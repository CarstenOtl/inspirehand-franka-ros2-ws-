// Copyright (c) 2026 Agile Robots SE
// Licensed under the Apache License, Version 2.0.
// See http://www.apache.org/licenses/LICENSE-2.0

#include "inspire_franka_sim/thread_pair_plugin.hpp"

#include <cstddef>

#include <pluginlib/class_list_macros.hpp>

namespace
{

int joint_id(const mjModel * model, const char * name)
{
  return mj_name2id(model, mjOBJ_JOINT, name);
}

int equality_id(const mjModel * model, const char * name)
{
  return mj_name2id(model, mjOBJ_EQUALITY, name);
}

/// MuJoCo keeps equality parameters in a flat neq x mjNEQDATA block, and
/// mjModel is handed to a plugin as const. Writing eq_data is nonetheless safe:
/// it is an input the solver reads each step, with nothing derived or cached
/// from it (unlike, say, body masses, which feed mj_setConst). This is the same
/// thing ThreadingScene._apply_thread_drive does from Python, where the model
/// is not const.
mjtNum * equality_data(const mjModel * model, int equality)
{
  return const_cast<mjtNum *>(model->eq_data) + static_cast<std::size_t>(equality) * mjNEQDATA;
}

}  // namespace

namespace inspire_franka_sim
{

bool ThreadPairPlugin::init(rclcpp::Node::SharedPtr node, const mjModel * model, mjData * data)
{
  node_ = node;
  logger_ = node->get_logger().get_child("thread_pair");

  const int axial = joint_id(model, "nut_axial");
  const int twist = joint_id(model, "nut_twist");
  coupling_ = equality_id(model, "thread_coupling");
  hold_ = equality_id(model, "thread_hold");
  if (axial < 0 || twist < 0 || coupling_ < 0 || hold_ < 0) {
    RCLCPP_INFO(
      logger_,
      "no thread pair in this scene (nut_axial/nut_twist/thread_coupling/thread_hold); "
      "the plugin will stay idle");
    return true;
  }
  if (model->jnt_type[axial] != mjJNT_SLIDE || model->jnt_type[twist] != mjJNT_HINGE) {
    RCLCPP_ERROR(
      logger_, "nut_axial must be a slide and nut_twist a hinge; refusing to drive the thread");
    return false;
  }

  axial_qpos_ = model->jnt_qposadr[axial];
  axial_dof_ = model->jnt_dofadr[axial];
  twist_qpos_ = model->jnt_qposadr[twist];
  twist_dof_ = model->jnt_dofadr[twist];
  // The scene is the authority on where "unturned, fully backed off" is.
  const mjtNum * coupling = equality_data(model, coupling_);
  start_offset_ = coupling[0];
  coupling_slope_ = coupling[1];

  state_publisher_raw_ = node->create_publisher<sensor_msgs::msg::JointState>(
    "/thread_state", rclcpp::SensorDataQoS());
  state_publisher_ =
    std::make_unique<realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>>(
      state_publisher_raw_);
  {
    // Size the message once; update() must not allocate.
    auto & message = state_publisher_->msg_;
    message.name = {"nut_axial", "nut_twist"};
    message.position.assign(2, 0.0);
    message.velocity.assign(2, 0.0);
  }

  hold_subscription_ = node->create_subscription<std_msgs::msg::Bool>(
    "/thread_hold", rclcpp::QoS(1).reliable().transient_local(),
    [this](const std_msgs::msg::Bool::SharedPtr message) {
      hold_requested_.store(message->data);
    });
  reset_service_ = node->create_service<std_srvs::srv::Trigger>(
    "/reset_thread",
    [this](
      const std_srvs::srv::Trigger::Request::SharedPtr,
      std_srvs::srv::Trigger::Response::SharedPtr response) {
      reset_requested_.store(true);
      response->success = true;
      response->message = "thread reset queued for the next simulation step";
    });

  reset_thread(model, data);
  present_ = true;
  RCLCPP_INFO(
    logger_,
    "thread pair ready: start offset %.5f m, %.8f m per rad; publishing /thread_state, "
    "listening on /thread_hold, resetting on /reset_thread",
    static_cast<double>(start_offset_), static_cast<double>(coupling_slope_));
  return true;
}

void ThreadPairPlugin::update(const mjModel * model, mjData * data)
{
  if (!present_) {
    return;
  }

  if (reset_requested_.exchange(false)) {
    hold_requested_.store(false);
    reset_thread(model, data);
  }

  const bool wanted = hold_requested_.load();
  if (wanted != hold_engaged_) {
    if (wanted) {
      engage_hold(model, data);
    } else {
      release_hold(model, data);
    }
    hold_engaged_ = wanted;
  }

  // data->time is the clock mujoco_ros2_control publishes on /clock, so the
  // stamp needs no ROS call from this realtime thread.
  if (data->time + 1e-9 < next_publish_time_) {
    return;
  }
  next_publish_time_ = data->time + publish_period_;
  if (state_publisher_->trylock()) {
    auto & message = state_publisher_->msg_;
    message.header.stamp =
      rclcpp::Time(static_cast<int64_t>(data->time * 1.0e9), RCL_ROS_TIME);
    message.position[0] = data->qpos[axial_qpos_];
    message.position[1] = data->qpos[twist_qpos_];
    message.velocity[0] = data->qvel[axial_dof_];
    message.velocity[1] = data->qvel[twist_dof_];
    state_publisher_->unlockAndPublish();
  }
}

void ThreadPairPlugin::engage_hold(const mjModel * model, mjData * data)
{
  // Freeze the coupling at the axial position the nut has reached, then pin the
  // twist there too. Both are consistent with each other by construction, so
  // the solver starts the next step with no constraint violation to resolve.
  mjtNum * coupling = equality_data(model, coupling_);
  coupling[0] = data->qpos[axial_qpos_];
  coupling[1] = 0.0;
  equality_data(model, hold_)[0] = data->qpos[twist_qpos_];
  data->eq_active[hold_] = 1;
}

void ThreadPairPlugin::release_hold(const mjModel * model, mjData * data)
{
  mjtNum * coupling = equality_data(model, coupling_);
  coupling[0] = start_offset_;
  coupling[1] = coupling_slope_;
  data->eq_active[hold_] = 0;
}

void ThreadPairPlugin::reset_thread(const mjModel * model, mjData * data)
{
  release_hold(model, data);
  hold_engaged_ = false;
  data->qpos[axial_qpos_] = start_offset_;
  data->qpos[twist_qpos_] = 0.0;
  data->qvel[axial_dof_] = 0.0;
  data->qvel[twist_dof_] = 0.0;
}

void ThreadPairPlugin::cleanup()
{
  state_publisher_.reset();
  state_publisher_raw_.reset();
  hold_subscription_.reset();
  reset_service_.reset();
  node_.reset();
  present_ = false;
}

}  // namespace inspire_franka_sim

PLUGINLIB_EXPORT_CLASS(
  inspire_franka_sim::ThreadPairPlugin,
  mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
