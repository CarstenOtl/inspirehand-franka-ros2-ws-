// Copyright (c) 2026 Agile Robots SE
// Licensed under the Apache License, Version 2.0.
// See http://www.apache.org/licenses/LICENSE-2.0

#include "inspire_franka_sim/fingertip_force_plugin.hpp"

#include <algorithm>
#include <cmath>

#include <pluginlib/class_list_macros.hpp>

namespace
{

constexpr const char * kTipBodies[2] = {"thumb_tip", "index_tip"};
constexpr const char * kChannels[2] = {
  "thumb_flexion_extension_force", "index_flexion_extension_force"};
/// ForgeUltra TIP_FRAME_LOCAL_ROT0_WXYZ, as yaw about the pad link's Z.
constexpr double kSensorYawRad[2] = {2.617993878, -0.261799388};

}  // namespace

namespace inspire_franka_sim
{

bool FingertipForcePlugin::init(
  rclcpp::Node::SharedPtr node, const mjModel * model, mjData * /*data*/)
{
  logger_ = node->get_logger().get_child("fingertip_force");
  for (int k = 0; k < 2; ++k) {
    tip_body_[k] = mj_name2id(model, mjOBJ_BODY, kTipBodies[k]);
    if (tip_body_[k] < 0) {
      RCLCPP_INFO(
        logger_, "no %s body in this scene; the plugin will stay idle", kTipBodies[k]);
      return true;
    }
    pad_body_[k] = model->body_parentid[tip_body_[k]];
  }

  publisher_raw_ = node->create_publisher<sensor_msgs::msg::JointState>(
    "/fingertip_force", rclcpp::SensorDataQoS());
  publisher_ =
    std::make_unique<realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>>(
      publisher_raw_);
  {
    // Size the message once; update() must not allocate.
    auto & message = publisher_->msg_;
    message.name = {kChannels[0], kChannels[1]};
    message.effort.assign(2, 0.0);
  }
  present_ = true;
  RCLCPP_INFO(
    logger_, "publishing /fingertip_force: pad forces on %s and %s along their sensor -X",
    mj_id2name(model, mjOBJ_BODY, pad_body_[0]), mj_id2name(model, mjOBJ_BODY, pad_body_[1]));
  return true;
}

void FingertipForcePlugin::update(const mjModel * model, mjData * data)
{
  if (!present_ || data->time + 1e-9 < next_publish_time_) {
    return;
  }
  next_publish_time_ = data->time + publish_period_;

  // The snapshot carries the step's contacts and solver forces, which is all
  // mj_contactForce reads.
  double value[2] = {0.0, 0.0};
  for (int k = 0; k < 2; ++k) {
    mjtNum force_world[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < data->ncon; ++i) {
      const mjContact & contact = data->contact[i];
      const bool on1 = model->geom_bodyid[contact.geom1] == pad_body_[k];
      const bool on2 = model->geom_bodyid[contact.geom2] == pad_body_[k];
      if (on1 == on2) {
        continue;
      }
      mjtNum wrench[6];
      mj_contactForce(model, data, i, wrench);
      // The contact-frame force acts on geom2 and its reaction on geom1;
      // frame rows are the contact axes in world coordinates.
      mjtNum force[3];
      mju_mulMatTVec(force, contact.frame, wrench, 3, 3);
      mju_addToScl3(force_world, force, on2 ? 1.0 : -1.0);
    }
    // Sensor X in world: the tip body's frame turned by the calibrated yaw.
    const mjtNum * R = data->xmat + 9 * tip_body_[k];
    const double c = std::cos(kSensorYawRad[k]);
    const double s = std::sin(kSensorYawRad[k]);
    const mjtNum sensor_x[3] = {
      c * R[0] + s * R[1], c * R[3] + s * R[4], c * R[6] + s * R[7]};
    value[k] = std::max(-mju_dot3(sensor_x, force_world), 0.0);
  }

  if (publisher_->trylock()) {
    auto & message = publisher_->msg_;
    message.header.stamp = rclcpp::Time(static_cast<int64_t>(data->time * 1.0e9), RCL_ROS_TIME);
    message.effort[0] = value[0];
    message.effort[1] = value[1];
    publisher_->unlockAndPublish();
  }
}

void FingertipForcePlugin::cleanup()
{
  publisher_.reset();
  publisher_raw_.reset();
  present_ = false;
}

}  // namespace inspire_franka_sim

PLUGINLIB_EXPORT_CLASS(
  inspire_franka_sim::FingertipForcePlugin,
  mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
