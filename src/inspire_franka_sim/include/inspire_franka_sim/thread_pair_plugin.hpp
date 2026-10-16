// Copyright (c) 2026 Agile Robots SE
// Licensed under the Apache License, Version 2.0.
// See http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <mujoco/mujoco.h>

#include <atomic>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_publisher.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp>

namespace inspire_franka_sim
{

/// Publishes the threaded nut's state and clamps the thread on request.
///
/// `inspire_franka_policy_scene.xml` carries ForgeUltra's simplified thread
/// pair: an axial slide coupled to a twist hinge by the `thread_coupling`
/// joint equality, plus an inactive `thread_hold` equality (see
/// apps/policy_rollout/utils/make_ros_sim_scene.py). The geometry is static
/// MJCF and needs nothing from here; the two things that cannot be expressed
/// declaratively are:
///
/// - **reading the turn.** The nut is not part of the robot, so no ros2_control
///   state interface reaches it and `joint_state_broadcaster` cannot see it.
///   Without this the rollout has to infer the turn from the grasp frame's own
///   yaw, a proxy that cannot tell a nut turn from the wrist yaw of the
///   descent -- and that fires spuriously when the fingertips converge. The
///   policy's own lifecycle releases on 55 degrees of turn, so it needs the
///   real angle.
/// - **holding the thread.** Isaac clamps the thread for the whole
///   release/return transition, so the nut cannot unwind while the fingers let
///   go of it (`_update_solver_thread_state`). The MuJoCo-only loop does it in
///   `ThreadingScene._apply_thread_drive`, which this mirrors: the coupling's
///   slope goes to zero at the held axial position and `thread_hold` pins the
///   twist. A holding PD torque on a 0.001 kg m^2 hinge is unstable at this
///   timestep; an implicitly solved equality is not.
///
/// Interface, all on sim time:
///
///   /thread_state   sensor_msgs/JointState   [nut_axial, nut_twist], position
///                                            and velocity, 100 Hz
///   /thread_hold    std_msgs/Bool            true clamps, false releases
///   /reset_thread   std_srvs/Trigger         back to the start pose on the bolt
///
/// A scene without a thread pair (every other MJCF in this package) loads the
/// plugin harmlessly: `init` says so once and `update` returns immediately.
class ThreadPairPlugin : public mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase
{
public:
  ThreadPairPlugin() = default;
  ~ThreadPairPlugin() override = default;

  bool init(rclcpp::Node::SharedPtr node, const mjModel * model, mjData * data) override;
  void update(const mjModel * model, mjData * data) override;
  void cleanup() override;

private:
  /// Clamp the thread where it stands, as ThreadingScene.hold_thread does.
  void engage_hold(const mjModel * model, mjData * data);
  /// Hand the nut back to the coupling.
  void release_hold(const mjModel * model, mjData * data);
  /// Start pose on the bolt: the coupling's own offset, twist zero.
  void reset_thread(const mjModel * model, mjData * data);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger logger_{rclcpp::get_logger("ThreadPairPlugin")};
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr hold_subscription_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr state_publisher_raw_;
  std::unique_ptr<realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>>
  state_publisher_;

  /// False when the scene has no thread pair; update() then does nothing.
  bool present_{false};

  int axial_qpos_{-1};
  int axial_dof_{-1};
  int twist_qpos_{-1};
  int twist_dof_{-1};
  int coupling_{-1};
  int hold_{-1};

  /// The coupling as the scene declared it, to restore on release.
  mjtNum start_offset_{0.0};
  mjtNum coupling_slope_{0.0};

  // Written by ROS executor threads, read in update().
  std::atomic_bool hold_requested_{false};
  std::atomic_bool reset_requested_{false};
  bool hold_engaged_{false};

  double publish_period_{0.01};
  double next_publish_time_{0.0};
};

}  // namespace inspire_franka_sim
