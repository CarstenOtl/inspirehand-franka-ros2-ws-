// Copyright (c) 2026 Agile Robots SE
// Licensed under the Apache License, Version 2.0.
// See http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <mujoco/mujoco.h>

#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_publisher.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp>

namespace inspire_franka_sim
{

/// Publishes the thumb and index pad forces the way ForgeUltra observes them.
///
/// A force-observing student sees ForgeUltra's `distillation_fingertip_force_n`
/// (forge_ultra/tasks/utils/fingertip_force.py): for each of thumb and index,
/// the net contact force on the pad link -- every contact, since the RH56
/// sensor cannot tell what presses it -- expressed in the calibrated tip
/// frame, as `max(-F_local_x, 0)` in N. The pad link is the parent of the
/// `thumb_tip` / `index_tip` body. The calibrated frames are yawed about the
/// pad link's Z relative to the URDF tip bodies this package's MJCFs carry
/// (ForgeUltra's TIP_FRAME_LOCAL_ROT0_WXYZ), so the yaw is applied here; the
/// MuJoCo-only rollout (policy_rollout/mujoco_threading_env.py) uses the same
/// two numbers.
///
/// Contact forces are not reachable through any ros2_control interface, hence
/// a plugin.
///
/// Interface, on sim time:
///
///   /fingertip_force   sensor_msgs/JointState   effort =
///                      [thumb_flexion_extension_force,
///                       index_flexion_extension_force] in N, 120 Hz
///                      (training's physics rate)
///
/// The physical hand's counterpart is the driver's ~/grip_force (raw grams);
/// run_policy_rollout.py reads either.
///
/// A scene without the two tip bodies loads the plugin harmlessly.
class FingertipForcePlugin : public mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase
{
public:
  FingertipForcePlugin() = default;
  ~FingertipForcePlugin() override = default;

  bool init(rclcpp::Node::SharedPtr node, const mjModel * model, mjData * data) override;
  void update(const mjModel * model, mjData * data) override;
  void cleanup() override;

private:
  rclcpp::Logger logger_{rclcpp::get_logger("FingertipForcePlugin")};
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_raw_;
  std::unique_ptr<realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>> publisher_;

  bool present_{false};
  /// thumb, index
  int tip_body_[2]{-1, -1};
  int pad_body_[2]{-1, -1};

  double publish_period_{1.0 / 120.0};
  double next_publish_time_{0.0};
};

}  // namespace inspire_franka_sim
