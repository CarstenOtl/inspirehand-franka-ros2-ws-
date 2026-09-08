// Copyright (c) 2026 Agile Robots SE
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <control_msgs/msg/joint_trajectory_controller_state.hpp>
#include <controller_interface/controller_interface.hpp>
#include <franka_msgs/srv/set_full_collision_behavior.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/srv/get_position_ik.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#include <realtime_tools/realtime_publisher.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>

#include "franka_semantic_components/franka_robot_model.hpp"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_repeatability {

/**
 * Joint impedance controller with an externally commanded target, derived from
 * franka_example_controllers::JointImpedanceWithIKExampleController.
 *
 * Differences from the upstream example, all of them required to measure anything:
 *
 *  - The target is an input, not a hard-coded sinusoid. Targets arrive either as a Cartesian
 *    pose on ``~/target_pose`` (resolved through the MoveIt ``compute_ik`` service) or as a
 *    joint configuration on ``~/target_joint_positions``.
 *  - IK is solved once per incoming pose in the (non-realtime) subscription callback rather
 *    than on every 1 kHz update cycle. The accepted joint target is echoed on ``~/active_goal``
 *    so a measurement script can capture it and replay the identical target on later visits.
 *  - The commanded joint position is ramped from where the command currently is to the new
 *    target with a quintic profile over ``motion_duration`` seconds, so a target that is far
 *    away does not turn into a torque step.
 *  - The reference actually applied by the control law is published together with the measured
 *    state on ``~/controller_state`` at the controller update rate. Nothing else in the stack
 *    exposes this: ``FrankaRobotState.desired_joint_state`` carries libfranka's ``q_d`` from the
 *    motion generator, which is not running in torque control mode.
 */
class CartesianTargetImpedanceIKController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;

  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  /// A joint-space target handed from a non-realtime callback to update().
  struct Goal {
    std::array<double, 7> positions{};
    uint64_t id{0};
  };

  bool assign_parameters();
  void update_joint_states();

  /// Torque command of the upstream example: stiffness/damping on the joint error plus coriolis.
  Vector7d compute_torque_command(const Vector7d& joint_positions_desired,
                                  const Vector7d& joint_positions_current,
                                  const Vector7d& joint_velocities_current);

  /// Limits how fast the commanded torque may change, in Nm per millisecond.
  Vector7d saturate_torque_rate(const Vector7d& tau_desired, const Vector7d& tau_previous) const;

  void target_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void target_joint_positions_callback(const sensor_msgs::msg::JointState::SharedPtr msg);

  /// Validates a candidate joint target and, if sane, hands it to update(). Non-realtime.
  bool accept_goal(const std::vector<double>& positions, const std::string& source);

  std::shared_ptr<moveit_msgs::srv::GetPositionIK::Request> create_ik_service_request(
      const geometry_msgs::msg::PoseStamped& pose) const;

  /**
   * Raises the collision thresholds to the same values the upstream
   * joint_impedance_with_ik example applies before it runs. Without this the robot keeps
   * whatever Desk last set, which is low enough that the restoring torque of the impedance law
   * itself can trip a cartesian_reflex and abort control.
   */
  bool apply_collision_behavior();

  std::vector<std::string> joint_names() const;

  static double quintic_blend(double s);

  // --- interfaces -------------------------------------------------------------------------
  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;

  static constexpr int kNumJoints = 7;
  const std::string k_robot_state_interface_name{"robot_state"};
  const std::string k_robot_model_interface_name{"robot_model"};

  // Offsets into state_interfaces_. state_interface_configuration() requests, in this order:
  // 7 positions, 7 velocities, 7 efforts, the robot model interfaces, then robot_time.
  static constexpr size_t kPositionOffset = 0;
  static constexpr size_t kVelocityOffset = 7;
  static constexpr size_t kEffortOffset = 14;
  size_t robot_time_index_{0};

  // --- parameters -------------------------------------------------------------------------
  std::string robot_type_;
  std::string arm_prefix_;
  std::string ik_group_name_;
  std::string ik_link_name_;
  bool avoid_collisions_{true};
  double motion_duration_{5.0};
  double max_joint_step_{1.5};
  double torque_rate_limit_{1.0};
  double ik_timeout_{0.05};
  Vector7d k_gains_;
  Vector7d d_gains_;

  // --- ROS entities -----------------------------------------------------------------------
  rclcpp::Client<moveit_msgs::srv::GetPositionIK>::SharedPtr compute_ik_client_;
  rclcpp::Client<franka_msgs::srv::SetFullCollisionBehavior>::SharedPtr collision_client_;
  bool set_collision_behavior_{true};
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_subscriber_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr target_joint_subscriber_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr active_goal_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr motion_active_publisher_;
  rclcpp::TimerBase::SharedPtr motion_active_timer_;
  std::unique_ptr<realtime_tools::RealtimePublisher<control_msgs::msg::JointTrajectoryControllerState>>
      state_publisher_;

  // --- cross-thread state -----------------------------------------------------------------
  realtime_tools::RealtimeBuffer<Goal> goal_buffer_;
  uint64_t next_goal_id_{0};  ///< only touched by the (single-threaded) executor callbacks
  std::atomic<bool> motion_active_{false};
  std::atomic<bool> command_initialized_{false};
  /// The publishers are lifecycle publishers; publishing while inactive only logs a complaint.
  std::atomic<bool> is_active_{false};
  /// Snapshot of the measured joint positions for the non-realtime IK seed. Written per element
  /// from update(); a torn read across elements is harmless for a seed and for the step check.
  std::array<std::atomic<double>, kNumJoints> measured_positions_snapshot_;

  // --- realtime state ---------------------------------------------------------------------
  uint64_t active_goal_id_{0};
  Vector7d position_command_;
  Vector7d position_command_previous_;
  Vector7d blend_start_;
  Vector7d blend_target_;
  Vector7d tau_command_previous_;
  Vector7d dq_filtered_;
  double blend_elapsed_{0.0};
  double robot_time_{0.0};
  double last_robot_time_{0.0};
  bool first_update_{true};

  std::array<double, kNumJoints> joint_positions_current_{};
  std::array<double, kNumJoints> joint_velocities_current_{};
  std::array<double, kNumJoints> joint_efforts_current_{};
};

}  // namespace franka_repeatability
