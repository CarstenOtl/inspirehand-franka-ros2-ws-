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
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <franka_msgs/srv/set_full_collision_behavior.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#include <realtime_tools/realtime_publisher.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/empty.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "franka_semantic_components/franka_robot_model.hpp"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_trajectory_replay {

/**
 * Plays back a joint-space trajectory on an FR3 and publishes, at the controller rate, the
 * reference it actually applied next to the measured state.
 *
 * Two command modes, selected with the ``command_interface`` parameter:
 *
 *  - ``position``: claims the joint position command interfaces. franka_hardware then runs
 *    libfranka's joint position motion generator with the robot's *internal* joint impedance
 *    controller (``ControllerMode::kJointImpedance``) - the default control mode of the arm.
 *    franka_hardware applies no filtering or rate limiting to position commands, so this
 *    controller ships its own limiter (``rate_limit``), which reproduces libfranka's
 *    ``limitRate`` for joint positions with the FR3 velocity/acceleration/jerk limits.
 *  - ``effort``: claims the effort interfaces and runs the joint impedance law of
 *    ``franka_example_controllers/JointImpedanceWithIKExampleController`` (stiffness and damping
 *    on the joint error plus coriolis compensation) on the same reference.
 *
 * Inputs (all non-realtime):
 *  - ``~/goto`` (sensor_msgs/JointState): quintic ramp to a joint target. The duration is
 *    derived from the step size and ``goto_max_velocity`` / ``goto_max_acceleration``.
 *  - ``~/trajectory`` (trajectory_msgs/JointTrajectory): the trajectory to replay. Must start
 *    within ``max_trajectory_start_error`` of the current command. Points carrying velocities
 *    are interpolated with cubic Hermite splines, points without them linearly.
 *  - ``~/abort`` (std_msgs/Empty): decelerate smoothly to a stop and hold.
 *
 * Outputs:
 *  - ``~/controller_state`` (control_msgs/JointTrajectoryControllerState) every update:
 *    reference (position, velocity, ``time_from_start`` = trajectory clock), feedback, error and
 *    output (the torque, or the position command after the rate limiter).
 *  - ``~/status`` (diagnostic_msgs/DiagnosticArray) at ``status_rate``: phase, command id,
 *    progress, limiter engagement count, last rejection reason.
 */
class TrajectoryReplayController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  static constexpr int kNumJoints = 7;

  enum class Phase : int { kIdle = 0, kGoto = 1, kTrajectory = 2, kStopping = 3 };
  static const char* phase_name(Phase phase);

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

  /// Quintic blend 10s^3 - 15s^4 + 6s^5: zero velocity and acceleration at both ends.
  static double quintic_blend(double s);
  /// Derivative of quintic_blend with respect to s.
  static double quintic_blend_derivative(double s);

  /**
   * Position-dependent joint velocity limits of the FR3, as libfranka's
   * computeUpperLimitsJointVelocity / computeLowerLimitsJointVelocity (rate_limiting.h).
   */
  static std::array<double, kNumJoints> upper_velocity_limits(const std::array<double, kNumJoints>& q);
  static std::array<double, kNumJoints> lower_velocity_limits(const std::array<double, kNumJoints>& q);

  /// A trajectory handed from the subscription callback to update().
  struct Trajectory {
    std::vector<double> times;                              ///< seconds from start, increasing
    std::vector<std::array<double, kNumJoints>> positions;  ///< one row per point
    std::vector<std::array<double, kNumJoints>> velocities; ///< empty when not provided
    bool has_velocities{false};
  };

  /// Interpolates the trajectory at time t (clamped to its ends). Public for unit testing.
  static void sample_trajectory(const Trajectory& trajectory, double t, size_t& segment_hint,
                                std::array<double, kNumJoints>& position);

 private:
  enum class CommandKind : int { kNone = 0, kGoto, kTrajectory, kAbort };

  struct Command {
    CommandKind kind{CommandKind::kNone};
    uint64_t id{0};
    std::array<double, kNumJoints> target{};
    double duration{0.0};
    std::shared_ptr<const Trajectory> trajectory;
  };

  bool assign_parameters();
  void update_joint_states();
  bool apply_collision_behavior();
  std::vector<std::string> joint_names() const;

  Vector7d compute_torque_command(const Vector7d& q_desired, const Vector7d& q_current,
                                  const Vector7d& dq_current);
  Vector7d saturate_torque_rate(const Vector7d& tau_desired, const Vector7d& tau_previous) const;

  /// libfranka's limitRate for joint positions. Returns the limited command and updates the
  /// limiter's own velocity/acceleration memory. Sets `engaged` when it changed anything.
  Vector7d limit_position_rate(const Vector7d& q_desired, bool& engaged);

  void goto_callback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void trajectory_callback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  void abort_callback(const std_msgs::msg::Empty::SharedPtr msg);
  void publish_status();
  void reject(const std::string& reason);

  /// Reorders a message's joints onto ours. Empty names mean "already in our order".
  bool joint_index_map(const std::vector<std::string>& names, std::array<size_t, kNumJoints>& map,
                       const std::string& source);

  // --- interfaces -------------------------------------------------------------------------
  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
  static constexpr size_t kPositionOffset = 0;
  static constexpr size_t kVelocityOffset = 7;
  static constexpr size_t kEffortOffset = 14;

  // --- parameters -------------------------------------------------------------------------
  std::string robot_type_;
  std::string arm_prefix_;
  bool effort_mode_{false};
  bool coriolis_compensation_{true};
  bool rate_limit_{true};
  double torque_rate_limit_{1.0};
  double goto_max_velocity_{0.5};
  double goto_max_acceleration_{1.0};
  double goto_min_duration_{2.0};
  double max_joint_step_{1.5};
  double max_trajectory_start_error_{0.05};
  double trajectory_velocity_scale_{1.0};
  double trajectory_acceleration_scale_{1.0};
  double abort_stop_duration_{0.5};
  /// One controller cycle. The arm delivers one state packet per millisecond and franka_hardware
  /// blocks on it, so a cycle is 1 ms of robot time whatever the wall clock measured.
  double cycle_time_{1e-3};
  std::array<double, kNumJoints> position_limits_lower_{};
  std::array<double, kNumJoints> position_limits_upper_{};
  std::array<double, kNumJoints> max_acceleration_{};
  std::array<double, kNumJoints> max_jerk_{};
  Vector7d k_gains_;
  Vector7d d_gains_;

  // --- ROS entities -----------------------------------------------------------------------
  rclcpp::Client<franka_msgs::srv::SetFullCollisionBehavior>::SharedPtr collision_client_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr goto_subscriber_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_subscriber_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr abort_subscriber_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr status_publisher_;
  rclcpp::TimerBase::SharedPtr status_timer_;
  std::unique_ptr<realtime_tools::RealtimePublisher<control_msgs::msg::JointTrajectoryControllerState>>
      state_publisher_;

  // --- cross-thread state -----------------------------------------------------------------
  realtime_tools::RealtimeBuffer<Command> command_buffer_;
  uint64_t next_command_id_{0};  ///< only touched by the single-threaded executor callbacks
  std::atomic<int> phase_{static_cast<int>(Phase::kIdle)};
  std::atomic<uint64_t> active_command_id_{0};
  std::atomic<uint64_t> completed_command_id_{0};
  std::atomic<double> phase_elapsed_{0.0};
  std::atomic<double> phase_duration_{0.0};
  std::atomic<uint64_t> rate_limit_engaged_{0};
  std::atomic<uint64_t> rate_limit_engaged_last_command_{0};
  std::atomic<bool> command_initialized_{false};
  std::atomic<bool> is_active_{false};
  std::array<std::atomic<double>, kNumJoints> measured_positions_snapshot_;
  std::array<std::atomic<double>, kNumJoints> command_snapshot_;
  std::string last_rejection_;  ///< executor thread only
  uint64_t rejections_{0};

  // --- realtime state ---------------------------------------------------------------------
  Phase rt_phase_{Phase::kIdle};
  uint64_t rt_command_id_{0};
  std::shared_ptr<const Trajectory> rt_trajectory_;
  size_t rt_segment_hint_{0};
  double rt_elapsed_{0.0};
  double rt_duration_{0.0};
  Vector7d blend_start_;
  Vector7d blend_target_;
  Vector7d stop_velocity_;
  Vector7d position_command_;
  Vector7d position_command_previous_;
  Vector7d velocity_command_;
  Vector7d tau_command_previous_;
  Vector7d dq_filtered_;
  // rate limiter memory (position mode)
  Vector7d limiter_q_;
  Vector7d limiter_dq_;
  Vector7d limiter_ddq_;
  bool first_update_{true};

  std::array<double, kNumJoints> joint_positions_current_{};
  std::array<double, kNumJoints> joint_velocities_current_{};
  std::array<double, kNumJoints> joint_efforts_current_{};
};

}  // namespace franka_trajectory_replay
