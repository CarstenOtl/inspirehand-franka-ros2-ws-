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
#include <franka_msgs/srv/set_cartesian_stiffness.hpp>
#include <franka_msgs/srv/set_full_collision_behavior.hpp>
#include <franka_trajectory_replay_msgs/msg/cartesian_goto.hpp>
#include <franka_trajectory_replay_msgs/msg/cartesian_replay_state.hpp>
#include <franka_trajectory_replay_msgs/msg/cartesian_trajectory.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#include <realtime_tools/realtime_publisher.hpp>
#include <std_msgs/msg/empty.hpp>

#include "franka_semantic_components/franka_cartesian_pose_interface.hpp"
#include "franka_semantic_components/franka_robot_model.hpp"
#include "franka_trajectory_replay/cartesian_impedance.hpp"
#include "franka_trajectory_replay/fr3_kinematics.hpp"

namespace franka_trajectory_replay {

/**
 * Plays back a Cartesian (6-DOF pose) trajectory on an FR3 with the torque law of
 * franka_example_controllers/CartesianImpedanceExampleController, and publishes, at the
 * controller rate, the reference it applied next to the measured pose.
 *
 * The law (cartesian_impedance.hpp) is the example's; what this controller adds is where its
 * reference comes from. The example's demo arc and activation-time nullspace pose are replaced
 * by a sampler over the same four phases as TrajectoryReplayController:
 *
 *  - ``~/goto`` (CartesianGoto): quintic ramp of the pose target (slerp for the orientation)
 *    and of the nullspace configuration. Reports idle once the example's reference filter has
 *    settled on the target.
 *  - ``~/trajectory`` (CartesianTrajectory): the trajectory to replay. Its first pose must be
 *    within ``max_trajectory_start_error_m`` / ``_rad`` of the current target. Positions are
 *    interpolated with cubic Hermite splines when the points carry linear velocities, linearly
 *    otherwise; orientations are slerped; nullspace configurations are linear.
 *  - ``~/pause`` / ``~/resume`` / ``~/abort`` (std_msgs/Empty): exactly the joint controller's
 *    clock-rate ramps. Abort ramps the phase clock to zero and holds.
 *  - live parameters ``translational_stiffness``, ``rotational_stiffness``,
 *    ``nullspace_stiffness``, ``stiffness_scale``, ``target_filter`` and the example's own
 *    ``~/set_cartesian_stiffness`` service. Gain changes take effect through the example's
 *    first-order filter.
 *
 * ``model_source`` selects where the pose and Jacobian come from: ``franka`` reads
 * franka_hardware's cartesian_pose_state and robot_model interfaces (the real arm);
 * ``dh`` computes both from the built-in FR3 DH model with an identity F_T_EE and no coriolis
 * term, for simulators that export only joint interfaces.
 *
 * Outputs: ``~/status`` (DiagnosticArray, the joint controller's keys plus the Cartesian
 * ones), ``~/controller_state`` (joint feedback, the torque output, the nullspace reference)
 * and ``~/cartesian_state`` (target, filtered reference, measured pose, the law's error and
 * its torque terms).
 */
class CartesianTrajectoryReplayController : public controller_interface::ControllerInterface {
 public:
  using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
  static constexpr int kNumJoints = 7;
  static constexpr int kPoseInterfaces = 16;

  enum class Phase : int { kIdle = 0, kGoto = 1, kTrajectory = 2, kStopping = 3 };
  static const char* phase_name(Phase phase);

  /// Quintic blend 10s^3 - 15s^4 + 6s^5: zero velocity and acceleration at both ends.
  static double quintic_blend(double s);

  /// A trajectory handed from the subscription callback to update().
  struct Trajectory {
    std::vector<double> times;                        ///< seconds from start, increasing
    std::vector<std::array<double, 3>> positions;     ///< base frame, metres
    std::vector<std::array<double, 4>> orientations;  ///< x y z w, sign-continuous
    std::vector<std::array<double, 3>> velocities;    ///< linear; empty when not provided
    std::vector<std::array<double, 7>> nullspace;     ///< one per point, forward-filled
    bool has_velocities{false};
  };

  /// Interpolates the trajectory at time t (clamped to its ends). Public for unit testing.
  static void sample_trajectory(const Trajectory& trajectory, double t, size_t& segment_hint,
                                Eigen::Vector3d& position, Eigen::Quaterniond& orientation,
                                Vector7d& nullspace);

  /// libfranka's FR3 Cartesian velocity limits (rate_limiting.h), packet-loss tolerance included.
  static constexpr double kMaxTranslationalVelocity = 3.0 - 1e-3 - 3.0 * 1e-3 * 9.0;
  static constexpr double kMaxRotationalVelocity = 2.5 - 1e-3 - 3.0 * 1e-3 * 17.0;

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
  enum class CommandKind : int { kNone = 0, kGoto, kTrajectory, kAbort };
  enum class Fault : int { kNone = 0, kPosition = 1, kOrientation = 2 };

  struct Command {
    CommandKind kind{CommandKind::kNone};
    uint64_t id{0};
    std::array<double, 3> position{};
    std::array<double, 4> orientation{0.0, 0.0, 0.0, 1.0};
    std::array<double, 7> nullspace{};
    bool has_nullspace{false};
    double duration{0.0};
    std::shared_ptr<const Trajectory> trajectory;
  };

  struct GainSettings {
    std::array<double, 6> stiffness{};  ///< diagonal, before stiffness_scale
    double stiffness_scale{1.0};
    double nullspace_stiffness{20.0};
    double target_filter{0.005};
    uint64_t revision{0};
  };

  bool assign_parameters();
  rcl_interfaces::msg::SetParametersResult gain_parameters_callback(
      const std::vector<rclcpp::Parameter>& parameters);
  bool publish_gain_settings(GainSettings settings, const char* origin);
  void set_cartesian_stiffness_callback(
      const std::shared_ptr<franka_msgs::srv::SetCartesianStiffness::Request> request,
      std::shared_ptr<franka_msgs::srv::SetCartesianStiffness::Response> response);
  void update_gains(bool initialise);
  void update_joint_states();
  std::vector<std::string> joint_names() const;

  Vector7d saturate_torque_rate(const Vector7d& tau_desired, const Vector7d& tau_previous) const;
  void begin_stopping();
  void advance_clock(double dt, double requested_rate);

  void goto_callback(const franka_trajectory_replay_msgs::msg::CartesianGoto::SharedPtr msg);
  void trajectory_callback(
      const franka_trajectory_replay_msgs::msg::CartesianTrajectory::SharedPtr msg);
  void pause_callback(const std_msgs::msg::Empty::SharedPtr msg);
  void resume_callback(const std_msgs::msg::Empty::SharedPtr msg);
  void abort_callback(const std_msgs::msg::Empty::SharedPtr msg);
  void publish_status();
  void reject(const std::string& reason);
  bool inside_workspace(const Eigen::Vector3d& position) const;
  bool read_nullspace(const std::vector<double>& values, std::array<double, 7>& out,
                      const std::string& source);

  // --- interfaces -------------------------------------------------------------------------
  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
  std::unique_ptr<franka_semantic_components::FrankaCartesianPoseInterface> franka_cartesian_pose_;
  static constexpr size_t kPositionOffset = 0;
  static constexpr size_t kVelocityOffset = 7;
  static constexpr size_t kEffortOffset = 14;

  // --- parameters -------------------------------------------------------------------------
  std::string arm_id_;
  std::string arm_prefix_;
  std::string base_frame_;
  bool model_from_dh_{false};
  bool nullspace_follows_trajectory_{true};
  bool coriolis_compensation_{true};
  double torque_rate_limit_{0.0};
  double goto_max_velocity_{0.10};
  double goto_max_angular_velocity_{0.50};
  double goto_max_nullspace_velocity_{0.50};
  double goto_min_duration_{3.0};
  double max_goto_step_m_{0.30};
  double max_goto_step_rad_{1.0};
  double max_trajectory_start_error_m_{0.002};
  double max_trajectory_start_error_rad_{0.01};
  double goto_settle_tolerance_m_{0.0005};
  double goto_settle_tolerance_rad_{0.002};
  double goto_settle_timeout_{2.0};
  std::array<double, 3> workspace_min_{};
  std::array<double, 3> workspace_max_{};
  double trajectory_velocity_scale_{1.0};
  double max_position_error_{0.08};
  double max_orientation_error_{0.35};
  double pause_ramp_duration_{0.5};
  double abort_stop_duration_{0.5};
  std::array<double, kNumJoints> position_limits_lower_{};
  std::array<double, kNumJoints> position_limits_upper_{};

  // --- ROS entities -----------------------------------------------------------------------
  rclcpp::Client<franka_msgs::srv::SetFullCollisionBehavior>::SharedPtr collision_client_;
  rclcpp::Subscription<franka_trajectory_replay_msgs::msg::CartesianGoto>::SharedPtr
      goto_subscriber_;
  rclcpp::Subscription<franka_trajectory_replay_msgs::msg::CartesianTrajectory>::SharedPtr
      trajectory_subscriber_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr pause_subscriber_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr resume_subscriber_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr abort_subscriber_;
  rclcpp::Service<franka_msgs::srv::SetCartesianStiffness>::SharedPtr stiffness_service_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr status_publisher_;
  rclcpp::TimerBase::SharedPtr status_timer_;
  std::unique_ptr<realtime_tools::RealtimePublisher<control_msgs::msg::JointTrajectoryControllerState>>
      state_publisher_;
  std::unique_ptr<realtime_tools::RealtimePublisher<
      franka_trajectory_replay_msgs::msg::CartesianReplayState>>
      cartesian_state_publisher_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;

  // --- cross-thread state -----------------------------------------------------------------
  realtime_tools::RealtimeBuffer<Command> command_buffer_;
  realtime_tools::RealtimeBuffer<GainSettings> gain_settings_buffer_;
  uint64_t next_command_id_{0};  ///< only touched by the single-threaded executor callbacks
  uint64_t next_gain_revision_{0};
  std::atomic<int> phase_{static_cast<int>(Phase::kIdle)};
  std::atomic<uint64_t> active_command_id_{0};
  std::atomic<uint64_t> processed_command_id_{0};
  std::atomic<uint64_t> completed_command_id_{0};
  std::atomic<double> phase_elapsed_{0.0};
  std::atomic<double> phase_duration_{0.0};
  std::atomic<bool> command_initialized_{false};
  std::atomic<bool> is_active_{false};
  std::atomic<bool> pause_requested_{false};
  std::atomic<bool> paused_{false};
  std::atomic<double> playback_rate_{1.0};
  std::atomic<int> fault_{static_cast<int>(Fault::kNone)};
  std::atomic<double> stiffness_scale_target_{1.0};
  std::array<std::atomic<double>, 3> target_position_snapshot_;
  std::array<std::atomic<double>, 4> target_orientation_snapshot_;
  std::array<std::atomic<double>, 7> target_nullspace_snapshot_;
  std::array<std::atomic<double>, 7> reference_pose_snapshot_;  ///< filtered p_d, q_d (xyzw)
  std::array<std::atomic<double>, 3> gains_applied_snapshot_;   ///< translational, rotational, nullspace
  std::atomic<double> position_error_{0.0};
  std::atomic<double> orientation_error_{0.0};
  std::atomic<double> joint_limit_margin_{0.0};
  std::string last_rejection_;  ///< executor thread only
  uint64_t rejections_{0};

  // --- realtime state ---------------------------------------------------------------------
  Phase rt_phase_{Phase::kIdle};
  Phase rt_stopping_from_{Phase::kIdle};
  uint64_t rt_command_id_{0};
  std::shared_ptr<const Trajectory> rt_trajectory_;
  size_t rt_segment_hint_{0};
  double rt_elapsed_{0.0};
  double rt_duration_{0.0};
  double rt_settle_elapsed_{0.0};
  double rt_playback_rate_{1.0};
  double rt_playback_target_{1.0};
  double rt_rate_ramp_start_{1.0};
  double rt_rate_ramp_elapsed_{0.0};
  double rt_rate_ramp_duration_{0.5};
  Eigen::Vector3d blend_position_start_;
  Eigen::Vector3d blend_position_target_;
  Eigen::Quaterniond blend_orientation_start_;
  Eigen::Quaterniond blend_orientation_target_;
  Vector7d blend_nullspace_start_;
  Vector7d blend_nullspace_target_;
  Eigen::Vector3d position_target_;       ///< sampler output
  Eigen::Quaterniond orientation_target_;
  Vector7d nullspace_target_;
  Eigen::Vector3d position_d_;            ///< after the example's filter
  Eigen::Quaterniond orientation_d_;
  Vector7d nullspace_d_;
  Matrix6d stiffness_;
  Matrix6d damping_;
  double nullspace_stiffness_{20.0};
  Matrix6d stiffness_target_;
  Matrix6d damping_target_;
  double nullspace_stiffness_target_{20.0};
  double filter_alpha_{0.005};
  uint64_t rt_gain_revision_{0};
  Vector7d tau_command_previous_;
  bool first_update_{true};

  std::array<double, kNumJoints> joint_positions_current_{};
  std::array<double, kNumJoints> joint_velocities_current_{};
  std::array<double, kNumJoints> joint_efforts_current_{};
};

}  // namespace franka_trajectory_replay
