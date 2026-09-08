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
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include <hardware_interface/handle.hpp>
#include <hardware_interface/hardware_info.hpp>
#include <hardware_interface/system_interface.hpp>
#include <hardware_interface/types/hardware_interface_return_values.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>

namespace franka_mujoco_hardware {

/**
 * ros2_control system that stands in for franka_hardware with a MuJoCo model of the arm.
 *
 * It exports the same joint interfaces as the real FR3 (position/velocity/effort command,
 * position/velocity/effort state) and reproduces the parts of libfranka's behaviour that
 * decide whether a trajectory runs or reflexes:
 *
 *  - position commands drive an emulated version of the robot's internal joint impedance
 *    controller: tau = K (q_d - q) + D (dq_d - dq) + bias(q, dq), with libfranka's default
 *    joint impedance stiffness. This is a plausibility model, not an identified one.
 *  - every position command is checked against the FR3 motion generator limits (position,
 *    position-dependent velocity, acceleration, jerk). A violation is logged with the name of
 *    the libfranka error the real robot would raise and, with `reflex_on_violation`, stops the
 *    hardware - exactly the failure mode that kills a real run.
 *  - effort commands are applied on top of gravity compensation, as libfranka does.
 *
 * Hardware parameters (all optional except model_path):
 *   model_path            MJCF file to load
 *   stiffness, damping    7 space-separated values for the emulated joint impedance
 *   initial_positions     7 values; otherwise the joints' position `initial_value`s
 *   reflex_on_violation   "true" to return ERROR from write() on a limit violation
 *   cycle_time            seconds of robot time per control cycle (default 0.001)
 *   gravity_compensation  "true" to add gravity torques to effort commands (libfranka does)
 */
class MujocoSystem : public hardware_interface::SystemInterface {
 public:
  RCLCPP_SHARED_PTR_DEFINITIONS(MujocoSystem)

  hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo& info) override;
  hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::return_type prepare_command_mode_switch(
      const std::vector<std::string>& start_interfaces,
      const std::vector<std::string>& stop_interfaces) override;
  hardware_interface::return_type perform_command_mode_switch(
      const std::vector<std::string>& start_interfaces,
      const std::vector<std::string>& stop_interfaces) override;

  hardware_interface::return_type read(const rclcpp::Time& time, const rclcpp::Duration& period) override;
  hardware_interface::return_type write(const rclcpp::Time& time, const rclcpp::Duration& period) override;

  static std::array<double, 7> upper_velocity_limits(const std::array<double, 7>& q);
  static std::array<double, 7> lower_velocity_limits(const std::array<double, 7>& q);

 private:
  enum class ControlMode { kNone, kPosition, kVelocity, kEffort };
  static constexpr size_t kNumJoints = 7;

  bool parse_array7(const std::string& name, std::array<double, 7>& out, bool required);
  void reset_state();
  void compute_gravity(std::array<double, 7>& gravity);
  bool check_motion_generator(const std::array<double, 7>& q_d, double dt);

  rclcpp::Logger logger() const { return rclcpp::get_logger("franka_mujoco_hardware"); }

  // model
  std::string model_path_;
  mjModel* model_{nullptr};
  mjData* data_{nullptr};
  mjData* scratch_{nullptr};
  std::array<int, 7> qpos_index_{};
  std::array<int, 7> dof_index_{};
  std::array<int, 7> actuator_index_{};
  int substeps_{1};

  // parameters
  std::array<double, 7> stiffness_{3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0};
  std::array<double, 7> damping_{60.0, 60.0, 60.0, 50.0, 40.0, 20.0, 15.0};
  std::array<double, 7> initial_positions_{0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785};
  std::array<double, 7> position_lower_{-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508};
  std::array<double, 7> position_upper_{2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508};
  std::array<double, 7> torque_limits_{87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0};
  bool reflex_on_violation_{true};
  bool gravity_compensation_{true};
  double max_acceleration_{10.0};
  double max_jerk_{5000.0};
  /// One control cycle in robot time. franka_hardware blocks on the arm's 1 kHz packets, so a
  /// cycle is 1 ms whatever the wall clock measured; the simulation does the same.
  double cycle_time_{1e-3};
  rclcpp::Clock steady_clock_{RCL_STEADY_TIME};

  // interfaces
  std::array<double, 7> hw_positions_{};
  std::array<double, 7> hw_velocities_{};
  std::array<double, 7> hw_efforts_{};
  std::array<double, 7> hw_position_commands_{};
  std::array<double, 7> hw_velocity_commands_{};
  std::array<double, 7> hw_effort_commands_{};
  double sim_time_{0.0};
  double violations_{0.0};

  // control
  ControlMode mode_{ControlMode::kNone};
  ControlMode requested_mode_{ControlMode::kNone};
  bool position_claimed_{false};
  bool velocity_claimed_{false};
  bool effort_claimed_{false};
  bool needs_initial_command_{true};
  std::array<double, 7> q_d_{};
  std::array<double, 7> dq_d_{};
  std::array<double, 7> ddq_d_{};
  std::array<double, 7> tau_applied_{};
  bool motion_generator_started_{false};
};

}  // namespace franka_mujoco_hardware
