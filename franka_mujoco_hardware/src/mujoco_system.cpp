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

#include <franka_mujoco_hardware/mujoco_system.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>

#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>

namespace franka_mujoco_hardware {

namespace {
constexpr double kLimitEps = 1e-3;
constexpr double kVelocityTolerance = kLimitEps + 3.0 * 1e-3 * 10.0;

std::string join(const std::array<double, 7>& values) {
  std::ostringstream out;
  out.precision(4);
  out << std::fixed << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    out << (i ? " " : "") << values[i];
  }
  out << "]";
  return out.str();
}
}  // namespace

std::array<double, 7> MujocoSystem::upper_velocity_limits(const std::array<double, 7>& q) {
  const auto limit = [](double vmax, double offset, double gain, double bound, double qi) {
    return std::min(vmax, std::max(0.0, -offset + std::sqrt(std::max(0.0, gain * (bound - qi))))) -
           kVelocityTolerance;
  };
  return {limit(2.62, 0.30, 12.0, 2.75010, q[0]), limit(2.62, 0.20, 5.17, 1.79180, q[1]),
          limit(2.62, 0.20, 7.00, 2.90650, q[2]), limit(2.62, 0.30, 8.00, -0.1458, q[3]),
          limit(5.26, 0.35, 34.0, 2.81010, q[4]), limit(4.18, 0.35, 11.0, 4.52050, q[5]),
          limit(5.26, 0.35, 34.0, 3.01960, q[6])};
}

std::array<double, 7> MujocoSystem::lower_velocity_limits(const std::array<double, 7>& q) {
  const auto limit = [](double vmax, double offset, double gain, double bound, double qi) {
    return std::max(-vmax, std::min(0.0, offset - std::sqrt(std::max(0.0, gain * (bound + qi))))) +
           kVelocityTolerance;
  };
  return {limit(2.62, 0.30, 12.0, 2.750100, q[0]), limit(2.62, 0.20, 5.17, 1.791800, q[1]),
          limit(2.62, 0.20, 7.00, 2.906500, q[2]), limit(2.62, 0.30, 8.00, 3.048100, q[3]),
          limit(5.26, 0.35, 34.0, 2.810100, q[4]), limit(4.18, 0.35, 11.0, -0.54092, q[5]),
          limit(5.26, 0.35, 34.0, 3.019600, q[6])};
}

bool MujocoSystem::parse_array7(const std::string& name, std::array<double, 7>& out, bool required) {
  const auto it = info_.hardware_parameters.find(name);
  if (it == info_.hardware_parameters.end()) {
    if (required) {
      RCLCPP_FATAL(logger(), "hardware parameter '%s' is required", name.c_str());
    }
    return !required;
  }
  std::istringstream stream(it->second);
  std::array<double, 7> values{};
  for (size_t i = 0; i < 7; ++i) {
    if (!(stream >> values[i])) {
      RCLCPP_FATAL(logger(), "hardware parameter '%s' needs 7 numbers, got '%s'", name.c_str(),
                   it->second.c_str());
      return false;
    }
  }
  out = values;
  return true;
}

hardware_interface::CallbackReturn MujocoSystem::on_init(const hardware_interface::HardwareInfo& info) {
  if (hardware_interface::SystemInterface::on_init(info) != hardware_interface::CallbackReturn::SUCCESS) {
    return hardware_interface::CallbackReturn::ERROR;
  }
  if (info_.joints.size() != kNumJoints) {
    RCLCPP_FATAL(logger(), "expected %zu joints in the ros2_control description, got %zu", kNumJoints,
                 info_.joints.size());
    return hardware_interface::CallbackReturn::ERROR;
  }

  const auto model_it = info_.hardware_parameters.find("model_path");
  if (model_it == info_.hardware_parameters.end() || model_it->second.empty()) {
    RCLCPP_FATAL(logger(), "hardware parameter 'model_path' (MJCF file) is required");
    return hardware_interface::CallbackReturn::ERROR;
  }
  model_path_ = model_it->second;

  const auto flag = [this](const char* name, bool& out) {
    const auto it = info_.hardware_parameters.find(name);
    if (it != info_.hardware_parameters.end()) {
      out = (it->second == "true" || it->second == "True" || it->second == "1");
    }
  };
  flag("reflex_on_violation", reflex_on_violation_);
  flag("gravity_compensation", gravity_compensation_);
  if (info_.hardware_parameters.count("cycle_time")) {
    cycle_time_ = std::stod(info_.hardware_parameters.at("cycle_time"));
  }
  if (!parse_array7("stiffness", stiffness_, false) || !parse_array7("damping", damping_, false)) {
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Initial configuration: explicit parameter, otherwise the joints' initial_value.
  bool have_initial = false;
  if (info_.hardware_parameters.count("initial_positions")) {
    if (!parse_array7("initial_positions", initial_positions_, true)) {
      return hardware_interface::CallbackReturn::ERROR;
    }
    have_initial = true;
  }
  if (!have_initial) {
    for (size_t i = 0; i < kNumJoints; ++i) {
      for (const auto& state_interface : info_.joints[i].state_interfaces) {
        if (state_interface.name == hardware_interface::HW_IF_POSITION &&
            !state_interface.initial_value.empty()) {
          initial_positions_[i] = std::stod(state_interface.initial_value);
        }
      }
    }
  }

  char error[1000] = "";
  model_ = mj_loadXML(model_path_.c_str(), nullptr, error, sizeof(error));
  if (model_ == nullptr) {
    RCLCPP_FATAL(logger(), "could not load MJCF '%s': %s", model_path_.c_str(), error);
    return hardware_interface::CallbackReturn::ERROR;
  }
  data_ = mj_makeData(model_);
  scratch_ = mj_makeData(model_);

  for (size_t i = 0; i < kNumJoints; ++i) {
    const std::string& joint = info_.joints[i].name;
    const int joint_id = mj_name2id(model_, mjOBJ_JOINT, joint.c_str());
    if (joint_id < 0) {
      RCLCPP_FATAL(logger(), "joint '%s' is not in the MJCF (ros2_control joint names must match "
                   "the MuJoCo joint names)", joint.c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }
    qpos_index_[i] = model_->jnt_qposadr[joint_id];
    dof_index_[i] = model_->jnt_dofadr[joint_id];
    const int actuator_id = mj_name2id(model_, mjOBJ_ACTUATOR, joint.c_str());
    if (actuator_id < 0) {
      RCLCPP_FATAL(logger(), "no actuator named '%s' in the MJCF (a <motor> per joint is expected)",
                   joint.c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }
    actuator_index_[i] = actuator_id;
    if (model_->jnt_limited[joint_id]) {
      position_lower_[i] = model_->jnt_range[2 * joint_id];
      position_upper_[i] = model_->jnt_range[2 * joint_id + 1];
    }
    if (model_->actuator_ctrllimited[actuator_id]) {
      torque_limits_[i] = model_->actuator_ctrlrange[2 * actuator_id + 1];
    }
  }

  RCLCPP_INFO(logger(),
              "MuJoCo %s loaded '%s' (timestep %.4f s). Joint impedance emulation K=%s D=%s, reflex on "
              "violation: %s",
              mj_versionString(), model_path_.c_str(), model_->opt.timestep, join(stiffness_).c_str(),
              join(damping_).c_str(), reflex_on_violation_ ? "yes" : "no");
  return hardware_interface::CallbackReturn::SUCCESS;
}

void MujocoSystem::reset_state() {
  mj_resetData(model_, data_);
  for (size_t i = 0; i < kNumJoints; ++i) {
    data_->qpos[qpos_index_[i]] = initial_positions_[i];
    data_->qvel[dof_index_[i]] = 0.0;
    data_->ctrl[actuator_index_[i]] = 0.0;
  }
  mj_forward(model_, data_);
  for (size_t i = 0; i < kNumJoints; ++i) {
    hw_positions_[i] = data_->qpos[qpos_index_[i]];
    hw_velocities_[i] = 0.0;
    hw_efforts_[i] = 0.0;
    hw_position_commands_[i] = hw_positions_[i];
    hw_velocity_commands_[i] = 0.0;
    hw_effort_commands_[i] = 0.0;
    q_d_[i] = hw_positions_[i];
    dq_d_[i] = 0.0;
    ddq_d_[i] = 0.0;
    tau_applied_[i] = 0.0;
  }
  sim_time_ = 0.0;
  violations_ = 0.0;
  motion_generator_started_ = false;
}

hardware_interface::CallbackReturn MujocoSystem::on_activate(const rclcpp_lifecycle::State& /*previous_state*/) {
  reset_state();
  mode_ = ControlMode::kNone;
  requested_mode_ = ControlMode::kNone;
  RCLCPP_INFO(logger(), "activated at q=%s", join(hw_positions_).c_str());
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn MujocoSystem::on_deactivate(const rclcpp_lifecycle::State& /*previous_state*/) {
  mode_ = ControlMode::kNone;
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> MujocoSystem::export_state_interfaces() {
  std::vector<hardware_interface::StateInterface> interfaces;
  for (size_t i = 0; i < kNumJoints; ++i) {
    interfaces.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_positions_[i]);
    interfaces.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_velocities_[i]);
    interfaces.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &hw_efforts_[i]);
  }
  interfaces.emplace_back(info_.name, "sim_time", &sim_time_);
  interfaces.emplace_back(info_.name, "motion_generator_violations", &violations_);
  return interfaces;
}

std::vector<hardware_interface::CommandInterface> MujocoSystem::export_command_interfaces() {
  std::vector<hardware_interface::CommandInterface> interfaces;
  for (size_t i = 0; i < kNumJoints; ++i) {
    interfaces.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_position_commands_[i]);
    interfaces.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_velocity_commands_[i]);
    interfaces.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &hw_effort_commands_[i]);
  }
  return interfaces;
}

hardware_interface::return_type MujocoSystem::prepare_command_mode_switch(
    const std::vector<std::string>& start_interfaces, const std::vector<std::string>& stop_interfaces) {
  bool position = position_claimed_;
  bool velocity = velocity_claimed_;
  bool effort = effort_claimed_;
  const auto apply = [](const std::vector<std::string>& names, bool value, bool& p, bool& v, bool& e) {
    for (const auto& name : names) {
      if (name.size() > 9 && name.compare(name.size() - 9, 9, "/position") == 0) {
        p = value;
      } else if (name.size() > 9 && name.compare(name.size() - 9, 9, "/velocity") == 0) {
        v = value;
      } else if (name.size() > 7 && name.compare(name.size() - 7, 7, "/effort") == 0) {
        e = value;
      }
    }
  };
  apply(stop_interfaces, false, position, velocity, effort);
  apply(start_interfaces, true, position, velocity, effort);
  if ((position && velocity) || (position && effort) || (velocity && effort)) {
    RCLCPP_ERROR(logger(), "only one of position/velocity/effort can be claimed at a time (like franka_hardware)");
    return hardware_interface::return_type::ERROR;
  }
  position_claimed_ = position;
  velocity_claimed_ = velocity;
  effort_claimed_ = effort;
  requested_mode_ = position ? ControlMode::kPosition
                    : velocity ? ControlMode::kVelocity
                    : effort   ? ControlMode::kEffort
                               : ControlMode::kNone;
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type MujocoSystem::perform_command_mode_switch(
    const std::vector<std::string>& /*start_interfaces*/, const std::vector<std::string>& /*stop_interfaces*/) {
  if (requested_mode_ != mode_) {
    mode_ = requested_mode_;
    needs_initial_command_ = true;
    motion_generator_started_ = false;
    const char* names[] = {"none", "position (internal joint impedance)", "velocity", "effort"};
    RCLCPP_INFO(logger(), "control mode -> %s", names[static_cast<int>(mode_)]);
  }
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type MujocoSystem::read(const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {
  for (size_t i = 0; i < kNumJoints; ++i) {
    hw_positions_[i] = data_->qpos[qpos_index_[i]];
    hw_velocities_[i] = data_->qvel[dof_index_[i]];
    hw_efforts_[i] = tau_applied_[i];
  }
  sim_time_ = data_->time;
  if (needs_initial_command_ && mode_ != ControlMode::kNone) {
    // Like franka_hardware::initializePositionCommands: the first command a controller sees is
    // the current state, so a controller that does not write yet holds position.
    for (size_t i = 0; i < kNumJoints; ++i) {
      hw_position_commands_[i] = hw_positions_[i];
      hw_velocity_commands_[i] = 0.0;
      hw_effort_commands_[i] = 0.0;
      q_d_[i] = hw_positions_[i];
      dq_d_[i] = 0.0;
      ddq_d_[i] = 0.0;
    }
    needs_initial_command_ = false;
  }
  return hardware_interface::return_type::OK;
}

void MujocoSystem::compute_gravity(std::array<double, 7>& gravity) {
  mju_copy(scratch_->qpos, data_->qpos, model_->nq);
  mju_zero(scratch_->qvel, model_->nv);
  mju_zero(scratch_->qacc, model_->nv);
  mj_forward(model_, scratch_);
  for (size_t i = 0; i < kNumJoints; ++i) {
    gravity[i] = scratch_->qfrc_bias[dof_index_[i]];
  }
}

bool MujocoSystem::check_motion_generator(const std::array<double, 7>& q_new, double dt) {
  // The checks libfranka's motion generator applies to consecutive joint position commands.
  // Names are the franka::Errors flags the real robot raises.
  const char* violation = nullptr;
  size_t joint = 0;
  double value = 0.0;
  double limit = 0.0;
  const auto upper = upper_velocity_limits(q_d_);
  const auto lower = lower_velocity_limits(q_d_);
  std::array<double, 7> dq{};
  std::array<double, 7> ddq{};
  for (size_t i = 0; i < kNumJoints && violation == nullptr; ++i) {
    if (!std::isfinite(q_new[i])) {
      violation = "command is not finite";
      joint = i;
      break;
    }
    if (q_new[i] < position_lower_[i] || q_new[i] > position_upper_[i]) {
      violation = "joint_motion_generator_position_limits_violation";
      joint = i;
      value = q_new[i];
      limit = q_new[i] < position_lower_[i] ? position_lower_[i] : position_upper_[i];
      break;
    }
    dq[i] = (q_new[i] - q_d_[i]) / dt;
    if (dq[i] > upper[i] + kVelocityTolerance || dq[i] < lower[i] - kVelocityTolerance) {
      violation = "joint_motion_generator_velocity_limits_violation";
      joint = i;
      value = dq[i];
      limit = dq[i] > 0 ? upper[i] : lower[i];
      break;
    }
    ddq[i] = (dq[i] - dq_d_[i]) / dt;
    if (motion_generator_started_ && std::abs(ddq[i]) > max_acceleration_ + 1e-6) {
      violation = "joint_motion_generator_velocity_discontinuity";  // acceleration limit
      joint = i;
      value = ddq[i];
      limit = max_acceleration_;
      break;
    }
    const double dddq = (ddq[i] - ddq_d_[i]) / dt;
    if (motion_generator_started_ && std::abs(dddq) > max_jerk_ + 1e-6) {
      violation = "joint_motion_generator_acceleration_discontinuity";  // jerk limit
      joint = i;
      value = dddq;
      limit = max_jerk_;
      break;
    }
  }
  if (violation != nullptr) {
    violations_ += 1.0;
    RCLCPP_ERROR_THROTTLE(logger(), steady_clock_, 500,
                          "motion generator: %s on joint %zu (value %.4f, limit %.4f) at t=%.3f s. The real "
                          "robot would reflex here.",
                          violation, joint + 1, value, limit, data_->time);
    return false;
  }
  return true;
}

hardware_interface::return_type MujocoSystem::write(const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {
  // One cycle of robot time per write(), regardless of the measured period (see cycle_time_).
  const double dt = cycle_time_;
  std::array<double, 7> tau{};
  std::array<double, 7> gravity{};

  switch (mode_) {
    case ControlMode::kNone:
      // Nothing claimed: hold with the impedance law on the last desired position.
      for (size_t i = 0; i < kNumJoints; ++i) {
        tau[i] = stiffness_[i] * (q_d_[i] - data_->qpos[qpos_index_[i]]) - damping_[i] * data_->qvel[dof_index_[i]] +
                 data_->qfrc_bias[dof_index_[i]];
      }
      break;
    case ControlMode::kPosition:
    case ControlMode::kVelocity: {
      std::array<double, 7> q_new{};
      for (size_t i = 0; i < kNumJoints; ++i) {
        q_new[i] = (mode_ == ControlMode::kPosition) ? hw_position_commands_[i]
                                                      : q_d_[i] + hw_velocity_commands_[i] * dt;
      }
      const bool ok = check_motion_generator(q_new, dt);
      if (!ok && reflex_on_violation_) {
        RCLCPP_FATAL(logger(), "stopping: the commanded motion violates the FR3 motion generator limits (set "
                     "reflex_on_violation:=false to keep simulating)");
        return hardware_interface::return_type::ERROR;
      }
      for (size_t i = 0; i < kNumJoints; ++i) {
        const double dq_new = (q_new[i] - q_d_[i]) / dt;
        ddq_d_[i] = (dq_new - dq_d_[i]) / dt;
        dq_d_[i] = dq_new;
        q_d_[i] = q_new[i];
      }
      motion_generator_started_ = true;
      for (size_t i = 0; i < kNumJoints; ++i) {
        tau[i] = stiffness_[i] * (q_d_[i] - data_->qpos[qpos_index_[i]]) +
                 damping_[i] * (dq_d_[i] - data_->qvel[dof_index_[i]]) + data_->qfrc_bias[dof_index_[i]];
      }
      break;
    }
    case ControlMode::kEffort:
      if (gravity_compensation_) {
        compute_gravity(gravity);
      }
      for (size_t i = 0; i < kNumJoints; ++i) {
        tau[i] = hw_effort_commands_[i] + gravity[i];
      }
      break;
  }

  for (size_t i = 0; i < kNumJoints; ++i) {
    tau[i] = std::clamp(tau[i], -torque_limits_[i], torque_limits_[i]);
    data_->ctrl[actuator_index_[i]] = tau[i];
    tau_applied_[i] = tau[i];
  }

  substeps_ = std::max(1, static_cast<int>(std::lround(cycle_time_ / model_->opt.timestep)));
  for (int k = 0; k < substeps_; ++k) {
    mj_step(model_, data_);
  }
  return hardware_interface::return_type::OK;
}

}  // namespace franka_mujoco_hardware

PLUGINLIB_EXPORT_CLASS(franka_mujoco_hardware::MujocoSystem, hardware_interface::SystemInterface)
