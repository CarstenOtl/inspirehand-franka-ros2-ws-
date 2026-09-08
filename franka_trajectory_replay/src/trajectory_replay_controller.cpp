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

#include <franka_trajectory_replay/trajectory_replay_controller.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include <pluginlib/class_list_macros.hpp>

using namespace std::chrono_literals;

namespace franka_trajectory_replay {

namespace {

// libfranka rate_limiting.h: tolerance subtracted from the velocity limits so that a few lost
// packets cannot push a command over the robot's own check.
constexpr double kLimitEps = 1e-3;
constexpr double kTolNumberPacketsLost = 3.0;
constexpr double kMaxJointAccelerationDefault = 10.0 - kLimitEps;
constexpr double kMaxJointJerkDefault = 5000.0 - kLimitEps;
constexpr double kVelocityTolerance = kLimitEps + kTolNumberPacketsLost * 1e-3 * 10.0;

// FR3 joint position limits, franka_description/robots/fr3/joint_limits.yaml.
constexpr std::array<double, 7> kPositionLower{-2.9007, -1.8361, -2.9007, -3.0770,
                                               -2.8763, 0.4398,  -3.0508};
constexpr std::array<double, 7> kPositionUpper{2.9007, 1.8361, 2.9007, -0.1169,
                                               2.8763, 4.6216, 3.0508};

std::string format_joints(const std::array<double, 7>& values) {
  char buffer[128];
  std::snprintf(buffer, sizeof(buffer), "[%.3f %.3f %.3f %.3f %.3f %.3f %.3f]", values[0],
                values[1], values[2], values[3], values[4], values[5], values[6]);
  return buffer;
}

}  // namespace

const char* TrajectoryReplayController::phase_name(Phase phase) {
  switch (phase) {
    case Phase::kIdle:
      return "idle";
    case Phase::kGoto:
      return "goto";
    case Phase::kTrajectory:
      return "trajectory";
    case Phase::kStopping:
      return "stopping";
  }
  return "unknown";
}

// --- static helpers -----------------------------------------------------------------------

double TrajectoryReplayController::quintic_blend(double s) {
  return s * s * s * (10.0 + s * (-15.0 + 6.0 * s));
}

double TrajectoryReplayController::quintic_blend_derivative(double s) {
  return s * s * (30.0 + s * (-60.0 + 30.0 * s));
}

std::array<double, 7> TrajectoryReplayController::upper_velocity_limits(
    const std::array<double, 7>& q) {
  const auto limit = [](double vmax, double offset, double gain, double bound, double qi) {
    return std::min(vmax, std::max(0.0, -offset + std::sqrt(std::max(0.0, gain * (bound - qi))))) -
           kVelocityTolerance;
  };
  return {limit(2.62, 0.30, 12.0, 2.75010, q[0]), limit(2.62, 0.20, 5.17, 1.79180, q[1]),
          limit(2.62, 0.20, 7.00, 2.90650, q[2]), limit(2.62, 0.30, 8.00, -0.1458, q[3]),
          limit(5.26, 0.35, 34.0, 2.81010, q[4]), limit(4.18, 0.35, 11.0, 4.52050, q[5]),
          limit(5.26, 0.35, 34.0, 3.01960, q[6])};
}

std::array<double, 7> TrajectoryReplayController::lower_velocity_limits(
    const std::array<double, 7>& q) {
  const auto limit = [](double vmax, double offset, double gain, double bound, double qi) {
    return std::max(-vmax, std::min(0.0, offset - std::sqrt(std::max(0.0, gain * (bound + qi))))) +
           kVelocityTolerance;
  };
  return {limit(2.62, 0.30, 12.0, 2.750100, q[0]), limit(2.62, 0.20, 5.17, 1.791800, q[1]),
          limit(2.62, 0.20, 7.00, 2.906500, q[2]), limit(2.62, 0.30, 8.00, 3.048100, q[3]),
          limit(5.26, 0.35, 34.0, 2.810100, q[4]), limit(4.18, 0.35, 11.0, -0.54092, q[5]),
          limit(5.26, 0.35, 34.0, 3.019600, q[6])};
}

void TrajectoryReplayController::sample_trajectory(const Trajectory& trajectory, double t,
                                                   size_t& segment_hint,
                                                   std::array<double, 7>& position) {
  const size_t n = trajectory.times.size();
  if (n == 0) {
    return;
  }
  if (t <= trajectory.times.front()) {
    position = trajectory.positions.front();
    segment_hint = 0;
    return;
  }
  if (t >= trajectory.times.back()) {
    position = trajectory.positions.back();
    segment_hint = n - 1;
    return;
  }
  // Time only moves forward, so start the search from the last segment.
  size_t i = std::min(segment_hint, n - 2);
  while (i > 0 && trajectory.times[i] > t) {
    --i;
  }
  while (i + 1 < n - 1 && trajectory.times[i + 1] <= t) {
    ++i;
  }
  segment_hint = i;

  const double t0 = trajectory.times[i];
  const double t1 = trajectory.times[i + 1];
  const double h = t1 - t0;
  const double s = (h > 0.0) ? (t - t0) / h : 1.0;
  const auto& p0 = trajectory.positions[i];
  const auto& p1 = trajectory.positions[i + 1];

  if (trajectory.has_velocities) {
    // Cubic Hermite: C1 continuous, so no velocity steps at the sample boundaries.
    const auto& v0 = trajectory.velocities[i];
    const auto& v1 = trajectory.velocities[i + 1];
    const double s2 = s * s;
    const double s3 = s2 * s;
    const double h00 = 2 * s3 - 3 * s2 + 1;
    const double h10 = s3 - 2 * s2 + s;
    const double h01 = -2 * s3 + 3 * s2;
    const double h11 = s3 - s2;
    for (int j = 0; j < kNumJoints; ++j) {
      position[j] = h00 * p0[j] + h10 * h * v0[j] + h01 * p1[j] + h11 * h * v1[j];
    }
  } else {
    for (int j = 0; j < kNumJoints; ++j) {
      position[j] = p0[j] + (p1[j] - p0[j]) * s;
    }
  }
}

// --- interfaces ---------------------------------------------------------------------------

std::vector<std::string> TrajectoryReplayController::joint_names() const {
  std::vector<std::string> names;
  names.reserve(kNumJoints);
  for (int i = 1; i <= kNumJoints; ++i) {
    names.push_back(arm_prefix_ + robot_type_ + "_joint" + std::to_string(i));
  }
  return names;
}

controller_interface::InterfaceConfiguration
TrajectoryReplayController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  const std::string suffix = effort_mode_ ? "/effort" : "/position";
  for (const auto& joint : joint_names()) {
    config.names.push_back(joint + suffix);
  }
  return config;
}

controller_interface::InterfaceConfiguration
TrajectoryReplayController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto& joint : joint_names()) {
    config.names.push_back(joint + "/position");
  }
  for (const auto& joint : joint_names()) {
    config.names.push_back(joint + "/velocity");
  }
  for (const auto& joint : joint_names()) {
    config.names.push_back(joint + "/effort");
  }
  if (franka_robot_model_) {
    for (const auto& name : franka_robot_model_->get_state_interface_names()) {
      config.names.push_back(name);
    }
  }
  return config;
}

void TrajectoryReplayController::update_joint_states() {
  for (int i = 0; i < kNumJoints; ++i) {
    joint_positions_current_[i] = state_interfaces_.at(kPositionOffset + i).get_value();
    joint_velocities_current_[i] = state_interfaces_.at(kVelocityOffset + i).get_value();
    joint_efforts_current_[i] = state_interfaces_.at(kEffortOffset + i).get_value();
    measured_positions_snapshot_[i].store(joint_positions_current_[i], std::memory_order_relaxed);
  }
}

// --- control laws -------------------------------------------------------------------------

TrajectoryReplayController::Vector7d TrajectoryReplayController::compute_torque_command(
    const Vector7d& q_desired, const Vector7d& q_current, const Vector7d& dq_current) {
  Vector7d coriolis = Vector7d::Zero();
  if (franka_robot_model_) {
    std::array<double, 7> coriolis_array = franka_robot_model_->getCoriolisForceVector();
    coriolis = Vector7d(coriolis_array.data());
  }
  const double kAlpha = 0.99;
  dq_filtered_ = (1 - kAlpha) * dq_filtered_ + kAlpha * dq_current;
  const Vector7d q_error = q_desired - q_current;
  return k_gains_.cwiseProduct(q_error) - d_gains_.cwiseProduct(dq_filtered_) + coriolis;
}

TrajectoryReplayController::Vector7d TrajectoryReplayController::saturate_torque_rate(
    const Vector7d& tau_desired, const Vector7d& tau_previous) const {
  if (!(torque_rate_limit_ > 0.0)) {
    return tau_desired;
  }
  Vector7d tau_saturated;
  for (int i = 0; i < kNumJoints; ++i) {
    const double difference = tau_desired(i) - tau_previous(i);
    tau_saturated(i) =
        tau_previous(i) + std::clamp(difference, -torque_rate_limit_, torque_rate_limit_);
  }
  return tau_saturated;
}

TrajectoryReplayController::Vector7d TrajectoryReplayController::limit_position_rate(
    const Vector7d& q_desired, bool& engaged) {
  // Mirrors franka::limitRate(max_velocity(q), min_velocity(q), max_acceleration, max_jerk,
  // commanded_positions, last_commanded_positions, last_commanded_velocities,
  // last_commanded_accelerations) with the libfranka cycle.
  const double dt = cycle_time_;
  engaged = false;
  std::array<double, 7> q_last{};
  for (int i = 0; i < kNumJoints; ++i) {
    q_last[i] = limiter_q_(i);
  }
  const auto upper = upper_velocity_limits(q_last);
  const auto lower = lower_velocity_limits(q_last);

  Vector7d q_limited;
  for (int i = 0; i < kNumJoints; ++i) {
    const double dq_desired = (q_desired(i) - limiter_q_(i)) / dt;
    const double safe_max_ddq = std::min(max_jerk_[i] * dt + limiter_ddq_(i), max_acceleration_[i]);
    const double safe_min_ddq = std::max(-max_jerk_[i] * dt + limiter_ddq_(i), -max_acceleration_[i]);
    double dq = std::clamp(dq_desired, limiter_dq_(i) + safe_min_ddq * dt,
                           limiter_dq_(i) + safe_max_ddq * dt);
    dq = std::clamp(dq, lower[i], upper[i]);
    if (std::abs(dq - dq_desired) > 1e-9) {
      engaged = true;
    }
    q_limited(i) = limiter_q_(i) + dq * dt;
    limiter_ddq_(i) = (dq - limiter_dq_(i)) / dt;
    limiter_dq_(i) = dq;
    limiter_q_(i) = q_limited(i);
  }
  return q_limited;
}

// --- non-realtime input -------------------------------------------------------------------

void TrajectoryReplayController::reject(const std::string& reason) {
  ++rejections_;
  last_rejection_ = reason;
  RCLCPP_ERROR(get_node()->get_logger(), "Rejected: %s", reason.c_str());
}

bool TrajectoryReplayController::joint_index_map(const std::vector<std::string>& names,
                                                 std::array<size_t, 7>& map,
                                                 const std::string& source) {
  if (names.empty()) {
    for (size_t i = 0; i < kNumJoints; ++i) {
      map[i] = i;
    }
    return true;
  }
  const auto ours = joint_names();
  for (size_t i = 0; i < kNumJoints; ++i) {
    const auto it = std::find(names.begin(), names.end(), ours[i]);
    if (it == names.end()) {
      reject(source + " does not name joint " + ours[i]);
      return false;
    }
    map[i] = static_cast<size_t>(std::distance(names.begin(), it));
  }
  return true;
}

void TrajectoryReplayController::goto_callback(const sensor_msgs::msg::JointState::SharedPtr msg) {
  if (!command_initialized_.load(std::memory_order_acquire)) {
    reject("goto arrived before the first update cycle");
    return;
  }
  if (static_cast<Phase>(phase_.load()) != Phase::kIdle) {
    reject("goto while busy (phase " +
           std::string(phase_name(static_cast<Phase>(phase_.load()))) + "); abort first");
    return;
  }
  std::array<size_t, 7> map{};
  if (!joint_index_map(msg->name, map, "goto")) {
    return;
  }
  Command command;
  command.kind = CommandKind::kGoto;
  double largest_step = 0.0;
  double duration = goto_min_duration_;
  for (int i = 0; i < kNumJoints; ++i) {
    if (map[i] >= msg->position.size()) {
      reject("goto has too few positions");
      return;
    }
    const double target = msg->position[map[i]];
    if (!std::isfinite(target)) {
      reject("goto target is not finite");
      return;
    }
    if (target < position_limits_lower_[i] || target > position_limits_upper_[i]) {
      reject("goto target for joint " + std::to_string(i + 1) + " (" + std::to_string(target) +
             ") is outside the position limits");
      return;
    }
    command.target[i] = target;
    const double step = std::abs(target - command_snapshot_[i].load(std::memory_order_relaxed));
    largest_step = std::max(largest_step, step);
    // Quintic peak velocity is 1.875 * step / T, peak acceleration 5.7735 * step / T^2.
    duration = std::max(duration, 1.875 * step / goto_max_velocity_);
    duration = std::max(duration, std::sqrt(5.7735 * step / goto_max_acceleration_));
  }
  if (largest_step > max_joint_step_) {
    reject("goto target is " + std::to_string(largest_step) +
           " rad from the current command, which exceeds max_joint_step");
    return;
  }
  command.duration = duration;
  command.id = ++next_command_id_;
  command_buffer_.writeFromNonRT(command);
  RCLCPP_INFO(get_node()->get_logger(),
              "Accepted goto (command %lu): largest step %.4f rad, duration %.2f s, target %s",
              static_cast<unsigned long>(command.id), largest_step, duration,
              format_joints(command.target).c_str());
}

void TrajectoryReplayController::trajectory_callback(
    const trajectory_msgs::msg::JointTrajectory::SharedPtr msg) {
  if (!command_initialized_.load(std::memory_order_acquire)) {
    reject("trajectory arrived before the first update cycle");
    return;
  }
  if (static_cast<Phase>(phase_.load()) != Phase::kIdle) {
    reject("trajectory while busy (phase " +
           std::string(phase_name(static_cast<Phase>(phase_.load()))) + "); abort first");
    return;
  }
  if (msg->points.size() < 2) {
    reject("trajectory needs at least two points");
    return;
  }
  std::array<size_t, 7> map{};
  if (!joint_index_map(msg->joint_names, map, "trajectory")) {
    return;
  }

  auto trajectory = std::make_shared<Trajectory>();
  const size_t n = msg->points.size();
  trajectory->times.reserve(n);
  trajectory->positions.reserve(n);
  trajectory->has_velocities = std::all_of(
      msg->points.begin(), msg->points.end(),
      [](const auto& point) { return point.velocities.size() >= kNumJoints; });
  if (trajectory->has_velocities) {
    trajectory->velocities.reserve(n);
  }

  double previous_time = -1.0;
  std::array<double, 7> previous_position{};
  std::array<double, 7> previous_velocity{};
  double peak_velocity_ratio = 0.0;
  double peak_acceleration_ratio = 0.0;
  for (size_t k = 0; k < n; ++k) {
    const auto& point = msg->points[k];
    const double t = rclcpp::Duration(point.time_from_start).seconds();
    if (!(t >= 0.0) || t <= previous_time) {
      reject("trajectory times must be non-negative and strictly increasing (point " +
             std::to_string(k) + ")");
      return;
    }
    std::array<double, 7> position{};
    std::array<double, 7> velocity{};
    for (int i = 0; i < kNumJoints; ++i) {
      if (map[i] >= point.positions.size()) {
        reject("trajectory point " + std::to_string(k) + " has too few positions");
        return;
      }
      position[i] = point.positions[map[i]];
      if (!std::isfinite(position[i])) {
        reject("trajectory point " + std::to_string(k) + " is not finite");
        return;
      }
      if (position[i] < position_limits_lower_[i] || position[i] > position_limits_upper_[i]) {
        reject("trajectory point " + std::to_string(k) + " joint " + std::to_string(i + 1) +
               " (" + std::to_string(position[i]) + ") is outside the position limits");
        return;
      }
      if (trajectory->has_velocities) {
        velocity[i] = point.velocities[map[i]];
        if (!std::isfinite(velocity[i])) {
          reject("trajectory point " + std::to_string(k) + " velocity is not finite");
          return;
        }
      }
    }
    if (k > 0) {
      const double h = t - previous_time;
      const auto upper = upper_velocity_limits(previous_position);
      const auto lower = lower_velocity_limits(previous_position);
      for (int i = 0; i < kNumJoints; ++i) {
        const double dq = (position[i] - previous_position[i]) / h;
        const double limit = dq >= 0.0 ? upper[i] : -lower[i];
        peak_velocity_ratio =
            std::max(peak_velocity_ratio, std::abs(dq) / (trajectory_velocity_scale_ * limit));
        if (k > 1) {
          const double ddq = (dq - previous_velocity[i]) / h;
          peak_acceleration_ratio = std::max(
              peak_acceleration_ratio,
              std::abs(ddq) / (trajectory_acceleration_scale_ * max_acceleration_[i]));
        }
        previous_velocity[i] = dq;
      }
    }
    previous_time = t;
    previous_position = position;
    trajectory->times.push_back(t);
    trajectory->positions.push_back(position);
    if (trajectory->has_velocities) {
      trajectory->velocities.push_back(velocity);
    }
  }

  double start_error = 0.0;
  for (int i = 0; i < kNumJoints; ++i) {
    start_error = std::max(start_error, std::abs(trajectory->positions.front()[i] -
                                                 command_snapshot_[i].load(std::memory_order_relaxed)));
  }
  if (start_error > max_trajectory_start_error_) {
    reject("trajectory starts " + std::to_string(start_error) +
           " rad away from the current command (max_trajectory_start_error); goto its first "
           "point first");
    return;
  }
  if (peak_velocity_ratio > 1.0) {
    reject("trajectory exceeds the joint velocity limit (" +
           std::to_string(100.0 * peak_velocity_ratio) + " % of the limit)");
    return;
  }
  if (peak_acceleration_ratio > 1.0) {
    reject("trajectory exceeds the joint acceleration limit (" +
           std::to_string(100.0 * peak_acceleration_ratio) + " % of the limit)");
    return;
  }

  Command command;
  command.kind = CommandKind::kTrajectory;
  command.trajectory = trajectory;
  command.duration = trajectory->times.back();
  command.id = ++next_command_id_;
  command_buffer_.writeFromNonRT(command);
  RCLCPP_INFO(get_node()->get_logger(),
              "Accepted trajectory (command %lu): %zu points, %.2f s, %s interpolation, peak "
              "%.0f %% of the velocity limit, %.0f %% of the acceleration limit, start error "
              "%.4f rad",
              static_cast<unsigned long>(command.id), n, command.duration,
              trajectory->has_velocities ? "cubic Hermite" : "linear", 100.0 * peak_velocity_ratio,
              100.0 * peak_acceleration_ratio, start_error);
}

void TrajectoryReplayController::abort_callback(const std_msgs::msg::Empty::SharedPtr /*msg*/) {
  Command command;
  command.kind = CommandKind::kAbort;
  command.id = ++next_command_id_;
  command_buffer_.writeFromNonRT(command);
  RCLCPP_WARN(get_node()->get_logger(), "Abort requested (command %lu)",
              static_cast<unsigned long>(command.id));
}

void TrajectoryReplayController::publish_status() {
  if (!is_active_.load(std::memory_order_acquire)) {
    return;
  }
  diagnostic_msgs::msg::DiagnosticArray msg;
  msg.header.stamp = get_node()->now();
  diagnostic_msgs::msg::DiagnosticStatus status;
  const auto phase = static_cast<Phase>(phase_.load());
  status.level = last_rejection_.empty() ? diagnostic_msgs::msg::DiagnosticStatus::OK
                                         : diagnostic_msgs::msg::DiagnosticStatus::WARN;
  status.name = std::string(get_node()->get_name());
  status.message = phase_name(phase);
  status.hardware_id = effort_mode_ ? "effort" : "position";

  const auto add = [&status](const char* key, const std::string& value) {
    diagnostic_msgs::msg::KeyValue kv;
    kv.key = key;
    kv.value = value;
    status.values.push_back(kv);
  };
  add("phase", std::to_string(static_cast<int>(phase)));
  add("phase_name", phase_name(phase));
  add("command_mode", effort_mode_ ? "effort" : "position");
  add("active_command_id", std::to_string(active_command_id_.load()));
  add("completed_command_id", std::to_string(completed_command_id_.load()));
  add("elapsed", std::to_string(phase_elapsed_.load()));
  add("duration", std::to_string(phase_duration_.load()));
  add("rate_limit_engaged_total", std::to_string(rate_limit_engaged_.load()));
  add("rate_limit_engaged_last_command", std::to_string(rate_limit_engaged_last_command_.load()));
  add("rejections", std::to_string(rejections_));
  add("last_rejection", last_rejection_);
  std::array<double, 7> command{};
  for (int i = 0; i < kNumJoints; ++i) {
    command[i] = command_snapshot_[i].load(std::memory_order_relaxed);
  }
  add("command", format_joints(command));
  msg.status.push_back(status);
  status_publisher_->publish(msg);
}

// --- realtime loop ------------------------------------------------------------------------

controller_interface::return_type TrajectoryReplayController::update(
    const rclcpp::Time& time, const rclcpp::Duration& /*period*/) {
  update_joint_states();
  Vector7d q_current(joint_positions_current_.data());
  Vector7d dq_current(joint_velocities_current_.data());

  // Advance by the nominal cycle, not the measured period: the reference has to be smooth in
  // the robot's 1 kHz clock, and a scheduling hiccup on the PC must not turn into a velocity
  // step that libfranka's motion generator would reject.
  const double dt = cycle_time_;

  if (first_update_) {
    // Hold wherever the arm is until somebody commands something.
    position_command_ = q_current;
    position_command_previous_ = q_current;
    velocity_command_.setZero();
    limiter_q_ = q_current;
    limiter_dq_.setZero();
    limiter_ddq_.setZero();
    tau_command_previous_.setZero();
    dq_filtered_.setZero();
    rt_phase_ = Phase::kIdle;
    first_update_ = false;
    for (int i = 0; i < kNumJoints; ++i) {
      command_snapshot_[i].store(position_command_(i), std::memory_order_relaxed);
    }
    command_initialized_.store(true, std::memory_order_release);
  }

  const Command* command = command_buffer_.readFromRT();
  if (command != nullptr && command->id != rt_command_id_) {
    rt_command_id_ = command->id;
    switch (command->kind) {
      case CommandKind::kGoto:
        blend_start_ = position_command_;
        blend_target_ = Vector7d(command->target.data());
        rt_duration_ = command->duration;
        rt_elapsed_ = 0.0;
        rt_phase_ = Phase::kGoto;
        break;
      case CommandKind::kTrajectory:
        rt_trajectory_ = command->trajectory;
        rt_segment_hint_ = 0;
        rt_duration_ = command->duration;
        rt_elapsed_ = 0.0;
        rt_phase_ = Phase::kTrajectory;
        break;
      case CommandKind::kAbort:
        if (rt_phase_ != Phase::kIdle) {
          blend_start_ = position_command_;
          stop_velocity_ = velocity_command_;
          rt_duration_ = abort_stop_duration_;
          rt_elapsed_ = 0.0;
          rt_phase_ = Phase::kStopping;
        }
        break;
      case CommandKind::kNone:
        break;
    }
    if (rt_phase_ != Phase::kIdle) {
      active_command_id_.store(command->id);
      rate_limit_engaged_last_command_.store(0);
    }
  }

  bool finished = false;
  switch (rt_phase_) {
    case Phase::kIdle:
      break;
    case Phase::kGoto: {
      rt_elapsed_ += dt;
      const double s = std::clamp(rt_elapsed_ / rt_duration_, 0.0, 1.0);
      position_command_ = blend_start_ + (blend_target_ - blend_start_) * quintic_blend(s);
      if (s >= 1.0) {
        position_command_ = blend_target_;
        finished = true;
      }
      break;
    }
    case Phase::kTrajectory: {
      rt_elapsed_ += dt;
      std::array<double, 7> sample{};
      sample_trajectory(*rt_trajectory_, rt_elapsed_, rt_segment_hint_, sample);
      position_command_ = Vector7d(sample.data());
      if (rt_elapsed_ >= rt_duration_) {
        position_command_ = Vector7d(rt_trajectory_->positions.back().data());
        finished = true;
      }
      break;
    }
    case Phase::kStopping: {
      // Velocity goes from v0 to zero along a smoothstep, so the position ends at
      // q0 + v0 * T / 2 with zero velocity and zero acceleration.
      rt_elapsed_ += dt;
      const double tau = std::clamp(rt_elapsed_ / rt_duration_, 0.0, 1.0);
      const double integral = tau - tau * tau * tau + 0.5 * tau * tau * tau * tau;
      position_command_ = blend_start_ + stop_velocity_ * rt_duration_ * integral;
      if (tau >= 1.0) {
        finished = true;
      }
      break;
    }
  }
  if (finished) {
    completed_command_id_.store(rt_command_id_);
    rt_phase_ = Phase::kIdle;
    rt_trajectory_.reset();
  }

  velocity_command_ = (position_command_ - position_command_previous_) / dt;
  position_command_previous_ = position_command_;

  Vector7d output;
  if (effort_mode_) {
    const Vector7d tau_desired = compute_torque_command(position_command_, q_current, dq_current);
    output = saturate_torque_rate(tau_desired, tau_command_previous_);
    tau_command_previous_ = output;
  } else {
    if (rate_limit_) {
      bool engaged = false;
      output = limit_position_rate(position_command_, engaged);
      if (engaged) {
        rate_limit_engaged_.fetch_add(1, std::memory_order_relaxed);
        rate_limit_engaged_last_command_.fetch_add(1, std::memory_order_relaxed);
      }
    } else {
      output = position_command_;
    }
  }
  for (int i = 0; i < kNumJoints; ++i) {
    command_interfaces_[i].set_value(output(i));
    command_snapshot_[i].store(position_command_(i), std::memory_order_relaxed);
  }

  phase_.store(static_cast<int>(rt_phase_));
  phase_elapsed_.store(rt_elapsed_);
  phase_duration_.store(rt_duration_);

  if (state_publisher_ && state_publisher_->trylock()) {
    auto& msg = state_publisher_->msg_;
    msg.header.stamp = time;
    for (int i = 0; i < kNumJoints; ++i) {
      msg.reference.positions[i] = position_command_(i);
      msg.reference.velocities[i] = velocity_command_(i);
      msg.feedback.positions[i] = q_current(i);
      msg.feedback.velocities[i] = dq_current(i);
      msg.feedback.effort[i] = joint_efforts_current_[i];
      msg.error.positions[i] = position_command_(i) - q_current(i);
      msg.error.velocities[i] = velocity_command_(i) - dq_current(i);
      if (effort_mode_) {
        msg.output.effort[i] = output(i);
      } else {
        msg.output.positions[i] = output(i);
      }
    }
    // The phase clock: trajectory time during replay, ramp time during goto, 0 when idle.
    msg.reference.time_from_start =
        rclcpp::Duration::from_seconds(rt_phase_ == Phase::kIdle ? 0.0 : rt_elapsed_);
    // The phase itself, so a recording can be split without cross-referencing ~/status.
    msg.output.time_from_start = rclcpp::Duration::from_seconds(static_cast<double>(rt_phase_));
    state_publisher_->unlockAndPublish();
  }

  return controller_interface::return_type::OK;
}

// --- lifecycle ----------------------------------------------------------------------------

CallbackReturn TrajectoryReplayController::on_init() {
  try {
    auto_declare<std::string>("robot_type", "fr3");
    auto_declare<std::string>("arm_prefix", "");
    auto_declare<std::string>("command_interface", "position");
    auto_declare<bool>("coriolis_compensation", true);
    auto_declare<bool>("rate_limit", true);
    auto_declare<double>("torque_rate_limit", 1.0);
    auto_declare<double>("goto_max_velocity", 0.5);
    auto_declare<double>("goto_max_acceleration", 1.0);
    auto_declare<double>("goto_min_duration", 5.0);
    auto_declare<double>("max_joint_step", 3.0);
    auto_declare<double>("max_trajectory_start_error", 0.05);
    auto_declare<double>("trajectory_velocity_scale", 1.0);
    auto_declare<double>("trajectory_acceleration_scale", 1.0);
    auto_declare<double>("abort_stop_duration", 0.5);
    auto_declare<double>("status_rate", 50.0);
    auto_declare<std::vector<double>>("position_limits_lower",
                                      std::vector<double>(kPositionLower.begin(), kPositionLower.end()));
    auto_declare<std::vector<double>>("position_limits_upper",
                                      std::vector<double>(kPositionUpper.begin(), kPositionUpper.end()));
    auto_declare<std::vector<double>>("max_acceleration",
                                      std::vector<double>(7, kMaxJointAccelerationDefault));
    auto_declare<std::vector<double>>("max_jerk", std::vector<double>(7, kMaxJointJerkDefault));
    auto_declare<std::vector<double>>("k_gains",
                                      {600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0});
    auto_declare<std::vector<double>>("d_gains", {30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 5.0});

    // Collision thresholds, the values franka_example_controllers/default_robot_behavior_utils.hpp
    // installs before the upstream examples run.
    auto_declare<bool>("set_collision_behavior", true);
    auto_declare<std::vector<double>>("lower_torque_thresholds_nominal",
                                      {25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0});
    auto_declare<std::vector<double>>("upper_torque_thresholds_nominal",
                                      {35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0});
    auto_declare<std::vector<double>>("lower_torque_thresholds_acceleration",
                                      {25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0});
    auto_declare<std::vector<double>>("upper_torque_thresholds_acceleration",
                                      {35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0});
    auto_declare<std::vector<double>>("lower_force_thresholds_nominal",
                                      {30.0, 30.0, 30.0, 25.0, 25.0, 25.0});
    auto_declare<std::vector<double>>("upper_force_thresholds_nominal",
                                      {40.0, 40.0, 40.0, 35.0, 35.0, 35.0});
    auto_declare<std::vector<double>>("lower_force_thresholds_acceleration",
                                      {30.0, 30.0, 30.0, 25.0, 25.0, 25.0});
    auto_declare<std::vector<double>>("upper_force_thresholds_acceleration",
                                      {40.0, 40.0, 40.0, 35.0, 35.0, 35.0});
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Exception during on_init: %s", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

bool TrajectoryReplayController::assign_parameters() {
  const auto node = get_node();
  robot_type_ = node->get_parameter("robot_type").as_string();
  arm_prefix_ = node->get_parameter("arm_prefix").as_string();
  arm_prefix_ = arm_prefix_.empty() ? "" : arm_prefix_ + "_";

  const auto mode = node->get_parameter("command_interface").as_string();
  if (mode == "position") {
    effort_mode_ = false;
  } else if (mode == "effort") {
    effort_mode_ = true;
  } else {
    RCLCPP_FATAL(node->get_logger(), "command_interface must be 'position' or 'effort', got '%s'",
                 mode.c_str());
    return false;
  }
  coriolis_compensation_ = node->get_parameter("coriolis_compensation").as_bool();
  rate_limit_ = node->get_parameter("rate_limit").as_bool();
  torque_rate_limit_ = node->get_parameter("torque_rate_limit").as_double();
  goto_max_velocity_ = node->get_parameter("goto_max_velocity").as_double();
  goto_max_acceleration_ = node->get_parameter("goto_max_acceleration").as_double();
  goto_min_duration_ = node->get_parameter("goto_min_duration").as_double();
  max_joint_step_ = node->get_parameter("max_joint_step").as_double();
  max_trajectory_start_error_ = node->get_parameter("max_trajectory_start_error").as_double();
  trajectory_velocity_scale_ = node->get_parameter("trajectory_velocity_scale").as_double();
  trajectory_acceleration_scale_ = node->get_parameter("trajectory_acceleration_scale").as_double();
  abort_stop_duration_ = node->get_parameter("abort_stop_duration").as_double();

  if (!(goto_max_velocity_ > 0.0) || !(goto_max_acceleration_ > 0.0) ||
      !(goto_min_duration_ > 0.0) || !(abort_stop_duration_ > 0.0)) {
    RCLCPP_FATAL(node->get_logger(), "goto_* and abort_stop_duration must all be > 0");
    return false;
  }

  const auto fill7 = [&node](const char* name, std::array<double, 7>& destination) {
    const auto values = node->get_parameter(name).as_double_array();
    if (values.size() != 7) {
      RCLCPP_FATAL(node->get_logger(), "%s must have 7 entries, got %zu", name, values.size());
      return false;
    }
    std::copy(values.begin(), values.end(), destination.begin());
    return true;
  };
  std::array<double, 7> k{};
  std::array<double, 7> d{};
  if (!fill7("position_limits_lower", position_limits_lower_) ||
      !fill7("position_limits_upper", position_limits_upper_) ||
      !fill7("max_acceleration", max_acceleration_) || !fill7("max_jerk", max_jerk_) ||
      !fill7("k_gains", k) || !fill7("d_gains", d)) {
    return false;
  }
  k_gains_ = Vector7d(k.data());
  d_gains_ = Vector7d(d.data());
  return true;
}

bool TrajectoryReplayController::apply_collision_behavior() {
  auto request = std::make_shared<franka_msgs::srv::SetFullCollisionBehavior::Request>();
  const auto fill = [this](const char* name, auto& destination) {
    const auto values = get_node()->get_parameter(name).as_double_array();
    if (values.size() != destination.size()) {
      RCLCPP_FATAL(get_node()->get_logger(), "%s must have %zu entries, got %zu.", name,
                   destination.size(), values.size());
      return false;
    }
    std::copy(values.begin(), values.end(), destination.begin());
    return true;
  };
  if (!fill("lower_torque_thresholds_nominal", request->lower_torque_thresholds_nominal) ||
      !fill("upper_torque_thresholds_nominal", request->upper_torque_thresholds_nominal) ||
      !fill("lower_torque_thresholds_acceleration",
            request->lower_torque_thresholds_acceleration) ||
      !fill("upper_torque_thresholds_acceleration",
            request->upper_torque_thresholds_acceleration) ||
      !fill("lower_force_thresholds_nominal", request->lower_force_thresholds_nominal) ||
      !fill("upper_force_thresholds_nominal", request->upper_force_thresholds_nominal) ||
      !fill("lower_force_thresholds_acceleration", request->lower_force_thresholds_acceleration) ||
      !fill("upper_force_thresholds_acceleration", request->upper_force_thresholds_acceleration)) {
    return false;
  }
  if (!collision_client_->wait_for_service(5s)) {
    // franka_hardware's service_server is only there on the real arm. Under mock or simulated
    // hardware there is nothing to set, and on the real arm nothing moves without
    // franka_hardware anyway - so this is a warning, not a failure.
    RCLCPP_WARN(get_node()->get_logger(),
                "service_server/set_full_collision_behavior is not available (mock or simulated "
                "hardware?). Collision thresholds stay whatever the robot has; on the real arm "
                "that is what Desk last set, which may be low enough to reflex.");
    return true;
  }
  auto future = collision_client_->async_send_request(request);
  if (future.wait_for(10s) != std::future_status::ready) {
    RCLCPP_FATAL(get_node()->get_logger(), "set_full_collision_behavior did not respond.");
    return false;
  }
  if (!future.get()->success) {
    RCLCPP_FATAL(get_node()->get_logger(), "set_full_collision_behavior was rejected.");
    return false;
  }
  RCLCPP_INFO(get_node()->get_logger(),
              "Collision behavior set (upper force thresholds %.0f/%.0f/%.0f N).",
              request->upper_force_thresholds_nominal[0],
              request->upper_force_thresholds_nominal[1],
              request->upper_force_thresholds_nominal[2]);
  return true;
}

CallbackReturn TrajectoryReplayController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  if (!assign_parameters()) {
    return CallbackReturn::FAILURE;
  }

  franka_robot_model_.reset();
  if (effort_mode_ && coriolis_compensation_) {
    franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
        arm_prefix_ + robot_type_ + "/robot_model", arm_prefix_ + robot_type_ + "/robot_state");
  }

  collision_client_ = get_node()->create_client<franka_msgs::srv::SetFullCollisionBehavior>(
      "service_server/set_full_collision_behavior");
  if (get_node()->get_parameter("set_collision_behavior").as_bool() &&
      !apply_collision_behavior()) {
    return CallbackReturn::FAILURE;
  }

  goto_subscriber_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
      "~/goto", rclcpp::QoS(1),
      [this](const sensor_msgs::msg::JointState::SharedPtr msg) { goto_callback(msg); });
  trajectory_subscriber_ = get_node()->create_subscription<trajectory_msgs::msg::JointTrajectory>(
      "~/trajectory", rclcpp::QoS(1).reliable(),
      [this](const trajectory_msgs::msg::JointTrajectory::SharedPtr msg) {
        trajectory_callback(msg);
      });
  abort_subscriber_ = get_node()->create_subscription<std_msgs::msg::Empty>(
      "~/abort", rclcpp::QoS(1),
      [this](const std_msgs::msg::Empty::SharedPtr msg) { abort_callback(msg); });

  status_publisher_ = get_node()->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      "~/status", rclcpp::QoS(10));
  const double status_rate = get_node()->get_parameter("status_rate").as_double();
  const auto status_period = std::chrono::duration<double>(1.0 / std::max(status_rate, 1.0));
  status_timer_ = get_node()->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(status_period),
      [this]() { publish_status(); });

  auto publisher = get_node()->create_publisher<control_msgs::msg::JointTrajectoryControllerState>(
      "~/controller_state", rclcpp::SystemDefaultsQoS());
  state_publisher_ = std::make_unique<
      realtime_tools::RealtimePublisher<control_msgs::msg::JointTrajectoryControllerState>>(
      publisher);
  {
    auto& msg = state_publisher_->msg_;
    msg.joint_names = joint_names();
    msg.reference.positions.assign(kNumJoints, 0.0);
    msg.reference.velocities.assign(kNumJoints, 0.0);
    msg.feedback.positions.assign(kNumJoints, 0.0);
    msg.feedback.velocities.assign(kNumJoints, 0.0);
    msg.feedback.effort.assign(kNumJoints, 0.0);
    msg.error.positions.assign(kNumJoints, 0.0);
    msg.error.velocities.assign(kNumJoints, 0.0);
    msg.output.positions.assign(kNumJoints, 0.0);
    msg.output.effort.assign(kNumJoints, 0.0);
  }

  RCLCPP_INFO(get_node()->get_logger(),
              "Configured: %s mode%s, rate limiter %s, goto <= %.2f rad/s / %.2f rad/s^2, "
              "max joint step %.2f rad.",
              effort_mode_ ? "effort (joint impedance law)" : "position (robot-internal joint impedance)",
              (effort_mode_ && coriolis_compensation_) ? " with coriolis compensation" : "",
              (rate_limit_ && !effort_mode_) ? "on" : "off", goto_max_velocity_,
              goto_max_acceleration_, max_joint_step_);
  return CallbackReturn::SUCCESS;
}

CallbackReturn TrajectoryReplayController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  if (state_interfaces_.size() < 3 * kNumJoints) {
    RCLCPP_FATAL(get_node()->get_logger(), "Got %zu state interfaces, expected at least %d.",
                 state_interfaces_.size(), 3 * kNumJoints);
    return CallbackReturn::ERROR;
  }
  const unsigned int update_rate = get_update_rate();
  cycle_time_ = update_rate > 0 ? 1.0 / static_cast<double>(update_rate) : 1e-3;
  RCLCPP_INFO(get_node()->get_logger(), "Cycle time %.4f s (controller manager update rate %u Hz).",
              cycle_time_, update_rate);
  first_update_ = true;
  command_initialized_.store(false, std::memory_order_release);
  phase_.store(static_cast<int>(Phase::kIdle));
  rt_command_id_ = 0;
  next_command_id_ = 0;
  active_command_id_.store(0);
  completed_command_id_.store(0);
  rate_limit_engaged_.store(0);
  rate_limit_engaged_last_command_.store(0);
  rt_trajectory_.reset();
  command_buffer_.writeFromNonRT(Command{});
  last_rejection_.clear();
  rejections_ = 0;
  for (auto& value : measured_positions_snapshot_) {
    value.store(0.0, std::memory_order_relaxed);
  }
  if (franka_robot_model_) {
    franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  }
  is_active_.store(true, std::memory_order_release);
  return CallbackReturn::SUCCESS;
}

CallbackReturn TrajectoryReplayController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  is_active_.store(false, std::memory_order_release);
  command_initialized_.store(false, std::memory_order_release);
  if (franka_robot_model_) {
    franka_robot_model_->release_interfaces();
  }
  return CallbackReturn::SUCCESS;
}

}  // namespace franka_trajectory_replay

// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_trajectory_replay::TrajectoryReplayController,
                       controller_interface::ControllerInterface)
