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

#include <franka_trajectory_replay/cartesian_trajectory_replay_controller.hpp>
#include <franka_trajectory_replay/collision_behavior.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <tuple>
#include <vector>

#include <pluginlib/class_list_macros.hpp>

using namespace std::chrono_literals;

namespace franka_trajectory_replay {

namespace {

// FR3 joint position limits, franka_description/robots/fr3/joint_limits.yaml.
constexpr std::array<double, 7> kPositionLower{-2.9007, -1.8361, -2.9007, -3.0770,
                                               -2.8763, 0.4398,  -3.0508};
constexpr std::array<double, 7> kPositionUpper{2.9007, 1.8361, 2.9007, -0.1169,
                                               2.8763, 4.6216, 3.0508};

std::string format_pose(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation) {
  char buffer[160];
  std::snprintf(buffer, sizeof(buffer), "[%.4f %.4f %.4f | %.4f %.4f %.4f %.4f]", position.x(),
                position.y(), position.z(), orientation.x(), orientation.y(), orientation.z(),
                orientation.w());
  return buffer;
}

std::string format_joints(const std::array<double, 7>& values) {
  char buffer[128];
  std::snprintf(buffer, sizeof(buffer), "[%.3f %.3f %.3f %.3f %.3f %.3f %.3f]", values[0],
                values[1], values[2], values[3], values[4], values[5], values[6]);
  return buffer;
}

Eigen::Quaterniond to_quaternion(const std::array<double, 4>& xyzw) {
  return Eigen::Quaterniond(xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
}

std::array<double, 4> to_array(const Eigen::Quaterniond& q) {
  return {q.x(), q.y(), q.z(), q.w()};
}

void fill_pose(geometry_msgs::msg::Pose& pose, const Eigen::Vector3d& position,
               const Eigen::Quaterniond& orientation) {
  pose.position.x = position.x();
  pose.position.y = position.y();
  pose.position.z = position.z();
  pose.orientation.x = orientation.x();
  pose.orientation.y = orientation.y();
  pose.orientation.z = orientation.z();
  pose.orientation.w = orientation.w();
}

bool read_pose(const geometry_msgs::msg::Pose& pose, Eigen::Vector3d& position,
               Eigen::Quaterniond& orientation) {
  position = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  orientation = Eigen::Quaterniond(pose.orientation.w, pose.orientation.x, pose.orientation.y,
                                   pose.orientation.z);
  if (!position.allFinite() || !orientation.coeffs().allFinite()) {
    return false;
  }
  if (orientation.coeffs().norm() < 1e-6) {
    return false;
  }
  orientation.normalize();
  return true;
}

}  // namespace

const char* CartesianTrajectoryReplayController::phase_name(Phase phase) {
  switch (phase) {
    case Phase::kIdle:
      return "idle";
    case Phase::kGoto:
      return "goto";
    case Phase::kTrajectory:
      return "trajectory";
    case Phase::kStopping:
      return "stopping";
    case Phase::kPolicy:
      return "policy";
  }
  return "unknown";
}

double CartesianTrajectoryReplayController::quintic_blend(double s) {
  return s * s * s * (10.0 + s * (-15.0 + 6.0 * s));
}

void CartesianTrajectoryReplayController::sample_trajectory(
    const Trajectory& trajectory, double t, size_t& segment_hint, Eigen::Vector3d& position,
    Eigen::Quaterniond& orientation, Vector7d& nullspace) {
  const size_t n = trajectory.times.size();
  if (n == 0) {
    return;
  }
  const auto take = [&](size_t k) {
    position = Eigen::Vector3d(trajectory.positions[k].data());
    orientation = to_quaternion(trajectory.orientations[k]);
    nullspace = Vector7d(trajectory.nullspace[k].data());
  };
  if (t <= trajectory.times.front()) {
    take(0);
    segment_hint = 0;
    return;
  }
  if (t >= trajectory.times.back()) {
    take(n - 1);
    segment_hint = n - 1;
    return;
  }
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
  const Eigen::Vector3d p0(trajectory.positions[i].data());
  const Eigen::Vector3d p1(trajectory.positions[i + 1].data());
  if (trajectory.has_velocities) {
    const Eigen::Vector3d v0(trajectory.velocities[i].data());
    const Eigen::Vector3d v1(trajectory.velocities[i + 1].data());
    const double s2 = s * s;
    const double s3 = s2 * s;
    const double h00 = 2 * s3 - 3 * s2 + 1;
    const double h10 = s3 - 2 * s2 + s;
    const double h01 = -2 * s3 + 3 * s2;
    const double h11 = s3 - s2;
    position = h00 * p0 + h10 * h * v0 + h01 * p1 + h11 * h * v1;
  } else {
    position = p0 + (p1 - p0) * s;
  }
  const Eigen::Quaterniond q0 = to_quaternion(trajectory.orientations[i]);
  Eigen::Quaterniond q1 = to_quaternion(trajectory.orientations[i + 1]);
  if (q0.coeffs().dot(q1.coeffs()) < 0.0) {
    q1.coeffs() = -q1.coeffs();
  }
  orientation = q0.slerp(s, q1);
  orientation.normalize();
  const Vector7d n0(trajectory.nullspace[i].data());
  const Vector7d n1(trajectory.nullspace[i + 1].data());
  nullspace = n0 + (n1 - n0) * s;
}

// --- interfaces ---------------------------------------------------------------------------

std::vector<std::string> CartesianTrajectoryReplayController::joint_names() const {
  std::vector<std::string> names;
  names.reserve(kNumJoints);
  for (int i = 1; i <= kNumJoints; ++i) {
    names.push_back(arm_prefix_ + arm_id_ + "_joint" + std::to_string(i));
  }
  return names;
}

controller_interface::InterfaceConfiguration
CartesianTrajectoryReplayController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto& joint : joint_names()) {
    config.names.push_back(joint + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
CartesianTrajectoryReplayController::state_interface_configuration() const {
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
  if (franka_cartesian_pose_) {
    for (const auto& name : franka_cartesian_pose_->get_state_interface_names()) {
      config.names.push_back(name);
    }
  }
  if (franka_robot_model_) {
    for (const auto& name : franka_robot_model_->get_state_interface_names()) {
      config.names.push_back(name);
    }
  }
  return config;
}

void CartesianTrajectoryReplayController::update_joint_states() {
  for (int i = 0; i < kNumJoints; ++i) {
    joint_positions_current_[i] = state_interfaces_.at(kPositionOffset + i).get_value();
    joint_velocities_current_[i] = state_interfaces_.at(kVelocityOffset + i).get_value();
    joint_efforts_current_[i] = state_interfaces_.at(kEffortOffset + i).get_value();
  }
}

// --- law helpers --------------------------------------------------------------------------

Vector7d CartesianTrajectoryReplayController::saturate_torque_rate(const Vector7d& tau_desired,
                                                          const Vector7d& tau_previous) const {
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

void CartesianTrajectoryReplayController::update_gains(bool initialise) {
  const GainSettings* settings = gain_settings_buffer_.readFromRT();
  if (settings != nullptr && (initialise || settings->revision != rt_gain_revision_)) {
    std::array<double, 6> scaled{};
    for (int i = 0; i < 6; ++i) {
      scaled[i] = settings->stiffness_scale * settings->stiffness[i];
    }
    example_cartesian_gains(scaled, stiffness_target_, damping_target_);
    nullspace_stiffness_target_ = settings->nullspace_stiffness;
    filter_alpha_ = settings->target_filter;
    rt_gain_revision_ = settings->revision;
    if (initialise) {
      stiffness_ = stiffness_target_;
      damping_ = damping_target_;
      nullspace_stiffness_ = nullspace_stiffness_target_;
    }
  }
}

void CartesianTrajectoryReplayController::begin_stopping() {
  if (rt_phase_ == Phase::kIdle) {
    return;
  }
  if (rt_phase_ != Phase::kStopping) {
    rt_stopping_from_ = rt_phase_;
  }
  rt_rate_ramp_start_ = rt_playback_rate_;
  rt_playback_target_ = 0.0;
  rt_rate_ramp_elapsed_ = 0.0;
  rt_rate_ramp_duration_ = abort_stop_duration_;
  rt_phase_ = Phase::kStopping;
}

void CartesianTrajectoryReplayController::advance_clock(double dt, double requested_rate) {
  if (rt_phase_ != Phase::kStopping && requested_rate != rt_playback_target_) {
    rt_rate_ramp_start_ = rt_playback_rate_;
    rt_playback_target_ = requested_rate;
    rt_rate_ramp_elapsed_ = 0.0;
    rt_rate_ramp_duration_ = pause_ramp_duration_;
  }
  const double previous_rate = rt_playback_rate_;
  if (rt_rate_ramp_elapsed_ < rt_rate_ramp_duration_) {
    rt_rate_ramp_elapsed_ = std::min(rt_rate_ramp_elapsed_ + dt, rt_rate_ramp_duration_);
    const double s = rt_rate_ramp_elapsed_ / rt_rate_ramp_duration_;
    rt_playback_rate_ =
        rt_rate_ramp_start_ + (rt_playback_target_ - rt_rate_ramp_start_) * quintic_blend(s);
  } else {
    rt_playback_rate_ = rt_playback_target_;
  }
  // Trapezoidal integration avoids a one-cycle clock jump at either end of the ramp.
  rt_elapsed_ += 0.5 * (previous_rate + rt_playback_rate_) * dt;
}

// --- non-realtime input -------------------------------------------------------------------

void CartesianTrajectoryReplayController::reject(const std::string& reason) {
  ++rejections_;
  last_rejection_ = reason;
  RCLCPP_ERROR(get_node()->get_logger(), "Rejected: %s", reason.c_str());
}

bool CartesianTrajectoryReplayController::inside_workspace(const Eigen::Vector3d& position) const {
  for (int i = 0; i < 3; ++i) {
    if (position(i) < workspace_min_[i] || position(i) > workspace_max_[i]) {
      return false;
    }
  }
  return true;
}

bool CartesianTrajectoryReplayController::read_nullspace(const std::vector<double>& values,
                                                         std::array<double, 7>& out,
                                                         const std::string& source) {
  if (values.size() != static_cast<size_t>(kNumJoints)) {
    reject(source + " nullspace_positions must hold exactly 7 values or be empty");
    return false;
  }
  for (int i = 0; i < kNumJoints; ++i) {
    if (!std::isfinite(values[i])) {
      reject(source + " nullspace position is not finite");
      return false;
    }
    if (values[i] < position_limits_lower_[i] || values[i] > position_limits_upper_[i]) {
      reject(source + " nullspace position for joint " + std::to_string(i + 1) + " (" +
             std::to_string(values[i]) + ") is outside the position limits");
      return false;
    }
    out[i] = values[i];
  }
  return true;
}

void CartesianTrajectoryReplayController::goto_callback(
    const franka_trajectory_replay_msgs::msg::CartesianGoto::SharedPtr msg) {
  if (!command_initialized_.load(std::memory_order_acquire)) {
    reject("goto arrived before the first update cycle");
    return;
  }
  if (static_cast<Phase>(phase_.load()) != Phase::kIdle) {
    reject("goto while busy (phase " +
           std::string(phase_name(static_cast<Phase>(phase_.load()))) + "); abort first");
    return;
  }
  Eigen::Vector3d position;
  Eigen::Quaterniond orientation;
  if (!read_pose(msg->pose, position, orientation)) {
    reject("goto pose is not finite or has a degenerate quaternion");
    return;
  }
  if (!inside_workspace(position)) {
    reject("goto target " + format_pose(position, orientation) + " is outside the workspace box");
    return;
  }
  if (!std::isfinite(msg->duration) || msg->duration < 0.0) {
    reject("goto duration must be finite and non-negative");
    return;
  }
  Eigen::Vector3d current_position;
  std::array<double, 4> current_orientation{};
  for (int i = 0; i < 3; ++i) {
    current_position(i) = target_position_snapshot_[i].load(std::memory_order_relaxed);
  }
  for (int i = 0; i < 4; ++i) {
    current_orientation[i] = target_orientation_snapshot_[i].load(std::memory_order_relaxed);
  }
  const double step = (position - current_position).norm();
  const double angle = quaternion_angle(to_quaternion(current_orientation), orientation);
  if (step > max_goto_step_m_ || angle > max_goto_step_rad_) {
    reject("goto target is " + std::to_string(step) + " m / " + std::to_string(angle) +
           " rad from the current reference, which exceeds max_goto_step");
    return;
  }

  Command command;
  command.kind = CommandKind::kGoto;
  for (int i = 0; i < 3; ++i) {
    command.position[i] = position(i);
  }
  command.orientation = to_array(orientation);
  double nullspace_step = 0.0;
  if (!msg->nullspace_positions.empty()) {
    if (!read_nullspace(msg->nullspace_positions, command.nullspace, "goto")) {
      return;
    }
    command.has_nullspace = true;
    for (int i = 0; i < kNumJoints; ++i) {
      nullspace_step = std::max(
          nullspace_step,
          std::abs(command.nullspace[i] - target_nullspace_snapshot_[i].load(std::memory_order_relaxed)));
    }
  }
  // Quintic peak velocity is 1.875 * step / T.
  double duration = std::max(goto_min_duration_, msg->duration);
  duration = std::max(duration, 1.875 * step / goto_max_velocity_);
  duration = std::max(duration, 1.875 * angle / goto_max_angular_velocity_);
  duration = std::max(duration, 1.875 * nullspace_step / goto_max_nullspace_velocity_);
  command.duration = duration;
  command.id = ++next_command_id_;
  command_buffer_.writeFromNonRT(command);
  RCLCPP_INFO(get_node()->get_logger(),
              "Accepted goto (command %lu): %.4f m / %.4f rad / %.4f rad nullspace step, "
              "duration %.2f s, target %s",
              static_cast<unsigned long>(command.id), step, angle, nullspace_step, duration,
              format_pose(position, orientation).c_str());
}

void CartesianTrajectoryReplayController::trajectory_callback(
    const franka_trajectory_replay_msgs::msg::CartesianTrajectory::SharedPtr msg) {
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
  if (msg->header.frame_id != base_frame_) {
    reject("trajectory frame_id '" + msg->header.frame_id + "' is not the base frame '" +
           base_frame_ + "'");
    return;
  }

  auto trajectory = std::make_shared<Trajectory>();
  const size_t n = msg->points.size();
  trajectory->times.reserve(n);
  trajectory->positions.reserve(n);
  trajectory->orientations.reserve(n);
  trajectory->nullspace.reserve(n);
  trajectory->has_velocities = std::any_of(msg->points.begin(), msg->points.end(), [](const auto& point) {
    const auto& v = point.twist.linear;
    return v.x != 0.0 || v.y != 0.0 || v.z != 0.0;
  });
  if (trajectory->has_velocities) {
    trajectory->velocities.reserve(n);
  }

  std::array<double, 7> nullspace{};
  for (int i = 0; i < kNumJoints; ++i) {
    nullspace[i] = target_nullspace_snapshot_[i].load(std::memory_order_relaxed);
  }
  double previous_time = -1.0;
  Eigen::Vector3d previous_position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond previous_orientation = Eigen::Quaterniond::Identity();
  double peak_linear_ratio = 0.0;
  double peak_angular_ratio = 0.0;
  const double linear_limit = trajectory_velocity_scale_ * kMaxTranslationalVelocity;
  const double angular_limit = trajectory_velocity_scale_ * kMaxRotationalVelocity;
  for (size_t k = 0; k < n; ++k) {
    const auto& point = msg->points[k];
    const double t = rclcpp::Duration(point.time_from_start).seconds();
    if (!(t >= 0.0) || t <= previous_time) {
      reject("trajectory times must be non-negative and strictly increasing (point " +
             std::to_string(k) + ")");
      return;
    }
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    if (!read_pose(point.pose, position, orientation)) {
      reject("trajectory point " + std::to_string(k) +
             " is not finite or has a degenerate quaternion");
      return;
    }
    if (!inside_workspace(position)) {
      reject("trajectory point " + std::to_string(k) + " " + format_pose(position, orientation) +
             " is outside the workspace box");
      return;
    }
    if (k > 0 && orientation.coeffs().dot(previous_orientation.coeffs()) < 0.0) {
      orientation.coeffs() = -orientation.coeffs();
    }
    if (!point.nullspace_positions.empty() &&
        !read_nullspace(point.nullspace_positions, nullspace,
                        "trajectory point " + std::to_string(k))) {
      return;
    }
    std::array<double, 3> velocity{};
    if (trajectory->has_velocities) {
      velocity = {point.twist.linear.x, point.twist.linear.y, point.twist.linear.z};
      if (!std::isfinite(velocity[0]) || !std::isfinite(velocity[1]) || !std::isfinite(velocity[2])) {
        reject("trajectory point " + std::to_string(k) + " velocity is not finite");
        return;
      }
    }
    if (k > 0) {
      const double h = t - previous_time;
      peak_linear_ratio =
          std::max(peak_linear_ratio, (position - previous_position).norm() / h / linear_limit);
      peak_angular_ratio = std::max(
          peak_angular_ratio, quaternion_angle(previous_orientation, orientation) / h / angular_limit);
    }
    previous_time = t;
    previous_position = position;
    previous_orientation = orientation;
    trajectory->times.push_back(t);
    trajectory->positions.push_back({position.x(), position.y(), position.z()});
    trajectory->orientations.push_back(to_array(orientation));
    trajectory->nullspace.push_back(nullspace);
    if (trajectory->has_velocities) {
      trajectory->velocities.push_back(velocity);
    }
  }

  Eigen::Vector3d current_position;
  std::array<double, 4> current_orientation{};
  for (int i = 0; i < 3; ++i) {
    current_position(i) = target_position_snapshot_[i].load(std::memory_order_relaxed);
  }
  for (int i = 0; i < 4; ++i) {
    current_orientation[i] = target_orientation_snapshot_[i].load(std::memory_order_relaxed);
  }
  const Eigen::Vector3d first_position(trajectory->positions.front().data());
  const double start_error_m = (first_position - current_position).norm();
  const double start_error_rad =
      quaternion_angle(to_quaternion(current_orientation), to_quaternion(trajectory->orientations.front()));
  if (start_error_m > max_trajectory_start_error_m_ || start_error_rad > max_trajectory_start_error_rad_) {
    reject("trajectory starts " + std::to_string(start_error_m) + " m / " +
           std::to_string(start_error_rad) +
           " rad away from the current reference (max_trajectory_start_error); goto its first "
           "point first");
    return;
  }
  if (peak_linear_ratio > 1.0) {
    reject("trajectory exceeds the Cartesian translational velocity limit (" +
           std::to_string(100.0 * peak_linear_ratio) + " % of the limit)");
    return;
  }
  if (peak_angular_ratio > 1.0) {
    reject("trajectory exceeds the Cartesian rotational velocity limit (" +
           std::to_string(100.0 * peak_angular_ratio) + " % of the limit)");
    return;
  }

  Command command;
  command.kind = CommandKind::kTrajectory;
  command.trajectory = trajectory;
  command.duration = trajectory->times.back();
  command.id = ++next_command_id_;
  command_buffer_.writeFromNonRT(command);
  RCLCPP_INFO(get_node()->get_logger(),
              "Accepted trajectory (command %lu): %zu points, %.2f s, %s position interpolation, "
              "peak %.0f %% of the translational and %.0f %% of the rotational velocity limit, "
              "start error %.4f m / %.4f rad",
              static_cast<unsigned long>(command.id), n, command.duration,
              trajectory->has_velocities ? "cubic Hermite" : "linear", 100.0 * peak_linear_ratio,
              100.0 * peak_angular_ratio, start_error_m, start_error_rad);
}

void CartesianTrajectoryReplayController::policy_command_callback(
    const franka_trajectory_replay_msgs::msg::CartesianGoto::SharedPtr msg) {
  if (!command_initialized_.load(std::memory_order_acquire)) {
    reject("policy command arrived before the first update cycle");
    return;
  }
  const auto phase = static_cast<Phase>(phase_.load(std::memory_order_acquire));
  if (phase != Phase::kIdle && phase != Phase::kPolicy) {
    reject("policy command while busy (phase " + std::string(phase_name(phase)) +
           "); abort first");
    return;
  }
  Eigen::Vector3d position;
  Eigen::Quaterniond orientation;
  if (!read_pose(msg->pose, position, orientation)) {
    reject("policy command pose is not finite or has a degenerate quaternion");
    return;
  }
  if (!inside_workspace(position)) {
    reject("policy target " + format_pose(position, orientation) +
           " is outside the workspace box");
    return;
  }

  Eigen::Vector3d previous;
  for (int i = 0; i < 3; ++i) {
    previous(i) = target_position_snapshot_[i].load(std::memory_order_relaxed);
  }
  std::array<double, 4> previous_q{};
  for (int i = 0; i < 4; ++i) {
    previous_q[i] = target_orientation_snapshot_[i].load(std::memory_order_relaxed);
  }
  const double position_step = (position - previous).norm();
  const double orientation_step = quaternion_angle(orientation, to_quaternion(previous_q));
  if (position_step > max_policy_step_m_ || orientation_step > max_policy_step_rad_) {
    reject("policy target step " + std::to_string(position_step) + " m / " +
           std::to_string(orientation_step) + " rad exceeds max_policy_step_m/rad");
    return;
  }

  Command command;
  command.kind = CommandKind::kPolicy;
  std::copy(position.data(), position.data() + 3, command.position.begin());
  command.orientation = to_array(orientation);
  if (!msg->nullspace_positions.empty()) {
    if (!read_nullspace(msg->nullspace_positions, command.nullspace, "policy command")) {
      return;
    }
    command.has_nullspace = true;
  }
  command.id = ++next_command_id_;
  command_buffer_.writeFromNonRT(command);
}

void CartesianTrajectoryReplayController::pause_callback(
    const std_msgs::msg::Empty::SharedPtr /*msg*/) {
  if (static_cast<Phase>(phase_.load(std::memory_order_acquire)) != Phase::kTrajectory) {
    reject("pause is only valid while a trajectory is running");
    return;
  }
  pause_requested_.store(true, std::memory_order_release);
  RCLCPP_INFO(get_node()->get_logger(), "Trajectory pause requested.");
}

void CartesianTrajectoryReplayController::resume_callback(
    const std_msgs::msg::Empty::SharedPtr /*msg*/) {
  if (static_cast<Phase>(phase_.load(std::memory_order_acquire)) != Phase::kTrajectory) {
    reject("resume is only valid while a trajectory is running");
    return;
  }
  pause_requested_.store(false, std::memory_order_release);
  RCLCPP_INFO(get_node()->get_logger(), "Trajectory resume requested.");
}

void CartesianTrajectoryReplayController::abort_callback(
    const std_msgs::msg::Empty::SharedPtr /*msg*/) {
  Command command;
  command.kind = CommandKind::kAbort;
  command.id = ++next_command_id_;
  command_buffer_.writeFromNonRT(command);
  RCLCPP_WARN(get_node()->get_logger(), "Abort requested (command %lu)",
              static_cast<unsigned long>(command.id));
}

void CartesianTrajectoryReplayController::publish_status() {
  if (!is_active_.load(std::memory_order_acquire)) {
    return;
  }
  diagnostic_msgs::msg::DiagnosticArray msg;
  msg.header.stamp = get_node()->now();
  diagnostic_msgs::msg::DiagnosticStatus status;
  const auto phase = static_cast<Phase>(phase_.load());
  const auto fault = static_cast<Fault>(fault_.load());
  status.level = (last_rejection_.empty() && fault == Fault::kNone)
                     ? diagnostic_msgs::msg::DiagnosticStatus::OK
                     : diagnostic_msgs::msg::DiagnosticStatus::WARN;
  status.name = std::string(get_node()->get_name());
  status.message = phase_name(phase);
  status.hardware_id = "cartesian_impedance";

  const auto add = [&status](const char* key, const std::string& value) {
    diagnostic_msgs::msg::KeyValue kv;
    kv.key = key;
    kv.value = value;
    status.values.push_back(kv);
  };
  add("phase", std::to_string(static_cast<int>(phase)));
  add("phase_name", phase_name(phase));
  add("command_mode", "cartesian_impedance");
  add("active_command_id", std::to_string(active_command_id_.load()));
  add("processed_command_id", std::to_string(processed_command_id_.load()));
  add("completed_command_id", std::to_string(completed_command_id_.load()));
  add("elapsed", std::to_string(phase_elapsed_.load()));
  add("duration", std::to_string(phase_duration_.load()));
  add("pause_requested", pause_requested_.load() ? "true" : "false");
  add("paused", paused_.load() ? "true" : "false");
  add("playback_rate", std::to_string(playback_rate_.load()));
  add("stiffness_scale_target", std::to_string(stiffness_scale_target_.load()));
  add("translational_stiffness_applied",
      std::to_string(gains_applied_snapshot_[0].load(std::memory_order_relaxed)));
  add("rotational_stiffness_applied",
      std::to_string(gains_applied_snapshot_[1].load(std::memory_order_relaxed)));
  add("nullspace_stiffness_applied",
      std::to_string(gains_applied_snapshot_[2].load(std::memory_order_relaxed)));
  {
    char buffer[160];
    std::snprintf(buffer, sizeof(buffer), "%.4f %.4f %.4f %.4f %.4f %.4f %.4f",
                  reference_pose_snapshot_[0].load(std::memory_order_relaxed),
                  reference_pose_snapshot_[1].load(std::memory_order_relaxed),
                  reference_pose_snapshot_[2].load(std::memory_order_relaxed),
                  reference_pose_snapshot_[3].load(std::memory_order_relaxed),
                  reference_pose_snapshot_[4].load(std::memory_order_relaxed),
                  reference_pose_snapshot_[5].load(std::memory_order_relaxed),
                  reference_pose_snapshot_[6].load(std::memory_order_relaxed));
    add("reference_pose", buffer);
  }
  add("position_error_m", std::to_string(position_error_.load()));
  add("orientation_error_rad", std::to_string(orientation_error_.load()));
  add("joint_limit_margin_rad", std::to_string(joint_limit_margin_.load()));
  add("policy_command_age", std::to_string(policy_command_age_.load()));
  add("policy_watchdog_stop", policy_watchdog_stop_.load() ? "true" : "false");
  add("tracking_fault", fault == Fault::kNone ? "false" : "true");
  add("last_fault", fault == Fault::kNone ? "" : (fault == Fault::kPosition
                                                       ? "position error exceeded max_position_error"
                                                       : "orientation error exceeded max_orientation_error"));
  add("rejections", std::to_string(rejections_));
  add("last_rejection", last_rejection_);
  std::array<double, 7> nullspace{};
  for (int i = 0; i < kNumJoints; ++i) {
    nullspace[i] = target_nullspace_snapshot_[i].load(std::memory_order_relaxed);
  }
  add("nullspace_target", format_joints(nullspace));
  msg.status.push_back(status);
  status_publisher_->publish(msg);
}

// --- realtime loop ------------------------------------------------------------------------

controller_interface::return_type CartesianTrajectoryReplayController::update(
    const rclcpp::Time& time, const rclcpp::Duration& period) {
  update_joint_states();
  const Vector7d q_current(joint_positions_current_.data());
  const Vector7d dq_current(joint_velocities_current_.data());

  Eigen::Quaterniond orientation;
  Eigen::Vector3d position;
  Matrix6x7d jacobian;
  Vector7d coriolis = Vector7d::Zero();
  if (model_from_dh_) {
    const Eigen::Matrix4d flange = fr3_flange_transform(q_current);
    position = flange.block<3, 1>(0, 3);
    orientation = Eigen::Quaterniond(Eigen::Matrix3d(flange.block<3, 3>(0, 0)));
    orientation.normalize();
    jacobian = fr3_zero_jacobian(q_current);
  } else {
    std::tie(orientation, position) = franka_cartesian_pose_->getCurrentOrientationAndTranslation();
    orientation.normalize();
    const std::array<double, 42> jacobian_array =
        franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);
    jacobian = Eigen::Map<const Matrix6x7d>(jacobian_array.data());
    if (coriolis_compensation_) {
      const std::array<double, 7> coriolis_array = franka_robot_model_->getCoriolisForceVector();
      coriolis = Vector7d(coriolis_array.data());
    }
  }

  // Move the controlled point from the flange onto the tool. The measured pose, the Jacobian
  // the law differentiates through, and therefore the whole impedance, all refer to the tool
  // frame from here on; the robot's own F_T_EE stays at identity.
  if (tool_active_) {
    const Eigen::Matrix3d flange_rotation = orientation.toRotationMatrix();
    const Eigen::Vector3d offset_base = flange_rotation * tool_translation_;
    position += offset_base;
    orientation = Eigen::Quaterniond(flange_rotation * tool_rotation_);
    orientation.normalize();
    jacobian = shift_jacobian(jacobian, offset_base);
  }

  const double dt = period.seconds();
  if (!std::isfinite(dt) || dt < 0.0) {
    return controller_interface::return_type::ERROR;
  }

  if (first_update_) {
    // Hold wherever the arm is until somebody commands something, as the example's
    // on_activate does with its initial pose.
    position_target_ = position;
    orientation_target_ = orientation;
    nullspace_target_ = q_current;
    position_d_ = position;
    orientation_d_ = orientation;
    nullspace_d_ = q_current;
    tau_command_previous_.setZero();
    rt_phase_ = Phase::kIdle;
    update_gains(true);
    first_update_ = false;
    for (int i = 0; i < 3; ++i) {
      target_position_snapshot_[i].store(position_target_(i), std::memory_order_relaxed);
    }
    const auto q = to_array(orientation_target_);
    for (int i = 0; i < 4; ++i) {
      target_orientation_snapshot_[i].store(q[i], std::memory_order_relaxed);
    }
    for (int i = 0; i < kNumJoints; ++i) {
      target_nullspace_snapshot_[i].store(nullspace_target_(i), std::memory_order_relaxed);
    }
    command_initialized_.store(true, std::memory_order_release);
  } else {
    update_gains(false);
  }

  const Command* command = command_buffer_.readFromRT();
  if (command != nullptr && command->id != rt_command_id_) {
    rt_command_id_ = command->id;
    switch (command->kind) {
      case CommandKind::kGoto:
        blend_position_start_ = position_target_;
        blend_position_target_ = Eigen::Vector3d(command->position.data());
        blend_orientation_start_ = orientation_target_;
        blend_orientation_target_ = to_quaternion(command->orientation);
        if (blend_orientation_start_.coeffs().dot(blend_orientation_target_.coeffs()) < 0.0) {
          blend_orientation_target_.coeffs() = -blend_orientation_target_.coeffs();
        }
        blend_nullspace_start_ = nullspace_target_;
        blend_nullspace_target_ =
            command->has_nullspace ? Vector7d(command->nullspace.data()) : nullspace_target_;
        rt_duration_ = command->duration;
        rt_elapsed_ = 0.0;
        rt_settle_elapsed_ = 0.0;
        rt_playback_rate_ = 1.0;
        rt_playback_target_ = 1.0;
        rt_rate_ramp_start_ = 1.0;
        rt_rate_ramp_elapsed_ = pause_ramp_duration_;
        rt_rate_ramp_duration_ = pause_ramp_duration_;
        rt_phase_ = Phase::kGoto;
        break;
      case CommandKind::kTrajectory:
        rt_trajectory_ = command->trajectory;
        rt_segment_hint_ = 0;
        rt_duration_ = command->duration;
        rt_elapsed_ = 0.0;
        rt_playback_rate_ = 1.0;
        rt_playback_target_ = 1.0;
        rt_rate_ramp_start_ = 1.0;
        rt_rate_ramp_elapsed_ = pause_ramp_duration_;
        rt_rate_ramp_duration_ = pause_ramp_duration_;
        pause_requested_.store(false, std::memory_order_release);
        paused_.store(false, std::memory_order_release);
        playback_rate_.store(1.0, std::memory_order_release);
        rt_phase_ = Phase::kTrajectory;
        break;
      case CommandKind::kAbort:
        if (rt_phase_ != Phase::kIdle) {
          begin_stopping();
        } else {
          completed_command_id_.store(command->id);
        }
        break;
      case CommandKind::kPolicy:
        position_target_ = Eigen::Vector3d(command->position.data());
        orientation_target_ = to_quaternion(command->orientation);
        if (command->has_nullspace) {
          nullspace_target_ = Vector7d(command->nullspace.data());
        }
        rt_elapsed_ = 0.0;
        rt_duration_ = policy_command_timeout_;
        rt_policy_command_age_ = 0.0;
        policy_watchdog_stop_.store(false, std::memory_order_release);
        rt_phase_ = Phase::kPolicy;
        break;
      case CommandKind::kNone:
        break;
    }
    if (rt_phase_ != Phase::kIdle) {
      active_command_id_.store(command->id);
      if (command->kind != CommandKind::kAbort) {
        fault_.store(static_cast<int>(Fault::kNone));
      }
    }
  }

  // The reference sampler: goto ramps and trajectories, both driven by the phase clock, and
  // stopping, which is either of them with the clock rate ramped to zero.
  bool finished = false;
  const Phase sampled_as = rt_phase_ == Phase::kStopping ? rt_stopping_from_ : rt_phase_;
  switch (rt_phase_) {
    case Phase::kIdle:
      break;
    case Phase::kPolicy:
      rt_policy_command_age_ += dt;
      rt_elapsed_ = rt_policy_command_age_;
      if (rt_policy_command_age_ > policy_command_timeout_) {
        policy_watchdog_stop_.store(true, std::memory_order_release);
        completed_command_id_.store(rt_command_id_);
        rt_phase_ = Phase::kIdle;
      }
      break;
    case Phase::kGoto:
    case Phase::kTrajectory:
    case Phase::kStopping: {
      double requested_rate = 1.0;
      if (rt_phase_ == Phase::kTrajectory && pause_requested_.load(std::memory_order_acquire)) {
        requested_rate = 0.0;
      }
      if (rt_phase_ == Phase::kStopping) {
        requested_rate = 0.0;
      }
      advance_clock(dt, requested_rate);
      if (sampled_as == Phase::kGoto) {
        const double s = std::clamp(rt_duration_ > 0.0 ? rt_elapsed_ / rt_duration_ : 1.0, 0.0, 1.0);
        const double blend = quintic_blend(s);
        position_target_ =
            blend_position_start_ + (blend_position_target_ - blend_position_start_) * blend;
        orientation_target_ = blend_orientation_start_.slerp(blend, blend_orientation_target_);
        orientation_target_.normalize();
        nullspace_target_ =
            blend_nullspace_start_ + (blend_nullspace_target_ - blend_nullspace_start_) * blend;
        if (rt_phase_ == Phase::kGoto && s >= 1.0) {
          position_target_ = blend_position_target_;
          orientation_target_ = blend_orientation_target_;
          nullspace_target_ = blend_nullspace_target_;
          // Idle only once the example's reference filter has arrived as well, so that the
          // next trajectory's start check sees a settled reference.
          rt_settle_elapsed_ += dt;
          const bool settled =
              (position_d_ - position_target_).norm() <= goto_settle_tolerance_m_ &&
              quaternion_angle(orientation_d_, orientation_target_) <= goto_settle_tolerance_rad_;
          if (settled || rt_settle_elapsed_ >= goto_settle_timeout_) {
            finished = true;
          }
        }
      } else if (sampled_as == Phase::kTrajectory && rt_trajectory_) {
        sample_trajectory(*rt_trajectory_, rt_elapsed_, rt_segment_hint_, position_target_,
                          orientation_target_, nullspace_target_);
        if (rt_phase_ == Phase::kTrajectory && rt_elapsed_ >= rt_duration_) {
          finished = true;
        }
      }
      if (rt_phase_ == Phase::kStopping && rt_rate_ramp_elapsed_ >= rt_rate_ramp_duration_) {
        finished = true;
      }
      const bool is_paused = rt_phase_ == Phase::kTrajectory &&
                             pause_requested_.load(std::memory_order_relaxed) &&
                             rt_playback_rate_ <= 1e-9;
      paused_.store(is_paused, std::memory_order_release);
      playback_rate_.store(rt_playback_rate_, std::memory_order_release);
      break;
    }
  }
  if (!nullspace_follows_trajectory_) {
    nullspace_target_ = nullspace_d_;
  }
  if (finished) {
    completed_command_id_.store(rt_command_id_);
    rt_phase_ = Phase::kIdle;
    rt_trajectory_.reset();
    pause_requested_.store(false, std::memory_order_release);
    paused_.store(false, std::memory_order_release);
    playback_rate_.store(1.0, std::memory_order_release);
  }

  // The example's law on the filtered reference, then its filters toward the new target.
  const CartesianImpedanceTerms terms = example_cartesian_impedance(
      position, orientation, jacobian, coriolis, q_current, dq_current, position_d_,
      orientation_d_, nullspace_d_, stiffness_, damping_, nullspace_stiffness_);
  const Vector7d output = saturate_torque_rate(terms.tau_command, tau_command_previous_);
  tau_command_previous_ = output;
  for (int i = 0; i < kNumJoints; ++i) {
    command_interfaces_[i].set_value(output(i));
  }

  const double position_error = terms.error.head(3).norm();
  const double orientation_error = quaternion_angle(orientation, orientation_d_);
  if ((rt_phase_ == Phase::kGoto || rt_phase_ == Phase::kTrajectory ||
       rt_phase_ == Phase::kPolicy) &&
      (position_error > max_position_error_ || orientation_error > max_orientation_error_)) {
    fault_.store(static_cast<int>(position_error > max_position_error_ ? Fault::kPosition
                                                                        : Fault::kOrientation));
    begin_stopping();
  }

  example_gain_filter(filter_alpha_, stiffness_target_, damping_target_, nullspace_stiffness_target_,
                      stiffness_, damping_, nullspace_stiffness_);
  example_reference_filter(filter_alpha_, position_target_, orientation_target_, position_d_,
                           orientation_d_);
  nullspace_d_ = filter_alpha_ * nullspace_target_ + (1.0 - filter_alpha_) * nullspace_d_;

  // --- snapshots for the executor thread ---------------------------------------------------
  for (int i = 0; i < 3; ++i) {
    target_position_snapshot_[i].store(position_target_(i), std::memory_order_relaxed);
    reference_pose_snapshot_[i].store(position_d_(i), std::memory_order_relaxed);
  }
  {
    const auto target = to_array(orientation_target_);
    const auto reference = to_array(orientation_d_);
    for (int i = 0; i < 4; ++i) {
      target_orientation_snapshot_[i].store(target[i], std::memory_order_relaxed);
      reference_pose_snapshot_[3 + i].store(reference[i], std::memory_order_relaxed);
    }
  }
  double margin = 1e9;
  for (int i = 0; i < kNumJoints; ++i) {
    target_nullspace_snapshot_[i].store(nullspace_target_(i), std::memory_order_relaxed);
    margin = std::min({margin, q_current(i) - position_limits_lower_[i],
                       position_limits_upper_[i] - q_current(i)});
  }
  gains_applied_snapshot_[0].store(stiffness_(0, 0), std::memory_order_relaxed);
  gains_applied_snapshot_[1].store(stiffness_(3, 3), std::memory_order_relaxed);
  gains_applied_snapshot_[2].store(nullspace_stiffness_, std::memory_order_relaxed);
  position_error_.store(position_error);
  orientation_error_.store(orientation_error);
  joint_limit_margin_.store(margin);
  phase_.store(static_cast<int>(rt_phase_));
  phase_elapsed_.store(rt_elapsed_);
  phase_duration_.store(rt_duration_);
  policy_command_age_.store(rt_policy_command_age_, std::memory_order_relaxed);
  processed_command_id_.store(rt_command_id_);

  if (state_publisher_ && state_publisher_->trylock()) {
    auto& msg = state_publisher_->msg_;
    msg.header.stamp = time;
    for (int i = 0; i < kNumJoints; ++i) {
      msg.reference.positions[i] = nullspace_d_(i);
      msg.reference.velocities[i] = 0.0;
      msg.feedback.positions[i] = q_current(i);
      msg.feedback.velocities[i] = dq_current(i);
      msg.feedback.effort[i] = joint_efforts_current_[i];
      msg.error.positions[i] = nullspace_d_(i) - q_current(i);
      msg.error.velocities[i] = -dq_current(i);
      msg.output.effort[i] = output(i);
    }
    msg.reference.time_from_start =
        rclcpp::Duration::from_seconds(rt_phase_ == Phase::kIdle ? 0.0 : rt_elapsed_);
    msg.output.time_from_start = rclcpp::Duration::from_seconds(static_cast<double>(rt_phase_));
    state_publisher_->unlockAndPublish();
  }
  if (cartesian_state_publisher_ && cartesian_state_publisher_->trylock()) {
    auto& msg = cartesian_state_publisher_->msg_;
    msg.header.stamp = time;
    msg.header.frame_id = base_frame_;
    msg.phase = static_cast<int>(rt_phase_);
    msg.trajectory_time = rt_phase_ == Phase::kIdle ? 0.0 : rt_elapsed_;
    fill_pose(msg.target, position_target_, orientation_target_);
    fill_pose(msg.reference, position_d_, orientation_d_);
    fill_pose(msg.measured, position, orientation);
    for (int i = 0; i < 6; ++i) {
      msg.error[i] = terms.error(i);
    }
    for (int i = 0; i < kNumJoints; ++i) {
      msg.nullspace_reference[i] = nullspace_d_(i);
      msg.tau_task[i] = terms.tau_task(i);
      msg.tau_nullspace[i] = terms.tau_nullspace(i);
      msg.tau_coriolis[i] = terms.tau_coriolis(i);
      msg.tau_command[i] = output(i);
    }
    cartesian_state_publisher_->unlockAndPublish();
  }

  return controller_interface::return_type::OK;
}

// --- lifecycle ----------------------------------------------------------------------------

CartesianTrajectoryReplayController::CallbackReturn CartesianTrajectoryReplayController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "fr3");
    auto_declare<std::string>("arm_prefix", "");
    auto_declare<std::string>("base_frame", "fr3_link0");
    auto_declare<std::string>("model_source", "franka");
    auto_declare<std::vector<double>>("tool_offset_xyz", {0.0, 0.0, 0.0});
    auto_declare<std::vector<double>>("tool_offset_rpy", {0.0, 0.0, 0.0});
    auto_declare<double>("translational_stiffness", 150.0);
    auto_declare<double>("rotational_stiffness", 10.0);
    auto_declare<double>("nullspace_stiffness", 20.0);
    auto_declare<double>("stiffness_scale", 1.0);
    auto_declare<double>("target_filter", 0.005);
    auto_declare<std::string>("nullspace_target", "trajectory");
    auto_declare<bool>("coriolis_compensation", true);
    auto_declare<double>("torque_rate_limit", 0.0);
    auto_declare<double>("goto_max_velocity", 0.10);
    auto_declare<double>("goto_max_angular_velocity", 0.50);
    auto_declare<double>("goto_max_nullspace_velocity", 0.50);
    auto_declare<double>("goto_min_duration", 3.0);
    auto_declare<double>("max_goto_step_m", 0.30);
    auto_declare<double>("max_goto_step_rad", 1.0);
    auto_declare<double>("max_trajectory_start_error_m", 0.002);
    auto_declare<double>("max_trajectory_start_error_rad", 0.01);
    auto_declare<double>("max_policy_step_m", 0.036);
    auto_declare<double>("max_policy_step_rad", 0.18);
    auto_declare<double>("policy_command_timeout", 0.25);
    auto_declare<double>("goto_settle_tolerance_m", 0.0005);
    auto_declare<double>("goto_settle_tolerance_rad", 0.002);
    auto_declare<double>("goto_settle_timeout", 2.0);
    auto_declare<std::vector<double>>("workspace_min", {-0.855, -0.855, -0.36});
    auto_declare<std::vector<double>>("workspace_max", {0.855, 0.855, 1.19});
    auto_declare<double>("trajectory_velocity_scale", 1.0);
    auto_declare<double>("max_position_error", 0.08);
    auto_declare<double>("max_orientation_error", 0.35);
    auto_declare<double>("pause_ramp_duration", 0.5);
    auto_declare<double>("abort_stop_duration", 0.5);
    auto_declare<double>("status_rate", 50.0);
    auto_declare<std::vector<double>>("position_limits_lower",
                                      std::vector<double>(kPositionLower.begin(), kPositionLower.end()));
    auto_declare<std::vector<double>>("position_limits_upper",
                                      std::vector<double>(kPositionUpper.begin(), kPositionUpper.end()));
    declare_collision_behavior_parameters(
        [this](const char* name, auto value) {
          this->template auto_declare<decltype(value)>(name, value);
        });
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Exception during on_init: %s", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

bool CartesianTrajectoryReplayController::assign_parameters() {
  const auto node = get_node();
  arm_id_ = node->get_parameter("arm_id").as_string();
  arm_prefix_ = node->get_parameter("arm_prefix").as_string();
  arm_prefix_ = arm_prefix_.empty() ? "" : arm_prefix_ + "_";
  base_frame_ = node->get_parameter("base_frame").as_string();

  std::array<double, 3> tool_xyz{};
  std::array<double, 3> tool_rpy{};
  {
    const auto read3 = [&node](const char* name, std::array<double, 3>& destination) {
      const auto values = node->get_parameter(name).as_double_array();
      if (values.size() != 3) {
        RCLCPP_FATAL(node->get_logger(), "%s must have 3 entries, got %zu", name, values.size());
        return false;
      }
      if (!std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); })) {
        RCLCPP_FATAL(node->get_logger(), "%s must be finite", name);
        return false;
      }
      std::copy(values.begin(), values.end(), destination.begin());
      return true;
    };
    if (!read3("tool_offset_xyz", tool_xyz) || !read3("tool_offset_rpy", tool_rpy)) {
      return false;
    }
  }
  tool_translation_ = Eigen::Vector3d(tool_xyz.data());
  tool_rotation_ = rpy_to_rotation(Eigen::Vector3d(tool_rpy.data()));
  tool_active_ = !tool_translation_.isZero(0.0) || !tool_rotation_.isIdentity(0.0);

  const auto model_source = node->get_parameter("model_source").as_string();
  if (model_source == "franka") {
    model_from_dh_ = false;
  } else if (model_source == "dh") {
    model_from_dh_ = true;
  } else {
    RCLCPP_FATAL(node->get_logger(), "model_source must be 'franka' or 'dh', got '%s'",
                 model_source.c_str());
    return false;
  }

  const auto nullspace_mode = node->get_parameter("nullspace_target").as_string();
  if (nullspace_mode == "trajectory") {
    nullspace_follows_trajectory_ = true;
  } else if (nullspace_mode == "fixed") {
    nullspace_follows_trajectory_ = false;
  } else {
    RCLCPP_FATAL(node->get_logger(), "nullspace_target must be 'trajectory' or 'fixed', got '%s'",
                 nullspace_mode.c_str());
    return false;
  }
  coriolis_compensation_ = node->get_parameter("coriolis_compensation").as_bool();
  torque_rate_limit_ = node->get_parameter("torque_rate_limit").as_double();
  goto_max_velocity_ = node->get_parameter("goto_max_velocity").as_double();
  goto_max_angular_velocity_ = node->get_parameter("goto_max_angular_velocity").as_double();
  goto_max_nullspace_velocity_ = node->get_parameter("goto_max_nullspace_velocity").as_double();
  goto_min_duration_ = node->get_parameter("goto_min_duration").as_double();
  max_goto_step_m_ = node->get_parameter("max_goto_step_m").as_double();
  max_goto_step_rad_ = node->get_parameter("max_goto_step_rad").as_double();
  max_trajectory_start_error_m_ = node->get_parameter("max_trajectory_start_error_m").as_double();
  max_trajectory_start_error_rad_ = node->get_parameter("max_trajectory_start_error_rad").as_double();
  max_policy_step_m_ = node->get_parameter("max_policy_step_m").as_double();
  max_policy_step_rad_ = node->get_parameter("max_policy_step_rad").as_double();
  policy_command_timeout_ = node->get_parameter("policy_command_timeout").as_double();
  goto_settle_tolerance_m_ = node->get_parameter("goto_settle_tolerance_m").as_double();
  goto_settle_tolerance_rad_ = node->get_parameter("goto_settle_tolerance_rad").as_double();
  goto_settle_timeout_ = node->get_parameter("goto_settle_timeout").as_double();
  trajectory_velocity_scale_ = node->get_parameter("trajectory_velocity_scale").as_double();
  max_position_error_ = node->get_parameter("max_position_error").as_double();
  max_orientation_error_ = node->get_parameter("max_orientation_error").as_double();
  pause_ramp_duration_ = node->get_parameter("pause_ramp_duration").as_double();
  abort_stop_duration_ = node->get_parameter("abort_stop_duration").as_double();

  const auto positive = [](double value) { return std::isfinite(value) && value > 0.0; };
  if (!positive(goto_max_velocity_) || !positive(goto_max_angular_velocity_) ||
      !positive(goto_max_nullspace_velocity_) || !positive(goto_min_duration_) ||
      !positive(pause_ramp_duration_) || !positive(abort_stop_duration_) ||
      !positive(goto_settle_timeout_) || !positive(trajectory_velocity_scale_) ||
      !positive(max_position_error_) || !positive(max_orientation_error_) ||
      !positive(max_policy_step_m_) || !positive(max_policy_step_rad_) ||
      !positive(policy_command_timeout_)) {
    RCLCPP_FATAL(node->get_logger(),
                 "goto_*, pause_ramp_duration, abort_stop_duration, trajectory_velocity_scale and "
                 "max_*_error must all be finite and > 0");
    return false;
  }

  const auto fill = [&node](const char* name, auto& destination) {
    const auto values = node->get_parameter(name).as_double_array();
    if (values.size() != destination.size()) {
      RCLCPP_FATAL(node->get_logger(), "%s must have %zu entries, got %zu", name,
                   destination.size(), values.size());
      return false;
    }
    std::copy(values.begin(), values.end(), destination.begin());
    return true;
  };
  if (!fill("workspace_min", workspace_min_) || !fill("workspace_max", workspace_max_) ||
      !fill("position_limits_lower", position_limits_lower_) ||
      !fill("position_limits_upper", position_limits_upper_)) {
    return false;
  }
  for (int i = 0; i < 3; ++i) {
    if (!(workspace_min_[i] < workspace_max_[i])) {
      RCLCPP_FATAL(node->get_logger(), "workspace_min must be below workspace_max on every axis");
      return false;
    }
  }

  GainSettings settings;
  const double translational = node->get_parameter("translational_stiffness").as_double();
  const double rotational = node->get_parameter("rotational_stiffness").as_double();
  settings.stiffness = {translational, translational, translational, rotational, rotational, rotational};
  settings.stiffness_scale = node->get_parameter("stiffness_scale").as_double();
  settings.nullspace_stiffness = node->get_parameter("nullspace_stiffness").as_double();
  settings.target_filter = node->get_parameter("target_filter").as_double();
  return publish_gain_settings(settings, "configuration");
}

bool CartesianTrajectoryReplayController::publish_gain_settings(GainSettings settings,
                                                                const char* origin) {
  const auto finite_nonnegative = [](double value) { return std::isfinite(value) && value >= 0.0; };
  if (!std::all_of(settings.stiffness.begin(), settings.stiffness.end(), finite_nonnegative) ||
      !finite_nonnegative(settings.stiffness_scale) ||
      !finite_nonnegative(settings.nullspace_stiffness)) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "%s: stiffness values, stiffness_scale and nullspace_stiffness must be finite and "
                 "non-negative",
                 origin);
    return false;
  }
  if (!std::isfinite(settings.target_filter) || !(settings.target_filter > 0.0) ||
      settings.target_filter > 1.0) {
    RCLCPP_ERROR(get_node()->get_logger(), "%s: target_filter must be in (0, 1]", origin);
    return false;
  }
  settings.revision = ++next_gain_revision_;
  gain_settings_buffer_.writeFromNonRT(settings);
  stiffness_scale_target_.store(settings.stiffness_scale, std::memory_order_release);
  RCLCPP_INFO(get_node()->get_logger(),
              "Cartesian impedance gains (%s): translational %.1f N/m, rotational %.1f Nm/rad, "
              "scale %.3f, nullspace %.1f, target filter %.4f.",
              origin, settings.stiffness[0], settings.stiffness[3], settings.stiffness_scale,
              settings.nullspace_stiffness, settings.target_filter);
  return true;
}

rcl_interfaces::msg::SetParametersResult
CartesianTrajectoryReplayController::gain_parameters_callback(
    const std::vector<rclcpp::Parameter>& parameters) {
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  GainSettings settings = *gain_settings_buffer_.readFromNonRT();
  bool changed = false;
  for (const auto& parameter : parameters) {
    const auto& name = parameter.get_name();
    if (name != "translational_stiffness" && name != "rotational_stiffness" &&
        name != "nullspace_stiffness" && name != "stiffness_scale" && name != "target_filter") {
      continue;
    }
    if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE) {
      result.successful = false;
      result.reason = name + " must be a double";
      return result;
    }
    const double value = parameter.as_double();
    if (name == "translational_stiffness") {
      settings.stiffness[0] = settings.stiffness[1] = settings.stiffness[2] = value;
    } else if (name == "rotational_stiffness") {
      settings.stiffness[3] = settings.stiffness[4] = settings.stiffness[5] = value;
    } else if (name == "nullspace_stiffness") {
      settings.nullspace_stiffness = value;
    } else if (name == "stiffness_scale") {
      settings.stiffness_scale = value;
    } else {
      settings.target_filter = value;
    }
    changed = true;
  }
  if (changed && !publish_gain_settings(settings, "parameter update")) {
    result.successful = false;
    result.reason = "values must be finite and non-negative; target_filter in (0, 1]";
  }
  return result;
}

void CartesianTrajectoryReplayController::set_cartesian_stiffness_callback(
    const std::shared_ptr<franka_msgs::srv::SetCartesianStiffness::Request> request,
    std::shared_ptr<franka_msgs::srv::SetCartesianStiffness::Response> response) {
  GainSettings settings = *gain_settings_buffer_.readFromNonRT();
  for (size_t i = 0; i < 6; ++i) {
    settings.stiffness[i] = request->cartesian_stiffness[i];
  }
  if (!publish_gain_settings(settings, "set_cartesian_stiffness")) {
    response->success = false;
    response->error = "Cartesian stiffness must contain 6 finite, non-negative values.";
    return;
  }
  response->success = true;
  response->error = "";
}

CartesianTrajectoryReplayController::CallbackReturn
CartesianTrajectoryReplayController::on_configure(const rclcpp_lifecycle::State& /*previous_state*/) {
  if (!assign_parameters()) {
    return CallbackReturn::FAILURE;
  }
  if (!parameter_callback_handle_) {
    parameter_callback_handle_ = get_node()->add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter>& parameters) {
          return gain_parameters_callback(parameters);
        });
  }

  franka_cartesian_pose_.reset();
  franka_robot_model_.reset();
  if (model_from_dh_) {
    RCLCPP_WARN(get_node()->get_logger(),
                "model_source is 'dh': pose and Jacobian come from the built-in FR3 DH model with "
                "an identity F_T_EE and no coriolis term. This is for simulators without "
                "franka_hardware's robot model; use model_source 'franka' on the real arm.");
  } else {
    franka_cartesian_pose_ =
        std::make_unique<franka_semantic_components::FrankaCartesianPoseInterface>(
            arm_prefix_, /*command_elbow_active=*/false);
    franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
        arm_prefix_ + arm_id_ + "/robot_model", arm_prefix_ + arm_id_ + "/robot_state");
  }

  collision_client_ = get_node()->create_client<franka_msgs::srv::SetFullCollisionBehavior>(
      "service_server/set_full_collision_behavior");
  if (get_node()->get_parameter("set_collision_behavior").as_bool() &&
      !apply_collision_behavior(get_node(), collision_client_)) {
    return CallbackReturn::FAILURE;
  }

  goto_subscriber_ = get_node()->create_subscription<franka_trajectory_replay_msgs::msg::CartesianGoto>(
      "~/goto", rclcpp::QoS(1),
      [this](const franka_trajectory_replay_msgs::msg::CartesianGoto::SharedPtr msg) {
        goto_callback(msg);
      });
  trajectory_subscriber_ =
      get_node()->create_subscription<franka_trajectory_replay_msgs::msg::CartesianTrajectory>(
          "~/trajectory", rclcpp::QoS(1).reliable(),
          [this](const franka_trajectory_replay_msgs::msg::CartesianTrajectory::SharedPtr msg) {
            trajectory_callback(msg);
          });
  policy_command_subscriber_ =
      get_node()->create_subscription<franka_trajectory_replay_msgs::msg::CartesianGoto>(
          "~/policy_command", rclcpp::QoS(1).reliable(),
          [this](const franka_trajectory_replay_msgs::msg::CartesianGoto::SharedPtr msg) {
            policy_command_callback(msg);
          });
  pause_subscriber_ = get_node()->create_subscription<std_msgs::msg::Empty>(
      "~/pause", rclcpp::QoS(1),
      [this](const std_msgs::msg::Empty::SharedPtr msg) { pause_callback(msg); });
  resume_subscriber_ = get_node()->create_subscription<std_msgs::msg::Empty>(
      "~/resume", rclcpp::QoS(1),
      [this](const std_msgs::msg::Empty::SharedPtr msg) { resume_callback(msg); });
  abort_subscriber_ = get_node()->create_subscription<std_msgs::msg::Empty>(
      "~/abort", rclcpp::QoS(1),
      [this](const std_msgs::msg::Empty::SharedPtr msg) { abort_callback(msg); });
  stiffness_service_ = get_node()->create_service<franka_msgs::srv::SetCartesianStiffness>(
      "~/set_cartesian_stiffness",
      [this](const std::shared_ptr<franka_msgs::srv::SetCartesianStiffness::Request> request,
             std::shared_ptr<franka_msgs::srv::SetCartesianStiffness::Response> response) {
        set_cartesian_stiffness_callback(request, response);
      });

  status_publisher_ = get_node()->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      "~/status", rclcpp::QoS(10));
  const double status_rate = get_node()->get_parameter("status_rate").as_double();
  const auto status_period = std::chrono::duration<double>(1.0 / std::max(status_rate, 1.0));
  status_timer_ = get_node()->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(status_period),
      [this]() { publish_status(); });

  auto state_publisher = get_node()->create_publisher<control_msgs::msg::JointTrajectoryControllerState>(
      "~/controller_state", rclcpp::SystemDefaultsQoS());
  state_publisher_ = std::make_unique<
      realtime_tools::RealtimePublisher<control_msgs::msg::JointTrajectoryControllerState>>(
      state_publisher);
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
    msg.output.effort.assign(kNumJoints, 0.0);
  }
  auto cartesian_publisher =
      get_node()->create_publisher<franka_trajectory_replay_msgs::msg::CartesianReplayState>(
          "~/cartesian_state", rclcpp::SystemDefaultsQoS());
  cartesian_state_publisher_ = std::make_unique<
      realtime_tools::RealtimePublisher<franka_trajectory_replay_msgs::msg::CartesianReplayState>>(
      cartesian_publisher);

  RCLCPP_INFO(get_node()->get_logger(),
              "Configured: Cartesian impedance (example law, %s model) about %s%s, nullspace "
              "target %s, target filter %.4f, goto <= %.2f m/s / %.2f rad/s, max goto step "
              "%.2f m / %.2f rad.",
              model_from_dh_ ? "built-in DH" : "franka_hardware",
              tool_active_ ? format_pose(tool_translation_,
                                         Eigen::Quaterniond(tool_rotation_)).c_str()
                           : "the flange",
              (coriolis_compensation_ && !model_from_dh_) ? " with coriolis compensation" : "",
              nullspace_follows_trajectory_ ? "follows the trajectory" : "fixed at activation",
              get_node()->get_parameter("target_filter").as_double(), goto_max_velocity_,
              goto_max_angular_velocity_, max_goto_step_m_, max_goto_step_rad_);
  return CallbackReturn::SUCCESS;
}

CartesianTrajectoryReplayController::CallbackReturn
CartesianTrajectoryReplayController::on_activate(const rclcpp_lifecycle::State& /*previous_state*/) {
  const size_t expected = 3 * kNumJoints + (model_from_dh_ ? 0 : kPoseInterfaces);
  if (state_interfaces_.size() < expected) {
    RCLCPP_FATAL(get_node()->get_logger(), "Got %zu state interfaces, expected at least %zu.",
                 state_interfaces_.size(), expected);
    return CallbackReturn::ERROR;
  }
  first_update_ = true;
  command_initialized_.store(false, std::memory_order_release);
  phase_.store(static_cast<int>(Phase::kIdle));
  rt_phase_ = Phase::kIdle;
  rt_stopping_from_ = Phase::kIdle;
  rt_command_id_ = 0;
  next_command_id_ = 0;
  active_command_id_.store(0);
  processed_command_id_.store(0);
  completed_command_id_.store(0);
  pause_requested_.store(false, std::memory_order_release);
  paused_.store(false, std::memory_order_release);
  playback_rate_.store(1.0, std::memory_order_release);
  fault_.store(static_cast<int>(Fault::kNone));
  policy_command_age_.store(0.0);
  policy_watchdog_stop_.store(false);
  rt_policy_command_age_ = 0.0;
  rt_trajectory_.reset();
  command_buffer_.writeFromNonRT(Command{});
  last_rejection_.clear();
  rejections_ = 0;
  for (auto& value : reference_pose_snapshot_) {
    value.store(0.0, std::memory_order_relaxed);
  }
  for (auto& value : gains_applied_snapshot_) {
    value.store(0.0, std::memory_order_relaxed);
  }
  if (franka_robot_model_) {
    franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  }
  if (franka_cartesian_pose_) {
    franka_cartesian_pose_->assign_loaned_state_interfaces(state_interfaces_);
  }
  is_active_.store(true, std::memory_order_release);
  return CallbackReturn::SUCCESS;
}

CartesianTrajectoryReplayController::CallbackReturn
CartesianTrajectoryReplayController::on_deactivate(const rclcpp_lifecycle::State& /*previous_state*/) {
  is_active_.store(false, std::memory_order_release);
  command_initialized_.store(false, std::memory_order_release);
  if (franka_robot_model_) {
    franka_robot_model_->release_interfaces();
  }
  if (franka_cartesian_pose_) {
    franka_cartesian_pose_->release_interfaces();
  }
  return CallbackReturn::SUCCESS;
}

}  // namespace franka_trajectory_replay

// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_trajectory_replay::CartesianTrajectoryReplayController,
                       controller_interface::ControllerInterface)
