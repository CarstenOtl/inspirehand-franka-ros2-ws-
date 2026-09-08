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

#include <franka_repeatability/cartesian_target_impedance_ik_controller.hpp>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>

#include <pluginlib/class_list_macros.hpp>

using namespace std::chrono_literals;

namespace franka_repeatability {

namespace {
constexpr std::chrono::duration<double> kServiceWaitStep{1s};
}  // namespace

std::vector<std::string> CartesianTargetImpedanceIKController::joint_names() const {
  std::vector<std::string> names;
  names.reserve(kNumJoints);
  for (int i = 1; i <= kNumJoints; ++i) {
    names.push_back(arm_prefix_ + robot_type_ + "_joint" + std::to_string(i));
  }
  return names;
}

controller_interface::InterfaceConfiguration
CartesianTargetImpedanceIKController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto& joint : joint_names()) {
    config.names.push_back(joint + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
CartesianTargetImpedanceIKController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // Order matters: kPositionOffset / kVelocityOffset / kEffortOffset index into this list.
  for (const auto& joint : joint_names()) {
    config.names.push_back(joint + "/position");
  }
  for (const auto& joint : joint_names()) {
    config.names.push_back(joint + "/velocity");
  }
  for (const auto& joint : joint_names()) {
    config.names.push_back(joint + "/effort");
  }
  for (const auto& name : franka_robot_model_->get_state_interface_names()) {
    config.names.push_back(name);
  }
  config.names.push_back(arm_prefix_ + robot_type_ + "/robot_time");

  return config;
}

void CartesianTargetImpedanceIKController::update_joint_states() {
  for (int i = 0; i < kNumJoints; ++i) {
    joint_positions_current_[i] = state_interfaces_.at(kPositionOffset + i).get_value();
    joint_velocities_current_[i] = state_interfaces_.at(kVelocityOffset + i).get_value();
    joint_efforts_current_[i] = state_interfaces_.at(kEffortOffset + i).get_value();
    measured_positions_snapshot_[i].store(joint_positions_current_[i], std::memory_order_relaxed);
  }
}

double CartesianTargetImpedanceIKController::quintic_blend(double s) {
  // 10s^3 - 15s^4 + 6s^5: zero velocity and acceleration at both ends.
  return s * s * s * (10.0 + s * (-15.0 + 6.0 * s));
}

CartesianTargetImpedanceIKController::Vector7d
CartesianTargetImpedanceIKController::compute_torque_command(
    const Vector7d& joint_positions_desired,
    const Vector7d& joint_positions_current,
    const Vector7d& joint_velocities_current) {
  std::array<double, 7> coriolis_array = franka_robot_model_->getCoriolisForceVector();
  Vector7d coriolis(coriolis_array.data());

  const double kAlpha = 0.99;
  dq_filtered_ = (1 - kAlpha) * dq_filtered_ + kAlpha * joint_velocities_current;

  Vector7d q_error = joint_positions_desired - joint_positions_current;
  return k_gains_.cwiseProduct(q_error) - d_gains_.cwiseProduct(dq_filtered_) + coriolis;
}

CartesianTargetImpedanceIKController::Vector7d
CartesianTargetImpedanceIKController::saturate_torque_rate(const Vector7d& tau_desired,
                                                           const Vector7d& tau_previous) const {
  // Zero or negative disables the limiter, which makes the applied torque exactly what the
  // upstream joint_impedance_with_ik example would apply. The example has no rate limit.
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

std::shared_ptr<moveit_msgs::srv::GetPositionIK::Request>
CartesianTargetImpedanceIKController::create_ik_service_request(
    const geometry_msgs::msg::PoseStamped& pose) const {
  auto request = std::make_shared<moveit_msgs::srv::GetPositionIK::Request>();

  request->ik_request.group_name = ik_group_name_;
  request->ik_request.avoid_collisions = avoid_collisions_;
  request->ik_request.timeout = rclcpp::Duration::from_seconds(ik_timeout_);
  request->ik_request.pose_stamped = pose;
  if (request->ik_request.pose_stamped.header.frame_id.empty()) {
    request->ik_request.pose_stamped.header.frame_id = arm_prefix_ + robot_type_ + "_link0";
  }
  if (!ik_link_name_.empty()) {
    request->ik_request.ik_link_name = ik_link_name_;
  }

  // Seed from the current measurement so the solver returns the branch the arm is already on.
  request->ik_request.robot_state.joint_state.name = joint_names();
  request->ik_request.robot_state.joint_state.position.resize(kNumJoints);
  for (int i = 0; i < kNumJoints; ++i) {
    request->ik_request.robot_state.joint_state.position[i] =
        measured_positions_snapshot_[i].load(std::memory_order_relaxed);
  }

  return request;
}

bool CartesianTargetImpedanceIKController::apply_collision_behavior() {
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
      !fill("lower_force_thresholds_acceleration",
            request->lower_force_thresholds_acceleration) ||
      !fill("upper_force_thresholds_acceleration",
            request->upper_force_thresholds_acceleration)) {
    return false;
  }

  if (!collision_client_->wait_for_service(5s)) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "service_server/set_full_collision_behavior is not available. The robot would "
                 "keep whatever collision thresholds Desk last set, and the impedance law can "
                 "trip a reflex against low ones.");
    return false;
  }

  // Bounded wait rather than the upstream example's unbounded future.get(), so a missing
  // response surfaces as an error instead of a hung configure.
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

bool CartesianTargetImpedanceIKController::accept_goal(const std::vector<double>& positions,
                                                       const std::string& source) {
  if (positions.size() != static_cast<size_t>(kNumJoints)) {
    RCLCPP_ERROR(get_node()->get_logger(), "%s target has %zu joints, expected %d - ignored.",
                 source.c_str(), positions.size(), kNumJoints);
    return false;
  }
  if (!command_initialized_.load(std::memory_order_acquire)) {
    RCLCPP_WARN(get_node()->get_logger(),
                "%s target arrived before the first update cycle - ignored.", source.c_str());
    return false;
  }

  double largest_step = 0.0;
  for (int i = 0; i < kNumJoints; ++i) {
    if (!std::isfinite(positions[i])) {
      RCLCPP_ERROR(get_node()->get_logger(), "%s target joint %d is not finite - ignored.",
                   source.c_str(), i + 1);
      return false;
    }
    const double measured = measured_positions_snapshot_[i].load(std::memory_order_relaxed);
    largest_step = std::max(largest_step, std::abs(positions[i] - measured));
  }
  if (largest_step > max_joint_step_) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "%s target is %.3f rad away from the current configuration, which exceeds "
                 "max_joint_step (%.3f rad) - ignored. Move closer or raise the limit.",
                 source.c_str(), largest_step, max_joint_step_);
    return false;
  }

  Goal goal;
  std::copy(positions.begin(), positions.end(), goal.positions.begin());
  goal.id = ++next_goal_id_;
  goal_buffer_.writeFromNonRT(goal);

  sensor_msgs::msg::JointState echo;
  echo.header.stamp = get_node()->now();
  echo.name = joint_names();
  echo.position = positions;
  active_goal_publisher_->publish(echo);

  RCLCPP_INFO(get_node()->get_logger(),
              "Accepted %s target (goal %lu), largest joint step %.4f rad, ramp %.2f s.",
              source.c_str(), static_cast<unsigned long>(goal.id), largest_step, motion_duration_);
  return true;
}

void CartesianTargetImpedanceIKController::target_joint_positions_callback(
    const sensor_msgs::msg::JointState::SharedPtr msg) {
  accept_goal(msg->position, "joint");
}

void CartesianTargetImpedanceIKController::target_pose_callback(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  if (!compute_ik_client_->service_is_ready()) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Pose target received but the compute_ik service is not available. Start this "
                 "controller together with move_group, or command joint targets instead.");
    return;
  }

  auto request = create_ik_service_request(*msg);
  compute_ik_client_->async_send_request(
      request, [this](rclcpp::Client<moveit_msgs::srv::GetPositionIK>::SharedFuture future) {
        const auto& response = future.get();
        if (response->error_code.val != response->error_code.SUCCESS) {
          RCLCPP_ERROR(get_node()->get_logger(), "Inverse kinematics failed with error code %d.",
                       response->error_code.val);
          return;
        }
        // The solution carries the whole robot state; pick out our joints by name.
        const auto& solution = response->solution.joint_state;
        std::vector<double> positions;
        positions.reserve(kNumJoints);
        for (const auto& joint : joint_names()) {
          const auto it = std::find(solution.name.begin(), solution.name.end(), joint);
          if (it == solution.name.end()) {
            RCLCPP_ERROR(get_node()->get_logger(), "IK solution does not contain joint %s.",
                         joint.c_str());
            return;
          }
          positions.push_back(solution.position.at(std::distance(solution.name.begin(), it)));
        }
        accept_goal(positions, "pose");
      });
}

controller_interface::return_type CartesianTargetImpedanceIKController::update(
    const rclcpp::Time& time,
    const rclcpp::Duration& period) {
  update_joint_states();

  Vector7d joint_positions_current(joint_positions_current_.data());
  Vector7d joint_velocities_current(joint_velocities_current_.data());

  robot_time_ = state_interfaces_.at(robot_time_index_).get_value();

  if (first_update_) {
    // Hold wherever the arm is until somebody commands a target.
    position_command_ = joint_positions_current;
    position_command_previous_ = joint_positions_current;
    blend_start_ = joint_positions_current;
    blend_target_ = joint_positions_current;
    tau_command_previous_.setZero();
    dq_filtered_.setZero();
    last_robot_time_ = robot_time_;
    first_update_ = false;
    command_initialized_.store(true, std::memory_order_release);
  }

  // Prefer the arm's own clock; fall back to the controller period if it did not advance.
  double dt = robot_time_ - last_robot_time_;
  if (!(dt > 0.0) || dt > 1.0) {
    dt = period.seconds();
  }
  last_robot_time_ = robot_time_;

  const Goal* goal = goal_buffer_.readFromRT();
  if (goal != nullptr && goal->id != active_goal_id_) {
    active_goal_id_ = goal->id;
    blend_start_ = position_command_;  // start from the command, not the measurement: no jump
    blend_target_ = Vector7d(goal->positions.data());
    blend_elapsed_ = 0.0;
    motion_active_.store(true, std::memory_order_relaxed);
  }

  if (motion_active_.load(std::memory_order_relaxed)) {
    blend_elapsed_ += dt;
    const double s = std::clamp(blend_elapsed_ / motion_duration_, 0.0, 1.0);
    position_command_ = blend_start_ + (blend_target_ - blend_start_) * quintic_blend(s);
    if (s >= 1.0) {
      position_command_ = blend_target_;
      motion_active_.store(false, std::memory_order_relaxed);
    }
  }

  Vector7d velocity_command = Vector7d::Zero();
  if (dt > 0.0) {
    velocity_command = (position_command_ - position_command_previous_) / dt;
  }
  position_command_previous_ = position_command_;

  Vector7d tau_desired =
      compute_torque_command(position_command_, joint_positions_current, joint_velocities_current);
  const Vector7d tau_command = saturate_torque_rate(tau_desired, tau_command_previous_);
  tau_command_previous_ = tau_command;

  for (int i = 0; i < kNumJoints; ++i) {
    command_interfaces_[i].set_value(tau_command(i));
  }

  if (state_publisher_ && state_publisher_->trylock()) {
    auto& msg = state_publisher_->msg_;
    msg.header.stamp = time;
    for (int i = 0; i < kNumJoints; ++i) {
      msg.reference.positions[i] = position_command_(i);
      msg.reference.velocities[i] = velocity_command(i);
      msg.feedback.positions[i] = joint_positions_current(i);
      msg.feedback.velocities[i] = joint_velocities_current(i);
      msg.feedback.effort[i] = joint_efforts_current_[i];
      msg.error.positions[i] = position_command_(i) - joint_positions_current(i);
      msg.error.velocities[i] = velocity_command(i) - joint_velocities_current(i);
      msg.output.effort[i] = tau_command(i);
    }
    // Carries the ramp clock so a recording can be split into motion and dwell without guessing.
    msg.reference.time_from_start = rclcpp::Duration::from_seconds(blend_elapsed_);
    state_publisher_->unlockAndPublish();
  }

  return controller_interface::return_type::OK;
}

CallbackReturn CartesianTargetImpedanceIKController::on_init() {
  try {
    auto_declare<std::string>("robot_type", "fr3");
    auto_declare<std::string>("arm_prefix", "");
    auto_declare<std::string>("ik_group_name", "");
    auto_declare<std::string>("ik_link_name", "");
    auto_declare<bool>("avoid_collisions", true);
    auto_declare<double>("motion_duration", 5.0);
    auto_declare<double>("max_joint_step", 1.5);
    auto_declare<double>("torque_rate_limit", 1.0);
    auto_declare<double>("ik_timeout", 0.05);
    auto_declare<double>("ik_service_timeout", 20.0);
    auto_declare<std::vector<double>>("k_gains", {});
    auto_declare<std::vector<double>>("d_gains", {});

    // Collision thresholds, defaulted to the values
    // franka_example_controllers/default_robot_behavior_utils.hpp applies before the upstream
    // joint_impedance_with_ik example runs.
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

bool CartesianTargetImpedanceIKController::assign_parameters() {
  robot_type_ = get_node()->get_parameter("robot_type").as_string();
  arm_prefix_ = get_node()->get_parameter("arm_prefix").as_string();
  arm_prefix_ = arm_prefix_.empty() ? "" : arm_prefix_ + "_";

  ik_group_name_ = get_node()->get_parameter("ik_group_name").as_string();
  if (ik_group_name_.empty()) {
    ik_group_name_ = arm_prefix_ + robot_type_ + "_arm";
  }
  ik_link_name_ = get_node()->get_parameter("ik_link_name").as_string();
  avoid_collisions_ = get_node()->get_parameter("avoid_collisions").as_bool();
  motion_duration_ = get_node()->get_parameter("motion_duration").as_double();
  max_joint_step_ = get_node()->get_parameter("max_joint_step").as_double();
  torque_rate_limit_ = get_node()->get_parameter("torque_rate_limit").as_double();
  ik_timeout_ = get_node()->get_parameter("ik_timeout").as_double();

  if (!(motion_duration_ > 0.0)) {
    RCLCPP_FATAL(get_node()->get_logger(), "motion_duration must be > 0, got %f.",
                 motion_duration_);
    return false;
  }

  const auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  const auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  if (k_gains.size() != static_cast<size_t>(kNumJoints) ||
      d_gains.size() != static_cast<size_t>(kNumJoints)) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "k_gains and d_gains must both have %d entries (got %zu and %zu).", kNumJoints,
                 k_gains.size(), d_gains.size());
    return false;
  }
  for (int i = 0; i < kNumJoints; ++i) {
    k_gains_(i) = k_gains.at(i);
    d_gains_(i) = d_gains.at(i);
  }
  return true;
}

CallbackReturn CartesianTargetImpedanceIKController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  if (!assign_parameters()) {
    return CallbackReturn::FAILURE;
  }

  franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
      arm_prefix_ + robot_type_ + "/" + k_robot_model_interface_name,
      arm_prefix_ + robot_type_ + "/" + k_robot_state_interface_name);

  compute_ik_client_ = get_node()->create_client<moveit_msgs::srv::GetPositionIK>("compute_ik");
  collision_client_ = get_node()->create_client<franka_msgs::srv::SetFullCollisionBehavior>(
      "service_server/set_full_collision_behavior");

  set_collision_behavior_ = get_node()->get_parameter("set_collision_behavior").as_bool();
  if (set_collision_behavior_ && !apply_collision_behavior()) {
    return CallbackReturn::FAILURE;
  }

  // Pose targets need move_group. Joint targets do not, so a missing service is a warning.
  const double ik_service_timeout = get_node()->get_parameter("ik_service_timeout").as_double();
  std::chrono::duration<double> waited{0s};
  while (!compute_ik_client_->wait_for_service(kServiceWaitStep)) {
    if (!rclcpp::ok()) {
      return CallbackReturn::ERROR;
    }
    waited += kServiceWaitStep;
    if (waited.count() >= ik_service_timeout) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "compute_ik unavailable after %.0f s. Pose targets will be rejected; joint "
                  "targets on ~/target_joint_positions still work.",
                  waited.count());
      break;
    }
    RCLCPP_INFO(get_node()->get_logger(), "Waiting for compute_ik (%.0f s)...", waited.count());
  }

  target_pose_subscriber_ = get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
      "~/target_pose", rclcpp::QoS(1),
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { target_pose_callback(msg); });
  target_joint_subscriber_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
      "~/target_joint_positions", rclcpp::QoS(1),
      [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
        target_joint_positions_callback(msg);
      });

  // Depth 1 and transient local: a late subscriber gets the current state, not a backlog of
  // goals from an earlier session that it would mistake for a fresh acknowledgement.
  active_goal_publisher_ = get_node()->create_publisher<sensor_msgs::msg::JointState>(
      "~/active_goal", rclcpp::QoS(1).transient_local());
  motion_active_publisher_ = get_node()->create_publisher<std_msgs::msg::Bool>(
      "~/motion_active", rclcpp::QoS(1).transient_local());
  motion_active_timer_ = get_node()->create_wall_timer(50ms, [this]() {
    if (!is_active_.load(std::memory_order_acquire)) {
      return;
    }
    std_msgs::msg::Bool msg;
    msg.data = motion_active_.load(std::memory_order_relaxed);
    motion_active_publisher_->publish(msg);
  });

  auto publisher = get_node()->create_publisher<control_msgs::msg::JointTrajectoryControllerState>(
      "~/controller_state", rclcpp::SystemDefaultsQoS());
  state_publisher_ = std::make_unique<
      realtime_tools::RealtimePublisher<control_msgs::msg::JointTrajectoryControllerState>>(
      publisher);

  // Size every array once, so update() never allocates.
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

  RCLCPP_INFO(get_node()->get_logger(),
              "Configured for group '%s', IK link '%s', ramp %.2f s, max joint step %.2f rad.",
              ik_group_name_.c_str(),
              ik_link_name_.empty() ? "<group tip>" : ik_link_name_.c_str(), motion_duration_,
              max_joint_step_);

  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianTargetImpedanceIKController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  const size_t expected_minimum = 3 * kNumJoints + 1;
  if (state_interfaces_.size() < expected_minimum) {
    RCLCPP_FATAL(get_node()->get_logger(), "Got %zu state interfaces, expected at least %zu.",
                 state_interfaces_.size(), expected_minimum);
    return CallbackReturn::ERROR;
  }
  robot_time_index_ = state_interfaces_.size() - 1;  // requested last

  first_update_ = true;
  command_initialized_.store(false, std::memory_order_release);
  motion_active_.store(false, std::memory_order_relaxed);
  active_goal_id_ = 0;
  next_goal_id_ = 0;
  blend_elapsed_ = 0.0;
  goal_buffer_.writeFromNonRT(Goal{});

  for (auto& value : measured_positions_snapshot_) {
    value.store(0.0, std::memory_order_relaxed);
  }

  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  is_active_.store(true, std::memory_order_release);
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianTargetImpedanceIKController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  is_active_.store(false, std::memory_order_release);
  command_initialized_.store(false, std::memory_order_release);
  franka_robot_model_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

}  // namespace franka_repeatability

// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_repeatability::CartesianTargetImpedanceIKController,
                       controller_interface::ControllerInterface)
