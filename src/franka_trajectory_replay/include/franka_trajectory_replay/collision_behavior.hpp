// Copyright (c) 2026 Agile Robots SE
// Licensed under the Apache License, Version 2.0.
// See http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <chrono>
#include <memory>
#include <vector>

#include <franka_msgs/srv/set_full_collision_behavior.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

namespace franka_trajectory_replay {

/// Declares the collision-threshold parameters with the values
/// franka_example_controllers/default_robot_behavior_utils.hpp installs before the upstream
/// examples run. ``declare`` is the controller's auto_declare.
template <typename Declare>
void declare_collision_behavior_parameters(Declare&& declare) {
  declare("set_collision_behavior", true);
  declare("lower_torque_thresholds_nominal",
          std::vector<double>{25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0});
  declare("upper_torque_thresholds_nominal",
          std::vector<double>{35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0});
  declare("lower_torque_thresholds_acceleration",
          std::vector<double>{25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0});
  declare("upper_torque_thresholds_acceleration",
          std::vector<double>{35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0});
  declare("lower_force_thresholds_nominal", std::vector<double>{30.0, 30.0, 30.0, 25.0, 25.0, 25.0});
  declare("upper_force_thresholds_nominal", std::vector<double>{40.0, 40.0, 40.0, 35.0, 35.0, 35.0});
  declare("lower_force_thresholds_acceleration",
          std::vector<double>{30.0, 30.0, 30.0, 25.0, 25.0, 25.0});
  declare("upper_force_thresholds_acceleration",
          std::vector<double>{40.0, 40.0, 40.0, 35.0, 35.0, 35.0});
}

/// Sends the declared thresholds over service_server/set_full_collision_behavior. Returns
/// false only on a malformed parameter or a refused/unanswered request; a missing service
/// (mock or simulated hardware) is a warning, since nothing moves without franka_hardware
/// on the real arm anyway.
inline bool apply_collision_behavior(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr& node,
    const rclcpp::Client<franka_msgs::srv::SetFullCollisionBehavior>::SharedPtr& client) {
  using namespace std::chrono_literals;
  auto request = std::make_shared<franka_msgs::srv::SetFullCollisionBehavior::Request>();
  const auto fill = [&node](const char* name, auto& destination) {
    const auto values = node->get_parameter(name).as_double_array();
    if (values.size() != destination.size()) {
      RCLCPP_FATAL(node->get_logger(), "%s must have %zu entries, got %zu.", name,
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
  if (!client->wait_for_service(5s)) {
    RCLCPP_WARN(node->get_logger(),
                "service_server/set_full_collision_behavior is not available (mock or simulated "
                "hardware?). Collision thresholds stay whatever the robot has; on the real arm "
                "that is what Desk last set, which may be low enough to reflex.");
    return true;
  }
  auto future = client->async_send_request(request);
  if (future.wait_for(10s) != std::future_status::ready) {
    RCLCPP_FATAL(node->get_logger(), "set_full_collision_behavior did not respond.");
    return false;
  }
  if (!future.get()->success) {
    RCLCPP_FATAL(node->get_logger(), "set_full_collision_behavior was rejected.");
    return false;
  }
  RCLCPP_INFO(node->get_logger(),
              "Collision behavior set (upper force thresholds %.0f/%.0f/%.0f N).",
              request->upper_force_thresholds_nominal[0],
              request->upper_force_thresholds_nominal[1],
              request->upper_force_thresholds_nominal[2]);
  return true;
}

}  // namespace franka_trajectory_replay
