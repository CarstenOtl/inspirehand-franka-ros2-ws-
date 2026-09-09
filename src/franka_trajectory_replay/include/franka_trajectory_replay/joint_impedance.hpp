// Copyright (c) 2023 Franka Robotics GmbH
// Copyright (c) 2026 Agile Robots SE
// Licensed under the Apache License, Version 2.0.
// See http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <Eigen/Core>

namespace franka_trajectory_replay {

// JointImpedanceExampleController::update(), with its sinusoidal reference
// supplied by the waypoint sampler instead. Gravity compensation and torque
// rate limiting remain in the same robot/franka_hardware path as the example.
inline Eigen::Matrix<double, 7, 1> example_joint_impedance(
    const Eigen::Matrix<double, 7, 1>& reference,
    const Eigen::Matrix<double, 7, 1>& position,
    const Eigen::Matrix<double, 7, 1>& velocity,
    const Eigen::Matrix<double, 7, 1>& stiffness,
    const Eigen::Matrix<double, 7, 1>& damping,
    Eigen::Matrix<double, 7, 1>& filtered_velocity) {
  constexpr double kAlpha = 0.99;
  filtered_velocity = (1 - kAlpha) * filtered_velocity + kAlpha * velocity;
  return stiffness.cwiseProduct(reference - position) - damping.cwiseProduct(filtered_velocity);
}

}  // namespace franka_trajectory_replay
