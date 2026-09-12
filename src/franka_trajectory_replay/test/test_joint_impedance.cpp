// Copyright (c) 2026 Agile Robots SE
// Licensed under the Apache License, Version 2.0.
// See http://www.apache.org/licenses/LICENSE-2.0

#include <gtest/gtest.h>
#include <cmath>
#include <franka_trajectory_replay/joint_impedance.hpp>

using Vector7d = Eigen::Matrix<double, 7, 1>;
using franka_trajectory_replay::example_joint_impedance;

TEST(ExampleJointImpedance, holds_initial_pose_with_zero_torque_at_rest) {
  const Vector7d q = Vector7d::Constant(0.7);
  Vector7d filtered = Vector7d::Zero();
  EXPECT_TRUE(example_joint_impedance(q, q, Vector7d::Zero(),
      Vector7d::Constant(24), Vector7d::Constant(2), filtered).isZero());
}

TEST(ExampleJointImpedance, matches_upstream_example_for_reference_and_feedback_sequence) {
  Vector7d stiffness, damping, reference, position, velocity;
  stiffness << 24, 24, 24, 24, 10, 6, 2;
  damping << 2, 2, 2, 1, 1, 1, 0.5;
  position << -0.3, 0.1, 0.2, -1.8, 1.0, 2.0, -0.2;
  Vector7d filtered = Vector7d::Zero();
  Vector7d upstream_filtered = Vector7d::Zero();
  for (int sample = 0; sample < 5000; ++sample) {
    const double t = sample * 0.001;
    reference = position;
    // Exercise all seven waypoint reference channels, not only joints 4/5.
    for (int joint = 0; joint < 7; ++joint) {
      reference[joint] += 0.1 * std::sin(t + joint);
      velocity[joint] = 0.2 * std::cos(2 * t + joint);
    }
    // Original example's torque equation, including its velocity filter.
    upstream_filtered = (1 - 0.99) * upstream_filtered + 0.99 * velocity;
    const Vector7d expected = stiffness.cwiseProduct(reference - position) +
        damping.cwiseProduct(-upstream_filtered);
    const auto actual = example_joint_impedance(
        reference, position, velocity, stiffness, damping, filtered);
    EXPECT_TRUE(actual.isApprox(expected, 1e-14)) << sample;
  }
}
