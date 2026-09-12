// Copyright (c) 2026 Agile Robots SE
// Licensed under the Apache License, Version 2.0.
// See http://www.apache.org/licenses/LICENSE-2.0

#include <gtest/gtest.h>

#include <franka_trajectory_replay/fr3_kinematics.hpp>
#include <franka_trajectory_replay/cartesian_impedance.hpp>

using franka_trajectory_replay::fr3_flange_transform;
using franka_trajectory_replay::fr3_zero_jacobian;
using franka_trajectory_replay::Vector7d;

namespace {

Vector7d joints(double a, double b, double c, double d, double e, double f, double g) {
  Vector7d q;
  q << a, b, c, d, e, f, g;
  return q;
}

}  // namespace

// Reference values from franka_trajectory_replay/kinematics.py (flange_transform), which is
// checked against pinocchio on the FR3 URDF in test/python/test_kinematics.py.
TEST(Fr3Kinematics, flange_transform_matches_the_python_dh_model) {
  const auto zero = fr3_flange_transform(Vector7d::Zero());
  EXPECT_NEAR(zero(0, 3), 0.088, 1e-12);
  EXPECT_NEAR(zero(1, 3), 0.0, 1e-12);
  EXPECT_NEAR(zero(2, 3), 0.926, 1e-12);
  EXPECT_NEAR(zero(0, 0), 1.0, 1e-12);

  const auto ready = fr3_flange_transform(joints(0.0, -M_PI / 4, 0.0, -3 * M_PI / 4, 0.0, M_PI / 2, M_PI / 4));
  EXPECT_NEAR(ready(0, 3), 0.306890567, 1e-9);
  EXPECT_NEAR(ready(1, 3), 0.0, 1e-9);
  EXPECT_NEAR(ready(2, 3), 0.590282052, 1e-9);
  EXPECT_NEAR(ready(0, 0), 0.707106781, 1e-9);
  EXPECT_NEAR(ready(0, 1), -0.707106781, 1e-9);

  const auto test = fr3_flange_transform(joints(0.3, -0.6, 0.2, -2.0, 0.4, 1.8, 0.9));
  EXPECT_NEAR(test(0, 3), 0.304506639, 1e-9);
  EXPECT_NEAR(test(1, 3), 0.238834613, 1e-9);
  EXPECT_NEAR(test(2, 3), 0.719535806, 1e-9);
  EXPECT_NEAR(test(0, 0), 0.909289161, 1e-9);
  EXPECT_NEAR(test(0, 1), -0.363460569, 1e-9);
  EXPECT_NEAR(test(0, 2), 0.202705787, 1e-9);
  // A rotation matrix.
  EXPECT_TRUE((test.block<3, 3>(0, 0) * test.block<3, 3>(0, 0).transpose())
                  .isApprox(Eigen::Matrix3d::Identity(), 1e-12));
}

TEST(Fr3Kinematics, jacobian_matches_the_python_finite_differences) {
  const auto jacobian = fr3_zero_jacobian(joints(0.3, -0.6, 0.2, -2.0, 0.4, 1.8, 0.9));
  // Column 0: joint 1 spins about base z, so v = z x p and w = z.
  EXPECT_NEAR(jacobian(0, 0), -0.238834613, 1e-8);
  EXPECT_NEAR(jacobian(1, 0), 0.304506639, 1e-8);
  EXPECT_NEAR(jacobian(2, 0), 0.0, 1e-8);
  EXPECT_NEAR(jacobian(5, 0), 1.0, 1e-8);
  // Column 3. The Python reference forms the angular rows from the rotation at q + step
  // rather than at q, so those rows carry a one-sided error of the step size (1e-6).
  EXPECT_NEAR(jacobian(0, 3), -0.099831863, 1e-8);
  EXPECT_NEAR(jacobian(1, 3), 0.010936342, 1e-8);
  EXPECT_NEAR(jacobian(2, 3), 0.483718408, 1e-8);
  EXPECT_NEAR(jacobian(3, 3), 0.446275026, 2e-6);
  EXPECT_NEAR(jacobian(4, 3), -0.887837298, 2e-6);
  EXPECT_NEAR(jacobian(5, 3), 0.112177539, 2e-6);
  // Column 6: the flange sits on joint 7's axis, so it contributes no linear velocity.
  EXPECT_NEAR(jacobian(0, 6), 0.0, 1e-8);
  EXPECT_NEAR(jacobian(2, 6), 0.0, 1e-8);
  EXPECT_NEAR(jacobian(3, 6), 0.202706157, 2e-6);
  EXPECT_NEAR(jacobian(4, 6), 0.417093534, 2e-6);
  EXPECT_NEAR(jacobian(5, 6), -0.885970455, 2e-6);
}

TEST(Fr3Kinematics, jacobian_is_the_derivative_of_the_forward_kinematics) {
  const Vector7d q = joints(-0.4, 0.5, 0.7, -1.6, -0.9, 2.2, -1.3);
  const auto jacobian = fr3_zero_jacobian(q);
  const double step = 1e-6;
  for (int j = 0; j < 7; ++j) {
    Vector7d forward = q;
    Vector7d backward = q;
    forward(j) += step;
    backward(j) -= step;
    const Eigen::Matrix4d t_f = fr3_flange_transform(forward);
    const Eigen::Matrix4d t_b = fr3_flange_transform(backward);
    const Eigen::Vector3d linear = (t_f.block<3, 1>(0, 3) - t_b.block<3, 1>(0, 3)) / (2 * step);
    const Eigen::Matrix3d d_rotation =
        (t_f.block<3, 3>(0, 0) - t_b.block<3, 3>(0, 0)) / (2 * step) * t_f.block<3, 3>(0, 0).transpose();
    const Eigen::Vector3d angular(d_rotation(2, 1), d_rotation(0, 2), d_rotation(1, 0));
    for (int r = 0; r < 3; ++r) {
      EXPECT_NEAR(jacobian(r, j), linear(r), 1e-6) << "joint " << j << " linear " << r;
      EXPECT_NEAR(jacobian(3 + r, j), angular(r), 1e-6) << "joint " << j << " angular " << r;
    }
  }
}


TEST(Fr3Kinematics, shifted_jacobian_is_the_derivative_of_the_tool_point) {
  using franka_trajectory_replay::rpy_to_rotation;
  using franka_trajectory_replay::shift_jacobian;
  using franka_trajectory_replay::skew_symmetric;

  // The Inspire hand's grasp centre in the flange frame.
  const Eigen::Vector3d tool(-0.0874, -0.0327, 0.1453);
  const Vector7d q = joints(0.3, -0.6, 0.2, -2.0, 0.4, 1.8, 0.9);

  const Eigen::Matrix4d flange = fr3_flange_transform(q);
  const Eigen::Vector3d offset_base = flange.block<3, 3>(0, 0) * tool;
  const auto shifted = shift_jacobian(fr3_zero_jacobian(q), offset_base);

  const double step = 1e-6;
  for (int j = 0; j < 7; ++j) {
    Vector7d forward = q;
    Vector7d backward = q;
    forward(j) += step;
    backward(j) -= step;
    const auto tool_point = [&tool](const Vector7d& value) {
      const Eigen::Matrix4d t = fr3_flange_transform(value);
      return Eigen::Vector3d(t.block<3, 1>(0, 3) + t.block<3, 3>(0, 0) * tool);
    };
    const Eigen::Vector3d linear = (tool_point(forward) - tool_point(backward)) / (2 * step);
    for (int r = 0; r < 3; ++r) {
      EXPECT_NEAR(shifted(r, j), linear(r), 1e-6) << "joint " << j << " linear " << r;
    }
    // A rigid body has one angular velocity: the angular rows never move.
    for (int r = 3; r < 6; ++r) {
      EXPECT_DOUBLE_EQ(shifted(r, j), fr3_zero_jacobian(q)(r, j));
    }
  }
  // A zero offset changes nothing.
  EXPECT_TRUE(shift_jacobian(fr3_zero_jacobian(q), Eigen::Vector3d::Zero())
                  .isApprox(fr3_zero_jacobian(q), 1e-15));
  // skew(a) b == a x b, and rpy follows the URDF convention Rz*Ry*Rx.
  const Eigen::Vector3d a(0.2, -0.5, 0.9);
  const Eigen::Vector3d b(-0.3, 0.7, 0.1);
  EXPECT_TRUE((skew_symmetric(a) * b).isApprox(a.cross(b), 1e-15));
  const Eigen::Vector3d rpy(0.3, -0.2, 1.1);
  const Eigen::Matrix3d expected =
      (Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX())).toRotationMatrix();
  EXPECT_TRUE(rpy_to_rotation(rpy).isApprox(expected, 1e-15));
  EXPECT_TRUE(rpy_to_rotation(Eigen::Vector3d::Zero()).isIdentity(1e-15));
}
