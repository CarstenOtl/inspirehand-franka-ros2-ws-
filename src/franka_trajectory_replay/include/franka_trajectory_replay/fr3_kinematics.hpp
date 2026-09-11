// Copyright (c) 2026 Agile Robots SE
// Licensed under the Apache License, Version 2.0.
// See http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <array>
#include <cmath>

#include <Eigen/Dense>

#include "franka_trajectory_replay/cartesian_impedance.hpp"

namespace franka_trajectory_replay {

// Forward kinematics and geometric Jacobian of the FR3 flange from Franka's published
// modified-DH (Craig) table, the same table as franka_trajectory_replay/kinematics.py, which
// is checked against pinocchio on the URDF. Used when the controller runs without
// franka_hardware's robot model (model_source: dh), i.e. in simulation.

constexpr double kFr3HalfPi = 1.5707963267948966;

/// a_{i-1}, d_i, alpha_{i-1}; the last row is the fixed flange transform (fr3_link8).
constexpr std::array<std::array<double, 3>, 8> kFr3Dh{{
    {0.0, 0.333, 0.0},
    {0.0, 0.0, -kFr3HalfPi},
    {0.0, 0.316, kFr3HalfPi},
    {0.0825, 0.0, kFr3HalfPi},
    {-0.0825, 0.384, -kFr3HalfPi},
    {0.0, 0.0, kFr3HalfPi},
    {0.088, 0.0, kFr3HalfPi},
    {0.0, 0.107, 0.0},
}};

inline Eigen::Matrix4d fr3_dh_transform(double a, double d, double alpha, double theta) {
  const double ca = std::cos(alpha);
  const double sa = std::sin(alpha);
  const double ct = std::cos(theta);
  const double st = std::sin(theta);
  Eigen::Matrix4d transform;
  transform << ct, -st, 0.0, a,
               st * ca, ct * ca, -sa, -d * sa,
               st * sa, ct * sa, ca, d * ca,
               0.0, 0.0, 0.0, 1.0;
  return transform;
}

/// The seven joint frames (index 0..6) and the flange (index 7), all in the base frame.
inline std::array<Eigen::Matrix4d, 8> fr3_frames(const Vector7d& q) {
  std::array<Eigen::Matrix4d, 8> frames;
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  for (int i = 0; i < 8; ++i) {
    const double theta = i < 7 ? q(i) : 0.0;
    transform = transform * fr3_dh_transform(kFr3Dh[i][0], kFr3Dh[i][1], kFr3Dh[i][2], theta);
    frames[i] = transform;
  }
  return frames;
}

/// Flange (fr3_link8) in the base frame (fr3_link0).
inline Eigen::Matrix4d fr3_flange_transform(const Vector7d& q) {
  return fr3_frames(q)[7];
}

/// Geometric Jacobian of the flange point: rows 0-2 linear, 3-5 angular, base frame. Each
/// joint's axis is the z axis of its own frame and its origin lies on that axis.
inline Matrix6x7d fr3_zero_jacobian(const Vector7d& q) {
  const auto frames = fr3_frames(q);
  const Eigen::Vector3d flange = frames[7].block<3, 1>(0, 3);
  Matrix6x7d jacobian;
  for (int i = 0; i < 7; ++i) {
    const Eigen::Vector3d axis = frames[i].block<3, 1>(0, 2);
    const Eigen::Vector3d origin = frames[i].block<3, 1>(0, 3);
    jacobian.block<3, 1>(0, i) = axis.cross(flange - origin);
    jacobian.block<3, 1>(3, i) = axis;
  }
  return jacobian;
}

}  // namespace franka_trajectory_replay
