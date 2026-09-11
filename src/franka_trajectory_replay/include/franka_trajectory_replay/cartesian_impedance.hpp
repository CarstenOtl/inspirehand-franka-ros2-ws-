// Copyright (c) 2023 Franka Robotics GmbH
// Copyright (c) 2026 Agile Robots SE
// Licensed under the Apache License, Version 2.0.
// See http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <array>
#include <cmath>

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace franka_trajectory_replay {

// franka_example_controllers/CartesianImpedanceExampleController (franka_ros2 v3.5.3),
// with its reference (p_d, q_d, q_null) supplied by the caller instead of the demo arc
// and the activation pose. Gravity compensation and the torque rate limiter remain in
// the same robot/franka_hardware path as the example.

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix7d = Eigen::Matrix<double, 7, 7>;
using Matrix6x7d = Eigen::Matrix<double, 6, 7>;
using Matrix7x6d = Eigen::Matrix<double, 7, 6>;

/// The example's damped pseudo-inverse regularisation for the nullspace projector.
constexpr double kNullspaceDampingLambda = 0.2;

struct CartesianImpedanceTerms {
  Vector7d tau_task{Vector7d::Zero()};
  Vector7d tau_nullspace{Vector7d::Zero()};
  Vector7d tau_coriolis{Vector7d::Zero()};
  Vector7d tau_command{Vector7d::Zero()};
  Vector6d error{Vector6d::Zero()};
};

/// buildGains(): diagonal stiffness from six values, damping 2 sqrt(k) (critically damped).
inline void example_cartesian_gains(const std::array<double, 6>& k, Matrix6d& stiffness,
                                    Matrix6d& damping) {
  stiffness.setZero();
  damping.setZero();
  for (int i = 0; i < 6; ++i) {
    const double ki = std::max(0.0, k[i]);
    stiffness(i, i) = ki;
    damping(i, i) = 2.0 * std::sqrt(ki);
  }
}

/// computeError(): [p - p_d ; -R * vec(q_c^-1 q_d)] with q_c on q_d's hemisphere.
inline Vector6d example_cartesian_error(const Eigen::Vector3d& position,
                                        const Eigen::Quaterniond& orientation,
                                        const Eigen::Vector3d& position_d,
                                        const Eigen::Quaterniond& orientation_d) {
  Vector6d error;
  error.head(3) = position - position_d;
  Eigen::Quaterniond orientation_corrected = orientation;
  if (orientation_d.coeffs().dot(orientation_corrected.coeffs()) < 0.0) {
    orientation_corrected.coeffs() = -orientation_corrected.coeffs();
  }
  const Eigen::Quaterniond error_quaternion(orientation_corrected.inverse() * orientation_d);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  error.tail(3) = -orientation.toRotationMatrix() * error.tail(3);
  return error;
}

/// The example's update() torque: task-space PD through J^T, a damped-pseudo-inverse
/// nullspace projection of a joint PD toward q_null, plus coriolis.
inline CartesianImpedanceTerms example_cartesian_impedance(
    const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation,
    const Matrix6x7d& jacobian, const Vector7d& coriolis, const Vector7d& q,
    const Vector7d& dq, const Eigen::Vector3d& position_d,
    const Eigen::Quaterniond& orientation_d, const Vector7d& q_nullspace,
    const Matrix6d& stiffness, const Matrix6d& damping, double nullspace_stiffness) {
  CartesianImpedanceTerms terms;
  terms.error = example_cartesian_error(position, orientation, position_d, orientation_d);

  // Damped pseudo-inverse of the Jacobian transpose, as the example computes it.
  const Matrix7x6d jacobian_transpose = jacobian.transpose();
  Eigen::JacobiSVD<Matrix7x6d> svd(jacobian_transpose, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto& singular_values = svd.singularValues();
  Matrix7x6d s_inverse = Matrix7x6d::Zero();
  for (int i = 0; i < singular_values.size(); ++i) {
    s_inverse(i, i) = singular_values(i) /
                      (singular_values(i) * singular_values(i) +
                       kNullspaceDampingLambda * kNullspaceDampingLambda);
  }
  const Matrix6x7d jacobian_transpose_pinv =
      svd.matrixV() * s_inverse.transpose() * svd.matrixU().transpose();

  terms.tau_task =
      jacobian_transpose * (-stiffness * terms.error - damping * (jacobian * dq));
  terms.tau_nullspace =
      (Matrix7d::Identity() - jacobian_transpose * jacobian_transpose_pinv) *
      (nullspace_stiffness * (q_nullspace - q) - 2.0 * std::sqrt(nullspace_stiffness) * dq);
  terms.tau_coriolis = coriolis;
  terms.tau_command = terms.tau_task + terms.tau_nullspace + terms.tau_coriolis;
  return terms;
}

/// The example's post-command reference filter: first-order on the position, slerp on the
/// orientation with the working quaternion kept on the target's hemisphere.
inline void example_reference_filter(double alpha, const Eigen::Vector3d& position_target,
                                     const Eigen::Quaterniond& orientation_target,
                                     Eigen::Vector3d& position_d,
                                     Eigen::Quaterniond& orientation_d) {
  position_d = alpha * position_target + (1.0 - alpha) * position_d;
  if (orientation_d.coeffs().dot(orientation_target.coeffs()) < 0.0) {
    orientation_d.coeffs() = -orientation_d.coeffs();
  }
  orientation_d = orientation_d.slerp(alpha, orientation_target);
  orientation_d.normalize();
}

/// The example's gain filter, applied to the stiffness and damping matrices alike.
inline void example_gain_filter(double alpha, const Matrix6d& stiffness_target,
                                const Matrix6d& damping_target, double nullspace_target,
                                Matrix6d& stiffness, Matrix6d& damping,
                                double& nullspace_stiffness) {
  stiffness = alpha * stiffness_target + (1.0 - alpha) * stiffness;
  damping = alpha * damping_target + (1.0 - alpha) * damping;
  nullspace_stiffness = alpha * nullspace_target + (1.0 - alpha) * nullspace_stiffness;
}

/// Angle of the rotation between two unit quaternions, in [0, pi].
inline double quaternion_angle(const Eigen::Quaterniond& a, const Eigen::Quaterniond& b) {
  const double dot = std::min(1.0, std::abs(a.coeffs().dot(b.coeffs())));
  return 2.0 * std::acos(dot);
}

}  // namespace franka_trajectory_replay
