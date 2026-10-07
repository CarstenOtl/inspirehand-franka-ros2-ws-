// Copyright (c) 2023 Franka Robotics GmbH
// Copyright (c) 2026 Agile Robots SE
// Licensed under the Apache License, Version 2.0.
// See http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace franka_forge_cartesian_impedance {

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

/// The example's damped pseudo-inverse regularisation for the nullspace projector. At this
/// value lambda sits on top of the three smallest singular values of J^T near the threading
/// posture, so the "projector" stops annihilating range(J^T) and the joint spring pushes on
/// the tool (12.5 N at a 1 rad wrist excursion). Pass 0.0 for the exact pseudo-inverse.
constexpr double kNullspaceDampingLambda = 0.2;

/// Singular values at or below this are treated as zero when lambda is 0.
constexpr double kNullspaceSingularTolerance = 1.0e-9;

/// Which orientation error the task-space spring sees.
enum class RotationErrorForm {
  /// The example's -R vec(q_c^-1 q_d) = sin(theta/2) * axis. Halves the effective rotational
  /// stiffness at small angles and saturates past 180 deg.
  kQuaternionVector,
  /// ForgeUltra's get_pose_error(rot_error_type="axis_angle"): theta * axis. What the policy
  /// was distilled against.
  kAxisAngle,
};

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

/// computeError(): [p - p_d ; -R * vec(q_c^-1 q_d)] with q_c on q_d's hemisphere. With
/// kAxisAngle the rotation block carries the full angle instead of its half-angle sine, which
/// is what ForgeUltra's compute_dof_torque feeds the rotational spring.
inline Vector6d example_cartesian_error(
    const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation,
    const Eigen::Vector3d& position_d, const Eigen::Quaterniond& orientation_d,
    RotationErrorForm form = RotationErrorForm::kQuaternionVector) {
  Vector6d error;
  error.head(3) = position - position_d;
  Eigen::Quaterniond orientation_corrected = orientation;
  if (orientation_d.coeffs().dot(orientation_corrected.coeffs()) < 0.0) {
    orientation_corrected.coeffs() = -orientation_corrected.coeffs();
  }
  const Eigen::Quaterniond error_quaternion(orientation_corrected.inverse() * orientation_d);
  if (form == RotationErrorForm::kAxisAngle) {
    // R * axis_angle(q_c^-1 q_d) == axis_angle(q_d q_c^-1), i.e. the world-frame rotation
    // vector ForgeUltra computes as axis_angle_from_quat(quat_mul(target, quat_inv)).
    const Eigen::AngleAxisd axis_angle(error_quaternion);
    error.tail(3) = axis_angle.axis() * axis_angle.angle();
  } else {
    error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  }
  error.tail(3) = -orientation.toRotationMatrix() * error.tail(3);
  return error;
}

/// ForgeUltra's nullspace term, `compute_dof_torque` in forge_ultra/tasks/utils/control.py:
///
///   M_task = (J M^-1 J^T)^-1                  the task-space (operational-space) mass
///   Jbar^T = M_task J M^-1                    the dynamically consistent generalised inverse
///   u_null = M (kp (q0 - q) - kd qdot)        a joint PD, weighted by the arm mass matrix
///   tau_null = (I - J^T Jbar^T) u_null
///
/// Two things separate this from the example's `I - J^T pinv_lambda(J^T)` term. The projector
/// here annihilates range(J^T) exactly for any posture, so the joint spring cannot push on the
/// tool -- the example's damped pseudo-inverse stops being a projector once lambda reaches the
/// smallest singular values of J^T and leaks up to 12.5 N at a 1 rad wrist excursion. And the
/// mass weighting makes the joint PD act on accelerations rather than torques, which is what
/// the policy was distilled against.
///
/// `arm_mass_matrix` is the arm block of the WHOLE articulation's generalized mass matrix, as
/// Isaac takes it (`forge_franka_env.py`: `mass_matrix[:, arm_joint_ids, :][:, :, arm_joint_ids]`
/// over `root_physx_view.get_generalized_mass_matrices()`), so it carries the hand's inertia
/// through the finger joints. An arm-only chain is not the same matrix.
///
/// Allocation-free: every inverse is on a fixed-size Eigen matrix.
inline Vector7d forge_nullspace_torque(const Matrix6x7d& jacobian,
                                       const Matrix7d& arm_mass_matrix, const Vector7d& q,
                                       const Vector7d& dq, const Vector7d& q_nullspace,
                                       double nullspace_stiffness, double nullspace_damping) {
  const Matrix7x6d jacobian_transpose = jacobian.transpose();
  const Matrix7d mass_inverse = arm_mass_matrix.inverse();
  const Matrix6d mass_task = (jacobian * mass_inverse * jacobian_transpose).inverse();
  const Matrix6x7d jacobian_eef_inverse = mass_task * jacobian * mass_inverse;

  // Forge normalises the joint distance to [-pi, pi] before the spring; std::remainder is the
  // same wrap as its `(d + pi) % (2 pi) - pi`. No FR3 joint pair can be more than pi apart
  // within the limits today, but the policy's q0 is a parameter and the wrap is free.
  Vector7d distance = q_nullspace - q;
  for (int i = 0; i < distance.size(); ++i) {
    distance(i) = std::remainder(distance(i), 2.0 * M_PI);
  }

  const Vector7d u_null =
      arm_mass_matrix * (nullspace_stiffness * distance - nullspace_damping * dq);
  return (Matrix7d::Identity() - jacobian_transpose * jacobian_eef_inverse) * u_null;
}

/// The example's update() torque: task-space PD through J^T, a damped-pseudo-inverse
/// nullspace projection of a joint PD toward q_null, plus coriolis.
///
/// With `arm_mass_matrix` non-null the nullspace term is Forge's instead
/// (`forge_nullspace_torque`), which is what the policy was distilled against; the task term
/// and coriolis are unchanged. `nullspace_damping_lambda` is then unused.
inline CartesianImpedanceTerms example_cartesian_impedance(
    const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation,
    const Matrix6x7d& jacobian, const Vector7d& coriolis, const Vector7d& q,
    const Vector7d& dq, const Eigen::Vector3d& position_d,
    const Eigen::Quaterniond& orientation_d, const Vector7d& q_nullspace,
    const Matrix6d& stiffness, const Matrix6d& damping, double nullspace_stiffness,
    double nullspace_damping_lambda = kNullspaceDampingLambda,
    RotationErrorForm rotation_error_form = RotationErrorForm::kQuaternionVector,
    const Matrix7d* arm_mass_matrix = nullptr) {
  CartesianImpedanceTerms terms;
  terms.error =
      example_cartesian_error(position, orientation, position_d, orientation_d, rotation_error_form);

  // Damped pseudo-inverse of the Jacobian transpose, as the example computes it.
  const Matrix7x6d jacobian_transpose = jacobian.transpose();
  Eigen::JacobiSVD<Matrix7x6d> svd(jacobian_transpose, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto& singular_values = svd.singularValues();
  Matrix7x6d s_inverse = Matrix7x6d::Zero();
  const double lambda = std::max(0.0, nullspace_damping_lambda);
  for (int i = 0; i < singular_values.size(); ++i) {
    if (lambda <= 0.0 && singular_values(i) <= kNullspaceSingularTolerance) {
      continue;
    }
    s_inverse(i, i) =
        singular_values(i) / (singular_values(i) * singular_values(i) + lambda * lambda);
  }
  const Matrix6x7d jacobian_transpose_pinv =
      svd.matrixV() * s_inverse.transpose() * svd.matrixU().transpose();

  terms.tau_task =
      jacobian_transpose * (-stiffness * terms.error - damping * (jacobian * dq));
  if (arm_mass_matrix != nullptr) {
    terms.tau_nullspace =
        forge_nullspace_torque(jacobian, *arm_mass_matrix, q, dq, q_nullspace,
                               nullspace_stiffness, 2.0 * std::sqrt(nullspace_stiffness));
  } else {
    terms.tau_nullspace =
        (Matrix7d::Identity() - jacobian_transpose * jacobian_transpose_pinv) *
        (nullspace_stiffness * (q_nullspace - q) - 2.0 * std::sqrt(nullspace_stiffness) * dq);
  }
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

/// Skew-symmetric matrix of a vector: skew(a) * b == a.cross(b).
inline Eigen::Matrix3d skew_symmetric(const Eigen::Vector3d& v) {
  Eigen::Matrix3d s;
  s << 0.0, -v.z(), v.y(),
       v.z(), 0.0, -v.x(),
       -v.y(), v.x(), 0.0;
  return s;
}

/// Rotation from roll-pitch-yaw in the URDF convention: Rz(yaw) * Ry(pitch) * Rx(roll).
inline Eigen::Matrix3d rpy_to_rotation(const Eigen::Vector3d& rpy) {
  return (Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
}

/// Moves a base-frame geometric Jacobian from a frame's origin to a point rigidly attached to
/// it. ``offset_base`` is the origin-to-point vector expressed in the base frame. The point's
/// velocity is v + omega x offset, so the linear rows pick up -skew(offset) times the angular
/// rows; the angular rows are unchanged, a rigid body having one angular velocity.
inline Matrix6x7d shift_jacobian(const Matrix6x7d& jacobian, const Eigen::Vector3d& offset_base) {
  Matrix6x7d shifted = jacobian;
  shifted.topRows(3) -= skew_symmetric(offset_base) * jacobian.bottomRows(3);
  return shifted;
}

/// Angle of the rotation between two unit quaternions, in [0, pi].
inline double quaternion_angle(const Eigen::Quaterniond& a, const Eigen::Quaterniond& b) {
  const double dot = std::min(1.0, std::abs(a.coeffs().dot(b.coeffs())));
  return 2.0 * std::acos(dot);
}

/// ForgeUltra's `arm_torque = clamp(arm_torque, -100, 100)` at the end of compute_dof_torque.
/// A non-positive limit disables the clamp. It never binds on an FR3 (87 / 12 Nm joints); it is
/// here so the law is the training one line for line.
inline Vector7d clamp_torque(const Vector7d& tau, double limit) {
  if (!(limit > 0.0)) {
    return tau;
  }
  return tau.cwiseMax(-limit).cwiseMin(limit);
}

// --- ForgeUltra's target decode ------------------------------------------------------------
//
// `_apply_action` (forge_ultra/tasks/mdp/robot_control.py) runs at every 120 Hz physics
// substep, not once per policy step: it decodes `bolt_tip + action` and then clips that goal
// against the LIVE grasp pose, 20 mm per axis and 0.097 rad per Euler angle. The clipped
// target therefore keeps leading the hand by up to one clip as it moves, instead of arriving
// at once and being held for the tick. These helpers let the controller do the same clip at
// its own rate, from the goal the policy process sends.

/// isaacsim.core.utils.torch.get_euler_xyz: (roll, pitch, yaw) of Rz(yaw) Ry(pitch) Rx(roll),
/// each wrapped to [0, 2 pi) exactly as the torch version does with `% (2 pi)`.
inline Eigen::Vector3d isaac_euler_xyz(const Eigen::Quaterniond& q) {
  const double qw = q.w(), qx = q.x(), qy = q.y(), qz = q.z();
  const double sinr_cosp = 2.0 * (qw * qx + qy * qz);
  const double cosr_cosp = qw * qw - qx * qx - qy * qy + qz * qz;
  const double roll = std::atan2(sinr_cosp, cosr_cosp);
  const double sinp = 2.0 * (qw * qy - qz * qx);
  const double pitch =
      std::abs(sinp) >= 1.0 ? std::copysign(M_PI / 2.0, sinp) : std::asin(sinp);
  const double siny_cosp = 2.0 * (qw * qz + qx * qy);
  const double cosy_cosp = qw * qw + qx * qx - qy * qy - qz * qz;
  const double yaw = std::atan2(siny_cosp, cosy_cosp);
  const auto wrap = [](double angle) {
    const double two_pi = 2.0 * M_PI;
    double r = std::fmod(angle, two_pi);
    if (r < 0.0) {
      r += two_pi;
    }
    return r;
  };
  return Eigen::Vector3d(wrap(roll), wrap(pitch), wrap(yaw));
}

/// isaacsim.core.utils.torch.quat_from_euler_xyz: the quaternion of Rz(yaw) Ry(pitch) Rx(roll).
inline Eigen::Quaterniond isaac_quat_from_euler_xyz(double roll, double pitch, double yaw) {
  const double cy = std::cos(yaw * 0.5), sy = std::sin(yaw * 0.5);
  const double cr = std::cos(roll * 0.5), sr = std::sin(roll * 0.5);
  const double cp = std::cos(pitch * 0.5), sp = std::sin(pitch * 0.5);
  return Eigen::Quaterniond(cy * cr * cp + sy * sr * sp, cy * sr * cp - sy * cr * sp,
                            cy * cr * sp + sy * sr * cp, sy * cr * cp - cy * sr * sp);
}

/// forge_tg2_utils.wrap_yaw: map [0, 2 pi) onto (-125, 235] degrees so the FR3's joint-7 limit
/// does not split the yaw range the policy works in.
inline double forge_wrap_yaw(double yaw) {
  return yaw > 235.0 * M_PI / 180.0 ? yaw - 2.0 * M_PI : yaw;
}

struct ForgeClippedTarget {
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
  bool position_clipped{false};
  bool orientation_clipped{false};
};

/// `_apply_action` step (2): clip a preclipped goal against the measured grasp pose. The
/// position clip is component-wise, the orientation clip is per Euler angle with the yaw
/// wrapped first, both in the frame `clip_frame` names (training's world frame, which is the
/// FR3 base yawed by pi: `clip_frame` rotates base-frame vectors into it). Inputs and the
/// result are in the base frame; the result's orientation is on the measured quaternion's
/// hemisphere so a caller can difference the two directly.
inline ForgeClippedTarget forge_clip_target(const Eigen::Vector3d& goal_position,
                                            const Eigen::Quaterniond& goal_orientation,
                                            const Eigen::Vector3d& measured_position,
                                            const Eigen::Quaterniond& measured_orientation,
                                            const Eigen::Quaterniond& clip_frame,
                                            double position_threshold,
                                            double rotation_threshold) {
  ForgeClippedTarget out;
  const Eigen::Quaterniond clip_frame_inverse = clip_frame.conjugate();

  // (2.a) position, per axis
  const Eigen::Vector3d delta = clip_frame * (goal_position - measured_position);
  Eigen::Vector3d clipped;
  for (int i = 0; i < 3; ++i) {
    clipped(i) = std::clamp(delta(i), -position_threshold, position_threshold);
    out.position_clipped = out.position_clipped || std::abs(delta(i)) > position_threshold;
  }
  out.position = measured_position + clip_frame_inverse * clipped;

  // (2.b) orientation, per Euler angle
  const Eigen::Vector3d current = isaac_euler_xyz(clip_frame * measured_orientation);
  const Eigen::Vector3d desired = isaac_euler_xyz(clip_frame * goal_orientation);
  double curr_roll = current.x(), curr_pitch = current.y(), curr_yaw = current.z();
  double desired_roll = desired.x(), desired_pitch = desired.y(), desired_yaw = desired.z();

  curr_yaw = forge_wrap_yaw(curr_yaw);
  desired_yaw = forge_wrap_yaw(desired_yaw);
  const double delta_yaw = desired_yaw - curr_yaw;
  const double yaw = curr_yaw + std::clamp(delta_yaw, -rotation_threshold, rotation_threshold);

  // get_euler_xyz already returns [0, 2 pi), so these two never fire; kept as written.
  if (desired_roll < 0.0) {
    desired_roll += 2.0 * M_PI;
  }
  if (desired_pitch < 0.0) {
    desired_pitch += 2.0 * M_PI;
  }
  const double delta_roll = desired_roll - curr_roll;
  const double roll = curr_roll + std::clamp(delta_roll, -rotation_threshold, rotation_threshold);

  if (curr_pitch > M_PI) {
    curr_pitch -= 2.0 * M_PI;
  }
  if (desired_pitch > M_PI) {
    desired_pitch -= 2.0 * M_PI;
  }
  const double delta_pitch = desired_pitch - curr_pitch;
  const double pitch =
      curr_pitch + std::clamp(delta_pitch, -rotation_threshold, rotation_threshold);

  out.orientation_clipped = std::abs(delta_yaw) > rotation_threshold ||
                            std::abs(delta_roll) > rotation_threshold ||
                            std::abs(delta_pitch) > rotation_threshold;
  out.orientation = clip_frame_inverse * isaac_quat_from_euler_xyz(roll, pitch, yaw);
  out.orientation.normalize();
  if (out.orientation.coeffs().dot(measured_orientation.coeffs()) < 0.0) {
    out.orientation.coeffs() = -out.orientation.coeffs();
  }
  return out;
}

}  // namespace franka_forge_cartesian_impedance
