// Copyright (c) 2026 Agile Robots SE
// Licensed under the Apache License, Version 2.0.
// See http://www.apache.org/licenses/LICENSE-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include <franka_trajectory_replay/cartesian_impedance.hpp>

using franka_trajectory_replay::example_cartesian_error;
using franka_trajectory_replay::example_cartesian_gains;
using franka_trajectory_replay::example_cartesian_impedance;
using franka_trajectory_replay::example_gain_filter;
using franka_trajectory_replay::example_reference_filter;
using franka_trajectory_replay::forge_nullspace_torque;
using franka_trajectory_replay::Matrix7d;
using franka_trajectory_replay::Matrix6d;
using franka_trajectory_replay::RotationErrorForm;
using franka_trajectory_replay::Matrix6x7d;
using franka_trajectory_replay::Vector6d;
using franka_trajectory_replay::Vector7d;

namespace {

// A Jacobian with the shape of the FR3's: full row rank, values of the right size.
Matrix6x7d synthetic_jacobian(double phase) {
  Matrix6x7d jacobian;
  for (int row = 0; row < 6; ++row) {
    for (int column = 0; column < 7; ++column) {
      jacobian(row, column) =
          0.4 * std::sin(1.3 * row + 0.7 * column + phase) + 0.1 * (row == column ? 1.0 : 0.0);
    }
  }
  return jacobian;
}

// CartesianImpedanceExampleController::update() as written upstream (v3.5.3), with the
// dynamic-size Eigen types it uses. The header uses fixed-size matrices; this is what it is
// checked against.
struct UpstreamExample {
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_;
  double nullspace_stiffness_{20.0};
  Eigen::Vector3d position_d_;
  Eigen::Quaterniond orientation_d_;
  Eigen::Matrix<double, 7, 1> q_d_nullspace_;
  double filter_params_{0.005};

  Eigen::Matrix<double, 6, 1> computeError(const Eigen::Vector3d& position,
                                           const Eigen::Quaterniond& orientation,
                                           const Eigen::Affine3d& transform) const {
    Eigen::Matrix<double, 6, 1> error;
    error.head(3) << position - position_d_;
    Eigen::Quaterniond orientation_corrected = orientation;
    if (orientation_d_.coeffs().dot(orientation_corrected.coeffs()) < 0.0) {
      orientation_corrected.coeffs() = -orientation_corrected.coeffs();
    }
    Eigen::Quaterniond error_quaternion(orientation_corrected.inverse() * orientation_d_);
    error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
    error.tail(3) << -transform.rotation() * error.tail(3);
    return error;
  }

  Eigen::Matrix<double, 7, 1> update(const Eigen::Vector3d& position,
                                     const Eigen::Quaterniond& orientation,
                                     const std::array<double, 42>& jacobian_array,
                                     const std::array<double, 7>& coriolis_array,
                                     const Eigen::Matrix<double, 7, 1>& q_,
                                     const Eigen::Matrix<double, 7, 1>& dq_,
                                     const Eigen::Vector3d& target_position,
                                     const Eigen::Quaterniond& target_orientation,
                                     const Eigen::Matrix<double, 6, 6>& target_stiffness,
                                     const Eigen::Matrix<double, 6, 6>& target_damping,
                                     double target_nullspace_stiffness) {
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
    Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());

    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    transform.translation() = position;
    transform.rotate(orientation.toRotationMatrix());

    const auto error = computeError(position, orientation, transform);

    Eigen::Matrix<double, 7, 1> tau_task, tau_nullspace, tau_d;

    Eigen::MatrixXd jacobian_transpose_pinv;
    double lambda = 0.2;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian.transpose(),
                                          Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType sing_vals = svd.singularValues();
    Eigen::MatrixXd S = jacobian.transpose();
    S.setZero();
    for (int i = 0; i < sing_vals.size(); i++) {
      S(i, i) = sing_vals(i) / (sing_vals(i) * sing_vals(i) + lambda * lambda);
    }
    jacobian_transpose_pinv = svd.matrixV() * S.transpose() * svd.matrixU().transpose();

    tau_task << jacobian.transpose() *
                    (-cartesian_stiffness_ * error - cartesian_damping_ * (jacobian * dq_));

    tau_nullspace << (Eigen::Matrix<double, 7, 7>::Identity() -
                      jacobian.transpose() * jacobian_transpose_pinv) *
                         (nullspace_stiffness_ * (q_d_nullspace_ - q_) -
                          2.0 * std::sqrt(nullspace_stiffness_) * dq_);

    tau_d << tau_task + tau_nullspace + coriolis;

    cartesian_stiffness_ =
        filter_params_ * target_stiffness + (1.0 - filter_params_) * cartesian_stiffness_;
    cartesian_damping_ =
        filter_params_ * target_damping + (1.0 - filter_params_) * cartesian_damping_;
    nullspace_stiffness_ = filter_params_ * target_nullspace_stiffness +
                           (1.0 - filter_params_) * nullspace_stiffness_;

    position_d_ = filter_params_ * target_position + (1.0 - filter_params_) * position_d_;
    if (orientation_d_.coeffs().dot(target_orientation.coeffs()) < 0.0) {
      orientation_d_.coeffs() = -orientation_d_.coeffs();
    }
    orientation_d_ = orientation_d_.slerp(filter_params_, target_orientation);
    orientation_d_.normalize();
    return tau_d;
  }
};

// A symmetric positive definite stand-in for the arm mass matrix, in the range the FR3's
// actually occupies (diagonal about 0.03 to 2.2 kg m^2 at the threading posture).
Matrix7d synthetic_mass_matrix(double phase) {
  Matrix7d a;
  for (int row = 0; row < 7; ++row) {
    for (int column = 0; column < 7; ++column) {
      a(row, column) = 0.3 * std::cos(0.9 * row + 1.1 * column + phase);
    }
  }
  Matrix7d mass = a.transpose() * a;
  for (int i = 0; i < 7; ++i) {
    mass(i, i) += 0.05 + 0.3 * (6 - i);
  }
  return mass;
}

// compute_dof_torque's nullspace block, transcribed from forge_ultra/tasks/utils/control.py.
Vector7d forge_reference_nullspace(const Matrix6x7d& jacobian, const Matrix7d& mass,
                                   const Vector7d& q, const Vector7d& dq,
                                   const Vector7d& q_null, double kp_null, double kd_null) {
  const Eigen::MatrixXd jacobian_t = jacobian.transpose();
  const Eigen::MatrixXd mass_inv = mass.inverse();
  const Eigen::MatrixXd mass_task = (jacobian * mass_inv * jacobian_t).inverse();
  const Eigen::MatrixXd j_eef_inv = mass_task * jacobian * mass_inv;
  Eigen::VectorXd distance = q_null - q;
  for (int i = 0; i < distance.size(); ++i) {
    distance(i) = std::fmod(distance(i) + M_PI, 2.0 * M_PI);
    if (distance(i) < 0.0) {
      distance(i) += 2.0 * M_PI;  // python's % returns the sign of the divisor
    }
    distance(i) -= M_PI;
  }
  const Eigen::VectorXd u_null = mass * (kd_null * -dq + kp_null * distance);
  const Eigen::VectorXd torque_null =
      (Eigen::MatrixXd::Identity(7, 7) - jacobian_t * j_eef_inv) * u_null;
  return Vector7d(torque_null);
}

}  // namespace

TEST(ExampleCartesianImpedance, gains_are_diagonal_and_critically_damped) {
  Matrix6d stiffness, damping;
  example_cartesian_gains({150.0, 150.0, 150.0, 10.0, 10.0, 10.0}, stiffness, damping);
  EXPECT_DOUBLE_EQ(stiffness(0, 0), 150.0);
  EXPECT_DOUBLE_EQ(stiffness(5, 5), 10.0);
  EXPECT_DOUBLE_EQ(damping(0, 0), 2.0 * std::sqrt(150.0));
  EXPECT_DOUBLE_EQ(damping(3, 3), 2.0 * std::sqrt(10.0));
  EXPECT_DOUBLE_EQ(stiffness(0, 1), 0.0);
  // Negative requests clamp to zero as upstream.
  example_cartesian_gains({-1.0, 0, 0, 0, 0, 0}, stiffness, damping);
  EXPECT_DOUBLE_EQ(stiffness(0, 0), 0.0);
}

TEST(ExampleCartesianImpedance, zero_task_and_nullspace_torque_at_rest_on_reference) {
  const Eigen::Vector3d p(0.4, 0.1, 0.5);
  const Eigen::Quaterniond q = Eigen::Quaterniond(Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()));
  Matrix6d stiffness, damping;
  example_cartesian_gains({150.0, 150.0, 150.0, 10.0, 10.0, 10.0}, stiffness, damping);
  const Vector7d q_joint = Vector7d::Constant(0.3);
  const Vector7d coriolis = Vector7d::Constant(0.05);
  const auto terms = example_cartesian_impedance(p, q, synthetic_jacobian(0.0), coriolis,
                                                 q_joint, Vector7d::Zero(), p, q, q_joint,
                                                 stiffness, damping, 20.0);
  EXPECT_TRUE(terms.error.isZero(1e-15));
  EXPECT_TRUE(terms.tau_task.isZero(1e-13));
  EXPECT_TRUE(terms.tau_nullspace.isZero(1e-13));
  EXPECT_TRUE(terms.tau_command.isApprox(coriolis, 1e-13));
}

TEST(ExampleCartesianImpedance, error_is_invariant_to_quaternion_sign) {
  const Eigen::Vector3d p(0.4, 0.1, 0.5);
  const Eigen::Quaterniond q(Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitY()));
  const Eigen::Quaterniond q_d(Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitY()));
  Eigen::Quaterniond minus_q = q;
  minus_q.coeffs() = -minus_q.coeffs();
  Eigen::Quaterniond minus_q_d = q_d;
  minus_q_d.coeffs() = -minus_q_d.coeffs();
  const Vector6d reference = example_cartesian_error(p, q, p, q_d);
  EXPECT_TRUE(example_cartesian_error(p, minus_q, p, q_d).isApprox(reference, 1e-14));
  EXPECT_TRUE(example_cartesian_error(p, q, p, minus_q_d).isApprox(reference, 1e-14));
  EXPECT_TRUE(example_cartesian_error(p, minus_q, p, minus_q_d).isApprox(reference, 1e-14));
}

TEST(ExampleCartesianImpedance, torque_acts_to_reduce_a_rotation_error) {
  // Measured orientation lags the reference by a small rotation about base z. With the
  // identity Jacobian rows for the angular part, the z torque must be positive.
  const Eigen::Vector3d p(0.4, 0.0, 0.5);
  const Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  const Eigen::Quaterniond q_d(Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitZ()));
  Matrix6x7d jacobian = Matrix6x7d::Zero();
  for (int i = 0; i < 6; ++i) {
    jacobian(i, i) = 1.0;
  }
  Matrix6d stiffness, damping;
  example_cartesian_gains({150.0, 150.0, 150.0, 10.0, 10.0, 10.0}, stiffness, damping);
  const auto terms = example_cartesian_impedance(p, q, jacobian, Vector7d::Zero(),
                                                 Vector7d::Zero(), Vector7d::Zero(), p, q_d,
                                                 Vector7d::Zero(), stiffness, damping, 0.0);
  // The error vector is the rotation *from* reference *to* measured, so it is negative here
  // and -K * error is a positive torque about z.
  EXPECT_LT(terms.error(5), 0.0);
  EXPECT_NEAR(terms.error(5), -std::sin(0.05 / 2.0), 1e-12);
  EXPECT_GT(terms.tau_task(5), 0.0);
  EXPECT_NEAR(terms.tau_task(5), 10.0 * std::sin(0.05 / 2.0), 1e-12);
  // A position error along x pulls back along x with the translational stiffness.
  const auto shifted = example_cartesian_impedance(p + Eigen::Vector3d(0.01, 0, 0), q, jacobian,
                                                   Vector7d::Zero(), Vector7d::Zero(),
                                                   Vector7d::Zero(), p, q, Vector7d::Zero(),
                                                   stiffness, damping, 0.0);
  EXPECT_NEAR(shifted.tau_task(0), -150.0 * 0.01, 1e-12);
}

TEST(ExampleCartesianImpedance, matches_upstream_update_over_a_moving_sequence) {
  std::mt19937 generator(7);
  std::uniform_real_distribution<double> noise(-1.0, 1.0);

  UpstreamExample upstream;
  example_cartesian_gains({150.0, 150.0, 150.0, 10.0, 10.0, 10.0}, upstream.cartesian_stiffness_,
                          upstream.cartesian_damping_);
  upstream.nullspace_stiffness_ = 20.0;
  upstream.position_d_ = Eigen::Vector3d(0.45, 0.05, 0.40);
  upstream.orientation_d_ = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
  upstream.q_d_nullspace_ << 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785;

  Matrix6d stiffness = upstream.cartesian_stiffness_;
  Matrix6d damping = upstream.cartesian_damping_;
  double nullspace_stiffness = upstream.nullspace_stiffness_;
  Eigen::Vector3d position_d = upstream.position_d_;
  Eigen::Quaterniond orientation_d = upstream.orientation_d_;
  const Vector7d q_nullspace = upstream.q_d_nullspace_;

  // A live gain change part-way through, filtered on both sides.
  Matrix6d stiffness_target = stiffness;
  Matrix6d damping_target = damping;
  double nullspace_target = nullspace_stiffness;

  for (int cycle = 0; cycle < 5000; ++cycle) {
    const double t = cycle * 1e-3;
    if (cycle == 2000) {
      example_cartesian_gains({300.0, 300.0, 300.0, 20.0, 20.0, 20.0}, stiffness_target,
                              damping_target);
      nullspace_target = 35.0;
    }
    // Reference: a slow arc; measurement: the reference plus a lagging error and noise.
    const Eigen::Vector3d target_position =
        upstream.position_d_ + Eigen::Vector3d(0.1 * std::sin(t), 0.05 * std::cos(0.7 * t), 0.0);
    const Eigen::Quaterniond target_orientation =
        Eigen::Quaterniond(Eigen::AngleAxisd(0.3 * std::sin(0.5 * t), Eigen::Vector3d::UnitZ())) *
        Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    Eigen::Vector3d position = target_position + Eigen::Vector3d(0.01 * noise(generator),
                                                                 0.01 * noise(generator),
                                                                 0.01 * noise(generator));
    Eigen::Quaterniond orientation =
        Eigen::Quaterniond(Eigen::AngleAxisd(0.05 * noise(generator), Eigen::Vector3d::UnitY())) *
        target_orientation;
    if (cycle % 3 == 0) {
      orientation.coeffs() = -orientation.coeffs();  // hemisphere flips must not matter
    }
    Vector7d q, dq, coriolis;
    for (int joint = 0; joint < 7; ++joint) {
      q(joint) = q_nullspace(joint) + 0.2 * std::sin(t + joint);
      dq(joint) = 0.3 * std::cos(2 * t + joint);
      coriolis(joint) = 0.1 * noise(generator);
    }
    const Matrix6x7d jacobian = synthetic_jacobian(0.1 * t);
    std::array<double, 42> jacobian_array{};
    Eigen::Map<Eigen::Matrix<double, 6, 7>>(jacobian_array.data()) = jacobian;
    std::array<double, 7> coriolis_array{};
    Eigen::Map<Vector7d>(coriolis_array.data()) = coriolis;

    const Vector7d expected = upstream.update(position, orientation, jacobian_array,
                                              coriolis_array, q, dq, target_position,
                                              target_orientation, stiffness_target,
                                              damping_target, nullspace_target);

    const auto terms = example_cartesian_impedance(position, orientation, jacobian, coriolis, q,
                                                   dq, position_d, orientation_d, q_nullspace,
                                                   stiffness, damping, nullspace_stiffness);
    example_gain_filter(0.005, stiffness_target, damping_target, nullspace_target, stiffness,
                        damping, nullspace_stiffness);
    example_reference_filter(0.005, target_position, target_orientation, position_d,
                             orientation_d);

    ASSERT_TRUE(terms.tau_command.isApprox(expected, 1e-10))
        << "cycle " << cycle << "\nexpected " << expected.transpose() << "\nactual   "
        << terms.tau_command.transpose();
    ASSERT_TRUE(position_d.isApprox(upstream.position_d_, 1e-12)) << cycle;
    ASSERT_NEAR(std::abs(orientation_d.coeffs().dot(upstream.orientation_d_.coeffs())), 1.0, 1e-12)
        << cycle;
  }
}

TEST(ExampleCartesianImpedance, reference_filter_converges_and_bypasses_at_alpha_one) {
  Eigen::Vector3d position_d(0.0, 0.0, 0.0);
  Eigen::Quaterniond orientation_d = Eigen::Quaterniond::Identity();
  const Eigen::Vector3d target(0.1, 0.0, 0.0);
  const Eigen::Quaterniond target_orientation(Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()));
  for (int i = 0; i < 200; ++i) {
    example_reference_filter(0.005, target, target_orientation, position_d, orientation_d);
  }
  // One time constant (200 cycles at alpha 0.005): 1 - 1/e of the way.
  EXPECT_NEAR(position_d.x(), 0.1 * (1.0 - std::pow(0.995, 200)), 1e-12);
  EXPECT_LT(franka_trajectory_replay::quaternion_angle(orientation_d, target_orientation),
            0.5 * std::pow(0.995, 200) + 1e-9);
  example_reference_filter(1.0, target, target_orientation, position_d, orientation_d);
  EXPECT_TRUE(position_d.isApprox(target, 1e-15));
  EXPECT_NEAR(franka_trajectory_replay::quaternion_angle(orientation_d, target_orientation), 0.0,
              1e-12);
}

// --- ForgeUltra's law -------------------------------------------------------------------
// The two options the policy profiles set so the student meets the law it was distilled
// against (forge_osc.compute_dof_torque / forge_ultra/tasks/utils/control.py).

TEST(ForgeCartesianImpedance, axis_angle_error_carries_the_full_angle) {
  const Eigen::Vector3d p(0.4, 0.0, 0.3);
  const Eigen::Vector3d axis = Eigen::Vector3d(0.3, -0.5, 0.81).normalized();
  const Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  for (const double angle : {0.02, 0.1, 0.3, 0.6, 1.2}) {
    const Eigen::Quaterniond q_d(Eigen::AngleAxisd(angle, axis));
    const Vector6d example = example_cartesian_error(p, q, p, q_d);
    const Vector6d forge =
        example_cartesian_error(p, q, p, q_d, RotationErrorForm::kAxisAngle);
    // Same axis, different magnitude: the example applies sin(theta/2), training theta. That
    // is the factor that turns a 28 Nm/rad rotational spring into an effective 14 Nm/rad.
    EXPECT_NEAR(example.tail(3).norm(), std::sin(0.5 * angle), 1e-12) << "angle " << angle;
    EXPECT_NEAR(forge.tail(3).norm(), angle, 1e-12) << "angle " << angle;
    EXPECT_NEAR(forge.tail(3).normalized().dot(example.tail(3).normalized()), 1.0, 1e-12);
    EXPECT_TRUE(forge.head(3).isApprox(example.head(3), 1e-15));
  }
}

TEST(ForgeCartesianImpedance, lambda_zero_keeps_the_nullspace_torque_out_of_the_task_space) {
  Matrix6d stiffness;
  Matrix6d damping;
  example_cartesian_gains({565.0, 565.0, 565.0, 28.0, 28.0, 28.0}, stiffness, damping);
  const Eigen::Vector3d p(0.45, 0.02, 0.32);
  const Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  Vector7d joints;
  joints << 0.1, -0.5, 0.0, -2.0, 0.9, 2.4, -0.7;
  Vector7d q_nullspace;
  q_nullspace << 0.0, -0.6, 0.0, -2.6, -0.5, 2.9, 0.0;

  for (double phase = 0.0; phase < 3.0; phase += 0.37) {
    const Matrix6x7d jacobian = synthetic_jacobian(phase);
    // How much of the nullspace torque a task-space wrench could account for: the residual
    // after removing the component orthogonal to range(J^T).
    const auto leak = [&jacobian](const Vector7d& tau) {
      return (jacobian.transpose() *
              jacobian.transpose().completeOrthogonalDecomposition().solve(tau))
          .norm();
    };
    const auto terms = [&](double lambda) {
      return example_cartesian_impedance(p, q, jacobian, Vector7d::Zero(), joints,
                                        Vector7d::Zero(), p, q, q_nullspace, stiffness, damping,
                                        10.0, lambda);
    };
    const double leak_example = leak(terms(0.2).tau_nullspace);
    const double leak_exact = leak(terms(0.0).tau_nullspace);
    EXPECT_LT(leak_exact, 1e-9) << "phase " << phase;
    EXPECT_GT(leak_example, 20.0 * std::max(leak_exact, 1e-12)) << "phase " << phase;
  }
}

TEST(ForgeNullspace, matches_compute_dof_torque) {
  const Matrix6x7d jacobian = synthetic_jacobian(0.3);
  const Matrix7d mass = synthetic_mass_matrix(0.2);
  Vector7d q, dq, q_null;
  q << 0.2, -0.4, 0.1, -2.1, 1.7, 2.0, -1.0;
  dq << 0.05, -0.1, 0.02, 0.3, -0.2, 0.1, 0.4;
  q_null << 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785;
  const double kp_null = 10.0;
  const double kd_null = 6.3246;  // vanilla_threading.yaml; 2 sqrt(10) to four places
  const Vector7d actual =
      forge_nullspace_torque(jacobian, mass, q, dq, q_null, kp_null, kd_null);
  const Vector7d expected =
      forge_reference_nullspace(jacobian, mass, q, dq, q_null, kp_null, kd_null);
  EXPECT_TRUE(actual.isApprox(expected, 1e-12)) << actual.transpose() << " vs " << expected.transpose();
}

TEST(ForgeNullspace, is_dynamically_consistent_so_it_cannot_push_on_the_tool) {
  // The point of the mass-weighted projector: mapping tau_null back through the dynamically
  // consistent inverse gives exactly no task wrench, at any posture. The example's damped
  // pseudo-inverse does not (see kNullspaceDampingLambda).
  Vector7d q, dq, q_null;
  q << 0.2, -0.4, 0.1, -2.1, 1.7, 2.0, -1.0;
  dq << 0.05, -0.1, 0.02, 0.3, -0.2, 0.1, 0.4;
  q_null << 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785;
  for (double phase = 0.0; phase < 3.0; phase += 0.5) {
    const Matrix6x7d jacobian = synthetic_jacobian(phase);
    const Matrix7d mass = synthetic_mass_matrix(0.4 * phase);
    const Vector7d tau_null =
        forge_nullspace_torque(jacobian, mass, q, dq, q_null, 10.0, 6.3246);
    const Matrix7d mass_inverse = mass.inverse();
    const Matrix6d mass_task = (jacobian * mass_inverse * jacobian.transpose()).inverse();
    const Matrix6x7d jacobian_eef_inverse = mass_task * jacobian * mass_inverse;
    const Vector6d wrench = jacobian_eef_inverse * tau_null;
    EXPECT_LT(wrench.norm(), 1e-9) << "phase " << phase << " leaked " << wrench.transpose();
  }
}

TEST(ForgeNullspace, reduces_to_the_exact_projector_when_the_mass_matrix_is_the_identity) {
  // With M = I the dynamically consistent inverse becomes the Moore-Penrose pseudo-inverse,
  // so Forge's term must agree with the example's at nullspace_damping_lambda = 0 -- the A1a
  // state. Any disagreement would mean the lambda-0 path is not the exact projector.
  const Eigen::Vector3d p(0.4, 0.1, 0.5);
  const Eigen::Quaterniond orientation(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()));
  const Eigen::Quaterniond orientation_d(Eigen::AngleAxisd(0.35, Eigen::Vector3d::UnitY()));
  Matrix6d stiffness, damping;
  example_cartesian_gains({565.0, 565.0, 565.0, 28.0, 28.0, 28.0}, stiffness, damping);
  Vector7d q, dq, q_null;
  q << 0.2, -0.4, 0.1, -2.1, 1.7, 2.0, -1.0;
  dq << 0.05, -0.1, 0.02, 0.3, -0.2, 0.1, 0.4;
  q_null << 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785;
  const Matrix6x7d jacobian = synthetic_jacobian(0.9);
  const Matrix7d identity = Matrix7d::Identity();
  const auto exact_pinv = example_cartesian_impedance(
      p, orientation, jacobian, Vector7d::Zero(), q, dq, p, orientation_d, q_null, stiffness,
      damping, 10.0, 0.0, RotationErrorForm::kAxisAngle);
  const auto forge = example_cartesian_impedance(
      p, orientation, jacobian, Vector7d::Zero(), q, dq, p, orientation_d, q_null, stiffness,
      damping, 10.0, 0.0, RotationErrorForm::kAxisAngle, &identity);
  EXPECT_TRUE(forge.tau_nullspace.isApprox(exact_pinv.tau_nullspace, 1e-9))
      << forge.tau_nullspace.transpose() << " vs " << exact_pinv.tau_nullspace.transpose();
  EXPECT_TRUE(forge.tau_task.isApprox(exact_pinv.tau_task, 1e-12));
}

TEST(ForgeNullspace, mass_weighting_changes_the_term_at_a_realistic_mass_matrix) {
  // Guard against the mass matrix being accepted and then ignored.
  const Matrix6x7d jacobian = synthetic_jacobian(0.9);
  Vector7d q, dq, q_null;
  q << 0.2, -0.4, 0.1, -2.1, 1.7, 2.0, -1.0;
  dq << 0.05, -0.1, 0.02, 0.3, -0.2, 0.1, 0.4;
  q_null << 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785;
  const Vector7d weighted =
      forge_nullspace_torque(jacobian, synthetic_mass_matrix(0.2), q, dq, q_null, 10.0, 6.3246);
  const Vector7d unweighted =
      forge_nullspace_torque(jacobian, Matrix7d::Identity(), q, dq, q_null, 10.0, 6.3246);
  EXPECT_GT((weighted - unweighted).norm(), 1e-3);
}

TEST(ForgeNullspace, wraps_the_joint_distance_to_pi) {
  const Matrix6x7d jacobian = synthetic_jacobian(0.1);
  const Matrix7d mass = synthetic_mass_matrix(0.0);
  const Vector7d q = Vector7d::Zero();
  Vector7d q_null = Vector7d::Zero();
  q_null(0) = 1.5 * M_PI;  // wraps to -0.5 pi
  const Vector7d wrapped =
      forge_nullspace_torque(jacobian, mass, q, Vector7d::Zero(), q_null, 10.0, 6.3246);
  Vector7d q_equivalent = Vector7d::Zero();
  q_equivalent(0) = -0.5 * M_PI;
  const Vector7d direct =
      forge_nullspace_torque(jacobian, mass, q, Vector7d::Zero(), q_equivalent, 10.0, 6.3246);
  EXPECT_TRUE(wrapped.isApprox(direct, 1e-12));
}
