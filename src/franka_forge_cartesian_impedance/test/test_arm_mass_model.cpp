// Copyright (c) 2026 Agile Robots SE
// Licensed under the Apache License, Version 2.0.
// See http://www.apache.org/licenses/LICENSE-2.0

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

#include <franka_forge_cartesian_impedance/arm_mass_model.hpp>

using franka_forge_cartesian_impedance::ArmMassModel;
using franka_forge_cartesian_impedance::Matrix7d;
using franka_forge_cartesian_impedance::Vector7d;

namespace {

// A seven-revolute serial chain, optionally with an extra movable "finger" link hanging off
// the last body. The point of the finger is that the controller never sees its joint, so the
// model must lock it and keep its inertia -- which is how the real hand reaches the arm block.
std::string chain_urdf(bool with_finger, double finger_mass = 0.4) {
  std::ostringstream urdf;
  urdf << "<robot name=\"chain\">";
  urdf << "<link name=\"base\"/>";
  for (int i = 1; i <= 7; ++i) {
    urdf << "<link name=\"link" << i << "\">"
         << "<inertial><origin xyz=\"0.1 0 0\"/><mass value=\"" << (3.0 - 0.2 * i) << "\"/>"
         << "<inertia ixx=\"0.02\" ixy=\"0\" ixz=\"0\" iyy=\"0.03\" iyz=\"0\" izz=\"0.04\"/>"
         << "</inertial></link>";
    urdf << "<joint name=\"joint" << i << "\" type=\"revolute\">"
         << "<parent link=\"" << (i == 1 ? std::string("base") : "link" + std::to_string(i - 1))
         << "\"/><child link=\"link" << i << "\"/>"
         << "<origin xyz=\"0.2 0 0.05\" rpy=\"0 0 0\"/>"
         << "<axis xyz=\"" << (i % 2 == 0 ? "0 1 0" : "0 0 1") << "\"/>"
         << "<limit lower=\"-2.8\" upper=\"2.8\" effort=\"80\" velocity=\"2\"/>"
         << "</joint>";
  }
  if (with_finger) {
    urdf << "<link name=\"finger\"><inertial><origin xyz=\"0.05 0 0\"/><mass value=\""
         << finger_mass << "\"/>"
         << "<inertia ixx=\"0.001\" ixy=\"0\" ixz=\"0\" iyy=\"0.001\" iyz=\"0\" izz=\"0.001\"/>"
         << "</inertial></link>";
    urdf << "<joint name=\"finger_joint\" type=\"revolute\">"
         << "<parent link=\"link7\"/><child link=\"finger\"/>"
         << "<origin xyz=\"0.08 0 0\" rpy=\"0 0 0\"/><axis xyz=\"0 1 0\"/>"
         << "<limit lower=\"0\" upper=\"1.5\" effort=\"5\" velocity=\"2\"/></joint>";
  }
  urdf << "</robot>";
  return urdf.str();
}

std::vector<std::string> arm_joints() {
  std::vector<std::string> names;
  for (int i = 1; i <= 7; ++i) {
    names.push_back("joint" + std::to_string(i));
  }
  return names;
}

Vector7d posture() {
  Vector7d q;
  q << 0.2, -0.4, 0.1, -2.1, 1.7, 2.0, -1.0;
  return q;
}

}  // namespace

TEST(ArmMassModel, builds_from_a_urdf_and_returns_a_symmetric_positive_definite_matrix) {
  ArmMassModel model;
  std::string error;
  ASSERT_TRUE(model.load(chain_urdf(true), arm_joints(), error)) << error;
  EXPECT_TRUE(model.loaded());

  Matrix7d mass;
  ASSERT_TRUE(model.compute(posture(), Vector7d::Zero(), mass));
  EXPECT_TRUE(mass.isApprox(mass.transpose(), 1e-12));
  const Eigen::LLT<Matrix7d> cholesky(mass);
  EXPECT_EQ(cholesky.info(), Eigen::Success) << "mass matrix is not positive definite";
  EXPECT_GT(mass.diagonal().minCoeff(), 0.0);
}

TEST(ArmMassModel, a_locked_joints_inertia_still_reaches_the_arm_block) {
  // The hand has no state interface on this controller, so its joints get locked. Locking
  // must fold the inertia in, not discard it -- otherwise the arm block is the one measured
  // at 10 to 14 % low against the training asset.
  ArmMassModel with_finger, without_finger;
  std::string error;
  ASSERT_TRUE(with_finger.load(chain_urdf(true), arm_joints(), error)) << error;
  ASSERT_TRUE(without_finger.load(chain_urdf(false), arm_joints(), error)) << error;

  Matrix7d heavy, light;
  ASSERT_TRUE(with_finger.compute(posture(), Vector7d::Zero(), heavy));
  ASSERT_TRUE(without_finger.compute(posture(), Vector7d::Zero(), light));
  EXPECT_GT((heavy - light).cwiseAbs().maxCoeff(), 1e-4)
      << "locking the finger dropped its inertia";
  // A heavier hand can only add inertia about the joints that carry it.
  EXPECT_GT(heavy(6, 6), light(6, 6));
}

TEST(ArmMassModel, armature_lands_on_the_diagonal_only) {
  ArmMassModel model;
  std::string error;
  ASSERT_TRUE(model.load(chain_urdf(true), arm_joints(), error)) << error;

  Vector7d armature;
  armature << 0.195, 0.195, 0.195, 0.195, 0.074, 0.074, 0.074;
  Matrix7d rigid, with_armature;
  ASSERT_TRUE(model.compute(posture(), Vector7d::Zero(), rigid));
  ASSERT_TRUE(model.compute(posture(), armature, with_armature));
  const Matrix7d difference = with_armature - rigid;
  EXPECT_TRUE(difference.isApprox(Matrix7d(armature.asDiagonal()), 1e-12));
}

TEST(ArmMassModel, rows_and_columns_follow_the_joint_order_the_caller_asked_for) {
  // The controller's joint order comes from its parameters, not from the URDF tree, so a
  // permuted request has to come back as the permuted matrix rather than a transposed one.
  std::vector<std::string> permuted = arm_joints();
  std::swap(permuted[1], permuted[5]);

  ArmMassModel natural, swapped;
  std::string error;
  ASSERT_TRUE(natural.load(chain_urdf(true), arm_joints(), error)) << error;
  ASSERT_TRUE(swapped.load(chain_urdf(true), permuted, error)) << error;

  Vector7d q = posture();
  Vector7d q_permuted = q;
  std::swap(q_permuted(1), q_permuted(5));

  Matrix7d expected, actual;
  ASSERT_TRUE(natural.compute(q, Vector7d::Zero(), expected));
  ASSERT_TRUE(swapped.compute(q_permuted, Vector7d::Zero(), actual));

  for (int row = 0; row < 7; ++row) {
    for (int column = 0; column < 7; ++column) {
      const int r = row == 1 ? 5 : (row == 5 ? 1 : row);
      const int c = column == 1 ? 5 : (column == 5 ? 1 : column);
      EXPECT_NEAR(actual(row, column), expected(r, c), 1e-12)
          << "entry (" << row << ", " << column << ")";
    }
  }
}

TEST(ArmMassModel, rejects_a_description_it_cannot_use) {
  ArmMassModel model;
  std::string error;

  EXPECT_FALSE(model.load("not xml at all", arm_joints(), error));
  EXPECT_FALSE(error.empty());
  EXPECT_FALSE(model.loaded());

  std::vector<std::string> missing = arm_joints();
  missing[3] = "fr3_joint_that_does_not_exist";
  EXPECT_FALSE(model.load(chain_urdf(true), missing, error));
  EXPECT_NE(error.find("fr3_joint_that_does_not_exist"), std::string::npos) << error;

  EXPECT_FALSE(model.load(chain_urdf(true), {"joint1", "joint2"}, error));
  EXPECT_FALSE(error.empty());

  // A failed load must leave nothing behind for update() to use.
  Matrix7d mass;
  EXPECT_FALSE(model.compute(posture(), Vector7d::Zero(), mass));
}
