// Copyright (c) 2026 Agile Robots SE
// Licensed under the Apache License, Version 2.0.
// See http://www.apache.org/licenses/LICENSE-2.0

#include <franka_forge_cartesian_impedance/arm_mass_model.hpp>

#include <algorithm>
#include <array>
#include <sstream>

#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace franka_forge_cartesian_impedance {

struct ArmMassModel::Impl {
  pinocchio::Model model;
  mutable pinocchio::Data data;
  // Column in `model`'s configuration/velocity vectors for each of the seven arm joints, in
  // the controller's joint order. buildReducedModel keeps the surviving joints in the
  // original tree order, which is the FR3's 1..7, but the mapping is read out rather than
  // assumed so a renamed or reordered description cannot silently transpose the matrix.
  std::array<int, 7> configuration_index{};
  std::array<int, 7> velocity_index{};
  mutable Eigen::VectorXd configuration;
};

ArmMassModel::ArmMassModel() = default;
ArmMassModel::~ArmMassModel() = default;
ArmMassModel::ArmMassModel(ArmMassModel&&) noexcept = default;
ArmMassModel& ArmMassModel::operator=(ArmMassModel&&) noexcept = default;

bool ArmMassModel::load(const std::string& urdf, const std::vector<std::string>& arm_joints,
                        std::string& error) {
  impl_.reset();
  if (arm_joints.size() != 7) {
    error = "arm_joints must name seven joints";
    return false;
  }

  pinocchio::Model full;
  try {
    pinocchio::urdf::buildModelFromXML(urdf, full);
  } catch (const std::exception& exception) {
    error = std::string("could not parse the robot description: ") + exception.what();
    return false;
  }

  for (const auto& name : arm_joints) {
    if (!full.existJointName(name)) {
      error = "the robot description has no joint '" + name + "'";
      return false;
    }
  }

  // Lock every movable joint that is not one of the seven. Their inertia is folded into the
  // bodies they are attached to, so the hand keeps contributing to the arm block.
  std::vector<pinocchio::JointIndex> locked;
  for (pinocchio::JointIndex index = 1; index < full.joints.size(); ++index) {
    const std::string& name = full.names[index];
    if (std::find(arm_joints.begin(), arm_joints.end(), name) == arm_joints.end()) {
      locked.push_back(index);
    }
  }

  auto impl = std::make_unique<Impl>();
  try {
    const Eigen::VectorXd reference = pinocchio::neutral(full);
    impl->model = pinocchio::buildReducedModel(full, locked, reference);
  } catch (const std::exception& exception) {
    error = std::string("could not reduce the robot description: ") + exception.what();
    return false;
  }

  if (impl->model.nv != 7 || impl->model.nq != 7) {
    std::ostringstream message;
    message << "the reduced model has " << impl->model.nq << " positions and " << impl->model.nv
            << " velocities, expected 7 and 7; are the arm joints all revolute?";
    error = message.str();
    return false;
  }

  for (std::size_t i = 0; i < arm_joints.size(); ++i) {
    const auto joint = impl->model.getJointId(arm_joints[i]);
    impl->configuration_index[i] = impl->model.idx_qs[joint];
    impl->velocity_index[i] = impl->model.idx_vs[joint];
  }

  impl->data = pinocchio::Data(impl->model);
  impl->configuration = pinocchio::neutral(impl->model);
  impl_ = std::move(impl);
  return true;
}

bool ArmMassModel::loaded() const { return impl_ != nullptr; }

bool ArmMassModel::compute(const Vector7d& q, const Vector7d& armature, Matrix7d& mass) const {
  if (impl_ == nullptr) {
    return false;
  }
  for (int i = 0; i < 7; ++i) {
    impl_->configuration(impl_->configuration_index[i]) = q(i);
  }
  pinocchio::crba(impl_->model, impl_->data, impl_->configuration);
  // crba fills the upper triangle only.
  for (int row = 0; row < 7; ++row) {
    for (int column = 0; column < 7; ++column) {
      const int i = impl_->velocity_index[row];
      const int j = impl_->velocity_index[column];
      mass(row, column) = j >= i ? impl_->data.M(i, j) : impl_->data.M(j, i);
    }
  }
  for (int i = 0; i < 7; ++i) {
    mass(i, i) += armature(i);
  }
  return true;
}

}  // namespace franka_forge_cartesian_impedance
