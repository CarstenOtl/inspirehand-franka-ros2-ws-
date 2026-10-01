// Copyright (c) 2026 Agile Robots SE
// Licensed under the Apache License, Version 2.0.
// See http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <franka_trajectory_replay/cartesian_impedance.hpp>

namespace franka_trajectory_replay {

/// The arm block of the whole articulation's generalized mass matrix, for Forge's
/// mass-weighted nullspace term (`forge_nullspace_torque`).
///
/// Isaac takes that block from PhysX over the full arm+hand articulation, so the hand's
/// inertia has to be in it. This builds the same thing from the URDF the controller is
/// handed at configure time: a rigid-body model of the whole tree with every non-arm joint
/// locked, whose composite-rigid-body mass matrix is then exactly the 7x7 arm block.
/// Verified against MuJoCo on 2026-10-01: with the hand bolted to the flange
/// (`hand_mount:=flange`) the two agree to 9.1e-07 kg m^2 on the ros-sim plant, and locking
/// the hand reproduces the full tree's arm block to 1.4e-17.
///
/// Locking the hand costs almost nothing: sweeping every finger joint from limit to limit
/// moves the arm block by at most 1.2e-03 kg m^2 (0.06 %), against the 0.125 kg m^2 (6.5 %)
/// by which the training asset's hand model and this one already disagree. The controller
/// has no hand state interfaces, so a live hand configuration is not available to it anyway.
///
/// `compute()` is allocation-free once `load()` has succeeded, so it is safe in update().
class ArmMassModel {
 public:
  ArmMassModel();
  ~ArmMassModel();
  ArmMassModel(ArmMassModel&&) noexcept;
  ArmMassModel& operator=(ArmMassModel&&) noexcept;
  ArmMassModel(const ArmMassModel&) = delete;
  ArmMassModel& operator=(const ArmMassModel&) = delete;

  /// Builds the model from a URDF string. `arm_joints` are the seven joints to keep, in the
  /// order the controller's state interfaces use; every other movable joint is locked at
  /// zero. Returns false and fills `error` if the URDF does not parse or a joint is missing.
  bool load(const std::string& urdf, const std::vector<std::string>& arm_joints,
            std::string& error);

  bool loaded() const;

  /// Composite-rigid-body mass matrix at `q`, plus `armature` on the diagonal. Returns false
  /// if the model was never loaded.
  bool compute(const Vector7d& q, const Vector7d& armature, Matrix7d& mass) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace franka_trajectory_replay
