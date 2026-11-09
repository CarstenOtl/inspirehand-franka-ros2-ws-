// Copyright (c) 2026 Agile Robots SE
// Licensed under the Apache License, Version 2.0.
// See http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <mujoco/mujoco.h>

#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp>

namespace inspire_franka_sim
{

/// The three parts of Isaac Lab's Inspire hand that MJCF cannot express.
///
/// inspire_hand_right_isaac.xml (make_hand_mjcf.py with the isaac flag) carries
/// Isaac's drives, armature and limits. What is left needs code, and runs here
/// on the physics thread immediately before every mj_step:
///
/// - Target-level mimic. Isaac strips the URDF's mimic tags and drives every
///   follower to multiplier * its leader's TARGET (`_expand_hand_mimic_targets`),
///   so a follower's actuator ctrl is set from its leader's ctrl. Chains (the
///   thumb's distal joint follows the intermediate) resolve in declaration order.
/// - The joint velocity limit (1 rad/s, PhysX maxJointVelocity). PhysX clamps
///   the velocity in the solve and integrates the position with the clamped
///   value. Here the step that just ran integrated qpos with the unclamped
///   qvel, so the excess displacement (qvel - clamp(qvel)) * dt is taken back
///   out of qpos as the velocity is clamped. That is exact for MuJoCo's Euler
///   and implicitfast integrators, which advance hinge positions by the new
///   velocity; it is wrong for RK4, which init() refuses. There is no post-step
///   hook, so the state mujoco_ros2_control publishes right after a step still
///   carries that step's excess: up to (qvel - limit) * dt of position (about
///   0.03 rad while a 20 N m pinch drive accelerates a free finger) and the
///   unclamped velocity, both gone at the next step.
/// - PhysX joint friction: the resisting effort is bounded by
///   coefficient * (|F| + |T|) of the spatial force the joint transmits from
///   parent to child (PxArticulationJointReducedCoordinate::setFrictionCoefficient).
///   MuJoCo's frictionloss is the same dry-friction bound with a fixed value, so
///   it is recomputed each step from the previous step's interaction force
///   (mj_rnePostConstraint's cfrc_int, moved to the joint anchor).
///
/// Configuration is read from the scene's <custom> block, written by
/// make_ros_sim_scene.py from make_hand_mjcf.py's ISAAC_* tables:
///
///   numeric isaac_hand/velocity_limit         [rad/s]
///   numeric isaac_hand/joint/<joint>          [friction coefficient]
///   numeric isaac_hand/mimic/<follower>       [multiplier, offset]
///   text    isaac_hand/mimic_leader/<follower> <leader joint>
///
/// A scene without them leaves the plugin idle.
class IsaacHandPlugin : public mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase
{
public:
  IsaacHandPlugin() = default;
  ~IsaacHandPlugin() override = default;

  bool init(rclcpp::Node::SharedPtr node, const mjModel * model, mjData * data) override;
  void pre_step(mjData * data) override;
  void cleanup() override;

private:
  struct Joint
  {
    std::string name;
    int id;
    int qpos;
    int dof;
    int body;
    mjtNum friction_coefficient;
  };

  struct Mimic
  {
    int follower_actuator;
    int leader_qpos;
    mjtNum multiplier;
    mjtNum offset;
  };

  /// The actuator transmitting to joint `joint`, or -1.
  static int actuator_for_joint(const mjModel * model, int joint);

  const mjModel * model_{nullptr};
  rclcpp::Logger logger_{rclcpp::get_logger("IsaacHandPlugin")};
  bool present_{false};
  mjtNum velocity_limit_{0.0};
  std::vector<Joint> joints_;
  std::vector<Mimic> mimics_;
};

}  // namespace inspire_franka_sim
