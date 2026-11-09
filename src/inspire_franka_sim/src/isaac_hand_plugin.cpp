// Copyright (c) 2026 Agile Robots SE
// Licensed under the Apache License, Version 2.0.
// See http://www.apache.org/licenses/LICENSE-2.0

#include "inspire_franka_sim/isaac_hand_plugin.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

#include <pluginlib/class_list_macros.hpp>

namespace
{

constexpr char kPrefix[] = "isaac_hand/";
constexpr char kVelocityLimit[] = "isaac_hand/velocity_limit";
constexpr char kJoint[] = "isaac_hand/joint/";
constexpr char kMimic[] = "isaac_hand/mimic/";
constexpr char kMimicLeader[] = "isaac_hand/mimic_leader/";

bool starts_with(const std::string & text, const char * prefix)
{
  return text.rfind(prefix, 0) == 0;
}

mjtNum norm3(const mjtNum * v)
{
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

}  // namespace

namespace inspire_franka_sim
{

int IsaacHandPlugin::actuator_for_joint(const mjModel * model, int joint)
{
  for (int i = 0; i < model->nu; ++i) {
    if (model->actuator_trntype[i] == mjTRN_JOINT && model->actuator_trnid[2 * i] == joint) {
      return i;
    }
  }
  return -1;
}

bool IsaacHandPlugin::init(rclcpp::Node::SharedPtr node, const mjModel * model, mjData *)
{
  logger_ = node->get_logger().get_child("isaac_hand");
  model_ = model;

  const int limit = mj_name2id(model, mjOBJ_NUMERIC, kVelocityLimit);
  if (limit < 0) {
    RCLCPP_INFO(logger_, "no %s in this scene; the plugin will stay idle", kVelocityLimit);
    return true;
  }
  if (model->opt.integrator == mjINT_RK4) {
    RCLCPP_ERROR(
      logger_, "the velocity limit's position correction assumes Euler or implicitfast; "
      "this scene integrates with RK4");
    return false;
  }
  velocity_limit_ = model->numeric_data[model->numeric_adr[limit]];

  for (int i = 0; i < model->nnumeric; ++i) {
    const char * raw = mj_id2name(model, mjOBJ_NUMERIC, i);
    const std::string name = raw ? raw : "";
    if (!starts_with(name, kPrefix) || name == kVelocityLimit) {
      continue;
    }
    const mjtNum * values = model->numeric_data + model->numeric_adr[i];
    const int size = model->numeric_size[i];
    if (starts_with(name, kJoint)) {
      const std::string joint = name.substr(sizeof(kJoint) - 1);
      const int id = mj_name2id(model, mjOBJ_JOINT, joint.c_str());
      if (id < 0 || model->jnt_type[id] != mjJNT_HINGE || size < 1) {
        RCLCPP_ERROR(logger_, "%s: no hinge joint '%s' to apply it to", name.c_str(), joint.c_str());
        return false;
      }
      joints_.push_back(
        {joint, id, model->jnt_qposadr[id], model->jnt_dofadr[id], model->jnt_bodyid[id], values[0]});
    } else if (starts_with(name, kMimic)) {
      const std::string follower = name.substr(sizeof(kMimic) - 1);
      const std::string leader_key = std::string(kMimicLeader) + follower;
      const int text = mj_name2id(model, mjOBJ_TEXT, leader_key.c_str());
      if (text < 0 || size < 2) {
        RCLCPP_ERROR(logger_, "%s needs [multiplier, offset] and a text %s", name.c_str(),
          leader_key.c_str());
        return false;
      }
      const std::string leader = model->text_data + model->text_adr[text];
      const int follower_actuator =
        actuator_for_joint(model, mj_name2id(model, mjOBJ_JOINT, follower.c_str()));
      const int leader_actuator =
        actuator_for_joint(model, mj_name2id(model, mjOBJ_JOINT, leader.c_str()));
      if (follower_actuator < 0 || leader_actuator < 0) {
        RCLCPP_ERROR(
          logger_, "%s: '%s' or its leader '%s' has no joint actuator", name.c_str(),
          follower.c_str(), leader.c_str());
        return false;
      }
      const int leader_joint = mj_name2id(model, mjOBJ_JOINT, leader.c_str());
      mimics_.push_back({follower_actuator, model->jnt_qposadr[leader_joint], values[0], values[1]});
    }
  }
  present_ = true;
  RCLCPP_INFO(
    logger_,
    "Isaac hand: %zu joints limited to %.3f rad/s with PhysX joint friction, %zu followers "
    "tracking their measured leader",
    joints_.size(), static_cast<double>(velocity_limit_), mimics_.size());
  return true;
}

void IsaacHandPlugin::pre_step(mjData * data)
{
  if (!present_) {
    return;
  }
  const mjModel * model = model_;
  const mjtNum dt = model->opt.timestep;

  // Velocity limit, as PhysX applies it: clamp, and integrate the position with
  // the clamped velocity rather than the one the last step used.
  for (const Joint & joint : joints_) {
    const mjtNum velocity = data->qvel[joint.dof];
    const mjtNum clamped = std::clamp(velocity, -velocity_limit_, velocity_limit_);
    if (clamped != velocity) {
      data->qpos[joint.qpos] -= (velocity - clamped) * dt;
      data->qvel[joint.dof] = clamped;
    }
  }

  // PhysX joint friction from the force each joint transmitted last step.
  // cfrc_int is the parent-on-child spatial force [torque; force], expressed in
  // world orientation about the tree's subtree centre of mass; the torque is
  // moved to the joint anchor before taking its magnitude.
  mj_rnePostConstraint(model, data);
  mjtNum * frictionloss = const_cast<mjtNum *>(model->dof_frictionloss);
  for (const Joint & joint : joints_) {
    const mjtNum * spatial = data->cfrc_int + 6 * static_cast<std::size_t>(joint.body);
    const mjtNum * force = spatial + 3;
    const mjtNum * centre = data->subtree_com + 3 * model->body_rootid[joint.body];
    const mjtNum * anchor = data->xanchor + 3 * joint.id;
    mjtNum lever[3] = {centre[0] - anchor[0], centre[1] - anchor[1], centre[2] - anchor[2]};
    mjtNum moment[3];
    mju_cross(moment, lever, force);
    mju_addTo3(moment, spatial);
    frictionloss[joint.dof] = joint.friction_coefficient * (norm3(force) + norm3(moment));
  }

  // The followers are a rigid linkage on the real hand: each one tracks its
  // leader's measured position. A leader stopped on the nut stops its followers.
  // MuJoCo clamps ctrl to each actuator's ctrlrange.
  for (const Mimic & mimic : mimics_) {
    data->ctrl[mimic.follower_actuator] =
      mimic.multiplier * data->qpos[mimic.leader_qpos] + mimic.offset;
  }
}

void IsaacHandPlugin::cleanup()
{
  present_ = false;
  joints_.clear();
  mimics_.clear();
  model_ = nullptr;
}

}  // namespace inspire_franka_sim

PLUGINLIB_EXPORT_CLASS(
  inspire_franka_sim::IsaacHandPlugin,
  mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
