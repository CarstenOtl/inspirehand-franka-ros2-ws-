"""MuJoCo port of ForgeUltra's FR3 M24 cyclic-threading scene and control path.

Ground truth is ``reference/forgeUltra`` (branch ``franka-chi``). This module
mirrors, in NumPy/MuJoCo:

- the corrected training scene (``mujoco_scene``) plus the simplified
  self-locking thread pair (``thread_pair.py`` / ``_update_solver_thread_state``):
  a virtual carrier with an axial slide driven to ``start - pitch*theta/2pi``
  by a 200 kN/m drive, and a passive twist hinge with tiny Coulomb/viscous
  resistance; nut/bolt contact is filtered;
- the robot without gravity (the training asset sets ``disable_gravity``),
  torque-driven arm joints with zero drive gains, PD hand joints with the
  threading gains (30/3, effort 20; inactive fingers 4.76/0.21, effort 1.864);
- the native OSC path at 120 Hz: filtered action -> clipped grasp-frame
  target -> operational-space torque -> joint-PD command (the deployment
  adapter) -> joint impedance torque;
- the cyclic release/return coordinator of ``forge_transitions.py``:
  release at 55 deg of directional turn progress, four 0.7 s waypoint phases,
  a 1.0 s return phase gated by the physical return check (5 mm, 5 deg,
  0.15 rad) with a 16 s timeout, thread hold during the transition, and the
  action-history seed at each cycle boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from . import forge_osc as fo
from .mujoco_scene import full_mass_matrix, training_scene_spec

ARM_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))
POLICY_JOINTS = (*ARM_JOINTS, *fo.PINCH_JOINTS)
HAND_POLICY_JOINTS = (
    "thumb_joint_0", "thumb_joint_1", "index_joint_0",
    "middle_joint_0", "ring_joint_0", "little_joint_0",
)
INACTIVE_FINGERS = ("middle_joint_0", "ring_joint_0", "little_joint_0")
INACTIVE_FINGER_TARGET = 1.333  # official-hand proximal upper limit
PHYSICS_DT = 1.0 / 120.0
DECIMATION = 8
POLICY_DT = PHYSICS_DT * DECIMATION
SUBSTEPS = 4

# Thread pair (vanilla_threading.yaml + env cfg defaults).
NUT_MASS_KG = 0.05
NUT_START_AXIAL_OFFSET_M = 0.00766
THREAD_UPPER_RELEASE_TURNS = 1.0
THREAD_HEAD_CLEARANCE_M = 0.0002
THREAD_AXIAL_KP = 200000.0
THREAD_AXIAL_KV = 200.0
THREAD_AXIAL_MAX_FORCE = 1000.0
THREAD_COULOMB_TORQUE = 0.002
THREAD_VISCOUS_DAMPING = 0.0002
THREADING_DIRECTION_SIGN = -1.0  # clockwise
THREAD_AXIAL_ARMATURE = 5.0      # kg, see _add_thread_pair
THREAD_TWIST_ARMATURE = 0.001    # kg m^2, keeps the hold equality stiff
CONTACT_FRICTION = 0.75

# Hand drives (ThreadingRobotControllerCfg).
PINCH_STIFFNESS, PINCH_DAMPING, PINCH_EFFORT = 30.0, 3.0, 20.0
INACTIVE_STIFFNESS, INACTIVE_DAMPING, INACTIVE_EFFORT = 4.76, 0.21, 1.864

# Cyclic release/return (waypoints_release_and_reset_franka.yaml).
RELEASE_MIN_TURN_DEG = 55.0
WAYPOINT_DURATION_S = 0.7
RETURN_DURATION_S = 1.0
RETURN_POSITION_TOLERANCE_M = 0.005
RETURN_ORIENTATION_TOLERANCE_DEG = 5.0
RETURN_HAND_TOLERANCE_RAD = 0.15
RETURN_TIMEOUT_S = 16.0
WAYPOINT_PHASES = 4



def _mimic_follower_targets(leaders: dict) -> dict:
    return fo.expand_hand_mimic(dict(leaders))


class ThreadingScene:
    """Compiled MuJoCo scene with the FR3, Inspire hand, camera, and threaded nut."""

    def __init__(self, *, nut_quat_wxyz, mjcf=None, camera=None):
        import mujoco

        self.mujoco = mujoco
        spec = training_scene_spec(mjcf)
        spec.option.timestep = PHYSICS_DT / SUBSTEPS
        self._configure_robot(spec)
        self._add_thread_pair(spec, nut_quat_wxyz)
        if camera is not None:
            camera(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.jid = {m.joint(j).name: j for j in range(m.njnt)}
        self.qadr = {n: m.jnt_qposadr[j] for n, j in self.jid.items()}
        self.vadr = {n: m.jnt_dofadr[j] for n, j in self.jid.items()}
        self.arm_dofs = [self.vadr[n] for n in ARM_JOINTS]
        self.aid = {m.actuator(a).name: a for a in range(m.nu)}
        self.flange = m.body("fr3_link8").id
        self.thumb_tip = m.body("thumb_tip").id
        self.index_tip = m.body("index_tip").id
        self.nut_body = m.body("m24_nut").id
        self.axial_dof = self.vadr["nut_axial"]
        self.twist_dof = self.vadr["nut_twist"]
        self.coupling_eq = m.equality("thread_coupling").id
        self.coupling_slope = float(m.eq_data[self.coupling_eq, 1])
        self.hold_eq = m.equality("thread_hold").id
        self.arm_ctrl_range = m.actuator_ctrlrange[[self.aid[n] for n in ARM_JOINTS]]
        self.sim_time = 0.0
        self.thread_hold: tuple[float, float] | None = None
        self.twist_reference = 0.0

    # --- scene construction ---------------------------------------------------
    @staticmethod
    def _configure_robot(spec) -> None:
        import mujoco

        scene_bodies = {"world", "table", "m24_bolt", "recorded_nut", "recorded_tcp_marker", "replayed_tcp_marker"}
        for body in spec.bodies:
            if body.name not in scene_bodies:
                body.gravcomp = 1.0  # training asset: disable_gravity=True
        for name in ARM_JOINTS:
            joint = spec.joint(name)
            joint.damping = np.zeros(3)  # arm drive gains are zeroed in training
            joint.frictionloss = 0.0
        for name in HAND_POLICY_JOINTS:
            act = spec.actuator(name)
            kp, kv, effort = (
                (INACTIVE_STIFFNESS, INACTIVE_DAMPING, INACTIVE_EFFORT)
                if name in INACTIVE_FINGERS
                else (PINCH_STIFFNESS, PINCH_DAMPING, PINCH_EFFORT)
            )
            act.gainprm[0] = kp
            act.biasprm[0] = 0.0
            act.biasprm[1] = -kp
            act.biasprm[2] = -kv
            act.forcerange = [-effort, effort]
        for geom in spec.geoms:
            if geom.classname is not None and geom.classname.name == "hand_collision":
                geom.friction = [CONTACT_FRICTION, 0.005, 0.0001]
        # The kinematic replay nut is replaced by the dynamic thread pair.
        old = spec.body("recorded_nut")
        old.mocap = False
        old.pos = [0.0, 0.0, -5.0]

    @staticmethod
    def _add_thread_pair(spec, nut_quat_wxyz) -> None:
        import mujoco

        lower = -fo.M24_BOLT_HEIGHT + 0.5 * fo.M24_NUT_HEIGHT + THREAD_HEAD_CLEARANCE_M
        upper = NUT_START_AXIAL_OFFSET_M + fo.M24_THREAD_PITCH * (THREAD_UPPER_RELEASE_TURNS + 0.5)
        carrier = spec.worldbody.add_body()
        carrier.name = "nut_carrier"
        carrier.pos = list(fo.BOLT_TIP_POSITION)
        carrier.explicitinertial = True
        carrier.mass = 0.001
        carrier.inertia = [1.0e-6, 1.0e-6, 1.0e-6]
        axial = carrier.add_joint()
        axial.name = "nut_axial"
        axial.type = mujoco.mjtJoint.mjJNT_SLIDE
        axial.axis = [0.0, 0.0, 1.0]
        axial.range = [lower, upper]
        # MuJoCo's equality stiffness scales with the dof inertia; armature on
        # the axial dof makes the thread coupling as stiff as Isaac's 200 kN/m
        # drive (0.03 mm per 20 N) without changing the nut's contact mass.
        axial.armature = THREAD_AXIAL_ARMATURE
        nut = carrier.add_body()
        nut.name = "m24_nut"
        nut.pos = [0.0, 0.0, 0.0]  # the axial slide carries the start offset
        nut.quat = [float(v) for v in nut_quat_wxyz]
        twist = nut.add_joint()
        twist.name = "nut_twist"
        twist.type = mujoco.mjtJoint.mjJNT_HINGE
        twist.axis = [0.0, 0.0, 1.0]
        twist.damping = np.array([THREAD_VISCOUS_DAMPING, 0.0, 0.0])
        twist.frictionloss = THREAD_COULOMB_TORQUE
        twist.armature = THREAD_TWIST_ARMATURE
        geom = nut.add_geom()
        geom.name = "m24_nut_geom"
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = "forge_m24_nut"
        geom.material = "forge_nut"
        geom.mass = NUT_MASS_KG
        geom.contype = 1
        geom.conaffinity = 1
        geom.friction = [CONTACT_FRICTION, 0.005, 0.0001]
        # Isaac drives the axial joint with an implicit 200 kN/m PD toward
        # start - pitch*theta/2pi. MuJoCo resolves an equality constraint
        # implicitly, so the same coupling is expressed as a joint equality
        # (axial = c0 + c1 * twist); the hold freezes it (c1 = 0).
        coupling = spec.add_equality()
        coupling.name = "thread_coupling"
        coupling.type = mujoco.mjtEq.mjEQ_JOINT
        coupling.name1 = "nut_axial"
        coupling.name2 = "nut_twist"
        coupling.data[0] = NUT_START_AXIAL_OFFSET_M
        coupling.data[1] = -THREADING_DIRECTION_SIGN * fo.M24_THREAD_PITCH / (2.0 * math.pi)
        coupling.solref = [0.004, 1.0]
        coupling.solimp = [0.99, 0.999, 0.0005, 0.5, 2.0]
        # Thread hold during release/return: an (initially inactive) equality
        # pins the twist joint; an explicit PD torque would be unstable.
        hold = spec.add_equality()
        hold.name = "thread_hold"
        hold.type = mujoco.mjtEq.mjEQ_JOINT
        hold.name1 = "nut_twist"
        hold.active = False
        hold.solref = [0.004, 1.0]
        hold.solimp = [0.99, 0.999, 0.0005, 0.5, 2.0]

    # --- state access -----------------------------------------------------------
    def q(self, names):
        return np.array([self.data.qpos[self.qadr[n]] for n in names])

    def dq(self, names):
        return np.array([self.data.qvel[self.vadr[n]] for n in names])

    def body_pose(self, body):
        return self.data.xpos[body].copy(), fo.quat_from_matrix(self.data.xmat[body].reshape(3, 3))

    def body_jacobian(self, body):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        self.mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body)
        return np.vstack((jacp, jacr))

    def grasp_state(self, z_transport) -> fo.GraspFrameState:
        thumb, _ = self.body_pose(self.thumb_tip)
        index, _ = self.body_pose(self.index_tip)
        flange_pos, flange_quat = self.body_pose(self.flange)
        j_flange = self.body_jacobian(self.flange)
        return fo.grasp_frame_state(
            thumb_pos=thumb,
            index_pos=index,
            flange_pos=flange_pos,
            flange_quat=flange_quat,
            z_transport=z_transport,
            thumb_linvel=self.body_jacobian(self.thumb_tip)[0:3] @ self.data.qvel,
            index_linvel=self.body_jacobian(self.index_tip)[0:3] @ self.data.qvel,
            flange_angvel=j_flange[3:6] @ self.data.qvel,
            flange_jacobian=j_flange[:, self.arm_dofs],
        )

    def arm_mass_matrix(self):
        return full_mass_matrix(self.model, self.data)[np.ix_(self.arm_dofs, self.arm_dofs)]

    @property
    def twist_angle(self) -> float:
        return float(self.data.qpos[self.qadr["nut_twist"]])

    @property
    def axial_position(self) -> float:
        return float(self.data.qpos[self.qadr["nut_axial"]])

    @property
    def turn_progress_rad(self) -> float:
        """ForgeUltra ``threading_directional_turn_progress`` (positive = clockwise)."""

        return THREADING_DIRECTION_SIGN * (self.twist_angle - self.twist_reference)

    def rebase_turn_reference(self) -> None:
        self.twist_reference = self.twist_angle

    # --- reset ------------------------------------------------------------------
    def reset(self, arm_joints, hand_posture: dict, settle_s: float = 0.25) -> None:
        d = self.data
        self.mujoco.mj_resetData(self.model, d)
        for name, value in zip(ARM_JOINTS, arm_joints):
            d.qpos[self.qadr[name]] = float(value)
        for name, value in _mimic_follower_targets(hand_posture).items():
            if name in self.qadr:
                d.qpos[self.qadr[name]] = float(value)
        d.qpos[self.qadr["nut_axial"]] = NUT_START_AXIAL_OFFSET_M
        d.qpos[self.qadr["nut_twist"]] = 0.0
        self.mujoco.mj_forward(self.model, d)
        self.sim_time = 0.0
        self.twist_reference = 0.0
        self.thread_hold = (NUT_START_AXIAL_OFFSET_M, 0.0)
        # Isaac's reset settle: arm pinned at the reset joints, hand servoing to
        # the grasp posture, nut pinned on the bolt.
        arm = np.asarray(arm_joints, dtype=float)
        for _ in range(int(round(settle_s / PHYSICS_DT))):
            q = self.q(ARM_JOINTS)
            dq = self.dq(ARM_JOINTS)
            self._apply_arm_torque(1000.0 * (arm - q) - 60.0 * dq)
            self._apply_hand_targets([hand_posture[n] for n in fo.PINCH_JOINTS])
            self._step_physics()
        self.thread_hold = None

    # --- actuation ----------------------------------------------------------------
    def _apply_arm_torque(self, torque) -> None:
        torque = np.clip(np.asarray(torque, dtype=float), self.arm_ctrl_range[:, 0], self.arm_ctrl_range[:, 1])
        for name, value in zip(ARM_JOINTS, torque):
            self.data.ctrl[self.aid[name]] = value

    def _apply_hand_targets(self, pinch_targets) -> None:
        for name, value in zip(fo.PINCH_JOINTS, pinch_targets):
            self.data.ctrl[self.aid[name]] = float(value)
        for name in INACTIVE_FINGERS:
            self.data.ctrl[self.aid[name]] = INACTIVE_FINGER_TARGET

    def _apply_thread_drive(self) -> None:
        d = self.data
        m = self.model
        if self.thread_hold is not None:
            axial, twist = self.thread_hold
            m.eq_data[self.coupling_eq, 0] = axial
            m.eq_data[self.coupling_eq, 1] = 0.0
            m.eq_data[self.hold_eq, 0] = twist
            d.eq_active[self.hold_eq] = 1
            return
        m.eq_data[self.coupling_eq, 0] = NUT_START_AXIAL_OFFSET_M
        m.eq_data[self.coupling_eq, 1] = self.coupling_slope
        d.eq_active[self.hold_eq] = 0

    def _step_physics(self) -> None:
        self._apply_thread_drive()
        for _ in range(SUBSTEPS):
            self.mujoco.mj_step(self.model, self.data)
        self.sim_time += PHYSICS_DT

    def hold_thread(self) -> None:
        self.thread_hold = (self.axial_position, self.twist_angle)

    def release_thread_hold(self) -> None:
        self.thread_hold = None

    def control_tick(self, filtered_action, z_transport, *, dead_zone=None):
        """One 120 Hz tick of the OSC -> joint-PD -> torque path; returns diagnostics."""

        q = self.q(ARM_JOINTS)
        dq = self.dq(ARM_JOINTS)
        grasp = self.grasp_state(z_transport)
        target = fo.decode_action_target(filtered_action, grasp)
        tau, wrench, pos_error, aa_error = fo.compute_dof_torque(
            dof_pos_arm=q,
            dof_vel_arm=dq,
            grasp=grasp,
            arm_mass_matrix=self.arm_mass_matrix(),
            target_pos=target.pos,
            target_quat=target.quat,
            dead_zone_thresholds=dead_zone,
        )
        pinch = fo.pinch_targets(filtered_action)
        command = fo.joint_pd_command_from_torque(q, dq, tau, pinch)
        arm_torque = fo.joint_pd_torque(q, dq, command)
        self._apply_arm_torque(arm_torque)
        self._apply_hand_targets(pinch)
        self._step_physics()
        return {
            "grasp": grasp,
            "target_pos": target.pos,
            "target_quat": target.quat,
            "osc_torque": tau,
            "pd_command": command,
            "wrench": wrench,
        }


@dataclass
class CyclicCoordinator:
    """``build_cyclic_release_teacher`` lifecycle without the teacher command."""

    max_cycles: int
    reset_grasp_pos: np.ndarray
    reset_grasp_quat: np.ndarray
    reset_hand: np.ndarray
    waypoint_steps: int = max(1, math.ceil(WAYPOINT_DURATION_S / POLICY_DT))
    return_steps: int = max(1, math.ceil(RETURN_DURATION_S / POLICY_DT))
    timeout_steps: int = max(1, math.ceil(RETURN_TIMEOUT_S / POLICY_DT))
    active: bool = False
    phase: int = -1
    phase_step: int = 0
    wait_steps: int = 0
    completed_cycles: int = 0
    failed: bool = False
    limit_reached: bool = False
    events: list = field(default_factory=list)

    @property
    def phase_count(self) -> int:
        return WAYPOINT_PHASES + 1

    def process_phase(self) -> str:
        if not self.active:
            return "policy"
        return "return_to_reset" if self.phase == WAYPOINT_PHASES else "follow_waypoints"

    def return_errors(self, grasp: fo.GraspFrameState, hand_joints):
        position_error = float(np.linalg.norm(grasp.pos - self.reset_grasp_pos))
        alignment = min(1.0, abs(float(np.dot(grasp.quat, self.reset_grasp_quat))))
        orientation_error_deg = math.degrees(2.0 * math.acos(alignment))
        hand_error = float(np.max(np.abs(np.asarray(hand_joints) - self.reset_hand)))
        return position_error, orientation_error_deg, hand_error

    def physically_returned(self, grasp, hand_joints) -> bool:
        p, o, h = self.return_errors(grasp, hand_joints)
        return (
            p <= RETURN_POSITION_TOLERANCE_M
            and o <= RETURN_ORIENTATION_TOLERANCE_DEG
            and h <= RETURN_HAND_TOLERANCE_RAD
        )

    def observe(self, *, step: int, turn_progress_rad: float, grasp, hand_joints, scene: ThreadingScene):
        """Advance after one policy step. Returns an event string or None."""

        event = None
        if self.active:
            self.phase_step += 1
            duration = self.waypoint_steps if self.phase < WAYPOINT_PHASES else self.return_steps
            if self.phase_step >= duration:
                if self.phase < WAYPOINT_PHASES:
                    self.phase += 1
                    self.phase_step = 0
                    self.wait_steps = 0
                elif self.physically_returned(grasp, hand_joints):
                    # on_finish: rebase, release the hold, count the cycle.
                    scene.rebase_turn_reference()
                    scene.release_thread_hold()
                    self.completed_cycles += 1
                    self.active = False
                    self.phase = -1
                    self.phase_step = 0
                    self.wait_steps = 0
                    self.limit_reached = self.completed_cycles >= self.max_cycles
                    event = "cycle_completed"
                else:
                    self.wait_steps += 1
                    if self.wait_steps >= self.timeout_steps:
                        scene.release_thread_hold()
                        self.active = False
                        self.phase = -1
                        self.failed = True
                        event = "return_failed"
        # Isaac evaluates the trigger on the post-step observation, i.e. after
        # a completed cycle has already rebased the turn reference.
        if (
            not self.active
            and not self.failed
            and not self.limit_reached
            and scene.turn_progress_rad >= math.radians(RELEASE_MIN_TURN_DEG)
        ):
            scene.hold_thread()
            self.active = True
            self.phase = 0
            self.phase_step = 0
            self.wait_steps = 0
            event = "release_started"
        if event is not None:
            self.events.append({"step": step, "event": event, "cycles": self.completed_cycles})
        return event


def seed_action_history(grasp: fo.GraspFrameState, hand_joints, closed_posture=None):
    """``seed_threading_policy_action_history``: encode the live pose as the previous action."""

    closed = closed_posture or fo.THREADING_GRASP_POSTURE
    seed = np.zeros(9)
    seed[0:3] = (grasp.pos - fo.BOLT_TIP_POSITION) / fo.POS_ACTION_BOUNDS
    base = fo.quat_from_euler_xyz(*fo.EE_BASE_ORN_EULER)
    rel = fo.quat_mul(fo.quat_conjugate(base), grasp.quat)
    roll, pitch, yaw = fo.get_euler_xyz(rel)
    if yaw > math.pi / 2:
        yaw -= 2.0 * math.pi
    if yaw < -math.pi:
        yaw += 2.0 * math.pi
    seed[5] = (yaw + math.radians(180.0)) / math.radians(270.0) * 2.0 - 1.0

    def wrap(x):
        return math.atan2(math.sin(x), math.cos(x))

    seed[3] = wrap(roll) / fo.ROT_ACTION_BOUNDS[0]
    seed[4] = wrap(pitch) / fo.ROT_ACTION_BOUNDS[1]
    # Hand: the recorded episodes seed the pinch actions so that the relative
    # pinch target equals the live reset hand pose.
    for k, (name, (lo, hi)) in enumerate(zip(fo.PINCH_JOINTS, fo.PINCH_RANGES)):
        seed[6 + k] = float(np.clip(2.0 * (hand_joints[k] - closed[name]) / (hi - lo), -1.0, 1.0))
    return seed


__all__ = [
    "ARM_JOINTS",
    "CyclicCoordinator",
    "DECIMATION",
    "HAND_POLICY_JOINTS",
    "PHYSICS_DT",
    "POLICY_DT",
    "POLICY_JOINTS",
    "ThreadingScene",
    "seed_action_history",
]
