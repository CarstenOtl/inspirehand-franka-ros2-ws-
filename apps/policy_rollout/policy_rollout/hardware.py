"""Real FR3/Inspire/RealSense adapter for the distilled threading policy.

The non-realtime policy process publishes bounded Cartesian setpoints. The
``CartesianImpedanceController`` owns the FR3 effort interfaces and
executes those setpoints in its 1 kHz loop. The RH56 remains on the same
independent ``/inspire_hand/command`` link used by trajectory replay.

ROS imports are deliberately lazy so unit tests continue to work on machines
without a ROS installation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import threading
import time

import numpy as np

from . import forge_osc as fo
from utils.camera_calibration import CameraCalibrationProfile


ARM_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))
HAND_JOINTS = (
    "pinky_proximal_joint",
    "ring_proximal_joint",
    "middle_proximal_joint",
    "index_proximal_joint",
    "thumb_proximal_pitch_joint",
    "thumb_proximal_yaw_joint",
)
SUPPORT_FINGER_TARGETS = np.array([1.333, 1.333, 1.333], dtype=float)
POLICY_HAND_JOINTS = (
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "index_proximal_joint",
)

# A force-observing student's pad forces, [thumb, index]. ros-sim's
# FingertipForcePlugin publishes them already projected the way ForgeUltra does,
# in N, under ForgeUltra's channel names; the RH56 driver's ~/grip_force carries
# raw FORCE_ACT registers in grams under its channel ids, thumb_bend = "5" and
# index = "4".
FINGERTIP_FORCE_CHANNELS = (
    "thumb_flexion_extension_force",
    "index_flexion_extension_force",
)
RH56_FORCE_CHANNELS = ("5", "4")
GRAM_FORCE_N = 9.80665e-3

# A force-observing student's seven arm torques. Training reads PhysX's
# projected joint-reaction torque with the robot's gravity off. Which FR3 signal
# matches it best is still being measured, so every candidate the backend has is
# recorded on every step, and --arm-torque-source picks the one the policy sees:
#
# - external:  libfranka's tau_ext_hat_filtered (tau_J minus the model's
#              gravity and dynamics), ForgeUltra's recommendation;
# - measured:  link-side tau_J minus the model's gravity, i.e. contact plus the
#              motion torques PhysX's transmitted torque also contains (the
#              controller low-passes tau_J at 1 kHz and publishes both);
# - desired:   libfranka's tau_J_d, the commanded torque without gravity;
# - commanded: the law's tau_task + tau_nullspace (no coriolis feed-forward,
#              which ForgeUltra's OSC does not have).
#
# external and desired come from franka_robot_state_broadcaster's convenience
# topics (100 Hz in controllers_policy.yaml); ros-sim has no broadcaster, and
# there measured is MuJoCo's applied actuator torque (gravity is off).
FRANKA_TORQUE_TOPICS = {
    "external": "/franka_robot_state_broadcaster/external_joint_torques",
    "desired": "/franka_robot_state_broadcaster/desired_joint_states",
}
ARM_TORQUE_SOURCES = ("auto", "external", "measured", "desired", "commanded")
#: First-order low-pass on the broadcaster torques before the 15 Hz sample:
#: about one 100 Hz message's worth of smoothing against aliasing. The
#: controller applies the same time constant to tau_J.
ARM_TORQUE_LOWPASS_S = 0.02


def arm_torque_candidates(*, sim: bool) -> tuple[str, ...]:
    """The torque candidates a backend records: all four on hardware, two in ros-sim."""

    return ("measured", "commanded") if sim else ("external", "measured", "desired", "commanded")


def resolve_arm_torque_source(source: str, *, sim: bool) -> str:
    """``auto`` is the external estimate on hardware and the law's torque in ros-sim."""

    if source not in ARM_TORQUE_SOURCES:
        raise ValueError("arm torque source must be one of " + ", ".join(ARM_TORQUE_SOURCES))
    if source == "auto":
        return "commanded" if sim else "external"
    if source not in arm_torque_candidates(sim=sim):
        raise ValueError(
            f"arm torque source {source!r} needs franka_robot_state_broadcaster, "
            "which ros-sim does not run"
        )
    return source


def arm_torque_candidates_from_controller(cartesian) -> dict[str, np.ndarray]:
    """``measured`` and ``commanded`` from one CartesianReplayState, N m."""

    if not hasattr(cartesian, "tau_measured_filtered"):
        raise RuntimeError(
            "CartesianReplayState has no tau_measured_filtered; rebuild "
            "franka_forge_cartesian_impedance_msgs and franka_forge_cartesian_impedance"
        )
    return {
        "measured": np.asarray(cartesian.tau_measured_filtered, dtype=float)
        - np.asarray(cartesian.tau_gravity, dtype=float),
        "commanded": np.asarray(cartesian.tau_task, dtype=float)
        + np.asarray(cartesian.tau_nullspace, dtype=float),
    }


def external_joint_torque_message_to_array(message) -> np.ndarray:
    """The seven FR3 efforts of a broadcaster JointState in ARM_JOINTS order, N m."""

    efforts = dict(zip(message.name, message.effort))
    if any(name not in efforts for name in ARM_JOINTS):
        raise RuntimeError("external joint torque message does not contain all FR3 joints")
    torque = np.array([efforts[name] for name in ARM_JOINTS], dtype=float)
    if not np.isfinite(torque).all():
        raise RuntimeError("external joint torque message contains non-finite values")
    return torque


class TorqueLowPass:
    """First-order low-pass over irregularly timed samples (``time_constant_s`` 0: none)."""

    def __init__(self, time_constant_s: float = ARM_TORQUE_LOWPASS_S) -> None:
        if not np.isfinite(time_constant_s) or time_constant_s < 0.0:
            raise ValueError("low-pass time constant must be a non-negative number")
        self.time_constant_s = float(time_constant_s)
        self.value: np.ndarray | None = None
        self._time_s: float | None = None

    def update(self, value, time_s: float) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        if self.value is None or self.time_constant_s == 0.0:
            self.value = value.copy()
        else:
            dt = max(float(time_s) - self._time_s, 0.0)
            alpha = 1.0 - np.exp(-dt / self.time_constant_s)
            self.value = self.value + alpha * (value - self.value)
        self._time_s = float(time_s)
        return self.value.copy()


#: How far past either end of the overlaid range the thumb-yaw feedback may
#: read, as a logical open ratio, before it counts as a fault: 30 register
#: counts, about 0.037 rad. 20 was too tight: on 2026-10-29 a thumb held at its
#: yaw limit settled 22.7 counts past it and ended two runs after 20 steps.
THUMB_FEEDBACK_SLACK_OPEN_RATIO = 0.03


def physical_hand_state_to_policy(
    hand_position, hand_velocity, *, thumb_overlay: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Map physical RH56 feedback into the checkpoint's logical coordinates.

    The driver contracts thumb-yaw commands into the physically usable final
    75 percent of travel. Its joint-state topic correctly reports that physical
    pose for TF, but the checkpoint was trained on the pre-overlay logical
    coordinate. Only the policy scalar and its derivative are inverted here;
    the physical TF tree remains untouched.

    ``thumb_overlay=False`` says the driver runs without the overlay (its
    ``thumb_abduction_overlay`` parameter, off in ros-sim): the reported pose
    then already is the logical one and nothing is inverted.
    """

    from inspire_hand_driver import command_overlays
    from inspire_hand_driver import kinematics as kin

    positions = np.asarray(hand_position, dtype=float).reshape(-1)
    velocities = np.asarray(hand_velocity, dtype=float).reshape(-1)
    if positions.shape != (len(HAND_JOINTS),) or velocities.shape != (
        len(HAND_JOINTS),
    ):
        raise ValueError("physical hand state must contain all six driven joints")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("physical hand state contains non-finite values")
    columns = {name: index for index, name in enumerate(HAND_JOINTS)}
    indices = [columns[name] for name in POLICY_HAND_JOINTS]
    q_policy = positions[indices].copy()
    dq_policy = velocities[indices].copy()
    if not thumb_overlay:
        return q_policy, dq_policy

    thumb_dof = kin.dof_index(command_overlays.THUMB_ABDUCTION_JOINT)
    physical_ratio = kin.rad_to_open_ratio(thumb_dof, q_policy[0])
    logical_ratio = command_overlays.invert_open_ratio_overlay(
        thumb_dof, physical_ratio
    )
    # A thumb driven onto the end of its range can settle a few register
    # counts (1/1000 each) past it: 2026-10-26 the student commanded 1.27 rad
    # against the RH56's 1.246 rad yaw limit and the feedback overshot the
    # overlay's zero. That is the limit, not a mapping fault; anything further
    # out still is.
    slack = THUMB_FEEDBACK_SLACK_OPEN_RATIO
    if not -slack <= logical_ratio <= 1.0 + slack:
        raise ValueError(
            "physical thumb-yaw feedback is outside the overlaid policy range "
            f"(logical open ratio {logical_ratio:.4f})"
        )
    logical_ratio = min(max(logical_ratio, 0.0), 1.0)
    q_policy[0] = kin.open_ratio_to_rad(thumb_dof, logical_ratio)
    travel_scale = 1.0 - command_overlays.THUMB_ABDUCTION_ZERO_OPEN_RATIO
    dq_policy[0] /= travel_scale
    return q_policy, dq_policy


def hand_joint_state_to_arrays(message) -> tuple[np.ndarray, np.ndarray]:
    """Pull the six driven RH56 joints out of a ``JointState``, in HAND_JOINTS order."""

    positions = dict(zip(message.name, message.position))
    velocities = dict(zip(message.name, message.velocity))
    if any(name not in positions for name in HAND_JOINTS):
        raise RuntimeError("hand state does not contain all driven RH56 joints")
    return (
        np.array([positions[name] for name in HAND_JOINTS], dtype=float),
        np.array([velocities.get(name, 0.0) for name in HAND_JOINTS], dtype=float),
    )


def fingertip_force_message_to_newtons(message) -> tuple[np.ndarray, bool]:
    """``[thumb, index]`` force in N from either source, and whether it is raw.

    Raw means the driver's untared FORCE_ACT, whose zero moves after a heavy
    load; the caller subtracts a zero captured with the pads free.
    """

    efforts = dict(zip(message.name, message.effort))
    if all(name in efforts for name in FINGERTIP_FORCE_CHANNELS):
        return np.array([efforts[name] for name in FINGERTIP_FORCE_CHANNELS], dtype=float), False
    if all(name in efforts for name in RH56_FORCE_CHANNELS):
        grams = np.array([efforts[name] for name in RH56_FORCE_CHANNELS], dtype=float)
        return GRAM_FORCE_N * grams, True
    raise RuntimeError(
        "fingertip force message has neither "
        + "/".join(FINGERTIP_FORCE_CHANNELS)
        + " nor the RH56 channels "
        + "/".join(RH56_FORCE_CHANNELS)
    )


def grasp_z_transport_at_reset(
    thumb_position, index_position, flange_position, flange_quaternion
) -> np.ndarray:
    """ForgeUltra's reset grasp approach axis, expressed in the flange frame.

    Training stores the untilted reset grasp Z (world down, orthogonalised
    against thumb->index) as a flange-frame vector once at reset and rotates it
    with the flange afterwards. On this hand that vector is mostly flange +Y;
    the flange's own -Z axis is about 107 degrees away from it at the M24 reset
    pose. fr3_link0 shares the training world's vertical, so base-frame inputs
    give the same flange-frame vector.
    """

    return fo.reset_z_transport(
        np.asarray(thumb_position, dtype=float),
        np.asarray(index_position, dtype=float),
        np.asarray(flange_position, dtype=float),
        np.asarray(flange_quaternion, dtype=float),
        tilt_deg=0.0,
    )


def grasp_frame_from_tips(
    thumb_position, index_position, flange_quaternion, z_transport
) -> tuple[np.ndarray, np.ndarray]:
    """``grasp_frame_state``'s pose: fingertip midpoint, approach from the reset transport."""

    approach = fo.quat_rotate(
        np.asarray(flange_quaternion, dtype=float), np.asarray(z_transport, dtype=float)
    )
    position, rotation = fo.hand_grasp_frame(
        np.asarray(thumb_position, dtype=float),
        np.asarray(index_position, dtype=float),
        approach,
    )
    return position, fo.quat_from_matrix(rotation)


def grasp_in_flange(
    grasp_position, grasp_quaternion, flange_position, flange_quaternion
) -> np.ndarray:
    """The grasp frame expressed on the flange, as ``PolicyGoal.tool_in_flange`` carries it.

    ``[x y z qx qy qz qw]``: the transform the controller composes onto its own
    flange pose to measure, clip and differentiate its law at the live fingertip
    midpoint, which is where training applied its wrench at every substep. The
    grasp frame is built from the tips in the flange frame and a flange-frame
    approach axis, so this depends on the hand joints only; the arm pose cancels.
    """

    positions = [np.asarray(v, dtype=float) for v in (grasp_position, flange_position)]
    quaternions = [np.asarray(v, dtype=float).copy() for v in (grasp_quaternion, flange_quaternion)]
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in positions):
        raise ValueError("grasp/flange positions must contain three finite values")
    if any(
        v.shape != (4,) or not np.isfinite(v).all() or np.linalg.norm(v) < 1.0e-8
        for v in quaternions
    ):
        raise ValueError("grasp/flange quaternions must contain four finite values")
    for v in quaternions:
        v /= np.linalg.norm(v)
    grasp_p, flange_p = positions
    grasp_q, flange_q = quaternions
    flange_inverse = fo.quat_conjugate(flange_q)
    translation = fo.quat_rotate(flange_inverse, grasp_p - flange_p)
    rotation = fo.quat_mul(flange_inverse, grasp_q)
    rotation /= np.linalg.norm(rotation)
    return np.array(
        [*translation, rotation[1], rotation[2], rotation[3], rotation[0]], dtype=float
    )


TRAINING_INACTIVE_FINGERS = {
    "middle_joint_0": 1.333,
    "ring_joint_0": 1.333,
    "little_joint_0": 1.333,
}


class TrainingHandKinematics:
    """Fingertip frames exactly as ForgeUltra's training asset defines them.

    Since 2026-10-16 the workspace URDF is generated from the same training
    URDF, so its ``thumb_tip``/``index_tip`` frames agree with these to 1e-7 m
    at the same joints (``tests/test_hardware.py``). Until then it was dex-urdf's
    derivation, whose tips sat about 15 mm and 21 degrees away.

    The rollout still builds the tips here rather than reading them out of TF,
    because TF is posed from the *physical* joint state and the policy works in
    its own logical coordinates: the driver's thumb-yaw overlay moves the
    physical thumb off the logical value, and the driver clamps the thumb's
    followers at their limits where training's mimic expansion does not. The
    hand is rigid on the flange, so tip positions in the flange frame depend
    only on the three policy hand joints and the arm pose cancels; TF still
    supplies the flange itself.

    Whether the *physical* fingers then land where these say depends on
    whether the training model describes this workcell's hand, which is still
    an open bench measurement.
    """

    def __init__(self) -> None:
        import mujoco

        from .mujoco_scene import load_training_scene

        self._mujoco = mujoco
        self.model, self.data = load_training_scene(urdf_tip_frames=True)
        self._address = {
            self.model.joint(index).name: self.model.jnt_qposadr[index]
            for index in range(self.model.njnt)
        }
        self._flange = self.model.body("fr3_link8").id
        self._thumb = self.model.body("thumb_tip").id
        self._index = self.model.body("index_tip").id

    def tips_in_flange(self, pinch_positions) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(thumb, index)`` tip positions in the flange frame."""

        values = np.asarray(pinch_positions, dtype=float).flatten()
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("pinch_positions must contain three finite values")
        data = self.data
        data.qpos[:] = 0.0
        targets = dict(zip(fo.PINCH_JOINTS, values))
        targets.update(TRAINING_INACTIVE_FINGERS)
        for name, value in fo.expand_hand_mimic(targets).items():
            address = self._address.get(name)
            if address is not None:
                data.qpos[address] = float(value)
        self._mujoco.mj_kinematics(self.model, data)
        rotation = data.xmat[self._flange].reshape(3, 3)
        origin = data.xpos[self._flange]
        return (
            rotation.T @ (data.xpos[self._thumb] - origin),
            rotation.T @ (data.xpos[self._index] - origin),
        )

    def frames_in_world(self, arm_positions, pinch_positions):
        """``(flange_position, flange_quaternion_wxyz, thumb, index)`` in the training world.

        Forward kinematics of the training asset at the given arm joints and
        pinch posture, in the world the student was trained in (the FR3 base at
        ``forge_osc.ROBOT_BASE_POSITION`` yawed by pi).
        """

        arm = np.asarray(arm_positions, dtype=float).flatten()
        values = np.asarray(pinch_positions, dtype=float).flatten()
        if arm.shape != (7,) or not np.isfinite(arm).all():
            raise ValueError("arm_positions must contain seven finite values")
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("pinch_positions must contain three finite values")
        data = self.data
        data.qpos[:] = 0.0
        for index, value in enumerate(arm):
            data.qpos[self._address[f"fr3_joint{index + 1}"]] = float(value)
        targets = dict(zip(fo.PINCH_JOINTS, values))
        targets.update(TRAINING_INACTIVE_FINGERS)
        for name, value in fo.expand_hand_mimic(targets).items():
            address = self._address.get(name)
            if address is not None:
                data.qpos[address] = float(value)
        self._mujoco.mj_kinematics(self.model, data)
        return (
            data.xpos[self._flange].copy(),
            data.xquat[self._flange].copy(),
            data.xpos[self._thumb].copy(),
            data.xpos[self._index].copy(),
        )


#: How far the homing goal may be shifted from the reset joints to cancel a
#: steady-state residual, per joint. The 2026-10-26 residual peaked at
#: 0.056 rad (joint 7); a larger one is a fault, not friction to lean on.
MAX_HOME_PRECOMPENSATION_RAD = 0.1


def next_home_command(command, home_arm, joint_residual) -> np.ndarray:
    """The next homing goal: the last one shifted against the residual it left.

    ``joint_residual`` is measured minus ``home_arm``. Each retry moves the goal
    by that amount the other way, so a PD controller held off the reset pose by
    friction or an unmodelled load comes to rest on it. The total shift from
    ``home_arm`` is clipped to MAX_HOME_PRECOMPENSATION_RAD per joint.
    """

    home_arm = np.asarray(home_arm, dtype=float)
    shifted = np.asarray(command, dtype=float) - np.asarray(joint_residual, dtype=float)
    offset = np.clip(shifted - home_arm, -MAX_HOME_PRECOMPENSATION_RAD, MAX_HOME_PRECOMPENSATION_RAD)
    return home_arm + offset


def trained_reset_grasp(training_hand, frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The grasp frame the student was reset to, in the FR3 base frame.

    ``(position, quaternion_wxyz, pinch_positions)``: forward kinematics of
    ``forge_osc.FRANKA_ARM_RESET_JOINTS_M24`` and Isaac's live entry pinch
    (``forge_osc.ISAAC_ENTRY_PINCH``) through the training asset, built exactly
    as the environment builds its grasp frame at reset (world-down approach
    orthogonalised against thumb->index). This is what the student steers back
    to in ``return_to_reset`` and what the return gate must be measured
    against; the pose the arm actually homed to is a measurement of the
    homing, not the reference (the 2026-10-16 hardware run homed 12.6 mm below
    it). Until 2026-10-26 it used the closed THREADING_GRASP_POSTURE, which
    sits 3.9 mm / 6.7 deg from where the teacher actually returns.
    """

    pinch = np.asarray(fo.ISAAC_ENTRY_PINCH, dtype=float)
    flange_position, flange_quaternion, thumb, index = training_hand.frames_in_world(
        fo.FRANKA_ARM_RESET_JOINTS_M24, pinch
    )
    z_transport = grasp_z_transport_at_reset(thumb, index, flange_position, flange_quaternion)
    position, quaternion = grasp_frame_from_tips(thumb, index, flange_quaternion, z_transport)
    base_position, base_quaternion = frame.pose_world_to_base(position, quaternion)
    return base_position, base_quaternion, pinch


# Training's release starts by opening the thumb and index (waypoint1 of
# forgeUltra assets/global_pose_settings/waypoints_release_and_reset_franka.yaml:
# thumb_joint_0 1.093, thumb_joint_1 0.0001, index_joint_0 0.03) and its reset
# commands the grasp posture, so the first observation of an episode shows the
# hand still closing (gap doc C2). ``--hand-start release`` reproduces that.
RELEASE_START_PINCH = np.array([1.093208432, 0.0001, 0.03])


# The Cartesian controller's compliance centre until the first policy goal.
#
# Training applied its operational-space wrench at the *live* fingertip
# midpoint, which travels about 20 mm over a threading cycle, and every policy
# goal now carries that live midpoint (``grasp_in_flange``) for the controller
# to act about. This is the same midpoint frozen at the threading grip --
# TrainingHandKinematics evaluated at forge_osc.THREADING_GRASP_POSTURE -- and
# it is what the controller holds about between activation and the first goal.
#
# It deliberately differs from `tcp.offset_xyz` in replay.yaml
# (-0.0874, -0.0327, 0.1453), which is the midpoint of a *closed* replay pinch
# and is 40 mm away. Trajectory replay keeps that one; its recorded Cartesian
# poses are expressed at it. Keep this equal to `tool_offset_xyz` in
# controllers_policy.yaml and controllers_sim_policy.yaml, which preflight
# checks, and to the `policy_grasp_tcp` marker in inspire_hand_on_flange.xml.
POLICY_TOOL_OFFSET_XYZ = (-0.059067, -0.028773, 0.173311)
POLICY_TOOL_OFFSET_RPY = (0.0, 0.0, 0.0)


def policy_tool_offset_config() -> dict:
    """The policy rollout's own flange-to-tool transform, in config form."""

    return {
        "tcp": {
            "offset_xyz": list(POLICY_TOOL_OFFSET_XYZ),
            "offset_rpy": list(POLICY_TOOL_OFFSET_RPY),
        }
    }


@dataclass(frozen=True)
class HardwareReadiness:
    ready: bool
    blockers: tuple[str, ...]


def assess_hardware_readiness(
    calibration: CameraCalibrationProfile,
) -> HardwareReadiness:
    """Static readiness only; graph/controller checks happen at run time."""

    blockers = tuple(calibration.hardware_blockers())
    return HardwareReadiness(ready=not blockers, blockers=blockers)


def image_message_to_numpy(message) -> tuple[np.ndarray, str]:
    """Decode the RealSense Image encodings used here without cv_bridge."""

    encoding = str(message.encoding).lower()
    height, width, step = int(message.height), int(message.width), int(message.step)
    raw = memoryview(message.data)
    if encoding in {"rgb8", "bgr8", "rgba8", "bgra8"}:
        channels = 4 if "a" in encoding else 3
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, step)
        image = rows[:, : width * channels].reshape(height, width, channels)[..., :3]
        if encoding.startswith("bgr"):
            image = image[..., ::-1]
        return np.ascontiguousarray(image), "rgb"
    if encoding in {"16uc1", "mono16"}:
        rows = np.frombuffer(raw, dtype=np.uint16).reshape(height, step // 2)
        return np.ascontiguousarray(rows[:, :width]), "millimetres"
    if encoding == "32fc1":
        rows = np.frombuffer(raw, dtype=np.float32).reshape(height, step // 4)
        return np.ascontiguousarray(rows[:, :width]), "metres"
    raise ValueError(f"unsupported RealSense image encoding {message.encoding!r}")


def _stamp_seconds(message) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + 1.0e-9 * float(stamp.nanosec)


def assert_policy_camera_frames(rgb, depth, camera_info, expected_frame_id: str) -> None:
    """Require every RGB-D input to use the calibrated colour pixel frame."""

    frames = {
        "RGB": str(rgb.header.frame_id),
        "depth": str(depth.header.frame_id),
        "CameraInfo": str(camera_info.header.frame_id),
    }
    mismatches = [
        f"{label}={frame_id!r}"
        for label, frame_id in frames.items()
        if frame_id != expected_frame_id
    ]
    if mismatches:
        raise RuntimeError(
            "policy RGB-D is not registered to the calibrated colour frame "
            f"{expected_frame_id!r}: " + ", ".join(mismatches)
        )


def policy_tool_transform(config) -> np.ndarray:
    """Build the configured flange-to-tool transform through the kinematics API."""

    from franka_trajectory_replay import kinematics

    return kinematics.tool_transform(
        config["tcp"]["offset_xyz"], config["tcp"]["offset_rpy"]
    )


class RgbdFrameSynchronizer:
    """Keep recent camera messages and return the closest timestamped pair."""

    def __init__(self, max_skew_s: float, queue_size: int = 8) -> None:
        if max_skew_s <= 0.0 or queue_size < 1:
            raise ValueError("RGB-D synchronization settings must be positive")
        self.max_skew_s = float(max_skew_s)
        self.queue_size = int(queue_size)
        self._queues = {"rgb": [], "depth": []}

    def add(self, stream: str, message, receive_time_s: float) -> None:
        if stream not in self._queues:
            raise ValueError(f"unknown camera stream {stream!r}")
        queue = self._queues[stream]
        queue.append((_stamp_seconds(message), float(receive_time_s), message))
        del queue[:-self.queue_size]

    def latest_pair(self):
        candidates = []
        for rgb in self._queues["rgb"]:
            for depth in self._queues["depth"]:
                skew = abs(rgb[0] - depth[0])
                if skew <= self.max_skew_s:
                    candidates.append((skew, min(rgb[0], depth[0]), rgb, depth))
        if not candidates:
            return None
        # Prefer the newest complete pair. Use the smaller skew as the
        # tie-breaker when two candidates share the same older timestamp; that
        # avoids selecting a new RGB frame with the previous depth frame while
        # its exact partner is still between callbacks.
        _skew, _stamp, rgb, depth = max(
            candidates, key=lambda item: (item[1], -item[0])
        )
        return rgb[2], depth[2], rgb[1], depth[1]


class TrainingFrameAdapter:
    """Convert between the policy's training world and the physical FR3 base."""

    # The training world is fr3_link0 yawed by this angle (and shifted, which no
    # rotation-only consumer cares about). The controller clips policy goals in
    # that world (policy_clip_frame_yaw), because ForgeUltra's Euler clip and its
    # yaw wrap are written in it.
    world_from_base_yaw = math.pi

    def __init__(self) -> None:
        self.world_from_base_position = fo.ROBOT_BASE_POSITION.copy()
        self.world_from_base_quaternion = fo.quat_from_euler_xyz(
            0.0, 0.0, self.world_from_base_yaw
        )

    def pose_base_to_world(self, position, quaternion):
        q = self.world_from_base_quaternion
        return (
            self.world_from_base_position
            + fo.quat_rotate(q, np.asarray(position, dtype=float)),
            fo.quat_mul(q, np.asarray(quaternion, dtype=float)),
        )

    def pose_world_to_base(self, position, quaternion):
        q_inv = fo.quat_conjugate(self.world_from_base_quaternion)
        return (
            fo.quat_rotate(
                q_inv,
                np.asarray(position, dtype=float) - self.world_from_base_position,
            ),
            fo.quat_mul(q_inv, np.asarray(quaternion, dtype=float)),
        )



def policy_goal_base(frame, filtered_action) -> tuple[np.ndarray, np.ndarray]:
    """The policy's preclipped goal (``bolt_tip + action``) in the base frame.

    This is ``_apply_action`` steps (0) and (1), the part that depends only on
    the filtered action and is constant over a policy tick. Step (2), the clip
    against the live grasp pose, is the controller's: it runs it at 1 kHz on
    every ``PolicyGoal`` (policy_clip_position_m / policy_clip_orientation_rad
    in the training world frame), as Isaac ran it at every 120 Hz substep.
    """

    # decode_action_target with clip=False never reads the grasp state.
    unused = fo.GraspFrameState(
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        linvel=np.zeros(3),
        angvel=np.zeros(3),
        jacobian=np.zeros((6, 7)),
    )
    goal = fo.decode_action_target(filtered_action, unused, clip=False)
    return frame.pose_world_to_base(goal.pos, goal.quat)


class GripCycleCoordinator:
    """Physical release/return phase state from measured grasp rotation.

    The physical workcell has no nut-angle topic. While the nut is grasped its
    directional turn is observable as grasp-frame yaw, so that signal drives
    the same transition threshold as the simulator. It is recorded as a proxy
    and is never promoted to task-success evidence.

    In ros-sim the nut's real twist IS available (``ThreadPairClient``), and
    ``update`` then gates on that instead; the proxy is still advanced and
    recorded alongside it.

    The release fires ``fo.RELEASE_MARGIN_RAD`` (5 deg) before the cycle's turn
    target. Training aims every turn after the first at the next 60 deg step of
    the nut's total turn (``fo.next_cycle_target_turn_rad``), which needs that
    total at the cycle boundary: ros-sim measures it, the proxy cannot (the
    wrist yaw returns to zero with the hand), so on hardware every cycle keeps
    the 60 deg target and releases at 55 deg.
    """

    def __init__(
        self,
        *,
        rate_hz,
        max_cycles,
        reset_position,
        reset_quaternion,
        reset_hand,
        min_fingertip_separation_m=None,
        max_proxy_step_rad=None,
    ):
        self.rate_hz = float(rate_hz)
        self.max_cycles = int(max_cycles)
        self.reset_position = np.asarray(reset_position, dtype=float).copy()
        self.reset_quaternion = np.asarray(reset_quaternion, dtype=float).copy()
        self.reset_hand = np.asarray(reset_hand, dtype=float).copy()
        # Guards on the proxy, for when it is what gates the release. The
        # 2026-10-16 hardware run released on a collapsed grasp frame: the
        # index closed on air, the fingertips converged 54 -> 7.9 mm, and the
        # proxy jumped -22 -> +74 deg in four ticks. The nut is 36 mm across
        # flats, so a pinch narrower than ~30 mm holds nothing, and a real turn
        # never moves 20 deg in one 15 Hz tick. None disables a guard.
        self.min_fingertip_separation_m = (
            None if min_fingertip_separation_m is None else float(min_fingertip_separation_m)
        )
        self.max_proxy_step_rad = (
            None if max_proxy_step_rad is None else float(max_proxy_step_rad)
        )
        self.active = False
        self.phase_index = -1
        self.phase_steps = 0
        self.wait_steps = 0
        self.completed_cycles = 0
        self.failed = False
        self._previous_yaw = 0.0
        self._unwrapped_yaw = 0.0
        self._cycle_yaw_origin = 0.0
        self.last_proxy_turn_rad = 0.0
        self.last_proxy_step_rad = 0.0
        self.gated_on_measured_turn = False
        self.proxy_vetoes = 0
        self.last_veto = None
        self.target_turn_rad = fo.THETA_MAX_RAD

    @property
    def release_turn_rad(self) -> float:
        """The turn at which this cycle's release fires."""

        return self.target_turn_rad - fo.RELEASE_MARGIN_RAD

    def process_phase(self) -> str:
        if not self.active:
            return "policy"
        return "return_to_reset" if self.phase_index == 4 else "follow_waypoints"

    def turn_progress_rad(self, quaternion) -> float:
        # The yaw below is measured about the reset grasp frame's own z, and at the threading
        # grip that axis points DOWN (grasp z . world z = -0.993), because the hand reaches
        # down onto the bolt. A tightening turn -- clockwise seen from above, the negative
        # direction about world z, the one training scores as positive progress -- therefore
        # already reads POSITIVE here. The former -1.0 factor flipped it, so a tightening
        # 55-degree turn reported -55 and `progress >= radians(55)` could never fire: the
        # rollout stayed in the `policy` phase for the whole run. Verified numerically against
        # the training reset pose: a -55 deg world-z turn now reads +54.5 deg (the 0.5 deg
        # shortfall is the ~7 deg tilt between grasp z and the bolt axis).
        relative = fo.matrix_from_quat(fo.quat_conjugate(self.reset_quaternion)) @ fo.matrix_from_quat(
            quaternion
        )
        yaw = math.atan2(relative[1, 0], relative[0, 0])
        delta = math.atan2(math.sin(yaw - self._previous_yaw), math.cos(yaw - self._previous_yaw))
        self._unwrapped_yaw += delta
        self._previous_yaw = yaw
        return self._unwrapped_yaw - self._cycle_yaw_origin

    def return_errors(self, position, quaternion, hand) -> tuple[float, float, float]:
        """``(metres, degrees, radians)`` from the reset grasp pose and hand posture."""

        position_error = float(np.linalg.norm(np.asarray(position) - self.reset_position))
        alignment = min(1.0, abs(float(np.dot(quaternion, self.reset_quaternion))))
        orientation_error = math.degrees(2.0 * math.acos(alignment))
        hand_error = float(np.max(np.abs(np.asarray(hand) - self.reset_hand)))
        return position_error, orientation_error, hand_error

    def _returned(self, position, quaternion, hand) -> bool:
        position_error, orientation_error, hand_error = self.return_errors(
            position, quaternion, hand
        )
        return position_error <= 0.005 and orientation_error <= 5.0 and hand_error <= 0.15

    def update(
        self,
        *,
        position,
        quaternion,
        hand,
        measured_turn_progress=None,
        measured_total_turn=None,
        fingertip_separation_m=None,
    ) -> tuple[str | None, float]:
        # The proxy is always advanced and recorded, even when it is not what
        # gates the release: it is the only turn signal hardware has, so how far
        # it drifts from the simulated nut's real twist is worth measuring.
        previous_unwrapped = self._unwrapped_yaw
        self.last_proxy_turn_rad = self.turn_progress_rad(quaternion)
        self.last_proxy_step_rad = self._unwrapped_yaw - previous_unwrapped
        progress = (
            self.last_proxy_turn_rad
            if measured_turn_progress is None
            else float(measured_turn_progress)
        )
        self.gated_on_measured_turn = measured_turn_progress is not None
        event = None
        if self.active:
            self.phase_steps += 1
            duration = math.ceil((0.7 if self.phase_index < 4 else 1.0) * self.rate_hz)
            if self.phase_steps >= duration:
                if self.phase_index < 4:
                    self.phase_index += 1
                    self.phase_steps = 0
                    self.wait_steps = 0
                elif self._returned(position, quaternion, hand):
                    self.completed_cycles += 1
                    self.active = False
                    self.phase_index = -1
                    self.phase_steps = 0
                    self.wait_steps = 0
                    self._cycle_yaw_origin = self._unwrapped_yaw
                    self.target_turn_rad = (
                        fo.THETA_MAX_RAD
                        if measured_total_turn is None
                        else fo.next_cycle_target_turn_rad(float(measured_total_turn))
                    )
                    event = "cycle_completed"
                else:
                    self.wait_steps += 1
                    if self.wait_steps >= math.ceil(16.0 * self.rate_hz):
                        self.active = False
                        self.failed = True
                        event = "return_failed"
        if (
            event is None
            and not self.active
            and not self.failed
            and self.completed_cycles < self.max_cycles
            and progress >= self.release_turn_rad
        ):
            veto = None if self.gated_on_measured_turn else self._proxy_veto(fingertip_separation_m)
            if veto is not None:
                self.proxy_vetoes += 1
                self.last_veto = veto
            else:
                self.active = True
                self.phase_index = 0
                self.phase_steps = 0
                self.wait_steps = 0
                event = "release_started"
        return event, progress

    def _proxy_veto(self, fingertip_separation_m) -> str | None:
        """Why the proxy's release turn must not be believed this tick, or None."""

        if (
            self.min_fingertip_separation_m is not None
            and fingertip_separation_m is not None
            and float(fingertip_separation_m) < self.min_fingertip_separation_m
        ):
            return (
                f"fingertips {1.0e3 * float(fingertip_separation_m):.1f} mm apart "
                f"(< {1.0e3 * self.min_fingertip_separation_m:.0f} mm): the pinch is "
                "empty and the grasp frame degenerate"
            )
        if (
            self.max_proxy_step_rad is not None
            and abs(self.last_proxy_step_rad) > self.max_proxy_step_rad
        ):
            return (
                f"proxy moved {math.degrees(self.last_proxy_step_rad):+.1f} deg in one tick "
                f"(limit {math.degrees(self.max_proxy_step_rad):.0f} deg): a collapsing "
                "grasp frame, not a turn"
            )
        return None


class ThreadPairClient:
    """The simulated nut's measured turn, and the clamp the transition needs.

    ros-sim's scene carries ForgeUltra's thread pair, and
    ``inspire_franka_sim``'s thread pair plugin publishes its state on
    ``/thread_state`` and clamps it on ``/thread_hold``. That replaces the
    grasp-frame yaw proxy with the quantity training actually scores -- the nut's
    own twist -- which matters because the proxy cannot distinguish a nut turn
    from the wrist yaw of the descent, and jumps ~100 degrees in one step when
    the fingertips converge.

    The physical workcell has no such topic, so hardware keeps the proxy and
    never constructs this. Both numbers are recorded in ros-sim, so how far
    apart they run is finally measurable.
    """

    def __init__(self, node, *, state_topic="/thread_state", hold_topic="/thread_hold",
                 reset_service="/reset_thread"):
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool
        from std_srvs.srv import Trigger

        # The plugin subscribes with TRANSIENT_LOCAL so a late or restarted
        # simulator still learns the current hold state. A VOLATILE publisher is
        # incompatible with that and DDS drops the connection silently -- it
        # reports "requesting incompatible QoS: DURABILITY" once and then no hold
        # ever arrives, which looks exactly like a thread that will not clamp.
        hold_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._node = node
        self._lock = threading.Lock()
        self._twist_rad = None
        self._axial_m = None
        self._reference_rad = 0.0
        self._held = None
        self._Bool = Bool
        self._hold_publisher = node.create_publisher(Bool, hold_topic, hold_qos)
        self._reset_client = node.create_client(Trigger, reset_service)
        node.create_subscription(
            JointState, state_topic, self._put, qos_profile_sensor_data
        )

    def _put(self, message):
        try:
            axial = message.position[list(message.name).index("nut_axial")]
            twist = message.position[list(message.name).index("nut_twist")]
        except (ValueError, IndexError):
            return
        with self._lock:
            self._axial_m = float(axial)
            self._twist_rad = float(twist)

    def wait_for_state(self, timeout_s: float) -> bool:
        deadline = self._node.now_s() + float(timeout_s)
        while self._node.now_s() < deadline:
            with self._lock:
                if self._twist_rad is not None:
                    return True
            time.sleep(0.02)
        with self._lock:
            return self._twist_rad is not None

    @property
    def axial_m(self):
        with self._lock:
            return self._axial_m

    @property
    def twist_rad(self):
        with self._lock:
            return self._twist_rad

    def turn_progress_rad(self):
        """Progress since the last rebase, in training's sign convention."""

        with self._lock:
            twist = self._twist_rad
            reference = self._reference_rad
        if twist is None:
            return None
        return fo.THREADING_DIRECTION_SIGN * (twist - reference)

    def total_turn_rad(self):
        """Turn since the run started, training's sign: the reset puts the nut at 0."""

        with self._lock:
            twist = self._twist_rad
        if twist is None:
            return None
        return fo.THREADING_DIRECTION_SIGN * twist

    def rebase(self) -> None:
        """Start a new cycle's turn count here, as ``rebase_turn_reference`` does."""

        with self._lock:
            if self._twist_rad is not None:
                self._reference_rad = self._twist_rad

    def hold(self, engaged: bool) -> None:
        """Clamp or free the thread. Idempotent, so it is safe to call per event."""

        engaged = bool(engaged)
        if self._held == engaged:
            return
        message = self._Bool()
        message.data = engaged
        self._hold_publisher.publish(message)
        self._held = engaged

    def reset(self, timeout_s: float = 5.0) -> bool:
        """Put the nut back at its start pose on the bolt, for a repeat run.

        The simulator is long-lived -- several rollouts share one
        sim_policy.launch.py -- so without this a second run would start on a
        nut the first one already turned.
        """

        if not self._reset_client.wait_for_service(timeout_sec=timeout_s):
            return False
        from std_srvs.srv import Trigger

        future = self._reset_client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        self._held = False
        with self._lock:
            self._reference_rad = 0.0
        return future.done() and bool(future.result() and future.result().success)


#: Field order of ``/fr3_policy_rollout/turn``. A JointState so rqt_plot and
#: PlotJuggler can plot each entry by name while the run streams.
TURN_MONITOR_FIELDS = (
    "nut_turn_deg",
    "grasp_yaw_proxy_deg",
    "gate_turn_deg",
    "fingertip_separation_mm",
    "index_joint_rad",
    "phase_index",
)
PHASE_INDEX = {"policy": 0.0, "follow_waypoints": 1.0, "return_to_reset": 2.0}


def turn_monitor_values(
    *, nut_turn_rad, proxy_turn_rad, gate_turn_rad, fingertip_separation_m, index_rad, phase
) -> list[float]:
    """One tick of the turn signals, in ``TURN_MONITOR_FIELDS`` order.

    The student has no nut-angle output; the only nut rotation in the loop is
    the one the coordinator gates the release on, which becomes the policy's
    phase input. ``nut_turn_rad`` is the simulated nut's measured turn (None on
    hardware, published as NaN), ``proxy_turn_rad`` the grasp-frame yaw that
    hardware has to use instead, and ``gate_turn_rad`` whichever of the two
    the coordinator actually compared with its release turn.
    """

    def degrees(value):
        return math.nan if value is None else math.degrees(float(value))

    return [
        degrees(nut_turn_rad),
        degrees(proxy_turn_rad),
        degrees(gate_turn_rad),
        math.nan if fingertip_separation_m is None else 1.0e3 * float(fingertip_separation_m),
        float(index_rad),
        PHASE_INDEX[phase],
    ]


@dataclass(frozen=True)
class LiveSample:
    arm_position: np.ndarray
    arm_velocity: np.ndarray
    hand_position: np.ndarray
    hand_velocity: np.ndarray
    rgb: np.ndarray
    depth: np.ndarray
    depth_units: str
    controller_target_position: np.ndarray
    controller_target_quaternion: np.ndarray
    controller_measured_position: np.ndarray
    controller_measured_quaternion: np.ndarray
    # Whether the controller's per-cycle clip of the last policy goal bound, at
    # its last state publish.
    controller_position_clipped: bool
    controller_orientation_clipped: bool
    sample_time_s: float
    # A force-observing student's inputs: the arm torque (the low-passed
    # external-torque estimate on hardware; in ros-sim the law's own task plus
    # nullspace torque, without the coriolis feed-forward ForgeUltra's OSC does
    # not have, gravity off as training's asset has it), and the zeroed pad
    # forces when a force topic is subscribed.
    arm_joint_torque_nm: np.ndarray | None = None
    fingertip_force_n: np.ndarray | None = None
    # Every recorded torque candidate by name (arm_torque_candidates), and the
    # model gravity the measured one had subtracted.
    arm_torque_candidates_nm: dict[str, np.ndarray] | None = None
    arm_torque_gravity_nm: np.ndarray | None = None


def _hardware_node_class():
    """Build the Node subclass only after ROS is known to be available."""

    from rclpy.node import Node

    class HardwarePolicyNode(Node):
        def __init__(
            self,
            calibration,
            config,
            *,
            hand_command_topic,
            hand_state_topic,
            max_state_age_s,
            max_frame_skew_s,
            use_sim_time=False,
            thumb_overlay=True,
            fingertip_force_topic=None,
            arm_torque_source=None,
            arm_torque_topics=None,
            arm_torque_lowpass_s=ARM_TORQUE_LOWPASS_S,
        ):
            from control_msgs.msg import JointTrajectoryControllerState
            from diagnostic_msgs.msg import DiagnosticArray
            from franka_forge_cartesian_impedance_msgs.msg import CartesianReplayState, PolicyGoal
            from realsense2_camera_msgs.msg import RGBD
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Empty
            import tf2_ros

            from rclpy.parameter import Parameter

            super().__init__(
                "fr3_policy_rollout",
                parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, use_sim_time)],
            )
            self.use_sim_time = bool(use_sim_time)
            # Whether the hand driver runs its thumb-yaw overlay, i.e. whether
            # its reported pose has to be inverted into the policy's logical
            # coordinate (physical_hand_state_to_policy).
            self.thumb_overlay = bool(thumb_overlay)
            self.last_tip_separation_m = None
            self.calibration = calibration
            self.config = config
            self.max_state_age_s = float(max_state_age_s)
            self.max_frame_skew_s = float(max_frame_skew_s)
            self._rgbd = RgbdFrameSynchronizer(self.max_frame_skew_s)
            # The grasp frame is rebuilt in the training asset's hand geometry,
            # not the workspace URDF's; see TrainingHandKinematics.
            self._training_hand = TrainingHandKinematics()
            self._policy_hand_position = None
            controller = "/" + config["cartesian"]["controller_name"].strip("/")
            # Preclipped goals plus the live grasp frame; the controller does
            # ForgeUltra's clip and runs its law at that frame (A3, A4).
            self.policy_publisher = self.create_publisher(
                PolicyGoal, controller + "/policy_goal", 1
            )
            self.abort_publisher = self.create_publisher(
                Empty, controller + "/abort", 1
            )
            self.hand_publisher = self.create_publisher(
                JointState, hand_command_topic, 10
            )
            self.turn_publisher = self.create_publisher(
                JointState, "/fr3_policy_rollout/turn", 10
            )
            self._lock = threading.Lock()
            self._arm = self._hand = None
            self._camera_info = self._cartesian = self._controller_status = None
            self._fingertip_force = None
            self._receive_time = {}
            # Subtracted from a raw (driver) force reading; see tare_fingertip_force.
            self.fingertip_force_zero_n = np.zeros(2)
            self.fingertip_force_topic = fingertip_force_topic
            if fingertip_force_topic:
                self.create_subscription(
                    JointState,
                    fingertip_force_topic,
                    lambda message: self._put("fingertip_force", message),
                    qos_profile_sensor_data,
                )
            # None: a student without force inputs, which records no torques.
            # Otherwise the broadcaster candidates in arm_torque_topics are
            # subscribed, and the controller ones come with ~/cartesian_state.
            self.arm_torque_source = arm_torque_source
            self.arm_torque_topics = dict(arm_torque_topics or {})
            self._arm_torques = {name: None for name in self.arm_torque_topics}
            self._arm_torque_filters = {
                name: TorqueLowPass(arm_torque_lowpass_s) for name in self.arm_torque_topics
            }
            for name, topic in self.arm_torque_topics.items():
                self.create_subscription(
                    JointState,
                    topic,
                    lambda message, name=name: self._put_arm_torque(name, message),
                    qos_profile_sensor_data,
                )
            self.create_subscription(
                JointTrajectoryControllerState,
                controller + "/controller_state",
                lambda message: self._put("arm", message),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                CartesianReplayState,
                controller + "/cartesian_state",
                lambda message: self._put("cartesian", message),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                DiagnosticArray,
                controller + "/status",
                self._put_controller_status,
                10,
            )
            self.create_subscription(
                JointState,
                hand_state_topic,
                lambda message: self._put("hand", message),
                qos_profile_sensor_data,
            )
            self.rgbd_subscription = self.create_subscription(
                RGBD,
                calibration.rgbd_topic,
                self._put_rgbd,
                qos_profile_sensor_data,
            )
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self._camera_checked = False
            self.grasp_z_transport = None

        def _put(self, name, message):
            with self._lock:
                setattr(self, "_" + name, message)
                self._receive_time[name] = self.now_s()

        def _put_arm_torque(self, name, message):
            # Filter every message, not the 15 Hz samples, so the low-pass sees
            # the stream's own rate.
            torque = external_joint_torque_message_to_array(message)
            received = self.now_s()
            with self._lock:
                self._arm_torques[name] = self._arm_torque_filters[name].update(torque, received)
                self._receive_time["torque_" + name] = received

        def _put_rgbd(self, message):
            received = self.now_s()
            with self._lock:
                self._rgbd.add("rgb", message.rgb, received)
                self._rgbd.add("depth", message.depth, received)
                self._camera_info = message.rgb_camera_info
                self._receive_time["camera_info"] = received

        def _put_controller_status(self, message):
            if not message.status:
                return
            values = {item.key: item.value for item in message.status[0].values}
            with self._lock:
                self._controller_status = values
                self._receive_time["status"] = self.now_s()

        def now_s(self) -> float:
            """Freshness and pacing clock: wall time on hardware, /clock in ros-sim.

            A slowed simulator (sim_speed_factor < 1) then gives the policy its
            trained period and input ages in simulated time, however long
            inference takes on the shared CPU.
            """

            if self.use_sim_time:
                return self.get_clock().now().nanoseconds * 1.0e-9
            return time.monotonic()

        def sleep_until(self, deadline_s: float) -> None:
            if not self.use_sim_time:
                remaining = deadline_s - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)
                return
            while self.now_s() < deadline_s:
                time.sleep(0.002)

        def wait_for_clock(self, timeout_s: float) -> None:
            deadline = time.monotonic() + timeout_s
            while self.use_sim_time and self.now_s() <= 0.0:
                if time.monotonic() > deadline:
                    raise TimeoutError("no /clock from the simulator")
                time.sleep(0.02)

        def ready(self) -> bool:
            with self._lock:
                return (
                    self._rgbd.latest_pair() is not None
                    and all(
                        value is not None
                        for value in (
                            self._arm,
                            self._cartesian,
                            self._hand,
                            self._camera_info,
                            self._controller_status,
                        )
                    )
                    and (not self.fingertip_force_topic or self._fingertip_force is not None)
                    and all(value is not None for value in self._arm_torques.values())
                )

        def wait_ready(self, timeout_s: float) -> None:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if self.ready() and self.policy_publisher.get_subscription_count() > 0:
                    return
                time.sleep(0.02)
            if self.rgbd_subscription.get_publisher_count() == 0:
                raise TimeoutError(
                    f"no composite RGB-D publisher on {self.calibration.rgbd_topic}; "
                    "start RealSense with enable_rgbd:=true, enable_sync:=true, "
                    "and align_depth.enable:=true"
                )
            raise TimeoutError("policy inputs/controller subscription did not become ready")

        def sample(self) -> LiveSample:
            with self._lock:
                pair = self._rgbd.latest_pair()
                messages = {
                    name: getattr(self, "_" + name)
                    for name in (
                        "arm",
                        "cartesian",
                        "hand",
                        "camera_info",
                        *(("fingertip_force",) if self.fingertip_force_topic else ()),
                    )
                }
                messages.update(
                    {"torque_" + name: value for name, value in self._arm_torques.items()}
                )
                receive = dict(self._receive_time)
                if pair is not None:
                    messages["rgb"], messages["depth"] = pair[:2]
                    receive["rgb"], receive["depth"] = pair[2:]
                else:
                    messages["rgb"] = messages["depth"] = None
            now = self.now_s()
            missing = [name for name, value in messages.items() if value is None]
            if missing:
                raise RuntimeError("missing live inputs: " + ", ".join(missing))
            stale = [
                name
                for name in (
                    "arm",
                    "cartesian",
                    "hand",
                    "rgb",
                    "depth",
                    *(("fingertip_force",) if self.fingertip_force_topic else ()),
                    *("torque_" + name for name in self.arm_torque_topics),
                )
                if now - receive[name] > self.max_state_age_s
            ]
            if stale:
                raise RuntimeError("stale live inputs: " + ", ".join(stale))
            rgb_stamp = _stamp_seconds(messages["rgb"])
            depth_stamp = _stamp_seconds(messages["depth"])
            if abs(rgb_stamp - depth_stamp) > self.max_frame_skew_s:
                raise RuntimeError(
                    f"RGB/depth timestamp skew {abs(rgb_stamp-depth_stamp):.4f} s "
                    f"exceeds {self.max_frame_skew_s:.4f} s"
                )
            if not self._camera_checked:
                info = messages["camera_info"]
                assert_policy_camera_frames(
                    messages["rgb"], messages["depth"], info, self.calibration.frame_id
                )
                self.calibration.assert_live_camera_info(info.width, info.height, info.k)
                self._camera_checked = True

            arm = messages["arm"]
            arm_columns = {name: index for index, name in enumerate(arm.joint_names)}
            if any(name not in arm_columns for name in ARM_JOINTS):
                raise RuntimeError("controller state does not contain all FR3 joints")
            indices = [arm_columns[name] for name in ARM_JOINTS]
            q_arm = np.asarray(arm.feedback.positions, dtype=float)[indices]
            dq_arm = np.asarray(arm.feedback.velocities, dtype=float)[indices]
            q_hand, dq_hand = hand_joint_state_to_arrays(messages["hand"])
            rgb, _ = image_message_to_numpy(messages["rgb"])
            depth, depth_units = image_message_to_numpy(messages["depth"])
            controller_pose = messages["cartesian"].target
            controller_target_position = np.array(
                [
                    controller_pose.position.x,
                    controller_pose.position.y,
                    controller_pose.position.z,
                ],
                dtype=float,
            )
            controller_target_quaternion = np.array(
                [
                    controller_pose.orientation.w,
                    controller_pose.orientation.x,
                    controller_pose.orientation.y,
                    controller_pose.orientation.z,
                ],
                dtype=float,
            )
            cartesian = messages["cartesian"]
            arm_torque_candidates = None
            arm_joint_torque = None
            if self.arm_torque_source is not None:
                arm_torque_candidates = arm_torque_candidates_from_controller(cartesian)
                for name in self.arm_torque_topics:
                    arm_torque_candidates[name] = messages["torque_" + name]
                arm_joint_torque = arm_torque_candidates[self.arm_torque_source]
            fingertip_force = None
            if self.fingertip_force_topic:
                fingertip_force, raw = fingertip_force_message_to_newtons(
                    messages["fingertip_force"]
                )
                if raw:
                    fingertip_force = fingertip_force - self.fingertip_force_zero_n
            controller_measured_pose = messages["cartesian"].measured
            controller_measured_position = np.array(
                [
                    controller_measured_pose.position.x,
                    controller_measured_pose.position.y,
                    controller_measured_pose.position.z,
                ],
                dtype=float,
            )
            controller_measured_quaternion = np.array(
                [
                    controller_measured_pose.orientation.w,
                    controller_measured_pose.orientation.x,
                    controller_measured_pose.orientation.y,
                    controller_measured_pose.orientation.z,
                ],
                dtype=float,
            )
            sample = LiveSample(
                arm_position=q_arm,
                arm_velocity=dq_arm,
                hand_position=q_hand,
                hand_velocity=dq_hand,
                rgb=rgb,
                depth=depth,
                depth_units=depth_units,
                controller_target_position=controller_target_position,
                controller_target_quaternion=controller_target_quaternion,
                controller_measured_position=controller_measured_position,
                controller_measured_quaternion=controller_measured_quaternion,
                controller_position_clipped=bool(messages["cartesian"].position_clipped),
                controller_orientation_clipped=bool(
                    messages["cartesian"].orientation_clipped
                ),
                sample_time_s=now,
                arm_joint_torque_nm=arm_joint_torque,
                fingertip_force_n=fingertip_force,
                arm_torque_candidates_nm=arm_torque_candidates,
                arm_torque_gravity_nm=(
                    None
                    if arm_torque_candidates is None
                    else np.asarray(cartesian.tau_gravity, dtype=float)
                ),
            )
            # The grasp frame is rebuilt from these logical hand joints rather
            # than read from TF, so keep the latest ones for _hand_frames.
            self._policy_hand_position = physical_hand_state_to_policy(
                sample.hand_position, sample.hand_velocity, thumb_overlay=self.thumb_overlay
            )[0]
            return sample

        def wait_for_fresh_sample(self, timeout_s: float) -> LiveSample:
            """Wait through transient callback gaps without hiding contract errors."""

            deadline = time.monotonic() + timeout_s
            last_error = None
            while time.monotonic() < deadline:
                try:
                    return self.sample()
                except RuntimeError as exc:
                    message = str(exc)
                    if not message.startswith(("missing live inputs:", "stale live inputs:")):
                        raise
                    last_error = exc
                    time.sleep(0.01)
            raise TimeoutError(
                "fresh policy inputs did not arrive"
                + ("" if last_error is None else f": {last_error}")
            )

        def tare_fingertip_force(self, samples: int = 15, timeout_s: float = 5.0) -> np.ndarray | None:
            """Zero a raw (driver) pad force reading; call with the pads touching nothing.

            Training's sensor reads exactly zero out of contact. The RH56's does
            not: its zero moves after a heavy load (inspire_hand_driver's
            driver_node docstring has -11 g before, +219 g after). Returns the
            zero in N, or None when the source is already projected (ros-sim).
            """

            readings = []
            deadline = time.monotonic() + timeout_s
            last_stamp = None
            while len(readings) < samples:
                if time.monotonic() > deadline:
                    raise TimeoutError("fingertip force did not arrive for the tare")
                with self._lock:
                    message = self._fingertip_force
                if message is not None and _stamp_seconds(message) != last_stamp:
                    last_stamp = _stamp_seconds(message)
                    value, raw = fingertip_force_message_to_newtons(message)
                    if not raw:
                        return None
                    readings.append(value)
                time.sleep(0.01)
            self.fingertip_force_zero_n = np.mean(readings, axis=0)
            return self.fingertip_force_zero_n

        def controller_status(self) -> dict[str, str]:
            with self._lock:
                status = (
                    None
                    if self._controller_status is None
                    else dict(self._controller_status)
                )
                received = self._receive_time.get("status", 0.0)
            if status is None or self.now_s() - received > 1.0:
                raise RuntimeError(
                    "Cartesian replay controller feedback stopped; check the hardware log"
                )
            return status

        def policy_hand_state(self, sample: LiveSample) -> tuple[np.ndarray, np.ndarray]:
            return physical_hand_state_to_policy(
                sample.hand_position, sample.hand_velocity, thumb_overlay=self.thumb_overlay
            )

        def policy_hand_positions(self) -> np.ndarray:
            """The three logical hand joints, from the hand topic alone.

            The grasp frame needs only the hand, so this deliberately does not
            require the arm, Cartesian and camera streams that ``sample``
            checks. The hand driver and TF are up well before the Cartesian
            controller starts publishing, and the reset grasp transport is
            captured in that window.
            """

            with self._lock:
                message = self._hand
            if message is None:
                raise RuntimeError("missing live inputs: hand")
            return physical_hand_state_to_policy(
                *hand_joint_state_to_arrays(message), thumb_overlay=self.thumb_overlay
            )[0]

        def _hand_frames(self):
            from rclpy.time import Time

            def lookup(child):
                transform = self.tf_buffer.lookup_transform("fr3_link0", child, Time())
                t = transform.transform.translation
                r = transform.transform.rotation
                return np.array([t.x, t.y, t.z]), np.array([r.w, r.x, r.y, r.z])

            flange_position, flange_quaternion = lookup("fr3_link8")
            # TF's thumb_tip/index_tip are posed from the physical joint state,
            # not the policy's logical one (thumb-yaw overlay, clamped thumb
            # followers). Rebuild the tips from the policy's own hand joints so
            # it sees and commands exactly its trained grasp frame.
            # Prefer the value that came with the sample the policy is about to
            # consume, so the grasp frame and the observation share an instant.
            # Before the first sample, read the hand topic on its own.
            hand = self._policy_hand_position
            if hand is None:
                hand = self.policy_hand_positions()
            thumb_flange, index_flange = self._training_hand.tips_in_flange(hand)
            rotation = fo.matrix_from_quat(flange_quaternion)
            thumb = flange_position + rotation @ thumb_flange
            index = flange_position + rotation @ index_flange
            return thumb, index, flange_position, flange_quaternion

        def capture_grasp_z_transport(self) -> np.ndarray:
            """Fix the grasp approach axis in the flange frame at the reset pose."""

            thumb, index, flange_position, flange_quaternion = self._hand_frames()
            self.grasp_z_transport = grasp_z_transport_at_reset(
                thumb, index, flange_position, flange_quaternion
            )
            return self.grasp_z_transport

        def grasp_pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if self.grasp_z_transport is None:
                raise RuntimeError("grasp frame used before its reset z transport was captured")
            thumb, index, _flange_position, flange_quaternion = self._hand_frames()
            grasp_position, grasp_quaternion = grasp_frame_from_tips(
                thumb, index, flange_quaternion, self.grasp_z_transport
            )
            return grasp_position, grasp_quaternion, flange_quaternion

        def grasp_pose_and_tool(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """The live grasp frame in the base frame, and the same frame on the flange.

            The third value is what ``PolicyGoal.tool_in_flange`` carries, so the
            controller measures, clips and differentiates its law at the live
            fingertip midpoint. Both come from one hand-frame evaluation.
            """

            if self.grasp_z_transport is None:
                raise RuntimeError("grasp frame used before its reset z transport was captured")
            thumb, index, flange_position, flange_quaternion = self._hand_frames()
            self.last_tip_separation_m = float(np.linalg.norm(index - thumb))
            grasp_position, grasp_quaternion = grasp_frame_from_tips(
                thumb, index, flange_quaternion, self.grasp_z_transport
            )
            tool_in_flange = grasp_in_flange(
                grasp_position, grasp_quaternion, flange_position, flange_quaternion
            )
            return grasp_position, grasp_quaternion, tool_in_flange

        def wait_for_policy_hand(self, target, timeout_s: float, tolerance: float, republish):
            """Block until the logical hand joints are within ``tolerance`` of ``target``."""

            target = np.asarray(target, dtype=float)
            deadline = time.monotonic() + timeout_s
            while True:
                try:
                    current = self.policy_hand_positions()
                except RuntimeError:
                    current = None
                if current is not None and np.max(np.abs(current - target)) <= tolerance:
                    return current
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "the Inspire hand did not reach the start posture "
                        f"{np.array2string(target, precision=3)}; last "
                        f"{None if current is None else np.array2string(current, precision=3)}"
                    )
                republish()
                time.sleep(0.2)

        def _wait_for_tf(self, function, timeout_s: float):
            deadline = time.monotonic() + timeout_s
            last_error = None
            while time.monotonic() < deadline:
                try:
                    return function()
                except Exception as exc:  # tf2 raises several lookup exception types
                    last_error = exc
                    time.sleep(0.02)
            raise TimeoutError(
                "arm-to-hand TF did not become ready"
                + ("" if last_error is None else f": {last_error}")
            )

        def wait_for_grasp_z_transport(self, timeout_s: float) -> np.ndarray:
            return self._wait_for_tf(self.capture_grasp_z_transport, timeout_s)

        def wait_for_grasp_pose(self, timeout_s: float):
            return self._wait_for_tf(self.grasp_pose, timeout_s)

        def publish_policy_goal(
            self, goal_position, goal_quaternion_wxyz, tool_in_flange, nullspace
        ):
            from franka_forge_cartesian_impedance_msgs.msg import PolicyGoal

            message = PolicyGoal()
            message.goal.position.x = float(goal_position[0])
            message.goal.position.y = float(goal_position[1])
            message.goal.position.z = float(goal_position[2])
            q = np.asarray(goal_quaternion_wxyz, dtype=float)
            message.goal.orientation.w = float(q[0])
            message.goal.orientation.x = float(q[1])
            message.goal.orientation.y = float(q[2])
            message.goal.orientation.z = float(q[3])
            message.tool_in_flange = [float(value) for value in tool_in_flange]
            message.nullspace_positions = [float(value) for value in nullspace]
            self.policy_publisher.publish(message)

        def publish_hand_target(self, pinch):
            from inspire_hand_driver import kinematics as kin
            from sensor_msgs.msg import JointState

            radians = np.concatenate(
                (SUPPORT_FINGER_TARGETS, [pinch[2], pinch[1], pinch[0]])
            )
            message = JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            message.name = list(HAND_JOINTS)
            message.position = [
                kin.rad_to_open_ratio(kin.dof_index(name), value)
                for name, value in zip(HAND_JOINTS, radians)
            ]
            self.hand_publisher.publish(message)

        def publish_turn(self, values):
            from sensor_msgs.msg import JointState

            message = JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            message.name = list(TURN_MONITOR_FIELDS)
            message.position = [float(value) for value in values]
            self.turn_publisher.publish(message)

        def abort(self):
            from std_msgs.msg import Empty

            self.abort_publisher.publish(Empty())

    return HardwarePolicyNode


def run_hardware_rollout(runner, calibration, args) -> dict:
    """Home through trajectory replay, switch controllers, and run the live policy."""

    readiness = assess_hardware_readiness(calibration)
    if not readiness.ready:
        details = "\n".join(f"- {item}" for item in readiness.blockers)
        raise RuntimeError(
            "physical policy execution is disabled by readiness checks:\n" + details
        )

    import rclpy
    from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
    from franka_trajectory_replay.cartesian_replay_client import CartesianReplayClient
    from franka_trajectory_replay.replay_client import Rejected
    from franka_trajectory_replay.runconfig import load_config
    from inspire_franka_trajectory_replay.replay import (
        CoordinatedReplayClient,
        load_home,
    )
    from .mujoco_threading_env import seed_action_history
    from .session import PolicyRolloutSession
    from utils.data_collection import RolloutDataCollector, create_rollout_dir

    sim = bool(getattr(args, "sim", False))
    # ros-sim's driver runs without the thumb-yaw overlay (sim_policy.launch.py);
    # the physical driver keeps it. --thumb-overlay on/off overrides.
    thumb_overlay = bool(getattr(args, "thumb_overlay_enabled", not sim))
    hand_start = str(getattr(args, "hand_start", "release"))
    config = load_config(args.config)
    home_arm, home_hand = load_home(args.home)
    if np.max(np.abs(home_arm - fo.FRANKA_ARM_RESET_JOINTS_M24)) > args.max_home_delta:
        raise ValueError("home arm pose does not match the policy's M24 reset pose")

    # Only a force-observing student subscribes, so the others run without a
    # force source. ros-sim's plugin publishes the projected channels; hardware
    # reads the RH56 driver.
    fingertip_force_topic = None
    if runner.config.force_observation == "included":
        fingertip_force_topic = getattr(args, "fingertip_force_topic", None) or (
            "/fingertip_force" if sim else "/inspire_hand/grip_force"
        )
    arm_torque_source = resolve_arm_torque_source(
        str(getattr(args, "arm_torque_source", "auto")), sim=sim
    )
    arm_torque_topics = {}
    if runner.config.force_observation == "included":
        arm_torque_topics = {
            name: topic
            for name, topic in FRANKA_TORQUE_TOPICS.items()
            if name in arm_torque_candidates(sim=sim)
        }
    else:
        arm_torque_source = None
    arm_torque_lowpass_s = float(getattr(args, "arm_torque_lowpass_s", ARM_TORQUE_LOWPASS_S))
    metadata = runner.metadata()
    run_dir = create_rollout_dir(args.recording_root, "ros_sim" if sim else "hardware")
    collector = RolloutDataCollector(
        run_dir,
        metadata={
            "checkpoint": metadata["checkpoint"],
            "checkpoint_sha256": metadata["sha256"],
            "checkpoint_weight_source": metadata["weight_source"],
            "flow_integration_steps": metadata["integration_steps"],
            "temporal_ensemble_decay": metadata["temporal_ensemble_decay"],
            "policy_device": metadata["device"],
            "camera_profile": str(calibration.source_path),
            "hand_velocity_observation": runner.config.hand_velocity_observation,
            "force_observation": runner.config.force_observation,
            "fingertip_force_topic": fingertip_force_topic,
            "arm_joint_torque_source": arm_torque_source,
            "arm_torque_candidates": (
                None if arm_torque_source is None else list(arm_torque_candidates(sim=sim))
            ),
            "arm_torque_topics": arm_torque_topics,
            "arm_torque_lowpass_s": arm_torque_lowpass_s,
            "collection_mode": (
                "ros_sim_closed_loop_student" if sim else "physical_closed_loop_student"
            ),
            "controller": config["cartesian"]["controller_name"],
            # Needed to rebuild the physical hand pose from the recorded
            # logical joints (analyze_rollout_motion) and to know how the
            # episode started.
            "thumb_abduction_overlay": thumb_overlay,
            "hand_start": hand_start,
        },
        record_rgbd=args.record_rgbd,
    )
    session = PolicyRolloutSession(runner, calibration, collector=collector)
    frame = TrainingFrameAdapter()

    # No rclpy signal handlers: with them, Ctrl-C shuts the ROS context down
    # before the handlers below run, so the abort never reaches the controller
    # and the recording is never written. Ctrl-C is then a plain
    # KeyboardInterrupt that the loop catches with ROS still up.
    from rclpy.signals import SignalHandlerOptions

    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    home_client = CoordinatedReplayClient(
        config, args.hand_topic, args.hand_state_topic, thumb_abduction_overlay=thumb_overlay
    )
    arm_client = CartesianReplayClient(config, node_name="policy_arm_setup")
    node = _hardware_node_class()(
        calibration,
        config,
        hand_command_topic=args.hand_topic,
        hand_state_topic=args.hand_state_topic,
        max_state_age_s=args.max_state_age,
        max_frame_skew_s=args.max_frame_skew,
        use_sim_time=sim,
        thumb_overlay=thumb_overlay,
        fingertip_force_topic=fingertip_force_topic,
        arm_torque_source=arm_torque_source,
        arm_torque_topics=arm_torque_topics,
        arm_torque_lowpass_s=arm_torque_lowpass_s,
    )
    debug_viz = None
    if getattr(args, "debug_viz", False):
        from .debug_viz import PolicyDebugPublisher

        debug_viz = PolicyDebugPublisher(
            node,
            calibration,
            runner.model.fusion.visual_encoder,
            publish_every=getattr(args, "debug_viz_every", 2),
        )
        print(
            "debug viz on: /fr3_policy_rollout/"
            "{dp3_points,scene_points,markers,dp3_overlay}\n"
            "  rqt_image_view /fr3_policy_rollout/dp3_overlay  "
            "(the sampled points in the camera view; nut candidates in red)",
            flush=True,
        )

    # The setup clients get their own executor, retired after homing. Every
    # policy-node callback only stores a message, and one spin thread for it
    # measured ~25 % faster and far less jittery than a shared five-thread
    # executor (loop p95 95 ms vs 176 ms against sim_policy.launch.py): the
    # extra threads mostly contend with inference for the GIL.
    setup_executor = MultiThreadedExecutor(num_threads=4)
    setup_executor.add_node(home_client)
    setup_executor.add_node(arm_client)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    setup_thread = threading.Thread(target=setup_executor.spin, daemon=True)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    setup_thread.start()
    spin_thread.start()
    status = "error"
    steps = 0
    missed_deadlines = 0
    clipped_policy_ticks = 0
    watchdog_stop = False
    live_preflight_s = []
    grasp_controlled_offset_m = None
    artifact = None
    error = None
    details = {
        "thumb_abduction_overlay": thumb_overlay,
        "hand_start": hand_start,
    }
    try:
        home_client.ensure_active(print)
        if not args.yes:
            input("Move the FR3 and Inspire hand to the policy reset pose? [Enter/Ctrl-C] ")
        home_client.goto(home_arm)
        home_client.wait_for_hand_link()
        home_client.command_hand(home_hand)
        home_client.wait_for_hand(
            home_hand, args.hand_timeout, args.hand_tolerance
        )
        print("policy homing complete")
        # The student returns to the grasp frame it was reset to in training,
        # so that frame -- not whatever pose the arm homed to -- is the return
        # gate's reference, and the homing has to put the live grasp frame on
        # it: joints 2/5/6/7 each move the grasp 3-6 mm per 0.01 rad.
        trained_position, trained_quaternion, trained_hand = trained_reset_grasp(
            node._training_hand, frame
        )
        details["trained_reset_grasp_base"] = {
            "position": [float(v) for v in trained_position],
            "quaternion_wxyz": [float(v) for v in trained_quaternion],
        }
        max_attempts = max(1, int(getattr(args, "home_attempts", 1)))
        home_command = np.asarray(home_arm, dtype=float).copy()
        for attempt in range(1, max_attempts + 1):
            z_transport = node.wait_for_grasp_z_transport(args.input_timeout)
            achieved_position, achieved_quaternion, _ = node.wait_for_grasp_pose(
                args.input_timeout
            )
            home_residual_m = float(np.linalg.norm(achieved_position - trained_position))
            alignment = min(1.0, abs(float(np.dot(achieved_quaternion, trained_quaternion))))
            home_residual_deg = math.degrees(2.0 * math.acos(alignment))
            joint_residual = np.asarray(home_client.current_joint_positions(), dtype=float)[:7] - home_arm
            print(
                f"homing attempt {attempt}: live grasp frame {1.0e3 * home_residual_m:.1f} mm / "
                f"{home_residual_deg:.2f} deg from the trained reset grasp "
                f"(base {np.array2string(trained_position, precision=4)}); joint residual "
                f"{np.array2string(joint_residual, precision=4)} rad",
                flush=True,
            )
            if home_residual_m <= args.max_home_grasp_error or attempt == max_attempts:
                break
            # The joint-impedance homing is a PD law: it settles where its error
            # torque balances static friction and any gravity the Desk load does
            # not compensate, and re-sending the same goal from rest lands on
            # the same spot (2026-10-26: 12.3 mm on all three attempts, joint 7
            # 0.056 rad short at 12.5 N m/rad). Shift the goal by the residual
            # instead, so the balance point is the reset pose.
            home_command = next_home_command(home_command, home_arm, joint_residual)
            home_client.goto(home_command)
            time.sleep(float(getattr(args, "home_settle_s", 1.0)))
        details["home_grasp_residual_m"] = home_residual_m
        details["home_grasp_residual_deg"] = home_residual_deg
        details["home_joint_residual_rad"] = [float(v) for v in joint_residual]
        details["home_command_offset_rad"] = [float(v) for v in home_command - home_arm]
        details["achieved_home_grasp_base"] = {
            "position": [float(v) for v in achieved_position],
            "quaternion_wxyz": [float(v) for v in achieved_quaternion],
        }
        if home_residual_m > args.max_home_grasp_error:
            raise Rejected(
                f"homing left the grasp frame {1.0e3 * home_residual_m:.1f} mm from the "
                f"trained reset grasp (limit {1.0e3 * args.max_home_grasp_error:.1f} mm, "
                f"--max-home-grasp-error); the return gate is 5 mm, so the student "
                "could not pass it from here. Improve the homing before overriding."
            )
        print(
            "grasp approach axis in the flange frame: "
            + np.array2string(z_transport, precision=4),
            flush=True,
        )

        # The policy controls the training grasp midpoint, not the replay TCP
        # that `config` carries; see POLICY_TOOL_OFFSET_XYZ.
        tool = policy_tool_transform(policy_tool_offset_config())
        arm_client.check_tool(tool, print)
        if sim:
            # MuJoCo has no franka_hardware robot model or FrankaRobotState: the
            # controller runs on its DH model and there is no F_T_EE to check.
            if not arm_client.uses_dh_model():
                raise Rejected("ros-sim expects the simulator profile (model_source=dh)")
        else:
            if arm_client.uses_dh_model():
                raise Rejected("physical rollout requires model_source=franka, not dh")
            arm_client.preflight(home_client.current_joint_positions(), print)
            # The load configured in Franka Desk is what libfranka compensates
            # for. With a bare-flange load the OSC spring carries the hand's
            # weight and holds short of every goal; the 2026-10-16 run held
            # 9-12 mm short in return_to_reset.
            details["configured_load"] = _configured_load(arm_client, args.input_timeout, print)
            expected_mass = getattr(args, "expected_load_mass_kg", None)
            if expected_mass is not None:
                load = details["configured_load"]
                if load is None:
                    raise Rejected("no FrankaRobotState to check the configured load mass against")
                # libfranka compensates the total: Desk may carry the hand as the
                # end-effector profile or as an extra load, and both are right.
                if abs(load["total_mass_kg"] - float(expected_mass)) > 0.1:
                    raise Rejected(
                        f"Franka Desk mass beyond the flange is {load['total_mass_kg']:.3f} kg "
                        f"(end effector {load['end_effector_mass_kg']:.3f} + load "
                        f"{load['load_mass_kg']:.3f}), expected {float(expected_mass):.3f} kg "
                        "(--expected-load-mass-kg): configure the hand's mass and centre of "
                        "mass in Desk before running the policy"
                    )
        arm_client.ensure_active(print)
        parameters = arm_client.controller_parameters()
        # ForgeUltra clips per axis with one threshold vector; the controller takes a
        # scalar per kind, so the training thresholds have to be uniform for the two
        # to be the same clip.
        if not (
            np.all(fo.POS_ACTION_THRESHOLD == fo.POS_ACTION_THRESHOLD[0])
            and np.all(fo.ROT_ACTION_THRESHOLD == fo.ROT_ACTION_THRESHOLD[0])
        ):
            raise Rejected("training's action thresholds are not uniform per axis")
        expected = {
            "translational_stiffness": float(fo.DEFAULT_TASK_PROP_GAINS[0]),
            "rotational_stiffness": float(fo.DEFAULT_TASK_PROP_GAINS[3]),
            "nullspace_stiffness": fo.KP_NULL,
            # The exact nullspace projector and ForgeUltra's rotation error; with the
            # example's lambda 0.2 the joint spring leaks more force onto the tool than the
            # policy itself commands, and the quaternion-vector error halves the rotational
            # stiffness. See controllers_sim_policy.yaml.
            "nullspace_damping_lambda": 0.0,
            "stiffness_scale": 1.0,
            "target_filter": 1.0,
            # compute_dof_torque's clamp, and its per-substep target clip run in the
            # controller at its rate (A4): the training thresholds, in the training world
            # frame, which is the base yawed by pi.
            "torque_limit": fo.ARM_TORQUE_LIMIT,
            "policy_clip_position_m": float(fo.POS_ACTION_THRESHOLD[0]),
            "policy_clip_orientation_rad": float(fo.ROT_ACTION_THRESHOLD[0]),
            "policy_clip_frame_yaw": frame.world_from_base_yaw,
            "torque_rate_limit": 1.0,
            "policy_command_timeout": 0.5,
            "state_publish_rate": 50.0,
        }
        mismatches = [
            f"{name}={parameters.get(name)!r} (expected {value})"
            for name, value in expected.items()
            if not math.isclose(
                float(parameters.get(name, float("nan"))), value, rel_tol=1e-6
            )
        ]
        if str(parameters.get("rotation_error")) != "axis_angle":
            mismatches.append(
                f"rotation_error={parameters.get('rotation_error')!r} (expected 'axis_angle')"
            )
        # A1b. Without the mass weighting the nullspace term is still a projector but not
        # Forge's: the joint PD acts on torques instead of accelerations.
        if parameters.get("mass_weighted_nullspace") is not True:
            mismatches.append(
                f"mass_weighted_nullspace={parameters.get('mass_weighted_nullspace')!r} "
                "(expected True)"
            )
        # compute_dof_torque has no Coriolis term; the PhysX arm felt its own
        # uncompensated, and so must this one. (Gravity is libfranka's, as it was
        # Isaac's disable_gravity.)
        if parameters.get("coriolis_compensation") is not False:
            mismatches.append(
                f"coriolis_compensation={parameters.get('coriolis_compensation')!r} "
                "(expected False: training's law has no Coriolis term)"
            )
        if mismatches:
            raise Rejected(
                "controller is not using the policy profile: " + "; ".join(mismatches)
            )

        # Homing and setup are complete. Keeping these helper nodes spinning
        # would continue deserializing joint/robot state that
        # the policy loop never reads. The main node owns the live state and
        # controller-status subscriptions from this point onward.
        setup_executor.shutdown()
        setup_thread.join(timeout=5.0)
        node.wait_for_clock(args.input_timeout)
        node.wait_ready(args.input_timeout)

        # Synthetic tensors do not exercise every content-dependent point-cloud
        # path. Run one warm-up and four measured real-frame passes before any
        # policy command. Physical execution requires the whole policy pass to
        # fit inside its actual command period, not merely the much looser
        # controller watchdog.
        preflight_session = PolicyRolloutSession(runner, calibration)
        for _ in range(5):
            sample = node.wait_for_fresh_sample(args.input_timeout)
            grasp_position, grasp_quaternion, _ = node.wait_for_grasp_pose(
                args.input_timeout
            )
            q_policy_hand, dq_policy_hand = node.policy_hand_state(sample)
            preflight_session.reset(
                previous_filtered_native_action=np.zeros(9), seed=args.seed
            )
            preflight_started = node.now_s() if sim else time.perf_counter()
            preflight_result = preflight_session.step(
                joint_position=np.concatenate((sample.arm_position, q_policy_hand)),
                joint_velocity=np.concatenate((sample.arm_velocity, dq_policy_hand)),
                rgb=sample.rgb,
                depth=sample.depth,
                depth_units=sample.depth_units,
                trajectory_progress=(
                    0.0 if runner.config.trajectory_progress_conditioning else None
                ),
                process_phase=(
                    "policy" if runner.config.cyclic_process_phase_conditioning else None
                ),
                fingertip_force_n=sample.fingertip_force_n,
                arm_joint_torque_nm=sample.arm_joint_torque_nm,
            )
            # The gate has to measure the tick the loop will actually run, not a
            # cheaper one: --debug-viz renders and publishes inside the loop, so
            # it is timed here too. force=True because the gate wants the worst
            # tick, not the publish_every duty cycle.
            if debug_viz is not None:
                debug_viz.publish(
                    prepared=preflight_result.prepared,
                    grasp_position=grasp_position,
                    grasp_quaternion=grasp_quaternion,
                    controller_target_position=sample.controller_target_position,
                    controller_measured_position=sample.controller_measured_position,
                    bolt_tip_base=frame.pose_world_to_base(
                        fo.BOLT_TIP_POSITION, np.array([1.0, 0.0, 0.0, 0.0])
                    )[0],
                    force=True,
                )
            live_preflight_s.append(
                (node.now_s() if sim else time.perf_counter()) - preflight_started
            )

        command_period = 1.0 / args.rate
        steady_budget = 0.9 * command_period
        steady_preflight_max = max(live_preflight_s[1:])
        print(
            f"live policy preflight: first {live_preflight_s[0]:.3f} s, "
            f"steady max {steady_preflight_max:.3f} s "
            f"(required < {steady_budget:.3f} s"
            + (", debug viz included" if debug_viz is not None else "")
            + ")",
            flush=True,
        )
        if steady_preflight_max >= steady_budget:
            message = (
                f"steady live policy pass took up to {steady_preflight_max:.3f} s; "
                f"must be below {steady_budget:.3f} s for {args.rate:g} Hz physical "
                "execution; reduce --integration-steps or --rate"
            )
            if not sim:
                raise Rejected(message)
            # The simulator shares this CPU (physics, camera rendering, viewer),
            # which the workcell does not. Overruns only stretch the policy
            # period; they are counted as missed deadlines in the report.
            print(f"WARNING (ros-sim): {message}", flush=True)

        if hand_start == "release":
            # C2: training's first observation shows the hand still closing
            # from the release posture into the grasp. Open thumb and index
            # like waypoint1, command the grasp, and start the loop on the
            # first sample after that instead of on a hand that has sat at the
            # grasp for the whole preflight.
            node.wait_for_policy_hand(
                RELEASE_START_PINCH,
                args.hand_timeout,
                args.hand_tolerance,
                lambda: node.publish_hand_target(RELEASE_START_PINCH),
            )
            if fingertip_force_topic:
                # Open, the pads touch nothing: the one moment training's zero holds.
                zero = node.tare_fingertip_force()
                details["fingertip_force_zero_n"] = None if zero is None else zero.tolist()
                if zero is not None:
                    print(
                        f"fingertip force zero (thumb, index): {np.array2string(zero, precision=3)} N",
                        flush=True,
                    )
            if arm_torque_source is not None:
                # At rest and touching nothing, training's reading is zero; what
                # each candidate shows here is its bias (payload/model error). It
                # is logged, not subtracted: training sees no such tare.
                from .observation import FR3_ARM_EFFORT_LIMITS_NM

                rest = node.wait_for_fresh_sample(args.input_timeout).arm_torque_candidates_nm
                details["arm_torque_at_rest_nm"] = {
                    name: value.tolist() for name, value in rest.items()
                }
                limits = np.asarray(FR3_ARM_EFFORT_LIMITS_NM)
                for name, value in rest.items():
                    print(
                        f"{name} torque at rest, normalized"
                        f"{' (policy input)' if name == arm_torque_source else ''}: "
                        + np.array2string(value / limits, precision=3),
                        flush=True,
                    )
            node.publish_hand_target(trained_hand)
            print(
                "hand start: opened to the release posture, grasp commanded; the first "
                "policy sample sees it closing",
                flush=True,
            )
        elif fingertip_force_topic and not sim:
            print(
                "WARNING: --hand-start grasp leaves the RH56 pad forces untared (the "
                "hand never opens before the first sample); their drifting zero goes "
                "straight into the policy",
                flush=True,
            )

        # Re-seed from a fresh measured state so dry preflight cannot affect the
        # first commanded action or the temporal action ensemble.
        sample = node.wait_for_fresh_sample(args.input_timeout)
        grasp_position, grasp_quaternion, _ = node.wait_for_grasp_pose(args.input_timeout)
        q_policy_hand, _ = node.policy_hand_state(sample)
        # Before the first goal the controller still acts about the frozen grip
        # midpoint; from the first goal on it acts about the live one, and the
        # report carries how far the two measurements then stay apart.
        print(
            "live grasp to frozen controller point before the first goal: "
            f"{1.0e3 * float(np.linalg.norm(sample.controller_measured_position - grasp_position)):.1f} mm",
            flush=True,
        )
        # Isaac's reset seeds prev_actions from the live pose (ForgeTg2Env._reset_idx),
        # and the dataset builder takes each sample's previous action from the
        # EMA-filtered stream that starts there, so training's episodes open with
        # it -- native z about 3.3 at the entry grasp, not 0. Zeros (the start
        # before 2026-10-26, matched to an earlier dataset) is kept for comparison.
        prev_action_seed = str(getattr(args, "prev_action_seed", "pose"))
        details["prev_action_seed"] = prev_action_seed
        if prev_action_seed == "pose":
            world_position, world_quaternion = frame.pose_base_to_world(
                grasp_position, grasp_quaternion
            )
            first_previous_action = seed_action_history(
                fo.GraspFrameState(
                    world_position, world_quaternion, np.zeros(3), np.zeros(3), np.zeros((6, 7))
                )
            )
        else:
            first_previous_action = np.zeros(9)
        print(
            f"cycle-1 previous action ({prev_action_seed}): "
            f"{np.array2string(first_previous_action, precision=3)}",
            flush=True,
        )
        session.reset(previous_filtered_native_action=first_previous_action, seed=args.seed)
        coordinator = GripCycleCoordinator(
            rate_hz=args.rate,
            max_cycles=args.cycles,
            reset_position=trained_position,
            reset_quaternion=trained_quaternion,
            reset_hand=trained_hand,
            min_fingertip_separation_m=getattr(args, "min_fingertip_separation", None),
            max_proxy_step_rad=(
                None
                if getattr(args, "max_proxy_step_deg", None) is None
                else math.radians(args.max_proxy_step_deg)
            ),
        )
        print(
            "return gate reference: the trained reset grasp "
            f"{np.array2string(trained_position, precision=4)} (base), hand "
            f"{np.array2string(trained_hand, precision=3)}; the live grasp frame at the "
            f"first sample is {np.array2string(grasp_position, precision=4)}, hand "
            f"{np.array2string(q_policy_hand, precision=3)}"
            + (" (still closing from the release posture)" if hand_start == "release" else ""),
            flush=True,
        )
        # ros-sim's nut sits on ForgeUltra's thread pair and reports its own
        # twist, so the release can be gated on the real turn. If the topic is
        # absent -- an older scene, or the plugin not configured -- say so and
        # fall back to the proxy rather than failing the run.
        thread_pair = None
        if sim:
            thread_pair = ThreadPairClient(node)
            if thread_pair.wait_for_state(args.input_timeout):
                if not thread_pair.reset():
                    print(
                        "warning: /reset_thread did not answer; the nut starts "
                        "wherever the previous run left it",
                        flush=True,
                    )
                print(
                    "thread pair live: nut twist "
                    f"{math.degrees(thread_pair.twist_rad or 0.0):.1f} deg, "
                    f"axial {1.0e3 * (thread_pair.axial_m or 0.0):.2f} mm; "
                    "the release gates on the measured turn",
                    flush=True,
                )
            else:
                thread_pair = None
                print(
                    "warning: no /thread_state in this simulator; falling back to "
                    "the grasp-frame yaw proxy. Regenerate the scene and set "
                    "mujoco_plugins in controllers_sim_policy.yaml.",
                    flush=True,
                )
        if not args.yes:
            input("Start closed-loop physical policy execution? [Enter/Ctrl-C] ")

        period = 1.0 / args.rate
        start = node.now_s()
        next_tick = start
        last_turn_log = -math.inf
        status = "step_budget"
        for steps in range(1, args.max_steps + 1):
            tick_started = node.now_s()
            # A starved simulator can render late; there that only stretches the
            # period, while hardware must stop on stale inputs.
            sample = (
                node.wait_for_fresh_sample(args.input_timeout) if sim else node.sample()
            )
            grasp_position, grasp_quaternion, tool_in_flange = node.grasp_pose_and_tool()
            if steps > 1:
                # A3's measure: the controller now acts about the live grasp frame,
                # so its measured pose and this one should agree up to sampling skew.
                offset = float(
                    np.linalg.norm(sample.controller_measured_position - grasp_position)
                )
                grasp_controlled_offset_m = (
                    offset
                    if grasp_controlled_offset_m is None
                    else max(grasp_controlled_offset_m, offset)
                )
            if sample.controller_position_clipped or sample.controller_orientation_clipped:
                clipped_policy_ticks += 1
            q_hand, dq_hand = node.policy_hand_state(sample)
            q10 = np.concatenate((sample.arm_position, q_hand))
            dq10 = np.concatenate((sample.arm_velocity, dq_hand))
            tip_separation_m = node.last_tip_separation_m
            # Per-tick diagnostics next to the policy's own inputs and outputs,
            # so the next analysis does not have to rebuild them from the
            # joints. The proxy is the previous tick's (this tick's is only
            # known after the coordinator update below).
            extras = {
                "grasp_position": grasp_position,
                "grasp_quaternion_wxyz": grasp_quaternion,
                "fingertip_separation_m": [tip_separation_m],
                "controller_measured_position": sample.controller_measured_position,
                "controller_target_position": sample.controller_target_position,
                "return_error_m_deg_rad": coordinator.return_errors(
                    grasp_position, grasp_quaternion, q_hand
                ),
                "proxy_turn_rad": [coordinator.last_proxy_turn_rad],
            }
            if fingertip_force_topic:
                extras["fingertip_force_n"] = sample.fingertip_force_n
                extras["arm_joint_torque_nm"] = sample.arm_joint_torque_nm
            if sample.arm_torque_candidates_nm is not None:
                for name, value in sample.arm_torque_candidates_nm.items():
                    extras[f"arm_torque_{name}_nm"] = value
                extras["arm_torque_gravity_nm"] = sample.arm_torque_gravity_nm
            phase = coordinator.process_phase()
            progress = min(
                1.0,
                (tick_started - start)
                / runner.config.trajectory_progress_duration_s,
            )
            result = session.step(
                joint_position=q10,
                joint_velocity=dq10,
                rgb=sample.rgb,
                depth=sample.depth,
                depth_units=sample.depth_units,
                trajectory_progress=(
                    progress
                    if runner.config.trajectory_progress_conditioning
                    else None
                ),
                process_phase=(
                    phase if runner.config.cyclic_process_phase_conditioning else None
                ),
                sample_time_s=tick_started,
                fingertip_force_n=sample.fingertip_force_n,
                arm_joint_torque_nm=sample.arm_joint_torque_nm,
                task_signals={
                    "completed_cycles": coordinator.completed_cycles,
                    "watchdog_stop": False,
                    # The turn the release actually gates on. Unmeasurable on
                    # hardware, which has no nut-angle topic; every ros-sim and
                    # hardware recording before this left it NaN, so the return
                    # failures had to be reconstructed offline from the joints.
                    # None rather than a NaN: the recorder stores NaN for an
                    # absent signal but rejects a non-finite one passed in, so
                    # handing it a NaN aborts the run on its first policy step.
                    "threading_turn_progress_rad": (
                        None
                        if thread_pair is None or thread_pair.turn_progress_rad() is None
                        else thread_pair.turn_progress_rad()
                    ),
                },
                extras=extras,
            )
            filtered = result.filtered_native_action.numpy()
            # Steps (0)-(1) of _apply_action here, once per tick; step (2), the clip
            # against the live grasp pose, and the law at that pose, in the controller
            # at 1 kHz with the grasp frame this tick measured.
            goal_position, goal_quaternion = policy_goal_base(frame, filtered)
            node.publish_policy_goal(goal_position, goal_quaternion, tool_in_flange, home_arm)
            node.publish_hand_target(fo.pinch_targets(filtered))
            if debug_viz is not None:
                debug_viz.publish(
                    prepared=result.prepared,
                    grasp_position=grasp_position,
                    grasp_quaternion=grasp_quaternion,
                    controller_target_position=sample.controller_target_position,
                    controller_measured_position=sample.controller_measured_position,
                    bolt_tip_base=frame.pose_world_to_base(
                        fo.BOLT_TIP_POSITION, np.array([1.0, 0.0, 0.0, 0.0])
                    )[0],
                    policy_goal_position=goal_position,
                    step=steps,
                )
            vetoes_before = coordinator.proxy_vetoes
            nut_turn = None if thread_pair is None else thread_pair.turn_progress_rad()
            event, turn_progress = coordinator.update(
                position=grasp_position,
                quaternion=grasp_quaternion,
                hand=q_hand,
                measured_turn_progress=nut_turn,
                measured_total_turn=(
                    None if thread_pair is None else thread_pair.total_turn_rad()
                ),
                fingertip_separation_m=tip_separation_m,
            )
            # Live view of the turn the release gates on (the policy's only
            # nut-rotation input, through its phase), next to the proxy.
            turn_values = turn_monitor_values(
                nut_turn_rad=nut_turn,
                proxy_turn_rad=coordinator.last_proxy_turn_rad,
                gate_turn_rad=turn_progress,
                fingertip_separation_m=tip_separation_m,
                index_rad=q_hand[2],
                phase=phase,
            )
            node.publish_turn(turn_values)
            turn_log_period = getattr(args, "turn_log_period", 1.0)
            if turn_log_period > 0 and tick_started - last_turn_log >= turn_log_period:
                last_turn_log = tick_started
                nut_deg, proxy_deg, _, separation_mm, index_rad, _ = turn_values
                print(
                    f"t={tick_started - start:6.2f}s step {steps:4d} {phase:16s} "
                    f"nut {nut_deg:7.1f} deg  proxy {proxy_deg:7.1f} deg  "
                    f"gate {math.degrees(turn_progress):7.1f}/"
                    f"{math.degrees(coordinator.release_turn_rad):.1f} deg  "
                    f"tips {separation_mm:5.1f} mm  index {index_rad:.2f} rad",
                    flush=True,
                )
            if coordinator.proxy_vetoes != vetoes_before and vetoes_before == 0:
                print(
                    f"proxy at {math.degrees(turn_progress):.1f} deg but the release was "
                    f"refused: {coordinator.last_veto}",
                    flush=True,
                )
            if event is not None:
                source = "nut twist" if coordinator.gated_on_measured_turn else "grasp-turn proxy"
                print(
                    f"policy event: {event}; cycles={coordinator.completed_cycles}, "
                    f"{source}={math.degrees(turn_progress):.1f} deg "
                    f"(proxy={math.degrees(coordinator.last_proxy_turn_rad):.1f} deg)"
                    + (
                        f"; next release at {math.degrees(coordinator.release_turn_rad):.1f} deg"
                        if event == "cycle_completed"
                        else ""
                    ),
                    flush=True,
                )
            if thread_pair is not None and event is not None:
                # Isaac clamps the thread for the whole release/return
                # transition, so the nut cannot unwind while the fingers let go.
                if event == "release_started":
                    thread_pair.hold(True)
                else:
                    thread_pair.hold(False)
                    if event == "cycle_completed":
                        thread_pair.rebase()
            if event == "cycle_completed":
                world_position, world_quaternion = frame.pose_base_to_world(
                    grasp_position, grasp_quaternion
                )
                session.action_filter.reset(
                    seed_action_history(
                        fo.GraspFrameState(
                            world_position,
                            world_quaternion,
                            np.zeros(3),
                            np.zeros(3),
                            np.zeros((6, 7)),
                        )
                    )
                )
            if coordinator.failed:
                status = "return_failed"
                break
            if coordinator.completed_cycles >= args.cycles:
                status = "completed"
                break
            controller_status = node.controller_status()
            if controller_status.get("tracking_fault") == "true":
                raise Rejected(
                    "the Cartesian controller stopped on a tracking fault: "
                    + controller_status.get("last_fault", "unknown")
                )
            if controller_status.get("policy_watchdog_stop") == "true":
                watchdog_stop = True
                status = "controller_watchdog"
                raise RuntimeError("Cartesian controller policy-command watchdog stopped")
            next_tick += period
            if next_tick > node.now_s():
                node.sleep_until(next_tick)
            else:
                missed_deadlines += 1
                next_tick = node.now_s()
    except KeyboardInterrupt:
        status = "interrupted"
        node.abort()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        node.abort()
        raise
    finally:
        # Stop accepting live policy motion before tearing down the ROS graph.
        # The controller ramps its phase clock to zero and then holds. Every
        # step is guarded, and a second Ctrl-C only skips the wait: the
        # recording and the report must survive any of them failing.
        _shutdown_step("abort", node.abort)
        _shutdown_step("abort ramp", lambda: time.sleep(0.6))
        artifact = None
        if collector.sample_count:
            try:
                artifact = collector.close()
            except Exception as exc:  # keep the report even if the data fails
                error = error or f"recording failed: {type(exc).__name__}: {exc}"
        report = _rollout_report(
            status=status,
            error=error,
            steps=steps,
            coordinator=locals().get("coordinator"),
            thread_pair=locals().get("thread_pair"),
            missed_deadlines=missed_deadlines,
            clipped_policy_ticks=clipped_policy_ticks,
            watchdog_stop=watchdog_stop,
            live_preflight_s=live_preflight_s,
            grasp_controlled_offset_m=grasp_controlled_offset_m,
            artifact=artifact,
            loop_period_s=1.0 / args.rate,
            **details,
        )
        (run_dir / "report.json").write_text(json.dumps(report, indent=2))
        _shutdown_step("setup executor", setup_executor.shutdown)
        _shutdown_step("executor", executor.shutdown)
        setup_thread.join(timeout=5.0)
        spin_thread.join(timeout=5.0)
        for item in (node, arm_client, home_client):
            _shutdown_step("destroy node", item.destroy_node)
        _shutdown_step("rclpy", rclpy.shutdown)
    return report


def _shutdown_step(name: str, action) -> None:
    """Run one teardown action, reporting instead of raising (Ctrl-C included)."""

    try:
        action()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"warning: shutdown step {name!r} failed: {type(exc).__name__}: {exc}", flush=True)


def _configured_load(arm_client, timeout_s: float, log) -> dict | None:
    """The load libfranka is compensating for, from FrankaRobotState, or None."""

    try:
        state = arm_client.robot_state_once(timeout_s)
    except TimeoutError as exc:
        log(f"warning: {exc}; the configured load mass goes unchecked", flush=True)
        return None
    load = {
        "load_mass_kg": float(state.inertia_load.inertia.m),
        "load_com_m": [
            float(state.inertia_load.inertia.com.x),
            float(state.inertia_load.inertia.com.y),
            float(state.inertia_load.inertia.com.z),
        ],
        "end_effector_mass_kg": float(state.inertia_ee.inertia.m),
        "total_mass_kg": float(state.inertia_total.inertia.m),
    }
    log(
        f"configured load (Franka Desk): {load['load_mass_kg']:.3f} kg at "
        f"{np.array2string(np.asarray(load['load_com_m']), precision=3)} m, end effector "
        f"{load['end_effector_mass_kg']:.3f} kg, total {load['total_mass_kg']:.3f} kg",
        flush=True,
    )
    return load


def _rollout_report(
    *,
    status,
    error,
    steps,
    coordinator,
    thread_pair=None,
    missed_deadlines,
    clipped_policy_ticks,
    watchdog_stop,
    live_preflight_s,
    grasp_controlled_offset_m,
    artifact,
    loop_period_s,
    **details,
) -> dict:
    return {
        "status": status,
        "error": error,
        "steps": steps,
        "requested_period_s": loop_period_s,
        "completed_cycles": 0 if coordinator is None else coordinator.completed_cycles,
        # What gated the release. "nut_twist" is the simulated thread pair's own
        # angle; "grasp_yaw_proxy" is the inferred one hardware has to use.
        "turn_source": "grasp_yaw_proxy" if thread_pair is None else "nut_twist",
        "final_nut_twist_deg": (
            None
            if thread_pair is None or thread_pair.twist_rad is None
            else math.degrees(thread_pair.twist_rad)
        ),
        "missed_policy_deadlines": missed_deadlines,
        # Ticks at which the controller's per-cycle clip of the goal was binding
        # when it last published its state: the target leading the hand, as in
        # training, rather than a step the runner had to shorten.
        "clipped_policy_ticks": clipped_policy_ticks,
        "watchdog_stop": watchdog_stop,
        "live_preflight_s": live_preflight_s,
        # Largest distance between the controller's measured controlled point and
        # the live grasp frame over the run, after the first goal (A3: should be
        # sampling skew only).
        "grasp_controlled_offset_m": grasp_controlled_offset_m,
        "proxy_vetoes": 0 if coordinator is None else coordinator.proxy_vetoes,
        "last_proxy_veto": None if coordinator is None else coordinator.last_veto,
        "recording": None if artifact is None else str(artifact.data_path),
        # Homing residuals, the trained reset grasp, the configured load, the
        # overlay and hand-start settings (run_hardware_rollout's details).
        **details,
    }


__all__ = [
    "HardwareReadiness",
    "MAX_HOME_PRECOMPENSATION_RAD",
    "RgbdFrameSynchronizer",
    "TrainingFrameAdapter",
    "assess_hardware_readiness",
    "grasp_frame_from_tips",
    "hand_joint_state_to_arrays",
    "grasp_z_transport_at_reset",
    "assert_policy_camera_frames",
    "grasp_in_flange",
    "image_message_to_numpy",
    "physical_hand_state_to_policy",
    "POLICY_TOOL_OFFSET_RPY",
    "POLICY_TOOL_OFFSET_XYZ",
    "TrainingHandKinematics",
    "RELEASE_START_PINCH",
    "next_home_command",
    "trained_reset_grasp",
    "policy_goal_base",
    "policy_tool_offset_config",
    "policy_tool_transform",
    "run_hardware_rollout",
]
