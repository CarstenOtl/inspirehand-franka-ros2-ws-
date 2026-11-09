"""The RH56 linkage-hand contract of students trained with ``hand_asset_profile: rh56``.

Source of truth: ForgeUltra 6c9b02a,

- ``assets/fr3_inspirehand/rh56/env_hand.py`` ``RegisterFingerActions``
- ``forge_ultra/tasks/mdp/robot_control.py`` ``generate_ctrl_signals`` (``rh56_register_finger_actions``)
- ``assets/fr3_inspirehand/rh56/deploy/README.md`` (the angle <-> register map)

On the mimic hand a pinch action is an angle about the grasp posture and the
driver turns angles into open ratios linearly. On the RH56 hand the action is a
register about the grasp register, ``reg = grasp - a * half_span`` clamped to the
stroke, and the leader angle a register reaches follows the measured four-bar /
six-bar linkage. The policy still observes leader angles (``hand_observation:
joint``), so the readout goes register -> angle through the same map.

The map and the follower table live in ``inspire_hand_driver.rh56_linkage``,
shared with ros-sim's RH56 hand, which has to reproduce the same linkage.

Driver side: the RH56 register is ``1000 * open_ratio``. The driver takes open
ratios on its command topic and reports the actual registers as open ratios on
``~/state`` (channels ``1``..``6``); that, not its linear ``~/joint_states``, is
what the runner reads. The driver's thumb-yaw overlay has to be off: it would
move the thumb register off the one the policy chose.
"""

from __future__ import annotations

import numpy as np

from inspire_hand_driver.rh56_linkage import (
    CHANNEL_OF_LEADER,
    REGISTER_FULL_SCALE,
    JointTable,
    load_hand_map,
    load_joint_table,
)

from . import forge_osc as fo

#: Policy leader joint -> this workspace's RH56 driver joint.
DRIVER_JOINT_OF_LEADER = {
    "thumb_joint_0": "thumb_proximal_yaw_joint",
    "thumb_joint_1": "thumb_proximal_pitch_joint",
    "index_joint_0": "index_proximal_joint",
    "middle_joint_0": "middle_proximal_joint",
    "ring_joint_0": "ring_proximal_joint",
    "little_joint_0": "pinky_proximal_joint",
}
SUPPORT_LEADERS = ("little_joint_0", "ring_joint_0", "middle_joint_0")
#: ``rh56_inactive_finger_register``: the support fingers tuck short of the
#: palm (the linkage's closed stop puts the fingertips into it).
INACTIVE_FINGER_REGISTER = 100.0


class LinearHandMap:
    """The old hand model's register <-> leader angle: the driver's linear map.

    Same interface as ``RH56HandMap`` (``rad_from_register``,
    ``register_from_rad``, ``open_ratio_from_rad``, keyed by leader joint), so
    the runner can read and command this workcell's hand through it instead of
    the linkage table. On 2026-11-06 the user found this map plus the mimic
    followers (``expand_followers_mimic``) closer to the real hand than
    ForgeUltra's linkage table (debug/rh56_thumb_model_compare.png).

    Its strokes are shorter than the linkage's (fully closed: thumb rotation
    1.246 vs 1.308 rad, thumb bend 0.48 vs 0.716, index 1.333 vs 1.47); a
    student angle past them is sent as register 0 and read back at the stroke.
    """

    followers = "mimic"

    @staticmethod
    def _dof(key: str) -> int:
        from inspire_hand_driver import kinematics as kin

        return kin.dof_index(DRIVER_JOINT_OF_LEADER[key])

    def rad_from_register(self, key: str, register):
        from inspire_hand_driver import kinematics as kin

        ratio = float(np.clip(register, 0.0, REGISTER_FULL_SCALE)) / REGISTER_FULL_SCALE
        return kin.open_ratio_to_rad(self._dof(key), ratio)

    def open_ratio_from_rad(self, key: str, rad):
        from inspire_hand_driver import kinematics as kin

        return kin.rad_to_open_ratio(self._dof(key), float(rad))

    def register_from_rad(self, key: str, rad):
        return REGISTER_FULL_SCALE * self.open_ratio_from_rad(key, rad)


def expand_followers_mimic(leaders: dict) -> dict:
    """Every hand joint at these leader angles under the old hand model's mimic couplings.

    Support fingers without a leader sit where the linear map puts register
    ``INACTIVE_FINGER_REGISTER``, as the runner commands them.
    """

    hand_map = LinearHandMap()
    targets = {
        name: float(hand_map.rad_from_register(name, INACTIVE_FINGER_REGISTER))
        for name in SUPPORT_LEADERS
    }
    targets.update({name: float(value) for name, value in leaders.items()})
    return fo.expand_hand_mimic(targets)


RH56_HAND_MODELS = ("linkage", "linear")


class RegisterPinchActions:
    """Pinch actions -> registers: ``RegisterFingerActions`` on the nominal table.

    ``a`` in [-1, 1] per pinch leader, positive = closing (register falls
    toward 0). The grasp register and the stroke come from the grasp posture and
    the pinch ranges through the nominal map, so the action -> register map is
    the one training used in every environment.
    """

    def __init__(self, hand_map, profile: fo.TaskProfile):
        if profile.hand != "rh56":
            raise ValueError(f"register pinch actions need the rh56 hand, not {profile.hand!r}")
        self.hand_map = hand_map
        grasp, reg_min, reg_max = [], [], []
        for name, (q_lo, q_hi) in zip(fo.PINCH_JOINTS, profile.pinch_ranges):
            grasp.append(float(hand_map.register_from_rad(name, profile.grasp_posture[name])))
            reg_min.append(float(hand_map.register_from_rad(name, q_hi)))  # closed end = low register
            reg_max.append(float(hand_map.register_from_rad(name, q_lo)))
        self.grasp = np.array(grasp)
        self.reg_min = np.array(reg_min)
        self.reg_max = np.array(reg_max)
        self.half_span = 0.5 * (self.reg_max - self.reg_min)

    def registers(self, filtered_action) -> np.ndarray:
        """The three commanded registers (thumb rotation, thumb bend, index), 0..1000."""

        a = np.asarray(filtered_action, dtype=float)[-3:]
        return np.clip(self.grasp - a * self.half_span, self.reg_min, self.reg_max)

    def pinch_targets(self, filtered_action) -> np.ndarray:
        """The leader angles those registers reach at nominal geometry (``leader_of_register``)."""

        registers = self.registers(filtered_action)
        return np.array(
            [
                float(self.hand_map.rad_from_register(name, reg))
                for name, reg in zip(fo.PINCH_JOINTS, registers)
            ]
        )


def command_open_ratios(hand_map, pinch, hand_joints) -> list[float]:
    """Driver open ratios, in ``hand_joints`` order, for the pinch leader angles.

    The support fingers sit at ``INACTIVE_FINGER_REGISTER``. Angles go through
    the map, so the driver writes exactly the register training commanded.
    """

    pinch = np.asarray(pinch, dtype=float).reshape(-1)
    if pinch.shape != (3,) or not np.isfinite(pinch).all():
        raise ValueError("pinch must contain three finite leader angles")
    ratios = {
        DRIVER_JOINT_OF_LEADER[name]: INACTIVE_FINGER_REGISTER / REGISTER_FULL_SCALE
        for name in SUPPORT_LEADERS
    }
    for name, rad in zip(fo.PINCH_JOINTS, pinch):
        ratios[DRIVER_JOINT_OF_LEADER[name]] = float(hand_map.open_ratio_from_rad(name, rad))
    return [ratios[joint] for joint in hand_joints]


def driver_radians_for_leaders(hand_map, pinch, hand_joints) -> np.ndarray:
    """The driver-convention radians that command the pinch through the linear map.

    For callers that only take radians (``CoordinatedReplayClient.command_hand``
    and ``wait_for_hand``): the driver's linear ``rad_to_open_ratio`` of these
    is the map's open ratio of the leader angle.
    """

    from inspire_hand_driver import kinematics as kin

    ratios = command_open_ratios(hand_map, pinch, hand_joints)
    return np.array(
        [kin.open_ratio_to_rad(kin.dof_index(joint), ratio) for joint, ratio in zip(hand_joints, ratios)]
    )


def register_state_to_open_ratios(message, hand_joints) -> np.ndarray:
    """The driver's ``~/state`` (open ratio per channel ``1``..``6``) in ``hand_joints`` order."""

    from inspire_hand_driver import kinematics as kin

    ratios = dict(zip(message.name, message.position))
    joint_of_channel = {dof.channel: dof.joint for dof in kin.DOFS}
    by_joint = {joint_of_channel[c]: float(r) for c, r in ratios.items() if c in joint_of_channel}
    if any(joint not in by_joint for joint in hand_joints):
        raise RuntimeError("hand register state does not contain all six RH56 channels")
    return np.array([by_joint[joint] for joint in hand_joints], dtype=float)


def leaders_from_open_ratios(hand_map, open_ratios, hand_joints, policy_hand_joints) -> np.ndarray:
    """The policy's leader angles from the actual registers, through the linkage map."""

    columns = {name: index for index, name in enumerate(hand_joints)}
    leader_of_driver = {joint: leader for leader, joint in DRIVER_JOINT_OF_LEADER.items()}
    return np.array(
        [
            float(
                hand_map.rad_from_register(
                    leader_of_driver[joint], REGISTER_FULL_SCALE * float(open_ratios[columns[joint]])
                )
            )
            for joint in policy_hand_joints
        ]
    )


def expand_followers(leaders: dict, joint_table: JointTable | None = None) -> dict:
    """Every hand joint's angle at these leader angles, along the linkage.

    The grasp frame is built from the training asset's fingertips, which on the
    RH56 hand sit where the linkage puts the followers (``expand_followers`` of
    ``RH56EnvHand`` at nominal geometry), not where the mimic multipliers do:
    at a fully bent thumb the mimic puts thumb_joint_3 at 0.61 rad, the linkage
    at 0.81. Leaders are clamped to their stroke; a finger without a leader in
    ``leaders`` sits at ``INACTIVE_FINGER_REGISTER``, as the support fingers do.
    """

    table = joint_table or load_joint_table()
    out = {}
    for leader, channel in CHANNEL_OF_LEADER.items():
        if leader in leaders:
            register = table.register_of_leader(channel, float(leaders[leader]))
        else:
            register = INACTIVE_FINGER_REGISTER
        out.update(table.joints_at(channel, register))
    return out


__all__ = [
    "DRIVER_JOINT_OF_LEADER",
    "LinearHandMap",
    "RH56_HAND_MODELS",
    "expand_followers_mimic",
    "INACTIVE_FINGER_REGISTER",
    "JointTable",
    "RegisterPinchActions",
    "command_open_ratios",
    "driver_radians_for_leaders",
    "expand_followers",
    "leaders_from_open_ratios",
    "load_hand_map",
    "load_joint_table",
    "register_state_to_open_ratios",
]
