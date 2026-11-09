"""The RH56 hand's measured linkage at nominal geometry: register <-> every joint angle.

The driver's own :mod:`inspire_hand_driver.kinematics` maps a register onto its
driven joint linearly and onto the followers with fixed mimic ratios. The real
hand is a four-bar (fingers) or six-bar (thumb bend) linkage per channel, and
policies trained on ForgeUltra's RH56 hand (``hand_asset_profile: rh56``) were
trained against that linkage. This package is the same function, for whoever
has to reproduce it: the policy runner (commands and observations) and the
ros-sim hand (``isaac_hand_relay.py`` and ``IsaacHandPlugin``).

Vendored verbatim from ForgeUltra ``assets/fr3_inspirehand/rh56/deploy``
(6c9b02a plus the joint table export, handed over 2026-11-06), both built from
``tables/rh56_tables.npz`` sha256
``7eec5b32dc8c2dea69fac39a56f26b834a7cad159fc59e9a72cdb89cb74441cc``:

- ``rh56_hand_map.py`` / ``rh56_hand_map.json``: leader angle <-> register;
- ``rh56_joint_table.json``: every joint of each channel per register.

Do not edit those three; re-export them on the training side. Conventions:
register 1000 = open stop, 0 = closed (``open_ratio = register / 1000``);
angles in rad, 0 = open, ForgeUltra joint names (``WORKSPACE_JOINT`` gives
this workspace's).
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
JOINT_TABLE_PATH = HERE / "rh56_joint_table.json"
HAND_MAP_PATH = HERE / "rh56_hand_map.json"
JOINT_TABLE_FORMAT = "rh56_joint_table/v1"
REGISTER_FULL_SCALE = 1000.0

#: Table channel of each leader joint (ForgeUltra names).
CHANNEL_OF_LEADER: Dict[str, str] = {
    "little_joint_0": "little",
    "ring_joint_0": "ring",
    "middle_joint_0": "middle",
    "index_joint_0": "index",
    "thumb_joint_1": "thumb_bend",
    "thumb_joint_0": "thumb_rotation",
}
#: Table channel -> the driver's DOF name (:data:`inspire_hand_driver.kinematics.DOFS`).
DOF_OF_CHANNEL: Dict[str, str] = {
    "little": "pinky",
    "ring": "ring",
    "middle": "middle",
    "index": "index",
    "thumb_bend": "thumb_bend",
    "thumb_rotation": "thumb_rotation",
}
#: ForgeUltra hand joint -> this workspace's (URDF/MJCF) joint name.
WORKSPACE_JOINT: Dict[str, str] = {
    "thumb_joint_0": "thumb_proximal_yaw_joint",
    "thumb_joint_1": "thumb_proximal_pitch_joint",
    "thumb_joint_2": "thumb_intermediate_joint",
    "thumb_joint_3": "thumb_distal_joint",
    "index_joint_0": "index_proximal_joint",
    "index_joint_1": "index_intermediate_joint",
    "middle_joint_0": "middle_proximal_joint",
    "middle_joint_1": "middle_intermediate_joint",
    "ring_joint_0": "ring_proximal_joint",
    "ring_joint_1": "ring_intermediate_joint",
    "little_joint_0": "pinky_proximal_joint",
    "little_joint_1": "pinky_intermediate_joint",
}
#: Training's nominal rod speed (ForgeUltra RH56 linkage drive), mm/s; the
#: register slews at ``1000 * v / stroke_mm`` per second.
NOMINAL_ROD_SPEED_MM_S = 20.0

# ForgeUltra's RH56 linkage drive (rh56/linkage_drive.py, robot_cfg.py), for a
# simulator that has to reproduce the hand's force stop: every joint of a
# channel is a stiff position drive on the linkage path, the rod force is
# estimated from their torques by virtual work, and a closing rod holds once it
# exceeds the stop.

#: Drive gains (N m/rad, N m s/rad) per table DOF type, robot_cfg.GAINS; effort 50 N m.
DRIVE_GAINS: Dict[str, Tuple[float, float]] = {
    "finger": (40.0, 0.4),
    "thumb_bend": (60.0, 0.6),
    "thumb_yaw": (60.0, 0.6),
}
DRIVE_EFFORT_LIMIT_NM = 50.0
#: The threading recipe's stop (``rh56_force_stop_range_n`` 42.5..57.5 N, drawn
#: per episode): its middle. N of rod force.
NOMINAL_FORCE_STOP_N = 50.0
#: A stalled rod moves again only below this fraction of the stop.
FORCE_RELEASE_FRAC = 0.9
#: First-order low pass on the rod force estimate: alpha 0.2 per training
#: physics step (1/120 s), i.e. a 37 ms time constant.
FORCE_FILTER_ALPHA = 0.2
FORCE_FILTER_DT_S = 1.0 / 120.0


def force_filter_alpha(dt_s: float) -> float:
    """The low pass's per-step alpha at step ``dt_s`` with training's time constant."""

    return 1.0 - (1.0 - FORCE_FILTER_ALPHA) ** (max(dt_s, 0.0) / FORCE_FILTER_DT_S)


def load_hand_map():
    """The vendored ``RH56HandMap`` (leader angle <-> register)."""

    from .rh56_hand_map import RH56HandMap

    return RH56HandMap(str(HAND_MAP_PATH))


class JointTable:
    """``rh56_joint_table.json``: every joint of each channel, piecewise linear in the register."""

    def __init__(self, path=JOINT_TABLE_PATH):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != JOINT_TABLE_FORMAT:
            raise ValueError(f"{path}: expected format {JOINT_TABLE_FORMAT!r}, got {data.get('format')!r}")
        self.tables_sha256 = data.get("tables_sha256")
        #: channel -> (leader joint, ascending registers, {joint: angles at those registers})
        self.channels: Dict[str, Tuple[str, np.ndarray, Dict[str, np.ndarray]]] = {}
        #: channel -> rod stroke, mm
        self.stroke_mm: Dict[str, float] = {}
        #: channel -> table DOF type (``finger``, ``thumb_bend``, ``thumb_yaw``)
        self.dof_type: Dict[str, str] = {}
        #: channel -> {joint: d angle / d register} at the ascending registers
        self._slope: Dict[str, Dict[str, np.ndarray]] = {}
        for name, ch in data["channels"].items():
            # The arrays run register 1000 -> 0; np.interp needs ascending samples.
            register = np.asarray(ch["register"], dtype=float)[::-1]
            if np.any(np.diff(register) <= 0.0):
                raise ValueError(f"{path}: channel {name!r} register is not strictly monotone")
            joints = {j: np.asarray(q, dtype=float)[::-1] for j, q in ch["joint_rad"].items()}
            if np.any(np.diff(joints[ch["leader_joint"]]) >= 0.0):
                raise ValueError(f"{path}: channel {name!r} leader does not close as the register falls")
            self.channels[name] = (ch["leader_joint"], register, joints)
            self.stroke_mm[name] = float(ch["stroke_mm"])
            self.dof_type[name] = str(ch["table_dof"])
            # Piecewise-linear table: the slope of each segment, looked up by segment.
            self._slope[name] = {j: np.diff(q) / np.diff(register) for j, q in joints.items()}

    def leader_range(self, channel: str) -> Tuple[float, float]:
        """``(open, closed)`` leader angle of ``channel``: its full stroke."""

        leader, _, joints = self.channels[channel]
        q = joints[leader]
        return float(q.min()), float(q.max())

    def joint_range(self, joint: str) -> Tuple[float, float]:
        """The stroke-wide range of any hand joint (ForgeUltra name)."""

        for _, _, joints in self.channels.values():
            if joint in joints:
                return float(joints[joint].min()), float(joints[joint].max())
        raise KeyError(joint)

    def register_of_leader(self, channel: str, rad: float) -> float:
        """The register (0..1000) at which ``channel``'s leader sits at ``rad``, clamped to the stroke."""

        leader, register, joints = self.channels[channel]
        # The leader angle falls as the register rises; interp needs it ascending.
        return float(np.interp(rad, joints[leader][::-1], register[::-1]))

    def joints_at(self, channel: str, register: float) -> Dict[str, float]:
        """``{joint: rad}`` for every joint of ``channel`` at ``register`` (clamped to 0..1000)."""

        _, reg, joints = self.channels[channel]
        r = float(np.clip(register, 0.0, REGISTER_FULL_SCALE))
        return {j: float(np.interp(r, reg, q)) for j, q in joints.items()}

    def dq_dL(self, channel: str, register: float) -> Dict[str, float]:
        """``{joint: d angle / d rod length}`` (rad/mm, closing positive) at ``register``.

        The register is linear in rod length and falls as the rod extends to
        close: ``dregister/dL = -1000 / stroke_mm``.
        """

        _, reg, _ = self.channels[channel]
        k = int(np.clip(np.searchsorted(reg, register, side="right") - 1, 0, len(reg) - 2))
        scale = -REGISTER_FULL_SCALE / self.stroke_mm[channel]
        return {j: float(s[k] * scale) for j, s in self._slope[channel].items()}

    def follower_curves(self, channel: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Per follower of ``channel``: (leader angles ascending, follower angles) along the stroke."""

        leader, _, joints = self.channels[channel]
        order = np.argsort(joints[leader])
        return {
            j: (joints[leader][order], q[order]) for j, q in joints.items() if j != leader
        }


@lru_cache(maxsize=1)
def load_joint_table() -> JointTable:
    """The vendored joint table, checked to share the hand map's linkage tables."""

    table = JointTable()
    map_meta = json.loads(HAND_MAP_PATH.read_text(encoding="utf-8"))
    if table.tables_sha256 != map_meta.get("tables_sha256"):
        raise ValueError(
            "rh56_joint_table.json and rh56_hand_map.json were exported from different "
            "linkage tables; re-export both"
        )
    return table


__all__ = [
    "CHANNEL_OF_LEADER",
    "DOF_OF_CHANNEL",
    "DRIVE_EFFORT_LIMIT_NM",
    "DRIVE_GAINS",
    "FORCE_RELEASE_FRAC",
    "JointTable",
    "NOMINAL_FORCE_STOP_N",
    "NOMINAL_ROD_SPEED_MM_S",
    "force_filter_alpha",
    "REGISTER_FULL_SCALE",
    "WORKSPACE_JOINT",
    "load_hand_map",
    "load_joint_table",
]
