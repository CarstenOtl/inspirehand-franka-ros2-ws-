"""RH56 register <-> leader joint angle map, shared by the forgeUltra sim and the hardware driver.

Standalone: numpy + json only, no forgeUltra import, so the deployment side can copy this file and
``rh56_hand_map.json`` verbatim.

The map replaces the linear ``open_ratio = clamp(1 - (rad - lower) / (upper - lower), 0, 1)`` for
policies trained on the RH56 linkage hand (task ``...-FR3v2-RH56-v0``, ``hand_asset_profile: rh56``).
It is the measured linkage at nominal geometry: the angle a policy commands or observes is the real
leader joint angle, and the sim drives its registers through exactly this function.

Conventions:
  * register 1000 = open stop, 0 = closed stop (the RH56 position register; open_ratio = register / 1000);
  * angle = leader joint angle in rad as in the sim (official fr3_inspirehand URDF joint, 0 = open);
  * one channel per driver DOF: little, ring, middle, index (leader ``<finger>_joint_0``),
    thumb_bend (``thumb_joint_1``), thumb_rotation (``thumb_joint_0``);
  * angles outside the stroke clamp to the stop registers, registers outside 0..1000 clamp too.

Policies trained on the mimic hand (official / calibrated profiles) keep the linear map.

    hand_map = RH56HandMap()
    register = hand_map.register_from_rad("index_joint_0", 0.60)   # command
    rad = hand_map.rad_from_register("index", 557.0)               # reading
"""
from __future__ import annotations

import json
import os

import numpy as np

FORMAT = "rh56_hand_map/v1"
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rh56_hand_map.json")


class RH56HandMap:
    def __init__(self, path: str = DEFAULT_PATH):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("format") != FORMAT:
            raise ValueError(f"{path}: expected format {FORMAT!r}, got {data.get('format')!r}")
        self.meta = {k: v for k, v in data.items() if k != "channels"}
        self._by_rad: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # rad ascending -> register
        self._by_reg: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # register ascending -> rad
        self.leader_joint: dict[str, str] = {}
        for name, ch in data["channels"].items():
            rad = np.asarray(ch["leader_rad"], dtype=np.float64)
            reg = np.asarray(ch["register"], dtype=np.float64)
            if rad.shape != reg.shape or rad.ndim != 1 or rad.size < 2:
                raise ValueError(f"{path}: channel {name!r} needs two equal 1-D arrays")
            o = np.argsort(rad)
            if np.any(np.diff(rad[o]) <= 0) or np.any(np.diff(reg[o]) >= 0):
                raise ValueError(f"{path}: channel {name!r} is not strictly monotone")
            self._by_rad[name] = (rad[o], reg[o])
            o = np.argsort(reg)
            self._by_reg[name] = (reg[o], rad[o])
            self.leader_joint[name] = ch["leader_joint"]
        self.channel_of_joint = {j: name for name, j in self.leader_joint.items()}

    @property
    def channels(self) -> list[str]:
        return list(self.leader_joint)

    def _channel(self, key: str) -> str:
        if key in self._by_rad:
            return key
        if key in self.channel_of_joint:
            return self.channel_of_joint[key]
        raise KeyError(f"unknown RH56 channel or leader joint {key!r}; have {self.channels}")

    def register_from_rad(self, key: str, rad):
        """Register (0..1000, float) that puts the leader of ``key`` (channel or joint name) at ``rad``."""
        x, y = self._by_rad[self._channel(key)]
        return np.interp(rad, x, y)

    def rad_from_register(self, key: str, register):
        """Leader angle [rad] of ``key`` (channel or joint name) at ``register`` (0..1000)."""
        x, y = self._by_reg[self._channel(key)]
        return np.interp(np.clip(register, 0.0, 1000.0), x, y)

    def open_ratio_from_rad(self, key: str, rad):
        """Drop-in for the linear ``open_ratio``: register / 1000."""
        return self.register_from_rad(key, rad) / 1000.0

    def rad_from_open_ratio(self, key: str, open_ratio):
        return self.rad_from_register(key, np.asarray(open_ratio, dtype=np.float64) * 1000.0)

    def stroke(self, key: str) -> tuple[float, float]:
        """(open, closed) leader angle of ``key``."""
        x, _ = self._by_rad[self._channel(key)]
        return float(x[0]), float(x[-1])
