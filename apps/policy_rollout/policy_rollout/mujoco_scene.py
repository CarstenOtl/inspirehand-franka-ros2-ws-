"""Build the ForgeUltra FR3 threading scene in MuJoCo from the replay MJCF.

The workspace's ``fr3_inspirehand_replay.xml`` was authored for the
pre-2026-09-09 Isaac scene and for kinematic nut replay. Two corrections make
it match the scene that produced the student's training data:

- the FR3 base plate sits on the tabletop at world z = 0 (the MJCF keeps it
  0.333 m lower);
- the ``thumb_tip``/``index_tip`` frames that define the grasp frame follow
  the official Inspire URDF (``inspirehand_right.urdf``: ``thumb_tip_fixed``
  and ``index_tip_fixed`` joints), not the MJCF's hand-authored offsets.

Everything else (hand mount at +10 mm with a half-turn, official mimic
coupling, M24 bolt at (0.61, 0, 0.05)) already agrees with the training asset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# thumb_tip_fixed / index_tip_fixed origins from the official URDF.
URDF_THUMB_TIP_OFFSET = (0.002144443, 0.017899759, -0.00745)
URDF_INDEX_TIP_OFFSET = (0.015744429, 0.031656168, -0.00605)


def default_replay_mjcf() -> Path:
    return Path(__file__).resolve().parents[3] / "assets" / "fr3_inspirehand" / "fr3_inspirehand_replay.xml"


def training_scene_spec(
    mjcf: str | Path | None = None,
    *,
    base_plate_z: float = 0.0,
    urdf_tip_frames: bool = True,
):
    """Return an ``MjSpec`` of the replay scene with the training corrections."""

    import mujoco

    spec = mujoco.MjSpec.from_file(str(mjcf or default_replay_mjcf()))
    base = spec.body("base")
    base.pos = [float(base.pos[0]), float(base.pos[1]), float(base_plate_z)]
    if urdf_tip_frames:
        spec.body("thumb_tip").pos = list(URDF_THUMB_TIP_OFFSET)
        spec.body("index_tip").pos = list(URDF_INDEX_TIP_OFFSET)
    return spec


def load_training_scene(mjcf: str | Path | None = None, **kwargs):
    """Compile the corrected scene and return ``(model, data)``."""

    import mujoco

    model = training_scene_spec(mjcf, **kwargs).compile()
    return model, mujoco.MjData(model)


# MuJoCo cameras look along -z with y up; a ROS optical frame looks along +z
# with y down. They differ by a half turn about x.
OPTICAL_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])


def link0_world_transform(base_plate_z: float = 0.0) -> np.ndarray:
    """fr3_link0 in the training world: base at (1.2, 0, z) yawed by pi."""

    t = np.eye(4)
    t[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    t[:3, 3] = [1.2, 0.0, base_plate_z]
    return t


def add_policy_camera(spec, profile, *, name="policy_calibrated", base_plate_z: float = 0.0):
    """Add the calibrated D415 colour camera (fovy model) to a scene spec."""

    import math

    import mujoco

    from .forge_osc import matrix_from_quat

    pose = profile.training_world_pose
    if pose.parent_frame_id != "fr3_link0":
        raise ValueError("camera profile pose must be expressed in fr3_link0")
    t_link0_optical = np.eye(4)
    t_link0_optical[:3, :3] = matrix_from_quat(np.asarray(pose.rotation_wxyz))
    t_link0_optical[:3, 3] = pose.translation_m
    t_world_optical = link0_world_transform(base_plate_z) @ t_link0_optical
    t_world_cam = t_world_optical.copy()
    t_world_cam[:3, :3] = t_world_optical[:3, :3] @ OPTICAL_TO_MUJOCO
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, t_world_cam[:3, :3].flatten())
    k = profile.source_intrinsics
    cam = spec.worldbody.add_camera()
    cam.name = name
    cam.pos = t_world_cam[:3, 3]
    cam.quat = quat
    cam.fovy = math.degrees(2.0 * math.atan(k.height / (2.0 * k.camera_matrix[4])))
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, k.width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, k.height)
    return t_world_optical


def full_mass_matrix(model, data) -> np.ndarray:
    import mujoco

    full = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, full)
    except TypeError:  # pragma: no cover - MuJoCo < 3.3 signature
        mujoco.mj_fullM(model, full, data.qM)
    return full


__all__ = [
    "OPTICAL_TO_MUJOCO",
    "add_policy_camera",
    "link0_world_transform",
    "URDF_INDEX_TIP_OFFSET",
    "URDF_THUMB_TIP_OFFSET",
    "default_replay_mjcf",
    "full_mass_matrix",
    "load_training_scene",
    "training_scene_spec",
]
