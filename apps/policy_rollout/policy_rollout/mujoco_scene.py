"""Build the ForgeUltra FR3 threading scene in MuJoCo from the replay MJCF.

The workspace's ``fr3_inspirehand_replay.xml`` was authored for the
pre-2026-09-09 Isaac scene and for kinematic nut replay. Three corrections
make it match the scene that produced the student's training data:

- the FR3 base plate sits on the tabletop at world z = 0 (the MJCF keeps it
  0.333 m lower);
- the ``thumb_tip``/``index_tip`` frames that define the grasp frame follow
  the official Inspire URDF (``inspirehand_right.urdf``: ``thumb_tip_fixed``
  and ``index_tip_fixed`` joints), not the MJCF's hand-authored offsets;
- the table is threading_v2's (forgeUltra 35373e0,
  ``relative_pose_table_in_front.yaml``): a legless 0.8 x 1.2 m walnut top
  centred at x = 0.335, not the MJCF's 0.8 x 0.6 m white bench on legs;
- the M24 bolt stands in the printed bolt holder (``Bolt-Holder-M24-v2lprt``)
  the hardware uses, and the tabletop is raised to the holder's foot, so the
  bolt keeps its training pose and the holder fills the 5 cm it used to float;
- the arm is MuJoCo Menagerie's FR3 v2 (inspire_franka_sim's ``fr3v2.xml``, the
  model training moves to): its link inertials, joint ranges, joint friction,
  damping and armature, and its meshes, replace the MJCF's FR3 (v1). The
  kinematics of the two are identical.

Everything else (hand mount at +10 mm with a half-turn, official mimic
coupling, M24 bolt at (0.61, 0, 0.05)) already agrees with the training asset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .forge_osc import BOLT_BASE_POSITION

# thumb_tip_fixed / index_tip_fixed origins from the official URDF.
URDF_THUMB_TIP_OFFSET = (0.002144443, 0.017899759, -0.00745)
URDF_INDEX_TIP_OFFSET = (0.015744429, 0.031656168, -0.00605)

# threading_v2's tabletop half extents (depth along x, width along y, 3 cm
# thick) and its walnut albedo at 3 repeats per metre
# (author_lab_bench_table.py). Training moves the top per reset; its eval and
# this scene keep it at the nominal centre the MJCF already uses.
TRAINING_TABLE_HALF_SIZE = (0.4, 0.6, 0.015)
TRAINING_TABLE_TEXREPEAT = (3.0, 3.0)

# Bolt-Holder-M24-v2lprt.STEP: an 80 x 80 mm foot plate under a 50 mm column,
# 60 mm tall, with a 16.5 mm deep hex pocket (36 mm across flats) the M24 head
# sits in. The mesh is tessellated from the STEP with its origin on the pocket
# floor, i.e. at the bolt's base frame. Its foot is 43.5 mm below that, which
# puts the tabletop at z = 0.0065 under the bolt at z = 0.05 (it was at z = 0
# with the bolt floating on a virtual 5 cm pedestal).
BOLT_HOLDER_MESH = Path("forge_nut_bolt") / "m24" / "m24_bolt_holder.stl"
BOLT_HOLDER_SEAT_HEIGHT = 0.0435
BOLT_HOLDER_RGBA = (0.62, 0.62, 0.64, 1.0)
TRAINING_TABLE_SURFACE_Z = BOLT_BASE_POSITION[2] - BOLT_HOLDER_SEAT_HEIGHT


# Menagerie's FR3 v2 as vendored in inspire_franka_sim, under the fr3_ names.
FR3V2_MJCF = Path(__file__).resolve().parents[3] / "src" / "inspire_franka_sim" / "mjcf" / "fr3v2.xml"
FR3_LINKS = tuple(f"fr3_link{i}" for i in range(8))


def fr3v2_arm():
    """Per-link inertial, joint and geoms of ``fr3v2.xml``, plus its joint default and materials."""

    import xml.etree.ElementTree as ET

    root = ET.parse(FR3V2_MJCF).getroot()
    default = root.find("./default/default[@class='fr3v2']/joint").attrib
    materials = {m.get("name"): m.get("rgba") for m in root.iter("material")}
    meshes = {m.get("name"): m.get("file") for m in root.find("asset").iter("mesh")}
    links = {}
    for body in root.iter("body"):
        name = body.get("name")
        if name not in FR3_LINKS:
            continue
        joint = body.find("joint")
        links[name] = {
            "inertial": body.find("inertial").attrib,
            "joint": None if joint is None else {**default, **joint.attrib},
            "geoms": [g.attrib for g in body.findall("geom")],
        }
    return links, materials, meshes


def use_fr3v2_arm(spec):
    """Give the replay MJCF's FR3 Menagerie's FR3 v2 dynamics and look."""

    import mujoco

    links, materials, meshes = fr3v2_arm()
    mesh_dir = FR3V2_MJCF.parent / "assets"
    for name, rgba in materials.items():
        material = spec.add_material(name=f"fr3v2_{name}")
        material.rgba = [float(v) for v in rgba.split()]
    for name, file in meshes.items():
        spec.add_mesh(name=f"fr3v2_{name}", file=str(mesh_dir / file))
    collision_class = spec.find_default("collision")
    for link, values in links.items():
        body = spec.body(link)
        for geom in list(body.geoms):
            spec.delete(geom)
        inertial = values["inertial"]
        body.explicitinertial = True
        body.mass = float(inertial["mass"])
        body.ipos = [float(v) for v in inertial["pos"].split()]
        body.iquat = [float(v) for v in inertial["quat"].split()]
        body.inertia = [float(v) for v in inertial["diaginertia"].split()]
        for geom in values["geoms"]:
            visual = geom.get("class") == "visual"
            added = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"fr3v2_{geom['mesh']}",
                **({} if visual else {"default": collision_class}),
            )
            if visual:
                added.contype, added.conaffinity, added.group = 0, 0, 2
                added.material = f"fr3v2_{geom['material']}"
            else:
                added.name = geom.get("name", "")
                added.group = 3
            added.mass = 0.0
            added.density = 0.0
        joint = values["joint"]
        if joint is not None:
            spec_joint = spec.joint(joint["name"])
            spec_joint.range = [float(v) for v in joint["range"].split()]
            spec_joint.frictionloss = float(joint["frictionloss"])
            # MjsJoint.damping is (linear, quadratic, cubic) in MuJoCo 3.12.
            spec_joint.damping = [float(joint["damping"]), 0.0, 0.0]
            spec_joint.armature = float(joint["armature"])
    return links


def default_replay_mjcf() -> Path:
    return Path(__file__).resolve().parents[3] / "assets" / "fr3_inspirehand" / "fr3_inspirehand_replay.xml"


def training_scene_spec(
    mjcf: str | Path | None = None,
    *,
    base_plate_z: float = 0.0,
    urdf_tip_frames: bool = True,
    training_table: bool = True,
    bolt_holder: bool = True,
    fr3v2: bool = True,
):
    """Return an ``MjSpec`` of the replay scene with the training corrections."""

    import mujoco

    spec = mujoco.MjSpec.from_file(str(mjcf or default_replay_mjcf()))
    base = spec.body("base")
    base.pos = [float(base.pos[0]), float(base.pos[1]), float(base_plate_z)]
    if urdf_tip_frames:
        spec.body("thumb_tip").pos = list(URDF_THUMB_TIP_OFFSET)
        spec.body("index_tip").pos = list(URDF_INDEX_TIP_OFFSET)
    if training_table:
        top = spec.geom("table_top")
        top.size = list(TRAINING_TABLE_HALF_SIZE)
        top.pos = [0.0, 0.0, -TRAINING_TABLE_HALF_SIZE[2]]
        top.material = "forge_table"
        spec.material("forge_table").texrepeat = list(TRAINING_TABLE_TEXREPEAT)
        for geom in list(spec.body("table").geoms):
            if geom.name.startswith("table_leg"):
                spec.delete(geom)
    if bolt_holder:
        add_bolt_holder(spec)
    if fr3v2:
        use_fr3v2_arm(spec)
    return spec


def add_bolt_holder(spec):
    """Stand ``m24_bolt`` in the hardware's bolt holder and raise the table to it.

    Visual only, like the table and the bolt: nothing in the threading task
    touches it, and its pocket's convex hull would swallow the nut's lowest
    travel. Square to the table, as it is mounted on the bench.
    """

    import mujoco

    spec.add_mesh(name="m24_bolt_holder", file=str(BOLT_HOLDER_MESH))
    bolt = spec.body("m24_bolt")
    holder = spec.worldbody.add_body(name="bolt_holder", pos=list(bolt.pos))
    holder.add_geom(
        name="bolt_holder_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="m24_bolt_holder",
        rgba=list(BOLT_HOLDER_RGBA),
        contype=0,
        conaffinity=0,
    )
    table = spec.body("table")
    table.pos = [float(table.pos[0]), float(table.pos[1]), float(TRAINING_TABLE_SURFACE_Z)]
    return holder


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
    "BOLT_HOLDER_MESH",
    "BOLT_HOLDER_RGBA",
    "BOLT_HOLDER_SEAT_HEIGHT",
    "OPTICAL_TO_MUJOCO",
    "FR3V2_MJCF",
    "TRAINING_TABLE_SURFACE_Z",
    "add_bolt_holder",
    "add_policy_camera",
    "fr3v2_arm",
    "use_fr3v2_arm",
    "link0_world_transform",
    "URDF_INDEX_TIP_OFFSET",
    "URDF_THUMB_TIP_OFFSET",
    "default_replay_mjcf",
    "full_mass_matrix",
    "load_training_scene",
    "training_scene_spec",
]
