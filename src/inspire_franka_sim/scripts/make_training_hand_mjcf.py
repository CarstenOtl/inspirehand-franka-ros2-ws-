#!/usr/bin/env python3
"""Vendor ForgeUltra's training hand geometry as a ros-sim-ready MJCF.

    python3 scripts/make_training_hand_mjcf.py

Writes ``mjcf/inspire_hand_right_training.xml`` and the meshes it needs into
``mjcf/assets_hand/training_right/``. Both are committed, so this only has to be
re-run when ``assets/fr3_inspirehand/`` is re-vendored.

Why a second hand model
-----------------------
The workspace has two independent descriptions of the same Inspire RH56.
``inspire_hand_right.xml`` comes from dex-urdf, which derives from Inspire's
published STEP files. The student policy was distilled on the other one: the
official Tiangong 2.0 Pro URDF, vendored here as
``assets/fr3_inspirehand/fr3_inspirehand_replay.xml``.

They are the same hand dimensionally -- thumb base to thumb pad agrees to
0.7 mm -- but they put the thumb's follower joints' zero at different
flexions, so dex-urdf carries the thumb pad about 15 degrees further from the
index. At the posture where the training hand pinches to 12 mm, the dex-urdf
hand's pads are 55 mm apart, which is why a policy that grips in training only
hovers in ros-sim. Teacher-policy rollouts on the bench say the training
geometry is the closer match to the physical hand, so this model is the one to
simulate against. ``apps/policy_rollout/utils/compare_hand_models.py`` draws
the two side by side.

What is taken from training, and what is not
--------------------------------------------
Taken: every link origin, mass and inertia, the mesh set, the joint axes and
ranges, and the coupling ratios (1.1169 on the fingers; the thumb's 1.1425 then
0.7508 *chained* off the intermediate, where dex-urdf drives both followers
straight off the pitch).

Not taken: the joint dynamics and torque limits. Training declares damping 1,
frictionloss 0.5 and a 50 N m effort limit, which suit PhysX's implicit drives.
ros-sim drives the hand through ``config/pids.yaml`` (p 20, d 0.4, output
clamped to 1 N m), so this model keeps the same dynamics and 1 N m actuator
range ``make_hand_mjcf.py`` uses for the dex-urdf hand. Swapping geometry and
drive tuning at once would leave nothing to attribute a change to.

Also not taken: the tip body offsets. The replay MJCF's own ``thumb_tip`` and
``index_tip`` are hand-authored and differ from the official URDF's, which is
why ``policy_rollout.mujoco_scene`` overrides them. The official values are
used here so the simulated tips mean the same points the policy's grasp frame
is built from.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
SOURCE_MJCF = REPO / "assets" / "fr3_inspirehand" / "fr3_inspirehand_replay.xml"
SOURCE_MESHES = REPO / "assets" / "fr3_inspirehand" / "hand"
MESH_SUBDIR = Path("assets_hand") / "training_right"
OUTPUT = PACKAGE / "mjcf" / "inspire_hand_right_training.xml"

#: Training name -> the name the rest of the workspace already speaks. The
#: controllers, config/pids.yaml, inspire_hand_sim_bridge.py and
#: inspire_hand_driver.kinematics all address these joints, so the geometry
#: changes underneath and nothing else has to.
BODIES = {
    "palm": "hand_base_link",
    "thumb_link_0": "thumb_proximal_base",
    "thumb_link_1": "thumb_proximal",
    "thumb_link_2": "thumb_intermediate",
    "thumb_link_3": "thumb_distal",
    "index_link_0": "index_proximal",
    "index_link_1": "index_intermediate",
    "middle_link_0": "middle_proximal",
    "middle_link_1": "middle_intermediate",
    "ring_link_0": "ring_proximal",
    "ring_link_1": "ring_intermediate",
    "little_link_0": "pinky_proximal",
    "little_link_1": "pinky_intermediate",
}
JOINTS = {
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

#: The six driven joints, in the register order the driver uses. Must match
#: make_hand_mjcf.py's DRIVEN and inspire_hand_sim_bridge.py's DRIVEN_JOINTS.
DRIVEN = (
    "pinky_proximal_joint",
    "ring_proximal_joint",
    "middle_proximal_joint",
    "index_proximal_joint",
    "thumb_proximal_pitch_joint",
    "thumb_proximal_yaw_joint",
)

#: ``thumb_tip_fixed`` / ``index_tip_fixed`` and the three support tips, from
#: the official URDF (``inspirehand_right.urdf``), parented as upstream has
#: them. Same constants as policy_rollout.mujoco_scene.
TIPS = {
    "thumb_distal": ("thumb_tip", (0.002144443, 0.017899759, -0.00745)),
    "index_intermediate": ("index_tip", (0.015744429, 0.031656168, -0.00605)),
    "middle_intermediate": ("middle_tip", (0.015745034, 0.031655867, -0.00605)),
    "ring_intermediate": ("ring_tip", (0.015743672, 0.031656544, -0.00605)),
    "pinky_intermediate": ("pinky_tip", (0.015739593, 0.031658573, -0.00505)),
}

# Kept identical to make_hand_mjcf.py so the two hand models differ only in
# geometry; see the module docstring.
JOINT_DYNAMICS = dict(damping=0.1, armature=0.002, frictionloss=0.005)
ACTUATOR_TORQUE = 1.0
CONTACT_FRICTION = (0.75, 0.005, 0.0001)
SHELL_RGBA = (0.15, 0.15, 0.17, 1.0)


def build(source: Path):
    import mujoco

    src = mujoco.MjSpec.from_file(str(source))
    spec = mujoco.MjSpec()
    spec.modelname = "inspire_hand_right_training"
    spec.compiler.degree = False
    spec.meshdir = f"{MESH_SUBDIR.parent}/"
    spec.compiler.meshdir = spec.meshdir

    mount = spec.worldbody.add_body(name="hand_mount")
    mount.add_frame().attach_body(src.body("palm"))

    # The palm carries the flange transform in the source scene. Strip it: the
    # on-flange wrapper supplies it, exactly as it does for the dex-urdf hand.
    palm = spec.body("palm")
    palm.pos = [0.0, 0.0, 0.0]
    palm.quat = [1.0, 0.0, 0.0, 0.0]

    for body in spec.bodies:
        if body.name not in BODIES:
            continue
        visual, collision = body.geoms[0], body.geoms[1]
        visual.group, visual.contype, visual.conaffinity = 1, 0, 0
        visual.density, visual.rgba = 0.0, list(SHELL_RGBA)
        collision.group, collision.contype, collision.conaffinity = 3, 1, 1
        collision.friction = list(CONTACT_FRICTION)
        collision.rgba = list(SHELL_RGBA)
        for joint in body.joints:
            joint.damping = [JOINT_DYNAMICS["damping"], 0.0, 0.0]
            joint.frictionloss = JOINT_DYNAMICS["frictionloss"]
            joint.armature = JOINT_DYNAMICS["armature"]
            joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
            joint.actfrcrange = [-ACTUATOR_TORQUE, ACTUATOR_TORQUE]

    for parent, (tip, offset) in TIPS.items():
        source_name = next(k for k, v in BODIES.items() if v == parent)
        existing = [b for b in spec.bodies if b.name == tip]
        if existing:
            existing[0].pos = list(offset)
        else:
            spec.body(source_name).add_body(name=tip, pos=list(offset))

    # Rename last, so the lookups above can use the source names.
    for body in spec.bodies:
        if body.name in BODIES:
            body.name = BODIES[body.name]
    for joint in spec.joints:
        if joint.name in JOINTS:
            joint.name = JOINTS[joint.name]
    for equality in spec.equalities:
        equality.name1 = JOINTS.get(equality.name1, equality.name1)
        equality.name2 = JOINTS.get(equality.name2, equality.name2)
        equality.name = f"{equality.name1}_follows_{equality.name2}"
    for exclude in spec.excludes:
        exclude.bodyname1 = BODIES.get(exclude.bodyname1, exclude.bodyname1)
        exclude.bodyname2 = BODIES.get(exclude.bodyname2, exclude.bodyname2)
        exclude.name = f"{exclude.bodyname1}_{exclude.bodyname2}"
    # The source is a whole workcell; attaching its palm brings the scene's
    # actuators, sensors and keyframes with it. None of them belong to a
    # reusable hand asset, and they all address the arm's joints.
    for collection in (spec.actuators, spec.sensors, spec.tendons,
                       spec.pairs, spec.keys):
        for element in list(collection):
            spec.delete(element)
    # A reusable hand asset must carry no keyframe: keyframes stay pending
    # through <attach>, and the flange model nests this one two levels deep,
    # which MuJoCo cannot namespace safely. Same rule as make_hand_mjcf.py.
    spec.nkey = 0

    # Drop every asset the hand does not use, so the model is self-contained
    # and its meshdir can point at the vendored copy.
    used = {geom.meshname for body in spec.bodies for geom in body.geoms}
    for mesh in list(spec.meshes):
        if mesh.name not in used:
            spec.delete(mesh)
        else:
            mesh.file = f"{MESH_SUBDIR.name}/{Path(mesh.file).name}"
    for material in list(spec.materials):
        spec.delete(material)
    for texture in list(spec.textures):
        spec.delete(texture)
    for body in spec.bodies:
        for geom in body.geoms:
            geom.material = ""

    for name in DRIVEN:
        actuator = spec.add_actuator()
        actuator.name = name
        actuator.target = name
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        actuator.ctrlrange = [-ACTUATOR_TORQUE, ACTUATOR_TORQUE]
    return spec


def tidy(xml: str) -> str:
    """Drop what the attach dragged in that a reusable hand asset must not have.

    Two things survive every spec-level delete because MuJoCo materialises them
    at compile time rather than holding them as spec elements: the source
    scene's ``start`` keyframe (with the arm columns sliced off, so it is not
    even a valid hand pose) and the ``fr3`` default class it was authored
    under. A keyframe in particular is not merely untidy -- it stays pending
    through ``<attach>``, and the flange model nests this file two levels deep,
    where MuJoCo cannot namespace it safely.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    for tag in ("keyframe", "size"):
        for element in root.findall(tag):
            root.remove(element)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "default" and child.get("class") == "fr3":
                parent.remove(child)
    # Removing the fr3 class leaves an anonymous <default> wrapping the one
    # that is left. MuJoCo rejects a nameless nested default outright when the
    # file is pulled in as a <model> asset, so flatten it away.
    for parent in root.findall("default"):
        for child in list(parent):
            if child.tag == "default" and child.get("class") is None:
                index = list(parent).index(child)
                parent.remove(child)
                for offset, grandchild in enumerate(list(child)):
                    parent.insert(index + offset, grandchild)
    # Every geom and joint below carries explicit values, so the inherited
    # official_hand class only has to stop contradicting them -- and stop
    # naming hand_shell, a material this file no longer declares.
    for default in root.iter("default"):
        if default.get("class") != "official_hand":
            continue
        for joint in default.findall("joint"):
            joint.set("actuatorfrcrange", f"{-ACTUATOR_TORQUE:g} {ACTUATOR_TORQUE:g}")
            joint.set("damping", f"{JOINT_DYNAMICS['damping']:g}")
            joint.set("frictionloss", f"{JOINT_DYNAMICS['frictionloss']:g}")
            joint.set("armature", f"{JOINT_DYNAMICS['armature']:g}")
        for nested in default.findall("default"):
            for geom in nested.findall("geom"):
                geom.attrib.pop("material", None)
                if nested.get("class") == "hand_collision":
                    geom.set("rgba", " ".join(f"{v:g}" for v in SHELL_RGBA))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    meshes = PACKAGE / "mjcf" / MESH_SUBDIR
    meshes.mkdir(parents=True, exist_ok=True)
    spec = build(SOURCE_MJCF)
    for mesh in spec.meshes:
        name = Path(mesh.file).name
        shutil.copyfile(SOURCE_MESHES / name, meshes / name)
    print(f"copied {len(list(spec.meshes))} meshes into {meshes.relative_to(PACKAGE)}")

    # Both compile() and to_xml() resolve mesh paths against the spec's own
    # model directory, which a spec built in memory does not have. Give them
    # absolute paths, then put the committed relative ones back in the text.
    absolute = PACKAGE / "mjcf" / MESH_SUBDIR
    for mesh in spec.meshes:
        mesh.file = str(absolute / Path(mesh.file).name)
    model = spec.compile()          # fails loudly rather than writing a broken file
    print(f"compiled: {model.nbody} bodies, {model.njnt} joints, "
          f"{model.nu} actuators, {model.neq} couplings")
    xml = tidy(spec.to_xml().replace(f"{absolute}/", f"{MESH_SUBDIR.name}/"))
    args.output.write_text(xml)
    print(f"wrote {args.output.relative_to(PACKAGE)}")


if __name__ == "__main__":
    main()
