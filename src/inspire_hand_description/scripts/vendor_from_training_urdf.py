#!/usr/bin/env python3
"""Regenerate this package's meshes and xacro from the training hand's URDF.

Everything under ``urdf/inspire_hand_{right,left}.macro.xacro`` and
``meshes/`` is *generated*. Edit this script, not the output. Run it from
anywhere, pointing it at forgeUltra's hand URDF:

    python3 scripts/vendor_from_training_urdf.py \\
        --source /path/to/forgeUltra/assets/fr3_inspirehand/urdf/inspirehand_right.urdf

See ``MODEL_PROVENANCE.md`` for where that URDF comes from.

Why this model
--------------
The student policy was trained on this URDF: forgeUltra's official Tiangong
2.0 Pro right hand, with the five tip frames forgeUltra adds. The workspace
used to describe the hand with dex-urdf's independent derivation, which puts
the pads 58 mm vs 114 mm apart at the same joint command, weighs 0.19 kg
against 0.54 kg, and maps the actuator range onto different joint angles. One
model now serves TF, MuJoCo, the driver's joint mapping and the policy.

What the transform does
-----------------------
1. **Names.** Links and joints are renamed to the names the workspace already
   speaks (``hand_base_link``, ``index_proximal_joint``, ...), so the driver,
   the controllers, ``config/pids.yaml`` and the sim bridge address the same
   joints as before. Only the geometry underneath changes. The map is
   ``LINKS``/``JOINTS`` below.

2. **Everything else is taken verbatim.** Link origins, masses, inertias, joint
   axes, ranges and mimic couplings -- including the thumb's chained coupling,
   where the distal joint follows the *intermediate* (0.7508) rather than the
   pitch. The one exception is ``<limit effort/velocity>``: training's 50 N m /
   1 rad/s suit PhysX's implicit drives, and the workspace keeps its own 1 N m /
   0.5 rad/s, which nothing on the real hand reads.

3. **Mount.** A massless ``hand_mount`` link is added above the palm with an
   identity joint. forgeUltra mounts the palm itself at the flange (10 mm out,
   yawed by pi), which is what ``hand_mount`` gets from the arm description, so
   the palm lands in the same place in both.

4. **Meshes.** The 13 official STL files are copied byte-for-byte and renamed
   after their link, under ``meshes/right/``. Each serves as both visual and
   collision geometry, as in the source.

5. **The left hand is the exact mirror of the right** across the hand_mount
   frame's XZ plane (y -> -y), the plane dex-urdf's two hands are mirrored
   across. Every link frame is conjugated by the reflection so it stays
   right-handed; joint axes become ``-M a`` so a positive angle still closes the
   finger; inertias become ``M I M``; mesh vertices are reflected and their
   winding flipped. Ranges and couplings are unchanged, so the driver's single
   side-agnostic table stays correct. There is no official left hand to check
   this against.

The ``<mimic>`` tags are carried through. Nothing in ROS propagates them at
runtime -- the driver and the MJCF generator each implement the coupling
themselves -- and ``test/test_mimic_matches_driver.py`` asserts the copies
agree.
"""

from __future__ import annotations

import argparse
import math
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PACKAGE = "inspire_hand_description"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

#: forgeUltra name -> workspace name.
LINKS = {
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
    "thumb_tip": "thumb_tip",
    "index_tip": "index_tip",
    "middle_tip": "middle_tip",
    "ring_tip": "ring_tip",
    "little_tip": "pinky_tip",
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
    "thumb_tip_fixed": "thumb_tip_joint",
    "index_tip_fixed": "index_tip_joint",
    "middle_tip_fixed": "middle_tip_joint",
    "ring_tip_fixed": "ring_tip_joint",
    "little_tip_fixed": "pinky_tip_joint",
}
TIP_LINKS = {"thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"}

# See point 2 of the module docstring.
EFFORT_LIMIT = "1"
VELOCITY_LIMIT = "0.5"
SHELL_RGBA = "0.15 0.15 0.17 1.0"

#: The left hand's reflection, in the hand_mount frame.
MIRROR = np.diag([1.0, -1.0, 1.0])


# --- small rigid-body helpers -------------------------------------------------

def _vec(text: str | None) -> np.ndarray:
    return np.array([float(v) for v in (text or "0 0 0").split()], dtype=float)


def _rpy_to_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _matrix_to_rpy(m: np.ndarray) -> np.ndarray:
    pitch = math.asin(max(-1.0, min(1.0, -m[2, 0])))
    if abs(math.cos(pitch)) > 1.0e-9:
        roll = math.atan2(m[2, 1], m[2, 2])
        yaw = math.atan2(m[1, 0], m[0, 0])
    else:  # gimbal lock: put everything in yaw
        roll = 0.0
        yaw = math.atan2(-m[0, 1], m[1, 1])
    return np.array([roll, pitch, yaw])


def _fmt(values) -> str:
    out = []
    for v in values:
        v = 0.0 if abs(v) < 5.0e-13 else float(v)
        out.append(f"{v:.12g}")
    return " ".join(out)


# --- reading the source -------------------------------------------------------

def read_source(path: Path) -> dict:
    """Pull links, joints and mesh names out of forgeUltra's URDF."""
    root = ET.parse(path).getroot()
    links = {}
    for link in root.findall("link"):
        name = link.get("name")
        if name not in LINKS:
            raise SystemExit(f"unexpected link {name!r} in {path}")
        entry = {"mesh": None, "inertial": None}
        inertial = link.find("inertial")
        if inertial is not None and name not in {"thumb_tip", "index_tip", "middle_tip",
                                                 "ring_tip", "little_tip"}:
            origin = inertial.find("origin")
            inertia = inertial.find("inertia").attrib
            entry["inertial"] = {
                "xyz": _vec(origin.get("xyz")),
                "rpy": _vec(origin.get("rpy")),
                "mass": float(inertial.find("mass").get("value")),
                "I": np.array([
                    [float(inertia["ixx"]), float(inertia["ixy"]), float(inertia["ixz"])],
                    [float(inertia["ixy"]), float(inertia["iyy"]), float(inertia["iyz"])],
                    [float(inertia["ixz"]), float(inertia["iyz"]), float(inertia["izz"])],
                ]),
            }
        meshes = {m.get("filename") for m in link.iter("mesh")}
        for geometry_parent in ("visual", "collision"):
            for element in link.findall(geometry_parent):
                origin = element.find("origin")
                if origin is not None and (
                    np.any(_vec(origin.get("xyz"))) or np.any(_vec(origin.get("rpy")))
                ):
                    raise SystemExit(f"{name}: non-identity {geometry_parent} origin")
        if len(meshes) > 1:
            raise SystemExit(f"{name}: visual and collision meshes differ: {meshes}")
        if meshes:
            entry["mesh"] = (path.parent / meshes.pop()).resolve()
        links[LINKS[name]] = entry

    joints = []
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in JOINTS:
            raise SystemExit(f"unexpected joint {name!r} in {path}")
        origin = joint.find("origin")
        entry = {
            "name": JOINTS[name],
            "type": joint.get("type"),
            "parent": LINKS[joint.find("parent").get("link")],
            "child": LINKS[joint.find("child").get("link")],
            "xyz": _vec(origin.get("xyz")),
            "rpy": _vec(origin.get("rpy")),
        }
        if entry["type"] == "revolute":
            limit = joint.find("limit")
            entry["axis"] = _vec(joint.find("axis").get("xyz"))
            entry["lower"] = float(limit.get("lower"))
            entry["upper"] = float(limit.get("upper"))
            mimic = joint.find("mimic")
            if mimic is not None:
                entry["mimic"] = (
                    JOINTS[mimic.get("joint")],
                    float(mimic.get("multiplier", "1")),
                    float(mimic.get("offset", "0")),
                )
        joints.append(entry)
    return {"links": links, "joints": joints}


# --- mirroring ----------------------------------------------------------------

def mirrored(model: dict) -> dict:
    """The left hand: conjugate every frame by MIRROR. See the docstring, point 5."""
    m = MIRROR
    links = {}
    for name, entry in model["links"].items():
        out = dict(entry)
        inertial = entry["inertial"]
        if inertial is not None:
            # Fold the inertial rotation into the tensor first, so the mirrored
            # one can be written with an identity origin rotation.
            r = _rpy_to_matrix(inertial["rpy"])
            tensor = r @ inertial["I"] @ r.T
            out["inertial"] = {
                "xyz": m @ inertial["xyz"],
                "rpy": np.zeros(3),
                "mass": inertial["mass"],
                "I": m @ tensor @ m,
            }
        links[name] = out
    joints = []
    for entry in model["joints"]:
        out = dict(entry)
        out["xyz"] = m @ entry["xyz"]
        out["rpy"] = _matrix_to_rpy(m @ _rpy_to_matrix(entry["rpy"]) @ m)
        if "axis" in entry:
            out["axis"] = -(m @ entry["axis"])
        joints.append(out)
    return {"links": links, "joints": joints}


def _read_binary_stl(path: Path):
    data = path.read_bytes()
    (count,) = struct.unpack_from("<I", data, 80)
    if len(data) != 84 + 50 * count:
        raise SystemExit(f"{path}: not a binary STL")
    records = np.frombuffer(
        data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
        count=count, offset=84,
    )
    return data[:80], records.copy()


def write_mirrored_stl(source: Path, target: Path) -> None:
    header, records = _read_binary_stl(source)
    m = MIRROR.astype(np.float32)
    records["n"] = records["n"] @ m.T
    vertices = records["v"] @ m.T
    # A reflection turns every triangle inside out; swap two corners to put
    # the outward normal back on the outside.
    records["v"] = vertices[:, [0, 2, 1], :]
    target.write_bytes(header + struct.pack("<I", len(records)) + records.tobytes())


# --- writing ------------------------------------------------------------------

def mesh_name(link: str) -> str:
    return f"{link}.stl"


def write_macro(model: dict, side: str, source_label: str) -> str:
    lines = [
        '<?xml version="1.0"?>',
        "<!--",
        "  GENERATED FILE - do not edit.",
        "",
        "  Produced by inspire_hand_description/scripts/vendor_from_training_urdf.py",
        f"  from {source_label}, the hand the policy was trained on.",
        "  See MODEL_PROVENANCE.md." + (
            " This left hand is the exact mirror of the right one;" if side == "left" else ""),
    ]
    if side == "left":
        lines.append("  there is no official left hand to check it against.")
    lines += [
        "",
        "  Regenerate by running that script; it takes the source URDF as its",
        '  "source" argument. XML comments cannot contain a double hyphen, which is',
        "  why the command line is described rather than shown.",
        "-->",
        '<robot xmlns:xacro="http://www.ros.org/wiki/xacro">',
        "",
        f'  <xacro:macro name="inspire_hand_{side}"',
        "    params=\"prefix:='' parent:='' xyz:='0 0 0' rpy:='0 0 0'\">",
        "",
        "    <!-- Mount. With no parent the hand stands alone and hand_mount is the",
        "         description's root link; with one, it is welded to that link. -->",
        "    <xacro:unless value=\"${parent == ''}\">",
        '      <joint name="${prefix}hand_mount_joint" type="fixed">',
        '        <origin xyz="${xyz}" rpy="${rpy}"/>',
        '        <parent link="${parent}"/>',
        '        <child link="${prefix}hand_mount"/>',
        "      </joint>",
        "    </xacro:unless>",
        "",
        '    <link name="${prefix}hand_mount" />',
        "",
    ]
    for name, entry in model["links"].items():
        if name in TIP_LINKS:
            continue
        inertial = entry["inertial"]
        tensor = inertial["I"]
        uri = f"package://{PACKAGE}/meshes/{side}/{mesh_name(name)}"
        lines += [
            f'    <link name="${{prefix}}{name}">',
            "      <inertial>",
            f'        <origin xyz="{_fmt(inertial["xyz"])}" rpy="{_fmt(inertial["rpy"])}" />',
            f'        <mass value="{inertial["mass"]:.12g}" />',
            f'        <inertia ixx="{tensor[0, 0]:.12g}" ixy="{tensor[0, 1]:.12g}"'
            f' ixz="{tensor[0, 2]:.12g}" iyy="{tensor[1, 1]:.12g}"'
            f' iyz="{tensor[1, 2]:.12g}" izz="{tensor[2, 2]:.12g}" />',
            "      </inertial>",
            "      <visual>",
            '        <origin xyz="0 0 0" rpy="0 0 0" />',
            "        <geometry>",
            f'          <mesh filename="{uri}" />',
            "        </geometry>",
            '        <material name="${prefix}inspire_hand_shell">',
            f'          <color rgba="{SHELL_RGBA}" />',
            "        </material>",
            "      </visual>",
            "      <collision>",
            '        <origin xyz="0 0 0" rpy="0 0 0" />',
            "        <geometry>",
            f'          <mesh filename="{uri}" />',
            "        </geometry>",
            "      </collision>",
            "    </link>",
            "",
        ]
    for name in sorted(TIP_LINKS, key=list(model["links"]).index):
        lines += [f'    <link name="${{prefix}}{name}" />', ""]

    lines += [
        '    <joint name="${prefix}base_joint" type="fixed">',
        '      <parent link="${prefix}hand_mount" />',
        '      <child link="${prefix}hand_base_link" />',
        '      <origin xyz="0 0 0" rpy="0 0 0" />',
        "    </joint>",
        "",
    ]
    for joint in model["joints"]:
        lines += [
            f'    <joint name="${{prefix}}{joint["name"]}" type="{joint["type"]}">',
            f'      <origin xyz="{_fmt(joint["xyz"])}" rpy="{_fmt(joint["rpy"])}" />',
            f'      <parent link="${{prefix}}{joint["parent"]}" />',
            f'      <child link="${{prefix}}{joint["child"]}" />',
        ]
        if joint["type"] == "revolute":
            lines += [
                f'      <axis xyz="{_fmt(joint["axis"])}" />',
                f'      <limit lower="{joint["lower"]:.12g}" upper="{joint["upper"]:.12g}"'
                f' effort="{EFFORT_LIMIT}" velocity="{VELOCITY_LIMIT}" />',
            ]
            if "mimic" in joint:
                driver, multiplier, offset = joint["mimic"]
                lines.append(
                    f'      <mimic joint="${{prefix}}{driver}"'
                    f' multiplier="{multiplier:.12g}" offset="{offset:.12g}" />'
                )
        lines += ["    </joint>", ""]
    lines += ["  </xacro:macro>", "</robot>", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True,
                        help="forgeUltra's assets/fr3_inspirehand/urdf/inspirehand_right.urdf")
    args = parser.parse_args()
    source = args.source.resolve()

    right = read_source(source)
    models = {"right": right, "left": mirrored(right)}
    for side, model in models.items():
        meshes = PACKAGE_ROOT / "meshes" / side
        if meshes.exists():
            shutil.rmtree(meshes)
        meshes.mkdir(parents=True)
        for name, entry in model["links"].items():
            if entry["mesh"] is None:
                continue
            target = meshes / mesh_name(name)
            if side == "right":
                shutil.copyfile(entry["mesh"], target)
            else:
                write_mirrored_stl(entry["mesh"], target)
        macro = PACKAGE_ROOT / "urdf" / f"inspire_hand_{side}.macro.xacro"
        macro.write_text(write_macro(model, side, "forgeUltra's inspirehand_right.urdf"))
        print(f"wrote {macro.relative_to(PACKAGE_ROOT)} and "
              f"{len(list(meshes.iterdir()))} meshes under {meshes.relative_to(PACKAGE_ROOT)}")


if __name__ == "__main__":
    main()
