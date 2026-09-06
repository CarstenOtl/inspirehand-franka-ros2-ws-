#!/usr/bin/env python3
"""Regenerate this package's meshes and xacro from the upstream dex-urdf model.

Everything under ``urdf/`` and ``meshes/`` is *generated*. Edit this script, not
the output. Run it from anywhere:

    python3 scripts/vendor_from_dex_urdf.py --source /path/to/dex_urdf/robots

See ``MODEL_PROVENANCE.md`` for where the source comes from and what licence it
carries.

What the transform does, and why
--------------------------------
The upstream model is a pair of flat, collision-only URDFs plus two disjoint
mesh sets. Four things have to change before it is usable as a ROS description
that can be bolted onto an arm:

1. **It becomes a xacro macro.** A flat URDF cannot be instantiated twice, given
   a prefix, or attached to a parent link. The macro takes ``prefix``, and a
   ``parent``/``xyz``/``rpy`` mount.

2. **Visuals are added.** Upstream's URDF has no ``<visual>`` at all -- it
   describes only collision geometry, so RViz would render nothing. The visual
   meshes exist upstream as glTF binaries in a separate tree; this script
   converts them to binary STL, which RViz and MuJoCo both read without
   argument, and attaches one to each link.

3. **Meshes are de-duplicated and re-rooted.** Four links share
   ``index_proximal`` and two share ``index_intermediate``; the copies land once
   each under ``meshes/<side>/``, and the ``../../meshes/...`` relative paths
   become ``package://`` URIs that resolve wherever the package is installed.

4. **``base`` is renamed to ``hand_mount``.** A link called ``base`` is fine in
   a standalone hand and a hazard in a combined description. Everything else
   keeps its upstream name, so joint names still match
   ``inspire_hand_driver.kinematics``.

The ``<mimic>`` tags are carried through unchanged. Nothing in ROS propagates
them at runtime -- the driver and the MJCF generator each implement the coupling
themselves -- but they are the authoritative record of what the linkage does,
and ``test/test_mimic_matches_driver.py`` asserts the three copies agree.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

PACKAGE = "inspire_hand_description"
SIDES = ("right", "left")

# Upstream calls the mount frame `base`. Too generic to let loose in a URDF that
# also contains an arm.
RENAMED_LINKS = {"base": "hand_mount"}

# Links whose visual mesh basename differs from their collision one. Only the
# palm, which upstream gives no collision mesh at all (it is a set of primitives)
# but does ship a visual for.
VISUAL_ONLY = {"hand_base_link": "base_link"}

HEADER = """<?xml version="1.0"?>
<!--
  GENERATED FILE - do not edit.

  Produced by inspire_hand_description/scripts/vendor_from_dex_urdf.py from the
  upstream dex-urdf Inspire RH56 model. See MODEL_PROVENANCE.md.

  Regenerate by running that script; it takes the upstream tree as its
  "source" argument. XML comments cannot contain a double hyphen, which is why
  the command line is described rather than shown.
-->
"""


def _strip_side(basename: str, side: str) -> str:
    """`right_index_proximal.obj` -> `index_proximal.obj`; the side is the directory."""
    prefix = f"{side}_"
    return basename[len(prefix):] if basename.startswith(prefix) else basename


def collect_meshes(root: ET.Element, side: str) -> Dict[str, str]:
    """Map link name -> collision mesh stem (no extension), for links that have one."""
    out: Dict[str, str] = {}
    for link in root.findall("link"):
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is not None:
                stem = Path(_strip_side(Path(mesh.get("filename")).name, side)).stem
                out[link.get("name")] = stem
    return out


def convert_meshes(source: Path, dest: Path, side: str, stems: List[str]) -> None:
    """Copy the OBJ collision meshes and convert the glTF visuals to binary STL."""
    import trimesh

    (dest / "collision").mkdir(parents=True, exist_ok=True)
    (dest / "visual").mkdir(parents=True, exist_ok=True)

    for stem in stems:
        obj = source / "meshes" / "obj_meshes" / "inspire_hand" / f"{side}_{stem}.obj"
        if obj.exists():
            shutil.copy2(obj, dest / "collision" / f"{stem}.obj")
        elif stem not in VISUAL_ONLY.values():
            print(f"  ! no collision mesh for {stem}", file=sys.stderr)

        glb = source / "meshes" / "glb_meshes" / "inspire_hand" / "visual" / f"{side}_{stem}.glb"
        if not glb.exists():
            print(f"  ! no visual mesh for {stem}", file=sys.stderr)
            continue
        # force='mesh' flattens the glTF scene graph into one mesh, baking in the
        # node transforms. Verified against the OBJ counterparts: same frame,
        # same units (metres), extents agreeing to well under a millimetre.
        mesh = trimesh.load(glb, force="mesh", process=False)
        mesh.export(dest / "visual" / f"{stem}.stl", file_type="stl")


def _fmt(element: ET.Element, indent: str) -> List[str]:
    """Serialise one element, pretty-printed, as a list of lines."""
    ET.indent(element, space="  ")
    text = ET.tostring(element, encoding="unicode").rstrip()
    return [indent + line for line in text.splitlines()]


def build_macro(root: ET.Element, side: str, meshes: Dict[str, str]) -> str:
    """Turn one flat URDF into a xacro macro."""
    def name(original: str) -> str:
        """Prefix a link or joint name, applying any rename."""
        return "${prefix}" + RENAMED_LINKS.get(original, original)

    lines: List[str] = [
        HEADER.rstrip(),
        '<robot xmlns:xacro="http://www.ros.org/wiki/xacro">',
        "",
        f'  <xacro:macro name="inspire_hand_{side}"',
        '    params="prefix:=\'\' parent:=\'\' xyz:=\'0 0 0\' rpy:=\'0 0 0\'">',
        "",
        "    <!-- Mount. With no parent the hand stands alone and hand_mount is the",
        "         description's root link; with one, it is welded to that link. -->",
        '    <xacro:unless value="${parent == \'\'}">',
        f'      <joint name="{name("hand_mount")}_joint" type="fixed">',
        '        <origin xyz="${xyz}" rpy="${rpy}"/>',
        '        <parent link="${parent}"/>',
        f'        <child link="{name("hand_mount")}"/>',
        "      </joint>",
        "    </xacro:unless>",
        "",
    ]

    for link in root.findall("link"):
        original = link.get("name")
        link.set("name", name(original))
        stem = meshes.get(original)
        visual_stem = VISUAL_ONLY.get(original, stem)

        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is not None:
                mesh.set(
                    "filename",
                    f"package://{PACKAGE}/meshes/{side}/collision/{stem}.obj",
                )

        if visual_stem is not None:
            visual = ET.Element("visual")
            ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            geometry = ET.SubElement(visual, "geometry")
            ET.SubElement(
                geometry,
                "mesh",
                {"filename": f"package://{PACKAGE}/meshes/{side}/visual/{visual_stem}.stl"},
            )
            material = ET.SubElement(visual, "material", {"name": "${prefix}inspire_hand_shell"})
            ET.SubElement(material, "color", {"rgba": "0.15 0.15 0.17 1.0"})
            # Before <collision>, which is the conventional URDF ordering.
            link.insert(len(link.findall("inertial")), visual)

        lines += _fmt(link, "    ") + [""]

    for joint in root.findall("joint"):
        joint.set("name", name(joint.get("name")))
        joint.find("parent").set("link", name(joint.find("parent").get("link")))
        joint.find("child").set("link", name(joint.find("child").get("link")))
        mimic = joint.find("mimic")
        if mimic is not None:
            mimic.set("joint", name(mimic.get("joint")))
        lines += _fmt(joint, "    ") + [""]

    lines += ["  </xacro:macro>", "</robot>", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory holding urdf/inspire_hand/ and meshes/{obj,glb}_meshes/.",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="This package's root (default: the parent of scripts/).",
    )
    args = parser.parse_args(argv)

    for side in SIDES:
        urdf = args.source / "urdf" / "inspire_hand" / f"inspire_hand_{side}.urdf"
        if not urdf.exists():
            print(f"missing {urdf}", file=sys.stderr)
            return 1
        print(f"{side}:")

        root = ET.parse(urdf).getroot()
        meshes = collect_meshes(root, side)
        stems = sorted(set(meshes.values()) | set(VISUAL_ONLY.values()))
        convert_meshes(args.source, args.package_root / "meshes" / side, side, stems)
        print(f"  {len(stems)} meshes")

        out = args.package_root / "urdf" / f"inspire_hand_{side}.macro.xacro"
        out.write_text(build_macro(root, side, meshes))
        print(f"  wrote {out.relative_to(args.package_root)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
