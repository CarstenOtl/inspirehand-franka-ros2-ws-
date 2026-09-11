"""Source-level contract for the physical FR3/Inspire hand mount.

This deliberately does not import xacro, so the mount cannot go unchecked in
Python environments that do not carry the ROS xacro package.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
FRANKA_XACRO = (
    REPO_ROOT
    / "src"
    / "inspire_franka_description"
    / "urdf"
    / "inspire_franka.urdf.xacro"
)
HAND_XACRO = (
    REPO_ROOT
    / "src"
    / "inspire_hand_description"
    / "urdf"
    / "inspire_hand_right.macro.xacro"
)
MUJOCO_ASSETS = (
    REPO_ROOT / "assets" / "fr3_inspirehand" / "fr3_inspirehand.xml",
    REPO_ROOT / "assets" / "fr3_inspirehand" / "fr3_inspirehand_replay.xml",
)


def _values(element: ET.Element, attribute: str) -> tuple[float, ...]:
    return tuple(float(value) for value in element.get(attribute).split())


def _property(root: ET.Element, name: str) -> tuple[float, ...]:
    element = next(
        candidate
        for candidate in root.iter()
        if candidate.tag.endswith("property") and candidate.get("name") == name
    )
    return _values(element, "value")


def _quat_from_rpy(rpy: tuple[float, ...]) -> tuple[float, ...]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _quat_multiply(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def test_urdf_source_has_the_corrected_physical_mount() -> None:
    franka = ET.parse(FRANKA_XACRO).getroot()
    hand = ET.parse(HAND_XACRO).getroot()

    assert _property(franka, "hand_flange_xyz") == pytest.approx(
        (0.0, 0.0, 0.010)
    )
    flange_rpy = _property(franka, "hand_flange_rpy")
    assert flange_rpy == pytest.approx((0.0, 0.0, math.pi))

    base_joint = next(
        joint
        for joint in hand.findall(".//joint")
        if joint.get("name") == "${prefix}base_joint"
    )
    base_rpy = _values(base_joint.find("origin"), "rpy")
    effective = _quat_multiply(_quat_from_rpy(flange_rpy), _quat_from_rpy(base_rpy))
    expected_hand_base = (2**-0.5, -(2**-0.5), 0.0, 0.0)
    assert abs(sum(a * b for a, b in zip(effective, expected_hand_base))) == pytest.approx(
        1.0, abs=1e-5
    )

    adapter_joint = next(
        joint
        for joint in franka.findall(".//joint")
        if joint.get("name") == "$(arg hand_prefix)hand_adapter_flange_joint"
    )
    assert _values(adapter_joint.find("origin"), "xyz") == pytest.approx(
        (0.0, 0.0, 0.005)
    )
    adapter_mesh = next(
        mesh
        for mesh in franka.findall(".//link/visual/geometry/mesh")
        if mesh.get("filename", "").endswith("/adapter_flange.stl")
    )
    assert adapter_mesh is not None


@pytest.mark.parametrize("asset", MUJOCO_ASSETS, ids=lambda path: path.name)
def test_mujoco_source_has_the_corrected_physical_mount(asset: Path) -> None:
    root = ET.parse(asset).getroot()
    palm = root.find(".//body[@name='hand_base_link']")
    if palm is None:
        palm = root.find(".//body[@name='palm']")
    assert palm is not None
    assert _values(palm, "pos") == pytest.approx((0.0, 0.0, 0.010))
    assert _values(palm, "quat") == pytest.approx((0.0, 0.0, 0.0, 1.0))

    adapter = root.find(".//geom[@name='hand_adapter_flange']")
    assert adapter is not None
    assert _values(adapter, "pos") == pytest.approx((0.0, 0.0, 0.005))
    assert adapter.get("mesh") == "hand_adapter_flange_mesh"
