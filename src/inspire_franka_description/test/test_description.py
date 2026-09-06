"""Expand the combined description in every supported shape and check it.

These are cheap structural checks, but they cover the failures that are
otherwise only found by launching something: a URDF with two root links, a
ros2_control block naming joints the description does not contain, or a hand
whose joint names have drifted away from the driver's.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

xacro = pytest.importorskip("xacro")

XACRO = (
    Path(__file__).resolve().parents[1] / "urdf" / "inspire_franka.urdf.xacro"
)

# The six the hand drives plus the six that follow, as the driver names them.
HAND_DRIVEN = {
    "pinky_proximal_joint",
    "ring_proximal_joint",
    "middle_proximal_joint",
    "index_proximal_joint",
    "thumb_proximal_pitch_joint",
    "thumb_proximal_yaw_joint",
}
HAND_PASSIVE = {
    "pinky_intermediate_joint",
    "ring_intermediate_joint",
    "middle_intermediate_joint",
    "index_intermediate_joint",
    "thumb_intermediate_joint",
    "thumb_distal_joint",
}
ARM_JOINTS = {f"fr3_joint{i}" for i in range(1, 8)}


def expand(**mappings) -> ET.Element:
    mappings.setdefault("hardware_type", "mock")
    return ET.fromstring(
        xacro.process_file(str(XACRO), mappings=mappings).toxml()
    )


def tree(root: ET.Element):
    """Return (link names, {child: parent})."""
    links = {link.get("name") for link in root.findall("link")}
    parents = {
        joint.find("child").get("link"): joint.find("parent").get("link")
        for joint in root.findall("joint")
    }
    return links, parents


@pytest.mark.parametrize(
    "mappings",
    [
        {},
        {"hand": "false"},
        {"arm": "false"},
        {"hand_mount": "flange"},
        {"hand_side": "left"},
        {"franka_gripper": "true"},
        {"arm_prefix": "a", "hand_prefix": "h_"},
        {"ros2_control": "true"},
        {"ros2_control": "true", "hand": "false"},
        {"ros2_control": "true", "arm": "false"},
    ],
)
def test_every_variant_has_exactly_one_root_and_no_dangling_links(mappings):
    root = expand(**mappings)
    links, parents = tree(root)

    referenced = set(parents) | set(parents.values())
    assert referenced <= links, f"joints reference undeclared links: {referenced - links}"

    roots = links - set(parents)
    assert roots == {"world"}, f"expected a single `world` root, got {roots}"


def test_the_bench_hand_hangs_off_the_world_not_the_arm():
    _, parents = tree(expand())
    assert parents["hand_mount"] == "world"


def test_the_flange_hand_hangs_off_the_arm():
    root = expand(hand_mount="flange")
    _, parents = tree(root)
    assert parents["hand_mount"] == "fr3_link8"

    mount = next(
        joint for joint in root.findall("joint")
        if joint.get("name") == "hand_mount_joint"
    )
    origin = mount.find("origin")
    assert [float(value) for value in origin.get("xyz").split()] == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    assert [float(value) for value in origin.get("rpy").split()] == pytest.approx(
        [0.0, 0.0, -math.pi / 2]
    )


def test_ros2_control_names_only_joints_that_exist():
    root = expand(ros2_control="true")
    described = {joint.get("name") for joint in root.findall("joint")}
    controlled = {
        joint.get("name")
        for block in root.findall("ros2_control")
        for joint in block.findall("joint")
    }
    assert controlled <= described, f"controlled but not described: {controlled - described}"


def test_both_assets_share_one_hardware_component():
    # mujoco_ros2_control builds one simulation per component, so a second block
    # would put the hand in a world of its own.
    root = expand(ros2_control="true")
    blocks = root.findall("ros2_control")
    assert len(blocks) == 1
    names = {joint.get("name") for joint in blocks[0].findall("joint")}
    assert ARM_JOINTS <= names
    assert (HAND_DRIVEN | HAND_PASSIVE) <= names


def test_only_the_driven_hand_joints_take_commands():
    # A follower cannot be commanded independently of the joint that drives it,
    # so it must not expose a command interface for a controller to claim.
    root = expand(ros2_control="true")
    block = root.find("ros2_control")
    commanded = {
        joint.get("name")
        for joint in block.findall("joint")
        if joint.find("command_interface") is not None
    }
    assert HAND_DRIVEN <= commanded
    assert not (HAND_PASSIVE & commanded)


def test_hand_joint_names_match_the_driver():
    kinematics = pytest.importorskip(
        "inspire_hand_driver.kinematics",
        reason="inspire_hand_driver not on the path; run from a built workspace",
    )
    described = {joint.get("name") for joint in expand().findall("joint")}
    assert set(kinematics.ALL_JOINTS) <= described


@pytest.mark.parametrize(
    "mappings",
    [
        {"arm": "false", "hand": "false"},
        {"arm": "false", "hand_mount": "flange"},
        {"hand_side": "bogus"},
        {"hand_mount": "bogus"},
    ],
)
def test_unsupported_combinations_fail_loudly(mappings):
    # Silently emitting a half-built description is much worse than refusing to
    # expand, because the failure then surfaces as a puzzling TF gap.
    with pytest.raises(Exception):
        expand(**mappings)
