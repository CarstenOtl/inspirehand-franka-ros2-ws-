"""The vendored URDF and the driver must agree about the finger coupling.

Three places state the same six multiplier/offset pairs:

* the vendored URDF's ``<mimic>`` tags (this package),
* ``inspire_hand_driver.kinematics.DOFS`` (which computes the follower angles
  the driver publishes),
* the MuJoCo equality constraints emitted by
  ``inspire_franka_sim/scripts/make_hand_mjcf.py``.

They are duplicated rather than shared because the consumers are a xacro file, a
Python module and an XML generator, with no natural common format. This test is
what stops the duplication from rotting: re-vendor the model with different
multipliers and it fails here rather than silently mis-reporting where the
fingertips are.

The MJCF's copy is checked by ``inspire_franka_sim``'s own test.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

# The URDFs are xacro macros, so they cannot simply be parsed as URDF. The
# <mimic> tags survive verbatim through xacro expansion though, and reading them
# out of the macro source keeps this test free of a xacro dependency.
URDF_DIR = Path(__file__).resolve().parents[1] / "urdf"

# The training hand's couplings. The thumb's distal joint follows the
# intermediate, not the pitch, so the "driver" column is the joint followed.
EXPECTED = {
    "thumb_intermediate_joint": ("thumb_proximal_pitch_joint", 1.1425, 0.0),
    "thumb_distal_joint": ("thumb_intermediate_joint", 0.7508, 0.0),
    "index_intermediate_joint": ("index_proximal_joint", 1.1169, 0.0),
    "middle_intermediate_joint": ("middle_proximal_joint", 1.1169, 0.0),
    "ring_intermediate_joint": ("ring_proximal_joint", 1.1169, 0.0),
    "pinky_intermediate_joint": ("pinky_proximal_joint", 1.1169, 0.0),
}


def _mimics(side: str):
    root = ET.parse(URDF_DIR / f"inspire_hand_{side}.macro.xacro").getroot()
    out = {}
    for joint in root.iter("joint"):
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        # Strip the ${prefix} the macro interpolates into every name.
        name = joint.get("name").replace("${prefix}", "")
        out[name] = (
            mimic.get("joint").replace("${prefix}", ""),
            float(mimic.get("multiplier")),
            float(mimic.get("offset")),
        )
    return out


@pytest.mark.parametrize("side", ["right", "left"])
def test_urdf_mimics_are_what_we_expect(side):
    assert _mimics(side) == pytest.approx(EXPECTED)


def test_both_hands_share_one_coupling():
    # If a re-vendored model ever mirrors the multipliers, the driver's single
    # side-agnostic table stops being correct and has to grow a `side` axis.
    assert _mimics("right") == _mimics("left")


def test_driver_kinematics_agree_with_the_urdf():
    kinematics = pytest.importorskip(
        "inspire_hand_driver.kinematics",
        reason="inspire_hand_driver not on the path; run this from a built workspace",
    )
    urdf = _mimics("right")
    driver = {
        coupling.joint: (coupling.source or dof.joint, coupling.multiplier, coupling.offset)
        for dof in kinematics.DOFS
        for coupling in dof.couplings
    }
    assert driver == pytest.approx(urdf)


def _limits(side: str):
    root = ET.parse(URDF_DIR / f"inspire_hand_{side}.macro.xacro").getroot()
    out = {}
    for joint in root.iter("joint"):
        limit = joint.find("limit")
        if limit is not None:
            name = joint.get("name").replace("${prefix}", "")
            out[name] = (float(limit.get("lower")), float(limit.get("upper")))
    return out


def test_driver_ranges_agree_with_the_urdf():
    # The driver maps open ratios onto exactly these ranges, so on hardware
    # they decide where the hand physically goes for a given joint command.
    kinematics = pytest.importorskip(
        "inspire_hand_driver.kinematics",
        reason="inspire_hand_driver not on the path; run this from a built workspace",
    )
    urdf = _limits("right")
    assert _limits("left") == urdf
    driver = {}
    for dof in kinematics.DOFS:
        driver[dof.joint] = (dof.lower, dof.upper)
        for coupling in dof.couplings:
            driver[coupling.joint] = (coupling.lower, coupling.upper)
    assert driver == pytest.approx(urdf)
