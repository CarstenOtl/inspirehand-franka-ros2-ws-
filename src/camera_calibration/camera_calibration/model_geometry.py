"""Fixed calibration-target geometry shared by the node and model tests."""

# T_fr3_link8_apriltag_0 for the printed tag frame used by OpenCV solvePnP.
#
# The MJCF makes the requested transform chain explicit:
#
#   fr3_link8 -> apriltag_rz1: Rz(-45 deg)
#   apriltag_rz1 -> apriltag_0: xyz=(0.060, 0, 0.035) m, vertical tag pose
#
# Thus the centre resolves to (0.060/sqrt(2), -0.060/sqrt(2), 0.035) m in
# fr3_link8. The visual mesh correction does not alter this TCP transform. Keep
# this value guarded by test_model_geometry.py whenever the holder changes.
DEFAULT_HAND_TO_TAG_XYZ = (
    0.042426406871192853,
    -0.042426406871192853,
    0.035,
)
DEFAULT_HAND_TO_TAG_QUATERNION_XYZW = (
    -0.270598050073099,
    0.653281482438188,
    -0.653281482438188,
    0.270598050073099,
)

# The MuJoCo body/site origin is now the printed tag centre and its axes match
# the detector axes. Plate thickness and OBJ UV corrections live on the geom.
TAG_BODY_TO_PRINTED_TAG_XYZ = (0.0, 0.0, 0.0)
TAG_BODY_TO_PRINTED_TAG_QUATERNION_XYZW = (
    0.0,
    0.0,
    0.0,
    1.0,
)

# The tag holder is calibration-only hardware, so the shared robot asset
# (which also renders the policy camera) no longer carries it. Calibration
# scenes add it back with add_apriltag_mount(). Paths are relative to the
# directory of assets/fr3_inspirehand/fr3_inspirehand.xml.
APRILTAG_TEXTURE_FILE = "textures/apriltag_36h11_id0.png"
APRILTAG_PLATE_MESH_FILE = "hand/apriltag_36h11_id0_dorsal.obj"


def add_apriltag_mount(root, asset_dir=None):
    """Mount the printed tag on ``fr3_link8`` of a parsed robot MJCF in place.

    ``root`` is an ``xml.etree.ElementTree`` element of the combined FR3/hand
    scene. Pass ``asset_dir`` to write absolute file paths, as needed when the
    result is loaded from a string. A scene that already has the mount is left
    unchanged. Returns ``root``.
    """
    from pathlib import Path
    import xml.etree.ElementTree as ET

    flange = root.find(".//body[@name='fr3_link8']")
    if flange is None:
        raise ValueError("MJCF has no fr3_link8 body to carry the AprilTag")
    if flange.find("body[@name='apriltag_rz1']") is not None:
        return root
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    def _file(relative):
        return str((Path(asset_dir) / relative).resolve()) if asset_dir else relative

    ET.SubElement(asset, "texture", {
        "type": "2d", "colorspace": "auto", "name": "apriltag_36h11_id0_texture",
        "file": _file(APRILTAG_TEXTURE_FILE),
    })
    ET.SubElement(asset, "material", {
        "name": "apriltag_36h11_id0", "texture": "apriltag_36h11_id0_texture",
        "rgba": "1 1 1 1", "texuniform": "false", "reflectance": "0",
        "specular": "0", "shininess": "0",
    })
    ET.SubElement(asset, "mesh", {
        "name": "hand_apriltag_36h11_id0_dorsal", "content_type": "model/obj",
        "file": _file(APRILTAG_PLATE_MESH_FILE),
    })

    # Rz1 is the first tag-mount frame: rotate -45 deg about fr3_link8 Z. The
    # tag centre is then translated 60 mm along Rz1 +X and 35 mm along Rz1 +Z.
    # The tag face is vertical with zero in-plane print clocking. The geom
    # transform puts the printed surface at the tag-body origin and cancels
    # the OBJ's +90 deg UV yaw.
    rz1 = ET.Element("body", {
        "name": "apriltag_rz1", "quat": "0.923879532511287 0 0 -0.382683432365090",
    })
    tag = ET.SubElement(rz1, "body", {
        "name": "apriltag_0", "pos": "0.060 0 0.035", "quat": "0.5 -0.5 0.5 -0.5",
    })
    ET.SubElement(tag, "geom", {
        "name": "apriltag_36h11_id0", "type": "mesh",
        "mesh": "hand_apriltag_36h11_id0_dorsal", "material": "apriltag_36h11_id0",
        "pos": "0 0 -0.0023", "quat": "0.707106781186548 0 0 -0.707106781186547",
        "rgba": "1 1 1 1", "contype": "0", "conaffinity": "0", "group": "2",
        "density": "0",
    })
    ET.SubElement(tag, "site", {
        "name": "apriltag_0_frame", "type": "box", "size": "0.020 0.020 0.0006",
        "rgba": "0 0 0 0", "group": "4",
    })
    # Before the hand, as in the asset the calibration was recorded with.
    hand = flange.find("body[@name='hand_base_link']")
    flange.insert(list(flange).index(hand) if hand is not None else len(flange), rz1)
    return root
