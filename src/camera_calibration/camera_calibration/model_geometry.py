"""Fixed calibration-target geometry shared by the node and model tests."""

# T_fr3_link8_apriltag_0 for the printed tag frame used by OpenCV solvePnP.
#
# The MJCF makes the requested transform chain explicit:
#
#   fr3_link8 -> apriltag_rz1: Rz(-45 deg)
#   apriltag_rz1 -> apriltag_0: xyz=(0.059, 0, 0.035) m, vertical tag pose
#
# Thus the centre resolves to (0.059/sqrt(2), -0.059/sqrt(2), 0.035) m in
# fr3_link8. The visual mesh correction does not alter this TCP transform. Keep
# this value guarded by test_model_geometry.py whenever the holder changes.
DEFAULT_HAND_TO_TAG_XYZ = (
    0.0417193000900063,
    -0.0417193000900063,
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
