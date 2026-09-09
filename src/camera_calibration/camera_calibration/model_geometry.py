"""Fixed calibration-target geometry shared by the node and model tests."""

# T_fr3_link8_apriltag_0 for the printed tag frame used by OpenCV solvePnP.
#
# This is composed from the flange-mounted hand in
# assets/fr3_inspirehand/fr3_inspirehand.xml:
#
#   fr3_link8 -> hand_base_link: xyz=(0, 0, 0.010), yaw=pi
#   hand_base_link -> apriltag_0: the measured pose stored in that MJCF
#   tag body -> printed tag: xyz=(0, 0, 0.002), yaw=pi/2
#
# Keep this value guarded by test_model_geometry.py whenever the physical hand,
# adapter, or AprilTag placement changes.
DEFAULT_HAND_TO_TAG_XYZ = (
    0.0187392994594171,
    0.000541679480600568,
    0.102009980013799,
)
DEFAULT_HAND_TO_TAG_QUATERNION_XYZW = (
    0.498190676972095,
    -0.499244910192612,
    0.501808192566079,
    -0.500748546576444,
)

# The surface-following mesh is 2 mm above the apriltag_0 body and its UV
# mapping rotates the printed detector axes +90 degrees around body +Z.
TAG_BODY_TO_PRINTED_TAG_XYZ = (0.0, 0.0, 0.002)
TAG_BODY_TO_PRINTED_TAG_QUATERNION_XYZW = (
    0.0,
    0.0,
    0.7071067811865475,
    0.7071067811865476,
)
