"""Fixed calibration-target geometry shared by the node and model tests."""

# T_fr3_link8_apriltag_0 for the printed tag frame used by OpenCV solvePnP.
#
# This is composed from the flange-mounted hand in
# assets/fr3_inspirehand/fr3_inspirehand.xml:
#
#   fr3_link8 -> hand_base_link: xyz=(0, 0, 0.010), yaw=pi
#   hand_base_link -> apriltag_0: the measured pose stored in that MJCF
#   tag body -> printed tag: xyz=(0, 0, 0.0023), yaw=pi/2
#
# Keep this value guarded by test_model_geometry.py whenever the physical hand,
# adapter, or AprilTag placement changes.
DEFAULT_HAND_TO_TAG_XYZ = (
    0.019039294855215,
    0.000545907195080103,
    0.109999737737734,
)
DEFAULT_HAND_TO_TAG_QUATERNION_XYZW = (
    0.498190676972095,
    -0.499244910192612,
    0.501808192566079,
    -0.500748546576444,
)

# The printed face is on top of the 2.3 mm flat plate. Its UV mapping rotates
# the printed detector axes +90 degrees around the plate body's +Z.
TAG_BODY_TO_PRINTED_TAG_XYZ = (0.0, 0.0, 0.0023)
TAG_BODY_TO_PRINTED_TAG_QUATERNION_XYZW = (
    0.0,
    0.0,
    0.7071067811865475,
    0.7071067811865476,
)
