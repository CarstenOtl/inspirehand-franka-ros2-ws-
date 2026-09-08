"""The generated pose program, judged the way the solver judges it.

The point of generating poses rather than hand-guiding them is that the set can
be checked before the arm moves: every pose reachable inside the joint limits,
the whole tag inside the image, the tag never grazing the camera, and the set as
a whole rotating about three axes. The last test is the one that matters most -
it feeds the program straight into the solver and asks whether the truth comes
back out, first exactly and then under realistic per-pose noise.
"""

import numpy as np
import pytest

from camera_calibration.calibration_math import (
    calibrate_eye_to_hand,
    invert_transform,
    make_transform,
    rotation_angle_deg,
)
from camera_calibration.pose_program import (
    CameraModel,
    PoseProgramError,
    PoseProgramLimits,
    generate_pose_program,
    orientation_excitation,
    pose_residual,
    solve_inverse_kinematics,
)
from camera_calibration.tag_pose import project_tag_corners
from franka_trajectory_replay.kinematics import READY_POSE, flange_transform
from franka_trajectory_replay.limits import POSITION_LOWER, POSITION_UPPER


TAG_SIZE = 0.040
# A camera on the table in front of the FR3, looking back at it: optical z towards
# the robot, optical y down.
WORLD_TO_CAMERA = make_transform(
    np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]), [0.95, 0.0, 0.55]
)
HAND_TO_TAG = make_transform(
    np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), [0.02, -0.03, 0.09]
)
CAMERA = CameraModel(
    matrix=np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]),
    distortion=np.zeros(5),
    width=640,
    height=480,
)


@pytest.fixture(scope="module")
def program():
    return generate_pose_program(
        WORLD_TO_CAMERA, HAND_TO_TAG, CAMERA, tag_size_m=TAG_SIZE, count=28, seed=3
    )


def test_inverse_kinematics_hits_a_reachable_flange_pose():
    target = flange_transform(READY_POSE + np.array([0.1, -0.2, 0.15, 0.2, -0.1, 0.1, 0.3]))
    solution = solve_inverse_kinematics(target, READY_POSE)
    assert solution is not None
    joint_positions, position_error, rotation_error = solution
    assert position_error < 1e-4
    assert rotation_error < 0.05
    np.testing.assert_allclose(
        pose_residual(flange_transform(joint_positions), target)[:3], 0.0, atol=1e-4
    )


def test_inverse_kinematics_refuses_an_unreachable_pose():
    far_away = make_transform(np.eye(3), [3.0, 0.0, 0.5])
    assert solve_inverse_kinematics(far_away, READY_POSE) is None


def test_every_pose_is_inside_the_joint_limits(program):
    for pose in program:
        assert np.all(pose.joint_positions >= POSITION_LOWER)
        assert np.all(pose.joint_positions <= POSITION_UPPER)


def test_every_pose_puts_the_whole_tag_in_the_image(program):
    limits = PoseProgramLimits()
    for pose in program:
        corners = project_tag_corners(
            pose.camera_to_tag, CAMERA.matrix, CAMERA.distortion, TAG_SIZE
        )
        assert corners[:, 0].min() >= limits.image_margin_px
        assert corners[:, 0].max() <= CAMERA.width - limits.image_margin_px
        assert corners[:, 1].min() >= limits.image_margin_px
        assert corners[:, 1].max() <= CAMERA.height - limits.image_margin_px
        assert pose.view_angle_deg <= limits.max_view_angle_deg
        assert limits.distance_range_m[0] <= pose.distance_m <= limits.distance_range_m[1]


def test_the_forward_kinematics_agree_with_the_predicted_tag_pose(program):
    for pose in program:
        np.testing.assert_allclose(
            pose.world_to_hand, flange_transform(pose.joint_positions), atol=1e-9
        )
        predicted = WORLD_TO_CAMERA @ pose.camera_to_tag
        from_robot = pose.world_to_hand @ HAND_TO_TAG
        assert np.linalg.norm(predicted[:3, 3] - from_robot[:3, 3]) < 1e-3


def test_consecutive_poses_stay_within_one_ramp(program):
    limits = PoseProgramLimits()
    for first, second in zip(program, program[1:]):
        step = np.abs(second.joint_positions - first.joint_positions).max()
        assert step <= limits.max_joint_step_rad


def test_the_set_rotates_about_three_axes(program):
    spread, informative = orientation_excitation([pose.world_to_hand for pose in program])
    assert spread > 0.05
    assert informative >= 3 * len(program)


def test_the_solver_recovers_the_truth_from_the_program(program):
    result, retained = calibrate_eye_to_hand(
        [pose.world_to_hand for pose in program],
        [pose.camera_to_tag for pose in program],
    )
    assert retained.all()
    assert np.linalg.norm(result.world_to_camera[:3, 3] - WORLD_TO_CAMERA[:3, 3]) < 1e-3
    assert (
        rotation_angle_deg((invert_transform(result.world_to_camera) @ WORLD_TO_CAMERA)[:3, :3])
        < 0.05
    )
    assert np.linalg.norm(result.hand_to_tag[:3, 3] - HAND_TO_TAG[:3, 3]) < 1e-3


def test_the_program_survives_realistic_per_pose_noise(program):
    """0.35 deg and 0.8 mm per pose is what an averaged burst of a 40 mm tag gives.

    The first hand-guided run landed at 85 mm and 20 deg. Anything near that here
    would mean the pose set, not the hardware, was the problem.
    """
    random = np.random.default_rng(0)
    observations = []
    for pose in program:
        axis = random.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = random.normal(scale=np.radians(0.35))
        skew = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
        )
        rotation = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew @ skew
        observations.append(
            pose.camera_to_tag @ make_transform(rotation, random.normal(scale=0.0008, size=3))
        )
    result, _ = calibrate_eye_to_hand(
        [pose.world_to_hand for pose in program], observations
    )
    assert np.linalg.norm(result.world_to_camera[:3, 3] - WORLD_TO_CAMERA[:3, 3]) < 0.005
    assert (
        rotation_angle_deg((invert_transform(result.world_to_camera) @ WORLD_TO_CAMERA)[:3, :3])
        < 0.5
    )
    assert result.translation_rmse_m < 0.005


def test_a_camera_that_cannot_see_the_workspace_is_refused():
    # Behind the robot and facing away: nothing it asks for is reachable.
    facing_away = make_transform(np.eye(3), [-2.0, 0.0, 0.5])
    with pytest.raises(PoseProgramError, match="usable"):
        generate_pose_program(
            facing_away,
            HAND_TO_TAG,
            CAMERA,
            tag_size_m=TAG_SIZE,
            count=8,
            seed=1,
            attempts_per_pose=8,
        )


def test_a_margin_larger_than_the_image_is_refused():
    with pytest.raises(PoseProgramError, match="no room"):
        generate_pose_program(
            WORLD_TO_CAMERA,
            HAND_TO_TAG,
            CAMERA,
            tag_size_m=TAG_SIZE,
            count=20,
            limits=PoseProgramLimits(image_margin_px=400.0),
        )


def test_too_few_poses_is_refused():
    with pytest.raises(PoseProgramError, match="at least 8"):
        generate_pose_program(WORLD_TO_CAMERA, HAND_TO_TAG, CAMERA, tag_size_m=TAG_SIZE, count=4)


def test_a_collision_veto_is_honoured():
    calls = []

    def veto(joint_positions):
        calls.append(np.asarray(joint_positions))
        return False

    with pytest.raises(PoseProgramError):
        generate_pose_program(
            WORLD_TO_CAMERA,
            HAND_TO_TAG,
            CAMERA,
            tag_size_m=TAG_SIZE,
            count=8,
            collision_check=veto,
            attempts_per_pose=8,
        )
    assert calls, "the veto was never consulted"
