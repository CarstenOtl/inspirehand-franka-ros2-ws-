from types import SimpleNamespace

import numpy as np
import pytest

from policy_rollout import forge_osc as fo
from policy_rollout.hardware import (
    GripCycleCoordinator,
    RgbdFrameSynchronizer,
    TrainingFrameAdapter,
    assert_policy_camera_frames,
    grasp_frame_from_tips,
    grasp_in_flange,
    grasp_z_transport_at_reset,
    image_message_to_numpy,
    physical_hand_state_to_policy,
    policy_goal_base,
    policy_tool_transform,
)


def _stamped(seconds, value):
    whole = int(seconds)
    nanoseconds = round((seconds - whole) * 1.0e9)
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=whole, nanosec=nanoseconds)
        ),
        value=value,
    )


def _image(array, encoding):
    return SimpleNamespace(
        encoding=encoding,
        height=array.shape[0],
        width=array.shape[1],
        step=array.strides[0],
        data=array.tobytes(),
    )


def test_image_decoder_converts_bgr_and_depth_units():
    bgr = np.array([[[3, 2, 1], [6, 5, 4]]], dtype=np.uint8)
    rgb, units = image_message_to_numpy(_image(bgr, "bgr8"))
    assert units == "rgb"
    assert rgb.tolist() == [[[1, 2, 3], [4, 5, 6]]]

    depth_mm = np.array([[100, 200]], dtype=np.uint16)
    decoded_mm, units_mm = image_message_to_numpy(_image(depth_mm, "16UC1"))
    assert units_mm == "millimetres"
    assert decoded_mm.tolist() == [[100, 200]]

    depth_m = np.array([[0.1, 0.2]], dtype=np.float32)
    decoded_m, units_m = image_message_to_numpy(_image(depth_m, "32FC1"))
    assert units_m == "metres"
    assert decoded_m == pytest.approx(depth_m)


def test_image_decoder_ignores_row_padding():
    padded = np.array([[[1, 2, 3], [4, 5, 6], [99, 99, 99]]], dtype=np.uint8)
    message = SimpleNamespace(
        encoding="rgb8", height=1, width=2, step=9, data=padded.tobytes()
    )
    image, _ = image_message_to_numpy(message)
    assert image.tolist() == [[[1, 2, 3], [4, 5, 6]]]


def test_physical_hand_feedback_is_mapped_back_to_policy_coordinates():
    from inspire_hand_driver import command_overlays
    from inspire_hand_driver import kinematics as kin

    logical_q = np.array([1.08, 0.05, 0.22])
    logical_dq = np.array([0.4, -0.2, 0.1])
    physical_positions = np.array([1.0, 1.1, 1.2, logical_q[2], logical_q[1], 0.0])
    physical_velocities = np.array([0.0, 0.0, 0.0, logical_dq[2], logical_dq[1], 0.0])
    thumb_dof = kin.dof_index(command_overlays.THUMB_ABDUCTION_JOINT)
    logical_ratio = kin.rad_to_open_ratio(thumb_dof, logical_q[0])
    physical_ratio = command_overlays.apply_open_ratio_overlay(
        thumb_dof, logical_ratio
    )
    physical_positions[-1] = kin.open_ratio_to_rad(thumb_dof, physical_ratio)
    physical_velocities[-1] = (
        1.0 - command_overlays.THUMB_ABDUCTION_ZERO_OPEN_RATIO
    ) * logical_dq[0]

    q_policy, dq_policy = physical_hand_state_to_policy(
        physical_positions, physical_velocities
    )

    assert q_policy == pytest.approx(logical_q)
    assert dq_policy == pytest.approx(logical_dq)


def test_policy_camera_frames_must_all_be_registered_to_colour():
    def message(frame_id):
        return SimpleNamespace(header=SimpleNamespace(frame_id=frame_id))

    colour = "camera_color_optical_frame"
    assert_policy_camera_frames(message(colour), message(colour), message(colour), colour)
    with pytest.raises(RuntimeError, match="not registered.*depth='camera_depth"):
        assert_policy_camera_frames(
            message(colour), message("camera_depth_optical_frame"), message(colour), colour
        )


def test_policy_tool_transform_uses_the_kinematics_api():
    pytest.importorskip("franka_trajectory_replay.kinematics")
    tool = policy_tool_transform(
        {"tcp": {"offset_xyz": [0.01, -0.02, 0.11], "offset_rpy": [0.0, 0.0, 0.0]}}
    )
    assert tool.shape == (4, 4)
    assert tool[:3, 3] == pytest.approx([0.01, -0.02, 0.11])


def test_rgbd_synchronizer_ignores_staggered_latest_callbacks():
    frames = RgbdFrameSynchronizer(max_skew_s=0.04)
    frames.add("rgb", _stamped(10.0, "rgb-0"), 100.0)
    frames.add("depth", _stamped(10.0, "depth-0"), 100.01)
    assert tuple(message.value for message in frames.latest_pair()[:2]) == (
        "rgb-0",
        "depth-0",
    )

    # Reproduce the live failure: RGB arrives 134.4 ms ahead of the newest
    # depth callback. The last synchronized pair remains valid meanwhile.
    frames.add("rgb", _stamped(10.1344, "rgb-1"), 100.13)
    assert tuple(message.value for message in frames.latest_pair()[:2]) == (
        "rgb-0",
        "depth-0",
    )
    frames.add("depth", _stamped(10.1344, "depth-1"), 100.14)
    pair = frames.latest_pair()
    assert tuple(message.value for message in pair[:2]) == ("rgb-1", "depth-1")
    assert pair[2:] == pytest.approx((100.13, 100.14))


def test_rgbd_synchronizer_prefers_fresh_pair_over_older_exact_pair():
    frames = RgbdFrameSynchronizer(max_skew_s=0.04)
    frames.add("rgb", _stamped(10.0, "rgb-old"), 100.0)
    frames.add("depth", _stamped(10.0, "depth-old"), 100.0)
    frames.add("rgb", _stamped(10.20, "rgb-new"), 100.20)
    frames.add("depth", _stamped(10.21, "depth-new"), 100.21)

    pair = frames.latest_pair()
    assert tuple(message.value for message in pair[:2]) == ("rgb-new", "depth-new")
    assert pair[2:] == pytest.approx((100.20, 100.21))


def test_training_frame_round_trip():
    adapter = TrainingFrameAdapter()
    position = np.array([0.5, -0.1, 0.4])
    quaternion = fo.quat_from_euler_xyz(0.2, -0.3, 0.4)
    world_position, world_quaternion = adapter.pose_base_to_world(position, quaternion)
    round_position, round_quaternion = adapter.pose_world_to_base(
        world_position, world_quaternion
    )
    assert round_position == pytest.approx(position)
    assert abs(float(np.dot(round_quaternion, quaternion))) == pytest.approx(1.0)
    # The yaw the controller's clip frame is configured with is this adapter's.
    roll, pitch, yaw = fo.get_euler_xyz(adapter.world_from_base_quaternion)
    assert (roll, pitch) == pytest.approx((0.0, 0.0), abs=1e-12)
    assert yaw == pytest.approx(adapter.world_from_base_yaw)


def test_policy_goal_is_the_preclipped_bolt_anchored_target():
    """The runner sends `_apply_action` steps (0)-(1); the clip is the controller's."""

    adapter = TrainingFrameAdapter()
    zero = np.zeros(9)
    goal_position, goal_quaternion = policy_goal_base(adapter, zero)
    bolt_tip_base, _ = adapter.pose_world_to_base(
        fo.BOLT_TIP_POSITION, np.array([1.0, 0.0, 0.0, 0.0])
    )
    # Zero action: the bolt tip itself, facing down with the yaw window's midpoint.
    assert goal_position == pytest.approx(bolt_tip_base)
    expected = fo.decode_action_target(
        zero,
        fo.GraspFrameState(np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3), np.zeros((6, 7))),
        clip=False,
    )
    _, expected_quaternion = adapter.pose_world_to_base(expected.pos, expected.quat)
    assert abs(float(np.dot(goal_quaternion, expected_quaternion))) == pytest.approx(1.0)

    # A full position action moves the goal by the 50 mm bound, never clipped here,
    # and the goal does not depend on where the grasp frame happens to be.
    action = np.zeros(9)
    action[0:3] = [1.0, -1.0, 0.5]
    far_position, _ = policy_goal_base(adapter, action)
    expected_world = fo.BOLT_TIP_POSITION + np.array([1.0, -1.0, 0.5]) * fo.POS_ACTION_BOUNDS
    expected_base, _ = adapter.pose_world_to_base(expected_world, np.array([1.0, 0.0, 0.0, 0.0]))
    assert far_position == pytest.approx(expected_base)
    assert np.linalg.norm(far_position - bolt_tip_base) > 0.07


def test_grasp_in_flange_composes_back_to_the_grasp_pose():
    """tool_in_flange is the grasp frame on the flange: flange o tool == grasp."""

    flange_position = np.array([0.45, 0.05, 0.35])
    flange_quaternion = fo.quat_from_euler_xyz(2.9, 0.1, -0.7)
    tool_translation = np.array([-0.059, -0.029, 0.173])
    tool_quaternion = fo.quat_from_euler_xyz(0.2, -1.3, 0.4)
    grasp_position = flange_position + fo.quat_rotate(flange_quaternion, tool_translation)
    grasp_quaternion = fo.quat_mul(flange_quaternion, tool_quaternion)

    tool = grasp_in_flange(grasp_position, grasp_quaternion, flange_position, flange_quaternion)
    assert tool.shape == (7,)
    assert tool[:3] == pytest.approx(tool_translation)
    recovered = np.array([tool[6], tool[3], tool[4], tool[5]])  # xyzw -> wxyz
    assert abs(float(np.dot(recovered, tool_quaternion))) == pytest.approx(1.0)
    assert np.linalg.norm(recovered) == pytest.approx(1.0)

    # The same hand posture on a different arm pose gives the same tool.
    other_flange_position = np.array([0.2, -0.3, 0.6])
    other_flange_quaternion = fo.quat_from_euler_xyz(-1.0, 0.4, 2.0)
    moved = grasp_in_flange(
        other_flange_position + fo.quat_rotate(other_flange_quaternion, tool_translation),
        fo.quat_mul(other_flange_quaternion, tool_quaternion),
        other_flange_position,
        other_flange_quaternion,
    )
    assert moved[:3] == pytest.approx(tool[:3])
    assert abs(float(np.dot(moved[3:], tool[3:]))) == pytest.approx(1.0)

    with pytest.raises(ValueError):
        grasp_in_flange(grasp_position, np.zeros(4), flange_position, flange_quaternion)


# The coordinator's proxy is only correct in the geometry it actually runs in, so these use
# the real threading-grip reset frame (URDF_RESET_* below, z pointing world-down) rather than
# an identity quaternion. With identity the sign error that kept the 2026-09-30 ros-sim run in
# the `policy` phase for all 427 steps is invisible: both signs pass.
def _reset_grasp_quaternion():
    z_transport = grasp_z_transport_at_reset(
        URDF_RESET_THUMB, URDF_RESET_INDEX, URDF_RESET_FLANGE_POSITION, URDF_RESET_FLANGE_QUATERNION
    )
    _, quaternion = grasp_frame_from_tips(
        URDF_RESET_THUMB, URDF_RESET_INDEX, URDF_RESET_FLANGE_QUATERNION, z_transport
    )
    return quaternion


def _turned_about_world_z(quaternion, degrees):
    return fo.quat_mul(fo.quat_from_euler_xyz(0.0, 0.0, np.deg2rad(degrees)), quaternion)


def _coordinator(reset_quaternion, *, max_cycles=1):
    return GripCycleCoordinator(
        rate_hz=10.0,
        max_cycles=max_cycles,
        reset_position=np.zeros(3),
        reset_quaternion=reset_quaternion,
        reset_hand=np.zeros(3),
    )


def test_cycle_coordinator_enters_release_after_tightening_turn():
    # An M24 right-hand thread tightens clockwise seen from above, i.e. negative about world z,
    # and that is the direction training scores as positive progress.
    reset_q = _reset_grasp_quaternion()
    coordinator = _coordinator(reset_q)
    event, progress = coordinator.update(
        position=np.zeros(3), quaternion=_turned_about_world_z(reset_q, -56.0), hand=np.zeros(3)
    )
    # The grasp z is tilted ~7 degrees off the bolt axis, so the proxy reads ~1 percent low.
    assert progress == pytest.approx(np.deg2rad(56.0), abs=np.deg2rad(2.0))
    assert event == "release_started"
    assert coordinator.process_phase() == "follow_waypoints"


def test_cycle_coordinator_ignores_a_loosening_turn():
    reset_q = _reset_grasp_quaternion()
    coordinator = _coordinator(reset_q)
    event, progress = coordinator.update(
        position=np.zeros(3), quaternion=_turned_about_world_z(reset_q, +56.0), hand=np.zeros(3)
    )
    assert progress < 0.0
    assert event is None
    assert coordinator.process_phase() == "policy"


def test_cycle_coordinator_gates_on_the_measured_turn_when_it_has_one():
    """In ros-sim the simulated nut reports its own twist, and that wins.

    The proxy is still advanced and recorded -- it is the only signal hardware
    has, so the gap between the two is worth measuring -- but it must not decide
    the release when the real angle is available.
    """

    reset_q = _reset_grasp_quaternion()
    coordinator = _coordinator(reset_q)

    # The hand has not turned at all, so the proxy says nothing happened; the nut
    # has gone past the 55-degree gate.
    event, progress = coordinator.update(
        position=np.zeros(3),
        quaternion=reset_q,
        hand=np.zeros(3),
        measured_turn_progress=np.deg2rad(56.0),
    )
    assert event == "release_started"
    assert progress == pytest.approx(np.deg2rad(56.0))
    assert coordinator.gated_on_measured_turn
    assert coordinator.last_proxy_turn_rad == pytest.approx(0.0, abs=1e-9)


def test_cycle_coordinator_ignores_a_proxy_spike_when_the_nut_has_not_turned():
    """The degenerate-fingertip artefact must not fire the release in ros-sim.

    A closing pinch used to collapse the fingertips to a few millimetres, where
    the thumb-to-index direction is meaningless and the proxy jumped ~100 degrees
    in one step -- which fired the release in two runs out of three. Gating on the
    nut's own twist makes that unreachable.
    """

    reset_q = _reset_grasp_quaternion()
    coordinator = _coordinator(reset_q)
    event, progress = coordinator.update(
        position=np.zeros(3),
        quaternion=_turned_about_world_z(reset_q, -100.0),
        hand=np.zeros(3),
        measured_turn_progress=0.0,
    )
    assert event is None
    assert progress == pytest.approx(0.0)
    assert coordinator.last_proxy_turn_rad > np.deg2rad(90.0)


def test_cycle_completion_rebases_without_retriggering_release():
    reset_q = _reset_grasp_quaternion()
    coordinator = _coordinator(reset_q, max_cycles=2)
    coordinator.active = True
    coordinator.phase_index = 4
    coordinator.phase_steps = 9
    coordinator._unwrapped_yaw = np.deg2rad(60.0)
    coordinator._previous_yaw = np.deg2rad(60.0)

    event, _ = coordinator.update(
        position=np.zeros(3), quaternion=reset_q, hand=np.zeros(3)
    )

    assert event == "cycle_completed"
    assert coordinator.completed_cycles == 1
    assert not coordinator.active


# fr3_link0 poses from inspire_franka.urdf.xacro (hand_mount:=flange) at the
# M24 reset joints and the training grasp posture.
URDF_RESET_THUMB = np.array([0.55681, 0.00573, 0.26632])
URDF_RESET_INDEX = np.array([0.67136, 0.00731, 0.25793])
URDF_RESET_FLANGE_POSITION = np.array([0.53776, -0.16679, 0.29261])
URDF_RESET_FLANGE_QUATERNION = np.array([0.535314, -0.756752, 0.240942, -0.287599])


def test_reset_grasp_frame_approaches_downward_like_training():
    z_transport = grasp_z_transport_at_reset(
        URDF_RESET_THUMB, URDF_RESET_INDEX, URDF_RESET_FLANGE_POSITION, URDF_RESET_FLANGE_QUATERNION
    )
    position, quaternion = grasp_frame_from_tips(
        URDF_RESET_THUMB, URDF_RESET_INDEX, URDF_RESET_FLANGE_QUATERNION, z_transport
    )
    z_axis = fo.matrix_from_quat(quaternion)[:, 2]
    np.testing.assert_allclose(position, 0.5 * (URDF_RESET_THUMB + URDF_RESET_INDEX))
    # Training's reset grasp Z is world-down, orthogonalised against thumb->index.
    assert np.degrees(np.arccos(-z_axis[2])) < 10.0
    # The flange's own -Z is not that axis on this hand mount; using it put the
    # 2026-09-12 hardware grasp frame ~107 degrees off the training frame.
    flange_minus_z = fo.quat_rotate(URDF_RESET_FLANGE_QUATERNION, np.array([0.0, 0.0, -1.0]))
    _, wrong = fo.hand_grasp_frame(URDF_RESET_THUMB, URDF_RESET_INDEX, flange_minus_z)
    angle = 2.0 * np.degrees(np.arccos(min(1.0, abs(np.dot(fo.quat_from_matrix(wrong), quaternion)))))
    assert angle > 90.0


def test_grasp_frame_transport_follows_flange_rotation():
    z_transport = grasp_z_transport_at_reset(
        URDF_RESET_THUMB, URDF_RESET_INDEX, URDF_RESET_FLANGE_POSITION, URDF_RESET_FLANGE_QUATERNION
    )
    _, reset_quaternion = grasp_frame_from_tips(
        URDF_RESET_THUMB, URDF_RESET_INDEX, URDF_RESET_FLANGE_QUATERNION, z_transport
    )
    turn = fo.quat_from_euler_xyz(0.0, 0.0, np.deg2rad(30.0))
    rotate = lambda p: fo.quat_rotate(turn, p)  # noqa: E731
    _, turned_quaternion = grasp_frame_from_tips(
        rotate(URDF_RESET_THUMB),
        rotate(URDF_RESET_INDEX),
        fo.quat_mul(turn, URDF_RESET_FLANGE_QUATERNION),
        z_transport,
    )
    np.testing.assert_allclose(
        fo.matrix_from_quat(turned_quaternion),
        fo.matrix_from_quat(turn) @ fo.matrix_from_quat(reset_quaternion),
        atol=1e-6,
    )


# The student was distilled on ForgeUltra's hand geometry, not the workspace
# URDF's. These pin the frame the rollout now commands in.
TRAINING_GRASP_MIDPOINT_IN_FLANGE = np.array([-0.059067, -0.028773, 0.173311])
TRAINING_TIP_SEPARATION_M = 0.0578


def _training_hand():
    mujoco = pytest.importorskip("mujoco")
    del mujoco
    from policy_rollout.hardware import TrainingHandKinematics

    return TrainingHandKinematics()


def test_training_hand_reproduces_the_distilled_grasp_midpoint():
    hand = _training_hand()
    posture = [fo.THREADING_GRASP_POSTURE[name] for name in fo.PINCH_JOINTS]
    thumb, index = hand.tips_in_flange(posture)
    np.testing.assert_allclose(
        0.5 * (thumb + index), TRAINING_GRASP_MIDPOINT_IN_FLANGE, atol=1e-5
    )
    assert np.linalg.norm(index - thumb) == pytest.approx(
        TRAINING_TIP_SEPARATION_M, abs=1e-4
    )


def test_training_hand_tips_do_not_depend_on_the_arm():
    """The hand is rigid on the flange, so the rollout may ignore arm joints."""

    mujoco = pytest.importorskip("mujoco")
    from policy_rollout.mujoco_scene import load_training_scene

    hand = _training_hand()
    model, data = load_training_scene(urdf_tip_frames=True)
    address = {model.joint(i).name: model.jnt_qposadr[i] for i in range(model.njnt)}
    inactive = {"middle_joint_0": 1.333, "ring_joint_0": 1.333, "little_joint_0": 1.333}
    generator = np.random.default_rng(3)
    for _ in range(4):
        arm = fo.FRANKA_ARM_RESET_JOINTS_M24 + generator.normal(0.0, 0.3, 7)
        posture = [generator.uniform(low, high) for low, high in fo.PINCH_RANGES]
        data.qpos[:] = 0.0
        for name, value in zip(
            (f"fr3_joint{i}" for i in range(1, 8)), arm, strict=True
        ):
            data.qpos[address[name]] = float(value)
        targets = dict(zip(fo.PINCH_JOINTS, posture, strict=True))
        targets.update(inactive)
        for name, value in fo.expand_hand_mimic(targets).items():
            if name in address:
                data.qpos[address[name]] = float(value)
        mujoco.mj_forward(model, data)
        flange = model.body("fr3_link8").id
        rotation = data.xmat[flange].reshape(3, 3)
        origin = data.xpos[flange]
        expected_thumb = rotation.T @ (data.xpos[model.body("thumb_tip").id] - origin)
        expected_index = rotation.T @ (data.xpos[model.body("index_tip").id] - origin)
        thumb, index = hand.tips_in_flange(posture)
        np.testing.assert_allclose(thumb, expected_thumb, atol=1e-9)
        np.testing.assert_allclose(index, expected_index, atol=1e-9)


def test_training_hand_differs_from_the_workspace_urdf():
    """Guard the reason this class exists: the two descriptions disagree."""

    hand = _training_hand()
    posture = [fo.THREADING_GRASP_POSTURE[name] for name in fo.PINCH_JOINTS]
    thumb, index = hand.tips_in_flange(posture)
    # Measured from inspire_franka.urdf.xacro with the thumb-yaw overlay.
    urdf_midpoint = np.array([-0.0638, -0.0422, 0.1773])
    offset = np.linalg.norm(0.5 * (thumb + index) - urdf_midpoint)
    assert 0.010 < offset < 0.020, f"unexpected URDF-to-training offset {offset:.4f} m"


def test_policy_tool_offset_is_the_training_grasp_midpoint():
    """The controller's compliance centre must be where training applied its wrench."""

    from policy_rollout.hardware import POLICY_TOOL_OFFSET_XYZ

    hand = _training_hand()
    posture = [fo.THREADING_GRASP_POSTURE[name] for name in fo.PINCH_JOINTS]
    thumb, index = hand.tips_in_flange(posture)
    np.testing.assert_allclose(
        np.asarray(POLICY_TOOL_OFFSET_XYZ), 0.5 * (thumb + index), atol=1e-6
    )


def test_policy_tool_offset_matches_the_controller_profiles():
    """hardware.py, controllers_policy.yaml and controllers_sim_policy.yaml agree."""

    yaml = pytest.importorskip("yaml")
    from pathlib import Path

    from policy_rollout.hardware import (
        POLICY_TOOL_OFFSET_RPY,
        POLICY_TOOL_OFFSET_XYZ,
    )

    config_root = (
        Path(__file__).resolve().parents[3]
        / "src/inspire_franka_trajectory_replay/config"
    )

    def find_parameters(node):
        """ros2 profiles nest controllers under a node name or a '/**' wildcard."""
        if isinstance(node, dict):
            controller = node.get("cartesian_trajectory_replay_controller")
            if isinstance(controller, dict) and "ros__parameters" in controller:
                return controller["ros__parameters"]
            for value in node.values():
                found = find_parameters(value)
                if found is not None:
                    return found
        return None

    for name in ("controllers_policy.yaml", "controllers_sim_policy.yaml"):
        document = yaml.safe_load((config_root / name).read_text())
        parameters = find_parameters(document)
        assert parameters is not None, f"no Cartesian controller block in {name}"
        np.testing.assert_allclose(
            parameters["tool_offset_xyz"], POLICY_TOOL_OFFSET_XYZ, atol=1e-9
        )
        np.testing.assert_allclose(
            parameters["tool_offset_rpy"], POLICY_TOOL_OFFSET_RPY, atol=1e-9
        )


def test_policy_profiles_run_forge_s_law_and_decode():
    """Both profiles carry compute_dof_torque's structure and _apply_action's clip.

    These are the values `run_hardware_rollout` refuses to run without; pinning
    them here means a profile edit fails a unit test before it fails on the bench.
    """

    yaml = pytest.importorskip("yaml")
    from pathlib import Path

    config_root = (
        Path(__file__).resolve().parents[3]
        / "src/inspire_franka_trajectory_replay/config"
    )

    def find_parameters(node):
        if isinstance(node, dict):
            controller = node.get("cartesian_trajectory_replay_controller")
            if isinstance(controller, dict) and "ros__parameters" in controller:
                return controller["ros__parameters"]
            for value in node.values():
                found = find_parameters(value)
                if found is not None:
                    return found
        return None

    adapter = TrainingFrameAdapter()
    for name in ("controllers_policy.yaml", "controllers_sim_policy.yaml"):
        parameters = find_parameters(yaml.safe_load((config_root / name).read_text()))
        assert parameters is not None, name
        assert parameters["translational_stiffness"] == pytest.approx(fo.DEFAULT_TASK_PROP_GAINS[0])
        assert parameters["rotational_stiffness"] == pytest.approx(fo.DEFAULT_TASK_PROP_GAINS[3])
        assert parameters["nullspace_stiffness"] == pytest.approx(fo.KP_NULL)
        assert parameters["rotation_error"] == "axis_angle", name
        assert parameters["nullspace_damping_lambda"] == 0.0, name
        assert parameters["mass_weighted_nullspace"] is True, name
        # compute_dof_torque has no Coriolis term and clamps at +-100 Nm.
        assert parameters["coriolis_compensation"] is False, name
        assert parameters["torque_limit"] == pytest.approx(fo.ARM_TORQUE_LIMIT), name
        # _apply_action's per-substep clip, now the controller's per-cycle clip, in
        # the training world frame.
        assert parameters["policy_clip_position_m"] == pytest.approx(fo.POS_ACTION_THRESHOLD[0]), name
        assert parameters["policy_clip_orientation_rad"] == pytest.approx(fo.ROT_ACTION_THRESHOLD[0]), name
        assert parameters["policy_clip_frame_yaw"] == pytest.approx(adapter.world_from_base_yaw), name
        # The setpoint is clipped in the controller; no second filter on it.
        assert parameters["target_filter"] == 1.0, name


def test_hand_joint_state_to_arrays_orders_the_driven_joints():
    from policy_rollout.hardware import HAND_JOINTS, hand_joint_state_to_arrays

    shuffled = list(reversed(HAND_JOINTS))
    message = SimpleNamespace(
        name=list(shuffled),
        position=[0.1 * (i + 1) for i in range(len(shuffled))],
        velocity=[0.01 * (i + 1) for i in range(len(shuffled))],
    )
    positions, velocities = hand_joint_state_to_arrays(message)
    expected = {n: 0.1 * (i + 1) for i, n in enumerate(shuffled)}
    np.testing.assert_allclose(positions, [expected[n] for n in HAND_JOINTS])
    assert velocities.shape == (len(HAND_JOINTS),)


def test_hand_joint_state_to_arrays_rejects_a_partial_hand():
    from policy_rollout.hardware import HAND_JOINTS, hand_joint_state_to_arrays

    message = SimpleNamespace(
        name=list(HAND_JOINTS[:-1]),
        position=[0.0] * (len(HAND_JOINTS) - 1),
        velocity=[0.0] * (len(HAND_JOINTS) - 1),
    )
    with pytest.raises(RuntimeError, match="all driven RH56 joints"):
        hand_joint_state_to_arrays(message)
