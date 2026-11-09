from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from policy_rollout import forge_osc as fo
from policy_rollout.hardware import (
    TorqueLowPass,
    arm_torque_candidates,
    arm_torque_candidates_from_controller,
    external_joint_torque_message_to_array,
    fingertip_force_message_to_newtons,
    resolve_arm_torque_source,
    TRAINING_INACTIVE_FINGERS,
    GripCycleCoordinator,
    joint_limit_violation,
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
    physical_velocities[-1] = logical_dq[0]
    flexion_dof = kin.dof_index(command_overlays.THUMB_FLEXION_JOINT)
    physical_positions[4] = kin.open_ratio_to_rad(
        flexion_dof,
        command_overlays.apply_open_ratio_overlay(
            flexion_dof, kin.rad_to_open_ratio(flexion_dof, logical_q[1])
        ),
    )
    physical_velocities[4] = logical_dq[1] / command_overlays.model_rate_gain(flexion_dof)

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


def test_cycle_coordinator_refuses_the_proxy_on_a_collapsed_pinch():
    """The 2026-10-16 hardware release: fingertips 7.9 mm apart, proxy +74 deg.

    The index was at 0.66 rad, so "index closed" would not have caught it; the
    pinch width and the proxy's jump rate do. The guard only applies while the
    proxy is what gates the release.
    """

    reset_q = _reset_grasp_quaternion()
    coordinator = GripCycleCoordinator(
        rate_hz=15.0,
        max_cycles=1,
        reset_position=np.zeros(3),
        reset_quaternion=reset_q,
        reset_hand=np.zeros(3),
        min_fingertip_separation_m=0.030,
        max_proxy_step_rad=np.deg2rad(20.0),
    )
    turned = _turned_about_world_z(reset_q, -74.0)
    event, progress = coordinator.update(
        position=np.zeros(3), quaternion=turned, hand=np.zeros(3), fingertip_separation_m=0.0079
    )
    assert event is None
    assert progress > np.deg2rad(55.0)
    assert coordinator.proxy_vetoes == 1
    assert "7.9 mm" in coordinator.last_veto
    # A wide-open pinch but the proxy jumped 74 deg in one tick: still refused.
    coordinator = GripCycleCoordinator(
        rate_hz=15.0,
        max_cycles=1,
        reset_position=np.zeros(3),
        reset_quaternion=reset_q,
        reset_hand=np.zeros(3),
        max_proxy_step_rad=np.deg2rad(20.0),
    )
    event, _ = coordinator.update(
        position=np.zeros(3), quaternion=turned, hand=np.zeros(3), fingertip_separation_m=0.055
    )
    assert event is None
    assert "in one tick" in coordinator.last_veto
    # The measured nut twist is not subject to the proxy guards.
    event, _ = coordinator.update(
        position=np.zeros(3),
        quaternion=turned,
        hand=np.zeros(3),
        measured_turn_progress=np.deg2rad(56.0),
        fingertip_separation_m=0.0079,
    )
    assert event == "release_started"


def test_cycle_coordinator_accepts_a_gradual_turn_with_a_full_pinch():
    reset_q = _reset_grasp_quaternion()
    coordinator = GripCycleCoordinator(
        rate_hz=15.0,
        max_cycles=1,
        reset_position=np.zeros(3),
        reset_quaternion=reset_q,
        reset_hand=np.zeros(3),
        min_fingertip_separation_m=0.030,
        max_proxy_step_rad=np.deg2rad(20.0),
    )
    event = None
    for degrees in range(-4, -60, -4):
        event, _ = coordinator.update(
            position=np.zeros(3),
            quaternion=_turned_about_world_z(reset_q, degrees),
            hand=np.zeros(3),
            fingertip_separation_m=0.040,
        )
        if event is not None:
            break
    assert event == "release_started"
    assert coordinator.proxy_vetoes == 0
    errors = coordinator.return_errors(np.array([0.004, 0.0, 0.0]), reset_q, np.zeros(3))
    assert errors[0] == pytest.approx(0.004)
    assert errors[1] == pytest.approx(0.0, abs=1e-6)
    assert errors[2] == pytest.approx(0.0)


# (fingertip separation m, [thumb, index] pad force N, index rad) per 15 Hz tick,
# from ros_sim-20261103-130005-909378 cycle 0 (ticks 22-39: the nut turned
# -15 -> +58 deg) and hardware-20261103-143701-495628 (ticks 28-39, the fingers
# stalled on the nut; ticks 212-226, they slipped past it unloaded).
SIM_FINGER_SWEEP_TURN = [
    (0.0334, [2.9, 4.5], 0.38), (0.0294, [1.1, 0.0], 0.41), (0.0254, [4.6, 4.1], 0.45),
    (0.0210, [4.6, 3.5], 0.49), (0.0149, [4.6, 3.0], 0.54), (0.0102, [2.1, 0.0], 0.59),
    (0.0061, [1.4, 0.0], 0.62), (0.0025, [4.6, 5.5], 0.67), (0.0026, [8.2, 11.9], 0.69),
    (0.0037, [7.3, 10.9], 0.70), (0.0058, [6.5, 11.9], 0.73), (0.0076, [7.1, 12.6], 0.75),
    (0.0093, [8.2, 12.8], 0.76), (0.0114, [8.5, 12.7], 0.78), (0.0127, [8.2, 12.3], 0.80),
    (0.0139, [8.3, 10.9], 0.81), (0.0153, [8.1, 8.7], 0.82), (0.0162, [6.0, 8.2], 0.83),
]
HARDWARE_STALLED_GRIP = [(0.0190, [7.5 + 0.1 * k, 7.5 + 0.2 * k], 0.49) for k in range(12)]
HARDWARE_SLIP = [
    (0.0162, [1.1, 0.4], 0.53), (0.0148, [1.2, 1.4], 0.55), (0.0138, [1.9, 3.0], 0.55),
    (0.0108, [1.9, 3.0], 0.59), (0.0089, [1.3, 1.4], 0.61), (0.0057, [1.3, 1.4], 0.65),
    (0.0053, [0.7, 0.7], 0.66), (0.0050, [0.3, 0.3], 0.67), (0.0055, [0.3, 0.3], 0.71),
    (0.0060, [0.0, 0.0], 0.72), (0.0069, [-0.1, 0.0], 0.72), (0.0084, [-0.2, 0.0], 0.84),
    (0.0230, [0.0, -0.3], 0.85),
]


def _sweep_coordinator(reset_q):
    return GripCycleCoordinator(
        rate_hz=15.0,
        max_cycles=2,
        reset_position=np.zeros(3),
        reset_quaternion=reset_q,
        reset_hand=np.zeros(3),
        min_fingertip_separation_m=0.030,
        max_proxy_step_rad=np.deg2rad(20.0),
        release_signal="finger_sweep",
    )


def _feed(coordinator, ticks, quaternion, **kwargs):
    events = []
    for separation, force, index in ticks:
        event, _ = coordinator.update(
            position=np.zeros(3),
            quaternion=quaternion,
            hand=np.zeros(3),
            fingertip_separation_m=separation,
            fingertip_force_n=np.array(force),
            index_rad=index,
            **kwargs,
        )
        events.append(event)
    return events


def test_finger_sweep_releases_at_the_end_of_a_simulated_turn():
    """The student turns the nut with its fingers, which the yaw proxy cannot see.

    Its tips end that turn ~16 mm apart, inside the proxy's collapsed-pinch veto,
    so on hardware the 2026-11-03 run could never release. The sweep fires on
    the tick the index reaches 0.82 rad (nut twist 46 deg in sim) and not before.
    """

    reset_q = _reset_grasp_quaternion()
    coordinator = _sweep_coordinator(reset_q)
    events = _feed(coordinator, SIM_FINGER_SWEEP_TURN, reset_q)
    assert events.index("release_started") == 16
    assert coordinator.gate_source == "finger_sweep"
    assert coordinator.proxy_vetoes == 0


def test_finger_sweep_ignores_a_stalled_grip_and_an_unloaded_slip():
    reset_q = _reset_grasp_quaternion()
    coordinator = _sweep_coordinator(reset_q)
    # A degenerate grasp frame reads a large proxy turn; the sweep ignores it.
    turned = _turned_about_world_z(reset_q, -100.0)
    events = _feed(coordinator, HARDWARE_STALLED_GRIP + HARDWARE_SLIP, turned)
    assert set(events) == {None}
    assert coordinator.process_phase() == "policy"


def test_finger_sweep_needs_the_pinch_and_the_close_in_one_grasp():
    reset_q = _reset_grasp_quaternion()
    coordinator = _sweep_coordinator(reset_q)
    pinch = SIM_FINGER_SWEEP_TURN[:9]
    opened = [(0.060, [0.0, 0.0], 0.20)]
    closed_loaded = [(0.016, [6.0, 8.2], 0.83)]
    assert set(_feed(coordinator, pinch + opened + closed_loaded, reset_q)) == {None}
    assert _feed(coordinator, pinch + closed_loaded, reset_q)[-1] == "release_started"


def test_finger_sweep_fires_when_the_contact_leaves_the_pads():
    """Under 2x arm friction the index pad read 0 N at the end of a 61 deg turn."""

    reset_q = _reset_grasp_quaternion()
    coordinator = _sweep_coordinator(reset_q)
    pinch = SIM_FINGER_SWEEP_TURN[:9]
    closed_unloaded = [(0.018, [6.3, 0.0], 0.83)]
    assert _feed(coordinator, pinch + closed_unloaded, reset_q)[-1] == "release_started"


def test_finger_sweep_accepts_the_rh56s_widest_loaded_pinch():
    """hardware-20261103-161413-264559: the real tips closed to 7.8 mm, not 2-3."""

    reset_q = _reset_grasp_quaternion()
    coordinator = _sweep_coordinator(reset_q)
    ticks = [
        (0.0220, [4.3, 10.2], 0.46), (0.0134, [0.6, 7.0], 0.57), (0.0099, [3.0, 8.0], 0.65),
        (0.0079, [6.9, 8.3], 0.72), (0.0079, [11.5, 10.7], 0.72), (0.0119, [5.1, 7.8], 0.74),
        (0.0165, [4.5, 7.3], 0.79), (0.0185, [4.7, 7.3], 0.82),
    ]
    events = _feed(coordinator, ticks, reset_q)
    assert events.index("release_started") == len(ticks) - 1


def test_measured_twist_still_wins_over_the_finger_sweep():
    reset_q = _reset_grasp_quaternion()
    coordinator = _sweep_coordinator(reset_q)
    events = _feed(coordinator, SIM_FINGER_SWEEP_TURN, reset_q, measured_turn_progress=0.0)
    assert set(events) == {None}
    assert coordinator.gate_source == "nut_twist"


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


@pytest.mark.parametrize(
    "total_deg, target_deg",
    [
        (0.0, 60.0),     # a turn exactly on the grid: the next is a full step
        (56.0, 64.0),    # four short is made up by the next turn
        (63.0, 57.0),    # an overshoot shortens it
        (50.0, 65.0),    # clamped to the trained 55-65 deg
        (128.0, 55.0),
        (178.0, 62.0),   # the residual is off the nearest grid point, not cycle 1
    ],
)
def test_the_next_turn_aims_at_the_60_degree_grid(total_deg, target_deg):
    target = fo.next_cycle_target_turn_rad(np.deg2rad(total_deg))
    assert np.rad2deg(target) == pytest.approx(target_deg)


def _complete_cycle(coordinator, reset_q, **kwargs):
    coordinator.active = True
    coordinator.phase_index = 4
    coordinator.phase_steps = 9
    return coordinator.update(
        position=np.zeros(3), quaternion=reset_q, hand=np.zeros(3), **kwargs
    )


def test_the_release_turn_follows_the_measured_total_turn():
    """ros-sim measures the nut's total turn, so cycle 2 makes up cycle 1's shortfall."""

    reset_q = _reset_grasp_quaternion()
    coordinator = _coordinator(reset_q, max_cycles=3)
    assert np.rad2deg(coordinator.release_turn_rad) == pytest.approx(55.0)

    event, _ = _complete_cycle(
        coordinator, reset_q, measured_turn_progress=0.0, measured_total_turn=np.deg2rad(56.0)
    )
    assert event == "cycle_completed"
    assert np.rad2deg(coordinator.release_turn_rad) == pytest.approx(59.0)
    event, _ = coordinator.update(
        position=np.zeros(3), quaternion=reset_q, hand=np.zeros(3),
        measured_turn_progress=np.deg2rad(57.0),
    )
    assert event is None
    event, _ = coordinator.update(
        position=np.zeros(3), quaternion=reset_q, hand=np.zeros(3),
        measured_turn_progress=np.deg2rad(59.5),
    )
    assert event == "release_started"


def test_without_a_measured_total_every_cycle_releases_at_55_degrees():
    """The proxy cannot see the nut's total turn, so hardware keeps the 60 deg target."""

    reset_q = _reset_grasp_quaternion()
    coordinator = _coordinator(reset_q, max_cycles=3)
    event, _ = _complete_cycle(coordinator, reset_q)
    assert event == "cycle_completed"
    assert np.rad2deg(coordinator.release_turn_rad) == pytest.approx(55.0)


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


def test_trained_reset_grasp_is_the_training_reset_frame_in_the_base():
    """The return gate's reference: the grasp frame of the reset joints and entry pinch.

    Checked against the same quantity measured the way the rollout measures it
    live (tips in the flange from the pinch posture, flange from the arm), and
    against the training world's bolt geometry: the student starts 0.26 m above
    the table, x 0.58 in the world, just behind the bolt at x 0.61.
    """

    from policy_rollout.hardware import TrainingFrameAdapter, trained_reset_grasp

    hand = _training_hand()
    frame = TrainingFrameAdapter()
    position, quaternion, pinch = trained_reset_grasp(hand, frame)
    np.testing.assert_allclose(pinch, fo.ISAAC_ENTRY_PINCH)
    assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    world_position, world_quaternion = frame.pose_base_to_world(position, quaternion)
    # The entry grasp sits (26.0, -0.4, 165.4) mm from the bolt tip in the base
    # frame, i.e. world (0.584, 0.000, 0.268); the closed posture's frame was
    # (0.5829, 0.0017, 0.2641).
    np.testing.assert_allclose(world_position, [0.584, 0.0004, 0.2677], atol=1e-3)
    # World-down orthogonalised against thumb->index: the entry pinch's more
    # open thumb tilts that line, so the approach axis leans 13.3 deg (the
    # closed posture's leaned under 10). The teacher's decoded return command
    # agrees with this frame to 0.8 deg.
    z_axis = fo.matrix_from_quat(world_quaternion)[:, 2]
    assert np.degrees(np.arccos(-z_axis[2])) == pytest.approx(13.3, abs=0.5)
    # Live measurement path: flange pose + tips in the flange.
    flange_position, flange_quaternion, thumb, index = hand.frames_in_world(
        fo.FRANKA_ARM_RESET_JOINTS_M24, pinch
    )
    thumb_flange, index_flange = hand.tips_in_flange(pinch)
    rotation = fo.matrix_from_quat(flange_quaternion)
    np.testing.assert_allclose(flange_position + rotation @ thumb_flange, thumb, atol=1e-9)
    np.testing.assert_allclose(flange_position + rotation @ index_flange, index, atol=1e-9)
    np.testing.assert_allclose(world_position, 0.5 * (thumb + index), atol=1e-9)


def test_the_teacher_s_return_command_lands_on_the_entry_grasp():
    """Isaac's teacher returns to its live entry pose; decoded here it must land on ours.

    The unified command is the teacher's at the end of all ten returns of the
    2026-10-25 force-obs training episode. Through this workspace's bolt frame
    and decoder it has to reproduce the entry grasp, hand included.
    """

    from policy_rollout.hardware import TrainingFrameAdapter, policy_goal_base, trained_reset_grasp

    frame = TrainingFrameAdapter()
    entry_position, entry_quaternion, entry_pinch = trained_reset_grasp(_training_hand(), frame)
    scale = np.array([3.0, 4.0, 16.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0])
    teacher = np.array([-0.175, 0.003, 0.207, 0.0, -0.077, 0.538, -0.091, -0.083, 0.0]) * scale
    goal_position, goal_quaternion = policy_goal_base(frame, teacher)
    assert np.linalg.norm(goal_position - entry_position) < 0.001
    alignment = min(1.0, abs(float(np.dot(goal_quaternion, entry_quaternion))))
    assert np.degrees(2.0 * np.arccos(alignment)) < 1.5
    np.testing.assert_allclose(fo.pinch_targets(teacher), entry_pinch, atol=2e-4)


def test_the_previous_action_seed_encodes_the_entry_grasp_like_isaac_s_reset():
    """Isaac's reset (and each cycle boundary) seeds prev_actions from the live pose.

    Position is the grasp relative to the bolt tip over the 5 cm action bounds,
    unclipped, so the entry grasp 165 mm above the tip seeds native z ~3.3;
    the pinch seeds are 0 under the relative-to-grasp finger map.
    """

    pytest.importorskip("mujoco")
    from policy_rollout.hardware import TrainingFrameAdapter, trained_reset_grasp
    from policy_rollout.mujoco_threading_env import seed_action_history

    frame = TrainingFrameAdapter()
    position, quaternion, _ = trained_reset_grasp(_training_hand(), frame)
    world_position, world_quaternion = frame.pose_base_to_world(position, quaternion)
    seed = seed_action_history(
        fo.GraspFrameState(world_position, world_quaternion, np.zeros(3), np.zeros(3), np.zeros((6, 7)))
    )
    np.testing.assert_allclose(
        seed[0:3], (world_position - fo.BOLT_TIP_POSITION) / fo.POS_ACTION_BOUNDS, atol=1e-9
    )
    assert seed[2] == pytest.approx(3.31, abs=0.02)
    np.testing.assert_allclose(seed[6:9], 0.0)
    # The teacher's own return command (unified x scale) sits at the same
    # entry grasp, so the seed must agree with it in position and yaw.
    scale = np.array([3.0, 4.0, 16.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0])
    teacher = np.array([-0.175, 0.003, 0.207, 0.0, -0.077, 0.538, -0.091, -0.083, 0.0]) * scale
    np.testing.assert_allclose(seed[0:3], teacher[0:3], atol=0.03)
    assert seed[5] == pytest.approx(teacher[5], abs=0.03)


def test_physical_hand_feedback_passes_through_without_the_overlay():
    """ros-sim runs the driver without the thumb-yaw overlay: nothing to invert."""

    from policy_rollout.hardware import HAND_JOINTS, physical_hand_state_to_policy

    positions = np.zeros(len(HAND_JOINTS))
    velocities = np.zeros(len(HAND_JOINTS))
    columns = {name: index for index, name in enumerate(HAND_JOINTS)}
    positions[columns["thumb_proximal_yaw_joint"]] = 1.185
    velocities[columns["thumb_proximal_yaw_joint"]] = 0.3
    positions[columns["index_proximal_joint"]] = 0.2147
    q, dq = physical_hand_state_to_policy(positions, velocities, thumb_overlay=False)
    np.testing.assert_allclose(q, [1.185, 0.0, 0.2147])
    np.testing.assert_allclose(dq, [0.3, 0.0, 0.0])
    # With the calibration on, the same logical 1.185 rad is register yaw 0.915.
    positions[columns["thumb_proximal_yaw_joint"]] = 1.185 - 0.27
    q_on, dq_on = physical_hand_state_to_policy(positions, velocities, thumb_overlay=True)
    assert q_on[0] == pytest.approx(1.185, abs=1e-6)
    assert dq_on[0] == pytest.approx(0.3)


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


def test_the_workspace_hand_is_the_training_hand():
    """The description TF and the simulator use must put the tips where training did.

    inspire_hand_description is generated from forgeUltra's URDF, and the
    simulator's flange model from that description, so the fingertips the
    plant and RViz show are the policy's own. Checked through the generated
    MJCF, which needs no xacro.
    """

    mujoco = pytest.importorskip("mujoco")
    hand = _training_hand()
    path = (
        Path(__file__).resolve().parents[3]
        / "src" / "inspire_franka_sim" / "mjcf" / "inspire_hand_on_flange.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    names = {
        "thumb_joint_0": "thumb_proximal_yaw_joint",
        "thumb_joint_1": "thumb_proximal_pitch_joint",
        "thumb_joint_2": "thumb_intermediate_joint",
        "thumb_joint_3": "thumb_distal_joint",
        "index_joint_0": "index_proximal_joint",
        "index_joint_1": "index_intermediate_joint",
    }
    grasp = [fo.THREADING_GRASP_POSTURE[name] for name in fo.PINCH_JOINTS]
    # Thumb pitch stays below 0.31 rad, where the intermediate's limit starts
    # to clamp it and the policy's unclamped mimic expansion stops applying.
    for posture in (grasp, [0.0, 0.0, 0.0], [1.2, 0.3, 1.3], [0.6, 0.15, 0.7]):
        data.qpos[:] = 0.0
        targets = dict(zip(fo.PINCH_JOINTS, posture, strict=True))
        targets.update(TRAINING_INACTIVE_FINGERS)
        expanded = fo.expand_hand_mimic(targets)
        for name, value in expanded.items():
            if name in names:
                data.qpos[model.jnt_qposadr[model.joint(names[name]).id]] = value
        mujoco.mj_kinematics(model, data)
        flange = model.body("fr3_link8").id
        rotation = data.xmat[flange].reshape(3, 3)
        origin = data.xpos[flange]
        thumb, index = hand.tips_in_flange(posture)
        for tip, expected in (("thumb_tip", thumb), ("index_tip", index)):
            measured = rotation.T @ (data.xpos[model.body(tip).id] - origin)
            np.testing.assert_allclose(measured, expected, atol=1e-6)


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


def test_turn_monitor_values_are_degrees_in_field_order_and_nan_without_a_nut():
    import math

    from policy_rollout.hardware import TURN_MONITOR_FIELDS, turn_monitor_values

    sim = turn_monitor_values(
        nut_turn_rad=math.radians(30.0),
        proxy_turn_rad=math.radians(45.0),
        gate_turn_rad=math.radians(30.0),
        fingertip_separation_m=0.0412,
        index_rad=0.6,
        phase="follow_waypoints",
    )
    assert len(sim) == len(TURN_MONITOR_FIELDS)
    np.testing.assert_allclose(sim, [30.0, 45.0, 30.0, 41.2, 0.6, 1.0])

    # Hardware has no nut-angle topic and may not have a tip separation yet.
    hardware = turn_monitor_values(
        nut_turn_rad=None,
        proxy_turn_rad=0.0,
        gate_turn_rad=0.0,
        fingertip_separation_m=None,
        index_rad=0.0,
        phase="policy",
    )
    assert math.isnan(hardware[0]) and math.isnan(hardware[3])
    assert hardware[5] == 0.0


def test_fingertip_force_reads_the_sim_channels_in_newtons():
    message = SimpleNamespace(
        name=["thumb_flexion_extension_force", "index_flexion_extension_force"],
        effort=[1.5, 0.25],
    )
    force, raw = fingertip_force_message_to_newtons(message)
    assert force.tolist() == pytest.approx([1.5, 0.25])
    assert not raw


def test_fingertip_force_reads_the_rh56_thumb_bend_and_index_in_grams():
    # Driver order: pinky, ring, middle, index, thumb_bend, thumb_rotation.
    message = SimpleNamespace(
        name=["1", "2", "3", "4", "5", "6"],
        effort=[0.0, 0.0, 0.0, 200.0, 1000.0, 50.0],
    )
    force, raw = fingertip_force_message_to_newtons(message)
    assert force.tolist() == pytest.approx([9.80665, 1.96133])
    assert raw


def test_external_torques_are_read_by_name_in_fr3_joint_order():
    names = [f"fr3_joint{i}" for i in (7, 6, 5, 4, 3, 2, 1)]
    message = SimpleNamespace(name=names, effort=[7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    assert external_joint_torque_message_to_array(message).tolist() == [1, 2, 3, 4, 5, 6, 7]
    with pytest.raises(RuntimeError, match="all FR3 joints"):
        external_joint_torque_message_to_array(SimpleNamespace(name=names[:6], effort=[0.0] * 6))


def test_the_torque_low_pass_starts_on_the_first_sample_and_follows_its_time_constant():
    lowpass = TorqueLowPass(0.02)
    assert lowpass.update([10.0], 0.0).tolist() == [10.0]
    # One time constant later a step has covered 1 - 1/e of the way.
    assert lowpass.update([0.0], 0.02)[0] == pytest.approx(10.0 / np.e)
    assert TorqueLowPass(0.0).update([3.0], 0.0).tolist() == [3.0]


def test_auto_torque_source_is_external_on_hardware_and_commanded_in_ros_sim():
    assert resolve_arm_torque_source("auto", sim=False) == "external"
    assert resolve_arm_torque_source("auto", sim=True) == "commanded"
    for name in ("external", "measured", "desired", "commanded"):
        assert resolve_arm_torque_source(name, sim=False) == name
    assert resolve_arm_torque_source("measured", sim=True) == "measured"
    with pytest.raises(ValueError, match="broadcaster"):
        resolve_arm_torque_source("external", sim=True)
    with pytest.raises(ValueError):
        resolve_arm_torque_source("raw", sim=False)


def test_hardware_records_all_four_torque_candidates_and_ros_sim_two():
    assert arm_torque_candidates(sim=False) == ("external", "measured", "desired", "commanded")
    assert arm_torque_candidates(sim=True) == ("measured", "commanded")


def test_controller_candidates_take_gravity_off_tau_j_and_leave_coriolis_out():
    state = SimpleNamespace(
        tau_measured_filtered=[30.0, -5.0, 0, 0, 0, 0, 1.0],
        tau_gravity=[28.0, -5.5, 0, 0, 0, 0, 0.0],
        tau_task=[1.0, 2.0, 0, 0, 0, 0, 0],
        tau_nullspace=[0.5, 0, 0, 0, 0, 0, 0],
        tau_coriolis=[9.0] * 7,
    )
    candidates = arm_torque_candidates_from_controller(state)
    assert candidates["measured"].tolist() == pytest.approx([2.0, 0.5, 0, 0, 0, 0, 1.0])
    assert candidates["commanded"].tolist() == pytest.approx([1.5, 2.0, 0, 0, 0, 0, 0])
    with pytest.raises(RuntimeError, match="rebuild"):
        arm_torque_candidates_from_controller(SimpleNamespace(tau_task=[0] * 7))


def test_a_failing_or_interrupted_shutdown_step_is_reported_not_raised(capsys):
    from policy_rollout.hardware import _shutdown_step

    def interrupted():
        raise KeyboardInterrupt

    def invalid_context():
        raise RuntimeError("publisher's context is invalid")

    _shutdown_step("abort ramp", interrupted)
    _shutdown_step("abort", invalid_context)
    out = capsys.readouterr().out
    assert "'abort ramp' failed: KeyboardInterrupt" in out
    assert "'abort' failed: RuntimeError" in out


def test_a_thumb_settled_just_past_its_yaw_limit_is_not_a_mapping_fault():
    pytest.importorskip("inspire_hand_driver")
    from inspire_hand_driver import command_overlays
    from inspire_hand_driver import kinematics as kin
    from policy_rollout.hardware import HAND_JOINTS

    thumb_dof = kin.dof_index(command_overlays.THUMB_ABDUCTION_JOINT)
    # 2026-10-29: logical open ratio -0.0227 at the yaw limit.
    physical_ratio = command_overlays.apply_open_ratio_overlay(thumb_dof, -0.0227)
    position = np.zeros(len(HAND_JOINTS))
    position[HAND_JOINTS.index("thumb_proximal_yaw_joint")] = kin.open_ratio_to_rad(thumb_dof, physical_ratio)
    q, _ = physical_hand_state_to_policy(position, np.zeros(len(HAND_JOINTS)))
    # Clamped onto the end of the logical range rather than refused.
    assert q[0] == pytest.approx(kin.open_ratio_to_rad(thumb_dof, 0.0))
    # 2026-11-03, force threshold 1000 g: -0.0307 at the yaw limit.
    position[HAND_JOINTS.index("thumb_proximal_yaw_joint")] = kin.open_ratio_to_rad(
        thumb_dof, command_overlays.apply_open_ratio_overlay(thumb_dof, -0.0307)
    )
    q, _ = physical_hand_state_to_policy(position, np.zeros(len(HAND_JOINTS)))
    assert q[0] == pytest.approx(kin.open_ratio_to_rad(thumb_dof, 0.0))
    position[HAND_JOINTS.index("thumb_proximal_yaw_joint")] = kin.open_ratio_to_rad(
        thumb_dof, command_overlays.apply_open_ratio_overlay(thumb_dof, -0.06)
    )
    with pytest.raises(ValueError, match="outside the overlaid policy range"):
        physical_hand_state_to_policy(position, np.zeros(len(HAND_JOINTS)))


def test_joint_limit_guard_stops_before_the_fr3_reflex():
    """hardware-20261103-154114-907622 drifted joint 5 to 2.875 of 2.8763 rad."""

    reflex = np.array([-0.574, 0.323, 0.389, -2.508, 2.875, 2.193, -2.534])
    assert "fr3_joint5" in joint_limit_violation(reflex, 0.15)
    # The closest a successful ros-sim cycle came: joint 5 at 0.30 rad from its limit.
    cycling = reflex.copy()
    cycling[4] = 2.8763 - 0.30
    assert joint_limit_violation(cycling, 0.15) is None
    # The home pose is nowhere near.
    home = np.array([0.0, -0.2, 0.0, -2.2, 1.68, 2.05, -1.02])
    assert joint_limit_violation(home, 0.15) is None


def test_return_gate_tolerance_is_configurable():
    """hardware-20261103-162709-785253 settled 8-10 mm and 3-5 deg from the reset."""

    reset_q = _reset_grasp_quaternion()
    settled = np.array([-0.0025, 0.0026, -0.0081])
    tilted = fo.quat_mul(fo.quat_from_euler_xyz(0.0, 0.0, np.deg2rad(3.5)), reset_q)
    strict = _coordinator(reset_q)
    assert not strict._returned(settled, tilted, np.zeros(3))
    loose = GripCycleCoordinator(
        rate_hz=15.0,
        max_cycles=1,
        reset_position=np.zeros(3),
        reset_quaternion=reset_q,
        reset_hand=np.zeros(3),
        return_tolerance_m=0.012,
        return_tolerance_deg=7.0,
    )
    assert loose._returned(settled, tilted, np.zeros(3))
