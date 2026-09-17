#!/usr/bin/env python3
"""Replay a coordinated trajectory in the policy-rollout MuJoCo scene.

Unlike ``test_mujoco_traj_replay.py``, this uses the corrected dynamic
``ThreadingScene`` from ``apps/policy_rollout``: the training-height FR3,
official Inspire fingertip frames, and the self-locking M24 thread pair.  Arm
and hand waypoints are tracked through the same joint-PD gains used by the
policy rollout.  Metadata release flags hold the thread during the scripted
release/retreat and release it when the next policy cycle starts.

This process has no ROS imports and cannot command physical hardware.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[3]
POLICY_APP = WORKSPACE / "apps/policy_rollout"
for path in (
    POLICY_APP,
    WORKSPACE / "src/inspire_franka_trajectory_replay",
    WORKSPACE / "src/franka_trajectory_replay",
    WORKSPACE / "src/inspire_hand_driver",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inspire_franka_trajectory_replay import release_phase  # noqa: E402
from inspire_franka_trajectory_replay.trajectory import (  # noqa: E402
    HAND_JOINTS,
    load_trajectory,
)
from policy_rollout import forge_osc as fo  # noqa: E402
from policy_rollout.mujoco_threading_env import (  # noqa: E402
    ARM_JOINTS,
    HAND_POLICY_JOINTS,
    PHYSICS_DT,
    ThreadingScene,
)


DEFAULT_NUT_SOURCE = (
    WORKSPACE / "apps/traj_replay/demo_trajs/traj_3_joint5_cap_2p8"
)
DRIVER_HAND_COLUMN = {
    "little_joint_0": HAND_JOINTS.index("pinky_proximal_joint"),
    "ring_joint_0": HAND_JOINTS.index("ring_proximal_joint"),
    "middle_joint_0": HAND_JOINTS.index("middle_proximal_joint"),
    "index_joint_0": HAND_JOINTS.index("index_proximal_joint"),
    "thumb_joint_1": HAND_JOINTS.index("thumb_proximal_pitch_joint"),
    "thumb_joint_0": HAND_JOINTS.index("thumb_proximal_yaw_joint"),
}


def _initial_nut_quaternion(path: Path, environment: int) -> np.ndarray:
    directory = path.expanduser().resolve()
    if directory.is_file():
        data_path = directory
    else:
        data_path = directory / "replay_data.npz"
    with np.load(data_path, allow_pickle=False) as data:
        if "nut_quat" not in data:
            raise ValueError(f"nut source has no nut_quat field: {data_path}")
        values = np.asarray(data["nut_quat"], dtype=float)
        if values.ndim == 3:
            if not 0 <= environment < values.shape[1]:
                raise ValueError(
                    f"environment {environment} is outside [0, {values.shape[1] - 1}]"
                )
            return values[0, environment].copy()
        if values.ndim == 2 and values.shape[1] == 4:
            return values[0].copy()
        raise ValueError(f"nut_quat has unsupported shape {values.shape}")


def _scene_hand(hand: np.ndarray) -> np.ndarray:
    return np.asarray(
        [hand[DRIVER_HAND_COLUMN[name]] for name in HAND_POLICY_JOINTS],
        dtype=float,
    )


def _target_at(trajectory, source_time: float) -> tuple[np.ndarray, np.ndarray, int]:
    index = int(np.searchsorted(trajectory.time, source_time, side="right") - 1)
    index = int(np.clip(index, 0, len(trajectory.time) - 1))
    if index == len(trajectory.time) - 1:
        arm = trajectory.arm[index]
        velocity = np.zeros(len(ARM_JOINTS))
        hand = trajectory.hand[index]
    else:
        dt = float(trajectory.time[index + 1] - trajectory.time[index])
        fraction = np.clip((source_time - trajectory.time[index]) / dt, 0.0, 1.0)
        arm = (1.0 - fraction) * trajectory.arm[index] + fraction * trajectory.arm[index + 1]
        velocity = (trajectory.arm[index + 1] - trajectory.arm[index]) / dt
        hand = (1.0 - fraction) * trajectory.hand[index] + fraction * trajectory.hand[index + 1]
    return np.asarray(arm), np.concatenate((velocity, _scene_hand(hand))), index


def _release_events(index):
    by_sample = {}
    for cycle in index.releases:
        by_sample.setdefault(cycle.release_sample, []).append("hold")
        if cycle.cycle > index.releases[0].cycle:
            by_sample.setdefault(cycle.start_sample, []).append("next_cycle")
    return by_sample


def _track_tick(scene, arm_target, velocity_and_hand, time_scale: float) -> None:
    velocity = velocity_and_hand[: len(ARM_JOINTS)] / time_scale
    hand = velocity_and_hand[len(ARM_JOINTS) :]
    q = scene.q(ARM_JOINTS)
    dq = scene.dq(ARM_JOINTS)
    command = np.concatenate((arm_target, velocity, hand[:3]))
    scene._apply_arm_torque(fo.joint_pd_torque(q, dq, command))
    scene._apply_hand_targets(hand[:3])
    scene._step_physics()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", help="generated coordinated trajectory directory")
    parser.add_argument("--nut-source", default=str(DEFAULT_NUT_SOURCE))
    parser.add_argument("--env", type=int, default=0)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--camera", default="policy_replay_front")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)
    if args.time_scale <= 0.0 or args.speed <= 0.0:
        parser.error("--time-scale and --speed must be positive")

    trajectory = load_trajectory(args.trajectory, environment=args.env)
    if trajectory.hand is None:
        parser.error("trajectory has no Inspire hand coordinates")
    releases = release_phase.load(trajectory.source)
    if releases is None or not releases.releases:
        parser.error("trajectory metadata has no per-cycle release flags")

    nut_quat = _initial_nut_quaternion(Path(args.nut_source), args.env)
    scene = ThreadingScene(nut_quat_wxyz=nut_quat)
    initial_hand = _scene_hand(trajectory.hand[0])
    reset_hand = dict(zip(HAND_POLICY_JOINTS, initial_hand))
    events = _release_events(releases)

    viewer_context = None
    if not args.headless:
        import mujoco
        import mujoco.viewer

        viewer_context = mujoco.viewer.launch_passive(scene.model, scene.data)
        viewer = viewer_context.__enter__()
        camera = scene.model.camera(args.camera)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera.id
    else:
        viewer = None

    loops = 0
    try:
        while True:
            scene.reset(trajectory.arm[0], reset_hand)
            playback_start = scene.sim_time
            last_sample = -1
            while True:
                tick_started = time.perf_counter()
                source_time = (scene.sim_time - playback_start) / args.time_scale
                if source_time > trajectory.time[-1] + 0.5 * PHYSICS_DT:
                    break
                arm, velocity_and_hand, sample = _target_at(trajectory, source_time)
                if sample != last_sample:
                    for crossed in range(last_sample + 1, sample + 1):
                        for event in events.get(crossed, ()):
                            if event == "hold":
                                scene.hold_thread()
                            else:
                                scene.rebase_turn_reference()
                                scene.release_thread_hold()
                    last_sample = sample
                _track_tick(scene, arm, velocity_and_hand, args.time_scale)
                if viewer is not None:
                    if not viewer.is_running():
                        return 0
                    viewer.sync()
                    remaining = PHYSICS_DT / args.speed - (
                        time.perf_counter() - tick_started
                    )
                    if remaining > 0.0:
                        time.sleep(remaining)
            loops += 1
            print(
                f"completed {len(releases.releases)} cycles; "
                f"nut turn={np.degrees(scene.twist_angle):.2f} deg, "
                f"axial={scene.axial_position:.5f} m"
            )
            if not args.loop:
                break
    finally:
        if viewer_context is not None:
            viewer_context.__exit__(None, None, None)

    print(
        f"scene=policy_rollout.ThreadingScene samples={len(trajectory.time)} "
        f"loops={loops} simulated={scene.sim_time:.2f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
