#!/usr/bin/env python3
"""Replay the retained forgeUltra FR3 threading trajectory in MuJoCo.

This is a kinematic compatibility viewer, not a second physics rollout. It
writes all 19 recorded joint poses into an FR3 model carrying the exact Inspire
hand kinematics and meshes used by the source training environment, and it
displays the recorded nut pose. A green sphere is the recorded TCP; a smaller
magenta sphere is the replayed thumb/index-tip midpoint. They should overlap.

Examples::

    python apps/traj_replay/tests/test_mujoco_traj_replay.py
    python apps/traj_replay/tests/test_mujoco_traj_replay.py --speed 2 --loop
    python apps/traj_replay/tests/test_mujoco_traj_replay.py --headless

The replay-only MJCF intentionally differs from the ROS-control MJCF: the
recording contains the older 12-DoF training hand, while the hardware workspace
uses a different six-driver linkage. Mixing those models visibly distorts the
grasp and introduces about 10 mm of TCP error.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

try:
    import mujoco
except ModuleNotFoundError as exc:
    mujoco = None
    MUJOCO_IMPORT_ERROR = exc
else:
    MUJOCO_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[3]
APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY = APP_DIR / "demo_trajs" / "traj_1"
DEFAULT_SCENE = REPO_ROOT / "assets" / "fr3_inspirehand" / "fr3_inspirehand_replay.xml"

# The replay scene intentionally uses the source environment's joint names, so
# every measured coordinate—including passive followers—is copied directly.
MODEL_TO_RECORDED_JOINT = {
    name: name
    for name in (
        *(f"fr3_joint{i}" for i in range(1, 8)),
        "index_joint_0",
        "little_joint_0",
        "middle_joint_0",
        "ring_joint_0",
        "thumb_joint_0",
        "index_joint_1",
        "little_joint_1",
        "middle_joint_1",
        "ring_joint_1",
        "thumb_joint_1",
        "thumb_joint_2",
        "thumb_joint_3",
    )
}


@dataclass(frozen=True)
class ReplayTrajectory:
    """One environment selected from the recorded batched trajectory."""

    directory: Path
    metadata: dict[str, Any]
    joint_names: tuple[str, ...]
    sample_time_s: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    nut_pos: np.ndarray
    nut_quat: np.ndarray
    tcp_pos: np.ndarray
    replay_phase: np.ndarray
    cycle: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.sample_time_s.shape[0])


@dataclass(frozen=True)
class JointBinding:
    """Recorded column and local scalar joint addresses."""

    local_name: str
    source_column: int
    joint_id: int
    qpos_address: int
    qvel_address: int


@dataclass(frozen=True)
class FitReport:
    sample_count: int
    tcp_rmse_m: float
    tcp_mean_error_m: float
    tcp_max_error_m: float
    limit_exceedances: int
    max_limit_excursion_rad: float


def _require_mujoco() -> Any:
    if mujoco is None:
        raise RuntimeError(
            "MuJoCo is not importable. Run this script in the workspace "
            "simulation environment."
        ) from MUJOCO_IMPORT_ERROR
    return mujoco


def _resolve_trajectory_directory(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        path = path.parent
    if not (path / "metadata.json").is_file():
        raise FileNotFoundError(f"trajectory metadata not found under {path}")
    return path


def load_trajectory(path: Path, environment: int = 0) -> ReplayTrajectory:
    """Load and validate one environment from a Forge replay artifact."""
    directory = _resolve_trajectory_directory(path)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    data_file = directory / metadata.get("data_file", "replay_data.npz")
    if not data_file.is_file():
        raise FileNotFoundError(f"trajectory data not found: {data_file}")

    required = {
        "sample_time_s",
        "joint_pos",
        "joint_vel",
        "nut_pos",
        "nut_quat",
        "tcp_pos",
        "replay_phase",
        "cycle",
    }
    with np.load(data_file, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"trajectory is missing fields: {sorted(missing)}")
        sample_time_s = np.asarray(archive["sample_time_s"], dtype=np.float64).copy()
        joint_pos_all = np.asarray(archive["joint_pos"], dtype=np.float64)
        joint_vel_all = np.asarray(archive["joint_vel"], dtype=np.float64)
        nut_pos_all = np.asarray(archive["nut_pos"], dtype=np.float64)
        nut_quat_all = np.asarray(archive["nut_quat"], dtype=np.float64)
        tcp_pos_all = np.asarray(archive["tcp_pos"], dtype=np.float64)
        replay_phase = np.asarray(archive["replay_phase"]).copy()
        cycle = np.asarray(archive["cycle"]).copy()

        if joint_pos_all.ndim != 3:
            raise ValueError(f"joint_pos must be [time, environment, joint], got {joint_pos_all.shape}")
        environment_count = joint_pos_all.shape[1]
        if not 0 <= environment < environment_count:
            raise ValueError(
                f"environment {environment} is outside [0, {environment_count - 1}]"
            )
        joint_pos = joint_pos_all[:, environment, :].copy()
        joint_vel = joint_vel_all[:, environment, :].copy()
        nut_pos = nut_pos_all[:, environment, :].copy()
        nut_quat = nut_quat_all[:, environment, :].copy()
        tcp_pos = tcp_pos_all[:, environment, :].copy()

    joint_names = tuple(metadata.get("joint_names", ()))
    sample_count = int(metadata.get("sample_count", sample_time_s.shape[0]))
    arrays = (joint_pos, joint_vel, nut_pos, nut_quat, tcp_pos, replay_phase, cycle)
    if sample_count != sample_time_s.shape[0] or any(
        array.shape[0] != sample_count for array in arrays
    ):
        raise ValueError("metadata sample_count does not match the trajectory arrays")
    if joint_pos.shape != joint_vel.shape:
        raise ValueError("joint_pos and joint_vel shapes differ")
    if joint_pos.shape[1] != len(joint_names):
        raise ValueError("metadata joint_names does not match the joint array width")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("metadata joint_names contains duplicates")
    if not np.all(np.isfinite(sample_time_s)) or np.any(np.diff(sample_time_s) < 0.0):
        raise ValueError("sample_time_s must be finite and monotonic")
    for name, array in (
        ("joint_pos", joint_pos),
        ("joint_vel", joint_vel),
        ("nut_pos", nut_pos),
        ("nut_quat", nut_quat),
        ("tcp_pos", tcp_pos),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")

    return ReplayTrajectory(
        directory=directory,
        metadata=metadata,
        joint_names=joint_names,
        sample_time_s=sample_time_s,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        nut_pos=nut_pos,
        nut_quat=nut_quat,
        tcp_pos=tcp_pos,
        replay_phase=replay_phase,
        cycle=cycle,
    )


def load_scene(path: Path) -> tuple[Any, Any, Path]:
    mj = _require_mujoco()
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MuJoCo replay scene not found: {path}")
    model = mj.MjModel.from_xml_path(str(path))
    data = mj.MjData(model)
    return model, data, path


def _named_id(model: Any, object_type: Any, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise ValueError(f"MuJoCo replay scene is missing {name!r}")
    return object_id


def build_joint_bindings(model: Any, trajectory: ReplayTrajectory) -> tuple[JointBinding, ...]:
    """Bind every recorded training joint to its namesake scalar hinge."""
    source_columns = {name: index for index, name in enumerate(trajectory.joint_names)}
    missing = set(MODEL_TO_RECORDED_JOINT.values()) - set(source_columns)
    if missing:
        raise ValueError(f"recorded trajectory is missing driver joints: {sorted(missing)}")

    bindings = []
    for local_name, recorded_name in MODEL_TO_RECORDED_JOINT.items():
        joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, local_name)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"{local_name!r} is not a scalar hinge")
        bindings.append(
            JointBinding(
                local_name=local_name,
                source_column=source_columns[recorded_name],
                joint_id=joint_id,
                qpos_address=int(model.jnt_qposadr[joint_id]),
                qvel_address=int(model.jnt_dofadr[joint_id]),
            )
        )
    return tuple(bindings)


def _mocap_id(model: Any, body_name: str) -> int:
    body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise ValueError(f"body {body_name!r} is not a mocap body")
    return mocap_id


def _set_normalized_quaternion(destination: np.ndarray, value: np.ndarray) -> None:
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("recorded nut quaternion is invalid")
    destination[:] = value / norm


def apply_sample(
    model: Any,
    data: Any,
    trajectory: ReplayTrajectory,
    bindings: Sequence[JointBinding],
    sample_index: int,
) -> tuple[float, int, float]:
    """Apply one sample and report TCP error and nominal-limit excursions."""
    if not 0 <= sample_index < trajectory.sample_count:
        raise IndexError(f"sample {sample_index} is outside the trajectory")

    data.qvel[:] = 0.0
    limit_exceedances = 0
    max_limit_excursion_rad = 0.0
    for binding in bindings:
        position = float(trajectory.joint_pos[sample_index, binding.source_column])
        if bool(model.jnt_limited[binding.joint_id]):
            low, high = model.jnt_range[binding.joint_id]
            excursion = max(float(low) - position, position - float(high), 0.0)
            if excursion > 0.0:
                limit_exceedances += 1
                max_limit_excursion_rad = max(
                    max_limit_excursion_rad, excursion
                )
        data.qpos[binding.qpos_address] = position
        data.qvel[binding.qvel_address] = trajectory.joint_vel[
            sample_index, binding.source_column
        ]
    nut_mocap = _mocap_id(model, "recorded_nut")
    data.mocap_pos[nut_mocap] = trajectory.nut_pos[sample_index]
    _set_normalized_quaternion(data.mocap_quat[nut_mocap], trajectory.nut_quat[sample_index])

    recorded_tcp_mocap = _mocap_id(model, "recorded_tcp_marker")
    data.mocap_pos[recorded_tcp_mocap] = trajectory.tcp_pos[sample_index]

    data.time = float(trajectory.sample_time_s[sample_index])
    mujoco.mj_forward(model, data)

    thumb_tip = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "thumb_tip")
    index_tip = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "index_tip")
    replayed_tcp = 0.5 * (data.xpos[thumb_tip] + data.xpos[index_tip])
    replayed_tcp_mocap = _mocap_id(model, "replayed_tcp_marker")
    data.mocap_pos[replayed_tcp_mocap] = replayed_tcp
    mujoco.mj_forward(model, data)

    tcp_error_m = float(np.linalg.norm(replayed_tcp - trajectory.tcp_pos[sample_index]))
    return tcp_error_m, limit_exceedances, max_limit_excursion_rad


def evaluate_fit(
    model: Any,
    data: Any,
    trajectory: ReplayTrajectory,
    bindings: Sequence[JointBinding],
    indices: Sequence[int] | range,
) -> FitReport:
    errors = []
    limit_exceedances = 0
    max_limit_excursion_rad = 0.0
    for index in indices:
        error, exceedances, excursion = apply_sample(
            model, data, trajectory, bindings, int(index)
        )
        errors.append(error)
        limit_exceedances += exceedances
        max_limit_excursion_rad = max(max_limit_excursion_rad, excursion)
    if not errors:
        raise ValueError("the selected replay range contains no samples")
    errors_array = np.asarray(errors)
    return FitReport(
        sample_count=len(errors),
        tcp_rmse_m=float(np.sqrt(np.mean(np.square(errors_array)))),
        tcp_mean_error_m=float(np.mean(errors_array)),
        tcp_max_error_m=float(np.max(errors_array)),
        limit_exceedances=limit_exceedances,
        max_limit_excursion_rad=max_limit_excursion_rad,
    )


def print_fit_report(trajectory: ReplayTrajectory, report: FitReport) -> None:
    duration = float(trajectory.sample_time_s[-1] - trajectory.sample_time_s[0])
    print(f"Task: {trajectory.metadata.get('threading_task', 'unknown')}")
    print(
        f"Trajectory: {trajectory.sample_count} samples, {duration:.2f} s, "
        f"{trajectory.metadata.get('recording_frequency_hz', 'unknown')} Hz"
    )
    print(
        "Training-hand tip midpoint vs recorded TCP: "
        f"RMSE={report.tcp_rmse_m * 1000.0:.4f} mm, "
        f"mean={report.tcp_mean_error_m * 1000.0:.4f} mm, "
        f"max={report.tcp_max_error_m * 1000.0:.4f} mm"
    )
    print(
        f"Recorded joint-limit excursions (not clipped): "
        f"{report.limit_exceedances} values, "
        f"largest={report.max_limit_excursion_rad:.3g} rad"
    )


def launch_viewer(
    model: Any,
    data: Any,
    scene: Path,
    trajectory: ReplayTrajectory,
    bindings: Sequence[JointBinding],
    start: int,
    end: int,
    speed: float,
    loop: bool,
) -> None:
    import mujoco.viewer

    print(f"Scene: {scene}")
    print("Green: recorded TCP; magenta: replayed thumb/index midpoint; orange: nut")
    print("Close the viewer or press Ctrl+C to stop.")

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=True, show_right_ui=False
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = _named_id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "policy_replay_front"
        )

        try:
            while viewer.is_running():
                replay_wall_start = time.monotonic()
                replay_time_start = float(trajectory.sample_time_s[start])
                previous_state = None
                for index in range(start, end):
                    if not viewer.is_running():
                        return
                    deadline = replay_wall_start + (
                        float(trajectory.sample_time_s[index]) - replay_time_start
                    ) / speed
                    delay = deadline - time.monotonic()
                    if delay > 0.0:
                        time.sleep(delay)

                    with viewer.lock():
                        apply_sample(model, data, trajectory, bindings, index)
                    viewer.sync()

                    state = (str(trajectory.replay_phase[index]), int(trajectory.cycle[index]))
                    if state != previous_state:
                        print(
                            f"sample {index:4d}/{end - 1}: "
                            f"cycle={state[1]} phase={state[0]}",
                            flush=True,
                        )
                        previous_state = state
                if not loop:
                    print("Replay complete; close the viewer to exit.", flush=True)
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.02)
                    return
        except KeyboardInterrupt:
            viewer.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--environment", type=int, default=0)
    parser.add_argument("--start", type=int, default=0, help="First sample index.")
    parser.add_argument("--end", type=int, default=None, help="Exclusive final sample index.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Validate every selected sample and print fit metrics without a viewer.",
    )
    parser.add_argument(
        "--max-tcp-error-mm",
        type=float,
        default=None,
        help="Optional headless failure threshold for maximum TCP position error.",
    )
    args = parser.parse_args(argv)
    if args.environment < 0:
        parser.error("--environment must be nonnegative")
    if args.start < 0:
        parser.error("--start must be nonnegative")
    if args.end is not None and args.end <= args.start:
        parser.error("--end must be greater than --start")
    if args.speed <= 0.0 or not np.isfinite(args.speed):
        parser.error("--speed must be finite and positive")
    if args.max_tcp_error_mm is not None and args.max_tcp_error_mm < 0.0:
        parser.error("--max-tcp-error-mm must be nonnegative")
    return args


def test_replay_contract_and_training_geometry() -> None:
    """Pytest smoke test; never opens the interactive viewer."""
    if mujoco is None:
        import pytest

        pytest.skip(f"MuJoCo is not importable: {MUJOCO_IMPORT_ERROR}")

    trajectory = load_trajectory(DEFAULT_TRAJECTORY)
    model, data, _ = load_scene(DEFAULT_SCENE)
    bindings = build_joint_bindings(model, trajectory)

    assert trajectory.sample_count == 1907
    assert len(bindings) == 19
    first_error, _, _ = apply_sample(model, data, trajectory, bindings, 0)
    assert np.isfinite(first_error)
    assert np.allclose(
        data.mocap_pos[_mocap_id(model, "recorded_nut")],
        trajectory.nut_pos[0],
    )

    for binding in bindings:
        expected = trajectory.joint_pos[0, binding.source_column]
        assert np.isclose(data.qpos[binding.qpos_address], expected)

    report = evaluate_fit(
        model, data, trajectory, bindings, range(0, trajectory.sample_count, 64)
    )
    assert report.sample_count > 20
    assert np.isfinite(report.tcp_rmse_m)
    assert np.isfinite(report.tcp_max_error_m)
    assert report.tcp_max_error_m < 2.0e-6


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        trajectory = load_trajectory(args.trajectory, args.environment)
        model, data, scene = load_scene(args.scene)
        bindings = build_joint_bindings(model, trajectory)
        end = trajectory.sample_count if args.end is None else min(
            args.end, trajectory.sample_count
        )
        if args.start >= end:
            raise ValueError(
                f"--start {args.start} is beyond the selected trajectory end {end}"
            )

        # Compute the fit once before visualization so the user has a numeric
        # answer as well as the overlaid green/magenta markers.
        report = evaluate_fit(
            model, data, trajectory, bindings, range(args.start, end)
        )
        print_fit_report(trajectory, report)
        if (
            args.max_tcp_error_mm is not None
            and report.tcp_max_error_m * 1000.0 > args.max_tcp_error_mm
        ):
            print(
                f"FAIL: maximum TCP error exceeds {args.max_tcp_error_mm:g} mm",
                file=sys.stderr,
            )
            return 1

        if not args.headless:
            apply_sample(model, data, trajectory, bindings, args.start)
            launch_viewer(
                model,
                data,
                scene,
                trajectory,
                bindings,
                args.start,
                end,
                args.speed,
                args.loop,
            )
        return 0
    except (FileNotFoundError, IndexError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
