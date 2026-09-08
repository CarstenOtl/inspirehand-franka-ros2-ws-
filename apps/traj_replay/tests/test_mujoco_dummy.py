#!/usr/bin/env python3
"""Standalone FR3 + Inspire Hand MuJoCo sidebar-control bringup.

Run this file directly to open MuJoCo's native viewer.  The right sidebar's
``Control`` section contains position sliders for the seven FR3 joints and the
six independently driven Inspire Hand joints.  This dummy is simulation-only:
it does not initialize ROS 2 or send commands to hardware.

Pytest only runs the non-interactive model contract test, so test discovery
never opens a window or waits for the viewer to close.

Examples::

    python3 apps/traj_replay/tests/test_mujoco_dummy.py
    python3 apps/traj_replay/tests/test_mujoco_dummy.py --headless --steps 100
    python3 apps/traj_replay/tests/test_mujoco_dummy.py \
        --scene /path/to/another_scene.xml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time
from typing import Any

import numpy as np

try:
    import mujoco
except ModuleNotFoundError as exc:
    mujoco = None
    MUJOCO_IMPORT_ERROR = exc
else:
    MUJOCO_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = REPO_ROOT / "assets" / "fr3_inspirehand"
DEFAULT_SCENE = ASSET_DIR / "fr3_inspirehand.xml"
DEFAULT_KEYFRAME = "start"

ARM_JOINTS = tuple(f"fr3_joint{index}" for index in range(1, 8))
HAND_JOINTS = (
    "pinky_proximal_joint",
    "ring_proximal_joint",
    "middle_proximal_joint",
    "index_proximal_joint",
    "thumb_proximal_pitch_joint",
    "thumb_proximal_yaw_joint",
)
HAND_FOLLOWERS = (
    "index_intermediate_joint",
    "middle_intermediate_joint",
    "ring_intermediate_joint",
    "pinky_intermediate_joint",
    "thumb_intermediate_joint",
    "thumb_distal_joint",
)
CONTROLLED_JOINTS = ARM_JOINTS + HAND_JOINTS
# FR3 adapter for the official TienKung 2 Pro hand coordinate frame.
TIENKUNG_FLANGE_TO_PALM_QUAT = np.array((2**-0.5, 0.0, 0.0, -(2**-0.5)))
PICKUP_INIT = dict(
    zip(
        CONTROLLED_JOINTS,
        (
            -0.392613,
            0.004288,
            -0.072713,
            -1.811251,
            0.592754,
            2.280553,
            -2.620279,
            1.0999,
            1.0999,
            1.0999,
            0.44,
            0.2,
            1.14,
        ),
        strict=True,
    )
)


def _require_mujoco() -> Any:
    if mujoco is None:
        raise RuntimeError(
            "MuJoCo is not importable in this shell. Install the Python "
            "'mujoco' package or run inside the workspace simulation "
            "environment, then retry."
        ) from MUJOCO_IMPORT_ERROR
    return mujoco


def resolve_scene(scene: Path) -> Path:
    """Resolve a scene name relative to the combined local asset bundle."""
    scene = scene.expanduser()
    if not scene.is_absolute():
        scene = ASSET_DIR / scene
    scene = scene.resolve()
    if not scene.is_file():
        raise FileNotFoundError(f"MuJoCo scene not found: {scene}")
    return scene


def _named_id(model: Any, object_type: Any, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise RuntimeError(f"MuJoCo model is missing {name!r}")
    return object_id


def _position_servo(
    model: Any,
    data: Any,
    joint_name: str,
    *,
    kp: float,
    kv: float,
) -> None:
    """Turn one compiled torque actuator into a position servo in memory.

    The production MJCF deliberately uses plain torque motors for
    ``mujoco_ros2_control`` compatibility.  This local viewer instead needs
    ``data.ctrl`` to remain a position target, because the native sidebar
    writes directly into that array.  MuJoCo's affine actuator equation below
    is ``force = kp * ctrl - kp * length - kv * velocity``.
    """
    actuator_id = _named_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
    )
    joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    transmission_joint = int(model.actuator_trnid[actuator_id, 0])
    if transmission_joint != joint_id:
        raise RuntimeError(
            f"actuator {joint_name!r} does not directly drive its namesake joint"
        )
    if not bool(model.jnt_limited[joint_id]):
        raise RuntimeError(f"joint {joint_name!r} needs limits for sidebar control")

    model.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_gainprm[actuator_id, :] = 0.0
    model.actuator_biasprm[actuator_id, :] = 0.0
    model.actuator_gainprm[actuator_id, 0] = kp
    model.actuator_biasprm[actuator_id, 1] = -kp
    model.actuator_biasprm[actuator_id, 2] = -kv

    low, high = model.jnt_range[joint_id]
    model.actuator_ctrllimited[actuator_id] = mujoco.mjtLimited.mjLIMITED_TRUE
    model.actuator_ctrlrange[actuator_id] = (low, high)
    qpos_address = int(model.jnt_qposadr[joint_id])
    data.ctrl[actuator_id] = np.clip(data.qpos[qpos_address], low, high)


def configure_sidebar_position_servos(
    model: Any,
    data: Any,
    *,
    arm_kp: float = 300.0,
    arm_kv: float = 30.0,
    hand_kp: float = 10.0,
    hand_kv: float = 0.4,
) -> None:
    """Expose stable, bounded joint-position targets in the Control sidebar."""
    gains = (arm_kp, arm_kv, hand_kp, hand_kv)
    if not all(np.isfinite(gain) and gain >= 0.0 for gain in gains):
        raise ValueError("sidebar servo gains must be finite and nonnegative")

    for joint_name in ARM_JOINTS:
        _position_servo(model, data, joint_name, kp=arm_kp, kv=arm_kv)
    for joint_name in HAND_JOINTS:
        _position_servo(model, data, joint_name, kp=hand_kp, kv=hand_kv)
    mujoco.mj_forward(model, data)


def load_dummy_scene(
    scene: Path = DEFAULT_SCENE,
    *,
    keyframe: str = DEFAULT_KEYFRAME,
    arm_kp: float = 300.0,
    arm_kv: float = 30.0,
    hand_kp: float = 10.0,
    hand_kv: float = 0.4,
) -> tuple[Any, Any, Path]:
    """Load and initialize the combined robot for local sidebar control."""
    mj = _require_mujoco()
    scene = resolve_scene(scene)
    model = mj.MjModel.from_xml_path(str(scene))
    data = mj.MjData(model)

    if keyframe:
        keyframe_id = int(
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, keyframe)
        )
        if keyframe_id < 0:
            raise RuntimeError(
                f"MuJoCo scene {scene} has no keyframe named {keyframe!r}"
            )
        mj.mj_resetDataKeyframe(model, data, keyframe_id)

    configure_sidebar_position_servos(
        model,
        data,
        arm_kp=arm_kp,
        arm_kv=arm_kv,
        hand_kp=hand_kp,
        hand_kv=hand_kv,
    )
    return model, data, scene


def launch_sidebar_control(model: Any, data: Any, scene: Path) -> None:
    """Run local physics while the native right sidebar edits position targets."""
    import mujoco.viewer

    print(f"Loaded scene: {scene}")
    print("Simulation only: ROS 2 and real-robot commands are disabled.")
    print("Open the right sidebar's Control section to move the arm and hand.")
    print("Close the window or press Ctrl+C in this terminal to exit.")

    shutdown_requested = False

    def request_shutdown(_signum: int, _frame: Any) -> None:
        nonlocal shutdown_requested
        shutdown_requested = True

    previous_sigint_handler = signal.signal(signal.SIGINT, request_shutdown)
    try:
        with mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=True,
            show_right_ui=True,
        ) as viewer:
            next_step = time.monotonic()
            while viewer.is_running() and not shutdown_requested:
                with viewer.lock():
                    mujoco.mj_step(model, data)
                viewer.sync()

                next_step += float(model.opt.timestep)
                delay = next_step - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
                else:
                    next_step = time.monotonic()

        # launch_passive() owns a daemon UI thread. Its context manager only
        # requests shutdown, so wait for the native viewer object to be
        # released before Python starts interpreter teardown. Without this,
        # MuJoCo 3.10 can segfault after an otherwise successful Ctrl+C.
        teardown_deadline = time.monotonic() + 5.0
        while viewer._sim() is not None and time.monotonic() < teardown_deadline:
            time.sleep(0.01)
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)

    if shutdown_requested:
        print("\nMuJoCo viewer stopped by user.", flush=True)


def test_dummy_scene_has_sidebar_position_controls() -> None:
    """Check the interactive dummy without opening its viewer."""
    if mujoco is None:
        import pytest

        pytest.skip(f"MuJoCo is not importable: {MUJOCO_IMPORT_ERROR}")

    model, data, _ = load_dummy_scene()
    assert model.nu == len(CONTROLLED_JOINTS)
    assert _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3_link7") >= 0
    flange_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3_link8")
    palm_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_base_link")

    assert np.allclose(data.xpos[palm_id], data.xpos[flange_id], atol=1e-9)
    flange_inverse = data.xquat[flange_id].copy()
    flange_inverse[1:] *= -1.0
    flange_to_palm = np.empty(4)
    mujoco.mju_mulQuat(flange_to_palm, flange_inverse, data.xquat[palm_id])
    assert np.isclose(
        abs(np.dot(flange_to_palm, TIENKUNG_FLANGE_TO_PALM_QUAT)),
        1.0,
        atol=1e-6,
    )

    hand_body_ids = set()
    for body_id in range(model.nbody):
        ancestor = body_id
        while ancestor > 0:
            if ancestor == palm_id:
                hand_body_ids.add(body_id)
                break
            ancestor = int(model.body_parentid[ancestor])
    hand_geom_ids = [
        geom_id for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in hand_body_ids
    ]
    assert hand_geom_ids
    for geom_id in hand_geom_ids:
        assert np.allclose(model.geom_rgba[geom_id], (1.0, 1.0, 1.0, 1.0))

    for joint_name in CONTROLLED_JOINTS:
        actuator_id = _named_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
        )
        joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_address = int(model.jnt_qposadr[joint_id])
        low, high = model.jnt_range[joint_id]
        assert np.isclose(data.qpos[qpos_address], PICKUP_INIT[joint_name])
        assert model.actuator_biastype[actuator_id] == mujoco.mjtBias.mjBIAS_AFFINE
        assert np.allclose(model.actuator_ctrlrange[actuator_id], (low, high))
        assert np.isclose(data.ctrl[actuator_id], data.qpos[qpos_address])

    for joint_name in HAND_FOLLOWERS:
        assert _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) >= 0
        assert (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            < 0
        )

    targets = {"fr3_joint1": 0.25, "index_proximal_joint": 0.7}
    initial_error = {}
    for joint_name, target in targets.items():
        actuator_id = _named_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
        )
        joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_address = int(model.jnt_qposadr[joint_id])
        initial_error[joint_name] = abs(float(data.qpos[qpos_address]) - target)
        data.ctrl[actuator_id] = target

    for _ in range(1_000):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
    for joint_name, target in targets.items():
        joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_address = int(model.jnt_qposadr[joint_id])
        final_error = abs(float(data.qpos[qpos_address]) - target)
        assert final_error < initial_error[joint_name]
        assert final_error < 0.01


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE,
        help=f"MJCF scene path or filename relative to {ASSET_DIR}",
    )
    parser.add_argument(
        "--keyframe",
        default=DEFAULT_KEYFRAME,
        help="Initial keyframe name; pass an empty string to use qpos0.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Load and step the scene without opening a viewer.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Physics steps to execute in headless mode (default: 10).",
    )
    parser.add_argument("--arm-kp", type=float, default=300.0)
    parser.add_argument("--arm-kv", type=float, default=30.0)
    parser.add_argument("--hand-kp", type=float, default=10.0)
    parser.add_argument("--hand-kv", type=float, default=0.4)
    args = parser.parse_args(argv)
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        model, data, scene = load_dummy_scene(
            args.scene,
            keyframe=args.keyframe,
            arm_kp=args.arm_kp,
            arm_kv=args.arm_kv,
            hand_kp=args.hand_kp,
            hand_kv=args.hand_kv,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"MuJoCo dummy ready: nbody={model.nbody} ngeom={model.ngeom} "
        f"njnt={model.njnt} nu={model.nu}"
    )
    if args.headless:
        for _ in range(args.steps):
            mujoco.mj_step(model, data)
        print(f"Headless validation complete: {args.steps} steps, t={data.time:.3f} s")
    else:
        launch_sidebar_control(model, data, scene)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
