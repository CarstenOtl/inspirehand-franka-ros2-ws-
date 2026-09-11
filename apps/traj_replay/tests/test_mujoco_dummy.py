#!/usr/bin/env python3
"""Standalone FR3 + Inspire Hand MuJoCo sidebar-control bringup.

Run this file directly to open MuJoCo's native viewer.  The right sidebar's
``Control`` section contains position sliders for the seven FR3 joints and the
six independently driven Inspire Hand joints.  The hand sliders are normalized:
1 is fully open and 0 is fully closed.  This dummy is simulation-only; it does
not initialize ROS 2 or send commands to hardware.

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
import struct
import sys
import time
from typing import Any
import xml.etree.ElementTree as ET

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
HAND_MIMICS = (
    ("index_intermediate_joint", "index_proximal_joint", 1.1169),
    ("middle_intermediate_joint", "middle_proximal_joint", 1.1169),
    ("ring_intermediate_joint", "ring_proximal_joint", 1.1169),
    ("pinky_intermediate_joint", "pinky_proximal_joint", 1.1169),
    ("thumb_intermediate_joint", "thumb_proximal_pitch_joint", 1.1425),
    ("thumb_distal_joint", "thumb_intermediate_joint", 0.7508),
)
HAND_FOLLOWERS = tuple(follower for follower, _, _ in HAND_MIMICS)
HAND_JOINT_LABELS = (
    ("Pinky curl", "pinky_proximal_joint"),
    ("Ring curl", "ring_proximal_joint"),
    ("Middle curl", "middle_proximal_joint"),
    ("Index curl", "index_proximal_joint"),
    ("Thumb pitch", "thumb_proximal_pitch_joint"),
    ("Thumb yaw", "thumb_proximal_yaw_joint"),
)
FINGERTIP_BODIES = ("thumb_tip", "index_tip")
FINGERTIP_MARKER_RADIUS_M = 0.006
FINGERTIP_MARKER_RGBA = np.array((1.0, 0.0, 0.0, 0.9))
CONTROLLED_JOINTS = ARM_JOINTS + HAND_JOINTS
DEFAULT_HAND_KP = 40.0
DEFAULT_HAND_KV = 2.0
# Current physical FR3 adapter for the official TienKung 2 Pro hand coordinate
# frame, clocked 90 degrees from the legacy installation.
TIENKUNG_FLANGE_TO_PALM_QUAT = np.array((0.0, 0.0, 0.0, 1.0))
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
            0.0,
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
    normalized: bool = False,
) -> None:
    """Turn one compiled torque actuator into a position servo in memory.

    The production MJCF deliberately uses plain torque motors for
    ``mujoco_ros2_control`` compatibility.  This local viewer instead needs
    ``data.ctrl`` to remain a position target, because the native sidebar
    writes directly into that array.  With ``normalized=True``, ``ctrl`` spans
    0 (the joint's upper/closed limit) to 1 (its lower/open limit), while the
    affine actuator still produces ``kp * (target - position) - kv * velocity``.
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
    low, high = model.jnt_range[joint_id]
    span = float(high - low)
    if not np.isfinite(span) or span <= 0.0:
        raise RuntimeError(f"joint {joint_name!r} has an invalid position range")

    model.actuator_gainprm[actuator_id, 0] = -kp * span if normalized else kp
    model.actuator_biasprm[actuator_id, 0] = kp * high if normalized else 0.0
    model.actuator_biasprm[actuator_id, 1] = -kp
    model.actuator_biasprm[actuator_id, 2] = -kv

    model.actuator_ctrllimited[actuator_id] = mujoco.mjtLimited.mjLIMITED_TRUE
    qpos_address = int(model.jnt_qposadr[joint_id])
    joint_position = float(data.qpos[qpos_address])
    if normalized:
        model.actuator_ctrlrange[actuator_id] = (0.0, 1.0)
        data.ctrl[actuator_id] = np.clip(
            (high - joint_position) / span, 0.0, 1.0
        )
    else:
        model.actuator_ctrlrange[actuator_id] = (low, high)
        data.ctrl[actuator_id] = np.clip(joint_position, low, high)


def configure_sidebar_position_servos(
    model: Any,
    data: Any,
    *,
    arm_kp: float = 300.0,
    arm_kv: float = 30.0,
    hand_kp: float = DEFAULT_HAND_KP,
    hand_kv: float = DEFAULT_HAND_KV,
) -> None:
    """Expose stable, bounded joint-position targets in the Control sidebar."""
    gains = (arm_kp, arm_kv, hand_kp, hand_kv)
    if not all(np.isfinite(gain) and gain >= 0.0 for gain in gains):
        raise ValueError("sidebar servo gains must be finite and nonnegative")

    for joint_name in ARM_JOINTS:
        _position_servo(model, data, joint_name, kp=arm_kp, kv=arm_kv)
    for joint_name in HAND_JOINTS:
        _position_servo(
            model,
            data,
            joint_name,
            kp=hand_kp,
            kv=hand_kv,
            normalized=True,
        )
    mujoco.mj_forward(model, data)


def load_dummy_scene(
    scene: Path = DEFAULT_SCENE,
    *,
    keyframe: str = DEFAULT_KEYFRAME,
    arm_kp: float = 300.0,
    arm_kv: float = 30.0,
    hand_kp: float = DEFAULT_HAND_KP,
    hand_kv: float = DEFAULT_HAND_KV,
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


def measure_hand(model: Any, data: Any) -> tuple[dict[str, float], float]:
    """Return active hand-joint angles in degrees and the fingertip gap in mm."""
    joint_degrees = {}
    for label, joint_name in HAND_JOINT_LABELS:
        joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_address = int(model.jnt_qposadr[joint_id])
        joint_degrees[label] = float(np.degrees(data.qpos[qpos_address]))

    thumb_tip_id, index_tip_id = (
        _named_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for body_name in FINGERTIP_BODIES
    )
    fingertip_distance_mm = 1_000.0 * float(
        np.linalg.norm(data.xpos[thumb_tip_id] - data.xpos[index_tip_id])
    )
    return joint_degrees, fingertip_distance_mm


def format_hand_measurements(model: Any, data: Any) -> tuple[str, str]:
    """Format the live hand measurements as the viewer's two text columns."""
    joint_degrees, fingertip_distance_mm = measure_hand(model, data)
    labels = ["MEASURED HAND STATE", *joint_degrees, "Thumb <-> index tips"]
    values = [
        "",
        *(f"{angle_degrees:7.2f} deg" for angle_degrees in joint_degrees.values()),
        f"{fingertip_distance_mm:7.2f} mm",
    ]
    return "\n".join(labels), "\n".join(values)


def update_fingertip_markers(user_scene: Any, model: Any, data: Any) -> None:
    """Draw non-colliding red markers at the measured fingertip positions."""
    user_scene.ngeom = 0
    for body_name in FINGERTIP_BODIES:
        body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        marker = user_scene.geoms[user_scene.ngeom]
        mujoco.mjv_initGeom(
            marker,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array((FINGERTIP_MARKER_RADIUS_M, 0.0, 0.0)),
            pos=data.xpos[body_id],
            mat=np.eye(3).ravel(),
            rgba=FINGERTIP_MARKER_RGBA,
        )
        user_scene.ngeom += 1


def launch_sidebar_control(model: Any, data: Any, scene: Path) -> None:
    """Run local physics while the native right sidebar edits position targets."""
    import mujoco.viewer

    print(f"Loaded scene: {scene}")
    print("Simulation only: ROS 2 and real-robot commands are disabled.")
    print("Open the right sidebar's Control section to move the arm and hand.")
    print("Hand controls: 1.0 = fully open, 0.0 = fully closed.")
    print("Measured hand angles and the thumb/index tip gap appear at top left.")
    print("Red spheres mark the thumb and index fingertips used for the gap.")
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
            next_measurement_update = 0.0
            while viewer.is_running() and not shutdown_requested:
                with viewer.lock():
                    mujoco.mj_step(model, data)
                    update_fingertip_markers(
                        viewer.user_scn, model, data
                    )

                now = time.monotonic()
                if now >= next_measurement_update:
                    labels, values = format_hand_measurements(model, data)
                    viewer.set_texts(
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            labels,
                            values,
                        )
                    )
                    next_measurement_update = now + 0.05
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

    flange_z = data.xmat[flange_id].reshape(3, 3)[:, 2]
    expected_palm_position = data.xpos[flange_id] + 0.010 * flange_z
    assert np.allclose(data.xpos[palm_id], expected_palm_position, atol=1e-9)
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
    hand_visual_geom_ids = [
        geom_id for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in hand_body_ids
        and int(model.geom_group[geom_id]) == 1
    ]
    assert hand_visual_geom_ids
    for geom_id in hand_visual_geom_ids:
        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            int(model.geom_bodyid[geom_id]),
        )
        material_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_MATERIAL,
            int(model.geom_matid[geom_id]),
        )
        expected = (
            "hand_rubber"
            if body_name in {"thumb_proximal", "thumb_distal"}
            else "hand_shell"
        )
        assert material_name == expected

    for joint_name in CONTROLLED_JOINTS:
        actuator_id = _named_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
        )
        joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_address = int(model.jnt_qposadr[joint_id])
        low, high = model.jnt_range[joint_id]
        assert np.isclose(data.qpos[qpos_address], PICKUP_INIT[joint_name])
        assert model.actuator_biastype[actuator_id] == mujoco.mjtBias.mjBIAS_AFFINE
        if joint_name in HAND_JOINTS:
            expected_control = (high - data.qpos[qpos_address]) / (high - low)
            assert np.allclose(model.actuator_ctrlrange[actuator_id], (0.0, 1.0))
            assert np.isclose(data.ctrl[actuator_id], expected_control)
        else:
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
        if joint_name in HAND_JOINTS:
            low, high = model.jnt_range[joint_id]
            data.ctrl[actuator_id] = (high - target) / (high - low)
        else:
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

    # Check the behavior, not just the declarations: every passive joint must
    # remain on its driver's mimic manifold while the hand moves under physics.
    for follower, driver, multiplier in HAND_MIMICS:
        follower_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, follower)
        driver_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, driver)
        follower_position = data.qpos[int(model.jnt_qposadr[follower_id])]
        driver_position = data.qpos[int(model.jnt_qposadr[driver_id])]
        assert np.isclose(
            follower_position,
            multiplier * driver_position,
            atol=0.01,
        ), f"{follower} did not follow {driver}"

    joint_degrees, fingertip_distance_mm = measure_hand(model, data)
    assert tuple(joint_degrees) == tuple(label for label, _ in HAND_JOINT_LABELS)
    assert all(np.isfinite(tuple(joint_degrees.values())))
    assert np.isfinite(fingertip_distance_mm)
    assert fingertip_distance_mm >= 0.0
    labels, values = format_hand_measurements(model, data)
    assert all(label in labels for label, _ in HAND_JOINT_LABELS)
    assert "Thumb <-> index tips" in labels
    assert values.count(" deg") == len(HAND_JOINTS)
    assert values.count(" mm") == 1


def test_only_the_thumb_uses_the_rubber_material() -> None:
    """Keep cosmetic finger coloring out of both distributed scenes."""
    scenes = (DEFAULT_SCENE, ASSET_DIR / "fr3_inspirehand_replay.xml")
    for scene in scenes:
        root = ET.parse(scene).getroot()
        colored_meshes = [
            geom.get("mesh")
            for geom in root.findall(".//geom[@material='hand_rubber']")
        ]
        assert colored_meshes == ["hand_right_thumb_2", "hand_right_thumb_4"]


def test_dummy_scene_starts_joint7_at_zero() -> None:
    root = ET.parse(DEFAULT_SCENE).getroot()
    start = root.find(".//key[@name='start']")
    assert start is not None
    qpos = [float(value) for value in start.get("qpos").split()]
    assert PICKUP_INIT["fr3_joint7"] == 0.0
    assert qpos[ARM_JOINTS.index("fr3_joint7")] == 0.0


def test_assets_have_the_black_ten_millimeter_adapter_flange() -> None:
    stl_path = ASSET_DIR / "hand" / "adapter_flange.stl"
    stl = stl_path.read_bytes()
    face_count = struct.unpack_from("<I", stl, 80)[0]
    assert face_count == 128
    assert len(stl) == 84 + face_count * 50
    vertices = []
    for face in range(face_count):
        values = struct.unpack_from("<12fH", stl, 84 + face * 50)
        vertices.extend(
            tuple(values[index:index + 3]) for index in (3, 6, 9)
        )
    assert vertices
    assert np.isclose(min(vertex[2] for vertex in vertices), -0.005)
    assert np.isclose(max(vertex[2] for vertex in vertices), 0.005)
    assert np.isclose(
        max(np.hypot(vertex[0], vertex[1]) for vertex in vertices), 0.038
    )

    scenes = (DEFAULT_SCENE, ASSET_DIR / "fr3_inspirehand_replay.xml")
    for scene in scenes:
        root = ET.parse(scene).getroot()
        mesh = root.find(".//asset/mesh[@name='hand_adapter_flange_mesh']")
        assert mesh is not None
        assert mesh.get("file") == "hand/adapter_flange.stl"

        flange = root.find(".//geom[@name='hand_adapter_flange']")
        assert flange is not None
        assert flange.get("type") == "mesh"
        assert flange.get("mesh") == "hand_adapter_flange_mesh"
        assert flange.get("pos") == "0 0 0.005"
        assert flange.get("material") == "flange_black"

        palm = root.find(".//body[@name='hand_base_link']")
        if palm is None:
            palm = root.find(".//body[@name='palm']")
        assert palm is not None
        assert palm.get("pos") == "0 0 0.010"


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
    parser.add_argument(
        "--hand-kp",
        type=float,
        default=DEFAULT_HAND_KP,
        help=f"Hand position stiffness (default: {DEFAULT_HAND_KP:g}).",
    )
    parser.add_argument(
        "--hand-kv",
        type=float,
        default=DEFAULT_HAND_KV,
        help=f"Hand velocity damping (default: {DEFAULT_HAND_KV:g}).",
    )
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
