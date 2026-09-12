#!/usr/bin/env python3
"""Inspect a calibrated RealSense pose with a MuJoCo-only robot preview.

The input may be either the JSON object printed after ``CALIBRATION_RESULT``
or an entire captured ROS log containing that line.  The default viewer is
strictly offline.  ``--live`` adds read-only ROS subscriptions for Franka joint
angles and the real RGB image, then presents the real and MuJoCo camera images
side by side. Neither mode creates a controller or application publisher, and
neither mode steps physics.

Keys in the MuJoCo window:

* C or 2: calibrated RGB optical point of view
* F or 1: free overview, including camera housing and view frustum
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence
import xml.etree.ElementTree as ET

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = REPO_ROOT / "assets" / "fr3_inspirehand" / "fr3_inspirehand.xml"
CAMERA_NAME = "calibrated_rgb_pov"
SUPPORTED_ROOT_FRAMES = {"world", "base", "fr3_link0"}
ARM_JOINT_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8))
LIVE_WINDOW_NAME = "Real camera | MuJoCo calibrated camera"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Show a CALIBRATION_RESULT camera pose and its RGB point of view in "
            "MuJoCo's passive viewer."
        )
    )
    parser.add_argument(
        "calibration",
        type=Path,
        help="JSON result file, or a ROS log containing a CALIBRATION_RESULT line",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"MJCF scene to decorate (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--start-in-pov",
        action="store_true",
        help="open in the calibrated RGB view instead of the free overview",
    )
    parser.add_argument(
        "--frustum-depth-m",
        type=float,
        default=0.35,
        help="length of the visible camera frustum in the overview (default: 0.35)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="compile and validate the decorated model without opening a window",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "subscribe to live Franka joints and RGB, then show the real and "
            "MuJoCo calibrated-camera images side by side"
        ),
    )
    parser.add_argument(
        "--joint-state-topic",
        default="/joint_states",
        help="Franka JointState topic used by --live (default: /joint_states)",
    )
    parser.add_argument(
        "--image-topic",
        default="/camera/camera/color/image_raw",
        help="real RGB topic used by --live",
    )
    parser.add_argument(
        "--render-width",
        type=int,
        default=640,
        metavar="PIXELS",
        help="width of each side of the live comparison (default: 640)",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=20.0,
        metavar="HZ",
        help="maximum live comparison refresh rate (default: 20)",
    )
    return parser


def load_calibration_result(path: Path) -> dict[str, Any]:
    """Read a pure result JSON file or recover the last result from a ROS log."""
    text = path.expanduser().read_text(encoding="utf-8")
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        marker = "CALIBRATION_RESULT "
        position = text.rfind(marker)
        if position < 0:
            raise ValueError(
                f"{path} is neither JSON nor a log containing {marker.strip()!r}"
            ) from None
        try:
            result, _ = json.JSONDecoder().raw_decode(text[position + len(marker) :])
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid CALIBRATION_RESULT JSON in {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return result


def _transform_matrix(value: Any, field: str) -> np.ndarray:
    if not isinstance(value, dict) or "matrix_4x4" not in value:
        raise ValueError(f"calibration result has no {field}.matrix_4x4")
    matrix = np.asarray(value["matrix_4x4"], dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{field}.matrix_4x4 must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-8):
        raise ValueError(f"{field}.matrix_4x4 is not a homogeneous transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1.0e-5
    ):
        raise ValueError(f"{field}.matrix_4x4 rotation is not orthonormal")
    return matrix


def calibration_poses(
    result: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Return world-to-mount, world-to-optical, K, and image (width, height)."""
    parent_frame = str(result.get("parent_frame", ""))
    if parent_frame not in SUPPORTED_ROOT_FRAMES:
        raise ValueError(
            f"calibration parent frame {parent_frame!r} is not coincident with this "
            f"scene's origin; expected one of {sorted(SUPPORTED_ROOT_FRAMES)}"
        )
    world_to_mount = _transform_matrix(result.get("transform"), "transform")
    if "world_to_camera_optical" not in result:
        raise ValueError(
            "this is a legacy calibration result without world_to_camera_optical; "
            "run calibration again so the RGB POV can use the exact camera-internal TF"
        )
    world_to_optical = _transform_matrix(
        result["world_to_camera_optical"], "world_to_camera_optical"
    )

    intrinsics = result.get("camera_intrinsics")
    if not isinstance(intrinsics, dict):
        raise ValueError("calibration result has no camera_intrinsics object")
    matrix = np.asarray(intrinsics.get("matrix_3x3"), dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("camera_intrinsics.matrix_3x3 must be a finite 3x3 matrix")
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    width = int(intrinsics.get("width", round(2.0 * float(matrix[0, 2]))))
    height = int(intrinsics.get("height", round(2.0 * float(matrix[1, 2]))))
    if width <= 0 or height <= 0:
        raise ValueError("camera image width and height must be positive")
    return world_to_mount, world_to_optical, matrix, (width, height)


def _matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to MuJoCo's normalized wxyz quaternion."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        other_a, other_b = ((1, 2), (2, 0), (0, 1))[axis]
        scale = math.sqrt(
            1.0
            + rotation[axis, axis]
            - rotation[other_a, other_a]
            - rotation[other_b, other_b]
        ) * 2.0
        quaternion = np.empty(4)
        quaternion[0] = (
            rotation[other_b, other_a] - rotation[other_a, other_b]
        ) / scale
        quaternion[axis + 1] = 0.25 * scale
        quaternion[other_a + 1] = (
            rotation[other_a, axis] + rotation[axis, other_a]
        ) / scale
        quaternion[other_b + 1] = (
            rotation[other_b, axis] + rotation[axis, other_b]
        ) / scale
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def _numbers(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _visual_geom(kind: str, **attributes: str) -> ET.Element:
    return ET.Element(
        "geom",
        {
            "type": kind,
            "contype": "0",
            "conaffinity": "0",
            # Keep all calibration-only geometry in one otherwise-unused
            # group so it can be hidden from the camera's own POV.
            "group": "5",
            "density": "0",
            **attributes,
        },
    )


def decorated_scene_xml(
    model_path: Path,
    world_to_mount: np.ndarray,
    world_to_optical: np.ndarray,
    camera_matrix: np.ndarray,
    image_size: tuple[int, int],
    frustum_depth_m: float,
) -> str:
    """Inject a camera housing, RGB optical camera, axes, and view frustum."""
    if not math.isfinite(frustum_depth_m) or frustum_depth_m <= 0.0:
        raise ValueError("--frustum-depth-m must be finite and positive")
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MuJoCo model does not exist: {model_path}")
    root = ET.fromstring(model_path.read_text(encoding="utf-8"))
    for element in root.iter():
        filename = element.get("file")
        if filename:
            asset_path = (model_path.parent / filename).resolve()
            if not asset_path.is_file():
                raise FileNotFoundError(f"MuJoCo asset does not exist: {asset_path}")
            element.set("file", str(asset_path))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{model_path} has no worldbody")

    # The housing is placed at the calibrated camera_link/mount frame.  Its
    # dimensions and axes follow the D4xx convention: +X forward, +Y left,
    # +Z up.  It is deliberately cosmetic and cannot collide with the scene.
    mount = ET.Element(
        "body",
        {
            "name": "calibrated_camera_mount",
            "pos": _numbers(world_to_mount[:3, 3]),
            "quat": _numbers(_matrix_to_quaternion_wxyz(world_to_mount[:3, :3])),
        },
    )
    mount.append(
        _visual_geom(
            "box",
            name="calibrated_camera_housing",
            size="0.0125 0.045 0.0125",
            pos="-0.008 0 0",
            rgba="0.12 0.14 0.16 1",
        )
    )
    mount.append(
        _visual_geom(
            "cylinder",
            name="calibrated_camera_lens",
            size="0.006 0.002",
            pos="0.006 -0.015 0",
            quat="0.707106781187 0 0.707106781187 0",
            rgba="0.10 0.35 0.75 1",
        )
    )
    worldbody.insert(0, mount)

    # OpenCV optical axes are +X right, +Y down, +Z forward.  MuJoCo cameras
    # are +X right, +Y up, -Z forward, so flip optical Y and Z.
    opencv_to_mujoco = np.diag((1.0, -1.0, -1.0))
    world_to_mujoco_camera = world_to_optical[:3, :3] @ opencv_to_mujoco
    optical = ET.Element(
        "body",
        {
            "name": "calibrated_camera_optical",
            "pos": _numbers(world_to_optical[:3, 3]),
            "quat": _numbers(_matrix_to_quaternion_wxyz(world_to_mujoco_camera)),
        },
    )
    width, height = image_size
    fy = float(camera_matrix[1, 1])
    fovy_deg = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
    optical.append(
        ET.Element(
            "camera",
            {"name": CAMERA_NAME, "mode": "fixed", "fovy": f"{fovy_deg:.12g}"},
        )
    )

    # Optical-axis marker: red=right, green=image-up (-OpenCV Y), blue=look.
    axis_length = 0.09
    optical.append(
        _visual_geom(
            "capsule",
            fromto=f"0 0 0 {axis_length} 0 0",
            size="0.002",
            rgba="1 0.1 0.1 1",
        )
    )
    optical.append(
        _visual_geom(
            "capsule",
            fromto=f"0 0 0 0 {axis_length} 0",
            size="0.002",
            rgba="0.1 1 0.1 1",
        )
    )
    optical.append(
        _visual_geom(
            "capsule",
            fromto=f"0 0 0 0 0 {-axis_length}",
            size="0.002",
            rgba="0.1 0.35 1 1",
        )
    )

    half_width = frustum_depth_m * width / (2.0 * float(camera_matrix[0, 0]))
    half_height = frustum_depth_m * height / (2.0 * fy)
    for x in (-half_width, half_width):
        for y in (-half_height, half_height):
            optical.append(
                _visual_geom(
                    "capsule",
                    fromto=_numbers((0.0, 0.0, 0.0, x, y, -frustum_depth_m)),
                    size="0.001",
                    rgba="0.1 0.85 0.95 0.55",
                )
            )
    worldbody.insert(1, optical)
    return ET.tostring(root, encoding="unicode")


def _require_mujoco() -> Any:
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MuJoCo is not importable; run this inside the workspace container"
        ) from exc
    return mujoco


def arm_joint_qpos_addresses(mujoco: Any, model: Any) -> np.ndarray:
    """Return MuJoCo qpos addresses in canonical Franka joint order."""
    addresses = []
    for name in ARM_JOINT_NAMES:
        joint_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        )
        if joint_id < 0:
            raise RuntimeError(f"MuJoCo model has no arm joint named {name!r}")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise RuntimeError(f"MuJoCo arm joint {name!r} is not a hinge")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(addresses, dtype=int)


def ordered_arm_joint_positions(
    names: Sequence[str], positions: Sequence[float]
) -> np.ndarray:
    """Extract the seven finite Franka angles from a ROS JointState payload."""
    if len(names) != len(positions):
        raise ValueError(
            f"JointState has {len(names)} names but {len(positions)} positions"
        )
    if len(set(names)) != len(names):
        raise ValueError("JointState contains duplicate joint names")
    by_name = {str(name): float(position) for name, position in zip(names, positions)}
    missing = [name for name in ARM_JOINT_NAMES if name not in by_name]
    if missing:
        raise ValueError("JointState is missing " + ", ".join(missing))
    ordered = np.asarray([by_name[name] for name in ARM_JOINT_NAMES], dtype=float)
    if not np.all(np.isfinite(ordered)):
        raise ValueError("JointState contains a non-finite Franka joint angle")
    return ordered


def _fit_rgb_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Aspect-fit an RGB image into a fixed dark canvas."""
    import cv2

    image = np.asarray(frame, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("RGB frame must have shape (height, width, 3)")
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, int(round(image.shape[1] * scale)))
    resized_height = max(1, int(round(image.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=interpolation
    )
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def comparison_frame(
    real_rgb: np.ndarray | None,
    simulated_rgb: np.ndarray,
    panel_width: int,
    panel_height: int,
    real_status: str,
    simulated_status: str,
) -> np.ndarray:
    """Build the labelled RGB side-by-side live comparison image."""
    import cv2

    if real_rgb is None:
        real_panel = np.full((panel_height, panel_width, 3), 20, dtype=np.uint8)
    else:
        real_panel = _fit_rgb_frame(real_rgb, panel_width, panel_height)
    simulated_panel = _fit_rgb_frame(
        simulated_rgb, panel_width, panel_height
    )
    image = np.concatenate((real_panel, simulated_panel), axis=1)
    title_height = 54
    title_bar = np.full((title_height, 2 * panel_width, 3), 32, dtype=np.uint8)
    image = np.concatenate((title_bar, image), axis=0)
    labels = (
        ("REAL CAMERA", real_status, 12),
        ("MUJOCO CAMERA", simulated_status, panel_width + 12),
    )
    for title, status, left in labels:
        cv2.putText(
            image,
            title,
            (left, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            status,
            (left, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (145, 205, 235),
            1,
            cv2.LINE_AA,
        )
    cv2.line(
        image,
        (panel_width, 0),
        (panel_width, image.shape[0]),
        (85, 85, 85),
        1,
    )
    return image


def run_live_comparison(
    mujoco: Any,
    model: Any,
    data: Any,
    camera_id: int,
    image_size: tuple[int, int],
    joint_state_topic: str,
    image_topic: str,
    render_width: int,
    max_fps: float,
) -> None:
    """Drive MuJoCo kinematics from ROS and compare its POV with live RGB."""
    if render_width <= 0:
        raise ValueError("--render-width must be greater than zero")
    if not math.isfinite(max_fps) or max_fps <= 0.0:
        raise ValueError("--max-fps must be finite and greater than zero")

    try:
        import cv2
        from cv_bridge import CvBridge
        import rclpy
        from rclpy.executors import ExternalShutdownException
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, JointState
    except ImportError as exc:
        raise RuntimeError(
            "--live requires ROS 2, sensor_msgs, cv_bridge, and OpenCV; source "
            "the ROS installation and this workspace"
        ) from exc

    render_height = max(1, int(round(render_width * image_size[1] / image_size[0])))
    if render_width > int(model.vis.global_.offwidth) or render_height > int(
        model.vis.global_.offheight
    ):
        raise ValueError(
            f"requested live render {render_width}x{render_height} exceeds the "
            f"model offscreen buffer {model.vis.global_.offwidth}x"
            f"{model.vis.global_.offheight}; lower --render-width"
        )

    addresses = arm_joint_qpos_addresses(mujoco, model)
    bridge = CvBridge()
    latest: dict[str, Any] = {
        "image": None,
        "image_time": None,
        "joints": None,
        "joint_time": None,
    }
    warned_joint_message = {"value": False}

    rclpy.init(args=[])
    node = rclpy.create_node("calibrated_camera_mujoco_live_view")

    def on_image(message: Any) -> None:
        try:
            latest["image"] = np.asarray(
                bridge.imgmsg_to_cv2(message, desired_encoding="rgb8"),
                dtype=np.uint8,
            ).copy()
            latest["image_time"] = time.monotonic()
        except Exception as exc:  # cv_bridge exceptions vary by ROS distro
            node.get_logger().error(f"Could not convert RGB frame: {exc}")

    def on_joint_state(message: Any) -> None:
        try:
            latest["joints"] = ordered_arm_joint_positions(
                message.name, message.position
            )
            latest["joint_time"] = time.monotonic()
        except ValueError as exc:
            if not warned_joint_message["value"]:
                node.get_logger().warning(
                    f"Ignoring incompatible {joint_state_topic}: {exc}"
                )
                warned_joint_message["value"] = True

    input_subscriptions = (
        node.create_subscription(
            Image, image_topic, on_image, qos_profile_sensor_data
        ),
        node.create_subscription(
            JointState, joint_state_topic, on_joint_state, qos_profile_sensor_data
        ),
    )

    render_option = mujoco.MjvOption()
    render_option.geomgroup[5] = 0
    try:
        renderer = mujoco.Renderer(
            model, height=render_height, width=render_width
        )
    except Exception as exc:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        raise RuntimeError(
            "MuJoCo could not create an RGB renderer; on a headless machine set "
            "MUJOCO_GL=egl"
        ) from exc

    print(f"Live joints: {joint_state_topic}")
    print(f"Real RGB:    {image_topic}")
    print("Preview:     MuJoCo calibrated RGB camera (read-only kinematics)")
    print("Press Q or Esc in the comparison window to close.")
    frame_period = 1.0 / max_fps
    next_frame = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=min(0.01, frame_period))
            now = time.monotonic()
            if now < next_frame:
                continue
            next_frame = now + frame_period

            if latest["joints"] is not None:
                data.qpos[addresses] = latest["joints"]
                mujoco.mj_forward(model, data)
            renderer.update_scene(
                data, camera=camera_id, scene_option=render_option
            )
            simulated_rgb = renderer.render().copy()

            image_age = (
                None
                if latest["image_time"] is None
                else now - float(latest["image_time"])
            )
            joint_age = (
                None
                if latest["joint_time"] is None
                else now - float(latest["joint_time"])
            )
            real_status = (
                f"waiting for {image_topic}"
                if image_age is None
                else f"live RGB | age {image_age:.2f} s"
            )
            simulated_status = (
                f"waiting for {joint_state_topic}"
                if joint_age is None
                else "q rad ["
                + ", ".join(f"{value:+.2f}" for value in latest["joints"])
                + f"] | age {joint_age:.2f} s"
            )
            combined = comparison_frame(
                latest["image"],
                simulated_rgb,
                render_width,
                render_height,
                real_status,
                simulated_status,
            )
            try:
                cv2.imshow(
                    LIVE_WINDOW_NAME, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                )
            except cv2.error as exc:
                raise RuntimeError(
                    "OpenCV could not open the live comparison window; check "
                    "DISPLAY/container GUI forwarding"
                ) from exc
            if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                break
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        renderer.close()
        try:
            cv2.destroyWindow(LIVE_WINDOW_NAME)
        except cv2.error:
            # No native window exists when the GUI backend failed before the
            # first imshow; preserve that original, more useful exception.
            pass
        for subscription in input_subscriptions:
            node.destroy_subscription(subscription)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _print_pose(label: str, transform: np.ndarray) -> None:
    quaternion = _matrix_to_quaternion_wxyz(transform[:3, :3])
    print(
        f"{label}: xyz={_numbers(transform[:3, 3])} m, "
        f"quaternion_wxyz={_numbers(quaternion)}"
    )


def run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.headless and args.live:
        raise ValueError("--headless and --live cannot be used together")
    result = load_calibration_result(args.calibration)
    world_to_mount, world_to_optical, camera_matrix, image_size = calibration_poses(
        result
    )
    xml = decorated_scene_xml(
        args.model,
        world_to_mount,
        world_to_optical,
        camera_matrix,
        image_size,
        args.frustum_depth_m,
    )
    mujoco = _require_mujoco()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    key_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "start"))
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    camera_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    )
    if camera_id < 0:
        raise RuntimeError(f"decorated MuJoCo model has no camera named {CAMERA_NAME!r}")

    _print_pose("camera mount", world_to_mount)
    _print_pose("RGB optical frame", world_to_optical)
    print(
        f"RGB image={image_size[0]}x{image_size[1]}, "
        f"fy={camera_matrix[1, 1]:.3f} px"
    )
    print(
        f"For matching horizontal coverage, keep the viewer window at the RGB "
        f"aspect ratio ({image_size[0]}:{image_size[1]})."
    )
    if args.headless:
        print("Decorated MuJoCo scene compiled successfully (headless validation).")
        return 0

    if args.live:
        run_live_comparison(
            mujoco,
            model,
            data,
            camera_id,
            image_size,
            args.joint_state_topic,
            args.image_topic,
            args.render_width,
            args.max_fps,
        )
        return 0

    import mujoco.viewer

    requested_mode = {"pov": bool(args.start_in_pov), "changed": True}

    def on_key(keycode: int) -> None:
        if keycode in (ord("C"), ord("2")):
            requested_mode.update(pov=True, changed=True)
        elif keycode in (ord("F"), ord("1")):
            requested_mode.update(pov=False, changed=True)

    print(
        "MuJoCo controls: C/2 = calibrated RGB POV, "
        "F/1 = free overview, Esc = close"
    )
    with mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=on_key,
    ) as viewer:
        free_camera = {
            "type": viewer.cam.type,
            "fixedcamid": viewer.cam.fixedcamid,
            "lookat": np.asarray(viewer.cam.lookat).copy(),
            "distance": viewer.cam.distance,
            "azimuth": viewer.cam.azimuth,
            "elevation": viewer.cam.elevation,
            "geomgroup": np.asarray(viewer.opt.geomgroup).copy(),
        }
        while viewer.is_running():
            with viewer.lock():
                if requested_mode["changed"]:
                    if requested_mode["pov"]:
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        viewer.cam.fixedcamid = camera_id
                        viewer.opt.geomgroup[5] = 0
                    else:
                        viewer.cam.type = free_camera["type"]
                        viewer.cam.fixedcamid = free_camera["fixedcamid"]
                        viewer.cam.lookat[:] = free_camera["lookat"]
                        viewer.cam.distance = free_camera["distance"]
                        viewer.cam.azimuth = free_camera["azimuth"]
                        viewer.cam.elevation = free_camera["elevation"]
                        viewer.opt.geomgroup[:] = free_camera["geomgroup"]
                    requested_mode["changed"] = False
            viewer.sync()
            time.sleep(1.0 / 60.0)
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
