#!/usr/bin/env python3
"""Inspect a calibrated RealSense pose in MuJoCo's passive viewer.

The input may be either the JSON object printed after ``CALIBRATION_RESULT``
or an entire captured ROS log containing that line.  The viewer is strictly
kinematic: it creates no ROS node, controller, publisher, or physics loop.

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
DEFAULT_CAMERA_MESH = REPO_ROOT / "assets" / "camera" / "mesh" / "d415.stl"
CAMERA_NAME = "calibrated_rgb_pov"
CAMERA_MESH_NAME = "calibrated_d415_mesh"
SUPPORTED_ROOT_FRAMES = {"world", "base", "fr3_link0"}
FRAME_AXIS_LENGTH_M = 0.12
FRAME_AXIS_RADIUS_M = 0.0025


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


def _append_frame_axes(body: ET.Element, name: str) -> None:
    """Draw an REP-103 axis triad: +X red, +Y green, and +Z blue."""
    axes = (
        ("x", (FRAME_AXIS_LENGTH_M, 0.0, 0.0), "1 0.1 0.1 1"),
        ("y", (0.0, FRAME_AXIS_LENGTH_M, 0.0), "0.1 1 0.1 1"),
        ("z", (0.0, 0.0, FRAME_AXIS_LENGTH_M), "0.1 0.35 1 1"),
    )
    for axis, endpoint, rgba in axes:
        body.append(
            _visual_geom(
                "capsule",
                name=f"{name}_{axis}_axis",
                fromto=_numbers((0.0, 0.0, 0.0, *endpoint)),
                size=f"{FRAME_AXIS_RADIUS_M:.12g}",
                rgba=rgba,
            )
        )


def _tf_connector(
    name: str,
    start: Sequence[float],
    end: Sequence[float],
    *,
    rgba: str,
    radius_m: float,
) -> ET.Element:
    """Draw the translation component between two TF frame origins."""
    return _visual_geom(
        "capsule",
        name=name,
        fromto=_numbers((*start, *end)),
        size=f"{radius_m:.12g}",
        rgba=rgba,
    )


def _free_view_geomgroup(current: Sequence[int]) -> np.ndarray:
    """Return viewer geometry groups with camera-visual group 5 enabled."""
    geomgroup = np.asarray(current).copy()
    geomgroup[5] = 1
    return geomgroup


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
    camera_mesh = DEFAULT_CAMERA_MESH.resolve()
    if not camera_mesh.is_file():
        raise FileNotFoundError(f"D415 camera mesh does not exist: {camera_mesh}")
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
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        worldbody_index = list(root).index(worldbody)
        root.insert(worldbody_index, asset)
    asset.append(
        ET.Element(
            "mesh",
            {"name": CAMERA_MESH_NAME, "file": str(camera_mesh)},
        )
    )

    # The housing is placed at the calibrated camera_link/mount frame. The
    # mesh transform matches realsense2_description's official D415 URDF:
    # xyz="0.00987 -0.020 0", rpy="pi/2 0 pi/2". It is deliberately cosmetic
    # and cannot collide with the scene.
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
            "mesh",
            name="calibrated_camera_housing",
            mesh=CAMERA_MESH_NAME,
            pos="0.00987 -0.020 0",
            quat="0.5 0.5 0.5 0.5",
            rgba="0.58 0.61 0.64 1",
        )
    )
    _append_frame_axes(mount, "calibrated_camera_link")
    worldbody.insert(0, mount)

    # Keep the body itself in the exact ROS/OpenCV optical TF: +X right,
    # +Y down, +Z forward. The child MuJoCo camera is rotated pi around X
    # because MuJoCo cameras look down local -Z with local +Y upward.
    optical = ET.Element(
        "body",
        {
            "name": "calibrated_camera_optical",
            "pos": _numbers(world_to_optical[:3, 3]),
            "quat": _numbers(_matrix_to_quaternion_wxyz(world_to_optical[:3, :3])),
        },
    )
    width, height = image_size
    fy = float(camera_matrix[1, 1])
    fovy_deg = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
    optical.append(
        ET.Element(
            "camera",
            {
                "name": CAMERA_NAME,
                "mode": "fixed",
                "quat": "0 1 0 0",
                "fovy": f"{fovy_deg:.12g}",
            },
        )
    )

    _append_frame_axes(optical, "calibrated_camera_optical")

    half_width = frustum_depth_m * width / (2.0 * float(camera_matrix[0, 0]))
    half_height = frustum_depth_m * height / (2.0 * fy)
    for x in (-half_width, half_width):
        for y in (-half_height, half_height):
            optical.append(
                _visual_geom(
                    "capsule",
                    fromto=_numbers((0.0, 0.0, 0.0, x, y, frustum_depth_m)),
                    size="0.001",
                    rgba="0.1 0.85 0.95 0.55",
                )
            )
    worldbody.insert(1, optical)
    worldbody.append(
        _tf_connector(
            "calibrated_camera_parent_to_link_tf",
            (0.0, 0.0, 0.0),
            world_to_mount[:3, 3],
            rgba="1 0.75 0.1 0.35",
            radius_m=0.0012,
        )
    )
    worldbody.append(
        _tf_connector(
            "calibrated_camera_link_to_optical_tf",
            world_to_mount[:3, 3],
            world_to_optical[:3, 3],
            rgba="1 0.45 0.05 0.9",
            radius_m=0.0018,
        )
    )
    return ET.tostring(root, encoding="unicode")


def _require_mujoco() -> Any:
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MuJoCo is not importable; run this inside the workspace container"
        ) from exc
    return mujoco


def _print_pose(label: str, transform: np.ndarray) -> None:
    quaternion = _matrix_to_quaternion_wxyz(transform[:3, :3])
    print(
        f"{label}: xyz={_numbers(transform[:3, 3])} m, "
        f"quaternion_wxyz={_numbers(quaternion)}"
    )


def run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
        # MuJoCo hides geom groups 3-5 by default. All calibrated-camera
        # visuals live in group 5, so explicitly show that group in the free
        # scene and hide it only while looking through the camera itself.
        free_geomgroup = _free_view_geomgroup(viewer.opt.geomgroup)
        free_camera = {
            "type": viewer.cam.type,
            "fixedcamid": viewer.cam.fixedcamid,
            "lookat": np.asarray(viewer.cam.lookat).copy(),
            "distance": viewer.cam.distance,
            "azimuth": viewer.cam.azimuth,
            "elevation": viewer.cam.elevation,
            "geomgroup": free_geomgroup,
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
