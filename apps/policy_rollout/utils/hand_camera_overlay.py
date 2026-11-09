#!/usr/bin/env python3
"""Draw the simulated hand over the live RealSense image.

    python3 apps/policy_rollout/utils/hand_camera_overlay.py          # one frame to debug/
    python3 apps/policy_rollout/utils/hand_camera_overlay.py --live   # window; s saves, q quits

Read-only: it subscribes to the camera and the joint states and commands nothing.

The MJCF policy scene is posed from the robot's measured joints and projected
through the live camera intrinsics and the calibrated camera pose. Where the
drawing and the photo disagree, one of three things is off, and the drawing
separates them:

- the arm outline (white) is off too: the camera calibration, whose own fit
  residual is 9.7 mm / 1.0 deg, so a few pixels here are expected;
- the arm lines up and the fingers do not: the hand model or the joint readings;
- the two thumbs (policy, driver) bracket the real one: the driver's thumb-yaw
  overlay. "policy" is the logical angle the student observes, "driver" is the
  angle the driver reports with the overlay still in it.

The nut and bolt (yellow) are where training puts them, not a measurement.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = APP_ROOT.parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from policy_rollout import forge_osc as fo  # noqa: E402

SCENE = WORKSPACE_ROOT / "src/inspire_franka_sim/mjcf/inspire_franka_policy_scene_windback45_tilt30.xml"
ARM_STATE_TOPIC = "/joint_states"
HAND_STATE_TOPIC = "/inspire_hand/joint_states"

# Geom groups in the policy scene's visual meshes.
HAND_VISUAL_GROUP = 1
ARM_VISUAL_GROUP = 2

ZOOM_CROP_PX = 120
ZOOM_FACTOR = 4
SCALE_BAR_M = 0.010

# BGR.
WHITE = (255, 255, 255)
GREEN = (60, 220, 60)
CYAN = (240, 220, 40)
MAGENTA = (240, 60, 240)
YELLOW = (40, 220, 240)
GREY = (170, 170, 170)


def project(points, rotation, translation, camera_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Base-frame points to pixels; ``(uv, depth)`` with depth along the optical axis."""

    camera = (np.asarray(points, dtype=float) - np.asarray(translation, dtype=float)) @ rotation
    depth = camera[:, 2]
    k = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k[0, 0] * camera[:, 0] / depth + k[0, 2]
        v = k[1, 1] * camera[:, 1] / depth + k[1, 2]
    return np.column_stack([u, v]), depth


def policy_hand_joints(hand_positions: dict[str, float], thumb_overlay: bool) -> dict[str, float]:
    """The logical hand the student observes, on the scene's joint names."""

    from policy_rollout.hardware import (
        HAND_JOINTS,
        TRAINING_INACTIVE_FINGERS,
        physical_hand_state_to_policy,
    )
    from utils.make_ros_sim_scene import hand as hand_tables

    leaders, _ = physical_hand_state_to_policy(
        [hand_positions[name] for name in HAND_JOINTS],
        np.zeros(len(HAND_JOINTS)),
        thumb_overlay=thumb_overlay,
    )
    targets = dict(zip(fo.PINCH_JOINTS, leaders))
    targets.update(TRAINING_INACTIVE_FINGERS)
    expanded = fo.expand_hand_mimic(targets)
    return {
        scene_name: float(expanded[training_name])
        for scene_name, training_name in hand_tables.ISAAC_JOINT_NAMES.items()
    }


@dataclass
class TipReading:
    label: str
    pixel: np.ndarray
    model_depth_m: float
    measured_depth_m: float


class SceneProjector:
    """The MJCF scene, posed from joint angles and projected into the camera."""

    def __init__(self, scene_path: Path, rotation, translation) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("policy_home").id)
        self.rotation = np.asarray(rotation, dtype=float)
        self.translation = np.asarray(translation, dtype=float)
        self.joint_names = {self.model.joint(i).name for i in range(self.model.njnt)}

    def set_joints(self, positions: dict[str, float]) -> None:
        for name, value in positions.items():
            if name in self.joint_names:
                self.data.qpos[self.model.joint(name).qposadr[0]] = float(value)
        self._mujoco.mj_kinematics(self.model, self.data)

    def body_position(self, name: str) -> np.ndarray:
        return self.data.xpos[self.model.body(name).id].copy()

    def geom_hull(self, geom: int, camera_matrix) -> np.ndarray | None:
        """A mesh geom's convex outline in pixels, or None when behind the camera."""

        import cv2

        model = self.model
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
        world = self.data.geom_xpos[geom] + vertices @ self.data.geom_xmat[geom].reshape(3, 3).T
        uv, depth = project(world, self.rotation, self.translation, camera_matrix)
        if not (depth > 0.0).all():
            return None
        return cv2.convexHull(np.round(uv).astype(np.int32))

    def geoms(self, group: int, body_prefixes: tuple[str, ...]) -> list[int]:
        import mujoco

        model = self.model
        return [
            geom
            for geom in range(model.ngeom)
            if model.geom_group[geom] == group
            and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
            and model.body(model.geom_bodyid[geom]).name.startswith(body_prefixes)
        ]


def measured_depth(depth_m: np.ndarray | None, pixel: np.ndarray, radius: int = 3) -> float:
    """Median valid depth around ``pixel``; NaN when there is none."""

    if depth_m is None or not np.isfinite(pixel).all():
        return float("nan")
    u, v = int(round(pixel[0])), int(round(pixel[1]))
    height, width = depth_m.shape
    if not (0 <= u < width and 0 <= v < height):
        return float("nan")
    window = depth_m[max(0, v - radius) : v + radius + 1, max(0, u - radius) : u + radius + 1]
    valid = window[window > 0.0]
    return float(np.median(valid)) if valid.size else float("nan")


def _outline(canvas, hull, colour, thickness=1) -> None:
    import cv2

    cv2.polylines(canvas, [hull], True, colour, thickness, cv2.LINE_AA)


def _cross(canvas, pixel, colour, label) -> None:
    import cv2

    if not np.isfinite(pixel).all():
        return
    point = (int(round(pixel[0])), int(round(pixel[1])))
    cv2.drawMarker(canvas, point, colour, cv2.MARKER_TILTED_CROSS, 9, 1, cv2.LINE_AA)
    cv2.putText(canvas, label, (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)


def compose(projector, image_bgr, depth_m, camera_matrix, arm, hand_reported, thumb_overlay):
    """The annotated frame and the tip readings for one set of inputs."""

    import cv2

    canvas = image_bgr.copy()
    model = projector.model
    outlines: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    marks: list[tuple[np.ndarray, tuple[int, int, int], str]] = []

    def hulls(geoms):
        return [hull for hull in (projector.geom_hull(g, camera_matrix) for g in geoms) if hull is not None]

    def tip(body, label):
        uv, depth = project(projector.body_position(body)[None], projector.rotation, projector.translation, camera_matrix)
        return TipReading(label, uv[0], float(depth[0]), measured_depth(depth_m, uv[0]))

    # The driver's own reading: only its thumb differs from the policy's hand.
    projector.set_joints({**arm, **hand_reported})
    outlines += [(hull, MAGENTA) for hull in hulls(projector.geoms(HAND_VISUAL_GROUP, ("thumb_",)))]
    driver_thumb = tip("thumb_tip", "thumb driver")

    # The hand the student believes it has.
    projector.set_joints({**arm, **policy_hand_joints(hand_reported, thumb_overlay)})
    for group, prefixes, colour in (
        (ARM_VISUAL_GROUP, ("fr3_link",), WHITE),
        (HAND_VISUAL_GROUP, ("hand_base", "hand_mount", "middle_", "ring_", "pinky_"), GREY),
        (HAND_VISUAL_GROUP, ("thumb_",), GREEN),
        (HAND_VISUAL_GROUP, ("index_",), CYAN),
    ):
        outlines += [(hull, colour) for hull in hulls(projector.geoms(group, prefixes))]
    for name in ("m24_nut_geom", "m24_bolt_geom"):
        hull = projector.geom_hull(model.geom(name).id, camera_matrix)
        if hull is not None:
            outlines.append((hull, YELLOW))

    thumb, index = tip("thumb_tip", "thumb policy"), tip("index_tip", "index")
    readings = [driver_thumb, thumb, index]
    marks += [(driver_thumb.pixel, MAGENTA, ""), (thumb.pixel, GREEN, "T"), (index.pixel, CYAN, "I")]

    zoom = _zoom_panel(canvas, thumb, index, camera_matrix, outlines, marks)
    for hull, colour in outlines:
        _outline(canvas, hull, colour)
    for pixel, colour, label in marks:
        _cross(canvas, pixel, colour, label)

    legend = [
        ("white arm  grey palm", WHITE),
        ("green thumb: policy angle", GREEN),
        ("magenta thumb: driver angle", MAGENTA),
        ("cyan index", CYAN),
        ("yellow nut/bolt: training spot", YELLOW),
    ]
    for row, (text, colour) in enumerate(legend):
        cv2.putText(canvas, text, (8, 16 + 15 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (8, 16 + 15 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

    return np.hstack([canvas, zoom[: canvas.shape[0]]]), readings


def _zoom_panel(image_bgr, thumb: TipReading, index: TipReading, camera_matrix, outlines, marks):
    """The plain photo magnified around the fingertips, with the model as thin outlines."""

    import cv2

    height, width = image_bgr.shape[:2]
    centre = np.nan_to_num(0.5 * (thumb.pixel + index.pixel), nan=0.0)
    half = ZOOM_CROP_PX // 2
    origin = np.array(
        [np.clip(centre[0] - half, 0, width - ZOOM_CROP_PX), np.clip(centre[1] - half, 0, height - ZOOM_CROP_PX)],
        dtype=np.int32,
    )
    crop = image_bgr[origin[1] : origin[1] + ZOOM_CROP_PX, origin[0] : origin[0] + ZOOM_CROP_PX]
    zoom = cv2.resize(crop, None, fx=ZOOM_FACTOR, fy=ZOOM_FACTOR, interpolation=cv2.INTER_CUBIC)

    def to_zoom(points):
        return np.round((np.asarray(points, dtype=float) - origin + 0.5) * ZOOM_FACTOR).astype(np.int32)

    for hull, colour in outlines:
        _outline(zoom, to_zoom(hull), colour)
    for pixel, colour, label in marks:
        _cross(zoom, (pixel - origin + 0.5) * ZOOM_FACTOR, colour, label)

    depth = 0.5 * (thumb.model_depth_m + index.model_depth_m)
    if np.isfinite(depth) and depth > 0.0:
        fx = float(np.asarray(camera_matrix).reshape(3, 3)[0, 0])
        bar = int(round(SCALE_BAR_M * fx / depth * ZOOM_FACTOR))
        y = zoom.shape[0] - 14
        cv2.line(zoom, (10, y), (10 + bar, y), WHITE, 2)
        cv2.putText(zoom, "10 mm", (10, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    return zoom


def _image_array(message) -> np.ndarray:
    channels = {"rgb8": 3, "bgr8": 3, "16UC1": 1, "mono16": 1}[message.encoding]
    dtype = np.uint16 if channels == 1 else np.uint8
    data = np.frombuffer(bytes(message.data), dtype=dtype)
    return data.reshape(message.height, message.width, -1) if channels == 3 else data.reshape(message.height, message.width)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="open a window instead of saving one frame")
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--out", type=Path, default=None, help="default: debug/hand_overlay_<time>.png")
    parser.add_argument(
        "--no-thumb-overlay",
        action="store_true",
        help="the hand driver runs without its thumb-yaw overlay (thumb_abduction_overlay false)",
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args(argv)

    import cv2
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, JointState

    from utils.camera_calibration import load_camera_calibration

    calibration = load_camera_calibration()
    pose = calibration.measured_world_pose or calibration.training_world_pose
    projector = SceneProjector(args.scene, fo.matrix_from_quat(np.asarray(pose.rotation_wxyz, float)), pose.translation_m)

    rclpy.init()
    node = rclpy.create_node("hand_camera_overlay")
    latest: dict = {}

    def store(key):
        return lambda message: latest.__setitem__(key, message)

    node.create_subscription(Image, calibration.color_topic, store("color"), qos_profile_sensor_data)
    node.create_subscription(Image, calibration.depth_topic, store("depth"), qos_profile_sensor_data)
    node.create_subscription(CameraInfo, calibration.camera_info_topic, store("info"), qos_profile_sensor_data)
    node.create_subscription(JointState, ARM_STATE_TOPIC, store("arm"), qos_profile_sensor_data)
    node.create_subscription(JointState, HAND_STATE_TOPIC, store("hand"), qos_profile_sensor_data)

    required = ("color", "info", "arm", "hand")

    def spin_until_ready() -> None:
        deadline = node.get_clock().now().nanoseconds + int(args.timeout * 1e9)
        while not all(key in latest for key in required):
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.get_clock().now().nanoseconds > deadline:
                missing = [key for key in required if key not in latest]
                raise TimeoutError(f"no message on: {missing}")

    def render():
        image = _image_array(latest["color"])
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if latest["color"].encoding == "rgb8" else image.copy()
        depth = _image_array(latest["depth"]).astype(float) * 1.0e-3 if "depth" in latest else None
        arm = {n: p for n, p in zip(latest["arm"].name, latest["arm"].position) if n.startswith("fr3_joint")}
        hand = dict(zip(latest["hand"].name, latest["hand"].position))
        if len(arm) != 7:
            raise RuntimeError(f"{ARM_STATE_TOPIC} carried {len(arm)} arm joints, not 7")
        return compose(projector, bgr, depth, latest["info"].k, arm, hand, not args.no_thumb_overlay)

    def save(frame, readings) -> Path:
        path = args.out or WORKSPACE_ROOT / "debug" / f"hand_overlay_{datetime.now():%Y%m%d_%H%M%S}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)
        print(f"saved {path}")
        for r in readings:
            print(
                f"  {r.label:13s} pixel ({r.pixel[0]:6.1f}, {r.pixel[1]:6.1f})  model depth "
                f"{1.0e3 * r.model_depth_m:6.1f} mm  camera depth there {1.0e3 * r.measured_depth_m:6.1f} mm"
            )
        return path

    try:
        spin_until_ready()
        if not args.live:
            # Let the arm, hand and image all come from the current moment.
            for _ in range(20):
                rclpy.spin_once(node, timeout_sec=0.02)
            save(*render())
            return 0
        window = "hand vs simulation (s: save, q: quit)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        last = None
        frame, readings = render()
        while True:
            rclpy.spin_once(node, timeout_sec=0.01)
            if latest["color"] is not last:
                last = latest["color"]
                frame, readings = render()
                cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord("s"):
                save(frame, readings)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
