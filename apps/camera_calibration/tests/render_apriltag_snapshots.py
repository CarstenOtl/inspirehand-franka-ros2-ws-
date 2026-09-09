#!/usr/bin/env python3
"""Regenerate the documented AprilTag measurement snapshots from MuJoCo."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = REPO_ROOT / "assets" / "fr3_inspirehand"
SCENE = ASSET_DIR / "fr3_inspirehand.xml"
WIDTH, HEIGHT, RENDER_WIDTH = 980, 640, 640
PANEL_X = RENDER_WIDTH
PANEL_BG = (249, 248, 244)
INK = (31, 31, 31)
MUTED = (78, 78, 78)
CYAN = (20, 203, 232)
MAGENTA = (242, 29, 130)
YELLOW = (255, 202, 0)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def _camera_for_direction(
    mujoco, target: np.ndarray, outward: np.ndarray, distance: float
):
    direction = np.asarray(outward, dtype=float)
    direction /= np.linalg.norm(direction)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = target
    camera.distance = distance
    camera.azimuth = math.degrees(math.atan2(-direction[1], -direction[0]))
    camera.elevation = -math.degrees(math.asin(direction[2]))
    return camera


def _project(mujoco, renderer, point: np.ndarray) -> tuple[int, int]:
    head = np.zeros(3)
    forward = np.zeros(3)
    up = np.zeros(3)
    mujoco.mjv_cameraInModel(head, forward, up, renderer.scene)
    right = np.cross(forward, up)
    delta = np.asarray(point) - head
    depth = float(np.dot(delta, forward))
    camera = renderer.scene.camera[0]
    focal = (HEIGHT / 2.0) * camera.frustum_near / camera.frustum_top
    return (
        int(round(RENDER_WIDTH / 2.0 + focal * np.dot(delta, right) / depth)),
        int(round(HEIGHT / 2.0 - focal * np.dot(delta, up) / depth)),
    )


def _line(draw: ImageDraw.ImageDraw, xy, fill, width=3) -> None:
    draw.line(xy, fill=fill, width=width, joint="curve")


def _panel_base(
    title: str, subtitle: str
) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), PANEL_BG)
    draw = ImageDraw.Draw(image)
    draw.text((675, 28), title, font=_font(28, bold=True), fill=INK)
    draw.text((675, 69), subtitle, font=_font(19), fill=MUTED)
    return image, draw


def _measurement_panel() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image, draw = _panel_base("AprilTag placement", "Dorsal view — current mount")
    entries = (
        (MAGENTA, "20.0 mm", "left rim → corner"),
        (CYAN, "72.0 mm", "flange → corner"),
        (YELLOW, "First black pixel", None),
    )
    y = 116
    for color, label, note in entries:
        draw.ellipse((675, y - 5, 694, y + 14), fill=color)
        draw.text((705, y - 8), label, font=_font(20, bold=note is not None), fill=INK)
        if note:
            draw.text((705, y + 19), note, font=_font(15), fill=MUTED)
            y += 68
        else:
            y += 49
    _line(draw, ((675, 294), (950, 294)), (190, 190, 185), 2)
    draw.text((675, 313), "Palm-frame coordinates", font=_font(21, bold=True), fill=INK)
    for index, text in enumerate((
        "rim y     −40.546 mm",
        "corner y  −20.546 mm",
        "corner z    72.000 mm",
    )):
        draw.text((675, 348 + index * 29), text, font=_font(18), fill=INK)
    draw.text((675, 448), "Current flange mount", font=_font(21, bold=True), fill=INK)
    draw.text((675, 484), "adapter   Ø76 × 10 mm", font=_font(18), fill=INK)
    draw.text((675, 515), "palm z     +10 mm", font=_font(18), fill=INK)
    draw.text((675, 546), "yaw         180°", font=_font(18), fill=INK)
    draw.text((675, 588), "90° from the legacy installation", font=_font(15), fill=MUTED)
    return image, draw


def _visibility_panel() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image, draw = _panel_base("Visibility check", "Oblique dorsal — 180° mount")
    _line(draw, ((675, 112), (950, 112)), (190, 190, 185), 2)
    draw.text((675, 144), "Surface-following sheet", font=_font(21, bold=True), fill=INK)
    lines = (
        "Every grid vertex is sampled",
        "from the official palm STL.",
        "",
        "The sheet remains outside the",
        "shell at every point:",
    )
    for index, text in enumerate(lines):
        draw.text((675, 182 + index * 29), text, font=_font(18), fill=INK)
    draw.text((675, 346), "Back face", font=_font(18), fill=MUTED)
    draw.text((820, 346), "+0.05 mm", font=_font(18, bold=True), fill=(13, 116, 143))
    draw.text((675, 386), "Visible face", font=_font(18), fill=MUTED)
    draw.text((820, 386), "+0.25 mm", font=_font(18, bold=True), fill=(13, 116, 143))
    _line(draw, ((675, 433), (950, 433)), (190, 190, 185), 2)
    draw.multiline_text(
        (675, 465),
        "No part of the AprilTag is inside\nthe hand mesh, and the black\n10 mm adapter is now included.",
        font=_font(18),
        fill=INK,
        spacing=8,
    )
    draw.text((675, 596), "Rendered from current MJCF", font=_font(15), fill=MUTED)
    return image, draw


def _render_snapshot(mujoco, model, data, *, oblique: bool) -> tuple[Image.Image, object]:
    tag_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "apriltag_0")
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_base_link")
    tag_rotation = data.xmat[tag_id].reshape(3, 3)
    palm_rotation = data.xmat[palm_id].reshape(3, 3)
    target = data.xpos[tag_id] + palm_rotation @ np.array((0.0, 0.0, -0.010))
    local_direction = np.array((0.75, 0.20, 1.0)) if oblique else np.array((0.12, -0.08, 1.0))
    outward = tag_rotation @ local_direction
    camera = _camera_for_direction(mujoco, target, outward, 0.31 if oblique else 0.34)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=RENDER_WIDTH)
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render().copy()
    return Image.fromarray(rgb), renderer


def generate() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "start")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    measurement, draw = _measurement_panel()
    render, renderer = _render_snapshot(mujoco, model, data, oblique=False)
    measurement.paste(render, (0, 0))
    draw = ImageDraw.Draw(measurement)
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_base_link")
    palm_position = data.xpos[palm_id]
    palm_rotation = data.xmat[palm_id].reshape(3, 3)
    local_points = {
        "corner": (-0.01673933, -0.020546, 0.072),
        "rim": (-0.01673933, -0.040546, 0.072),
        "flange": (-0.01673933, -0.020546, 0.0),
    }
    points = {
        name: _project(mujoco, renderer, palm_position + palm_rotation @ np.asarray(point))
        for name, point in local_points.items()
    }
    _line(draw, (points["rim"], points["corner"]), MAGENTA, 4)
    _line(draw, (points["flange"], points["corner"]), CYAN, 4)
    for name, color in (("rim", MAGENTA), ("flange", CYAN), ("corner", YELLOW)):
        x, y = points[name]
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
    renderer.close()
    measurement.save(ASSET_DIR / "apriltag_measurement_snapshot.png", optimize=True)

    visibility, _ = _visibility_panel()
    render, renderer = _render_snapshot(mujoco, model, data, oblique=True)
    visibility.paste(render, (0, 0))
    renderer.close()
    visibility.save(ASSET_DIR / "apriltag_visibility_snapshot.png", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    generate()
    print("Updated AprilTag measurement and visibility snapshots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
