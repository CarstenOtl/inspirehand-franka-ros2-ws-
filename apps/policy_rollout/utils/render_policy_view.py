#!/usr/bin/env python3
"""Draw what the DP3 encoder saw in a recorded rollout, next to training's nut.

    python3 apps/policy_rollout/utils/render_policy_view.py RUN_DIR \\
        [--times 0,4,8,16] [--out view.png] [--checkpoint CKPT]

For each requested time it re-samples the recorded RGB-D frame with the
checkpoint's own DP3 encoder (deterministic at evaluation, so these are the
4096 points the policy got) and draws two panels:

- the policy camera view with the sampled points, and crosses where training's
  nut centre (red), nut top (magenta) and bolt base (yellow) appear in this
  same camera, plus the recorded grasp point (cyan). Training rendered its
  images from this camera pose, so this comparison does not depend on how
  accurate the calibration is: a real nut away from the red cross is a scene
  that differs from training;
- a side view in fr3_link0 (x forward, z up) of the sampled points within 8 cm
  of the bolt plane, with training's table, bolt and nut and the grasp point.
  This one goes through the camera calibration, so it carries that error too
  (the 2026-10-29 fit has 9.7 mm / 1.0 deg residuals).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from policy_rollout import DEFAULT_CHECKPOINT  # noqa: E402
from policy_rollout.forge_osc import matrix_from_quat  # noqa: E402
from utils.camera_calibration import load_camera_calibration  # noqa: E402
from utils.data_collection import DATA_FILENAME, METADATA_FILENAME  # noqa: E402

# Training's M24 pair in fr3_link0 (training world (0.61, 0, z), base at x 1.2
# yawed by pi): nut centre at its first recorded pose, 36 mm across flats and
# 21 mm tall; bolt base on the table.
TRAINING_NUT_CENTRE = np.array([0.59, 0.0, 0.110])
NUT_HALF_WIDTH_M = 0.018
NUT_HALF_HEIGHT_M = 0.0105
TRAINING_BOLT_BASE = np.array([0.59, 0.0, 0.05])
SIDE_VIEW_HALF_WIDTH_M = 0.08


def sample_dp3_points(encoder, rgb, depth, valid) -> np.ndarray:
    """The encoder's evaluation sample of one frame, in camera metres."""

    import torch

    with torch.no_grad():
        points, sampled_valid = encoder._sample_points(
            torch.as_tensor(rgb, dtype=torch.float32)[None],
            torch.as_tensor(depth, dtype=torch.float32)[None],
            torch.as_tensor(valid)[None],
        )
    centre = encoder.xyz_center_m.cpu().numpy().reshape(3)
    scale = encoder.xyz_scale_m.cpu().numpy().reshape(3)
    xyz = points[0, :, :3].cpu().numpy() * scale + centre
    return xyz[sampled_valid[0].cpu().numpy().astype(bool)]


def load_encoder(checkpoint: Path):
    import torch

    from policy_rollout.dp3 import build_dp3_encoder

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder, _ = build_dp3_encoder(payload["config"]["vision_encoder_config"])
    return encoder.eval()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--times", default="0,4,8,16", help="seconds into the run")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)

    data = np.load(args.run / "data" / DATA_FILENAME)
    metadata = json.loads((args.run / "data" / METADATA_FILENAME).read_text())
    if "head_depth" not in data.files:
        raise SystemExit("this run has no RGB-D frames; record with --record-rgbd")
    checkpoint = args.checkpoint or Path(metadata.get("checkpoint", DEFAULT_CHECKPOINT))
    if not checkpoint.is_file():
        checkpoint = DEFAULT_CHECKPOINT
    encoder = load_encoder(checkpoint)

    profile = load_camera_calibration()
    k = np.asarray(profile.policy_intrinsics.camera_matrix, dtype=float).reshape(3, 3)
    pose = profile.training_world_pose
    r_base_cam = matrix_from_quat(np.asarray(pose.rotation_wxyz, dtype=float))
    t_base_cam = np.asarray(pose.translation_m, dtype=float)

    def to_pixel(base_point):
        camera = r_base_cam.T @ (np.asarray(base_point) - t_base_cam)
        return (k @ (camera / camera[2]))[:2]

    from matplotlib import use

    use("Agg")
    from matplotlib import pyplot
    from matplotlib.patches import Rectangle

    elapsed = data["sample_time_s"] - data["sample_time_s"][0]
    times = [float(value) for value in args.times.split(",")]
    figure, axes = pyplot.subplots(2, len(times), figsize=(4.2 * len(times), 6.4), squeeze=False)
    for column, when in enumerate(times):
        index = int(np.argmin(np.abs(elapsed - when)))
        xyz = sample_dp3_points(
            encoder, data["head_rgb"][index], data["head_depth"][index], data["depth_valid"][index]
        )
        grasp = data["grasp_position"][index]

        image = axes[0, column]
        image.imshow(np.clip(data["head_rgb"][index].transpose(1, 2, 0), 0.0, 1.0))
        uv = (k @ (xyz / xyz[:, 2:3]).T)[:2]
        image.scatter(uv[0], uv[1], s=0.4, c="#2a78d6", alpha=0.5, linewidths=0)
        for point, colour, label in (
            (TRAINING_NUT_CENTRE, "#e34948", "training nut centre"),
            (TRAINING_NUT_CENTRE + [0, 0, NUT_HALF_HEIGHT_M], "#c040c0", "training nut top"),
            (TRAINING_BOLT_BASE, "#e0c000", "training bolt base"),
            (grasp, "#1bae7a", "recorded grasp"),
        ):
            u, v = to_pixel(point)
            image.scatter([u], [v], marker="+", s=90, c=colour, linewidths=1.6, label=label)
        image.set_title(f"t = {elapsed[index]:.1f} s  ({len(xyz)} DP3 points)", fontsize=9)
        image.set_axis_off()

        side = axes[1, column]
        base = xyz @ r_base_cam.T + t_base_cam
        near = np.abs(base[:, 1] - TRAINING_NUT_CENTRE[1]) < SIDE_VIEW_HALF_WIDTH_M
        side.scatter(base[near, 0], base[near, 2], s=1.0, c="#2a78d6", alpha=0.6, linewidths=0)
        side.axhline(0.0, color="#888888", linewidth=0.8)
        side.add_patch(Rectangle(
            (TRAINING_NUT_CENTRE[0] - NUT_HALF_WIDTH_M, TRAINING_NUT_CENTRE[2] - NUT_HALF_HEIGHT_M),
            2 * NUT_HALF_WIDTH_M, 2 * NUT_HALF_HEIGHT_M, fill=False, edgecolor="#e34948", linewidth=1.4,
        ))
        side.plot([TRAINING_BOLT_BASE[0]] * 2, [0.0, TRAINING_BOLT_BASE[2]], color="#e0c000", linewidth=3)
        side.scatter([grasp[0]], [grasp[2]], marker="+", s=90, c="#1bae7a", linewidths=1.6)
        side.set_xlim(0.40, 0.80)
        side.set_ylim(-0.06, 0.30)
        side.set_aspect("equal")
        side.set_xlabel("fr3_link0 x [m]", fontsize=8)
        side.set_ylabel("z [m]", fontsize=8)
        side.tick_params(labelsize=7)
        side.grid(alpha=0.25)
    axes[0, 0].legend(loc="lower left", fontsize=6, framealpha=0.8)
    figure.suptitle(
        f"{args.run.name}: DP3 points (blue) vs training nut (red box/cross); "
        "bottom row via the camera calibration",
        fontsize=10,
    )
    figure.tight_layout()
    out = args.out or args.run / "analysis" / "policy_view.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=130)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
