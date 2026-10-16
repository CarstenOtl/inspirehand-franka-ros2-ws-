#!/usr/bin/env python3
"""See the two Inspire hand descriptions side by side, at the same joint command.

    python3 utils/compare_hand_models.py --render     # writes PNGs
    python3 utils/compare_hand_models.py              # interactive overlay

The policy observes and commands in the TRAINING hand's frame
(``TrainingHandKinematics`` loads ``assets/fr3_inspirehand/fr3_inspirehand_replay.xml``,
the official Tiangong 2.0 Pro geometry the student was distilled on), while ros-sim
simulates the WORKSPACE hand (``inspire_hand_on_flange.xml``, vendored from dex-urdf).
Both mount on ``fr3_link8``, and that frame is identical in the two models -- link7 to
link8 is (0, 0, 107) mm with identity rotation in each, and both seat the hand 10 mm
along +z -- so posing both at the same three driven joints and drawing them in that
frame is a like-for-like comparison.

What it shows: the two agree dimensionally (thumb base to thumb pad 93.3 vs 94.0 mm)
but the workspace hand carries its thumb pad about 15 degrees further from the index,
so at the posture where the training hand pinches to 12 mm its pads are 55 mm apart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
TRAINING_MJCF = REPO / "assets" / "fr3_inspirehand" / "fr3_inspirehand_replay.xml"
WORKSPACE_MJCF = REPO / "src" / "inspire_franka_sim" / "mjcf" / "inspire_hand_on_flange.xml"

TRAINING_HAND_BODIES = (
    "palm", "thumb_link_0", "thumb_link_1", "thumb_link_2", "thumb_link_3",
    "index_link_0", "index_link_1", "middle_link_0", "middle_link_1",
    "ring_link_0", "ring_link_1", "little_link_0", "little_link_1",
)
WORKSPACE_HAND_BODIES = (
    "hand_mount", "hand_base_link", "thumb_proximal_base", "thumb_proximal",
    "thumb_intermediate", "thumb_distal", "index_proximal", "index_intermediate",
    "middle_proximal", "middle_intermediate", "ring_proximal", "ring_intermediate",
    "pinky_proximal", "pinky_intermediate",
)

TRAINING_GREEN = [0.10, 0.72, 0.35, 1.0]
WORKSPACE_ORANGE = [0.98, 0.45, 0.10, 1.0]

#: (thumb yaw, thumb pitch, index), radians -- the postures worth comparing.
POSTURES = {
    "open": (1.0978, 0.0441, 0.0),
    "training_grip": (1.0978, 0.0441, 0.668),
    "ros_sim_max_curl": (1.2470, 0.1908, 0.7159),
}

#: Cameras in the fr3_link8 frame: (eye, target, up), metres.
VIEWS = {
    "pinch_plane": ((0.080, -0.400, 0.130), (-0.050, -0.025, 0.120), (0, 0, 1)),
    "pinch_closeup": ((0.010, -0.230, 0.185), (-0.062, -0.026, 0.158), (0, 0, 1)),
    "thumb_side": ((-0.430, -0.025, 0.130), (-0.050, -0.025, 0.120), (0, 0, 1)),
}


def training_joints(yaw: float, pitch: float, index: float) -> dict[str, float]:
    """The training asset's own coupling: 1.1169 fingers, 1.1425 then 0.7508 chained."""
    yaw, pitch, index = min(yaw, 1.246165), min(pitch, 0.48), min(index, 1.333)
    second = 1.1425 * pitch
    joints = {
        "thumb_joint_0": yaw, "thumb_joint_1": pitch,
        "thumb_joint_2": second, "thumb_joint_3": 0.7508 * second,
        "index_joint_0": index, "index_joint_1": 1.1169 * index,
    }
    for finger in ("middle", "ring", "little"):
        joints[f"{finger}_joint_0"] = 1.333
        joints[f"{finger}_joint_1"] = 1.1169 * 1.333
    return joints


def workspace_joints(yaw: float, pitch: float, index: float) -> dict[str, float]:
    """The workspace URDF's coupling: 1.06399 with a -0.04545 offset, 1.334, 0.667."""
    clamp = lambda value, low, high: max(low, min(high, value))
    joints = {
        "thumb_proximal_yaw_joint": min(yaw, 1.308),
        "thumb_proximal_pitch_joint": min(pitch, 0.6),
        "thumb_intermediate_joint": clamp(1.334 * pitch, 0.0, 0.8),
        "thumb_distal_joint": clamp(0.667 * pitch, 0.0, 0.4),
        "index_proximal_joint": min(index, 1.47),
        "index_intermediate_joint": clamp(1.06399 * index - 0.04545, -0.04545, 1.56),
    }
    for finger in ("middle", "ring", "pinky"):
        joints[f"{finger}_proximal_joint"] = 1.333
        joints[f"{finger}_intermediate_joint"] = clamp(
            1.06399 * 1.333 - 0.04545, -0.04545, 1.56
        )
    return joints


def _look_at(eye, target, up) -> np.ndarray:
    """MuJoCo cameras look down their own -z with +y up."""
    import mujoco

    forward = np.asarray(target, float) - np.asarray(eye, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, float))
    right /= np.linalg.norm(right)
    rotation = np.column_stack([right, np.cross(right, forward), -forward])
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, rotation.flatten())
    return quaternion


def hand_spec(path: Path, hand_bodies, rgba, *, cameras: bool):
    """Load one description, hide everything that is not the hand, colour the rest."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(path))
    keep = set(hand_bodies)
    for body in spec.bodies:
        for geom in body.geoms:
            if body.name not in keep or geom.group == 3:
                geom.group = 4          # not the hand, or a collision shell
            else:
                geom.group = 0
                geom.rgba = rgba
                geom.material = ""
    if cameras:
        flange = spec.body("fr3_link8")
        for name, (eye, target, up) in VIEWS.items():
            camera = flange.add_camera()
            camera.name = name
            camera.pos = list(eye)
            camera.quat = list(_look_at(eye, target, up))
            camera.fovy = 45
    spec.visual.headlight.ambient = [0.6, 0.6, 0.6]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]
    spec.visual.global_.offwidth = 1000
    spec.visual.global_.offheight = 1000
    return spec


def posed(model, joints):
    import mujoco

    data = mujoco.MjData(model)
    address = {model.joint(i).name: model.jnt_qposadr[i] for i in range(model.njnt)}
    data.qpos[:] = 0.0
    for name, value in joints.items():
        if name in address:
            data.qpos[address[name]] = float(value)
    mujoco.mj_forward(model, data)
    return data


def _whiten(image: np.ndarray) -> np.ndarray:
    """Each scene brings its own background; key it out so the two can be blended."""
    values = image.astype(np.int16)
    flat = (values.max(axis=2) - values.min(axis=2)) < 12
    background = flat & ((values.max(axis=2) < 40) | (values.min(axis=2) > 200))
    out = image.copy()
    out[background] = 255
    return out


def render(output: Path) -> None:
    import mujoco
    from PIL import Image

    training = hand_spec(TRAINING_MJCF, TRAINING_HAND_BODIES, TRAINING_GREEN, cameras=True).compile()
    workspace = hand_spec(WORKSPACE_MJCF, WORKSPACE_HAND_BODIES, WORKSPACE_ORANGE, cameras=True).compile()
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1
    output.mkdir(parents=True, exist_ok=True)

    def shot(model, data, view):
        with mujoco.Renderer(model, 1000, 1000) as renderer:
            renderer.update_scene(data, camera=view, scene_option=option)
            return _whiten(renderer.render().copy())

    for posture, values in POSTURES.items():
        left = posed(training, training_joints(*values))
        right = posed(workspace, workspace_joints(*values))
        for view in VIEWS:
            a, b = shot(training, left, view), shot(workspace, right, view)
            overlay = np.minimum(a.astype(np.int16), b.astype(np.int16)).astype(np.uint8)
            path = output / f"{posture}__{view}.png"
            Image.fromarray(np.hstack([a, b, overlay])).save(path)
            print(f"wrote {path}  (training | workspace | overlay)")


def interactive() -> None:
    """Both hands in one scene, SPACE steps through the postures."""
    import mujoco
    import mujoco.viewer

    combined = mujoco.MjSpec()
    combined.modelname = "inspire_hand_model_comparison"
    combined.compiler.degree = False

    # The workspace file is already hand-on-flange, with fr3_link8 at its root,
    # so attaching it puts that flange at the origin. The training file is a
    # whole arm, so lift just the palm subtree onto a flange of our own -- the
    # palm body carries its own +10 mm and half-turn, the same transform the
    # replay scene applies, so the two flanges coincide.
    workspace = hand_spec(WORKSPACE_MJCF, WORKSPACE_HAND_BODIES, WORKSPACE_ORANGE, cameras=False)
    combined.attach(workspace, prefix="workspace_", frame=combined.worldbody.add_frame())

    training = hand_spec(TRAINING_MJCF, TRAINING_HAND_BODIES, TRAINING_GREEN, cameras=False)
    flange = combined.worldbody.add_body(name="training_fr3_link8")
    flange.add_frame().attach_body(training.body("palm"), prefix="training_")

    combined.option.gravity = [0.0, 0.0, 0.0]
    model = combined.compile()
    data = mujoco.MjData(model)
    address = {model.joint(i).name: model.jnt_qposadr[i] for i in range(model.njnt)}

    names = list(POSTURES)
    state = {"index": 1}

    def apply():
        posture = names[state["index"]]
        values = POSTURES[posture]
        data.qpos[:] = 0.0
        for prefix, joints in (("training_", training_joints(*values)),
                               ("workspace_", workspace_joints(*values))):
            for name, value in joints.items():
                key = prefix + name
                if key in address:
                    data.qpos[address[key]] = float(value)
        mujoco.mj_forward(model, data)
        print(f"posture: {posture}  (thumb yaw {values[0]:.3f}, pitch {values[1]:.3f}, "
              f"index {values[2]:.3f})   green = training, orange = workspace")

    def on_key(keycode):
        if keycode in (32, ord(" ")):
            state["index"] = (state["index"] + 1) % len(names)
            apply()

    apply()
    print("SPACE cycles the posture. The hands are held, not simulated.")
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            viewer.sync()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="write PNGs instead of opening a viewer")
    parser.add_argument("--output", type=Path, default=REPO / "logs" / "hand_model_overlay")
    args = parser.parse_args()
    render(args.output) if args.render else interactive()


if __name__ == "__main__":
    main()
