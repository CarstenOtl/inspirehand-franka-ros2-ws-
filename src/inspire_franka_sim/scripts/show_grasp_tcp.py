#!/usr/bin/env python3
"""Show the grasp-centre TCP the Cartesian impedance replay controls.

A static demonstration: one hand-picked arm pose and the threading grip, no
trajectory, no ROS. The orange sphere and its drawn axes are the ``grasp_tcp``
site in ``inspire_hand_on_flange.xml``; the grey sphere at the wrist is
``attachment_site``, the flange ``fr3_link8``. The script also reports how far
the site sits from the thumb/index fingertip midpoint it is meant to mark, so
the picture is checkable rather than merely decorative.

    ./show_grasp_tcp.py --view              # interactive MuJoCo window
    ./show_grasp_tcp.py --out /tmp/tcp.png  # offscreen render

Both need a display; the container inherits the host's through ``DISPLAY``.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# The threading grip, mean over the grip samples of the traj_3 capture. Hard-coded
# on purpose: this demonstration must not depend on a recording being present.
GRIP = {
    "pinky_proximal_joint": 1.3325,
    "ring_proximal_joint": 1.3323,
    "middle_proximal_joint": 1.3325,
    "index_proximal_joint": 0.6777,
    "thumb_proximal_pitch_joint": 0.0214,
    "thumb_proximal_yaw_joint": 1.0987,
}
OPEN = {name: 0.0 for name in GRIP}
# A pose that simply presents the hand to the camera. Nothing depends on it.
DEMO_ARM = np.array([0.0, -0.35, 0.0, -2.2, 0.0, 1.9, 0.785])

# inspire_hand_description's URDF mimic tags, so the intermediate links follow.
COUPLING = {
    "index_intermediate_joint": ("index_proximal_joint", 1.06399, -0.04545),
    "middle_intermediate_joint": ("middle_proximal_joint", 1.06399, -0.04545),
    "ring_intermediate_joint": ("ring_proximal_joint", 1.06399, -0.04545),
    "pinky_intermediate_joint": ("pinky_proximal_joint", 1.06399, -0.04545),
    "thumb_intermediate_joint": ("thumb_proximal_pitch_joint", 1.334, 0.0),
    "thumb_distal_joint": ("thumb_proximal_pitch_joint", 0.667, 0.0),
}


def scene_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(here, "..", "mjcf", "inspire_franka_flange_scene.xml"),
        os.path.join(here, "..", "share", "inspire_franka_sim", "mjcf",
                     "inspire_franka_flange_scene.xml"),
    ):
        if os.path.exists(candidate):
            return os.path.normpath(candidate)
    from ament_index_python.packages import get_package_share_directory

    return os.path.join(get_package_share_directory("inspire_franka_sim"), "mjcf",
                        "inspire_franka_flange_scene.xml")


def pose(model, data, mujoco, arm, hand):
    address = {model.joint(i).name: model.joint(i).qposadr[0] for i in range(model.njnt)}
    for index, value in enumerate(arm, start=1):
        data.qpos[address[f"fr3_joint{index}"]] = value
    for name, value in hand.items():
        data.qpos[address[name]] = value
    for name, (source, gain, offset) in COUPLING.items():
        data.qpos[address[name]] = gain * data.qpos[address[source]] + offset
    mujoco.mj_forward(model, data)


def report(model, data, mujoco):
    site = data.site(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_tcp")).xpos
    tips = [data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)]
            for n in ("thumb_tip", "index_tip")]
    midpoint = 0.5 * (tips[0] + tips[1])
    flange = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3_link8")]
    return {
        "site": site.copy(),
        "midpoint": midpoint.copy(),
        "site_to_midpoint_mm": 1000.0 * float(np.linalg.norm(site - midpoint)),
        "flange_to_site_mm": 1000.0 * float(np.linalg.norm(site - flange)),
        "tip_separation_mm": 1000.0 * float(np.linalg.norm(tips[0] - tips[1])),
    }


def options(mujoco):
    """Draw the frames of the hidden site group 4, and nothing else new."""
    option = mujoco.MjvOption()
    option.frame = mujoco.mjtFrame.mjFRAME_SITE
    for group in range(len(option.sitegroup)):
        option.sitegroup[group] = 1 if group == 4 else 0
    return option


def look_at(mujoco, target, distance, azimuth, elevation):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = target
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", action="store_true", help="open an interactive window")
    parser.add_argument("--out", default=None, help="write a PNG instead of opening a window")
    parser.add_argument("--open-hand", action="store_true",
                        help="show the open hand rather than the threading grip")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--distance", type=float, default=0.42)
    parser.add_argument("--azimuth", type=float, default=150.0)
    parser.add_argument("--elevation", type=float, default=-18.0)
    parser.add_argument("--frame-scale", type=float, default=0.12,
                        help="drawn frame axis length, relative to MuJoCo's default")
    args = parser.parse_args(argv)

    os.environ.setdefault("MUJOCO_GL", "glfw")
    import mujoco

    model = mujoco.MjModel.from_xml_path(scene_path())
    # MuJoCo sizes frame axes from the model extent, which for a whole arm makes
    # them longer than the hand itself.
    model.vis.scale.framelength *= args.frame_scale
    model.vis.scale.framewidth *= args.frame_scale
    data = mujoco.MjData(model)
    pose(model, data, mujoco, DEMO_ARM, OPEN if args.open_hand else GRIP)
    measured = report(model, data, mujoco)

    print("grasp_tcp site, %s hand:" % ("open" if args.open_hand else "threading grip"))
    print("  world position          %s m" % np.array2string(measured["site"], precision=4))
    print("  flange to the site      %.1f mm" % measured["flange_to_site_mm"])
    print("  thumb/index midpoint    %s m" % np.array2string(measured["midpoint"], precision=4))
    print("  site to that midpoint   %.1f mm" % measured["site_to_midpoint_mm"])
    print("  fingertip separation    %.1f mm" % measured["tip_separation_mm"])

    camera = look_at(mujoco, measured["site"], args.distance, args.azimuth, args.elevation)
    if args.view:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                          show_right_ui=False) as viewer:
            with viewer.lock():
                source = options(mujoco)
                viewer.opt.frame = source.frame
                for group in range(len(viewer.opt.sitegroup)):
                    viewer.opt.sitegroup[group] = source.sitegroup[group]
                viewer.cam.type = camera.type
                viewer.cam.lookat[:] = camera.lookat
                viewer.cam.distance = camera.distance
                viewer.cam.azimuth = camera.azimuth
                viewer.cam.elevation = camera.elevation
            viewer.sync()
            print("\nclose the window to exit")
            while viewer.is_running():
                viewer.sync()
        return 0

    # The offscreen framebuffer is a model property and the scenes leave it at
    # MuJoCo's 640x480 default; raise it here rather than in the shared MJCF.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(model, args.height, args.width)
    renderer.update_scene(data, camera=camera, scene_option=options(mujoco))
    image = renderer.render()
    if args.out:
        try:
            from PIL import Image

            Image.fromarray(image).save(args.out)
        except ImportError:
            import imageio.v3 as iio

            iio.imwrite(args.out, image)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
