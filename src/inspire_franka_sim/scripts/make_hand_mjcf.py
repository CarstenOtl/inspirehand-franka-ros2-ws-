#!/usr/bin/env python3
"""Turn the Inspire hand's URDF into a physics-ready MJCF.

    python3 scripts/make_hand_mjcf.py --side right

Writes ``mjcf/inspire_hand_<side>.xml`` and the meshes it needs into
``mjcf/assets_hand/``. Both are committed, so this only has to be re-run when
the description is re-vendored.

Why a generator and not a hand-written model
--------------------------------------------
MuJoCo will import the URDF directly, and the import is good: 12 joints with the
right ranges, 9 meshes, visual and collision geometry correctly separated. What
it cannot give is anything the URDF does not say, and the URDF is silent on four
things that matter:

1. **The finger coupling.** MuJoCo's URDF importer drops ``<mimic>`` without a
   word. Six joints would become free-swinging, and the hand would hang open.
   They are re-added here as ``<equality><joint>`` constraints, whose quartic
   ``polycoef`` expresses exactly the affine map the ``<mimic>`` did.

2. **Actuators.** A URDF has no notion of one. Six ``<motor>`` actuators are
   added, one per driven joint.

3. **Joint dynamics.** Upstream declares no damping, armature or friction
   anywhere. Undamped, mesh-collided finger joints at 5 g apiece are a stiff
   system that rings; see JOINT_DYNAMICS.

The reusable hand asset intentionally has no keyframe. Keyframes remain pending
through ``<attach>``, and the flange model passes the hand through two nested
attachment levels; MuJoCo cannot namespace such a keyframe safely. The final
hand-only and combined scenes own their complete ``open``/``start`` poses
instead.

Why ``<motor>`` and not ``<position>``
--------------------------------------
mujoco_ros2_control's actuator/command-interface compatibility matrix makes an
``effort`` command interface **unsupported** on a ``<position>`` actuator, and an
incompatible pairing is a hard error at controller_manager startup, not a
warning. A ``<motor>`` supports effort natively and position/velocity through
the PIDs in ``config/pids.yaml``. This is the same trade the FR3 model makes -
see ``mjcf/MJCF_PROVENANCE.md``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# The coupling, restated for MuJoCo. Must agree with the URDF's <mimic> tags and
# with inspire_hand_driver.kinematics; test/test_mjcf.py checks that it does.
#
# MuJoCo's equality/joint constraint is
#     y - y0 = a0 + a1*(x - x0) + a2*(x - x0)^2 + ...
# with y0/x0 the joints' qpos0. Both are 0 here (the URDF import sets no `ref`),
# so an affine mimic is just polycoef = [offset, multiplier, 0, 0, 0].
COUPLINGS = (
    # (follower, driver, multiplier, offset)
    ("index_intermediate_joint", "index_proximal_joint", 1.06399, -0.04545),
    ("middle_intermediate_joint", "middle_proximal_joint", 1.06399, -0.04545),
    ("ring_intermediate_joint", "ring_proximal_joint", 1.06399, -0.04545),
    ("pinky_intermediate_joint", "pinky_proximal_joint", 1.06399, -0.04545),
    ("thumb_intermediate_joint", "thumb_proximal_pitch_joint", 1.334, 0.0),
    ("thumb_distal_joint", "thumb_proximal_pitch_joint", 0.667, 0.0),
)

#: The six driven joints, in the register order the driver uses.
DRIVEN = (
    "pinky_proximal_joint",
    "ring_proximal_joint",
    "middle_proximal_joint",
    "index_proximal_joint",
    "thumb_proximal_pitch_joint",
    "thumb_proximal_yaw_joint",
)

# Torque clamp per driven joint, N m. This is upstream's declared <limit effort>,
# which is a placeholder rather than a measurement - but it is at least the right
# order of magnitude: the RH56 manages roughly 10 N at a fingertip on a ~50 mm
# lever, so ~0.5 N m, and 1.0 leaves headroom for the PIDs to accelerate through.
ACTUATOR_TORQUE = 1.0

# Unidentified stabilisers, not measurements. The description declares no joint
# dynamics at all, and an undamped chain of 5 g links on mesh contacts rings
# badly. These are the smallest values found to hold a curl steady at a 2 ms
# timestep without visibly slowing the finger down; retune them if you ever
# characterise the real hand.
JOINT_DYNAMICS = dict(damping=0.1, armature=0.002, frictionloss=0.005)

# Body pairs whose collision geometry overlaps no matter what the joints do, so
# a contact between them is always an artefact of the coarse geometry rather
# than the hand touching itself.
#
# There is exactly one, and it is the palm's fault. `hand_base_link` has no
# collision mesh - upstream gives it a cylinder and seven boxes, an envelope
# rather than the true shell - and one of those boxes swallows the thumb's
# proximal segment. Measured by sweeping all six driven joints across their full
# range (3000 random in-range poses): this pair sits at -12.1 mm at every one of
# them, including the rest pose, while every other overlap varies with the pose
# and bottoms out only where the thumb is genuinely driven into the index
# finger. Those are real self-collisions and are left alone.
#
# Re-run that sweep if the model is ever re-vendored; a finer palm mesh would
# make this entry unnecessary.
SPURIOUS_PAIRS = (("hand_base_link", "thumb_proximal"),)

MESH_SUBDIR = "assets_hand"


def rewrite_for_mujoco(urdf: str, meshdir: Path) -> str:
    """Make a ROS URDF loadable by MuJoCo's importer.

    Two changes. MuJoCo does not resolve ``package://`` URIs, so they are cut
    back to paths relative to a ``meshdir``; and MuJoCo reads its import options
    from a ``<mujoco>`` block inside the URDF, which ROS tooling ignores.
    """
    urdf = urdf.replace("package://inspire_hand_description/meshes/", "")
    options = (
        "<mujoco>\n"
        f'  <compiler meshdir="{meshdir}" balanceinertia="true"'
        ' discardvisual="false" fusestatic="false" strippath="false"/>\n'
        "</mujoco>\n"
    )
    return urdf.replace("</robot>", options + "</robot>")


def build(spec, side: str) -> None:
    """Add everything the URDF could not say."""
    import mujoco

    for joint in spec.joints:
        if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        # damping is a 3-vector in MjSpec (ball and free joints use all three);
        # a hinge reads only the first element.
        joint.damping[0] = JOINT_DYNAMICS["damping"]
        joint.armature = JOINT_DYNAMICS["armature"]
        joint.frictionloss = JOINT_DYNAMICS["frictionloss"]

    # MuJoCo's URDF importer leaves collision geoms in group 0, which the viewer
    # draws by default - so the hand renders twice, the collision hulls sitting
    # over the visual shells. Menagerie's convention is group 3 for collision,
    # which the viewer hides unless asked. Visual geoms (which the importer
    # already marks contype=0) keep group 1.
    for geom in spec.geoms:
        if geom.contype != 0 or geom.conaffinity != 0:
            geom.group = 3

    # Restore the parent/child contact filter by hand.
    #
    # MuJoCo normally excludes contacts between a parent body and its child -
    # except when the parent belongs to the world weld, which is exactly the
    # case for a hand standing on a bench: every joint from `world` down to
    # `hand_base_link` is fixed, so the palm IS the world body as far as the
    # filter is concerned. Without these excludes the palm's collision
    # primitives (a cylinder and seven boxes, a coarse envelope rather than the
    # true shell) intersect every finger root by 11-14 mm at the OPEN pose. The
    # fingers then start the simulation jammed: they close sluggishly against a
    # constraint force of order the entire actuator budget, and will not reopen
    # at all.
    #
    # Every direct parent/child pair is excluded, not just the palm's, so the
    # model behaves identically whether it is bolted to the arm's flange (where
    # the filter would have applied) or standing on the bench (where it would
    # not). Excluding a pair MuJoCo already excludes is a no-op.
    def exclude_pair(first: str, second: str) -> None:
        exclude = spec.add_exclude()
        exclude.name = f"{first}_{second}"
        exclude.bodyname1 = first
        exclude.bodyname2 = second

    for body in spec.bodies:
        parent = body.parent
        if parent is None or not body.geoms or not parent.geoms:
            continue
        exclude_pair(parent.name, body.name)

    # The palm's coarse envelope also engulfs a body that is not its direct
    # child, which the loop above therefore misses. See SPURIOUS_PAIRS.
    for first, second in SPURIOUS_PAIRS:
        exclude_pair(first, second)

    joint_names = {joint.name for joint in spec.joints}
    missing = (set(DRIVEN) | {c[0] for c in COUPLINGS}) - joint_names
    if missing:
        raise SystemExit(f"URDF is missing expected joints: {sorted(missing)}")

    for name in DRIVEN:
        actuator = spec.add_actuator()
        actuator.name = name
        actuator.set_to_motor()
        actuator.target = name
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.ctrlrange = [-ACTUATOR_TORQUE, ACTUATOR_TORQUE]
        actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE

    for follower, driver, multiplier, offset in COUPLINGS:
        equality = spec.add_equality()
        equality.name = f"{follower}_follows_{driver}"
        equality.type = mujoco.mjtEq.mjEQ_JOINT
        equality.objtype = mujoco.mjtObj.mjOBJ_JOINT
        equality.name1 = follower
        equality.name2 = driver
        equality.data[:5] = [offset, multiplier, 0.0, 0.0, 0.0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", default="right", choices=("right", "left"))
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="inspire_franka_sim's root (default: the parent of scripts/).",
    )
    parser.add_argument(
        "--description",
        type=Path,
        default=None,
        help="inspire_hand_description's root. Defaults to a sibling in src/.",
    )
    args = parser.parse_args(argv)

    import mujoco
    import xacro

    description = args.description or (
        args.package_root.parent / "inspire_hand_description"
    )
    if not (description / "urdf").is_dir():
        raise SystemExit(f"no inspire_hand_description at {description}")

    urdf = xacro.process_file(
        str(description / "urdf" / "inspire_hand.urdf.xacro"),
        mappings={
            "side": args.side,
            "ros2_control": "false",
            # No world link and no offset: the scene decides where the hand goes.
            "mount_to_world": "false",
        },
    ).toxml()

    mjcf_dir = args.package_root / "mjcf"
    scratch = mjcf_dir / f".inspire_hand_{args.side}.urdf"
    scratch.write_text(rewrite_for_mujoco(urdf, description / "meshes"))
    try:
        spec = mujoco.MjSpec.from_file(str(scratch))
    finally:
        scratch.unlink(missing_ok=True)

    spec.modelname = f"inspire_hand_{args.side}"
    build(spec, args.side)

    # Compile before writing: a spec that will not compile is not worth saving,
    # and the error is far more legible here than at controller_manager startup.
    model = spec.compile()

    # The saved XML has to find its meshes relative to itself, not relative to
    # wherever the description happened to be when this ran. The mesh elements
    # already carry `<side>/<kind>/<name>` paths from the import, so copying
    # that structure under assets_hand/ and pointing meshdir there keeps them
    # valid without rewriting every element.
    assets = mjcf_dir / MESH_SUBDIR
    for kind in ("visual", "collision"):
        source = description / "meshes" / args.side / kind
        target = assets / args.side / kind
        target.mkdir(parents=True, exist_ok=True)
        for mesh in sorted(source.glob("*")):
            shutil.copy2(mesh, target / mesh.name)
    spec.meshdir = MESH_SUBDIR

    # to_xml() opens each mesh to re-derive its path, and resolves meshdir
    # against the process's working directory rather than the output file. Run
    # it from mjcf/ so the relative meshdir it is about to write is the one it
    # validates against.
    out = mjcf_dir / f"inspire_hand_{args.side}.xml"
    cwd = Path.cwd()
    os.chdir(mjcf_dir)
    try:
        xml = spec.to_xml()
    finally:
        os.chdir(cwd)
    out.write_text(xml)

    print(
        f"{out.relative_to(args.package_root)}: "
        f"{model.nbody - 1} bodies, {model.njnt} joints, {model.nu} actuators, "
        f"{model.neq} equality constraints, {model.nmesh} meshes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
