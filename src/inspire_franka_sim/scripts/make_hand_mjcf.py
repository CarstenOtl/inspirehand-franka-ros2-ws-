#!/usr/bin/env python3
"""Turn the Inspire hand's URDF into a physics-ready MJCF.

    python3 scripts/make_hand_mjcf.py --side right
    python3 scripts/make_hand_mjcf.py --isaac

Writes ``mjcf/inspire_hand_<side>.xml`` and the meshes it needs into
``mjcf/assets_hand/``. Both are committed, so this only has to be re-run when
the description is re-vendored. ``--isaac`` writes the same hand with Isaac
Lab's training-time dynamics instead (``mjcf/inspire_hand_right_isaac.xml`` and
its flange wrapper), for the policy rollout scene; see ISAAC_DRIVES.

Why a generator and not a hand-written model
--------------------------------------------
MuJoCo will import the URDF directly, and the import is good: 12 joints with the
right ranges, 13 meshes, visual and collision geometry correctly separated. What
it cannot give is anything the URDF does not say, and the URDF is silent on four
things that matter:

1. **The finger coupling.** MuJoCo's URDF importer drops ``<mimic>`` without a
   word. Six joints would become free-swinging, and the hand would hang open.
   They are re-added here as ``<equality><joint>`` constraints, whose quartic
   ``polycoef`` expresses exactly the affine map the ``<mimic>`` did.

2. **Actuators.** A URDF has no notion of one. Six ``<motor>`` actuators are
   added, one per driven joint.

3. **Joint dynamics.** The description declares no damping, armature or friction
   anywhere. Undamped, mesh-collided finger links of 5 to 60 g are a stiff
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
#
# The thumb's distal joint follows the intermediate, not the pitch, as in the
# training hand's URDF; an equality on a follower chains the way the mimic does.
COUPLINGS = (
    # (follower, driver, multiplier, offset)
    ("index_intermediate_joint", "index_proximal_joint", 1.1169, 0.0),
    ("middle_intermediate_joint", "middle_proximal_joint", 1.1169, 0.0),
    ("ring_intermediate_joint", "ring_proximal_joint", 1.1169, 0.0),
    ("pinky_intermediate_joint", "pinky_proximal_joint", 1.1169, 0.0),
    ("thumb_intermediate_joint", "thumb_proximal_pitch_joint", 1.1425, 0.0),
    ("thumb_distal_joint", "thumb_intermediate_joint", 0.7508, 0.0),
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

# Torque clamp per driven joint, N m. This is the description's <limit effort>,
# which is a placeholder rather than a measurement - but it is at least the right
# order of magnitude: the RH56 manages roughly 10 N at a fingertip on a ~50 mm
# lever, so ~0.5 N m, and 1.0 leaves headroom for the PIDs to accelerate through.
ACTUATOR_TORQUE = 1.0

# Unidentified stabilisers, not measurements. The description declares no joint
# dynamics at all, and an undamped chain of 5 to 60 g links on mesh contacts rings
# badly. These are the smallest values found to hold a curl steady at a 2 ms
# timestep without visibly slowing the finger down; retune them if you ever
# characterise the real hand.
JOINT_DYNAMICS = dict(damping=0.1, armature=0.002, frictionloss=0.005)

# Sliding friction of the finger pads, from the training asset rather than from
# a measurement. ForgeUltra sets 0.75 on every geom of its hand's
# `hand_collision` class and on the nut
# (`policy_rollout.mujoco_threading_env.CONTACT_FRICTION`), so a hand that
# keeps MuJoCo's default 1.0 grips the threading nut harder than the plant the
# student was distilled on -- MuJoCo takes the element-wise MAXIMUM of the two
# geoms' friction, so lowering the nut alone cannot fix it.
#
# Real pad rubber on a steel nut is unmeasured; see the sim-to-real table in
# docs/policy_rollout_training_gap_todo.md. This is training's number, chosen
# so the two simulators agree, not a claim about the hardware.
CONTACT_FRICTION = (0.75, 0.005, 0.0001)

# Body pairs whose collision geometry overlaps at rest, so a contact between them
# is an artefact of the coarse geometry rather than the hand touching itself.
#
# There is exactly one, and it is the palm's fault. MuJoCo collides a mesh as
# its convex hull, and the hull of the palm shell fills the hollow the thumb's
# proximal segment sits in. Measured by sweeping all six driven joints across
# their full range (1500 random in-range poses plus the open pose): this pair
# overlaps by 3.5 mm at the open pose and by up to 7.2 mm elsewhere, and only
# clears by 3 mm at the far end of the thumb's travel. Every other overlap is
# the thumb genuinely driven into the index finger -- 44 mm apart or more at
# the open pose -- and is left alone.
#
# Re-run that sweep if the model is ever re-vendored; a convex decomposition of
# the palm would make this entry unnecessary.
SPURIOUS_PAIRS = (("hand_base_link", "thumb_proximal"),)

MESH_SUBDIR = "assets_hand"

# --- The hand as Isaac Lab simulates it for the threading student (--isaac) ---
#
# ForgeUltra's official profile (FR3_INSPIREHAND_ASSET_PROFILE unset, the
# default since 2026-09-08; the M24 reset pose is written for it). Geometry,
# masses and joint limits are already the training URDF's; what the URDF does
# not carry is set at spawn, and these are those values:
#
# - assets/fr3_inspirehand/build_fr3_inspirehand.py strips every <mimic> before
#   the USD conversion. The followers are free joints with their own drives,
#   and forge_ultra/tasks/mdp/robot_control.py `_expand_hand_mimic_targets`
#   couples them at the TARGET level only: follower target = multiplier *
#   leader target. No equality holds them; contact can bend a distal joint off
#   its leader. inspire_franka_sim's IsaacHandPlugin does that expansion.
# - assets/fr3_inspirehand/robot.py: armature, PhysX joint friction
#   (coefficient), stiffness/damping, effort 50 and velocity 1 rad/s on all
#   twelve joints.
# - forge_franka_threading_v2/forge_franka_env.py overrides the six leaders at
#   reset: stiffness 30, damping 0.1 * 30, effort 20 (frozen_hand_*), and the
#   reset pose's inactive fingers (middle, ring, little) 4.76 / 0.21 / 1.864.
#
# A PhysX "force" drive is f = k (target - q) - d qdot clamped at the effort
# limit, which is exactly a MuJoCo <position> actuator with kp = k, kv = d and
# forcerange = +-effort. The velocity limit and the joint friction have no
# MuJoCo counterpart and are applied by the plugin, from the <custom> block the
# policy scene carries (ISAAC_VELOCITY_LIMIT, ISAAC_JOINT_FRICTION).

#: Isaac's name for each workspace joint, for tracing the tables below.
ISAAC_JOINT_NAMES = {
    "thumb_proximal_yaw_joint": "thumb_joint_0",
    "thumb_proximal_pitch_joint": "thumb_joint_1",
    "thumb_intermediate_joint": "thumb_joint_2",
    "thumb_distal_joint": "thumb_joint_3",
    "index_proximal_joint": "index_joint_0",
    "index_intermediate_joint": "index_joint_1",
    "middle_proximal_joint": "middle_joint_0",
    "middle_intermediate_joint": "middle_joint_1",
    "ring_proximal_joint": "ring_joint_0",
    "ring_intermediate_joint": "ring_joint_1",
    "pinky_proximal_joint": "little_joint_0",
    "pinky_intermediate_joint": "little_joint_1",
}

#: (stiffness N m/rad, damping N m s/rad, effort limit N m) per joint.
ISAAC_DRIVES = {
    # Pinch leaders: frozen_hand_stiffness / 0.1 * that / frozen_hand_effort.
    "thumb_proximal_yaw_joint": (30.0, 3.0, 20.0),
    "thumb_proximal_pitch_joint": (30.0, 3.0, 20.0),
    "index_proximal_joint": (30.0, 3.0, 20.0),
    # Inactive leaders (reset_pose_m24.yaml: mode inactive).
    "middle_proximal_joint": (4.76, 0.21, 1.864),
    "ring_proximal_joint": (4.76, 0.21, 1.864),
    "pinky_proximal_joint": (4.76, 0.21, 1.864),
    # Followers keep robot.py's _HAND_STIFFNESS / _HAND_DAMPING and effort 50.
    "thumb_intermediate_joint": (4.76, 0.21, 50.0),
    "thumb_distal_joint": (0.9, 0.042, 50.0),
    "index_intermediate_joint": (0.9, 0.042, 50.0),
    "middle_intermediate_joint": (0.9, 0.042, 50.0),
    "ring_intermediate_joint": (0.9, 0.042, 50.0),
    "pinky_intermediate_joint": (0.9, 0.042, 50.0),
}

#: robot.py _HAND_ARMATURE, kg m^2.
ISAAC_ARMATURE = {
    "thumb_proximal_yaw_joint": 0.0032,
    "thumb_proximal_pitch_joint": 0.0032,
    "thumb_intermediate_joint": 0.00265,
    "thumb_distal_joint": 0.0006,
    "index_proximal_joint": 0.00265,
    "middle_proximal_joint": 0.00265,
    "ring_proximal_joint": 0.00265,
    "pinky_proximal_joint": 0.00265,
    "index_intermediate_joint": 0.0006,
    "middle_intermediate_joint": 0.0006,
    "ring_intermediate_joint": 0.0006,
    "pinky_intermediate_joint": 0.0006,
}

#: robot.py _HAND_FRICTION: PhysX's unitless joint friction coefficient, which
#: bounds the resisting effort by coefficient * (|F| + |T|) of the spatial force
#: the joint transmits (PxArticulationJointReducedCoordinate::
#: setFrictionCoefficient). Load-dependent, so it is not a MuJoCo frictionloss;
#: the plugin recomputes the frictionloss from that bound every step.
ISAAC_JOINT_FRICTION = {
    "thumb_proximal_yaw_joint": 0.132,
    "thumb_proximal_pitch_joint": 0.132,
    "thumb_intermediate_joint": 0.07456,
    "thumb_distal_joint": 0.01276,
    "index_proximal_joint": 0.07456,
    "middle_proximal_joint": 0.07456,
    "ring_proximal_joint": 0.07456,
    "pinky_proximal_joint": 0.07456,
    "index_intermediate_joint": 0.01276,
    "middle_intermediate_joint": 0.01276,
    "ring_intermediate_joint": 0.01276,
    "pinky_intermediate_joint": 0.01276,
}

#: robot.py _HAND_VELOCITY_LIMITS (the official URDF's <limit velocity>), rad/s.
ISAAC_VELOCITY_LIMIT = 1.0


#: The hand-written flange wrapper (tool sites, adapter frame); the Isaac variant
#: is the same file pointed at the Isaac hand, so the two cannot drift apart.
FLANGE_WRAPPER = "inspire_hand_on_flange.xml"


def isaac_flange_wrapper(text: str, hand_file: str) -> str:
    """``inspire_hand_on_flange.xml`` mounting ``hand_file`` instead."""

    source = '<model name="inspire_hand" file="inspire_hand_right.xml"/>'
    if text.count(source) != 1:
        raise SystemExit(f"{FLANGE_WRAPPER} no longer mounts inspire_hand_right.xml once")
    text = text.replace(source, f'<model name="inspire_hand" file="{hand_file}"/>')
    text = text.replace(
        '<mujoco model="inspire_hand_on_flange">',
        '<mujoco model="inspire_hand_on_flange_isaac">\n'
        "  <!-- GENERATED by inspire_franka_sim/scripts/make_hand_mjcf.py "
        f"from {FLANGE_WRAPPER}\n       (make_hand_mjcf.py with the isaac flag). "
        "Edit that file, not this one. -->",
        1,
    )
    return text


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


def build(spec, side: str, isaac: bool = False) -> None:
    """Add everything the URDF could not say.

    ``isaac`` builds the hand as Isaac Lab simulates it instead (see
    ISAAC_DRIVES): position actuators on all twelve joints, Isaac's armature,
    no passive damping or friction, and no equality couplings.
    """
    import mujoco

    for joint in spec.joints:
        if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        # damping is a 3-vector in MjSpec (ball and free joints use all three);
        # a hinge reads only the first element.
        if isaac:
            joint.damping[0] = 0.0
            joint.armature = ISAAC_ARMATURE[joint.name]
            joint.frictionloss = 0.0
            # The description's <limit effort="1"> imports as a joint-level
            # actuator force clamp, which would cut Isaac's 20 N m pinch drive
            # to 1 N m; the drive's own effort limit is the clamp here.
            effort = ISAAC_DRIVES[joint.name][2]
            joint.actfrcrange = [-effort, effort]
            joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        else:
            joint.damping[0] = JOINT_DYNAMICS["damping"]
            joint.armature = JOINT_DYNAMICS["armature"]
            joint.frictionloss = JOINT_DYNAMICS["frictionloss"]

    # MuJoCo's URDF importer leaves collision geoms in group 0, which the viewer
    # draws by default - so the hand renders twice, the collision hulls sitting
    # over the visual shells. Menagerie's convention is group 3 for collision,
    # which the viewer hides unless asked. Visual geoms (which the importer
    # already marks contype=0) keep group 1.
    #
    # The same pass sets the pad friction, which the URDF is also silent on; see
    # CONTACT_FRICTION.
    for geom in spec.geoms:
        if geom.contype != 0 or geom.conaffinity != 0:
            geom.group = 3
            geom.friction = list(CONTACT_FRICTION)

    # Restore the parent/child contact filter by hand.
    #
    # MuJoCo normally excludes contacts between a parent body and its child -
    # except when the parent belongs to the world weld, which is exactly the
    # case for a hand standing on a bench: every joint from `world` down to
    # `hand_base_link` is fixed, so the palm IS the world body as far as the
    # filter is concerned. Without these excludes the palm's collision hull (the
    # convex hull of its shell, which fills the hollows the fingers sit in)
    # intersects the finger roots it carries. The fingers then start the
    # simulation jammed: they close sluggishly against a constraint force of
    # order the entire actuator budget, and may not reopen at all.
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

    # The palm's hull also engulfs a body that is not its direct
    # child, which the loop above therefore misses. See SPURIOUS_PAIRS.
    for first, second in SPURIOUS_PAIRS:
        exclude_pair(first, second)

    joint_names = {joint.name for joint in spec.joints}
    missing = (set(DRIVEN) | {c[0] for c in COUPLINGS}) - joint_names
    if missing:
        raise SystemExit(f"URDF is missing expected joints: {sorted(missing)}")

    if isaac:
        # One drive per joint, followers included, in the URDF's joint order.
        for joint in spec.joints:
            if joint.name not in ISAAC_DRIVES:
                continue
            stiffness, damping, effort = ISAAC_DRIVES[joint.name]
            actuator = spec.add_actuator()
            actuator.name = joint.name
            actuator.set_to_position(kp=stiffness, kv=damping)
            actuator.target = joint.name
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.ctrlrange = list(joint.range)
            actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            actuator.forcerange = [-effort, effort]
            actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        return

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
    parser.add_argument(
        "--isaac",
        action="store_true",
        help="write inspire_hand_<side>_isaac.xml (the hand as Isaac Lab simulates "
        "it, see ISAAC_DRIVES) and inspire_hand_on_flange_isaac.xml instead",
    )
    args = parser.parse_args(argv)
    if args.isaac and args.side != "right":
        raise SystemExit("--isaac: training only has the right hand")

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

    variant = f"inspire_hand_{args.side}" + ("_isaac" if args.isaac else "")
    spec.modelname = variant
    build(spec, args.side, isaac=args.isaac)

    # Compile before writing: a spec that will not compile is not worth saving,
    # and the error is far more legible here than at controller_manager startup.
    model = spec.compile()

    # The saved XML has to find its meshes relative to itself, not relative to
    # wherever the description happened to be when this ran. The mesh elements
    # already carry `<side>/<name>` paths from the import, so copying that
    # structure under assets_hand/ and pointing meshdir there keeps them valid
    # without rewriting every element. The side's directory is replaced
    # whole, so meshes from an earlier model do not linger.
    assets = mjcf_dir / MESH_SUBDIR
    target = assets / args.side
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for mesh in sorted((description / "meshes" / args.side).glob("*")):
        shutil.copy2(mesh, target / mesh.name)
    spec.meshdir = MESH_SUBDIR

    # to_xml() opens each mesh to re-derive its path, and resolves meshdir
    # against the process's working directory rather than the output file. Run
    # it from mjcf/ so the relative meshdir it is about to write is the one it
    # validates against.
    out = mjcf_dir / f"{variant}.xml"
    cwd = Path.cwd()
    os.chdir(mjcf_dir)
    try:
        xml = spec.to_xml()
    finally:
        os.chdir(cwd)
    out.write_text(xml)
    if args.isaac:
        flange = mjcf_dir / FLANGE_WRAPPER
        isaac_flange = mjcf_dir / FLANGE_WRAPPER.replace(".xml", "_isaac.xml")
        isaac_flange.write_text(isaac_flange_wrapper(flange.read_text(), out.name))
        print(f"{isaac_flange.relative_to(args.package_root)}: {FLANGE_WRAPPER} on {out.name}")

    print(
        f"{out.relative_to(args.package_root)}: "
        f"{model.nbody - 1} bodies, {model.njnt} joints, {model.nu} actuators, "
        f"{model.neq} equality constraints, {model.nmesh} meshes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
