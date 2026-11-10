# Hardware rollout notes, 2026-11-10

The exp20 fitted-hand DP3 students on the real FR3 and Inspire hand. What worked,
what caused each failure, and what is still open.

## Result

Run `hardware-20261110-173006-400353`: the proprio-friction student completed
**3 of 3 cycles**, with a joint-space re-home between cycles. It is the first
multi-cycle hardware run of this student.

```bash
# from apps/policy_rollout, on the friction-compensated controller (controllers_policy.yaml)
python3 run_policy_rollout.py hardware --viewer --record-rgbd --cycles 3 \
  --rehome-between-cycles --max-home-grasp-error 0.003 \
  --release-signal grasp-yaw --release-proxy-deg 30 \
  --checkpoint <m24-dp3-exp20r01-proprio-friction-...>/nn/checkpoint.pt
```

| Student | Outcome on hardware |
|---|---|
| proprio-friction (reduced randomization) | Good grip and turn with friction compensation; 3 cycles with the command above |
| extended randomization | Fingers slide under the nut and never grip its sides, with or without friction compensation. It also aims low in ros-sim (tips 88-94 mm, nut 101-119 mm). Untested explanation: nut depth randomization |

Both students stop on the fr3_joint5 limit in ros-sim after 1-2 cycles. That limit
is still the next risk for longer runs.

## Why the Cartesian controller is inaccurate at rest

The return to the reset pose stalls 12-14 mm and up to about 20 deg off, while
joint-space homing reaches about 2 mm. Decomposing the commanded torque at a stall
(run 140157, reproducing all seven logged torques):

- The position spring (13.7 mm, 7.7 N) and the rotation spring (about 3.5 N m) reach
  3.8 N m at single joints, but they largely cancel. The net torque is 1.5 N m at
  most.
- The rotation target is clipped to 0.097 rad per axis, about 2.7 N m. Past 5.6 deg
  of error, more error adds no push.
- There is no integral term, as in training, so anything joint friction holds
  becomes a permanent error.
- The exact nullspace projector gives at most 0.04 N m, even with joints 0.2-0.4 rad
  off, so posture is not restored.
- 1 N m held at one joint maps to 2-6 mm and up to 2.8 deg at the grasp point.

These gains come from a frictionless simulated arm. Friction magnitude here is
inferred from zero motion, not measured; `arm_torque_external_nm` is sensor minus
model and is not a friction measurement.

## Re-homing between cycles

`--rehome-between-cycles` lets the student's return clear the nut (gate 30 mm /
25 deg by default), then switches to the joint-impedance controller, homes, and
switches back.

Across the pause the policy state is kept consistent:

- The progress clock holds.
- The action ensemble is cleared and the previous action is reseeded from the homed
  pose.
- The grasp-yaw turn proxy restarts, otherwise the wrist's return of up to 20 deg
  would count as turn in the next cycle.

Observed: 11.5 mm / 25 deg before, 4.5 mm / 3.3 deg after, 6.2 s per re-home.

The handback once got no reply from the controller manager for 15 s (run 172327),
on the Cartesian client, which also subscribes to the full FrankaRobotState. The
cause is suspected rather than proven. The handback now goes through the homing
client, and switches retry on timeout.

## Release signal

The automatic hardware release is a finger sweep (separation 10 mm or less, index
0.82 or more), and this student holds 33-35 mm, so it never fires. Use
`--release-signal grasp-yaw --release-proxy-deg 30`: the wrist turns less than the
nut (proxy 33-34 deg at 56-59 deg of real twist in ros-sim).

## Checks that saved time, and false leads

**The bolt holder moved, and it looked like an inertia problem.** A run of missed
grips began right after the Desk end-effector inertia was edited. Reverting the
inertia, confirmed by reading `inertia_ee` from the robot state, did not help. The
first camera frames showed the holder and its paper shifted 1.1-1.4 px left and
9-15 mm further from the camera, while the table edge stayed put. The arrival
height and the unloaded thumb pad matched that shift, and restoring the holder
position restored the grip.

To check the fixture, compare a frame with a good run by phase correlation on the
holder region, with a static region as a control. 0.2 px at this image scale is
already meaningful.

**Desk end-effector inertia.** The two parameter sets differ by about 0.11 N m
static (20 mm COM offset, 0.58 kg), too small to explain the return error. The
values currently in Desk are the original ones: mass 0.5789 kg, COM (0.00358,
-0.00413, 0.09805) m, Ixx/Iyy/Izz 0.000408/0.000538/0.000197, no cross terms.

**F_T_EE must stay identity.** The runner applies the tool offset itself and refuses
to start otherwise.

**`frame_id: fr3_link2` on robot state is cosmetic.** It comes from a
`link_names[kFlangeLinkIndex]` lookup that assumes a linear URDF.

## Open

- A re-home stops correcting once it is within 5 mm. It lands about 4.5 mm / 3.3 deg
  off, against about 2 mm at startup, and leaves 0.5 mm of margin before
  `rehome_failed`.
- The fr3_joint5 limit in longer runs.
- The extended student's low aim.
