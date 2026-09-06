# Model provenance

Everything under `urdf/` and `meshes/` is **generated**. The generator is
`scripts/vendor_from_dex_urdf.py`; edit that, not its output.

| | |
|---|---|
| Model | Inspire Robotics RH56 dexterous hand, left and right |
| Upstream | `dex-urdf` (dexsuite), whose authors derived it from the STEP files Inspire publishes at <https://www.inspire-robots.com/download/frwz/> |
| Reached this repo via | `tiangong_infra_ws/asset/tiangong2pro` |
| Licence | See `UPSTREAM_LICENSE.txt` |
| Vendored on | 2026-09-06 |

## What the source looks like

Two flat, **collision-only** URDFs (`inspire_hand_{left,right}.urdf`) plus two
disjoint mesh trees:

- `meshes/obj_meshes/inspire_hand/` — OBJ, referenced by the URDFs' `<collision>`
  elements. Eight per side: four links share `index_proximal` and two share
  `index_intermediate`, so there are fewer meshes than links.
- `meshes/glb_meshes/inspire_hand/visual/` — glTF binary, referenced by nothing.
  Nine per side; the extra one is the palm, which has no collision mesh (its
  collision is a cylinder and seven boxes).

The two sets agree: same frame, same units (metres), extents matching to well
under a millimetre. That was checked before adopting the glTF set as the visual
geometry.

## What the generator changes, and why

1. **Flat URDF → xacro macro.** A flat URDF cannot be instantiated twice, given a
   prefix, or attached to a parent link — all three of which this workspace
   needs. The macro takes `prefix` and a `parent`/`xyz`/`rpy` mount.

2. **Visuals added.** Upstream has no `<visual>` at all, so RViz would render
   nothing. The glTF visuals are converted to **binary STL**, which RViz and
   MuJoCo both read without argument, and one is attached to each link. The
   conversion flattens the glTF scene graph (`trimesh.load(force='mesh')`),
   baking in node transforms; it also discards upstream's per-part colours, so
   the URDF paints the whole hand one dark grey. Re-materialising from the glTF
   would mean shipping Collada instead, which RViz reads but MuJoCo does not.

3. **Meshes de-duplicated and re-rooted.** The shared meshes land once each under
   `meshes/<side>/{visual,collision}/`, and the upstream `../../meshes/...`
   relative paths become `package://` URIs that resolve wherever the package is
   installed. The `right_`/`left_` filename prefixes are dropped — the side is
   the directory now.

4. **`base` renamed to `hand_mount`.** A link called `base` is fine in a
   standalone hand and a hazard in a description that also contains an arm.
   Every other link and **every joint** keeps its upstream name, which matters:
   the joint names are the contract between the description, the driver
   (`inspire_hand_driver.kinematics`) and the MJCF.

`<mimic>` tags are carried through unchanged.

## Twelve joints, six actuators

The hand has six actuators. Each finger's `*_intermediate` joint follows its
`*_proximal` joint through a four-bar linkage, and the thumb has two such
followers. Upstream records this as `<mimic>`, with these values (identical for
both hands):

| follower | driver | multiplier | offset |
|---|---|---|---|
| `*_intermediate_joint` (four fingers) | `*_proximal_joint` | 1.06399 | −0.04545 |
| `thumb_intermediate_joint` | `thumb_proximal_pitch_joint` | 1.334 | 0 |
| `thumb_distal_joint` | `thumb_proximal_pitch_joint` | 0.667 | 0 |

A `<mimic>` is a statement, not a mechanism — nothing in ROS propagates it at
runtime. So the same numbers appear in `inspire_hand_driver.kinematics` (which
computes the follower angles the driver publishes) and in the MuJoCo equality
constraints that `inspire_franka_sim/scripts/make_hand_mjcf.py` emits.
`test/test_mimic_matches_driver.py` fails if the copies drift apart.

Two details worth knowing before trusting the numbers:

- The finger multiplier is **not** what you get by assuming a follower reaches
  its limit exactly when its driver does — that gives 1.09214. At the closed
  pose the follower stops at 1.5186 rad, about 2.4° inside its own 1.56 rad
  limit. The limits are loose; the multiplier is the linkage.
- The thumb multipliers are rounded (1.334 and 0.667 rather than 4/3 and 2/3),
  which *overshoots* both follower limits by 4e-4 rad at the closed pose. The
  driver clamps to the limits, because otherwise TF would report a joint outside
  the range the description declares while MuJoCo clamped it — and the two would
  disagree about where the thumb is.

## Known gaps

- **No joint dynamics.** Upstream declares no damping or friction anywhere, and
  every joint carries `effort="1"` / `velocity="0.5"`, which are placeholders
  rather than measurements. The simulation supplies its own; see
  `inspire_franka_sim`.
- **Collision meshes are not convex.** MuJoCo takes the convex hull of a mesh
  geom automatically, so this costs nothing there, but a planner using the raw
  meshes will see fingers slightly fatter than they are.
- **The palm has no collision mesh**, only a cylinder and seven boxes. They are a
  reasonable envelope, not the true shell.
