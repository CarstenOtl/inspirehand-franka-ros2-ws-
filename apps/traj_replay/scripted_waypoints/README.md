# Scripted threading waypoints

These are vendored snapshots of ForgeUltra's FR3 sequential-threading release,
retreat, and reset recipe for reproducible trajectory-replay testing.

- `waypoints_release_and_reset_franka_20260905_traj1.yaml` is copied byte-for-byte
  from ForgeUltra commit `82461f8` (Git blob
  `fb6eaeabc21625d789f1d8191b4292d144515bb7`). This is the historical recipe
  contemporary with the September 5 `traj_1` recording and its validated
  `threading_*_flange180` derivatives.
- `waypoints_release_and_reset_franka_20260910_current.yaml` is copied
  byte-for-byte from the current ForgeUltra checkout at commit
  `bc77a941a333768da5bb47c64085174f4b2dbf13` (Git blob
  `022df0233a42864bb0c7c9466ba761938a8dddda`). The waypoint file itself was
  last changed on September 10.

Do not silently substitute one snapshot for the other. The validated replay
artifacts contain measured executions of the historical recipe; the current
recipe has different arm targets and a different declared trajectory space.
A future generator or test should record the selected snapshot's filename and
hash in its output metadata.

`repeat_policy_trajectory` uses the September 10 joint-PD snapshot by default
and copies the exact selected recipe into every generated artifact.
