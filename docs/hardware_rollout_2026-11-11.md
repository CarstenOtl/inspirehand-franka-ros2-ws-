# Hardware rollouts 2026-11-11: transition clip on the real FR3

Written for: the training team.

## Setup

- **Student:** `m24-dp3-exp20r01-proprio-friction-rollback3-clock-force-nohv-da-s0-20261110-120354-491945`,
  task profile `mimic_windback45_tilt30` (yaw seam 280°).
- **Clip:**
  - `policy` phase: unchanged, 0.02 m / 0.097 rad, tracking-fault limits 0.08 m / 0.35 rad.
  - `follow_waypoints` and `return_to_reset`: 0.05 m / 0.25 rad, fault limits 0.1 m / 0.5 rad.
  - Applied per goal on every 1 kHz controller tick, against the measured grasp pose, per Isaac Euler angle.
- **Controller:**
  - Gains 565 N/m and 28 N·m/rad, damping 2√K, null-space stiffness 10.
  - Friction feedforward 0.3 N·m on every joint.
- **Payload:** Desk payload 0.579 kg (hand + cable).
- **Return gate:** 20 mm / 10° (15 mm in run 1). A cycle's return ends the moment both are met.
- **Logging:** 1 kHz controller-state bags for runs 1–3. Runs 4–6 have the 15 Hz rollout log only.

## Results

| Run | Re-home between cycles | Cycles | Return time | Where the return ended |
|---|---|---|---|---|
| 1 | no | 1 of 2 | 2.5 s; cycle 2 timed out after 16.9 s | 13.3 mm / 9.8°; cycle 2 stuck at 15.2 mm / 6.7° |
| 2 | no | 3/3 | 2.3–2.7 s | 13.0 → 15.6 → 17.3 mm, at about 9.8° |
| 3 | yes | 3/3 | 2.3–2.5 s | 12.5 / 12.9 / 13.2 mm |
| 4 | yes | 6/6 | 2.3–2.5 s | 12.7–13.5 mm, 9.5–9.8° |
| 5 | yes | 3/3 | 2.0–2.5 s | 12.9–13.3 mm |
| 6 | yes | stopped in cycle 1 | — | grip failure during the turn (see below) |

- **No tracking fault and no missed control deadlines.** Each run missed at most 3 of about 500–1000 policy steps.
- **With re-homing, 12 of 12 cycles completed.** Each re-home is a joint-space move to the reset joints:
  - it takes 6.2 s and lands 2.4–3.1 mm / 1.1–1.4° from the reset pose;
  - 8 of 9 re-homes worked on the first attempt; one took two (12.2 s).

## Findings

**1. The transition clip fixes the orientation stall.**
- On 2026-11-10, with the 0.097 rad clip, orientation stuck at 16–20° during `return_to_reset`.
- With 0.25 rad, orientation reaches the 10° gate within 2.0–2.7 s every time.

**2. Position stalls about 13 mm short of the reset pose, and this is not the policy's aim.**
- The student's return goal is within 0.5 mm / 0.5° of the fixed reset pose in every cycle, so the per-episode reset fix is in effect.
- The arm stops on the same side every time (base frame about −12, +6, −2 mm). While it is stopped:
  - the spring force pulling it toward the goal is 6.3–6.8 N, spread as 0.5–1.3 N·m over the joints, mostly j1, j4 and j5;
  - the joints are not moving.
- **That is more than training's dead-zone maximum.** 13 mm × 565 N/m ≈ 7 N, against 5 N translational.
- **Orientation probably exceeds it too.** In run 1 cycle 2, the return ran 16.9 s, and orientation settled at 6.7°. That is about 3.3 N·m at 28 N·m/rad, against 1 N·m rotational.
- **Caveat:** both numbers come from one pose. The static spring load also contains any payload or gravity model error, so they are an upper bound on friction alone.

**3. The arm's posture drifts within each cycle.**
- Over the policy phase and the transitions, j1 turns about −0.2 rad and j3 about +0.2 rad, with smaller changes on j4 and j6. The grasp pose stays put during this.
- Without re-homing the drift carries into the next cycle, and the stall grows: 13 → 15.6 → 17.3 mm in run 2. The spring force at the stall rose from 6.8 to 9.1 N with no motion, so the friction the arm has to overcome grows with the drift.
- With re-homing the stall stays at 12.5–13.5 mm over 12 cycles.

**4. The fault margin is thin.**
- The peak error against the clipped target was 0.41–0.44 rad against the 0.5 rad limit, always in `follow_waypoints`. Your Isaac worst case was 0.42 rad.
- It did not get worse over cycles (runs 1–3; runs 4–6 have no 1 kHz log).

**5. Run 6 failed on the grip, not on the controller.**
- About 4.8 s into the turn, the thumb pad unloaded and the fingertip separation closed from 40 mm to 6 mm. The turn proxy reached 27° and then fell back.
- The release then fired and the run was stopped. This is the same failure pattern as on 2026-11-10 (the fingers close past the nut).

## Questions for the training team

1. **Dead-zone randomization.** The real arm holds about 7 N and up to about 3 N·m of static spring load without moving, against training's U(0, 5 N / 1 N·m). Can the range be widened toward about 8 N / 3.5 N·m?
   - We are building a dead-zone probe that measures this at several poses, separated from payload model error. It should give a measured range within days.
2. **Posture drift.** At null-space stiffness 10, does the Isaac arm also turn j1/j3 by about 0.2 rad over a cycle? Does it come back during `return_to_reset`?
   - If not, Isaac holds the posture more firmly than the real arm, and a randomized null-space stiffness or joint friction would cover it.
   - Until this is settled, multi-cycle runs use the joint-space re-home.
3. **Recollection with `cyclic_transition_clip` {transition phases, 0.05 m, 0.25 rad}.** From our side this is good to go. The hardware confirms both the faster return and your 0.42 rad peak error.
