#!/usr/bin/env python3
"""Verify the ported OSC -> joint-PD adapter against a recorded teacher episode.

For every policy-phase row of the reference episode the teacher's filtered
native OSC action, joint state, and the resulting 17-D impedance joint-PD
command were recorded. This script recomputes that command from the ported
controller using MuJoCo kinematics (grasp frame, Jacobian, mass matrix) and
reports the residual. Rows in the scripted release/return phases used cubic
joint trajectories instead of the OSC, so they are reported separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = APP_ROOT.parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from policy_rollout import forge_osc as fo  # noqa: E402

DEFAULT_MJCF = WS_ROOT / "assets" / "fr3_inspirehand" / "fr3_inspirehand_replay.xml"
DEFAULT_EPISODE = APP_ROOT / "checkpoints" / "reference_episode" / "episode_000_sequential_threading.npz"
ARM = [f"fr3_joint{i}" for i in range(1, 8)]
POLICY_HAND = ["thumb_joint_0", "thumb_joint_1", "index_joint_0"]
POSTURE_ONLY = ["middle_joint_0", "ring_joint_0", "little_joint_0"]


class MujocoKinematics:
    """Kinematic quantities the controller needs, from the replay MJCF."""

    def __init__(self, mjcf: Path, base_z: float = 0.0):
        import mujoco

        from policy_rollout.mujoco_scene import training_scene_spec

        self.mujoco = mujoco
        self.model = training_scene_spec(mjcf, base_plate_z=base_z).compile()
        self.data = mujoco.MjData(self.model)
        self.jid = {self.model.joint(j).name: j for j in range(self.model.njnt)}
        self.qadr = {n: self.model.jnt_qposadr[j] for n, j in self.jid.items()}
        self.vadr = {n: self.model.jnt_dofadr[j] for n, j in self.jid.items()}
        self.arm_dofs = [self.vadr[n] for n in ARM]
        self.flange = self.model.body("fr3_link8").id
        self.thumb_tip = self.model.body("thumb_tip").id
        self.index_tip = self.model.body("index_tip").id

    def set_state(self, q_arm, dq_arm, hand_pos: dict, hand_vel: dict | None = None):
        d, m = self.data, self.model
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        for n, v in zip(ARM, q_arm):
            d.qpos[self.qadr[n]] = v
        for n, v in zip(ARM, dq_arm):
            d.qvel[self.vadr[n]] = v
        full = fo.expand_hand_mimic(hand_pos)
        for n, v in full.items():
            if n in self.qadr:
                d.qpos[self.qadr[n]] = v
        if hand_vel:
            # followers scale with their leaders
            vel = dict(hand_vel)
            for follower, (leader, mult, _off) in fo.MIMIC_JOINT_MAP.items():
                vel[follower] = vel.get(leader, 0.0) * mult
            for n, v in vel.items():
                if n in self.vadr:
                    d.qvel[self.vadr[n]] = v
        self.mujoco.mj_forward(m, d)

    def body_pose(self, body):
        return self.data.xpos[body].copy(), fo.quat_from_matrix(self.data.xmat[body].reshape(3, 3))

    def body_jacobian(self, body):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        self.mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body)
        return np.vstack((jacp, jacr))

    def body_linvel(self, body):
        return self.body_jacobian(body)[0:3] @ self.data.qvel

    def arm_mass_matrix(self):
        from policy_rollout.mujoco_scene import full_mass_matrix

        return full_mass_matrix(self.model, self.data)[np.ix_(self.arm_dofs, self.arm_dofs)]

    def grasp(self, z_transport):
        thumb, _ = self.body_pose(self.thumb_tip)
        index, _ = self.body_pose(self.index_tip)
        flange_pos, flange_quat = self.body_pose(self.flange)
        J_flange = self.body_jacobian(self.flange)
        return fo.grasp_frame_state(
            thumb_pos=thumb,
            index_pos=index,
            flange_pos=flange_pos,
            flange_quat=flange_quat,
            z_transport=z_transport,
            thumb_linvel=self.body_linvel(self.thumb_tip),
            index_linvel=self.body_linvel(self.index_tip),
            flange_angvel=J_flange[3:6] @ self.data.qvel,
            flange_jacobian=J_flange[:, self.arm_dofs],
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", default=str(DEFAULT_EPISODE))
    parser.add_argument("--mjcf", default=str(DEFAULT_MJCF))
    parser.add_argument("--dead-zone", choices=["none", "default"], default="none")
    parser.add_argument("--base-z", type=float, default=0.0, help="world z of the FR3 base plate")
    parser.add_argument("--tilt-deg", type=float, default=0.0, help="reset grasp-Z outward tilt")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    kin = MujocoKinematics(Path(args.mjcf), base_z=args.base_z)
    ep = np.load(args.episode, allow_pickle=False)
    proprio = ep["student_osc_proprio"].astype(float)
    filtered = ep["osc_filtered_action"].astype(float)
    pd = ep["impedance_joint_pd_command"].astype(float)
    phase = ep["replay_phase"]
    n = len(proprio)

    # Reset-time grasp Z transport: reset arm joints, closed hand posture.
    posture = dict(fo.THREADING_GRASP_POSTURE)
    kin.set_state(fo.FRANKA_ARM_RESET_JOINTS_M24, np.zeros(7), posture)
    thumb, _ = kin.body_pose(kin.thumb_tip)
    index, _ = kin.body_pose(kin.index_tip)
    flange_pos, flange_quat = kin.body_pose(kin.flange)
    z_transport = fo.reset_z_transport(thumb, index, flange_pos, flange_quat, tilt_deg=args.tilt_deg)

    dead_zone = fo.DEFAULT_DEAD_ZONE if args.dead_zone == "default" else None
    predicted = np.zeros((n, 17))
    torque = np.zeros((n, 7))
    grasp_pos = np.zeros((n, 3))
    for t in range(n):
        q = proprio[t, :7]
        dq = proprio[t, 10:17]
        hand_pos = dict(posture)
        hand_vel = {}
        for k, name in enumerate(POLICY_HAND):
            hand_pos[name] = proprio[t, 7 + k]
            hand_vel[name] = proprio[t, 17 + k]
        kin.set_state(q, dq, hand_pos, hand_vel)
        grasp = kin.grasp(z_transport)
        grasp_pos[t] = grasp.pos
        target = fo.decode_action_target(filtered[t], grasp)
        tau, _wrench, _pe, _ae = fo.compute_dof_torque(
            dof_pos_arm=q,
            dof_vel_arm=dq,
            grasp=grasp,
            arm_mass_matrix=kin.arm_mass_matrix(),
            target_pos=target.pos,
            target_quat=target.quat,
            dead_zone_thresholds=dead_zone,
        )
        torque[t] = tau
        predicted[t] = fo.joint_pd_command_from_torque(q, dq, tau, fo.pinch_targets(filtered[t]))

    def stats(mask, shift):
        rows = np.where(mask)[0]
        rows = rows[rows + shift < n]
        if len(rows) == 0:
            return None
        rec = pd[rows + shift]
        pre = predicted[rows]
        # The arm position target encodes tau/Kp; report it in torque units too.
        arm_err = pre[:, :7] - rec[:, :7]
        tau_rec = (rec[:, :7] - proprio[rows, :7]) * fo.IMPEDANCE_JOINT_PD_ARM_STIFFNESS
        tau_err = torque[rows] - tau_rec
        return {
            "rows": int(len(rows)),
            "arm_position_target_mae_rad": float(np.abs(arm_err).mean()),
            "arm_position_target_max_rad": float(np.abs(arm_err).max()),
            "arm_torque_mae_nm": float(np.abs(tau_err).mean()),
            "arm_torque_rms_nm": float(np.sqrt((tau_err**2).mean())),
            "arm_torque_mae_per_joint_nm": np.abs(tau_err).mean(0).round(3).tolist(),
            "recorded_torque_rms_nm": float(np.sqrt((tau_rec**2).mean())),
            "arm_velocity_target_mae_rad_s": float(np.abs(pre[:, 7:14] - rec[:, 7:14]).mean()),
            "hand_target_mae_rad": float(np.abs(pre[:, 14:17] - rec[:, 14:17]).mean()),
            "hand_target_max_rad": float(np.abs(pre[:, 14:17] - rec[:, 14:17]).max()),
            "torque_r2_per_joint": [
                float(1.0 - ((tau_err[:, j]) ** 2).sum() / max(((tau_rec[:, j] - tau_rec[:, j].mean()) ** 2).sum(), 1e-9))
                for j in range(7)
            ],
        }

    policy = phase == "policy"
    report = {
        "episode": str(args.episode),
        "dead_zone": args.dead_zone,
        "base_z": args.base_z,
        "tilt_deg": args.tilt_deg,
        "policy_rows_same_row_alignment": stats(policy, 0),
        "policy_rows_next_row_alignment": stats(policy, 1),
        "scripted_rows_same_row_alignment": stats(~policy, 0),
        "grasp_origin_first_row_world_m": grasp_pos[0].round(4).tolist(),
        "bolt_tip_world_m": fo.BOLT_TIP_POSITION.round(4).tolist(),
        "reset_z_transport_flange_frame": z_transport.round(4).tolist(),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
