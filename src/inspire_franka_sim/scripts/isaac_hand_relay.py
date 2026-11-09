#!/usr/bin/env python3
"""The Inspire hand's ROS interface, served the way Isaac Lab drives the hand.

    ros2 run inspire_franka_sim isaac_hand_relay.py

Stands in for the RS485 driver in ``sim_policy.launch.py hand_chain:=isaac``.
The policy runner talks to the same two topics it uses on hardware, but the
hand behind them is the training simulator's:

- ``/inspire_hand/command`` (open ratios, the driver's contract) is converted to
  joint targets with the driver's own ``open_ratio_to_rad`` and sent to the
  position actuators at once. Isaac writes the policy's pinch target straight
  into the joint drive; there is no slew, no register quantisation and no wall
  clock in between, which is what the driver chain (``hand_chain:=driver``:
  mock driver plus ``inspire_hand_sim_bridge.py``) adds.
- ``/inspire_hand/joint_states`` republishes the SIMULATED hand joints from
  ``/joint_states``. Isaac's student observes its finger joints, which stop on
  the nut; the mock driver reports where its slew has got to, contact or not.

The followers are not commanded here: IsaacHandPlugin sets their targets from
the leaders' inside the simulator, as Isaac's ``_expand_hand_mimic_targets``
does.

``hand_model:=rh56`` serves the RH56 linkage hand of the RH56-trained students
(ForgeUltra ``hand_asset_profile: rh56``; the scene must be the RH56 one, whose
IsaacHandPlugin moves the followers along the linkage). Then the relay speaks
the real hand's register interface, as the RS485 driver does:

- a command's open ratio is a register (``1000 * ratio``, rounded as the driver
  does). The register slews at training's nominal rod speed
  (``rh56_linkage.NOMINAL_ROD_SPEED_MM_S`` over the channel's stroke) in sim
  time, and each step's register goes to the leader drive as the angle the
  linkage reaches there (``rh56_linkage``'s joint table);
- the force stop, as ForgeUltra's linkage drive has it: every simulation step
  the rod force is estimated from the torques the channel's drives apply, by
  virtual work (``1000 * sum(tau * dq/dL)``, low-passed with training's 37 ms),
  and a closing rod holds while it exceeds ``force_stop_n`` (released below 90 %
  of it). A finger blocked by the nut therefore stops squeezing at the stop
  instead of driving its position servo into the nut;
- the state is the simulated leaders read back to registers through the same
  table: ``~/state`` carries them as open ratios per channel ``1``..``6``
  (``frame_id`` ``open_ratio``), and ``~/joint_states`` the driver's linear
  conversion of those ratios (``kinematics.joint_positions``), exactly what the
  driver publishes. The simulated joint angles themselves are not reported:
  the hand cannot report them either.

Not modelled: the per-episode randomization of the stop, the rod speed and the
linkage geometry; the nominal values are used.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from inspire_hand_driver import kinematics as kin
from inspire_hand_driver.rh56_linkage import force_filter_alpha

# The forward command controller's joint order (controllers_sim_policy.yaml),
# which is the driver's register order.
DRIVEN_JOINTS = tuple(kin.DRIVEN_JOINTS)
ANGLE_MAX = 1000


class Rh56Registers:
    """The RH56 register side of the relay: registers <-> joint angles along the linkage, and the
    rod's force stop (ForgeUltra ``RH56LinkageDrive.step`` at nominal geometry)."""

    def __init__(self, force_stop_n=None) -> None:
        from inspire_hand_driver.rh56_linkage import (
            DOF_OF_CHANNEL,
            DRIVE_EFFORT_LIMIT_NM,
            DRIVE_GAINS,
            FORCE_RELEASE_FRAC,
            NOMINAL_FORCE_STOP_N,
            NOMINAL_ROD_SPEED_MM_S,
            WORKSPACE_JOINT,
            load_joint_table,
        )

        table = load_joint_table()
        channel_of_dof = {dof: channel for channel, dof in DOF_OF_CHANNEL.items()}
        self.table = table
        self.workspace_joint = WORKSPACE_JOINT
        #: per DOF (register order): the table channel, and its slew rate in register/s
        self.channels = [channel_of_dof[d.name] for d in kin.DOFS]
        self.rates = [
            ANGLE_MAX * NOMINAL_ROD_SPEED_MM_S / table.stroke_mm[c] for c in self.channels
        ]
        #: the drive gains per DOF
        self.gains = [DRIVE_GAINS[table.dof_type[c]] for c in self.channels]
        self.effort_limit = DRIVE_EFFORT_LIMIT_NM
        self.force_stop = float(NOMINAL_FORCE_STOP_N if force_stop_n is None else force_stop_n)
        if not self.force_stop > 0.0:
            raise ValueError("the RH56 force stop must be positive")
        self.release_frac = FORCE_RELEASE_FRAC
        #: the leader's (ForgeUltra) name per DOF, to read the table
        self.leaders = [table.channels[c][0] for c in self.channels]
        for dof, leader in zip(kin.DOFS, self.leaders):
            if WORKSPACE_JOINT[leader] != dof.joint:
                raise RuntimeError(f"RH56 table leader {leader} is not the driver's {dof.joint}")
        self.registers = None
        self.targets = None
        #: per DOF: the low-passed rod force (N, closing positive) and whether the rod is stalled
        self.rod_force = [0.0] * len(self.channels)
        self.stalled = [False] * len(self.channels)

    def registers_of(self, leader_angles):
        return [
            self.table.register_of_leader(c, float(q)) for c, q in zip(self.channels, leader_angles)
        ]

    def leader_angles(self, registers):
        return [
            self.table.joints_at(c, r)[leader]
            for c, r, leader in zip(self.channels, registers, self.leaders)
        ]

    def rod_force_now(self, k: int, positions, velocities) -> float:
        """Unfiltered rod force of DOF ``k`` from the torques its drives apply at the current register.

        ``positions`` / ``velocities``: workspace joint name -> simulated value. The drives are
        MuJoCo position actuators, ``kp (target - q) - kd qdot`` clamped at the effort limit, on
        the targets the linkage gives at the current register (IsaacHandPlugin sets the
        followers'). Virtual work: ``F = 1000 * sum(tau * dq/dL)``, N.
        """

        channel, register = self.channels[k], self.registers[k]
        kp, kd = self.gains[k]
        targets = self.table.joints_at(channel, register)
        jacobian = self.table.dq_dL(channel, register)
        force = 0.0
        for joint, target in targets.items():
            name = self.workspace_joint[joint]
            tau = kp * (target - positions[name]) - kd * velocities.get(name, 0.0)
            tau = min(max(tau, -self.effort_limit), self.effort_limit)
            force += tau * jacobian[joint]
        return 1000.0 * force

    def step(self, dt: float, positions, velocities, filter_alpha: float) -> bool:
        """One simulation step of every rod: force estimate, then slew unless stalled.

        True if any register moved. A rod stalls only while closing (register falling): over the
        stop, or still above ``release_frac`` of it after stalling, as the training drive does.
        """

        moved = False
        for k, (r, t, rate) in enumerate(zip(self.registers, self.targets, self.rates)):
            force = self.rod_force_now(k, positions, velocities)
            self.rod_force[k] = filter_alpha * force + (1.0 - filter_alpha) * self.rod_force[k]
            step = min(max(t - r, -rate * max(dt, 0.0)), rate * max(dt, 0.0))
            closing = step < 0.0
            over = self.rod_force[k] > self.force_stop
            still_loaded = self.stalled[k] and self.rod_force[k] > self.release_frac * self.force_stop
            self.stalled[k] = closing and (over or still_loaded)
            if step != 0.0 and not self.stalled[k]:
                self.registers[k] = r + step
                moved = True
        return moved


class IsaacHandRelay(Node):
    def __init__(self) -> None:
        super().__init__("inspire_hand")
        self.declare_parameter(
            "command_topic", "/hand_position_forward_command_controller/commands"
        )
        # /joint_states arrives at the controller manager's 1 kHz. Republishing
        # all of it buried the runner's Python loop in callbacks (the policy fell
        # to ~5 Hz and tripped the controller watchdog), so publish at the
        # driver's own rate (inspire_hand_driver publish_rate_hz), in sim time.
        self.declare_parameter("publish_rate_hz", 50.0)
        # "mimic": the official hand (joint-angle interface); "rh56": the RH56
        # linkage hand behind its register interface (see the module docstring).
        self.declare_parameter("hand_model", "mimic")
        # hand_model rh56: the rod force stop, N (rh56_linkage.NOMINAL_FORCE_STOP_N).
        self.declare_parameter("force_stop_n", 50.0)
        self._period_ns = int(1.0e9 / float(self.get_parameter("publish_rate_hz").value))
        self._last_ns = None
        self._last_slew_ns = None
        self._targets = None
        hand_model = str(self.get_parameter("hand_model").value)
        if hand_model not in ("mimic", "rh56"):
            raise ValueError(f"hand_model must be 'mimic' or 'rh56', got {hand_model!r}")
        self._rh56 = (
            Rh56Registers(float(self.get_parameter("force_stop_n").value))
            if hand_model == "rh56"
            else None
        )
        self._publisher = self.create_publisher(
            Float64MultiArray, str(self.get_parameter("command_topic").value), 10
        )
        self._state_publisher = self.create_publisher(JointState, "~/joint_states", 10)
        self._register_publisher = self.create_publisher(JointState, "~/state", 10)
        self.create_subscription(JointState, "~/command", self._on_command, 10)
        self.create_subscription(JointState, "/joint_states", self._on_state, 10)
        if self._rh56 is None:
            self.get_logger().info(
                "Isaac hand chain: ~/command open ratios -> joint targets, "
                "/joint_states -> ~/joint_states"
            )
        else:
            self.get_logger().info(
                "RH56 hand chain: ~/command registers -> rod-speed slew with a "
                f"{self._rh56.force_stop:g} N rod force stop -> linkage leader targets; "
                "simulated leaders -> registers -> ~/state, ~/joint_states"
            )

    def _on_state(self, message: JointState) -> None:
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if self._rh56 is not None:
            self._on_state_rh56(message, stamp_ns)
            return
        if (
            self._targets is not None
            and self._last_ns is not None
            and 0 <= stamp_ns - self._last_ns < self._period_ns
        ):
            return
        self._last_ns = stamp_ns
        names = list(message.name)
        hand = [n for n in kin.ALL_JOINTS if n in names]
        if not hand:
            return
        index = {n: i for i, n in enumerate(names)}
        if self._targets is None and all(n in index for n in DRIVEN_JOINTS):
            # Until the first command, hold where the hand is.
            self._targets = [float(message.position[index[n]]) for n in DRIVEN_JOINTS]
        out = JointState()
        out.header = message.header
        out.name = hand
        out.position = [float(message.position[index[n]]) for n in hand]
        if len(message.velocity) == len(names):
            out.velocity = [float(message.velocity[index[n]]) for n in hand]
        self._state_publisher.publish(out)

    def _on_state_rh56(self, message: JointState, stamp_ns: int) -> None:
        rh56 = self._rh56
        index = {n: i for i, n in enumerate(message.name)}
        if not all(n in index for n in DRIVEN_JOINTS):
            return
        leaders = [float(message.position[index[n]]) for n in DRIVEN_JOINTS]
        if rh56.registers is None:
            # Until the first command, hold the registers the hand is at.
            rh56.registers = rh56.registers_of(leaders)
            rh56.targets = list(rh56.registers)
            self._last_slew_ns = stamp_ns
        # The rod slews, and its force stop acts, every simulation step, not at the state rate.
        dt = 1.0e-9 * (stamp_ns - self._last_slew_ns)
        self._last_slew_ns = stamp_ns
        positions = dict(zip(message.name, message.position))
        velocities = (
            dict(zip(message.name, message.velocity))
            if len(message.velocity) == len(message.name)
            else {}
        )
        if rh56.step(dt, positions, velocities, force_filter_alpha(dt)):
            self._publish_targets(rh56.leader_angles(rh56.registers))
        if self._last_ns is not None and 0 <= stamp_ns - self._last_ns < self._period_ns:
            return
        self._last_ns = stamp_ns
        ratios = [r / ANGLE_MAX for r in rh56.registers_of(leaders)]
        # The hand reports integer registers.
        ratios = [round(ANGLE_MAX * r) / ANGLE_MAX for r in ratios]
        state = JointState()
        state.header.stamp = message.header.stamp
        state.header.frame_id = "open_ratio"
        state.name = [d.channel for d in kin.DOFS]
        state.position = ratios
        self._register_publisher.publish(state)
        joints = JointState()
        joints.header.stamp = message.header.stamp
        joints.name = list(kin.ALL_JOINTS)
        joints.position = kin.joint_positions(ratios)
        self._state_publisher.publish(joints)

    def _publish_targets(self, targets) -> None:
        command = Float64MultiArray()
        command.data = [float(v) for v in targets]
        self._publisher.publish(command)

    def _on_command(self, message: JointState) -> None:
        if self._rh56 is not None:
            self._on_command_rh56(message)
            return
        if self._targets is None:
            self.get_logger().warn(
                "command before the first /joint_states; ignored", throttle_duration_sec=2.0
            )
            return
        for name, ratio in zip(message.name, message.position):
            try:
                dof = kin.dof_index(name)
            except KeyError:
                self.get_logger().warn(f"not a driven joint: {name!r}", throttle_duration_sec=2.0)
                continue
            self._targets[DRIVEN_JOINTS.index(kin.DOFS[dof].joint)] = kin.open_ratio_to_rad(
                dof, ratio
            )
        self._publish_targets(self._targets)

    def _on_command_rh56(self, message: JointState) -> None:
        rh56 = self._rh56
        if rh56.registers is None:
            self.get_logger().warn(
                "command before the first /joint_states; ignored", throttle_duration_sec=2.0
            )
            return
        for name, ratio in zip(message.name, message.position):
            try:
                dof = kin.dof_index(name)
            except KeyError:
                self.get_logger().warn(f"not a driven joint: {name!r}", throttle_duration_sec=2.0)
                continue
            # open_ratio_to_angle: the driver writes integer registers.
            rh56.targets[dof] = float(round(min(max(float(ratio), 0.0), 1.0) * ANGLE_MAX))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IsaacHandRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
