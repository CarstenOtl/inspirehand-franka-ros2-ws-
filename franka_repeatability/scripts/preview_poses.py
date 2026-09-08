#!/usr/bin/env python3
# Copyright (c) 2026 Agile Robots SE
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Show the configured poses in RViz, and optionally drive the arm through them.

Publishes an axis triad and label for every pose plus the cycle path between them, latched, so
RViz picks them up whenever it connects. With --drive it also animates the robot model through
the poses, using the same quintic joint-space blend the measurement controller applies - so what
you watch in RViz is the path the arm will actually take, bulges included, not a planned one.

This drives the *model*, by publishing joint states. It does not command hardware. Use
goto_pose.py for that. Run it against preview.launch.py, which brings up robot_state_publisher,
move_group and RViz without any ros2_control stack to compete with.
"""

import argparse
import sys
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

from franka_repeatability.runconfig import default_config_path, load_config, namespaced
from franka_repeatability.target_client import TargetClient

READY = [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]

LATCHED = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)

AXES = (
    ((0.0, 0.0, 0.0, 1.0), (1.0, 0.2, 0.2)),      # x, red
    ((0.0, 0.0, 0.7071, 0.7071), (0.2, 1.0, 0.2)),  # y, green
    ((0.0, -0.7071, 0.0, 0.7071), (0.2, 0.4, 1.0)),  # z, blue
)


def quaternion_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


class PosePreview(TargetClient):
    def __init__(self, config):
        super().__init__(config, node_name='preview_poses')
        self._markers = self.create_publisher(MarkerArray, '~/poses', LATCHED)
        # robot_state_publisher listens here directly. preview.launch.py runs no ros2_control and
        # no joint_state_publisher, so nothing else writes this topic and the model follows us.
        self._joint_states = self.create_publisher(
            JointState, namespaced(config['namespace'], 'joint_states'), 10
        )
        self._displayed = list(READY)
        self.create_timer(1.0 / 30.0, self._publish_joint_states)

    def _publish_joint_states(self):
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(self.config['joint_names'])
        message.position = [float(value) for value in self._displayed]
        self._joint_states.publish(message)

    def build_markers(self, axis_length=0.08):
        array = MarkerArray()
        frame = self.config['base_frame']
        now = self.get_clock().now().to_msg()
        marker_id = 0

        for pose in self.config['poses']:
            position = pose['position']
            orientation = tuple(pose['orientation'])

            for rotation, colour in AXES:
                arrow = Marker()
                arrow.header.frame_id = frame
                arrow.header.stamp = now
                arrow.ns = 'axes'
                arrow.id = marker_id
                marker_id += 1
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                (
                    arrow.pose.position.x,
                    arrow.pose.position.y,
                    arrow.pose.position.z,
                ) = position
                combined = quaternion_multiply(orientation, rotation)
                (
                    arrow.pose.orientation.x,
                    arrow.pose.orientation.y,
                    arrow.pose.orientation.z,
                    arrow.pose.orientation.w,
                ) = combined
                arrow.scale.x = axis_length
                arrow.scale.y = axis_length * 0.12
                arrow.scale.z = axis_length * 0.12
                arrow.color.r, arrow.color.g, arrow.color.b = colour
                arrow.color.a = 1.0
                array.markers.append(arrow)

            label = Marker()
            label.header.frame_id = frame
            label.header.stamp = now
            label.ns = 'labels'
            label.id = marker_id
            marker_id += 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = position[0]
            label.pose.position.y = position[1]
            label.pose.position.z = position[2] + 0.06
            label.pose.orientation.w = 1.0
            label.scale.z = 0.04
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = '%s  [%.0f, %.0f, %.0f] mm' % (
                pose['name'],
                position[0] * 1000,
                position[1] * 1000,
                position[2] * 1000,
            )
            array.markers.append(label)

        # The cycle order, closed back to the first pose - this is the path the arm takes.
        path = Marker()
        path.header.frame_id = frame
        path.header.stamp = now
        path.ns = 'cycle'
        path.id = marker_id
        path.type = Marker.LINE_STRIP
        path.action = Marker.ADD
        path.pose.orientation.w = 1.0
        path.scale.x = 0.004
        path.color.r, path.color.g, path.color.b, path.color.a = (1.0, 0.8, 0.1, 0.7)
        for pose in self.config['poses'] + [self.config['poses'][0]]:
            point = Marker().pose.position.__class__()
            point.x, point.y, point.z = pose['position']
            path.points.append(point)
        array.markers.append(path)

        return array

    def publish_markers(self):
        array = self.build_markers()
        self._markers.publish(array)
        return len(array.markers)

    def solve_all(self):
        """One IK solution per pose, seeded exactly as the cached measurement run seeds it."""
        solutions = []
        seed = list(self._displayed)
        for pose in self.config['poses']:
            solution = self.solve_ik(pose, seed)
            self.get_logger().info(
                '%s -> %s' % (pose['name'], ' '.join('%+.3f' % value for value in solution))
            )
            solutions.append(solution)
            seed = solution
        return solutions

    @staticmethod
    def _blend(s):
        # The controller's ramp, repeated here so the preview shows the real transition path.
        return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))

    def move_to(self, target, seconds, rate=50.0):
        start = list(self._displayed)
        steps = max(int(seconds * rate), 1)
        for step in range(1, steps + 1):
            fraction = self._blend(step / steps)
            self._displayed = [
                a + (b - a) * fraction for a, b in zip(start, target)
            ]
            time.sleep(1.0 / rate)
        self._displayed = list(target)

    def drive(self, solutions, seconds_per_move, settle):
        for pose, solution in zip(self.config['poses'], solutions):
            self.get_logger().info('moving to %s' % pose['name'])
            self.move_to(solution, seconds_per_move)
            time.sleep(settle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None, help='run configuration YAML')
    parser.add_argument(
        '--namespace',
        default=None,
        help='override the namespace; pass "" for franka_fr3_moveit_config/moveit.launch.py',
    )
    parser.add_argument(
        '--drive',
        action='store_true',
        help='also animate the robot model through the poses (visualisation only)',
    )
    parser.add_argument('--loops', type=int, default=1, help='how many times to drive the cycle')
    parser.add_argument('--seconds', type=float, default=4.0, help='seconds per move when driving')
    parser.add_argument('--settle', type=float, default=1.0, help='pause at each pose when driving')
    args = parser.parse_args()

    config = load_config(args.config or default_config_path())
    if args.namespace is not None:
        config['namespace'] = args.namespace

    rclpy.init()
    node = PosePreview(config)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        count = node.publish_markers()
        topic = node.get_namespace().rstrip('/') + '/preview_poses/poses'
        print(
            '\nPublished %d markers for %s on %s (latched).\n'
            'Add a MarkerArray display on that topic in RViz, with Fixed Frame %s.\n'
            % (
                count,
                ', '.join(pose['name'] for pose in config['poses']),
                topic,
                config['base_frame'],
            )
        )

        if args.drive:
            solutions = node.solve_all()
            for loop in range(args.loops):
                print('animating the cycle, pass %d of %d' % (loop + 1, args.loops))
                node.drive(solutions, args.seconds, args.settle)
            print('done - holding the last pose, Ctrl-C to stop')
            while rclpy.ok():
                time.sleep(0.5)
        else:
            print('Holding to keep the markers latched. Ctrl-C to stop.')
            while rclpy.ok():
                time.sleep(0.5)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as error:  # noqa: BLE001 - report and exit rather than dump a traceback
        print('failed: %s' % error, file=sys.stderr)
        return 1
    finally:
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
