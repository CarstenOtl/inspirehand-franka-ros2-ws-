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

"""
Bring up the FR3 with the repeatability measurement controller.

This deliberately does not start the measurement run. It only brings up the robot, move_group
(needed for compute_ik) and the controller, which holds the arm at whatever configuration it is
in. Start the run separately once the arm is somewhere sensible:

    ros2 launch franka_repeatability repeatability.launch.py \
        robot_config_file:=/ros2_ws/src/franka_bringup/config/tekken.config.yaml

    ros2 run franka_repeatability run_repeatability.py --dry-run   # check the poses first
    ros2 run franka_repeatability run_repeatability.py

Unlike franka_bringup's example launch files this passes its own controllers.yaml through
franka.launch.py's `controllers_yaml` argument, so franka_bringup's copy stays untouched.
"""

import franka_bringup.launch_utils as launch_utils
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

load_yaml = launch_utils.load_yaml

CONTROLLER_NAME = 'repeatability_ik_controller'


def generate_robot_nodes(context):
    robot_config_file = LaunchConfiguration('robot_config_file').perform(context)
    controllers_yaml = LaunchConfiguration('controllers_yaml').perform(context)
    robot_ips = LaunchConfiguration('robot_ips').perform(context)

    configs = load_yaml(robot_config_file)
    nodes = []

    for index, (_, config) in enumerate(configs.items()):
        namespace = str(config.get('namespace', ''))
        robot_ip = (
            launch_utils.get_parameter_for_config(
                robot_ips, num_configs=len(configs), config_index=index
            )
            if robot_ips
            else str(config['robot_ip'])
        )

        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare('franka_bringup'), 'launch', 'franka.launch.py']
                    )
                ),
                launch_arguments={
                    'robot_type': str(config['robot_type']),
                    'arm_prefix': str(config['arm_prefix']),
                    'namespace': namespace,
                    'robot_ip': robot_ip,
                    'load_gripper': str(config['load_gripper']),
                    'use_fake_hardware': str(config['use_fake_hardware']),
                    'fake_sensor_commands': str(config['fake_sensor_commands']),
                    'joint_state_rate': str(config['joint_state_rate']),
                    'controllers_yaml': controllers_yaml,
                }.items(),
            )
        )

        # move_group provides the compute_ik service the controller resolves pose targets with.
        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare('franka_fr3_moveit_config'),
                            'launch',
                            'move_group.launch.py',
                        ]
                    )
                ),
                launch_arguments={
                    'robot_ip': robot_ip,
                    'namespace': namespace,
                    'load_gripper': str(config['load_gripper']),
                    'use_fake_hardware': str(config['use_fake_hardware']),
                    'fake_sensor_commands': str(config['fake_sensor_commands']),
                    'use_rviz': str(config['use_rviz']),
                    'arm_prefix': str(config['arm_prefix']),
                }.items(),
            )
        )

        nodes.append(
            Node(
                package='controller_manager',
                executable='spawner',
                namespace=namespace,
                arguments=[CONTROLLER_NAME, '--controller-manager-timeout', '30'],
                parameters=[controllers_yaml],
                output='screen',
            )
        )

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'robot_config_file',
                default_value=PathJoinSubstitution(
                    [FindPackageShare('franka_bringup'), 'config', 'franka.config.yaml']
                ),
                description=(
                    'Absolute path to the robot configuration file. franka_bringup opens this '
                    'directly, so it has to be a filesystem path, not a package-relative name.'
                ),
            ),
            DeclareLaunchArgument(
                'controllers_yaml',
                default_value=PathJoinSubstitution(
                    [FindPackageShare('franka_repeatability'), 'config', 'controllers.yaml']
                ),
                description='Controller manager configuration, including the measurement gains.',
            ),
            DeclareLaunchArgument(
                'robot_ips',
                default_value='',
                description='Override robot_ip from the robot configuration file.',
            ),
            OpaqueFunction(function=generate_robot_nodes),
        ]
    )
