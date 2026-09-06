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
Bring up the FR3 (real or fake hardware) with the trajectory replay controller.

This does not move the arm. The controller activates and holds position; everything else is
driven from replay_trajectory.py in a second terminal:

    ros2 launch franka_trajectory_replay replay.launch.py \
        robot_config_file:=/ros2_ws/src/franka_bringup/config/tekken.config.yaml

    ros2 run franka_trajectory_replay replay_trajectory.py ~/trajs/policy.npz

Pass controllers_yaml:=.../controllers_effort.yaml for the effort (joint impedance law) mode.
The robot config file decides between real and fake hardware (use_fake_hardware).
"""

import franka_bringup.launch_utils as launch_utils
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

CONTROLLER_NAME = 'trajectory_replay_controller'


def generate_robot_nodes(context):
    robot_config_file = LaunchConfiguration('robot_config_file').perform(context)
    controllers_yaml = LaunchConfiguration('controllers_yaml').perform(context)
    robot_ips = LaunchConfiguration('robot_ips').perform(context)
    configs = launch_utils.load_yaml(robot_config_file)
    nodes = []
    for index, (_, config) in enumerate(configs.items()):
        namespace = str(config.get('namespace', ''))
        robot_ip = (
            launch_utils.get_parameter_for_config(robot_ips, num_configs=len(configs), config_index=index)
            if robot_ips else str(config['robot_ip'])
        )
        nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([FindPackageShare('franka_bringup'), 'launch', 'franka.launch.py'])),
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
        ))
        nodes.append(Node(
            package='controller_manager',
            executable='spawner',
            namespace=namespace,
            arguments=[CONTROLLER_NAME, '--controller-manager-timeout', '30'],
            parameters=[controllers_yaml],
            output='screen',
        ))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_config_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('franka_bringup'), 'config', 'franka.config.yaml']),
            description='Absolute path to the robot configuration file (franka_bringup opens it '
                        'directly, so it has to be a filesystem path).'),
        DeclareLaunchArgument(
            'controllers_yaml',
            default_value=PathJoinSubstitution(
                [FindPackageShare('franka_trajectory_replay'), 'config', 'controllers.yaml']),
            description='Controller manager configuration (controllers.yaml: position mode, '
                        'controllers_effort.yaml: effort mode).'),
        DeclareLaunchArgument('robot_ips', default_value='',
                              description='Override robot_ip from the robot configuration file.'),
        OpaqueFunction(function=generate_robot_nodes),
    ])
