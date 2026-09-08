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
The same ROS stack as replay.launch.py, with MuJoCo (franka_mujoco_hardware) in place of the
arm. Same namespace, same controller, same topics - so replay_trajectory.py runs unchanged:

    ros2 launch franka_trajectory_replay sim.launch.py
    ros2 run franka_trajectory_replay replay_trajectory.py ~/trajs/policy.npz --yes

Arguments: namespace (NS_1), initial_positions (FR3 ready pose), reflex_on_violation (true:
a motion generator limit violation stops the hardware like the real arm would),
use_rviz (false).
"""

import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CONTROLLER_NAME = 'trajectory_replay_controller'


def generate_nodes(context):
    namespace = LaunchConfiguration('namespace').perform(context)
    controllers_yaml = LaunchConfiguration('controllers_yaml').perform(context)
    mujoco_share = get_package_share_directory('franka_mujoco_hardware')
    robot_description = xacro.process_file(
        os.path.join(mujoco_share, 'urdf', 'fr3_mujoco.urdf.xacro'),
        mappings={
            'robot_type': 'fr3',
            'hand': 'false',
            'ee_id': 'none',
            'model_path': LaunchConfiguration('model_path').perform(context),
            'initial_positions': LaunchConfiguration('initial_positions').perform(context),
            'reflex_on_violation': LaunchConfiguration('reflex_on_violation').perform(context),
            'stiffness': LaunchConfiguration('stiffness').perform(context),
            'damping': LaunchConfiguration('damping').perform(context),
        },
    ).toprettyxml(indent='  ')

    nodes = [
        Node(package='robot_state_publisher', executable='robot_state_publisher', namespace=namespace,
             parameters=[{'robot_description': robot_description}], output='screen'),
        Node(package='controller_manager', executable='ros2_control_node', namespace=namespace,
             parameters=[controllers_yaml, {'robot_description': robot_description}],
             output='screen'),
        Node(package='controller_manager', executable='spawner', namespace=namespace,
             arguments=['joint_state_broadcaster', '--controller-manager-timeout', '30'], output='screen'),
        Node(package='controller_manager', executable='spawner', namespace=namespace,
             arguments=[CONTROLLER_NAME, '--controller-manager-timeout', '30'],
             parameters=[controllers_yaml], output='screen'),
        Node(package='rviz2', executable='rviz2', name='rviz2', namespace=namespace,
             arguments=['--display-config', os.path.join(
                 get_package_share_directory('franka_description'), 'rviz', 'visualize_franka.rviz')],
             condition=IfCondition(LaunchConfiguration('use_rviz')), output='screen'),
    ]
    return nodes


def generate_launch_description():
    mujoco_share = get_package_share_directory('franka_mujoco_hardware')
    replay_share = get_package_share_directory('franka_trajectory_replay')
    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='NS_1'),
        DeclareLaunchArgument('controllers_yaml',
                              default_value=os.path.join(replay_share, 'config', 'controllers_sim.yaml')),
        DeclareLaunchArgument('model_path', default_value=os.path.join(mujoco_share, 'mujoco', 'scene.xml')),
        DeclareLaunchArgument('initial_positions', default_value='0 -0.785398 0 -2.356194 0 1.570796 0.785398',
                              description='7 joint positions the simulated arm starts in'),
        DeclareLaunchArgument('reflex_on_violation', default_value='true'),
        DeclareLaunchArgument('stiffness', default_value='3000 3000 3000 2500 2500 2000 2000',
                              description='emulated internal joint impedance stiffness [Nm/rad]'),
        DeclareLaunchArgument('damping', default_value='60 60 60 50 40 20 15'),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        OpaqueFunction(function=generate_nodes),
    ])
