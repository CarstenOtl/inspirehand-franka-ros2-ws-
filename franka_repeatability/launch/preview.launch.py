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

"""Visualise the configured poses in RViz, with no robot and no ros2_control.

Brings up just enough to see something: robot_state_publisher for the model, move_group for
compute_ik, and RViz. Deliberately no controller_manager - the FR3 MoveIt setup drives the arm
through *effort* interfaces, and mock_components has no dynamics, so under fake hardware a
trajectory controller holds the model perfectly still no matter what you send it. Here
preview_poses.py publishes joint states directly instead, so the model actually moves.

    ros2 launch franka_repeatability preview.launch.py
    ros2 run franka_repeatability preview_poses.py --namespace "" --drive
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    load_gripper = LaunchConfiguration('load_gripper')
    ee_id = LaunchConfiguration('ee_id')

    franka_xacro = os.path.join(
        get_package_share_directory('franka_bringup'), 'urdf', 'franka_arm.urdf.xacro'
    )
    robot_description = {
        'robot_description': ParameterValue(
            Command(
                [
                    FindExecutable(name='xacro'),
                    ' ',
                    franka_xacro,
                    ' hand:=',
                    load_gripper,
                    ' robot_type:=fr3 robot_ip:=preview-only',
                    ' ee_id:=',
                    ee_id,
                    ' use_fake_hardware:=true fake_sensor_commands:=false',
                ]
            ),
            value_type=str,
        )
    }

    rviz_config = os.path.join(
        get_package_share_directory('franka_repeatability'), 'rviz', 'repeatability.rviz'
    )

    return LaunchDescription(
        [
            # Defaults match the measurement configuration: no gripper, so the group tip and the
            # frame the poses refer to are both fr3_link8.
            DeclareLaunchArgument('load_gripper', default_value='false'),
            DeclareLaunchArgument('ee_id', default_value='none'),
            DeclareLaunchArgument('use_rviz', default_value='true'),
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                parameters=[robot_description],
                output='screen',
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('franka_fr3_moveit_config'),
                        'launch',
                        'move_group.launch.py',
                    )
                ),
                launch_arguments={
                    'robot_ip': 'preview-only',
                    'namespace': '',
                    'load_gripper': load_gripper,
                    'use_fake_hardware': 'true',
                    'fake_sensor_commands': 'false',
                    'use_rviz': 'false',
                    'arm_prefix': '',
                }.items(),
            ),
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_config],
                output='screen',
            ),
        ]
    )
