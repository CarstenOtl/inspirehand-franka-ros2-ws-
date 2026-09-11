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

"""Loading and validation of the run configuration (config/replay.yaml)."""

import copy
import os

import yaml

JOINT_NAME_TEMPLATE = '{prefix}{robot_type}_joint{index}'

DEFAULTS = {
    'namespace': 'NS_1',
    'robot_type': 'fr3',
    'arm_prefix': '',
    'controller_name': 'trajectory_replay_controller',
    'controller_manager': 'controller_manager',
    'joint_state_topic': 'joint_states',
    'tcp': {
        'frame': 'fr3_link8',
        'offset_xyz': [0.0, 0.0, 0.0],
        'offset_rpy': [0.0, 0.0, 0.0],
    },
    'prepare': {
        'rate': 1000,
        'interpolation': 'cubic',
        'blend_time': 0.04,
        'cutoff_hz': 0.0,
        'hold_start': 0.5,
        'hold_end': 0.5,
        'lead_in': 0.5,
        'lead_out': 0.5,
        'lead_max_acceleration': 2.5,
        'time_scale': 1.0,
        'auto_scale': True,
        'velocity_margin': 0.8,
        'acceleration_margin': 0.5,
        'jerk_margin': 0.5,
        'send_rate': 1000,
    },
    'cartesian': {
        'controller_name': 'cartesian_trajectory_replay_controller',
        'base_frame': 'fr3_link0',
        'robot_state_topic': 'franka_robot_state_broadcaster/robot_state',
        'velocity_margin': 1.0,
        'acceleration_margin': 0.5,
        'jerk_margin': 0.5,
        'send_rate': 1000,
        'tool_tolerance_m': 1e-4,
        'tool_tolerance_rad': 1e-4,
        'fk_tolerance_m': 0.003,
        'fk_tolerance_deg': 0.5,
    },
    'recording': {
        'storage_id': 'sqlite3',
        'topics': [
            'trajectory_replay_controller/controller_state',
            'trajectory_replay_controller/status',
            'franka_robot_state_broadcaster/robot_state',
            'joint_states',
        ],
        'required': ['trajectory_replay_controller/controller_state'],
    },
    'timing': {
        'settle_seconds': 1.0,
        'cycles': 1,
        'goto_timeout': 60.0,
    },
    'output_dir': '~/franka_replay_runs',
}


def joint_names(robot_type='fr3', arm_prefix=''):
    prefix = '' if not arm_prefix else arm_prefix + '_'
    return [
        JOINT_NAME_TEMPLATE.format(prefix=prefix, robot_type=robot_type, index=i)
        for i in range(1, 8)
    ]


def namespaced(namespace, *parts):
    """Join a namespace and topic/service segments into an absolute name."""
    namespace = (namespace or '').strip('/')
    segments = [str(part).strip('/') for part in parts if part]
    return '/' + '/'.join(([namespace] if namespace else []) + segments)


def default_config_path():
    from ament_index_python.packages import get_package_share_directory

    return os.path.join(
        get_package_share_directory('franka_trajectory_replay'), 'config', 'replay.yaml'
    )


def _merge(base, override):
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path=None):
    """Load replay.yaml on top of the defaults. ``path=None`` means the packaged file."""
    loaded = {}
    if path is None:
        try:
            path = default_config_path()
        except Exception:  # noqa: BLE001 - outside a ROS install the defaults are enough
            path = None
    if path:
        with open(os.path.expanduser(path), 'r') as handle:
            loaded = yaml.safe_load(handle) or {}
    config = _merge(DEFAULTS, loaded)
    config['joint_names'] = joint_names(config['robot_type'], config['arm_prefix'])
    config['output_dir'] = os.path.expanduser(config['output_dir'])
    config['config_path'] = path
    return config
