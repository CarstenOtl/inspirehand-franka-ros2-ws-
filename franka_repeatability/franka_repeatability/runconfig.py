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

"""Loading and validation of the run configuration."""

import os

import yaml

JOINT_NAME_TEMPLATE = '{prefix}{robot_type}_joint{index}'


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
        get_package_share_directory('franka_repeatability'), 'config', 'repeatability.yaml'
    )


def load_config(path):
    with open(path, 'r') as handle:
        config = yaml.safe_load(handle)

    if not config.get('poses'):
        raise ValueError('%s defines no poses' % path)

    for index, pose in enumerate(config['poses']):
        pose.setdefault('name', 'P%d' % (index + 1))
        if len(pose.get('position', [])) != 3:
            raise ValueError('pose %r needs a position of three numbers' % pose['name'])
        if len(pose.get('orientation', [])) != 4:
            raise ValueError(
                'pose %r needs an orientation quaternion of four numbers (x, y, z, w)'
                % pose['name']
            )

    names = [pose['name'] for pose in config['poses']]
    if len(set(names)) != len(names):
        raise ValueError('pose names must be unique, got %r' % names)

    ik_mode = config.setdefault('ik_mode', 'cached')
    if ik_mode not in ('cached', 'per_visit'):
        raise ValueError("ik_mode must be 'cached' or 'per_visit', got %r" % ik_mode)

    config.setdefault('namespace', '')
    config.setdefault('controller_name', 'repeatability_ik_controller')
    config.setdefault('base_frame', 'fr3_link0')
    config.setdefault('ee_link', 'fr3_link8')
    config.setdefault('robot_type', 'fr3')
    config.setdefault('arm_prefix', '')
    config.setdefault('cycles', 10)

    timing = config.setdefault('timing', {})
    timing.setdefault('motion_duration', 5.0)
    timing.setdefault('settle_seconds', 1.5)
    timing.setdefault('dwell_seconds', 2.0)

    recording = config.setdefault('recording', {})
    recording.setdefault('storage_id', 'sqlite3')
    recording.setdefault(
        'topics',
        [
            'franka_robot_state_broadcaster/robot_state',
            '%s/controller_state' % config['controller_name'],
        ],
    )

    config['output_dir'] = os.path.expanduser(
        config.get('output_dir', '~/franka_repeatability_runs')
    )
    config['joint_names'] = joint_names(config['robot_type'], config['arm_prefix'])

    if config['cycles'] < 2:
        raise ValueError('cycles must be at least 2 for repeatability to mean anything')

    return config
