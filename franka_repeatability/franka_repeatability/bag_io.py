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

"""Reading the rosbag2 recording produced by a run."""

import os

import yaml


def stamp_to_ns(stamp):
    return int(stamp.sec) * 1000000000 + int(stamp.nanosec)


def storage_id_from_metadata(bag_dir, fallback='sqlite3'):
    metadata_path = os.path.join(str(bag_dir), 'metadata.yaml')
    try:
        with open(metadata_path, 'r') as handle:
            metadata = yaml.safe_load(handle)
        return metadata['rosbag2_bagfile_information']['storage_identifier']
    except (OSError, KeyError, TypeError):
        return fallback


def read_messages(bag_dir, topics=None, storage_id=None):
    """Yield ``(topic, message, receive_time_ns)`` for every message in the bag.

    Deserialisation happens here rather than during the run: at 1 kHz a Python subscriber drops
    samples, which would silently bias the very averages the run is trying to measure.
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    bag_dir = str(bag_dir)
    if storage_id is None:
        storage_id = storage_id_from_metadata(bag_dir)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr', output_serialization_format='cdr'
        ),
    )

    type_map = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    wanted = set(topics) if topics else None
    if wanted:
        missing = wanted - set(type_map)
        if missing:
            raise KeyError(
                'the bag does not contain %s. Recorded topics: %s'
                % (sorted(missing), sorted(type_map))
            )
        message_classes = {topic: get_message(type_map[topic]) for topic in wanted}
    else:
        message_classes = {topic: get_message(kind) for topic, kind in type_map.items()}

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if wanted is not None and topic not in wanted:
            continue
        yield topic, deserialize_message(data, message_classes[topic]), timestamp
