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

"""Recording with ``ros2 bag record`` and reading the result back.

Recording goes through the rosbag2 CLI rather than a Python subscriber: at 1 kHz rclpy drops
messages, which would silently bias the very tracking numbers a run exists to produce.
"""

import os
import signal
import subprocess
import time

import yaml


def stamp_to_ns(stamp):
    return int(stamp.sec) * 1000000000 + int(stamp.nanosec)


class BagRecorder:
    def __init__(self, bag_dir, topics, storage_id='sqlite3', logger=None):
        self.bag_dir = str(bag_dir)
        self.topics = list(topics)
        self.storage_id = storage_id
        self.logger = logger
        self.process = None

    def _log(self, level, text):
        if self.logger is not None:
            getattr(self.logger, level)(text)
        else:
            print('[%s] %s' % (level, text))

    def start(self, timeout=20.0):
        command = ['ros2', 'bag', 'record', '-o', self.bag_dir, '-s', self.storage_id]
        command += self.topics
        self._log('info', 'recording: %s' % ' '.join(command))
        # No stdin: the recorder otherwise installs its own keyboard handler (space to pause)
        # on the shared terminal, which swallows the run script's Enter prompts and turns them
        # into an end-of-file, i.e. an abort.
        self.process = subprocess.Popen(command, preexec_fn=os.setsid, stdin=subprocess.DEVNULL)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    'ros2 bag record exited immediately with code %d' % self.process.returncode
                )
            if os.path.isdir(self.bag_dir) and any(
                name.endswith('.db3') or name.endswith('.mcap') for name in os.listdir(self.bag_dir)
            ):
                time.sleep(1.0)  # let the subscriptions finish matching before anything moves
                return
            time.sleep(0.2)
        raise TimeoutError('ros2 bag record did not start writing within %.0f s' % timeout)

    def stop(self, timeout=30.0):
        if self.process is None or self.process.poll() is not None:
            return
        os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._log('warning', 'ros2 bag record did not stop on SIGINT, terminating')
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=10.0)


def storage_id_from_metadata(bag_dir, fallback='sqlite3'):
    try:
        with open(os.path.join(str(bag_dir), 'metadata.yaml'), 'r') as handle:
            metadata = yaml.safe_load(handle)
        return metadata['rosbag2_bagfile_information']['storage_identifier']
    except (OSError, KeyError, TypeError):
        return fallback


def read_messages(bag_dir, topics=None, storage_id=None):
    """Yield ``(topic, message, receive_time_ns)`` for every message in the bag."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    bag_dir = str(bag_dir)
    if storage_id is None:
        storage_id = storage_id_from_metadata(bag_dir)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id),
        rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'),
    )
    type_map = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    wanted = set(topics) if topics else None
    if wanted:
        wanted = {topic for topic in wanted if topic in type_map}
    classes = {topic: get_message(kind) for topic, kind in type_map.items() if wanted is None or topic in wanted}
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic not in classes:
            continue
        yield topic, deserialize_message(data, classes[topic]), timestamp
