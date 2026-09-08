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

"""Reading Isaac Sim trajectory captures (.npy / .npz) into one common structure.

Accepted layouts:

* ``.npy`` holding an (N, 7) array of joint positions. The sample rate has to be given
  (``rate=``) unless the array has 8 columns and one of them is a monotonically increasing time
  column (``time_column=`` picks it; column 0 is tried by default).
* ``.npz`` with a 3-D ``joint_pos`` of shape (N, environments, dofs) as the Isaac ``replay_data``
  export writes it: ``env=`` picks the environment (default 0). A ``metadata.json`` next to the
  file supplies ``dt`` / ``recording_frequency_hz``, ``joint_names`` and ``arm_joint_ids`` when the
  npz itself carries none.
* ``.csv`` segment exports with a header ``time_s, pos_<joint>..., vel_<joint>..., cmd_<joint>...``;
  ``csv_prefix`` selects ``pos`` (measured, default) or ``cmd`` (the policy target).
* ``.npy`` holding a pickled dict, or ``.npz`` with named arrays. The joint positions are looked
  up under ``q``, ``joint_pos``, ``joint_pos_arm``, ``qpos``, ``joint_positions``; the time base
  under ``t``/``time``/``timestamps`` (per sample), ``dt`` (scalar) or ``rate``/``hz``/``fps``;
  optional joint names under ``arm_joint_names``/``joint_names``; optional end-effector pose
  under ``ee_pos``/``ee_quat``; optional joint velocities under ``qd``/``joint_vel``/``dq``.

Joint order: if names are present they are matched to the FR3 joints by their trailing index
(``fr3_joint3``, ``panda_joint3``, ``joint3``, ``joint_3``, ``A3`` all map to joint 3). Names
that do not look like Franka joints are refused unless ``joint_map`` (a list of 7 column
indices) or ``assume_order=True`` is given, because silently replaying another robot's joint
angles on the FR3 is exactly the mistake this loader exists to catch.
"""

import os
import re
from dataclasses import dataclass, field

import numpy as np

Q_KEYS = ('q', 'joint_pos', 'joint_pos_arm', 'qpos', 'joint_positions', 'positions', 'arm_q')
QD_KEYS = ('qd', 'joint_vel', 'joint_vel_arm', 'dq', 'joint_velocities', 'velocities')
T_KEYS = ('t', 'time', 'timestamps', 'times')
DT_KEYS = ('dt', 'timestep', 'step_dt')
RATE_KEYS = ('rate', 'hz', 'fps', 'frequency')
NAME_KEYS = ('arm_joint_names', 'joint_names', 'names', 'dof_names')
EE_POS_KEYS = ('ee_pos', 'tcp_pos', 'ee_position', 'eef_pos')
EE_QUAT_KEYS = ('ee_quat', 'tcp_quat', 'ee_orientation', 'eef_quat')

FRANKA_NAME_PREFIXED = re.compile(r'^(?:.*?)(?:fr3|panda|fer|fp3)_?(?:joint|A|j)_?([1-7])$', re.IGNORECASE)
FRANKA_NAME = re.compile(r'^(?:joint|A|j)_?([1-7])$', re.IGNORECASE)


@dataclass
class Trajectory:
    t: np.ndarray                        # (N,) seconds, starting at 0
    q: np.ndarray                        # (N, 7) joint positions, FR3 order, radians
    qd: np.ndarray = None                # (N, 7) joint velocities, if the capture had them
    ee_pos: np.ndarray = None            # (N, 3) end-effector position as the source saw it
    ee_quat: np.ndarray = None           # (N, 4) end-effector quaternion (x, y, z, w) from source
    source: str = ''
    meta: dict = field(default_factory=dict)

    @property
    def duration(self):
        return float(self.t[-1] - self.t[0])

    @property
    def rate(self):
        return float((len(self.t) - 1) / self.duration) if len(self.t) > 1 else 0.0


def _first(mapping, keys):
    for key in keys:
        if key in mapping:
            return key, mapping[key]
    return None, None


def _as_dict(path):
    loaded = np.load(path, allow_pickle=True)
    if hasattr(loaded, 'files'):
        return {key: loaded[key] for key in loaded.files}
    array = np.asarray(loaded)
    if array.dtype == object and array.shape == ():
        item = array.item()
        if isinstance(item, dict):
            return {str(key): np.asarray(value) for key, value in item.items()}
    return {'__array__': array}


def joint_order_from_names(names, joint_map=None, assume_order=False):
    """Column indices that put the source columns into FR3 joint order."""
    names = [str(name) for name in names]
    if joint_map is not None:
        joint_map = [int(index) for index in joint_map]
        if len(joint_map) != 7:
            raise ValueError('joint_map needs exactly 7 column indices')
        return joint_map
    matched = {}
    for pattern in (FRANKA_NAME_PREFIXED, FRANKA_NAME):
        for column, name in enumerate(names):
            match = pattern.match(name.strip())
            if match:
                matched.setdefault(int(match.group(1)), column)
        if len(matched) == 7:
            return [matched[index] for index in range(1, 8)]
    if assume_order:
        if len(names) < 7:
            raise ValueError('need at least 7 joint columns, got %d' % len(names))
        return list(range(7))
    raise ValueError(
        'the joint names in the capture do not look like FR3 joints: %s. If this really is an '
        'FR3 trajectory pass joint_map=[...] (7 column indices in FR3 order) or '
        'assume_order=True.' % names
    )


def _load_csv(path, csv_prefix='pos'):
    """Isaac segment export: time_s, pos_<joint>..., vel_<joint>..., cmd_<joint>..."""
    import csv

    with open(path, newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [row for row in reader if row]
    table = np.asarray(rows, dtype=float)
    columns = {name.strip(): k for k, name in enumerate(header)}
    time_key = next((k for k in ('time_s', 'time', 't') if k in columns), None)
    if time_key is None:
        raise ValueError('%s: no time_s column' % path)
    prefix = csv_prefix.rstrip('_') + '_'
    joint_columns = [name for name in header if name.startswith(prefix)]
    if not joint_columns:
        raise ValueError('%s: no columns starting with %r (have %s)' % (path, prefix, header[:12]))
    names = [name[len(prefix):] for name in joint_columns]
    q = table[:, [columns[name] for name in joint_columns]]
    qd = None
    vel_columns = ['vel_' + name for name in names]
    if all(name in columns for name in vel_columns):
        qd = table[:, [columns[name] for name in vel_columns]]
    return {'q': q, 'qd': qd, 't': table[:, columns[time_key]], 'names': names,
            'meta': {'csv_prefix': csv_prefix, 'csv_joints': names}}


def _sidecar_metadata(path):
    """metadata.json written next to an Isaac replay_data.npz."""
    import json

    for candidate in (os.path.join(os.path.dirname(path), 'metadata.json'),
                      os.path.join(os.path.dirname(path), 'manifest.json')):
        if os.path.exists(candidate):
            try:
                with open(candidate) as handle:
                    return json.load(handle), candidate
            except (OSError, ValueError):
                continue
    return {}, None


def load_trajectory(path, rate=None, time_column=None, joint_map=None, assume_order=False,
                    degrees=False, env=0, key=None, csv_prefix='pos'):
    """Load a capture into a :class:`Trajectory` with FR3 joint order and a zero-based clock."""
    path = os.path.expanduser(str(path))
    meta = {}

    if path.lower().endswith('.csv'):
        loaded = _load_csv(path, csv_prefix)
        q, qd, t, names = loaded['q'], loaded['qd'], loaded['t'], loaded['names']
        t = t - t[0]
        ee_pos = ee_quat = None
        meta.update(loaded['meta'])
        sidecar, sidecar_path = _sidecar_metadata(path)
        if sidecar:
            meta['sidecar'] = sidecar_path
            for k in ('task', 'checkpoint', 'rate_hz'):
                if k in sidecar:
                    meta[k] = sidecar[k]
        data = None
    else:
        data = _as_dict(path)

    if data is None:
        pass
    elif '__array__' in data:
        array = np.asarray(data['__array__'], dtype=float)
        if array.ndim != 2:
            raise ValueError('%s: expected a 2-D array, got shape %s' % (path, array.shape))
        t = None
        if array.shape[1] == 7:
            q = array
        elif array.shape[1] >= 8:
            column = 0 if time_column is None else int(time_column)
            candidate = array[:, column]
            if np.all(np.diff(candidate) > 0):
                t = candidate - candidate[0]
                q = np.delete(array, column, axis=1)[:, :7]
            elif time_column is not None:
                raise ValueError('column %d is not monotonically increasing' % column)
            else:
                q = array[:, :7]
        else:
            raise ValueError('%s: need at least 7 columns, got %d' % (path, array.shape[1]))
        qd = None
        names = None
        ee_pos = None
        ee_quat = None
    else:
        if key is not None:
            if key not in data:
                raise ValueError('%s has no array %r (keys: %s)' % (path, key, sorted(data)))
            q = data[key]
        else:
            key, q = _first(data, Q_KEYS)
        if q is None:
            raise ValueError(
                '%s: no joint position array under any of %s (keys: %s)'
                % (path, Q_KEYS, sorted(data))
            )
        meta['q_key'] = key
        q = np.asarray(q, dtype=float)
        _, qd = _first(data, QD_KEYS)
        _, ee_pos = _first(data, EE_POS_KEYS)
        _, ee_quat = _first(data, EE_QUAT_KEYS)
        if q.ndim == 3:
            # (N, environments, dofs): the Isaac multi-env replay export.
            if not 0 <= int(env) < q.shape[1]:
                raise ValueError('%s holds %d environments, env=%d does not exist' % (path, q.shape[1], env))
            meta['environments'] = int(q.shape[1])
            meta['env'] = int(env)
            q = q[:, int(env), :]
            qd = None if qd is None else np.asarray(qd, dtype=float)[:, int(env), :]
            ee_pos = None if ee_pos is None else np.asarray(ee_pos, dtype=float)[:, int(env), :]
            ee_quat = None if ee_quat is None else np.asarray(ee_quat, dtype=float)[:, int(env), :]
        if q.ndim != 2:
            raise ValueError('%s[%s] has shape %s, expected (N, 7)' % (path, key, q.shape))

        _, names = _first(data, NAME_KEYS)
        _, t = _first(data, T_KEYS)
        sidecar, sidecar_path = _sidecar_metadata(path)
        if sidecar:
            meta['sidecar'] = sidecar_path
            for k in ('task', 'checkpoint', 'recording_clock', 'rollout_mode', 'physics_dt', 'decimation'):
                if k in sidecar:
                    meta[k] = sidecar[k]
            if names is None:
                if sidecar.get('arm_joint_ids') and sidecar.get('joint_names'):
                    names = [sidecar['joint_names'][i] for i in sidecar['arm_joint_ids']]
                    q = q[:, sidecar['arm_joint_ids']]
                    if qd is not None:
                        qd = np.asarray(qd, dtype=float)[:, sidecar['arm_joint_ids']]
                elif sidecar.get('joint_names'):
                    names = sidecar['joint_names']
            if t is None and not any(k in data for k in DT_KEYS + RATE_KEYS):
                if sidecar.get('dt'):
                    t = np.arange(len(q)) * float(sidecar['dt'])
                    meta['dt'] = float(sidecar['dt'])
                elif sidecar.get('recording_frequency_hz') or sidecar.get('rate_hz'):
                    hz = float(sidecar.get('recording_frequency_hz') or sidecar.get('rate_hz'))
                    t = np.arange(len(q)) / hz
                    meta['dt'] = 1.0 / hz
        if t is not None:
            t = np.asarray(t, dtype=float).reshape(-1)
            t = t - t[0]
        else:
            _, dt = _first(data, DT_KEYS)
            _, hz = _first(data, RATE_KEYS)
            if dt is not None:
                t = np.arange(len(q)) * float(dt)
                meta['dt'] = float(dt)
            elif hz is not None:
                t = np.arange(len(q)) / float(hz)
        for key in ('task_id', 'checkpoint_path', 'warning', 'dt', 'chosen_env_idx'):
            if key in data:
                value = data[key]
                meta[key] = value.item() if getattr(value, 'shape', None) == () else value.tolist()

    if t is None:
        if rate is None:
            raise ValueError(
                '%s carries no time base; pass the capture rate (rate=..., e.g. 15 for a 15 Hz '
                'policy)' % path
            )
        t = np.arange(len(q)) / float(rate)
    if len(t) != len(q):
        raise ValueError('time base has %d samples, joint positions %d' % (len(t), len(q)))
    if np.any(np.diff(t) <= 0):
        raise ValueError('%s: time stamps are not strictly increasing' % path)

    if names is not None:
        order = joint_order_from_names(np.asarray(names).tolist(), joint_map, assume_order)
        meta['source_joint_names'] = [str(n) for n in np.asarray(names).tolist()]
    elif joint_map is not None:
        order = [int(i) for i in joint_map]
    else:
        order = list(range(7))
    if q.shape[1] < max(order) + 1:
        raise ValueError('joint order %s needs %d columns, the capture has %d' % (order, max(order) + 1, q.shape[1]))
    q = q[:, order]
    if qd is not None:
        qd = np.asarray(qd, dtype=float)[:, order]

    if degrees:
        q = np.deg2rad(q)
        if qd is not None:
            qd = np.deg2rad(qd)
    elif np.abs(q).max() > 2.0 * np.pi + 0.05:
        raise ValueError(
            '%s: joint positions reach %.1f, which is not radians. Pass degrees=True if the '
            'capture is in degrees.' % (path, np.abs(q).max())
        )

    return Trajectory(
        t=np.asarray(t, dtype=float),
        q=np.ascontiguousarray(q, dtype=float),
        qd=None if qd is None else np.ascontiguousarray(qd, dtype=float),
        ee_pos=None if ee_pos is None else np.asarray(ee_pos, dtype=float),
        ee_quat=None if ee_quat is None else np.asarray(ee_quat, dtype=float),
        source=path,
        meta=meta,
    )


def save_prepared(path, prepared, source=None, meta=None):
    """Write the commanded (dense) trajectory so a run is reproducible from its directory."""
    payload = {
        't': prepared.t,
        'q': prepared.q,
        'qd': prepared.qd,
        'qdd': prepared.qdd,
        'qddd': prepared.qddd,
        'joint_names': np.asarray(prepared.joint_names),
    }
    if source is not None:
        payload['source_t'] = source.t
        payload['source_q'] = source.q
        if source.ee_pos is not None:
            payload['source_ee_pos'] = source.ee_pos
        if source.ee_quat is not None:
            payload['source_ee_quat'] = source.ee_quat
        payload['source_path'] = np.asarray(source.source)
    if meta:
        payload['meta_json'] = np.asarray(__import__('json').dumps(meta))
    np.savez_compressed(os.path.expanduser(str(path)), **payload)


def load_prepared(path):
    from franka_trajectory_replay.prepare import Prepared

    data = np.load(os.path.expanduser(str(path)), allow_pickle=False)
    prepared = Prepared(
        t=data['t'], q=data['q'], qd=data['qd'], qdd=data['qdd'], qddd=data['qddd'],
        joint_names=[str(n) for n in data['joint_names'].tolist()],
    )
    extras = {key: data[key] for key in data.files if key.startswith('source_')}
    meta = {}
    if 'meta_json' in data.files:
        meta = __import__('json').loads(str(data['meta_json']))
    return prepared, extras, meta
