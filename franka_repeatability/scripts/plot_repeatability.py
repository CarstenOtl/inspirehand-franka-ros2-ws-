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

"""Plot the figures behind a repeatability run.

Reads the run directory written by run_repeatability.py and analysed by
analyze_repeatability.py, and writes a set of PNG/PDF figures. Runs entirely offline.

Most figures need only report.json, visits.csv and run.json. Two of them need more:

- ``joint_contribution`` maps each joint's spread through the Jacobian, so it needs the
  captured robot.urdf and pinocchio (the same dependency analyze_repeatability.py has).
- ``joint_tracking`` needs the recorded bag, because the per-cycle time series is not in
  the summary files. Pass --no-bag to skip it.

Unlike analyze_repeatability.py, the bag reader here keeps the settle phase as well, so a
tracking trace runs unbroken from the pose command to the end of the averaging window.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

NS_PER_S = 1e9
NUM_JOINTS = 7

# One hue per pose, held across every figure so a colour always means the same pose.
POSE_COLORS = ('#2a78d6', '#eb6834', '#1baf7a')
AXIS_COLORS = ('#2a78d6', '#eb6834', '#1baf7a')
AXIS_NAMES = ('x', 'y', 'z')
INK = '#15171a'
INK_2 = '#565a55'
RULE = '#d5d7cf'
FLAG = '#c8322f'

ALL_FIGURES = (
    'spread',
    'drift',
    'repeatability',
    'joint_contribution',
    'joint_deviation',
    'joint_tracking',
)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


def load_run(run_dir):
    """Read the three summary files and reshape them into per-pose arrays."""
    missing = [
        name
        for name in ('run.json', 'report.json', 'visits.csv')
        if not os.path.exists(os.path.join(run_dir, name))
    ]
    if missing:
        raise SystemExit(
            '%s is missing %s. run.json comes from the run itself; report.json and visits.csv '
            'come from analyze_repeatability.py, so run that over this directory first.'
            % (run_dir, ', '.join(missing))
        )

    with open(os.path.join(run_dir, 'run.json'), 'r') as handle:
        metadata = json.load(handle)
    with open(os.path.join(run_dir, 'report.json'), 'r') as handle:
        report = json.load(handle)
    with open(os.path.join(run_dir, 'visits.csv'), 'r') as handle:
        visits = list(csv.DictReader(handle))

    if not visits:
        raise RuntimeError('visits.csv in %s is empty - was the run analysed?' % run_dir)

    # Elapsed minutes since the first averaging window, so drift can be plotted against wall
    # clock rather than against cycle index. The two differ once a cycle is retried.
    windows = sorted(metadata['windows'], key=lambda window: window['window_start_ns'])
    origin_ns = windows[0]['window_start_ns']
    elapsed = {
        (window['pose'], window['cycle']): (window['window_start_ns'] - origin_ns) / NS_PER_S / 60.0
        for window in windows
    }

    poses = []
    for name in [pose['name'] for pose in metadata['config']['poses']]:
        if name in report['per_pose']:
            poses.append(name)

    data = {}
    for name in poses:
        rows = sorted(
            [row for row in visits if row['pose'] == name], key=lambda row: int(row['cycle'])
        )
        cycles = [int(row['cycle']) for row in rows]
        positions = np.array(
            [[float(row['fk_%s_m' % axis]) for axis in AXIS_NAMES] for row in rows]
        )
        joints = np.array(
            [[float(row['q%d_mean_rad' % (j + 1)]) for j in range(NUM_JOINTS)] for row in rows]
        )
        summary = report['per_pose'][name]
        # report.json lists per-visit distances in the order it saw the cycles, not sorted.
        order = [summary['cycles'].index(cycle) for cycle in cycles]
        data[name] = {
            'cycles': np.array(cycles, dtype=float),
            'elapsed_min': np.array([elapsed[(name, cycle)] for cycle in cycles]),
            'position_m': positions,
            'joint_rad': joints,
            'distance_um': np.array(summary['position_fk']['distances_m'])[order] * 1e6,
            'angle_mdeg': np.degrees(np.array(summary['orientation_fk']['angles_rad'])[order])
            * 1000.0,
            'rp_um': summary['position_fk']['repeatability_rp_m'] * 1e6,
            'orientation_rp_mdeg': np.degrees(summary['orientation_fk']['repeatability_rad'])
            * 1000.0,
            'target_m': next(
                pose['position'] for pose in metadata['config']['poses'] if pose['name'] == name
            ),
        }

    return metadata, report, data, poses


def detrend(values, times):
    """Remove the least-squares line in time. Returns (residual, slope per unit time)."""
    if len(values) < 3:
        return np.asarray(values, dtype=float), 0.0
    slope, intercept = np.polyfit(times, values, 1)
    return np.asarray(values) - (slope * np.asarray(times) + intercept), float(slope)


def repeatability_from_points(points):
    """ISO 9283 RP: mean distance from the barycentre plus three standard deviations."""
    barycentre = points.mean(axis=0)
    distances = np.linalg.norm(points - barycentre, axis=1)
    return float(distances.mean() + 3.0 * distances.std(ddof=1)), distances


def detrended_repeatability(entry):
    """RP with the linear-in-time component taken out of each axis first."""
    residual = np.column_stack(
        [detrend(entry['position_m'][:, axis], entry['elapsed_min'])[0] for axis in range(3)]
    )
    value, _ = repeatability_from_points(residual)
    return value * 1e6


# --------------------------------------------------------------------------------------
# bag extraction (only needed by joint_tracking)
# --------------------------------------------------------------------------------------


def extract_tracking(run_dir, metadata, stride):
    """Per-window controller time series, from the pose command to the end of the dwell.

    Returns a list of dicts, one per window, holding time relative to the pose command plus
    the reference, feedback, error and commanded torque for all seven joints.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from franka_repeatability.bag_io import read_messages, stamp_to_ns

    controller_topic = next(
        topic for topic in metadata['topics'] if topic.endswith('controller_state')
    )
    windows = sorted(metadata['windows'], key=lambda window: window['window_start_ns'])

    samples = {index: [] for index in range(len(windows))}
    starts = [window['commanded_ns'] for window in windows]
    ends = [window['window_end_ns'] for window in windows]

    for topic, message, _ in read_messages(
        os.path.join(run_dir, 'bag'), topics=[controller_topic]
    ):
        timestamp_ns = stamp_to_ns(message.header.stamp)
        # Windows are disjoint and sorted, so at most one can contain this sample.
        index = np.searchsorted(starts, timestamp_ns, side='right') - 1
        if index < 0 or timestamp_ns > ends[index]:
            continue
        samples[index].append(
            (
                timestamp_ns,
                list(message.reference.positions),
                list(message.feedback.positions),
                list(message.error.positions),
                list(message.output.effort),
            )
        )

    tracks = []
    for index, window in enumerate(windows):
        rows = sorted(samples[index], key=lambda row: row[0])
        if not rows:
            continue
        rows = rows[::stride]
        tracks.append(
            {
                'pose': window['pose'],
                'cycle': window['cycle'],
                't_s': np.array([row[0] for row in rows], dtype=np.int64).astype(float)
                / NS_PER_S
                - window['commanded_ns'] / NS_PER_S,
                'reference': np.array([row[1] for row in rows]),
                'feedback': np.array([row[2] for row in rows]),
                # error.positions is published as reference - feedback by the controller.
                'error': np.array([row[3] for row in rows]),
                'torque': np.array([row[4] for row in rows]),
                'dwell_start_s': (window['window_start_ns'] - window['commanded_ns']) / NS_PER_S,
                'dwell_end_s': (window['window_end_ns'] - window['commanded_ns']) / NS_PER_S,
            }
        )
    return tracks


def cached_tracking(run_dir, metadata, out_dir, stride, refresh):
    """extract_tracking with an npz cache, so re-plotting does not re-read the bag."""
    cache = os.path.join(out_dir, 'tracking_cache_stride%d.npz' % stride)
    if os.path.exists(cache) and not refresh:
        stored = np.load(cache, allow_pickle=True)
        return list(stored['tracks'])
    tracks = extract_tracking(run_dir, metadata, stride)
    np.savez_compressed(cache, tracks=np.array(tracks, dtype=object))
    return tracks


# --------------------------------------------------------------------------------------
# shared styling
# --------------------------------------------------------------------------------------


def annotate_notes(axes, lines, headroom=0.30):
    """Park a small fixed-width table under the data, making room for it first."""
    bottom, top = axes.get_ylim()
    axes.set_ylim(bottom - headroom * (top - bottom), top)
    axes.annotate(
        '\n'.join(lines),
        (0.99, 0.02),
        xycoords='axes fraction',
        ha='right',
        va='bottom',
        fontsize=7,
        color=INK_2,
        family='monospace',
        bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none', 'pad': 2.5},
    )


def style_axes(axes, grid_axis='y'):
    axes.set_facecolor('none')
    for side in ('top', 'right'):
        axes.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axes.spines[side].set_color(RULE)
    axes.tick_params(colors=INK_2, labelsize=8, length=3, width=0.8)
    axes.grid(True, axis=grid_axis, color=RULE, linewidth=0.7, alpha=0.9)
    axes.set_axisbelow(True)
    axes.xaxis.label.set_color(INK_2)
    axes.yaxis.label.set_color(INK_2)
    axes.title.set_color(INK)


def figure_title(figure, title, subtitle=None):
    """Left-aligned title block above the axes.

    The offsets are in inches divided by the figure height, so the gap between title and
    subtitle stays constant however tall the figure is - a fixed figure fraction collapses
    on a short figure and the two lines overlap.
    """
    height = figure.get_size_inches()[1]
    figure.suptitle(
        title, x=0.012, y=1.0 + 0.34 / height, ha='left', va='bottom',
        fontsize=13, fontweight='semibold', color=INK,
    )
    if subtitle:
        figure.text(
            0.012, 1.0 + 0.12 / height, subtitle, ha='left', va='bottom',
            fontsize=9, color=INK_2,
        )


def save(figure, out_dir, name, formats, dpi, show):
    paths = []
    for suffix in formats:
        path = os.path.join(out_dir, '%s.%s' % (name, suffix))
        figure.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
        paths.append(path)
    if not show:
        import matplotlib.pyplot as plt

        plt.close(figure)
    return paths


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------


def figure_spread(data, poses):
    """Box plot of the visit spread: distance from the barycentre, and the same per axis."""
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={'wspace': 0.22})
    figure_title(
        figure,
        'Spread of the visits, per pose',
        'Boxes are the interquartile range with the median; whiskers reach 1.5 x IQR; '
        'every visit is drawn.',
    )

    # --- distance from the barycentre, the quantity ISO 9283 reduces to RP
    values = [data[name]['distance_um'] for name in poses]
    boxes = left.boxplot(
        values, widths=0.5, patch_artist=True, medianprops={'color': INK, 'linewidth': 1.6}
    )
    for patch in boxes['boxes']:
        patch.set(facecolor='none', edgecolor=INK_2, linewidth=1.2)
    for key in ('whiskers', 'caps'):
        for line in boxes[key]:
            line.set(color=INK_2, linewidth=1.2)
    for line in boxes['fliers']:
        line.set(marker='', linestyle='none')

    rng = np.random.default_rng(0)
    for index, name in enumerate(poses):
        entry = data[name]
        jitter = rng.uniform(-0.13, 0.13, size=len(entry['distance_um']))
        left.plot(
            index + 1 + jitter,
            entry['distance_um'],
            'o',
            markersize=4.5,
            color=POSE_COLORS[index % len(POSE_COLORS)],
            markeredgecolor='white',
            markeredgewidth=0.6,
            zorder=3,
        )
        left.hlines(
            entry['rp_um'], index + 0.72, index + 1.28, color=FLAG, linewidth=1.6, linestyle=(0, (5, 3))
        )
        left.annotate(
            'RP %.1f' % entry['rp_um'],
            (index + 1, entry['rp_um']),
            textcoords='offset points',
            xytext=(0, 6),
            ha='center',
            fontsize=8,
            color=FLAG,
        )

    left.set_xticks(range(1, len(poses) + 1))
    left.set_xticklabels(['%s\nn = %d' % (name, len(data[name]['cycles'])) for name in poses])
    left.set_ylabel('distance from barycentre (um)')
    # The RP marker sits above every observed visit by construction, so make room for it.
    left.set_ylim(0, max(data[name]['rp_um'] for name in poses) * 1.18)
    left.set_title('distance from barycentre', fontsize=10, loc='left', pad=8)
    style_axes(left)

    # --- the same visits resolved onto each axis, signed
    positions = []
    values = []
    colors = []
    for index, name in enumerate(poses):
        entry = data[name]
        centred = (entry['position_m'] - entry['position_m'].mean(axis=0)) * 1e6
        for axis in range(3):
            positions.append(index * 4 + axis)
            values.append(centred[:, axis])
            colors.append(AXIS_COLORS[axis])

    boxes = right.boxplot(
        values, positions=positions, widths=0.7, patch_artist=True
    )
    for patch, color in zip(boxes['boxes'], colors):
        patch.set(facecolor=color, alpha=0.16, edgecolor=color, linewidth=1.2)
    for key in ('whiskers', 'caps'):
        for index, line in enumerate(boxes[key]):
            line.set(color=colors[index // 2], linewidth=1.2)
    for line, color in zip(boxes['medians'], colors):
        line.set(color=color, linewidth=2.0)
    for line in boxes['fliers']:
        line.set(marker='', linestyle='none')

    for slot, series, color in zip(positions, values, colors):
        jitter = rng.uniform(-0.16, 0.16, size=len(series))
        right.plot(slot + jitter, series, 'o', markersize=3, color=color, alpha=0.85, zorder=3)

    right.axhline(0, color=INK_2, linewidth=1.0)
    right.set_xticks(positions)
    right.set_xticklabels(['d%s' % AXIS_NAMES[index % 3] for index in range(len(positions))],
                          fontsize=8)
    for index, name in enumerate(poses):
        right.annotate(
            name,
            (index * 4 + 1, 0),
            xycoords=('data', 'axes fraction'),
            xytext=(0, -28),
            textcoords='offset points',
            ha='center',
            fontsize=10,
            color=INK,
        )
    right.set_ylabel('signed deviation (um)')
    right.set_title('signed deviation, per axis', fontsize=10, loc='left', pad=8)
    style_axes(right)
    return figure


def figure_drift(data, poses):
    """Deviation from the barycentre against elapsed time, with the fitted trend."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1, len(poses), figsize=(4.0 * len(poses), 3.9), sharey=True, gridspec_kw={'wspace': 0.12}
    )
    axes = np.atleast_1d(axes)
    figure_title(
        figure,
        'Deviation from the barycentre, against elapsed time',
        'A stationary process scatters about zero. Dashed lines are the least-squares trend.',
    )

    for index, name in enumerate(poses):
        entry = data[name]
        axis_handle = axes[index]
        centred = (entry['position_m'] - entry['position_m'].mean(axis=0)) * 1e6
        for axis in range(3):
            residual, slope = detrend(centred[:, axis], entry['elapsed_min'])
            axis_handle.plot(
                entry['elapsed_min'],
                centred[:, axis],
                '-o',
                markersize=4,
                linewidth=1.5,
                color=AXIS_COLORS[axis],
                markeredgecolor='white',
                markeredgewidth=0.6,
                label='d%s' % AXIS_NAMES[axis] if index == 0 else None,
            )
            fitted = centred[:, axis] - residual
            axis_handle.plot(
                entry['elapsed_min'],
                fitted,
                linestyle=(0, (5, 3)),
                linewidth=1.3,
                color=AXIS_COLORS[axis],
                alpha=0.8,
            )
        axis_handle.axhline(0, color=INK_2, linewidth=1.0)
        axis_handle.set_title(
            '%s   RP %.1f -> %.1f um'
            % (name, entry['rp_um'], detrended_repeatability(entry)),
            fontsize=9.5,
            loc='left',
            pad=8,
        )
        axis_handle.set_xlabel('elapsed (min)')
        style_axes(axis_handle)
    axes[0].set_ylabel('deviation from barycentre (um)')
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc='upper left', ncol=3)
    return figure


def figure_repeatability(data, poses, spec_um):
    """RP as reported against RP with the drift removed, per pose."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(7.6, 4.2))
    figure_title(
        figure,
        'Repeatability, before and after detrending',
        'ISO 9283 RP = mean distance from the barycentre + 3 sigma.',
    )

    width = 0.34
    slots = np.arange(len(poses))
    reported = [data[name]['rp_um'] for name in poses]
    corrected = [detrended_repeatability(data[name]) for name in poses]

    for offset, values, color, label in (
        (-width / 2 - 0.02, reported, POSE_COLORS[0], 'as reported'),
        (width / 2 + 0.02, corrected, POSE_COLORS[2], 'detrended'),
    ):
        bars = axes.bar(slots + offset, values, width=width, color=color, label=label)
        for rect, value in zip(bars, values):
            axes.annotate(
                '%.1f' % value,
                (rect.get_x() + rect.get_width() / 2, value),
                textcoords='offset points',
                xytext=(0, 4),
                ha='center',
                fontsize=9,
                color=INK,
            )

    if spec_um:
        axes.axhline(spec_um, color=FLAG, linewidth=1.4, linestyle=(0, (6, 4)))
        axes.annotate(
            'datasheet +-%.2f mm' % (spec_um / 1000.0),
            (len(poses) - 0.5, spec_um),
            textcoords='offset points',
            xytext=(0, 5),
            ha='right',
            fontsize=8.5,
            color=FLAG,
        )
        axes.set_ylim(0, spec_um * 1.15)

    axes.set_xticks(slots)
    axes.set_xticklabels(
        [
            '%s\n(%s) m' % (name, ', '.join('%.1f' % value for value in data[name]['target_m']))
            for name in poses
        ]
    )
    axes.set_ylabel('RP (um)')
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc='upper left')
    style_axes(axes)
    return figure


def figure_joint_contribution(run_dir, metadata, data, poses):
    """How much of the Cartesian spread each joint's own spread accounts for."""
    import matplotlib.pyplot as plt

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from franka_repeatability.kinematics import ForwardKinematics

    config = metadata['config']
    forward = ForwardKinematics(os.path.join(run_dir, 'robot.urdf'), config['joint_names'])

    def jacobian(joints, epsilon=1e-6):
        base, _ = forward.pose(joints, config['ee_link'], config['base_frame'])
        columns = []
        for index in range(NUM_JOINTS):
            perturbed = np.array(joints, dtype=float)
            perturbed[index] += epsilon
            moved, _ = forward.pose(perturbed, config['ee_link'], config['base_frame'])
            columns.append((moved - base) / epsilon)
        return np.column_stack(columns)

    figure, axes = plt.subplots(figsize=(9.2, 4.4))
    figure_title(
        figure,
        "Contribution of each joint's repeatability to flange position",
        'Per-joint standard deviation across the visits, mapped through the Jacobian column '
        'at that pose.',
    )

    width = 0.8 / len(poses)
    slots = np.arange(NUM_JOINTS)
    for index, name in enumerate(poses):
        entry = data[name]
        columns = jacobian(entry['joint_rad'].mean(axis=0))
        spread = entry['joint_rad'].std(axis=0, ddof=1)
        contribution = np.linalg.norm(columns, axis=0) * spread * 1e6
        axes.bar(
            slots + (index - (len(poses) - 1) / 2) * width,
            contribution,
            width=width * 0.9,
            color=POSE_COLORS[index % len(POSE_COLORS)],
            label=name,
        )

    axes.set_xticks(slots)
    gains = metadata.get('controller_parameters', {}).get('k_gains')
    axes.set_xticklabels(
        ['J%d\nk=%g' % (index + 1, gains[index]) if gains else 'J%d' % (index + 1)
         for index in range(NUM_JOINTS)]
    )
    axes.set_ylabel('contribution to flange position (um)')
    # Parked above the axes: the interior is where the J7 annotation has to go.
    axes.legend(
        frameon=False, fontsize=9, labelcolor=INK_2, ncol=len(poses),
        loc='lower right', bbox_to_anchor=(1.0, 1.005),
    )
    style_axes(axes)
    top = axes.get_ylim()[1]
    axes.annotate(
        'a joint on the flange axis contributes 0 to position\n'
        'however badly it repeats - it shows up in orientation',
        (NUM_JOINTS - 1, top * 0.02),
        xytext=(NUM_JOINTS - 0.6, top * 0.62),
        textcoords='data',
        ha='right',
        va='bottom',
        fontsize=8.5,
        color=FLAG,
        arrowprops={'arrowstyle': '-', 'color': FLAG, 'linewidth': 1.0,
                    'linestyle': (0, (3, 2))},
    )
    return figure


def figure_joint_deviation(data, poses):
    """Seven subplots: visit-to-visit deviation of each joint's dwell mean."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 2, figsize=(11, 11.5), sharex=True)
    figure_title(
        figure,
        'Per-joint deviation across the visits',
        'Dwell-mean joint angle minus that pose\'s mean, against elapsed time. Dashed lines '
        'are the least-squares trend; sigma is quoted per pose.',
    )
    flat = axes.ravel()

    for joint in range(NUM_JOINTS):
        axis_handle = flat[joint]
        notes = []
        for index, name in enumerate(poses):
            entry = data[name]
            centred = (entry['joint_rad'][:, joint] - entry['joint_rad'][:, joint].mean()) * 1e6
            color = POSE_COLORS[index % len(POSE_COLORS)]
            axis_handle.plot(
                entry['elapsed_min'],
                centred,
                '-o',
                markersize=4,
                linewidth=1.4,
                color=color,
                markeredgecolor='white',
                markeredgewidth=0.6,
                label=name,
            )
            residual, slope = detrend(centred, entry['elapsed_min'])
            axis_handle.plot(
                entry['elapsed_min'],
                centred - residual,
                linestyle=(0, (5, 3)),
                linewidth=1.2,
                color=color,
                alpha=0.8,
            )
            notes.append('%-4s %7.0f %+9.0f' % (name, centred.std(ddof=1), slope))

        axis_handle.axhline(0, color=INK_2, linewidth=1.0)
        axis_handle.set_title('joint %d' % (joint + 1), fontsize=10, loc='left', pad=6)
        axis_handle.set_ylabel('deviation (urad)')
        style_axes(axis_handle)
        annotate_notes(axis_handle, ['%-4s %7s %9s' % ('', 'sigma', 'drift')] + notes
                       + ['%-4s %7s %9s' % ('', 'urad', 'urad/min')])
        if joint >= NUM_JOINTS - 2:
            axis_handle.set_xlabel('elapsed (min)')

    legend_cell = flat[-1]
    legend_cell.axis('off')
    handles, labels = flat[0].get_legend_handles_labels()
    legend_cell.legend(
        handles, labels, frameon=False, fontsize=10, labelcolor=INK_2, loc='upper left',
        title='pose', title_fontsize=9,
    )
    legend_cell.annotate(
        'Each panel shares the joint-space view of the same visits that the\n'
        'Cartesian figures summarise. A joint whose scatter is dominated by\n'
        'its trend line is drifting, not repeating badly.',
        (0.02, 0.55),
        xycoords='axes fraction',
        fontsize=8.5,
        color=INK_2,
        va='top',
    )
    return figure


def figure_joint_tracking(tracks, poses, mode):
    """Seven subplots: the controller's tracking error per joint, every visit overlaid."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 2, figsize=(11, 11.5), sharex=True)
    if mode == 'error':
        subtitle = (
            'Reference minus measured, for every visit, aligned on the pose command. '
            'The shaded band is the averaging window.'
        )
        label = 'tracking error (mrad)'
    else:
        subtitle = (
            'Reference (dashed) against measured (solid) for every visit, aligned on the '
            'pose command.'
        )
        label = 'joint angle (rad)'
    figure_title(figure, 'Per-joint tracking through the move and the dwell', subtitle)
    flat = axes.ravel()

    dwell_start = np.median([track['dwell_start_s'] for track in tracks])
    dwell_end = np.median([track['dwell_end_s'] for track in tracks])

    for joint in range(NUM_JOINTS):
        axis_handle = flat[joint]
        axis_handle.axvspan(dwell_start, dwell_end, color='#0d366b', alpha=0.07, linewidth=0)
        for track in tracks:
            index = poses.index(track['pose']) if track['pose'] in poses else 0
            color = POSE_COLORS[index % len(POSE_COLORS)]
            if mode == 'error':
                axis_handle.plot(
                    track['t_s'], track['error'][:, joint] * 1000.0,
                    linewidth=0.9, color=color, alpha=0.55,
                )
            else:
                axis_handle.plot(
                    track['t_s'], track['feedback'][:, joint],
                    linewidth=0.9, color=color, alpha=0.55,
                )
                axis_handle.plot(
                    track['t_s'], track['reference'][:, joint],
                    linewidth=0.8, color=color, alpha=0.5, linestyle=(0, (4, 3)),
                )

        notes = []
        if mode == 'error':
            axis_handle.axhline(0, color=INK_2, linewidth=1.0)
            for index, name in enumerate(poses):
                selected = [track for track in tracks if track['pose'] == name]
                if not selected:
                    continue
                settled = np.concatenate(
                    [
                        track['error'][track['t_s'] >= track['dwell_start_s'], joint]
                        for track in selected
                    ]
                )
                peak = max(np.abs(track['error'][:, joint]).max() for track in selected)
                notes.append(
                    '%-4s %+8.2f %8.2f' % (name, settled.mean() * 1000.0, peak * 1000.0)
                )

        axis_handle.set_title('joint %d' % (joint + 1), fontsize=10, loc='left', pad=6)
        axis_handle.set_ylabel(label)
        style_axes(axis_handle)
        if mode == 'error' and notes:
            annotate_notes(
                axis_handle,
                ['%-4s %8s %8s' % ('', 'steady', 'peak')] + notes
                + ['%-4s %8s %8s' % ('', 'mrad', 'mrad')],
            )
        if joint >= NUM_JOINTS - 2:
            axis_handle.set_xlabel('time since the pose command (s)')

    legend_cell = flat[-1]
    legend_cell.axis('off')
    handles = [
        plt.Line2D([], [], color=POSE_COLORS[index % len(POSE_COLORS)], linewidth=2)
        for index in range(len(poses))
    ]
    legend_cell.legend(
        handles, poses, frameon=False, fontsize=10, labelcolor=INK_2, loc='upper left',
        title='pose', title_fontsize=9,
    )
    legend_cell.annotate(
        'A joint impedance law is a spring: the steady-state offset inside\n'
        'the shaded window is the torque the joint needs to hold position\n'
        'divided by its stiffness. Its sign follows the approach direction,\n'
        'so it cancels out of a unidirectional repeatability figure.',
        (0.02, 0.55),
        xycoords='axes fraction',
        fontsize=8.5,
        color=INK_2,
        va='top',
    )
    return figure


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('run_dir', help='run directory holding run.json, report.json, visits.csv')
    parser.add_argument('--out', default=None, help='output directory (default: RUN_DIR/plots)')
    parser.add_argument(
        '--only', nargs='+', choices=ALL_FIGURES, default=None, help='draw only these figures'
    )
    parser.add_argument(
        '--format', nargs='+', default=['png'], choices=['png', 'pdf', 'svg'],
        help='output formats (default: png)',
    )
    parser.add_argument('--dpi', type=int, default=160)
    parser.add_argument('--show', action='store_true', help='open the figures instead of exiting')
    parser.add_argument(
        '--no-bag', action='store_true', help='skip figures that need the recorded bag'
    )
    parser.add_argument(
        '--stride', type=int, default=10,
        help='keep every Nth control sample in the tracking figure (default: 10, i.e. 100 Hz)',
    )
    parser.add_argument(
        '--tracking-mode', choices=['error', 'absolute'], default='error',
        help='plot the tracking error, or reference against measured (default: error)',
    )
    parser.add_argument(
        '--refresh', action='store_true', help='re-read the bag instead of using the npz cache'
    )
    parser.add_argument(
        '--spec-um', type=float, default=100.0,
        help='datasheet repeatability drawn as a reference line, in um (0 to omit)',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])

    import matplotlib

    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            'font.size': 9,
            'axes.titleweight': 'semibold',
            'figure.facecolor': 'white',
            'savefig.facecolor': 'white',
            'text.color': INK,
        }
    )

    run_dir = os.path.abspath(args.run_dir)
    metadata, report, data, poses = load_run(run_dir)
    out_dir = args.out or os.path.join(run_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    wanted = list(args.only) if args.only else list(ALL_FIGURES)
    written = []
    skipped = []

    def emit(name, builder):
        if name not in wanted:
            return
        try:
            figure = builder()
        except Exception as error:  # a missing optional dependency must not lose the rest
            skipped.append((name, '%s: %s' % (type(error).__name__, error)))
            return
        written.extend(save(figure, out_dir, name, args.format, args.dpi, args.show))

    emit('spread', lambda: figure_spread(data, poses))
    emit('drift', lambda: figure_drift(data, poses))
    emit('repeatability', lambda: figure_repeatability(data, poses, args.spec_um))
    emit('joint_contribution', lambda: figure_joint_contribution(run_dir, metadata, data, poses))
    emit('joint_deviation', lambda: figure_joint_deviation(data, poses))

    if 'joint_tracking' in wanted:
        if args.no_bag:
            skipped.append(('joint_tracking', 'skipped: --no-bag'))
        else:
            try:
                tracks = cached_tracking(run_dir, metadata, out_dir, args.stride, args.refresh)
                figure = figure_joint_tracking(tracks, poses, args.tracking_mode)
                written.extend(
                    save(figure, out_dir, 'joint_tracking', args.format, args.dpi, args.show)
                )
            except Exception as error:
                skipped.append(('joint_tracking', '%s: %s' % (type(error).__name__, error)))

    for path in written:
        print('wrote %s' % path)
    for name, reason in skipped:
        print('skipped %s (%s)' % (name, reason), file=sys.stderr)

    if args.show:
        plt.show()
    return 0 if written else 1


if __name__ == '__main__':
    sys.exit(main())
