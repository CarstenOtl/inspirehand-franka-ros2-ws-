"""Live RViz view of what the policy sees and where it is steering.

Publishes, from inside the rollout loop so the values are the ones the policy
actually used rather than a reconstruction:

``~/dp3_points``   the exact 4096 XYZRGB points the DP3 encoder sampled, in the
                   camera optical frame (the launch already publishes the
                   static ``fr3_link0`` -> camera transform RViz needs);
``~/scene_points`` those points above the table, in the base frame;
``~/markers``      the controller's controlled point, the controller target,
                   the policy grasp frame, and the nut estimated from the cloud.
"""

from __future__ import annotations

import numpy as np

BASE_FRAME = "fr3_link0"

# dataviz reference palette, categorical slots 1-3 plus status red.
_BLUE = (0.165, 0.471, 0.839)
_ORANGE = (0.922, 0.408, 0.204)
_AQUA = (0.106, 0.686, 0.478)
_RED = (0.890, 0.286, 0.282)
_GREEN = (0.0, 0.514, 0.0)
_PAPER = (0.97, 0.97, 0.96)

TABLE_CLEARANCE_M = 0.02      # above the bench top, in base z
NUT_SEARCH_RADIUS_M = 0.05    # around the bolt axis, for the nut centroid
NUT_MAX_HEIGHT_M = 0.20


class PolicyDebugPublisher:
    """Publish the policy's live point cloud and control frames for RViz."""

    def __init__(self, node, calibration, encoder, *, publish_every=2):
        from sensor_msgs.msg import PointCloud2
        from visualization_msgs.msg import MarkerArray

        self.node = node
        self.calibration = calibration
        self.encoder = encoder
        self.publish_every = max(1, int(publish_every))
        self._tick = 0

        pose = calibration.training_world_pose
        if pose.parent_frame_id != BASE_FRAME:
            raise ValueError(
                f"camera profile pose must be in {BASE_FRAME}, got {pose.parent_frame_id}"
            )
        from .forge_osc import matrix_from_quat

        self.r_base_cam = matrix_from_quat(np.asarray(pose.rotation_wxyz, dtype=float))
        self.t_base_cam = np.asarray(pose.translation_m, dtype=float)

        self.points_publisher = node.create_publisher(PointCloud2, "~/dp3_points", 1)
        self.scene_publisher = node.create_publisher(PointCloud2, "~/scene_points", 1)
        self.marker_publisher = node.create_publisher(MarkerArray, "~/markers", 1)

    # --- helpers ---------------------------------------------------------------
    def _cloud(self, xyz, rgb01, frame_id, stamp):
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Header

        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        colour = (np.asarray(rgb01, dtype=np.float64).reshape(-1, 3) * 255.0).astype(np.uint32)
        packed = (colour[:, 0] << 16) | (colour[:, 1] << 8) | colour[:, 2]
        data = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
        data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        data["rgb"] = packed
        message = PointCloud2()
        message.header = Header(frame_id=frame_id)
        message.header.stamp = stamp
        message.height = 1
        message.width = len(xyz)
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        message.is_bigendian = False
        message.point_step = 16
        message.row_step = 16 * len(xyz)
        message.is_dense = True
        message.data = data.tobytes()
        return message

    def _marker(self, marker_id, kind, stamp, *, colour, scale, namespace="policy"):
        from visualization_msgs.msg import Marker

        marker = Marker()
        marker.header.frame_id = BASE_FRAME
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = kind
        marker.action = Marker.ADD
        marker.color.r, marker.color.g, marker.color.b = colour
        marker.color.a = 0.95
        marker.scale.x, marker.scale.y, marker.scale.z = scale
        marker.pose.orientation.w = 1.0
        return marker

    @staticmethod
    def _point(vector):
        from geometry_msgs.msg import Point

        return Point(x=float(vector[0]), y=float(vector[1]), z=float(vector[2]))

    # --- main entry point ------------------------------------------------------
    def publish(
        self,
        *,
        rgb,
        depth,
        depth_units,
        grasp_position,
        grasp_quaternion,
        controller_target_position,
        controller_measured_position,
        bolt_tip_base,
        policy_goal_position=None,
        step=None,
    ):
        self._tick += 1
        if self._tick % self.publish_every:
            return
        import torch
        from visualization_msgs.msg import Marker, MarkerArray

        from utils.camera_calibration import prepare_rgbd
        from .forge_osc import matrix_from_quat

        stamp = self.node.get_clock().now().to_msg()
        prepared = prepare_rgbd(rgb, depth, self.calibration, depth_units=depth_units)
        with torch.inference_mode():
            sampled, valid = self.encoder._sample_points(
                torch.from_numpy(prepared.rgb)[None],
                torch.from_numpy(prepared.depth)[None],
                torch.from_numpy(prepared.valid_mask)[None],
            )
        sampled = sampled[0].numpy().astype(np.float64)
        valid = valid[0].numpy().astype(bool)
        centre = self.encoder.xyz_center_m.numpy().reshape(3)
        scale = self.encoder.xyz_scale_m.numpy().reshape(3)
        xyz_cam = sampled[:, :3] * scale + centre
        rgb01 = np.clip((sampled[:, 3:6] + 1.0) * 0.5, 0.0, 1.0)
        xyz_cam, rgb01 = xyz_cam[valid], rgb01[valid]

        self.points_publisher.publish(
            self._cloud(xyz_cam, rgb01, self.calibration.frame_id, stamp)
        )

        points_base = (self.r_base_cam @ xyz_cam.T).T + self.t_base_cam
        above = points_base[:, 2] > TABLE_CLEARANCE_M
        above &= points_base[:, 2] < NUT_MAX_HEIGHT_M
        self.scene_publisher.publish(
            self._cloud(points_base[above], rgb01[above], BASE_FRAME, stamp)
        )

        bolt = np.asarray(bolt_tip_base, dtype=float)
        near = above & (np.linalg.norm(points_base[:, :2] - bolt[:2], axis=1) <= NUT_SEARCH_RADIUS_M)
        nut = points_base[near].mean(axis=0) if int(near.sum()) >= 3 else None

        grasp = np.asarray(grasp_position, dtype=float)
        target = np.asarray(controller_target_position, dtype=float)
        measured = np.asarray(controller_measured_position, dtype=float)

        markers = MarkerArray()

        controlled = self._marker(0, Marker.SPHERE, stamp, colour=_BLUE, scale=(0.018,) * 3)
        controlled.pose.position = self._point(measured)
        markers.markers.append(controlled)

        goal = self._marker(1, Marker.SPHERE, stamp, colour=_ORANGE, scale=(0.018,) * 3)
        goal.pose.position = self._point(target)
        markers.markers.append(goal)

        travel = self._marker(2, Marker.ARROW, stamp, colour=_ORANGE, scale=(0.004, 0.009, 0.0))
        travel.points = [self._point(measured), self._point(target)]
        markers.markers.append(travel)

        origin = self._marker(3, Marker.CUBE, stamp, colour=_GREEN, scale=(0.016,) * 3)
        origin.pose.position = self._point(grasp)
        markers.markers.append(origin)

        rotation = matrix_from_quat(np.asarray(grasp_quaternion, dtype=float))
        for axis, colour in enumerate((_RED, _GREEN, _BLUE)):
            arrow = self._marker(4 + axis, Marker.ARROW, stamp, colour=colour,
                                 scale=(0.003, 0.007, 0.0), namespace="grasp_axes")
            arrow.points = [self._point(grasp), self._point(grasp + rotation[:, axis] * 0.05)]
            markers.markers.append(arrow)

        if nut is not None:
            blob = self._marker(7, Marker.SPHERE, stamp, colour=_AQUA, scale=(0.022,) * 3)
            blob.pose.position = self._point(nut)
            markers.markers.append(blob)

        goal_unclipped = None
        if policy_goal_position is not None:
            goal_unclipped = np.asarray(policy_goal_position, dtype=float)
            blob = self._marker(9, Marker.SPHERE, stamp, colour=_RED, scale=(0.026,) * 3)
            blob.color.a = 0.75
            blob.pose.position = self._point(goal_unclipped)
            markers.markers.append(blob)
            reach = self._marker(10, Marker.ARROW, stamp, colour=_RED, scale=(0.005, 0.011, 0.0))
            reach.points = [self._point(grasp), self._point(goal_unclipped)]
            markers.markers.append(reach)

        label = self._marker(8, Marker.TEXT_VIEW_FACING, stamp, colour=_PAPER, scale=(0.0, 0.0, 0.014))
        label.pose.position = self._point(grasp + np.array([0.0, 0.0, 0.10]))
        lines = [f"grasp      z={grasp[2]*1000:7.1f} mm",
                 f"step target z={target[2]*1000:7.1f} mm  (clipped to 20 mm of grasp)"]
        if goal_unclipped is not None:
            lines.append(f"POLICY GOAL z={goal_unclipped[2]*1000:7.1f} mm  "
                         f"|goal-grasp|={np.linalg.norm(goal_unclipped-grasp)*1000:6.1f} mm")
        if nut is not None:
            lines.append(f"nut (cloud) z={nut[2]*1000:7.1f} mm  from {int(near.sum())} pts")
            lines.append(f"            |grasp-nut|={np.linalg.norm(grasp-nut)*1000:6.1f} mm")
            if goal_unclipped is not None:
                lines.append(f"            |GOAL-nut| ={np.linalg.norm(goal_unclipped-nut)*1000:6.1f} mm")
        else:
            lines.append("nut (cloud) not found near the bolt axis")
        label.text = "\n".join(lines)
        if step is not None:
            g = "  goal %s" % np.array2string(goal_unclipped, precision=4) if goal_unclipped is not None else ""
            n = ("  |goal-nut| %6.1f mm" % (np.linalg.norm(goal_unclipped - nut) * 1000)
                 if (nut is not None and goal_unclipped is not None) else "")
            print(f"[viz {step:4d}] grasp {np.array2string(grasp, precision=4)}{g}{n}"
                  f"  |grasp-nut| {np.linalg.norm(grasp-nut)*1000:6.1f} mm" if nut is not None
                  else f"[viz {step:4d}] grasp {np.array2string(grasp, precision=4)}{g}  nut not found",
                  flush=True)
        markers.markers.append(label)

        self.marker_publisher.publish(markers)


__all__ = ["PolicyDebugPublisher", "BASE_FRAME"]
