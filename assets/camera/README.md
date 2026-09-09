# Shared camera assets

Applications must load reusable camera geometry from this directory instead of
reaching into a ROS package or submodule.

`mesh/d415.stl` is derived from
`realsense2_description/meshes/d415.stl`. It was reduced from 425,160 to 99,997
triangles using quadric edge-collapse decimation because MuJoCo rejects STL
files containing more than 200,000 faces. Its units and coordinate convention
remain those of the upstream RealSense D415 mesh.

The source checkout is pinned at RealSense ROS commit
`60c850958d651130fc2cc3d10efb37ff5be93da5`. Redistribution is covered by the
Apache-2.0 text in `mesh/LICENSE.Apache-2.0.txt`.

## Camera information and workcell calibration

Static, device-related information is kept in `info/d415.yaml`. It contains the
current color intrinsics and the camera-internal
`camera_link -> camera_color_optical_frame` transform extracted from the
calibration run.

Measured workcell transforms are stored in UTC timestamp directories:

```text
calibration/
  20260909T175352_443744Z/
    calibrated_tf.yaml
```

The timestamped YAML preserves both `fr3_link0 -> camera_link` and the composed
`fr3_link0 -> camera_color_optical_frame` transform, together with the tag
fixture and fit quality. Source logs remain unchanged under
`apps/camera_calibration/logs/`. Extraction does not mark the result as
deployment-validated; its measured fit RMSE is retained for review.
