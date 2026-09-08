# TienKung 2 Pro right-hand meshes

The 13 STL files in this directory are copied byte-for-byte from
`tiangong2pro_urdf/meshes` in Open-X-Humanoid's official `TienKung_URDF`
repository:

- Source: https://github.com/Open-X-Humanoid/TienKung_URDF/tree/main/tiangong2pro_urdf
- Source commit: `5c221783fb92fcc4af891ef1dc0502963caf2266`

The link transforms, inertias, joint axes, limits, and mimic ratios used by the
adjacent MJCF scenes come from
`tiangong2pro_urdf/urdf/tiangong2.0_complete_with_hands.xacro` at that commit.
Only the FR3 flange adapter and the workcell AprilTag/tip markers are local
integration details.
