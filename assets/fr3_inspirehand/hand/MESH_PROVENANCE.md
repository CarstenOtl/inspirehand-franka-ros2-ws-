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

`apriltag_36h11_id0_dorsal.obj` is not a replacement hand mesh. It is a local,
surface-following visual sheet sampled from `R_base_link.STL`, with its back
0.05 mm outside the shell so the marker is visible without a floating gap. The
36h11 id 0 black square is 40 mm; its one-cell quiet zone makes the complete
textured sheet 50 mm square. Its print is rotated 180 degrees to match the
physical workcell hand.
