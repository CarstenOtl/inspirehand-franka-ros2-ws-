# TienKung 2 Pro right-hand meshes

The 13 vendor STL files in this directory are copied byte-for-byte from
`tiangong2pro_urdf/meshes` in Open-X-Humanoid's official `TienKung_URDF`
repository:

- Source: https://github.com/Open-X-Humanoid/TienKung_URDF/tree/main/tiangong2pro_urdf
- Source commit: `5c221783fb92fcc4af891ef1dc0502963caf2266`

The link transforms, inertias, joint axes, limits, and mimic ratios used by the
adjacent MJCF scenes come from
`tiangong2pro_urdf/urdf/tiangong2.0_complete_with_hands.xacro` at that commit.
Only the FR3 flange adapter and the workcell AprilTag/tip markers are local
integration details.

`adapter_flange.stl` is the locally generated FR3-to-hand mounting spacer. Its
compact binary STL is a closed 32-sided cylinder, 76 mm in diameter and 10 mm
thick, centred on its mesh origin. The MJCF places that origin 5 mm above
`fr3_link8`, making the part span the full 0--10 mm gap to the palm.

`apriltag_36h11_id0_dorsal.obj` is not a replacement hand mesh. It is the local,
flat 2.3 mm mounting plate used by the physical flange-mounted holder. The 36h11
id 0 black square is 40 mm; its one-cell quiet zone makes the complete textured
face 50 mm square. The mount first creates an `Rz1=-45 deg` frame about
`fr3_link8` Z, then places the printed face centre 60 mm along Rz1 +X and 35 mm
along Rz1 +Z. The print has zero in-plane clocking, with one edge parallel to
flange Z. The existing print/UV orientation is preserved.
