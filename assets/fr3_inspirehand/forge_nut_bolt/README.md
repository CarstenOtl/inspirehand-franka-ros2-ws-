# ForgeUltra nut and bolt meshes

These STL files are native-metre exports of the active `geom_baked` meshes in
ForgeUltra's physics-ready USD assets on the `chi_testing` branch at commit
`c50fb8a7b2f1539dc449daa24de90209bc40e95d`:

- `M24_nut_sdf.usd` / `M24_bolt35_sdf.usd`
- `M30_nut_sdf.usd` / `M30_bolt35_sdf.usd`
- `M36_nut_sdf.usd` / `M36_bolt35_sdf.usd`

Each nut contains 15,000 triangles and each bolt contains 26,000 triangles.
The USD local-to-world transform was baked into the STL vertices without
rescaling or recentering. The nut origin remains at its center; each bolt runs
from its base at `z=0` to its physical tip.

MuJoCo renders the complete triangle mesh. Its rigid mesh collision uses a
convex hull, while the replay's one-way twist-to-axial constraint supplies the
thread mechanics previously implemented with PhysX SDF contact.
