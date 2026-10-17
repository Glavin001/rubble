# Shardwright assets through full destruction: findings (2026-10-11)

To continue the fixes or verify them, see [HANDOFF_SHARDWRIGHT.md](HANDOFF_SHARDWRIGHT.md).

Question: does a Shardwright pre-fractured asset run through full destruction on the new
stress solver, with PhysX as the rigid-body engine? Asset: `two_storey_building`, baked
with `benchmarks/configs/support_clusters.toml`. Its L2 level has 1,182 chunks
(fragments made of 2–16 convex hulls, 3,655 hulls in all) and 3,635 bonds plus 23 anchors.

All runs used the CPU only: stress-ref, and PhysX CPU from the native macOS install
`out/install/macos-cumetal/release`. The GPU was left to the performance sessions.
Corrections and performance work on the solver are happening in other sessions, so
solver bugs found here are documented with a proposed patch, not committed.

## Reproduce

```bash
# 1. Bake (Shardwright fix/closed-hull-faces, see finding 4) and export with faces and names
prefracture bake --input benchmarks/assets/two_storey_building.glb \
  --meta benchmarks/assets/two_storey_building.glb.meta.json \
  --config benchmarks/configs/support_clusters.toml --out /tmp/sw-verify/bake2
shard-export bake2/two_storey_building.fracphys two_storey_L2.json 2 bake2/two_storey_building.asset.json
# 2. Standalone world (stress-ref) / PhysX CPU as the rigid-body engine
cargo run --release -p stress-ref --example shardwright_destruction -- two_storey_L2.json <idle|ram|projectile|remove> --duration 2 --out OUT
PHYSX_ROOT=$SDK DYLD_LIBRARY_PATH=$SDK/lib cargo run --release -p stress-physx --example shardwright_physx -- two_storey_L2.json <case> --duration 2 --out OUT
# 3. Energy budget (dissipated vs impactor KE + potential released)
python3 findings/energy_budget.py OUT/<case>.scene.json OUT/<case>[.physx].obs.json
# Per-frame balance
cargo run --release -p stress-ref --example energy_trace -- OUT/<case>.scene.json
```

The cases (`stress_ref::shardwright::destruction_case`):

| Case | Load |
|---|---|
| `idle` | Gravity only. |
| `ram` | stress-gpu's `ram_at`: a 400 kg box at 12 m/s into the −x face. |
| `projectile` | frac-sim's default: a steel ball, r = 0.4 m, 15 m/s. |
| `remove` | Sudden removal of the column `f0_col_21` at 0.1 s. |

## Results (PhysX CPU, compound hulls taken as given, 2 s; idle 1 s)

| Case | Broken / 3,635 | Fragments | Available (J) | Dissipated (J) | Budget | Peak frame |
|---|---|---|---|---|---|---|
| idle | 0 | 1 | 213 | ≈0 | OK (max utilization 0.46) | 2.4 s |
| projectile | 14 | 9 | 2.39e5 | 1.03e5 | OK | 3.0 s |
| ram | 3,371 | 1,007 | 3.42e6 | 2.05e8 | **energy created** | 15.9 s |
| remove | 3,571 | 1,134 | −2.1e7 (rose) | 2.9e10 | **energy created** | 15.0 s |

The standalone world agrees on projectile: 15 broken bonds and 10 fragments, budget OK.
It also creates energy on ram (contact: 7.1e8 J) and remove (bond ledger: 2.2e9 J).

Every frame runs the explicit stress solve at about 3,170 substeps per frame on 8
threads. The peaks are destruction frames, with up to 1,071 bodies in impact islands.

## Findings

1. **The pipeline works.**
   - Shardwright `.fracphys` → `shard-export` (hull points and polygon faces, component
     names) → `stress_ref::shardwright::import` (one compound chunk per fragment, hulls
     taken as given) → `World::with_engine` with `PhysxEngine` (one PhysX body per
     cluster, one convex shape per hull).
   - The intact asset stands, and a moderate impact gives local, energy-consistent damage.

2. **The equivalent-inertia boxes overlapped** (importer, fixed in `8ef56d55d`).
   - The first importer replaced each non-convex fragment with one box. 3,001 of the
     3,635 bonded pairs then overlapped at rest, with a median depth of 82 mm.
   - Once bonds broke, the overlap fired as contact energy (column removal dissipated
     525 MJ out of a 4.5 MJ potential).
   - The asset's own hulls do not overlap: 0 of the inter-fragment pairs, checked by LP.
   - Fragments are now compound chunks (`ChunkDesc::compound`): contact is per hull, and
     mass and inertia come from the hulls. stress-gpu refuses them for now.

3. **`ConvexHull::new` was not robust to f32 input** (fixed in `8ef56d55d`; the A7 note
   on a 15/14 face flip).
   - Faces that were planar only to rounding split into overlapping planes, and repeated
     corners made noise triangles.
   - Under a rotation, Shardwright hull volumes changed by up to ×100, sometimes going
     negative.
   - It is now an incremental hull on the exact `robust::orient3d` predicate, with mass
     from the closed triangle mesh. Rotation now changes volume by less than 5e-14.
   - Shardwright hulls no longer go through it: `ConvexHull::from_faces` (`d48fdcc23`)
     takes the asset's faces as given and checks the surface is closed. 3,655 of 3,655
     are accepted; the worst departure from convex is 30 µm, against a 0.5 mm hull margin.

4. **Shardwright emitted a non-closed hull** (Shardwright `fix/closed-hull-faces`,
   `9b5701e`).
   - Fragment 1715 had a triangle face inside a coplanar quad face: duplicate half-spaces
     of a clipped polytope.
   - `stored_hull_is_convex` now also requires a closed surface
     (`frac_geom::polygon::closed_surface`). The output rebuild loop repairs such hulls,
     and the `collision_shapes` gate reports them. The rebake passes every gate.

5. **Split frame fold puts strain into bonds** (stress-ref solver; **not committed**;
   proposed patch `fold-rigid-drift-linear-transfer.patch`).
   - In the standalone column removal, 100% of the bad bond dissipation (2.33e8 of
     2.34e8 J) is booked within 0.2 ms after a split.
   - Single bonds jump from intact to a 23–30 mm opening in one substep and book 2–5 MJ
     each, mostly rebar rupture work.
   - **Cause:** `fold_rigid_drift(exact = true)` re-expresses the world positions
     `center + u` in a frame turned by the finite rotation R(φ). The hidden field is
     linear: a rigid rotation is `u = φ × r`, which is strain-free in the linear bond
     kinematics. Taking out R(φ) leaves O(φ²·L) in the bonds.
   - **Measured:** the mean-translation fold changes bond kinematics by about 1e-17 m;
     the drift fold, at φ up to 0.062 rad, by up to 47 mm (1.44 mm at φ = 0.053 on a
     494-chunk child).
   - **Patch:** displacements and rotations take the linear transfer (`u − t − φ×r`,
     `th − φ`), which leaves every bond's kinematics unchanged. Velocities keep the exact
     world-preserving re-expression, so momentum stays exact. Positions move by O(φ²)
     instead, about 1 mm.
   - **With the patch:** post-split jumps fall to about 1e-16 m, and the column removal
     balances through frame 28 (0.47 s). At frame 29 a separate contact jump of 62 kJ
     appears (contact dissipation 0.5 → 71 kJ in one frame); see finding 6.

6. **Contact in impact islands creates energy in cascades** (open; stress-ref contact).
   - In the coupled ram and removal runs, nearly the whole building ends up in impact
     islands (`max_island_bodies` 959 and 1,071). Their contact is stress-ref's penalty
     contact, not PhysX's.
   - The standalone ram dissipates 7.1e8 J through contact, against 3.7e6 J available.
     The PhysX ram dissipates 2.0e8 J the same way.
   - The fold patch removes the bond-ledger term but not this one. A candidate is the
     per-point offset memory for compound parts (sample indices concatenated over parts),
     or impactor/chunk contact without an overlap memory. Not diagnosed.

7. **Smaller observations.**
   - `max_failure_index` reads `inf` from the first crack onwards in every loaded run
     (some bond's capacity reaches 0), so it says nothing about the rest of the run.
   - Masses: the calibrated densities give 114.1 t against the asset's 115.6 t.
   - Cost: in the explicit solve, about 2.3 s per intact frame at 3,170 substeps;
     destruction frames reach 15 s.

## Files

**Commits on `feat/shardwright-destruction-verify` (PhysX):**

| Commit | What it adds |
|---|---|
| `7cc440a26` | Case runner, component names. |
| `8ef56d55d` | Exact hulls, compound chunks. |
| `d48fdcc23` | Hulls taken as given, `shardwright_physx`. |

**ShardWright:** `fix/closed-hull-faces` (`9b5701e`), in the worktree `../ShardWright-closed-hulls`.

**This directory:**
- `fold-rigid-drift-linear-transfer.patch`: the proposed solver fix (finding 5).
- `energy_budget.py`: the budget check.
