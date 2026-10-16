# Hand-off: Shardwright full destruction on the new stress solver

For a session that continues the fixes, or verifies fixes made elsewhere. What was found
and measured is in [SHARDWRIGHT_DESTRUCTION.md](SHARDWRIGHT_DESTRUCTION.md); this file
is the how. Written 2026-10-11. Check the dated status against current source first:
other sessions are correcting the solver and optimising it.

**Goal:** a Shardwright asset (`two_storey_building`) runs idle, ram, projectile and
column removal through the stress solver with PhysX CPU as the rigid-body engine. Every
case must pass its energy budget: dissipated ≤ impactor kinetic energy + potential
released (5% slack).

## Status

| # | Item | Where | Status |
|---|---|---|---|
| 1 | Importer: fragments as compound chunks of their own hulls (no box overlap) | stress-ref `shardwright.rs`, `contact.rs`, `structure.rs`, `scene.rs` | Done, committed |
| 2 | `ConvexHull::new` exact (`robust::orient3d`); `ConvexHull::from_faces` takes the asset's faces as given | stress-ref `hull.rs` | Done, committed |
| 3 | Shardwright hulls must be closed surfaces | Shardwright `frac-geom`, `frac-collision`, `frac-validate` | Done, committed (not merged) |
| 4 | Split frame fold puts O(φ²L) strain into bonds | stress-ref `solver.rs` `fold_rigid_drift` | **Patch proposed, not applied** |
| 5 | Impact-island contact creates energy in cascades | stress-ref `world.rs` contact | **Open, not diagnosed** |
| 6 | `max_failure_index` is `inf` after the first crack | stress-ref solver | Open, minor |
| 7 | stress-gpu has no compound chunks (refuses them) | stress-gpu `world.rs` `unsupported` | Open, by design for now |

Pass/fail on the last full run (before item 4), PhysX CPU, 2 s:

| Case | Budget |
|---|---|
| idle | OK |
| projectile | OK |
| ram | Fails: 2.0e8 J dissipated through contact, 3.4e6 J available. |
| remove | Fails: 2.9e10 J dissipated, pieces end higher than they started. |

## Where things are

**PhysX worktree:** `~/Development/PhysX/.claude/worktrees/shardwright-verify`, branch
`feat/shardwright-destruction-verify`. Work from `destruction/stress-oracle`. It was
branched from `86ee3026b`, which comes before the `perf/stress-realtime` experiments.

| Commit | What it adds |
|---|---|
| `7cc440a26` | Case runner, component names. |
| `8ef56d55d` | Exact hulls, compound chunks. |
| `d48fdcc23` | Hulls taken as given, PhysX CPU runner. |
| `c60bf68ff` | Findings. |

None of it is pushed or merged.

- **Shared solver files touched:** `hull.rs`, `contact.rs`, `structure.rs`, `scene.rs`,
  `world.rs`, `world_coupling.rs`, `shardwright.rs`, stress-gpu `loads.rs`/`world.rs`.
  The `brave-mendel-e3a2f0` worktree had uncommitted edits in several of these.
  Expect conflicts.

**Shardwright worktree:** `~/Development/ShardWright-closed-hulls`, branch
`fix/closed-hull-faces`, commit `9b5701e` (on `f10ce21`). The main `ShardWright`
checkout belongs to another session (`structures-vibe-land`, uncommitted work); don't
edit it.

**PhysX CPU SDK** (native macOS; no Docker and no GPU needed):
`SDK=~/Development/PhysX/out/install/macos-cumetal/release`. HANDOFF.md §6 ("no macOS
preset, use Docker") is out of date. `PhysxEngine::new` / `boxed` run on the CPU from
this install. Build with `PHYSX_ROOT=$SDK` and run with `DYLD_LIBRARY_PATH=$SDK/lib`.

**Ephemeral artefacts** in `/tmp/sw-verify` (bake, exports, run outputs) may be gone.
Regenerate them as below.

## Regenerate inputs (about 2 min)

```bash
W=~/Development/PhysX/.claude/worktrees/shardwright-verify/destruction/stress-oracle
SW=~/Development/ShardWright-closed-hulls          # or wherever fix/closed-hull-faces is checked out
OUT=/tmp/sw-verify; mkdir -p $OUT

# Bake with the closed-hull fix (gates must all PASS)
(cd $SW && CARGO_TARGET_DIR=$OUT/sw-target cargo build --release -p frac-cli)
(cd $SW && $OUT/sw-target/release/prefracture bake --input benchmarks/assets/two_storey_building.glb \
   --meta benchmarks/assets/two_storey_building.glb.meta.json \
   --config benchmarks/configs/support_clusters.toml --out $OUT/bake2)

# Exporter: its Cargo.toml path deps assume Shardwright beside the *main* PhysX checkout,
# so build a copy pointed at $SW (scripts/shardwright_verify.sh does the same).
rm -rf $OUT/export-src && cp -R $W/tools/shardwright-export $OUT/export-src
sed -i '' "s#../../../../../Shardwright#$SW#" $OUT/export-src/Cargo.toml
CARGO_TARGET_DIR=$OUT/export-target cargo build --release --manifest-path $OUT/export-src/Cargo.toml
$OUT/export-target/release/shard-export $OUT/bake2/two_storey_building.fracphys $OUT/two_storey_L2.json 2 \
   $OUT/bake2/two_storey_building.asset.json
#   expect: 1848 fragments (levels 0..=2), 3658 bonds at level 2
```

The export carries each hull's `points` and `faces`, plus `component_names`, which
gives the `name:f0_col_21` groups that column removal needs.

## Run

```bash
cd $W
SDK=~/Development/PhysX/out/install/macos-cumetal/release
# PhysX CPU as the rigid-body engine (the production path)
PHYSX_ROOT=$SDK CARGO_TARGET_DIR=target/physx-cpu cargo build --release -p stress-physx --example shardwright_physx
for c in idle ram projectile remove; do d=2; [ $c = idle ] && d=1
  RAYON_NUM_THREADS=8 DYLD_LIBRARY_PATH=$SDK/lib target/physx-cpu/release/examples/shardwright_physx \
    $OUT/two_storey_L2.json $c --duration $d --out $OUT/physx > $OUT/physx/$c.summary 2> $OUT/physx/$c.log
done
# Standalone world (oracle): same options
cargo build --release -p stress-ref --example shardwright_destruction
target/release/examples/shardwright_destruction $OUT/two_storey_L2.json <case> --duration 2 --out $OUT/ref

# Energy budget (needs numpy + scipy; e.g. uv run --with numpy --with scipy python ...)
python findings/energy_budget.py $OUT/physx/<case>.scene.json $OUT/physx/<case>.physx.obs.json
```

- **Cost:** about 2.3 s per intact frame on 8 threads (explicit solve, about 3,170
  substeps per frame). Destruction frames reach 15 s. A 2 s case takes 10–25 min, so run
  the cases one after another, in the background.
- **GPU:** don't run GPU jobs alongside the GPU performance sessions (AGENTS.md).
- **Shorter runs** for quick checks: `--duration 0.5`. The first big split in column
  removal happens at 0.449 s.

## Item 4: split fold strain (fix, then verify)

- **Symptom:** in column removal, the bond-dissipation ledger jumps in single frames.
  Frame 47 books +2.3e8 J while mechanical energy barely moves. All of it is booked
  within 0.2 ms after a split.
- **Cause:** `ReferenceSolver::fold_rigid_drift(ci, exact = true)` (called from
  `split_cluster` under `methods.exact_frame_transfer`) re-expresses the world positions
  `center + u` in a frame rotated by the finite rotation R(φ). The hidden field is linear
  (a rigid rotation is `u = φ×r`, zero strain in the linear bond kinematics), so this
  leaves O(φ²·L) strain in the bonds.
- **Patch:** `findings/fold-rigid-drift-linear-transfer.patch` (`git apply` from the
  repository root). It applies the linear transfer to `u` and `th` and keeps the exact
  velocity re-expression.

**Verify:**
1. Apply `findings/split-debug-instrumentation.patch` instead; it contains the same fix
   plus the instrumentation. Then run, from `$W`:
   ```bash
   python3 -c "import json;s=json.load(open('$OUT/physx/remove.scene.json'));s['sim']['duration']=0.5;json.dump(s,open('$OUT/remove_0.5.json','w'))"
   cargo build --release -p stress-ref --example energy_trace
   STRESS_DEBUG_SPLIT=1 target/release/examples/energy_trace $OUT/remove_0.5.json > $OUT/trace.txt 2>&1
   grep '^CHILD' $OUT/trace.txt     # "drift-fold" jump per child: must be ~1e-16 (was up to 4.7e-2 m)
   grep '^FOLD'  $OUT/trace.txt     # |phi| at each fold (up to 0.062 rad here)
   grep '^BOND'  $OUT/trace.txt     # single bonds dissipating >10 kJ in one evaluation
   grep '^frame' $OUT/trace.txt     # residual = mechanical + dissipated - initial
   ```
2. **Pass:** the residual stays within a few kJ of −9.4 kJ (the removed column's own
   energy) through frame 28. No `BOND` lines appear right after `SPLIT` lines.
   - With the patch, the residual jumps to +61.8 kJ at frame 29. That jump is contact
     (contact dissipation 0.5 → 71 kJ), item 5, not the fold.
3. **Regression:** `cargo test --release -p stress-ref` covers the library, `hulls`,
   `principles` and `conservation`; the conservation tests exercise splits. Also run
   `stress-ref check scenes golden` (generate scenes with `stress-ref gen-scenes scenes`)
   for the golden catalogue, and the stress-gpu gate (`stress-gpu-gate --micro`, GPU,
   when the GPU is free).
   - stress-gpu's mirror runs the same `split_cluster`, so GPU results can move too.
4. **Before applying:** check whether another session has already changed
   `fold_rigid_drift`. If so, run step 1 with only the instrumentation part of the debug
   patch.

## Item 5: impact-island contact energy (diagnose)

- **Symptom:** ram, both standalone and PhysX-coupled, dissipates 2–7e8 J through
  contact against 3.4–3.7e6 J available. Coupled column removal ends with pieces higher
  than they started.
- **Context:** in these runs nearly the whole building ends up in impact islands
  (`max_island_bodies` 959 and 1,071 in the observation `values`), so contact is
  stress-ref's penalty contact, not PhysX's.

Suggested path:
1. Apply item 4's fix first, so the bond ledger is clean.
2. Run `energy_trace` on the ram scene (`$OUT/physx/ram.scene.json`, or `ref/ram.scene.json`)
   and find the first frame where the residual rises with the contact term.
3. Suspects, in order:
   - **The overlap memory for compound parts.** `pair_offsets` holds one entry per
     sample point, with sample indices concatenated over a compound's parts
     (`OBox::for_each_sample` in `contact.rs`). Check that the indices stay stable while
     the active set changes.
   - **Impactor–chunk contact has no pre-existing-overlap memory.** Only chunk pairs have
     one; see `apply_contacts` in `world.rs`.
   - **The step bound for compound chunks uses the bounding box.** That's
     `contact_stable_dt` and the `half` extents. A chunk thinner than its box gets a
     stiffer real contact than the bound assumes.
4. Instrument as in item 4: print pairs whose contact work in one substep exceeds a
   threshold, with their kind (pair or impactor), depth and offset.

**Pass for the whole effort:** all four PhysX CPU cases pass `energy_budget.py`, and
idle stays at 0 broken bonds with max utilization 0.46. Then compare PhysX against
standalone on projectile (now 14 broken bonds / 9 fragments against 15 / 10) and on ram
and remove once they balance.

## Not done / later

- **PhysX gets hull points, not polygons.** It cooks its own convex mesh, which is the
  same hull. Passing Shardwright's polygons would need a `ShapeGeom` variant in
  `blast-stress-solver-rs`.
- **stress-gpu compound chunks:** the GPU contact samples one box or hull per chunk.
  Needed before a GPU run of Shardwright assets can be compared.
- **Merging:** `fix/closed-hull-faces` needs review into Shardwright `main`;
  `feat/shardwright-destruction-verify` needs reconciling with
  `integration/oracle-stress-gpu` / `perf/stress-realtime`.
