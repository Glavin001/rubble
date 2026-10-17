# Handoff: stress-oracle (2026-10-10)

Read this first, then `README.md` (model, validation tables, known gaps). Treat every
dated status here as a claim to re-verify against the current source and a fresh run.

## 1. What this is and what the owner wants

`destruction/stress-oracle` is a Rust reference ("oracle") for destruction physics:
a rigid-body-spring stress solver (`crates/stress-ref`) validated against closed forms
and four external solvers (OpenCourant/OpenRadioss, OpenSees, LMGC90, Kratos DEM;
results in `golden/`). It also runs with PhysX CPU as the rigid-body engine
(`crates/stress-physx`), and has a scientific debug renderer (`crates/stress-viz`).

The owner's standards, in their words or close to it:

* **Accuracy first.** "I do not want arbitrary caps. I want whatever is physically
  accurate and faithful to reality." Do not tune constants to pass a gate; find the
  cause. Report regressions plainly.
* **End-to-end output is what matters.** For the PhysX path: same inputs, matching
  correct outputs. Re-computing inside the oracle to override PhysX's assumptions is
  fine. Note accuracy differences and anything that needs more than PhysX's public API.
* **Speed matters, but only after accuracy**, and the oracle must stay readable.
* Showcase film: scientific debug view (not realistic colours), realistic behaviour,
  complex structures from convex hulls (up to 64 vertices, no trimesh), PhysX + oracle.
  The owner will run the long simulations on their MacBook (see section 6).

Working rules: branch `clean/stress-oracle`; no pull request unless asked; sibling
checkouts `vibe-land-4` and `blast-stress-solver-2` are read-only; the PhysX backend
adapter in `blast/blast-stress-solver-rs` (`physx_backend.rs`) is shared with the native
GPU pipeline, so change it only with the owner's go-ahead (per-shape friction needs it).

## 2. State at handoff

Branch `clean/stress-oracle`, pushed. Recent commits:

| commit | content |
|---|---|
| `34dc53a` | earlier speedups, adaptive solve with PhysX, convex hull chunks, scene-pack import |
| `2ba97c7` | bit-identical speedups (every benchmark output identical bit for bit) |
| `801aa76` | per-chunk damped step bound, quasi-static mechanism fix, contact-set sharing, pack import fixes |
| (this commit) | this handoff, `scripts/bench_identical.sh`, `examples/energy_trace.rs`, README gap update |

Verification at `801aa76` (4-core container, all sequential):

* `cargo test --release -p stress-ref`: all pass (~12 min). Clippy clean.
* `cargo test --release -p stress-physx`: all pass (~5 min).
* `stress-ref check scenes golden`: **91 of 95 gated metrics pass.** Failures: the three
  b5 40 m/s metrics (long-standing gap, README) and **b5 2 m/s `speed_lost`** (new; see
  section 3.1).
* `stress-physx check scenes golden` (PhysX path vs standalone oracle): **76 of 76 pass,
  but the largest gap grew from 0.03% to 2.7%** (b5 2 m/s `speed_lost` 2.7%, b7 masonry
  debris speed 0.74% and 1.31%). See section 3.2.

## 3. Open work, in priority order

### 3.1 b5 2 m/s: push-over not converged at the default substep (in progress)

Scene `scenes/b5_wall_impact_v02.json`: 1000 kg steel box ram at 2 m/s into an anchored
840-chunk concrete wall, 0.3 s. Gate: `speed_lost` within 30% of OpenCourant (0.951 m/s).

Facts established:

* **First impact is converged**: ram 2.0 -> 0.848 m/s at every Courant safety 0.1-0.5
  (`speed_lost` 1.152, 21% from OpenCourant: passes).
* The ram then follows the wall and **hits it a second time**. Time of that second drop:

  | courant_safety | 0.5 | 0.45 | 0.4 | 0.3 | 0.2 | 0.15 | 0.1 |
  |---|---|---|---|---|---|---|---|
  | second impact (s) | 0.283 | 0.283 | 0.310 | 0.334 | 0.334 | 0.317 | 0.323 |

  Converged ~0.32 s; the default (0.5) is ~12% early, so the 0.3 s window catches it
  (`speed_lost` 1.34-1.50, fails). OpenCourant shows no large second drop before 0.3 s
  (1.070 -> 1.049 m/s between 0.01 and 0.3 s).
* **Not an instability**: the energy balance closes in every run (residual ~18 J of
  30 kJ, flat; `examples/energy_trace.rs`). The second impact releases gravitational
  energy (wall base fails, 66 bonds in a frame), physically.
* With `--set sim.contact_friction=0.0`, safety 0.5 and 0.3 both give 1.152 at 0.3 s
  (second impact after the window). Timing with friction off was not measured.
* Feature ablation, second-impact time at safety 0.5 vs 0.2 (partial when handed over):
  rate effects off 0.283 vs 0.334 (no change: not the cause); damping off 0.300 vs
  0.329 (narrower). Softening off and crack contact off change the physics itself
  (first impact 2.0 -> 1.52 and -> 0.65 m/s), so they are not clean ablations. Static
  fatigue off: not run to completion.

Main suspect, **step-dependent contact friction** (`world.rs`, `penalty_force`): friction
is regularised Coulomb, viscous below `FRICTION_REGULARIZATION = 1e-3 m/s` with
coefficient `mu fn / v_reg`, capped at `MAX_SET_VISCOSITY / max(points, 10) * m_red / dt`
for explicit stability. With `fn ~ 1e4 N` the cap always binds, so below ~0.1 m/s of
slip the friction force is `c_max * v_t`, proportional to `1/dt` and weaker than Coulomb.
How hard sliding debris and cracked faces resist slow sliding then depends on the substep.

Suggested next steps:

1. Finish the ablation; also measure second-impact timing with friction off at 0.5 vs 0.2
   (`--set sim.duration=0.4`). If friction off makes the timing step-independent, the
   suspect is confirmed.
2. Make friction step-independent yet stable. Options: (a) velocity-level stick, giving
   the tangential force that cancels the pair's relative slip within the substep (`m_eff
   v_t / dt`, with the effective mass including rotation at the contact point), clipped
   to `mu fn`, and apportioned when a chunk has several contact sets; (b) an elastic
   tangential spring with Coulomb return mapping (stores slip like the normal penalty
   offset in `pair_offsets`), stable under the same substep as the normal spring. (b) is
   closer to how DEM codes do it and keeps a Coulomb limit independent of `dt`.
3. Accept only when b5 v02 gives the same second-impact time (and `speed_lost`) at safety
   0.5, 0.3 and 0.15 within a few percent, then rerun everything in section 5.

Diagnostic data for these runs was in the session scratchpad and is not carried over;
the commands in section 5 regenerate it.

### 3.2 PhysX path drift (0.03% -> 2.7%)

`world_coupling.rs` picks `self.solver.stable_dt().min(fdt)` when no impact island is
active, while the standalone world also takes `contact_dt` (`World::substep_dt`). So the
two paths take different substeps, and the step-sensitive outcomes of 3.1 differ. Fix
3.1 first, then re-measure; if the drift remains, consider using the same substep rule in
both paths.

### 3.3 Known, documented gaps (README "Known gaps")

* b5 at 40 m/s: hole (ours) vs push-over (OpenCourant); the oracle's far field is not
  mesh-converged and has no crushing. Open.
* No Poisson effect in the chunk network (b6 spall corner ligaments; b9 displacement +24%).
* PhysX backend uses one material (friction 0.25) for every shape; per-shape materials
  need the shared adapter changed (owner's go-ahead).

### 3.4 Showcase film (approved in outline, not started)

Plan agreed with the owner: separate acts joined with title cards, about 2-2.5 min at
1080p, scientific debug view (the existing renderer's heat maps, crack lines, side-by-
side "mechanism off" panels), every act backed by a behaviour test, honest labels ("our
model with X off", never a competitor). Structures from the repository's assets rather
than new ones: `house-2story` (599 hull chunks: brick, stone, timber, glass, steel,
concrete), `comp-*` and `rig-*` components, plus the box-built arch and overhang showcases.
Candidate acts: standing under self-weight (stress paths); wrecking ball fast vs slow;
car into the house; blast beside it (glass first, shadowing); keystone removal (arch);
sustained-load creak-crack-snap (static fatigue, existing overhang showcase).

Still to build: act scenes and tests, renderer captions/title cards, camera motion,
clip assembly (ffmpeg), and a run script for the owner's Mac. Cost: the house's stable
substep is ~3e-7 s (its stiffest timber planks' rotational rows, with stiffness-
proportional damping at zeta ~1), so ~30 min of compute per 0.5 s of impact here.

The villa (`villa-savoye`) does not stand under calibrated materials with code-minimum
reinforcement (its stairs and roof parapets have no static equilibrium), so do not use
it as an intact building without addressing that.

## 4. How the pieces fit (orientation)

* `solver.rs`: chunk states, clusters, explicit substep (phased: rigid accelerations,
  bond responses, chunk loads from the start-of-substep state, then applied in index
  order; `par.rs` maps in parallel above 4096 items, bit-identical for any thread count),
  timestep bound (`chunk_stable_dts`), splitting.
* `joint.rs`: the joint law (tension cutoff, Mohr-Coulomb, crushing, buckling cap, DIF,
  Weibull, static fatigue, softening, crack-contact spring patch, rebar).
* `statics.rs`: quasi-static equilibrium (PCG), `equilibrate_or_keep` (a cluster with no
  equilibrium keeps its last one; never commit damage from a non-converged iterate).
* `world.rs`: scene runner: contacts (impactor, ground, chunk pairs; penalty with per-set
  shared stiffness and viscosity), blasts, events, observation. `world_coupling.rs`: the
  PhysX-coupled path (impact islands, verification and redo).
* `hull.rs`, `contact.rs`: convex hulls and box/hull contact sampling, separating-axis test.
* `scene_pack.rs`: import of `packs/*.json` (from `blast/blast-stress-demo-rs/assets/scenes`) (Y-up to
  Z-up; ground at the assets' grade y = 0; chunks reaching grade are supports; materials
  mapped to calibrated presets; reinforced-concrete joints get EN 1992 minimum rebar).
  `examples/import_pack.rs` writes a pack as a scene file.
* `crates/stress-physx`: `PhysxEngine` (implements `RigidEngine`), `stress-physx` CLI.

## 5. Commands

```bash
cd destruction/stress-oracle
cargo build --release -p stress-ref                    # oracle CLI: target/release/stress-ref
cargo test  --release -p stress-ref                    # ~12 min on 4 cores
cargo clippy --release -p stress-ref --all-targets
target/release/stress-ref check scenes golden --json check.json   # ~9 min, gated table

# PhysX path (needs a PhysX CPU SDK build; this container had it at /home/user/physx-cpu-install)
export PHYSX_ROOT=/path/to/physx-cpu-install
cargo test  --release -p stress-physx -- --test-threads 3         # ~5 min
target/release/stress-physx check scenes golden --json physx_check.json   # ~14 min

# One scene with overrides, and the energy trace
target/release/stress-ref run scenes/b5_wall_impact_v02.json \
  --set sim.courant_safety=0.3 --set sim.duration=0.45 --out v02.json
cargo run --release -p stress-ref --example energy_trace -- scenes/b5_wall_impact_v02.json 0.5

# Speed work: bit-identity and wall time against a baseline build's output directory
scripts/bench_identical.sh target/release/stress-ref /tmp/bench_new /tmp/bench_baseline
```

Pitfalls learned the hard way:

* Benchmark only on a quiet machine; a concurrent test run inflated timings by 30-60%.
* Parallelism lost on 4 cores (rayon wake-up costs more than the work at <= 2000 bonds):
  check `RAYON_NUM_THREADS=1` against the default before claiming a parallel gain.
* `pkill -f` / `pgrep -f` patterns match the invoking shell; use
  `ps -eo pid,args | awk '/pattern/ && !/awk/'`.
* Instruction counts (callgrind) and wall time disagreed by 3x on the contact path;
  confirm with wall time (or gdb stack sampling) before optimizing.
* `World` is not `Sync` (it can hold the PhysX engine): parallel code takes the solver
  and scene by reference via free functions (see `pair_contact`).

## 6. Running long simulations on the owner's MacBook

This repository's PhysX 5 has build presets for Linux (x86-64 and aarch64) and Windows
only, no macOS. The oracle and renderer are plain Rust and build natively on macOS
(oracle-only runs). For the PhysX path, use Docker Desktop with an aarch64 Linux image
and the `linux-aarch64-clang-cpu-only` preset (`physx/buildtools/presets/public/`), then
set `PHYSX_ROOT` to the install (headers incl. `PxConfig.h`, static libs; see README
"PhysX"). Run one act per core in parallel rather than threading one run. Before
handing over a script, ask the owner for their chip, core count and memory and whether
Docker is installed (asked, not yet answered).
