# Destruction reactions

Agreed with the user on 2026-10-25. One strike, blast or collapse can produce every reaction a
real structure shows, and each is decided by the engine and reported as an event:

| # | Reaction | Decided by | Event |
|---|---|---|---|
| 1 | **Crushed**: the chunk's material is comminuted and it stops being a rigid chunk | Compressive overstress of the chunk, integrated to damage | `ChunkCrushed` |
| 2 | **Knocked off**: bonds break under the contact or impact force | Threshold predictor / breaks (exists) | `BondsBroken`, `Split` |
| 3 | **Blast loading**: overpressure on exposed faces | Friedlander face pressures (oracle `loads.rs`), now in `World` | the breaks it causes |
| 4 | **Support loss**: the weight left without a load path breaks bonds, pieces tip | `support_loss.rs` load flow and tipping | `BondsBroken`, `Split` |
| 5 | **Dynamic amplification** of a sudden loss | explicit, opt-in | — |
| 6 | **Debris loads** on the structure below | contact-to-load path (exists) | — |

Rubble decides and reports; the host draws dust. Simulating dust particles is optional.

## Rules every part follows

- **Forward-only** (AGENTS.md): a strike's crushes are predicted with its breaks at the end of the
  frame before it and applied before the engine steps. No rewind, no re-solve.
- **Capability flags.** Each reaction is an opt-in switch (`SolverConfig`, `Methods`,
  `SimFeatures` or the `DestructibleDef`), off by default so every existing golden and digest is
  unchanged. Not part of `blast_parity`.
- **No silent defaults.** A crush-enabled material without crush parameters is an error, never a
  guessed value.
- **No lossy surprises.** A crushed chunk's mass is **conserved by default**: it leaves the
  structure as a few small rigid fragments carrying its mass and momentum, so its weight still
  loads the floor below. Removing the mass entirely (dust) is an explicit `CrushMass::Remove`.
- **Energy.** The impactor pays the crush energy (`dD · crushEnergy · V`) out of its kinetic
  energy, before the step, as `J = M (v − sqrt(v² − 2 dE / M))`, capped at `M v` (it can stop a
  striker, never bounce it).

## 1. Chunk crushing

The model is the one PhysX's Blast stress extension and vibe-land ship
(`extStressCrushStep`; PhysX `blast/blast-stress-solver/docs/chunk-crushing.md`), written fresh
for rubble:

```
sigma = (1/V) sum sym(branch (x) force)      contact and bond tractions, no body forces
p = -tr(sigma)/3 (compression positive)      q = sqrt(1.5 s:s)
crushing needs p > 0
excess = max(q - (cohesion + frictionSlope p), p - capPressure)
D += excess^2 dt / (crushViscosity crushEnergy)       crushed at D >= 1
```

Material parameters are per material (`crush` block), derived from the material's own
compressive strength with cited sources (vibe-land `structures/town-kit/src/materials.mjs`
`CRUSH`: EN 1996-1-1, EN 1992-1-1, Bond 1961 comminution energy, CEB-FIP MC90 rate factor).

`ChunkCrushed { destructible, chunk, mass, volume, pose, linear_velocity (at the chunk),
angular_velocity, peak_pressure, peak_deviator, cause: Impact | Blast | Contact | Load }`.

## 3. Blast loading (shipped 2026-10-25)

`World::detonate(&BlastDef { position, tnt_kg, radial })` loads every threshold destructible
placed with `DestructibleDef::blast_loading` (off by default). Code: `engine-world`
`src/blast.rs` (API and the per-chunk window model) and `src/blast_threshold.rs` (faces,
breaks, split, impulses; a child of `threshold.rs`'s `imp`).

- **Physics is the oracle's.** `stress_ref::blast::FaceBlast` per exposed face (Kinney-Graham,
  Friedlander, oblique reflection, `SHADOW_FACTOR`, clearing); exposure and clearing distances
  are stress-gpu `loads::exposure` / `clearing_distances` on a mirror of the oracle solver with
  the GPU's bond states; shadowing culls occluders by reach as stress-ref `rebuild_blast`
  does, against every blast-loaded destructible's chunk boxes at their GPU poses. Faces are
  rebuilt when the topology changes, as the oracle does; a face turned away carries no
  pressure and is skipped.
- **Forward-only.** The detonation is blast time 0 at the start of the next step. Each step its
  pulse reaches, before the engine steps: the host waits for the GPU (readbacks in flight),
  evaluates each chunk's load over the step's window, breaks bonds by the threshold rule
  (`fr_breaks`: either chunk's load above the tensile capacity; the load is the window's
  peak, as `impact_peak` takes an impact's), splits on the GPU (`GpuFracture::break_bonds` +
  `encode_split`), adopts the pieces (re-framed from this step) and queues each piece's share
  of the window's impulse (linear and angular, about its centre of mass). The events arrive
  in that step's frame. Nothing is re-simulated.
- **Composes with crushing.** Each chunk's load (`ChunkBlastLoad`: peak force, its moment,
  `Σ sym(r ⊗ F)` over the faces at the largest face pressure, impulse, angular impulse, peak
  pressure, `LoadCause::Blast`) is reported by `World::destructible_blast_loads` and written
  to the slot's GPU buffer `BlastSlot::loads` (16 floats per chunk: force + cause code;
  moment + peak pressure; σV xx yy zz xy; σV xz yz), zero in steps no blast reaches.
- **Errors, not caps.** `tnt_kg` must be positive and finite, the position finite; a loaded
  face nearer than the fits' `Z = 0.05 m/kg^(1/3)` fails `detonate`; at most `BLAST_CAP` (16)
  blasts at once; a blast pushing more bodies than one step's impulse records (1024) fails
  the step.
- **Free bodies** take `explode`'s radial impulse when `BlastDef::radial` is set (it reaches
  every shape its mask meets: leave destructibles' categories out of it for their pieces to
  take only the face-pressure impulse).
- **Tests** (`engine-world` `blast` unit tests and `tests/blast.rs`): reflected Kinney-Graham
  peak at the scaled distance; a face's impulse is the fitted Friedlander integral (incident
  impulse × reflection factor × area, × cos θ toward the charge) and windows sum to it; the
  World chunk facing the charge takes exactly that peak force and impulse; the oracle's
  two-wall showcase (facing wall breaches, shadowed wall survives, alone it breaches; 60 kg
  where the oracle's stress solve takes 30, since the threshold rule is local); a far wall
  survives the charge that breaches a near one; the knocked-off pieces' momentum equals the
  impulse their faces took; with the flag off the world is unchanged.

## Tests and oracle

Each part has unit tests from the law's definition and physical invariants (a structure under
its own weight never crushes; tension never crushes; damage is quadratic in overstress; mass and
momentum are conserved in `Conserve`; the impactor's energy drops by the crush energy), and the
crush-wall and reference-building-crush scenes run end to end with every reaction counted.

### Reference rigs and the reactions harness

The two P0 reference rigs of `docs/scene-port-catalog.md` are rubble scene packs in
[`engine-webgpu/fracture/packs/`](../crates/engine-webgpu/fracture/packs/), ported from
blast-stress-solver (`perf/rubble-field-60hz`) by `port-reference-packs.mjs`:

| Pack | Chunks / bonds | Shots (`defaults.shots`) |
|---|---|---|
| `crush-wall` (v3, crush on `block-wall`) | 216 / 464 | `1.0x`, `1.1x`, `1.4x` (speed scales; 4.8 t, r 0.45) |
| `reference-building-crush` (v3) | 64 / 132 | `ordinary`, `2x`, `8x`, `16x` (mass scales; 2 t at 18.9 m/s, r 0.5) |
| `reference-building` (v2, the crush-off twin) | 64 / 132 | as above |

The packs are as authored (the `crush` blocks and `nodes[].m` kept) plus `defaults.shots`:
the source demo's own launcher for one structure and one wave, as data (origin, velocity,
spin, mass, radius, launch after the 2 s settle), read by `engine_fracture::pack::pack_shots`.

**Harness.** `engine-world/tests/support/reactions.rs` runs a pack as a
`StrengthModel::Threshold` destructible, fires a shot and counts per frame: bonds broken,
splits, pieces created, actors, support-loss breaks, crushed chunks and mass, blast breaks,
the impactor's speed and kinetic energy, and the system's mass (every actor plus the
impactor). The crush, blast and support counters are hooks behind the harness features
`reactions-crush`, `reactions-blast`, `reactions-support` (off; they read zero, or `n/a`)
until those branches' events land.

```sh
cd engine
cargo run --release -p engine-world --example destruction_reactions -- --list
cargo run --release -p engine-world --example destruction_reactions -- \
  crush-wall:1.4x reference-building-crush:8x reference-building-crush:gravity \
  --seconds 8 --json /tmp/reactions.json --md /tmp/reactions.md
cargo test --release -p engine-world --test reactions -- --test-threads=1
```

The tests gate invariants: both packs load (crush blocks iff v3, one stress chunk per node,
shots aimed at the structure), the reference building stands 5 s under gravity with no
bond broken, mass is conserved every frame through every split, an ordinary shot on each
rig breaks bonds and knocks pieces off (none before it arrives, and the impactor leaves
slower), and two runs give identical events and per-frame counts.

**Lab and recording.** The Fracture Lab's Scene menu has a *Reference rigs* group; the URL
`fracture/?pack=crush-wall&shot=1.4x` (`shot=1.4` also matches; `bonds=stress` for
`blast_parity`, default `threshold`) loads a rig in the stress-solver mode and fires the
authored shot at its launch time; the panel's *Authored shot* fires one by hand.
`fracture/record-reactions.mjs crush-wall:1.4x [--seconds 5] [--png] [--out DIR]` records it
frame-locked in headless Chrome to an MP4 (and PNGs), by default under `$TMPDIR/rubble-reactions`.

**Reference observations, not gates.** blast-stress-solver's `docs/chunk-crushing.md` (PhysX,
CPU, rewind/resimulation pass 1, a different engine): reference building 0/64 crushed under
gravity, ordinary impact 0, 2x 3, 8x 4, 16x 7; crush-wall at 1.4x with crush off the plug is
blown through and the ball exits at 36.5 m/s, with crush on the wall absorbs 5.2 of the
ball's 6.3 MJ and the ball leaves at 2.7 m/s. Rubble is forward-only and has no crushing
yet; its numbers are measured, not tuned toward these.

Rubble on this branch (threshold model, Metal, 8 s, crushing not yet merged; speed out is the
last frame's, after the ball has rolled or come to rest):

| Scene | Shot | Speed in → out m/s | KE in → out MJ | Bonds broken | Splits | Actors at end |
|---|---|---|---|---|---|---|
| crush-wall | gravity | — | — | 0 | 0 | 1 |
| crush-wall | 1.0x | 36.7 → 5.3 | 3.24 → 0.07 | 161 | 4 | 57 |
| crush-wall | 1.1x | 40.4 → 2.2 | 3.92 → 0.01 | 13 | 1 | 4 |
| crush-wall | 1.4x | 51.5 → 11.6 | 6.35 → 0.32 | 246 | 2 | 85 |
| reference-building-crush | gravity | — | — | 0 | 0 | 1 |
| reference-building-crush | ordinary | 18.9 → 2.0 | 0.36 → 0.00 | 14 | 3 | 5 |
| reference-building-crush | 2x | 18.9 → 2.0 | 0.71 → 0.01 | 13 | 3 | 4 |
| reference-building-crush | 8x | 18.9 → 0.1 | 2.86 → 0.00 | 35 | 6 | 9 |
| reference-building-crush | 16x | 18.9 → 2.0 | 5.72 → 0.06 | 13 | 3 | 4 |

Structure mass is conserved in every run (crush-wall 27,033.5 kg, the building 67,127.0 kg:
the packs' dynamic masses plus the static footing chunks at their material's density).
