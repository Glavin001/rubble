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

### As built (threshold mode, 2026-10-25)

**Switch.** `DestructibleDef::crush: Option<Crush>` (`StrengthModel::Threshold` only; `None`
by default). `Crush { mass: CrushMass, materials, max_per_step }`:

- `CrushMass::Conserve { fragments, budget }` (default 4 fragments, 256 reserved fragment
  bodies) or `CrushMass::Remove`.
- `materials`: `CrushMaterial` by the stress import's chunk material name (a pack import's
  per-node copies such as `concrete#mass12` match their base name), supplying or replacing
  the import's own.
- `max_per_step`: crushes one step reports at most (0: 256).

A material without parameters never crushes (stress-ref `Material::crush: None`). Turning
crushing on where no chunk can crush is an error naming the materials. Exceeding the
fragment budget or the per-step crushes fails the step with an error naming the limit.

**Parameters.**
- stress-ref `crush::CrushParams` on `Material::crush`; `crush::presets` is the cited table
  (vibe-land `CRUSH`: brick masonry, gypsum, concrete, glass, stone masonry, softwood, roof
  tile, slate, with `bond_energy` and `mc90_viscosity`). `presets::for_material` maps the
  calibrated materials (concrete, brick, stone, glass, timber; steel and mortar never crush).
- ScenePack v3 import: `materials[].crush` (camelCase), chosen per chunk by `nodes[].m`.
  - A crush block in a v1/v2 pack is an error, as is a missing required field.
  - Without crush blocks the import is unchanged.
  - `strainRateExponent` gets an import note: rate hardening is not modelled in v1.
- `strength_scale` scales bond strengths, not crush parameters.

**Where it runs.** `stress-gpu` `fracture.slang` (`GpuFracture`), which both `World` and the
lab's `ThresholdWorld` run; `crush.slang` holds the law, mirrored in stress-ref
`crush::crush_step_f32` and compared on the GPU (`tests/crush_parity.rs`). The resident
`GpuWorld`'s own threshold kernels (`threshold.slang`) do not crush.

**The stress estimate.** The threshold mode solves no bond forces, so the virial is built from
equilibrium. A chunk's external forces at their points are its frame's contact loads, as
`fr_loads` defines them:
- the sustained forces and the carried pulses' means, for the frame;
- each new impact at its Hertz peak force, for its Hertz duration (capped at the frame);
- for a predicted strike, the predictor's Hertz peak force `m_eff v / tau` at the contact,
  for `tau`.

They are balanced by reactions through the chunk's intact bonds, shared by bond area at the
bond centroids. Since `sym()` is bilinear, those reactions add `sym(r_A (x) -F)`, where `r_A`
is the bonds' area-weighted centroid relative to the chunk and `F` the net external force. A
chunk with no intact bond has only its inertia as a reaction: the net force is balanced at the
loads' own (|f|-weighted) centroid, so one contact or one strike never crushes a lone chunk.
Body forces (gravity, inertia) are excluded, as the law says.

Consequences:
- Loads through bonds alone (a column under a building's weight) are not estimated: there is
  no bond solve. Cause `Load` is reserved, as is `Blast`.
- Area share counts a block's side bonds too. On the free wall's bottom row the estimate came
  to the same order as `W / 3A`: measured 1.06 W/(3A) at peak, settling included.
  A projected share (`A |n . F|`) would load only the bonds facing the load; not built.

**Forward only.** Everything runs in the fracture's encode after the engine's step and before
the next one; nothing is rewound.
1. `fr_crush_loads` integrates the frame's contact loads, before the frame's bond breaks.
2. The predicted strike (`fr_strike_crush`) loads the struck body's entry chunk, the one
   `fr_strike_nearest` picks for the strike's bond breaks. It runs before `fr_strike_force`,
   so the struck chunk is crushed before the engine's step meets it.
3. A crushed chunk's bonds break with the frame's other breaks (`fr_crush_bonds`). The split
   then leaves it alone on a body.
4. After the split, `fr_crush_apply` parks that body out of the world and writes the event.
   With `Conserve` it also turns `fragments` reserved bodies into boxes:
   - a grid of the chunk's box, halved along its longest cell until there are enough;
   - centred on its centre of mass and shrunk to at most its volume;
   - each with its share of the mass, at the chunk-point velocity `v + w x r`.
   So the fragments carry its mass and momentum exactly. Ranks come from a scan, so event
   order and fragment slots are deterministic.

**Energy.** The strike's damage increment books `dD crushEnergy V` to its shot (`fr_strike_crush`).
Each shot pays its booked total once, as `J = M (v - sqrt(v^2 - 2 dE / M))` capped at `M v`,
against its speed (`fr_crush_bill`). The struck dynamic bodies receive `J` by their share of
the energy, at the contact (`fr_crush_push`), so momentum is kept. Contact-path crushes
charge nobody: debris, sustained pressure and impacts the predictor did not see are already
inside the engine's step.

**v1 limits.**
- A strike crushes only its entry chunk; the neighbours its ball also covers crush from the
  next frame's contact loads, uncharged.
- Strain-rate hardening is omitted.
- The lab (`ThresholdWorld::new_with_crush`, Fracture Lab `Crush`) removes crushed chunks;
  it keeps no fragments. It reads each frame's crush events back with its counters
  (`ThresholdWorld::take_crush_events`) and draws a dust puff at each (stress-web
  `stress_dust`, render only); its status line counts them.

**A finding, fixed 2026-10-25.** The strike prediction picked its entry chunk by matching a
distance recomputed in a second kernel bit for bit (`fr_strike_entry`, `fr_strike_nearest`).
Metal's fast-math compiles the same expression differently in the two kernels, so on some
strikes no chunk matched and the strike was dropped: the contact broke the bonds a frame late
(on the crush wall, the pack's 1.0x shot broke its first bonds in frame 18 rather than 17;
the support-loss frame's shot from exactly one frame's travel met its column still welded).
`fr_strike_entry` now stores each chunk's distance and `fr_strike_nearest` compares the stored
bits, so the minimum always has its chunk. Crushing takes the same entry chunk (its own
`fr_crush_entry` and `fr_crush_nearest` are gone). Tests: `engine-world`
`tests/strike_prediction.rs`, and `tests/support_loss.rs` back at its 2.5 m start.

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

Crushing's, as built:
- stress-ref `crush::tests`: the law. Uniaxial yield at fc, tension never crushes, damage
  quadratic in overstress, the cap, the f32 mirror, the impulse, the presets.
- `tests/crush_pack.rs`: the v3 import.
- stress-gpu `tests/crush_parity.rs`: the Slang law against the f32 mirror.
- engine-world `tests/crush.rs`:
  - the free wall under its own weight never crushes, and its bottom row's pressure matches
    `W / 3A`;
  - the pack's 4.8 t ball crushes the struck block with cause Impact and pays exactly its
    crush energy;
  - `Conserve` keeps the free wall's mass and the system's momentum in zero g;
  - `Remove` takes the chunk out;
  - crushing that crushes nothing changes no body bit for bit;
  - missing parameters and the wrong model are errors.
- stress-rubble `tests/threshold_crush.rs`: the lab's world.

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
## 4. Support loss (`DestructibleDef::support_loss`, threshold model, opt-in)

`engine-world/src/support_loss.rs`, hooked after each frame's breaks in `threshold.rs`. The
pieces those breaks touched carry their weight down their intact bonds to their anchors: a flow,
breadth first from the supports, each chunk handing its weight and what reached it from above to
the neighbours one bond nearer an anchor in proportion to those bonds' compressive capacities
(`compressive_strength * area`). A bond whose load exceeds its capacity breaks with the next
frame (`GpuFracture::break_bonds_sparse`), so a collapse cascades forward a frame at a time
(after a sudden loss of support, with the dynamic factor on the load's increase: section 5).
Tipping: a piece whose centre of mass, along gravity, is outside the hull of its loaded support
bonds lets go of the members standing on those supports (their bonds into it break). The flow
conserves weight (the load reaching the anchors is the piece's weight); it has no bending, no
tension and no redundancy beyond the sharing. An error under the Bonds model. C ABI:
`RbDestructibleDef::support_loss`.

## 5. Dynamic amplification (`DestructibleDef::support_loss_dynamic_factor`, opt-in)

The flow is quasi-static. A load suddenly applied to an elastic system overshoots its static
effect: the single-degree-of-freedom step response peaks at `1 + exp(-zeta pi / sqrt(1 - zeta^2))`
times the step (2 undamped). There are two conventions for applying that to a support loss:

- **On the whole post-loss demand.** Progressive-collapse guidance's linear static procedure
  multiplies the gravity load over a removed column by a dynamic increase factor of up to 2
  (GSA 2003 *Progressive Collapse Analysis and Design Guidelines*, `2(DL + 0.25 LL)`; DoD UFC
  4-023-03 *Design of Buildings to Resist Progressive Collapse*, load increase factor
  `Omega_LD`). It is a design check: conservative for a member that already carried load.
- **On the change of load.** The step response itself: a member that carried `before` and
  carries `after` once the support is gone peaks at `before + k (after - before)`.

We use the second. The pass predicts what an event breaks rather than sizing members, so it
follows the physics; it is exact for a member that carried nothing (`k after`) and adds nothing
for one whose load does not change. With factor `k`, a piece whose supports changed this frame
(the piece it was part of before held anchors it no longer holds; a chip knocked off changes
nothing) breaks a bond whose demand `before + k max(after - before, 0)` exceeds its capacity.
`before` is the bond's flow load the frame before the loss: the pass keeps every bond's load
from the last time it ran on the bond's piece, starting from the intact structure's flow at
placement; a bond that carried nothing has `before = 0`. Only an increase is amplified: a bond
whose load drops peaks at `before`, which it already held. Range 1 to 2, anything else an
error; `k != 1` needs `support_loss`. Default 1: the quasi-static flow. Tipping takes no
factor. C ABI: `RbDestructibleDef::support_loss_dynamic_factor`.

The stress oracle (stress-ref, explicit mode) amplifies the change of load, which is what the
convention reproduces:

| Oracle scene | before | static after | peak | (peak - before) / (after - before) | peak / after |
|---|---|---|---|---|---|
| b4 sudden support loss (zeta 0.02), reaction | 29.4 N | 39.2 N | 48.5 N | 1.94 | 1.24 |
| b8 sudden column removal (elastic), column 0 axial | 20.2 kN | 33.2 kN | 54.5 kN | 2.63 | 1.64 |
| b8 sudden column removal (elastic), column 2 axial | 31.3 kN | 48.2 kN | 74.2 kN | 2.54 | 1.54 |

(b8's static-after is the mean of its last 50 ms.) b4, a single degree of freedom, gives an
increment ratio of 1.944 against the analytic 1.939 at `zeta = 0.02`: `k = 2` on the change
predicts its peak (49.0 N against 48.5 N), where `k = 2` on the whole demand asks 78.4 N. b8, an
elastic frame, has increment ratios above 2, beyond the factor's range under either convention:
`k = 2` on the change gives 46.2 kN and 65.1 kN against its peaks of 54.5 kN and 74.2 kN (low),
on the whole demand 66.4 kN and 96.4 kN (high). The factor is not tuned to these.

## Seeing the reactions (videos)

- Fracture Lab, frame-locked: `node engine/crates/engine-webgpu/fracture/record-reactions.mjs
  crush-wall:1.4x:crush0 crush-wall:1.4x:crush1 --orbit 400,-60` (`:crushN` or `--crush N`
  sets the lab's Crush; the URL is `fracture/?pack=crush-wall&shot=1.4x&crush=1`).
- `World` only (blast loading, support loss, `Conserve` fragments): `cargo run --release -p
  engine-world --example reaction_replay -- <scenario>` dumps poses, shapes and events to
  JSON; `engine/tools/reaction-replay/` plays it in three.js and records it.
  `bash engine/tools/reaction-replay/record-all.sh OUT_DIR` runs every scenario. Support-loss
  breaks are told apart with `World::destructible_support_loss_breaks`.
