# Shardwright → GLB → Slang AVBD / WebGPU

A local, dependency-free browser demo importing **one self-contained GLB** with
`SHARDWRIGHT_fracture` v1. Shardwright does the slow fracture/hull cooking offline.
The lab reads baked meshes, convex hulls, full mass/inertia tensors, bonds,
hierarchy, and the resolved physical material library. Reading, scene building
and the per-tick host plan run in Rust compiled to wasm (no fracture bake runs
in the browser); the same Rust runs every scene natively on Metal
([`fracture_viewer`](../../../README.md#native-fracture-viewer-macos--metal)).
GPU shader compilation is a separate startup cost.

## Run

The `assets/*.glb` files are stored in Git LFS. If they are ~130-byte pointer files,
run `git lfs install && git lfs pull` first.

From the Rubble repository root, with the Slang toolchain described in
[`engine/TOOLCHAIN.md`](../../../TOOLCHAIN.md):

```sh
export RUSTUP_TOOLCHAIN=stable
export WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
cargo run --manifest-path engine/Cargo.toml -p engine-webgpu --bin wgsl_export
wasm-pack build engine/crates/engine-fracture-wasm --release --target web \
  --out-dir ../engine-webgpu/generated/fracture-wasm --no-typescript
python3 -m http.server 8777 --bind 127.0.0.1 --directory engine/crates/engine-webgpu
```

Open <http://127.0.0.1:8777/fracture/> in a WebGPU-enabled browser. Generated WGSL
and wasm are ignored by git, so rerun the export after checkout or shader changes,
and the `wasm-pack` build after changes to `engine-fracture`. `wasm-pack` comes
from `web/`'s devDependencies (`npm --prefix web ci`) or `cargo install wasm-pack`.

## Shared architecture

Each piece of logic has one source, used by both hosts:

| Piece | Source | Browser (WebGPU) | Native (Metal) |
|---|---|---|---|
| GLB / FRAC reader, metadata | `engine-fracture` `asset.rs` | wasm `Asset` | direct |
| Scene building: levels, compound hulls, bonds, rows/grids/suburbs, roads, cuboid house, Scene menu | `engine-fracture` `scene.rs`, `cuboid.rs`, `bonds.rs`, `samples.rs` | wasm `Scene` | direct |
| Hull cooking and the hull arena (ABI v2) | `engine-schema` `convex.rs` | wasm `HullArena` | direct |
| Per-tick host plan: buffer sizes, host words, step state, ordered dispatches | `engine-fracture` `plan.rs` | `simulation.mjs` replays it | `engine-viewer` `sim.rs` replays it |
| Physics kernels | `slang/kernels/*.slang` (`ENGINE_WGSL` layout) | WGSL (`wgsl_export`) | MSL (`engine-viewer` `build.rs`, from the plan's entry list) |
| Camera (orbit, first-person walk and fly), shot rays, floor, indexed mesh | `engine-fracture` `view.rs` | wasm `Orbit`, `camera-input.mjs` | direct (orbit) |
| Draw shader | `slang/render/fracture_draw.slang` | WGSL (`wgsl_export`) | MSL |

`engine-fracture-wasm` is the wasm-bindgen surface; `wasm.mjs` loads it and adapts
types. `Planner` emits each tick as flat command words: a copy (from the tick's host
upload or between buffers), a direct dispatch (kernel, workgroups, pass-uniform
slot), an indirect dispatch (kernel, argument slot, pass slot) or a phase marker
(profiling and frame-slicing boundaries). The hosts only create buffers and
pipelines, replay commands, read counters back into the plan, and draw. The JS
left is GPU I/O, the page and input. On a recording mock device the
replayed commands are byte-identical to the former JS host's, uploads included.

**Deterministic parity shot.** Both hosts can run the same scene, deterministic
budget, shot and tick count:

```sh
cargo run -p engine-viewer --release --example fracture_compare -- arch 0 240
```
```js
(await import('./parity.mjs')).parityShot(__FRACTURE_DEMO__.device,{scene:'arch',level:0,ticks:240})
```

Measured on an M3 Max after 240 ticks:

| Scene | Metal standing | Browser standing | Metal contacts @240 | Browser contacts @240 |
|---|---|---|---|---|
| arch, level 0 | 5 of 23 | 5 of 23 | 85 | 95 |
| house, level 0 | 70 of 131 | 70 of 131 | 969 | 780 |
| cuboid 7×7 grid | 31,907 of 35,035 | 31,927 of 35,035 | 46,760 | 47,286 |

The arch's counters are identical through tick 120. Each host repeats its own
run exactly; between hosts, f32 differences in the generated shader code (Slang →
MSL directly vs Slang → WGSL → Dawn) grow through contact-rich, chaotic motion.
The default **House** scene contains 131 building parts, 163 baked collision
hulls and 454 bonds. The Scene selector also retains the 23-piece masonry arch.
Both start at rest on the floor. Click to shoot an orange sphere; drag to orbit
and scroll to zoom. Impacts wake sleeping pieces. Reset reloads 16 shots.
Sphere trajectories compensate for gravity to aim near the clicked scene depth.

**Camera** (the panel's Camera menu; `V` cycles it, remembered per viewer). Every bond
mode shares the same camera (`engine-fracture` `view.rs` `ViewMode`, input in
`camera-input.mjs`):

| Mode | Look | Move | Shoot |
|---|---|---|---|
| Orbit | drag to orbit the scene centre | scroll to zoom | click: through the pointer |
| First person · walk | click to lock the mouse, then mouse-look (drag without a mouse) | WASD / arrows on the floor at 1.7 m eye height; Shift sprints; scroll sets the speed | click: through the crosshair (tap: through the tap) |
| First person · fly | as walk | as walk, along the look; Space / E up, C / Q down (stays above the floor) | as walk |

Entering first person starts at the orbit's eye (walk: dropped to the floor) looking at
the scene centre; Esc releases the mouse; back in orbit, the orbit view is as it was.
The camera only moves the eye and the shots' origin: it never collides with or moves
the pieces.

Live counters update approximately once per second: **TPS** counts completed
physics ticks (target 120), **FPS** counts submitted render frames, and **Sim
speed** is TPS / 120 (1× means real time). Pausing physics leaves rendering active,
so TPS drops to zero while FPS continues. Measurements exclude loading and hidden
tabs. Physics ms/tick includes CPU encoding/submission and the GPU/readback wait;
these are wall-clock measurements, not GPU timestamps. Frame ms is the interval
between rendered frames, including physics blocking, not isolated rendering cost.

The house includes four levels: **131 / 567 / 3,158 / 5,905 fragments**. The arch
has **23 / 165 / 171 / 392**. Only one level is active at a time. The house defaults
to level 0 because the current browser host and convex path are still experimental;
finer levels increase collision work substantially. Assets load only on selection.
Levels 2 and 3 bake 35 and 36 flat (zero-volume) sheets into compound fragments;
the importer skips those hulls (every fragment keeps real ones) and says so.

**Veneer houses · row of 20 (level 3)** places twenty level-3 houses along +z (131,840
bodies, 133,280 hulls) from `assets/veneer-house-hull16.glb`. Copies share one
hull arena and draw instanced. “Fire the wrecking ball” rolls a 1.6 m, 200 t ball
at 40 m/s along the ground through the whole row (`__FRACTURE_DEMO__.fireRow()`;
`benchmarkRow()` records per-phase GPU timings). The scene budgets 8 candidates
and 8 contacts per collider and 64 colours and colouring rounds, and runs with a
16-vertex hull limit: a larger hull is a load error, never reduced.

**Veneer houses · 7×7 grid of 49 (level 2)** uses the bundled `veneer-house.glb` (64-vertex
hull limit; 154,742 bodies, 245,882 hulls) and fires the ball down the middle
column. On an M3 Max an all-asleep step costs 2.6 ms of GPU time and the peak after
the shot ~38 ms (~114k awake, ~880k contacts). Unbonded pieces knock over
neighbouring houses, so most of the grid ends up awake. The 16-vertex re-bake
measured the same peak with 15% more pieces, so the grid keeps the original asset.

**Veneer houses · suburb of 48 (level 2)** (`build_suburb` in `engine-fracture`) spaces the same houses
out: 4 × 3 blocks of 2 × 2 lots with 5-6 m yards, separated by 8 m roads with
sidewalks and dashed centre lines (drawn on the ground; the physics ground is
still the plane). The ball rolls down one column of lots and hits 6 houses. On an
M3 Max 24 of 48 houses collapse (debris still topples unbonded lot neighbours;
43 of 49 on the grid), ~55k pieces stay awake, and the step settles at ~15 ms
(~66 ticks/s) by 30 s. 10 m yards still lose 22 houses.

**Cuboid houses · suburb of 48** uses the same layout (**Cuboid houses · large
suburb of 96**: twice the size, 6 × 4 blocks of 2 × 2 lots, about 69k pieces) with a
procedural house made only of box colliders (`engine-fracture` `cuboid.rs`, ~715 pieces): running-bond
brick walls with interlocking corners, door and windows with lintels, sills and
glass, stepped brick gables, a ridge beam, rafters and roof boards, an exterior
chimney, a porch on posts with a step, and a picket fence with a mailbox. Every
piece is a loose body held by gravity and friction (μ = 0.5); `engine-fracture` `tests/cuboid.rs`
checks that no two pieces start interpenetrating.
- The roof pitch is 12°: woken whole, a house stands and sleeps again in 2.6 s.
  At 18° the settling jolt starts the rafters sliding and the roof falls in.
- The scene uses zero-gap contacts (`FLAG_CONTACT_GAP`). With the default margin
  each course rests 2 mm deep, so the walls and the taller gables sink unevenly.
- On an M3 Max the shot collapses 12 of 48 houses, 6-12k pieces stay awake, and
  the GPU step is 3.5-4.2 ms (box-box contacts, no hull SAT), about 4x faster than
  the hull suburb. The remaining cost is mostly the primal's per-colour dispatches.

**Cuboid houses · 7×7 grid of 49** puts the same cuboid house on the dense
grid layout. Measured live in the browser on an M3 Max after the row shot
(wall-clock ticks including counter readback): 145-166 ticks/s for physics alone
and 79-86 ticks/s with a frame rendered every tick; 19 of 49 houses collapse by
20 s and up to ~20k pieces are awake. The hull grid of the same layout runs at
~33 ticks/s.

**Overlap physics ticks** (checkbox, opt-in) stops waiting for each tick's
counter readback: up to three ticks are queued ahead of the GPU and counters
are read in the background. The GPU runs the same ticks in the same order;
only host-side uses of the counters lag a tick or two (HUD, overflow report,
colour-slot estimate, few-awake solver choice, the all-asleep skip, which now
applies only when the counters describe the latest tick). On the cuboid grid
after the shot, 1 tick per call goes from 100 ticks/s waiting to 274
overlapped; the hull grid, GPU-bound at ~28 ms per tick, goes from 28.9 to
30.2. In lockstep, overlapped and waiting runs stay bit-identical until the
engine's own run-to-run nondeterminism appears (two plain runs of the cuboid
grid diverge at tick ~296).

### Scene menu

The **Scene** menu groups the bundled scenes (`engine-fracture` `samples.rs`, `GROUPS`),
each named for what it is, then how many:
- **One structure:** veneer house, cuboid house, masonry arch.
- **Rows and grids:** veneer houses in a row of 20 and a 7×7 grid; cuboid houses in a 7×7
  grid.
- **Suburbs (lots and roads):** veneer houses (48), cuboid houses (48, and a large
  suburb of 96).
- **Wrecking Yard:** tall structures for heavy balls, built in Rust as scene packs
  (`PACK_SAMPLES`; below).
- **Bridge Out:** a highway viaduct for a pier hit, also a Rust-built pack (below).
- **Topple:** a high-rise beside a street with demolition charges (below).
- **Ram Raid** and **Fly-through:** the vehicle's storefront and the aircraft's two
  towers, built the same way (below).
- **Vibe City presets:** generated scene packs, when they are built.
- **Your file:** a dropped or chosen GLB.

The native viewer numbers the same scenes 1–9 in this order (the Wrecking Yard and Topple are
the web lab's).

### Wrecking Yard

Three structures for the heavy-ball beats (`stress_ref::wrecking`). Each is built in Rust and
written as a scene pack, then loaded like any pack: `pack::build_pack_scene` for the
lab's own modes, stress-web `stress_set_pack` for the stress modes. The page asks for the
JSON with `packSampleJson(key)`; nothing is fetched.

| Scene | What | Chunks · bonds |
|---|---|---|
| 10-floor concrete frame | 3 × 3 bays of 6 m, 3.5 m storeys. Columns, beam-column joints, beams (two chunks a span), slabs (four a bay) and a walled lift core in the central bay, on pad and strip footings | 1,264 · 2,408 |
| Round tower on columns | Eight 8 m round floors on a 16-sided core and a ring of eight round columns: the red-sphere target | 337 · 648 |
| Brick tower | A 4 m square brick shaft, 12 m tall, in running bond with interlocked corners | 1,444 · 4,290 |

**Materials** are the calibrated ones (`material.rs`): concrete for the frame and the
round tower, brick with mortar joints for the tower. Each pack carries the strengths and
stiffness, and each node its mass at the calibrated density. The stress modes' authored
import therefore gives the calibrated values, and the welds in the lab's D6 mode do too
(the pack's Blast `limits`). The footings are the supports. The weight is carried as in
every pack: the stress solve takes gravity from the first frame.

**Ball presets** (stress modes; the **Ball** menu, which reloads the level):

| Preset | Ball |
|---|---|
| 10 t wrecking ball | solid steel, r 0.67 m, 40 m/s |
| 50 t wrecking ball | solid steel, r 1.15 m, 20 m/s |
| 15 m demolition sphere | a 5 cm steel shell (277 t) that rolls in along the ground at 12 m/s |
| Vehicle stand-in | 2 t, r 0.9 m, 15 m/s (54 km/h): the Ram Raid's click ball |
| Aircraft stand-in | 40 t, r 2 m, 80 m/s: the Fly-through's click ball |

The table is `stress_ref::wrecking::BALLS`; stress-web's `stress_set_shot_preset` applies
it. The mass, speed and radius sliders still scale it. **Fire the ball along the line of
attack** launches it at the structure from 6 m out (`stress_fire`, `wrecking::launch`),
as the row's wrecking-ball button does:
- the frame's corner column line along the front facade, at the ground storey;
- the round tower's -x column, with the core behind it;
- the middle of the brick tower's -x wall, a quarter of the way up.

Clicking still shoots the preset ball along the camera ray.

**Checks.**
- `cargo test --release -p stress-ref --test wrecking` (in `engine/stress-oracle`):
  - chunk and bond counts and total mass from the layouts' geometry;
  - the packs import back unchanged;
  - each structure stands in the reference under its own weight.
- `cargo test --release -p stress-rubble --features gpu --test wrecking_yard`: each
  structure settles on the GPU in both stress modes, with no bond broken and under 1 cm of
  movement in 2 s.
- `cargo test --release -p engine-fracture --test pack_samples`: the menu entries build
  as lab scenes.

**Hero check** (native, M3 Max). Run:
`cargo run --release -p stress-rubble --features gpu --example wrecking_yard -- wy_frame10 ball50t 600`.
It prints bonds broken, fragments, ms per frame, and per storey when a tenth of its
chunks had dropped a metre.

The 50 t ball into the frame's corner column line under `blast_parity` gives a
collapse that climbs:
- storeys 1–2 drop by frame 90 (1.3 s after the shot at frame 10), and storeys 3–6 by
  frame 120;
- storeys 7–10, the core and the far bays stay standing;
- 684 of 2,408 bonds break, leaving 340 fragments;
- median 3.4 ms per frame after the shot, p95 7.8 ms, worst 34 ms;
- the run repeats exactly.

The 10 t ball only breaks the corner (77 bonds). The demolition sphere flattens the
frame (over 2,300 bonds by frame 240). In the round tower, the 50 t ball and the sphere drop every floor's
slabs and leave the core standing. The brick tower is holed through courses 6–24 and
stays up above the hole.

`threshold` breaks bonds on contact force alone, with no stress solve, so there it is
local damage only. In Chrome, after 90 frames the 50 t ball had broken 4 of the frame's
bonds, 18 of the round tower's and 15 of the brick tower's.

### Bridge Out

A reinforced-concrete highway viaduct for the pier-hit beat (`stress_ref::viaduct`). It is
built and loaded like the Wrecking Yard's structures, and it has its own Scene menu group.
It has **347 chunks and 575 bonds** (about 1,890 t):
- four 16 m spans of a 9 m wide, 0.9 m deep deck, its top 10 m above grade, in chunks of
  about 2.5 × 1.5 m, with parapets;
- three hammerhead piers (a 1.8 m column in five chunks on a pad footing, a pier cap in
  three chunks, bearing seats) and two abutments;
- a lower road crossing at grade under span 2: slabs bonded to a sub-base of supports.

The deck is bonded across every span joint, but the piers differ:
- **Piers 1 and 3 are seat piers.** The spans' ends rest 1 m over a 3 m cap on bearings,
  joined by a link slab the top third of the deck deep.
- **Pier 2, the one the ball hits, is a Gerber (half-joint) hinge.** Spans 2 and 3 are
  drop-in spans. They hang by their bonds alone on the 0.15 m nibs of a 2 m link slab,
  over a diaphragm, with no seat under their ends.

Why the hinge pier: in `blast_parity` the struck structure is anchored, so the ball
rebounds and the hit only breaks bonds. The pier's broken column chunks stay stacked
under its cap. A span seated on that pier would stay up on contact at both ends and
break at midspan. A drop-in span has nothing under its end once the nibs go. The nibs are
thinner than the seat piers' links, so they break first, and each span swings down about
its far seat as one slab.

The ball (the 50 t preset by default) comes in across the deck (+y) at pier 2's column,
one column chunk above its footing.

**Checks:**
- `cargo test --release -p stress-ref --test wrecking` covers the counts, the mass, grade
  and the seats (`viaduct_counts_and_mass`), along with the shared pack and
  reference-settle tests.
- `cargo test --release -p stress-rubble --features gpu --test wrecking_yard viaduct`
  covers two things:
  - the settle in both stress modes;
  - the 50 t pier hit in `blast_parity`, with each frame's splits followed as the lab's
    view does. At frame 90 spans 2 and 3 are swinging down, still 90% intact, and spans 1
    and 4 stand. At frame 150 spans 2 and 3 are on the ground and broken up.

**Hero check:**
`cargo run --release -p stress-rubble --features gpu --example wrecking_yard -- bo_viaduct ball50t 600`.
For each span that moves, it also prints how far its two ends have dropped and how many
of its bonded pairs are still at their distance.

The 50 t ball under `blast_parity`, fired at frame 10:
- **Frame 30:** the ball arrives at about frame 28, and 21 bonds break: pier 2's column,
  its bearings, its diaphragm and half the nibs. The far ends' links and bearings at
  piers 1 and 3 let go within a few frames.
- **Frame 60:** spans 2 and 3 hinge down. Their pier-2 ends are 1.1 m down and their
  seated ends 0.15 m, with 80 and 81 of 82 pairs intact.
- **Frame 90:** the pier-2 ends are 5.2 m down and the seated ends 0.6 m, with 78 and 80 of
  82 pairs intact.
- **About frame 110:** the pier-2 ends land, span 2's on the lower road. The slabs break up
  there (88 deck bonds in frames 101–110, and 20 of the road's bonds by frame 140), and
  their seated ends slide off the seats.
- **Frames 120–180:** pier 3 and then pier 1 give way, and spans 4 and 1 follow: a
  progressive collapse. By frame 300, 446 bonds have broken (232 fragments); the abutments
  and the road stay.

The other balls, under `blast_parity`:
- **10 t ball:** the same sequence. The spans stay whole in the air (78 and 80 of 82
  pairs at frame 90) and break up when they land (frames 100–110). Spans 1 and 4 follow.
- **Demolition sphere:** it takes pier 2 and both drop-in spans down by frame 60.
  Spans 1 and 4 and piers 1 and 3 stay standing.

Under `threshold` nothing comes down: the strike breaks the struck column chunk's 3 bonds
(51 with the sphere). There is no stress solve, and the rest of the viaduct is one
anchored, static body, so no span drops.

### Topple

A narrow high-rise beside a street, brought down by demolition charges (The Finals'
street-building topple, Teardown's high-rise topple). `stress_ref::topple`, built and loaded
like the Wrecking Yard (Scene menu group **Topple**).

| Scene | What | Chunks · bonds |
|---|---|---|
| High-rise beside a street, with charges | The Wrecking Yard frame (`wrecking::rc_frame`) at 2 × 1 bays and 10 storeys: 12.5 × 6.5 m in plan, 35.5 m tall, about 1,000 t. Shear walls across both short ends from the second storey up, a ground-storey back wall on strip footings, pad footings. A 20 m street of 4 m slab tiles (supports) along its long street side (-y) | 420 · 714 |

**Charges.** The pack carries them as `defaults.charges`, in firing order:

```json
{"name": "street-side columns", "delay": 0, "bonds": [[node0, node1], ...], "debris": [node, ...], "kick": {"x": 0, "y": 1, "z": 12}}
```

- `bonds` are node pairs, not bond indices. A pair names its bond whatever order an import
  keeps the bonds in, and whichever bonds it leaves out (`PackCharge::bond_indices`
  resolves them and names any pair that is not a bond). Node indices are stable: every
  import keeps the pack's node order.
- `debris` are the chunks the blast throws out. `kick` is the velocity it adds to them
  (Y-up, m/s) once they come free.
- `delay` is in seconds after the charges are set off.

The tower's two charges:
1. **street-side columns** (at once): every bond of the three street-side ground-storey
   columns, to their footings and to the joints above. The columns are thrown out over the
   street at 12 m/s, as a demolition's kicker charges do. A rigid column cut loose would
   otherwise stand where it was and carry the storeys above as before.
2. **back-row footings** (0.25 s later, once the columns are clear): the back line's bonds
   to its footings (3 columns, the back wall's 4 halves to their strips and pads). The tower
   then stands on its back line alone, its weight over the empty street side.

**Setting them off.** Use **Detonate charges** (stress modes, shown only for a scene with
charges), or call stress-web `stress_detonate(name)` with an empty name for every charge,
each at its delay. `stress_charges()` lists the charges, and `stats.charges` holds their
names. The cut happens between two frames, at the start of the next one, and never by
stepping a frame again:
- `threshold` (GPU-resident world): `ThresholdWorld::cut_bonds` / `detonate`. The bonds break
  as if the last frame had broken them (`GpuFracture::break_bonds`, after one wait for the
  GPU's bond and body state). The next frame splits them after its step. The kick is added
  before the frame after that, to bodies made of debris alone.
- `blast_parity` (resident GpuWorld): `GpuWorld::cut_bonds`. The host splits the bonds as the
  punch path does (`flush_pending`, the engine's motion, `break_bonds`, `rebuild`), sets the
  debris clusters' kick, and writes the bodies back to the engine.
- `stress_rubble::demolition::Demolition` schedules the charges (by name, or all at their
  delays) for natively run hosts.

**What it does.** Native, M3 Max:
`cargo run --release -p stress-rubble --features gpu --example topple -- 480 30`
sets the charges off before frame 30. Add `STRESS_PROFILE=blast_parity` for the other
profile and `TOPPLE_TRACE=1` for the bonds broken by storey and kind. Both profiles stand
under self-weight, and both repeat exactly run to run.

- **threshold**: the tower topples across the street.
  - Frame 31: 6 bonds are cut. The columns are 2.7 m out by frame 45 and about 20 m out in
    the end.
  - Frame 46: 11 bonds are cut. The tower then leans out over the street as one body.
    | Frame | Lean | Largest body |
    |---|---|---|
    | 105 | 3.9° | 92% of the storeys' chunks |
    | 180 | 16° | 84% |
    | 285 | 51° | 72% |
  - Frame 330 (5.5 s): it lands with its top storey 21 m out over the street. The landing
    breaks bonds from 300 to 402 by frame 375.
  - The end: 405 of 714 bonds broken and 233 bodies.
  - It does not break into storey-sized slabs. The landing strips the street face, and
    storeys 3–10 stay one 188-chunk piece. This profile breaks a bond only when a *contact*
    load on one of its chunks beats it, so the bonds between storeys on the far side and in
    the end walls never see a load.
  - The base also erodes while the tower leans: the free tower's weight on its back line is
    a sustained contact load above those chunks' bond capacity.
- **blast_parity**: the charges cut as above, but the tower does not topple.
  - The cut of the street-side columns breaks 16 more bonds around them.
  - Once the back row is cut, the quasi-static solve of the free tower breaks about 140 bonds in
    storeys 1–5 within 15 frames, and the whole tower by frame 135. It collapses almost in
    place: the top storey moves 6 m toward the street, and 680 of 714 bonds break.
  - Walls along the long faces as well did not change this.
- **In Chrome (threshold)** the lab's run diverged from the native one. The back line
  crushed faster than the tower tipped, and the tower came down nearly vertically (704
  bonds broken by 9 s). The outcome sits close to that crushing threshold, and Chrome's
  WebGPU runs other shader code (WGSL rather than Metal), so treat the native run as the
  reference.

**Checks** (in `engine/stress-oracle`):
- `cargo test --release -p stress-ref --test topple`:
  - counts and mass from the layout;
  - what each charge cuts; both together leave the tower one piece, bonded to no support,
    its centre of mass over the street side of its back line;
  - the charges survive both profiles' imports.
- `cargo test --release -p stress-rubble --features gpu --test topple`:
  - threshold: a second after the release the tower leans out as one body. In the end the
    top storey is more than 15 m across the street, on the ground, and over 100 bonds past
    the charges' are broken.
  - blast_parity: the charges' bonds are broken before the next frame, the columns are
    thrown and the tower is free of its footings.
- `wrecking_yard` (settle) and `pack_samples` include the tower.

**Scripted takes and replay.** A take step `act: {detonate: name}` (`''` for every charge
at its delay) sets the charges off, and the Detonate button's charges are logged. Either
way, instant replay sets them off at the same frame (see "Takes, instant replay and slow
motion"). The `topple_detonate` take (`takes/lab.mjs`) records the threshold run.

### Ram Raid and Fly-through

The targets of the vehicle and aircraft beats (`stress_ref::buildings`), built and loaded
as the Wrecking Yard's are, in their own Scene menu groups. Until the combined host drives
a real vehicle or aircraft, a ball preset stands in for each (the table above).

| Scene | What | Chunks · bonds | Mass |
|---|---|---|---|
| Storefront row (Ram Raid) | Five two-storey shops, 5.5 × 7 m each, in one 27.8 m terrace. Brick party walls, gables, back wall and upper facade in running bond (1 × 0.5 m units, 0.3 m walls), two upper windows a shop under concrete lintels; on the ground storey a glazed door and a two-pane shop window between brick piers, on a brick stallriser, under a concrete fascia; 0.2 m concrete first-floor slab and roof on the walls; concrete strip footings | 1,203 · 3,464 | 498 t |
| 7-storey apartment block (Fly-through) | A 24.5 × 12.5 m RC frame, 4 × 2 bays of 6 m, 3 m storeys (0.45 m columns and beam-column joints, 0.45 × 0.5 m beams, 0.2 m slabs, as the 10-floor frame is cut). Every perimeter bay is filled with 0.25 m brick panels round a 2 × 1.3 m window under a concrete lintel; a glazed entrance on the ground floor | 1,355 · 2,874 | 1,920 t |
| 11-storey glass tower (Fly-through) | 12 m square, 40 m tall. 0.25 m slabs (32 cells a floor) on a 5 m walled core (0.3 m walls) and eight 0.45 m perimeter columns, in a curtain wall: 25 mm panes in six 2 m bays a side between steel mullions (80 × 160 mm), steel transoms at the slab edges, corner posts | 1,332 · 2,856 | 1,842 t |

Every chunk is a box, so the bonds are `auto_bonds` between touching faces, with the joint
material of whichever governs (`wrecking::joint_of`): mortar wherever brick meets
anything, glass where a pane meets its frame, concrete where concrete meets anything
(steel is anchored in it), steel between steel members. Glazing carries none of the
building's load, so it lacks some joints (`buildings::glazing_joints`):
- a pane in a masonry or concrete opening is bonded at its sill only (setting blocks);
  it is not bonded at its head (clearance) or its sides (a sealed frame with a movement
  gap). Bonded at its sides, it took the piers' settlement in shear and broke under its
  own weight in the reference;
- a curtain-wall pane is bonded only at its sides, to its mullions and posts (structural
  silicone);
- each storey's curtain wall hangs from its own slab edge over a 20 mm open stack joint
  (`TowerLayout::STACK_JOINT`). Without the gap, the rings of curtain wall stacked on each
  other as rigid tubes: once the floors below fell away, they held the tower's upper
  block up on glass.

**Materials.** `wrecking::calibrated` adds glass and structural steel to concrete,
footing and brick:
- glass is annealed soda-lime float (`Material::glass`, EN 572-1): 2,500 kg/m³,
  E = 70 GPa, 40 MPa tension, 1,000 MPa compression, 40 MPa shear, brittle, 8 J/m²;
- steel is S355 (`Material::steel`, EN 10025-2 / EN 1993-1-1): 7,850 kg/m³,
  E = 200 GPa, 355 MPa tension and compression, 205 MPa shear (f_y/√3), ductile.

The sources are cited in `calibrated`'s comment. Glass bonds are of the glass,
the rest as in the yard. There is no rebar in glass or steel joints.

**Blast limits per material.** The D6 mode's Blast `limits` were pack-wide, and these
packs mix two or three joint materials. So each pack material carries its own `limits`
(its calibrated strengths over the 60 Hz step), and `pack.rs` gives each bond its
material's. A pack of one joint material, which covers every Wrecking Yard structure,
still has the pack-wide `limits` too, so its packs load unchanged. A bond whose material
has no `limits` in a pack without pack-wide ones is a load error naming the bond.
Incomplete `limits` are a load error naming the material (`tests/pack_samples.rs`).

The D6 welds' springs still use one modulus (`PACK_YOUNGS`) for every material. Their
capacities follow the material; their stiffness ratios follow geometry only.

**Lines of attack** (`defaults.wreckingYard`, along +y into the street front):
- the storefront: the middle shop's window, 1.25 m up (a car's bonnet);
- the apartment block: the front's middle column at mid-height (the fourth storey);
- the glass tower: the front's middle mullion at mid-height (the sixth storey).

**Checks.** The Wrecking Yard's checks cover these too:
- `tests/wrecking.rs` pins the chunk and bond counts and cross-checks the bonds against
  a brute-force count of touching faces under the glazing rules, by joint material. It
  checks the total mass from the layout's volumes, that the boxes are solid (no overlap)
  and stand at or above grade, and that every chunk is bonded through to a footing. It
  checks the `storey-N` groups and the aim, that the packs import with each bond's own
  calibrated material, and that each building stands in the reference;
- `tests/wrecking_yard.rs` (GPU) shows each one settle in `blast_parity` and
  `threshold`: no bond broken, under 2 mm of movement in 2 s.

**Hits** (native, M3 Max, `blast_parity` unless stated). Run:
`cargo run --release -p stress-rubble --features gpu --example wrecking_yard -- KEY BALL 600`.
The example also prints each storey's mean displacement and tilt at the end, and the top
storey's drift on every line, so a tilting upper block shows. `STRESS_WY_KIND=core`
limits the table to one node type, and `STRESS_WY_AIM_X` / `STRESS_WY_AIM_Z` move the
target. Results:
- **Storefront, vehicle stand-in.** The ball smashes the middle shop window: 20 bonds
  break and the two panes and the stallrisers come out (7 fragments). The piers, fascia
  and upper storey stand. Under `threshold`: 111 bonds, 29 bodies, still local.
- **Apartment block.**
  - The vehicle stand-in at the fourth storey breaks 2 bonds.
  - A 10 t ball at 20 m/s breaks 95 bonds round the struck column (8 chunks of storeys
    3–4 fall), and the block stands.
  - The 10 t preset (40 m/s) breaks 261 and the block holds for 2 s. Then a progressive
    collapse brings all of it down by frame 480.
  - The aircraft stand-in breaks 2,261 bonds within 20 frames of contact, in every
    storey, and the block falls at once.
- **Glass tower.**
  - The 10 t ball goes through the sixth storey's curtain wall into the core: 159 bonds
    at contact, 278 by frame 180. Then a progressive collapse starts, and from frame 240
    everything comes down (2,540 bonds by frame 600).
  - The aircraft stand-in breaks 1,965 bonds at contact, and the tower is flat by frame
    240.
  - **No preset makes a mid-height cut tilt the upper block.** The storeys under the
    impact fail first, and the upper block drops straight down: its top storey drifts
    under 0.3 m sideways while falling 16–23 m. It tips over only once it lands on the
    debris.
  - Under `threshold` (no stress solve), the aircraft stand-in breaks 347 bonds at
    contact. Falling debris brings that to 922 by frame 480, and a quarter of the chunks
    of storeys 1–7 fall. The top four storeys stay where they were, on what remains.

A strike in `blast_parity` loads the whole island it hits. A fast, heavy one, such as
the aircraft stand-in, therefore breaks bonds in every storey within the contact frame,
not just along the impactor's path. The same 10 t ball at mid-height partly collapses
the 10-floor frame too (577 bonds; about half the chunks of storeys 1–5 fall). Broken pieces are rigid in
contact, so stacked debris cannot crush. A tilting upper block would need the far side's
columns and walls to hinge in compression while the near side goes, and these profiles
do not produce that.

### Launched bonded bodies

A pack can carry **free islands**: chunks bonded to each other but, through no path of
bonds, to a support. Both stress profiles already make each one its own rigid body (the
reference world's clusters are the connected components of the bonds, and a component
with no support is not anchored). A car, a bus or an aircraft is such an island: the
structure's own chunks and bonds, so it collides with the rest of the structure and with
itself, and its bonds break as any bond does. `stress_ref::launch` sets them moving.

The pack carries its launches as `defaults.launches` (`launch::write`, read with
`launch::launches_of`), Y-up, m/s:

```json
{"name": "car", "nodes": [node, ...], "velocity": {"x": 0, "y": 0, "z": 15},
 "angularVelocity": {"x": 0, "y": 0, "z": 0}, "delay": 0,
 "strikes": [{"centre": {"x": 0, "y": 0.6, "z": 2.0}, "radius": 0.9}]}
```

- `nodes` are the island's nodes (`group`, a `nodeGroups` name, may stand in for them).
- `velocity` is the island's centre-of-mass velocity and `angularVelocity` its spin. Each body
  the island is made of when it is released gets `v + w × (c − c_island)` at its centre `c`.
- `delay` is in seconds after the launch is asked for. Until it is released the island is a
  free body like any other: a car stands on its wheels, and an aircraft placed in the air
  falls, so launch an airborne island at once.
- `strikes` are spheres fixed in the island (its nose, an aircraft's wings), in the frame it
  was built in. The threshold profile's strike prediction checks them as it checks a shot,
  at the sphere's point of the body that holds the nearest island chunk, moving with that
  body (spin included). A strike they will make within the next frame breaks the struck
  body's bonds before the engine steps, so a fast island goes through a shop window as a
  ball does instead of stopping dead on it. A body's own chunks are never its strike's.

**Releasing them.** Use **Launch vehicles** (stress modes, shown only for a scene with
launches), or call stress-web `stress_launch(name)`, with an empty name for every launch,
each at its delay. `stress_launches()` lists them and `stats.launches` holds their names.
The island is set moving between two frames, at the start of the next one, and never by
stepping a frame again:
- `threshold`: `ThresholdWorld::launch_island` writes the bodies' motion in both pose
  regions (after one wait for the GPU's body state) and adds the strike spheres to the
  prediction's shots (`stress-gpu` `fracture::Shot::at`; 64 shots in all, the 16 balls'
  included).
- `blast_parity`: `GpuWorld::launch_island` sets the clusters' motion as a charge's kick is
  set, rebuilds the solver's tables and writes the bodies back to the engine. No strike is
  predicted for an island here: its contacts load the stress solve after the engine's step,
  as every contact does, so an island stops at what it hits as the profile's balls rebound.
- `stress_rubble::launch::Launcher` schedules them for natively run hosts.

A take step `act: {launch: name}` releases them, and the button's launches are logged, so
instant replay releases them at the same frame.

**Check scene.** `launch::ram_and_wall`: a concrete wall of 4 × 4 blocks on a footing and a
free concrete ram (a T of five 0.5 m cubes, 1.5 t) launched at it at 30 m/s.
- `cargo test --release -p stress-ref --test launch`: the launches' pack round trip (nodes or
  group, Y-up), malformed launches named, and the ram imports as its own unanchored body.
- `cargo test --release -p stress-rubble --features gpu --test launch`: in both profiles
  the ram flies at its launch velocity (3.000 m in 6 frames, its fall included) and breaks
  bonds of the wall and of itself (threshold: the wall in 5 bodies and the ram in 2;
  blast_parity: 14 of the wall's bonds and 4 of the ram's).

### Bonded vehicles

`stress_ref::vehicles` builds a car, a bus and an airliner as free bonded islands. Each part
is a chunk or a few: a convex prism or loft, so a vehicle can be placed at any heading.
Parts are bonded wherever their faces touch (`wrecking::face_bonds`), and the joint is chosen
by `vehicles::joint`:
- `glass` where a pane meets its frame;
- `riveted` in an airframe;
- `bolted` where a door, wheel, bonnet, boot lid, bumper, bolted wing or engine is attached;
- `spot-weld` elsewhere in a road vehicle's welded shell.

| Vehicle | What | Parts · bonds | Mass |
|---|---|---|---|
| `car` | A saloon, 4.5 × 1.8 × 1.45 m: floor, sills, subframes and rails; engine, firewall, bonnet, bolted wings; A-, B- and C-pillars, roof; four doors on their hinge and latch pillars; boot; bumpers; four wheels; windscreen, rear and side glass | 50 · 115 | 1.4 t |
| `bus` | A single-deck bus, 12 × 2.5 × 3.2 m: underframes (the engine in the rear one), axles, floor; side panels below and above the windows, window pillars, glass; roof; front with windscreen, rear; four wheels | 87 · 214 | 10.4 t |
| `airliner` | A regional jet (Embraer E175 class), 30 m long, 26 m span: an octagonal fuselage barrel in upper and lower halves, nose and tail cones, wings in four pieces a side, engines on pylons, fin and tailplane | 36 · 46 | 34.7 t |

**Materials** (`wrecking::calibrated`, cited in `material.rs`). A vehicle's parts are solid
chunks standing for thin metal shells, so the metal is smeared over each chunk: a smeared
property is the metal's own value times its solidity, the metal's share of the volume or of
the section.
- `sheet-steel`: the body shell, HC340LA to EN 10268, 1.5% steel by volume (a 1.4 t car in its
  4.5 × 1.8 × 1.45 m envelope). That gives E = 3.2 GPa, the parts' contact stiffness.
- `spot-weld`, the welded shell's joints, at 0.3% steel by section:
  - shear 0.375 MPa: about 6 kN a weld (AWS D8.9M and ISO 18278-2 tensile-shear tests) at a
    40 mm pitch;
  - tension 0.15 MPa: cross-tension is 0.4 of tensile-shear (ISO 14272);
  - compression 0.37 MPa: the shell's progressive folding, a third of its squash load
    (Wierzbicki & Abramowicz). Euro NCAP's 56 km/h stop over 0.6 m averages about 0.28 MPa
    over the front section.
- `bolted`: doors, wheels and bolted panels. Tension 0.14 MPa and shear 0.11 MPa are FMVSS 206's
  11,000 N and 8,900 N door hinge and latch loads over a door's 0.08 m² edge face. This is
  weaker than the welded shell in every mode: doors and wheels come off first.
- `aluminium` and `riveted`: the airframe, 2024-T3 (MMPDS-01: F_tu 483, F_su 283,
  F_cy 269 MPa, E 72.4 GPa).
  - The airframe is 4% aluminium by volume, and its joints are 0.4% by section at a riveted
    lap joint's efficiency of 0.7 (Niu).
  - The joints hold 1.35 MPa in tension, 0.79 MPa in shear and 0.38 MPa in compression (the
    thin shell crushing).
- `wheel`: the shell's material, under its own name for its colour.

Each part's mass is the vehicle's own (its curb mass shared out by part; the car's floor
takes the rest), written as the pack's node masses, which both profiles import
(`pack_masses`). A scene (`vehicles::compose`) places vehicles beside a structure. Each one
is bonded within itself only, launched where it has a `Motion`, and its strike spheres go
with it. Its chunks follow the structure's, so the structure's node indices are unchanged.

**Wheels do not roll.** They are rigid parts of the vehicle, so a car launched on its wheels
slides on them, braked by the ground's friction like a car on locked wheels.
`vehicles::sliding_launch` gives the launch speed that arrives at a given speed. Measured,
the car loses about 5 m/s² (0.5 g).

**Crash test** (Scene menu group): the car or the bus slides 12 m into a rigid barrier (a
support) and arrives at about 60 km/h (`vh_car_barrier`, 51 chunks; `vh_bus_barrier`, 88).
Native, M3 Max (`cargo run --release -p stress-rubble --features gpu --example vehicles --
vh_car_barrier 120`, `STRESS_PROFILE=blast_parity` for the other profile):
- **threshold**: 37 of the car's 115 bonds break. Its front wheels and a wing come off and the
  bonnet lifts, and 45 of its 50 parts stay one body: the cabin holds. In the bus, the
  wheels, the rear axle and the bumper tear off, and the body (93% of its parts) holds.
- **blast_parity**: the car's four wheels shear off at the barrier, and the rest stays one
  body (4 bonds). With a 6 m run-up (16.4 m/s), the whole front and floor came apart instead
  (52 bonds): how much breaks depends on how the contact meets the barrier. The pillars, roof,
  doors and glass held in both runs.
- In Chrome (takes `vh_car_barrier_crash` and `vh_bus_barrier_crash`) the threshold runs break
  45 and 39 bonds.

The bus rocks on its wheels in the reference world's own contact and loses them after half a
second. On the GPU, both vehicles stand still in both stress modes.

**Ram Raid with vehicles** (V5, Scene menu group Ram Raid). Both buildings stand on 0.5 m
footings above grade, so each scene adds a street: one support slab, level with the
footings' tops, which the vehicles drive on. Each vehicle starts 12 m out and hits at about
54 km/h, one launch each (`car`, `bus`).
- `rr_vehicles_storefront` (1,341 chunks): the car is aimed at the middle shop's window and
  the bus at the fourth shop's.
- `rr_vehicles_apartment` (1,493 chunks): the car is aimed at front bay 0's brick sill and
  window, the bus at bay 3's.

Threshold, recorded in Chrome (takes `rr_storefront_car_bus` and `rr_apartment_car_bus`: the
car, then the bus, the camera following each in):
- **Storefront.** The car's strike breaks the stallriser and the panes before it reaches them.
  It goes through the window into the shop (natively, 8.9 m in by frame 100) and crumples
  against the interior and the back wall. The bus breaks the window, the piers and the fascia
  and drives into the shop with the fascia on its roof. 442 bonds are broken, the same in
  three recordings.
- **Apartment block.** Its infill is coarse: a bay's brick sill is two 1.15 t panels. The car
  breaks the sill loose but does not get through, and the bus breaks the infill under the
  first-floor beam and is thrown back. 90 bonds are broken, the same in three recordings.
- At the glazed entrance the car stops dead on the pane. A pane is one chunk bonded at its
  sill, and that bond holds 1.6 MN (40 MPa over a 20 mm by 2 m edge), so the pane cannot
  shatter. It is the buildings' glazing model, not changed here.

Recording these takes exposed a bug in strike prediction that was older than them. Under the
WGSL builds' fast math (Chrome, and naga's Metal compile), the nearest struck chunk was
matched by the bits of a recomputed distance. That found no chunk, so every strike was
dropped: the car stopped on the shop window in Chrome. `fr_strike_entry` now keeps the
distance and `fr_strike_nearest` reads it. Metal's results are unchanged. In Chrome, the
threshold car stand-in now breaks 129 of the storefront's bonds by frame 100 (native WGSL
133, Metal 104) where it broke 11 before.

`cargo test --release -p stress-rubble --features gpu --test vehicles car_drives`: the car
alone goes more than 2 m past the facade, and both the shopfront and the car break.

**Fly-through with the airliner** (S3, Scene menu group Fly-through). The airliner flies
along the building's line of attack (`defaults.wreckingYard`, +y), its fuselage axis at the
aim's height:
- `ft_airliner_tower` (1,368 chunks): the glass tower's sixth storey, between two slabs;
- `ft_airliner_apartment` (1,391 chunks): the apartment block's fourth storey.

It is thrown like a paper aeroplane: its nose 40 m out, at 70 m/s, with the lift that brings it
to the facade's height under gravity (2.8 m/s up: its path is within 2.3° of level and 0.2 m
of the aim's height). It is released at once, since nothing holds it up before it is released.
Its strike spheres are its nose, both wing roots and both engines. A struck body takes one
strike a frame: the first sphere in order that reaches its bounds, usually the nose. So the
wings meet the building by contact loads.

No kinematic flight up to first contact. Holding a body on a path and releasing it at
contact would need the engine to drive the island's motion until the routing check sees the
contact coming. That is forward-only in principle, but neither world has a kinematic island,
so it is not done here.

What happens, threshold, recorded in Chrome (takes `ft_glass_tower_airliner` and
`ft_apartment_airliner`, which replace the 40 t ball stand-in's `ft_glass_tower_jet`):
- **Glass tower.** The nose breaks up in the curtain wall and the floors around the cut. The
  jet stops within the tower's first bays: natively it gets 8 m in on WGSL and 14 m on Metal,
  and it stops at the core and the slab pieces it pushes. In Chrome it pitches up and slides down the
  facade. Natively its nose, three wing pieces and an engine break off. 710 bonds break by 7 s. The tower
  stands: the upper block does not fail across the cut band. The threshold profile has no
  stress solve, and the README's earlier note on this tower stands.
- **Apartment block.** It punches through the middle bays of five storeys, so you can see
  through the block. Its engines, nose and wing pieces fly off. The bays above the hole hang
  on, and the jet falls back out in front. 1,148 bonds break by 7 s.
- **blast_parity** (native): the jet shatters on contact (all 46 of its bonds) and the tower
  comes down whole (2,566 bonds by frame 240). Its pieces are thrown up out of the tower, 86 m
  in 4 s, an artefact of the quasi-static solve of a free island stopped in one frame. That
  is why the takes are threshold.
- The airliner breaks less than a real one would. The threshold rule breaks a bond when a
  chunk's contact load beats its tension capacity. A riveted barrel section holds 9.5 MN in
  tension (1.35 MPa over 7 m²), and the crushing of the thin shell in compression is not a
  failure mode the profile has.

`cargo test --release -p stress-rubble --features gpu --test vehicles airliner`: the jet
flies in level at 70 m/s (within 0.5 m of its height), breaks over 300 of the apartment
block's bonds and loses its nose.

**Checks:**
- `cargo test --release -p stress-ref --test vehicles`:
  - each vehicle's parts, mass and size, as one free island with every joint as `joint` picks;
  - the bolted joints weaker than the welds, each door on two 11 kN joints;
  - the scenes' packs import as built in both profiles (masses, launches, free islands);
  - the car stands on its wheels in the reference.
- `cargo test --release -p stress-rubble --features gpu --test vehicles`:
  - the car and the bus stand on their wheels for a second in both stress modes (no bond
    broken, under 1 cm of movement);
  - threshold: the car arrives at 56-60 km/h, stops, loses front parts and keeps its cabin;
  - blast_parity: parts break off and the cabin's pillars, roof, doors and glass stay one body.
- `pack_samples` builds the scenes as lab scenes.

### Bond modes

The **Bonds** menu picks how the baked bonds hold the fragments, on the same asset,
level, camera and shot rule, so the three can be compared:

| Mode | What holds the fragments | Where it runs |
|---|---|---|
| **None** (default) | Nothing: pieces start asleep and wake when hit | rubble (this page's WGSL host) |
| **Breakable D6 joints** | Each bond a breakable AVBD weld (below) | rubble (this page's WGSL host) |
| **Stress solver** | The GPU stress solver's bonds (`engine/stress-oracle`, profile `blast_parity`): rubble steps the bodies, its contacts load the bond graph, and a quasi-static stress cascade breaks bonds and splits fragments on the GPU | Rust on wasm with its own WebGPU device and canvas (`stress-oracle/crates/stress-web`) |
| **Stress solver · simple threshold (v0)** | The same pipeline with no stress solve (profile `threshold`), forward only. A strike is predicted before rubble steps (its Hertz force against each bond's strength × area) and its island split first. After each step, a contact force beyond a bond's capacity breaks it. The breaks split fragments into new bodies on the GPU, exactly as the stress solver's do | As **Stress solver** |
| **Learned impact** | **Compute only** — lean MGN-T forward (H=48, P=64, ~139k random weights) on the house-1story pack graph. Does not break bonds or drive rubble; the HUD shows **ms/forward** so browser WebGPU latency can be compared to the native Metal M0 finding | This page's WebGPU device; Slang → WGSL (`slang/kernels/learned_impact.slang` → `generated/wgsl/learned_impact.wgsl`) |

**Browser support (stress modes).** The two stress-solver modes wait on the GPU from Rust.
Chrome and Edge 137+ suspend the wasm stack there with WebAssembly JavaScript Promise
Integration (JSPI). Other WebGPU browsers (Safari, including iPhone) load an Asyncify build of
the same Rust (`stress_web_bg.asyncify.wasm`), whose stack the page unwinds and rewinds
(`gpu-wait.mjs`). `?asyncify=1` forces that build in Chrome, to test it.

**Balls.** **Ball mass**, **Ball speed** and **Ball radius** are power-of-two multiples of
the scene's ball, in every mode. Mass and speed apply to the next shots. Radius reloads the
level, because the ball colliders and the broadphase grid are sized when it loads; the mass
stays as Ball mass sets it. The default speed is the scene's own, else three scene extents a
second within 5–18 m/s (`plan::default_shot_speed`). rubble has no swept collision, so a
very fast, small ball can pass through thin pieces.

**View.** The **View** menu picks what the pieces are coloured by: **Materials** (the
default), **Sleep state** (awake orange, asleep blue, static grey) or **Bodies** (one colour
per rigid body). In rubble's modes every piece is its own body, so Bodies gives each piece a
colour. The stress modes' Bodies view follows splits (see Stress solver, View options).

**Bond graph.** "Inspect bond graph" draws each bond as a line, hidden once the bond breaks.
In **Breakable D6 joints** line k is the engine's joint k (every builder emits them in the
same order, `engine-fracture` `tests/bond_lines.rs`), and the draw reads that joint's broken
flag (`fracture_draw.slang`). With **None** there are no joints, so a line hides once its
pieces part. The stress modes draw their own (see Stress solver, View options).

**Physics rate.** The **Physics** menu sets the simulated frames per second and reloads
the level:
- **Auto** keeps each mode's own rate: 120 Hz for rubble's modes, 60 Hz for the stress
  solver.
- **60 Hz** and **120 Hz** force the rate (rubble's `params.dt`, or the stress scene's
  `sim.frame_dt`).

Breakable D6 joints are tuned for 120 Hz. In the stress modes, each animation frame catches
up on up to 0.1 s of simulated time. Before, it caught up on at most two physics frames, so
a 60 Hz display running a 120 Hz simulation dropped time even when a frame took 3 ms.

**Ground bonds (stress modes).** A Shardwright piece bonded to the world (the door frame,
the fence posts, the walls' bottom courses) is held by a breakable bond to a thin fixed
ground chunk (`stress_ref::shardwright::import_value_with(.., true)`), using the bond's own
area, material and strength. The piece comes free when that bond breaks. Before, the piece
was fixed for good and could not be destroyed. Unhit, the cuboid house stands with no bond
broken. The default import, which the benches and the guard use, is unchanged.

**Scene packs (stress modes).** A Vibe City pack runs in both stress modes
(`stress_set_pack`): the lab's own scene of it gives the camera and shots, and stress-ref
imports it under the mode's profile (`Profile::import_options`), as `stress_bench --pack
--config` does. The pack's node masses and supports (nodes of mass 0) are used; the bonds
to a support break like any bond. Packs without a per-material list (all of
`generated/vibe-city`) take stress-ref's calibrated strengths, since their scene-wide
Blast limits are not stresses. The status line shows (⚠) what the import could not take
from the pack as given:
- bond patches from the pack's own area and normal where no shared face is found (bonds
  between diagonal neighbours, which touch along an edge);
- bonds with no normal in the pack (7 in `fracturedTower`), taken from the face the two
  chunks share, else along the line between their centres.

Unlike Breakable D6, which uses the pack limits, `townhouse` sheds about 350 bonds under
its own weight before it settles in the stress modes.

**Learned impact mode.**
- Physics is the same as **None** (unbonded fragments). The network run is timed and discarded.
- Pack: `engine/stress-oracle/packs/house-1story.json` (319 chunks / 877 bonds).
- Rebuild WGSL after editing the Slang:
  ```sh
  slangc engine/slang/kernels/learned_impact.slang -I engine/slang -target wgsl \
    -line-directive-mode none -o engine/crates/engine-webgpu/generated/wgsl/learned_impact.wgsl
  ```
- Bindings are packed to 8 storage buffers so Chrome's per-stage limit is met.

**Simple threshold mode (v0).**
- A deliberately trivial bond rule, forward only. It proves the path any stress model
  shares: contacts in, bonds broken, structures split into child bodies with their
  colliders re-parented, rubble continuing. A learned model would take the same slot.
- `Profile::Threshold` (`stress-ref` `capability.rs`); kernels in
  `stress-gpu/shaders/threshold.slang`, with strike detection shared with the GPU punch
  (`strike.slang`).
- **Predicted strikes, before rubble steps (no correction pass, no re-simulation):**
  - At the end of each resident frame, in its submission, `routing.slang`'s check flags a
    projectile that reaches an island within the next frame.
  - `threshold_predict` takes the strike's force: the impulse of an inelastic hit over the
    Hertz contact time, from the closing speed, the masses and the moduli.
  - It breaks every intact bond of the struck island whose tensile capacity
    (strength × area × factor) is below the force reaching it. That force spreads from
    where the projectile meets the island as a point load on a half-space does:
    F·A / (2π d²).
  - The GPU split pass then splits the island in the same submission, so rubble's next
    step meets the pieces, not the intact wall.
- **After each step:** a bond breaks when the contact force on either of its chunks exceeds
  its tensile capacity (`threshold_breaks`, in place of the static solve). This covers
  debris and settling.
- Every frame after the first is resident: one submission and one read, no coupled or
  host-orchestrated island frame. `GpuWorld::correct_frame` (PhysX's correction limit)
  exists only behind `STRESS_CORRECTION_LIMIT=1`, a measurement, off by default; see
  AGENTS.md.
- **Exit speeds (native):**

  | Shot | `blast_parity` | Threshold |
  |---|---|---|
  | Free-wall ram (reference 35.1 m/s) | 11.2 m/s | 33.3 m/s, 151 fragments |
  | Cuboid house, lab ball 1400 kg at 60 m/s | 18 m/s | 56.4 m/s |

  On the veneer house the lab ball meets a broken-off level-0 panel heavier than itself
  and goes on at 18 m/s in both profiles, without bouncing.
- **Frame time, native, cuboid house, frame 30:**
  - threshold 4.2 ms: rubble 2.4, stress cascade 0.85, 408 bodies;
  - `blast_parity` 5.8 ms: rubble 2.65, cascade 2.1, 219 bodies.
- The **Bond strength** slider scales the capacities.
- **Native:**
  - `STRESS_PROFILE=threshold cargo run --release -p stress-rubble --features view --example fracture_glb`;
  - test `stress-rubble/tests/threshold_split.rs`.

**Stress solver mode.**
- It runs the same Rust as the native reference (`stress-rubble` example
  `fracture_glb`): the level's Shardwright stress export, stress-ref's import, the
  resident GPU frame, and `stress_rubble::render`.
- It uses the lab's camera (every view mode) and shot rule (`engine-fracture` `view.rs`,
  `plan::shot_velocity`).
- It runs one copy of the structure; the samples' layouts of copies are the other modes'.
- **JSPI or Asyncify.** The solver reads the GPU synchronously, so in the browser each read
  suspends the wasm stack to the event loop (`engine-gpu-wait`, `gpu-wait.mjs`, the page's
  import map): through JavaScript Promise Integration in Chrome and Edge 137+, else through
  the Asyncify build. `build-web.sh` makes that build with Binaryen's `wasm-opt --asyncify`
  (pinned, downloaded once). Only the import `rubble-gpu-wait.yield_to_browser` unwinds.
  wasm-bindgen's externref shims are left out (Asyncify cannot instrument externref; they
  never wait). It is about 6% larger than the JSPI build.
- **Build:**
  ```sh
  CARGO_PROFILE_RELEASE_DEBUG=0 wasm-pack build engine/stress-oracle/crates/stress-web --release --target web \
    --out-dir ../../../crates/engine-webgpu/generated/stress-web --no-typescript
  ```
  `engine/scripts/build-web.sh` includes this step.

**Native reference** (house, level 0, one cannonball, 120 frames, M3 Max): 15 fragments,
151 of 453 bonds broken, median frame 4.2 ms.

**Status in Chrome (M3 Max):**
- It loads, draws, aims and fires like the other modes, and it holds real time (60 physics
  frames/s, 2.3 ms each) until a ball strikes.
- **Bonds break and split on the GPU.** A ×64 ball on the veneer house, level 0: 140
  bonds broken, 64 fragments, 61 physics frames/s.
- **Chrome allows 10 storage buffers per shader stage; the stress kernels fit it two ways:**
  - Each pipeline's layout holds only the bindings its entry point uses, when its module
    declares more than the device allows. `build.rs` reflects them from the generated WGSL
    with Naga.
  - The kernels that use more than 10 on their own run as phases: `pass_snapshot`,
    `pass_log`, `layout_parent`, `spawn_bodies`, `spawn_bodies_group`,
    `spawn_colliders`. They do the same work in order (`reflection::PHASED`,
    `Gpu::staged`), and the fused kernel still runs where the device allows it (Metal: 31).
  - `tests/binding_limits.rs` keeps every entry point within 10.
  - Lossless: the guard's digests are unchanged. Forcing the phases and per-entry layouts
    natively (`STRESS_GPU_STORAGE_LIMIT=10`) gives the fused kernels' digest on both Metal
    and native WGSL.
- The same WGSL kernels run natively (`STRESS_GPU_SHADERS=wgsl`, where Metal allows 31)
  with the Metal library's result: a lab shot (×64 ball) breaks 210 bonds into 17
  fragments on both.
- **WGSL rules the Slang source now meets:**
  - no constant infinity or NaN (`common.slang` `float_bits`);
  - barriers only under workgroup-uniform control flow (`group_uniform`,
    `group_uniform4`, `group_uniform_island`, and uniform reductions in `group_sum2`);
  - storage access read from the shader's declarations (`gpu.rs` `wgsl_access`).
  Metal's values are unchanged (guard).
- **View options.** Both stress modes draw "Inspect bond graph" and the **View** menu
  with their own renderer (`stress-rubble` `render.rs`, set by `stress_set_view`):
  - **Bond graph:** each intact bond is a line between its chunks' centres, drawn over the
    geometry and gone once the bond breaks.
  - **Sleep state:** the lab's colours. The coupled engine keeps every body awake, because
    a sleeping body's support loads would leave the stress solve. So here "asleep" (blue)
    means at rest, slower than 0.05 m/s and 0.05 rad/s; moving bodies are orange and
    static ones grey.
  - **Bodies:** one colour per body (cluster). A structure starts as one body, so one
    colour. A split retires the body and gives each piece a new cluster id. The piece with
    most of the old body's chunks keeps its colour, and the others take new ones
    (`BodyColors`), so a body keeps its colour from frame to frame and you can watch it
    break up.

### Takes, instant replay and slow motion

`director.mjs` sequences what every bond mode already does (steps, shots along rays, the
orbit camera, the view); the physics is the mode's own.

- **Speed** (the menu): real time, ½ or ¼. Slow motion steps fewer physics ticks per
  displayed frame at the same step size, so the physics is unchanged.
- **Instant replay** (the button, or R) re-runs the scene from its load, firing the same
  shots (logged as rays, with the ball settings of the time) at the same ticks:
  1. it catches up to half a second before the last shot without drawing much;
  2. it plays 2.5 s at quarter speed while the camera swings back to the live view;
  3. it catches up to where the run was, and play carries on live.

  The replay never feeds back into the run: it is a fresh run. The status line reports
  whether it reached the same state, which is:
  - the stress modes' frame, fragment, body and broken-bond counts;
  - in rubble's modes, a hash of every body's record.

  The threshold and stress modes repeat exactly in recorded takes. Rubble's modes repeated
  on one house (bit-identical body records). On the 7 × 7 cuboid grid (about 11k awake
  bodies), three runs of the same take gave three different hashes, so a replay there is
  reported as reaching a different state.
- **`window.__record(script, opts)`** plays a take frame-locked and uploads each frame for
  `record-takes.mjs` (`takes/lab.mjs` documents the segment fields: video frames, `slow`,
  camera moves, shots, view). The stress modes step exactly the frames asked for
  (`stress_frame_steps`), never dropping time to the wall clock.

**Dust and debris** (View options, stress modes, on by default): dust puffs and chips
where a chunk's velocity jumps (an impact, a break, a landing). They are computed on the
GPU from the poses the renderer draws (`stress-rubble` `fx.rs`, `Renderer::set_fx`) and
advance with simulated time, so they slow down in slow motion and freeze when paused.
It is drawing only: the physics is the same either way (the recorded takes break the same
bonds with it on or off). Native renderers leave it off.

### Breakable D6 joints (experimental)

“Breakable D6 joints” turns each baked bond of the selected level into
a D6 bond joint (`JointDesc::bond`, `JOINT_FLAG_BOND`; rows in
`slang/physics/joints.slang`, see `engine/docs/JOINTS.md`):

- **Rows.** Six AVBD rows in the bond frame (normal, u, v; bending about u and v;
  torsion) about the baked rest pose. Each row has a fixed penalty and an
  accumulated multiplier.
- **Penalties.** They are Shardwright's elastic springs (`kn = A/ce`, `ks = A/cg`,
  `ktu,v = Iuu,vv/ce`, `kt = J/cg`; `bonds.rs bond_springs`) times one
  scene-wide factor, so redundant bonds share load in the bake's proportions.
  Rows stiffer than 5× their lighter body's inertial term are capped:
  stiffer rows lock bonded clusters in Gauss-Seidel.
- **Breaking.** A bond breaks when `max(Fn⁺/T + M/Mcap, Fn⁻/C, V/S) > 1` on the
  step's final iteration. The capacities come from Shardwright's reference
  stress solver: material and interface strengths × area × strength scale
  (`bond_capacities`). Broken bonds leave the exact joint-pair hash, so their
  pieces collide again.
- **Sleep and wake.** Bonds join sleep islands and wake their partners.
- **Preloading.** The scene first settles one house with breaking off
  (`Scene::preload_variant` / `settled`, driven by each host) and starts every copy from those poses.

`bond-regression.mjs` checks the welds on the GPU:

| Case | Result |
|---|---|
| Strong hang | holds within 0.03 mm |
| Weak hang | breaks |
| Resting in compression | holds |
| Cantilever | 0.1° tilt |
| Weak bending | breaks |
| Bonded pair released from rest | falls and lands intact |

**Measured on house scenes (M3 Max), not yet good enough to enable by default.**
With 8 iterations a whole house of per-fragment welds never fully settles:
- long spans creep (roof pieces still sag after 6 s);
- 20 g slivers jitter;
- a woken house therefore never sleeps, and about 1% of its bonds break under
  self-weight.

| | 49-house grid | 20-house row |
|---|---|---|
| Peak GPU per step | ~30 ms (38 ms unbonded) | ~36 ms |
| Bonds broken | 11% | 47% (Level 3) |
| Idle (all asleep) per step | 6.4 ms | 5.7 ms |

The likely fix is structural: simulate each intact bonded island as one compound
rigid body (exactly rigid, sleeps, one body per standing house), evaluate bonds
with a stress solver, and split islands where bonds break, as Shardwright's own
reference simulator does.

`veneer-house-hull16.glb` is a fresh Shardwright bake of the same source
(`benchmarks/assets/veneer_house.glb`, seed 1337) with `assets/veneer-house-hull16.toml`
(building_lite plus `max_hull_vertices = 16`, enforced by its `collision_shapes`
gate), exported with `export-glb --uncompressed`. Current Shardwright fractures
this source into 142 / 614 / 3,619 / 6,592 fragments, more than the bundled
house's bake, independent of the cap. Level 0 has one 7 m slab hull whose stored
faces are 16 µm non-planar, beyond the f32 allowance of the convexity check, so
that level is rejected; the row uses level 3.

Drop another Shardwright fracture GLB onto the page or use the file picker.
“Download extracted physics metadata” exposes the parsed physical asset and
materials for another consumer, including a future stress solver.

## Export your already-baked asset

In the Shardwright checkout (the CLI binary is named `prefracture`):

```sh
export RUSTUP_TOOLCHAIN=stable
export WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
cargo run -p frac-cli --release -- export-glb \
  --input out/my-model.glb \
  --physics out/my-model.fracphys \
  --out out/my-model.fracture.glb \
  --uncompressed
```

`--physics` defaults to the input path with `.fracphys` suffix. Supply
`--materials path/to/materials.toml` when the bake used a custom material library.
`--uncompressed` is required by this minimal demo's accessor reader; the exporter
also supports preserving meshopt for other consumers. Packaging never regenerates
fractures. The resulting file is the only asset the page needs.

Bundled asset provenance: Shardwright's `out/structs/masonry_arch/masonry_arch`
render/physics pair, seed 1337, built-in material library 1.0.0. Exported using the
command above with `--uncompressed`; all four levels are included (751 fragments,
2,821 bonds total). The asset's bake settings hash remains in its physics payload.

The house is Shardwright's `veneer_house` asset, not the separate
`two_storey_building` benchmark. Source: `out/structs/building_lite/veneer_house/`
render/physics pair, seed 1337, built-in library 1.0.0, exported with
`prefracture export-glb --uncompressed`. Its 54.1 MB GLB includes all four levels;
no source sidecars or runtime bake are required.

## Contract and implementation

- `engine-fracture/src/asset.rs`: bounds-checked GLB / schema-1 FlatBuffers reader;
  reusable independently of any GPU. Exposes full tensors, bond frames/areas/moments,
  composition, strength scales, reinforcement, boundary loops and child mappings.
  Interface material IDs share the material ID space:
  `library.interface_material[id - library.material.length]`; -1 means unspecified.
- `engine-fracture/src/scene.rs`: selects a hierarchy level, converts full inertia
  to a principal basis, transforms both render and collision geometry to that
  basis, and cooks baked convex topology into the **packed hull ABI v2** with
  `engine-schema`'s cooker (`hull-golden.json` pins its bytes). Each compound child
  is recentred on its own bounds and placed by the collider's local position. CPU
  work is decoding/conversion, not hull generation.
- `simulation.mjs`: replays the plan (`engine-fracture/src/plan.rs`), which
  schedules Rubble's **core** Slang AVBD kernels with
  persistent hot/cold contacts, the hierarchical size-level broadphase, the convex
  narrowphase (`convex_classify` / `convex_sat_small|large` / `convex_manifold`),
  the block CSR scan, stable colour tiles, degree-aware primal lanes and sized
  sleep work — the same schedule as `engine-webgpu/src/gpu.rs`. Rendering reads
  GPU body poses directly. Batches contain 1–4 ticks with one counter readback per
  batch; rendering is submitted before awaiting that readback. Slow ticks reduce
  the batch size, and fully sleeping scenes skip GPU physics.
- Convex collision lives in the core library (`slang/physics/convex.slang`,
  `slang/kernels/convex_step.slang`); see
  [`engine/docs/convex-hulls.md`](../../../docs/convex-hulls.md). The former
  `fracture_step.slang` kernels and `fracture_export` were retired.

The vendor extension stores the original FRAC bytes in a standard GLB bufferView
and embeds the full material library in root JSON. It is optional for rendering
and is **not a Khronos standard**. See Shardwright's `docs/GLB_FRACTURE.md` and
`schemas/frac.fbs` for the versioned contract. Generic optimizers must preserve or
remap the custom bufferView and embedded glTF node references.

## Scope

This demonstrates **import and rigid-body collision**, not bond failure or a stress
solver. Bonds are inspection lines. Fragments start sleeping at their baked poses on the
floor and wake through contact with a projectile or moving neighbor. This initial
rest state is not a solved bonded equilibrium; the world anchors are visualized
but are not fixed constraints. Physical strength properties are available
to consumers but are not converted into AVBD breakable constraints. Contact
friction is currently a demo-wide 0.5; surface shading uses glTF base colors.
Textures, arbitrary edited node transforms, mixed hierarchy frontiers and runtime
fracture refinement are not implemented. Input must follow Shardwright's baked
asset/COM coordinate contract (meters, Y-up, right-handed).

Convex limits are 64 vertices, 128 faces and 192 edges per hull. Hull pairs are
compacted into a small bucket (≤16 vertices: 16 lanes per pair, 4 pairs per
workgroup) and a large bucket (64 lanes per pair). SAT works in one hull's local
frame from shared memory and prunes edge axes with adjacent-face Gauss maps; the
previous frame's feature gives hysteresis and a cached-face fast path that skips
the edge query. Clipping streams points into a deepest + extremes reducer (no
polygon buffers) and keeps at most four points (`MAX_CONVEX_CONTACTS`).
Generation-gated contact reuse skips SAT/clipping for unchanged contact patches.
Compound mass and inertia remain on the body; child hulls are never replaced by
spheres.

At import, exact duplicate vertex records are welded for adjacency, and coplanar
face patches are normalized. Original vertex coordinates/order remain available.
Uncertifiable edge adjacency selects exhaustive vertex-projection SAT instead of
unsafe Gauss-map pruning. Degenerate zero-volume hulls are rejected with their
fragment/hull ID. The bundled house's levels 2 and 3 contain such source hulls and
need recooking; levels 0 and 1 remain usable. A failed level selection preserves
the previous scene. No GLB extension change or runtime fracture is involved.

The hierarchy inserts each collider into one power-of-two size level. Queries
visit their own and coarser levels; fixed/sleeping fine colliders also probe coarse
movers. Exact cell coordinates disambiguate hash collisions. Same-body pairs are
excluded and pair ownership prevents duplicates. Cell widths conservatively cover
proxy diameters, including f32 padding. There is no fixed-capacity large-body list
or pair-truncating chain shortcut. Storage capacities still report overflow.

The target remains **10,000+ active rigid bodies**. Correctness at that count does
not imply a 120 TPS performance guarantee. Use the scale regression below and
record hardware, active counts, candidates, contacts and timings. The optional
`simulation.profileStep()` reports GPU timestamp phases when supported; ordinary
HUD timings include queue/readback waits. Projectiles still use discrete collision,
not CCD. Bonds remain inspection metadata rather than stress constraints.

The Gauss-map/contact-selection port is based on three-avbd at `2bb1a318`;
its MIT copyright/license is preserved in `engine/slang/THIRD_PARTY_LICENSES.txt`.

## Headless physics benchmark

Physics only: no canvas and no renderer, with WebGPU on the real GPU in
headless Chrome. It prints JSON with ms/tick (ticks queued back to back),
the counters, and per-kernel GPU times.

```sh
nvm use
node engine/crates/engine-webgpu/fracture/headless.mjs grid warm=420 measure=120
node engine/crates/engine-webgpu/fracture/headless.mjs cuboidGrid budget='{"deterministic":true}' trace=60
```

- Scenes: `grid`, `suburb`, `house`, `cuboidGrid`, `cuboids`.
- `shot=none` skips the row shot. `profile=0` skips the kernel timings.
- `budget` is merged into the scene capacity. Use
  `{"deterministic":true}` for run-to-run identical results, so two shader
  builds can be compared tick for tick.
- `CHROME=` picks the browser (default: Google Chrome).
- `bonds=1` uses the lab's breakable bonds. `columns=`/`rows=` change the
  grid layout.
- `staticUnhit=1` makes the houses outside the shot's column static. This
  changes behaviour and is only for estimates.
- `CHROME_ARGS` passes extra browser flags, for example Dawn toggles.
- No npm dependencies; see `headless-page.mjs` for the protocol.

## Verification

```sh
nvm use   # .nvmrc: Node 22.22 (node:module stripTypeScriptTypes needs >= 22.13)
node --test engine/crates/engine-webgpu/fracture/*.test.mjs
cargo test --manifest-path engine/Cargo.toml -p engine-oracles --test convex_slang_cpu
```

`simulation.test.mjs` drives the browser host with a mock WebGPU device against
the exported WGSL (no GPU). The node tests run the lab's wasm path (build it
first, see *Run*); the scene, bond, hull and cuboid-house logic is tested in Rust:

```sh
cargo test --manifest-path engine/Cargo.toml -p engine-fracture --release
cargo test --manifest-path engine/Cargo.toml -p engine-viewer --release --test fracture_regression
``` `convex_slang_cpu` runs the production Slang convex
and broadphase kernels on the CPU (`slangc -target cpp`) against exhaustive
oracles; see `engine/docs/convex-hulls.md`.

Checks the real bundled asset's hierarchy/materials, asymmetric inertia conversion,
render/collision frame agreement and malformed containers. In the browser console,
after selecting **Masonry arch**, level 0:

```js
await (await import('./regression.mjs')).runFractureRegression()
```

Checks 120 GPU steps without pre-impact drift, then shoots the crown and runs
720 more steps. Checks wake-up, fragment motion, contacts, finite poses, unit
orientations, no overflow, hull/sphere floor support, and exact reset of all body
fields and ammunition. Shardwright's `cargo test -p frac-io --test
fracture_glb` checks byte-preserving physics round trips, compressed/uncompressed
render compatibility, determinism and invalid pairings.


For GPU convex conformance and a 10,000-active-body, 64-vertex-hull workload:

```js
await (await import('./convex-regression.mjs')).runConvexGpuTests({scale: 10000})
// add boundary:true for work-list row-limit cases (32,768+ SAT pairs)
```

The conformance suite compares actual GPU contacts with exhaustive CPU SAT,
checks exact bounds, arbitrary rotations, mixed-size/static/sleeping/compound
pair ownership and duplicates, and covers 64-vertex prisms plus 124-face/186-edge
bipyramids. It also checks an open compound frame (a sphere in the hole must not
collide), a sphere touching its jamb, and batched contact-reuse pose parity.
It also covers hull/box, hull/sphere, hull/capsule and hull/plane pairs, dense
compounds, SAT work-list sizes, and reuse/hint on versus off. Scale results
(dense 64-vertex prisms with reuse off, 4-child compounds, small rubble) include
active count, contacts, candidates, overflow, wall time and per-phase plus
single-pass GPU timestamps; no asleep benchmark shortcut. Shader changes require
rerunning `wgsl_export`. The packed hull ABI is v2; do not mix older host code or
generated shaders with the new importer.

The measurements below predate the core convex path (they were taken with the
retired `fracture_step` kernels) and are kept as the baseline to beat.

The recorded [local validation](./convex-validation.json) on 2026-10-14 used
Chrome WebGPU with an Apple / metal-3 adapter. The dense scale workload kept
10,000 bodies active, with about 35,900 candidates and 112,100 contacts, and no
overflow. Five ticks averaged **282 ms/tick**; a separate timestamp sample spent
about **96 ms in narrowphase and 106 ms in AVBD solving**. These are local,
non-isolated measurements, not a real-time result. Further optimization must
address both collision work and the shared solver/dispatch costs.

For the impacted house level 1, five timestamp samples had roughly 7,600–7,700
candidates and 7,700 contacts; narrowphase took 0.8–15 ms and solving 86–136 ms.
The timings varied substantially. This confirms that tighter bounds/contact
reuse do not by themselves eliminate the solver bottleneck. The core convex path
also runs on the native Metal and WebGPU (Rust) hosts; CUDA only compiles it.
