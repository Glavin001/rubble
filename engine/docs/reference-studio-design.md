# Reference Studio, step 2: recreate a reference crash (design note)

Step 1 (built, `apps/reference-studio/`): a library of reference images in shots, regions,
schema-driven tracings, a hand-set camera per image, a link to a Rust-defined scenario, and
the comparison computed in Rust (`crash_reference::shape`). Step 2 makes *recreating* the
crash easy: from a traced image, pick the vehicle that matches, place what it hits, set the
camera, press Simulate, and compare, all in the studio, all live.

**Status (2026-11-27): built**, with the owner's answers to the open questions below and two
later decisions: the studio is client-side only (a static site over a workspace folder,
everything computed by Rust in wasm, the run on the page's WebGPU) and the Scene view has
Setup and Review (images pinned to a time and a camera of the recorded run). See
`apps/reference-studio/README.md`. Deviations from this note: a data scenario places
bodies and boxes absolutely (no `gap_s`/`face` relative placement); `CrashScenario` keeps a
`fn` for its criteria (a data scenario's are one of the law sets); the soft bodies are drawn
by three's `WebGLRenderer` from node positions the recording gives (not a TSL storage
buffer); data scenarios live in the studio's workspace (`apps/reference-studio/workspace/
scenarios/`), not `engine/docs/`; no camera solve and no ffmpeg ingestion yet.

## The direction (owner, 2026-11-27)

- **(a) A TypeScript library generated from the Rust**, types and calls, the Rust the only
  source of truth. **Recommendation: `ts-rs` for the types and `wasm-bindgen` exports for
  the calls.** `ts-rs` is already a workspace dependency (engine `Cargo.toml`) and in use
  (`engine-world/src/play/api.rs`, `#[derive(TS)]` on the play API types); `wasm-bindgen`
  already exports the crash lab (`engine-world-wasm/src/crash_ref.rs`:
  `crashReferenceRegistry`, `crashInfo`, `crashSample`). `tsify` is not used in the repo and
  would be a second way to do what `ts-rs` does. So: derive `TS` on the scenario description,
  `Shot`/`Frame`-side types that Rust reads (the structure schema, `Instance`), the comparison
  result; generate `.ts` into `apps/reference-studio/src/gen/` from a test (`cargo test
  export_bindings`, as ts-rs does) and fail CI when the generated files differ from the
  committed ones; the wasm exports take and return those types as JSON. The thin
  hand-written TS layer is only the wasm loader and a typed `call()` wrapper. The step 1
  types in `shared/model.ts` then become generated (the Rust `shape::Instance`,
  `ElementValue` and `Schema` gain the frame and shot types).
- **(b) Editor components on top of it**: the page calls the TS library, which calls the
  precompiled Rust in wasm, which builds the scene. React owns the panels and forms; the 3D
  view is one component owning a canvas (below).
- **(c) Truly real-time editing**: click a box, move it, change its size, mass or fixed;
  pick a vehicle by design, set its pose and speed; press Simulate and it runs at once, no
  rebuild, no reload (scenario parameters apply at runtime).
- **(d) Per-image cameras moved by hand in the 3D view**, the frame's traced outline drawn
  over the live view, saved back to the frame's `camera` (the step 1 field).
- **(e) The scenario description is data** the editor reads and writes and that the same
  `CrashScenario::build`/`run` code executes in tests, on CUDA and in the page.

## (e) Scenarios as data, beside the Rust-defined ones

Today a scenario is Rust: `CrashScenario { layout: fn(&dyn Fn(&str) -> Extent, Jitter) ->
Layout, cameras: &'static [Camera], moments: &'static [Moment], criteria: fn() -> Vec<Criterion>, .. }`.
The layout is a function because positions depend on the designs' extents (S02: the wall's
face is the SUV's nose plus 0.1 s of travel) and on the run's jitter.

Changes in `engine/crates/engine-softbody/src/crash_reference/mod.rs`:

1. **`LayoutSpec` as data** (serde + ts-rs), the function's inputs made explicit:
   - bodies: `{ name, design, at, yaw_deg, speed_kmh | velocity, carrier }`, where `at` may
     be relative to another body's or box's face (`{ "gap_s": 0.1, "to": "wall", "face": "-z" }`)
     so a data scenario keeps "a tenth of a second before the wall" whatever the design;
   - boxes: `{ name, centre, half, fixed: true }` (later `mass` for movable rigid blocks,
     which need the rigid coupling the World has, not the fleet's `StaticBox`);
   - `impact_s`, `secs_after_impact`, cameras (`FreeCamera`: eye, target, up, fov),
     moments, and which criteria apply (below).
   `LayoutSpec::resolve(&self, extent, jitter) -> Layout` does what each `layout` fn does
   today, generically: relative placements, the jitter on the striking body.
2. **Two sources, one type**: `CrashScenario` keeps its fields but `layout` becomes an enum
   `LayoutSource::Code(fn ..) | LayoutSource::Data(LayoutSpec)`, cameras and moments become
   owned (`Vec<FreeCamera>`, `Vec<Moment>` with `String` names) so data can fill them, and
   `all()` stays the Rust list. `from_desc(ScenarioDesc) -> CrashScenario` builds a data
   one. `build` and every runner (CPU, CUDA test, wasm) take a `CrashScenario` and do not
   care where it came from. A Rust scenario can be exported as data (`to_desc`) where its
   layout is expressible, which is also the test that the data path reproduces it bit for
   bit (S02 data vs S02 code: same fleet, same state words after N steps).
3. **Criteria stay in Rust.** They have bases (laws, cited data) that data cannot vouch for.
   A data scenario names generic law-based criteria only: momentum conserved, energy not
   created, nothing passes through, centre of mass of a symmetric head-on at rest (the
   `Criterion::law` set), plus "report" measures with no band. A recreated crash that earns
   cited bands is promoted to a Rust file.
4. **Where data scenarios live**: `engine/docs/crash-reference/studio/scenarios/<id>.json`
   (committed, beside shots and frames: a shot's `rubble.scenario` may name either kind).
   Tests: a native test loads every data scenario, builds it and runs a few hundred steps
   (no NaN, the law criteria); the CUDA test takes `--scenario-file`.

## (c) What is live and what needs a new fleet

| Edit | Cost |
|---|---|
| Move or resize a static box | update the fleet's box list and the contact pass's box buffer; no kernel compile, no respawn (boxes are per-fleet data, `StaticBox`) |
| Body pose or speed | reset that body's node positions and velocities from its template (the respawn path without re-reading the design); state buffers rewritten, same layout |
| Camera, moment, playback | nothing physical |
| Change a vehicle's design, add or remove a vehicle | a new fleet (`ExplicitBeams` for that design: new node and beam counts, new buffers); the kernels are already compiled for the solver, so it is a respawn of a fraction of a second, not a reload |
| Contact setting (discrete / edge / continuous) | a respawn with the flag; same compiled kernels |
| Box mass / movable boxes | needs rigid bodies in the World coupling (the soft-rigid World, `docs/handoff/soft-rigid-world.md`), not the crash fleet's static boxes: a later slice |

Simulate = rebuild the fleet from the current `ScenarioDesc` in wasm (`crashSpawnFromDesc`)
and step; Reset = respawn from the same desc. The page never builds geometry itself.

Changes in `engine-world-wasm` (`src/crash_ref.rs`): `crashSpawnFromDesc(desc)` beside
the id-based spawn; `crashSetBox(i, centre, half)`, `crashSetBody(i, at, yaw, speed)` for
live edits; `crashOutline(schema, camera)` (the step 1 `rubble_instance` computed in wasm on
the live state, so the ghost follows the simulation); `crashCompare(...)` (the step 1
`compare`, in wasm, so the studio's table no longer needs the native server for maths).
The crash lab page (`softbody.html?crashRef=`) gets the same desc path so a studio scenario
opens there too.

## The 3D view: plain three.js in a React component (confirmed, with one change)

What the repo has: `rubble-three.mjs` draws a World's rigid colliders in three's
`WebGPURenderer` on the World's own device, reading body poses from the engine's buffers in
the vertex stage (no readback); `three.html`/`play.html` use it. Soft-body skins are drawn by
the WebGPU renderer `showcase-car-render.mjs`, not by three.

- **react-three-fiber** would put the per-frame buffer updates and the physics-driven loop
  through its reconciler; the loop here is driven by wasm stepping and by buffers updated
  every frame, which is simpler imperatively. Not recommended.
- **Embed the crash lab's WebGPU renderer** in a canvas: the soft bodies are drawn today,
  exactly as the lab draws them; but gizmos (move a box, aim a camera), picking and the
  traced-outline overlay would all be new code in a renderer that has none of them.
- **Plain three.js, owned by a React component** (`<SceneView desc onEdit />`): three's
  `TransformControls`, `OrbitControls`, raycast picking and a 2D overlay for the traced
  outline come free; rigid boxes are `rubble-three.mjs` as is. Drawing a soft body in three
  takes a mesh whose vertex stage reads the engine's node-position buffer (the
  `rubble-three` pattern: a storage attribute over the engine's `GPUBuffer`, TSL indexing by
  vertex), first with the collision triangles flat-shaded (what `crash_outlines` renders:
  enough to edit and to align a camera), then the skinned visual mesh by porting
  `showcase-car-render.mjs`'s skinning to TSL.
- **Recommendation: plain three.js from a React component**, as the coordinator proposed,
  with the soft body drawn by a TSL node-buffer mesh (collision surface first). Embedding
  the lab renderer is the fallback if the TSL skin port stalls; the editor's controls would
  then be HTML overlays.

## (d) Cameras by hand, against a live ghost

The frame's `camera` (eye, target, up, fov, the region it applies to) is edited in the 3D
view: orbit/dolly/roll with the reference image as a translucent backdrop (or the traced
outline drawn over the 3D view), the Rubble outline in the same schema updated live from
`crashOutline`. Saving writes the frame's `camera` through the same tool (`set_frame_camera`).

**Camera solve (optional, a first guess only)**, reusing `engine/tools/vehicle-gen/compare.py`'s
Levenberg-Marquardt solve (position, orientation, focal length from pixel ↔ 3D pairs, priors
on height and focal length) moved to Rust beside `shape` (one implementation, the server and
wasm both call it). What a vehicle tracing gives, with the design's dimensions and the
scenario pose at that moment:
- wheel centres: 3D points known from the design (hubs in the car frame) and the body's pose;
- wheel radii in the image: each wheel a known 3D circle, so its image size gives depth over
  focal length at that wheel;
- the ground: the tyres' contact points (bottom of each traced circle) lie on y = 0;
- the sill and roofline give the car's vertical plane and the up direction.
Seven unknowns (3 position, 3 orientation, focal length); two wheels give four points
(centres, contacts) = eight equations, just determined. **One frame cannot separate focal
length from distance** in a near-side view (the car is nearly planar: an affine view fits
as well), and roll vs a sloped ground is weak with one wheel pair. First guess: fix the fov
at the lab's 60° (or a prior from the video), solve the other six, then adjust by hand
against the ghost. A three-quarter view (both wheel pairs) or two frames of the same fixed
camera constrain focal length.

## (b) Choosing the generated design from a tracing

Each generated design's `crash.json` (`ZoneMap.gauges`: tape measures between named nodes,
wheelbase, roof height, door opening...) gives its dimensions. A side tracing gives
dimensionless ratios: front overhang / wheelbase, rear overhang / wheelbase, roof height /
wheelbase, wheel radius / wheelbase, hood length / wheelbase, windscreen angle. Rank the
designs by the distance between those ratios (each weighted by its element's visibility:
hidden and blurred elements count less), show the top three with the design's own outline
in the schema drawn over the tracing (normalised to the wheelbase frame), and let the owner
pick and tweak (the generator's parameters are the tweak; a tweaked design is a new
generated design, never an edited template). This needs `crash.json` to carry overall
length and roof height gauges for every generated design (some carry only door gauges).

## Obstacle placement

The tracing does not say where the wall is in 3D; the scenario does. For a frontal test the
contact face is placed relative to the body (the `gap_s`/`face` form above); for an
offset or pole test, its lateral position comes from the overlap the source states (40 %
offset) or from the top view's tracing (the struck side line against the barrier edge). In
the 3D view a box is moved with a gizmo, and its contact face is snapped to a body's face
plus a gap. Obstacles are traced too: a `barrier-side` structure schema (two keypoints for
its face's top and bottom corners) lets the comparison check that the recreated face sits
where the footage's does through the same camera.

## (e) Video ingestion

The user supplies a local video file. The server runs `ffmpeg -ss T -i FILE -frames:v 1`
for chosen times (or `-vf fps=N` over a range) into the gitignored library, each frame
created with its shot and `time.video_s` set exactly (no more "approximate" times). The page
shows a scrubber (an HTML `<video>` element over the local file, served by the server from
an allowed folder) to pick the moments. Nothing is downloaded by the studio; footage stays
out of git.

## Build order (smallest useful slice first)

1. `LayoutSpec` + `ScenarioDesc` in Rust with `ts-rs`, `to_desc` for S02 and the
   bit-for-bit test (data S02 == code S02). Generated TS committed and checked.
2. `crashSpawnFromDesc` in `engine-world-wasm`; the crash lab opens a desc.
3. The studio's 3D view: three.js in a React component, collision surface via a node-buffer
   mesh, boxes via `rubble-three`, orbit; Simulate / Reset; load a shot's scenario.
4. Live edits: move/resize boxes, body pose and speed (no respawn), design change (respawn).
5. Per-frame camera in 3D, the traced outline as an overlay, save to the frame; the live
   ghost and `crashCompare` in wasm (the step 1 table without the native server).
6. Design ranking from the tracing; camera first guess; ffmpeg ingestion.
7. Movable/massive boxes through the soft-rigid World.

## Open questions for the owner

1. Data scenarios: committed beside the studio's shots (`docs/crash-reference/studio/scenarios/`)
   or as Rust-reviewable files in `crash_reference/` (data, but next to the code)?
2. Should a recreated scenario ever carry asserted criteria, or only law-based ones until
   someone writes it up with cited data in Rust?
3. Movable boxes with mass (a barrier on a sled, a parked car as a rigid block): needed in
   the first slices, or static boxes until the soft-rigid World hosts the crash lab?
4. The 3D view's soft body: is the collision surface (flat-shaded) good enough for the
   first slices, with the skinned mesh after, or is the skinned look needed from the start?
5. Camera fov: assume 60° unless known, or ask per shot (a video's lens is usually fixed
   across its frames: a per-shot fov would tie frames together)?
6. Design matching: rank the existing generated designs only, or also propose generator
   parameters (a new design) to fit a tracing?
7. Should the studio keep a native server once the maths is in wasm (it still has to write
   files and run ffmpeg), or become a static page writing through the File System Access
   API?
