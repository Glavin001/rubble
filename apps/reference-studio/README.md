# Reference Studio

A static web app for crash (and later any node-beam structure) reference images: collect
screenshots and video frames, trace each structure like a stick figure, **recreate the
crash** in Rubble (pick the vehicles, place what they hit, simulate on the page's GPU), set
each image's camera and moment against the recorded run, and compare the shapes. People
and AI agents use the same actions (WebMCP, `window.studio`).

Everything runs in the browser: the simulation is Rubble's explicit solver on WebGPU (what
production runs), and every number shown (scene placement, first contact, the report,
Rubble's outlines, the comparison, design ranking) is Rust compiled to wasm. There is no
server: data goes to a folder you choose.

## Run it

```bash
cd apps/reference-studio && nvm use && npm ci          # once
# The engine (from the repo root; about 3 min the first time, about 70 s after a Rust edit):
RUBBLE_WEB_PROFILE=fast RUBBLE_WEB_PARTS="kernels world" bash engine/scripts/build-web.sh
npm run engine      # links engine/crates/engine-webgpu/generated/{world-wasm,kernels} into public/rubble/
npm run designs     # generates the vehicles into public/designs/ (vehicle_gen_rig; about 30 s, 10.5 MB)
npm run dev         # http://127.0.0.1:8770/
```

On first open the studio asks for its **workspace folder**: choose
`apps/reference-studio/workspace/` in the repo (it remembers the folder and asks for
permission again on return), or use this browser's own storage. Use Chrome or Edge with
WebGPU.

`public/rubble/` and `public/designs/` are not committed (`npm run engine` and
`npm run designs` rebuild them from the repo's code). `npm run engine` links the fast
build's files; for a static build that copies them, `node scripts/engine-assets.ts engine`.
The fast build's world wasm is 23 MB and reads its kernels (60 MB of `kernels/`) at run
time; a release build (`bash engine/scripts/build-web.sh` with `RUBBLE_WEB_PARTS="kernels
world"`, no fast profile) embeds them. `npm run designs -- --visual` also keeps each car's
render model and skins (about 200 MB a car; the Scene view does not draw them yet).

## Storage and browsers

One workspace folder holds everything:

| folder | holds | committed |
|---|---|---|
| `shots/`, `frames/`, `structures/`, `scenarios/` | points, text and numbers | yes |
| `library/` | the images | never (gitignored) |
| `recordings/` | cached runs, `<scenario hash>.rrec` | never (gitignored) |

Two ways to open it (`shared/workspace.ts`, one interface):

- **A folder you choose** (File System Access API, `showDirectoryPicker`): Chrome and Edge
  86+ only; not Firefox, not Safari (caniuse, November 2026). The folder's handle is kept in
  IndexedDB; read-write permission is asked again when you come back.
- **This browser's storage** (the origin private file system): every current browser
  (MDN: Baseline since March 2023; writing a file, `createWritable`, needs Chrome/Edge 86+,
  Firefox 111+, Safari 26+). The data then lives only in this browser: the header says
  "this browser only". The smoke test uses it.

Both need a secure context (`https://` or `http://127.0.0.1`). The simulation needs
WebGPU; without it the Scene view says so. A wasm CPU fallback exists for automated tests
only (`?cpu-fallback`), and its result is labelled "CPU fallback (tests)" wherever shown.

## Use it

1. **Library**: shots (one crash run each) and their frames. New shot; add images (drop
   them anywhere).
2. **Annotate**: mark regions (real, BeamNG...), choose the structure, trace it (guided).
3. **Link**: link the shot to a scenario: a built-in one (S01...S14) or a recreated one.
4. **Scene** (Alt+5), two activities:
   - **Setup**: open a scenario (built-in ones are read-only: *Copy as data to edit*), or
     *New scenario* (a generated sedan at 50 km/h and a wall). Select a body or a box in
     the view or the lists, drag it along the ground or type its numbers; change a body's
     design (the panel's *Match a design from a tracing* ranks the generated designs
     against a traced side view), speed, yaw; add and remove boxes and bodies. Every edit is
     saved at once and undoable. **Simulate** runs it on WebGPU and records every 4 ms.
   - **Review**: scrub, play (slow motion), step 4 ms at a time; move the camera (orbit,
     right-drag to pan, wheel, field of view, roll). Pick a reference image: its traced
     outline and Rubble's outline at the shown time are drawn over the 3D view, which takes
     the image's aspect. When they line up, **Pin this image here** stores the time and the
     camera on the frame. Images are listed pinned, not pinned, or *check after
     re-simulation* (pinned on an earlier recording: a setup edit keeps every pin).
5. **Compare**: each source against its own "before", Rubble from the recording at each
   image's pin and camera, and the shape numbers.

What is live and what is not (measured in the smoke test and by hand, Chrome on an M-series
Mac): moving or resizing a box, moving or turning a body or changing its speed re-places
the scene in Rust in 2 to 8 ms (no respawn, no kernel compile); changing a design loads that
design once (a fetch and a parse) and re-places; Simulate builds the GPU fleet from the
description: S02 (an SUV, 0.6 s) took 21 s the first time (kernel compile) and 0.6 s after;
the CPU fallback takes about 2 s. Camera moves, scrubbing and playback never simulate.

## Where the maths lives

All Rust, compiled into the page's wasm (`engine-world-wasm`, `src/studio.rs`):
`engine_softbody::crash_reference::studio` (placing a scene, the recording, positions at a
time, outlines, the comparison), `desc` (scenarios as data, first contact, law criteria),
`recording`, `registry`, `outline::BodyParts::outlines` (the one body-outline function,
also `crash_outlines`'s), `shape` (tracings, frames, measures, the moment/camera rule) and
`shape::design` (design ranking). TypeScript holds UI state and calls them; the types that
cross are generated from the Rust (`src/generated/crash-reference.ts`, `ts-rs`; `BLESS=1
cargo test -p engine-softbody --test studio_types` regenerates it and the test fails when
it is stale).

## For agents

The tool list is `shared/tools.ts` (name, title, description, input schema, read-only or
not): *data* tools edit the workspace (shots, frames, tracings, scenarios, boxes, bodies,
pins, cameras) and read computed results (outline, comparison, design matches); *page*
tools drive the views and the Scene (open_scene, set_mode, simulate,
get_recording_status, seek, step, play, pause, reset, set_playback_speed, get_report,
set_scene_camera, select_image, pin_image, jump_to_pin, select_object, undo_scene,
redo_scene, describe_scene). Two routes:

- **WebMCP**: each tool registered with `document.modelContext.registerTool` (the W3C
  polyfill `@w3c/webmcp-polyfill` 0.2.0 where the browser has none; `window.studio.webmcp`
  says which).
- **`window.studio`**: `studio.tools`, `await studio.call(name, input)` or `await studio.<name>(input)`.

A recreate session: `create_scenario` → `open_scene` → `set_box` / `set_body` →
`simulate` → `seek` → `set_scene_camera` → `pin_image` → `get_rubble_outline` →
`get_comparison`.

## Checks

- `npm run typecheck`, `npm run build` (the static site in `dist/`)
- `npm test`: workspace paths, image headers, the data check, the data tools on an
  in-memory workspace
- `npm run check-data`: the committed workspace holds points and text only
- `npm run smoke` (after `npm run build`): headless Chrome on `dist/` with OPFS, no server
  but a static file server; the whole recreate flow (see `tests/smoke.ts`) on the CPU
  fallback; screenshots in `engine/out/reference-studio/screenshots/step2/generated/`.
  `python3 engine/scripts/with-gpu-lock.py --limit 300 -- npm run smoke -- --webgpu` runs it
  on WebGPU; `--real` recreates S02 with the repo's S02 frames (third-party footage: not for
  sharing).
- Rust: `cargo test --release -p engine-softbody-lab --test crash_scenario_data`,
  `cargo test -p engine-softbody --lib crash_reference`, `cargo test -p engine-softbody --test studio_types`

## Deploying

`npm run build` writes a static site to `dist/` (relative paths, any static host). On
Vercel: project root `apps/reference-studio`, `vercel.json` sets the build command, the
output directory and `application/wasm` for `.wasm`. The engine files must be in
`public/` before the build (run `npm run engine` without `--link`, i.e. `node
scripts/engine-assets.ts engine`, and `npm run designs` in the build, or upload a
prebuilt `dist/`); prefer a release world build (kernels embedded, no 60 MB `kernels/`).
No cross-origin isolation headers are needed: the wasm uses no shared memory or threads
(`SharedArrayBuffer` is not used). Deploying is the owner's action.

Safety: no server; the page reads and writes only the folder you chose (paths checked to
stay inside it) or its own storage.
