# Reference Studio

A local app for crash (and later any node-beam structure) reference images: collect
screenshots and video frames, trace each structure on them like a stick figure, link them
to a Rubble scenario, moment and camera, and compare the shapes with Rubble's own, which it
outlines automatically. People and AI agents use the same actions (WebMCP, `window.studio`,
HTTP).

## Run it

```bash
cd apps/reference-studio && nvm use && npm ci   # once
npm run dev                                      # then open http://127.0.0.1:8770/
```

One command starts everything: Vite serves the React page and, in the same process, the
studio's file-system API (`server/`, bound to 127.0.0.1 only). The first comparison of a
linked shot builds the Rust tools if needed (`cargo build --release -p engine-softbody-lab
--example reference_shape --example crash_outlines`, about a minute from a warm cache) and
runs Rubble's outlines (about 10 s a scenario, then cached in `engine/out/reference-studio/`).

For a `.claude/launch.json` (the desktop app's Browser pane):
`{ "name": "reference-studio", "runtimeExecutable": "npm", "runtimeArgs": ["--prefix", "apps/reference-studio", "run", "dev"], "port": 8770 }`.
Another port: `STUDIO_PORT=8780 npm run dev`.

## Use it (ten steps)

1. **Library** lists every shot (one crash run: a video or a set of screenshots) with its
   frames as thumbnails and status badges: *untraced*, *traced*, *paired*, *linked*, *no image*.
   Search or filter; click any frame to reopen it. Links are shareable (`#/annotate/<frame>`).
2. **New shot**: a title and, if it came from one, the video URL.
3. **Add images…** or drop files anywhere on the page, choosing the shot they go into. They
   are copied into `library/` (gitignored); each becomes a frame.
4. **Annotate** a frame: *Draw region* (R) over the part one source shows and tag it
   *real*, *beamng*, *rubble* or *other* (an image with a real test half and a BeamNG half
   gets two regions), or *Whole image*.
5. Choose the region's **structure** (vehicle side, top, three-quarter, a truss…) and
   which side its front is on.
6. Trace: the element list walks the schema in order (guided mode, G). Click to place a
   keypoint; a circle is its centre then its rim; a polyline is clicks in order, Enter
   when done. Drag to adjust, alt-click to delete, shift-click to insert, V to flag
   *blurred* or *hidden/guessed*. Zoom with the wheel, pan with space- or right-drag, 0 fits.
   ⌘Z / ⇧⌘Z undo and redo. Every change saves at once.
7. *Copy from previous frame* (C) starts a frame from the shot's previous tracing.
8. **Link**: link the shot to a Rubble scenario (also in the library), and the frame to a
   moment (automatic from its label unless chosen). Give the image **its own camera** by
   hand (eye, target, up, field of view; or paste one as JSON) and Rubble's outline is
   projected through it; the annotate view's *Rubble ghost* draws it over the photo.
9. **Compare**: every source (each traced region and Rubble) in one normalised frame, each
   against its own *before* frame (the earliest traced frame of the shot and view, or the
   one you choose), and the shape numbers: hood fold height, chord, creases, wheel setback
   and drop, windscreen angle, tilts. The annotate view shows the same table live.
10. Commit `engine/docs/crash-reference/studio/` (points and text only); figures for docs:
    `python3 engine/tools/crash-reference/outlines.py all S01 S02` and `sheet.py`.

## Data model

One location, committed, points and text only: `engine/docs/crash-reference/studio/`
(JSON Schema files in `schema/`; TypeScript types in `shared/model.ts`; `npm run
check-data` asserts no image or pixel data is in it). Images live only in `library/`.

- **Shot** (`shots/<id>.json`): one crash run seen in footage. `title`, `source` (`video`
  with `url`, or `screenshots`), `rubble` (the scenario setup: `scenario`, `contact`,
  `run`, or null), `before_frame` (or null: automatic), `notes`.
- **Frame** (`frames/<id>.json`): an image of the library (`image.file`, size, sha256) and
  what is known about it: `shot`, `moment` (`label`, `after_impact_ms` if known), `time`
  (`video_s`), `view` (side, top, three-quarter…), `notes`, `rubble` (the scenario `moment`
  and named `camera`, null for automatic), `camera` (its own hand-set camera: `eye`,
  `target`, `up`, `fov_y_deg`, and the `region` whose crop is its image; null until set),
  `before_frame`, and `regions`.
- **Region**: a crop rectangle `[x0, y0, x1, y1]` (normalised image coordinates) with a
  `source` tag (real, beamng, rubble, other), a label and its **Tracing**: `instances`,
  `traced_by`, `approximate`, `note`.
- **Instance**: one structure in a region: its `schema`, `front` (left or right) and
  `elements` by name: a keypoint `{p}`, a polyline `{pts}`, a circle `{c, rim}`, each with
  `vis`: clear, blurred or hidden (hidden includes guessed). Points are in the region's
  normalised coordinates (0..1 of its width and height, from its top-left corner).
- **StructureSchema** (`structures/<id>.json`), data, not code: the elements in tracing
  order (`name`, `type`, `label`, `colour`, `hint`, and the `rubble` rule that reads it off
  Rubble's part outlines), the `frames` that normalise a shape (an origin anchor and a unit;
  the first that every source has is used), and the `measures`. Shipped: `vehicle-side`,
  `vehicle-top`, `vehicle-three-quarter` and the generic `truss-side`. A new structure kind
  is a new JSON file; only a new kind of measure or Rubble rule needs Rust.

## Where the maths lives

In Rust, once: `engine/crates/engine-softbody/src/crash_reference/shape/` (tracings into
structure axes, frames, normalisation, measures, Rubble's elements from its outlines, the
reference-moment rule), with unit tests on synthetic shapes. The example `reference_shape`
serves it as JSON lines; the studio's server keeps one process and passes requests through,
and `engine/tools/crash-reference/outlines.py` calls the same binary. The page and the
server compute no geometry: TypeScript only draws (image and screen transforms). Rubble's
outlines come from `crash_outlines` (`--cameras FILE` adds each frame's own camera).

## For agents: three routes to the same tools

The tool list is `shared/tools.ts` (name, title, description, JSON input schema, read-only
or not). The same functions are reachable three ways:

- **WebMCP.** Each tool is registered with `document.modelContext.registerTool({ name,
  title, description, inputSchema, annotations: { readOnlyHint }, execute })` (spec
  sections followed: "4.1 Extensions to Document", "4.2 ModelContext Interface", "4.2.1
  ModelContextTool Dictionary", "4.2.3 ModelContextRegisterToolOptions Dictionary": the
  registration's `signal` unregisters on page hide). `execute` returns plain JSON. Where
  the browser has no `document.modelContext` the page installs the W3C polyfill
  (`@w3c/webmcp-polyfill` 0.2.0, pinned, bundled by Vite: no network at run time), which
  also gives `document.modelContext.getTools()` and `executeTool(tool, input)` for testing.
  `window.studio.webmcp` says which: `native` or `polyfill`. Sources:
  <https://webmachinelearning.github.io/webmcp/> (Draft Community Group Report, 24 November
  2026) and <https://github.com/webmachinelearning/webmcp>.
- **`window.studio`** (any script driving the page, e.g. over the DevTools protocol):
  `studio.tools` (the list), `await studio.call(name, input)` or `await studio.<name>(input)`.
  Page tools (`open_view`, `open_frame`, `select_element`, `set_zoom`, `describe_view`)
  change what the page shows.
- **HTTP** (no page needed): `POST /api/tool/<name>` with the input as `application/json`
  returns `{ok, result}` or `{ok: false, error}`; `GET /api/tools`, `GET /api/state`,
  `POST /api/upload?name=&shot=` with raw image bytes. Server tools only.

A typical agent session: `list_untraced` → `open_frame` → `get_frame` (regions, image
size) → `mark_region` → `set_region_schema` → `list_schemas` (elements and hints) →
`set_keypoint` / `add_polyline_point` / `set_visibility` → `link_scenario`, `link_frame`,
`set_frame_camera` → `get_rubble_outline` → `get_comparison`. `describe_view` says what the
page shows and which element the guided mode asks for next.

## Checks

- `npm run typecheck`, `npm run build`
- `npm test`: unit tests of the server (path safety, image headers, the data check, the
  editing tools on a scratch folder)
- `npm run check-data`: the committed data holds points and text only
- `npm run smoke`: headless Chrome (no GPU) drives the page through `window.studio`,
  WebMCP and HTTP on generated test images (a drawn car before and after a crash) with a
  scratch data folder, checks the table against values worked by hand, and saves
  screenshots of every view to `engine/out/reference-studio/screenshots/generated/`
  (`--real`: of the real library, which shows third-party footage: not for sharing)
- Rust: `cargo test --release -p engine-softbody --lib crash_reference::shape`

Safety: the server binds 127.0.0.1, refuses other Host headers, reads and writes only its
data folder, library and cache (ids checked, paths resolved and kept inside), and accepts
writes only as JSON or octet-stream bodies (a cross-site form cannot send them).
