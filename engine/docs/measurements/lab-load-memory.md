# Vehicle lab: the generated sedan's load memory (2026-11-14)

An iPhone (Chrome on iOS, so WebKit, WebGPU on) closed the tab ("Can't open this
page") at the step "compiling the Generated sedan's kernels and building the lab…"
of `vehicle-lab.html?mode=mountain&car=gen`; the same page with `car=sedan` (the
small authored car) loads and runs there. These are the page's memory measurements
in desktop Chrome and what was cut, losslessly. Nothing here was run on a phone.

## Method

* `engine/tools/vehicle-lab/lab-memory.mjs`: headless Chrome (Apple M-series Mac,
  Metal) on the page served from `engine/web-dist`. Injected before the page's
  scripts: GPU bytes summed at `createBuffer` / `createTexture` (less `destroy()`),
  by label; the wasm modules' `memory`. Per load step (the page's progress text):
  `performance.memory.usedJSHeapSize` (precise), the wasm linear memory, the GPU
  bytes. Through the whole load, every 150 ms: the renderer and GPU processes'
  physical footprint (`footprint`, the number iOS's jetsam limits a process by);
  the peak is the largest sample. Settled: 8 s after ready, after a forced GC. The
  page's own breakdown of what it keeps (`window.__mem()`).
* `engine-softbody/examples/skinned_visual_memory.rs`: the wasm loader's Rust
  steps natively under a counting allocator (live and peak bytes per stage). The
  wasm heap never shrinks, so its size after the load is this peak (plus
  fragmentation: 1000 MB native, 1077 MB grown in the browser).
* The car: `sedan_new` (`~/Development/rubble-artifacts/sedan-live/mods`):
  `sedan.glb` 16 MB, `sedan.rskin` 124 MB (version 2), 490,997 vertices, 465,998
  triangles, 13,945,318 skin entries (28.4 a vertex, at most 67), 548 skin nodes.

## Before and after (MiB; mountain, one car)

| | before | JS fixes (cc8f89e3, 16aaaa04) | after (+ cf30b0b9, aabf452a) | authored sedan (after) |
|---|---:|---:|---:|---:|
| **Renderer footprint, peak** | **2247** | 1857 | **1227** | 415 |
| Renderer footprint, settled | 2185 | 1796 | 1113 | 385 |
| JS heap, peak at a step's end | 719 | 416 | 310 | 111 |
| JS heap, settled | 674 | 284 | 178 | 69 |
| wasm memory (never shrinks) | 1118 | 1118 | 515 | 140 |
| GPU buffers, settled | 375 | 374 | 375 | 150 |
| GPU textures | 38.5 | 38.5 | 38.5 | 38.5 |
| GPU process footprint, peak (macOS) | 3905 | 3906 | 3913 | 3316 |

The authored sedan's renderer peak was 422 MB before (unchanged within noise).

### By owner

JS heap, settled (the page's `__mem()` and the uploads' retention test):

| | before | after |
|---|---:|---:|
| glb + rskin bytes kept on the car's entry | 134 | 0 (released; fetched again for another paint) |
| the skin's bind data (255 MB) kept by a closure past its upload | 255 | 0 |
| the visual (kept for in-place rebuilds and the cutaway) | 279 | 173 (the weights as u16 nodes and i16 grid steps, below) |
| jbeam + pc text | 1.5 | 1.5 |

The visual's weights: `wtNode` u32 (53 MB) and `wtW` 3 f32 (160 MB) before;
`wtNode16` u16 (27 MB) and `wtQ` 3 i16 (80 MB) after, the same values exactly.

JS heap, transient during the load: the 255 MB bind array the renderer packed is
now staged 8 MB at a time.

wasm heap high-water (native counting allocator, MB; cumulative live at each stage):

| stage | before | after |
|---|---:|---:|
| glb + rskin bytes copied in (wasm-bindgen) | 134 | 134 |
| desc, rows, solver spawn | 137 | 137 |
| glb parsed (and as flexmesh meshes) | 186 | 186 |
| rskin decoded (`Skin::decode`: a Vec a vertex, 16 B an entry) | 510 | read in place (`SkinBytes`): 186 |
| visual built (peak includes the weight lists' doubling) | 841 (peak 894) | 473 (peak 476) |
| `visual_js`'s flattened temporaries (largest: wtW, 160 MB) | 1000 | none (copied from the lists in place) |

GPU (settled, the lab's own buffers over 1 MB): `showcase-binds` 255 (64 B a
vertex and a 64 B slot of 4 entries), textures 38.5 (MSAA colour and depth at
1280×720 and the 1024² car layers), `showcase-attrs` 18.7, `showcase-out` 15,
the World's buffers (`b33` 16, `readRanges` 10–11, `b27` 8, four of 4) and index
lists (5.3 + 5.3 + 3.8 + 3.8). Unchanged by this work.

The GPU process's footprint on macOS is about 3.3 GB for either car (the authored
one included) and is not what differs between them; on iOS the GPU process is
WebKit's own and was not measured.

## What changed (all lossless)

1. Renderer (cc8f89e3): `addCar` packed the bind data in the level's scope, which
   its `params` closure kept alive: the 255 MB array outlived its upload. Packed in
   a function of its own. Uploads pinned bit for bit (`tests/showcase_car_memory.test.mjs`).
2. Lab (16aaaa04): the glb and rskin bytes are dropped once the visual is built
   (`beamng-mod.mjs releaseModels`); a second paint fetches them again
   (`modModels`, the HTTP cache); `modVisual` refuses a released vehicle rather
   than build its Collada fallback.
3. wasm (cf30b0b9): the rskin read in place (`flexskin::SkinBytes`, decode's bits
   and errors, versions 1 and 2); the weight lists reserved at the entry count; the
   flat lists copied to JS from where they lie; the weights sent as the sidecar's
   grid steps when `flexskin::steps` is exact for every entry (else f32 as before).
4. Renderer (aabf452a): the bind buffer packed and written 8 MB at a time.
5. Guard (9f37b1f8): before downloading, the car's footprint estimated from its
   files' sizes (`modFootprint`: tab ≈ 450 MB + 6.0 × model bytes, GPU ≈ 170 MB +
   2.15 × rskin bytes, largest buffer ≈ 2.16 × rskin bytes, from the table above;
   1.26 and 1.28 since the bind buffer's entries are grid steps, §8.30)
   against `navigator.deviceMemory` (half of it for a tab) and the adapter's
   `maxStorageBufferBindingSize` / `maxBufferSize`. Refused: the error line and its
   banner give the numbers and a "Load it anyway" button. Checked with
   `lab-memory.mjs --deviceMemory 2`. Safari and iOS give no `deviceMemory`: there
   only the GPU limits are judged.

## Identity

`engine/tools/vehicle-lab/skin-upload-identity.mjs` builds the car's visual with a
World wasm and packs it with a renderer on a stand-in device (node), hashing every
buffer's final contents. Old (5b5a527f's wasm, built apart, and renderer) and new
give the same bytes for all 13 buffers (`showcase-binds` 267,358,144 bytes,
`showcase-attrs`, `-out`, `-index`, `-floor-index`, `-topology`, `-params`, …).
Natively, `gen3_rig::the_sidecar_read_in_place_binds_the_same_visual` checks the
in-place skin binds the same visual (its `Debug`, every float) as the decoded one.

## Not done, for decision or later

Lossless, not done (each needs a design choice or touches drawing code):

* Done since (`vehicle-perf-first-principles.md` §8.30): the bind buffer at 8 B an
  entry (u16 node, 3 i16 steps; the shader scales by powers of two, exact):
  `showcase-binds` 267 → 157 MB a car on the GPU, and half the bytes the skinning pass
  reads a frame. `modFootprint`'s GPU figures follow.
* The visual's weights held as steps in Rust (`ShowcaseVisual`): the wasm
  high-water 515 → about 405 MB. Its CPU users (`vehicle_compare`, `skin::deform`,
  lint) read f32 today.
* The rskin streamed into the wasm heap as it downloads (no JS copy, no second
  copy): about 124 MB off the JS peak during the bind.
* Not keeping the visual in JS after its upload (173 MB): in-place rebuilds and the
  cutaway would rebuild it (about 1 s) or keep it in wasm.

Lossy (would need the user's decision; not measured beyond this): f16 weights
would halve the steps again but A spans ±4, where f16's step is 2⁻⁹ (16× the
sidecar's grid); quantised detail offsets likewise. Not proposed further: the
lossless items above are larger.

## Phone estimate (unverified)

The tab's peak went from 2.2 GB to 1.2 GB (the authored sedan, which runs on the
phone: 0.42 GB). WebKit kills a tab's WebContent process near a limit that depends
on the device's memory (commonly reported as roughly 1.5–2 GB on 6 GB iPhones,
less on 4 GB ones), and JavaScriptCore's heap is not V8's, so whether 1.2 GB fits
the phone in question is not known until it is tried there. The kernels' shader
compile (WebKit's GPU process) was not measured either: if the phone still closes
the tab at the compile step with these numbers, that points at the GPU process
rather than the tab's memory.

## The iPhone's `GPUOutOfMemoryError` (2026-11-17)

The same iPhone, with the cuts above, loaded `vehicle-lab.html?mode=mountain&car=gen`
further and stopped with `gpu: [object GPUOutOfMemoryError]`: WebKit's out-of-memory
error had an empty message, and the page printed the object. Nothing here was run on
a phone; the iOS figures are WebKit's code and a public survey, the rest is desktop
Chrome (`lab-memory.mjs`, M3 Max) and Chrome with iOS's limits forced (`--ios`).

### What iOS allows

* One buffer: WebKit (`HardwareCapabilities.mm`) sets `maxBufferSize` and
  `maxStorageBufferBindingSize` both to max(256 MiB, min(1 GiB, Metal's
  `maxBufferLength` / 3)). Every iOS device reports at least 268,435,456 bytes, 96 %
  at least 644 MB, 82 % 1 GiB (web3dsurvey). The car's largest buffer, the skin's
  bind records, was 267,358,144 bytes on main (1 MB under the least limit) and is
  157 MB since #116: **no single buffer is over a limit**, and no solver buffer is
  near one (the largest, `b33`, 16 MB).
* All of them together: WebKit raises `GPUOutOfMemoryError` ("Allocation failure.",
  which reached the page empty) when Metal's `newBufferWithLength` (or a texture's
  allocation) returns nil. WebKit charges Metal's allocations to the tab's WebContent
  process (its memory attribution), so they count, with the tab's JS and wasm heaps,
  against the limit iOS refuses or kills a process at. **What pushed it over is the
  total**: the tab's own memory (1.0 GB settled for the full car, 528 MB of it a wasm
  heap that never shrinks) plus the GPU's 312 MB.

### Per car, 1 and 8 cars (MB; mountain; desktop Chrome)

| | 1 car, before | 1 car, after | 8 cars, before | 8 cars, after |
|---|---:|---:|---:|---:|
| GPU buffers, settled | 274 | 274 | 1741 | 690 |
| of which `showcase-binds` (the skin, 150 MiB a car) | 150 | 150 | 1200 | 150 (shared) |
| `showcase-attrs`, `-out` and index lists (per car) | 52 | 52 | 416 | 416 |
| GPU textures | 38.5 | 38.5 | 21.6 | 21.6 |
| wasm heap (the page's) | 528 | 77 | 532 | 532 |
| JS heap, settled | 178 | 178 | 1388 | 1389 |
| renderer footprint, settled | 1034 | 584 | 2480 | 2317 |
| renderer footprint, peak | 1230 | 947 | 2906 | 2568 |

(8 cars, after: measured before the worker, with the shared bind buffer only.) The
World's own buffers are ~100 MB whatever the car (`b33` 16, `readRanges` 14–44, `b27`
8, …). 8 cars before: the same bind records packed eight times, and eight 173 MB
visuals in JS (one a paint; not changed here).

### What changed (lossless; the desktop default draws the same pixels)

1. The error line (`gpu-errors.mjs`): the error's kind, the browser's text or that
   it gave none, the phase (the renderers' and the cars' uploads run in their own
   error scopes), the buffers and textures created then or last (labels, sizes),
   what is live on the device, and its `maxBufferSize` / `maxStorageBufferBindingSize`.
2. The visual is built in a worker of its own (`visual-worker.mjs`, the same wasm and
   the same call): the ~450 MB of wasm heap its build grows go with the worker. The
   page's wasm heap 528 → 77 MB; the tab settled 1034 → 584 MB, its peak 1230 → 947.
3. Cars of one model share the skin's bind buffer (`showcase-car-render.mjs`
   `bindsLike`): a paint and a car index do not enter the bind records
   (`skin-upload-identity.mjs --paint --vehicle`: the same hash; only `showcase-attrs`
   and `-params` differ). 8 cars: 1741 → 690 MB.

Before / after, desktop default (`vehicle-lab-record.mjs`, the generated sedan at rest
and in the 56 km/h offset crash): the same stats JSON, and the same stills byte for
byte but one (one frame grab 1.5 s into the crash: 5 levels at most, mean 0.0001).

### The lighter car detail (the generated sedan's own, for small devices)

`vehicle_gen_rig` also writes `sedan.lite.glb` / `sedan.lite.rskin`: the same design
built with its body meshed at a 35 mm target edge instead of 20 mm (`LITE_EDGE`; the
longest-edge cap scales with it and is unchanged at 20 mm), every part under the
same name, bound by the same binder to the same lattice. The JBeam and `.pc` are the
same bytes (`rig_export.rs`), the lab's damage units are the same, and the crash and
rest runs' physics readouts are the same number for number. 280,629 vertices and
6.5 M skin entries (57 % and 47 % of the art's); the files 68 MB instead of 140.
Coarser targets drop parts (the door cards at 40 mm), so 35 mm is the coarsest
tried that keeps every mesh.

The lab's "Car detail" control: auto (full unless the device cannot hold it), full,
lighter. Auto picks the lighter one where the footprint guard refuses the full one,
on an iPhone or iPad's WebKit (no `deviceMemory`, so the tab's budget cannot be read,
and the full car is what failed there), or after the full one's build ran out of GPU
memory on this device (rebuilt once, lighter, saying why). With the lighter detail
chosen that way the lab runs one car, and says so.

| mountain, 1 car (MB) | full, after | lighter, `--ios` |
|---|---:|---:|
| renderer footprint, peak / settled | 947 / 584 | 428 / 409 |
| GPU buffers + textures | 274 + 38.5 | 175 + 38.5 |
| `showcase-binds` | 150 | 74 |
| visual kept in JS | 173 | 88 |

With `--ios` (an iPhone's user agent, no `deviceMemory`, the adapter's two buffer
limits at 256 MiB and the device created with them) the lab picks the lighter detail,
loads in 2.9 s and runs (17 → 549 World steps in 8 s). The authored sedan, which
runs on that iPhone, measured 415 MB peak and 150 MB GPU. Whether the phone holds
the lighter generated sedan is the owner's check on the phone.
