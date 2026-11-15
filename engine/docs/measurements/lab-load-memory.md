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
   2.15 × rskin bytes, largest buffer ≈ 2.16 × rskin bytes, from the table above)
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

## Second pass (2026-11-15): the phone's "GPU device was lost"

The iPhone then lost its WebGPU device while loading (the banner's "The GPU device
was lost (no reason given)", 0 steps taken). The four lossless items below were
done on `perf/gen-car-phone-memory`; same method, same car, served from a local
build (`build-web.sh`'s wgsl_export, world wasm and assembly; `__mods` linked to
the artifacts' folder). MiB, desktop Chrome on an M-series Mac:

| | mountain gen, before | after | shoot gen surface, before | after | authored sedan, before | after |
|---|---:|---:|---:|---:|---:|---:|
| **Renderer footprint, peak** | **1170** | **919** | **1235** | **969** | 407 | 435 |
| Renderer footprint, settled | 1136 | 719 | 1134 | 728 | 400 | 397 |
| JS heap, peak at a step's end | 310 | 241 | 310 | 239 | 90 | 89 |
| JS heap, settled | 177 | 5 | 178 | 6 | 68 | 69 |
| wasm memory | 529 | 419 | 529 | 419 | 150 | 149 |
| GPU buffers, peak | 377 | 260 | 370 | 259 | 144 | 143 |
| `showcase-binds` | 255 | 136 | 255 | 136 | 23 | 23 |

(The authored sedan's peak moves between runs by about 30 MiB; its settled numbers
are unchanged.)

1. **The bind buffer at 8 B an entry** (1332af91). When the wasm sends the weights
   as grid steps (`wtQ`) on the grids the shader decodes (2⁻¹³, 2⁻⁹), a weighted
   vertex's entries are packed end to end: node u16 and A i16 in one word, ∂A/∂T
   and ∂A/∂B i16 in the next (flags bit 27). The shader decodes a step count s by
   lowering f32(s)'s exponent (integer arithmetic): the same f32 bits, and no float
   operation a compiler could fold into the sums. Other weights keep the f32 layout.
   267,358,144 → 142,986,368 bytes.
2. **The visual not kept after its upload** (8926eac3). The renderer's per-level
   Params closure kept `addCar`'s whole scope, the visual among it, while the car
   was drawn; it is built outside that scope now. The lab drops a mod car's visuals
   once uploaded and builds them again for an in-place rebuild, the cutaway or a
   second paint (the same files and wasm: the same bytes).
3. **The model files streamed into the wasm heap** (80e47910): `HeldBytes`, written
   chunk by chunk as they download (`loadMod(…, { held: true })`), read there by
   `jbeamVisualSkinnedHeld`, freed by `releaseModels`.
4. **The weights as steps in Rust** (80e47910): `ShowcaseVisual::wt` is a
   `flexskin::WtEntries`, u16 + 3 × i16 while every entry is exact, else u32 + f32;
   `w(k)` gives back the pushed f32 bits, so the CPU users read the same values.
   Native wasm-heap peak (`skinned_visual_memory`): 476 → 369 MB.

Also: the device-lost banner names the load step last shown (or "while running, N
steps taken") and the last memory reading (JS heap where given, wasm heap, the car's
estimated footprint, device memory), and a loss while the lab is still being built on
the device is reported too. The fit check's estimate is the measured one (tab
450 MB + 3.7 × model bytes; bind buffer 1.16 × rskin), and on iOS (`isIOS()`: the
UA, or a Mac UA with touch) a car estimated past 1 GB of tab is warned about with a
"Load it anyway" button. That 1 GB is a judgement from one report (0.42 GB ran,
1.2 GB did not), not a known WebKit limit; the generated sedan now estimates 0.97 GB
and is not warned about.

Proofs (all on the generated sedan, 490,997 vertices, 13,945,318 entries):

* GPU: `engine/tools/vehicle-lab/skin-gpu-identity.mjs` skins the same deformed
  node positions with the old renderer and the new one (headless Chrome, Metal):
  `showcase-out` (15,711,904 bytes, every vertex moved off its rest) is
  byte-identical.
* CPU mirror: `skin-upload-identity.mjs` hashes the entries as `wEntry` decodes
  them from `showcase-binds`: the same old and new. With the new renderer, every
  buffer's hash is the same from the old wasm, the new wasm, and the new wasm
  reading `HeldBytes` (`--held`).
* Tests: `tests/showcase_car_memory.test.mjs` (the step layout decodes to the f32
  layout's entries, in bounded pieces; another grid keeps the f32 bytes; addCar
  keeps nothing of the visual), `beamng_mod_models` (held files, retry,
  truncation), `beamng_mod_footprint` (iOS), `flexskin::wt_entries_give_back_what_was_pushed`,
  `gen3_rig` (the in-place sidecar binds the same visual).

What is left of the load peak: the visual as JS arrays while it is uploaded (about
170 MB, transient now) and the bind staging. Packing the bind buffer in Rust and
uploading it from the wasm heap would remove the first; not done (a change of where
the renderer's packing lives).

## Not done, for decision or later

Lossy (would need the user's decision; not measured beyond this): f16 weights
would halve the steps again but A spans ±4, where f16's step is 2⁻⁹ (16× the
sidecar's grid); quantised detail offsets likewise.

## Phone estimate (unverified)

The tab's peak went from 2.2 GB to 1.2 GB (the authored sedan, which runs on the
phone: 0.42 GB). WebKit kills a tab's WebContent process near a limit that depends
on the device's memory (commonly reported as roughly 1.5–2 GB on 6 GB iPhones,
less on 4 GB ones), and JavaScriptCore's heap is not V8's, so whether 1.2 GB fits
the phone in question is not known until it is tried there. The kernels' shader
compile (WebKit's GPU process) was not measured either: if the phone still closes
the tab at the compile step with these numbers, that points at the GPU process
rather than the tab's memory.
