# The explicit frame: one workgroup a car or wide passes? Design decided by runtime (2026-11-15, interim)

Owner's order: physical accuracy (bit-identical results, never traded), then the highest
real-time runtime, then everything else (compile time, CI minutes, tidiness). So this
document decides pass by pass on **measured runtime**, native and Chrome, and reports
compile time and variant count as secondary costs of whichever wins.

Machine: M3 Max (40 GPU cores), native wgpu on Metal and headless Chrome (WebGPU on the
same GPU). Branch `integrate/perf-all` at `c72f501f`. Every number below is in a log in
`engine/out/` named in its row.

## 1. Runtime table (lead)

### 1.1 The same work, two structures (the structural question, isolated)

`examples/wide_proto.rs` (native) and `tools/wide-frame/proto.html` (Chrome, the same
WGSL): the generated sedan's real topology (`sedan_new`: 804 nodes, 6,427 beams, its
incidence lists), a 2 kHz substep as **a beam pass** (spring-damper force a beam) and **a
node pass** (each node gathers its incident beams' forces in incidence order, gravity,
symplectic Euler). Wide: one thread an element over every car, two dependent dispatches a
substep, 34 substeps in one compute pass. Workgroup: the same two phases in one workgroup
a car, a barrier after each, 34 substeps in one dispatch. Wall time to completion, median
of 30 frames, ms per 34-substep frame. Logs `wide_proto_0617.log`,
`wide_proto_chrome_0622.log` (Chrome's timer rounds to 0.1 ms).

| Cars apart | Wide, native | Wide, Chrome | Workgroup 512 lanes, native | Workgroup 512, Chrome | Workgroup 256, native | Workgroup 256, Chrome | Winner |
|---|---|---|---|---|---|---|---|
| 1 | 0.74 (16.6 µs a substep) | 0.70 (15.2 µs) | **0.62** | **0.60** | 1.01 | 0.90 | workgroup 512 |
| 8 | 0.89 (21.0 µs) | 0.90 (21.2 µs) | **0.64** | **0.60** | 1.02 | 0.90 | workgroup 512 |
| 32 | 1.38 (35.4 µs) | 1.40 (36.4 µs) | **0.65** | **0.60** | 1.05 | 0.90 | workgroup 512 |
| 128 | 5.12 (145 µs) | 4.90 (139 µs) | 2.95 | **2.30** | **2.72** | 2.60 | workgroup (256 native, 512 Chrome) |
| 256 | 11.6 (333 µs) | — | — | — | **8.93** | — | workgroup 256 |

**The one-workgroup-a-car structure wins at every fleet size, natively and in Chrome, by
1.2× (1 car) to 2.1× (32 cars).** Two dependent dispatches cost ≈15–17 µs a substep for one
car (each dispatch's latency plus its loads), while the same two phases in one workgroup
cost ≈18 µs a substep including the barriers, and the workgroup's cost stays flat to 32
cars (they run on separate cores) where the wide passes' grows. Past ≈100 cars both are
throughput-bound and the workgroup form still wins (fewer global round trips: a car's
nodes stay in its core's cache across phases).

### 1.2 The existing wide passes against the in-workgroup frame (the real solver)

`examples/wide_vs_workgroup.rs`, the repo's sedan (367 nodes, 2,170 beams; the generated
sedan is not wide-capable: driven wheels), self-collision on, no coupling, the frame alone.
Log `wide_vs_wg_0610.log`; and `explicit_vehicles_bench` "frame alone", log
`wide_bench_repo_0603.log`.

| Cars apart | In-workgroup frame | Wide passes | Ratio |
|---|---|---|---|
| 1 | 0.91–1.22 | 6.73–7.21 | 6–7× slower wide |
| 8 | 1.26 | 7.38 | 6× |
| 32 | 1.17–1.47 | 8.05 | 5.5–7× |

The existing wide passes (`explicit_beams_wide_*`, kept "for vehicles too big for one
workgroup") are far slower: their self-collision is a grid of fixed bucket slots per step
with no reused pair list, plus per-pass record loads. They are not a candidate.

### 1.3 Where one generated sedan's frame goes today (the real solver, late mountain crash)

`vehicle_lab --measure mountain-perf`, page-default mountain, surface contact off,
self-collision split on (`RUBBLE_SELF_SPLIT=1`), ms per 34-step frame, 20 frames from one
snapshot each way, wall time; skip switches are time-only (`set_self_split_skip`). Native
log `profile_page_2.log`; Chrome (the page's GPU segment "explicit self split frame") logs
`chrome_seg_0.log`/`chrome_seg_1.log` (before the walk change), `chrome_skip_1.log`.

| t (s) | Split off native | Split on native | step dispatches only | walks | pairs | builds a frame | Chrome split off | Chrome split on |
|---|---|---|---|---|---|---|---|---|
| 9 | 10.8 | 7.3 | 4.5 | 0.1 | 2.6 | 4.8 | 8.8 | 7.4 |
| 20 | 11.7 | 8.0 | 4.6 | 0.2 | 2.7 | 4.5 | 9.8 | 7.9 |
| 36 | 14.6 | 8.1 | 4.7 | 0.2 | 3.3 | 6.3 | 11.8 | 8.1 |
| 40 | 12.7 | 7.8 | 4.9 | 0.1 | 2.8 | 3.0 | 12.0 | 8.7 |
| 44 | 11.3 | 7.9 | 5.0 | 0.1 | 2.9 | 1.3 | 11.1 | 8.0 |

(Chrome's columns are at the nearest sampled sim time of two separate runs; the late
crash's state differs a little between runs.)

Inside the pairs dispatches (ablation builds, t 36/44): the ~10k listed pairs' box tests
cost about nothing; ≈2.1 ms is the anti-clip hit test itself (`selfClip`'s two
`triangleHit`s, one on the previous step's geometry) on ≈4.5k pairs a substep for 47–75
hits.

**The step dispatches' ≈4.5–5 ms are now the largest cost, and §1.1 says why it is not
the structure:** the same beam and node phases cost 0.6 ms a frame in one workgroup. The
other ≈4 ms are the solver's content inside the substep: ≈9–21 phases a substep (coupling
4, drive 1, grid count, scan and scatter 5 on builds, self-collision's apply, slides,
surface), each phase as long as its lane's longest chain (`vehicle-perf-first-principles.md`
§2.1–2.2): beams with yield, break groups, dampers and extension records; nodes with
ground, boxes and the coupling's contacts. The floor for one car is 0.3–0.5 ms
(the audit, §1.3).

### 1.4 Decision per pass

| Pass | Structure | Why (measured) |
|---|---|---|
| Beams, nodes, coupling's per-car step, drive, aero | **one workgroup a car** (as now) | §1.1: 1.2–2.1× faster than wide at 1–32 cars, both platforms; never slower to 256 |
| Self-collision search and tests (listed pairs, the walk) | **across the GPU** (the self split, built) | 3–6.5 ms a frame saved natively, 1.4–3.7 ms in Chrome (§1.3) |
| Vehicle–vehicle contact (groups' tests and levels) | **across the GPU** (the group split) | contact-groups-design.md; a group of k cars on one workgroup puts k cars on one core |
| The per-car step's remaining ≈4 ms | **fewer phases and shorter chains inside the workgroup** | the cost is content, not structure (§1.3 against §1.1) |

So the answer to "is the frame-sized kernel the wrong way to write this": **for runtime,
the one-workgroup-a-car frame is the right structure for a car's own phases** (no trick:
a barrier is cheaper than a dependent dispatch, and a car's arrays are too small to fill
the GPU). The pairwise searches are already across the GPU. The large-kernel problem is a
*compile* problem (§2), which is solvable without changing that structure.

## 2. Compile time and module count (secondary costs of the winner)

Measured (`examples/contact_compile_time.rs`, native Metal; the page's loading card in
Chrome, log `chromeB_0545.log`):

| Module | WGSL | Native cold | Native warm (driver cache) | Chrome, fresh profile (Metal cache warm) |
|---|---|---|---|---|
| frame (`explicit_beams_wide_nosurf`, 512 lanes) | 477 KB | 8.5 s | 0.0–1.0 s | 0.2 s |
| self split step | 469 KB | 8.4–9.6 s | 0.0 s | 1.1 s |
| self split walk / pairs | 59 / 49 KB | 0.3 / 0.2 s | 0.0 | 0.1 / 0.5 s |
| self split vehicles' step with contact | 530 KB | 29.8 s | 0.0 | — |
| split group members | 778 KB | 50–61 s | 0.1 s | — |
| prototype beam pass / node pass | 0.8 / 1.1 KB | 0.01–0.05 s | — | 0.06 / 0.11 s |

A page load compiles only its options' modules (one car, surface off: the frame plus
three deferred split modules; surface on: the segment frame plus nine small surface
pipelines); first load of a new kernel version in Chrome took 17.6–25.9 s here, later
loads 2.3–3.2 s to ready.

The build compiles **474 WGSL modules** (`OUT_DIR/wgsl`), of which **104 are shipped
explicit frame-family modules (42.8 MB of WGSL)** and **84 are test-only explicit modules**
(ablation 12, deterministic-hits 12, frozen baseline 39, frozen reference 15, self split
ablation 6: ≈20 MB). The shipped ones multiply four axes: lanes {256, 512} × aero {off, on}
× surface {full, without surface (`_nosurf`), without coupling (`_nocpl`), segments
(`_seg`)} × kind {frame, solo, contact, contact-big, split member/group/tests, self split
step/walk/pairs, …}.

What can go without runtime cost (each to be **measured per switch before adopting**: a
kernel that carries a disabled feature's code can lose occupancy to registers):

| Change | Modules removed | Runtime risk | Status |
|---|---|---|---|
| Test-only modules (ablation, det, baseline, reference) built only for the test build (a `test-kernels` feature the crate's own dev-dependency turns on), not for release/web/CI's non-test jobs | 84 of 474 from every non-test build | none (not shipped) | **done** (with the explicit modules compiled in parallel, which they were not): release lib build script 113.8 → 41.9 s, test build 46.1 s, 72.9 → 51.6 MB of WGSL, world wasm 63.0 → 61.9 MB |
| Lanes 256/512 as a WGSL `override` constant: one module, two pipelines | ≈half the shipped modules' *translation* (not their pipeline compiles) | none expected (the same code after specialisation) | to measure: Metal compiles each specialisation; Tint/naga translate once |
| `_nocpl` (frame without the coupling's proxy paths: a smaller kernel for browsers) | ≈20 | compile only: it exists for browser compile time | keep while it shortens the page's first load |
| aero as data (no wings: empty ranges) | ≈half | **must measure**: aero code in a non-aero car's kernel | to measure |
| Surface pass as its own pipelines everywhere (already so with segments) instead of `_nosurf` twins | ≈20 | the segment split costs a dispatch per substep (≈2 µs × 34) | to measure |

## 3. What this design does next (after the owner has seen this table)

1. **Per-car step: fewer phases, shorter chains, inside the workgroup** (audit #5, #6): the
   coupling's folded step is done (2 barriers, not 4); next the grid count / scan / scatter
   (5 syncs on builds), self-collision's apply rounds, and the beam phase's records sorted by
   kind (`beam-records.md`, measured separately). Bitwise against today's frame per change.
2. **Anti-clip hit test** (≈2.1–2.3 ms). Split measured (ablation builds of the pairs
   module, t 36/44, log `profile_clip_0627.log`): without the hit test past the box 5.1 ms
   (from 7.4); without the crossing test (the previous step's frame) unchanged; without
   `clipRecord`'s atomics unchanged (≤0.2 ms). So it is the first `triangleHit` and its
   loads on ≈4.5k pairs a substep: compute and latency, not contention. Lossless options
   are scheduling only (the triangle's geometry once per triangle per substep,
   triangle-major order), with a bitwise A/B.
3. **Surface contact with the self split**: the page's other mode (surface on) runs the
   frame in segments without the split; cutting the self-collision passes around the
   surface pass gives it the same saving.
4. **Compile**: the test-kernel feature first (certain, no runtime risk); then each
   variant-to-switch row of §2 measured one at a time.

Not measured yet (honest gaps): 32 cars in contact on today's code (the group split's
numbers are in contact-groups-design.md, older); Dawn's indirect-dispatch validation cost
per indirect dispatch; nvcc / CUDA compile times; hundreds of cars with the real solver
(the prototype goes to 256).

## 4. Requirements of the original brief, answered briefly

- (a) Bit-identical: the decision keeps every pass that is already bitwise (per-node sums in
  incidence order inside the car's workgroup; the split passes' order-free effects and
  `selfApply`'s ordered application). Nothing here changes an equation or an order.
- (b) Runtime switches instead of variants: §2's table, each row measured before adoption.
- (c) Dispatches: one car's frame is ≈1 dispatch (in-workgroup) or ≈103 (self split:
  step, walk, pairs a substep; ≈2 µs each natively and in Chrome, 0.1–0.2 ms a frame in
  all); a wide frame would be ≥2 a substep a phase pair (§1.1: 15–17 µs a substep for two).
  Encoding once and replaying is not available in WebGPU (no command-buffer reuse); the
  page's encode of ≈100 dispatches costs ≈0.1 ms of CPU.
- (d) Compile time per kernel: §2 (native cold, Chrome); nvcc not measured.
- (e) Migration: none for the per-car phases (they stay); the split passes are already behind
  switches with bitwise A/B against the monolith.
- (f) Risks: occupancy of small arrays across workgroups is exactly what §1.1 measured (the
  wide form loses); the 2-in-flight readback gate is unchanged; inherently serial passes
  (vehicle-contact levels) stay bounded per group as in contact-groups-design.md.
