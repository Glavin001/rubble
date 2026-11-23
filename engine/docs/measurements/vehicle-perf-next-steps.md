# Vehicle GPU performance: the next steps that can move the frame by multiples (2026-11-23)

This document is written against `origin/main` at ba8355e4 (#152 merged). It asks
whether the per-switch work of `vehicle-perf-first-principles.md` §8 can still reach the
goal, and if not, which larger changes can. Every figure cited here is from main's
docs; nothing new was measured for it. Section numbers (§8.xx) refer to
`vehicle-perf-first-principles.md`; "the floor review" is `gpu-floor-vs-production.md`.

## 1. The answer in short

- **The goal is 3–5× away, and iteration will not close it.** The goal: 20 soft cars with
  surface contact on, worst step ≤ 8.3 ms of the vehicles' GPU time (§8.52). The derby
  today is 25–27 ms a frame and the mountain 38–44 ms, on the Mac (§8.60). The last ten
  switches gave 1–25 % each, mostly on one-car or no-contact rows.
- **The measured floor says the hardware can do it.** A hand-written kernel ran the same
  physics, bit for bit against the same CPU reference. It ran one car in 1.1–1.2 ms a
  frame, 40 cars apart in about the same 1.21 ms, and a head-on pair in 3.1–4.0 ms. The
  floor-review table puts production at 3.2, 4.1 and 15.4 ms for its comparable rows.
- **There are two separate gaps, and the launch chain is only part of one of them.**
  - **Gap A, a car's own step.** Twenty cars that never touch run as one dispatch a
    frame: `Solo`, a workgroup a car, all 34 steps (§8.54). That dispatch still costs
    13.0 of a 13.8 ms frame, against the floor's 1.21 ms for 40 cars. So launches cannot
    be the cause there. A car's step inside a workgroup costs 0.4–0.8 ms a step in every
    contact row as well (§8.54), so the same kernel also dominates contact frames.
  - **Gap B, the contact frame's structure and work.**
    - About 850 launches a frame, at 2–5 µs each on the Mac's host; hundreds of them
      have no workgroup (§8.57).
    - Vehicle contact's tests grow with the pile: 59 of 126 ms on the T4's
      mountain20_t20 (§8.57).
    - The surface pass's pair path: 23 ms of that same frame.
- **So the swings are:**
  1. find out why the per-car kernel is 10× its floor, then rebuild it natively;
  2. cut the contact frame's launches structurally;
  3. make the contact tests' work proportional to real contact.

  Each is lossless and gated bit for bit, as AGENTS.md requires. One part, lockstep
  across workgroups on native backends, needs the owner's decision (§6).

## 2. Where 20 cars' time goes today

ms a frame of 34 steps, bit-exact on; Mac: M3 Max, Metal; T4: Modal, CUDA (§8.60,
generated sedan):

| row | Mac | T4 | target |
|---|---|---|---|
| drive (1 car, no surface contact) | 3.14 | 3.92 | |
| apart1 (1 car, surface contact) | 5.72 | 7.40 | |
| apart20 (20 cars, never touching) | 14.60 | 25.94 | 8.3 |
| clash20 (10 head-on pairs mid-crush) | 21.37 | 37.99 | 8.3 |
| derby20_t5 | 26.96 | 53.47 | 8.3 |
| mountain20_t20 | 43.82 | 85.52 | 8.3 |
| pile32 (no surface contact) | 41.12 | 80.14 | |

By dispatch kind (§8.54, Mac, the older car; the proportions are what matter):

| row | frame | members' / solo step | fleet walk / tests | levels | the rest |
|---|---|---|---|---|---|
| apart20 | 13.8 | 13.0 (one `Solo` dispatch) | — | — | 334 of 378 dispatches empty |
| clash20 | 25.2 | 17.3 (34 × 0.51) | 0.2 | 3.8 | 3.6 |
| mountain20_t20 | 66.9 | 26.0 (34 × 0.76) | 30.7 (34 × 0.90) | 6.4 | 7.5 |

On the T4's mountain20_t20 (§8.57): the fleet's tests are 59 of 126 ms, the surface
pair path 23 ms and the levels 10 ms.

Three readings:

1. **The per-car step is the largest single cost in every row**, and it does not shrink
   with fewer launches. Its cost per car is the problem:
   - the floor runs 40 cars in the time of one;
   - production's `Solo` costs 2.5× more at 20 cars than at one (14.6 against 5.7);
   - with surface contact on, its one-car step costs 1.8× the step without (5.7 against
     3.1).
2. **The contact work that grows with the pile is the fleet's tests and walk.** Kept
   candidates fix clash20 (96.7 % of passes reuse them) but never run on the mountain
   (§8.55–§8.56).
3. **Launches set a floor of about 5 ms a frame** for a surface car alone: 19 a step
   (§8.54). Hundreds of a frame's launches carry no workgroup (§8.57).

## 3. The colleague's proposal against these numbers

| claim | verdict |
|---|---|
| "The cost isn't arithmetic, it's structure; ~850 dependent launches a frame" | **Half right.** About 850 launches is §8.57's own figure, and they cost the Mac's host 2–5 µs each, empty or not. But apart20 is a single launch and still 12× the floor. The largest cost is the per-car kernel's structure (where state lives, spills, occupancy), not the launch chain. |
| "Each car runs in a single workgroup, so 20 cars use half of a T4's 40 cores" | True of the `Solo`/member paths. But the floor has the same shape (a workgroup a car) and runs 40 cars in the time of one. So idle cores are not what costs: each workgroup is slow. |
| 1. One persistent kernel per frame, CUDA first, 2–4× expected | **Agreed as a spike, with the expectation lowered and the scope changed.** For cars apart this kernel already exists (`Solo`) and is slow, so persistence alone buys nothing there. In contact frames it removes launch overhead, about 2–4 ms of a 21–44 ms Mac frame. Its big win comes only if it also keeps each car's state on chip (item 2) and syncs only cars that touch. Pairs are solved by the floor kernel; piles of three or more are not (floor review §6.4). Grid-wide syncs at every phase would cost about as much as the launches they replace. |
| 2. Keep each car's state on chip across substeps; never tried on CUDA or native Metal | **Agreed, and this is the main bet** (§4.1). The floor review's correction says the same: #139 lost only through Slang → WGSL → naga with wgpu's injected checks. On `engine-metal` and CUDA it has never been measured. |
| 3. More cars per launch, fewer launches per substep (fuse passes) | Agreed (§4.2). The lossless forms are named there. |
| 4. Chrome's bottlenecks: drawing 8 cars 12–19.5 ms, the mountain 6–8× slower in Chrome, Chrome runs none of the switches | **Not found in main's docs**; possibly from #148's (profiling tools) baselines, still open. Worth checking, since the browser is the product. But "none of the switches" is wrong: the wide schedule, the fleet passes and fast arithmetic are the page's defaults (§8.8, §8.29, §8.40). The surface pipelines' vehicle-contact part is off for wgpu (§8.54). Chrome cannot turn off its injected bounds checks, a measured 5–12 % natively (§8.48). |
| Low-hanging fruit: switches on in Chrome, drawing cost, the Chrome mountain | Fine as a parallel browser track (§5). None of it moves the 20-car GPU frame, which is the goal's measure. |

Not in the proposal, and larger than items 3–4 on the mountain and the derby:
- the contact tests' work on piles (§4.3);
- the surface pass's cost (§4.4);
- CUDA Graphs, a nearly free launch cut on the platform the A/B loop runs on (§4.2).

## 4. The steps, in order

Each step lists its gate. The bar does not move: lossless, bit for bit with bit-exact
on, scenario matrix with no row slower (AGENTS.md's ladder; `run.sh --cuda --base A
--head B` on Modal first, the Mac last).

### 4.0 Two days of measurement that decide the shape of 4.1 and 4.2

Cheap, decisive, nothing built:

1. **Why `Solo` costs 2.5× at 20 cars than at one, and 12× the floor's 40 cars.**
   - On the T4: `COUNTERS=1` on apart1, apart8 and apart20 for occupancy, registers and
     spills, achieved bandwidth, L1/L2 hit rates, and warp stall reasons.
   - On the Mac: #148's Metal kernel replay (`metal_profile`) for the same rows.
   - Hypotheses, each with a counter that confirms or kills it:
     - register spills from the kernel's size ("Long-lived kernel values spill" memory;
       `explicit_beams_frame` is one of the largest functions);
     - workgroup memory or registers limiting residency, so cars share cores;
     - device-memory bandwidth from re-reading every node and beam each step;
     - the surface pass inside the step (apart1 against drive: 2.6 ms a frame for one
       car).
2. **The floor's layout on native.** Build #139's core kernel and today's frame kernel
   on `engine-metal` (no injected checks, strict arithmetic by compile option) and on
   CUDA, and time both. This is the one measurement the floor review leaves open (its
   §8, step 4). A 1.5× or larger result makes §4.1 a go.
3. **The launch floor on CUDA.** Run a frame with every dispatch sized to zero (the
   gate's buffer already allows that): this is the launch overhead alone on the T4.
   Compare it with the same frame captured as a CUDA graph.

### 4.1 A native per-car step built to the floor's layout (the main bet; lossless)

Built only if 4.0.2 says it pays.

- A new module behind a switch, native first (CUDA, then `engine-metal`): a workgroup a
  car, all steps, state in registers and workgroup memory. The floor review §6.3 lists
  everything it must add: wheels, proxies, sleep, debris, loop caps, uniform barriers.
- The existing kernels stay the GPU reference: bit for bit with `fast-check.sh` from the
  same state, then the matrix.
- Applies directly to apart*, and to the members' step of every contact row: the
  members' step is the solo step plus the member box, the throttle reset and the list
  state (§8.57's map).
- Expected: the floor review's one-car gap (2.6×), and more at 20 cars if 4.0.1 shows
  that residency or spills are what cost.
- Chrome keeps today's path unless a WGSL build of it also wins. #139 says it does not
  today, so expect two step kernels: native and portable.

### 4.2 The contact frame's launches cut structurally (lossless)

In order of cost:

1. **CUDA Graphs** (CUDA only, host-side only). `engine-cuda` uses none today. Capture
   each schedule's frame once and replay it, with the indirect counts still read on the
   device. No kernel change, no physics change, bit for bit by construction. Gate: the
   two-build CUDA A/B.
2. **No empty launches.** About 340 of a frame's launches get no workgroup (§8.57). The
   host already reads the plan's group count back without waiting
   (`set_vv_wide_seen`, `level_want`). Generalise that hint so that each schedule
   encodes only the launch kinds the last plan used. A frame whose plan changed still
   runs correctly, because the full launch set stays as the fallback, which is exactly
   how §8.57's dual schedule works.
3. **Fuse dependent per-step passes that run on the same elements.** The surface
   pipelines are 11 launches a step (§8.53). The node, coupling and wheels passes can be
   fused where §8.57's map shows no reader of another car's workgroup.
4. **A native per-frame kernel for cars in contact** (one launch, cars meeting inside
   it): the floor review §6.4, pairs first, with a fallback to today's path when the
   meeting is missed. This is the colleague's item 1. **It needs the owner's decision**
   (§6), because no API but CUDA's cooperative launch promises co-residency.

### 4.3 Vehicle contact's tests proportional to contact (lossless)

- The mountain's frame grows with the pile because the fleet's walk runs on every pass
  (§8.55, §8.56). Kept candidates never build there: a pile of 20 has every car near
  several others, and the list voids past its room or an owner's share, then backs off
  up to 2,176 passes.
- Steps:
  - read the fixtures' `fkCtl(FK_WAIT)` and the last build's counts by owner;
  - bound the return by doing the walk twice (time only);
  - make the list hold on piles, either with a room sized from the pile or with
    per-owner lists, so that a large owner does not void the rest.
- Upper bound by clash20's ratio: most of the walk's 10–30 ms a frame on the Mac.
- Gate: mountain20_t12/t20, derby20 and pile32, bit for bit.

### 4.4 The surface pass's cost (lossless)

- One car's surface contact against the ground adds 2.6 ms a frame on the Mac
  (apart1 5.72 against drive 3.14), and its pair path is 23 ms of the T4's
  mountain20_t20.
- Full fidelity means surface contact on, so this pass is paid in every goal row.
- Measure its parts with `PROFILE_SPLIT=1` and #148's replay: the prep's snapshot of
  every node, the find over every skin triangle, the pair path's listing. Then apply
  the same reuse argument as the self-collision list (a conservative skin, rebuilt only
  on motion past it).

### 4.5 Already measured, small: take or drop, don't rework

- `claude/jovial-moser-8fc7c2` (`PERF_CPL_RANGES`, unmerged, no PR): tumble −5.0 to
  −5.7 %, pile8 −2.9 %, no row slower, off by default. Open a PR or drop it.
- Vehicles' own step on the wide passes under contact (§8.57's map): bounded at
  3–7 % on the 20-car rows with surface contact. Not worth building ahead of 4.1.

## 5. The browser track, in parallel

The product runs in Chrome, but the goal's measure is the vehicles' GPU frame. These
items are real; none is a multiple on that frame.

- Check the colleague's figures (drawing 12–19.5 ms for 8 cars; the mountain 6–8×
  slower than native) against #148's WebGPU baselines and the lab's probe. If they hold,
  the Chrome-only gap is the largest one for the demo.
- Compile the World's kernels up front in the page through the async path (#152 left
  this for the Chrome work; known bug F10's lesson).
- Drawing gives way to physics (memory "Physics first"); skinning in one pass is done
  (§8.41).

## 6. Decisions for the owner

1. **Lockstep across workgroups on native backends** (§4.2.4): CUDA cooperative
   launch, and Metal relying on observed co-residency with a loud fallback. It is not
   possible in the shared WebGPU kernels (#138).
2. **Two step kernels**, native (4.1) and portable (today's), if 4.0.2 shows that the
   layout pays only natively. This goes against "one source" in spirit; both would be
   Slang.
3. **The lossy options** stay options: vehicle sleep thresholds (§8.27), the rigid
   stand-in for intact cars (plan C1, not started), and the RoR profile's idle-triangle
   skip. None is proposed here.

## 7. What would be wrong to do next

- More switches of 1–5 % on one-car rows, ahead of 4.0.
- A persistent CUDA kernel that keeps today's per-car layout and adds grid syncs at
  every phase: launch cost moves into sync cost, and the per-car step stays 10× its
  floor.
- Anything lossy without the owner's decision (AGENTS.md).
