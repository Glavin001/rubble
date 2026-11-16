# Deformable vehicles: the physical floor, where the time goes, and what to build (2026-11-14)

This is a read-only audit. No kernel or host code was changed, and no GPU run was made.
It is grounded in the kernel (`slang/kernels/explicit_beams.slang` with
`explicit_beams_rigid.slang`, `_drive`, `_surface`, `_sleep`), the layout
(`crates/engine-exec/src/explicit_plan.rs`), the host (`crates/engine-webgpu/src/explicit_beams_gpu.rs`,
`crates/engine-world/src/explicit_vehicles.rs`) and the CPU reference
(`crates/engine-softbody/src/explicit_beams.rs`). Measured costs come from
`lab-mountain-perf.md`, `fleet-benchmark.md`, `explicit-2khz.md`,
`explicit-vehicles-in-world.md` and `handoff/known-bugs.md`. The only new measurement is a
CPU microbenchmark (§1.4, source in Appendix A).

Each claim is marked as one of:
- **[code]**: read in the code at `c01d4715`;
- **[doc]**: measured in the repo's docs;
- **[assumed]**: a hardware figure that was not measured here;
- **[inferred]**: a conclusion drawn from the above.

The machine is an M3 Max: 40 GPU cores, 12 performance CPU cores, 400 GB/s unified memory.

## 0. Corrections to the brief

- **The lab's sedan runs 512 lanes a car, not 256** [code]. `explicit_plan::frame_lanes` gives
  512 to fleets of up to 32 vehicles, and to any fleet holding a vehicle with more than
  `WIDE_BEAMS` = 2,048 beams. Larger fleets of small cars get 256. The doc comment there
  records the measurement behind this choice, in µs of GPU time per step at 256 / 512 / 1,024
  lanes [doc]: Viper 23.0 / 20.3 / 24.6, E36 39.4 / 31.0 / 38.8, Mustang 44.6 / 40.4 / 46.0.
  Doubling or quadrupling the lanes barely changes a step's time. That alone says the step is
  bound by latency, not by throughput (§2).
- **With surface contact on, a frame is not one dispatch** [code, doc]. Under
  `EB_FRAME_SEG` each step runs as one segment, with the surface pass as nine pipelines
  between segments: 34 × 10 dispatches a frame (`lab-mountain-perf.md` §1a). Without surface
  contact, all 34 steps run in one dispatch.
- **A rebuild is no longer triggered whole-car by any one node** [code]. The reuse rule keeps
  one spread per piece label (`lab-mountain-perf.md` §6A). A piece's own violation still
  rebuilds the whole list, though. The "about 5 ms" is the walk's total per frame on the
  steps that cannot reuse the list (Chrome stub, §8), not the cost of one rebuild.
- **The CPU figure (45 s of mountain in 34 s wall) is not in the repo's docs.** It is used
  here as given. The docs' CPU figure is 12 s simulated in 12.0 s of wall time, with surface
  contact (§2 there).
- **Steps a frame: 34** (`gP.steps`; 34 × 60 = 2,040 Hz in the lab).

## 1. The physical floor

### 1.1 The car and its work per substep [code + doc]

The car is the generated sedan as the lab runs it: 548 nodes, 3,581 beams, 528 collision
triangles, raycast wheels. The self-collision pair list holds 5,000 pairs in free fall and
5,800–10,500 late on the mountain (margin 0.1 m, `lab-mountain-perf.md` §8).

| Work item | Count a substep | Bytes the kernel touches per item, as laid out today [code] |
|---|---|---|
| Beam evaluation (`beamUpdate`) | 3,581 | 184 B. Reads `s1`, `bn`, `c1`, `c2`, `s0` (5 × 16), the two ends' position and velocity (4 × 16) and two slot indices (8). Writes two incidence entries (2 × 16). |
| Node integration (`nodeUpdate`, `beamSum` gather) | 548 | 305 B. Reads P, V, F (48) and ≈13.1 incidence entries (13.1 × 16 = 209); `info` stays in registers for the frame. Writes P, V, F (48). |
| Listed pair test (`pairReuseTest` → `tryContact`) | 5,000–10,500 | ≈290 B. Liveness (16), the box (32), the node (16), `triGeoLoad`'s 9 records plus `tn` (160), the group word, the node's `.w` and exclusions (48). |
| Grid counting sort | 548 candidates, 1,024 buckets | small (workgroup atomics) |

Summed: the beams 0.66 MB, the nodes 0.17 MB and the pairs 1.45–3.0 MB give **2.3–3.8 MB a
substep, or 77–130 MB a 1/60 s frame**. That is traffic through the caches, not DRAM. The
car's whole working set (nodes 26 KB, beam state and statics ≈290 KB, incidence 115 KB,
triangle geometry 93 KB, pair list ≤84 KB) is **about 0.6 MB**. It stays in cache across
substeps.

**Arithmetic** [inferred, from the code's operations]. A beam is about 60–80 flops in
strict form: a sqrt, a divide and the yield and break branches. A pair test is about
100–150 flops; a node about 100. Altogether that is roughly **30–60 MFLOP a frame**.

### 1.2 Hardware figures used [assumed]

- GPU: 40 cores. FP32 ≈ 14 TFLOPS for the whole GPU, so ≈0.36 TFLOPS a core. DRAM 400 GB/s.
- One core's load bandwidth from its caches: ≈64 B a clock at ≈1.4 GHz, so ≈90 GB/s. Not
  measured; this is the figure the "one workgroup" bounds below depend on most.
- A dependent load that hits L2 costs ≈0.2–0.35 µs, and one that goes to DRAM ≈0.5–0.7 µs.
- One `sync()` costs ≈1–3 µs with its phase's minimum work. This is backed by measurements:
  the page car's frame without coupling took 0.72 ms for 33 steps of about 6–8 syncs
  (`explicit-2khz.md`), and the Viper took 20 µs a step.
- A dependent dispatch costs 5.3 µs natively (wgpu on Metal, a trivial kernel:
  `examples/dispatch_cost.rs`, 340 dispatches in 1.8 ms) [doc]. A dispatch whose kernel uses
  large workgroup arrays costs ≈0.16 ms natively, because naga's MSL writer zeroes them from
  one thread [doc]. Chrome's dispatch cost has not been measured.
- CPU: one P-core of the M3 Max; Appendix A measures it.

### 1.3 The GPU floor

**One car** (per 1/60 s frame of 34 substeps):

| Bound | Basis | Floor |
|---|---|---|
| Whole-GPU bandwidth (work spread over 40 cores) | 77–130 MB at 400 GB/s, and the working set is cache-resident | 0.02–0.3 ms |
| Whole-GPU arithmetic | 30–60 MFLOP at 14 TFLOPS | 2–4 µs |
| One core (one workgroup), as laid out | 77–130 MB at ≈90 GB/s | **0.85–1.45 ms** |
| One core, compact records (§3.4: pair 24 B, beam 56 B, node ≈100 B) | ≈0.6 MB a substep, ≈20 MB a frame | ≈0.23 ms |
| One core, arithmetic | 30–60 MFLOP at 0.36 TFLOPS | 0.08–0.17 ms |
| Dependency chain | 34 substeps × at least 3–4 dependent phases (node → beam → contact test and apply, plus the coupling) × 1.5–3 µs | **0.15–0.4 ms** |

**The floor for one car is about 0.3–0.5 ms a frame on this GPU.** It is set by latency
(the 34 serial substeps), not by work. The car's work is about 1 % of what the GPU could do
in that time.

**N cars apart.** The latency floor (≈0.3–0.5 ms) applies until the GPU fills. After that,
each car costs its own throughput: ≈20 MB a frame with compact records, at aggregate cache
bandwidth (≈1–2 TB/s assumed) or at worst DRAM. That is **≈10–50 µs a car**. So:
- 32 cars: ≈0.5–1.6 ms;
- 100 cars: ≈1–5 ms;
- 1,000 cars: ≈10–50 ms, which is memory-bound.

**Cars in contact.** Contact adds a vehicle–vehicle pair test set comparable to
self-collision, per touching pair. Spread over cores, a touching pair costs about one car's
latency floor × 1.5, so **≈0.5–0.8 ms**. Thirty-two cars in contact cost **≈1–2 ms**.

### 1.4 The CPU floor (one core) [measured here]

Appendix A is a C model of one substep with this car's counts: 548 nodes, 3,581 springs with
damping and a break check, 7,000 point–triangle plane tests and the integration. It is built
`clang -O2 -ffp-contract=off` (no FMA contraction, as the bitwise contract needs) and runs
20,000 substeps on one M3 Max P-core, with other agents' builds running.

| Part | Per substep | Per item |
|---|---|---|
| Beams | 6.2–6.3 µs | 1.74–1.77 ns |
| 7,000 pair tests | 26.9–27.0 µs | 3.84–3.86 ns |
| Nodes | 0.9 µs | |
| **Total** | **34.0–34.2 µs, so 1.16 ms a frame** | |

The real beam has kinds, plasticity, break groups and strict helpers, and real contacts
carry friction laws: allow 2–3× on the beams and contacts. That puts **a well-written
single-core CPU car at ≈1.5–2.5 ms a frame**.

The reference today runs ≈12.6 ms a frame: 34 s for 2,700 frames, which is 378 µs a
substep. That is 5–8× the CPU floor. It was written for clarity and for the bitwise
contract, not for speed. One example: `step()` allocates `Vec`s per substep for centred
damping [code].

### 1.5 Today against the floor

| Case | Measured [doc] | Floor | Ratio |
|---|---|---|---|
| One car parked (Chrome GPU ms per World step) | 4.5–4.8 | 0.3–0.5 | 10–15× |
| Free fall | 5.0 (Chrome), 6.3 (native) | 0.3–0.5 | 10–20× |
| Late mountain crash | 12.8–14.2 (Chrome) | 0.3–0.5 (plus more pairs: ≤0.7) | 20–45× |
| 32 cars apart | 11.75 (native, 1/8/32 = 9.34/10.12/11.75) | 0.5–1.6 | 7–20× |
| Cars in contact, 8 / 16 / 32 | 53.6 / 58.8 / 103.3 | ≈1 / 1–1.5 / 1–2 | 50–100× |
| CPU reference, one car | 12.6 ms a frame | 1.5–2.5 | 5–8× |

**By phase** (native, one snapshot at t = 35 s, the car before re-export, 20 frames each way
with ablations, ms a frame; `lab-mountain-perf.md` §7) [doc]:

| Phase | Measured | One-core floor as laid out (§1.3) | Ratio |
|---|---|---|---|
| Whole frame | 19.6 | ≈1 | ≈20× |
| Self-collision search (listing, walk, tests) | ≈14.2 (19.6 − 5.4) | pairs 0.3–1.0 | 15–45× |
| Coupling per-step passes | ≈5.7 (19.6 − 13.9) | well under 0.1 (548 nodes × a few shapes × 34) | >50× |
| Beams | ≈2.5 (19.6 − 17.1) | 0.25 | ≈10× |
| Everything except self-collision | 5.4 | 0.3–0.5 | 10–18× |

The ablation shares overlap: a step waits on its slowest phase, so the shares do not add.

**Contact group, 8 cars** (same state, ms a frame, `fleet-benchmark.md`) [doc]: 68.4 before
the grid fixes, 52.0 after. In the 52.0: members' self-collision ≈16, vehicle–vehicle
triangle tests ≈10, the levels' application ≈12 (chains serial per contact point), the rest
14.

## 2. Where the gap goes, by cause

### 2.1 Barriers per substep [code]

`sync()` is two barriers: `DeviceMemoryBarrierWithGroupSync` then
`GroupMemoryBarrierWithGroupSync`. These are the solo frame's syncs per substep, by
condition:

| Site | syncs | When |
|---|---|---|
| `coupleStep`: reactBy, integrate, spheres, count | 4 | coupled car (the lab); 3 on the first step |
| `driveStep` | 1 | driven car |
| `centredPhase` | 2 | centred damping |
| After the node phase and clears | 1 | always |
| Aero / commands | 1 | `EB_AERO` modules |
| After the beam phase | 1 | always |
| Grid scan and scatter | 5 | grid steps that build (not on reuse) |
| `pairReuseTests` | 1 | grid with reuse |
| `selfApply` | ≥2 (1 a round, up to 32 rounds, then the serial tail; 3 more with `selfSplit`) | steps with a hit |
| `surfaceStep` | up to 5 | surface contact |
| End of step | 1 | always |

The coupled lab car with drive and no hit, reusing its list, runs about **9 syncs (18
barriers) a substep**. A rebuild with hits runs about 16, and surface contact adds up to 5,
for 21. Over 34 substeps that is **≈300–700 syncs a frame**. At 1.5–3 µs each, the barriers
with their minimum phase latency cost **≈0.5–2 ms a frame**: about the whole floor, spent on
synchronisation alone.

### 2.2 Work per lane between barriers: serial dependent loads [code + inferred]

At 512 lanes:
- **Beams:** 3,581 / 512 = 7 beams a lane. Each is a chain of about four dependent loads
  (`s1` → `bn` → the four node records → the slot indices), and nothing overlaps them across
  iterations. That is ≈7 × 4 × 0.3 µs ≈ **8 µs a substep**.
- **Pair tests:** 5,000–10,500 pairs give 10–20 a lane. Each is about five dependent levels:
  liveness, box, node, then `tryContact`'s loads, then `triGeoLoad`'s 9 records. That is
  ≈10–20 × 1.5 µs ≈ **15–30 µs a substep**, or 0.5–1 ms a frame.
- **Walk (rebuild):** `pairReuseList` gives a triangle 2 lanes. Each lane walks up to 64 cells
  and their bucket runs serially, so the step waits for the triangle with the most
  candidates.
- **Coupling count (`cplCountBy`):** each lane loops over every proxy shape for its node:
  O(nodes × shapes) every substep. There were 100+ shapes on the mountain before per-piece
  windows, and still 14–73 at 30–37 s.
- **Group application (`selfApply`, vv levels):** the hits' chains run in rounds, one
  barrier a round. Past 32 rounds one lane finishes them, and in groups the chains run
  serially per contact point.
- **Lane-0 serial work:** `grabApply`, `brakeApply`, `resolvePending`, the 16-block scan
  total and the reuse decision all run on one lane.

With one 512-lane workgroup on a core, there are 16 SIMD groups to hide memory latency. A
core can hold several times that, so every dependent load is paid almost in full. **This is
the main cause [inferred]: per substep, the step is a chain of ≈9–21 phases, each as long as
its slowest lane's dependent-load chain.** The lane measurements fit this: 256 → 512 →
1,024 lanes gives −10 %, then +20 %.

### 2.3 Idle lanes and idle GPU [code + inferred]

- **One car uses 1 of 40 cores.** For 5–14 ms a World step, 39 cores idle while the frame
  runs.
- **The frame does not overlap the rigid step.** `encode_frame` runs cpl_plan → share → frame
  → loads after the rigid step on one queue, and wgpu barriers storage writes between
  passes. A car's frame is added to the World step's critical path [code].
- **Within the car, work is short and uneven.** Phases with fewer items than lanes leave
  lanes idle: a few dozen bodies (`cplIntegrate`), 16-lane scan blocks, lane-0 sections.
  The early-out `triangleFinish` and the throttle make work per lane uneven.
- **A contact group of k cars runs on one workgroup** [code: `contactGroup`;
  doc: `fleet-benchmark.md`]. Each member gets lanes / k. Eight cars in groups of 2 put 2×
  each member's work on one core, and the frame waits on the slowest group. That is why 8
  cars cost 5× what 4 cars do.

### 2.4 Data layout [code]

- **Nodes:** AoS of 3 × float4: (P, mass), (V, friction), (F, self group). That is
  48 B, vector-friendly; good for the GPU.
- **Beams:** 3 static uint4 (48 B: ends, kind/type, group, k, c, limits), 2 state float4
  (32 B) and 2 slot indices. `c2` is loaded before the kind branch, though only LIMITED and
  SUPPORT use it (the compiler may sink the load; not verified). Beam kinds are mixed in file
  order, so a SIMD group runs several kinds' branches.
- **Forces:** gathered, not scattered. Each beam writes its two ends' forces into the CSR
  incidence entries (16 B each), and the node sums them in a fixed order. This is already
  the best design for bitwise determinism, with no atomics. Its cost is 2 × 16 B written
  and read per beam, against ≈2 × 12 B at minimum.
- **Pair records:** 2 words (triangle, node) in `gNext`. Each test reloads the triangle's 9
  geometry records: **≈290 B a pair, of which ≈180 B is the same triangle's geometry**,
  reloaded once per candidate.
- **Indirections:** `gLists[slotBase + 2j]` for the incidence slots; `gStatic[nodeInfoBase]`;
  per-pair `gMisc` liveness and `gStatic` exclusions.

### 2.5 Host work, readbacks and what gates the next step [code + doc]

- **Per World step** (`explicit_vehicles.rs`): `write_params` (one uniform write), two buffer
  clears, cpl_plan (2–3 dispatches of nv workgroups), a copy into indirect args, cpl_share
  (indirect), the frame, cpl_loads and sleep. The status is read without waiting every 16
  frames (`STATUS_EVERY`). **Nothing in the frame waits on the host.**
- **The lab, CPU side:** a union-find over the beams for piece labels every 6 windows (this
  needs the beams' broken state read back), plus retired-bit uploads.
- **What gates the next step on the page:** `canStep` allows at most 2 readbacks in flight,
  and the World's ring holds 3. When the GPU is the limit, a step is refused and is not owed
  again (§8–9 there). At 120 Hz this starved steps until the draw-skipping fix. **The host is
  not a cost; GPU latency is** [doc].
- **Native split frames:** 340 dispatches with surface contact, ≈55 ms a World step of naga
  workgroup zeroing [doc]. Native-only, and bit-identical with `RUBBLE_SURFACE_SPLIT=0`.

### 2.6 Rebuilt every step that could be kept [code]

- **The grid's bucket counts:** cleared, and every candidate counted, every substep, even when
  the list is reused (reuse == 2 skips only the scan and the scatter). That is 548 workgroup
  atomics and 1,024 clears a substep; cheap but useless on reuse steps.
- **The whole pair list on any violation.** A list whose margin one piece exceeds is rebuilt
  for every triangle. The standard alternative rebuilds only the rows of the elements that
  moved (§3.1).
- **Every listed pair re-tested every substep,** including pairs whose gap at the last test
  is many times the most the two could approach since. Parked, the list is reused on 97 % of
  steps, but all 5,000 pairs are still tested 34 times a frame.
- **The coupling's node-against-shape count, every substep from scratch.** Every node is
  tested against every shape's sphere.
- **`triGeoFill` for every tested triangle, every substep.** This is needed: geometry changes.

### 2.7 Work done for parts at rest [code + doc]

- **Whole-car sleep exists** (`explicit_beams_sleep.slang`, an engine-wide setting). A
  sleeping car is not stepped. Whether the lab turns it on was not checked; the parked
  figures above are of an awake car.
- **Inside an awake car, everything is evaluated every substep:** all beams (needed: loaded
  structure at rest still carries forces), all pair tests (not needed, §2.6), all coupling
  counts (not needed).
- **Torn pieces at rest** are retired by the debris time to live. That is lossy by design,
  and the owner chose it: off by default, on in the lab.

## 3. Best known technique for each cost

For each, the technique (with its standard source), the design here, the expected cost
after, and whether the result stays the same.

### 3.1 Self-collision: Verlet lists done per element

**Sources:**
- Verlet lists with a skin, rebuilt on displacement (Verlet 1967; Allen & Tildesley,
  *Computer Simulation of Liquids*, §5.3);
- per-particle "half-skin" displacement tracking (the same text; LAMMPS and HOOMD's
  neighbour-list `check yes`);
- spatial hashing with a counting sort (Teschner et al. 2003, *Optimized Spatial Hashing for
  Collision Detection of Deformable Objects*; Green 2010, *Particle Simulation using CUDA*,
  NVIDIA);
- conservative advancement / time-of-impact bounds (Mirtich 1996; Redon et al. 2002).

**(a) A safe-until bound per pair.** At a pair's test, store its gap g past the contact band
(the plane distance less max(r, CLIP_DEPTH), as `pairFar` already computes). Skip the pair
until the accumulated relative-motion bound since then (the same per-axis spread and h·v
terms the list's rule uses, per piece) exceeds g. This is the list's own proof applied per
pair instead of per list.
- Expected: most of the 5,000–10,500 tests a substep drop when parked or cruising. Late, a
  fraction drops (only the crush zone keeps testing). The pair tests (0.5–1 ms a frame plus
  the waits) fall to ≈0.1–0.3 ms.
- **Identical (bitwise):** a skipped pair provably meets the band in neither path. It needs a
  rounding allowance like `fqSeparated`'s, and a CPU twin.
- Size: medium. A gap word a pair, a check in `pairReuseTest`, the CPU twin and tests.

**(b) An incremental (row-wise) rebuild.** When the bound fails for some elements, rebuild only
the rows of the triangles and candidates whose own displacement exceeded the half-margin,
not the whole list. The grid is already counted every step.
- Expected: the late rebuild total (≈5 ms in Chrome) scales with the moved fraction:
  ≈1–2 ms in a crash, ≈0 at cruise.
- Identical: the list stays a superset with the same box test, and selfApply orders the hits.
- Size: medium-large (the list's rows need stable slots; compaction is part of the build).

**(c) A balanced build.** Replace the per-triangle walk (2 lanes a triangle, ≤64 cells) with
node-centric queries: each candidate reads its one cell's triangle entries, after a counting
sort of (cell, triangle) entries. Or allocate per-triangle work by a prefix sum of its cell
counts (Green 2010).
- Expected: the rebuild costs its total work over 512 lanes, not its worst triangle. ≈2–4×
  on rebuild steps.
- Identical (same pairs).

**(d) Triangle-major list order with the geometry loaded once.** Sort the list by triangle (the
counting sort already exists). Each lane takes a fixed chunk of consecutive pairs and
reloads the geometry only when the triangle changes. Bytes a pair fall from ≈290 to ≈24.
- Expected: pair traffic ÷10. On one core: −0.5 to −1 ms a frame late.
- Identical: tests are pure, and selfApply applies the hits in the specification's order.

### 3.2 The coupling: a Verlet list and fewer barriers

The coupled car pays 4 syncs a substep plus O(nodes × shapes) in `cplCountBy` [code].

- **A per-frame (node, shape) list.** Build it once a frame from the shapes' swept spheres
  and the nodes' frame travel bound (the same proof as 3.1, with the proxies' velocities
  bounded over the frame). The per-substep count tests only listed pairs.
- **Fold `cplReactBy` and `cplSpheres` into neighbouring phases.** reactBy can go into the end
  of the previous step's last phase, and the spheres into the integrate phase with lanes
  split. That gives 4 → 2 syncs.
- Expected: 1.8–5.7 ms → ≈0.3–0.8 ms a frame.
- Identical (superset list; the same order of sums: fixed-point `gAcc`).
- Size: medium.

### 3.3 Barriers and phase structure

**Sources:**
- Harris, *Optimizing Parallel Reduction in CUDA*, and subgroup scans (Merrill & Garland
  2016, single-pass scan);
- software pipelining of loads (Volkov 2010, *Better performance at lower occupancy*).

**What to do:**
- Merge the clears into the node phase's barrier.
- Run `driveStep` beside the coupling's integrate.
- Scan the grid with subgroup operations: WebGPU `subgroups` has shipped in Chrome since
  M128 and wgpu has it. That takes the scan's 5 syncs to 2.
- Skip the bucket counting on reuse steps.
- Issue the next item's loads before the current item's arithmetic in the beam and pair
  loops (unroll by 2), so two dependent chains are in flight a lane.

Expected: 9–16 → 5–8 syncs a substep (−0.3 to −1 ms a frame), and −30–50 % on the beam and
pair phases' latency. Identical. Each change is small. **Measure first:** the barrier cost
with the existing `EB_ABLATE` bits 64 (one barrier instead of two) and 128 (none),
time-only. No doc reports this yet.

### 3.4 Data layout

**Sources:** Bell & Garland 2008 (sparse matrix–vector on GPUs: CSR, ELL, sorting rows by
length); Cuthill–McKee reordering for locality.

**What to do:**
- Sort beams by kind (and by type) within a car, so a SIMD group runs one branch.
- Renumber nodes by reverse Cuthill–McKee, so a group's endpoints share cache lines.
- Pack the beam statics into 32 B: local u16 ends plus a vehicle base, k, c, and limits only
  for the kinds that use them, in a side table.
- Store incidence forces as 12 B entries.

**Identical:** beamSum's order is the CSR's per-node order, which the beam evaluation order
does not change. Break groups use `min(local)`, which is order-free.

**Not identical, and needs the owner:** a renumbering changes nothing in the arithmetic. But a
beam's file index appears in the spec (§17 masters, `min(local)`), so the permutation must be
held beside the beams and mapped back. That is bookkeeping, not physics.

Expected: beams −30–50 % (≈1 ms a frame in the late phase). Size: medium (the layout, the CPU
twin's indexing, the readbacks).

### 3.5 Cars in contact: one car a workgroup, and contact across workgroups

**Source:** the standard multi-body GPU pattern. Per-body local solves run in their own
blocks, and interactions are exchanged through global memory at a kernel boundary (one
kernel launch a phase, as in PhysX GPU and Bullet's GPU pipeline). Global barriers inside a
kernel are not allowed in WebGPU: there is no forward-progress guarantee across workgroups.

**Design: C2's same-substep design** (`fleet-benchmark.md` "A contact group across
workgroups"). Each substep is two dispatches:
- (i) every member's step, one workgroup a car;
- (ii) the vehicle-contact pass, one thread a candidate pair across all groups, with
  fixed-point sums.

That is 68 dispatches a frame, whatever the number of cars. State that lives across substeps
moves to storage: the reuse state, the coupling's sphere copies, the per-frame drive words.

Expected:
- 34 × (2 × 5.3 µs dispatch + the member step + the pass);
- with today's member step (≈0.27 ms) that is 25–35 ms for 8 cars, the doc's estimate;
- **after 3.1–3.3 bring a member's substep to ≈20–40 µs, ≈1.5–3 ms for any number of
  touching cars** until the GPU fills.

**Identical** (the doc's analysis: per-element updates, fixed-point sums, hits in the
specification's order).

**What it forces:** the group kernels must use small workgroup arrays, or native pays ≈0.16 ms
of naga zeroing a dispatch [doc]. That would be 11 ms a frame at 68 dispatches. The fix is
either `create_shader_module_trusted` after a read-before-write audit, or the state in
storage. Also needed: Chrome's dispatch cost measured with the real bind group. Size: large
(the whole group path).

**Lossy alternative (owner's decision):** a two-phase substep, where each car tests
contact against its partner's state one substep old. It needs no barrier inside a substep,
so groups dissolve into solo frames. Not identical.

### 3.6 Fleets of cars apart

- One workgroup a car is the persistent-block pattern. It avoids 34 × k dispatches and keeps
  per-substep data local.
- It scales across cores until residency fills: ≈40 cores × 2–4 resident 512-lane groups,
  so 80–160 cars at one car's latency (1 → 32 cars: 9.3 → 11.8 ms [doc]).
- Past that, throughput binds, and **256 lanes for large fleets** (the current rule) is right.
- The existing **wide passes** (a dispatch per phase per step across all vehicles,
  `WIDE_PASSES`) are the phase-dispatch alternative. They run no coupling, aero, raycast
  wheels or surface contact, and were slower for one sedan (0.63 vs 0.44 ms [doc]).

Expected after 3.1–3.4: per car ≈20–50 µs of throughput, so ≈2–5 ms for 100 cars. Measure:
the fleet bench at 64/128/256 generated sedans (only 32 has been measured).

### 3.7 CPU threads per car (BeamNG's public description)

BeamNG publicly describes its physics as running at 2,000 Hz with vehicles spread over CPU
cores. That is public material only; no source was read.

On this machine (§1.4) a well-written car costs ≈1.5–2.5 ms a frame on a P-core, so 12
P-cores carry ≈12 cars at ≈2 ms. For a single car that is worse than the GPU's floor
(0.3–0.5 ms) but better than today's GPU (4.5–14 ms).

**What it forces:**
- The rigid world's contacts and proxies live on the GPU. A CPU car can exchange with them
  once a frame, not every substep. That changes the coupling's time resolution: **lossy, the
  owner's decision**.
- It also needs a CPU reference 5–8× faster than today's (SoA, no per-step allocations,
  SIMD), which is worth doing anyway for validation speed. It stays bitwise if the SIMD keeps
  the scalar order per lane.

## 4. Is one workgroup a car the right architecture?

**One car: yes in principle; the cost is the work inside it, not the shape.** A workgroup
barrier (≈1–3 µs) is cheaper than a dependent dispatch (5.3 µs native; more in Chrome,
unmeasured). One car's 3,581 beams would fill only ≈3 % of the GPU in a phase-dispatch
design, which therefore cannot beat one workgroup's latency. The floor is ≈0.3–0.5 ms, and
today's 4.5–14 ms comes from:
- 9–21 phases a substep;
- each phase as long as a lane's chain of 7–20 dependent loads;
- O(nodes × shapes) coupling;
- whole-list rebuilds.

More lanes do not help (measured). Fewer phases, shorter chains and less work do.

**(b) Fewer or more lanes sized to the car.** No real effect (−10 % at 512, worse at 1,024).
Keep 512 for small fleets and 256 for large ones.

**Hundreds of cars apart: yes.** Throughput-bound past ≈100 cars, with no dispatch overhead
and data local to each car. (a) one dispatch a phase across many workgroups would add 100–200
dispatches a frame, and wins only if per-car latency stays where it is. Keep (a) as the wide
passes for vehicles too big for one workgroup.

**Cars in contact: no.** Putting a group on one workgroup puts k cars' work on one core, and
the frame waits on the biggest group: 8 cars cost 53 ms against ≈1 ms of floor. Contact
belongs across workgroups (3.5), with each car keeping its own workgroup for its local
phases.

**(c) Few cars on CPU threads, crowds on GPU.** Not as the base: it makes the rigid coupling
frame-rate (lossy) and duplicates the hot path's tuning on two targets. It is a fallback for
cars that touch no dynamic rigid body, if the owner wants it.

**If starting today, I would build:**
- **Solo cars:** one workgroup a car for the whole frame (as now), designed for ≤4–5 syncs a
  substep. Every contact search runs as a per-element Verlet list with safe-until bounds and
  row-wise rebuilds, and the coupling's (node, shape) pairs are listed once a frame.
- **Cars in contact:** the same per-car kernel run a substep at a time, alternating with a
  global contact dispatch: two dispatches a substep, ≈68 a frame. The host picks per frame
  which cars run solo (all 34 substeps in one dispatch) and which run in the contact
  schedule, as it picks groups now.
- **Records compact and sorted:** beams by kind and locality, triangle-major pair lists.
- **One CPU twin, kept bitwise and made fast** (SoA, SIMD in scalar order), so validation runs
  at 5–10× real time.

This is today's architecture for solo cars and fleets, and C2's design for groups, with the
inner loops rebuilt. It is not a rewrite.

## 5. Ranked recommendations

Ranked by expected gain × how often it is paid. "Identical" means bitwise, the same as
today's contract.

| # | Change | Expected gain | Identical? | Risk | Depends on | Measure before committing |
|---|---|---|---|---|---|---|
| 1 | **Self-collision per pair:** safe-until bound (3.1a) + triangle-major list with the geometry loaded once (3.1d) | Every car every frame. Parked 4.5 → ≈3 ms. Late crash −2 to −4 ms (12.8–14.2 → ≈9–11) | Yes | Low–medium (rounding allowance; CPU twin) | none | On the CPU twin: per substep, the share of listed pairs whose gap exceeds the bound (`--tri-census` style). GPU: the 2^26 / 2^27 ablations' share at parked, 9 s and 35 s |
| 2 | **Incremental, balanced rebuild** (3.1b, c) | Late ≈5 ms → ≈1–2 ms (−3 to −4) | Yes | Medium (stable row slots) | 1 (shared bounds) | Rebuild census: the elements past the half-margin at each rebuild; the walk's per-lane spread |
| 3 | **Coupling list + barrier folding** (3.2) | 1.8–5.7 → ≈0.3–0.8 ms in coupled scenes (the World, mountain, town) | Yes | Medium | none | Ablation 32768 (count) and 8192/16384 shares at t 10, 20, 35 s; shapes a node meets |
| 4 | **Contact groups across workgroups** (3.5, C2) | 8/16/32 touching cars 53/59/103 → ≈25–35 now, ≈2–4 ms after 1–3 | Yes | High (whole group path; workgroup zeroing natively; Chrome dispatch cost) | 1–3 for the big gain | A Chrome dependent-dispatch microbench with the frame's bind group; the member step's cost after 1–3 |
| 5 | **Fewer syncs, pipelined loads, subgroup scan** (3.3) | −0.3 to −1 ms a frame, every car | Yes | Low | none | `EB_ABLATE` 64/128 barrier cost (time-only), parked and late |
| 6 | **Compact, sorted beam records** (3.4) | Beams −30–50 % (≈−1 ms late; more for 6k-beam cars) | Yes (with the index map) | Medium (touches every reader of beam records) | none | The beam ablation (32) share on the re-exported car; SIMD divergence by kind census |
| 7 | **A fast CPU twin** (SoA, no per-step allocation, SIMD in scalar order) | Validation 5–8× faster; CPU-hosted cars possible | Yes | Low–medium | none | A profile of the CPU twin on the mountain |
| 8 | **Large-fleet bench** (64–256 generated sedans, 256 vs 512 lanes) | Decides whether the phase dispatch matters for fleets | n/a | none | 1–6 | The bench itself |

**Owner decisions (not lossless, not recommended without a decision):**
- vehicle contact one substep behind (dissolves groups);
- CPU cars with frame-rate rigid coupling;
- sleep for torn pieces at rest without the debris TTL's retirement;
- fast-math or FMA in the kernels (breaks CPU==GPU).

Expected total for one car after 1, 2, 3, 5 and 6: **parked ≈1.5–2.5 ms, late crash ≈4–6 ms.**
That is still ≈5–10× the floor. Closing the rest needs the phase count at 3–5 a substep and
compact records end to end, which this plan works toward but does not fully reach. The
latency floor (≈0.3–0.5 ms) is not reachable while the frame holds 9+ syncs a substep.

## 6. What is verified and what is inferred

**Verified in code:**
- the lane policy (512 / 256);
- the barrier sites and counts per substep;
- `sync()` = two barriers;
- the node, beam and pair load sets and record sizes;
- the CSR gather (no atomics for beams);
- `cplCountBy`'s O(nodes × shapes) loop;
- bucket counting on reuse steps;
- the per-triangle walk at 2 lanes;
- `triGeoLoad`'s 9 records a pair;
- that the frame is encoded after the rigid step with no host wait inside;
- status reads every 16 frames;
- contact groups on one workgroup;
- the wide passes' existence and limits;
- the CPU reference's per-substep allocations under centred damping.

**Measured here:** the CPU microbenchmark (§1.4).

**From the docs:** every GPU millisecond quoted, the ablation shares, dispatch and zeroing
costs, and the lanes measurements.

**Inferred or assumed:**
- per-core GPU load bandwidth (≈90 GB/s);
- load latencies (0.2–0.7 µs);
- the per-sync cost (1.5–3 µs, fitted to the page car and the Viper);
- every floor number;
- every expected gain;
- that dependent loads, rather than barriers alone, dominate a phase;
- residency (2–4 workgroups of 512 lanes a core);
- the share of pairs a safe-until bound would skip.

**Not verified:**
- the CPU figure of 34 s for 45 s (not in the docs);
- Chrome's dispatch cost;
- whether the compiler sinks `c2`'s load;
- the lab renderer's per-frame readback volume (the skin path was not read);
- BeamNG's threading details beyond its public statements.

## Appendix A: the CPU microbenchmark

`clang -O2 -ffp-contract=off beambench.c && ./beambench` (M3 Max, one P-core, a busy
machine). Output, two runs: 34.0 and 34.2 µs a substep; beams 1.74–1.77 ns each, pair tests
3.84–3.86 ns each, nodes 0.86–0.88 µs; 1.16 ms per 34-substep frame. A model of the work's
shape (scalar, scatter-add, plane test only), not of the full beam laws.

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define N 548
#define B 3581
#define T 528
#define P 7000
static float px[N],py[N],pz[N],vx[N],vy[N],vz[N],fx[N],fy[N],fz[N],im[N];
static int ba[B],bb[B]; static float rest[B],kk[B],cc[B],str[B]; static unsigned char brk[B];
static int ta[T],tb[T],tc[T]; static int pt[P],pn[P]; static int hits;
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(){
  srand(1);
  for(int i=0;i<N;i++){px[i]=(i%10)*0.3f;py[i]=((i/10)%6)*0.3f;pz[i]=(i/60)*0.3f;vx[i]=vy[i]=vz[i]=0;im[i]=1.0f/2.5f;}
  for(int j=0;j<B;j++){int a=rand()%N,b=(a+1+rand()%40)%N;ba[j]=a;bb[j]=b;
    float dx=px[a]-px[b],dy=py[a]-py[b],dz=pz[a]-pz[b];rest[j]=sqrtf(dx*dx+dy*dy+dz*dz)*1.001f;kk[j]=2e6f;cc[j]=200;str[j]=1e9f;brk[j]=0;}
  for(int t=0;t<T;t++){ta[t]=rand()%N;tb[t]=rand()%N;tc[t]=rand()%N;}
  for(int p=0;p<P;p++){pt[p]=rand()%T;pn[p]=rand()%N;}
  const float h=1.0f/2000; int steps=20000;
  double t0=now(), tb_=0, tn=0, tp=0;
  for(int s=0;s<steps;s++){
    double a0=now();
    for(int i=0;i<N;i++){fx[i]=0;fy[i]=-9.81f/im[i];fz[i]=0;}
    for(int j=0;j<B;j++){ if(brk[j])continue; int a=ba[j],b=bb[j];
      float dx=px[a]-px[b],dy=py[a]-py[b],dz=pz[a]-pz[b];
      float q=dx*dx+dy*dy+dz*dz; float inv=1.0f/sqrtf(q); float len=q*inv;
      float e=len-rest[j]; float u=((vx[a]-vx[b])*dx+(vy[a]-vy[b])*dy+(vz[a]-vz[b])*dz)*inv;
      float f=-kk[j]*e-cc[j]*u; if(fabsf(f)>str[j]){brk[j]=1;f=0;}
      float s_=f*inv; fx[a]+=dx*s_;fy[a]+=dy*s_;fz[a]+=dz*s_; fx[b]-=dx*s_;fy[b]-=dy*s_;fz[b]-=dz*s_; }
    double a1=now();
    for(int p=0;p<P;p++){int t=pt[p],n=pn[p]; int A=ta[t],Bq=tb[t],C=tc[t];
      float e1x=px[Bq]-px[A],e1y=py[Bq]-py[A],e1z=pz[Bq]-pz[A],e2x=px[C]-px[A],e2y=py[C]-py[A],e2z=pz[C]-pz[A];
      float nx=e1y*e2z-e1z*e2y,ny=e1z*e2x-e1x*e2z,nz=e1x*e2y-e1y*e2x; float nl=sqrtf(nx*nx+ny*ny+nz*nz)+1e-20f;
      float d=((px[n]-px[A])*nx+(py[n]-py[A])*ny+(pz[n]-pz[A])*nz)/nl; if(fabsf(d)<0.025f) hits++; }
    double a2=now();
    for(int i=0;i<N;i++){ vx[i]+=fx[i]*im[i]*h; vy[i]+=fy[i]*im[i]*h; vz[i]+=fz[i]*im[i]*h;
      px[i]+=vx[i]*h; py[i]+=vy[i]*h; pz[i]+=vz[i]*h; if(py[i]<-1){py[i]=-1;vy[i]=0;} }
    double a3=now(); tb_+=a1-a0; tp+=a2-a1; tn+=a3-a2;
  }
  double t1=now();
  printf("per substep: total %.2f us  beams %.2f us (%.2f ns/beam)  pairs %.2f us (%.2f ns/pair)  nodes %.2f us; hits %d\n",
    (t1-t0)/steps*1e6, tb_/steps*1e6, tb_/steps*1e9/B, tp/steps*1e6, tp/steps*1e9/P, tn/steps*1e6, hits);
  printf("per 1/60 s frame (34 substeps): %.2f ms\n",(t1-t0)/steps*34*1e3);
}
```

## 7. First pass against this audit: measured phases, one change adopted, where the targets stand (2026-11-15)

Branch `claude/rubble-vehicle-gpu-perf-2d0f9c`. M3 Max, native wgpu on Metal and headless
Chrome (page visible to the compositor, 60 Hz). Benchmark: the vehicle lab's mountain at the
page's defaults (seed 1, 50°, roughness 0.6, launch 14 m/s, no rockslide, surface contact
off, self-collision across the GPU on), the generated sedan (548 nodes, 4,251 beams, 528
collision triangles), 40 s. With these defaults the car is at rest, crumpled, from t ≈ 10 s,
so three quarters of the run is a wreck lying still.

New in the lab: a **"cars" control** (1–8; `LabOptions::cars`, `vehicle_lab --cars N`): the
cars leave the lip together, side by side and stacked, closing on the first. It rebuilds
the lab in place (the fleet is fixed when the World is built). The run's GPU time a step
as median / p95 / worst: `vehicle_lab --measure mountain-perf --timing step` natively, the
page's `__gpuRun()` (printed by `vehicle-lab-record.mjs --probe`).

### 7.1 Where the targets stand

Vehicle GPU ms per World step (the explicit frame and the coupling's passes), steps from
t = 0.5 s, whole run:

| Case | Before: median / p95 / worst | After: median / p95 / worst | Target (worst) |
|---|---|---|---|
| 1 car, native | 6.71 / 6.94 / 7.37 | 5.82 / 6.02 / 6.55 | 1.0 |
| 1 car, Chrome | 6.81–6.87 / 7.23–8.76 / 7.77–12.29 (two runs) | 5.97–6.02 / 6.44–7.88 / 6.92 (15.2 in one run, one step) | 1.0 |
| 8 cars, native, default paths | 312 / 534 / 553 | unchanged | 3 |
| 8 cars, native, `RUBBLE_GROUP_SPLIT=1` and self split (14 s) | 168 / 257 / 270 | unchanged | 3 |

**Neither target is met.** One car is 6–7× over its target; eight cars in a pile are
50–180× over theirs. 60 steps/s held for one car in Chrome (10–20 steps refused in the
first second of each run). Eight cars were not run in Chrome, and 120 steps/s was not
tested.

### 7.2 One car's frame by phase (native, measured)

`vehicle_lab --ab T --ab-ablate …`: 20 frames of 34 steps from one snapshot under each
mask, wall ms a frame, self split on. Skipping a phase removes its chain and what it
feeds, so shares overlap; rows marked ≤ include downstream work.

| Phase | Free fall, t 2.5 s | Wreck at rest, t 20 s | Measured how |
|---|---|---|---|
| Whole frame | 4.5 | 7.1 → **6.2** | |
| Self-collision hits applied (`selfApply`) | ≈0 | 2.1 → **1.1** | second mask 64 |
| Listed pairs' tests and walks (their dispatches) | 0.4 | ≈0.45 | `set_self_split_skip` 3, less `selfApply` |
| Raycast wheels | 0.56 | 0.58 | second mask 1 |
| Nodes | 0.5 | 0.6 | mask 2 |
| Beams | 0.5 | 0.4 | mask 32 |
| Coupling (step passes and node contacts) | 0.4 | 0.5 | mask 122880 |
| 35 step dispatches that return at once | 0.33–0.38 | same | second mask 4096 |
| … with the frame's start in each (records, rebase check, 3 barriers) | 0.45–0.53 | same | second mask 8192 |
| Everything else in a step (the reuse decision and tracking, bucket counts, triangle throttles, volatile triangles, barriers) | ≈1.1 | ≈1.2 | remainder |

The in-workgroup frame (self split off) is 6.2 ms in free fall and 8.7–9.1 ms from the
first impact on; a frame whose step loop is empty costs 0.47 ms there (one dispatch).

Two corrections to §1–§2 from these numbers:
- **No phase is under 0.3 ms a frame, and none is over 1.1 ms after the change below.**
  There is no single cause left to remove for one car; the frame is about a dozen
  dependent phases a step of 10–30 µs each.
- **A phase's cost is its longest lane's chain, and a core is the unit.** The repo's own
  prototype (`wide-frame-design.md` §1.1: beams and nodes only, simple springs, 804 nodes,
  6,427 beams) measured 18 µs a substep for two phases in one workgroup and 15–17 µs as
  two dispatches across the GPU: **about 9 µs a dependent phase either way, 0.3 ms a
  frame**. §1.2 assumed 1.5–3 µs. A step with self-collision needs at least four phases
  in sequence (nodes, beams, the pair tests, the hits applied), so the floor of a crash
  frame is about 4 × 0.3 = **1.2 ms** before the hit rounds, the coupling and the wheels,
  and free flight about 0.6 ms. **The 1.0 ms worst-frame target is below that for crash
  frames on this GPU** with any schedule built from these phases. The realistic target
  for one car is 1.2–1.5 ms; today is 6.2–6.5.

### 7.3 Adopted: `selfApply`'s node forces summed once a hit (perf flag 1, on by default)

The two-body rule's round and the residual pass's two phases each summed the beam forces
of a hit's four nodes again, in series on the hit's lane (12 `beamSum`s a hit). Now a
phase before the rounds computes each (hit, role)'s sum on its own lane into the hit list's
region (`hitFbFill`, 12 words a hit after the pending beams), and the three readers load
it. Nothing in `selfApply` writes what `beamSum` reads (the incidence entries and the
step's group slots), so the stored bits are the bits each reader computed.
`explicit_plan::PERF_HIT_FB`, `ExplicitBeamsGpu::set_perf_flags`, `RUBBLE_PERF_FLAGS=0`
off, the page's "hit forces summed once" (live), `vehicle_lab --ab-switch perf=1`.

Bit for bit (every state word, 20 frames of 34 steps each way from one snapshot):
- mountain, one car, self split on: t 2.5, 4, 6, 8, 20 s; with flag 2 as well, self split
  on and off: t 0.5, 2.5, 4, 6, 8, 20 s;
- two cars in contact: t 1, 2, 3, 5 s; eight cars in contact: t 1, 2, 3 s (flags 1 and 2);
- against the CPU reference and the frozen kernel with the flags on
  (`tools/explicit-2khz/perf-verify.sh 3`): `explicit_loop_cap` 5, `explicit_beams_bitwise`
  12, `explicit_contact_gpu` 13, `explicit_pair_reuse_gpu` 2, `explicit_drive_gpu` 3,
  `showcase_car_gpu` 6 (1 ignored, as before), `vv_grid_gpu` 2, `explicit_jbeam_gpu` 6;
  engine-world `explicit_vehicles` 11, `vehicle_lab_reset` 3, `vehicle_lab_piece_windows` 1,
  `explicit_drive_world` 1. The same suites and `explicit_plan_modules` pass with every
  flag off (`perf-verify.sh 0`: the old path after the kernel edits).
- **Not run:** the `engine-metal` parity suites (Metal and CUDA hosts take the flag from
  the shared plan, so they run the new path by default), and the Chrome module check
  `softbody-check.mjs` (the lab page itself loaded and ran with both flags on, no console
  or WGSL error).

Time, same snapshots, ms a frame, off → on: free fall 4.40 → 4.41; t 4 s 6.69 → 5.90; 6 s
7.08 → 6.15; 8 s 6.96 → 6.00; 20 s 7.17 → 6.20 (−12 to −14 %). Chrome, whole run: median
6.81 → 5.97–6.02, p95 7.23 → 6.44 in the clean run.

### 7.4 Tried, bit for bit, not faster: the wheels beside other phases (perf flag 2, off)

The wheels' rays and bars for step s + 1 beside step s's beam phase and after it, their
tyres beside the coupling's first phase (`driveAheadPart`; three barriers a step fewer).
Every state word the same at all six times, split and in-workgroup. Time: −2 % to +7 %
natively, 6.02 against 5.97 ms median in Chrome. A wheel's tyre chain is longer than the
coupling phase it shares, and its rays lengthen the beam phase on their lanes, so the
frame waits as long as before: removing barriers does not pay when the guest's chain is
the longer one. Kept behind its switch (the page's "wheels beside other phases").

Also measured: one barrier instead of two per `sync()` (mask 64) changes the state on
Metal, so its timing (−1.3 ms at rest) is not a result; both barriers are needed.

### 7.5 Remaining gap per phase, and what would close it

| Phase | Now (rest) | Floor (0.3 ms a dependent phase) | Why it is above |
|---|---|---|---|
| Hits applied | 1.1 | 0.3–0.6 | five phases in sequence (fill, round, residual, guard, merge); each is a real dependency of the specification's residual pass |
| Pair tests | 0.45 | 0.3 | at its floor within 2× |
| Nodes, beams | 0.6, 0.4 | 0.3 each | 548 nodes on 512 lanes put two nodes on 36 lanes; 8 beams a lane |
| Coupling | 0.5 | 0.3 | two phases (folded) |
| Wheels | 0.58 | 0 beside another phase | the tyre chain; see §7.4 |
| Dispatches and frame start | 0.5 | 0.07 (34 × 2 µs) | 10 µs a step dispatch of the 470 KB kernel, not the 2 µs of a trivial one |
| Rest of a step | 1.2 | ≈0.1 | not split further |

Eight cars: the default path steps a group of eight on one workgroup (the big-group pass);
the existing split paths halve it and leave the pass of a group past four members in one
workgroup (`contact-groups-design.md` §5.3). Reaching 3 ms needs that pass across
workgroups for any group size, then every per-car phase above. Not started here.

## 8. The architecture from the GPU's measured cost model: one schedule for the whole fleet (2026-11-15)

§7 treated the frame as given. This section derives the structure from what the GPU
charges, measured (`examples/gpu_cost_model.rs`, native Metal, M3 Max; logs
`out/perf/cost_model*.log`): synthetic kernels on a 1 MB table (a car's working set),
dependent dispatches in one pass, and barrier phases in one workgroup.

### 8.1 The cost model

| What | Cost |
|---|---|
| A dependent dispatch of a small kernel, 64 to 4,096 threads | 2.2–2.7 µs |
| The same, 32,768 threads (eight cars' beams) | 4.4–5.0 µs |
| Each buffer load a thread makes in sequence, in such a dispatch | +0.2 µs (streamed), +0.4 µs through an index, +0.27 µs chained; the dispatch waits for its longest thread |
| A barrier phase in a workgroup, nothing to do | 0.12–0.3 µs |
| A barrier phase, 4 to 64 lanes active, one item of L loads | 0.3 + 0.11·L µs |
| A barrier phase, every lane active (512 lanes × 8 items × 4 loads) | 10.3 µs: one core's throughput, ≈0.6 ns a load |
| 8 or 32 workgroups in one dispatch | the same per phase: cores in parallel |
| A dispatch of the 470 KB frame kernel that returns at once (§7.2) | ≈10 µs |
| Native only: a dispatch of a kernel with workgroup arrays, zeroed by naga | ≈70 µs (measured below) |

So: **threads are cheap and loads are dear.** A thread pays about 0.2 µs for every load it
makes in a row, dependent or not; a workgroup of 512 lanes is one core, and a full phase
there is bound by that core's throughput. Work with many items (beams, nodes, pairs)
belongs one item a thread across the GPU; work with a few items and several dependent
phases (hits, wheels, proxy bodies) belongs in a small workgroup, where a phase costs
0.3 µs plus its chain. Kernels must be small (a dispatch of the big frame kernel costs
four times a small one's) and carry no workgroup arrays natively.

### 8.2 What the repo's wide passes already do

The wide passes (`explicit_beams_wide_*`: a thread an element over every vehicle, four
dispatches a step, the same per-element functions as the frame) were measured 6–7× slower
than a workgroup a car (`wide-frame-design.md` §1.2). **That was naga's zeroing of their
workgroup arrays, per dispatch, not their structure.** `examples/wide_vs_workgroup.rs`,
the repo's sedan (367 nodes, 2,170 beams, self-collision on), ms a 34-step frame:

| Cars | Workgroup a car | Wide, zeroed (as shipped natively) | Wide, not zeroed (`RUBBLE_WIDE_ZERO=0`) |
|---|---|---|---|
| 1 | 1.17–1.21 | 7.34 | **1.08** |
| 8 | 1.24–1.27 | 7.34 | 1.32 |
| 32 | 1.47 | 7.89 | 1.93 |
| 128 | 4.55 | — | 4.26 |

Per pass at one car (each dispatch timed apart): nodes 0.42, beams 0.29, triangles 0.17,
hits 0.22 ms a frame. Browsers never paid the zeroing (Tint zeroes across the lanes).

### 8.3 The lab's car on the wide passes (built, bit for bit)

The wide passes could not run raycast wheels or the rigid coupling. Now they do: an **aux
pass** a step (`explicit_beams_wide_aux`, a workgroup a vehicle: `coupleStep` then
`driveStep` exactly as the frame calls them; after the last step the coupling's end), the
coupling's header loaded in `wide_nodes` (its proxy contacts read spheres and lists from
the buffers), the coupling's rebase in `wide_begin`. `ExplicitBeamsGpu::set_wide`,
`RUBBLE_WIDE=1`; off by default.

Mountain, the lab's generated sedan (`vehicle_lab --ab-switch wide`, 20 frames of 34 steps
each way from one snapshot, self split off, `RUBBLE_WIDE_ZERO=0`): **every state word the
same as the workgroup frame** at t 0.5, 2.5, 4, 6, 8 and 20 s; 5.1–5.4 ms a frame wide
against 5.8–6.1 in free fall, 6.4–6.7 against 8.3–9.0 from the first impact on.

Per pass (`--ab-wide-profile`, GPU ms a frame, timed apart), against what §8.1 charges
for the same work done as it should be:

| Pass | Free fall, t 2.5 | Wreck at rest, t 20 | By the cost model | What gets it there |
|---|---|---|---|---|
| Self-collision triangles (grid walk and tests, every step) | 2.18 | 2.81 | ≈0.3 (a thread a listed pair: 2.5 + 0.2 × 15 µs) | the reused pair list's walk and pairs kernels (the self split's, already across the GPU) in this schedule |
| Aux (coupling, wheels) | 1.52 | 1.58 | ≈0.3–0.4 | its own small kernels: the frame's functions inline a 10 µs dispatch and five phases |
| Hits applied | 0.20 | 1.85 | ≈0.2–0.4 (five phases of ≤64 active lanes) | not yet split by cause |
| Nodes | 0.72 | 0.90 | ≈0.3 (2.5 + 0.2 × 25 µs) | records streamed for the node thread |
| Beams | 0.52 | 0.52 | ≈0.15 (2.5 + 0.2 × 10 µs) | the sorted hot records (the frame's `jbeamHot`), not file-order `beamUpdate` |
| Centred damping (on in this run) | 0.50 | 0.50 | | |
| **Frame** | 5.2 | 6.7 | **≈1.2–1.6** | |

The design, then: **every car in the same few dispatches a substep** (nodes; beams with
the pair tests and the wheels' rays; hits and the coupling's bodies in small workgroups),
small kernels, no workgroup arrays, hit lists and pair lists in storage. Its cost is
dispatches × (2.5 µs + 0.2 µs × the longest thread's loads), flat in the number of cars
until a dispatch passes about ten thousand threads (§8.1: 2.5 → 5 µs at 32k), so eight cars
cost about what one does. Cars in contact need no contact groups in this structure: their
pairs are more threads in the pairs dispatch, and the specification's owner order is extra
short hit rounds. That part is not built.

### 8.4 Order of work

1. The reused pair list in the wide schedule (−1.5 to −2 ms).
2. The aux pass as its own small kernels (−1 ms); hits by cause (−1 to −1.5 ms in a crash).
3. Hot beam records and streamed node records in the wide passes (−0.6 ms).
4. Native: the wide pipelines without zeroing by default, after each kernel's workgroup
   reads are audited (today `RUBBLE_WIDE_ZERO=0`).
5. Vehicle contact in the wide schedule (pairs across the GPU, ordered hit rounds), in
   place of contact groups: the 8-car target.
6. Chrome at each step.

### 8.5 Step 1 of §8.4 built: the reused pair list in the wide schedule

`ExplicitBeamsGpu::set_wide_list` (`RUBBLE_WIDE_LIST=1`, `vehicle_lab --wide 1 --wide-list 1`;
off by default): after a step's `wide_beams`, a small list kernel a vehicle
(`explicit_beams_wide_list`: the list's tracking, the triangles' throttles, the grid's
counts and scan, the reuse decision, a build's records, the volatile triangles' tests; its
state loaded and saved in storage), then the self split's walk and pairs dispatches, then
`wide_hits` (which records a walk's list in the saved state). `wide_tris` is not dispatched.

Bit for bit against the wide passes' own walk (`--ab-switch widelist`, every state word,
20 frames each way) at t 0.5, 2.5, 4, 6, 8, 20 s; 4.40 / 6.00 / 6.20 / 6.09 / 6.30 ms a
frame against 5.20 / 6.74 / 6.74 / 6.66 / 7.21.

A first version reused the self split's step kernel for the list (two thinned dispatches
of the 470 KB kernel a step): bit for bit, 0.35 ms faster. The small kernel: 0.5–0.9 ms
faster. Kernel size is a cost of its own (§8.1).

**Whole run, one car, native** (40 s, vehicles' GPU ms a step from t 0.5 s, the same
trajectory to every printed figure):

| Path | median / p95 / worst |
|---|---|
| A workgroup a car, self split on (the default, §7.1) | 5.82 / 6.02 / 6.55 |
| Wide passes with the list (`RUBBLE_WIDE_ZERO=0`) | **5.73 / 5.93 / 6.10** |

Per pass at rest (t 40, timed apart): the list, walk and pairs with the beams 2.29 (beams
0.52 of it), hits 1.90, aux 1.60, nodes 0.91, centred damping 0.52 (on in the native
example's settings, off on the page). The list kernel's dispatch is ≈26 µs, the walk and
pairs ≈12 µs: the list kernel is the next thing to cut (its lane-0 decision and the
triangles' geometry belong in threads of their own).

Frames back to back hide a GPU effect that every path pays in a display's frame loop: with
the GPU idle 10 ms before each frame (`AB_GAP_MS=10`), the same frames take 15–20 % longer
(wide 6.28 → 7.40, self split 6.27 → 7.24, one workgroup 8.58 → 9.86 ms).

**Where this leaves the targets.** Not met: one car is 5.7–6.1 ms on either path. What
changed is the structure: the wide schedule now runs the lab's car exactly, its cost is a
sum of passes that are each measured against a model, and it is flat in the number of cars
apart (§8.2). The remaining passes (list, hits, aux, node and beam records) are each 3–8×
over what §8.1 charges for their work; cars in contact still run the old contact groups
(a frame with vehicle contact is not wide yet).

### 8.6 Hard stop (2026-11-15): three attempts in a row under 5 %

The brief's stop rule was reached. Since the hit forces' cache (§7.3, −12 %), three
attempts each gained under 5 % of one car's frame against the best default:

| Attempt | Bit for bit | Gain against the default path |
|---|---|---|
| Wheels beside other phases (§7.4) | yes | −2 to +7 % (none) |
| The wide schedule with the aux pass and the reused pair list (§8.3, §8.5) | yes | 5.82 → 5.73 ms median, 6.55 → 6.10 worst (1.5 % / 7 % of the worst step) |
| Hits packed into vector records (each hit's list words and node records in 9 `float4` loads a phase in place of ≈40 loads; tried, reverted) | yes, t 2.5, 4, 6, 8, 20 s | 6.21 against 6.20 ms: none |

The last one says the load-count model of §8.1 does not explain `selfApply`: removing
about 80 loads a hit changed nothing. What it does not model is atomics: a hit makes
about 80–100 atomic adds, stores and exchanges on storage (the fixed-point accumulators,
the residual's B and D words, the shifts, the contact-force log). Their cost was not
measured; `gpu_cost_model` has no atomic case. That is the first measurement to make
before touching hits again.

**Best result per criterion**

| Criterion | Best | Target |
|---|---|---|
| 1 car, worst step, native | 6.10 ms (wide schedule, opt-in), 6.55 ms (default) | 1.0 |
| 1 car, worst step, Chrome | 6.92 ms (default; the wide schedule is not measured in Chrome) | 1.0 |
| 8 cars crashing, worst step, native | 270 ms (existing splits on, 14 s), 553 ms (default) | 3 |
| 8 cars, 120 steps/s, zero refused | not run | |
| Every phase within 2× its floor | no phase is (§7.5, §8.3) | |
| Bit-for-bit proofs | every change: mountain at 5–6 times, 2- and 8-car contact for the flags, the GPU suites; tiny and wall crash by the suites only | |

**Remaining gap per phase** (one car at rest, the wide schedule, ms a frame, against §8.1's
charge): list, walk and pairs 1.8 / 0.3; hits 1.9 / 0.2–0.4; aux (coupling, wheels) 1.6 /
0.3–0.4; nodes 0.9 / 0.3; beams 0.5 / 0.15.

**Why.** Three causes, in order of size. (1) Cars in contact still run as contact groups
on one workgroup; the wide schedule has no vehicle contact yet, so the 8-car case has not
moved. (2) Each remaining pass is the frame's per-element functions called from a wide
thread, with their original record layouts and, for hits, their atomics; the two rewrites
tried at that level (the list as a small kernel: 0.5–0.9 ms; hits packed: nothing) show the
gain is real where the model applies and absent where it does not. (3) 1.0 ms itself is
under the measured cost of four dependent dispatches a substep (§7.2); 1.2–1.6 ms is what
the model supports for this car.

### 8.7 After the stop: atomics measured, the wide schedule in Chrome (2026-11-15)

**Atomics** (`examples/gpu_atomic_cost.rs`, one 512-lane workgroup, µs a phase per
operation a lane, 4–64 lanes active): a plain load at a scattered address 0.17–0.26, an
`atomicLoad` 0.21–0.24, an `atomicAdd` 0.25–0.28 (the same when every lane hits the same
eight words), a plain store 0.10. So an atomic costs about what a scattered load does, and
loads of one record's neighbouring words cost almost nothing: the unit of §8.1 is a
scattered access, not a load. That is why packing a hit's own words (§8.6) gained nothing:
a hit's cost is its ≈90 atomics and its four nodes' scattered records, about 30 µs a step
of `selfApply`'s 35.

**The wide schedule in Chrome: 12 % faster, once measured fairly.** A first probe read it
1.5 ms slower (7.40–7.65 against 5.92 ms median). That was the probe, not the schedule:
with `timing=phases` the page times every dispatch in a pass of its own, about 240 passes
a wide frame against one for the default's split frame. With the frame as one timed
segment (`timing=step`), headless, the 1-car mountain, 42 s, vehicles' GPU ms a step from
t 0.5 s, three rounds interleaved:

| Path | median / p95 / worst |
|---|---|
| A workgroup a car, self split on | 5.93 / 6.36 / 6.83, 5.91 / 6.35 / 6.80, 5.93 / 6.37 / 7.06 |
| Wide passes with the list | **5.18 / 5.62 / 6.23**, 5.19 / 5.67 / 6.39, 5.19 / 5.63 / 6.43 |

60 steps/s held in every run (14–17 steps refused in the first second).

### 8.8 The wide schedule is the lab's default

`LabOptions::wide` and `wide_list` (on; the page's "fleet-wide passes" and "with the
reused pair list", live; `vehicle_lab --wide 0`). Frames with car-to-car contact, sleep or
surface contact keep a workgroup a car, so more than one car in the lab is not wide yet.
The engine's own default is unchanged (`ExplicitBeamsGpu::set_wide`, off except for
vehicles past 2,048 beams).

Native wide pipelines are now created without workgroup zeroing (`RUBBLE_WIDE_ZERO=1`
zeroes). The audit, from each wide module's WGSL: begin, rebase, beams, end, close and the
centred passes use no workgroup variable; `wide_nodes` and `wide_aux` read the coupling's
header and table after lane 0 wrote them before a barrier (`coupleBegin`, `cplGsClear`)
and sphere copies only where the table lists them; `wide_hits` its two hit flags after
lane 0 set them; `wide_tris` only writes its two; the list kernel is the self split's
family, already unzeroed and written the same way.

Proofs: every state word the same as the self split frame, 20 frames of 34 steps each way
from one snapshot, at t 0.5, 2.5, 4, 6, 8, 10, 20 and 30 s (`--ab-switch wide`); the GPU
suites with the lab on the wide schedule: `explicit_loop_cap` 5, `explicit_beams_bitwise`
12, `explicit_beams_gpu` 5, `explicit_contact_gpu` 13, `explicit_plan_modules` 1,
engine-world `explicit_vehicles` 11, `vehicle_lab_reset` 3, `vehicle_lab_piece_windows` 1,
`vehicle_lab_pose_time` 2, `explicit_drive_world` 1.

**One car, whole run, vehicles' GPU ms a step (median / p95 / worst):**

| | Start of this work | Now |
|---|---|---|
| Native | 6.71 / 6.94 / 7.37 | 5.77 / 5.96 / 6.14 |
| Chrome | 6.81–6.87 / 7.23–8.76 / 7.77–12.29 | 5.19 / 5.63 / 6.43 |

Natively the wide schedule and the self split frame are within 2 % of each other frame for
frame (wall, back to back: 4.33–6.35 against 4.39–6.20 ms); the whole run's worst step is
6.14 against 6.55. The gain is Chrome's.

### 8.9 The aux pass's work on the list and hits kernels' lanes (lab default)

The aux dispatch a step existed for order only: step s + 1's coupling phases and its
wheels' rays and bars read what step s's node phase left, and its tyres what step s's hits
left. With `set_wide_fused` (`LabOptions::wide_fused`, the page's "coupling and wheels in
those passes", live; `vehicle_lab --wide-fused 0`) they run on the list kernel's lanes
beside step s's list phases (the coupling's two folded phases on the shapes' and nodes'
lanes, the rays and bars on the top lanes) and at the end of step s's `wide_hits` (the
tyres). The aux dispatch remains for step 0 and the coupling's end.

Bit for bit against the aux dispatch (`--ab-switch fused`) at t 0.5, 2.5, 4, 6, 8, 20 s,
and the whole wide default against the self split frame at t 0.5, 2.5, 4, 6, 8, 10, 20,
30 s (4.15–6.09 against 4.40–6.18 ms a frame). Natively 0.25–0.47 ms a frame faster than
the aux dispatch (3.97 / 4.25 / 5.60 / 5.95 / 5.85 / 6.11 against 4.24 / 4.49 / 6.07 /
6.28 / 6.16 / 6.36); Chrome 5.08–5.10 / 5.50 / 6.05–6.23 against 5.23 / 5.71 / 6.83
(median / p95 / worst). Suites green with it on (`explicit_loop_cap`,
`explicit_beams_bitwise`, `explicit_plan_modules`, `explicit_vehicles`, `vehicle_lab_reset`,
`vehicle_lab_piece_windows`, `explicit_drive_world`).

**One car, whole run, vehicles' GPU ms a step (median / p95 / worst):**

| | Start of this work | Now |
|---|---|---|
| Native | 6.71 / 6.94 / 7.37 | 5.57 / 5.76 / 6.00 |
| Chrome | 6.81–6.87 / 7.23–8.76 / 7.77–12.29 | 5.08 / 5.49 / 6.05–6.23 |

Per pass at rest now (timed apart): the list kernel with the walk and pairs 2.6, hits with
the tyres 2.4, nodes 0.9, beams 0.5.

### 8.10 The list kernel split by part; bucket counts only on building steps

Time-only switches in the list kernel (since removed), free fall / at rest, ms a frame:
the coupling's and wheels' work fused there costs ≈0.5 (it was ≈1.5 as a dispatch of its
own); the grid's bucket counting ≈0.3, on every step, though only a step that scans the
grid (a build, or a walk without the list) reads the counts and ranks. The counting now
runs inside that branch (one more barrier on those steps). Bit for bit against the self
split frame at t 0.5, 2.5, 4, 6, 8, 10, 20, 30 s. Its gain is inside the run-to-run spread
of this harness (the same build's frames at rest read 5.6 and 6.1 ms in two runs an hour
apart), so no number is claimed for it.
