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

### 8.11 Verification closed, and cars in contact across workgroups as the lab's default

**Suites on the final kernels** (default switches, the lab on the wide schedule):
engine-webgpu `explicit_loop_cap` 5, `explicit_beams_bitwise` 12, `explicit_contact_gpu`
13, `explicit_pair_reuse_gpu` 2, `explicit_drive_gpu` 3, `showcase_car_gpu` 6 (1 ignored),
`explicit_plan_modules` 1, `explicit_beams_gpu` 5, `explicit_rigid_gpu` 6, `vv_grid_gpu` 2,
`debris_retire_gpu` 1, `explicit_beam_order_gpu` 3, `group_split_gpu` 2, `self_split_gpu`
8, `self_split_gate` 1; engine-world `explicit_vehicles` 11, `vehicle_lab_reset` 3,
`vehicle_lab_piece_windows` 1, `explicit_drive_world` 1; engine-metal `vehicle_parity` 4,
`explicit_parity` 7, `explicit_drive_parity` 1. All pass.

**Cars in contact.** The existing split group path (`set_group_split`: a member's step a
workgroup, the contact tests across the GPU) with self-collision split is now the lab's
default (`LabOptions::group_split`, the page's "touching cars across the GPU", live;
`vehicle_lab --group-split 0`), gated in browsers on its modules being compiled
(prewarmed, deferred, for any fleet of more than one car). The engine's default is
unchanged (`RUBBLE_GROUP_SPLIT`).

Bit for bit against a workgroup a group (`--ab-switch groupsplit`, every state word, 20
frames each way): two cars at t 1, 2, 3 s (24.4 / 25.6 against 26.4 / 28.3 ms a frame);
eight cars at t 1, 3, 6 s (**23.8 / 18.7 / 70.8 against 44.7 / 48.7 / 106.4**).

**Eight cars, whole 40 s run, native, vehicles' GPU ms a step:**

| | median / p95 / worst |
|---|---|
| Start (a workgroup a group) | 312 / 534 / 553 |
| Now (split) | 264 / 474 / 490 |

The gain is in flight and the crash (2–3×); the run is dominated by what follows: from
t ≈ 10 s the eight cars lie in one pile, one contact group of eight, whose contact pass
past four members is still one workgroup (owner after owner, `contact-groups-design.md`
§5.3), and its cost climbs from 224 ms at 20 s to 480 ms at 40 s with the cars at rest
(not investigated: more held contacts as the wrecks settle is the likely cause).

**Eight cars in Chrome** (headless, 25 s of wall time, `timing=step`): the page loads in
64–74 s, runs 7–12 steps/s (the simulation reached t ≈ 4 s), 42 ms median and 103 ms worst
a step, about 1,000 steps refused. 120 steps/s is not held; criterion 2 fails there by
the same cause.

### 8.12 The eight-car pile: what its frame is (measured), and one thing that does not fix it

Ablations in the pile (`vehicle_lab --cars 8 --ab 12,30 --ab-ablate …`, a workgroup a
group, 20 frames from one snapshot, ms a frame):

| t (s) | Frame | Without the vehicle-contact pass (512) | Without the members' self-collision tests, the levels or the tests of the small-group pass |
|---|---|---|---|
| 12 | 305–308 | **39.5** | 305–307 (those masks belong to the small-group pass) |
| 30 | 470–472 | **38.7** | 470 |

So **87–92 % of the pile's frame is the contact pass of a group past four members**
(`vvPass`: owner after owner, each owner's listed triangles against every partner's
contact points in its box, on one workgroup), about 1 ms an owner a substep at 12 s and
1.6 ms at 30 s. At 12 s the state after 20 frames is the same word for word with the pass
left out: the cars lie within each other's frame boxes without touching, and the pass
finds nothing. Everything else for eight cars on one workgroup is 39 ms.

Tried: `VV_PART` 4 → 8, so a group of eight takes the small-group pass (candidates in a
grid, a pair list, levels). Its buffers are laid out for four owners in five places
(`vvRunSmall`, `vvHitSmall`, `vvSortAt`, `vvRankAt`, `vvOrderAt`, and the plan's sizes); the
loop caps stopped the first run at once. With those generalised it runs on one workgroup
but slower in flight (128 against 45 ms a step: each of eight members steps on 64 lanes),
and with the split group path on it ends in non-finite state at t 4.5 s (that path's
scratch also assumes four owners). Reverted. The fix is the one §8.3 names: the contact
tests of any group as threads across the GPU from a grid, not a larger side-by-side group.

### 8.13 The larger groups' contact pass: candidates streamed (on by default)

In `vvPass` a triangle's candidate loop loaded, for every candidate before its box test,
the node's record (for its vehicle), that vehicle's flags and the node's velocity: three
scattered loads, about two million times an owner a substep in the pile. With
`PERF_VV_STREAM` (perf flag 64, on by default; the page's "contact candidates streamed",
live) the gather writes the candidate's travel over the step, its vehicle and that
vehicle's anti-clip into a record beside it (the group's slice of the small groups' sort
records, unused by a larger group), and the loop reads the two streams: the same
arithmetic of `vvInBox` on the same values (`vvInBoxTravel`).

Bit for bit (eight cars, `--ab-switch perf=64`, every state word, 20 frames each way) at
t 2, 4, 6, 9, 12 s; 47.9 / 79.5 / 107.5 / 283.3 / 296.9 ms a frame against 48.1 / 88.4 /
116.3 / 322.3 / 355.0 (−10 to −16 % once the cars are in one group). Suites with it on:
`explicit_loop_cap`, `explicit_contact_gpu` 13, `group_split_gpu` 2 (its column of six is a
larger group), `vv_grid_gpu`, `explicit_beams_bitwise` 12, `explicit_vehicles` 11, engine-metal
`explicit_parity` 7.

It is not the fix: the loop still visits every listed triangle against every candidate
(≈2 M visits an owner a substep, one core). The candidates in a grid, as the small groups
have them, would cut the visits about sevenfold (`fleet-benchmark.md`: 11.1 M → 1.5 M), and
the tests across workgroups would spread what is left; neither is built for groups past
four.

### 8.14 The larger groups' contact pass with a grid (on by default), and what is left in it

`PERF_VV_BIG_GRID` (perf flag 128, on by default; the page's "contact candidates in a
grid", live): in `vvPass` each owner's candidates are counted into the workgroup's 1,024
buckets, scanned and copied in bucket order (`vvBigGridBuild`: the small groups'
`vvGridBuild` for one owner), and a triangle visits the cells of its box padded by the
most `vvInBox` pads it (`vvTriangleCandsBigGrid`), each bucket once; a box over more than
64 cells visits every candidate as before. The cell is 0.75 m (any size gives the same
candidates; the self-collision cell left most contact triangles over the 64 cells).

Bit for bit (eight cars, `--ab-switch perf=128` against the streamed loop, every state
word) at t 2, 4, 6, 9, 12 s. Time, ms a frame, grid against streamed: 52.6 / 104.1 /
245.2 / 238.9 against 48.1 / 108.5 / 289.1 / 300.0: 9 % slower in flight (the build's
barriers an owner, few candidates), 15–20 % faster in the pile, where the worst frames are.
Suites with it on (`perf-verify.sh 193`): `explicit_loop_cap`, `explicit_contact_gpu` 13,
`group_split_gpu` 2, `vv_grid_gpu` 2, `explicit_beams_bitwise` 12, `explicit_vehicles` 11.

Inside the pass with the grid (t 12 s, ablation masks of the same snapshot): the frame 249,
without the pass 39.6, without the triangles' loop (2048) 59, without the candidates'
gather (4096) 70. **The triangles' loop is still ≈190 ms a frame** (0.7 ms an owner a
substep): in the pile the candidates that remain are really inside the triangles' boxes,
and each takes the anti-clip test (`vvFind`: the held sides, the neighbours' vote) on one
core. That is work to spread, not to cull: the pairs listed and tested a thread each
across the GPU, as the small groups' split path does (`contact-groups-design.md` §5), for
groups of any size.

Eight cars at t 12 s, a workgroup a group: 355 ms a frame at the start of this section's
work, 239 now.

### 8.15 A frame by dispatch kind, closing to 100 %; groups of five to eight on the split path (2026-11-16)

**The measurement.** `vehicle_lab --ab T --ab-profile` (`ExplicitBeamsGpu::profile_dispatches`,
any frame path): from the snapshot at T, 10 frames untimed (wall), then the same 10 with
every dispatch in its own timed pass, summed by kind. The table closes: the kinds' sum
plus the time between passes is the timed span (within 1 % on eight cars), and the span
against the untimed frame is what the timing costs (2–9 %). A pass whose indirect
dispatch is empty has no work to stamp on Metal (its stamps land on its neighbours'
work): a pass whose interval holds another's is counted as empty, time 0. Below a
dispatch (the phases between a kernel's barriers) there is no clock; those stay
ablations, which do not add up. `contact_group_counts` prints, per group, each member's
listed pairs, hits, listed triangles and candidates of the last pass.

**Groups of five to eight members on the split path's pass** (`VV_PART` 8: every layout
sized by it; `VV_SIDE` 4: the members a workgroup steps side by side, and the
one-workgroup path's small pass). A group of five to eight was the larger groups' pass,
owner after owner on one core; it now takes the small groups' pass with its tests
across workgroups, its members each on a whole workgroup. Bit for bit against the
one-workgroup path (`--ab-switch groupsplit`, eight cars, every state word but the
contact plan's records) at t 2, 3.7, 5, 8, 12, 16 s. The plan's records are outside the
verdict: a vehicle's "big" flag is whichever vehicle overflowed a broadphase bucket
first, and differs between two runs of one path (`--ab-same`); the pairs are the same.

What the first table showed (eight cars; one group of eight from t ≈ 2 s, while the cars
still fall): at t 3.7 s the levels' dispatch was 87 % of a 164 ms frame with 17 hits in
the whole group; in the pile the tests' walk was 55 % of 92 ms. Three changes from it,
each bit for bit at the same times:

1. **The walk's parts by the group's threads.** A triangle's cells or candidates were
   split in `LANES / triangles` parts (one workgroup's lanes: 1 part in a pile) although
   the walk runs on 16 workgroups; now `16 · LANES / triangles`. Tests' walk 51 → 31 ms.
2. **The levels once, and only those in use.** Every lane recomputed every owner's level
   (`vvLevel`, k³) for every owner of every level, and the loop ran k levels of bucket
   builds and residual phases whatever the hits. Lane 0 now computes the levels once
   (four bits an owner in `gsUniformLanes`: Metal has no workgroup variable to spare),
   and the loop stops at the highest in use (level 1 always: the throttle updates).
3. **The two-body counts cleared where hits wrote them.** After a pass with any hit,
   `vvSplitClear` zeroed nine words of every node of every member (39,456 scattered
   atomic stores for eight sedans, on one core: the 3.6 ms a substep at t 3.7 s). The
   counts are only written by a found hit (`vvSplitCount`), so the levels now zero the
   listed hits' triangle nodes and points (every node's only when a list overflowed).

Eight cars, ms a frame (untimed wall of the same snapshots; old: the one-workgroup path
at the start of this section):

| t s | old | split, VV_PART 8 | + walk parts | + levels, clear |
|---|---|---|---|---|
| 2.0 (falling, one group of 8) | 52.7 | 37.6 | 35.4 | 15.7 |
| 3.7 (first collisions) | 96.6 | 153.4 | 150.2 | 35.3 |
| 5.0 | 76.5 | 61.4 | 55.6 | 24.8 |
| 8.0 | 198.7 | 101.8 | 79.2 | 49.4 |
| 12.0 (pile) | 237.0 | 86.5 | 74.1 | 54.7 |

The pile's frame by dispatch kind now (t 12 s, 48.3 ms untimed on the profiled
snapshot, 34 dispatches of each a frame):

| kind | ms a frame | µs each | share |
|---|---|---|---|
| tests' walk (16 workgroups a group) | 31.0 | 912 | 59 % |
| front (candidates, grids; the group's workgroup) | 8.9 | 261 | 17 % |
| every car's step (a workgroup each) | 7.7 | 226 | 15 % |
| levels (hits applied; the group's workgroup) | 2.6 | 77 | 5 % |
| tests' listed pairs | 1.9 | 56 | 4 % |

Next from this table: the walk (a triangle's thread scans up to 3,500 candidates in a
pile, 4,160 listed triangles on 8,192 threads, so one part each), then the front.

4. **The tests on 64 workgroups a group** (`SPLIT_TEST_WGS`, was 16: 16 of the GPU's
   cores at most, and one part a triangle in a pile). Bit for bit at the same five times.
   Eight cars, ms a frame at t 2 / 3.7 / 5 / 8 / 12 s: 16.8 / 33.7 / 22.1 / 33.1 / 30.3
   (15.7 / 35.3 / 24.8 / 49.4 / 54.7 before); the first 12 s' worst frame 53 ms (300 at
   the start of this section). The pile's frame by kind now (30.0 ms untimed): tests'
   walk 11.3 ms (34 %), front 8.9 (27 %), the cars' steps 7.7 (23 %), levels 2.6 (8 %),
   listed pairs 2.1 (6 %).

5. **The two-body pass's force snapshot for the listed hits' nodes.** The front took
   every member's every node's force (`vvSnapshot`: a beam sum a node, 4,384 nodes on
   one core, every substep) for a pass that reads it at its hits' nodes alone. Nothing
   between the pass's start and its first applied hit writes a force, so the levels take
   it for the listed hits' triangle nodes and points (every node's when a list
   overflowed). Bit for bit at the same five times; the front 8.9 → 6.9 ms a frame in
   the pile, 5.7 → 3.5 falling; frames at t 2 / 3.7 / 5 / 8 / 12 s: 14.6 / 33.0 / 20.6 /
   31.6 / 28.2 ms.

6. **A small group's member on every lane of its own workgroup.** On the split path a
   member of a group of two to four stepped on its partition's lanes alone (256, 128,
   128 of the workgroup's 512: the step written for members sharing one workgroup), so
   two to four touching cars were the slowest case left: four cars 53 ms a frame
   against eight cars' 28, two cars 25 ms while grouped against 6.8 apart. The same
   step now takes every lane (`gVvWhole`: its items shared over 512 lanes, its
   workgroup arrays in slot 0; the split modules' measurement word 4096 gives the
   partition back). Bit for bit against the groups' own workgroup (`group_split_gpu`,
   `self_split_gpu`; `--ab-switch groupsplit`, two and four cars, t 2, 4, 6, 9 s).
   Four cars 53 → 27 ms a frame (the cars' steps 1,390 → 593 µs a substep); two cars
   grouped 24.8 → 17.4 ms. A group's member's step is still 2.6 times a larger group's
   member's (226 µs: `contactMemberStep`, with the reused pair list). The two are not
   interchangeable as written: `contactMemberStep` for a group of two differs from the
   small groups' step from the first frame with a hit (ignored test
   `whole_workgroup_members_first_difference` was written against that attempt; cause
   not found).

**Chrome (2026-11-16).** Two breaks from this section's changes, found on the page and
fixed: the frame kernel's workgroup storage went 96 bytes past the browser's 32,768
(arrays sized by `VV_PART` that only serve `VV_SIDE` members), and the levels' loop
bound was not uniform to Tint. `explicit-wgsl-check.mjs` (modules only: 5 s) now runs
after every kernel change. Cold compiles on Metal: `explicit_beams_split_member_wide`
209 s (1.44 M inlined lines; used only with self-collision split off: now compiled
last), `explicit_beams_selfsplit_member_vv_wide` 35 s (535 k inlined lines: three
copies of a car's step), `explicit_beams_contact_wide_nosurf` 29–37 s. Until the
member module is compiled a page runs touching cars in the groups' own workgroup: the
user's two-car page held 60 steps a second apart and 21–26 touching.

**Eight cars, the whole 40 s run after this section** (native, vehicle GPU ms a World
step, t ≥ 0.5 s, `--timing step`; 2,383 steps): median 31.5, p95 39.2, p99 40.4, worst
41.7 at t 3.70 s. At the start of the section (§8.11's rerun): 222.6 / 399.7 / — /
417.4. The frame grows slowly through the rest (29 ms at t 12 s, 41 at t 40 s). The
targets (3 ms worst, 120 steps a second) are not met: a step is ten times the target.
Per kind in the pile (t 12 s, µs a substep; a substep's budget at the target is 88 µs
for everything): tests' walk 330, the cars' steps 227, front 203, levels 79, listed
pairs 62. One car's frame (§8.9, untouched in this section) is 164 µs a substep.

**Chrome, eight cars, 200 s of page time (40 s simulated), page visible** (probe,
`timing=step`, vehicle GPU ms a step, t ≥ 0.5 s):

| build | ready | median | p95 | worst | steps refused |
|---|---|---|---|---|---|
| this section's kernels, the fallback's modules compiled first | 74 s (cold) | 34.9 | 73.1 | 175.7 at t 5.9 s | 8,130 |
| the self split's contact modules first (`prefer_self_split`, the lab) | 16 s (the driver's cache warm) | 41.0 | 46.1 | 51.1 | 8,100 |

With the fallback first, touching cars ran in the groups' own workgroup (49–95 ms a
step) for the first 45–55 s after the page was ready. With the self split's contact
modules compiled before the first frame they run across the GPU from the first step.
The second run's later frames are slower (39–46 ms against 33–37): the fallback's
modules were compiling behind it, and the two runs were not controlled against each
other. 120 steps a second is not held: 11–14 steps a second, every other step refused.

### 8.16 Per phase: before, floor, after (2026-11-16)

Native M3 Max, ms a frame of 34 substeps. "Before": the start of this work (§7.1, the
8-car column §8.11/§8.12's first table). "Floor": §8.1's cost model for the phase as a
dependent dispatch across the GPU, 34 × (2.5–5 µs a dispatch + its longest thread's
chain at ≈0.2 µs a scattered access); a phase that needs k dependent sub-phases is k
times that. "Within 2×" is against the floor's upper figure.

**One car** (wreck at rest, the wide schedule; §8.9's per-pass profile; no change in §8.15):

| phase | before | floor | after | within 2× | why not |
|---|---|---|---|---|---|
| self-collision: list, walk, pairs (3 dispatches) | ≈2.9 (in the workgroup) | 0.3–0.5 | 2.6 | no | the list kernel is 65 µs a dispatch: the reuse decision, tracking and throttles are barrier phases of one workgroup, not threads over elements |
| hits applied, with the tyres | 2.1 | 0.3–0.6 (five dependent sub-phases of the residual pass) | 2.4 with the tyres' 0.6 | no | ≈90 atomics a hit on one thread (0.25 µs each, §8.7): 22 µs a hit's chain |
| nodes | 0.6 | 0.1–0.2 | 0.9 with the coupling's fold | no | a node's record is ≈12 scattered loads; the streamed record of §3.4 is not built |
| beams | 0.4–0.5 | 0.1–0.2 | 0.5 | no (2.5×) | a beam's record and two nodes' loads |
| dispatches, frame start and end | 0.5 | 0.1 | ≈0.4 (≈10 dispatches a substep) | no | count of dispatches, 2.5–5 µs each |
| **frame, worst of the 40 s run** | 7.37 | 1.2–1.5 (§7.2) | **6.00** | no | |

**Eight cars in a pile** (t 12 s, one contact group of eight; §8.15's profile, a kind's
34 dispatches):

| phase | before | floor | after | within 2× | why not |
|---|---|---|---|---|---|
| contact tests: triangles' walk | ≈190 (in `vvPass`, one core) | 0.2–0.4 (a thread a listed triangle's cell) | 11.2 | no | 4,160 triangles share 32,768 threads in whole parts; a part still scans up to ≈450 candidates (every candidate of a box over 64 cells) |
| every car's step | ≈40 (one after another) | as one car's frame, 1.2–1.5 (the wide schedule over the fleet) | 7.7 | no | members step a workgroup each, not on the fleet-wide passes |
| contact front (scans, candidates, grids) | in the pass | 0.2–0.4 | 6.9 | no | one workgroup a group: 4,224 triangles' box tests and 4,384 points' gather on one core |
| levels (hits applied) | in the pass | 0.2–0.4 a level in use | 2.6 | no | one workgroup a group; ≈8 barrier phases a level |
| contact tests: listed pairs | in the pass | 0.2 | 2.1 | no | 56 µs a dispatch of 64 workgroups a group, most of them idle |
| **frame, worst of the 40 s run** | 553 | 2–3 | **41.7** | no | |

No phase is within twice its floor. None of the "why not" entries is a physical limit:
each names work that still runs on one core or as barrier phases of one workgroup. The
exception already recorded is the one-car crash frame's floor itself (§7.2: 1.2–1.5 ms,
above the 1.0 ms target on this GPU). What the gap is made of, in order of size: for
eight cars the walk's parts (a finer split, or the candidates culled by the 0.75 m grid
the larger groups' pass has), the members' steps on the wide passes, the front's scans
on the tests' dispatch; for one car the list kernel and the hits' atomics.

Worst and p95 per car count, whole 40 s run, vehicle GPU ms a World step (t ≥ 0.5 s):

| cars | native median / p95 / worst | Chrome median / p95 / worst | steps a second (Chrome), refused |
|---|---|---|---|
| 1 | 5.57 / 5.76 / 6.00 (§8.9) | 5.08 / 5.49 / 6.05–6.23 (§8.9) | the World's 60 held, 14–17 refused in the first second (§8.7) |
| 8 | 31.5 / 39.2 / 41.7 | 41.0 / 46.1 / 51.1 | 11–14 of the World's 60, 8,100 of 10,400 refused |

(These pages ran the World at its default 60 steps a second. The brief's 120 steps a
second was not run: at 31–41 ms a step eight cars hold neither.)

### 8.17 Thirty-two cars: the group is the cost (2026-11-16)

The lab's "cars" control goes to 32 (the first eight as before; further cars in columns
to either side, two high, closing on the first). All 32 are one contact group from the
start (their boxes overlap side by side), a group past `VV_PART` members takes the
larger groups' pass in the group's own workgroup, and that pass is the frame:

| t s | frame, ms | the groups' pass (one core) | the 32 cars' own steps |
|---|---|---|---|
| 1.0 (falling, no hit) | 189 | 181 (95 %) | 7.7 |
| 2.5 | 125 | 117 (93 %) | 7.4 |
| 4.0 (crashing) | 452 | 494 timed (96 %) | 15.5 |

(`--cars 32 --ab 1,2.5,4 --ab-switch groupsplit --ab-profile`; split against the groups'
own workgroup bit for bit at the three times, 185 / 134 / 472 against 332 / 288 / 652 ms.)
The cars' own steps hold at one core a car: 7–15 ms for 32, as `wide-frame-design.md`
§1.1's prototype has it. Contact as a per-group pass does not: its cost follows the
group's size, hits or no hits. This is the measurement behind the plan's first step,
vehicle contact as fleet-wide lists with no group as a unit of execution.

Also here: the split modules' measurement word 8192 steps a small group's members with
the larger groups' member step (`contactMemberStep`), for the ignored diagnostic test
`whole_workgroup_members_first_difference`. Finding so far: for two sedans the pass
finds the same vehicle-vehicle hits but their forces do not reach the accumulators
(zero for the hit triangle's nodes and the point); cause not yet found.

### 8.18 Correction: a larger group's members tested stale triangles (fixed), and what it does to §8.15–§8.17's numbers (2026-11-16)

**The bug (older than this work).** `contactMemberStep` (a member of a group past four
cars, and since §8.15 a member of a group of five to eight on the split path) walks
its self-collision grid with `triangleGridWalk`, which loads each triangle's geometry of
the step from `gBeams[gP.triGeo …]` (`triGeoLoad`) whenever that cache exists. Only the
solo step and the wide passes fill the cache (`triGeoFill`); no member step did. A
member's grid walk therefore tested its triangles as the last solo or wide frame left
them. The small groups' step (`contactMembersStepAt`) computes them live. Found by
stepping two sedans' members with `contactMemberStep` (the split modules' diagnostic
word 8192, since removed): identical while they approached, then the pass found the same
vehicle hits but the members' own self-collision hits were missing.

**The fix.** `contactMemberStep` fills the cache for its tested triangles in its beam
phase when its grid is on (the solo step's loop). With it the two member steps agree bit
for bit (the diagnostic over 12 frames; `group_split_gpu`, `self_split_gpu`), and
GPU == CPU holds (`explicit_beams_bitwise` 12, `explicit_contact_gpu` 13,
`explicit_vehicles` 11, Metal `vehicle_parity` 4). **Results change for any group past
four cars** in a build with the cache: that is the bug's removal, not a regression.

**One member step.** The split path now steps every member with `contactMemberStep`,
whatever the group's size (the partitioned step is no longer in the split modules:
`explicit_beams_selfsplit_member_vv_wide` 535 k → 408 k inlined lines,
`explicit_beams_split_member_wide` 1.44 M → 1.31 M). Bit for bit against the groups' own
workgroup at 2, 4 and 8 cars, t 2, 4, 6, 9 s.

**What the earlier numbers were.** Stale triangles mostly do not overlap their car's
nodes, so a member's self-collision found little and cost little. With the triangles of
the step a member's step in the eight-car pile is 611 µs a substep, not the 227 µs of
§8.15–§8.16; the small groups' step was never slower than the larger groups', it was
doing the work. Corrected:

| eight cars, whole 40 s run, native, vehicle GPU ms a step | median | p95 | worst |
|---|---|---|---|
| §8.15's figure (with the bug) | 31.5 | 39.2 | 41.7 |
| now (fixed) | 55.6 | 65.2 | 70.3 |

The start-of-work figures (312 / 534 / 553) and every "old path" column of §8.11–§8.17
for more than four cars were also measured with the bug, so they understate the old
path's cost too (the one-workgroup path at t 9 s: 200 ms then, 332 ms fixed). The frame
by kind at t 9 s, fixed (45.9 ms): the cars' steps 20.8 (42 %), tests' walk 14.5, front
7.3, levels 4.4, listed pairs 2.2. Four cars: 25–30 ms a frame, the cars' steps 18.4
(70 %). Two cars grouped: 17 ms, apart 7.4.

**What this changes in the order of work.** In a group the cars' own steps are again
the largest part: a member's step is 540–610 µs a substep against 164–218 µs for the same
car solo, because a member walks its self-collision every substep and the solo step
reuses its pair list on 92 % of them. So members on the solo step (with its reused pair
list) comes before, not after, the contact front: it needs the vehicle contact's listed
pairs and the group's scratch out of the regions the solo step's pair list and saved
state use (`triPairsAt`, `gridCountBase`), which is stage 1 of
`contact-groups-design.md` §8.

### 8.19 A group's members on the solo step's pair list (2026-11-16)

§8.18's first item, built in three steps, each bit for bit (`group_split_gpu`,
`self_split_gpu`; `--ab-switch groupsplit` at 2, 4 and 8 cars, t 2, 4, 6, 9 s; the
GPU == CPU suites; Chrome's validation of every module).

1. **Vehicle contact's scratch in regions of its own.** The pass's listed (triangle,
   candidate) pairs were in each owner's self-collision pair region (`triPairsAt`) and a
   group's counters and grid offsets between its dispatches in its first member's
   self-collision grid words (`gridCountBase`, `gridOffBase`): the places a solo step
   keeps its reused pair list and its state between dispatches. Now `vvTriPairs`
   (gNext, 2 · `TRI_PAIR_CAP` words a vehicle), `vvScratch` (gAcc, `GRID_BUCKETS` a
   vehicle) and `vvScratchOff` (gNext, `GRID_BUCKETS` + 1 a vehicle); Params words
   101, 99, 100 (`PARAM_WORDS` 102), fleets of two vehicles or more. 1 MB of gNext more a
   vehicle. No change in time.
2. **The member keeps its pair list over the frame's steps** (`PERF_MEMBER_LIST`, perf
   flag 256, on by default; `--ab-switch perf=256`). `contactMemberStep` runs the solo
   step's list (`pairReuseTrack`, `pairReuseDecide`, build or reuse), its state saved to
   the member's solo-step record after each step's dispatch and loaded before the next
   (`memberListLoad`, `memberListSave`). Everything else of the member's step is as it
   was (the pending throttle reset, the box keys, the node phase's `hits`, the coupling
   and drive before it, the frame's end in `contactMemberEnd`). With the list's tests in
   the member's own workgroup (the split path without self-collision split, and this
   step's first form) a member's step in the pile went 600 → 467 µs a substep for eight
   cars and 543 → 358 for four: the dispatch waits for its slowest workgroup, and with a
   list built on 9–16 % of a member's steps some member of eight builds in three steps
   of four.
3. **With self-collision split, the member's walk and pairs in the fleet's dispatches.**
   The member's step ends at its tests as a solo vehicle's does (`selfSplitBuild`,
   `selfSplitTests`; `selfSplitVehicle` takes members), the walk and pairs dispatches
   test its pairs across the GPU, and its hits are applied in a dispatch of their own
   before the groups' pass (`memberRest`, the vehicles' module with the step word's
   `SPLIT_LEVELS`; the host's `GroupOp::SelfRest`, no workgroups without a group). A
   solo vehicle's hits wait for its next step's dispatch; a member's cannot: the
   groups' pass of the step runs after them in the groups' own workgroup, and that
   order is kept. One dispatch more a substep.

Native M3 Max, the pile at t 9 s, a frame of 34 substeps (`--ab-profile`):

| | before | after |
|---|---|---|
| 8 cars: the vehicles' step, µs a substep | 599.8 | 164.6, and the members' hits 56.5, self pairs 25.5, self walk 37.2 (22 of 34 steps) |
| 8 cars: the vehicles' own work, ms a frame | 20.4 | 9.2 |
| 8 cars: frame, ms (untimed) | 45.1 | 33.8 |
| 4 cars: the vehicles' step, µs a substep | 542.7 | 173.6, and the members' hits 57.8, self pairs 17.2, self walk 33.6 (10 of 34) |
| 4 cars: frame, ms | 24.8 | 15.3 |
| 2 cars grouped (t 2 s): frame, ms | 15.5 | 6.9 |

The first 9.3 s of the run, the explicit frame's GPU ms a World step (t ≥ 0.5 s; median /
p95 / worst): 2 cars 6.37 / 17.2 / 18.6 → 6.31 / 8.1 / 9.7; 4 cars 22.3 / 25.2 / 31.0 →
12.5 / 16.8 / 22.3; 8 cars 32.1 / 46.4 / 57.4 → 22.2 / 34.8 / 50.0. Eight cars' frame
by kind is now the contact tests' walk 14.2 ms (38 %), the front 7.0, the vehicles' step
5.6, the levels 4.2, the tests' pairs 2.2, the members' hits 1.9.

### 8.20 The eight-car benchmark's scene, and where it stands (2026-11-16)

Three things about the scene, not the solver, found on the way to a whole run:

- **The page's stability check stopped the run in Chrome at t ≈ 9.2 s** (twice): a car
  slowed by a crash to 47 kJ, 104 m across with parts left behind, left at 568 kJ after
  being struck by another. Its own fall does not explain the rise and the box rule
  counted nothing else. Natively the fleet's kinetic + potential energy only falls
  (1,388 → −18,417 kJ over 13 s) and so does each car's own (`vehicle_lab`'s energy
  line). The box rule now also needs the read vehicles' mechanical energy together to
  have risen past launches and external work (`stability.rs`; two tests).
- **The mountain's ground followed the first car alone.** The other cars' chunks were
  dropped as it ran on, and whole cars fell without end (median node speeds of
  350 km/h, fastest past 1,000). Each car now keeps its own window of chunks (from
  every vehicle's origin record, read with the first car's nodes), and the shapes'
  room is 1,024 with more than one car.
- **Still open:** a few debris nodes a car leave over the canyon's walls and fall
  without end (the lab's debris time to live reads the first car only). A wide apron
  under each chunk was tried and removed: it became a contact candidate for every car
  (median 56, worst 88 ms) and did not catch them all.

Whole 40 s run with §8.19's members' pair list and the ground under every car, vehicle
GPU ms a World step (t ≥ 0.5 s):

| | median | p95 | worst | |
|---|---|---|---|---|
| 8 cars, native | 37.6 | 60.3 | 62.9 | 55.6 / 65.2 / 70.3 in §8.18 |
| 8 cars, Chrome (page visible, 50 s simulated) | 40.4 | 49.6 | 53.3 | no stop; 11–15 steps a second, 8,219 of 10,500 steps refused |
| 1 car, native | 5.55 | 5.74 | 5.94 | unchanged |

The frame grows through the run although the cars are at rest from about t 16 s
(26 ms at t 4 s, 32 at 12, 40 at 20, 62 at 40): at t 20–30 s the cars' steps are
530–540 µs a substep again and the levels 365 (the members' pair list on or off:
57 against 58–63 ms a frame). Not yet explained; the debris in free fall and resting
cars that never sleep are the two candidates.

### 8.21 Every car's debris; the levels' cost by part; a hit's nodes' beam sums kept (2026-11-16)

**Every car's debris.** The lab's piece labels and debris time to live were the first
car's alone, so another car's torn-off part was never retired and never apart in its
pair list. Each copy now has both, from one beam reading. Eight cars' whole run:
median 37.5, p95 44.4, worst 52.7 ms a step (37.6 / 60.3 / 62.9 before); the pair
lists reused on 93–94 % of car-steps late in the run (86–91 %). A few larger parts
past the time to live's limits (10 kg, 40 nodes) still fall off the canyon's sides.

**The resting pile's frame by kind** (t 30 s, 42.8 ms): tests' walk 12.8, levels 11.2,
front 7.7, the cars' steps 6.2, listed pairs 2.2, members' hits 2.0. Vehicle contact is
34 of the 43 ms; the cars' own steps are 183 µs a substep for all eight.

**Inside the levels** (`vehicle_lab --ab-split-ablate`, the split modules' measurement
word: time only, the same 20 frames from one snapshot with a part skipped):

| skipped | t 3.7 s (17 hits a pass) | t 30 s (10 hits) |
|---|---|---|
| nothing | 40.3 ms a frame | 42.7 |
| the whole pass | 9.9 | 10.8 |
| residual phases (16384) | 26.8 (−13.4) | 37.8 (−5.0) |
| the snapshot (65536) | 33.6 (−6.7) | 41.9 (−0.9) |
| buckets and application (32768) | 35.4 (−4.9) | 40.3 (−2.5) |
| level 1's throttle updates (131072) | 37.1 (−3.2) | 41.6 (−1.2) |
| all four | 20.9 (−19.3) | 33.7 (−9.0) |

Ten to seventeen hits cost 9–19 ms a frame because the levels run in the group's one
workgroup: every lane's work on one core, a hit after another. A hit takes its
triangle, its clip test and its four nodes' forces again in each of four passes over
the list.

**A hit's nodes' beam sums kept for the pass** (perf flag 512, on by default;
`--ab-switch perf=512`). A node's force is its beams' sum plus its accumulators, and
the sum does not change within a pass, but each use took the node's whole beam list
again. The levels now keep it beside the snapshot when they take it (the snapshot's
unused word tags the pass) and later uses load it. Bit for bit at t 2, 3.7, 5, 8, 12,
30 s; ms a frame 35.1 against 39.5 at t 3.7 s and 41.6 against 43.4 at t 30 s, within
noise elsewhere. Small: the rest of a hit's chain (the triangle, the clip test, the
staging) is still taken four times, on one core. The remedy is §8 of
`contact-groups-design.md` stage 3 (a thread a hit across the GPU), not more caching.

### 8.22 The tiny and wall-crash proofs, run on their own (2026-11-16)

The brief's bit-for-bit list names a tiny scene and a wall crash beside the mountain
and the 2- and 8-car contact A/Bs. Until now they were covered only inside whole
suites. Run separately at commit `0e698895` (release build, under the GPU lock, a clean
worktree), each twice: with the host's default perf flags and with
`RUBBLE_PERF_FLAGS=0` (every switch of this work off):

| scene | test | default flags | every switch off |
|---|---|---|---|
| wall crash | `engine-webgpu explicit_beams_bitwise sedan_alone_and_into_a_wall_bitwise` | pass | pass |
| tiny | `engine-webgpu self_split_gpu tiny_split_is_bit_for_bit` | pass | pass |

Both runs of the wall crash print the same lines: the idle sedan 40 frames
deterministic and equal to the frozen reference, the wall crash 263 frames
deterministic and equal to the frozen reference, its hits in the baseline's one-lane
order in 229. Both runs of the tiny scene: 2 frames of 4 steps bit for bit. So on these
scenes the paths with the switches on and off both equal the frozen reference, hence
each other. Limits: one run each; the tiny test is two frames of four steps; the commit
predates the fast-check tooling (no kernel change after it on this branch).

### 8.23 One car's substep: the passes dissected on GPU timestamps; two small switches; three ideas that do not pay (2026-11-16)

**The tool.** `vehicle_lab --ab-perf-masks M1,M2,…` replays 20 frames from one snapshot
with the perf flags XOR each mask, three times round, and prints per mask the wall
median and fastest tenth and (on the wide schedule) the GPU ms a frame of every pass
from timestamps, each dispatch a timed pass (`profile_wide_frame`, now with the list
kernel, the walk and the pairs apart). Wall time moves 5–40 % with other load on the
machine; the per-pass timestamps repeat to 0.01 ms. Time-only switches in a kernel
(skip a part, or run an idempotent part twice) give a part's cost; they were removed
after use.

**What one car's frame is made of** (wreck at rest, t 20 s, 13–16 hits a step, native M3
Max, ms a frame of 34 substeps, each dispatch timed apart: the passes sum to about 7 where
the untimed frame is 5.9–6.1):

| pass | ms | of which |
|---|---|---|
| hits (`wide_hits`) | 2.48 | the hits applied (`selfApply`) 1.68: beam forces' fill 0.23, the round (two-body force, shifts, first adds) 0.55–0.63, the residual pass 0.79, barriers and the rest 0.06; the next step's tyres 0.50; the dispatch with nothing to apply 0.29 |
| list kernel | 2.08 | the coupling's and wheels' fused work ≈0.5 (wall); tracking, twice for the price of 0.09 (wall); the triangles' geometry 0.11; the rest (state load and save, throttles, decision, an empty dispatch) not split further |
| nodes | 0.88 | |
| beams | 0.55 | |
| pairs | 0.52 | 0.19 is the indirect dispatch with no pair to test |
| centred damping (on in the native lab, off on the page) | 0.50 | |

Three findings that correct §8.1–§8.16:
- **Barriers are free here.** Eight more `sync()` (sixteen barriers) a step in the hits
  kernel or in the list kernel change the frame by less than 0.05 ms. A phase costs its
  longest lane's work, not its barrier.
- **The list kernel has no expensive part.** Its triangles' geometry is 0.11 ms, its
  tracking under 0.1, the volatile triangles' tests nothing measurable. A first reading
  (skip the geometry: −1.4 ms) was the downstream effect of stale geometry (no hits to
  apply), not the fill. The 65 µs a dispatch of §8.16 is mostly work it was given (the
  fused coupling and wheels) and the dispatch itself.
- **The hits' cost is the round and the residual pass at 13–16 hits, not atomics alone.**
  The merge (36 atomics a hit) is 0.1 ms; the round is 0.55.

**Kept, bit for bit, on by default**

*Hits a (hit, role) a lane* (perf flag 1024, `PERF_HIT_ROLES`, with flag 1;
`selfResidualRoles`, `hitRolePhase`, `hitCombinePhase`). The first pass's adds, each
node's acceleration (from A, then A + B), the residual's and the total's shares run on
the (hit, role)'s own lane; the two steps that combine a hit's four nodes (the residual
r, the total with its guard) on the hit's lane, from four role records kept after the
beam forces (20 words a hit). The total's difference goes straight into the
accumulator: nothing reads A or B after the guard's accelerations, so the D words and
the merge phase are not needed (the same integers added). Seven short phases in place
of three long ones and the adds in the round. The hits pass 2.47–2.48 → 2.26 ms a frame
(−0.21, timed apart; −8 % of the pass, −3.5 % of the frame); wall at rest 5.49 → 5.38 and
6.14 → 6.00 (median of 60 frames), nothing in free fall (no hits).

*The tested triangles' geometry beside the beams* (perf flag 2048, `PERF_WIDE_GEO`;
`wideGeoThread`, `triangleFinishPeek`). A thread a triangle on the beams' dispatch
(which already had the threads: they returned at once) in place of the list kernel's
lanes; the same triangles (those the throttle update will leave tested, read ahead
without writing). The list pass 2.05–2.08 → 1.93–1.97 (−0.11), beams unchanged
(0.546 → 0.543). Wall: inside the noise.

Both together: wall median 6.02 → 5.79 (t 6 s) and 6.21 → 5.81 (t 20 s) in one run of the
mask harness; `fast-check.sh perf=3072 cars1`, 20 frames: 5.83 → 5.58 (t 6 s),
5.65 → 5.57 (t 20 s), 4.31 = 4.31 (free fall).

Proofs: `fast-check.sh perf=1024` and `perf=2048`, 20 frames each way, every state word
the same on all eleven fixtures (one car at t 2.5, 6, 20 s; two cars at 2, 3.7, 8, 16 s;
eight cars at 2, 3.7, 8, 16 s); the suites, Chrome's validation and the whole runs are
below.

**Tried, not kept**

- *Strict arithmetic's opaque zero from a thread-local set at entry* (in place of the
  uniform `gP.vehicles >> 31` at every strict operation; the suspicion was a uniform
  load per operation). Not faster (5.61 against 5.69, 5.77 against 5.79 ms a frame) and
  **not bit for bit**: at t 20 s 103 node words and 114 beam words differ after 20
  frames (the first, node word 5304: 0x4135f235 against 0x4135f244). The compilers see
  further through a local than through the uniform. Removed.
- *The list kernel's parts moved off its workgroup*: only the geometry was worth trying
  (above); the rest is under 0.1 ms each.
- *Fewer barriers by fusing phases*: nothing to gain (barriers are free).

**Per pass: before, floor, after** (one car at rest, ms a frame, timed apart; floor as
§8.16)

| pass | before | floor | after |
|---|---|---|---|
| hits applied, with the tyres | 2.48 | 0.3–0.6 | 2.26 |
| list kernel | 2.08 | 0.3–0.5 with walk and pairs | 1.97 |
| pairs | 0.52 | | 0.52 |
| nodes | 0.88 | 0.1–0.2 | 0.88 |
| beams | 0.55 | 0.1–0.2 | 0.55 |
| centred damping | 0.50 | | 0.50 |

**Stop.** Three attempts in a row gained under 5 % of one car's frame (−3.5 %, −1.9 %, 0 %).
What the dissection says about the next step: no single pass has a cause left that a
schedule change removes. The hits' round and residual pass cost 0.1–0.2 µs for every
access in the longest lane's chain whether the access is an atomic, a scattered load or
a neighbouring word (the role split cut atomics in series from about 130 to 30 a hit and
gained a fifth of what a model that charges only atomics predicts). The remaining
levers are each 0.1–0.3 ms: the round's four accelerations and its position shifts on
the roles' lanes; the tyres (0.5) and the fused coupling (0.5), which earlier attempts
did not move (§7.4); an in-box pair's chain in the pairs dispatch (the triangle's nodes
loaded again by `ownTravelOut` and the hit's record, which the geometry records could
carry). None reaches the 1.2–1.5 ms of §7.2 alone or together.

**Whole run and verification** (both switches off, `RUBBLE_PERF_FLAGS=961`, against the
new default, the same build; one car, 40.2 s, vehicles' GPU ms a step from t 0.5 s,
two rounds interleaved; the scene is this branch's lab, with the mountain's lid, so
the figures are not §8.9's):

| | median / p95 / worst |
|---|---|
| off | 5.05 / 5.45 / 9.13 (the worst is the step at t 0.50 s), 5.06 / 5.44 / 5.64 |
| on | **4.80 / 5.05 / 5.35**, 4.78 / 5.02 / 5.31 |

The rest fixture alone, 60 frames a way: 5.53 → 5.25 ms a frame (median), hits pass
2.34 → 2.15, list pass 2.11 → 1.99, every other pass within 0.003.

With the switches on by default: `perf-verify.sh default` passes `explicit_loop_cap` 5,
`explicit_beams_bitwise` 12, `explicit_contact_gpu` 13, `explicit_pair_reuse_gpu` 2,
`explicit_beam_order_gpu` 3, `explicit_drive_gpu` 3, `explicit_rigid_gpu` 6,
`showcase_car_gpu` 6 (1 ignored, as before), `self_split_gpu` 8, engine-world
`explicit_vehicles` 11, `explicit_drive_world` 1, engine-metal `vehicle_parity` 4,
`explicit_parity` 7; Chrome compiles all 108 modules. **Not run:** the lab page in
Chrome (its two new checkboxes are untested), the CUDA host, more than eight cars.

**Follow-up found, not fixed here.** The lab page sends its perf flags as the sum of its
checkboxes, which clears the flags without a checkbox (256 `PERF_MEMBER_LIST`, 512
`PERF_VV_BEAM_SUM`): the page runs without them. Flags 1024 and 2048 have checkboxes.


### 8.24 The lid, the page's switches, and the whole runs after them (2026-11-16)

**The canyon's lid.** A slab across the canyon on its walls' tops (a collider of
density 1, which the scene renderer does not draw). Parts thrown over a wall had no
ground and fell without end; one such node set its car's contact reach and pair-list
rebuilds. With the lid no node passes 25 km/h late in the run.

*Replaced 2026-11-17:* on the rolling terrain the lid was an invisible ceiling the cars
met in mid-air (one car, 40 s: four touches, the first at 1.25 s right off the lip; at
13.3 s its spin fell from 15.6 to 4.3 rad/s while it slid along the lid). The canyon now
has no ceiling: its walls go on undrawn above their drawn rock to 500 m over the fall line
(`mountain.rs` `WALL_TOP`). `LAB_NO_LID` is gone.

**The page's switches.** The page sends its perf flags as its checkboxes' sum, so a
switch without a checkbox was off on the page: the members' pair lists (256, §8.19) and
the contact hits' beam sums (512, §8.21) never ran in Chrome until they got theirs
here. Every Chrome figure for more than one car before this section is without them.

**Whole 40 s run, vehicle GPU ms a World step (t ≥ 0.5 s):**

| | median | p95 | worst |
|---|---|---|---|
| 8 cars, native, before the lid (§8.21) | 37.5 | 44.4 | 52.7 |
| 8 cars, native, with the lid | 26.7 | 28.1 | 32.0 |
| 8 cars, Chrome, with the lid, the two switches off on the page | 35.2 | 39.6 | 47.1 |
| 8 cars, Chrome, every switch on (51 s simulated, no stop) | 23.5 | 25.8 | 32.4 |
| 1 car, native, §8.23's two switches on / off (its agent's runs) | 4.80 / 5.05 | 5.05 / 5.44 | 5.35 / 5.64 |

Chrome still takes about 15 World steps a second with eight cars (8,069 of 10,950
steps refused). Debris retirement is refused by the engine while vehicle contact is on
(`set_retired`: "its contact groups do not skip retired nodes"), so with several cars
only the piece labels take effect; the lid is what keeps debris from running away.

**The fast loop** (AGENTS.md, "Checking a kernel change"): `fast-check.sh` on 18 saved
states (1, 2, 4, 8, 32 cars) in place of driving up to each moment; a cache for each
kernel compile; a `fast` build profile. A rebuild after a kernel edit 91 s → 5 s (a
small file) or 25 s (`explicit_beams.slang`); a bit-for-bit check of four eight-car
moments 4–6 minutes → 5 s.

### 8.25 Vehicle contact's state by vehicle; its front and tests as dispatches over the fleet (2026-11-16)

`contact-groups-design.md` §8, stages 1 and 2 (what is built: its §8.6; stage 3, the
levels, is not: its §8.7). Perf flags 4096 (`PERF_FLEET_STATE`) and 8192
(`PERF_FLEET_FRONT`, with 4096), both on by default, each a checkbox on the lab page.
Kernel `explicit_beams_fleet.slang`; modules `explicit_beams_split_fleetfront` and
`explicit_beams_split_fleetplan` (four variants each; 116 explicit modules).

**Proofs** (native M3 Max; the GPU shared with another session throughout):

- `fast-check.sh perf=12288` (both against neither): 18 fixtures (1, 2, 4, 8, 32 cars)
  and the four pre-lid four-car fixtures at 20 frames, every state word the same.
- `vehicle_lab … --ab 2,3.7,6,9 --ab-switch perf=…`, 2, 4 and 8 cars: `perf=8192` (both
  against stage 1 alone), `perf=4096` (both against neither) and `perf=4096` with
  `RUBBLE_PERF_FLAGS=8129` (stage 1 alone against neither): 36 lines, every one "state
  bit for bit: true". `--ab-switch groupsplit` at eight cars: true at the four times.
  32 cars, `--ab 1,2.5,4`: true (one group of 32: the larger groups' pass either way;
  two words of the plan's records differ at t 1, outside the verdict as before).
- `fleet_contact_gpu` (new: two sedans head on, a column of six; each switch setting,
  with and without self split), `group_split_gpu` (3), `self_split_gpu` (8): pass.
- `perf-verify.sh default …`: `explicit_loop_cap` 5, `explicit_beams_bitwise` 12,
  `explicit_contact_gpu` 13, `explicit_pair_reuse_gpu` 2, `vv_grid_gpu` 2,
  `explicit_vehicles` 11, Metal `vehicle_parity` 4: PASS.
- Chrome: "all 116 modules compile in Chrome".

**Eight cars, t 9 s, ms a frame of 34 substeps by dispatch kind** (`--ab-profile`; the
scene with the lid, §8.24):

| kind | before (both off) | after |
|---|---|---|
| front (one workgroup a group) | 6.81 | scans 0.29, owners' parts 0.61, triangles and points 0.89, grids 0.46, sorted copy 0.32: 2.57 |
| tests' walk | 4.09 | 1.49, and its prefix 0.27 |
| tests' listed pairs | 1.80 | 0.52, and its prefix 0.30 |
| levels (a workgroup a group) | 2.81 | 3.29 |
| the cars' steps, their self-collision walk, pairs and hits | 8.86 | 8.89 |
| **frame, untimed** | **23.8** | **16.3** |

A/B from one snapshot, on / off, ms a frame (`perf=4096`: both against neither):

| cars | t 2 | t 3.7 | t 6 | t 9 |
|---|---|---|---|---|
| 2 | 7.41 / 7.83 | 11.48 / 11.97 | 10.88 / 11.56 | 10.03 / 11.06 |
| 4 | 8.21 / 8.34 | 17.08 / 18.10 | 17.32 / 18.71 | 12.81 / 14.92 |
| 8 | 12.71 / 14.87 | 19.97 / 23.98 | 15.51 / 22.72 | 16.28 / 23.66 |

Stage 1 alone is within noise of the groups' slots (eight cars at t 9: 22.0 against
23.6); the gain is stage 2's. The first 9.3 s of the eight-car run, the explicit
frame's GPU ms a World step (t ≥ 0.5 s): median 21.7, p95 25.1, worst 31.6 with both
off; 14.8 / 20.7 / 23.8 in the `perf=8192` A/B run (both on from t 2 s). Not measured:
a whole 40 s run, and Chrome's frame times (the page was not opened; only the modules'
compile was checked).

**Thirty-two cars are unchanged**: one group past `VV_PART`, so `vvPass` in one
workgroup, 150 of 166 ms at t 2.5 s (187 / 187, 234 / 234, 271 / 271 ms a frame on /
off at t 2.5, 1 and 4 s). That is stage 3.

**What cost what on the way.**
- A group whose members have not met ran every fleet dispatch for nothing (four cars
  falling: 9.5 against 8.5 ms a frame). The first plan now sizes everything after it
  and writes zeros when no owner is offered a partner: 8.5 against 8.6.
- The walk's parts: 64 workgroups' worth of threads, as a group had, left the walk at
  8.2 ms a frame for eight cars at rest (the t 16 s fixture); 128 gives 3.0, and that
  frame 29.4 → 18.8 ms (256: the same within noise).
- The plan's pair list can hold a pair twice (`contact-groups-design.md` §8.6): hits
  applied twice at four cars. Found by the long harness run, not by the suites. The
  fast rung that catches it: `FIXTURES=$PWD/out/fixtures_nolid LAB_NO_LID=1
  fast-check.sh --make 4`, then the same with `FRAMES=20` and `perf=12288` (the scene
  before the lid, where the duplicate arises 18 frames after t 3.7 s; `LAB_NO_LID` is
  the lab's switch for it).

**Open, not caused by this work.** `group_split_gpu`'s three tests pass in one process;
with a fourth test beside them that also toggles a perf flag frame by frame (tried with
this work's flags, and with flag 256 alone and no fleet kernel in use) the tests
diverge from frames 0 to 5, each on its own device. One at a time
(`--test-threads=1`) all pass. Zeroing the split pipelines' workgroup memory does not
change it. Cause not found; the new test is in a file of its own (`fleet_contact_gpu`).

### 8.26 Whole runs with the fleet-wide front and the one-car switches merged (2026-11-16)

After §8.23 (one car's two switches) and §8.25 (vehicle contact's state by owner
vehicle, its front and tests as fleet-wide dispatches) were merged and checked here
(all 18 fixtures bit for bit for each set of switches; 13 suites; 116 modules validate
in Chrome). Vehicle GPU ms a World step, whole 40 s run, t ≥ 0.5 s:

| | median | p95 | worst | before these merges (§8.24) |
|---|---|---|---|---|
| 8 cars, native | 17.5 | 18.2 | 24.2 | 26.7 / 28.1 / 32.0 |
| 8 cars, Chrome (page visible, 50 s simulated) | 16.8 | 20.5 | 26.9 | 23.5 / 25.8 / 32.4 |
| 1 car, native | 4.77 | 5.03 | 5.30 | 5.03 / 5.33 / 5.66 (the same scene, switches off) |

From the start of this work: eight cars 312 / 534 / 553 native (understated by §8.18's
bug), one car 6.71 / 6.94 / 7.37. The targets (3 ms and 1 ms worst) are not met: eight
cars are 8 times the target, one car 5 times. Chrome takes about 17 World steps a second
with eight cars (7,740 of 10,770 steps refused).

Where the eight-car frame is now (§8.25's profile at t 9 s, 16.3 ms): the cars' own
steps 5.8, the levels 3.3, the front 2.6, members' self-collision hits 1.7, the walk
1.8, listed pairs 0.8, self pairs and walk 1.4. Thirty-two cars are unchanged
(139–270 ms): a group past eight members still takes the larger groups' pass on one
core, which the next stage removes.

### 8.27 Vehicle sleep on the fast paths: what it costs, what it saves, what it loses (2026-11-16)

What was built and its proofs: `explicit-vehicles-in-world.md` §10.7. In short: with
vehicle sleep on, contact groups across workgroups, self-collision split, the fleet-wide
contact front and (for a lone car) the wide passes now run, bit for bit as the workgroup
paths with sleep on; a sleeping car or group is not dispatched; the thresholds are a
live setting (the lab page's "vehicle sleep" row; `vehicle_lab --vehicle-sleep 1
--sleep-speed V --sleep-reach-mm R --sleep-dwell N`). Owner's decision: sleep is
acceptable, not lossless against a run that never sleeps; this section is the
measurement of what each setting loses. Native, M3 Max, the GPU shared with two other
sessions throughout (times are for ranking, not a quiet machine's).

**Before this work, sleep on cost ten times the frame.** With sleep on the split paths
stood down and touching cars ran one workgroup a group: 187 and 199 ms a frame of 34
steps for the eight-car pile at t 20 and 30 s against 18.4 and 19.0 on the split paths
(both with sleep on, from one snapshot); eight cars parked on the flat arena 73.7 against
7.2.

**Awake, sleep on against off** (nobody asleep; the same physics state bit for bit), ms a
frame of 34 steps from one snapshot:

| cars, moment | sleep on | sleep off |
|---|---|---|
| 1, t 2.5 / 6 / 20 s (the wide passes; 40 frames) | 5.26 / 5.79 / 5.62 | 4.68 / 5.56 / 5.39 |
| 8, t 2 / 3.7 / 8 / 16 / 20 / 30 s | 13.46 / 17.67 / 16.50 / 19.23 / 18.57 / 19.00 | 13.28 / 17.37 / 16.25 / 19.03 / 18.35 / 18.84 |
| 32, t 1 / 2.5 / 4 s | 213.2 / 139.3 / 270.0 | 212.9 / 138.5 / 269.5 |

Whole 40 s eight-car run, vehicle GPU ms a World step (t ≥ 0.5 s): median 18.13, p95
18.87, worst 25.15 with sleep on; 17.91 / 18.64 / 24.74 with it off (+1.2 % at the
median). One car pays 4 to 12 % on the wide passes (what would trim it:
`explicit-vehicles-in-world.md` §10.7).

**The eight-car pile on the mountain is not at rest by sleep's rule.** The rule: no node
faster than `speed` along an axis at any step, and no node farther than `reach` from its
anchor, for `dwell` steps; the members of a contact group sleep together. What it sees
of the pile (`vehicle_lab --rest-probe 120`, the solver alone from the saved states,
sleep off; per car the fastest node at a frame's end, and the farthest any node
travelled within a dwell of 0.5 s):

| state | fastest node, m/s (by car) | nodes ever past 0.25 m/s | farthest travel in 0.5 s, mm | nodes past 0.1 mm |
|---|---|---|---|---|
| 8 cars, t 20 s | 44.4, 2.9, 2.4, 3.4, 13.6, 2.3, 1.5, 29.7 | 536, 66, 59, 265, 269, 156, 40, 536 | 7321, 297, 5.8, 64.8, 40.9, 52.7, 24.4, 4500 | 532 to 541 of 548 |
| 8 cars, t 30 s | 2.7, 2.0, 1.5, 21.5, 15.9, 2.7, 1.5, 3.6 | 77, 55, 98, 539, 279, 123, 31, 173 | 95.5, 315, 6.7, 4714, 32.8, 18.9, 9.5, 191 | 530 to 539 of 548 |
| 1 car, t 20 s | 3.6 | 94 | 12.1 | 530 of 548 |
| 8 cars parked on the flat arena, t 5 s | 0.006 to 0.048 | 0 | 0.113 to 0.128 | 2 of 548 |

On the mountain every car has dozens of nodes moving faster than the default bound and
nearly all of its nodes travel more than 0.1 mm in half a second: loose parts of the
wrecks keep moving at metres a second, whole cars creep on the slope, and cars still
slide or tumble long after "rest" (two cars moved 4.5 and 7.3 m in half a second at
t 20 s, another 4.7 m at t 30 s, in the probe's frames). In the never-sleep World run,
from t 25 s seven cars' median node speed is 0 km/h while their fastest nodes are at up
to 41 km/h (most seconds 1 to 10) to t 40 s; the eighth creeps at 3 to 4 km/h and slides
at up to 28 km/h between t 29 and 32 s. The pile is one contact group of eight, so one
moving member keeps all eight awake. Whether that residual motion is
right physics (no static friction in the node contact, parts flailing on broken beams)
is a separate question, not looked into here.

**The loss table, the lab's eight-car mountain run** (whole 40 s World runs, one a
setting; these runs repeat exactly here: the default-threshold run, in which no car
sleeps, has the never-sleep run's energy at every second and its final positions).
Thresholds as speed m/s, reach mm, dwell steps:

| setting | each car first asleep, s | wakes | largest node displacement from the never-sleep run at t 40 s, mm (by car) | kinetic + potential at t 40 s, kJ | World step at rest (t 37 to 40 s), wall ms |
|---|---|---|---|---|---|
| sleep off | n/a | n/a | reference | −16,283.974 | 20.1 |
| tighter: 0.05, 0.02, 4000 and 0.125, 0.05, 2000 | never (not run in the World: a tighter bound holds whenever a looser one does and a longer dwell is reached later, so they cannot sleep where the default never does; in the frozen-proxy sweep below neither slept) | 0 | 0 | as off | as off, plus the awake cost |
| default: 0.25, 0.1, 1000 | never | 0 | 0.0 for every car | −16,283.974 | 20.3 to 20.4 |
| looser: 1, 1, 1000 | never | 0 | 0.0 for every car | −16,283.974 | 20.3 |
| far looser: 15, 400, 1000 | all eight at 24.37 (one group) | 0 | 355.3, 138.4, 27.4, 21,435.3, 66.6, 30.8, 76.6, 544.0 | −16,052.398 | 0.03 |

So on the mountain nothing between the default and a hundredfold looser setting sleeps
the pile, and the setting that does (a car may have nodes at 15 m/s and travel 40 cm in
half a second and still count as still) freezes a pile that was going to move: in the
never-sleep run one car slides 21 m further after t 24 s and the fleet gives up another
232 kJ of potential energy. A sweep of the solver alone from the t 16 s state to t 40 s
(`--sleep-loss`, seven settings from the two tighter ones to 10 m/s, 100 mm, 1000 steps
and 5 m/s, 2 mm, 100 steps) had no car asleep under any of them, each ending in the
never-sleep state to 0.000 mm (that sweep keeps the World's proxies as they were at t 16 s, so a car that
leaves them falls: it shows only that those settings do not sleep the pile).

**The loss table where there is rest: eight cars parked on the flat arena** (the solver
alone from the saved state at t 5 s, 900 frames = 15.3 s under each setting,
`--sleep-loss`; car 0 stands alone, cars 1 to 7 are one contact group):

| setting | first asleep after, s (car 0; cars 1 to 7) | wakes | asleep at the end | largest node displacement from the never-sleep run, mm (by car) | kinetic + potential, kJ | frame, first 30 / last 60, ms |
|---|---|---|---|---|---|---|
| sleep off | n/a | n/a | 0 of 8 | reference | 58.276 | 7.10 / 7.10 |
| tighter: 0.05, 0.02, 4000 | 5.58; never | 0 | 1 of 8 | 0.006; 0 | 58.276 | 7.31 / 6.89 |
| tighter: 0.125, 0.05, 2000 | 2.70; never | 0 | 1 of 8 | 0.033; 0 | 58.276 | 7.32 / 6.88 |
| default: 0.25, 0.1, 1000 | 1.02; 2.38 | 0 | 8 of 8 | 0.034, 0.019, 0.027 ×5, 0.046 | 58.276 | 7.31 / 1.24 |
| looser: 0.5, 0.5, 500 | 0.27; 0.27 | 0 | 8 of 8 | 0.116, 0.104, 0.121 ×5, 0.105 | 58.276 | 7.24 / 1.26 |
| looser: 1, 2, 250 | 0.15; 0.15 | 0 | 8 of 8 | 0.080, 0.075, 0.061 ×5, 0.059 | 58.275 | 1.29 / 1.24 |

One car alone gives car 0's column (asleep after 1.02 s at the default, 0.034 mm). The
default loses under 0.05 mm against never sleeping and no energy to the joule; the
tighter settings do not sleep the row of seven in 15 s (some member always moves past
the reach within the dwell); the looser ones sleep the cars while they are still settling and lose 0.06 to
0.12 mm. (The 1.24 ms is the harness's frame with every dispatch empty; in the World a
fleet asleep is not stepped at all: below.)

**At rest in the World, sleep on (default thresholds) against off**, wall ms a World step:

| scene | awake | asleep | sleep off |
|---|---|---|---|
| flat arena, 1 car (asleep at 6.05 s) | 4.3 to 4.9 | 0.03 | not run |
| flat arena, 8 cars (asleep at 6.05 and 6.32 s) | 7.8 to 8.0 | 0.03 | 7.5 |
| flat arena, 32 cars (asleep at 6.05 and 6.83 s) | 51.1 | 0.03 | 50.5 |
| mountain, 8 cars | 18.6 to 20.4 | never asleep | 18.4 to 20.2 |

**The coupling plan** (`cpl_plan`), GPU ms a World step: 0.03 to 0.05 for 1, 8 and 32
cars on the flat arena; 0.13 to 0.33 for eight cars on the mountain beside a 15 to 18 ms
frame; 0.09 to 0.10 ms beside a 191 to 302 ms frame (0.05 %) for 32 cars on the mountain. Not material, and not changed
(`explicit-vehicles-in-world.md` §10.7 for what an exact skip of sleeping vehicles there
would need).

**The lab's default.** The page's "vehicle sleep" control starts off, as `LabOptions`
and the engine have it (hosts and tests ask for it). Turned on, at the engine's
thresholds (0.25 m/s, 0.1 mm, 1000 steps): parked on level ground the cars sleep 6 to 7 s
after they are set down (a second or two after they stop settling) and the World stops
stepping; on the mountain nothing sleeps and the run costs 1 % more (one car on the wide
passes 4 to 12 %). Left for the owner: whether the mountain pile
should sleep at all. No threshold short of freezing visibly moving wreckage does it; the
other route is the residual motion itself (the flailing parts and the creep), which is a
physics question with its own evidence to gather.

**Commands.** `tools/explicit-2khz/sleep-check.sh` (the first rung); the tables:
`vehicle_lab --mode gpu --measure mountain-perf --cars 8 --car … --rockslide 0
--self-split 1 --timing step` with `--vehicle-sleep 0|1 [--sleep-speed … --sleep-reach-mm
… --sleep-dwell …] --secs 40 [--save-final F | --compare-final F]` for the World runs,
`--scene flat` for the arena, and `--from-fixture F --rest-probe 120 --sleep-loss
"off;0.25,0.1,1000;…" --loss-frames N` for the solver alone from a saved state.

**By part, and a contact fix (2026-11-16, later; `explicit-vehicles-in-world.md` §10.8).**
`vehicle_lab --from-fixture F --rest-parts N [--rest-every K] [--rest-vv 0]
[--rest-coupling 0]`: from a saved state, the solver alone, sleep off. Per car, every K
frames: its main body's (heaviest piece over unbroken beams) centre-of-mass speed and
angular velocity, its body nodes' speed relative to that rigid motion, its loose pieces,
whether each fast body node has another car's node within 15 cm or a node of its own not
beamed to it within twice the contact band, and its kinetic + potential energy. The
machine's GPU was shared with three other sessions throughout (times for ranking).

*The pile on the 50° mountain at t 30 s, as §8.27 had it (state saved before the fix),
3 s:*

| part | measured |
|---|---|
| car 3 | slides: centre of mass 5.8 to 20.9 m/s (the probe's proxies stay where the World left them, so it also falls off them; in the World it slides at up to 28 km/h) |
| the seven others, rigid motion | centre of mass under 0.1 m/s (0 to 8 mm in half a second), under 0.12 rad/s |
| their loose pieces (3 to 5 a car, 0.9 to 25.5 kg) | under 0.09 m/s |
| their bodies' own nodes | 17 to 230 a car past 0.25 m/s within half a second, the fastest 0.7 to 16 m/s relative to the body, all beams intact; 0.4 to 32 J relative to the body, not decaying over 3 s; none near another car, at most 2 near a self-contact |
| vehicle contact off | every number the same: no vehicle contact force acts; the eight are one contact group by overlapping frame bounds only |

*One car at rest at t 20 s (the same state each way), 1 s:*

| | fast body nodes at a line | fastest, relative to the body, m/s | relative kinetic energy, J |
|---|---|---|---|
| before the fix | 3 to 30 | 0.8 to 2.8 | 2.3 to 6.4, for good |
| rigid coupling off (free fall; before the fix) | 27, 6, 7, 5, 2, 2 | 1.7 falling to 0.45 | 14.5 falling to 1.2 |
| after the fix | 0 to 2 | 0.27 to 0.76 | 0.2 to 1.4 |

The rock kept the bodies ringing: the World's static bodies took RoR's contact rule on
BeamNG-profile vehicles (jbeam-semantics-spec.md §37.1, "Fix (2026-11-16)"). A physics
fix, its own commit, changing results for BeamNG-profile vehicles against static and
kinematic World bodies.

*Whole runs after the fix* (eight cars, 40 s, `--rockslide 0 --self-split 1`; vehicle
GPU ms a World step on the shared machine):

| scene | what rests | default sleep | sleep off | vehicle GPU ms a step |
|---|---|---|---|---|
| ledges 50° (the measurements' mountain) | six cars median 0 km/h from t 20 to 24 s, fastest nodes 1 to 8 km/h; car 2 slides on at 30 to 134 km/h; cars 3 and 7 have torn-off parts sliding hundreds of metres below (frame bounds 290 to 567 m) | nothing sleeps; final positions those of sleep off (0.0 mm), energy the same | reference | 25.8 median (sleep on and off alike) |
| rolling gully (the page's default) | nothing: every car slides on at 40 to 140 km/h at t 33 to 36 s; torn-off parts leave the terrain's window and fall (nodes at 500 to 720 km/h) | n/a: the run fails at t 36.05 (non-finite node state); before the fix it fails at t 37.3 (more colliders than the shapes room): the fallen debris, a pre-existing failure | | 30 to 36 wall (no GPU summary: the run failed) |
| ledges at 25° (below the kinetic friction angle) | all eight at rest by t 14 s; 4 J kinetic in all at t 40 s | nothing sleeps in 40 s; final positions those of sleep off (0.0 mm) | reference | 28.5 / 27.8 median |
| one car, ledges 50° (30 s) | at rest on a shelf from t 12 s (0.0 km/h; under 3 J kinetic at t 30 s) | never sleeps | not run | 5.53 median |
| one car, ledges 25° (30 s) | at rest from t 9 s (0.0 to 0.3 km/h; under 0.5 J at t 30 s) | never sleeps | not run | 5.45 median |

*The 25° pile at t 25 s, by part (1.5 s):* centres of mass under 0.014 m/s; 0.05 to
0.58 J relative kinetic energy a car; 0 to 18 body nodes past 0.25 m/s, the fastest
light skin nodes (0.27 to 0.54 kg; median node 1.4 kg) at 0.7 to 2.1 m/s with every
beam intact, under 1 cm of travel; dangling panels (2 of 6 beams) swinging 4 cm in half a
second. Contact groups: one of six, two cars alone.

*Sleep loss there* (`--sleep-loss`, 300 frames = 5.1 s each, the solver alone):

| setting | first asleep after, s (by car) | asleep at the end | largest node difference from never sleeping, mm |
|---|---|---|---|
| off | n/a | 0 of 8 | reference |
| default 0.25, 0.1, 1000 | none | 0 of 8 | 0 |
| 0.5, 0.5, 1000 | car 5: 1.28 | 1 | 2.9 (car 5) |
| 1, 1, 1000 | car 5: 1.09 | 1 | 2.9 |
| 2, 2, 1000 | car 5: 0.53 | 1 | 3.0 |
| 3, 5, 1000 | car 4: 0.97, car 5: 0.53 | 2 | 7.5, 3.0 |

*One car at rest on the 50° mountain's shelf* (after the fix, saved at t 20 s, at rest
since t 12 s; `--rest-probe 60`, then `--sleep-loss` 5.1 s each): its fastest node at a
frame's end 1.47 m/s, 86 nodes past 0.25 m/s within a second, the farthest 7.2 mm in a
dwell; no setting of the sweep above (default to 3 m/s, 5 mm) sleeps it.

**The 25° pile's ringing, by energy term (2026-11-16, later;
`explicit-vehicles-in-world.md` §10.8 "Where the 25° pile's ringing comes from").** From
the t 25 s fixture, the cars' states carried to the CPU reference:

| run (3 s) | car 7: nodes past 0.25 m/s a quarter second | kinetic energy, J |
|---|---|---|
| GPU fleet as saved (`--rest-parts`) | 18 (up to 1.9 m/s) | 0.58 relative to the body |
| CPU, car 7 alone (with or without its anti-clip memory) | 232, 14, 3, 1, then 0 | 3.0 (the carry's jolt), then 0.1 |
| GPU standalone twin of that CPU state | the CPU's, to 0.03 J | 0.1 to 0.4 |
| GPU fleet, vehicle contact off for the probe | 0 (cars 0, 1, 3 collapse: 539 to 545 nodes at 5 to 6 m/s) | 0.09 |
| GPU fleet, any lossless switch off (pair reuse, candidate list, fold, group or self split) | 18: every number the same | 0.58 |
| CPU fleet of cars 0, 1, 3, 6, 7 with vehicle contact | 15 to 29 | 0.6 to 1.4 |
| the same, final-velocity guard on | 17 to 26 | 0.6 to 1.4 |
| the same, position steps' budget × 0.5 (`SHIFT_BUDGET`) | 10 to 28 | 0.6 to 1.4 |
| the same, no position-step budget (`SHIFT_BUDGET=0`) | 19, 13, 16, 14, 14, 2, 1, 1, 3, 1, 4 | 0.06 to 0.15 after 1.5 s; the cars sink into each other, −10 to −80 J of potential energy a quarter |

The CPU fleet's ledger, a quarter second (rule as shipped): vehicle contact's normal force
−84 to −122 J, its friction −5 to −7 J, its position steps +89 to +135 J; self-contact
+5 to +31 J; the rock −3 to −5 J; mechanical energy −8 to +4 J; 13,100 vehicle-contact
hits a quarter. The cars rest on each other: pair gaps (`--pair-gaps`) of 0 between cars
0–3, 0–7, 1–3, 3–7 and 6–7; car 2 is 6 m off.

*Sleep where the cars do rest, longer runs* (25°, default thresholds, 90 s World runs):

| run | asleep | vehicle GPU at rest, ms a World step | loss against never sleeping |
|---|---|---|---|
| one car (at rest from t 9 s) | at t 73.6 s, never woken | 4.9 awake (median); asleep the World skips its step: 0.03 ms wall | largest node 2.1 mm at t 90 s; 3 J of potential energy the frozen car did not lose |
| eight cars | none in 90 s | 28.6 (median, shared GPU) | 0 |

**The gate (the pile at rest ≤ 1 ms of vehicle GPU a step) is not met.** On the 50°
and rolling mountains the pile is not at rest: the physics (slopes past the friction
angle) keeps a car and torn-off parts sliding. Where it rests (25°) the per-node rule
does not sleep a contact group whose members ring under a centimetre or carry a
swinging panel, at any threshold up to 3 m/s and 5 mm. Asleep, a World step costs
0.03 ms (§8.27, the flat arena), so the gate needs the rule, not the dispatch: §10.8 for
the two designs (a group member sleeping inside an awake group; a stillness test a
ringing node or a swinging panel does not fail), each a measured, configurable loss.

### 8.28 Vehicle contact's candidates kept across substeps: bit for bit natively and in Chrome, off by default (2026-11-16, 17)

`contact-groups-design.md` §8.10 (the design, the soundness argument and the limits). Perf
flag 32768 (`PERF_FLEET_KEEP`, with 8192), kernel functions `fk*` in
`explicit_beams_fleet.slang`; a checkbox on the lab page, unchecked. The (triangle,
contact point) pairs the front and the walk find every substep are kept as a Verlet list
with conservative allowances (calm 5 cm, wild 35 cm, between vehicles 5–30 cm chosen at
the build, 30 cm more for vehicles not near each other). The exact tests run on the kept
pairs every pass, and the list is built again when a node or a pair of vehicles may
have moved past its allowance.

**Proofs, natively** (M3 Max; the GPU shared with other sessions):

- `fast-check.sh perf=32768`: the 18 fixtures (1, 2, 4, 8 and 32 cars), the eight
  first-collision fixtures (four and eight cars at t 3, 3.4, 4.2, 5) at 4, 20 and 60
  frames, eight cars at t 9 s, and 12 early fixtures (2, 4 and 8 cars at t 0.5, 0.9,
  1.1, 1.3 s, 60 frames each: the first builds, an owner past its room, the front's own
  path after it). Every state word the same. Also with a build's room cut down
  (a temporary build): the overflow path, its wait and the build after it, bit for bit.
- `fleet_contact_gpu`: two sedans head-on (60 frames) and a column of six (40), with and
  without self split, the flag against the front's search every substep: bit for bit,
  and the list was built and reused (5 builds and 68 passes reused; 6 and 115).
- `perf-verify.sh` with the flag on (the host's default set plus 32768):
  `explicit_loop_cap` 5, `fleet_contact_gpu` 1, `group_split_gpu` 3, `self_split_gpu` 8,
  `explicit_contact_gpu` 13, `explicit_beams_bitwise` 12, `explicit_pair_reuse_gpu` 2,
  `vv_grid_gpu` 2, engine-world `explicit_vehicles` 11, Metal `vehicle_parity` 4: PASS.
- Chrome: all 116 modules compile.

**Reuse and time, from one state each way, ms a frame of 34 substeps, on / off**
(4 frames, after merging main; the on run is read 0.1–0.3 ms slow by the harness
itself: `perf=0`, the same both ways, gives 4.77 / 4.48 for one car):

| fixture | passes reusing the list | on / off |
|---|---|---|
| 1 car, t 2.5, 6, 20 | (no vehicle pairs) | 4.78 / 4.56, 6.24 / 5.98, 6.00 / 5.82 (20 frames: the harness's bias) |
| 2 cars, t 2, 3.7, 8, 16 | 97.1, 96.3, 95.6, 95.6 % | 9.28 / 10.53, 11.06 / 11.31, 10.69 / 11.18, 11.13 / 11.31 |
| 4 cars, t 2, 3.7, 8, 16 | (none at t 2), 94.1, 94.9, 95.6 % | 9.52 / 8.90, 17.63 / 19.28, 14.85 / 16.68, 12.80 / 14.68 |
| 8 cars, t 2, 3.7, 8, 16 | 97.1, 94.9, 94.1, 94.9 % | 13.15 / 14.47, 17.11 / 18.62, 15.39 / 17.31, 15.61 / 20.21 |
| 4 cars, first collisions, t 3, 3.4, 4.2, 5 (60 frames) | 95.2, 94.6, 93.5, 92.5 % | 16.71 / 18.44, 17.24 / 18.83, 17.01 / 18.63, 17.64 / 19.20 |
| 8 cars, first collisions, t 3, 3.4, 4.2, 5 (60 frames) | 95.9, 95.9, 95.3, 95.3 % | 14.61 / 16.10, 20.11 / 21.94, 17.95 / 19.97, 14.88 / 17.06 |
| 32 cars, t 1, 2.5, 4 | (one group past `VV_PART`: not on the fleet-wide front) | 213.1 / 212.7, 139.1 / 139.1, 270.5 / 269.9 |

Eight cars at t 16 s by dispatch kind (20 frames): the walk 4.48 → 0.34 ms a frame
(2 of 34 substeps build), the listed pairs, both prefixes and the sorted copy 1.79 →
0.42; the scans 0.34 → 0.51 and the grids 0.52 → 0.67 (the bound's thread a node and
the kept pairs' tests), owners' parts 0.70 → 0.98. Untimed frame 20.9 → 14.8 ms.

**Whole 40 s run, eight cars, native** (vehicle GPU ms a World step, t ≥ 0.5 s, 2,383
steps): on 14.05 / 16.73 / 26.6 (median / p95 / worst), off 18.68 / 19.53 / 25.7. Both
runs end in the same state (the energies a car and their total, −16,284 kJ, the same).

**Chrome: first not the same run; the cause, found and fixed.** (The earlier
finding: the lab page, eight cars, real-time loop, the first car at rest at y −164.66
with the flag and at y −311 without it, every run.)

- *The tool.* `WebWorld.explicitAb(mask, frames, steps)` (engine-world-wasm): from the
  page's state, the vehicles' frames with the flag on, the state staged
  (`ExplicitBeamsGpu::stage_state`, a MAP_READ copy), the state put back, the same
  frames with it off, staged, put back; resolved to the differing runs of state words
  (`state_differences`; the planning records masked as `vehicle_lab --ab-switch` does).
  `WebVehicleLab.fixture` saves the page's state in the native fixture format.
  `vehicle-lab-record.mjs --probe --ab MASK` drives it; `--locked` runs the page
  frame-locked; `--ab 0 --abSaveAll TAG` saves a state every second; natively
  `vehicle_lab --from-fixture A --compare-fixture B` compares two. The command is in
  AGENTS.md (rung 3).
- *The page in real time is not one run.* Its readbacks land a speed-dependent number
  of steps later, so two runs with the flag off differed (coupling records at t 4 to
  8 s) in one pair of runs and were equal in another. The end positions of real-time runs
  therefore cannot date a divergence.
- *Frame-locked* (each step's readback awaited), the page is one run: flag on and off
  every state word the same at every second to t 20, nodes apart at t 21.
- *The in-page A/B* (20 frames a second) missed it: the difference begins in frame 31 of
  that second. The page's t 20 state replayed natively (`fast-check.sh perf=32768`)
  differs too, so it is not Chrome's: frames 1 to 30 bit for bit, frame 31 differs, at
  its step 27 (`AB_PRE=30` saves the state before that frame; `AB_STEPS` bisects it).
- *Which test.* A build every pass still differs, so it is not the reuse bound. The
  walk's own pairs tested by a copy of `fkPairTry` differ from `vvPairTest`; with the
  gathered position (the run's `E.xyz`) in place of the recomputed one, equal. The two
  are the same source, `nf3(vvRel(vb, va) + x)`, but two inlined copies, and Metal's
  compiler (naga natively, Tint in Chrome) may reassociate: one copy rounded
  `(o_vb − o_va) + x` in another order, a contact point had two positions differing in
  the last bit, and a pair at a test's edge hit on one path only. Native fixtures never
  met it: it needs the origins far apart (t 20 s down the mountain) and a point at an edge.
- *The fix.* `vvRel` is unfusable (`nf3(o_v − o_a)`): every copy rounds the difference
  alone. No other change; the flag's design was sound.
- *The default path (flag off) against main's kernels* (`fast-check.sh kernel`,
  `--ab-kernel-dir` a copy of main's build's `OUT_DIR/wgsl`; 44 of the WGSL modules
  differ in text): bit for bit on 19 of the 20 fixtures, the two Chrome states included.
  The one-car t 20 fixture differs (103 node and 114 beam words), and it differs the
  same way, word for word, with this build's own kernels as the other side: the
  harness's second lab, not the change (the one-car case another agent reported). So
  the fix changed no result on any fixture; where Metal reassociated `vvRel` before it
  did so in the same way in both of the walk's own copies.

**Proofs after the fix.**

- `fast-check.sh perf=32768`: the 18 standard fixtures and the two Chrome states (t 20.0
  and 20.5 s, now in `out/fixtures`: cars8_t20.0chrome, cars8_t20.5chrome) bit for bit;
  60 frames on the Chrome states and on 8 cars t 8 and 16, 4 cars t 8: bit for bit.
- Chrome, the lab page frame-locked, eight cars, the mountain with ledges: flag on and
  off saved every second to t 40 and compared natively: every state word the same at
  all 40 seconds; the first car ends at (−6.33, −356.70, 301.84) both ways. The in-page
  A/B (`--locked --ab 32768`, 20 frames every second to t 40): 40 of 40 bit for bit,
  and the run ends at the same place (the A/B does not perturb the page).
- `perf-verify.sh 49089` (the host's default set plus 32768): `explicit_loop_cap` 5,
  `fleet_contact_gpu` 1, `group_split_gpu` 3, `self_split_gpu` 8,
  `explicit_contact_gpu` 13, `explicit_beams_bitwise` 12, `explicit_pair_reuse_gpu` 2,
  `vv_grid_gpu` 2, `staged_state_gpu` 1 (new: the browser readers equal the blocking
  ones), engine-world `explicit_vehicles` 11; engine-metal `vehicle_parity` 4. Chrome: all 116 modules
  compile.

**Time after the fix, before the worst step's work** (vehicles' GPU ms a World step,
t ≥ 0.5 s, median / p95 / worst): native, whole 40 s eight-car run, on 13.47 / 16.81 /
26.91 (worst at t 1.02), off 18.73 / 19.52 / 22.99. Chrome, the page's own real-time
loop, 170 s: off 17.82 / 20.65 / 26.00 and 17.74 / 20.66 / 25.46, on 15.81 / 19.42 /
30.99 and 15.86 / 19.46 / 30.77 (both at t 28.17). Faster at the median, slower at the
worst.

**The worst step** (`AB_EACH=1`: each frame's time and its builds, reused passes and
passes on today's path; states saved just before, natively and from the page):

- *What it was.* Not a build that keeps its list: a build *past an owner's room*
  (`FK_OVER`, an owner over 32,767 pairs: the cars' first collisions at t 1, a pile at
  t 28 on the ledges and t 30 on the rolling gully), retried every 136 passes. Its pass
  listed pairs it then threw away and walked its tests in the build's padded boxes,
  a thread a triangle: 7 to 10 ms in one pass (Chrome t 28: 31.2 ms frames against
  24 ms beside them; rolling t 30, natively: 42 against 32 ms, every fourth frame).
  On the rolling gully, tumbling cars also built every third pass (a wild node past
  35 cm), a build costing more than the walk it saves.
- *What changed* (behind the same flag; each a choice of path or an order of work, the
  tests the same):
  1. that pass's tests walk the *walk's own* boxes (the triangle's box grown by its
     range, its travel, the owner's fastest candidate's travel and `CLIP_DEPTH`,
     `vvTriangleCandsList`'s), over the build's candidates, a part of a *listed*
     triangle a thread as today's walk; `fkPairTry` decides each pair exactly;
  2. the build's listing stops once an owner is past its room (the list is void);
  3. a build with no fewer candidates than the last one past the room skips its listing
     (it would be past it again);
  4. builds past the room in a row double the next wait, up to 16 times 136 passes;
  5. a list that lasted fewer than 8 passes sends the next 68 to today's front;
  6. no tracking while the passes left are today's path.
- *After* (frames around the same moments): Chrome t 28 attempts 24.7 / 24.3 ms beside
  24.1; native t 0.9 attempts no longer above their neighbours; rolling t 30 attempts
  33.1 against 32.5 ms.

**Proofs at the final kernel** (commit 40a7c4fe):

- `fast-check.sh perf=32768`: 23 fixtures bit for bit; new rung-1 fixtures: eight cars at
  t 0.9 (native, builds past the room), the page's ledges state at t 28 and its rolling
  state at t 30 (`cars8_t28.0chrome`, `cars8_t30.0rolling`); 60 frames on the Chrome
  states and on t 0.9: bit for bit.
- Chrome, frame-locked, flag on and off, every second compared natively: ledges 40 of 40
  bit for bit; rolling gully 36 of 36 (the page stops at t ≈ 36.5 in both runs, flag on
  or off: a read-back map fails as the World passes its coupling's shapes room, which
  the native lab reports at t 37 on rolling ground: "more colliders than the shapes
  room", flag on or off; not this flag).
- `perf-verify.sh default` and `perf-verify.sh 49089`: `explicit_loop_cap` 5,
  `fleet_contact_gpu` 1, `group_split_gpu` 3, `self_split_gpu` 8, `explicit_contact_gpu`
  13, `explicit_beams_bitwise` 12, `explicit_pair_reuse_gpu` 2, `vv_grid_gpu` 2,
  `staged_state_gpu` 1, engine-world `explicit_vehicles` 11: PASS both ways; engine-metal
  `vehicle_parity` 4: PASS both ways. Chrome: all 116 modules compile.

**Time at the final kernel** (vehicles' GPU ms a World step, t ≥ 0.5 s, median / p95 /
worst):

| run | flag on | flag off |
|---|---|---|
| native, ledges, 40 s | 13.34 / 16.09 / 22.65 | 18.51 / 19.34 / 25.29 |
| native, rolling gully, 36 s | 23.81 / 39.61 / 49.02 | 24.06 / 39.24 / 49.10 |
| Chrome real time, ledges, 170 s (off / on / on / off) | 15.87 / 19.22 / 25.21; 15.84 / 19.23 / 25.46 | 17.80 / 20.73 / 25.52; 17.74 / 21.22 / 57.30 |
| Chrome real time, rolling gully, 120 s (off / on / on / off; t 30.6–31.9 s reached) | 18.25 / 35.19 / 43.44; 18.30 / 35.11 / 43.53 | 21.75 / 33.10 / 39.39; 21.48 / 34.93 / 42.94 |

Both native pairs end with the same energies (ledges −16,284 kJ, rolling −85,260 kJ).
(The ledges' 57.3 ms off is one step of one run, not seen in the other three. Rolling
runs in Chrome are 120 s of page time: at 170 s the page reaches t 37 to 40, where the
scene itself passes its coupling's shapes room and the page stops, flag on or off.)

**Default: still off.** Bit for bit natively and in Chrome on both terrains, and faster
at the median everywhere (ledges 28 % natively and 11 % in Chrome, rolling 1 % natively
and 15 % in Chrome); the worst step is now better on the ledges both ways (22.7 against
25.3 ms natively; 25.2 and 25.5 against 25.5 ms in Chrome). On the page's own terrain,
the rolling gully, it is not: natively the worst is a tie (49.02 against 49.10 ms, the
same step at t 30.33) and p95 0.4 ms higher; in Chrome the worst is 43.4 and 43.5 against
39.4 and 42.9 ms and p95 about 1 ms higher. There the cars pile and tumble, the list is
past an owner's room or outlived in a few passes, and most passes run today's front, likely with
the flag's own costs on top (the scans' tracking and the first plan's checks). The rule
for a default is faster at the median and the worst on the page's terrains, so it stays
a switch and the page's box stays unchecked.

**Per car, tried and reverted (2026-11-17; commit 36173635, reverted by d3ac5fe2).** The
next step named above: a car past its room, or a culprit of a list outlived in a few
passes, is *excluded* (its pairs on today's front, for 272 passes, or the overflow wait
doubled) while every other car keeps reusing its list. A pass that reuses the list with a
car excluded (`FK_MIX`) runs the kept pairs' tests and today's front for the excluded
cars' pairs: their points gathered alone, an excluded owner's walk listing into its own
pair region, another owner's tested inline so its kept list stays; a car is back only
through a build; a build's tests walk the walk's own boxes (the new list is for the next
pass). Bit for bit on the 23 fixtures (60 frames on the Chrome and overflow ones), but
slower than the fleet-wide list everywhere, natively (median / p95 / worst ms):

| run | per car | fleet-wide list (kept) | flag off |
|---|---|---|---|
| ledges, 40 s | 14.42 / 19.65 / 25.85 | 13.34 / 16.09 / 22.65 | 17.94 / 19.38 / 25.52 |
| rolling gully, 36 s | 25.56 / 41.07 / 50.81 | 23.81 / 39.61 / 49.02 | 24.43 / 39.17 / 49.21 |

From the page's rolling states, 20 frames: t 20 22.26 against 22.08 ms off, t 25 24.19
against 23.51, t 28 33.53 against 32.95, with 99.6 % of passes reusing the list: on the
gully almost every pair of contact is a piled car's, so a mixed pass is today's front
plus the kept tests and the inline walks beside it (with lists outlived after 8 passes a
build came every 8 passes: 28.6 against 24.0 ms frames at t 25; 32 passes and 272 for the
exclusion brought that down, not below the flag off). The fleet-wide list is kept; the
kernel at the branch's head is 40a7c4fe's, whose proofs and timings are above. What
would be needed for per car to pay: the excluded cars' pairs listed beside the kept list
(a pair region shared, not tested inline) and a build that keeps reading its list in its
own pass; both change today's walk's code, measured first for its own cost.

**Per car, listed beside the kept list: built, bit for bit, slower; reverted (2026-11-18;
commits 14cbaa61 and after on `perf/keep-rolling`, reverted there).** The design named above,
on main after #119, #122 and #123, behind a companion flag (8, `PERF_FLEET_CAR`, with 32768).

- *Today's walk, measured first* (main's kernels, flag off, `--ab-profile`, 10 frames of 34
  substeps, ms a frame / µs a pass): rolling gully t 25 6.19 / 182 (19 % of a 31.0 ms
  frame), t 30 6.71 / 197 (20 %); ledges t 16 2.71 / 80 (14 %). The change to the walk itself
  is one term: `vvPairAdd` adds the owner's first place after its kept list
  (`fkWalkBase`, 0 unless a pass reuses the list with a car excluded); `vvTriangleCandsList`
  is unchanged.
- *What was built.* One predicate splits the pairs in every dispatch of a pass: a pair with
  an excluded car (as owner or as the point's vehicle) is today's front's, every other pair
  the kept list's. A pass that reuses the list with a car excluded (`FK_MIX`): the kept
  pairs' tests for the others; today's gather (those pairs alone), grids, sorted copy and
  walk for the excluded cars' pairs, the walk's pairs listed in the owner's pair region
  after its kept list and tested in the listed pairs' dispatch (no inline tests). A build:
  the padded gather of every pair (its box now also holds every point `flPoint`'s would,
  whatever its speed), the list of the pairs of two cars not excluded, then in the same
  pass the tests of its new list and, a part of a listed triangle a thread, the excluded
  cars' pairs in the walk's own boxes; an owner past its room is excluded at that build
  (FK_BACKOFF passes, doubled per build past it in a row), the others keep their lists. A
  car with a node too fast for any build is excluded instead of sending the fleet to
  today's path. A culprit of a list outlived in fewer than 8 passes is excluded for 68
  passes; a car comes back only through a build; an excluded car's nodes are not tracked.
- *Proofs.* `fast-check.sh` on 22 fixtures (1, 2, 4, 8 cars, ledges; the overflow moment,
  eight cars at t 0.9; the rolling gully at t 20, 25, 28, 30 saved natively with
  `--terrain rolling`): bit for bit with `RUBBLE_PERF_FLAGS=30932929 perf=8` (per car
  against the fleet-wide list) and `perf=32776` (per car against off), 4 frames; 20 frames
  on ledges t 16 and t 30, rolling t 25 and t 30, and t 0.9: bit for bit. Chrome's module
  check found one barrier: the build's plan read a workgroup value inside lane 0's branch
  (`workgroupUniformLoad` from non-uniform control flow, `explicit_beams_split_fleetplan`
  and its variants); read before the branch, fixed: all 124 modules compile. The suites, Chrome's A/B and the 40 s
  runs were not run: the change is slower on every fixture (below).
- *Time* (ms a frame, 20 frames from one state each way, per car / fleet-wide list):

| fixture | attempt 1 (as designed) | attempt 2 (+ the mixed walk culled by its candidates' box) | attempt 3 (+ the mixed walk's parts over the owners with candidates alone) |
|---|---|---|---|
| ledges, 8 cars, t 16 | 17.79 untimed (fleet-wide 14.60, off 16.85) | 19.19 / 15.50 | 19.41 / 15.45 |
| rolling, t 25 | 36.42 untimed (fleet-wide 28.72, off 27.33) | 36.85 / 28.60 | 36.72 / 28.60 |
| ledges, t 30 | | 25.90 / 19.45 | 26.08 / 19.59 |
| ledges, t 0.9 (overflow) | | 13.57 / 11.71 | 13.44 / 11.81 |
| rolling, t 30 | | | 35.18 / 29.32 |

  Three attempts in a row, each slower, none a gain: stopped there.
- *Why.* The per-owner counts of a mixed pass at ledges t 16 (`AB_COUNTS`): every one of
  the eight owners had candidates of today's front (316 to 3,773), with all 528 of its
  triangles listed, and a build every second frame for cover. One car is excluded almost
  all the time (the culprit of the lists the fleet-wide run outlives: its 69 passes on
  today's path in 20 frames), it returns through a build every 68 passes and is excluded
  again at once; in a pile every other car touches it, so every owner gathers its points
  and walks every listed triangle: the walk is today's (2.80 ms, 82 µs a pass, against
  2.63 / 77 off), with the kept tests, the tracking and the builds on top. Culling a mixed
  walk's triangles by the box of the owner's front candidates (attempt 2) and giving the
  walk's parts only to owners with such candidates (attempt 3) changed nothing: those
  candidates are a whole car beside every owner. On the rolling gully the list is now
  kept (90 % of passes reuse it, where the fleet-wide list never builds: every build there
  is past an owner's room), but builds come every 10 passes (65 in 20 frames, wild nodes)
  and each one's tests of the excluded cars in the walk's own boxes cost 7.1 ms a frame in
  the listed pairs' dispatch (0.6 off); the walk stays 7.3 to 7.8 ms.
- *Found on the way: the fleet-wide list's own cost on today's main.* With 32768 the host
  drops `PERF_FLEET_POST` (the levels' throttle updates) and the fused scans
  (`fuseScansOn`), both default since §8.29 and §8.32. Rolling t 25, flag on against off:
  the split levels 8.06 against 7.16 ms a frame, the scans' dispatch 0.37 against none, the
  owners' parts 1.08 against 0.88, the walk the same (6.19): 28.72 against 27.33 ms
  untimed. Ledges t 16: 14.60 against 16.85 (the walk 0.82 against 2.63). So on the
  page's terrain the fleet-wide flag now costs about 1.4 ms a frame before any walk is
  saved; making the kept list compatible with the throttle updates across the fleet and
  with the fused scans is the cheaper next step than per car.

The fleet-wide list (40a7c4fe's) stays as it was, off by default.

### 8.29 Contact groups of any size on the fleet-wide passes; the levels visit the owners with hits; the throttle updates across the fleet (2026-11-16)

`contact-groups-design.md` §8 stage 3 (what is built: its §8.7). Two perf flags, both
on by default, each a checkbox on the lab page:

- **16384 `PERF_FLEET_ANY`** (with 8192): a contact group of any size takes the
  fleet-wide front and tests and its levels in `fleetLevelsAny`, in place of the larger
  groups' pass (`vvPass`) on one core past `VV_PART` (8) members. The levels: the
  group's owners with hits are listed first (usually a few of the group's), and only
  they are visited, on the lanes their hit lists reach; an owner's partners are two
  words of bits by member, and its level (one past the highest earlier owner with hits
  that is its partner or shares one: `vvLevelsSet`'s rule) is relaxed over those bit
  sets until none changes. A small group takes these levels too, unless one of its
  owners lists more than 128 hits (then the small groups' own levels, whose hits are
  ordered by buckets). Params word 105 (`vvRoom`): the run, hit list and two-body point
  slots, eight times the fleet's contact points before, which 32 crashing cars pass.
- **65536 `PERF_FLEET_POST`** (with 8192): every listed triangle's throttle update and
  a later owner's reset, which ran in the group's workgroup with its first level (528
  triangles a car on one core every substep, hits or not), are a thread each in a
  dispatch of their own after the listed pairs' tests (`explicit_beams_split_fleetpost`,
  four variants; 120 explicit modules). (32768 is `perf/fleet-candidates`'
  `PERF_FLEET_KEEP`.)

**Proofs** (native M3 Max, the GPU shared with other sessions through a lock):

- `fast-check.sh` on the 18 fixtures (1, 2, 4, 8, 32 cars): `perf=16384`,
  `perf=65536` and `perf=81920` (both), each fixture "bit for bit: true"; `perf=81920`
  again at `FRAMES=20`, and on the four pre-lid four-car fixtures at 20 frames: true.
- `fleet_contact_gpu`: a column of ten (a group past eight members, asserted) with
  16384 and with both, a column of six with 65536 alone and with the levels for any
  size forced on small groups, each with and without self split: pass, with the
  default flags and with `RUBBLE_PERF_FLAGS=98241` (the defaults before, plus both).
- `perf-verify.sh 98241`: `group_split_gpu` 3, `self_split_gpu` 8,
  `explicit_contact_gpu` 13, `explicit_beams_bitwise` 12, `explicit_loop_cap` 5,
  `explicit_vehicles` 11, Metal `vehicle_parity` 4: PASS. (98241 is now the default.)
- Chrome: "all 120 modules compile in Chrome".

**Native, ms a frame of 34 substeps, on / off** (fast-check, 4 frames each way from one
state; `perf=81920`):

| cars | fixtures (t s) | on / off |
|---|---|---|
| 1 | 2.5, 6, 20 | 5.6 / 4.4, 7.2 / 5.9, 6.7 / 5.9 |
| 2 | 2, 3.7, 8, 16 | 10.2 / 10.0, 11.7 / 12.0, 10.7 / 10.9, 10.4 / 10.9 |
| 4 | 2, 3.7, 8, 16 | 9.5 / 8.9, 17.9 / 18.4, 15.5 / 16.2, 13.3 / 14.2 |
| 8 | 2, 3.7, 8, 16 | 12.6 / 14.0, 16.5 / 18.1, 14.9 / 16.8, 16.7 / 18.8 |
| 32 | 1, 2.5, 4 | **37.2 / 198.7, 30.8 / 131.8, 49.6 / 249.3** |

One car has no contact group, so neither flag runs; the same switch with an unused bit
(`perf=1048576`) reads 6.4 / 4.4, 7.1 / 5.8, 6.2 / 5.9: the "on" side is measured
first from a cold state, a bias of the harness, not a cost. At 20 frames the 32-car
fixtures read 49 / 218, 50 / 175, 51 / 252, eight cars 12.5–19.6 against 13.9–21.3.

**Where the 32-car frame is now** (`--ab-profile`, both flags on, ms a frame): the
levels 35.4 / 21.4 / 32.7 at t 1, 2.5, 4 (one workgroup for the one group of 32: the
levels in turn, three residual phases each, a barrier between), the cars' own steps
6.0–8.3, the walk 1.8–4.1, self-collision 3.5–5.0, the rest of the front 3–4, the
throttle dispatch 0.25. Before, the larger groups' pass was 141–244 ms of it. Eight cars
at t 16: the levels 3.38 → 0.99 ms a frame, the throttle dispatch 0.23.

**Chrome, eight cars** (`vehicle-lab-record.mjs --probe --secs 40`,
`mode=mountain&car=gen&cars=8&timing=step`, page visible, vehicle GPU ms a World step,
t ≥ 0.5 s, about 10 s simulated each):

| | median | p95 | worst |
|---|---|---|---|
| both off | 17.2 | 23.1 | 25.3 |
| both on | 15.5 | 21.8 | 24.4 |
| both on (second run) | 15.6 | 21.6 | 25.1 |
| both off (second run) | 16.5 | 22.7 | 25.0 |

Not measured in Chrome: more than eight cars (the page builds at most eight, whatever
its car count says).

**Default on**: every state word the same, faster natively from two cars up (32 cars
four to five times) and faster in Chrome at eight cars.

**Not built: the levels across the GPU (stage 3c).** At 32 cars the levels are now
two-thirds of the frame, all in the one group's workgroup. What is left, precisely:

1. *The order (b).* The levels are already order-independent in what they compute (a
   level is a set of owners whose touched vehicles are disjoint, so within a level
   nothing depends on the order of owners or hits); what is ordered is the levels
   themselves (level l reads the nodes level l − 1 wrote) and, inside an owner, the
   chain of hits on one contact point (`vvApplyOrdered`'s scan). Both stay.
2. *The levels as dispatches (c).* A dispatch a level and phase (application, three
   residual phases) over every hit of that level in the fleet, a thread a hit, sized
   from a prefix of the hits by level (`flPrefixAt(3, v)` is reserved). A dispatch
   needs its level: the step's three parameter copies are used up as phases, so the
   levels need copies of their own (a level and phase each, past `PARAM_COPIES`) or a
   module a phase reading the level from an argument word the plan writes. With L
   level slots that is 4 L dispatches a substep, empty past the highest level in use,
   and the group's workgroup looping the levels past L (lossless, slow, rare). The
   snapshot and the two-body clears are threads over hits already independent of the
   levels (before the first and after the last).
3. *The levels' rounds* run in the workgroup today (a lane an owner with hits); over
   the fleet they are one workgroup of the plan, rounds bounded by the longest chain
   of touching owners (`EB_MAX_MEMBERS + 2`).

### 8.30 The vehicle lab's drawing: skinning a third of what it was, the same picture (2026-11-16)

The lab's report column "GPU ms a drawn frame" with five generated sedans
(`vehicle-lab.html?mode=mountain&car=gen&cars=5`) read about 14 ms for "skinning and
shadow map", and it did not fall with the render scale. Split into three timers
(skinning / shadow map / the picture), it was the skin compute pass: 13.3 ms, the
shadow map 0.5. Two causes, both removed without changing what is drawn:

* `cutTris` rewrote both drawn index lists of every car on every drawn frame, each
  triangle reading its three vertices' whole weighted entry lists, though the lists
  depend only on the piece labels (written when a piece comes off). `cutCheck` (one
  invocation a car) compares the car's label words with the copy the lists were cut
  by and dispatches the cut indirectly only when they differ.
* A weighted vertex's entries were 16 B (u32 node, three f32), 4 to a 64-byte slot.
  They are now the `.rskin`'s own 8 B (u16 node, three i16 steps of 2^-13 and 2^-9),
  8 to a slot, scaled in the shader (powers of two: exact). The bind buffer is
  267 → 157 MB a car (`modFootprint` follows: 1.26 / 1.28 a rskin byte).

Skinning runs once a drawn frame, only on drawn frames (`cars.update` is called from
`draw()` alone), and does not depend on the canvas (render scale 0.5 and 1 give the
same ms). Chrome, M3 Max, 1280×720, page visible (headless, not throttled), the World
paused and draws submitted alone (`engine/tools/vehicle-lab/draw-cost.js`; before =
main with the timers only):

| GPU ms a drawn frame | 1 car | 5 cars | 8 cars |
|---|---|---|---|
| skinning, before | 2.59 | 13.2 | 21.0 |
| skinning, floor | 0.43 | 2.2 | 3.5 |
| skinning, after | 0.75 | 3.73 | 5.96 |
| shadow map, before / after | 0.24 / 0.24 | 0.47 / 0.49 | 0.66 / 0.68 |
| picture, before / after | 3.1 / 3.3 | 9.2 / 8.5–9.0 | 12.4–12.9 / 11.3–11.9 |
| World steps a second while stepping, before / after | 60 / 60 | 14 / 22 | 10 / 16 |
| FPS while stepping, before / after | 60 / 60 | 35 / 54 | 24 / 38 |

The skinning floor is bandwidth: per car per frame about 157 MB of binds read (64 B a
vertex and 8 B an entry) and 15.7 MB written (490,997 vertices × 32 B), about 173 MB,
0.43 ms at 400 GB/s. After is 1.7× it at every car count. The shadow map (427,478
opaque triangles a car, depth only, about 0.06 ms a car) is not worth an option
trading its resolution or rate, so none was added. The picture did not change and is
the largest part now: with five cars the opaque cars are about 6.6 of its 8.5–9 ms,
and a quarter of the pixels (render scale 0.5) takes off only 1.2–1.5 ms; leaving out
the fragment discards changes nothing. Its cost follows the triangles (2 cm edges, far
below a pixel at the lab's distances), which is what the renderer's levels of detail
(`LOD_PX` = 3) are for; the generated car has one level.

The picture is the same. `skin_parity_gpu`: 4000 of 4000 vertices bit for bit from the
8-byte entries as from the f32 ones. The recut check (crash for 5 s, pause, cut every
list again from the labels as they are): 0 of 921,600 pixels differ. Stills of every
frame of the first 1.2 s, five cars, frame-locked recording, two runs of main and two of
this branch: the first 48 frames (0.8 s) are the same bytes in all four, and after that
the runs drift apart as two runs of main do (main against main 89 → 56 dB by 1.2 s,
main against this branch 89–94 → 46–69 dB): Chrome's World is not reproducible run to
run, so later frames are within run-to-run noise and exact identity under deformation
rests on the parity test and the recut check.

Not done (follow-ups): skip skinning while nothing it reads has changed (the World
paused, or a car asleep), which needs a GPU-side check like `cutCheck` because the
level pick follows the camera; levels of detail for the generated car (changes the
picture by design, needs the owner's decision and work on the asset side).

### 8.31 One car's own step: fixed cost against work, and the wheels off the list's and the hits' core (2026-11-16)

The lab's car alone (`sedan_new`: 548 nodes, 4,251 beams, 528 collision triangles, four
raycast wheels, the World coupling on), native M3 Max, ms a frame of 34 substeps, from the
saved moments of `fast-check.sh` (t 2.5, 6 and 20 s; t 20 is the wreck at rest, 9 hits a
step). Wall figures are `--ab-perf-masks` (60 frames a mask from one snapshot, median /
fastest tenth); pass figures are its per-pass timestamps (each dispatch a timed pass of its
own). The GPU was shared with another session's jobs throughout (through the lock), so wall
figures move by a few percent between runs; the A/B pairs below ran interleaved.

**What a dispatch costs here.** `RUBBLE_EXTRA_DISPATCH=n` (a measurement-only host switch,
`ExplicitBeamsGpu::extra_dispatch`) adds n dependent dispatches of a trivial kernel (one
workgroup of 64 threads adding one to a word) after each of the frame's 279 dispatch
points (176 wide commands, 103 list, walk and pairs dispatches); `_INDIRECT=1` makes them
indirect dispatches of zero workgroups. t 20, fastest tenth: 5.72–5.74 without; one more
6.38 (+0.65 ms, 2.3 µs each), two 6.76 (+1.03, 1.85 µs), four 7.79 (+2.06, 1.85 µs); two
empty indirect 6.34 (+0.61, 1.1 µs each). **A dependent dispatch costs about 2 µs in the
frame, an empty indirect one about 1 µs**, not the 5–8 µs of a dispatch timed in a pass of
its own (§8.23: the timestamps add ≈3.4 µs a pass, (6.64 − 5.81) / 243). The frame's 175
dispatches with work and 68 empty indirect ones are **≈0.4–0.5 ms of fixed cost (8 %)**.
Fusing passes saves ≈0.07 ms a frame per dispatch removed from each substep; it is not where
most of the frame goes.

**Per pass: fixed, work, floor** (t 20, timed apart, ms a frame; the "of which" splits are
time-only switches that skip a part (removed after use; they change results, so each
reading is that pass's own time only); floor: the work's bytes at 400 GB/s and its longest
dependent chain at 0.15–0.2 µs an access (§8.23's measured per-access cost), plus the
in-frame boundary of 2 µs, × 34):

| pass | dispatches a frame | timed apart | of which | floor |
|---|---|---|---|---|
| nodes | 34 | 0.84–0.88 | the dispatch with no node updated (boundary, timestamps, the coupling header's lane and barrier) 0.20; node updates 0.67, of which the static shapes' and the coupling's proxy contacts 0.38 | 0.15 (0.17 MB a substep: 0.4 µs; ≈10 accesses in series) |
| beams | 34 | 0.52–0.55 | with no beam updated (the triangles' geometry threads, §8.23's 2048) 0.35; the beam updates 0.18 | 0.15 (0.78 MB a substep: 2 µs) |
| list kernel | 35 | 1.97–2.05 | step s + 1's coupling and wheels (the rays, the bars) 0.68–0.69, of which the rays' loop over the static shapes 0.13; the list's own phases 1.28 (the pieces' tests 0.02) | 0.2–0.3 |
| walk and pairs | 68 | 0.49–0.51 | 0.19 of it the empty indirect walks and pairs (§8.23) | 0.1 |
| hits | 34 | 2.16–2.21 | the next step's tyres 0.50; `selfApply` ≈1.4 (§8.23: fill 0.23, round 0.55, residual 0.79); the dispatch with nothing to apply 0.29 | 0.3–0.6 |
| centred damping (two passes; native only) | 68 | 0.50 | | 0.1 |
| begin, rebase, step 0's aux, end, close | 6 | 0.09 | | |
| **frame (wall)** | 243 | 5.72–5.83 | boundaries ≈0.45 | §1.3: 0.3–0.5 |

The two one-workgroup kernels (list, hits) are 4.2 of the 6.6 ms timed. Inside them the
largest single item was not self-collision: **the four raycast wheels cost 1.19 ms a frame
(0.69 in the list kernel, 0.50 in the hits kernel), 0.85 ms of wall time**, a serial chain
of a few lanes (drvCtx's loops over the hubs and curves, a ray per wheel, the bars, the
tyre: about a hundred dependent accesses a step) that rides on the critical path of the
workgroup it shares.

**Built: the wheels off the list's core, then the tyres split at their hubs' forces.**

*Perf flag 131072, `PERF_AHEAD_APART`* (with `PERF_WIDE_FUSED`; `wideAheadApart`). The list
kernel's dispatch gets a second workgroup a vehicle (vehicles + v) that runs step s + 1's
coupling (its step or its fold, phases B and C) and its wheels' rays and bars: the same
calls on the same lanes with the same barrier between, in place of on the list workgroup's
lanes beside its phases. They read the nodes after the step's node phase, the coupling's
records and sums, the wheels' records and the static shapes, and write the coupling's
bodies, spheres and counts and the wheels' scratch records; the list's phases write none
of these and read none of what they write (they ran beside them already, between the same
barriers); the tyres that read the scratch records are in the next dispatch. On a core of
their own they no longer lengthen the list's phases. The list workgroup's code changes
only in that `ahead` is false with the flag; the new path is its own function behind an
early return.

*Perf flag 262144, `PERF_TYRE_AHEAD`* (with 131072; `drvTyrePre`, `drvTyrePost`,
`driveTyresPre`, `driveTyresPost` in `explicit_beams_drive.slang`). A tyre reads its hubs'
forces (`drvForceNow`: the beam sums and the contact accumulators after the step's hits)
and nothing else that the walk, the pairs or the hits change. `drvTyrePre`, on the wheels'
workgroup after the bars, runs everything that does not need them (the frame, hub and axle,
the steer angle with Ackermann, the tyre's frame, spin drive and brake, slip ratio and slip
angle, both friction curves, the friction ellipse's shape, the outputs 17..19, the
vehicle's end record); `drvTyrePost`, on the hits kernel's lanes after `selfApply`, reads
the hubs' forces and runs the rest (the spin solves, the ellipse's cap, the state records,
the hub forces). The values cross in eight float records a wheel after the vehicle's wheels
(`DriveLayout::stride_for`: 20 + 31 records a wheel, `TYRE_PRE_RECS`), written as floats
and read back exactly, cleared by the second part, so a frame leaves them zero as without
the split. `drvTyre`'s arithmetic in its order, the ellipse's shape computed always where it
was computed only when the ellipse binds (pure, read only there).

**Proofs.**
- `fast-check.sh perf=131072`: the seven fixtures (one car at t 2.5, 6, 20; two cars at 2,
  3.7, 8, 16 s), 4 and 20 frames, every state word the same. After the drive layout change
  (new fixtures, `out/fixtures2`): `perf=262144` with `RUBBLE_PERF_FLAGS=147393` (the tyre
  split against the wheels apart alone) and `perf=393216` (both against neither), seven
  fixtures, 20 frames, every word the same. (Two cars run the contact path, where neither
  flag acts: those lines prove only that nothing else moved.)
- The old path against main's kernels after the first change (`--ab-switch kernel
  --ab-kernel-dir` on a copy of main's WGSL, the flag off; before the drive layout change,
  after which main's kernels and these share no fixtures): one car at t 2.5 and 6, two cars at all four moments, every
  word the same. One car at t 20 differs (103 node words, 114 beam words, first node word
  5304: 0x4135f235 / 0x4135f244), **and so does this build's WGSL against itself** (the same
  words): the kernel-dir harness is not bit-reproducible at that moment, independent of any
  kernel change. §8.23's rejection of the thread-local opaque zero quotes exactly these words,
  so that verdict was the harness's, not the change's (not re-measured here).
- Chrome: all 116 modules compile after each step (`explicit-wgsl-check.mjs`; no new
  module, no new workgroup variable).
- Suites with both flags on (`perf-verify.sh 409537`): `explicit_loop_cap` 5,
  `explicit_beams_bitwise` 12, `explicit_contact_gpu` 13, `self_split_gpu` 8,
  `explicit_pair_reuse_gpu` 2, `explicit_beam_order_gpu` 3, `showcase_car_gpu` 6 (1
  ignored, as before), `explicit_drive_gpu` 3, engine-world `explicit_vehicles` 11,
  `explicit_drive_world` 1, `vehicle_lab_reset` 3, engine-metal `vehicle_parity` 4,
  `explicit_drive_parity` 1: PASS. With the default flags (the old path's Metal codegen
  after the edits): `vehicle_parity` 4, `explicit_drive_parity` 1, `explicit_parity` 7:
  PASS.
- Of these only `vehicle_lab_reset` runs the lab's fused wide schedule (the World with
  `wide_fused`), where the flags act; the GPU == CPU suites do not. The new paths' proof is
  the A/B against the shipped path on the same state, which the suites and §8.8–§8.9 hold
  to the CPU reference.

**Gain** (`--ab-perf-masks` from the saved moments, interleaved, fastest tenth / median, ms
a frame):

| moment | neither | 131072 | both (393216) |
|---|---|---|---|
| t 2.5 (falling, first contact) | 4.31 / 4.41 | 4.05 / 4.16 | 3.89 / 3.99 |
| t 6 (crash) | 5.43 / 5.47 | 4.85 / 4.90 | 4.63 / 4.67 |
| t 20 (wreck at rest) | 5.31 / 5.76 | 5.07 / 5.25 | 4.99 / 5.08 |

Passes at t 20 (timed apart): list 2.01 → 1.35 → 1.51, hits 2.19 → 2.19 → 1.75; every
other pass within 0.005. The first change is −5 to −11 % of the frame; the second −2 to
−5 % more (the wheels' workgroup becomes the list dispatch's longest at 1.51 against the
list's own 1.3, so part of what leaves the hits kernel is paid there). Spreading the four
wheels over four SIMD groups (wheel w on lane 32w) in both places: no gain (tried, removed).

**Whole 40 s run** (native, release, `--secs 40`, the vehicles' GPU ms a World step from
t 0.5 s, three rounds interleaved, each run alone on the GPU through the lock):

| flags | median | p95 | p99 | worst |
|---|---|---|---|---|
| default (16321) | 5.23, 5.24, 5.28 | 5.52, 5.50, 5.53 | 5.65, 5.64, 5.66 | 5.84, 8.83, 6.96 |
| + 131072 | 4.87, 4.72, 4.72 | 5.11, 4.97, 4.93 | 5.23, 5.06, 5.04 | 5.51, 8.12, 5.49 |
| + 131072 + 262144 | **4.55, 4.56, 4.55** | **4.74, 4.74, 4.72** | 4.86, 4.86, 4.82 | 7.52, 7.87, 5.25 |

Median −13 %, p95 −14 %, p99 −14 %. The worst is one step near t 7.13 s (the crash): 5.8
to 8.8 ms with the default flags as well, so it moves with something other than these
switches (not investigated here); without it the worst is 5.25–5.49 with the flags against
5.84 without. (The scene is main's after PR #117, whose rolling terrain is the lab's
default now: these are not §8.26's figures, 4.77 / 5.03 / 5.30 on the terrain before it.)

**Default: both off.** Bit for bit and faster natively at every moment and over the
whole run, but not measured in Chrome (the page's checkboxes "wheels apart from the list"
and "tyres mostly ahead" are untested in a browser; Chrome only compiled the modules), and
the rule for a default is both. To turn them on: the lab page with both boxes ticked
against unticked (`vehicle-lab-record.mjs --probe` on the mountain, one car), then add
`PERF_AHEAD_APART | PERF_TYRE_AHEAD` to `perf_flags_default` and tick the boxes by default.

**What stands between this and 1 ms** (one car at rest, ≈4.6–5.0 ms a frame now):
- The hits kernel, 1.75: `selfApply`'s fill, rounds and residual pass (≈1.4) at 9–16 hits,
  each phase a chain of device accesses at 0.15–0.2 µs whatever their kind (§8.23), and
  the dispatch with nothing to apply (0.29). Its floor is 0.3–0.6: it needs the hits'
  working set in workgroup memory for the rounds (an access there is tens of
  nanoseconds), not fewer atomics.
- The list dispatch, 1.5: now the wheels' workgroup (rays, bars, tyre's first part) sets
  it; the list's own phases are 1.3. Both are long single-lane chains (drvCtx alone loops
  over the hubs and curves in series); the same remedy as the hits.
- The nodes, 0.85 against 0.15: the coupling's and static shapes' contacts (0.38) and the
  dispatch's own lane-0 header and barrier (0.20 with no node updated).
- Fixed cost: ≈0.45 ms of boundaries (243 dispatches), the centred damping's two passes
  (0.5 natively), the empty walks and pairs (0.19).
- The sum of the per-pass floors with today's eight dependent dispatches a substep
  (natively; six on the page, without the centred damping) is 1.0–1.4 ms (8 × 2 µs × 34 =
  0.54 of it boundaries): the 1 ms target needs both the chains in workgroup memory and
  fewer dependent dispatches a substep (four would be 0.27 ms).

### 8.32 The levels across the GPU (stage 3c): built and bit for bit; the levels are chains, so off by default (2026-11-16)

`contact-groups-design.md` §8.8 (what is built). Perf flag 524288 (`PERF_FLEET_LEVELS`,
with 16384 and 65536), a checkbox on the lab page (unchecked). Module
`explicit_beams_split_fleetlevels` (four variants; 124 explicit modules). The fixtures
were made again: the fleet's gNext region and gBeams grew (the levels' state, the staged
hits kept).

**Measured first: the levels in use** (`vehicle_lab … --ab-profile` with `AB_LEVELS=n`:
each frame's last pass, its owners with hits by level; the levels' dispatch timed alone):

| scene | levels (top) | owners a level | hits a level | the levels, ms a frame (µs a substep) | untimed frame, ms |
|---|---|---|---|---|---|
| 8 cars, t 3.7 | 1–2 | 1 | 1–4 | 7.10 (212) | 19.3 |
| 8 cars, t 16 | 0 (in the sampled passes) | – | – | 1.04 (31) | 17.6 |
| 32 cars, t 1 | 2–9 | 1–4 | 1–10 | 36.0 (1,075) | 53.0 |
| 32 cars, t 2.5 | 2–6 | 1–2 | 1–8 | 21.5 (646) | 42.5 |
| 32 cars, t 4 | 5–8 | 1–2 | 1–14 | 33.0 (987) | 54.7 |
| 32 cars, t 12 | 6–7 | 1 | 1–27 | 27.6 (824); the walk 29.1 | 75.9 |
| 32 cars, t 25 | 11–12 | 1 | 1–8 | 49.8 (1,486); the walk 48.4 | 117.8 |

A few hits, in a chain of levels. Ten hits cost a millisecond a substep because one hit's
phase is one thread's chain of dependent loads (the triangle, the clip test or the vote,
four nodes' forces, the snapshot): 25 to 40 µs measured as a dispatch of its own (about
10 µs of it the timing's own pass). A level is four such phases in turn (the hits, three
residual phases); the levels of a substep are in turn.

**The floor** (with today's work a hit): levels × 4 phases × one hit's latency, plus a
dispatch's 2 µs for each when they are dispatches. At t 25 (12 levels, one owner each):
12 × 4 × ~30 µs ≈ 1.4 ms a substep, 49 ms a frame; measured 49.8. The group's one
workgroup is already at that floor whenever a level holds one owner: spreading a level's
hits across the GPU removes nothing from a chain and adds 4 L + 3 dispatches a substep.
It removes the owners' turns within a level (the workgroup takes a level's owners one
after another) only where a level holds several.

**Proofs** (native M3 Max; the GPU shared with other sessions through the lock):

- `fast-check.sh perf=524288`, the 18 fixtures (1, 2, 4, 8, 32 cars): with 16 slots forced
  (`RUBBLE_FLEET_LEVEL_SLOTS=16`), with the slots as the levels ask, and before the staged
  hits were kept: every fixture "bit for bit: true"; 8 and 32 cars at `FRAMES=20` with 16
  slots; 32 cars at 20 frames with 2 slots (levels 3 and up in the end dispatch's
  workgroup): true.
- A loop-budget run (`RUBBLE_LOOP_BUDGET=200000`, 32 cars, the flag on): no cap reached.
- `perf-verify.sh 622529` (the defaults and the flag, 16 slots forced): `fleet_contact_gpu`
  (new: a column of ten pushed from behind, with 16, 2 and 1 slots, with and without self
  split, its levels at least 3 deep, asserted; two columns of six side by side: two
  groups in the same dispatches), `group_split_gpu` 3, `self_split_gpu` 8,
  `explicit_contact_gpu` 13, `explicit_beams_bitwise` 12, `explicit_loop_cap` 5,
  `explicit_vehicles` 11, Metal `vehicle_parity` 4: PASS.
- Chrome: "all 124 modules compile in Chrome".
- Whole 40 s runs with it on end in the same state as off (8 and 32 cars: every car's
  speed and energy the same at t 40).

**Native, ms a frame, on / off** (fast-check A/B from one state, 20 frames, 16 slots
forced): 32 cars t 1, 2.5, 4: 41.2 / 49.1, 41.3 / 50.6, 47.4 / 50.8; 8 cars t 2, 3.7, 8,
16: 16.5 / 12.7, 22.9 / 19.8, 19.2 / 15.3, 20.6 / 17.7 (four slots' dispatches, 67 a
substep, cost 3 to 4 ms; with 5 slots the eight-car frames were 1.2 to 1.9 ms slower).
The untimed frame after it (`--ab-profile`, 10 frames, 5 slots): 32 cars t 1 41.9 against
53.1, t 4 47.1–48.0 against 54.7; 8 cars t 3.7 21.1 against 19.9, t 16 19.4 against 17.7.

Inside it at 32 cars t 4 (timed apart; 5 slots): hits 7.1 ms a frame (170 dispatches of
42 µs), residual phases 13.3 (510 of 26), the end 7.5 (the levels past 5 in the group's
workgroup), the groups' part 2.3, snapshot 0.7, plan 0.5. Keeping the staged hit for the
residual phases took a residual phase from 36–38 to 26–31 µs.

**So the slots follow the levels**: a group reports its highest level and whether one
of its levels held two owners or more; the host reads it with the status (every 16
frames) and gives the next frames that many slots, or none (the groups' own levels, no
extra dispatch) when every level held one owner, for four reads after the last such
report. With that, at 8 cars the flag costs nothing (fast-check, 18 fixtures, within
±0.9 ms either way) and the whole runs:

| vehicles GPU ms a step, t ≥ 0.5 s, 40 s | median | p95 | worst |
|---|---|---|---|
| 8 cars, off | 15.73 | 17.14 | 22.17 |
| 8 cars, on | 15.95 | 16.90 | 23.39 |
| 32 cars, off | 115.4 | 131.5 | 164.3 |
| 32 cars, on | 116.9 | 133.4 | 142.1 |

At 32 cars the run is dominated by its later part, where the levels are a chain (and
the walk as large as the levels: 29–48 ms a frame); the slots stay off there. The crash's
first seconds, where levels hold several owners, are faster (the fixtures above), and the
worst step drops 164 → 142 ms; the median does not move.

**Chrome, eight cars** (`run.sh --probe --secs 40`,
`mountain:mode=mountain&car=gen&timing=step&cars=8`, `&fleetLevels=1` for on; vehicle GPU
ms a World step, t ≥ 0.5 s, about 10 s simulated each): off 16.32 / 22.31 / 25.40
(median / p95 / worst), on 15.95 / 21.38 / 24.14: within the runs' noise (§8.29's two
runs of one setting differ by 0.7 ms at the median).

**Default: off.** Bit for bit, but not faster over whole runs natively (32 cars: the median
and p95 a little slower, the worst 22 ms faster; 8 cars: the same) nor, beyond noise, in
Chrome.

**Eight cars above 10 ms, what the rest is** (`--ab-profile`, defaults, ms a frame of 34
substeps; t 3.7 / t 16): the cars' own steps (self step) 5.7 / 6.0; the levels 7.1 / 1.0;
the members' self-collision hits 1.7 / 1.7; the walk 1.0 / 4.4; self pairs 0.8 / 0.9;
triangles and points 0.7 / 1.0; owners' parts 0.6 / 0.6; listed pairs 0.6 / 0.5; grids
0.46 / 0.47; the two prefixes 0.67 / 0.68; sorted copy 0.31 / 0.32; scans 0.30 / 0.30;
throttles 0.23 / 0.23; untimed 19.3 / 17.6 ms (whole run median 15.7). Without the levels
and the walk the rest is about 11 ms, half of it the cars' own steps. The fleet front's
eleven small dispatches are 3.9 ms (most 10–20 µs a substep each, latency, not work).

**What would cut the levels further**: one hit's latency, not parallelism across hits
(the staged geometry kept in the group's own levels too, where the lane that stages a
hit is the lane that finishes it; a hit's four nodes' forces on four threads), and fewer
passes over a hit (three residual phases and the application each take its triangle and
clip test again).

**Then: one hit's latency, and two front dispatches fused (2026-11-17).**
`contact-groups-design.md` §8.9. Perf flags 8388608 (`PERF_FLEET_HITLAT`: the group's
hits flattened over its lanes, the snapshot a lane a node, a hit's staged geometry kept in
its lane's registers across the residual phases) and 16777216 (`PERF_FLEET_FUSE`: the
pairs' scans in the plan's workgroup, the sorted copy in the grids' workgroups), both on
by default, a checkbox each on the lab page (checked).

Proofs: `fast-check.sh` with `perf=8388608`, `perf=16777216` and both, all 18 fixtures
(1, 2, 4, 8, 32 cars), and both at `FRAMES=20` on the 8- and 32-car fixtures: every
fixture "bit for bit: true". `perf-verify.sh 25264065` (the defaults before and both):
`fleet_contact_gpu` 1, `group_split_gpu` 3, `self_split_gpu` 8, `explicit_contact_gpu`
13, `explicit_beams_bitwise` 12, `explicit_loop_cap` 5, `explicit_vehicles` 11, Metal
`vehicle_parity` 4: PASS. `fleet_contact_gpu` adds each flag and both against off on the column of ten (at
least 3 levels deep, asserted) and two columns of six, with and without self split
(passes with the defaults and with `RUBBLE_PERF_FLAGS=25264065`). Chrome: all 124 modules compile. Whole 40 s runs end in the same state.

ms a frame by dispatch kind (`--ab-profile`, the same fixtures, both off / both on):

| | 8 cars t 3.7 | 8 cars t 16 | 32 cars t 4 | 32 cars t 25 |
|---|---|---|---|---|
| the levels (`split levels`) | 7.09 / 5.49 | 1.04 / 1.04 | 32.9 / 17.4 | 49.3 / 31.3 |
| scans + sorted copy | 0.61 / 0 | 0.63 / 0 | 0.92 / 0 | 1.92 / 0 |
| owners' parts + grids (now with them) | 1.10 / 1.33 | 1.15 / 1.47 | 2.01 / 2.49 | 2.46 / 3.49 |
| untimed frame | 20.0 / 18.0 | 17.8 / 17.4 | 54.9 / 39.3 | 117.8 / 97.5 |

fast-check at 20 frames, on / off: 32 cars t 1, 2.5, 4: 32.6 / 49.2, 33.5 / 50.6,
38.8 / 51.2; 8 cars t 2, 3.7, 8, 16: 12.3 / 12.7, 17.9 / 19.7, 15.0 / 15.3, 17.4 / 17.7.
The level's floor moved: at t 25 (12 levels a substep, one owner each) 0.94 ms a substep
against 1.47; a level is now about 77 µs (its application and three residual phases).

Whole 40 s runs (vehicle GPU ms a step, t ≥ 0.5 s, median / p95 / worst):

| | off | on |
|---|---|---|
| 8 cars, native | 16.51 / 17.48 / 23.28 | 16.38 / 17.09 / 23.10 |
| 32 cars, native | 115.4 / 131.5 / 164.3 | 95.2 / 109.6 / 118.0 |
| 8 cars, Chrome (page visible, ~10 s simulated) | 15.60 / 21.90 / 24.38 | 15.50 / 19.94 / 24.80 |

**Default on**: bit for bit; faster natively (32 cars by a fifth, 8 cars a little) and in
Chrome at the median and p95 (its worst within the runs' noise).

**Eight cars with the other two branches** (a local merge of this branch,
`perf/fleet-candidates` and `perf/own-step`, not pushed; native, 8 cars, mountain):

| ms a frame (`--ab-profile`) | merge, kept candidates and these off | with `PERF_FLEET_KEEP` and `PERF_FLEET_HITLAT` |
|---|---|---|
| t 3.7: the cars' own steps (self step) | 5.75 | 5.76 |
| levels | 7.03 | 5.61 |
| members' self-collision hits | 1.65 | 1.67 |
| walk | 1.00 | 0.25 (one build) |
| owners' parts, grids, scans, triangles and points | 2.23 | 2.58 |
| listed prefix, pairs' prefix, pairs, sorted copy, throttles | 2.02 | 0.14 |
| self pairs + self walk | 1.26 | 1.26 |
| untimed frame | 20.1 | 16.8 |
| t 16: untimed frame | 17.9 | 13.1 (self step 5.98, members' hits 1.71, levels 1.05, owners' parts 1.05, self pairs 0.86) |
| whole 40 s, vehicles median / p95 / worst | 16.16 / 17.49 / 23.80 | 11.97 / 14.48 / 18.97 |

(`PERF_FLEET_FUSE` is not in the merge's measurement: its sorted copy and the kept
candidates' build share the grids' dispatch, which needs joining first.) With the three
the eight-car frame is 12–17 ms; the gate (10 ms) is not met. What is left, at t 16:
the cars' own steps (6.0 ms, the same in the merge as without `perf/own-step`: its
changes did not show on this path), the members' self-collision hits (1.7), the front's
remaining dispatches (3.2, ten small ones), the levels (1.0) and self pairs (0.9). At t 3.7 the levels are still 5.6 ms: two
levels of one owner each, each about 75 µs a substep of one thread's chains.

### 8.33 One car's own step: §8.31's switches on by default, the hits from workgroup memory (2026-11-16)

The lab's car alone (`sedan_new`, as §8.31), native M3 Max and Chrome on the same machine.
The GPU was shared with three other sessions' jobs through the lock throughout: each
measurement below ran alone on the GPU, but wall figures move by a few percent between runs,
and the A/B pairs ran interleaved in one hold of the lock.

**§8.31's two switches in Chrome.** The lab page (`vehicle-lab-record.mjs --probe`, mountain,
one car, 40 s, the page's own loop, vehicles' GPU ms a World step from t 0.5 s), with the
boxes "wheels apart from the list" and "tyres mostly ahead" off and on. Two rounds. The first
page of each round is slower than any later one whatever its flags (a cold GPU: 6.03 and 5.75
ms where the same flags give 5.40 and 4.50 warm), so the comparison is between warm pages:

| flags | median | p95 | p99 | worst |
|---|---|---|---|---|
| neither (warm, round 2) | 5.40 | 6.10 | 6.32 | 11.25 |
| both (rounds 1, 2, 2) | **4.73, 4.87, 4.82** | 5.31, 5.56, 5.47 | 5.55, 5.88, 5.66 | 8.22, 10.57, 10.38 |

−10 to −12 % of the median and −9 to −13 % of p95: faster in Chrome as natively (§8.31's
−13 %), and bit for bit (§8.31's proofs, and the suites below with them on). **Both are on by
default now** (`perf_flags_default`, and their boxes ticked). The page's worst step, near
t 31.9 s in every warm run whatever the flags, is the page's.

**Built: the hits from workgroup memory** (perf flag 1048576, `PERF_HITS_LOCAL`, with
`PERF_HIT_ROLES` and `PERF_HIT_FB`; `selfApplyLocal` in the wide hits kernel). With those
switches a two-body dissipative car's `selfApply` is the beam forces' fill, one round (no hit
reads another's force, so round 1 computes them all) and the residual pass's seven phases.
Every value the phases pass to each other is a node's (beam force, mass, hit count,
accumulators A and B) or a hit's (record, role records), and each phase read them again from
device memory. Now item q (hit q >> 2, role q & 3) on lane q loads its node's values once; a
node named by several items has one copy (the first item naming it); the round and the seven
phases run on the copies (the integer adds as workgroup atomics on the copy: the same
integers); the accumulators are written back once, as the device's value plus the copy's
adds. Every float is the same strict arithmetic on the same bits in the same order as
`selfTwoBodyCore`, `hitRolePhase` and `hitCombinePhase`; the scratch records those leave in
`gNext` are written as they leave them. 1..48 hits a step (192 items on the kernel's 256
lanes); more, or another kind of car, takes `selfApply` as before. The arrays are the list
kernels' (`gsTested`, `gsPairs`, `gsCount`; 17 KB with the module's others, under Chrome's
32 KB), which the hits kernel did not use, so no new workgroup variable. Two refinements,
each measured alone: its barriers are workgroup barriers (no lane there reads a device word
another lane wrote there; `sync()` also waits for the phase's device writes): no change. The
round's four position steps a hit are added by the items' own lanes (`shiftAdd`'s integer
adds, from workgroup memory) instead of twelve atomics in a row on the hit's lane: hits pass
−0.07 ms.

Proofs (the flag against without it, on top of §8.31's flags):
- `fast-check.sh perf=1048576` on all fifteen fixtures (one car at t 2.5, 6, 20 s; two, four
  and eight cars at 2, 3.7, 8, 16 s), 4 and 20 frames, every state word the same; again after
  each refinement. (Two or more cars run the contact path, where the flag does not act.)
- The `--ab-perf-masks` replays (60 frames a mask, three rounds): every mask's frames end in
  the same state ("=").
- Chrome compiles all 116 modules (`EXPLICIT_WGSL_MODULES_ONLY=1 explicit-wgsl-check.mjs`).
- Suites with every switch of this section on (`RUBBLE_PERF_FLAGS=3555265`): engine-webgpu
  `explicit_loop_cap` 5, `explicit_beams_bitwise` 12, `explicit_contact_gpu` 13,
  `self_split_gpu` 8, `explicit_pair_reuse_gpu` 2, `explicit_beam_order_gpu` 3,
  `showcase_car_gpu` 6 (1 ignored, as before), `explicit_drive_gpu` 3; engine-world
  `explicit_vehicles` 11, `vehicle_lab_reset` 3; engine-metal `vehicle_parity` 4,
  `explicit_drive_parity` 1: PASS. (Of these only `vehicle_lab_reset` runs the lab's fused
  wide schedule where the flags act; the A/B on the same state is the new path's proof, as
  §8.31 says.)

**Tried, not kept on** (perf flag 2097152, `PERF_TYRE_EARLY`, off; bit for bit on all fifteen
fixtures, 20 frames): the tyres' second part loading its records, its hubs' masses and beam
sums on the wheels' lanes while the other lanes apply the hits, and reading only the hubs'
accumulators after (`drvTyreLoad`, `drvTyreFinish`). Hits pass −0.02 ms, wall unchanged: the
tyres were 0.08 ms of the pass (below), not on its critical path.

**Where the hits pass goes now** (t 20, timed apart; time-only switches, removed after use):
the dispatch with nothing to apply, its lane-0 header and the tyres 0.41 (the tyres 0.08);
the load 0.22; the round 0.45 before the position steps moved, ≈0.38 after; the seven phases
and the write-back 0.21. The round is one lane a hit running `selfTwoBodyCore`'s strict
contact rule (`contactResponse`, `noPositiveWork`) on values already in workgroup memory: its
time is that chain's arithmetic, not memory.

**The list dispatch** (time-only switches at t 20): without the coupling's and wheels'
workgroup 1.31–1.32 ms; without its coupling 1.30; without its wheels 1.30; without the list
workgroup 1.52. Each of the three (the list's phases, the coupling's, the wheels') is about
1.3 ms on its own, so shortening any one of them alone does not shorten the dispatch.

**Per pass: before, floor, after** (one car at rest, t 20, ms a frame, timed apart; before:
§8.31's both flags; floor: §8.31's):

| pass | before | floor | after |
|---|---|---|---|
| hits | 1.78–1.79 | 0.3–0.6 | **1.22** |
| list (list workgroup; coupling and wheels workgroup) | 1.50–1.51 (1.31; ≈1.5) | 0.2–0.3 | 1.51 |
| nodes | 0.85 | 0.15 | 0.85 |
| beams | 0.54 | 0.15 | 0.54 |
| walk and pairs | 0.51 | 0.1 | 0.51 |
| centred damping (native only) | 0.50 | 0.1 | 0.50 |

Wall at the saved moments (`--ab-perf-masks`, median / fastest tenth, ms a frame): t 20 5.09 /
5.01 → **4.68 / 4.45**; t 6 5.15 / 5.10 → **4.77 / 4.66**; t 2.5 (6 hits, falling) 4.05 →
4.08 (no change). −8 % at the crash and at rest.

**Whole 40 s run, one car** (vehicles' GPU ms a World step from t 0.5 s; native
`vehicle_lab --measure mountain-perf --secs 40`, release, two rounds interleaved; Chrome as
above):

| flags | median | p95 | p99 | worst |
|---|---|---|---|---|
| native, the old default (16321) | 5.25, 5.25 | 5.52, 5.56 | 5.64, 5.65 | 8.85, 8.24 |
| native, + §8.31's two | 4.57, 4.58 | 4.76, 4.79 | 4.88, 4.89 | 7.96, 6.30 |
| native, + the hits from workgroup memory (the new default) | **4.22, 4.25** | **4.40, 4.45** | 4.50, 4.58 | 4.75, 7.87 |
| Chrome, neither of §8.31's (warm) | 5.40 | 6.10 | 6.32 | 11.25 |
| Chrome, + §8.31's two (warm) | 4.73, 4.87, 4.82 | 5.31, 5.56, 5.47 | 5.55, 5.88, 5.66 | 8.22, 10.57, 10.38 |
| Chrome, the new default (warm) | **4.57, 4.50** | 5.36, 5.42 | 5.57, 5.65 | 7.60, 10.57 |

The new default against the old: native median −19 %, p95 −20 %; Chrome median −15 to −17 %.
The hits switch alone: native −7.5 % median and p95; Chrome −3 to −7 % median, p95 unchanged
within the runs' spread. The native worst is still the crash step near t 7.13 s in three of
four runs (§8.31: it moves with something other than these switches).

**Default: `PERF_HITS_LOCAL` on** (bit for bit, faster natively and in Chrome), its box
ticked; `PERF_TYRE_EARLY` off, its box unticked. With the new default: engine-metal
`vehicle_parity` 4, engine-world `vehicle_lab_reset` 3 and `explicit_vehicles` 11: PASS.
With the old default (16321: the old path's Metal codegen after the edits): engine-metal
`vehicle_parity` 4, `explicit_parity` 7, `explicit_drive_parity` 1: PASS.

**Stop.** After the hits from workgroup memory (−7 to −8 %), three attempts in a row gained
under 5 % of the frame: the tyres loaded early (0 %), the workgroup barriers (0 %), the
position steps on the items' lanes (−1.5 %). What the measurements say about the next step:
- No single pass now holds a cause that a change of where values wait removes. The hits
  pass's largest part (0.38) is a chain of strict arithmetic, not memory; its fixed part
  (0.41) is the dispatch, the header and the tyres.
- The list dispatch is three workgroups of about 1.3 ms each (the list's phases, the
  coupling's step or fold, the wheels' rays, bars and first tyre part). Only shortening all
  three moves it; each is a chain of a few lanes through its own records.

**What stands between this and 2.5 ms and 1 ms** (one car, ≈4.2 ms a step natively, ≈4.5 in
Chrome):
- To 2.5 ms (−1.7 ms natively, −2.0 in Chrome): the list dispatch (1.5 → its floor 0.3 would
  be −1.2), the hits (1.22 → 0.6: −0.6), the nodes (0.85 → 0.3: the coupling's contacts and
  the node record's loads, −0.5). Each needs its single-lane chains split across lanes (the
  list's tracking, decision and build; the coupling's bodies and folds; drvCtx's loops over
  the hubs and curves, a lane a hub), not workgroup memory alone: the hits show that once the
  values are local, the arithmetic chain is what remains.
- To 1 ms: §8.31's sum of floors with eight dependent dispatches a substep is 1.0–1.4 ms;
  1 ms needs the floors and about four dependent dispatches a substep (each ≈2 µs × 34), and
  natively no centred damping's two passes (0.5).

**The kernel-dir harness at t 20** (`--ab-switch kernel --ab-kernel-dir` with this build's own
WGSL differs at the one-car t 20 fixture, §8.31). Not reproduced here, and no race found in
the one-car path: every same-build replay at t 20 in this section (fast-check with four
switches, 4 and 20 frames, and every `--ab-perf-masks` replay, eleven runs) ended bit for
bit. The kernel-dir A/B is a second lab given A's state at the saved moment (`vehicle_lab.rs`:
B is "a second lab of the same options ... given A's state"), so state that the snapshot does
not carry, or the override's own compile of the same WGSL, would differ there and nowhere
else; which one is not established.

### 8.34 The list dispatch's three workgroups, measured and split: under 5 %, stopped (2026-11-16)

The owner's request after §8.33: split the single-lane chains of the list dispatch's three
workgroups (the list's own phases, the coupling, the wheels) as one attempt, designed from a
measurement of each chain first; then the nodes pass's coupling contacts; stop if the
combined change gains under 5 %. It did (−2 to −3 %), so the nodes pass was not started.

**Measured first** (one car, t 20 and t 6, the new default flags; time-only switches that
skip or end a part early, removed after use; each figure the list pass timed apart, ms a
frame of 34 steps; a skipped part changes the state, so each figure is that pass's time
only):

| list dispatch with | t 20 | t 6 |
|---|---|---|
| all three workgroups (the default) | 1.49–1.51 | 1.51–1.52 |
| the dispatch, no work in any workgroup | 0.31–0.36 | 0.31–0.36 |
| the coupling's and wheels' workgroup alone | 1.50–1.52 | 1.52 |
| … its wheels only (coupling off) | 1.05–1.07 | 1.07 |
| … its coupling only (wheels off) | 0.92–0.94 | 0.94 |
| … its wheels without the rays | 0.87 | 0.90 |
| … its wheels without the tyres' first part | 0.90 | 0.92 |
| the list workgroup alone | 1.31–1.33 | 1.39 |
| … ended after its header and state load | 0.37 | 0.36 |
| … ended after the tracking | 0.68 | 0.69 |
| … ended after the throttles, tested list and clears | 0.78 | 0.79 |
| … ended after the decision and the brute pairs | 1.21 | 1.08 |
| … without its tests (the save kept) | 1.22 | 1.18 |

Above the dispatch's own 0.31–0.36: the wheels' work 0.70 (context and bars ≈0.37, rays
≈0.18, tyres' first part ≈0.15), the coupling's 0.57, and together on one workgroup 1.15:
**they add, not overlap**, because every lane runs its share of the coupling's per-node
count and lanes 255..252 also run the wheels. The list workgroup's work is 1.0: the tracking
0.32, the throttles and tested list 0.10, the decision with the brute pairs 0.30–0.43, the
tests 0.13–0.21, the header and load 0.06.

**Built and kept (off): perf flag 4194304, `PERF_WHEELS_APART`** (with `PERF_AHEAD_APART`;
`wideCoupleApart`, `wideWheelsApart`; the host dispatches three workgroups a vehicle). The
coupling on workgroup vehicles + v, the wheels on 2 · vehicles + v: the same calls on the
same lanes between the same barriers (neither reads what the other writes; they already ran
beside each other). Bit for bit (`fast-check.sh perf=4194304`, every fixture, 20 frames;
Chrome compiles all 116 modules; `explicit_loop_cap`). The list pass: 1.51 → 1.29–1.37 (now
the list workgroup's 1.31 sets it; the wheels' workgroup alone is 0.94–0.97, the coupling's
0.88–0.89). Wall at the saved moments (`--ab-perf-masks`, median): t 20 4.80 → 4.67–4.68, t 6
4.84 → 4.79, t 2.5 4.05 → 3.67 (falling, no hits: the list is the frame's longest pass there).
−2 to −3 % at rest and at the crash.

**Tried, not kept** (each bit for bit on the one-car fixtures, each removed):
- *The tested triangles listed by a scan* in place of one shared counter's atomic add a
  tested triangle: the list pass 1.29 → 1.48, slower (three more barriers and two 16-step
  scans cost more than ≈260 atomics on one workgroup word).
- *The brute pairs' contacters from a workgroup copy* (`trianglePairs` reads each
  contacter's node and position; every lane reads the same ones in the same order): list
  pass unchanged (1.29 → 1.31).
- *The tracking's main-body position keys as each lane's maximum, added once* (in place of
  an atomic max a node and key on six shared words): unchanged (1.29 → 1.30).

Neither of the last two was the chain: two dependent device loads a contacter and the
atomics on six words were not what the list workgroup's phases wait on. What they do wait
on was not found: each phase is a lane's two or three nodes or triangles, a handful of
dependent loads each, which the per-access cost of §8.23 puts at 2–4 µs a phase, not the
9–13 µs measured for the tracking and the decision.

**Stop.** The combined change (wheels apart, scan, contacters local, tracking local) gained
under 5 % (the wheels apart −2 to −3 %; the rest 0 or a loss; §8.35 turns the wheels apart on). Not built: the coupling's
per-node count on spare workgroups of the hits dispatch (its inputs are complete there and
nothing between reads its outputs; it would take the coupling's workgroup from 0.88 toward
0.6, but the list workgroup's 1.31 sets the dispatch), the wheels' context with its hubs a lane
each (≈0.1 of the wheels' 0.70), and the nodes pass's coupling contacts. Its whole runs and suites are in §8.35.

**What stands between this and 2.5 ms** (one car, ≈4.2 ms natively): the list dispatch is
now the list workgroup's 1.31 (of which 0.31 is the dispatch with its timestamps); below it
the wheels' 0.95 and the coupling's 0.88. Shortening the dispatch past 0.95 needs the list
workgroup's phases understood first: they cost three to four times what their loads
predict, and two lossless changes aimed at the obvious chains moved nothing. The next
measurement is a lane-level one (a timestamp per phase and lane on Metal), not another
ablation (done in §8.35, as a standalone microbenchmark).

### 8.35 Why a single workgroup's phase costs 3× its loads: a microbenchmark (2026-11-16)

`cargo run --release -p engine-webgpu-lab --example phase_cost` (native wgpu on Metal, M3 Max,
through the GPU lock). One dispatch of 256-lane workgroups runs 1 or 1025 barrier phases;
in each, every lane takes `items` items of `k` dependent loads (each load's value addresses
the next), optionally with a store, an atomic or strict arithmetic, then a barrier; µs a
phase = the difference / 1024. One factor varies at a time from the base (a read-write
storage buffer of 2^16 records, 1 MB; 2 items; storage and workgroup barriers as `sync()`;
one workgroup). A second part times one workgroup running n dependent fma steps, written
out (n distinct instructions) or as a loop, each dispatch after an empty one (code hot) or
after a 16,384-statement kernel (code cold).

| µs a phase (1 MB table) | k 0 | k 1 | k 4 | k 16 |
|---|---|---|---|---|
| base (read-write, 2 items) | 0.44 | 1.11 | 2.39 | 6.96 |
| read-only buffer | 0.43 | 1.09 | 2.37 | 6.97 |
| atomic loads | 0.35 | 1.08 | 2.77 | 9.65 |
| strict arithmetic on each load (an XOR with a uniform) | 0.35 | 1.07 | 2.38 | 7.17 |
| a uniform word in each load's index | 0.36 | 1.07 | 2.36 | 7.09 |
| workgroup barrier only | 0.40 | 1.06 | 2.36 | 6.95 |
| + a store an item | 0.79 | 1.32 | 2.06 | 4.55 |
| + a device atomic add an item | 0.66 | 1.32 | 1.89 | 4.30 |
| + a shared atomic add on one word an item | 0.56 | 1.24 | 1.85 | 4.26 |
| + a device atomic max on 8 words an item | 0.44 | 1.13 | 1.73 | 4.14 |

Base with k 4: 1 item 1.40, 3 items 3.35; 3 workgroups 2.36, 16 workgroups 2.37, 128
workgroups 5.79. The 64 MB table doubles the per-load cost (k 16: 14.25) and 128 workgroups
on it 25.55.

| one workgroup, n dependent fma | written out, hot / cold | as a loop (two integer `%` and a conversion an iteration), hot / cold |
|---|---|---|
| 256 | 1.97 / 2.43 | 14.65 / 14.02 |
| 1,024 | 8.38 / 8.16 | 55.89 / 55.39 |
| 4,096 | 32.09 / 31.49 | 220.72 / 220.40 |

What they say:
- **A dependent device load costs 0.2 µs** in a phase (the k slope: (6.96 − 1.11) / 30 items' loads
  ≈ 0.2 µs at 1 MB, 0.4 at 64 MB): §8.23's figure. A read-only buffer, a uniform in the
  index, the strict arithmetic's opaque zero and the barrier kind change nothing; an atomic
  load costs 1.4×. A barrier phase with nothing in it is 0.35–0.44 µs. Stores and atomics
  after the chain overlap it (k 16 drops to 4–4.5: the next item's chain starts while the
  write drains). So none of the factors the brief named makes a phase 3× its loads.
- **The code's size does not matter**: 4,096 distinct instructions run cold cost what they
  cost hot (32 µs either way). The instruction cache is not it.
- **A dependent ALU step costs ≈8 ns** on a single workgroup (1.97 µs / 256 fma: the chain's
  latency, nothing hides it), and an integer division or modulo far more (the loop's
  iteration, two `%` and a conversion around one fma: 57 ns). Twenty-five dependent ALU
  steps cost as much as one device load.

So the list workgroup's phases (9–13 µs where their loads give 2–4) are paying for
**dependent arithmetic, not memory**: the rest is ≈700–1,100 dependent ALU steps a lane a
phase, or ≈100–150 integer divisions. The kernels' code is built of such chains: every
strict float operation is the operation and an XOR (twice the chain), every array access
carries a bounds clamp (WGSL's), every capped loop computes its end (`ebEnd`'s selects and,
for a runtime step, a division), and the index arithmetic of the list's phases divides by
runtime values (`p / lanes`, `p % lanes`, `i / yz`, `(i / span.z) % span.y`). The same
explains §8.33's round (0.38 ms, its values already in workgroup memory): the strict contact
rule is a long dependent chain at 8 ns a step.

**Does a lossless change follow?** Not a small one. The chains' length is their arithmetic:
the strict operations' XORs are what keeps them bit for bit across targets (a thread-local
opaque zero was tried in §8.23 and moved bits); the bounds clamps are WGSL's. What would
shorten them is splitting a lane's chain across lanes (a phase's items over more lanes, a
contact rule's independent sub-expressions on separate lanes) or removing integer division
from the index arithmetic of the hottest loops (divisions by `lanes`, `subs`, `yz`, which
are runtime values there; replaced by a precomputed reciprocal or a nested loop, the same
indices). The latter is lossless by construction (integer indices only) but each site is a
few percent of one phase; none was built here. Flag 16777216 stays unused.

**`PERF_WHEELS_APART` (4194304), decided by the rule.** Suites with it on
(`RUBBLE_PERF_FLAGS=5652417`): engine-webgpu `explicit_loop_cap` 5, `explicit_beams_bitwise`
12, `explicit_contact_gpu` 13, `self_split_gpu` 8, `explicit_pair_reuse_gpu` 2,
`explicit_beam_order_gpu` 3, `showcase_car_gpu` 6 (1 ignored), `explicit_drive_gpu` 3;
engine-world `explicit_vehicles` 11, `vehicle_lab_reset` 3; engine-metal `vehicle_parity` 4,
`explicit_drive_parity` 1; with the old default `vehicle_parity` 4: PASS. Chrome (the lab
page, mountain, one car, 40 s, warm pages, vehicles' GPU ms a step): default 4.60, 4.58
median (p95 5.36, 5.51); wheels apart 4.52, 4.43 (p95 5.36, 5.36): −2 to −3 % median, p95
the same. Native whole runs (vehicles' GPU ms a World step from t 0.5 s, five rounds interleaved,
median / p95): default 3.96 / 4.38 (the session's first run), 4.27 / 4.45, 4.26 / 4.46,
4.27 / 4.46, 4.24 / 4.41; wheels apart 4.15 / 4.37, 4.17 / 4.37, 4.17 / 4.39, 4.17 / 4.37,
4.13 / 4.30: −2.4 % median and −2 % p95 in the four rounds after the first. Bit for bit,
faster natively and in Chrome: **on by default** (`perf_flags_default`, its box ticked).
One car's step is now ≈4.15 ms natively and ≈4.45–4.5 in Chrome.


### 8.36 A car's own step on the multi-car path: what carries it, two switches built, both off (2026-11-17)

On the integration branch of #115, #118, #119 and #120 (`perf/vehicles-integrated`), eight
cars on the mountain, native M3 Max. The GPU was shared with other sessions' jobs through
the lock, so wall figures are for ranking. §8.33's one-car gains do not reach cars that
touch. Touching cars run the self-collision split path, where a car's own step is the
members' dispatch (`explicit beams split member` with `EB_SELF_SPLIT`, labelled
`explicit self step`: `splitMember`, which runs `splitCoupleStep`, `splitDriveStep` and
`contactMemberStep`) plus the members' rest dispatch (`memberRest`, labelled `explicit self
members' hits`). None of these is one of the wide passes that §8.31–§8.33 changed.

**Measured first.** At t 3.7, 16 and 30 s every car is a member of a contact group: the
step dispatch's solo-car parts cost nothing there. The tools were `vehicle_lab --ab-profile`
(each dispatch a timed pass, 10 frames) and `--ab-split-ablate M:M2`. The ablation used
time-only bits of the second measurement word (`AB_PROFILE_A2`) that skip one part each;
they were removed after use. ms a frame of 34 steps:

| the members' step dispatch with | t 3.7 | t 16 |
|---|---|---|
| everything (the default) | 5.92 | 6.05 |
| no node updates | 4.64 | 4.69 |
| no beam updates | 5.14 | 5.28 |
| no coupling step and no wheels | 4.42 | 4.52 |
| `contactMemberStep` returning at once (dispatch, prologue, coupling, wheels) | 1.93 | 1.99 |
| everything returning at once (the dispatch alone) | 0.32 | 0.32 |

| the members' rest dispatch with | t 3.7 | t 16 |
|---|---|---|
| everything | 1.68 | 1.75 |
| no `selfApply` | 0.35 | 0.35 |

So the 6 ms are: the dispatch 0.32 and the prologue ≈0.1; the coupling's step and the
wheels ≈1.5 (the wheels' rays and bars ≈0.2 of that, below); the node phase 1.3; the beam
phase 0.75; and the rest of `contactMemberStep` (the triangles' finish, the grid's counts,
scan and sorted copy, the list's build or tests) ≈2.0. The members' hits are another 1.7,
of which `selfApply` is 1.35.

**Built (each lossless, its own function behind an early return, off by default):**

- *Perf flag 33554432, `PERF_SELF_HITS_LOCAL`*, with `PERF_HIT_ROLES` and `PERF_HIT_FB`. This
  is §8.33's `selfApplyLocal` on the self split path: a member's rest and a solo car's
  step k − 1 rest, for a two-body dissipative car with 1..48 hits. On this path `gsCount`
  keeps the pair list's state words past the buckets (`soloStepLoad`), and
  `selfApplyLocal`'s accumulator copies overlap them, so each lane keeps one of those words
  in a register across the call and puts it back (`selfApplyLocalKeep`). `gsTested` and
  `gsPairs` hold nothing yet at that point.
- *Perf flag 67108864, `PERF_SPLIT_WHEELS_AHEAD`*, with `PERF_MEMBER_LIST` and self split. A
  member's raycast wheels' rays and bars of step s + 1 run on a workgroup of their own in
  step s's rest dispatch (a second row of its arguments, `splitWheelsAhead`). Step s + 1
  then runs only the tyres (`splitTyres`), which read their hubs' forces after the levels.
  The rays and bars read the nodes, which nothing writes between a node phase and the next
  (the walk, the pairs, the rest and the levels write accumulators and position steps), and
  they write the wheels' scratch records, which only the tyres read. These are the same
  calls as `driveAheadPart`, which the solo frame's `PERF_DRIVE_AHEAD` and §8.31 run. The
  members' and rest dispatches' arguments now always have two rows, and every other use
  returns at once on the second.
- *Not built: the coupling's step off the member's workgroup*, the larger part of the
  1.5 ms. On this path `splitCoupleStep` leaves its spheres, the nodes' ranges and the
  reaction sums in the member workgroup's own memory (`gsCpl`), and the node phase reads
  them from there. A coupling step on another workgroup would have to reload all of that
  into the member's workgroup, and that reload is most of the coupling's chain. §8.31's
  `PERF_AHEAD_APART` could move it only because the wide node phase reads the buffers.

**Proofs.**
- `fast-check.sh perf=33554432`, `perf=67108864` and `perf=100663296` (both), all 20
  fixtures (1, 2, 4, 8 and 32 cars): every state word the same.
- Chrome compiles all 124 modules.
- Suites with both on (`RUBBLE_PERF_FLAGS=131563457`), all PASS:
  - webgpu: `explicit_loop_cap` 5, `explicit_beams_bitwise` 12, `self_split_gpu` 8, `group_split_gpu` 3, `fleet_contact_gpu` 1, `explicit_contact_gpu` 13, `explicit_drive_gpu` 3, `sleep_fast_gpu` 2, `explicit_sleep_gpu` 3.
  - world: `explicit_vehicles` 11, `vehicle_lab_reset` 3.
  - metal: `vehicle_parity` 4, `explicit_drive_parity` 1.
- `PERF_SELF_HITS_LOCAL` alone passed the same subset, without the bitwise, drive, sleep and reset suites.

**Per phase at eight cars** (`--ab-profile`, ms a frame; floors from §8.31's per-pass floors, one car's work
on a workgroup of its own):

| phase | before (t 3.7 / t 16) | floor | after, both on (t 3.7 / t 16) |
|---|---|---|---|
| members' rest (`explicit self members' hits`) | 1.69 / 1.76 | 0.35 (its dispatch and load, no `selfApply`) | **1.18 / 1.35** |
| members' step (`explicit self step`) | 5.89 / 6.05 | ≈1.1 (the dispatch 0.32; nodes, beams, list ≈0.15 + 0.15 + 0.3; the chains in workgroup memory) | **5.65 / 5.83** |
| of which the wheels' rays and bars | ≈0.2 | 0 (beside the rest) | ≈0 |
| of which the coupling's step | ≈1.3 | — | ≈1.3 (not moved) |
| untimed frame | 14.77 / 25.01 | | 14.07 / 24.59 |

**Whole 40 s runs** (vehicle GPU ms a World step from t ≥ 0.5 s, median / p95 / worst;
off is the integration default `30900161`, on adds both switches `131563457`; runs interleaved, each alone on the GPU):

| | off | on |
|---|---|---|
| native, 8 cars (`sedan_new`), round 1 | 22.24 / 24.68 / 29.55 | 21.80 / 24.15 / 29.06 |
| native, 8 cars, round 2 | 22.22 / 24.66 / 29.13 | 21.85 / 24.08 / 29.05 |
| native, 1 car (the wide passes: neither acts) | 4.42 / 4.58 / 4.81 | 4.46 / 4.61 / 5.16 |
| Chrome, 8 cars (`car=gen`, ~18.6 s simulated), round 1 | 18.95 / 23.25 / 27.99 | 18.76 / 23.28 / 27.32 |
| Chrome, 8 cars, round 2 | 18.71 / 23.08 / 26.94 | 18.70 / 23.08 / 28.54 |
| Chrome, 1 car | 3.61 / 4.37 / 6.88 | 3.56 / 3.95 / 6.02 |

Natively the pair gains −1.7 to −2.0 % at the median and −2.2 to −2.4 % at p95 for eight cars.
In Chrome it is −1 % in one round and level in the other. Both switches stay **off**: the
rule for a default is faster in wall time natively and in Chrome, and the Chrome gain is
inside the runs' spread. The lab page's boxes "touching cars' hits in core memory" and
"touching cars' wheels ahead" are unticked.

**What remains above 10 ms at eight cars** (t 16, `--ab-profile`, ms a frame of 34 steps,
both switches on; the untimed frame is 24.6):
- the groups' levels (`split levels`) 7.0;
- the fleet walk 6.3;
- the members' step 5.8: the coupling ≈1.3, the nodes 1.3, the beams 0.75, the rest of
  `contactMemberStep` ≈2.0, the dispatch 0.3;
- the members' hits 1.35;
- the front's other dispatches 4.4 (triangles and points 1.2, owners' parts 0.9, grids 0.7,
  pairs 0.6, the two prefixes 0.9, throttles 0.24);
- self pairs and walk 1.3.

At t 3.7 the frame is 14.1: the members' step 5.7, members' hits 1.2, the fleet walk 1.3,
the levels 1.0, and the front's dispatches 3.3. The levels and the walk at t 16 are the
pile's (§8.32: one owner a level, latency-bound chains). They are larger than §8.32's
figures on this branch's scene, main's rolling terrain since #117.

**What remains above 2.5 ms for one car's own step.** Natively, one car is 4.4 ms a step
on the wide passes (§8.33's passes: hits 1.2, list 1.5, nodes 0.85, beams 0.54, walk and
pairs 0.51, centred damping 0.5). On the multi-car path, a member's own step is 5.8 + 1.35
ms a frame for all eight members side by side. What is left is single-workgroup chains:
the coupling, and the node and beam phases (one workgroup over 548 nodes and 4,251
beams). The next levers would be the member's node and beam phases on the wide passes'
threads, since §8.31's wide node and beam passes are 0.85 and 0.54 for one car, and the
coupling's workgroup copies built where its node phase runs.

### 8.37 The contact levels' hits worked out once, ahead of the levels (2026-11-18)

`contact-groups-design.md` §8.11 (what is built and why it is the same). Perf flag 4
(`PERF_LEVELS_ONCE`, with `PERF_FLEET_HITLAT`), kernel `explicit_beams_fleet_once.slang`
(`fleetLevelsOnce`), a checkbox on the lab page ("contact levels' hits worked out
ahead"). Main at 366bc068, the eight- and 32-car mountain (main's rolling gully), native
M3 Max; the GPU shared with another session's jobs through the lock (wall figures for
ranking).

**Measured first: where a hit's time goes.** A level is the hits' application, then three
residual phases, each behind a barrier and each as long as its slowest hit. In each
phase the hit's lane found its hit again (owner, list entry, triangle, node: four loads
in a row), and the application and residual phase 0 each worked out the triangle, the
clip test or vote, the two-body counts' slot (a walk of the owner's partners) and the
reduced mass, then the force, in two coordinate frames (owner, World); phases 0 and 1
each summed the four nodes' accelerations from loads of a flag, a tag, a beam sum and a
mass before the accumulators. None of that changes within the levels; only the
accumulators do. The levels in use (`AB_LEVELS`, owners / hits by level, four frames'
last passes): 8 cars t 3.7: four levels of 1–2 owners and 4–22 hits; t 16: none or one;
t 30: two levels of one hit; 32 cars t 1: 1–2 levels; t 4: 14 levels of 1–2 owners and
1–33 hits.

**Built, two steps** (each bit for bit on all 20 fixtures before the next):
1. the hit found and both stagings (application in the owner's coordinates, residual in
   the World's) taken once after the snapshot for every hit of the group side by side, in
   registers; each two-body hit applied by its own lane; phase 1's r kept from phase 0;
   a level's merge beside the next level's application (three barriers a level, not
   four);
2. the residual phases' unchanging values staged too (each node's immovable flag, beam
   sum, mass and force-log flag; `vvResOf`'s and `vvGuardOf`'s inverse masses), so a
   residual phase loads only the accumulators it reads.

**Per phase** (`--ab-profile`, the levels' dispatch `explicit split levels`, ms a frame of
34 substeps; before: main's defaults 30900161; the untimed frame in brackets):

| fixture | levels (top) | before | step 1 | step 2 (the flag) | a level, µs before → after |
|---|---|---|---|---|---|
| 8 cars t 3.7 | 4 | 9.63 (23.1) | 6.69 (20.3) | **6.21 (19.6)** | 71 → 46 |
| 8 cars t 16 | 0–1 | 1.06 (17.1) | 1.12 (17.1) | 1.18 (17.2) | — |
| 8 cars t 30 | 2 | 5.42 (22.9) | 4.33 (22.1) | **4.27 (21.9)** | — |
| 32 cars t 1 | 1–2 | 5.67 (23.4) | 4.67 (21.7) | **4.51 (21.6)** | — |
| 32 cars t 2.5 | | 2.03 (19.7) | 1.89 (18.9) | 2.02 (19.0) | — |
| 32 cars t 4 | 14 | 31.6 (54.7) | 16.1 (38.7) | **13.7 (36.7)** | 66 → 29 |

(A level's µs: the dispatch a substep over the levels, the substep's fixed part, the
snapshot, the rule's rounds and the staging, included.)

**The floor** with this work a hit: a level is three barriers, and between them one
dependent round trip each: the application's atomic adds (beside the level before's
merge: an exchange, then an add), phase 0's accumulator loads then its adds into B,
phase 1's loads of A and B then its adds into D. About 3 × (a barrier and two dependent
device-memory accesses, ~3–4 µs) ≈ 10–12 µs a level, plus the substep's fixed part (the
snapshot, the rounds, one staging: about 1.5–2 levels' worth at 8 cars t 30). At 32 cars
t 4 the level is 29 µs, at 8 cars t 3.7 46 µs (more owners and hits a level, the
slowest lane's hit sets it): within 2.5–4× of that floor, from 6–7×. What is left in a
lane is the residual phases' arithmetic (the guard's divisions) and the force log's adds.

**Proofs.**
- `fast-check.sh perf=4`, all 20 fixtures (1, 2, 4, 8 and 32 cars), after each step: every
  state word the same; 8 and 32 cars at `FRAMES=20`: the same.
- Chrome: all 124 modules compile (`EXPLICIT_WGSL_MODULES_ONLY=1`), after each step.
- `perf-verify.sh 30900165` (the defaults and the flag): `fleet_contact_gpu` 1 (new: the
  flag against off on the column of ten, at least 3 levels deep, with and without self
  split, and the two columns of six), `group_split_gpu` 3, `self_split_gpu` 8,
  `explicit_contact_gpu` 13, `explicit_beams_bitwise` 12, `free_flight_gpu` 1,
  `explicit_loop_cap` 5, `explicit_vehicles` 11, Metal `vehicle_parity` 4: PASS.
- The scenario matrix (`tools/scenario-matrix/run.sh perf=4`, PR #127): PASS, every row bit
  for bit, none slower past its threshold:

| row | on ms | off ms | delta |
|---|---|---|---|
| drive | 3.700 | 3.672 | +0.8 % |
| tumble | 3.487 | 3.478 | +0.3 % |
| wall | 4.655 | 4.624 | +0.7 % |
| pile2 | 8.660 | 8.494 | +2.0 % |
| pile8 | 15.146 | 15.030 | +0.8 % |
| pile32 | 38.184 | 57.261 | **−33.3 %** |
| headon | 13.838 | 16.327 | **−15.2 %** |
| side | 9.809 | 9.900 | −0.9 % |
| bricks | 3.947 | 3.911 | +0.9 % |
| rocks | 4.361 | 4.334 | +0.6 % |
| barrier | 7.293 | 7.269 | +0.3 % |
| truss | 1.774 | 1.756 | +1.0 % |

- Whole 40 s runs end in the same state with the flag as without (`--compare-final`: every
  car's largest node difference 0.0 mm, 8 and 32 cars).

**Whole 40 s runs, native** (release, vehicles' GPU ms a World step, t ≥ 0.5 s, 2,371
steps; runs interleaved, each alone on the GPU through the lock):

| | median | p95 | p99 | worst |
|---|---|---|---|---|
| 8 cars, off, round 1 | 18.10 | 23.26 | 23.81 | 28.27 |
| 8 cars, on, round 1 | **17.77** | **21.95** | 22.45 | **22.78** |
| 8 cars, off, round 2 | 18.00 | 23.23 | 23.74 | 28.14 |
| 8 cars, on, round 2 | **17.74** | **22.00** | 22.50 | **23.55** |
| 32 cars, off | 95.41 | 105.88 | 409.6 | 423.9 (t 18.32) |
| 32 cars, on | **77.36** | **86.50** | 390.3 | 405.0 (t 18.32) |

Eight cars: −1.6 % at the median, −5.4 % at p95, −17 to −19 % at the worst (the crash's
levels at t 5.35). Thirty-two cars: −19 % at the median and −18 % at p95; the worst is a
step at t 18.32 that is not the levels' (400 ms either way).

**Chrome.** Rung 3's frame-locked A/B (`run.sh --probe --locked --ab 4 --abUntil 40`,
`mode=mountain&cars=8&timing=step`, the ledges and the rolling gully): 40 samples each, 0
differ. Whole page runs (`run.sh --probe --secs 40`,
`mode=mountain&car=gen&timing=step&cars=8`, `&levelsOnce=1` for on; vehicle GPU ms a World
step, t ≥ 0.5 s, about 19–20 s simulated; interleaved off, on, off, on):

| | steps | median | p95 | p99 | worst |
|---|---|---|---|---|---|
| off, round 1 | 1,111 | 20.26 | 24.13 | 26.56 | 28.55 |
| on, round 1 | 1,176 | **16.87** | **18.48** | 19.42 | **20.51** |
| off, round 2 | 1,112 | 20.30 | 24.44 | 26.77 | 29.86 |
| on, round 2 | 1,175 | **16.84** | **18.45** | 19.20 | **19.99** |

−17 % at the median, −24 % at p95, −29 % at the worst; the page also simulates 6 % more
in the same 40 s. (A real-time page run is not frame-locked; the second round's on run
ended with the first car elsewhere, so its cars took another path; the frame-locked A/B is
the proof of sameness.)

**Default: on** (main's default becomes 30900165). Bit for bit natively (fixtures, suites,
the scenario matrix, whole runs' final states) and in Chrome (frame-locked A/B on both
terrains), and faster in wall time natively and in Chrome at the median, p95 and the
worst. The lab page's box "contact levels' hits worked out ahead" is ticked.

**What remains in the levels.** At 32 cars t 4 the levels are still 13.7 ms of a 37 ms
frame (14 levels a substep, 29 µs each); at 8 cars t 3.7 6.2 ms of 20. A level is now three
barriers around short chains; cutting further means fewer barriers a level (phase 0's
residual and phase 1's guard need every hit's B, so a barrier between them is inherent;
two levels whose owners share no vehicle could be one, which the rule already makes
them) or fewer levels (the rule's levels are the owners' partner conflicts; a finer rule,
by contact point and triangle node rather than by vehicle, would let more hits share a
level).

### 8.38 Physics first: what the lab's drawing costs the World, and the drawing giving way (2026-11-19)

The owner's Chrome (M3 Max, a 120 Hz display), the derby with four cars apart: 121 FPS,
World 52 steps a second (86 % of real time), GPU not ready 30 %. Measured here in
headless Chrome on the same machine (`engine/tools/vehicle-lab/physics-first.sh`: the
page's own real-time loop for 30 s, drawing as the page draws, then 8 s with nothing
drawn, `window.__drawOff`; the drawing's parts with the World paused, `draw-cost.js`),
1280×720, GPU timing per step on. Headless Chrome shows 60 frames a second; a faster
display is `vehicle-lab-record.mjs --uncapped` (frames as fast as Chrome makes them, 80
to 110 a second here). Other sessions' GPU jobs shared the GPU during some runs (through
the lock, and the owner's own page): steps a second move by a few between runs.

**What a picture costs** (GPU ms, the World paused, drawn alone):

| | 4 cars | 8 cars | mountain, 8 cars |
|---|---|---|---|
| skinning (all cars) | 2.9 | 6.1 | 5.8 |
| shadow map | 0.35 | 0.6 | 0.8 |
| picture: the cars' opaque triangles | 3.2–4.2 | 4.3–6 | 6.7–10.6 |
| picture: glass; colliders and sky (the props are instanced already) | 0.4; 0.4 | 0.4; 0.3 | 1.1; 0.6 |
| **a picture** | **7.5–8.5** | **12.2–14** | **15.5–19.5** |
| a World step (physics), cars apart / in contact | 6.3 / 12.5–13 | 7.2–8.8 / 16.4 | 16–20 |

The drawing is not advanced graphics but it is not small either: 427,478 opaque
triangles a car (2 cm edges, one level of detail) and a skin of 490,997 vertices a car
read from 157 MB of binds. Readback copies are a few kB. A picture costs more than a
World step while the cars are apart. Drawn every display frame it takes half the GPU's
second (four cars apart at 60 Hz: 61 pictures × 8.3 ms = 50 %, the physics 38 %); at
120 Hz it takes more than the GPU has, which is the owner's 86 %. Physics is delayed
behind draws: steps and pictures share one queue, a picture submitted between two steps
runs before the second, and a step the World is not ready for is lost time.

**Built.**
* `rubble::DrawPacer` (engine-world-wasm `draw_pacer.rs`, with tests): which display
  frames are drawn. No picture queued behind one still on the GPU, or in a frame whose
  step was refused, until the picture is older than the floor's interval; while steps
  are refused (two in half a second) the drawing gives way a level at a time: every
  display frame, 60 pictures a second, 30 (the floor, a control: 60, 30, 20 or 15), then
  75 % and 50 % of the pixels (automatic render scale only); below every frame only
  frames that took a step are drawn (each car skinned once a World step, each picture
  the latest state). It comes back a level after 2 s with no step refused; a level that
  loses again within its try waits twice as long (to 16 s). The page's controls:
  "drawing gives way to physics" (on), "at least N pictures a second", live.
* Three World steps in flight (`setStepsInFlight`, a control: 2 to 4; pacing only, the
  same steps). Chrome lands a step's readback well after the GPU is done with it, and
  with two in flight the World refused steps while the GPU was idle: four cars in
  contact, nothing drawn, 47 steps a second with two (8.9–13 ms a step), 60 with three.
  A/B in one run (giving way on, two then three, three times): 41–45 against 49–57
  steps a second. The lab's readings (measurements, the camera, the derby's AI) arrive
  one step (16.7 ms) later. The frame-locked recorder waits for every readback and is
  unchanged.
* A frame not drawn still reads back what its steps left (measurements, the derby's AI):
  the lab's inputs no longer follow the drawing rate. The skin and the shadow map are
  not redone when nothing they show changed (no step, the same lead, one level of
  detail: the World paused). The debris effects' time step spans the steps of frames
  not drawn. Polling a landed readback before refusing a step made no measurable
  difference (kept: it is the right order).

**Before and after** (headless Chrome; real time is 60 steps a second; frame interval
p50 / p95 / worst ms after; drawing's GPU ms a picture and its share of the GPU's second):

| | World steps/s, before → after | pictures/s | frame, after | drawing, before → after | physics alone (nothing drawn), before → after |
|---|---|---|---|---|---|
| derby, 4 cars apart, 60 Hz | 60 (99 %) → 59 (99 %) | 61 → 31 | 16.7 / 16.8 / 17 | 8.3 ms (50 %) → 8.3 (26 %) | |
| derby, 4 cars apart, fast display (80–110 Hz) | 42 (70 %) → 61 (100 %) | 42 → 27 | 17.4 / 18.8 / 19 | 8.0 (34 %) → 8.5 (23 %) | |
| derby, 4 cars in contact | 38 (63 %) → 48 (81 %) | 33 → 30 | 16.7 / 16.7 / 17 | 8.3 (27 %) → 9.2 (28 %) | 47 (79 %) → 60 (100 %); 12.4–13 ms a step |
| derby, 4 cars in contact, fast display | 36 (61 %) → 47 (78 %) | 28 → 27 | 17.2 / 18.7 / 19 | 7.8 (22 %) → 9.3 (25 %) | 38 → 61 |
| derby, 8 cars apart | 40 (67 %) → 59 (99 %) | 48 → 30 | 16.7 / 16.7 / 17 | 12.1 (58 %) → 12.1 (37 %) | |
| derby, 8 cars in contact | 31 (52 %) → 36 (60 %) | 29 → 28 | 16.7 / 33.3 / 50 | 10.6 (31 %) → 12.5 (35 %) | 40 (67 %) → 60 (99 %); 16.4 ms a step |
| mountain, 8 cars | 28 (46 %) → 35 (59 %) | 31 → 28 | 16.7 / 33.3 / 50 | 15.7 (49 %) → 14.5 (40 %) | 40 (67 %) → 48 (80 %); 17–20 ms a step |

(Before, the page already halved its scale and drew every other step while steps were
refused, so the pictures a second are similar in contact. The frame p95 of 33 ms with
eight cars is the floor's picture drawn while the GPU is still busy: 30 pictures a
second, as the floor asks; drawing only while the GPU is free gave 15 pictures a second,
16.7 ms frames and 1–2 more steps a second.)

**What is left, and whether it is physics.** Four cars: with nothing drawn the physics
keeps real time in contact (13 ms a step); the gap left (81 %) is the 30 pictures a
second at 9 ms. Eight cars in contact: the physics alone fills the GPU (16.4 ms × 60 =
98 % of it), so any drawing costs steps; the floor's 30 pictures at 12.5 ms are the
40 % missing. The mountain's eight cars: physics alone is 17–20 ms a step, 80 % of real
time with nothing drawn: physics is the limit there. To make a picture close to free,
the work left is in the drawing itself, lossless first:
* skinning: 0.73 ms a car, 1.7× its bandwidth floor, and the cars of one model each read
  the same 157 MB of binds; one pass reading a bind once for every car would take eight
  cars' 6.1 ms toward 1 ms;
* the cars' opaque picture follows the triangles (a quarter of the pixels takes off only
  10–15 %): levels of detail for the generated car (a different picture by design, the
  asset side: §8.30's follow-up).

### 8.39 The kept candidates with today's defaults (2026-11-18, 19)

`contact-groups-design.md` §8.12 (what changed and why the results are the same). Perf flag
32768 (`PERF_FLEET_KEEP`), the lab's "contact candidates kept between steps". Branch
`perf/keep-with-defaults`; native M3 Max, the GPU shared with other sessions' jobs through
the lock (wall figures for ranking; the fixtures' and the matrix's are A/B from one state).

**What blocked it.** On main, turning 32768 on made the host drop two of today's defaults:
the throttles across the fleet (`PERF_FLEET_POST`: its dispatch is sized and indexed by the
walk's prefix of the listed triangles, which a reused pass never computes; the updates went
back into the levels, +0.7 to 0.9 ms a frame) and the fused scans (the kept candidates'
node tracking lived in the scans' dispatch, which came back, +0.37 to 0.53 ms). On the
rolling gully a third thing kept the list from working at all: a piled owner's padded
list was past its region's 32,767 pairs (33,103 to 66,207 measured), so every build was
void and the list was never reused (0 builds kept and 677 of 680 passes on today's front,
t 20 to 30, 20 frames each).

**What changed** (§8.12): the throttles a thread a triangle in the elements' size
(`fkPost`); the tracking in the members' rest, per lane, and none after a pass with nothing
offered (`fkTrackVehicle`, the host's `PERF_KEEP_TRACK_REST`), so the scans are fused again;
the kept pairs in the fleet's whole pair room, each owner a share from its last build's count
(`fkAlloc`, `fkShareAt`; a change of the buffers' layout: fixtures and matrix states
remade); a list kept down to 3 passes (`FK_SHORT`, 8 before). The partial rebuild (only the
cars past their allowance, one list) was weighed and not built (§8.12).

**Per phase** (`--ab-profile`, ms a frame of 34 substeps by dispatch kind, 8 cars; "main"
columns from main's binary on its own fixtures, the others from this branch's on its own;
the default (off) columns agree to 0.1 ms; untimed frame in the last row):

| phase | ledges t 16: off / main keep / keep | ledges t 30 | rolling t 25 | rolling t 28 |
|---|---|---|---|---|
| fleet walk | 2.59 / 0.69 / 0.69 | 4.81 / 0.68 / 0.69 | 6.06 / 6.02 / **2.19** | 6.88 / 6.88 / **2.80** |
| split levels | 1.17 / 1.90 / **1.21** | 4.23 / 5.07 / **4.33** | 5.14 / 5.88 / **5.12** | 5.04 / 5.86 / **5.05** |
| fleet scans | 0 / 0.53 / **0** | 0 / 0.52 / **0** | 0 / 0.37 / **0** | 0 / 0.36 / **0** |
| fleet throttles | 0.23 / 0 / 0.22 | 0.24 / 0 / 0.22 | 0.24 / 0 / 0.22 | 0.24 / 0 / 0.22 |
| self members' hits (the rest; tracking with keep) | 1.73 / 1.70 / 2.01 | 1.77 / 1.77 / 2.05 | 1.75 / 1.72 / 2.04 | 1.78 / 1.76 / 2.06 |
| fleet owners' parts (first plan; fused scans) | 0.89 / 1.05 / 1.22 | 0.86 / 1.04 / 1.21 | 0.89 / 1.00 / 1.26 | 0.88 / 1.00 / 1.25 |
| fleet triangles, points | 1.08 / 0.56 / 0.56 | 1.56 / 0.60 / 0.59 | 1.60 / 1.62 / 0.73 | 1.64 / 1.66 / 0.81 |
| fleet grids (reused: the kept tests) | 0.72 / 0.60 / 0.61 | 0.70 / 0.59 / 0.61 | 0.73 / 0.75 / 1.41 | 0.73 / 0.75 / 1.82 |
| fleet pairs, both prefixes | 1.25 / 0.14 / 0.14 | 1.32 / 0.11 / 0.11 | 1.41 / 1.69 / 0.31 | 1.56 / 1.86 / 0.43 |
| **untimed frame (wall)** | 17.40 / 15.18 / **14.88** | 22.23 / 17.85 / **17.52** | 25.43 / 25.74 / **21.70** | 26.63 / 27.99 / **23.50** |

The levels are back at the default's, the scans' dispatch is gone, and on the rolling gully
the list now works (builds every 7 passes, 82 to 85 % of passes reused; the kept tests
over 140,000 to 210,000 pairs a build are the grids' 1.4 to 1.8 ms). What the composition
costs: the members' rest +0.3 ms (the tracking, a few dependent loads behind the rest's
barrier: 9 µs a pass; per lane maxima and sums did not change it) and the first plan
+0.2 to 0.35 ms (the scans fused in it again, the kept checks).

**Proofs.**
- `fast-check.sh perf=32768`, the 20 standard fixtures (1, 2, 4, 8, 32 cars, remade for the
  layout) and the rolling gully's four (`cars8_t{20,25,28,30}rolling`, native
  `--terrain rolling`, replayed with `EXTRA="--terrain rolling"`): every state word the same
  after each step; 20 frames on 8 cars ledges t 3.7, 16, 30 and the four rolling: the same.
- Chrome: all 124 modules compile (`EXPLICIT_WGSL_MODULES_ONLY=1`) after each step.
- `perf-verify.sh 30932933` (the defaults and 32768) and `perf-verify.sh default`:
  `explicit_loop_cap` 5, `fleet_contact_gpu` 1 (the kept candidates against off on the two
  sedans head-on and the column of six, with and without self split, now with the throttles
  across the fleet, the fused scans and the tracking in the rest all on), `group_split_gpu`
  3, `self_split_gpu` 8, `explicit_contact_gpu` 13, `explicit_beams_bitwise` 12,
  `explicit_pair_reuse_gpu` 2, `vv_grid_gpu` 2, `staged_state_gpu` 1, `free_flight_gpu` 1,
  engine-world `explicit_vehicles` 11, engine-metal `vehicle_parity` 4: PASS.
- Chrome, rung 3 (`run.sh --probe --locked --ab 32768 --abUntil 40`, 8 cars): ledges 40
  samples, 0 differ; rolling gully 40 samples, 0 differ.
- Whole 40 s native runs end in the same state with the flag as without
  (`--compare-final`: every car's largest node difference 0.0 mm; 8 cars ledges and rolling,
  32 cars).

**The scenario matrix** (`tools/scenario-matrix/run.sh perf=32768`, states remade for the
layout): PASS, every row bit for bit, none slower past its threshold.

| row | on ms | off ms | delta |
|---|---|---|---|
| drive | 3.300 | 3.817 | −13.5 % (one car: the switch does nothing; noise) |
| tumble | 3.803 | 3.814 | −0.3 % |
| wall | 4.106 | 4.110 | −0.1 % |
| pile2 | 9.004 | 8.884 | +1.4 % |
| pile8 | 13.743 | 14.828 | **−7.3 %** |
| pile32 | 37.777 | 37.480 | +0.8 % |
| headon | 13.157 | 14.197 | **−7.3 %** |
| side | 9.732 | 9.957 | −2.3 % |
| bricks | 3.923 | 3.986 | −1.6 % |
| rocks | 4.384 | 4.388 | −0.1 % |
| barrier | 7.858 | 7.745 | +1.5 % |
| truss | 1.805 | 1.807 | −0.1 % |

(Before the tracking was skipped after idle passes the matrix failed on pile2, +9.4 %: two
cars grouped but apart paid the tracking every pass for nothing.)

**Whole 40 s runs, native** (release, vehicles' GPU ms a World step, t ≥ 0.5 s, 2,371 steps;
off, on, off, on, each alone on the GPU through the lock):

| run | median | p95 | p99 | worst |
|---|---|---|---|---|
| 8 cars ledges, off | 17.37, 17.54 | 21.49, 21.49 | 21.96, 21.91 | 22.99, 23.01 (t 5.37, 5.35) |
| 8 cars ledges, on | **14.87, 14.86** | **18.00, 17.77** | **19.79, 18.90** | 23.57, 23.46 (t 5.37, 5.35) |
| 8 cars rolling, off | 24.38, 24.44 | 28.06, 28.07 | 29.56, 29.31 | 41.83, 41.36 (t 27.5) |
| 8 cars rolling, on | **20.71, 20.76** | 27.65, 27.81 | 29.01, 29.05 | 33.19 (t 27.05), 42.35 (t 27.50) |
| 32 cars ledges, off | 77.73, 79.17 | 87.48, 88.86 | 391.2, 397.1 | 405.4, 412.0 (t 18.3) |
| 32 cars ledges, on | 78.11, 78.19 | 87.35, 87.72 | 391.9, 399.3 | 407.2, 412.4 (t 18.3) |

Eight cars: −15 % at the median on both terrains, −16 % at p95 on the ledges, −1 % on the
rolling gully. The worst step on the ledges (the crash's levels, t 5.35) is +2 % in both
rounds; replayed from t 5.2 frame by frame (`AB_EACH=1`, 12 frames) the frames there are no
slower with the flag (median 21.50 against 22.08 ms), so it is one step's build beside the
levels' peak. Thirty-two cars: a tie (+0.5 %, −1.2 %); the kept list does nothing there
(every build past the fleet's room or a car with more partners than a build covers, 8:
0 builds in 2,000 passes). Covering 16 partners was tried (rung 1, 32-car fixtures): t 2.5
builds 505,000 pairs and is 8 % slower, t 1 and t 4 still past the room; reverted.

**Chrome** (the page's real-time loop, `run.sh --probe --secs N`, vehicles' GPU ms a World
step, t ≥ 0.5 s; `&fleetKeep=0|1`, interleaved; the page runs at most 8 cars on the mountain):

| run | steps | median | p95 | p99 | worst |
|---|---|---|---|---|---|
| ledges, 90 s, off | 2,755, 2,779 | 15.66, 15.61 | 16.81, 16.76 | 17.88, 17.66 | 21.01, 21.52 |
| ledges, 90 s, on | 2,884, 2,801 | **14.47, 14.70** | **16.39, 16.62** | 17.87, 17.89 | **20.54, 21.22** |
| rolling, 90 s, off | 2,683, 2,678 | 17.00, 17.16 | 18.28, 18.32 | 18.96, 18.78 | 22.16, 23.52 |
| rolling, 90 s, on | 2,846, 2,860 | **14.92, 14.90** | 18.37, 18.38 | 19.04, 19.05 | **20.59, 21.21** |
| rolling, 40 s (t ≤ 19.5), off | 1,178, 1,176, 1,174, 1,172 | 16.71, 16.81, 16.84, 16.91 | 18.42, 18.49, 18.46, 18.56 | | 19.99, 20.32, 19.67, 19.92 |
| rolling, 40 s (t ≤ 19.5), on | 1,175, 1,172, 1,172, 1,168, 1,166 | 16.99, 16.98, 16.90, 17.05, 16.97 | 18.71, 18.76, 18.73, 18.84, 18.76 | | 20.38, 20.49, 20.29, 20.79, 20.27 |

Over the whole run: −6 to −8 % at the median on the ledges, −12 % on the rolling gully, and
the page simulates 1 to 6 % more in the same time; the worst step lower on both. In the
rolling gully's first 19.5 s (the falls and the first collisions) the flag is 0.4 to 1.6 %
slower at the median and 1 to 2 % at p95: there the builds are past the fleet's room or the
lists short, and what is left is the flag's own cost (the plan's checks, the per-triangle
listing marks; the tracking is skipped while the passes wait).

**The Demolition Derby** (Chrome, `mode=derby&derbyPlayer=0`, spectator, the derby probe,
`run.sh --probe --secs 40`, one run each way; the AI's paths differ run to run, so a pair is
not one run): 8 cars off 13.77 / 15.25 / 17.04 (median / p95 / worst, 1,373 steps), on
14.60 / 16.01 / 19.00 (1,311 steps): **6 % slower**. 4 cars on 11.74 / 13.16 / 14.12 (1,626
steps) against 9.35 / 12.23 / 13.17 off (1,512 steps, a run before a machine restart). On a
flat arena the cars ram at speed, as in the mountain's first collisions: lists past the
fleet's room or outlived in a few passes, and the flag's own cost on top.

**Default: on** (the owner's decision, 2026-11-19: "just as good or faster and prepares us for
the future"; main's default becomes 30932933, the lab page's box is ticked). Bit for bit
everywhere it was checked (fixtures on both terrains, the suites, the matrix, whole runs'
final states, Chrome's frame-locked A/B on both terrains), and faster at the median on both
terrains natively (−15 %) and in Chrome (−6 to −12 %). The earlier rule was faster
everywhere, and these places are not: the worst step of the native ledges run
(+2 % in both rounds: the crash's levels and a build in one step), 32 cars (a tie, +0.5 % and
−1.2 %: the list never holds there), the rolling gully's first 20 s in Chrome (+0.4 to
1.6 %, the flag's own cost before the cars pile), and the derby (+6 % at 8 cars, one noisy real-time run each way). The
stop rule ended the work: after the composition (−12 to −19 % on the 8-car fixtures) and the
fleet's room (−12 to −16 % on the rolling gully), three attempts in a row each under 5 %
(the per-lane tracking, no tracking after idle passes, 16 partners covered).

What would be next, measured first: the first collisions (8 cars ledges t 3.7: the first two
builds are past the fleet's room, 262,136 pairs, while the cars close at speed and the
allowance between vehicles is 30 cm; 12 frames on today's front with the flag's costs on
top), and 32 cars (more partners than a build covers, and past the room when covered): a
list whose allowance between two vehicles is the pair's own rather than the fleet's
fastest pair's would shrink both.

### 8.40 Bit-exact results as a setting, off by default: the fast arithmetic (2026-11-19)

**Owner decision (2026-11-19).** Bit-identical results across machines are needed for
testing and reproduction, not at run time. They become an engine-wide setting, **off by
default**; on, everything works exactly as before (bit for bit, GPU == CPU reference),
and every bit-for-bit suite runs with it on. The equations, the 2 kHz rate, the contacts
and their rules (anti-clip, the two-body mass split, the no-positive-work residual), the
iterations and the tolerances do not change. Runtime quality is that destruction looks
and behaves right; last-bit variation between machines and runs is acceptable.

**What is built** (branch `perf/fast-determinism`):
- `ExplicitVehiclesDef::bit_exact` (World; `World::set_explicit_bit_exact` at run time),
  `LabOptions::bit_exact` (lab JSON `bitExact`, the live command `{ bitExact }`, the
  page's box "bit-exact results (testing)", `vehicle_lab --bit-exact 0|1`), default off;
  `RUBBLE_BIT_EXACT=1` turns the default on. `ExplicitBeamsGpu`'s own constructors stay
  bit-exact (its suites are the GPU == CPU contract); `with_settings(…, bit_exact)` and
  `set_bit_exact` choose. A change remakes the pipelines (each compiled at first use) and
  no state. `compute_specs_for(…, bit_exact)` gives a browser host the modules it will
  create.
- **Strict float operations** (`slang/physics/strict.slang`): every intermediate passes
  through `nf`, an XOR with `gP.vehicles >> 31` that the compilers cannot know is zero,
  and `sdiv` / `ssqrt` take one exact fma correction. With bit-exact off the host creates
  its pipelines from the same WGSL with those three helpers' bodies replaced at run time
  (`engine-webgpu/src/fast_float.rs`, `fast_float_wgsl`): `nf(x)` returns `x`, `sdiv` is
  `a / b`, `ssqrt` is `sqrt`. Nothing else changes: every equation, operand and branch
  is the kernel's; the compiler may now fuse multiply-adds and reassociate. `noFuseBits`
  and `opaqueCount` keep the uniform (`opaqueCount` keeps loops loops for compile time,
  `kernel-compile-time.md`). Strict mode's modules are the build's text byte for byte:
  no Slang file changed, so engine-metal and CUDA (strict, parity hosts) are untouched.
  The rewrite fails loudly if Slang's emitted shape changes (a unit test runs it over
  every shipped module).
- Harnesses that compare bit for bit turn it on themselves: `perf-verify.sh`
  (`RUBBLE_BIT_EXACT=1`), `fast-check.sh` (`BIT_EXACT`, default 1, and `--bit-exact`),
  the scenario matrix (`BIT_EXACT`, default 1), the Chrome A/B (`vehicle-lab-record.mjs
  --ab` adds `bitExact=1`), and the World suites' definitions (`bit_exact: true`).
- `--ab-switch bitexact` (on = bit-exact, off = fast; prints each side's physical
  difference: largest node distance, broken beams, kinetic energy, finiteness) and
  `--ab-perf-masks …,2147483648` (bit 31: that mask with bit-exact off, per wide pass).

**Fixed-order contact sums: nothing converted, and why.** Every contact force already
goes into integer fixed-point atomics (`accAdd`, `toFixed`), so the order of the adds is
already free. What the levels (`vvLevelsSet`, `fleetLevels*`, `vvApplyOrderedBuckets`)
and `selfApply`'s rounds order is not the sum but the **contact rule's input**: a hit's
force reads its node's force with every earlier hit's share on that node added
(`hitForce`'s `sum`; the vehicle-contact spec §3.1, the CPU reference, Rigs of Rods'
sequential loop), and `contactResponse` stops the approach with `r = −f·n − (vn +
max(vn, 0))·m/h` from that force. Applying every hit in one parallel pass would give each
hit the node's force without the others' shares: two hits on one node would each cancel
the same normal force (twice the stopping force, energy put in) and each friction cone
would be sized from the wrong normal force. That changes the rule, not the order of
additions, so they stay ordered in both settings. The residual and guard phases read the
accumulated A and B for the same reason. The other order-keeping parts the brief asked
about: the beams' CSR gather sums each node's incidence forces in a fixed order (in the
fast setting the compiler may already reassociate that sum; a scatter would need float
atomics, which WGSL lacks, or fixed-point beam forces, a change of the arithmetic: not
built); `resolvePending` (break exceptions in spawn order: the break groups' rule); the
grid scans (integer; they place candidates, not sums); `selfSerial` past `HIT_CAP` (the
rule's one-lane fallback). None is order kept only for the bits.

**Tolerance tests** (`engine-webgpu/tests/fast_arithmetic_gpu.rs`). The yardstick is not
chosen by hand: the bit-exact kernel run as spawned and from four spawn states no one
could tell apart (every node velocity one unit in the last place up, down; every
velocity times 1 ± 1e-5). The fast setting must end within 4× the largest distance those
runs end from the bit-exact one (floor 2 mm and 2 beams); its energy (kinetic plus
gravity) may not rise past the start by more than 1 % of the start's kinetic energy plus
the weight times 2 cm (the spawn's own elastic energy); every position finite; and it is
deterministic run to run on one machine (every state word, twice). One second (60 frames
of 34 steps), the repo sedan:

| scene | fast vs bit-exact (mm): largest node / centroid / front / ride height | spread, the largest of 4 nudges (mm) | broken beams: bit-exact / fast / nudges |
|---|---|---|---|
| parked | 0.47 / 0.29 / 0.03 / 0.003 | 0 (a car at rest is not moved by 1e-5 of zero): the floor | 0 / 0 / 0 |
| wall, 100 km/h | 229 / 5.4 / 1.4 / 4.3 | 270 / 3.6 / 12.3 / 2.7 | 266 / 266 / 266 ×4 |
| head-on, 50 km/h each | 1418 / 8.3 / 10.6 / 1.1 | 1481 / 19.1 / 22.9 / 4.3 | 359 / 364 / 359, 359, 319, 364 |

All three pass. They show: the fast setting is deterministic on one machine (its own
run-to-run spread is zero; it differs only from the strict order), and a crash amplifies
differences far below measurement to the size of a torn piece's flight (1.4 m at the
largest node) and to tens of broken beams (319 to 364 from a 1e-5 change of speed). By
that yardstick the fast setting is one more indistinguishable start, not another
behaviour. (A single 1-ulp nudge was too narrow a yardstick: the head-on's front moved
10.6 mm in the fast setting and 0.8 mm under one nudge; four nudges, two of them 1e-5 of
speed, give 22.9 mm.)

**Fixtures** (`fast-check.sh bitexact`, 20 frames each way from one state, native, cargo
profile `fast`, M3 Max, under the lock; ms a frame of 34 steps; the largest node
distance after 20 frames was 0.05 mm with one car, 0.21 with two, 0.13 with eight):

| fixture | bit-exact | fast | |
|---|---|---|---|
| 1 car t 2.5 (free fall) | 3.26 | 3.51 | +8 % |
| 1 car t 6 | 4.78 | 4.63 | −3 % |
| 1 car t 20 (wreck at rest) | 4.64 | 4.52 | −3 % |
| 2 cars t 2 / 3.7 / 8 / 16 | 7.25 / 8.26 / 11.96 / 12.00 | 7.15 / 7.78 / 11.26 / 11.69 | −1 / −6 / −6 / −3 % |
| 8 cars t 2 / 3.7 / 8 / 16 / 20 / 30 | 11.81 / 18.83 / 14.90 / 16.70 / 17.30 / 21.46 | 11.18 / 17.12 / 13.89 / 15.63 / 16.20 / 19.70 | −5 / −9 / −7 / −6 / −6 / −8 % |
| 32 cars t 1 / 2.5 / 4 | 23.34 / 19.10 / 35.89 | 19.97 / 17.93 / 33.35 | −14 / −6 / −7 % |

**One car's passes** (`--ab-perf-masks 0,2147483648`, GPU ms a frame, each dispatch timed
apart, 24 frames; the wall line: median / fastest tenth of 60 frames):

| pass | t 20, bit-exact → fast | t 6 | t 2.5 |
|---|---|---|---|
| hits | 1.605 → 1.361 | 1.535 → 1.287 | 0.411 → 0.389 |
| list | 1.387 → 1.358 | 1.371 → 1.345 | 0.965 → 0.872 |
| nodes | 0.872 → 0.805 | 1.033 → 0.900 | 0.801 → 0.775 |
| beams | 0.548 → 0.453 | 0.544 → 0.451 | 0.546 → 0.448 |
| pairs | 0.553 → 0.503 | 0.528 → 0.476 | 0.465 → 0.430 |
| centred beams / nodes | 0.220 / 0.310 → 0.198 / 0.316 | 0.219 / 0.302 → 0.197 / 0.303 | 0.220 / 0.302 → 0.196 / 0.304 |
| aux | 0.073 → 0.064 | 0.073 → 0.063 | 0.074 → 0.065 |
| wall | 4.71 / 4.64 → 4.39 / 4.33 | 4.71 / 4.64 → 4.35 / 4.27 | 3.25 / 3.18 → 3.13 / 3.06 |

The hits (−15 %: the contact rule's long dependent chain, §8.35) and the beams (−17 %)
shrink most; the list and the centred nodes (loads and integer index arithmetic) hardly
move. That bears out §8.35 only in part: the XOR doubles a dependent chain only where the
chain is float arithmetic; most phases are bound by loads and integer index arithmetic.

**Scenario matrix** (`run.sh bitexact`; several values where a row was run again):

| row | bit-exact ms | fast ms | fast vs bit-exact | largest node difference after 10 frames |
|---|---|---|---|---|
| drive | 3.310 / 3.524 | 3.231 / 3.331 | −2.4 / −5.5 % | 0.016 mm |
| tumble | 3.241 / 3.390 | 3.278 / 3.218 | +1.1 / −5.1 % | 0.055 mm |
| wall | 4.321 | 4.092 | −5.3 % | 8.2 mm |
| pile2 | 8.330 | 7.900 | −5.2 % | 30.9 mm |
| pile8 | 15.204 | 14.300 | −5.9 % | 16.6 mm |
| pile32 | 39.550 | 36.207 | −8.5 % | 399 mm (broken beams 1113 / 1117) |
| headon | 13.355 | 12.335 | −7.6 % | 126 mm |
| side | 9.248 | 8.531 | −7.8 % | 2.9 mm |
| bricks | 4.349 / 3.718 / 3.842 | 4.710 / 3.835 / 3.812 | +8.3 / +3.1 / −0.8 % | 0.29 mm |
| rocks | 4.872 / 4.202 / 4.322 | 5.281 / 3.963 / 3.952 | +8.4 / −5.7 / −8.6 % | 11.7 mm |
| barrier | 7.385 | 7.038 | −4.7 % | 1.5 mm |
| truss | 2.706 / 1.777 / 1.924 | 3.131 / 1.844 / 2.000 | +16 / +3.8 / +4.0 % | 0.002 mm |

The first round of bricks, rocks and truss ran beside other load (their slower fast
halves did not repeat); in two reruns of five rounds bricks and rocks are level or
faster. The truss (no vehicle, no self-collision, its frame the beam and node passes) is
3–4 % slower in the fast setting both times, inside the matrix's 5 % threshold.

**Whole 40 s runs, native** (release, `vehicle_lab --measure mountain-perf`, vehicles' GPU
ms a World step from t 0.5 s, median / p95 / worst; each alone under the GPU lock, but
the machine's user pages are not under it: a run whose p95 is far over its median had
company):

| | bit-exact | fast |
|---|---|---|
| 1 car, round 1 | 4.63 / 4.87 / 5.39 | 4.82 / 11.91 / 20.74 (company) |
| 1 car, round 2 | 5.33 / 9.50 / 17.65 (company) | **3.95 / 4.46** / 11.57 |
| 8 cars, round 1 | 18.17 / 28.86 / 52.89 | **15.97 / 19.36 / 23.71** |
| 8 cars, round 2 | 17.74 / 21.96 / 23.33 | **15.91 / 18.76 / 23.03** |
| 8 cars, launch speed × (1 ± 1e-5) | 17.39 / 18.95 / 21.62; 19.37 / 21.67 / 25.06 | |
| 32 cars | 79.29 / 88.93 / 436.4 | stopped by the 20-minute limit at t 20 (below) |

Each setting is deterministic run to run (both rounds of a setting end in the same state,
0.0 mm); the two settings end in different states, and so do bit-exact runs whose launch
speed differs by 1e-5: the eight cars end 0.06 to 1.2 km from the reference run's places
either way (cars that slid on down the mountain in one run stop in another). A whole
run's work therefore differs between the settings as it does between two launch speeds;
its timing compares workloads as well as arithmetic. The fixtures above (one state, both
ways) are the like-for-like figures.

The 32-car fast run's frame rose to 700–860 ms from t 11 s (the bit-exact run had such
frames only at t 18–19, 250 ms). The same state at t 12 run both ways (`--ab 12
--ab-switch bitexact --ab-profile`): bit-exact 637 ms a frame, fast 548 ms; `explicit
fleet pairs` alone 528 ms (15.5 ms a dispatch) and the fleet walk 87 ms. The cliff is the
state's (a pile whose vehicle-contact pairs explode), reached by the fast run's path and
not by this bit-exact one; both settings pay it. A follow-up for the fleet pairs, not a
fast-setting issue.

**Chrome** (headless, the lab page's own real-time loop, `vehicle-lab-record.mjs --probe`,
vehicles' GPU ms a World step from t 0.5 s, median / p95; the page is not frame-locked, so
two runs take two paths):

| | bit-exact | fast |
|---|---|---|
| 1 car mountain, 40 s, round 1 | 4.27 / 5.29 | 4.80 / 6.19 |
| 1 car mountain, 40 s, round 2 | 4.30 / 5.66 | 4.39 / 5.32 |
| 8 cars mountain, first 5 s | 15.55 / 18.34 | **14.39 / 17.92** |
| derby 4 cars, first 5 s | 5.97 / 7.16 | **5.52 / 6.17** |
| 8 cars mountain, 40 s (bit-exact only) | 16.51 / 20.45; 16.53 / 20.57 | |
| derby 4 / 8 cars, 40 s (bit-exact only) | 11.27 / 16.16; 16.50 / 20.19 | |

All 124 fast modules pass Chrome's validation (`EXPLICIT_WGSL_MODULES_ONLY=1`, Tint,
uniformity included). One car in Chrome does not gain (+12 % and +2 % median in two
rounds): its frame is the wide passes, and Chrome's Metal back end compiles other MSL
than naga's; the native gain (−7 %) does not carry over. The multi-car paths gain in
Chrome as natively (−7 to −8 % in the first seconds). **A cold start in the fast setting
is long in Chrome**: its modules are other text than the bit-exact ones, so nothing is
cached the first time; the 8-car page took 190 s to start (two earlier attempts passed
the recorder's 4-minute start limit while compiling), and the deferred modules compile
while the page runs. The whole 40 s Chrome runs of 8 cars and the derby in the fast
setting were not completed (the cold compiles and the session's shared GPU queue); only
their first 5 s, warm.

**What pays, what does not, what is kept.** Kept: the fast arithmetic as the World's and
the lab's default (bit-exact off). Natively it is 3–14 % faster on every fixture but
one-car free fall (+8 %), 5–9 % on the matrix's contact rows, and 7–8 % in Chrome on the
multi-car paths, within the scene's own spread by the suite's yardstick. Not a gain: one
car in Chrome (level to slower), the truss (3–4 % slower, inside the noise threshold),
one car in free fall natively. No fixed-order contact sum was converted (the rule reads
them). §8.35's reading that the strict XOR doubles every dependent chain overstated its
share: one car's frame drops 7 %, not the 30–50 % a doubled ALU chain would give, because
most phases wait on loads and integer index arithmetic.

**Proofs that bit-exact is unchanged.** No Slang source changed; with bit-exact on the
host passes the build's WGSL as before (`host_wgsl(src, true)` is the uniform-load
rewrite alone, as on main). Suites with `RUBBLE_BIT_EXACT=1` (`perf-verify.sh default`):
explicit_loop_cap 5, explicit_beams_bitwise 12, explicit_contact_gpu 13, explicit_pair_reuse_gpu 2, explicit_beam_order_gpu 3, debris_retire_gpu 1, showcase_car_gpu 6 (1 ignored, as before), vv_grid_gpu 2, fleet_contact_gpu 1, self_split_gpu 8, engine-world explicit_vehicles 11, vehicle_lab_reset 3, vehicle_lab_piece_windows 1, explicit_drive_world 1: all PASS. engine-metal `vehicle_parity` was not run: no file it compiles or
runs changed.

### 8.41 The cars of one model skinned in one pass, and a retired check read once (2026-11-19)

§8.38 left the skin as the drawing's largest lossless part: eight generated cars 6.1 ms a
picture, each car reading the shared 157 MB of binds. Headless Chrome on the M3 Max,
1280×720, the GPU shared with other sessions' jobs through the lock (rounds interleaved
to rank).

**Built** (`showcase-car-render.mjs`; the page's control "Skin cars of one model":
together (wide, the default), together (loop), one by one; live):
* *The batch* (`skinBatchWgsl`): the cars of one model (one bind buffer, one level of
  detail each) skinned in one dispatch with SKIN_WGSL's own `skin` code. A car's skinned
  vertices, node state and tables become 256-byte aligned ranges of one pooled buffer each
  (moved there once, at the update after cars are added or removed); every other pass
  binds the car's range and sees what it saw before; the batch binds the pools whole and
  a table (`carTab`, a uniform) holds each car's Params and range bases. Wide: a workgroup
  a (car, block of vertices), the cars of a block side by side, so a block's binds come
  from memory for the first car and from the cache for the others. Loop: an invocation a
  vertex, every car in turn.
* *A weighted vertex's retired check in its entries' pass* (`weightedPacked`,
  `weightedVertex` return it): `vertexRetired` walked the entries again and read the
  whole 64-byte slot for each entry (32 slot reads for a vertex of 32 entries), for every
  vertex of the generated car. The same answer from the pass that already reads them.

**Lossless, checked.** `tools/vehicle-lab/skin-batch.sh` (`skin-batch.js`): at two
states per scene (cars apart at t 1.5 s; in contact, or down the mountain, at t 9 s), the
World paused, every car's skinned vertices read back and hashed under five settings on
the same state: one by one with the base commit's kernel (`BASE`, loaded from
`generated/skin-ref`), one by one with this kernel, wide, loop, one by one again. Derby 4,
derby 8, mountain 8, both states: every car's bytes the same (PASS). The draw passes are
unchanged and read the same vertex bytes, so the picture is the same. Two other variants
were built and dropped: each node's shown position computed once a car before the skin
(the same bits, no faster); a packed vertex's entries read once for every car with
per-car sums in registers (different bits under Chrome's Metal compiler, and slower).

**GPU ms a picture, World paused** (medians of 3 interleaved rounds; skinning includes
the node passes, ≈0.36 ms for eight cars):

| | skinning, before → after | shadow map | picture | a picture, before → after |
|---|---|---|---|---|
| derby, 4 cars apart | 2.96 → 1.25 | 0.35 | 4.5 | 7.8 → 6.0 |
| derby, 4 cars in contact | 2.98 → 1.25 | 0.37 | 5.8 | 9.2 → 7.4 |
| derby, 8 cars apart | 6.03 → 2.50 | 0.61 | 5.3–6.7 | 12.4 → 9.4 |
| derby, 8 cars in contact | 6.00 → 2.49 | 0.68 | 4.4 | 11.1 → 7.6 |
| mountain, 8 cars (t 1.5 / 9 s) | 5.85 → 2.47 | 0.82 / 0.70 | 10.0 / 4.9 | 17.1 → 13.1 / 11.4 → 8.1 |

One by one with this kernel (the retired check alone): 2.1–2.2 ms (4 cars), 4.2–4.5
(8 cars). Wide and loop are within 0.1 ms of each other; wide is the default.

**The page's real-time loop** (`skin-batch-live.sh`: one run a scene, before and after
alternating in 6 s windows, 4 rounds; real time is 60 steps a second; the drawing gives
way, 30 pictures a second at its floor):

| | World steps/s, before → after | pictures/s |
|---|---|---|
| derby, 4 cars in contact | 47 (78 %) → 49 (82 %) | 30 → 30 |
| derby, 8 cars in contact | 35 (58 %) → 39 (65 %) | 27 → 30 |
| mountain, 8 cars | 36 (59 %) → 36 (60 %) | 28 → 28 |

Cars apart keep real time before and after (§8.38). The page's per-second GPU ms of a
drawn frame is a rolling median of timer readbacks, skipped while the GPU is busy, so it
lags across 6 s windows: the paused table is the clean one. The mountain is physics
bound (§8.38: 80 % of real time with nothing drawn).

**What is left.** The batched skin is ≈0.27 ms a car and grows with the cars, so it is
not the binds' bandwidth (157 MB once is ≈0.5 ms): it is the per-car work of ≈25 entries
a vertex (each a gathered node position and three multiply-adds) over 491k vertices. The
picture (4.4–6.7 ms for eight cars, 10 on the mountain at its start) follows the
triangles, not the pixels (§8.38): levels of detail are the lever, a different picture by
design (the "Car detail: lighter" control already draws about half the vertices).

### 8.42 The floor measured directly: a purpose-built kernel for one sedan (2026-11-19)

`engine/experiments/gpu-floor` (a workspace member nothing depends on): a hand-written
Metal kernel for the lab's generated sedan alone, checked against the CPU reference, to
measure what the car's physics costs once the production kernels' generality is gone. The
car as the reference steps it: 548 nodes, 4,251 beam records (3,581 beams and 670
aero-triangle records), 528 collision triangles, 424 contacters; centred damping, anti-clip
with the swept test, the two-body rule, the residual pass and the work guard, fixed-point
contact sums, dissipative exact-friction contact. Left out, as the brief asked: the World's
coupling, raycast wheels, sleep, debris, surface contact, other cars. The ground is the
reference's own plane and the wall its static box (the same contact law), 34 substeps a
frame. Native M3 Max, through the GPU lock; logs in
`~/Development/rubble-artifacts/gpu-floor/`.

**Three mechanisms.** What a soft body can touch decides how its step is laid out, and
there are three cases that differ in mechanism, not in wording: (1) one body and a fixed
world (the ground, static boxes): nothing leaves its workgroup; (2) a body against another
soft body (a car, a node-beam structure, the same thing): two workgroups on two cores must
step in lockstep and exchange state every substep; (3) a body against the rigid World: the
rigid bodies come in as proxies once a frame, live in the body's workgroup for the frame's
34 substeps and go back with the impulses they took. Each has its own kernel path
(`kernels/floor2.metal` with `pair.metal` or the `RIGID` blocks compiled in) and its own
scenes (`floor`, `pair --scene headon|side|rear|apart`, `rigid --scene bricks|rocks`).

**How it is checked.**
- *The kernel's logic, with no GPU*: `--emu` compiles the Metal source as C++ (a small
  header stands in for the shading language; each lane is a coroutine, a barrier the point
  where every lane has finished its phase, a cross-workgroup barrier the point where every
  workgroup has) and compares every state word with the CPU reference after each frame or
  each substep, lanes and workgroups in either order. It found three bugs before any GPU
  time: a start-up race on a counter, a workgroup memory shared between emulated groups,
  and a real race in the two-workgroup contact pass (a body one barrier ahead of its
  partner overwrote what it had published). `EMU_TRACE=1` prints the kernel's own counts,
  which is how the wasted candidates below were found.
- *On the GPU, strict*: fast math off **and** `#pragma clang fp contract(off)`. With fast
  math off alone Metal still emits `llvm.fmuladd` (9 % of a·b + c differ from the CPU's);
  with the pragma a·b + c, a/b and √a are the CPU's bits over 4 M random inputs
  (`mathprobe`). No XOR barriers are needed for a strict Metal build.
- *Closeness, for builds that are not bit for bit*: the one-frame defect (a second car
  restarted from the reference's state every frame: worst node position and velocity
  error) against the reference's own one-frame spread from starts moved by a micron; and
  the free run's deviation (farthest node, centre of mass, kinetic energy, broken and bent
  beams) against the reference's own spread over three such starts.

**Strict is bit for bit on the GPU, all three mechanisms**, every frame, every state word
(node positions, velocities and forces; live beams' rest lengths, thresholds, strengths and
slopes; broken flags; deform groups; dead triangles; fired groups; contact sums and
position steps; anti-clip memory; for mechanism 3 the rigid bodies' poses, velocities,
counts, impulses, moments and peaks):
- one body: a 5 cm drop to rest, a 3 m fall and landing, a 16 m/s wall crash (120 frames,
  up to 27 self-contacts a step), and a restart from the CPU's state 12 frames into that
  crash;
- two bodies: head-on at 14 m/s each (30 frames, about 2,000 body-body hits a frame, 52
  beams broken), a side impact (20 frames, and 204 substeps one dispatch each), a rear
  impact (20 frames), and two bodies apart;
- a body and rigid bodies: the car at 11 m/s into 24 loose 25 kg boxes (30 frames), and four
  0.5 to 1 t spheres falling on the parked car (30 frames). The coupling is two-way, as the
  reference's: each node inside a shape gets the two-body contact rule against that body's
  mass and inertia at the contact point, the reaction and its moment are summed per shape
  in fixed point, and the bodies integrate them every substep.

**The cost model, measured at 1,024 lanes** (`microbench`, µs a barrier phase of one
threadgroup; 40 threadgroups give the same figures, one a core):

| what | measured | so |
|---|---|---|
| an empty barrier phase | 0.073 | barriers are free |
| 128 dependent fma: 1 / 8 / 32 warps active | 0.24 / 0.42 / 1.51 | a phase is bound by the core's throughput, 0.35 ns a warp-instruction, not by its longest chain (1.3 ns a step with one warp) |
| 4 chains of 32 fma one after another / interleaved | 1.47 / 1.44 | interleaving buys nothing |
| the same work in 16 of 32 warps | half | an idle warp is free |
| threadgroup-memory loads: a scalar / 3 adjacent floats / 6 adjacent | 3.1 / 2.5 / 4.9 ns a warp | about ten instructions an access, the same as a streamed device load; a load is three or four floats, so position and velocity side by side are two loads, not one |
| device loads: 4 streamed / 4 scattered / 16 scattered a lane | 3.9 / 23 / 29 ns a warp-load | 10 to 80 instructions each |
| device atomic add: own word / 4 scattered | 2.6 / 38 ns | |
| live registers a lane, each updated a phase: 16 / 48 / 80 / 112 / 160 | 0.19 / 0.32 / 1.1 / 5.0 / 4.2 ns an update | past about 48 they spill: 3× at 80, 15× at 112 |
| 16 dependent √ or ÷: fast math / strict | 0.7 / 2.7 (√), 4.1 (÷) ns a warp | strict √ and ÷ cost 4 to 6× |
| a barrier across workgroups on different cores (a device counter and a bounded spin) | +0.1 to 0.7 µs, 0 missed for 2 to 40 groups | lockstep across cores is affordable |
| a word one workgroup stores, read by a workgroup on another core in the same dispatch (`xsync --data`, 2,000 rounds) | plain store: wrong on every read; atomic store: right on every read | see below |

This corrects §8.35's reading (8 ns a dependent step was eight warps sharing the core) and
§2.2's cause (it is not latency): **a car's step on one core costs its warps' instructions,
its memory accesses at about ten instructions each (threadgroup or streamed device; 20 to
80 for scattered device), and whatever its lanes keep alive past about 48 registers.** A
core is saturated by about eight busy warps, so a workgroup's phases do not overlap and a
second workgroup on the same core waits.

Three things the compiler and the GPU do that the source does not show:
- with fast math off it still fuses a·b + c (above);
- it does not unroll a constant loop with a large body, and an array indexed by the loop
  variable then stays in stack memory: the kernel's four "register" slots a lane were
  loads and stores until the loop was written out once a slot (both seen in
  `xcrun metal -S -emit-llvm`; they cost the first kernels most of their time);
- **what two workgroups share within a dispatch must be atomic on both sides.** A plain
  store by one workgroup was never seen by a workgroup on another core (4,095,999 of
  4,096,000 reads stale with two groups, the same with 3, 8, 39 and 40), not even through
  an atomic load, and a plain load did not see another group's atomic add to a word its own
  group had stored. With atomic stores every read was right. The first body-against-body
  kernel was bit for bit in the emulator and wrong on the GPU for exactly this (its bodies
  published their nodes with plain stores); with atomic stores and loads it is bit for bit.

**The kernels.** `kernels/floor.metal` (first layout, kept for the ledger) and
`kernels/floor2.metal` (second layout, from the cost model), one workgroup of 1,024 lanes a
body, all 34 substeps a dispatch, every count a compile-time constant:
- a beam's hot words (its ends, k, c, rest length, last extension rate) in registers, four
  slots a lane; everything else of a beam read only on the step it yields, breaks, or is
  one of the car's 71 special elements; a lane's own node in registers too;
- a beam hands its ends one scalar (force over length) through threadgroup memory; a node
  gathers its beams in the reference's order from a streamed list as (x_self − x_other)·g,
  which is the reference's push bit for bit at both ends (IEEE subtraction and
  multiplication are sign-symmetric), and a broken beam's scalar is zero (adding ±0 changes
  no bit of a sum that began at +0 or at the weight);
- self-collision: a lane a triangle over its own row of an exact pair list (every pair the
  reference's box test could pass, rebuilt when a node has moved more than the skin
  allows), each pair filtered by the rule's own cheap tests (crossed the plane over the
  step; or holds a side for this triangle; or inside the band **and over the triangle**);
  pairs that pass, and nodes in ground or box contact, go on compact lists and get their
  full rule a lane an item (a warp pays for its slowest lane). The last test matters: with
  the band alone the filter passed 722 pairs a substep at rest (points of the same panel,
  beside the triangle), all of which the full rule then dropped; with it 0 to 2 at rest
  and 5 to 13 in the wall crash (was about 800), the same bits;
- contact sums, side memory, break groups and triangle breaks live in device memory and are
  touched only on steps flagged as having them; a hot-state image starts a dispatch with six
  streamed loads a lane.

A substep without contacts is seven barrier phases (0.5 µs of barriers).

**One body and a fixed world (mechanism 1), measured.** ms a frame of 34 substeps, one car,
medians (`batch-nine`, `batch-thirteen`; a state's figure repeats to 1 %, a change of
compile to about 5 %); production's figures are the scenario matrix's rows on the same machine
(`scenario-matrix.md`, not re-run here):

| | at rest | wall crash, 16 m/s | 3 m fall | bit for bit |
|---|---|---|---|---|
| production (matrix rows drive / tumble / wall) | 3.2 | 4.1 | | (its own suites) |
| first layout, strict | 3.00 | 3.68 | 3.17 | yes |
| second layout, strict | 1.25 | 2.2 | 1.07 (1.44 at the landing) | yes |
| second layout, fast math + native functions | 1.04–1.10 | 1.83 mid-crush, 1.66 after | | no: see closeness |
| the reference on one CPU core | ≈10.5 | ≈10.4 | ≈10.5 | |

A dispatch with every phase left out is 0.10 ms. At rest the phases are (ms a frame, fast,
by the profiler below): node pass 0.08, centred scalars 0.08, centred gather 0.17, beams
0.24, force gather 0.19, pair rows 0.27, the candidates' rule 0.09 and the hit passes 0.10
(a settling car still has a contact or two), the list's rebuilds 0.03. In the wall crash,
mid-crush: beams 0.31, pair rows 0.46, hit passes 0.39, rebuilds 0.14, candidates 0.11,
the rest as at rest. The crumpled car's pair rows are no longer than at rest (2,899 listed
pairs in 110 warp-iterations a substep against 2,827 in 99): what a crash adds is the
contacts' own phases and the list's rebuilds, which is what the next paragraph but one is
about.

**It did not reach §1.3's 0.3–0.5 ms, and why.** That estimate charged memory at cache
bandwidth and arithmetic at the core's FLOPS. The core charges otherwise: a substep at rest
is about 3,400 warp memory accesses and about 30,000 warp instructions, which the table
above prices at about 20 µs; the measured substep is 35 to 38 µs. The gathers cost about
29 ns a warp-entry and the pair rows about 90 ns a warp-pair, two to five times their
loads and instructions, the same unexplained factor as §8.35's. Changes that cut
instructions (unrolled gathers, compact lists, a two-level list build, static slot
classes) did not move the total. Three layouts aimed at memory were then built and
measured, and none paid:

| layout | what it removes | at rest, ms a frame |
|---|---|---|
| a lane's own node in registers; the held byte read only after a contact | about 7 % of a substep's accesses | in the figures above |
| position and velocity side by side in threadgroup memory (`--tgpv`) | an end of a beam in one load: but a load is 3 to 4 floats | 1.29 against 1.29 |
| each node's lane owns three of its beams (`--own`, every node can: 1,644 of 3,164 plain beams), adding their force from registers | 3 of a node's 13 gathered entries twice a substep; half the loads of those beams | 1.14 against 1.11 (fast): the gathers fall by 0.19 ms and the beam phases take it back |

So the floor found for this car on this GPU is **about 1.1 ms a frame fast, 1.25 strict**:
a third of production, three to four times §1.3's estimate. What is left is not waste the
counts show; it is the price of a warp-iteration here, which the microbenchmark's
primitives under-predict. Two guesses at the cause were tested and are not it
(`batch-rows`, strict, at rest): the slot words' registers (twelve fewer live a lane,
`--regmode 2`: 1.39 against 1.32, the rows' phase unchanged at 0.34) and the row's vector
indexed by a variable (four points a load by name: 1.29 against 1.32, kept as the
default). It stays open.

**Crash states: a profiler that can be trusted, and what it found.** Leaving one phase out
of a whole frame (how the phase figures above were first taken) mis-measures any phase the
later substeps depend on: they then run from another state (the centred scalars came out at
−6 ms, the two-body pass anywhere from 1.2 to 2.0). `--profile F` runs a kernel of one
substep a dispatch: each of frame F + 1's 34 substeps from its own saved state, cut short
after one more phase group at a time; a group's cost is the difference, and nothing before
it changes. The groups sum to the measured frame within a few percent (4.46 against 4.51 ms).
It still is not a full-frame A/B: two changes its sums favoured made the frame slower
(below), so every change here was kept or dropped on the whole frame, side by side in one
batch.

The head-on frame (two cars, 14 m/s each, mid-crush, fast), ms a frame by group, when the
profiler first ran and now:

| group | then | now |
|---|---|---|
| the two-body owners' turns | 1.53 | 0.87 |
| the bodies' own hit passes | 0.46 | 0.24–0.35 |
| two-body: boxes, publication, the partner's nodes, the triangles' cull and clusters | 0.56 | 0.43 |
| two-body: cluster points' box test, candidates' rule | 0.20, 0.21 | 0.20, 0.18 |
| the pair list's rebuilds | 0.38 (0.6 with one list) | 0.29–0.40 |
| beams, gathers, rows, node pass, self candidates | 1.1 | 1.1 |
| **the frame** | **4.5** (5.7 before the profiler) | **3.5** |

What it showed is one thing: **a phase in which a few lanes each run a long chain costs
that chain, however few the lanes.** A chain runs at about 1.3 ns an instruction and about
37 ns a device atomic one after another (`xsync --spin`: forty scattered atomic adds a
lane in one warp, 1.5 µs), against 0.35 ns a warp-instruction when the core is full.
"A lane an item" lists (above) take rare work out of the wide phases, and each such phase
then costs its longest chain: 1 to 3 µs, twenty times a substep. The hits were the worst
case: each hit's lane walked some 170 scattered device words over its passes (the nodes'
forces, masses, hit counts, staged sums, contact sums). What was changed, each bit for bit
with the reference on the GPU, with its effect on the head-on frame:
- *only the nodes a contact wrote read their contact words*: on any step after a contact
  every node cleared and read its contact sums, position step, side memory and staging, up
  to 22 device atomics a node; now four bits a node say what was written for it
  (5.66 → 5.22 ms);
- *what another workgroup stored atomically read with one plain vector load* in place of
  three atomic loads (5.22 → 4.87);
- *two pair lists*: a wide one (skin 0.30 m) built from the clusters, the working one
  (skin 0.05 m) filtered from it: the same rows in the same order. In the head-on the
  working list is rebuilt on 15 of 34 substeps (a node may move 15 mm against the reference
  node before the list must be rebuilt, and the crush zone does that in two or three
  steps); from the clusters that cost a warp every cluster's nodes for its one lane that
  needs them (4.87 → 4.43; the wall crash's wide builds 135 → 41 in 20 frames);
- *the hit passes from threadgroup memory*: each hit node gets a slot in the beam scalars'
  array (free between two beam passes), named by an atomic minimum over the hits that touch
  it and filled by the node's own lane from its registers; the passes read and sum there
  (4.51 → 3.60; the turns 1.53 → 0.85);
- *a body's clusters' boxes published by the body* instead of rebuilt by its partner from
  28 scattered loads a lane (3.60 → 3.50).
What was tried and dropped, measured side by side (`batch-eleven`): the hits' sums carried
on to the nodes' lanes' registers, and a row's first candidate run by the row's own lane
(3.60 → 3.84, and the car at rest 1.09 → 1.17: six more live values a lane and a long
branch inside the widest phase cost more than the phases they removed); the two-body rule
run by the lane that finds a point in a triangle's box (3.84 → 4.03: a warp waits for its
slowest lane, which is what the lists were for); velocity and force published only for
the nodes a partner can reach (no change: a streamed atomic store is cheap, it is the
chains that are not). A workgroup waiting at the cross-group barrier does not slow the one
still working (`xsync --spin`).

**Body against body (mechanism 2).** Two workgroups a pair, one a body. Metal has no
barrier across workgroups: each group's lane 0 adds one to a device counter after its own
barrier and waits (a bounded number of loads) until every group has; a miss is a status
bit and was never seen. After a step each body reduces its box and top speed on one warp
and publishes them; the bodies meet; if the boxes are apart (the reference's own test)
that is all, one meeting a substep. If they meet, each publishes its nodes (atomic
stores), they meet again, and each body does the half of the reference's contact pass
that its own triangles own, both at once: the partner's nodes copied into the beam
scalars' array (free between two beam passes); its triangles against the partner's box
and then against the partner's clusters of contact points (their boxes published by the
partner); each (triangle, cluster) against the cluster's points by the reference's padded
box test; each candidate through the contact rule (frame, held side, swept crossing, side
vote). Every stage's survivors
are a compact list, a lane an item. The culls before the reference's test only drop what
that test drops (the same rounded terms, a pad no smaller). Then the owners' turns, owner
0's and then owner 1's as the reference orders them (first pass from the forces at the
pass's start, residual, work guard, the sums), the other body waiting at the barrier;
integer sums, so a turn is order-free inside. What a body publishes lives in two slots by
the substep's parity, because a body can run one barrier ahead of its partner.

The first design had a third workgroup do the whole pass with both bodies copied into its
memory: it cost about 110 µs a substep in contact while the two bodies' cores idled, and
13 pairs filled the GPU's 40 cores. On the bodies' own cores (first 5.3 to 6.4 ms for the
head-on; the paragraph above is how it came down), ms a frame for the pair (two cars), bit
for bit strict in every scene (`batch-twelve`, `batch-thirteen`, `batch-final-b`):

| scene | strict | fast + native | production (matrix) |
|---|---|---|---|
| two bodies apart (boxes apart) | 1.43 | 1.33 | |
| head-on, 14 m/s each: mid-crush / the 30 frames' median | 4.2 (4.0 after the second round below) | 3.5 / 3.4 (3.2 / 3.1) | 15.4 |
| side impact: at the impact / after it | / 2.1 | 3.6 / 1.9 | 9.3 |
| rear impact, after it | 2.15 | | |

Two cars crushing each other cost about twice one car crushing against a wall (3.5 against
1.83: each body is in its own crash, with its own self-contacts, breaks and list
rebuilds, and the pass on top: about 600 candidates and 10 to 20 hits a body a substep).

**Body against rigid bodies (mechanism 3).** The reference's proxy set in the body's
workgroup: each substep a warp reduces the body's box, a lane a shape places its shape
(glam's NEON quaternion arithmetic written out in its operation order, so the poses are
the reference's bits) and marks it active if its bounds meet the box; a node tests the
active shapes (bounds first), counts into the body it is inside (the rule's mass split)
and, on the contact nodes' compact list, takes the two-body rule against that body at the
contact point; reactions and moments are summed per shape in 1/32 N fixed point; a lane a
body integrates them. Boxes and spheres on dynamic bodies are ported; static bodies, hulls
and capsules are not. ms a frame, one car, bit for bit strict in both scenes
(`batch-rigid`):

| scene | strict | fast + native | production (matrix) |
|---|---|---|---|
| the car at 11 m/s into 24 boxes, mid-impact | 2.08 | 1.83 | 3.4 (60 bricks) |
| four rocks onto the parked car, under the rocks | 2.06 | 1.66 | 3.8 |

(First measured at 3.0 and 3.1 for the boxes, 2.2 and 2.0 for the rocks, and read as "a node
tests every active shape". It was not that: 1.0 ms of the boxes' frame was the 24 bodies'
integration, each body's lane scanning the shapes for its own. See "a loop in which each
lane's turn is a different iteration" under surface contact below.)

**The scaling staircase, each mechanism's scenes.** One workgroup is one core's work and
the GPU has 40, so N workgroups cost ⌈N / 40⌉ times one (fast + native, ms a frame, all
copies in one dispatch; a pair is two workgroups, so its tread is 20 pairs):

| workgroups | 1 | 20 | 40 | 41 | 80 | 81 | 120 | 200 |
|---|---|---|---|---|---|---|---|---|
| one body at rest | 1.10 | 1.12 | 1.21 | 2.31 | 2.57 | 3.63 | 3.83 | 6.23 |
| one body, wall, mid-crash | 1.83 | 1.90 | 2.20 | 4.02 | 4.51 | 6.33 | 6.86 | 11.3 |
| a body into 24 boxes, mid-impact | 1.83 | | 2.20 | 4.02 | 4.62 | | | 13.1 |
| a body under four rocks | 1.66 | | 1.96 | 3.60 | 4.17 | | | 11.8 |

| pairs (two workgroups each) | 1 | 8 | 20 | 21 | 40 | 41 | 60 |
|---|---|---|---|---|---|---|---|
| head-on, mid-crush | 3.49 | 3.64 | 4.53 | 7.87 | 9.26 | 12.7 | 13.8 |
| side impact, at the impact | 3.58 | | 4.49 | 7.96 | 9.43 | | |
| bodies apart | 1.33 | | 1.47 | 2.74 | 2.98 | | |

(`batch-final-a`, `-b`.) A tread rises by 10 to 30 % from its first workgroup to its
fortieth (the cores share memory bandwidth and the clock), then the next workgroup starts
a new tread. 200 cars at rest are 6.2 ms and 200 in a wall crash 11.3; 120 cars in 60
head-on crashes are 13.8 ms. Production's 32 cars apart were 11.75 ms (§1.5).

**Closeness of the fast builds** (`batch-final-a`, `-b`). One body at rest: the one-frame
defect is inside the reference's own from a micron (median 0.026 mm against 0.074, p90
0.10 against 0.24, worst 0.49 against 0.72); the free run's positions stay inside the
reference's spread (5.5 mm against 97 at the frame it is flagged), and the check's
kinetic-energy term is what flags it (28 % against 4 % of a kinetic energy that is nearly
zero in a settled car: a weakness of that metric at rest, not of the build). Wall crash:
the defect is the size of the reference's own over the run (median 0.06 mm against 0.22,
p90 2.7 against 4.9, worst 8.1 against 9.7) though single frames differ by more than ten
times that frame's reference value (a contact or a yield lands a frame apart); the free
run's positions and centre of mass stay inside the reference's spread (21 mm against 74,
8.5 against 11.5), its kinetic energy does not at one frame (2.7 % against 0.8 %), so by
the check's own rule the run is outside three times the spread. Two bodies head-on: the
farthest node from the reference is 410 to 710 mm over 30 frames against 470 to 900 mm
for the reference's own from a micron (a head-on crash is chaotic; the same broken-beam
counts); side impact 0.01 mm against 7.0 at the impact and 12 to 36 mm against 27 to 92
after it. Fast math with native functions is worth 10 to 20 %.

**The ledger (first layout, wall crash, fast math, ms a frame; `batch-v1`):** counts as
uniforms instead of constants +6 % (3.40 → 3.60); a dispatch a substep instead of a frame
+72 % (5.84: 34 loads of the state); unordered force sums through device atomics +18 %
(slower, not faster: device atomics cost more than the ordered gather); native functions
in place of the specified sequences 0 % there (memory-bound) and 5 % on the second layout;
strict against fast +8 %. Second layout: the filter's over-the-triangle test −9 % at rest
and −5 % after the crash; position and velocity side by side 0 %; the owning layout 0 %.

**What to take back to production, ranked.** Gains are this kernel's own, on the scene
named, each measured with only that change between the two figures (ms a frame, one sedan a
body); "bit for bit" means the strict build with the technique in it matched the CPU
reference's every state word on the GPU. Production's gain will differ: its kernels are
Slang through wgpu, one dispatch a substep, and some of this it already has in another form.

| # | technique | measured here | bit for bit | to port |
|---|---|---|---|---|
| 1 | All 34 substeps in one dispatch; the body's state in registers and workgroup memory; a hot-state image between dispatches | wall crash 5.84 → 3.40 (−42 %; a dispatch a substep costs +72 %) | yes | the largest single item, and structural: the World's per-substep host coupling has to move to once a frame |
| 2 | A beam's hot words in registers (four slots a lane), one scalar an end through workgroup memory, each node gathering its beams in the reference's order; slot loops written out by hand | rest 3.00 → 1.32, wall 3.68 → 2.60 (−56 %, −29 %) | yes (the gather is the reference's sum, both ends) | needs every count a compile-time constant (+6 % as uniforms) and no array indexed by a loop variable |
| 3 | Vehicle contact on the two bodies' own workgroups, in lockstep across cores (a device counter and a bounded spin), each body searching its own triangles, the owners' turns one after the other | head-on 3.5 fast / 4.2 strict against production's 15.4 (with 4 to 9 below); the change from a third workgroup doing the pass alone, 6.4 → 5.5 | yes | every word two workgroups share stored atomically (a plain store is not seen on another core: a silent wrong answer); two bodies a group so far |
| 4 | Hit passes from workgroup memory: a slot a hit node (force, mass, hit count, staged sums), named by an atomic minimum, filled by the node's own lane | head-on 4.51 → 3.60 (−20 %); the owners' turns 1.53 → 0.85 | yes | the slots live in the beam scalars' array between two beam passes; at most 80 self hits and 63 a turn a substep (a status bit beyond) |
| 5 | Two pair lists: a wide one from the clusters rarely, the working one filtered from it | head-on 4.88 → 4.43 (−9 %); wall crash's run 2.46 → 2.30 (−7 %) | yes (the same rows in the same order) | applies wherever a Verlet list is rebuilt often in a crash |
| 6 | Only the nodes a contact wrote read and clear their contact words (four bits a node, by kind) | head-on 5.66 → 5.22 (−8 %); wall 2.52 → 2.24 strict (−11 %) | yes | check first whether production's node pass reads them for every node |
| 7 | The pair filter also applies the rule's over-the-triangle test and checks a held side against this triangle | rest 1.41 → 1.29 (−9 %), wall −5 %; candidates 722 → 0–2 a substep at rest | yes (the rule's own tests) | only where a cheap filter feeds a separate full-rule pass; production runs the rule per pair |
| 8 | What another workgroup stored atomically read with one plain vector load | head-on 5.22 → 4.87 (−7 %) | yes (measured; not guaranteed by the language) | with 3 only |
| 9 | A body publishes its clusters' boxes for its partner's cull | head-on 3.60 → 3.50 (−3 %) | yes | with 3 only |
| 10 | A row loop with no vector indexed by a variable (four points a load, by name) | rest 1.32 → 1.29 (−2 %) | yes | trivial |
| 11 | Fast math with native functions | 10 to 20 % (rest 1.25 → 1.09, head-on 4.2 → 3.5) | **no**: within the closeness report above | an option, as now |
| 12 | A lane's item given to it (a list, or `ctz` over a mask of hits) instead of a scan with a skip; appends after the loop, not inside it | 24 proxy bodies' integration 1.0 → 0.09 ms a frame (node contact on the bricks 2.69 → 1.83); surface search 0.57 → 0.17 | yes | per loop: the production kernels' own scans (a body's shapes, candidates appended inside a row or cluster loop) are to be read for it |
| 13 | In a pass whose owners take turns, the part of every turn that reads only the pass's start (two-body: slots, hit counts, first pass) done by all owners at once, before they meet | head-on 3.40 → 3.32; those stages 0.46 → 0.32 ms | yes | with 3 only; a turn's values cross the meeting in the hit's record, not in registers |
| 14 | A reduction done by the lanes that hold the values (atomic min and max into workgroup memory) instead of one warp walking them | head-on 3.31 → 3.23; two bodies apart 1.47 → 1.43; surface contact's box 0.02 | yes | the body's box, its clusters' boxes, surface contact's box |
| 15 | What a phase's loop reads that does not change (the lane's roles and entry counts, the bases) taken again at the phase's start, not kept from the kernel's | 2 to 7 %: parked with surface contact 1.51 → 1.41, bricks 3.47 → 3.36, at rest 1.09 → 1.06 | yes | grows with the kernel's size, so production's general kernels have the most to gain; needs a load the compiler cannot fold |
| 16 | The kernel chosen a body a frame by what is in reach: a contact-free kernel with a one-compare guard, the whole kernel taking the frame up at the substep the guard ended it | a frame with nothing in reach 1.41 → 1.07 (surface contact), 1.23 → 1.07 (node contact); the post's run 1.95 → 1.78; frames in contact unchanged | yes | the World's proxy gather gives the plan; the guard, not the plan, carries the safety; pairs not covered |

The table read against production's code as it stands (what production does at each of
these points, what applies to its kernels as they are, what needs a new kernel, what does
not carry to Chrome or CUDA, and an order of work): `gpu-floor-vs-production.md`.

Two that are not kernel changes and paid for all of the above: the strict Metal build is
fast math off plus `#pragma clang fp contract(off)` (no XOR barriers; that is what made
"bit for bit" checkable at all), and the two instruments, an emulator that runs the kernel
source on the CPU against the reference (every logic bug here, a cross-core race among
them, found without the GPU queue) and the profiler above with its rule (a change is kept
only on a whole-frame comparison in one batch).

Measured and not worth taking: position and velocity side by side (0 %); each node's lane
owning three of its beams (0 %, and not bit for bit); unordered force sums through device
atomics (+18 % slower, not bit for bit); the hits' sums held in the nodes' lanes'
registers and a row's first candidate run inline (+7 %); the two-body rule run where a
point passes its box (+5 %); a phase fewer in the hit passes (+1 to +5 %); velocity and
force published only for reachable nodes (0 %, kept: it is exact and cheap).

**Where the head-on's 3.5 ms still is, and what would be tried next.** By the profiler: the
owners' turns 0.87 (ten short phases a substep for some thirty hits: the rule's passes each
need the one before, summed over every hit), the two-body search 0.8, the bodies' own hit
passes 0.3, the list's rebuilds 0.3 to 0.4, and 1.1 for the two bodies' beams, gathers and
rows, which is the car at rest. Next: the wide list built a lane an item, as the two-body
search is (its clusters still cost a warp every cluster for one lane); the list's test of
how far nodes have moved taken against a reference near the pair, not one node of the
car (the crush zone moves 7 mm a step against the far end and is rebuilt every two or
three steps for it). Each is tens of percent of its group, not of the frame: the frame
is now spread over a dozen groups, none above a quarter.

**The two-body pass, second round (2026-11-19, after surface contact's).** The expectation
was that the search's two loops with an append inside (triangle against the partner's 20
clusters, a cluster's 32 points) held the head-on's 0.8 ms of search, as the same shape had
held 1 ms in mechanism 3. They did not: they held 0.29 ms, and listing from a mask took
0.13 of it. The profiler's per-substep dispatches had overstated the pass (two workgroups
started apart wait at every meeting: 2.1 ms by its sums against 1.28 on whole frames), so
the pass is now taken apart on whole frames: the frame from one saved state with the pass
cut short after one more stage (`pair --gpu` prints it; `--def NAME=V` switches a kernel
option from one binary). The head-on frame's pass, ms a frame (fast, frame 10; a substep
is this over 34), before this round and now:

| stage | before | now |
|---|---|---|
| the second meeting (the first is in every frame) | 0.06 | 0.07–0.14 |
| nodes published (548 lanes, 3 to 9 atomic stores each) | 0.14 | 0.12 |
| cluster boxes published | 0.09 | 0.035 |
| the partner's nodes copied; triangles' cull | 0.045; 0.02 | 0.04; 0.03 |
| triangles × clusters (285 items a body a substep) | 0.09 | 0.04 |
| cluster points' box test (600 candidates) | 0.20 | 0.13 |
| candidates' rule, hits recorded (10 to 20 hits) | 0.19 | 0.16–0.18 |
| hit counts, three meetings | 0.09 | 0.10 |
| turns: slots filled, hits counted | 0.23 | 0.18 |
| turns: first pass, its shifts | 0.23 | 0.135 |
| turns: residual pass | 0.08 | 0.09 |
| **the pass** (the stages; the guard's own cost does not show, the frame is slower without it) | **1.28** | **1.09** |

Kept, each bit for bit with the reference (the emulator: head-on and side in both lane
orders, rear, a substep at a time; the GPU strict: head-on 30 frames, side 20, rear 20,
apart 10, head-on a substep a dispatch, four pairs), with the head-on frame:
- *The clusters met and the points passed as bits, then listed* (3.50 → 3.40). Small,
  because every lane's turn in those loops already is the same iteration; only the store
  ran once an iteration.
- *Every owner's slots, hit counts and first pass at once, before the bodies meet*
  (3.40 → 3.32; those stages 0.46 → 0.32). The owners' turns are one after the other, as
  the reference's, but only the residual pass and the work guard read what the other
  owner's turn left (the contact sums). The first pass is from the forces at the pass's
  start, which the candidates' stage records, and its shifts are atomic adds that
  commute. So each body does that part on its own core while the other does the same;
  then owner 0's residual and guard, a meeting, owner 1's contact sums read again (each
  node's by its own lane), its residual and guard. What a hit's lane needs after the
  meeting (slots, first-pass force, the point's velocity) goes through the hit's record:
  kept in registers across the meetings, those stages were 0.23 but the frame *without
  the pass* was 0.08 ms slower and the meetings 0.07 dearer (3.36 in all).
- *The body's box and its clusters' boxes reduced by the nodes' own lanes* (atomic min
  and max into workgroup memory, a lane a cluster then publishing): 3.31 → 3.23, and two
  bodies apart 1.47 → 1.43. One warp walking 17 nodes a lane, or a cluster's lane its 28
  points, was that phase's chain at 1.3 ns a step while the core's other warps idled.

Measured and not kept: the candidates' rule in two stages, a hit-to-be then its own lane
(3.49 → 3.57: the hits' forty device words are not what a warp of candidates pays for);
what a body publishes laid out by lane, so its stores are adjacent (3.31 → 3.37: the
stage did not move, 0.134 both ways, and the readers' table loads cost; an atomic store to
device memory is about 20 ns a warp wherever the words are); the reach test's shared
words read once a lane instead of eighteen times (its stage 0.134 → 0.118, the frame
3.227 → 3.243); every slot's sums re-read by its hit's lane (+0.05 ms on its stage: one
warp's chain of scattered loads against 17 warps' three).

The pair, ms a frame, the round's start against its end in one batch (`batch-pairx12`,
`-x14`; today's figures at 20 pairs and more are above the table's further up, for both
kernels and for the older pinned one alike):

| | fast + native | strict |
|---|---|---|
| head-on, mid-crush (frame 10) | 3.50 → **3.22** | 4.37 → **3.99** |
| head-on, the 30 frames' median | 3.36 → 3.09 | 3.88 now |
| side impact, at the impact | 3.58 → 3.28 | 4.33 → 4.00 |
| two bodies apart | 1.47 → 1.43 | 1.70 → 1.67 |
| head-on, 8 / 20 / 21 / 40 / 60 pairs | 3.66 → 3.39 / 4.64 → 4.55 / 8.12 → 7.82 / 9.38 → 9.37 / 14.3 → 14.0 | |

Eight percent for a pair, up to eight pairs; two to four percent at 20 pairs and none at
40, where every core is busy. What this round cut was waiting, not work: a body idle
through the other's turn, a core idle while one warp walked a chain. When all 40 cores
are busy the frame is bound by what they share, and only less work would show.

Two things about the model. *Values alive across a barrier and not touched cost nothing*
(`microbench only=live`: 1 to 128 of them a lane, 0.088 µs a phase throughout), so the
0.2 ms the registers cost above is not the barriers'. *The code's shape moves the frame
with no change in the work done*: across this round's variants the frame without the pass
(none of the pass's stages run) was 2.17 to 2.25 ms, and the same turns compiled in a
restructured loop 3.40 or 3.60. It is the same thing as surface contact's 0.14 ms for its
code's presence, and it is as large as most of the changes tried here; the paragraph
after the next finds its cause in part. A change of a few hundredths of a millisecond
cannot be judged on this kernel while it lasts.

What is left in the pass, by the table: five meetings a substep in contact (about 0.3 ms;
the reference's order fixes three of them: owner 0's guard after owner 1's candidates
have read the start's sums, owner 1's residual after owner 0's guard); the nodes'
publication (0.12: every node's position goes to the partner, for the side vote's
neighbours; publishing only what the partner's box can reach, with those neighbours,
would cut the stores, not done); the turns' 0.4, short chains on one warp of hits.

**What code that does not run costs, and why (2026-11-19).** Two paragraphs here left a
cost with no cause: a parked car pays 0.14 to 0.2 ms for surface contact's code being in
its kernel, and the two-body frame moved by 0.05 to 0.2 ms with the code's shape. The
cause, as far as it was found: **the compiler keeps a value in one place for its whole
life.** A value read at the kernel's start and used in a phase's loop (a lane's entry
counts, its node, its triangle) is alive through every phase of every substep. While the
kernel is small it has a register; as code is added, more such values are kept in memory
instead, and a loop that uses one then reads it from memory every turn. Nothing is
slower where the new code is; the old loops are.

How it was found. First what it is not, each measured:

| suspected | measure | result |
|---|---|---|
| values alive across a barrier, not touched | `microbench only=live`, 1 to 128 a lane | 0.088 µs a phase throughout |
| values alive through the whole kernel, never used | `LIVE_DUMMY` 0, 4, 8, 16, 32, parked car | 1.23 throughout (1.44 with surface contact) |
| the stack frame's size | `microbench only=stack:`, a dead array of 64 to 2,048 floats | none |
| the kernel's length; a barrier that never runs | 2,000 dead instructions; a dead barrier | none (1.51 µs for 1.51) |
| the pass's addresses computed before the substeps' loop | `COLD_OPAQUE`: its bases behind a load the compiler cannot see through | 1.512 and 1.515 |
| helpers not inlined (the compiler's output keeps 77 functions) | `GPU_FLOOR_INLINE=1`: all marked always-inline | node contact 1.26 → 1.22, surface contact 1.52 → 1.54 |

Then what it is. In the microbenchmark a phase that updates 48 values (0.57 µs alone) ran
anywhere from 0.58 to 1.56 µs beside code that never runs, with no order in it (16
threadgroup loads in the dead code 1.03, 48 of them 0.69; fast and strict builds
differently), while a phase of 128 operations on one value did not move. So it is the
running code with many values that is touched, not anything the dead code does. In the
kernel, the parked car with the surface pass's stages left out of the build one after
another (`SURF_CUT`, `SURF_SUB`): 1.26 with node contact alone, 1.33 with the pass's code
left out, 1.38 with its search and narrow tests, 1.50 with its staged stages, 1.51 with
all of it; any one staged stage alone adds 0.03 to 0.09 and all three 0.07. No one
construct, and not additive. By group (`--profile`), with the pass in the kernel and not
run: the two gathers slower (0.17 → 0.22 and 0.19 → 0.29 ms a frame), the beams faster
(0.22 → 0.17). The gathers' loops are bounded by the lane's entry counts, read at the
kernel's start. Reading them again at the gather's start (one load of the lane's table
row, behind a load the compiler cannot see through, so that the copy is the phase's own)
made every build faster, the larger the more. That is the split the compiler does not
make: one value for the long life, another for the loop.

Applied (`LANE_LOCAL`, on by default): the node pass, both gathers, the list check and
the pair rows take their lane's roles, entry counts and bases at the phase's start; and
surface contact's box is reduced by the nodes' own lanes, as the two-body pass's (0.02).
Bit for bit in every scene of every mechanism (the emulator's three suites; strict on the
GPU). ms a frame, before → after, one batch (`batch-shape13`):

| scene | fast + native | strict |
|---|---|---|
| one body at rest | 1.09 → 1.06 | 1.26 → 1.22 |
| one body, wall, mid-crash | 1.83 → 1.74 | 2.26 → 2.22 |
| parked beside the post, node contact | 1.26 → 1.23 | 1.46 → 1.41 |
| parked beside the post, surface contact | 1.51 → **1.41** | 1.68 → 1.58 |
| the post, surface contact | 3.54 → 3.40 | 4.35 → 4.24 |
| 24 bricks, surface contact | 3.47 → 3.36 | 4.22 → 4.10 |
| four rocks, surface contact | 2.52 → 2.40 | 3.04 → 2.93 |
| 24 bricks / four rocks, node contact | 1.83 → 1.80 / 1.66 → 1.62 | 2.08 → 2.00 / 2.06 → 1.97 |
| two bodies apart / head-on / side | 1.43 → 1.41 / 3.22 → 3.14 / 3.27 → 3.21 | no change |

Two to seven percent, most where the kernel is largest. What remains: the parked car
with surface contact is 1.41 against 1.26 with the pass's code left out of the same
build, of which the pass's first phase runs for some 0.05; about 0.1 is still the code's
presence. The values it would take are the lane's state (position, force, the beams'
slots), which cannot be read again from a table; a bit-for-bit copy of the node's
position made for the gathers alone was erratic (`LANE_LOCAL=3`: 1.437 → 1.429 with
surface contact, 1.270 → 1.306 with its code left out) and is off. The compiler's own
output cannot be read here (the intermediate code can, `GPU_FLOOR_DUMP` and `xcrun metal
-S -emit-llvm`; it is before the inlining and the register assignment that matter), so
what is left is found one experiment at a time.

What follows for a kernel that holds every mechanism: its always-run loops should read
their bounds and bases at the phase's start, not from the kernel's; and a body with
nothing in reach in a frame is cheapest in a kernel that has no contact code at all (1.06
at rest against 1.23 with node contact compiled in and 1.41 with surface contact). The
World builds each body's proxy set every frame; an empty set could pick that kernel. Built
next.

**The kernel by what is in reach (2026-11-19).** Three pipelines from the one source
(`floor2.metal` `REACH`), a body's choice made every frame:

- *the one kernel*, as it was;
- *the contact-free kernel* (`REACH 1`): no code for the proxies' shapes and none for
  surface contact. It keeps the ground, the world's boxes by node, self-collision, and
  the window's start; it has a guard; and at its end it does the proxies' free flight for
  the substeps it did (the reference's `end_step` with no reaction, a substep after
  another, the same arithmetic);
- *the kernel that takes a frame up* (`REACH 2`): the whole kernel starting at the substep
  a word gives it, and returning at once when none is left.

*The plan* (`floor.rs` `reach_plan`, the host, from what it knows at the frame's start;
in the World, where it gathers the body's proxies). Everything the body could touch in
the frame goes in one box: each proxy shape's extent as posed, widened by what its
turning can move in the frame, swept along its free flight; with surface contact the
world's fixed boxes too. The plane is the box's face the body is farthest outside of,
moved out by the contact's reach (the 2.5 cm band and a millimetre). The margin is four
times what the fastest node moves a substep, 2 cm at least. The contact-free kernel is
chosen when the body is clear of the plane by the margin and by 1.25 times what it would
travel in the frame.

*The guard* makes that choice safe whatever the host gave it: one compare a node at each
substep's end (n·x − offset against the margin). A node within the margin ends the
dispatch with that substep; the body's count of substeps done is written, and the whole
kernel's dispatch, next in the same command buffer, takes the frame up there. Its
workgroups are read from three words (an indirect dispatch): none unless a body was given
the whole kernel or a guard asked. A plan that does not hold at the start ends the first
kernel at substep 0. Nothing is done twice and nothing is undone.

*Why the frame is the one kernel's.* At the start of a substep the contact-free kernel
runs, every node is farther from the plane than the margin, so none is inside a shape's
bounds and node contact would have found nothing. Surface contact's first test is the
body's box, taken where the nodes are after the node pass, against the shapes' boxes
padded by the band: the contact-free kernel checks there that every node is still on the
clear side, so no shape would have been near and the pass would have done nothing. If
that check fails (a node crossed the whole margin in one substep, four times what the
fastest node was moving) the kernel's status says so (0x1000) and the frame does not
count; no run here set it. The proxies have no reaction, so their substeps are free
flight, and what the frame's later substeps need of them is where they are, which the
whole kernel reads at its start.

*Checked.* Emulator (`reach-emu.sh`, a minute): 24 runs bit for bit with the reference in
every state word of the body and the proxies; with `--reach-eager` frames change kernel
part way, at substeps 1, 6, 12, 13, 15, 16, 17, 19, 22, 25, 26, 31 and 33; both lane
orders; proxies at rest, falling, and spinning after being struck; a fixed pole with
surface contact and a wall by node contact (the one-body driver). GPU strict, bit for bit
every frame: the post over 60 frames (approach, contact, left behind), bricks and rocks
with and without surface contact, the hand-over on the GPU; two bodies given different
kernels in one dispatch against each alone, no differing word. The one kernel's own
suites unchanged (three emulator suites, the strict GPU runs of every scene).

*What it gives* (ms a frame at a fixed state; fast | strict):

| scene | fast: the one kernel → the choice | strict |
|---|---|---|
| one body, no proxies in the build (the bound) | 1.05 | 1.21 |
| parked by the post, surface contact | 1.41 → 1.07 | 1.58 → 1.22 |
| parked by the post, node contact | 1.23 → 1.07 | 1.40 → 1.21 |
| parked by 24 bricks, surface contact | 1.44 → 1.08 | 1.61 → 1.22 |
| 8 bodies parked, surface contact, one dispatch | 1.43 → 1.10 | 1.62 → 1.25 |
| in contact (post / bricks / rocks with surface contact; bricks by node) | 3.40 / 3.36 / 2.40 / 1.80, the same | 4.24 / 4.10 / 2.93 / 2.01, the same |

A frame with nothing in reach is within 0.02 ms of a build that has no proxies at all:
the 0.35 (surface contact) and 0.16 (node contact) the contact code cost a frame that did
not use it are gone. A frame in contact is the one kernel's pipeline and costs what it
did. Whole runs, each frame timed (`--list 7`):

| run | frames by kernel | fast, mean ms a frame: one kernel → the choice | strict |
|---|---|---|---|
| the post 2 m ahead at 11 m/s, surface contact, 60 frames | 10 contact-free (approach), 28 whole, 22 contact-free (the post left behind) | 1.95 → 1.78 | 2.34 → 2.17 |
| four rocks 1.5 m above at 6 m/s, surface contact, 40 frames | 14 contact-free, 26 whole | 2.00 → 1.87 | 2.36 → 2.25 |

The approach frames are 1.30–1.51 → 0.99–1.17 fast; the frames after the post is left
behind 1.74–1.86 → 1.42–1.51 (the car is bent and in self-contact there).

*What its parts cost.* The guard: under 0.01 ms a frame, inside the noise (1.069 with,
1.078 without, fast; 1.220 and 1.213 strict). The whole kernel's dispatch after a
contact-free one that finished: nothing measurable, by indirect count or over every
group (1.153, 1.149; 1.148 with no such dispatch). The proxies' free flight: 0.04 ms a
frame done a substep at a time in the loop, nothing measurable done at the dispatch's
end. The kernel that takes a frame up, used from substep 0 in place of the one kernel
(`--reach-cont`): within 0.035 ms either way (post +0.03, bricks −0.01, rocks −0.01,
bricks by node +0.03 fast; +0.02, +0.03, +0.02, −0.02 strict), which is this kernel's
code-shape noise; the default uses the one kernel for a frame with no contact-free body,
so such a frame cannot be slower.

*A measurement trap found on the way.* The first numbers had every in-contact frame 0.02
to 0.03 ms slower with the choice. It was the host: reading the body's block (its nodes
and proxies, for the plan) from the shared buffer just before the dispatch adds that to
the dispatch's GPU time with any kernel (`--reach-host`, `REACH_HOST`: the read alone
does it; writing eight words does not, nor a 20 µs wait). Timed frames are given plans
worked out before; in the World the plan comes from what the host already holds.

*Not covered.* Two soft bodies (a pair's bodies would have to end at the same substep,
which takes a meeting a substep; not built: bodies with no partner in the frame are this
kernel's, and that is the broad phase's choice). One box for everything a body could
touch: the car among the scattered bricks is inside it and runs the whole kernel (10 of
60 frames contact-free in that run); the World's gather of proxies a body is the finer
filter. A frame handed over part way pays both kernels' start and end (once an
approach; not measured apart).

### 8.43 The floor kernel with surface contact: the car's skin against the world's shapes (2026-11-19)

(#137's second part, §8.42's kernel with the engine's opt-in surface contact.)

**Surface contact (the car's skin against the world's shapes), added 2026-11-19.**
Mechanism 3 above is the engine's default rule against a rigid shape, *node contact*: a
node takes a force once the node itself is strictly inside the shape. Nothing between the
nodes is tested, so a shape thinner than the node spacing passes into the car. The
engine's opt-in *surface contact* (`surface_contact`, spec §38, `explicit_surface.rs`)
also tests every collision triangle, every beam that is no triangle's edge (2,796 on the
sedan) and every node within the 2.5 cm band, against every shape. What it changes, on the
CPU reference (`surfdemo`: the sedan into a fixed 12 cm square pole that passes between
the nose's nodes; "inside" is how far the pole's near face is past the car's front-most
node, sampled every four frames):

| speed | node contact only: deepest inside, speed when 0.4 m in | surface contact: deepest inside | steps a node is inside the pole |
|---|---|---|---|
| 5 m/s | 0.41 m | 0.12 m | 0 (node only: up to 170 node-steps a frame) |
| 11 m/s | 0.99 m, still 10.2 m/s at 0.43 m | 0.26 m | 0 |
| 20 m/s | 1.53 m, still 16.0 m/s at 0.97 m | 0.44 m | 0 |

With node contact the pole is a metre into the car before the car has slowed; with
surface contact it is stopped at the skin and no node is ever inside it. One thing to
check on the engine side, not this kernel's: with surface contact the car leaves the pole
again at 2.6, 4.6 and 5.1 m/s (from 5, 11 and 20 m/s). That is the reference's own
response (the kernel reproduces it bit for bit) and I have no reference run that says
what it should be.

*In the kernel (`floor2.metal` `SURF`).* Boxes and spheres, on rigid bodies (two-way) and
fixed in the world; the reference's casts for fast bodies (§38.4) are not ported and the
kernel's status says so if a body is fast enough to need them. The geometry is not
rewritten: `slang/physics/feature_query.slang` (the engine's strict closest-feature
queries, the twin of `engine_schema::feature_query`) is converted to Metal as text when
the kernel is assembled (`floor.rs` `fq_msl`: `out` parameters to references, constants to
macros, literals to single precision), so there is one source. Checks, all 0 differing:
the converted queries against the Rust module on 300,000 random triangle, beam and point
cases against boxes and spheres, compiled as C++ and by Metal on the GPU (`fqcheck`,
`fqcheck --gpu`); the whole pass against the reference in the emulator, every state word
and the proxies', on a loose 3 t post (24 frames, both lane orders, by step, at 11 and
25 m/s), the brick wall (14), the falling rocks (16) and a fixed pole (14, both orders);
and strict on the GPU, bit for bit every frame, on all four.

*What a query costs alone* (`fqcheck --gpu --latency`: one warp, every lane the same 32
cases; µs a query, fast | strict):

| element against shape | apart, in the band | touching or crossing | near, no contact |
|---|---|---|---|
| triangle, box | 20–23 \| 34–37 | 47–52 \| 74–81 | 11 \| 19 |
| beam, box | 11 \| 20 | 22 \| 34 | 5 \| 8 |
| triangle, sphere | 1.8 \| 2.7 | 2.8 \| 4.9 | 1.2 \| 1.6 |
| beam, sphere | 1.4 \| 2.0 | 2.3 \| 2.9 | 1.1 \| 1.3 |
| point, either | 0.6 \| 0.8 | | 0.6 \| 0.7 |

A box query is 12 box edges of six (triangle) or three (beam) closest-point candidates
plus the run inside, walked up to four times (the least gap; the candidates at it; then,
crossing, the support along the overlap axis and the candidates at it). One such query is
as long as a whole substep of the car at rest (32 µs), and a stage that runs a lane a pair
pays its slowest lane, so a warp holding a triangle pair and a beam pair pays both. Forty
values kept live across the queries (as the car's kernel keeps) changed nothing (20.4
against 19.9 µs).

*The pass.* After the self hits, before the proxies integrate:
1. the body's box (a warp reduces it in the S1 phase, where the nodes are final), and the
   slots whose cull boxes meet it within the band: none, and the pass is one barrier;
2. the search: each lane's triangle, node and beams against the near slots' boxes (in
   workgroup memory), the pairs that pass on a list;
3. a lane a pair: the separation cull; a point, or anything against a sphere, tested
   here; a triangle or a beam against a box becomes a staged pair with a task for each
   box edge within the band of the element's box, and one for its run inside;
4. stage A, a lane a task: the task's six candidates' gaps, and the pair's least gap by
   an atomic min (the walk's first pass);
5. stage B: a task that holds a candidate within the tie of that gap walks it again alone
   and records it with its place in the walk;
6. stage R, a lane a pair: the query's steps between its walks and its later passes, over
   the few recorded candidates in the walk's order (the overlap axis only when touching
   or crossing);
7. a lane a hit: the response from the nodes' forces (published once to workgroup
   memory), the counts and the body, into the fixed-point sums.

Every candidate that the reference's pass would count is counted (an unculled edge only
adds candidates past the band), and each is evaluated by the engine's own function, so
the result is the reference's. Overflow of a staged list is not a failure: the pair takes
the whole query in its lane.

*Cost* (ms a frame, one sedan, fast | strict; the node-contact column is the same scene
and frame without surface contact, which is another trajectory by then):

| state | node contact | surface contact | first version, a lane a pair |
|---|---|---|---|
| parked 0.3 m from a post, nothing near | 1.26 \| 1.46 | 1.51 \| 1.69 | |
| a loose 3 t post, 11 m/s, frame 5 | 2.13 \| 2.66 | 3.53 \| 4.35 | 3.97 \| 5.53 |
| 24 bricks, frame 12 | 1.83 \| 2.08 | 3.47 \| 4.22 | 5.06 \| 6.76 |
| four falling rocks (spheres), frame 14 | 1.66 \| 2.06 | 2.52 \| 3.04 | 2.78 \| 3.34 |

Surface contact in contact is 0.9 (spheres) to 1.6 ms (boxes) a frame on top of node
contact, 0.25 when nothing is near. In copies (fast, in contact): post 1: 3.54, 40: 4.67,
41: 8.11, 80: 9.59; bricks 1: 3.47, 40: 4.51, 41: 7.93, 80: 9.20; rocks 1: 2.51,
40: 3.15, 41: 5.64, 80: 6.59. The same staircase as without it: a body a core.

*How it got there, and what each step measured* (whole frames, two builds in one batch;
fast | strict where both were taken):

| step | post | bricks | rocks |
|---|---|---|---|
| a lane a pair, the whole query | 3.97 \| 5.53 | 5.06 \| 6.76 | 2.78 \| 3.34 |
| the box tests staged (a lane a pair and box edge) | 3.65 \| 4.68 | 4.72 \| 5.68 | 2.77 \| 3.36 |
| a body's shapes from its own list (the proxies' integration) | 3.68 \| 4.69 | 3.87 \| 4.79 | 2.70 \| 3.24 |
| the search's pairs listed from a mask | 3.63 \| 4.62 | 3.56 \| 4.47 | 2.49 \| 3.04 |
| the run inside a box apart from the edges' tasks, the tasks reserved from a mask | 3.56 \| 4.40 | 3.47 \| 4.26 | 2.53 \| 3.06 |

**A loop in which each lane's turn is a different iteration.** This is what the last three
steps have in common, and it is the finding of the round. A warp runs a loop's body once
for every iteration in which *any* of its lanes takes it. Written as "scan the items, skip
the ones that are not mine, do the work on mine", 24 lanes with one item each run the
work 24 times, in series, each for one lane:

- *The proxies' integration* scanned the shapes for the body's own (`if (shape.body !=
  lid) continue;`), then exchanged six sums, integrated and normalised. 24 bodies: 30 µs a
  substep, 1.0 ms a frame, a third of the node-contact frame, and until the profiler
  split the group it had been read as the nodes' shape tests. With each body's shapes in a list
  of its own (`T_BSH`) the loop is one iteration: 0.09 ms. Node contact on the boxes
  2.69 → 1.83; mechanism 3's table above is the new figures.
- *The search* stored a pair inside the loop over the near slots: every (slot, lane) hit
  of a warp ran the store and its atomic alone. The slots an element is near are now bits
  of a mask, and the pairs are listed afterwards in a loop over the mask's bits, where
  each lane's j-th pair is the same iteration: the search 0.57 → 0.17 on the bricks.
- *Stage B* looped over a task's six candidates and walked the kept ones: six walks a warp
  for lanes that keep one or two each. A mask of the kept, then its bits in turn.
- *Stage A* held the run-inside task among the edges' tasks, so every warp with one ran
  both walks six times. The run inside is the pair's own lane's task now, and a single
  candidate of it clips one edge, not three.

The rule for a kernel: **give a lane its item directly (a list, or `ctz` over a mask),
never a scan with a skip.** A scan is right only when most lanes take most iterations.
The two-body pass has an append inside its loops over 20 clusters and over 32 points
(`pair.metal`); that was done next and was worth far less than this paragraph expected
(the second round above: those loops held 0.29 ms, not 0.8, and every lane's turn in them
is already the same iteration).

Measured and not kept, each because the whole frame said so: the node pass's shape tests
behind the active shapes' union box, with the shapes' boxes in workgroup memory (the
bricks' union is the whole car: 4.71 → 4.76); a hit's node counts in a lane-a-hit phase
before the apply (rocks 2.49 → 2.56); the lane's own state written to its rows for the
length of the pass (below).

*Where the time is now* (the profiler, fast, ms a frame; its groups trade a tenth with
their neighbours, the whole frame is the measure):

| group | post | bricks | rocks |
|---|---|---|---|
| near slots, search | 0.21 | 0.33 | 0.26 |
| cull, tasks, direct tests | 0.35 | 0.32 | 0.32 |
| A, an edge's gaps | 0.38 | 0.36 | 0 |
| B, kept candidates | 0.19 | 0.14 | 0 |
| R, a pair's contact | 0.42 | 0.42 | 0 |
| hits applied | 0.27 | 0.42 | 0.27 |
| the proxies' integration | 0.09 | 0.09 | 0.11 |

*With nothing near* the pass costs 0.25 ms a frame, and that is three things, measured
apart on the parked car (fast): the pass's first phase, 0.05 (the frame with it left out
by a switch); its bookkeeping (the box reduction, the fast-body test, the counters' resets),
0.06 (a build with the pass's code removed: 1.32 against 1.26); and 0.14 to 0.2 for the
pass's code being in the kernel at all, not run (1.46 with the pass switched off against
1.32 with its code removed). It is not the lane's state being live across that code:
writing the state to its rows before the pass and reading it back after left the parked
frame where it was and made the frames in contact 0.36 ms longer. It does not follow the
kernel's size either (the build without the fallback queries is the same). That was
taken apart later ("What code that does not run costs", above): the parked car with
surface contact is 1.41 now, node contact alone 1.23.

*What is left, in order.* What remains of that 0.14 (about 0.1). Then the stages are each a few tenths with no one
cause: the kept candidates' rows are read from device memory in stage R (three scattered
rows a candidate a pass); a hit's 24 fixed-point adds are device atomics; the search runs
every substep where the body's other searches keep a list over several. And the query
itself: four walks of up to 78 candidates is the reference's definition, so a cheaper
exact box test is a change to both twins on the engine side, not to a kernel.

**Not built, and what the numbers do not cover.** Raycast wheels and the drivetrain (the
scenes' cars have none, as the scenario matrix's soft rows); piles of three or more bodies
(the owners' turns generalise, the pairing does not yet); static, hull and capsule
proxies, and for surface contact cylinders, triangles (a trimesh's, a heightfield's) and
the casts; a car's beams against another car's triangles (car against car is node
against triangle only); Chrome and WGSL (Metal only, by the brief); a body split over
several cores.
Production's figures are the matrix's, on its own scenes: the same car and speeds, not the
same frames.

### 8.44 Porting the floor kernel's methods into production: what maps, what does not, measured (2026-11-19)

PR #137 (`claude/gpu-floor-kernel-cc397d`, §8.42 and §8.43 here) measured the floor of one sedan in a
purpose-built Metal kernel: one car at rest 1.1 ms a frame fast, 1.25 strict; two cars head-on
3.5 / 4.2. This section asks how much of that the production explicit kernels can take, one
lossless switch at a time, and answers from measurements on `main` (96904a44). Native M3 Max,
`vehicle_lab --profile fast`, every GPU run through `with-gpu-lock.py`; the fixtures are
`fast-check.sh --make 1 8` on this commit (`out/fixtures`, one car at t 2.5, 6 and 20 s; eight
cars at t 2, 3.7, 8, 16, 20 and 30 s). Wall figures are `--ab-perf-masks` medians of 60 frames
unless marked. No kernel or default changes in this section: every switch considered below
either already exists in production, or its measured ceiling was too small to build, or it
needs the dedicated kernel described at the end.

**Decision first: two workgroups in lockstep across cores is not safe in Chrome, so vehicle
contact keeps its per-substep dispatches.** The floor's mechanism 2 (each body on its own
workgroup, a device counter and a bounded spin as a barrier across cores, every shared word
stored atomically) is what takes the head-on pair from production's 15.4 ms to 3.5–4.2. It
rests on two properties WebGPU does not give:
- *Forward progress across workgroups.* WGSL's only barriers are workgroup-scoped
  (`workgroupBarrier`, `storageBarrier`); neither the WGSL nor the WebGPU specification promises
  that two workgroups of one dispatch are resident at once. A spin that waits for another
  workgroup is correct only on a GPU that happens to schedule both. On Apple's 40 cores with
  two to forty workgroups it held (#137's `xsync`, 0 missed); a smaller GPU, a busy one, or a
  software adapter that runs workgroups on a few threads (SwiftShader, which the Cursor Cloud
  VMs and CI use, `AGENTS.md`) need not, and the bounded spin then ends with a status bit, i.e.
  a failed frame, every frame.
- *Visibility of what another workgroup stored.* WGSL atomics are relaxed and there is no
  device-scope fence, so a workgroup's plain stores are not ordered for another workgroup by
  anything the language offers. #137 measured the consequence on Metal (plain stores by one
  workgroup were stale on another core on 4,095,999 of 4,096,000 reads; atomic stores were
  right on every read). A WGSL port would have to make every exchanged word an `atomic<u32>`
  scalar (no vector or float atomics), and its technique 8 (reading atomically stored words
  with one plain vector load) is not expressible: a WebGPU dispatch may not bind one buffer
  range as both atomic and plain writable storage.

So for cars that touch, production keeps a dispatch a substep (the self-split members' step,
the fleet front, the levels; §8.25–§8.37), and only the floor's in-workgroup parts can be
taken over. That also bounds what this port can do for the eight-car mountain and the derby:
the floor's 3.5 ms head-on figure needs the lockstep and is not reachable in Chrome. A native
host could use it (a Metal or CUDA path of its own, where forward progress and memory scopes
are explicit); that is a separate decision and not built here.

**What each of the floor's ranked techniques is in production today** (#137 §8.42's table,
in its order):

| # | floor technique | floor's gain | in production | port as a switch? |
|---|---|---|---|---|
| 1 | all 34 substeps in one dispatch, state in registers and workgroup memory | −42 % (wall crash) | the in-workgroup frame kernel (`explicit_beams_frame`) already steps a car's whole frame in one dispatch, but its state lives in device memory | no: see "one dispatch a frame" below; its gain is #1 and #2 together |
| 2 | a beam's hot words in registers, one scalar an end through workgroup memory, a gather per node in the reference's order | −56 % at rest | the gather per node in the reference's order is production's (`beamSum` over the node's incidence entries, four a pass); beams write float4 forces to device memory each step | only inside a kernel that keeps the beams on fixed lanes across substeps (#1) |
| 3 | vehicle contact on both bodies' workgroups in lockstep | head-on 15.4 → 3.5 | — | no: the decision above |
| 4 | hit passes from workgroup memory | −20 % (head-on) | `PERF_HITS_LOCAL` (§8.33, on), `PERF_SELF_HITS_LOCAL` (§8.36, off) | already ported; what is left is device latency, below |
| 5 | two pair lists, the working one filtered from a wide one | −7 to −9 % in crashes | the reused pair list (`pairReuse*`, `PERF_WIDE_LIST`), kept candidates across substeps (`PERF_FLEET_KEEP`) | the wreck at rest builds rarely (t 20: the list kernel 1.34 ms a frame timed apart, its builds a small part); not measured in a crash here |
| 6 | only nodes a contact wrote read and clear their contact words | −8 to −11 % | every node reads and clears its accumulators (on a step with hits), its §37 position step, its two-body count and rolls its anti-clip memory, every step | measured ceiling 1–3 %: not built (below) |
| 7 | the pair filter applies the rule's over-the-triangle test | −9 % at rest | each listed pair goes through `pairReuseTest` (the full rule) a thread a pair over the GPU | not measured here |
| 8, 9 | atomic stores read with one vector load; partner's cluster boxes | −7 %, −3 % | — | only with 3 |
| 10 | no vector indexed by a variable in the row loop | −2 % | | nothing comparable found |
| 11 | fast math | 10–20 %, not bit for bit | PR #135 (bit-exact as an engine-wide setting) | not this PR's |

**One car today, by pass** (the wide passes, the lab's default; `--ab-wide-profile`, each
dispatch timed apart, ms a frame of 34 substeps; the walk's figure is the timestamps of
empty indirect dispatches, §8.23, and is left out):

| pass | t 2.5 (driving) | t 6 | t 20 (wreck at rest, hits every step) | floor's phase (#137, at rest, fast) |
|---|---|---|---|---|
| nodes | 0.78 | 1.05 | 0.77 | node pass 0.08 + force gather 0.19 |
| beams | 0.53 | 0.53 | 0.53 | beams 0.24 |
| centred damping (two passes) | 0.51 | 0.51 | 0.52 | centred scalars 0.08 + gather 0.17 |
| list kernel (self-collision's list; the next step's coupling and wheels on two more workgroups) | 0.95 | 1.34 | 1.34 | pair rows 0.27, rebuilds 0.03 |
| pairs | 0.45 | 0.52 | 0.54 | candidates' rule 0.09 |
| hits (self-collision hits, tyres) | 0.41 | 1.51 | 1.63 | hit passes 0.10 |
| aux, begin, end | 0.10 | 0.10 | 0.10 | |
| **frame (wall, untimed)** | **3.8–4.0** | **5.3** | **5.3** | **1.1** |

The floor's scene has no wheels, no World coupling and no terrain proxies; production's
one-car figures include them (§8.31: the wheels ≈0.85 ms of wall time before
`PERF_AHEAD_APART`; the node pass's proxy contacts ≈0.38).

**One dispatch a frame (technique 1), measured.** Production already has it: with
`--wide 0` a car's frame is one dispatch of one 512-lane workgroup (`explicit_beams_frame`).
It is slower than the wide passes, bit for bit the same state (`fast-check.sh wide`, 20
frames): t 2.5 4.64 against 4.02 ms, t 6 6.28 against 5.38, t 20 6.26 against 5.54. Where its
frame goes (`--ab-ablate` on the ablation kernel, time only, 20 frames; t 2.5 / t 20, against
4.65 / 6.14 with nothing skipped):

| skipped | t 2.5 | t 20 |
|---|---|---|
| everything (the dispatch alone) | 0.47 | 0.47 |
| all but the frame's start | 0.65 | 0.67 |
| the whole step loop (the frame's start and end, a barrier a step) | 1.56 | 1.83 |
| node updates | 4.16 | 5.66 |
| beam updates | 4.22 | 5.91 |
| the coupling's step and the wheels ahead | 4.28 | 5.91 |
| the wheels' step | 4.18 | 5.85 |
| selfApply | 4.74 | 5.21 |
| the grid's build | 4.41 | 4.76 |
| the pair tests | 4.55 | 4.67 |

A step is 118 µs on one core against the floor's 32: no single part carries it. The floor got
from its first layout (3.0 ms at rest, about this kernel's shape: state in device memory) to
1.25 by #1 and #2 together: the beams on fixed lanes for the whole frame with their hot words
in registers, positions and velocities in workgroup memory, one scalar a beam end. Without the
registers a dispatch a frame saves only the dispatch boundaries (§8.31: ≈2 µs each, 0.4–0.5 ms
a frame for the wide path's 243), which the wide passes' spread over many cores outweighs.
And the frame kernel cannot simply be widened: its modules declare 31.6 KB of the 32 KB of
workgroup memory the device grants (32 KB on this adapter; `wg_mem.py`: `explicit_beams_solo_aero_wide` 31,628 B, the
grid's buckets and scan, the coupling's spheres, the hit copies), so neither 1,024 lanes nor
the floor's 30 KB of positions, velocities and beam scalars fit beside what it holds now.

**Hit passes from workgroup memory (technique 4), what is left.** `PERF_HITS_LOCAL` is the
floor's idea already (each hit's nodes copied into workgroup memory once, the rule's phases on
the copies, one write back), yet one car's hits still cost about 1 ms a frame with 9 hits a
substep. Time-only cuts of `selfApplyLocal` (each one returns at a barrier; the frame then
runs from another state, so these are attributions, not savings), ms a frame:

| `selfApplyLocal` | t 6 | t 20 |
|---|---|---|
| left out | 4.41 | 4.30 |
| to the end of its loads | 4.70 | 4.56–4.58 |
| and the slots (first item naming a node; the loads' results first used here) | 5.03 | 4.95 |
| and the round (the first pass, the lead lane's rule) | 5.18–5.20 | 5.11 |
| and the role and combine phases | 5.34 | 5.31 |
| all of it | 5.36 | 5.40 |

About 0.6 of the 1.0 ms is the loads: each item's lane walks a chain of device reads (its
record, its node, the node's info, the incidence entries four at a time, the accumulators by
atomic loads, which are not reordered) before the first barrier can use them. The floor has no
such chain because its nodes' state is already in its workgroup's memory. Taking these
loads out needs the same thing #1 needs.

**Contact words read only by nodes a contact wrote (technique 6), its ceiling.** On the lab
car every node of every step rolls its anti-clip memory (two atomic loads, four stores), takes
its §37 position step (three loads, three stores), clears its two-body count, and on a step
with hits reads and clears its three accumulators: up to 19 device atomics a node. A
time-only build that skipped all of them (not lossless; ceiling only), one car, median /
fastest tenth: t 2.5 3.84 / 3.73 → 3.78 / 3.68; t 6 5.34 / 5.23 → 5.28 / 5.17; t 20 5.30 / 4.58 →
4.49 / 4.43 (the baseline's frames were bimodal there). So 1–3 % at most. Doing it losslessly
needs a written-bits word a node set by every writer (`accAdd`, three direct-add sites,
`shiftAdd`, `clipRecord`, the two-body counts) and kept conservative across restores and A/B
switching; for 1–3 % it was not built. In the floor it was 8–11 % because its frame had little
else left; here the node pass's own cost is mostly the World coupling's proxy contacts and the
incidence gather.

**Eight cars on the mountain today** (`--ab-profile`, each dispatch timed apart, ms a frame;
untimed frame 19.8 ms at t 3.7, 17.2 at t 16):

| dispatch kind | t 3.7 | t 16 |
|---|---|---|
| members' step (`explicit self step`, a workgroup a car a substep) | 5.78 | 6.16 |
| contact levels | 6.21 | 1.16 |
| fleet walk | 1.39 | 2.60 |
| members' self-collision hits | 1.68 | 1.70 |
| the front's other dispatches (triangles and points, owners' parts, grids, pairs, prefixes, throttles) | 3.71 | 4.22 |
| self pairs, self walk | 1.55 | 1.55 |

The members' step is about 170 µs a car a substep, the floor's first layout with a dispatch a
substep (5.84 ms a frame in #137's ledger is 172 µs a substep): the same shape and the same
cost. By the decision above these cars stay on a dispatch a substep, so of the floor's methods
only #2's layout inside a substep and #4–#7 could apply; each is a few tenths of a millisecond
of a 17–20 ms frame (§8.36 has the step's split: coupling 1.3, nodes 1.3, beams 0.75).

**Whole runs, before** (native, 40 s, `--self-split 1`, the defaults; vehicle GPU ms a World
step from t ≥ 0.5 s, median / p95 / worst; this section changes nothing, so after is before):

| | explicit frame | vehicles (frame + coupling passes) | wall ms a World step |
|---|---|---|---|
| 1 car, mountain | 4.66 / 4.86 / 5.27 | 4.84 / 5.12 / 5.38 | 5.89 / 6.20 / 6.76 |
| 8 cars, mountain | 17.28 / 20.84 / 23.65 | 18.04 / 22.17 / 23.91 | 19.36 / 23.51 / 25.16 |
| 8 cars, derby arena (native: parked, the derby's AI runs in the page) | 5.73 / 5.91 / 5.93 | — | 7.12 / 7.57 / 19.6 |

The derby with its AI is a Chrome scene; §8.38 (physics first) has it: eight cars in contact
16.4 ms a step. Floor for comparison: one car 1.1–1.25 (at rest) and 1.83–2.2 (wall crash); two
cars head-on 3.5–4.2 (lockstep, not available in Chrome).

**What would reach the floor for a solo car, and what it needs.** A dedicated solo-car frame
kernel: one workgroup a car for all 34 substeps, the floor's layout (beams on fixed lanes with
their hot words in registers, slot loops written out; positions, velocities and beam scalars
in workgroup memory; hits named into the scalars' array), and production's other work run as
phases inside it at the points the frame kernel runs them today (the coupling's step and
proxy contacts, the raycast wheels and drivetrain, sleep, debris retirement, surface
contact, aero triangles, centred damping, deform and break groups, slides). Its
prerequisites, each measured above: the workgroup memory budget (32 KB in Chrome) shared in
time between the beam scalars and the grid and coupling arrays, which today hold it for the
whole frame; 1,024 lanes requested from the device in Chrome; every count of the car a
specialisation constant or a compile per car shape (#137: +6 % as uniforms); and the
emulator's discipline (#137) for checking it on the CPU before the GPU. It is a kernel of the
floor's size (2,166 lines of Metal there, without wheels, coupling or terrain), not a switch,
and it helps a car only while it touches nothing but the World; a car in vehicle contact goes
back to the per-substep path. That is the next step if the solo car's 4–5 ms matters more than
the touching cars' 17–20.

### 8.45 Status against the floor goal: per phase and per car count (2026-11-19)

**Correction (§8.46, PR #141, merged with this section).** The hits' "14–16×" below set
production's t 20 (22–25 hits a step) against #137's floor at rest with fast arithmetic
(0–2 hits). Against #137's kernel run again strict on states with hits, its hit passes
cost ≈0.75 ms a frame (0.74 mid-crush, 0.78 the wreck); production's hits dispatch after
#141's slot map is 1.24 ms at t 20: **1.6×**, not 14–16×. The rows below are left as
measured, with a pointer to §8.46 where this changes them.

No new measurement: this section collects the open PRs' numbers to close the goal's
criteria 3 and 5, and gives a verdict on all five. The criteria are §8.6's table: (1) one
car's worst step 1.0 ms, natively and in Chrome; (2) eight cars crashing, worst step 3 ms,
120 steps a second with none refused; (3) every phase within 2× its floor, or a measured
reason it cannot be; (4) bit-for-bit proofs for every change; (5) per phase before / floor /
after, and worst and p95 per car count. Sources are cited by PR and branch (written while the PRs were open; their sections
are numbered here as merged together):

- main (96904a44): §1.3 (the estimated floor), §8.16 (the first before / floor / after),
  §8.23 (one car's passes), §8.35 (the dependent-ALU microbenchmark), §8.37 (levels once),
  §8.38 (physics first);
- #137 `claude/gpu-floor-kernel-cc397d` §8.42, §8.43: the **measured** floor, from a purpose-built
  Metal kernel for the same sedan (no wheels, no World coupling, no terrain proxies);
- #138 `perf/floor-port` §8.44: production per pass, whole runs on main's defaults, and
  the decision that lockstep across workgroups is not safe in Chrome;
- #139 `perf/solo-car-kernel` (not merged; its record is PR #139): the floor's layout as a WGSL kernel (no-go);
- #135 `perf/fast-determinism` §8.40: bit-exact against fast arithmetic, per pass and per
  car count;
- #134 `perf/keep-with-defaults` §8.39: kept candidates on by default, 8 and 32 cars;
- #136 `perf/draw-batched-skin` §8.41: what the drawing costs the World's steps.

**The floor used.** §1.3's 0.3–0.5 ms was an estimate (cache bandwidth and FLOPS). #137
measured the floor directly and found it 3–4× higher: one car at rest 1.04–1.10 ms a frame
fast, 1.25 strict (bit for bit with the CPU reference); a 16 m/s wall crash 1.83 fast
mid-crush, 2.2 strict. Its cost model also corrects §8.35's reading: a phase on one core is
bound by the core's throughput (0.35 ns a warp-instruction, about ten instructions a memory
access), not by one lane's dependent chain; §8.35's 8 ns a step was eight warps sharing the
core. #135 bears this out from the other side: taking the strict XOR out of every operation
takes one car's frame down 7 %, not the 30–50 % a doubled chain would.

#### One car, per phase (criteria 3 and 5)

ms a frame of 34 substeps, native M3 Max, the wide passes (the lab's default), each
dispatch timed apart (timed apart, the passes sum to 15–20 % more than the untimed frame,
§8.23). Production is at t 20 s (the wreck at rest, 13–16 hits a step). The floor's phases
are #137's profiler on its fast build at rest; no per-phase strict figures were taken
(strict is about 1.15× fast for the whole frame), so each ratio is against the lower,
fast floor. Mapping, production pass → floor phase:
- *nodes* → node pass + force gather (production's node pass gathers the beams' incidence
  forces; it also carries the World coupling's proxy contacts, ≈0.38 ms, §8.31, which the
  floor's scene does not have);
- *beams* → beams;
- *centred damping* (two passes) → centred scalars + centred gather;
- *list kernel* → pair rows + list rebuilds (production's list dispatch also runs the next
  step's coupling and wheels on two more workgroups, ≈0.5 ms, §8.23);
- *pairs* (the full rule, a thread a listed pair) → the candidates' rule (the floor filters
  the rows first and runs the rule only on a compact list);
- *hits* (self-collision hits and the tyres, ≈0.5 of it, §8.23) → hit passes;
- *aux, begin, end* → a dispatch with every phase left out.

| phase | before (§8.16, 2026-11-15) | after, bit-exact (#138) | after, bit-exact → fast (#135, one run) | floor (#137, fast) | ratio: bit-exact (#138) / fast (#135) | within 2×? |
|---|---|---|---|---|---|---|
| nodes | 0.6 (the coupling not folded in) | 0.77 | 0.872 → 0.805 | 0.08 + 0.19 = 0.27 | 2.9× / 3.0× | no (1.4× without the coupling's 0.38) |
| beams | 0.4–0.5 | 0.53 | 0.548 → 0.453 | 0.24 | 2.2× / 1.9× | fast only |
| centred damping | (off then) | 0.52 | 0.530 → 0.514 | 0.08 + 0.17 = 0.25 | 2.1× / 2.1× | no, by 5 % |
| list kernel | ≈2.9 with walk and pairs, in the workgroup | 1.34 | 1.387 → 1.358 | 0.27 + 0.03 = 0.30 | 4.5× / 4.5× | no |
| pairs | | 0.54 | 0.553 → 0.503 | 0.09 | 6.0× / 5.6× | no |
| self-collision together (list + pairs) | ≈2.9 | 1.88 | 1.94 → 1.86 | 0.39 | 4.8× / 4.8× | no |
| hits, with the tyres | 2.1 | 1.63 | 1.605 → 1.361 | 0.10 (settled: 0–2 candidates) | 16× / 14× | no |
| aux, begin, end | 0.5 (≈10 dispatches a substep) | 0.10 | 0.073 → 0.064 | 0.10 | 1.0× / 0.6× | yes |
| **frame, untimed** | 7.37 (the run's worst) | 5.3 | 4.71 → 4.39 | 1.04–1.10 fast, 1.25 strict | 4.2× against strict / 4.0–4.2× against fast | no |

The same in a crash, as near as the sources come (production at t 6 s, after the
mountain's tumble, #138; the floor's 16 m/s wall crash mid-crush, #137; not the same frames):
nodes 1.05 against 0.27 (3.9×), beams 0.53 against 0.31 (1.7×), centred damping 0.51
against 0.25 (2.0×), list and pairs 1.86 against 0.71 (2.6×), hits 1.51 against 0.39
(3.9×), the frame 5.3 against 1.83 fast / 2.2 strict (2.4–2.9×).

**The core, measured like for like.** #139 timed today's core (node pass, centred damping
and beam pass; contact off, so no proxies, hits or wheels) on the lab's states: 1.32–1.37 ms
a frame natively (the frame kernel; the wide passes 1.34–1.41) and 1.57–1.58 in Chrome,
against the floor's same phases at 0.08 + 0.08 + 0.17 + 0.24 + 0.19 = 0.76 (fast). That is
**1.75–1.8× natively (within 2×) and 2.1× in Chrome** (just over). Against a strict floor
(≈1.15 × 0.76 = 0.87, inferred) the native core is 1.5×.

**Each phase over 2×, and why** (a measured reason, or open):

| phase | over by | can it be closed losslessly in Chrome / WGSL? | evidence |
|---|---|---|---|
| the core in Chrome (nodes, centred damping, beams) | 2.1× (1.58 against 0.76) | **No, by the known route.** The floor's own layout (positions and velocities in workgroup memory, beams on fixed lanes with their hot words in registers, one scalar a beam end) was built in WGSL and is slower: 2.04 against 1.58 in Chrome, 2.37 against 1.33 natively, with seven variants each measured on the whole frame. Fast arithmetic does not help one car in Chrome (+2 to +12 % at the median, #135) | PR #139; #135 §8.40 |
| nodes, the coupling's part | 2.9× with it, 1.4× without | The proxy contacts are the World coupling, which the floor's scene does not have. **Open**: no floor was measured for the coupling | #138 §8.44; §8.31 |
| self-collision list and pairs | 4.8× at rest, 2.6× in a crash | **Open.** The floor's lever here is its filter: the rule's over-the-triangle test and a held side checked before the full rule (at rest 722 candidates a substep → 0–2, −9 % of the frame). Production runs the full rule on every listed pair; the filter was not measured in production (#138). Part of the list dispatch is the next step's coupling and wheels (≈0.5, §8.23), which the floor does not have. §8.34 split the list dispatch's workgroups and stopped under 5 % | #137 §8.42 technique 7; #138 §8.44; §8.23; §8.34 |
| hits, with the tyres | 14–16× at rest, 3.9× in a crash (**corrected by §8.46**: 1.6× against a strict floor with hits, after #141) | **Open; the only known route was measured unfavourable.** Hits already run from workgroup memory (`PERF_HITS_LOCAL`, the floor's technique 4). Of `selfApplyLocal`'s ≈1.0 ms, 0.6 is each item's chain of device loads (record, node, info, incidence, accumulators) before its first barrier. The floor has no such chain because its nodes' state lives in its workgroup for the whole frame. That needs the floor's techniques 1 and 2, and #139 measured that layout slower for the core. Strictness is not the cause: fast arithmetic takes 15 % off the hits (#135). The tyres (≈0.5) have no floor phase (the floor's car has no wheels). The workloads also differ: 13–16 hits a step at t 20 against 0–2 at the floor's rest | #138 §8.44; PR #139; #135 §8.40; §8.23 |
| two cars in contact (frame level) | 3.1–3.7× (matrix head-on 13.2–15.4 against 4.2 strict; fast 12.3 against 3.5) | **No: it cannot be expressed safely in WebGPU.** The floor's mechanism 2 steps both bodies' workgroups in lockstep across cores (a device counter and a spin) and exchanges state through atomically stored words. WGSL and WebGPU promise neither forward progress across workgroups (a spin can hang, e.g. on SwiftShader) nor a device-scope fence, and technique 8 (one plain vector load of atomically stored words) cannot be bound in WebGPU. So vehicle contact keeps a dispatch a substep, and a dispatch a substep costs what the floor's own first layout cost (+72 %; 172 µs a substep in #137's ledger, against production's 170 µs a car a substep in the eight-car members' step) | #138 §8.44; #137 §8.42 |

**Frame level, per mechanism.** Production is the scenario matrix's rows: bit-exact (the
matrix's default), the same car and speeds, but not the same frames. The floor is #137's:
strict for the bit-exact ratio, fast for the fast one.

| mechanism, scene | production bit-exact (as #137 cites; #135's matrix) | floor strict | ratio | production fast (#135) / floor fast | ratio fast |
|---|---|---|---|---|---|
| one body, at rest (matrix drive) | 3.2; 3.31–3.52 | 1.25 | 2.6–2.8× | 3.23–3.33 / 1.04–1.10 | 2.9–3.2× |
| one body, wall crash 16 m/s | 4.1; 4.32 | 2.2 | **1.9–2.0×** | 4.09 / 1.83 | 2.2× |
| two bodies head-on, 14 m/s each | 15.4; 13.36 (13.16 with #134's keep) | 4.2 | 3.1–3.7× | 12.34 / 3.5 | 3.5× |
| two bodies, side impact | 9.3; 9.25 | 2.1 (after the impact) | 4.4× | 8.53 / 3.6 (at the impact) | 2.4× |
| a body into boxes (60 bricks in production, 24 boxes in the floor) | 3.4; 3.72–4.35 | 3.0 | **1.1–1.45×** | 3.81–4.71 / 3.1 | 1.2–1.5× |
| rocks onto the parked car | 3.8; 4.20–4.87 | 2.2 | **1.7–2.2×** | 3.95–5.28 / 2.0 | 2.0–2.6× |

**Eight cars, per phase.** No floor per phase exists: #137 built pairs of bodies, not piles
of three or more. What it gives is a frame-level bound: eight cars in four head-on pairs,
3.49–3.64 ms fast (its staircase, one to eight pairs), with lockstep (Metal only). Production's
eight cars on the mountain (#138, `--ab-profile`, t 3.7 / t 16): members' step 5.78 / 6.16
(a workgroup a car a substep: the floor's first-layout shape and cost), contact levels
6.21 / 1.16, fleet walk 1.39 / 2.60, members' hits 1.68 / 1.70, the front's other dispatches
3.71 / 4.22, self pairs and walk 1.55 / 1.55; the frame 19.8 / 17.2 untimed, about 5× the
pairs' bound. The members' step cannot take the floor's technique 1 while its cars touch
(the lockstep decision above). #134's kept candidates take the t 16 frame to 14.9.

#### Per car count (criterion 5)

Vehicles' GPU ms a World step (the explicit frame and the coupling's passes), steps from
t ≥ 0.5 s, whole runs unless marked; median / p95 / worst. Bold: the best configuration
measured for that scene and host. A range is two rounds of the same run. "Mountain" is
the source's terrain where it does not name one.

| cars | scene | host | configuration | median | p95 | worst | source |
|---|---|---|---|---|---|---|---|
| 1 | mountain, 40 s | native | main's defaults, bit-exact | 4.84 | 5.12 | **5.38** | #138 §8.44 (explicit frame alone 4.66 / 4.86 / 5.27) |
| 1 | mountain, 40 s | native | bit-exact (#135's round 1) | 4.63 | 4.87 | 5.39 | #135 §8.40 |
| 1 | mountain, 40 s | native | **fast** (#135) | **3.95** | **4.46** | not measured cleanly (11.57 with another job on the GPU) | #135 §8.40, round 2 |
| 1 | mountain, 40 s | Chrome | **bit-exact** | **4.27–4.30** | **5.29–5.66** | not reported | #135 §8.40 |
| 1 | mountain, 40 s | Chrome | fast | 4.39–4.80 | 5.32–6.19 | not reported | #135 §8.40 |
| 1 | mountain, 40 s | Chrome | main's defaults (wheels apart) | 4.43–4.52 | 5.36 | not reported | main §8.35 |
| 2 | any | native, Chrome | | not measured as a whole run | | | single frames only: fixtures t 2 / 3.7 / 8 / 16 s, 7.25 / 8.26 / 11.96 / 12.00 bit-exact, 7.15 / 7.78 / 11.26 / 11.69 fast (#135); matrix head-on 13.16 (#134), pile2 8.33 bit-exact / 7.90 fast (#135) |
| 4 | derby, 40 s | Chrome | **keep off** | **9.35** | **12.23** | **13.17** | #134 §8.39 (a run before a machine restart) |
| 4 | derby, 40 s | Chrome | keep on (#134's default) | 11.74 | 13.16 | 14.12 | #134 §8.39 |
| 4 | derby, 40 s | Chrome | bit-exact | 11.27 | 16.16 | not reported | #135 §8.40 (fast: the first 5 s only, 5.52 / 6.17) |
| 8 | mountain ledges, 40 s | native | main's defaults, bit-exact | 18.04 | 22.17 | 23.91 | #138 §8.44 |
| 8 | mountain ledges, 40 s | native | **keep on, bit-exact** | **14.86–14.87** | **17.77–18.00** | 23.46–23.57 | #134 §8.39 |
| 8 | mountain ledges, 40 s | native | fast, keep off | 15.91–15.97 | 18.76–19.36 | **23.03–23.71** | #135 §8.40 |
| 8 | mountain ledges | native | keep on and fast together | not measured | | | |
| 8 | rolling gully, 40 s | native | keep on, bit-exact | 20.71–20.76 | 27.65–27.81 | 33.19–42.35 | #134 §8.39 |
| 8 | mountain ledges, 90 s | Chrome | **keep on** | **14.47–14.70** | **16.39–16.62** | 20.54–21.22 | #134 §8.39 |
| 8 | mountain, about 20 s simulated | Chrome | main's defaults (levels once) | 16.84–16.87 | 18.45–18.48 | **19.99–20.51** | main §8.37 |
| 8 | mountain, the first 5 s | Chrome | fast | 14.39 | 17.92 | not reported | #135 §8.40 |
| 8 | derby, 40 s | Chrome | **keep off** | **13.77** | **15.25** | **17.04** | #134 §8.39 (keep on 14.60 / 16.01 / 19.00; bit-exact 40 s in #135: 16.50 / 20.19) |
| 8 | derby, parked (no AI) | native | main's defaults | 5.73 | 5.91 | 5.93 | #138 §8.44 (explicit frame; not a crash) |
| 32 | mountain, 40 s | native | **main's defaults (levels once)** | **77.36** | **86.50** | **405.0** | main §8.37 (ledges with keep on: 78.11–78.19 / 87.35–87.72 / 407.2–412.4, #134; bit-exact 79.29 / 88.93 / 436.4, #135; fast stopped at t 20 by the time limit) |
| 32 | any | Chrome | | not measured (the page runs at most eight cars on the mountain) | | | #134 §8.39 |
| 32 | derby | native, Chrome | | not measured | | | |

Steps a second in Chrome (real time is 60; 120 was never run): one car keeps 60 (§7.1).
Eight cars on the mountain: 48 (80 %) with nothing drawn, 35–36 drawn (main §8.38, #136
§8.41). Eight cars in contact in the derby: 60 with nothing drawn (16.4 ms a step), 36 → 39
drawn with the batched skin (#136). Four cars in contact: 60 with nothing drawn, 47 → 49
drawn.

#### Verdict per criterion

| # | criterion | verdict | numbers |
|---|---|---|---|
| 1 | one car, worst step ≤ 1.0 ms, natively and in Chrome | **proven unreachable on this hardware** | the measured floor for this car is above it even with no wheels or coupling: at rest 1.04–1.10 fast, 1.25 strict; wall crash 1.83 fast, 2.2 strict (#137). Today's worst is 5.38 native (#138); Chrome's p95 is 5.3–5.7 (#135), its worst not reported since §8.16 |
| 2 | eight cars crashing, worst ≤ 3 ms; 120 steps a second, none refused | **proven unreachable on this hardware** (the 3 ms); 120 steps a second **not met** | one head-on pair alone floors at 3.49 fast / 4.2 strict, and four pairs (eight cars) at ≈3.6 fast, with a lockstep WebGPU cannot do safely (#137, #138). That a pile costs each of its cars no less than a pair does is assumed (#137 did not build piles). Today's worst is 23.0–23.9 native, 20.0–21.2 in Chrome. 120 steps a second needs ≤ 8.3 ms a step against a 14.5–17 median, and Chrome holds 48 of 60 with nothing drawn |
| 3 | every phase within 2× its floor, or proof it cannot be | **not met**: part proven, part open | Within: aux and dispatch 1.0×; the core natively 1.75–1.8× (like for like, #139); beams fast 1.9×; and at frame level the wall crash 1.9–2.0×, boxes 1.1–1.45×, rocks 1.7–2.2×. Over, with proof: vehicle contact 3.1–3.7× (no lockstep in WebGPU, #138); the core in Chrome 2.1× (the floor's layout in WGSL measured slower, #139). Over and **open**: self-collision list and pairs 4.8× (the floor's filter not tried in production); hits 14–16× at rest, 3.9× in a crash (the load chain needs workgroup-resident state, and the one route built for it lost on the core; **corrected by §8.46**: against a strict floor with hits, ≈0.75 ms, production's hits are 1.24 ms at t 20 after #141's slot map, 1.6×); the nodes' coupling part (no floor measured); centred damping 2.1× (5 % over) |
| 4 | bit-for-bit proofs for every change | **met** | every default change carries fixtures, suites, the matrix and, where it touches Chrome, the frame-locked A/B (§8.23, §8.35, §8.37; #134). With #135 bit-exact becomes a setting, off by default: the suites run with it on and pass, and the fast setting is held to measured tolerance tests (within 4× the spread of starts no one could tell apart) |
| 5 | per phase before / floor / after; worst and p95 per car count | **met, with gaps marked** | the tables above. Not measured: two cars as a whole run; 32 cars in Chrome or in the derby; one car's worst step in Chrome since §8.16; keep and fast together; 120 steps a second; eight cars per phase against a floor (no pile floor exists) |

**What would move a verdict.** Only criterion 3's open rows could still be closed by a
lossless change in Chrome: the floor's pair filter (technique 7) in production's pair pass,
and a measured floor for the coupling's proxy contacts. Criteria 1 and 2 need a target at
or above the measured floor (one car 1.25 strict at rest, 2.2 in a crash; a head-on pair
4.2), or a native host with its own lockstep path (Metal or CUDA), which is a separate
decision (#138).

### 8.46 Hits and the self-collision list against their floor, like for like; the hits' slot map (2026-11-19)

The question (after PR #140's §8.45): the two phases of one car's frame furthest above their
floor with no proof they could not close, the self-collision hits (1.63 ms a frame at t 20,
"14–16×") and the self-collision list and pair tests (1.34 + 0.54, "4.8×"). Close them
toward 2× losslessly, or measure why not. Native M3 Max, `vehicle_lab --profile fast`, every
GPU run under `with-gpu-lock.py`; the states are `fast-check.sh --make 1` (one car on the
mountain at t 2.5, 6 and 20 s) and the scenario matrix's rows (`wall`: the crash preset at
56 km/h, t 0.1). Figures are ms a frame of 34 substeps; "timed apart" is each dispatch in a
timed pass of its own (`--ab-perf-masks`, 24 frames); wall figures are medians of 60 frames.
Bit-exact is the default here (#135 is not merged).

**First, the floor was the wrong one.** §8.45 set production's t 20 (the wreck after the
mountain's tumble, 22–25 hits a step) against #137's floor *at rest* (a pristine car, 0–2
hits) with fast arithmetic. #137's kernel was run again here, unchanged (`floor --v2
--profile F`, built from `claude/gpu-floor-kernel-cc397d` outside the workspace), on states
with hits, strict (bit for bit with the CPU reference, as production is) and fast; the hits a
step from its emulator:

| floor phase (#137, ms a frame) | at rest (frame 41), strict / fast | wall crash 16 m/s mid-crush (frame 17, up to 20 hits), strict / fast | the wreck against the wall (150 frames after, 4 hits a step), strict / fast |
|---|---|---|---|
| self hit passes | 0.0005 / 0.002 | **0.74** / 0.39 | **0.78** / 0.37 |
| pair rows + candidates' rule + list rebuilds | 0.43 / 0.35 | **0.91** / 0.78 | **0.51** / 0.46 |
| the whole frame (all groups) | 1.34 / 1.11 | 2.75 / 2.07 | 2.22 / 1.52 |

The floor's hit passes cost about 0.75 ms strict whether a step has 4 hits or 20: like
production's they are a fixed chain of phases (the round and the residual pass's seven), not
work per hit. Against a strict floor with hits the comparison is far from §8.45's 0.10.

**Where production's hits pass goes** (t 20, 22–25 hits; the wall row, 4–10 hits; time-only
cuts, removed after use; a cut that ends a phase early changes the state, so each figure is an
attribution):

| hits dispatch, timed apart | t 20 | wall |
|---|---|---|
| the dispatch alone (return at entry) | 0.19 | 0.20 |
| + lane 0's start (deferred breaks, the list's record) and its barrier | +0.09 | +0.07 |
| + the items' loads (record, node, beam sum, accumulators) | +0.22 | +0.20 |
| + the slots (each item scans every item before it) | **+0.39** | +0.18 |
| + the round | +0.19 | +0.23 |
| + the residual pass's phases and the write-back | +0.34 | +0.22 |
| + the next step's tyres | +0.15 | +0.49 |
| **all** | **1.62** | **1.61** |

Leaving resolvePending out changed nothing; the beam sum is 0.16 of the loads; selfApply's
general path and the tyres compiled out of the kernel (not only skipped) gave the same as
skipping them, so code size and registers are not it.

**The slot map** (perf flag 134217728, `PERF_HIT_SLOT_MAP`). The slot phase was each (hit,
role) item scanning every item before it for the first one naming its node: a serial chain
of workgroup loads that grows with the hits squared, 12 µs a substep at 22 hits. Now the
hits kernel clears a map of the car's nodes in workgroup memory (gsCount's first words, a
word a node, up to 1,024 nodes) before its first barrier on a step with hits; each item takes
an atomic maximum of 256 − q at its node's word and reads its slot back after the barrier;
the slots' copies of the accumulators are written one phase later (the round reads none of
them, and by then no lane reads the map). The same slots, so the same sums of the same bits.

| one car | off | on |
|---|---|---|
| t 20: hits pass, timed apart | 1.665 | **1.244** |
| t 20: the frame, wall median | 5.51 | **5.03** (−9 %) |
| t 6: hits pass / frame | 1.596 / 5.46 | 1.218 / 5.11 |
| wall row: hits pass / frame | 1.674 / 4.81 | 1.612 / 4.75 |
| t 2.5 and drive (no hits): hits pass | 0.41, 0.77 | 0.42, 0.79 |

The first matrix run had the drive row (no hits) +0.3 / +5.1 / +9.5 % over its three runs:
the map was cleared on every step. It is now cleared only on a step with hits (each lane
reads the count); the rerun passed.

**Proofs.** `fast-check.sh perf=134217728` (one car, t 2.5, 6 and 20; 20 frames): bit for
bit. The scenario matrix: every row bit for bit (12 rows; with the map cleared on hit steps
only: drive −4.1 %, tumble −8.8 %, wall −2.4 %, bricks +1.0 %, rocks −1.6 %, PASS). The GPU
suites (`perf-verify.sh` with the defaults and the map, flags 165117893: the loop-cap census,
explicit_beams_bitwise, explicit_contact_gpu, explicit_pair_reuse_gpu,
explicit_beam_order_gpu, debris_retire_gpu, showcase_car_gpu, vv_grid_gpu,
explicit_vehicles, vehicle_lab_reset, vehicle_lab_piece_windows): all pass. Chrome: all 124
modules validate (uniformity included); the frame-locked page A/B (`run.sh --probe --locked
--ab 134217728 --abUntil 40`, one car on the mountain ledges): 40 samples, every state word
the same. `engine-metal` vehicle_parity: 4 of 4 pass. The whole matrix again with the map in the
defaults (`run.sh perf=134217728`, on against off): every row bit for bit, PASS (wall −2.5 %,
headon −3.5 %, tumble +4.4 %, the others within ±3 %; tumble had −8.8 % in the run before:
noise on a row with few hits).

**Whole runs** (one car, the mountain, 40 s; vehicles' GPU ms a World step, median / p95;
the World's wall ms a step, worst; two rounds each, the map on is today's defaults plus the
map):

| | map off (main's defaults) | map on |
|---|---|---|
| native, vehicles | 4.92–4.96 / 5.16–5.21 | **4.25–4.55 / 4.92** |
| native, wall worst a step | 6.91–6.93 | **6.44–6.48** |
| Chrome (frame-locked page probe), vehicles | 4.64, 3.94, 4.62 / 5.96, 5.56, 6.15 | 3.86, 4.88, 3.52 / 4.78, 6.12, 4.30 |

Chrome's whole runs drift by ±20 % from run to run with the same flags (3.5 to 4.9 ms
median), more than the change; over three pairs the map's runs average 4.09 against 4.40
(−7 %), faster in two of the three pairs. On by default (`perf_flags_default`; the lab's
checkbox checked).

**Hits after the map, against the like-for-like floor** (strict). t 20: the dispatch 1.24,
of which the tyres ≈0.15 and the dispatch boundary with lane 0's start 0.28; the hit
application itself ≈0.8 against the floor's 0.74–0.78. **The hit application is at its
strict floor (≈1.05×); the whole dispatch is 1.6× (1.4× without the tyres, which the floor's
car does not have).** At the wall: 1.61 with 0.49 of tyres (the wheels against the wall) and
≈0.82 of hits against 0.74: 1.1×. What is left over the floor is the dispatch a substep (the
floor runs its frame in one) and the wheels.

**What did not pay (hits).** *The items' loads before lane 0's barrier* (a perf flag in a
time-only build, `hitLoad`, when no beam is pending: the same loads beside lane 0's chain):
bit for bit, 1.274 → 1.276 at t 20, 1.627 → 1.662 at the wall; not kept. *Device barriers*:
`sync()` reduced to a workgroup barrier everywhere (time only): no change in any pass.

**The self-collision list and pairs.** §8.34 split the list dispatch (the list's workgroup,
the coupling's and the wheels' workgroups side by side; the dispatch costs its slowest).
Measured again (timed apart):

| list dispatch | t 20 | wall |
|---|---|---|
| all three workgroups | 1.38 | 1.12 |
| the list's workgroup alone | 1.42 | 0.95 |
| the coupling and wheels alone | 0.98 | **1.13** |
| the list workgroup with the coupling's and wheels' code compiled out of the entry | 1.27 | 0.81 |

At the wall the critical path is the wheels' workgroup (the rays against the wall), which no
floor has. The list workgroup's four phases cost about 0.3 each (header and state load; the
tracking and throttles; lane 0's decision; the tests' record and the save), each a short
chain. The pairs dispatch (a thread a listed pair over the GPU) is 0.52: 0.16 the dispatch,
0.10 the list's box-of-cells test, 0.15 tryContact's tests before the anti-clip rule and 0.11
the rule (`selfClip`; its crossing test 0.02).

Against the floor, strict: at the wall 0.95 + 0.52 = 1.47 against 0.91 (**1.6×**); at t 20
1.42 + 0.54 = 1.96 against the wall wreck's 0.51 (3.8×), but these are not the same wreck:
t 20 lists 8,254 pairs a build and rebuilds 4 times a frame (the mountain wreck's torn
parts), the floor's wreck about never.

**What did not pay (list and pairs).** *The list's margin* (lossless: the list is exact at
any margin), 0.05, 0.07, 0.10 (today's) and 0.14 m: at 0.05 the list falls from 8,254 to
6,248 pairs and the builds rise from 4.0 to 6.9 a frame, and the pairs pass stays 0.54–0.56,
the list 1.35–1.44, the frame within noise at every margin: the pairs pass is a fixed chain,
not work a pair. *The floor's filter (technique 7)*: its ceiling here is the anti-clip
rule's 0.11 of the pairs pass, under 5 %; not built. *Lane 0's coupling setup in the list
workgroup* (unused there with the coupling on its own workgroup): −0.055 of the list
workgroup alone (4 % at t 20; nothing at the wall, where the wheels' workgroup is the
critical path); not built. *Shared-atomic contention in the tracking* (eight maxima from
every lane; lane 0's alone): no change. *Device barriers*: no change. Three consecutive
attempts under 5 % of the phase: stopped.

**Why the list stays over 2× at t 20.** It is four phases of short chains in one workgroup
plus a dispatch a substep for the pairs. On the floor's layout those chains read workgroup
memory; production's read device memory because the car's state lives there between
dispatches, and moving it into the workgroup is #139's solo kernel, measured slower for the
core in WGSL. The one lever found is the coupling's and wheels' code sharing the list's entry
(0.15 of the list workgroup), and it is not separable: a second entry is a second dispatch,
which runs after the first, not beside it.

| phase, one car | before (§8.45 / main) | floor, like for like (#137 strict) | after | ratio after |
|---|---|---|---|---|
| hits dispatch, t 20 | 1.63–1.67 | 0.78 (wreck) | **1.24** | 1.6× (hit application ≈1.05×) |
| hits dispatch, wall | 1.61–1.67 | 0.74 (mid-crush) | 1.61 (0.49 tyres) | 2.2×; 1.1× without the tyres and the dispatch |
| list workgroup + pairs, wall | 1.47 | 0.91 | 1.47 | 1.6× |
| list workgroup + pairs, t 20 | 1.96 | 0.51 (another wreck) | 1.96 | 3.8× (a measured reason above) |

### 8.47 Three small lessons of the gpu-floor review tried on production's kernels: none pays (2026-11-20)

(The gpu-floor experiment itself is §8.42 and §8.43.)

The gpu-floor experiment's review (`gpu-floor-vs-production.md` on PR #137, §6.1) lists
three small changes to production's kernels as they are: (a) contact words read and
cleared only for nodes a contact wrote; (b) loops where each lane's turn is a different
iteration given their items directly; (c) what a phase's hot loop reads that does not
change read again at the phase's start. Each was measured before anything was built.
**None reaches 5 % of a frame on its best row, so none is kept: `main`'s kernels and
defaults are as they were, and no perf switch was added.** Native M3 Max, `main` at
96904a44, the scenario matrix (`--profile fast`), the page's sedan.

| | bound or result, of a frame | built | kept |
|---|---|---|---|
| (a) contact words only for written nodes | 0 to 2 % a row (0.01 to 0.17 ms), the whole read-and-clear | no | no |
| (b) the proxies' integration | 0 (−0.2 to +0.5 %): each body already scans its own range | no | no |
| (b) the pair list's walk | under 3.5 % for the whole walk (0.1 to 1.1 ms); its appends are a part | no | no |
| (b) surface contact's loops, the in-workgroup walks | not in any saved state: not timed, not changed | no | no |
| (c) bounds and bases read again a phase | member step −0.7 to +0.6 %, frame kernel −0.8 to +0.3 %: nothing | yes, bit for bit | no |

**How it was measured.** Temporary perf bits in a build of its own, A/B through the matrix
(`run.sh perf=BIT`), each build's binary kept apart so that nothing compiled while a
matrix ran. Two kinds of time-only bit:
- *leave the work out* (results wrong): what the perf doc's ablations do. It is a bound
  only while the scene stays the same. Leaving out the nodes' force sums took the
  head-on's contact forces away and the row came out −19 %: that is another scene, not a
  bound.
- *do the work once more* (results the same): where the work is idempotent, the kernel
  repeats it and the difference is its cost. Every state word stays the same, so both
  halves run the same physics, and the matrix prints its usual verdict. Used for (a) and
  for the walk of (b).

The two measurement builds are commits c2b8b478 and 05b15b41 on
`claude/port-gpu-floor-lessons-1289bd`, each reverted by the commit after it.

**Baseline** (`run.sh --baseline`, ms a frame of 34 steps, two runs in a row):

| row | run 1 | run 2 | | row | run 1 | run 2 |
|---|---|---|---|---|---|---|
| drive | 3.80 | 3.29 | | headon | 14.00 | 13.82 |
| tumble | 3.55 | 3.48 | | side | 9.30 | 9.85 |
| wall | 4.75 | 4.68 | | bricks | 4.03 | 4.05 |
| pile2 | 8.64 | 8.65 | | rocks | 4.39 | 4.33 |
| pile8 | 14.99 | 14.60 | | barrier | 7.51 | 7.73 |
| pile32 | 39.10 | 38.54 | | truss | 1.81 | 1.80 |

A one-car row's level moves between about 3.2 to 3.5 and 3.9 to 4.1 ms from one process
to the next with the same binary (drive, tumble, bricks; same-against-same runs showed
both levels). The two halves of one process sit at the same level, so the matrix's A/B
holds; a one-car row's baseline needs several runs.

**(a) Contact words only for nodes a contact wrote.** On the wide passes and in the
member step every node, every substep, rolls its anti-clip memory, clears its two-body
count, takes its position steps and reads and clears its force sums: 8 atomic loads and
11 stores (`nodeUpdateShapes`, `nodeUpdateCoupled`). Two measurements:

- *the rolls, the count and the position steps left out* (5 loads, 8 stores; perf bit
  2^27 of 05b15b41): −1.7 to +1.2 % on every row, the size of the matrix's noise. (The
  first run's −11 % on the drive row was the level change above: three more runs gave
  −0.2, −0.8 and +0.3 %.)
- *every contact word read and written once more*, each store the value already there
  (`contactWordsAgain`, bit 2^28; bit for bit on all 20 fixtures and all 12 rows). What
  it adds is what the node phase's whole read-and-clear costs:

| row | once more, % (runs) | ms a frame |
|---|---|---|
| drive | +0.4, +0.8, −0.6, +0.3 | 0.01 |
| tumble | +1.4, +1.8, +1.5 | 0.06 |
| wall | +1.1, +1.4 | 0.06 |
| pile2 | +1.4, +1.7, +1.0, +1.3, +1.6 | 0.12 |
| pile8 | +0.6, +1.3 | 0.15 |
| pile32 | +0.5, +0.4 | 0.17 |
| headon | +0.8, +0.7, −0.1, +1.0, +1.4 | 0.11 |
| side | +1.8, +2.7, +0.6, +1.2, +1.4 | 0.14 |
| bricks | +2.9, +2.9, −2.3, +1.7, +0.6, +1.2, +1.8 | 0.07 |
| rocks | +2.8, +0.2, −0.1, +2.3 | 0.06 |
| barrier | +2.0, +2.1, +1.7, +2.1, +2.3 | 0.16 |
| truss | +1.3, −0.1, −1.3, −0.5 | 0 |

So the bound is 0.01 to 0.17 ms a frame, 1 to 2 % of most rows. A switch could keep less
than that: it adds a flag word a node that the node phase loads and every writer sets,
and a node written in the last two steps still pays in full (the anti-clip roll moves
last step's recorded words to the held ones, so a node written at step s resets its held
words at s + 2). The review's estimate was "at most 0.2 ms of the 0.85"; measured, it is
under that. The experiment's own kernel paid 1 to 2 ms a frame for the same words
(−8 % on its head-on and −11 % on its wall crash when it stopped); production's node
phase pays 0.05 to 0.16. One car's 548 nodes are 18 warps of 19 accesses, 11,600 warp
accesses a frame: 0.06 to 0.12 ms is 5 to 10 ns each, inside the experiment's own 4 to
29 ns for a warp's access to device memory.

*Not built.* What it would take, for whoever returns to it: one word a node in `gAcc`
(bits: force sums written, anti-clip recorded, anti-clip held, count, position steps),
set inside `accAdd` (the hits', the levels', surface contact's and the wheels'),
`clipRecord`, `shiftAdd` and the count's adds, and by the adds to the force sums that do
not go through `accAdd` (`explicit_beams.slang` near 6278, 6414 and 8296); the node phase
loads it once and, when it is zero, touches nothing. The clear state of every word is
then what it is today, so every word stays bit for bit. The bound does not shrink with
the frame: at the one-car target of 1 ms a frame the same 0.06 ms is 6 %, and that is
when to build it.

**(b) Loops where each lane's turn is a different iteration.**

- *The proxies' integration* (`cplIntegrateBody`, `cplIntegrateBodySplit`). The plan's
  shape range a body is set on the lab's path: the fallback to every shape
  (`gsCpl[0] & 4`) is the tests' cache-off bit (`set_coupling_workgroup_cache`), off by
  default. A lossless A/B that forces every shape (bit 2^29 of 05b15b41): wall −0.2 %,
  bricks +0.5 %, rocks +0.2 %, all bit for bit. So even the whole scan costs nothing
  here: the shapes' reactions are already a lane a shape (`cplReactBy`) and the body's
  loop reads workgroup memory. The experiment's 1.0 → 0.09 ms was its own loop;
  production does not have it.
- *The pair list's walk.* On the lab's path (wide passes with the reused list, self
  split) the list is built by `selfWalkList` in the walk dispatch
  (`explicit_beams_self_walk`: 64-thread workgroups, 32 shares a triangle), the same
  loop as `triangleGridWalk`'s and `pairReuseList`'s with the append inside the bucket
  loop. It runs on building steps only: 9, 16 and 29 of a frame's 34 dispatches at
  pile2, pile8 and pile32; 3 at the head-on. The whole walk once more with every append
  an atomic that changes nothing (bit 2^29 of c2b8b478, bit for bit on all 12 rows), two
  runs: pile2 +1.7 and +3.5 %, pile8 +2.4 and 0.0 %, pile32 +2.9 and +3.0 %, every other
  row between −1.1 and +3.0 % and none above +2 % in both runs. GPU timestamps agree
  (`PROFILE_SPLIT=1`): the walk dispatch is 0.24, 0.62 and 1.82 ms a frame at pile2,
  pile8 and pile32 (2.1, 3.8 and 4.7 % of the timed span), under 0.1 ms elsewhere.
  Moving the appends after the loop cannot gain more than a part of that. Not changed.
- *`triangleGridWalk`'s `triPairAdd`, `pairReuseList`, `trianglePairs`.* None runs in a
  saved state. The first lists only with the reused pair list off
  (`RUBBLE_PAIR_REUSE=0`); the second builds the list inside a workgroup, which the
  lab's self split replaces with the walk dispatch above; the third is the frame
  kernel's early list on a step with few listed triangles, and the matrix's rows do not
  run the frame kernel (one car runs the wide passes, cars in contact the member step).
  Not timed, not changed.
- *Surface contact's `surfPairList` and static-slot loops.* Surface contact is off in
  every matrix scene and every fast-check fixture (`--surface 1` is in none). Left for a
  task that first saves a state with it on.

**(c) Bounds and bases read again at a phase's start.** Built as the review describes
(c2b8b478): `memberNodesLocal` and `memberBeamsLocal` for `contactMemberStep`,
`frameNodesLocal` and `frameBeamsLocal` for the frame kernel, each its own function
behind a perf bit (2^27 the node phase, 2^28 the beam phase). Each loads the vehicle's
four records, its origin and its step again through `v + phaseZero()`, where `phaseZero`
is a workgroup atomic's top bit (`gsTestedCount[0].load() >> 31`, always 0), so no
compiler can fold the copy into the entry's.

- *Does it survive the toolchain?* Yes. The WGSL holds the call and the separate loads
  (`phaseZero_0()` four times in the member module), Chrome compiles all 124 modules
  (`EXPLICIT_WGSL_MODULES_ONLY=1`), and it is bit for bit on all 20 fixtures and all 12
  rows.
- *Does it pay?* No.

| row (cars in contact: the member step) | node phase | beam phase | both |
|---|---|---|---|
| pile2 | +0.4 % | −0.0 % | +0.4 % |
| pile8 | +0.6 % | +0.2 % | +0.5 % |
| pile32 | +0.4 % | −0.1 % | −0.2 % |
| headon | −0.7 % | +0.5 % | −0.0 % |
| side | −0.3 % | +0.0 % | −0.7 % |
| barrier | −0.6 % | +0.0 % | −0.1 % |

| the frame kernel, one car (`--wide 0`), on / off ms a frame, 3 rounds | t 2.5 | t 6 | t 20 |
|---|---|---|---|
| node phase | 4.66 / 4.66 | 6.33 / 6.31 | 6.27 / 6.27 |
| beam phase | 4.65 / 4.67 | 6.29 / 6.31 | 6.25 / 6.25 |
| both | 4.67 / 4.67 | 6.29 / 6.29 | 6.19 / 6.24 |

  (The one-car rows of the matrix run the wide passes, which hold neither function;
  their deltas in those runs, −9.9 to +4.9 % on a different row each time, are the
  level changes described under the baseline.)
- *Why not, as far as it was looked at.* A measurement, then a guess. Measured: the
  member step is 5.0 to 5.9 ms a frame at two to eight cars (its node phase 1.3 and its
  beam phase 0.75 of that, §8.36), and neither moved. The guess: a lane here takes one
  or two nodes and eight beams a phase, each hundreds of instructions and tens of device
  loads, so a bound or a base loaded from the stack once a turn is too small to see. The
  experiment's kernel is 1.1 to 1.3 ms for the whole frame with its counts known at
  compile time; the same load is a larger share there. Not kept: a switch that changes
  nothing is only more code in the largest kernels.

**What else the runs showed** (`PROFILE_SPLIT=1`, GPU ms a frame; for ranking):

| row | member step | members' hits | split levels | fleet walk | self walk + pairs |
|---|---|---|---|---|---|
| pile2 (8.6) | 5.76 | 1.65 | 0.53 | 0.04 | 0.77 |
| pile8 (15.4) | 5.94 | 1.74 | 1.14 | 1.45 | 1.49 |
| pile32 (39.3) | 7.15 | 3.00 | 14.10 | 3.68 | 4.37 |
| headon (14.0) | 5.39 | 1.85 | 4.80 | 0.73 | 0.66 |
| side (9.9) | 5.01 | 1.71 | 0.71 | 0.15 | 0.63 |

The member step is a third to a half of every row with cars in contact, and the levels
are a third of the head-on and of pile32. The three changes above act on parts of the
node phase and of the walk that are a few per cent of that. The gap to the experiment
(1.7 to 3.9 times) is in the schedule and the layout, the review's §6.2 to §6.4, not in
these.

**Defaults.** Nothing to decide: no switch exists. Recommendation: keep (a) as a note
for when one car's frame is near 1 ms; drop (b) for the lab's path and take the surface
loops with the surface pass's own item (the review's §6.1 d); drop (c).

**To run again.**
```sh
cd engine
git checkout 05b15b41 -- slang/kernels && tools/scenario-matrix/run.sh perf=268435456    # (a) once more
tools/scenario-matrix/run.sh perf=134217728                                               # (a) the rolls left out
tools/scenario-matrix/run.sh perf=536870912 wall bricks rocks                             # (b) every shape
git checkout c2b8b478 -- slang/kernels && tools/scenario-matrix/run.sh perf=402653184     # (c) both phases
tools/scenario-matrix/run.sh perf=536870912                                               # (b) the walk once more
EXTRA="--wide 0" FRAMES=10 AB_ROUNDS=3 AB_WARM_MS=150 tools/explicit-2khz/fast-check.sh perf=402653184 cars1   # (c) the frame kernel
git checkout HEAD -- slang/kernels
```

### 8.48 What the native toolchain adds to every kernel: wgpu's runtime checks, 5 to 12 % of every row (2026-11-21)

Native M3 Max, main's kernels (96904a44), every GPU run under
`with-gpu-lock.py`.

**A correction to §8.47 first.** §8.47 ends by saying the gap to the gpu-floor
experiment "is in the schedule and the layout", the review's §6.2 to §6.4. Two of those
were already measured on production, on branches the review does not cite:

- the experiment's layout as a Slang kernel (#139, not merged: its own §8.43 on `perf/solo-car-kernel`): bit for bit with today's
  frame kernel and the CPU, and slower (2.37 ms against 1.33 natively, 2.04 against
  1.58 in Chrome);
- two bodies in lockstep across workgroups (§8.44): not safe in WebGPU, which
  promises neither forward progress across workgroups nor a device-scope fence.

So the same layout is fast as hand-written Metal and slow through Slang, WGSL and naga.
This section measures what that path itself adds.

**What wgpu adds to a kernel on Metal** (`wgpu-hal` 30, `naga::back::msl`):

| check | what is emitted | where |
|---|---|---|
| bounds checks | every index of an array or a buffer clamped to the last element (a buffer's length read from a sizes uniform) | every load and store |
| forced loop bound | a two-word counter a loop, compared and decremented every turn | every loop, nested ones too |
| integer-division guard | a divisor of zero replaced | every `/` and `%` on integers |
| workgroup memory zeroed | one lane zeroes it at a workgroup's start | already off on the wide and split paths (§8.8, `contact-groups-design.md` §4); still on for the frame kernel, where the dispatch alone is 0.47 ms (§8.44) |

The engine's kernels cap their own loops (`explicit_loop_cap.slang`) and index within
their own counts, so the first three change no result. They had been timed once, for
compile time on a T4 (`kernel-compile-time.md`: no change); never for run time.

**The measurement.** Every explicit module now goes through one helper
(`engine_webgpu::explicit_shader_module`) that can compile it without a mask of the
checks (1 bounds, 2 loop bound, 4 integer division; `RUBBLE_SHADER_CHECKS_OFF`, or
`set_explicit_shader_checks_off`; at the time of the measurement 0, every check, by
default). `vehicle_lab --ab-switch
checks=MASK` builds a second GPU object with the mask, restores both from one snapshot,
runs them in turn and compares every state word, so the scenario matrix times it as it
does a perf flag. Control (`checks=0`, checked against checked, two objects): bit for
bit, pile8 +0.8 %, headon +1.0 %.

ms a frame, on against off within one process (3 rounds of 10 frames each way):

| row | checked | all three out | delta | bounds out alone | loop bound out alone |
|---|---|---|---|---|---|
| drive | 3.87 | 3.46 | −10.6 % | −10.3 % | −5.9 % |
| tumble | 3.71 | 3.35 | −9.7 % | −9.3 % | −1.5 % |
| wall | 4.12 | 3.81 | −7.7 % | −8.0 % | −3.2 % |
| pile2 | 8.72 | 7.89 | −9.5 % | −3.5 % | −4.2 % |
| pile8 | 15.45 | 14.03 | −9.2 % | −3.5 % | −6.4 % |
| pile32 | 37.96 | 34.15 | −10.1 % | −5.2 % | −9.1 % |
| headon | 13.15 | 12.08 | −8.1 % | −4.0 % | −3.1 % |
| side | 9.74 | 9.09 | −6.7 % | −3.6 % | −2.5 % |
| bricks | 3.96 | 3.57 | −9.8 % | −4.4 % | −4.1 % |
| rocks | 4.34 | 3.82 | −12.1 % | −5.8 % | −1.9 % |
| barrier | 7.61 | 7.05 | −7.3 % | −4.0 % | −3.2 % |
| truss | 1.78 | 1.69 | −5.3 % | −4.4 % | −9.6 % |

Every row bit for bit in all three runs; the matrix's verdict is PASS for each. The two
parts roughly add. The integer-division guard was not timed alone.

**The rest of the path costs little.** `engine-metal` compiles the same Slang straight
to Metal, with no WGSL, no naga and none of the checks. `explicit_bench` (the plan's
default schedule, the frame kernel; a procedural sedan of 367 nodes and 2,170 beams,
parked; a frame submitted and waited for, median of 120):

| | one car, wall ms | 16 cars, wall ms |
|---|---|---|
| wgpu, checked (today) | 1.44 | 1.21 |
| wgpu, the three checks out | 1.29 | 1.20 |
| Metal direct | 0.95 to 1.15 (GPU time 0.86) | 0.85 to 0.92 (GPU time 0.71) |

What is left between wgpu without the checks and Metal direct on this path is about
the frame kernel's workgroup zeroing (0.47 ms a dispatch, one dispatch a frame) and the
two hosts' own submission. So Slang through WGSL and naga does not produce slower code
than Slang straight to Metal: the injected checks and the zeroing are the difference.
(This benchmark's wgpu column is wall time only: the default device has no timestamp
queries.)

**Proofs, with all three checks out.**
- `fast-check.sh checks=7`: all 20 fixtures bit for bit (1, 2, 4, 8 and 32 cars).
- The scenario matrix: 12 rows bit for bit, PASS (the table).
- The suites that hold the GPU to the CPU reference and the paths to each other
  (`RUBBLE_SHADER_CHECKS_OFF=7 perf-verify.sh default`): all 11 pass (the loop-cap
  census 5, explicit_beams_bitwise 12, explicit_contact_gpu 13, explicit_pair_reuse_gpu
  2, explicit_beam_order_gpu 3, debris_retire_gpu 1, showcase_car_gpu 6, vv_grid_gpu 2,
  explicit_vehicles 11, vehicle_lab_reset 3, vehicle_lab_piece_windows 1).
- Not run at the time: `engine-metal` parity, the 40 s whole runs, and any kernel outside
  the explicit vehicles (the rigid solver's modules are created in `gpu.rs` and still
  carry every check).
- Chrome: nothing changes there. A page cannot turn Tint's own checks off, so this is a
  native gain only (native hosts, and what the matrix and every native figure in this
  doc measure).

**A harness bug found on the way.** The first run failed one fixture (one car, t 20).
The control failed the same way: the second GPU object had not been given that
fixture's retired nodes. Fixed in the lab's A/B (`fixture_host`); the kernel A/B
(`--ab-switch kernel`) has the same gap on a replayed fixture with retired nodes and is
not changed here.

**What this does and does not explain.** It is 5 to 12 % of every native row, on every
kernel, with no kernel changed. It is not the 2 to 4 times between production and the
experiment: #139's layout was 1.8 times slower than today's frame kernel with the
checks in both. Whether the checks weigh more on that layout (its hot loops index
workgroup arrays every turn) was not measured: #139's kernel is not on this branch.

**Default: the bounds checks and the division guards out on native Metal, the loop
bound kept** (mask 5; the owner's decision for the checks, 2026-11-21;
`SHADER_CHECKS_OFF_METAL`, `explicit_shader_checks_off`). On `main`'s kernels all three
were out (mask 7) and every proof below this paragraph is that build's. After the merge
with the integration branch (#145) the 32-car pile does not repeat between two GPU
objects when the bounds checks and the loop bound are both out (`known-bugs.md` F9, open;
the other 11 rows are bit for bit at 7), so the loop bound stays in until that is found.
With mask 5 on the merged code, against the checked build:

| row | checked | mask 5 | delta |
|---|---|---|---|
| drive | 4.02 | 3.64 | −9.3 % |
| tumble | 3.84 | 3.57 | −7.0 % |
| wall | 4.56 | 4.24 | −7.0 % |
| pile2 | 8.92 | 8.22 | −7.8 % |
| pile8 | 13.72 | 12.89 | −6.1 % |
| pile32 | 38.52 | 36.42 | −5.5 % |
| headon | 13.34 | 12.80 | −4.1 % |
| side | 9.59 | 9.30 | −3.0 % |
| bricks | 4.03 | 3.70 | −8.1 % |
| rocks | 4.43 | 4.09 | −7.7 % |
| barrier | 7.74 | 7.44 | −3.8 % |
| truss | 1.89 | 1.79 | −5.2 % |

12 rows bit for bit, PASS (the loop bound is worth another 2 to 6 % on the piles, the
first table's last column). Also with mask 5 on the merged code: pile32 with 5 on
both sides and 5 against checked, 3 runs each, bit for bit; the 20 fixtures bit for bit;
`perf-verify.sh default`'s 11 suites; `engine-metal` `vehicle_parity` 4; Chrome's
validation of the 128 modules, strict and fast (unchanged WGSL); the 11 rigid scenes'
hashes the same as the fully checked build's. The whole 40 s runs were not repeated. The checked build is the
debugging mode (`RUBBLE_SHADER_CHECKS_OFF=0`): without the checks an index out of range
reads or writes other memory where the checked build clamped it to the last element,
which hid the bug as well. Scope, each a measured limit and not a preference:
- *Metal only.* The device's backend decides. Every figure and proof here is Metal's;
  Vulkan and DX12 keep every check until they are measured and proven the same way
  (`RUBBLE_SHADER_CHECKS_OFF=7` turns them off there for that measurement).
- *The explicit vehicles' kernels and the World's coupling only.* The rigid solver's
  modules are not changed by this default (§8.49 is theirs).
- *Proven in the bit-exact mode.* The kernels' strict helpers hold their rounding with
  and without the checks. Under fast arithmetic (#135's setting) a change of generated
  code can move a rounding (`gpu-determinism-wgsl`), so there the checked and unchecked
  builds are each valid and need not agree bit for bit.

With the default in (the lab's checked half built with every check, `checks=7`): the
matrix again, 12 rows bit for bit, PASS, −5.4 to −14.9 % (drive 3.18 against 3.73, pile8
13.41 against 14.96, pile32 34.57 against 38.85, truss 1.69 against 1.79); the 20
fixtures bit for bit; the 11 suites of `perf-verify.sh default` with no variable set
pass; `engine-metal` `explicit_parity` 7, `vehicle_parity` 4, `explicit_drive_parity` 1
pass.

*Whole runs* (the mountain, 40 s, `--rockslide 0 --self-split 1 --timing step`; the
vehicles' GPU ms a World step from t 0.5 s, median / p95; the fully checked build is
`RUBBLE_SHADER_CHECKS_OFF=0 RUBBLE_RIGID_SHADER_CHECKS_OFF=0`, the default is this
section's and §8.49's; the two in turn):

| | fully checked | the default |
|---|---|---|
| 8 cars, 2 rounds | 17.30 / 21.66, 17.27 / 21.69 | **15.60 / 19.97, 15.79 / 20.06** (−9 % median, −8 % p95) |
| 8 cars, the World's wall ms a step, median | 19.3, 20.1 | 17.4, 19.4 |
| 1 car, 4 rounds | 4.22 / 4.57, 4.22 / 4.66, 4.22 / 4.68, 4.23 / 4.58 | **3.78 / 4.14, 3.77 / 4.12, 3.82 / 4.32, 3.81 / 4.31** (−10 % median) |
| 1 car, the World's wall ms a step, median | 7.31, 7.16, 6.19, 5.73 | 6.72, 6.66, 5.36, 5.17 |

One car's level moves between processes (§8.47): in five other runs of the default it
was 3.74, 3.75, 4.14, 4.16 and 4.18, and in one other of the checked build 4.17. So over
whole runs the default is 10 % faster in most processes and level in the rest; within
one process (the matrix) it is faster in every row. A whole run's final state is not
compared: native wgpu is not reproducible run to run (the fixtures and the matrix hold
the bits).

**To run again** (from `engine/`):
```sh
tools/scenario-matrix/run.sh checks=7                  # all three out (the default) against checked, every row
tools/scenario-matrix/run.sh checks=1                  # bounds checks alone (2: loop bound, 4: division)
tools/scenario-matrix/run.sh checks=0                  # control: checked against checked
tools/explicit-2khz/fast-check.sh checks=7             # every fixture
tools/explicit-2khz/perf-verify.sh default             # the suites on the default (RUBBLE_SHADER_CHECKS_OFF=0: checked)
RUBBLE_BENCH_CARS=1,16 cargo run --profile fast -p engine-metal --example explicit_bench
```
The first run with a new mask compiles every pipeline again (4 to 6 minutes).

### 8.49 The same checks on the World's own kernels (the rigid solver), and the zeroing that was left (2026-11-21)

The question (§8.48's open rows): the rigid solver's modules (`gpu.rs`, 283 of the build's
409) still carried every check and zeroed their workgroup memory at every dispatch, and
five kinds of explicit pipeline still zeroed theirs. Native M3 Max. Rigid figures are
`rigid_gate` (`engine-world/examples`): ms a World step (step and wait), median of 240
steps after 60, sleep off, a process a figure, the masks in turn for 4 to 6 rounds; its
hash mode is a hash of every body at 12 checkpoints to step 300 in 11 rigid scenes (two
runs of the checked build agree at every one). `RUBBLE_RIGID_SHADER_CHECKS_OFF=MASK`: 1
bounds, 2 loop bound, 4 integer division, 8 the zeroing.

**What each check does to the rigid scenes' results** (11 scenes, sleep off, against the
checked build):

| left out | end states |
|---|---|
| nothing, run twice | the same |
| the loop bound (2) | the same in all 11 |
| the integer-division guards (4) | the same in all 11 |
| the zeroing of workgroup memory (8) | the same in all 11 |
| **the bounds checks (1)** | **differ in 10 of 11**, first at the checkpoint of step 20 (both brick rings), 30 (brick walls), 50 (pyramid, box pile, ragdoll pile), 75 (wall smash, 100k grid), 100 (chain mail), 200 (sphere pile); hinge chains the same |

So unlike the explicit kernels (strict arithmetic, every word the same without the
checks), a rigid result depends on the bounds checks. It is not a rounding of the whole
build: the first ten steps are bit for bit and the scenes part at different steps.

**Which kernels** (`RUBBLE_RIGID_BOUNDS_OFF_NAMES=a,b`: the bounds checks out of those
kernels alone; `tools/rigid-gate/bounds-search.py`'s halving over the 79 kernels the 11 scenes build):
two, **`narrowphase_fresh_copy_persistent`** (the 28k ring, from step 20) and
**`ccd_emit_copy_persistent`** (the box pile, from step 150, after the ball). Both run
the shapes' manifold code (`narrowphaseShapes`, `box_collide.slang`). With the checks
out of the other 77 the 11 scenes are bit for bit, and the step is no faster (six paired
rounds: the 110k ring +0.3 %, the 32k pile −0.7 %). So the bounds checks' 5 to 6 % of a
rigid step is in those two kernels, the ones whose result depends on them, and nothing
of it can be taken until that is resolved. Not determined: whether it is an index the
clamp makes harmless or a rounding that the compiler chooses differently around the
clamps (the kernels are not strict; `boxClipPoly` and `boxAddPt` guard their own counts,
and the scenes stay bit for bit while their boxes are axis-aligned, which points at
rounding; this code has a history on Metal, see the note above `closingPoints`).

**Time** (ms a step; the level of the machine moved between rounds, so the deltas are
each round's pair, and the later rounds are the steadier):

| | Brick Ring 110k | Box Pile 32k |
|---|---|---|
| checked, 4 rounds | 16.48 | 5.36 |
| the three checks out (7): results differ | 15.58 (−5.5 %) | 5.04 (−6.1 %) |
| the zeroing out (8) | 15.86 (−3.8 %) | 5.05 (−5.8 %) |
| all four out (15): results differ | 15.16 (−8.0 %) | 4.69 (−12.6 %) |
| the loop bound out alone (2), 4 rounds | −1.7 to −2.6 % a round | −1.0 to −3.3 % a round |
| the result-preserving three on every module (14), 6 paired rounds | −3.7 % (median of rounds) | −3.9 % |
| **the default as built, 6 paired rounds** | **16.14 → 15.18; −4.7, −4.1, −3.8 % in the three steady rounds** | **4.92 → 4.52; −7.8, −8.8, −7.3 %** |

**The default** (`rigid_shader_checks_off` in `gpu.rs`):
- *The loop bound out on native Metal*, every World kernel. It changes nothing of a loop
  that ends (the counter allows 2^64 turns), and the 11 scenes are bit for bit.
- *The zeroing out, natively, for the `solve_primal*` kernels alone.* Their workgroup
  memory is 864 floats a workgroup (`gsA`, `gsRhs`), zeroed by one lane at every
  dispatch of every colour of every iteration. Read: every value they read from it was
  written in the same dispatch before a barrier. A body's lanes are live together
  (`live` is a function of the body's slot), so in `primalReduceStep` the lane that
  reads slot `slot * s + lane` has a live partner `lane + s` that wrote it before the
  barrier; `gsUniformLanes` is written by lane 0 before each read. The other World
  kernels with workgroup memory (46 modules, most of them 64 words) keep the zeroing:
  not read one by one, and little to gain.
- *The bounds checks and the division guards stay* (results differ; unproven and
  untimed alone).
- `RUBBLE_RIGID_SHADER_CHECKS_OFF=0` is the fully checked build.

**Proofs with the default in.** The 11 scenes' hashes: the same as the checked build's
at every checkpoint. Suites (release, one at a time): engine-webgpu `ring_hold` 1,
`columns_hold` 4, `gables_hold` 1, `invariants_small_gpu` 6, `sleep_fast_gpu` 2,
`breakable_joints` 4, `pipelined_gate` 1, `broadphase_oracle` 2, `ccd_unit` 5,
`ccd_tunnel` 5, `kinematic` 3, `queries` 4; engine-world `world` 18, `contacts` 7,
`post_stabilization` 1, `explicit_vehicles` 11: all pass. (`destruction`: 7 pass, 3 fail
on "Invalid GLB header": this worktree's fracture assets are LFS pointers, not the
change.) The scenario matrix again, the vehicles' checks against checked: 12 rows bit
for bit, PASS; `run.sh --world`: PASS, the World adds 0.44 to 1.46 ms a step. The WGSL
is unchanged (compile options only), and no GPU call is added or removed. Not run: the
rigid gate's sleep-on rows and its Chrome rows (Chrome is not changed: a page cannot set
either option).

**The explicit pipelines that still zeroed** (the frame kernel, the vehicle pairs' grid,
the variants, the first split path, the World's coupling; mask bit 8 of
`RUBBLE_SHADER_CHECKS_OFF`, `checks=15:7` on the matrix): 12 rows bit for bit, −0.8 to
+1.7 %, no row moves. Nothing there on the default paths (the wide and split paths
already skip it, §8.8). Not adopted; the bit stays as a measurement.

**What is left of this lesson.** One thing: the bounds checks in the manifold code of
the narrowphase and the CCD's emit, 5 to 6 % of a rigid step on Metal (the 110k ring 16.5
→ 15.6 ms). If the dependence is an index, it is a bug to fix and the checks can go; if
it is a rounding, the two builds are each valid and taking the checks out changes the
rigid scenes' bits, which is the owner's decision (the rigid gate holds them to `main`).

**To run again** (from `engine/`):
```sh
GATE_MODE=hash GATE_SLEEP=off cargo run --profile fast -p engine-world-lab --example rigid_gate   # the default
RUBBLE_RIGID_SHADER_CHECKS_OFF=0 GATE_MODE=hash GATE_SLEEP=off cargo run --profile fast -p engine-world-lab --example rigid_gate
RUBBLE_RIGID_SHADER_CHECKS_OFF=0 GATE_MODE=time GATE_SLEEP=off GATE_SCENES="Brick Ring (110k),Box Pile (32k)" cargo run --profile fast -p engine-world-lab --example rigid_gate
tools/scenario-matrix/run.sh checks=15:7     # the explicit pipelines' zeroing (ON:OFF masks)
RUBBLE_RIGID_BOUNDS_OFF_NAMES=narrowphase_fresh_copy_persistent GATE_MODE=hash GATE_SLEEP=off GATE_SCENES="Brick Ring (28k)" cargo run --profile fast -p engine-world-lab --example rigid_gate   # differs from step 20
```

### 8.50 A module without the coupling's code for a vehicle with an empty proxy set: measured, not built (2026-11-21)

The gpu-floor review (`gpu-floor-vs-production.md`, §6.2 step 1)
proposes that a vehicle whose proxy set is empty for a coupling window be sent to a
module without the coupling's code, chosen a frame by an indirect count. Its own kernel
without that code was 0.16 ms a frame cheaper. This section measures what the same choice
gives production, on `main` at 96904a44. (It follows §8.47 and §8.48; measured on the
sibling branch `claude/jovial-moser-8fc7c2`, whose measurement build is reverted.)

**What production does today** (read, each with its place at 96904a44):

* **No module without the coupling's node contact exists.** The `_nocpl` twins
  (`EB_NO_COUPLING`, twelve frame and contact modules, `engine-webgpu/build.rs`) leave out
  only the surface pass's proxy paths, and the host picks one once an upload, for a fleet
  with surface contact and no coupling room (`explicit_beams_gpu.rs:1165` `frame_module`).
  Every other module carries `explicit_beams_rigid.slang` whole.
* **The node phase already chooses a vehicle a step, by a branch.** `nodeUpdate` calls the
  coupled function only when `cplHas(vehicle)` (`explicit_beams.slang:1182`, `:1261`), and
  `cplHas` is false for a vehicle whose record has no shape
  (`explicit_beams_rigid.slang:1506`). A vehicle with an empty proxy set runs
  `nodeUpdateShapes`, the same function a World without coupling runs.
* **On the lab's path (what the matrix times) the coupling's step is not on the vehicle's
  own workgroup.** The wide passes with the reused pair list run step s + 1's coupling on a
  workgroup of its own in the list kernel's dispatch (`wideCoupleApart`,
  `explicit_beams.slang:8537`; the host dispatches three workgroups a vehicle,
  `explicit_beams_gpu.rs:3513`). With no shape its count pass returns at its first test
  (`explicit_beams_rigid.slang:932`).
* So with an empty proxy set what is left of the coupling on that path is: in the node
  pass a header read by lane 0 and one barrier a workgroup, and two record reads a node
  (`cplHas`); on the coupling's workgroup the header, an empty table and four barriers a
  step (a vehicle with no shape does not take the folded step: `cplSplit` needs a shape);
  and the code itself, in three modules: the node pass (`explicit_beams_wide_nodes`), the
  list kernel (`explicit_beams_selfsplit_list`) and the aux pass (`explicit_beams_wide_aux`,
  two dispatches a frame).
* A World outside the lab runs the other path: `World::add_explicit_vehicles` turns the
  coupling on, which turns the wide passes off (`explicit_beams_gpu.rs:2651`; the lab
  turns them on again, `vehicle_lab/mod.rs:2719`). There the coupling's step, with its
  four barriers a substep for an empty set, is on the vehicle's own workgroup.

**The matrix's one-car rows as saved** (each vehicle's proxy bodies / shapes in the saved
state): drive 0 / 0 (flat ground is the solver's own plane); tumble 7 / 7 (the mountain's
ground is static rigid chunks; the car is in the air); wall 1 / 1 (the wall is a rigid
body, mid-crush); bricks 34 / 34; rocks 3 / 3. Only the drive row has an empty proxy set,
so only there would the review's choice act. For the other two the same comparison was
made from the saved state with the proxy set emptied (`--ab-empty-proxies`: the record's
counts zero, as the coupling's emit writes a vehicle with nothing in reach,
`vehicle_coupling.slang:937`); the car then flies on or drives through where the wall was,
the same in both halves.

**The measurement build** (commit 8417ad38, reverted by the commit after it): the three
modules compiled a second time with `EB_NO_PROXY`: `cplHas` false, `coupleBegin` without
the header, the coupling's step, tables and end empty, the node pass without its coupled
branch and its barrier, the coupling's workgroup returning at once (still dispatched).
The WGSL: node pass 100,219 → 66,026 bytes, list kernel 271,340 → 205,568, aux pass
119,007 → 62,534. A host switch swaps them in (`set_noproxy_modules`), and the lab times
both from one state in one process (`--ab-switch noproxy`). Without the define the two
kernel files are `main`'s text (`unifdef -UEB_NO_PROXY`). A second switch
(`--ab-switch cplhdr`) turns the coupling header's `on` word off in the shipped modules:
what the coupling's code executes for a vehicle without proxies, on either path.

**Results** (native Metal, M3 Max, `--profile fast`, the lab's default path: the wide
passes with the reused pair list, as the matrix runs a row). Each run is one process: the
two halves in turn from the saved state, 11 rounds of 10 frames, each frame's best round,
their mean. Eight runs a line, the comparisons taken in turn; the control is the same
chain with `perf=0` (both halves the same code). ms a frame of 34 steps:

| row, proxy set | shipped | without the coupling's code | change (median of the runs' own) | range of the runs | control | state |
|---|---|---|---|---|---|---|
| drive, empty as saved | 3.303 | 3.232 | **−0.062 ms, −1.9 %** | −2.7 to +0.6 % | +0.1 % | bit for bit, 8 of 8 |
| tumble, emptied | 2.963 | 2.883 | **−0.069 ms, −2.4 %** | −2.7 to +1.1 % | −0.1 % | bit for bit, 8 of 8 |
| wall, emptied | 3.912 | 3.847 | **−0.063 ms, −1.6 %** | −2.1 to −1.1 % | +0.2 % | bit for bit, 8 of 8 |
| tumble as saved, 7 static shapes, none touched (not step 1) | 3.322 | 2.893 | −0.363 ms, −11.2 % | −15.9 to −8.4 % | 0.0 % | bit for bit, 8 of 8 |

A first chain of five runs a line (3 rounds a process) agrees: pooled over the thirteen,
−1.8, −2.4, −1.7 and −11.1 %. The wall, bricks and rocks rows as saved differ with the
modules swapped, as they must: their proxies are being touched or are in free flight.

What the shipped modules execute for a vehicle without proxies (`--ab-switch cplhdr`, the
header's `on` word on against off, five runs a line, 3 rounds; every run bit for bit):

| path | row | on | off | the coupling's executed work |
|---|---|---|---|---|
| the lab's (wide passes) | drive | 3.726 | 3.769 | +0.007 ms, +0.2 % (−1.8 to +2.3) |
| a workgroup a vehicle (`--wide 0`) | drive | 4.479 | 4.324 | **+0.157 ms, +3.6 %** (3.3 to 3.8) |
| | tumble, emptied | 4.121 | 3.965 | **+0.167 ms, +4.4 %** (2.8 to 4.6) |
| | wall, emptied | 5.636 | 5.455 | **+0.220 ms, +4.1 %** (2.8 to 7.2) |

Where the lab path's gain is, by module (`RUBBLE_NOPROXY_PARTS`, five runs, 3 rounds):
on the drive row the node pass alone −1.3 % (−2.9 to +12.5, one run an outlier) and the
list kernel alone +0.1 %; on the tumble row as saved the node pass −7.9 % (−9.3 to −6.8),
the list kernel −2.3 % (−2.5 to −1.2), the aux pass −0.7 %.

**Reading.**

* **The choice is exact with no guard, as the review argued.** Every run with an empty
  proxy set left the same state words and coupling records with the coupling's code out
  (39 runs over three rows) or its header off (20 runs).
* **On the path the matrix times it is worth 0.06 to 0.07 ms a frame, 1.6 to 2.4 %.**
  Nothing of that is executed work (the header's word: +0.2 %): the node phase already
  takes the uncoupled function for such a vehicle and the coupling's step sits on a
  workgroup beside the list kernel. What is left is the node pass's barrier and the size
  of its code.
* **The review's 0.16 ms is production's too, on the other path.** Where the coupling's
  step is on the vehicle's own workgroup an empty set costs 0.16 to 0.22 ms a frame, 3.6
  to 4.4 % of that path's 4.0 to 5.5 ms (by the code, the step's four barriers a substep
  and the node pass's test; not timed apart). The experiment's kernel has that layout, which is why it saw the cost.
* **The row where the coupling costs most is not step 1's.** A car in the air over the
  mountain has seven ground chunks in its proxy set and pays 0.36 ms a frame (11 %) for
  them without touching one, most of it in the node pass. On the mountain no car's set is
  ever empty, so the choice by an empty set never acts there.

**Decision: nothing is built.** The stop rule was 5 % of a frame on a one-car row; the
choice gives 1.6 to 2.4 % on the matrix's path and at most 4.4 % as a bound on the other.
Two more reasons it would not have been built as the review describes it:

* On the wide path a module a vehicle does not exist for the node pass: one dispatch runs
  a thread a node over the whole fleet. A module there is chosen a fleet (every vehicle's
  set empty), or the pass is split by vehicle; the choice a vehicle is the branch it
  already has.
* It costs four more modules for a browser to compile (66, 63 and twice 206 kB of WGSL)
  and a second set of pipelines.

`main`'s kernels and defaults are unchanged; no switch was added.

**Left for a decision, neither built nor verified beyond the runs above:**

1. *The workgroup path's 0.16 ms without a new module.* The header-off half above is the
   whole change for a vehicle alone on its workgroup: lane 0 at the frame's start leaves
   the uniform word zero when the vehicle's record has no body and no shape (as
   `cplFoldMark` sets the fold bit there), so the step returns before its barriers.
   Uniform for Chrome, bit for bit in the 15 runs above. It is under the 5 % rule today
   and would be 16 % of a 1 ms frame. A contact group's workgroup needs every member's set
   empty. It does not show on the matrix (the lab's path), so its gate needs rows on
   `--wide 0`.
2. *The untouched ground of the tumble row, 0.36 ms.* This is the review's step 2. A
   hypothesis for it that needs no new guard: production's per-node candidate lists
   already say which nodes can touch a shape this step, so a vehicle none of whose nodes
   lists a shape could take the uncoupled node function for that step. That the two
   functions agree bit for bit when no node lists a shape is shown here only for this
   state (the 13 runs above and the 5 with the node pass alone); it has to be argued from the list's rule and checked on the bricks
   and rocks rows before anything is built.

**Method notes.** The same-against-same control in the same chain is what makes a 2 %
reading usable: single runs of the control ranged −6.8 to +7.0 % with 3 rounds a process
and −3.7 to +4.6 % with 11, while the medians of eight were within ±0.2 %. The rounds are
cheap (a process is mostly its start), so `AB_ROUNDS=11` costs a second a run. Only native
Metal was measured; Chrome was not.

**To run it again** (the measurement build; from `engine/`):

```sh
git checkout 8417ad38
tools/scenario-matrix/run.sh --make drive tumble wall
ROUNDS=11 tools/scenario-matrix/run.sh noproxy drive       # the row with an empty set as saved
ROUNDS=11 tools/scenario-matrix/run.sh perf=0 drive        # its control
# A row with its set emptied, the header's word, the other path, one module alone: the
# lab as matrix.py runs it, with the option (wall adds --scenario tools/scenario-matrix/scenes/wall.json):
python3 scripts/with-gpu-lock.py --limit 300 -- env RUBBLE_GPU_LOCK=0 AB_WARM_MS=150 AB_ROUNDS=11 \
  target/fast/examples/vehicle_lab --mode gpu --measure mountain-perf --rockslide 0 --self-split 1 \
  --timing step --car ~/Development/rubble-artifacts/sedan-live/mods/sedan_new --cars 1 \
  --from-fixture out/scenario-matrix/tumble.fx --ab-frames 10 --ab-switch noproxy --ab-empty-proxies
#   --ab-switch cplhdr [--wide 0]     the shipped modules, the coupling header on against off
#   RUBBLE_NOPROXY_PARTS=1|2|4        the node pass, the list kernel or the aux pass alone
```

The lab prints each vehicle's proxy bodies and shapes at the state; a run whose state has
a touched or moving proxy reports the words that differ and fails.

### 8.51 A cold load in Chrome: the device loss at start, and the groups' kernel split by dispatch kind (2026-11-21)

**What was wrong.** On #145's preview a windowed Chrome lost its WebGPU device on the
derby's first steps ("A valid external Instance reference no longer exists"), then
refused WebGPU on every reload ("no WebGPU adapter") until it was restarted. Headless
Chrome, which the probes use, did not: it arms no GPU watchdog (`compute_spec.rs`).

**Cause.** `Contact::new` created the contact groups', solo and larger groups' frame
pipelines synchronously when the fleet was built. Their modules are the largest (about
1 MB of WGSL each). With self-collision split (`prefer_self_split`) the page compiles them
in the background after start, because the default path (the fleet-wide passes with self
split) never dispatches them. So at start-up Chrome compiled them cold on its GPU
process's main thread, for longer than the watchdog allows, and killed the GPU process.
Fast arithmetic (§8.40) roughly doubles Metal's compile times, which made it fatal.

Found by timing every synchronous pipeline creation in headless Chrome
(`run.sh --probe --preload tools/vehicle-lab/sync-pipelines.js`): the frame pipelines
were created at start, and after the fix, on the 4-car derby through its first
collision (t = 6.3 s), they are never created at all.

**Fix (lossless).** Each of the three is created at its first dispatch (`LazyPipe`):
the same pipelines whenever they are made.

**The longest compile of the load.** The next slowest kernel was the groups' pass,
`explicit_beams_split_group` (412 KB of WGSL). Cold in Chrome on an M3 Max, fast
arithmetic, one sample each:

| Kernel, one `splitGroupStep` pass alone | Cold compile |
|---|---|
| the whole kernel (every pass) | 203 to 284 s (three runs) |
| `fleetLevelsOnce` (the default levels) | 35.5 s |
| `fleetLevelsLat` | 9.3 s |
| `vvPassSmall` | 4.5 s |
| `vvPass` | 2.9 s |
| `fleetGroupLevels` | 2.3 s |
| `fleetLevelsAny`, `vvPassSmallLevels` | 1.8, 1.7 s |
| `fleetLevelsEnd` (the levels' end) | 1.3 s |
| the frame's end, the front, the defer check, the begin | 0.2 to 0.3 s each |
| **sum of the passes alone** | **59 s** |

The passes alone add up to 59 s; together in one kernel they take 203 to 284 s. Apple's
compiler slows down sharply on one giant function, so the fix is to stop compiling
passes the frame never runs together.

**The split.** Within one dispatch different groups can take different passes (a group's
size, how many hits an owner lists), so the kernels are split by the dispatch's kind,
which the host knows, not by pass. Three modules, each with one entry, beside the one
with every pass (the same Slang, the same functions on the same words):

- `explicit_beams_split_group_begin`: the frame's start;
- `explicit_beams_split_group_levels`: a step's levels under the default fleet flags
  (owners' state, front fleet-wide, groups of any size, hits by latency, levels once;
  not the levels across the GPU, no ablation 512): `fleetGroupLevels` or
  `fleetLevelsOnce`, then the frame's end;
- `explicit_beams_split_group_lv`: the levels across the GPU's end.

The schedule (`engine-exec`, so every host) picks them per dispatch
(`schedule::GroupEntry`, `group_entries_apart`, `levels_once_entry`) only where they run
what the kernel with every pass would, and counts them in readiness. The kernel with
every pass stays for every other setting; a browser compiles it in the background.

| Cold in Chrome (fast arithmetic) | Before | After |
|---|---|---|
| the groups' kernels the default path compiles before start | 203 to 284 s (one kernel) | 49.1 s (0.1 + 47.8 + 1.2 s) |

**Checked.**
- `fast-check.sh groupapart` (the switch `RUBBLE_GROUP_APART=0`, `set_group_apart`):
  every fixture, 1 to 8 cars (falling, first collisions, pile, rest), bit for bit, in
  bit-exact and in fast arithmetic. Each side made its own kernels (the "on" side the
  begin and levels kernels, the "off" side the one with every pass).
- Chrome validates all 140 modules (`explicit-wgsl-check.mjs`; its dump now takes each
  split module's entry from `plan::split_modules`, which a pipeline compile of the
  fleet, wheels and groups' modules needed).
- `perf-verify.sh default`: all 11 suites pass (GPU == CPU reference, bit-exact).
  `engine-metal` parity: vehicle, explicit, drive, loop cap and webgpu parity pass.
- On CUDA (Modal T4, `ENGINE_CUDA_FMAD=0 run.sh --cuda --base 5ab97e41 --head bf83a455`,
  two builds): all 12 matrix rows bit for bit, none slower (pile2 −3.0 %, side −1.9 %,
  headon −1.2 %, barrier −1.1 %, the rest within ±0.5 %), verdict PASS, $0.02.
- A windowed Chrome (the desktop app's browser pane) loaded the 4-car derby, compiled
  the new kernels cold, ran its first collisions and every background compile with no
  device loss. Headless probe, 20 s: real time, GPU 11.4 ms a step median, 14.8 ms worst.

**The next slowest: the members' step, not split.** `explicit_beams_selfsplit_member_vv`
(613 KB) took 118 s cold. Its dispatch kinds alone, cold (measured while other GPU
jobs ran, so slower than alone): the members' rest 1.3 s, the step 157 s, and the
step's two halves alone 48 s (a solo vehicle's frame step) and 32 s (a group member's).
Both halves run in the same dispatch, each vehicle's workgroup taking one, so splitting
them costs a second dispatch every substep (34 a frame) for good, to save a compile that
happens once and that Chrome then caches. Not done; it stays the longest kernel of a
cold load.

### 8.52 The goal's baseline: 20 cars, surface contact on, by peak and by car count (2026-11-22)

The goal: 20 soft cars on the mountain and in the derby in real time with headroom, at
full fidelity (surface contact on), judged by the **worst** step of a run, not its median.
The target is a worst step of 8.3 ms of vehicles' GPU time (half of 1/60 s).

New rows in the scenario matrix (`tools/scenario-matrix/matrix.py`, `GOAL`; `run.sh --make
goal`, `--baseline goal`), every one with surface contact on: `apartN` (N cars driving
8 m apart, never touching), `clashN` (N / 2 head-on pairs all mid-crush in the same
steps), `mountain20_tT`, `derby20_tT` (the AI driving, which the native harness now
feeds from the cars' poses), `pole`, and `wall_s`, `bricks_s`, `pile8_s` (existing rows
with surface contact). CUDA now builds the segmented frame and the surface pass's
pipelines (known bug P1's surface half), so these rows replay on Modal.

Measured at 369b6472 plus this section's commits, bit-exact on, ms a frame of 34 steps
(mean of each frame's best round; in brackets the worst frame). Mac: M3 Max, native
Metal, `run.sh --baseline`. T4: Modal, CUDA, `ENGINE_CUDA_FMAD=0 run.sh --cuda --baseline`.

| row | Mac | T4 |
|---|---|---|
| drive (1 car, no surface contact) | 3.14 (3.52) | 3.89 (7.14) |
| apart1 | 69.72 (70.33) | 31.40 (32.06) |
| apart2 | 8.33 (8.45) | 14.51 (14.76) |
| apart4 | 12.87 (14.34) | 23.92 (25.87) |
| apart8 | 13.59 (14.93) | 25.85 (27.95) |
| apart16 | 14.07 (15.41) | 28.04 (30.52) |
| apart20 | 14.30 (15.59) | 30.10 (33.74) |
| apart32 | 15.26 (16.59) | 38.01 (41.90) |
| clash2 | 15.86 (16.84) | 24.96 (29.40) |
| clash4 | 16.44 (17.75) | 28.55 (33.69) |
| clash8 | 23.41 (24.14) | 40.19 (43.53) |
| clash16 | 24.85 (25.62) | 46.04 (49.74) |
| clash20 | 25.41 (26.25) | 49.98 (54.16) |
| clash32 | 30.79 (33.37) | 61.74 (66.49) |
| mountain20_t2 | 19.32 (22.44) | 36.58 (38.54) |
| mountain20_t4 | 38.31 (41.89) | 78.10 (95.03) |
| mountain20_t7 | 37.89 (39.66) | 75.87 (77.69) |
| mountain20_t12 | 41.40 (46.03) | 86.15 (91.96) |
| mountain20_t20 | 68.11 (70.38) | not run |
| derby20_t2 / t5 / t10 (AI driving) | 23.03 / 30.61 / 29.32 (24.53 / 33.62 / 31.20) | not run with the AI |
| pole | 82.64 (99.09) | 48.36 (78.64) |
| wall_s | 73.73 (79.23) | 34.37 (35.21) |
| bricks_s | 78.91 (82.42) | 48.12 (49.84) |
| pile8 / pile8_s | 13.13 / 25.34 | 20.87 / 48.28 |
| pile32 (no surface contact) | 36.15 (40.29) | 75.74 (80.58) |

Whole runs on the Mac, 20 s, 20 cars, surface contact on (`vehicle_lab --measure
mountain-perf --scenario ... --secs 20`, GPU ms a World step):

| scene | vehicles: median / p95 / worst | whole step: median / p95 / worst |
|---|---|---|
| mountain, ledges | 40.4 / 66.0 / 71.8 (at t 19.6; it grows with the pile) | 43.2 / 69.6 / 98.9 (t 14.6) |
| derby | 29.6 / 32.5 / 35.3 | 33.5 / 40.4 / 383.0 (t 1.95) |

What the baseline says:

1. **The explicit frame is the cost.** The rigid step is 0.4 to 1.2 ms and the coupling's
   passes 0.1 ms of a 30 to 70 ms step.
2. **One car alone with surface contact on is the slowest row: 70 to 83 ms on the Mac.**
   A fleet with surface contact and no contact plan runs the frame in segments
   (`Kernel::FrameSeg`) with the surface pass's nine pipelines between them, 341 launches
   a frame. On the T4 the segments are 28.0 of 31.7 ms (0.8 ms each); the nine surface
   pipelines together are 3.6 ms.
3. **Cars that touch nothing cost 14 ms on the Mac** (apart2 to apart32: 8.3 to 15.3). The
   count scales almost flat past four cars, so the GPU does take cars side by side; the
   floor is the cost. On the T4, apart20 is one `Solo` launch of 28.8 ms: every car alone
   in its workgroup for 34 steps, 0.85 ms a step, with self-collision and the surface
   pass inside the workgroup. Without surface contact one car is 3.1 ms on the wide
   passes: `has_surface()` refuses the wide list, self-collision split and self split
   with vehicle contact (`explicit_plan/schedule.rs`).
4. **In contact, the members' step is 64 to 74 % of the frame** (T4: clash20 36.3 of 49.8
   ms, mountain20_t12 55.1 of 85.9), then vehicle contact's tests (5.3; 22.5 on the
   mountain) and the groups' levels (5.9). Surface contact doubles a pile (pile8 13.1 →
   25.3 on the Mac) for the same reason as in 3.
5. **A peak that is not physics:** the derby's whole step is 383 ms once, at t 1.95 s,
   while its vehicles' time stays under 36 ms. Its cause is not yet identified (the
   first step that needs a pipeline not used before is the candidate).

### 8.53 Surface contact on the fast schedules: the pass as pipelines after each step's hits (2026-11-22)

§8.52's baseline says a fleet with surface contact pays for the schedule it is refused,
not for the pass. This section lets such a fleet run the same schedules as one without:
the surface pass runs as its own pipelines (`explicit_beams_surface_pipes.slang`,
`SURF_PASSES`: the prep, each kind's find, each kind's apply, a workgroup a vehicle)
at the point of each step where the frame kernel runs `surfaceStep`.
`FramePlanner::set_surface_fast` (`--ab-switch surffast` in `vehicle_lab` and in
engine-cuda's `explicit_replay`; `RUBBLE_SURFACE_FAST=0|1`).

**What the pass reads and writes.** All of it is device memory; the pass keeps nothing
in workgroup memory but uniform flags (`explicit_beams_surface.slang`, its header).

| | reads | writes |
|---|---|---|
| prep (the snapshot) | each node's force slot and its beams' forces of this step's parity (`beamSum`), its accumulator (`accForce`), the groups broken this step | `gBeams[surfSnap]` a node, `gAcc[surfSplit]` a node (cleared), a proxy body's hit count and this step's earliest cast (cleared); with a fast proxy the casts' times; on the pair path each shape's bounds and the listed pairs (`gNext`), their count; its decisions in `gMisc[segRec(v)]` |
| find | positions, the static slots, the proxies' bodies and shapes (`gMisc`), the spheres the coupling's step wrote (`gNext`), the listed pairs | each node's hit count, each body's, the elements' slot masks (`gAcc[surfMark]`), the pair path's hits |
| apply | the same, the snapshot and the counts, velocities and masses, the bodies' velocities | the nodes' accumulators and contact-force readout, the position steps (`shiftAt`), a proxy shape's reaction sums of this step (`cplAcc`) |

What it needs current when it starts: the step's beam forces and group breaks (after
the beam phase), the accumulators after the step's self-collision hits (`selfApply`),
the proxies' poses and spheres as this step's coupling integrated them. What must not
have run yet: anything of the next step that reads what it writes, which is the next
node phase (accumulators, position steps), the next coupling step (the shapes' reaction
sums, the bodies' poses: it integrates them) and the next step's tyres (their hubs'
accumulators). The pass writes no position and no velocity, so vehicle sleep's motion
mark, the pair lists' tracking, the kept candidates' tracking and a wheel's rays and bars
(which read nodes, static shapes and the drive's records) are indifferent to which side
of it they run on.

**The hit flag.** The frame kernel's node phase skips a node's accumulators when no hit
was possible since the last one (`gsHit`: set by a self-collision hit, the wheels, the
surface pass for a vehicle with the flag). The pipelines carry it in the vehicle's
segment record (`segRec(v)`, `.x`): the dispatch before them saves the flag, the prep
writes 1 for a vehicle with surface contact (or leaves it), the next step's dispatch
loads it. The record is the wide passes' rebase shift, which those passes read only at
the frame's start (`wide_begin`, `wide_rebase`); a wide frame's node pass takes every
node's accumulators whatever the flag, so there the record is the prep's alone and
needs no place of its own.

**Each schedule.**

| schedule | where the pass goes | before it | after it |
|---|---|---|---|
| The wide passes, with or without the reused pair list (a lone car, or vehicle contact off) | after each step's `wide_hits` (a workgroup a vehicle: the deferred breaks, `selfApply`) | nothing: every value is in device memory, the step is `gMisc[v].w + pad2` until `wide_close` | nothing: the next step's aux pass (the coupling's step, the wheels) and node pass follow |
| … with the fused aux work (`PERF_WIDE_FUSED`, and with it `PERF_AHEAD_APART`, `PERF_TYRE_AHEAD`, `PERF_TYRE_EARLY`, `PERF_WHEELS_APART`) | **cannot host it as built**: step s + 1's coupling step runs on step s's list kernel and its tyres at the end of step s's hits kernel, both before the pass could run. The coupling's step integrates the proxies the pass must meet at step s's pose and reads the shapes' reaction sums the pass adds to; the tyres read their hubs' accumulators, which the pass adds to | | a surface fleet's wide frames run the aux pass every step (`Switches::wide_fused_on`) |
| Self split with vehicle contact, a solo vehicle (`explicit_beams_selfsplit_member_vv`: its step cut at its tests; the walk and the pairs; step s's rest at the start of step s + 1's dispatch) | between the rest's `selfApply` and its slide nodes, which are in one dispatch | a dispatch of its own for the rest's first half (`explicit_beams_solo_rest`, a small module: the walk's list recorded, the hit flag, `selfApply` or its local form; the flag and the list's state saved as the step saves them) | step s + 1's dispatch skips that half (`surfPiped`) and runs the slide nodes, the closing barrier and the step |
| … a group's member (its step, the walk, the pairs, its rest `memberRest`, then the groups' pass) | after the members' rest, before the groups' pass: where `contactGroup` and `splitMember` run `surfaceStep`, after the member's step (its slide nodes included) | nothing (the member's node pass takes its accumulators whatever the flag) | nothing |
| The split group path without self split (`explicit_beams_split_member`) | unchanged: that module holds the pass (it is not built without it) | | |
| Self split without vehicle contact (several cars with contact off, or a fleet the wide passes do not take) | the same cut as the solo vehicle above | **left gated** (`self_split_on`): the kernel's rest entry serves it, the hosts' launch of it without vehicle contact is not built. No goal or default row takes this path; such a fleet keeps the frame in segments | |

Vehicle sleep: the pipelines skip a sleeping vehicle themselves (`surfPipeSkip`), are
dispatched over every vehicle and bind the step's own parameter copy; the solo rest is
dispatched as the vehicles' step is (awake-indexed when the list is on).

**The pipelines' own cost, found on the way.** Two things made the pass as pipelines
dearer than the pass in a kernel, and both are fixed here for every path that runs them.

1. *naga's zeroing of workgroup memory* (Metal, natively). The pipelines hold the
   coupling's workgroup copies (2,080 words). naga's MSL zeroes them from one lane at
   every workgroup's start: 0.16 to 0.19 ms a dispatch on the M3 Max, nine dispatches a
   step. Of the one-car segmented frame's 69.5 ms, 52 ms were this (each dispatch timed
   apart: the nine pipelines 5.5 to 6.4 ms a frame each, the segments 19.2). Every
   workgroup value the pipelines read is written in the same dispatch before the
   barrier the reads follow (the header's copies and the empty sphere table by
   `coupleBegin` and `cplGsClear` in `surfPipeSkip`, the prep's pair-path flag, the
   sleep rule's flag), so they are now made without the zeroing, as the wide and split
   pipelines are (`surface_pipes_zeroed`; `--ab-switch checks=21:5` puts it back for an
   A/B). The frame in segments, which is still what a fleet off the fast schedules
   runs, drops from 69.5 to 21.4 ms for one car with no other change.
2. *A workgroup a vehicle.* A vehicle's elements were walked `LANES` at a time, each
   lane a chain (the sedan's 4,251 segments: 17 a lane), and 20 vehicles filled 20
   workgroups of a GPU that runs hundreds. On the T4, 20 cars apart: the nine pipelines
   12.4 of the frame's 29.9 ms, the segments' find alone 4.2. An element's find and
   apply read what the prep and the finds left and write order-free sums, counts and
   masks, so the fast schedules' pipelines (`SURF_FAST_PASSES`) run **a thread an
   element across the GPU**, with no barrier and no workgroup memory (the coupling's
   header from the buffer, `EB_CPL_DIRECT`), each dispatch **sized on the GPU** by the
   prep (`gSurfArgs`, as the self split's walk and pairs are sized): a kind no vehicle
   has work for in a step gets no workgroup, exactly (the per-element loops need static
   slots or proxies left to them; the pair path a listed pair). The pair path's listing
   (`surfPairCull`, every element against every proxy shape's bounds: the prep's
   longest phase beside proxies, 19.3 of the 20-car mountain frame's 89 ms on the T4)
   is a dispatch a thread an element too (`explicit_surface_cull`), and its decision a
   small one after it (`explicit_surface_plan`). Eleven launches a step in place of
   nine; on the mountain the prep's 19.3 ms become 8.6 (the prep 3.2, the listing 3.8,
   the plan 1.6).

**What a host launches.** A wide frame of a lone car: each step's aux pass, node pass,
beam pass, list kernel, walk, pairs and hits pass as before, then the eleven surface
launches. Self split with vehicle contact: the vehicles' step, the walk, the pairs, the
members' rest, the solo vehicles' rest, the eleven surface launches, then the groups'
pass. The eleven: the prep (a workgroup a vehicle; it binds the argument buffer), the
listing (indirect), the plan (a workgroup a vehicle; it binds the buffer too), four
finds and four applies (indirect). No launch binds as storage the buffer it is sized
from (WebGPU forbids it; `schedule::surf_args_offset`, held by the schedule's tests).
Until a browser has compiled the pipelines, a fleet keeps the pass in its kernels (a
wide frame: the nine pipelines of a workgroup a vehicle).

**Left gated, and why.**

- *The wide passes' fused aux work with surface contact* (`wide_fused_on`): step s + 1's
  coupling and tyres would run before step s's surface pass (the table above). A lone
  car with surface contact pays one aux dispatch a step for it.
- *Self split without vehicle contact* (`self_split_on`): can host the pass the same way
  (the kernel's rest entry serves it); the launch of that entry outside the vehicle
  contact schedule is not built. No goal or default row takes it.
- *The split group path without self split*: its member module holds the pass; unchanged.
- *A browser*: `surface_fast_default` is off on wasm32. Chrome accepts every module
  (below); its bit-for-bit A/B on the lab page (AGENTS.md rung 3) has not been run.
- *engine-metal*: its host replays the planner's own commands and never turns the
  setting on.

**Results.** Bit-exact on, ms a frame of 34 steps. Every row of both tables ends bit
for bit, every state word and every coupling word (`--ab-switch surffast`, one build,
both ways from the row's saved state).

T4 (Modal, CUDA, `ENGINE_CUDA_FMAD=0`; both parts on, the CUDA host's default). Before:
§8.52's table (7459c5ce). After: `run.sh --cuda --baseline` at f12eee6c, mean of each
frame's best round (worst frame). On / off: `run.sh --cuda surffast`, 10 frames, 3
rounds.

| row | before | after | on / off in one build |
|---|---|---|---|
| apart1 | 31.40 (32.06) | 7.34 (7.54) | 7.35 / 31.46, −76.6 % |
| apart2 | 14.51 (14.76) | 12.81 (12.94) | 12.74 / 14.45, −11.8 % |
| apart4 | 23.92 (25.87) | 18.12 (20.14) | 18.11 / 23.81, −23.9 % |
| apart8 | 25.85 (27.95) | 19.62 (21.66) | 19.62 / 25.71, −23.7 % |
| apart16 | 28.04 (30.52) | 23.91 (25.79) | 23.98 / 28.31, −15.3 % |
| apart20 | 30.10 (33.74) | 26.79 (28.49) | 26.82 / 30.38, −11.7 % |
| apart32 | 38.01 (41.90) | 35.55 (37.98) | 35.57 / 37.78, −5.8 % |
| clash2 | 24.96 (29.40) | 20.65 (23.34) | 20.66 / 24.97, −17.3 % |
| clash4 | 28.55 (33.69) | 23.39 (25.69) | 23.35 / 28.47, −18.0 % |
| clash8 | 40.19 (43.53) | 32.77 (35.17) | 32.79 / 39.74, −17.5 % |
| clash16 | 46.04 (49.74) | 39.55 (42.51) | 39.52 / 45.97, −14.0 % |
| clash20 | 49.98 (54.16) | 43.96 (47.33) | 43.99 / 50.01, −12.0 % |
| clash32 | 61.74 (66.49) | 57.31 (61.48) | 57.36 / 61.52, −6.8 % |
| mountain20_t2 | 36.58 (38.54) | 34.87 (36.07) | 34.85 / 36.48, −4.5 % |
| mountain20_t4 | 78.10 (95.03) | 65.11 (75.29) | 65.22 / 77.82, −16.2 % |
| mountain20_t7 | 75.87 (77.69) | 65.52 (66.76) | 65.54 / 76.04, −13.8 % |
| mountain20_t12 | 86.15 (91.96) | 77.55 (83.54) | 77.44 / 86.50, −10.5 % |
| pole | 48.36 (78.64) | 22.44 (51.34) | 22.10 / 48.56, −54.5 % |
| wall_s | 34.37 (35.21) | 8.97 (9.93) | 8.95 / 34.38, −74.0 % |
| bricks_s | 48.12 (49.84) | 14.71 (15.15) | 14.72 / 48.28, −69.5 % |
| pile8_s | 48.28 | 31.67 (33.27) | 31.68 / 48.10, −34.1 % |
| drive (no surface contact) | 3.89 (7.14) | 3.90 (3.94) | not on the path |
| pile8 / pile32 (no surface contact) | 20.87 / 75.74 (80.58) | 20.91 (21.52) / 75.83 (80.49) | not on the path |

mountain20_t20 and the derby rows were not run on the T4 (as in §8.52). The 12 default
rows, base 7459c5ce against f12eee6c as two builds (`run.sh --cuda --base 7459c5ce
--head f12eee6c`): every end state word for word, −0.5 to +0.2 %, PASS.

Mac (M3 Max, native Metal, the GPU shared with other jobs through the lock). Before:
§8.52's table. On / off: `run.sh surffast`, 10 frames, 3 rounds, both parts switched;
its off half already has the pipelines without the zeroing (above), which is why a
lone car's off is 21 ms and not §8.52's 70.

| row | before (§8.52) | after, the default | both parts: on / off in one build |
|---|---|---|---|
| apart1 | 69.72 (70.33) | 6.43 (6.61) | 6.51 / 21.49, −69.7 % |
| apart2 | 8.33 (8.45) | 8.34 (8.45) | 7.57 / 8.35, −9.3 % |
| apart4 | 12.87 (14.34) | 12.86 (14.29) | 10.89 / 12.91, −15.7 % |
| apart8 | 13.59 (14.93) | 13.53 (14.85) | 12.68 / 13.63, −7.0 % |
| apart16 | 14.07 (15.41) | 14.07 (15.35) | 13.33 / 14.16, −5.8 % |
| apart20 | 14.30 (15.59) | 14.27 (15.60) | 13.93 / 14.37, −3.0 % |
| apart32 | 15.26 (16.59) | 15.30 (16.62) | 15.61 / 15.39, +1.4 % |
| clash2 | 15.86 (16.84) | 15.73 (16.79) | 14.49 / 15.80, −8.3 % |
| clash4 | 16.44 (17.75) | 16.39 (17.70) | 14.78 / 16.16, −8.5 % |
| clash8 | 23.41 (24.14) | 23.48 (24.17) | 23.55 / 23.35, +0.9 % |
| clash16 | 24.85 (25.62) | 24.54 (25.58) | 25.50 / 24.61, +3.6 % |
| clash20 | 25.41 (26.25) | 25.14 (26.05) | 26.51 / 25.31, +4.7 % |
| clash32 | 30.79 (33.37) | 31.14 (33.23) | 29.99 / 32.83, −8.6 % |
| mountain20_t2 | 19.32 (22.44) | 19.01 (19.86) | 18.02 / 18.60, −3.1 % |
| mountain20_t4 | 38.31 (41.89) | 37.04 (40.54) | 33.83 / 36.72, −7.9 % |
| mountain20_t7 | 37.89 (39.66) | 36.74 (37.63) | 33.32 / 36.19, −7.9 % |
| mountain20_t12 | 41.40 (46.03) | 40.73 (44.63) | 39.53 / 40.36, −2.0 % |
| mountain20_t20 | 68.11 (70.38) | 66.32 (67.65) | 65.32 / 66.72, −2.1 % |
| derby20_t2 | 23.03 (24.53) | 21.95 (24.63) | 22.99 / 22.40, +2.6 % |
| derby20_t5 | 30.61 (33.62) | 29.74 (31.60) | 32.78 / 29.88, +9.7 % |
| derby20_t10 | 29.32 (31.20) | 28.32 (30.02) | 30.52 / 28.40, +7.5 % |
| pole | 82.64 (99.09) | 16.81 (31.24) | 16.10 / 32.90, −51.1 % |
| wall_s | 73.73 (79.23) | 7.50 (9.31) | 7.51 / 24.55, −69.4 % |
| bricks_s | 78.91 (82.42) | 11.19 (11.89) | 11.28 / 29.64, −61.9 % |
| pile8_s | 25.34 | 25.07 (26.69) | 20.92 / 25.22, −17.1 % |

"After, the default" is `run.sh --baseline` at c76d09ba: a wide frame's part on, the
vehicle-contact part off, so the rows of several cars run the kernels of 7459c5ce and
keep its times. The 12 default rows (no surface contact), this build's lab against
7459c5ce's built beside it, each `run.sh --baseline` in a run of its own on the shared
GPU (not a paired A/B): drive 3.26 / 3.55, tumble 3.36 / 3.45, wall 3.91 / 3.89, pile2
8.02 / 7.92, pile8 12.47 / 12.34, pile32 35.57 / 36.28, headon 11.76 / 12.10, side 8.47
/ 8.44, bricks 3.36 / 3.62, rocks 3.68 / 4.07, barrier 7.01 / 7.32, truss 1.51 / 1.69:
−10.6 to +1.2 %, none slower past the runs' own spread.
The pipelines with naga's zeroing put back against without it, 6 frames, 2 rounds,
every state word the same on every row: the frame in segments
(`RUBBLE_SURFACE_FAST=0 run.sh checks=21:5`) apart1 69.58 / 21.62 ms, pole 87.30 /
39.18, wall_s 73.27 / 25.08; the fast schedules (`run.sh checks=21:5`) apart1 17.31 /
6.55, pole 33.42 / 23.30, bricks_s 22.24 / 11.63.

The Mac gate fails on time on two rows, **derby20_t5 (+9.7 %) and derby20_t10
(+7.5 %)**, and clash16 and clash20 are 3.6 and 4.7 % slower (inside the gate's 5 %).
Every one of them is a row of 16 to 20 cars in contact groups. Each dispatch timed
apart on derby20_t5 (the timing inflates each): the eleven surface launches are 12.6 of
the frame's 33 ms (the pair path's find 3.7, the prep 1.5, the segments' find and apply
1.4 and 1.3, the triangles' find 1.3), where the same work inside 20 members'
workgroups costs less on this GPU; the T4 is the other way round (20 workgroups fill
half of it). So the setting has two parts with their own defaults
(`schedule::surface_fast_default`, `surface_fast_contact_default`):

- **A wide frame's part** (a lone car, or vehicle contact off): on natively and on
  CUDA. Faster on every row of both machines that takes it.
- **The vehicle-contact part** (self split with vehicle contact): on for CUDA, **off
  for wgpu** unless `RUBBLE_SURFACE_FAST=1`. A change that slows a row is not made a
  default. On Metal, of the 21 rows that take it, 14 are 2 to 17 % faster, three are
  within 3 % (clash8 +0.9, apart32 +1.4, derby20_t2 +2.6), clash16 and clash20 are 3.6
  and 4.7 % slower and the two derby rows are past the gate. With it off those rows
  run exactly the kernels of 7459c5ce (every module they launch is byte for byte the
  same).

What would make the vehicle-contact part pass on Metal (not built): fewer launches a
step there (the three per-element kinds' finds in one dispatch over one index, as the
listing is, and their applies in another: eleven to seven, if the fused kernels stay
under the browser's compile cap); the solo vehicles' rest in the members' rest dispatch
(one launch fewer when a group exists); an early return for a pair thread past its
vehicle's list before it loads the vehicle's records (the pair kinds are sized by the
longest list of the fleet). Tried and taken out: the applies sized by their finds (a
second argument buffer the finds write: no apply for a kind that found nothing); bit
for bit, no gain on the T4 and 3 % slower on the Mac's clash20, since a car's wheels
and skin meet a static slot on most steps.

**Proofs** (2026-11-22; commands from `engine/`):

- `tools/scenario-matrix/run.sh surffast ROW...` on the Mac: 25 goal rows bit for bit.
- `ENGINE_CUDA_FMAD=0 tools/scenario-matrix/run.sh --cuda surffast ROW...`: 21 goal
  rows bit for bit, PASS.
- `ENGINE_CUDA_FMAD=0 tools/scenario-matrix/run.sh --cuda --base 7459c5ce --head
  f12eee6c`: 12 default rows bit for bit, PASS.
- `cargo test --profile fast -p engine-webgpu --test explicit_surface_fast_gpu` (new):
  a lone car into a wall slot and a static or a free proxy pole on the wide passes; three
  cars with vehicle contact, a head-on pair and one alone, with and without vehicle
  sleep: every state and coupling word against the pass in the kernels, 6 tests.
- `explicit_loop_cap` 6, `explicit_surface_proxy_gpu` 4 (GPU against the CPU reference),
  engine-world-wasm `surface_contact_gpu` 3, `self_split_gpu` 8, `group_split_gpu` 3,
  engine-softbody `host explicit_surface` 21 (CPU), engine-exec 21 (the schedule's
  launches for both paths), engine-webgpu's library tests 72: pass.
- Chrome: `EXPLICIT_WGSL_MODULES_ONLY=1 node --experimental-websocket
  explicit-wgsl-check.mjs`: all 155 modules. `tools/build-time/compile_cost.py --check`:
  the new modules are predicted 0.0 to 2.2 s (cap 10 s). `tools/build-time/check.py`:
  nothing new (it fails on `vehicle_lab.rs`, 3,365 lines against its 3,357, as it does
  at 7459c5ce; this change adds no line there).
- Of the 425 modules of 7459c5ce, eight differ: the self split's step and vehicles'
  step modules (the rest's skip, a uniform test of a parameter). Fifteen are new.

**Not verified.** Chrome's bit-for-bit A/B on the lab page and whole 40 s runs (rungs 3
and 4); engine-metal's parity suite (its host is unchanged); the T4's derby rows.
One run of the T4's `pole` row of fourteen (three rounds of ten frames) ended with one
word different between the halves, the car's step count (200 on, 540 off: the
snapshot's own count on the on half, every other word the same), and did not repeat in
44 further rounds of that row or in any other run; its cause was not found. `explicit_replay` now also fails a half that
does not repeat its own last round and names it, so that a difference between the
halves is not read as the switch's.

### 8.54 After §8.53: where 20 cars' time is on the Mac (2026-11-22)

Checked at the merge of §8.53 (7dff0d41), Mac, `run.sh surffast` (both parts on / off):
apart1 5.72 / 21.41, pole 17.10 / 33.26, wall_s 8.41 / 24.62 (bit for bit), and the
vehicle-contact part slower on clash20 (+7.0 %) and derby20_t5 (+10.9 %), as §8.53 says;
it stays off for wgpu.

By dispatch kind (`PROFILE_SPLIT=1 run.sh perf=0`, wgpu's default, each dispatch a timed
pass; Metal's timestamps misattribute between kinds, the dispatch counts are exact):

| row | frame, untimed | members' step | fleet walk | levels | the rest |
|---|---|---|---|---|---|
| apart20 | 13.8 | one `Solo` dispatch of 13.0 (every car alone in its workgroup, 34 steps: 0.38 ms a step) | — | — | 334 of 378 dispatches are empty |
| clash20 | 25.2 | 17.3 (34 × 0.51) | 0.2 | 3.8 | 3.6 |
| mountain20_t12 | 42.8 | 23.8 (34 × 0.70) | 10.2 (34 × 0.30) | 3.3 | 6.5 |
| mountain20_t20 | 66.9 | 26.0 (34 × 0.76) | 30.7 (34 × 0.90) | 6.4 | 7.5 |

Cars apart with vehicle contact off, so that the fleet runs the wide passes with the
surface pipelines (live runs of 2 s, GPU ms a step of the explicit frame, median / p95):
1 car 5.47 / 5.50, 8 cars 6.40 / 6.47, 20 cars 9.07 / 12.36, 32 cars 10.74 / 14.28. That
is about 5.3 ms of launches (19 a step) plus 0.17 ms a car. With vehicle contact on and
no group the same cars are 9.2 / 14.3 at 20.

Reading:

1. A car's whole step in one workgroup costs 0.4 to 0.8 ms a step on the Mac whatever
   else happens; across the GPU (the wide passes) 20 cars' steps cost about 0.1 ms a
   step together. The members' step is the largest part of every row in contact.
2. The fleet walk grows with the pile: 0.3 ms a dispatch at t 12, 0.9 at t 20. It is
   the mountain run's growth from 40 to 70 ms.
3. The frame's floor is its launches: 19 a step for a lone surface car is 5.3 ms before
   any car's work.

### 8.55 The fleet walk on the mountain: the kept candidates are never used there (2026-11-22)

§8.54's second reading asked why `explicit fleet walk` grows with the pile. Measured at
the merge of §8.54, Mac, wgpu's default, each row's fixture replayed for 6 frames (204
contact passes; `vehicle_lab --from-fixture ROW.fx --ab-switch perf=0 --ab-frames 6`,
its `kept candidates` line):

| row | builds | passes that reused the kept pairs | passes on the front's own path |
|---|---|---|---|
| clash20 | 17 (232,378 pairs a build, most an owner 17,385) | 502 of 519, 96.7 % | 0 |
| mountain20_t12 | 0 | 0 | 204 of 204 |
| mountain20_t20 | 0 | 0 | 204 of 204 |

Measured: on both mountain rows vehicle contact's kept candidates (`PERF_FLEET_KEEP`,
§8.28, §8.39) do no pass at all. Every pass takes the front's own path: the gather, the
grids, the sorted copy and the walk of every listed triangle (`fleetWideTests`, `flWalk`),
which is the dispatch §8.54 timed at 0.30 ms (t 12) and 0.90 ms (t 20), against 0.2 ms a
frame on clash20 where 96.7 % of the passes test the kept pairs alone. So the walk does
not scale badly in itself; it runs on every pass because the list that replaces it is
off, and its work is the pile's listed triangles.

Not yet measured (hypothesis): why no build is tried in 204 passes. The code has two
ways there: a build past the fleet's room or an owner's share voids the list and is
followed by a wait on the front's own path that doubles with each such build in a row,
to `FK_BACKOFF << 4` = 2,176 passes (`explicit_beams_fleet_track.slang` `fkWaitLeft`,
its setter; the room is `vehicles · FK_CAP`, `fkRoom`), and a list that lasted fewer
than `FK_SHORT` passes waits `FK_THRASH_WAIT`. A pile of 20 cars on ledges has every
car near several others, where clash20 has ten separate pairs. The next step is the
fixture's own control words (`fkCtl(FK_WAIT)`, the last build's counts by owner), then
the bound: the mountain rows with the walk's dispatch done twice (time only), which
says what a pass that reused its pairs would give back at most (clash20's ratio
suggests most of 10.2 ms a frame at t 12 and of 30.7 ms at t 20).

Not built in this section: every car's own step across the GPU under vehicle contact
(§8.54's first reading), and fewer surface launches a step. What the first needs that
the wide passes do not hold today, from the members' step (`contactMemberStep`,
`explicitBeamsFrameStep` under `EB_VV_SOLO`): the solo vehicles' frame-bounds check in
the node pass (`nodeUpdateSolo`, `vvReportOutside`, the allowed box in workgroup
memory), each member's contact box by step parity (`memberBox`, `vvBoxKey`), the
pending-beam reset of a member's triangle throttles (`VV_FLAGS.w`), the members' pair
lists' state kept between steps in the solo-step words (`memberListLoad`), the kept
candidates' tracking of every node (`fkTrackVehicle`), and wheels on each other
(`wheelsPartners`) between a step's coupling and its nodes. Each is per element or per
vehicle and none reads workgroup state of another, so none is a reason it cannot be
hosted; each is kernel work in the wide node pass and its proof.

### 8.56 The fleet walk on a pile: why the kept candidates are off, and the walk made cheap without them (2026-11-22)

§8.55 left two questions: which rule keeps the kept candidates off on the mountain, and
what the walk costs when it has to run. Branch `claude/fleet-walk-kept`, from 25ea30b4.
Mac: M3 Max, native Metal, the GPU shared with other sessions through the lock. T4:
Modal, CUDA, `ENGINE_CUDA_FMAD=0`. Bit-exact on everywhere.

**The cause (measured).** `FK_DEBUG=1 vehicle_lab <row> --from-fixture ROW.fx --ab-switch
perf=0 --ab-frames 3` prints the kept candidates' control words and the plan's adjacency
(`ExplicitBeamsGpu::fleet_keep_debug`, new):

| row | a vehicle's partners in the plan (20 vehicles) | partners a build covers | the control words |
|---|---|---|---|
| mountain20_t12 | 19, 19, 8, 17, 19, 12, 14, 18, 18, 18, 15, 13, 15, 17, 17, 18, 15, 15, 19, 14 | 8 (`FK_COVER`) | no list; 1,897 passes of the wait left (of 2,176), 4 failed builds in a row; the last build covered 0 partners for 19 vehicles |
| mountain20_t20 | 18, 19, 8, 17, 19, 14, 14, 18, 18, 18, 17, 13, 18, 17, 18, 18, 18, 15, 19, 18 | 8 | no list; 1,136 left, 4 in a row |
| clash20 | 1 or 0 | 8 | list valid, 15 builds and 334 reused passes in 10 frames |
| derby20_t5 | 0 to 4 | 8 | list valid, 1 build and 33 reused passes a frame |
| derby20_t10 | 0 to 6 | 8 | list valid, 1 build and 33 reused passes a frame; most pairs an owner kept 43,495 |

So it is neither the fleet's pair room nor an owner's share nor a short life: it is the
third rule of `fkDecide`, **a vehicle with more partners in the plan than a build covers**
(`tooMany`: `FK_COVER` = 8, the words a vehicle has for its build's partners). In a pile of
20 on the ledges every car but one has 12 to 19 partners (their boxes with the 5 cm margin
overlap), so every build attempt ends before its gather, takes the back-off (136 passes
doubling to 2,176) and the fleet stays on the front's own path: one attempt in 2,176
passes, 64 frames. §8.39 met the same rule at 32 cars ("a car with more partners than a
build covers, 8"). The derby keeps its list: at t 5 and t 10 it is rebuilt about once a
frame and reused on 33 passes of 34.

**What a build would need** (not built): the cover's words a vehicle are part of the
buffers' layout (`fleet_keep_words`), so 19 partners is a layout change; the build's run
is sized `VV_PART` (8) times the fleet's contact points (87,680 here) where the front's own
run has the fleet's room (19 times: 208,240); and a build pads every triangle's box by
0.25 to 0.9 m, in a pile where a triangle's cells hold candidates of every neighbour. §8.39
tried 16 partners at 32 cars: 505,000 pairs a build, 8 % slower, reverted. With the walk
at its cost below, what a kept list could still give back on these rows is the front's
other dispatches, 6 to 7 of 34 ms (below).

**What the walk does on a pile (measured, counters in a time-only build; a frame is 34
passes).**

| a frame of 34 passes | mountain20_t12 | mountain20_t20 |
|---|---|---|
| listed triangles walked | 336,396 (9,894 a pass: every owner lists its 528; a partner approaches fast on every pass and resets the throttles) | 334,662 |
| cells their boxes cover | 3.43 million (10 a triangle) | 3.34 million |
| candidates visited in those cells' buckets | 9.20 million | 12.48 million |
| of them inside the triangle's box at its largest pad | **698** | **55,397** |
| triangles over the grid's 64 cells, which test every candidate of their owner | 544 (16 a pass, every one over 4,096 cells: a torn panel's triangle, metres long) | 1,444 (42 a pass: the same 16, and 26 of 64 to 4,096 cells) |
| candidates those test | 1.64 million (3,000 a triangle) | 7.84 million (5,400 a triangle) |
| iterations of the walk's search for a bucket met twice | 19.5 million | 19.0 million |

Two things, both of the repo's known kinds. (1) A visited candidate cost three loads, two
of them dependent (`vvInBox` loads the candidate's vehicle for the anti-clip flag and its
velocity for its pad) before the comparison that rejects 9,199,000 of 9,200,000. (2) A
triangle over the cells tests every candidate of its owner on its three threads, chains
of 1,000 to 3,000 loads each, and the dispatch waits for its longest thread: a loop that
one lane runs for the whole GPU.

**The bound, time only** (20 frames from the fixture, wall): the walk's work done twice
(a second, dry copy of `vvTriangleCandsList` that lists nothing and keeps a checksum
alive) adds 7.3 ms a frame at t 12 (43.3 → 50.5) and 19.4 ms at t 20 (67.1 → 86.5). So
the walk costs that much, and a walk at clash20's 0.2 ms a frame would give 36.2 ms at
t 12 and 47.9 ms at t 20. (§8.54's timed passes said 10.2 and 30.7 ms: Metal's
timestamps overstate it, as that section warned.) With the candidates' position test in
(below), each part left out in turn (time only, results differ): no test of every
candidate −3.9 ms at t 12 and −23.0 at t 20 (the hits those triangles find go with it);
no bucket scans −3.0 and −2.5; no search for a bucket met twice −0.1 and −0.1; the
thread's own start (the triangle's frame, three nodes) 1.7 ms.

**What changed** (`explicit_beams_fleet.slang`: `flWalkBoxed`, `flWalkBig`, `flBigHint`;
perf flag 536870912, `PERF_FLEET_WALK_BOX`; the walk of the front's own path only: the
kept candidates' build and tests, the groups' own dispatches and the larger groups' pass
are untouched). Each lossless: the same (triangle, candidate) pairs listed, each once.

1. **A candidate's position first.** The walk already has the triangle's box grown by
   the largest pad any candidate of the owner can have (it sizes the cells from it: the
   owner's fastest candidate, `FL_FAST`). `vvInBox`'s pad for a candidate is the same
   expression of that candidate's own speed, which is at most the fastest's, and every
   operation in it is monotone, so a candidate outside the larger box fails `vvInBox`:
   it is skipped before the two dependent loads. A NaN compares false and goes on to
   `vvInBox` as before. The first test's box is 0.1 mm and a millionth of the
   coordinates larger still: with fast arithmetic (the default outside the tests) a
   compiler may fuse the multiply-add of one pad and not of the other, a last place
   apart; inside that room `vvInBox` alone decides, so the pairs are the same in either
   setting. Alone: mountain20_t12 39.7 → 36.6 ms, mountain20_t20 66.8 → 64.7.
2. **A big triangle's test on threads of its own.** The throttles' dispatch marks a listed
   triangle whose box covers more than the walk's 64 cells (`flBigHint`, in a pass that
   walks: bit 1 of the triangle's listed word, `fkListed`) and puts it in the fleet's list
   of big triangles (`flListAt`: a region of a word a triangle that the layout has held
   since stage 2 and nothing used; an atomic count in a free control word, cleared by the
   first plan). The walk's plan adds 128 threads for each (`FL_BIG_PARTS`) after the
   listed triangles' own, and thread p of a big triangle tests candidates p, p + 128, ...
   of its owner, 20 to 70 each. The stored mark alone decides who tests a triangle: the
   walk's threads leave a marked one to its own threads, which test every candidate
   whatever the walk's own count of cells would be (the cells only leave out candidates
   that fail the box test, so testing every candidate lists the same pairs). No value is
   worked out in two places and compared, so fast arithmetic cannot make two threads
   disagree about a triangle. Which triangles are marked is a choice of schedule. No
   change of the buffers' layout.

The fleet's parameter copies have no free bit in their perf word (bits 27 to 31 are the
level slots there), so the planner keeps the switch in bit 30 of the word after the beam
order (`FramePlanner::set_perf_flags` and `perf_flags` carry it as bit 29 for their
callers: `perf=536870912`, `RUBBLE_PERF_FLAGS`, the lab page's `--ab`).

**Results.** On against off in one build from each row's saved state, 3 rounds of 10
frames, every state word the same on every row of both machines
(`tools/scenario-matrix/run.sh perf=536870912`, and `--cuda`). ms a frame of 34 steps.

| row | Mac on / off (worst frame on / off) | Mac, default before → after: mean (worst frame) | T4 on / off (worst frame on / off) |
|---|---|---|---|
| mountain20_t2 | 19.13 / 19.31, −0.9 % (19.82 / 19.95) | 18.84 (19.87) → 19.19 (19.87) | 34.61 / 34.89, −0.8 % (35.73 / 36.13) |
| mountain20_t4 | 37.04 / 37.12, −0.2 % (40.37 / 40.43) | 36.94 (40.48) → 36.60 (40.39) | 64.40 / 64.98, −0.9 % (75.92 / 75.86) |
| mountain20_t7 | 35.66 / 36.80, −3.1 % (36.40 / 37.67) | 36.83 (37.71) → 35.18 (36.35) | 62.91 / 65.34, −3.7 % (64.33 / 66.93) |
| **mountain20_t12** | **34.24 / 40.84, −16.1 % (37.91 / 44.84)** | **40.57 (44.85) → 34.21 (37.99)** | **64.72 / 77.44, −16.4 % (70.10 / 83.35)** |
| **mountain20_t20** | **43.09 / 66.60, −35.3 % (45.98 / 74.89)** | **66.52 (69.13) → 43.16 (44.41)** | **81.00 / 125.46, −35.4 % (83.37 / 129.71)** |
| derby20_t2 | 22.23 / 22.03, +0.9 % (31.88 / 30.37) | 22.02 (23.97) → 22.40 (23.42) | 37.78 / 37.78, +0.0 % (39.46 / 39.47) |
| derby20_t5 | 30.00 / 30.31, −1.0 % (38.09 / 38.19) | 29.43 (31.12) → 29.76 (31.61) | 55.80 / 55.80, −0.0 % (58.55 / 58.51) |
| derby20_t10 | 30.26 / 29.80, +1.5 % (37.23 / 35.72) | 28.20 (29.67) → 28.69 (30.21) | 51.77 / 51.81, −0.1 % (55.09 / 55.30) |
| clash2 / 4 / 8 | +0.1 / +0.1 / +0.3 % | 15.41 (16.30) → 15.60 (16.76); 16.11 (17.76) → 16.35 (17.62); 23.55 (24.17) → 23.52 (24.19) | +0.0 / +0.1 / +0.0 % |
| clash16 / 20 / 32 | +0.1 / +0.2 / −0.2 % (clash20 25.39 / 25.34, worst 26.16 / 26.18) | 24.64 (27.64) → 24.71 (25.56); 25.17 (26.15) → 25.45 (26.25); 30.93 (33.20) → 30.83 (33.32) | −0.0 / +0.1 / +0.0 % (clash20 43.81 / 43.79, worst 47.18 / 47.09) |
| apart1 to apart32 | −0.1 to +0.8 % | 6.17 (6.61) → 6.41 (6.61); 8.33 (8.41) → 8.33 (8.46); 12.86 (14.24) → 12.87 (14.28); 13.54 (14.81) → 13.56 (14.92); 14.06 (15.36) → 14.05 (15.39); 14.26 (15.64) → 14.25 (15.66); 15.27 (16.62) → 15.27 (16.60) | −0.2 to +0.1 % |
| pole / wall_s / bricks_s / pile8_s | −0.2 / +0.1 / −0.0 / −0.0 % | 16.13 (30.40) → 16.68 (31.45); 7.51 (8.85) → 8.12 (9.34); 10.24 (10.80) → 11.15 (11.98); 24.98 (26.62) → 25.17 (26.76) | within 0.6 % |
| drive / tumble / wall | +1.1 / +0.4 / +1.6 % | 3.52 (3.62) → 3.42 (3.67); 3.38 (3.54) → 3.43 (3.52); 4.05 (4.41) → 4.24 (4.45) | +0.0 / +0.4 / +0.1 % |
| pile2 / pile8 / pile32 | +0.2 / +0.0 / −0.2 % | 8.24 (8.44) → 8.22 (8.41); 12.80 (13.25) → 12.99 (13.36); 36.13 (40.07) → 35.51 (39.39) | +0.0 / +0.1 / −3.0 % (pile32 73.42 / 75.71, worst 78.98 / 81.04) |
| headon / side / bricks | −0.1 / −0.1 / +3.1 % (bricks 3.78 / 3.67) | 11.81 (12.53) → 11.89 (12.52); 8.41 (11.16) → 8.83 (11.67); 3.34 (3.50) → 3.72 (3.83) | within 0.1 % |
| rocks / barrier / truss | +0.4 / +1.1 / +0.5 % | 4.01 (4.27) → 3.96 (4.18); 7.35 (8.27) → 7.29 (8.25); 1.71 (1.84) → 1.70 (1.91) | within 0.2 % |

"On / off" is the paired A/B (the verdict: PASS on both machines, 37 rows, none slower
past 5 % and 0.05 ms; `WORST=1` prints the slowest frame of any round of each half). Both
columns are the last run, on the kernel as committed; two earlier runs of each machine on
the kernel before the first test's rounding room gave the same verdicts and the same
deltas to 0.5 % (mountain20_t12 −16.2 and −16.5 %, mountain20_t20 −32.9 and −35.5 %). On
the T4 the other rows' worst frames are within 0.3 ms between the halves but for one
frame of the on half on three rows: drive 7.61, wall 9.78 and pile2 24.10 ms against 3.94,
6.69 and 12.34 in one run, 9.28, 8.68 and 28.57 in the next. It is not the switch: the
same rows with `perf=0`, same against same, give pile2 22.46 / 12.26 (the lone cars 4.59 /
4.60 and 3.93 / 3.92 that time). The on half is the first of each round; why its pile2
has one slow frame was not looked into (a hypothesis: a module's first load in a timed
frame). Each frame's best round, the number compared, does not see it. "Default before →
after" is two `run.sh --baseline` runs of this
build one after the other on the shared GPU, the first with `RUBBLE_PERF_FLAGS=165150661`
(the defaults without the switch), each row a process of its own: not a paired A/B, and
the one-car rows change level between processes by more than this change could (bricks
3.34 → 3.72 and bricks_s 10.24 → 11.15 there are −0.6 % and −0.8 % in the paired run; a
lone car launches no fleet dispatch).

The worst frame of the mountain's last fixture goes from 69.1 to 44.4 ms on the Mac. From
t 12 to t 20 the frame now grows from 34.2 to 43.2 ms where it grew from 40.6 to 66.5.

By dispatch kind on the Mac (`--ab-profile`, each dispatch a timed pass, ms a frame;
Metal's timestamps overstate a kind, the comparison is like for like):

| kind | t 12 before → after | t 20 before → after |
|---|---|---|
| fleet walk | 10.17 → 2.25 (299 → 66 µs a dispatch) | 30.97 → 3.31 (911 → 98 µs) |
| fleet triangles, points (the marks are made here) | 1.87 → 1.90 | 2.80 → 2.82 |
| members' step | 23.89 → 23.83 | 25.98 → 25.98 |
| levels | 3.31 → 3.31 | 6.42 → 6.41 |

**Default: on natively and on CUDA** (`perf_flags_default`), off in a browser until
Chrome's bit-for-bit A/B on the lab page (AGENTS.md rung 3) is run.

**What is left of 20 cars on the mountain, and next.** After this the walk is within 1
to 2 ms a frame of no walk at all (the time-only "no walk" frames above also lose the
hits' work, so that is an upper bound of what is left). The frame at t 12 is the members'
step (23.8 ms in §8.54's split), the levels (3.3) and the front's seven other dispatches
a pass (the triangles and points, the owners' parts, the grids, three prefixes, the
pairs: 6 to 7 ms a frame, mostly launches). A kept list that holds on a pile would
replace those with one dispatch; it needs the cover for 19 partners (a layout change)
and a build whose pairs fit the fleet's room, which §8.39's attempt at 16 partners did
not reach. Not built here.

**A bug found on the way, fixed because the proof needed it.** `explicit_replay` (the
CUDA matrix's row runner) set a `perf=MASK` switch, ran one untimed frame, put the state
back and timed: `CudaExplicitBeams::restore` puts the planner's words back with the
bindings, the perf flags among them, so **both halves of every `perf=MASK` A/B on CUDA ran
the snapshot's flags**, same against same (this switch's first T4 run: mountain20_t20
124.86 / 124.79 ms, every row within 0.3 %). The switch is now applied again after the
state is put back (80.94 / 125.56). Earlier T4 verdicts of `perf=` switches were
same-against-same and say nothing; the host switches (`surffast`, `groupsplit`,
`selfsplit`, `widelist`, `fused`, `wide`, `devindirect`) are fields of the host and were
not affected.

**Proofs** (2026-11-22; commands from `engine/`):

- `WORST=1 tools/scenario-matrix/run.sh perf=536870912 ROW...` (all 37 rows named), Mac:
  every row bit for bit, `verdict: PASS; 37 rows in 95.2 s` (before the rounding room:
  `run.sh perf=536870912`, `verdict: PASS; 12 rows in 20.3 s`, and `... goal`, `verdict:
  PASS; 25 rows in 97.8 s`).
- `ENGINE_CUDA_FMAD=0 MODAL_OPTS="--pack 4 --env WORST=1" tools/scenario-matrix/run.sh
  --cuda perf=536870912 ROW...` (all 37 rows named): every row bit for bit, `verdict:
  PASS; 37 rows on Modal T4, in 4 container(s)`, three times (twice before the room).
- `tools/explicit-2khz/perf-verify.sh default webgpu:fleet_walk_gpu webgpu:fleet_contact_gpu
  webgpu:group_split_gpu webgpu:explicit_contact_gpu` (release, bit-exact on, the switch
  on by default): PASS, 1, 1, 3 and 13 tests. `perf-verify.sh 701988805` (the defaults
  with the kept candidates off, so that every contact pass walks) on `fleet_contact_gpu`,
  `group_split_gpu` and `explicit_contact_gpu`: PASS, 1, 3 and 13. (`fleet_walk_gpu` and
  the second run's `group_split_gpu` and `explicit_contact_gpu` on the kernel as
  committed, the others before the rounding room.) On the kernel as committed again:
  `perf-verify.sh default webgpu:fleet_contact_gpu metal:vehicle_parity`: PASS, 1 and 4.
- `cargo test --profile fast -p engine-webgpu --test fleet_walk_gpu` (new, bit-exact on):
  a column of ten and a column of six with torn panels (a triangle's node of two cars
  moved 4 m, so that triangles are over the cells), the kept candidates off so that every
  pass walks, with and without self split: 40 frames each, every state word the same with
  the switch on as off; up to 9 and 18 big triangles a pass on threads of their own.
- `cargo test --profile fast -p engine-webgpu --test explicit_loop_cap`: 6 passed (the
  new loops are capped, ids 640 to 643). `cargo test --profile fast -p engine-exec`: 25.
- Chrome: `EXPLICIT_WGSL_MODULES_ONLY=1 node --experimental-websocket
  explicit-wgsl-check.mjs`: all 155 modules compile. `tools/build-time/compile_cost.py
  --check`: 0 failing. `tools/build-time/check.py`: the two `vehicle_lab.rs` lines of the
  base (3,383 lines against 3,375; `mountain_perf`), nothing new.

**Not verified.** Chrome's bit-for-bit A/B on the lab page (rung 3: the web build needs
the fracture lab's Git LFS assets, which this worktree does not hold; the switch stays
off in a browser, where Chrome accepts every module); whole 20 and 40 s runs (rung 4)
and the lab's own runs with the AI driving; the fast setting's tolerance suite
(`fast_arithmetic_gpu`, `run.sh bitexact`). The time-only
figures (the bound, the parts left out) are from builds that were not committed.

### 8.57 Every car's own step on the wide passes under vehicle contact: the map, and the frames without a group (2026-11-22)

§8.54's first reading: a car's whole step in one workgroup costs 0.4 to 0.8 ms a step on
the Mac, the same cars' steps across the GPU (the wide passes) about 0.1 ms a step
together. A fleet with vehicle contact on never takes the wide passes today
(`FramePlanner::is_wide` needs contact off). This section is the map of what a car's own
step is on each path, then the first part built: the frames in which the contact plan
holds no group.

**The paths today** (contact on, `schedule::frame`):

| the fleet | the plan holds no group | the plan holds a group |
|---|---|---|
| surface contact, its pass in the kernels (wgpu's default: `surface_fast_contact_default(false)`) | one `Solo` dispatch, a workgroup a vehicle for the whole frame (`explicit_beams_frame` under `EB_VV_SOLO`), sized by the plan (`vvArgs` words 0..2: the vehicles, or 0 with a group) | every vehicle a step a dispatch on `explicit_beams_split_member` (`Member(s)`, sized by words 20..22: 0 without a group): a solo vehicle `explicitBeamsFrameStep`, a member `splitMember` |
| no surface contact, or its pass as pipelines with vehicle contact (CUDA's default): self split with vehicle contact | every vehicle a step a dispatch whatever the plan (`SelfMember(s)`, words 28..30: the vehicles always), the walk, the pairs, the members' rest (none without a group), the solo vehicles' rest and the surface pipelines (surface fleets) | the same dispatches, then the fleet's front, tests and levels |

**The map.** What a vehicle's own step runs on the members' path and on the solo
vehicles' (`contactMemberStep`, `splitMember`, `memberRestStep`, `contactMemberBegin`,
`contactMemberEnd`, `splitGroupBegin`, `splitGroupEnd`; `explicitBeamsFrameStep` under
`EB_VV_SOLO`), against the wide passes (`explicit_beams_wide_*`, `explicit_beams_wide_list`).
"Reading" marks what is read from the code and not yet held by a test.

| item | today: where, what it reads and writes | on the wide step | status |
|---|---|---|---|
| the frame's start: the origin's rebase | solo: the step dispatch of step 0 (every lane reads node 0, the nodes shifted, `gMisc[v].xyz`, `cplRebase`); member: `contactMemberBegin` in the groups' begin dispatch | `wide_begin` (the shift into `gMisc[wideMisc + 1 + v]`, `cplRebase`), `wide_rebase` (a thread a node), then the host's copy of the origins and step counts into the vehicle table | built for a frame without a group (the same arithmetic: `P − first` when `length(first) > REBASE_RADIUS`). With a group: the groups' begin must then skip `contactMemberBegin` (reading) |
| the coupling's step, the wheels | solo: `coupleStep`, `driveStep` (or `PERF_DRIVE_AHEAD`'s parts); member: `splitCoupleStep` (the members' partition of lanes; a group past `CPL_GS_VEHICLES` members or header bit 4 runs `cplIntegrate` with no `cplReactBy` before it, where `coupleStep` runs both) and `splitDriveStep` / `splitTyres` | `wide_aux` (a workgroup a vehicle: `coupleStep`, `driveStep`); with `PERF_WIDE_FUSED` on the list and hits kernels | built without a group. With a group (reading): the fused form cannot be taken, since step s + 1's tyres would read their hubs' accumulators before step s's contact levels add to them; and `splitCoupleStep`'s large-group branch must be shown the same as `coupleStep` or kept |
| the solo vehicles' frame-bounds check | `nodeUpdateSolo` / `nodeUpdateShapes` / `nodeUpdateCoupled` with `solo`: each node's position before its update against the allowed box (`gsVvBounds`: centre and half extents from `vvRec(v, VV_ALLOWED_LO/HI)` minus the origin, in workgroup memory), not for a retired node; every node again at the frame's end. Writes `gAcc[vvCtl(VV_OUTSIDE)]` by an atomic maximum | the node pass's thread, before its `nodeUpdate`: the box from the two records and the table's origin, a thread a node; the frame's last positions in the close pass | **built** (`explicit_beams_wide_nodes_vv`, `explicit_beams_wide_close_vv`): the same expression per node; the half sum times 0.5 is exact, so no rounding differs from the workgroup's stored copy |
| each member's contact box by step parity | `memberBox` after the member's node phase (and again after centred damping): atomic minima and maxima of `nf3(origin + P)` into `vvBoxKey(v, parity)`, the other parity's keys cleared for the next step, the allowed box checked (`vvReportOutside`); `contactMemberEnd` clears all twelve | the node pass's thread after `nodeUpdate` (and the centred nodes' pass): order-free atomics a node; the other parity's clear in the pass's per-vehicle part | not built. Reading: exact (minima, maxima); the fleet's front reads the keys after the members' steps, which the wide step keeps |
| the pending-beam reset of a member's triangle throttles | `VV_FLAGS.w`, set by the contact pass on a fast approach (`explicit_beams.slang` 10085, 11085; `explicit_beams_fleet.slang` 121): at the next step's throttle update every triangle's wait is zeroed and the triangle listed; cleared after the update (`contactMemberStep`, lane 0, s > 0), and in `contactMemberEnd` | the list kernel's throttle loop (it reads the flag per vehicle) and `wide_end`; `PERF_WIDE_GEO`'s `triangleFinishPeek` in the beams pass must then fill every triangle of a pending member | not built. Reading: exact, per vehicle |
| the members' pair-list state between steps | `memberListLoad` / `memberListKeep` / `memberListSave`: the list's state (`gsCount` past the buckets, `REUSE_WORDS` words) in `soloStepAt(v)` | the list kernel keeps the same words there (`soloStepSave`, `soloStepLoad`) | the same words already: nothing to host. `pairReuseBegin` at step 0 on both |
| the kept candidates' tracking of every node | `fkTrackVehicle` at the end of `memberRest` when the host sets `PERF_KEEP_TRACK_REST`; otherwise the scans' dispatch tracks (`fkTrack`, a thread a node) | not hosted: without the members' rest the host does not set the flag and the scans' dispatch tracks, one launch a step in a frame with a group | a schedule choice, no kernel work (the fleet's kept candidates are being changed in parallel: left to that work) |
| wheels on each other | `wheelsPartners` in a dispatch of its own a workgroup a group before each step's vehicles (`GroupOp::Wheels`): the staged reactions into the partners' accumulators, the rays | the same dispatch before the step's aux pass | not built: a fleet whose wheels see bodies stays on today's path |
| the node, beam, centred and list phases, the walk, the pairs, the hits | the member's or the solo vehicle's workgroup; a member's node phase always takes the accumulators (`hits` true) | `wide_nodes` (always `hits` true), `wide_centred_*`, `wide_beams`, the list kernel, the walk and pairs dispatches, `wide_hits` | the wide frame against the frame kernel is held bit for bit already with contact off (`explicit_beams_bitwise`, the matrix's one-car rows) |
| the surface pass | in the member's or solo vehicle's kernel, or as pipelines after the rests (§8.53) | the pipelines after `wide_hits` (§8.53) | built without a group |
| the frame's end | solo: `coupleEnd`, `pairReuseEnd`, the triangles' last throttle update, the last group breaks, the step count, the last bounds check; member: `splitGroupEnd` (`coupleEndMembers`, `contactMemberEnd`) | the list kernel's end, `wide_aux` at `steps`, `wide_end`, `wide_close` | built without a group. With a group: `splitGroupEnd` must skip what the wide end ran (reading) |
| the step parity | a member's own step uses its own count (`gMisc[v].w + s`); the groups' pass its first member's | the wide passes clear every vehicle's group slots by vehicle 0's parity: every vehicle must have run the same steps | holds while no vehicle sleeps (every vehicle steps every frame): vehicle sleep stays on today's path |

Nothing in the list reads another vehicle's workgroup memory; the two rows marked as
needing a proof before they move are the coupling's large-group branch and the order of
the tyres against the contact levels.

**Measured before building: what the wide passes cost these cars.** §8.54's figures for
cars apart with contact off (20 cars 9.07 ms a step at the median, 12.36 at p95) are a
live run's GPU time. The matrix's rows are one saved moment timed by wall clock
(encode, submit, wait), and that moment is at the run's p95. The same fixtures replayed
with vehicle contact off, so that the fleet runs the wide passes
(`EXTRA="--vv 0" run.sh perf=0 ROW`, Mac, bit-exact on), against today's frame with
contact on:

| row | contact on today (`Solo`) | contact off (the wide passes) |
|---|---|---|
| apart2 | 8.29 | 6.07 to 6.34 |
| apart8 | 13.61 | 9.92 to 10.55 |
| apart20 | 14.30 | 12.50 to 12.52 |
| apart32 | 15.34 | 14.64 to 14.83 |

So the most this part can give on the Mac is 14.3 → 12.5 ms at 20 cars (−12.5 %), 27 %
at 8, 4 % at 32: a workgroup a vehicle costs that GPU the same for 2 cars as for 32
(8.3 to 15.3 ms), the wide passes' work grows with the cars.

**Built: the frames without a group** (`FramePlanner::set_vv_wide`, `--ab-switch vvwide`
in `vehicle_lab` and in engine-cuda's `explicit_replay`; `RUBBLE_VV_WIDE=0|1`;
`explicit_plan/vv_wide.rs`, `explicit_beams_vvwide.slang`). The plan decides on the GPU,
so such a frame holds both schedules and each is sized there:

- `explicit_vv_gate` (one workgroup after the plan's passes, every frame): from the
  plan's group count (`vvArgs` word 3, which every snapshot carries) the wide passes'
  workgroup counts into a buffer of its own (`ArgBuf::Wide`, ten slots): each pass's
  count when the plan holds no group, else 0; and the vehicles' count for today's
  launches that run every frame (`SelfMember`, `SoloRest`) when it holds one, else 0.
- Every wide launch is sized from that buffer, in the order of a wide frame with the
  reused pair list. The node and close passes are two new modules, the wide ones with
  the bounds check ahead of them (`explicit_beams_wide_nodes_vv`,
  `explicit_beams_wide_close_vv`); every other launch is a module the hosts had.
- With self split and vehicle contact (no surface contact, or CUDA's surface pipelines)
  the walk, the pairs and the surface pipelines are today's own dispatches, and one
  launch of each serves both schedules: a step is the wide passes up to the list
  kernel, the vehicles' step, the walk, the pairs, the wide hits, then the rests, the
  surface pipelines and the groups' passes. With the surface pass in the kernels (wgpu's
  default) the wide frame brings its own walk, pairs and surface pipelines ahead of each
  step's members' dispatch, the prep and the plan sized by the gate; their argument
  slots (the self split's and the surface pass's) are zeroed at the frame's start by the
  gate's module again, since a slot is otherwise cleared by the step before it, which
  gets no workgroup in a frame with a group. Zeroing every slot at a frame's start
  changes nothing for a frame that writes them.
- The steps' parameter copies carry the wide passes' flags (`PERF_WIDE_LIST`, the fused
  aux work where a wide frame takes it, `PERF_HIT_SLOT_MAP`); no kernel of the split
  group path reads them (the groups' pass reads `lvSlots` from other copies when its
  tests are split, which this schedule requires).
- No launch binds as storage the buffer it is sized from (held by the schedule's tests).

**A launch is not free when the GPU gives it no workgroup.** The brief for this part
took §8.54's "334 of 378 dispatches are empty" as costing nothing; that was GPU time.
With both schedules in every frame (the first build), every row with a group was slower
by wall clock, bit for bit:

| | Mac (Metal), on / off, ms a frame | T4 (CUDA) |
|---|---|---|
| pile2 | 10.48 / 9.28, +13.0 % | 12.91 / 11.53, +11.9 % |
| barrier | 7.66 / 7.06, +8.5 % | 11.78 / 10.62, +10.9 % |
| side | 9.08 / 8.47, +7.2 % | 14.78 / 13.53, +9.3 % |
| headon | 12.56 / 11.84, +6.1 % | 18.65 / 17.41, +7.1 % |
| clash2 | 19.11 / 17.46, +9.4 % | 22.20 / 20.90, +6.2 % |
| clash20 | 29.62 / 27.73, +6.8 % | 45.08 / 44.02, +2.4 % |
| mountain20_t12 | 42.21 / 40.74, +3.6 % | 79.21 / 78.08, +1.4 % |

That is 0.6 to 1.9 ms a frame for 200 to 700 launches without a workgroup: 2 to 5 µs
each on the Mac (wgpu's encoding and Metal's, not the GPU), and on CUDA an indirect
launch always runs its resident blocks to read a count of zero. The same holds for
today's frames: a frame without a group already carries about 340 launches of the
groups' path that get no workgroup, which is the difference between this part's result
and the contact-off column above (13.6 against 12.5 ms at 20 cars).

So a frame holds both schedules only when the host expects no group: the host reads the
plan's group count back without waiting (wgpu: four bytes copied after a frame's passes
and mapped asynchronously, every fourth frame; CUDA: at `wait`, where the stream is
idle) and gives it to the planner (`FramePlanner::set_vv_wide_seen`). After a plan
without a group the next frames hold both; until then, and after a plan with a group,
a frame is today's, which serves a plan without a group as before. Either frame is
right for either plan, so the hint costs or saves time and cannot change a result: a
frame that holds both schedules while the plan forms a group runs today's launches
(sized by the gate), and `explicit_vv_wide_gpu` holds such frames bit for bit. This is
the same kind of read as the levels' slots (`level_want`).

**Results** (2026-11-22, bit-exact on, ms a frame of 34 steps; `run.sh vvwide ROW...`, 10
frames, 3 rounds, the switch forced on whatever the fleet's size). Every one of the 37
rows ends bit for bit on both machines, every state word and every coupling word.

| row | Mac (Metal), on / off | T4 (CUDA, `ENGINE_CUDA_FMAD=0`), on / off |
|---|---|---|
| apart2 | 7.59 / 8.33, −8.9 % | 10.75 / 12.91, −16.7 % |
| apart4 | 10.97 / 12.86, −14.7 % | 15.97 / 18.19, −12.2 % |
| apart8 | 11.52 / 13.63, −15.5 % | 18.20 / 19.64, −7.3 % |
| apart16 | 13.69 / 14.16, −3.3 % | 23.18 / 23.98, −3.3 % |
| apart20 | 13.59 / 14.39, −5.6 % | 25.97 / 26.87, −3.4 % |
| apart32 | 15.89 / 15.32, +3.8 % | 33.64 / 35.68, −5.7 % |
| apart1, drive, tumble, wall, bricks, rocks, truss, pole, wall_s, bricks_s (one vehicle: contact not active) | −4.8 to +2.2 % | −0.1 to +0.2 % |
| pile2, pile8, pile32, headon, side, barrier | −0.8 to +0.6 % | −0.1 to +0.2 % |
| clash2 to clash32 | −0.5 to +0.3 % | 0.0 to +0.1 % |
| mountain20_t2, t4, t7, t12, t20 | −0.6 to +0.1 % | −0.1 to +0.3 % |
| derby20_t2, t5, t10 | −0.1 to +0.1 % | +0.1 % |
| pile8_s | −0.1 % | 0.0 % |
| verdict | PASS (37 rows) | PASS (37 rows) |

The rows with a group run today's frame (the host has read a plan with a group back),
so they are the same launches as before plus the read. On the Mac the cars-apart rows
were measured five more times on a GPU other sessions shared: apart2 −6 to −19 %,
apart4 −8 to −15 %, apart8 −12 to −20 %, apart16 −3.3 to +1.3 %, apart20 −5.6 to
+4.5 %, apart32 0.0 to +10.3 % (past the gate's 5 % in three of five runs). On the T4
every count is faster.

The defaults, before → after, mean of each frame's best round (worst frame), each
`run.sh --baseline` a run of its own (`RUBBLE_VV_WIDE=0` before; the two T4 runs were
two containers, whose rows off this path differ by 1 to 6 % between them, so only the
rows this path changes are listed):

| row | Mac before → after | T4 before → after |
|---|---|---|
| apart2 | 8.31 (8.46) → 7.53 (7.91) | 13.20 (13.38) → 10.92 (11.15) |
| apart4 | 12.84 (14.21) → 10.31 (12.56) | 18.41 (20.44) → 15.88 (17.96) |
| apart8 | 13.54 (14.84) → 11.51 (13.30) | 20.49 (22.66) → 18.28 (20.32) |
| apart16 | 14.07 (15.34) → 14.07 (15.37), not taken | 26.02 (28.81) → 23.17 (25.27) |
| apart20 | 14.24 (15.64) → 14.26 (15.62), not taken | 29.54 (32.00) → 25.96 (27.74) |
| apart32 | 15.22 (16.54) → 15.24 (16.60), not taken | 39.70 (42.82) → 33.59 (36.31) |

**Defaults.**

- *CUDA*: on (`vv_wide_default(true)`). Passes the gate on every row.
- *wgpu, natively*: on for a fleet of at most 8 vehicles (`DEFAULT_MOST_VEHICLES`,
  `explicit_vv_wide_gpu.rs`), where every measured row is faster; off above, where it
  is a tie at 16 and 20 and slower at 32. `RUBBLE_VV_WIDE=1` any fleet, 0 none. It acts
  only where the host's wide passes with the reused pair list are on (the lab's
  options; the World's own defaults leave them off).
- *wasm*: off. Chrome accepts the three new modules; its bit-for-bit A/B on the lab
  page (AGENTS.md rung 3) was not run, and a browser's compiles ahead do not list them.
- *engine-metal*: its host replays the planner's commands and never turns it on.

**Proofs** (commands from `engine/`).

- `tools/scenario-matrix/run.sh vvwide ROW...` on the Mac, 37 rows: bit for bit, PASS.
- `ENGINE_CUDA_FMAD=0 tools/scenario-matrix/run.sh --cuda vvwide ROW...`, 37 rows
  (mountain20_t20 and the derby rows among them): bit for bit, PASS.
- `ENGINE_CUDA_FMAD=0 tools/scenario-matrix/run.sh --cuda --base 25ea30b4 --head
  a973e8cf`, the 12 default rows as two builds: every end state word for word, −0.2 to
  +0.2 %, PASS.
- `cargo test --profile fast -p engine-webgpu --test explicit_vv_wide_gpu` with
  `RUBBLE_BIT_EXACT=1` (new, 6 tests): three cars apart with and without surface
  contact, 24 frames; a pair closing head-on and one alone, 40 frames, with and without
  surface contact (24 frames held the wide schedule, 18 had a group, 3 held the wide
  schedule while the plan formed a group); a vehicle falling out of its frame bounds
  reported in the same words from the solo frame and from the wide passes.
- `cargo test --profile fast -p engine-exec` 28 (three new: the gate's slots are the wide
  plan's counts; each path's launches in order; no launch binds the buffer it is sized
  from), engine-webgpu `explicit_plan_modules` 1, `explicit_loop_cap` 6: pass.
- The suites that hold the GPU to the CPU reference, bit-exact, with the host's
  defaults and again with the wide frame forced where the plan allows it
  (`RUBBLE_VV_WIDE=1 RUBBLE_WIDE=1 RUBBLE_WIDE_LIST=1 RUBBLE_GROUP_SPLIT=1
  RUBBLE_SELF_SPLIT=1`), `--profile fast`, each under the GPU lock: `fleet_contact_gpu` 1,
  `group_split_gpu` 3, `self_split_gpu` 8, `explicit_contact_gpu` 13,
  `explicit_surface_fast_gpu` 6 pass both ways. (How many of their frames took the wide
  schedule in the forced runs was not counted; `explicit_vv_wide_gpu` counts its own.)
- Chrome: `EXPLICIT_WGSL_MODULES_ONLY=1 node --experimental-websocket
  explicit-wgsl-check.mjs`: all 158 modules. `tools/build-time/compile_cost.py --check`:
  0 failing (the new modules 0.0 to 0.5 s). `tools/build-time/check.py`: the two
  `vehicle_lab.rs` lines of the base, nothing new. `explicit_beams.slang` keeps its
  11,645 lines: its include of the fleet's file moved to the end of
  `explicit_beams_split.slang`, and of the base's 440 generated WGSL modules every one
  is byte for byte the same (three are new).
- Not added: budget lines for the three new modules in
  `tools/build-time/inline-budget.txt` (116, 300 and 34,203 inlined lines); that check
  already fails at the base on 15 modules of §8.53 without a line.

**Left on today's path, and why.**

- *Every frame with a group* (the map above: nothing of it is built).
- *A fleet whose wheels see bodies* (`Switches::wheels_bodies`): the wheels' dispatch is
  a group's; its place in a wide step is not built.
- *Vehicle sleep*: the wide passes need one step parity for the fleet.
- *A fleet past 8 vehicles on wgpu, a browser*: the defaults above.

**What the frames with a group would gain, measured before building them.** The same
fixtures with vehicle contact off run every car's own step on the wide passes (the
surface pass as pipelines) and nothing of vehicle contact. Against today's frame with
its own-step dispatches named (Mac: §8.54's table, the members' step holds the surface
pass; T4: `PROFILE_SPLIT=1 run.sh --cuda perf=0`, the vehicles' step, the walk, the
pairs, the rests and the surface pipelines):

| row | Mac: today, of it the members' step | Mac: own steps on the wide passes | T4: today, of it the cars' own dispatches | T4: own steps on the wide passes |
|---|---|---|---|---|
| clash20 | 25.2, 17.3 | 16.3 | 44.3, 31.4 | 28.0 |
| mountain20_t12 | 42.8, 23.8 | 22.3 | 78.0, 46.8 | 44.0 |
| mountain20_t20 | 66.9, 26.0 | 23.9 | 125.9, about 50 | 46.9 |
| derby20_t5 | 29.8 | 19.1 | 56.7 | 32.7 |
| pile8_s | 25.1 | 13.3 | 32.2 | 21.9 |
| pile8 (no surface contact) | 12.4, 9.1 (timed apart: each inflated) | 5.4 | 21.2, 15.7 | 10.2 |
| pile32 (no surface contact) | 37.7, 13.0 (timed apart) | 10.9 | 77.2, 35.4 | 30.3 |

On the Mac the 20-car rows with surface contact would gain at most 1.0 to 2.1 ms a
frame (3 to 4 %): the wide frame runs the surface pass as pipelines, which at 16 to 20
cars costs that GPU about what the pass inside the members' workgroups does (§8.53's
finding for the vehicle-contact part), so the cars' own steps across the GPU are not
"far less" there. On the T4, where §8.53 already took the surface pass out of the
vehicles' step, the step dispatch is 12 to 33 % of these frames and the wide passes
would save 3 to 5 ms: 7 % of clash20, 4 % of mountain20_t12, 2 % of mountain20_t20,
and more only without surface contact (pile8 about 26 %, pile32 about 7 %; on the Mac
pile8 at most 30 %, pile32 6 %). These are upper bounds: with contact off the wide
frame of a fleet without surface contact also fuses its aux work, which a frame with a
group cannot (the map's second row). The frames
are elsewhere: on the T4's mountain20_t20 the fleet's tests are 59 of 126 ms, the
levels 10, the surface pass's pair path (its find, listing, prep and plan) 23.

So the frames with a group were not built: the map above is what they need, and the
table is what they would return. In order of return: the fleet's tests on the mountain
(§8.55), the surface pass's pair path, the launches a step (a launch costs the Mac's
host 2 to 5 µs with or without a workgroup, and a frame has 850), and only then the
members' step.

**Not verified.** Chrome's bit-for-bit A/B on the lab page and whole 40 s runs (rungs 3
and 4); engine-metal's parity suite (its host is unchanged); `explicit_beams_bitwise`
and the World's suites (`explicit_vehicles`, `vehicle_lab_reset`); the T4's rows with
wgpu through Vulkan. The Mac's timings are from a GPU other sessions were using through
the lock: deltas within one process are paired, the two baseline runs are not.
