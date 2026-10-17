# AVBD Slang / WebGPU progress and worker handoff

> **Continuation update (2026-10-14, after 20:20 UTC):** The experiment described as unfinished in the snapshot below has now been measured and **rejected**. It costs **12.70 ms** versus fresh production **11.91 ms**: narrowphase increases **2.03→3.74 ms**, exceeding removal of the **1.11 ms** warm pass. All45 candidate GPU tests passed, including newly added cold-contact force/penalty assertions; no full-scene accuracy claim was made. The seven production files were restored, candidate wrapper/generated entry removed, and useful cold-contact assertions retained. Browser unit tests pass218/218. Restored production GPU tests pass44/44 (8.15s), and69 production modules were re-exported successfully. The live ring was then opened for the user; stop or navigate away from that live workload before any further GPU timing/tests. See `avbd-ring-inline-narrowphase-trial-2026-10-14.json` and `.patch` for the completed experiment. The sections below preserve the earlier handoff snapshot; **do not resume or adopt that candidate as if it remains in the source tree**. Next optimization should address measured collision/contact traffic with a materially different design. Goal remains incomplete.

**Snapshot:** 2026-10-14, after the 20:13:42 UTC production ring benchmark and subsequent experimental shader export.
**Repository:** the rubble repository root
**Branch:** `feat/avbd-attempt-2`
**Git HEAD:** `501485f4318fe736e2d0610bd6268a4590c67408`
**Status:** substantial uncommitted implementation and test changes; an explicitly opt-in optimization experiment is unfinished. **The ≥120 physics FPS / three-avbd parity goal is NOT complete.**

This document is the standalone handoff. It includes the essential findings, measurements, implementation contracts, test results, open issues, and continuation procedure. Repository paths below are relative to the repository root unless otherwise stated. Supporting artifacts contain additional raw evidence, but no prior chat is required to understand the next steps. Historical files and early entries in `engine/DECISIONS.md` sometimes describe superseded implementations; this snapshot distinguishes current behavior from those earlier states.

## 1. User objective and priorities

The user wants the Slang → WGSL WebGPU implementation to match or beat the pinned `inspiration/pinned/three-avbd` implementation in correctness, scene scale, and speed:

1. First make the **full 110k brick ring** correct, stable, and capable of **at least 120 physics steps per second**.
2. Then pursue the **full 506k brick gables** under the same accuracy and performance principles.
3. Preserve the columns fixes and existing Metal physics oracles.
4. Add high-quality automated regression tests whose failures explain the violated physical or scheduling invariant, affected body/pair/component, expected versus actual results, and likely failure mechanism.
5. Use fair performance measurements. Do not reduce bodies, iterations, contacts, collision coverage, or the active workload to manufacture speed.
6. Do not declare the broader objective complete merely because a focused test, shorter scene, or isolated kernel meets its budget.

“100k ring” in conversational requests means the pinned **110,440-brick** scene, not a reduced 100,000-body substitute. The user explicitly requested a deep reference audit; that audit was completed and is summarized below. The latest user request is this handoff, so the current optimization was left inspectable rather than advanced into an unvalidated default.

## 2. Executive state

- The severe **~200–250 ms/step slowdown is absent** in current measurements. The latest production ring run is **11.89 ms/step ≈84.1 physics FPS**. Earlier recent production controls ranged approximately **11.41–12.08 ms**.
- The refreshed reference ring UI measured **9.99 ms/step ≈100.1 physics FPS** on the same machine/browser family. Neither current local result establishes 120 FPS.
- The target is **8.333 ms/step**. Relative to the latest 11.89 ms control, Rubble needs approximately **29.9% less time per step**, or **42.7% more throughput**. It is approximately **19.0% slower in step time** than the 9.99 ms reference result, though these last two readings are not a tightly bracketed final A/B/A.
- Ring scene construction now matches the pinned builder body-for-body, including handedness conversion, inertia, rolling-ball velocity, and ground dimensions.
- Ring480, canonical gables240, columns, and shared Metal oracles passed after the last adopted optimization. Those are bounded correctness results, not proof of exact long-term trajectory identity.
- Current ring defaults include persistent 64-byte hot / 32-byte cold contacts, cached solver pairs, stable color packing, physical adjacency ordering, incremental coloring, a bounded sparse tail starting at color 10, and cooperative color-prefix scanning.
- **Gables stays on canonical contacts by default.** Native compact/persistent gables has an unresolved drift failure; do not relax its thresholds or enable persistent gables by default.
- The remaining main performance opportunity is **collision/contact preparation**, especially the separate warm-start pass, rather than another blind primal arithmetic rewrite.
- An **opt-in narrowphase-integrated warm-start candidate** is present, compiles, and passes focused tests. It has **not been timed in the full browser ring**, has **not passed full-scene gates**, and is **not adopted**.

## 3. Exact machine, source, and measurement context

### Machine and software

- User machine: **Apple M3 Max, 128 GB**, macOS.
- Browser used for current measurements: Codex in-app browser, Chromium/Chrome **155.0.0.0**, WebGPU adapter reported as vendor `apple`, architecture `metal-3`.
- User restarted the Mac during the investigation. Both reference and Rubble recovered afterward. This is evidence of recovery, **not a diagnosis of the original slowdown**.
- Pinned three-avbd HEAD: `2bb1a318e9aad03640b4bce5b5d222484f0925f7`.
- The pinned checkout already has local changes in `package.json`, `pnpm-lock.yaml`, `src/avbd3d/bench-cases.ts`, `src/avbd3d/bench-scenes.ts`, plus untracked `settle-pyramid.html`. This work did not alter those files. Use actual working-tree source hashes, not HEAD alone, when reproducing a comparison.
- Rubble has many pre-existing modified/untracked files, including independent fracture work. **Do not broadly reset, clean, stage, or revert the repository.** No new commit was created for this handoff.

### Fair physics throughput protocol

Both implementations should run serially, reset from the same scene, with:

| Setting | Value |
| --- | ---: |
| Total ring bodies | 110,442 |
| Dynamic/awake ring bodies | 110,441 |
| Bricks | 110,440 |
| Dynamic spheres | 1 |
| Static ground bodies | 1 |
| Solver iterations per step | 4 |
| Physics dt | 1/60 s |
| Lead-in | 60 steps |
| Rounds | 2 |
| Untimed steps before each round | 3 |
| Batches per round | 5 |
| Steps per timed batch | 10 |
| Total timed steps | 100 |
| Reference-compatible wall statistic | Minimum of the two round medians |

The timed windows are steps **64–113** and **117–166**, counting completed steps from one. Raw batch times and both medians must be retained. Each batch queues physics and drains actual GPU completion; no render or diagnostic readback belongs inside the timed window. `bench=only` stops after the benchmark.

Separate concepts:

- **Physics throughput:** the batched wall-clock metric above.
- **GPU phase diagnostics:** ten later steps, 167–176, with extra compute-pass boundaries and per-step readback. These perturb execution. Medians need not sum to median total and are not the throughput window.
- **Live rendered FPS:** also includes rendering, UI/readback scheduling, and presentation. The live loop's per-frame readback remains a separate optimization opportunity.

All comparisons require complete current-step GPU dispatch coverage and zero overflow. Read awake/contact/manifold/color counts and color bucket coverage outside timing. A very fast collapsed scene is not a successful optimization. Prefer A/B/A or B/A/B with fresh scene resets and stable controls. Do not run native GPU tests, another live demo, or compilation concurrently with browser timing.

### Slowdown investigation and recovery

Before reboot, examples included reference **216.77 ms** and Rubble **227.20 ms**; several experiments in that period were unusable for speed claims. An independent browser/native dispatch-and-memory probe was added, using a simple increment shader with no physics or Slang. It validates every output word, including 8,388,608 words in a 32 MiB case.

During the slow period, 128 tiny dispatches cost about **0.19 ms**, whereas one full 32 MiB update cost **11.71 ms** in browser and **11.936 ms** in native wgpu/Metal. A 256-thread native variant did not fix it (**12.799 ms**). Browser large-workload batches varied **2.79–21.59 ms**. This argues against simple empty-dispatch overhead explaining hundreds of milliseconds, but does not establish a hardware fault, driver fault, or external GPU contention. No pre-incident historical baseline exists for that probe.

After the full reboot, serial measurements were reference **10.22 ms**, Rubble canonical append-order **19.99 ms**, and canonical physical-order **20.29 ms**. Later code optimizations reduced Rubble further to its present ~11–12 ms range. Do not confuse reboot recovery with subsequent code improvements.

The final handoff process check found **no running `ring_hold`, `columns_hold`, `gables_hold`, `oracle_accuracy`, `avbd_slice`, `wgsl_export`, `slangc`, `cargo`, `engine_webgpu`, or `gpu_dispatch_probe` process**. A separate Chrome GPU process had been active in earlier checks, but CPU percentage is not GPU utilization, and causation was not established. Do not blame it or kill unrelated apps based on that observation.

## 4. Latest production measurement and reference comparison

### Latest Rubble control

URL:

`http://127.0.0.1:8765/demo.html?scene=ring110k&bench=only&profile=detail`

Measured at **2026-10-14T20:13:42.822Z**. This used the production contact-persistence path, **not the new inline candidate**. It ran before the subsequent 70-module export.

- Wall: **11.890000000037253 ms/step**.
- Round medians: **12.070000000111758**, **11.890000000037253 ms**.
- Round 1 batch means, ms: `[11.82, 12.00, 12.07, 12.20, 12.34]`.
- Round 2 batch means, ms: `[12.09, 11.97, 11.89, 11.88, 11.79]`.
- Every batch exceeds the 8.333 ms budget.
- All 110,441 dynamics awake, four iterations, zero overflow.

Workload snapshots:

| Field | After lead-in | End of timed window |
| --- | ---: | ---: |
| Contacts | 1,184,245 | 1,174,316 |
| Broadphase candidates | 230,244 | 256,703 |
| Awake dynamics | 110,441 | 110,441 |
| Colors | 15 | 11 |
| Manifolds | 227,155 | 246,481 |
| Overflow | 0 | 0 |

Nonzero color bucket sizes:

- Start: `[38216,36641,21457,9780,2747,1068,297,121,63,28,9,6,4,2,2]`.
- End: `[36437,33636,20681,10740,4398,2539,1244,548,183,33,2]`.
- Remaining buckets through palette size 32 are zero. Each complete bucket vector sums to 110,441.

Later phase medians, ms:

| Stage | Latest control |
| --- | ---: |
| AABBs | 0.196608 |
| Grid / broadphase | 1.179648 |
| Reference-pose updates | 0.065536 |
| Contact hash clear | 0.065536 |
| Contact hash insert | 0.393216 |
| Narrowphase | 2.031616 |
| Separate warm start | 1.114112 |
| Graph / coloring / preparation | 0.851968 |
| Primal, all four iterations | 4.063232 |
| Dual, all four iterations | 1.703936 |
| Sleep-related work | 0.065536 |

Profiled GPU span was **11.730944 ms**, profiled completion **15.10 ms**. The completion number includes later diagnostic overhead and must not replace the 11.89 ms throughput result.

### Refreshed reference

The freshly displayed three-avbd ring result immediately preceding an earlier Rubble **11.80 ms** run was:

| Metric | Reference value |
| --- | ---: |
| Wall | 9.99 ms |
| GPU total | 9.90 ms |
| Collision | 3.54 ms |
| Adjacency | 0.26 ms |
| Coloring | 0.52 ms |
| Solve | 5.57 ms |
| Contacts | 1,169,844 |
| Colors | 11 |
| GPU allocation | 279 MiB |

These reference values are **UI-rounded to two decimals**; raw reference batches were not available. Its reported contacts come after later profiling, not necessarily the exact end of the timed window.

The paired earlier Rubble run at **20:07:38.693Z** measured **11.80 ms**, with primal+dual **5.701632 ms**, close to reference solve **5.57 ms**. Rubble collision preparation (grid, refs, hash clear/insert, narrowphase, warm) totaled approximately **4.9152 ms**, versus reference collision **3.54 ms**. Stage definitions are not perfectly identical, so use this to prioritize investigation rather than as exact accounting of every gap.

A separate downloaded reference report contains exact values **10.31000000005588 ms** ring and **26.320000000018627 ms** gables. That export was delayed approximately 48 minutes. It is **not the fresh 9.99 ms report**. Its gables result has 505,923 total bodies, 3 iterations, 4,102,693 contacts, 9 colors, zero clashes/overflow, GPU 26.607616 ms, and allocation approximately 1,218 MiB. Those gables timings are useful baseline evidence, not a contemporaneous final comparison against current Rubble.

**Performance implication:** warm start costs ~1.11 ms while hash clearing costs only ~0.07 ms. Prioritize repeated contact traffic/initialization, not a speculative hash-clear rewrite. Even eliminating all 1.11 ms would not by itself reach 8.333 ms.

## 5. Scene parity and correctness repairs already implemented

### Exact ring construction

- Radius 80, 40 courses, tier 4, ten rows at the base.
- Bricks full dimensions **1 × 0.5 × 0.5**, mass **0.25**, 2% circumferential gaps, radial pitch **0.52 m**.
- Rolling sphere radius **8**, density **10**, speed **30**, initial Rubble position `[0,8.5,-95.2]`, initial angular velocity `[3.75,0,0]`.
- Ring ground dimensions **360 × 1 × 360**, centered at zero, top Y=0.5.
- Reference uses a different up axis. Swapping Y/Z reverses handedness; yaw and axial angular velocity must change sign. Merely swapping coordinates caused physically incorrect behavior.
- `scene-builders.mjs` is tested against the actual pinned builder for every body, quaternion, mass, inverse inertia, velocity, and angular velocity. Ground and zero floor-contact rolling velocity are independently checked.
- Ring defaults: dt 1/60, gravity -10, 4 iterations, alpha .99, gamma .999, betaLin 10000, margin .01, friction .5, contact reuse enabled, sleep time sufficiently large to keep all dynamics active.
- Columns intentionally retain their separate deep static slab. Do not replace the ring ground with the columns slab or vice versa.

### Important fixed defects and invariants

1. **CSR prefix publication race:** in-place publication raced neighboring reads across workgroups. Finalize exclusive prefixes into the other scan buffer, then publish offsets/cursors in a separate dispatch. Tests cover all entries through 524,288 bodies. A maximum-height-only columns check was inadequate because upper bricks could remain while lower layers fell.
2. **Current-step dispatch coverage:** host clamps/previous CPU estimates could omit work. Direct dispatch now tiles across Y at 32,768 groups per row, with matching Slang flattening. GPU-generated indirect arguments cover current candidate/contact/manifold/awake counts. Browser batches copy previous GPU contact counts into the next uniforms before resetting counters. Reset seeds zero history. Do not reintroduce truncation to satisfy the 65,535 dimension limit.
3. **Warm-start corner identity:** quantized feature IDs can collide. Among equal feature IDs, match nearest anchor pairs so distinct support corners remain distinct. History generation, normal agreement, both halves, and reuse tags matter.
4. **Full support polygon:** retain up to eight clipped face vertices. Taking the first four biases support. Cross-product quaternion rotation preserves exact touching axes despite f32 quaternion-norm rounding. Exact sphere/box touch is included.
5. **Friction:** test raw tangent force against the cone before projection; projected force is always in-bounds. Ramp two independent tangent penalties. Preserve both Hessian rows during sliding. Earlier shared-penalty behavior caused widespread collapse and wrong impactor motion.
6. **Graph coloring:** run 32 JP rounds independently of resulting color count. Reject unfinished/out-of-range colors with overflow rather than packing an uncolored body into color zero and racing connected solves. GPU dispatch must cover newly created colors even inside ten-step batches.
7. **Large-body pair ownership:** small bodies own small→large pairs regardless of ID; only large→large uses ID ordering. Large bodies skip redundant small-grid scans. Endpoint fallback is static-only. This fixed duplicated sphere-last pairs and missed sphere-first pairs.
8. **Exact cell identity:** a hashed linked bucket must reject bodies from a different coordinate cell, even when neighboring cells alias to the same bucket. Geometry rejection alone does not prevent duplicate emissions.
9. **Bulk native scene upload:** individual staging writes stalled startup for 506k gables. Batch APIs upload current bodies, previous bodies, and colliders contiguously, preserving append IDs and associations.
10. **Color argument bounds:** a 64-thread launch does not imply 64 palette entries. Guard palette reads, preserve indirect work slots 64+, and launch full workgroups where cooperative shaders require them.
11. **Weak CAS:** WGSL weak CAS can fail spuriously. Retry on the same open-address slot and confirm ownership; advancing past an EMPTY hole breaks lookup. The old simple exchange replacement regressed Metal and was reverted. The historical weak-CAS theory alone did not explain all columns failures; CSR and support identity fixes were also necessary.

Mandatory phase order remains: **AABB → broadphase → narrowphase/persistence at step-start poses → CSR/color → body prediction/warm start → primal/dual iterations → velocity extraction → island sleep**. Moving prediction ahead of contact initialization changes C0 and previously exploded drops.

## 6. Adopted optimizations and evidence

These results were obtained at different points in implementation and machine state. Use each bracketed control comparison, not sums of historical gains.

| Change retained | Evidence / result | Important contract |
| --- | --- | --- |
| Block-scan CSR | Replaced repeated full-buffer prefix passes; full-scale prefix correctness passes | 256 bodies per 64-lane group; Metal must launch complete 64-thread groups |
| Incremental coloring with compaction | Full ring/gables gates pass; gables step240 only 285 of 505,922 bodies needed recoloring in one measured run | Reuse-only trial failed gables; compaction and immutable conflict snapshot are required |
| GPU-driven sleep dispatch | Skips eight no-op manifold-wide propagation passes when there are no sleep candidates | Moving-neighbor veto and current-step work args remain correct |
| Remove unused counter snapshots/statistics | Generated-IR tests prevent unnecessary atomic loads and global diagnostic contention | Keep required physics counters and overflow reporting |
| Plain primal topology loads; pose-only body stores | Removes immutable atomic loads and 84 logical store bytes/body/iteration | Construction resources remain atomic; non-pose bytes unchanged |
| Stable color packing | Deterministic body-index buckets; locality and coverage tests | Early ~225 ms A/B was contaminated, not proof of speed |
| Physical adjacency ordering | Stable neighbor/pair order across append permutations, joints, compounds, very long rows | No independently established speed gain; retained reproducibility contract |
| Compact 64-byte solver points | Compact/canonical/compact **17.93/19.98/17.90 ms** after reboot | Initially kept canonical persistence and extra scratch; superseded ring default by persistent layout |
| Clipping arrays passed by reference | New/old/new **15.12/17.73/14.91 ms**; NP ~4.52→~2.0 ms | Preserve all eight vertices; emitted WGSL pointer contract matters |
| 32-byte cached solver pair | Cached/control/cached **14.78/14.97/14.78 ms** | Cache current body IDs/range/sphere flag/normal/friction once; adds 64 MiB at ring capacity |
| Persistent hot/cold contacts | Persistent/compact/persistent **12.76/14.75/12.71 ms**, retry 12.69 | Two 64-byte hot halves plus two 32-byte cold halves; avoid repack/final copyback; saves 256 MiB versus preceding ring layout |
| Early linked-grid ownership filter | Optimized/original/optimized **12.26/12.74/12.32 ms**, grid ~1.64→1.18 ms | Bit31 stores awake-dynamic eligibility; mask IDs; retain lower-ID sleepers |
| Bounded sparse color tail from color10 | Candidate/control/repeat **11.86/12.19/11.88 ms** | One 64-thread group, eight-body chunks, current GPU color count, ordered buffer/workgroup barriers |
| Cooperative color tile-prefix scan | Candidate/control/repeat **11.50/11.89/11.61 ms**, final earlier control 11.43 | One full 64-thread group per color; preserves exclusive prefix and stable order; packing ~0.46→0.07 ms |

Current ring production defaults are persistent contacts, physical adjacency, stable packing, incremental coloring, sparseStart=10, parallel prefix. Browser controls retain canonical/compact storage, append ordering, atomic packing, full separate-color scheduling, and serial prefix for A/B tests. Native sparse scheduling remains explicit via environment variable. Metal keeps its existing serial color-prefix host path.

## 7. Experiments rejected or left opt-in, and why

A passing focused test is necessary but does not make a slower or neutral variant worth adopting.

| Experiment | Observation / disposition |
| --- | --- |
| Simple exchange hash claim | Broke shared Metal behavior; reverted. Reference uses displaced-occupant carry, which is not equivalent to overwriting a slot. |
| Contact/support reduction shortcuts | Fewer vertices or aggressive shallow-edge shortcuts could look faster but broke support/oracles. Removed; full geometry is mandatory. |
| Reuse-only incremental colors | Gables unhit maximum drift 2.045 m; compaction required. |
| Manifold or fused warm scheduling | Manifold/contact/manifold **17.98/17.80/17.93 ms**; fused/contact/fused **17.53/17.52/17.49 ms**. No clear gain; variants stay opt-in. |
| Morton upload ordering | Morton/construction/Morton **17.63/17.78/17.74 ms**. No material gain; construction order default. Identity-safe mapping and tests retained. |
| Separate contact-reuse pass | Covered ~44% of candidates but no material wall gain; removed. |
| Direct compact-point primal math | Neutral; reverted. |
| Tangent basis hoisted per manifold | **12.81 vs 12.70 ms** control; primal unchanged. Reverted. |
| Persistent fused manifold warm start | **13.44 ms**, warm ~1.51 vs 1.05 ms. Keep separate per-contact persistence. Focused regression retained. |
| Counting-sort grid prototype | Scan/membership tests passed, but additional work did not beat filtered linked grid. Removed. |
| Disjoint reduction scratch / fewer barriers | **12.26 vs 12.29/12.30 ms**, primal unchanged; no established gain. Reverted. |
| Lane tuning | Fewer lanes **12.51 ms**, primal 4.46; four lanes for large colors **12.19**, identical to restored control. Reverted; stronger lane tests retained. |
| Residual triple-product factoring | **12.23 vs 12.19 ms**, primal worse. Reverted; independent moving-contact oracle retained. |
| Per-body motion cache | **12.47 vs 12.19 ms**, primal/dual both increased. Removed allocation/routing. |
| Flat point traversal | First exposed a stale manifold cursor after a joint (2.1 mm pose error). Corrected version still **12.93 vs 12.16 ms**, primal 4.85 vs4.13. Reverted; mixed-edge tests retained. |
| Packed/scalar 21-entry matrix | Packed 12.07, zero-init11.98, scalar11.82 vs restored11.87 and recent11.80–11.88. No reliable gain. Reverted; oblique dense oracle retained. |
| Slang optimization levels | Levels1/2/3 produced byte-identical probed raw WGSL; no flag change adopted. |
| Hoist current-body motion / fused three-row accumulation | **11.97 / 11.87 ms** vs restored11.87; second slightly changed trajectory. Reverted. |
| Split cold/reused warm passes | **11.71 / 11.70 ms**; forced combined-helper inlining11.43 vs restored11.41. No gain. Removed. |
| Force-inline contact loader | **11.46 ms**, unused calculations still in emitted WGSL. Reverted. |
| Identity-only contact header helper | Removes apparent unused hash loads but **11.43 vs11.41 ms**. No demonstrated runtime gain; reverted. |
| Sparse tail starts at color8 | **11.95 ms**, increasingly serializes larger buckets. Keep color10. |
| Subgroup primal reduction | **12.21 vs fresh restored12.08 ms**, same workload; prior11.44 control too old for clean attribution. Reverted all four source changes. |

The subgroup trial used actual subgroup ticket allocation, did not assume workgroup-local IDs map to subgroup lanes, preserved the 4/2/1 reduction tree, and fell back to shared reduction for small subgroups. Native advertised sizes4–64, browser32. It required explicit unsigned shuffle intrinsics and a Naga24 `enable subgroups` parsing shim. All44 tests passed, but speed did not improve. Its reproducible patch is saved; do not redo it as an untried obvious win.

## 8. Current unfinished experiment: inline warm initialization in narrowphase

### Hypothesis

The separate persistence pass revisits ~1.2 million contacts and costs ~1.11 ms. Reference narrowphase initializes/warm-starts points while generating or reusing them. Rubble's candidate calls the existing contact persistence helper before the first current-half store, potentially removing repeated loads/stores and one pass.

This is **different from earlier fused persistence experiments**, which combined warm start with compact packing after narrowphase.

### Present state

- Implemented in the working tree, **explicitly opt-in** using `persistence=narrowphase`.
- Default production behavior remains `persistence=contact`.
- **45/45 GPU/shader library tests passed**, 0 failed, test execution10.26s after a44.42s build.
- **252/252 browser unit tests passed**, 0 failed, for the three benchmark/profiling/scheduling files.
- Shader export completed successfully: **70 WGSL modules**, including `narrowphase_initialized_persistent.wgsl`.
- **No candidate browser timing yet.** The visible11.89ms result is production, not this candidate.
- **No full ring480/gables/columns/Metal gate has been run for this candidate.** Earlier full-scene results belong to the adopted baseline.
- No native host environment switch selects this candidate for `ring_hold` yet. The focused GPU fixture dispatches the new entry explicitly.

### Exact source changes

New file `engine/slang/kernels/initialized_narrowphase.slang`:

```c
#define ENGINE_COMPACT_SOLVE 1
#define ENGINE_PERSISTENT_CONTACTS 1
#define ENGINE_INLINE_NARROWPHASE_WARM 1
#define narrowphase narrowphase_initialized_persistent
#include "world_step.slang"
```

Other changed files and purpose:

1. `engine/slang/kernels/world_step.slang`
   - `emitContactManifold`: construct each contact, call `persistContact(cStart+k,c,false,ba,bb)` under the new define, then `STORE_NEW_CONTACT` with current owner.
   - `tryReuseContacts`: load prior contact, call the same helper with `alreadyReused=true`, then store in current half; baseline still uses its reuse marking path.
   - `persistContact`: suppress its own two current-half store sites under the new define, returning the initialized value for the caller to store. Baseline stores remain.
2. `engine/crates/engine-webgpu/build.rs`
   - Add `narrowphase_initialized_persistent` to world entries; route it to the new wrapper for SPIR-V/WGSL source generation before the generic persistent suffix handling.
3. `engine/crates/engine-webgpu/src/gpu.rs`
   - Permit loading the explicit candidate entry for persistent mode, while retaining generic default aliases.
4. `engine/crates/engine-webgpu/src/gpu_tests.rs`
   - Extend `mixed_contact_reuse_fixture` with an `initialized` boolean.
   - New test: `narrowphase_initialization_preserves_mixed_pairs_and_exactly_once_warmstart`.
   - Dispatch candidate NP and skip separate persistence for that case; prior cases use false.
5. `engine/crates/engine-webgpu/demo.html`
   - Parse `inlineNarrowphase` from query.
   - Reject candidate unless `scene=ring110k` and persistent contacts selected.
   - Route generic narrowphase to the candidate entry.
   - Skip separate persistence dispatch; keep diagnostic phase boundary, so warm should be empty.
   - Label benchmark metadata `persistence:"narrowphase"`.
6. `tests/persistence_schedule.test.mjs`
   - Exercise `persistent-inline` across profiling/reuse/graph/sparse/prefix configurations.
   - Require no second persistence pass, correct pair preparation before solve, correct alias/storage ABI, and rejection of unsupported scene/storage combinations.
7. `tests/demo_profiling.test.mjs`
   - Supply the new false-by-default mode in the benchmark fixture.

All test paths in items6–7 are under `engine/crates/engine-webgpu/`.

### Known correctness risk before adoption

Baseline persistence runs after all narrowphase wakeups and reads resulting body flags. Inline persistence can see a pair's pre-wakeup flags; another pair may wake one endpoint later. This can change whether warm decay/initialization is applied to a formerly sleeping pair. **Do not generalize to sleeping scenes without resolving this ordering contract.**

The experimental browser guard confines it to the persistent ring, whose current parameters keep every dynamic awake for the measured/tested horizon. This is not a general proof that any future ring configuration cannot sleep. The candidate must either enforce its all-awake precondition robustly or gain a tested wake-safe implementation before broad adoption.

### Existing candidate fixture coverage and missing follow-up

The mixed-history fixture has67 candidates crossing the64-thread boundary; cold, reused, generation-invalid and separated pairs; spheres and boxes; poisoned tail candidate; both contact halves; full feature bit31/generation; exact current owner tags; consumed transient reuse tags; distinct support corners; independent tangent penalty decay exactly once; and unchanged previous history.

Recommended next test enhancement, **not yet implemented**: cold pairs (`id % 5 == 0`) should explicitly have zero normal/tangent forces, `kN = max(1,m_min/dt²)` (fixture dynamic mass1 versus static ground), and both tangent penalties1. This catches omission of initialization even if geometry/counts look right. Keep diagnostic actual/expected values. Consider a mixed sleeping/waking pair fixture before expanding scope.

### Reversible snapshot

Seven-file pre-experiment backup exists under `/tmp/rubble-before-inline-narrowphase/`, mirroring repository paths for all changed files above except the new wrapper. It contains the adopted baseline, including the44th hash regression. Temp directories may disappear; a durable unified patch and before/candidate SHA256 map were also saved:

- `engine/docs/avbd-ring-inline-narrowphase-in-progress.patch`
- `engine/docs/avbd-ring-inline-narrowphase-in-progress-2026-10-14.json`

The patch includes the new wrapper and latest252-test scheduling changes. It is against the pre-experiment working tree, **not against Git HEAD**. If discarding, first use `git apply --reverse --check` against that patch and inspect scope. Only reverse it if no later edits conflict. Never use `git reset --hard` to restore this baseline. Re-export production69 modules afterward, and remove only the obsolete generated candidate file if desired; the exporter does not clean stale filenames automatically. Verify `manifest.json` and shader routing rather than relying on directory file count.

## 9. Automated regression coverage and latest results

### Focused GPU / shader suite

Current candidate tree: **45 pass**. Baseline immediately before candidate: **44 pass**.

Coverage includes:

- Broadphase unique ownership, exact hash-cell filtering, sleeping endpoints, large-body order, and independent CPU pair sets. The larger lattice fixture checks **6,084 expected pairs across641 bodies**, bucket membership, and tail canaries.
- Complete current-step work arguments, dispatch dimensions, palette bounds and work-slot preservation.
- CSR exclusive prefixes through524,288 bodies, ragged boundaries, zero degrees, sentinels, and physical sorting including65,537-contact rows.
- Stable packing and both serial/cooperative color prefixes at ring/gables scale, partial tiles, empty colors, zero-body reset and padded invocations.
- Color history repair, conflict rejection, sparse tail dependency ordering across chunks, and buffer visibility.
- Independent dense f64 Newton/contact equations across canonical/compact/persistent layouts, endpoints, all1/2/4/8 lanes, partial workgroups, negative quaternion hemisphere, spheres, static/sleeping neighbors, mixed joints and contacts.
- Moving oblique contact coverage: **288 primal and144 dual cases**, independent row-wise Jacobians, matrix lever-arm rotation, cone projection and dense Gaussian elimination.
- Contact reuse/corner identity, both halves, cold/generation-invalid pairs, ragged manifold ranges, exact-once warm decay, hot/cold ABI, cached pairs, dead points, and final-only copyback for compact mode.
- Exact touching and eight-point clipped support geometry.
- Independent friction penalty ramp/cone behavior.
- Bulk body upload preserving offsets/history/associations.
- Generated-WGSL validation, bounded loops, no forbidden atomic snapshots/diagnostic buffers, topology load types, pose-store fields, expected clipping pointers and compact/persistent buffer bindings.

The44th retained test is `contact_hash_preserves_colliding_pair_heads_across_dead_gaps_and_history_halves`:32 keys share a bucket, duplicate blocks contain dead gaps, irrelevant payloads include poison, both layouts/halves are exercised, and insertion repeats four times. It checks unique entries, earliest live head, reachability before EMPTY, and unmodified history. Mutation omitting head5 failed for pair `(1,1026)`, actual head249 versus expected5, explaining warm-start support loss. Mutation removed.

Other demonstrated test sensitivity: omitting angular motion fails the independent moving-contact oracle; omitting the yz Hessian term fails the oblique basis with approximately1.92mm position error; stale manifold traversal after a joint failed by2.1mm. These tests are stronger than implementation-text mirror checks.

### Browser unit suite

Current candidate: **252 pass** across `demo_profiling.test.mjs`, `physics_benchmark.test.mjs`, and `persistence_schedule.test.mjs`. Baseline before candidate:218 pass.

These execute shipped encoder/mode-selection code and protect dependency order, exact-once persistence, shader/storage ABI selection, profiling attribution, zero extra pass boundaries in throughput mode, GPU draining, raw samples, readback cleanup, counter/bucket reads outside timing, full-scale workload rejection, and candidate query guards.

Scene-builder and spatial-order tests are separate from that252 count. They compare actual pinned builders/order routines and preserved upload/readback identities. Do not describe the252 run as a fresh rerun of every JavaScript test file.

### Latest full-scene baseline gates, after adopted cooperative prefix

These gates passed before the unfinished inline experiment; thresholds were not loosened.

| Gate | Result / checkpoint |
| --- | --- |
| Native persistent ring480 + sparse10 | PASS; mean flank drift1.1497m, ball `[-0.035727184,8.4884205,105.45146]`, contacts1,031,897 |
| Native columns suite | Four tests PASS; 100k awake step180 maxY9.8998, drift0.1017m, KE1.0531, max speed0.0081, contacts400,000, awake100,000, overflow0 |
| Native canonical gables240 | PASS; unhit mean/max drift0.118626/0.330369m; contacts3,903,667; awake505,922 |
| Native canonical gables balls | `[-47.914364,4.7769637,-73.88008]`, `[47.994953,4.7769713,-73.753204]` |
| Browser ring480 + reset, 18:32:37 UTC | PASS; mean flank1.181787461m, ball `[-0.030252265,8.469187737,105.191246033]`; contacts1,037,330; candidates273,773; awake110,441; colors13; manifolds259,920; overflow0 |
| Browser canonical gables240 + reset, 18:33:10 UTC | PASS; unhit mean/max0.117217656/0.326606750m; contacts4,060,011; awake505,922; colors9; manifolds981,491; overflow0 |
| Browser canonical gables balls | `[-48.019500732,4.776892662,-73.832588196]`, `[48.073787689,4.776981831,-73.761627197]` |
| Shared Metal checks |18 PASS: library3, avbd_slice7, columns_wgpu_parity3, oracle_accuracy5; 100k position delta approximately0.000213m |

Ring limits: all110,441 dynamics remain awake; first-frame contacts1.25–1.32M; all poses/velocities finite; overflow0; valid contact coloring. For flanks with initial `abs(x)>40`, max vertical drift<0.5m through120 steps, mean<0.2m through120, <0.5m at240, <1.6m at480. Ball Y stays7.5–9.5m and final Z>90m. These allow the reference's later impact propagation, not widespread early collapse.

Gables limits: all505,922 dynamics awake; initial3,916,802 contacts; all-brick max drift<0.3m through20; at240 unhit mean<0.2m and max<0.6m; reference-based ball position bounds±1.5m; finite values, zero overflow, valid native coloring. Browser checks also exercise reset.

### Known unresolved gables failure

Native persistent gables240 has **five unhit bricks above0.60m**, maximum **3.00113m**, worst body **318697**, with10 colors in the recorded failure. Native compact gables also exposed drift reproduced with the prior clipping code, so the pointer-clipping optimization did not uniquely cause it. Browser persistent gables passed; native canonical gables passed. The backend/layout/order discrepancy remains unexplained. **Keep canonical gables default and retain the failing gate unchanged.**

## 10. Reference audit: remaining advantages and tradeoffs

The completed read-only subagent audit is `engine/docs/avbd-ring-reference-audit.md`. Several findings have since been implemented; its old status paragraphs are not current. Essential remaining considerations:

- **Contact initialization during narrowphase:** reference advantage now being tested; avoid claiming a win until full timing/gates pass.
- **Counting-sort grid:** reference avoids linked-list pointer chasing and filters exact cells. Rubble has fixed cell identity and filtered early ownership. A counting-sort prototype did not win locally; revisiting needs a materially different design and evidence.
- **Memory capacity/residency:** reference sizes capacity from initial touching-pair estimates and grows before overflow, while Rubble uses rounded fixed capacities. Reference ring allocation ~279MiB. Persistent contacts reduced Rubble allocation, but full memory parity is not established. Reducing allocation alone does not necessarily reduce active bandwidth.
- **Live loop:** reference submits without per-step waiting and reads viewer stats asynchronously. Rubble's live frame waits for staging readback. A bounded staging ring/nonblocking HUD could improve live FPS separately from physics throughput; do not confuse a UI scheduling improvement with solver speed.
- **Pair-centric algorithms:** reference processes compact pair/point state efficiently. Rubble grouping trials repeatedly lost occupancy or moved costs. Copying source structure is not evidence of a win.
- **Reference numerical differences still worth investigating carefully:** Rubble's slow-body criterion uses radius0.5 rather than actual radius (~0.612372 for brick,8 for ball); tangent cap1e6 versus reference1e10; different LDLT equilibration/pivot floor and position/angular clamps. These are verified source differences, not proven slowdown causes. Instrument whether they matter before changing physics; require independent ill-conditioned/dense tests and scene gates.
- **General large-body capacity:** audit found only16 large dynamics/eight planes and endpoint-only recovery for some large statics. Adequate for this ring's one ball/ground, not general mixed-size parity. Add explicit overflow/fallback before advertising generality.
- **Reference hash algorithm:** exchange with displaced-occupant carry and full probing is more elaborate than simple exchange. Any replacement must prove reachability, duplicate handling, wraparound/high load, exhaustion reporting and both backends.
- **Reference sparse fusion is not an unquestioned model:** its optional implementation was restricted to≤1,024 active bodies and disabled by default after replay divergence. Rubble's retained full-ring tail is separately bounded/tested.
- **Rubble strength:** cached80-byte world OBBs avoid repeated quaternion geometry work. They cost bandwidth/storage; removing them is a hypothesis, not obvious cleanup.
- **Shared engine scope:** Rubble supports compounds, joints, sleeping, commands and multiple backends. Ring-specific specialization cannot silently remove those capabilities from general paths.

## 11. Recommended continuation sequence

1. **Establish source/export state before measuring.** The working tree contains the unfinished inline candidate, but defaults remain production.70 modules are exported. Confirm manifest, query and metadata; do not accidentally benchmark stale generated WGSL.
2. **Add the explicit cold-contact initialization assertions** described in section8. Keep focused mixed-history and shader-validation tests green.
3. **Measure candidate vs fresh controls serially**, ideally production/candidate/production with `profile=detail`. Candidate URL is below. Require unchanged scale, active bodies, iterations, zero overflow and valid work; compare trajectory/counter changes rather than assuming counts must be bit-identical forever. Warm should become empty; inspect whether NP grows enough to erase the saved pass.
4. **If neutral/slower, reject promptly.** Reverse only the saved candidate patch, retain independently useful tests if applicable, re-export69 production entries, rerun appropriate focused/browser tests, and record evidence. Several previous “obvious” optimizations failed by shifting cost.
5. **If a repeatable win, resolve eligibility/wakeup semantics** and expose a narrowly scoped native candidate path for full ring validation. Then run ring480, columns180, canonical gables240, browser ring/gables reset, and shared Metal oracles serially. Do not adopt from45 focused tests alone.
6. **Continue against measured remaining costs:** collision/contact traffic first; investigate new register/occupancy or data-layout evidence before another arithmetic rewrite. Hash clearing is too small to be the main target. Preserve independent oracles when changing equations or reduction order.
7. **Refresh three-avbd and Rubble back-to-back** once controls are stable. Record adapter/UA/source/shader hashes, raw Rubble batches, reference precision limits, workload snapshots and memory capacities. The9.99 reference result is useful, but not the final publication-grade pairing with11.89.
8. **After ring physics throughput meets target with correctness**, separately address live rendering/readback costs and move to gables. Resolve native persistent-gables drift before changing its default. Do not turn the broader goal complete until its explicitly required work is done.

## 12. Files and commands for the next worker

### Important implementation files

| Path | Purpose |
| --- | --- |
| `engine/slang/kernels/world_step.slang` | Main physics phase graph kernels, hash, contact generation/persistence, graph, solve |
| `engine/slang/physics/atomics.slang` | WGSL atomic compatibility/CAS contract |
| `engine/slang/physics/contact_storage.slang` | Canonical/compact/persistent storage abstraction |
| `engine/slang/physics/box_collide.slang` | SAT and eight-point clipping |
| `engine/slang/physics/common.slang`, `math3d.slang` | Layout/math/solver helpers |
| `engine/slang/kernels/compact_solve.slang`, `persistent_solve.slang` | Specialized entry wrappers |
| `engine/slang/kernels/initialized_narrowphase.slang` | Unfinished candidate wrapper |
| `engine/crates/engine-webgpu/build.rs`, `src/lib.rs` | Shader generation/translation/IR checks |
| `engine/crates/engine-webgpu/src/gpu.rs` | Native wgpu host, buffers, shader aliases, scheduling |
| `engine/crates/engine-webgpu/src/gpu_tests.rs` | Focused actual-GPU tests and independent numerical oracles |
| `engine/crates/engine-webgpu/demo.html` | Browser scenes, buffers, scheduler, render/HUD, benchmark modes |
| `engine/crates/engine-webgpu/scene-builders.mjs`, `spatial-order.mjs` | Pure scene and identity-safe permutation logic |
| `engine/crates/engine-webgpu/physics-benchmark.mjs` | Fair timing protocol/workload validation |
| `engine/crates/engine-webgpu/ring-regression.mjs`, `gables-regression.mjs` | Browser physical checks/reset |
| `engine/crates/engine-webgpu/tests/{ring_hold,columns_hold,gables_hold}.rs` | Full-scale native gates |
| `engine/crates/engine-metal/tests/{oracle_accuracy,avbd_slice,columns_wgpu_parity}.rs` | Shared backend/oracle protection |
| `engine/crates/engine-webgpu/dispatch-diagnostic.html`, `.wgsl`, `src/bin/gpu_dispatch_probe.rs` | Independent performance probe |
| `engine/DECISIONS.md` | Chronological decisions; newest adopted/rejected entry is8dt, next available8du |

### Shell prerequisites

Use repository root as cwd. Per provided AGENTS instructions, export these **before every Rust shell command**:

```sh
export RUSTUP_TOOLCHAIN=stable
export WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
```

These instructions originated in cloud setup; this current host is macOS, and actual tested GPU adapters report Metal. Do not assume the Linux ICD path makes the local adapter Vulkan. Native GPU tests need sandbox escalation on this machine; the `cargo test` prefix was approved. Query actual tool approval state rather than assuming it transfers to another worker/session.

Node24 used here:

`node`

Slang compiler:

`slangc`

### Focused build/test/export

Run serially with GPU work stopped; prepend the exports above to each Rust invocation:

```sh
cargo test --manifest-path engine/Cargo.toml -p engine-webgpu --release --lib -- --test-threads=1
cargo run --manifest-path engine/Cargo.toml -p engine-webgpu --release --bin wgsl_export
```

Export directory is `engine/crates/engine-webgpu/generated/wgsl/`, gitignored. Current manifest has70 entries; adopted pre-candidate baseline has69. Exporter writes manifest but does not delete obsolete generated files.

```sh
node --test \
  engine/crates/engine-webgpu/tests/demo_profiling.test.mjs \
  engine/crates/engine-webgpu/tests/physics_benchmark.test.mjs \
  engine/crates/engine-webgpu/tests/persistence_schedule.test.mjs

node --experimental-transform-types --test \
  engine/crates/engine-webgpu/tests/scene_builders.test.mjs

node --test \
  engine/crates/engine-webgpu/tests/spatial_order.test.mjs
```

### Full native gates

Prepend the Rust exports to each shell. Run these **one at a time**, not alongside browser benchmarks:

```sh
ENGINE_PERSISTENT_CONTACTS=1 ENGINE_SPARSE_COLORS=10 \
  cargo test --manifest-path engine/Cargo.toml -p engine-webgpu --release \
  --test ring_hold --test columns_hold -- --nocapture --test-threads=1

cargo test --manifest-path engine/Cargo.toml -p engine-webgpu --release \
  --test gables_hold -- --nocapture --test-threads=1

cargo test --manifest-path engine/Cargo.toml -p engine-metal --release \
  --lib --test avbd_slice --test oracle_accuracy --test columns_wgpu_parity \
  -- --nocapture --test-threads=1
```

The gables command assumes no inherited `ENGINE_PERSISTENT_CONTACTS` or `ENGINE_COMPACT_CONTACTS`; unset those if necessary to run canonical. The first command does **not** yet select inline NP. Useful existing optional host flags: `ENGINE_COMPACT_CONTACTS=1`, `ENGINE_PERSISTENT_CONTACTS=1`, `ENGINE_MANIFOLD_PERSISTENCE=fused`, `ENGINE_SPARSE_COLORS=10`. Diagnostic `WG_ONLY`/`WG_SKIP` can omit phases; never use them in a valid performance claim.

### Servers and browser URLs

Rubble static server was running on port8765, tool session67690, serving `engine/crates/engine-webgpu`. If absent, start from that directory:

```sh
python3 -m http.server 8765 --bind 127.0.0.1
```

Reference Vite server was running on5317, tool session59802, cwd `inspiration/pinned/three-avbd`:

```sh
node node_modules/vite/bin/vite.js \
  --host 127.0.0.1 --port 5317 --strictPort
```

Reuse an existing server rather than launching duplicates. Session IDs may not transfer to a new worker; confirm actual availability. These idle servers are not continuously running a physics simulation.

- Production ring timing: `http://127.0.0.1:8765/demo.html?scene=ring110k&bench=only`
- Collision detail: append `&profile=detail`.
- Graph detail: append `&profile=graph`.
- **Unfinished candidate:** `http://127.0.0.1:8765/demo.html?scene=ring110k&bench=only&profile=detail&persistence=narrowphase`
- Ring480/reset: `http://127.0.0.1:8765/demo.html?scene=ring110k&check=ring`
- Gables240/reset: `http://127.0.0.1:8765/demo.html?scene=gables506k&check=gables`
- Reference: `http://127.0.0.1:5317/bench3d.html`, select paper tier for ring and the 510k tier for full gables.
- Independent diagnostic: `http://127.0.0.1:8765/dispatch-diagnostic.html`.

Useful browser controls: `contacts=canonical`, `contacts=compact`, `contacts=persistent`, `persistence=manifold`, `persistence=fused`, `prefix=serial`, `sparse=32`, `packing=atomic`, `adjacency=append`, `coloring=rebuild`, `order=morton`. Check reported metadata after navigation. Regression pages benchmark only with explicit `bench=1`; `check=ring` is not a fair throughput mode.

### Browser automation state and pitfalls

Use the supported CUA browser tool. Existing selected IAB browser ID is`2`, tabID`3`, persistent handle was `retryTab`. Current page is the stopped production detailed benchmark showing11.89ms. It was marked deliverable. New workers should rebind from actual browser inventory; do not assume a JavaScript handle transfers. After compaction, call `cua.rewriteDocumentation()` before continuing browser control.

Read current AX/DOM completion state, then captured console `BENCH ` or `REGRESSION ` logs. Logs survive navigation, so verify timestamps and URL before accepting results. A prior user-assisted normal-HTTP navigation resolved access to a restored internal error page; do not request that again unless a real browser blocker recurs.

Avoid waiting on a browser download event here: one call unexpectedly blocked about48 minutes. The reference “Copy results” UI reported copied while the exposed clipboard read was empty. A “Saved” button did not guarantee a fresh file. `~/Downloads/apple-metal-3.json` (a local download, not in the repo) is the older exact10.31ms/26.32ms report, **not** the fresh9.99ms result. Prefer visible values with stated precision over falsely attributing an old download to a new run.

Read `inspiration/pinned/three-avbd/docs/PLAN.md` and `docs/FINDINGS.md` before work inside the pinned repo. Its AGENTS requires `pnpm check` before declaring changes done and says unavailable GPU tests skip, never pass. This work used it as a reference and did not modify it. Preserve oracle fixtures; never regenerate them to make a failing solver pass.

## 13. Evidence inventory and final handoff checklist

Key evidence under `engine/docs/`:

- `avbd-ring-inline-narrowphase-in-progress-2026-10-14.json`: latest11.89ms control, raw batches, counters, phase medians, candidate test status and before/candidate source hashes.
- `avbd-ring-inline-narrowphase-in-progress.patch`: reversible unfinished candidate, relative to adopted working-tree baseline.
- `avbd-reference-refresh-2026-10-14.json`: exact older reference export plus explicitly separated fresh UI-rounded9.99ms result and actual reference source hashes.
- `avbd-ring-post-reboot-2026-10-14.json`: slowdown recovery, then-canonical baseline and raw batches.
- `avbd-dispatch-diagnostic-2026-10-14.json`: independent no-physics/no-Slang probe.
- `avbd-ring-parallel-color-prefix-2026-10-14.json`: last adopted optimization's A/B/A and full-scene gates.
- `avbd-ring-persistent-storage-ab-2026-10-14.json`: persistent layout gain and native gables failure.
- `avbd-ring-clipping-ab-2026-10-14.json`, `avbd-ring-cached-pairs-ab-2026-10-14.json`, `avbd-ring-grid-filter-ab-2026-10-14.json`, `avbd-ring-sparse-colors-ab-2026-10-14.json`: retained optimizations and validation.
- `avbd-ring-contact-header-trials-2026-10-14.json`: neutral loader trials and retained hash mutation regression.
- `avbd-ring-subgroup-primal-trial-2026-10-14.json` and `.patch`: rejected subgroup/sparse8 experiment and reproducible candidate.
- `avbd-ring-split-warm-trials-2026-10-14.json`, `avbd-ring-primal-trials-2026-10-14.json`, `avbd-ring-contact-motion-trial-2026-10-14.json`, `avbd-ring-motion-cache-ab-2026-10-14.json`, `avbd-ring-flat-primal-ab-2026-10-14.json`, `avbd-ring-matrix-trials-2026-10-14.json`, `avbd-ring-primal-hoist-rows-2026-10-14.json`: rejected trials and useful retained tests.
- `avbd-ring-reference-audit.md`, `avbd-regression-notes.md`, `avbd-gables-reference.json`, `avbd-compact-solver.md`: detailed architecture and historical correctness evidence; some default/status language is superseded by this handoff.

At handoff:

- [x] Production slowdown recovery rechecked:11.89ms, not~200ms.
- [x] Latest focused candidate GPU tests45/45 and browser unit tests252/252 pass.
- [x]70 shader modules exported successfully.
- [x] No native GPU test/compiler process left running in final process check.
- [x] Current experiment and rollback patch preserved; default stays production.
- [ ] Candidate full-ring performance measured.
- [ ] Candidate sleep/wakeup eligibility fully resolved.
- [ ] Candidate full-scene gates passed/adoption decision made.
- [ ] Native persistent/compact gables drift resolved.
- [ ] Ring ≥120 physics FPS achieved with fair measurements and correctness.
- [ ] Full three-avbd parity / subsequent506k gables objective achieved.

The standing goal remains active and incomplete. The next worker should begin with section8 and section11, keeping the proven baseline and its regression contracts intact.
