# Rubble Performance Plan — maximize real-time rigid-body runtime

Scope: **rigid-body collision performance only**. Goal: match/beat the two reference AVBD solvers
(`avbd-metal`, `webphysics`) on real-time throughput while the test net in
[`test-coverage-gaps.md`](./test-coverage-gaps.md) guarantees no physics regressions.

> **North star.** Both references run the *entire* substep fully GPU-resident with **zero hot-path
> CPU↔GPU readbacks** and **indirect dispatch** off GPU-side counts. Rubble's correctness is fine; its
> speed ceiling is set by per-frame CPU round-trips that neither reference pays. The whole plan is:
> *make the frame GPU-resident, drive dispatch from GPU counts, and keep the only CPU↔GPU transfer an
> async readback for rendering.*

---

## 1. Where the time goes today (measured, not guessed)

Rubble already attributes every phase with a CPU/GPU lane (`StepTimingsMs`, overlay is the single source of
truth for web + native). The **CPU lanes are the target**:

| Phase (overlay) | Lane | Cost source | Code |
|---|---|---|---|
| `Upload` | **CPU→GPU** | full body state + props + shapes re-uploaded **every frame** (World is CPU-authoritative) | `rubble3d/src/lib.rs:898` (`gpu_pipeline.upload(...)`) |
| `Broadphase · Bounds` | **CPU** | AABB staging | `mod.rs:751` (`aabbs.download`) |
| `Broadphase · Build` | **CPU+GPU** | hybrid LBVH: Morton readback + **CPU Karras tree** | `gpu_lbvh.rs:1552-1558` (`morton_keys.download_with` → `build_tree_cpu`) |
| `Broadphase · Readback` | **GPU→CPU** | Morton / pair-count readback | `gpu_lbvh.rs:1552`, pair-count `read` |
| `Contacts` | **GPU→CPU** | full contact buffer downloaded every frame (for CPU coloring/adjacency/compound-merge/persistence) | `mod.rs:1402,1457` (`contacts.download`) + `contact_count.read` `mod.rs:824,1399,1447` |
| (inside `Solve`) | **CPU** | `build_body_contact_adjacency` rebuilt every frame, `body_order` re-uploaded, coloring readback | `mod.rs:1271-1274`, `gpu_color_bodies` |
| `Extract` | **GPU→CPU** | final `body_states.download` every frame | `mod.rs:1417` |

A **fully-GPU LBVH path already exists** (`gpu_lbvh.rs:1454` "no CPU readback for tree construction",
GPU Karras build + refit + GPU traversal) — it's built but the engine still uses the hybrid path in places.
That is the cheapest first win.

There is **no perf-regression guard** in the suite (only `debug_assert!` timer sanity). Add one in Milestone 0.

---

## 2. Best practices to adopt (from the two references, mapped to rubble)

| Technique | Source | Rubble application |
|---|---|---|
| **Fully GPU-resident state across frames** | both | Stop re-uploading state each frame; bodies live on GPU; CPU readback only for rendering/queries, **async** |
| **Indirect dispatch off GPU counts** | both (`bp_finalize_pairs`, `color_scan`, `buildContactDispatchArgsKernel`) | Write workgroup counts into an indirect buffer on GPU; never `contact_count.read()` to size a dispatch |
| **GPU CSR adjacency (clear→count→scan→scatter)** | avbd-metal `GPUSolver.swift:2058`, webphysics `buildPairBodyListsKernel` | Replace CPU `build_body_contact_adjacency` |
| **GPU incremental/persistent coloring + counting-sort body order** | avbd-metal `color_iterate`/`color_count/scan/scatter`, webphysics `greedyBodyColorsKernel` | Replace Luby-with-readback; reuse the existing `GpuRadixSort` for the order |
| **CAS open-addressing persistence map** for O(1) warm-start | avbd-metal `pm_insert`/`pairMapFind` | Replace the O(new×prev) brute-force warm-start scan (`warmstart_wgsl.rs:65`) |
| **Double-buffered async LBVH** (build overlaps sim) | webphysics `broadPhase.ts:43,91` | Build next frame's BVH while traversing this one; don't stall |
| **Counting-sort spatial hash** (alt to BVH for uniform scenes) | avbd-metal `20_broadphase.metal` | Optional second broadphase for dense uniform scenes; pick per-scene |
| **Encoder split at stage boundaries** (2.5–3× measured) | avbd-metal `GPUSolver.swift:1785` | Split the one mega solve-encoder per stage |
| **Persistent single-dispatch solver for small scenes** | avbd-metal `solve_persistent` | Collapse `iters×colors×dual` into one workgroup with barriers at low body counts |
| **Subgroup lane-split primal** (8 threads/body) | avbd-metal `primal_particles_split` | For bodies with many contacts; gate behind wgpu subgroup support |
| **SoA hot/cold buffer split, fat-margin + velocity-expanded AABBs** | both | Already partly present; finish (separate hot solver lanes; reuse contact_offset) |

---

## 3. Milestones (sequenced by impact ÷ risk, each gated by tests)

### M0 — Measurement + safety net (prerequisite, do first)
- **Perf harness:** a `criterion`-style bench (or a `#[ignore]` timing test) that runs fixed scenes
  (16/64/256/1024 bodies; a 10-box stack; a 1k-box grid) for K steps and records `StepTimingsMs` per phase to a
  committed baseline JSON. Add a CI job (non-blocking initially) that prints per-phase deltas vs baseline.
- **Perf-regression guard:** an assertion test (`heavy` lane) that the **CPU lanes** of the step are below a
  budget that shrinks as milestones land (e.g. after M2, assert `upload_ms` ≈ 0; after M3, assert `contact_fetch_ms`
  ≈ 0). This converts each milestone's win into a permanent guard.
- **Physics safety net:** land the Part C set from `test-coverage-gaps.md` — at minimum **CA1** (two-world
  determinism), **CA2** (manifold count/location), **CA5** (broadphase vs brute-force), **CA4** (scale rest-heights),
  **CB9** (wasm testkit lane). These are the tests every milestone below must keep green.
- *Risk:* none (tests only). *Exit:* baseline recorded; net green on native + web.

### M1 — Switch broadphase to the existing fully-GPU LBVH (low risk, quick win)
- **Change:** route the step through the already-built GPU Karras path (`gpu_lbvh.rs:1454`); delete the Morton
  readback + `build_tree_cpu` from the hot path (keep it behind a `cfg`/feature for debugging only). Adopt
  webphysics's **double-buffered async** build (`broadPhase.ts:43,91`) so the tree builds while last frame traverses.
- **Phase impact:** `Broadphase · Build (CPU)` → ~0, `Broadphase · Readback` → ~0, `Bounds (CPU)` shrinks (move AABB
  staging fully on-GPU).
- **Guard:** CA5 (pair-set vs brute force under motion) — this is exactly the test that proves the GPU tree finds
  the same pairs. Plus the manifold/scale tests.
- *Risk:* low (path exists, behavior-equivalent). *Exit:* broadphase CPU lanes ≈ 0; CA5 green; no narrowphase delta.

### M2 — GPU-resident body state (the foundational win)
- **Change:** keep `body_states`, `props`, shape buffers **persistent on GPU across frames**. `World::step` stops
  re-uploading the full state each frame (`lib.rs:898`); instead it uploads **only deltas** (new/removed bodies,
  `set_position`/`set_velocity` pokes) and the per-step uniforms. The final `body_states.download` (`mod.rs:1417`)
  becomes an **async readback used only for rendering/queries** (mirror webphysics's `debugReadbackInFlight` + the
  existing `step_async`). Predict/extract already run on GPU; just stop the round-trip.
- **Phase impact:** `Upload (CPU)` → ~0 (delta-only), `Extract` readback moves off the critical path (async).
- **Guard:** CB3/CB4 (`rubble-wasm` marshaling + sync/async equivalence), CA1 (determinism — residency must not
  change results), the full oracle ladder (rest heights, energy, momentum) which depends on readbacks still being
  correct. **CB9 (wasm testkit lane) is critical here** — the web path uses `step_async`, and residency is where
  native/web could silently diverge.
- *Risk:* medium-high (touches World ownership model, add/remove paths, `compact_states`). Land behind a flag;
  flip once the net is green on both lanes.
- *Exit:* `upload_ms` ≈ 0 in the perf guard; CA1 + CB9 green; identical trajectories pre/post.

### M3 — GPU contact pipeline: coloring + adjacency + indirect dispatch (kills the contact readback)
- **Change:** move three things on-GPU so contacts never come back to the CPU:
  1. **CSR adjacency** via clear→count→scan→scatter (replace `build_body_contact_adjacency`, `mod.rs:2893`).
  2. **Incremental/persistent coloring** (reuse previous colors; early-out when unchanged) + **counting-sort body
     order** via the existing `GpuRadixSort` (replace Luby-with-readback `coloring_wgsl.rs` + the CPU order build).
  3. **Indirect dispatch** for primal/dual/solve passes from a GPU-written args buffer (remove `contact_count.read`
     at `mod.rs:824,1399,1447` and the `contacts.download` at `:1402,1457`). Compound contacts (currently
     CPU-generated) move to a GPU emit pass or a clearly-bounded secondary buffer so they don't force a readback.
- **Phase impact:** `Contacts (GPU→CPU)` → ~0; the CPU adjacency/order/coloring cost inside `Solve` → ~0; the
  mid-frame sync stalls disappear (big real-time latency win).
- **Guard:** CA1 (determinism — GPU coloring order & atomic adjacency are the #1 nondeterminism risk; the
  two-world + reversed-insertion test is the gate), CA2 (manifold survival through the new contact path), CA4
  (scale — color-bucket/pair-buffer overflow), CA3 (warm-start neutrality), CA5.
- *Risk:* high (most complex; determinism-sensitive). Land sub-steps independently (adjacency, then order, then
  indirect dispatch), each behind the perf guard + CA1.
- *Exit:* `contact_fetch_ms` ≈ 0; no `contact_count.read` in the hot path; CA1/CA2/CA4 green on native + web.

### M4 — O(1) warm-start via CAS persistence map
- **Change:** replace the O(new×prev) brute-force match (`warmstart_wgsl.rs:65`) with avbd-metal's pair-keyed
  **CAS open-addressing map** (`pm_insert`/`pairMapFind`, capacity next-pow2(2·maxPairs), linear probing). Carry λ
  and penalty by O(1) lookup; keep the stable-feature-ID work as a separate contact-robustness task.
- **Phase impact:** removes quadratic warm-start cost (dominant in dense scenes); shrinks `Solve`/contact prep.
- **Guard:** CA3 (warm-start converges to the same rest state — directly proves the new map is correctness-neutral),
  CA1, and the existing `warm_start_matches_by_feature` regression.
- *Risk:* medium. *Exit:* warm-start cost ~flat with contact count; CA3 green.

### M5 — Throughput micro-opts (after the round-trips are gone)
- **M5a Encoder split at stage boundaries** (`mod.rs:1280` mega-encoder → per-stage) — avbd-metal measured
  2.5–3×; verify on wgpu/Vulkan with the perf harness before keeping.
- **M5b Persistent single-dispatch solver for small scenes** (`solve_persistent`-style: whole `iters×colors×dual`
  in one workgroup with `workgroupBarrier`) — dominant win at low body counts (interactive case); gate by a
  body-count threshold and wgpu workgroup-size limits.
- **M5c Subgroup lane-split primal** (8 threads/body + `subgroupAdd`/shuffle reduction) for bodies with many
  contacts; gate behind wgpu subgroup feature detection, fall back to the scalar path.
- **M5d SoA hot/cold split + fat/velocity-expanded AABBs** finish (separate the hot solver lanes; reuse
  `contact_offset` for the velocity margin) — coalesced access + fewer pair drops in resting stacks.
- **Guard:** entire physics net unchanged; perf harness shows the win per opt. Each is independently revertible.
- *Risk:* low-medium each. *Exit:* per-opt measured improvement, net green.

---

## 4. Expected end state vs the references

After M1–M3 the hot path is **GPU-resident with zero hot-path readbacks and indirect dispatch** — structurally
matching `avbd-metal`/`webphysics`. M4 removes the last super-linear cost. M5 closes the constant-factor gap
(lane-split, persistent dispatch, encoder structure). Combined with rubble's existing advantages (native + WebGPU
+ wasm, 2D+3D, multi-GPU, broad shape coverage) and the stability core already landed, this puts rubble at or
beyond both references on real-time rigid-body throughput **without giving up correctness**, because every
milestone lands against the oracle-backed, dual-target (native + browser) test net.

## 5. Sequencing summary
```
M0  safety net + perf harness + perf guard      (tests only; prerequisite)
M1  fully-GPU LBVH (use existing path) + async  → broadphase CPU lanes ≈ 0      [low risk]
M2  GPU-resident state, delta upload, async RB   → upload ≈ 0, extract off-path  [foundational]
M3  GPU coloring+adjacency+indirect dispatch     → contact readback ≈ 0          [highest risk, biggest latency win]
M4  CAS persistence-map warm-start               → removes O(N²)                 [medium]
M5  encoder split / persistent dispatch / subgroup / SoA  → constant-factor      [incremental]
```
Hard rule: **no milestone lands without its named guard tests green on both the native (lavapipe) and web
(SwiftShader-WebGPU) lanes.**
</content>
