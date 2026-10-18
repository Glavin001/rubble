# Stress baseline: the remaining work

Written 2026-10-18, when PR Glavin001/rubble#39 was readied for merging into `main`. This
document lists every item of `engine/docs/stress-baseline-goal.md` (the charter) that is not
done, with the plan and the acceptance for each. An item is done only when its acceptance
holds and its evidence (commands, numbers, commit) is in the charter's log. The rules at the
end of `engine/docs/stress-5d-one-command-buffer.md` apply to all of them.

## Where things stand at the merge

| Milestone | State |
|---|---|
| M0–M4 | Done |
| M6 | Done for `blast_parity`. Standard median 10.4, p95 15.9–16.7, worst 17.6–18.8 ms; quick 7.8 / 11.0 / 14.7–15.3 ms (targets 11.46 / 37.88 / 48.24 and 11.38 / 15.32 / 35.5) |
| M5 | **Not done.** You closed it as it stood; that waived its criteria and did not meet them. The items below are what meeting them takes |

Everything experimental is behind flags that are off by default. The default build gives the
recorded digests: guard 32/32 identical; stress-gpu and stress-rubble suites pass.

## R1. One command buffer and one read per tick, on Metal, WebGPU and CUDA

The Goal's "resident" requirement and M5's first step. Plan, work items W1–W6 and acceptance:
`engine/docs/stress-5d-one-command-buffer.md`. In short:

- one submission and one status read per resident tick, **counted**, on every frame kind;
- the stress stage as a backend-neutral command list, with no hand-recorded dispatches;
- Apple Metal, web WebGPU and Linux CUDA each build, run and agree;
- speed parity with today;
- deterministic.

**Started:** the WIP single-encoder path, opt-in as `STRESS_FRAME_SUBMITS=one` (lossless on the
quick tier, slower: median 8.1 against 7.3 ms).

## R2. The barrage's GPU race (determinism)

The 36-shot barrage (`vibe-town_barrage`, `blast_parity`) does not repeat run to run.
- The recorded suites do repeat.
- The race is timing-dependent: it doesn't show with `STRESS_GPU_DET_TRACE`/`DET_END` on.
- The charter log entry "Barrage nondeterminism" has the bisect.
- Likely the race the in-place split path documents ("a GPU race after it, not yet found").

- **Plan:**
  1. A low-overhead probe: a per-frame hash of a few tables folded into the frame's one read
     without extra passes.
  2. Find the first stage whose output differs, then find the missing barrier or ordering in
     that stage.
  3. Fix it losslessly.
- **Accept when:**
  - six barrage runs (`STRESS_RUBBLE_MAX_BODIES=65536 STRESS_RUBBLE_MAX_COLLIDERS=262144`)
    give one digest;
  - the guard is all identical;
  - the suites pass.

  This is 5d's W6.

## R3. Punch-through fidelity in real time (M5's fidelity bar)

M5's bar: the free-wall ram and the wall row each closer to the reference than the native
stage's correction-limit-1 pass, on ram speed and fragment count. References:
- free wall 35.1 m/s / 6 fragments (correction pass 10.9 / 23);
- wall row 23.6 / 133 (correction pass 4.9 / 240).

| Option (all flags, off) | Free wall | Wall row | Frame cost | Bar |
|---|---|---|---|---|
| `blast_parity` (default) | 8.6 / 1 | 10.2 / 43 | real time | free wall misses on speed |
| Routed, explicit island | 35.0 / 3 | 31.7 / 41 | 275 ms | met, not real time |
| Impact window, 3 ms, explicit contact | 35.08 / 3 | 26.66 / 124 | 54–527 ms | met, not real time |
| GPU punch (`STRESS_IMPACT_PUNCH=gpu`) | 25.4 / 3 | 26.0 / 19 | ≤ 10 ms | free wall met; wall row misses fragments (needs > 26) |

- **Open sub-items:**
  - **R3a. Wall-row fragments under the GPU punch.**
    - The second wall shatters dynamically in the reference; one plug per hit does not
      reproduce that.
    - Plan: a physically derived model of a struck plate's breakup under the plug's impact.
      For example, the stress solve of the struck region with the plug's contact force as a
      load, cascaded within the frame, in place of a single plug.
    - Measured against the reference, without fitting a parameter to the reference.
  - **R3b. Collider leak in long barrages under the GPU punch.**
    - Every split re-adds every child's colliders, the large remainder's included, and
      rubble's collider arrays are append-only.
    - Plan: the largest child keeps its parent's engine body and colliders (only smaller
      pieces are spawned), in `spawn.slang` and the engine's adoption.
    - Lossless for the split's physics.
  - **R3c. Adoption.** You decided the GPU punch stays behind its flag. Adopting it into a
    profile is your decision once R3a and R3b are done.
- **Accept when:**
  - with the chosen profile, both scenes beat the correction pass on both measures;
  - every impact frame is within the M6 worst-frame target;
  - vibe-town quick and standard meet all M6 targets with the profile on;
  - the 36-shot barrage runs to the end with the default collider capacity;
  - all deterministic.

## R4. Routed shots on vibe-land's town (needs you)

The town's assets set its explicit stress step near 1e-7 s: 2 mm road markings, mailbox flags,
timber posts. 159 structures certify below 1 µs. Routed island frames there need 20k–100k
substeps.

- **Plan:**
  1. You re-export the town pack from Shardwright: road markings merged into their slab or
     made non-structural, and a minimum chunk thickness. Data at the source, per AGENTS.md.
  2. Then re-measure the routed profile on the town.
- **Accept when:**
  - `STRESS_BENCH_STRICT_STEP=1e-6` passes on the re-exported pack;
  - routed quick and standard tiers meet the M6 targets (with whichever impact model R3
    adopts).

## R5. Real-time impact frames for the explicit model (research)

An explicit impact window costs about 46 µs per substep. Every lever measured left it there:
- fewer passes;
- feature code compiled out;
- one threadgroup;
- per-thread chunk caching;
- wide islands;
- a coarser step (lossy).

The log has each experiment.

- **Plan:** only if you want the explicit model in real time, a different formulation:
  - fewer substeps (sliver elimination at the source, as R4);
  - or an on-chip island design, which needs a new idea for the dependent per-substep chain.
- **Accept when:** an explicit-window impact frame on the free wall and the wall row is within
  16.7 ms on the M3 Max, with the same fidelity as today's window.
- R3's GPU punch may make this unnecessary: it's your call.

## Order

1. R1 (5d: W1 → W6, of which W6 is R2).
2. R3b.
3. R3a.
4. R3c (your decision).
5. R4 (needs the re-export).
6. R5 (only if you ask for it).
