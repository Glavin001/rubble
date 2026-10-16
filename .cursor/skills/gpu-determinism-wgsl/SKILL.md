---
name: gpu-determinism-wgsl
description: Write GPU compute kernels (WGSL, or Slang in engine/slang) with atomics, hash maps, and parallel emission that stay bit-identical across runs on the same hardware, and prove that an optimization is lossless. Use when adding or modifying GPU kernels in crates/rubble3d/src/gpu, crates/rubble-gpu or engine/slang, when runs diverge, or when claiming a performance change does not alter results.
---

# Deterministic WGSL Kernels

The engine guarantees bit-identical results across runs on the same hardware (`crates/rubble3d/tests/determinism_probe_tests.rs`). GPU parallelism breaks this in known ways; these are the patterns that keep it.

## Sources of nondeterminism and their antidotes

1. **`atomicAdd` slot allocation** (e.g. contact emission) assigns scheduling-dependent indices. Antidote: sort afterwards with a **total-order key**. If the natural key `(pair, feature_id)` can tie (capsule caps share a feature ID), add a content-derived disambiguator — a hash of the quantized contact position (`contact_pos_hash` in `narrowphase_wgsl.rs`) — so the sort key is a total order regardless of emission order.

2. **Concurrent hash-map insertion.** Never let inserters read another inserter's non-atomic writes within the same dispatch — that visibility is undefined. The warm-start pair map pattern (`crates/rubble3d/src/gpu/warmstart_wgsl.rs`):
   - CAS only a 32-bit hash word to claim a bucket (`atomicCompareExchangeWeak`); force the hash nonzero so 0 means empty.
   - Write the full key/value non-atomically *after* winning the claim.
   - Verify the full key on lookup in a **subsequent dispatch** — writes are visible across the pass boundary, not within it.
   - Handle spurious CAS failure: `old_value == 0 && !exchanged` means retry the same bucket, don't probe on.
   - For unique keys each key claims one deterministic base bucket → bit-identical to a serial scan. Genuinely duplicate keys resolve scheduling-dependently; make sure only tolerance-tested paths (oracle tests), not the bitwise probe, depend on them.

3. **Coloring / graph partitioning races.** Any "first thread wins" structure must be restated as CAS-claim or a deterministic reduction.

4. **Dispatch bounds vs live counts.** When a kernel is dispatched by an upper bound (count lives on the GPU), guard with `if i >= min(count_buf[0], params.upper_bound) { return; }`. Both operands matter: the buffer is the truth, the param bounds the dispatch.

5. **Float reduction order.** Don't sum across threads with atomics-on-float workarounds; use a fixed-tree reduction or keep per-item results and reduce in a deterministic pass.

## Verification checklist for any new/changed kernel

```bash
# WGSL compiles under naga (add a case in wgsl_validation_tests.rs for each new entry point)
cargo test -p rubble3d --test wgsl_validation_tests
# Bit-identical across runs
cargo test -p rubble3d --test determinism_probe_tests --release
# Full suite
cargo nextest run -p rubble3d
```

If the probe fails, bisect by disabling the new kernel's effect (feed it zero counts) to confirm attribution before hunting inside it.

## Engine (Slang, `engine/slang/kernels`): lessons from the fracture lab

Two races made identical fracture-lab runs diverge after ~120-200 ticks:

6. **Iterated in-place propagation.**
   - `sleep_island_gate` revokes sleep candidates in place over N rounds.
     A revocation written earlier in the same dispatch may already be read, so
     how far it travels depends on scheduling.
   - Antidote: mark a write with the round that made it and treat it as unseen
     until the next round (`budget.deterministicSleepGate`).
   - The same applies to any multi-round label or flood propagation.
7. **Direct-mapped caches used as early-outs.**
   - In the SAT separating-feature cache, pairs that share a slot overwrite
     each other in scheduling order.
   - A cached proof can skip a pair that the full test (which skips
     near-parallel edges) would have given a fallback contact.
   - A cache is lossless only if every hit agrees with the full computation
     it skips.
   - `budget.deterministic` turns on both fixes.

**Floating-point codegen.** On Chrome → Dawn → Metal, *any* change to a
kernel's generated code can change its rounding, even with no maths changed:
- unrolling a loop that only zeroes values;
- turning off Dawn's bounds checks.

So "same source maths" does not prove "same results". Unchanged shaders do
reproduce exactly.

**Proving an optimization lossless.** Don't compare old and new shader builds.
Put the optimization behind a host flag and compare off vs on in one build:
1. Use `budget.deterministic`.
2. Run lockstep in one page or headless run.
3. Compare body snapshots bit for bit, and candidate and contact counts every
   tick.

Example: every broadphase change of 2026-10-15 was proven over 700 hull-grid
ticks this way. Pair-set checks are also useful:
- run `inspectCollisions` on one state with the optimization on, then off;
- call `restoreBodies(snapshot)` between the two runs, because the
  narrowphase wakes bodies.
