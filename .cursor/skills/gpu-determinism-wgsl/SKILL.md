---
name: gpu-determinism-wgsl
description: Write WGSL compute kernels with atomics, hash maps, and parallel emission that stay bit-identical across runs on the same hardware. Use when adding or modifying GPU kernels in crates/rubble3d/src/gpu or crates/rubble-gpu, when the determinism probe fails, or when replacing CPU logic with GPU parallel equivalents.
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
