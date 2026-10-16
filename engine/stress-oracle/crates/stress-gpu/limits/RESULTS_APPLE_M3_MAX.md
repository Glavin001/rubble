# GPU limits and costs: Apple M3 Max (measured)

Reference numbers for reasoning about what is expensive in the stress-gpu kernels on
Apple silicon, and the current standing of those kernels against the hard limits. The raw
data is [`m3_max.toml`](m3_max.toml) (regenerate with `scripts/limits.sh`); the probe
methods are in [`README.md`](README.md); the RTX 4090 counterpart (documented, not yet
measured) is [`rtx4090.toml`](rtx4090.toml).

**Hardware and conditions.** Apple M3 Max with the 40-core GPU (Apple GPU family 9,
`applegpu_g15s`), unified memory, macOS 14.4 (Metal Shading Language up to 3.1), Xcode
Metal compiler 32023.155, high-power mode. One threadgroup of 256 threads unless stated;
GPU warmed first (cold runs read up to 4x slower); every timed value is the minimum of 7
repeats (medians agree within ~1%). The GPU was shared with other processes at ~25%
utilization during the run (flagged `contended = true`); two full runs agreed within ~2%.
Driver statistics (instructions, registers, spills) come from a private Metal API
(pipeline option `0x40000`); machine-code size from Xcode's offline translator.

## Hard limits

| Limit | Value | Consequence beyond it |
|---|---|---|
| Instruction cache, straight-line code | knee at 16.97-17.63 KB of machine code (2,110-2,192 driver instructions) | 1.6 -> 3.0 ns per instruction (code streams from memory) |
| Instruction cache, mixed code (loads every 8th, branch every 16th) | knee at 9.8-11.3 KB (1,106-1,284 instructions) | 1.4 -> 4.6 ns per instruction |
| Temporary registers per thread | 96 (the driver's cap; spilling starts at 96 live floats) | ~1.6 -> ~3.0 ns per update between 88 and 96 live values |
| Spill stack per threadgroup (spilled bytes x threads) | knee at 184-201 KB (array sweep), 200-233 KB (thread sweep) | dependent stack load 55 -> 96-150 ns |
| Threadgroup memory | 32 KB per threadgroup | compile/launch failure |
| Threads per threadgroup | 1,024 (every current kernel's pipeline accepts 1,024) | launch failure |
| SIMD width | 32 | - |
| Co-resident threads (opt-in probe) | 3,072 per core: 48 x 64, 12 x 256 or 3 x 1,024 groups per core | not a guarantee: Metal publishes no occupancy or forward-progress rule between threadgroups |
| Device-scope memory fence, `coherent(device)` | not in MSL 3.1 publicly (MSL 3.2 / macOS 15+) | cross-threadgroup spin barriers are unsupported here |
| CPU <-> GPU visibility inside a running kernel | none observed, either direction | no host/kernel mailboxes or persistent-kernel polling |

## Operation costs (dependent chains, ns per operation)

"ns" includes the share of loop overhead; "excl. loop" subtracts it (approximate).

| Operation | ns | excl. loop | Driver instructions per op |
|---|---|---|---|
| FMA | 1.90 | 1.00 | ~2 |
| add + precise divide | 37.8 | 36.9 | ~28 |
| add + `fast::divide` | 6.9 | 6.0 | ~4 |
| sqrt (precise and `fast::` are identical) | 26.6 | 25.7 | ~16 |
| add + precise `pow` | **403.6** | 402.7 | ~159 |
| add + `fast::pow` | 11.8 | 10.9 | ~5 |
| precise `exp` / `fast::exp` | 12.3 / 6.0 | 11.4 / 5.1 | ~9 / ~3 |
| `simd_sum` (+ FMA) | 16.2 | 14.4 | ~4 |
| dependent load: threadgroup / device (cached) / constant / stack | 48.6 / 54.6 / 54.5 / 44.8 | 46.8 / 52.8 / 52.7 / 43.0 | - |
| threadgroup exchange (store, barrier, load, barrier) | 126.8 | 112.3 | - |
| device exchange (device-memory barrier) | 146.5 | 132.0 | - |
| dynamic loop iteration | 14.5 | - | ~20 |

Rules of thumb drawn from these:
- A precise `/` costs ~20 FMAs and a precise `pow` ~200; use `fast::` forms where the
  accuracy gate allows. Note: replacing divisions globally (`-freciprocal-math`,
  `-ffast-math`) changed fracture cascades far beyond the reference's spread
  (house settlement 542 -> 274 broken bonds), so precise division must be replaced site by
  site, gated.
- Every dependent memory access is ~45-55 ns wherever it lives: chains of loads (index ->
  address -> value) and spill reloads dominate latency-bound code.
- A loop iteration costs ~14 FMAs: unroll short fixed loops, but unrolling adds code size
  (see the instruction cache).
- Barriers cost ~110-130 ns each with an exchange; the device-memory flavour ~15% more.

## Launch and host costs

| Item | Value |
|---|---|
| Dispatch inside one serial compute encoder | 1.03 us each (intercept 0.25 us); one encoder per dispatch: 1.04 us |
| One command buffer per dispatch | ~15 us GPU span, ~17.4 us wall per dispatch |
| wgpu compute pass | two Metal command buffers per pass (a pre-pass barrier buffer plus the pass): never one pass per dispatch outside profiling |
| Submit + `waitUntilCompleted`, empty command buffer | 9.7 us min, 14.6 us median |
| Submit + `waitUntilCompleted`, one tiny dispatch | 103 us min, 116 us median, 142 us p90 (~52 us commit -> GPU start, ~60 us GPU end -> CPU resumes) |
| Same, CPU spinning on a shared event | 81 us min, 91 us median |
| GPU clock after idle | no slowdown for gaps up to 4 ms; kernels ~4.1x slower (worst 4.3x) after 8-50 ms idle; commit -> start ~500 us from a 2 ms gap |
| Memory bandwidth (streaming copy, read + write) | 356 GB/s max, 352 median |

Implications: batch every substep's dispatches into one pass; count blocking host waits
per frame (each ~100 us); a real-time loop that idles 8 ms or more between frames pays the
clock ramp unless the GPU is kept busy (another queue's work or a keep-alive).

## Where the stress-gpu kernels stand (static, precise build)

From `scripts/kernel_budget.sh` (baseline `kernel_budget.toml`) and the hard-limit tests
(`tests/kernel_limits.rs`). Hot = runs every substep.

| Kernel | Hot | Instructions | Code KB | / i-cache knee | Registers | Spilled B/thread | Spill x threads KB | / stack knee |
|---|---|---|---|---|---|---|---|---|
| island_frame | yes | 12,424 | 90.6 | 5.3x | 96 | 1,104 | 283 | 1.5x |
| contact_forces | yes | 7,481 | 52.9 | 3.1x | 96 | 512 | 33 | 0.2x |
| island_statics | per frame | 5,925 | 46.0 | 2.7x | 96 | 944 | 242 | 1.3x |
| wide_bonds | yes | 4,715 | 35.3 | 2.1x | 96 | 608 | 156 | 0.84x |
| wide_rigid | yes | 3,127 | 22.9 | 1.3x | 96 | 144 | 37 | 0.2x |
| settled_fatigue | per frame | 2,924 | 21.9 | 1.3x | 96 | 480 | 123 | 0.7x |
| wide_chunks | yes | 2,435 | 18.8 | 1.1x | 95 | 0 | 0 | 0 |
| impactor_shares | yes | 1,943 | 13.8 | 0.8x | 96 | 208 | 13 | 0.1x |
| impactor_integrate | yes | 1,573 | 11.7 | 0.7x | 96 | 336 | 86 | 0.5x |
| contact_sums | yes | 1,214 | 9.1 | 0.5x | 83 | 112 | 7 | 0.04x |
| wide_wake | yes | 970 | 7.5 | 0.4x | 56 | 0 | 0 | 0 |
| wide_end | yes | 563 | 4.4 | 0.3x | 95 | 192 | 49 | 0.3x |
| wide_drift | yes | 422 | 3.4 | 0.2x | 81 | 0 | 0 | 0 |
| impactor_crush | yes | 240 | 2.2 | 0.1x | 95 | 48 | 12 | 0.07x |

Hard-limit test status (`tests/kernel_limits.rs`, 2026-10-11):

| Test | Status |
|---|---|
| hot kernels fit the instruction cache | **failing**: 7 of 12 fit (island_frame 5.3x, contact_forces 3.1x, wide_bonds 2.1x, wide_rigid 1.35x, wide_chunks 1.1x over) |
| hot kernels do not spill | **failing**: 3 of 12 spill-free (wide_chunks, wide_drift, wide_wake) |
| spill stack per threadgroup within the knee | **failing**: 12 of 14 within (island_frame 1.53x, island_statics 1.31x over) |
| hot kernels below the register cap | **failing**: 6 of 12 below 96 |
| threadgroup memory within the device limit | passing (largest 1,040 B of 32 KB) |
| pipelines accept their threadgroup size | passing (14 of 14) |

Cross-check against measured time (`stress-gpu-latency`): the island kernel costs
8.4 us per substep for a 4-bond island, rising to ~22 us for large islands. At the
measured 1.6-3.0 ns per instruction, a substep executes ~2.8k instructions (~20 KB), past
the 17 KB knee, while island_frame's spill footprint is 1.5x the stack knee: both cliffs
apply. The joint law alone (wide_bonds, 4.7k instructions) is twice the cache.

## RTX 4090 (documented only)

128 SMs, warp 32, 64K 32-bit registers per SM (255 per thread), up to 100 KB shared memory
per SM (99 KB per block, dynamic), 72 MB L2, ~1 TB/s. Cooperative launch with grid-wide
sync is officially supported (grid sized by the occupancy API). The instruction-cache size
is not published; earlier NVIDIA parts are reported with a ~12 KB L0 per SM partition and
~128 KB L1 instruction cache per SM (third-party microbenchmarks, unverified for Ada), so
the code-size penalty is likely smaller there. Measure before relying on any of this.
