# GPU limits

Measured hardware limits the stress-gpu kernels have to fit, and a static guard that
keeps the world kernels inside them.

| file | what |
|---|---|
| `m3_max.toml` | measured on Apple M3 Max (40-core GPU) by `scripts/limits.sh` |
| `rtx4090.toml` | RTX 4090 / sm_89 **documented** values (`measured = false`), with sources |
| `kernel_budget.toml` | committed per-kernel baseline for `scripts/kernel_budget.sh` |

Tools: `tools/limits/limits.m` (Objective-C, Metal + Foundation only, built by
`tools/limits/build.sh`), `tools/limits/codesize.py` (machine-code size),
`tools/limits/budget.py` (budget table and baseline compare).

**Private API.** Instruction counts, temporary registers and spilled bytes come from the
undocumented Metal pipeline option `0x40000`, which makes
`MTLComputePipelineReflection` answer `performanceStatistics`. Machine-code size comes
from Xcode's offline AGX translator `applegpu-nt`. Both can disappear with an OS or Xcode
update; the tools then report `-1`, warn, and skip those comparisons
(`LIMITS_NO_PRIVATE_STATS=1` simulates it).

## Running

```bash
# from crates/stress-gpu, macOS + Xcode
scripts/limits.sh                     # default suite (~30 s) -> limits/<target>.toml
scripts/limits.sh --repeats 11 --out /tmp/m3.toml
scripts/limits.sh --probes ops,dispatch
scripts/limits.sh --coresidency       # + OPT-IN co-residency probe
scripts/limits.sh --watchdog 8        # + OPT-IN watchdog probe (see below; idle machine only)

scripts/kernel_budget.sh              # compare world kernels with the baseline; exit 1 on regression
scripts/kernel_budget.sh --update     # accept the current kernels as the baseline
scripts/kernel_budget.sh --margin 10  # allow +10% instructions / code bytes
```

The GPU is shared (another developer's server, WindowServer, Chrome). The default suite
runs one threadgroup at a time except the bandwidth copy (~1 ms per repeat), runs every
job sequentially, and never stops other processes. Each probe warms the GPU first
(~0.8 ms single-threadgroup runs for 300 ms, plus 30 ms before every timed point: the
clock falls within milliseconds of idling and short warm-up kernels never raise it), and
every point is the **minimum** (and median) of N repeats (default 7). Times are GPU
timestamps (`GPUEndTime - GPUStartTime`); per-operation costs are the slope between two
loop counts, so dispatch overhead cancels.

## Reading the toml

`[control]` records the conditions: chip, GPU cores, OS, `pmset` power mode, Metal
compiler, commit, repeats, and `Device Utilization %` from `ioreg -r -d 1 -c
IOAccelerator` sampled 5 times before and after the run (when the tool is not running,
so: other processes). `contended = true` when either median exceeds
`contention_threshold_pct` (15). A contended run still reports min-of-N, but
issue-bound numbers (ALU throughput) can read high; latency- and fetch-bound numbers
are steadier.

Sweeps are stored as `<name>_columns` plus `<name> = [[...], ...]` rows. Knees are
`[last point below, first point above]` the threshold named in each `method`.

| section | probe | key results |
|---|---|---|
| `instruction_cache` | (a) straight-line loop bodies, 512..4096 statements in steps of 64 (4 independent FMA/MUL chains), and a mixed body (a cached device load every 8th, a uniform branch every 16th statement) | `alu_knee_code_bytes`, `alu_knee_instructions`, `alu_ns_per_instr_below/above`, same for `mixed_` |
| `live_registers` | (b) K floats live across a loop, K = 16..256 | `first_spill_live_floats`, `max_temp_registers_seen`, `time_knee_live_floats`, `first_reduced_max_threads_*` (absent: the driver never reduced maxTotalThreadsPerThreadgroup below 1024) |
| `stack` | (c) per-thread array chased in full-period order; sweep 1 array size at 256 threads, sweep 2 1 KiB/thread with 32..1024 threads | `array_sweep_knee_bytes_per_threadgroup`, `thread_sweep_knee_bytes_per_threadgroup` (footprint = driver spilled bytes x threads, the unit `kernel_budget.sh` uses) |
| `ops` | (d) dependent chains, 1 x 256 threads: FMA, precise/fast divide, sqrt, pow, exp, simd_sum, dependent loads by address space, threadgroup and device barrier exchanges, loop iteration; plus FMA on one SIMD | per op `ns_min`, `ns_median`, `ns_excl_loop` (slope minus the loop-iteration slope, divided by ops per iteration; approximate) |
| `dispatch` | (e) N tiny dependent dispatches in one serial encoder (N = 1..512); 64 dispatches as one encoder, one encoder each, or one command buffer each | `serial_encoder_us_per_dispatch` (least-squares slope), `cb_per_dispatch_*` |
| `round_trip` | (f) commit -> host resumes: empty command buffer, tiny dispatch + `waitUntilCompleted`, tiny dispatch + `MTLSharedEvent` spin (bounded at 1 s) | min/median/p90/max microseconds, split into commit->GPU start and GPU end->resume |
| `clock_ramp` | (f) idle gap 0.1..50 ms before a fixed ~200 us kernel | median/max exec per gap and `median_over_warm`; min-of-N would hide the ramp, so read the medians |
| `bandwidth` | (g) float4 streaming copy, 128 MiB -> 128 MiB, whole GPU | `copy_gb_per_s` (read + write) |
| `coresidency` | (h) OPT-IN | see below |
| `watchdog` | (i) OPT-IN | see below |

### Opt-in probes

* **Co-residency** (`--coresidency`). Thread 0 of every group increments an arrival
  counter, then polls it until all groups arrived or a poll limit passes (~3 ms; the limit
  shrinks with the group count because polls of one atomic serialise), and records what
  it saw. If every group saw the total, all were co-resident; otherwise the minimum seen
  is about the first wave. There is no unbounded spin: it cannot hang. Sweeps 32/64/256/1024
  threads x 1..64 groups per core and stops a group size at its first incomplete point.
  Observed 2026-10-11 (contended, not in `m3_max.toml`): 3072 threads per core for this
  tiny kernel (48 x 64, 12 x 256, 3 x 1024 groups per core; 64 x 32-thread groups per
  core all resident). Kernels with more registers or threadgroup memory fit fewer.
* **Watchdog** (`--watchdog S`). One command buffer per duration 0.25 s, 0.5 s, ... up to
  S, each a single-threadgroup FMA loop, until the driver fails one; reports the longest
  completed and the error. It can stall the desktop for up to S seconds; run it only on an
  idle machine you own, never alongside other developers' GPU work. Not run so far.

## Kernel budget

`scripts/kernel_budget.sh` builds `shaders/generated/world.metal` exactly as `build.rs`
does (`xcrun metal -fno-fast-math`, `metallib`) and prints, per kernel: driver
instructions, temporary registers, spilled bytes, threads per threadgroup (from
`[numthreads]` in `shaders/world.slang`), spilled bytes x threads, static threadgroup
memory, maxTotalThreadsPerThreadgroup, machine-code bytes, and the ratios
code bytes / instruction-cache knee (`alu_knee_code_bytes[0]`) and
spill x threads / stack knee (`array_sweep_knee_bytes_per_threadgroup[0]`). It fails when
a kernel grows more than the margin (+5%) in instructions or code bytes, gains spilled
bytes, or its maxTotalThreadsPerThreadgroup falls below its threadgroup size; more
registers and new or removed kernels are notes. The ratio columns are information, not
gates: several kernels already exceed the knees.

## Cross-check: world kernels against the knees (2026-10-11, commit cb63e436b)

From `kernel_budget.sh` and `target/release/stress-gpu-latency` (one island, one
threadgroup of 256, 2000 substeps, min of 5 repeats; the GPU was contended at ~25%):

| kernel | instr | code bytes | x i-cache knee (17.0 KB) | spill x 256 thr | x stack knee (184 KB) |
|---|---:|---:|---:|---:|---:|
| island_frame | 12424 | 90570 | 5.3 | 282624 | 1.5 |
| wide_bonds | 4715 | 35342 | 2.1 | 155648 | 0.84 |

`stress-gpu-latency` island kernel: 8.4 us/substep at 4 bonds, 11.2 at 212 bonds (0.83
bonds/thread), 16.5 at 324, ~22 us/substep from 592 bonds up (to 8560 bonds, 15 batches).
At 7.3 code bytes per instruction, the knee (17.0 KB) is ~2300 instructions; island_frame
is 12.4k static instructions. Executing all of them straight through would cost 20 us at
the below-knee rate (1.62 ns/instr) and 37 us above it (3.02 ns/instr); the measured
8.4-11.2 us for small islands means a substep executes only part of the static code
(~2.8k instructions at the above-knee rate), which is itself past the knee. So
island_frame runs in the fetch-bound regime, and with 1104 spilled bytes per thread its
282 KB per threadgroup stack is 1.5x the stack knee, where a dependent stack load costs
96-150 ns instead of 55 ns. wide_bonds is also past the instruction-cache knee (2.1x) but
inside the stack knee. Shrinking island_frame's executed path below ~17 KB or its spill
below ~720 B/thread are the two measured cliffs; neither alone was verified by changing
the kernel.

## Differences from the earlier ad-hoc measurements

Measured here (contended, ~25% GPU utilization by other processes) vs the earlier scratch
probes: the ALU i-cache knee is 17.0-17.6 KB (earlier 11-17.5 KB: the mixed body with
loads and branches has its knee at 9.8-11.3 KB, the straight-line ALU body at the top of
the earlier range); below-knee cost is 1.62 ns per *driver* instruction (earlier 1.0-1.2,
counted differently and uncontended), above-knee 3.02 (same as earlier); dependent loads
are 6-8 ns slower (threadgroup/device/constant/stack 48.6/54.6/54.5/44.8, earlier
41/47/47/37) with loop overhead amortised over 8 loads included; the stack knee is
184-200 KB per threadgroup counted as spilled bytes x threads (earlier 135-166 KB);
dispatch in one serial encoder 1.03 us (earlier 1.2-2.2).
