# Build time: edit to ready

What an edit costs before the fast loop's harness is built again
(`cargo build --profile fast -p engine-world --example vehicle_lab`, the build
`fast-check.sh` and the scenario matrix run). Measured with
`python3 tools/build-time/measure.py` (from `engine/`): each edit is new (a kernel
comment or an unused Rust function no build has seen, so caches miss as after a real
edit), three runs, the median. M-series Mac, 16 cores, nothing else building. The
architecture and its rules: [../build-architecture.md](../build-architecture.md).

## 2026-11-22: kernel cache for every compile, kernel lists and authoring split out

Before: `d5a1d480` (main). After: this branch's per-compile cache for every slangc run,
`engine-kernel-manifest` and `engine-softbody-authoring`.

| Edit | File | Before s | After s | Crates compiled | Build scripts run | slangc runs |
|---|---|---|---|---|---|---|
| a small kernel file | `slang/kernels/soft_skin.slang` | 4.1 | 3.6 | 3 → 3 | 1 → 1 | 3 → 1 |
| the explicit vehicles' kernel | `slang/kernels/explicit_beams.slang` | 28.9 | 18.5 | 3 → 3 | 1 → 1 | 175 → 172 |
| the most-included kernel file | `slang/physics/common.slang` | 28.2 | 27.0 | 3 → 3 | 1 → 1 | 558 → 555 |
| soft-body runtime model | `engine-softbody/src/explicit_beams.rs` | 6.1 | 3.3 | 6 → 7 | 1 → 0 | 3 → 0 |
| vehicle authoring | `authoring/vehicle.rs` | 6.1 | 1.8 | 6 → 2 | 1 → 0 | 3 → 0 |
| the explicit plan | `engine-exec/src/explicit_plan.rs` | 4.8 | 2.7 | 5 → 5 | 1 → 0 | 3 → 0 |
| the explicit GPU host | `engine-webgpu/src/explicit_beams_gpu.rs` | 2.6 | 2.6 | 3 → 3 | 0 → 0 | 0 → 0 |
| a World module | `engine-world/src/threshold.rs` | 1.7 | 1.7 | 1 → 1 | 0 → 0 | 0 → 0 |

- The explicit kernel: the modules the build script compiled one at a time (the frame,
  its 512-lane and aero variants, the wide passes) now compile in parallel with the
  rest.
- Rust edits below the executors no longer rerun the kernel build script: it depends
  on `engine-kernel-manifest`, not on `engine-exec` (and through it the soft-body
  model). A Rust edit no longer starts any slangc (before: the three uncached
  standalone modules every time).
- An authoring edit compiles the authoring crate and the harness only (the harness
  uses the generated sedan): not the soft-body model, the plans, the GPU host or the
  World.
- The soft-body model's dependents include `engine-softbody-authoring` now (7 crates,
  the authoring crate in parallel with the plans).

Every generated kernel is the same byte for byte (`tools/build-time/kernel_digest.py`,
2856 files of the shipped and test sets, `#line` names compared from the Slang root):
the only difference is engine-metal's 145 SPIR-V and CUDA copies that nothing read and
that are no longer compiled.

## Where the time still goes

- A kernel edit is slangc: one process per module, each parsing the module's whole
  include closure (14–25k lines). `common.slang` reaches 555 modules of the fast loop;
  `explicit_beams.slang` 172, 30 of them the test-only modules (`test-kernels`, on in
  the harness through engine-world's dev-dependency).
- Every kernel edit recompiles engine-webgpu and everything above it: the build
  script embeds the kernels with `include_str!`.
