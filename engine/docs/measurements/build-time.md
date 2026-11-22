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

## 2026-11-22: many entries of one source in one process (`compile_many`)

The rigid kernels' WGSL and SPIR-V (engine-webgpu, every world entry of each source in
one call) through `slang_batch`, one parse per batch instead of one per entry.

| Edit | Main s | Before this s | After s | slangc processes |
|---|---|---|---|---|
| `slang/physics/common.slang` | 28.2 | 27.0 | 14.7 | 555 → 12 (and 44 batches) |
| `slang/kernels/explicit_beams.slang` | 28.9 | 18.5 | 18.8 | 172 → 173 |
| `slang/kernels/soft_skin.slang` | 4.1 | 3.6 | 3.7 | 1 → 2 |

The explicit modules each have defines of their own, so they share no parse: one slangc
each, as before. Every kernel of the four crates compiled from an empty cache: 86 s with
slangc alone, 69 s with batches (MSL and CUDA are still a slangc a compile).

Every generated kernel is the same byte for byte with and without batches
(`kernel_digest.py`, from an empty cache); in the Linux build image (Slang 2026.19),
the 499 rigid WGSL and SPIR-V outputs a batch writes are all slangc's.

## 2026-11-22: the web build's fast profile (`RUBBLE_WEB_PROFILE=fast`)

The world wasm (`wasm-pack build crates/engine-world-wasm`), rebuilt after an edit:

| Edit | `--release` s | `--profile fast` s | `--profile fast --no-opt` s |
|---|---|---|---|
| a kernel file (`soft_skin.slang`) | 44 | 34 | 5 |
| the soft-body model | – | – | 4 |

Release is thin LTO and one codegen unit. With `--profile fast` cargo finishes in 4 s;
the rest was wasm-opt, which the crates turn off for release only
(`[package.metadata.wasm-pack.profile.release]`), so the fast build passes `--no-opt`.
`build-web.sh` with `RUBBLE_WEB_PROFILE=fast` builds the engine's wasm and host tools
this way and skips the world wasm's Asyncify build (Safari). For iterating only: host
code runs slower than in release (no LTO), so pages are measured with release builds.

## 2026-11-22: loop ids, kernel parts behind their define, test kernels on demand

- The loop caps carry permanent ids instead of `__LINE__`: a line added to a kernel file
  changes no WGSL module it did not change before (a blank line at the top of
  `explicit_beams_drive.slang`: every explicit module below it before, none now), so
  Chrome's pipeline cache keeps them.
- The wings (`EB_AERO`) and self-split (`EB_SELF_SPLIT`) parts of `explicit_beams.slang`
  are files of their own under their define: modules without it do not depend on them.
- The test-only modules (84: the frozen baseline and reference, the deterministic-hits
  and ablation builds) are no longer a cargo feature compiled by the build: a test or a
  measurement compiles them on first use (`engine_webgpu::test_kernels_dir`), so the
  fast loop does not, and the library, the harness and the tests share one build of
  engine-webgpu and everything above it (before: two, with and without the feature).

| Edit | Main s | Before these s | After s | slangc runs |
|---|---|---|---|---|
| `slang/kernels/explicit_beams.slang` | 28.9 | 18.8 | 15.9 | 173 → 143 |
| the wings (`explicit_beams_aero.slang`; in the main file before) | 28.9 | 18.8 | 10.9 | 173 → 59 |
| `slang/physics/common.slang` | 28.2 | 14.7 | 15.6 | 12 → 14 (batches) |

Every kernel the same: `kernel_digest.py` 2856 files identical, the 84 test modules
compiled on demand byte for byte the build's; the moved parts identical in WGSL and, with
`#line` dropped, in MSL and CUDA. The loop ids change the kernels' constants only:
checked on the GPU (fast-check against the previous kernels: 16 of 17 fixtures bit for
bit, the 17th differing identically with the same kernels on both sides; the scenario
matrix's 12 rows bit for bit; no row slower over seven samples of the closest).

`engine-schema` embeds `vehicle_layout.slang` and `character_layout.slang` in
`#[cfg(test)]` code only: an edit there recompiles no crate (checked: schema stays
fresh), so nothing to move.

## 2026-11-22: kernels compiled at run time in the fast profile

engine-webgpu's kernels in the `fast` profile on native are compiled on first use into a
store keyed by the Slang sources (`engine_webgpu::kernel_store`), not by its build script
and embedded: a kernel edit compiles no Rust crate. The measured time includes compiling
the store (`--bin kernels`), which the harness scripts do before the GPU lock.

| Edit | Main s | Before this s | After s | Crates compiled |
|---|---|---|---|---|
| a small kernel file (`soft_skin.slang`) | 4.1 | 3.7 | 1.2 | 3 → 0 |
| `slang/kernels/explicit_beams.slang` | 28.9 | 15.9 | 13.5 | 3 → 0 |
| `slang/physics/common.slang` | 28.2 | 15.6 | 9.1 | 3 → 0 |
| the wings (`explicit_beams_aero.slang`) | 28.9 | 10.9 | 8.5 | 3 → 0 |
| the soft-body model | 6.1 | 3.3 | 3.9 | 7 |
| vehicle authoring | 6.1 | 1.8 | 1.8 | 2 |
| the explicit plan | 4.8 | 2.7 | 2.9 | 5 |
| the explicit GPU host | 2.6 | 2.6 | 2.6 | 3 |
| a World module | 1.7 | 1.7 | 1.8 | 1 |

(The Rust edits' times now include the store check of `--bin kernels`, 0.3–0.5 s.)

The same bytes: `the_embedded_kernels_are_the_run_time_compile` (every module of
`OUT_DIR`, 500+, equal to the run-time compile's); the store against the previous build's
embedded WGSL, every file the same; on the GPU, fast-check with the store's kernels
against those (16 of 17 fixtures bit for bit, the 17th as with the same kernels on both
sides) and the bitwise and loop-cap suites.

## 2026-11-22: the fast web build's world wasm reads its kernels at run time

With `RUBBLE_WEB_PROFILE=fast`, the world wasm is built with `RUBBLE_KERNELS=runtime`: it
embeds no WGSL (18 MB instead of 78 MB) and reads each module the first time the engine
asks, from `generated/kernels/` beside it (engine-world-wasm installs the reader,
`engine_webgpu::set_kernel_loader`; build-web.sh copies the native store there). A kernel
edit for the pages is then:

    cargo run --profile fast -p engine-webgpu --bin kernels -- --copy-to crates/engine-webgpu/generated/kernels

and a reload: 9.2 s after a `common.slang` edit (the slangc work), the world wasm not
rebuilt (before: wgsl_export, then the world wasm, 5 s fast or 44 s release, plus the
same slangc work). Checked in Chrome: demo.html (235 modules read, no errors) and
vehicle-lab.html (the generated sedan driving) run from it; with the pane hidden both
this build and the same build with embedded kernels step once a second (the browser's
throttling), so the page's speed is the same. Release (Vercel, CI) embeds as before.

## Where the time still goes

- An explicit kernel edit is slangc: 143 modules of distinct defines in the fast loop,
  about 1.3 s each (a third of it parsing).
- Release, the C library and wasm still embed the kernels: a kernel edit there
  recompiles engine-webgpu and everything above it (the web build's fast profile:
  about 5 s for the world wasm).
