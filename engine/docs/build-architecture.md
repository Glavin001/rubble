# Build architecture

How the engine's crates and kernels are laid out so that an edit rebuilds only what
it changes, and the rules that keep it that way. Measurements:
[measurements/build-time.md](measurements/build-time.md). Tools:
`engine/tools/build-time/` (`measure.py`, `kernel_digest.py`).

Every rule here is about build time only. None of them may change what the engine
computes: a build change is proven lossless with the kernel digest (every generated
kernel byte for byte) and, when host code moves, the scenario matrix (AGENTS.md).

## The layers

```
engine-kernel-manifest     which kernels exist: names, entries, defines, sources (data only)
engine-slang-build         slangc / nvcc driver and the per-compile cache
engine-schema              layouts, units, commands (leaf)
engine-softbody            the soft-body runtime model (explicit_*, jbeam, truck, ...)
engine-softbody-authoring  the generated cars, visual meshes (flexmesh, glb), comparisons
engine-exec                the plans (rigid, explicit, terrain), built on the runtime model
engine-webgpu / -metal / -cuda   executors; their build scripts compile the kernels
engine-world, -ffi, -world-wasm  the World and its bindings
```

Direction is downward only. In particular:

- **Build scripts depend on data, not on plans or models.** A build script reruns,
  and the crate it builds recompiles, whenever any of its build-dependencies changes.
  The kernel build scripts depend on `engine-kernel-manifest` and
  `engine-slang-build` only. (Before: on `engine-exec`, and through it on
  `engine-softbody`, so every edit to the soft-body model compiled it a second time
  for the build scripts and reran all three of them.)
- **The runtime does not depend on authoring.** `engine-exec`, the executors, the
  World and the FFI build on `engine-softbody`; authoring (about 50k lines) is in
  `engine-softbody-authoring`, which only tools, tests, examples and the showcase
  (`engine-world-wasm`) use.
- **`engine-kernel-manifest` depends on nothing.** It is the bottom of every kernel
  build; anything it imports would rerun every kernel build script when edited.

## Kernels

- **Every slangc and nvcc run goes through `engine-slang-build`**, which caches each
  compile by everything its output depends on (slangc, arguments, the source's
  include closure by name and bytes). No build script spawns `slangc` itself.
- **slangc runs from the Slang root with relative paths**, so its output (the `#line`
  names of MSL and CUDA) does not depend on where the checkout is. One cache,
  `~/.cache/rubble/slang-obj` (`RUBBLE_SLANG_OBJ_CACHE`; trimmed least recently used
  past `RUBBLE_SLANG_OBJ_CACHE_GIB`, default 8), serves every worktree: a new worktree
  compiles no kernel another one already compiled.
- **Compile in parallel** (`engine_slang_build::par_map`), never in a loop of single
  compiles.
- **Many entries of one source: `compile_many`.** slangc spends about half a second
  parsing and checking a source's include closure before it emits anything, once per
  process. `compile_many` compiles the entries of one source, target and define set in
  a few `slang_batch` processes (`engine-slang-build/batch/slang_batch.cpp`, the Slang
  library, one parse each) instead of a slangc each. It is used where the output is
  slangc's byte for byte: WGSL and SPIR-V (checked on macOS and Linux with every rigid
  entry; the MSL a batch writes has some `#line` numbers of its own, so MSL and CUDA
  stay one slangc a compile). Each process also checks one batched entry per target
  against slangc and drops the batch if they differ; an entry calling another entry
  point (which the library's link step does not finish) and anything a batch does not
  write in its time limit compile with slangc. Without the Slang SDK beside slangc
  (headers and library) or with `RUBBLE_SLANG_BATCH=0`, every compile is a slangc.
  `RUBBLE_SLANG_TIMING=1` prints the phases on the build script's stderr.
- **Do not compile what nothing reads.** Every output a build script writes has a
  reader.
- **Test-only kernels are not a build feature.** A cargo feature that tests turn on
  splits the build in two (the library without it, tests and examples with it, each its
  own engine-webgpu and everything above). The test-only modules are compiled when a
  test or measurement first asks (`engine_webgpu::test_kernels_dir`, from
  `engine_kernel_manifest::explicit::test_modules`, cached per compile).
- **An environment variable is a rerun trigger only if the output depends on it.**
  Never `PATH`, `SLANGC` or the like in a crate low in the graph: they differ between
  shells and tools, and every rerun recompiles everything above.
  (`RUBBLE_SLANG_CACHE` changes where outputs come from, not what they are.)

## Worktrees

Each worktree has its own `engine/target` (cargo locks a target directory while it
builds, so a shared one would make parallel sessions wait on each other). What is
shared instead:

- the kernels: the per-compile cache in `~/.cache/rubble/slang-obj` (above), so a new
  worktree compiles no kernel another one already compiled;
- optionally, the third-party crates: install sccache (`brew install sccache`) and set
  `RUSTC_WRAPPER=sccache` in your shell. It caches rustc's output by its inputs, so a
  new worktree's wgpu, naga, winit and the rest come from the cache. It does not help
  the workspace's own crates (incremental builds, which sccache does not cache).

## Checked in CI

`python3 tools/build-time/check.py` (from `engine/`; the "Build architecture" job):

- `engine-kernel-manifest` depends on nothing; the kernel build scripts build-depend on
  no engine crate but it and `engine-slang-build`; the runtime (plans, executors,
  World, FFI) does not depend on `engine-softbody-authoring`.
- Nothing but `engine-slang-build` spawns slangc or nvcc.
- Sources stay under 1500 lines (Rust) and 3000 (Slang). The files over it when the
  rule came in are in `tools/build-time/size-allowlist.txt`, each at most its listed
  size: they may shrink (`check.py --shrink` lowers the numbers), not grow. Split a
  file that has to grow, or raise its line in the same change, where review sees it.
  Large files cost twice: every edit recompiles all of it, and the editor and the
  reader carry all of it.

Also in CI (the web build's kernels job): `tools/build-time/inline_budget.py`, every
explicit module's inlined size (`scripts/wgsl-inline-size.py`: the entry point with every
call inlined, which a GPU driver's compile time follows) within its line in
`tools/build-time/inline-budget.txt` (its size when set, plus 10 %). Past it: shrink the
module (one call site per heavy helper, a split by dispatch kind:
`docs/measurements/kernel-compile-time.md`), or raise its line in the same change.

## Kernel source

- A loop cap names its loop by a permanent id (`EB_END(ID, ..)`, `EB_PASS(ID, ..)`),
  never `__LINE__`: a line added anywhere would change every module below it, and with
  them Chrome's and nvcc's caches. New loops take the next free id (the census test and
  `check.py` name it).
- A part that only some modules compile lives in a file of its own, included under its
  define (`#ifdef EB_AERO` / `#include "explicit_beams_aero.slang"`): the cache skips the
  include for modules without the define, so an edit there recompiles only the modules
  that have it. Moving code into such a file is checked with
  `kernel_digest.py` (WGSL identical; MSL and CUDA identical with `STRIP_LINES=1`).

## Measuring

- `python3 tools/build-time/measure.py [--only NAME,...] [--runs N]` (from `engine/`):
  for each of a fixed set of edits (kernel files, the soft-body model, authoring,
  the plan, the host, the World), the seconds from the edit to a built fast-loop
  harness, the crates compiled, the build scripts run and the slangc processes
  started. `CMD="cargo build --release ..."` measures another build.
- `python3 tools/build-time/kernel_digest.py OUT.txt` then `--compare A B`: every
  generated kernel's hash, the shipped set and the test set. Run it before and after a
  change to how kernels are built: anything but "0 changed, 0 removed, 0 added" needs
  an explanation in the commit.
