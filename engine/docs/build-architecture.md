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
- **An environment variable is a rerun trigger only if the output depends on it.**
  (`RUBBLE_SLANG_CACHE` changes where outputs come from, not what they are.)

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
