---
name: fast-kernels-and-builds
description: Write and change Slang kernels, kernel plumbing and Rust host code so that runtime stays fast, Chrome's first-load compile stays under 10 s a kernel, and an edit rebuilds little. Use before adding or editing any Slang kernel or helper, splitting or adding a source file, adding a kernel variant or define, touching engine-slang-build / engine-kernel-manifest / build scripts, or when CI fails a build-time check (check.py, compile_cost.py, inline_budget.py, fanout.py, the Chrome WGSL check).
---

# Fast kernels and fast builds

Runtime performance comes first; never trade it (or physics results) for build time.
Everything here is lossless: same results, less work for the compiler or the build.
Background and numbers: `engine/docs/build-architecture.md`,
`engine/docs/measurements/build-time.md`, `engine/docs/measurements/kernel-compile-time.md`.

## What makes a kernel slow to compile

A GPU compiler inlines every call into the entry point. Chrome's (and Metal's) compile
time grows faster than linearly with what is left after inlining, so the cost is
**copies**, not lines: a 300-statement helper called from 5 places is 1,500 statements
for the compiler. Code removed only by constant folding (a constant argument, a masked
uniform) does not shorten Chrome's compile, so splitting a helper by a constant
argument gains nothing.

### Rules for Slang

1. **One call site per heavy helper in a kernel.** Instead of `if (a) f(x); else if (b)
   f(y);`, choose the arguments, then call once: `mode`/`count`/`stride` variables and one
   loop (`triangleCandidates` with `cmode`; `splitDrive`; `memberTail`). Helpers return
   decisions (bits, counts) and one place acts on them.
2. **A barrier must be reached by uniform control flow.** Chrome rejects barriers that
   naga accepts. Uniformity survives through return values, not `out` parameters. Read
   shared decisions through `gsUniform*` (rewritten to `workgroupUniformLoad`,
   `uniformize_workgroup_lanes`; add new names to its list).
3. **Every loop has a cap with a permanent id**: `EB_END(ID, ..)`, `EB_END_UNIFORM`,
   `EB_PASS(ID, guard)` as the loop's first statement; ids are unique (`check.py` names
   the next free one). Never `__LINE__`: a moved line would change every module below it
   and miss every cache.
4. **An unconditional `EB_PASS` in a hot inner loop can cost seconds of compile**; use
   `EB_END_UNIFORM` with `ebSpend()` where the loop was budgeted before.
5. **A feature path goes in its own function behind an early return**, or its own file
   behind its define, never threaded through a shared function's body.
6. **Files stay near 500 lines, 1,000 at most.** Split by `#include`ing parts at the same
   place, cut between top-level sections, outside any `#if` and any function; keep an
   include guard's `#endif` in its file. Prove it with `kernel_digest.py` (below).
7. **Include a part only where it is used** (`#ifdef EB_X` / `#include "..._x.slang"`):
   the cache key is every file a compile includes, so an unconditional include makes an
   edit rerun every module that has it (`fanout.py` shows the count).

### When a merge for compile time misbehaves natively

Twice a logically identical merge in the solo frame (the pieces' loop, the wheels'
call site) failed native suites (`explicit_pair_reuse_gpu`, `self_split_gpu`) while
Chrome compiled it faster. Do not keep such a change: run the suites in bit-exact mode
after every merge, and drop the merge if they fail. Write down the result in
`kernel-compile-time.md`.

## Rules for Rust and the build

- Crate direction: leaf data crates (`engine-kernel-manifest`, `engine-schema`) stay
  leaves; build scripts depend only on `engine-slang-build` and the manifest; runtime
  crates never depend on `engine-softbody-authoring`.
- Only `engine-slang-build` spawns slangc or nvcc (cached by content, batched).
- Functions at most 200 lines, files at most 1,000 (`check.py`; allowlists only shrink).
- Tools that read kernel source follow `#include` (`bind_slots`, `vk_binding_slots`, the
  loop-cap census): never assume one file holds a kernel.
- The fast loop (`--profile fast`) compiles kernels at run time into a content-addressed
  store; `cargo run --profile fast -p engine-webgpu --bin kernels` builds it before running
  a test binary directly.

## Checks (cheapest first)

| Check | Command (from `engine/`) | Time |
|---|---|---|
| Architecture, sizes, loop ids | `python3 tools/build-time/check.py` | 1 s |
| Predicted Chrome compile, 10 s cap | `python3 tools/build-time/compile_cost.py --check DIR` | 10 s |
| One-call-site rule | `python3 tools/build-time/compile_cost.py --dup-check DIR` | 10 s |
| What to fix first | `python3 tools/build-time/compile_cost.py --burndown 15 DIR` | 10 s |
| Where a kernel's code comes from (Mac) | `python3 tools/build-time/kernel_profile.py MODULE.wgsl --attr 25 --who FN` | 15 s |
| Edit fan-out | `cargo run -q --profile fast -p engine-webgpu --bin kernels -- --closures F && python3 tools/build-time/fanout.py F --check` | 5 s |
| Chrome's WGSL rules | `cargo run ... --bin kernels -- --browser-wgsl D` then `EXPLICIT_WGSL_DIR=D EXPLICIT_WGSL_MODULES_ONLY=1 node crates/engine-webgpu/explicit-wgsl-check.mjs` | 10 s |
| Refactor is lossless | `STRIP_LINES=1 python3 tools/build-time/kernel_digest.py A.txt` before and after, `--compare A.txt B.txt` | 1 min |
| Real compile time | Chrome cold: `EXPLICIT_WGSL_COLD=1 EXPLICIT_WGSL_DIR=D node ... explicit-wgsl-check.mjs` on `explicit_wgsl_dump -- --fast` output, A/B/A/B | minutes |
| Edit-to-ready | `python3 tools/build-time/measure.py --budget` | 10 min |

`DIR` is the build's WGSL: `cargo run -q --profile fast -p engine-webgpu --bin kernels`
prints it. Judge a compile-time change by the cold Chrome compile, never by the
prediction alone; keep it only if Chrome improves and the change is bit for bit (the
ladder in AGENTS.md: fast-check, scenario matrix, suites, sleep check).

## When CI fails a build-time check

- **Over a size line**: split the file (rule 6), or raise the line in the same change
  with the reason, where review sees it.
- **compile_cost / dup-check**: `--burndown` and `kernel_profile.py --who` name the
  helper and its call paths; give it one call site.
- **fanout**: a file now reaches more compiles; include it only where used, or add the
  new file with `fanout.py --init` (its number shows in review).
- **Chrome WGSL**: usually a barrier after a branch on a non-uniform value (rule 2).
