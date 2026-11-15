# Explicit kernels: first-load compile time in Chrome on Metal

Measured 2026-11-13 on an M3 Max, Chrome (Metal), branch `perf/kernel-compile-time`. The user's report: the two-car head-on with surface contact took minutes to load, looked frozen, and lost the GPU device during the wait.

## What made the compile slow

**Metal's compiler inlines every call into the kernel's entry point, and its compile time follows the size of that one function.** A module's size in bytes says little. A helper called from five places, inside a function called from four, is twenty copies.

`engine/scripts/wgsl-inline-size.py` measures this from the WGSL text alone (no GPU, no compiler): the entry point's size once every call is inlined, the functions that contribute most, and the callers that multiply them.

| Module (512 lanes) | Inlined lines before | After |
|---|---|---|
| frame, no surface contact (`explicit_beams_wide_nosurf`) | 145,652 | 145,652 |
| frame with surface contact (`explicit_beams_wide`) | 1,537,990 | 486,377 |
| the same without the coupling's proxy paths (`_nocpl`) | none | 199,501 |
| contact, with surface (`explicit_beams_contact_wide`) | 3,113,989 | 1,010,763 |
| the same, `_nocpl` | none | 437,011 |
| large contact groups, with surface (`explicit_beams_contact_big_wide`) | 1,586,149 | 534,536 |

Later the same day the centred damping's objective form (spec §39.6: `centredKeep`, `centredBeamObjective`) added about 2,056 inlined lines per frame loop, about 1 %: `explicit_beams_wide_nosurf` 147,708, `explicit_beams_wide_nocpl` 201,557, `explicit_beams_contact_wide_nocpl` 441,123. Chrome's compile time was not measured again.

The surface pass was ten times the rest of the frame. Inside it, the closest-feature queries (`slang/physics/feature_query.slang`) were copied hundreds of times: `fqSegTri` 80 times (5 call sites in `fqCandidates`, itself at 4 call sites in `fqContact`), `fqClosestTri` 208, `fqSegSeg` 300, `fqVisit` 528, and the strict multiply 72,000 times against 7,200 in a frame without surface contact. The contact module holds the surface pass twice (the solo vehicles' path and the groups' path).

## What changed (all lossless)

1. **One call site per heavy query** (`feature_query.slang`). Where the Rust twin calls a function several times in a row, the kernel runs one loop over the same calls in the same order and picks the arguments by index: `seg_triangle`'s six candidates, the candidate walk's shape edges, a contact's four passes, the overlap axes. Same operations, same order, same bits.
2. **No proxy paths without the coupling** (`EB_NO_COUPLING`, the `_nocpl` modules, WGSL only). An upload without the World's coupling can have no proxies, but its surface pass carried the per-element proxy loops, the casts and the pair path. They are compiled out for such uploads; the code that runs is the same text. With these modules the surface pass stays in the frame (one pipeline) instead of running as ten.
3. **No large-group module for a fleet of at most four vehicles.** It cannot form a group of more than four, so that dispatch never had a workgroup.
4. **Kernels compile one at a time, smallest first** (`warm_pipelines`), and the loading card shows each one. See "Compiles left behind" below.

Proof of bit-identity (native wgpu on Metal, CPU against GPU): `surface_contact_gpu` (3), `surface_pairs`, `explicit_surface_proxy_gpu` (4), `explicit_contact_gpu` (12), `engine-schema` `feature_query` (5), `explicit_loop_cap` (5).

## Chrome compile times

Cold compiles (`explicit-wgsl-check.mjs` with `EXPLICIT_WGSL_COLD=1`), one module at a time. The user's own page was running on the same GPU during these, so treat them as within a few seconds.

| Module | Before | After |
|---|---|---|
| `explicit_beams_wide_nosurf` (one car, no surface contact) | 7.2 s (user's page) | 7.0 s |
| one car with surface contact, softbody.html | about 45 s (user's page: the frame in segments plus nine surface pipelines) | 10.0 s (`explicit_beams_wide_nocpl`, one pipeline) |
| the nine surface pipelines (World pages, with coupling) | 17.3 s side by side; one alone up to 89 s (commit `3e0aa017`) | 0.6 to 1.7 s each, 11 s in all one after another |
| `explicit_beams_solo_wide` | still compiling at 78 s beside two others (user's page) | 31.8 s; `_nocpl` 10.3 s |
| `explicit_beams_contact_wide` | still compiling at 128 to 210 s beside two others | 166.3 s; `_nocpl` 33.6 s |
| `explicit_beams_contact_big_wide` | 243.1 s (user's page) | 52.3 s; not compiled for 4 vehicles or fewer |
| `explicit_beams_solo_wide_nosurf` | not measured | 7.0 s |
| `explicit_beams_contact_wide_nosurf` | not measured | 24.0 s |
| `explicit_beams_contact_big_wide_nosurf` | not measured | 12.9 s; not compiled for 4 vehicles or fewer |

What a first load compiles on softbody.html (no coupling), one after another:

| Scene | Before | After |
|---|---|---|
| one car, no surface contact | 7 s | 7 s |
| one car, surface contact | about 45 s | 10 s |
| two cars, no surface contact | solo + contact + large groups (about 44 s by the "after" times, not measured before) | 7 + 24 = 31 s |
| two cars head-on, surface contact | over 243 s with all three compiling at once; device lost on one attempt | 10 + 34 = 44 s |

Whole page loads, headless Chrome, a fresh profile, the generated sedan (`mod=/__mods/sedan_new`, surface contact on), from the built site (`explicit-load-time.mjs`), with other compiles and the user's page running on the machine:

| Page | Kernels | First frame |
|---|---|---|
| `xscene=sidepole` (one car) | `explicit_beams_wide_nocpl` 8.8 s | 13.8 s |
| the yard (several cars: the same kernels as the two-car head-on) | `explicit_beams_solo_wide_nocpl` 12.2 s, `explicit_beams_contact_wide_nocpl` 40.0 s, four plan passes under 0.1 s each | 59.6 s |

The targets were 10 s of compile for one car and 20 s for the two-car head-on with surface contact. The first is met (8.8 to 10 s). **The second is not: 44 to 52 s.**

## What is left for the two-car target

The contact module is the whole remaining cost, and it is two frames in one: the solo vehicles' path and the contact groups' path, so that one dispatch steps both side by side.

1. **Groups in a module of their own** (as the large groups already are), with the solo module stepping the vehicles outside a group. For two cars this costs no GPU time: either both are solo or both are in one group, so only one of the two dispatches has work. Estimated from the sizes above: about 10 s + 17 s with surface contact, 7 s + 13 s without. For larger fleets the two dispatches run one after the other where they ran side by side, which needs measuring before it is adopted there. It needs a flag in the plan pass's dispatch arguments (`vvArgs`) and care with the sleep list, which rewrites those arguments.
2. **The same one-call-site treatment for the surface pass itself** (`explicit_beams_surface.slang`): `fqContact`, `fqSegContact` and `fqPointContact` still have 6 copies each (static slots, proxies, the pair path), `surfApplyTo` 12, `surfCastToi` 6. This is what the World pages (with coupling) still pay: 32 s for the solo module, 166 s for the contact module.
3. **The frame without surface contact** has the same pattern on a smaller scale: `contactResponse` 14 copies, `tryContact` 6, `hitForce` 7 (about 70,000 of its 146,000 lines).

## Compiles left behind

On macOS the Metal compiler service (`MTLCompilerService`) goes on compiling a pipeline after the page that asked for it is closed, reloaded or has lost its device. It was seen at 100 % CPU for 9.5, 22 and 28 minutes after its page was gone. A cancelled or lost load cannot stop its compile, and the leftovers slow every later load. The pages therefore start one compile at a time: an abandoned load leaves at most one behind. The cost is that kernels no longer compile side by side (the two-car load above would be about 34 s if they did).

## Not verified

- Page loads in the user's own (windowed) browser after these changes, and the `xscene=headon` page itself (the yard above compiles the same kernels).
- `one-world.html`, `explicit-world.html`, `brick-ring-drive.html` and `vehicle-lab.html` were not loaded in a browser after their loading card and stop were wired in; they build, and the card and the stop panel are tested on their own (`load-progress-check.mjs`).
- Compile times on other GPUs and browsers. The inlined-size measure is about Metal; other drivers may not inline everything.
- `explicit_plan_modules` fails on `feat/sedan-body` before these changes (the shared module table lacks the surface split's modules) and still does; the `_nocpl` modules are WGSL only and are not in that table either.

## 2026-11-15: NVIDIA's driver, nvcc and Metal (the vehicle lab's mountain page)

Goal (owner): a cold compile of every pipeline the vehicle lab's mountain page needs
(generated sedan, page defaults: the frame module `explicit_beams_wide_nosurf` and the
deferred self-split modules `explicit_beams_selfsplit_{step,walk,pairs}_wide`) in under
60 s on NVIDIA (T4, Vulkan), proportionally less on Metal and nvcc. Lossless only: every
change bit for bit, runtime no worse.

### Where the time goes

GPU drivers inline every call, so a module costs what its entry point is after full
inlining. `tools/explicit-compile/inline_size.py MODULE.wgsl` counts it (non-blank WGSL
lines, each function's own lines times the call paths that reach it):

| Module | WGSL lines | after inlining |
|---|---|---|
| frozen reference `explicit_ref_solo` (T4: 2.5–3.3 s) | 2,966 | 26,764 |
| `explicit_beams_wide_nosurf` (T4: 171 s) | 15,766 | 262,497 |
| `explicit_beams_solo` (with surface, T4: 504–518 s) | 23,997 | 736,404 |
| `explicit_beams_contact_wide_nosurf` (T4: 1,024 s) | 22,793 | 561,783 |

Tested and ruled out:

- **Unrolling.** The substep loop's count is a run-time value (`gP.steps`); Slang keeps
  every loop a loop (136 in the frame's WGSL, 146 `for(;;)` in its CUDA). Forcing
  `#pragma unroll 1` on all 146 CUDA loops: PTX −1.3 % lines, nvcc 44.9 → 42.4–43.4 s.
- **wgpu's runtime checks** (bounds, loop bounding, division): no change on the T4
  (tools/modal/README.md).

Proxy (nvcc, sm_75 alone, Docker on the M3 Max; `tools/cuda/measure_nvcc.py`): marking
every function of 20+ lines `__noinline__` (`tools/explicit-compile/cu_noinline.py`)
takes the frame from 44.9 to 9.9 s, so the copies are the cost. One function at a time
(noinline, frame, s): cplHdr 38.2, tryContact 34.6, beamUpdate 38.8, selfTwoBodyCore 37.5,
ebEnd 36.1, triangleGeo 40.5. WGSL has no noinline, so the fix is fewer call sites in the
source: the same calls in the same order from one site (a loop over the cases, or a
select of the arguments), or a branch-free helper.

### Changes (each bit for bit; mountain A/B against the previous build's WGSL)

| Step | Frame lines after inlining | nvcc frame / step (s) | Metal cold, page's 4 modules (s) |
|---|---|---|---|
| baseline | 262,497 | 44.9 / 45 (alone); 60.3 / 60.8 (4 at once) | 19.4 (frame 9.3, step 9.5) |
| 1. cplHdr: always the workgroup copy (no branch, ~630 copies) | 235,407 | 34.9 | |
| 2. ebEnd / ebEndMost: selects, not ifs (~220 copies) | 226,851 | 33.5 | 15.1 |
| 3. the four accelerations of selfTwoBodyCore, resDrive, guardOf from one call site (opaqueCount loop); the node pass's ground and box contacts from one contactForce | 214,711 | | |
| 4. the brute and every-candidate tests from one site; the frame's listing walk without tests | 201,261 | 27.6 / 26.5 | 12.9 |
| 5. the step's listed pairs (reused, walk's, early) from one tryContact site; the testing walk from one site | 187,232 | 28.3 / 26.4 | |
| 6. one beamUpdate site for sorted and file order; selfSerial's hits and residual modes as one pass | 169,029 | 24.7 / 23.8 | 12.0 (frame 5.9, step 5.6) |

`opaqueCount(n)` is n plus `noFuseBits()` (0 at run time from a uniform), so no compiler
can see the trip count and unroll the merged call sites back into copies.

Strict arithmetic: every merged call passes values that are the results of strict
operations (`nf`, a bitcast XOR) or loads, never a plain product, so a merge cannot
change what a driver contracts into an FMA. (The coupling's two `contactForce` calls,
one with `f * share`, are left alone for that reason.)

Runtime, mountain (vehicle_lab --measure mountain-perf --self-split 1 --ab-switch kernel
--ab-kernel-dir <base WGSL>, native Metal, ms a frame, this kernel / base, one snapshot
each way): see the commit messages and the table in the final report.

### Not done / open

- Chrome runtime A/B (no harness runs two kernels from one snapshot in the page).
- The page could start on the self-split modules alone (no frame module), halving its
  cold compile; that changes the page's start, an owner decision.
