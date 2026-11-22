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
| 4. the brute and every-candidate tests from one site; the frame's listing walk without tests (**reverted**) | 201,261 | 27.6 / 26.5 | 12.9 |
| 5. the step's listed pairs (reused, walk's, early) from one tryContact site; the testing walk from one site (**reverted**) | 187,232 | 28.3 / 26.4 | |
| 6. one beamUpdate site for sorted and file order; selfSerial's hits and residual modes as one pass | 169,029 | 24.7 / 23.8 | 12.0 (frame 5.9, step 5.6) |
| **kept: 1, 2, 3, 6 without 4 and 5** | **196,508** | **27.5 / 26.3** (base 38.1 / 38.0 in the same session) | **14.0** (frame 6.8, step 6.6) |

Steps 4 and 5 are reverted: on NVIDIA (CUDA route, T4, `explicit_ab` with the base
build's fatbin loaded through `RUBBLE_CUDA_EXPLICIT_DIR`, generated sedan, tumble and
crash, 60 frames) the frame with them took 6.63 ms a frame against 5.50 (+20 %), bit for
bit; without them 5.53–5.55 against 5.59–5.60. (Bisected with fatbins per variant: steps
1–2 5.73, step 3 5.42–5.50, step 6's beam merge removed still 6.59.) Metal ran them
faster, so they were a register-allocation loss on NVIDIA's compiler, not more work. An
intermediate form of step 4 also made group_split's two-sedan head-on run past 420 s on
Metal (bit for bit); the form committed and then reverted ran it in 133 s.

### NVIDIA T4 (Vulkan, driver 620.70.06), cold, the page's four modules

`gpu_run.sh --allow-cold-compile --no-warm --no-shader-cache --cpus 8 -- -p engine-webgpu
--bin gpu_adapter -- --compile --jobs 8` (the four kept modules and the base build's
four side by side, a core each):

| Module | Base | Kept |
|---|---|---|
| `explicit_beams_wide_nosurf` (frame) | 186.2 s | 68.4 s |
| `explicit_beams_selfsplit_step_wide` | 182.8 s | 71.6 s |
| `explicit_beams_selfsplit_walk_wide` | 1.5 s | 1.5 s |
| `explicit_beams_selfsplit_pairs_wide` | 1.5 s | 1.5 s |
| sum (one after another) / longest (side by side) | 372 s / 186 s | 143 s / 72 s |

The 60 s bar is not met: 2.6x less, still 143 s one after another.

`opaqueCount(n)` is n plus `noFuseBits()` (0 at run time from a uniform), so no compiler
can see the trip count and unroll the merged call sites back into copies.

Strict arithmetic: every merged call passes values that are the results of strict
operations (`nf`, a bitcast XOR) or loads, never a plain product, so a merge cannot
change what a driver contracts into an FMA. (The coupling's two `contactForce` calls,
one with `f * share`, are left alone for that reason.)

Runtime, mountain (vehicle_lab --measure mountain-perf --self-split 1 --ab-switch kernel
--ab-kernel-dir <base WGSL>, native Metal, ms a frame, one snapshot each way), kept
kernel against base: t 9 5.48 / 5.58, t 20 5.45 / 5.55, t 36 5.64 / 5.86, t 44 5.71 /
5.89, every state word bit for bit.

### Not done / open

- Chrome runtime A/B (no harness runs two kernels from one snapshot in the page).
- The page could start on the self-split modules alone (no frame module), halving its
  cold compile; that changes the page's start, an owner decision.

## 2026-11-16: `char_move` on GitHub's macOS runners (the C library job's "Compilation failed")

The job "C library (macos)" failed about half its runs in `examples/hello`, which prewarms the character kernels:

```
In Device::create_compute_pipeline, label = 'char_move'
  Internal error: new_compute_pipeline_state: NSError { code: 2, localizedDescription: "Compilation failed", domain: "CompilerError", ... }
```

That `NSError` is all Metal returns (its `userInfo` holds only the same description). The reason is in the macOS unified log of the process (`engine/scripts/metal-compiler-log.sh`, which the CI job now runs when it fails):

```
10:25:46.500 MTLCompilerService (MTLCompiler) Compilation SUCCESS (TYPE=MTLBuildLibraryFromSourceToArchive - source): Time 983 ms
10:25:46.624 MTLCompilerService (MTLCompiler) Compilation SUCCESS (TYPE=MTLBuildFunctions - pipeline): Time 114 ms
10:25:57.595 hello (Metal) Execution of the command buffer was aborted due to an error during execution. Caused GPU Timeout Error (00000002:kIOGPUCommandBufferCallbackErrorTimeout)
10:25:57.597 hello (AppleParavirtGPUMetalIOGPUFamily) Failed to get compute pipeline state info
```

The runner is a virtual machine ("Apple M1 (Virtual)", `VirtualMac2,1`, 3 cores, 7 GB) with Apple's paravirtual GPU. The guest's compiler service finishes in about a second; the pipeline is then built outside the guest, and the guest's wait for it ends in a GPU timeout 10.5 to 11.0 s later (two attempts: 10.97 s and 10.49 s). It is not memory (81 % free, no memory kill), not concurrency (no other kernel was compiling), and no compiler crashed.

`char_move` was the one kernel whose build took about that long there:

- Four fresh runners, eight runs of `hello` each: the first run failed both attempts on all four; runs 2 to 8 passed on all four (15 to 22 s a run). One earlier runner passed ten of ten.
- Eleven earlier CI jobs: the first attempt failed in nine, the engine's one retry in five of those.
- A real GPU (M3 Max): five of five pass.

### The fix: one call site for the casts (inlined size 78,391 → 21,836 lines)

`wgsl-inline-size.py`: `char_move` was 78,391 lines once inlined. `castCandidates` (the capsule cast against the gathered candidates, 8,138 lines inlined) was written out six times (verify, step up, across, down, stick to floor, ground probe: 48.8k), and the cast straight down at a winner with its surface normal three more times (about 24k).

`char_move` now runs the six casts from one loop (`CAST_VERIFY` … `CAST_GROUND`) that picks each cast's arguments by index, and the cast down at the winner from one call (`castDownSurface`, lane 0, shared through `gsShare[1]`; the step-down's was computed by every lane before, with the same value). Same casts, same arguments, same order, same tolerances.

| `char_move` | Before | After |
|---|---|---|
| Inlined lines | 78,391 | 21,836 |
| Pipeline, M3 Max, cold (`gpu_adapter --compile <wgsl>`) | 7.09 s | 2.00 s |
| Pipeline, GitHub macOS runner (same command, four runners, three builds each, alternating) | fails, 12 of 12 (the GPU timeout, 10.7 to 15.1 s after the start) | 2.46 to 4.10 s, 12 of 12 (first build on a runner 3.59 to 4.10 s) |
| `hello` on a fresh GitHub macOS runner, eight runs each | first run failed both attempts on 4 of 4 runners | 64 of 64 runs pass with no failed attempt (8 runners, two workflow runs; first run 14 to 22 s, later 8 to 12 s) |

The runner numbers are from a temporary workflow on the fix's branch (runs 43854810639 and 43911463486, removed before the pull request): it ran `hello` eight times on each of four fresh runners, then built the old kernel's WGSL (main at 977ff3ac) and the new one's with `gpu_adapter --compile`. So the new pipeline has about 2.5 times the room under the roughly 10.5 s limit, where the old one was at or over it.

### Results against the old kernel

Compared on every `CharacterState` the `character` and `heightfield` tests of engine-webgpu decode (1,424 states, raw bits, native Metal; the tests are deterministic run to run):

- **Bit for bit in eight of the eleven character tests** (flat walk, stairs, elevator, moving platform and jump, debris, crush, CCD rocket, compaction) after two things the old straight-line code got from the compiler's algebra and the loop has to say itself: a vertical cast's length is `|disp.y|` (`castCandidates`'s `vertical`; the compiler knew the direction was the constant up axis, the loop's `length()` is the rounded root of a square), and the stepped move is summed as displacements (`Dh * across + up − down`, where `P3 − P` adds and removes `P`).
- **Not bit for bit on the two slope tests and the dynamic plank.** The first difference is the ground probe's hit distance against the tilted box, 0.8 µm apart with identical inputs (position and velocity equal in bits in that step): the shape cast's iteration, compiled in a different context, converging within its own tolerance. From there the 30° slope's character comes to rest 1.3 mm from where it did (3.1 mm apart at most on the way), the plank's 5 µm; the 55° slide's positions stay equal. Keeping the three casts down as three call sites did not change this (measured), so it is not that merge.
- All of `character`, `heightfield`, `queries`, `kinematic` (engine-webgpu) and `motion`, `prewarm` (engine-world) pass. engine-metal builds.

No tolerance, iteration count or contact was changed; the remaining difference is the platform compiler's floating-point code for the same source.

### Not done / open

- `query_batch` (63,092 inlined lines) and `projectile_step` (46,267) carry the same casts several times over. No CI job compiles them on the virtual GPU today; a host that prewarms them there would be near the same limit.
- Chrome was not measured.

## 2026-11-22: measuring without a browser, and the burn-down list

The vehicle lab's derby (`mode=derby&car=gen&cars=5`) and mountain pages create the same
28 pipelines (the fast setting). Cold in Chrome on an M3 Max (`explicit-wgsl-check.mjs`,
`EXPLICIT_WGSL_COLD=1`): 335 s in all, four modules past 10 s, every other at most 2.5 s.

| Module | Chrome s | native wgpu s | inlined statements | predicted s |
|---|---|---|---|---|
| `explicit_beams_contact_wide_nosurf` | 118.8 | 47.9 | 83,896 | 105.1 |
| `explicit_beams_selfsplit_member_vv_wide` | 114.0 | 45.8 | 88,441 | 115.4 |
| `explicit_beams_split_grouplevels_wide` | 49.3 | 32.9 | 51,765 | 45.0 |
| `explicit_beams_solo_wide_nosurf` | 41.1 | 15.4 | 45,525 | 35.9 |
| `explicit_beams_wide_hits` | 2.5 | 2.3 | 12,703 | 3.8 |

Three ways to measure, cheapest first:

- **Predicted, from the WGSL** (`tools/build-time/compile_cost.py`, a second, in CI):
  seconds = K x statements^B over the entry point with every call inlined, one-line
  helpers not counted. Fitted on the eight modules of 1 s or more: the four past 10 s
  within 15 %, the 1-3 s ones within 1.5x. Inlined lines fit worse (2x): `nf(x)` is
  `return x;` in the fast setting and costs the compiler nothing, but is 25-30 % of the
  lines. Other counts (loops, barriers, array accesses) predicted no better.
- **Native wgpu on Metal** (`EXPLICIT_WGSL_DIR=DIR cargo run -p engine-webgpu-lab
  --example explicit_compile_time`, no browser; it takes the GPU lock itself, so do not
  wrap it in `with-gpu-lock.py`): the driver's real compile, but of naga's MSL, not
  Tint's. Same ranking, 1.5-2.6x faster than Chrome on the large modules: a lower bound.
- **Chrome** (`explicit-wgsl-check.mjs`, EXPLICIT_WGSL_COLD=1): the user's first load.

Ruled out: Apple's offline compiler (`xcrun metal -c` on the module's MSL) takes 1 s for
the 41 s solo module. The time is in the driver's back end, which runs only when a
pipeline is created on the device.

`compile_cost.py --burndown` over the 142 shipped modules: 70 predicted past 10 s (the
surface-contact, wing and coupling variants too). The helpers that cost most across them,
each as if it had one copy: `driveAheadPart` (5 call sites in the frame step, 6 copies of
3,047 statements), `beamSum` (275 copies of 111, through `vvBeamSumOf`), `drvTyre`,
`surfaceStep` (twice in the contact module), `ebEnd` (1,358 copies of 12), `toFixed`
(882 of 11), `surfApplyTo`, `tryContact`.

### The burn-down from the compiler's own output (`tools/build-time/kernel_profile.py`)

The static count overstates a call that passes a constant: splitting `driveAheadPart`
(5 call sites, each with a constant `part`) by its part cut the counted statements of
`explicit_beams_solo_wide_nosurf` by 24 % and changed nothing the compiler sees (its
optimized entry point 83,059 instructions both ways; Chrome 41.1 -> 37.9 s, noise). The
driver folds each site down to its branch before the costly passes.

`kernel_profile.py` measures what is left: the module through naga's MSL, Apple's
offline compiler and `air-opt` (every call inlined, `-O2`), the entry point's
instructions, and from its line tables each source function's instructions, the copies
that survived and the call paths that made them (`--attr N --who FN`). 15 s a module,
no GPU, no browser (macOS with Xcode). Chrome's cold compile follows it as
`1.07e-8 x instructions^1.914` s (the 28 lab modules: within 1.3x on average): 10 s at
about 49,000 instructions.

| Module (fast setting) | optimized instructions | Chrome cold |
|---|---|---|
| `explicit_beams_contact_wide_nosurf` | 173,434 | 118.8 s |
| `explicit_beams_selfsplit_member_vv_wide` | 172,170 | 114.0 s |
| `explicit_beams_split_grouplevels_wide` | 123,679 | 49.3 s |
| `explicit_beams_solo_wide_nosurf` | 83,059 | 41.1 s |

First fix, from its list: `fleetLevelsAny` was four copies of 15,625 instructions in the
levels kernel, half of it: `fleetLevelsOnce` and `fleetLevelsLat` each called it at two
fallbacks (a hit list past its room) and returned. They now return "fall back" and their
callers call it once (`fleetLevelsOnceTry`, `fleetLevelsLatTry`); nothing ran after a
fallback, so the same work in the same order. `explicit_beams_split_grouplevels_wide`
73,206 instructions, Chrome cold 49.3 -> 15.4 s; `explicit_beams_split_group_wide`
predicted 230 -> 108 s. Checked: fast-check against the previous kernels (16 of 17
fixtures bit for bit, `cars1_t20.0` differing as with the same kernels on both sides),
the scenario matrix (12 rows bit for bit, none slower; `--ab-switch kernel` now times its
halves in rounds as the other switches do), Chrome's validation of all 140 modules in
both settings.

### Tried and dropped: masking the experiment-only switches

Five perf switches are on in no default (drive-ahead, the levels across the GPU,
tyre-early, self-hits-local, split-wheels-ahead), yet every pipeline carries their code:
the flags word is a run-time uniform. Making the pipelines with each read of the word
masked (`gP.perfFlags & mask`) was bit for bit (fast-check, the widening path, the
scenario matrix, Chrome's validation) and cut the offline optimized size (solo 83,059 ->
70,536, member 172,170 -> 150,870, contact 173,434 -> 160,905 instructions), but not
Chrome's cold compile: solo 41.1 -> 39.9 s, contact 118.8 -> 129.9 s, member 114.0 ->
123.0 s (noise). Like splitting `driveAheadPart` by its constant, code the compiler can
only drop by folding a constant does not shorten Chrome's compile; removing copies from
the source (`fleetLevelsAny`, 49 -> 15 s) does. So `kernel_profile.py`'s burn-down is read
for duplicated copies, and a fix is judged by Chrome's cold compile, not the profile.

### Every vehicle a group member: the members' module without the solo step

The split group path's vehicles' module (`explicit_beams_split_member`, the vehicle
lab's `explicit_beams_selfsplit_member_vv_wide`, compiled before the page runs) held two
steps: a solo vehicle's (`explicitBeamsFrameStep`) and a group member's
(`contactMemberStep`), each returning at once for the other kind. With vehicle contact,
the grouping pass (`explicit_beams_vv_group`) now makes every vehicle a member, one
without partners a group of its own, and the module holds the member step alone. A
group past `VV_MAX_GROUP` members is the host's error as before (`contact_status`:
"more than 64"); its vehicles still run alone in the contact module, and on the split
group path are not stepped that frame. Any other vehicle reaching this module in no
group fails the frame loudly (loop-cap id 599). The contact module (deferred on the lab's
pages) keeps its solo half: dropping it would turn that error into a failed frame.

| | before | after |
|---|---|---|
| optimized instructions (`kernel_profile.py`) | 172,170 | 88,433 |
| Chrome cold compile | 114.0-123.0 s | 32.0-34.1 s |

The member step gives a solo vehicle's results bit for bit, and a group of one costs
the same at run time. Checked against the previous kernels: fast-check (16 of 17
fixtures bit for bit, the 17th the known quirk), with the split group path off too
(`--ab-switch groupsplit` in the new build: bit for bit), the scenario matrix (12 rows
bit for bit, none slower), `sleep-check.sh` (21 checks), the contact, group-split and
vehicle-grid GPU suites in bit-exact mode (18 tests, including a group over the limit),
Chrome's validation of all 140 modules in both settings.

### The members' module: one selfApply

The members' module's entry runs either a member's step (`splitMemberHead`, through
`contactMemberStepHead`) or, in the self split's rest dispatch, the member's rest
(`memberRestStepHead`); each ended in the same tail with its own `selfApply`. The tail is
now one function the entry calls once (`memberTail`,
`slang/kernels/explicit_beams_member_tail.slang`), the heads returning what it is to do
(the `MT_*` bits: Chrome sees a barrier's uniformity through a return value, not through
the struct handed back by pointer, which it rejected). `contactMemberStep` keeps its
form for the contact module (its head, then the tail).

`explicit_beams_selfsplit_member_vv_wide`: 88,433 -> 76,220 optimized instructions, Chrome
cold 32.0 -> 26.3 s. Checked against the kernels before the members' changes:
fast-check (16 of 17 bit for bit, the known quirk), `groupsplit` in the new build, the
scenario matrix (12 rows bit for bit, none slower), `sleep-check.sh` (21), the
group-split, self-split and contact GPU suites in bit-exact mode (24 tests), Chrome's
validation of all 140 modules in both settings.

### The residual and the guard share their beam sums

A staged hit's residual (`vvResOf`) and its guard (`vvGuardOf`) each summed the beam
forces of the hit's four nodes (`vvResAcc`, through `vvBeamSumOf`): eight copies a call
site, and eight sums at run time where four give the same values (nothing writes the
beams' forces between the two). The four sums are now taken once, in one loop
(`vvSumsOf`, `slang/kernels/explicit_beams_vv_sums.slang`), and both read them.
`explicit_beams_split_grouplevels_wide`: 73,206 -> 65,284 optimized instructions, Chrome
cold 15.4 -> 12.9 s. Checked: fast-check (16 of 17 bit for bit, the known quirk), the
scenario matrix (12 rows bit for bit, none slower; the contact rows 1-9 % faster), the
group-split, fleet-contact and contact GPU suites in bit-exact mode (17 tests), Chrome's
validation of all 140 modules.

Tried and dropped: one function for the members' two drive schedules (the normal step
and PERF_SPLIT_WHEELS_AHEAD's tyres alone, one copy of `drvTyre`). Bit for bit, 76,220 ->
70,924 optimized instructions, but Chrome cold the same (before 25.4 / 25.8 s, after
25.4 / 25.7 s, alternating): the copy it removed is not where the compile spends its
time. The profile ranks candidates; only a cold compile decides.

### The levels kernel: one call site for the snapshot's nodes and the lane's residual

Five places snapshot a hit's four nodes (`vvSnapNode`) with four calls, or pick one of
two calls by the node; each is now one call with the node chosen by index (in a loop
kept a loop by `opaqueCount`), the same calls in the same order. `latResidual` was called
twice, for the lane's own hit and for another; it is now one call on a copy, kept again
only for the lane's own (another's copy was dropped before too).
`explicit_beams_split_grouplevels_wide`: 65,284 -> 56,910 optimized instructions, Chrome
cold 12.9 -> 10.3 s (two samples each step). Checked: fast-check (16 of 17 bit for bit,
the known quirk), the scenario matrix (12 rows bit for bit, none slower), the
group-split, fleet-contact, contact and vehicle-grid GPU suites in bit-exact mode (19
tests), `sleep-check.sh` (21), Chrome's validation of all 140 modules. Tried and dropped:
`opaqueCount` on `vvSumsOf`'s loop (the same instructions and time: not unrolled).

### The levels kernel under 10 s: one ordered application for both scans

`vvApplyOrderedBuckets` was `vvApplyOrdered` with each hit's earlier and later hits of
its point searched among its bucket's instead of all n: the same answers (any earlier
hit, the smallest later one), so the same tries in the same order. Both are one function
now (`vvApplyOrderedIn`, `bk` choosing the range), and the small groups' pass calls it
once: one copy of `vvTry` where it had two. `explicit_beams_split_grouplevels_wide`:
56,910 -> 53,664 optimized instructions, Chrome cold 10.3 -> 9.8 / 9.6 s. Checked:
fast-check (16 of 17 bit for bit, the known quirk), `groupsplit`, the scenario matrix
(12 rows bit for bit, none slower), the group-split, fleet-contact, contact and
vehicle-grid GPU suites in bit-exact mode (19 tests), `sleep-check.sh` (21), Chrome's
validation of all 140 modules.

### The members' module: what its compile spends, by ablation

Cutting one part of the member step at a time (measurement builds, their results
meaningless) and compiling cold in Chrome: all of it 26 s; without the coupling's step
24.9 s, the wheels' 24.4 s, the hits' tail 18.6 s, the self split's tests
(`selfSplitTests`) 12.1 s. The tests cost half the compile for a fraction of the
instructions; inside them the pieces' tests (`piecesTests`) were called at two places
(the building step and a listed one) and the grid's walk at two (the self split's
fallback past the list's room, and the step's own).

Both are one call site now: `selfSplitTests` calls `piecesTests` once, and returns "walk"
(`selfSplitTestsOrWalk`) for the member step to run its own grid walk instead of a copy.
The same tests in the same order. `explicit_beams_selfsplit_member_vv_wide` Chrome cold
26 -> 19.8 s (the pieces) -> 16.5 s (the walk), while the instructions fell only 76k ->
73k: which copies cost the compile shows in a cold compile, not in the count. Checked:
fast-check (16 of 17 bit for bit, the known quirk), `groupsplit`, the scenario matrix (12
rows bit for bit, none slower), the self-split, group-split, contact and pair-reuse GPU
suites in bit-exact mode (26 tests), `sleep-check.sh` (21), Chrome's validation of all
140 modules.

What is left of the members' module's 16.5 s, by the same ablation: the pieces' tests
4.5 s, the grid walk 1.8 s, the brute pairs 1.8 s, the single triangle test 0.2 s; the
hits' tail (selfApply: its rounds and serial residual, one copy each) about 7 s. The tests'
cost is `tryContact` (about a thousand instructions with its force) inside each
strategy's own nested candidate loops: one call site for it across the strategies means
the strategies producing pairs for one consuming loop, a change to the narrow phase's
shape (and its run time) of its own.
