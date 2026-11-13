# GPU overload: how Rubble can take the machine's GPU away, and what to do about it

Written 2026-11-13 after the vehicle lab lost the GPU device and macOS logged the user out. It applies to every back end: WebGPU in the browser, and native wgpu, Metal and CUDA.

**Status: documented, not solved.** Nothing here was run on a GPU. Everything comes from reading the code and from runs of the CPU reference. A prototype of three guards exists on the branch `fix/gpu-overload-guards` (commit `dd586715`, labelled "prototype, not adopted"); it is unverified and is not part of the engine.

Each claim below is marked where it matters:

- **Read:** the code says so (file and line given).
- **Measured (CPU):** a run of the CPU reference in this investigation.
- **Observed:** what the user or an earlier session saw on the machine.
- **Inferred:** reasoning from the above, not checked on a GPU.
- **General knowledge:** how the platforms are known to behave, not re-checked against their documents when this was written. Verify before relying on a number.

## Summary

- **Submitted GPU work cannot be taken back.** Closing the page or killing the process does not stop it on Apple Silicon. The compositor shares the GPU, so the desktop stops with it.
- **Rubble has no limit on the GPU time one World step may take, and no limit in time on the steps queued.** The readback gate limits the queue by count (3 steps; 8 in the vehicle lab). The kernel loop budget is 2^22 to 2^31 passes for each of up to 512 invocations of a vehicle's workgroup, which is seconds to minutes of GPU time.
- **Nothing stops a page from stepping a broken simulation.** On the CPU reference the lab's default mountain run reaches a non-finite (NaN) car state after 54.6 s. The CPU lab stops there. The page shows a line in the HUD and keeps submitting.
- **Most likely cause of the 2026-11-13 crash** (not confirmed): the car's state degraded on the mountain (the default-on rockslide striking it at rest, parts flung hundreds of metres, then NaN), frames from that state cost far more than a normal one, and the page kept eight of them in flight and refilled the queue every displayed frame. Second candidate: a pipeline compile stall, which was seen on its own the same night on another page.
- **A separate failure the same night was not overload:** `softbody.html` lost the device while Metal was still compiling very large kernels (over two minutes for one module).
- **Recommended order of work:** stop on invalid state, pace by completed work with two steps in flight at most, pause on a wall-time budget, handle device loss without panics, then tie the pass ceiling to measured throughput, then split long dispatches and shrink the kernels.

## 1. What happened

### 2026-11-13, about 02:35: the vehicle lab, Mountain mode (Observed)

Page `engine/crates/engine-webgpu/vehicle-lab.html` on `feat/vehicle-lab` (head `5d4ba111`), in the user's Chrome: the generated sedan (554 nodes, 4,228 beams), surface contact on, Mountain mode at its defaults (50° slope, rockslide on).

1. The lab ran for a while, the car tumbling down the slope.
2. The console showed `device lost (Unknown): A valid external Instance reference no longer exists`.
3. Then `no WebGPU adapter…`.
4. `ioreg` showed the GPU's "Device Utilization %" at 100 for at least 10 s with no GPU page or process running.
5. Shortly after, macOS ended the session and logged the user out.

Minutes before, the lab's pacing had been changed (`899461d0`, `26f5cb98`, `32b98ec1`): eight steps' results in flight instead of three, and steps the World refused carried over to the next frame. The page had not been run headless with those changes; the lab's own status section lists "the mountain for 5 minutes" and "a tiny loop budget ending a step with its error" as not yet verified (`engine/docs/vehicle-lab.md`, Status).

### 2026-11-13, about 02:50: `softbody.html`, two cars head on (Observed)

The older page, generated sedan, `xscene=headon` (two cars), surface contact on, first load of those kernels:

1. The loader showed "Compiling GPU solver kernels (4 of 7 done) 78 s".
2. The console later showed `explicit_beams_solo_wide 127.8 s`.
3. Then `pipeline warm-up failed (explicit_beams_frame / explicit_beams_contact): GPUPipelineError: A valid external Instance reference no longer exists`, and "The GPU device was lost".
4. Then a wasm panic inside wgpu: `createBuffer … size (582176) is too large … mappedAtCreation`, ending in `unreachable`.

At that moment two `MTLCompilerService` processes were at 100 % CPU, still compiling after the loss. The machine's load average was 12–18: the compiles, plus a CPU-reference run of this investigation. GPU utilisation was about 27 %.

**This one is not GPU overload.** It is the compile of very large kernels taking minutes, during which Chrome lost or killed its GPU process. Section 3h covers it.

### Earlier incidents

| Date | Trigger | Mechanism | How it was noticed | What fixed or guarded it |
|---|---|---|---|---|
| 2026-11-02 | A 64-car town run with one workgroup per vehicle | One long Metal dispatch | The Mac froze | Wide passes for big vehicles; runs bounded to 10–20 minutes with a kill timer |
| 2026-11-06/07 | A coupled vehicle crossing 100 m from its origin | `cplRebase` read the coupling header from workgroup memory before it was written; garbage counts fed nested loops | GPU at 100 % with nothing running; reboot | `cd371f29` (read the header from the buffer), `26e3b700` (counts held to the region's room). The fix sat unmerged on one branch for days while other branches hit it |
| 2026-11-07 | Bad state (NaN, corrupt counts) in the explicit kernels | Nested loops capped at 2^20 passes each multiply | Three reboots in one day | Per-frame pass budget (`01c50662`, `c2fad337`), the loop-cap status word, `World::step` failing on it, the loop census test, the GPU lock |
| 2026-11-07 | A loop with a barrier that did not end alike on every lane | Non-uniform control flow | Same day | Uniform caps (`ebEndMost`) |
| Before 2026-11-07 | A large pipeline compiled synchronously in Chrome | The compile ran on the GPU process's main thread past its watchdog | "Device lost" | `warm_pipelines` (`createComputePipelineAsync`) |
| 2026-11-13 02:35 | The vehicle lab, Mountain | Section 4 | Device lost, GPU at 100 % with no process, logout | Not fixed |
| 2026-11-13 02:50 | `softbody.html`, two cars, cold kernels | Compile of very large kernels for minutes | Device lost during loading, then a wasm panic | Not fixed |

## 2. The problem, stated generally

A real-time engine that computes on the GPU can ask for more work than the GPU finishes, in two ways:

- **Too many submissions:** more steps queued per second than complete per second. The queue grows.
- **One submission that is too long:** a single dispatch that runs for seconds.

Either is harmless on a CPU, where a thread can be stopped. On a GPU it is not, for four reasons.

**Submitted work cannot be recalled.**

- **Metal (Observed here, four times):** a committed command buffer runs to its end. Killing the process that made it does not stop it. Only a reboot cleared a runaway kernel.
- **Vulkan and Direct3D 12 (General knowledge):** neither has a call that cancels work already submitted to a queue. A driver or the OS may reset the device after a timeout. On Windows that is Timeout Detection and Recovery, about 2 s by default: the device is removed and every application using it loses its device.
- **CUDA (General knowledge):** a running kernel cannot be interrupted from the host. Destroying the context or killing the process ends it on Linux and Windows.
- **WebGPU (General knowledge):** the API has no cancel. The specification's security section treats long-running shaders as a denial-of-service risk and leaves the defence to the implementation, which may lose the device.

**The display shares the GPU.** The window server composites every window on the same GPU. On Apple Silicon there is one GPU and no pre-emption that we can rely on, so a kernel that holds it holds the desktop. (Observed: the Mac froze on 2026-11-02; the session ended on 2026-11-13.)

**The protections act after the fact.**

- **Chrome (General knowledge):** a watchdog thread in the GPU process kills the process when its main thread does not respond for some seconds (on the order of 10–15 s). Every WebGPU device then reports loss; the message "A valid external Instance reference no longer exists" is what a page sees when the GPU process is gone. The watchdog kills a process. It does not stop work already handed to the driver (Observed: 100 % utilisation with no process).
- **macOS (General knowledge, consistent with what was observed):** a watchdog restarts the window server when it stops responding for long enough, which ends the login session.
- **Sandboxing** covers memory, files and the network. It does not ration GPU time.

**Killing our own process is not a remedy.** A kill timer bounds how long a run submits. It does not stop what is queued.

**Why a physics engine is the bad case:**

- It submits many compute dispatches every frame, forever, with no natural pause.
- The loops' counts depend on the simulation's state: contacts, proxies, grid cells, pairs. A state that breaks (NaN, an explosion, a part flung far away) changes the cost of the next frame, and the next frame is submitted before the host has read anything back.
- A simulation that falls behind real time is tempted to catch up, which submits more work exactly when the GPU is slow.
- One vehicle runs all 33 of its 2 kHz substeps in a single dispatch, so the unit of work that cannot be interrupted is a whole World step.

## 3. Failure modes in Rubble today

"Today" is `feat/vehicle-lab` at `5d4ba111`.

### (a) Submit-rate runaway and catch-up pacing

**How a page paces (Read).**

- `FramePacer::advance(dt, slow, max)` (`engine/crates/engine-world-wasm/src/frame_pacer.rs:119`) turns wall time into steps owed and caps them per frame. The pages pass `max = 4`.
- Its GPU budget (the measured milliseconds per step against the frame) only applies when the page gives a frame-rate floor. `new FramePacer(rate, 0)` has none. The vehicle lab passes 0 (`vehicle-lab.html:347`); `brick-ring-drive.html:655`, `one-world.html:459` and `explicit-world.html:261` pass `?minFps`, which defaults to 0. So on these pages the pacer never looks at GPU time.
- The only gate on a step is `WebWorld::canStep` (`engine-world-wasm/src/lib.rs`), which is `World::can_step_without_waiting`: a readback slot is free (`engine-world/src/lib.rs:2182`). A slot frees when its `mapAsync` resolves.
- Slots: `WorldDef::readback_frames`, 3 by default (`engine-world/src/lib.rs:246`). The lab sets 8 (`engine-world/src/vehicle_lab/mod.rs:1772`).

**The lab's loop (Read, `vehicle-lab.html:806-830`).** Each displayed frame asks for `pacer.advance(dt, 1 / scale, 4) + debt` steps, where `debt` is up to 3 steps the World refused last frame. That is up to 7 `World::step` submissions in one frame, each a rigid step plus the vehicles' 33-substep frame. The frame then submits its render pass and, at most, one page readback.

**What that bounds, and what it does not.**

- The queue is bounded **in count**: 8 World steps in flight in the lab, 3 elsewhere, plus the frames being drawn.
- It is not bounded **in time**. The gate does not know what a step costs.

| GPU cost of one step | Queue at 8 in flight | At 3 in flight |
|---|---|---|
| 3.7–4.0 ms (lab, idle; Measured earlier, headless Chrome) | 32 ms | 12 ms |
| 6.5–8.9 ms (lab, in contact) | 71 ms | 27 ms |
| 100 ms | 0.8 s | 0.3 s |
| 1 s | 8 s | 3 s |

- At the measured cost the lab is not saturated: 60 steps/s × 8.9 ms is 534 ms of GPU time a second, plus drawing. The pacing change alone does not explain a wedge.
- When a step's cost rises above the displayed frame (16.7 ms), the page fills all its slots and keeps them full: a slot frees, the next frame submits into it. The GPU stays at 100 % for as long as the page runs, and when the page dies the queue still drains. The eight-slot gate and the owed steps make that queue 2.7 times deeper than before, and make the page refill it faster.
- The readback that frees a slot is also the only way an error reaches the host. With 8 in flight, a frame that failed is known 8 steps later (Read: the loop-cap flag and the coupling status are extras of the step's own readback, `engine-world/src/lib.rs:2380-2389`, decoded in `World::frame`, `lib.rs:2986-2996`).

**Which pages share it (Read).**

| Page | Steps asked per frame | In flight | Carries refused steps |
|---|---|---|---|
| `vehicle-lab.html` | 4 + 3 owed | 8 | Yes (`debt`) |
| `brick-ring-drive.html:692-695` | 4 | 3 | No |
| `one-world.html:489-492` | 4 | 3 | No |
| `explicit-world.html:300-304` | 4 | 3 | No (it counts them) |
| `softbody.html:2125-2137` (World mode) | 2 × steps per frame | 3 | Remainder capped at one frame's share |
| `softbody.html` (standalone `ExplicitSim`) | The pacer's, with a GPU budget only with `?minFps` | Its own; not audited here | – |
| `vehicle.html:1483-1486`, `demo.html`, `city.html`, `impact.html`, `motion.html`, `showcase.html` | Fixed or pacer | 3 | No |

All of them gate on a free readback slot and none on GPU time. The lab is the only one with 8 slots and owed steps.

**Native (Read).** `World::step` itself has no gate. With `Backpressure::Block` (the default) a step with no free slot waits for the oldest (`wait_oldest`, `lib.rs:2679`; `ReadbackRing::ready` polls with `wait_indefinitely`), so the host thread stops until the GPU finishes: back-pressure, at the price of a hung game thread. With `Backpressure::Drop` the step is submitted without a readback (`lib.rs:2412-2419`), so a native host that calls `step` every frame in Drop mode has no limit at all on what it queues. Only the wasm wrapper checks `canStep` first.

### (b) Long single dispatches: the loop budgets

**The mechanism (Read).** Every loop in the explicit kernels is capped at 2^20 passes (`EB_LOOP_CAP`, `engine/slang/kernels/explicit_loop_cap.slang`). Nested loops multiply, so each invocation also draws from a per-frame pass budget (`ebSpent`). The host sets the budget per frame:

- Explicit frame: `loop_budget(unit, steps) = clamp(2 × unit × (steps + 2), 2^22, 2^31)` (`engine-exec/src/explicit_plan.rs:182`), where `unit` sums, per vehicle, `nodes × (coupling shapes + planes + 16) + 4 × beams + triangles × candidates + 1024` (`explicit_plan.rs:1857-1871`).
- Coupling passes: `clamp(4 × (colliders + 33 × hash + s × (s + planes + bodies) + cand² + 64 × vehicles + 4096), 2^22, 2^31)` (`engine-world/src/explicit_vehicles.rs:134`).

**The budget is per invocation.** A vehicle's workgroup has 512 lanes for fleets of up to 32 vehicles (`SMALL_FLEET`, `explicit_plan.rs`), 256 otherwise. Each lane may spend the whole budget. Loops that contain barriers are not budgeted at all; they take their real maxima (`ebEndMost`).

**The lab's numbers (computed from the formula).** Coupling room 256 shapes and 4,096 planes (`vehicle_lab/mod.rs:82-86`), 33 substeps:

| Car | `unit` | Budget, passes per invocation per frame | All 512 lanes |
|---|---|---|---|
| Generated sedan (554 nodes, 4,228 beams; triangle and candidate counts not in the repository, so the last term is left out) | ≥ 2.44 million | ≥ 171 million (2^27.3) | ≥ 8.7 × 10^10 |
| Authored sedan (987 nodes, 6,865 beams, 608 triangles, 306 candidates) | 4.53 million | 317 million (2^28.2) | 1.6 × 10^11 |
| The clamp's top | – | 2^31 = 2.1 billion | 1.1 × 10^12 |

The coupling's own budget for the lab is at least `4 × (256 × 4,608 + 768² + …)` = 7.1 million plus `4 × colliders + 132 × hash size`.

**What a sound frame takes (Inferred from the kernels' structure).** A lane handles one or two nodes, a few beams and one or two triangles a substep. Against a full coupling room that is on the order of 10^4 passes a substep, 3 × 10^5 a frame. The largest legitimate count recorded is about 6 million in one invocation (the ETK's serial self-collision fallback; `engine/docs/handoff/soft-rigid-world.md` §4). The budget is 30 to 350 times that. It was sized to never stop a sound frame, not to bound time.

**In seconds (Inferred; the time per pass has never been measured).** Assume 1–10 ns per pass on one lane:

| Case | Passes | GPU time |
|---|---|---|
| One lane runs to the lab's budget | 1.7–3.2 × 10^8 | 0.2–3 s |
| One lane runs to 2^31 | 2.1 × 10^9 | 2–21 s |
| All 512 lanes run to the lab's budget | 0.9–1.6 × 10^11 | Between the single-lane time (if the lanes run fully in parallel) and 512 times it: seconds to tens of minutes |
| The same, eight frames in flight | × 8 | × 8 |

So a frame that reaches its budget is not "failed in milliseconds". The budget turns an endless hang into a long one. A frame at the budget, eight deep, is enough on its own to hold the GPU past any watchdog.

**Legitimate worst cases that are far above the measured ones (Read, then computed).**

- **Node contact against proxies:** every node tests every proxy shape. At the lab's room, 554 × 256 × 33 = 4.7 million node–shape tests a frame; the mountain measured 3–41 shapes (Measured, CPU), so 0.05–0.75 million.
- **Self-collision, every-candidate path:** a triangle whose padded box covers more than 64 grid cells tests every candidate instead of a few buckets (`explicit_beams.slang:3218-3223`, the CPU's `CollisionGrid::gather`). At worst that is `triangles × candidates` a substep: 608 × 306 × 33 = 6.1 million tests a frame for the authored sedan.
- **The pair list:** more than 8,192 listed pairs for a vehicle in a step runs the grid walk again with its tests (`explicit_beams.slang:5430-5445`).
- **Contact groups:** touching vehicles merge into one group on one workgroup; two ETKs cost 21.3 ms a step (known-bugs C2), and a lane walks every member (`explicit_beams_rigid.slang:768-829`).
- **Surface contact:** each triangle, segment and node against each proxy (`explicit_beams_surface.slang:916-983`), so the cost scales with the proxy count like node contact does.

None of these has a time limit. They are bounded by capacities, which fail loudly when exceeded, but a frame at the capacities' edge can cost tens of times a typical one.

### (c) State-dependent blow-ups

**The mountain degrades the car's state within a minute (Measured, CPU).** `examples/vehicle_lab.rs --measure mountain --surface 1 --secs 90`, the authored sedan, defaults (seed, 50°, rockslide on), with a diagnostic added for this investigation:

| Time | Car | Beams broken | Node box (largest extent) | Proxy shapes |
|---|---|---|---|---|
| 0–9 s | Falling, tumbling | 0 → 333 | 3.6–4.7 m | 3–13 |
| 12 s | | 365 | 19 m | 9 |
| 21 s | Nearly at rest | 442 | 37 m | 36 |
| 33 s | | 578 | 49 m | 41 (the peak) |
| 39 s | At rest at y = −165 m | 607 | 63 m | 14 |
| 45 s | At rest; rockslide boulders arriving | 716 | 151 m | 15 |
| 51 s | A boulder strike (888 → 967 broken) | 967 | 260 m | 16 |
| 54.5 s | At rest | 970 | 323 m | 15 |
| **54.6 s** | **Non-finite node state: the run stops** | | | |

- **Parts fly off and keep going.** From 39 s the box grows by a steady 18 m/s while the car is at rest: a detached part falling at its terminal speed toward the car's own ground plane, which the mountain puts at y = −100 km (`vehicle_lab/mod.rs:1698`). Nothing removes it or stops it.
- **The state becomes NaN.** The CPU lab stops with an error (`vehicle_lab/cpu.rs:305-309`), and so does the offline GPU example (`examples/vehicle_lab.rs:128-132`). **The page does not** (Read): `CarStats::read` sets a flag and logs one line (`vehicle_lab/monitor.rs:440-448`), the HUD shows "NON-FINITE (NaN)", and `beforeStep` and `step` go on every frame.
- **The kernels have no NaN test** (Read: no `isnan`, `isinf` or `isfinite` in `explicit_beams*.slang` or `vehicle_coupling.slang`). The explicit solver's own status reports loop caps and vehicle-contact bounds, not a non-finite state.
- **The rockslide is what does it (Measured, CPU).** With the rockslide off, the same run stays finite for 90 s with the box steady at 35 m (appendix). The rockslide is on by default and throws a 3.4 t boulder at a car at rest every 4 s, for ever.
- This was the authored sedan. The generated sedan the user ran is a local export and was not run here.

**What a degraded state does to the next frame (Read for the paths, Inferred for the cost).**

- **The proxy query takes the car's box from all of its nodes** (`cpl_bounds`, `vehicle_coupling.slang:233-243`), grown by the fastest node's speed times the window (`sweptBox`, `:277-287`). A part 300 m away makes the box 300 m long. Every collider whose bounds touch that box becomes a proxy, up to the room (then the step fails by name). On the CPU run the count stayed at 41 or less because the flung part fell away from the rocks; a part that stays on the slope, or an exploded state whose fastest node has an enormous speed, takes in every live rock (about 180 bodies at the defaults, 975 at the clamps).
- **An exploded or NaN state sends the bounded paths to their bounds:** the every-candidate self-collision path, every node against every proxy. The frame then costs what section (b) lists as the worst cases rather than what was measured. How much on this GPU is not known; from the pass counts, tens to hundreds of times a typical frame.
- **`cpl_query` counts cells in integers** (Read, `vehicle_coupling.slang:427-432` and `:455-457`): `uint cells = uint(span.x) * uint(span.y) * uint(span.z)`. For a very large or non-finite box this product can wrap. A wrapped count below the table size walks only that many cells and silently misses colliders; it does not hang (the loop is bounded by the table size either way). The explicit kernels' grid walks count in floating point for exactly this reason (`explicit_beams.slang:3218-3223`). This is a correctness bug in the coupling's query for boxes past about 1,600 cells a side.
- **Garbage headers:** the coupling's record counts are held to the region's room (`26e3b700`), and the loops that read them are capped.
- **Capacity overflows fail by name** (Read): shapes, bodies, planes, impulse records, chunk records and the loop flag are bits of the coupling status (`explicit_vehicles.rs:68-75`), read with every step. They fail the step whose readback carries them, which is up to 8 steps after the frame that set them.

**Giant coordinates by design (Read).** The mountain moves the arena's ground and fixtures 200 km down, the car's own static shapes 10 km up, the solver's ground plane 100 km down, and a parked second car 10 km away; destroyed bodies wait at y = −10 km. Contact is computed relative to the car's origin, so these are offsets rather than jitter. They do mean a part that leaves the slope falls for 100 km.

### (d) Streaming and rebuild churn

**The mountain's streaming is bounded (Read).**

- A chunk is 8–23 static bodies (2–3 slabs, 2 walls, 4–18 rocks by roughness) plus up to 16 loose rocks (`vehicle_lab/mountain.rs:270-378`). The window is 2 behind and 8 ahead: 11 chunks, about 180 bodies at the defaults; 25 chunks and about 975 bodies at the clamps, against a World capacity of 4,096 bodies and 8,192 colliders (`mod.rs:1765-1779`).
- Chunks outside the window are all removed in the same call; at most one is made per step once running (`mod.rs:1201-1242`).
- The 12 spike hulls are shared; the hull arena dedups them (`engine-schema/src/convex.rs:673-691`). Body and collider uploads are ranges, not whole arrays. No pipeline or buffer is made per chunk.
- Exceeding capacity is an error, not a silent skip.

**Smaller findings (Read).**

- A chunk whose spawn fails part way leaks the bodies already made until the next restart (`mod.rs:1236`).
- Each hull insert marks the broadphase's resting table stale, so the step after a new chunk plans the coupling by scanning every collider instead of querying the table (`engine-webgpu/src/gpu.rs:1485-1551`, `explicit_vehicles.rs:716`). Bounded by the collider count.
- The rockslide throws one 3.4 t boulder, as a bullet, at most every 4 s while the car is at rest, and removes it after 8 s (`mod.rs:1248-1281`). Two or three live at most. Its aim uses the car's centre without checking it is finite.
- Restart re-streams every chunk in one call: a burst of up to about 975 bodies, once.

Streaming is not a candidate for the crash.

**Rebuilds.** Every "build" option of the lab (car, surface contact, second car, drag, damping, sleep, GPU timing, the coupling's room) makes a new device and compiles every pipeline again, 21 s measured for a surface-contact toggle (`engine/docs/vehicle-lab.md`). Section 3h.

### (e) Device loss and rebuild loops

- **The lab (Read, `vehicle-lab.html:371-379`):** on loss it drops its reference to the World, shows the message and a Rebuild button. It does not rebuild by itself, so there is no loop. It does not free the old World's wasm objects, and the button is offered immediately, while the GPU may still be busy with the queue that caused the loss. A rebuild then queues a full pipeline compile behind it.
- **Other pages:** `device.lost.then(fail)`, which stops their loop.
- **Mid-load loss panics the wasm (Observed, 02:50):** creating a mapped buffer on a lost device is an error that wgpu's web back end unwraps, so the module ends in `unreachable` instead of an error the page can show.
- **`mapAsync` on a lost device rejects;** the lab swallows that for rebuilt worlds (`vehicle-lab.html:434`).
- **Native:** not audited. wgpu reports loss through a callback; `World` has no path that stops stepping on it.

### (f) Several GPU users at once

- `~/.rubble/gpu.lock` (`hold_gpu_lock`, `engine-webgpu/src/gpu.rs:656`) is taken by native processes that create a device, and by anything wrapped in `engine/scripts/with-gpu-lock.py`.
- **A browser page takes no lock.** The user's page and an agent's headless Chrome or native test can share the GPU, each seeing the other's work as its own slowness. The lab's 02:35 run and this night's agents are an example of the exposure, though no agent GPU job is known to have been running at that minute.
- The lock serialises jobs. It does not bound any of them.
- CPU load matters too: the 02:50 compile stall happened at a load average of 12–18.

### (g) Native specifics

- **No browser watchdog.** Nothing kills a native process whose GPU work runs long. On macOS the first protection to act is the window server's own.
- **wgpu native:** as (a): Block mode hangs the calling thread on the GPU; Drop mode does not gate.
- **Metal back end (Read):** `engine-metal` keeps rings of command buffers (`RING = 4` in `sim.rs:38`, `PARAM_RING = 3` in `explicit.rs:42`) and waits for the oldest when the ring is full (`sim.rs:494`, `explicit.rs:208`). That is back-pressure by completion with 3–4 in flight, by blocking the caller.
- **CUDA back end:** not audited here. A CUDA kernel that runs long also cannot be interrupted; destroying the context ends it.
- **A host application (Unreal):** the engine runs on the game's device or beside it. A long Rubble dispatch stalls the game's own rendering on the same GPU; a blocked `step` stalls the game thread; on Windows a dispatch past the OS timeout removes the device for the whole game. The host has no way today to tell Rubble how much GPU time it may use.
- **Pipeline compiles** happen at startup with no cache (3h).

### (h) Pipeline compile stalls (not overload, same symptom)

**The kernels are very large (Read; sizes from this build's generated WGSL).**

- `engine_exec::explicit_plan::MODULES` lists 32 frame modules: 16 combinations of lanes (256/512), aero, solo, contact groups (small/big), each with and without surface contact, plus the `_seg` variants and 7 small passes (`explicit_plan.rs:411-460`).
- Each frame module is one entry point holding the whole 33-substep frame with every call inlined:

| Module | WGSL size |
|---|---|
| `explicit_beams_contact_aero_wide` | 812 KB |
| `explicit_beams_contact`, `_contact_wide` | 741 KB |
| `explicit_beams_contact_big` | 683 KB |
| `explicit_beams_solo_aero` | 652 KB |
| `explicit_beams`, `_wide`, `_solo` | 595–597 KB |
| The same without surface contact | 367–595 KB |

- A scene compiles the variants it uses: one car needs the solo frame; two cars that can touch need the contact frame as well; surface contact selects the larger twins.
- The path is Slang → WGSL (at build time) → Tint → MSL → the Metal compiler (`MTLCompilerService`) at page load.

**What is known and what is inferred.**

- **Observed:** one module took 127.8 s; the load was at 78 s with 4 of 7 done; the device was lost during the compiles; the Metal compiler processes were still running after the loss; GPU utilisation was low.
- **Read:** the pages already compile with `createComputePipelineAsync` first (`warm_pipelines`, `engine-world-wasm/src/lib.rs:149-160`), all the specs at once. Its comment records that a synchronous compile of the contact frame, about a minute cold, kills the GPU process.
- **Inferred:** Chrome lost or killed its GPU process while the compiles ran. Whether by its watchdog (the async path should keep the main thread free, so this would mean the compile still blocks something the watchdog checks) or by memory (several Metal compiles of 0.6–0.8 MB sources at once) is **not known**. The async warm-up did not prevent it.
- **Inferred:** the browser caches compiled pipelines, so this is a first-load problem, and a load that ends in a lost device may leave nothing cached, so the next load is cold again.
- **Read:** a device lost mid-load ends in a wasm panic (3e).
- **Native equivalent:** the same compiles stall startup, with no cache unless we add one (a wgpu pipeline cache, or precompiled Metal libraries). The handoff notes Metal parity kernel compiles running for many minutes.

## 4. Root cause of the 2026-11-13 lab crash

> **User's observation (added 2026-11-13, after this analysis).** "It got really corrupted looking a moment before it froze and crashed." The car was visibly corrupted on screen just before the freeze. That is direct evidence for hypothesis 1 (the page kept stepping an invalid state) and against a pure compile stall. It does not show which kernel then ran long. Confidence in hypothesis 1 is raised from moderate to high.

Not confirmed. Ranked by the evidence:

**H1. The page kept stepping a degraded state (most likely; moderate confidence).**

- *Chain:* on the mountain the car breaks up, parts fly off, and within about a minute the state is non-finite. Frames computed from the exploding and then NaN state take the bounded-but-large paths of section 3c. The page does not stop on NaN, learns of any step error 8 steps late, keeps 8 steps in flight and asks for up to 7 more each frame. The GPU saturates; Chrome's GPU process stops responding and is killed (device lost, then no adapter); the queued steps keep the GPU at 100 % with no process; the window server starves and the session ends.
- *For:* the CPU reference reaches NaN at 54.6 s on the default mountain run, and stays finite for 90 s with the rockslide off (Measured): the default-on rockslide keeps striking a car at rest until it comes apart. "Works for a while and then crashes" matches a state that degrades over a minute. The page has no stop on NaN (Read). 100 % for 10 s after the page died needs about 10 s of queued work, which is 8 frames at roughly a second each: in the range section 3b gives for frames near their bounds.
- *Against:* the CPU run was the authored sedan, not the generated one. The GPU cost of a NaN-state frame was not measured; it rests on pass counts. The CPU run's NaN came with rockslide strikes on a car at rest, and the user's car may not have come to rest.

**H2. A legitimately heavy state, with no NaN (possible; lower confidence).** The same chain without the NaN: a flung part or a pile of rocks takes the proxy count toward the room, the step cost passes the displayed frame, and the eight-deep queue does the rest. *For:* the box grew to 323 m on the CPU. *Against:* the proxy count stayed at 41 or less there, and a step would need to cost about a second for the 10 s that was observed.

**H3. A pipeline compile stall (possible; lower confidence).** The user was changing options, and each build option recompiles every pipeline for about 21 s. The 02:50 failure shows a compile can lose the device with the same message. *For:* the same console message; the lab rebuilds on a new device each time. *Against:* a compile stall leaves the GPU nearly idle (27 % at 02:50), and the lab crash had the GPU at 100 % with no process. The report says the lab was running, not loading.

**H4. A kernel loop running to its budget from a corrupt count (possible; cannot be ranked).** The 2026-11-07 class of bug, in a path the mountain exercises that earlier scenes did not: the origin rebase with static proxies every 100 m, bullets against a car, streaming. *For:* it produces exactly these symptoms, and section 3b shows the budget does not bound time. *Against:* no such path was found by reading; the earlier rebase bug is fixed on this branch.

**H5. The pacing change alone (unlikely).** At the measured 6.5–8.9 ms a step the lab is not saturated (section 3a). The change is an amplifier of H1, H2 and H4, not a cause.

**H6. Streaming or the precision of far coordinates (unlikely).** Section 3d.

H1 and H3 can both be true of one night. Keep both open.

**What would confirm it: a staged experiment (described, not run; it needs the user's approval and a quiet machine).**

1. **CPU first.** Run the generated sedan on the CPU reference on the mountain (the offline example needs a loader for the local export). Record when the box grows and whether the state goes non-finite.
2. **Calibrate the time per pass.** The runaway test kernel (`explicit_loop_cap_test.slang`) natively, under the GPU lock and a 60 s kill timer, at budgets 2^12, 2^16 and 2^20 with one lane and with all lanes running. This gives nanoseconds per pass and how lanes share a core; nothing else in this document can replace it. Abort if a run at one budget takes more than 10 times the one before.
3. **Cost of a degraded frame, one frame at a time.** Take a state from step 1 just before the NaN (the CPU reference can save it), load it natively, and run a single frame with `RUBBLE_LOOP_BUDGET` at 2^16, then 2^20, then the real budget, waiting for each to finish and reading the status before the next. One frame in flight, never a loop. Watch `ioreg -r -d 1 -w 0 -c IOAccelerator | grep "Device Utilization"` in another terminal. Abort when a frame takes over 250 ms or utilisation stays at 100 % for 2 s after the process exits.
4. **The page, headless, never in the user's browser.** The lab on the mountain with a readback every step and a harness that stops stepping on the first NaN, the first error, or a step over 50 ms; 3 slots, no owed steps; 90 s at most. Record GPU time per step against the box and proxy counts.
5. **Only then** the page as it was (8 slots, owed steps), headless, with the same stops, to see whether the queue is what turns a slow frame into a loss.

Stop the whole experiment at the first sign of a wedge. Do not start a later stage until utilisation is back to idle.

## 5. What is already guarded, and what each guard does not cover

| Guard | What it does | What it does not do |
|---|---|---|
| Loop caps (2^20 a loop) and the per-frame pass budget (`explicit_loop_cap.slang`) | Ends every kernel loop; turns an endless loop into a failed frame | Bound time: the budget is 2^22–2^31 passes per invocation, times up to 512 lanes. Barrier loops are outside the budget |
| The loop census test (`engine-webgpu/tests/explicit_loop_cap.rs`) | Fails CI on an uncapped loop in the explicit kernels | Check the size of a budget, or kernels outside the explicit set |
| Loop-cap flag and coupling status, read with every step | `World::step`'s frame fails with the error | Arrive in time: the error comes with the readback, after every step already in flight (3, or 8). Report NaN |
| The explicit solver's status ring (every 8th frame) | Reports vehicle-contact bounds and the loop flag for hosts without the World | Run every frame |
| Capacity rooms (coupling shapes, bodies, planes; World capacities) | Fail by name when exceeded | Limit the cost of a frame within the room |
| Float cell counts in the explicit grid walks | A stretched or NaN triangle takes a bounded path | Exist in `cpl_query`, which counts in integers |
| The readback gate (`canStep`) | Bounds the steps in flight by count | Know what a step costs; apply to native `step` in Drop mode |
| `FramePacer`'s GPU budget | Slows the simulation to what the GPU keeps up with | Apply without a frame-rate floor (the default on every vehicle page) |
| `warm_pipelines` | Compiles off the GPU process's main thread | Prevent the 02:50 loss; limit how many compile at once |
| `~/.rubble/gpu.lock` | One native or wrapped GPU job at a time | Cover a browser page; bound a job |
| Kill timers | End a run that keeps submitting | Stop work already queued |
| The lab's device-loss handler | No rebuild loop | Free the World; wait for the GPU before offering a rebuild |
| The CPU lab's and the offline example's NaN stop | End the run on a non-finite state | Exist on the page |

## 6. Options (not implemented; for a later decision)

"Lossless" means the physics a step computes is unchanged. Slowing down, pausing and failing are not lossy in the sense of `AGENTS.md`: no step computes anything different, and limits fail loudly.

| # | Option | Protects against | Cost | Lossless | Web / native |
|---|---|---|---|---|---|
| 1 | **Stop on invalid state.** Read each vehicle's bounds record (it exists: `cpl_bounds` writes it) with the step; a non-finite box or speed fails the step. The lab's page stops on `nonFinite` | 3c: stepping a NaN state for ever | 28 bytes a vehicle a step; a day | Yes | Both |
| 2 | **Back-pressure on completed work, two in flight at most.** A step is submitted only when the one before the last has been seen to complete (`onSubmittedWorkDone`; natively a fence or the command buffer's completion). No owed steps | 3a: the queue's depth in time becomes two steps | Small; the simulation runs slower when the GPU is slower, and says so | Yes | Both; for native it should not block the caller |
| 3 | **Slow down, never catch up.** The pacer drops what it cannot take, shows "running at X % of real time" | 3a | Trivial with 2 | Yes | Both |
| 4 | **A wall-time budget per step with automatic pause.** Two steps in a row over a budget (50 ms was the proposal), or a step that does not complete, pauses stepping with a message; the user resumes | 3a, 3b, 3c: whatever makes steps slow, the host stops feeding it | Small; needs a rule for the first frames after a build and for hidden tabs | Yes | Both |
| 5 | **A pass ceiling tied to measured throughput.** An explicit engine-wide setting for the most passes an invocation may take in a frame, set from a measured time per pass so that a frame at the ceiling is, say, 30 ms; a frame that needs more fails by name | 3b: the single long dispatch | Needs the calibration of section 4 step 2. A legitimate serial fallback of 6 million passes may not fit a 30 ms ceiling: then that path must be made parallel first | Yes (it fails; it never drops work) | Both |
| 6 | **A budget shared by the workgroup,** not per lane | 3b: the 512× factor | A kernel change (an atomic counter changes Metal's code generation; needs the Metal parity run) | Yes | Both |
| 7 | **Split long dispatches.** A frame as several dispatches (the plan already replays commands; the 33 substeps could be 3–4 dispatches), so the GPU has points where the compositor can run and the host can stop | 3b, and the desktop freezing during a legitimate heavy frame | Per-dispatch overhead (measured before as a cost; needs an A/B); barrier state must live in buffers between dispatches | Yes | Both |
| 8 | **Validation before submit** of what the host already knows: counts against rooms, finite inputs, header sanity | 3c: garbage in | Small | Yes | Both |
| 9 | **Count cells in floating point in `cpl_query`,** as the explicit kernels do | 3c: silently missed colliders for huge boxes | One kernel edit and its parity runs | Fixes a bug | Both |
| 10 | **Remove or settle flung parts** (a kill bound for vehicle nodes, or the mountain's ground plane nearer) | 3c: the growing box | A design decision: it changes what happens to debris | A limit; must be explicit | Both |
| 11 | **A watchdog thread for native** that sees a step outstanding too long and stops submitting, tells the host, and never blocks the game thread | 3g | Moderate | Yes | Native |
| 12 | **An engine-level GPU budget API for hosts:** "this much GPU time per frame", steps in flight, a callback on overload | 3g: Unreal and other hosts | Design work on top of 2, 4 and 5 | Yes | Native (and the wasm host) |
| 13 | **Device loss:** stop submitting, free everything, never panic (no unwrap on a lost device), no rebuild until the GPU is idle or the user insists | 3e | Small in the pages; the wgpu unwrap needs a guard before each creation | Yes | Both |
| 14 | **Compile less at once:** only the variants a scene needs, one at a time with yields, progress and a cancel button | 3h | Small to moderate | Yes | Web |
| 15 | **Smaller kernels:** split the frame into smaller modules or entry points so each compiles in seconds | 3h, and helps 7 | Large; touches the kernels' structure and Metal's code generation | Yes if bit-identical; must be shown | Both |
| 16 | **A persistent pipeline cache** (native), or precompiled Metal libraries | 3h on native | Moderate | Yes | Native |
| 17 | **A lock that covers browser pages,** or a rule (section 7) | 3f | A rule is free; a real lock for pages needs a helper | – | Both |
| 18 | **Tests:** a simulated slow GPU for the pacer (in-flight never above two, steps per frame capped, pause on budget); a tiny-budget run of every new kernel; a soak test of every page headless for minutes with the stops of section 4; a CPU soak of each scene that fails on NaN | Regressions of all of the above | Moderate | – | Both |

**Recommended order.**

1. **1, 13 and the page's NaN stop.** Cheap, lossless, and they remove the most likely cause.
2. **2 and 3.** Two in flight, no catch-up. This bounds the queue in time whatever a step costs.
3. **4.** The pause on a wall-time budget. With 2 it means the engine stops feeding a GPU that is falling behind, from any cause, including ones not yet understood.
4. **18,** the tests for 1–4, and the headless soak of every vehicle page before the user opens it.
5. **The calibration** (section 4 step 2), then **5**: a pass ceiling with a real time behind it.
6. **14** now, then **16** for native: the compile stall is a separate failure that 1–5 do not touch.
7. **6, 7, 15:** the kernel work that makes a single dispatch short. This is the only thing that protects the desktop from a legitimately heavy frame, and it is the most expensive.
8. **9, 10, 11, 12** as their owners need them (9 is a bug and can go with any kernel work).

**The prototype.** Commit `dd586715` on `fix/gpu-overload-guards` sketches options 1, 2, 4 and a first cut of 5 (a ceiling of 2^26 with no calibration behind it): `engine-world-wasm/src/step_gate.rs` with 8 CPU tests against a simulated slow GPU, the ceiling in `explicit_plan.rs`, and the bounds check in `explicit_vehicles.rs`. It compiles natively and its unit tests pass. It was never built for wasm or run on a GPU, no page uses it, and it is not adopted.

## 7. Operating rules until this is solved

For people and agents.

**Do not run:**

- **`vehicle-lab.html` in Mountain mode** in a browser you care about. If it must be looked at, turn the rockslide off, set readbacks to every step, watch the HUD's state row, and close the tab when the car comes apart.
- **Any vehicle page for minutes unattended.** A car that has broken up is the risk.
- **Two-car scenes with surface contact on a cold cache** while the machine is busy (the 02:50 failure). Let the first load finish with nothing else running.
- **Anything on the GPU while the user has a GPU page open,** and no GPU page in the user's browser while agents use the GPU.

**Testing a new GPU page or kernel:**

1. The CPU reference first, for as long as the page is meant to run. A NaN or an error there is a stop.
2. A tiny native run with a deliberately low `RUBBLE_LOOP_BUDGET`.
3. Headless Chrome through `engine/scripts/with-gpu-lock.py` with a kill timer, with a harness that stops on the first error, NaN or slow step.
4. Only then the user's browser.

Follow `.cursor/skills/gpu-safe-runs/SKILL.md` for every run.

**Signs of a wedge:**

- `ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*'` stays at 100 with no GPU process running.
- A page reports "device lost", then "no WebGPU adapter".
- Tests that used to pass stall; the pointer or windows stutter.

**What to do:**

- Stop submitting: close the page, end the run. Do not press Rebuild, and do not start another GPU job "to check": it queues behind the wedge.
- Watch utilisation. If it falls to idle within a minute, the queue drained; wait for it before anything else.
- If it stays at 100 %, save work and reboot. Tell the user rather than retrying.
- Write down what was running (page, mode, car, options, how long) in this document's incident table.

## Appendix: the CPU measurements

Diagnostic added for this investigation: `ExplicitBeams::self_collision_load` (read-only) and the offline example's progress line (`engine/crates/engine-world/examples/vehicle_lab.rs`), on the prototype branch.

```bash
cd engine
cargo run -j 4 --release -p engine-world --example vehicle_lab -- --measure mountain --surface 1 --secs 90 --every 30
```

- **Rockslide on (defaults):** the table in section 3c. Non-finite at 54.6 s. No triangle took the every-candidate path at any sample (0 of 566–608): a broken triangle is removed rather than stretched, so that path is not what grows on the mountain.
- **Rockslide off:** the same run from a script with `"mountain": { "rockslide": false }`, 90 s. The car came to rest after about 25 s at y = −128 m and stayed there: 452 beams broken, the node box 35 m and steady, 37 proxy shapes at the peak, **no non-finite state**. So on the CPU reference the growth of the box past 35 m and the NaN both come with the rockslide's boulders striking the car at rest (one 3.4 t bullet at 14 m/s every 4 s, for as long as the car sits still).
