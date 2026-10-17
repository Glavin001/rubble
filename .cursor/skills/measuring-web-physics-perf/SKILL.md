---
name: measuring-web-physics-perf
description: Measure the WASM/WebGPU demo and the engine fracture lab (engine/crates/engine-webgpu/fracture) without being fooled by tab throttling, display refresh rate, dev-server staleness or diverged simulation states. Use when benchmarking the web demos or the fracture lab, A/B-testing a shader or host change, or when reported step times or sim speed look wrong (too fast, too slow, hanging).
---

# Measuring Web Physics Performance

Browser measurements lie in specific, repeatable ways. Every wrong conclusion this project drew about web perf came from one of the pitfalls below, not from the engine.

## Setup

```bash
cd web && npm run dev   # predev rebuilds the WASM; stale WASM is a silent source of "no change"
```

Open `http://localhost:5173/src/3d/index.html`. The overlay shows per-phase step timings; treat it as the primary metric, not FPS.

## Pitfalls that invalidate measurements

1. **Background-tab throttling.** Browsers throttle `requestAnimationFrame` to ~1 Hz when the tab (or the IDE webview containing it) is hidden or minimized. Symptoms: demo "hangs" on its loading message, `stepsPerSec` collapses, timings spike. Always bring the tab to the foreground before measuring (`Page.bringToFront` via CDP, or ask the user to keep it visible). If the user minimizes the window mid-run, discard those samples.

2. **Display refresh rate sets sim speed, not step cost.** With a fixed `dt` per rAF, a 120 Hz ProMotion display runs the sim 2× real time — this looks like "physics too fast" but is a stepping-rate bug, not a solver bug. The demo uses a fixed-timestep accumulator (`FIXED_DT` / `simAccumulator` in `web/src/3d/main.ts`) precisely so sim speed is independent of refresh rate. If sim speed looks wrong, check that pattern first.

3. **Averaging over warm-up.** The first ~100 frames include pipeline compilation and pile settling. Sample steady-state windows.

4. **FPS as a proxy.** FPS saturates at the refresh rate; a 10× step-time improvement can show zero FPS change. Measure step milliseconds.

## Measuring via CDP

Use `Runtime.evaluate` against the page to read the demo's timing state directly instead of screenshotting the overlay repeatedly. Poll a steady-state window, report median and p95 step time and per-phase breakdown (Upload, Predict, Broadphase, Narrowphase, Contacts, Solve, Extract).

## Cross-checking

- Native comparison: `DISPLAY`-independent perf tests live in `crates/rubble3d/tests/perf_baseline_tests.rs`. Do not compare local Metal/proprietary-driver numbers against committed lavapipe CI baselines — different backends are not comparable.
- E2E sanity: `cd web && npm run test:e2e` (Playwright, SwiftShader) verifies behavior, not speed; SwiftShader timings are meaningless as benchmarks.

## Reporting

Always report: scene + body count, browser + adapter, tab-visible confirmation, steady-state window, median step ms with phase breakdown, and the comparison baseline. A number without these is not reproducible.

## Engine fracture lab (Slang kernels, JS host)

Prefer the headless runner to the lab page for physics numbers. It uses
WebGPU on the real GPU, has no renderer, prints JSON, and doesn't compete with
the lab or the user for the GPU:

```bash
nvm use
node engine/crates/engine-webgpu/fracture/headless.mjs grid warm=420 measure=120
```

See the fracture README, "Headless physics benchmark", for options: scenes,
`budget`, `bonds`, `columns`/`rows`, `trace`, `CHROME_ARGS`.

Pitfalls specific to this engine:

1. **Compare the same state.** A shader or host change that alters
   floating-point rounding even slightly sends a chaotic collapse down a
   different path within a few hundred ticks. Timing two runs at "tick 420" can
   then time two different scenes; one claimed "9% faster" was an artefact.
   - Run A/B timings with `budget='{"deterministic":true}'`.
   - Check that the counters (candidates, contacts, awake) match at the
     measured tick before trusting the timing difference.
   - If they differ, compare in default mode over several runs.
2. **`profileKernels` puts every dispatch in its own pass.** Use it to find
   which kernels matter, not as a total. Timestamps are quantised to about
   65 µs. For totals use ms/tick with ticks queued back to back
   (`step(1,{wait:false})` then `flush()`), as the headless runner does.
3. **The browser pane throttles in the background.** When the app window
   loses focus, `requestAnimationFrame` drops to 1-4 per second and the lab
   takes almost no physics steps. Count rAF calls inside every sample and
   discard throttled ones. Never conclude anything from a run where the step
   count did not advance.
4. **Lab FPS and TPS are different things.** The lab slices heavy steps across
   frames, so FPS can stay at 60+ while physics runs at a fraction of real
   time. Report both, together with the awake-body count.
5. **The demo page holds its own simulation and scene.** Page-level calls such
   as `D.load(...)` or a user click change `D.scene` under a running test.
   Load the scene inside the test and check `scene.rowShot` before using it.
