---
name: measuring-web-physics-perf
description: Measure the WASM/WebGPU demo performance in a real browser without being fooled by tab throttling, display refresh rate, or dev-server staleness. Use when benchmarking the web demos (http://localhost:5173/src/3d/index.html), verifying a performance change, or when reported step times or sim speed look wrong (too fast, too slow, hanging).
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
