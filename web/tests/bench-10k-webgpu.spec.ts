import { test, expect } from "@playwright/test";

/**
 * WebGPU benchmark for the 10k Grid scene.
 *
 * Uses benchStep() to run physics steps directly, bypassing Three.js rendering.
 * This gives clean physics-only timings without animation frame throttling.
 *
 * Usage:
 *   cd web && npx playwright test tests/bench-10k-webgpu.spec.ts
 *
 * The test outputs JSON timing stats to stdout via console.log so you can
 * capture it with:
 *   npx playwright test tests/bench-10k-webgpu.spec.ts --reporter=list 2>&1 | grep '^{'
 */

const WARMUP_STEPS = 60;
const BENCH_STEPS = 200;

/** Timing phase indices in the 7-float lastStepTimingsMs array */
const PHASE = {
  upload: 0,
  predict_aabb: 1,
  broadphase: 2,
  narrowphase: 3,
  contact_fetch: 4,
  solve: 5,
  extract: 6,
} as const;

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  if (n === 0) return 0;
  if (n % 2 === 0) return (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  return sorted[Math.floor(n / 2)];
}

function p95(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.ceil(sorted.length * 0.95) - 1;
  return sorted[Math.max(0, idx)];
}

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function stddev(values: number[]): number {
  const m = mean(values);
  const variance = values.reduce((sum, v) => sum + (v - m) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function min(values: number[]): number {
  return Math.min(...values);
}

function max(values: number[]): number {
  return Math.max(...values);
}

test.describe("WebGPU 10k Grid Benchmark", () => {
  // Generous timeout — 10k bodies on WebGPU can be slow on CI
  test.setTimeout(600_000);

  test("benchmark 10k grid scene", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (err) => errors.push(err.message));
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });

    // Navigate to the 3D demo
    await page.goto("/src/3d/index.html");

    // Wait for WASM + world init
    await page.waitForFunction(
      () => {
        const t = window.__rubble_test;
        if (t?.error) throw new Error(`init failed: ${t.error}`);
        return t?.ready === true;
      },
      null,
      { timeout: 60_000 },
    );

    // Wait for default scene to load so UI is interactive
    await page.waitForFunction(
      () => document.getElementById("loading")?.style.display === "none",
      null,
      { timeout: 60_000 },
    );

    // Switch to 10k Grid scene
    await page.selectOption("#scene-select", "10k Grid");

    // Wait for 10k Grid to fully load (expect ~10000 bodies)
    await page.waitForFunction(
      () => {
        const t = window.__rubble_test;
        return (
          (t?.bodyCount ?? 0) >= 10000 &&
          document.getElementById("loading")?.style.display === "none"
        );
      },
      null,
      { timeout: 120_000 },
    );

    const bodyCount = await page.evaluate(
      () => window.__rubble_test?.bodyCount ?? 0,
    );
    // eslint-disable-next-line no-console
    console.log(`Loaded ${bodyCount} bodies`);

    // Verify benchStep is available
    const hasBenchStep = await page.evaluate(
      () => typeof window.__rubble_test?.benchStep === "function",
    );
    expect(hasBenchStep, "benchStep() not available on test hooks").toBe(true);

    // Stop the render loop so benchStep can run exclusively
    await page.evaluate(() => window.__rubble_test!.stopLoop!());

    // --- Warmup using benchStep (no rendering) ---
    for (let i = 0; i < WARMUP_STEPS; i++) {
      await page.evaluate(() => window.__rubble_test!.benchStep!());
    }
    // eslint-disable-next-line no-console
    console.log(`Warmup complete (${WARMUP_STEPS} steps)`);

    // --- Benchmark using benchStep ---
    const timings: number[][] = [];

    for (let i = 0; i < BENCH_STEPS; i++) {
      const t = await page.evaluate(() => window.__rubble_test!.benchStep!());
      timings.push(t);

      if ((i + 1) % 20 === 0) {
        const stepTotal = t.reduce((a, b) => a + b, 0);
        // eslint-disable-next-line no-console
        console.log(
          `  step ${i + 1}: total=${stepTotal.toFixed(1)}ms ` +
            `solve=${t[PHASE.solve].toFixed(1)}ms ` +
            `narrow=${t[PHASE.narrowphase].toFixed(1)}ms ` +
            `broad=${t[PHASE.broadphase].toFixed(1)}ms`,
        );
      }
    }

    // --- Compute stats ---
    const byPhase = (idx: number) => timings.map((t) => t[idx]);
    const totalPerStep = timings.map((t) => t.reduce((a, b) => a + b, 0));

    function phaseStats(name: string, values: number[]) {
      return {
        [`${name}_median`]: median(values),
        [`${name}_mean`]: mean(values),
        [`${name}_stddev`]: stddev(values),
        [`${name}_p95`]: p95(values),
        [`${name}_min`]: min(values),
        [`${name}_max`]: max(values),
      };
    }

    const result = {
      bodies: bodyCount,
      warmup_steps: WARMUP_STEPS,
      bench_steps: BENCH_STEPS,
      runtime: "webgpu-wasm-browser",
      ...phaseStats("step_ms", totalPerStep),
      ...phaseStats("upload_ms", byPhase(PHASE.upload)),
      ...phaseStats("predict_ms", byPhase(PHASE.predict_aabb)),
      ...phaseStats("broadphase_ms", byPhase(PHASE.broadphase)),
      ...phaseStats("narrowphase_ms", byPhase(PHASE.narrowphase)),
      ...phaseStats("contact_fetch_ms", byPhase(PHASE.contact_fetch)),
      ...phaseStats("solve_ms", byPhase(PHASE.solve)),
      ...phaseStats("extract_ms", byPhase(PHASE.extract)),
    };

    // Print JSON result (mirrors bench_10k.rs output format)
    // eslint-disable-next-line no-console
    console.log(JSON.stringify(result, null, 2));

    // --- Assertions (sanity checks, not perf gates) ---
    expect(bodyCount).toBeGreaterThanOrEqual(10000);
    expect(timings).toHaveLength(BENCH_STEPS);

    // All timings should be finite and non-negative
    for (let s = 0; s < timings.length; s++) {
      for (let p = 0; p < 7; p++) {
        expect(
          Number.isFinite(timings[s][p]),
          `timings[step=${s}][phase=${p}] = ${timings[s][p]} is not finite`,
        ).toBe(true);
        expect(timings[s][p]).toBeGreaterThanOrEqual(0);
      }
    }

    // No runtime errors
    const realErrors = errors.filter(
      (e) => !e.includes("DevTools") && !e.includes("favicon"),
    );
    expect(realErrors, `Errors during benchmark:\n${realErrors.join("\n")}`).toHaveLength(0);
  });
});
