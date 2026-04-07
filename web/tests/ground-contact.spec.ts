import { test, expect } from "@playwright/test";

/**
 * Ground-contact acceptance test.
 *
 * Loads the "Ground" scene (a static floor + one dynamic box at y=4) and
 * asserts the dynamic body falls and makes contact with the ground.
 * This catches regressions where the simulation runs without errors but
 * objects never actually move or collide.
 */

test.describe("Ground contact acceptance", () => {
  test.beforeEach(async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (err) => errors.push(err.message));
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });

    await page.goto("/src/3d/index.html");
    // Wait for WASM + world init; check for error so we fail fast with a
    // useful message instead of timing out silently.
    await page.waitForFunction(
      () => {
        const t = window.__rubble_test;
        if (t?.error) throw new Error(`init failed: ${t.error}`);
        return t?.ready === true;
      },
      null,
      { timeout: 30_000 },
    );

    (page as any).__errors = errors;
  });

  test("dynamic body contacts ground within 5s sim time", async ({ page }) => {
    // Wait for the initial scene (Pyramid) to finish its first step so the
    // loading overlay hides and scene-select is interactive.
    await page.waitForFunction(
      () => document.getElementById("loading")?.style.display === "none",
      null,
      { timeout: 30_000 },
    );

    // Switch to the Ground scene (1 static floor + 1 dynamic box).
    // The Ground scene has exactly 2 bodies.
    await page.selectOption("#scene-select", "Ground");

    // Wait for the Ground scene to fully load:
    // - bodyCount drops to exactly 2 (from Pyramid's ~137)
    // - loading overlay hidden (first GPU step completed)
    await page.waitForFunction(
      () => {
        const t = window.__rubble_test;
        return (
          (t?.bodyCount ?? 0) === 2 &&
          document.getElementById("loading")?.style.display === "none"
        );
      },
      null,
      { timeout: 30_000 },
    );

    // Read initial transforms. Ground scene: 2 bodies, 7 floats each.
    // Body 0 = static floor (y≈0), Body 1 = dynamic box (y≈4).
    const initial = await page.evaluate(() => {
      const transforms = window.__rubble_test!.getTransforms!();
      const n = transforms.length / 7;
      const ys: number[] = [];
      for (let i = 0; i < n; i++) ys.push(transforms[i * 7 + 1]);
      return { ys, count: n, stepCount: window.__rubble_test!.stepCount };
    });

    expect(initial.count).toBe(2);

    // Find the dynamic body (highest initial y)
    let maxY = -Infinity;
    let dynamicIdx = 0;
    for (let i = 0; i < initial.count; i++) {
      if (initial.ys[i] > maxY) {
        maxY = initial.ys[i];
        dynamicIdx = i;
      }
    }

    // The dynamic box should start well above the floor
    expect(maxY).toBeGreaterThan(2.0);

    // Poll until the body falls and contacts the floor.
    // Floor top surface at y=0.5, box half-height=0.5 → resting y ≈ 1.0.
    // Accept y < 2.0 as "contacted ground" (started at ~4).
    const pollStart = Date.now();
    const deadline = pollStart + 30_000;
    let lastY = maxY;
    let lastStepCount = initial.stepCount;
    while (Date.now() < deadline) {
      const state = await page.evaluate((bodyIdx) => {
        const t = window.__rubble_test;
        const transforms = t?.getTransforms?.();
        const y =
          transforms && transforms.length >= (bodyIdx + 1) * 7
            ? transforms[bodyIdx * 7 + 1]
            : null;
        return {
          y,
          stepCount: t?.stepCount ?? 0,
          error: t?.error ?? null,
        };
      }, dynamicIdx);

      lastY = state.y ?? lastY;
      lastStepCount = state.stepCount;

      if (state.error) {
        throw new Error(`Simulation error: ${state.error}`);
      }
      if (state.y !== null && state.y < 2.0) {
        break;
      }
      await page.waitForTimeout(200);
    }

    // Diagnostic info on failure
    expect(
      lastY,
      `Body did not fall to ground. lastY=${lastY}, startY=${maxY}, ` +
        `steps=${lastStepCount}, elapsed=${Date.now() - pollStart}ms, ` +
        `bodyIdx=${dynamicIdx}, initialYs=${JSON.stringify(initial.ys)}`,
    ).toBeLessThan(2.0);

    // Body should be near the floor surface, not fallen through
    expect(lastY).toBeGreaterThan(-1.0);

    // No errors during simulation
    const errors = ((page as any).__errors as string[]).filter(
      (e) => !e.includes("DevTools") && !e.includes("favicon"),
    );
    expect(errors).toHaveLength(0);
  });
});
