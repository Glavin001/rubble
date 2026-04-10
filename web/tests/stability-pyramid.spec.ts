import { test, expect } from "@playwright/test";

/**
 * WASM physics stability test: Pyramid scene.
 *
 * Verifies that a 16-row pyramid of boxes settles stably:
 * - Most bodies stay within the original footprint (no explosion)
 * - Center of mass doesn't rise (no energy pumping)
 * - Bottom rows remain intact (top rows may shed a few boxes)
 *
 * Usage:
 *   cd web && npx playwright test tests/stability-pyramid.spec.ts --reporter=list
 */

const SETTLE_STEPS = 600; // 5 seconds at 120Hz

test.describe("WASM Pyramid Stability", () => {
  test.setTimeout(180_000);

  test("pyramid holds shape after settling", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (err) => errors.push(err.message));
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });

    await page.goto("/src/3d/index.html?webgl=1");

    await page.waitForFunction(
      () => {
        const t = window.__rubble_test;
        if (t?.error) throw new Error(`init failed: ${t.error}`);
        return t?.ready === true;
      },
      null,
      { timeout: 60_000 },
    );

    // Default scene is Pyramid
    await page.waitForFunction(
      () => document.getElementById("loading")?.style.display === "none",
      null,
      { timeout: 60_000 },
    );

    const hasBenchStep = await page.evaluate(
      () => typeof window.__rubble_test?.benchStep === "function",
    );
    expect(hasBenchStep, "benchStep() not available").toBe(true);

    await page.evaluate(() => window.__rubble_test!.stopLoop!());

    // Record initial state
    const initialState = await page.evaluate(() => {
      const transforms = window.__rubble_test!.getTransforms!();
      const n = transforms.length / 7;
      const bodies: { x: number; y: number; z: number }[] = [];
      for (let i = 0; i < n; i++) {
        bodies.push({
          x: transforms[i * 7 + 0],
          y: transforms[i * 7 + 1],
          z: transforms[i * 7 + 2],
        });
      }
      return bodies;
    });

    const dynamicInitial = initialState.slice(1); // skip ground
    const initialMaxY = Math.max(...dynamicInitial.map((b) => b.y));
    const initialMaxAbsX = Math.max(...dynamicInitial.map((b) => Math.abs(b.x)));
    const initialComY =
      dynamicInitial.reduce((sum, b) => sum + b.y, 0) / dynamicInitial.length;

    // eslint-disable-next-line no-console
    console.log(
      `Pyramid: ${initialState.length} bodies, ` +
        `initialComY=${initialComY.toFixed(2)}, maxY=${initialMaxY.toFixed(2)}, ` +
        `maxAbsX=${initialMaxAbsX.toFixed(2)}`,
    );

    // --- Settle ---
    for (let i = 0; i < SETTLE_STEPS; i++) {
      await page.evaluate(() => window.__rubble_test!.benchStep!());
      if ((i + 1) % 200 === 0) {
        const snap = await page.evaluate(() => {
          const transforms = window.__rubble_test!.getTransforms!();
          const n = transforms.length / 7;
          const ys: number[] = [];
          const xs: number[] = [];
          for (let j = 1; j < n; j++) {
            ys.push(transforms[j * 7 + 1]);
            xs.push(Math.abs(transforms[j * 7 + 0]));
          }
          return {
            maxY: Math.max(...ys),
            minY: Math.min(...ys),
            maxAbsX: Math.max(...xs),
            comY: ys.reduce((a, b) => a + b, 0) / ys.length,
          };
        });
        // eslint-disable-next-line no-console
        console.log(
          `  step ${i + 1}: comY=${snap.comY.toFixed(2)}, maxY=${snap.maxY.toFixed(2)}, ` +
            `maxAbsX=${snap.maxAbsX.toFixed(2)}`,
        );
      }
    }

    // --- Read final state ---
    const finalState = await page.evaluate(() => {
      const transforms = window.__rubble_test!.getTransforms!();
      const n = transforms.length / 7;
      const bodies: { x: number; y: number; z: number }[] = [];
      for (let i = 0; i < n; i++) {
        bodies.push({
          x: transforms[i * 7 + 0],
          y: transforms[i * 7 + 1],
          z: transforms[i * 7 + 2],
        });
      }
      return bodies;
    });

    const dynamicFinal = finalState.slice(1);
    const finalYs = dynamicFinal.map((b) => b.y);
    const finalXs = dynamicFinal.map((b) => b.x);
    const finalComY = finalYs.reduce((a, b) => a + b, 0) / finalYs.length;
    const finalMaxY = Math.max(...finalYs);
    const finalMinY = Math.min(...finalYs);
    const finalMaxAbsX = Math.max(...dynamicFinal.map((b) => Math.abs(b.x)));

    // eslint-disable-next-line no-console
    console.log(
      `Final: comY=${finalComY.toFixed(2)}, maxY=${finalMaxY.toFixed(2)}, ` +
        `minY=${finalMinY.toFixed(2)}, maxAbsX=${finalMaxAbsX.toFixed(2)}`,
    );

    // --- Assertions ---

    // 1. No body should fall through the ground
    expect(
      finalMinY,
      `Body fell through ground: minY=${finalMinY.toFixed(2)}`,
    ).toBeGreaterThan(-1.0);

    // 2. No body should fly far above the initial pyramid height
    // Allow some margin for top-row boxes bouncing, but not explosion
    expect(
      finalMaxY,
      `Body flew to y=${finalMaxY.toFixed(2)} (initial max was ${initialMaxY.toFixed(2)}). Explosion.`,
    ).toBeLessThan(initialMaxY + 5.0);

    // 3. Horizontal footprint should not expand dramatically
    // Pyramid base spans ~[-8, 8]. Allow some spread but not explosion.
    const MAX_FOOTPRINT = initialMaxAbsX + 5.0;
    expect(
      finalMaxAbsX,
      `Horizontal spread=${finalMaxAbsX.toFixed(2)} exceeds ${MAX_FOOTPRINT.toFixed(2)}. Bodies flew sideways.`,
    ).toBeLessThan(MAX_FOOTPRINT);

    // 4. Center of mass Y should not have risen (energy conservation)
    // It can drop (boxes settling/compacting) but shouldn't rise much
    const comRise = finalComY - initialComY;
    expect(
      comRise,
      `Center of mass rose by ${comRise.toFixed(2)} ` +
        `(initial=${initialComY.toFixed(2)}, final=${finalComY.toFixed(2)}). Energy pumping.`,
    ).toBeLessThan(1.0);

    // 5. Most bodies should still be stacked (above y=0.5, the rest surface)
    // In a stable pyramid, all dynamic bodies rest above ground
    const bodiesOnGround = finalYs.filter((y) => y >= 0.3).length;
    const requiredOnGround = Math.floor(dynamicFinal.length * 0.9);
    expect(
      bodiesOnGround,
      `Only ${bodiesOnGround}/${dynamicFinal.length} bodies above ground. ` +
        `Need ${requiredOnGround}. Pyramid collapsed.`,
    ).toBeGreaterThanOrEqual(requiredOnGround);

    // 6. Bottom row should be intact: bodies in the y=0.3..1.5 band
    // Pyramid SIZE=16: bottom row has 16 bodies at y≈0.5
    const bottomRowBodies = finalYs.filter(
      (y) => y >= 0.3 && y <= 1.5,
    ).length;
    expect(
      bottomRowBodies,
      `Only ${bottomRowBodies} bodies in bottom row (expected ~16). Base disintegrated.`,
    ).toBeGreaterThanOrEqual(14);

    const realErrors = errors.filter(
      (e) => !e.includes("DevTools") && !e.includes("favicon"),
    );
    expect(
      realErrors,
      `Errors during test:\n${realErrors.join("\n")}`,
    ).toHaveLength(0);
  });
});
