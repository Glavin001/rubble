import { test, expect } from "@playwright/test";

/**
 * WASM pyramid stability test using the LIVE animation loop (not benchStep).
 *
 * This matches what the user sees in the browser — physics + rendering together.
 * If benchStep tests pass but this fails, the issue is in the rendering/physics
 * interaction (e.g., GPU buffer contention between Three.js and physics).
 */

const SETTLE_SECONDS = 8; // wall-clock seconds to let it settle

test.describe("WASM Pyramid Stability (live loop)", () => {
  test.setTimeout(120_000);

  test("pyramid holds shape with live rendering", async ({ page }) => {
    const errors: string[] = [];
    const stabilityWarnings: string[] = [];
    page.on("pageerror", (err) => errors.push(err.message));
    const stepLogs: string[] = [];
    const allLogs: string[] = [];
    page.on("console", (msg) => {
      const text = msg.text();
      allLogs.push(`[${msg.type()}] ${text}`);
      if (msg.type() === "error") errors.push(text);
      if (text.includes("[STABILITY]")) stabilityWarnings.push(text);
      if (text.includes("[STEP]") || text.includes("[STABILITY]")) {
        stepLogs.push(text);
      }
    });

    await page.goto("/src/3d/index.html?webgl=1&norender=1");

    // Wait for init
    await page.waitForFunction(
      () => {
        const t = window.__rubble_test;
        if (t?.error) throw new Error(`init failed: ${t.error}`);
        return t?.ready === true;
      },
      null,
      { timeout: 60_000 },
    );

    // Wait for Pyramid (default scene) to start rendering
    await page.waitForFunction(
      () => document.getElementById("loading")?.style.display === "none",
      null,
      { timeout: 60_000 },
    );

    // Record initial body count
    const bodyCount = await page.evaluate(
      () => window.__rubble_test?.bodyCount ?? 0,
    );
    // eslint-disable-next-line no-console
    console.log(`Pyramid (live): ${bodyCount} bodies`);

    // Let the animation loop run for SETTLE_SECONDS
    await page.evaluate(
      (ms) => new Promise((r) => setTimeout(r, ms)),
      SETTLE_SECONDS * 1000,
    );

    // Log stability warnings
    if (stabilityWarnings.length > 0) {
      // eslint-disable-next-line no-console
      console.log(`Stability warnings (first 10):`);
      for (const w of stabilityWarnings.slice(0, 10)) {
        // eslint-disable-next-line no-console
        console.log(`  ${w}`);
      }
    }

    const stepCount = await page.evaluate(
      () => window.__rubble_test?.stepCount ?? 0,
    );
    // eslint-disable-next-line no-console
    console.log(`After ${SETTLE_SECONDS}s: ${stepCount} steps completed`);
    // eslint-disable-next-line no-console
    console.log(`All logs (last 30):\n${allLogs.slice(-30).join("\n")}`);
    // eslint-disable-next-line no-console
    console.log(`Step logs:\n${stepLogs.join("\n")}`);

    // Read final state
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

    const dynamicBodies = finalState.slice(1); // skip ground
    const ys = dynamicBodies.map((b) => b.y);
    const maxY = Math.max(...ys);
    const minY = Math.min(...ys);
    const maxAbsX = Math.max(...dynamicBodies.map((b) => Math.abs(b.x)));
    const comY = ys.reduce((a, b) => a + b, 0) / ys.length;

    // eslint-disable-next-line no-console
    console.log(
      `Final: comY=${comY.toFixed(2)}, maxY=${maxY.toFixed(2)}, ` +
        `minY=${minY.toFixed(2)}, maxAbsX=${maxAbsX.toFixed(2)}`,
    );

    // Bodies flying above y=18 = explosion (initial max ~13.25)
    const exploded = ys.filter((y) => y > 18).length;
    expect(
      exploded,
      `${exploded} bodies flew above y=18 (explosion). maxY=${maxY.toFixed(2)}`,
    ).toBe(0);

    // Horizontal footprint: initial ~8, allow up to 13
    expect(
      maxAbsX,
      `Bodies spread to x=${maxAbsX.toFixed(2)}. Explosion.`,
    ).toBeLessThan(13.0);

    // No bodies through floor
    expect(minY, `Body fell through ground: y=${minY.toFixed(2)}`).toBeGreaterThan(-1.0);

    // Bottom row should be mostly intact (at least 12 of 16 boxes at y<1.5)
    const bottomRow = ys.filter((y) => y >= 0.3 && y <= 1.5).length;
    expect(
      bottomRow,
      `Only ${bottomRow} bodies in bottom row (expected ~16). Base disintegrated.`,
    ).toBeGreaterThanOrEqual(12);

    // Center of mass should not have risen much from initial ~4.75
    expect(
      comY,
      `Center of mass at ${comY.toFixed(2)}, expected near 4.75. Energy pumping.`,
    ).toBeLessThan(6.0);

    const realErrors = errors.filter(
      (e) => !e.includes("DevTools") && !e.includes("favicon"),
    );
    expect(
      realErrors,
      `Errors:\n${realErrors.join("\n")}`,
    ).toHaveLength(0);
  });
});
