import { test, expect } from "@playwright/test";

/**
 * WASM physics stability test: Stack scene.
 *
 * Verifies that a 10-box vertical stack settles stably:
 * - The top-placed box must remain the highest body (stack didn't topple)
 * - All boxes must stay within a tight horizontal footprint (no flying apart)
 * - No body falls through the ground
 *
 * Usage:
 *   cd web && npx playwright test tests/stability-stack.spec.ts --reporter=list
 */

const SETTLE_STEPS = 600; // 5 seconds at 120Hz

test.describe("WASM Stack Stability", () => {
  test.setTimeout(180_000);

  test("stack remains upright after settling", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (err) => errors.push(err.message));
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });

    await page.goto("/src/3d/index.html");

    await page.waitForFunction(
      () => {
        const t = window.__rubble_test;
        if (t?.error) throw new Error(`init failed: ${t.error}`);
        return t?.ready === true;
      },
      null,
      { timeout: 60_000 },
    );

    await page.waitForFunction(
      () => document.getElementById("loading")?.style.display === "none",
      null,
      { timeout: 60_000 },
    );

    // Switch to Stack scene (1 ground + 10 boxes = 11 bodies)
    await page.selectOption("#scene-select", "Stack");

    // Wait for scene to actually switch — body count must drop from Pyramid's ~137
    await page.waitForFunction(
      () => {
        const t = window.__rubble_test;
        return (
          (t?.bodyCount ?? 0) <= 15 &&
          (t?.bodyCount ?? 0) >= 10 &&
          document.getElementById("loading")?.style.display === "none"
        );
      },
      null,
      { timeout: 60_000 },
    );

    const hasBenchStep = await page.evaluate(
      () => typeof window.__rubble_test?.benchStep === "function",
    );
    expect(hasBenchStep, "benchStep() not available").toBe(true);

    await page.evaluate(() => window.__rubble_test!.stopLoop!());

    // Record initial state (before any physics)
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

    // Find which body index is initially the highest (the top of the stack)
    // Skip index 0 (ground plane)
    let topBodyIdx = 1;
    for (let i = 2; i < initialState.length; i++) {
      if (initialState[i].y > initialState[topBodyIdx].y) {
        topBodyIdx = i;
      }
    }
    const initialTopY = initialState[topBodyIdx].y;
    // eslint-disable-next-line no-console
    console.log(
      `Stack: ${initialState.length} bodies, top body idx=${topBodyIdx} at y=${initialTopY.toFixed(2)}`,
    );

    // --- Settle ---
    for (let i = 0; i < SETTLE_STEPS; i++) {
      await page.evaluate(() => window.__rubble_test!.benchStep!());
      // Log progress every 100 steps
      if ((i + 1) % 200 === 0) {
        const snap = await page.evaluate(() => {
          const transforms = window.__rubble_test!.getTransforms!();
          const n = transforms.length / 7;
          const ys: number[] = [];
          for (let j = 1; j < n; j++) ys.push(transforms[j * 7 + 1]);
          return {
            maxY: Math.max(...ys),
            minY: Math.min(...ys),
          };
        });
        // eslint-disable-next-line no-console
        console.log(
          `  step ${i + 1}: maxY=${snap.maxY.toFixed(2)}, minY=${snap.minY.toFixed(2)}`,
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

    // Dynamic bodies only (skip ground at index 0)
    const dynamicBodies = finalState.slice(1);

    // --- Assertions ---

    // 1. Top body must still be THE highest (or within 0.5 of highest)
    const finalYs = dynamicBodies.map((b) => b.y);
    const finalMaxY = Math.max(...finalYs);
    const topBodyFinalY = finalState[topBodyIdx].y;

    // eslint-disable-next-line no-console
    console.log(
      `Final: top body y=${topBodyFinalY.toFixed(2)}, ` +
        `highest y=${finalMaxY.toFixed(2)}, ` +
        `xs=[${dynamicBodies.map((b) => b.x.toFixed(1)).join(",")}]`,
    );

    expect(
      topBodyFinalY,
      `Top body (idx=${topBodyIdx}) is at y=${topBodyFinalY.toFixed(2)} ` +
        `but highest body is at y=${finalMaxY.toFixed(2)}. Stack toppled.`,
    ).toBeGreaterThanOrEqual(finalMaxY - 0.5);

    // 2. All dynamic bodies must stay within a horizontal footprint
    // Stack is centered at x=0,z=0 with box half-extent=1.0
    // Allow generous 3.0 units of drift (but not 10+)
    const MAX_HORIZONTAL_DRIFT = 3.0;
    for (let i = 0; i < dynamicBodies.length; i++) {
      const b = dynamicBodies[i];
      const hDist = Math.sqrt(b.x * b.x + b.z * b.z);
      expect(
        hDist,
        `Body ${i + 1} drifted ${hDist.toFixed(2)} from center ` +
          `(x=${b.x.toFixed(2)}, z=${b.z.toFixed(2)}). Stack exploded.`,
      ).toBeLessThan(MAX_HORIZONTAL_DRIFT);
    }

    // 3. No body fell through the ground
    const finalMinY = Math.min(...finalYs);
    expect(
      finalMinY,
      `Body fell through ground: minY=${finalMinY.toFixed(2)}`,
    ).toBeGreaterThan(-1.0);

    // 4. Top body should still be well above the ground (stack didn't fully collapse)
    // 10 boxes stacked: top should be at least y=5 (half the stack height)
    expect(
      topBodyFinalY,
      `Top body too low at y=${topBodyFinalY.toFixed(2)}. Stack collapsed.`,
    ).toBeGreaterThan(5.0);

    const realErrors = errors.filter(
      (e) => !e.includes("DevTools") && !e.includes("favicon"),
    );
    expect(
      realErrors,
      `Errors during test:\n${realErrors.join("\n")}`,
    ).toHaveLength(0);
  });
});
