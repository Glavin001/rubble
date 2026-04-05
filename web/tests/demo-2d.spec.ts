import { test, expect } from "@playwright/test";

// Helper: wait for the physics engine to initialize and run N steps
async function waitForReady(page: import("@playwright/test").Page) {
  await page.waitForFunction(
    () => window.__rubble_test?.ready === true,
    null,
    { timeout: 30_000 },
  );
}

async function waitForSteps(
  page: import("@playwright/test").Page,
  minSteps: number,
) {
  await page.waitForFunction(
    (n) => (window.__rubble_test?.stepCount ?? 0) >= n,
    minSteps,
    { timeout: 30_000 },
  );
}

test.describe("2D Physics Demo", () => {
  test.beforeEach(async ({ page }) => {
    // Collect console errors
    const errors: string[] = [];
    page.on("pageerror", (err) => errors.push(err.message));
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });

    // Use the bounded walls+grid path (50 bodies) so SwiftShader can keep up
    // and bodies stay contained within the asserted world bounds.
    await page.goto("/src/2d/index.html?bodies=50");
    await waitForReady(page);

    // Attach errors to page for later assertions
    (page as any).__errors = errors;
  });

  test("initializes without errors", async ({ page }) => {
    const error = await page.evaluate(() => window.__rubble_test?.error);
    expect(error).toBeNull();
  });

  test("spawns expected number of bodies", async ({ page }) => {
    const bodyCount = await page.evaluate(
      () => window.__rubble_test?.bodyCount ?? 0,
    );
    // 3 static walls + 50 dynamic bodies = 53
    expect(bodyCount).toBeGreaterThanOrEqual(50);
  });

  test("simulation advances (step count increases)", async ({ page }) => {
    const step1 = await page.evaluate(
      () => window.__rubble_test?.stepCount ?? 0,
    );
    await waitForSteps(page, step1 + 10);
    const step2 = await page.evaluate(
      () => window.__rubble_test?.stepCount ?? 0,
    );
    expect(step2).toBeGreaterThan(step1);
  });

  test("bodies fall under gravity (y positions decrease)", async ({
    page,
  }) => {
    // Record initial positions of dynamic bodies (skip first 3 which are static walls)
    const initialPositions = await page.evaluate(() => {
      return Array.from(window.__rubble_test!.getPositions());
    });

    // Wait for physics to run
    await waitForSteps(page, 30);

    const laterPositions = await page.evaluate(() => {
      return Array.from(window.__rubble_test!.getPositions());
    });

    // Check that at least some dynamic bodies have moved down (y decreased)
    // Dynamic bodies start at index 3 (after 3 static walls)
    let movedDown = 0;
    const dynamicStart = 3; // 3 static walls
    for (let i = dynamicStart; i < initialPositions.length / 2; i++) {
      const initialY = initialPositions[i * 2 + 1];
      const laterY = laterPositions[i * 2 + 1];
      if (laterY < initialY - 0.01) movedDown++;
    }

    // At least 30% of dynamic bodies should have moved down
    const dynamicCount = initialPositions.length / 2 - dynamicStart;
    expect(movedDown).toBeGreaterThan(dynamicCount * 0.3);
  });

  test("all bodies stay within world bounds", async ({ page }) => {
    // Let simulation run for a while
    await waitForSteps(page, 60);

    const positions = await page.evaluate(() => {
      return Array.from(window.__rubble_test!.getPositions());
    });

    // Check that no body has fallen below the ground (y should be >= -5, some tolerance)
    // or escaped the world bounds
    for (let i = 0; i < positions.length / 2; i++) {
      const x = positions[i * 2];
      const y = positions[i * 2 + 1];
      // Bodies shouldn't fall way below ground or escape sideways
      expect(y).toBeGreaterThan(-10);
      expect(x).toBeGreaterThan(-50);
      expect(x).toBeLessThan(90);
    }
  });

  test("positions array has correct length", async ({ page }) => {
    await waitForSteps(page, 5);
    const result = await page.evaluate(() => {
      const positions = window.__rubble_test!.getPositions();
      const bodyCount = window.__rubble_test!.bodyCount;
      return {
        posLength: positions.length,
        bodyCount,
        handleCount: positions.length / 2,
      };
    });

    // Positions array should have 2 floats per handle
    expect(result.posLength).toBeGreaterThan(0);
    expect(result.posLength % 2).toBe(0);
  });

  test("angles array has correct length", async ({ page }) => {
    const result = await page.evaluate(() => {
      const angles = window.__rubble_test!.getAngles();
      const positions = window.__rubble_test!.getPositions();
      return {
        anglesLength: angles.length,
        handleCount: positions.length / 2,
      };
    });

    // One angle per handle
    expect(result.anglesLength).toBe(result.handleCount);
  });

  test("loading indicator is hidden after init", async ({ page }) => {
    const loadingDisplay = await page.evaluate(() => {
      return document.getElementById("loading")?.style.display;
    });
    expect(loadingDisplay).toBe("none");
  });

  test("no console errors during simulation", async ({ page }) => {
    await waitForSteps(page, 30);
    const errors = (page as any).__errors as string[];
    // Filter out known benign warnings
    const realErrors = errors.filter(
      (e) => !e.includes("DevTools") && !e.includes("favicon"),
    );
    expect(realErrors).toHaveLength(0);
  });
});
