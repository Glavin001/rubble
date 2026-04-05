import { test, expect } from "@playwright/test";

// Helper: wait for the physics engine to initialize
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
  // SwiftShader WebGPU broadphase readback can take ~0.5s/step even at 50 bodies,
  // so 120 steps needs ~60s. Use a generous timeout to keep CI reliable.
  await page.waitForFunction(
    (n) => (window.__rubble_test?.stepCount ?? 0) >= n,
    minSteps,
    { timeout: 90_000 },
  );
}

test.describe("3D Physics Demo", () => {
  test.beforeEach(async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (err) => errors.push(err.message));
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });

    // Use a small body count so SwiftShader (CI) can step fast enough for
    // the waitForSteps timeouts below.
    await page.goto("/src/3d/index.html?bodies=50");
    await waitForReady(page);

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
    // 1 ground plane + 50 dynamic bodies = 51 (see ?bodies=50 above)
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
    // Record initial transforms (7 floats per body: x,y,z,qx,qy,qz,qw)
    const initialTransforms = await page.evaluate(() => {
      return Array.from(window.__rubble_test!.getTransforms());
    });

    // Wait for physics to simulate
    await waitForSteps(page, 30);

    const laterTransforms = await page.evaluate(() => {
      return Array.from(window.__rubble_test!.getTransforms());
    });

    // Check that dynamic bodies (index 1+, skip ground plane at 0) have fallen
    let movedDown = 0;
    const dynamicStart = 1; // skip ground plane
    const numBodies = initialTransforms.length / 7;
    for (let i = dynamicStart; i < numBodies; i++) {
      const initialY = initialTransforms[i * 7 + 1];
      const laterY = laterTransforms[i * 7 + 1];
      if (laterY < initialY - 0.01) movedDown++;
    }

    // At least 30% of dynamic bodies should have moved down
    const dynamicCount = numBodies - dynamicStart;
    expect(movedDown).toBeGreaterThan(dynamicCount * 0.3);
  });

  test("bodies stay above ground plane (y >= -1 tolerance)", async ({
    page,
  }) => {
    // Let simulation run and bodies settle
    await waitForSteps(page, 120);

    const transforms = await page.evaluate(() => {
      return Array.from(window.__rubble_test!.getTransforms());
    });

    const numBodies = transforms.length / 7;
    for (let i = 1; i < numBodies; i++) {
      // skip ground plane
      const y = transforms[i * 7 + 1];
      // Bodies should not penetrate far below ground (y=0)
      // Allow some tolerance for solver imprecision
      expect(y).toBeGreaterThan(-2.0);
    }
  });

  test("quaternions are valid (unit length)", async ({ page }) => {
    await waitForSteps(page, 10);

    const transforms = await page.evaluate(() => {
      return Array.from(window.__rubble_test!.getTransforms());
    });

    const numBodies = transforms.length / 7;
    for (let i = 0; i < numBodies; i++) {
      const qx = transforms[i * 7 + 3];
      const qy = transforms[i * 7 + 4];
      const qz = transforms[i * 7 + 5];
      const qw = transforms[i * 7 + 6];
      const len = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
      // Quaternion should be approximately unit length (within floating point tolerance)
      expect(len).toBeGreaterThan(0.9);
      expect(len).toBeLessThan(1.1);
    }
  });

  test("transforms array has correct structure (7 floats per body)", async ({
    page,
  }) => {
    await waitForSteps(page, 5);
    const result = await page.evaluate(() => {
      const transforms = window.__rubble_test!.getTransforms();
      return {
        transformsLength: transforms.length,
        bodyCount: window.__rubble_test!.bodyCount,
      };
    });

    expect(result.transformsLength).toBeGreaterThan(0);
    expect(result.transformsLength % 7).toBe(0);
  });

  test("no positions are NaN", async ({ page }) => {
    await waitForSteps(page, 30);

    const transforms = await page.evaluate(() => {
      return Array.from(window.__rubble_test!.getTransforms());
    });

    for (let i = 0; i < transforms.length; i++) {
      expect(Number.isNaN(transforms[i])).toBe(false);
      expect(Number.isFinite(transforms[i])).toBe(true);
    }
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
    const realErrors = errors.filter(
      (e) => !e.includes("DevTools") && !e.includes("favicon"),
    );
    expect(realErrors).toHaveLength(0);
  });
});
