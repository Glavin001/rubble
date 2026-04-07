import { test, expect } from "@playwright/test";

/**
 * WebGPU narrowphase regression test.
 *
 * Loads the real 3D demo with the "Scatter" scene (mixed spheres, boxes,
 * capsules) and asserts that:
 *  - No WebGPU validation errors/warnings are emitted
 *  - The narrowphase compute phase runs and produces sane timings
 *  - The simulation advances without errors
 *
 * This catches browser-only failures like exceeding
 * maxStorageBuffersPerShaderStage that unit/WASM tests cannot detect.
 */

const WEBGPU_VALIDATION_PATTERNS = [
  "exceeds the maximum per-stage limit",
  "Invalid ComputePipeline",
  "Invalid BindGroupLayout",
  "Invalid CommandBuffer",
  "While calling [Device].CreateComputePipeline",
  "While calling [Device].CreateBindGroupLayout",
  "is not valid for the bind group layout",
];

function isWebGPUValidationMessage(text: string): boolean {
  return WEBGPU_VALIDATION_PATTERNS.some((pattern) => text.includes(pattern));
}

test.describe("WebGPU narrowphase regression", () => {
  let gpuErrors: string[] = [];
  let gpuWarnings: string[] = [];
  let pageErrors: string[] = [];

  test.beforeEach(async ({ page }) => {
    gpuErrors = [];
    gpuWarnings = [];
    pageErrors = [];

    // Capture console errors and warnings before navigation
    page.on("pageerror", (err) => pageErrors.push(err.message));
    page.on("console", (msg) => {
      const text = msg.text();
      if (msg.type() === "error") {
        gpuErrors.push(text);
      }
      if (msg.type() === "warning" && isWebGPUValidationMessage(text)) {
        gpuWarnings.push(text);
      }
    });

    // Navigate to the 3D demo (no ?bodies= param — use scene picker)
    await page.goto("/src/3d/index.html");

    // Wait for WASM + world init
    await page.waitForFunction(
      () => window.__rubble_test?.ready === true,
      null,
      { timeout: 30_000 },
    );
  });

  test("Scatter scene runs without WebGPU validation errors", async ({
    page,
  }) => {
    // Switch to the Scatter scene (mixed shape types → exercises narrowphase)
    await page.selectOption("#scene-select", "Scatter");

    // Wait for the scene to load and simulation to advance
    await page.waitForFunction(
      () => {
        const t = window.__rubble_test;
        const loadingHidden =
          document.getElementById("loading")?.style.display === "none";
        return (
          loadingHidden &&
          (t?.bodyCount ?? 0) >= 2500 &&
          (t?.stepCount ?? 0) >= 5
        );
      },
      null,
      { timeout: 180_000 },
    );

    // Assert no init error
    const error = await page.evaluate(() => window.__rubble_test?.error);
    expect(error).toBeNull();

    // Assert no WebGPU validation errors or warnings
    const validationErrors = gpuErrors.filter(isWebGPUValidationMessage);
    const allValidationIssues = [...validationErrors, ...gpuWarnings];
    expect(
      allValidationIssues,
      `WebGPU validation failures detected:\n${allValidationIssues.join("\n")}`,
    ).toHaveLength(0);

    // Assert no page-level errors
    const realPageErrors = pageErrors.filter(
      (e) => !e.includes("DevTools") && !e.includes("favicon"),
    );
    expect(
      realPageErrors,
      `Page errors detected:\n${realPageErrors.join("\n")}`,
    ).toHaveLength(0);
  });

  test("narrowphase timing is present and sane after Scatter steps", async ({
    page,
  }) => {
    await page.selectOption("#scene-select", "Scatter");

    // Wait for scene load + several steps
    await page.waitForFunction(
      () => {
        const t = window.__rubble_test;
        const loadingHidden =
          document.getElementById("loading")?.style.display === "none";
        return (
          loadingHidden &&
          (t?.bodyCount ?? 0) >= 2500 &&
          (t?.stepCount ?? 0) >= 10
        );
      },
      null,
      { timeout: 180_000 },
    );

    // Sample narrowphase timing (index 3)
    const timings = await page.evaluate(() => {
      const t = window.__rubble_test?.lastStepTimingsMs;
      return t ? Array.from(t) : null;
    });

    expect(timings).not.toBeNull();
    expect(timings!).toHaveLength(7);

    // [3] = narrowphase — should be finite and non-negative
    const narrowphaseMs = timings![3];
    expect(Number.isFinite(narrowphaseMs)).toBe(true);
    expect(narrowphaseMs).toBeGreaterThanOrEqual(0);

    // All other phases should also be finite
    for (let i = 0; i < 7; i++) {
      expect(
        Number.isFinite(timings![i]),
        `timings[${i}] should be finite, got ${timings![i]}`,
      ).toBe(true);
    }
  });

  test("no WebGPU validation errors in console across all loaded scenes", async ({
    page,
  }) => {
    // After initial scene load, check for any validation errors
    // that appeared during init (default scene)
    await page.waitForFunction(
      () => {
        const t = window.__rubble_test;
        return (
          document.getElementById("loading")?.style.display === "none" &&
          (t?.stepCount ?? 0) >= 3
        );
      },
      null,
      { timeout: 60_000 },
    );

    const validationErrors = gpuErrors.filter(isWebGPUValidationMessage);
    const allIssues = [...validationErrors, ...gpuWarnings];
    expect(
      allIssues,
      `WebGPU validation failures on default scene:\n${allIssues.join("\n")}`,
    ).toHaveLength(0);
  });
});
