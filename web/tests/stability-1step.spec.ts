import { test, expect } from "@playwright/test";

test.describe("incremental diagnostic", () => {
  test.setTimeout(120_000);

  test("positions stable through 100 steps", async ({ page }) => {
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
    await page.waitForFunction(
      () => document.getElementById("loading")?.style.display === "none",
      null,
      { timeout: 60_000 },
    );

    await page.evaluate(() => window.__rubble_test!.stopLoop!());

    const getStats = async () => {
      return await page.evaluate(() => {
        const t = window.__rubble_test!.getTransforms!();
        const n = t.length / 7;
        const ys = [];
        const xs = [];
        for (let i = 1; i < n; i++) {
          ys.push(t[i * 7 + 1]);
          xs.push(Math.abs(t[i * 7 + 0]));
        }
        return {
          maxY: Math.max(...ys),
          minY: Math.min(...ys),
          maxAbsX: Math.max(...xs),
          comY: ys.reduce((a, b) => a + b, 0) / ys.length,
        };
      });
    };

    const init = await getStats();
    console.log(`Init: comY=${init.comY.toFixed(2)}, maxY=${init.maxY.toFixed(2)}`);

    for (let chunk = 0; chunk < 10; chunk++) {
      for (let i = 0; i < 10; i++) {
        await page.evaluate(() => window.__rubble_test!.benchStep!());
      }
      const stats = await getStats();
      console.log(`step ${(chunk + 1) * 10}: comY=${stats.comY.toFixed(2)}, maxY=${stats.maxY.toFixed(2)}, minY=${stats.minY.toFixed(2)}, maxAbsX=${stats.maxAbsX.toFixed(2)}`);
      expect(stats.maxY, `maxY explosion at step ${(chunk + 1) * 10}`).toBeLessThan(init.maxY + 10.0);
    }

    const realErrors = errors.filter(e => !e.includes("DevTools") && !e.includes("favicon"));
    expect(realErrors, `Errors: ${realErrors.join("\n")}`).toHaveLength(0);
  });
});
