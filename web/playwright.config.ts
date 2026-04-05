import { defineConfig } from "@playwright/test";
import { existsSync } from "fs";
import { join } from "path";
import { homedir } from "os";

// Find a working Chromium binary: env var > Playwright cache > system
function findChromium(): string | undefined {
  if (process.env.PLAYWRIGHT_CHROMIUM_PATH) {
    return process.env.PLAYWRIGHT_CHROMIUM_PATH;
  }
  // Search Playwright cache for any installed chromium
  const cacheDir = join(homedir(), ".cache", "ms-playwright");
  try {
    const { readdirSync } = require("fs");
    const entries = readdirSync(cacheDir) as string[];
    // Find chromium dirs (not headless_shell), sorted descending to pick newest
    const chromiumDirs = entries
      .filter((e: string) => e.startsWith("chromium-") && !e.includes("headless"))
      .sort()
      .reverse();
    for (const dir of chromiumDirs) {
      const chromePath = join(cacheDir, dir, "chrome-linux", "chrome");
      if (existsSync(chromePath)) return chromePath;
    }
  } catch {}
  return undefined;
}

const chromiumPath = findChromium();

export default defineConfig({
  testDir: "./tests",
  timeout: 120_000,
  retries: 1,
  use: {
    baseURL: "http://localhost:4173",
  },
  projects: [
    {
      name: "chromium-webgpu",
      use: {
        browserName: "chromium",
        launchOptions: {
          ...(chromiumPath ? { executablePath: chromiumPath } : {}),
          args: [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--headless=new",
            // Enable WebGPU with SwiftShader CPU emulation
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan",
            "--use-vulkan=swiftshader",
            "--use-angle=swiftshader",
            "--disable-vulkan-fallback-to-gl-for-testing",
            "--disable-gpu-sandbox",
            // Stability flags for CI
            "--disable-dev-shm-usage",
          ],
        },
      },
    },
  ],
  // Start Vite preview server before tests
  webServer: {
    command: "npx vite preview --port 4173",
    port: 4173,
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
