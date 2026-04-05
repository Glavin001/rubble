import { chromium } from "playwright";
import { appendFileSync, createReadStream, existsSync, statSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, extname, join, normalize } from "path";
import http from "http";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const webRoot = join(__dirname, "..", "dist");
const logPath = "/opt/cursor/logs/debug.log";
const port = Number(process.env.REPRO_PORT ?? "4174");

function writeLog(entry) {
  appendFileSync(logPath, JSON.stringify({ timestamp: Date.now(), ...entry }) + "\n");
}

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".wasm": "application/wasm",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

const server = http.createServer((request, response) => {
  const url = new URL(request.url ?? "/", "http://127.0.0.1:4173");
  const pathname = decodeURIComponent(url.pathname);
  const requested = pathname === "/" ? "/index.html" : pathname;
  const normalized = normalize(requested).replace(/^(\.\.[/\\])+/, "");
  let filePath = join(webRoot, normalized);
  if (existsSync(filePath) && statSync(filePath).isDirectory()) {
    filePath = join(filePath, "index.html");
  }
  if (!existsSync(filePath)) {
    response.statusCode = 404;
    response.end("Not found");
    return;
  }
  response.setHeader(
    "Content-Type",
    mimeTypes[extname(filePath).toLowerCase()] ?? "application/octet-stream",
  );
  createReadStream(filePath).pipe(response);
});

await new Promise((resolve) => server.listen(port, "127.0.0.1", resolve));

const browser = await chromium.launch({
  executablePath: "/usr/local/bin/google-chrome",
  headless: true,
  args: [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--headless=new",
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan",
    "--use-vulkan=swiftshader",
    "--use-angle=swiftshader",
    "--disable-vulkan-fallback-to-gl-for-testing",
    "--disable-gpu-sandbox",
    "--disable-dev-shm-usage",
  ],
});

try {
  const page = await browser.newPage();
  page.on("console", async (msg) => {
    writeLog({
      source: "console",
      type: msg.type(),
      text: msg.text(),
      location: msg.location(),
    });
  });
  page.on("pageerror", (error) => {
    writeLog({
      source: "pageerror",
      text: error.message,
      stack: error.stack,
    });
  });

  const path = process.argv[2] ?? "/src/3d/index.html";
  await page.goto(`http://127.0.0.1:${port}${path}`, { waitUntil: "networkidle" });
  await page.waitForTimeout(6000);
  const testState = await page.evaluate(() => ({
    ready: window.__rubble_test?.ready ?? null,
    stepCount: window.__rubble_test?.stepCount ?? null,
    bodyCount: window.__rubble_test?.bodyCount ?? null,
    error: window.__rubble_test?.error ?? null,
  }));
  writeLog({ source: "state", path, state: testState });
} finally {
  await browser.close();
  await new Promise((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
}
