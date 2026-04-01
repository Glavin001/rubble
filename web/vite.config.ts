import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";
import { resolve } from "path";

export default defineConfig({
  plugins: [wasm(), topLevelAwait()],
  build: {
    target: "esnext",
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        "demo-2d": resolve(__dirname, "src/2d/index.html"),
        "demo-3d": resolve(__dirname, "src/3d/index.html"),
      },
    },
  },
  optimizeDeps: {
    exclude: ["rubble-wasm"],
  },
});
