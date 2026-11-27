import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// A static site: `npm run dev` serves it, `npm run build` writes dist/ for any static host.
// There is no server side: the studio stores to a folder the user picks (or this browser's
// storage), and the engine is wasm under public/rubble/ (scripts/engine-assets.ts).
const port = Number(process.env.STUDIO_PORT || 8770);

export default defineConfig({
  base: './',
  plugins: [react()],
  server: { host: '127.0.0.1', port, strictPort: true, watch: { ignored: ['**/workspace/library/**', '**/workspace/recordings/**', '**/public/**'] } },
  preview: { host: '127.0.0.1', port, strictPort: true },
  build: { outDir: 'dist', sourcemap: true, target: 'es2022' },
});
