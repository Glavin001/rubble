import react from '@vitejs/plugin-react';
import { defineConfig, type Plugin } from 'vite';
import { createApi } from './server/api.ts';

/** The studio's file-system API inside Vite's own server: one command starts everything. */
function studioApi(): Plugin {
  let api: ReturnType<typeof createApi> | null = null;
  const get = () => (api ??= createApi());
  return {
    name: 'reference-studio-api',
    configureServer(server) {
      server.middlewares.use((req, res, next) => get().handle(req, res, next));
      server.httpServer?.on('close', () => api?.close());
    },
    configurePreviewServer(server) {
      server.middlewares.use((req, res, next) => get().handle(req, res, next));
      server.httpServer?.on('close', () => api?.close());
    },
  };
}

const port = Number(process.env.STUDIO_PORT || 8770);

export default defineConfig({
  plugins: [react(), studioApi()],
  server: { host: '127.0.0.1', port, strictPort: true, watch: { ignored: ['**/library/**'] } },
  preview: { host: '127.0.0.1', port, strictPort: true },
  build: { outDir: 'dist', sourcemap: true },
});
