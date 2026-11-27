// The studio's HTTP API, as connect middleware (mounted by vite.config.ts in `npm run dev`
// and `npm run preview`). Local only: the dev server binds 127.0.0.1, and requests whose
// Host is not a loopback name are refused (no DNS rebinding); writes need a JSON or
// octet-stream body, which a cross-site form cannot send without a CORS preflight.
//
//   GET  /api/state                  shots, frame summaries, structure schemas, tools
//   GET  /api/tools                  the tool list (shared/tools.ts)
//   POST /api/tool/<name>            run a server tool: JSON input -> {ok, result | error}
//   POST /api/upload?name=&shot=&view=&moment=   raw image bytes -> the new frame
//   GET  /library/<path>             an image of the library
//   GET  /cache/<path>               a still Rubble's crash_outlines rendered

import { readFileSync, statSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { extname } from 'node:path';
import { TOOLS } from '../shared/tools.ts';
import { Ops } from './ops.ts';
import { Rust } from './rust.ts';
import { IMAGE_TYPES, Store, StudioError, safeJoin, type Paths } from './store.ts';

const LOOPBACK = /^(127\.0\.0\.1|localhost|\[::1\])(:\d+)?$/;
const MAX_UPLOAD = 64 << 20;

function body(req: IncomingMessage, limit: number): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const parts: Buffer[] = [];
    let size = 0;
    req.on('data', (c: Buffer) => {
      size += c.length;
      if (size > limit) {
        reject(new StudioError(`body over ${limit >> 20} MB`, 413));
        req.destroy();
      } else parts.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(parts)));
    req.on('error', reject);
  });
}

function send(res: ServerResponse, code: number, data: unknown, type = 'application/json'): void {
  const b = Buffer.isBuffer(data) ? data : Buffer.from(type === 'application/json' ? JSON.stringify(data) : String(data));
  res.writeHead(code, { 'content-type': type, 'content-length': b.length, 'cache-control': 'no-store', 'x-content-type-options': 'nosniff' });
  res.end(b);
}

function file(res: ServerResponse, p: string): void {
  const type = IMAGE_TYPES[extname(p).toLowerCase()];
  if (!type) throw new StudioError('not an image', 403);
  let st;
  try {
    st = statSync(p);
  } catch {
    throw new StudioError('not found', 404);
  }
  if (!st.isFile()) throw new StudioError('not found', 404);
  send(res, 200, readFileSync(p), type);
}

export interface Api {
  handle: (req: IncomingMessage, res: ServerResponse, next: () => void) => void;
  ops: Ops;
  close: () => void;
}

export function createApi(paths?: Paths): Api {
  const store = new Store(paths);
  const rust = new Rust(store.paths.cache);
  const ops = new Ops(store, rust);

  async function route(req: IncomingMessage, res: ServerResponse): Promise<boolean> {
    const url = new URL(req.url ?? '/', 'http://127.0.0.1');
    const path = decodeURIComponent(url.pathname);
    const isApi = path.startsWith('/api/') || path.startsWith('/library/') || path.startsWith('/cache/');
    if (!isApi) return false;
    if (!LOOPBACK.test(req.headers.host ?? '')) throw new StudioError('the studio answers on 127.0.0.1 only', 403);
    if (req.method === 'GET' && path === '/api/state') {
      send(res, 200, { shots: ops.t_list_shots(), frames: ops.summaries(), schemas: store.structures(), tools: TOOLS, paths: { data: store.paths.data, library: store.paths.library } });
      return true;
    }
    if (req.method === 'GET' && path === '/api/tools') {
      send(res, 200, TOOLS);
      return true;
    }
    if (req.method === 'GET' && path.startsWith('/library/')) {
      file(res, store.imagePath(path.slice('/library/'.length)));
      return true;
    }
    if (req.method === 'GET' && path.startsWith('/cache/')) {
      file(res, safeJoin(store.paths.cache, path.slice('/cache/'.length)));
      return true;
    }
    if (req.method === 'POST' && path.startsWith('/api/tool/')) {
      if (!(req.headers['content-type'] ?? '').startsWith('application/json')) throw new StudioError('send the input as application/json', 415);
      const text = (await body(req, MAX_UPLOAD * 2)).toString('utf8');
      let input: Record<string, unknown>;
      try {
        input = text ? JSON.parse(text) : {};
      } catch {
        throw new StudioError('the body is not JSON');
      }
      send(res, 200, { ok: true, result: await ops.call(path.slice('/api/tool/'.length), input) });
      return true;
    }
    if (req.method === 'POST' && path === '/api/upload') {
      if (req.headers['content-type'] !== 'application/octet-stream') throw new StudioError('send the image as application/octet-stream', 415);
      const name = url.searchParams.get('name') ?? '';
      const input = Object.fromEntries(['shot', 'view', 'moment'].map((k) => [k, url.searchParams.get(k) ?? undefined]));
      send(res, 200, { ok: true, result: ops.addImage(name, await body(req, MAX_UPLOAD), input) });
      return true;
    }
    throw new StudioError(`no route ${req.method} ${path}`, 404);
  }

  return {
    ops,
    close: () => rust.stop(),
    handle(req, res, next) {
      route(req, res)
        .then((done) => {
          if (!done) next();
        })
        .catch((e: unknown) => {
          const err = e instanceof StudioError ? e : new StudioError(String((e as Error)?.message ?? e), 500);
          if (!res.headersSent) send(res, err.status, { ok: false, error: err.message });
        });
    },
  };
}
