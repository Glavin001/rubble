// The studio's documents, held in memory and written through to the workspace folder
// (shared/workspace.ts): shots, frames, structure schemas, data scenarios, and the images'
// names. Reads are synchronous (the tools are); every write is queued to the workspace in
// order (`flush()` waits for them). Ids are checked so a document stays a file of its folder.

import type { Frame, ScenarioDesc, Shot, StructureSchema } from './model.ts';
import type { Workspace } from './workspace.ts';

export class StudioError extends Error {
  status: number;
  constructor(message: string, status = 400) {
    super(message);
    this.status = status;
  }
}

const ID = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$/;

export function checkId(id: unknown, what = 'id'): string {
  if (typeof id !== 'string' || !ID.test(id) || id.includes('..')) {
    throw new StudioError(`${what} must be letters, digits, '_', '-' or '.', starting with a letter or digit (got ${JSON.stringify(id)})`);
  }
  return id;
}

export const IMAGE_TYPES: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
};

export function extOf(name: string): string {
  const i = name.lastIndexOf('.');
  return i < 0 ? '' : name.slice(i).toLowerCase();
}

const u32be = (b: Uint8Array, i: number) => ((b[i] << 24) | (b[i + 1] << 16) | (b[i + 2] << 8) | b[i + 3]) >>> 0;
const u16be = (b: Uint8Array, i: number) => (b[i] << 8) | b[i + 1];
const u16le = (b: Uint8Array, i: number) => b[i] | (b[i + 1] << 8);
const u24le = (b: Uint8Array, i: number) => b[i] | (b[i + 1] << 8) | (b[i + 2] << 16);
const ascii = (b: Uint8Array, i: number, n: number) => String.fromCharCode(...b.subarray(i, i + n));

/** Width and height from an image file's header (PNG, JPEG, WebP), or null. */
export function imageSize(b: Uint8Array): { width: number; height: number } | null {
  if (b.length > 24 && u32be(b, 0) === 0x89504e47) return { width: u32be(b, 16), height: u32be(b, 20) };
  if (b.length > 30 && ascii(b, 0, 4) === 'RIFF' && ascii(b, 8, 4) === 'WEBP') {
    const kind = ascii(b, 12, 4);
    if (kind === 'VP8X') return { width: 1 + u24le(b, 24), height: 1 + u24le(b, 27) };
    if (kind === 'VP8L') {
      const v = (b[21] | (b[22] << 8) | (b[23] << 16) | (b[24] << 24)) >>> 0;
      return { width: 1 + (v & 0x3fff), height: 1 + ((v >> 14) & 0x3fff) };
    }
    if (kind === 'VP8 ') return { width: u16le(b, 26) & 0x3fff, height: u16le(b, 28) & 0x3fff };
    return null;
  }
  if (b.length > 4 && b[0] === 0xff && b[1] === 0xd8) {
    let i = 2;
    while (i + 9 < b.length) {
      if (b[i] !== 0xff) return null;
      const m = b[i + 1];
      const len = u16be(b, i + 2);
      if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc) return { width: u16be(b, i + 7), height: u16be(b, i + 5) };
      i += 2 + len;
    }
  }
  return null;
}

export async function sha256(b: Uint8Array): Promise<string> {
  const d = await crypto.subtle.digest('SHA-256', b as Uint8Array<ArrayBuffer>);
  return [...new Uint8Array(d)].map((x) => x.toString(16).padStart(2, '0')).join('');
}

const dec = new TextDecoder();

/** The workspace's folders. */
export const DIRS = { shots: 'shots', frames: 'frames', structures: 'structures', scenarios: 'scenarios', library: 'library', recordings: 'recordings' } as const;

export class Store {
  ws: Workspace;
  private shotsM = new Map<string, Shot>();
  private framesM = new Map<string, Frame>();
  private structsM = new Map<string, StructureSchema>();
  private scenariosM = new Map<string, ScenarioDesc>();
  /** Every image file in library/ (paths relative to it). */
  images = new Set<string>();
  /** Problems met while loading (a file that does not parse). */
  problems: string[] = [];
  private queue: Promise<void> = Promise.resolve();
  private writeError: string | null = null;
  /** Called after any change (the page re-renders). */
  onChange: (what: string) => void = () => {};

  private constructor(ws: Workspace) {
    this.ws = ws;
  }

  /** Every document of the workspace, read into memory. */
  static async open(ws: Workspace): Promise<Store> {
    const s = new Store(ws);
    const load = async <T>(dir: string, into: Map<string, T>) => {
      for (const f of await ws.list(dir)) {
        if (!f.endsWith('.json')) continue;
        try {
          const v = JSON.parse(dec.decode((await ws.read(`${dir}/${f}`))!)) as T & { id: string };
          into.set(v.id, v);
        } catch (e) {
          s.problems.push(`${dir}/${f}: ${(e as Error).message}`);
        }
      }
    };
    await load(DIRS.shots, s.shotsM);
    await load(DIRS.frames, s.framesM);
    await load(DIRS.structures, s.structsM);
    await load(DIRS.scenarios, s.scenariosM);
    const walk = async (dir: string, pre: string) => {
      for (const f of await ws.list(dir)) if (IMAGE_TYPES[extOf(f)]) s.images.add(pre + f);
      for (const d of await ws.dirs(dir)) await walk(`${dir}/${d}`, `${pre}${d}/`);
    };
    await walk(DIRS.library, '');
    return s;
  }

  private write(path: string, data: string | Uint8Array): void {
    this.queue = this.queue.then(() => this.ws.write(path, data)).catch((e) => {
      this.writeError = `${path}: ${(e as Error).message}`;
    });
  }

  /** Every write so far is in the workspace (or the first error is thrown). */
  async flush(): Promise<void> {
    await this.queue;
    if (this.writeError) {
      const e = this.writeError;
      this.writeError = null;
      throw new StudioError(`could not save ${e}`, 500);
    }
  }

  private putJson<T extends { id: string; updated?: string }>(dir: string, m: Map<string, T>, v: T, what: string): T {
    checkId(v.id, what);
    v.updated = new Date().toISOString();
    m.set(v.id, structuredClone(v));
    this.write(`${dir}/${v.id}.json`, JSON.stringify(v, null, 1) + '\n');
    this.onChange(what);
    return v;
  }

  shots(): Shot[] {
    return [...this.shotsM.values()].sort((a, b) => a.id.localeCompare(b.id)).map((x) => structuredClone(x));
  }
  shot(id: string): Shot {
    const x = this.shotsM.get(checkId(id, 'shot'));
    if (!x) throw new StudioError(`no shot ${id}`, 404);
    return structuredClone(x);
  }
  hasShot(id: string): boolean {
    return this.shotsM.has(id);
  }
  putShot(s: Shot): Shot {
    return this.putJson(DIRS.shots, this.shotsM, s, 'shot');
  }

  frames(): Frame[] {
    return [...this.framesM.values()].sort((a, b) => a.id.localeCompare(b.id)).map((x) => structuredClone(x));
  }
  frame(id: string): Frame {
    const x = this.framesM.get(checkId(id, 'frame'));
    if (!x) throw new StudioError(`no frame ${id}`, 404);
    return structuredClone(x);
  }
  hasFrame(id: string): boolean {
    return this.framesM.has(id);
  }
  putFrame(f: Frame): Frame {
    return this.putJson(DIRS.frames, this.framesM, f, 'frame');
  }

  structures(): StructureSchema[] {
    return [...this.structsM.values()].sort((a, b) => a.id.localeCompare(b.id));
  }
  structure(id: string): StructureSchema {
    const x = this.structsM.get(checkId(id, 'schema'));
    if (!x) throw new StudioError(`no structure schema ${id}`, 404);
    return x;
  }
  /** A structure schema written into the workspace (a new workspace starts with the shipped ones). */
  putStructure(s: StructureSchema): void {
    checkId(s.id, 'schema');
    this.structsM.set(s.id, s);
    this.write(`${DIRS.structures}/${s.id}.json`, JSON.stringify(s, null, 1) + '\n');
  }

  scenarios(): ScenarioDesc[] {
    return [...this.scenariosM.values()].sort((a, b) => a.id.localeCompare(b.id)).map((x) => structuredClone(x));
  }
  scenario(id: string): ScenarioDesc {
    const x = this.scenariosM.get(checkId(id, 'scenario'));
    if (!x) throw new StudioError(`no data scenario ${id}`, 404);
    return structuredClone(x);
  }
  hasScenario(id: string): boolean {
    return this.scenariosM.has(id);
  }
  putScenario(d: ScenarioDesc): ScenarioDesc {
    checkId(d.id, 'scenario');
    this.scenariosM.set(d.id, structuredClone(d));
    this.write(`${DIRS.scenarios}/${d.id}.json`, JSON.stringify(d, null, 1) + '\n');
    this.onChange('scenario');
    return d;
  }
  removeScenario(id: string): void {
    this.scenariosM.delete(checkId(id, 'scenario'));
    this.queue = this.queue.then(() => this.ws.remove(`${DIRS.scenarios}/${id}.json`));
    this.onChange('scenario');
  }

  imageExists(rel: string): boolean {
    return this.images.has(rel);
  }
  image(rel: string): Promise<Uint8Array | null> {
    if (!this.images.has(rel)) return Promise.resolve(null);
    return this.ws.read(`${DIRS.library}/${rel}`);
  }

  /** Save an image under `dir/name` in the library (never overwriting another one). */
  async saveImage(name: string, bytes: Uint8Array, dir = 'added'): Promise<{ file: string; width: number; height: number; sha256: string }> {
    const ext = extOf(name);
    if (!IMAGE_TYPES[ext]) throw new StudioError(`images only (png, jpg, webp): ${name}`);
    const size = imageSize(bytes);
    if (!size) throw new StudioError(`not a readable png, jpg or webp: ${name}`);
    const stem = name.slice(0, name.length - ext.length).replace(/[^A-Za-z0-9_.-]+/g, '_').replace(/^[._-]+/, '').slice(0, 80) || 'image';
    checkId(dir.replace(/\//g, '_'), 'folder');
    const hash = await sha256(bytes);
    let file = `${dir}/${stem}${ext}`;
    for (let k = 2; this.images.has(file); k++) {
      const there = await this.image(file);
      if (there && (await sha256(there)) === hash) break;
      file = `${dir}/${stem}-${k}${ext}`;
    }
    await this.ws.write(`${DIRS.library}/${file}`, bytes);
    this.images.add(file);
    return { file, ...size, sha256: hash };
  }

  /** A cached recording (recordings/<hash>.rrec), or null. */
  recording(hash: string): Promise<Uint8Array | null> {
    checkId(hash, 'recording');
    return this.ws.read(`${DIRS.recordings}/${hash}.rrec`);
  }
  putRecording(hash: string, bytes: Uint8Array): Promise<void> {
    checkId(hash, 'recording');
    return this.ws.write(`${DIRS.recordings}/${hash}.rrec`, bytes);
  }
}
