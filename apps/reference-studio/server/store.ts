// The studio's files: committed points-only data (shots, frames, structures), the
// gitignored image library and the gitignored cache of Rubble's outlines. Every path a
// request names is checked to stay inside its folder.

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync, renameSync, statSync, unlinkSync, writeFileSync } from 'node:fs';
import { dirname, extname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Frame, Shot, StructureSchema } from '../shared/model.ts';

export const PACKAGE = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const REPO = resolve(PACKAGE, '..', '..');
export const ENGINE = join(REPO, 'engine');

export interface Paths {
  /** Committed: shots/, frames/, structures/. */
  data: string;
  /** Gitignored: the images. */
  library: string;
  /** Gitignored: Rubble's outlines and stills, made on demand. */
  cache: string;
}

export function defaultPaths(): Paths {
  return {
    data: resolve(process.env.STUDIO_DATA || join(ENGINE, 'docs', 'crash-reference', 'studio')),
    library: resolve(process.env.STUDIO_LIBRARY || join(PACKAGE, 'library')),
    cache: resolve(process.env.STUDIO_CACHE || join(ENGINE, 'out', 'reference-studio')),
  };
}

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

/** `rel` inside `base`, or an error: no absolute paths, no '..', no NUL, no symlinked escape by name. */
export function safeJoin(base: string, rel: string): string {
  if (typeof rel !== 'string' || rel === '' || rel.includes('\0') || isAbsolute(rel) || rel.split(/[\\/]/).includes('..')) {
    throw new StudioError(`path not allowed: ${JSON.stringify(rel)}`, 403);
  }
  const p = resolve(base, rel);
  const r = relative(base, p);
  if (r === '' || r.startsWith('..') || isAbsolute(r) || r.split(sep).includes('..')) {
    throw new StudioError(`path not allowed: ${JSON.stringify(rel)}`, 403);
  }
  return p;
}

export const IMAGE_TYPES: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
};

/** Width and height from an image file's header (PNG, JPEG, WebP), or null. */
export function imageSize(b: Buffer): { width: number; height: number } | null {
  if (b.length > 24 && b.readUInt32BE(0) === 0x89504e47) return { width: b.readUInt32BE(16), height: b.readUInt32BE(20) };
  if (b.length > 30 && b.toString('ascii', 0, 4) === 'RIFF' && b.toString('ascii', 8, 12) === 'WEBP') {
    const kind = b.toString('ascii', 12, 16);
    if (kind === 'VP8X') return { width: 1 + b.readUIntLE(24, 3), height: 1 + b.readUIntLE(27, 3) };
    if (kind === 'VP8L') {
      const v = b.readUInt32LE(21);
      return { width: 1 + (v & 0x3fff), height: 1 + ((v >> 14) & 0x3fff) };
    }
    if (kind === 'VP8 ') return { width: b.readUInt16LE(26) & 0x3fff, height: b.readUInt16LE(28) & 0x3fff };
    return null;
  }
  if (b.length > 4 && b[0] === 0xff && b[1] === 0xd8) {
    let i = 2;
    while (i + 9 < b.length) {
      if (b[i] !== 0xff) return null;
      const m = b[i + 1];
      const len = b.readUInt16BE(i + 2);
      if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc) {
        return { width: b.readUInt16BE(i + 7), height: b.readUInt16BE(i + 5) };
      }
      i += 2 + len;
    }
  }
  return null;
}

export function sha256(b: Buffer): string {
  return createHash('sha256').update(b).digest('hex');
}

function readJson<T>(p: string): T {
  return JSON.parse(readFileSync(p, 'utf8')) as T;
}

/** Write through a temporary file so a crash never leaves half a file. */
export function writeJson(p: string, v: unknown): void {
  mkdirSync(dirname(p), { recursive: true });
  const tmp = `${p}.tmp-${process.pid}`;
  writeFileSync(tmp, JSON.stringify(v, null, 1) + '\n');
  renameSync(tmp, p);
}

function listJson(dir: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir).filter((f) => f.endsWith('.json')).sort();
}

export class Store {
  paths: Paths;
  constructor(paths: Paths = defaultPaths()) {
    this.paths = paths;
    for (const d of [join(paths.data, 'shots'), join(paths.data, 'frames'), paths.library, paths.cache]) mkdirSync(d, { recursive: true });
  }

  shots(): Shot[] {
    return listJson(join(this.paths.data, 'shots')).map((f) => readJson<Shot>(join(this.paths.data, 'shots', f)));
  }
  shot(id: string): Shot {
    const p = join(this.paths.data, 'shots', `${checkId(id, 'shot')}.json`);
    if (!existsSync(p)) throw new StudioError(`no shot ${id}`, 404);
    return readJson<Shot>(p);
  }
  hasShot(id: string): boolean {
    return existsSync(join(this.paths.data, 'shots', `${checkId(id, 'shot')}.json`));
  }
  putShot(s: Shot): Shot {
    checkId(s.id, 'shot');
    s.updated = new Date().toISOString();
    writeJson(join(this.paths.data, 'shots', `${s.id}.json`), s);
    return s;
  }
  deleteShot(id: string): void {
    unlinkSync(join(this.paths.data, 'shots', `${checkId(id, 'shot')}.json`));
  }

  frames(): Frame[] {
    return listJson(join(this.paths.data, 'frames')).map((f) => readJson<Frame>(join(this.paths.data, 'frames', f)));
  }
  frame(id: string): Frame {
    const p = join(this.paths.data, 'frames', `${checkId(id, 'frame')}.json`);
    if (!existsSync(p)) throw new StudioError(`no frame ${id}`, 404);
    return readJson<Frame>(p);
  }
  hasFrame(id: string): boolean {
    return existsSync(join(this.paths.data, 'frames', `${checkId(id, 'frame')}.json`));
  }
  putFrame(f: Frame): Frame {
    checkId(f.id, 'frame');
    f.updated = new Date().toISOString();
    writeJson(join(this.paths.data, 'frames', `${f.id}.json`), f);
    return f;
  }

  structures(): StructureSchema[] {
    return listJson(join(this.paths.data, 'structures')).map((f) => readJson<StructureSchema>(join(this.paths.data, 'structures', f)));
  }
  structure(id: string): StructureSchema {
    const p = join(this.paths.data, 'structures', `${checkId(id, 'schema')}.json`);
    if (!existsSync(p)) throw new StudioError(`no structure schema ${id}`, 404);
    return readJson<StructureSchema>(p);
  }

  /** An image in the library: its path, checked to stay inside. */
  imagePath(rel: string): string {
    const p = safeJoin(this.paths.library, rel);
    if (!IMAGE_TYPES[extname(p).toLowerCase()]) throw new StudioError(`not an image type: ${rel}`, 403);
    return p;
  }

  /** Save an uploaded image under `dir/name` in the library (never overwriting another one). */
  saveImage(name: string, bytes: Buffer, dir = 'added'): { file: string; width: number; height: number; sha256: string } {
    const ext = extname(name).toLowerCase();
    if (!IMAGE_TYPES[ext]) throw new StudioError(`images only (png, jpg, webp): ${name}`);
    const size = imageSize(bytes);
    if (!size) throw new StudioError(`not a readable png, jpg or webp: ${name}`);
    const stem = name.slice(0, name.length - ext.length).replace(/[^A-Za-z0-9_.-]+/g, '_').replace(/^[._-]+/, '').slice(0, 80) || 'image';
    checkId(dir.replace(/\//g, '_'), 'folder');
    const hash = sha256(bytes);
    let file = `${dir}/${stem}${ext}`;
    for (let k = 2; existsSync(this.imagePath(file)); k++) {
      if (sha256(readFileSync(this.imagePath(file))) === hash) break;
      file = `${dir}/${stem}-${k}${ext}`;
    }
    const p = this.imagePath(file);
    mkdirSync(dirname(p), { recursive: true });
    writeFileSync(p, bytes);
    return { file, ...size, sha256: hash };
  }

  imageExists(rel: string): boolean {
    try {
      return statSync(this.imagePath(rel)).isFile();
    } catch {
      return false;
    }
  }
}
