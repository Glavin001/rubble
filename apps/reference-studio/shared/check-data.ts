// Checks a workspace's committable data (shots/, frames/, structures/, scenarios/): points,
// text and numbers only, no image or pixel data, every frame consistent with the structure
// schemas. library/ and recordings/ (never committed) are not checked. `npm run check-data`.

import type { Frame, Shot, StructureSchema } from './model.ts';
import { DIRS, IMAGE_TYPES, extOf } from './store.ts';
import type { Workspace } from './workspace.ts';

/** A run of base64 this long is image data, not a point or a note. */
const BASE64_RUN = /[A-Za-z0-9+/]{200,}={0,2}/;
const BINARY = /[\x00-\x08\x0e-\x1f]/;
const MEDIA = ['.gif', '.bmp', '.tif', '.tiff', '.avif', '.heic', '.mp4', '.webm'];

async function walk(ws: Workspace, dir: string): Promise<string[]> {
  const out = (await ws.list(dir)).map((f) => (dir ? `${dir}/${f}` : f));
  for (const d of await ws.dirs(dir)) {
    if (!dir && (d === DIRS.library || d === DIRS.recordings)) continue;
    out.push(...(await walk(ws, dir ? `${dir}/${d}` : d)));
  }
  return out;
}

export async function checkData(ws: Workspace): Promise<string[]> {
  const problems: string[] = [];
  const say = (p: string, m: string) => problems.push(`${p}: ${m}`);
  const files = (await walk(ws, '')).filter((p) => !p.startsWith('.') && p !== 'README.md');
  const docs = new Map<string, unknown>();
  const dec = new TextDecoder();
  for (const p of files) {
    const ext = extOf(p);
    if (IMAGE_TYPES[ext] || MEDIA.includes(ext)) {
      say(p, 'an image or video file: images belong in library/ (never committed)');
      continue;
    }
    if (ext !== '.json') {
      say(p, 'only .json files belong here');
      continue;
    }
    const b = (await ws.read(p))!;
    const text = dec.decode(b);
    if (BINARY.test(text)) say(p, 'binary content');
    if (/data:[a-z]+\/[a-z0-9.+-]+;base64,/i.test(text)) say(p, 'a data: URI (embedded image data)');
    if (BASE64_RUN.test(text)) say(p, 'a long base64 run (embedded image data?)');
    if (b.length > 512 * 1024) say(p, 'over 512 KB: points and text are much smaller');
    try {
      docs.set(p, JSON.parse(text));
    } catch (e) {
      say(p, `not JSON: ${(e as Error).message}`);
    }
  }
  const read = <T>(sub: string): [string, T][] => [...docs].filter(([p]) => p.startsWith(sub + '/')).map(([p, v]) => [p, v as T]);
  const schemas = new Map(read<StructureSchema>(DIRS.structures).map(([, s]) => [s.id, s]));
  const shots = new Set(read<Shot>(DIRS.shots).map(([, s]) => s.id));
  for (const [p, f] of read<Frame>(DIRS.frames)) {
    if (f.shot && !shots.has(f.shot)) say(p, `shot ${f.shot} does not exist`);
    for (const r of f.regions ?? []) {
      const [x0, y0, x1, y1] = r.crop;
      if (!(x0 >= 0 && y0 >= 0 && x1 <= 1 && y1 <= 1 && x1 > x0 && y1 > y0)) say(p, `region ${r.id}: crop ${JSON.stringify(r.crop)} is not inside the image`);
      for (const i of r.tracing.instances) {
        const sc = schemas.get(i.schema);
        if (!sc) {
          say(p, `region ${r.id}: no structure schema ${i.schema}`);
          continue;
        }
        for (const [name, v] of Object.entries(i.elements)) {
          const def = sc.elements.find((e) => e.name === name);
          if (!def) say(p, `region ${r.id}: ${i.schema} has no element ${name}`);
          else if ((def.type === 'point' && v.pts) || (def.type === 'polyline' && (v.p || v.c)) || (def.type === 'circle' && (v.p || v.pts)))
            say(p, `region ${r.id}: ${name} is a ${def.type} but holds other data`);
        }
      }
    }
  }
  return problems;
}
