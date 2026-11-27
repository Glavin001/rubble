// Checks the committed data: points and text only, no image or pixel data, and every frame
// consistent with the structure schemas. `npm run check-data` (and tests/studio.test.ts).

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { extname, join, relative } from 'node:path';
import type { Frame, Shot, StructureSchema } from '../shared/model.ts';
import { IMAGE_TYPES } from './store.ts';

/** A run of base64 this long is image data, not a point or a note. */
const BASE64_RUN = /[A-Za-z0-9+/]{200,}={0,2}/;
const BINARY = /[\x00-\x08\x0e-\x1f]/;

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}

export function checkData(dir: string): string[] {
  const problems: string[] = [];
  const files = walk(dir);
  const say = (p: string, m: string) => problems.push(`${relative(dir, p)}: ${m}`);
  for (const p of files) {
    if (IMAGE_TYPES[extname(p).toLowerCase()] || ['.gif', '.bmp', '.tif', '.tiff', '.avif', '.heic', '.mp4', '.webm'].includes(extname(p).toLowerCase())) {
      say(p, 'an image or video file: images belong in the gitignored library');
      continue;
    }
    if (extname(p) !== '.json') {
      say(p, 'only .json files belong here');
      continue;
    }
    const text = readFileSync(p, 'utf8');
    if (BINARY.test(text)) say(p, 'binary content');
    if (/data:[a-z]+\/[a-z0-9.+-]+;base64,/i.test(text)) say(p, 'a data: URI (embedded image data)');
    if (BASE64_RUN.test(text)) say(p, 'a long base64 run (embedded image data?)');
    if (statSync(p).size > 512 * 1024) say(p, 'over 512 KB: points and text are much smaller');
  }
  const read = <T>(sub: string): [string, T][] =>
    files.filter((p) => relative(dir, p).startsWith(sub + '/') && p.endsWith('.json')).map((p) => [p, JSON.parse(readFileSync(p, 'utf8')) as T]);
  const schemas = new Map(read<StructureSchema>('structures').map(([, s]) => [s.id, s]));
  const shots = new Set(read<Shot>('shots').map(([, s]) => s.id));
  for (const [p, f] of read<Frame>('frames')) {
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
