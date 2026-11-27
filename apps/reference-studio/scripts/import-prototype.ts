// One-shot importer from the tracing prototype (engine/tools/crash-reference/trace.html):
// copies the reference frames into the library (keeping the originals) and turns the old
// frame index and tracings into shots and frames.
//
//   npm run import-prototype [-- --frames DIR] [--ref GIT_REF] [--dry-run]
//
// The old files (engine/docs/crash-reference/index.json, tracings/*.json) were removed when
// they were migrated; they are read from git at --ref (default 4773e6f1b, the last commit
// that has them) when they are not on disk. Existing shots and frames are not overwritten.

import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import type { ElementValue, Frame, Instance, Region, Shot, Vis, Xy } from '../shared/model.ts';
import { ENGINE, REPO, Store, imageSize, sha256 } from '../server/store.ts';

const args = process.argv.slice(2);
const opt = (k: string, d: string) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const FRAMES = opt('--frames', join(homedir(), 'Development', 'rubble-artifacts', 'crash-reference'));
const REF = opt('--ref', '4773e6f1b');
const DRY = args.includes('--dry-run');
const OLD = 'engine/docs/crash-reference';

function old(rel: string): string | null {
  const p = join(REPO, rel);
  if (existsSync(p)) return readFileSync(p, 'utf8');
  try {
    return execFileSync('git', ['show', `${REF}:${rel}`], { cwd: REPO, encoding: 'utf8', maxBuffer: 1 << 26 });
  } catch {
    return null;
  }
}

function oldTracings(): string[] {
  const dir = join(REPO, OLD, 'tracings');
  if (existsSync(dir)) return readdirSync(dir).filter((f) => f.endsWith('.json'));
  try {
    return execFileSync('git', ['ls-tree', '--name-only', `${REF}:${OLD}/tracings`], { cwd: REPO, encoding: 'utf8' })
      .split('\n')
      .filter((f) => f.endsWith('.json'));
  } catch {
    return [];
  }
}

interface OldFrame { scenario: string; file: string | null; video_s: number; shows: string; camera: string; moment: string; look_at: string }
interface OldHalf { crop: Xy[]; lines: Record<string, Xy[]>; wheels?: Record<string, { c: Xy; rim: Xy }>; blur?: string }
interface OldTracing { frame: string; camera: string; front: 'left' | 'right'; approximate: boolean; note: string; traced_by: string; image: { width: number; height: number }; halves: Record<string, OldHalf> }

const index = JSON.parse(old(`${OLD}/index.json`) ?? 'null') as { video: string; frames: OldFrame[] } | null;
if (!index) throw new Error(`no ${OLD}/index.json on disk or at ${REF}`);
const tracings = new Map<string, OldTracing>();
for (const f of oldTracings()) {
  const t = JSON.parse(old(`${OLD}/tracings/${f}`)!) as OldTracing;
  tracings.set(t.frame, t);
}

// Rubble's registry, for the shots' titles and which scenarios exist.
let registry: { id: string; title: string }[] = [];
const bin = join(ENGINE, 'target', 'release', 'examples', 'reference_shape');
if (existsSync(bin)) {
  const r = JSON.parse(execFileSync(bin, [], { input: '{"op":"scenarios"}', encoding: 'utf8' }));
  if (r.ok) registry = r.result;
}

const store = new Store();
const SIDE: Record<string, string> = {
  hood: 'hood', fender_top: 'fender_top', wheel_arch: 'wheel_arch', bumper_front: 'bumper_front',
  windscreen: 'windscreen', roofline: 'roofline', door_front_edge: 'door_front_edge', sill: 'sill',
};
const TOP: Record<string, string> = {
  body_outline: 'outline', hood_outline: 'hood_outline', windscreen_base: 'windscreen_base',
  roof_front_edge: 'roof_front_edge', roof_outline: 'roof_outline', side_line: 'side_line',
};
const r4 = (v: number) => Math.round(v * 1e6) / 1e6;

function vis(blur: string | undefined, name: string): Vis {
  if (!blur || blur === 'none') return 'clear';
  const key = name.split('_')[0];
  return blur.toLowerCase().includes(key) ? 'blurred' : 'clear';
}

function region(id: string, h: OldHalf, t: OldTracing): Region {
  const xs = h.crop.map((p) => p[0]);
  const ys = h.crop.map((p) => p[1]);
  const crop: [number, number, number, number] = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  const toRegion = (p: Xy): Xy => [r4((p[0] - crop[0]) / (crop[2] - crop[0])), r4((p[1] - crop[1]) / (crop[3] - crop[1]))];
  const side = t.camera === 'side';
  const map = side ? SIDE : TOP;
  const elements: Record<string, ElementValue> = {};
  for (const [name, line] of Object.entries(h.lines)) {
    const to = map[name];
    if (!to) throw new Error(`${t.frame}: no element for line ${name}`);
    elements[to] = { pts: line.map(toRegion), vis: vis(h.blur, name) };
  }
  for (const [name, w] of Object.entries(h.wheels ?? {})) {
    elements[`${name}_wheel`] = { c: toRegion(w.c), rim: toRegion(w.rim), vis: vis(h.blur, 'wheel') };
  }
  // The prototype's S02 origin: the door's front edge at its lowest point, now a keypoint of its own.
  const door = elements.door_front_edge?.pts;
  if (side && door?.length) {
    const low = door.reduce((a, b) => (b[1] > a[1] ? b : a));
    elements.door_sill_corner = { p: low, vis: elements.door_front_edge.vis };
  }
  const instance: Instance = { id: 'car', schema: side ? 'vehicle-side' : 'vehicle-top', front: t.front, elements };
  const note = [t.note, h.blur && h.blur !== 'none' ? `blur: ${h.blur}` : '', side && door?.length ? 'door_sill_corner: the lowest point of door_front_edge (migrated from the prototype).' : '']
    .filter(Boolean)
    .join(' ');
  return {
    id,
    label: id === 'real' ? 'real test' : id === 'beamng' ? 'BeamNG' : id,
    source: id === 'real' || id === 'beamng' ? id : 'other',
    crop,
    tracing: { instances: [instance], traced_by: t.traced_by, approximate: t.approximate, note },
  };
}

let copied = 0;
let made = 0;
const shots = new Map<string, Shot>();
for (const fr of index.frames) {
  const sid = fr.scenario;
  if (!shots.has(sid)) {
    const reg = registry.find((s) => s.id === sid);
    shots.set(sid, {
      id: sid,
      title: reg ? `${sid}: ${reg.title}` : `${sid} (no Rubble scenario yet)`,
      source: { kind: 'video', url: index.video, note: 'Real test beside BeamNG in one video; video times are approximate.' },
      rubble: reg ? { scenario: sid, contact: 'edge', run: 0 } : null,
      before_frame: null,
      notes: '',
    });
  }
  const id = fr.file ? fr.file.split('/').pop()!.replace(/\.[^.]+$/, '') : `${sid}_${fr.video_s}s_${fr.camera.replace(/\W+/g, '-')}`;
  let image: Frame['image'] = null;
  if (fr.file) {
    const src = join(FRAMES, fr.file);
    if (existsSync(src)) {
      const bytes = readFileSync(src);
      const size = imageSize(bytes);
      if (!size) throw new Error(`${src}: unreadable image`);
      const dst = store.imagePath(fr.file);
      if (!DRY && !existsSync(dst)) {
        mkdirSync(join(dst, '..'), { recursive: true });
        copyFileSync(src, dst);
        copied++;
      }
      image = { file: fr.file, ...size, sha256: sha256(bytes) };
    } else {
      const t = tracings.get(fr.file);
      if (t) image = { file: fr.file, width: t.image.width, height: t.image.height, sha256: '' };
    }
  }
  const t = fr.file ? tracings.get(fr.file) : undefined;
  const frame: Frame = {
    id,
    shot: sid,
    image,
    moment: { label: fr.moment, after_impact_ms: null },
    time: { video_s: fr.video_s, approximate: true },
    view: fr.camera,
    notes: fr.look_at + (fr.file ? '' : ' (no image captured yet: add one)') + ` Shows: ${fr.shows}.`,
    rubble: { moment: null, camera: null },
    camera: null,
    regions: t ? Object.entries(t.halves).map(([h, v]) => region(h, v, t)) : [],
  };
  if (!DRY && !store.hasFrame(id)) {
    store.putFrame(frame);
    made++;
  }
}
for (const s of shots.values()) if (!DRY && !store.hasShot(s.id)) store.putShot(s);
console.log(`${shots.size} shots, ${made} frames written, ${copied} images copied into ${store.paths.library}, ${tracings.size} tracings migrated${DRY ? ' (dry run)' : ''}`);
