// The server tools (shared/tools.ts): each takes the tool's JSON input and returns JSON.
// They read and write the studio's files; geometry goes to Rust (rust.ts).

import { existsSync } from 'node:fs';
import { join, relative } from 'node:path';
import type { ElementValue, Frame, FrameSummary, Instance, Region, Scenario, Shot, SourceTag, Status, StructureSchema, Tracing, Vis } from '../shared/model.ts';
import { SOURCES, VIS } from '../shared/model.ts';
import { TOOLS } from '../shared/tools.ts';
import { Rust } from './rust.ts';
import { Store, StudioError, checkId } from './store.ts';

type Input = Record<string, unknown>;

const s = (v: unknown, k: string): string => {
  if (typeof v !== 'string') throw new StudioError(`${k} must be a string`);
  return v;
};
const n = (v: unknown, k: string): number => {
  if (typeof v !== 'number' || !Number.isFinite(v)) throw new StudioError(`${k} must be a number`);
  return v;
};
const unit = (v: unknown, k: string): number => {
  const x = n(v, k);
  if (x < -0.5 || x > 1.5) throw new StudioError(`${k} = ${x}: normalised coordinates are 0..1 (a little outside is allowed for parts out of frame)`);
  return Math.round(x * 1e6) / 1e6;
};
const vec3 = (v: unknown, k: string): [number, number, number] => {
  if (!Array.isArray(v) || v.length !== 3) throw new StudioError(`${k} must be [x, y, z]`);
  return [n(v[0], k), n(v[1], k), n(v[2], k)];
};

export function emptyTracing(): Tracing {
  return { instances: [], traced_by: '', approximate: false, note: '' };
}

function hasData(v: ElementValue | undefined): boolean {
  return !!v && (!!v.p || (!!v.pts && v.pts.length > 0) || !!v.c);
}

export function tracedCount(r: Region): number {
  return r.tracing.instances.reduce((k, i) => k + Object.values(i.elements).filter(hasData).length, 0);
}

/** Frames in time order: video time, then time after impact, then a "before"-like label
 * first (before, setup, first contact), then id. */
export function byTime(a: Frame, b: Frame): number {
  const early = (f: Frame) => (/^(before|setup|(first )?contact|rest before)$/i.test(f.moment.label.trim()) ? 0 : 1);
  return (
    (a.time?.video_s ?? Infinity) - (b.time?.video_s ?? Infinity) ||
    (a.moment.after_impact_ms ?? Infinity) - (b.moment.after_impact_ms ?? Infinity) ||
    early(a) - early(b) ||
    a.id.localeCompare(b.id)
  ) || 0;
}

function slug(t: string): string {
  return t.replace(/[^A-Za-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'shot';
}

export class Ops {
  store: Store;
  rust: Rust;
  constructor(store: Store, rust: Rust) {
    this.store = store;
    this.rust = rust;
  }

  // ------------------------------------------------------------------ lookups

  region(f: Frame, id: unknown): Region {
    const r = f.regions.find((x) => x.id === id);
    if (!r) throw new StudioError(`frame ${f.id} has no region ${JSON.stringify(id)} (regions: ${f.regions.map((x) => x.id).join(', ') || 'none: mark_region first'})`, 404);
    return r;
  }

  instance(r: Region, id: unknown): Instance {
    const i = id ? r.tracing.instances.find((x) => x.id === id) : r.tracing.instances[0];
    if (!i) throw new StudioError(`region ${r.id} has no structure yet: set_region_schema first`, 404);
    return i;
  }

  element(i: Instance, name: unknown): { def: StructureSchema['elements'][number]; v: ElementValue } {
    const sc = this.store.structure(i.schema);
    const def = sc.elements.find((e) => e.name === name);
    if (!def) throw new StudioError(`${i.schema} has no element ${JSON.stringify(name)} (elements: ${sc.elements.map((e) => e.name).join(', ')})`, 404);
    i.elements[def.name] ??= { vis: 'clear' };
    return { def, v: i.elements[def.name] };
  }

  /** The frame compared against as "before": the frame's own, the shot's, or the earliest
   * frame of the same shot and view (by video time, then id) with a traced region. */
  beforeOf(f: Frame, frames: Frame[] = this.store.frames()): Frame | null {
    const shot = f.shot && this.store.hasShot(f.shot) ? this.store.shot(f.shot) : null;
    const named = f.before_frame || (shot?.before_frame && frames.find((x) => x.id === shot.before_frame)?.view === f.view ? shot.before_frame : null);
    if (named) return frames.find((x) => x.id === named) ?? null;
    const same = frames.filter((x) => x.shot === f.shot && f.shot && x.view === f.view && x.regions.some((r) => tracedCount(r) > 0));
    same.sort(byTime);
    return same[0] ?? null;
  }

  status(f: Frame, frames: Frame[], shots: Map<string, Shot>): Status[] {
    if (!f.image || !this.store.imageExists(f.image.file)) return ['no image'];
    const traced = f.regions.some((r) => tracedCount(r) > 0);
    const out: Status[] = [traced ? 'traced' : 'untraced'];
    const b = this.beforeOf(f, frames);
    if (traced && (f.regions.filter((r) => tracedCount(r) > 0).length > 1 || (b && b.id !== f.id))) out.push('paired');
    if (f.shot && shots.get(f.shot)?.rubble) out.push('linked');
    return out;
  }

  summaries(): FrameSummary[] {
    const frames = this.store.frames();
    const shots = new Map(this.store.shots().map((x) => [x.id, x]));
    return frames.map((f) => ({
      id: f.id,
      shot: f.shot,
      image: f.image,
      thumb: f.image ? `/library/${f.image.file}` : null,
      moment: f.moment.label,
      view: f.view,
      regions: f.regions.length,
      traced: f.regions.reduce((k, r) => k + tracedCount(r), 0),
      status: this.status(f, frames, shots),
    }));
  }

  // ------------------------------------------------------------------ tools

  async call(name: string, input: Input): Promise<unknown> {
    const t = TOOLS.find((x) => x.name === name);
    if (!t) throw new StudioError(`no tool ${name}`, 404);
    if (t.side !== 'server') throw new StudioError(`${name} acts on the open page: call it through window.studio or WebMCP`, 400);
    const fn = (this as unknown as Record<string, (i: Input) => unknown>)[`t_${name}`];
    return fn.call(this, input ?? {});
  }

  t_list_shots() {
    const sums = this.summaries();
    return this.store.shots().map((x) => {
      const mine = sums.filter((f) => f.shot === x.id);
      const has = (st: Status) => mine.filter((f) => f.status.includes(st)).length;
      return { ...x, frames: mine.length, traced: has('traced'), paired: has('paired'), linked: !!x.rubble, no_image: has('no image') };
    });
  }

  t_list_frames(i: Input) {
    let out = this.summaries();
    if (typeof i.shot === 'string') out = out.filter((f) => (i.shot === '' ? !f.shot : f.shot === i.shot));
    if (typeof i.status === 'string') out = out.filter((f) => f.status.includes(i.status as Status));
    return out;
  }

  t_list_untraced() {
    return this.summaries().filter((f) => f.status.includes('untraced'));
  }

  t_get_frame(i: Input) {
    const f = this.store.frame(s(i.frame, 'frame'));
    const b = this.beforeOf(f);
    return { ...f, before_resolved: b?.id ?? null, image_url: f.image ? `/library/${f.image.file}` : null, image_on_disk: !!f.image && this.store.imageExists(f.image.file) };
  }

  t_get_tracing(i: Input) {
    const f = this.store.frame(s(i.frame, 'frame'));
    const regions = i.region ? [this.region(f, i.region)] : f.regions;
    return regions.map((r) => ({ region: r.id, source: r.source, crop: r.crop, ...r.tracing }));
  }

  t_list_schemas() {
    return this.store.structures();
  }

  async t_list_scenarios() {
    return this.rust.scenarios<Scenario>();
  }

  t_create_shot(i: Input) {
    const title = s(i.title, 'title');
    let id = i.id ? checkId(i.id, 'shot') : slug(title);
    if (!i.id) for (let k = 2; this.store.hasShot(id); k++) id = `${slug(title)}-${k}`;
    if (this.store.hasShot(id)) throw new StudioError(`shot ${id} exists`, 409);
    const shot: Shot = {
      id,
      title,
      source: i.video_url ? { kind: 'video', url: s(i.video_url, 'video_url') } : { kind: 'screenshots' },
      rubble: null,
      before_frame: null,
      notes: typeof i.notes === 'string' ? i.notes : '',
    };
    return this.store.putShot(shot);
  }

  t_update_shot(i: Input) {
    const x = this.store.shot(s(i.shot, 'shot'));
    if (typeof i.title === 'string') x.title = i.title;
    if (typeof i.notes === 'string') x.notes = i.notes;
    if (typeof i.video_url === 'string') x.source = i.video_url ? { kind: 'video', url: i.video_url } : { kind: 'screenshots' };
    if (typeof i.before_frame === 'string') x.before_frame = i.before_frame ? (this.store.frame(i.before_frame), i.before_frame) : null;
    return this.store.putShot(x);
  }

  addImage(name: string, bytes: Buffer, i: Input): Frame {
    const shot = typeof i.shot === 'string' && i.shot ? (this.store.shot(i.shot), i.shot) : null;
    const img = this.store.saveImage(name, bytes, shot ?? 'added');
    let id = img.file.split('/').pop()!.replace(/\.[^.]+$/, '');
    const existing = this.store.frames().find((f) => f.image?.sha256 === img.sha256);
    if (existing) return existing;
    for (let k = 2; this.store.hasFrame(id); k++) id = `${img.file.split('/').pop()!.replace(/\.[^.]+$/, '')}-${k}`;
    const f: Frame = {
      id,
      shot,
      image: img,
      moment: { label: typeof i.moment === 'string' ? i.moment : '', after_impact_ms: null },
      time: null,
      view: typeof i.view === 'string' ? i.view : 'side',
      notes: '',
      rubble: { moment: null, camera: null },
      camera: null,
      regions: [],
    };
    return this.store.putFrame(f);
  }

  t_add_image(i: Input) {
    const b64 = s(i.data_base64, 'data_base64').replace(/^data:[^,]*,/, '');
    return this.addImage(s(i.name, 'name'), Buffer.from(b64, 'base64'), i);
  }

  t_assign_frame(i: Input) {
    const f = this.store.frame(s(i.frame, 'frame'));
    const shot = s(i.shot, 'shot');
    if (shot) this.store.shot(shot);
    f.shot = shot || null;
    return this.store.putFrame(f);
  }

  t_set_frame_info(i: Input) {
    const f = this.store.frame(s(i.frame, 'frame'));
    if (typeof i.moment === 'string') f.moment.label = i.moment;
    if (i.after_impact_ms === null || typeof i.after_impact_ms === 'number') f.moment.after_impact_ms = i.after_impact_ms as number | null;
    if (typeof i.view === 'string') f.view = i.view;
    if (typeof i.notes === 'string') f.notes = i.notes;
    if (typeof i.video_s === 'number') f.time = { video_s: i.video_s, approximate: f.time?.approximate ?? false };
    if (typeof i.before_frame === 'string') {
      if (i.before_frame) this.store.frame(i.before_frame);
      f.before_frame = i.before_frame || null;
    }
    return this.store.putFrame(f);
  }

  t_mark_region(i: Input) {
    const f = this.store.frame(s(i.frame, 'frame'));
    const source = s(i.source, 'source') as SourceTag;
    if (!SOURCES.includes(source)) throw new StudioError(`source is one of ${SOURCES.join(', ')}`);
    const has = ['x0', 'y0', 'x1', 'y1'].every((k) => typeof i[k] === 'number');
    let crop: Region['crop'] = [0, 0, 1, 1];
    if (has) {
      const [x0, y0, x1, y1] = ['x0', 'y0', 'x1', 'y1'].map((k) => Math.min(1, Math.max(0, unit(i[k], k))));
      crop = [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)];
      if (crop[2] - crop[0] < 0.01 || crop[3] - crop[1] < 0.01) throw new StudioError('a region must be at least 1 % of the image across');
    }
    let id = typeof i.region === 'string' && i.region ? checkId(i.region, 'region') : '';
    let r = id ? f.regions.find((x) => x.id === id) : undefined;
    if (!r) {
      if (!id) {
        id = source;
        for (let k = 2; f.regions.some((x) => x.id === id); k++) id = `${source}-${k}`;
      }
      r = { id, label: '', source, crop, tracing: emptyTracing() };
      f.regions.push(r);
    }
    r.source = source;
    if (has || !r.crop) r.crop = crop;
    r.label = typeof i.label === 'string' ? i.label : r.label || { real: 'real test', beamng: 'BeamNG', rubble: 'Rubble', other: 'other' }[source];
    this.store.putFrame(f);
    return r;
  }

  t_remove_region(i: Input) {
    const f = this.store.frame(s(i.frame, 'frame'));
    this.region(f, i.region);
    f.regions = f.regions.filter((r) => r.id !== i.region);
    return this.store.putFrame(f);
  }

  t_set_region_schema(i: Input) {
    const f = this.store.frame(s(i.frame, 'frame'));
    const r = this.region(f, i.region);
    const sc = this.store.structure(s(i.schema, 'schema'));
    const front = i.front === 'left' || i.front === 'right' ? i.front : sc.orientation.default;
    let inst = i.instance ? r.tracing.instances.find((x) => x.id === i.instance) : r.tracing.instances[0];
    if (!inst) {
      inst = { id: typeof i.instance === 'string' ? checkId(i.instance, 'instance') : 'car', schema: sc.id, front, elements: {} };
      if (sc.kind !== 'vehicle' && !i.instance) inst.id = 'structure';
      r.tracing.instances.push(inst);
    }
    inst.schema = sc.id;
    inst.front = front;
    for (const k of Object.keys(inst.elements)) if (!sc.elements.some((e) => e.name === k)) delete inst.elements[k];
    this.store.putFrame(f);
    return inst;
  }

  private edit(i: Input, fn: (v: ElementValue, kind: string, inst: Instance) => void) {
    const f = this.store.frame(s(i.frame, 'frame'));
    const r = this.region(f, i.region);
    const inst = this.instance(r, i.instance);
    const { def, v } = this.element(inst, i.element);
    fn(v, def.type, inst);
    r.tracing.updated = new Date().toISOString();
    this.store.putFrame(f);
    return { region: r.id, instance: inst.id, element: def.name, type: def.type, value: inst.elements[def.name] ?? null };
  }

  t_set_keypoint(i: Input) {
    return this.edit(i, (v, kind) => {
      const p: [number, number] = [unit(i.x, 'x'), unit(i.y, 'y')];
      if (kind === 'point') v.p = p;
      else if (kind === 'circle') {
        const part = i.part ?? (v.c ? 'rim' : 'c');
        if (part !== 'c' && part !== 'rim') throw new StudioError('part is "c" or "rim"');
        if (part === 'c' && v.c && v.rim) v.rim = [v.rim[0] + p[0] - v.c[0], v.rim[1] + p[1] - v.c[1]];
        v[part] = p;
      } else throw new StudioError(`${String(i.element)} is a polyline: use add_polyline_point or move_polyline_point`);
    });
  }

  t_add_polyline_point(i: Input) {
    return this.edit(i, (v, kind) => {
      if (kind !== 'polyline') throw new StudioError(`${String(i.element)} is a ${kind}: use set_keypoint`);
      v.pts ??= [];
      const p: [number, number] = [unit(i.x, 'x'), unit(i.y, 'y')];
      const at = typeof i.index === 'number' ? Math.max(0, Math.min(v.pts.length, Math.round(i.index))) : v.pts.length;
      v.pts.splice(at, 0, p);
    });
  }

  t_move_polyline_point(i: Input) {
    return this.edit(i, (v, kind) => {
      const k = Math.round(n(i.index, 'index'));
      if (kind !== 'polyline' || !v.pts || k < 0 || k >= v.pts.length) throw new StudioError(`${String(i.element)} has no point ${k}`, 404);
      v.pts[k] = [unit(i.x, 'x'), unit(i.y, 'y')];
    });
  }

  t_remove_point(i: Input) {
    return this.edit(i, (v, kind, inst) => {
      if (typeof i.index === 'number' && kind === 'polyline') {
        const k = Math.round(i.index);
        if (!v.pts || k < 0 || k >= v.pts.length) throw new StudioError(`${String(i.element)} has no point ${k}`, 404);
        v.pts.splice(k, 1);
      } else delete inst.elements[String(i.element)];
    });
  }

  t_set_visibility(i: Input) {
    return this.edit(i, (v) => {
      if (!VIS.includes(i.vis as Vis)) throw new StudioError(`vis is one of ${VIS.join(', ')}`);
      v.vis = i.vis as Vis;
    });
  }

  t_set_tracing_info(i: Input) {
    const f = this.store.frame(s(i.frame, 'frame'));
    const r = this.region(f, i.region);
    if (typeof i.traced_by === 'string') r.tracing.traced_by = i.traced_by;
    if (typeof i.approximate === 'boolean') r.tracing.approximate = i.approximate;
    if (typeof i.note === 'string') r.tracing.note = i.note;
    this.store.putFrame(f);
    return r.tracing;
  }

  t_copy_tracing_from_previous(i: Input) {
    const frames = this.store.frames();
    const f = frames.find((x) => x.id === i.frame) ?? this.store.frame(s(i.frame, 'frame'));
    const r = this.region(f, i.region);
    const order = frames
      .filter((x) => x.shot && x.shot === f.shot)
      .sort(byTime);
    const at = order.findIndex((x) => x.id === f.id);
    for (let k = at - 1; k >= 0; k--) {
      const src = order[k].regions.find((x) => (x.id === r.id || x.source === r.source) && tracedCount(x) > 0);
      if (src) {
        r.tracing = { ...structuredClone(src.tracing), note: `copied from ${order[k].id} (${src.id}); adjust every point`, approximate: true };
        this.store.putFrame(f);
        return { copied_from: order[k].id, region: src.id, tracing: r.tracing };
      }
    }
    throw new StudioError(`no earlier frame of shot ${f.shot ?? '(none)'} has a traced ${r.source} region`, 404);
  }

  async t_link_scenario(i: Input) {
    const x = this.store.shot(s(i.shot, 'shot'));
    const id = s(i.scenario, 'scenario');
    if (!id) x.rubble = null;
    else {
      const sc = (await this.rust.scenarios<Scenario>()).find((y) => y.id.toLowerCase() === id.toLowerCase());
      if (!sc) throw new StudioError(`no Rubble scenario ${id}`, 404);
      x.rubble = { scenario: sc.id, contact: typeof i.contact === 'string' ? i.contact : x.rubble?.contact ?? 'edge', run: typeof i.run === 'number' ? i.run : 0 };
    }
    return this.store.putShot(x);
  }

  t_link_frame(i: Input) {
    const f = this.store.frame(s(i.frame, 'frame'));
    if (typeof i.moment === 'string') f.rubble.moment = i.moment || null;
    if (typeof i.camera === 'string') f.rubble.camera = i.camera || null;
    return this.store.putFrame(f);
  }

  t_set_frame_camera(i: Input) {
    const f = this.store.frame(s(i.frame, 'frame'));
    if (i.clear) f.camera = null;
    else {
      const prev = f.camera;
      const region = typeof i.region === 'string' ? i.region || null : prev?.region ?? null;
      if (region) this.region(f, region);
      const fov = typeof i.fov_y_deg === 'number' ? i.fov_y_deg : prev?.fov_y_deg ?? 60;
      if (!(fov > 1 && fov < 170)) throw new StudioError('fov_y_deg is between 1 and 170');
      f.camera = {
        eye: i.eye !== undefined ? vec3(i.eye, 'eye') : prev?.eye ?? (() => { throw new StudioError('eye is needed for a new camera'); })(),
        target: i.target !== undefined ? vec3(i.target, 'target') : prev?.target ?? (() => { throw new StudioError('target is needed for a new camera'); })(),
        up: i.up !== undefined ? vec3(i.up, 'up') : prev?.up ?? [0, 1, 0],
        fov_y_deg: fov,
        region,
      };
    }
    return this.store.putFrame(f);
  }

  t_put_frame(i: Input) {
    const f = i.frame as Frame;
    if (!f || typeof f !== 'object' || !Array.isArray(f.regions)) throw new StudioError('frame must be a Frame document');
    checkId(f.id, 'frame');
    const { before_resolved: _a, image_url: _b, image_on_disk: _c, ...clean } = f as Frame & Record<string, unknown>;
    return this.store.putFrame(clean as Frame);
  }

  // ------------------------------------------------------------------ Rubble

  /** Pixel size of the image a camera or region covers. */
  private regionSize(f: Frame, regionId: string | null): [number, number] {
    if (!f.image) throw new StudioError(`frame ${f.id} has no image`);
    const r = regionId ? f.regions.find((x) => x.id === regionId) : null;
    const c = r?.crop ?? [0, 0, 1, 1];
    return [(c[2] - c[0]) * f.image.width, (c[3] - c[1]) * f.image.height];
  }

  private frameCameras(shot: Shot): unknown[] {
    return this.store
      .frames()
      .filter((x) => x.shot === shot.id && x.camera && x.image)
      .sort((a, b) => a.id.localeCompare(b.id))
      .map((x) => {
        const [w, h] = this.regionSize(x, x.camera!.region);
        return { name: `frame_${x.id}`, eye: x.camera!.eye, target: x.camera!.target, up: x.camera!.up, fov_y_deg: x.camera!.fov_y_deg, width: Math.round(w), height: Math.round(h) };
      });
  }

  /** The scenario, moment and camera a frame is compared with, and the outlines folder. */
  async rubbleLink(f: Frame) {
    const shot = f.shot ? this.store.shot(f.shot) : null;
    if (!shot?.rubble) throw new StudioError(`frame ${f.id}: its shot is not linked to a Rubble scenario (link_scenario)`, 409);
    const { scenario, contact } = shot.rubble;
    const pick = await this.rust.call<{ moment: string; camera: string }>('pick', { scenario, moment: f.moment.label || 'peak', view: f.view || 'side' });
    const moment = f.rubble.moment ?? pick.moment;
    const camera = f.camera ? `frame_${f.id}` : f.rubble.camera ?? pick.camera;
    const dir = await this.rust.outlines(scenario, contact, this.frameCameras(shot));
    return { scenario, contact, moment, camera, dir, auto: { moment: f.rubble.moment === null, camera: !f.camera && f.rubble.camera === null }, own_camera: !!f.camera };
  }

  private schemaOf(f: Frame, want?: unknown): StructureSchema {
    if (typeof want === 'string' && want) return this.store.structure(want);
    const inst = f.regions.flatMap((r) => r.tracing.instances)[0];
    return this.store.structure(inst?.schema ?? (f.view === 'top' ? 'vehicle-top' : f.view.includes('three') ? 'vehicle-three-quarter' : 'vehicle-side'));
  }

  async t_get_rubble_outline(i: Input) {
    const f = this.store.frame(s(i.frame, 'frame'));
    const link = await this.rubbleLink(f);
    const sc = this.schemaOf(f, i.schema);
    const inst = await this.rust.call<Record<string, unknown>>('rubble_instance', { schema: sc, outlines: join(link.dir, 'outlines.json'), moment: link.moment, camera: link.camera });
    const still = join(link.dir, `${link.moment}-${link.camera}.png`);
    return {
      ...link,
      dir: relative(this.store.paths.cache, link.dir),
      schema: sc.id,
      instance: inst,
      still: existsSync(still) ? `/cache/${relative(this.store.paths.cache, still)}` : null,
      region: f.camera?.region ?? null,
    };
  }

  async t_get_comparison(i: Input) {
    const frames = this.store.frames();
    const f = frames.find((x) => x.id === i.frame) ?? this.store.frame(s(i.frame, 'frame'));
    const before = this.beforeOf(f, frames) ?? f;
    const sources: unknown[] = [];
    const used: { name: string; label: string; source: SourceTag; region: string; before_region: string }[] = [];
    let schema: StructureSchema | null = null;
    for (const r of f.regions) {
      const inst = r.tracing.instances[0];
      if (!inst || tracedCount(r) === 0) continue;
      schema ??= this.store.structure(inst.schema);
      if (inst.schema !== schema.id) continue;
      const br = before.regions.find((x) => x.id === r.id && x.tracing.instances[0]?.schema === schema!.id) ??
        before.regions.find((x) => x.source === r.source && x.tracing.instances[0]?.schema === schema!.id);
      if (!br) continue;
      const shape = (fr: Frame, rg: Region) => {
        const [w, h] = this.regionSize(fr, rg.id);
        return { tracing: { instance: rg.tracing.instances[0], width: w, height: h } };
      };
      sources.push({ name: r.id, before: shape(before, br), now: shape(f, r) });
      used.push({ name: r.id, label: r.label, source: r.source, region: r.id, before_region: br.id });
    }
    if (!schema) throw new StudioError(`frame ${f.id} has no traced region to compare`, 409);
    let rubble: unknown = null;
    const shot = f.shot ? this.store.shot(f.shot) : null;
    if (shot?.rubble) {
      try {
        const now = await this.rubbleLink(f);
        const was = await this.rubbleLink(before);
        const view = (l: typeof now) => (schema!.rubble_view === 'car_frame' ? 'car_frame' : `camera:${l.camera}`);
        sources.push({
          name: 'rubble',
          before: { rubble: { outlines: join(was.dir, 'outlines.json'), moment: was.moment, view: view(was) } },
          now: { rubble: { outlines: join(now.dir, 'outlines.json'), moment: now.moment, view: view(now) } },
        });
        used.push({ name: 'rubble', label: `Rubble ${now.scenario} ${now.moment}`, source: 'rubble', region: '', before_region: '' });
        rubble = { scenario: now.scenario, contact: now.contact, moment: now.moment, before_moment: was.moment, camera: now.camera, view: view(now) };
      } catch (e) {
        rubble = { error: (e as Error).message };
      }
    }
    const result = await this.rust.call<Record<string, unknown>>('compare', { schema, sources });
    // result: { frame (the common frame used: origin and unit), shapes, rows }
    return { frame_id: f.id, before_frame: before.id, sources: used, rubble, ...result, schema: schema.id };
  }
}
