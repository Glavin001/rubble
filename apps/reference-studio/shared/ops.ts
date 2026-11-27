// The data tools (shared/tools.ts, side "data"): each takes the tool's JSON input and returns
// JSON. They read and write the workspace's documents (shared/store.ts); anything computed
// (the registry, the moment/camera rule, Rubble's outline, the comparison, design ranking,
// scenario checks) is Rust in wasm, reached through `EngineApi` (src/engine.ts in the page).

import type { Camera, CompareSource, Contact, Frame, FrameCamera, FrameSummary, Instance, Moment, RecordingStatus, Region, Registry, RubbleSpec, ScenarioDesc, SceneInfo, Shot, SourceTag, Status, StructureSchema, Tracing, Vis, ElementValue } from './model.ts';
import { SOURCES, VIS, hasData } from './model.ts';
import { TOOLS } from './tools.ts';
import { Store, StudioError, checkId } from './store.ts';

type Input = Record<string, unknown>;

/** What the data tools need computed: Rust in wasm (src/engine.ts). */
export interface EngineApi {
  registry(data: ScenarioDesc[]): Registry;
  pick(moments: Moment[], cameras: Camera[], scenario: string, label: string, view: string): { moment: string | null; camera: string | null };
  validId(id: string): void;
  checkDesc(d: ScenarioDesc): void;
  /** A built-in scenario as data (run 0). */
  exportBuiltin(id: string): Promise<ScenarioDesc>;
  /** The scenario placed in the page's scene and its recording ready (the workspace's cached
   * one, or simulated now on the page's GPU). */
  recorded(scenario: string, contact: Contact): Promise<{ info: SceneInfo; status: RecordingStatus }>;
  outline(schema: StructureSchema, spec: RubbleSpec): Record<string, unknown>;
  compare(schema: StructureSchema, sources: CompareSource[]): Record<string, unknown>;
  /** The generated designs ranked by proportion against a traced instance (Rust). */
  rankDesigns(schema: StructureSchema, instance: Instance, width: number, height: number): Promise<unknown>;
}

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
/** Metres and degrees as stored: to the millimetre and the hundredth of a degree. */
const mm = (x: number) => Math.round(x * 1000) / 1000;
const v3mm = (v: [number, number, number]): [number, number, number] => [mm(v[0]), mm(v[1]), mm(v[2])];

export function emptyTracing(): Tracing {
  return { instances: [], traced_by: '', approximate: false, note: '' };
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

const b64 = (t: string): Uint8Array => {
  const bin = atob(t);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
};

/** A new data scenario's default: the generated sedan at 50 km/h toward +Z, a rigid wall
 * ahead (its face 3.35 m from the car's origin: about 0.17 s of travel from the sedan's nose). */
export function defaultScenario(id: string, title: string): ScenarioDesc {
  return {
    id,
    title,
    notes: '',
    bodies: [{ name: 'car', design: 'sedan', at: [0, 0, 0], yaw_deg: 0, speed_kmh: 50, carrier: false }],
    boxes: [{ name: 'wall', centre: [0, 1, 3.6], half: [2.5, 1, 0.25], yaw_deg: 0 }],
    cameras: [{ name: 'side', eye: [7, 1.2, 1.2], target: [0, 0.6, 1.2] }],
    moments: [
      { name: 'before', after_impact_ms: -10 },
      { name: 'peak', after_impact_ms: 70 },
      { name: 'rest', after_impact_ms: 450 },
    ],
    secs_after_impact: 0.5,
    contact: 'edge',
  };
}

export class Ops {
  store: Store;
  engine: EngineApi;
  constructor(store: Store, engine: EngineApi) {
    this.store = store;
    this.engine = engine;
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
    if (f.pin && f.camera) out.push('pinned');
    return out;
  }

  summaries(thumb: (file: string) => string | null = () => null): FrameSummary[] {
    const frames = this.store.frames();
    const shots = new Map(this.store.shots().map((x) => [x.id, x]));
    return frames.map((f) => ({
      id: f.id,
      shot: f.shot,
      image: f.image,
      thumb: f.image ? thumb(f.image.file) : null,
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
    if (t.side !== 'data') throw new StudioError(`${name} acts on the open page: call it through window.studio or WebMCP`, 400);
    const fn = (this as unknown as Record<string, (i: Input) => unknown>)[`t_${name}`];
    return fn.call(this, input ?? {});
  }

  t_list_shots() {
    const sums = this.summaries();
    return this.store.shots().map((x) => {
      const mine = sums.filter((f) => f.shot === x.id);
      const has = (st: Status) => mine.filter((f) => f.status.includes(st)).length;
      return { ...x, frames: mine.length, traced: has('traced'), paired: has('paired'), linked: !!x.rubble, no_image: has('no image'), pinned: has('pinned') };
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
    return { ...f, before_resolved: b?.id ?? null, image_on_disk: !!f.image && this.store.imageExists(f.image.file) };
  }

  t_get_tracing(i: Input) {
    const f = this.store.frame(s(i.frame, 'frame'));
    const regions = i.region ? [this.region(f, i.region)] : f.regions;
    return regions.map((r) => ({ region: r.id, source: r.source, crop: r.crop, ...r.tracing }));
  }

  t_list_schemas() {
    return this.store.structures();
  }

  registry(): Registry {
    return this.engine.registry(this.store.scenarios());
  }

  t_list_scenarios() {
    return this.registry().scenarios.map(({ desc: _d, ...x }) => x);
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

  async addImage(name: string, bytes: Uint8Array, i: Input): Promise<Frame> {
    const shot = typeof i.shot === 'string' && i.shot ? (this.store.shot(i.shot), i.shot) : null;
    const img = await this.store.saveImage(name, bytes, shot ?? 'added');
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
      pin: null,
      regions: [],
    };
    return this.store.putFrame(f);
  }

  t_add_image(i: Input) {
    return this.addImage(s(i.name, 'name'), b64(s(i.data_base64, 'data_base64').replace(/^data:[^,]*,/, '')), i);
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
    const order = frames.filter((x) => x.shot && x.shot === f.shot).sort(byTime);
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

  t_link_scenario(i: Input) {
    const x = this.store.shot(s(i.shot, 'shot'));
    const id = s(i.scenario, 'scenario');
    if (!id) x.rubble = null;
    else {
      const sc = this.registry().scenarios.find((y) => y.id.toLowerCase() === id.toLowerCase());
      if (!sc) throw new StudioError(`no Rubble scenario ${id} (built-in or in scenarios/)`, 404);
      x.rubble = { scenario: sc.id, contact: typeof i.contact === 'string' ? i.contact : x.rubble?.contact ?? sc.contact, run: typeof i.run === 'number' ? i.run : 0 };
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
      const need = (k: string): never => {
        throw new StudioError(`${k} is needed for a new camera`);
      };
      f.camera = {
        eye: i.eye !== undefined ? v3mm(vec3(i.eye, 'eye')) : prev?.eye ?? need('eye'),
        target: i.target !== undefined ? v3mm(vec3(i.target, 'target')) : prev?.target ?? need('target'),
        up: i.up !== undefined ? vec3(i.up, 'up').map((x) => Math.round(x * 1e4) / 1e4) as [number, number, number] : prev?.up ?? [0, 1, 0],
        fov_y_deg: Math.round(fov * 100) / 100,
        region,
      };
    }
    return this.store.putFrame(f);
  }

  t_copy_camera(i: Input) {
    const from = this.store.frame(s(i.from, 'from'));
    if (!from.camera) throw new StudioError(`frame ${from.id} has no camera to copy`, 409);
    const f = this.store.frame(s(i.to, 'to'));
    const region = from.camera.region && f.regions.some((r) => r.id === from.camera!.region) ? from.camera.region : null;
    f.camera = { ...structuredClone(from.camera), region };
    return this.store.putFrame(f);
  }

  /** Pin an image at a time of the recorded run (its camera is `Frame.camera`). */
  t_set_frame_pin(i: Input) {
    const f = this.store.frame(s(i.frame, 'frame'));
    if (i.clear) f.pin = null;
    else f.pin = { at_ms: Math.round(n(i.at_ms, 'at_ms') * 100) / 100, recording: s(i.recording, 'recording'), label: typeof i.label === 'string' ? i.label : null, pinned: new Date().toISOString() };
    return this.store.putFrame(f);
  }

  t_list_pins(i: Input) {
    return this.store
      .frames()
      .filter((f) => typeof i.shot !== 'string' || !i.shot || f.shot === i.shot)
      .map((f) => ({ frame: f.id, shot: f.shot, view: f.view, moment: f.moment.label, pinned: !!(f.pin && f.camera), at_ms: f.pin?.at_ms ?? null, recording: f.pin?.recording ?? null, label: f.pin?.label ?? null, camera: f.camera }));
  }

  t_put_frame(i: Input) {
    const f = i.frame as Frame;
    if (!f || typeof f !== 'object' || !Array.isArray(f.regions)) throw new StudioError('frame must be a Frame document');
    checkId(f.id, 'frame');
    const { before_resolved: _a, image_on_disk: _c, ...clean } = f as Frame & Record<string, unknown>;
    return this.store.putFrame(clean as Frame);
  }

  // ------------------------------------------------------------------ data scenarios

  t_get_scenario(i: Input) {
    const id = s(i.scenario, 'scenario');
    if (this.store.hasScenario(id)) return this.store.scenario(id);
    throw new StudioError(`no data scenario ${id}${/^S\d\d$/i.test(id) ? ' (built-in scenarios are Rust: create_scenario with from: "' + id + '" copies one as data)' : ''}`, 404);
  }

  async t_create_scenario(i: Input) {
    const title = typeof i.title === 'string' && i.title ? i.title : 'Recreated crash';
    let id = typeof i.id === 'string' && i.id ? i.id : slug(title).toLowerCase();
    if (!i.id) for (let k = 2; this.store.hasScenario(id); k++) id = `${slug(title).toLowerCase()}-${k}`;
    this.engine.validId(id);
    checkId(id, 'scenario');
    if (this.store.hasScenario(id)) throw new StudioError(`scenario ${id} exists`, 409);
    let d: ScenarioDesc;
    const from = typeof i.from === 'string' ? i.from : '';
    if (!from) d = defaultScenario(id, title);
    else if (this.store.hasScenario(from)) d = { ...this.store.scenario(from), id, title };
    else d = { ...(await this.engine.exportBuiltin(from)), id, title: typeof i.title === 'string' && i.title ? i.title : `${from} as data` };
    this.engine.checkDesc(d);
    return this.store.putScenario(d);
  }

  /** Replace a scenario description whole (the Scene view's undo and redo use it). */
  t_put_scenario(i: Input) {
    const d = i.scenario as ScenarioDesc;
    if (!d || typeof d !== 'object') throw new StudioError('scenario must be a ScenarioDesc document');
    checkId(d.id, 'scenario');
    this.engine.checkDesc(d);
    return this.store.putScenario(d);
  }

  t_delete_scenario(i: Input) {
    const id = s(i.scenario, 'scenario');
    this.store.scenario(id);
    this.store.removeScenario(id);
    return { deleted: id };
  }

  private scenarioEdit(i: Input, fn: (d: ScenarioDesc) => unknown) {
    const d = this.store.scenario(s(i.scenario, 'scenario'));
    const r = fn(d);
    this.engine.checkDesc(d);
    this.store.putScenario(d);
    return r ?? d;
  }

  t_set_scenario_info(i: Input) {
    return this.scenarioEdit(i, (d) => {
      if (typeof i.title === 'string') d.title = i.title;
      if (typeof i.notes === 'string') d.notes = i.notes;
      if (typeof i.secs_after_impact === 'number') d.secs_after_impact = n(i.secs_after_impact, 'secs_after_impact');
      if (typeof i.contact === 'string') {
        if (!['discrete', 'edge', 'continuous'].includes(i.contact)) throw new StudioError('contact is discrete, edge or continuous');
        d.contact = i.contact as Contact;
      }
    });
  }

  private boxOf(d: ScenarioDesc, name: unknown) {
    const b = d.boxes.find((x) => x.name === name);
    if (!b) throw new StudioError(`${d.id} has no box ${JSON.stringify(name)} (boxes: ${d.boxes.map((x) => x.name).join(', ') || 'none'})`, 404);
    return b;
  }
  private bodyOf(d: ScenarioDesc, name: unknown) {
    const b = d.bodies.find((x) => x.name === name);
    if (!b) throw new StudioError(`${d.id} has no body ${JSON.stringify(name)} (bodies: ${d.bodies.map((x) => x.name).join(', ')})`, 404);
    return b;
  }
  private fresh(names: string[], base: string) {
    let id = base;
    for (let k = 2; names.includes(id); k++) id = `${base}${k}`;
    return id;
  }

  t_add_box(i: Input) {
    return this.scenarioEdit(i, (d) => {
      const name = typeof i.name === 'string' && i.name ? i.name : this.fresh(d.boxes.map((b) => b.name), 'box');
      if (d.boxes.some((b) => b.name === name)) throw new StudioError(`box ${name} exists`, 409);
      const half = i.half !== undefined ? v3mm(vec3(i.half, 'half')) : ([1, 1, 0.25] as [number, number, number]);
      d.boxes.push({ name, centre: i.centre !== undefined ? v3mm(vec3(i.centre, 'centre')) : [0, half[1], 8], half, yaw_deg: typeof i.yaw_deg === 'number' ? Math.round(i.yaw_deg * 100) / 100 : 0 });
    });
  }

  t_set_box(i: Input) {
    return this.scenarioEdit(i, (d) => {
      const b = this.boxOf(d, i.box);
      if (i.centre !== undefined) b.centre = v3mm(vec3(i.centre, 'centre'));
      if (i.half !== undefined) b.half = v3mm(vec3(i.half, 'half'));
      if (typeof i.yaw_deg === 'number') b.yaw_deg = Math.round(i.yaw_deg * 100) / 100;
      if (typeof i.name === 'string' && i.name && i.name !== b.name) {
        if (d.boxes.some((x) => x.name === i.name)) throw new StudioError(`box ${i.name} exists`, 409);
        b.name = i.name;
      }
    });
  }

  t_remove_box(i: Input) {
    return this.scenarioEdit(i, (d) => {
      this.boxOf(d, i.box);
      d.boxes = d.boxes.filter((b) => b.name !== i.box);
    });
  }

  t_add_body(i: Input) {
    return this.scenarioEdit(i, (d) => {
      const name = typeof i.name === 'string' && i.name ? i.name : this.fresh(d.bodies.map((b) => b.name), 'car');
      if (d.bodies.some((b) => b.name === name)) throw new StudioError(`body ${name} exists`, 409);
      d.bodies.push({ name, design: typeof i.design === 'string' && i.design ? i.design : 'sedan', at: i.at !== undefined ? v3mm(vec3(i.at, 'at')) : [4, 0, 0], yaw_deg: typeof i.yaw_deg === 'number' ? i.yaw_deg : 0, speed_kmh: typeof i.speed_kmh === 'number' ? i.speed_kmh : 0, carrier: i.carrier === true });
    });
  }

  t_set_body(i: Input) {
    return this.scenarioEdit(i, (d) => {
      const b = this.bodyOf(d, i.body);
      if (typeof i.design === 'string' && i.design) b.design = i.design;
      if (i.at !== undefined) b.at = v3mm(vec3(i.at, 'at'));
      if (typeof i.yaw_deg === 'number') b.yaw_deg = Math.round(n(i.yaw_deg, 'yaw_deg') * 100) / 100;
      if (typeof i.speed_kmh === 'number') b.speed_kmh = Math.round(n(i.speed_kmh, 'speed_kmh') * 1000) / 1000;
      if (typeof i.carrier === 'boolean') b.carrier = i.carrier;
      if (i.velocity === null) delete b.velocity;
      else if (i.velocity !== undefined) b.velocity = vec3(i.velocity, 'velocity');
    });
  }

  t_remove_body(i: Input) {
    return this.scenarioEdit(i, (d) => {
      this.bodyOf(d, i.body);
      if (d.bodies.length === 1) throw new StudioError('a scenario keeps at least one body');
      d.bodies = d.bodies.filter((b) => b.name !== i.body);
    });
  }

  // ------------------------------------------------------------------ Rubble (from the recording)

  /** Pixel size of the image a camera or region covers. */
  regionSize(f: Frame, regionId: string | null): [number, number] {
    if (!f.image) throw new StudioError(`frame ${f.id} has no image`);
    const r = regionId ? f.regions.find((x) => x.id === regionId) : null;
    const c = r?.crop ?? [0, 0, 1, 1];
    return [(c[2] - c[0]) * f.image.width, (c[3] - c[1]) * f.image.height];
  }

  /** The time and camera a frame is compared at: its pin and own camera when it has them,
   * else the scenario moment and named camera its labels pick (Rust's rule). */
  async rubbleLink(f: Frame) {
    const shot = f.shot ? this.store.shot(f.shot) : null;
    if (!shot?.rubble) throw new StudioError(`frame ${f.id}: its shot is not linked to a Rubble scenario (link_scenario)`, 409);
    const { scenario } = shot.rubble;
    const contact = (shot.rubble.contact || 'edge') as Contact;
    const { info, status } = await this.engine.recorded(scenario, contact);
    const pick = this.engine.pick(info.moments, info.cameras, info.id, f.moment.label || 'peak', f.view || 'side');
    const momentName = f.rubble.moment ?? pick.moment;
    const named = info.moments.find((m) => m.name === momentName);
    const at_ms = f.pin ? f.pin.at_ms : named?.after_impact_ms ?? 0;
    const camName = f.rubble.camera ?? pick.camera;
    const cam = info.cameras.find((c) => c.name === camName);
    const camera: FrameCamera | null = f.camera ?? (cam ? { eye: cam.eye, target: cam.target, up: [0, 1, 0], fov_y_deg: 60, region: null } : null);
    return {
      scenario: info.id,
      contact,
      at_ms,
      moment: f.pin ? null : momentName,
      camera: f.camera ? 'own' : camName,
      camera_def: camera,
      pinned: !!f.pin,
      check: !!f.pin && f.pin.recording !== status.hash,
      recording: { executor: status.executor, hash: status.hash, fresh: status.fresh },
      own_camera: !!f.camera,
    };
  }

  private schemaOf(f: Frame, want?: unknown): StructureSchema {
    if (typeof want === 'string' && want) return this.store.structure(want);
    const inst = f.regions.flatMap((r) => r.tracing.instances)[0];
    return this.store.structure(inst?.schema ?? (f.view === 'top' ? 'vehicle-top' : f.view.includes('three') ? 'vehicle-three-quarter' : 'vehicle-side'));
  }

  private spec(f: Frame, link: Awaited<ReturnType<Ops['rubbleLink']>>, schema: StructureSchema): RubbleSpec {
    if (schema.rubble_view === 'car_frame' || !link.camera_def) return { at_ms: link.at_ms, width: 1, height: 1, body: 0 };
    const [w, h] = this.regionSize(f, link.camera_def.region);
    return { at_ms: link.at_ms, camera: link.camera_def, width: Math.round(w), height: Math.round(h), body: 0 };
  }

  async t_get_rubble_outline(i: Input) {
    const f = this.store.frame(s(i.frame, 'frame'));
    const link = await this.rubbleLink(f);
    if (!link.camera_def) throw new StudioError(`frame ${f.id}: no camera (set_frame_camera, or a scenario camera)`, 409);
    const sc = this.schemaOf(f, i.schema);
    const [w, h] = this.regionSize(f, link.camera_def.region);
    const inst = this.engine.outline(sc, { at_ms: link.at_ms, camera: link.camera_def, width: Math.round(w), height: Math.round(h), body: 0 });
    return { ...link, schema: sc.id, instance: inst, region: link.camera_def.region ?? null };
  }

  async t_get_comparison(i: Input) {
    const frames = this.store.frames();
    const f = frames.find((x) => x.id === i.frame) ?? this.store.frame(s(i.frame, 'frame'));
    const before = this.beforeOf(f, frames) ?? f;
    const sources: CompareSource[] = [];
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
        // A pinned image against an unpinned "before" (or the other way) would compare a time and
        // camera chosen by hand with a named moment and camera: meaningless numbers. Say so.
        if (before.id !== f.id && now.pinned !== was.pinned) {
          const missing = now.pinned ? before.id : f.id;
          throw new StudioError(`pin the before image to compare: ${missing} has no pin (time and camera) in the Scene view, ${now.pinned ? f.id : before.id} has one`);
        }
        sources.push({ name: 'rubble', before: { rubble: this.spec(before, was, schema) }, now: { rubble: this.spec(f, now, schema) } });
        const when = now.pinned ? `${now.at_ms} ms (pinned)` : `${now.moment} (${now.at_ms} ms)`;
        used.push({ name: 'rubble', label: `Rubble ${now.scenario} ${when}`, source: 'rubble', region: '', before_region: '' });
        rubble = { scenario: now.scenario, contact: now.contact, at_ms: now.at_ms, before_ms: was.at_ms, moment: now.moment, pinned: now.pinned, check: now.check, camera: now.camera, view: schema.rubble_view, recording: now.recording };
      } catch (e) {
        rubble = { error: (e as Error).message };
      }
    }
    const result = this.engine.compare(schema, sources.filter((x) => x.name !== 'rubble' || (rubble && !(rubble as { error?: string }).error)));
    return { frame_id: f.id, before_frame: before.id, sources: used.filter((u) => u.name !== 'rubble' || (rubble && !(rubble as { error?: string }).error)), rubble, ...result, schema: schema.id };
  }

  async t_list_design_matches(i: Input) {
    const f = this.store.frame(s(i.frame, 'frame'));
    const r = i.region ? this.region(f, i.region) : f.regions.find((x) => tracedCount(x) > 0);
    if (!r) throw new StudioError(`frame ${f.id} has no traced region`, 409);
    const inst = this.instance(r, i.instance);
    const [w, h] = this.regionSize(f, r.id);
    return this.engine.rankDesigns(this.store.structure(inst.schema), inst, w, h);
  }
}
