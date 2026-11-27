// The page's data: the open workspace (src/workspace.ts) in a Store, the data tools over it
// (shared/ops.ts) and the engine they compute with (Rust in wasm). The studio is a static
// site: there is no server; this is the only route to the documents.

import type { Frame, FrameSummary, Instance, RecordingStatus, Registry, SceneInfo, Shot, StructureSchema } from '../shared/model.ts';
import { Ops, type EngineApi } from '../shared/ops.ts';
import { Store, StudioError } from '../shared/store.ts';
import type { ToolDef } from '../shared/tools.ts';
import { TOOLS } from '../shared/tools.ts';
import type { Workspace } from '../shared/workspace.ts';
import { engineModule, loadEngine } from './engine.ts';
import { scene } from './scene.ts';

export interface ShotRow extends Shot {
  frames: number;
  traced: number;
  paired: number;
  linked: boolean;
  no_image: number;
  pinned: number;
}

export interface State {
  shots: ShotRow[];
  frames: FrameSummary[];
  schemas: StructureSchema[];
  tools: ToolDef[];
  workspace: { kind: Workspace['kind']; name: string };
}

export interface FrameDoc extends Frame {
  before_resolved: string | null;
  image_url: string | null;
  image_on_disk: boolean;
}

export class ApiError extends Error {}

let store: Store | null = null;
let ops: Ops | null = null;
const urls = new Map<string, string>();

/** The engine the data tools compute with: Rust in wasm, and the Scene view's recording. */
const engineApi: EngineApi = {
  registry: (data) => engineModule().studioRegistry(data) as Registry,
  pick: (m, c, s, l, v) => engineModule().studioPick(m, c, s, l, v),
  validId: (id) => engineModule().studioValidId(id),
  checkDesc: (d) => engineModule().studioCheckDesc(d),
  exportBuiltin: async (id) => {
    const e = await scene.ready();
    await e.ensureTemplates({ id });
    return e.studio.exportBuiltin(id);
  },
  recorded: (scenario) => scene.ensureRecorded(scenario) as Promise<{ info: SceneInfo; status: RecordingStatus }>,
  outline: (schema, spec) => scene.engine!.studio.outline(schema, spec),
  compare: (schema, sources) => scene.engine!.studio.compare(schema, sources),
  rankDesigns: async (schema: StructureSchema, inst: Instance, w: number, h: number) => {
    const e = await scene.ready();
    await e.loadAllDesigns();
    return e.studio.rankDesigns(schema, inst, w, h);
  },
};

/** Open a workspace: every document read into memory, every image given a URL. */
export async function openWorkspace(ws: Workspace, onChange: () => void): Promise<Store> {
  await loadEngine();
  for (const u of urls.values()) URL.revokeObjectURL(u);
  urls.clear();
  store = await Store.open(ws);
  ops = new Ops(store, engineApi);
  scene.attach(store);
  store.onChange = () => {
    scene.onStoreChange();
    onChange();
  };
  for (const f of store.images) {
    const b = await store.image(f);
    if (b) urls.set(f, URL.createObjectURL(new Blob([b as Uint8Array<ArrayBuffer>])));
  }
  return store;
}

export function currentStore(): Store {
  if (!store) throw new ApiError('no workspace open');
  return store;
}

export function currentOps(): Ops {
  if (!ops) throw new ApiError('no workspace open');
  return ops;
}

export function imageUrl(file: string | null | undefined): string | null {
  return file ? urls.get(file) ?? null : null;
}

export async function getState(): Promise<State> {
  const o = currentOps();
  const shots = o.t_list_shots() as ShotRow[];
  return { shots, frames: o.summaries(imageUrl), schemas: o.store.structures(), tools: TOOLS, workspace: { kind: o.store.ws.kind, name: o.store.ws.name } };
}

/** Run a data tool; its writes are in the workspace when this resolves. */
export async function tool<T = unknown>(name: string, input: Record<string, unknown> = {}): Promise<T> {
  const o = currentOps();
  try {
    const r = await o.call(name, input);
    await o.store.flush();
    if (name === 'add_image') {
      const f = r as Frame;
      if (f.image && !urls.has(f.image.file)) {
        const b = await o.store.image(f.image.file);
        if (b) urls.set(f.image.file, URL.createObjectURL(new Blob([b as Uint8Array<ArrayBuffer>])));
      }
    }
    if (name === 'get_frame') {
      const f = r as FrameDoc;
      f.image_url = imageUrl(f.image?.file);
    }
    return r as T;
  } catch (e) {
    throw new ApiError(e instanceof StudioError ? e.message : String((e as Error)?.message ?? e));
  }
}

export async function upload(file: File, opts: { shot?: string | null; view?: string; moment?: string } = {}): Promise<Frame> {
  const o = currentOps();
  const bytes = new Uint8Array(await file.arrayBuffer());
  const f = await o.addImage(file.name, bytes, opts);
  await o.store.flush();
  if (f.image && !urls.has(f.image.file)) urls.set(f.image.file, URL.createObjectURL(new Blob([bytes])));
  return f;
}

/** A frame's image as an <img> can load it. */
export function frameImage(f: Frame | null): string | null {
  return imageUrl(f?.image?.file);
}
