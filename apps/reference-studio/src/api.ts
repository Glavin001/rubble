// The page's only route to data: the studio server's HTTP API (server/api.ts).

import type { Frame, FrameSummary, Shot, StructureSchema } from '../shared/model.ts';
import type { ToolDef } from '../shared/tools.ts';

export interface ShotRow extends Shot {
  frames: number;
  traced: number;
  paired: number;
  linked: boolean;
  no_image: number;
}

export interface State {
  shots: ShotRow[];
  frames: FrameSummary[];
  schemas: StructureSchema[];
  tools: ToolDef[];
  paths: { data: string; library: string };
}

export interface FrameDoc extends Frame {
  before_resolved: string | null;
  image_url: string | null;
  image_on_disk: boolean;
}

export class ApiError extends Error {}

async function unwrap<T>(r: Response): Promise<T> {
  const j = (await r.json().catch(() => ({ ok: false, error: `${r.status} ${r.statusText}` }))) as { ok: boolean; result?: T; error?: string };
  if (!r.ok || !j.ok) throw new ApiError(j.error ?? `${r.status}`);
  return j.result as T;
}

export async function getState(): Promise<State> {
  const r = await fetch('/api/state');
  if (!r.ok) throw new ApiError(`state: ${r.status}`);
  return r.json();
}

export function tool<T = unknown>(name: string, input: Record<string, unknown> = {}): Promise<T> {
  return fetch(`/api/tool/${encodeURIComponent(name)}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(input),
  }).then((r) => unwrap<T>(r));
}

export function upload(file: File, opts: { shot?: string | null; view?: string; moment?: string } = {}): Promise<Frame> {
  const q = new URLSearchParams({ name: file.name });
  if (opts.shot) q.set('shot', opts.shot);
  if (opts.view) q.set('view', opts.view);
  if (opts.moment) q.set('moment', opts.moment);
  return fetch(`/api/upload?${q}`, { method: 'POST', headers: { 'content-type': 'application/octet-stream' }, body: file }).then((r) => unwrap<Frame>(r));
}
