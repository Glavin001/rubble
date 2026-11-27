import { useCallback, useEffect, useLayoutEffect, useRef, useState, type PointerEvent as RPE } from 'react';
import type { ElementValue, Instance, Region, SourceTag, StructureSchema, Vis, Xy } from '../shared/model.ts';
import { SOURCES, VIS } from '../shared/model.ts';
import { tool, type FrameDoc } from './api.ts';
import type { Ctx } from './App.tsx';
import { FrameInfo } from './FrameInfo.tsx';
import { SOURCE_VAR, hasData, imageToRegion, insertIndex, regionToImage } from './geom.ts';
import { NumbersTable, useComparison } from './Compare.tsx';

/** Image pixels -> screen: screen = (x, y) + s * p. `fit` is the scale that fits the image. */
export interface ViewBox {
  s: number;
  x: number;
  y: number;
  fit: number;
}

/** The next element of the schema without data after `current` (in schema order). */
export function nextElement(sc: StructureSchema, inst: Instance | undefined, current: string | null): string | null {
  const els = sc.elements;
  const at = Math.max(-1, els.findIndex((e) => e.name === current));
  for (let k = 1; k <= els.length; k++) {
    const e = els[(at + k) % els.length];
    if (!hasData(inst?.elements[e.name])) return e.name;
  }
  return current ?? els[0]?.name ?? null;
}

type Drag =
  | { kind: 'point'; region: string; element: string; part: 'p' | 'c' | 'rim' | number; last: number }
  | { kind: 'pan'; sx: number; sy: number; x: number; y: number }
  | { kind: 'crop'; a: Xy; b: Xy };

interface RubbleOverlay {
  instance: { elements: Record<string, ElementValue>; loops: Record<string, Xy[][]> };
  region: string | null;
}

function Stage({ ctx, frame, viewBox, setViewBox, mode, setMode, newSource, guided, overlay }: {
  ctx: Ctx; frame: FrameDoc; viewBox: ViewBox | null; setViewBox: (v: ViewBox) => void; mode: 'trace' | 'region'; setMode: (m: 'trace' | 'region') => void;
  newSource: SourceTag; guided: boolean; overlay: RubbleOverlay | null;
}) {
  const host = useRef<HTMLDivElement>(null);
  const drag = useRef<Drag | null>(null);
  const space = useRef(false);
  const [cropDraft, setCropDraft] = useState<[Xy, Xy] | null>(null);
  const [img, setImg] = useState<{ w: number; h: number } | null>(frame.image ? { w: frame.image.width, h: frame.image.height } : null);
  const W = img?.w ?? 1;
  const H = img?.h ?? 1;
  const vb = viewBox;
  const region = ctx.region;
  const inst = region?.tracing.instances[0];

  const fit = useCallback(() => {
    const r = host.current?.getBoundingClientRect();
    if (!r || !img) return;
    const s = Math.min(r.width / img.w, r.height / img.h) * 0.96;
    setViewBox({ s, fit: s, x: (r.width - img.w * s) / 2, y: (r.height - img.h * s) / 2 });
  }, [img, setViewBox]);

  useLayoutEffect(() => {
    if (!vb) fit();
  }, [vb, fit]);
  useEffect(() => {
    const ro = new ResizeObserver(() => fit());
    if (host.current) ro.observe(host.current);
    return () => ro.disconnect();
  }, [fit]);
  useEffect(() => setImg(frame.image ? { w: frame.image.width, h: frame.image.height } : null), [frame.image]);

  const zoomAt = useCallback((k: number, sx: number, sy: number) => {
    if (!vb) return;
    const s = Math.min(vb.fit * 40, Math.max(vb.fit * 0.5, vb.s * k));
    const kk = s / vb.s;
    setViewBox({ ...vb, s, x: sx - (sx - vb.x) * kk, y: sy - (sy - vb.y) * kk });
  }, [vb, setViewBox]);

  useEffect(() => {
    const on = (e: Event) => {
      const d = (e as CustomEvent).detail as { zoom: number; x?: number; y?: number };
      const r = host.current?.getBoundingClientRect();
      if (!vb || !r) return;
      const s = vb.fit * Math.max(0.5, Math.min(40, d.zoom));
      const cx = (d.x ?? 0.5) * W;
      const cy = (d.y ?? 0.5) * H;
      setViewBox({ ...vb, s, x: r.width / 2 - cx * s, y: r.height / 2 - cy * s });
    };
    addEventListener('studio-zoom', on);
    return () => removeEventListener('studio-zoom', on);
  }, [vb, W, H, setViewBox]);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
    };
    el.addEventListener('wheel', wheel, { passive: false });
    return () => el.removeEventListener('wheel', wheel);
  }, [zoomAt]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest('input, textarea, select')) return;
      if (e.code === 'Space') {
        space.current = true;
        e.preventDefault();
      }
      const r = host.current?.getBoundingClientRect();
      if (!r) return;
      if (e.key === '0') fit();
      if (e.key === '+' || e.key === '=') zoomAt(1.25, r.width / 2, r.height / 2);
      if (e.key === '-') zoomAt(0.8, r.width / 2, r.height / 2);
    };
    const up = (e: KeyboardEvent) => e.code === 'Space' && (space.current = false);
    addEventListener('keydown', down);
    addEventListener('keyup', up);
    return () => {
      removeEventListener('keydown', down);
      removeEventListener('keyup', up);
    };
  }, [fit, zoomAt]);

  const toImage = (e: { clientX: number; clientY: number }): Xy => {
    const r = host.current!.getBoundingClientRect();
    return [(e.clientX - r.left - vb!.x) / vb!.s, (e.clientY - r.top - vb!.y) / vb!.s];
  };
  const toR = (r: Region, p: Xy) => imageToRegion(r.crop, W, H, p);
  const toI = (r: Region, p: Xy) => regionToImage(r.crop, W, H, p);

  const local = (mut: (v: ElementValue) => void, element: string) => {
    const f = structuredClone(frame);
    const r = f.regions.find((x) => x.id === region!.id)!;
    const i = r.tracing.instances[0];
    i.elements[element] ??= { vis: 'clear' };
    mut(i.elements[element]);
    ctx.draft(f);
  };

  const save = (d: Extract<Drag, { kind: 'point' }>, p: Xy, final: boolean) => {
    const base = { frame: frame.id, region: d.region, element: d.element, x: p[0], y: p[1] };
    const [name, input] = typeof d.part === 'number' ? ['move_polyline_point', { ...base, index: d.part }] : ['set_keypoint', d.part === 'p' ? base : { ...base, part: d.part }];
    if (final) return ctx.edit(name, input, { noUndo: true });
    return tool(name, input).catch(() => undefined);
  };

  const advance = (element: string, done: boolean) => {
    if (guided && done && ctx.schema && inst) {
      const after = structuredClone(inst);
      after.elements[element] = { ...(after.elements[element] ?? { vis: 'clear' }), p: [0, 0] };
      ctx.setElement(nextElement(ctx.schema, after, element));
    }
  };

  const onDown = (e: RPE<SVGSVGElement>) => {
    if (!vb || !img) return;
    const r = host.current!.getBoundingClientRect();
    (e.target as Element).setPointerCapture?.(e.pointerId);
    if (e.button === 1 || e.button === 2 || space.current) {
      drag.current = { kind: 'pan', sx: e.clientX - r.left, sy: e.clientY - r.top, x: vb.x, y: vb.y };
      return;
    }
    const p = toImage(e);
    if (mode === 'region') {
      drag.current = { kind: 'crop', a: p, b: p };
      setCropDraft([p, p]);
      return;
    }
    const handle = (e.target as Element).closest('[data-handle]')?.getAttribute('data-handle');
    if (handle && region) {
      const [element, part] = handle.split('|');
      const pp = /^\d+$/.test(part) ? Number(part) : (part as 'p' | 'c' | 'rim');
      if (e.altKey) {
        ctx.edit('remove_point', { frame: frame.id, region: region.id, element, ...(typeof pp === 'number' ? { index: pp } : {}) });
        return;
      }
      ctx.setElement(element);
      ctx.snapshot();
      drag.current = { kind: 'point', region: region.id, element, part: pp, last: 0 };
      return;
    }
    if (!region || !inst || !ctx.schema || !ctx.element) {
      ctx.toast(region ? 'choose a structure for this region first (side panel)' : 'mark a region first: “Draw region”, or “Whole image” in the side panel', true);
      return;
    }
    const def = ctx.schema.elements.find((x) => x.name === ctx.element);
    if (!def) return;
    const q = toR(region, p);
    const v = inst.elements[def.name];
    const base = { frame: frame.id, region: region.id, element: def.name, x: q[0], y: q[1] };
    if (def.type === 'point') ctx.edit('set_keypoint', base).then(() => advance(def.name, true));
    else if (def.type === 'circle') {
      const part = !v?.c ? 'c' : 'rim';
      ctx.edit('set_keypoint', { ...base, part }).then(() => advance(def.name, part === 'rim'));
    } else {
      const pts = v?.pts ?? [];
      const index = e.shiftKey ? insertIndex(pts.map((x) => toI(region, x)), p) : pts.length;
      ctx.edit('add_polyline_point', { ...base, index });
    }
  };

  const onMove = (e: RPE<SVGSVGElement>) => {
    const d = drag.current;
    if (!d || !vb) return;
    const r = host.current!.getBoundingClientRect();
    if (d.kind === 'pan') {
      setViewBox({ ...vb, x: d.x + e.clientX - r.left - d.sx, y: d.y + e.clientY - r.top - d.sy });
    } else if (d.kind === 'crop') {
      d.b = toImage(e);
      setCropDraft([d.a, d.b]);
    } else if (region) {
      const q = toR(region, toImage(e));
      local((v) => {
        if (typeof d.part === 'number') v.pts![d.part] = q;
        else if (d.part === 'c' && v.c && v.rim) {
          v.rim = [v.rim[0] + q[0] - v.c[0], v.rim[1] + q[1] - v.c[1]];
          v.c = q;
        } else v[d.part] = q;
      }, d.element);
      const now = performance.now();
      if (now - d.last > 120) {
        d.last = now;
        save(d, q, false);
      }
    }
  };

  const onUp = async (e: RPE<SVGSVGElement>) => {
    const d = drag.current;
    drag.current = null;
    if (!d) return;
    if (d.kind === 'crop') {
      setCropDraft(null);
      const [x0, y0, x1, y1] = [d.a[0] / W, d.a[1] / H, d.b[0] / W, d.b[1] / H].map((v) => Math.min(1, Math.max(0, v)));
      if (Math.abs(x1 - x0) < 0.01 || Math.abs(y1 - y0) < 0.01) return ctx.toast('drag a rectangle over the part of the image the region covers');
      const reg = (await ctx.edit('mark_region', { frame: frame.id, source: newSource, x0, y0, x1, y1 })) as Region;
      ctx.setRegion(reg.id);
      setMode('trace');
    } else if (d.kind === 'point' && region) {
      save(d, toR(region, toImage(e)), true);
    }
  };

  const s = vb?.s ?? 1;
  const hr = 6 / s;
  const lw = (k: number) => k / s;
  return (
    <div className="stage" ref={host}>
      {img && vb && (
        <svg
          role="application"
          aria-label={`Image ${frame.id}: click to place the active element`}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onContextMenu={(e) => e.preventDefault()}
          style={{ cursor: mode === 'region' ? 'crosshair' : space.current ? 'grab' : 'crosshair' }}
        >
          <g transform={`translate(${vb.x} ${vb.y}) scale(${s})`}>
            {frame.image_on_disk && frame.image_url ? <image href={frame.image_url} x={0} y={0} width={W} height={H} /> : <rect width={W} height={H} fill="#333" />}
            {frame.regions.map((r) => {
              const [a, b] = [toI(r, [0, 0]), toI(r, [1, 1])];
              const act = r.id === region?.id;
              return (
                <g key={r.id} opacity={act ? 1 : 0.5}>
                  <rect x={a[0]} y={a[1]} width={b[0] - a[0]} height={b[1] - a[1]} fill="none" stroke={SOURCE_VAR[r.source]} strokeWidth={lw(act ? 2.5 : 1.5)} strokeDasharray={`${lw(10)} ${lw(6)}`} />
                  <text x={a[0] + lw(6)} y={a[1] + lw(18)} fill={SOURCE_VAR[r.source]} fontSize={lw(14)} fontWeight={700} stroke="#000" strokeWidth={lw(3)} paintOrder="stroke">
                    {r.label || r.id}
                  </text>
                  <Tracing r={r} schema={ctx.state.schemas.find((x) => x.id === r.tracing.instances[0]?.schema)} toI={toI} lw={lw} hr={hr} active={act} element={ctx.element} />
                </g>
              );
            })}
            {overlay && <Ghost overlay={overlay} frame={frame} schema={ctx.schema} W={W} H={H} lw={lw} />}
            {cropDraft && (
              <rect x={Math.min(cropDraft[0][0], cropDraft[1][0])} y={Math.min(cropDraft[0][1], cropDraft[1][1])} width={Math.abs(cropDraft[1][0] - cropDraft[0][0])} height={Math.abs(cropDraft[1][1] - cropDraft[0][1])} fill="rgba(255,255,255,0.08)" stroke={SOURCE_VAR[newSource]} strokeWidth={lw(2)} />
            )}
          </g>
        </svg>
      )}
      {!frame.image_on_disk && (
        <div className="stage-hint" style={{ top: 50, bottom: 'auto' }}>
          This frame's image is not in the library{frame.image ? ` (${frame.image.file})` : ''}. Add it from the library page (or run the importer); its tracing is kept.
        </div>
      )}
    </div>
  );
}

function Tracing({ r, schema, toI, lw, hr, active, element }: {
  r: Region; schema: StructureSchema | undefined; toI: (r: Region, p: Xy) => Xy; lw: (k: number) => number; hr: number; active: boolean; element: string | null;
}) {
  const inst = r.tracing.instances[0];
  if (!inst || !schema) return null;
  return (
    <>
      {schema.elements.map((def) => {
        const v = inst.elements[def.name];
        if (!hasData(v)) return null;
        const col = def.colour;
        const on = active && def.name === element;
        const dash = v.vis === 'hidden' ? `${lw(4)} ${lw(5)}` : v.vis === 'blurred' ? `${lw(10)} ${lw(4)}` : undefined;
        const handle = (key: string, p: Xy, rr = hr) => (
          <circle key={key} data-handle={`${def.name}|${key}`} cx={p[0]} cy={p[1]} r={on ? rr * 1.25 : rr} fill={on ? '#fff36b' : col} stroke="#000" strokeWidth={lw(1.5)} style={{ cursor: 'move' }} />
        );
        const label = (p: Xy) => (
          <text x={p[0] + lw(8)} y={p[1] - lw(8)} fill={col} fontSize={lw(13)} fontWeight={700} stroke="#000" strokeWidth={lw(3)} paintOrder="stroke" pointerEvents="none">
            {def.label}
          </text>
        );
        if (def.type === 'point' && v.p) {
          const p = toI(r, v.p);
          return <g key={def.name}>{active && handle('p', p)}{!active && <circle cx={p[0]} cy={p[1]} r={hr * 0.7} fill={col} />}{active && label(p)}</g>;
        }
        if (def.type === 'circle' && v.c) {
          const c = toI(r, v.c);
          const rim = v.rim ? toI(r, v.rim) : null;
          return (
            <g key={def.name}>
              {rim && <circle cx={c[0]} cy={c[1]} r={Math.hypot(rim[0] - c[0], rim[1] - c[1])} fill="none" stroke={col} strokeWidth={lw(on ? 3 : 2)} strokeDasharray={dash} />}
              {active && handle('c', c)}
              {active && rim && handle('rim', rim, hr * 0.8)}
              {active && label(c)}
            </g>
          );
        }
        if (def.type === 'polyline' && v.pts?.length) {
          const pts = v.pts.map((p) => toI(r, p));
          return (
            <g key={def.name}>
              <polyline points={pts.map((p) => p.join(',')).join(' ')} fill="none" stroke={col} strokeWidth={lw(on ? 4 : 2.5)} strokeLinejoin="round" strokeDasharray={dash} />
              {active && pts.map((p, k) => handle(String(k), p, hr * 0.85))}
              {active && label(pts[0])}
            </g>
          );
        }
        return null;
      })}
    </>
  );
}

function Ghost({ overlay, frame, schema, W, H, lw }: { overlay: RubbleOverlay; frame: FrameDoc; schema: StructureSchema | null; W: number; H: number; lw: (k: number) => number }) {
  const r = overlay.region ? frame.regions.find((x) => x.id === overlay.region) : null;
  const crop = r?.crop ?? [0, 0, 1, 1];
  const toI = (p: Xy) => regionToImage(crop, W, H, p);
  const col = (n: string) => schema?.elements.find((e) => e.name === n)?.colour ?? '#3ddc84';
  return (
    <g pointerEvents="none" opacity={0.95}>
      {Object.entries(overlay.instance.loops).flatMap(([g, ls]) =>
        ls.map((l, k) => <polyline key={`${g}${k}`} points={l.map((p) => toI(p).join(',')).join(' ')} fill="none" stroke="#3ddc84" strokeOpacity={0.45} strokeWidth={lw(1)} />),
      )}
      {Object.entries(overlay.instance.elements).map(([n, v]) => {
        if (v.pts) return <polyline key={n} points={v.pts.map((p) => toI(p).join(',')).join(' ')} fill="none" stroke={col(n)} strokeWidth={lw(2.5)} strokeDasharray={`${lw(8)} ${lw(5)}`} />;
        if (v.c && v.rim) {
          const c = toI(v.c);
          const m = toI(v.rim);
          return <circle key={n} cx={c[0]} cy={c[1]} r={Math.hypot(m[0] - c[0], m[1] - c[1])} fill="none" stroke={col(n)} strokeWidth={lw(2.5)} strokeDasharray={`${lw(8)} ${lw(5)}`} />;
        }
        if (v.p) {
          const p = toI(v.p);
          return <rect key={n} x={p[0] - lw(5)} y={p[1] - lw(5)} width={lw(10)} height={lw(10)} fill="none" stroke={col(n)} strokeWidth={lw(2)} />;
        }
        return null;
      })}
    </g>
  );
}

function Regions({ ctx, frame, newSource, setNewSource, setMode }: { ctx: Ctx; frame: FrameDoc; newSource: SourceTag; setNewSource: (s: SourceTag) => void; setMode: (m: 'trace' | 'region') => void }) {
  return (
    <div className="section">
      <h3>Regions</h3>
      <p className="muted small" style={{ margin: '4px 0' }}>A region is the part of the image one source shows: a real-test half, a BeamNG half.</p>
      <div className="regions">
        {frame.regions.map((r) => (
          <div key={r.id} className="region" aria-current={r.id === ctx.region?.id}>
            <div className="row">
              <button className="btn small" onClick={() => ctx.setRegion(r.id)} aria-pressed={r.id === ctx.region?.id}>
                <span className="dot" style={{ background: SOURCE_VAR[r.source] }} /> {r.label || r.id}
              </button>
              <select value={r.source} aria-label={`Source of region ${r.id}`} onChange={(e) => ctx.edit('mark_region', { frame: frame.id, region: r.id, source: e.target.value })}>
                {SOURCES.map((s) => <option key={s}>{s}</option>)}
              </select>
              <button className="btn small danger" aria-label={`Remove region ${r.id}`} onClick={() => confirm(`Remove region ${r.id} and its tracing?`) && ctx.edit('remove_region', { frame: frame.id, region: r.id })}>
                ✕
              </button>
            </div>
            <label className="row small">
              <span className="muted">Structure</span>
              <select
                value={r.tracing.instances[0]?.schema ?? ''}
                aria-label={`Structure of region ${r.id}`}
                onChange={async (e) => {
                  await ctx.edit('set_region_schema', { frame: frame.id, region: r.id, schema: e.target.value });
                  ctx.setRegion(r.id);
                }}
              >
                <option value="" disabled>choose…</option>
                {ctx.state.schemas.map((s) => <option key={s.id} value={s.id}>{s.title}</option>)}
              </select>
            </label>
          </div>
        ))}
      </div>
      <div className="row" style={{ marginTop: 6 }}>
        <select value={newSource} onChange={(e) => setNewSource(e.target.value as SourceTag)} aria-label="Source of the new region">
          {SOURCES.map((s) => <option key={s}>{s}</option>)}
        </select>
        <button className="btn small" onClick={() => setMode('region')} title="Drag a rectangle on the image (R)">Draw region</button>
        <button className="btn small" onClick={async () => ctx.setRegion(((await ctx.edit('mark_region', { frame: frame.id, source: newSource })) as Region).id)}>Whole image</button>
      </div>
    </div>
  );
}

function Structure({ ctx, frame, guided, setGuided }: { ctx: Ctx; frame: FrameDoc; guided: boolean; setGuided: (g: boolean) => void }) {
  const r = ctx.region;
  const sc = ctx.schema;
  const inst = r?.tracing.instances[0];
  if (!r) return <div className="section"><p className="muted">Mark a region to trace it.</p></div>;
  if (!sc || !inst) return <div className="section"><p className="muted">Choose the structure this region shows (Regions, above).</p></div>;
  const cur = sc.elements.find((e) => e.name === ctx.element);
  const v = cur ? inst.elements[cur.name] : undefined;
  const done = sc.elements.filter((e) => hasData(inst.elements[e.name])).length;
  return (
    <div className="section">
      <div className="row">
        <h3 style={{ flex: 1 }}>{sc.title}</h3>
        <span className="muted small">{done} / {sc.elements.length}</span>
      </div>
      <div className="row small" style={{ margin: '6px 0' }}>
        <label className="row">
          <span className="muted">Front at</span>
          <select value={inst.front} onChange={(e) => ctx.edit('set_region_schema', { frame: frame.id, region: r.id, schema: sc.id, front: e.target.value })} aria-label="Which side the front is on">
            {sc.orientation.front.map((f) => <option key={f}>{f}</option>)}
          </select>
        </label>
        <label className="row">
          <input type="checkbox" checked={guided} onChange={(e) => setGuided(e.target.checked)} /> guided (G)
        </label>
      </div>
      <ul className="elements" aria-label="Elements, in tracing order">
        {sc.elements.map((e) => {
          const ev = inst.elements[e.name];
          return (
            <li key={e.name}>
              <button className="el" aria-current={e.name === ctx.element} onClick={() => ctx.setElement(e.name)}>
                <span className="tick" aria-label={hasData(ev) ? 'placed' : 'not placed'}>{hasData(ev) ? '✓' : '·'}</span>
                <span>
                  <span className="dot" style={{ background: e.colour, marginRight: 6 }} />
                  {e.label} <span className="muted small">({e.type}{e.type === 'polyline' && ev?.pts ? `, ${ev.pts.length} pts` : ''})</span>
                </span>
                <span className={`vis ${ev?.vis ?? ''}`}>{hasData(ev) && ev.vis !== 'clear' ? ev.vis : ''}</span>
              </button>
            </li>
          );
        })}
      </ul>
      {cur && (
        <div className="notice small">
          <b>{cur.label}</b>: {cur.hint}
          <div className="row" style={{ marginTop: 6 }}>
            <label className="row">
              <span className="muted">Visibility (V)</span>
              <select value={v?.vis ?? 'clear'} disabled={!hasData(v)} onChange={(e) => ctx.edit('set_visibility', { frame: frame.id, region: r.id, element: cur.name, vis: e.target.value })}>
                {VIS.map((x) => <option key={x}>{x}</option>)}
              </select>
            </label>
            <button className="btn small" disabled={!hasData(v)} onClick={() => ctx.edit('remove_point', { frame: frame.id, region: r.id, element: cur.name })}>Clear</button>
            {cur.type === 'polyline' && <button className="btn small" onClick={() => ctx.setElement(nextElement(sc, inst, cur.name))}>Done: next (Enter)</button>}
          </div>
        </div>
      )}
      <div className="row" style={{ marginTop: 6 }}>
        <button className="btn small" onClick={() => ctx.edit('copy_tracing_from_previous', { frame: frame.id, region: r.id })} title="Copy this region's tracing from the previous frame of the shot (C)">
          Copy from previous frame
        </button>
      </div>
      <TracingInfo ctx={ctx} frame={frame} r={r} />
    </div>
  );
}

function TracingInfo({ ctx, frame, r }: { ctx: Ctx; frame: FrameDoc; r: Region }) {
  const [by, setBy] = useState(r.tracing.traced_by);
  useEffect(() => setBy(r.tracing.traced_by), [r.tracing.traced_by]);
  return (
    <div className="row small" style={{ marginTop: 8 }}>
      <label className="row">
        <span className="muted">Traced by</span>
        <input type="text" value={by} size={14} onChange={(e) => setBy(e.target.value)} onBlur={() => by !== r.tracing.traced_by && ctx.edit('set_tracing_info', { frame: frame.id, region: r.id, traced_by: by })} />
      </label>
      <label className="row">
        <input type="checkbox" checked={r.tracing.approximate} onChange={(e) => ctx.edit('set_tracing_info', { frame: frame.id, region: r.id, approximate: e.target.checked })} /> approximate
      </label>
    </div>
  );
}

const KEYS: [string, string][] = [
  ['click', 'place the active element (a circle: centre, then rim)'],
  ['shift-click', 'insert a polyline point where it fits'],
  ['drag', 'move a point'],
  ['alt-click', 'delete a point'],
  ['Enter / N', 'next element; P: previous'],
  ['Backspace', 'remove the active polyline\'s last point'],
  ['V', 'cycle visibility: clear, blurred, hidden'],
  ['wheel, + −, 0', 'zoom, fit'],
  ['space-drag, right-drag', 'pan'],
  ['R', 'draw a region; G: guided mode; C: copy previous'],
  ['⌘Z / ⇧⌘Z', 'undo / redo'],
  ['Alt+1…4', 'library, annotate, link, compare'],
];

export function Annotate({ ctx, frame, viewBox, setViewBox }: { ctx: Ctx; frame: FrameDoc; viewBox: ViewBox | null; setViewBox: (v: ViewBox) => void }) {
  const [mode, setMode] = useState<'trace' | 'region'>('trace');
  const [newSource, setNewSource] = useState<SourceTag>(frame.regions.some((r) => r.source === 'real') ? 'beamng' : 'real');
  const [guided, setGuided] = useState(true);
  const [ghost, setGhost] = useState(false);
  const [overlay, setOverlay] = useState<RubbleOverlay | null>(null);
  const cmp = useComparison(frame.id, ctx.revision, frame.regions.some((r) => r.tracing.instances.length));
  const shot = ctx.state.shots.find((s) => s.id === frame.shot);
  const canGhost = !!shot?.rubble && !!frame.camera;

  useEffect(() => {
    if (!ghost || !canGhost) return setOverlay(null);
    let live = true;
    tool<RubbleOverlay>('get_rubble_outline', { frame: frame.id }).then((o) => live && setOverlay(o)).catch((e) => ctx.toast(e.message, true));
    return () => {
      live = false;
    };
  }, [ghost, canGhost, frame.id, frame.camera, ctx]);

  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest('input, textarea, select') || e.metaKey || e.ctrlKey || e.altKey) return;
      const r = ctx.region;
      const sc = ctx.schema;
      const inst = r?.tracing.instances[0];
      const k = e.key.toLowerCase();
      if (k === 'r') setMode((m) => (m === 'region' ? 'trace' : 'region'));
      else if (k === 'escape') setMode('trace');
      else if (k === 'g') setGuided((g) => !g);
      else if (k === 'c' && r) ctx.edit('copy_tracing_from_previous', { frame: frame.id, region: r.id });
      if (!sc || !inst || !r) return;
      const els = sc.elements.map((x) => x.name);
      const at = els.indexOf(ctx.element ?? '');
      if (k === 'enter' || k === 'n') {
        e.preventDefault();
        ctx.setElement(k === 'enter' ? nextElement(sc, inst, ctx.element) : els[(at + 1) % els.length]);
      } else if (k === 'p') ctx.setElement(els[(at - 1 + els.length) % els.length]);
      else if (k === 'v' && ctx.element && hasData(inst.elements[ctx.element])) {
        const cur = inst.elements[ctx.element].vis;
        ctx.edit('set_visibility', { frame: frame.id, region: r.id, element: ctx.element, vis: VIS[(VIS.indexOf(cur) + 1) % VIS.length] as Vis });
      } else if (k === 'backspace' && ctx.element) {
        const v = inst.elements[ctx.element];
        if (v?.pts?.length) ctx.edit('remove_point', { frame: frame.id, region: r.id, element: ctx.element, index: v.pts.length - 1 });
        else if (hasData(v)) ctx.edit('remove_point', { frame: frame.id, region: r.id, element: ctx.element });
      }
    };
    addEventListener('keydown', on);
    return () => removeEventListener('keydown', on);
  }, [ctx, frame.id]);

  const def = ctx.schema?.elements.find((e) => e.name === ctx.element);
  const v = def && ctx.region?.tracing.instances[0]?.elements[def.name];
  const next =
    mode === 'region'
      ? `Drag a rectangle over the ${newSource} part of the image (Esc: cancel).`
      : !ctx.region
        ? 'Mark a region first: “Draw region” or “Whole image” in the side panel.'
        : !def
          ? 'Choose a structure for this region in the side panel.'
          : def.type === 'circle'
            ? `${def.label}: click the ${v?.c ? 'rim (tyre edge)' : 'centre'}.`
            : def.type === 'point'
              ? `${def.label}: click where it is.`
              : `${def.label}: click to add points in order (${v?.pts?.length ?? 0} so far); Enter when done.`;
  return (
    <div className="work annotate">
      <div className="stage-wrap">
        <Stage ctx={ctx} frame={frame} viewBox={viewBox} setViewBox={setViewBox} mode={mode} setMode={setMode} newSource={newSource} guided={guided} overlay={overlay} />
        <div className="stage-bar">
          <button className="btn small" onClick={() => window.dispatchEvent(new CustomEvent('studio-zoom', { detail: { zoom: 1 } }))}>Fit (0)</button>
          <button className="btn small" aria-pressed={mode === 'region'} onClick={() => setMode(mode === 'region' ? 'trace' : 'region')}>Draw region (R)</button>
          <button className="btn small" aria-pressed={ghost} disabled={!canGhost} onClick={() => setGhost(!ghost)} title={canGhost ? 'Rubble\'s outline through this frame\'s own camera' : 'Needs a linked shot and this frame\'s own camera (Link view)'}>
            Rubble ghost
          </button>
          <span className="btn small" style={{ cursor: 'default' }}>zoom {viewBox ? (viewBox.s / viewBox.fit).toFixed(1) : '1.0'}×</span>
        </div>
        <div className="stage-hint" aria-live="polite">{next}</div>
      </div>
      <aside className="side" aria-label="Frame, regions and elements">
        <Regions ctx={ctx} frame={frame} newSource={newSource} setNewSource={setNewSource} setMode={setMode} />
        <Structure ctx={ctx} frame={frame} guided={guided} setGuided={setGuided} />
        {cmp.data && (
          <div className="section">
            <div className="row">
              <h3 style={{ flex: 1 }}>Shape numbers (live)</h3>
              <button className="btn small" onClick={() => ctx.open('compare')}>Compare view</button>
            </div>
            <NumbersTable data={cmp.data} compact />
          </div>
        )}
        <FrameInfo ctx={ctx} frame={frame} />
        <div className="section">
          <h3>Keys</h3>
          <div className="keys">
            {KEYS.map(([k, d]) => (
              <span key={k} style={{ display: 'contents' }}>
                <kbd>{k}</kbd>
                <span>{d}</span>
              </span>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}
