import { useEffect, useState } from 'react';
import type { StructureSchema, Vis, Xy } from '../shared/model.ts';
import { tool, type FrameDoc } from './api.ts';
import type { Ctx } from './App.tsx';
import { SOURCE_HEX } from './geom.ts';

/** A normalised shape as Rust's compare returns it (x forward, y up). */
interface Shape {
  points: Record<string, Xy>;
  lines: Record<string, Xy[]>;
  circles: Record<string, { c: Xy; r: number }>;
  loops?: Record<string, Xy[][]>;
  vis: Record<string, Vis>;
}

export interface Comparison {
  frame_id: string;
  before_frame: string;
  schema: string;
  frame: { id: string; label: string } | null;
  sources: { name: string; label: string; source: string }[];
  rubble: { scenario: string; at_ms: number; before_ms: number; moment: string | null; pinned: boolean; check: boolean; camera: string | null; view: string; recording: { executor: string; fresh: boolean } } | { error: string } | null;
  shapes: Record<string, { before: Shape; now: Shape }>;
  rows: { id: string; label: string; values: Record<string, { v: number; vis: Vis; int?: boolean } | null> }[];
}

/** get_comparison for a frame, again whenever the frame changes (debounced). */
export function useComparison(frame: string, revision: number, enabled = true) {
  const [data, setData] = useState<Comparison | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!enabled) return setData(null);
    let live = true;
    const t = setTimeout(() => {
      tool<Comparison>('get_comparison', { frame })
        .then((d) => live && (setData(d), setError(null)))
        .catch((e) => live && (setData(null), setError(e.message)));
    }, 150);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [frame, revision, enabled]);
  return { data, error };
}

const fmt = (x: { v: number; int?: boolean } | null, label: string) =>
  x == null ? 'n/a' : x.int ? String(x.v) : label.includes('deg') ? `${x.v >= 0 ? '+' : ''}${x.v.toFixed(1)}` : `${x.v >= 0 ? '+' : ''}${x.v.toFixed(3)}`;

export function NumbersTable({ data, compact = false }: { data: Comparison; compact?: boolean }) {
  return (
    <div className="table-wrap">
      <table className="nums">
        <thead>
          <tr>
            <th scope="col">{compact ? '' : 'measure'}</th>
            {data.sources.map((s) => (
              <th key={s.name} scope="col" style={{ color: SOURCE_HEX[s.source] }}>
                {s.label || s.name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((r) => (
            <tr key={r.id}>
              <th scope="row" style={{ fontWeight: 400 }}>{r.label}</th>
              {data.sources.map((s) => {
                const x = r.values[s.name];
                return (
                  <td key={s.name} className={x?.vis && x.vis !== 'clear' ? x.vis : ''} title={x?.vis && x.vis !== 'clear' ? `measured from ${x.vis} elements` : undefined}>
                    {fmt(x, r.label)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small" style={{ margin: '4px 0 0' }}>
        {data.frame ? data.frame.label : 'no common frame: every source lacks its origin or unit, only angles and counts are compared'}. Orange: measured from blurred
        elements; red: from hidden or guessed ones. Reported comparisons, not criteria.
      </p>
    </div>
  );
}

type Layer = { shape: Shape; colour: string; dashed: boolean; key: string };

/** A plot of normalised shapes, front at the left as filmed (screen x grows toward the rear). */
function Plot({ title, layers, schema, height = 300 }: { title: string; layers: Layer[]; schema: StructureSchema | undefined; height?: number }) {
  const xs: number[] = [];
  const ys: number[] = [];
  const add = (p: Xy) => (xs.push(p[0]), ys.push(p[1]));
  for (const { shape } of layers) {
    Object.values(shape.lines).forEach((l) => l.forEach(add));
    Object.values(shape.points).forEach(add);
    Object.values(shape.circles).forEach((c) => (add([c.c[0] - c.r, c.c[1] - c.r]), add([c.c[0] + c.r, c.c[1] + c.r])));
  }
  const W = 640;
  if (!xs.length) return <div className="plot"><h3>{title}</h3><p className="muted small" style={{ padding: 10 }}>nothing to draw</p></div>;
  let [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const pad = 0.06 * Math.max(x1 - x0, y1 - y0, 1e-6);
  [x0, x1, y0, y1] = [x0 - pad, x1 + pad, y0 - pad, y1 + pad];
  const sc = Math.min((W - 20) / (x1 - x0), (height - 20) / (y1 - y0));
  const tx = (p: Xy): Xy => [10 + (x1 - p[0]) * sc, 10 + (y1 - p[1]) * sc];
  const pts = (l: Xy[]) => l.map((p) => tx(p).join(',')).join(' ');
  const grid = [];
  const step = [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 50, 100].find((g) => (x1 - x0) / g < 14) ?? 100;
  for (let g = Math.ceil(x0 / step) * step; g <= x1; g += step) {
    const [a, b] = [tx([g, y0]), tx([g, y1])];
    grid.push(<line key={`x${g}`} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="#2a3344" strokeWidth={Math.abs(g) < 1e-9 ? 2 : 1} />);
  }
  for (let g = Math.ceil(y0 / step) * step; g <= y1; g += step) {
    const [a, b] = [tx([x0, g]), tx([x1, g])];
    grid.push(<line key={`y${g}`} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="#2a3344" strokeWidth={Math.abs(g) < 1e-9 ? 2 : 1} />);
  }
  return (
    <div className="plot">
      <h3>{title}</h3>
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={title}>
        {grid}
        {layers.map(({ shape, colour, dashed, key }) => (
          <g key={key} opacity={dashed ? 0.75 : 1}>
            {Object.entries(shape.loops ?? {}).flatMap(([g, ls]) => ls.map((l, k) => <polyline key={`${g}${k}`} points={pts(l)} fill="none" stroke={colour} strokeOpacity={0.25} strokeWidth={1} />))}
            {Object.entries(shape.lines).map(([n, l]) => (
              <polyline key={n} points={pts(l)} fill="none" stroke={colour} strokeWidth={n === 'hood' ? 4 : 2.2} strokeDasharray={dashed ? '7 5' : undefined} strokeLinejoin="round">
                <title>{schema?.elements.find((e) => e.name === n)?.label ?? n}</title>
              </polyline>
            ))}
            {Object.entries(shape.circles).map(([n, c]) => {
              const p = tx(c.c);
              return <circle key={n} cx={p[0]} cy={p[1]} r={c.r * sc} fill="none" stroke={colour} strokeWidth={2.2} strokeDasharray={dashed ? '7 5' : undefined} />;
            })}
            {Object.entries(shape.points).map(([n, q]) => {
              const p = tx(q);
              return <rect key={n} x={p[0] - 5} y={p[1] - 5} width={10} height={10} fill="none" stroke={colour} strokeWidth={2} />;
            })}
          </g>
        ))}
      </svg>
    </div>
  );
}

export function Compare({ ctx, frame }: { ctx: Ctx; frame: FrameDoc }) {
  const traced = frame.regions.some((r) => r.tracing.instances.length);
  const { data, error } = useComparison(frame.id, ctx.revision, traced);
  const schema = ctx.state.schemas.find((s) => s.id === data?.schema);
  const shot = ctx.state.shots.find((s) => s.id === frame.shot);
  if (!traced) return <div className="page"><div className="notice">Trace at least one region of this frame first (Annotate).</div></div>;
  if (error) return <div className="page"><div className="notice bad">{error}</div></div>;
  if (!data) return <div className="page"><div className="notice">Computing (Rubble's shapes come from the recorded run: the scene is simulated on this page's GPU first when it has no recording)…</div></div>;
  const col = (name: string) => SOURCE_HEX[data.sources.find((s) => s.name === name)?.source ?? 'other'];
  const rub = data.rubble && 'error' in data.rubble ? data.rubble.error : null;
  return (
    <div className="compare">
      <div className="row">
        <h2 style={{ flex: 1 }}>
          {frame.id}: {frame.moment.label || 'moment ?'} against {data.before_frame === frame.id ? 'itself (it is the before frame)' : data.before_frame}
        </h2>
        <button className="btn small" onClick={() => ctx.open('annotate')}>Annotate</button>
        <button className="btn small" onClick={() => ctx.open('link')}>Link</button>
      </div>
      <div className="legend">
        {data.sources.map((s) => (
          <span key={s.name}>
            <span className="dot" style={{ background: col(s.name) }} /> {s.label}
          </span>
        ))}
        {data.rubble && !rub && 'at_ms' in data.rubble && (
          <span className="muted">
            Rubble: {data.rubble.scenario} at {data.rubble.at_ms} ms{data.rubble.pinned ? ' (pinned)' : data.rubble.moment ? ` (${data.rubble.moment})` : ''} against {data.rubble.before_ms} ms ({data.rubble.view}) · recording {data.rubble.recording.executor === 'webgpu' ? 'WebGPU' : 'CPU fallback (tests only)'}
            {data.rubble.check ? ' · check after re-simulation' : ''}
          </span>
        )}
      </div>
      {rub && <div className="notice bad">Rubble's outline: {rub}</div>}
      {!shot?.rubble && <div className="notice">This shot is not linked to a Rubble scenario: link it (library or Link view) to add Rubble's automatic outline.</div>}
      <div className="plots">
        <Plot title={`Common frame: every source now (${data.frame?.label ?? 'raw axes'})`} layers={data.sources.map((s) => ({ shape: data.shapes[s.name].now, colour: col(s.name), dashed: false, key: s.name }))} schema={schema} height={340} />
      </div>
      <h3>Change: each source against its own before (dashed)</h3>
      <div className="plots">
        {data.sources.map((s) => (
          <Plot
            key={s.name}
            title={s.label}
            schema={schema}
            layers={[
              { shape: data.shapes[s.name].before, colour: '#8a93a6', dashed: true, key: 'b' },
              { shape: data.shapes[s.name].now, colour: col(s.name), dashed: false, key: 'n' },
            ]}
          />
        ))}
      </div>
      <div className="card" style={{ padding: 12 }}>
        <h3>Shape numbers</h3>
        <NumbersTable data={data} />
      </div>
    </div>
  );
}
