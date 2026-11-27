import { useEffect, useState } from 'react';
import type { FrameDoc } from './api.ts';
import type { Ctx } from './App.tsx';

const MOMENTS = ['before', 'first contact', 'early crush', 'mid crush', 'peak crush', 'peak', 'rebound', 'rotation', 'aftermath'];
const VIEWS = ['side', 'top', 'three-quarter', 'front three-quarter', 'three-quarter rear', 'front', 'rear', 'wide'];

/** A text field saved when it loses focus or on Enter. */
function Lazy({ label, value, onSave, type = 'text', list }: { label: string; value: string; onSave: (v: string) => void; type?: string; list?: string }) {
  const [v, setV] = useState(value);
  useEffect(() => setV(value), [value]);
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type={type}
        value={v}
        list={list}
        onChange={(e) => setV(e.target.value)}
        onBlur={() => v !== value && onSave(v)}
        onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
      />
    </label>
  );
}

export function FrameInfo({ ctx, frame }: { ctx: Ctx; frame: FrameDoc }) {
  const set = (input: Record<string, unknown>) => ctx.edit('set_frame_info', { frame: frame.id, ...input });
  const sameShot = ctx.state.frames.filter((f) => f.shot && f.shot === frame.shot && f.id !== frame.id && f.view === frame.view);
  return (
    <div className="section">
      <h3>Frame</h3>
      <label className="field">
        <span>Shot</span>
        <select value={frame.shot ?? ''} onChange={(e) => ctx.act('assign_frame', { frame: frame.id, shot: e.target.value })}>
          <option value="">(no shot)</option>
          {ctx.state.shots.map((s) => (
            <option key={s.id} value={s.id}>
              {s.title}
            </option>
          ))}
        </select>
      </label>
      <datalist id="moments">{MOMENTS.map((m) => <option key={m} value={m} />)}</datalist>
      <datalist id="views">{VIEWS.map((m) => <option key={m} value={m} />)}</datalist>
      <div className="grid3" style={{ gridTemplateColumns: '1fr 1fr' }}>
        <Lazy label="Moment" value={frame.moment.label} list="moments" onSave={(v) => set({ moment: v })} />
        <Lazy label="View" value={frame.view} list="views" onSave={(v) => set({ view: v })} />
        <Lazy label="ms after impact" type="number" value={frame.moment.after_impact_ms?.toString() ?? ''} onSave={(v) => set({ after_impact_ms: v === '' ? null : Number(v) })} />
        <Lazy label="Video time (s)" type="number" value={frame.time?.video_s?.toString() ?? ''} onSave={(v) => v !== '' && set({ video_s: Number(v) })} />
      </div>
      <label className="field">
        <span>Compared against (before)</span>
        <select value={frame.before_frame ?? ''} onChange={(e) => set({ before_frame: e.target.value })}>
          <option value="">automatic{frame.before_resolved ? `: ${frame.before_resolved}` : ''}</option>
          {sameShot.map((f) => (
            <option key={f.id} value={f.id}>
              {f.id} ({f.moment})
            </option>
          ))}
        </select>
      </label>
      <Lazy label="Notes" value={frame.notes} onSave={(v) => set({ notes: v })} />
    </div>
  );
}
