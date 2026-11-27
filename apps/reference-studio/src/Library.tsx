import { useEffect, useRef, useState } from 'react';
import type { FrameSummary, ScenarioInfo as Scenario, Status } from '../shared/model.ts';
import { tool, upload, type ShotRow } from './api.ts';
import type { Ctx } from './App.tsx';

const STATUSES: Status[] = ['untraced', 'traced', 'paired', 'linked', 'no image'];

function Thumb({ f, ctx }: { f: FrameSummary; ctx: Ctx }) {
  return (
    <div className="thumb" role="group" aria-label={f.id}>
      <button className="pic" style={{ border: 0, padding: 0, cursor: 'pointer' }} onClick={() => ctx.open('annotate', f.id)} aria-label={`Annotate ${f.id}`}>
        {f.thumb && !f.status.includes('no image') ? <img src={f.thumb} alt="" loading="lazy" /> : <span className="muted small">no image</span>}
      </button>
      <div className="meta">
        <span className="name">{f.id}</span>
        <span className="muted small">
          {f.moment || 'moment ?'} · {f.view || 'view ?'} · {f.regions} region{f.regions === 1 ? '' : 's'}
        </span>
        <span className="badges">
          {f.status.map((s) => (
            <span key={s} className={`badge ${s.replace(' ', '-')}`}>
              {s}
            </span>
          ))}
        </span>
        <span className="row">
          <button className="btn small" onClick={() => ctx.open('annotate', f.id)}>Annotate</button>
          <button className="btn small" onClick={() => ctx.open('link', f.id)}>Link</button>
          <button className="btn small" onClick={() => ctx.open('compare', f.id)} disabled={!f.status.includes('traced')}>Compare</button>
        </span>
        <label className="small">
          <span className="sr-only">Shot of {f.id}</span>
          <select value={f.shot ?? ''} onChange={(e) => ctx.act('assign_frame', { frame: f.id, shot: e.target.value })} aria-label={`Shot of ${f.id}`}>
            <option value="">(no shot)</option>
            {ctx.state.shots.map((s) => (
              <option key={s.id} value={s.id}>
                {s.id}
              </option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
}

function ShotCard({ shot, frames, ctx, scenarios }: { shot: ShotRow | null; frames: FrameSummary[]; ctx: Ctx; scenarios: Scenario[] }) {
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState(shot?.title ?? '');
  useEffect(() => setTitle(shot?.title ?? ''), [shot?.title]);
  const id = shot ? `shot-${shot.id}` : 'shot-none';
  return (
    <section className="card shot" aria-labelledby={id}>
      <div className="shot-head">
        {editing && shot ? (
          <form
            className="row"
            style={{ flex: '1 1 260px' }}
            onSubmit={async (e) => {
              e.preventDefault();
              await ctx.act('update_shot', { shot: shot.id, title });
              setEditing(false);
            }}
          >
            <input type="text" value={title} onChange={(e) => setTitle(e.target.value)} aria-label="Shot title" style={{ flex: 1 }} autoFocus />
            <button className="btn small primary">Save</button>
            <button type="button" className="btn small" onClick={() => setEditing(false)}>Cancel</button>
          </form>
        ) : (
          <h2 id={id}>{shot ? shot.title : 'Not in a shot'}</h2>
        )}
        {shot && !editing && <button className="btn small" onClick={() => setEditing(true)}>Rename</button>}
        {shot && (
          <label className="row small">
            <span className="muted">Rubble scenario</span>
            <select value={shot.rubble?.scenario ?? ''} onChange={(e) => ctx.act('link_scenario', { shot: shot.id, scenario: e.target.value })} aria-label={`Rubble scenario of ${shot.id}`}>
              <option value="">(not linked)</option>
              {scenarios.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.id}: {s.title}
                </option>
              ))}
              {shot.rubble && !scenarios.some((s) => s.id === shot.rubble!.scenario) && <option value={shot.rubble.scenario}>{shot.rubble.scenario}</option>}
            </select>
          </label>
        )}
      </div>
      {shot && (
        <p className="muted small" style={{ margin: '0 0 8px' }}>
          {shot.source.kind === 'video' ? (
            <>
              Video: <a href={shot.source.url} target="_blank" rel="noreferrer noopener">{shot.source.url}</a>
            </>
          ) : (
            'Screenshots'
          )}
          {' · '}
          {shot.frames} frames, {shot.traced} traced, {shot.paired} paired{shot.rubble ? `, linked to ${shot.rubble.scenario} (${shot.rubble.contact}, run ${shot.rubble.run})` : ''}
          {shot.no_image ? ` · ${shot.no_image} without an image` : ''}
        </p>
      )}
      {frames.length ? (
        <div className="thumbs">
          {frames.map((f) => (
            <Thumb key={f.id} f={f} ctx={ctx} />
          ))}
        </div>
      ) : (
        <p className="muted small">No frames match. Drop images here with this shot chosen as the target, or move frames into it.</p>
      )}
    </section>
  );
}

export function Library({ ctx }: { ctx: Ctx }) {
  const { state } = ctx;
  const [filter, setFilter] = useState<string>('');
  const [query, setQuery] = useState('');
  const [target, setTarget] = useState<string>('');
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [newShot, setNewShot] = useState<{ title: string; url: string } | null>(null);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const picker = useRef<HTMLInputElement>(null);

  useEffect(() => {
    tool<Scenario[]>('list_scenarios').then(setScenarios).catch(() => setScenarios([]));
  }, []);

  const add = async (files: FileList | File[]) => {
    const list = [...files].filter((f) => /\.(png|jpe?g|webp)$/i.test(f.name));
    if (!list.length) return ctx.toast('only png, jpg and webp images', true);
    setBusy(true);
    let last = '';
    try {
      for (const f of list) last = (await upload(f, { shot: target || null })).id;
      await ctx.act('list_shots', {});
      ctx.toast(`added ${list.length} image${list.length === 1 ? '' : 's'}${target ? ` to ${target}` : ''}`);
      if (list.length === 1 && last) ctx.open('annotate', last);
    } catch (e) {
      ctx.toast((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  };

  const q = query.trim().toLowerCase();
  const show = (f: FrameSummary) => (!filter || f.status.includes(filter as Status)) && (!q || `${f.id} ${f.moment} ${f.view}`.toLowerCase().includes(q));
  const groups: { shot: ShotRow | null; frames: FrameSummary[] }[] = state.shots.map((s) => ({ shot: s, frames: state.frames.filter((f) => f.shot === s.id && show(f)) }));
  const loose = state.frames.filter((f) => !f.shot && show(f));
  if (loose.length || !state.shots.length) groups.push({ shot: null, frames: loose });
  const untraced = state.frames.filter((f) => f.status.includes('untraced'));

  return (
    <div
      className="page"
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        add(e.dataTransfer.files);
      }}
    >
      <div className="lib-head">
        <button className="btn primary" onClick={() => picker.current?.click()} disabled={busy}>
          {busy ? 'Adding…' : 'Add images…'}
        </button>
        <input ref={picker} type="file" accept="image/png,image/jpeg,image/webp" multiple hidden onChange={(e) => e.target.files && add(e.target.files)} />
        <label className="row small">
          <span className="muted">into</span>
          <select value={target} onChange={(e) => setTarget(e.target.value)} aria-label="Shot new images go into">
            <option value="">(no shot)</option>
            {state.shots.map((s) => (
              <option key={s.id} value={s.id}>
                {s.id}
              </option>
            ))}
          </select>
        </label>
        <button className="btn" onClick={() => setNewShot({ title: '', url: '' })}>New shot</button>
        <span className="spacer" />
        <label className="row small">
          <span className="sr-only">Search frames</span>
          <input type="text" placeholder="Search frames" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Search frames" />
        </label>
        <label className="row small">
          <span className="sr-only">Status filter</span>
          <select value={filter} onChange={(e) => setFilter(e.target.value)} aria-label="Status filter">
            <option value="">All frames ({state.frames.length})</option>
            {STATUSES.map((s) => (
              <option key={s} value={s}>
                {s} ({state.frames.filter((f) => f.status.includes(s)).length})
              </option>
            ))}
          </select>
        </label>
      </div>
      {newShot && (
        <form
          className="card shot row"
          onSubmit={async (e) => {
            e.preventDefault();
            const s = (await ctx.act('create_shot', { title: newShot.title, ...(newShot.url ? { video_url: newShot.url } : {}) })) as { id: string };
            setTarget(s.id);
            setNewShot(null);
          }}
        >
          <input type="text" required placeholder="What the run is, e.g. Sedan into a pole at 50 km/h" value={newShot.title} onChange={(e) => setNewShot({ ...newShot, title: e.target.value })} aria-label="New shot title" style={{ flex: '2 1 280px' }} autoFocus />
          <input type="text" placeholder="Video URL (optional)" value={newShot.url} onChange={(e) => setNewShot({ ...newShot, url: e.target.value })} aria-label="Video URL" style={{ flex: '1 1 200px' }} />
          <button className="btn primary">Create shot</button>
          <button type="button" className="btn" onClick={() => setNewShot(null)}>Cancel</button>
        </form>
      )}
      <div className={`drop${over ? ' over' : ''}`}>
        Drop screenshots or video frames anywhere on this page to add them to the library{target ? ` (into ${target})` : ''}. Images go to the workspace's
        <code>library/</code> (never committed); tracings are saved as points in its <code>frames/</code>.
      </div>
      {!state.frames.length && !state.shots.length ? (
        <div className="card empty">
          <h2>Nothing here yet</h2>
          <ol>
            <li>Create a shot for one crash run (<b>New shot</b>), with its video URL if it came from one.</li>
            <li>Add its screenshots or frames (<b>Add images…</b> or drop them here).</li>
            <li>Open a frame, mark its regions (a real test half, a BeamNG half) and trace each with a structure.</li>
            <li>Link the shot to a Rubble scenario, then open <b>Compare</b>.</li>
          </ol>
          <p className="muted small">Existing crash reference frames: <code>npm run import-prototype</code>.</p>
        </div>
      ) : (
        <>
          {untraced.length > 0 && !filter && (
            <p className="muted small">
              {untraced.length} frame{untraced.length === 1 ? '' : 's'} still untraced.{' '}
              <button className="btn small" onClick={() => setFilter('untraced')}>Show them</button>
            </p>
          )}
          {groups
            .filter((g) => g.frames.length || !filter)
            .map((g) => (
              <ShotCard key={g.shot?.id ?? '-'} shot={g.shot} frames={g.frames} ctx={ctx} scenarios={scenarios} />
            ))}
        </>
      )}
    </div>
  );
}
