import { useEffect, useState } from 'react';
import type { ElementValue, FrameCamera, ScenarioInfo as Scenario, Xy } from '../shared/model.ts';
import { tool, type FrameDoc } from './api.ts';
import type { Ctx } from './App.tsx';
import { FrameInfo } from './FrameInfo.tsx';
import { regionToImage } from './geom.ts';

interface Outline {
  scenario: string;
  contact: string;
  at_ms: number;
  /** The named moment used (null when the frame is pinned at a time). */
  moment: string | null;
  camera: string | null;
  pinned: boolean;
  check: boolean;
  own_camera: boolean;
  recording: { executor: string; hash: string; fresh: boolean };
  schema: string;
  instance: { elements: Record<string, ElementValue>; loops: Record<string, Xy[][]>; width: number; height: number; front: string };
  region: string | null;
}

/** Elements and part loops over an image of `w` × `h` (instance coordinates are 0..1 of it). */
export function Overlay({ inst, w, h, colour, crop = [0, 0, 1, 1] }: { inst: Outline['instance']; w: number; h: number; colour: (n: string) => string; crop?: [number, number, number, number] }) {
  const t = (p: Xy) => regionToImage(crop, w, h, p);
  const lw = Math.max(1, w / 640);
  return (
    <g pointerEvents="none">
      {Object.entries(inst.loops).flatMap(([g, ls]) => ls.map((l, k) => <polyline key={`${g}${k}`} points={l.map((p) => t(p).join(',')).join(' ')} fill="none" stroke="#3ddc84" strokeOpacity={0.5} strokeWidth={lw} />))}
      {Object.entries(inst.elements).map(([n, v]) => {
        if (v.pts) return <polyline key={n} points={v.pts.map((p) => t(p).join(',')).join(' ')} fill="none" stroke={colour(n)} strokeWidth={lw * 3} strokeLinejoin="round" />;
        if (v.c && v.rim) {
          const [c, m] = [t(v.c), t(v.rim)];
          return <circle key={n} cx={c[0]} cy={c[1]} r={Math.hypot(m[0] - c[0], m[1] - c[1])} fill="none" stroke={colour(n)} strokeWidth={lw * 3} />;
        }
        if (v.p) {
          const p = t(v.p);
          return <rect key={n} x={p[0] - 6 * lw} y={p[1] - 6 * lw} width={12 * lw} height={12 * lw} fill="none" stroke={colour(n)} strokeWidth={lw * 2} />;
        }
        return null;
      })}
    </g>
  );
}

const v3 = (a: number[]) => a.map((x) => +x.toFixed(3)).join(', ');
const parse3 = (s: string): [number, number, number] | null => {
  const a = s.split(/[\s,]+/).filter(Boolean).map(Number);
  return a.length === 3 && a.every(Number.isFinite) ? (a as [number, number, number]) : null;
};

function CameraForm({ ctx, frame, scenario }: { ctx: Ctx; frame: FrameDoc; scenario: Scenario | undefined }) {
  const cam = frame.camera;
  const [eye, setEye] = useState(cam ? v3(cam.eye) : '');
  const [target, setTarget] = useState(cam ? v3(cam.target) : '');
  const [up, setUp] = useState(cam ? v3(cam.up) : '0, 1, 0');
  const [fov, setFov] = useState(String(cam?.fov_y_deg ?? 60));
  const [region, setRegion] = useState(cam?.region ?? '');
  const [paste, setPaste] = useState('');
  useEffect(() => {
    setEye(cam ? v3(cam.eye) : '');
    setTarget(cam ? v3(cam.target) : '');
    setUp(cam ? v3(cam.up) : '0, 1, 0');
    setFov(String(cam?.fov_y_deg ?? 60));
    setRegion(cam?.region ?? '');
  }, [cam]);
  const save = (c: Partial<FrameCamera>) => ctx.edit('set_frame_camera', { frame: frame.id, ...c });
  const submit = () => {
    const [e, t, u] = [parse3(eye), parse3(target), parse3(up)];
    if (!e || !t || !u) return ctx.toast('eye, target and up are three numbers each', true);
    save({ eye: e, target: t, up: u, fov_y_deg: Number(fov), region });
  };
  const fromJson = () => {
    try {
      const j = JSON.parse(paste);
      const c = j.camera ?? j;
      if (!Array.isArray(c.eye) || !Array.isArray(c.target)) throw new Error('needs eye and target arrays');
      save({ eye: c.eye, target: c.target, up: c.up ?? [0, 1, 0], fov_y_deg: c.fov_y_deg ?? c.fov ?? 60, region: c.region ?? region });
      setPaste('');
    } catch (e) {
      ctx.toast(`not a camera: ${(e as Error).message}`, true);
    }
  };
  const nudge = (axis: number, d: number) => {
    const e = parse3(eye);
    const t = parse3(target);
    if (!e || !t) return;
    e[axis] += d;
    t[axis] += d;
    save({ eye: e, target: t, up: parse3(up) ?? [0, 1, 0], fov_y_deg: Number(fov), region });
  };
  return (
    <div className="section">
      <h3>This image's own camera</h3>
      <p className="muted small" style={{ margin: '4px 0' }}>
        Set by hand: metres in the scenario's world (+Y up). When set, Rubble's outline is projected through it instead of a named camera, and the annotate view can show it
        as a ghost over the image. A camera solve may give a first guess later; this is the primary path.
      </p>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <label className="field"><span>Eye (x, y, z)</span><input type="text" value={eye} onChange={(e) => setEye(e.target.value)} placeholder="4.6, 1.1, -0.3" /></label>
        <label className="field"><span>Target (x, y, z)</span><input type="text" value={target} onChange={(e) => setTarget(e.target.value)} placeholder="0, 0.8, -0.3" /></label>
        <div className="grid3" style={{ gridTemplateColumns: '2fr 1fr' }}>
          <label className="field"><span>Up</span><input type="text" value={up} onChange={(e) => setUp(e.target.value)} /></label>
          <label className="field"><span>Field of view (° vertical)</span><input type="number" step="0.5" min="1" max="170" value={fov} onChange={(e) => setFov(e.target.value)} /></label>
        </div>
        <label className="field">
          <span>Applies to (its image's aspect)</span>
          <select value={region} onChange={(e) => setRegion(e.target.value)}>
            <option value="">whole image</option>
            {frame.regions.map((r) => <option key={r.id} value={r.id}>region {r.label || r.id}</option>)}
          </select>
        </label>
        <div className="row">
          <button className="btn primary">Save camera</button>
          {cam && <button type="button" className="btn" onClick={() => ctx.edit('set_frame_camera', { frame: frame.id, clear: true })}>Clear</button>}
          {scenario && (
            <select
              aria-label="Start from a named camera"
              value=""
              onChange={(e) => {
                const c = scenario.cameras.find((x) => x.name === e.target.value);
                if (c) save({ eye: c.eye as FrameCamera['eye'], target: c.target as FrameCamera['target'], up: [0, 1, 0], fov_y_deg: 60, region });
              }}
            >
              <option value="">start from a named camera…</option>
              {scenario.cameras.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
            </select>
          )}
        </div>
      </form>
      {cam && (
        <div className="row small" style={{ marginTop: 6 }} aria-label="Move the camera">
          <span className="muted">Move eye and target:</span>
          {(['x', 'y', 'z'] as const).map((a, i) => (
            <span key={a} className="row">
              <button className="btn small" onClick={() => nudge(i, -0.1)}>{a} −0.1</button>
              <button className="btn small" onClick={() => nudge(i, 0.1)}>{a} +0.1</button>
            </span>
          ))}
        </div>
      )}
      <label className="field">
        <span>Paste a camera as JSON (eye, target, up, fov_y_deg), e.g. copied from the crash lab</span>
        <textarea rows={2} value={paste} onChange={(e) => setPaste(e.target.value)} placeholder='{"eye":[4.6,1.1,-0.3],"target":[0,0.8,-0.3],"fov_y_deg":60}' />
      </label>
      <div className="row">
        <button className="btn small" disabled={!paste.trim()} onClick={fromJson}>Use pasted camera</button>
        {cam && <button className="btn small" onClick={() => navigator.clipboard?.writeText(JSON.stringify(cam)).then(() => ctx.toast('camera copied'))}>Copy as JSON</button>}
      </div>
    </div>
  );
}

export function Link({ ctx, frame }: { ctx: Ctx; frame: FrameDoc }) {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [outline, setOutline] = useState<Outline | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const shot = ctx.state.shots.find((s) => s.id === frame.shot);
  const scenario = scenarios.find((s) => s.id === shot?.rubble?.scenario);
  const schema = ctx.state.schemas.find((s) => s.id === outline?.schema);
  const colour = (n: string) => schema?.elements.find((e) => e.name === n)?.colour ?? '#3ddc84';

  useEffect(() => {
    tool<Scenario[]>('list_scenarios').then(setScenarios).catch((e) => setErr(e.message));
  }, []);
  useEffect(() => {
    if (!shot?.rubble) return setOutline(null);
    let live = true;
    setBusy(true);
    tool<Outline>('get_rubble_outline', { frame: frame.id })
      .then((o) => live && (setOutline(o), setErr(null)))
      .catch((e) => live && (setOutline(null), setErr(e.message)))
      .finally(() => live && setBusy(false));
    return () => {
      live = false;
    };
  }, [frame.id, ctx.revision, shot?.rubble]);

  const camRegion = frame.camera?.region ? frame.regions.find((r) => r.id === frame.camera!.region) : null;
  return (
    <div className="work">
      <div className="link-main">
        {!shot && <div className="notice">Put this frame in a shot first (side panel), then link the shot to a Rubble scenario.</div>}
        {shot && !shot.rubble && <div className="notice">Link shot “{shot.title}” to a Rubble scenario (side panel) to see Rubble's automatic outline.</div>}
        {err && <div className="notice bad">{err}</div>}
        {busy && <div className="notice">Rubble's outline from the recorded run (simulated on this page's GPU first when there is no recording)…</div>}
        {outline && (
          <>
            <div className="plot">
              <h3>
                Rubble {outline.scenario} at {outline.pinned ? `${outline.at_ms} ms (pinned)` : `${outline.moment} (${outline.at_ms} ms)`}, camera {outline.own_camera ? 'this frame\'s own' : outline.camera}: its outline in the {outline.schema} schema
              </h3>
              <p className="muted small">
                From the recording ({outline.recording.executor === 'webgpu' ? 'WebGPU' : 'CPU fallback, tests only'}){outline.check ? ' · pinned on an earlier recording: check after re-simulation' : ''}. Pin a time and camera in the Scene view.
              </p>
              <svg viewBox={`0 0 ${outline.instance.width} ${outline.instance.height}`} role="img" aria-label="Rubble's outline">
                <rect width={outline.instance.width} height={outline.instance.height} fill="var(--canvas)" />
                <Overlay inst={outline.instance} w={outline.instance.width} h={outline.instance.height} colour={colour} />
              </svg>
            </div>
            {frame.camera && frame.image && frame.image_url && (
              <div className="plot">
                <h3>The reference image with Rubble's outline through its camera</h3>
                <svg viewBox={`0 0 ${frame.image.width} ${frame.image.height}`} role="img" aria-label="Reference image with Rubble's ghost">
                  {frame.image_on_disk && <image href={frame.image_url} width={frame.image.width} height={frame.image.height} />}
                  <Overlay inst={outline.instance} w={frame.image.width} h={frame.image.height} colour={colour} crop={camRegion?.crop ?? [0, 0, 1, 1]} />
                </svg>
              </div>
            )}
          </>
        )}
      </div>
      <aside className="side" aria-label="Scenario, moment and camera">
        {shot && (
          <div className="section">
            <h3>Shot: {shot.title}</h3>
            <label className="field">
              <span>Rubble scenario</span>
              <select value={shot.rubble?.scenario ?? ''} onChange={(e) => ctx.act('link_scenario', { shot: shot.id, scenario: e.target.value })}>
                <option value="">(not linked)</option>
                {scenarios.map((s) => <option key={s.id} value={s.id}>{s.id}: {s.title}</option>)}
              </select>
            </label>
            {shot.rubble && (
              <div className="grid3" style={{ gridTemplateColumns: '1fr 1fr' }}>
                <label className="field">
                  <span>Contact</span>
                  <select value={shot.rubble.contact} onChange={(e) => ctx.act('link_scenario', { shot: shot.id, scenario: shot.rubble!.scenario, contact: e.target.value })}>
                    {['discrete', 'edge', 'continuous'].map((c) => <option key={c}>{c}</option>)}
                  </select>
                </label>
                <label className="field">
                  <span>Run</span>
                  <input type="number" min={0} value={shot.rubble.run} onChange={(e) => ctx.act('link_scenario', { shot: shot.id, scenario: shot.rubble!.scenario, run: Number(e.target.value) })} />
                </label>
              </div>
            )}
          </div>
        )}
        {scenario && (
          <div className="section">
            <h3>This frame in {scenario.id}</h3>
            <label className="field">
              <span>Moment {frame.pin ? `(unused: pinned at ${frame.pin.at_ms} ms)` : ''}</span>
              <select value={frame.rubble.moment ?? ''} onChange={(e) => ctx.edit('link_frame', { frame: frame.id, moment: e.target.value })}>
                <option value="">automatic from “{frame.moment.label}”{outline && !frame.rubble.moment && outline.moment ? `: ${outline.moment}` : ''}</option>
                {scenario.moments.map((m) => <option key={m.name} value={m.name}>{m.name} ({m.after_impact_ms > 0 ? '+' : ''}{m.after_impact_ms} ms)</option>)}
              </select>
            </label>
            <label className="field">
              <span>Named camera {frame.camera ? '(unused: the frame has its own)' : ''}</span>
              <select value={frame.rubble.camera ?? ''} disabled={!!frame.camera} onChange={(e) => ctx.edit('link_frame', { frame: frame.id, camera: e.target.value })}>
                <option value="">automatic from view “{frame.view}”</option>
                {scenario.cameras.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
              </select>
            </label>
          </div>
        )}
        {shot?.rubble && <CameraForm ctx={ctx} frame={frame} scenario={scenario} />}
        <FrameInfo ctx={ctx} frame={frame} />
      </aside>
    </div>
  );
}
