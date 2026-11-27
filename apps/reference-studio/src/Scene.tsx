// The Scene view: a recreated crash set up and reviewed in 3D. Plain three.js in a canvas
// this component owns (the design note's choice): the ground, the static boxes and each
// body's collision surface, flat shaded, its vertices the node positions Rust gives every
// frame (the placed spawn state in Setup, the recording at the shown time in Review).
//
// Two activities: Setup (bodies, boxes, speeds, contact; any edit marks the recording stale)
// and Review (a timeline over the recording and a free camera; nothing re-simulates). A
// reference image binds to the recording by a time and its own camera, both chosen in
// Review ("Pin this image here"). Every number shown is Rust's (scene placement, first
// contact, positions, Rubble's outline, design ranking); this file draws and edits.

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { DesignMatch } from './generated/crash-reference.ts';
import type { ElementValue, Frame, ScenarioInfo, Xy } from '../shared/model.ts';
import { frameImage, tool, currentStore } from './api.ts';
import type { Ctx } from './App.tsx';
import { designs, type DesignRow } from './engine.ts';
import { Overlay } from './Link.tsx';
import { scene, type View3 } from './scene.ts';

const useScene = () => useSyncExternalStore((f) => scene.subscribe(f), () => scene.state);

const fmt = (x: number, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : '—');
const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';

/** The region of a frame the camera's image is (its crop) and its pixel size. */
function cropOf(f: Frame | null): { crop: [number, number, number, number]; w: number; h: number } | null {
  if (!f?.image) return null;
  const r = f.camera?.region ? f.regions.find((x) => x.id === f.camera!.region) : f.regions.find((x) => x.tracing.instances.length) ?? null;
  const crop = r?.crop ?? [0, 0, 1, 1];
  return { crop, w: (crop[2] - crop[0]) * f.image.width, h: (crop[3] - crop[1]) * f.image.height };
}

// ---------------------------------------------------------------------------- the 3D view

interface Three {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  bodies: THREE.Mesh[];
  boxes: THREE.Mesh[];
  version: number;
}

function View3D({ frameAspect, onPick, onDrag, children }: { frameAspect: number | null; onPick: (sel: { kind: 'body' | 'box'; name: string } | null) => void; onDrag: (kind: 'body' | 'box', name: string, at: [number, number, number]) => void; children?: React.ReactNode }) {
  const host = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const t = useRef<Three | null>(null);
  const st = useScene();
  const [size, setSize] = useState({ w: 640, h: 360 });

  // The drawing area: the whole box, or the image region's aspect when an image is overlaid
  // (so its outline and the 3D view share one projection).
  useEffect(() => {
    const el = host.current!;
    const ro = new ResizeObserver(() => {
      const W = el.clientWidth, H = el.clientHeight;
      if (!frameAspect) return setSize({ w: W, h: H });
      const w = Math.min(W, H * frameAspect);
      setSize({ w: Math.round(w), h: Math.round(w / frameAspect) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [frameAspect]);

  useEffect(() => {
    const renderer = new THREE.WebGLRenderer({ canvas: canvas.current!, antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(2, devicePixelRatio));
    const s = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.05, 500);
    s.add(new THREE.HemisphereLight(0xdde6ff, 0x404040, 1.4));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(6, 10, 4);
    s.add(sun);
    const grid = new THREE.GridHelper(60, 60, 0x6d7891, 0x3a4256);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.55;
    s.add(grid);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: 0x2a2f3a, roughness: 1 }));
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.002;
    ground.name = 'ground';
    s.add(ground);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;
    controls.addEventListener('change', () => {
      const c = camera;
      const dir = new THREE.Vector3();
      c.getWorldDirection(dir);
      const target = controls.target;
      scene.cameraMoved({ eye: [c.position.x, c.position.y, c.position.z].map((x) => +x.toFixed(4)) as View3['eye'], target: [target.x, target.y, target.z].map((x) => +x.toFixed(4)) as View3['target'], up: [c.up.x, c.up.y, c.up.z].map((x) => +x.toFixed(4)) as View3['up'], fov_y_deg: c.fov });
    });
    t.current = { renderer, scene: s, camera, controls, bodies: [], boxes: [], version: -1 };
    let raf = 0;
    const loop = () => {
      draw();
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      controls.dispose();
      renderer.dispose();
      t.current = null;
    };
  }, []);

  // The camera set from outside (a pin, a tool, an image): applied to the view.
  useEffect(() => {
    const x = t.current;
    if (!x) return;
    const c = st.camera;
    x.camera.up.set(...c.up);
    x.camera.position.set(...c.eye);
    x.camera.fov = c.fov_y_deg;
    x.controls.target.set(...c.target);
    // OrbitControls keeps the up it was made with: a new up (roll) needs new controls.
    const ctl = x.controls as unknown as { _quat?: THREE.Quaternion; _quatInverse?: THREE.Quaternion };
    if (ctl._quat && ctl._quatInverse) {
      ctl._quat.setFromUnitVectors(x.camera.up, new THREE.Vector3(0, 1, 0));
      ctl._quatInverse.copy(ctl._quat).invert();
    }
    x.camera.updateProjectionMatrix();
    x.camera.lookAt(...c.target);
    x.controls.update();
  }, [st.cameraVersion]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const x = t.current;
    if (!x) return;
    x.renderer.setSize(size.w, size.h, false);
    x.camera.aspect = size.w / Math.max(1, size.h);
    x.camera.updateProjectionMatrix();
  }, [size]);

  // The scene's objects: rebuilt when Rust places a new scene.
  useEffect(() => {
    const x = t.current;
    const info = st.info;
    if (!x) return;
    for (const m of [...x.bodies, ...x.boxes]) {
      x.scene.remove(m);
      m.geometry.dispose();
    }
    x.bodies = [];
    x.boxes = [];
    if (!info) return;
    const palette = [0x4f8cff, 0xff8a3d, 0x3ddc84, 0xd96bff, 0xffd24a];
    info.bodies.forEach((b, i) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(b.nodes * 3), 3));
      g.setIndex(b.triangles);
      const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: palette[i % palette.length], flatShading: true, side: THREE.DoubleSide, roughness: 0.75, metalness: 0.05 }));
      m.userData = { kind: 'body', name: b.name, offset: b.node_offset, nodes: b.nodes };
      m.frustumCulled = false;
      x.scene.add(m);
      x.bodies.push(m);
    });
    for (const b of info.boxes) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(2 * b.half[0], 2 * b.half[1], 2 * b.half[2]), new THREE.MeshStandardMaterial({ color: 0x8c93a3, roughness: 0.9, transparent: true, opacity: 0.92 }));
      m.position.set(...b.centre);
      m.rotation.y = (b.yaw_deg * Math.PI) / 180;
      m.userData = { kind: 'box', name: b.name };
      x.scene.add(m);
      x.boxes.push(m);
    }
    x.version = -1;
  }, [st.info]);

  // Selection highlight.
  useEffect(() => {
    const x = t.current;
    if (!x) return;
    for (const m of [...x.bodies, ...x.boxes]) {
      const mat = m.material as THREE.MeshStandardMaterial;
      const on = st.selected && m.userData.kind === st.selected.kind && m.userData.name === st.selected.name;
      mat.emissive.setHex(on ? 0x335599 : 0x000000);
    }
  }, [st.selected, st.info]);

  const lastPos = useRef<Float32Array | null>(null);
  const draw = () => {
    const x = t.current;
    if (!x) return;
    const pos = scene.positions;
    if (pos && pos !== lastPos.current) {
      lastPos.current = pos;
      for (const m of x.bodies) {
        const a = m.geometry.getAttribute('position') as THREE.BufferAttribute;
        const { offset, nodes } = m.userData as { offset: number; nodes: number };
        if (pos.length >= 3 * (offset + nodes)) (a.array as Float32Array).set(pos.subarray(3 * offset, 3 * (offset + nodes)));
        a.needsUpdate = true;
        m.geometry.computeBoundingSphere();
      }
    }
    x.renderer.setClearColor(new THREE.Color(css('--canvas')));
    x.renderer.render(x.scene, x.camera);
  };

  // Picking and dragging along the ground (Setup only).
  const drag = useRef<{ kind: 'body' | 'box'; name: string; mesh: THREE.Object3D; offset: THREE.Vector3; y: number; moved: boolean; start: THREE.Vector3; startPos: Float32Array | null } | null>(null);
  const ray = (e: React.PointerEvent) => {
    const x = t.current!;
    const r = canvas.current!.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    const rc = new THREE.Raycaster();
    rc.setFromCamera(ndc, x.camera);
    return rc;
  };
  const onDown = (e: React.PointerEvent) => {
    const x = t.current;
    if (!x || e.button !== 0) return;
    const rc = ray(e);
    const hit = rc.intersectObjects([...x.bodies, ...x.boxes], false)[0];
    if (!hit) {
      if (!e.shiftKey) onPick(null);
      return;
    }
    const { kind, name } = hit.object.userData as { kind: 'body' | 'box'; name: string };
    onPick({ kind, name });
    if (st.mode !== 'setup' || !st.editable) return;
    const y = kind === 'box' ? hit.object.position.y : 0;
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -y);
    const p = new THREE.Vector3();
    rc.ray.intersectPlane(plane, p);
    const anchor = kind === 'box' ? hit.object.position.clone() : new THREE.Vector3(...(st.info!.bodies.find((b) => b.name === name)!.at));
    drag.current = { kind, name, mesh: hit.object, offset: anchor.clone().sub(p), y, moved: false, start: anchor.clone(), startPos: scene.positions ? scene.positions.slice() : null };
    x.controls.enabled = false;
    (e.target as Element).setPointerCapture(e.pointerId);
  };
  const onMove = (e: React.PointerEvent) => {
    const d = drag.current;
    const x = t.current;
    if (!d || !x) return;
    const p = new THREE.Vector3();
    if (!ray(e).ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), -d.y), p)) return;
    const at = p.add(d.offset);
    d.moved = true;
    if (d.kind === 'box') d.mesh.position.set(at.x, d.y, at.z);
    else {
      // The body's surface moved rigidly with the pointer until Rust re-places it on release.
      const dx = at.x - d.start.x, dz = at.z - d.start.z;
      const m = d.mesh as THREE.Mesh;
      const a = m.geometry.getAttribute('position') as THREE.BufferAttribute;
      const { offset, nodes } = m.userData as { offset: number; nodes: number };
      const arr = a.array as Float32Array;
      for (let i = 0; i < nodes; i++) {
        arr[3 * i] = d.startPos![3 * (offset + i)] + dx;
        arr[3 * i + 2] = d.startPos![3 * (offset + i) + 2] + dz;
      }
      a.needsUpdate = true;
    }
  };
  const onUp = () => {
    const d = drag.current;
    const x = t.current;
    drag.current = null;
    if (x) x.controls.enabled = true;
    if (!d?.moved) return;
    const at = d.kind === 'box' ? d.mesh.position : new THREE.Vector3(...st.info!.bodies.find((b) => b.name === d.name)!.at).add(new THREE.Vector3(0, 0, 0));
    if (d.kind === 'box') onDrag('box', d.name, [at.x, d.y, at.z]);
    else {
      const m = d.mesh as THREE.Mesh;
      const arr = (m.geometry.getAttribute('position') as THREE.BufferAttribute).array as Float32Array;
      const { offset } = m.userData as { offset: number };
      const dx = arr[0] - d.startPos![3 * offset], dz = arr[2] - d.startPos![3 * offset + 2];
      onDrag('body', d.name, [d.start.x + dx, 0, d.start.z + dz]);
    }
  };

  return (
    <div ref={host} className="view3d" data-testid="scene-canvas">
      <div className="drawarea" style={{ width: size.w, height: size.h }}>
        <canvas ref={canvas} style={{ width: size.w, height: size.h }} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} aria-label="3D scene: drag to orbit, right-drag to pan, wheel to zoom; click a body or box to select it, drag it along the ground in Setup" />
        {children}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------- panels

function Num({ label, value, step = 0.1, onCommit, unit, disabled }: { label: string; value: number; step?: number; onCommit: (v: number) => void; unit?: string; disabled?: boolean }) {
  const [v, setV] = useState(String(value));
  useEffect(() => setV(String(+value.toFixed(4))), [value]);
  return (
    <label className="num">
      <span>{label}</span>
      <input type="number" step={step} value={v} disabled={disabled} onChange={(e) => setV(e.target.value)} onBlur={() => Number(v) !== value && Number.isFinite(Number(v)) && onCommit(Number(v))} onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()} />
      {unit && <span className="muted small">{unit}</span>}
    </label>
  );
}

function Props({ ctx, list }: { ctx: Ctx; list: DesignRow[] }) {
  const st = useScene();
  const info = st.info!;
  const id = st.scenario!;
  const run = (name: string, input: Record<string, unknown>) => tool(name, { scenario: id, ...input }).catch((e) => ctx.toast(e.message, true));
  if (!st.selected) return <p className="muted small">Click a body or a box in the view, or in the lists, to edit it.</p>;
  if (st.selected.kind === 'box') {
    const b = info.boxes.find((x) => x.name === st.selected!.name);
    if (!b) return null;
    const ro = !st.editable;
    const setC = (k: number, v: number) => run('set_box', { box: b.name, centre: b.centre.map((x, i) => (i === k ? v : x)) });
    const setH = (k: number, v: number) => run('set_box', { box: b.name, half: b.half.map((x, i) => (i === k ? v : x)) });
    return (
      <div className="props" aria-label={`Box ${b.name}`}>
        <h3>Box {b.name}</h3>
        <div className="grid3">
          <Num label="x" value={b.centre[0]} onCommit={(v) => setC(0, v)} unit="m" disabled={ro} />
          <Num label="y" value={b.centre[1]} onCommit={(v) => setC(1, v)} unit="m" disabled={ro} />
          <Num label="z" value={b.centre[2]} onCommit={(v) => setC(2, v)} unit="m" disabled={ro} />
          <Num label="half x" value={b.half[0]} onCommit={(v) => setH(0, v)} unit="m" disabled={ro} />
          <Num label="half y" value={b.half[1]} onCommit={(v) => setH(1, v)} unit="m" disabled={ro} />
          <Num label="half z" value={b.half[2]} onCommit={(v) => setH(2, v)} unit="m" disabled={ro} />
          <Num label="yaw" value={b.yaw_deg} step={1} onCommit={(v) => run('set_box', { box: b.name, yaw_deg: v })} unit="°" disabled={ro} />
        </div>
        {st.editable && <button className="btn small danger" onClick={() => run('remove_box', { box: b.name }).then(() => scene.select(null))}>Remove box</button>}
      </div>
    );
  }
  const b = info.bodies.find((x) => x.name === st.selected!.name);
  if (!b) return null;
  const ro = !st.editable;
  return (
    <div className="props" aria-label={`Body ${b.name}`}>
      <h3>Body {b.name}</h3>
      <label className="num">
        <span>design</span>
        <select value={b.design} disabled={ro} onChange={(e) => run('set_body', { body: b.name, design: e.target.value })} aria-label="Design">
          {list.map((d) => <option key={d.name} value={d.name}>{d.name}: {d.title}</option>)}
        </select>
      </label>
      <div className="grid3">
        <Num label="x" value={b.at[0]} onCommit={(v) => run('set_body', { body: b.name, at: [v, b.at[1], b.at[2]] })} unit="m" disabled={ro} />
        <Num label="z" value={b.at[2]} onCommit={(v) => run('set_body', { body: b.name, at: [b.at[0], b.at[1], v] })} unit="m" disabled={ro} />
        <Num label="yaw" value={b.yaw_deg} step={1} onCommit={(v) => run('set_body', { body: b.name, yaw_deg: v })} unit="°" disabled={ro} />
        <Num label="speed" value={b.speed_kmh} step={1} onCommit={(v) => run('set_body', { body: b.name, speed_kmh: v })} unit="km/h" disabled={ro} />
      </div>
      <p className="muted small">{fmt(b.mass_kg, 0)} kg · {b.nodes} nodes · {b.triangles.length / 3} surface triangles</p>
      {st.editable && <DesignMatches ctx={ctx} body={b.name} />}
      {st.editable && info.bodies.length > 1 && <button className="btn small danger" onClick={() => run('remove_body', { body: b.name }).then(() => scene.select(null))}>Remove body</button>}
    </div>
  );
}

function DesignMatches({ ctx, body }: { ctx: Ctx; body: string }) {
  const st = useScene();
  const frames = useMemo(() => currentStore().frames().filter((f) => f.view === 'side' && f.regions.some((r) => r.tracing.instances.some((i) => i.schema === 'vehicle-side' && i.elements.front_wheel?.c && i.elements.rear_wheel?.c))), [ctx.revision, ctx.state]); // eslint-disable-line react-hooks/exhaustive-deps
  const [frame, setFrame] = useState('');
  const [rows, setRows] = useState<DesignMatch[] | null>(null);
  const [busy, setBusy] = useState(false);
  if (!frames.length) return <p className="muted small">Match a design: trace a side view (both wheels) of a "before" frame first.</p>;
  const go = async () => {
    setBusy(true);
    try {
      setRows(await tool<DesignMatch[]>('list_design_matches', { frame: frame || frames[0].id }));
    } catch (e) {
      ctx.toast((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  };
  return (
    <details className="matches" open={!!rows}>
      <summary>Match a design from a tracing</summary>
      <div className="row">
        <select value={frame} onChange={(e) => setFrame(e.target.value)} aria-label="Traced frame">
          {frames.map((f) => <option key={f.id} value={f.id}>{f.id} ({f.moment.label || 'no moment'})</option>)}
        </select>
        <button className="btn small" disabled={busy} onClick={go}>{busy ? 'Ranking…' : 'Rank designs'}</button>
      </div>
      {rows && (
        <table className="table small">
          <thead><tr><th>design</th><th title="root of the weighted mean squared difference of the ratios">dist.</th>{rows[0]?.rows.map((r) => <th key={r.name} title={r.name}>{r.name.split(' / ')[0]}</th>)}<th /></tr></thead>
          <tbody>
            {rows.slice(0, 5).map((m) => (
              <tr key={m.design}>
                <td>{m.design}</td>
                <td>{fmt(m.distance, 3)}</td>
                {m.rows.map((r) => <td key={r.name} title={`traced ${fmt(r.traced, 3)}, weight ${r.weight}`}>{fmt(r.design, 2)}<span className="muted"> / {fmt(r.traced, 2)}</span></td>)}
                <td><button className="btn small" disabled={st.busy !== null} onClick={() => tool('set_body', { scenario: st.scenario, body, design: m.design }).catch((e) => ctx.toast(e.message, true))}>Use</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {rows && <p className="muted small">Ratios over the wheelbase: design / traced (side view). Computed in Rust.</p>}
    </details>
  );
}

function Timeline() {
  const st = useScene();
  const s = st.status;
  if (!s?.present) return <p className="muted small">No recording yet: Simulate records the run (every 4 ms sample); Review then scrubs it.</p>;
  const moments = st.info?.moments ?? [];
  return (
    <div className="timeline" aria-label="Timeline">
      <div className="row">
        <button className="btn small" onClick={() => (st.playing ? scene.pause() : scene.play())} aria-label={st.playing ? 'Pause' : 'Play'}>{st.playing ? '❚❚' : '▶'}</button>
        <button className="btn small" onClick={() => scene.step(-1)} aria-label="Back one sample">◀ 4 ms</button>
        <button className="btn small" onClick={() => scene.step(1)} aria-label="Forward one sample">4 ms ▶</button>
        <button className="btn small" onClick={() => scene.reset()}>Reset</button>
        <label className="row small">speed
          <select value={st.speed} onChange={(e) => scene.setSpeed(Number(e.target.value))} aria-label="Playback speed">
            {[0.02, 0.05, 0.1, 0.25, 0.5, 1].map((x) => <option key={x} value={x}>{x === 1 ? 'real time' : `${x}×`}</option>)}
          </select>
        </label>
        <span className="mono">{fmt(st.ms, 1)} ms</span>
        <span className="muted small">after first contact</span>
      </div>
      <input type="range" className="scrub" min={s.start_ms} max={s.end_ms} step={0.5} value={st.ms} onChange={(e) => scene.seek(Number(e.target.value))} aria-label="Time after first contact (ms)" />
      <div className="ticks small">
        {moments.map((m) => (
          <button key={m.name} className="tick" style={{ left: `${(100 * (m.after_impact_ms - s.start_ms)) / (s.end_ms - s.start_ms)}%` }} onClick={() => scene.seek(m.after_impact_ms)} title={`${m.name}: ${m.after_impact_ms} ms`}>{m.name}</button>
        ))}
      </div>
    </div>
  );
}

function Images({ ctx }: { ctx: Ctx }) {
  const st = useScene();
  const frames = useMemo(() => (st.shot ? currentStore().frames().filter((f) => f.shot === st.shot && f.image) : []), [st.shot, ctx.revision, ctx.state]); // eslint-disable-line react-hooks/exhaustive-deps
  const shots = useMemo(() => currentStore().shots(), [ctx.state]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!shots.length) return null;
  const cur = frames.find((f) => f.id === st.image) ?? null;
  const hash = st.status?.hash;
  return (
    <div className="images" aria-label="Reference images">
      <div className="row">
        <h3>Reference images</h3>
        <select value={st.shot ?? ''} onChange={(e) => scene.setShot(e.target.value || null)} aria-label="Shot">
          <option value="">(no shot)</option>
          {shots.map((s) => <option key={s.id} value={s.id}>{s.title}{s.rubble?.scenario === st.scenario ? ' (linked)' : ''}</option>)}
        </select>
      </div>
      <ul className="imglist">
        {frames.map((f) => {
          const pinned = !!(f.pin && f.camera);
          const check = pinned && hash && f.pin!.recording !== hash;
          return (
            <li key={f.id}>
              <button className={`imgrow${st.image === f.id ? ' on' : ''}`} onClick={() => window.studio.call('select_image', { frame: f.id }).catch((e: Error) => ctx.toast(e.message, true))}>
                {frameImage(f) && <img src={frameImage(f)!} alt="" />}
                <span>{f.id}<br /><span className="muted small">{f.view} · {f.moment.label || '—'}</span></span>
                <span className={`badge ${pinned ? (check ? 'warn' : 'ok') : ''}`}>{pinned ? (check ? 'check after re-simulation' : `pinned ${fmt(f.pin!.at_ms, 0)} ms`) : 'not pinned'}</span>
              </button>
            </li>
          );
        })}
      </ul>
      {cur && (
        <div className="row wrap">
          <button className="btn primary small" disabled={!st.status?.present} onClick={() => window.studio.call('pin_image', { frame: cur.id }).then(() => ctx.toast(`pinned ${cur.id} at ${fmt(scene.state.ms, 1)} ms`)).catch((e: Error) => ctx.toast(e.message, true))}>Pin this image here</button>
          {cur.pin && <button className="btn small" onClick={() => window.studio.call('jump_to_pin', { frame: cur.id })}>Jump to its pin</button>}
          <label className="row small">copy camera from
            <select value="" onChange={(e) => e.target.value && tool('copy_camera', { from: e.target.value, to: cur.id }).then(() => window.studio.call('jump_to_pin', { frame: cur.id })).catch((er) => ctx.toast(er.message, true))} aria-label="Copy the camera from another image">
              <option value="">…</option>
              {frames.filter((f) => f.camera && f.id !== cur.id).map((f) => <option key={f.id} value={f.id}>{f.id}</option>)}
            </select>
          </label>
          <button className="btn small" onClick={() => scene.setImage(null)}>Hide image</button>
        </div>
      )}
    </div>
  );
}

/** The image's traced outline and Rubble's at the shown time through the 3D camera, over the canvas. */
function ImageOverlay({ frame, opacity }: { frame: Frame; opacity: number }) {
  const st = useScene();
  const c = cropOf(frame)!;
  const [rubble, setRubble] = useState<{ elements: Record<string, ElementValue>; loops: Record<string, Xy[][]> } | null>(null);
  const region = frame.camera?.region ? frame.regions.find((r) => r.id === frame.camera!.region) : frame.regions.find((r) => r.tracing.instances.length);
  const inst = region?.tracing.instances[0];
  useEffect(() => {
    if (!st.status?.present || !inst || !scene.engine) return setRubble(null);
    const id = requestAnimationFrame(() => {
      try {
        const schema = currentStore().structure(inst.schema);
        if (schema.rubble_view === 'car_frame') return setRubble(null);
        const o = scene.engine!.studio.outline(schema, { at_ms: st.ms, camera: scene.frameCamera(region?.id ?? null), width: Math.round(c.w), height: Math.round(c.h), body: 0 }) as { elements: Record<string, ElementValue>; loops: Record<string, Xy[][]> };
        setRubble(o);
      } catch {
        setRubble(null);
      }
    });
    return () => cancelAnimationFrame(id);
  }, [st.ms, st.camera, st.status, inst, frame]); // eslint-disable-line react-hooks/exhaustive-deps
  const img = frameImage(frame);
  const colour = (n: string) => currentStore().structure(inst?.schema ?? 'vehicle-side').elements.find((e) => e.name === n)?.colour ?? '#ffcc00';
  return (
    <svg className="overlay" viewBox={`0 0 ${c.w} ${c.h}`} preserveAspectRatio="none" aria-label={`${frame.id}: its traced outline (colours) and Rubble's at ${fmt(st.ms, 1)} ms (green)`}>
      {img && frame.image && opacity > 0 && (
        <image href={img} x={-c.crop[0] * frame.image.width} y={-c.crop[1] * frame.image.height} width={frame.image.width} height={frame.image.height} opacity={opacity} />
      )}
      {rubble && <Overlay inst={{ ...rubble, width: c.w, height: c.h, front: 'right' }} w={c.w} h={c.h} colour={() => '#3ddc84'} />}
      {inst && <Overlay inst={{ elements: inst.elements, loops: {}, width: c.w, height: c.h, front: inst.front }} w={c.w} h={c.h} colour={colour} />}
    </svg>
  );
}

function CameraPanel() {
  const st = useScene();
  const c = st.camera;
  // Roll: the up vector turned about the view direction.
  const roll = useMemo(() => {
    const f = new THREE.Vector3(...c.target).sub(new THREE.Vector3(...c.eye)).normalize();
    const right = new THREE.Vector3().crossVectors(f, new THREE.Vector3(0, 1, 0)).normalize();
    const up0 = new THREE.Vector3().crossVectors(right, f);
    const up = new THREE.Vector3(...c.up);
    return (Math.atan2(up.dot(right), up.dot(up0)) * 180) / Math.PI;
  }, [c]);
  const setRoll = (deg: number) => {
    const f = new THREE.Vector3(...c.target).sub(new THREE.Vector3(...c.eye)).normalize();
    const right = new THREE.Vector3().crossVectors(f, new THREE.Vector3(0, 1, 0)).normalize();
    const up0 = new THREE.Vector3().crossVectors(right, f);
    const r = (deg * Math.PI) / 180;
    const up = up0.multiplyScalar(Math.cos(r)).add(right.multiplyScalar(Math.sin(r)));
    scene.setCamera({ ...c, up: [up.x, up.y, up.z] });
  };
  return (
    <div className="campanel" aria-label="Camera">
      <label className="row small">fov <input type="range" min={10} max={100} step={0.5} value={c.fov_y_deg} onChange={(e) => scene.setCamera({ ...c, fov_y_deg: Number(e.target.value) })} aria-label="Vertical field of view (degrees)" /> <span className="mono">{fmt(c.fov_y_deg, 1)}°</span></label>
      <label className="row small">roll <input type="range" min={-30} max={30} step={0.25} value={roll} onChange={(e) => setRoll(Number(e.target.value))} aria-label="Roll (degrees)" /> <span className="mono">{fmt(roll, 1)}°</span></label>
      <span className="muted small mono" title="eye → target">eye {c.eye.map((x) => fmt(x, 2)).join(', ')} → {c.target.map((x) => fmt(x, 2)).join(', ')}</span>
    </div>
  );
}

export function SceneView({ ctx }: { ctx: Ctx }) {
  const st = useScene();
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([]);
  const [list, setList] = useState<DesignRow[]>([]);
  const [opacity, setOpacity] = useState(0.4);
  useEffect(() => {
    tool<ScenarioInfo[]>('list_scenarios').then(setScenarios).catch((e) => ctx.toast(e.message, true));
    designs().then(setList);
  }, [ctx.state]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!st.scenario && !st.busy) {
      const first = currentStore().scenarios()[0];
      if (first) window.studio.call('open_scene', { scenario: first.id }).catch((e: Error) => ctx.toast(e.message, true));
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const frame = st.image ? (() => {
    try {
      return currentStore().frame(st.image);
    } catch {
      return null;
    }
  })() : null;
  const crop = cropOf(frame);
  const onDrag = useCallback((kind: 'body' | 'box', name: string, at: [number, number, number]) => {
    const round = at.map((x) => Math.round(x * 1000) / 1000);
    (kind === 'box' ? tool('set_box', { scenario: scene.state.scenario, box: name, centre: round }) : tool('set_body', { scenario: scene.state.scenario, body: name, at: round })).catch((e) => ctx.toast(e.message, true));
  }, [ctx]);
  const info = st.info;
  const s = st.status;
  const recLabel = !s?.present ? 'no recording' : s.fresh ? `recorded · ${s.executor === 'webgpu' ? 'WebGPU' : 'CPU fallback (tests only)'} · ${s.samples} samples · ${fmt(s.duration_s, 2)} s` : 'stale: the setup changed since this recording · re-simulate';
  return (
    <div className="sceneview">
      <div className="scenebar">
        <label className="row small">
          <span className="sr-only">Scenario</span>
          <select value={st.scenario ?? ''} onChange={(e) => e.target.value && window.studio.call('open_scene', { scenario: e.target.value }).catch((er: Error) => ctx.toast(er.message, true))} aria-label="Scenario">
            <option value="">Open a scenario…</option>
            <optgroup label="Recreated (data, editable)">{scenarios.filter((x) => x.recreated).map((x) => <option key={x.id} value={x.id}>{x.id}: {x.title}</option>)}</optgroup>
            <optgroup label="Built-in (Rust, read-only)">{scenarios.filter((x) => !x.recreated).map((x) => <option key={x.id} value={x.id}>{x.id}: {x.title}</option>)}</optgroup>
          </select>
        </label>
        <button className="btn small" onClick={() => tool<{ id: string }>('create_scenario', { title: 'Recreated crash', ...(st.scenario && !st.editable ? { from: st.scenario, title: `${st.scenario} recreated` } : {}) }).then((d) => window.studio.call('open_scene', { scenario: d.id })).catch((e) => ctx.toast(e.message, true))}>
          {st.scenario && !st.editable ? 'Copy as data to edit' : 'New scenario'}
        </button>
        <div className="seg" role="group" aria-label="Mode">
          <button aria-pressed={st.mode === 'setup'} onClick={() => scene.setMode('setup')}>Setup</button>
          <button aria-pressed={st.mode === 'review'} disabled={!s?.present} onClick={() => scene.setMode('review')} title={s?.present ? '' : 'Simulate first'}>Review</button>
        </div>
        <button className="btn primary" disabled={!info || !!st.busy} onClick={() => window.studio.call('simulate', {}).catch((e: Error) => ctx.toast(e.message, true))}>
          {s?.present && !s.fresh ? 'Re-simulate' : 'Simulate'}
        </button>
        <span className={`small ${s?.present && !s.fresh ? 'warnt' : 'muted'}`} data-testid="rec-status">{recLabel}</span>
        {st.webgpu === false && <span className="badge warn" title="The studio's solver is WebGPU">no WebGPU in this browser{st.cpuFallback ? ' · CPU fallback (tests)' : ''}</span>}
        {st.cpuFallback && st.webgpu !== false && <span className="badge warn">CPU fallback on (tests)</span>}
      </div>
      {(st.busy || st.error) && (
        <div className={`notice ${st.error ? 'bad' : ''}`} role="status">
          {st.busy}{st.sim ? ` · sample ${st.sim.sample} of ${st.sim.samples} (${st.sim.stage})` : ''}{st.error}
        </div>
      )}
      <div className="scenemain">
        <div className="sc-stage">
          <View3D frameAspect={crop ? crop.w / crop.h : null} onPick={(sel) => scene.select(sel)} onDrag={onDrag}>
            {frame && crop && <ImageOverlay frame={frame} opacity={opacity} />}
          </View3D>
          <div className="sc-foot">
            {st.mode === 'review' ? <Timeline /> : <p className="muted small">Setup: drag a body or a box along the ground, or edit its numbers. {info && `First contact at ${fmt(info.impact_s * 1000, 1)} ms (Rust, from the layout) · placed in ${fmt(st.placeMs ?? 0, 1)} ms (edit to redraw ${fmt(st.editMs ?? 0, 0)} ms)`}</p>}
            <CameraPanel />
            {frame && <label className="row small">image <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} aria-label="Reference image opacity" /></label>}
          </div>
        </div>
        <aside className="side sceneside" aria-label="Scene">
          {info && (
            <>
              <div className="row">
                <h2 title={info.id}>{info.title}</h2>
                {st.editable && <span className="small muted">undo {st.undo} · redo {st.redo}</span>}
                {st.editable && <button className="btn small" disabled={!st.undo} onClick={() => window.studio.call('undo_scene', {})}>Undo</button>}
                {st.editable && <button className="btn small" disabled={!st.redo} onClick={() => window.studio.call('redo_scene', {})}>Redo</button>}
              </div>
              {!st.editable && <p className="muted small">A built-in scenario (Rust): read-only. "Copy as data to edit" makes a recreation of it.</p>}
              <h3>Bodies {st.editable && <button className="btn small" onClick={() => tool('add_body', { scenario: st.scenario, at: [4, 0, 0] }).catch((e) => ctx.toast(e.message, true))}>+</button>}</h3>
              <ul className="objlist">
                {info.bodies.map((b) => (
                  <li key={b.name}><button className={st.selected?.kind === 'body' && st.selected.name === b.name ? 'on' : ''} onClick={() => scene.select({ kind: 'body', name: b.name })}>{b.name} · {b.design} · {fmt(b.speed_kmh, 1)} km/h</button></li>
                ))}
              </ul>
              <h3>Boxes {st.editable && <button className="btn small" onClick={() => tool('add_box', { scenario: st.scenario }).catch((e) => ctx.toast(e.message, true))}>+</button>}</h3>
              <ul className="objlist">
                {info.boxes.map((b) => (
                  <li key={b.name}><button className={st.selected?.kind === 'box' && st.selected.name === b.name ? 'on' : ''} onClick={() => scene.select({ kind: 'box', name: b.name })}>{b.name} · {b.half.map((x) => fmt(2 * x, 2)).join(' × ')} m</button></li>
                ))}
                {!info.boxes.length && <li className="muted small">none</li>}
              </ul>
              <Props ctx={ctx} list={list} />
              <Images ctx={ctx} />
            </>
          )}
          {!info && !st.busy && <p className="muted">Open a scenario, or make a new one.</p>}
        </aside>
      </div>
    </div>
  );
}
