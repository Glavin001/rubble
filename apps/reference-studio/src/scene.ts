// The Scene view's state, outside React: the open scenario placed in the engine, the mode
// (Setup: edit the scenario; Review: the recording's timeline and a free camera), the
// recording's status, the playback clock, the 3D camera, the selection and the image
// overlaid. Data edits arrive through the store (the data tools); this re-places the scene in
// Rust after each one (a box or a pose is live; a design change loads that design first).
// Playback, scrubbing and camera moves only read the recording: nothing re-simulates.

import type { FrameCamera, RecordingStatus, ScenarioDesc, SceneInfo, SceneSource } from '../shared/model.ts';
import { StudioError, type Store } from '../shared/store.ts';
import { Engine, hasWebGpu, loadEngine, type SimProgress } from './engine.ts';

export type Mode = 'setup' | 'review';
export interface View3 {
  eye: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
  fov_y_deg: number;
}

export interface SceneState {
  scenario: string | null;
  /** A data scenario (editable) or a built-in one (read-only: copy it as data to edit). */
  editable: boolean;
  shot: string | null;
  mode: Mode;
  info: SceneInfo | null;
  status: RecordingStatus | null;
  /** Shown time (ms after first contact). */
  ms: number;
  playing: boolean;
  /** Simulated seconds per real second. */
  speed: number;
  selected: { kind: 'body' | 'box'; name: string } | null;
  /** What the engine is doing (placing, loading a design, simulating), or null. */
  busy: string | null;
  sim: SimProgress | null;
  error: string | null;
  /** The last placement's cost: Rust's place (ms) and the whole edit to redraw (ms). */
  placeMs: number | null;
  editMs: number | null;
  /** The reference image overlaid on the 3D view. */
  image: string | null;
  camera: View3;
  /** Bumped when the camera is set from outside the 3D view (a pin, a tool). */
  cameraVersion: number;
  webgpu: boolean | null;
  /** Use the wasm CPU reference instead of WebGPU (automated tests only; labelled). */
  cpuFallback: boolean;
  undo: number;
  redo: number;
}

const DEFAULT_VIEW: View3 = { eye: [7, 2.2, -2], target: [0, 0.7, 1.5], up: [0, 1, 0], fov_y_deg: 60 };

export class SceneController {
  store: Store | null = null;
  engine: Engine | null = null;
  state: SceneState = {
    scenario: null, editable: false, shot: null, mode: 'setup', info: null, status: null, ms: 0, playing: false, speed: 0.1, selected: null, busy: null, sim: null, error: null,
    placeMs: null, editMs: null, image: null, camera: DEFAULT_VIEW, cameraVersion: 0, webgpu: null, cpuFallback: new URLSearchParams(location.search).has('cpu-fallback'), undo: 0, redo: 0,
  };
  /** Positions drawn now (every body's nodes, x y z) and the beam flags. */
  positions: Float32Array | null = null;
  private listeners = new Set<() => void>();
  private undoStack: ScenarioDesc[] = [];
  private redoStack: ScenarioDesc[] = [];
  private lastDesc: string | null = null;
  private placing: Promise<void> | null = null;
  private again = false;
  private raf = 0;

  subscribe(f: () => void): () => void {
    this.listeners.add(f);
    return () => this.listeners.delete(f);
  }
  private set(p: Partial<SceneState>): void {
    this.state = { ...this.state, ...p };
    for (const f of this.listeners) f();
  }

  attach(store: Store): void {
    this.store = store;
  }

  async ready(): Promise<Engine> {
    if (!this.engine) {
      await loadEngine();
      this.engine = new Engine();
      hasWebGpu().then((w) => this.set({ webgpu: w }));
    }
    return this.engine;
  }

  private source(): SceneSource {
    const id = this.state.scenario!;
    if (this.store!.hasScenario(id)) return { desc: this.store!.scenario(id) };
    return { id };
  }

  /** Open a scenario (data or built-in) in the scene, its cached recording loaded when there is one. */
  async open(scenario: string, shot: string | null = null): Promise<SceneInfo> {
    const e = await this.ready();
    const editable = this.store!.hasScenario(scenario);
    const linked = shot ?? this.store!.shots().find((s) => s.rubble?.scenario === scenario)?.id ?? null;
    this.undoStack = [];
    this.redoStack = [];
    this.lastDesc = editable ? JSON.stringify(this.store!.scenario(scenario)) : null;
    this.set({ scenario, editable, shot: linked, selected: null, image: null, error: null, playing: false, undo: 0, redo: 0 });
    await this.replace(true);
    const info = this.state.info!;
    if (!this.state.status?.fresh) {
      const cached = await this.store!.recording(info.hash).catch(() => null);
      if (cached) {
        try {
          this.set({ status: e.studio.loadRecording(cached, info.hash) });
        } catch {
          /* a recording of an older layout: ignored */
        }
      }
    }
    if (this.state.status?.fresh) this.seek(this.state.status.start_ms);
    const cam = info.cameras[0];
    if (cam) this.setCamera({ eye: cam.eye, target: cam.target, up: [0, 1, 0], fov_y_deg: 60 });
    return info;
  }

  /** Place the scene again from the scenario as it is now. */
  async replace(first = false): Promise<void> {
    if (this.placing) {
      this.again = true;
      return this.placing;
    }
    this.placing = (async () => {
      do {
        this.again = false;
        const e = await this.ready();
        const t0 = performance.now();
        const src = this.source();
        const missing = e.studio.missing(src);
        if (missing.length) this.set({ busy: `respawning: loading ${missing.join(', ')}` });
        try {
          const info = await e.place(src, (s) => this.set({ busy: `respawning: ${s}` }));
          const status = e.studio.recordingStatus();
          this.positions = e.studio.spawnPositions();
          const sel = this.state.selected;
          const keep = sel && (sel.kind === 'body' ? info.bodies.some((b) => b.name === sel.name) : info.boxes.some((b) => b.name === sel.name));
          this.set({ info, status, busy: null, error: null, placeMs: info.place_ms, editMs: performance.now() - t0, selected: keep ? sel : null, ...(first ? {} : { mode: this.state.mode }) });
          if (this.state.mode === 'review' && status.present) this.showAt(this.state.ms);
        } catch (err) {
          this.set({ busy: null, error: (err as Error).message });
        }
      } while (this.again);
    })();
    try {
      await this.placing;
    } finally {
      this.placing = null;
    }
  }

  /** The store changed: if it is the open data scenario, re-place it (and keep an undo step). */
  onStoreChange(): void {
    const id = this.state.scenario;
    if (!id || !this.store?.hasScenario(id)) return;
    const now = JSON.stringify(this.store.scenario(id));
    if (now === this.lastDesc) return;
    if (this.lastDesc && !this.undoing) {
      this.undoStack.push(JSON.parse(this.lastDesc));
      if (this.undoStack.length > 200) this.undoStack.shift();
      this.redoStack = [];
    }
    this.lastDesc = now;
    this.set({ undo: this.undoStack.length, redo: this.redoStack.length });
    if (this.state.mode === 'review') this.set({ mode: 'setup', playing: false });
    void this.replace();
  }

  private undoing = false;
  /** Undo or redo a scenario edit (through the store, as any edit). */
  history(dir: 'undo' | 'redo'): ScenarioDesc | null {
    const id = this.state.scenario;
    if (!id || !this.store?.hasScenario(id)) return null;
    const from = dir === 'undo' ? this.undoStack : this.redoStack;
    const to = dir === 'undo' ? this.redoStack : this.undoStack;
    const d = from.pop();
    if (!d) return null;
    to.push(this.store.scenario(id));
    this.undoing = true;
    try {
      this.store.putScenario(d);
    } finally {
      this.undoing = false;
    }
    this.set({ undo: this.undoStack.length, redo: this.redoStack.length });
    return d;
  }

  setMode(mode: Mode): void {
    if (mode === 'review' && !this.state.status?.present) throw new StudioError('no recording yet: simulate first');
    this.set({ mode, playing: false });
    if (mode === 'review') this.showAt(this.state.ms);
    else this.positions = this.engine?.studio.spawnPositions() ?? this.positions;
    this.set({});
  }

  /** Simulate the scene and record it (cached in the workspace by its hash). */
  async simulate(opts: { cpu?: boolean } = {}): Promise<RecordingStatus> {
    const e = await this.ready();
    if (!this.state.info) throw new StudioError('no scene open');
    const cpu = opts.cpu ?? this.state.cpuFallback;
    if (!cpu && !(await hasWebGpu())) throw new StudioError('this browser has no WebGPU: the studio simulates on the GPU (Chrome or Edge with WebGPU). The CPU fallback is for automated tests only (?cpu-fallback).');
    this.set({ busy: cpu ? 'simulating (CPU fallback, for tests)' : 'simulating on WebGPU', playing: false, error: null });
    try {
      const t0 = performance.now();
      const status = await e.simulate({ cpu, onProgress: (p) => this.set({ sim: p }) });
      const bytes = e.studio.recordingBytes();
      await this.store!.putRecording(status.hash, bytes).catch(() => undefined);
      this.set({ status, busy: null, sim: null, mode: 'review', error: null });
      this.seek(this.state.ms < status.start_ms || this.state.ms > status.end_ms ? status.start_ms : this.state.ms);
      console.info(`simulated in ${((performance.now() - t0) / 1000).toFixed(1)} s (${status.executor})`);
      return status;
    } catch (err) {
      this.set({ busy: null, sim: null, error: (err as Error).message });
      throw err;
    }
  }

  /** The scenario open with a fresh recording (for the comparison and outlines). */
  async ensureRecorded(scenario: string): Promise<{ info: SceneInfo; status: RecordingStatus }> {
    if (this.state.scenario !== scenario || !this.state.info) await this.open(scenario);
    if (!this.state.status?.fresh) await this.simulate();
    return { info: this.state.info!, status: this.state.status! };
  }

  private showAt(ms: number): void {
    if (!this.engine || !this.state.status?.present) return;
    this.positions = this.engine.studio.positionsAtMs(ms);
  }

  seek(ms: number): void {
    const st = this.state.status;
    const t = st?.present ? Math.min(st.end_ms, Math.max(st.start_ms, ms)) : 0;
    this.showAt(t);
    this.set({ ms: t });
  }

  step(samples: number): void {
    const st = this.state.status;
    if (!st?.present) return;
    const k = Math.round((this.state.ms - st.start_ms) / (st.sample_s * 1000)) + samples;
    this.seek(st.start_ms + k * st.sample_s * 1000);
  }

  play(): void {
    if (!this.state.status?.present) throw new StudioError('no recording yet: simulate first');
    if (this.state.mode !== 'review') this.setMode('review');
    if (this.state.ms >= this.state.status.end_ms) this.seek(this.state.status.start_ms);
    this.set({ playing: true });
    let last = performance.now();
    const tick = (now: number) => {
      if (!this.state.playing) return;
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const ms = this.state.ms + dt * this.state.speed * 1000;
      this.seek(ms);
      if (ms >= this.state.status!.end_ms) this.set({ playing: false });
      else this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  pause(): void {
    cancelAnimationFrame(this.raf);
    this.set({ playing: false });
  }

  reset(): void {
    this.pause();
    if (this.state.status?.present) this.seek(this.state.status.start_ms);
    else this.positions = this.engine?.studio.spawnPositions() ?? null;
    this.set({});
  }

  setSpeed(speed: number): void {
    if (!(speed > 0 && speed <= 4)) throw new StudioError('speed is in (0, 4]');
    this.set({ speed });
  }

  select(sel: SceneState['selected']): void {
    this.set({ selected: sel });
  }

  /** The 3D view moved its camera (by hand). */
  cameraMoved(c: View3): void {
    this.state = { ...this.state, camera: c };
    for (const f of this.listeners) f();
  }

  /** Set the camera from outside the 3D view. */
  setCamera(c: View3): void {
    this.set({ camera: c, cameraVersion: this.state.cameraVersion + 1 });
  }

  setShot(shot: string | null): void {
    this.set({ shot, image: null });
  }

  setImage(frame: string | null): void {
    this.set({ image: frame });
  }

  /** The current camera as an image's camera (region: the frame's camera region kept). */
  frameCamera(region: string | null): FrameCamera {
    const c = this.state.camera;
    return { eye: c.eye, target: c.target, up: c.up, fov_y_deg: c.fov_y_deg, region };
  }
}

export const scene = new SceneController();
