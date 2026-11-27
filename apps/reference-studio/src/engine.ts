// The engine in the page: Rust compiled to wasm (engine-world-wasm, the crash lab's module,
// served from public/rubble/ by scripts/engine-assets.ts). This file only loads it, fetches
// the designs' files it asks for and moves bytes between it, the GPU run and the workspace.
// Everything computed is Rust: placing a scene, first contact, the GPU run's sampling and
// report, positions at a time, outlines, the comparison, design ranking (types generated
// from the Rust: src/generated/crash-reference.ts).

import type {
  Camera,
  CompareSource,
  Contact,
  CpuProgress,
  DesignMatch,
  Instance,
  Moment,
  RecordingStatus,
  Registry,
  RubbleSpec,
  ScenarioDesc,
  SceneInfo,
  SceneSource,
} from './generated/crash-reference.ts';
import type { StructureSchema } from '../shared/model.ts';

// The wasm module's calls the studio uses (wasm-bindgen's own declarations are in the
// built module's .d.ts, which is not committed).
interface CrashStudioWasm {
  missing(src: SceneSource): string[];
  addTemplate(design: string, files: [string, string][] | null): void;
  hasTemplate(design: string): boolean;
  place(src: SceneSource): SceneInfo;
  spawnPositions(): Float32Array;
  exportBuiltin(id: string): ScenarioDesc;
  cpuStart(): void;
  cpuStep(samples: number): CpuProgress;
  loadRecording(bytes: Uint8Array, hash?: string | null): RecordingStatus;
  recordingBytes(): Uint8Array;
  recordingStatus(): RecordingStatus;
  report(): unknown;
  positionsAtMs(ms: number): Float32Array;
  brokenAtMs(ms: number): Uint8Array;
  outline(schema: unknown, spec: RubbleSpec): Record<string, unknown>;
  compare(schema: unknown, sources: CompareSource[]): Record<string, unknown>;
  rankDesigns(schema: unknown, instance: Instance, width: number, height: number): DesignMatch[];
}
interface ExplicitSimWasm {
  frame(steps: number): void;
  crashSample(): Promise<number | null>;
  crashReport(): unknown;
  crashInfo(): { samples: number; sampleSteps: number; impactS: number } | null;
  crashTakeRecording(): Uint8Array | null;
  free(): void;
}
interface WebWorldWasm {
  explicitSimAsync(spec: unknown, progress?: (line: string) => void): Promise<ExplicitSimWasm>;
}
interface WorldModule {
  default: (path?: unknown) => Promise<unknown>;
  CrashStudio: new () => CrashStudioWasm;
  WebWorld: { create(def: unknown): Promise<WebWorldWasm> };
  studioRegistry(data: ScenarioDesc[]): Registry;
  studioCheckDesc(desc: ScenarioDesc): void;
  studioValidId(id: string): void;
  studioPick(moments: Moment[], cameras: Camera[], scenario: string, label: string, view: string): { moment: string | null; camera: string | null };
  lastPanic(): string;
}

const BASE = import.meta.env.BASE_URL;
let mod: WorldModule | null = null;
let loading: Promise<WorldModule> | null = null;

/** The wasm module, loaded once. */
export function loadEngine(): Promise<WorldModule> {
  loading ??= (async () => {
    const url = new URL(`${BASE}rubble/world-wasm/engine_world_wasm.js`, location.href).href;
    // A plain dynamic import the bundler leaves alone: the module is a static file of the
    // site (public/), not part of the bundle.
    const load = new Function('u', 'return import(u)') as (u: string) => Promise<WorldModule>;
    const m = await load(url);
    await m.default({ module_or_path: new URL(`${BASE}rubble/world-wasm/engine_world_wasm_bg.wasm`, location.href).href });
    mod = m;
    return m;
  })();
  return loading;
}

export function engineModule(): WorldModule {
  if (!mod) throw new Error('the engine (wasm) is not loaded yet');
  return mod;
}

export interface DesignRow {
  name: string;
  title: string;
  folder: string;
}

let designList: Promise<DesignRow[]> | null = null;
/** The generated designs the static build carries (public/designs/designs.json) plus the
 * barrier, which is built from code. */
export function designs(): Promise<DesignRow[]> {
  designList ??= fetch(`${BASE}designs/designs.json`)
    .then((r) => (r.ok ? (r.json() as Promise<DesignRow[]>) : []))
    .catch(() => [] as DesignRow[])
    .then((l) => [...l, { name: 'mdb214', title: 'FMVSS 214 moving deformable barrier (from code)', folder: '' }]);
  return designList;
}

/** Whether this browser has WebGPU (the studio's solver; without it only the labelled CPU fallback runs). */
export async function hasWebGpu(): Promise<boolean> {
  const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return false;
  try {
    return !!(await gpu.requestAdapter());
  } catch {
    return false;
  }
}

export type SimProgress = { sample: number; samples: number; executor: 'webgpu' | 'cpu-fallback'; stage: string };

/** The studio's scene in the engine: one at a time. */
export class Engine {
  studio: CrashStudioWasm;
  private world: Promise<WebWorldWasm> | null = null;
  scene: SceneInfo | null = null;
  source: SceneSource | null = null;
  constructor() {
    this.studio = new (engineModule().CrashStudio)();
  }

  /** Load the templates `src` needs (a fetch and a Rust parse per design not yet loaded). */
  async ensureTemplates(src: SceneSource, say: (s: string) => void = () => {}): Promise<number> {
    const t0 = performance.now();
    const list = await designs();
    for (const d of this.studio.missing(src)) {
      say(`loading design ${d}`);
      if (d === 'mdb214') this.studio.addTemplate(d, null);
      else {
        const row = list.find((x) => x.name === d);
        if (!row) throw new Error(`design ${d} is not in this build (public/designs/: node scripts/engine-assets.ts designs)`);
        const r = await fetch(`${BASE}designs/${row.folder}/physics.json`);
        if (!r.ok) throw new Error(`designs/${row.folder}/physics.json: ${r.status}`);
        this.studio.addTemplate(d, (await r.json()) as [string, string][]);
      }
    }
    return performance.now() - t0;
  }

  /** Place a scene (templates first). */
  async place(src: SceneSource, say?: (s: string) => void): Promise<SceneInfo> {
    await this.ensureTemplates(src, say);
    this.scene = this.studio.place(src);
    this.source = src;
    return this.scene;
  }

  /** Every design's template loaded (for ranking). */
  async loadAllDesigns(say?: (s: string) => void): Promise<void> {
    for (const d of await designs()) await this.ensureTemplates({ desc: { id: 'x', title: '', notes: '', bodies: [{ name: 'b', design: d.name, at: [0, 0, 0], yaw_deg: 0, speed_kmh: 0, carrier: false }], boxes: [], cameras: [], moments: [], secs_after_impact: 1, contact: 'edge' } }, say);
  }

  /** Run the placed scene to its end and record it: on the page's WebGPU (the crash lab's
   * ExplicitSim, `crashRef` with `record`), or the wasm CPU reference (`cpu`: the automated
   * smoke test's fallback, labelled `cpu-fallback`). Returns the recording's status. */
  async simulate(opts: { cpu?: boolean; onProgress?: (p: SimProgress) => void; cancelled?: () => boolean } = {}): Promise<RecordingStatus> {
    if (!this.scene || !this.source) throw new Error('no scene placed');
    const hash = this.scene.hash;
    const say = (p: Omit<SimProgress, 'executor'> & { executor?: SimProgress['executor'] }) => opts.onProgress?.({ executor: opts.cpu ? 'cpu-fallback' : 'webgpu', ...p });
    if (opts.cpu) {
      this.studio.cpuStart();
      for (;;) {
        const p = this.studio.cpuStep(5);
        say({ sample: p.sample, samples: p.samples, stage: 'cpu' });
        if (p.done) break;
        if (opts.cancelled?.()) throw new Error('cancelled');
        await new Promise((r) => setTimeout(r, 0));
      }
      return this.studio.recordingStatus();
    }
    const m = engineModule();
    this.world ??= m.WebWorld.create({});
    const world = await this.world;
    say({ sample: 0, samples: this.scene.samples, stage: 'compiling kernels' });
    const files: Record<string, [string, string][]> = {};
    const list = await designs();
    for (const b of this.scene.bodies) {
      if (b.design === 'mdb214' || files[b.design]) continue;
      const row = list.find((x) => x.name === b.design)!;
      files[b.design] = (await (await fetch(`${BASE}designs/${row.folder}/physics.json`)).json()) as [string, string][];
    }
    const spec = { crashRef: { ...this.source, run: 0, contact: this.scene.contact, designs: files, record: true }, timing: false };
    const sim = await world.explicitSimAsync(spec, (line) => say({ sample: 0, samples: this.scene!.samples, stage: line.split(' ')[0] }));
    try {
      const info = sim.crashInfo()!;
      const pending: Promise<unknown>[] = [];
      for (let k = 0; k < info.samples; k++) {
        if (k > 0) sim.frame(info.sampleSteps);
        pending.push(sim.crashSample());
        if (pending.length >= 16) {
          await Promise.all(pending.splice(0));
          say({ sample: k + 1, samples: info.samples, stage: 'webgpu' });
          if (opts.cancelled?.()) throw new Error('cancelled');
        }
      }
      await Promise.all(pending);
      const bytes = sim.crashTakeRecording();
      if (!bytes) throw new Error(`the GPU run gave no recording${m.lastPanic() ? `: ${m.lastPanic()}` : ''}`);
      return this.studio.loadRecording(bytes, hash);
    } finally {
      sim.free();
    }
  }
}

export type { Contact };
export type { StructureSchema };
