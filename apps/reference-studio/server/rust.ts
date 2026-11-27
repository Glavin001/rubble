// The Rust side: the shape maths (`reference_shape --serve`, one long-lived process) and
// Rubble's outlines (`crash_outlines`, run on demand, cached). The server never computes
// geometry itself; it passes data between the files and these tools.

import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { createInterface } from 'node:readline';
import { ENGINE, StudioError } from './store.ts';

const EXAMPLES = join(ENGINE, 'target', 'release', 'examples');
const SOURCES = [
  'crates/engine-softbody/src/crash_reference/shape/mod.rs',
  'crates/engine-softbody/src/crash_reference/shape/measures.rs',
  'crates/engine-softbody/src/crash_reference/shape/rubble.rs',
  'crates/engine-softbody/src/crash_reference/shape/link.rs',
  'crates/engine-softbody/src/crash_reference/outline.rs',
  'crates/engine-softbody-lab/examples/reference_shape.rs',
  'crates/engine-softbody-lab/examples/crash_outlines.rs',
];

function run(cmd: string, args: string[], cwd: string, timeoutMs: number): Promise<string> {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { cwd, stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    let err = '';
    const timer = setTimeout(() => p.kill('SIGKILL'), timeoutMs);
    p.stdout.on('data', (d) => (out += d));
    p.stderr.on('data', (d) => (err += d));
    p.on('error', (e) => {
      clearTimeout(timer);
      reject(e);
    });
    p.on('close', (code) => {
      clearTimeout(timer);
      if (code === 0) resolve(out);
      else reject(new StudioError(`${cmd} ${args.join(' ')} failed (${code}): ${err.slice(-1500)}`, 500));
    });
  });
}

/** The release examples, rebuilt when a source is newer than the binary. */
async function ensureBuilt(log: (s: string) => void): Promise<void> {
  const bins = ['reference_shape', 'crash_outlines'].map((b) => join(EXAMPLES, b));
  const newest = Math.max(...SOURCES.map((s) => statSync(join(ENGINE, s)).mtimeMs));
  if (bins.every((b) => existsSync(b) && statSync(b).mtimeMs >= newest)) return;
  log('building the Rust tools (cargo build --release -p engine-softbody-lab --example reference_shape --example crash_outlines)...');
  await run('cargo', ['build', '--release', '-p', 'engine-softbody-lab', '--example', 'reference_shape', '--example', 'crash_outlines'], ENGINE, 15 * 60_000);
}

export class Rust {
  private proc: ChildProcessWithoutNullStreams | null = null;
  private pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
  private next = 1;
  private building: Promise<void> | null = null;
  private outlineJobs = new Map<string, Promise<string>>();
  log: (s: string) => void;
  cache: string;

  constructor(cache: string, log: (s: string) => void = (s) => console.log(`[reference-studio] ${s}`)) {
    this.cache = cache;
    this.log = log;
  }

  private async built(): Promise<void> {
    this.building ??= ensureBuilt(this.log).finally(() => (this.building = null));
    await this.building;
  }

  private async start(): Promise<ChildProcessWithoutNullStreams> {
    if (this.proc && this.proc.exitCode === null) return this.proc;
    await this.built();
    const p = spawn(join(EXAMPLES, 'reference_shape'), ['--serve'], { cwd: ENGINE });
    createInterface({ input: p.stdout }).on('line', (line) => {
      const r = JSON.parse(line) as { id: number; ok: boolean; result?: unknown; error?: string };
      const w = this.pending.get(r.id);
      if (!w) return;
      this.pending.delete(r.id);
      if (r.ok) w.resolve(r.result);
      else w.reject(new StudioError(`reference_shape: ${r.error}`, 422));
    });
    p.stderr.on('data', (d) => this.log(`reference_shape: ${d}`));
    p.on('exit', () => {
      for (const w of this.pending.values()) w.reject(new StudioError('reference_shape exited', 500));
      this.pending.clear();
      this.proc = null;
    });
    this.proc = p;
    return p;
  }

  /** One request to the Rust shape service. */
  async call<T>(op: string, body: Record<string, unknown> = {}): Promise<T> {
    const p = await this.start();
    const id = this.next++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
      p.stdin.write(JSON.stringify({ id, op, ...body }) + '\n');
    });
  }

  private scenarioCache: unknown[] | null = null;
  async scenarios<T>(): Promise<T[]> {
    this.scenarioCache ??= await this.call<unknown[]>('scenarios');
    return this.scenarioCache as T[];
  }

  /**
   * Rubble's outlines for a scenario run, with extra (per-frame) cameras: the folder holding
   * outlines.json and the stills, made by crash_outlines when not cached. Keyed by the
   * scenario, contact and the cameras, so a moved camera makes a new folder.
   */
  async outlines(scenario: string, contact: string, cameras: unknown[]): Promise<string> {
    const camText = JSON.stringify(cameras);
    const key = cameras.length ? '-' + createHash('sha256').update(camText).digest('hex').slice(0, 12) : '';
    const dir = join(this.cache, 'outlines', `${scenario}-${contact}${key}`);
    const file = join(dir, 'outlines.json');
    if (existsSync(file) && this.fresh(file)) return dir;
    const job = this.outlineJobs.get(dir);
    if (job) return job;
    const p = (async () => {
      await this.built();
      mkdirSync(dir, { recursive: true });
      const args = [scenario, '--contact', contact, '--out', dir];
      if (cameras.length) {
        writeFileSync(join(dir, 'cameras.json'), camText);
        args.push('--cameras', join(dir, 'cameras.json'));
      }
      this.log(`crash_outlines ${args.join(' ')}`);
      await run(join(EXAMPLES, 'crash_outlines'), args, ENGINE, 5 * 60_000);
      return dir;
    })().finally(() => this.outlineJobs.delete(dir));
    this.outlineJobs.set(dir, p);
    return p;
  }

  /** Cached outlines older than the binary that made them are made again. */
  private fresh(file: string): boolean {
    const bin = join(EXAMPLES, 'crash_outlines');
    return !existsSync(bin) || statSync(file).mtimeMs >= statSync(bin).mtimeMs;
  }

  readOutlinesMeta(dir: string): { moments: { name: string; after_impact_ms: number }[] } {
    return JSON.parse(readFileSync(join(dir, 'outlines.json'), 'utf8'));
  }

  stop(): void {
    this.proc?.kill();
  }
}
