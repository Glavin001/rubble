// The engine's files the static studio serves (public/, gitignored), from the repo's own builds:
//
//   node scripts/engine-assets.ts engine [--link]   the world wasm (engine-world-wasm, the crash lab's
//       module), its kernels when it is a fast build, and its GPU-wait helper, from
//       engine/crates/engine-webgpu/generated/ (made by engine/scripts/build-web.sh with
//       RUBBLE_WEB_PARTS="kernels world") into public/rubble/. --link: symbolic links instead
//       of copies (local work: a rebuild is picked up at once; a static build copies them).
//   node scripts/engine-assets.ts designs [NAME ...] [--visual]  the generated designs (every registered one by
//       default): engine-softbody-lab's vehicle_gen_rig, the rigger the site build runs, each
//       design's mod folder into public/designs/<design>_new/ and the list into
//       public/designs/designs.json; each folder's physics files also as one physics.json. The
//       render model and skins (.glb, .rskin, ~200 MB a car) are left out unless --visual.
//
// Neither is committed: both are rebuilt from the repo's code.

import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, symlinkSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const PKG = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const ENGINE = resolve(PKG, '..', '..', 'engine');
const GEN = join(ENGINE, 'crates', 'engine-webgpu', 'generated');
const PUB = join(PKG, 'public');

function size(p: string): number {
  const st = statSync(p);
  if (st.isDirectory()) return readdirSync(p).reduce((k, f) => k + size(join(p, f)), 0);
  return st.size;
}

const mb = (b: number) => `${(b / 1048576).toFixed(1)} MB`;

function place(src: string, dst: string, link: boolean): void {
  rmSync(dst, { recursive: true, force: true });
  mkdirSync(dirname(dst), { recursive: true });
  if (link) symlinkSync(src, dst);
  else cpSync(src, dst, { recursive: true, dereference: true });
}

function engine(link: boolean): void {
  const wasm = join(GEN, 'world-wasm');
  if (!existsSync(join(wasm, 'engine_world_wasm_bg.wasm'))) {
    throw new Error(`${wasm}: no world wasm. Build it first (from the repo root):\n  RUBBLE_WEB_PROFILE=fast RUBBLE_WEB_PARTS="kernels world" bash engine/scripts/build-web.sh`);
  }
  const out = join(PUB, 'rubble');
  mkdirSync(out, { recursive: true });
  place(wasm, join(out, 'world-wasm'), link);
  // A fast build reads its kernels at run time from kernels/ beside world-wasm/; a release
  // build embeds them (no kernels/ needed).
  if (existsSync(join(GEN, 'kernels'))) place(join(GEN, 'kernels'), join(out, 'kernels'), link);
  place(join(ENGINE, 'crates', 'engine-webgpu', 'fracture', 'gpu-wait.mjs'), join(out, 'gpu-wait.mjs'), link);
  console.log(`public/rubble/: world wasm ${mb(statSync(join(wasm, 'engine_world_wasm_bg.wasm')).size)}${existsSync(join(GEN, 'kernels')) ? ', kernels (fast build)' : ''}${link ? ' (linked)' : ''}`);
}

function designs(names: string[], visual: boolean): void {
  const rig = (args: string[]) => execFileSync('cargo', ['run', '--release', '-q', '-p', 'engine-softbody-lab', '--example', 'vehicle_gen_rig', '--', ...args], { cwd: ENGINE, encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'], maxBuffer: 1 << 26 });
  const list = JSON.parse(rig(['--list-json'])) as { name: string; folder: string }[];
  const want = names.length ? list.filter((d) => names.includes(d.name)) : list;
  const out = join(PUB, 'designs');
  mkdirSync(out, { recursive: true });
  for (const d of want) {
    const t = Date.now();
    const dir = join(out, d.folder);
    rig([d.name, '--out', dir]);
    // What the studio builds a template from, in one file: every .jbeam part, the .pc and
    // crash.json as [[path, text], ...] (crash_templates::template_from_files' input). The
    // render model and skins (.glb, .rskin: about 200 MB a car) are dropped unless --visual.
    const files: [string, string][] = [];
    const walk = (rel: string) => {
      for (const f of readdirSync(join(dir, rel))) {
        const r = rel ? `${rel}/${f}` : f;
        if (statSync(join(dir, r)).isDirectory()) walk(r);
        else if (/\.(jbeam|pc)$|crash\.json$/.test(f)) files.push([r, readFileSync(join(dir, r), 'utf8')]);
        else if (!visual && /\.(glb|rskin)$/.test(f)) rmSync(join(dir, r));
      }
    };
    walk('');
    files.sort((a, b) => a[0].localeCompare(b[0]));
    writeFileSync(join(dir, 'physics.json'), JSON.stringify(files));
    console.log(`  ${d.name}: ${mb(size(dir))} (${files.length} physics files, ${((Date.now() - t) / 1000).toFixed(0)} s)`);
  }
  writeFileSync(join(out, 'designs.json'), JSON.stringify(list.filter((d) => existsSync(join(out, d.folder))), null, 1) + '\n');
  console.log(`public/designs/: ${mb(size(out))}`);
}

const [what, ...rest] = process.argv.slice(2);
if (what === 'engine') engine(rest.includes('--link'));
else if (what === 'designs') designs(rest.filter((x) => !x.startsWith('--')), rest.includes('--visual'));
else {
  console.error('usage: node scripts/engine-assets.ts engine [--link] | designs [NAME ...] [--visual]');
  process.exit(2);
}
