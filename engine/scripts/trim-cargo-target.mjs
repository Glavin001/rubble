#!/usr/bin/env node
// Starts the workspace's own crates over in a cargo target dir and leaves every
// third-party crate's artifacts where they are:
//
//   node engine/scripts/trim-cargo-target.mjs TARGET_DIR ENGINE_DIR [--dry-run]
//
// Cargo never deletes an artifact. When a crate's dependencies or features change, its
// outputs get a new hash in their names and the old ones stay. For this workspace the
// bulk of that is its own crates' (measured on a 3.2 GB dir: 1.4 GB, 1.2 GB of it
// engine-webgpu's, whose libraries embed the shaders and whose build script writes them
// out again). Which of a crate's variants a build still uses cannot be told from the
// files (two feature sets of one crate are both current and are rebuilt at different
// times; a rule by age deleted a current one when tried), so every variant of the
// workspace's packages goes, in every profile directory: the same files as
// `cargo clean -p` for each of them. The next build compiles the workspace's crates
// again on warm third-party crates, which is what a build after a change to engine-webgpu
// does anyway.
//
// build-web.sh runs it when the build cache's target dir is over its cap. It used to
// delete the whole dir there, and a build from nothing no longer fits Vercel's 45
// minutes.
import fs from 'node:fs';
import path from 'node:path';

const [target, engine, flag] = process.argv.slice(2);
if (!target || !engine) {
  console.error('usage: trim-cargo-target.mjs TARGET_DIR ENGINE_DIR [--dry-run]');
  process.exit(2);
}
const dry = flag === '--dry-run';

// The workspace's package names, with `_` for `-` (cargo uses both spellings in names).
const names = new Set();
for (const dir of [engine, path.join(engine, 'crates'), path.join(engine, 'stress-oracle'), path.join(engine, 'stress-oracle', 'crates')]) {
  let entries = [];
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { continue; }
  for (const manifest of [path.join(dir, 'Cargo.toml'), ...entries.filter((e) => e.isDirectory()).map((e) => path.join(dir, e.name, 'Cargo.toml'))]) {
    let text;
    try { text = fs.readFileSync(manifest, 'utf8'); } catch { continue; }
    const m = /^\[package\][^[]*?^name\s*=\s*"([^"]+)"/ms.exec(text);
    if (m) names.add(m[1].replace(/-/g, '_'));
  }
}
if (names.size === 0) {
  console.error(`trim-cargo-target: no packages found under ${engine}`);
  process.exit(2);
}

function size(p) {
  const st = fs.lstatSync(p);
  if (!st.isDirectory()) return st.size;
  let bytes = 0;
  for (const e of fs.readdirSync(p)) bytes += size(path.join(p, e));
  return bytes;
}

// The profile directories: any directory at most three levels down that has a `deps`.
const INSIDE = ['deps', 'build', 'incremental', '.fingerprint', 'examples'];
function profiles(dir, depth) {
  const out = [];
  let entries = [];
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return out; }
  if (entries.some((e) => e.isDirectory() && e.name === 'deps')) out.push(dir);
  if (depth > 0) {
    for (const e of entries) {
      if (e.isDirectory() && !INSIDE.includes(e.name)) out.push(...profiles(path.join(dir, e.name), depth - 1));
    }
  }
  return out;
}

// `libNAME-HASH.rlib`, `NAME-HASH.d`, `NAME-HASH` (a fingerprint or build directory), ...
const ARTIFACT = /^(?:lib)?(.+)-[0-9a-f]{16}(?:\..+)?$/;
let freed = 0, removed = 0;
for (const profile of profiles(target, 3)) {
  for (const sub of ['deps', '.fingerprint', 'build']) {
    let entries = [];
    try { entries = fs.readdirSync(path.join(profile, sub)); } catch { continue; }
    for (const name of entries) {
      const m = ARTIFACT.exec(name);
      if (!m || !names.has(m[1].replace(/-/g, '_'))) continue;
      const p = path.join(profile, sub, name);
      freed += size(p);
      removed += 1;
      if (!dry) fs.rmSync(p, { recursive: true, force: true });
    }
  }
}
console.log(`    ${dry ? 'would free' : 'freed'} ${(freed / 1e6).toFixed(0)} MB: ${removed} artifacts of the workspace's ${names.size} packages (third-party crates kept)`);
