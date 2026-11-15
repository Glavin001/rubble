#!/usr/bin/env node
// Write the showcase car renderer's skin modules (SHOWCASE_SKIN_WGSL: the
// solver-buffer and positions-buffer variants) to a directory, for
// `cargo run -p engine-webgpu --release --example wgsl_validate`.
//   node showcase-wgsl-dump.mjs <dir>
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { SHOWCASE_SKIN_WGSL } from './showcase-car-render.mjs';
const dir = process.argv[2] ?? '.';
mkdirSync(dir, { recursive: true });
// naga 24 does not convert an abstract-int const in a float comparison
// (`LOD_PX`, compared with an f32 in pick); Tint does. Written as a float
// literal here so naga checks the rest.
const naga = (code) => code.replace(/const LOD_PX = (\d+);/, 'const LOD_PX = $1.0;');
for (const [k, code] of Object.entries(SHOWCASE_SKIN_WGSL)) writeFileSync(join(dir, `showcase-skin-${k}.wgsl`), naga(code));
console.log(Object.keys(SHOWCASE_SKIN_WGSL).map((k) => join(dir, `showcase-skin-${k}.wgsl`)).join(' '));
