// Copies the pinned three.js (package.json) into generated/three, where the pages' import
// map finds it ("three/webgpu", "three/tsl", "three/addons/…"). The site is static and has no
// bundler: these are three's own ES module builds, served as is.
//
//   npm ci && node vendor-three.mjs      (engine/scripts/build-web.sh runs both)
import { cpSync, mkdirSync, rmSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const pkg = join(here, 'node_modules', 'three');
const out = join(here, 'generated', 'three');
if (!existsSync(pkg)) throw new Error('node_modules/three is missing: run `npm ci` in engine/crates/engine-webgpu first');

// three's builds and the addons the pages import (examples/jsm paths).
const BUILD = ['three.webgpu.js', 'three.tsl.js', 'three.core.js'];
const ADDONS = ['controls/OrbitControls.js'];

rmSync(out, { recursive: true, force: true });
mkdirSync(join(out, 'addons'), { recursive: true });
for (const f of BUILD) cpSync(join(pkg, 'build', f), join(out, f));
cpSync(join(pkg, 'LICENSE'), join(out, 'LICENSE'));
for (const f of ADDONS) {
  mkdirSync(dirname(join(out, 'addons', f)), { recursive: true });
  cpSync(join(pkg, 'examples', 'jsm', f), join(out, 'addons', f));
}
console.log(`three ${JSON.parse((await import('node:fs')).readFileSync(join(pkg, 'package.json'), 'utf8')).version} → ${out}`);
