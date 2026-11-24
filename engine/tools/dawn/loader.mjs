// `--import ./loader.mjs`: resolve the World wasm's one bare import, `rubble-gpu-wait`
// (the pages map it with an import map), to the bundle's fracture/gpu-wait.mjs.
import { register } from 'node:module';
import { pathToFileURL } from 'node:url';
import { join } from 'node:path';

const target = pathToFileURL(join(process.env.BUNDLE, 'fracture/gpu-wait.mjs')).href;
register('data:text/javascript,' + encodeURIComponent(`
export async function resolve(spec, ctx, next) {
  if (spec === 'rubble-gpu-wait') return { url: ${JSON.stringify(target)}, shortCircuit: true };
  return next(spec, ctx);
}`));
