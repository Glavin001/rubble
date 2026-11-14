// A converted BeamNG vehicle folder (engine-softbody's community_cache writes them to
// engine-webgpu/__mods/<folder>, each with a files.json of its files' paths): fetched
// for the wasm, which parses and builds everything (the JBeam vehicle, its flexbody
// visual, its textures). Shared by the pages that load one (softbody.html ?mod=,
// one-world.html). Community content is data: only its .jbeam, .pc, .dae,
// .materials.json, .glb/.rskin and image files are read, and nothing in it runs.
import { jbeamVisual, jbeamVisualSkinned, decodeTexture } from './rubble-world.mjs';

/**
 * The vehicle folder at `base` (a URL): `{ kind: 'jbeam', files: [[name, text]] (its
 * .jbeam parts), pc (the .pc text: `pc` its path, else the folder's first), pcName,
 * meshes: [[name, Uint8Array]] (.dae), materials: [[name, text]], glb, skin (a glTF
 * render model with weighted skins: the folder's .glb and .rskin, else null), base,
 * list, label }`.
 */
/** Why a converted vehicle can be missing: they live only in a developer's checkout. */
export const LOCAL_MODS = 'Converted community vehicles are local only (engine-webgpu/__mods, never committed or deployed); CONTRIBUTING.md "Local BeamNG vehicles" says how to make one with community_cache.';

/**
 * One file's bytes: an HTTP error throws, and a body shorter than its Content-Length
 * (a connection cut mid-file, seen over a proxy as "rskin: truncated") is fetched once
 * more, then throws naming the file and both sizes. (A compressed response's
 * Content-Length is the encoded size, so the check applies only to identity bodies.)
 */
export async function fetchFile(url) {
  for (let attempt = 1; ; attempt++) {
    const r = await fetch(url, attempt > 1 ? { cache: 'no-store' } : undefined);
    if (!r.ok) throw new Error(`${url}: ${r.status} ${r.statusText}`);
    const bytes = new Uint8Array(await r.arrayBuffer());
    const encoded = (r.headers.get('Content-Encoding') ?? 'identity') !== 'identity';
    const want = encoded ? NaN : Number(r.headers.get('Content-Length'));
    if (!(want > bytes.length)) return bytes;
    if (attempt >= 2) throw new Error(`${url}: truncated, ${bytes.length} of ${want} bytes (after a retry)`);
  }
}

export async function loadMod(base, { pc = null } = {}) {
  base = base.replace(/\/$/, '');
  const r = await fetch(`${base}/files.json`);
  if (!r.ok) throw new Error(`${base}/files.json: ${r.status} ${r.statusText}. ${LOCAL_MODS}`);
  const list = await r.json();
  const text = async (f) => [f, new TextDecoder().decode(await fetchFile(`${base}/${f}`))];
  const bin = async (f) => (f ? await fetchFile(`${base}/${f}`) : null);
  const pcName = pc ?? list.find((f) => f.endsWith('.pc'));
  const skinName = list.find((f) => f.endsWith('.rskin'));
  const glbName = skinName && list.find((f) => f.endsWith('.glb') && f.includes('/'));
  const [files, pcText, meshes, materials, glb, skin] = await Promise.all([
    Promise.all(list.filter((f) => f.endsWith('.jbeam')).map(text)),
    pcName && list.includes(pcName) ? text(pcName).then((t) => t[1]) : undefined,
    Promise.all(list.filter((f) => f.endsWith('.dae')).map(async (f) => [f, await bin(f)])),
    Promise.all(list.filter((f) => f.endsWith('.materials.json')).map(text)),
    bin(glbName),
    bin(skinName),
  ]);
  return { kind: 'jbeam', files, pc: pcText, pcName, meshes, materials, base, list, label: base.split('/').pop(), glb, skin, glbName, skinName,
    modelBytes: (glb?.length ?? 0) + (skin?.length ?? 0) };
}

/**
 * Drop a glTF-skinned vehicle's model bytes (`mod.glb`, `mod.skin`: a generated car's
 * are ~140 MB) once its visuals are built; `modModels` fetches them again (the HTTP
 * cache's copy, normally) before another is built. A phone's tab cannot keep them.
 */
export function releaseModels(mod) {
  if (mod.skinName) mod.glb = mod.skin = null;
}

/** The model bytes `releaseModels` dropped, fetched again (nothing when they are held). */
export async function modModels(mod) {
  if (!mod.skinName || mod.skin) return mod;
  const [glb, skin] = await Promise.all([mod.glbName ? fetchFile(`${mod.base}/${mod.glbName}`) : null, fetchFile(`${mod.base}/${mod.skinName}`)]);
  mod.glb = glb; mod.skin = skin;
  return mod;
}

/**
 * Its flexbodies and props bound to the explicit solver's nodes (wasm jbeamVisual, or
 * jbeamVisualSkinned for a glTF model): `native` the JBeam-mode spawn (pressure wheels),
 * `raycast` its raycast-wheel spawn, `paint` [r, g, b]. Returns the wasm's result with
 * `visual.lod` set to its distant levels (`[]` when it has none).
 */
export function modVisual(mod, { native = true, raycast = false, paint = null } = {}) {
  if (mod.skinName && !mod.skin) throw new Error(`modVisual: ${mod.label}'s model files were released (releaseModels); await modModels(mod) first`);
  const spec = { jbeam: mod.files, pc: mod.pc, native, ...(raycast ? { wheels: 'raycast' } : {}), ...(paint ? { paint } : {}), materials: mod.materials ?? [] };
  const r = mod.skin ? jbeamVisualSkinned(spec, mod.meshes, mod.glb, mod.skin) : jbeamVisual(spec, mod.meshes);
  r.visual.lod = r.visualLod ? [r.visualLod, r.visualFar] : [];
  return r;
}

/**
 * The folder's images named by `names` (a visual's `textures` or `dataTextures`), each
 * decoded in the wasm to a mipped `size`² layer; null for a missing or unreadable one.
 */
export function modTextures(mod, names, size = 1024) {
  return Promise.all((names ?? []).map(async (t) => {
    const f = mod.list.find((x) => x.split('/').pop().toLowerCase() === t.toLowerCase());
    if (!f) return null;
    try { return decodeTexture(new Uint8Array(await (await fetch(`${mod.base}/${f}`)).arrayBuffer()), size); } catch { return null; }
  }));
}
