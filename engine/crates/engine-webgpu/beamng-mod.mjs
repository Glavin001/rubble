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
export async function loadMod(base, { pc = null } = {}) {
  base = base.replace(/\/$/, '');
  const r = await fetch(`${base}/files.json`);
  if (!r.ok) throw new Error(`${base}/files.json: ${r.status} ${r.statusText}`);
  const list = await r.json();
  const text = async (f) => [f, await (await fetch(`${base}/${f}`)).text()];
  const bin = async (f) => (f ? new Uint8Array(await (await fetch(`${base}/${f}`)).arrayBuffer()) : null);
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
  return { kind: 'jbeam', files, pc: pcText, pcName, meshes, materials, base, list, label: base.split('/').pop(), glb, skin };
}

/**
 * Its flexbodies and props bound to the explicit solver's nodes (wasm jbeamVisual, or
 * jbeamVisualSkinned for a glTF model): `native` the JBeam-mode spawn (pressure wheels),
 * `raycast` its raycast-wheel spawn, `paint` [r, g, b]. Returns the wasm's result with
 * `visual.lod` set to its distant levels (`[]` when it has none).
 */
export function modVisual(mod, { native = true, raycast = false, paint = null } = {}) {
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
