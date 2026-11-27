// A converted BeamNG vehicle folder (engine-softbody's community_cache writes them to
// engine-webgpu/__mods/<folder>, each with a files.json of its files' paths): fetched
// for the wasm, which parses and builds everything (the JBeam vehicle, its flexbody
// visual, its textures). Shared by the pages that load one (crash-lab.html ?mod=,
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

/**
 * A vehicle folder's render model files for a detail: `'full'` its art (`<x>.glb`,
 * `<x>.rskin`), `'lite'` the lighter render beside it (`<x>.lite.glb`, `.lite.rskin`:
 * the vehicle generator's same design meshed coarser and bound to the same lattice; the
 * physics files are the same), or the art when the folder has no lighter one.
 * `{ glbName, skinName, detail (the one found), hasLite }`.
 */
export function modModelNames(list, detail = 'full') {
  const lite = (f) => /\.lite\.(glb|rskin)$/.test(f);
  const pick = (want) => {
    const skinName = list.find((f) => f.endsWith('.rskin') && lite(f) === want);
    const glbName = skinName && list.find((f) => f.endsWith('.glb') && f.includes('/') && lite(f) === want);
    return { glbName, skinName };
  };
  const hasLite = !!pick(true).skinName;
  const useLite = detail === 'lite' && hasLite;
  return { ...pick(useLite), detail: useLite ? 'lite' : 'full', hasLite };
}

export async function loadMod(base, { pc = null, detail = 'full' } = {}) {
  base = base.replace(/\/$/, '');
  const r = await fetch(`${base}/files.json`);
  if (!r.ok) throw new Error(`${base}/files.json: ${r.status} ${r.statusText}. ${LOCAL_MODS}`);
  const list = await r.json();
  const text = async (f) => [f, new TextDecoder().decode(await fetchFile(`${base}/${f}`))];
  const bin = async (f) => (f ? await fetchFile(`${base}/${f}`) : null);
  const pcName = pc ?? list.find((f) => f.endsWith('.pc'));
  const names = modModelNames(list, detail);
  const { skinName, glbName } = names;
  const [files, pcText, meshes, materials, glb, skin] = await Promise.all([
    Promise.all(list.filter((f) => f.endsWith('.jbeam')).map(text)),
    pcName && list.includes(pcName) ? text(pcName).then((t) => t[1]) : undefined,
    Promise.all(list.filter((f) => f.endsWith('.dae')).map(async (f) => [f, await bin(f)])),
    Promise.all(list.filter((f) => f.endsWith('.materials.json')).map(text)),
    bin(glbName),
    bin(skinName),
  ]);
  return { kind: 'jbeam', files, pc: pcText, pcName, meshes, materials, base, list, label: base.split('/').pop(), glb, skin, glbName, skinName, detail: names.detail, hasLite: names.hasLite,
    modelBytes: (glb?.length ?? 0) + (skin?.length ?? 0) };
}

/**
 * `loadMod` for a vehicle folder held in memory: `files` is `[[path, bytes]]` (what the
 * vehicle rigger returns in the browser: engine-vehiclerig-wasm, vehicle-design-worker.mjs;
 * `files.json` among them, as `loadMod` reads it). The same result shape as `loadMod`'s,
 * with `base: null` and `store` (path -> Blob: the model files, read again by `modModels`
 * for each paint, as `loadMod`'s are fetched again).
 */
export function modFromFiles(files, { pc = null, detail = 'full', label = 'custom' } = {}) {
  const byPath = new Map(files.map(([p, b]) => [p, b]));
  const index = byPath.get('files.json');
  const list = index ? JSON.parse(new TextDecoder().decode(index)) : [...byPath.keys()];
  for (const f of list) if (!byPath.has(f)) throw new Error(`${label}: files.json lists ${f}, which the files do not contain`);
  const text = (f) => [f, new TextDecoder().decode(byPath.get(f))];
  const pcName = pc ?? list.find((f) => f.endsWith('.pc'));
  const names = modModelNames(list, detail);
  const { skinName, glbName } = names;
  const store = new Map();
  for (const f of [glbName, skinName]) if (f) store.set(f, new Blob([byPath.get(f)]));
  const glb = glbName ? byPath.get(glbName) : null, skin = skinName ? byPath.get(skinName) : null;
  return { kind: 'jbeam', files: list.filter((f) => f.endsWith('.jbeam')).map(text), pc: pcName && list.includes(pcName) ? text(pcName)[1] : undefined, pcName,
    meshes: list.filter((f) => f.endsWith('.dae')).map((f) => [f, byPath.get(f)]), materials: list.filter((f) => f.endsWith('.materials.json')).map(text),
    base: null, store, images: new Map(list.filter((f) => /\.(png|jpe?g|dds|webp)$/i.test(f)).map((f) => [f, byPath.get(f)])), list, label, glb, skin, glbName, skinName, detail: names.detail, hasLite: names.hasLite,
    modelBytes: (glb?.length ?? 0) + (skin?.length ?? 0) };
}

/** The model files' sizes of a mod held in memory (`modModelSizes` for a folder at a URL). */
export function modModelSizesOf(mod) {
  return { glb: mod.store?.get(mod.glbName)?.size ?? 0, skin: mod.store?.get(mod.skinName)?.size ?? 0, detail: mod.detail, hasLite: mod.hasLite };
}

// A glTF-skinned vehicle's cost as the vehicle lab measured it
// (engine/docs/measurements/lab-load-memory.md: the generated sedan, 16 MB glb and
// 124 MB rskin; the authored sedan for the base): the tab's peak physical footprint
// (JS and wasm heaps, the browser's own), the GPU memory the lab holds, and the skin's
// bind buffer (64 bytes a vertex and a slot of 8 entries, the sidecar's own grid steps:
// 157 MB for the sedan's 124 MB rskin; it was 267 MB at 4 f32 entries a slot, and the
// GPU's figure is the measured one less that difference).
const FOOTPRINT = { tabBase: 450e6, tabPerModelByte: 6.0, gpuBase: 170e6, gpuPerSkinByte: 1.26, bindPerSkinByte: 1.28 };
// The share of a device's memory a tab can count on (a judgement: browsers kill a
// tab well before it has the device's whole memory).
const TAB_SHARE = 0.5;

/**
 * What loading a glTF-skinned vehicle will cost, estimated from its model files' sizes
 * (bytes: `modModelSizes`): `{ tab, gpu, largestBuffer }` in bytes.
 */
export function modFootprint({ glb = 0, skin = 0 }) {
  const f = FOOTPRINT;
  return { tab: f.tabBase + f.tabPerModelByte * (glb + skin), gpu: f.gpuBase + f.gpuPerSkinByte * skin, largestBuffer: f.bindPerSkinByte * skin };
}

/**
 * Whether a footprint (`modFootprint`) fits this device: `deviceMemory` (GB,
 * navigator.deviceMemory where the browser gives it; Safari does not), `limits` (the
 * WebGPU adapter's, or null). `{ fits, message }`: the message names the vehicle and
 * the numbers. Without a deviceMemory the memory is not judged (nothing to judge by).
 */
export function footprintCheck(fp, { deviceMemory, limits, name }) {
  const mb = (b) => `${Math.round(b / 2 ** 20)} MB`;
  const need = `the ${name} needs about ${mb(fp.tab)} of this tab's memory and ${mb(fp.gpu)} of GPU memory to load`;
  if (limits) {
    for (const k of ['maxStorageBufferBindingSize', 'maxBufferSize']) {
      if (limits[k] !== undefined && fp.largestBuffer > limits[k]) {
        return { fits: false, message: `${need}, with one buffer of ${mb(fp.largestBuffer)}: over this GPU's ${k} of ${mb(limits[k])}.` };
      }
    }
  }
  if (deviceMemory && fp.tab + fp.gpu > TAB_SHARE * deviceMemory * 2 ** 30) {
    return { fits: false, message: `${need}; this device reports ${deviceMemory} GB of memory, and a tab can count on about ${Math.round(100 * TAB_SHARE)} % of it. Loading it would likely crash the tab.` };
  }
  return { fits: true, message: '' };
}

/**
 * A vehicle folder's model files' sizes for a detail (`modModelNames`; bytes, from HEAD:
 * nothing is downloaded): `{ glb, skin, detail, hasLite }`.
 */
export async function modModelSizes(base, detail = 'full') {
  base = base.replace(/\/$/, '');
  const r = await fetch(`${base}/files.json`);
  if (!r.ok) throw new Error(`${base}/files.json: ${r.status} ${r.statusText}. ${LOCAL_MODS}`);
  const names = modModelNames(await r.json(), detail);
  const size = async (f) => (f ? Number((await fetch(`${base}/${f}`, { method: 'HEAD' })).headers.get('Content-Length')) || 0 : 0);
  const [glb, skin] = await Promise.all([size(names.glbName), size(names.skinName)]);
  return { glb, skin, detail: names.detail, hasLite: names.hasLite };
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
  if (mod.store) {
    const blob = async (f) => (f ? new Uint8Array(await mod.store.get(f).arrayBuffer()) : null);
    [mod.glb, mod.skin] = await Promise.all([blob(mod.glbName), blob(mod.skinName)]);
    return mod;
  }
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
  r.visual.closedPanels = closedSolids(mod.glb);
  return r;
}

/**
 * `modVisual` for a glTF-skinned vehicle, built in a worker of its own
 * (visual-worker.mjs: the same wasm, the same call) so the wasm heap the build grows
 * (about 530 MB for the generated sedan) is freed with the worker instead of staying
 * the page's. The model bytes go to the worker (`mod.glb` and `mod.skin` are released,
 * as `releaseModels` would; `modModels` fetches them again). Falls back to `modVisual`
 * on the page where a module worker cannot run.
 */
export async function modVisualApart(mod, { native = true, raycast = false, paint = null } = {}) {
  if (!mod.skin || typeof Worker === 'undefined') return modVisual(mod, { native, raycast, paint });
  const spec = { jbeam: mod.files, pc: mod.pc, native, ...(raycast ? { wheels: 'raycast' } : {}), ...(paint ? { paint } : {}), materials: mod.materials ?? [] };
  const closed = closedSolids(mod.glb);
  const here = (p) => new URL(p, import.meta.url).href;
  let worker;
  try { worker = new Worker(here('./visual-worker.mjs'), { type: 'module' }); } catch { return modVisual(mod, { native, raycast, paint }); }
  const { glb, skin } = mod;
  mod.glb = mod.skin = null;
  try {
    const answer = await new Promise((resolve, reject) => {
      worker.onmessage = (e) => resolve(e.data);
      worker.onerror = (e) => reject(new Error(`the visual's worker failed: ${e.message || 'no message'}`));
      worker.postMessage({ wasm: here('./generated/world-wasm/engine_world_wasm_bg.wasm'), glue: here('./generated/world-wasm/engine_world_wasm.js'), gpuWait: here('./fracture/gpu-wait.mjs'), spec, glb, skin }, [glb.buffer, skin.buffer]);
    });
    if (answer.error) throw new Error(answer.error);
    const r = answer.result;
    r.visual.lod = r.visualLod ? [r.visualLod, r.visualFar] : [];
    r.visual.closedPanels = closed;
    return r;
  } finally {
    worker.terminate();
  }
}

/**
 * Whether a glTF model's parts are closed solids: the vehicle generator tags every
 * vertex of a solid part outer skin, edge wall or inner face (the `_SURFACE` attribute,
 * engine-vehiclegen export.rs), and writes no tag for sheet parts. The renderer then
 * knows a paint face's back is inside a sealed panel (showcase-car-render.mjs).
 */
export function closedSolids(glb) {
  if (!glb || glb.byteLength < 20) return false;
  const view = new DataView(glb.buffer, glb.byteOffset, glb.byteLength);
  // Header (magic, version, length), then the JSON chunk (length, type 'JSON').
  if (view.getUint32(0, true) !== 0x46546c67 || view.getUint32(16, true) !== 0x4e4f534a) return false;
  const json = JSON.parse(new TextDecoder().decode(glb.subarray(20, 20 + view.getUint32(12, true))));
  return (json.meshes ?? []).some((m) => (m.primitives ?? []).some((p) => p.attributes && '_SURFACE' in p.attributes));
}

/**
 * The folder's images named by `names` (a visual's `textures` or `dataTextures`), each
 * decoded in the wasm to a mipped `size`² layer; null for a missing or unreadable one.
 */
export function modTextures(mod, names, size = 1024) {
  return Promise.all((names ?? []).map(async (t) => {
    const f = mod.list.find((x) => x.split('/').pop().toLowerCase() === t.toLowerCase());
    if (!f) return null;
    if (mod.images) { try { return decodeTexture(mod.images.get(f), size); } catch { return null; } }
    try { return decodeTexture(new Uint8Array(await (await fetch(`${mod.base}/${f}`)).arrayBuffer()), size); } catch { return null; }
  }));
}
