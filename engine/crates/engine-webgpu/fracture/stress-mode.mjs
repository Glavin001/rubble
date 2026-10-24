// "Stress solver" bond mode: the GPU stress solver coupled to rubble, run in Rust on its own
// WebGPU device and canvas (engine/stress-oracle/crates/stress-web). This file loads the
// module, forwards input (camera-input.mjs, as the page's renderer) and drives one frame per
// animation frame; the physics, camera, shot rule and drawing are the Rust's.
import init,{stress_init,stress_set_glb,stress_set_cuboid,stress_set_pack,stress_error,stress_stats,stress_set_running,stress_running,stress_resize,stress_drag,stress_zoom,stress_shoot,stress_set_view_mode,stress_mouse_look,stress_travel,stress_set_shot_mass_scale,stress_set_shot_speed_scale,stress_set_shot_radius_scale,stress_set_bond_strength_scale,stress_set_profile,stress_set_view,stress_set_physics_hz,stress_panic} from '../generated/stress-web/stress_web.js';
// Chrome and Edge 137+ suspend the wasm stack natively (JSPI). Elsewhere (Safari) the
// Asyncify build of the same Rust does, driven by gpu-wait.mjs, which picks the build.
import {JSPI,bindAsyncify,asyncifyPromising} from './gpu-wait.mjs';
import {attachCameraInput} from './camera-input.mjs';
let module=null;
export async function createStressMode(canvas){
 if(!module){
  const wasm=JSPI?'stress_web_bg.wasm':'stress_web_bg.asyncify.wasm';
  module=await init({module_or_path:new URL(`../generated/stress-web/${wasm}`,import.meta.url)});
  if(!JSPI)bindAsyncify(module,module.stress_asyncify_buffer());
 }
 // Exports that may wait on the GPU (they suspend; see gpu-wait.mjs).
 const suspendable=JSPI?f=>WebAssembly.promising(f):asyncifyPromising;
 // A trap (a Rust panic: "unreachable") reports the panic's message, which only the console
 // shows otherwise.
 const why=f=>async(...a)=>{try{return await f(...a)}catch(e){const p=stress_panic();throw p?new Error(`${p} (${e?.message??e})`):e}};
 const load=why(suspendable(module.stress_load)),frame=why(suspendable(module.stress_frame));
 const fit=()=>{const r=devicePixelRatio||1,w=Math.max(1,Math.round(canvas.clientWidth*r)),h=Math.max(1,Math.round(canvas.clientHeight*r));
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;stress_resize(w,h)}};
 fit();await stress_init(canvas);
 let viewMode=0;
 const input=attachCameraInput(canvas,{get mode(){return viewMode},drag:stress_drag,mouseLook:stress_mouse_look,zoom:stress_zoom,travel:stress_travel},(u,v)=>stress_shoot(u,v));
 return {
  async load(bytes,level){stress_set_glb(new Uint8Array(bytes));if(!await load(level))throw new Error(stress_error())},
  // The lab's cuboid houses in a sample's layout (null, {columns, rows} or {suburb: true}).
  async loadCuboid(layout){stress_set_cuboid(JSON.stringify(layout??null));if(!await load(0))throw new Error(stress_error())},
  // A scene pack's JSON bytes (vibe-city / blast-stress-solver).
  async loadPack(bytes){stress_set_pack(new Uint8Array(bytes));if(!await load(0))throw new Error(stress_error())},
  async frame(elapsedMs){fit();input.update();if(!await frame(elapsedMs))throw new Error(stress_error())},
  get stats(){return JSON.parse(stress_stats())},
  setShotMassScale(scale){stress_set_shot_mass_scale(scale)},
  // The shots' speed (from the next shot) and radius (from the next load), as multiples
  // of the lab's.
  setShotSpeedScale(scale){stress_set_shot_speed_scale(scale)},
  setShotRadiusScale(scale){stress_set_shot_radius_scale(scale)},
  setBondStrengthScale(scale){stress_set_bond_strength_scale(scale)},
  // The capability profile of the next load: blast_parity, threshold (the v0 mode),
  // load_path or impact_local.
  setProfile(name){stress_set_profile(name)},
  // The physics frame rate of the next load (Hz; 0 keeps the scene's).
  setPhysicsHz(hz){stress_set_physics_hz(hz)},
  // The lab's view: the bond graph, and the colour mode (0 materials, 1 body state,
  // 2 one colour per body, kept as it breaks).
  setView(bonds,colorMode){stress_set_view(bonds,colorMode)},
  // The camera's view mode (0 orbit, 1 first-person walk, 2 fly), kept across loads.
  get viewMode(){return viewMode},
  set viewMode(mode){viewMode=mode|0;stress_set_view_mode(viewMode);input.modeChanged()},
  get running(){return stress_running()},set running(on){stress_set_running(on)},
  dispose(){input.dispose()}};
}
