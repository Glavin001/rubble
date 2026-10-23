// "Stress solver" bond mode: the GPU stress solver coupled to rubble, run in Rust on its own
// WebGPU device and canvas (engine/stress-oracle/crates/stress-web). This file loads the
// module, forwards input and drives one frame per animation frame; the physics, camera,
// shot rule and drawing are the Rust's.
import init,{stress_init,stress_set_glb,stress_set_cuboid,stress_set_pack,stress_error,stress_stats,stress_set_running,stress_running,stress_resize,stress_drag,stress_zoom,stress_shoot,stress_set_shot_mass_scale,stress_set_shot_speed_scale,stress_set_shot_radius_scale,stress_set_bond_strength_scale,stress_set_profile,stress_set_view,stress_set_physics_hz} from '../generated/stress-web/stress_web.js';
// Chrome and Edge 137+ suspend the wasm stack natively (JSPI). Elsewhere (Safari) the
// Asyncify build of the same Rust does, driven by gpu-wait.mjs, which picks the build.
import {JSPI,bindAsyncify,asyncifyPromising} from './gpu-wait.mjs';
let module=null;
export async function createStressMode(canvas){
 if(!module){
  const wasm=JSPI?'stress_web_bg.wasm':'stress_web_bg.asyncify.wasm';
  module=await init({module_or_path:new URL(`../generated/stress-web/${wasm}`,import.meta.url)});
  if(!JSPI)bindAsyncify(module,module.stress_asyncify_buffer());
 }
 // Exports that may wait on the GPU (they suspend; see gpu-wait.mjs).
 const suspendable=JSPI?f=>WebAssembly.promising(f):asyncifyPromising;
 const load=suspendable(module.stress_load),frame=suspendable(module.stress_frame);
 const fit=()=>{const r=devicePixelRatio||1,w=Math.max(1,Math.round(canvas.clientWidth*r)),h=Math.max(1,Math.round(canvas.clientHeight*r));
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;stress_resize(w,h)}};
 fit();await stress_init(canvas);
 let drag=null;
 const down=e=>{if(e.button!==0)return;drag={start:[e.clientX,e.clientY],last:[e.clientX,e.clientY],moved:false};canvas.setPointerCapture(e.pointerId)};
 const move=e=>{if(!drag)return;if(Math.hypot(e.clientX-drag.start[0],e.clientY-drag.start[1])>5)drag.moved=true;
  if(drag.moved)stress_drag(e.clientX-drag.last[0],e.clientY-drag.last[1]);drag.last=[e.clientX,e.clientY]};
 const up=e=>{if(!drag)return;const shot=!drag.moved;drag=null;if(shot){const r=canvas.getBoundingClientRect();stress_shoot((e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height)}};
 const cancel=()=>drag=null,wheel=e=>{e.preventDefault();stress_zoom(e.deltaY)};
 canvas.addEventListener('pointerdown',down);canvas.addEventListener('pointermove',move);canvas.addEventListener('pointerup',up);canvas.addEventListener('pointercancel',cancel);canvas.addEventListener('wheel',wheel,{passive:false});
 return {
  async load(bytes,level){stress_set_glb(new Uint8Array(bytes));if(!await load(level))throw new Error(stress_error())},
  // The lab's cuboid houses in a sample's layout (null, {columns, rows} or {suburb: true}).
  async loadCuboid(layout){stress_set_cuboid(JSON.stringify(layout??null));if(!await load(0))throw new Error(stress_error())},
  // A scene pack's JSON bytes (vibe-city / blast-stress-solver).
  async loadPack(bytes){stress_set_pack(new Uint8Array(bytes));if(!await load(0))throw new Error(stress_error())},
  async frame(elapsedMs){fit();if(!await frame(elapsedMs))throw new Error(stress_error())},
  get stats(){return JSON.parse(stress_stats())},
  setShotMassScale(scale){stress_set_shot_mass_scale(scale)},
  // The shots' speed (from the next shot) and radius (from the next load), as multiples
  // of the lab's.
  setShotSpeedScale(scale){stress_set_shot_speed_scale(scale)},
  setShotRadiusScale(scale){stress_set_shot_radius_scale(scale)},
  setBondStrengthScale(scale){stress_set_bond_strength_scale(scale)},
  // The capability profile of the next load: blast_parity, or threshold (the v0 mode).
  setProfile(name){stress_set_profile(name)},
  // The physics frame rate of the next load (Hz; 0 keeps the scene's).
  setPhysicsHz(hz){stress_set_physics_hz(hz)},
  // The lab's view: the bond graph, and the colour mode (0 materials, 1 body state,
  // 2 one colour per body, kept as it breaks).
  setView(bonds,colorMode){stress_set_view(bonds,colorMode)},
  get running(){return stress_running()},set running(on){stress_set_running(on)},
  dispose(){canvas.removeEventListener('pointerdown',down);canvas.removeEventListener('pointermove',move);canvas.removeEventListener('pointerup',up);canvas.removeEventListener('pointercancel',cancel);canvas.removeEventListener('wheel',wheel)}};
}
