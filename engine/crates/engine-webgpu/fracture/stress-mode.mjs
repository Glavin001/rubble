// "Stress solver" bond mode: the GPU stress solver coupled to rubble, run in Rust on its own
// WebGPU device and canvas (engine/stress-oracle/crates/stress-web). This file loads the
// module, forwards input (camera-input.mjs, as the page's renderer) and drives one frame per
// animation frame; the physics, camera, shot rule and drawing are the Rust's.
import init,{stress_init,stress_set_glb,stress_set_cuboid,stress_set_pack,stress_error,stress_stats,stress_set_running,stress_running,stress_resize,stress_drag,stress_zoom,stress_set_view_mode,stress_mouse_look,stress_travel,stress_set_shot_mass_scale,stress_set_shot_speed_scale,stress_set_shot_radius_scale,stress_set_bond_strength_scale,stress_set_profile,stress_set_view,stress_set_physics_hz,stress_panic,stress_set_orbit,stress_orbit_state,stress_shoot_ray,stress_aim,stress_set_fx,stress_shot_presets,stress_set_shot_preset,stress_fire,stress_charges,stress_detonate} from '../generated/stress-web/stress_web.js';
// Chrome and Edge 137+ suspend the wasm stack natively (JSPI). Elsewhere (Safari) the
// Asyncify build of the same Rust does, driven by gpu-wait.mjs, which picks the build.
import {JSPI,bindAsyncify,asyncifyPromising} from './gpu-wait.mjs';
import {attachCameraInput} from './camera-input.mjs';
let module=null;
// onShot(origin, direction): each click's shot, as a lab-frame ray (the replay log).
export async function createStressMode(canvas,onShot=()=>{}){
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
 const load=why(suspendable(module.stress_load)),frame=why(suspendable(module.stress_frame)),frameSteps=why(suspendable(module.stress_frame_steps));
 const fit=()=>{const r=devicePixelRatio||1,w=Math.max(1,Math.round(canvas.clientWidth*r)),h=Math.max(1,Math.round(canvas.clientHeight*r));
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;stress_resize(w,h)}};
 fit();await stress_init(canvas);
 let viewMode=0;
 const input=attachCameraInput(canvas,{get mode(){return viewMode},drag:stress_drag,mouseLook:stress_mouse_look,zoom:stress_zoom,travel:stress_travel},(u,v)=>{const r=stress_aim(u,v);if(r.length===6&&stress_shoot_ray(r.subarray(0,3),r.subarray(3)))onShot([...r.subarray(0,3)],[...r.subarray(3)])});
 return {
  async load(bytes,level){stress_set_glb(new Uint8Array(bytes));if(!await load(level))throw new Error(stress_error())},
  // The lab's cuboid houses in a sample's layout (null, {columns, rows} or {suburb: true}).
  async loadCuboid(layout){stress_set_cuboid(JSON.stringify(layout??null));if(!await load(0))throw new Error(stress_error())},
  // A scene pack's JSON bytes (vibe-city / blast-stress-solver).
  async loadPack(bytes){stress_set_pack(new Uint8Array(bytes));if(!await load(0))throw new Error(stress_error())},
  async frame(elapsedMs){fit();input.update();if(!await frame(elapsedMs))throw new Error(stress_error())},
  // Frame-locked (recorded takes): exactly `steps` physics frames, running or paused, then draw.
  async frameSteps(steps){fit();if(!await frameSteps(steps>>>0))throw new Error(stress_error())},
  // The orbit camera: [yaw, pitch, distance, tx, ty, tz] (lab frame); a target is optional.
  setOrbit(yaw,pitch,distance,target=[]){stress_set_orbit(yaw,pitch,distance,new Float64Array(target))},
  get orbit(){return [...stress_orbit_state()]},
  // A shot along a lab-frame ray (camera independent: replays and takes).
  // The shot ray through a viewport point (u, v in 0..1): {origin, direction}, or null.
  aim(u,v){const r=stress_aim(u,v);return r.length===6?{origin:[...r.subarray(0,3)],direction:[...r.subarray(3)]}:null},
  shootRay(origin,direction){return stress_shoot_ray(new Float64Array(origin),new Float64Array(direction))},
  get stats(){return JSON.parse(stress_stats())},
  setShotMassScale(scale){stress_set_shot_mass_scale(scale)},
  // The shots' speed (from the next shot) and radius (from the next load), as multiples
  // of the lab's.
  setShotSpeedScale(scale){stress_set_shot_speed_scale(scale)},
  setShotRadiusScale(scale){stress_set_shot_radius_scale(scale)},
  setBondStrengthScale(scale){stress_set_bond_strength_scale(scale)},
  // The ball presets ([{key, menu, mass, radius, speed, rolling}]); the next load's ball
  // (an index, or -1 for the scene's); fire it along the pack's line of attack (stats.canFire).
  get shotPresets(){return JSON.parse(stress_shot_presets())},
  setShotPreset(index){stress_set_shot_preset(index)},
  fire(){return stress_fire()},
  // The loaded pack's demolition charges ([{name, delay, bonds}], firing order); set one off
  // by name, or '' for all, each at its delay: cut between frames before the next steps.
  get charges(){return JSON.parse(stress_charges())},
  detonate(name=''){return stress_detonate(name)},
  // The capability profile of the next load: blast_parity, or threshold (the v0 mode).
  setProfile(name){stress_set_profile(name)},
  // The physics frame rate of the next load (Hz; 0 keeps the scene's).
  setPhysicsHz(hz){stress_set_physics_hz(hz)},
  // The lab's view: the bond graph, and the colour mode (0 materials, 1 body state,
  // 2 one colour per body, kept as it breaks, 3 materials untinted).
  setView(bonds,colorMode){stress_set_view(bonds,colorMode)},
  // Dust and chips from the fragments (render only).
  setFx(on){stress_set_fx(!!on)},
  // The camera's view mode (0 orbit, 1 first-person walk, 2 fly), kept across loads.
  get viewMode(){return viewMode},
  set viewMode(mode){viewMode=mode|0;stress_set_view_mode(viewMode);input.modeChanged()},
  get running(){return stress_running()},set running(on){stress_set_running(on)},
  dispose(){input.dispose()}};
}
