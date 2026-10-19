// "Stress solver" bond mode: the GPU stress solver coupled to rubble, run in Rust on its own
// WebGPU device and canvas (engine/stress-oracle/crates/stress-web). This file loads the
// module, forwards input and drives one frame per animation frame; the physics, camera,
// shot rule and drawing are the Rust's.
import init,{
  stress_init,stress_set_glb,stress_set_cuboid,stress_error,stress_stats,stress_set_running,stress_running,
  stress_resize,stress_drag,stress_zoom,stress_shoot,stress_set_shot_mass_scale,stress_set_bond_strength_scale,
  stress_set_learned_gate,stress_learned_gate,stress_learned_capture,stress_learned_dims,
  stress_learned_node_in,stress_learned_edge_in,stress_learned_endpoints,stress_set_pending_breaks,
  stress_learned_feature_gaps,
} from '../generated/stress-web/stress_web.js';
if(!('Suspending' in WebAssembly))throw new Error('Stress-solver mode needs WebAssembly JavaScript Promise Integration (Chrome 137 or later).');
let module=null;
export async function createStressMode(canvas){
 module??=await init({module_or_path:new URL('../generated/stress-web/stress_web_bg.wasm',import.meta.url)});
 // Exports that may wait on the GPU (they suspend; see gpu-wait.mjs).
 const load=WebAssembly.promising(module.stress_load),frame=WebAssembly.promising(module.stress_frame);
 const breakEdges=WebAssembly.promising(module.stress_break_edges);
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
  async load(bytes,level,{learnedGate=false}={}){
   stress_set_learned_gate(!!learnedGate);
   stress_set_glb(new Uint8Array(bytes));
   if(!await load(level))throw new Error(stress_error());
  },
  // The lab's cuboid houses in a sample's layout (null, {columns, rows} or {suburb: true}).
  async loadCuboid(layout,{learnedGate=false}={}){
   stress_set_learned_gate(!!learnedGate);
   stress_set_cuboid(JSON.stringify(layout??null));
   if(!await load(0))throw new Error(stress_error());
  },
  async frame(elapsedMs){fit();if(!await frame(elapsedMs))throw new Error(stress_error())},
  get stats(){return JSON.parse(stress_stats())},
  setShotMassScale(scale){stress_set_shot_mass_scale(scale)},
  setBondStrengthScale(scale){stress_set_bond_strength_scale(scale)},
  get learnedGate(){return stress_learned_gate()},
  featureGaps(){return stress_learned_feature_gaps()},
  /** Pack MGN features from the last drawn stress mirror (CPU). */
  captureLearned(){
   if(!stress_learned_capture())return null;
   const [nNodes,nEdges]=stress_learned_dims();
   return {
    nNodes,nEdges,
    nodeIn:new Float32Array(stress_learned_node_in()),
    edgeIn:new Float32Array(stress_learned_edge_in()),
    endpoints:new Int32Array(stress_learned_endpoints()),
   };
  },
  /** Break dense edge indices from the last capture; returns bonds broken. */
  async breakLearnedEdges(indices){
   stress_set_pending_breaks(Uint32Array.from(indices));
   const code=await breakEdges();
   if(!code)throw new Error(stress_error());
   return code-1;
  },
  get running(){return stress_running()},set running(on){stress_set_running(on)},
  dispose(){canvas.removeEventListener('pointerdown',down);canvas.removeEventListener('pointermove',move);canvas.removeEventListener('pointerup',up);canvas.removeEventListener('pointercancel',cancel);canvas.removeEventListener('wheel',wheel)}};
}
