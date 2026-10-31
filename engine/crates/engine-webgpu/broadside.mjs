// Broadside: the galleon stern shot apart around a walking captain. The scene (the ship as
// a threshold destructible, the captain, the cues, the camera and the slow-motion rate) is
// Rust's (engine-world-wasm `broadside`), stepped on the wasm module's own WebGPU device;
// this page builds the ship's render scene from the same GLB (fracture/wasm.mjs), uploads
// each frame's poses and draws them with the fracture lab's renderer. `window.__film`
// renders the take frame by frame at a fixed clock rate for recording (record-broadside.mjs).
import init,{broadside_init,broadside_set_asset,broadside_error,broadside_panic,broadside_body_base,broadside_extra_vertices,broadside_extras_version,broadside_poses,broadside_camera,broadside_time_scale,broadside_set_scripted,broadside_shoot_ray,broadside_stats} from './generated/world-wasm/engine_world_wasm.js';
import {JSPI} from './fracture/gpu-wait.mjs';
import {parseFractureGlb,buildScene} from './fracture/wasm.mjs';
import {createRenderer} from './fracture/render.mjs';

const $=id=>document.getElementById(id);
const canvas=$('view'),status=$('status');
const setStatus=t=>{status.textContent=t;status.classList.remove('error')};
function fail(e){console.error(e);status.textContent=String(e?.message??e);status.classList.add('error')}

// The ship's leaf level (broadside::LEVEL) and the most bodies the pose buffer holds: the
// render scene's, the captain, MAX_BALLS balls and the crush splinter budget.
const LEVEL=2,POSE_FLOATS=28,EXTRA_BODIES=1+128+4096;

let device,renderer=null,poseBuffer=null,scene=null,glb=null;
let running=true,last=0,credit=0,extrasVersion=-1,frame=Promise.resolve();

const wasm=await init({module_or_path:new URL('./generated/world-wasm/engine_world_wasm_bg.wasm',import.meta.url)});
const why=f=>async(...a)=>{try{return await f(...a)}catch(e){const p=broadside_panic();throw p?new Error(`${p} (${e?.message??e})`):e}};
const load=why(WebAssembly.promising?WebAssembly.promising(wasm.broadside_load):()=>false);
const stepFrames=why(WebAssembly.promising?WebAssembly.promising(wasm.broadside_frame):()=>false);
const stats=()=>JSON.parse(broadside_stats());

// Upload the poses `alpha` of the way between the last two steps, the extras' mesh when
// it changed, the director's camera (`wall` seconds on), and draw.
function draw(alpha,wall){
 const poses=broadside_poses(alpha);device.queue.writeBuffer(poseBuffer,0,poses);
 const v=broadside_extras_version();
 if(v!==extrasVersion){extrasVersion=v;renderer.setDynamic(broadside_extra_vertices())}
 if($('director').checked){const c=broadside_camera(wall);renderer.lookAt([c[0],c[1],c[2]],[c[3],c[4],c[5]])}
 renderer.draw(false);
}

async function advance(steps,alpha,wall){
 if(steps>0&&!await stepFrames(steps))throw new Error(broadside_error());
 draw(alpha,wall);
}

async function open(){
 setStatus('Loading the galleon…');
 broadside_set_asset(new Uint8Array(glb));
 if(!await load())throw new Error(broadside_error());
 if(!scene)scene=buildScene(parseFractureGlb(glb),LEVEL);
 const bodies=broadside_body_base()+EXTRA_BODIES;
 poseBuffer?.destroy();poseBuffer=device.createBuffer({size:bodies*POSE_FLOATS*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
 renderer?.dispose();renderer=await createRenderer(device,canvas,{bodyBuffer:poseBuffer,shots:0},scene,shoot);
 extrasVersion=-1;last=0;credit=0;
 broadside_set_scripted(true);
 const s=stats();
 setStatus(`${s.chunks} chunks · ${s.bonds} bonds`);
 draw(0,0);
}

function shoot(ray){
 if($('director').checked)return;
 frame=frame.then(()=>{broadside_set_scripted(false);broadside_shoot_ray(new Float32Array(ray.origin),new Float32Array(ray.direction))});
}

function hud(){
 const s=stats();
 $('stats').textContent=`t ${s.simSeconds.toFixed(2)} s   x${broadside_time_scale().toFixed(2)}\nshots ${s.shots}   bonds broken ${s.brokenBonds}\npieces ${s.pieces}   crushed ${s.crushed}   splinters ${s.splinters}`;
}

function tick(t){
 frame=frame.then(async()=>{
  try{
   let steps=0;const wall=last?Math.min(0.1,(t-last)/1000):0;
   if(running&&last){credit+=wall*stats().frameHz*broadside_time_scale();steps=Math.floor(credit);credit-=steps}
   last=t;await advance(steps,credit,wall);hud();
  }catch(e){fail(e);running=false}
 });
 frame.finally(()=>{if(!film)requestAnimationFrame(tick)});
}

// Offline recording: each call renders the next film frame `1/fps` of clock time on (the
// take's slow motion included), however long the physics takes, and returns it as a PNG
// data URL. `done` is true once the take is over.
let film=false;
window.__film={
 async start(){film=true;document.body.classList.add('film');await frame;await open();credit=0},
 async frame(fps=30){
  const wall=1/fps;
  credit+=wall*stats().frameHz*broadside_time_scale();
  const steps=Math.floor(credit);credit-=steps;
  await advance(steps,credit,wall);
  return {png:canvas.toDataURL('image/png'),done:stats().over,stats:stats()};
 },
};

$('restart').onclick=()=>{frame=frame.then(open).catch(fail)};
$('run').onclick=()=>{running=!running;last=0;$('run').textContent=running?'Pause':'Run'};
$('director').onchange=()=>{if(!$('director').checked)renderer.viewMode=0};

try{
 if(!navigator.gpu)throw new Error('WebGPU is unavailable. Open this page in a WebGPU-enabled browser.');
 if(!JSPI)throw new Error('Broadside waits for the GPU with JavaScript Promise Integration: open it in Chrome or Edge 137+.');
 await broadside_init();
 const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw new Error('No WebGPU adapter available');
 device=await adapter.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(16,adapter.limits.maxStorageBuffersPerShaderStage),maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize,maxBufferSize:adapter.limits.maxBufferSize}});
 device.addEventListener('uncapturederror',e=>fail(e.error));
 const r=await fetch('./fracture/assets/galleon-stern.glb');if(!r.ok)throw new Error(`galleon-stern.glb: ${r.status}`);
 glb=await r.arrayBuffer();
 if(new URLSearchParams(location.search).get('film')!=='1'){await open();requestAnimationFrame(tick)}
 else window.__filmReady=true;
}catch(e){fail(e)}
