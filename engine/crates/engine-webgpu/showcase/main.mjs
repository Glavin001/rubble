// The showcase: rubble's world API in the browser. A scenario (a scene pack with driven
// vehicles, engine-world `scenario`) is stepped in Rust on the wasm module's own WebGPU
// device (engine-world-wasm); each step's poses come back as the fracture lab's draw buffer
// and this page draws them with the lab's renderer (fracture/render.mjs), the structure's
// meshes built from the same pack (fracture/wasm.mjs buildPackScene). Takes and instant
// replay are the lab's director (fracture/director.mjs) over this page. The physics, the
// poses and the shots are the Rust's; this file forwards input and uploads.
import init,{showcase_init,showcase_scenarios,showcase_select,showcase_pack,showcase_error,showcase_panic,showcase_vertices,showcase_poses,showcase_shoot_ray,showcase_fire,showcase_charges,showcase_detonate,showcase_stats} from '../generated/world-wasm/engine_world_wasm.js';
import {JSPI} from '../fracture/gpu-wait.mjs';
import {buildPackScene} from '../fracture/wasm.mjs';
import {createRenderer} from '../fracture/render.mjs';
import {createDirector} from '../fracture/director.mjs';

const $=id=>document.getElementById(id);
const status=$('status'),canvas=$('view');
const setStatus=text=>{status.textContent=text;status.classList.remove('error')};
function fail(e){console.error(e);status.textContent=String(e?.message??e);status.classList.add('error')}

let device,renderer=null,scene=null,poseBuffer=null,key=null;
// `held`: the director has the page; `loading`: a scenario is being built. Either stops the
// frame loop. `frame`: the frame in flight (wasm calls wait for it: a frame suspends in
// the module, which holds the scenario meanwhile).
let running=true,held=false,loading=false,last=0,credit=0,frame=Promise.resolve();
const after=f=>frame.then(f);
// Physics steps per simulated second (the scenario's rate) and simulated time per wall second.
const stats=()=>JSON.parse(showcase_stats());
const timeScale=()=>Number($('time-scale').value);

const wasm=await init({module_or_path:new URL('../generated/world-wasm/engine_world_wasm_bg.wasm',import.meta.url)});
// Exports that wait for the GPU suspend the wasm stack (JSPI, fracture/gpu-wait.mjs).
const why=f=>async(...a)=>{try{return await f(...a)}catch(e){const p=showcase_panic();throw p?new Error(`${p} (${e?.message??e})`):e}};
const loadSelected=why(WebAssembly.promising?WebAssembly.promising(wasm.showcase_load):()=>false);
const load=key=>{showcase_select(key);return loadSelected()};
const stepFrames=why(WebAssembly.promising?WebAssembly.promising(wasm.showcase_frame):()=>false);

// Advance `steps` steps (0: none), then upload the poses and draw.
async function advance(steps){
 if(steps>0&&!await stepFrames(steps))throw new Error(showcase_error());
 const poses=showcase_poses();device.queue.writeBuffer(poseBuffer,0,poses);renderer.draw(false);
}

// The scenario `next`: its world (Rust), its render scene (the pack's meshes, then the
// vehicles' and balls'), the pose buffer and the renderer; running live, or paused (`run`
// false: a take's setup, which must not step before the take starts).
async function open(next,run=true){
 loading=true;
 try{await frame;await build(next,run)}finally{loading=false}
}
async function build(next,run){
 controls(false);setStatus(`Loading ${next}…`);
 if(!await load(next))throw new Error(showcase_error());
 // The pack's render scene (camera, floor), every mesh from the Rust (coloured by material).
 const built=buildPackScene(showcase_pack(next).buffer);
 scene={handle:built.handle,...built.handle.meta(),hullData:built.hullData,vertices:showcase_vertices(),lines:built.lines,projectileVertices:built.projectileVertices,roadVertices:built.roadVertices};
 const poses=showcase_poses();
 poseBuffer?.destroy();poseBuffer=device.createBuffer({size:poses.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
 device.queue.writeBuffer(poseBuffer,0,poses);
 renderer?.dispose();renderer=await createRenderer(device,canvas,{bodyBuffer:poseBuffer,shots:0},scene,shoot);
 key=next;director.reset();last=0;credit=0;running=run;$('run').textContent=running?'Pause':'Run';
 const s=stats();
 $('detonate').hidden=!s.charges.length;$('fire').hidden=!s.canFire;
 controls(true);setStatus(`${next}: ${s.drives.length} vehicles · ${s.chunks} chunks · ${s.bonds} bonds`);
 renderer.draw(false);
}

function controls(on){for(const id of ['run','restart','detonate','fire','replay'])$(id).disabled=!on}

function shoot(ray){if(!scene||held)return;after(()=>{if(showcase_shoot_ray(new Float64Array(ray.origin),new Float64Array(ray.direction)))director.logShot(ray.origin,ray.direction)})}

// The lab's director over this page: a stress-mode-like view of the scenario.
const view={
 get stats(){const s=stats();return {...s,statsFrame:s.frames,fragments:s.bodies}},
 frameSteps:n=>advance(n),
 detonate:name=>showcase_detonate(name),
 launch:()=>false,
 fire:()=>showcase_fire(),
 shootRay:(o,d)=>showcase_shoot_ray(new Float64Array(o),new Float64Array(d)),
 setOrbit:(...a)=>renderer.setOrbit(...a),
 get orbit(){return renderer.orbit},
};
const director=createDirector({
 stressOn:true,stress:view,get renderer(){return renderer},get scene(){return scene},canvas:()=>canvas,stepHz:()=>stats().frameHz,showBonds:()=>false,
 async pause(){held=true;await frame},
 resume(){held=false;last=0;credit=0;running=true;$('run').textContent='Pause'},
 reload:()=>open(key,false),
 aim:(u,v)=>renderer.aimAt(u,v),
 // The draw's colour modes (0 materials, 1 sleep state, 2 bodies); a take's untinted materials are 0.
 setView({color}={}){if(color!==undefined&&renderer)renderer.colorMode=color===3?0:color},
 shotSettings:()=>({mass:0,speed:0}),applyShotSettings(){},
 stats(){const s=stats();return `${s.bodies} pieces · ${s.brokenBonds} bonds broken · ${s.contacts} contacts · ${s.simSeconds.toFixed(2)} s`},
 async fullFrame(on){const was=document.body.classList.contains('collapsed');document.body.classList.toggle('collapsed',on);await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));return was},
 counters(){const s=stats();return {frames:s.frames,statsFrame:s.frames,simSeconds:s.simSeconds,brokenBonds:s.brokenBonds,fragments:s.bodies,bodies:s.bodies,chunks:s.chunks,lastShot:null}},
 async summary(){const s=stats();return {frames:s.frames,fragments:s.bodies,brokenBonds:s.brokenBonds,bodies:s.bodies}},
});
window.__record=(script,opts)=>director.record(script,opts);
window.__replay=opts=>director.replay(opts);
window.__SHOWCASE__={open:k=>open(k,false),get ready(){return !!scene&&!held},stats};

function hud(){
 const s=stats();
 $('bonds-broken').textContent=s.brokenBonds;$('bodies').textContent=s.bodies;$('contacts').textContent=s.contacts;
 $('drives').textContent=s.drives.map((d,i)=>`${d.padEnd(12)} ${Math.round(s.driveKmh[i]??0).toString().padStart(4)} km/h`).join('\n');
 $('detonate').disabled=!s.charges.length||s.chargesArmed>0;$('fire').disabled=!s.canFire;
}

function tick(t){
 if(scene&&!held&&!loading){
  frame=(async()=>{
   try{
    let steps=0;
    if(running&&last){credit+=Math.min(0.1,(t-last)/1000)*stats().frameHz*timeScale();steps=Math.floor(credit);credit-=steps}
    last=t;
    await advance(steps);hud();
   }catch(e){fail(e);running=false;$('run').textContent='Run'}
  })();
  frame.finally(()=>requestAnimationFrame(tick));
 }else requestAnimationFrame(tick);
}

$('scenario').onchange=()=>open($('scenario').value).catch(fail);
$('restart').onclick=()=>open(key).catch(fail);
$('run').onclick=()=>{running=!running;last=0;$('run').textContent=running?'Pause':'Run'};
$('detonate').onclick=()=>after(()=>{if(showcase_detonate(''))director.logDetonate('')});
$('fire').onclick=()=>after(()=>{if(showcase_fire())director.logFire()});
async function replay(){
 if(!director.shots)return;
 $('replay').disabled=true;
 try{setStatus('Replay: re-running the scenario from its load…');
  const r=await director.replay({onProgress:p=>{status.textContent=`Replay ${Math.round(p*100)}%`}});
  if(r)setStatus(r.same?'Replay done: it reached the same state, carrying on live':`Replay done, but it ended in a different state (${JSON.stringify(r.after)} vs ${JSON.stringify(r.before)})`)}
 catch(e){fail(e)}finally{$('replay').disabled=false}
}
$('replay').onclick=replay;
addEventListener('keydown',e=>{if(e.code==='KeyR'&&!e.repeat&&!e.ctrlKey&&!e.metaKey&&!e.altKey&&!/^(INPUT|SELECT|TEXTAREA)$/.test(e.target?.tagName))replay()});

try{
 if(!navigator.gpu)throw new Error('WebGPU is unavailable. Open this page in a WebGPU-enabled browser.');
 if(!JSPI)throw new Error('The showcase waits for the GPU with JavaScript Promise Integration: open it in Chrome or Edge 137+.');
 await showcase_init();
 const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw new Error('No WebGPU adapter available');
 device=await adapter.requestDevice();
 device.addEventListener('uncapturederror',e=>fail(e.error));
 const list=JSON.parse(showcase_scenarios());
 $('scenario').innerHTML=list.map(s=>`<option value="${s.key}">${s.group} · ${s.menu}</option>`).join('');
 const wanted=new URLSearchParams(location.search).get('scenario');
 await open(list.some(s=>s.key===wanted)?wanted:list[0].key);
 $('scenario').value=key;
 requestAnimationFrame(tick);
}catch(e){fail(e);controls(false)}
