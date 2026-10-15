import {parseFractureGlb} from './asset.mjs';
import {buildScene} from './geometry.mjs';
import {createSimulation} from './simulation.mjs';
import {createRenderer} from './render.mjs';
import {createRates} from './rates.mjs';
const $=id=>document.getElementById(id),status=$('status');
let device,asset,scene,sim,renderer,running=false,released=false,loading=true,frame=Promise.resolve(),loadTail=Promise.resolve(),last=0,accumulator=0;
let currentRates=null;
const rates=createRates(value=>{
 currentRates=value;
 $('tps').textContent=value?value.tps.toFixed(value.tps>0&&value.tps<10?1:0):'—';
 $('fps').textContent=value?value.fps.toFixed(value.fps>0&&value.fps<10?1:0):'—';
 $('sim-speed').textContent=value?`${value.speed.toFixed(2)}×`:'—';
 $('step-ms').textContent=value?.stepMs?`${value.stepMs.toFixed(1)} ms/tick`:'target 120 TPS';
 $('frame-ms').textContent=value?`${value.frameMs.toFixed(1)} ms/frame`:'frames / second';
 $('timing').textContent=value?.stepMs?`Physics: ${value.encodeMs.toFixed(1)} ms CPU submit + ${value.waitMs.toFixed(1)} ms GPU / readback wait`:'TPS counts physics ticks; FPS counts rendered frames.';
});
document.addEventListener('visibilitychange',()=>{last=0;accumulator=0;rates.reset()});
const setStatus=text=>{status.textContent=text;status.classList.remove('error')};
const fail=e=>{console.error(e);running=false;status.textContent=e.message;status.classList.add('error');$('release').textContent=released?'Resume':'Start simulation'};
const controls=enabled=>{for(const id of ['release','reset','level','metadata'])$(id).disabled=!enabled;for(const id of ['file','sample','scene'])$(id).disabled=!device||loading};
function serializeLoad(task){const result=loadTail.catch(()=>{}).then(async()=>{loading=true;running=false;controls(false);await frame;rates.reset();try{await task()}catch(e){fail(e)}finally{loading=false;last=0;rates.reset();controls(!!sim)}});loadTail=result;return result}
let shotQueue=Promise.resolve();
function shoot(ray){
 if(loading||!sim)return;
 shotQueue=shotQueue.then(async()=>{await frame;if(loading||!sim)return;try{sim.launch(ray.origin,ray.direction);if(!running)rates.reset();released=true;running=true;accumulator=0;$('release').textContent='Pause'}catch(e){fail(e)}});
 return shotQueue;
}
async function loadLevel(level){
 let next;try{next=buildScene(asset,level)}catch(e){if(scene?.selected.length)$('level').value=String(scene.selected[0].level);throw e}
 renderer?.dispose();renderer=null;sim?.dispose();sim=null;released=false;accumulator=0;
 scene=next;sim=await createSimulation(device,scene,setStatus);renderer=await createRenderer(device,$('view'),sim,scene,shoot);
 $('fragments').textContent=scene.selected.length;$('hulls').textContent=scene.colliders.filter(c=>c.shape===3).length;$('bonds').textContent=scene.bonds.length;$('release').textContent='Start simulation';
 setStatus(`${(asset.bytes/1048576).toFixed(2)} MB · ${asset.library.material.length} materials · click the model to shoot`);
}
async function importBytes(bytes,name,sampleKey=""){
 const start=performance.now(),next=parseFractureGlb(bytes);const ms=performance.now()-start;
 asset=next;$('scene').value=sampleKey;const levels=[...new Set(asset.physics.fragments.map(f=>f.level))].sort((a,b)=>a-b);
 $('level').replaceChildren(...levels.map(l=>new Option(`Level ${l} · ${asset.physics.fragments.filter(f=>f.level===l).length} fragments`,l)));$('filename').textContent=name;
 await loadLevel(levels[0]);window.__FRACTURE_DEMO__.parseMs=ms;
}
const samples={house:{file:'veneer-house.glb',label:'house'},arch:{file:'masonry-arch.glb',label:'arch'}};
const sample=(key=$('scene').value||'house')=>serializeLoad(async()=>{
 const entry=samples[key];if(!entry)throw new Error('Unknown bundled scene');
 setStatus(`Fetching baked ${entry.label}…`);
 const response=await fetch(`./assets/${entry.file}`);
 if(!response.ok)throw new Error('Bundled asset missing. Follow README.md to export it.');
 await importBytes(await response.arrayBuffer(),entry.file,key);
});
$('sample').onclick=()=>sample();
$('scene').onchange=()=>sample($('scene').value);
$('file').onchange=()=>{const f=$('file').files[0];if(f)serializeLoad(async()=>importBytes(await f.arrayBuffer(),f.name))};
$('level').onchange=()=>serializeLoad(()=>loadLevel(Number($('level').value)));
$('release').onclick=()=>{released=true;running=!running;$('release').textContent=running?'Pause':'Resume';accumulator=0;rates.reset()};
$('reset').onclick=()=>serializeLoad(async()=>{sim.reset();released=false;$('release').textContent='Start simulation';setStatus('Reset · click the model to shoot')});
$('metadata').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify({physics:asset.physics,materialLibrary:asset.library},null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='fracture-metadata.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
for(const type of ['dragenter','dragover'])window.addEventListener(type,e=>{e.preventDefault();document.body.classList.add('drop')});
window.addEventListener('dragleave',()=>document.body.classList.remove('drop'));
window.addEventListener('drop',e=>{e.preventDefault();document.body.classList.remove('drop');const f=e.dataTransfer.files[0];if(f&&device)serializeLoad(async()=>importBytes(await f.arrayBuffer(),f.name))});
window.__FRACTURE_DEMO__={get ready(){return !loading&&!!sim},get asset(){return asset},get scene(){return scene},get simulation(){return sim},get renderer(){return renderer},get rates(){return currentRates},
 async shootRay(origin,direction){running=false;await frame;return sim.launch(origin,direction)},
 async stepFrames(count){running=false;await frame;for(let i=0;i<count;i++)await sim.step();renderer.draw($('show-bonds').checked);return sim.counters},
 async snapshot(){await frame;return sim.snapshot()},reset:()=>serializeLoad(async()=>sim.reset()),load:sample,selectLevel:level=>serializeLoad(async()=>{$("level").value=String(level);await loadLevel(level)})};
function tick(now){
 const elapsed=last?Math.min(.04,(now-last)/1000):0;last=now;
 if(!loading&&renderer){frame=(async()=>{try{
  let work=null,ticks=0;
  if(running){
   accumulator=Math.min(accumulator+elapsed,4/120);
   // Bound outstanding work. Slow ticks get one tick per frame, rather than
   // four serial GPU waits starving camera input and rendering.
   const budget=Math.max(1,Math.min(4,Math.floor(12/Math.max(sim.timing.wallMs,0.1))));
   ticks=Math.min(budget,Math.floor(accumulator*120));
   if(ticks){accumulator-=ticks/120;work=sim.step(ticks)}
  }
  // Queue the draw immediately behind physics; do not wait for CPU stats first.
  renderer.draw($('show-bonds').checked);
  if(work){await work;if(!document.hidden)for(let i=0;i<ticks;i++)rates.step(sim.timing)}
  if(running){const c=sim.counters;setStatus(`${sim.shots}/16 shots · ${sim.steps} steps · ${c.contact} contacts · ${c.awake} awake · ${(sim.steps/120).toFixed(1)} s`)}
  if(!document.hidden)rates.frame();
 }catch(e){fail(e)}})();frame.finally(()=>requestAnimationFrame(tick))}else requestAnimationFrame(tick);
}
try{
 if(!navigator.gpu)throw new Error('WebGPU is unavailable. Open this localhost page in a WebGPU-enabled browser.');
 const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw new Error('No WebGPU adapter available');
 device=await adapter.requestDevice({requiredFeatures:adapter.features.has('timestamp-query')?['timestamp-query']:[],requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(16,adapter.limits.maxStorageBuffersPerShaderStage),maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize,maxBufferSize:adapter.limits.maxBufferSize}});
 device.addEventListener('uncapturederror',e=>fail(e.error));device.lost.then(info=>fail(new Error(`GPU device lost: ${info.message}. Reload this page.`)));
 await sample();requestAnimationFrame(tick);
}catch(e){fail(e);controls(false)}
