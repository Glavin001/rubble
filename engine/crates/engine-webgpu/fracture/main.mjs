import {parseFractureGlb} from './asset.mjs';
import {buildScene,buildGrid,buildSuburb} from './geometry.mjs';
import {buildCuboidHouse} from './cuboid-house.mjs';
import {createSimulation} from './simulation.mjs';
import {preloadBonds} from './bonds.mjs';
import {createRenderer} from './render.mjs';
import {createRates} from './rates.mjs';
const $=id=>document.getElementById(id),status=$('status');
let device,asset,scene,sim,renderer,running=false,released=false,loading=true,frame=Promise.resolve(),loadTail=Promise.resolve(),last=0,accumulator=0;
// Heavier steps than this are split across frames (see simulation encodeSlices);
// back to whole steps once few bodies are awake.
const SLICED_STEP_MS=12,SLICE_EXIT_AWAKE=20000;
// Slices submitted per frame adapt to the frame time: more while frames stay
// under the 60 FPS budget, fewer above it, so physics gets every spare
// millisecond without dropping rendering below ~60 FPS.
const FRAME_BUDGET_MS=16.0;
let sliceRate=1,sliceCredit=0,frameEma=FRAME_BUDGET_MS;
let currentRates=null,layout=null,maxHullVertices,procedural=null;
// Physics ticks per second follow the scene's dt (120 Hz unless the scene says otherwise).
const stepHz=()=>Math.round(1/(scene?.params?.dt||1/120));
const rates=createRates(value=>{
 currentRates=value;
 $('tps').textContent=value?value.tps.toFixed(value.tps>0&&value.tps<10?1:0):'—';
 $('fps').textContent=value?value.fps.toFixed(value.fps>0&&value.fps<10?1:0):'—';
 $('sim-speed').textContent=value?`${value.speed.toFixed(2)}×`:'—';
 $('step-ms').textContent=value?.stepMs?`${value.stepMs.toFixed(1)} ms/tick`:`target ${stepHz()} TPS`;
 $('frame-ms').textContent=value?`${value.frameMs.toFixed(1)} ms/frame`:'frames / second';
 $('timing').textContent=value?.stepMs?`Physics: ${value.encodeMs.toFixed(1)} ms CPU submit + ${value.waitMs.toFixed(1)} ms GPU / readback wait`:'TPS counts physics ticks; FPS counts rendered frames.';
},undefined,()=>stepHz());
document.addEventListener('visibilitychange',()=>{last=0;accumulator=0;rates.reset()});
const setStatus=text=>{status.textContent=text;status.classList.remove('error')};
const fail=e=>{console.error(e);running=false;status.textContent=e.message;status.classList.add('error');$('release').textContent=released?'Resume':'Start simulation'};
const controls=enabled=>{for(const id of ['release','reset','level','metadata','fire'])$(id).disabled=!enabled;for(const id of ['file','sample','scene'])$(id).disabled=!device||loading};
function serializeLoad(task){const result=loadTail.catch(()=>{}).then(async()=>{loading=true;running=false;controls(false);await frame;rates.reset();try{await task()}catch(e){fail(e)}finally{loading=false;last=0;rates.reset();controls(!!sim)}});loadTail=result;return result}
let shotQueue=Promise.resolve();
function shoot(ray){
 if(loading||!sim)return;
 shotQueue=shotQueue.then(async()=>{await frame;if(loading||!sim)return;try{sim.launch(ray.origin,ray.direction);if(!running)rates.reset();released=true;running=true;accumulator=0;$('release').textContent='Pause'}catch(e){fail(e)}});
 return shotQueue;
}
async function loadLevel(level){
 let next;try{
  // Settle one house under its own weight first, so every copy starts asleep in
  // static equilibrium (bonds carrying their load) rather than at zero force.
  // Breakable bonds are experimental (opt-in): see README "Breakable bonds".
  const built=procedural?buildCuboidHouse():buildScene(asset,level,{maxHullVertices});
  if($('solve-bonds').checked)setStatus('Settling bonds under self-weight…');
  const one=$('solve-bonds').checked?await preloadBonds(device,built):{...built,joints:[]};
  next=layout?(layout.suburb?buildSuburb:buildGrid)(asset,level,{...layout,maxHullVertices,base:one}):one;
  // A brick in a wall touches up to six neighbours (four contacts each).
  if(procedural)next={...next,capacity:{...next.capacity,candidatesPerCollider:12,contactsPerCollider:24}}}catch(e){if(scene?.selected.length)$('level').value=String(scene.selected[0].level);throw e}
 renderer?.dispose();renderer=null;sim?.dispose();sim=null;released=false;accumulator=0;
 scene=next;sim=await createSimulation(device,scene,setStatus);renderer=await createRenderer(device,$('view'),sim,scene,shoot);
 renderer.colorMode=$('show-sleep').checked?1:0;
 const copies=scene.copies||1;$('fragments').textContent=scene.selected.length*copies;$('hulls').textContent=scene.colliders.filter(c=>c.shape===(procedural?2:3)).length;$('bonds').textContent=scene.bonds.length*copies;$('release').textContent='Start simulation';
 $('fire').hidden=!scene.rowShot;$('awake').textContent='0';$('contacts').textContent='0';
 const skipped=scene.skippedHulls.length?` · ${scene.skippedHulls.length} flat zero-volume hulls skipped${copies>1?' per house':''}`:'';
 const what=copies>1?`${copies} houses · ${scene.bodies.length-1-scene.projectileCount} bodies · `:'';
 const source=procedural?'procedural cuboid houses (box colliders)':`${(asset.bytes/1048576).toFixed(2)} MB · ${asset.library.material.length} materials${skipped}`;
 setStatus(`${what}${source} · ${scene.rowShot?'fire the ball through the row or click to shoot':'click the model to shoot'}`);
}
async function importBytes(bytes,name,sampleKey="",entry={}){
 layout=entry.layout??null;maxHullVertices=entry.maxHullVertices;procedural=null;
 const start=performance.now(),next=parseFractureGlb(bytes);const ms=performance.now()-start;
 asset=next;$('scene').value=sampleKey;const levels=[...new Set(asset.physics.fragments.map(f=>f.level))].sort((a,b)=>a-b);
 $('level').replaceChildren(...levels.map(l=>new Option(`Level ${l} · ${asset.physics.fragments.filter(f=>f.level===l).length} fragments`,l)));$('filename').textContent=name;
 const level=levels.includes(entry.level)?entry.level:levels[0];$('level').value=String(level);
 await loadLevel(level);window.__FRACTURE_DEMO__.parseMs=ms;
}
const samples={house:{file:'veneer-house.glb',label:'house'},row:{file:'veneer-house-hull16.glb',label:'20 houses',layout:{columns:1,rows:20},level:3,maxHullVertices:16},
 grid:{file:'veneer-house.glb',label:'49 houses',layout:{columns:7,rows:7},level:2},
 suburb:{file:'veneer-house.glb',label:'48 houses',layout:{suburb:true},level:2},
 cuboids:{procedural:'cuboid',label:'48 cuboid houses',layout:{suburb:true}},
 cuboidGrid:{procedural:'cuboid',label:'49 cuboid houses',layout:{columns:7,rows:7}},arch:{file:'masonry-arch.glb',label:'arch'}};
const sample=(key=$('scene').value||'house')=>serializeLoad(async()=>{
 const entry=samples[key];if(!entry)throw new Error('Unknown bundled scene');
 if(entry.procedural){
  // Built in code from box colliders: no GLB, one level, no baked bonds.
  layout=entry.layout??null;procedural=entry.procedural;asset=null;$('scene').value=key;$('filename').textContent='Procedural cuboid house';
  $('level').replaceChildren(new Option('Cuboids · box colliders only',0));$('level').value='0';
  await loadLevel(0);return;
 }
 setStatus(`Fetching baked ${entry.label}…`);
 const response=await fetch(`./assets/${entry.file}`);
 if(!response.ok)throw new Error('Bundled asset missing. Follow README.md to export it.');
 await importBytes(await response.arrayBuffer(),entry.file,key,entry);
});
$('sample').onclick=()=>sample();
$('fire').onclick=()=>scene?.rowShot&&shoot(scene.rowShot);
$('scene').onchange=()=>sample($('scene').value);
$('file').onchange=()=>{const f=$('file').files[0];if(f)serializeLoad(async()=>importBytes(await f.arrayBuffer(),f.name))};
$('level').onchange=()=>serializeLoad(()=>loadLevel(Number($('level').value)));
$('show-sleep').onchange=()=>{if(renderer)renderer.colorMode=$('show-sleep').checked?1:0};
$('solve-bonds').onchange=()=>serializeLoad(()=>loadLevel(Number($('level').value)));
$('release').onclick=()=>{released=true;running=!running;$('release').textContent=running?'Pause':'Resume';accumulator=0;rates.reset()};
$('reset').onclick=()=>serializeLoad(async()=>{sim.reset();released=false;$('release').textContent='Start simulation';setStatus('Reset · click the model to shoot')});
$('metadata').onclick=()=>{if(!asset)return;const url=URL.createObjectURL(new Blob([JSON.stringify({physics:asset.physics,materialLibrary:asset.library},null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='fracture-metadata.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
for(const type of ['dragenter','dragover'])window.addEventListener(type,e=>{e.preventDefault();document.body.classList.add('drop')});
window.addEventListener('dragleave',()=>document.body.classList.remove('drop'));
window.addEventListener('drop',e=>{e.preventDefault();document.body.classList.remove('drop');const f=e.dataTransfer.files[0];if(f&&device)serializeLoad(async()=>importBytes(await f.arrayBuffer(),f.name))});
window.__FRACTURE_DEMO__={get sliceRate(){return sliceRate},get ready(){return !loading&&!!sim},get device(){return device},get asset(){return asset},get scene(){return scene},get simulation(){return sim},get renderer(){return renderer},get rates(){return currentRates},
 async shootRay(origin,direction){running=false;await frame;return sim.launch(origin,direction)},
 async fireRow(){running=false;await frame;return sim.launch(scene.rowShot.origin,scene.rowShot.direction)},
 // Repeatable row benchmark: fire, then every `every` steps record wall ms per
 // step, the ball, counters and (optionally) a per-phase GPU profile.
 async benchmarkRow({steps=600,every=60,profile=true}={}){
  running=false;await frame;const id=sim.launch(scene.rowShot.origin,scene.rowShot.direction),log=[];
  for(let done=0;done<steps;){
   const n=Math.min(every,steps-done),t=performance.now();for(let k=0;k<n;k+=4)await sim.step(Math.min(4,n-k));done+=n;
   const ms=(performance.now()-t)/n,snap=await sim.snapshot(),c=sim.counters,o=id*28;
   const p=profile?await sim.profileStep():null;if(p)done++;
   log.push({step:done,wallMs:+ms.toFixed(2),ballZ:+snap[o+2].toFixed(1),ballVz:+snap[o+14].toFixed(1),awake:c.awake,contacts:c.contact,
    candidates:c.candidate,small:c.convexSmall,large:c.convexLarge,colors:c.color,overflow:c.overflow,
    gpuMs:p?+p.gpuMs.toFixed(2):null,phases:p?Object.fromEntries(Object.entries(p.phases).map(([k,v])=>[k,+v.toFixed(2)])):null});
  }
  renderer.draw($('show-bonds').checked);return log;
 },
 async stepFrames(count){running=false;await frame;for(let i=0;i<count;i++)await sim.step();renderer.draw($('show-bonds').checked);return sim.counters},
 async snapshot(){await frame;return sim.snapshot()},reset:()=>serializeLoad(async()=>sim.reset()),load:sample,selectLevel:level=>serializeLoad(async()=>{$("level").value=String(level);await loadLevel(level)})};
function tick(now){
 const frameMs=last?now-last:FRAME_BUDGET_MS;
 const elapsed=last?Math.min(.04,(now-last)/1000):0;last=now;
 // Damped control on a smoothed frame time: cut quickly when over budget,
 // raise slowly when under, so one heavy frame does not swing the rate.
 frameEma+=0.2*(frameMs-frameEma);
 sliceRate=Math.min(4,Math.max(0.25,sliceRate*Math.min(1.03,Math.max(0.7,FRAME_BUDGET_MS/frameEma))));
 if(!loading&&renderer){frame=(async()=>{try{
  let work=null,ticks=0;
  if(running){
   accumulator=Math.min(accumulator+elapsed,4/stepHz());
   if(sim.slicing){
    // A heavy step in flight: this frame's share of slices, then its counters.
    sliceCredit+=sliceRate;const n=Math.floor(sliceCredit);sliceCredit-=n;
    // Counters arrive asynchronously, so the queue keeps flowing.
    if(sim.submitSlices(n))sim.finishSlicedStep().then(()=>{if(!document.hidden)rates.step(sim.timing)}).catch(fail);
   }else{
    // Bound outstanding work. Slow ticks get one tick per frame, rather than
    // four serial GPU waits starving camera input and rendering.
    const budget=Math.max(1,Math.min(4,Math.floor(12/Math.max(sim.timing.wallMs,0.1))));
    ticks=Math.min(budget,Math.floor(accumulator*stepHz()));
    if(ticks){
     accumulator-=ticks/stepHz();
     // Steps over ~12 ms run sliced across frames so rendering keeps its rate;
     // the physics is the same, only spread over several frames.
     const heavy=sim.timing.wallMs>SLICED_STEP_MS&&!(sim.timing.sliced&&sim.counters.awake<SLICE_EXIT_AWAKE);
     if(heavy){ticks=0;if(sim.beginSlicedStep()&&sim.submitSlices(1))sim.finishSlicedStep().then(()=>{if(!document.hidden)rates.step(sim.timing)}).catch(fail)}
     else work=sim.step(ticks);
    }
   }
  }
  // Queue the draw immediately behind physics; do not wait for CPU stats first.
  renderer.draw($('show-bonds').checked);
  if(work){await work;if(!document.hidden)for(let i=0;i<ticks;i++)rates.step(sim.timing)}
  if(running){const c=sim.counters;$("awake").textContent=c.awake;$("contacts").textContent=c.contact}
  if(running){const c=sim.counters;setStatus(`${sim.shots}/16 shots · ${sim.steps} steps · ${c.contact} contacts · ${c.awake} awake · ${(sim.steps/stepHz()).toFixed(1)} s`)}
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
