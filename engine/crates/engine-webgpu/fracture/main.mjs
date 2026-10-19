import {parseFractureGlb,buildScene,buildCuboidHouse,buildPackScene,layoutScene,withoutJoints,withBondStrength,SAMPLES} from './wasm.mjs';
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
// millisecond without dropping rendering below ~60 FPS. The budget sits just
// above one 60 Hz vsync interval (16.7 ms): at 16.0 every vsync-paced frame
// read as over budget and the rate sank to its floor (2-4 ticks/s).
const FRAME_BUDGET_MS=17.5,MAX_SLICE_RATE=24;
let sliceRate=1,sliceCredit=0,frameEma=FRAME_BUDGET_MS;
// Whether the GPU finished the work submitted two frames ago before this
// frame began (queue.onSubmittedWorkDone): then it is about to run dry, so
// submit more slices; otherwise fewer. One frame of work stays queued, so the
// GPU never idles between frames. Vsync-paced frame times jump between 16.7
// and 33 ms, too coarse to steer by.
let workDone=[{done:true},{done:true}];
let currentRates=null,layout=null,maxHullVertices,procedural=null,assetBytes=null,packBytes=null;
// Bond mode (the Bonds menu): 'none' (pieces sleep until hit), 'd6' (breakable D6
// joints, AVBD welds) or 'stress' (the GPU stress solver, Rust on its own canvas:
// stress-mode.mjs). The stress mode keeps its own frame loop state.
const bondMode=()=>$('bond-mode').value;
let stress=null,stressOn=false,stressHud={t:0,frames:0,renders:0};
// The bond-graph and sleep-colour views are this page's renderer's; the stress mode draws
// with its own (Rust), which has neither: the boxes are off there, saying so.
const STRESS_VIEW_NOTE='Not in the stress-solver mode: it draws with its own renderer';
function viewOptions(stressMode){for(const id of ['show-bonds','show-sleep']){const box=$(id),label=box.closest('label');box.disabled=stressMode;if(stressMode){label.dataset.title??=label.title;label.title=STRESS_VIEW_NOTE}else if(label.dataset.title!==undefined){label.title=label.dataset.title;delete label.dataset.title}}}
// Ball mass: a power-of-two multiple of the scene's default ball, applied to the next shots.
const massScale=()=>2**Number($('ball-mass').value);
// Bond strength: a power-of-two multiple of every bond's break strength (D6 joints and
// the stress solver), applied when the level loads.
const strengthScale=()=>2**Number($('bond-strength').value);
const scaleLabel=k=>`×${k>=1?k:`1/${1/k}`}`;
function applyBallMass(){
 const k=massScale(),kg=(stressOn?stress?.stats.ballMass:scene?.projectileMass)*k;
 $('ball-mass-value').textContent=`×${k>=1?k:`1/${1/k}`}${kg?` · ${Math.round(kg).toLocaleString()} kg`:''}`;
 if(stressOn)stress?.setShotMassScale(k);else if(sim&&scene?.projectileMass)sim.setProjectileMass(scene.projectileMass*k);
}
// Physics ticks per second follow the scene's dt (120 Hz unless the scene says otherwise).
const stepHz=()=>Math.round(1/(scene?.params?.dt||1/120));
const rates=createRates(value=>{
 currentRates=value;
 $('tps').textContent=value?value.tps.toFixed(value.tps>0&&value.tps<10?1:0):'—';
 $('fps').textContent=value?value.fps.toFixed(value.fps>0&&value.fps<10?1:0):'—';
 $('sim-speed').textContent=value?`${value.speed.toFixed(2)}×`:'—';
 // GPU time from timestamps (sampled ticks and frames) and the rate it alone
 // would allow if nothing else used the GPU (physics and rendering share it).
 const tickMs=sim?.gpuTickMs,frameMs=renderer?.gpuFrameMs;
 $('step-ms').textContent=value&&!value.stepMs&&sim?.counters.awake===0?'all asleep · no GPU work':
  value&&tickMs?`${tickMs.toFixed(1)} ms GPU/tick · max ${Math.floor(1000/tickMs)} TPS`:
  value?.stepMs?`${value.stepMs.toFixed(1)} ms/tick`:`target ${stepHz()} TPS`;
 $('frame-ms').textContent=value&&frameMs?`${frameMs.toFixed(1)} ms GPU/frame · max ${Math.floor(1000/frameMs)} FPS`:
  value?`${value.frameMs.toFixed(1)} ms/frame`:'frames / second';
 // Per tick: CPU time encoding and submitting the GPU work, then time spent
 // waiting on the GPU (the readback, or a full queue of overlapped ticks).
 // Sliced ticks run between frames, so no wait is measured for them.
 $('timing').textContent=!value?.stepMs?'TPS counts physics ticks; FPS counts rendered frames.':
  value.sliced?`Physics per tick: ${value.encodeMs.toFixed(1)} ms CPU encode; GPU work spread across frames`:
  `Physics per tick: ${value.encodeMs.toFixed(1)} ms CPU encode + ${value.waitMs.toFixed(1)} ms waiting on the GPU`;
},undefined,()=>stepHz());
document.addEventListener('visibilitychange',()=>{last=0;accumulator=0;rates.reset()});
const setStatus=text=>{status.textContent=text;status.classList.remove('error')};
const fail=e=>{console.error(e);running=false;status.textContent=e.message;status.classList.add('error');$('release').textContent=released?'Resume':'Start simulation'};
const controls=enabled=>{enabled||=stressOn&&!loading;for(const id of ['release','reset','level','metadata','fire'])$(id).disabled=!enabled;for(const id of ['file','sample','scene'])$(id).disabled=!device||loading};
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
  if(bondMode()==='stress')return loadStress(level);
  stressOn=false;$('view').hidden=false;$('stress-view').hidden=true;if(stress)stress.running=false;viewOptions(false);
  const built=procedural==='pack'?buildPackScene(packBytes):procedural?buildCuboidHouse():buildScene(asset,level,{maxHullVertices});
  if(bondMode()==='d6')setStatus('Settling bonds under self-weight…');
  const one=bondMode()==='d6'?await preloadBonds(device,strengthScale()===1?built:withBondStrength(built,strengthScale())):withoutJoints(built);
  // Copies laid out as the sample says (procedural houses get a larger contact budget).
  next=layoutScene(one,layout,!!procedural)}catch(e){if(scene)$('level').value=String(scene.level??0);throw e}
 renderer?.dispose();renderer=null;sim?.dispose();sim=null;released=false;accumulator=0;
 scene=next;sim=await createSimulation(device,scene,setStatus);renderer=await createRenderer(device,$('view'),sim,scene,shoot);
 renderer.colorMode=$('show-sleep').checked?1:0;applyBallMass();
 const copies=scene.copies||1;$('fragments').textContent=scene.selected.length*copies;$('hulls').textContent=procedural?scene.pieces*copies:scene.hullCount;$('bonds').textContent=scene.bondCount*copies;$('release').textContent='Start simulation';
 $('fire').hidden=!scene.rowShot;$('awake').textContent='0';$('contacts').textContent='0';
 const skipped=scene.skippedHulls.length?` · ${scene.skippedHulls.length} flat zero-volume hulls skipped${copies>1?' per house':''}`:'';
 const what=copies>1?`${copies} houses · ${scene.bodyCount-1-scene.projectileCount} bodies · `:'';
 const source=procedural==='pack'?`scene pack · ${scene.pieces} nodes`:procedural?'procedural cuboid houses (box colliders)':`${(asset.bytes/1048576).toFixed(2)} MB · ${asset.materialCount} materials${skipped}`;
 setStatus(`${what}${source} · ${scene.rowShot?'fire the ball through the row or click to shoot':'click the model to shoot'}`);
}
// The stress-solver mode, built and run by stress-web: a GLB's level (one copy of the
// structure; the sample's layout of copies is the other modes'), or the cuboid houses in
// their sample's layout (every copy).
async function loadStress(level){
 // A fracture GLB's level (one copy of the structure), or the cuboid houses in their
 // sample's layout (every copy). Scene packs are not in this mode yet.
 const cuboid=procedural==='cuboid';
 if(!cuboid&&(procedural||!assetBytes))throw new Error('The stress-solver mode runs a fracture GLB or the cuboid houses: pick one of those scenes.');
 renderer?.dispose();renderer=null;sim?.dispose();sim=null;released=false;
 $('view').hidden=true;$('stress-view').hidden=false;viewOptions(true);
 setStatus('Loading the stress solver (Rust and Slang on WebGPU)…');
 // The mode is on only once its module loaded: a failed import (an old build, no JSPI)
 // leaves the page in its own renderer with the error shown.
 try{if(!stress){const {createStressMode}=await import('./stress-mode.mjs');stress=await createStressMode($('stress-view'))}}
 catch(e){$('view').hidden=false;$('stress-view').hidden=true;viewOptions(false);throw new Error(`Stress-solver mode unavailable: ${e.message}. Rebuild it (README, Bond modes).`)}
 stressOn=true;
 stress.setBondStrengthScale(strengthScale());
 if(cuboid)await stress.loadCuboid(layout);else await stress.load(assetBytes,level);
 await stress.frame(0);applyBallMass();
 const st=stress.stats;stressHud={t:performance.now(),frames:0,renders:0};
 $('fragments').textContent=st.chunks;$('bonds').textContent=st.bonds;$('hulls').textContent=st.hulls;$('awake').textContent='0';$('contacts').textContent='—';
 $('release').textContent='Start simulation';$('fire').hidden=true;
 setStatus(`Stress solver · ${st.chunks} chunks · ${st.bonds} bonds · blast_parity, resident GPU frame at ${Math.round(st.frameHz)} Hz${cuboid?'':' · one copy'} · click the model to shoot`);
}
async function importBytes(bytes,name,sampleKey="",entry={}){
 assetBytes=bytes;layout=entry.layout??null;maxHullVertices=entry.maxHullVertices;procedural=null;
 const start=performance.now(),next=parseFractureGlb(bytes);const ms=performance.now()-start;
 asset?.handle.free();asset=next;$('scene').value=sampleKey;
 $('level').replaceChildren(...asset.levels.map(l=>new Option(`Level ${l} · ${asset.levelFragments(l)} fragments`,l)));$('filename').textContent=name;
 const level=asset.initialLevel(entry.level);$('level').value=String(level);
 await loadLevel(level);window.__FRACTURE_DEMO__.parseMs=ms;
}
// The Scene menu is the shared catalogue (engine-fracture samples.rs), as in the Metal viewer.
const samples=Object.fromEntries(SAMPLES.map(s=>[s.key,{...s,procedural:s.file?null:'cuboid'}]));
// Grouped as the catalogue groups them (engine-fracture samples.rs GROUPS).
const optgroup=(label,options)=>{const g=document.createElement('optgroup');g.label=label;g.append(...options);return g};
const sceneGroups=[...new Set(SAMPLES.map(s=>s.group))].map(label=>optgroup(label,SAMPLES.filter(s=>s.group===label).map(s=>new Option(s.menu,s.key,s.key==='house',s.key==='house'))));
$('scene').replaceChildren(...sceneGroups,optgroup('Your file',[Object.assign(new Option('Custom GLB (choose or drop one below)',''),{disabled:true})]));
// vibe-city /destructible-stress presets, generated into ../generated/vibe-city (optional).
const VIBE_CITY='../generated/vibe-city/';
try{const index=await (await fetch(`${VIBE_CITY}index.json`)).json();
 const group=document.createElement('optgroup');group.label='Vibe City presets (generated scene packs)';
 for(const p of index.presets.filter(p=>!p.failed&&!p.pending)){
  const key=`vc:${p.id}`;samples[key]={key,label:p.id,pack:`${VIBE_CITY}${p.id}.json`,nodes:p.nodes,bonds:p.bonds};
  group.append(new Option(`${p.id} · ${p.nodes} nodes`,key));
 }
 if(group.children.length)$('scene').insertBefore(group,$('scene').lastElementChild);
}catch{}
const sample=(key=$('scene').value||'house')=>serializeLoad(async()=>{
 const entry=samples[key];if(!entry)throw new Error('Unknown bundled scene');
 if(entry.pack){
  // A generated scene pack (engine/scenes/vibe-city, bun run export.ts): one level, bonds from the pack.
  setStatus(`Fetching ${entry.label}…`);
  const response=await fetch(entry.pack);
  if(!response.ok)throw new Error('Scene pack missing: run bun run export.ts in engine/scenes/vibe-city.');
  packBytes=await response.arrayBuffer();assetBytes=null;layout=null;procedural='pack';asset?.handle.free();asset=null;$('scene').value=key;$('filename').textContent=entry.pack.split('/').pop();
  $('level').replaceChildren(new Option(`Scene pack · ${entry.nodes} nodes · ${entry.bonds} bonds`,0));$('level').value='0';
  await loadLevel(0);return;
 }
 if(entry.procedural){
  // Built in code from box colliders: no GLB, one level, no baked bonds.
  layout=entry.layout??null;procedural=entry.procedural;assetBytes=null;asset?.handle.free();asset=null;$('scene').value=key;$('filename').textContent='Procedural cuboid house';
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
$('ball-mass').oninput=applyBallMass;
$('bond-strength').oninput=()=>{$('bond-strength-value').textContent=scaleLabel(strengthScale())};
$('bond-strength').onchange=()=>{if(bondMode()!=='none')serializeLoad(()=>loadLevel(Number($('level').value)))};
$('bond-mode').onchange=()=>serializeLoad(()=>loadLevel(Number($('level').value)));
$('release').onclick=()=>{if(stressOn&&stress){stress.running=!stress.running;$('release').textContent=stress.running?'Pause':'Resume';return}released=true;running=!running;$('release').textContent=running?'Pause':'Resume';accumulator=0;rates.reset()};
$('reset').onclick=()=>serializeLoad(async()=>{if(stressOn){await loadStress(Number($('level').value));return}sim.reset();released=false;$('release').textContent='Start simulation';setStatus('Reset · click the model to shoot')});
$('metadata').onclick=()=>{if(!asset)return;const url=URL.createObjectURL(new Blob([asset.metadataJson()],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='fracture-metadata.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
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
// One stress-mode frame: the Rust steps what fell due and draws; the HUD counts physics
// frames and rendered frames per second from its stats.
function stressTick(now,elapsedMs){
 frame=(async()=>{try{
  const before=stress.stats.frames??0;await stress.frame(elapsedMs);const st=stress.stats;
  stressHud.frames+=(st.frames??0)-before;stressHud.renders++;
  if(now-stressHud.t>=1000){const s=(now-stressHud.t)/1000;$('tps').textContent=(stressHud.frames/s).toFixed(0);$('fps').textContent=(stressHud.renders/s).toFixed(0);$('sim-speed').textContent=`${(stressHud.frames/s/st.frameHz).toFixed(2)}×`;stressHud={t:now,frames:0,renders:0}}
  $('step-ms').textContent=st.stepMs?`${st.stepMs.toFixed(1)} ms/frame (stress + rubble)`:`target ${Math.round(st.frameHz)} frames/s`;
  $('frame-ms').textContent='frames / second';
  $('awake').textContent=st.moving;
  if(stress.running||st.shots){$('release').textContent=stress.running?'Pause':'Resume';setStatus(`${st.shots}/16 shots · ${st.frames} frames · ${st.fragments} fragments · ${st.brokenBonds} bonds broken · ${st.simSeconds.toFixed(1)} s`)}
 }catch(e){fail(e)}})();
 frame.finally(()=>requestAnimationFrame(tick));
}
function tick(now){
 if(stressOn&&!loading&&stress){const ms=last?Math.min(40,now-last):0;last=now;return stressTick(now,ms)}
 const frameMs=last?now-last:FRAME_BUDGET_MS;
 const elapsed=last?Math.min(.04,(now-last)/1000):0;last=now;
 // Damped control on a smoothed frame time: cut quickly when over budget,
 // raise slowly when under, so one heavy frame does not swing the rate.
 frameEma+=0.2*(frameMs-frameEma);
 if(!sim?.slicing)sliceRate=Math.min(MAX_SLICE_RATE,Math.max(0.25,sliceRate*Math.min(1.03,Math.max(0.7,FRAME_BUDGET_MS/frameEma))));
 if(!loading&&renderer){frame=(async()=>{try{
  let work=null,ticks=0;
  if(running){
   accumulator=Math.min(accumulator+elapsed,4/stepHz());
   if(sim.slicing){
    // A heavy step in flight: this frame's share of slices, then its counters.
    // Additive increase while the GPU keeps up, multiplicative cut when it
    // does not, with the frame-time cut above as a backstop.
    sliceRate=workDone[0].done&&frameEma<FRAME_BUDGET_MS?Math.min(MAX_SLICE_RATE,sliceRate+0.1):Math.max(0.25,sliceRate*0.85);
    sliceCredit+=sliceRate;const n=Math.floor(sliceCredit);sliceCredit-=n;
    const mark={done:false};workDone=[workDone[1],mark];device.queue.onSubmittedWorkDone().then(()=>{mark.done=true});
    // Counters arrive asynchronously, so the queue keeps flowing.
    if(sim.submitSlices(n))sim.finishSlicedStep().then(()=>{if(!document.hidden)rates.step(sim.timing)}).catch(fail);
   }else{
    // Bound outstanding work. Slow ticks get one tick per frame, rather than
    // four serial GPU waits starving camera input and rendering.
    // Overlapped ticks (opt-in) queue without waiting for counters, so the
    // frame can carry the whole backlog (the simulation caps ticks in flight).
    const overlap=$('overlap-ticks').checked;
    const budget=overlap?4:Math.max(1,Math.min(4,Math.floor(12/Math.max(sim.timing.wallMs,0.1))));
    ticks=Math.min(budget,Math.floor(accumulator*stepHz()));
    if(ticks){
     accumulator-=ticks/stepHz();
     // Steps over ~12 ms run sliced across frames so rendering keeps its rate;
     // the physics is the same, only spread over several frames.
     const heavy=sim.timing.wallMs>SLICED_STEP_MS&&!(sim.timing.sliced&&sim.counters.awake<SLICE_EXIT_AWAKE);
     if(heavy){ticks=0;if(sim.beginSlicedStep()&&sim.submitSlices(1))sim.finishSlicedStep().then(()=>{if(!document.hidden)rates.step(sim.timing)}).catch(fail)}
     else work=sim.step(ticks,{wait:!overlap});
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
