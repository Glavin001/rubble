import {parseFractureGlb,buildScene,buildCuboidHouse,buildPackScene,layoutScene,withoutJoints,withBondStrength,withProjectileRadius,withParams,SAMPLES,PACK_SAMPLES,packSampleJson} from './wasm.mjs';
import {createSimulation} from './simulation.mjs';
import {preloadBonds} from './bonds.mjs';
import {createRenderer} from './render.mjs';
import {createRates} from './rates.mjs';
import {createDirector} from './director.mjs';
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
// joints, AVBD welds), 'stress' (GPU stress solver on its own canvas), 'threshold' (the
// same, with the v0 rule: a bond breaks on contact force alone, no solve), or 'learned'
// (lean MGN-T forward on this page's WebGPU — compute/timing only; rubble stays unbonded).
const bondMode=()=>$('bond-mode').value;
// The controls panel collapses for a full view (remembered per viewer; phones start
// collapsed). Collapsed, a quick button mirrors Start / Pause.
const PANEL_KEY='fracture-lab.panel';
function setPanel(open){
 document.body.classList.toggle('collapsed',!open);const t=$('panel-toggle');
 t.setAttribute('aria-expanded',String(open));t.textContent=open?'✕':'☰ Controls';t.title=open?'Hide controls (full view)':'Show controls';
 try{localStorage.setItem(PANEL_KEY,open?'1':'0')}catch{}
}
{let stored=null;try{stored=localStorage.getItem(PANEL_KEY)}catch{}
 setPanel(stored===null?!matchMedia('(max-width: 720px)').matches:stored==='1')}
$('panel-toggle').onclick=()=>setPanel(document.body.classList.contains('collapsed'));
{const quick=$('quick-release'),release=$('release'),sync=()=>{quick.textContent=release.textContent;quick.disabled=release.disabled};
 new MutationObserver(sync).observe(release,{childList:true,characterData:true,subtree:true,attributes:true,attributeFilter:['disabled']});sync();quick.onclick=()=>release.click()}

// The stress solver's modes, and the capability profile each runs (stress-web).
const STRESS_PROFILES={stress:'blast_parity',threshold:'threshold'};
const stressMode=()=>bondMode() in STRESS_PROFILES;
// The stress-solver modes wait on the GPU from Rust: through WebAssembly JavaScript Promise
// Integration where the browser has it, else (Safari) through the Asyncify build
// (gpu-wait.mjs, stress-mode.mjs).
let stress=null,stressOn=false,stressHud={t:0,frames:0,renders:0};
let learned=null,learnedMs=0,learnedBusy=false;
// The bond graph and the View menu (0 materials, 1 sleep state, 2 bodies): this page's
// renderer reads them as it draws; the stress modes draw with their own (Rust, stress-web),
// told here.
const viewMode=()=>Number($('view-mode').value);
function applyView(){if(stressOn){stress?.setView($('show-bonds').checked,viewMode());stress?.setFx($('fx').checked)}}
// The Camera menu (engine-fracture view.rs ViewMode: 0 orbit, 1 first-person walk, 2 fly),
// remembered per viewer and kept across loads. V cycles it.
const CAMERA_KEY='fracture-lab.camera';
const CAMERA_HINTS=['CLICK TO SHOOT · DRAG TO ORBIT · SCROLL TO ZOOM · V CAMERA',
 'CLICK: LOCK MOUSE, THEN SHOOT · WASD WALK · SHIFT SPRINT · SCROLL SPEED · ESC FREES MOUSE · V CAMERA',
 'CLICK: LOCK MOUSE, THEN SHOOT · WASD FLY · SPACE/E UP · C/Q DOWN · SHIFT SPRINT · SCROLL SPEED · ESC FREES MOUSE · V CAMERA'];
const cameraMode=()=>Number($('camera-mode').value);
function applyCamera(){
 const mode=cameraMode();if(renderer)renderer.viewMode=mode;if(stress)stress.viewMode=mode;
 document.body.classList.toggle('first-person',mode!==0);$('hint').textContent=CAMERA_HINTS[mode];
 try{localStorage.setItem(CAMERA_KEY,String(mode))}catch{}
}
{let stored=null;try{stored=localStorage.getItem(CAMERA_KEY)}catch{}
 if(stored&&$('camera-mode').querySelector(`option[value="${stored}"]`))$('camera-mode').value=stored}
$('camera-mode').onchange=()=>{applyCamera();$('camera-mode').blur()};
addEventListener('keydown',e=>{
 if(e.code!=='KeyV'||e.repeat||e.ctrlKey||e.metaKey||e.altKey||/^(INPUT|SELECT|TEXTAREA)$/.test(e.target?.tagName))return;
 const options=$('camera-mode').options;$('camera-mode').value=options[(cameraMode()+1)%options.length].value;applyCamera()});
applyCamera();
// Ball mass: a power-of-two multiple of the scene's default ball, applied to the next shots.
const massScale=()=>2**Number($('ball-mass').value);
// Ball speed: a power-of-two multiple of the scene's launch speed, applied to the next shots.
const speedScale=()=>2**Number($('ball-speed').value);
// Ball radius: a power-of-two multiple of the scene's ball (the mass stays), applied when
// the level loads.
const radiusScale=()=>2**Number($('ball-radius').value);
// Bond strength: a power-of-two multiple of every bond's break strength (D6 joints and
// the stress solver), applied when the level loads.
const strengthScale=()=>2**Number($('bond-strength').value);
// Physics rate: 0 (Auto) keeps the scene's dt, else 60 or 120 ticks per second of
// simulated time, applied when the level loads.
const physicsHz=()=>Number($('physics-hz').value);
const scaleLabel=k=>`×${k>=1?k:`1/${1/k}`}`;
function applyBallSpeed(){
 const k=speedScale(),ms=(stressOn?stress?.stats.ballSpeed:sim?.shotSpeed)*k;
 $('ball-speed-value').textContent=`${scaleLabel(k)}${ms?` · ${ms.toFixed(ms<10?1:0)} m/s`:''}`;
 if(stressOn)stress?.setShotSpeedScale(k);else sim?.setProjectileSpeedScale(k);
}
function showBallRadius(){
 const r=stressOn?stress?.stats.ballRadius:scene?.projectileRadius;
 $('ball-radius-value').textContent=`${scaleLabel(radiusScale())}${r?` · ${r.toFixed(2)} m`:''}`;
}
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
 $('step-ms').textContent=learned?`learned ${learnedMs.toFixed(2)} ms/fwd · ${learned.graph.nNodes} n · ${learned.params} w`:
  value&&!value.stepMs&&sim?.counters.awake===0?'all asleep · no GPU work':
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
const controls=enabled=>{enabled||=stressOn&&!loading;for(const id of ['release','reset','level','metadata','fire','detonate','launch'])$(id).disabled=!enabled;for(const id of ['file','sample','scene'])$(id).disabled=!device||loading};
function serializeLoad(task){const result=loadTail.catch(()=>{}).then(async()=>{loading=true;running=false;controls(false);await frame;rates.reset();try{await task()}catch(e){fail(e)}finally{loading=false;last=0;rates.reset();controls(!!sim)}});loadTail=result;return result}
let shotQueue=Promise.resolve();
function shoot(ray){
 if(loading||!sim)return;
 shotQueue=shotQueue.then(async()=>{await frame;if(loading||!sim)return;try{sim.launch(ray.origin,ray.direction);director.logShot(ray.origin,ray.direction);if(!running)rates.reset();released=true;running=true;accumulator=0;$('release').textContent='Pause'}catch(e){fail(e)}});
 return shotQueue;
}
async function loadLevel(level){
 let next;try{
  // Settle one house under its own weight first, so every copy starts asleep in
  // static equilibrium (bonds carrying their load) rather than at zero force.
  // Breakable bonds are experimental (opt-in): see README "Breakable bonds".
  if(stressMode())return loadStress(level);
  stressOn=false;$('view').hidden=false;$('stress-view').hidden=true;$('ball-preset-row').hidden=true;$('detonate').hidden=true;$('launch').hidden=true;$('fire').textContent='Fire the wrecking ball through the row';if(stress)stress.running=false;
  learned?.dispose();learned=null;learnedMs=0;
  const raw=procedural==='pack'?buildPackScene(packBytes):procedural?buildCuboidHouse():buildScene(asset,level,{maxHullVertices});
  const built=physicsHz()?withParams(raw,{dt:1/physicsHz()}):raw;
  if(bondMode()==='d6')setStatus('Settling bonds under self-weight…');
  // Learned mode: rubble stays unbonded (same as none); the MGN-T forward is timed separately.
  const one=bondMode()==='d6'?await preloadBonds(device,strengthScale()===1?built:withBondStrength(built,strengthScale())):withoutJoints(built);
  // Copies laid out as the sample says (procedural houses get a larger contact budget).
  const laid=layoutScene(one,layout,!!procedural);
  next=radiusScale()===1?laid:withProjectileRadius(laid,radiusScale())}catch(e){if(scene)$('level').value=String(scene.level??0);throw e}
 renderer?.dispose();renderer=null;sim?.dispose();sim=null;released=false;accumulator=0;
 scene=next;director.reset();sim=await createSimulation(device,scene,setStatus);renderer=await createRenderer(device,$('view'),sim,scene,shoot);
 renderer.colorMode=viewMode();applyCamera();applyBallMass();applyBallSpeed();showBallRadius();
 if(bondMode()==='learned'){
  setStatus('Loading learned-impact kernels (Slang→WGSL)…');
  const {createLearnedImpact}=await import('./learned-impact.mjs');
  learned=await createLearnedImpact(device,{status:setStatus});
  learnedMs=await learned.run();
 }
 const copies=scene.copies||1;$('fragments').textContent=scene.selected.length*copies;$('hulls').textContent=procedural?scene.pieces*copies:scene.hullCount;$('bonds').textContent=scene.bondCount*copies;$('release').textContent='Start simulation';
 $('fire').hidden=!scene.rowShot;$('awake').textContent='0';$('contacts').textContent='0';
 const skipped=scene.skippedHulls.length?` · ${scene.skippedHulls.length} flat zero-volume hulls skipped${copies>1?' per house':''}`:'';
 const what=copies>1?`${copies} houses · ${scene.bodyCount-1-scene.projectileCount} bodies · `:'';
 const source=procedural==='pack'?`scene pack · ${scene.pieces} nodes`:procedural?'procedural cuboid houses (box colliders)':`${(asset.bytes/1048576).toFixed(2)} MB · ${asset.materialCount} materials${skipped}`;
 const learnedNote=learned?` · learned-impact ${learnedMs.toFixed(2)} ms/fwd (${learned.graph.nNodes} chunks, ${learned.params} weights, compute only)`:'';
 setStatus(`${what}${source}${learnedNote} · ${scene.rowShot?'fire the ball through the row or click to shoot':'click the model to shoot'}`);
}
// The stress-solver mode, built and run by stress-web: a GLB's level (one copy of the
// structure; the sample's layout of copies is the other modes'), the cuboid houses in
// their sample's layout (every copy), or a scene pack.
async function loadStress(level){
 // A fracture GLB's level (one copy of the structure), the cuboid houses in their
 // sample's layout (every copy), or a scene pack.
 const cuboid=procedural==='cuboid',pack=procedural==='pack';
 if(!cuboid&&!pack&&(procedural||!assetBytes))throw new Error('The stress-solver mode runs a fracture GLB, a scene pack or the cuboid houses: pick one of those scenes.');
 learned?.dispose();learned=null;learnedMs=0;
 renderer?.dispose();renderer=null;sim?.dispose();sim=null;released=false;
 $('view').hidden=true;$('stress-view').hidden=false;
 setStatus('Loading the stress solver (Rust and Slang on WebGPU)…');
 // The mode is on only once its module loaded: a failed import (an old build)
 // leaves the page in its own renderer with the error shown.
 try{if(!stress){const {createStressMode}=await import('./stress-mode.mjs');stress=await createStressMode($('stress-view'),(o,d)=>director.logShot(o,d))}}
 catch(e){$('view').hidden=false;$('stress-view').hidden=true;throw new Error(`Stress-solver mode unavailable: ${e?.message??e}. Rebuild it (README, Bond modes).`)}
 stressOn=true;director.reset();
 // The ball presets (stress-web, stress_ref::wrecking::BALLS), listed once.
 if($('ball-preset').options.length===1)$('ball-preset').append(...stress.shotPresets.map((b,i)=>new Option(b.menu,i)));
 $('ball-preset-row').hidden=false;stress.setShotPreset(Number($('ball-preset').value));
 const profile=STRESS_PROFILES[bondMode()];
 stress.viewMode=cameraMode();stress.setProfile(profile);stress.setPhysicsHz(physicsHz());stress.setBondStrengthScale(strengthScale());stress.setShotRadiusScale(radiusScale());applyView();
 if(cuboid)await stress.loadCuboid(layout);else if(pack)await stress.loadPack(packBytes);else await stress.load(assetBytes,level);
 await stress.frame(0);applyBallMass();applyBallSpeed();showBallRadius();
 const st=stress.stats;stressHud={t:performance.now(),frames:0,renders:0};
 $('fragments').textContent=st.chunks;$('bonds').textContent=st.bonds;$('hulls').textContent=st.hulls;$('awake').textContent='0';$('contacts').textContent='—';
 $('release').textContent='Start simulation';$('fire').hidden=!st.canFire;$('fire').textContent='Fire the ball along the line of attack';
 // A scene with demolition charges (the pack's defaults.charges: Topple) gets the detonator.
 $('detonate').hidden=!st.charges?.length;
 // A scene with launched bonded bodies (the pack's defaults.launches: a car, an aircraft) gets the launcher.
 $('launch').hidden=!st.launches?.length;
 setStatus(`Stress solver · ${st.chunks} chunks · ${st.bonds} bonds · ${profile==='threshold'?'simple threshold (v0): a bond breaks when a contact force beats its strength':profile}, resident GPU frame at ${Math.round(st.frameHz)} Hz${cuboid||pack?'':' · one copy'}${stressWarning(st)} · click the model to shoot`);
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
// Scene packs built in Rust (engine-fracture samples.rs PACK_SAMPLES: the Wrecking Yard), loaded as any pack.
for(const s of PACK_SAMPLES)samples[s.key]={...s,pack:true,builtin:true};
const packGroups=[...new Set(PACK_SAMPLES.map(s=>s.group))].map(label=>optgroup(label,PACK_SAMPLES.filter(s=>s.group===label).map(s=>new Option(s.menu,s.key))));
$('scene').replaceChildren(...sceneGroups,...packGroups,optgroup('Your file',[Object.assign(new Option('Custom GLB (choose or drop one below)',''),{disabled:true})]));
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
  // A scene pack, built in Rust (the Wrecking Yard) or generated (engine/scenes/vibe-city, bun run export.ts): one level, bonds from the pack.
  if(entry.builtin)packBytes=packSampleJson(key).buffer;
  else{setStatus(`Fetching ${entry.label}…`);
   const response=await fetch(entry.pack);
   if(!response.ok)throw new Error('Scene pack missing: run bun run export.ts in engine/scenes/vibe-city.');
   packBytes=await response.arrayBuffer()}
  assetBytes=null;layout=null;procedural='pack';asset?.handle.free();asset=null;$('scene').value=key;$('filename').textContent=entry.builtin?`${entry.label} (built in Rust)`:entry.pack.split('/').pop();
  $('level').replaceChildren(new Option(entry.builtin?entry.menu:`Scene pack · ${entry.nodes} nodes · ${entry.bonds} bonds`,0));$('level').value='0';
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
$('fire').onclick=()=>{if(stressOn){if(stress?.fire())director.logFire();return}scene?.rowShot&&shoot(scene.rowShot)};
// Every charge, each at its delay (stress-web stress_detonate: cut between frames); logged
// so a replay sets them off at the same frame.
$('detonate').onclick=()=>{if(stressOn&&stress?.detonate(''))director.logDetonate('')};
// Every launched bonded body, each at its delay (stress-web stress_launch: set moving between
// frames); logged so a replay releases them at the same frame.
$('launch').onclick=()=>{if(stressOn&&stress?.launch(''))director.logLaunch('')};
$('ball-preset').onchange=()=>serializeLoad(()=>loadLevel(Number($('level').value)));
$('scene').onchange=()=>sample($('scene').value);
$('file').onchange=()=>{const f=$('file').files[0];if(f)serializeLoad(async()=>importBytes(await f.arrayBuffer(),f.name))};
$('level').onchange=()=>serializeLoad(()=>loadLevel(Number($('level').value)));
$('view-mode').onchange=()=>{if(renderer)renderer.colorMode=viewMode();applyView()};
$('show-bonds').onchange=applyView;
$('fx').onchange=applyView;
$('ball-mass').oninput=applyBallMass;
$('ball-speed').oninput=applyBallSpeed;
$('ball-radius').oninput=()=>{$('ball-radius-value').textContent=scaleLabel(radiusScale())};
$('ball-radius').onchange=()=>serializeLoad(()=>loadLevel(Number($('level').value)));
$('bond-strength').oninput=()=>{$('bond-strength-value').textContent=scaleLabel(strengthScale())};
$('bond-strength').onchange=()=>{if(bondMode()!=='none')serializeLoad(()=>loadLevel(Number($('level').value)))};
$('bond-mode').onchange=()=>serializeLoad(()=>loadLevel(Number($('level').value)));
$('physics-hz').onchange=()=>serializeLoad(()=>loadLevel(Number($('level').value)));
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
// What the stress import could not take from the scene as given (stress-web `warnings`).
const stressWarning=st=>st.warnings?.length?` · ⚠ ${st.warnings.join('; ')}`:'';
function stressTick(now,elapsedMs){
 frame=(async()=>{try{
  const before=stress.stats.frames??0;await stress.frame(elapsedMs);const st=stress.stats;
  stressHud.frames+=(st.frames??0)-before;stressHud.renders++;
  if(now-stressHud.t>=1000){const s=(now-stressHud.t)/1000;$('tps').textContent=(stressHud.frames/s).toFixed(0);$('fps').textContent=(stressHud.renders/s).toFixed(0);$('sim-speed').textContent=`${(stressHud.frames/s/st.frameHz).toFixed(2)}×`;stressHud={t:now,frames:0,renders:0,stepMs:stressHud.stepMs}}
  // A display frame that stepped no physics frame (a 120 Hz display, a 60 Hz simulation)
  // reports no step time: the last measured one stays, so the label does not flip.
  if(st.stepMs)stressHud.stepMs=st.stepMs;
  $('step-ms').textContent=stressHud.stepMs?`${stressHud.stepMs.toFixed(1)} ms/frame (stress + rubble)`:`target ${Math.round(st.frameHz)} frames/s`;
  $('frame-ms').textContent='frames / second';
  $('awake').textContent=st.moving??st.awake??'—';
  if(stress.running||st.shots){$('release').textContent=stress.running?'Pause':'Resume';if(performance.now()>=noticeUntil)setStatus(`${st.shots} shots · ${st.frames} frames · ${st.fragments} fragments · ${st.bodies??'—'} bodies · ${st.brokenBonds} bonds broken · ${st.simSeconds.toFixed(1)} s${stressWarning(st)}`)}
 }catch(e){fail(e)}})();
 frame.finally(()=>requestAnimationFrame(tick));
}
function tick(now){
 // The director (a take or a replay) steps and draws on its own meanwhile.
 if(held){last=0;requestAnimationFrame(tick);return}
 if(stressOn&&!loading&&stress){const ms=last?Math.min(250,now-last):0;last=now;return stressTick(now,ms*timeScale())}
 const frameMs=last?now-last:FRAME_BUDGET_MS;
 const elapsed=(last?Math.min(.04,(now-last)/1000):0)*timeScale();last=now;
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
  // Learned-impact: time one forward per frame when idle enough (discard output).
  if(learned&&!learnedBusy){learnedBusy=true;learned.run().then(ms=>{learnedMs=ms;learnedBusy=false;$('step-ms').textContent=`learned ${ms.toFixed(2)} ms/fwd · ${learned.graph.nNodes} n · ${learned.params} w`}).catch(e=>{learnedBusy=false;fail(e)})}
  // Queue the draw immediately behind physics; do not wait for CPU stats first.
  renderer.draw($('show-bonds').checked);
  if(work){await work;if(!document.hidden)for(let i=0;i<ticks;i++)rates.step(sim.timing)}
  if(running){const c=sim.counters;$("awake").textContent=c.awake;$("contacts").textContent=c.contact}
  if(running&&performance.now()>=noticeUntil){const c=sim.counters;setStatus(`${sim.shots} shots · ${sim.steps} steps · ${c.contact} contacts · ${c.awake} awake · ${(sim.steps/stepHz()).toFixed(1)} s`)}
  if(!document.hidden)rates.frame();
 }catch(e){fail(e)}})();frame.finally(()=>requestAnimationFrame(tick))}else requestAnimationFrame(tick);
}
// Takes and instant replay (director.mjs). While it holds the page, the tick loop idles.
let held=false;
// A replay's outcome stays on the status line this long before the live counters return.
let noticeUntil=0;
// Live slow motion (the Speed menu): simulated time per wall second; the step size is
// unchanged, fewer steps run per frame.
const timeScale=()=>Number($('time-scale').value);
const shotSettings=()=>({mass:Number($('ball-mass').value),speed:Number($('ball-speed').value)});
function hashFloats(f){let h=0x811c9dc5;const u=new Uint32Array(f.buffer,f.byteOffset,f.length);for(let i=0;i<u.length;i++){h^=u[i];h=Math.imul(h,0x01000193)}return(h>>>0).toString(16)}
const director=createDirector({
 get stressOn(){return stressOn},get stress(){return stress},get sim(){return sim},get renderer(){return renderer},get scene(){return scene},
 canvas:()=>$(stressOn?'stress-view':'view'),stepHz,showBonds:()=>$('show-bonds').checked,
 async pause(){held=true;await frame;await shotQueue;await frame},
 resume(){held=false;last=0;accumulator=0;rates.reset();
  if(stressOn){if(stress){stress.running=true;$('release').textContent='Pause'}}else if(sim){released=running=true;$('release').textContent='Pause'}},
 reload:()=>serializeLoad(()=>loadLevel(Number($('level').value))),
 aim:(u,v)=>stressOn?stress.aim(u,v):renderer.aimAt(u,v),
 setView({bonds,color,fx}={}){if(bonds!==undefined)$('show-bonds').checked=!!bonds;if(fx!==undefined)$('fx').checked=!!fx;if(color!==undefined){$('view-mode').value=String(color);if(renderer)renderer.colorMode=viewMode()}applyView()},
 shotSettings,applyShotSettings(s){$('ball-mass').value=String(s.mass);$('ball-speed').value=String(s.speed);applyBallMass();applyBallSpeed()},
 stats(){if(stressOn){const st=stress.stats;
  // The GPU-resident world's counters arrive a frame or two late: none yet reads '—', not 0.
  const known=v=>st.gpuResident&&!st.statsFrame?'—':(v??'—');
  return `${known(st.fragments)} fragments · ${st.brokenBonds??0} bonds broken · ${known(st.bodies)} bodies · ${(st.simSeconds??0).toFixed(2)} s`}
  const c=sim.counters;return `${sim.shots} shots · ${c.awake} awake · ${c.contact} contacts · ${(sim.steps/stepHz()).toFixed(2)} s`},
 // A recorded take's frame: the view over the whole window (the panel hidden, as collapsed)
 // or back as it was. Returns whether it was full before.
 async fullFrame(on){const was=document.body.classList.contains('collapsed');document.body.classList.toggle('collapsed',on);
  await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));return was},
 // The counters a recorded take logs per video frame (the take check: they never go back).
 counters(){if(stressOn){const st=stress.stats;return {frames:st.frames??0,statsFrame:st.statsFrame??st.frames??0,simSeconds:st.simSeconds??0,brokenBonds:st.brokenBonds??0,fragments:st.fragments??0,bodies:st.bodies??0,chunks:st.chunks??0,lastShot:st.lastShot??null}}
  return {frames:sim.steps,simSeconds:sim.steps/stepHz(),shots:sim.shots}},
 // What a replay must reproduce: the stress modes' counters, or every body's pose and velocity.
 async summary(){if(stressOn){const st=stress.stats;return {frames:st.frames,fragments:st.fragments,brokenBonds:st.brokenBonds,bodies:st.bodies}}
  return {steps:sim.steps,shots:sim.shots,bodies:hashFloats(await sim.snapshot())}}});
async function replay(){
 if(director.busy||!director.shots)return;
 $('replay').disabled=true;const was=status.textContent;
 try{setStatus('Replay: re-running the scene from its load…');
  const r=await director.replay({onProgress:p=>{status.textContent=`Replay ${Math.round(p*100)}%`}});
  noticeUntil=performance.now()+6000;
  if(r)setStatus(r.same?'Replay done: it reached the same state, carrying on live':`Replay done, but it ended in a different state (${JSON.stringify(r.after)} vs ${JSON.stringify(r.before)}): this mode did not repeat the run`);
  else status.textContent=was}
 catch(e){fail(e)}finally{$('replay').disabled=false}
}
$('replay').onclick=replay;
addEventListener('keydown',e=>{if(e.code==='KeyR'&&!e.repeat&&!e.ctrlKey&&!e.metaKey&&!e.altKey&&!/^(INPUT|SELECT|TEXTAREA)$/.test(e.target?.tagName))replay()});
window.__record=(script,opts)=>director.record(script,opts);
window.__replay=opts=>director.replay(opts);
try{
 if(!navigator.gpu)throw new Error('WebGPU is unavailable. Open this localhost page in a WebGPU-enabled browser.');
 const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw new Error('No WebGPU adapter available');
 device=await adapter.requestDevice({requiredFeatures:adapter.features.has('timestamp-query')?['timestamp-query']:[],requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(16,adapter.limits.maxStorageBuffersPerShaderStage),maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize,maxBufferSize:adapter.limits.maxBufferSize}});
 device.addEventListener('uncapturederror',e=>fail(e.error));device.lost.then(info=>fail(new Error(`GPU device lost: ${info.message}. Reload this page.`)));
 await sample();requestAnimationFrame(tick);
}catch(e){fail(e);controls(false)}
