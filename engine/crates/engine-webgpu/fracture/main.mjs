import {parseFractureGlb,buildScene,buildCuboidHouse,buildPackScene,layoutScene,withoutJoints,withBondStrength,withProjectileRadius,withParams,buildTowersScene,towersVolley,withCity,meteorShot,toppleShots,nearMissSettings,sceneHandle,glassTowerPack,SAMPLES} from './wasm.mjs';
import {createLabWorld,settleBonds,labDevice,bondAsset,thresholdAsset} from './lab-world.mjs';
import {createRenderer} from './render.mjs';
import {createRates} from './rates.mjs';
const $=id=>document.getElementById(id),status=$('status');
// The physics is rubble's World (lab-world.mjs): `sim` steps it, `renderer` draws its buffers.
let device,asset,scene,sim,renderer,running=false,released=false,loading=true,frame=Promise.resolve(),loadTail=Promise.resolve(),last=0,accumulator=0;
// Physics ticks one frame may take (World steps whole and never waits: a tick with no free
// readback slot stays due for the next frame).
const MAX_TICKS_PER_FRAME=4;
let currentRates=null,layout=null,maxHullVertices,procedural=null,assetBytes=null,packBytes=null;
// The towers showcase (engine-fracture towers.rs): a block of a pack's towers,
// its own ball, bonds at 0.2× the pack's strength unless Bonds is None.
const TOWERS={columns:2,rows:2,gap:6,radius:1,mass:30000,strength:0.2};
let towersBlock=false;
// The meteor variant of the block: a 5 m, 30,000 t ball (engine-fracture towers.rs METEOR).
let towersMeteor=false;const METEOR={radius:5,mass:3e7};
// The scripted volley in flight: shots with absolute steps, the next to fire.
let volley=null;
// Ball trails: per projectile slot, its path and whether it was a CCD shot;
// positions come from a small async readback of the ball bodies each frame.
let trails=[],trailRead=null;
function resetTrails(){trails=scene?Array.from({length:scene.projectileCount},()=>({path:[],ccd:false})):[];renderer?.setTrails(new Float32Array(0))}
function trailLaunched(id){const t=trails[id-scene.projectileStart];if(t){t.path=[];t.ccd=thresholdOn||$('ball-ccd').checked&&!stressOn}}
function updateTrails(){
 if(!sim||!renderer||trailRead||!sim.shots)return;
 const n=Math.min(sim.shots,scene.projectileCount);
 const r=renderer;trailRead=sim.shotRecords().then(f=>{
  if(r!==renderer)return;
  const out=[];
  for(let k=0;k<n;k++){
   const t=trails[k];if(!t)continue;const p=[f[k*28],f[k*28+1],f[k*28+2]];
   if(p[1]>-100){const q=t.path[t.path.length-1];if(!q||Math.hypot(p[0]-q[0],p[1]-q[1],p[2]-q[2])>0.01){t.path.push(p);if(t.path.length>600)t.path.shift()}}
   const m=t.path.length,c=t.ccd?[0.2,0.95,1]:[1,0.25,0.2];
   for(let i=1;i<m;i++){const a=0.25+0.75*i/m;for(const v of [t.path[i-1],t.path[i]])out.push(v[0],v[1],v[2],0,c[0],c[1],c[2],a)}
  }
  renderer.setTrails(new Float32Array(out));
 }).catch(()=>{}).finally(()=>{trailRead=null});
}
// Bond mode (the Bonds menu): 'none' (pieces sleep until hit), 'd6' (breakable D6
// joints, AVBD welds), 'threshold' (the v0 stress rule: a bond breaks on contact force
// alone, no solve), all three on rubble's World (lab-world.mjs: StrengthModel::Bonds and
// StrengthModel::Threshold) and this page's renderer; 'stress' (GPU stress solver on its own
// canvas), 'load_path' and 'impact_local' (the same GPU world with a load path, or with the
// analytic local impact model for strikes: stress-web, which World has no equivalent of),
// 'blast' (the same GPU world, Blast's stress solver deciding the bonds: profile blast_gpu), or
// 'learned' (lean MGN-T forward on this page's WebGPU — compute/timing only; rubble stays
// unbonded).
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

// Directed debris, "Near Miss" (engine-fracture guidance.rs): each shot finds a
// crowd of bystanders downrange (orange rings). On, debris that can reach one
// lands around them rather than on open ground; off, it flies as it would. The
// rings and the count of pieces resting near a bystander show either way, so
// the two can be compared shot for shot. Each shot re-arms the pieces.
let bystanders=[],nearAt=0,nearBusy=false;
function applyGuide(){
 if(!sim||!scene||stressOn||thresholdOn)return;
 sim.setGuidance($('guide').checked?nearMissSettings($('guide-strength').value):null,bystanders);
 renderer?.setZones(sceneHandle(scene).zoneOutline(bystanders));
}
function placeCrowd(ray){
 // The city block's zones sit on its cars and pedestrians.
 if(!scene||stressOn||thresholdOn||towersBlock)return;
 bystanders=sceneHandle(scene).nearMissZonesFor(Float64Array.from(ray.origin),Float64Array.from(ray.direction));
 applyGuide();$('near').textContent='0';
}
function updateNear(now){
 if(nearBusy||!sim||!bystanders.length||now-nearAt<500)return;nearAt=now;nearBusy=true;
 sim.snapshot().then(s=>{if(sim&&scene)$('near').textContent=sceneHandle(scene).nearCounts(s,bystanders).reduce((a,b)=>a+b,0)}).catch(()=>{}).finally(()=>{nearBusy=false});
}
for(const id of ['guide','guide-strength'])$(id).onchange=()=>{applyGuide();$(id).blur()};
// The stress solver's modes on stress-web, and the capability profile each runs. The simple
// threshold runs on World (thresholdMode).
const STRESS_PROFILES={stress:'blast_parity',load_path:'load_path',impact_local:'impact_local',blast:'blast_gpu'};
const thresholdMode=()=>bondMode()==='threshold';
const PROFILE_NOTES={threshold:'simple threshold (v0): a bond breaks when a contact force beats its strength',load_path:'load path: gravity and contacts carried through the bonds, each checked for tension, shear, compression and bending',impact_local:'local impact: a shot holes, spalls or craters the wall by its ballistic limit, before the engine steps',blast_gpu:"Blast's stress solver (vibe-land's model, Slang)"};
const stressMode=()=>bondMode() in STRESS_PROFILES;
// The stress-solver modes wait on the GPU from Rust: through WebAssembly JavaScript Promise
// Integration where the browser has it, else (Safari) through the Asyncify build
// (gpu-wait.mjs, stress-mode.mjs).
let stress=null,stressOn=false,stressHud={t:0,frames:0,renders:0,gaps:[],cpuMs:0,last:0};
// The simple threshold on World: a StrengthModel::Threshold destructible drawn by this page.
let thresholdOn=false;
let learned=null,learnedMs=0,learnedBusy=false;
// The bond graph and the View menu (0 materials, 1 sleep state, 2 bodies): this page's
// renderer reads them as it draws; the stress modes draw with their own (Rust, stress-web),
// told here.
const viewMode=()=>Number($('view-mode').value);
function applyView(){if(stressOn)stress?.setView($('show-bonds').checked,viewMode())}
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
// Chunk crushing in the threshold mode (engine/docs/destruction-reactions.md): 0 off, 1 the
// scene's own crush parameters (a ScenePack v3's materials[].crush), 2 those and the cited presets.
const crushMode=()=>Number($('crush').value);
const scaleLabel=k=>`×${k>=1?k:`1/${1/k}`}`;
function applyBallSpeed(){
 const k=speedScale(),ms=(stressOn?stress?.stats.ballSpeed:sim?.shotSpeed)*k;
 $('ball-speed-value').textContent=`${scaleLabel(k)}${ms?` · ${ms.toFixed(ms<10?1:0)} m/s`:''}`;
 if(stressOn)stress?.setShotSpeedScale(k);else sim?.setProjectileSpeedScale(k);
}
// CCD balls: the next shots carry BODY_CCD (rubble modes; the stress modes fire their own).
function applyBallCcd(){if(!stressOn&&!thresholdOn)sim?.setProjectileCcd($('ball-ccd').checked)}
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
const stepHz=()=>Math.round(1/(sim?.dt||scene?.params?.dt||1/120));
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
  value?.stepMs?`${value.stepMs.toFixed(2)} ms/tick CPU`:`target ${stepHz()} TPS`;
 $('frame-ms').textContent=value&&frameMs?`${frameMs.toFixed(1)} ms GPU/frame · max ${Math.floor(1000/frameMs)} FPS`:
  value?`${value.frameMs.toFixed(1)} ms/frame`:'frames / second';
 // Per tick: CPU time encoding and submitting the GPU work, then time spent
 // waiting on the GPU (the readback, or a full queue of overlapped ticks).
 // Sliced ticks run between frames, so no wait is measured for them.
 $('timing').textContent=!value?.stepMs?'TPS counts physics ticks; FPS counts rendered frames.':
  `Physics per tick: ${value.encodeMs.toFixed(2)} ms CPU (World.step: encode and submit; results arrive a frame or two later, never waited for)`;
},undefined,()=>stepHz());
document.addEventListener('visibilitychange',()=>{last=0;accumulator=0;rates.reset()});
const setStatus=text=>{status.textContent=text;status.classList.remove('error')};
const fail=e=>{console.error(e?.message??e);running=false;status.textContent=e.message;status.classList.add('error');$('release').textContent=released?'Resume':'Start simulation'};
const controls=enabled=>{enabled||=stressOn&&!loading;for(const id of ['release','reset','level','metadata','fire','volley','meteor','topple'])$(id).disabled=!enabled;for(const id of ['file','sample','scene'])$(id).disabled=!device||loading};
function serializeLoad(task){const result=loadTail.catch(()=>{}).then(async()=>{loading=true;running=false;controls(false);await frame;rates.reset();try{await task()}catch(e){fail(e)}finally{loading=false;last=0;rates.reset();controls(!!sim)}});loadTail=result;return result}
let shotQueue=Promise.resolve();
function shoot(ray){
 if(loading||!sim)return;
 shotQueue=shotQueue.then(async()=>{await frame;if(loading||!sim)return;try{placeCrowd(ray);trailLaunched(sim.launch(ray.origin,ray.direction));if(!running)rates.reset();released=true;running=true;accumulator=0;$('release').textContent='Pause'}catch(e){fail(e)}});
 return shotQueue;
}
async function loadLevel(level){
 let next;try{
  // Settle one house under its own weight first, so every copy starts asleep in
  // static equilibrium (bonds carrying their load) rather than at zero force.
  // Breakable bonds are experimental (opt-in): see README "Breakable bonds".
  if(stressMode())return loadStress(level);
  stressOn=false;$('view').hidden=false;$('stress-view').hidden=true;if(stress)stress.running=false;$('strike-prediction').disabled=true;$('contact-fracture').disabled=true;$('pack-shot-row').hidden=true;pendingShot=null;
  if(thresholdMode())return loadThreshold(level);
  thresholdOn=false;
  learned?.dispose();learned=null;learnedMs=0;
  const raw=procedural==='pack'?buildPackScene(packBytes):procedural?buildCuboidHouse():buildScene(asset,level,{maxHullVertices});
  const built=physicsHz()?withParams(raw,{dt:1/physicsHz()}):raw;
  if(towersBlock){
   // The pack's welds as authored (no settling pass: the towers start at rest),
   // weakened so the volley can bring them down; loose pieces with Bonds: None.
   const one=bondMode()==='none'?withoutJoints(built):withBondStrength(built,TOWERS.strength*strengthScale());
   // Parked cars and pedestrians on the streets: what the debris is steered at.
   const laid=withCity(buildTowersScene(one,towersMeteor?{...TOWERS,...METEOR}:TOWERS));
   next=radiusScale()===1?laid:withProjectileRadius(laid,radiusScale());
  }else{
  if(bondMode()==='d6'&&!towersBlock)setStatus('Settling bonds under self-weight…');
  // Learned mode: rubble stays unbonded (same as none); the MGN-T forward is timed separately.
  // D6 bonds: one house settled on World (settleBonds), then laid out.
  const one=bondMode()==='d6'?await settleBonds(strengthScale()===1?built:withBondStrength(built,strengthScale()),setStatus):withoutJoints(built);
  // Copies laid out as the sample says (procedural houses get a larger contact budget).
  const laid=layoutScene(one,layout,!!procedural);
  next=radiusScale()===1?laid:withProjectileRadius(laid,radiusScale())}}catch(e){if(scene)$('level').value=String(scene.level??0);throw e}
 renderer?.dispose();renderer=null;sim?.dispose();sim=null;released=false;accumulator=0;
 scene=next;volley=null;setStatus('Building the World…');sim=await createLabWorld(scene,bondAsset(scene),{gpuTiming:GPU_TIMING});renderer=await createRenderer(device,$('view'),sim,scene,shoot);resetTrails();
 renderer.colorMode=viewMode();applyCamera();applyBallMass();applyBallSpeed();applyBallCcd();showBallRadius();bystanders=towersBlock?sceneHandle(scene).cityZones():[];applyGuide();$('near').textContent='—';$('meteor').hidden=!towersMeteor;$('topple').hidden=!towersBlock;
 if(bondMode()==='learned'){
  setStatus('Loading learned-impact kernels (Slang→WGSL)…');
  const {createLearnedImpact}=await import('./learned-impact.mjs');
  learned=await createLearnedImpact(device,{status:setStatus});
  learnedMs=await learned.run();
 }
 const copies=scene.copies||1;$('fragments').textContent=scene.selected.length*copies;$('hulls').textContent=procedural?scene.pieces*copies:scene.hullCount;$('bonds').textContent=scene.bondCount*copies;$('release').textContent='Start simulation';
 $('fire').hidden=!scene.rowShot||towersBlock;$('volley').hidden=!towersBlock;$('awake').textContent='0';$('contacts').textContent='0';
 const skipped=scene.skippedHulls.length?` · ${scene.skippedHulls.length} flat zero-volume hulls skipped${copies>1?' per house':''}`:'';
 const what=towersBlock?`${copies} fractured towers · ${scene.bodyCount-1-scene.projectileCount} pieces · ${scene.jointCount} bonds · `:copies>1?`${copies} houses · ${scene.bodyCount-1-scene.projectileCount} bodies · `:'';
 const source=procedural==='pack'?`scene pack · ${scene.pieces} nodes`:procedural?'procedural cuboid houses (box colliders)':`${(asset.bytes/1048576).toFixed(2)} MB · ${asset.materialCount} materials${skipped}`;
 const learnedNote=learned?` · learned-impact ${learnedMs.toFixed(2)} ms/fwd (${learned.graph.nNodes} chunks, ${learned.params} weights, compute only)`:'';
 setStatus(`${what}${source}${learnedNote} · ${towersBlock?'fire the volley or click to shoot':scene.rowShot?'fire the ball through the row or click to shoot':'click the model to shoot'}`);
}
// The stress-solver mode, built and run by stress-web: a GLB's level (one copy of the
// structure; the sample's layout of copies is the other modes'), the cuboid houses in
// their sample's layout (every copy), or a scene pack.
async function loadStress(level){
 // A fracture GLB's level (one copy of the structure), the cuboid houses in their
 // sample's layout (every copy), or a scene pack.
 const cuboid=procedural==='cuboid',pack=procedural==='pack';
 if(!cuboid&&!pack&&(procedural||!assetBytes))throw new Error('The stress-solver mode runs a fracture GLB, a scene pack or the cuboid houses: pick one of those scenes.');
 thresholdOn=false;learned?.dispose();learned=null;learnedMs=0;
 renderer?.dispose();renderer=null;sim?.dispose();sim=null;released=false;
 $('view').hidden=true;$('stress-view').hidden=false;
 setStatus('Loading the stress solver (Rust and Slang on WebGPU)…');
 // The mode is on only once its module loaded: a failed import (an old build)
 // leaves the page in its own renderer with the error shown.
 try{if(!stress){const {createStressMode}=await import('./stress-mode.mjs');stress=await createStressMode($('stress-view'))}}
 catch(e){$('view').hidden=false;$('stress-view').hidden=true;throw new Error(`Stress-solver mode unavailable: ${e?.message??e}. Rebuild it (README, Bond modes).`)}
 stressOn=true;
 const profile=STRESS_PROFILES[bondMode()];
 stress.viewMode=cameraMode();stress.setProfile(profile);stress.setPhysicsHz(physicsHz());stress.setCrush(profile==='threshold'?crushMode():0);stress.setBondStrengthScale(strengthScale());stress.setShotRadiusScale(radiusScale());applyView();
 if(cuboid)await stress.loadCuboid(layout);else if(pack)await stress.loadPack(packBytes);else await stress.load(assetBytes,level);
 await stress.frame(0);applyBallMass();applyBallSpeed();showBallRadius();
 const st=stress.stats;stressHud=stressHudReset(performance.now());
 $('fragments').textContent=st.chunks;$('bonds').textContent=st.bonds;$('hulls').textContent=st.hulls;$('awake').textContent='0';$('contacts').textContent='—';
 $('release').textContent='Start simulation';$('fire').hidden=true;
 $('strike-prediction').checked=stress.strikePrediction;$('strike-prediction').disabled=false;
 $('contact-fracture').checked=stress.contactFracture;$('contact-fracture').disabled=!stress.contactFractureAvailable;
 setStatus(`Stress solver · ${st.chunks} chunks · ${st.bonds} bonds · ${PROFILE_NOTES[profile]??profile}, resident GPU frame at ${Math.round(st.frameHz)} Hz${cuboid||pack?'':' · one copy'}${stressWarning(st)} · click the model to shoot`);
 pendingShot=null;showPackShots();
}
// The simple threshold (the stress mode's v0 rule) on World: a StrengthModel::Threshold
// destructible per copy on the world's shared GPU fracture, drawn by this page's renderer
// from World's poses. A fracture GLB's level (one copy, as the stress mode imports it), the
// cuboid houses in their sample's layout (every copy) or a scene pack.
async function loadThreshold(level){
 const cuboid=procedural==='cuboid',pack=procedural==='pack';
 if(!cuboid&&!pack&&(procedural||!assetBytes))throw new Error('The threshold mode runs a fracture GLB, a scene pack or the cuboid houses: pick one of those scenes.');
 learned?.dispose();learned=null;learnedMs=0;
 renderer?.dispose();renderer=null;sim?.dispose();sim=null;released=false;accumulator=0;
 setStatus('Building the threshold World…');
 // The scene World's asset is built from (DestructibleAssetDesc::from_glb: the level at the
 // default hull budget), so its chunks are the drawn ones.
 const raw=pack?buildPackScene(packBytes):cuboid?layoutScene(buildCuboidHouse(),layout,true):buildScene(asset,level);
 const lab=radiusScale()===1?raw:withProjectileRadius(raw,radiusScale());
 const source=thresholdAsset(pack?'pack':cuboid?'cuboid':'glb',pack?packBytes:cuboid?null:assetBytes,level);
 const totals=source.totals();
 try{sim=await createLabWorld(lab,source,{copies:lab.copies||1,physicsHz:physicsHz(),crush:crushMode(),strengthScale:strengthScale()})}finally{source.free()}
 thresholdOn=true;scene=lab;volley=null;
 renderer=await createRenderer(device,$('view'),sim,scene,shoot);resetTrails();
 renderer.colorMode=viewMode();applyCamera();applyBallMass();applyBallSpeed();showBallRadius();bystanders=[];renderer.setZones(new Float32Array(0));
 const copies=lab.copies||1;
 $('fragments').textContent=totals[0]*copies;$('hulls').textContent=totals[1]*copies;$('bonds').textContent=totals[2]*copies;$('awake').textContent='0';$('contacts').textContent='0';$('near').textContent='—';
 $('release').textContent='Start simulation';$('fire').hidden=true;$('volley').hidden=true;$('meteor').hidden=true;$('topple').hidden=true;
 // World predicts every projectile's strike (World::body_set_projectile); it has no
 // in-step contact fracture.
 $('strike-prediction').checked=true;$('strike-prediction').disabled=true;$('contact-fracture').checked=false;$('contact-fracture').disabled=true;
 const shots=pack?sim.packShots:[];
 $('pack-shot').replaceChildren(...shots.map((s,i)=>new Option(`${s.label} · ${(s.mass/1000).toFixed(1)} t at ${s.speed.toFixed(1)} m/s`,i)));
 $('pack-shot-row').hidden=!shots.length;$('pack-shot').title=shots.map(s=>`${s.label}: ${s.note}`).join('\n');
 pendingShot=null;
 if(urlShot!==null){const i=shots.findIndex(s=>s.label===urlShot||s.label===`${urlShot}x`);
  if(i<0)throw new Error(`No authored shot ${urlShot}: ${shots.map(s=>s.label).join(', ')}`);
  $('pack-shot').value=String(i);pendingShot={index:i,at:shots[i].launchAt};urlShot=null;released=true;running=true;$('release').textContent='Pause'}
 setStatus(`Threshold (World) · ${totals[0]*copies} chunks · ${totals[2]*copies} bonds · ${PROFILE_NOTES.threshold}, ${stepHz()} Hz${cuboid||pack?'':' · one copy'} · click the model to shoot`);
}
// The threshold mode's status line.
const thresholdStatus=()=>{const st=sim.stats(),b=sim.breaks;
 return `${sim.shots} shots · ${sim.steps} frames · ${st.actors} bodies · ${st.bonds-st.bondsIntact} bonds broken${crushMode()?` · ${st.crushed} crushed`:' · crush off'} · ${(sim.steps/stepHz()).toFixed(2)} s`};
async function importBytes(bytes,name,sampleKey="",entry={}){
 assetBytes=bytes;layout=entry.layout??null;maxHullVertices=entry.maxHullVertices;procedural=null;towersBlock=false;
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
// Generated in Rust (engine-fracture glass_tower.rs): a pack built on the spot, no file.
samples['glass-tower']={key:'glass-tower',label:'glass tower and pavilion',generated:glassTowerPack};
$('scene').replaceChildren(...sceneGroups,optgroup('Generated',[new Option('Glass tower · 10 storeys of shattering glass, a pavilion beside it','glass-tower')]),optgroup('Your file',[Object.assign(new Option('Custom GLB (choose or drop one below)',''),{disabled:true})]));
// The punch-through wall (ScenePack v3 with crush blocks, blast-stress-solver crush-wall.json):
// chunk crushing's showcase (Bonds: simple threshold, Crush: on).
samples['crush-wall']={key:'crush-wall',label:'punch-through wall',pack:'./packs/crush-wall.json',nodes:216,bonds:464};
$('scene').insertBefore(optgroup('Chunk crushing (ScenePack v3)',[new Option('Punch-through wall · 12×8×2 blocks','crush-wall')]),$('scene').lastElementChild);
// vibe-city /destructible-stress presets, generated into ../generated/vibe-city (optional).
const VIBE_CITY='../generated/vibe-city/';
try{const index=await (await fetch(`${VIBE_CITY}index.json`)).json();
 const group=document.createElement('optgroup');group.label='Vibe City presets (generated scene packs)';
 const tower=index.presets.find(p=>p.id==='fracturedTower'&&!p.failed&&!p.pending);
 if(tower){
  // The CCD showcase: the fractured tower as a 2×2 city block, fast off-angle balls.
  samples['vc-towers']={key:'vc-towers',label:'fractured towers block',pack:`${VIBE_CITY}fracturedTower.json`,nodes:tower.nodes*4,bonds:tower.bonds*4,towers:true};
  group.append(new Option(`Fractured towers · 2×2 city block · ${tower.nodes*4} nodes (CCD volley)`,'vc-towers'));
  // The same block, cars and pedestrians in its streets, and a meteor (Directed debris demo).
  samples['vc-meteor']={...samples['vc-towers'],key:'vc-meteor',label:'meteor strike',meteor:true};
  group.append(new Option(`Meteor strike · fractured towers, cars, pedestrians`,'vc-meteor'));
 }
 for(const p of index.presets.filter(p=>!p.failed&&!p.pending)){
  const key=`vc:${p.id}`;samples[key]={key,label:p.id,pack:`${VIBE_CITY}${p.id}.json`,nodes:p.nodes,bonds:p.bonds};
  group.append(new Option(`${p.id} · ${p.nodes} nodes`,key));
 }
 if(group.children.length)$('scene').insertBefore(group,$('scene').lastElementChild);
}catch{}
// The reference destruction rigs (packs/index.json, port-reference-packs.mjs): scene packs
// with authored shots, run in the stress-solver modes (destruction-reactions.md).
try{const index=await (await fetch('./packs/index.json')).json();
 const group=document.createElement('optgroup');group.label='Reference rigs (scene packs, authored shots)';
 for(const p of index.packs){const key=`ref:${p.key}`;samples[key]={key,label:p.title,pack:`./packs/${p.file}`,nodes:p.nodes,bonds:p.bonds,reference:true,shots:p.shots};
  group.append(new Option(`${p.key} · ${p.nodes} chunks · ${p.bonds} bonds`,key))}
 if(group.children.length)$('scene').insertBefore(group,$('scene').lastElementChild);
}catch{}
// URL: ?pack=crush-wall&shot=1.4x[&bonds=threshold|stress][&camera=x,y,z,tx,ty,tz] picks a
// reference rig, the stress-solver mode (threshold unless bonds= says otherwise) and fires
// the authored shot at its launch time (the source's 2 s settle); shot=1.4 matches 1.4x.
const query=new URLSearchParams(location.search);
// ?gputime=phases: World times each phase of the step (sim.gpuTime.segments); default the step.
const GPU_TIMING=query.get('gputime')==='phases'?'phases':'step';
let urlShot=null;
{const pack=query.get('pack'),bonds=query.get('bonds');
 if(bonds&&$('bond-mode').querySelector(`option[value="${bonds}"]`))$('bond-mode').value=bonds;
 if(pack&&samples[`ref:${pack}`]){$('scene').value=`ref:${pack}`;if(!bonds)$('bond-mode').value='threshold';urlShot=query.get('shot')}}
// The pack's authored shots in the panel (stress modes), and firing one.
function showPackShots(){
 const shots=stressOn&&procedural==='pack'?stress.packShots:[];
 $('pack-shot').replaceChildren(...shots.map((s,i)=>new Option(`${s.label} · ${(s.mass/1000).toFixed(1)} t at ${s.speed.toFixed(1)} m/s`,i)));
 $('pack-shot-row').hidden=!shots.length;$('pack-shot').title=shots.map(s=>`${s.label}: ${s.note}`).join('\n');
 if(urlShot!==null){const i=shots.findIndex(s=>s.label===urlShot||s.label===`${urlShot}x`);
  if(i<0)throw new Error(`No authored shot ${urlShot}: ${shots.map(s=>s.label).join(', ')}`);
  $('pack-shot').value=String(i);pendingShot={index:i,at:shots[i].launchAt};urlShot=null;stress.running=true}
}
// An authored shot waiting for its launch time (s of simulated time).
let pendingShot=null;
function firePendingShot(st){
 if(thresholdOn){if(pendingShot&&sim&&sim.steps*sim.dt>=pendingShot.at){trailLaunched(sim.firePackShot(pendingShot.index));pendingShot=null}return}
 if(!pendingShot||(st.frames??0)/st.frameHz<pendingShot.at)return;
 stress.firePackShot(pendingShot.index);pendingShot=null;
}
$('fire-pack-shot').onclick=()=>{
 if(thresholdOn&&sim){trailLaunched(sim.firePackShot(Number($('pack-shot').value)));released=true;running=true;$('release').textContent='Pause';return}
 if(stressOn&&stress){stress.firePackShot(Number($('pack-shot').value));stress.running=true}};
const sample=(key=$('scene').value||'house')=>serializeLoad(async()=>{
 const entry=samples[key];if(!entry)throw new Error('Unknown bundled scene');
 if(entry.pack||entry.generated){
  // A scene pack, one level with the pack's bonds: built in Rust (entry.generated), or
  // generated into ../generated (engine/scenes/vibe-city, bun run export.ts).
  if(entry.generated)packBytes=entry.generated().buffer;
  else{
   setStatus(`Fetching ${entry.label}…`);
   const response=await fetch(entry.pack);
   if(!response.ok)throw new Error(entry.reference?`Scene pack missing: ${entry.pack}`:'Scene pack missing: run bun run export.ts in engine/scenes/vibe-city.');
   packBytes=await response.arrayBuffer();
  }
  assetBytes=null;layout=null;procedural='pack';
  // The towers block: breakable welds, CCD balls at 16× unless the panel says otherwise.
  if(entry.towers&&(!towersBlock||towersMeteor!==!!entry.meteor)){$('bond-mode').value='d6';$('ball-speed').value=entry.meteor?'0':'4';$('ball-ccd').checked=true}
  towersBlock=!!entry.towers;towersMeteor=!!entry.meteor;asset?.handle.free();asset=null;$('scene').value=key;$('filename').textContent=entry.pack?entry.pack.split('/').pop():'generated in Rust (glass_tower.rs)';
  $('level').replaceChildren(new Option(entry.nodes!=null?`Scene pack · ${entry.nodes} nodes · ${entry.bonds} bonds`:'Scene pack · generated',0));$('level').value='0';
  await loadLevel(0);return;
 }
 if(entry.procedural){
  towersBlock=false;
  // Built in code from box colliders: no GLB, one level, no baked bonds.
  layout=entry.layout??null;procedural=entry.procedural;assetBytes=null;asset?.handle.free();asset=null;$('scene').value=key;$('filename').textContent='Procedural cuboid house';
  $('level').replaceChildren(new Option('Cuboids · box colliders only',0));$('level').value='0';
  await loadLevel(0);return;
 }
 towersBlock=false;
 setStatus(`Fetching baked ${entry.label}…`);
 const response=await fetch(`./assets/${entry.file}`);
 if(!response.ok)throw new Error('Bundled asset missing. Follow README.md to export it.');
 await importBytes(await response.arrayBuffer(),entry.file,key,entry);
});
$('sample').onclick=()=>sample();
$('fire').onclick=()=>scene?.rowShot&&shoot(scene.rowShot);
// The volley's shots fire from the frame loop at their steps (never inside a
// step split across frames), counted from the step the button was pressed at.
function startShots(shots){applyBallCcd();applyBallSpeed();
 const base=sim.steps;volley={shots:shots.map(s=>({...s,step:s.step+base})),next:0};released=true;running=true;accumulator=0;$('release').textContent='Pause'}
$('volley').onclick=()=>{if(sim&&towersBlock)startShots(towersVolley(scene))};
// The meteor scene's strike, and base shots that topple the first tower over its street.
$('meteor').onclick=()=>{if(sim&&towersMeteor)startShots([meteorShot(scene)])};
$('topple').onclick=()=>{if(sim&&towersBlock)startShots(toppleShots(scene))};
function fireVolley(){
 if(!volley)return;
 while(volley.next<volley.shots.length&&volley.shots[volley.next].step<=sim.steps){const s=volley.shots[volley.next++];trailLaunched(sim.launch(s.origin,s.dir,s.speed?{speed:s.speed,arc:false}:undefined))}
 if(volley.next>=volley.shots.length)volley=null;
}
$('scene').onchange=()=>sample($('scene').value);
$('file').onchange=()=>{const f=$('file').files[0];if(f)serializeLoad(async()=>importBytes(await f.arrayBuffer(),f.name))};
$('level').onchange=()=>serializeLoad(()=>loadLevel(Number($('level').value)));
$('view-mode').onchange=()=>{if(renderer)renderer.colorMode=viewMode();applyView()};
$('show-bonds').onchange=applyView;
$('strike-prediction').onchange=()=>{if(stressOn&&stress)stress.setStrikePrediction($('strike-prediction').checked)};
$('contact-fracture').onchange=()=>{if(stressOn&&stress){stress.setContactFracture($('contact-fracture').checked);serializeLoad(()=>loadLevel(Number($('level').value)))}};
$('ball-mass').oninput=applyBallMass;
$('ball-speed').oninput=applyBallSpeed;
$('ball-ccd').onchange=applyBallCcd;
$('ball-radius').oninput=()=>{$('ball-radius-value').textContent=scaleLabel(radiusScale())};
$('ball-radius').onchange=()=>serializeLoad(()=>loadLevel(Number($('level').value)));
$('bond-strength').oninput=()=>{$('bond-strength-value').textContent=scaleLabel(strengthScale())};
$('bond-strength').onchange=()=>{if(bondMode()!=='none')serializeLoad(()=>loadLevel(Number($('level').value)))};
$('bond-mode').onchange=()=>serializeLoad(()=>loadLevel(Number($('level').value)));
$('crush').onchange=()=>serializeLoad(()=>loadLevel(Number($('level').value)));
$('physics-hz').onchange=()=>serializeLoad(()=>loadLevel(Number($('level').value)));
$('release').onclick=()=>{if(stressOn&&stress){stress.running=!stress.running;$('release').textContent=stress.running?'Pause':'Resume';return}released=true;running=!running;$('release').textContent=running?'Pause':'Resume';accumulator=0;rates.reset()};
$('reset').onclick=()=>serializeLoad(async()=>{if(stressOn){await loadStress(Number($('level').value));return}if(thresholdOn){await loadThreshold(Number($('level').value));return}
 sim.reset();applyGuide();volley=null;resetTrails();released=false;$('release').textContent='Start simulation';setStatus('Reset · click the model to shoot')});
$('metadata').onclick=()=>{if(!asset)return;const url=URL.createObjectURL(new Blob([asset.metadataJson()],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='fracture-metadata.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
for(const type of ['dragenter','dragover'])window.addEventListener(type,e=>{e.preventDefault();document.body.classList.add('drop')});
window.addEventListener('dragleave',()=>document.body.classList.remove('drop'));
window.addEventListener('drop',e=>{e.preventDefault();document.body.classList.remove('drop');const f=e.dataTransfer.files[0];if(f&&device)serializeLoad(async()=>importBytes(await f.arrayBuffer(),f.name))});
window.__FRACTURE_DEMO__={
 // The stress modes, frame-locked for recording: ready once loaded; each advance runs one
 // display frame of `ms` (the physics frames that fell due, a URL shot at its launch time,
 // the draw). orbit/zoom move the stress camera as a drag would. The threshold mode (World)
 // steps those frames and waits for their results before drawing.
 get stressReady(){return !loading&&(stressOn&&!!stress||thresholdOn&&!!sim)},
 async stressAdvance(ms=1000/60){stressCapture=true;await frame;
  if(thresholdOn){
   lockedBehind+=ms/1000;const n=Math.floor(lockedBehind*stepHz()+1e-9);lockedBehind-=n/stepHz();
   if(running&&n)await sim.step(n);firePendingShot(null);renderer.draw($('show-bonds').checked);setStatus(thresholdStatus());return thresholdStats()}
  await stress.frame(ms);const st=stress.stats;firePendingShot(st);setStatus(stressStatus(st));return st},
 // Back to the page's own frame loop after stressAdvance (frame-locked capture) took it over.
 stressRelease(){stressCapture=false;last=0},
 stressOrbit(dx,dy){if(thresholdOn)renderer?.orbit(dx,dy);else stress.orbit(dx,dy)},stressZoom(delta){if(thresholdOn)renderer?.zoom(delta);else stress.zoom(delta)},
 get status(){return $('status').textContent},
 // The page's own shot (a click's path: queued behind the frame, the loop keeps running).
 shoot:ray=>shoot(ray),
 // Run or pause the frame loop's physics, as the Start / Pause button does.
 setRunning(on){released=true;running=!!on;accumulator=0;$('release').textContent=running?'Pause':'Resume'},
 get running(){return running},
 // The towers volley still firing (the Fire the volley button).
 get volleyActive(){return !!volley},get ready(){return !loading&&!!sim},get device(){return device},get asset(){return asset},get scene(){return scene},get simulation(){return sim},get renderer(){return renderer},get rates(){return currentRates},
 async shootRay(origin,direction){running=false;await frame;placeCrowd({origin,direction});return sim.launch(origin,direction)},
 // The current Near Miss crowd (one ring zone per bystander).
 get bystanders(){return bystanders},
 async fireRow(){running=false;await frame;return sim.launch(scene.rowShot.origin,scene.rowShot.direction)},
 // Repeatable row benchmark: fire, then every `every` steps record wall ms per step, the
 // ball and the counters (World's per-phase GPU split: `profile` takes
 // GpuTiming::Phases, which this page does not ask for; `gpuMs` is the step's).
 async benchmarkRow({steps=600,every=60}={}){
  running=false;await frame;const id=sim.launch(scene.rowShot.origin,scene.rowShot.direction),log=[];
  for(let done=0;done<steps;){
   const n=Math.min(every,steps-done),t=performance.now();await sim.step(n);done+=n;
   const ms=(performance.now()-t)/n,snap=await sim.snapshot(),c=sim.counters,o=id*28;
   log.push({step:done,wallMs:+ms.toFixed(2),ballZ:+snap[o+2].toFixed(1),ballVz:+snap[o+14].toFixed(1),awake:c.awake,contacts:c.contact,
    candidates:c.candidate,colors:c.color,overflow:c.overflow,gpuMs:sim.gpuTickMs==null?null:+sim.gpuTickMs.toFixed(2)});
  }
  renderer.draw($('show-bonds').checked);return log;
 },
 // `count` steps as fast as World's readbacks allow, every frame read, then a draw.
 async stepFrames(count){running=false;await frame;await sim.step(count);renderer.draw($('show-bonds').checked);return sim.counters},
 async snapshot(){await frame;return sim.snapshot()},reset:()=>serializeLoad(async()=>{if(thresholdOn)await loadThreshold(Number($('level').value));else{sim.reset();applyGuide()}}),load:sample,selectLevel:level=>serializeLoad(async()=>{$("level").value=String(level);await loadLevel(level)})};
// page-bench.mjs: frame-locked steps.
window.__step=n=>window.__FRACTURE_DEMO__.stepFrames(n);
// Checks and benches (the before/after comparisons): steps, the breaks since the shot.
let shotStep=0;
window.__lab={
 async step(n){await sim.step(n);return sim.counters},async one(){await sim.step(1);return sim.counters},
 broken(){const st=sim.stats(),f=sim.breaks.firstStep;return {broken:st.bonds-st.bondsIntact,actors:st.actors,crushed:st.crushed,firstStep:f===null?null:f-shotStep+1}},
 async fireRay(){const d=window.__FRACTURE_DEMO__,c=scene.center,e=Math.max(scene.extent,1);const id=await d.shootRay([c[0]+0.13,c[1],c[2]+3*e],[0,0,-1]);shotStep=sim.steps;return id},
 reset(){return window.__FRACTURE_DEMO__.reset()},
};
// The threshold mode's numbers, as the stress modes' stats.
let lockedBehind=0;
const thresholdStats=()=>{const st=sim.stats();return {frames:sim.steps,shots:sim.shots,bodies:st.actors,brokenBonds:st.bonds-st.bondsIntact,crush:crushMode(),crushed:st.crushed,chunks:st.chunks,bonds:st.bonds,simSeconds:sim.steps*sim.dt,frameHz:stepHz()}};
// One stress-mode frame: the Rust steps what fell due and draws; the HUD counts physics
// frames and rendered frames per second from its stats.
// What the stress import could not take from the scene as given (stress-web `warnings`).
const stressWarning=st=>st.warnings?.length?` · ⚠ ${st.warnings.join('; ')}`:'';
// The running stress mode's status line (the frame loop's and the frame-locked capture's).
const stressStatus=st=>`${st.shots} shots · ${st.frames} frames · ${st.bodies??'—'} bodies · ${st.brokenBonds} bonds broken${st.crush?` · ${st.crushed} crushed`:' · crush off'} · ${st.simSeconds.toFixed(2)} s${stressWarning(st)}`;
// The stress HUD, from one second of display frames: physics frames and display frames per
// second as they ran; the GPU time of a physics frame and of a draw (timestamps, a running
// mean); the display's refresh (the shorter gaps between display frames); and what the
// machine would allow uncapped: physics at real time takes frameHz × its GPU frame time of
// each second, the draws the rest, and the page's own CPU time per display frame (stepping,
// drawing, encoding; the GPU runs that work later, never waited for) bounds it too.
function stressHudReset(now){return {t:now,frames:0,renders:0,gaps:[],cpuMs:0,last:now,stepMs:stressHud?.stepMs}}
function stressHudShow(st,s){
 const h=stressHud,tps=h.frames/s,fps=h.renders/s,hz=st.frameHz;
 $('tps').textContent=tps.toFixed(0);$('fps').textContent=fps.toFixed(0);$('sim-speed').textContent=`${(tps/hz).toFixed(2)}×`;
 // The refresh: the median of the gaps within 1.5× the shortest (one refresh each; longer
 // gaps are frames that missed one).
 const gaps=h.gaps.sort((a,b)=>a-b).filter((g,_,a)=>g<=1.5*a[0]),gap=gaps[gaps.length>>1];
 const display=gap?Math.round(1000/gap):null,cpu=h.renders?h.cpuMs/h.renders:0;
 const phys=st.gpuStepMs,draw=st.gpuDrawMs;
 $('step-ms').textContent=phys?`${phys.toFixed(2)} ms GPU/tick · max ${Math.floor(1000/phys)} TPS`:
  h.stepMs?`${h.stepMs.toFixed(2)} ms CPU/tick (no GPU timestamps)`:`target ${Math.round(hz)} TPS`;
 $('frame-ms').textContent=draw?`${draw.toFixed(2)} ms GPU/draw${display?` · display ${display} Hz`:''}`:display?`display ${display} Hz`:'frames / second';
 if(!phys||!draw){$('timing').textContent=`CPU ${cpu.toFixed(2)} ms per display frame (stepping, drawing, encoding). No GPU timestamps here: the GPU's time is not measured.`;return}
 // GPU ms per wall second that physics at real time takes, and the draws the rest allows.
 const physBusy=hz*phys,gpuFps=physBusy<1000?(1000-physBusy)/draw:0,cpuFps=cpu>0?1000/cpu:Infinity,uncapped=Math.min(gpuFps,cpuFps);
 $('timing').textContent=physBusy>=1000?
  `GPU-bound: real-time physics needs ${physBusy.toFixed(0)} ms of GPU each second (${Math.round(hz)} ticks × ${phys.toFixed(2)} ms), so it runs at most ${(1000/physBusy).toFixed(2)}× real time. Draw ${draw.toFixed(2)} ms GPU; CPU ${cpu.toFixed(2)} ms per display frame.`:
  `Each second at real time: physics ${physBusy.toFixed(0)} ms GPU (${Math.round(hz)} ticks × ${phys.toFixed(2)} ms), draws ${draw.toFixed(2)} ms GPU each in the rest. Uncapped ≈ ${Math.floor(uncapped)} FPS (${gpuFps<=cpuFps?'GPU':'CPU'}-bound; CPU ${cpu.toFixed(2)} ms per display frame)${display?`; the display caps it at ${display} Hz`:''}.`;
}
function stressTick(now,elapsedMs){
 frame=(async()=>{try{
  const before=stress.stats.frames??0,t0=performance.now();await stress.frame(elapsedMs);const cpuMs=performance.now()-t0;const st=stress.stats;firePendingShot(st);
  const h=stressHud;h.frames+=(st.frames??0)-before;h.renders++;h.cpuMs+=cpuMs;if(now>h.last)h.gaps.push(now-h.last);h.last=now;
  // A display frame that stepped no physics frame (a 120 Hz display, a 60 Hz simulation)
  // reports no step time: the last measured one stays.
  if(st.stepMs)h.stepMs=st.stepMs;
  if(now-h.t>=1000){stressHudShow(st,(now-h.t)/1000);stressHud=stressHudReset(now)}
  $('awake').textContent=st.moving??st.awake??'—';
  if(stress.running||st.shots){$('release').textContent=stress.running?'Pause':'Resume';setStatus(stressStatus(st))}
 }catch(e){fail(e)}})();
 frame.finally(()=>requestAnimationFrame(tick));
}
// Frame-locked capture (record-reactions.mjs): the stress mode steps only when asked.
let stressCapture=false;
function tick(now){
 if((stressOn||thresholdOn)&&stressCapture){requestAnimationFrame(tick);return}
 if(stressOn&&!loading&&stress){const ms=last?Math.min(250,now-last):0;last=now;return stressTick(now,ms)}
 const elapsed=last?Math.min(.04,(now-last)/1000):0;last=now;
 if(!loading&&renderer){frame=(async()=>{try{
  let ticks=0;
  if(running)fireVolley();
  if(running){
   accumulator=Math.min(accumulator+elapsed,MAX_TICKS_PER_FRAME/stepHz());
   // The due ticks World can take without waiting; the rest stay due (never dropped).
   const due=Math.min(MAX_TICKS_PER_FRAME,Math.floor(accumulator*stepHz()+1e-9));
   ticks=sim.tick(due);accumulator-=ticks/stepHz();
   if(!document.hidden)for(let i=0;i<ticks;i++)rates.step(sim.timing);
   if(thresholdOn)firePendingShot(null);
  }else sim.poll();
  // Learned-impact: time one forward per frame when idle enough (discard output).
  if(learned&&!learnedBusy){learnedBusy=true;learned.run().then(ms=>{learnedMs=ms;learnedBusy=false;$('step-ms').textContent=`learned ${ms.toFixed(2)} ms/fwd · ${learned.graph.nNodes} n · ${learned.params} w`}).catch(e=>{learnedBusy=false;fail(e)})}
  updateTrails();renderer.draw($('show-bonds').checked);
  if(running){const c=sim.counters;$("awake").textContent=c.awake;$("contacts").textContent=c.contact;updateNear(now)}
  if(running)setStatus(thresholdOn?thresholdStatus():(()=>{const c=sim.counters;return `${sim.shots} shots · ${sim.steps} steps · ${c.contact} contacts · ${c.awake} awake · ${(sim.steps/stepHz()).toFixed(1)} s`})());
  if(!document.hidden)rates.frame();
 }catch(e){fail(e)}})();frame.finally(()=>requestAnimationFrame(tick))}else requestAnimationFrame(tick);
}
try{
 if(!navigator.gpu)throw new Error('WebGPU is unavailable. Open this localhost page in a WebGPU-enabled browser.');
 // World's device (lab-world.mjs): the physics and this page's renderer share it.
 device=await labDevice();
 device.addEventListener('uncapturederror',e=>fail(e.error));device.lost.then(info=>fail(new Error(`GPU device lost: ${info.message}. Reload this page.`)));
 // ?scene=<key>&bonds=<mode>&crush=<0|1|2> preselect the panel (e.g. scene=crush-wall&bonds=threshold&crush=1).
 const q=new URLSearchParams(location.search);
 if(q.has('scene')&&samples[q.get('scene')])$('scene').value=q.get('scene');
 if(q.has('bonds'))$('bond-mode').value=q.get('bonds');
 if(q.has('crush'))$('crush').value=q.get('crush');
 await sample();requestAnimationFrame(tick);
}catch(e){fail(e);controls(false)}
