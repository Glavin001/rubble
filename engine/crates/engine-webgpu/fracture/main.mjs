import {parseFractureGlb,sceneHandle,glassTowerPack,nearMissSettings,labFractureDefaults,labDefaultPhysicsHz,SAMPLES} from './wasm.mjs';
import {createLab,labDevice} from './lab-world.mjs';
import {createRenderer} from './render.mjs';
import {createRates} from './rates.mjs';
// The fracture lab: one World (rubble's Rust engine, the one the C library gives Unreal) per
// run, drawn by this page's renderer. The page wires its controls to the lab API
// (engine-world `rubble::lab`, through lab-world.mjs) and shows what it reports: Rust decides
// every scene, bond model, setting, shot and number. The one exception is the blast_parity
// stress solver, which runs on its own engine (stress-web) until it is ported.
const $=id=>document.getElementById(id),status=$('status');
let device,asset,scene,sim,renderer,running=false,released=false,loading=true,frame=Promise.resolve(),loadTail=Promise.resolve(),last=0,accumulator=0;
// Physics ticks one frame may take (World steps whole and never waits: a tick with no free
// readback slot stays due for the next frame).
const MAX_TICKS_PER_FRAME=4;
// The scene's source: a GLB (assetBytes, the level menu), a pack (packBytes) or the cuboid
// house; how its copies are arranged (layout, or the tower block: towers, meteor).
let currentRates=null,layout=null,maxHullVertices,procedural=null,assetBytes=null,packBytes=null,towers=null;
// Ball trails: per ball slot, its path and whether it was a CCD shot; positions come from a
// small async readback of the ball bodies each frame.
let trails=[],trailRead=null;
function resetTrails(){trails=scene?Array.from({length:scene.projectileCount},()=>({path:[],ccd:false})):[];renderer?.setTrails(new Float32Array(0))}
function trailLaunched(id){const t=trails[id-scene.projectileStart];if(t){t.path=[];t.ccd=$('ball-ccd').checked}}
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
// The Bonds menu: the lab's bond models (engine-world `LabBonds`: none, d6, threshold,
// load_path, impact_local, blast_gpu), all on World; 'stress' (the blast_parity stress
// solver, its own engine: stress-web); 'learned' (MGN-T forward on this page's WebGPU,
// compute and timing only, the pieces loose on World).
const bondMode=()=>$('bond-mode').value;
const BLAST_PARITY='stress';
// 'rubble' is a panel preset over load_path (see showFractureSwitches), not a model of its own.
const labBonds=()=>({learned:'none',blast:'blast_gpu',rubble:'load_path'})[bondMode()]??bondMode();
const STRESS_MODELS=new Set(['threshold','load_path','impact_local','blast_gpu']);
const stressModel=()=>STRESS_MODELS.has(labBonds());
const PROFILE_NOTES={threshold:'simple threshold (v0): a bond breaks when a contact force beats its strength',load_path:'load path: gravity and contacts carried through the bonds, each checked for tension, shear, compression and bending',impact_local:'local impact: a shot holes, spalls or craters the wall by its ballistic limit, before the engine steps',blast_gpu:"vibe-land's GPU stress solver (Blast's model, Slang)",d6:'breakable D6 joints',none:'loose pieces',rubble:'Rubble (full): load path for sustained load, analytic impact for strikes with their structural response, predicted before the engine steps'};
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

// Directed debris, "Near Miss" (engine-fracture guidance.rs): each shot finds a crowd of
// bystanders downrange (orange rings); the tower block's are its cars and pedestrians. On,
// debris that can reach one lands around them rather than on open ground; off, it flies as
// it would. Every bond model takes it.
let bystanders=[],nearAt=0,nearBusy=false;
function applyGuide(){
 if(!sim||!scene||stressOn)return;
 sim.setGuidance($('guide').checked?nearMissSettings($('guide-strength').value):null,bystanders);
 renderer?.setZones(sceneHandle(scene).zoneOutline(bystanders));
}
function placeCrowd(ray){
 if(!scene||stressOn||towers)return;
 bystanders=sceneHandle(scene).nearMissZonesFor(Float64Array.from(ray.origin),Float64Array.from(ray.direction));
 applyGuide();$('near').textContent='0';
}
function updateNear(now){
 if(nearBusy||!sim||!bystanders.length||now-nearAt<500)return;nearAt=now;nearBusy=true;
 sim.snapshot().then(s=>{if(sim&&scene)$('near').textContent=sceneHandle(scene).nearCounts(s,bystanders).reduce((a,b)=>a+b,0)}).catch(()=>{}).finally(()=>{nearBusy=false});
}
for(const id of ['guide','guide-strength'])$(id).onchange=()=>{applyGuide();$(id).blur()};
// blast_parity: the stress solver on its own engine and canvas (stress-mode.mjs).
let stress=null,stressOn=false,stressHud={t:0,frames:0,renders:0,gaps:[],cpuMs:0,last:0};
let learned=null,learnedMs=0,learnedBusy=false;
// The View menu (0 materials, 1 sleep state, 2 bodies).
const viewMode=()=>Number($('view-mode').value);
// The stress view (value 3) is the World's (lab-world): each step then carries every chunk's
// load over strength; the separate stress engine has no such mode and draws materials.
function applyView(){if(stressOn)stress?.setView($('show-bonds').checked,viewMode()===3?0:viewMode());sim?.world?.setStressView?.(viewMode()===3);$('stress-scale').disabled=viewMode()!==3;if(sim?.poseDraw)sim.setStressScale?.($('stress-scale').value)}
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
// Ball mass, speed and radius: power-of-two multiples of the scene's ball (mass and speed
// apply to the next shots; the radius when the level loads). Bond strength: a multiple of
// every bond's break strength (reloads).
const massScale=()=>2**Number($('ball-mass').value);
const speedScale=()=>2**Number($('ball-speed').value);
const radiusScale=()=>2**Number($('ball-radius').value);
const strengthScale=()=>2**Number($('bond-strength').value);
// Physics steps per second of simulated time: one explicit rate for every bond model.
const physicsHz=()=>Number($('physics-hz').value)||labDefaultPhysicsHz();
// Chunk crushing (the stress models): 0 off, 1 the scene's own crush parameters, 2 those
// and the cited presets.
const crushMode=()=>stressModel()?Number($('crush').value):0;
const scaleLabel=k=>`×${k>=1?k:`1/${1/k}`}`;
// The ball settings for the next shots (Rust: `Lab::set_ball`).
function applyBall(){
 const k=speedScale(),ms=(stressOn?stress?.stats.ballSpeed:sim?.shotSpeed)*k;
 $('ball-speed-value').textContent=`${scaleLabel(k)}${ms?` · ${ms.toFixed(ms<10?1:0)} m/s`:''}`;
 const m=massScale(),kg=(stressOn?stress?.stats.ballMass:scene?.projectileMass)*m;
 $('ball-mass-value').textContent=`${scaleLabel(m)}${kg?` · ${Math.round(kg).toLocaleString()} kg`:''}`;
 if(stressOn){stress?.setShotSpeedScale(k);stress?.setShotMassScale(m)}else sim?.setBall(m,k,$('ball-ccd').checked);
}
function showBallRadius(){
 const r=stressOn?stress?.stats.ballRadius:scene?.projectileRadius;
 $('ball-radius-value').textContent=`${scaleLabel(radiusScale())}${r?` · ${r.toFixed(2)} m`:''}`;
}
// The stress models' fracture switches: each starts at the model's own (Rust:
// `fracture_defaults`), the panel's choice sent explicitly.
let wasFull=false;
function showFractureSwitches(){
 const own=labFractureDefaults(labBonds());
 for(const [id,key] of [['strike-prediction','strikePrediction'],['contact-fracture','contactFracture']]){
  $(id).disabled=!own;if(own)$(id).checked=own[key];
 }
 // Rubble (full): the compare page's Rubble column (Rust: `play::compare::Approach::Rubble`).
 const full=bondMode()==='rubble';
 if(full){$('strike-prediction').checked=true;$('contact-fracture').checked=false;$('strike-rule').value='analytic';$('impact-response').value='dynamic';$('ball-ccd').checked=false}
 else if(wasFull){$('strike-rule').value='';$('impact-response').value='off';$('ball-ccd').checked=true}
 wasFull=full;
 $('crush').disabled=!stressModel();
 $('strike-rule').disabled=$('support-loss').disabled=!own;
 $('strike-radius').disabled=!own||$('strike-rule').value!=='radial';
 // A strike's structural response is the load path's (Rust: `FractureOptions::impact_response`).
 const lp=labBonds()==='load_path';if(!lp)$('impact-response').value='off';
 $('impact-response').disabled=!lp;$('impact-scale').disabled=!lp||$('impact-response').value==='off';
}
const stepHz=()=>Math.round(sim?1/sim.dt:physicsHz());
const rates=createRates(value=>{
 currentRates=value;
 $('tps').textContent=value?value.tps.toFixed(value.tps>0&&value.tps<10?1:0):'—';
 $('fps').textContent=value?value.fps.toFixed(value.fps>0&&value.fps<10?1:0):'—';
 $('sim-speed').textContent=value?`${value.speed.toFixed(2)}×`:'—';
 // GPU time per physics step and what it allows (World's numbers: LabStats), and the
 // page's own draw.
 const st=sim?.stats(),tickMs=st?.gpuStepMs??sim?.gpuTickMs,frameMs=renderer?.gpuFrameMs;
 $('step-ms').textContent=learned?`learned ${learnedMs.toFixed(2)} ms/fwd · ${learned.graph.nNodes} n · ${learned.params} w`:
  value&&tickMs?`${tickMs.toFixed(2)} ms GPU/tick · max ${Math.floor(st?.maxStepsPerSecond??1000/tickMs)} TPS`:
  value?.stepMs?`${value.stepMs.toFixed(2)} ms/tick CPU`:`target ${stepHz()} TPS`;
 $('frame-ms').textContent=value&&frameMs?`${frameMs.toFixed(2)} ms GPU/frame · max ${Math.floor(1000/frameMs)} FPS`:
  value?`${value.frameMs.toFixed(1)} ms/frame`:'frames / second';
 if(!value?.stepMs){$('timing').textContent='TPS counts physics ticks; FPS counts rendered frames.';return}
 const share=st?.realtimeShare;
 $('timing').textContent=share&&frameMs?
  `Each second at real time: physics ${(1000*share).toFixed(0)} ms GPU (${stepHz()} ticks × ${tickMs.toFixed(2)} ms)${share>=1?', more than the second: it runs slower than real time':`, draws ${frameMs.toFixed(2)} ms GPU each in the rest: uncapped ≈ ${Math.floor((1000-1000*share)/frameMs)} FPS`}. CPU ${value.encodeMs.toFixed(2)} ms per tick (encode and submit).`:
  `Physics per tick: ${value.encodeMs.toFixed(2)} ms CPU (World.step: encode and submit; results arrive a frame or two later, never waited for)`;
},undefined,()=>stepHz());
document.addEventListener('visibilitychange',()=>{last=0;accumulator=0;rates.reset()});
const setStatus=text=>{status.textContent=text;status.classList.remove('error')};
const fail=e=>{console.error(e?.message??e);running=false;status.textContent=e?.message??String(e);status.classList.add('error');$('release').textContent=released?'Resume':'Start simulation'};
const controls=enabled=>{enabled||=stressOn&&!loading;for(const id of ['release','reset','level','metadata','fire','volley','meteor','topple','suggested'])$(id).disabled=!enabled;for(const id of ['file','sample','scene'])$(id).disabled=!device||loading};
function serializeLoad(task){const result=loadTail.catch(()=>{}).then(async()=>{loading=true;running=false;controls(false);await frame;rates.reset();try{await task()}catch(e){fail(e)}finally{loading=false;last=0;rates.reset();controls(!!sim)}});loadTail=result;return result}
// A click's shot: queued on World, fired by the next step (Rust's schedule).
function shoot(ray){
 if(loading)return;
 if(stressOn)return;
 if(!sim)return;
 try{placeCrowd(ray);sim.shoot(ray.origin,ray.direction);if(!running)rates.reset();released=true;running=true;accumulator=0;$('release').textContent='Pause'}catch(e){fail(e)}
}
// Run the queued actions now: start the simulation if it is paused.
function go(){released=true;running=true;accumulator=0;$('release').textContent='Pause'}
// The run's choices, as the lab API takes them (engine-world-wasm lab.rs `DefJs`).
function labDef(level){
 const source=procedural==='cuboid'?'cuboid':procedural==='pack'?'pack':'glb';
 const def={source,level,maxHullVertices,bonds:labBonds(),physicsHz:physicsHz(),strengthScale:strengthScale(),
  ballRadiusScale:radiusScale(),ballMassScale:massScale(),ballSpeedScale:speedScale(),ccd:$('ball-ccd').checked,crush:crushMode(),gpuTiming:GPU_TIMING,
  arrangement:towers?towers:layout?'layout':'one',...(layout&&!towers?{layout}:{})};
 if(stressModel()){def.strikePrediction=$('strike-prediction').checked;def.contactFracture=$('contact-fracture').checked;
  // The strike rule apart from the bond model's load rule (Rust: `FractureOptions::strike`), and support loss.
  const strike=$('strike-rule').value;if(strike){def.strike=strike;if(strike==='radial')def.strikeRadius=Number($('strike-radius').value)}
  const response=$('impact-response').value;if(response!=='off'){def.impactResponse=response;def.impactScale=Number($('impact-scale').value)}
  def.supportLoss=$('support-loss').checked}
 return {def,bytes:source==='glb'?assetBytes:source==='pack'?packBytes:null};
}
async function loadLevel(level){
 if(bondMode()===BLAST_PARITY)return loadStress(level);
 stressOn=false;$('view').hidden=false;$('stress-view').hidden=true;if(stress)stress.running=false;
 learned?.dispose();learned=null;learnedMs=0;
 const {def,bytes}=labDef(level);
 renderer?.dispose();renderer=null;sim?.dispose();sim=null;released=false;accumulator=0;
 setStatus('Building the World…');
 try{sim=await createLab(def,bytes,setStatus)}catch(e){if(scene)$('level').value=String(scene.level??0);throw e}
 scene=sim.scene;sim.onLaunch(trailLaunched);
 renderer=await createRenderer(device,$('view'),sim,scene,shoot);resetTrails();
 renderer.colorMode=viewMode();applyView();applyCamera();applyBall();showBallRadius();
 bystanders=towers?sceneHandle(scene).cityZones():[];applyGuide();$('near').textContent='—';
 if(bondMode()==='learned'){
  setStatus('Loading learned-impact kernels (Slang→WGSL)…');
  const {createLearnedImpact}=await import('./learned-impact.mjs');
  learned=await createLearnedImpact(device,{status:setStatus});
  learnedMs=await learned.run();
 }
 const st=sim.stats();
 $('fragments').textContent=st.chunks;$('hulls').textContent=scene.colliderCount;$('bonds').textContent=st.bonds;$('release').textContent='Start simulation';
 $('fire').hidden=!scene.rowShot||!!towers;$('volley').hidden=!towers;$('topple').hidden=!towers;$('meteor').hidden=towers!=='meteor';$('awake').textContent='0';$('contacts').textContent='0';
 // The scene's suggested settings (the tower block's), applied only when asked.
 $('suggested').hidden=!Object.keys(sim.suggested??{}).length;
 const shots=sim.packShots;
 $('pack-shot').replaceChildren(...shots.map((s,i)=>new Option(`${s.label} · ${(s.mass/1000).toFixed(1)} t at ${s.speed.toFixed(1)} m/s`,i)));
 $('pack-shot-row').hidden=!shots.length;$('pack-shot').title=shots.map(s=>`${s.label}: ${s.note}`).join('\n');
 if(urlShot!==null){const i=shots.findIndex(s=>s.label===urlShot||s.label===`${urlShot}x`);
  if(i<0)throw new Error(`No authored shot ${urlShot}: ${shots.map(s=>s.label).join(', ')}`);
  // The authored shot at its launch time (the source's settle), on World's schedule.
  $('pack-shot').value=String(i);sim.queue({pack:i},Math.max(0,Math.round(shots[i].launchAt*stepHz())-sim.steps));urlShot=null;go()}
 const what=towers?`4 towers and the city · `:scene.copies>1?`${scene.copies} copies · `:'';
 const source=procedural==='pack'?`scene pack · ${scene.pieces} nodes`:procedural?'procedural cuboid houses (box colliders)':`${(asset.bytes/1048576).toFixed(2)} MB · ${asset.materialCount} materials`;
 const learnedNote=learned?` · learned-impact ${learnedMs.toFixed(2)} ms/fwd (${learned.graph.nNodes} chunks, ${learned.params} weights, compute only)`:'';
 setStatus(`${PROFILE_NOTES[bondMode()]??PROFILE_NOTES[labBonds()]??labBonds()} · ${stepHz()} Hz · ${what}${st.chunks} chunks · ${st.bonds} bonds · ${source}${learnedNote} · ${towers?'fire the volley or click to shoot':scene.rowShot?'fire the ball through the row or click to shoot':'click the model to shoot'}`);
}
// blast_parity: its own engine (stress-web), one structure, labelled as such.
async function loadStress(level){
 const cuboid=procedural==='cuboid',pack=procedural==='pack';
 if(!cuboid&&!pack&&(procedural||!assetBytes))throw new Error('blast_parity runs a fracture GLB, a scene pack or the cuboid houses: pick one of those scenes.');
 learned?.dispose();learned=null;learnedMs=0;
 renderer?.dispose();renderer=null;sim?.dispose();sim=null;released=false;
 $('view').hidden=true;$('stress-view').hidden=false;
 setStatus('Loading blast_parity (its own engine: Rust and Slang on WebGPU)…');
 try{if(!stress){const {createStressMode}=await import('./stress-mode.mjs');stress=await createStressMode($('stress-view'))}}
 catch(e){$('view').hidden=false;$('stress-view').hidden=true;throw new Error(`blast_parity unavailable: ${e?.message??e}. Rebuild it (README, Bond modes).`)}
 stressOn=true;
 stress.viewMode=cameraMode();stress.setProfile('blast_parity');stress.setPhysicsHz(physicsHz());stress.setCrush(0);stress.setBondStrengthScale(strengthScale());stress.setShotRadiusScale(radiusScale());applyView();
 if(cuboid)await stress.loadCuboid(layout);else if(pack)await stress.loadPack(packBytes);else await stress.load(assetBytes,level);
 await stress.frame(0);applyBall();showBallRadius();
 const st=stress.stats;stressHud=stressHudReset(performance.now());
 $('fragments').textContent=st.chunks;$('bonds').textContent=st.bonds;$('hulls').textContent=st.hulls;$('awake').textContent='0';$('contacts').textContent='—';
 $('release').textContent='Start simulation';for(const id of ['fire','volley','meteor','topple','suggested','pack-shot-row'])$(id).hidden=true;
 setStatus(`blast_parity (a separate engine, not World: it waits on the GPU every frame) · ${st.chunks} chunks · ${st.bonds} bonds · ${Math.round(st.frameHz)} Hz · one structure${stressWarning(st)} · click the model to shoot`);
}
async function importBytes(bytes,name,sampleKey="",entry={}){
 assetBytes=bytes;layout=entry.layout??null;maxHullVertices=entry.maxHullVertices;procedural=null;towers=null;
 const start=performance.now(),next=parseFractureGlb(bytes);const ms=performance.now()-start;
 asset?.handle.free();asset=next;$('scene').value=sampleKey;
 $('level').replaceChildren(...asset.levels.map(l=>new Option(`Level ${l} · ${asset.levelFragments(l)} fragments`,l)));$('filename').textContent=name;
 const level=asset.initialLevel(entry.level);$('level').value=String(level);
 await loadLevel(level);window.__FRACTURE_DEMO__.parseMs=ms;
}
// The Scene menu is the shared catalogue (engine-fracture samples.rs), as in the Metal viewer.
const samples=Object.fromEntries(SAMPLES.map(s=>[s.key,{...s,procedural:s.file?null:'cuboid'}]));
const optgroup=(label,options)=>{const g=document.createElement('optgroup');g.label=label;g.append(...options);return g};
const sceneGroups=[...new Set(SAMPLES.map(s=>s.group))].map(label=>optgroup(label,SAMPLES.filter(s=>s.group===label).map(s=>new Option(s.menu,s.key,s.key==='house',s.key==='house'))));
// Generated in Rust (engine-fracture glass_tower.rs): a pack built on the spot, no file.
samples['glass-tower']={key:'glass-tower',label:'glass tower and pavilion',generated:glassTowerPack};
$('scene').replaceChildren(...sceneGroups,optgroup('Generated',[new Option('Glass tower · 10 storeys of shattering glass, a pavilion beside it','glass-tower')]),optgroup('Your file',[Object.assign(new Option('Custom GLB (choose or drop one below)',''),{disabled:true})]));
// The punch-through wall (ScenePack v3 with crush blocks): chunk crushing's showcase.
samples['crush-wall']={key:'crush-wall',label:'punch-through wall',pack:'./packs/crush-wall.json',nodes:216,bonds:464};
$('scene').insertBefore(optgroup('Chunk crushing (ScenePack v3)',[new Option('Punch-through wall · 12×8×2 blocks','crush-wall')]),$('scene').lastElementChild);
// vibe-city /destructible-stress presets, generated into ../generated/vibe-city (optional).
const VIBE_CITY='../generated/vibe-city/';
try{const index=await (await fetch(`${VIBE_CITY}index.json`)).json();
 const group=document.createElement('optgroup');group.label='Vibe City presets (generated scene packs)';
 const tower=index.presets.find(p=>p.id==='fracturedTower'&&!p.failed&&!p.pending);
 if(tower){
  // The 2×2 tower block of the fractured tower, its city's cars and pedestrians.
  samples['vc-towers']={key:'vc-towers',label:'fractured towers block',pack:`${VIBE_CITY}fracturedTower.json`,nodes:tower.nodes*4,bonds:tower.bonds*4,towers:'towers'};
  group.append(new Option(`Fractured towers · 2×2 city block · ${tower.nodes*4} nodes (CCD volley)`,'vc-towers'));
  samples['vc-meteor']={...samples['vc-towers'],key:'vc-meteor',label:'meteor strike',towers:'meteor'};
  group.append(new Option(`Meteor strike · fractured towers, cars, pedestrians`,'vc-meteor'));
 }
 for(const p of index.presets.filter(p=>!p.failed&&!p.pending)){
  const key=`vc:${p.id}`;samples[key]={key,label:p.id,pack:`${VIBE_CITY}${p.id}.json`,nodes:p.nodes,bonds:p.bonds};
  group.append(new Option(`${p.id} · ${p.nodes} nodes`,key));
 }
 if(group.children.length)$('scene').insertBefore(group,$('scene').lastElementChild);
}catch{}
// The reference destruction rigs (packs/index.json): scene packs with authored shots.
try{const index=await (await fetch('./packs/index.json')).json();
 const group=document.createElement('optgroup');group.label='Reference rigs (scene packs, authored shots)';
 for(const p of index.packs){const key=`ref:${p.key}`;samples[key]={key,label:p.title,pack:`./packs/${p.file}`,nodes:p.nodes,bonds:p.bonds,reference:true,shots:p.shots};
  group.append(new Option(`${p.key} · ${p.nodes} chunks · ${p.bonds} bonds`,key))}
 if(group.children.length)$('scene').insertBefore(group,$('scene').lastElementChild);
}catch{}
// URL: ?pack=crush-wall&shot=1.4x[&bonds=<model>][&camera=...] picks a reference rig and
// fires the authored shot at its launch time; ?gputime=phases times each step's phases.
const query=new URLSearchParams(location.search);
const GPU_TIMING=query.get('gputime')==='phases'?'phases':'step';
let urlShot=null;
{const pack=query.get('pack'),bonds=query.get('bonds');
 if(bonds&&$('bond-mode').querySelector(`option[value="${bonds}"]`))$('bond-mode').value=bonds;
 if(pack&&samples[`ref:${pack}`]){$('scene').value=`ref:${pack}`;if(!bonds)$('bond-mode').value='threshold';urlShot=query.get('shot')}}
$('fire-pack-shot').onclick=()=>{if(sim){sim.queue({pack:Number($('pack-shot').value)});go()}};
const sample=(key=$('scene').value||'house')=>serializeLoad(async()=>{
 const entry=samples[key];if(!entry)throw new Error('Unknown bundled scene');
 if(entry.pack||entry.generated){
  if(entry.generated)packBytes=entry.generated().buffer;
  else{
   setStatus(`Fetching ${entry.label}…`);
   const response=await fetch(entry.pack);
   if(!response.ok)throw new Error(entry.reference?`Scene pack missing: ${entry.pack}`:'Scene pack missing: run bun run export.ts in engine/scenes/vibe-city.');
   packBytes=await response.arrayBuffer();
  }
  // Selecting a scene never changes the bond model or the settings: the tower block's
  // suggested ones are a button (Use the scene's suggested settings).
  assetBytes=null;layout=null;procedural='pack';towers=entry.towers??null;asset?.handle.free();asset=null;$('scene').value=key;$('filename').textContent=entry.pack?entry.pack.split('/').pop():'generated in Rust (glass_tower.rs)';
  $('level').replaceChildren(new Option(entry.nodes!=null?`Scene pack · ${entry.nodes} nodes · ${entry.bonds} bonds`:'Scene pack · generated',0));$('level').value='0';
  await loadLevel(0);return;
 }
 if(entry.procedural){
  towers=null;layout=entry.layout??null;procedural=entry.procedural;assetBytes=null;asset?.handle.free();asset=null;$('scene').value=key;$('filename').textContent='Procedural cuboid house';
  $('level').replaceChildren(new Option('Cuboids · box colliders only',0));$('level').value='0';
  await loadLevel(0);return;
 }
 towers=null;
 setStatus(`Fetching baked ${entry.label}…`);
 const response=await fetch(`./assets/${entry.file}`);
 if(!response.ok)throw new Error('Bundled asset missing. Follow README.md to export it.');
 await importBytes(await response.arrayBuffer(),entry.file,key,entry);
});
$('sample').onclick=()=>sample();
// Actions: each queues its plan on World (engine-world `LabAction`); Rust fires every shot
// at its step.
const act=action=>{if(sim){try{applyBall();sim.queue(action);go()}catch(e){fail(e)}}};
$('fire').onclick=()=>act('row');
$('volley').onclick=()=>act('volley');
$('meteor').onclick=()=>act('meteor');
$('topple').onclick=()=>act('topple');
// The scene's suggested settings (the tower block's D6 bonds at 0.2×, CCD balls at 16×),
// applied to the panel only now, then the scene reloads.
$('suggested').onclick=()=>{const s=sim?.suggested;if(!s)return;
 if(s.bonds)$('bond-mode').value=s.bonds==='blast_gpu'?'blast':s.bonds;
 if(s.strengthScale)$('bond-strength').value=String(Math.round(Math.log2(s.strengthScale)));
 if(s.ballSpeedScale)$('ball-speed').value=String(Math.round(Math.log2(s.ballSpeedScale)));
 if(s.ccd!==undefined)$('ball-ccd').checked=s.ccd;
 $('bond-strength-value').textContent=scaleLabel(strengthScale());showFractureSwitches();
 serializeLoad(()=>loadLevel(Number($('level').value)))};
$('scene').onchange=()=>sample($('scene').value);
$('file').onchange=()=>{const f=$('file').files[0];if(f)serializeLoad(async()=>importBytes(await f.arrayBuffer(),f.name))};
$('level').onchange=()=>serializeLoad(()=>loadLevel(Number($('level').value)));
$('view-mode').onchange=()=>{if(renderer)renderer.colorMode=viewMode();applyView()};
$('show-bonds').onchange=applyView;
$('stress-scale').onchange=applyView;
const reload=()=>serializeLoad(()=>loadLevel(Number($('level').value)));
$('strike-prediction').onchange=reload;
$('contact-fracture').onchange=reload;
$('impact-response').onchange=()=>{$('impact-scale').disabled=$('impact-response').value==='off';reload()};
$('impact-scale').onchange=reload;
// A chosen strike rule only acts on predicted strikes: picking one turns prediction on.
$('strike-rule').onchange=()=>{$('strike-radius').disabled=$('strike-rule').value!=='radial';if($('strike-rule').value)$('strike-prediction').checked=true;reload()};
$('strike-radius').onchange=reload;
$('support-loss').onchange=reload;
$('ball-mass').oninput=applyBall;
$('ball-speed').oninput=applyBall;
$('ball-ccd').onchange=applyBall;
$('ball-radius').oninput=()=>{$('ball-radius-value').textContent=scaleLabel(radiusScale())};
$('ball-radius').onchange=reload;
$('bond-strength').oninput=()=>{$('bond-strength-value').textContent=scaleLabel(strengthScale())};
$('bond-strength').onchange=()=>{if(bondMode()!=='none')reload()};
$('bond-mode').onchange=()=>{showFractureSwitches();reload()};
$('crush').onchange=reload;
$('physics-hz').onchange=reload;
$('release').onclick=()=>{if(stressOn&&stress){stress.running=!stress.running;$('release').textContent=stress.running?'Pause':'Resume';return}released=true;running=!running;$('release').textContent=running?'Pause':'Resume';accumulator=0;rates.reset()};
// Reset: the bond models restore the placed scene; the stress models place it again.
$('reset').onclick=()=>serializeLoad(async()=>{if(stressOn||sim?.poseDraw){await loadLevel(Number($('level').value));return}
 sim.clearQueue();sim.reset();applyGuide();resetTrails();released=false;$('release').textContent='Start simulation';setStatus('Reset · click the model to shoot')});
$('metadata').onclick=()=>{if(!asset)return;const url=URL.createObjectURL(new Blob([asset.metadataJson()],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='fracture-metadata.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
for(const type of ['dragenter','dragover'])window.addEventListener(type,e=>{e.preventDefault();document.body.classList.add('drop')});
window.addEventListener('dragleave',()=>document.body.classList.remove('drop'));
window.addEventListener('drop',e=>{e.preventDefault();document.body.classList.remove('drop');const f=e.dataTransfer.files[0];if(f&&device)serializeLoad(async()=>importBytes(await f.arrayBuffer(),f.name))});
// The status line while it runs: World's numbers.
const runStatus=()=>{const st=sim.stats();
 return `${st.shots} shots · ${st.steps} steps · ${st.bodies} bodies · ${st.bonds-st.bondsIntact} of ${st.bonds} bonds broken${st.crushed?` · ${st.crushed} crushed`:''} · ${st.awake} awake · ${st.contacts} contacts · ${(st.steps/stepHz()).toFixed(2)} s`};
// The numbers a frame-locked recording reads (record-reactions.mjs, stress-modes-check.mjs).
const runStats=()=>{const st=sim.stats();return {frames:st.steps,shots:st.shots,bodies:st.bodies,brokenBonds:st.bonds-st.bondsIntact,crush:crushMode(),crushed:st.crushed,chunks:st.chunks,bonds:st.bonds,simSeconds:st.steps*sim.dt,frameHz:stepHz(),gpuStepMs:st.gpuStepMs,gpuDrawMs:renderer?.gpuFrameMs,awake:st.awake}};
let lockedBehind=0;
window.__FRACTURE_DEMO__={
 // Frame-locked capture: ready once loaded; each advance runs one display frame of `ms` (the
 // physics frames that fell due, then the draw).
 get stressReady(){return !loading&&(stressOn&&!!stress||!!sim)},
 async stressAdvance(ms=1000/60){stressCapture=true;await frame;
  if(!stressOn){
   lockedBehind+=ms/1000;const n=Math.floor(lockedBehind*stepHz()+1e-9);lockedBehind-=n/stepHz();
   if(running&&n)await sim.step(n);renderer.draw($('show-bonds').checked);setStatus(runStatus());return runStats()}
  await stress.frame(ms);const st=stress.stats;setStatus(stressStatus(st));return st},
 stressRelease(){stressCapture=false;last=0},
 stressOrbit(dx,dy){if(!stressOn)renderer?.orbit(dx,dy);else stress.orbit(dx,dy)},stressZoom(delta){if(!stressOn)renderer?.zoom(delta);else stress.zoom(delta)},
 get status(){return $('status').textContent},
 // The page's own shot (a click's path).
 shoot:ray=>shoot(ray),
 setRunning(on){released=true;running=!!on;accumulator=0;$('release').textContent=running?'Pause':'Resume'},
 get running(){return running},
 // Shots queued on World and not yet fired (the volley in flight).
 get volleyActive(){return !!sim&&sim.queued>0},get ready(){return !loading&&!!sim},get device(){return device},get asset(){return asset},get scene(){return scene},get simulation(){return sim},get renderer(){return renderer},get rates(){return currentRates},
 // A ball along a ray, fired by the next step; returns its scene body.
 async shootRay(origin,direction){running=false;await frame;placeCrowd({origin,direction});return sim.shoot(origin,direction)},
 get bystanders(){return bystanders},
 async fireRow(){running=false;await frame;const id=scene.projectileStart+(sim.shots+sim.queued)%Math.max(1,scene.projectileCount);sim.queue('row');return id},
 // Repeatable row benchmark: fire, then every `every` steps record wall ms per step, the
 // ball and the counters.
 async benchmarkRow({steps=600,every=60}={}){
  running=false;await frame;const id=await this.fireRow(),log=[];
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
 async snapshot(){await frame;return sim.snapshot()},reset:()=>serializeLoad(async()=>{if(sim.poseDraw)await loadLevel(Number($('level').value));else{sim.clearQueue();sim.reset();applyGuide()}}),load:sample,selectLevel:level=>serializeLoad(async()=>{$("level").value=String(level);await loadLevel(level)})};
// page-bench.mjs: frame-locked steps.
window.__step=n=>window.__FRACTURE_DEMO__.stepFrames(n);
// Checks and benches (the before/after comparisons): steps, the breaks since the shot.
let shotStep=0;
window.__lab={
 async step(n){await sim.step(n);return sim.counters},async one(){await sim.step(1);return sim.counters},
 broken(){const st=sim.stats(),f=sim.breaks.firstStep;return {broken:st.bonds-st.bondsIntact,actors:st.bodies,crushed:st.crushed,firstStep:f===null?null:f-shotStep+1}},
 async fireRay(){const d=window.__FRACTURE_DEMO__,c=scene.center,e=Math.max(scene.extent,1);const id=await d.shootRay([c[0]+0.13,c[1],c[2]+3*e],[0,0,-1]);shotStep=sim.steps;return id},
 reset(){return window.__FRACTURE_DEMO__.reset()},
};
// blast_parity's HUD (its own engine): physics frames and display frames per second, GPU
// time per physics frame and per draw, the display's refresh, the uncapped estimate.
const stressWarning=st=>st.warnings?.length?` · ⚠ ${st.warnings.join('; ')}`:'';
const stressStatus=st=>`blast_parity · ${st.shots} shots · ${st.frames} frames · ${st.bodies??'—'} bodies · ${st.brokenBonds} bonds broken · ${st.simSeconds.toFixed(2)} s${stressWarning(st)}`;
function stressHudReset(now){return {t:now,frames:0,renders:0,gaps:[],cpuMs:0,last:now,stepMs:stressHud?.stepMs}}
function stressHudShow(st,s){
 const h=stressHud,tps=h.frames/s,fps=h.renders/s,hz=st.frameHz;
 $('tps').textContent=tps.toFixed(0);$('fps').textContent=fps.toFixed(0);$('sim-speed').textContent=`${(tps/hz).toFixed(2)}×`;
 const gaps=h.gaps.sort((a,b)=>a-b).filter((g,_,a)=>g<=1.5*a[0]),gap=gaps[gaps.length>>1];
 const display=gap?Math.round(1000/gap):null,cpu=h.renders?h.cpuMs/h.renders:0;
 const phys=st.gpuStepMs,draw=st.gpuDrawMs;
 $('step-ms').textContent=phys?`${phys.toFixed(2)} ms GPU/tick · max ${Math.floor(1000/phys)} TPS`:h.stepMs?`${h.stepMs.toFixed(2)} ms CPU/tick (no GPU timestamps)`:`target ${Math.round(hz)} TPS`;
 $('frame-ms').textContent=draw?`${draw.toFixed(2)} ms GPU/draw${display?` · display ${display} Hz`:''}`:display?`display ${display} Hz`:'frames / second';
 if(!phys||!draw){$('timing').textContent=`CPU ${cpu.toFixed(2)} ms per display frame. No GPU timestamps here: the GPU's time is not measured.`;return}
 const physBusy=hz*phys,gpuFps=physBusy<1000?(1000-physBusy)/draw:0,cpuFps=cpu>0?1000/cpu:Infinity,uncapped=Math.min(gpuFps,cpuFps);
 $('timing').textContent=physBusy>=1000?
  `GPU-bound: real-time physics needs ${physBusy.toFixed(0)} ms of GPU each second (${Math.round(hz)} ticks × ${phys.toFixed(2)} ms), so it runs at most ${(1000/physBusy).toFixed(2)}× real time. Draw ${draw.toFixed(2)} ms GPU; CPU ${cpu.toFixed(2)} ms per display frame.`:
  `Each second at real time: physics ${physBusy.toFixed(0)} ms GPU (${Math.round(hz)} ticks × ${phys.toFixed(2)} ms), draws ${draw.toFixed(2)} ms GPU each in the rest. Uncapped ≈ ${Math.floor(uncapped)} FPS (${gpuFps<=cpuFps?'GPU':'CPU'}-bound; CPU ${cpu.toFixed(2)} ms per display frame)${display?`; the display caps it at ${display} Hz`:''}.`;
}
function stressTick(now,elapsedMs){
 frame=(async()=>{try{
  const before=stress.stats.frames??0,t0=performance.now();await stress.frame(elapsedMs);const cpuMs=performance.now()-t0;const st=stress.stats;
  const h=stressHud;h.frames+=(st.frames??0)-before;h.renders++;h.cpuMs+=cpuMs;if(now>h.last)h.gaps.push(now-h.last);h.last=now;
  if(st.stepMs)h.stepMs=st.stepMs;
  if(now-h.t>=1000){stressHudShow(st,(now-h.t)/1000);stressHud=stressHudReset(now)}
  $('awake').textContent=st.moving??st.awake??'—';
  if(stress.running||st.shots){$('release').textContent=stress.running?'Pause':'Resume';setStatus(stressStatus(st))}
 }catch(e){fail(e)}})();
 frame.finally(()=>requestAnimationFrame(tick));
}
// Frame-locked capture (record-reactions.mjs): the page steps only when asked.
let stressCapture=false;
function tick(now){
 if(stressCapture){requestAnimationFrame(tick);return}
 if(stressOn&&!loading&&stress){const ms=last?Math.min(250,now-last):0;last=now;return stressTick(now,ms)}
 const elapsed=last?Math.min(.04,(now-last)/1000):0;last=now;
 if(!loading&&renderer){frame=(async()=>{try{
  let ticks=0;
  if(running){
   accumulator=Math.min(accumulator+elapsed,MAX_TICKS_PER_FRAME/stepHz());
   // The due ticks World can take without waiting; the rest stay due (never dropped).
   const due=Math.min(MAX_TICKS_PER_FRAME,Math.floor(accumulator*stepHz()+1e-9));
   ticks=sim.tick(due);accumulator-=ticks/stepHz();
   if(!document.hidden)for(let i=0;i<ticks;i++)rates.step(sim.timing);
  }else sim.poll();
  if(learned&&!learnedBusy){learnedBusy=true;learned.run().then(ms=>{learnedMs=ms;learnedBusy=false}).catch(e=>{learnedBusy=false;fail(e)})}
  updateTrails();renderer.draw($('show-bonds').checked);
  if(running){const c=sim.counters;$("awake").textContent=c.awake;$("contacts").textContent=c.contact;updateNear(now);setStatus(runStatus())}
  if(!document.hidden)rates.frame();
 }catch(e){fail(e)}})();frame.finally(()=>requestAnimationFrame(tick))}else requestAnimationFrame(tick);
}
try{
 if(!navigator.gpu)throw new Error('WebGPU is unavailable. Open this localhost page in a WebGPU-enabled browser.');
 // World's device (lab-world.mjs): the physics and this page's renderer share it.
 device=await labDevice();
 device.addEventListener('uncapturederror',e=>fail(e.error));device.lost.then(info=>fail(new Error(`GPU device lost: ${info.message}. Reload this page.`)));
 // ?scene=<key>&bonds=<mode>&crush=<0|1|2>&hz=<60|120>&view=<0..3> preselect the panel.
 const q=new URLSearchParams(location.search);
 if(q.has('scene')&&samples[q.get('scene')])$('scene').value=q.get('scene');
 if(q.has('bonds'))$('bond-mode').value=q.get('bonds');
 if(q.has('view'))$('view-mode').value=q.get('view');
 if(q.has('stressScale'))$('stress-scale').value=q.get('stressScale');
 if(q.has('strike'))$('strike-rule').value=q.get('strike');
 if(q.has('radius'))$('strike-radius').value=q.get('radius');
 if(q.has('response'))$('impact-response').value=q.get('response');
 if(q.has('responseScale'))$('impact-scale').value=q.get('responseScale');
 if(q.has('support'))$('support-loss').checked=q.get('support')==='1';
 if(q.has('crush'))$('crush').value=q.get('crush');
 if(q.has('hz'))$('physics-hz').value=q.get('hz');
 showFractureSwitches();
 await sample();requestAnimationFrame(tick);
}catch(e){fail(e);controls(false)}
