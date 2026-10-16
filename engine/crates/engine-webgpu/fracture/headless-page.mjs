// Physics-only benchmark page (no canvas, no renderer). headless.mjs opens it
// in headless Chrome and reads window.__result. Query parameters:
//   scene    grid | suburb | house | cuboidGrid | cuboids   (default grid)
//   shot     row (the ball through the row) | none          (default row)
//   warm     ticks before measuring                          (default 420)
//   measure  ticks timed back to back, overlapped            (default 120)
//   profile  per-kernel GPU timestamps after measuring, 0/1  (default 1)
//   trace    log counters every N warm ticks, 0 = off        (default 0)
//   budget   JSON merged into scene.capacity, e.g. {"deterministic":true}
//   awake    1: every dynamic body starts awake (scenes start asleep)
//   params   JSON merged into scene.params, e.g. {"flags":97} (measurement only)
//   sleepdiag=N [every=120]   why bodies stay awake (sleep-diag.mjs), N ticks after the shot
//   sleeptrace=T [ticks pick bodies]   per-tick trace of the worst movers from tick T
//   wakewatch=T [to]    bodies woken each tick and their awake partners
//   sleepwatch=T [to]   bodies that start each burst of motion between ticks T and to
//   bonds    1: the lab's breakable bonds (experimental), preloaded as in the lab
//   level    fracture level override for the GLB scenes
//   columns, rows   grid layout override (grid scenes only)
//   scene=pairs a=<shape> b=<shape> count=N reuse=0|1   collision-cost benchmark
//            (engine-fracture pairs.rs): N static bases of shape a, each with shape b resting on it.
//   staticUnhit  1: houses outside the shot's column become static (immovable).
//            Behaviour-changing; only to estimate compound houses that stay intact.
import {parseFractureGlb,buildScene,buildCuboidHouse,buildPairScene,layoutScene,withoutJoints,withCapacity,withParams,wakeAll,sceneHandle,hullView,hullBase,SAMPLES} from './wasm.mjs';
import {createSimulation} from './simulation.mjs';
import {preloadBonds} from './bonds.mjs';
import {runSleepDiag,runSleepTrace,runSleepWatch,runWakeWatch} from './sleep-diag.mjs';

// The lab's bundled scenes (engine-fracture samples.rs); bonds only with bonds=1.
async function buildNamed(name,device,bonds,layout,levelOverride=null,paramsOverride={}){
 const entry=SAMPLES.find(s=>s.key===name);if(!entry)throw new Error(`Unknown scene ${name}`);
 const lay=layout&&entry.layout&&!entry.layout.suburb?{...entry.layout,...layout}:entry.layout;
 if(!entry.file)return layoutScene(buildCuboidHouse(),lay,true);
 const response=await fetch(`./assets/${entry.file}`);if(!response.ok)throw new Error(`Missing asset ${entry.file}`);
 const asset=parseFractureGlb(await response.arrayBuffer());
 const level=levelOverride??asset.initialLevel(entry.level);
 const built=withParams(buildScene(asset,level,{maxHullVertices:entry.maxHullVertices}),paramsOverride);
 const scene=layoutScene(bonds?await preloadBonds(device,built):withoutJoints(built),lay,false);
 // For sleep-diag's per-fragment breakdown: copy 0's fragment -> body map and the materials.
 const byId=new Map(scene.selected.map(f=>[f,{id:scene.handle.bodyOfFragment(f),f:asset.handle.fragment(f)}]));
 return Object.assign(scene,{byId,library:asset.handle.library()});
}

async function run(){
 const q=new URLSearchParams(location.search),num=(k,d)=>q.has(k)?Number(q.get(k)):d;
 const name=q.get('scene')??'grid',warm=num('warm',420),measure=num('measure',120),trace=num('trace',0);
 const budget=q.has('budget')?JSON.parse(q.get('budget')):{};
 const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw new Error('No WebGPU adapter');
 const device=await adapter.requestDevice({requiredFeatures:adapter.features.has('timestamp-query')?['timestamp-query']:[],
  requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(16,adapter.limits.maxStorageBuffersPerShaderStage),
   maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize,maxBufferSize:adapter.limits.maxBufferSize}});
 let gpuError=null;device.addEventListener('uncapturederror',e=>{gpuError??=String(e.error.message)});
 const layout=q.has('columns')||q.has('rows')?{...(q.has('columns')?{columns:num('columns',7)}:{}),...(q.has('rows')?{rows:num('rows',7)}:{})}:null;
 const paramsOverride=q.has('params')?JSON.parse(q.get('params')):{};
 let built=name==='pairs'?buildPairScene({a:q.get('a')??'box',b:q.get('b')??'box',count:num('count',4096)}):await buildNamed(name,device,num('bonds',0)===1,layout,q.has('level')?num('level',0):null,paramsOverride);
 if(name==='pairs'&&num('reuse',0)===1)built=withParams(built,{flags:built.params.flags|1});
 let scene=withParams(withCapacity(built,budget),paramsOverride);
 if(num('awake',0)===1)scene=wakeAll(scene);
 const frozen=num('staticUnhit',0)===1?sceneHandle(scene).freezeUnhitCopies():0;
 // lockstep=N: N ticks of every optimization off vs on (both deterministic),
 // counts compared every tick and body words every 35; reports the first difference.
 if(num('lockstep',0)>0){
  const off={deterministic:true,movedOnlyBounds:false,awakeQueries:false,persistentRest:false,taggedWarmHash:false,awakeListPasses:false};
  const on={deterministic:true,movedOnlyBounds:true};
  const A=await createSimulation(device,withCapacity(scene,off)),B=await createSimulation(device,withCapacity(scene,on));
  if((q.get('shot')??'row')==='row'&&scene.rowShot)for(const S of [A,B])S.launch(scene.rowShot.origin,scene.rowShot.direction,{arc:false});
  let first=null,maxAwake=0,k=0;
  for(k=1;k<=num('lockstep',0)&&!first;k++){await A.step(1);await B.step(1);maxAwake=Math.max(maxAwake,A.counters.awake);
   const a=A.counters,b=B.counters;if(a.candidate!==b.candidate||a.contact!==b.contact||a.awake!==b.awake){first={tick:k,a,b};break}
   if(k%35===0){const [x,y]=await Promise.all([A,B].map(async S=>new Uint32Array((await S.snapshot()).buffer).slice()));
    for(let i=0;i<x.length;i++)if(x[i]!==y[i]){first={tick:k,word:i%28,body:Math.floor(i/28)};break}}}
  A.dispose();B.dispose();return {scene:name,lockstep:{ticks:k-1,firstDifference:first,maxAwake},gpuError};
 }
 const sim=await createSimulation(device,scene);
 if(num('sleeptrace',0)>0){
  if((q.get('shot')??'row')==='row'&&scene.rowShot)sim.launch(scene.rowShot.origin,scene.rowShot.direction,{arc:false});
  const report=await runSleepTrace(sim,scene,{at:num('sleeptrace',0),ticks:num('ticks',30),pick:num('pick',3),bodies:q.has('bodies')?q.get('bodies').split(',').map(Number):null,settling:num('settling',0)===1,hullView,hullBase});
  sim.dispose();return {scene:name,trace:report,gpuError};
 }
 // stepdebug=T [to]: run to T, then single steps printing the counters; on a
 // failure report bodies with non-finite or extreme state.
 if(num('stepdebug',0)>0){
  if((q.get('shot')??'row')==='row'&&scene.rowShot)sim.launch(scene.rowShot.origin,scene.rowShot.direction,{arc:false});
  const from=num('stepdebug',0),to=num('to',from+10),rows=[];let error=null;
  try{for(let t=0;t<from;t++)await sim.step(1)}catch(e){error='before: '+String(e)}
  for(let t=from;t<to&&!error;t++){try{await sim.step(1);rows.push({t:t+1,...sim.counters})}catch(e){error=String(e);rows.push({t:t+1,...sim.counters})}}
  const f=await sim.snapshot(),bad=[];let vmax=0,wmax=0;
  for(let i=0;i<scene.bodyCount;i++){const o=i*28,v=Math.hypot(f[o+12],f[o+13],f[o+14]),w=Math.hypot(f[o+16],f[o+17],f[o+18]);
   if(![f[o],f[o+1],f[o+2],v,w].every(Number.isFinite)&&bad.length<10)bad.push(i);if(v>vmax)vmax=v;if(w>wmax)wmax=w}
  sim.dispose();return {error,rows,bad,vmax,wmax,gpuError};
 }
 if(num('wakewatch',0)>0){
  if((q.get('shot')??'row')==='row'&&scene.rowShot)sim.launch(scene.rowShot.origin,scene.rowShot.direction,{arc:false});
  const report=await runWakeWatch(sim,scene,{from:num('wakewatch',0),to:num('to',1500)});
  sim.dispose();return {scene:name,wake:report,gpuError};
 }
 if(num('sleepwatch',0)>0){
  if((q.get('shot')??'row')==='row'&&scene.rowShot)sim.launch(scene.rowShot.origin,scene.rowShot.direction,{arc:false});
  const report=await runSleepWatch(sim,scene,{from:num('sleepwatch',0),to:num('to',2400)});
  sim.dispose();return {scene:name,watch:report,gpuError};
 }
 if(num('sleepdiag',0)>0){
  if((q.get('shot')??'row')==='row'&&scene.rowShot)sim.launch(scene.rowShot.origin,scene.rowShot.direction,{arc:false});
  const report=await runSleepDiag(sim,scene,{ticks:num('sleepdiag',0),every:num('every',120)});
  sim.dispose();return {scene:name,bodies:scene.bodyCount,params:scene.params,budget,sleepdiag:report,gpuError};
 }
 if((q.get('shot')??'row')==='row'&&scene.rowShot)sim.launch(scene.rowShot.origin,scene.rowShot.direction,{arc:false});
 const log=[],t0=performance.now();
 // timeline=N: every N ticks, one step timed in a single pass (true GPU ms)
 // and the next profiled per kernel, grouped by stage, with the counters.
 const timeline=num('timeline',0),stages=[];
 const stageOf=k=>/^(build_aabbs|broadphase|write_moved|rest_|write_finer)/.test(k)?'broadphase':
  /^(narrowphase|contact_hash|persist_contacts|update_refs|reuse_all)/.test(k)?'contacts':
  /^convex_/.test(k)?'hullCollision':/^(solve_primal)/.test(k)?'solverPrimal':/^(solve_dual)/.test(k)?'solverDual':
  /^(pack_solver|predict_bodies|extract_velocity)/.test(k)?'solverOther':
  /^(csr_|build_awake)/.test(k)?'contactGraph':/^(color_|init_colors|count_color|prefix_color|scatter_color|record_color|write_color)/.test(k)?'coloring':
  /^(sleep_|write_sleep)/.test(k)?'sleep':'other';
 if(timeline)for(let k=0;k<warm;){
  if(k%timeline===0){const w=await sim.profileStep({singlePass:true});const p=await sim.profileKernels();const st={};
   for(const [n,v] of Object.entries(p.kernels??{}))st[stageOf(n)]=+((st[stageOf(n)]??0)+v.ms).toFixed(3);
   const c=sim.counters;stages.push({tick:sim.steps,stepMs:+w.gpuMs.toFixed(2),awake:c.awake,contacts:c.contact,candidates:c.candidate,hullPairs:c.convexSmall+c.convexLarge,stages:st,kernels:Object.entries(p.kernels??{}).sort((a,b)=>b[1].ms-a[1].ms).slice(0,10).map(([n,v])=>n+" "+v.ms.toFixed(3)+" x"+v.count)});k+=2}
  else{await sim.step(1);k++}
 }
 else for(let k=0;k<warm;k+=4){await sim.step(Math.min(4,warm-k),{wait:false});
  if(trace&&Math.floor((k+4)/trace)>Math.floor(k/trace)){await sim.flush();const c=sim.counters;log.push({tick:sim.steps,awake:c.awake,contacts:c.contact,candidates:c.candidate})}}
 await sim.flush();const warmMs=performance.now()-t0;
 const t1=performance.now();for(let k=0;k<measure;k++)await sim.step(1,{wait:false});await sim.flush();
 const msPerTick=(performance.now()-t1)/Math.max(1,measure),counters={...sim.counters};
 let kernels=null;
 if(num('profile',1)){const agg={};for(let r=0;r<3;r++){const p=await sim.profileKernels();if(!p.supported)break;for(const [k,v] of Object.entries(p.kernels))agg[k]=(agg[k]??0)+v.ms/3}
  kernels=Object.fromEntries(Object.entries(agg).sort((a,b)=>b[1]-a[1]).map(([k,v])=>[k,+v.toFixed(3)]))}
 const result={scene:name,timeline:stages,frozenHouses:frozen,bodies:scene.bodyCount,colliders:scene.colliderCount,joints:scene.jointCount,budget,warm,measure,
  msPerTick:+msPerTick.toFixed(3),ticksPerSecond:+(1000/msPerTick).toFixed(1),warmMs:Math.round(warmMs),counters,kernels,trace:log,gpuError};
 sim.dispose();return result;
}
run().then(r=>{window.__result=r},e=>{window.__result={error:String(e?.stack??e)}});
