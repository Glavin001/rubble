// Physics-only benchmark page (no canvas, no renderer): the lab's scenes on rubble's World
// (lab-world.mjs). headless.mjs opens it in headless Chrome and reads window.__result.
// Query parameters:
//   scene    grid | suburb | house | cuboidGrid | cuboids   (default grid)
//   shot     row (the ball through the row) | none          (default row)
//   warm     ticks before measuring                          (default 420)
//   measure  ticks timed back to back, pipelined             (default 120)
//   profile  World's per-phase GPU times of the last measured tick, 0/1 (default 1)
//   trace    log counters every N warm ticks, 0 = off        (default 0)
//   budget   JSON merged into scene.capacity (its contact and candidate budgets size World)
//   awake    1: every dynamic body starts awake (scenes start asleep)
//   params   JSON merged into scene.params, e.g. {"flags":97} (World's solver; measurement only)
//   bonds    1: the lab's breakable bonds (experimental), preloaded as in the lab
//   level    fracture level override for the GLB scenes
//   columns, rows   grid layout override (grid scenes only)
//   scene=pairs a=<shape> b=<shape> count=N reuse=0|1   collision-cost benchmark
//            (engine-fracture pairs.rs): N static bases of shape a, each with shape b resting on it.
//   staticUnhit  1: houses outside the shot's column become static (immovable).
//            Behaviour-changing; only to estimate compound houses that stay intact.
import {parseFractureGlb,buildScene,buildCuboidHouse,buildPairScene,layoutScene,withoutJoints,withCapacity,withParams,wakeAll,sceneHandle,SAMPLES} from './wasm.mjs';
import {createLabFromScene,settleScene} from './lab-world.mjs';

// The lab's bundled scenes (engine-fracture samples.rs); bonds only with bonds=1.
async function buildNamed(name,bonds,layout,levelOverride=null,paramsOverride={}){
 const entry=SAMPLES.find(s=>s.key===name);if(!entry)throw new Error(`Unknown scene ${name}`);
 const lay=layout&&entry.layout&&!entry.layout.suburb?{...entry.layout,...layout}:entry.layout;
 if(!entry.file)return layoutScene(buildCuboidHouse(),lay,true);
 const response=await fetch(`./assets/${entry.file}`);if(!response.ok)throw new Error(`Missing asset ${entry.file}`);
 const asset=parseFractureGlb(await response.arrayBuffer());
 const level=levelOverride??asset.initialLevel(entry.level);
 const built=withParams(buildScene(asset,level,{maxHullVertices:entry.maxHullVertices}),paramsOverride);
 return layoutScene(bonds?await settleScene(built):withoutJoints(built),lay,false);
}

async function run(){
 const q=new URLSearchParams(location.search),num=(k,d)=>q.has(k)?Number(q.get(k)):d;
 const name=q.get('scene')??'grid',warm=num('warm',420),measure=num('measure',120),trace=num('trace',0);
 const budget=q.has('budget')?JSON.parse(q.get('budget')):{};
 for(const k of ['sleepdiag','sleeptrace','wakewatch','sleepwatch','lockstep','timeline'])
  if(q.has(k))throw new Error(`${k}: a diagnostic of the old JavaScript host's plan (contact rows, kernel profiles), gone with it (web-rust-host.md)`);
 const layout=q.has('columns')||q.has('rows')?{...(q.has('columns')?{columns:num('columns',7)}:{}),...(q.has('rows')?{rows:num('rows',7)}:{})}:null;
 const paramsOverride=q.has('params')?JSON.parse(q.get('params')):{};
 let built=name==='pairs'?buildPairScene({a:q.get('a')??'box',b:q.get('b')??'box',count:num('count',4096)}):await buildNamed(name,num('bonds',0)===1,layout,q.has('level')?num('level',0):null,paramsOverride);
 if(name==='pairs'&&num('reuse',0)===1)built=withParams(built,{flags:built.params.flags|1});
 let scene=withParams(withCapacity(built,budget),paramsOverride);
 if(num('awake',0)===1)scene=wakeAll(scene);
 const frozen=num('staticUnhit',0)===1?sceneHandle(scene).freezeUnhitCopies():0;
 const sim=await createLabFromScene(scene,{bonds:scene.jointCount?'d6':'none',gpuTiming:num('profile',1)?'phases':'step'});
 let gpuError=null;sim.device.addEventListener('uncapturederror',e=>{gpuError??=String(e.error.message)});
 if((q.get('shot')??'row')==='row'&&scene.rowShot)sim.queue('row');
 const log=[],t0=performance.now();
 for(let k=0;k<warm;k+=4){await sim.step(Math.min(4,warm-k));
  if(trace&&Math.floor((k+4)/trace)>Math.floor(k/trace)){const c=sim.counters;log.push({tick:sim.steps,awake:c.awake,contacts:c.contact,candidates:c.candidate})}}
 const warmMs=performance.now()-t0;
 const t1=performance.now();await sim.step(measure);
 const msPerTick=(performance.now()-t1)/Math.max(1,measure),counters={...sim.counters};
 const g=sim.gpuTime,phases=g?.segments?Object.entries(g.segments.reduce((a,[k,v])=>(a[k]=(a[k]??0)+v,a),{})).sort((a,b)=>b[1]-a[1]).map(([k,v])=>[k,+v.toFixed(3)]):null;
 const result={scene:name,frozenHouses:frozen,bodies:scene.bodyCount,colliders:scene.colliderCount,joints:scene.jointCount,budget,warm,measure,
  msPerTick:+msPerTick.toFixed(3),ticksPerSecond:+(1000/msPerTick).toFixed(1),warmMs:Math.round(warmMs),counters,gpuMs:g?+g.totalMs.toFixed(3):null,phases,trace:log,gpuError};
 sim.dispose();return result;
}
run().then(r=>{window.__result=r},e=>{window.__result={error:String(e?.stack??e)}});
