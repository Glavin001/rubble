// Physics-only benchmark page (no canvas, no renderer). headless.mjs opens it
// in headless Chrome and reads window.__result. Query parameters:
//   scene    grid | suburb | house | cuboidGrid | cuboids   (default grid)
//   shot     row (the ball through the row) | none          (default row)
//   warm     ticks before measuring                          (default 420)
//   measure  ticks timed back to back, overlapped            (default 120)
//   profile  per-kernel GPU timestamps after measuring, 0/1  (default 1)
//   trace    log counters every N warm ticks, 0 = off        (default 0)
//   budget   JSON merged into scene.capacity, e.g. {"deterministic":true}
//   bonds    1: the lab's breakable bonds (experimental), preloaded as in the lab
//   columns, rows   grid layout override (grid scenes only)
//   scene=pairs a=<shape> b=<shape> count=N reuse=0|1   collision-cost benchmark
//            (shape-bench.mjs): N static bases of shape a, each with shape b resting on it.
//   staticUnhit  1: houses outside the shot's column become static (immovable).
//            Behaviour-changing; only to estimate compound houses that stay intact.
import {parseFractureGlb} from './asset.mjs';
import {buildScene,buildGrid,buildSuburb} from './geometry.mjs';
import {buildCuboidHouse} from './cuboid-house.mjs';
import {createSimulation} from './simulation.mjs';
import {preloadBonds} from './bonds.mjs';
import {buildPairScene} from './shape-bench.mjs';

// Same bundled scenes as the lab (main.mjs samples), without bond preloading.
const SCENES={house:{file:'veneer-house.glb'},grid:{file:'veneer-house.glb',layout:{columns:7,rows:7},level:2},
 suburb:{file:'veneer-house.glb',layout:{suburb:true},level:2},
 cuboidGrid:{procedural:true,layout:{columns:7,rows:7}},cuboids:{procedural:true,layout:{suburb:true}}};

async function buildNamed(name,device,bonds,layout){
 const known=SCENES[name];if(!known)throw new Error(`Unknown scene ${name}`);
 const entry=layout&&known.layout&&!known.layout.suburb?{...known,layout:{...known.layout,...layout}}:known;
 if(entry.procedural){
  const one=buildCuboidHouse(),scene=(entry.layout.suburb?buildSuburb:buildGrid)(null,0,{...entry.layout,base:one});
  return {...scene,capacity:{...scene.capacity,candidatesPerCollider:12,contactsPerCollider:24}};
 }
 const response=await fetch(`./assets/${entry.file}`);if(!response.ok)throw new Error(`Missing asset ${entry.file}`);
 const asset=parseFractureGlb(await response.arrayBuffer());
 const level=entry.level??Math.min(...asset.physics.fragments.map(f=>f.level));
 const built=buildScene(asset,level),one=bonds?await preloadBonds(device,built):{...built,joints:[]};
 return entry.layout?(entry.layout.suburb?buildSuburb:buildGrid)(asset,level,{...entry.layout,base:one}):one;
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
 const built=name==='pairs'?buildPairScene({a:q.get('a')??'box',b:q.get('b')??'box',count:num('count',4096)}):await buildNamed(name,device,num('bonds',0)===1,layout);
 if(name==='pairs'&&num('reuse',0)===1)built.params={...built.params,flags:built.params.flags|1};
 const scene={...built,capacity:{...built.capacity,...budget}};
 let frozen=0;
 if(num('staticUnhit',0)===1&&scene.rowShot&&scene.instanceStride){
  const frags=scene.instanceStride,x=scene.rowShot.origin[0],r=scene.projectileRadius,ground=scene.bodies[0];
  scene.bodies=scene.bodies.slice();
  for(let k=0;k<scene.copies;k++){
   let lo=1/0,hi=-1/0;for(let b=1+k*frags;b<1+(k+1)*frags;b++){lo=Math.min(lo,scene.bodies[b].p[0]);hi=Math.max(hi,scene.bodies[b].p[0])}
   if(x+r>lo-0.5&&x-r<hi+0.5)continue;
   for(let b=1+k*frags;b<1+(k+1)*frags;b++)scene.bodies[b]={...scene.bodies[b],invMass:0,invI:[0,0,0],flags:ground.flags};
   frozen++;
  }
 }
 const sim=await createSimulation(device,scene);
 if((q.get('shot')??'row')==='row'&&scene.rowShot)sim.launch(scene.rowShot.origin,scene.rowShot.direction,{arc:false});
 const log=[],t0=performance.now();
 for(let k=0;k<warm;k+=4){await sim.step(Math.min(4,warm-k),{wait:false});
  if(trace&&Math.floor((k+4)/trace)>Math.floor(k/trace)){await sim.flush();const c=sim.counters;log.push({tick:sim.steps,awake:c.awake,contacts:c.contact,candidates:c.candidate})}}
 await sim.flush();const warmMs=performance.now()-t0;
 const t1=performance.now();for(let k=0;k<measure;k++)await sim.step(1,{wait:false});await sim.flush();
 const msPerTick=(performance.now()-t1)/Math.max(1,measure),counters={...sim.counters};
 let kernels=null;
 if(num('profile',1)){const agg={};for(let r=0;r<3;r++){const p=await sim.profileKernels();if(!p.supported)break;for(const [k,v] of Object.entries(p.kernels))agg[k]=(agg[k]??0)+v.ms/3}
  kernels=Object.fromEntries(Object.entries(agg).sort((a,b)=>b[1]-a[1]).map(([k,v])=>[k,+v.toFixed(3)]))}
 const result={scene:name,frozenHouses:frozen,bodies:scene.bodies.length,colliders:scene.colliders.length,joints:scene.joints?.length??0,budget,warm,measure,
  msPerTick:+msPerTick.toFixed(3),ticksPerSecond:+(1000/msPerTick).toFixed(1),warmMs:Math.round(warmMs),counters,kernels,trace:log,gpuError};
 sim.dispose();return result;
}
run().then(r=>{window.__result=r},e=>{window.__result={error:String(e?.stack??e)}});
