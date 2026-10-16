// Deterministic parity shot, the browser half of engine-viewer's
// fracture_compare example: the same scene (shared Rust code), budget, shot
// and tick count, and the same printout to diff against the Metal run.
//   (await import('./parity.mjs')).parityShot(__FRACTURE_DEMO__.device,{scene:'arch',level:0,ticks:240})
import {parseFractureGlb,buildScene,buildCuboidHouse,layoutScene,withoutJoints,withCapacity,SAMPLES} from './wasm.mjs';
import {createSimulation} from './simulation.mjs';

export async function parityShot(device,{scene:key='arch',level,ticks=240}={}){
 const entry=SAMPLES.find(s=>s.key===key);if(!entry)throw new Error(`Unknown scene ${key}`);
 let one;
 if(entry.file){
  const asset=parseFractureGlb(await (await fetch(`./assets/${entry.file}`)).arrayBuffer());
  level??=asset.initialLevel(entry.level);one=buildScene(asset,level,{maxHullVertices:entry.maxHullVertices});
 }else{level??=0;one=buildCuboidHouse()}
 // Deterministic budget: no SAT-cache early-out, exact sleep gate.
 const scene=withCapacity(layoutScene(withoutJoints(one),entry.layout,!entry.file),{deterministic:true});
 const sim=await createSimulation(device,scene),lines=[];
 try{
  // The row shot, else a ball at the highest piece from in front.
  const top=scene.bodies.slice(1,scene.projectileStart).reduce((a,b)=>b.p[1]>a.p[1]?b:a).p;
  const shot=scene.rowShot??{origin:[top[0],top[1],top[2]+scene.extent],direction:[0,0,-1]};
  sim.launch(shot.origin,shot.direction);
  for(let k=1;k<=ticks;k++){await sim.step(1);const c=sim.counters;
   if(k%60===0||k===ticks)lines.push(`tick ${k}: awake ${c.awake} contacts ${c.contact} candidates ${c.candidate} colors ${c.color} manifolds ${c.manifold}`)}
  const snap=await sim.snapshot(),bodies=scene.bodies,n=scene.bodyCount;
  let standing=0,sum=0;
  for(let i=1;i<scene.projectileStart;i++)if(Math.abs(snap[i*28+1]-bodies[i].p[1])<0.05)standing++;
  for(let i=1;i<n;i++)sum+=snap[i*28]+snap[i*28+1]+snap[i*28+2];
  lines.push(`${key} level ${level}: ${n} bodies · standing ${standing} of ${scene.projectileStart-1} · position sum ${sum.toFixed(4)}`);
  const r=await sim.inspectCollisions(),pairs=new Set();
  for(let i=0;i<r.contacts.length;i+=8)pairs.add(`${Math.min(r.contacts[i],r.contacts[i+1])},${Math.max(r.contacts[i],r.contacts[i+1])}`);
  lines.push(`inspect: ${r.counters.contact} contacts · ${r.counters.candidate} candidates · ${r.counters.manifold} manifolds · ${pairs.size} colliding pairs`);
  return lines.join('\n');
 }finally{sim.dispose()}
}
