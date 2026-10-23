import {add,sub,norm} from './vec.mjs';
import {colliderVertices} from './wasm.mjs';
// Run in this demo's browser console, or import from a browser automation test.
export async function runFractureRegression(demo=window.__FRACTURE_DEMO__) {
 const require=(ok,message)=>{if(!ok)throw new Error(`Fracture regression: ${message}`)};
 require(demo.ready,'demo is not ready');
 require(demo.scene.selected.length===23,'select the bundled arch at level 0');
 await demo.reset();const before=await demo.snapshot();
 await demo.stepFrames(120);const rest=await demo.snapshot();
 for(let i=1;i<demo.scene.projectileStart;i++)for(let k=0;k<3;k++)require(rest[i*28+k]===before[i*28+k],'arch moved before impact');
 // Aim a real sphere at the crown from in front of the arch.
 const target=demo.scene.bodies.slice(1,demo.scene.projectileStart).reduce((a,b)=>a.p[1]>b.p[1]?a:b).p;
 const origin=add(target,[0,0,3]);const projectile=await demo.shootRay(origin,norm(sub(target,origin)));
 let peakContacts=0,peakAwake=0;
 for(let n=0;n<12;n++){const c=await demo.stepFrames(60);require(c.overflow===0,'GPU capacity overflow');peakContacts=Math.max(peakContacts,c.contact);peakAwake=Math.max(peakAwake,c.awake)}
 const after=await demo.snapshot();let minHullY=Infinity,maxDisplacement=0;
 for(let i=1;i<demo.scene.projectileStart;i++){
  const o=i*28;for(const k of [0,1,2,4,5,6,7,12,13,14,16,17,18])require(Number.isFinite(after[o+k]),`nonfinite state in body ${i}`);
  require(Math.abs(Math.hypot(...after.slice(o+4,o+8))-1)<.001,`invalid orientation in body ${i}`);
  maxDisplacement=Math.max(maxDisplacement,Math.hypot(...sub(Array.from(after.slice(o,o+3)),Array.from(before.slice(o,o+3)))));
 }
 for(const c of demo.scene.colliders.filter(c=>c.shape===3)){
  const o=c.body*28,body={q:Array.from(after.slice(o+4,o+8)),p:Array.from(after.slice(o,o+3))};
  for(const v of colliderVertices(demo.scene,c,body))minHullY=Math.min(minHullY,v[1]);
 }
 require(peakContacts>0&&peakAwake>1,'projectile did not wake fragments');require(maxDisplacement>.1,'projectile did not move the arch');
 require(minHullY>-.025,`hulls penetrated the floor: ${minHullY} m`);
 require(after[projectile*28+1]>=demo.scene.projectileRadius-.025,'sphere fell through floor');
 const result={passed:true,steps:demo.simulation.steps,peakContacts,peakAwake,maxDisplacement,minHullY,counters:demo.simulation.counters};
 await demo.reset();const reset=await demo.snapshot();require(reset.every((x,i)=>x===before[i]),'reset did not restore every body field');require(demo.simulation.shots===0,'reset did not reload projectiles');
 return result;
}
// A 300 m/s shot at the arch crown (2.5 m a tick at 120 Hz), without and with
// CCD balls (engine-viewer fracture_regression a_ccd_shot_does_not_tunnel_through_the_arch):
// the plain ball passes the arch between ticks, the CCD ball stops at the crown.
export async function runCcdShotRegression(demo=window.__FRACTURE_DEMO__,{speed=300,ticks=30}={}) {
 const require=(ok,message)=>{if(!ok)throw new Error(`CCD shot regression: ${message}`)};
 require(demo.ready,'demo is not ready');
 const sim=()=>demo.simulation,start=demo.scene.projectileStart,bodies=demo.scene.bodies.slice(1,start);
 const crown=bodies.reduce((a,b)=>a.p[1]>b.p[1]?a:b).p,back=Math.min(...bodies.map(b=>b.p[2]));
 const runs={};
 for(const ccd of [false,true]){
  await demo.reset();await demo.stepFrames(10);const before=await demo.snapshot();
  sim().setProjectileCcd(ccd);sim().setProjectileSpeedScale(speed/sim().shotSpeed);
  const id=await demo.shootRay(add(crown,[0,0,3]),[0,0,-1]);let minZ=Infinity;
  for(let t=0;t<ticks;t++){const c=await demo.stepFrames(1);require(c.overflow===0,'GPU capacity overflow');minZ=Math.min(minZ,(await demo.snapshot())[id*28+2])}
  const after=await demo.snapshot();let moved=0;
  for(let i=1;i<start;i++)moved=Math.max(moved,Math.hypot(...sub(Array.from(after.slice(i*28,i*28+3)),Array.from(before.slice(i*28,i*28+3)))));
  runs[ccd?'ccd':'control']={minZ,moved,finite:after.slice(id*28,id*28+28).every(Number.isFinite)};
 }
 sim().setProjectileSpeedScale(1);
 require(runs.control.minZ<back-2,`the plain ball should pass the arch (z ${runs.control.minZ})`);
 require(runs.ccd.minZ>back-1&&runs.ccd.finite,`the CCD ball passed the arch (z ${runs.ccd.minZ})`);
 require(runs.ccd.moved>1&&runs.ccd.moved>5*runs.control.moved,`the CCD ball should knock the crown out (${runs.ccd.moved} vs ${runs.control.moved})`);
 return {passed:true,back,...runs};
}
