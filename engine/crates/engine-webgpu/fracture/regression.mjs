import {rotate,add,sub,norm} from './geometry.mjs';
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
  const o=c.body*28,q=Array.from(after.slice(o+4,o+8)),p=Array.from(after.slice(o,o+3));
  for(let i=0;i<c.params[1];i++){const v=(c.params[0]+1+i)*4;minHullY=Math.min(minHullY,add(rotate(q,Array.from(demo.scene.hullData.slice(v,v+3))),p)[1])}
 }
 require(peakContacts>0&&peakAwake>1,'projectile did not wake fragments');require(maxDisplacement>.1,'projectile did not move the arch');
 require(minHullY>-.025,`hulls penetrated the floor: ${minHullY} m`);
 require(after[projectile*28+1]>=demo.scene.projectileRadius-.025,'sphere fell through floor');
 const result={passed:true,steps:demo.simulation.steps,peakContacts,peakAwake,maxDisplacement,minHullY,counters:demo.simulation.counters};
 await demo.reset();const reset=await demo.snapshot();require(reset.every((x,i)=>x===before[i]),'reset did not restore every body field');require(demo.simulation.shots===0,'reset did not reload projectiles');
 return result;
}
