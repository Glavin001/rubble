// Bond joints through the wasm build (capacity and spring formulas:
// engine/crates/engine-fracture/tests/bonds.rs).
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {parseFractureGlb,buildScene,scaleBondSprings,preloadDone,sceneHandle} from './wasm.mjs';
import {rotate,quatMul,add,sub} from './vec.mjs';

const data=await readFile(new URL('./assets/veneer-house.glb',import.meta.url));
const house=parseFractureGlb(data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength));

test('bond joints weld at the centroid with the baked rest pose and capped springs',()=>{
 const sc=buildScene(house,2),dt=1/120;
 assert.equal(sc.jointCount,sc.bondCount);assert.equal(sc.joints.length,sc.bondCount);
 for(const j of sc.joints.slice(0,200)){
  const A=sc.bodies[j.bodyA],B=sc.bodies[j.bodyB];
  assert.ok(Math.hypot(...sub(add(A.p,rotate(A.q,j.anchorA)),add(B.p,rotate(B.q,j.anchorB))))<1e-5);
  const qb=quatMul(A.q,j.rest);assert.ok(Math.abs(Math.abs(qb.reduce((s,x,k)=>s+x*B.q[k],0))-1)<1e-6);
  const m=Math.min(1/A.invMass,B.invMass>0?1/B.invMass:Infinity);
  j.k.slice(0,3).forEach(k=>assert.ok(k>0&&k<=5*m/(dt*dt)*(1+1e-6)));
 }
 // One scene-wide factor: uncapped rows keep their ratios.
 const raw=sc.joints.slice(0,50).map(j=>({...j,k:j.k.map(x=>x*3)}));
 const scaled=scaleBondSprings(raw,sc.bodies,dt,0.5,Infinity);
 const f=scaled[0].k[0]/raw[0].k[0];
 for(let i=0;i<50;i++)raw[i].k.forEach((k,r)=>assert.ok(Math.abs(scaled[i].k[r]-k*f)<=1e-9*k*f));
});

test('preloading wakes every piece with breaking off and restarts asleep at the settled poses',()=>{
 const sc=buildScene(house,0),awake=sceneHandle(sc).preloadVariant().meta();
 assert.equal(awake.params.flags&2,2);
 const snap=new Float32Array(sc.bodyCount*28);for(let i=0;i<sc.bodyCount;i++){snap[i*28+1]=i;snap[i*28+7]=1}
 const settled=sceneHandle(sc).settled(snap,120,true),bodies=settled.bodies();
 assert.ok(bodies.slice(1,sc.projectileStart).every((b,i)=>b.flags===5&&b.p[1]===i+1));
 assert.deepEqual(settled.meta().preload,{steps:120,settled:true});
 assert.equal(preloadDone(60,0),true);assert.equal(preloadDone(64,0),false);assert.equal(preloadDone(600,9),true);
});
