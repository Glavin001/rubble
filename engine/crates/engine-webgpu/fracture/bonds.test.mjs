// Bond -> joint conversion (CPU): capacities and springs follow Shardwright's
// reference formulas, scaling keeps row ratios, and capping bounds the solver.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {parseFractureGlb} from './asset.mjs';
import {buildScene,bondCapacities,bondSprings,scaleBondSprings,rotate,quatMul,inverseRotate,JOINT_BOND,BOND_PENALTY_CAP} from './geometry.mjs';

const data=await readFile(new URL('./assets/veneer-house.glb',import.meta.url));
const house=parseFractureGlb(data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength));
const close=(a,b,rel=1e-6)=>assert.ok(Math.abs(a-b)<=rel*Math.max(Math.abs(a),Math.abs(b),1e-30),`${a} vs ${b}`);

test('bond capacities follow the reference stress solver (tension, compression, shear, bending)',()=>{
  const lib=house.library;
  const mortar=house.physics.bonds.find(b=>b.composition[0]?.interfaceMaterial===lib.material.length&&b.b>=0);
  const [T,C,S,M]=bondCapacities(house,mortar);
  const ft=lib.interface_material[0].tensile_strength,k=mortar.strengthScale,A=mortar.area;
  const fc=Math.min(...[mortar.a,mortar.b].map(f=>lib.material[house.physics.fragments[f].materialIds[0]].compressive_strength));
  close(T,ft*A*k);close(C,fc*A*k);close(S,1.5*ft*A*k);
  close(M,ft*Math.min(mortar.iUU,mortar.iVV)/Math.max(...mortar.extentHalf)*k);
  // A bearing (ft = 0) resists with 0.05 fc instead of nothing.
  const bearing=house.physics.bonds.find(b=>{const i=b.composition[0]?.interfaceMaterial;return i>=0&&lib.interface_material[i-lib.material.length].tensile_strength===0});
  if(bearing)assert.ok(bondCapacities(house,bearing)[0]>0);
});

test('bond springs are series compliances of both fragments and the interface',()=>{
  const b=house.physics.bonds.find(x=>x.b>=0&&x.composition[0]?.interfaceMaterial>=0);
  const [kn,ks,ks2,ku,kv,kt]=bondSprings(house,b);
  assert.equal(ks,ks2);
  assert.ok([kn,ks,ku,kv,kt].every(x=>Number.isFinite(x)&&x>0));
  close(ku/kn,b.iUU/b.area);close(kv/kn,b.iVV/b.area);close(kt/ks,b.polarMoment/b.area);
  // Every bond of the house has finite springs (incl. anisotropic timber).
  for(const level of [0,1,2,3])for(const x of house.physics.bonds.filter(y=>y.level===level))
    assert.ok(bondSprings(house,x).every(Number.isFinite),`bond ${x.id}`);
});

test('bond joints weld at the centroid with the baked rest pose and scaled, capped springs',()=>{
  const sc=buildScene(house,2),dt=1/120;
  assert.equal(sc.joints.length,house.physics.bonds.filter(b=>b.level===2).length);
  for(const j of sc.joints.slice(0,200)){
    const A=sc.bodies[j.bodyA],B=sc.bodies[j.bodyB];
    // Both anchors are the same world point at rest.
    const wa=A.p.map((x,k)=>x+rotate(A.q,j.anchorA)[k]),wb=B.p.map((x,k)=>x+rotate(B.q,j.anchorB)[k]);
    assert.ok(Math.hypot(...wa.map((x,k)=>x-wb[k]))<1e-5);
    // qB = qA * rest.
    const qb=quatMul(A.q,j.rest);assert.ok(Math.abs(Math.abs(qb.reduce((s,x,k)=>s+x*B.q[k],0))-1)<1e-6);
    // Unit normal and u, orthogonal, in A's frame.
    assert.ok(Math.abs(Math.hypot(...j.normal)-1)<1e-5&&Math.abs(j.normal.reduce((s,x,k)=>s+x*j.frameU[k],0))<1e-5);
    const m=Math.min(1/A.invMass,B.invMass>0?1/B.invMass:Infinity),I=Math.min(1/Math.max(...A.invI),B.invMass>0?1/Math.max(...B.invI):Infinity);
    j.k.forEach((k,r)=>assert.ok(k>0&&k<=BOND_PENALTY_CAP*(r<3?m:I)/(dt*dt)*(1+1e-6)));
  }
  // Uncapped rows keep the physical ratios exactly (one scene-wide factor).
  const raw=house.physics.bonds.filter(b=>b.level===2).map(b=>bondSprings(house,b));
  const scaled=scaleBondSprings(raw.map((k,i)=>({...sc.joints[i],k})),sc.bodies,dt,0.5,Infinity);
  const s=scaled[0].k[0]/raw[0][0];for(let i=0;i<50;i++)raw[i].forEach((k,r)=>close(scaled[i].k[r],k*s,1e-5));
});
