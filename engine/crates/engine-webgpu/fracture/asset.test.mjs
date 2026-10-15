import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {parseFractureGlb,EXTENSION} from './asset.mjs';
import {buildScene,principalAxes,rotate,inverseRotate,add,sub,hullView,hullBase,colliderVertices} from './geometry.mjs';
const bytes=await readFile(new URL('./assets/masonry-arch.glb',import.meta.url));
const buffer=bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength);
const asset=parseFractureGlb(buffer);
test('actual Shardwright bake retains hierarchy, material IDs and bond topology',()=>{
 assert.equal(asset.physics.name,'masonry_arch');assert.equal(asset.physics.seed,'1337');assert.equal(asset.physics.fragments.length,751);assert.equal(asset.physics.bonds.length,2821);
 assert.deepEqual(asset.physics.levelFragmentCount,[23,165,171,392]);
 const bond=asset.physics.bonds.find(b=>b.composition.some(c=>c.interfaceMaterial>=0));
 const id=bond.composition.find(c=>c.interfaceMaterial>=0).interfaceMaterial;
 assert.equal(asset.library.interface_material[id-asset.library.material.length].id,'lime_mortar');
 assert.equal(asset.physics.bonds.filter(b=>b.level===0&&b.anchor).length,2);
 assert.equal(asset.physics.fragments[0].mass>0,true);
});
test('principal inertia basis reconstructs a tensor with off-diagonal terms',()=>{
 const inertia=[4,1,.3,1,2,.2,.3,.2,3],{q,moments}=principalAxes(inertia);
 for(let col=0;col<3;col++){const axis=[0,0,0];axis[col]=1;const v=rotate(q,inverseRotate(q,axis).map((x,i)=>x*moments[i]));for(let row=0;row<3;row++)assert.ok(Math.abs(v[row]-inertia[row*3+col])<1e-7)}
});
test('selected levels never instantiate both a fragment and its ancestor; hull coordinates survive COM/eigenbasis conversion',()=>{
 for(let l=0;l<4;l++){
  const scene=buildScene(asset,l);assert.equal(scene.projectileStart-1,asset.physics.levelFragmentCount[l]);assert.equal(scene.selected.every(f=>f.level===l),true);
  for(const f of scene.selected){assert.ok(!scene.byId.has(f.parent));const {id}=scene.byId.get(f.id),body=scene.bodies[id],c=scene.colliders.find(c=>c.body===id);
   const local=add(c.localPos,hullView(scene.hullData,hullBase(c.params)).vertex(0));const reconstructed=add(rotate(body.q,local),f.com),original=asset.physics.hulls[f.hullBegin].vertices[0];assert.ok(Math.hypot(...sub(reconstructed,original))<1e-5);
  }
 }
});
test('rejects corrupt, plain or unsupported GLBs before allocating GPU resources',()=>{
 assert.throws(()=>parseFractureGlb(buffer.slice(0,-4)),/header or length/);
 const copy=buffer.slice(0);new DataView(copy).setUint32(0,0,true);assert.throws(()=>parseFractureGlb(copy),/header/);
 const text=Buffer.from(buffer).toString('utf8');assert.ok(text.includes(EXTENSION));
 const bad=Buffer.from(buffer);const offset=bad.indexOf(Buffer.from('SHARDWRIGHT_fracture'));bad[offset]=88;assert.throws(()=>parseFractureGlb(bad.buffer.slice(bad.byteOffset,bad.byteOffset+bad.length)),/export/);
});

test('floor meets hulls and fragments start asleep; projectile slots are inactive',()=>{
 const scene=buildScene(asset,0);let minY=Infinity;
 for(const c of scene.colliders.filter(c=>c.shape===3)){
  const body=scene.bodies[c.body];assert.equal(body.flags,5);
  for(const v of colliderVertices(scene,c))minY=Math.min(minY,v[1]);
 }
 assert.ok(Math.abs(minY)<1e-6);
 assert.equal(scene.bodies.length-scene.projectileStart,16);
 assert.ok(scene.bodies.slice(scene.projectileStart).every(b=>b.flags===0&&b.invMass===0));
});

test('bundled house imports its building parts, compound hulls and finer hierarchy',async()=>{
 const data=await readFile(new URL('./assets/veneer-house.glb',import.meta.url));
 const house=parseFractureGlb(data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength));
 assert.equal(house.physics.name,'veneer_house');
 assert.deepEqual(house.physics.levelFragmentCount,[131,567,3158,5905]);
 const scene=buildScene(house,0);
 assert.equal(scene.selected.length,131);
 assert.equal(scene.colliders.filter(c=>c.shape===3).length,163);
 assert.equal(scene.bonds.length,454);
 assert.ok(scene.extent>6&&scene.height>4);
 assert.ok(scene.selected.some(f=>f.hullCount>1));
 assert.ok(scene.selected.every(f=>scene.bodies[scene.byId.get(f.id).id].flags===5));
 const fine=buildScene(house,1);
 assert.equal(fine.selected.length,567);
 assert.equal(fine.colliders.filter(c=>c.shape===3).length,2253);
 assert.throws(()=>buildScene(house,2),/Fragment 1401, hull 3590: Degenerate zero-volume hull/);
 assert.throws(()=>buildScene(house,3),/Fragment 5162, hull 8777: Degenerate zero-volume hull/);
});
