// The lab's wasm path for real Shardwright bakes: parsing, level scenes and
// layouts come from engine-fracture (Rust; full tests in
// engine/crates/engine-fracture/tests). These check the bindings the lab uses.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {parseFractureGlb,buildScene,layoutScene,colliderVertices,SAMPLES} from './wasm.mjs';
const load=async file=>{const b=await readFile(new URL(`./assets/${file}`,import.meta.url));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)};
const arch=parseFractureGlb(await load('masonry-arch.glb'));

test('the bundled bake parses with its hierarchy, materials and metadata',()=>{
 assert.equal(arch.name,'masonry_arch');assert.equal(arch.seed,'1337');assert.equal(arch.fragments,751);assert.equal(arch.bonds,2821);
 assert.deepEqual(arch.levelFragmentCount,[23,165,171,392]);assert.deepEqual(arch.levels,[0,1,2,3]);
 assert.equal(arch.levelFragments(1),165);assert.equal(arch.initialLevel(2),2);assert.equal(arch.initialLevel(9),0);
 const meta=JSON.parse(arch.metadataJson());
 assert.equal(meta.physics.fragments.length,751);assert.ok(meta.materialLibrary.material.length===arch.materialCount);
 assert.ok(meta.physics.bonds.some(b=>b.composition.some(c=>c.interfaceMaterial>=0)));
});

test('corrupt, plain or unsupported GLBs are rejected before any GPU work',async()=>{
 const buffer=await load('masonry-arch.glb');
 assert.throws(()=>parseFractureGlb(buffer.slice(0,-4)),/header or length/);
 const copy=buffer.slice(0);new DataView(copy).setUint32(0,0,true);assert.throws(()=>parseFractureGlb(copy),/header/);
 const bad=Buffer.from(buffer),at=bad.indexOf(Buffer.from('SHARDWRIGHT_fracture'));bad[at]=88;
 assert.throws(()=>parseFractureGlb(bad.buffer.slice(bad.byteOffset,bad.byteOffset+bad.length)),/export/);
});

test('a level scene stands on the floor asleep with inactive projectile slots',()=>{
 const scene=buildScene(arch,0);
 assert.equal(scene.selected.length,23);assert.equal(scene.projectileStart,24);assert.equal(scene.bodyCount-scene.projectileStart,16);
 let minY=Infinity;
 for(const c of scene.colliders.filter(c=>c.shape===3)){assert.equal(scene.bodies[c.body].flags,5);for(const v of colliderVertices(scene,c))minY=Math.min(minY,v[1])}
 assert.ok(Math.abs(minY)<1e-6);
 assert.ok(scene.bodies.slice(scene.projectileStart).every(b=>b.flags===0&&b.invMass===0));
 assert.equal(scene.vertices.length%10,0);assert.equal(scene.hullData.length,scene.hullRecords*4);
});

test('the house, its flat-hull skips and the layouts of the Scene menu',async()=>{
 const house=parseFractureGlb(await load('veneer-house.glb'));
 const one=buildScene(house,0);
 assert.deepEqual([one.selected.length,one.hullCount,one.bondCount],[131,163,454]);
 const fine=buildScene(house,2);
 assert.equal(fine.skippedHulls.length,35);assert.deepEqual(fine.skippedHulls[0],{fragment:1401,hull:3590});
 const frags=fine.projectileStart-1;
 const grid=layoutScene(fine,{columns:7,rows:7}),suburb=layoutScene(fine,{suburb:true});
 assert.equal(grid.copies,49);assert.equal(grid.instanceStride,frags);assert.equal(grid.projectileStart,1+49*frags);
 assert.equal(grid.rowShot.origin[0],0);assert.deepEqual(grid.rowShot.direction,[0,0,1]);assert.equal(grid.projectileArc,false);
 assert.deepEqual(grid.capacity,{contactsPerCollider:8,candidatesPerCollider:8,maxColors:64,colorRounds:64});
 assert.equal(suburb.copies,48);assert.ok(suburb.roadVertices.length>0&&suburb.roadVertices.length%60===0);
 assert.equal(grid.hullData.length,fine.hullData.length,'copies share one hull arena');
 assert.deepEqual(SAMPLES.map(s=>s.key),['house','cuboid','arch','row','grid','cuboidGrid','suburb','cuboids','cuboids2x']);
});

test('a 16-vertex bake enforces its hull limit with a clear error',async()=>{
 const house=parseFractureGlb(await load('veneer-house-hull16.glb'));
 assert.ok(house.maxHullVertices<=16);
 assert.throws(()=>buildScene(house,3,{maxHullVertices:8}),/configured limit is 8 \(hard maximum 64\)/);
 const row=layoutScene(buildScene(house,3,{maxHullVertices:16}),{columns:1,rows:20});
 assert.equal(row.copies,20);assert.equal(row.skippedHulls.length,23);
});
