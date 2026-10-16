// Hull cooking through the wasm build is the engine-schema cooker (the
// source of hull-golden.json); the arena and recentring come from Rust too.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {HullArena,appendHull,hullView,hullBase,colliderBounds} from './wasm.mjs';
import {packHull} from '../generated/fracture-wasm/engine_fracture_wasm.js';
import {prism} from './convex-regression.mjs';
import {add,rotate} from './vec.mjs';

test('the wasm cooker writes the golden words of hull ABI v2',()=>{
 for(const g of JSON.parse(readFileSync(new URL('./hull-golden.json',import.meta.url)))){
  const words=packHull(Float32Array.from(g.vertices.flat()),Uint32Array.from(g.sizes),Uint32Array.from(g.indices));
  assert.deepEqual(Array.from(words),g.words,g.name);
 }
});

test('the arena keeps the unit cube at record 0, deduplicates and refcounts',()=>{
 const arena=new HullArena(),cube=hullView(arena.data,0);
 assert.deepEqual([cube.nv,cube.nf,cube.ne,cube.closed],[8,6,12,true]);
 const h=prism(6),a=appendHull(arena,h.vertices,h.topology),b=appendHull(arena,h.vertices,h.topology);
 assert.equal(hullBase(a.params),hullBase(b.params));assert.equal(arena.refs(a.base),2);
 assert.ok(Number.isFinite(a.params[0])&&Math.abs(a.params[0])>1e-30,'tagged base is a normal float');
 arena.takeDirty();assert.equal(arena.release(a.base),false);assert.equal(arena.release(a.base),true);assert.equal(arena.release(0),false);
 assert.ok(arena.records<a.base+1||arena.freeRecords>=0);
});

test('offset compound children are recentred with tight local bounds',()=>{
 const arena=new HullArena(),h=prism(8),{params,localPos}=appendHull(arena,h.vertices.map(v=>add(v,[50,0,0])),h.topology);
 assert.ok(params[3]<2);assert.ok(Math.abs(localPos[0]-50)<1e-5);
 const scene={hullData:arena.data},q=[0,Math.sin(.3),0,Math.cos(.3)],body={p:[1,2,3],q};
 const {lo,hi}=colliderBounds(scene,{body:0,shape:3,params,localPos},body);
 for(const v of h.vertices){const p=add(body.p,rotate(q,add(v,[50,0,0])));assert.ok(p.every((x,k)=>x>=lo[k]-1e-4&&x<=hi[k]+1e-4))}
});

test('the hull vertex limit rejects larger hulls with a clear error, never reduces them',()=>{
 assert.throws(()=>new HullArena({maxVertices:7}),/outside 8\.\.=64/);
 const arena=new HullArena({maxVertices:16});
 assert.equal(appendHull(arena,prism(8,.5,1).vertices,prism(8,.5,1).topology).vertexCount,16);
 const records=arena.records;
 assert.throws(()=>appendHull(arena,prism(9,.5,1).vertices,prism(9,.5,1).topology),/18 vertices; the configured limit is 16 \(hard maximum 64\)/);
 assert.equal(arena.records,records,'a rejected hull leaves the arena unchanged');
});

test('flat collision geometry is rejected instead of inventing a volume',()=>{
 const h=prism(4);assert.throws(()=>appendHull(new HullArena(),h.vertices.map(v=>[v[0],0,v[2]]),h.topology),/zero-volume|convex topology/);
});
