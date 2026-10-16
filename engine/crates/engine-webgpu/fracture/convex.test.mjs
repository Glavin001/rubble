import test from 'node:test';
import assert from 'node:assert/strict';
import {appendHull,colliderBounds,rotate,add} from './geometry.mjs';
import {prism,bipyramid} from './convex-regression.mjs';

test('64-vertex hull packs outward closed edge adjacency without simplifying vertices',()=>{
 const hull=prism(32),data=[],params=appendHull(data,hull.vertices,hull.topology);
 assert.equal(params[1],64);assert.equal(data[2],96);
 const nv=data[0],nf=data[1],ne=data[2],edgeBase=1+nv+2*nf;
 for(let e=0;e<ne;e++){
  const [a,b,fa]=data.slice((edgeBase+e*2)*4,(edgeBase+e*2)*4+3),fb=data[(edgeBase+e*2+1)*4];
  assert.notEqual(a,b);assert.notEqual(fa,fb);assert.ok(fa<nf&&fb<nf);
 }
});
test('offset compound hull bounds contain vertices at rotations and do not enclose its body origin',()=>{
 const hull=prism(8),data=[],params=appendHull(data,hull.vertices.map(v=>add(v,[50,0,0])),hull.topology);
 assert.ok(params[3]<2);
 for(let i=0;i<20;i++){
  const q=[0,Math.sin(i*.1),0,Math.cos(i*.1)],body={p:[1,2,3],q},c={body:0,shape:3,params},scene={bodies:[body],hullData:new Float32Array(data)};
  const {lo,hi}=colliderBounds(scene,c);
  for(const v of hull.vertices){const p=add(body.p,rotate(q,add(v,[50,0,0])));assert.ok(p.every((x,k)=>x>=lo[k]-1e-5&&x<=hi[k]+1e-5))}
  assert.ok(hi.every((x,k)=>x-lo[k]<3.1));
 }
});
test('repeated face/vertex records are normalized without adding a hull or changing vertices',()=>{
 const h=prism(4);const copy=h.vertices.map(v=>[...v]);const duplicate=h.topology.indices.map(i=>i+copy.length);
 const data=[];appendHull(data,[...h.vertices,...copy],{sizes:[...h.topology.sizes,...h.topology.sizes],indices:[...h.topology.indices,...duplicate]});
 assert.equal(data[0],16);assert.equal(data[1],6);assert.equal(data[2],12);
});
test('flat collision geometry is rejected instead of inventing a volume',()=>{
 const h=prism(4);assert.throws(()=>appendHull([],h.vertices.map(v=>[v[0],0,v[2]]),h.topology),/zero-volume|convex topology/);
});

test('64-vertex triangulated hull retains 124 faces and 186 manifold edges',()=>{
 const h=bipyramid(),data=[];appendHull(data,h.vertices,h.topology);
 assert.deepEqual(data.slice(0,3),[64,124,186]);
});
