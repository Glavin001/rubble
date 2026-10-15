import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {HullArena,appendHull,cookHull,packHull,hullView,hullBase,colliderBounds,colliderVertices,rotate,add,dot,cross,sub} from './geometry.mjs';
import {prism,bipyramid} from './convex-regression.mjs';

const golden=JSON.parse(readFileSync(new URL('./hull-golden.json',import.meta.url)));

test('JS packer matches the Rust cooker golden words (hull ABI v2)',()=>{
 const f32=new Float32Array(1),u32=new Uint32Array(f32.buffer),asF=w=>{u32[0]=w;return f32[0]};
 for(const g of golden){
  const vertices=g.vertices.map(v=>v.map(Math.fround));
  const words=packHull(cookHull(vertices,{sizes:g.sizes,indices:g.indices}));
  assert.equal(words.length,g.words.length,`${g.name}: record count`);
  const view=hullView(Uint32Array.from(g.words),0),floatWord=new Set();
  // Vertex/plane/meta float lanes; everything else is an exact integer.
  for(let i=4;i<16;i++)if(i%4!==3||i<12)floatWord.add(i);
  floatWord.add(7);floatWord.add(11);
  for(let i=0;i<view.nv;i++)for(let k=0;k<3;k++)floatWord.add((4+i)*4+k);
  for(let i=0;i<view.nf;i++)for(let k=0;k<4;k++)floatWord.add((4+view.nv+i)*4+k);
  for(let i=0;i<words.length;i++){
   if(floatWord.has(i))assert.ok(Math.abs(asF(words[i])-asF(g.words[i]))<=1e-6*Math.max(1,Math.abs(asF(g.words[i]))),`${g.name}: float word ${i}`);
   else assert.equal(words[i],g.words[i],`${g.name}: word ${i}`);
  }
 }
});

test('64-vertex prism packs closed adjacency, incident edges and CCW loops',()=>{
 const arena=new HullArena(),hull=prism(32),{params}=appendHull(arena,hull.vertices,hull.topology);
 const view=hullView(arena.data,hullBase(params));
 assert.equal(view.nv,64);assert.equal(view.ne,96);assert.ok(view.closed);
 for(let e=0;e<view.ne;e++){const [a,b,l,r]=view.edge(e);assert.notEqual(a,b);assert.notEqual(l,r);assert.ok(l<view.nf&&r<view.nf)}
 for(let i=0;i<view.nv;i++)assert.deepEqual(view.adjacency(i),Array.from({length:view.ne},(_,e)=>e).filter(e=>view.edge(e)[0]===i||view.edge(e)[1]===i));
 for(let k=0;k<view.nf;k++){const ids=view.face(k),{n}=view.plane(k),p=ids.map(view.vertex);
  let s=[0,0,0];for(let i=1;i<p.length-1;i++)s=add(s,cross(sub(p[i],p[0]),sub(p[i+1],p[0])));assert.ok(dot(s,n)>0,'loop winds CCW about the outward normal')}
});

test('offset compound children are recentred into local positions with tight bounds',()=>{
 const arena=new HullArena(),hull=prism(8),{params,localPos}=appendHull(arena,hull.vertices.map(v=>add(v,[50,0,0])),hull.topology);
 assert.ok(params[3]<2,'bound radius stays local to the child');assert.ok(Math.abs(localPos[0]-50)<1e-6);
 const scene={bodies:[],hullData:arena.data};
 for(let i=0;i<20;i++){
  const q=[0,Math.sin(i*.1),0,Math.cos(i*.1)],body={p:[1,2,3],q},c={body:0,shape:3,params,localPos};scene.bodies[0]=body;
  const {lo,hi}=colliderBounds(scene,c);
  for(const v of hull.vertices){const p=add(body.p,rotate(q,add(v,[50,0,0])));assert.ok(p.every((x,k)=>x>=lo[k]-1e-4&&x<=hi[k]+1e-4))}
  assert.ok(hi.every((x,k)=>x-lo[k]<3.1));
  assert.equal(colliderVertices(scene,c).length,16);
 }
});

test('repeated face/vertex records are normalized without changing vertex records',()=>{
 const h=prism(4),copy=h.vertices.map(v=>[...v]),duplicate=h.topology.indices.map(i=>i+copy.length);
 const hull=cookHull([...h.vertices,...copy],{sizes:[...h.topology.sizes,...h.topology.sizes],indices:[...h.topology.indices,...duplicate]});
 assert.equal(hull.vertices.length,16);assert.equal(hull.faces.length,6);assert.equal(hull.edges.length,12);assert.ok(hull.closed);
});

test('flat collision geometry is rejected instead of inventing a volume',()=>{
 const h=prism(4);assert.throws(()=>cookHull(h.vertices.map(v=>[v[0],0,v[2]]),h.topology),/zero-volume|convex topology/);
});

test('64-vertex triangulated hull retains 124 faces and 186 manifold edges',()=>{
 const h=bipyramid(),hull=cookHull(h.vertices,h.topology);
 assert.deepEqual([hull.vertices.length,hull.faces.length,hull.edges.length],[64,124,186]);assert.ok(hull.closed);
});

test('arena keeps the unit cube at record 0 and deduplicates identical hulls',()=>{
 const arena=new HullArena(),cube=hullView(arena.data,0);
 assert.deepEqual([cube.nv,cube.nf,cube.ne,cube.closed],[8,6,12,true]);
 const h=prism(6),a=appendHull(arena,h.vertices,h.topology),b=appendHull(arena,h.vertices,h.topology);
 assert.equal(hullBase(a.params),hullBase(b.params));assert.ok(Number.isFinite(a.params[0])&&Math.abs(a.params[0])>1e-30,'tagged base is a normal float');
});

test('arena refcounts, reuses freed blocks first-fit, merges holes and reports dirty ranges',()=>{
 const arena=new HullArena();arena.takeDirty();
 const a=cookHull(prism(4).vertices,prism(4).topology),b=cookHull(prism(6).vertices,prism(6).topology),c=cookHull(prism(5).vertices,prism(5).topology);
 const ba=arena.insert(a),ba2=arena.insert(a);assert.equal(ba,ba2);assert.equal(arena.refs(ba),2);
 const bb=arena.insert(b),bc=arena.insert(c);
 assert.deepEqual(arena.takeDirty(),[[ba,bc+packHull(c).length/4]]);
 assert.equal(arena.release(ba),false);assert.equal(arena.release(ba),true);assert.equal(arena.release(0),false);
 assert.equal(arena.release(bb),true);
 assert.equal(arena.freeRecords,(packHull(a).length+packHull(b).length)/4,'adjacent holes merge');
 const d=cookHull(prism(7).vertices,prism(7).topology),bd=arena.insert(d);
 assert.equal(bd,ba,'first fit reuses the hole');assert.deepEqual(arena.takeDirty(),[[bd,bd+packHull(d).length/4]]);
 assert.equal(hullView(arena.data,bd).nv,14);
 const records=arena.records;assert.equal(arena.release(bc),true);assert.ok(arena.records<records,'free tail shrinks');
});

test('filter words match the Rust CollisionFilter encoding',async()=>{
 const {filterWords}=await import('./geometry.mjs');
 assert.deepEqual(filterWords(undefined,3),{generation:0,material:3});
 assert.deepEqual(filterWords({categories:1,mask:2,group:-3},5),{generation:(1|2<<16)>>>0,material:(5|0xfffd<<16)>>>0});
});

test('convexity is judged at the f32 precision of the input frame (as engine-schema)',()=>{
  const box=(center,half,lift)=>{
    const vertices=Array.from({length:8},(_,i)=>[0,1,2].map(k=>center[k]+((i>>k)&1?half:-half)));
    vertices.push([center[0],center[1]+half+lift,center[2]]);
    const faces=[[0,2,6,4],[1,5,7,3],[0,4,5,1],[2,3,7,6],[0,1,3,2],[4,6,7,5]];
    return {vertices,topology:{sizes:faces.map(()=>4),indices:faces.flat()}};
  };
  let b=box([6,0,0],.25,3e-6);
  assert.deepEqual(cookHull(b.vertices,b.topology).vertices.slice(0,8),b.vertices.slice(0,8));
  b=box([6,0,0],.25,1e-3);assert.throws(()=>cookHull(b.vertices,b.topology),/Non-convex/);
  b=box([0,0,0],.25,3e-6);assert.throws(()=>cookHull(b.vertices,b.topology),/Non-convex/);
  // Data stored as f32 in a 6 m frame, cooked after moving to the origin.
  assert.doesNotThrow(()=>cookHull(b.vertices,b.topology,{sourceScale:6.25}));
});

test('hull arena vertex limit rejects larger hulls with a clear error, never reduces them',()=>{
  assert.throws(()=>new HullArena({maxVertices:7}),/outside 8\.\.64/);
  assert.throws(()=>new HullArena({maxVertices:65}),/outside 8\.\.64/);
  assert.equal(new HullArena().maxVertices,64);
  const arena=new HullArena({maxVertices:16});
  const small=prism(8,.5,1),large=prism(9,.5,1);
  assert.equal(appendHull(arena,small.vertices,small.topology).hull.vertices.length,16);
  const records=arena.records;
  assert.throws(()=>appendHull(arena,large.vertices,large.topology),/18 vertices; the configured limit is 16 \(hard maximum 64\)/);
  assert.equal(arena.records,records,'a rejected hull leaves the arena unchanged');
  assert.doesNotThrow(()=>appendHull(new HullArena(),large.vertices,large.topology));
});

// Mirrors hullSupportIndex (physics/convex.slang). Some cooked house hulls keep
// a redundant vertex on no face loop as vertex 0; a climb starting there never
// moved and missed the ground by up to 15 cm, so plates sank through the floor.
test('hill-climb supports match brute force on every certified house hull',async()=>{
 const {parseFractureGlb}=await import('./asset.mjs');const {buildScene}=await import('./geometry.mjs');
 const bytes=readFileSync(new URL('./assets/veneer-house.glb',import.meta.url));
 const asset=parseFractureGlb(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
 const dirs=Array.from({length:64},(_,i)=>{const y=1-2*(i+.5)/64,r=Math.sqrt(1-y*y),t=i*2.399963;return [r*Math.cos(t),y,r*Math.sin(t)]});
 let isolatedStart=0;
 for(const level of [1,2,3]){
  const scene=buildScene(asset,level),seen=new Set();
  for(const c of scene.colliders){if(c.shape!==3||seen.has(hullBase(c.params)))continue;seen.add(hullBase(c.params));
   const h=hullView(scene.hullData,hullBase(c.params));if(!h.closed)continue;
   if(!h.adjacency(0).length)isolatedStart++;
   for(const d of dirs){
    let cur=h.edge(0)[0],curD=dot(h.vertex(cur),d);
    for(let step=0;step<h.nv;step++){let next=cur,nextD=curD;
     for(const e of h.adjacency(cur)){const [a,b]=h.edge(e),o=a===cur?b:a,od=dot(h.vertex(o),d);if(od>nextD){nextD=od;next=o}}
     if(next===cur)break;cur=next;curD=nextD}
    let best=-Infinity;for(let i=0;i<h.nv;i++)best=Math.max(best,dot(h.vertex(i),d));
    assert.ok(best-curD<=1e-6,`level ${level} hull ${hullBase(c.params)}: climb stopped ${best-curD} short`);
   }
  }
 }
 assert.ok(isolatedStart>0,'the asset still exercises hulls whose vertex 0 is on no face');
});
