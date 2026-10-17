import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {mortonSceneOrder,restoreSceneOrder} from '../spatial-order.mjs';
import {buildRing,buildGables,brickQuaternion,boxInvInertia,sphereMassProperties} from '../scene-builders.mjs';

// Specification of the GPU upload order, written independently of the
// implementation (bit-by-bit, no magic-number spreading):
// 1. Quantize each engine axis to 10 bits over the scene's bounding box:
//    q = min(1023, floor((p - lo) / max(hi - lo, 1e-9) * 1024)).
// 2. Interleave from the least significant bit in the axis order X, Z, Y
//    (bit b of X -> 3b, of Z -> 3b + 1, of Y -> 3b + 2), so vertical stacks
//    vary slowest within a cell.
// 3. Stable sort the dynamic bodies by code (ties keep scene order); the static
//    ground stays at GPU slot 0 for the renderer.
function mortonSpec(positions) {
  const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
  for(const p of positions)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],p[k]);hi[k]=Math.max(hi[k],p[k]);}
  const quant=(p,k)=>Math.min(1023,Math.floor((p[k]-lo[k])/Math.max(hi[k]-lo[k],1e-9)*1024));
  const codes=positions.map(p=>{
    const q=[quant(p,0),quant(p,2),quant(p,1)];
    let c=0;
    for(let b=0;b<10;b++)for(let s=0;s<3;s++)c+=((q[s]>>b)&1)*2**(3*b+s);
    return c;
  });
  return [0,...positions.map((_,i)=>i).slice(1).sort((a,b)=>codes[a]-codes[b]||a-b)];
}

function expectOrder(actual,expected,label) {
  assert.equal(actual.length,expected.length,`${label}: body count`);
  for(let gpu=0;gpu<actual.length;gpu++)
    if(actual[gpu]!==expected[gpu])assert.fail(`${label}: GPU slot ${gpu} holds scene body ${actual[gpu]}; the Morton specification uploads ${expected[gpu]}.`);
}

function verify(bodies,colliders,label) {
  const before = bodies.map(b=>b.p.slice());
  const {gpuToScene,sceneToGpu} = mortonSceneOrder(bodies,colliders);
  expectOrder(Array.from(gpuToScene),mortonSpec(bodies.map(b=>b.p)),label);
  assert.equal(gpuToScene[0],0,'Ground must stay at renderer/collider index zero.');
  const packed = new Uint32Array(bodies.length * 28);
  for (let gpu = 0; gpu < bodies.length; gpu++) {
    const scene = gpuToScene[gpu];
    assert.equal(sceneToGpu[scene],gpu,`${label}: inverse mapping lost body ${scene}`);
    assert.equal(colliders[scene].body,scene,`${label}: source collider identity was mutated`);
    for (let word=0;word<28;word++) packed[gpu*28+word]=(Math.imul(scene+1,2654435761)^word)>>>0;
  }
  const restored = new Uint32Array(restoreSceneOrder(packed.buffer,gpuToScene));
  for (let scene = 0; scene < bodies.length; scene++) {
    assert.deepEqual(bodies[scene].p,before[scene],`${label}: scene-space initial pose was mutated`);
    for (let word=0;word<28;word++) {
      const want=(Math.imul(scene+1,2654435761)^word)>>>0;
      if (restored[scene*28+word] !== want) assert.fail(`${label}: readback body ${scene}, word ${word} lost identity; drift/ball tests would inspect the wrong body.`);
    }
  }
  return {gpuToScene,sceneToGpu};
}

test('Morton upload handles negative coordinates, coincident bodies, degenerate axes and renderer ground',()=>{
  for (const positions of [ [[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]],
    [[0,0,0],[12,0,-8],[-12,0,8],[1,0,3],[1,0,3],[0,0,-8]] ]) {
    const bodies=positions.map((p,id)=>({p,invMass:id?1:0}));
    verify(bodies,bodies.map((_,body)=>({body})), 'edge case');
  }
  // Hand-checked: one body per corner of a unit cube. Each axis quantizes to
  // 0 or 1023, so at every bit level Y outranks Z outranks X: the order is
  // (0,0,0) (1,0,0) (0,0,1) (1,0,1) (0,1,0) (1,1,0) (0,1,1) (1,1,1).
  const corners=[[0,0,0],[0,1,0],[1,1,1],[0,0,1],[1,0,0],[1,1,0],[0,1,1],[1,0,1],[0,0,0]];
  const bodies=corners.map((p,id)=>({p,invMass:id?1:0}));
  const {gpuToScene}=mortonSceneOrder(bodies,bodies.map((_,body)=>({body})));
  assert.deepEqual(Array.from(gpuToScene),[0,8,4,3,7,1,5,6,2],'axis interleave order X, Z, Y from the least significant bit');
});

for (const [name,build,count] of [
  ['ring110k',(brick,ball)=>buildRing(brick,ball,80,40,4,10),110442],
  ['gables506k',(brick,ball)=>buildGables(brick,ball,16,68,30),505923],
]) {
  test(`${name}: full-scale permutation matches the Morton specification and preserves every readback word`,()=>{
    const bodies=[{p:[0,0,0],invMass:0,flags:0}],colliders=[{body:0,shape:2}];
    const add=(b,c)=>{colliders.push({...c,body:bodies.length});bodies.push(b);};
    build((p,a,mass)=>add({p,q:brickQuaternion(a),invMass:1/mass,invI:boxInvInertia(mass,.5,.25,.25),flags:9},{shape:2,params:[.5,.25,.25]}),
      (p,r,density,v,w)=>{const m=sphereMassProperties(r,density);add({p,invMass:1/m.mass,invI:[m.invInertia,m.invInertia,m.invInertia],v,w,flags:9},{shape:0,params:[r,0,0]});});
    assert.equal(bodies.length,count,'Do not reduce scene scale to improve timing.');
    const map=verify(bodies,colliders,name);
    for(let id=count-2;id<count;id++) assert.equal(map.gpuToScene[map.sceneToGpu[id]],id,'Impactor identity must survive spatial upload.');
    assert.ok(map.gpuToScene.some((id,gpu)=>id!==gpu),'This scene must actually exercise a nonidentity permutation.');
  });
}

// Deterministic pseudo-random scene (no Math.random: failures must reproduce).
function cloud(n,seed,scale=40,integer=false){
  let s=seed>>>0;const r=()=>((s=(Math.imul(s,1664525)+1013904223)>>>0)/2**32-.5)*scale;
  return [[0,0,0],...Array.from({length:n-1},()=>[r(),r()*.3,r()].map(x=>integer?Math.round(x):x))];
}
const upload=positions=>{const bodies=positions.map((p,id)=>({p,invMass:id?1:0}));return Array.from(mortonSceneOrder(bodies,bodies.map((_,body)=>({body}))).gpuToScene);};

test('upload order is a bijection with the ground fixed at GPU slot 0',()=>{
  for(const seed of [1,7,99]){
    const order=upload(cloud(5000,seed));
    assert.equal(order[0],0);
    assert.deepEqual([...order].sort((a,b)=>a-b),order.map((_,i)=>i),'every scene body appears exactly once');
  }
});

test('order depends only on positions relative to the scene bounds (integer translation invariance)',()=>{
  const base=cloud(3000,5,64,true);
  const moved=base.map(p=>[p[0]+1024,p[1]-512,p[2]+256]);
  assert.deepEqual(upload(moved),upload(base));
});

test('a flat (single-height) scene orders by its horizontal axes only',()=>{
  const flat=cloud(2000,11).map(p=>[p[0],3,p[2]]);
  assert.deepEqual(upload(flat),mortonSpec(flat));
});

test('Morton upload keeps spatial neighbours adjacent in memory',()=>{
  const positions=cloud(20000,3);
  // Scene order of a random cloud is spatially incoherent; the Morton order
  // must cut the mean distance between consecutive GPU bodies several-fold.
  const meanStep=order=>{let d=0;for(let i=2;i<order.length;i++){const a=positions[order[i-1]],b=positions[order[i]];d+=Math.hypot(a[0]-b[0],a[1]-b[1],a[2]-b[2]);}return d/(order.length-2);};
  const scene=meanStep(positions.map((_,i)=>i)),morton=meanStep(upload(positions));
  assert.ok(morton*5<scene,`Morton mean step ${morton.toFixed(3)} vs scene order ${scene.toFixed(3)}`);
});

test('invalid collider mappings and corrupt/truncated readbacks fail with identity diagnostics',()=>{
  const b=[{p:[0,0,0],invMass:0},{p:[1,2,3],invMass:1}];
  assert.throws(()=>mortonSceneOrder(b,[{body:0},{body:0}]),/collider 1 belongs to body 0/);
  assert.throws(()=>mortonSceneOrder(b,[]),/one collider per body/);
  assert.throws(()=>mortonSceneOrder([{p:[0,0,0],invMass:1}],[{body:0}]),/static ground/);
  assert.throws(()=>restoreSceneOrder(new ArrayBuffer(112),[0,1]),/cannot hold 2 records/);
  for(const map of [[0,0],[0,2],[0,-1],[0,NaN]]) assert.throws(()=>restoreSceneOrder(new ArrayBuffer(224),map),/invalid or duplicate scene identity/);
});

const demo = readFileSync(new URL('../demo.html',import.meta.url),'utf8');
const uploadStart = demo.indexOf('const storageOrder =');
const uploadEnd = demo.indexOf('device.queue.writeBuffer(bufs[0]',uploadStart);
assert.ok(uploadStart >= 0 && uploadEnd > uploadStart);
for (const spatialOrder of [false,true]) {
  test(`shipped upload keeps renderer colliders and previous poses paired (Morton=${spatialOrder})`,()=>{
    const bodies=[{p:[0,0,0],invMass:0},{p:[8,2,-9],invMass:1},{p:[-9,1,8],invMass:2},{p:[0,3,-4],invMass:3}];
    const colliders=bodies.map((_,body)=>({body,shape:body===3?0:2,params:[body+1,body+2,body+3],density:body+4}));
    const bodyWrites=new Map(), colliderWrites=new Map();
    const context={spatialOrder,mortonSceneOrder,bodies,colliders,cap:{maxBodies:64},bodyBytes:{},colBytes:{},
      writeBody(_bytes,id,b){bodyWrites.set(id,b);},writeCollider(_bytes,id,c){colliderWrites.set(id,c);}};
    runInNewContext(demo.slice(uploadStart,uploadEnd),context);
    const map=spatialOrder?mortonSceneOrder(bodies,colliders).sceneToGpu:bodies.map((_,i)=>i);
    assert.equal(bodyWrites.size,bodies.length*2,'Both current and previous-pose halves must be initialized.');
    assert.equal(colliderWrites.size,bodies.length);
    for(let scene=0;scene<bodies.length;scene++){
      const gpu=map[scene];
      assert.equal(bodyWrites.get(gpu),bodies[scene],`scene ${scene}: wrong current body uploaded`);
      assert.equal(bodyWrites.get(64+gpu),bodies[scene],`scene ${scene}: reset history uses a different identity`);
      const collider=colliderWrites.get(gpu);
      assert.equal(collider.body,gpu,`scene ${scene}: renderer and collision body index disagree`);
      assert.equal(collider.shape,colliders[scene].shape,'Moving the sphere must preserve its collision/render shape.');
      assert.deepEqual(collider.params,colliders[scene].params,'Spatial ordering must not change dimensions.');
      assert.equal(collider.density,colliders[scene].density);
      assert.equal(colliders[scene].body,scene,'Upload must not mutate the original scene identity.');
    }
  });
}
