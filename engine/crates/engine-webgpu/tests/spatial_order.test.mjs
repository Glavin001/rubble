import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {runInNewContext} from 'node:vm';
import {mortonSceneOrder,restoreSceneOrder} from '../spatial-order.mjs';
import {buildRing,buildGables,brickQuaternion,boxInvInertia,sphereMassProperties} from '../scene-builders.mjs';

// Expected orders come from the pinned three-avbd Morton routine, recorded in
// fixtures/three-avbd-reference.json by fixtures/generate-three-avbd-reference.ts
// (the only code that reads the gitignored inspiration/ checkout).
const reference = JSON.parse(readFileSync(new URL('./fixtures/three-avbd-reference.json',import.meta.url),'utf8'));
const sha256 = order => createHash('sha256').update(new Uint8Array(Uint32Array.from(order).buffer)).digest('hex');

function expectOrder(actual,expected,label) {
  if (Array.isArray(expected)) {
    assert.deepEqual(actual,expected,`${label}: Z-up Morton order or stable ties differ from pinned three-avbd.`);
    return;
  }
  assert.equal(actual.length,expected.count,`${label}: body count`);
  for (const [gpu,scene] of expected.samples)
    assert.equal(actual[gpu],scene,`${label}: GPU slot ${gpu} holds scene body ${actual[gpu]}, pinned three-avbd uploads ${scene}.`);
  assert.equal(sha256(actual),expected.sha256,`${label}: Morton order differs from pinned three-avbd ${reference.revision}.`);
}

function verify(bodies,colliders,label,expected) {
  const before = bodies.map(b=>b.p.slice());
  const {gpuToScene,sceneToGpu} = mortonSceneOrder(bodies,colliders);
  expectOrder(Array.from(gpuToScene),expected,label);
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
  for (const {positions,order} of reference.morton.edgeCases) {
    const bodies=positions.map((p,id)=>({p,invMass:id?1:0}));
    verify(bodies,bodies.map((_,body)=>({body})), 'edge case', order);
  }
});

for (const [name,build,count] of [
  ['ring110k',(brick,ball)=>buildRing(brick,ball,80,40,4,10),110442],
  ['gables506k',(brick,ball)=>buildGables(brick,ball,16,68,30),505923],
]) {
  test(`${name}: full-scale permutation matches pinned ordering and preserves every readback word`,()=>{
    const bodies=[{p:[0,0,0],invMass:0,flags:0}],colliders=[{body:0,shape:2}];
    const add=(b,c)=>{colliders.push({...c,body:bodies.length});bodies.push(b);};
    build((p,a,mass)=>add({p,q:brickQuaternion(a),invMass:1/mass,invI:boxInvInertia(mass,.5,.25,.25),flags:9},{shape:2,params:[.5,.25,.25]}),
      (p,r,density,v,w)=>{const m=sphereMassProperties(r,density);add({p,invMass:1/m.mass,invI:[m.invInertia,m.invInertia,m.invInertia],v,w,flags:9},{shape:0,params:[r,0,0]});});
    assert.equal(bodies.length,count,'Do not reduce scene scale to improve timing.');
    const map=verify(bodies,colliders,name,reference.morton[name]);
    for(let id=count-2;id<count;id++) assert.equal(map.gpuToScene[map.sceneToGpu[id]],id,'Impactor identity must survive spatial upload.');
    assert.ok(map.gpuToScene.some((id,gpu)=>id!==gpu),'This fixture must actually exercise a nonidentity permutation.');
  });
}

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
