// The lab's World bookkeeping (lab-sim.mjs) against a mock World: the frame loop takes only
// the steps a free readback slot allows and keeps the rest due, frames feed the counters
// and the breaks, queued shots launch with the step that fires them, captured steps and
// launches stay in order, and the stress models' poses are written only when a frame brought
// new ones. The physics itself is Rust's (engine-world `rubble::lab`: its Rust tests).
import test from 'node:test';
import assert from 'node:assert/strict';
import {labSimulation} from './lab-sim.mjs';

globalThis.GPUBufferUsage??={MAP_READ:1,COPY_SRC:4,COPY_DST:8,STORAGE:128};

// A World whose readbacks take `latency` polls to arrive, `slots` of them in flight.
function mockWorld({slots=3,latency=2,breaks={}}={}){
 const log={steps:0,writes:0,launches:[],reads:0};
 let inFlight=[],polls=0;
 const device={createBuffer:d=>({...d,destroy(){}}),queue:{writeBuffer(){log.writes++}}};
 const world={
  log,
  device:()=>device,solver:()=>({dt:1/60}),
  // labStep: undefined without a free readback slot; else the queued shots due, then the step.
  labStep(lab){if(inFlight.length>=slots)return undefined;const fired=lab.queue.splice(0).map(()=>7);inFlight.push({step:log.steps++,due:polls+latency});return fired},
  labQueue(lab,action){log.launches.push(action);lab.queue.push(action)},
  labClearQueue(lab){lab.queue.length=0},
  stepCount:()=>BigInt(log.steps),
  poll(){
   polls++;const ready=inFlight.filter(f=>f.due<=polls);inFlight=inFlight.filter(f=>f.due>polls);
   const last=ready.length?ready[ready.length-1].step:-1;
   return {empty:!ready.length,lastStep:last,free(){},
    counters:()=>ready.length?{awake:last+1,contacts:2*last,candidates:0,manifolds:0,colors:0,overflow:0,bodies:9}:null,
    labBreaks:()=>ready.flatMap(f=>breaks[f.step]?[f.step,breaks[f.step],0]:[]),
    gpuTimes:()=>ready.length?[{totalMs:0.5,segments:[['solve',0.5]]}]:[]};
  },
  readBodySlots:slots=>{log.reads++;return Promise.resolve(new Float32Array(slots.length*28).fill(log.steps))},
  labPoses:()=>new Float32Array(28*4),
  labStats:()=>({chunks:3,bonds:2,bondsIntact:1,bodies:2,crushed:0}),
  labReset(){log.reset=true},
  bodyBuffer:()=>'bodies',jointBuffer:()=>'joints',
  free(){log.freed=true},
 };
 return world;
}
const lab=drawsPoses=>({slotOf:()=>drawsPoses?[]:[0,1,2,3],drawsPoses,physicsHz:60,queue:[],shots:0,shotSpeed:10,free(){}});
const scene={bodyCount:4,jointCount:1,projectileStart:3,projectileCount:1};

test('the frame loop takes only the steps a readback slot allows, never waiting',()=>{
 const w=mockWorld({slots:3,latency:5}),sim=labSimulation(w,lab(false),scene);
 assert.equal(sim.tick(4),3,'three slots, three steps; the fourth stays due');
 assert.equal(sim.tick(4),0,'no slot free: nothing stepped, nothing waited for');
 assert.equal(w.log.steps,3);
});

test('step(n) takes every step and returns once their frames are read',async()=>{
 const w=mockWorld({slots:2,latency:2,breaks:{4:3,6:1}}),sim=labSimulation(w,lab(false),scene);
 await sim.step(10);
 assert.equal(w.log.steps,10);
 assert.equal(w.lastPolledStep,9,'the last step\'s frame arrived');
 assert.deepEqual({awake:sim.counters.awake,contact:sim.counters.contact},{awake:10,contact:18});
 assert.deepEqual(sim.breaks,{broken:4,crushed:0,firstStep:4});
 assert.equal(sim.gpuTickMs,0.5);
 assert.equal(sim.steps,10);
 sim.reset();
 assert.ok(w.log.reset);assert.deepEqual(sim.breaks,{broken:0,crushed:0,firstStep:null});
});

test('a queued shot launches with the step that fires it; captures stay in order',async()=>{
 const w=mockWorld(),sim=labSimulation(w,lab(false),scene),launched=[];
 sim.onLaunch(id=>launched.push(id));
 sim.captureSteps(true);
 await sim.step(2);
 sim.shoot([1,2,3],[0,0,-1]);
 assert.deepEqual(launched,[],'queued, not yet fired');
 await sim.step(1);
 assert.deepEqual(launched,[7],'fired by the next step');
 const out=await sim.takeCaptured();
 assert.deepEqual(out.map(e=>e.launch!==undefined?'launch':e.queued?'queued':'step'),['step','step','queued','launch','step']);
 assert.equal(out[0].bodies.length,4*28,'every scene body');
 assert.deepEqual(w.log.launches,[{shot:{origin:[1,2,3],direction:[0,0,-1]}}]);
});

test('bond models draw the world\'s buffers; the stress models their poses, written on new frames only',async()=>{
 const bond=labSimulation(mockWorld(),lab(false),scene);
 assert.equal(bond.bodyBuffer,'bodies');assert.equal(bond.jointBuffer,'joints');
 const w=mockWorld({latency:1}),sim=labSimulation(w,lab(true),scene);
 assert.notEqual(sim.bodyBuffer,'bodies');assert.equal(sim.jointBuffer,null);
 const first=w.log.writes;
 sim.poll();
 assert.equal(w.log.writes,first,'no frame, no write');
 sim.tick(1);sim.poll();
 assert.equal(w.log.writes,first+1,'one new frame, one write');
 assert.throws(()=>sim.reset(),/reload/);
 sim.dispose();assert.ok(w.log.freed);
});
