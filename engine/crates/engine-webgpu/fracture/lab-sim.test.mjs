// The lab's World bookkeeping (lab-sim.mjs) against a mock World: the frame loop takes only
// the steps a free readback slot allows and keeps the rest due, frames feed the counters
// and the breaks, captured steps and launches stay in order, and the threshold mode's poses
// are written only when a frame brought new ones. The physics itself is Rust's (engine-world,
// engine-world-wasm lab.rs: its Rust tests).
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
  canStep:()=>inFlight.length<slots,
  step(){if(inFlight.length>=slots)throw new Error('stepped without a free readback slot');inFlight.push({step:log.steps++,due:polls+latency})},
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
  labLaunch:(lab,o,d)=>{log.launches.push([...o]);return 7},
  labStats:()=>({chunks:3,bonds:2,bondsIntact:1,actors:2,crushed:0}),
  labReset(){log.reset=true},
  bodyBuffer:()=>'bodies',jointBuffer:()=>'joints',
  free(){log.freed=true},
 };
 return world;
}
const bondLab={slotOf:()=>[0,1,2,3],shots:0,shotSpeed:10,free(){}};
const thresholdLab={slotOf:()=>[],shots:0,shotSpeed:10,free(){}};
const scene={bodyCount:4,jointCount:1,projectileStart:3,projectileCount:1};

test('the frame loop takes only the steps a readback slot allows, never waiting',()=>{
 const w=mockWorld({slots:3,latency:5}),sim=labSimulation(w,bondLab,scene);
 assert.equal(sim.tick(4),3,'three slots, three steps; the fourth stays due');
 assert.equal(sim.tick(4),0,'no slot free: nothing stepped, nothing waited for');
 assert.equal(w.log.steps,3);
});

test('step(n) takes every step and returns once their frames are read',async()=>{
 const w=mockWorld({slots:2,latency:2,breaks:{4:3,6:1}}),sim=labSimulation(w,bondLab,scene);
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

test('captured steps and launches stay in order, one snapshot per step',async()=>{
 const w=mockWorld(),sim=labSimulation(w,bondLab,scene);
 sim.captureSteps(true);
 await sim.step(2);
 assert.equal(sim.launch([1,2,3],[0,0,-1]),7);
 await sim.step(1);
 const out=await sim.takeCaptured();
 assert.deepEqual(out.map(e=>e.launch!==undefined?'launch':'step'),['step','step','launch','step']);
 assert.equal(out[0].bodies.length,4*28,'every scene body');
 assert.deepEqual(w.log.launches,[[1,2,3]]);
});

test('bond modes draw the world\'s buffers; the threshold mode its poses, written on new frames only',async()=>{
 const bond=labSimulation(mockWorld(),bondLab,scene);
 assert.equal(bond.bodyBuffer,'bodies');assert.equal(bond.jointBuffer,'joints');
 const w=mockWorld({latency:1}),sim=labSimulation(w,thresholdLab,scene);
 assert.notEqual(sim.bodyBuffer,'bodies');assert.equal(sim.jointBuffer,null);
 const first=w.log.writes;
 sim.poll();
 assert.equal(w.log.writes,first,'no frame, no write');
 sim.tick(1);sim.poll();
 assert.equal(w.log.writes,first+1,'one new frame, one write');
 assert.throws(()=>sim.reset(),/reloads/);
 sim.dispose();assert.ok(w.log.freed);
});
