// The fracture lab's World as the page and its checks use it (lab-world.mjs makes it):
// the frame loop's steps and polls, the counters and breaks the frames carry, snapshots
// and captured steps. Bookkeeping only (no wasm here, so it is unit-tested with a mock
// world: lab-sim.test.mjs); every physics decision is in Rust (engine-world `rubble::lab`).
import {pollFrame,macrotask} from '../world-loop.mjs';

const RECORD_BYTES=112;

export function labSimulation(world,lab,scene){
 const device=world.device(),poseDraw=lab.drawsPoses,dt=1/lab.physicsHz;
 const bodyCount=scene.bodyCount;
 // The stress models: the pose buffer the renderer reads (labPoses, a record per scene body).
 const poses=poseDraw?device.createBuffer({label:'lab poses',size:Math.max(RECORD_BYTES,bodyCount*RECORD_BYTES),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC}):null;
 const writePoses=()=>{if(poses)device.queue.writeBuffer(poses,0,world.labPoses(lab))};
 writePoses();
 const slots=poseDraw?null:Int32Array.from(lab.slotOf());
 let counters={awake:0,contact:0,candidate:0,manifold:0,color:0,overflow:0},gpuTickMs=null,gpuTime=null,timing={wallMs:0,encodeMs:0,waitMs:0};
 // Bond breaks seen in frames: the total, the first step that broke one.
 const breaks={broken:0,crushed:0,firstStep:null};
 let capture=null,disposed=false;
 // Balls launched (scene bodies), as the steps that fired them report them.
 const launchListeners=new Set();
 const onFrame=f=>{
  const c=f.counters();
  if(c)counters={awake:c.awake,contact:c.contacts,candidate:c.candidates,manifold:c.manifolds,color:c.colors,overflow:c.overflow,bodies:c.bodies};
  const b=f.labBreaks();
  for(let i=0;i<b.length;i+=3){breaks.broken+=b[i+1];breaks.crushed+=b[i+2];if(breaks.firstStep===null&&b[i+1]>0)breaks.firstStep=b[i]}
  const t=f.gpuTimes?.();if(t?.length){gpuTime=t[t.length-1];gpuTickMs=gpuTime.totalMs}
 };
 // Poses are written when a frame brought new ones.
 let fresh=false;
 const poll=()=>{pollFrame(world,f=>{fresh||=!f.empty;onFrame(f)});if(fresh){fresh=false;writePoses()}};
 // One step if a readback slot is free (the step is deferred, never dropped, otherwise): the
 // shots due launched first (Rust's schedule), then the step.
 const tryStep=()=>{
  const t=performance.now(),fired=world.labStep(lab),ms=performance.now()-t;
  if(fired===undefined)return false;
  timing={wallMs:ms,encodeMs:ms,waitMs:0};
  for(const id of fired){capture?.push({launch:id});for(const f of launchListeners)f(id)}
  if(capture)capture.push({bodies:snapshotNow()});
  return true;
 };
 const snapshotNow=()=>poseDraw?Promise.resolve(world.labPoses(lab)):world.readBodySlots(slots);
 return {
  world,lab,device,
  // Whether the renderer draws from labPoses (the stress models).
  poseDraw,
  bodyBuffer:poses??world.bodyBuffer(),
  jointBuffer:!poseDraw&&scene.jointCount?world.jointBuffer():null,
  // Streams drawn once (the balls, the street props) with world slots for body ids.
  remap:data=>poseDraw?data:Float32Array.from(lab.remap(data)),
  get shots(){return lab.shots},get queued(){return lab.queued},get steps(){return Number(world.stepCount())},
  get counters(){return {...counters}},get timing(){return {...timing}},get gpuTickMs(){return gpuTickMs},
  // The last step's GPU time as World measured it ({totalMs, segments: [[phase, ms]]}).
  get gpuTime(){return gpuTime},
  get breaks(){return {...breaks}},
  // World steps whole: nothing is ever split across frames.
  slicing:false,
  get shotSpeed(){return lab.shotSpeed},get shotMass(){return lab.shotMass},get shotRadius(){return lab.shotRadius},
  // The balls fired from now on: multiples of the scene's mass and speed, continuous collision.
  setBall(massScale,speedScale,ccd){lab.setBall(massScale,speedScale,!!ccd)},
  // Queue an action (engine-world-wasm lab.rs `action_of`) `delay` steps from now: a shot
  // ray, 'row', 'volley', 'topple', 'meteor', {pack: index}, {rocket: {origin, direction, tntKg}}
  // or {blast: {position, tntKg}}. Its balls launch, and its charges go off, at their steps.
  queue(action,delay=0){world.labQueue(lab,action,delay);capture?.push({queued:action})},
  // A ball along a ray at the next step. Returns the scene body it will be: the next ball
  // slot in order, after those already queued (Rust fires them in that order).
  shoot(origin,direction){
   const slot=scene.projectileStart+(lab.shots+lab.queued)%Math.max(1,scene.projectileCount);
   this.queue({shot:{origin:Array.from(origin),direction:Array.from(direction)}});return slot},
  launch(origin,direction){return this.shoot(origin,direction)},
  clearQueue(){world.labClearQueue(lab)},
  // Called with each ball's scene body as the step that fires it is taken.
  onLaunch(f){launchListeners.add(f);return()=>launchListeners.delete(f)},
  get packShots(){return lab.packShots()},
  // The frame loop: up to `count` due steps without waiting, then the ready frames.
  // Returns the steps taken (the rest stay due).
  tick(count){let n=0;while(n<count&&tryStep())n++;poll();return n},
  // Checks and automation: `count` steps as fast as readbacks allow, every frame read.
  async step(count=1){
   for(let i=0;i<count;){if(tryStep())i++;else await macrotask();poll()}
   await this.flush();
  },
  async flush(){
   const last=Number(world.stepCount())-1;
   while((world.lastPolledStep??-1)<last){await macrotask();poll()}
  },
  poll,
  // Every body record (28 floats, BodyState) in scene body order, as the queue leaves it.
  async snapshot(){return snapshotNow()},
  // The ball slots' records (28 floats each, projectile order), for the ball trails.
  async shotRecords(){
   const n=Math.min(lab.shots,scene.projectileCount),start=scene.projectileStart;
   if(poseDraw)return world.labPoses(lab).slice(start*28,(start+n)*28);
   return world.readBodySlots(slots.slice(start,start+n));
  },
  // The step's fixed dt (s).
  get dt(){return dt},
  // Every number the HUD shows (engine-world `LabStats`): {steps, shots, chunks, bonds,
  // bondsIntact, bodies, crushed, awake, contacts, gpuStepMs, maxStepsPerSecond,
  // realtimeShare, physicsHz}.
  stats(){return world.labStats(lab)},
  // Per bond, 1 if intact (as of the last polled frame).
  bondsIntact(){return world.labBondsIntact(lab)},
  reset(){
   if(poseDraw)throw new Error('the stress models reload to reset');
   world.labReset(lab);breaks.broken=breaks.crushed=0;breaks.firstStep=null;counters={...counters,awake:0};
  },
  // Directed debris: settings (GuidanceSettings) or null, and zones (ZoneDesc, scene bodies).
  setGuidance(settings,zones=[]){world.labSetGuidance(lab,settings,zones)},
  // Test hook: every step's bodies (scene order) and every launch, in order.
  captureSteps(on){capture=on?[]:null},
  async takeCaptured(){
   const list=capture??[];capture=capture&&[];
   const out=[];for(const e of list)out.push(e.bodies?{bodies:await e.bodies}:e);
   return out;
  },
  dispose(){
   if(disposed)return;disposed=true;poses?.destroy();lab.free();
   world.free();
  },
 };
}
