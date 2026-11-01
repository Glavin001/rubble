// The fracture lab's World as the page and its checks use it (lab-world.mjs makes it):
// the frame loop's steps and polls, the counters and breaks the frames carry, snapshots
// and captured steps. Bookkeeping only (no wasm here, so it is unit-tested with a mock
// world: lab-sim.test.mjs); every physics decision is in Rust (engine-world-wasm lab.rs).
import {pollFrame,macrotask} from '../world-loop.mjs';

const RECORD_BYTES=112;

export function labSimulation(world,lab,scene){
 const device=world.device(),threshold=lab.slotOf().length===0,dt=world.solver().dt;
 const bodyCount=scene.bodyCount;
 // Threshold: the pose buffer the renderer reads (labPoses, a record per scene body).
 const poses=threshold?device.createBuffer({label:'lab poses',size:Math.max(RECORD_BYTES,bodyCount*RECORD_BYTES),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC}):null;
 const writePoses=()=>{if(poses)device.queue.writeBuffer(poses,0,world.labPoses(lab))};
 writePoses();
 const slots=threshold?null:Int32Array.from(lab.slotOf());
 let counters={awake:0,contact:0,candidate:0,manifold:0,color:0,overflow:0},gpuTickMs=null,gpuTime=null,timing={wallMs:0,encodeMs:0,waitMs:0};
 // Bond breaks seen in frames: the total, the first step that broke one.
 const breaks={broken:0,crushed:0,firstStep:null};
 let capture=null,disposed=false;
 const onFrame=f=>{
  const c=f.counters();
  if(c)counters={awake:c.awake,contact:c.contacts,candidate:c.candidates,manifold:c.manifolds,color:c.colors,overflow:c.overflow,bodies:c.bodies};
  const b=f.labBreaks();
  for(let i=0;i<b.length;i+=3){breaks.broken+=b[i+1];breaks.crushed+=b[i+2];if(breaks.firstStep===null&&b[i+1]>0)breaks.firstStep=b[i]}
  const t=f.gpuTimes?.();if(t?.length){gpuTime=t[t.length-1];gpuTickMs=gpuTime.totalMs}
 };
 // Threshold poses are written when a frame brought new ones.
 let fresh=false;
 const poll=()=>{pollFrame(world,f=>{fresh||=!f.empty;onFrame(f)});if(fresh){fresh=false;writePoses()}};
 // One step if a readback slot is free (the step is deferred, never dropped, otherwise).
 const tryStep=()=>{
  if(!world.canStep())return false;
  const t=performance.now();world.step();const ms=performance.now()-t;
  timing={wallMs:ms,encodeMs:ms,waitMs:0};
  if(capture)capture.push({bodies:snapshotNow()});
  return true;
 };
 const snapshotNow=()=>threshold?Promise.resolve(world.labPoses(lab)):world.readBodySlots(slots);
 return {
  world,lab,device,threshold,
  bodyBuffer:poses??world.bodyBuffer(),
  jointBuffer:!threshold&&scene.jointCount?world.jointBuffer():null,
  // Streams drawn once (the shots, the street props) with world slots for body ids.
  remap:data=>threshold?data:Float32Array.from(lab.remap(data)),
  get shots(){return lab.shots},get steps(){return Number(world.stepCount())},
  get counters(){return {...counters}},get timing(){return {...timing}},get gpuTickMs(){return gpuTickMs},
  // The last step's GPU time as World measured it ({totalMs, segments: [[phase, ms]]}).
  get gpuTime(){return gpuTime},
  get breaks(){return {...breaks}},
  // World steps whole: nothing is ever split across frames.
  slicing:false,
  get shotSpeed(){return lab.shotSpeed},
  setProjectileMass(kg){lab.setShotMass(kg)},setProjectileSpeedScale(k){lab.setSpeedScale(k)},setProjectileCcd(on){lab.setCcd(!!on)},
  launch(origin,direction,{speed,arc}={}){
   const id=world.labLaunch(lab,Float64Array.from(origin),Float64Array.from(direction),speed,arc);
   capture?.push({launch:id});return id;
  },
  firePackShot(index){const id=world.labFirePackShot(lab,index);capture?.push({launch:id});return id},
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
  // The shot slots' records (28 floats each, projectile order), for the ball trails.
  async shotRecords(){
   const n=Math.min(lab.shots,scene.projectileCount),start=scene.projectileStart;
   if(threshold)return world.labPoses(lab).slice(start*28,(start+n)*28);
   return world.readBodySlots(slots.slice(start,start+n));
  },
  // The step's fixed dt (s).
  get dt(){return dt},
  // The destructibles' counts: {chunks, bonds, bondsIntact, actors, crushed}.
  stats(){return world.labStats(lab)},
  // Per bond, 1 if intact (as of the last polled frame).
  bondsIntact(){return world.labBondsIntact(lab)},
  reset(){
   if(threshold)throw new Error('the threshold mode reloads to reset');
   world.labReset(lab);breaks.broken=breaks.crushed=0;breaks.firstStep=null;counters={...counters,awake:0};
  },
  // Directed debris: settings (GuidanceSettings) or null, and zones (ZoneDesc).
  setGuidance(settings,zones=[]){world.labSetGuidance(lab,settings,zones)},
  // Test hook: every step's bodies (scene order) and every launch, in order.
  captureSteps(on){capture=on?[]:null},
  async takeCaptured(){
   const list=capture??[];capture=capture&&[];
   const out=[];for(const e of list)out.push(e.launch!==undefined?e:{bodies:await e.bodies});
   return out;
  },
  dispose(){
   if(disposed)return;disposed=true;poses?.destroy();lab.free();
   world.free();
  },
 };
}

