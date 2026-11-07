// The fracture lab's physics: rubble's Rust World on the browser's WebGPU, through the lab
// API every host calls (engine-world `rubble::lab`; engine-world-wasm lab.rs binds it). The
// page gives a run's choices (scene, arrangement, bond model, settings: `createLab`); Rust
// builds the scene, the world, the D6 settle and the placement, and decides every physics
// question. This file only drives the settle a frame at a time and hands the placed run to
// lab-sim.mjs (the page's bookkeeping).
import {createWorld,WebWorld,macrotask} from '../rubble-world.mjs';
import {labSimulation} from './lab-sim.mjs';
import {LabRun,sceneObject} from './wasm.mjs';

// A small world that only holds the page's device: every lab world runs on it, so the
// renderers and the learned-impact kernels keep one GPUDevice across loads.
// The promise is kept, not its result: callers that arrive while it is being made (a page
// changing its scene before its first load finished) share the one device.
let holder=null,holding=null;
const TINY={capacity:{maxBodies:64,maxColliders:64,maxContacts:4096,maxCandidates:4096,maxJoints:1,maxColors:8,maxCommands:16,maxHullVertices:1024}};
export async function labDevice(){holding??=createWorld(TINY).then(w=>holder=w);await holding;return holder.device()}
// ?broadphase=full: the full broadphase instead of the awake-indexed one (A/B, same pairs).
const FULL_BROADPHASE=typeof location!=='undefined'&&new URLSearchParams(location.search).get('broadphase')==='full';

// A lab run placed in its own World: `def` the page's choices (engine-world-wasm lab.rs
// `DefJs`: source, level, arrangement, layout, bonds, physicsHz, strengthScale, ball scales,
// ccd, contactFracture, strikePrediction, crush, gpuTiming), `bytes` the GLB or pack. D6 bonds
// settle first, a few steps a frame (never waiting). A combination that cannot work rejects
// with Rust's reason.
export async function createLab(def,bytes,onProgress=()=>{}){
 await labDevice();
 return placeRun(LabRun.create(def,bytes?new Uint8Array(bytes):undefined),onProgress);
}

// A lab scene the page built itself (benchmarks, synthetic bond tests) placed as given under
// a bond model (`def`: bonds, physicsHz, ...; the scene's rate by default).
export async function createLabFromScene(scene,def={}){
 await labDevice();
 return placeRun(LabRun.fromScene(scene.handle,def,false));
}

// The lab's bond preload of one copy: `scene` with its D6 bonds settled under self-weight
// (every piece asleep at its settled pose), a few steps a frame (never waiting).
export async function settleScene(scene,onProgress=()=>{}){
 if(!scene.jointCount)return scene;
 await labDevice();
 const run=LabRun.fromScene(scene.handle,{bonds:'d6'},true);
 try{await settle(run,onProgress);return sceneObject(run.scene())}finally{run.free()}
}

async function settle(run,onProgress){
 if(!run.needsSettle)return;
 const job=run.beginSettle(holder);
 if(!job)return;
 try{for(;job.advance()===undefined;){onProgress(`Settling bonds under self-weight · ${job.steps} steps…`);await macrotask()}run.finishSettle(job)}
 finally{job.free()}
}

async function placeRun(run,onProgress=()=>{}){
 try{
  await settle(run,onProgress);
  const scene=sceneObject(run.scene()),world=WebWorld.createForLab(run,holder,FULL_BROADPHASE);
  world.flags=holder.flags;
  try{
   const sim=labSimulation(world,world.labPlace(run),scene);
   // The scene drawn, and the scene's suggested settings (applied only when asked).
   sim.scene=scene;sim.suggested=run.suggestedSettings();
   return sim;
  }catch(e){world.free();throw e}
 }finally{run.free()}
}
