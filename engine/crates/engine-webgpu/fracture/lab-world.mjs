// The fracture lab's physics: rubble's Rust World (engine-world-wasm lab.rs) on the
// browser's WebGPU, the host native and Unreal run. A lab scene (engine-fracture, built in
// Rust: wasm.mjs) becomes one World destructible: StrengthModel::Bonds for the bond modes
// (None, breakable D6 bonds), StrengthModel::Threshold for the stress mode's simple
// threshold. This file makes the worlds (lab-sim.mjs keeps the page's bookkeeping); every
// physics decision is in Rust.
//
// Bond modes draw from the world's body and joint buffers in place (scene body i is world
// slot i). The threshold mode's chunks change bodies as they split, so the page draws them
// from a pose buffer filled each frame from World's polled poses (labPoses).
import {createWorld,WebWorld,macrotask} from '../rubble-world.mjs';
import {labSimulation} from './lab-sim.mjs';
import {LabAsset,sceneObject} from './wasm.mjs';

// A small world that only holds the page's device: every lab world runs on it, so the
// renderers and the learned-impact kernels keep one GPUDevice across loads.
let holder=null;
const TINY={capacity:{maxBodies:64,maxColliders:64,maxContacts:4096,maxCandidates:4096,maxJoints:1,maxColors:8,maxCommands:16,maxHullVertices:1024}};
export async function labDevice(){holder??=await createWorld(TINY);return holder.device()}
// ?broadphase=full: the full broadphase instead of the awake-indexed one (A/B, same pairs).
const FULL_BROADPHASE=typeof location!=='undefined'&&new URLSearchParams(location.search).get('broadphase')==='full';
const worldFor=async def=>{await labDevice();if(FULL_BROADPHASE)def={...def,features:{...(def.features??{}),awakeBroadphase:false}};const w=WebWorld.createOnDeviceOf(def,holder);w.flags=holder.flags;return w};

// A World for `scene`: `asset` (a LabAsset) and the load options (copies, physicsHz, crush,
// strengthScale, gpuTiming).
export async function createLabWorld(scene,asset,opts={}){
 const world=await worldFor(asset.worldDef(scene.handle,opts));
 try{
  world.labRegister(asset);
  const lab=world.labPlace(asset,scene.handle,opts);
  return labSimulation(world,lab,scene);
 }catch(e){world.free();throw e}
}

// The lab's bond preloading on World: `scene` (one house, D6 bonds) settled under its own
// weight in a helper world, a few steps a frame (never waiting), then returned with every
// piece asleep at its settled pose (the page lays out copies of it).
export async function settleBonds(scene,onProgress=()=>{}){
 if(!scene.jointCount)return scene;
 const asset=LabAsset.bonds(scene.handle),world=await worldFor(asset.worldDef(scene.handle,{}));
 try{
  world.labRegister(asset);
  const job=world.labPreloadBegin(asset);
  if(job){
   for(;world.labPreloadAdvance(job)===undefined;){onProgress(`Settling bonds under self-weight · ${job.steps} steps…`);await macrotask()}
   job.free();
  }
  return sceneObject(world.labSettledScene(asset,scene.handle));
 }finally{asset.free();world.free()}
}

// The lab's asset for a scene: the bond modes' (Bonds) or a threshold source.
export const bondAsset=scene=>LabAsset.bonds(scene.handle);
export const thresholdAsset=(kind,bytes,level=0)=>LabAsset.threshold(kind,new Uint8Array(bytes??new ArrayBuffer(0)),level);
