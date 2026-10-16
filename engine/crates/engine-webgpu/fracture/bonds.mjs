// Bond preloading (engine-fracture Scene::preload_variant / settled). Elastic
// bonds carry load only once they deform, so a baked structure released at its
// rest pose drops onto its springs and the transient overloads bonds that hold
// it fine at rest (Shardwright's self-weight gate). Instead, settle one copy
// under gravity with breaking off and start every copy asleep from the settled,
// statically loaded poses.
import {createSimulation} from './simulation.mjs';
import {sceneHandle,sceneObject,preloadDone} from './wasm.mjs';

export async function preloadBonds(device, scene) {
  if(!scene.jointCount)return scene;
  const handle=sceneHandle(scene),sim=await createSimulation(device,{handle:handle.preloadVariant()});
  try{
    let steps=0;
    do{await sim.step(4);steps+=4}while(!preloadDone(steps,sim.counters.awake));
    return sceneObject(handle.settled(await sim.snapshot(),steps,sim.counters.awake===0));
  }finally{sim.dispose()}
}
