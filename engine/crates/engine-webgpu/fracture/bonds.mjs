// Bond preloading. Elastic bonds carry load only once they deform, so a baked
// structure released at its rest pose drops onto its springs and the transient
// overloads bonds that hold it fine at rest (Shardwright's self-weight gate).
// Instead, settle one copy under gravity with breaking off (FLAG_BOND_PRELOAD)
// and start every copy asleep from the settled, statically loaded poses.
import {createSimulation} from './simulation.mjs';

export const FLAG_BOND_PRELOAD=2;

export async function preloadBonds(device, scene, {maxSteps=600}={}) {
  if(!scene.joints?.length)return scene;
  const dynamic=i=>i>0&&i<scene.projectileStart&&scene.bodies[i].invMass>0;
  const awake={...scene,bodies:scene.bodies.map((b,i)=>dynamic(i)?{...b,flags:9}:b),
    params:{...scene.params,flags:(scene.params.flags??0)|FLAG_BOND_PRELOAD}};
  const sim=await createSimulation(device,awake);
  try{
    let steps=0;
    // Settled once the bonded island falls asleep (sleep keeps its bond state).
    while(steps<maxSteps){await sim.step(4);steps+=4;if(steps%60===0&&sim.counters.awake===0)break}
    const snap=await sim.snapshot(),state=await sim.jointState();
    const bodies=scene.bodies.map((b,i)=>dynamic(i)?{...b,flags:5,v:[0,0,0],w:[0,0,0],
      p:[snap[i*28],snap[i*28+1],snap[i*28+2]],q:[snap[i*28+4],snap[i*28+5],snap[i*28+6],snap[i*28+7]]}:b);
    return {...scene,bodies,preload:{steps,settled:sim.counters.awake===0,forces:state.lambda}};
  }finally{sim.dispose()}
}
