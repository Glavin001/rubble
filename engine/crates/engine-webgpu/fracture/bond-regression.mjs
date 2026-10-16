// Breakable bond joints (JOINT_BOND) on the GPU, run in the browser:
//   import('./bond-regression.mjs').then(m=>m.runBondGpuTests(device))
// Each case is a small scene of unit-density boxes welded by bonds, judged by
// physical invariants (a strong weld holds, a weak one breaks the right way).
import {createSimulation} from './simulation.mjs';
import {quatMul,quatConjugate,inverseRotate,sub} from './vec.mjs';
import {scaleBondSprings} from './wasm.mjs';

const H=0.25,MASS=1000*(2*H)**3,G=9.81;
const box=(p,q=[0,0,0,1])=>({p,q,invMass:1/MASS,invI:Array(3).fill(1/(MASS*(2*H)**2/6)),flags:9});

// A bond between bodies a and b at world point c with world normal n (A->B).
// Physical springs (N/m, N m/rad) like a short masonry joint; each scene maps
// them to solver penalty with scaleBondSprings, as buildScene does.
const SPRINGS=[1e8,5e7,5e7,2e6,2e6,1e6];
function bond(bodies,a,b,c,n,strength){
  const qa=bodies[a].q,qb=bodies[b].q;
  return {bodyA:a,bodyB:b,anchorA:inverseRotate(qa,sub(c,bodies[a].p)),anchorB:inverseRotate(qb,sub(c,bodies[b].p)),
    rest:quatMul(quatConjugate(qa),qb),normal:inverseRotate(qa,n),strength,k:SPRINGS,
    frameU:inverseRotate(qa,Math.abs(n[0])<0.9?[1,0,0]:[0,1,0])};
}

function scene(bodies,joints,extra=[]){
  const colliders=[{body:0,shape:4,params:[0,1,0,0],density:0},...bodies.slice(1).map((b,i)=>({body:i+1,shape:2,params:[H,H,H,0],density:1000})),...extra];
  return {bodies,colliders,joints,hullData:new Uint32Array(4),projectileStart:bodies.length,projectileCount:0,
    params:{dt:1/120,iterations:8,gravityY:-G,flags:1,sleepLin:.03,sleepAng:.05,sleepTime:1}};
}

let penaltyScale;
async function run(device,sc,steps){
  sc={...sc,joints:scaleBondSprings(sc.joints,sc.bodies,1/120,penaltyScale)};
  const sim=await createSimulation(device,sc);
  try{for(let s=0;s<steps;s+=4)await sim.step(4);const snap=await sim.snapshot(),flags=await sim.jointFlags();
    return {pos:i=>[snap[i*28],snap[i*28+1],snap[i*28+2]],rot:i=>[snap[i*28+4],snap[i*28+5],snap[i*28+6],snap[i*28+7]],flags:[...flags],counters:sim.counters};}
  finally{sim.dispose()}
}

export async function runBondGpuTests(device,{scale}={}){
  penaltyScale=scale;
  const out={},weight=MASS*G;
  const ground={p:[0,0,0],q:[0,0,0,1],invMass:0,invI:[0,0,0],flags:0};
  // 1/2. A box hangs from a world weld 3 m up: holds when strong, falls when the
  // tension capacity is below its weight (and only then).
  for(const [name,tension] of [['hangStrong',100*weight],['hangWeak',0.3*weight]]){
    const bodies=[ground,box([0,3,0])];
    const j=bond(bodies,0,1,[0,3+H,0],[0,-1,0],[tension,1e9,1e9,1e9]);
    const r=await run(device,scene(bodies,[j]),240);
    out[name]={y:r.pos(1)[1],broken:r.flags[0]&1,overflow:r.counters.overflow};
  }
  // 3. Sign check: a box resting on the ground welded below is compressed, not
  // pulled: a tiny tension capacity must not break it.
  {
    const bodies=[ground,box([0,H,0])];
    const j=bond(bodies,0,1,[0,0,0],[0,1,0],[1e-3,1e9,1e9,1e9]);
    const r=await run(device,scene(bodies,[j]),240);
    out.restingCompression={y:r.pos(1)[1],broken:r.flags[0]&1};
  }
  // 4/5. Cantilever: a box welded by its side face to a world wall. A strong
  // weld keeps it level (all rotations locked); a weak bending capacity breaks.
  for(const [name,moment] of [['cantileverStrong',1e9],['cantileverWeak',0.05*weight*H]]){
    const bodies=[ground,box([H,2,0])];
    const j=bond(bodies,0,1,[0,2,0],[1,0,0],[1e9,1e9,1e9,moment]);
    const r=await run(device,scene(bodies,[j]),240);
    const q=r.rot(1),tilt=2*Math.acos(Math.min(1,Math.abs(q[3])));
    out[name]={y:r.pos(1)[1],tiltDeg:tilt*180/Math.PI,broken:r.flags[0]&1};
  }
  // 6. Two bonded boxes: the pair falls as one rigid body, the bond intact,
  // and the bonded faces never collide (exact pair exclusion).
  {
    const bodies=[ground,box([0,2,0]),box([0,2+2*H,0])];
    const j=bond(bodies,1,2,[0,2+H,0],[0,1,0],[1e9,1e9,1e9,1e9]);
    const r=await run(device,scene(bodies,[j]),360);
    const a=r.pos(1),b=r.pos(2);
    out.bondedPair={gap:b[1]-a[1],dx:Math.hypot(b[0]-a[0],b[2]-a[2]),lowY:a[1],broken:r.flags[0]&1};
  }
  const pass={
    hangStrong:out.hangStrong.broken===0&&Math.abs(out.hangStrong.y-3)<5e-4,
    hangWeak:out.hangWeak.broken===1&&out.hangWeak.y<2,
    restingCompression:out.restingCompression.broken===0&&Math.abs(out.restingCompression.y-H)<0.01,
    cantileverStrong:out.cantileverStrong.broken===0&&out.cantileverStrong.tiltDeg<0.2&&Math.abs(out.cantileverStrong.y-2)<5e-4,
    cantileverWeak:out.cantileverWeak.broken===1,
    // Released from rest, the bonded pair must fall and land (no Gauss-Seidel lock).
    bondedPair:out.bondedPair.broken===0&&Math.abs(out.bondedPair.gap-2*H)<5e-4&&out.bondedPair.dx<0.001&&Math.abs(out.bondedPair.lowY-H)<0.02,
  };
  return {pass,ok:Object.values(pass).every(Boolean),out};
}
