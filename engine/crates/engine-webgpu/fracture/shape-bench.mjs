// Collision-cost benchmark scenes (headless-page scene=pairs): `count`
// separated stacks, each a static base of shape `a` with a dynamic body of
// shape `b` resting on it, sleep disabled so every pair stays in contact.
// Measuring two counts and taking the difference gives the marginal GPU cost
// of one live pair of that shape combination.
//
// Shapes: box, sphere, capsule, hull<N> (an N-gon prism, 2N vertices, N = 3-32),
// box4 (a compound body: four boxes tiling the same footprint as one box).
import {HullArena,gridConfig,appendHull} from './geometry.mjs';

const HALF=[0.5,0.25,0.5];   // box half extents (1 x 0.5 x 1 m slab)
const RADIUS=0.25;           // sphere radius; capsule radius and half length

// N-gon prism circumscribing the box footprint, same height as the box.
function prism(n){
  const r=0.5/Math.cos(Math.PI/n),v=[],faces=[];
  for(let i=0;i<n;i++){const t=(i+0.5)/n*2*Math.PI;v.push([r*Math.cos(t),-HALF[1],r*Math.sin(t)])}
  for(let i=0;i<n;i++){const [x,,z]=v[i];v.push([x,HALF[1],z])}
  faces.push([...Array(n).keys()]);                 // bottom
  faces.push([...Array(n).keys()].map(i=>n+i).reverse()); // top
  for(let i=0;i<n;i++){const j=(i+1)%n;faces.push([i,n+i,n+j,j])}
  return {v,faces};
}

function colliderFor(shape,body,arena){
  if(shape==='box')return [{body,shape:2,density:1000,params:[...HALF,0],material:0}];
  if(shape==='sphere')return [{body,shape:0,density:1000,params:[RADIUS,0,0,0],material:0}];
  if(shape==='capsule')return [{body,shape:1,density:1000,params:[RADIUS,RADIUS,0,0],material:0,localRot:[0,0,Math.SQRT1_2,Math.SQRT1_2]}];
  if(shape==='box4'){
    const h=[HALF[0]/2,HALF[1],HALF[2]/2],out=[];
    for(const sx of [-1,1])for(const sz of [-1,1])out.push({body,shape:2,density:1000,params:[...h,0],localPos:[sx*h[0],0,sz*h[2]],material:0});
    return out;
  }
  const m=/^hull(\d+)$/.exec(shape);if(!m)throw new Error(`Unknown bench shape ${shape}`);
  const {v,faces}=prism(Number(m[1])),c=appendHull(arena,v,{sizes:faces.map(f=>f.length),indices:faces.flat()});
  return [{body,shape:3,density:1000,params:c.params,localPos:c.localPos,material:0}];
}

// Half height above the base top at which shape b rests.
const restHalf=shape=>shape==='sphere'||shape==='capsule'?RADIUS:HALF[1];

export function buildPairScene({a='box',b='box',count=4096,spacing=2.5}={}){
  const arena=new HullArena(),bodies=[],colliders=[];
  const side=Math.ceil(Math.sqrt(count));
  for(let k=0;k<count;k++){
    const x=(k%side-side/2)*spacing,z=(Math.floor(k/side)-side/2)*spacing;
    const base=bodies.length;
    bodies.push({p:[x,HALF[1],z],q:[0,0,0,1],invMass:0,invI:[0,0,0],flags:0});
    colliders.push(...colliderFor(a,base,arena));
    // Every base shape reaches y = 2 * HALF[1] (= 2 * RADIUS) at its top.
    const top=bodies.length,m=1000*8*HALF[0]*HALF[1]*HALF[2];
    const baseTop=2*HALF[1];
    bodies.push({p:[x,baseTop+restHalf(b),z],q:[0,0,0,1],invMass:1/m,
      invI:[3/(m*(HALF[1]**2+HALF[2]**2)),3/(m*(HALF[0]**2+HALF[2]**2)),3/(m*(HALF[0]**2+HALF[1]**2))],flags:1});
    colliders.push(...colliderFor(b,top,arena));
  }
  const extent=side*spacing;
  return {bodies,colliders,joints:[],hullData:arena.data,skippedHulls:[],
    projectileStart:bodies.length,projectileCount:0,projectileRadius:0.25,projectileVertices:new Float32Array(0),
    vertices:new Float32Array(0),lines:new Float32Array(0),selected:[],bonds:[],byId:new Map(),
    extent,extentX:extent,extentZ:extent,height:1,copies:1,instanceStride:0,center:[0,0.5,0],
    gridCellSize:gridConfig(colliders).base,capacity:{candidatesPerCollider:8,contactsPerCollider:16},
    // Flags: 64 world-frame inertia (+1 contact reuse when reuse=1). Sleep off.
    params:{flags:64,iterations:8,dt:1/120,gravityY:-9.81,contactOffset:.002,alpha:.95,sleepLin:-1,sleepAng:-1,sleepTime:1e9}};
}
