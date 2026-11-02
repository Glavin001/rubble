import {HullArena,appendHull} from './wasm.mjs';
import {add,norm} from './vec.mjs';
// Convex test fixtures (hull prisms and bipyramids, hull fields, compounds, chains) for the
// unit tests and for building GPU scenes by hand. The GPU conformance checks that ran here
// on the old JavaScript host (its collision-only pass, `inspectCollisions`, against CPU SAT
// oracles) went with that host: World's narrowphase runs only inside a step. The same
// kernels are checked natively (engine-webgpu's tests, the Metal viewer's fracture
// regressions).

export function prism(sides=4,radius=1,height=2){
 const vertices=[];for(const y of [-height/2,height/2])for(let i=0;i<sides;i++)vertices.push([radius*Math.cos(i*2*Math.PI/sides),y,radius*Math.sin(i*2*Math.PI/sides)]);
 const faces=[Array.from({length:sides},(_,i)=>i),Array.from({length:sides},(_,i)=>sides+i)];
 for(let i=0;i<sides;i++)faces.push([i,(i+1)%sides,(i+1)%sides+sides,i+sides]);
 return {vertices,topology:{sizes:faces.map(f=>f.length),indices:faces.flat()}};
}
export function bipyramid(radius=1){
 const vertices=Array.from({length:62},(_,i)=>[radius*Math.cos(i*2*Math.PI/62),0,radius*Math.sin(i*2*Math.PI/62)]);
 vertices.push([0,radius,0],[0,-radius,0]);const faces=[];
 for(let i=0;i<62;i++){faces.push([i,(i+1)%62,62],[(i+1)%62,i,63])}
 return {vertices,topology:{sizes:faces.map(f=>3),indices:faces.flat()}};
}
// ---- fixtures ----------------------------------------------------------------------
export function hullFixture(count=160,sides=4,scaleTest=false){
 const arena=new HullArena(),bodies=[],colliders=[];let seed=123;
 const random=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/2**32);
 for(let i=0;i<count;i++){
  const size=scaleTest?1:[.1,.5,2][i%3],group=Math.floor(i/2),active=scaleTest||i%5!==0;
  const q=scaleTest?norm([random()*.15,random()*.2,random()*.1,1]):norm(Array.from({length:4},()=>random()-.5));
  const p=scaleTest?[(i%40)*2.01,1+Math.floor(i/1600)*2.01,Math.floor(i/40)%40*2.01]:[group*15+(i%2)*size*(group%4===0?.3:1.5+random()*2),3,(i%2)*size*(random()-.5)];
  const fixed=!scaleTest&&i%11===0;
  bodies.push({p,q,invMass:fixed?0:1,invI:fixed?[0,0,0]:[1,1,1],flags:fixed?0:active?9:5});
  const h=sides===62?bipyramid(size):prism(sides,size,2*size);
  const child=appendHull(arena,h.vertices,h.topology);colliders.push({body:i,shape:3,params:child.params,localPos:child.localPos,density:1});
  // Additional child far from the body origin. Children never self-collide.
  if(!scaleTest&&i%7===0){const c=appendHull(arena,h.vertices.map(v=>add(v,[0,0,4])),h.topology);colliders.push({body:i,shape:3,params:c.params,localPos:c.localPos,density:1})}
 }
 return {bodies,colliders,hullData:arena.data,params:{dt:1/120,iterations:0,gravityY:0},projectileCount:0};
}
// Hulls next to boxes, spheres, capsules and a ground plane.
function mixedFixture(){
 const arena=new HullArena(),bodies=[{p:[0,0,0],q:[0,0,0,1],invMass:0,invI:[0,0,0],flags:0}],colliders=[{body:0,shape:4,params:[0,1,0,0],density:0}];
 let seed=7;const random=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/2**32);
 for(let i=0;i<60;i++){
  const x=(i%10)*3,z=Math.floor(i/10)*3,q=norm(Array.from({length:4},()=>random()-.5));
  const hull=prism(3+i%9,.5,1),id=bodies.length;bodies.push({p:[x,.4+random()*.3,z],q,invMass:1,invI:[1,1,1],flags:9});
  const c=appendHull(arena,hull.vertices,hull.topology);colliders.push({body:id,shape:3,params:c.params,localPos:c.localPos,density:1});
  const other=bodies.length,d=.7+random()*.6,pos=[x+d,.6,z+(random()-.5)*.4];bodies.push({p:pos,q:norm(Array.from({length:4},()=>random()-.5)),invMass:1,invI:[1,1,1],flags:9});
  if(i%3===0)colliders.push({body:other,shape:2,params:[.3,.4,.25,Math.hypot(.3,.4,.25)],density:1});
  else if(i%3===1)colliders.push({body:other,shape:0,params:[.35,0,0,0],density:1});
  else colliders.push({body:other,shape:1,params:[.2,.4,0,.6],density:1});
 }
 return {bodies,colliders,hullData:arena.data,params:{dt:1/120,iterations:0,gravityY:0},projectileCount:0};
}
// Compound bodies with several children each, densely packed (scale or quality).
export function compoundFixture(count,children=4,spacing=2.2){
 const arena=new HullArena(),bodies=[],colliders=[];let seed=99;
 const random=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/2**32);
 const side=Math.ceil(Math.cbrt(count));
 for(let i=0;i<count;i++){
  const p=[(i%side)*spacing,1+Math.floor(i/(side*side))*spacing,Math.floor(i/side)%side*spacing];
  bodies.push({p,q:norm([random()*.2,random()*.2,random()*.2,1]),invMass:1,invI:[1,1,1],flags:9});
  for(let k=0;k<children;k++){
   const h=prism(4+(k%5),.35,.7),off=[(k%2-.5)*.75,((k>>1)%2-.5)*.75,0];
   const c=appendHull(arena,h.vertices.map(v=>add(v,off)),h.topology);colliders.push({body:i,shape:3,params:c.params,localPos:c.localPos,density:1});
  }
 }
 return {bodies,colliders,hullData:arena.data,params:{dt:1/120,iterations:0,gravityY:0},projectileCount:0};
}
// A line of overlapping hulls: exactly `pairs` hull/hull contacts.
export function chainFixture(pairs,sides){
 const arena=new HullArena(),bodies=[],colliders=[],h=prism(sides,.5,1);
 for(let i=0;i<=pairs;i++){
  bodies.push({p:[i*.95,1,0],q:[0,0,0,1],invMass:1,invI:[1,1,1],flags:9});
  const c=appendHull(arena,h.vertices,h.topology);colliders.push({body:i,shape:3,params:c.params,localPos:c.localPos,density:1});
 }
 return {bodies,colliders,hullData:arena.data,params:{dt:1/120,iterations:0,gravityY:0},projectileCount:0};
}

