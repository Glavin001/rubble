import {HullArena,appendHull,colliderBounds,colliderVertices,hullView,hullBase,cuboidHull,add,sub,scale,rotate,cross,dot,norm} from './geometry.mjs';
import {createSimulation} from './simulation.mjs';
// GPU conformance and scale checks for the core convex path (hull/hull, hull/box,
// hull/plane, hull/sphere, hull/capsule). CPU oracles below are independent of the
// Slang code: exhaustive SAT with full projections, polygon closest points and
// segment sampling. Run in a WebGPU page: runConvexGpuTests({scale:10000}).

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
const require=(ok,msg)=>{if(!ok)throw new Error(msg)};
const MARGIN=.01; // simulation.mjs contactOffset
const AMBIGUOUS=2e-4; // |separation| band where f32 GPU and f64 oracle may disagree

// ---- CPU oracles -----------------------------------------------------------------
// World polytope for a hull or box collider: vertices, face planes, edge vectors.
function polytope(scene,c){
 const body=scene.bodies[c.body],q=body.q||[0,0,0,1],lq=c.localRot||[0,0,0,1];
 if(c.shape===2){
  const box=cuboidHull(c.params.slice(0,3));const lp=c.localPos||[0,0,0];
  const vertices=box.vertices.map(v=>add(body.p,rotate(q,add(lp,rotate(lq,v)))));
  return {vertices,normals:box.faces.map(f=>rotate(q,rotate(lq,f.n))),edges:box.edges.map(([a,b])=>sub(vertices[b],vertices[a]))};
 }
 const view=hullView(scene.hullData,hullBase(c.params)),vertices=colliderVertices(scene,c),s=c.params[1]||1;
 return {vertices,normals:Array.from({length:view.nf},(_,k)=>rotate(q,rotate(lq,scale(view.plane(k).n,1/s)))).map(norm),
  faces:Array.from({length:view.nf},(_,k)=>view.face(k)),edges:Array.from({length:view.ne},(_,k)=>{const [a,b]=view.edge(k);return sub(vertices[b],vertices[a])})};
}
// Largest separation over every face normal and edge-pair axis (full projections).
function satOracle(a,b){
 let best=-Infinity,axis=null;
 const ca=scale(a.vertices.reduce((s,v)=>add(s,v),[0,0,0]),1/a.vertices.length),cb=scale(b.vertices.reduce((s,v)=>add(s,v),[0,0,0]),1/b.vertices.length);
 for(const n of [...a.normals,...b.normals,...a.edges.flatMap(e=>b.edges.map(f=>cross(e,f)))]){
  const l=Math.hypot(...n);if(l<1e-9)continue;let x=n.map(v=>v/l);if(dot(x,sub(cb,ca))<0)x=x.map(v=>-v);
  const sep=sepAlong(a,b,x);if(sep>best){best=sep;axis=x}
 }
 return {sep:best,axis};
}
function sepAlong(a,b,n){
 const pa=a.vertices.map(v=>dot(v,n)),pb=b.vertices.map(v=>dot(v,n));
 return Math.max(Math.min(...pb)-Math.max(...pa),Math.min(...pa)-Math.max(...pb));
}
// Distance from p to a convex polytope (0 inside), via its face polygons.
function pointPolytope(poly,p){
 const inside=poly.normals.every((n,k)=>dot(n,sub(p,poly.vertices[poly.faces?poly.faces[k][0]:0]))<=0);
 if(poly.faces&&inside)return 0;
 let best=Infinity;
 const faces=poly.faces||[];
 for(let k=0;k<faces.length;k++){
  const ids=faces[k],n=poly.normals[k],v0=poly.vertices[ids[0]],s=dot(n,sub(p,v0));if(s<=0)continue;
  const q=sub(p,scale(n,s));let inPoly=true,d=Infinity;
  for(let i=0;i<ids.length;i++){const a=poly.vertices[ids[i]],b=poly.vertices[ids[(i+1)%ids.length]],e=sub(b,a);
   if(dot(cross(e,n),sub(q,a))>0){inPoly=false;const t=Math.max(0,Math.min(1,dot(sub(p,a),e)/dot(e,e)));d=Math.min(d,Math.hypot(...sub(p,add(a,scale(e,t)))))}}
  best=Math.min(best,inPoly?s:d);
 }
 return best;
}
const moving=(scene,i)=>(scene.bodies[i].flags&1)&&!(scene.bodies[i].flags&4);
const overlaps=(a,b)=>a.lo.every((x,k)=>x<=b.hi[k]&&b.lo[k]<=a.hi[k]);

// GPU contact records -> pair set and per-pair manifold summaries.
function gpuContacts(result){
 const pairs=new Map();
 for(let k=0;k<result.contacts.length;k+=8){
  const a=result.contacts[k],b=result.contacts[k+1],f=new Float32Array(result.contacts.buffer,(k+4)*4,4);
  const pt=new Float32Array(result.points.buffer,(k/8)*64,16);
  const key=`${a},${b}`;if(!pairs.has(key))pairs.set(key,{normal:[f[0],f[1],f[2]],c0:[]});
  pairs.get(key).c0.push(pt[3]);
 }
 return pairs;
}

// ---- fixtures ----------------------------------------------------------------------
function hullFixture(count=160,sides=4,scaleTest=false){
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
function compoundFixture(count,children=4,spacing=2.2){
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
function chainFixture(pairs,sides){
 const arena=new HullArena(),bodies=[],colliders=[],h=prism(sides,.5,1);
 for(let i=0;i<=pairs;i++){
  bodies.push({p:[i*.95,1,0],q:[0,0,0,1],invMass:1,invI:[1,1,1],flags:9});
  const c=appendHull(arena,h.vertices,h.topology);colliders.push({body:i,shape:3,params:c.params,localPos:c.localPos,density:1});
 }
 return {bodies,colliders,hullData:arena.data,params:{dt:1/120,iterations:0,gravityY:0},projectileCount:0};
}

// ---- checks ------------------------------------------------------------------------
async function conformance(device,scene,label){
 const sim=await createSimulation(device,scene);
 try{
  const result=await sim.inspectCollisions(),actual=gpuContacts(result),pairs=new Set();
  for(let k=0;k<result.pairs.length;k+=4){const key=`${result.pairs[k]},${result.pairs[k+1]}`;require(!pairs.has(key),`${label}: duplicate candidate ${key}`);pairs.add(key)}
  const bounds=scene.colliders.map(c=>colliderBounds(scene,c));let expected=0,checkedDepth=0,worstAxis=0;
  for(let i=0;i<scene.colliders.length;i++){
   const a=scene.colliders[i];
   if(a.shape===3){const ga=result.bounds.slice(i*8,i*8+8);for(let k=0;k<3;k++){require(Math.abs(ga[k]-bounds[i].lo[k])<.002,`${label}: GPU min ${i}`);require(Math.abs(ga[k+4]-bounds[i].hi[k])<.002,`${label}: GPU max ${i}`)}}
   for(let j=i+1;j<scene.colliders.length;j++){
    const b=scene.colliders[j];
    if(a.body===b.body||(!moving(scene,a.body)&&!moving(scene,b.body)))continue;
    if(a.shape!==3&&b.shape!==3)continue;
    let sep;
    const key=`${i},${j}`;
    if(a.shape===4||b.shape===4){const h=a.shape===4?b:a;sep=Math.min(...polytope(scene,h).vertices.map(v=>v[1]))}
    else{
     if(!overlaps(bounds[i],bounds[j]))continue;
     const [h,o]=a.shape===3?[a,b]:[b,a],ph=polytope(scene,h);
     if(o.shape===0){const c=add(scene.bodies[o.body].p,rotate(scene.bodies[o.body].q,o.localPos||[0,0,0]));sep=pointPolytope(ph,c)-o.params[0]}
     else if(o.shape===1){
      const body=scene.bodies[o.body],axis=rotate(body.q,[0,o.params[1],0]),p0=sub(body.p,axis),p1=add(body.p,axis);
      sep=Infinity;for(let s=0;s<=64;s++)sep=Math.min(sep,pointPolytope(ph,add(p0,scale(sub(p1,p0),s/64)))-o.params[0]);
     }else{
      const pa=polytope(scene,a),pb=polytope(scene,b),sat=satOracle(pa,pb);sep=sat.sep;
      // Manifold quality: the GPU normal must be (near) a minimal-penetration
      // axis; face bias allows edges up to 0.95·face + 0.01 (CONVEX_EDGE_*).
      const got=actual.get(key);
      if(got&&sep<-AMBIGUOUS){
       const n=got.normal;require(Math.abs(Math.hypot(...n)-1)<1e-3,`${label}: non-unit normal ${key}`);
       const along=sepAlong(pa,pb,n);worstAxis=Math.min(worstAxis,along-sat.sep);
       require(along>=sat.sep-.012,`${label}: normal ${key} penetrates ${along} vs best ${sat.sep}`);
       const deepest=Math.min(...got.c0)-MARGIN;require(Math.abs(deepest-along)<.02+.05*Math.abs(along),`${label}: depth ${key} ${deepest} vs ${along}`);
       checkedDepth++;
      }
     }
    }
    if(Math.abs(sep)<AMBIGUOUS)continue;
    if(sep<0){expected++;require(actual.has(key),`${label}: missed contact ${key} (sep ${sep})`);if(a.shape!==4&&b.shape!==4)require(pairs.has(key),`${label}: missed broadphase ${key}`)}
    else require(!actual.has(key),`${label}: false contact ${key} (sep ${sep})`);
   }
  }
  return {label,expectedPairs:expected,gpuPairs:actual.size,checkedDepth,worstAxis,counters:result.counters};
 }finally{sim.dispose()}
}

async function timeScene(device,scene,steps=5){
 const sim=await createSimulation(device,scene);
 try{
  await sim.step();const start=performance.now();for(let i=0;i<steps;i++)await sim.step();
  const wallMs=(performance.now()-start)/steps,poses=await sim.snapshot();require(poses.every(Number.isFinite),'nonfinite scale state');
  const counters=sim.counters;require(counters.overflow===0,`scale overflow 0x${counters.overflow.toString(16)}`);
  return {bodies:scene.bodies.length,colliders:scene.colliders.length,wallMs,counters,
   profile:await sim.profileStep(),singlePass:await sim.profileStep({singlePass:true})};
 }finally{sim.dispose()}
}

export async function runConvexGpuTests({scale=10000,boundary=false}={}){
 const adapter=await navigator.gpu.requestAdapter(),device=await adapter.requestDevice({requiredFeatures:adapter.features.has('timestamp-query')?['timestamp-query']:[],requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(10,adapter.limits.maxStorageBuffersPerShaderStage),maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize,maxBufferSize:adapter.limits.maxBufferSize}});
 const errors=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message));const results=[];
 try{
  for(const [count,sides] of [[520,4],[64,32],[24,62]])results.push(await conformance(device,hullFixture(count,sides),`hulls ${count}x${sides===62?64:sides*2}v`));
  results.push(await conformance(device,mixedFixture(),'mixed box/sphere/capsule/plane'));
  results.push(await conformance(device,compoundFixture(64,4,1.3),'dense compounds'));
  // A compound frame must leave its aperture open. One sphere is in the hole,
  // another intersects a jamb: body-level spherical/convex proxies fail this.
  {
   const arena=new HullArena(),colliders=[],h=prism(4),bodies=[{p:[0,0,0],q:[0,0,0,1],invMass:0,invI:[0,0,0],flags:0}];
   for(const [center,half] of [[[0,2,0],[2,.2,.4]],[[0,-2,0],[2,.2,.4]],[[-2,0,0],[.2,2,.4]],[[2,0,0],[.2,2,.4]]]){
    const vertices=h.vertices.map(v=>add(center,[(v[0]+v[2])*half[0],v[1]*half[1],(v[0]-v[2])*half[2]]));
    const c=appendHull(arena,vertices,h.topology);colliders.push({body:0,shape:3,params:c.params,localPos:c.localPos,density:1});
   }
   for(const p of [[0,0,0],[2,0,0]]){const body=bodies.length;bodies.push({p,q:[0,0,0,1],invMass:1,invI:[1,1,1],flags:9});colliders.push({body,shape:0,params:[.15,0,0,0],density:1})}
   const scene={bodies,colliders,hullData:arena.data,params:{dt:1/120,iterations:0,gravityY:0}};
   const sim=await createSimulation(device,scene);try{
    const r=await sim.inspectCollisions();let jamb=false;
    for(let k=0;k<r.contacts.length;k+=8){const ids=[r.contacts[k],r.contacts[k+1]];require(!ids.includes(4),'compound aperture falsely blocked');if(ids.includes(5))jamb=true}
    require(jamb,'sphere did not contact the compound jamb');results.push({test:'compound aperture',passed:true});
   }finally{sim.dispose()}
  }
  // SAT work-list sizing: empty, one pair, and both buckets around row limits.
  const workCases=[[0,4],[1,4],[1,32],...(boundary?[[32768,32],[32769,32],[131073,4]]:[])];
  for(const [pairs,sides] of workCases){
   const scene=chainFixture(pairs,sides),sim=await createSimulation(device,scene);
   try{const r=await sim.inspectCollisions(),got=gpuContacts(r).size;
    require(got===pairs,`work list ${pairs}x${sides*2}v: ${got} manifolds`);
    require((sides*2<=16?r.counters.convexSmall:r.counters.convexLarge)===pairs,`work list ${pairs}: bucket count`);
    results.push({test:'work list',pairs,vertices:sides*2,passed:true});
   }finally{sim.dispose()}
  }
  // Batched vs individual submission with contact reuse (previous-count parity).
  {
   const scene=hullFixture(2,32);scene.params={...scene.params,flags:1,iterations:4};
   scene.bodies.forEach(b=>{b.flags=9;b.invMass=1;b.invI=[1,1,1]});
   const sim=await createSimulation(device,scene);try{
    for(let i=0;i<8;i++)await sim.step();const separate=await sim.snapshot();
    sim.reset();await sim.step(4);await sim.step(4);const batched=await sim.snapshot();
    let error=0;for(let i=0;i<scene.bodies.length;i++)for(const k of [0,1,2,4,5,6,7])error=Math.max(error,Math.abs(separate[i*28+k]-batched[i*28+k]));
    require(batched.every(Number.isFinite)&&error<.002,`batched contact reuse diverged: ${error}`);
    require(sim.counters.overflow===0,'batched contact overflow');results.push({test:'batched contact reuse',maxPoseError:error,passed:true});
   }finally{sim.dispose()}
  }
  // Settling with reuse/SAT hints on versus off: the cached-feature fast path
  // must not change resting behaviour beyond solver noise.
  {
   const run=async flags=>{const scene=mixedFixture();scene.params={dt:1/120,iterations:8,gravityY:-9.81,flags};const sim=await createSimulation(device,scene);
    try{for(let i=0;i<30;i++)await sim.step(4);require(sim.counters.overflow===0,'settle overflow');return await sim.snapshot()}finally{sim.dispose()}};
   const on=await run(1),off=await run(0);let worst=0;
   for(let i=0;i<on.length/28;i++)for(const k of [0,1,2])worst=Math.max(worst,Math.abs(on[i*28+k]-off[i*28+k]));
   require(on.every(Number.isFinite)&&off.every(Number.isFinite),'settle nonfinite');
   require(worst<.05,`reuse/hint path diverged from full SAT by ${worst} m`);results.push({test:'cached feature vs full SAT',maxPositionDelta:worst,passed:true});
  }
  if(scale){
   const dense=hullFixture(scale,32,true);dense.params.iterations=8;
   results.push({test:'dense 64-vertex prisms, reuse off',...await timeScene(device,dense)});
   const compounds=compoundFixture(scale,4,1.45);compounds.params={dt:1/120,iterations:8,gravityY:-9.81,flags:1};
   results.push({test:'compound bodies x4 children, reuse on',...await timeScene(device,compounds)});
   const rubble=hullFixture(scale*3,4,true);rubble.params={dt:1/120,iterations:8,gravityY:0,flags:1};
   results.push({test:'small rubble hulls',...await timeScene(device,rubble)});
  }
  require(!errors.length,errors.join('\n'));
  return {passed:true,adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,description:adapter.info.description},results};
 }finally{device.destroy()}
}
