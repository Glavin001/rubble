import {appendHull,colliderBounds,gridConfig,add,rotate,cross,sub,dot,norm} from './geometry.mjs';
import {createSimulation} from './simulation.mjs';
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
function world(scene,c){const body=scene.bodies[c.body],d=scene.hullData,base=c.params[0],nv=d[base*4],nf=d[base*4+1],ne=Math.abs(d[base*4+2]);
 const vertex=i=>add(body.p,rotate(body.q,Array.from(d.slice((base+1+i)*4,(base+1+i)*4+3))));
 return {vertices:Array.from({length:nv},(_,i)=>vertex(i)),normals:Array.from({length:nf},(_,i)=>rotate(body.q,Array.from(d.slice((base+1+nv+i)*4,(base+1+nv+i)*4+3)))),edges:Array.from({length:ne},(_,i)=>{const at=(base+1+nv+2*nf+2*i)*4;return sub(vertex(d[at+1]),vertex(d[at]))})};
}
function sat(a,b){for(const n of [...a.normals,...b.normals,...a.edges.flatMap(e=>b.edges.map(f=>cross(e,f)))]){
 const l=Math.hypot(...n);if(l<1e-10)continue;const axis=n.map(x=>x/l),pa=a.vertices.map(v=>dot(v,axis)),pb=b.vertices.map(v=>dot(v,axis));
 if(Math.max(Math.min(...pb)-Math.max(...pa),Math.min(...pa)-Math.max(...pb))>1e-5)return false;
 }return true;
}
function fixture(count=160,sides=4,scaleTest=false){
 const bodies=[],colliders=[],data=[];let seed=123;
 const random=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/2**32);
 for(let i=0;i<count;i++){
  const size=scaleTest?1:[.1,.5,2][i%3],group=Math.floor(i/2),active=scaleTest||i%5!==0;
  const q=scaleTest?norm([random()*.15,random()*.2,random()*.1,1]):norm(Array.from({length:4},()=>random()-.5));
  const p=scaleTest?[(i%40)*2.01,1+Math.floor(i/1600)*2.01,Math.floor(i/40)%40*2.01]:[group*15+(i%2)*size*(group%4===0?.3:1.5+random()*2),3,(i%2)*size*(random()-.5)];
  const fixed=!scaleTest&&i%11===0;
  bodies.push({p,q,invMass:fixed?0:1,invI:fixed?[0,0,0]:[1,1,1],flags:fixed?0:active?9:5});
  const h=sides===62?bipyramid(size):prism(sides,size,2*size);
  const params=appendHull(data,h.vertices,h.topology);colliders.push({body:i,shape:3,params,density:1});
  // Additional hull on the same rigid body, far from its COM. Never self-collide.
  if(!scaleTest&&i%7===0){const params=appendHull(data,h.vertices.map(v=>add(v,[0,0,4])),h.topology);colliders.push({body:i,shape:3,params,density:1})}
 }
 return {bodies,colliders,hullData:new Float32Array(data),gridCellSize:gridConfig(colliders).base,params:{dt:1/120,iterations:0,gravityY:0},projectileCount:0};
}
export async function runConvexGpuTests({scale=10000}={}){
 const adapter=await navigator.gpu.requestAdapter(),device=await adapter.requestDevice({requiredFeatures:adapter.features.has('timestamp-query')?['timestamp-query']:[],requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(16,adapter.limits.maxStorageBuffersPerShaderStage),maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize,maxBufferSize:adapter.limits.maxBufferSize}});
 const errors=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message));const results=[];
 try{
 for(const [count,sides] of [[520,4],[64,32],[24,62]]){
  const scene=fixture(count,sides),sim=await createSimulation(device,scene);
  try{
   const result=await sim.inspectCollisions(),actual=new Set();
   for(let k=0;k<result.contacts.length;k+=32){const ids=new Uint32Array(result.contacts.buffer,k*4,2);actual.add(`${ids[0]},${ids[1]}`)}
   const pairs=new Set();for(let k=0;k<result.pairs.length;k+=4){const key=`${result.pairs[k]},${result.pairs[k+1]}`;require(!pairs.has(key),'duplicate candidate');pairs.add(key)}
   const bounds=scene.colliders.map(c=>colliderBounds(scene,c)),shapes=scene.colliders.map(c=>world(scene,c));let expected=0;
   for(let i=0;i<scene.colliders.length;i++){
    const a=scene.colliders[i],ga=result.bounds.slice(i*8,i*8+8);
    for(let k=0;k<3;k++){require(Math.abs(ga[k]-bounds[i].lo[k])<.002,'wrong GPU minimum');require(Math.abs(ga[k+4]-bounds[i].hi[k])<.002,'wrong GPU maximum')}
    for(let j=i+1;j<scene.colliders.length;j++){
     const b=scene.colliders[j],moving=i=>(scene.bodies[i].flags&1)&&!(scene.bodies[i].flags&4);
     if(a.body===b.body||(!moving(a.body)&&!moving(b.body)))continue;
     if(!bounds[i].lo.every((x,k)=>x<=bounds[j].hi[k]&&bounds[j].lo[k]<=bounds[i].hi[k]))continue;
     if(sat(shapes[i],shapes[j])){expected++;require(pairs.has(`${i},${j}`),`missed broadphase ${i},${j}`);require(actual.has(`${i},${j}`),`missed contact ${i},${j}`)}
     else require(!actual.has(`${i},${j}`),`false contact ${i},${j}`);
    }
   }
   require(actual.size===expected,`extra contacts ${actual.size} vs ${expected}`);
   results.push({bodies:count,vertices:sides===62?64:sides*2,expectedPairs:expected,counters:result.counters});
  }finally{sim.dispose()}
 }
 // A compound frame must leave its aperture open. One sphere is in the hole,
 // another intersects a jamb: body-level spherical/convex proxies fail this.
 {
  const data=[],colliders=[],h=prism(4),bodies=[{p:[0,0,0],q:[0,0,0,1],invMass:0,invI:[0,0,0],flags:0}];
  for(const [center,half] of [[[0,2,0],[2,.2,.4]],[[0,-2,0],[2,.2,.4]],[[-2,0,0],[.2,2,.4]],[[2,0,0],[.2,2,.4]]]){
   // Rotate the square cross-section 45 degrees into an axis-aligned box.
   const vertices=h.vertices.map(v=>add(center,[(v[0]+v[2])*half[0],v[1]*half[1],(v[0]-v[2])*half[2]]));
   colliders.push({body:0,shape:3,params:appendHull(data,vertices,h.topology),density:1});
  }
  for(const p of [[0,0,0],[2,0,0]]){const body=bodies.length;bodies.push({p,q:[0,0,0,1],invMass:1,invI:[1,1,1],flags:9});colliders.push({body,shape:0,params:[.15,0,0,0],density:1})}
  const scene={bodies,colliders,hullData:new Float32Array(data),params:{dt:1/120,iterations:0,gravityY:0},gridCellSize:gridConfig(colliders).base};
  const sim=await createSimulation(device,scene);try{
   const r=await sim.inspectCollisions();let jamb=false;
   for(let k=0;k<r.contacts.length;k+=32){const ids=new Uint32Array(r.contacts.buffer,k*4,2);require(!ids.includes(4),'compound aperture falsely blocked');if(ids.includes(5))jamb=true}
   require(jamb,'sphere did not contact the compound jamb');results.push({test:'compound aperture',passed:true});
  }finally{sim.dispose()}
 }
 // Compare queue-batched steps against individual submissions with contact
 // reuse enabled. This catches stale previous-contact counts/parity in a batch.
 {
  const scene=fixture(2,32);scene.params={...scene.params,flags:1,iterations:4};
  scene.bodies.forEach(b=>{b.flags=9;b.invMass=1;b.invI=[1,1,1]});
  const sim=await createSimulation(device,scene);try{
   for(let i=0;i<8;i++)await sim.step();const separate=await sim.snapshot();
   sim.reset();await sim.step(4);await sim.step(4);const batched=await sim.snapshot();
   let error=0;for(let i=0;i<scene.bodies.length;i++)for(const k of [0,1,2,4,5,6,7])error=Math.max(error,Math.abs(separate[i*28+k]-batched[i*28+k]));
   require(batched.every(Number.isFinite)&&error<.002,`batched contact reuse diverged: ${error}`);
   require(sim.counters.overflow===0,'batched contact overflow');results.push({test:'batched contact reuse',maxPoseError:error,passed:true});
  }finally{sim.dispose()}
 }
 if(scale){
  const scene=fixture(scale,32,true);scene.params.iterations=8;
  const sim=await createSimulation(device,scene);try{
   await sim.step();const start=performance.now();for(let i=0;i<5;i++)await sim.step();
   const wallMs=(performance.now()-start)/5,poses=await sim.snapshot();require(poses.every(Number.isFinite),'nonfinite scale state');
   results.push({bodies:scale,verticesPerHull:64,active:sim.counters.awake,wallMs,counters:sim.counters,profile:await sim.profileStep()});
  }finally{sim.dispose()}
 }
 require(!errors.length,errors.join('\n'));return {passed:true,adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture},results};
 }finally{device.destroy()}
}
