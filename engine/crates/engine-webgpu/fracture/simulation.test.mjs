// GPU-free smoke test of the browser host: drive simulation.mjs with a mock
// WebGPU device that records every call, using the real exported WGSL
// (cargo run -p engine-webgpu --bin wgsl_export). Checks that every pipeline
// exists and fits the storage budget, every binding is backed by a large-enough
// buffer, and indirect slots stay inside the argument buffer.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,existsSync} from 'node:fs';
import {createSimulation} from './simulation.mjs';
import {HullArena,appendHull} from './geometry.mjs';
import {prism} from './convex-regression.mjs';

const wgslDir=new URL('../generated/wgsl/',import.meta.url);
const haveWgsl=existsSync(new URL('convex_classify_persistent.wgsl',wgslDir));

function mockDevice(){
  const log={dispatches:[],buffers:[],pipelines:new Map(),errors:[]};
  globalThis.GPUBufferUsage={MAP_READ:1,MAP_WRITE:2,COPY_SRC:4,COPY_DST:8,INDEX:16,VERTEX:32,UNIFORM:64,STORAGE:128,INDIRECT:256,QUERY_RESOLVE:512};
  globalThis.GPUShaderStage={VERTEX:1,FRAGMENT:2,COMPUTE:4};
  globalThis.GPUMapMode={READ:1,WRITE:2};
  globalThis.fetch=async url=>{const path=new URL(String(url).replace('../generated/wgsl/',''),wgslDir);
    return existsSync(path)?{ok:true,text:async()=>readFileSync(path,'utf8')}:{ok:false}};
  const buffer=desc=>{const b={...desc,destroyed:false,destroy(){this.destroyed=true},
    async mapAsync(){},getMappedRange(){const r=new ArrayBuffer(this.size);if(this.size===80)new Uint32Array(r)[5]=1;return r},unmap(){}};log.buffers.push(b);return b};
  const pass=()=>{let pipe=null,bound=null;return{
    setPipeline(p){pipe=p},setBindGroup(i,g,offsets){bound={g,offsets}},
    dispatchWorkgroups(x,y=1,z=1){log.dispatches.push({name:pipe.label,x,y,z,bound})},
    dispatchWorkgroupsIndirect(buf,offset){log.dispatches.push({name:pipe.label,indirect:true,buf,offset,bound})},end(){}}};
  const device={
    limits:{maxStorageBufferBindingSize:2**31,maxBufferSize:2**31},
    features:new Set(),
    createBuffer:buffer,
    createShaderModule:({code,label})=>({code,label,async getCompilationInfo(){return {messages:[]}}}),
    createBindGroupLayout:d=>({...d}),
    createPipelineLayout:d=>({...d}),
    createComputePipeline:d=>{log.pipelines.set(d.label,d);return {label:d.label,desc:d}},
    createBindGroup:d=>({...d}),
    pushErrorScope(){},async popErrorScope(){return null},
    createCommandEncoder:()=>({beginComputePass:pass,copyBufferToBuffer(){},clearBuffer(){},resolveQuerySet(){},finish(){return {}}}),
    queue:{writeBuffer(){},submit(){}},
  };
  return {device,log};
}

function scene(n=40){
  const arena=new HullArena(),bodies=[{p:[0,0,0],q:[0,0,0,1],invMass:0,invI:[0,0,0],flags:0}],colliders=[{body:0,shape:4,params:[0,1,0,0],density:0}];
  for(let i=0;i<n;i++){
    bodies.push({p:[(i%50)*.9,1+Math.floor(i/50)*1.2,0],q:[0,0,0,1],invMass:1,invI:[1,1,1],flags:9});
    const h=prism(4+i%20,.5,1),c=appendHull(arena,h.vertices,h.topology);
    colliders.push({body:i+1,shape:3,params:c.params,localPos:c.localPos,density:1});
  }
  bodies.push({p:[0,3,0],q:[0,0,0,1],invMass:1,invI:[1,1,1],flags:9});colliders.push({body:bodies.length-1,shape:0,params:[.3,0,0,0],density:1});
  return {bodies,colliders,hullData:arena.data,params:{dt:1/120,iterations:4,gravityY:-9.81,flags:1}};
}

test('browser host schedules every core + convex kernel against valid bindings',{skip:!haveWgsl&&'run wgsl_export first'},async()=>{
  const {device,log}=mockDevice(),sim=await createSimulation(device,scene());
  await sim.step(2);
  await sim.inspectCollisions();
  sim.reset();await sim.step(1);
  const names=new Set(log.dispatches.map(d=>d.name));
  for(const k of ['convex_classify','convex_write_args','convex_sat_small','convex_sat_large','convex_manifold','record_color_degrees','csr_prefix','count_color_tiles','pack_solver_contacts','solve_primal_color','solve_dual','write_sleep_args'])
    assert.ok(names.has(k),`missing dispatch ${k}`);
  const argsBuffer=log.buffers.find(b=>b.label==='AVBD 23');
  for(const d of log.dispatches){
    if(d.indirect){assert.equal(d.buf,argsBuffer);assert.ok(d.offset+12<=argsBuffer.size,`${d.name} slot ${d.offset/12} outside args`)}
    else assert.ok(d.x>=1&&d.x<=65535&&d.y>=1&&d.y<=65535,`${d.name} groups ${d.x}x${d.y}`);
    for(const e of d.bound.g.entries){const b=e.resource.buffer;assert.ok(b&&!b.destroyed,`${d.name} binding ${e.binding}`)}
  }
  for(const [name,p] of log.pipelines){
    const code=p.compute.module.code,storage=(code.match(/var<storage/g)||[]).length;
    assert.ok(storage<=10,`${name}: ${storage} storage buffers`);
    assert.equal(p.compute.entryPoint,code.match(/@compute[^\n]*\n\s*fn (\w+)/)?.[1]??p.compute.entryPoint);
  }
  // Every binding of every pipeline is a buffer at least as large as the WGSL needs per element.
  for(const d of log.dispatches)for(const e of d.bound.g.entries)assert.ok((e.resource.size??e.resource.buffer.size)>=4);
  sim.dispose();
});

test('bodies above 512 use the three-dispatch block CSR scan',{skip:!haveWgsl&&'run wgsl_export first'},async()=>{
  const {device,log}=mockDevice(),sim=await createSimulation(device,scene(600));
  await sim.step(1);
  const names=new Set(log.dispatches.map(d=>d.name));
  for(const k of ['csr_prefix_blocks','csr_prefix_block_sums','csr_prefix_add'])assert.ok(names.has(k),k);
  assert.ok(!names.has('csr_prefix'));
  const profile=await sim.profileStep();assert.equal(profile.supported,false);
  sim.dispose();
});

test('whole-step manifold reuse, hull growth and collider swaps schedule correctly',{skip:!haveWgsl&&'run wgsl_export first'},async()=>{
  const {device,log}=mockDevice(),sc=scene(),sim=await createSimulation(device,sc);
  log.dispatches.length=0;await sim.step(1,{reuseManifolds:true});
  assert.ok(log.dispatches.some(d=>d.name==='broadphase_query'),'first step cannot reuse');
  log.dispatches.length=0;await sim.step(1,{reuseManifolds:true});
  const names=log.dispatches.map(d=>d.name);
  assert.ok(names.includes('reuse_all_manifolds'));
  for(const k of ['build_aabbs','broadphase_query','narrowphase','convex_sat_large'])assert.ok(!names.includes(k),`${k} skipped while reusing`);
  // Grow the hull arena past the buffer: a new buffer replaces binding 18 everywhere.
  const arena=new HullArena();let child;
  for(let i=0;i<60;i++)child=appendHull(arena,prism(20+i%12,.3+i*.01,.6).vertices,prism(20+i%12,.3+i*.01,.6).topology);
  const before=log.buffers.filter(b=>b.label==='AVBD 18');sim.syncHulls(arena);
  const after=log.buffers.filter(b=>b.label==='AVBD 18');
  assert.equal(after.length,before.length+1,'hull buffer grew');assert.ok(before[0].destroyed);
  sim.setCollider(1,{...sc.colliders[1],params:child.params,localPos:child.localPos,filter:{categories:1,mask:1,group:0}});
  log.dispatches.length=0;await sim.step(1,{reuseManifolds:true});
  assert.ok(log.dispatches.some(d=>d.name==='broadphase_query'),'scene change forces a full step');
  for(const d of log.dispatches)for(const e of d.bound.g.entries)assert.ok(!e.resource.buffer.destroyed,`${d.name} still bound to a destroyed buffer`);
  sim.dispose();
});
