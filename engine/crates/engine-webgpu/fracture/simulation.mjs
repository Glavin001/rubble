import {Plan,sceneHandle} from './wasm.mjs';
// WebGPU executor of the fracture lab's host plan. Rust (engine-fracture
// plan.rs, through wasm) owns the buffer sizes, the host-uploaded words, the
// step state and the ordered dispatch schedule; this file creates the
// buffers and pipelines and replays each tick's commands. The Metal viewer
// (engine-viewer/src/fracture/sim.rs) replays the same commands.
const PASS_ALIGN=256,COUNTER_BYTES=80,UPLOAD_BYTES=1120,BODY_BYTES=112,JOINT_BYTES=288;
const OP_DISPATCH=0,OP_INDIRECT=1,OP_COPY=2,HOST_UPLOAD=255;
const KERNELS=Plan.kernels(),PHASES=Plan.phases();
// One step encoded as several command buffers, split after these phases. Submitted
// on successive frames they run the same dispatches in the same queue order as one
// submit, so results are identical; frames just stop waiting on whole steps.
const SLICE_AFTER=new Set(PHASES.filter(p=>p[1]).map(p=>p[0]));

export async function createSimulation(device, scene, onProgress = () => {}) {
const plan=new Plan(sceneHandle(scene),device.limits.maxStorageBufferBindingSize,device.limits.maxBufferSize);
const sizes=Array.from(plan.bufferSizes()),cap=plan.cap,bodyCount=plan.bodyCount,jointCount=plan.jointCount;
const bufs=sizes.map((size,i)=>device.createBuffer({label:`AVBD ${i}`,size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC|(Plan.isUniform(i)?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|(i===23?GPUBufferUsage.INDIRECT:0)}));
try {
const upload=()=>{device.queue.writeBuffer(bufs[0],0,plan.bodyBytes());device.queue.writeBuffer(bufs[2],0,plan.colliderBytes());device.queue.writeBuffer(bufs[5],0,plan.jointBytes())};
device.queue.writeBuffer(bufs[22],0,Plan.passTable());
device.queue.writeBuffer(bufs[3],0,plan.materialTable());
upload();
const hullData=scene.hullData??sceneHandle(scene).hullData();
device.queue.writeBuffer(bufs[18],0,hullData);
let hullRecordsCapacity=cap.maxHull;

const sources=[];
for(const [name,entry] of KERNELS){
  onProgress('compiling '+name);
  const response=await fetch(`../generated/wgsl/${entry}.wgsl`);
  if(!response.ok)throw new Error(`Missing ${entry}; run the shader export commands in fracture/README.md`);
  const src=await response.text();
  const module=device.createShaderModule({label:name,code:src});
  // Force frontend validation now so createComputePipeline cannot hang forever
  // on a deferred compile error (Chrome async path has been wedging here).
  if(module.getCompilationInfo){const err=(await module.getCompilationInfo()).messages.find(m=>m.type==='error');if(err)throw new Error(`${name}: ${err.message}`)}
  sources.push({src,module});
}
onProgress('creating pipelines…');
const bindingsOf=wgsl=>{
  const out=[];
  for(const line of wgsl.split('\n')){const i=line.indexOf('@binding(');if(i<0)continue;
    const rest=line.slice(i+9),n=Number(rest.slice(0,rest.indexOf(')')));
    if(!out.some(b=>b.index===n))out.push({index:n,uniform:line.includes('var<uniform')})}
  return out.sort((a,b)=>a.index-b.index);
};
const bindGroup=(name,layout,binds)=>device.createBindGroup({label:name,layout,entries:binds.map(b=>({binding:b.index,resource:b.index===22?{buffer:bufs[22],size:16}:{buffer:bufs[b.index]}}))});
const pipes=[];
for(const [k,[name,entry]] of KERNELS.entries()){
  const binds=bindingsOf(sources[k].src);
  const layout=device.createBindGroupLayout({label:name,entries:binds.map(b=>({binding:b.index,visibility:GPUShaderStage.COMPUTE,
    buffer:b.uniform?{type:'uniform',hasDynamicOffset:b.index===22,minBindingSize:b.index===22?16:undefined}:{type:'storage'}}))});
  onProgress('pipeline '+name);
  const t=performance.now();
  // Sync create: createComputePipelineAsync was hanging on sleep_island_commit
  // after the large narrowphase / solve_primal modules on Chrome Metal.
  device.pushErrorScope('validation');
  const pipeline=device.createComputePipeline({label:name,layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module:sources[k].module,entryPoint:entry}});
  const pipeErr=await device.popErrorScope();if(pipeErr)throw new Error(`${name}: ${pipeErr.message}`);
  pipes.push({name,pipeline,bind:bindGroup(name,layout,binds),dynamic:binds.some(b=>b.index===22),layout,binds});
  onProgress(`pipeline ${name} ${(performance.now()-t).toFixed(0)}ms`);
  // setTimeout, not rAF: background tabs throttle rAF and freeze load between kernels.
  await new Promise(r=>setTimeout(r,0));
}
onProgress('pipelines ready · building renderer…');
await new Promise(r=>setTimeout(r,0));

const staging=device.createBuffer({size:COUNTER_BYTES,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
const snapshot=device.createBuffer({size:Math.max(BODY_BYTES,bodyCount*BODY_BYTES),usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
const hostUpload=device.createBuffer({size:UPLOAD_BYTES,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
let last=plan.counters,timing={wallMs:0,encodeMs:0,waitMs:0},sliceEncodeMs=0,countedSteps=0;

// Replay plan commands: copies on the encoder, dispatches in compute passes.
// begin(name) ends a profiled phase `name` and opens the next pass ('start'
// opens the first); begin.split(name) false keeps encoding into the current
// pass. Without begin, one pass (optionally timestamped by `descriptor`).
function replay(enc,words,begin,descriptor){
  device.queue.writeBuffer(hostUpload,0,plan.hostUpload());
  let pass=null;
  for(let i=0;i<words.length;){
    const op=words[i];
    if(op===OP_COPY){
      const src=words[i+1]===HOST_UPLOAD?hostUpload:bufs[words[i+1]];
      enc.copyBufferToBuffer(src,words[i+2],bufs[words[i+3]],words[i+4],words[i+5]);i+=6;continue;
    }
    pass??=begin?begin('start'):enc.beginComputePass(descriptor);
    if(op===OP_DISPATCH||op===OP_INDIRECT){
      const p=pipes[words[i+1]],slot=op===OP_DISPATCH?words[i+4]:words[i+3];
      pass.setPipeline(p.pipeline);
      if(p.dynamic)pass.setBindGroup(0,p.bind,[slot*PASS_ALIGN]);else pass.setBindGroup(0,p.bind);
      if(op===OP_DISPATCH){pass.dispatchWorkgroups(words[i+2],words[i+3],1);i+=5}
      else{pass.dispatchWorkgroupsIndirect(bufs[23],words[i+2]*12);i+=4}
    }else{
      const name=PHASES[words[i+1]][0];i+=2;
      if(begin&&(!begin.split||begin.split(name))){pass.end();pass=begin(name)}
    }
  }
  (pass??(begin?begin('start'):enc.beginComputePass(descriptor))).end();
}

// The renderer draws a copy of the bodies taken at the end of each step,
// so a step split across frames is never seen half-solved.
const renderBodies=device.createBuffer({label:'render bodies',size:Math.max(BODY_BYTES,bodyCount*BODY_BYTES),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
const writeRender=()=>device.queue.writeBuffer(renderBodies,0,plan.bodyBytes(),0,bodyCount*BODY_BYTES);
writeRender();
const copyRender=enc=>enc.copyBufferToBuffer(bufs[0],0,renderBodies,0,bodyCount*BODY_BYTES);

// Overlapped stepping (step({wait:false})): queued-tick completions, the
// background counter read and an overflow seen in the background (thrown by
// the next call).
const MAX_IN_FLIGHT=3;
let inFlight=[],overlapRead=null,overlapError=null,lastRead=null;
async function flushOverlap(){while(inFlight.length)await inFlight.shift();if(overlapRead)await overlapRead}

// GPU time of one sampled tick at a time (stats panel), about four times a
// second: each pass of the tick writes begin/end timestamps, the sum is the
// tick's GPU time. One sample in flight; nothing waits on its readback.
const tickTimer=device.features.has('timestamp-query')?{query:device.createQuerySet({type:'timestamp',count:32}),
  resolve:device.createBuffer({size:256,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),
  read:device.createBuffer({size:256,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),busy:false,last:0,ms:null}:null;
const timeThisTick=()=>tickTimer&&!tickTimer.busy&&performance.now()-tickTimer.last>250;
const tickStamps=k=>({timestampWrites:{querySet:tickTimer.query,beginningOfPassWriteIndex:2*k,endOfPassWriteIndex:2*k+1}});
function resolveTickStamps(enc,passes){
  tickTimer.busy=true;tickTimer.last=performance.now();
  enc.resolveQuerySet(tickTimer.query,0,2*passes,tickTimer.resolve,0);
  enc.copyBufferToBuffer(tickTimer.resolve,0,tickTimer.read,0,16*passes);
}
// After the submit that holds the copy.
function readTickStamps(passes){
  tickTimer.read.mapAsync(GPUMapMode.READ,0,16*passes).then(()=>{
    const t=new BigUint64Array(tickTimer.read.getMappedRange(0,16*passes));
    let ns=0;for(let i=0;i<passes;i++)ns+=Number(t[2*i+1]-t[2*i]);
    tickTimer.read.unmap();tickTimer.ms=ns/1e6;
  }).catch(()=>{}).finally(()=>{tickTimer.busy=false});
}

function submitPhysics(reuseManifolds=false){
  const enc=device.createCommandEncoder(),timed=timeThisTick();
  replay(enc,plan.encodeStep(reuseManifolds),null,timed?tickStamps(0):undefined);
  copyRender(enc);
  if(timed)resolveTickStamps(enc,1);
  device.queue.submit([enc.finish()]);
  if(timed)readTickStamps(1);
  plan.markSubmitted();
}

let slices=null,sliceStart=0,readPending=false,slicePasses=0;
function encodeSlices(){
  const out=[],timed=timeThisTick();let enc=device.createCommandEncoder(),passes=0;
  const begin=name=>{
    if(SLICE_AFTER.has(name)){out.push(enc.finish());enc=device.createCommandEncoder()}
    return enc.beginComputePass(timed?tickStamps(passes++):undefined);
  };
  begin.split=name=>name==='start'||SLICE_AFTER.has(name);
  replay(enc,plan.encodeStep(false),begin);
  copyRender(enc);
  if(timed)resolveTickStamps(enc,passes);
  slicePasses=timed?passes:0;
  out.push(enc.finish());
  return out;
}

// Counter readbacks share one staging buffer, so they run one at a time:
// a copy into staging must never be submitted while it is mapped.
let readChain=Promise.resolve();
function readCounters(keep=false){const run=readChain.then(()=>readCountersNow(keep));readChain=run.catch(()=>{});return run}
// The plan matches each read to the ticks submitted when its copy was queued.
async function readCountersNow(keep){
  const token=plan.beginRead(),enc=device.createCommandEncoder();
  enc.copyBufferToBuffer(bufs[7],0,staging,0,COUNTER_BYTES);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const words=new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  countedSteps=token[0];
  if(keep)return plan.countRead(token,words);
  return last=plan.finishRead(token,words);
}
// Throws the plan's capacity error after a read reporting an overflow.
const checkOverflow=()=>{try{plan.overflowError()}catch(e){throw new Error(e.message)}};

async function readBuffer(buffer,offset,size){
  if(!size)return new ArrayBuffer(0);
  const temp=device.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  try{const e=device.createCommandEncoder();e.copyBufferToBuffer(buffer,offset,temp,0,size);device.queue.submit([e.finish()]);await temp.mapAsync(GPUMapMode.READ);return temp.getMappedRange().slice(0)}finally{temp.destroy()}
}
const rebind=()=>{for(const p of pipes)if(p.binds.some(b=>b.index===18))p.bind=bindGroup(p.name,p.layout,p.binds)};

return {
 bodyBuffer:bufs[0],renderBodyBuffer:renderBodies,cap,plan,
 // Sliced stepping (see encodeSlices): begin, submit slices frame by frame,
 // then finish to read the counters. A sleeping world needs no step.
 get slicing(){return slices!==null},
 // GPU ms of the latest sampled tick (timestamps), or null if unsupported.
 get gpuTickMs(){return tickTimer?.ms??null},
 beginSlicedStep(){
   if(slices)throw new Error('A sliced step is already in flight');
   if(plan.skipAsleep(1)){timing={wallMs:0,encodeMs:0,waitMs:0};return false}
   sliceStart=performance.now();slices=encodeSlices();sliceEncodeMs=performance.now()-sliceStart;return true;
 },
 submitSlices(n=1){
   if(!slices)return true;
   if(n>0)device.queue.submit(slices.splice(0,n));
   if(slices.length===0){plan.markSubmitted();if(slicePasses){readTickStamps(slicePasses);slicePasses=0}}
   return slices.length===0;
 },
 async finishSlicedStep(){
   if(slices?.length)throw new Error('Slices still pending');
   slices=null;
   // The frame loop does not wait for this; a step finishing while the last
   // readback is still mapping keeps the previous counters one step longer.
   if(readPending)return;
   readPending=true;try{await readCounters()}finally{readPending=false}
   // Slices are submitted over several frames: no GPU wait to report.
   timing={wallMs:performance.now()-sliceStart,encodeMs:sliceEncodeMs,waitMs:0,sliced:true};
   checkOverflow();
 },
 // One timestamped pass per dispatch, totals by kernel name (ms). Pass
 // boundaries add a little per dispatch, so compare kernels, not the sum.
 async profileKernels(){
   if(!device.features.has('timestamp-query'))return {supported:false};
   const max=4096,query=device.createQuerySet({type:'timestamp',count:max});
   const resolve=device.createBuffer({size:max*8,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
   const read=device.createBuffer({size:max*8,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
   try{
     const enc=device.createCommandEncoder(),names=[];let pipe=null,group=null,offsets;
     const run=fn=>{const i=names.length;if(2*i+1>=max)throw new Error('too many dispatches to profile');names.push(pipe.label);
       const p=enc.beginComputePass({timestampWrites:{querySet:query,beginningOfPassWriteIndex:2*i,endOfPassWriteIndex:2*i+1}});
       p.setPipeline(pipe);if(offsets)p.setBindGroup(0,group,offsets);else p.setBindGroup(0,group);fn(p);p.end()};
     const proxy={setPipeline(x){pipe=x},setBindGroup(i,g,o){group=g;offsets=o},
       dispatchWorkgroups(...a){run(p=>p.dispatchWorkgroups(...a))},dispatchWorkgroupsIndirect(b,o){run(p=>p.dispatchWorkgroupsIndirect(b,o))},end(){}};
     replay(enc,plan.encodeStep(false),()=>proxy);
     const n=names.length;
     enc.resolveQuerySet(query,0,n*2,resolve,0);enc.copyBufferToBuffer(resolve,0,read,0,n*16);
     device.queue.submit([enc.finish()]);
     await read.mapAsync(GPUMapMode.READ);const t=new BigUint64Array(read.getMappedRange().slice(0));
     const kernels={};for(let i=0;i<n;i++){const k=kernels[names[i]]??={ms:0,count:0};k.ms+=Number(t[2*i+1]-t[2*i])/1e6;k.count++}
     await readCounters();
     const list=names.map((name,i)=>[name,Number(t[2*i+1]-t[2*i])/1e6]);
     return {supported:true,dispatches:n,kernels,list,counters:{...last}};
   }finally{query.destroy();resolve.destroy();read.destroy()}
 },
 // Advances one real step with GPU timestamps. Default: one compute pass per
 // phase (bounds, broadphase, narrowphase, convex classify/SAT/manifold,
 // persistence, topology, solve, sleep). Pass boundaries add latency, so
 // {singlePass:true} times the whole step in one pass for an honest total.
 async profileStep({singlePass=false}={}){
   if(!device.features.has('timestamp-query'))return {supported:false};
   const query=device.createQuerySet({type:'timestamp',count:64});
   const resolve=device.createBuffer({size:512,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
   const read=device.createBuffer({size:512,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
   try{
     const enc=device.createCommandEncoder(),names=[],start=performance.now();let created=0;
     const stamps=i=>({timestampWrites:{querySet:query,beginningOfPassWriteIndex:i*2,endOfPassWriteIndex:i*2+1}});
     if(singlePass){names.push('step');created=1;replay(enc,plan.encodeStep(false),null,stamps(0))}
     else replay(enc,plan.encodeStep(false),name=>{if(name!=='start')names.push(name);return enc.beginComputePass(stamps(created++))});
     const n=names.length;
     enc.resolveQuerySet(query,0,created*2,resolve,0);enc.copyBufferToBuffer(resolve,0,read,0,created*16);
     device.queue.submit([enc.finish()]);
     await read.mapAsync(GPUMapMode.READ);const times=new BigUint64Array(read.getMappedRange().slice(0));
     const phases={};for(let i=0;i<n;i++)phases[names[i]]=Number(times[i*2+1]-times[i*2])/1e6;
     await readCounters();
     return {supported:true,singlePass,wallMs:performance.now()-start,phases,gpuMs:Object.values(phases).reduce((a,b)=>a+b,0),counters:{...last}};
   }finally{query.destroy();resolve.destroy();read.destroy()}
 },
 // Diagnostics run the actual Slang collision kernels, without advancing poses.
 // Explicit opt-in; no readback/allocation on the interactive hot path.
 // Contacts are persistent cold records: 8 words (colliderA, colliderB,
 // featureId, generation, normal xyz, friction).
 async inspectCollisions({flags=0,awakeQueries=plan.awakeQueries}={}){
   const enc=device.createCommandEncoder();
   replay(enc,plan.encodeInspect(flags,awakeQueries));
   device.queue.submit([enc.finish()]);plan.afterInspect();
   // Report this pass's counters but keep the stepping state: a collision-only
   // pass counts no awake bodies, and step() skips work when last.awake is 0.
   const counters=await readCounters(true);
   if(counters.overflow)throw new Error(`Collision overflow ${counters.overflow}`);
   const base=plan.contactBase(false);
   return {counters:{...counters},base,pairs:new Uint32Array(await readBuffer(bufs[10],0,counters.candidate*16)),
     bounds:new Float32Array(await readBuffer(bufs[9],cap.maxBodies*32,plan.colliderCount*32)),
     contacts:new Uint32Array(await readBuffer(bufs[4],base*32,counters.contact*32)),
     points:new Float32Array(await readBuffer(bufs[27],base*64,counters.contact*64)),
     manifolds:new Uint32Array(await readBuffer(bufs[25],0,counters.manifold*32))};
 },
 // Read-only view of the last step's contacts (no collision pass), for debugging.
 async readContacts(){
   await readCounters();
   const base=plan.contactBase(true);
   return {counters:{...last},base,meta:new Uint32Array(await readBuffer(bufs[4],base*32,last.contact*32)),
     points:new Float32Array(await readBuffer(bufs[27],base*64,last.contact*64))};
 },
 get shots(){return plan.shots},
 // Bond row forces (lambda) and penalties (k) per joint, D6 slot order: normal,
 // u, v, about normal, about u, about v (engine-schema Joint: flags word 2,
 // lambda words 48.., penalty words 56..).
 async jointState(){
   if(!jointCount)return {lambda:[],k:[]};
   const f=new Float32Array(await readBuffer(bufs[5],0,jointCount*JOINT_BYTES)),w=JOINT_BYTES/4;
   const u=new Uint32Array(f.buffer);
   return {lambda:Array.from({length:jointCount},(_,i)=>Array.from(f.subarray(i*w+48,i*w+54))),k:Array.from({length:jointCount},(_,i)=>Array.from(f.subarray(i*w+56,i*w+62))),flags:Array.from({length:jointCount},(_,i)=>u[i*w+2])};
 },
 // Joint flags (bit 0 = broken), read back for tests and the HUD.
 async jointFlags(){
   if(!jointCount)return new Uint32Array(0);
   const v=new Uint32Array(await readBuffer(bufs[5],0,jointCount*JOINT_BYTES));
   return Uint32Array.from({length:jointCount},(_,i)=>v[i*JOINT_BYTES/4+2]);
 },
 set incrementalColoring(v){plan.incrementalColoring=!!v},
 // Solver parameters (dt, iterations, ...), packed every step; setParams merges.
 get params(){return plan.params},
 setParams(partial){plan.setParams(partial)},
 launch(origin,direction,{speed,arc}={}){
   const {id,bytes,offsets}=plan.launch(Float64Array.from(origin),Float64Array.from(direction),speed,arc),data=Uint8Array.from(bytes);
   for(const o of offsets)device.queue.writeBuffer(bufs[0],o,data);
   device.queue.writeBuffer(renderBodies,offsets[0],data);
   last=plan.counters;return id;
 },
 get steps(){return plan.steps},get counters(){return {...last}},get timing(){return {...timing}},
 // {reuseManifolds:true}: three-avbd whole-step manifold reuse (skips collision
 // detection; falls back to a full step after any scene change).
 // wait: true (default) waits for each batch and its counters. wait: false
 // overlaps ticks: it returns once the batch is queued, at most MAX_IN_FLIGHT
 // ticks ahead of the GPU, and reads counters in the background. The GPU runs
 // the same ticks in the same order either way; only host-side uses of the
 // counters (HUD, overflow report, colour-slot estimate, the few-awake solver
 // choice, the all-asleep skip) see values up to a few ticks old.
 async step(count=1,{reuseManifolds=false,wait=true}={}){
   if(!Number.isInteger(count)||count<1||count>4)throw new Error('Physics batch must have 1–4 ticks');
   if(overlapError){const e=overlapError;overlapError=null;throw e}
   if(plan.skipAsleep(count)){timing={wallMs:0,encodeMs:0,waitMs:0};return}
   const start=performance.now();
   // CPU encode/submit time only; a full overlap queue's wait for the GPU is
   // timed separately (it is GPU time, not CPU work).
   let encode=0,queueWait=0;
   for(let i=0;i<count;i++){
     const t0=performance.now();submitPhysics(reuseManifolds);encode+=performance.now()-t0;
     if(!wait){inFlight.push(device.queue.onSubmittedWorkDone());if(inFlight.length>MAX_IN_FLIGHT){const t1=performance.now();await inFlight.shift();queueWait+=performance.now()-t1}}
   }
   if(!wait){
     overlapRead??=readCounters().then(()=>{const end=performance.now(),counted=countedSteps;
       // Per-tick time from throughput between background reads: a single
       // read also waits for the ticks queued ahead of it.
       if(lastRead&&counted>lastRead.steps)timing={wallMs:(end-lastRead.time)/(counted-lastRead.steps),encodeMs:encode/count,waitMs:queueWait/count,overlapped:true};
       lastRead={time:end,steps:counted};
       try{checkOverflow()}catch(e){overlapError=e}
     }).finally(()=>{overlapRead=null});
     return;
   }
   await flushOverlap();
   lastRead=null;await readCounters();const end=performance.now();
   // Includes queue execution and mapAsync synchronization, not pure GPU time.
   timing={wallMs:(end-start)/count,encodeMs:encode/count,waitMs:(end-start-encode)/count};
   checkOverflow();
 },
 // Wait until every overlapped tick has run and the counters are current.
 async flush(){await flushOverlap();await readCounters();if(overlapError){const e=overlapError;overlapError=null;throw e}},
 // Upload the arena's changed records; grows the hull buffer (power of two,
 // contents copied) and rebuilds the bind groups that reference it.
 syncHulls(arena){
   plan.markStale(true);
   const needed=arena.records;
   if(needed>hullRecordsCapacity){
     const records=2**Math.ceil(Math.log2(needed)),size=records*16;
     if(size>device.limits.maxStorageBufferBindingSize)throw new Error('Hull arena exceeds the storage binding limit');
     const grown=device.createBuffer({label:'AVBD 18',size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC|GPUBufferUsage.STORAGE});
     const enc=device.createCommandEncoder();enc.copyBufferToBuffer(bufs[18],0,grown,0,hullRecordsCapacity*16);device.queue.submit([enc.finish()]);
     bufs[18].destroy();bufs[18]=grown;hullRecordsCapacity=records;rebind();
   }
   const words=arena.data;
   for(const [a,b] of arena.takeDirty())device.queue.writeBuffer(bufs[18],a*16,words.buffer,words.byteOffset+a*16,(b-a)*16);
 },
 // Replace a collider (hull swap, filter change). Release old hulls yourself.
 setCollider(index,c){device.queue.writeBuffer(bufs[2],index*64,plan.setCollider(index,c))},
 // Diagnostics: write back a snapshot() of the bodies (e.g. between two
 // inspectCollisions calls, whose narrowphase may wake bodies).
 restoreBodies(data){device.queue.writeBuffer(bufs[0],0,data);plan.markStale(false)},
 async snapshot(){const enc=device.createCommandEncoder();enc.copyBufferToBuffer(bufs[0],0,snapshot,0,bodyCount*BODY_BYTES);device.queue.submit([enc.finish()]);await snapshot.mapAsync(GPUMapMode.READ);const bytes=snapshot.getMappedRange().slice(0);snapshot.unmap();return new Float32Array(bytes)},
 reset(){
   const enc=device.createCommandEncoder();
   for(let i=0;i<bufs.length;i++)if(Plan.clearedOnReset(i))enc.clearBuffer(bufs[i]);
   device.queue.submit([enc.finish()]);
   upload();writeRender();slices=null;
   plan.reset();last=plan.counters;lastRead=null;inFlight=[];overlapError=null;
 },
 dispose(){for(const b of [...bufs,staging,snapshot,hostUpload,renderBodies])b.destroy();if(tickTimer){tickTimer.query.destroy();tickTimer.resolve.destroy();tickTimer.read.destroy()}plan.free()},
};
} catch (error) { for (const buffer of bufs) buffer.destroy(); plan.free(); throw error; }
}
