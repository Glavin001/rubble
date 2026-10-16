import {colliderBounds,gridConfig} from './geometry.mjs';
// WebGPU host submission of Rubble's shared Slang AVBD kernels.
// Adapted from demo.html; kept separate from its scene/benchmark UI.
export async function createSimulation(device, scene, onProgress = () => {}) {
const statusEl = {set textContent(v) {onProgress(v)}};
const cfg = {friction:0.5, iterations:8, sleepRounds:8, ...scene.params};
const MAX_WARM_HASH = 1 << 21;
const PASS_ALIGN=256;
const nextPow2=n=>2**Math.ceil(Math.log2(Math.max(1,n)));
const cap={maxBodies:nextPow2(Math.max(64,scene.bodies.length)),maxColliders:nextPow2(Math.max(64,scene.colliders.length)),
 maxContacts:nextPow2(Math.max(4096,scene.colliders.length*32)),maxCandidates:nextPow2(Math.max(4096,scene.colliders.length*64)),
 maxJoints:1,maxColors:32,maxCommands:1,maxHull:Math.max(64,scene.hullData.length/4)};
const constraintIdWords=nextPow2(Math.max(32768,2*(cap.maxContacts+cap.maxJoints)));
const csrBase=Math.max(constraintIdWords,131072);

    function bufferSizes() {
      const b = (n, stride) => n * stride;
      const s = new Array(28).fill(256);
      s[0] = b(cap.maxBodies * 2, 112);
      s[1] = b(cap.maxBodies, 112);
      s[2] = b(cap.maxColliders, 64);
      s[3] = b(64, 16);
      s[4] = b(cap.maxContacts * 2, 128);
      s[5] = b(cap.maxJoints, 256);
      s[6] = 256;
      s[7] = 256;
      s[8] = b(cap.maxCommands, 64);
      s[9] = b(cap.maxBodies + cap.maxColliders, 32);
      s[10] = b(cap.maxCandidates, 16);
      s[11] = b(cap.maxBodies, 4);
      s[12] = b(cap.maxBodies, 4);
      s[13] = b(cap.maxBodies + 1, 4);
      s[14] = (csrBase + cap.maxBodies + 1) * 4;
      s[15] = b(Math.max(cap.maxColliders,cap.maxBodies * cap.maxColors), 4);
      s[16] = b(cap.maxColors, 4);
      s[17] = b(cap.maxColors, 4);
      s[18] = b(cap.maxHull, 16);
      s[19] = 65536 * 4;
      s[20] = b(cap.maxBodies, 4);
      s[21] = 64 * 4;
      s[22] = 96 * PASS_ALIGN;
      s[23] = b(Math.max(cap.maxColors, 70) * 3, 4);
      s[24] = b(cap.maxColliders, 80);
      s[25] = b(cap.maxCandidates, 32);
      s[26] = b(cap.maxBodies, 32);
      s[27] = b(cap.maxCandidates, 32);
      return s;
    }

    const K = [
      'build_aabbs', 'broadphase_clear', 'broadphase_insert', 'broadphase_query',
      'update_refs', 'prepare_candidates', 'write_sat_args', 'sat', 'narrowphase', 'contact_hash_clear', 'contact_hash_insert', 'persist_contacts',
      'build_awake_list', 'csr_clear', 'csr_degree_contacts', 'csr_prefix',
      'csr_prefix_init', 'csr_prefix_scan', 'csr_prefix_finalize', 'csr_prefix_swap',
      'csr_scatter_contacts', 'csr_sort', 'init_colors', 'color_bodies',
      'pack_color_lists', 'write_color_args', 'write_work_args', 'predict_bodies', 'solve_primal_color',
      'solve_dual', 'extract_velocity_sleep', 'sleep_island_gate', 'sleep_island_commit',
    ];

    function bindingsOf(wgsl) {
      const out = [];
      for (const line of wgsl.split('\n')) {
        const i = line.indexOf('@binding(');
        if (i < 0) continue;
        const rest = line.slice(i + '@binding('.length);
        const n = Number(rest.slice(0, rest.indexOf(')')));
        if (out.some((b) => b.index === n)) continue;
        out.push({ index: n, uniform: line.includes('var<uniform') });
      }
      out.sort((a, b) => a.index - b.index);
      return out;
    }

    function f32(view, o, v) { view.setFloat32(o, v, true); }
    function u32(view, o, v) { view.setUint32(o, v, true); }

    function writeBody(buf, index, body) {
      const o = index * 112;
      const v = new DataView(buf);
      const q = body.q || [0, 0, 0, 1];
      const w = body.w || [0, 0, 0];
      f32(v, o + 0, body.p[0]); f32(v, o + 4, body.p[1]); f32(v, o + 8, body.p[2]); f32(v, o + 12, body.invMass);
      f32(v, o + 16, q[0]); f32(v, o + 20, q[1]); f32(v, o + 24, q[2]); f32(v, o + 28, q[3]);
      f32(v, o + 32, body.invI[0]); f32(v, o + 36, body.invI[1]); f32(v, o + 40, body.invI[2]); u32(v, o + 44, body.flags);
      const vel = body.v || [0, 0, 0];
      f32(v, o + 48, vel[0]); f32(v, o + 52, vel[1]); f32(v, o + 56, vel[2]); f32(v, o + 60, 0);
      f32(v, o + 64, w[0]); f32(v, o + 68, w[1]); f32(v, o + 72, w[2]); u32(v, o + 76, 0);
      // initialPos/Rot = pose so frame-0 x⁻ / warm-start match Metal (zeros left a 0-quat).
      f32(v, o + 80, body.p[0]); f32(v, o + 84, body.p[1]); f32(v, o + 88, body.p[2]); f32(v, o + 92, 0);
      f32(v, o + 96, q[0]); f32(v, o + 100, q[1]); f32(v, o + 104, q[2]); f32(v, o + 108, q[3]);
    }

    function writeCollider(buf, index, c) {
      const o = index * 64;
      const v = new DataView(buf);
      u32(v, o + 0, c.body); u32(v, o + 4, c.shape); u32(v, o + 8, 0); u32(v, o + 12, 0);
      f32(v, o + 16, 0); f32(v, o + 20, 0); f32(v, o + 24, 0); f32(v, o + 28, c.density);
      f32(v, o + 32, 0); f32(v, o + 36, 0); f32(v, o + 40, 0); f32(v, o + 44, 1);
      f32(v, o + 48, c.params[0]); f32(v, o + 52, c.params[1]); f32(v, o + 56, c.params[2]); f32(v, o + 60, c.params[3]);
    }

    const params = {
      dt: 1 / 60, gravityY: -10, iterations: cfg.iterations, flags: cfg.flags || 0,
      alpha: cfg.alpha ?? 0.99, gamma: 0.999, betaLin: cfg.betaLin ?? 10000, betaAng: 100,
      kStart: 1000, kMaxN: 1e10, kMaxT: 1e6, contactOffset: 0.01,
      sleepLin: cfg.sleepLin ?? 0.15, sleepAng: cfg.sleepAng ?? 0.15, sleepTime: cfg.sleepTime ?? 0.5, residualExit: 1e-4,
      trustLin: 0.35, trustAng: 0.5, pivotFloor: 1e-6, gridCellSize: 2,
    };

    const packHostBytes = new ArrayBuffer(208);
    const packHostView = new DataView(packHostBytes);
    const packHostU8 = new Uint8Array(packHostBytes);
    function packHost(scratch) {
      const v = packHostView;
      const nums = [
        params.dt, params.gravityY, params.iterations, params.flags,
        params.alpha, params.gamma, params.betaLin, params.betaAng,
        params.kStart, params.kMaxN, params.kMaxT, params.contactOffset,
        params.sleepLin, params.sleepAng, params.sleepTime, params.residualExit,
        params.trustLin, params.trustAng, params.pivotFloor, params.gridCellSize,
      ];
      const uintFields = new Set([2, 3]);
      nums.forEach((n, i) => uintFields.has(i) ? u32(v, i * 4, n) : f32(v, i * 4, n));
      for (let i = 0; i < 32; i++) u32(v, 80 + i * 4, scratch[i] || 0);
      return packHostBytes;
    }


Object.assign(params,scene.params,{gridCellSize:scene.gridCellSize});
const sizes=bufferSizes();
for(const [i,size] of sizes.entries())if(size>device.limits.maxStorageBufferBindingSize||size>device.limits.maxBufferSize)throw new Error(`GPU buffer ${i} exceeds this adapter's limits`);
const bufs=sizes.map((size,i)=>device.createBuffer({label:`AVBD ${i}`,size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC|((i===6||i===22)?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|(i===23?GPUBufferUsage.INDIRECT:0)}));

try {
    const pass = new Uint8Array(96 * PASS_ALIGN);
    const passView = new DataView(pass.buffer);
    for (let i = 0; i < 96; i++) {
      let v = i;
      if (i >= 48 && i < 68) { const k = i - 48; v = (1 << k) | ((k & 1) << 31); }
      else if (i >= 68) v = 0;
      passView.setUint32(i * PASS_ALIGN, v >>> 0, true);
    }
    device.queue.writeBuffer(bufs[22], 0, pass);

    const mats = new ArrayBuffer(64 * 16);
    const mv = new DataView(mats);
    for (let i = 0; i < 64; i++) {
      f32(mv, i * 16, cfg.friction); f32(mv, i * 16 + 4, 0); f32(mv, i * 16 + 8, 0.01); f32(mv, i * 16 + 12, 0);
    }
    device.queue.writeBuffer(bufs[3], 0, mats);


const bodyBytes=new ArrayBuffer(sizes[0]),colBytes=new ArrayBuffer(sizes[2]);
const bodies=scene.bodies,colliders=scene.colliders;
bodies.forEach((b,i)=>{writeBody(bodyBytes,i,b);writeBody(bodyBytes,cap.maxBodies+i,b)});
colliders.forEach((c,i)=>writeCollider(colBytes,i,c));
device.queue.writeBuffer(bufs[0],0,bodyBytes);device.queue.writeBuffer(bufs[2],0,colBytes);
device.queue.writeBuffer(bufs[18],0,scene.hullData);
// Sleeping bodies skip AABB refit, so publish their initial cached bounds.
const initialAabbs=new ArrayBuffer(sizes[9]);
for(const [i,c] of colliders.entries()){
 const v=new DataView(initialAabbs),o=(cap.maxBodies+i)*32,{lo,hi}=colliderBounds(scene,c);
 for(let k=0;k<3;k++){v.setFloat32(o+k*4,lo[k],true);v.setFloat32(o+16+k*4,hi[k],true)}
 v.setFloat32(o+12,c.params[3]||c.params[0],true);v.setUint32(o+28,c.body,true);
}
device.queue.writeBuffer(bufs[9],0,initialAabbs);
const grid=gridConfig(colliders),planes=Uint32Array.from(colliders.flatMap((c,i)=>c.shape===4?[i]:[]));
params.gridCellSize=grid.base;
if(planes.length)device.queue.writeBuffer(bufs[19],0,planes);

    const sources = {};
    for (const name of K) {
      statusEl.textContent = 'compiling ' + name;
      const entry=['prepare_candidates','write_sat_args','sat','narrowphase','broadphase_query','broadphase_insert','build_aabbs'].includes(name)?`fracture_${name}`:name;
      const response=await fetch(`../generated/wgsl/${entry}.wgsl`);
      if(!response.ok)throw new Error(`Missing ${entry}; run the shader export commands in fracture/README.md`);
      const src=await response.text();
      const module = device.createShaderModule({ label: name, code: src });
      // Force frontend validation now so createComputePipeline cannot hang forever
      // on a deferred compile error (Chrome async path has been wedging here).
      if (module.getCompilationInfo) {
        const info = await module.getCompilationInfo();
        const err = info.messages.find((m) => m.type === 'error');
        if (err) throw new Error(`${name}: ${err.message}`);
      }
      sources[name] = { src, module };
    }

    statusEl.textContent = 'creating pipelines…';
    const pipes = {};
    for (const name of K) {
      const binds = bindingsOf(sources[name].src);
      const layout = device.createBindGroupLayout({
        label: name,
        entries: binds.map((b) => ({
          binding: b.index,
          visibility: GPUShaderStage.COMPUTE,
          buffer: b.uniform
            ? { type: 'uniform', hasDynamicOffset: b.index === 22, minBindingSize: b.index === 22 ? 16 : undefined }
            : { type: 'storage' },
        })),
      });
      statusEl.textContent = 'pipeline ' + name;
      const tPipe = performance.now();
      // Sync create: createComputePipelineAsync was hanging on sleep_island_commit
      // after the large narrowphase / solve_primal modules on Chrome Metal.
      device.pushErrorScope('validation');
      const pipeline = device.createComputePipeline({
        label: name,
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: sources[name].module, entryPoint: ["prepare_candidates","write_sat_args","sat","narrowphase","broadphase_query","broadphase_insert","build_aabbs"].includes(name) ? `fracture_${name}` : name },
      });
      const pipeErr = await device.popErrorScope();
      if (pipeErr) throw new Error(`${name}: ${pipeErr.message}`);
      const bind = device.createBindGroup({
        label: name,
        layout,
        entries: binds.map((b) => ({
          binding: b.index,
          resource: b.index === 22 ? { buffer: bufs[22], size: 16 } : { buffer: bufs[b.index] },
        })),
      });
      pipes[name] = { pipeline, bind, dynamic: binds.some((b) => b.index === 22) };
      statusEl.textContent = `pipeline ${name} ${(performance.now() - tPipe).toFixed(0)}ms`;
      // setTimeout — not rAF: background tabs throttle rAF and freeze load between kernels.
      await new Promise((r) => setTimeout(r, 0));
    }

    statusEl.textContent = 'pipelines ready · building renderer…';
    await new Promise((r) => setTimeout(r, 0));


const staging=device.createBuffer({size:64,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
const bodyRead=device.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
const snapshot=device.createBuffer({size:scene.bodies.length*112,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
const bodyCount=bodies.length,colliderCount=colliders.length,sleepRounds=8;
let last={contact:0,candidate:0,awake:bodyCount-1,color:16,manifold:0,overflow:0},steps=0,parity=0,shots=0;
let timing={wallMs:0,encodeMs:0,waitMs:0},initialized=false;

    // Reused every step — allocating these in writeHost showed up as wall−GPU gap on ring.
    const scratchU32 = new Uint32Array(64);
    // Must match bufs[17] size from bufferSizes (maxColors × 4), not a larger pad.
    const colorCountWords = cap.maxColors;
    const colorCountsBytes = colorCountWords * 4;
    // One queue upload + GPU copies into the four step buffers (cheaper than 4 writeBuffers).
    const hostUpload = device.createBuffer({
      size: 256 + 256 + 64 + colorCountsBytes,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const hostUploadCpu = new ArrayBuffer(256 + 256 + 64 + colorCountsBytes);
    const hostScratchView = new Uint32Array(hostUploadCpu, 0, 64);
    const hostParamsView = new Uint8Array(hostUploadCpu, 256, 208);
    const hostCountersView = new Uint32Array(hostUploadCpu, 512, 16);
    const hostColorsView = new Uint32Array(hostUploadCpu, 576, colorCountWords);

    function scratchWords() {
      const idsLen = csrBase;
      const prevN = cap.maxContacts;
      const s = scratchU32;
      s.fill(0);
      s[1] = cap.maxBodies;
      s[2] = cap.maxCandidates;
      s[3] = cap.maxContacts;
      s[4] = Math.min(cap.maxColors, 32);
      s[7] = steps === 0 ? 0 : Math.min(last.contact, cap.maxContacts);
      s[8] = parity === 0 ? cap.maxContacts : 0;
      // One hash entry per manifold; host field and clear dispatch must agree.
      const hash = Math.min(nextPow2(Math.max(prevN * 4, 1024)), idsLen, MAX_WARM_HASH);
      s[9] = hash;
      s[10] = cap.maxCandidates;
      s[11] = parity === 0 ? 0 : cap.maxContacts;
      let log = 0;
      while ((1 << log) < Math.max(bodyCount, 1) && log < 20) log++;
      s[12] = log === 0 ? 0 : (log - 1) & 1;
      s[17] = initialized?Math.max(1,steps):0;
      s[18] = grid.levels;
      s[19] = planes.length;
      s[24] = csrBase;
      return s;
    }

    function writeHost(enc) {
      const s = scratchWords();
      packHost(s);
      hostScratchView.set(s);
      hostParamsView.set(packHostU8);
      hostCountersView.fill(0);
      hostCountersView[0] = bodyCount;
      hostCountersView[1] = colliderCount;
      hostColorsView.fill(0);
      device.queue.writeBuffer(hostUpload, 0, hostUploadCpu);
      // Prefer GPU copies from one upload — four tiny writeBuffers were ~0.3 ms of ring wall.
      enc.copyBufferToBuffer(hostUpload, 0, bufs[21], 0, 256);
      enc.copyBufferToBuffer(hostUpload, 256, bufs[6], 0, 208);
      // Keep the previous GPU count before clearing counters. A benchmark queues
      // ten steps between CPU readbacks; last.contact is stale within that batch.
      if (steps > 0) enc.copyBufferToBuffer(bufs[7], 8, bufs[6], 80 + 7 * 4, 4);
      enc.copyBufferToBuffer(hostUpload, 512, bufs[7], 0, 64);
      enc.copyBufferToBuffer(hostUpload, 576, bufs[17], 0, colorCountsBytes);
    }

    function groups(threads) {
      const n = Math.ceil(Math.max(threads, 1) / 64);
      return [Math.min(n, 32768), Math.ceil(n / 32768), 1];
    }
    function disp(pass, name, threads, passIdx) {
      const p = pipes[name];
      pass.setPipeline(p.pipeline);
      if (p.dynamic) pass.setBindGroup(0, p.bind, [(passIdx || 0) * PASS_ALIGN]);
      else pass.setBindGroup(0, p.bind);
      pass.dispatchWorkgroups(...groups(threads));
    }

    function dispWork(pass, name, slot, passIdx = 0) {
      const p = pipes[name];
      pass.setPipeline(p.pipeline);
      if (p.dynamic) pass.setBindGroup(0, p.bind, [passIdx * PASS_ALIGN]);
      else pass.setBindGroup(0, p.bind);
      pass.dispatchWorkgroupsIndirect(bufs[23], slot * 12);
    }

    function encodeStep(enc, readback, beginPass) {
      const nBodies = Math.max(bodyCount, 1);
      const nCols = Math.max(colliderCount, 1);
      const prevContacts = cap.maxContacts;
      const idsLen = nextPow2(Math.max(32768, 2 * (cap.maxContacts + cap.maxJoints)));
      // Hash size is a power of two (mask = n-1).
      const hashN = Math.min(
        nextPow2(Math.max(Math.max(prevContacts, 256) * 4, 1024)),
        idsLen,
        MAX_WARM_HASH,
      );

      writeHost(enc);
      let pass = beginPass ? beginPass() : enc.beginComputePass();
      const nextPhase = () => {
        if (!beginPass) return;
        pass.end();
        pass = beginPass();
      };
      disp(pass, 'build_aabbs', nCols);
      disp(pass, 'broadphase_clear', 131072);
      nextPhase();
      disp(pass, 'broadphase_insert', nCols);
      disp(pass, 'broadphase_query', nCols);
      disp(pass, 'write_work_args', 64);
      nextPhase();
      const reuse = (params.flags & 1) !== 0;
      if (reuse) {
        disp(pass, 'update_refs', nBodies);
        disp(pass, 'contact_hash_clear', hashN);
        dispWork(pass, 'contact_hash_insert', 68);
      }
      dispWork(pass, 'prepare_candidates', 64);
      disp(pass, 'write_sat_args', 1);
      dispWork(pass, 'sat', 69);
      dispWork(pass, 'narrowphase', 64);
      disp(pass, 'write_work_args', 64);
      nextPhase();
      if (!reuse) {
        disp(pass, 'contact_hash_clear', hashN);
        dispWork(pass, 'contact_hash_insert', 68);
      }
      dispWork(pass, 'persist_contacts', 65);
      nextPhase();
      disp(pass, 'build_awake_list', nBodies);
      disp(pass, 'write_work_args', 64);
      disp(pass, 'csr_clear', nBodies + 1);
      dispWork(pass, 'csr_degree_contacts', 66);
      if (bodyCount <= 512) {
        disp(pass, 'csr_prefix', 1);
      } else {
        disp(pass, 'csr_prefix_init', nBodies);
        let log = 0;
        while ((1 << log) < bodyCount && log < 20) {
          disp(pass, 'csr_prefix_scan', nBodies, 48 + log);
          log++;
        }
        disp(pass, 'csr_prefix_finalize', nBodies + 1);
        disp(pass, 'csr_prefix_swap', nBodies);
      }
      dispWork(pass, 'csr_scatter_contacts', 66);
      dispWork(pass, 'csr_sort', 67);
      dispWork(pass, 'init_colors', 67);
      // Complete graph coloring before solving; the shader reports unfinished colors.
      const colorRounds = 32;
      for (let round = 0; round < colorRounds; round++) dispWork(pass, 'color_bodies', 67, round % 16);
      dispWork(pass, 'pack_color_lists', 67);
      disp(pass, 'write_color_args', 64);
      disp(pass, 'predict_bodies', nBodies);
      nextPhase();
      // Current-step indirect arguments skip empty colors without a CPU estimate.
      const colors = Math.min(cap.maxColors, 32);
      for (let iter = 0; iter < params.iterations; iter++) {
        if (beginPass && iter > 0) nextPhase();
        const primal = pipes.solve_primal_color;
        for (let color = 0; color < colors; color++) {
          pass.setPipeline(primal.pipeline);
          pass.setBindGroup(0, primal.bind, [color * PASS_ALIGN]);
          pass.dispatchWorkgroupsIndirect(bufs[23], color * 12);
        }
        if (beginPass) nextPhase();
        dispWork(pass, 'solve_dual', 65);
      }
      disp(pass, 'extract_velocity_sleep', nBodies);
      nextPhase();
      {
        for (let i = 0; i < sleepRounds; i++) dispWork(pass, 'sleep_island_gate', 66);
      }
      disp(pass, 'sleep_island_commit', nBodies);
      pass.end();
      if (readback) {
        enc.copyBufferToBuffer(bufs[7], 0, staging, 0, 64);
        enc.copyBufferToBuffer(bufs[0], 112, bodyRead, 0, 16);
      }
    }

    function submitPhysics() {
      const enc = device.createCommandEncoder();
      encodeStep(enc, false);
      device.queue.submit([enc.finish()]);
      initialized=true;
      parity ^= 1;
      steps++;
    }

    async function readCounters() {
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(bufs[7], 0, staging, 0, 64);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const c = new Uint32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      last = { contact: c[2], candidate: c[4], awake: c[5], color: c[6], manifold: c[12], overflow: c[7] };
    }


return {
 bodyBuffer:bufs[0],cap,
 async profileStep(){
   if(!device.features.has('timestamp-query'))return {supported:false};
   const query=device.createQuerySet({type:'timestamp',count:128});
   const resolve=device.createBuffer({size:1024,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
   const read=device.createBuffer({size:1024,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
   try{
     const enc=device.createCommandEncoder();let phases=0;const start=performance.now();
     encodeStep(enc,false,()=>{const i=phases++;return enc.beginComputePass({timestampWrites:{querySet:query,beginningOfPassWriteIndex:i*2,endOfPassWriteIndex:i*2+1}})});
     enc.resolveQuerySet(query,0,phases*2,resolve,0);enc.copyBufferToBuffer(resolve,0,read,0,phases*16);
     device.queue.submit([enc.finish()]);initialized=true;parity^=1;steps++;
     await read.mapAsync(GPUMapMode.READ);const times=new BigUint64Array(read.getMappedRange().slice(0));
     const ms=Array.from({length:phases},(_,i)=>Number(times[i*2+1]-times[i*2])/1e6);
     await readCounters();return {supported:true,wallMs:performance.now()-start,boundsMs:ms[0],broadphaseMs:ms[1],narrowphaseMs:ms[2],persistMs:ms[3],topologyMs:ms[4],solveMs:ms.slice(5,-1).reduce((a,b)=>a+b,0),sleepMs:ms.at(-1),gpuMs:ms.reduce((a,b)=>a+b,0),counters:{...last}};
   }finally{query.destroy();resolve.destroy();read.destroy()}
 },
 // Diagnostics run the actual Slang collision kernels, without advancing poses.
 // Explicit opt-in; no readback/allocation on the interactive hot path.
 async inspectCollisions(){
   const enc=device.createCommandEncoder(),savedFlags=params.flags;params.flags=0;writeHost(enc);params.flags=savedFlags;const pass=enc.beginComputePass();
   disp(pass,'build_aabbs',colliderCount);disp(pass,'broadphase_clear',131072);
   disp(pass,'broadphase_insert',colliderCount);disp(pass,'broadphase_query',colliderCount);
   disp(pass,'write_work_args',64);dispWork(pass,'prepare_candidates',64);disp(pass,'write_sat_args',1);dispWork(pass,'sat',69);dispWork(pass,'narrowphase',64);pass.end();device.queue.submit([enc.finish()]);
   await readCounters();
   if(last.overflow)throw new Error(`Collision overflow ${last.overflow}`);
   const read=async(buffer,offset,size)=>{
     if(!size)return new ArrayBuffer(0);
     const temp=device.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
     try{const e=device.createCommandEncoder();e.copyBufferToBuffer(buffer,offset,temp,0,size);device.queue.submit([e.finish()]);await temp.mapAsync(GPUMapMode.READ);return temp.getMappedRange().slice(0)}finally{temp.destroy()}
   };
   return {counters:{...last},pairs:new Uint32Array(await read(bufs[10],0,last.candidate*16)),
     bounds:new Float32Array(await read(bufs[9],cap.maxBodies*32,colliderCount*32)),
     contacts:new Float32Array(await read(bufs[4],(parity===0?0:cap.maxContacts)*128,last.contact*128))};
 },
 get shots(){return shots},
 launch(origin,direction){
   if(shots>=scene.projectileCount)throw new Error('All 16 projectiles fired. Reset to reload.');
   if(![...origin,...direction].every(Number.isFinite))throw new Error('Invalid projectile ray');
   const id=scene.projectileStart+shots,mass=scene.projectileMass,r=scene.projectileRadius;
   const speed=Math.max(5,Math.min(18,scene.extent*3));
   const distance=Math.max(r,scene.center.reduce((sum,x,i)=>sum+(x-origin[i])*direction[i],0));
   const v=direction.map(x=>x*speed);v[1]+=.5*9.81*distance/speed;
   const body={p:origin,q:[0,0,0,1],v,invMass:1/mass,invI:Array(3).fill(1/(.4*mass*r*r)),flags:9};
   const bytes=new ArrayBuffer(112);writeBody(bytes,0,body);
   device.queue.writeBuffer(bufs[0],id*112,bytes);device.queue.writeBuffer(bufs[0],(cap.maxBodies+id)*112,bytes);
   last.awake=Math.max(1,last.awake);shots++;return id;
 },
 get steps(){return steps},get counters(){return {...last}},get timing(){return {...timing}},
 async step(count=1){
   if(!Number.isInteger(count)||count<1||count>4)throw new Error('Physics batch must have 1–4 ticks');
   if(initialized&&last.awake===0){steps+=count;timing={wallMs:0,encodeMs:0,waitMs:0};return;}
   const start=performance.now();for(let i=0;i<count;i++)submitPhysics();const submitted=performance.now();
   await readCounters();const end=performance.now();
   // Includes queue execution and mapAsync synchronization, not pure GPU time.
   timing={wallMs:(end-start)/count,encodeMs:(submitted-start)/count,waitMs:(end-submitted)/count};
   if(last.overflow)throw new Error(`Physics capacity exceeded (0x${last.overflow.toString(16)}); choose a coarser level`);
 },
 async snapshot(){const enc=device.createCommandEncoder();enc.copyBufferToBuffer(bufs[0],0,snapshot,0,scene.bodies.length*112);device.queue.submit([enc.finish()]);await snapshot.mapAsync(GPUMapMode.READ);const bytes=snapshot.getMappedRange().slice(0);snapshot.unmap();return new Float32Array(bytes);},
 reset(){
   const enc=device.createCommandEncoder();
   for(let i=0;i<bufs.length;i++)if(![0,2,3,18,19,22].includes(i))enc.clearBuffer(bufs[i]);
   device.queue.submit([enc.finish()]);
   device.queue.writeBuffer(bufs[0],0,bodyBytes);device.queue.writeBuffer(bufs[2],0,colBytes);
   device.queue.writeBuffer(bufs[9],0,initialAabbs);
   steps=0;parity=0;shots=0;initialized=false;last={contact:0,candidate:0,awake:bodyCount-1,color:16,manifold:0,overflow:0};
 },
 dispose(){for(const b of [...bufs,staging,bodyRead,snapshot,hostUpload])b.destroy();},
};
} catch (error) { for (const buffer of bufs) buffer.destroy(); throw error; }
}
