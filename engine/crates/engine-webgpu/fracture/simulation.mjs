import {gridConfig,filterWords,JOINT_BYTES} from './geometry.mjs';
// WebGPU host submission of Rubble's shared Slang AVBD kernels (persistent
// hot/cold contacts, convex hull narrowphase, hierarchical broadphase). The
// schedule mirrors engine-webgpu/src/gpu.rs in its persistent-contact mode.
export async function createSimulation(device, scene, onProgress = () => {}) {
const statusEl = {set textContent(v) {onProgress(v)}};
const cfg = {friction:0.5, iterations:8, sleepRounds:8, ...scene.params};
const MAX_WARM_HASH = 1 << 21;
const BOND_WAKE_ROUNDS = 8;
const PASS_ALIGN=256;
const nextPow2=n=>2**Math.ceil(Math.log2(Math.max(1,n)));
const budget={contactsPerCollider:32,candidatesPerCollider:64,...scene.capacity};
// Incremental colouring recolours only changed neighbourhoods but lets the
// palette drift (more solve passes); fresh colouring is the default.
let incremental=budget.incrementalColoring??false;
const cap={maxBodies:nextPow2(Math.max(64,scene.bodies.length)),maxColliders:nextPow2(Math.max(64,scene.colliders.length)),
 maxContacts:nextPow2(Math.max(4096,scene.colliders.length*budget.contactsPerCollider)),maxCandidates:nextPow2(Math.max(4096,scene.colliders.length*budget.candidatesPerCollider)),
 maxJoints:Math.max(1,scene.joints?.length??0),maxColors:budget.maxColors??32,maxCommands:1,maxHull:Math.max(64,scene.hullData.length/4)};
const jointCount=scene.joints?.length??0;
// Joint-pair hash (world_step jointHashSize): >= 65536 slots and 2x the joints.
const jointHashSlots=nextPow2(Math.max(2*jointCount,65536));
const constraintIdWords=nextPow2(Math.max(32768,2*(cap.maxContacts+cap.maxJoints)));
const csrBase=Math.max(constraintIdWords,131072);
// Broadphase grid heads: twice the colliders, within the pre-CSR id region.
const gridHashWords=Math.min(nextPow2(Math.max(2*scene.colliders.length,131072)),csrBase);

    function bufferSizes() {
      const b = (n, stride) => n * stride;
      const s = new Array(30).fill(256);
      s[0] = b(cap.maxBodies * 2, 112);
      s[1] = b(cap.maxBodies, 112);
      s[2] = b(cap.maxColliders, 64);
      s[3] = b(64, 16);
      s[4] = b(cap.maxContacts * 2, 32);   // persistent ContactMeta (cold)
      s[5] = b(cap.maxJoints, JOINT_BYTES);
      s[6] = 256;
      s[7] = 256;
      s[8] = b(cap.maxCommands, 64);
      s[9] = b(cap.maxBodies + cap.maxColliders, 32);
      s[10] = b(cap.maxCandidates, 16);
      s[11] = b(cap.maxBodies, 4);
      s[12] = b(cap.maxBodies, 4);
      s[13] = b(cap.maxBodies + 1, 4);
      s[14] = (csrBase + cap.maxBodies + 1) * 4;
      // Colour lists + history rows; broadphase links reuse it per collider.
      s[15] = b(Math.max(cap.maxColliders, cap.maxBodies * (cap.maxColors + 2)), 4);
      s[16] = b(cap.maxColors * Math.max(1, Math.ceil(cap.maxBodies / 64)), 4);
      s[17] = b(128, 4);                     // colour counts 0..63 + max degrees 64..127
      s[18] = b(cap.maxHull, 16);
      s[19] = jointHashSlots * 3 * 4;      // joint hash: index + body pair per slot
      s[20] = b(cap.maxBodies, 4);
      s[21] = 96 * 4; // scratch: host-reset words 0..63, uniform-grid large list 64..79
      s[22] = 96 * PASS_ALIGN;
      s[23] = b(Math.max(cap.maxColors, 77) * 3, 4); // colours, work 64..73, convex 74..76
      s[24] = b(cap.maxColliders, 80);
      s[25] = b(cap.maxCandidates, 32);
      s[26] = b(cap.maxBodies, 32);
      s[27] = b(cap.maxContacts * 2, 64);  // persistent SolverPoint (hot)
      s[28] = b(cap.maxCandidates, 32);    // solver pairs
      s[29] = b(cap.maxCandidates, 32);    // convex SAT work list
      return s;
    }

    // Logical kernel -> exported entry (persistent contact storage).
    const ENTRY = {
      narrowphase: 'narrowphase_persistent', contact_hash_insert: 'contact_hash_insert_persistent',
      persist_contacts: 'persist_contacts_persistent', pack_solver_contacts: 'pack_solver_pairs_persistent',
      solve_primal_color: 'solve_primal_color_persistent', solve_dual: 'solve_dual_persistent',
      convex_classify: 'convex_classify_persistent', convex_manifold: 'convex_manifold_persistent',
      reuse_all_manifolds: 'reuse_all_manifolds_persistent',
    };
    const K = [
      'build_aabbs', 'broadphase_clear', 'broadphase_insert', 'broadphase_query', 'write_work_args',
      'update_refs', 'contact_hash_clear', 'contact_hash_insert', 'narrowphase',
      'convex_classify', 'convex_write_args', 'convex_sat_small', 'convex_sat_large', 'convex_manifold', 'reuse_all_manifolds',
      'persist_contacts', 'build_awake_list', 'csr_clear', 'csr_degree_contacts', 'csr_prefix',
      'csr_prefix_blocks', 'csr_prefix_block_sums', 'csr_prefix_add', 'csr_scatter_contacts', 'csr_sort',
      'init_colors', 'color_bodies', 'color_bodies_local', 'count_color_tiles', 'prefix_color_tiles_parallel', 'scatter_color_tiles',
      'record_color_degrees', 'write_color_args', 'predict_bodies', 'pack_solver_contacts', 'solve_primal_color',
      'solve_dual', 'extract_velocity_sleep',
      'joint_hash_clear', 'joint_hash_insert', 'bond_wake', 'warmstart_joints', 'csr_degree_joints', 'csr_scatter_joints', 'write_sleep_args', 'sleep_island_gate', 'sleep_island_commit',
    ];
    let hasConvex = scene.colliders.some((c) => c.shape === 3);
    const filterOf = (c) => filterWords(c.filter, c.material || 0);
    let hasFilters = scene.colliders.some((c) => { const f = filterOf(c); return f.generation !== 0 || (f.material >>> 16) !== 0; });

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
      const lp = c.localPos || [0, 0, 0], lq = c.localRot || [0, 0, 0, 1];
      const f = filterOf(c);
      u32(v, o + 0, c.body); u32(v, o + 4, c.shape); u32(v, o + 8, f.material); u32(v, o + 12, f.generation);
      f32(v, o + 16, lp[0]); f32(v, o + 20, lp[1]); f32(v, o + 24, lp[2]); f32(v, o + 28, c.density);
      f32(v, o + 32, lq[0]); f32(v, o + 36, lq[1]); f32(v, o + 40, lq[2]); f32(v, o + 44, lq[3]);
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
// Bond joints (Joint, 192 bytes; see common.slang): fixed penalty k0..k5
// (normal, u, v, about u, about v, about normal) in the bond frame.
const jointBytes=new ArrayBuffer(Math.max(JOINT_BYTES,jointCount*JOINT_BYTES));
(scene.joints??[]).forEach((j,i)=>{const v=new DataView(jointBytes,i*JOINT_BYTES);
  v.setUint32(0,j.bodyA,true);v.setUint32(4,j.bodyB,true);v.setUint32(8,7,true);v.setUint32(12,0,true);
  j.anchorA.forEach((x,k)=>v.setFloat32(16+4*k,x,true));v.setFloat32(28,1e30,true);
  j.anchorB.forEach((x,k)=>v.setFloat32(32+4*k,x,true));
  j.rest.forEach((x,k)=>v.setFloat32(48+4*k,x,true));      // axisA.xyz + limitMin: rest relative rotation
  j.normal.forEach((x,k)=>v.setFloat32(64+4*k,x,true));    // axisB: bond normal in A's frame
  (j.frameU??[0,0,0]).forEach((x,k)=>v.setFloat32(144+4*k,x,true)); // frameU: bond u axis in A's frame
  j.strength.forEach((x,k)=>v.setFloat32(160+4*k,x,true));
  for(let k=0;k<6;k++)v.setFloat32(120+4*k,j.k[k],true);
});
colliders.forEach((c,i)=>writeCollider(colBytes,i,c));
device.queue.writeBuffer(bufs[0],0,bodyBytes);device.queue.writeBuffer(bufs[2],0,colBytes);
device.queue.writeBuffer(bufs[5],0,jointBytes);
device.queue.writeBuffer(bufs[18],0,scene.hullData);
let hullRecordsCapacity=cap.maxHull;
// Hierarchical size-level grid; build_aabbs initialises sleeping bounds on the
// first step (HOST_S(17) == 0), so no CPU-side AABB upload is needed.
const grid=gridConfig(colliders);
params.gridCellSize=grid.base;

    const sources = {};
    for (const name of K) {
      statusEl.textContent = 'compiling ' + name;
      const entry=ENTRY[name]||name;
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
        compute: { module: sources[name].module, entryPoint: ENTRY[name] || name },
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
      pipes[name] = { pipeline, bind, dynamic: binds.some((b) => b.index === 22), layout, binds };
      statusEl.textContent = `pipeline ${name} ${(performance.now() - tPipe).toFixed(0)}ms`;
      // setTimeout — not rAF: background tabs throttle rAF and freeze load between kernels.
      await new Promise((r) => setTimeout(r, 0));
    }

    statusEl.textContent = 'pipelines ready · building renderer…';
    await new Promise((r) => setTimeout(r, 0));


const COUNTER_BYTES=80; // GpuCounters: 20 words (convexLarge at byte 64)
const staging=device.createBuffer({size:COUNTER_BYTES,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
const snapshot=device.createBuffer({size:scene.bodies.length*112,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
const bodyCount=bodies.length,colliderCount=colliders.length,sleepRounds=Math.max(0,Math.min(16,cfg.sleepRounds??8));
let last={contact:0,candidate:0,awake:bodyCount-1,color:16,manifold:0,overflow:0,convexSmall:0,convexLarge:0},steps=0,parity=0,shots=0;
let timing={wallMs:0,encodeMs:0,waitMs:0},initialized=false;

    // Reused every step — allocating these in writeHost showed up as wall−GPU gap on ring.
    const scratchU32 = new Uint32Array(64);
    const colorCountWords = 128;
    const colorCountsBytes = colorCountWords * 4;
    // One queue upload + GPU copies into the four step buffers (cheaper than 4 writeBuffers).
    const hostUpload = device.createBuffer({
      size: 256 + 256 + 96 + colorCountsBytes,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const hostUploadCpu = new ArrayBuffer(256 + 256 + 96 + colorCountsBytes);
    const hostScratchView = new Uint32Array(hostUploadCpu, 0, 64);
    const hostParamsView = new Uint8Array(hostUploadCpu, 256, 208);
    const hostCountersView = new Uint32Array(hostUploadCpu, 512, COUNTER_BYTES / 4);
    const hostColorsView = new Uint32Array(hostUploadCpu, 608, colorCountWords);
    // Previous contacts bound the warm hash; host field and clear dispatch must agree.
    const hashN = Math.min(nextPow2(Math.max(cap.maxContacts * 4, 1024)), csrBase, MAX_WARM_HASH);

    function scratchWords() {
      const s = scratchU32;
      s.fill(0);
      s[1] = cap.maxBodies;
      s[2] = cap.maxCandidates;
      s[3] = cap.maxContacts;
      s[4] = Math.min(cap.maxColors, 64);
      s[7] = steps === 0 ? 0 : Math.min(last.contact, cap.maxContacts);
      s[8] = parity === 0 ? cap.maxContacts : 0;
      s[9] = hashN;
      s[10] = cap.maxCandidates;
      s[11] = parity === 0 ? 0 : cap.maxContacts;
      let log = 0;
      while ((1 << log) < Math.max(bodyCount, 1) && log < 20) log++;
      s[12] = log === 0 ? 0 : (log - 1) & 1;
      s[16] = gridHashWords;
      s[17] = initialized ? Math.max(1, steps) : 0;
      // Incremental colouring (as the Rust host): colours persist in a history
      // row and only bodies whose neighbourhood changed recolour (work slot 69);
      // small repair lists are coloured by one workgroup (local repair).
      s[26] = incremental ? 1 : 0;
      s[27] = incremental ? 1 : 0;
      s[18] = grid.levels;   // hierarchical broadphase levels
      s[29] = 1;             // degree-aware primal lanes (bufs[17] holds 128 words)
      s[30] = hasFilters ? 1 : 0; // collision filters present
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
      hostCountersView[3] = jointCount;
      hostColorsView.fill(0);
      device.queue.writeBuffer(hostUpload, 0, hostUploadCpu);
      // Prefer GPU copies from one upload — four tiny writeBuffers were ~0.3 ms of ring wall.
      enc.copyBufferToBuffer(hostUpload, 0, bufs[21], 0, 256);
      enc.copyBufferToBuffer(hostUpload, 256, bufs[6], 0, 208);
      // Keep the previous GPU count before clearing counters. A benchmark queues
      // ten steps between CPU readbacks; last.contact is stale within that batch.
      if (steps > 0) enc.copyBufferToBuffer(bufs[7], 8, bufs[6], 80 + 7 * 4, 4);
      enc.copyBufferToBuffer(hostUpload, 512, bufs[7], 0, COUNTER_BYTES);
      enc.copyBufferToBuffer(hostUpload, 608, bufs[17], 0, colorCountsBytes);
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

    // Collision only (bounds, broadphase, narrowphase incl. convex). `phase`
    // marks profiling boundaries; it is a no-op for ordinary steps.
    function encodeCollision(pass, phase, reuse) {
      const nCols = Math.max(colliderCount, 1), nBodies = Math.max(bodyCount, 1);
      disp(pass, 'build_aabbs', nCols);
      if (jointCount) {
        // Exact bonded-pair set for the broadphase (broken bonds are left out).
        disp(pass, 'joint_hash_clear', jointHashSlots);
        disp(pass, 'joint_hash_insert', jointCount);
      }
      pass = phase('bounds');
      disp(pass, 'broadphase_clear', gridHashWords);
      disp(pass, 'broadphase_insert', nCols);
      pass = phase('broadphaseInsert');
      disp(pass, 'broadphase_query', nCols);
      disp(pass, 'write_work_args', 64);
      pass = phase('broadphase');
      if (reuse) {
        disp(pass, 'update_refs', nBodies);
        disp(pass, 'contact_hash_clear', hashN);
        dispWork(pass, 'contact_hash_insert', 68);
      }
      dispWork(pass, 'narrowphase', 64);
      pass = phase('narrowphase');
      if (hasConvex) {
        dispWork(pass, 'convex_classify', 64);
        disp(pass, 'convex_write_args', 64);
        pass = phase('convexClassify');
        dispWork(pass, 'convex_sat_small', 74);
        pass = phase('convexSatSmall');
        dispWork(pass, 'convex_sat_large', 75);
        pass = phase('convexSat');
        dispWork(pass, 'convex_manifold', 76);
        pass = phase('convexManifold');
      }
      return pass;
    }

    // beginPass(name) ends a profiled phase `name` and opens the next pass
    // ('start' opens the first). Without it, one pass (optionally timestamped).
    function encodeStep(enc, readback, beginPass, passDescriptor, reuseManifolds = false) {
      const nBodies = Math.max(bodyCount, 1);
      writeHost(enc);
      let pass = beginPass ? beginPass('start') : enc.beginComputePass(passDescriptor);
      const phase = (name) => {
        if (!beginPass) return pass;
        pass.end();
        pass = beginPass(name);
        return pass;
      };
      const reuse = (params.flags & 1) !== 0;
      if (reuseManifolds) {
        // three-avbd reuseManifolds: no bounds, broadphase or narrowphase.
        disp(pass, 'write_work_args', 64);
        dispWork(pass, 'reuse_all_manifolds', 73);
        pass = phase('reuseManifolds');
      } else pass = encodeCollision(pass, phase, reuse);
      disp(pass, 'write_work_args', 64);
      if (!reuse && !reuseManifolds) {
        disp(pass, 'contact_hash_clear', hashN);
        dispWork(pass, 'contact_hash_insert', 68);
      }
      dispWork(pass, 'persist_contacts', 65);
      if (jointCount) {
        // Bonded bodies wake together (one bond hop per round), then the
        // awake bonds capture their step-start violation.
        for (let round = 0; round < BOND_WAKE_ROUNDS; round++) disp(pass, 'bond_wake', jointCount);
        disp(pass, 'warmstart_joints', jointCount);
      }
      pass = phase('persistence');
      disp(pass, 'build_awake_list', nBodies);
      disp(pass, 'write_work_args', 64);
      disp(pass, 'csr_clear', nBodies + 1);
      dispWork(pass, 'csr_degree_contacts', 66);
      if (jointCount) disp(pass, 'csr_degree_joints', jointCount);
      if (bodyCount <= 512) {
        disp(pass, 'csr_prefix', 1);
      } else {
        // Three-dispatch block scan (not log N Hillis–Steele passes).
        disp(pass, 'csr_prefix_blocks', Math.ceil(nBodies / 4));
        disp(pass, 'csr_prefix_block_sums', 64);
        disp(pass, 'csr_prefix_add', nBodies);
      }
      dispWork(pass, 'csr_scatter_contacts', 66);
      if (jointCount) disp(pass, 'csr_scatter_joints', jointCount);
      dispWork(pass, 'csr_sort', 67);
      pass = phase('graph');
      dispWork(pass, 'init_colors', 67);
      disp(pass, 'write_work_args', 64);
      // Complete graph coloring before solving; the shader reports unfinished colors.
      // Slot 69 is zero when the local pass handled the repair list. Dense debris
      // (100k awake pieces) can need more than 32 Jones-Plassmann rounds.
      if (incremental) disp(pass, 'color_bodies_local', 64);
      for (let round = 0; round < (budget.colorRounds ?? 32); round++) dispWork(pass, 'color_bodies', incremental ? 69 : 67, round % 16);
      pass = phase('coloring');
      // Stable per-colour body order (locality across every primal iteration).
      disp(pass, 'count_color_tiles', nBodies);
      disp(pass, 'prefix_color_tiles_parallel', Math.min(cap.maxColors, 64) * 64);
      disp(pass, 'scatter_color_tiles', nBodies);
      dispWork(pass, 'record_color_degrees', 67);
      disp(pass, 'write_color_args', 64);
      disp(pass, 'predict_bodies', nBodies);
      dispWork(pass, 'pack_solver_contacts', 66);
      pass = phase('topology');
      // Current-step indirect arguments skip empty colors without a CPU estimate.
      const colors = Math.min(cap.maxColors, 64);
      for (let iter = 0; iter < params.iterations; iter++) {
        const primal = pipes.solve_primal_color;
        for (let color = 0; color < colors; color++) {
          pass.setPipeline(primal.pipeline);
          pass.setBindGroup(0, primal.bind, [color * PASS_ALIGN]);
          pass.dispatchWorkgroupsIndirect(bufs[23], color * 12);
        }
        dispWork(pass, 'solve_dual', 65, iter + 1 === params.iterations ? 1 : 0);
      }
      pass = phase('solve');
      disp(pass, 'extract_velocity_sleep', nBodies);
      disp(pass, 'write_sleep_args', 64);
      for (let i = 0; i < sleepRounds; i++) dispWork(pass, 'sleep_island_gate', 66);
      dispWork(pass, 'sleep_island_commit', 67);
      if (beginPass) phase('sleep');
      pass.end();
      if (readback) enc.copyBufferToBuffer(bufs[7], 0, staging, 0, COUNTER_BYTES);
    }

    let manifoldsValid = false;
    function submitPhysics(reuseManifolds = false) {
      const enc = device.createCommandEncoder();
      encodeStep(enc, false, null, undefined, reuseManifolds && manifoldsValid);
      manifoldsValid = true;
      device.queue.submit([enc.finish()]);
      initialized=true;
      parity ^= 1;
      steps++;
    }

    async function readCounters() {
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(bufs[7], 0, staging, 0, COUNTER_BYTES);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const c = new Uint32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      last = { contact: c[2], candidate: c[4], awake: c[5], color: c[6], manifold: c[12], overflow: c[7], convexSmall: c[11], convexLarge: c[16] };
    }


return {
 bodyBuffer:bufs[0],cap,
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
     if(singlePass){names.push('step');created=1;encodeStep(enc,false,null,stamps(0));}
     else encodeStep(enc,false,(name)=>{if(name!=='start')names.push(name);return enc.beginComputePass(stamps(created++))});
     const n=names.length;
     enc.resolveQuerySet(query,0,created*2,resolve,0);enc.copyBufferToBuffer(resolve,0,read,0,created*16);
     device.queue.submit([enc.finish()]);initialized=true;parity^=1;steps++;manifoldsValid=true;
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
 async inspectCollisions(){
   const enc=device.createCommandEncoder(),savedFlags=params.flags;params.flags=0;writeHost(enc);params.flags=savedFlags;let pass=enc.beginComputePass();
   pass=encodeCollision(pass,()=>pass,false);pass.end();device.queue.submit([enc.finish()]);manifoldsValid=false;
   await readCounters();
   if(last.overflow)throw new Error(`Collision overflow ${last.overflow}`);
   const read=async(buffer,offset,size)=>{
     if(!size)return new ArrayBuffer(0);
     const temp=device.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
     try{const e=device.createCommandEncoder();e.copyBufferToBuffer(buffer,offset,temp,0,size);device.queue.submit([e.finish()]);await temp.mapAsync(GPUMapMode.READ);return temp.getMappedRange().slice(0)}finally{temp.destroy()}
   };
   const base=parity===0?0:cap.maxContacts;
   return {counters:{...last},pairs:new Uint32Array(await read(bufs[10],0,last.candidate*16)),
     bounds:new Float32Array(await read(bufs[9],cap.maxBodies*32,colliderCount*32)),
     contacts:new Uint32Array(await read(bufs[4],base*32,last.contact*32)),
     points:new Float32Array(await read(bufs[27],base*64,last.contact*64)),
     manifolds:new Uint32Array(await read(bufs[25],0,last.manifold*32))};
 },
 get shots(){return shots},
 // Joint lambda0..5 and k0..5 per joint (preloading, diagnostics).
 async jointState(){
   if(!jointCount)return {lambda:[],k:[]};
   const size=jointCount*JOINT_BYTES,read=device.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
   try{const enc=device.createCommandEncoder();enc.copyBufferToBuffer(bufs[5],0,read,0,size);device.queue.submit([enc.finish()]);
     await read.mapAsync(GPUMapMode.READ);const f=new Float32Array(read.getMappedRange().slice(0));read.unmap();const w=JOINT_BYTES/4;
     return {lambda:Array.from({length:jointCount},(_,i)=>Array.from(f.subarray(i*w+24,i*w+30))),k:Array.from({length:jointCount},(_,i)=>Array.from(f.subarray(i*w+30,i*w+36)))};}
   finally{read.destroy()}
 },
 // Joint flags (bit 0 = broken), read back for tests and the HUD.
 async jointFlags(){
   if(!jointCount)return new Uint32Array(0);
   const size=jointCount*JOINT_BYTES,read=device.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
   try{const enc=device.createCommandEncoder();enc.copyBufferToBuffer(bufs[5],0,read,0,size);device.queue.submit([enc.finish()]);
     await read.mapAsync(GPUMapMode.READ);const v=new Uint32Array(read.getMappedRange().slice(0));read.unmap();
     return Uint32Array.from({length:jointCount},(_,i)=>v[i*JOINT_BYTES/4+3]);}finally{read.destroy()}
 },
 set incrementalColoring(v){incremental=!!v},
 // Live solver parameters (dt, iterations, ...), packed again every step.
 get params(){return params},
 launch(origin,direction,{speed,arc=scene.projectileArc??true}={}){
   if(shots>=scene.projectileCount)throw new Error('All 16 projectiles fired. Reset to reload.');
   if(![...origin,...direction].every(Number.isFinite))throw new Error('Invalid projectile ray');
   const id=scene.projectileStart+shots,mass=scene.projectileMass,r=scene.projectileRadius;
   speed??=scene.projectileSpeed??Math.max(5,Math.min(18,scene.extent*3));
   const distance=Math.max(r,scene.center.reduce((sum,x,i)=>sum+(x-origin[i])*direction[i],0));
   const v=direction.map(x=>x*speed);if(arc)v[1]+=.5*9.81*distance/speed;
   const body={p:origin,q:[0,0,0,1],v,invMass:1/mass,invI:Array(3).fill(1/(.4*mass*r*r)),flags:9};
   const bytes=new ArrayBuffer(112);writeBody(bytes,0,body);
   device.queue.writeBuffer(bufs[0],id*112,bytes);device.queue.writeBuffer(bufs[0],(cap.maxBodies+id)*112,bytes);manifoldsValid=false;
   last.awake=Math.max(1,last.awake);shots++;return id;
 },
 get steps(){return steps},get counters(){return {...last}},get timing(){return {...timing}},
 // {reuseManifolds:true}: three-avbd whole-step manifold reuse (skips collision
 // detection; falls back to a full step after any scene change).
 async step(count=1,{reuseManifolds=false}={}){
   if(!Number.isInteger(count)||count<1||count>4)throw new Error('Physics batch must have 1–4 ticks');
   if(initialized&&last.awake===0){steps+=count;timing={wallMs:0,encodeMs:0,waitMs:0};return;}
   const start=performance.now();for(let i=0;i<count;i++)submitPhysics(reuseManifolds);const submitted=performance.now();
   await readCounters();const end=performance.now();
   // Includes queue execution and mapAsync synchronization, not pure GPU time.
   timing={wallMs:(end-start)/count,encodeMs:(submitted-start)/count,waitMs:(end-submitted)/count};
   if(last.overflow){manifoldsValid=false;throw new Error(`Physics capacity exceeded (0x${last.overflow.toString(16)}); choose a coarser level`);}
 },
 // Upload the arena's changed records; grows the hull buffer (power of two,
 // contents copied) and rebuilds the bind groups that reference it.
 syncHulls(arena){
   const needed=arena.records;
   if(needed>hullRecordsCapacity){
     const records=2**Math.ceil(Math.log2(needed)),size=records*16;
     if(size>device.limits.maxStorageBufferBindingSize)throw new Error('Hull arena exceeds the storage binding limit');
     const grown=device.createBuffer({label:'AVBD 18',size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC|GPUBufferUsage.STORAGE});
     const enc=device.createCommandEncoder();enc.copyBufferToBuffer(bufs[18],0,grown,0,hullRecordsCapacity*16);device.queue.submit([enc.finish()]);
     bufs[18].destroy();bufs[18]=grown;hullRecordsCapacity=records;
     for(const [name,p] of Object.entries(pipes))if(p.binds.some(b=>b.index===18))
       p.bind=device.createBindGroup({label:name,layout:p.layout,entries:p.binds.map(b=>({binding:b.index,resource:b.index===22?{buffer:bufs[22],size:16}:{buffer:bufs[b.index]}}))});
   }
   const words=arena.data;
   for(const [a,b] of arena.takeDirty())device.queue.writeBuffer(bufs[18],a*16,words.buffer,a*16,(b-a)*16);
   manifoldsValid=false;
 },
 // Replace a collider (hull swap, filter change). Release old hulls yourself.
 setCollider(index,c){
   if(!(index>=0&&index<colliderCount))throw new Error('collider out of range');
   colliders[index]=c;const bytes=new ArrayBuffer(64);writeCollider(bytes,0,c);
   device.queue.writeBuffer(bufs[2],index*64,bytes);writeCollider(colBytes,index,c);
   hasConvex||=c.shape===3;
   hasFilters=colliders.some((x)=>{const f=filterOf(x);return f.generation!==0||(f.material>>>16)!==0});
   manifoldsValid=false;
 },
 async snapshot(){const enc=device.createCommandEncoder();enc.copyBufferToBuffer(bufs[0],0,snapshot,0,scene.bodies.length*112);device.queue.submit([enc.finish()]);await snapshot.mapAsync(GPUMapMode.READ);const bytes=snapshot.getMappedRange().slice(0);snapshot.unmap();return new Float32Array(bytes);},
 reset(){
   manifoldsValid=false;
   const enc=device.createCommandEncoder();
   for(let i=0;i<bufs.length;i++)if(![0,2,3,18,22].includes(i))enc.clearBuffer(bufs[i]);
   device.queue.submit([enc.finish()]);
   device.queue.writeBuffer(bufs[0],0,bodyBytes);device.queue.writeBuffer(bufs[2],0,colBytes);
   device.queue.writeBuffer(bufs[5],0,jointBytes);
   steps=0;parity=0;shots=0;initialized=false;last={contact:0,candidate:0,awake:bodyCount-1,color:16,manifold:0,overflow:0,convexSmall:0,convexLarge:0};
 },
 dispose(){for(const b of [...bufs,staging,snapshot,hostUpload])b.destroy();},
};
} catch (error) { for (const buffer of bufs) buffer.destroy(); throw error; }
}
