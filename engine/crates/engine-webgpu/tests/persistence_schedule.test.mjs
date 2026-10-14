import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

// Execute the shipped encoder, not a copy of its scheduling logic. Stale compact
// tags in reserved holes become phantom contacts if clear moves after persistence.
const html = readFileSync(new URL('../demo.html', import.meta.url), 'utf8');
const start = html.indexOf('function encodeStep(');
const end = html.indexOf('function submitPhysics()', start);
assert.ok(start >= 0 && end > start);
const source = html.slice(start, end);

for (const [query, expected] of [
  ['?scene=ring110k', true], ['?scene=gables506k', true],
  ['?scene=columns', true], ['?scene=ring110k&prefix=serial', false],
  ['?scene=ring110k&prefix=parallel', true],
]) {
  test(`color prefix default and control are explicit (${query})`, () => {
    const mode=html.slice(html.indexOf('const qs ='),html.indexOf('const regressionKind ='));
    assert.equal(runInNewContext(`${mode}\nparallelColorPrefix`,{URLSearchParams,location:{search:query}}),expected);
  });
}

for (const [query, expected] of [
  ['?scene=ring110k',10], ['?scene=columns',32], ['?scene=gables506k',32],
  ['?scene=ring110k&sparse=32',32], ['?scene=ring110k&sparse=0',0],
  ['?scene=ring110k&sparse=8',8],
]) {
  test(`sparse scheduling mode preserves scene defaults and explicit controls (${query})`, () => {
    const mode=html.slice(html.indexOf('const qs ='),html.indexOf('const regressionKind ='));
    assert.equal(runInNewContext(`${mode}\nsparseStart`,{URLSearchParams,location:{search:query}}),expected);
  });
}
for(const value of ['-1','33','2.5','invalid']) {
  test(`invalid sparse color ${value} fails before allocating GPU work`,()=>{
    const mode=html.slice(html.indexOf('const qs ='),html.indexOf('const regressionKind ='));
    assert.throws(()=>runInNewContext(mode,{URLSearchParams,location:{search:`?scene=ring110k&sparse=${value}`}}),/sparse must be an integer color index from 0 through 32/);
  });
}

// Execute shipped mode selection, allocation and shader routing together. A
// wrong alias can bind 32-byte metadata to a shader expecting 128-byte contacts;
// WebGPU robust access may hide that mistake as missing collision work.
for (const [query, expectedMode] of [
  ['?scene=ring110k', 'persistent'], ['?scene=gables506k', 'persistent'],
  ['?scene=gables506k&contacts=canonical', 'canonical'], ['?scene=columns', 'canonical'],
  ['?scene=ring110k&contacts=canonical', 'canonical'],
  ['?scene=ring110k&contacts=compact', 'compact'],
  ['?scene=gables506k&contacts=persistent', 'persistent'],
]) {
  test(`contact allocation and shader ABI agree (${query})`, () => {
    const mode = html.slice(html.indexOf('const qs ='), html.indexOf('const regressionKind ='));
    const allocation = html.slice(html.indexOf('function bufferSizes()'), html.indexOf('const K = ['));
    const routing = html.slice(html.indexOf('const persistentEntries ='), html.indexOf('const sources ='));
    const cap = {maxBodies:64,maxColliders:128,maxContacts:257,maxCandidates:129,maxJoints:3,
      maxColors:32,maxCommands:8,maxHull:16};
    const output = runInNewContext(`${mode}\n${allocation}\n${routing}\n({
      persistentContacts, compactContacts, sizes:bufferSizes(),
      entries:['narrowphase','contact_hash_insert','persist_contacts','persist_manifolds_compact',
        'pack_solver_contacts','solve_primal_color_compact','solve_dual_compact'].map(shaderEntry)
    })`, {URLSearchParams,location:{search:query},cap,csrBase:32768,PASS_ALIGN:256});
    assert.equal(output.persistentContacts,expectedMode === 'persistent');
    assert.equal(output.compactContacts,expectedMode !== 'canonical');
    assert.equal(output.sizes[4],cap.maxContacts*2*(expectedMode === 'persistent' ? 32 : 128),
      'Metadata/canonical allocation must match the selected shader stride and cover both history halves.');
    assert.equal(output.sizes[27],expectedMode === 'persistent' ? cap.maxContacts*2*64 : expectedMode === 'compact' ? cap.maxContacts*64 : 64,
      'Persistent solver points must allocate both halves; single-frame scratch is insufficient after the first swap.');
    const expected = expectedMode === 'persistent'
      ? ['narrowphase_persistent','contact_hash_insert_persistent','persist_contacts_persistent',
        'persist_manifolds_persistent','pack_solver_pairs_persistent','solve_primal_color_persistent','solve_dual_persistent']
      : ['narrowphase','contact_hash_insert','persist_contacts','persist_manifolds_compact',
        'pack_solver_contacts','solve_primal_color_compact','solve_dual_compact'];
    assert.deepEqual(Array.from(output.entries),expected,'Every contact reader/writer must select the same storage ABI.');
  });
}

for (const mode of ['canonical', 'compact', 'manifold', 'fused', 'persistent', 'persistent-fused']) {
  for (const profiling of [false, true]) {
    for (const [detailedProfile,reuse,graphProfile] of [[false,true,false],[true,true,false],[true,false,false],[false,true,true],[false,false,true]]) {
    for (const [sparseStart,parallelColorPrefix] of [[8,false],[32,false],[8,true]]) {
    test(`shipped encoder preserves persistence dependencies (${mode}, profiling=${profiling}, detail=${detailedProfile}, graph=${graphProfile}, reuse=${reuse}, sparseStart=${sparseStart}, parallelPrefix=${parallelColorPrefix})`, () => {
      const persistentContacts = mode.startsWith('persistent');
      const fused = mode === 'fused' || mode === 'persistent-fused';
      const trace = [], passes = [];
      const pass = () => {
        const p = {id:passes.length,
          setPipeline() {}, setBindGroup() {},
          dispatchWorkgroupsIndirect() { trace.push({name: 'primal'}); },
          end() { this.ended = true; },
        };
        passes.push(p);
        return p;
      };
      const context = {
        detailedProfile, graphProfile, colorProfile:false, bodyCount:110442, colliderCount:110442, last:{contact:1_170_000},
        cap:{maxContacts:2097152,maxJoints:64,maxColors:32},
        nextPow2:n => 2 ** Math.ceil(Math.log2(n)), MAX_WARM_HASH:2097152,
        persistentContacts, params:{flags:reuse?1:0,iterations:4}, compactContacts:mode !== 'canonical',
        manifoldPersistence:mode === 'manifold', fusedPersistence:fused,
        manifoldHashInsert:false,
        physicalAdjacency:true, incrementalColors:true, stablePacking:true, sparseStart, parallelColorPrefix,
        sleepRounds:8, PASS_ALIGN:256, bufs:[],
        pipes:{solve_primal_color:{},solve_primal_color_compact:{}},
        writeHost() { trace.push({name:'host'}); },
        disp(_pass,name,threads,arg) { trace.push({name,threads,arg,phase:_pass.id}); },
        dispWork(_pass,name,slot,arg) { trace.push({name,slot,arg,phase:_pass.id}); },
        encoder:{beginComputePass:pass}, begin:profiling ? pass : undefined,
      };
      runInNewContext(`${source}\nencodeStep(encoder, false, begin)`, context);
      const positions = name => trace.flatMap((call,i) => call.name === name ? [i] : []);
      const persistence = fused ? 'persist_manifolds_compact' : mode === 'manifold' ? 'persist_manifolds' : 'persist_contacts';
      assert.equal(positions(persistence).length,1,'Warm-start must run once, not decay forces twice.');
      const at = positions(persistence)[0];
      assert.equal(trace[at].slot,(fused || mode === 'manifold') ? 66 : 65,
        'Persistence dispatch must use the current GPU manifold/contact count, not a stale CPU estimate.');
      assert.ok(positions('narrowphase')[0] < at && at < positions('predict_bodies')[0],
        'Warm start evaluates current step-start poses after collision generation and before prediction.');
      const clears = positions('clear_solver_contacts'), packs = positions('pack_solver_contacts');
      if (persistentContacts) {
        assert.equal(clears.length,0,'Persistent points are initialized by narrowphase; clearing here destroys current ownership and warm-start state.');
        assert.equal(packs.length,fused ? 0 : 1,'Persistent mode prepares pairs once, without repacking points.');
        if (!fused) assert.ok(at < packs[0] && packs[0] < positions('primal')[0], 'Pair metadata must be prepared before solving.');
      } else if (mode === 'fused') {
        assert.equal(clears.length,1);
        assert.ok(positions('narrowphase')[0] < clears[0] && clears[0] < at,
          'Clear dead tags after current contact counts exist but before fused writes; otherwise holes become live or valid tags are erased.');
        assert.equal(packs.length,0,'Fused points must not be repacked from canonical state later.');
      } else if (mode !== 'canonical') {
        assert.equal(clears.length,1); assert.equal(packs.length,1);
        assert.ok(at < clears[0] && clears[0] < packs[0] && packs[0] < positions('primal')[0],
          'Separate compact preparation must consume persisted state and finish before the first primal.');
      } else {
        assert.equal(clears.length,0); assert.equal(packs.length,0);
      }
      const prefixCall = trace.find(c => c.name === (parallelColorPrefix ? 'prefix_color_tiles_parallel' : 'prefix_color_tiles'));
      assert.equal(prefixCall.threads, parallelColorPrefix ? 32 * 64 : 64,
        'Cooperative prefix needs one complete workgroup per color; a single group silently omits every later color.');
      const dual = trace.filter(c => c.name === (mode === 'canonical' ? 'solve_dual' : 'solve_dual_compact'));
      assert.deepEqual(dual.map(c=>c.arg),[0,0,0,1], 'Publish canonical state only on the final dual iteration.');
      assert.equal(positions('primal').length,4*sparseStart,'Every prefix color must remain encoded in each iteration.');
      const sparseName = mode === 'canonical' ? 'solve_primal_sparse' : 'solve_primal_sparse_compact';
      const tail = trace.filter(c => c.name === sparseName);
      assert.equal(tail.length,sparseStart < 32 ? 4 : 0,'Dispatch one complete tail per iteration, never drop unobserved GPU colors.');
      for (const c of tail) { assert.equal(c.threads,64,'Sparse tail MUST use exactly one workgroup; multiple groups race colors.'); assert.equal(c.arg,sparseStart); }
      if(sparseStart < 32) {
        const dualName = mode === 'canonical' ? 'solve_dual' : 'solve_dual_compact';
        const d = positions(dualName), t = positions(sparseName);
        for(let i=0;i<4;i++) {
          const begin = i===0 ? 0 : d[i-1]+1;
          assert.equal(trace.slice(begin,t[i]).filter(c=>c.name==='primal').length,sparseStart);
          assert.equal(t[i]+1,d[i],'Tail colors must complete immediately before dual, inside the same primal phase.');
        }
      }
      assert.equal(passes.length,profiling ? 14 + (graphProfile ? 8 : 0) + (detailedProfile ? (reuse?3:2) : 0) : 1,'Detailed timestamps must add boundaries only to profiling, never to throughput batches.');
      if(profiling && detailedProfile){
        const phase = name => trace.find(c=>c.name===name).phase;
        assert.equal(phase('contact_hash_clear'),3,'Hash clearing must have its own timestamp interval.');
        assert.equal(phase('contact_hash_insert'),4,'Hash insertion must have its own timestamp interval.');
        assert.equal(phase('narrowphase'),reuse?5:2,'Narrowphase timing must exclude hash/reference work.');
        assert.equal(phase(persistence),reuse?6:5,'Persistence must not be charged to hash insertion.');
        if(reuse) assert.equal(phase('update_refs'),2);
      }
      if (profiling && graphProfile) {
        const expected = [
          ['build_awake_list','write_work_args'],
          ['csr_clear','csr_degree_contacts'],
          ['csr_prefix_blocks','csr_prefix_block_sums','csr_prefix_add'],
          ['csr_scatter_contacts'], ['csr_sort'], ['init_colors','write_work_args'],
          Array(32).fill('color_bodies'),
          ['count_color_tiles',parallelColorPrefix ? 'prefix_color_tiles_parallel' : 'prefix_color_tiles','scatter_color_tiles','write_color_args'],
          ['predict_bodies'],
        ];
        for (let stage=0; stage<expected.length; stage++) {
          assert.deepEqual(trace.filter(c=>c.phase===stage+4).map(c=>c.name), expected[stage],
            `Graph timestamp stage ${stage} has moved across a dependency or charges unrelated work; fix boundaries before trusting performance.`);
        }
      }
      assert.ok(passes.every(p=>p.ended));
    });
    }
  }
}

}

// Opt-in three-avbd hashInsert shape. Runs the shipped mode parser, host upload
// and encoder: per-manifold insertion reads the previous step's gManifolds, so it
// must use the copied GPU manifold count and run before narrowphase overwrites it.
const modeSource = html.slice(html.indexOf('const qs ='), html.indexOf('const regressionKind ='));
for (const [query, expected] of [['?scene=ring110k','manifold'],['?scene=gables506k','manifold'],
  ['?scene=columns','contact'],['?scene=ring110k&hashInsert=contact','contact'],
  ['?scene=gables506k&hashInsert=contact','contact'],['?scene=columns&hashInsert=manifold','manifold']]) {
  test(`hash insert mode parses explicitly (${query})`, () => {
    assert.equal(runInNewContext(`${modeSource}\nhashInsertMode`,{URLSearchParams,location:{search:query}}),expected);
  });
}
test('invalid hash insert mode fails before allocating GPU work', () => {
  assert.throws(()=>runInNewContext(modeSource,{URLSearchParams,location:{search:'?scene=ring110k&hashInsert=pair'}}),
    /hashInsert must be contact or manifold/);
});

function encodeTrace({reuse, manifoldHashInsert, detailedProfile=false}) {
  const trace = [], passes = [];
  const pass = () => { const p = {id:passes.length, setPipeline() {}, setBindGroup() {},
    dispatchWorkgroupsIndirect() { trace.push({name:'primal'}); }, end() {}}; passes.push(p); return p; };
  const context = {
    detailedProfile, graphProfile:false, colorProfile:false, bodyCount:110442, colliderCount:110442, last:{contact:1_170_000},
    cap:{maxContacts:2097152,maxJoints:64,maxColors:32},
    nextPow2:n => 2 ** Math.ceil(Math.log2(n)), MAX_WARM_HASH:2097152,
    persistentContacts:true, params:{flags:reuse?1:0,iterations:4}, compactContacts:true,
    manifoldPersistence:false, fusedPersistence:false, manifoldHashInsert,
    physicalAdjacency:true, incrementalColors:true, stablePacking:true, sparseStart:10, parallelColorPrefix:true,
    sleepRounds:8, PASS_ALIGN:256, bufs:[], pipes:{solve_primal_color:{},solve_primal_color_compact:{}},
    writeHost() {}, disp(p,name,threads) { trace.push({name,threads,phase:p.id}); },
    dispWork(p,name,slot) { trace.push({name,slot,phase:p.id}); },
    encoder:{beginComputePass:pass}, begin:detailedProfile ? pass : undefined,
  };
  runInNewContext(`${source}\nencodeStep(encoder, false, begin)`, context);
  return trace;
}
for (const detailedProfile of [false, true]) {
  test(`per-manifold hash insert replaces the per-contact insert before narrowphase (detail=${detailedProfile})`, () => {
    const trace = encodeTrace({reuse:true, manifoldHashInsert:true, detailedProfile});
    const at = name => trace.flatMap((c,i)=>c.name===name?[i]:[]);
    assert.equal(at('contact_hash_insert').length,0,'Per-contact insert must not also run; two inserters race the same table.');
    assert.equal(at('contact_hash_insert_manifolds').length,1);
    const insert = trace[at('contact_hash_insert_manifolds')[0]];
    assert.equal(insert.slot,68,'Insert must use GPU work slot 68 (previous manifold count when HOST_S(14) is set).');
    assert.ok(at('contact_hash_clear')[0] < at('contact_hash_insert_manifolds')[0] &&
      at('contact_hash_insert_manifolds')[0] < at('narrowphase')[0],
      'Previous manifolds are overwritten by narrowphase: insert must follow the clear and precede narrowphase.');
    if (detailedProfile) assert.equal(insert.phase,4,'Per-manifold insert keeps the hashInsert timestamp interval.');
  });
}
test('per-manifold hash insert is rejected when reuse would place it after narrowphase', () => {
  assert.throws(()=>encodeTrace({reuse:false, manifoldHashInsert:true}),/requires contact reuse/);
  assert.doesNotThrow(()=>encodeTrace({reuse:false, manifoldHashInsert:false}));
});

test('host upload publishes the previous GPU manifold count before counters reset', () => {
  const hostSource = html.slice(html.indexOf('function scratchWords()'), html.indexOf('function groups('));
  for (const [manifoldHashInsert, steps] of [[true,5],[true,0],[false,5]]) {
    const copies = [];
    const context = {
      manifoldHashInsert, steps, parity:1, bodyCount:10, colliderCount:10, incrementalColors:true,
      csrBase:32768, MAX_WARM_HASH:2097152, nextPow2:n => 2 ** Math.ceil(Math.log2(n)),
      last:{contact:100}, cap:{maxBodies:64,maxCandidates:128,maxContacts:256,maxColors:32},
      scratchU32:new Uint32Array(64), packHost() {}, packHostU8:new Uint8Array(208),
      hostScratchView:new Uint32Array(64), hostParamsView:new Uint8Array(208),
      hostCountersView:new Uint32Array(16), hostColorsView:new Uint32Array(32),
      device:{queue:{writeBuffer() {}}}, hostUpload:'upload', hostUploadCpu:null, colorCountsBytes:128,
      bufs:Array.from({length:32},(_,i)=>`b${i}`),
    };
    const s = runInNewContext(`${hostSource}\nconst s = scratchWords(); writeHost({copyBufferToBuffer(...a){ copies.push(a); }}); s`,
      {...context, copies});
    assert.equal(s[14], manifoldHashInsert ? 1 : 0, 'HOST_S(14) selects the per-manifold slot-68 count.');
    const manifoldCopy = copies.findIndex(c=>c[0]==='b7' && c[1]===48 && c[2]==='b6' && c[3]===80+13*4 && c[4]===4);
    const reset = copies.findIndex(c=>c[0]==='upload' && c[2]==='b7');
    if (manifoldHashInsert && steps > 0) {
      assert.ok(manifoldCopy >= 0,'Previous manifoldCount (counter byte 48) must reach HOST_S(13); a stale CPU count omits batch work.');
      assert.ok(manifoldCopy < reset,'Copy the manifold count before the host counter reset zeroes it.');
    } else {
      assert.equal(manifoldCopy,-1,'No previous manifolds exist on step 0, and the control must not change uniforms.');
    }
  }
});


// profile=colors splits every primal color (and the sparse tail) into its own
// timestamped pass; the profiler's pass count must match the shipped encoder.
for (const sparseStart of [10, 32, 0]) {
  test(`color profile isolates every encoded primal color (sparseStart=${sparseStart})`, () => {
    const passes = [], trace = [];
    const pass = () => { const p = {id:passes.length, setPipeline() {}, setBindGroup() {},
      dispatchWorkgroupsIndirect() { trace.push({name:'primal', phase:p.id}); }, end() {}}; passes.push(p); return p; };
    const context = {
      detailedProfile:false, graphProfile:false, colorProfile:true, bodyCount:110442, colliderCount:110442,
      last:{contact:1_170_000}, cap:{maxContacts:2097152,maxJoints:64,maxColors:32},
      nextPow2:n => 2 ** Math.ceil(Math.log2(n)), MAX_WARM_HASH:2097152,
      persistentContacts:true, params:{flags:1,iterations:4}, compactContacts:true,
      manifoldPersistence:false, fusedPersistence:false, manifoldHashInsert:true,
      physicalAdjacency:true, incrementalColors:true, stablePacking:true, sparseStart, parallelColorPrefix:true,
      sleepRounds:8, PASS_ALIGN:256, bufs:[], pipes:{solve_primal_color:{},solve_primal_color_compact:{}},
      writeHost() {}, disp(p,name) { trace.push({name, phase:p.id}); },
      dispWork(p,name) { trace.push({name, phase:p.id}); }, encoder:{beginComputePass:pass},
    };
    runInNewContext(`${source}\nencodeStep(encoder, false, encoder.beginComputePass)`, context);
    const prefix = Math.min(32, sparseStart), tail = sparseStart < 32 ? 1 : 0;
    const primalPasses = Math.max(prefix + (tail && sparseStart > 0 ? 1 : 0), 1);
    // Same formula as profileOneStep: parts(9: aabbs,grid,narrowphase,warm,graph) + iterations*(primal+dual) + sleep.
    assert.equal(passes.length, 5 + 4 * (primalPasses + 1) + 1,
      'profile=colors pass count must equal the profiler expectation or every color label shifts.');
    for (const p of new Set(trace.filter(c=>c.name==='primal').map(c=>c.phase)))
      assert.equal(trace.filter(c=>c.phase===p && c.name==='primal').length, 1, 'Each color must own one timestamp interval.');
    assert.ok(trace.filter(c=>c.name==='solve_dual_compact').every(c=>!trace.some(o=>o.phase===c.phase && o.name.startsWith('primal'))),
      'Dual stays in its own interval, never charged to a color.');
  });
}
