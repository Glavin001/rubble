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
// The encoder's dispatch-count constants (8ev), read from the shipped page, with the
// fixed-palette controls on (colors=full, rounds=full): these tests check the full
// schedule; adaptive counts are exercised by the browser regressions.
const dispatchSource = html.slice(html.indexOf('const COLOR_ROUND_LIMIT'), html.indexOf('const colorsFull'));
const DISPATCH = {
  ...runInNewContext(`${dispatchSource}({COLOR_ROUND_LIMIT, COLOR_ROUND_HEADROOM, COLOR_HEADROOM, SLEEP_NEVER_S})`),
  colorsFull: true, roundsFull: true,
};

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
  ['?scene=gables506k&contacts=canonical', 'canonical'], ['?scene=columns', 'persistent'],
  ['?scene=columns&contacts=canonical', 'canonical'],
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
      const context = {...DISPATCH,
        detailedProfile, graphProfile, colorProfile:false, bodyCount:110442, colliderCount:110442, jointCount:0, last:{contact:1_170_000},
        cap:{maxContacts:2097152,maxJoints:64,maxColors:32},
        nextPow2:n => 2 ** Math.ceil(Math.log2(n)), MAX_WARM_HASH:2097152,
        persistentContacts, params:{flags:reuse?1:0,iterations:4}, compactContacts:mode !== 'canonical',
        manifoldPersistence:mode === 'manifold', fusedPersistence:fused,
        manifoldHashInsert:false, gridHashWords:() => 0, gridSplit:false, gridClearWords:() => 131072, reuseCopy:false, npSplit:false, earlyPairPack:false, persistList:false, localRepair:false, carryRest:false, warmRecordHash:false, warmHashSlots:n => Math.min(2 ** Math.ceil(Math.log2(Math.max(n * 4, 1024))), 2097152),
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
for (const [query, expected] of [['?scene=ring110k','record'],['?scene=gables506k','record'],
  ['?scene=ring110k&hashInsert=manifold','manifold'],
  ['?scene=columns','record'],['?scene=ring110k&hashInsert=contact','contact'],
  ['?scene=gables506k&hashInsert=contact','contact'],['?scene=columns&hashInsert=manifold','manifold']]) {
  test(`hash insert mode parses explicitly (${query})`, () => {
    assert.equal(runInNewContext(`${modeSource}\nhashInsertMode`,{URLSearchParams,location:{search:query}}),expected);
  });
}
test('invalid hash insert mode fails before allocating GPU work', () => {
  assert.throws(()=>runInNewContext(modeSource,{URLSearchParams,location:{search:'?scene=ring110k&hashInsert=pair'}}),
    /hashInsert must be contact, manifold or record/);
});

// Split broadphase grid (8ej): every scene's default since P0 (8et).
for (const [query, mode, split] of [['?scene=ring110k','split',true],['?scene=gables506k','split',true],
  ['?scene=columns','split',true],['?scene=ring110k&grid=scaled','scaled',false],
  ['?scene=gables506k&grid=split','split',true],['?scene=ring110k&grid=fixed','fixed',false]]) {
  test(`grid mode parses explicitly (${query})`, () => {
    assert.equal(runInNewContext(`${modeSource}\nJSON.stringify([gridMode, gridSplit])`,{URLSearchParams,location:{search:query}}),JSON.stringify([mode, split]));
  });
}
test('invalid grid mode fails before allocating GPU work', () => {
  assert.throws(()=>runInNewContext(modeSource,{URLSearchParams,location:{search:'?scene=ring110k&grid=half'}}),
    /grid must be fixed, scaled or split/);
});

// Two-pass narrowphase (8ek): default wherever reuse=copy runs.
for (const [query, mode] of [['?scene=ring110k','split'],['?scene=gables506k','split'],
  ['?scene=columns','split'],['?scene=columns&contacts=canonical','fused'],['?scene=ring110k&np=fused','fused'],['?scene=ring110k&reuse=pass','fused']]) {
  test(`narrowphase mode parses explicitly (${query})`, () => {
    assert.equal(runInNewContext(`${modeSource}\nnpMode`,{URLSearchParams,location:{search:query}}),mode);
  });
}
test('np=split without reuse=copy fails before allocating GPU work', () => {
  assert.throws(()=>runInNewContext(modeSource,{URLSearchParams,location:{search:'?scene=ring110k&reuse=pass&np=split'}}),
    /np=split requires reuse=copy/);
  assert.throws(()=>runInNewContext(modeSource,{URLSearchParams,location:{search:'?scene=ring110k&np=pair'}}),
    /np must be fused or split/);
});

function encodeTrace({reuse, manifoldHashInsert, detailedProfile=false, npSplit=false, earlyPairPack=false, persistList=false, localRepair=false, carryRest=false}) {
  const trace = [], passes = [];
  const pass = () => { const p = {id:passes.length, setPipeline() {}, setBindGroup() {},
    dispatchWorkgroupsIndirect() { trace.push({name:'primal'}); }, end() {}}; passes.push(p); return p; };
  const context = {...DISPATCH,
    detailedProfile, graphProfile:false, colorProfile:false, bodyCount:110442, colliderCount:110442, jointCount:0, last:{contact:1_170_000},
    cap:{maxContacts:2097152,maxJoints:64,maxColors:32},
    nextPow2:n => 2 ** Math.ceil(Math.log2(n)), MAX_WARM_HASH:2097152,
    persistentContacts:true, params:{flags:reuse?1:0,iterations:4}, compactContacts:true,
    manifoldPersistence:false, fusedPersistence:false, manifoldHashInsert, gridHashWords:() => 0, gridSplit:false, gridClearWords:() => 131072, reuseCopy:npSplit, npSplit, earlyPairPack, persistList, localRepair, carryRest, warmRecordHash:false, warmHashSlots:n => Math.min(2 ** Math.ceil(Math.log2(Math.max(n * 4, 1024))), 2097152),
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
for (const detailedProfile of [false, true]) {
  test(`two-pass narrowphase defers then tests shapes inside the narrowphase interval (detail=${detailedProfile})`, () => {
    const trace = encodeTrace({reuse:true, manifoldHashInsert:true, detailedProfile, npSplit:true});
    const at = name => trace.flatMap((c,i)=>c.name===name?[i]:[]);
    const defer = at('narrowphase_defer_copy_persistent'), fresh = at('narrowphase_fresh_copy_persistent');
    assert.equal(defer.length,1); assert.equal(fresh.length,1);
    assert.equal(at('narrowphase').length + at('narrowphase_copy_persistent').length,0,
      'The fused narrowphase must not also run; it would emit every pair twice.');
    assert.ok(at('contact_hash_insert_manifolds')[0] < defer[0] && defer[0] < fresh[0] &&
      at('write_work_args').some(i => i > fresh[0]) && !at('write_work_args').some(i => i > defer[0] && i < fresh[0]),
      'Reuse reads the warm hash, shape tests read the deferred tail, and work args need the final manifold count.');
    assert.equal(trace[defer[0]].slot,64); assert.equal(trace[fresh[0]].slot,64);
    if (detailedProfile) assert.equal(trace[defer[0]].phase, trace[fresh[0]].phase,
      'Both passes belong to the narrowphase timestamp interval.');
  });
}

// Packed-pair CSR endpoints (8el): default under persistent contacts.
for (const [query, mode] of [['?scene=ring110k','pairs'],['?scene=gables506k','pairs'],
  ['?scene=columns','pairs'],['?scene=columns&contacts=canonical','manifold'],['?scene=ring110k&endpoints=manifold','manifold'],
  ['?scene=ring110k&persistence=manifold','pairs']]) {
  test(`CSR endpoint mode parses explicitly (${query})`, () => {
    assert.equal(runInNewContext(`${modeSource}\nendpointsMode`,{URLSearchParams,location:{search:query}}),mode);
  });
}
test('endpoints=pairs without persistent contacts fails before allocating GPU work', () => {
  assert.throws(()=>runInNewContext(modeSource,{URLSearchParams,location:{search:'?scene=columns&contacts=canonical&endpoints=pairs'}}),
    /endpoints=pairs requires persistent contacts/);
});
for (const earlyPairPack of [false, true]) {
  test(`solver pairs are packed once, before CSR when its endpoints read them (early=${earlyPairPack})`, () => {
    const trace = encodeTrace({reuse:true, manifoldHashInsert:true, npSplit:true, earlyPairPack});
    const at = name => trace.flatMap((c,i)=>c.name===name?[i]:[]);
    const pack = at('pack_solver_contacts');
    assert.equal(pack.length,1,'Exactly one pack per step: a second pack is wasted work, none leaves stale pairs.');
    const warm = at('persist_contacts_copy_persistent')[0];
    if (earlyPairPack) assert.ok(warm < pack[0] && pack[0] < at('csr_degree_contacts')[0],
      'CSR degree and scatter read body ids from solver pairs: the pack must follow persistence and precede the graph.');
    else assert.ok(pack[0] > at('predict_bodies')[0], 'Manifold endpoints keep the pack beside the solve.');
  });
}

// Persist work list (8em).
for (const [query, mode] of [['?scene=ring110k','list'],['?scene=gables506k','list'],
  ['?scene=columns','list'],['?scene=columns&contacts=canonical','scan'],['?scene=ring110k&persist=scan','scan'],['?scene=ring110k&reuse=pass','scan']]) {
  test(`persist mode parses explicitly (${query})`, () => {
    assert.equal(runInNewContext(`${modeSource}\npersistMode`,{URLSearchParams,location:{search:query}}),mode);
  });
}
test('persist=list without reuse=copy fails before allocating GPU work', () => {
  assert.throws(()=>runInNewContext(modeSource,{URLSearchParams,location:{search:'?scene=ring110k&reuse=pass&persist=list'}}),
    /persist=list requires reuse=copy/);
});
test('persist list runs from slot 70 with the overflow scan on slot 71, after narrowphase work args', () => {
  const trace = encodeTrace({reuse:true, manifoldHashInsert:true, npSplit:true, earlyPairPack:true, persistList:true});
  const at = name => trace.flatMap((c,i)=>c.name===name?[i]:[]);
  const list = at('persist_list_copy_persistent'), scan = at('persist_contacts_copy_persistent');
  assert.equal(list.length,1); assert.equal(scan.length,1);
  assert.equal(trace[list[0]].slot,70,'The list pass must be sized by the listed count, not every contact.');
  assert.equal(trace[scan[0]].slot,71,'The scan must use slot 71, which is zero unless the list overflowed.');
  const fresh = at('narrowphase_fresh_copy_persistent')[0];
  assert.ok(at('write_work_args').some(i => i > fresh && i < list[0]),
    'Slots 70/71 are written after narrowphase; a stale count would skip or repeat persistence.');
  assert.ok(list[0] < scan[0] && scan[0] < at('pack_solver_contacts')[0]);
});

// Local incremental color repair (8en).
for (const [query, mode] of [['?scene=ring110k','local'],['?scene=gables506k','local'],
  ['?scene=ring110k&repair=rounds','rounds'],['?scene=ring110k&coloring=rebuild','rounds']]) {
  test(`color repair mode parses explicitly (${query})`, () => {
    assert.equal(runInNewContext(`${modeSource}\nrepairMode`,{URLSearchParams,location:{search:query}}),mode);
  });
}
test('repair=local without incremental coloring fails before allocating GPU work', () => {
  assert.throws(()=>runInNewContext(modeSource,{URLSearchParams,location:{search:'?scene=ring110k&coloring=rebuild&repair=local'}}),
    /repair=local requires incremental coloring/);
});
for (const localRepair of [false, true]) {
  test(`local repair runs once between slot-69 work args and the global rounds (local=${localRepair})`, () => {
    const trace = encodeTrace({reuse:true, manifoldHashInsert:true, npSplit:true, earlyPairPack:true, persistList:true, localRepair});
    const at = name => trace.flatMap((c,i)=>c.name===name?[i]:[]);
    const local = at('color_bodies_local'), rounds = at('color_bodies');
    assert.equal(rounds.length, 32, 'The global rounds stay as the fallback for large repair lists.');
    assert.equal(local.length, localRepair ? 1 : 0);
    if (localRepair) {
      const init = at('init_colors')[0];
      assert.ok(init < local[0] && at('write_work_args').some(i => i > init && i < local[0]) && local[0] < rounds[0],
        'Local repair needs the repair list and its slot-69 decision, and must finish before any global round.');
      assert.equal(trace[local[0]].threads, 64, 'One workgroup owns the whole repair list.');
    }
  });
}

// Resting-pair carry across sleep (8ep).
for (const [query, mode] of [['?scene=ring110k','carry'],['?scene=ring','carry'],['?scene=ring110k&rest=drop','drop']]) {
  test(`rest mode parses explicitly (${query})`, () => {
    assert.equal(runInNewContext(`${modeSource}\nrestMode`,{URLSearchParams,location:{search:query}}),mode);
  });
}
test('invalid rest mode fails before allocating GPU work', () => {
  assert.throws(()=>runInNewContext(modeSource,{URLSearchParams,location:{search:'?scene=ring&rest=keep'}}),
    /rest must be carry or drop/);
});
for (const carryRest of [false, true]) {
  test(`resting pairs are carried after the broadphase and before narrowphase reads them (carry=${carryRest})`, () => {
    const trace = encodeTrace({reuse:true, manifoldHashInsert:true, npSplit:true, earlyPairPack:true, persistList:true, localRepair:true, carryRest});
    const at = name => trace.flatMap((c,i)=>c.name===name?[i]:[]);
    const carry = at('carry_rest_pairs');
    assert.equal(carry.length, carryRest ? 1 : 0);
    if (carryRest) {
      assert.equal(trace[carry[0]].slot, 72, 'Carry runs over the previous manifolds published in slot 72.');
      assert.ok(at('broadphase_query')[0] < carry[0] && carry[0] < at('narrowphase_defer_copy_persistent')[0],
        'Carry appends candidates after the broadphase and must read gManifolds before narrowphase overwrites them.');
      assert.ok(at('write_work_args').some(i => i > carry[0] && i < at('narrowphase_defer_copy_persistent')[0]),
        'Slot 64 must be recounted after carried candidates, or narrowphase skips them.');
    }
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
    const context = {...DISPATCH,
      manifoldHashInsert, gridHashWords:() => 0, gridSplit:false, gridClearWords:() => 131072, reuseCopy:false, npSplit:false, earlyPairPack:false, persistList:false, localRepair:false, carryRest:false, warmRecordHash:false, warmHashSlots:n => Math.min(2 ** Math.ceil(Math.log2(Math.max(n * 4, 1024))), 2097152), steps, parity:1, bodyCount:10, colliderCount:10, incrementalColors:true,
      csrBase:32768, MAX_WARM_HASH:2097152, nextPow2:n => 2 ** Math.ceil(Math.log2(n)),
      last:{contact:100}, jointCount:0, cap:{maxBodies:64,maxCandidates:128,maxContacts:256,maxColors:32},
      scratchU32:new Uint32Array(64), packHost() {}, packHostU8:new Uint8Array(208),
      hostScratchView:new Uint32Array(64), hostParamsView:new Uint8Array(208),
      hostCountersView:new Uint32Array(16), hostColorsView:new Uint32Array(32),
      device:{queue:{writeBuffer() {}}}, hostUpload:'upload', hostUploadCpu:null, colorCountsBytes:128,
      bufs:Array.from({length:32},(_,i)=>`b${i}`),
    };
    const s = runInNewContext(`${hostSource}\nconst s = scratchWords(); writeHost({copyBufferToBuffer(...a){ copies.push(a); }}); s`,
      {...context, copies});
    assert.equal(s[14], manifoldHashInsert ? 1 : 0, 'HOST_S(14) selects the per-manifold slot-68 count.');
    assert.equal(s[20], 0, 'HOST_S(20) stays 0 unless grid=split.');
    assert.equal(s[22], 0, 'HOST_S(22) stays 0 unless persist=list.');
    assert.equal(s[27], 0, 'HOST_S(27) stays 0 unless repair=local.');
    assert.equal(s[28], 0, 'HOST_S(28) stays 0 unless rest=carry.');
    assert.equal(runInNewContext(`${hostSource}\nscratchWords()[28]`, {...context, carryRest:true}), 1,
      'rest=carry must reach HOST_S(28), or sleepers drop resting pairs and nothing carries them.');
    assert.equal(runInNewContext(`${hostSource}\nscratchWords()[27]`, {...context, localRepair:true}), 1,
      'repair=local must reach HOST_S(27), or slot 69 keeps the global rounds and the local pass exits.');
    assert.equal(runInNewContext(`${hostSource}\nscratchWords()[22]`, {...context, persistList:true}), 1,
      'persist=list must reach HOST_S(22), or narrowphase never lists contacts and slot 70 stays empty.');
    assert.equal(runInNewContext(`${hostSource}\nscratchWords()[20]`, {...context, gridSplit:true}), 1,
      'grid=split must reach HOST_S(20), or the shader keeps one table while the host clears two.');
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
    const context = {...DISPATCH,
      detailedProfile:false, graphProfile:false, colorProfile:true, bodyCount:110442, colliderCount:110442, jointCount:0,
      last:{contact:1_170_000}, cap:{maxContacts:2097152,maxJoints:64,maxColors:32},
      nextPow2:n => 2 ** Math.ceil(Math.log2(n)), MAX_WARM_HASH:2097152,
      persistentContacts:true, params:{flags:1,iterations:4}, compactContacts:true,
      manifoldPersistence:false, fusedPersistence:false, manifoldHashInsert:true, gridHashWords:() => 0, gridSplit:false, gridClearWords:() => 131072, reuseCopy:false, npSplit:false, earlyPairPack:false, persistList:false, localRepair:false, carryRest:false, warmRecordHash:false, warmHashSlots:n => Math.min(2 ** Math.ceil(Math.log2(Math.max(n * 4, 1024))), 2097152),
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

// Grid table sizing shipped in the host: >=2x colliders, never below the
// shader default, never past the CSR region, and explicit in the uniforms.
test('broadphase grid table scales with colliders and clears every head', () => {
  const sizing = html.slice(html.indexOf('function gridHashWords()'), html.indexOf('function scratchWords()'));
  const nextPow2 = n => 2 ** Math.ceil(Math.log2(n));
  for (const [gridMode, colliderCount, expected] of [['scaled', 110442, 262144], ['scaled', 505923, 1048576],
    ['scaled', 1000, 131072], ['scaled', 4_000_000, 4194304], ['fixed', 505923, 0]]) {
    const words = runInNewContext(`${sizing}\ngridHashWords()`, {gridMode, gridSplit:false, colliderCount, nextPow2, csrBase: 4194304});
    assert.equal(words, expected, `${gridMode} grid for ${colliderCount} colliders`);
  }
  // Split grids: two tables, each a power of two, together inside the pre-CSR region.
  const prevPow2 = n => { let p = 1; while (p * 2 <= n) p <<= 1; return p; };
  for (const [colliderCount, csrBase, expected] of [[110442, 4194304, 262144], [4_000_000, 4194304, 2097152],
    [505923, 1572864, 524288]]) {
    const ctx = {gridMode:'split', gridSplit:true, colliderCount, nextPow2, prevPow2, csrBase};
    const words = runInNewContext(`${sizing}\ngridHashWords()`, ctx);
    assert.equal(words, expected, `split grid for ${colliderCount} colliders below csrBase ${csrBase}`);
    assert.ok(2 * words <= csrBase, 'Both split tables must end before the CSR index region.');
    assert.equal(runInNewContext(`${sizing}\ngridClearWords()`, ctx), 2 * words,
      'The clear must cover the awake and the other table.');
  }
  const clearSource = html.slice(html.indexOf('function gridClearWords()'), html.indexOf('function scratchWords()'));
  for (const gridSplit of [false, true]) {
  const trace = [];
  const pass = () => ({id:0, setPipeline() {}, setBindGroup() {}, dispatchWorkgroupsIndirect() {}, end() {}});
  runInNewContext(`${clearSource}\n${source}\nencodeStep(encoder, false)`, {...DISPATCH,
    detailedProfile:false, graphProfile:false, colorProfile:false, bodyCount:505923, colliderCount:505923, jointCount:0,
    last:{contact:4_000_000}, cap:{maxContacts:6291456,maxJoints:64,maxColors:32},
    nextPow2, MAX_WARM_HASH:2097152, persistentContacts:true, params:{flags:1,iterations:3}, compactContacts:true,
    manifoldPersistence:false, fusedPersistence:false, manifoldHashInsert:true, gridHashWords:() => 1048576, gridSplit, reuseCopy:false, npSplit:false, earlyPairPack:false, persistList:false, localRepair:false, carryRest:false, warmRecordHash:false, warmHashSlots:n => Math.min(2 ** Math.ceil(Math.log2(Math.max(n * 4, 1024))), 2097152),
    physicalAdjacency:true, incrementalColors:true, stablePacking:true, sparseStart:32, parallelColorPrefix:true,
    sleepRounds:8, PASS_ALIGN:256, bufs:[], pipes:{solve_primal_color:{},solve_primal_color_compact:{}},
    writeHost() {}, disp(p,name,threads) { trace.push({name,threads}); }, dispWork() {}, encoder:{beginComputePass:pass},
  });
  assert.equal(trace.find(c=>c.name==='broadphase_clear').threads, gridSplit ? 2097152 : 1048576,
    'The clear must cover every head HOST_S(16) addresses (both tables when split); stale heads link last step\'s colliders into new buckets.');
  }
});

// Record warm hash sizing: slots hold one previous manifold each. The CPU
// contact count is a low seed during the first batch, so the body floor keeps
// the table from exhausting its probe limit; four words per slot stay pre-CSR.
test('record warm hash slots cover manifolds even with a stale contact seed', () => {
  const sizing = html.slice(html.indexOf('function warmHashSlots('), html.indexOf('// 0 keeps the shader'));
  const nextPow2 = n => 2 ** Math.ceil(Math.log2(n));
  const slots = (prev, bodyCount, warmRecordHash=true, csrBase=8388608) =>
    runInNewContext(`${sizing}\nwarmHashSlots(${prev})`, {warmRecordHash, bodyCount, nextPow2, csrBase, MAX_WARM_HASH:2097152});
  assert.equal(slots(441764, 110442), 524288, 'Ring seed (bodies x4) alone would give 262144 slots for ~230k manifolds.');
  assert.ok(slots(1184245, 110442) >= 2 * 246481, 'Steady ring keeps load at or below one half.');
  assert.equal(slots(4062927, 505923, true, 16777216), 2097152);
  assert.ok(slots(4062927, 505923, true, 16777216) * 4 <= 16777216, 'Record words must end before the CSR index region.');
  assert.equal(slots(1184245, 110442, false), 2097152, 'Head mode keeps one word per slot and the 2^21 cap.');
});

// Live playback pipelines counter readback (two frames in flight). Execute the
// shipped helpers: draining must wait for every queued readback, decoded counters
// must use the GPU counter layout, and an incomplete coloring must still stop
// playback even though it is reported a frame late.
test('live readback ring drains queued frames and surfaces late coloring overflow', async () => {
  const block = html.slice(html.indexOf('const LIVE_IN_FLIGHT'), html.indexOf('async function frame('));
  // The full 80-byte counter block (20 words) since 8ev reads colorRounds (word 17).
  const counters = new Uint32Array(20);
  counters[2] = 1174316; counters[4] = 256703; counters[5] = 110441; counters[6] = 11; counters[12] = 246481;
  const makeBuffer = () => ({ getMappedRange: () => counters.buffer.slice(0), unmap() { this.unmapped = true; } });
  const device = { createBuffer: makeBuffer };
  const ctx = runInNewContext(`${block}\n({ liveFree, livePending, drainLive, readLiveCounters, get last() { return last; }, setError(e) { liveError = e; } })`,
    { device, GPUBufferUsage: { MAP_READ: 1, COPY_DST: 8 }, last: null, Promise, Set, Array, Uint32Array, Error });
  assert.equal(ctx.liveFree.length, 2, 'Two frames in flight bound latency to two physics steps.');
  let resolve;
  const pending = new Promise(r => { resolve = r; });
  ctx.livePending.add(pending);
  let drained = false;
  const drain = ctx.drainLive().then(() => { drained = true; });
  await new Promise(r => setTimeout(r, 0));
  assert.equal(drained, false, 'Benchmarks must not start while live frames are still queued.');
  resolve();
  await drain;
  assert.equal(drained, true);
  const buf = makeBuffer();
  ctx.readLiveCounters(buf);
  assert.deepEqual({ ...ctx.last }, { contact: 1174316, candidate: 256703, awake: 110441, color: 11, manifold: 246481, overflow: 0, colorRounds: 0 });
  assert.ok(buf.unmapped, 'Readback buffers must be unmapped before reuse.');
  counters[7] = 8;
  assert.throws(() => ctx.readLiveCounters(makeBuffer()), /Graph coloring is incomplete/);
  ctx.setError(new Error('late coloring overflow'));
  await assert.rejects(ctx.drainLive(), /late coloring overflow/, 'A deferred readback failure must stop the next drain.');
});

for (const [query, expected] of [['?scene=ring110k','copy'],['?scene=gables506k','copy'],['?scene=columns','copy'],
  ['?scene=columns&contacts=canonical','pass'],['?scene=ring110k&reuse=pass','pass'],['?scene=ring110k&contacts=canonical','pass'],['?scene=gables506k&contacts=compact','pass']]) {
  test(`reuse copy mode parses explicitly (${query})`, () => {
    assert.equal(runInNewContext(`${modeSource}\nreuseMode`,{URLSearchParams,location:{search:query}}),expected);
  });
}
test('reuse copy requires persistent per-contact persistence', () => {
  assert.throws(()=>runInNewContext(modeSource,{URLSearchParams,location:{search:'?scene=ring110k&contacts=canonical&reuse=copy'}}),
    /reuse=copy requires persistent contacts/);
});
