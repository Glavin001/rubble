// The cars of one model skinned together (showcase-car-render.mjs skinBatchWgsl) against
// one by one: an expression for the recorder's --eval.
//
//   LIMIT=600 bash engine/tools/vehicle-lab/run.sh --probe --secs 1 --limit 10 \
//     --eval engine/tools/vehicle-lab/skin-batch.js --out engine/out/skin-batch \
//     "derby8:mode=derby&car=gen&surface=0&derbyCars=8&timing=step"
//
// At each of SAMPLES (simulated seconds) it pauses the World and, on that one state,
// skins with every setting of "skin cars of one model" (one by one, wide, loop, one by
// one again; with window.__skinRefUrl, also one by one with that module's kernel, e.g.
// the code before a change: skin-batch.sh puts the base commit's there) and reads back every car's skinned vertices (window.__skinOut): the bytes
// must be the same bit for bit. Then it times the drawing alone with each setting,
// ROUNDS rounds interleaved (window.__drawCost; the GPU ms of skinning, shadow map and
// picture from the page's timers). Returns the table as text.
(async () => {
  const SAMPLES = window.__skinBatchSamples ?? [1.5, 5, 9], ROUNDS = 3, DRAWS = 32, BATCH = 8;
  // Parts left out per row (the car renderer's update passes), to see what skinning is.
  const PARTS = window.__skinBatchParts ?? [[]];
  const MODES = ['off', 'wide', 'loop'];
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const f = (x) => (x == null ? '   -  ' : x.toFixed(2).padStart(6));
  const set = (mode) => { const c = document.getElementById('skinBatch'); c.value = mode; c.dispatchEvent(new Event('change', { bubbles: true })); };
  const pause = () => document.getElementById('pause').click();
  const out = [];
  let same = true;
  const ref = window.__skinRefUrl ? (await import(window.__skinRefUrl)).SHOWCASE_SKIN_WGSL.skin : null;
  if (ref) out.push(`reference kernel: ${window.__skinRefUrl}`);
  for (const t of SAMPLES) {
    for (let i = 0; i < 600 && !(window.__state().t >= t); i++) await sleep(100);
    pause();
    await sleep(300);
    const st = window.__state();
    const hashes = {};
    for (const mode of [...(ref ? ['ref'] : []), 'off', 'wide', 'loop', 'off']) {
      if (mode === 'ref') { set('off'); window.__skinCode(ref); } else set(mode);
      const h = await window.__skinOut();
      if (mode === 'ref') window.__skinCode(null);
      const key = h.map((x) => x.sha.slice(0, 12)).join(' ');
      if (hashes[mode] && hashes[mode] !== key) { same = false; out.push(`t ${st.t.toFixed(2)}: one by one twice differs`); }
      hashes[mode] = key;
    }
    const cars = window.__drawStats().cars;
    const ok = hashes.off === hashes.wide && hashes.off === hashes.loop && (!ref || hashes.ref === hashes.off);
    if (!ok) same = false;
    out.push(`t ${st.t.toFixed(2)} s, ${cars.length} cars (${cars.filter((c) => c.batched).length} batched under the last setting), skinned vertices ${ok ? 'the same bit for bit' : 'DIFFER'} (${ref ? 'reference | ' : ''}off | wide | loop):`);
    for (const m of [...(ref ? ['ref'] : []), ...MODES]) out.push(`  ${m.padEnd(5)} ${hashes[m]}`);
    out.push('  setting | skinning | shadow | picture | sum    | wall ms a draw   (World paused, draws alone)');
    for (let r = 0; r < ROUNDS; r++) {
      for (const parts of PARTS) for (const m of [...(ref ? ['ref'] : []), ...MODES]) {
        if (m === 'ref') { set('off'); window.__skinCode(ref); } else set(m);
        window.__drawSkip(parts);
        await window.__drawCost(BATCH, BATCH);
        window.__drawSkip(parts);
        const wall = await window.__drawCost(DRAWS, BATCH);
        await sleep(100);
        const g = window.__drawGpu();
        if (m === 'ref') window.__skinCode(null);
        out.push(`  ${(m + (parts.length ? ' -' + parts.join(',') : '')).padEnd(7)} | ${f(g.skin)}   | ${f(g.shadow)} | ${f(g.picture)}  | ${f(g.skin + g.shadow + g.picture)} | ${f(wall)}`);
      }
    }
    set('wide');
    window.__drawSkip([]);
    pause();
  }
  out.push(same ? 'PASS: every state the same bit for bit' : 'FAIL: skinned vertices differ');
  return out.join('\n');
})()
