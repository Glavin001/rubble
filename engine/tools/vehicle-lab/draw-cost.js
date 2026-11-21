// The vehicle lab's drawing cost, part by part: an expression for the recorder's --eval
// (it runs in the page once it is ready, before the probe's seconds).
//
//   LIMIT=560 bash engine/tools/vehicle-lab/run.sh --probe --secs 3 --eval engine/tools/vehicle-lab/draw-cost.js \
//     --out engine/out/draw-cost "c5:mode=mountain&car=gen&cars=5&renderScale=1"
//
// It lets the run reach SETTLE simulated seconds (the cars on their way down), reads the
// drawn frames' GPU ms while the World steps (the page's own report: skinning, shadow
// map, the picture), then pauses the World and, for each list of parts left out
// (window.__drawSkip), measures the drawing alone (window.__drawCost: the frame loop
// held, draws submitted back to back in batches and waited for, so the GPU is busy
// with nothing else and does not clock down between them): the wall time a draw and
// the medians of its three timers. A part's cost is the whole minus the run without it.
// Returns the table as text.
(async () => {
  const SETTLE = 1.5, DRAWS = 48, BATCH = 8;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const f = (x) => (x == null ? '   -  ' : x.toFixed(2).padStart(6));
  for (let i = 0; i < 400 && !(window.__state().t >= SETTLE); i++) await sleep(100);
  const stats = window.__drawStats();
  const out = [];
  out.push(`canvas ${stats.canvas.join('x')}, colliders ${stats.colliders}, bodies ${stats.bodies}, cars ${stats.cars.length}`);
  const c0 = stats.cars[0];
  if (c0) out.push(`car 0: ${c0.nodes} nodes, ${c0.units} damage units, triangle cut ${c0.cut ? 'on' : 'off'}; levels: ` + c0.levels.map((l, k) => `[${k}] ${l.vertices} vertices, ${l.opaqueTriangles} opaque + ${l.glassTriangles} blended triangles, binds ${(l.bindBytes / 1e6).toFixed(1)} MB, edge ${(1000 * l.edge).toFixed(1)} mm`).join('; '));
  window.__drawSkip([]);
  await sleep(2500);
  const live = window.__drawGpu(), st = window.__state();
  out.push(`World stepping (t ${st.t.toFixed(2)} s): ${st.simRate.toFixed(0)} steps/s, ${st.fps.toFixed(0)} FPS; GPU ms a drawn frame: skinning ${f(live.skin)} shadow map ${f(live.shadow)} picture ${f(live.picture)} (${live.n} frames)`);
  document.getElementById('pause').click();
  await sleep(300);
  out.push('World paused, draws alone. Left out | skinning | shadow | picture | sum    | wall ms a draw');
  const all = ['normals', 'comps', 'frames', 'skin', 'cut'];
  const sets = [[], ['skin'], ['cut'], ['normals', 'comps', 'frames'], all, ['depth'], ['boxes', 'spheres'], ['opaque'], ['glass'],
    [...all, 'depth', 'boxes', 'spheres', 'opaque', 'glass'], []];
  for (const s of sets) {
    window.__drawSkip(s);
    await window.__drawCost(BATCH, BATCH);
    window.__drawSkip(s);
    const wall = await window.__drawCost(DRAWS, BATCH);
    await sleep(100);
    const g = window.__drawGpu();
    out.push(`${(s.join(',') || '(nothing)').padEnd(35)} | ${f(g.skin)}   | ${f(g.shadow)} | ${f(g.picture)}  | ${f(g.skin + g.shadow + g.picture)} | ${f(wall)}`);
  }
  // The picture against its pixels: the same draws at half the render scale (a quarter
  // of the pixels). What does not fall with them is the geometry's part.
  const scale = document.getElementById('renderScale'), was = scale.value;
  for (const v of ['0.5', was]) {
    scale.value = v; scale.dispatchEvent(new Event('change', { bubbles: true }));
    await sleep(200);
    window.__drawSkip([]);
    await window.__drawCost(BATCH, BATCH);
    window.__drawSkip([]);
    const wall = await window.__drawCost(DRAWS, BATCH);
    await sleep(100);
    const g = window.__drawGpu(), c = window.__drawStats().canvas;
    out.push(`${('render scale ' + v + ' (' + c.join('x') + ')').padEnd(35)} | ${f(g.skin)}   | ${f(g.shadow)} | ${f(g.picture)}  | ${f(g.skin + g.shadow + g.picture)} | ${f(wall)}`);
  }
  window.__drawSkip([]);
  document.getElementById('pause').click();
  return out.join('\n');
})()
