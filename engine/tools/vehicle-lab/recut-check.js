// A check of the car renderer's "cut the drawn triangle lists only when the piece labels
// change" (showcase-car-render.mjs cutCheck): an expression for the recorder's --eval.
//
//   bash engine/tools/vehicle-lab/run.sh --probe --secs 1 --eval engine/tools/vehicle-lab/recut-check.js \
//     --out engine/out/recut "m5:mode=mountain&car=gen&cars=5&renderScale=1"
//
// Lets the cars crash for UNTIL simulated seconds (pieces come off: the lab's "mass on
// the main body" falls below 100 %), pauses the World, reads the picture's pixels twice
// (they must agree: nothing moves), then has every car's lists cut again from the
// labels as they are (window.__recut) and reads the pixels again. The lists the frames
// kept must be the lists a fresh cut gives: no pixel may differ. Returns the counts.
(async () => {
  const UNTIL = 5;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  for (let i = 0; i < 1200 && !(window.__state().t >= UNTIL); i++) await sleep(100);
  document.getElementById('pause').click();
  await sleep(500);
  const src = document.getElementById('c');
  const grab = async () => {
    // Two frames on: this callback runs after the page's own draw of the same frame.
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    const c = document.createElement('canvas'); c.width = src.width; c.height = src.height;
    const x = c.getContext('2d', { willReadFrequently: true }); x.drawImage(src, 0, 0);
    return x.getImageData(0, 0, c.width, c.height).data;
  };
  const differ = (a, b) => { let n = 0; for (let k = 0; k < a.length; k += 4) if (a[k] !== b[k] || a[k + 1] !== b[k + 1] || a[k + 2] !== b[k + 2]) n++; return n; };
  const lit = (a) => { let n = 0; for (let k = 0; k < a.length; k += 4) if (a[k] + a[k + 1] + a[k + 2] > 0) n++; return n; };
  const a1 = await grab(), a2 = await grab();
  window.__recut();
  await sleep(200);
  const b = await grab();
  const s = window.__stats();
  document.getElementById('pause').click();
  const still = differ(a1, a2), cut = differ(a2, b);
  const text = `recut check at t ${s.t.toFixed(2)} s: ${s.car.beams.broken} beams broken, mass on the main body ${(100 * s.car.attachedMass).toFixed(2)} %, ${s.car.partsDetached} parts detached; `
    + `${src.width}x${src.height}, ${lit(a1)} pixels drawn; two frames of the paused picture differ in ${still} pixels; after cutting every list again ${cut} pixels differ`;
  if (cut > still) console.error('recut check failed: ' + text);
  return text;
})()
