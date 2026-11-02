// Grid peak profiling for optimization work, run in the demo page (?gputime=phases for the
// per-phase split):
//   const m=await import('./perf-bench.mjs'); await m.gridPeak(window.__FRACTURE_DEMO__)
// Loads the 49-house grid, fires the ball, steps to the busiest phase (~420 steps), then
// reports World's GPU time of a step (its phases when the page asked for them), render
// time, and the counters.
export async function gridPeak(D,{scene='grid',warm=420,samples=5}={}){
  await D.load(scene);for(let i=0;i<600&&!D.ready;i++)await new Promise(r=>setTimeout(r,200));
  const s=D.simulation,R=D.renderer;
  await D.fireRow();await s.step(warm);
  const whole=[];let phases=null;
  for(let r=0;r<samples;r++){await s.step(1);whole.push(s.gpuTickMs);phases=s.gpuTime?.segments??phases}whole.sort((a,b)=>a-b);
  const draw=[];for(let r=0;r<3;r++)draw.push(await R.profileDraw());draw.sort((a,b)=>a-b);
  const agg={};for(const [k,v] of phases??[])agg[k]=(agg[k]??0)+v;
  return {awake:s.counters.awake,contacts:s.counters.contact,stepMs:whole[Math.floor(samples/2)]==null?null:+whole[Math.floor(samples/2)].toFixed(2),drawMs:+draw[1].toFixed(2),
    phases:Object.entries(agg).sort((a,b)=>b[1]-a[1]).slice(0,12).map(([k,v])=>`${k} ${v.toFixed(2)}`)};
}
