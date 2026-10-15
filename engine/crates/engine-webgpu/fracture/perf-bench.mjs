// Grid peak profiling for optimization work, run in the demo page:
//   const m=await import('./perf-bench.mjs'); await m.gridPeak(window.__FRACTURE_DEMO__)
// Loads the 49-house grid, fires the ball, steps to the busiest phase (~420
// steps), then reports per-kernel GPU time (two steps averaged), the whole
// step in one timed pass, render time, and the counters.
export async function gridPeak(D,{scene='grid',warm=420,samples=2}={}){
  await D.load(scene);for(let i=0;i<600&&!D.ready;i++)await new Promise(r=>setTimeout(r,200));
  const s=D.simulation,R=D.renderer;
  await D.fireRow();for(let k=0;k<warm;k+=4)await s.step(4);
  const agg={};for(let r=0;r<samples;r++){await s.step(4);const p=await s.profileKernels();
    for(const [k,v] of Object.entries(p.kernels)){(agg[k]??={ms:0,count:0}).ms+=v.ms/samples;agg[k].count=v.count}}
  const whole=[];for(let r=0;r<5;r++){await s.step(4);whole.push((await s.profileStep({singlePass:true})).gpuMs)}whole.sort((a,b)=>a-b);
  const draw=[];for(let r=0;r<3;r++)draw.push(await R.profileDraw());draw.sort((a,b)=>a-b);
  return {awake:s.counters.awake,contacts:s.counters.contact,stepMs:+whole[2].toFixed(2),drawMs:+draw[1].toFixed(2),
    kernels:Object.entries(agg).sort((a,b)=>b[1].ms-a[1].ms).slice(0,12).map(([k,v])=>`${k} ${v.ms.toFixed(2)}`)};
}
