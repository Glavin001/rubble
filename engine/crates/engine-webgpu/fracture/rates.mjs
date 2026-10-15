// Wall-clock rates: count completed physics ticks and submitted render frames.
// Do not derive FPS from CPU draw time: WebGPU rendering is asynchronous.
// `stepHz` is the physics rate (1/dt); speed 1 means real time.
export function createRates(publish, now = () => performance.now(), stepHz = () => 120) {
 let start, frames, steps, stepMs, encodeMs, waitMs;
 function reset() {
  start=now();frames=steps=stepMs=encodeMs=waitMs=0;
  publish(null);
 }
 reset();
 return {reset,
  step(timing) {steps++;stepMs+=timing.wallMs;encodeMs+=timing.encodeMs;waitMs+=timing.waitMs},
  frame() {
   frames++;
   const end=now(),elapsed=end-start;
   if(elapsed<1000)return;
   const tps=steps*1000/elapsed,fps=frames*1000/elapsed;
   publish({tps,fps,speed:tps/stepHz(),stepMs:steps?stepMs/steps:0,
    encodeMs:steps?encodeMs/steps:0,waitMs:steps?waitMs/steps:0,frameMs:elapsed/frames});
   start=end;frames=steps=stepMs=encodeMs=waitMs=0;
  }
 };
}
