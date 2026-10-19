// The browser half of engine-gpu-wait: the wasm import `rubble-gpu-wait.yield_to_browser`
// (resolved by the page's import map). Suspends the calling wasm stack for one macrotask
// (JavaScript Promise Integration), so a GPU map's callback can run before the solver
// reads the mapped buffer. Only valid under an export called through
// WebAssembly.promising (stress-mode.mjs). `waitStats` counts the suspensions and the time
// spent in them (the stress mode's HUD).
const channel=new MessageChannel();let waiting=[];
export const waitStats={yields:0,ms:0};
channel.port1.onmessage=()=>{const ready=waiting;waiting=[];for(const resolve of ready)resolve()};
export const yield_to_browser=new WebAssembly.Suspending(()=>{const t=performance.now();waitStats.yields++;return new Promise(resolve=>{waiting.push(()=>{waitStats.ms+=performance.now()-t;resolve()});if(waiting.length===1)channel.port2.postMessage(0)})});
globalThis.__gpuWaitStats=waitStats;
