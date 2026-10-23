// The browser half of engine-gpu-wait: the wasm import `rubble-gpu-wait.yield_to_browser`
// (resolved by the page's import map). Suspends the calling wasm stack for one macrotask so a
// GPU map's callback can run before the solver reads the mapped buffer. Two suspensions, by
// the browser. JavaScript Promise Integration (Chrome 137+): a WebAssembly.Suspending, under an
// export called through WebAssembly.promising. Without it (Safari): the Asyncify build of
// stress-web (build-web.sh), whose yield unwinds the stack to the page; asyncifyPromising
// rewinds it once the macrotask has run. `waitStats` counts the suspensions and the time spent
// in them (the stress mode's HUD).
const channel=new MessageChannel();let waiting=[];
export const waitStats={yields:0,ms:0};
channel.port1.onmessage=()=>{const ready=waiting;waiting=[];for(const resolve of ready)resolve()};
// One macrotask: the promise resolves once the channel's message has been handled.
const wake=()=>{const t=performance.now();waitStats.yields++;return new Promise(resolve=>{waiting.push(()=>{waitStats.ms+=performance.now()-t;resolve()});if(waiting.length===1)channel.port2.postMessage(0)})};
// Asyncify: the build's exports and its save area (bindAsyncify), the wake-up of the unwound
// stack (pending), and the queue that keeps one saved stack in use at a time.
let asyncify=null,save=0,pending=null,queue=Promise.resolve();
export function bindAsyncify(exports,area){asyncify=exports;save=area}
const unwind=()=>{
 if(!asyncify)throw new Error('the Asyncify build of stress-web is not bound');
 // Rewinding (state 2): the stack is restored up to this call; the wait is over, so
 // normal execution resumes from here.
 if(asyncify.asyncify_get_state()===2){asyncify.asyncify_stop_rewind();return}
 pending=wake();asyncify.asyncify_start_unwind(save);
};
// Which build the page runs: JSPI where the browser has it, else Asyncify. `?asyncify=1`
// forces Asyncify (to test it in Chrome). Both builds import this one name: a Suspending
// for JSPI, the unwind for Asyncify.
export const JSPI='Suspending' in WebAssembly&&new URLSearchParams(globalThis.location?.search??'').get('asyncify')!=='1';
export const yield_to_browser=JSPI?new WebAssembly.Suspending(wake):unwind;
// An export that may reach yield_to_browser, called as WebAssembly.promising would call it:
// the stack unwinds to here, the event loop turns, and the export is called again to rewind
// until it returns. Calls run one after another (one saved stack).
export function asyncifyPromising(exp){
 const run=async(args)=>{
  let r=exp(...args);
  while(asyncify.asyncify_get_state()===1){
   asyncify.asyncify_stop_unwind();
   const wait=pending;pending=null;await wait;
   asyncify.asyncify_start_rewind(save);
   r=exp(...args);
  }
  return r;
 };
 return (...args)=>{const next=queue.then(()=>run(args));queue=next.catch(()=>{});return next};
}
globalThis.__gpuWaitStats=waitStats;
