// Pointer, wheel and keyboard input for the lab's camera (engine-fracture view.rs), shared by
// this page's renderer and the stress modes'. `camera` is {mode, drag, mouseLook, zoom,
// travel}: mode 0 orbits (drag to orbit, click to shoot at the pointer); 1 walks and 2 flies
// in first person (click to lock the mouse, then mouse-look and click to shoot through the
// crosshair; Esc releases; without a mouse, drag to look and tap to shoot). Keys move the
// first-person eye: WASD / arrows, Space or E up, C or Q down, Shift sprints.
const KEYS={KeyW:[0,1],ArrowUp:[0,1],KeyS:[0,-1],ArrowDown:[0,-1],KeyD:[1,1],ArrowRight:[1,1],KeyA:[1,-1],ArrowLeft:[1,-1],
 Space:[2,1],KeyE:[2,1],KeyC:[2,-1],KeyQ:[2,-1]};
const typing=e=>e.target instanceof HTMLElement&&(e.target.isContentEditable||/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName));
export function attachCameraInput(canvas,camera,shoot){
 let drag=null,last=0,settle=false;const held=new Set();
 const firstPerson=()=>camera.mode!==0,locked=()=>document.pointerLockElement===canvas;
 const at=e=>{const r=canvas.getBoundingClientRect();return [(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]};
 const down=e=>{if(e.button!==0)return;
  if(locked()){shoot(.5,.5);return}
  drag={start:[e.clientX,e.clientY],last:[e.clientX,e.clientY],moved:false,mouse:e.pointerType==='mouse'};canvas.setPointerCapture(e.pointerId)};
 // The first move after the mouse locks carries the cursor's jump to the lock point
 // (Chromium), not a look: it is dropped.
 const lockChange=()=>{settle=locked()};
 const move=e=>{
  if(locked()){if(settle){settle=false;return}camera.mouseLook(e.movementX,e.movementY);return}
  if(!drag)return;if(Math.hypot(e.clientX-drag.start[0],e.clientY-drag.start[1])>5)drag.moved=true;
  if(drag.moved)camera.drag(e.clientX-drag.last[0],e.clientY-drag.last[1]);drag.last=[e.clientX,e.clientY]};
 const up=e=>{if(!drag)return;const click=!drag.moved,mouse=drag.mouse;drag=null;if(!click)return;
  // First person with a mouse: the first click locks it (the crosshair aims from then on).
  if(firstPerson()&&mouse&&canvas.requestPointerLock){try{const p=canvas.requestPointerLock();p?.catch?.(()=>{})}catch{}return}
  shoot(...at(e))};
 const cancel=()=>drag=null;
 const wheel=e=>{e.preventDefault();camera.zoom(e.deltaY)};
 // Movement keys only steer the first-person modes, and never while typing in the panel.
 const key=pressed=>e=>{
  if(!pressed){held.delete(e.code);if(e.key==='Shift')held.delete('Shift')}
  if(!firstPerson()||typing(e)||e.ctrlKey||e.metaKey||e.altKey)return;
  if(e.key==='Shift'){if(pressed)held.add('Shift');return}
  if(!(e.code in KEYS))return;
  e.preventDefault();if(pressed)held.add(e.code)};
 const keydown=key(true),keyup=key(false),blur=()=>held.clear();
 canvas.addEventListener('pointerdown',down);canvas.addEventListener('pointermove',move);canvas.addEventListener('pointerup',up);canvas.addEventListener('pointercancel',cancel);canvas.addEventListener('wheel',wheel,{passive:false});
 addEventListener('keydown',keydown);addEventListener('keyup',keyup);addEventListener('blur',blur);document.addEventListener?.('pointerlockchange',lockChange);
 return {
  // Travel by the keys held since the last update (once per drawn frame).
  update(now=performance.now()){
   const dt=last?Math.min(.1,(now-last)/1000):0;last=now;
   if(!firstPerson()||!held.size)return;
   const axes=[0,0,0];for(const code of held){const k=KEYS[code];if(k)axes[k[0]]+=k[1]}
   camera.travel(Math.sign(axes[0]),Math.sign(axes[1]),Math.sign(axes[2]),held.has('Shift'),dt)},
  // A view-mode change: orbit releases the mouse; held keys are dropped.
  modeChanged(){held.clear();if(!firstPerson()&&locked())document.exitPointerLock?.()},
  dispose(){canvas.removeEventListener('pointerdown',down);canvas.removeEventListener('pointermove',move);canvas.removeEventListener('pointerup',up);canvas.removeEventListener('pointercancel',cancel);canvas.removeEventListener('wheel',wheel);
   removeEventListener('keydown',keydown);removeEventListener('keyup',keyup);removeEventListener('blur',blur);document.removeEventListener?.('pointerlockchange',lockChange)}};
}
