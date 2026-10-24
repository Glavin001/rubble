// camera-input.mjs on stand-in DOM objects: what each view mode does with clicks, drags,
// mouse-look and held keys.
import test from 'node:test';
import assert from 'node:assert/strict';

const listeners=(target)=>{const map={};target.addEventListener=(t,f)=>{(map[t]??=[]).push(f)};target.removeEventListener=(t,f)=>{map[t]=(map[t]??[]).filter(g=>g!==f)};
 target.fire=(t,e)=>{for(const f of map[t]??[])f(e)};return target};
globalThis.HTMLElement??=class{};
const win=listeners({});
for(const k of ['addEventListener','removeEventListener'])globalThis[k]=win[k];
globalThis.document={pointerLockElement:null,exitPointerLock(){this.pointerLockElement=null}};
const {attachCameraInput}=await import('./camera-input.mjs');

function setup(mode){
 const canvas=listeners({getBoundingClientRect:()=>({left:0,top:0,width:200,height:100}),setPointerCapture(){},
  requestPointerLock(){document.pointerLockElement=canvas}});
 const calls=[],camera={mode,drag:(...a)=>calls.push(['drag',...a]),mouseLook:(...a)=>calls.push(['look',...a]),zoom:(...a)=>calls.push(['zoom',...a]),travel:(...a)=>calls.push(['travel',...a])};
 const shots=[];document.pointerLockElement=null;
 const input=attachCameraInput(canvas,camera,(u,v)=>shots.push([u,v]));
 const pointer=(type,x,y,extra={})=>canvas.fire(type,{button:0,pointerId:1,pointerType:'mouse',clientX:x,clientY:y,...extra});
 const key=(type,code,key=code)=>win.fire(type,{code,key,target:{},preventDefault(){}});
 return {canvas,camera,calls,shots,input,pointer,key};
}

test('orbit: a click shoots through the pointer, a drag orbits, keys do nothing',()=>{
 const {calls,shots,input,pointer,key}=setup(0);
 pointer('pointerdown',50,25);pointer('pointerup',50,25);
 assert.deepEqual(shots,[[.25,.25]]);
 pointer('pointerdown',50,25);pointer('pointermove',70,25);pointer('pointerup',70,25);
 assert.deepEqual(calls,[['drag',20,0]]);assert.equal(shots.length,1);
 key('keydown','KeyW');input.update(1000);input.update(1100);
 assert.equal(calls.length,1);input.dispose();
});

test('first person: the first click locks the mouse, then clicks shoot through the centre',()=>{
 const {canvas,camera,calls,shots,input,pointer}=setup(1);
 pointer('pointerdown',10,10);pointer('pointerup',10,10);
 assert.equal(document.pointerLockElement,canvas);assert.deepEqual(shots,[]);
 canvas.fire('pointermove',{movementX:3,movementY:-2});
 assert.deepEqual(calls,[['look',3,-2]]);
 pointer('pointerdown',10,10);
 assert.deepEqual(shots,[[.5,.5]]);
 // Another first-person mode keeps the lock; orbit releases it.
 camera.mode=2;input.modeChanged();assert.equal(document.pointerLockElement,canvas);
 camera.mode=0;input.modeChanged();assert.equal(document.pointerLockElement,null);
 input.dispose();
});

test('first person without a mouse: a tap shoots through the tap, a drag looks',()=>{
 const {calls,shots,input,pointer}=setup(2);
 pointer('pointerdown',100,50,{pointerType:'touch'});pointer('pointerup',100,50,{pointerType:'touch'});
 assert.deepEqual(shots,[[.5,.5]]);
 pointer('pointerdown',100,50,{pointerType:'touch'});pointer('pointermove',100,80,{pointerType:'touch'});
 assert.deepEqual(calls,[['drag',0,30]]);input.dispose();
});

test('first person: held keys travel each update, opposite keys cancel, Shift sprints',()=>{
 const {calls,input,key}=setup(2);
 input.update(1000);
 key('keydown','KeyW');key('keydown','KeyD');key('keydown','Space');key('keydown','ShiftLeft','Shift');
 input.update(1050);
 assert.deepEqual(calls.at(-1),['travel',1,1,1,true,.05]);
 key('keydown','KeyS');key('keyup','ShiftLeft','Shift');key('keydown','KeyQ');
 input.update(1550);
 // A long frame moves at most 0.1 s.
 assert.deepEqual(calls.at(-1),['travel',0,1,0,false,.1]);
 key('keyup','KeyD');key('keyup','KeyW');key('keyup','KeyS');key('keyup','Space');key('keyup','KeyQ');
 const n=calls.length;input.update(1600);assert.equal(calls.length,n);
 input.dispose();
});

test('first person: keys typed into the panel do not move the camera',()=>{
 const {calls,input}=setup(1);
 const field=Object.assign(new HTMLElement(),{tagName:'SELECT'});
 win.fire('keydown',{code:'KeyW',key:'w',target:field,preventDefault(){throw new Error('kept from the panel')}});
 input.update(1000);input.update(1016);
 assert.equal(calls.length,0);input.dispose();
});
