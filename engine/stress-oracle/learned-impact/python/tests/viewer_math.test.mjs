import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const source=await readFile(new URL('../learned_impact/viewer/scene.js',import.meta.url),'utf8');
const {rotation,interval,identity}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
test('rigid pose interpolation preserves orthonormal rotation through 180 degrees',()=>{
 const end=[[-1,0,0],[0,-1,0],[0,0,1]];
 for(const t of [0,.25,.5,.75,1]){
  const r=rotation(identity,end,t);
  for(let i=0;i<3;i++)for(let j=0;j<3;j++)near(r[i].reduce((s,x,k)=>s+x*r[j][k],0),i===j?1:0);
  near(r[0][0],Math.cos(Math.PI*t));near(r[1][0],Math.sin(Math.PI*t));
 }
});
test('playhead maps initial pose and final pose to valid adjacent snapshots',()=>{
 assert.deepEqual(interval(0,24),{i:0,alpha:0});
 assert.deepEqual(interval(2.5,24),{i:2,alpha:.5});
 assert.deepEqual(interval(24,24),{i:23,alpha:1});
});
