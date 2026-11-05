import test from 'node:test';
import assert from 'node:assert/strict';
import {restoreSceneOrder} from '../spatial-order.mjs';

// The upload order itself (engine-scenes `web::upload_order`) is specified and tested in
// Rust: engine-scenes tests/web_scenes.rs. Here: the readback's return to scene order.

test('restoring scene order preserves every readback word of every body',()=>{
  const gpuToScene=[0,3,1,4,2];
  const packed=new Uint32Array(gpuToScene.length*28);
  for(let gpu=0;gpu<gpuToScene.length;gpu++)for(let word=0;word<28;word++)
    packed[gpu*28+word]=(Math.imul(gpuToScene[gpu]+1,2654435761)^word)>>>0;
  const restored=new Uint32Array(restoreSceneOrder(packed.buffer,gpuToScene));
  for(let scene=0;scene<gpuToScene.length;scene++)for(let word=0;word<28;word++)
    assert.equal(restored[scene*28+word],(Math.imul(scene+1,2654435761)^word)>>>0,`body ${scene}, word ${word} lost identity`);
});

test('corrupt and truncated readbacks fail with identity diagnostics',()=>{
  assert.throws(()=>restoreSceneOrder(new ArrayBuffer(112),[0,1]),/cannot hold 2 records/);
  for(const map of [[0,0],[0,2],[0,-1],[0,NaN]]) assert.throws(()=>restoreSceneOrder(new ArrayBuffer(224),map),/invalid or duplicate scene identity/);
});
