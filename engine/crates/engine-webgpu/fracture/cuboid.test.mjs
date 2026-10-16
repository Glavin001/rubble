// The procedural cuboid house through the wasm build (layout and overlap
// checks: engine/crates/engine-fracture/tests/cuboid.rs).
import test from 'node:test';
import assert from 'node:assert/strict';
import {buildCuboidHouse,layoutScene} from './wasm.mjs';

test('cuboid house: box colliders only, all pieces asleep and dynamic',()=>{
 const house=buildCuboidHouse();
 assert.ok(house.pieces>700,`a detailed house (${house.pieces} pieces)`);
 assert.equal(house.projectileStart,house.pieces+1);
 assert.ok(house.colliders.slice(1,house.projectileStart).every(c=>c.shape===2&&c.params.slice(0,3).every(h=>h>0)));
 assert.ok(house.bodies.slice(1,house.projectileStart).every(b=>b.flags===5&&b.invMass>0));
 assert.equal(house.params.flags,97,'reuse, zero-gap contacts, world-frame inertia');
 assert.equal(buildCuboidHouse({colliders:'hull'}).hullCount,house.pieces);
});

test('cuboid suburb: 48 houses, procedural budget, the shot crosses one column of lots',()=>{
 const one=buildCuboidHouse(),suburb=layoutScene(one,{suburb:true},true),frags=one.projectileStart-1;
 assert.equal(suburb.copies,48);assert.equal(suburb.projectileStart,1+48*frags);
 assert.equal(suburb.capacity.candidatesPerCollider,12);assert.equal(suburb.capacity.contactsPerCollider,24);
 const x=suburb.rowShot.origin[0],r=suburb.projectileRadius,bodies=suburb.bodies;
 const hit=Array.from({length:48},(_,k)=>bodies[1+k*frags].p[0]-one.bodies[1].p[0]).filter(dx=>x+r>dx-one.extentX/2&&x-r<dx+one.extentX/2).length;
 assert.equal(hit,6);
});
