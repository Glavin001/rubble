import test from 'node:test';
import assert from 'node:assert/strict';
import {buildCuboidHouse,corners} from './cuboid-house.mjs';
import {buildSuburb} from './geometry.mjs';

// Penetration depth of two pieces (box with rotation about +x): the smallest
// overlap over the 15 separating axes, or 0 when some axis separates them.
function penetration(a,b){
  const axes=p=>{const c=Math.cos(p.angle),s=Math.sin(p.angle);return [[1,0,0],[0,c,s],[0,-s,c]]};
  const A=axes(a),B=axes(b),cand=[...A,...B];
  for(const u of A)for(const v of B){const n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]],l=Math.hypot(...n);if(l>1e-6)cand.push(n.map(x=>x/l))}
  const ca=corners(a),cb=corners(b);let depth=Infinity;
  for(const n of cand){
    const pa=ca.map(c=>c[0]*n[0]+c[1]*n[1]+c[2]*n[2]),pb=cb.map(c=>c[0]*n[0]+c[1]*n[1]+c[2]*n[2]);
    const overlap=Math.min(Math.max(...pa),Math.max(...pb))-Math.max(Math.min(...pa),Math.min(...pb));
    if(overlap<=0)return 0;depth=Math.min(depth,overlap);
  }
  return depth;
}

test('cuboid house: box colliders only, no piece starts inside another, all asleep above ground',()=>{
  const house=buildCuboidHouse(),pieces=house.selected;
  assert.ok(pieces.length>700,`a detailed house (${pieces.length} pieces)`);
  const boxes=house.colliders.slice(1,house.projectileStart);
  assert.ok(boxes.every(c=>c.shape===2&&c.params.slice(0,3).every(h=>h>0)));
  assert.ok(house.bodies.slice(1,house.projectileStart).every(b=>b.flags===5&&b.invMass>0&&b.invI.every(Number.isFinite)));
  // Nothing below the floor.
  for(const p of pieces)assert.ok(Math.min(...corners(p).map(c=>c[1]))>-1e-9,'a piece starts below the ground');
  // Touching is fine (resting contact); interpenetration over 0.1 mm is a layout bug.
  const bounds=pieces.map(p=>{const c=corners(p);return [0,1,2].map(k=>[Math.min(...c.map(v=>v[k])),Math.max(...c.map(v=>v[k]))])});
  let worst=0,pair=null;
  for(let i=0;i<pieces.length;i++)for(let j=i+1;j<pieces.length;j++){
    if(bounds[i].some(([lo,hi],k)=>hi<=bounds[j][k][0]||bounds[j][k][1]<=lo))continue;
    const d=penetration(pieces[i],pieces[j]);if(d>worst){worst=d;pair=[i,j]}
  }
  assert.ok(worst<1e-4,`pieces ${pair} interpenetrate by ${(worst*1000).toFixed(2)} mm`);
});

test('cuboid suburb: 48 houses on lots, the shot crosses one column of lots',()=>{
  const one=buildCuboidHouse(),suburb=buildSuburb(null,0,{base:one});
  assert.equal(suburb.copies,48);
  const frags=one.projectileStart-1;assert.equal(suburb.projectileStart,1+48*frags);
  const x=suburb.rowShot.origin[0],r=suburb.projectileRadius;
  const hit=Array.from({length:48},(_,k)=>suburb.bodies[1+k*frags].p[0]-one.bodies[1].p[0])
    .filter(dx=>x+r>dx-one.extentX/2&&x-r<dx+one.extentX/2).length;
  assert.equal(hit,6);
});
