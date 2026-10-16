export const add=(a,b)=>a.map((x,i)=>x+b[i]);
export const sub=(a,b)=>a.map((x,i)=>x-b[i]);
export const scale=(a,s)=>a.map(x=>x*s);
export const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
export const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export const norm=a=>scale(a,1/(Math.hypot(...a)||1));
export const rotate=(q,v)=>add(v,scale(cross(q.slice(0,3),add(cross(q.slice(0,3),v),scale(v,q[3]))),2));
export const inverseRotate=(q,v)=>rotate([-q[0],-q[1],-q[2],q[3]],v);
export const quatMul=(a,b)=>[a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],
  a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]];
export const quatConjugate=q=>[-q[0],-q[1],-q[2],q[3]];

// Break capacities of a baked bond, as Shardwright's reference stress solver
// computes them (tools/sim): ft is the bond's interface-material tensile
// strength, else the weaker fragment material's; fc the weaker compressive
// strength; k the baked strength scale. Bearings (ft = 0) resist with 0.05 fc.
// Returns [tension N, compression N, shear N, bending N m].
export function bondCapacities(asset, bond, strengthScale=1) {
  const lib=asset.library,fragments=asset.physics.fragments;
  const strengths=f=>{const m=lib.material[fragments[f].materialIds[0]??0]??{};const ft=m.tensile_strength??1e6;return [ft,m.compressive_strength??10*ft]};
  const [fta,fca]=strengths(bond.a),[ftb,fcb]=bond.b>=0?strengths(bond.b):[fta,fca];
  const iface=bond.composition[0]?.interfaceMaterial??-1;
  const ifm=iface>=0?lib.interface_material[iface-lib.material.length]:null;
  const ft=ifm?.tensile_strength??Math.min(fta,ftb),fc=Math.min(fca,fcb);
  const k=bond.strengthScale*strengthScale,c=Math.max(bond.extentHalf[0],bond.extentHalf[1],1e-6);
  const fte=ft>0?ft:0.05*fc;
  return [Math.max(fte*bond.area*k,1e-6),Math.max(fc*bond.area*k,1e-6),Math.max(1.5*fte*bond.area*k,1e-6),Math.max(fte*Math.min(bond.iUU,bond.iVV)/c*k,1e-9)];
}

export const JOINT_BOND=7, JOINT_BYTES=192;

// AVBD penalty for bonds: one scene-wide factor maps the physical springs to
// solver stiffness, so rows keep the bake's relative stiffness (and so their
// share of load in redundant structures); the multipliers carry the load. The
// median bond's normal row sits at BOND_PENALTY_SCALE x the median lighter
// body's m/h^2. At steady state a bond acts like a spring of iterations x k:
// 0.5 holds statics to ~0.03 mm / 0.1 deg under 125 kg (bond-regression.mjs).
// Rows stiffer than BOND_PENALTY_CAP x their lighter body's inertial term
// (m/h^2, smallest I/h^2) are capped: past ~20x a bonded cluster locks in
// Gauss-Seidel (a released pair hangs in the air) and f32 6x6 solves lose
// precision; the few capped rows are still the stiffest in the structure.
export const BOND_PENALTY_SCALE=0.5, BOND_PENALTY_CAP=5;
export function scaleBondSprings(joints, bodies, dt, scale=BOND_PENALTY_SCALE, cap=BOND_PENALTY_CAP) {
  if(!joints.length)return joints;
  const median=v=>{const s=[...v].sort((a,b)=>a-b);return s[s.length>>1]};
  const inertial=i=>bodies[i]?.invMass>0?{m:1/bodies[i].invMass,I:1/Math.max(...bodies[i].invI)}:{m:Infinity,I:Infinity};
  const lighter=j=>{const a=inertial(j.bodyA),b=inertial(j.bodyB);return {m:Math.min(a.m,b.m),I:Math.min(a.I,b.I)}};
  const s=scale*median(joints.map(j=>lighter(j).m))/(dt*dt)/median(joints.map(j=>j.k[0]));
  return joints.map(j=>{const {m,I}=lighter(j),capLin=cap*m/(dt*dt),capAng=cap*I/(dt*dt);
    return {...j,k:j.k.map((x,r)=>Math.min(x*s,r<3?capLin:capAng))}});
}

// Elastic bond springs, Shardwright's reference model (frac-validate
// network.rs, StiffnessModel::Spec): series compliances along the normal from
// each fragment's centroid (distA, distB) plus the interface layer,
//   ce = dA/E_A + dB/E_B + t/E_i,  cg = dA/G_A + dB/G_B + t/G_i,
//   kn = A/ce, ks = A/cg, ktu = Iuu/ce, ktv = Ivv/ce, kt = J/cg.
// World bonds (b = -1) have no B side. Returns [kn, ks, ks, ktu, ktv, kt] in
// the bond frame (normal, u, v | bending about u, v, torsion about normal).
export function bondSprings(asset, bond) {
  const lib=asset.library,fragments=asset.physics.fragments;
  const elastic=f=>{const m=lib.material[fragments[f].materialIds[0]??0]??{};
    // Anisotropic (wood): longitudinal E and shear G, as frac-material elastic().
    if(m.anisotropy)return [m.anisotropy.E_long??m.anisotropy.e_long,m.anisotropy.G??m.anisotropy.g];
    if(m.youngs_modulus!=null)return [m.youngs_modulus,m.youngs_modulus/(2*(1+(m.poisson??0.2)))];
    return [1e9,1e9/2.4]};
  const [ea,ga]=elastic(bond.a),[eb,gb]=bond.b>=0?elastic(bond.b):[Infinity,Infinity];
  const iface=bond.composition[0]?.interfaceMaterial??-1,im=iface>=0?lib.interface_material[iface-lib.material.length]:null;
  const em=im?(im.youngs_modulus??ea):1,gm=im?(im.shear_modulus??em/2.4):1,tm=im?(im.thickness??0):0;
  const da=Math.max(bond.distA,1e-9),db=bond.b>=0?Math.max(bond.distB,1e-9):0;
  const ce=da/ea+(Number.isFinite(eb)?db/eb:0)+(tm>0?tm/em:0),cg=da/ga+(Number.isFinite(gb)?db/gb:0)+(tm>0?tm/gm:0);
  const ks=bond.area/cg;
  return [bond.area/ce,ks,ks,bond.iUU/ce,bond.iVV/ce,bond.polarMoment/cg];
}

// Symmetric inertia eigenbasis: keep the off-diagonals, then rotate geometry
// into principal axes because the AVBD body ABI stores diagonal inverse inertia.
export function principalAxes(inertia) {
  const a=Array.from({length:3},(_,i)=>inertia.slice(i*3,i*3+3));
  const v=[[1,0,0],[0,1,0],[0,0,1]];
  for(let iteration=0;iteration<32;iteration++) {
    let p=0,q=1;
    for(const [i,j] of [[0,2],[1,2]])if(Math.abs(a[i][j])>Math.abs(a[p][q])){p=i;q=j;}
    if(Math.abs(a[p][q])<1e-10*Math.max(1,...a.map((r,i)=>Math.abs(r[i]))))break;
    const theta=.5*Math.atan2(2*a[p][q],a[q][q]-a[p][p]),c=Math.cos(theta),s=Math.sin(theta);
    const ap=a[p][p],aq=a[q][q],apq=a[p][q];
    for(let k=0;k<3;k++)if(k!==p&&k!==q){const kp=a[k][p],kq=a[k][q];a[k][p]=a[p][k]=c*kp-s*kq;a[k][q]=a[q][k]=s*kp+c*kq;}
    a[p][p]=c*c*ap-2*s*c*apq+s*s*aq;a[q][q]=s*s*ap+2*s*c*apq+c*c*aq;a[p][q]=a[q][p]=0;
    for(let k=0;k<3;k++){const kp=v[k][p],kq=v[k][q];v[k][p]=c*kp-s*kq;v[k][q]=s*kp+c*kq;}
  }
  const m=v.flat(),trace=m[0]+m[4]+m[8];let q;
  if(trace>0){const s=Math.sqrt(trace+1)*2;q=[(m[7]-m[5])/s,(m[2]-m[6])/s,(m[3]-m[1])/s,s/4];}
  else {let i=0;if(m[4]>m[0])i=1;if(m[8]>m[i*3+i])i=2;const j=(i+1)%3,k=(i+2)%3,s=Math.sqrt(1+m[i*3+i]-m[j*3+j]-m[k*3+k])*2;q=[0,0,0,(m[k*3+j]-m[j*3+k])/s];q[i]=s/4;q[j]=(m[j*3+i]+m[i*3+j])/s;q[k]=(m[k*3+i]+m[i*3+k])/s;}
  const moments=a.map((r,i)=>r[i]);
  if(moments.some(x=>!(x>0)||!Number.isFinite(x)))throw new Error('Fragment inertia is not positive definite');
  return {q:norm(q),moments};
}

// ---- packed hull ABI v2 (engine-schema/src/convex.rs; slang/physics/convex.slang) ----
// One uint32 arena of 16-byte records. Floats are raw bits; indices never pass
// through f32. Record H: V|F<<8|E<<16|flags<<24, loop/edge/adjacency word
// offsets; H+1 local AABB centre + bound radius; H+2 half extents + inner
// radius; H+3 interior point; vertices (w = adjacency start | degree<<16);
// planes (n, d); then words: face entries, u8 loops, edges, u8 adjacency.
export const HULL_FLAG_CLOSED=1, HULL_PARAM_TAG=0x40000000, HULL_BASE_MASK=0x1fffffff;
export const HULL_MAX_VERTICES=64, HULL_MAX_FACES=128, HULL_MAX_EDGES=192;
const bitsF32=new Float32Array(1),bitsU32=new Uint32Array(bitsF32.buffer);
const f32bits=x=>{bitsF32[0]=x;return bitsU32[0]};
const bitsf32=x=>{bitsU32[0]=x>>>0;return bitsF32[0]};

// Normalise a baked face soup into a convex hull. Exact duplicate positions
// weld for adjacency only; vertex records and order are kept. Coplanar patches
// merge into planar convex loops; when that fails ('Degenerate convex face')
// the raw faces are kept and adjacency is left uncertified.
// sourceScale: largest |coordinate| of the frame the vertices were stored in as
// f32 (e.g. the asset frame before moving a fragment into its body frame); the
// convexity check allows that frame's rounding. Defaults to these vertices.
export function cookHull(vertices, topology, {sourceScale=0}={}) {
  try{return cookFaces(vertices,topology,true,sourceScale)}catch(error){
    if(error.message!=='Degenerate convex face')throw error;
    return cookFaces(vertices,topology,false,sourceScale);
  }
}
function cookFaces(vertices, topology, normalizeFaces, sourceScale=0) {
  if(vertices.length<4||vertices.length>HULL_MAX_VERTICES)throw new Error('Convex hull needs 4–64 finite vertices');
  if(vertices.some(v=>v.length!==3||!v.every(Number.isFinite)))throw new Error('Convex hull has non-finite coordinates');
  const center=scale(vertices.reduce((a,v)=>add(a,v),[0,0,0]),1/vertices.length);
  const lo=[0,1,2].map(k=>Math.min(...vertices.map(v=>v[k]))),hi=[0,1,2].map(k=>Math.max(...vertices.map(v=>v[k])));
  const first=new Map(),ids=vertices.map((v,i)=>{const key=Array.from(new Uint32Array(new Float32Array(v).buffer)).join(',');if(!first.has(key))first.set(key,i);return first.get(key)});
  const canonical=[...first.values()].sort((a,b)=>a-b);
  const span=Math.hypot(...sub(hi,lo)),relative=vertices.map(v=>sub(v,center));
  const axis=relative.reduce((a,b)=>dot(a,a)>=dot(b,b)?a:b);
  const plane=relative.map(v=>cross(axis,v)).reduce((a,b)=>dot(a,a)>=dot(b,b)?a:b);
  const planeLength=Math.hypot(...plane);
  const thickness=planeLength>0?Math.max(...relative.map(v=>Math.abs(dot(v,plane))/planeLength)):0;
  if(!planeLength||thickness<=Math.max(span*1e-8,1e-10))throw new Error('Degenerate zero-volume hull: recook the source collision asset');
  const tol=Math.max(span*1e-6,1e-9),planes=[],raws=[];let offset=0;
  // Convexity is judged against the input's f32 precision (~|x|·2^-24 per
  // coordinate, amplified by small faces); merging and packed bytes are unchanged.
  const convexTol=Math.max(tol,Math.max(sourceScale,...vertices.map(v=>Math.max(...v.map(Math.abs))))*2**-20);
  for(const count of topology.sizes){
    if(count<3||count>64)throw new Error('Convex face size outside 3..64');
    const face=topology.indices.slice(offset,offset+count);offset+=count;
    if(face.length!==count||face.some(i=>!Number.isInteger(i)||!vertices[i]))throw new Error('Invalid convex face indices');
    const f=face.map(i=>ids[i]);
    let sum=[0,0,0];for(let k=1;k<count-1;k++)sum=add(sum,cross(sub(vertices[f[k]],vertices[f[0]]),sub(vertices[f[k+1]],vertices[f[0]])));
    if(Math.hypot(...sum)<1e-14)continue;
    let n=norm(sum);if(dot(n,sub(vertices[f[0]],center))<0)n=scale(n,-1);
    const clean=[...new Set(f)];if(dot(sum,n)<0)clean.reverse();
    const d=dot(n,vertices[f[0]]);
    if(!planes.some(p=>dot(p.n,n)>1-1e-7&&Math.abs(p.d-d)<tol))planes.push({n,d});
    raws.push({ids:clean,n});
  }
  if(offset!==topology.indices.length)throw new Error('Invalid convex face indices');
  const faces=[];
  for(const entry of (normalizeFaces?planes:raws.map(r=>({n:r.n,d:dot(r.n,vertices[r.ids[0]]),ids:r.ids})))){
    const {n,d}=entry;let loop=entry.ids;
    if(!loop){
      const u=norm(cross(n,Math.abs(n[0])<.8?[1,0,0]:[0,1,0])),v=cross(n,u);
      const points=canonical.filter(i=>Math.abs(dot(n,vertices[i])-d)<tol).map(i=>({i,x:dot(vertices[i],u),y:dot(vertices[i],v)})).sort((a,b)=>a.x-b.x||a.y-b.y);
      const turn=(a,b,c)=>(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x),lower=[],upper=[];
      for(const p of points){while(lower.length>1&&turn(lower.at(-2),lower.at(-1),p)<=1e-14)lower.pop();lower.push(p)}
      for(const p of points.toReversed()){while(upper.length>1&&turn(upper.at(-2),upper.at(-1),p)<=1e-14)upper.pop();upper.push(p)}
      loop=lower.slice(0,-1).concat(upper.slice(0,-1)).map(p=>p.i);
    }
    if(loop.length<3)throw new Error('Degenerate convex face');
    if(normalizeFaces&&vertices.some(p=>dot(n,p)-d>convexTol*2))throw new Error('Non-convex collision hull');
    faces.push({n,ids:loop});
  }
  if(faces.length<4)throw new Error('Degenerate zero-volume hull: recook the source collision asset');
  let closed=normalizeFaces;const edgeMap=new Map(),edges=[];
  faces.forEach(({ids},face)=>{for(let k=0;k<ids.length;k++){
    const a=ids[k],b=ids[(k+1)%ids.length],key=`${Math.min(a,b)},${Math.max(a,b)}`,e=edgeMap.get(key);
    if(e===undefined){edgeMap.set(key,edges.length);edges.push([a,b,face,-1])}
    else{const edge=edges[e];if(edge[3]>=0||edge[0]!==b||edge[1]!==a)closed=false;if(edge[3]<0)edge[3]=face}
  }});
  if(faces.length>HULL_MAX_FACES||edges.length>HULL_MAX_EDGES)throw new Error('Invalid convex topology');
  if(edges.some(e=>e[3]<0))closed=false;
  return {vertices:vertices.map(v=>[...v]),faces:faces.map(f=>({n:f.n,d:dot(f.n,vertices[f.ids[0]]),ids:f.ids})),edges:edges.map(e=>[e[0],e[1],e[2],e[3]<0?255:e[3]]),closed};
}

// Shift a hull so its local AABB is centred on the origin; returns the offset
// for the collider's local position. Compound children cooked in a body frame
// must be recentred or their bound radius spans the whole body.
export function recenterHull(hull){
  const lo=[0,1,2].map(k=>Math.min(...hull.vertices.map(v=>v[k]))),hi=[0,1,2].map(k=>Math.max(...hull.vertices.map(v=>v[k])));
  const c=scale(add(lo,hi),.5);
  return {hull:{...hull,vertices:hull.vertices.map(v=>sub(v,c)),faces:hull.faces.map(f=>({...f,d:f.d-dot(f.n,c)}))},offset:c};
}

export function cuboidHull(half=[1,1,1]){
  const vertices=[];for(let i=0;i<8;i++)vertices.push([i&1?half[0]:-half[0],i&2?half[1]:-half[1],i&4?half[2]:-half[2]]);
  // Same faces the Rust cooker derives (from_points); order differs but every
  // consumer addresses faces through the packed record.
  const faces=[[0,2,6,4],[1,5,7,3],[0,4,5,1],[2,3,7,6],[0,1,3,2],[4,6,7,5]];
  return cookHull(vertices,{sizes:faces.map(f=>f.length),indices:faces.flat()});
}

export function hullBoundingRadius(hull){return Math.max(...hull.vertices.map(v=>Math.hypot(...v)))}

// Pack one hull (v2). Returns uint32 words, a multiple of four.
export function packHull(hull){
  const v=hull.vertices.length,f=hull.faces.length,e=hull.edges.length;
  const lo=[0,1,2].map(k=>Math.min(...hull.vertices.map(p=>Math.fround(p[k])))),hi=[0,1,2].map(k=>Math.max(...hull.vertices.map(p=>Math.fround(p[k]))));
  const interior=scale(hull.vertices.reduce((a,p)=>add(a,p),[0,0,0]),1/v);
  const inner=Math.max(0,Math.min(...hull.faces.map(x=>x.d-dot(x.n,interior))));
  const loopBytes=hull.faces.reduce((s,x)=>s+x.ids.length,0),adjBytes=hull.closed?2*e:0;
  const faceWord=(4+v+f)*4,loopWord=faceWord+f,edgeWord=loopWord+Math.ceil(loopBytes/4),adjWord=edgeWord+e;
  const words=new Uint32Array(Math.ceil((adjWord+Math.ceil(adjBytes/4))/4)*4);
  const put4=(at,p,w)=>{words[at]=f32bits(p[0]);words[at+1]=f32bits(p[1]);words[at+2]=f32bits(p[2]);words[at+3]=w>>>0};
  words[0]=(v|f<<8|e<<16|(hull.closed?HULL_FLAG_CLOSED:0)<<24)>>>0;words[1]=loopWord;words[2]=edgeWord;words[3]=adjWord;
  put4(4,scale(add(lo,hi),.5),f32bits(hullBoundingRadius(hull)));
  put4(8,scale(sub(hi,lo),.5),f32bits(inner));
  put4(12,interior,0);
  const incident=hull.vertices.map(()=>[]);
  if(hull.closed)hull.edges.forEach(([a,b],k)=>{incident[a].push(k);incident[b].push(k)});
  let adj=0;hull.vertices.forEach((p,i)=>{put4((4+i)*4,p,adj|incident[i].length<<16);adj+=incident[i].length});
  hull.faces.forEach((x,k)=>put4((4+v+k)*4,x.n,f32bits(x.d)));
  const putByte=(word,b,value)=>{words[word+(b>>2)]|=value<<((b&3)*8)};
  let start=0;hull.faces.forEach((x,k)=>{words[faceWord+k]=start|x.ids.length<<16;for(const i of x.ids)putByte(loopWord,start++,i)});
  hull.edges.forEach(([a,b,l,r],k)=>{words[edgeWord+k]=(a|b<<8|l<<16|r<<24)>>>0});
  let at=0;for(const list of incident)for(const k of list)putByte(adjWord,at++,k);
  return words;
}

// Hull arena with runtime management (mirrors engine-schema HullArena):
// identical hulls are shared and reference-counted, released blocks go to a
// first-fit free list (adjacent blocks merge, a free tail shrinks the arena),
// and writes are recorded as dirty record ranges for incremental uploads.
// The unit cube always sits at record 0 (box/hull pairs) and is permanent.
export class HullArena {
  // maxVertices: engine-wide hull limit (8..64, matching engine-schema's
  // HullArena::with_max_vertices). Larger hulls are rejected, never reduced.
  constructor({maxVertices=HULL_MAX_VERTICES}={}){
    if(!(maxVertices>=8&&maxVertices<=HULL_MAX_VERTICES))throw new Error(`Hull vertex limit ${maxVertices} is outside 8..${HULL_MAX_VERTICES}`);
    this.maxVertices=maxVertices;this.words=[];this.dedup=new Map();this.blocks=new Map();this.free=[];this.dirty=[];if(this.insert(cuboidHull())!==0)throw new Error('unit cube must be record 0')}
  insert(hull){
    if(hull.vertices.length>this.maxVertices)throw new Error(`Convex hull has ${hull.vertices.length} vertices; the configured limit is ${this.maxVertices} (hard maximum ${HULL_MAX_VERTICES}). Re-export the asset with fewer hull vertices or raise the limit`);
    const words=packHull(hull),key=words.join(',');
    if(this.dedup.has(key)){const base=this.dedup.get(key);this.blocks.get(base).refs++;return base}
    const len=words.length/4,base=this.#allocate(len);
    for(let i=0;i<words.length;i++)this.words[base*4+i]=words[i];
    this.dirty.push([base,base+len]);this.dedup.set(key,base);this.blocks.set(base,{len,refs:1,key});return base;
  }
  release(base){
    if(base===0)return false;const block=this.blocks.get(base);if(!block)return false;
    if(--block.refs>0)return false;
    this.blocks.delete(base);this.dedup.delete(block.key);this.#freeRange(base,block.len);return true;
  }
  refs(base){return this.blocks.get(base)?.refs??0}
  takeDirty(){
    const d=this.dirty.sort((a,b)=>a[0]-b[0]),out=[];this.dirty=[];
    for(const r of d){const last=out.at(-1);if(last&&r[0]<=last[1])last[1]=Math.max(last[1],r[1]);else out.push([...r])}
    return out;
  }
  markAllDirty(){this.dirty=[[0,this.records]]}
  get freeRecords(){return this.free.reduce((s,f)=>s+f[1],0)}
  #allocate(len){
    const k=this.free.findIndex(f=>f[1]>=len);
    if(k>=0){const [start,n]=this.free[k];if(n===len)this.free.splice(k,1);else this.free[k]=[start+len,n-len];return start}
    const base=this.words.length/4;if(base+len>HULL_BASE_MASK)throw new Error('Hull arena exceeds the 29-bit base range');
    for(let i=0;i<len*4;i++)this.words.push(0);return base;
  }
  #freeRange(start,len){
    let k=this.free.findIndex(f=>f[0]>start);if(k<0)k=this.free.length;this.free.splice(k,0,[start,len]);
    if(k+1<this.free.length&&this.free[k][0]+this.free[k][1]===this.free[k+1][0]){this.free[k][1]+=this.free[k+1][1];this.free.splice(k+1,1)}
    if(k>0&&this.free[k-1][0]+this.free[k-1][1]===this.free[k][0]){this.free[k-1][1]+=this.free[k][1];this.free.splice(k,1)}
    const last=this.free.at(-1);if(last&&last[0]+last[1]===this.records){this.free.pop();this.words.length=last[0]*4}
  }
  get data(){return Uint32Array.from(this.words)}
  get records(){return this.words.length/4}
}

// Per-collider collision filter (mirrors engine-schema CollisionFilter):
// categories/mask (16 bits each) in Collider.generation, 0 = all; a signed
// group in Collider.material bits 16..31 (shared positive: always collide,
// shared negative: never collide, overriding categories).
export function filterWords({categories=0xffff,mask=0xffff,group=0}={},material=0){
  const generation=(categories===0xffff&&mask===0xffff)?0:((categories&0xffff)|((mask&0xffff)<<16))>>>0;
  return {generation,material:((material&0xffff)|((group&0xffff)<<16))>>>0};
}

// Collider params for a hull at `base`: tagged base bits (a normal float),
// uniform scale, 0, bound radius (grid level classification only).
export const hullParams=(base,scaleFactor,radius)=>[bitsf32((base|HULL_PARAM_TAG)>>>0),scaleFactor,0,radius*scaleFactor];
export const hullBase=params=>f32bits(params[0])&HULL_BASE_MASK;

// Cook, recentre and insert one compound child; returns collider fields.
export function appendHull(arena, vertices, topology, cookOptions) {
  const {hull,offset}=recenterHull(cookHull(vertices,topology,cookOptions)),base=arena.insert(hull);
  return {params:hullParams(base,1,hullBoundingRadius(hull)),localPos:offset,hull};
}

// Decoder mirroring the Slang accessors (tests, diagnostics, CPU bounds).
export function hullView(data, base) {
  const w=i=>data[base*4+i],h=w(0),nv=h&255,nf=h>>>8&255,ne=h>>>16&255,flags=h>>>24;
  const f=i=>bitsf32(w(i)),byte=(word,b)=>w(word+(b>>2))>>>((b&3)*8)&255;
  return {nv,nf,ne,closed:(flags&HULL_FLAG_CLOSED)!==0,
    vertex:i=>[f((4+i)*4),f((4+i)*4+1),f((4+i)*4+2)],
    plane:k=>{const o=(4+nv+k)*4;return {n:[f(o),f(o+1),f(o+2)],d:f(o+3)}},
    face:k=>{const entry=w((4+nv+nf)*4+k),start=entry&0xffff,count=entry>>>16;return Array.from({length:count},(_,i)=>byte(w(1),start+i))},
    edge:k=>{const x=w(w(2)+k);return [x&255,x>>>8&255,x>>>16&255,x>>>24]},
    adjacency:i=>{const x=w((4+i)*4+3),start=x&0xffff,deg=x>>>16;return Array.from({length:deg},(_,j)=>byte(w(3),start+j))}};
}

// World vertices of a convex collider (body pose + collider local pose + scale).
export function colliderVertices(scene,c,body=scene.bodies[c.body]){
  const view=hullView(scene.hullData,hullBase(c.params)),s=c.params[1]||1,lp=c.localPos||[0,0,0],lq=c.localRot||[0,0,0,1],q=body.q||[0,0,0,1];
  return Array.from({length:view.nv},(_,i)=>add(body.p,rotate(q,add(lp,rotate(lq,scale(view.vertex(i),s))))));
}

export function colliderBounds(scene,c,body=scene.bodies[c.body]) {
  const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
  if(c.shape===3){for(const v of colliderVertices(scene,c,body))for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],v[k]);hi[k]=Math.max(hi[k],v[k])}}
  else{const r=c.shape===4?1e6:c.params[0],p=add(body.p,rotate(body.q||[0,0,0,1],c.localPos||[0,0,0]));for(let k=0;k<3;k++){lo[k]=p[k]-r;hi[k]=p[k]+r}}
  return {lo:lo.map(x=>x-.0001),hi:hi.map(x=>x+.0001)};
}

// Hierarchical broadphase sizing: base = median diameter; levels cover every
// collider (mirrors hierarchical_grid in engine-schema and hierLevel in Slang).
export function gridConfig(colliders) {
  // Matches build_aabbs: capsules (shape 1) reach radius + half height.
  const radius=c=>c.shape===1?c.params[0]+c.params[1]:c.params[3]>0?c.params[3]:Math.hypot(c.params[0],c.params[1],c.params[2]);
  const diameters=colliders.filter(c=>c.shape!==4).map(c=>2*radius(c)+0.0004).sort((a,b)=>a-b);
  const base=Math.fround(Math.max(.001,diameters[Math.floor(diameters.length/2)]||1)*(1+1e-5));
  const levels=Math.min(31,Math.max(2,2+Math.ceil(Math.log2(Math.max(1,(diameters.at(-1)||base)/base)))));
  return {base,levels};
}

// A grid of identical houses (columns along +x, rows along +z). Every copy
// shares the single house's hull records (same arena offsets) and render mesh
// (drawn instanced), so GPU memory grows only with bodies and colliders. The
// projectile is a heavy ball rolled along the ground down the middle column.
// base: an already built (e.g. bond-preloaded) single-house scene to replicate.
// `positions` ([dx, dz] per copy) replaces the regular grid; `shotX` is then
// the x of the row shot and `roads` are drawn on the ground (buildSuburb).
export function buildGrid(asset, level, {columns=1, rows=20, gap=2.5, projectileRadius=1.6, projectileMass=200000, projectileSpeed=40, maxHullVertices, base, positions, shotX:shotAt, roads=[]}={}) {
  const one=base??buildScene(asset,level,{maxHullVertices}),frags=one.projectileStart-1;
  const pitchX=one.extentX+gap,pitchZ=one.extentZ+gap;
  const offsetX=c=>(c-(columns-1)/2)*pitchX;
  const offsets=positions??Array.from({length:columns*rows},(_,k)=>[offsetX(k%columns),(Math.floor(k/columns)-(rows-1)/2)*pitchZ]);
  const copies=offsets.length;
  const spanOf=k=>Math.max(...offsets.map(o=>o[k]))-Math.min(...offsets.map(o=>o[k]));
  const width=spanOf(0)+one.extentX,length=spanOf(1)+one.extentZ;
  const bodies=[one.bodies[0]],colliders=[one.colliders[0]],joints=[];
  const houseColliders=one.colliders.filter(c=>c.body>=1&&c.body<one.projectileStart);
  for(let k=0;k<copies;k++){
    const [dx,dz]=offsets[k],base=k*frags;
    for(let i=1;i<one.projectileStart;i++){const b=one.bodies[i];bodies.push({...b,p:[b.p[0]+dx,b.p[1],b.p[2]+dz]})}
    for(const col of houseColliders)colliders.push({...col,body:col.body+base});
    // World-anchored bonds keep body 0 and move their world anchor with the copy.
    for(const j of one.joints)joints.push(j.world?{...j,bodyA:j.bodyA+base,anchorB:[j.anchorB[0]+dx,j.anchorB[1],j.anchorB[2]+dz]}
      :{...j,bodyA:j.bodyA+base,bodyB:j.bodyB+base});
  }
  const projectileStart=bodies.length,projectileVertices=[];
  for(let slot=0;slot<one.projectileCount;slot++){
    const id=bodies.length;
    bodies.push({p:[slot*4,-1000,0],q:[0,0,0,1],invMass:0,invI:[0,0,0],flags:0});
    colliders.push({body:id,shape:0,density:0,params:[projectileRadius,0,0,0]});
    const src=one.projectileVertices,stride=src.length/one.projectileCount,r0=one.projectileRadius;
    for(let v=0;v<stride;v+=10){const o=slot*stride+v;projectileVertices.push(src[o]/r0*projectileRadius,src[o+1]/r0*projectileRadius,src[o+2]/r0*projectileRadius,src[o+3],src[o+4],src[o+5],src[o+6],src[o+7],src[o+8],id)}
  }
  const span=Math.max(width,length),shotX=shotAt??offsetX(Math.floor(columns/2));
  return {...one,bodies,colliders,joints,projectileStart,projectileRadius,projectileMass,projectileSpeed,projectileArc:false,roadVertices:roadMesh(roads),
    projectileVertices:new Float32Array(projectileVertices),copies,columns,rows,instanceStride:frags,
    rowPitch:pitchZ,columnPitch:pitchX,rowLength:length,gridWidth:width,
    extent:Math.max(one.extent,span),center:[0,one.height*.5,0],gridCellSize:gridConfig(colliders).base,
    // Measured on house level 2 after impact: ~3 candidates and ~5 contacts per
    // collider. Eight each keeps headroom without gigabytes of contact buffers.
    // ~100k pieces wake behind the ball; their contact graph can need more than
    // 32 Jones-Plassmann rounds (and headroom past 32 colours; kernels allow 64).
    capacity:{candidatesPerCollider:8,contactsPerCollider:8,maxColors:64,colorRounds:64},
    camera:{yaw:columns>1?.75:1.15,pitch:columns>1?.5:.32,distance:span*.8,minDistance:3},
    // Ready-made shot: from beyond the first house, rolling along +z at ground level.
    rowShot:{origin:[shotX,projectileRadius+.02,-length/2-projectileRadius-8],direction:[0,0,1]}};
}

// A suburb: blocksX × blocksZ blocks of lotsX × lotsZ houses, with yards between
// houses and roads (asphalt, sidewalks, dashed centre lines) between blocks.
// The row shot rolls down one column of lots, through a subset of the houses.
export function buildSuburb(asset, level, {blocksX=4, blocksZ=3, lotsX=2, lotsZ=2, yardX=5, yardZ=6, road=8, sidewalk=1.5, shotBlock, ...options}={}) {
  const one=options.base??buildScene(asset,level,{maxHullVertices:options.maxHullVertices});
  const lotX=one.extentX+yardX,lotZ=one.extentZ+yardZ;
  const blockW=lotsX*lotX,blockL=lotsZ*lotZ,pitchX=blockW+road,pitchZ=blockL+road;
  const cx=b=>(b-(blocksX-1)/2)*pitchX,cz=b=>(b-(blocksZ-1)/2)*pitchZ;
  const positions=[];
  for(let bz=0;bz<blocksZ;bz++)for(let bx=0;bx<blocksX;bx++)for(let lz=0;lz<lotsZ;lz++)for(let lx=0;lx<lotsX;lx++)
    positions.push([cx(bx)+(lx-(lotsX-1)/2)*lotX,cz(bz)+(lz-(lotsZ-1)/2)*lotZ]);
  // Roads run between and around the blocks: x0, z0, x1, z1 and orientation.
  const halfW=blocksX*pitchX/2,halfL=blocksZ*pitchZ/2,roads=[];
  for(let i=0;i<=blocksX;i++){const x=-halfW+i*pitchX;roads.push({x0:x-road/2,x1:x+road/2,z0:-halfL-road/2,z1:halfL+road/2,alongZ:true})}
  for(let i=0;i<=blocksZ;i++){const z=-halfL+i*pitchZ;roads.push({x0:-halfW-road/2,x1:halfW+road/2,z0:z-road/2,z1:z+road/2,alongZ:false,sidewalk})}
  for(const r of roads)r.sidewalk=sidewalk;
  const column=shotBlock??Math.floor((blocksX-1)/2);
  const scene=buildGrid(asset,level,{...options,base:one,positions,shotX:cx(column)+(0-(lotsX-1)/2)*lotX,roads});
  return {...scene,columns:blocksX*lotsX,rows:blocksZ*lotsZ,extent:Math.max(scene.extent,2*halfW,2*halfL)};
}

// Flat road surfaces drawn on the static ground (render vertices: position,
// normal, colour, body 0). Roads are decoration only: the physics ground stays
// the plane y = 0.
export function roadMesh(roads) {
  const v=[];
  const quad=(x0,z0,x1,z1,y,c)=>{for(const [x,z] of [[x0,z0],[x0,z1],[x1,z1],[x0,z0],[x1,z1],[x1,z0]])v.push(x,y,z,0,1,0,...c,0)};
  const asphalt=[.17,.18,.2],walk=[.42,.42,.4],paint=[.86,.72,.25];
  const cross=(r,s)=>r.alongZ!==s.alongZ&&s.x0<=r.x1&&s.x1>=r.x0&&s.z0<=r.z1&&s.z1>=r.z0;
  for(const r of roads){
    quad(r.x0,r.z0,r.x1,r.z1,.004,asphalt);
    const w=r.sidewalk??0;
    if(r.alongZ){quad(r.x0-w,r.z0,r.x0,r.z1,.003,walk);quad(r.x1,r.z0,r.x1+w,r.z1,.003,walk)}
    else{quad(r.x0,r.z0-w,r.x1,r.z0,.003,walk);quad(r.x0,r.z1,r.x1,r.z1+w,.003,walk)}
    // Dashed centre line, broken at intersections.
    const others=roads.filter(s=>cross(r,s)),mid=r.alongZ?(r.x0+r.x1)/2:(r.z0+r.z1)/2;
    const [a,b]=r.alongZ?[r.z0,r.z1]:[r.x0,r.x1];
    for(let t=a+1;t+2<=b;t+=4){
      if(others.some(s=>r.alongZ?(t+2>=s.z0&&t<=s.z1):(t+2>=s.x0&&t<=s.x1)))continue;
      if(r.alongZ)quad(mid-.08,t,mid+.08,t+2,.006,paint);else quad(t,mid-.08,t+2,mid+.08,.006,paint);
    }
  }
  return new Float32Array(v);
}

// A single row of `copies` houses along +z (a one-column grid).
export function buildRow(asset, level, {copies=20, ...options}={}) {
  return buildGrid(asset, level, {...options, columns:1, rows:copies});
}

export function buildScene(asset, level, {maxHullVertices}={}) {
  const selected=asset.physics.fragments.filter(f=>f.level===level);
  if(!selected.length)throw new Error('No fragments at this level');
  const bodies=[{p:[0,0,0],q:[0,0,0,1],invMass:0,invI:[0,0,0],flags:0}];
  const colliders=[{body:0,shape:4,density:0,params:[0,1,0,0]}];
  const arena=new HullArena({maxVertices:maxHullVertices}),vertices=[],lines=[],byId=new Map(),skippedHulls=[];
  let lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
  for(const f of selected)for(const h of asset.physics.hulls.slice(f.hullBegin,f.hullBegin+f.hullCount))for(const v of h.vertices)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],v[k]);hi[k]=Math.max(hi[k],v[k]);}
  if(!lo.every(Number.isFinite))throw new Error('Selected level has no collision hulls');
  const center=scale(add(lo,hi),.5),height=hi[1]-lo[1],extent=Math.max(...sub(hi,lo));
  const placement=[-center[0],-lo[1],-center[2]];
  for(const f of selected){
    if(!(f.mass>0)||!f.hullCount)throw new Error(`Fragment ${f.id} has no dynamic mass or hull`);
    const {q,moments}=principalAxes(f.inertia),id=bodies.length;
    const body={p:add(f.com,placement),q,invMass:1/f.mass,invI:moments.map(x=>1/x),flags:5};
    bodies.push(body);byId.set(f.id,{id,f,q});
    let kept=0;
    for(let h=f.hullBegin;h<f.hullBegin+f.hullCount;h++){
      const source=asset.physics.hulls[h].vertices,verts=source.map(v=>inverseRotate(q,sub(v,f.com)));
      // The asset stores hulls as f32 in its own frame; judge convexity at that precision.
      const sourceScale=source.reduce((m,v)=>Math.max(m,Math.abs(v[0]),Math.abs(v[1]),Math.abs(v[2])),0);
      let child;
      try{child=appendHull(arena,verts,asset.physics.hullFaces[h],{sourceScale})}catch(e){
        // A zero-volume child (a flat sheet baked into a compound) adds no
        // collision volume and no mass: mass/inertia come from the fragment.
        // Drop it only while the fragment keeps real hulls; never thicken it.
        if(/zero-volume/.test(e.message)){skippedHulls.push({fragment:f.id,hull:h});continue}
        throw new Error(`Fragment ${f.id}, hull ${h}: ${e.message}`);
      }
      colliders.push({body:id,shape:3,density:f.mass/Math.max(f.volume,1e-12),params:child.params,localPos:child.localPos});
      kept++;
    }
    if(!kept)throw new Error(`Fragment ${f.id}: every collision hull has zero volume; recook the source collision asset`);
    const node=asset.json.nodes[f.node],mesh=asset.json.meshes[node?.mesh];
    if(!mesh)throw new Error(`Fragment ${f.id} has no render mesh`);
    for(const prim of mesh.primitives){
      if((prim.mode??4)!==4)throw new Error('Demo supports triangle meshes');
      const positions=asset.accessor(prim.attributes.POSITION),normals=prim.attributes.NORMAL===undefined?null:asset.accessor(prim.attributes.NORMAL);
      const indices=prim.indices===undefined?positions.map((_,i)=>[i]):asset.accessor(prim.indices);
      const color=asset.json.materials?.[prim.material]?.pbrMetallicRoughness?.baseColorFactor?.slice(0,3)??[.65,.58,.47];
      for(let t=0;t<indices.length;t+=3){const ids=indices.slice(t,t+3).map(v=>v[0]);if(ids.length!==3||ids.some(i=>!positions[i]))throw new Error('Invalid triangle indices');
        const normal=norm(cross(sub(positions[ids[1]],positions[ids[0]]),sub(positions[ids[2]],positions[ids[0]])));
        for(const i of ids)vertices.push(...inverseRotate(q,positions[i]),...inverseRotate(q,normals?.[i]??normal),...color,id);
      }
    }
  }
  const bonds=asset.physics.bonds.filter(b=>b.level===level);
  // Each bond becomes a breakable weld (JOINT_BOND) at its centroid: all six
  // DOF locked at the baked rest pose, breaking at the bond's capacities. World
  // bonds (b = -1, anchors) weld to the static ground body 0.
  const joints=[];
  for(const b of bonds){
    const A=byId.get(b.a),B=b.b>=0?byId.get(b.b):null;if(!A||(b.b>=0&&!B))continue;
    const c=add(b.centroid,placement),qA=A.q,qB=B?B.q:[0,0,0,1];
    joints.push({bodyA:A.id,bodyB:B?B.id:0,
      anchorA:inverseRotate(qA,sub(c,bodies[A.id].p)),anchorB:B?inverseRotate(qB,sub(c,bodies[B.id].p)):c,
      rest:quatMul(quatConjugate(qA),qB),normal:inverseRotate(qA,b.normal),frameU:inverseRotate(qA,b.frameU),
      strength:bondCapacities(asset,b),k:bondSprings(asset,b),world:!B});
  }
  const scaledJoints=scaleBondSprings(joints,bodies,1/120);
  for(const b of bonds){
    const a=byId.get(b.a),other=byId.get(b.b);if(!a)continue;
    const endpoints=other?[[a,[0,0,0]],[other,[0,0,0]]]:[[a,[0,0,0]],[null,add(b.centroid,placement)]];
    for(const [body,p] of endpoints)lines.push(...p,0,1,0,...(b.anchor?[.95,.6,.2]:[.25,.9,.8]),body?.id??0);
  }
  const projectileStart=bodies.length,projectileCount=16;
  const projectileRadius=Math.max(.08,Math.min(.4,extent*.04));
  const projectileMass=selected.reduce((sum,f)=>sum+f.mass,0)/selected.length*1.5;
  const projectileVertices=[];
  for(let slot=0;slot<projectileCount;slot++){
    const id=bodies.length;
    // Inactive slots are static and below the world. Activated slots get real mass.
    bodies.push({p:[slot*2,-1000,0],q:[0,0,0,1],invMass:0,invI:[0,0,0],flags:0});
    colliders.push({body:id,shape:0,density:0,params:[projectileRadius,0,0,0]});
    const point=(u,v)=>[Math.sin(v)*Math.cos(u),Math.cos(v),Math.sin(v)*Math.sin(u)];
    for(let y=0;y<12;y++)for(let x=0;x<24;x++){
      const a=point(x/24*Math.PI*2,y/12*Math.PI),b=point((x+1)/24*Math.PI*2,y/12*Math.PI);
      const c=point((x+1)/24*Math.PI*2,(y+1)/12*Math.PI),d=point(x/24*Math.PI*2,(y+1)/12*Math.PI);
      for(const n of [a,b,c,a,c,d])projectileVertices.push(...scale(n,projectileRadius),...n,1,.3,.08,id);
    }
  }
  return {projectileStart,projectileCount,projectileRadius,projectileMass,projectileVertices:new Float32Array(projectileVertices),bodies,colliders,joints:scaledJoints,hullData:arena.data,skippedHulls,vertices:new Float32Array(vertices),lines:new Float32Array(lines),selected,bonds,byId,
    extent,extentX:hi[0]-lo[0],extentZ:hi[2]-lo[2],height,copies:1,instanceStride:0,center:[0,height*.5+placement[1]+lo[1],0],gridCellSize:gridConfig(colliders).base,
    // Flags: 1 reuse contacts, 64 world-frame inertia (FLAG_WORLD_INERTIA).
    params:{flags:438465,iterations:8,dt:1/120,gravityY:-9.81,contactOffset:.002,alpha:.95,sleepLin:.03,sleepAng:.05,sleepTime:1}};
}
