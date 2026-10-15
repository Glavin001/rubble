export const add=(a,b)=>a.map((x,i)=>x+b[i]);
export const sub=(a,b)=>a.map((x,i)=>x-b[i]);
export const scale=(a,s)=>a.map(x=>x*s);
export const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
export const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export const norm=a=>scale(a,1/(Math.hypot(...a)||1));
export const rotate=(q,v)=>add(v,scale(cross(q.slice(0,3),add(cross(q.slice(0,3),v),scale(v,q[3]))),2));
export const inverseRotate=(q,v)=>rotate([-q[0],-q[1],-q[2],q[3]],v);

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
export function cookHull(vertices, topology) {
  try{return cookFaces(vertices,topology,true)}catch(error){
    if(error.message!=='Degenerate convex face')throw error;
    return cookFaces(vertices,topology,false);
  }
}
function cookFaces(vertices, topology, normalizeFaces) {
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
    if(normalizeFaces&&vertices.some(p=>dot(n,p)-d>tol*2))throw new Error('Non-convex collision hull');
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
  constructor(){this.words=[];this.dedup=new Map();this.blocks=new Map();this.free=[];this.dirty=[];if(this.insert(cuboidHull())!==0)throw new Error('unit cube must be record 0')}
  insert(hull){
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
export function appendHull(arena, vertices, topology) {
  const {hull,offset}=recenterHull(cookHull(vertices,topology)),base=arena.insert(hull);
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
  const radius=c=>c.params[3]>0?c.params[3]:Math.hypot(c.params[0],c.params[1],c.params[2]);
  const diameters=colliders.filter(c=>c.shape!==4).map(c=>2*radius(c)+0.0004).sort((a,b)=>a-b);
  const base=Math.fround(Math.max(.001,diameters[Math.floor(diameters.length/2)]||1)*(1+1e-5));
  const levels=Math.min(31,Math.max(2,2+Math.ceil(Math.log2(Math.max(1,(diameters.at(-1)||base)/base)))));
  return {base,levels};
}

export function buildScene(asset, level) {
  const selected=asset.physics.fragments.filter(f=>f.level===level);
  if(!selected.length)throw new Error('No fragments at this level');
  const bodies=[{p:[0,0,0],q:[0,0,0,1],invMass:0,invI:[0,0,0],flags:0}];
  const colliders=[{body:0,shape:4,density:0,params:[0,1,0,0]}];
  const arena=new HullArena(),vertices=[],lines=[],byId=new Map(),skippedHulls=[];
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
      const verts=asset.physics.hulls[h].vertices.map(v=>inverseRotate(q,sub(v,f.com)));
      let child;
      try{child=appendHull(arena,verts,asset.physics.hullFaces[h])}catch(e){
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
  // Bonds are inspection-only; initially sleeping pieces wake on projectile impact.
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
  return {projectileStart,projectileCount,projectileRadius,projectileMass,projectileVertices:new Float32Array(projectileVertices),bodies,colliders,hullData:arena.data,skippedHulls,vertices:new Float32Array(vertices),lines:new Float32Array(lines),selected,bonds,byId,
    extent,height,center:[0,height*.5+placement[1]+lo[1],0],gridCellSize:gridConfig(colliders).base,
    params:{flags:1,iterations:8,dt:1/120,gravityY:-9.81,contactOffset:.002,alpha:.95,sleepLin:.03,sleepAng:.05,sleepTime:1}};
}
