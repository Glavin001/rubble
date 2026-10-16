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

// Padded float3 hull ABI: counts, vertices, outward normals, face ranges,
// two records per edge (vertices + adjacent faces), interior/lo/hi, face loops.
// Vertices remain in the body's principal frame; compound mass/inertia are unchanged.
export function appendHull(data, vertices, topology) {
  const length=data.length;
  try{return packHull(data,vertices,topology,true)}catch(error){
    data.length=length;
    if(error.message!=='Degenerate convex face')throw error;
    // Keep a conservative exhaustive SAT path for imperfect baked face soups.
    // Never apply adjacency-based pruning to topology we cannot certify.
    return packHull(data,vertices,topology,false);
  }
}
function packHull(data, vertices, topology, normalizeFaces) {
  if(vertices.length<4||vertices.length>64||vertices.some(v=>v.length!==3||!v.every(Number.isFinite)))throw new Error('Convex hull needs 4–64 finite vertices');
  const center=scale(vertices.reduce((a,v)=>add(a,v),[0,0,0]),1/vertices.length);
  const lo=[0,1,2].map(k=>Math.min(...vertices.map(v=>v[k]))),hi=[0,1,2].map(k=>Math.max(...vertices.map(v=>v[k])));
  // Shardwright may duplicate a geometric vertex across face loops. Weld
  // exact positions for adjacency only; retain original vertex coordinates/order.
  const canonical=new Map(),vertexIds=vertices.map((v,i)=>{const key=v.join(',');if(!canonical.has(key))canonical.set(key,i);return canonical.get(key)});
  const span=Math.hypot(...sub(hi,lo)),relative=vertices.map(v=>sub(v,center));
  const axis=relative.reduce((a,b)=>dot(a,a)>dot(b,b)?a:b);
  const plane=relative.map(v=>cross(axis,v)).reduce((a,b)=>dot(a,a)>dot(b,b)?a:b);
  const planeLength=Math.hypot(...plane);
  if(!planeLength||Math.max(...relative.map(v=>Math.abs(dot(v,plane))/planeLength))<=Math.max(1e-10,span*1e-8))throw new Error('Degenerate zero-volume hull: recook the source collision asset');
  const base=data.length/4,faces=[],edges=new Map(),planes=[],rawFaces=[];let offset=0,closed=normalizeFaces;
  const tolerance=Math.max(1e-9,Math.hypot(...sub(hi,lo))*1e-6);
  for(const count of topology.sizes){
    if(count<3||count>64)throw new Error('Convex face size outside 3..64');
    const ids=topology.indices.slice(offset,offset+count).map(i=>vertexIds[i]);offset+=count;
    if(ids.length!==count||ids.some(i=>!Number.isInteger(i)||!vertices[i]))throw new Error('Invalid convex face indices');
    let sum=[0,0,0];for(let i=1;i<count-1;i++)sum=add(sum,cross(sub(vertices[ids[i]],vertices[ids[0]]),sub(vertices[ids[i+1]],vertices[ids[0]])));
    if(Math.hypot(...sum)<1e-14)continue;
    let n=norm(sum);if(dot(n,sub(vertices[ids[0]],center))<0)n=scale(n,-1);
    const clean=[...new Set(ids)];
    if(dot(sum,n)<0)clean.reverse();
    rawFaces.push({ids:clean,n});
    const d=dot(n,vertices[ids[0]]);
    if(!planes.some(p=>dot(p.n,n)>1-1e-7&&Math.abs(p.d-d)<tolerance))planes.push({n,d});
  }
  // Merge duplicate/coplanar face patches and remove T-junction/collinear edge
  // vertices using a planar convex loop. This does not alter the convex volume.
  for(const entry of (normalizeFaces?planes:rawFaces)){
    const {n,d}=entry;
    const u=norm(cross(n,Math.abs(n[0])<.8?[1,0,0]:[0,1,0])),v=cross(n,u);
    const points=[...canonical.values()].filter(i=>Math.abs(dot(n,vertices[i])-d)<tolerance).map(i=>({i,x:dot(vertices[i],u),y:dot(vertices[i],v)})).sort((a,b)=>a.x-b.x||a.y-b.y);
    const turn=(a,b,c)=>(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
    const lower=[],upper=[],areaTolerance=1e-14;
    for(const p of points){while(lower.length>1&&turn(lower.at(-2),lower.at(-1),p)<=areaTolerance)lower.pop();lower.push(p)}
    for(const p of points.toReversed()){while(upper.length>1&&turn(upper.at(-2),upper.at(-1),p)<=areaTolerance)upper.pop();upper.push(p)}
    const ids=normalizeFaces?lower.slice(0,-1).concat(upper.slice(0,-1)).map(p=>p.i):entry.ids;
    if(ids.length<3)throw new Error('Degenerate convex face');
    if(normalizeFaces&&vertices.some(p=>dot(n,p)-d>tolerance*2))throw new Error('Non-convex collision hull');
    const face=faces.length;faces.push({ids,n});
    for(let i=0;i<ids.length;i++){
      const a=ids[i],b=ids[(i+1)%ids.length],key=[Math.min(a,b),Math.max(a,b)].join(',');
      const edge=edges.get(key);
      if(edge){if(edge.length!==3||edge[0]!==b||edge[1]!==a)closed=false;if(edge.length===3)edge.push(face)}
      else edges.set(key,[a,b,face]);
    }
  }
  if(offset!==topology.indices.length||faces.length>128||edges.size>192)throw new Error('Invalid convex topology');
  if([...edges.values()].some(e=>e.length!==4))closed=false;
  if(faces.length<4)throw new Error('Degenerate zero-volume hull: recook the source collision asset');
  const push=v=>data.push(...v,0);
  push([vertices.length,faces.length,closed?edges.size:-edges.size]);vertices.forEach(push);faces.forEach(f=>push(f.n));
  let start=base+1+vertices.length+2*faces.length+2*edges.size+3;
  for(const f of faces){push([start,f.ids.length,0]);start+=f.ids.length;}
  edges.forEach(e=>{push(e.slice(0,3));push([e[3]??0,0,0])});
  push(center);push(lo);push(hi);faces.forEach(f=>f.ids.forEach(i=>push(vertices[i])));
  if(data.length/4>=2**24)throw new Error('Hull arena exceeds exact float offset range');
  // Radius only classifies grid levels; it is not a collision shape.
  return [base,vertices.length,faces.length,Math.hypot(...sub(hi,lo))*.5];
}

export function colliderBounds(scene,c,body=scene.bodies[c.body]) {
  const p=body.p,q=body.q||[0,0,0,1],lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
  if(c.shape===3){for(let i=0;i<c.params[1];i++){
    const at=(c.params[0]+1+i)*4,v=add(p,rotate(q,Array.from(scene.hullData.slice(at,at+3))));
    for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],v[k]);hi[k]=Math.max(hi[k],v[k])}
  }}else{const r=c.shape===4?1e6:c.params[0];for(let k=0;k<3;k++){lo[k]=p[k]-r;hi[k]=p[k]+r}}
  return {lo:lo.map(x=>x-.0001),hi:hi.map(x=>x+.0001)};
}

export function gridConfig(colliders) {
  const diameters=colliders.filter(c=>c.shape!==4).map(c=>2*(c.shape===0?c.params[0]:c.params[3])+0.0004).sort((a,b)=>a-b);
  const base=Math.fround(Math.max(.001,diameters[Math.floor(diameters.length/2)]||1)*(1+1e-5));
  const levels=Math.max(2,2+Math.ceil(Math.log2((diameters.at(-1)||base)/base)));
  return {base,levels};
}

export function buildScene(asset, level) {
  const selected=asset.physics.fragments.filter(f=>f.level===level);
  if(!selected.length)throw new Error('No fragments at this level');
  const bodies=[{p:[0,0,0],q:[0,0,0,1],invMass:0,invI:[0,0,0],flags:0}];
  const colliders=[{body:0,shape:4,density:0,params:[0,1,0,0]}];
  const hullData=[],vertices=[],lines=[],byId=new Map();
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
    for(let h=f.hullBegin;h<f.hullBegin+f.hullCount;h++){
      const verts=asset.physics.hulls[h].vertices.map(v=>inverseRotate(q,sub(v,f.com)));
      let params;try{params=appendHull(hullData,verts,asset.physics.hullFaces[h])}catch(e){throw new Error(`Fragment ${f.id}, hull ${h}: ${e.message}`)}
      colliders.push({body:id,shape:3,density:f.mass/Math.max(f.volume,1e-12),params});
    }
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
  return {projectileStart,projectileCount,projectileRadius,projectileMass,projectileVertices:new Float32Array(projectileVertices),bodies,colliders,hullData:new Float32Array(hullData),vertices:new Float32Array(vertices),lines:new Float32Array(lines),selected,bonds,byId,
    extent,height,center:[0,height*.5+placement[1]+lo[1],0],gridCellSize:gridConfig(colliders).base,
    params:{flags:1,iterations:8,dt:1/120,gravityY:-9.81,contactOffset:.002,alpha:.95,sleepLin:.03,sleepAng:.05,sleepTime:1}};
}
