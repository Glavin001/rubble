// SHARDWRIGHT_fracture v1. CPU-only reader; reusable by stress-solver clients.
export const EXTENSION = 'SHARDWRIGHT_fracture';
const fail = message => { throw new Error(message); };
const decoder = new TextDecoder('utf-8', { fatal: true });

// Bounds-checked FlatBuffers reader for schemas/frac.fbs (schema version 1).
// No generated code or npm runtime is needed by the standalone demo.
class FracBuffer {
  constructor(bytes) { this.bytes = bytes; this.d = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength); }
  check(p, n) { if (!Number.isSafeInteger(p) || p < 0 || n < 0 || p + n > this.d.byteLength) fail('Truncated FRAC payload'); return p; }
  u64(p) { return this.d.getBigUint64(this.check(p,8),true).toString(); }
  u32(p) { return this.d.getUint32(this.check(p,4),true); }
  i32(p) { return this.d.getInt32(this.check(p,4),true); }
  u16(p) { return this.d.getUint16(this.check(p,2),true); }
  u8(p) { return this.d.getUint8(this.check(p,1)); }
  f32(p) { const x=this.d.getFloat32(this.check(p,4),true); if(!Number.isFinite(x)) fail('Non-finite FRAC value'); return x; }
  field(t, i) { const v=t-this.i32(t), n=this.u16(v); this.check(v,n); const o=4+2*i; const offset=o<n?this.u16(v+o):0; return offset?this.check(t+offset,1):0; }
  scalar(t,i,type='u32',fallback=0) { const p=this.field(t,i); return p?this[type](p):fallback; }
  struct(t,i,n=3) { const p=this.field(t,i); return p?Array.from({length:n},(_,k)=>this.f32(p+4*k)):Array(n).fill(0); }
  vector(t,i,stride,read) { const p=this.field(t,i); if(!p)return []; const v=p+this.u32(p), n=this.u32(v); this.check(v+4,n*stride); return Array.from({length:n},(_,k)=>read(v+4+k*stride)); }
  tables(t,i,read) { return this.vector(t,i,4,p=>read(p+this.u32(p))); }
  string(t,i) { const p=this.field(t,i); if(!p)return ''; const v=p+this.u32(p), n=this.u32(v); this.check(v+4,n); return decoder.decode(this.bytes.subarray(v+4,v+4+n)); }
}
export function readPhysics(bytes) {
  const r=new FracBuffer(bytes);
  if(bytes.length<8 || decoder.decode(bytes.subarray(4,8))!=='FRAC') fail('Missing FRAC identifier');
  const root=r.u32(0);
  if(r.scalar(root,0,'u16')!==1) fail('Unsupported physics schema');
  const fragments=r.tables(root,6,t=>({
    id:r.scalar(t,0), level:r.scalar(t,1,'u8'),parent:r.scalar(t,2,'i32',-1),
    childBegin:r.scalar(t,3), childCount:r.scalar(t,4), mass:r.scalar(t,5,'f32'),
    com:r.struct(t,6), inertia:r.struct(t,7,9), materialIds:r.vector(t,8,2,p=>r.u16(p)),
    materialFractions:r.vector(t,9,4,p=>r.f32(p)),hullBegin:r.scalar(t,10),hullCount:r.scalar(t,11,'u16'),
    particleCandidate:!!r.scalar(t,12,'u8'),role:r.scalar(t,13,'u8'),node:r.scalar(t,14,'i32',-1),
    volume:r.scalar(t,15,'f32'), interiorArea:r.scalar(t,16,'f32'),component:r.scalar(t,17),
    cellBegin:r.scalar(t,18),cellCount:r.scalar(t,19),
  }));
  const bonds=r.tables(root,7,t=>({
    id:r.scalar(t,0),level:r.scalar(t,1,'u8'),a:r.scalar(t,2),b:r.scalar(t,3,'i32'),
    area:r.scalar(t,4,'f32'),centroid:r.struct(t,5),normal:r.struct(t,6),planarity:r.scalar(t,7,'f32'),
    frameU:r.struct(t,8),frameV:r.struct(t,9),iUU:r.scalar(t,10,'f32'),iVV:r.scalar(t,11,'f32'),
    iUV:r.scalar(t,12,'f32'),polarMoment:r.scalar(t,13,'f32'),extentCenter:r.struct(t,14),extentHalf:r.struct(t,15,2),
    distA:r.scalar(t,16,'f32'),distB:r.scalar(t,17,'f32'),
    composition:r.tables(t,18,c=>({kind:r.scalar(c,0,'u8'),interfaceMaterial:r.scalar(c,1,'i32',-1),fraction:r.scalar(c,2,'f32')})),
    rebarCount:r.scalar(t,19,'u16'),rebarArea:r.scalar(t,20,'f32'),rebarDirection:r.struct(t,21),
    strengthScale:r.scalar(t,22,'f32'),parentBond:r.scalar(t,23,'i32',-1),childBegin:r.scalar(t,24),childCount:r.scalar(t,25),
    loopBegin:r.scalar(t,26),loopCount:r.scalar(t,27,'u16'),spawnBegin:r.scalar(t,28),spawnCount:r.scalar(t,29),
    anchor:!!r.scalar(t,30,'u8'),interfaceCount:r.scalar(t,31),
  }));
  const hulls=r.tables(root,8,t=>({fragment:r.scalar(t,1),vertices:r.vector(t,0,12,p=>[r.f32(p),r.f32(p+4),r.f32(p+8)])}));
  const hullFaces=r.tables(root,9,t=>({indices:r.vector(t,0,4,p=>r.u32(p)),sizes:r.vector(t,1,1,p=>r.u8(p))}));
  const loops=r.tables(root,10,t=>r.vector(t,0,12,p=>[r.f32(p),r.f32(p+4),r.f32(p+8)]));
  const spawn=r.vector(root,12,24,p=>({p:[r.f32(p),r.f32(p+4),r.f32(p+8)],n:[r.f32(p+12),r.f32(p+16),r.f32(p+20)]}));
  const bondChildren=r.vector(root,17,4,p=>r.u32(p));
  for (const [i,f] of fragments.entries()) {
    if(f.id!==i || f.mass<0 || f.parent < -1 || f.parent>=fragments.length || f.materialIds.length!==f.materialFractions.length || f.hullBegin+f.hullCount>hulls.length || f.childBegin+f.childCount>fragments.length) fail('Invalid fragment ranges');
  }
  for(const [i,b] of bonds.entries()) if(b.id!==i || b.a>=fragments.length || b.b < -1 || b.b>=fragments.length || b.area<0 || b.childBegin+b.childCount>bondChildren.length || b.loopBegin+b.loopCount>loops.length || b.spawnBegin+b.spawnCount>spawn.length) fail('Invalid bond references');
  if(hullFaces.length!==hulls.length)fail('Missing hull faces');
  for(const [i,h] of hulls.entries()) if(h.fragment>=fragments.length || h.vertices.length<4 || h.vertices.length>64 || hullFaces[i].sizes.reduce((a,b)=>a+b,0)!==hullFaces[i].indices.length || hullFaces[i].indices.some(v=>v>=h.vertices.length)) fail('Invalid convex hull');
  return {schemaVersion:1,seed:r.scalar(root,3,'u64','0'),variant:r.scalar(root,14),levelFragmentBegin:r.vector(root,15,4,p=>r.u32(p)),levelFragmentCount:r.vector(root,16,4,p=>r.u32(p)),name:r.string(root,13),toolVersion:r.string(root,1),settingsHash:r.string(root,2),
    materialLibraryId:r.string(root,4),materialLibraryVersion:r.string(root,5),
    levels:r.scalar(root,11,'u8'),fragments,bonds,hulls,hullFaces,loops,spawn,bondChildren};
}
export function parseFractureGlb(buffer) {
  const d=new DataView(buffer);
  if(buffer.byteLength<20 || d.getUint32(0,true)!==0x46546c67 || d.getUint32(4,true)!==2 || d.getUint32(8,true)!==buffer.byteLength)fail('Invalid GLB header or length');
  let json,bin;
  for(let p=12;p<buffer.byteLength;) {
    if(p+8>buffer.byteLength)fail('Truncated GLB chunk');
    const n=d.getUint32(p,true),type=d.getUint32(p+4,true);p+=8;
    if(n%4 || p+n>buffer.byteLength)fail('Invalid GLB chunk range');
    if(type===0x4e4f534a){if(json)fail('Duplicate JSON');json=JSON.parse(decoder.decode(new Uint8Array(buffer,p,n)));}
    if(type===0x004e4942){if(bin)fail('Duplicate BIN');bin=new Uint8Array(buffer,p,n);}
    p+=n;
  }
  const ext=json?.extensions?.[EXTENSION];
  if(!bin || ext?.version!==1 || ext.encoding!=='flatbuffers' || ext.physicsSchemaVersion!==1)fail('This GLB needs the SHARDWRIGHT_fracture v1 export');
  if(ext.coordinates?.units!=='meters' || ext.coordinates?.up!=='Y' || ext.coordinates?.handedness!=='right')fail('Unsupported fracture coordinate convention');
  if(json.extensionsRequired?.includes('EXT_meshopt_compression'))fail('Export with --uncompressed for this demo');
  if(json.buffers?.length!==1 || json.buffers[0].uri || json.buffers[0].byteLength>bin.length || json.images?.some(i=>i.uri))fail('Expected a self-contained GLB');
  const slice=(index)=>{
    const v=json.bufferViews?.[index];
    if(!v || v.buffer!==0 || !Number.isSafeInteger(v.byteLength) || !Number.isSafeInteger(v.byteOffset??0) || (v.byteOffset??0)<0 || v.byteLength<0 || (v.byteOffset??0)+v.byteLength>json.buffers[0].byteLength)fail('Invalid bufferView');
    return bin.subarray(v.byteOffset??0,(v.byteOffset??0)+v.byteLength);
  };
  const physics=readPhysics(slice(ext.bufferView));
  const library=ext.materialLibrary;
  if(library?.library_id!==physics.materialLibraryId || library.library_version!==physics.materialLibraryVersion)fail('Mismatched material library');
  for(const f of physics.fragments) {
    if(f.node>=0 && json.nodes?.[f.node]?.extras?.fragment_id!==f.id)fail('Mismatched fragment node');
    if(f.materialIds.some(id=>!library.material?.[id]))fail('Missing physical material');
  }
  for(const b of physics.bonds) if(b.composition.some(c=>c.interfaceMaterial>=0 && !library.interface_material?.[c.interfaceMaterial-library.material.length]))fail('Missing interface material');
  const accessor=(index)=>{
    const a=json.accessors?.[index];if(!a||a.sparse)fail('Missing or sparse accessor unsupported in this demo');
    const v=json.bufferViews[a.bufferView];if(v?.extensions?.EXT_meshopt_compression)fail('Export with --uncompressed for this demo');
    const bytes=slice(a.bufferView),dv=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
    const width={SCALAR:1,VEC2:2,VEC3:3,VEC4:4}[a.type];
    const size={5121:1,5123:2,5125:4,5126:4}[a.componentType];
    if(!width||!size||!Number.isSafeInteger(a.count)||a.count<0)fail('Unsupported accessor');
    const stride=v.byteStride??width*size,off=a.byteOffset??0;
    if(!Number.isSafeInteger(off)||!Number.isSafeInteger(stride)||stride<width*size||off<0 || (a.count && off+(a.count-1)*stride+width*size>bytes.length))fail('Accessor outside bufferView');
    const get={5121:'getUint8',5123:'getUint16',5125:'getUint32',5126:'getFloat32'}[a.componentType];
    return Array.from({length:a.count},(_,i)=>Array.from({length:width},(_,k)=>{const x=dv[get](off+i*stride+k*size,true);if(!Number.isFinite(x))fail('Non-finite mesh attribute');return x}));
  };
  return {json,physics,library,accessor,bytes:buffer.byteLength};
}
