// Thin typed layer over the shared Rust host logic (engine-fracture, built to
// wasm by engine-fracture-wasm): GLB parsing, scene building, hull cooking and
// the per-tick dispatch plan. The native Metal viewer runs the same Rust.
// Build: see fracture/README.md (wasm-pack ... --out-dir ../engine-webgpu/generated/fracture-wasm).
import init,{initSync,Asset,Scene,Plan,HullArena as RustArena,hullView as rustHullView,hullBase,samples,colliderWorldVertices,colliderWorldBounds,preloadDone,scaleBondSprings,Orbit,IndexedMesh,floorVertices} from '../generated/fracture-wasm/engine_fracture_wasm.js';

const wasmUrl=new URL('../generated/fracture-wasm/engine_fracture_wasm_bg.wasm',import.meta.url);
if(globalThis.process?.versions?.node){const {readFileSync}=await import('node:fs');initSync({module:readFileSync(wasmUrl)})}
else await init({module_or_path:wasmUrl});

export {Plan,hullBase,preloadDone,scaleBondSprings,Orbit,IndexedMesh,floorVertices};
export const SAMPLES=samples();

// A parsed fracture GLB: the Rust Asset plus the fields the lab shows.
export function parseFractureGlb(buffer){
 const handle=new Asset(new Uint8Array(buffer)),summary=handle.summary();
 return {handle,...summary,bytes:handle.bytes,materialCount:handle.materialCount,levels:Array.from(handle.levels()),
  levelFragments:level=>handle.levelFragments(level),initialLevel:preferred=>handle.initialLevel(preferred),metadataJson:()=>handle.metadataJson()};
}

// A Rust Scene as the object the hosts use: small fields as plain values,
// render streams and the hull arena as typed arrays; per-body records are
// decoded on first use (diagnostics and tests only).
export function sceneObject(handle){
 const scene={handle,...handle.meta(),hullData:handle.hullData(),vertices:handle.vertices(),lines:handle.lines(),
  projectileVertices:handle.projectileVertices(),roadVertices:handle.roadVertices()};
 let bodies,colliders,joints;
 Object.defineProperties(scene,{
  bodies:{get:()=>bodies??=handle.bodies(),enumerable:true},
  colliders:{get:()=>colliders??=handle.colliders(),enumerable:true},
  joints:{get:()=>joints??=handle.joints(),enumerable:true},
 });
 return scene;
}

// The Rust scene of a host scene object (plain JS scenes are converted once).
export function sceneHandle(scene){
 if(scene.handle)return scene.handle;
 const {hullData,...rest}=scene;
 return Scene.fromJs(rest,hullData?Uint32Array.from(hullData):undefined);
}

export const buildScene=(asset,level,{maxHullVertices}={})=>sceneObject(Scene.fromAsset(asset.handle,level,maxHullVertices));
export const buildCuboidHouse=({colliders='box',pitchDeg}={})=>sceneObject(Scene.cuboidHouse(colliders==='hull',pitchDeg));
export const buildPairScene=({a='box',b='box',count=4096}={})=>sceneObject(Scene.pairs(a,b,count));
// Copies of one house: {columns, rows} grid/row, {suburb:true}, or null.
export const layoutScene=(one,layout,procedural=false)=>sceneObject(sceneHandle(one).layout(layout??null,procedural));
export const withoutJoints=scene=>sceneObject(sceneHandle(scene).withoutJoints());
export const withCapacity=(scene,partial)=>sceneObject(sceneHandle(scene).withCapacity(partial));
export const withParams=(scene,partial)=>sceneObject(sceneHandle(scene).withParams(partial));
export const wakeAll=scene=>sceneObject(sceneHandle(scene).wakeAll());

// World vertices / AABB of a collider at a body pose (default: the scene pose).
export function colliderVertices(scene,c,body=scene.bodies[c.body]){
 const v=colliderWorldVertices(scene.hullData,c,body.p,body.q??[0,0,0,1]);
 return Array.from({length:v.length/3},(_,i)=>[v[3*i],v[3*i+1],v[3*i+2]]);
}
export function colliderBounds(scene,c,body=scene.bodies[c.body]){
 const b=colliderWorldBounds(scene.hullData,c,body.p,body.q??[0,0,0,1]);
 return {lo:[b[0],b[1],b[2]],hi:[b[3],b[4],b[5]]};
}

// Hull arena (engine-schema HullArena): cook baked faces, recentre, insert.
export class HullArena{
 constructor({maxVertices}={}){this.rust=new RustArena(maxVertices)}
 // topology: {sizes, indices}; returns collider fields {params, localPos} and the hull's vertex count.
 append(vertices,topology,{sourceScale}={}){
  return this.rust.appendHull(Float64Array.from(vertices.flat()),Uint32Array.from(topology.sizes),Uint32Array.from(topology.indices),sourceScale);
 }
 release(base){return this.rust.release(base)}
 refs(base){return this.rust.refs(base)}
 takeDirty(){const d=this.rust.takeDirty();return Array.from({length:d.length/2},(_,i)=>[d[2*i],d[2*i+1]])}
 markAllDirty(){this.rust.markAllDirty()}
 get data(){return this.rust.data}
 get records(){return this.rust.records}
 get freeRecords(){return this.rust.freeRecords}
 get maxVertices(){return this.rust.maxVertices}
}
export const appendHull=(arena,vertices,topology,options)=>arena.append(vertices,topology,options);

// Decoder mirroring the Slang accessors (tests, CPU oracles).
export function hullView(data,base){
 const h=rustHullView(data,base);
 return {...h,vertex:i=>h.vertices[i],plane:k=>h.planes[k],face:k=>h.faces[k],edge:k=>h.edges[k],adjacency:i=>h.adjacency[i]};
}
