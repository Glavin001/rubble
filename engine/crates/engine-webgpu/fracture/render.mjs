import {Orbit,IndexedMesh,floorVertices,sceneHandle} from './wasm.mjs';
import {attachCameraInput} from './camera-input.mjs';
// WebGPU draw of the fracture lab. The shader is the shared Slang draw shader
// (slang/render/fracture_draw.slang, exported to WGSL), reading live poses
// from the body buffer; the camera (orbit, first-person walk and fly), shot
// rays, floor and indexed mesh come from engine-fracture (view.rs). The Metal viewer draws the same way.
export async function createRenderer(device,canvas,simulation,scene,onShoot=()=>{}){
 const context=canvas.getContext('webgpu'),format=navigator.gpu.getPreferredCanvasFormat();context.configure({device,format,alphaMode:'opaque'});
 const response=await fetch('../generated/wgsl/fracture_draw.wgsl');
 if(!response.ok)throw new Error('Missing fracture_draw.wgsl; run the shader export commands in fracture/README.md');
 const module=device.createShaderModule({code:await response.text()});const info=await module.getCompilationInfo();if(info.messages.some(m=>m.type==='error'))throw new Error(info.messages.map(m=>m.message).join('\n'));
 const vertex=GPUShaderStage.VERTEX;
 const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:vertex,buffer:{type:'read-only-storage'}},{binding:1,visibility:vertex,buffer:{type:'read-only-storage'}},{binding:2,visibility:vertex,buffer:{type:'uniform'}},{binding:3,visibility:vertex,buffer:{type:'read-only-storage'}}]});
 // View: column-major view-projection, then instance stride, colour mode and joints per copy.
 const uniform=device.createBuffer({size:80,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
 const pipeline=(entry,decal=false)=>device.createRenderPipeline({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),vertex:{module,entryPoint:'vs'},fragment:{module,entryPoint:entry,targets:[{format}]},primitive:{topology:entry==='wire'?'line-list':'triangle-list',cullMode:'none'},
  // Road decals lie millimetres above the ground: a depth bias keeps them in
  // front of it at any distance (the far plane grows with the scene).
  depthStencil:{format:'depth24plus',depthWriteEnabled:entry!=='wire',depthCompare:entry==='wire'?'always':decal?'less-equal':'less',...(decal?{depthBias:-16,depthBiasSlopeScale:-2}:{})}});
 const solid=pipeline('fs'),ground=pipeline('ground'),wire=pipeline('wire'),decal=pipeline('fs',true);
 // The engine's joints: a bond line hides once its joint breaks (joints per copy 0: the
 // scene has none, and the line hides once its pieces part).
 const joints=simulation.jointBuffer??device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE});
 const jointsPerCopy=simulation.jointBuffer&&scene.jointCount?scene.jointCount/(scene.copies||1):0;
 // Render streams (10 floats per vertex) are storage buffers the shader pulls from.
 const streams=[];
 const stream=data=>{const buffer=device.createBuffer({size:Math.max(16,data.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});if(data.length)device.queue.writeBuffer(buffer,0,data);streams.push(buffer);
  return {count:data.length/10,group:device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:simulation.renderBodyBuffer??simulation.bodyBuffer}},{binding:1,resource:{buffer}},{binding:2,resource:{buffer:uniform}},{binding:3,resource:{buffer:joints}}]})}};
 const indexed=new IndexedMesh(scene.vertices),indices=indexed.index();
 const mesh=stream(indexed.vertices()),projectiles=stream(scene.projectileVertices),props=stream(scene.propVertices??new Float32Array(0)),lines=stream(scene.lines),roads=stream(scene.roadVertices),floor=stream(floorVertices(scene.extent));
 // Directed-debris zone outlines (engine-fracture guidance.rs), drawn over the scene.
 let zones=stream(new Float32Array(0));
 indexed.free();
 const index=device.createBuffer({size:Math.max(4,indices.byteLength),usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(index,0,indices);
 const orbit=new Orbit(sceneHandle(scene));let depth,width=0,height=0;
 const aspect=()=>canvas.clientWidth/Math.max(1,canvas.clientHeight);
 const shotAt=(u,v)=>orbit.aim(u,v,aspect(),scene.projectileRadius);
 function aim(clientX,clientY){
  const rect=canvas.getBoundingClientRect();
  return shotAt((clientX-rect.left)/rect.width,(clientY-rect.top)/rect.height);
 }
 const input=attachCameraInput(canvas,orbit,(u,v)=>onShoot(shotAt(u,v)));
 // Ball trails (setTrails): line segments, 8 floats per vertex (xyz, pad, rgba),
 // drawn four times a pixel apart so they read at any size.
 const trailModule=device.createShaderModule({code:`struct V{vp:mat4x4f,px:vec4f};@group(0)@binding(0)var<uniform> v:V;@group(0)@binding(1)var<storage,read> t:array<vec4f>;
struct O{@builtin(position)p:vec4f,@location(0)c:vec4f};
@vertex fn vs(@builtin(vertex_index)i:u32,@builtin(instance_index)k:u32)->O{var o:O;o.p=v.vp*vec4f(t[2u*i].xyz,1.);let off=vec2f(f32(k&1u),f32(k>>1u))-.5;o.p=vec4f(o.p.xy+off*v.px.xy*2.*o.p.w,o.p.zw);o.c=t[2u*i+1u];return o;}
@fragment fn fs(i:O)->@location(0)vec4f{return i.c;}`});
 const trailPipe=device.createRenderPipeline({layout:'auto',vertex:{module:trailModule,entryPoint:'vs'},fragment:{module:trailModule,entryPoint:'fs',targets:[{format}]},primitive:{topology:'line-list'},depthStencil:{format:'depth24plus',depthWriteEnabled:false,depthCompare:'less'}});
 const trailView=device.createBuffer({size:80,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
 let trailBuf=null,trailGroup=null,trailVerts=0;
 // A mesh the page replaces at will (setDynamic): render vertices, 10 floats each.
 let dynBuf=null,dynGroup=null,dynVerts=0;
 let stamps=null,colorMode=0;
 // GPU time of a sampled frame's render pass, about four times a second (stats
 // panel). One sample in flight; nothing waits on its readback.
 const frameTimer=device.features.has('timestamp-query')?{query:device.createQuerySet({type:'timestamp',count:2}),
  resolve:device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),
  read:device.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),busy:false,last:0,ms:null}:null;
 return {aim,
 // Line-list vertices outlining the guidance zones (empty: none).
 setZones(vertices){zones=stream(vertices)},
 // Trail segments for the next frames (8 floats per vertex: xyz, pad, rgba).
 setTrails(data){
  trailVerts=data.length/8;if(!trailVerts)return;
  if(!trailBuf||trailBuf.size<data.byteLength){trailBuf?.destroy();trailBuf=device.createBuffer({size:Math.max(65536,data.byteLength*2),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
   trailGroup=device.createBindGroup({layout:trailPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:trailView}},{binding:1,resource:{buffer:trailBuf}}]})}
  device.queue.writeBuffer(trailBuf,0,data);
 },
 // Render vertices (10 floats each, body-local) drawn with the scene's meshes until
 // replaced: bodies the page adds while it runs (a character, balls, debris).
 setDynamic(data){
  dynVerts=data.length/10;if(!dynVerts)return;
  if(!dynBuf||dynBuf.size<data.byteLength){dynBuf?.destroy();dynBuf=device.createBuffer({size:Math.max(65536,data.byteLength*2),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
   dynGroup=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:simulation.renderBodyBuffer??simulation.bodyBuffer}},{binding:1,resource:{buffer:dynBuf}},{binding:2,resource:{buffer:uniform}},{binding:3,resource:{buffer:joints}}]})}
  device.queue.writeBuffer(dynBuf,0,data);
 },
 // A scripted camera (a take's shot): fly mode, the eye at `eye` looking at `target`.
 lookAt(eye,target){orbit.lookAt(eye[0],eye[1],eye[2],target[0],target[1],target[2]);input.modeChanged()},
 // The camera's view mode (0 orbit, 1 first-person walk, 2 fly: engine-fracture ViewMode).
 get viewMode(){return orbit.mode},
 set viewMode(mode){orbit.setMode(mode|0,aspect());input.modeChanged()},
 // GPU ms of the latest sampled frame's render pass, or null if unsupported.
 get gpuFrameMs(){return frameTimer?.ms??null},
 // 0: materials; 1: sleep state (awake orange, asleep blue, static grey).
 set colorMode(mode){colorMode=mode|0},
 // GPU time of one frame's render pass (ms), via timestamps.
 async profileDraw(showBonds=false){
  if(!device.features.has('timestamp-query'))return null;
  const q=device.createQuerySet({type:'timestamp',count:2}),res=device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),rd=device.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  try{stamps={querySet:q,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1};this.draw(showBonds);stamps=null;
   const enc=device.createCommandEncoder();enc.resolveQuerySet(q,0,2,res,0);enc.copyBufferToBuffer(res,0,rd,0,16);device.queue.submit([enc.finish()]);
   await rd.mapAsync(GPUMapMode.READ);const t=new BigUint64Array(rd.getMappedRange().slice(0));return Number(t[1]-t[0])/1e6}
  finally{stamps=null;q.destroy();res.destroy();rd.destroy()}
 },
 draw(showBonds){
  const dpr=Math.min(devicePixelRatio,2),w=Math.max(1,Math.round(canvas.clientWidth*dpr)),h=Math.max(1,Math.round(canvas.clientHeight*dpr));
  if(w!==width||h!==height){width=canvas.width=w;height=canvas.height=h;depth?.destroy();depth=device.createTexture({size:[w,h],format:'depth24plus',usage:GPUTextureUsage.RENDER_ATTACHMENT});}
  input.update();
  device.queue.writeBuffer(uniform,0,orbit.viewProj(w/h));device.queue.writeBuffer(uniform,64,new Uint32Array([scene.instanceStride||0,colorMode,jointsPerCopy,0]));
  const timed=!stamps&&frameTimer&&!frameTimer.busy&&performance.now()-frameTimer.last>250;
  const writes=stamps??(timed?{querySet:frameTimer.query,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}:null);
  const enc=device.createCommandEncoder(),pass=enc.beginRenderPass({colorAttachments:[{view:context.getCurrentTexture().createView(),clearValue:[.035,.055,.075,1],loadOp:'clear',storeOp:'store'}],depthStencilAttachment:{view:depth.createView(),depthClearValue:1,depthLoadOp:'clear',depthStoreOp:'store'},...(writes?{timestampWrites:writes}:{})});
  pass.setPipeline(ground);pass.setBindGroup(0,floor.group);pass.draw(floor.count);
  if(roads.count){pass.setPipeline(decal);pass.setBindGroup(0,roads.group);pass.draw(roads.count)}
  pass.setPipeline(solid);pass.setBindGroup(0,mesh.group);pass.setIndexBuffer(index,'uint32');pass.drawIndexed(indices.length,scene.copies||1);
  if(dynVerts){pass.setBindGroup(0,dynGroup);pass.draw(dynVerts)}
  // The newest projectileCount balls (a shot past them reuses the oldest).
  // Street props (parked cars, pedestrians): their own bodies, drawn once.
  if(props.count){pass.setBindGroup(0,props.group);pass.draw(props.count)}
  if(simulation.shots){pass.setBindGroup(0,projectiles.group);pass.draw(Math.min(simulation.shots,scene.projectileCount)*projectiles.count/scene.projectileCount)}
  if(trailVerts){const vp=orbit.viewProj(w/h);device.queue.writeBuffer(trailView,0,vp);device.queue.writeBuffer(trailView,64,new Float32Array([1/w,1/h,0,0]));pass.setPipeline(trailPipe);pass.setBindGroup(0,trailGroup);pass.draw(trailVerts,4)}
  if(showBonds&&lines.count){pass.setPipeline(wire);pass.setBindGroup(0,lines.group);pass.draw(lines.count,scene.copies||1)}
  if(zones.count){pass.setPipeline(wire);pass.setBindGroup(0,zones.group);pass.draw(zones.count)}pass.end();
  if(timed){frameTimer.busy=true;frameTimer.last=performance.now();enc.resolveQuerySet(frameTimer.query,0,2,frameTimer.resolve,0);enc.copyBufferToBuffer(frameTimer.resolve,0,frameTimer.read,0,16)}
  device.queue.submit([enc.finish()]);
  if(timed)frameTimer.read.mapAsync(GPUMapMode.READ).then(()=>{const t=new BigUint64Array(frameTimer.read.getMappedRange());frameTimer.ms=Number(t[1]-t[0])/1e6;frameTimer.read.unmap()}).catch(()=>{}).finally(()=>{frameTimer.busy=false});
 },dispose(){input.dispose();for(const b of [uniform,index,trailView,...streams])b.destroy();trailBuf?.destroy();if(joints!==simulation.jointBuffer)joints.destroy();orbit.free();if(frameTimer){frameTimer.query.destroy();frameTimer.resolve.destroy();frameTimer.read.destroy()}depth?.destroy();dynBuf?.destroy()}};
}
