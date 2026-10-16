import {add,scale,sub,dot,cross,norm} from './geometry.mjs';
const shader=`
struct Body {p:vec4f,q:vec4f,invI:vec4f,v:vec4f,w:vec4f,initialP:vec4f,initialQ:vec4f};
@group(0) @binding(0) var<storage,read> bodies:array<Body>;
@group(0) @binding(1) var<uniform> vp:mat4x4f;
struct Out {@builtin(position) p:vec4f,@location(0) n:vec3f,@location(1) color:vec3f,@location(2) world:vec3f};
fn rotate(q:vec4f,v:vec3f)->vec3f{return v+2.*cross(q.xyz,cross(q.xyz,v)+q.w*v);}
@vertex fn vs(@location(0) p:vec3f,@location(1) n:vec3f,@location(2) color:vec3f,@location(3) id:f32)->Out{
 let body=bodies[u32(id)];let world=body.p.xyz+rotate(body.q,p);var o:Out;
 o.p=vp*vec4f(world,1.);o.n=rotate(body.q,n);o.color=color;o.world=world;return o;
}
@fragment fn fs(o:Out)->@location(0) vec4f{
 let light=.38+.62*max(0.,dot(normalize(o.n),normalize(vec3f(-.5,1.,.7))));
 return vec4f(pow(o.color*light,vec3f(.8)),1.);
}
@fragment fn ground(o:Out)->@location(0) vec4f{
 let coord=o.world.xz;let grid=abs(fract(coord-.5)-.5)/max(fwidth(coord),vec2f(.0001));
 let line=1.-min(min(grid.x,grid.y),1.);return vec4f(mix(vec3f(.075,.1,.125),vec3f(.13,.18,.21),line*.55),1.);
}
@fragment fn wire(o:Out)->@location(0) vec4f{return vec4f(o.color,1.);}
`;
function matrix(eye,target,aspect,near,far){
 const z=norm(sub(eye,target)),x=norm(cross([0,1,0],z)),y=cross(z,x);
 const view=[x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1];
 const f=1/Math.tan(.7/2),proj=[f/aspect,0,0,0,0,f,0,0,0,0,far/(near-far),-1,0,0,near*far/(near-far),0];
 return new Float32Array(Array.from({length:16},(_,i)=>[0,1,2,3].reduce((s,k)=>s+proj[k*4+i%4]*view[Math.floor(i/4)*4+k],0)));
}
export async function createRenderer(device,canvas,simulation,scene,onShoot=()=>{}){
 const context=canvas.getContext('webgpu'),format=navigator.gpu.getPreferredCanvasFormat();context.configure({device,format,alphaMode:'opaque'});
 const module=device.createShaderModule({code:shader});const info=await module.getCompilationInfo();if(info.messages.some(m=>m.type==='error'))throw new Error(info.messages.map(m=>m.message).join('\n'));
 const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:'read-only-storage'}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:'uniform'}}]});
 const uniform=device.createBuffer({size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
 const bind=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:simulation.bodyBuffer}},{binding:1,resource:{buffer:uniform}}]});
 const buffers=[{arrayStride:40,attributes:[{shaderLocation:0,offset:0,format:'float32x3'},{shaderLocation:1,offset:12,format:'float32x3'},{shaderLocation:2,offset:24,format:'float32x3'},{shaderLocation:3,offset:36,format:'float32'}]}];
 const pipeline=entry=>device.createRenderPipeline({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),vertex:{module,entryPoint:'vs',buffers},fragment:{module,entryPoint:entry,targets:[{format}]},primitive:{topology:entry==='wire'?'line-list':'triangle-list',cullMode:'none'},depthStencil:{format:'depth24plus',depthWriteEnabled:entry!=='wire',depthCompare:entry==='wire'?'always':'less'}});
 const solid=pipeline('fs'),ground=pipeline('ground'),wire=pipeline('wire');
 const make=data=>{const b=device.createBuffer({size:Math.max(4,data.byteLength),usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});if(data.length)device.queue.writeBuffer(b,0,data);return b;};
 const projectiles=make(scene.projectileVertices);
 const mesh=make(scene.vertices),lines=make(scene.lines),g=[];
 const size=Math.max(20,scene.extent*8);for(const p of [[-size,0,-size],[-size,0,size],[size,0,size],[-size,0,-size],[size,0,size],[size,0,-size]])g.push(...p,0,1,0,.1,.15,.18,0);
 const floor=make(new Float32Array(g));let depth,width=0,height=0,yaw=.25,pitch=.22,distance=scene.extent*1.9,drag;
 function camera(){
  const aspect=canvas.clientWidth/canvas.clientHeight,fitted=distance/Math.min(1,aspect),target=scene.center;
  const eye=[target[0]+Math.sin(yaw)*Math.cos(pitch)*fitted,target[1]+Math.sin(pitch)*fitted,target[2]+Math.cos(yaw)*Math.cos(pitch)*fitted];
  return {eye,target,aspect};
 }
 function aim(clientX,clientY){
  const {eye,target,aspect}=camera(),rect=canvas.getBoundingClientRect();
  const forward=norm(sub(target,eye)),right=norm(cross(forward,[0,1,0])),up=cross(right,forward);
  const x=(2*(clientX-rect.left)/rect.width-1)*aspect*Math.tan(.7/2),y=(1-2*(clientY-rect.top)/rect.height)*Math.tan(.7/2);
  const direction=norm(add(forward,add(scale(right,x),scale(up,y))));
  return {origin:add(eye,scale(direction,scene.projectileRadius*2)),direction};
 }
 const down=e=>{if(e.button!==0)return;drag={start:[e.clientX,e.clientY],last:[e.clientX,e.clientY],moved:false};canvas.setPointerCapture(e.pointerId)};
 const move=e=>{if(!drag)return;if(Math.hypot(e.clientX-drag.start[0],e.clientY-drag.start[1])>5)drag.moved=true;
  if(drag.moved){yaw-=(e.clientX-drag.last[0])*.006;pitch=Math.max(-.1,Math.min(1.3,pitch+(e.clientY-drag.last[1])*.006))}drag.last=[e.clientX,e.clientY]};
 const up=e=>{if(!drag)return;const shoot=!drag.moved;drag=null;if(shoot)onShoot(aim(e.clientX,e.clientY))};
 const cancel=()=>drag=null;
 const wheel=e=>{e.preventDefault();distance=Math.max(scene.extent*.35,Math.min(scene.extent*8,distance*Math.exp(e.deltaY*.001)))};
 canvas.addEventListener('pointerdown',down);canvas.addEventListener('pointermove',move);canvas.addEventListener('pointerup',up);canvas.addEventListener('pointercancel',cancel);canvas.addEventListener('wheel',wheel,{passive:false});
 return {aim,draw(showBonds){
  const dpr=Math.min(devicePixelRatio,2),w=Math.max(1,Math.round(canvas.clientWidth*dpr)),h=Math.max(1,Math.round(canvas.clientHeight*dpr));
  if(w!==width||h!==height){width=canvas.width=w;height=canvas.height=h;depth?.destroy();depth=device.createTexture({size:[w,h],format:'depth24plus',usage:GPUTextureUsage.RENDER_ATTACHMENT});}
  const {eye,target}=camera();
  device.queue.writeBuffer(uniform,0,matrix(eye,target,w/h,.02,Math.max(500,scene.extent*50)));
 const enc=device.createCommandEncoder(),pass=enc.beginRenderPass({colorAttachments:[{view:context.getCurrentTexture().createView(),clearValue:[.035,.055,.075,1],loadOp:'clear',storeOp:'store'}],depthStencilAttachment:{view:depth.createView(),depthClearValue:1,depthLoadOp:'clear',depthStoreOp:'store'}});
  pass.setBindGroup(0,bind);pass.setPipeline(ground);pass.setVertexBuffer(0,floor);pass.draw(6);pass.setPipeline(solid);pass.setVertexBuffer(0,mesh);pass.draw(scene.vertices.length/10);
  if(simulation.shots){pass.setVertexBuffer(0,projectiles);pass.draw(simulation.shots*scene.projectileVertices.length/scene.projectileCount/10)}
  if(showBonds&&scene.lines.length){pass.setPipeline(wire);pass.setVertexBuffer(0,lines);pass.draw(scene.lines.length/10)}pass.end();device.queue.submit([enc.finish()]);
 },dispose(){for(const b of [uniform,mesh,lines,floor,projectiles])b.destroy();depth?.destroy();canvas.removeEventListener('pointerdown',down);canvas.removeEventListener('pointermove',move);canvas.removeEventListener('pointerup',up);canvas.removeEventListener('pointercancel',cancel);canvas.removeEventListener('wheel',wheel)}};
}
