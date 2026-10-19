// Exact rigid rotations, smooth pose interpolation, and instanced WebGL boxes.
export function quaternion(m) {
 const t=m[0][0]+m[1][1]+m[2][2]; let q;
 if(t>0){const s=2*Math.sqrt(t+1);q=[(m[2][1]-m[1][2])/s,(m[0][2]-m[2][0])/s,(m[1][0]-m[0][1])/s,s/4]}
 else {const i=m[0][0]>m[1][1]?(m[0][0]>m[2][2]?0:2):(m[1][1]>m[2][2]?1:2),j=(i+1)%3,k=(i+2)%3,s=2*Math.sqrt(1+m[i][i]-m[j][j]-m[k][k]);q=[0,0,0,0];q[i]=s/4;q[j]=(m[j][i]+m[i][j])/s;q[k]=(m[k][i]+m[i][k])/s;q[3]=(m[k][j]-m[j][k])/s;}
 return q;
}
export function rotation(a,b,t){
 let qa=quaternion(a),qb=quaternion(b),d=qa.reduce((s,x,i)=>s+x*qb[i],0);
 if(d<0){qb=qb.map(x=>-x);d=-d}
 let u=1-t,v=t;
 if(d<.9995){const angle=Math.acos(Math.min(1,d)),s=Math.sin(angle);u=Math.sin((1-t)*angle)/s;v=Math.sin(t*angle)/s}
 let q=qa.map((x,i)=>u*x+v*qb[i]),len=Math.hypot(...q);q=q.map(x=>x/len);
 const [x,y,z,w]=q;
 return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]];
}
export const mix=(a,b,t)=>a.map((v,k)=>v+(b[k]-v)*t);
export function interval(playhead,count){const i=Math.min(count-1,Math.max(0,Math.floor(playhead)));return {i,alpha:Math.max(0,Math.min(1,playhead-i))}}
export const identity=[[1,0,0],[0,1,0],[0,0,1]];
export class Scene {
 constructor(canvas){
  this.canvas=canvas;const gl=this.gl=canvas.getContext('webgl2',{antialias:true});if(!gl)throw Error('This playback needs WebGL 2.');
  const vs=`#version 300 es
  layout(location=0) in vec3 position;layout(location=1) in vec3 normal;
  layout(location=2) in vec3 center;layout(location=3) in vec3 row0;layout(location=4) in vec3 row1;layout(location=5) in vec3 row2;layout(location=6) in vec3 halfSize;layout(location=7) in vec3 color;layout(location=8) in float broken;
  uniform vec3 target,right,up,eye;uniform vec3 scale;out vec3 tint;out vec3 norm;out vec3 local;out vec3 faceNormal;flat out float isBroken;
  void main(){vec3 v=position*halfSize;vec3 p=center+vec3(dot(row0,v),dot(row1,v),dot(row2,v))-target;gl_Position=vec4(dot(p,right)*scale.x,dot(p,up)*scale.y,-dot(p,eye)*scale.z,1.);tint=color;local=position;faceNormal=normal;isBroken=broken;norm=vec3(dot(row0,normal),dot(row1,normal),dot(row2,normal));}`;
  const fs=`#version 300 es
  precision highp float;in vec3 tint;in vec3 norm;in vec3 local;in vec3 faceNormal;flat in float isBroken;out vec4 outputColor;
  void main(){float light=.42+.58*max(0.,dot(normalize(norm),normalize(vec3(-.4,-.5,1.))));vec3 e=abs(local)*(vec3(1.)-abs(faceNormal));float edge=smoothstep(.96,.995,max(e.x,max(e.y,e.z)));vec3 fill=tint*light;outputColor=vec4(mix(fill,isBroken>.5?vec3(1.,.94,.82):fill*.55,edge),1.);}`;
  const shader=(type,src)=>{const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s};
  const p=this.program=gl.createProgram();gl.attachShader(p,shader(gl.VERTEX_SHADER,vs));gl.attachShader(p,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));
  const verts=[];
  for(let axis=0;axis<3;axis++)for(const sign of [-1,1]){const j=(axis+1)%3,k=(axis+2)%3,n=[0,0,0];n[axis]=sign;for(const [a,b] of [[-1,-1],[1,-1],[1,1],[-1,-1],[1,1],[-1,1]]){const v=[0,0,0];v[axis]=sign;v[j]=a;v[k]=b;verts.push(...v,...n)}}
  this.vao=gl.createVertexArray();gl.bindVertexArray(this.vao);gl.bindBuffer(gl.ARRAY_BUFFER,gl.createBuffer());gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(verts),gl.STATIC_DRAW);
  for(let i=0;i<2;i++){gl.enableVertexAttribArray(i);gl.vertexAttribPointer(i,3,gl.FLOAT,false,24,i*12)}
  this.instances=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,this.instances);
  for(let i=2;i<8;i++){gl.enableVertexAttribArray(i);gl.vertexAttribPointer(i,3,gl.FLOAT,false,76,(i-2)*12);gl.vertexAttribDivisor(i,1)}
  gl.enableVertexAttribArray(8);gl.vertexAttribPointer(8,1,gl.FLOAT,false,76,72);gl.vertexAttribDivisor(8,1);
  const sphere=[];const point=(a,b)=>[Math.cos(a)*Math.sin(b),Math.sin(a)*Math.sin(b),Math.cos(b)];
  for(let a=0;a<32;a++)for(let b=0;b<16;b++)for(const [x,y] of [[a,b],[a+1,b],[a+1,b+1],[a,b],[a+1,b+1],[a,b+1]]){const p=point(x*Math.PI/16,y*Math.PI/16);sphere.push(...p,...p)}
  this.sphereCount=sphere.length/6;this.sphereVao=gl.createVertexArray();gl.bindVertexArray(this.sphereVao);gl.bindBuffer(gl.ARRAY_BUFFER,gl.createBuffer());gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(sphere),gl.STATIC_DRAW);
  for(let i=0;i<2;i++){gl.enableVertexAttribArray(i);gl.vertexAttribPointer(i,3,gl.FLOAT,false,24,i*12)}
  gl.bindBuffer(gl.ARRAY_BUFFER,this.instances);
  for(let i=2;i<8;i++){gl.enableVertexAttribArray(i);gl.vertexAttribPointer(i,3,gl.FLOAT,false,76,(i-2)*12);gl.vertexAttribDivisor(i,1)}
  gl.enableVertexAttribArray(8);gl.vertexAttribPointer(8,1,gl.FLOAT,false,76,72);gl.vertexAttribDivisor(8,1);
  this.uniforms=Object.fromEntries(['target','right','up','eye','scale'].map(k=>[k,gl.getUniformLocation(p,k)]));
 }
 draw(boxes,camera){
  const gl=this.gl,c=this.canvas,dpr=Math.min(devicePixelRatio||1,2),w=Math.round(c.clientWidth*dpr),h=Math.round(c.clientHeight*dpr);
  if(c.width!==w||c.height!==h){c.width=w;c.height=h}
  gl.viewport(0,0,w,h);gl.clearColor(.065,.09,.12,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);gl.useProgram(this.program);gl.bindVertexArray(this.vao);
  const {yaw,pitch,center,radius,zoom}=camera,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),s=zoom/(radius*1.35);
  const uniforms={target:center,right:[cy,sy,0],up:[sy*sp,-cy*sp,cp],eye:[-sy*cp,cy*cp,sp],scale:[s/Math.max(1,w/h),s*Math.min(1,w/h),1/(radius*40)]};
  for(const [k,v] of Object.entries(uniforms))gl.uniform3fv(this.uniforms[k],v);
  for(const sphere of [false,true]){const group=boxes.filter(b=>!!b.sphere===sphere);if(!group.length)continue;
   const data=new Float32Array(group.length*19);let at=0;
   for(const b of group){data.set([...b.center,...b.rotation.flat(),...b.half,...b.color,b.broken?1:0],at);at+=19}
   gl.bindVertexArray(sphere?this.sphereVao:this.vao);gl.bindBuffer(gl.ARRAY_BUFFER,this.instances);gl.bufferData(gl.ARRAY_BUFFER,data,gl.DYNAMIC_DRAW);gl.drawArraysInstanced(gl.TRIANGLES,0,sphere?this.sphereCount:36,group.length);
  }
 }
}
