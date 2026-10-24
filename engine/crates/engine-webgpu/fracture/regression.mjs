import {add,sub,norm} from './vec.mjs';
import {colliderVertices} from './wasm.mjs';
// Run in this demo's browser console, or import from a browser automation test.
export async function runFractureRegression(demo=window.__FRACTURE_DEMO__) {
 const require=(ok,message)=>{if(!ok)throw new Error(`Fracture regression: ${message}`)};
 require(demo.ready,'demo is not ready');
 require(demo.scene.selected.length===23,'select the bundled arch at level 0');
 await demo.reset();const before=await demo.snapshot();
 await demo.stepFrames(120);const rest=await demo.snapshot();
 for(let i=1;i<demo.scene.projectileStart;i++)for(let k=0;k<3;k++)require(rest[i*28+k]===before[i*28+k],'arch moved before impact');
 // Aim a real sphere at the crown from in front of the arch.
 const target=demo.scene.bodies.slice(1,demo.scene.projectileStart).reduce((a,b)=>a.p[1]>b.p[1]?a:b).p;
 const origin=add(target,[0,0,3]);const projectile=await demo.shootRay(origin,norm(sub(target,origin)));
 let peakContacts=0,peakAwake=0;
 for(let n=0;n<12;n++){const c=await demo.stepFrames(60);require(c.overflow===0,'GPU capacity overflow');peakContacts=Math.max(peakContacts,c.contact);peakAwake=Math.max(peakAwake,c.awake)}
 const after=await demo.snapshot();let minHullY=Infinity,maxDisplacement=0;
 for(let i=1;i<demo.scene.projectileStart;i++){
  const o=i*28;for(const k of [0,1,2,4,5,6,7,12,13,14,16,17,18])require(Number.isFinite(after[o+k]),`nonfinite state in body ${i}`);
  require(Math.abs(Math.hypot(...after.slice(o+4,o+8))-1)<.001,`invalid orientation in body ${i}`);
  maxDisplacement=Math.max(maxDisplacement,Math.hypot(...sub(Array.from(after.slice(o,o+3)),Array.from(before.slice(o,o+3)))));
 }
 for(const c of demo.scene.colliders.filter(c=>c.shape===3)){
  const o=c.body*28,body={q:Array.from(after.slice(o+4,o+8)),p:Array.from(after.slice(o,o+3))};
  for(const v of colliderVertices(demo.scene,c,body))minHullY=Math.min(minHullY,v[1]);
 }
 require(peakContacts>0&&peakAwake>1,'projectile did not wake fragments');require(maxDisplacement>.1,'projectile did not move the arch');
 require(minHullY>-.025,`hulls penetrated the floor: ${minHullY} m`);
 require(after[projectile*28+1]>=demo.scene.projectileRadius-.025,'sphere fell through floor');
 const result={passed:true,steps:demo.simulation.steps,peakContacts,peakAwake,maxDisplacement,minHullY,counters:demo.simulation.counters};
 await demo.reset();const reset=await demo.snapshot();require(reset.every((x,i)=>x===before[i]),'reset did not restore every body field');require(demo.simulation.shots===0,'reset did not reload projectiles');
 return result;
}
// A 300 m/s shot at the arch crown (2.5 m a tick at 120 Hz), without and with
// CCD balls (engine-viewer fracture_regression a_ccd_shot_does_not_tunnel_through_the_arch):
// the plain ball passes the arch between ticks, the CCD ball strikes the crown and
// knocks it out (time-of-impact contact, then the inelastic momentum exchange).
export async function runCcdShotRegression(demo=window.__FRACTURE_DEMO__,{speed=300,ticks=30}={}) {
 const require=(ok,message)=>{if(!ok)throw new Error(`CCD shot regression: ${message}`)};
 require(demo.ready,'demo is not ready');
 const sim=()=>demo.simulation,start=demo.scene.projectileStart,bodies=demo.scene.bodies.slice(1,start);
 const crown=bodies.reduce((a,b)=>a.p[1]>b.p[1]?a:b).p,back=Math.min(...bodies.map(b=>b.p[2]));
 const runs={};
 for(const ccd of [false,true]){
  await demo.reset();await demo.stepFrames(10);const before=await demo.snapshot();
  sim().setProjectileCcd(ccd);sim().setProjectileSpeedScale(speed/sim().shotSpeed);
  const id=await demo.shootRay(add(crown,[0,0,3]),[0,0,-1]);let minZ=Infinity;
  for(let t=0;t<ticks;t++){const c=await demo.stepFrames(1);require(c.overflow===0,'GPU capacity overflow');minZ=Math.min(minZ,(await demo.snapshot())[id*28+2])}
  const after=await demo.snapshot();let moved=0;
  for(let i=1;i<start;i++)moved=Math.max(moved,Math.hypot(...sub(Array.from(after.slice(i*28,i*28+3)),Array.from(before.slice(i*28,i*28+3)))));
  runs[ccd?'ccd':'control']={minZ,moved,finite:after.slice(id*28,id*28+28).every(Number.isFinite)};
 }
 sim().setProjectileSpeedScale(1);
 require(runs.control.minZ<back-2,`the plain ball should pass the arch (z ${runs.control.minZ})`);
 require(runs.ccd.finite,'the CCD ball has a non-finite state');
 require(runs.ccd.moved>5&&runs.ccd.moved>20*runs.control.moved,`the CCD ball should knock the crown out (${runs.ccd.moved} vs ${runs.control.moved})`);
 return {passed:true,back,...runs};
}

// Phase-through acceptance in the browser lab (engine-viewer tests/fracture_tower_ccd.rs):
// CCD balls at the loaded structure from four sides at three heights, four corners and
// two plunging roof shots, at each of `scales` × the scene's shot speed. A phase-through
// is exact: in some tick the ball's centre path (its pose before → after the tick)
// meets a piece that did not move during that tick (GJK distance between the segment
// and the piece's hull is zero). Every shot must have none.
const dot3=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const mul3=(a,s)=>[a[0]*s,a[1]*s,a[2]*s];
function simplexClosest(pts){
 let best=null;
 for(let mask=1;mask<1<<pts.length;mask++){
  const idx=pts.map((_,i)=>i).filter(i=>mask&(1<<i)),p0=pts[idx[0]],e=idx.slice(1).map(i=>sub(pts[i],p0)),m=e.length;
  const a=e.map(r=>e.map(c=>dot3(r,c))),b=e.map(r=>-dot3(r,p0));
  let ok=true;
  for(let col=0;col<m&&ok;col++){
   let piv=col;for(let r=col+1;r<m;r++)if(Math.abs(a[r][col])>Math.abs(a[piv][col]))piv=r;
   if(Math.abs(a[piv][col])<1e-14){ok=false;break}
   [a[col],a[piv]]=[a[piv],a[col]];[b[col],b[piv]]=[b[piv],b[col]];
   for(let r=0;r<m;r++)if(r!==col){const f=a[r][col]/a[col][col];for(let c=0;c<m;c++)a[r][c]-=f*a[col][c];b[r]-=f*b[col]}
  }
  if(!ok)continue;
  const w=b.map((x,r)=>x/a[r][r]),w0=1-w.reduce((s,x)=>s+x,0);
  if(w0<-1e-12||w.some(x=>x<-1e-12))continue;
  let p=mul3(p0,w0);w.forEach((x,r)=>{p=add(p,mul3(pts[idx[r+1]],x))});
  const d=dot3(p,p);if(!best||d<best.d)best={d,p,idx};
 }
 return best;
}
const qrot=(q,v)=>{const u=[q[0],q[1],q[2]],w=q[3],c=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];return add(add(mul3(u,2*dot3(u,v)),mul3(v,w*w-dot3(u,u))),mul3(c,2*w))};
const qmul=(a,b)=>[a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]];
// World vertices of a hull or box piece at a body pose.
function pieceVertices(scene,c,body){
 if(c.shape===3)return colliderVertices(scene,c,body);
 const centre=add(body.p,qrot(body.q,c.localPos??[0,0,0])),rot=qmul(body.q,c.localRot??[0,0,0,1]);
 return Array.from({length:8},(_,i)=>add(centre,qrot(rot,[i&1?c.params[0]:-c.params[0],i&2?c.params[1]:-c.params[1],i&4?c.params[2]:-c.params[2]])));
}
export function segmentHullDistance(s0,s1,verts){
 const support=d=>{let v=verts[0];for(const x of verts)if(dot3(x,d)>dot3(v,d))v=x;return sub(v,dot3(s0,d)<dot3(s1,d)?s0:s1)};
 let simplex=[support([1,0,0])],closest=simplex[0];
 for(let it=0;it<64;it++){
  const d2=dot3(closest,closest);if(d2<1e-18)return 0;
  const p=support(mul3(closest,-1));
  if(d2-dot3(closest,p)<=1e-12*Math.max(d2,1))return Math.sqrt(d2);
  simplex.push(p);const r=simplexClosest(simplex);simplex=r.idx.map(i=>simplex[i]);closest=r.p;
  if(simplex.length===4)return 0;
 }
 return Math.sqrt(dot3(closest,closest));
}
// ccd: false and a shot limit make the control (the detector must find phase-throughs).
export async function runPhaseThroughRegression(demo=window.__FRACTURE_DEMO__,{scales=[16,64],maxTicks=120,ccd=true,limit=Infinity,log=()=>{}}={}) {
 const require=(ok,message)=>{if(!ok)throw new Error(`Phase-through regression: ${message}`)};
 require(demo.ready,'demo is not ready');
 const scene=demo.scene,start=scene.projectileStart,sim=()=>demo.simulation,dt=scene.params.dt;
 const pieces=scene.colliders.map((c,i)=>({c,i})).filter(({c})=>(c.shape===3||c.shape===2)&&c.body>=1&&c.body<start);
 let lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
 for(const b of scene.bodies.slice(1,start))for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],b.p[k]);hi[k]=Math.max(hi[k],b.p[k])}
 // A little off the axis: piece grids have seams on it (a centre path along a seam only touches hulls).
 const c=[0,1,2].map(k=>(lo[k]+hi[k])/2+[.137,0,.071][k]),back=Math.max(hi[0]-lo[0],hi[2]-lo[2])/2+6,h=f=>lo[1]+f*(hi[1]-lo[1])+.053;
 const shots=[];
 for(const [side,d] of [['+x',[-1,0,0]],['-x',[1,0,0]],['+z',[0,0,-1]],['-z',[0,0,1]]])
  for(const [hn,f] of [['low',.12],['mid',.5],['high',.88]])shots.push({label:`${side} ${hn}`,origin:[c[0]-d[0]*back,h(f),c[2]-d[2]*back],dir:d});
 for(const [n,d0] of [['+x+z',[-1,0,-1]],['-x+z',[1,0,-1]],['+x-z',[-1,0,1]],['-x-z',[1,0,1]]]){const d=norm(d0);shots.push({label:`corner ${n}`,origin:[c[0]-d[0]*back*1.42,h(.5),c[2]-d[2]*back*1.42],dir:d})}
 for(const [n,d0] of [['roof +x',[-1,-1,0]],['roof -z',[0,-1,1]]]){const d=norm(d0);shots.push({label:n,origin:sub([c[0],hi[1],c[2]],mul3(d,back)),dir:d})}
 const pose=(s,b)=>({p:[s[b*28],s[b*28+1],s[b*28+2]],q:[s[b*28+4],s[b*28+5],s[b*28+6],s[b*28+7]]});
 const results=[];
 for(const scale of scales)for(const shot of shots.slice(0,limit)){
  await demo.reset();await demo.stepFrames(2);
  sim().setProjectileCcd(ccd);sim().setProjectileSpeedScale(scale);
  const speed=sim().shotSpeed*scale,ticks=Math.min(maxTicks,Math.max(30,Math.ceil((Math.max(hi[0]-lo[0],hi[1]-lo[1])+20)/(speed*dt))));
  const id=await demo.shootRay(shot.origin,shot.dir);
  let prev=await demo.snapshot(),phase=0,first=null,hit=false;
  for(let t=0;t<ticks;t++){
   const cnt=await demo.stepFrames(1);require(cnt.overflow===0,`${shot.label}: GPU capacity overflow`);
   const now=await demo.snapshot(),b0=pose(prev,id).p,b1=pose(now,id).p;
   for(const {c} of pieces){
    const P0=pose(prev,c.body),P1=pose(now,c.body);
    const moved=Math.hypot(...sub(P1.p,P0.p))>1e-3||Math.abs(dot3(P0.q,P1.q)+P0.q[3]*P1.q[3])<1-1e-6;
    if(moved){hit=true;continue}
    // Bounding-sphere prefilter (the hull's bound radius about its body).
    const seg=sub(b1,b0),w=sub(P0.p,b0),tt=Math.min(1,Math.max(0,dot3(w,seg)/Math.max(dot3(seg,seg),1e-20)));
    const bound=c.shape===2?Math.hypot(c.params[0],c.params[1],c.params[2]):c.params[3];
    if(Math.hypot(...sub(add(b0,mul3(seg,tt)),P0.p))>bound+Math.hypot(...c.localPos??[0,0,0])+1e-3)continue;
    if(segmentHullDistance(b0,b1,pieceVertices(scene,c,P0))<=1e-6){phase++;first??=`tick ${t}: ball ${b0.map(x=>x.toFixed(2))} -> ${b1.map(x=>x.toFixed(2))} crossed still piece ${c.body}`}
   }
   prev=now;
   const v=prev.slice(id*28+12,id*28+15);if(Math.hypot(...v)<.5)break;
  }
  const r={scale,speed,label:shot.label,phase,first,hit};results.push(r);
  log(`×${scale} (${speed.toFixed(0)} m/s) ${shot.label}: ${phase} phase-throughs${first?` — ${first}`:''}`);
 }
 sim().setProjectileSpeedScale(1);sim().setProjectileCcd(true);
 const failed=results.filter(r=>r.phase>0||!r.hit);
 return {passed:failed.length===0,shots:results.length,failed,phaseThroughs:results.reduce((s,r)=>s+r.phase,0)};
}

// Phase-throughs of ball `id` between two body snapshots: still pieces (moved
// pieces are hits) whose hull the ball's centre path meets (exact GJK).
function stepCrossings(scene,pieces,prev,now,id){
 const pose=(s,b)=>({p:[s[b*28],s[b*28+1],s[b*28+2]],q:[s[b*28+4],s[b*28+5],s[b*28+6],s[b*28+7]]});
 const b0=pose(prev,id).p,b1=pose(now,id).p,seg=sub(b1,b0),out=[];
 if(b0[1]<-100||b1[1]<-100)return out;
 for(const {c} of pieces){
  const P0=pose(prev,c.body),P1=pose(now,c.body);
  if(Math.hypot(...sub(P1.p,P0.p))>1e-3||Math.abs(dot3(P0.q,P1.q)+P0.q[3]*P1.q[3])<1-1e-6)continue;
  const w=sub(P0.p,b0),tt=Math.min(1,Math.max(0,dot3(w,seg)/Math.max(dot3(seg,seg),1e-20)));
  const bound=c.shape===2?Math.hypot(c.params[0],c.params[1],c.params[2]):c.params[3];
  if(Math.hypot(...sub(add(b0,mul3(seg,tt)),P0.p))>bound+Math.hypot(...c.localPos??[0,0,0])+1e-3)continue;
  if(segmentHullDistance(b0,b1,pieceVertices(scene,c,P0))<=1e-6)out.push(`ball ${id} ${b0.map(x=>x.toFixed(2))} -> ${b1.map(x=>x.toFixed(2))} crossed still piece ${c.body}`);
 }
 return out;
}

// The lab as a person uses it: the frame loop running (heavy steps sliced
// across frames), shots fired through the page's own click path at points
// around the view's centre while earlier balls are still flying and the
// structure is already broken. Every step's bodies are captured and every ball
// checked in every step. `ccd` and `scale` (Ball speed) as the panel sets them.
export async function runLiveShotRegression(demo=window.__FRACTURE_DEMO__,{shots=24,everyMs=150,settleMs=3000,scale=16,ccd=true,log=()=>{}}={}){
 const require=(ok,message)=>{if(!ok)throw new Error(`Live shot regression: ${message}`)};
 require(demo.ready,'demo is not ready');
 await demo.reset();
 const scene=demo.scene,sim=demo.simulation,start=scene.projectileStart;
 const pieces=scene.colliders.map((c,i)=>({c,i})).filter(({c})=>(c.shape===3||c.shape===2)&&c.body>=1&&c.body<start);
 sim.setProjectileCcd(ccd);sim.setProjectileSpeedScale(scale);
 const canvas=document.getElementById('view'),rect=canvas.getBoundingClientRect();
 sim.captureSteps(true);demo.setRunning(true);
 let k=0x5eed;const rnd=()=>{k=(k*1103515245+12345)>>>0;return k/4294967296};
 // The page stops on an error (a capacity limit, a GPU error): report it.
 const pageError=()=>{const st=document.getElementById('status');if(st.classList.contains('error'))throw new Error(`page stopped: ${st.textContent}`)};
 for(let i=0;i<shots;i++){
  pageError();
  // Points over the middle of the view, where the structure is.
  const x=rect.left+rect.width*(0.3+0.4*rnd()),y=rect.top+rect.height*(0.3+0.4*rnd());
  await demo.shoot(demo.renderer.aim(x,y));
  await new Promise(r=>setTimeout(r,everyMs));
 }
 await new Promise(r=>setTimeout(r,settleMs));
 pageError();
 demo.setRunning(false);
 while(demo.simulation.slicing)await new Promise(r=>setTimeout(r,20));
 await new Promise(r=>setTimeout(r,100));
 const entries=await sim.takeCaptured();sim.captureSteps(false);
 let prev=null,steps=0,sliced=0;const fresh=new Set(),flying=new Set(),failures=[];
 for(const e of entries){
  if(e.launch!==undefined){fresh.add(e.launch);flying.add(e.launch);if(e.sliced)sliced++;continue}
  if(prev)for(const id of flying)if(!fresh.has(id))for(const f of stepCrossings(scene,pieces,prev,e.bodies,id))failures.push(`step ${steps}: ${f}`);
  fresh.clear();prev=e.bodies;steps++;
 }
 sim.setProjectileSpeedScale(1);sim.setProjectileCcd(true);
 const r={passed:failures.length===0,shots,steps,launchedMidSlicedStep:sliced,phaseThroughs:failures.length,first:failures.slice(0,5)};
 log(JSON.stringify(r));return r;
}

// The towers showcase in the browser: the "Fire the volley" button on the
// towers block scene, the frame loop running as a person watches it. Every
// step captured, every ball checked; the page's own rates (TPS, × real time)
// sampled each second while the volley flies.
export async function runTowersVolleyRegression(demo=window.__FRACTURE_DEMO__,{ccd=true,settleMs=4000,check=true,log=()=>{}}={}){
 const require=(ok,message)=>{if(!ok)throw new Error(`Towers volley regression: ${message}`)};
 require(demo.ready,'demo is not ready');
 await demo.reset();
 const scene=demo.scene,sim=demo.simulation,start=scene.projectileStart;
 const pieces=scene.colliders.map((c,i)=>({c,i})).filter(({c})=>(c.shape===3||c.shape===2)&&c.body>=1&&c.body<start);
 document.getElementById('ball-ccd').checked=ccd;
 // check: false measures the page's rates alone (no per-step capture).
 if(check)sim.captureSteps(true);
 document.getElementById('volley').click();
 const rates=[];
 while(demo.volleyActive){await new Promise(r=>setTimeout(r,1000));if(demo.rates)rates.push({tps:demo.rates.tps,speed:demo.rates.speed,gpu:demo.simulation.gpuTickMs,fps:demo.rates.fps,sliced:demo.simulation.timing.sliced,wall:demo.simulation.timing.wallMs})}
 const t0=performance.now();
 while(performance.now()-t0<settleMs){await new Promise(r=>setTimeout(r,1000));if(demo.rates)rates.push({tps:demo.rates.tps,speed:demo.rates.speed})}
 demo.setRunning(false);
 while(demo.simulation.slicing)await new Promise(r=>setTimeout(r,20));
 await new Promise(r=>setTimeout(r,100));
 const entries=await sim.takeCaptured();sim.captureSteps(false);
 let prev=null,steps=0;const fresh=new Set(),flying=new Set(),failures=[];
 for(const e of entries){
  if(e.launch!==undefined){fresh.add(e.launch);flying.add(e.launch);continue}
  if(prev)for(const id of flying)if(!fresh.has(id))for(const f of stepCrossings(scene,pieces,prev,e.bodies,id))failures.push(`step ${steps}: ${f}`);
  fresh.clear();prev=e.bodies;steps++;
 }
 document.getElementById('ball-ccd').checked=true;
 const speeds=rates.map(r=>r.speed).filter(Number.isFinite);
 const r={passed:failures.length===0,steps,shots:sim.shots,phaseThroughs:failures.length,first:failures.slice(0,5),
  minSpeed:speeds.length?Math.min(...speeds):null,meanSpeed:speeds.length?speeds.reduce((a,b)=>a+b,0)/speeds.length:null,
  meanTps:rates.length?rates.reduce((a,b)=>a+b.tps,0)/rates.length:null,samples:rates};
 log(JSON.stringify(r));return r;
}
