import {Scene,rotation,mix,identity,interval} from '/scene.js';
const $=id=>document.getElementById(id),scenes={oracle:new Scene($('oracle')),model:new Scene($('model'))};
let data,head=0,playing=false,request=0,last=0;
const camera={center:[0,0,0],radius:1,yaw:0,pitch:.4,zoom:.95};
const percent=x=>(100*x).toFixed(1)+'%',num=x=>Number(x).toLocaleString(undefined,{maximumFractionDigits:1});
async function json(url){const r=await fetch(url);if(!r.ok)throw Error(await r.text());return r.json()}
function pause(){playing=false;$('play').textContent='▶ Play both'}
function reset(){
 const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
 data.oracle[0].centers.forEach((p,i)=>p.forEach((v,k)=>{lo[k]=Math.min(lo[k],v-data.half_extents[i][k]);hi[k]=Math.max(hi[k],v+data.half_extents[i][k])}));
 camera.center=lo.map((v,k)=>(v+hi[k])/2);camera.radius=Math.max(...lo.map((v,k)=>hi[k]-v))/2||1;camera.zoom=.95;camera.pitch=.4;
 const v=data.oracle[0].impactors[0]?.velocity;camera.yaw=v&&Math.hypot(v[0],v[1])>1e-8?Math.atan2(v[0],-v[1])+.35:2.7;draw();
}
function fragmentColor(id){const h=(Number(id)*.61803398875)%1;return [0,1,2].map(k=>.35+.5*(.5+.5*Math.cos(2*Math.PI*(h-k/3))))}
function heat(u){const t=Math.min(1,Math.max(0,u));return t<.5?mix([.16,.4,.73],[.98,.69,.21],t*2):mix([.98,.69,.21],[.9,.22,.16],t*2-1)}
function svgNode(tag,attrs,text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n}
function chart(id,metric,key){
 const svg=$(id);svg.replaceChildren();if(!metric?.axis)return;
 const t=data.metrics.interactions.times_s,series=['oracle','model'].map(s=>metric[s][key]);
 let lo=Math.min(0,...series.flat()),hi=Math.max(0,...series.flat());if(hi===lo){lo-=1;hi+=1}const pad=(hi-lo)*.08;lo-=pad;hi+=pad;
 const X=i=>54+530*(t[i]-t[0])/(t.at(-1)-t[0]),Y=v=>142-124*(v-lo)/(hi-lo);
 for(const value of [lo,(lo+hi)/2,hi]){const y=Y(value);svg.append(svgNode('line',{x1:54,x2:584,y1:y,y2:y,stroke:'#2c3947'}),svgNode('text',{x:48,y:y+4,'text-anchor':'end',fill:'#9aacbd','font-size':11},value.toFixed(1)))}
 svg.append(svgNode('line',{x1:54,x2:584,y1:Y(0),y2:Y(0),stroke:'#9aacbd','stroke-dasharray':'4 4'}));
 series.forEach((values,k)=>svg.append(svgNode('path',{d:values.map((v,i)=>`${i?'L':'M'}${X(i)},${Y(v)}`).join(' '),fill:'none',stroke:k?'#f1a563':'#72d7bb','stroke-width':2.5})));
 svg.append(svgNode('text',{x:54,y:166,fill:'#9aacbd','font-size':11},'0 ms'),svgNode('text',{x:584,y:166,'text-anchor':'end',fill:'#9aacbd','font-size':11},((t.at(-1)-t[0])*1000).toFixed(1)+' ms'));
 svg.append(svgNode('line',{class:'time-cursor',x1:54,x2:54,y1:12,y2:145,stroke:'#e3eaf0','stroke-width':1}));
}
function distance(v){return Math.abs(v)<0.01?`${(v*1000).toFixed(2)} mm`:`${v.toFixed(2)} m`}
function actorMetrics(){
 const all=data?.metrics.interactions,m=all?.actors.find(a=>a.name===$('actor').value);$('outcome').classList.toggle('outcome-failure',Boolean(m?.final_direction_disagreement||data?.assessment?.status==='rejected'));
 if(!m){$('outcome').textContent='No external actor motion is available for this run. Structural response needs a separate assessment.';$('interactionNote').textContent='';}
 else{
  $('outcome').textContent=m.driven?'Actor motion is prescribed: use forces and structural response to evaluate this interaction.':m.final_direction_disagreement?'Interaction mismatch: the oracle and model finish moving in different directions.':m.axis?'Final directions agree. This alone does not establish correct penetration, fracture or energy transfer.':'This actor starts stationary, so an impact direction is not defined.';
  $('interactionNote').textContent=`${m.name} · Final separation ${distance(m.final_position_error_m)} · Trajectory position error ${distance(m.position_vector_rmse_m)} · Velocity error ${m.velocity_vector_rmse_mps.toFixed(2)} m/s (time-weighted vector RMSE). ${data.integration==='native-stress-phase-trajectory/1'?'Native stress integration applies predicted forces and work. Forecasts remain fixed within each interval; this is an experimental learned run.':'This experimental adapter has known motion errors even with exact oracle outputs; these are system errors, not isolated network errors.'}`;
 }
 if(data?.assessment?.status==='rejected')$('outcome').textContent=`Rejected model: ${data.assessment.reason}`;
 for(const side of ['oracle','model']){const v=m?.[side];$(side+'Direction').textContent=v?.final_direction??'—';$(side+'Travel').textContent=v?.forward_displacement_m?num(v.forward_displacement_m.at(-1))+' m':'—';$(side+'Speed').textContent=v?.forward_velocity_mps?num(v.forward_velocity_mps.at(-1))+' m/s':'—'}
 chart('travelPlot',m,'forward_displacement_m');chart('velocityPlot',m,'forward_velocity_mps');draw();
}
function draw(){
 if(!data)return;const {i,alpha}=interval(head,data.count),index=alpha===1?i+1:i,view=$('view').value;
 for(const side of ['oracle','model']){
  const a=data[side][i],b=data[side][i+1],f=data[side][index],util=new Float64Array(data.half_extents.length),broken=new Set();
  data.bonds.forEach(([p,q],j)=>{const u=f.utilization[j]||0;util[p]=Math.max(util[p],u);util[q]=Math.max(util[q],u)});
  for(const j of f.new_breaks){for(const p of data.bonds[j])broken.add(p)}
  const boxes=[];let fragments=new Set();
  f.active.forEach((active,j)=>{if(!active)return;fragments.add(f.clusters[j]);boxes.push({center:mix(a.centers[j],b.centers[j],alpha),rotation:rotation(a.rotations[j],b.rotations[j],alpha),half:data.half_extents[j],color:view==='fragments'?fragmentColor(f.clusters[j]):view==='stress'?heat(util[j]):broken.has(j)?[1,.4,.16]:[.28,.37,.46],broken:broken.has(j)})});
  a.impactors.forEach((p,j)=>{const q=b.impactors[j];boxes.push({center:mix(p.center,q.center,alpha),rotation:rotation(p.rotation,q.rotation,alpha),half:p.shape.half_extents||[p.shape.radius,p.shape.radius,p.shape.radius],sphere:p.shape.type==='sphere',color:[.85,.88,.91]})});
  const r=camera.radius,g=data.ground??Math.min(...data.oracle[0].centers.map((c,j)=>c[2]-data.half_extents[j][2]));
  boxes.push({center:[camera.center[0],camera.center[1],g-.025*r],half:[r*3,r*3,.015*r],rotation:identity,color:[.15,.19,.24]});
  scenes[side].draw(boxes,camera);$(side+'Count').textContent=`${fragments.size} fragments · ${f.new_breaks.length} new breaks`;
 }
 $('frame').value=head;$('time').textContent=`${(head*data.dt*1000).toFixed(1)} / ${(data.count*data.dt*1000).toFixed(1)} ms`;
 for(const cursor of document.querySelectorAll('.time-cursor')){const x=54+530*head/data.count;cursor.setAttribute('x1',x);cursor.setAttribute('x2',x)}
 $('legend').textContent=view==='fragments'?'Each connected fragment has one color within its own world. Colors do not identify matching fragments across worlds.':view==='stress'?'Blue → amber → red: peak load / strength, 0 → 0.5 → 1 or higher. Peak history is cumulative; this is not an instantaneous stress tensor.':'Orange + white edges: pieces attached to bonds that broke at the displayed snapshot. Geometry and connected fragments belong to each world independently.';
}
async function load(){
 const seq=++request;pause();data=null;for(const k of ['play','back','next','frame','reset'])$(k).disabled=true;$('error').hidden=true;
 try{
  const next=await json('/api/rollout?event='+encodeURIComponent($('event').value));if(seq!==request)return;data=next;head=0;
  for(const k of ['play','back','next','frame','reset'])$(k).disabled=false;$('frame').max=data.count;
  const m=data.metrics,e=m.eventual_breaks;$('actual').textContent=e.tp+e.fn;$('predicted').textContent=e.tp+e.fp;$('matched').textContent=e.tp;$('extra').textContent=e.fp;
  const native=data.integration==='native-stress-phase-trajectory/1';$('integrationKind').textContent=native?'Native stress integration · experimental model':'Experimental endpoint integration';
  $('integrationLimits').textContent=native?'This experimental model forecasts joint history, total and elastic forces, stored energy and work from each live frame start. The native integrator applies them at its actual stress phases; contacts, rigid motion and splits remain native. Breaks can occur within a frame. Forecasts do not react to new within-frame contacts. No validated production accuracy, conservation or runtime speed claim.':'This experimental endpoint adapter interpolates hidden deformation increments and commits breaks at interval boundaries. Exact oracle endpoints also produce motion errors through it. Model and integration errors are mixed; it is not a validated replacement.';
  $('scope').textContent='Experimental independent rollout · '+data.count+' intervals · '+(data.dt*1000).toFixed(2)+' ms per model prediction · zero oracle resets';
  $('story').textContent=`Final outcome: oracle ${m.oracle_fragments.count} fragments, largest holds ${percent(m.oracle_fragments.largest_mass_fraction)} of active mass. Model ${m.model_fragments.count} fragments, largest holds ${percent(m.model_fragments.largest_mass_fraction)}. Final position error (per-axis RMSE): ${(m.final_position_rmse_m*100).toFixed(1)} cm.`;
  $('timing').textContent=m.timing.map(t=>`${t.tolerance_seconds===0?'Same frame':'Within ±'+(t.tolerance_seconds*1000).toFixed(1)+' ms'}: ${percent(t.f1)} F1`).join(' · ')+` · Any frame: ${percent(e.f1)} F1.`;
  $('energy').textContent=`Initial mechanical energy: ${num(m.oracle_initial_mechanical_energy_j)} J. Final oracle: ${num(m.oracle_final_mechanical_energy_j)} J; model: ${num(m.model_final_mechanical_energy_j)} J. ${m.analytic_bond_evaluations} analytic bond evaluations, ${num(m.contact_substeps)} native substeps. ${native?'Work is predicted and integrated at native stress phases; this does not by itself prove energy consistency.':'Energy differences include missing learned dissipation and integration effects; they are not an accuracy score by themselves.'}`;
  $('checkpoint').textContent=`Checkpoint ${data.checkpoint.sha256}. ${data.scope}`;
  $('actor').replaceChildren();for(const a of m.interactions?.actors??[])$('actor').add(new Option(a.name,a.name));$('actor').disabled=!$('actor').options.length;actorMetrics();
  const url=new URL(location.href);url.searchParams.set('event',$('event').value);history.replaceState(null,'',url);reset();
 }catch(e){$('error').hidden=false;$('error').textContent=e.message}
}
$('event').onchange=load;$('view').onchange=draw;$('reset').onclick=reset;
$('actor').onchange=actorMetrics;
for(const id of ['travelPlot','velocityPlot'])$(id).onclick=e=>{if(!data)return;const point=$(id).createSVGPoint();point.x=e.clientX;point.y=e.clientY;const x=point.matrixTransform($(id).getScreenCTM().inverse()).x;pause();head=Math.max(0,Math.min(data.count,(x-54)/530*data.count));draw()};
$('play').onclick=()=>{if(!data)return;if(head>=data.count)head=0;playing=!playing;$('play').textContent=playing?'Ⅱ Pause':'▶ Play both'};
$('frame').oninput=()=>{pause();head=Number($('frame').value);draw()};
for(const [id,step] of [['back',-1],['next',1]])$(id).onclick=()=>{pause();head=Math.max(0,Math.min(data.count,Math.round(head)+step));draw()};
for(const side of ['oracle','model']){const c=$(side);let drag=null;c.onpointerdown=e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointerup=()=>drag=null;c.onpointercancel=()=>drag=null;c.onpointermove=e=>{if(!drag)return;camera.yaw+=(e.clientX-drag[0])*.008;camera.pitch=Math.max(-1.4,Math.min(1.4,camera.pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];draw()};c.addEventListener('wheel',e=>{e.preventDefault();camera.zoom=Math.max(.15,Math.min(8,camera.zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false})}
window.addEventListener('resize',draw);
function tick(t){if(playing&&data){head=Math.min(data.count,head+(t-last)/1000*data.count/Number($('duration').value));draw();if(head>=data.count)pause()}last=t;requestAnimationFrame(tick)}requestAnimationFrame(tick);
try{const catalog=await json('/api/rollouts');if(!catalog.length)throw Error('No completed independent runs are published yet.');for(const x of catalog)$('event').add(new Option(x.label,x.id));const key=new URL(location.href).searchParams.get('event');if(catalog.some(x=>x.id===key))$('event').value=key;await load()}catch(e){$('error').hidden=false;$('error').textContent=e.message}
