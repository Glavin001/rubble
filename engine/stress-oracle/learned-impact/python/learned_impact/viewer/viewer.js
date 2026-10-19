import {Scene,rotation,mix,interval,identity} from './scene.js';
const $=id=>document.getElementById(id);
let index,data,playhead=0,playing=false,last=0,request=0,scenes,currentPose,initialSelection=true;
let camera={center:[0,0,0],radius:1,yaw:0,pitch:.45,zoom:1};
const rgb=hex=>[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)/255);
const colors={unscored:rgb('#88909b'),idle:rgb('#345e85'),caught:rgb('#64d6ac'),missed:rgb('#ed6b89'),extra:rgb('#e9914e'),break:rgb('#fff1d2'),projectile:rgb('#dce9f3')};
const heat=v=>v<.5?mix(rgb('#326caa'),rgb('#4ab7c2'),Math.max(0,v)*2):v<1?mix(rgb('#4ab7c2'),rgb('#f6bc55'),(v-.5)*2):mix(rgb('#f6bc55'),rgb('#f15c58'),Math.min(1,(v-1)*2));
const magnitude=t=>t<.5?mix(rgb('#326caa'),rgb('#f6bc55'),Math.max(0,t)*2):mix(rgb('#f6bc55'),rgb('#f15c58'),Math.min(1,(t-.5)*2));
function responseInfo(f){
 const mode=$('view').value;if(!mode.startsWith('response_')||!f.response)return null;
 const field=mode.includes('energy')?'energy':'force',error=mode.endsWith('_error');
 return {...f.response[field],errorValues:f.response[field].error,valid:f.response.valid,field,error,unit:field==='force'?'N':'J',max:data.response_scales[field]};
}
function fail(e){$('error').hidden=false;$('error').textContent=e.message||String(e)}
async function json(url){const r=await fetch(url);if(!r.ok)throw Error(`${r.status}: ${await r.text()}`);return r.json()}
function option(value,text){const o=document.createElement('option');o.value=value;o.textContent=text;return o}
function syncUrl(){
 if(!data||!index)return;
 const model=index.models[+$('model').value];if(!model)return;
 const url=new URL(location.href),short=model.sha256.slice(0,8);
 url.searchParams.set('checkpoint',index.models.filter(m=>m.sha256.startsWith(short)).length===1?short:model.sha256);
 for(const key of ['dataset','event','view','geometry'])url.searchParams.set(key,$(key).value);
 history.replaceState(null,'',url);
}
function pause(){playing=false;$('play').textContent=playhead>=data?.frames.length?'↺ Replay impact':'▶ Play impact'}
function reset(){
 const pts=data.replay?.frames[0].centers||data.geometry.centers,lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
 pts.forEach((p,i)=>p.forEach((v,k)=>{lo[k]=Math.min(lo[k],v-data.half_extents[i][k]);hi[k]=Math.max(hi[k],v+data.half_extents[i][k])}));
 camera.center=lo.map((v,k)=>(v+hi[k])/2);camera.radius=Math.max(...lo.map((v,k)=>hi[k]-v))/2||1;camera.zoom=.95;camera.pitch=.40;
 const blasts=data.replay?.blasts||[];
 if(blasts.length){const distance=Math.max(...blasts.map(b=>Math.hypot(...b.position.map((v,k)=>v-camera.center[k]))));camera.zoom=Math.min(camera.zoom,1.15*camera.radius/(distance+.1))}
 const impactor=data.replay?.frames[0].impactors[0],p=impactor?.center||blasts[0]?.position,v=impactor?.velocity;
 // An off-centre ram starts almost touching the wall; its position can give an
 // edge-on camera. Look along its approach direction when it is moving.
 camera.yaw=v&&Math.hypot(v[0],v[1])>1e-8?Math.atan2(v[0],-v[1])+.35:p?Math.atan2(-(p[0]-camera.center[0]),p[1]-camera.center[1])+.35:2.7;
 draw();
}
async function load(){
 const seq=++request;pause();data=null;for(const id of ['play','frame','back','next','firstBreak','reset'])$(id).disabled=true;for(const id of ['oracle','prediction'])$(id).style.opacity='.35';$('description').textContent='Running checkpoint on recorded frames…';$('error').hidden=true;
 try{
  const next=await json(`/api/event?dataset=${encodeURIComponent($('dataset').value)}&event=${encodeURIComponent($('event').value)}&model=${$('model').value}`);if(seq!==request)return;
  data=next;playhead=0;
  data.response_scales={force:0,energy:0};
  for(const f of data.frames)if(f.response)for(const field of ['force','energy'])for(const kind of ['oracle','model','error'])for(const value of f.response[field][kind])data.response_scales[field]=Math.max(data.response_scales[field],value||0);
  for(const o of $('view').options)if(o.value.startsWith('response_'))o.disabled=!data.response_available;
  if(!data.response_available&&$('view').value.startsWith('response_'))$('view').value='stress';
  for(const select of ['view','geometry'])for(const o of $(select).options)if(o.value.startsWith('world_')||o.value==='predicted')o.disabled=!data.world_motion_available;
  if(!data.world_motion_available){if($('view').value.startsWith('world_'))$('view').value='stress';if($('geometry').value==='predicted')$('geometry').value='replay'}for(const id of ['play','frame','back','next','firstBreak','reset'])$(id).disabled=false;for(const id of ['oracle','prediction'])$(id).style.opacity='1';$('frame').max=data.frames.length;$('frame').value=0;
  data.world_motion_control=data.world_motion_available&&index.models[+$('model').value].run.world_motion_weight===0;
  const m=data.manifest,model=index.models[+$('model').value],same=model.run.manifest_sha256===index.dataset_sha256;
  $('description').textContent=`${m.structure_type} · ${m.load_type} · ${m.n_chunks.toLocaleString()} pieces · ${m.n_bonds.toLocaleString()} bonds · ${m.n_frames} recorded intervals`;
  const partition={train:'Training example · measures fit',validation:'Validation example · used for model selection',test:'Held-out test example',unseen:'Outside this checkpoint’s training corpus',unknown:'Training coverage unknown'};
  $('scope').textContent=partition[data.partition]||'Recorded-frame comparison';
  $('method').textContent=data.evaluation;
  $('provenance').textContent=`Checkpoint SHA-256: ${model.sha256}. ${partition[data.partition]||'Training coverage unknown'}. ${model.run.architecture?.startsWith('physics-')?`Separate break probability and timing; break decision threshold ${model.run.threshold.toFixed(2)}. `:''}${data.replay?data.replay.verification:'World motion unavailable: this legacy event could not be reproduced exactly. Static geometry is shown.'}`;
  const params=data.context?.generator?.parameters;
  $('recipe').textContent=params?Object.entries(params).map(([k,v])=>`${k.replaceAll('_',' ')}: ${Array.isArray(v)?v.join(', '):v}`).join(' · ')+'. Positions in meters, mass in kilograms, speed in m/s, angles in degrees. Material values are multipliers of the recorded base scene.':'Legacy scenario: exact parameter recipe was not captured with this recording.';
  const coverage=index.coverage,metrics=model.metrics||{},score=metrics.test||metrics.all_recorded_frames;
  $('datasetReport').textContent=coverage?`${coverage.events} scenarios · ${coverage.frames.toLocaleString()} intervals · ${coverage.complete_v2_recordings} new complete recordings · ${coverage.chunks_min_max.join('–')} pieces per scenario. ${coverage.events_without_breaks} scenarios have no breaks; ${coverage.events_still_breaking_at_end} still have new breaks at the end and need longer continuations. Remaining priorities: independent structural layouts, larger connected graphs, repeated impacts, and full feedback between contacts and fracture.`:`${index.events.length} scenarios · ${index.frames} intervals.`;
  $('checkpointReport').textContent=score?`${metrics.test?'Held-out test':'All-data training fit'}: precision ${(score.precision*100).toFixed(1)}%, recall ${(score.recall*100).toFixed(1)}%, F1 ${score.f1==null?"—":score.f1.toFixed(3)} across ${score.frames} intervals. This reports the checkpoint’s saved evaluation corpus, not only the selected scenario. ${data.input_contract==='frame_start_forcing'?'Oracle start states and prescribed forcing descriptors; no contact-preview inputs.':data.input_contract==='causal_preview'?'Oracle start states with causal contact preview inputs.':'Oracle start states and future interval force histories.'} Autonomous collapse is not validated. ${score.state_rmse?.v_mps!=null&&score.zero_increment_rmse?.v_mps!=null?`Local velocity RMSE ${score.state_rmse.v_mps.toPrecision(3)} m/s; no-change baseline ${score.zero_increment_rmse.v_mps.toPrecision(3)} m/s (lower is better).`:''}`:'This legacy checkpoint has no grouped evaluation report attached. Use the interval and event diagnostics, and do not treat training-fit metrics as generalization.';
  $('playbackNote').textContent=data.replay?`Verified oracle motion · ${(m.dt*m.n_frames*1000).toFixed(1)} ms of physics, slowed down for inspection. Impact recordings begin just before contact; poses are interpolated, interval peak stresses are not.`:'Static-only legacy recording: world motion could not be verified. The timeline still compares recorded stress and break labels.';
  reset();syncUrl();
 }catch(e){if(seq===request)fail(e)}
}
function pose(prediction=false){
 const {i,alpha}=interval(playhead,data.frames.length),r=data.replay;
 if(!r||$('geometry').value==='static')return {centers:data.geometry.centers,rotations:data.geometry.rotations,active:data.half_extents.map(()=>true),impactors:[]};
 const a=r.frames[i],recorded=r.frames[i+1],world=data.frames[i].world;
 const b=prediction&&world&&$('geometry').value==='predicted'?{...recorded,centers:world.centers.map((p,n)=>world.valid[n]?p:recorded.centers[n]),rotations:world.rotations.map((p,n)=>world.valid[n]?p:recorded.rotations[n])}:recorded;
 return {centers:a.centers.map((p,n)=>mix(p,b.centers[n],alpha)),rotations:a.rotations.map((r,n)=>rotation(r,b.rotations[n],alpha)),active:alpha<1?a.active:b.active,impactors:a.impactors.map((p,n)=>({...p,center:mix(p.center,b.impactors[n].center,alpha),rotation:rotation(p.rotation,b.impactors[n].rotation,alpha)}))};
}
function marker(point,other,p,n){
 // Attach a bond-end mark to the chunk, so split fragments cannot be joined by a long false bond.
 const initial=data.replay?.frames[0]||data.geometry,offset=initial.centers[other].map((v,k)=>(v-initial.centers[n][k])*.5),r0=initial.rotations[n],r=p.rotations[n];
 const local=[0,1,2].map(k=>r0.reduce((s,row,j)=>s+row[k]*offset[j],0));
 return point.map((v,k)=>v+r[k].reduce((s,x,j)=>s+x*local[j],0));
}
function boxes(p,prediction){
 const f=data.frames[interval(playhead,data.frames.length).i],mode=$('view').value,values=new Float32Array(p.centers.length),priority=new Uint8Array(p.centers.length),broken=new Uint8Array(p.centers.length),marks=[];
 const response=responseInfo(f);
 const active=playhead>0,showBreaks=['stress','breaks','failure'].includes(mode);
 const loadMax=Math.max(1e-12,...f.truth_load,...f.model_load,...(mode==='load_error'?f.load_error:[]));
 const world=f.world,worldMax=world?Math.max(1e-12,...(mode==='world_error'?[...world.position_error,...world.baseline_position_error]:[...world.truth_travel,...world.predicted_travel])):1;
 const deltaMax=Math.max(1e-12,...f.truth_delta,...f.predicted_delta,...(mode==='delta_error'?f.delta_error:[]));
 if(active)f.active.forEach((edge,j)=>{
  const ends=data.endpoints[edge],truth=f.truth[j]>0,pred=f.predicted[j]>0,value=(prediction?f.predicted_util:f.truth_util)[j];
  let status=truth&&pred?1:pred?2:truth?3:0;
  for(const n of ends){values[n]=Math.max(values[n],value);if(prediction&&mode==='breaks')priority[n]=Math.max(priority[n],status)}
  if(showBreaks&&(prediction?pred:truth))for(let k=0;k<2;k++){const n=ends[k];broken[n]=1;if(p.active[n])marks.push({center:marker(p.centers[n],ends[1-k],p,n),rotation:p.rotations[n],half:[.009,.009,.009].map(x=>Math.max(x,camera.radius*.007)),color:colors.break})}
 });
 const list=p.centers.flatMap((center,n)=>p.active[n]?[{center,rotation:p.rotations[n],half:data.half_extents[n],broken:broken[n],color:!active?colors.idle:response?(response.valid[n]?magnitude((prediction?(response.error?response.errorValues[n]:response.model[n]):response.oracle[n])/Math.max(1e-12,response.max)):colors.unscored):mode.startsWith('world_')&&world?(world.valid[n]?magnitude((mode==='world_error'?(prediction?world.position_error:world.baseline_position_error):(prediction?world.predicted_travel:world.truth_travel))[n]/worldMax):colors.unscored):mode.startsWith('delta')&&f.motion_valid?.[n]===false?colors.unscored:mode==='failure'?(broken[n]?colors.extra:colors.idle):mode==='loads'||mode==='load_error'?magnitude((prediction?(mode==='load_error'?f.load_error:f.model_load):f.truth_load)[n]/loadMax):mode==='delta'||mode==='delta_error'?magnitude((prediction?(mode==='delta_error'?f.delta_error:f.predicted_delta):f.truth_delta)[n]/deltaMax):prediction&&mode==='breaks'?(priority[n]?[null,colors.caught,colors.extra,colors.missed][priority[n]]:colors.idle):heat(values[n])}]:[]);
 for(const im of p.impactors)list.push({center:im.center,rotation:im.rotation,half:im.shape.half_extents||[im.shape.radius,im.shape.radius,im.shape.radius],sphere:im.shape.type==='sphere',color:colors.projectile});
 for(const blast of data.replay?.blasts||[])list.push({center:blast.position,rotation:identity,half:[1,1,1].map(()=>camera.radius*.035),sphere:true,color:rgb('#f6bc55')});
 // A reference floor/grid adds depth; it is scenery, not a simulated collider.
 const r=camera.radius,ground=data.replay?.ground??Math.min(...data.geometry.centers.map((c,n)=>c[2]-data.half_extents[n][2]));
 list.push({center:[camera.center[0],camera.center[1],ground-.025*r],half:[r*3,r*3,.015*r],rotation:identity,color:rgb('#26303c')});
 for(let j=-6;j<=6;j++){const coord=j*r/2;list.push({center:[camera.center[0]+coord,camera.center[1],ground-.009*r],half:[.001*r,r*3,.001*r],rotation:identity,color:rgb('#374451')},{center:[camera.center[0],camera.center[1]+coord,ground-.009*r],half:[r*3,.001*r,.001*r],rotation:identity,color:rgb('#374451')})}
 return list.concat(marks);
}
function draw(){
 if(!data||!scenes)return;
 const {i}=interval(playhead,data.frames.length),f=data.frames[i],m=playhead>0?f.metrics:{tp:0,fn:0,fp:0},p=pose();
 const predictedPose=pose(true);currentPose=[p,predictedPose];scenes[0].draw(boxes(p,false),camera);scenes[1].draw(boxes(predictedPose,true),camera);
 $('frame').value=playhead;if(!playing)$('play').textContent=playhead>=data.frames.length?'↺ Replay impact':'▶ Play impact';
 $('time').textContent=`${(playhead*data.manifest.dt*1000).toFixed(2)} ms · ${playhead===0?'start':`interval ${i+1}/${data.frames.length}`}`;
 $('actual').textContent=m.tp+m.fn;$('caught').textContent=m.tp;$('missed').textContent=m.fn;$('extra').textContent=m.fp;
 $('oracleCount').textContent=`${m.tp+m.fn} new breaks`;$('modelCount').textContent=`${m.tp+m.fp} predicted breaks`;
 $('story').textContent=playhead===0?'Ready at impact. Press Play, or jump to the first recorded break.':`In interval ${i+1}, the oracle broke ${m.tp+m.fn} bonds. The model caught ${m.tp}, missed ${m.fn}, and predicted ${m.fp} additional breaks.`;
 const blastNote=(data.replay?.blasts||[]).map(b=>`${b.tnt_mass} kg TNT · amber dot marks source`).join(', ');
 const note=data.replay&&$('geometry').value!=='static'?`${p.impactors.length?p.impactors.map(x=>x.name).join(', ')+' · ':blastNote?blastNote+' · ':''}verified oracle motion`:$('geometry').value==='static'?'Undeformed geometry · same recorded labels':'Static geometry · motion unavailable';
 $('motionContext').textContent=data.replay&&$('geometry').value!=='static'?'Same motion, different predictions. Both views follow recorded oracle motion; the model predicts fields on those frames.':'Undeformed inspection. Both views use the same static geometry to compare the selected oracle interval with the model prediction.';
 $('oracleNote').textContent=note;$('modelNote').textContent=note;
 if($('geometry').value==='predicted'&&f.world){$('motionContext').textContent='One-step position and rotation prediction. Right: recorded start → model endpoint, then reset to the next recorded start. Projectiles and activity follow the oracle. This is not an independent collapse simulation.';$('modelNote').textContent='Predicted endpoint · resets each interval'}
 if(data.world_motion_control)$('motionContext').textContent+=' This checkpoint uses the constant-velocity world-motion control (loss weight 0).';
 const mode=$('view').value,deltaMax=Math.max(0,...f.truth_delta,...f.predicted_delta,...(mode==='delta_error'?f.delta_error:[]));
 const loadMax=Math.max(0,...f.truth_load,...f.model_load,...(mode==='load_error'?f.load_error:[]));
 const world=f.world,worldMax=world?Math.max(0,...(mode==='world_error'?[...world.position_error,...world.baseline_position_error]:[...world.truth_travel,...world.predicted_travel])):0;
 let keys=mode.startsWith('world_')?[['#326caa','0 mm'],['#f6bc55',`${(worldMax*500).toPrecision(3)} mm`],['#f15c58',`${(worldMax*1000).toPrecision(3)} mm · shared maximum`],['#88909b','Inactive endpoint · not scored']]:mode==='loads'||mode==='load_error'?[['#326caa','0 N'],['#f6bc55',`${(loadMax*.5).toPrecision(3)} N`],['#f15c58',`${loadMax.toPrecision(3)} N · shared maximum`]]:mode==='failure'?[['#345e85','No new failure'],['#e9914e','New failure this interval']]:mode==='delta'||mode==='delta_error'?[['#326caa','0 mm'],['#f6bc55',`${(deltaMax*500).toPrecision(3)} mm`],['#f15c58',`${(deltaMax*1000).toPrecision(3)} mm · shared maximum`]]:mode==='breaks'?[['#64d6ac','Caught'],['#ed6b89','Missed'],['#e9914e','Extra'],['#345e85','No new break'],['#fff1d2','Pieces attached to new breaks']]:[['#326caa','Low load'],['#4ab7c2','50% of strength'],['#f6bc55','100% · strength limit'],['#f15c58','150%+'],...(mode==='stress'?[['#fff1d2','Pieces attached to new breaks']]:[])];
 const help={world_travel:'Distance from the recorded start to the endpoint, including rigid motion and fragment motion. Left: oracle. Right: model. Shared scale in millimeters. Choose Model endpoint in Geometry to inspect the predicted positions and rotations; each interval starts fresh from the oracle.',world_error:'Endpoint position error, in millimeters; lower is better. Left: constant-velocity baseline error. Right: model error. Both compare with the same recorded oracle endpoint. Choose Model endpoint in Geometry to see predicted poses. Colors describe the endpoint even during interpolation.',loads:`Largest force magnitude among the four bin averages, in newtons. Left: recorded oracle force. Right: ${data.input_contract==='frame_start_forcing'?'unused contact-input slots shown as zero; this is not a prediction of zero force. Prescribed forcing is supplied as scene descriptors':data.input_contract==='causal_preview'?'causal rigid-contact preview supplied to this model':'the same recorded force supplied to this model (future interval information)'}. These are model inputs, not predicted stress. Torques are available on hover.`,load_error:'Left: recorded force magnitude. Right: largest vector difference between model-input and oracle force across the four bins. Blue on the right means little input error. Shared scale in newtons; stress and break predictions remain available in other views.',stress:'Stress and new breaks together. Colors use the same fixed strength scale in both panels.',breaks:'Model panel: green = caught, pink = missed, orange = extra. Oracle panel keeps the stress field. If a piece touches several categories, missed takes priority.',failure:'Each panel highlights pieces attached to its own newly broken bonds. Previously broken bonds are excluded; the counts refer to bonds, not pieces.',util:'Peak load divided by bond strength, shown as the maximum over each piece’s attached live bonds. 1.0 means the strength limit. Break outlines are hidden to expose the field.',delta:'Magnitude of local deformation change during this interval, in millimeters. This excludes cluster rigid motion. Both panels share one scale; it can change between intervals.',delta_error:'Left: actual local displacement increment. Right: length of the vector difference between predicted and actual local increments, in millimeters. Blue on the right means little error. Both use the displayed shared scale.'};
 if(mode.startsWith('delta')){if(f.metrics.motion_scored_chunk_intervals===0)keys.splice(0,keys.length,['#88909b','No comparable local-motion targets']);else keys.push(['#88909b','Coordinate reset · not scored'])}
 const response=responseInfo(f);
 if(response){
  keys=[['#326caa',`0 ${response.unit}`],['#f6bc55',`${(response.max*.5).toPrecision(3)} ${response.unit}`],['#f15c58',`${response.max.toPrecision(3)} ${response.unit} · fixed for this replay`],['#88909b','No active endpoint bond']];
  help[mode]=`Endpoint ${response.field==='force'?'bond force magnitude':'stored elastic energy per bond'}. Left: oracle. Right: ${response.error?(response.field==='force'?'length of the force-vector error':'absolute energy error'):'model prediction'}. A piece shows its largest attached active-bond value, not its total ${response.field==='force'?'net force':'energy'}. Shared ${response.unit} scale stays fixed through this replay. Both panels use the oracle endpoint activity mask; ${f.response.retired_caches_excluded} retired caches are excluded in this interval. Colors describe the endpoint during motion interpolation.`;
 }
 $('viewHelp').textContent=help[mode]+(mode.startsWith('delta')?` ${f.metrics.motion_scored_chunk_intervals} of ${data.half_extents.length} pieces have comparable targets. Gray pieces changed fragment coordinates; their local increments cannot be compared. Recorded world motion is still shown.`:'');
 $('oracleSubtitle').textContent=mode==='world_error'?'Constant-velocity error':mode==='world_travel'?'Recorded world travel':mode.startsWith('load')?'Recorded interval force':mode.startsWith('delta')?'Recorded local increment':'What actually happened';
 $('modelSubtitle').textContent=mode==='world_error'?'Predicted position error':mode==='world_travel'?'Predicted world travel':mode==='load_error'?'Contact input error':mode==='loads'?(data.input_contract==='frame_start_forcing'?'No contact-force input':data.input_contract==='causal_preview'?'Causal contact preview':'Recorded force input'):mode==='delta_error'?'Local displacement error':mode==='delta'?'Predicted local increment':mode==='breaks'?'Caught / missed / extra':'What it predicted';
 if(response){$('oracleSubtitle').textContent=`Recorded endpoint · ${response.unit}`;$('modelSubtitle').textContent=`${response.error?'Endpoint error':'Predicted endpoint'} · ${response.unit}`;}
 const fm=x=>x==null?'—':x.toFixed(3);
 $('diagnostics').textContent=`Interval precision ${fm(f.metrics.precision)} · recall ${fm(f.metrics.recall)} · F1 ${fm(f.metrics.f1)} · event pooled F1 ${fm(data.metrics.f1)} · utilization MAE in log1p space ${fm(f.metrics.util_log_mae)} · local displacement RMSE ${f.metrics.displacement_rmse_m==null?'unavailable':(f.metrics.displacement_rmse_m*1000).toPrecision(4)+' mm'} (${f.metrics.motion_scored_chunk_intervals} pieces scored; ${f.metrics.motion_fragment_reset_excluded_chunk_intervals} coordinate resets excluded). Metrics describe the selected interval, including at its starting pose.`;
 if(world)$('diagnostics').textContent+=` World endpoint position RMSE ${world.position_rmse_m==null?'unavailable':(world.position_rmse_m*1000).toPrecision(4)+' mm'}; constant-velocity baseline ${world.baseline_position_rmse_m==null?'unavailable':(world.baseline_position_rmse_m*1000).toPrecision(4)+' mm'}; orientation RMSE ${world.orientation_rmse_rad==null?'unavailable':(world.orientation_rmse_rad*180/Math.PI).toPrecision(4)+'°'} (${world.scored_pieces} pieces scored, including fragment resets).`;
 if(response){const fmt=x=>x==null?'unavailable':x.toPrecision(4);$('diagnostics').textContent+=` Active bond ${response.field==='force'?'force-component':'stored-energy'} RMSE: model ${fmt(response.rmse)} ${response.unit}; zero baseline ${fmt(response.zero_rmse)} ${response.unit}; previous-response baseline ${fmt(response.persistence_rmse)} ${response.unit}. Lower is better. ${f.response.active_bonds} active bonds scored.`;}
 $('legend').replaceChildren(...($('view').value==='breaks'?[document.createTextNode('Model colors:')]:[]),...keys.map(([c,text])=>{const s=document.createElement('span'),b=document.createElement('i');b.className='key';b.style.background=c;s.append(b,document.createTextNode(text));return s}));
}
$('event').onchange=load;$('model').onchange=load;$('view').onchange=()=>{draw();syncUrl()};$('geometry').onchange=()=>{draw();syncUrl()};$('reset').onclick=reset;
$('frame').oninput=()=>{pause();playhead=+$('frame').value;draw()};
$('back').onclick=()=>{pause();playhead=Math.max(0,Math.ceil(playhead)-1);draw()};$('next').onclick=()=>{pause();playhead=Math.min(data.frames.length,Math.floor(playhead)+1);draw()};
$('firstBreak').onclick=()=>{pause();const i=data.frames.findIndex(f=>f.truth.some(x=>x>0));playhead=i<0?data.frames.length:i+.999;draw()};
$('play').onclick=()=>{if(!data)return;if(playing){pause();return}if(playhead>=data.frames.length)playhead=0;playing=true;last=performance.now();$('play').textContent='Ⅱ Pause'};
function tick(t){if(playing&&data){playhead=Math.min(data.frames.length,playhead+(t-last)/1000*data.frames.length/(+$('duration').value));last=t;draw();if(playhead>=data.frames.length)pause()}requestAnimationFrame(tick)}requestAnimationFrame(tick);

function inspect(canvas,x,y){
 if(!data||!currentPose)return;
 const p=currentPose[canvas.id==='prediction'?1:0],c=camera,w=canvas.clientWidth,h=canvas.clientHeight,s=c.zoom/(c.radius*1.35),cy=Math.cos(c.yaw),sy=Math.sin(c.yaw),cp=Math.cos(c.pitch),sp=Math.sin(c.pitch);
 const right=[cy,sy,0],up=[sy*sp,-cy*sp,cp],eye=[-sy*cp,cy*cp,sp],dx=(2*x/w-1)*Math.max(1,w/h)/s,dy=(1-2*y/h)/(s*Math.min(1,w/h));
 const origin=c.center.map((v,k)=>v+right[k]*dx+up[k]*dy+eye[k]*c.radius*40),dir=eye.map(x=>-x);let hit=-1,nearest=Infinity;
 p.centers.forEach((center,n)=>{if(!p.active[n])return;const r=p.rotations[n],v=origin.map((x,k)=>x-center[k]),o=[0,1,2].map(k=>r.reduce((s,row,j)=>s+row[k]*v[j],0)),d=[0,1,2].map(k=>r.reduce((s,row,j)=>s+row[k]*dir[j],0));let lo=0,hi=Infinity;
 for(let k=0;k<3;k++){const half=data.half_extents[n][k];if(Math.abs(d[k])<1e-10){if(Math.abs(o[k])>half)return}else{const a=(-half-o[k])/d[k],b=(half-o[k])/d[k];lo=Math.max(lo,Math.min(a,b));hi=Math.min(hi,Math.max(a,b))}}if(lo<=hi&&lo<nearest){nearest=lo;hit=n}});
 if(hit<0)return;
 const f=data.frames[interval(playhead,data.frames.length).i];let bond=-1,value=-1;
 f.active.forEach((edge,j)=>{if(data.endpoints[edge].includes(hit)){const u=Math.max(f.truth_util[j],f.predicted_util[j]);if(u>value){value=u;bond=j}}});
 let text=`Piece ${hit}: force oracle ${f.truth_load[hit].toPrecision(3)} N / input ${f.model_load[hit].toPrecision(3)} N; torque ${f.truth_torque[hit].toPrecision(3)} / ${f.model_torque[hit].toPrecision(3)} N·m. ${f.motion_valid?.[hit]===false?'Local displacement comparison unavailable: fragment coordinates reset.':`Local Δu oracle ${(f.truth_delta[hit]*1000).toPrecision(3)} mm / model ${(f.predicted_delta[hit]*1000).toPrecision(3)} mm; vector error ${(f.delta_error[hit]*1000).toPrecision(3)} mm.`}`;
 if(f.world){const w=f.world;text+=w.valid[hit]?` World endpoint error ${(w.position_error[hit]*1000).toPrecision(3)} mm; constant-velocity baseline ${(w.baseline_position_error[hit]*1000).toPrecision(3)} mm.`:' World endpoint inactive and unscored.'}
 const response=responseInfo(f);if(response){const fmt=x=>x==null?'unavailable':x.toPrecision(4);text+=response.valid[hit]?` Largest attached active-bond ${response.field}: oracle ${fmt(response.oracle[hit])}, model ${fmt(response.model[hit])}, error ${fmt(response.errorValues[hit])} ${response.unit}.`:' No active endpoint bond; cached retired responses are excluded.';}
 if(bond>=0){const bin=x=>x===0?'no break':`break in quarter ${x}/4`;text+=` Most loaded attached live bond ${f.active[bond]}: load/strength ${f.truth_util[bond].toPrecision(3)} / ${f.predicted_util[bond].toPrecision(3)}; oracle ${bin(f.truth[bond])}, model ${bin(f.predicted[bond])}.`}else text+=' No live attached bonds.';
 $('inspect').textContent=text;
}

for(const id of ['oracle','prediction']){const c=$(id);let drag=null;c.onpointerdown=e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointerup=c.onpointercancel=()=>drag=null;c.onpointermove=e=>{if(!drag){const r=c.getBoundingClientRect();inspect(c,e.clientX-r.left,e.clientY-r.top);return}camera.yaw-=(e.clientX-drag[0])*.008;camera.pitch=Math.max(-.1,Math.min(1.45,camera.pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];draw()};c.addEventListener('wheel',e=>{e.preventDefault();camera.zoom=Math.max(.15,Math.min(8,camera.zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false})}
new ResizeObserver(draw).observe($('oracle'));
async function loadDataset(){
 const seq=++request;pause();data=null;
 for(const id of ['play','frame','back','next','firstBreak','reset'])$(id).disabled=true;
 const nextIndex=await json(`/api/index?dataset=${encodeURIComponent($('dataset').value)}`);
 if(seq!==request)return;index=nextIndex;
 for(const name of index.datasets||[])if(!Array.from($('dataset').options).some(o=>o.value===name))$('dataset').append(option(name,index.dataset_labels?.[name]||name));
 $('event').replaceChildren();$('model').replaceChildren();
 const groups=new Map();
 for(const e of index.events){const group=`${e.motion?'Motion replay':'Static only'} · ${e.structure_type}`;let g=groups.get(group);if(!g){g=document.createElement('optgroup');g.label=group;groups.set(group,g);$('event').append(g)}g.append(option(e.path,`${e.event_id} · ${e.dt?Number((e.dt*1000).toPrecision(4)):e.n_frames} ${e.dt?'ms / frame':'frames'}`))}
 index.models.forEach((m,i)=>$('model').append(option(i,`${m.name} · ${m.sha256.slice(0,8)}`)));
 const previewModel=index.models.findIndex(m=>m.run.architecture==='physics-preview-v3');if(previewModel>=0)$('model').value=previewModel;
 const initial=($('dataset').value==='tails'?index.events.find(e=>e.event_id==='v3_house_04_s101_f120'):null)||($('dataset').value==='preview'?index.events.find(e=>e.event_id==='v3_house_04_s101'):null)||index.events.find(e=>e.motion&&e.event_id==='v3_house_06_s101')||index.events.find(e=>e.motion&&e.event_id==='v2_house_ring_25mps_f24')||index.events.find(e=>e.motion&&e.structure_type==='masonry_house')||index.events.find(e=>e.motion);
 if(initial)$('event').value=initial.path;
 if(initialSelection){
  initialSelection=false;const query=new URLSearchParams(location.search),hash=query.get('checkpoint');
  const matches=hash?index.models.map((m,i)=>({m,i})).filter(({m})=>m.sha256.startsWith(hash)):[];
  if(matches.length===1)$('model').value=matches[0].i;
  if(index.events.some(e=>e.path===query.get('event')))$('event').value=query.get('event');
  for(const key of ['view','geometry'])if([...$(key).options].some(o=>o.value===query.get(key)))$(key).value=query.get(key);
 }
 await load();
}
$('dataset').onchange=()=>loadDataset().catch(fail);
try{scenes=[new Scene($('oracle')),new Scene($('prediction'))];const requested=new URLSearchParams(location.search).get('dataset');if(['preview','tails','scale','layouts','responses','impact-responses'].includes(requested)){$('dataset').append(option(requested,requested==='impact-responses'?'Corrected oracle · 128 scenarios · full response model':requested==='responses'?'Corrected oracle · experimental response models':requested==='layouts'?'New wall layouts · 48 impacts':requested==='scale'?'Physical scale test · 10,400 chunks':requested==='tails'?'Longer impact recordings · 120 ms':'Causal input experiment'));$('dataset').value=requested}await loadDataset()}catch(e){fail(e)}
