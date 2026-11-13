// The comparison page: one scenario under several destruction approaches, side by side.
// The catalogue, each scenario's scene pack, each approach's lab settings, the script and
// every verdict number are Rust's (engine-world `rubble::play::compare`); each column is a
// lab run in its own World on one shared device (lab-world.mjs), drawn by the lab's
// renderer. This file lays out the columns, paces the steps and prints the numbers.
//   ?scene=balcony|pillar|speed_ladder|equal_momentum  &approaches=a,b,...  &speed=0.25
import {createLab,labDevice} from './lab-world.mjs';
import {createRenderer} from './render.mjs';
import {compareCatalog,comparePack,compareDef} from './wasm.mjs';

const $=id=>document.getElementById(id);
const status=$('status');
const say=t=>{status.textContent=t;status.classList.remove('error')};
const fail=e=>{console.error(e);status.textContent=String(e?.message??e);status.classList.add('error')};
const query=new URLSearchParams(location.search);

const catalog=compareCatalog();
const scenarioPick=$('scenario');
for(const s of catalog.scenarios)scenarioPick.add(new Option(s.title,s.name));
scenarioPick.value=catalog.scenarios.some(s=>s.name===query.get('scene'))?query.get('scene'):catalog.scenarios[0].name;
const wanted=(query.get('approaches')??'').split(',').filter(Boolean);
for(const a of catalog.approaches){
 const label=document.createElement('label'),box=document.createElement('input');
 box.type='checkbox';box.value=a.name;box.checked=!wanted.length||wanted.includes(a.name);box.onchange=()=>load();
 label.append(box,a.title);$('approaches').append(label);
}
if(query.get('speed'))$('speed').value=query.get('speed');
const scenario=()=>catalog.scenarios.find(s=>s.name===scenarioPick.value);
const chosen=()=>[...$('approaches').querySelectorAll('input:checked')].map(b=>catalog.approaches.find(a=>a.name===b.value));

let columns=[],running=true,generation=0,last=0,credit=0;

const fmt=(x,d=1)=>Number(x).toFixed(d);
// A column's card: the verdict's headline for the scenario, then its numbers.
function headline(name,v){
 if(name==='balcony')return v.fell?`Snapped ${fmt(v.firstBreak??0,2)} s in`:v.bondsBroken?'Cracked, still hanging':'Holds';
 if(name==='pillar')return v.fell?'Canopy came down':'Canopy still up';
 const through=v.shots.filter(s=>s.through).length;
 return v.shots.length?`${through} of ${v.shots.length} shots went through`:'Waiting for the first shot';
}
function card(col,v,stats){
 const rows=[];
 for(const s of v.shots){
  const moving=s.launchSpeed>0;
  const down=s.wallFell?' · wall down':'';
  const what=(!moving?`${s.broke} bonds`:s.through?`through at ${fmt(s.along,0)} m/s · ${s.broke} bonds`:`stopped · ${s.broke} bonds`)+down;
  rows.push([s.label,what]);
 }
 rows.push(['Bonds broken',`${v.bondsBroken} of ${v.bonds}`]);
 col.card.querySelector('.headline').textContent=headline(scenario().name,v);
 col.card.querySelector('table').replaceChildren(...rows.map(([k,x])=>{const tr=document.createElement('tr'),a=document.createElement('td'),b=document.createElement('td');a.textContent=k;b.textContent=x;tr.append(a,b);return tr}));
 const ms=stats.gpuStepMs;
 col.card.querySelector('.perf').textContent=`${stats.bodies} bodies · ${stats.contacts} contacts${ms!=null?` · ${fmt(ms,2)} ms GPU per step (shared GPU: not a benchmark)`:''}`;
}

function dispose(){
 for(const c of columns){c.renderer?.dispose();c.tracker?.free();c.sim?.dispose()}
 columns=[];$('columns').replaceChildren();
}

async function load(){
 const mine=++generation,sc=scenario(),approaches=chosen();
 $('caption').textContent=sc.caption;
 history.replaceState(null,'',`?scene=${sc.name}&approaches=${approaches.map(a=>a.name).join(',')}&speed=${$('speed').value}`);
 dispose();
 $('columns').style.setProperty('--n',String(Math.max(1,approaches.length)));
 say(`Building ${sc.title}…`);
 try{
  const device=await labDevice(),pack=comparePack(sc.name),built=[];
  for(const a of approaches){
   const el=document.createElement('section');el.className='col'+(a.name==='rubble'?' ours':'');
   el.innerHTML='<h2></h2><p class="blurb"></p><canvas></canvas><div class="card"><p class="headline">…</p><table></table><div class="perf"></div></div>';
   el.querySelector('h2').textContent=a.title;el.querySelector('.blurb').textContent=a.blurb;
   $('columns').append(el);
   const sim=await createLab(compareDef(sc.name,a.name),pack);
   if(mine!==generation){sim.dispose();return}
   const col={approach:a,sim,el,card:el.querySelector('.card'),due:0};
   // A click shoots the scene's ball along the same ray in every column.
   col.renderer=await createRenderer(device,el.querySelector('canvas'),sim,sim.scene,ray=>{for(const c of columns)c.sim.shoot(ray.origin,ray.direction)});
   col.tracker=sim.world.compareTrack(sim.lab,sc.name);
   sim.world.setStressView($('view').value==='3');col.renderer.colorMode=Number($('view').value);
   built.push(col);
  }
  if(mine!==generation){for(const c of built){c.renderer.dispose();c.tracker.free();c.sim.dispose()}return}
  columns=built;
  for(const c of columns)c.sim.world.compareQueueScript(c.sim.lab,c.tracker);
  credit=0;last=0;running=true;$('pause').textContent='Pause';
  say(`${sc.title}: ${approaches.length} approaches, the same script in each. Click a view to fire the scene's ball in all of them; drag to orbit.`);
 }catch(e){fail(e)}
}

function frame(now){
 requestAnimationFrame(frame);
 if(!columns.length)return;
 const sc=scenario(),dt=last?Math.min(.1,(now-last)/1000):0;last=now;
 try{
  if(running){
   credit+=dt*Number($('speed').value)*sc.physicsHz;
   const steps=Math.min(8,Math.floor(credit));credit-=steps;
   for(const c of columns){c.due=Math.min(16,c.due+steps);c.due-=c.sim.tick(c.due)}
  }else for(const c of columns)c.sim.poll();
  for(const c of columns){
   c.sim.world.compareObserve(c.sim.lab,c.tracker);
   card(c,c.sim.world.compareVerdict(c.sim.lab,c.tracker),c.sim.stats());
   c.renderer.draw(false);
  }
 }catch(e){fail(e);columns=[]}
}

scenarioPick.onchange=()=>load();
// The view: stress asks each World for its chunks' load over strength from the next step on.
$('view').onchange=()=>{const m=Number($('view').value);$('legend').style.visibility=m===3?'visible':'hidden';for(const c of columns){c.sim.world.setStressView(m===3);c.renderer.colorMode=m}};
$('run').onclick=()=>load();
$('pause').onclick=()=>{running=!running;$('pause').textContent=running?'Pause':'Resume'};
// Checks and recordings: the columns' verdicts, and stepping without the frame clock.
window.__compare={
 get ready(){return columns.length>0},
 verdicts:()=>columns.map(c=>({approach:c.approach.name,...c.sim.world.compareVerdict(c.sim.lab,c.tracker)})),
 async step(n){running=false;for(const c of columns){for(let i=0;i<n;i++){await c.sim.step(1);c.sim.world.compareObserve(c.sim.lab,c.tracker)}}},
};
load().then(()=>requestAnimationFrame(frame));
