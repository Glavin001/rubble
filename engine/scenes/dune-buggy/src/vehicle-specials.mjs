import {trailerSpec} from './vehicle-catalog.mjs';
/** Structural accessories are separate convex solids; no merged vehicle meshes. */
export function buildSprintBody(p,{box,beam,quad}){
 const f=-p.wheelbase/2,r=p.wheelbase/2,roof=p.cageHeight;
 const tube=(n,a,b,rad=.022,mat='frame')=>beam(n,a,b,rad,mat);
 // Tapered nose, open cockpit and low side pods.
 quad('Sprint nose',[[-.42,.9,f-.30],[.42,.9,f-.30],[.63,1.02,f+.58],[-.63,1.02,f+.58]],.022,'frame');
 for(const s of [-1,1]){
  tube('Sprint lamp support',[s*.43,.91,f-.3],[s*.37,.93,f-.21],.018,'steel');
  tube('Nose flange',[s*.43,.91,f-.3],[s*.65,1.02,f+.58]);
  quad('Cockpit side skin',[[s*.65,p.tireRadius+.09,f+.43],[s*.65,.93,f+.58],[s*.65,.93,.63],[s*.65,p.tireRadius+.09,.63]],.018,'frame');
  tube('Side nerf bar',[s*.65,.64,f+.55],[s*.85,.64,.30],.03,'alloy');
  tube('Nerf rear return',[s*.85,.64,.30],[s*.65,.64,.63],.03,'alloy');
  tube('Wing front support',[s*.54,roof,-.23],[s*.55,roof+.33,-.33],.021,'steel');
  tube('Wing rear support',[s*.54,roof,.72],[s*.55,roof+.45,.65],.021,'steel');
 }
 // Three connected chord panels make a shallow cambered wing.
 for(let i=0;i<3;i++){
  const a=-.55+i*.55,b=a+.55,ya=roof+.30+(a+.55)*.13,yb=roof+.30+(b+.55)*.13;
  quad('Top wing chord '+i,[[-1.0,ya,a],[1.0,ya,a],[1.0,yb,b+.003],[-1.0,yb,b+.003]],.026,'frame');
 }
 for(const s of [-1,1])quad('Wing end plate',[[s*1.0,roof+.29,-.57],[s*1.0,roof+.82,-.57],[s*1.0,roof+.82,1.11],[s*1.0,roof+.50,1.11]],.019,'frame');
 for(const s of [-1,1]){
  tube('Front wing support',[s*.43,.91,f-.3],[s*.52,.66,f-.48],.02,'steel');
  box('Front wing end plate',[.018,.16,.38],[s*.75,.72,f-.49],'frame');
 }
 box('Front wing',[1.52,.03,.36],[0,.65,f-.49],'race');
 for(const s of [-1,1])tube('Rear push bar stay',[s*.5,.7,r+.35],[s*.55,.82,r+.55],.03,'alloy');
 tube('Rear push bar',[-.55,.82,r+.55],[.55,.82,r+.55],.03,'alloy');
 // Wing race marking, each stroke a separate plate attached to its end plate.
 for(const s of [-1,1])for(const dz of [0,.18])box('Wing number stroke',[.012,.25,.055],[s*1.014,roof+.66,.12+dz],'race');
}

export function buildSemiRear(p,{box,beam,quad,cylinder,ring,motion},width,rearCab){
 const r=p.wheelbase/2,t=trailerSpec({...p,vehicle:'semi'}),h=t.hitch,front=h[2]-.04,tail=h[2]+3.8,axle=t.axleZ;
 const tube=(n,a,b,rad=.027,mat='dark')=>beam(n,a,b,rad,mat);
 // Tractor cab back, steps, exhaust stacks and fifth-wheel support.
 box('Cab back',[width*2,p.cageHeight-1.12,.022],[0,(p.cageHeight+1.12)/2,rearCab],'frame');
 for(const s of [-1,1]){
  tube('Fifth wheel rail',[s*.5,p.tireRadius+.20,r+.35],[s*.36,h[1]-.07,h[2]],.055);
  tube('Stack mount',[s*.65,1.02,.72],[s*.78,1.03,.83],.025);
  tube('Exhaust stack',[s*.78,1.03,.83],[s*.78,p.cageHeight+.12,.83],.045,'alloy');
  tube('Exhaust tip',[s*.78,p.cageHeight+.12,.83],[s*.78,p.cageHeight+.12,.94],.045,'alloy');
  box('Cab step',[.32,.06,.55],[s*.78,.66,-.1],'alloy');
  tube('Step mount',[s*.65,p.tireRadius+.09,.05],[s*.78,.66,.05],.025);
 }
 box('Fifth wheel plate',[.8,.08,.52],[0,h[1]-.055,h[2]],'steel');
 // The kingpin is the trailer root; the rig rotates all tagged meshes about h.
 motion('trailer');
 box('Trailer kingpin',[.09,.15,.09],h,'steel');
 for(const s of [-1,1])tube('Trailer drawbar',h,[s*.65,1.14,front+.45],.05,'steel');
 for(const s of [-1,1]){
  box('Trailer frame rail',[.09,.14,tail-front],[s*.65,1.08,(tail+front)/2],'steel');
  for(let i=0;i<=8;i++){
   const z=front+i*(tail-front)/8;
   tube('Trailer stake',[s*.87,1.17,z],[s*.87,1.42,z],.018,'alloy');
  }
  tube('Trailer side rail',[s*.87,1.42,front],[s*.87,1.42,tail],.022,'frame');
  box('Trailer mud flap',[.32,.24,.022],[s*t.halfTrack,.42,axle+.52],'rubber');
  tube('Trailer flap hanger',[s*.65,1.08,axle+.52],[s*t.halfTrack,.54,axle+.52],.018,'steel');
 }
 for(let i=0;i<=8;i++)box('Trailer crossmember',[1.76,.07,.07],[0,1.135,front+i*(tail-front)/8],'steel');
 // Deck planks have small seams and bear directly on the crossmembers.
 for(let i=0;i<8;i++)box('Trailer deck plank',[.215,.045,tail-front+.03],[(i-3.5)*.218,1.185,(tail+front)/2],'race');
 for(const s of [-1,1]){
  tube('Trailer spring hanger',[s*.65,1.06,axle-.3],[s*.65,.61,axle],.043,'steel');
  tube('Trailer rear hanger',[s*.65,1.06,axle+.3],[s*.65,.61,axle],.036,'steel');
  for(let i=0;i<4;i++)box('Trailer leaf spring',[.075,.018,.7-i*.1],[s*.65,.64+i*.016,axle],'dark');
  tube('Trailer axle saddle',[s*.65,.69,axle],[s*.65,t.wheelRadius,axle],.042,'steel');
 }
 tube('Trailer axle',[-t.halfTrack,t.wheelRadius,axle],[t.halfTrack,t.wheelRadius,axle],.048,'steel');
 for(const s of [-1,1]){
  const c=[s*t.halfTrack,t.wheelRadius,axle],label='Trailer '+(s<0?'left':'right');motion('trailerWheel',{center:c,radius:t.wheelRadius});
  ring(label+' tire',t.wheelRadius,.23,.30,c,'rubber','Wheels');
  cylinder(label+' rim',.232,.304,c,'alloy','Wheels');
  cylinder(label+' hub',.09,.34,c,'steel','Wheels');
  for(let i=0;i<32;i++){
   const a=i*Math.PI/16;
   cylinder(label+' tread',.032,.31,[c[0],c[1]+Math.cos(a)*.42,c[2]+Math.sin(a)*.42],'rubber','Wheels');
  }
 }
 motion('trailer');
 box('Trailer rear bumper',[1.8,.12,.10],[0,1.1,tail],'steel');
 for(const s of [-1,1])box('Trailer rear lamp',[.24,.055,.025],[s*.67,1.11,tail+.055],'red');
 motion(null);
}
