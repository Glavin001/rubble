import * as T from 'three';
import { buildVehicleBody } from './vehicle-body.mjs';
import { vehicleById, vehicleFields, vehicleLift, mechanicalParameters, tireWidthScale } from './vehicle-catalog.mjs';
import { cornerHardpoints } from './vehicle-rig.mjs';
import { materialVolume } from './construction.mjs';

export const defaults = { wheelbase: 2.62, track: 1.88, tireRadius: .395, cageHeight: 1.78, tubeRadius: .026, seed: 17 };
export const materials = {
  frame:{color:'#344a4b',metalness:.72,roughness:.36,density:7850},
  orange:{color:'#f36b29',metalness:.52,roughness:.31,density:7850},
  rubber:{color:'#202224',metalness:0,roughness:.92,density:1100},
  alloy:{color:'#b0afa4',metalness:.9,roughness:.27,density:2700},
  steel:{color:'#69737a',metalness:.92,roughness:.25,density:7850},
  dark:{color:'#242a2d',metalness:.65,roughness:.45,density:7850},
  // Lamp and mirror bodies are moulded shells, not solid steel: a solid
  // 'dark' lamp weighed 12 kg on a 1 cm tube (see vehicle destruction tests).
  housing:{color:'#242a2d',metalness:.35,roughness:.5,density:1000},
  seat:{color:'#242729',metalness:0,roughness:.93,density:140},
  belt:{color:'#d96b35',metalness:0,roughness:.9,density:1200},
  red:{color:'#b62e25',metalness:.25,roughness:.38,density:1800},
  glass:{color:'#203c47',metalness:.65,roughness:.09,density:2500},
  race:{color:'#eee7d4',metalness:.3,roughness:.4,density:1200},
  lens:{color:'#e7e3c7',metalness:.25,roughness:.14,density:2400},
};

/** All measurements are metres. X=width, Y=up, negative Z=front.
 * Each new solid is partitioned against all earlier solids. The removed shared
 * volume becomes an exact matching contact surface; parts are never merged.
 * Contact graph edges are measured from real solid intersections before cutting.
 */
export function buildBuggy(wasm, input={}, progress=(phase="",percent=0)=>{}, audit=false, options={preview:false,lowDetail:true}) {
 const live=options.live;
 live?.begin();
 const preview=options.preview===true;
 const lowDetail=preview&&options.lowDetail!==false;
 const segments=n=>lowDetail?Math.max(6,Math.min(n,Math.ceil(n/3))):n;
 const parameters={...defaults,...input};
 if(!vehicleById(parameters.vehicle))throw Error('Unknown vehicle');
 for(const [k,,min,max] of [...vehicleFields(parameters.vehicle),['tubeRadius','Tube radius',.022,.034]])if(!Number.isFinite(parameters[k])||parameters[k]<min||parameters[k]>max)throw Error(`Invalid ${k}`);
 const p=mechanicalParameters(parameters),lift=vehicleLift(parameters),tireWidth=tireWidthScale(parameters);
 const {Manifold:M,Mesh}=wasm||{};
 const solids=[], originals=[], parts=[], joints=[], skipped=[]; let candidates=0, trims=0;
 const f=-p.wheelbase/2, r=p.wheelbase/2, y=p.tireRadius, w=.65, floor=y+.09, roof=p.cageHeight, tr=p.tubeRadius;
 const v=a=>new T.Vector3(...a);
 const overlap=(a,b)=>a.min.every((x,i)=>x<=b.max[i]+1e-8&&a.max[i]>=b.min[i]-1e-8);
 const cube=live?live.cube:(size,c,rot=[0,0,0],radius=0)=>{if(radius){const h=size.map(x=>x/2),r=Math.min(radius,...h.map(x=>x*.45)),points=[];for(const x of [-1,1])for(const y of [-1,1])for(const z of [-1,1])for(let k=0;k<3;k++)points.push([x*(h[0]-(k===0?0:r)),y*(h[1]-(k===1?0:r)),z*(h[2]-(k===2?0:r))]);const a=M.hull(points),b=a.rotate(rot),d=b.translate(c);a.delete();b.delete();return d;}const a=M.cube(size,true),b=a.rotate(rot),d=b.translate(c);a.delete();b.delete();return d;};
 const cyl=live?live.cyl:(radius,len,c,rot=[90,0,0],n=32)=>{const a=M.cylinder(len,radius,radius,segments(n),true),b=a.rotate(rot),d=b.translate(c);a.delete();b.delete();return d;};
 const ring=live?live.ring:(out,inside,len,c,rot=[0,90,0],n=64)=>{const a=M.cylinder(len,out,out,segments(n),true),b=M.cylinder(len+.02,inside,inside,segments(n),true),d=a.subtract(b),e=d.rotate(rot),g=e.translate(c);[a,b,d,e].forEach(x=>x.delete());return g;};
 const beam=live?live.beam:(a,b,rad=tr,hollow=false)=>{const A=v(a),B=v(b),d=B.clone().sub(A),q=new T.Quaternion().setFromUnitVectors(new T.Vector3(0,0,1),d.clone().normalize()),matrix=new T.Matrix4().compose(A.add(B).multiplyScalar(.5),q,new T.Vector3(1,1,1));let s=M.cylinder(d.length(),rad,rad,segments(32),true);if(hollow){const h=M.cylinder(d.length()+.01,rad-.003,rad-.003,segments(32),true),t=s.subtract(h);s.delete();h.delete();s=t;}const out=s.transform(matrix.elements);s.delete();return out;};
 const fromGeom=(geo)=>{const a=geo;const mesh=new Mesh({numProp:3,vertProperties:new Float32Array(a.attributes.position.array),triVerts:a.index?new Uint32Array(a.index.array):Uint32Array.from({length:a.attributes.position.count},(_,i)=>i)});mesh.merge();const s=new M(mesh);geo.dispose();return s;};
 let binding=null;
 const motion=(corner,role,extra={})=>{binding=role?{corner,role,...extra}:null};
 const add=(name,system,mat,solid,meta=binding,functionality=null)=>{
  if(lift){const shifted=solid.translate([0,lift,0]);solid.delete();solid=shifted;if(meta?.endpoints)meta={...meta,endpoints:meta.endpoints.map(a=>[a[0],a[1]+lift,a[2]])};}
  if(live)return live.add(name,system,mat,solid,meta,functionality);
  // S\(A∪B) equals (S\A)\B: uncut tools give the same partition
  // without re-intersecting fragile, already-cut thin panel surfaces.
  const original=parameters.vehicle&&parameters.vehicle!=='buggy'?solid.translate([0,0,0]):null;
  const id=`${system.toLowerCase()}-${String(parts.length).padStart(4,'0')}`, bounds=solid.boundingBox(); const contacts=[];
  for(let i=0;!preview&&i<solids.length;i++)if(overlap(bounds,parts[i].bounds)){
   candidates++; const cut=solid.intersect(originals[i]??solids[i]),volume=cut.volume();
   if(volume>1e-12){const b=cut.boundingBox(),other=originals[i]??solids[i],next=solid.subtract(other);
    // Exact shared surface of the two solids: (|A| + |B'| - |A u B'|) / 2.
    // Bond areas come from this, not from coplanar faces of simplified hulls.
    const union=other.add(next),interfaceAreaM2=Math.max(0,(other.surfaceArea()+next.surfaceArea()-union.surfaceArea())/2);union.delete();
    contacts.push({a:parts[i].id,b:id,anchor:b.min.map((x,k)=>(x+b.max[k])/2),kind:system==='Suspension'?'mechanical':'fixed',overlapRemovedM3:volume,interfaceAreaM2});solid.delete();solid=next;trims++;} cut.delete();
  }
  if(solid.isEmpty()||solid.volume()<1e-12){skipped.push(name);solid.delete();original?.delete();return null;}
  const volume=solid.volume(), mesh=solid.getMesh(), pos=new Float32Array(mesh.numVert*3);
  for(let i=0;i<mesh.numVert;i++)for(let k=0;k<3;k++)pos[i*3+k]=mesh.vertProperties[i*mesh.numProp+k];
  const box=solid.boundingBox(), center=box.min.map((x,i)=>(x+box.max[i])/2);
  parts.push({id,name,system,material:mat,motion:meta,functionality,position:pos,indices:new Uint32Array(mesh.triVerts),bounds:box,center,volume,mass:materialVolume(name,volume,solid.surfaceArea())*materials[mat].density});if(preview){solid.delete();original?.delete()}else{solids.push(solid);originals.push(original)}joints.push(...contacts);return id;
 };
 const b=(name,a,z,mat='frame',rad=tr,system='Frame',hollow=true,functionality=null)=>add(name,system,mat,beam(a,z,rad,hollow),binding?{...binding,endpoints:[a,z],radius:rad}:null,functionality);
 const box=(name,size,c,mat='dark',system='Cabin',rot,functionality=null)=>add(name,system,mat,cube(size,c,rot,mat==='seat'?.035:0),binding,functionality);
 const bolt=(name,c,rot=[90,0,0],system='Hardware',rad=.012)=>add(name,system,'steel',cyl(rad,.012,c,rot,6));
 // Split cage and chassis: every rail, diagonal and crossmember is its own tube.
 for(const s of [-1,1]){
  const x=s*w;
  const n={A:[s*.43,f-.26,floor+.06],B:[x,f+.43,floor],C:[x,.58,floor],D:[s*.5,r+.35,floor+.11]};
  // Coordinate helper: longitudinal positions are explicitly in Z.
  const A=[s*.43,floor+.06,f-.26],B=[x,floor,f+.43],C=[x,floor,.63],D=[s*.5,floor+.11,r+.35];
  const E=[x,1.02,f+.58],F=[x,1.02,.72],G=[s*.54,roof,-.23],H=[s*.54,roof,.72],I=[s*.5,1.2,r+.22];
  const label=s<0?'Left':'Right';
  [[A,B],[B,C],[C,D],[B,E],[E,F],[F,C],[E,G],[G,H],[H,F],[F,I],[I,D],[D,H],[B,F],[C,E]].forEach((pair,i)=>b(`${label} cage member ${i+1}`,pair[0],pair[1]));
  b(`${label} hood rail`,A,E);b(`${label} front bumper upright`,A,[s*.43,.91,f-.3]);

 }
 for(const [z,height,width,label] of [[f-.26,floor+.06,.43,'Front chassis'],[f+.43,floor,w,'Pedal bulkhead'],[-.1,floor,w,'Seat front'],[.58,floor,w,'Seat rear'],[.72,1.02,w,'Harness bar'],[.72,roof,.54,'Rear roof'],[-.23,roof,.54,'Front roof'],[r+.35,floor+.11,.5,'Rear chassis'],[r+.22,1.2,.5,'Engine upper'],[f-.3,.91,.43,'Front bumper']]) b(`${label} crossmember`,[-width,height,z],[width,height,z],'frame',tr,'Frame',true,label==='Rear chassis'?'chassis':null);
 b('Roof diagonal',[-.54,roof,-.23],[.54,roof,.72]);
 b('Rear firewall diagonal',[-w,floor,.58],[w,1.02,.72]);
 b('Rear bumper',[-.57,.71,r+.46],[.57,.71,r+.46]);
 for(const s of [-1,1])b('Rear bumper stay',[s*.5,floor+.11,r+.35],[s*.57,.71,r+.46]);
 progress('Suspension',18);
 // Long-travel double wishbones, individual hubs, steering links and coilovers.
 for(const z of [f,r])for(const s of [-1,1]){
  const front=z===f,label=`${front?'Front':'Rear'} ${s<0?'left':'right'}`, wx=s*p.track/2;
  const corner=(front?'f':'r')+(s<0?'l':'r');
  const hardpoints=cornerHardpoints(p,corner);
  const inner=front?.39:.47, pivotY=hardpoints.lowerPivot[1], lower=hardpoints.lower, upper=hardpoints.upper;
  for(const dz of [-.23,.23]){
   const mount=[s*inner,pivotY,z+dz];
   motion(corner,'lowerArm');
   b(`${label} lower wishbone`,mount,lower,'orange',.027,'Suspension',true);
   motion(null,null);
   b(`${label} chassis mount`,mount,[s*(front?.55:.58),floor,z+(front?.4:-.55)],'frame',.025,'Suspension');
   bolt(`${label} pivot bolt`,mount,[0,0,0],'Suspension',.021);
   motion(corner,'upperArm');
   b(`${label} upper wishbone`,[s*inner,pivotY+.24,z+dz*.8],upper,'orange',.021,'Suspension');
   motion(null,null);
   b(`${label} upper mount`,[s*inner,pivotY+.24,z+dz*.8],[s*inner,pivotY,z+dz],'dark',.024,'Suspension');
  }
  motion(corner,'upright');
  b(`${label} upright`,lower,upper,'dark',.045,'Suspension',false);
  const top=hardpoints.shockTop,bottom=hardpoints.shockBottom;
  motion(null,null);
  b(`${label} shock tower`,top,[s*.55,floor,z+(front?.43:-.4)],'frame',.032,'Suspension');
  const V=v(bottom).sub(v(top)),L=V.length(),dir=V.clone().normalize(),point=t=>v(top).addScaledVector(dir,t).toArray();
  motion(corner,'damper');
  b(`${label} damper body`,top,point(L*.67),'alloy',.039,'Suspension',false);
  motion(corner,'piston');
  b(`${label} chrome piston`,point(L*.6),bottom,'steel',.016,'Suspension',false);
  motion(corner,'spring');
  if(live)add(`${label} coil spring`,'Suspension','orange',live.spring(top,bottom));
  else {
  const axis=new T.Quaternion().setFromUnitVectors(new T.Vector3(0,1,0),dir),points=[];
  for(let k=0;k<=240;k++){const t=k/240,a=t*Math.PI*2*10;points.push(new T.Vector3(Math.cos(a)*.059,t*L*.68,Math.sin(a)*.059).applyQuaternion(axis).add(v(top)));}
  const springSegments=lowDetail?80:320;
  const curve=new T.CatmullRomCurve3(points),geo=new T.TubeGeometry(curve,springSegments,.009,8,false);
  // TubeGeometry has open ends; close with matching disk fans for a watertight spring.
  const positions=Array.from(geo.attributes.position.array),idx=Array.from(geo.index.array);
  for(const [off,reverse] of [[0,false],[springSegments*9,true]]){const c=positions.length/3;const pp=points[off===0?0:240];positions.push(pp.x,pp.y,pp.z);for(let j=0;j<8;j++)idx.push(c,off+(reverse?j+1:j),off+(reverse?j:j+1));}
  geo.setAttribute('position',new T.Float32BufferAttribute(positions,3));geo.setIndex(idx);
  add(`${label} coil spring`,'Suspension','orange',fromGeom(geo));
  }
  for(const t of [.02,L*.68]){motion(corner,t===.02?'topSeat':'bottomSeat');b(`${label} spring seat`,point(t-.012),point(t+.012),'dark',.071,'Suspension',false);}
  motion(corner,'shockEye');
  b(`${label} shock eye`,point(L-.06),lower,'steel',.023,'Suspension',false);
  motion(corner,'tieRod');
  b(`${label} tie rod`,[s*.18,y+.07,z+.1],[wx-s*.15,y+.06,z],'steel',.012,'Suspension',false);
  motion(corner,'axle');
  b(`${label} axle`,[0,y,z],[wx,y,z],'steel',.025,'Drivetrain',false);
  motion(corner,'cvBoot');
  for(let i=0;i<6;i++)add(`${label} CV boot rib ${i+1}`,'Drivetrain','rubber',cyl(.047-Math.abs(i-2.5)*.003,.017,[wx-s*(.21+i*.017),y,z],[0,90,0]));
  motion(corner,'hub',{component:'hub'});
  add(`${label} hub`,'Wheels','dark',cyl(.075,.21,[wx-s*.1,y,z],[0,90,0]));
  motion(corner,'hub',{component:'rotor'});
  add(`${label} brake rotor`,'Wheels','steel',ring(.151,.068,.013,[wx-s*.125,y,z]));
  motion(corner,'knuckle',{component:'caliper'});
  box(`${label} brake caliper`,[.07,.1,.06],[wx-s*.13,y+.115,z+.025],'red','Wheels');
  motion(corner,'wheel');
  // Hollow rim, six independent spokes, beadlock rings, lug nuts.
  add(`${label} rim barrel`,'Wheels','alloy',ring(.235,.214,.275,[wx,y,z]));
  for(const side of [-1,1]){
   const face=wx+side*.138;
   add(`${label} beadlock ${side}`,'Wheels','alloy',ring(.252,.219,.019,[face,y,z]));
   for(let k=0;k<16;k++){const a=k*Math.PI/8;bolt(`${label} beadlock bolt ${side}-${k}`,[face+side*.012,y+Math.cos(a)*.238,z+Math.sin(a)*.238],[0,90,0],'Wheels',.007);}
  }
  const face=wx+s*.09;
  b(`${label} hub extension`,[wx-s*.01,y,z],[face,y,z],'steel',.035,'Wheels',false);
  add(`${label} wheel center`,'Wheels','alloy',cyl(.087,.04,[face,y,z],[0,90,0]));
  for(let k=0;k<6;k++){const a=k*Math.PI/3;b(`${label} spoke ${k+1}`,[face,y+Math.cos(a)*.063,z+Math.sin(a)*.063],[face-s*.036,y+Math.cos(a)*.226,z+Math.sin(a)*.226],'alloy',.027,'Wheels',false);bolt(`${label} lug nut ${k+1}`,[face+s*.024,y+Math.cos(a)*.06,z+Math.sin(a)*.06],[0,90,0],'Wheels',.013);}
  add(`${label} center cap`,'Wheels','dark',cyl(.045,.018,[face+s*.026,y,z],[0,90,0]));
  // Revolved carcass with a real inner void. Bead rests against rim.
  const rr=parameters.tireRadius,profile=[[.235,-.135],[.26,-.169],[rr-.075,-.183],[rr-.029,-.148],[rr-.012,-.10],[rr-.007,0],[rr-.012,.10],[rr-.029,.148],[rr-.075,.183],[.26,.169],[.235,.135]].map(([r,z])=>[r,z*tireWidth]);
  const tire0=live?live.revolve(profile,96):M.revolve([profile],segments(96)),tire1=tire0.rotate([0,90,0]),tire=tire1.translate([wx,y,z]);tire0.delete();tire1.delete();add(`${label} tire carcass`,'Wheels','rubber',tire);
  for(let k=0;k<32;k++)for(let row=-1;row<=1;row++){
   const a=(k+(row===0?.42:0))*Math.PI/16,rad=rr-.004,center=[wx+row*.097*tireWidth,y+Math.cos(a)*rad,z+Math.sin(a)*rad];
   // Angularly spaced lugs cut into the carcass then partitioned: touching, no penetration.
   add(`${label} tread ${k+1}.${row+2}`,'Wheels','rubber',cube([.086*tireWidth,.045,.061],center,[a*180/Math.PI,0,row*12],.006));
  }
 }
 motion(null,null);
 progress('Cabin',55);
 // Thin individual floor plates and two bucket seats with bolsters and webbing.
 for(const s of [-1,1]){
  const x=s*.325;
  box(`${s<0?'Driver':'Passenger'} floor`,[.59,.004,1.42],[x,floor-.005,-.07],'dark');
  for(const z of [-.18,.4])box('Seat mounting rail',[.38,.05,.06],[x,floor+.02,z],'steel');
  box('Seat pan',[.45,.11,.57],[x,floor+.09,.14],'seat');
  box('Seat back',[.43,.64,.105],[x,floor+.42,.43],'seat','Cabin',[12,0,0]);
  b('Headrest support',[x,floor+.65,.47],[x,floor+.77,.5],'steel',.018,'Cabin',false);
  box('Head restraint',[.25,.21,.115],[x,floor+.81,.50],'seat');
  for(const side of [-1,1]){
   box('Seat base side bolster',[.076,.14,.53],[x+side*.207,floor+.17,.14],'seat');
   box('Seat shoulder bolster',[.07,.51,.15],[x+side*.203,floor+.46,.435],'seat','Cabin',[12,0,side*-5]);
   box('Shoulder harness',[.047,.60,.014],[x+side*.094,floor+.43,.364],'belt','Cabin',[12,0,side*3]);
   box('Lap harness',[.21,.012,.046],[x+side*.112,floor+.151,.12],'belt','Cabin',[0,0,side*-5]);
  }
  box('Harness buckle',[.074,.019,.067],[x,floor+.16,.12],'steel');
  // Back straps physically join the seatback to the harness crossbar.
  for(const side of [-1,1])b('Harness attachment',[x+side*.094,floor+.60,.48],[x+side*.094,1.02,.72],'belt',.014,'Cabin',false);
 }
 // Faceted hood, inset dashboard, switches, gauges, pedal assembly.
 for(const s of [-1,1])for(const dz of [-.17,.28]){const t=(dz+.26)/.84; b('Hood mounting standoff',[s*(.43+.22*t),floor+.06+(1.02-floor-.06)*t,f+dz],[s*.36,.965+(dz-.06)*.176,f+dz],'steel',.014,'Body',false);}
 box('Nose panel',[.83,.017,.62],[0,.965,f+.06],'frame','Body',[-10,0,0]);
 box('Hood accent stripe',[.12,.0016,.60],[-.16,.9733,f+.06],'orange','Body',[-10,0,0]);
 for(const s of [-1,1])for(const z of [f-.17,f+.28])bolt('Hood fastener',[s*.36,.971+(z-f-.06)*.176,z],[90,0,0],'Body',.008);
 b('Dashboard cross support',[-w,.955,f+.58],[w,.955,f+.58],'frame',tr,'Cabin');
 box('Dashboard',[1.32,.19,.048],[0,.955,f+.57],'dark','Cabin',[-15,0,0]);
 for(const x of [-.32,-.13]){
  add('Instrument bezel','Cabin','alloy',cyl(.065,.02,[x,.978,f+.597],[0,0,0],48));
  add('Gauge face','Cabin','dark',cyl(.057,.006,[x,.978,f+.608],[0,0,0],48));
  b('Gauge needle',[x-.023,.96,f+.612],[x+.027,1.0,f+.612],'lens',.0025,'Cabin',false);
 }
 for(let i=0;i<4;i++)bolt('Dashboard toggle',[.16+i*.065,.977,f+.59],[0,0,0],'Cabin',.009);
 const stA=[-.325,.77,f+.6],stB=[-.325,1.08,-.31];
 b('Steering column bracket',[-.325,.955,f+.58],stA,'steel',.024,'Cabin');
 motion(null,'steering');
 b('Steering column',stA,stB,'steel',.018,'Cabin',false);
 const steer0=ring(.18,.158,.023,[0,0,0],[0,0,0],72);const q=new T.Quaternion().setFromUnitVectors(new T.Vector3(0,0,1),v(stB).sub(v(stA)).normalize());const mm=new T.Matrix4().compose(v(stB),q,new T.Vector3(1,1,1));const steer=steer0.transform(mm.elements);steer0.delete();add('Steering wheel grip','Cabin','rubber',steer);
 for(let k=0;k<3;k++){const a=k*Math.PI*2/3;const end=new T.Vector3(Math.cos(a)*.167,Math.sin(a)*.167,0).applyQuaternion(q).add(v(stB));b('Steering wheel spoke',stB,end.toArray(),'alloy',.013,'Cabin',false);}
 add('Steering center','Cabin','dark',cyl(.042,.033,stB,[0,0,0]));
 motion(null,null);
 for(let i=0;i<3;i++) {b('Pedal arm',[-.5+i*.12,floor,-.77],[-.5+i*.12,floor+.18,-.65],'steel',.013,'Cabin',false);box('Pedal pad',[.075,.025,.11],[-.5+i*.12,floor+.18,-.65],'alloy','Cabin',[-35,0,0]);}
 box('Shifter mounting plate',[.12,.012,.18],[.025,floor,.05],'steel','Cabin');
 b('Gear lever',[.025,floor,.05],[.025,.8,-.1],'steel',.012,'Cabin',false);const knob=live?live.sphere(.032,32):M.sphere(.032,32);add('Gear knob','Cabin','rubber',knob.translate([.025,.8,-.1]));knob.delete();
 progress('Engine & details',72);
 // Rear-mounted boxer-style engine: block, individual cooling fins, heads and exhaust.
 box('Transmission case',[.25,.23,.55],[0,floor+.08,r-.16],'alloy','Drivetrain',undefined,'driveline');
 box('Engine crankcase',[.45,.32,.34],[0,floor+.23,r+.12],'dark','Drivetrain',undefined,'engine');
 for(const s of [-1,1]){
  b('Engine mounting strut',[s*.18,floor+.06,r],[s*.5,floor+.11,r+.35],'steel',.025,'Drivetrain');
  for(let bank=0;bank<2;bank++){
   const z=r+.015+bank*.18;
   add('Cylinder barrel','Drivetrain','dark',cyl(.092,.29,[s*.31,floor+.26,z],[0,90,0]));
   for(let i=0;i<8;i++)add(`Cooling fin ${i+1}`,'Drivetrain','alloy',cyl(.109,.009,[s*(.22+i*.027),floor+.26,z],[0,90,0]));
  }
  box('Cylinder head',[.09,.23,.38],[s*.49,floor+.26,r+.1],'alloy','Drivetrain');
  box('Valve cover',[.025,.17,.29],[s*.548,floor+.26,r+.1],'dark','Drivetrain');
  b('Exhaust header',[s*.47,floor+.20,r+.21],[s*.42,floor+.10,r+.43],'steel',.026,'Drivetrain');
  b('Exhaust collector',[s*.42,floor+.10,r+.43],[s*.25,floor+.14,r+.51],'steel',.027,'Drivetrain');
 }
 b('Exhaust muffler',[-.3,floor+.14,r+.51],[.3,floor+.14,r+.51],'alloy',.065,'Drivetrain',false);
 b('Tailpipe',[.27,floor+.14,r+.51],[.48,floor+.19,r+.61],'steel',.029,'Drivetrain',true);
 add('Fan shroud','Drivetrain','dark',cyl(.215,.13,[0,floor+.52,r+.12],[0,0,0],64));
 add('Fan pulley','Drivetrain','alloy',cyl(.075,.04,[0,floor+.52,r+.20],[0,0,0],48));
 for(const s of [-1,1]){
  b('Air intake riser',[s*.3,floor+.3,r+.1],[s*.3,floor+.58,r+.1],'dark',.043,'Drivetrain');
  add('Air filter','Drivetrain','red',cyl(.076,.13,[s*.3,floor+.62,r+.1],[90,0,0]));
  for(let i=0;i<10;i++)add('Filter pleat','Drivetrain','dark',ring(.078,.066,.003,[s*.3,floor+.566+i*.012,r+.1],[90,0,0],32));
 }
 // Rear fuel cell and straps sit on their own saddle mounts.
 for(const x of [-.34,.34]) b('Fuel tank saddle',[x,1.02,.72],[x,1.27,.92],'steel',.026,'Drivetrain');
 add('Fuel tank','Drivetrain','alloy',cyl(.147,.83,[0,1.285,.95],[0,90,0],64));
 for(const x of [-.30,.30])add('Tank strap','Drivetrain','dark',ring(.153,.14,.03,[x,1.285,.95]));
 add('Fuel cap','Drivetrain','dark',cyl(.043,.024,[0,1.432,.95],[90,0,0]));
 // Lamps, mirrors and extinguisher are actual geometry, each with its own mount.
 for(const s of [-1,1]){
  const lampLift=parameters.vehicle&&parameters.vehicle!=='buggy'?.13:0;
  // The bracket starts inside the nose panel (its top is ~.927 here), so it
  // is welded to it rather than grazing it 3 mm above.
  b('Headlamp bracket',[s*.37,.915,f-.21],[s*.40,1.01+lampLift,f-.21],'steel',.013,'Body',false);
  add('Headlight housing','Body','housing',cyl(.087,.065,[s*.40,1.075+lampLift,f-.23],[0,0,0],48));
  add('Headlight lens','Body','lens',cyl(.075,.011,[s*.40,1.075+lampLift,f-.266],[0,0,0],48));
  b('Mirror stalk',[s*.65,1.04,f+.58],[s*.81,1.2,f+.63],'steel',.012,'Body',false);
  box('Mirror housing',[.135,.09,.047],[s*.82,1.22,f+.64],'housing','Body',[0,s*15,0]);
  box('Mirror glass',[.119,.074,.008],[s*.823,1.22,f+.666],'steel','Body',[0,s*15,0]);
  box('Rear lamp mount',[.12,.06,.025],[s*.49,.75,r+.45],'dark','Body');
  box('Rear lamp lens',[.09,.032,.02],[s*.49,.75,r+.469],'red','Body');
 }
 b('Extinguisher bracket',[.62,1.05,.72],[.65,1.32,.72],'steel',.019,'Body',false);
 add('Fire extinguisher','Body','red',cyl(.046,.28,[.67,1.25,.72],[90,0,0]));
 add('Extinguisher neck','Body','steel',cyl(.017,.05,[.67,1.411,.72],[90,0,0]));
 box('Extinguisher handle',[.07,.015,.027],[.67,1.439,.72],'dark','Body');
 if(parameters.vehicle&&parameters.vehicle!=='buggy')buildVehicleBody(parameters.vehicle,p,{
  box:(name,size,c,mat)=>box(name,size,c,mat,'Body'),
  cylinder:(name,r,l,c,mat,system='Body')=>add(name,system,mat,cyl(r,l,c,[0,90,0])),
  ring:(name,r,inner,l,c,mat,system='Body')=>add(name,system,mat,ring(r,inner,l,c,[0,90,0])),
  motion:(role,extra={})=>motion(null,role,extra),
  beam:(name,a,c,r,mat)=>b(name,a,c,mat,r,'Body',false),
  quad:(name,points,thickness,mat)=>{const normal=v(points[1]).sub(v(points[0])).cross(v(points[2]).sub(v(points[0]))).normalize().multiplyScalar(thickness/2);const vertices=[-1,1].flatMap(s=>points.map(a=>v(a).addScaledVector(normal,s).toArray()));add(name,'Body',mat,live?live.poly(vertices):M.hull(vertices));}
 });
 if(live)return live.finish(parameters);
 progress('Checking connections',91);
 if(preview)return {parameters,parts,joints:[],report:{parts:parts.length,triangles:parts.reduce((n,x)=>n+x.indices.length/3,0),joints:null,components:null},quality:'preview',detail:lowDetail?'low':'high',units:'metres',axes:{up:'+Y',front:'-Z'},version:1};
 // Audit intersections independently of construction when requested by offline tests.
 let maxOverlap=0,overlapCount=0;
 if(audit)for(let i=0;i<solids.length;i++)for(let j=0;j<i;j++)if(overlap(parts[i].bounds,parts[j].bounds)){const a=solids[i].intersect(solids[j]),vol=a.volume();maxOverlap=Math.max(maxOverlap,vol);if(vol>1e-7)overlapCount++;a.delete();}
 const adjacent=new Map(parts.map(x=>[x.id,new Set()]));for(const e of joints){adjacent.get(e.a)?.add(e.b);adjacent.get(e.b)?.add(e.a);}
 const visited=new Set(),components=[];
 for(const part of parts)if(!visited.has(part.id)){const queue=[part.id];visited.add(part.id);for(let i=0;i<queue.length;i++)for(const id of adjacent.get(queue[i]))if(!visited.has(id)){visited.add(id);queue.push(id);}components.push(queue);}
 const report={parts:parts.length,joints:joints.length,triangles:parts.reduce((n,x)=>n+x.indices.length/3,0),components:components.length,disconnected:components.slice(1).map(g=>g.map(id=>parts.find(p=>p.id===id).name)),isolated:components.filter(x=>x.length<5).map(g=>g.map(id=>parts.find(p=>p.id===id).name)),skipped,candidates,trims,auditVolumeToleranceM3:1e-7,maxOverlapM3:audit?maxOverlap:null,overlapCount:audit?overlapCount:null,massKg:parts.reduce((n,x)=>n+x.mass,0)};
 solids.forEach(x=>x.delete());originals.forEach(x=>x?.delete());progress('Ready',100);
 return {parameters,parts,joints,report,quality:'final',units:'metres',axes:{up:'+Y',front:'-Z'},version:1};
}
