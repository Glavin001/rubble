import {buildSprintBody,buildSemiRear} from './vehicle-specials.mjs';
/** Separate thin panels and convex frame members; shared by visual/CSG/collider backends. */
export function buildVehicleBody(kind,p,api){
 if(kind==='sprint')return buildSprintBody(p,api);
 const {box,beam,quad}=api;
 const f=-p.wheelbase/2,r=p.wheelbase/2,y=p.tireRadius,floor=y+.09,roof=p.cageHeight,bodyX=kind==='monster'?.88:kind==='rally'?.80:.86;
 const belt=kind==='rally'?1.03:1.12,frontGlass=f+.62,rearCab=kind==='rally'?.88:.63;
 const tube=(name,a,b,rad=.018,mat='dark')=>beam(name,a,b,rad,mat);
 const plate=(name,points,mat='frame',thick=.016)=>quad(name,points,thick,mat);
 // Mount bodywork to known cage hardpoints before adding panel skin.
 for(const s of [-1,1]){
  tube('Body outrigger front',[s*.65,1.02,f+.58],[s*bodyX,belt,frontGlass],.024);
  tube('Body outrigger rear',[s*.65,1.02,.72],[s*bodyX,belt,rearCab],.024);
  tube('Body sill front mount',[s*.65,floor,f+.43],[s*bodyX,floor+.1,frontGlass],.023);
  tube('Body sill rear mount',[s*.65,floor,.63],[s*bodyX,floor+.1,rearCab],.023);
  tube('Body sill',[s*bodyX,floor+.1,frontGlass],[s*bodyX,floor+.1,rearCab],.023);
  tube('Door belt rail',[s*bodyX,belt,frontGlass],[s*bodyX,belt,rearCab],.019);
  plate('Door outer skin',[[s*bodyX,floor+.09,frontGlass],[s*bodyX,belt,frontGlass],[s*bodyX,belt,rearCab],[s*bodyX,floor+.09,rearCab]]);
  box('Door recessed handle',[.022,.027,.14],[s*(bodyX+.018),belt-.10,rearCab-.20],'dark');
  const a=[s*bodyX,belt,frontGlass],b=[s*.59,roof,-.23],c=[s*.59,roof,.72],d=[s*bodyX,belt,rearCab];
  for(const [name,u,v] of [['A pillar',a,b],['Roof side rail',b,c],['B pillar',c,d]])tube(name,u,v,.025,'frame');
  if(kind!=='derby')plate('Side window',[a,b,c,d],'glass',.009);
  else{tube('Door impact rail',[s*(bodyX+.025),belt-.23,frontGlass],[s*(bodyX+.025),belt-.23,rearCab],.04,'alloy');tube('Window guard',a,c,.018,'steel');}
  tube('Window divider',[s*(bodyX+.59)/2,(roof+belt)/2,(frontGlass-.23)/2],[s*bodyX,belt,(frontGlass+rearCab)/2],.011,'dark');
 }
 plate('Roof panel',[[-.61,roof+.02,-.25],[.61,roof+.02,-.25],[.61,roof+.02,.75],[-.61,roof+.02,.75]],'frame',.025);
 if(kind!=='derby')plate('Windshield',[[-bodyX,belt,frontGlass],[bodyX,belt,frontGlass],[.59,roof,-.23],[-.59,roof,-.23]],'glass',.011);
 tube('Windshield lower rail',[-bodyX,belt,frontGlass],[bodyX,belt,frontGlass],.022,'dark');
 tube('Windshield upper rail',[-.59,roof,-.23],[.59,roof,-.23],.021,'frame');
 if(kind!=='derby')plate('Cab rear window',[[-.59,roof,.72],[.59,roof,.72],[bodyX,belt,rearCab],[-bodyX,belt,rearCab]],'glass',.011);
 tube('Rear cab crossbar',[-bodyX,belt,rearCab],[bodyX,belt,rearCab],.023);
 const nose=f-.48,hoodY=kind==='rally'?.97:1.04;
 for(const s of [-1,1]){
  tube('Hood support',[s*.43,floor+.06,f-.26],[s*bodyX,hoodY,nose],.025);
  tube('Hood edge',[s*bodyX,hoodY,nose],[s*bodyX,belt,frontGlass],.019,'frame');
  tube('Front bumper stay',[s*.43,.91,f-.3],[s*(bodyX-.1),.72,nose-.10],.03);
  // Convex arc sections leave wheel wells open instead of spanning the tires.
  for(const z of [f,r])for(let k=0;k<10;k++){
   const a=Math.max(0,k*Math.PI/10-.003),b=Math.min(Math.PI,(k+1)*Math.PI/10+.003),inner=p.tireRadius+.05,outer=inner+.09;
   const point=(t,rad)=>[s*bodyX,y+Math.sin(t)*rad,z+Math.cos(t)*rad];
   plate(`${z===f?'Front':'Rear'} wheel arch ${k+1}`,[point(a,inner),point(a,outer),point(b,outer),point(b,inner)],'frame',.025);
  }
  // Fender supports intersect the arch at its crown and attach to chassis tubes.
  for(const z of [f,r])tube('Fender bracket',[s*(z===f?.43:.5),z===f?floor+.06:floor+.11,z+(z===f?-.26:.35)],[s*bodyX,y+p.tireRadius+.105,z],.019);
  box('Front lamp housing',[.25,.13,.07],[s*(bodyX-.18),hoodY-.11,nose-.008],'dark');
  box('Front lamp lens',[.22,.10,.012],[s*(bodyX-.18),hoodY-.11,nose-.048],'lens');
  tube('Door mirror arm',[s*bodyX,belt,frontGlass+.06],[s*(bodyX+.16),belt+.1,frontGlass+.1],.014);
  box('Door mirror',[.12,.085,.055],[s*(bodyX+.18),belt+.11,frontGlass+.1],'dark');
 }
 plate('Hood panel',[[-bodyX,hoodY,nose],[bodyX,hoodY,nose],[bodyX,belt,frontGlass],[-bodyX,belt,frontGlass]],'frame',.024);
 box('Front fascia',[bodyX*2,.25,.035],[0,hoodY-.135,nose],'frame');
 box('Recessed grille',[bodyX*1.15,.12,.018],[0,hoodY-.14,nose-.025],'dark');
 for(let i=-4;i<=4;i++)box('Grille slat',[.028,.105,.012],[i*bodyX*.13,hoodY-.14,nose-.0395],'alloy');
 tube('Front impact bumper',[-bodyX,.72,nose-.1],[bodyX,.72,nose-.1],kind==='monster'?.055:.04,'dark');
 const tail=r+.52;
 if(kind==='semi'){buildSemiRear(p,api,bodyX,rearCab);}
 else if(kind==='derby'){
  const back=1.05,deckY=1.04;
  for(const s of [-1,1]){
   tube('Rear window pillar',[s*.59,roof,.72],[s*bodyX,deckY,back],.025,'frame');
   tube('Trunk side rail',[s*bodyX,deckY,back],[s*bodyX,deckY,tail],.025,'frame');
   tube('Trunk mount',[s*.5,1.2,r+.22],[s*bodyX,deckY,tail],.027);
   plate('Rear quarter panel',[[s*bodyX,deckY,back],[s*bodyX,deckY,tail],[s*bodyX,.82,tail],[s*bodyX,.82,back]],'frame',.022);
   tube('Trunk rear stay',[s*.57,.71,r+.46],[s*bodyX,.82,tail],.035);
   box('Trunk retaining strap',[.06,.012,tail-back],[s*.48,deckY+.01,(back+tail)/2],'alloy');
   tube('Windshield crash bar',[s*.25,belt,frontGlass],[s*.25,roof,-.23],.021,'steel');
  }
  box('Trunk lid',[bodyX*2,.024,tail-back],[0,deckY,(tail+back)/2],'frame');
  box('Rear crash bumper',[bodyX*2+.12,.15,.12],[0,.80,tail+.04],'steel');
  tube('Trunk cross rail',[-bodyX,deckY,back],[bodyX,deckY,back],.025);
  for(const s of [-1,1])box('Race number upright',[.018,.3,.07],[s*.606,roof+.02,.19],'race');
 }
 else if(kind==='rally'){
  for(const s of [-1,1]){
   tube('Hatch pillar',[s*.59,roof,.72],[s*bodyX,1.02,tail],.025,'frame');
   plate('Rear quarter upper',[[s*bodyX,belt,rearCab],[s*.59,roof,.72],[s*bodyX,1.02,tail],[s*bodyX,.97,rearCab]],'frame',.022);
   tube('Tail support',[s*.57,.71,r+.46],[s*bodyX,.90,tail],.025);
   box('Tail light',[.15,.13,.03],[s*(bodyX-.1),.92,tail+.020],'red');
   tube('Wing pedestal',[s*.48,1.02,tail-.1],[s*.48,1.25,tail-.04],.016);
   box('Roof intake side',[.06,.075,.28],[s*.2,roof+.065,.15],'dark');
  }
  plate('Hatch glass',[[-.59,roof,.72],[.59,roof,.72],[bodyX,1.02,tail],[-bodyX,1.02,tail]],'glass',.010);
  box('Tail panel',[bodyX*2,.30,.036],[0,.87,tail],'frame');
  box('Rear wing',[bodyX*2+.14,.025,.20],[0,1.26,tail-.04],'dark');
  box('Roof intake cap',[.46,.025,.28],[0,roof+.114,.15],'frame');
  for(const s of [-1,1]){box('Rally mud flap',[.26,.20,.018],[s*p.track/2,y-.15,r+p.tireRadius+.06],'rubber');tube('Mud flap hanger',[s*.5,floor+.11,r+.35],[s*p.track/2,y-.05,r+p.tireRadius+.06],.016);}
 }else{
  const deck=1.0,bedTop=1.27;
  for(const s of [-1,1]){
   tube('Bed front post',[s*bodyX,belt,rearCab],[s*bodyX,bedTop,rearCab],.024,'frame');
   tube('Bed upper rail',[s*bodyX,bedTop,rearCab],[s*bodyX,bedTop,tail],.025,'frame');
   tube('Bed rear support',[s*.57,.71,r+.46],[s*bodyX,deck,tail],.026);
   plate('Bed side panel',[[s*bodyX,deck,rearCab],[s*bodyX,bedTop,rearCab],[s*bodyX,bedTop,tail],[s*bodyX,deck,tail]],'frame',.023);
   box('Truck tail light',[.1,.17,.025],[s*(bodyX-.12),bedTop-.1,tail+.020],'red');
   tube('Bed diagonal',[s*bodyX,deck,tail],[s*.5,1.2,r+.22],.02);
  }
  box('Tailgate',[bodyX*2,bedTop-deck-.002,.026],[0,(bedTop+deck)/2,tail+.002],'frame');
  box('Bed floor',[bodyX*2,.03,tail-rearCab],[0,deck,(tail+rearCab)/2],'dark');
  box('Tailgate handle',[.23,.035,.018],[0,bedTop-.07,tail+.019],'dark');
  for(let i=-3;i<=3;i++)box('Bed floor rib',[.024,.01,tail-rearCab-.08],[i*.19,deck+.019,(tail+rearCab)/2],'alloy');
  tube('Rear impact bumper',[-bodyX,.78,tail+.1],[bodyX,.78,tail+.1],.045);
  for(const s of [-1,1])tube('Rear bumper mount',[s*.57,.71,r+.46],[s*bodyX,.78,tail+.1],.025);
 }
 // Separate roof-mounted light bar and brackets.
 for(const s of [-1,1])tube('Light bar foot',[s*.48,roof+.02,-.20],[s*.48,roof+.13,-.20],.016);
 box('Light bar housing',[1.04,.075,.07],[0,roof+.14,-.20],'dark');
 for(let i=0;i<10;i++)box('Light bar cell',[.074,.044,.012],[(i-4.5)*.095,roof+.14,-.240],'lens');
 // Contrasting race graphics are thin geometric strips, also separate parts.
 plate('Hood race stripe',[[-.10,hoodY+.011,nose],[.10,hoodY+.011,nose],[.10,belt+.011,frontGlass],[-.10,belt+.011,frontGlass]],'race',.008);
}
