// A detailed house built only from cuboids (SHAPE_BOX colliders): running-bond
// brick walls with interlocking corners, door and window openings with
// lintels, sills and glass, stepped brick gables, a ridge beam, rafters, roof
// boards and a ridge cap, an exterior chimney, a porch on posts with a step,
// and a picket fence with a mailbox. Every piece is its own dynamic body and
// starts asleep, touching but never overlapping its neighbours.
//
// Box pairs take the engine's box-box path instead of the convex-hull SAT.
// Rafters and boards are loose and held only by friction (μ = 0.5). Woken whole,
// the house stands and sleeps again at a 12° pitch; at 18° the settling jolt
// starts the rafters sliding and the roof falls in. Contacts use zero-gap mode
// (FLAG_CONTACT_GAP): with the default margin every course rests 2 mm deep, an
// 11-course wall sinks ~2 cm unevenly against the taller gables, and the roof
// shifts.
import {HullArena,gridConfig,scale,appendHull} from './geometry.mjs';

const BRICK_L=.5,BRICK_H=.25,WALL_T=.25;   // brick length, course height, wall thickness
const HALF_X=4,HALF_Z=3,COURSES=11;           // outer walls at x = ±4, z = ±3
const EAVE=COURSES*BRICK_H;                    // top of the walls
const GAP=.001;                                // side gap between neighbouring pieces
const DENSITY={brick:1900,concrete:2300,wood:550,glass:2500};

// Deterministic colour jitter (the same house every load).
function rng(seed){let s=seed>>>0;return ()=>{s=(s*1664525+1013904223)>>>0;return s/4294967296}}

// `parts` limits the build (diagnostics): walls, openings, gables, roof,
// chimney, porch, fence.
// colliders: 'box' (SHAPE_BOX) or 'hull' (each box as an 8-vertex convex hull:
// the same shapes through the convex path, to compare the two).
export function buildCuboidHouse({seed=7,pitchDeg=12,colliders:colliderKind='box',parts=['walls','openings','gables','roof','chimney','porch','fence']}={}){
  const PITCH=pitchDeg*Math.PI/180,TAN=Math.tan(PITCH),COS=Math.cos(PITCH),SIN=Math.sin(PITCH);
  const has=new Set(parts),rand=rng(seed),pieces=[];
  const jitter=(c,a)=>c.map(x=>Math.min(1,Math.max(0,x*(1+(rand()-.5)*a))));
  const brick=()=>jitter([.62,.25,.18],.25),concrete=[.62,.62,.6],trim=[.9,.88,.82];
  // center, half extents, rotation about +x (radians), material, colour
  const box=(center,half,material,color,angle=0)=>pieces.push({center,half,material,color,angle});

  // ---- walls ----------------------------------------------------------------
  // A wall runs along u (x for front/back, z for the sides) at a fixed coordinate.
  // Even courses: front/back span the full width and the sides fit between
  // them; odd courses swap, so the corners interlock.
  const walls=[
    {axis:'x',at:HALF_Z-WALL_T/2,openings:[
      {u0:-.5,u1:.5,c0:0,c1:8,kind:'door'},{u0:-3,u1:-1.8,c0:3,c1:8,kind:'window'},{u0:1.8,u1:3,c0:3,c1:8,kind:'window'}]},
    {axis:'x',at:-HALF_Z+WALL_T/2,openings:[{u0:-2.6,u1:-1.4,c0:3,c1:8,kind:'window'},{u0:1.4,u1:2.6,c0:3,c1:8,kind:'window'}]},
    {axis:'z',at:HALF_X-WALL_T/2,openings:[{u0:-.6,u1:.6,c0:3,c1:8,kind:'window'}]},
    {axis:'z',at:-HALF_X+WALL_T/2,openings:[]},
  ];
  const place=(wall,u,y,du,dy,depth,material,color)=>wall.axis==='x'
    ?box([u,y,wall.at],[du/2,dy/2,depth/2],material,color)
    :box([wall.at,y,u],[depth/2,dy/2,du/2],material,color);
  // Fill [a, b] of one course with bricks; odd courses start with a half brick.
  const fill=(wall,a,b,course)=>{
    const len=b-a;if(len<.1)return;
    const n=Math.max(1,Math.round(len/BRICK_L)),bl=len/n,cuts=[a];
    if(course%2&&n>1){for(let k=0;k<n;k++)cuts.push(a+bl*(k+.5));}else for(let k=1;k<n;k++)cuts.push(a+bl*k);
    cuts.push(b);
    for(let k=0;k+1<cuts.length;k++)place(wall,(cuts[k]+cuts[k+1])/2,(course+.5)*BRICK_H,cuts[k+1]-cuts[k]-GAP,BRICK_H,WALL_T,'brick',brick());
  };
  const minus=(a,b,holes)=>{
    let runs=[[a,b]];
    for(const [h0,h1] of holes)runs=runs.flatMap(([r0,r1])=>h1<=r0||h0>=r1?[[r0,r1]]:[[r0,Math.max(r0,h0)],[Math.min(r1,h1),r1]].filter(([x,y])=>y-x>1e-9));
    return runs;
  };
  if(has.has('walls'))for(let c=0;c<COURSES;c++)for(const wall of walls){
    const full=(wall.axis==='x')===(c%2===0);
    const half=wall.axis==='x'?HALF_X:HALF_Z,extent=full?half:half-WALL_T;
    const holes=[];
    for(const o of wall.openings){
      if(c>=o.c0&&c<o.c1)holes.push([o.u0,o.u1]);
      else if(c===o.c1)holes.push([o.u0-WALL_T,o.u1+WALL_T]); // lintel course
    }
    for(const [a,b] of minus(-extent,extent,holes))fill(wall,a,b,c);
  }
  if(has.has('openings'))for(const wall of walls)for(const o of wall.openings){
    // Lintel over the opening, bearing a brick depth on each side.
    place(wall,(o.u0+o.u1)/2,(o.c1+.5)*BRICK_H,o.u1-o.u0+2*WALL_T-GAP,BRICK_H,WALL_T,'concrete',concrete);
    const top=o.c1*BRICK_H,bottom=o.c0*BRICK_H,w=o.u1-o.u0;
    if(o.kind==='door')place(wall,(o.u0+o.u1)/2,(top-.01)/2,w-.02,top-.01,.05,'wood',[.2,.38,.42]);
    else{
      const sill=.06;
      place(wall,(o.u0+o.u1)/2,bottom+sill/2,w-.01,sill,WALL_T,'wood',trim);
      place(wall,(o.u0+o.u1)/2,(bottom+sill+top-.01)/2,w-.02,top-.01-bottom-sill,.03,'glass',[.62,.8,.9]);
    }
  }

  // ---- gables and roof --------------------------------------------------------
  // P(z): the plane the rafters sit on, through the outer wall top edges.
  const plane=z=>EAVE+(HALF_Z-Math.abs(z))*TAN+GAP;
  let gableTop=EAVE;const gableCourses=[];
  for(let c=COURSES;;c++){
    const y1=(c+1)*BRICK_H,span=HALF_Z-(y1-EAVE)/TAN-.002;
    if(span<.3)break;gableCourses.push({c,span});gableTop=y1;
  }
  // Ridge beam: rests on the top gable course, its top corners just under P.
  let ridgeH=plane(.1)-.002-gableTop;
  while(ridgeH<.06&&gableCourses.length){gableCourses.pop();gableTop=EAVE+gableCourses.length*BRICK_H;ridgeH=plane(.1)-.002-gableTop}
  if(has.has('gables'))for(const x of [HALF_X-WALL_T/2,-HALF_X+WALL_T/2])for(const {c,span} of gableCourses)fill({axis:'z',at:x},-span,span,c);
  if(has.has('roof'))box([0,gableTop+ridgeH/2,0],[HALF_X-.05,ridgeH/2,.1],'wood',[.5,.36,.22]);
  // Rafters and roof boards along each slope, from just off the ridge to the eave overhang.
  const zLo=.02,zHi=HALF_Z+.35,rafterT=.12,boardT=.03,strip=.62;
  if(has.has('roof'))for(const side of [1,-1]){
    const angle=side*PITCH;
    const onSlope=(z0,z1,lift,thick)=>{ // centre of a piece whose bottom face lies on P + lift between z0 and z1
      const zm=(z0+z1)/2,y=plane(zm)+lift;
      return [side*zm+side*SIN*thick/2,y+COS*thick/2];
    };
    for(const x of [-3.6,-1.8,0,1.8,3.6]){
      const [z,y]=onSlope(zLo,zHi,0,rafterT);
      box([x,y,z],[.04,rafterT/2,(zHi-zLo)/COS/2],'wood',[.55,.4,.25],angle);
    }
    const lift=rafterT/COS+GAP;
    for(let z0=zLo;z0<zHi-1e-6;z0+=strip*COS){
      const z1=Math.min(zHi,z0+strip*COS),[z,y]=onSlope(z0,z1,lift,boardT),shade=jitter([.33,.29,.28],.15);
      for(const [x0,x1] of [[-HALF_X,-GAP],[GAP,HALF_X]])
        box([(x0+x1)/2,y,z],[(x1-x0)/2,boardT/2,(z1-z0)/COS/2-GAP],'wood',shade,angle);
    }
  }
  const capY=plane(zLo)+rafterT/COS+GAP+boardT/COS+.002;
  if(has.has('roof'))box([0,capY+.02,0],[HALF_X,.02,.18],'wood',[.25,.22,.22]);

  // ---- chimney (outside the left gable) ----------------------------------------
  const chimneyX=-HALF_X-.26,chimneyTop=plane(0)+.6;
  if(has.has('chimney'))for(let c=0;(c+1)*BRICK_H<=chimneyTop;c++){
    const y=(c+.5)*BRICK_H,col=jitter([.5,.2,.15],.2);
    if(c%2===0)for(const z of [-.125,.125])box([chimneyX,y,z],[.25-GAP,BRICK_H/2,.125-GAP/2],'brick',col);
    else for(const dx of [-.125,.125])box([chimneyX+dx,y,0],[.125-GAP/2,BRICK_H/2,.25-GAP],'brick',col);
  }

  // ---- porch and step ---------------------------------------------------------------
  const postH=2.3;
  if(has.has('porch')){
  for(const x of [-1.1,1.1])for(const z of [HALF_Z+.2,HALF_Z+1.4])box([x,postH/2,z],[.075,postH/2,.075],'wood',trim);
  box([0,postH+GAP+.04,HALF_Z+.825],[1.3,.04,.775],'wood',[.4,.4,.42]);
  box([0,.075,HALF_Z+.3+GAP],[.7,.075,.3],'concrete',concrete);
  }

  // ---- picket fence, gate gap and mailbox --------------------------------------
  const fenceZ=HALF_Z+3.2,posts=[-5,-2.5,-.75,.75,2.5,5];
  if(has.has('fence')){
  for(const x of posts)box([x,.475,fenceZ],[.05,.475,.05],'wood',trim);
  for(let k=0;k+1<posts.length;k++){
    const a=posts[k],b=posts[k+1];if(a===-.75)continue; // the gate
    box([(a+b)/2,.95+GAP+.02,fenceZ],[(b-a)/2-GAP,.02,.03],'wood',trim);
    for(let x=a+.15;x<b-.12;x+=.18)box([x,.4,fenceZ],[.04,.4,.0125],'wood',trim);
  }
  box([1.3,.5,fenceZ+.45],[.04,.5,.04],'wood',[.45,.32,.2]);
  box([1.3,1+GAP+.11,fenceZ+.45],[.1,.11,.225],'wood',[.18,.2,.24]);
  }

  return sceneFromPieces(pieces,{hulls:colliderKind==='hull'});
}

// Bodies, box colliders and the render mesh, recentred so the footprint's
// bounding box is centred on x = z = 0 (layouts assume a centred house).
export function sceneFromPieces(pieces,{hulls=false}={}){
  const arena=new HullArena();
  const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
  for(const p of pieces)for(const c of corners(p))for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],c[k]);hi[k]=Math.max(hi[k],c[k])}
  const shift=[-(lo[0]+hi[0])/2,0,-(lo[2]+hi[2])/2];
  const bodies=[{p:[0,0,0],q:[0,0,0,1],invMass:0,invI:[0,0,0],flags:0}];
  const colliders=[{body:0,shape:4,density:0,params:[0,1,0,0]}],vertices=[];
  for(const p of pieces){
    const [hx,hy,hz]=p.half,rho=DENSITY[p.material],m=rho*8*hx*hy*hz,id=bodies.length;
    const q=[Math.sin(p.angle/2),0,0,Math.cos(p.angle/2)];
    bodies.push({p:p.center.map((x,k)=>x+shift[k]),q,invMass:1/m,invI:[3/(m*(hy*hy+hz*hz)),3/(m*(hx*hx+hz*hz)),3/(m*(hx*hx+hy*hy))],flags:5});
    if(hulls){
      const v=[];for(let i=0;i<8;i++)v.push([i&1?hx:-hx,i&2?hy:-hy,i&4?hz:-hz]);
      const faces=[[0,2,6,4],[1,5,7,3],[0,4,5,1],[2,3,7,6],[0,1,3,2],[4,6,7,5]];
      const c=appendHull(arena,v,{sizes:faces.map(f=>f.length),indices:faces.flat()});
      colliders.push({body:id,shape:3,density:rho,params:c.params,localPos:c.localPos,material:0});
    }else colliders.push({body:id,shape:2,density:rho,params:[hx,hy,hz,0],material:0});
    for(const [n,quad] of FACES)for(const k of [0,1,2,0,2,3]){const s=quad[k];vertices.push(s[0]*hx,s[1]*hy,s[2]*hz,...n,...p.color,id)}
  }
  const width=hi[0]-lo[0],depth=hi[2]-lo[2],height=hi[1]-lo[1],extent=Math.max(width,depth,height);
  const projectileStart=bodies.length,projectileCount=16,projectileRadius=.4,projectileVertices=[];
  for(let slot=0;slot<projectileCount;slot++){
    const id=bodies.length;
    bodies.push({p:[slot*2,-1000,0],q:[0,0,0,1],invMass:0,invI:[0,0,0],flags:0});
    colliders.push({body:id,shape:0,density:0,params:[projectileRadius,0,0,0]});
    const point=(u,v)=>[Math.sin(v)*Math.cos(u),Math.cos(v),Math.sin(v)*Math.sin(u)];
    for(let y=0;y<12;y++)for(let x=0;x<24;x++){
      const a=point(x/24*Math.PI*2,y/12*Math.PI),b=point((x+1)/24*Math.PI*2,y/12*Math.PI);
      const c=point((x+1)/24*Math.PI*2,(y+1)/12*Math.PI),d=point(x/24*Math.PI*2,(y+1)/12*Math.PI);
      for(const n of [a,b,c,a,c,d])projectileVertices.push(...scale(n,projectileRadius),...n,1,.3,.08,id);
    }
  }
  return {projectileStart,projectileCount,projectileRadius,projectileMass:500,projectileVertices:new Float32Array(projectileVertices),
    bodies,colliders,joints:[],hullData:arena.data,skippedHulls:[],vertices:new Float32Array(vertices),lines:new Float32Array(0),
    selected:pieces,bonds:[],byId:new Map(),pieces:pieces.length,boxes:pieces.length,
    extent,extentX:width,extentZ:depth,height,copies:1,instanceStride:0,center:[0,height/2,0],gridCellSize:gridConfig(colliders).base,
    // Flags: 1 reuse contacts, 32 zero-gap contacts, 64 world-frame inertia.
    params:{flags:2027745,iterations:8,dt:1/120,gravityY:-9.81,contactOffset:.002,alpha:.95,sleepLin:.03,sleepAng:.05,sleepTime:1}};
}

// World corners of a piece (rotation about +x).
export function corners(p){
  const [hx,hy,hz]=p.half,c=Math.cos(p.angle),s=Math.sin(p.angle),out=[];
  for(const sx of [-1,1])for(const sy of [-1,1])for(const sz of [-1,1]){
    const y=sy*hy,z=sz*hz;out.push([p.center[0]+sx*hx,p.center[1]+c*y-s*z,p.center[2]+s*y+c*z]);
  }
  return out;
}

const FACES=[
  [[1,0,0],[[1,-1,-1],[1,1,-1],[1,1,1],[1,-1,1]]],[[-1,0,0],[[-1,-1,1],[-1,1,1],[-1,1,-1],[-1,-1,-1]]],
  [[0,1,0],[[-1,1,-1],[-1,1,1],[1,1,1],[1,1,-1]]],[[0,-1,0],[[-1,-1,1],[-1,-1,-1],[1,-1,-1],[1,-1,1]]],
  [[0,0,1],[[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]],[[0,0,-1],[[1,-1,-1],[-1,-1,-1],[-1,1,-1],[1,1,-1]]],
];
