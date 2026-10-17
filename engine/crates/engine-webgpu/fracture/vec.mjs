// Vector helpers for the browser camera, picking and diagnostics.
export const add=(a,b)=>a.map((x,i)=>x+b[i]);
export const sub=(a,b)=>a.map((x,i)=>x-b[i]);
export const scale=(a,s)=>a.map(x=>x*s);
export const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
export const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export const norm=a=>scale(a,1/(Math.hypot(...a)||1));
export const rotate=(q,v)=>add(v,scale(cross(q.slice(0,3),add(cross(q.slice(0,3),v),scale(v,q[3]))),2));
export const inverseRotate=(q,v)=>rotate([-q[0],-q[1],-q[2],q[3]],v);
export const quatMul=(a,b)=>[a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],
  a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]];
export const quatConjugate=q=>[-q[0],-q[1],-q[2],q[3]];
