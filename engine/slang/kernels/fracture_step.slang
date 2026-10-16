// Compound convex collision kernels (up to 64 vertices per child hull).
// Gauss-map edge pruning/contact selection adapted from three-avbd (MIT),
// Copyright (c) 2026 Steven Bobyn; see ../THIRD_PARTY_LICENSES.txt.
// Reuses the shared AVBD solver and contact ABI; independent broadphase entries
// keep the production primitive path compatible while this path is validated.
#include "world_step.slang"

struct FracHit { float4 normal; uint4 feature; };
BIND(27) RWStructuredBuffer<FracHit> gFracHits;

// Pose first: with `float3 p` after the scalar/bool fields, Apple's Metal compiler
// (Slang → MSL, macOS 14) miscompiled fracContacts' hull copies and dropped `p`
// once the clip loop was present; WGSL was unaffected. Keep this field order.
struct FracHull { float4 q; float3 p; uint base; uint nv; uint nf; uint ne; bool closed; };
FracHull fracHull(Collider c, BodyState b) {
    FracHull h;
    h.base=uint(c.params.x); float3 counts=gHullVerts[h.base];
    h.nv=min(uint(counts.x),64u); h.nf=min(uint(counts.y),128u); h.ne=min(uint(abs(counts.z)),192u);h.closed=counts.z>0;
    h.p=transformPoint(b.position,b.orientation,c.localPos); h.q=quatMul(b.orientation,c.localRot);
    return h;
}
float3 fracVertex(FracHull h,uint i) { return transformPoint(h.p,h.q,gHullVerts[h.base+1u+i]); }
float3 fracNormal(FracHull h,uint i) { return quatRotate(h.q,gHullVerts[h.base+1u+h.nv+i]); }
uint2 fracFaceRange(FracHull h,uint i) { return uint2(gHullVerts[h.base+1u+h.nv+h.nf+i].xy); }
float3 fracFaceVertex(FracHull h,uint face,uint i) { uint2 r=fracFaceRange(h,face);return transformPoint(h.p,h.q,gHullVerts[r.x+i]); }
uint fracMeta(FracHull h) { return h.base+1u+h.nv+2u*h.nf+2u*h.ne; }
uint4 fracEdge(FracHull h,uint i) {
    uint at=h.base+1u+h.nv+2u*h.nf+2u*i;
    return uint4(uint3(gHullVerts[at]),uint(gHullVerts[at+1u].x));
}
float3 fracInterior(FracHull h) { return transformPoint(h.p,h.q,gHullVerts[fracMeta(h)]); }
// Gregorius/three-avbd Gauss-map pruning: only Minkowski-face edge axes matter.
bool fracMinkowski(float3 a,float3 b,float3 c,float3 d) {
    float3 ba=cross(b,a),dc=cross(d,c);
    float cba=dot(c,ba),dba=dot(d,ba),adc=dot(a,dc),bdc=dot(b,dc);
    return cba*dba<0 && adc*bdc<0 && cba*bdc>0;
}
void fracAdd(inout BoxManifold mf,float3 a,float3 b,uint feature) {
    if(mf.count>=8)return;
    for(int i=0;i<mf.count;i++)if(dot(mf.pts[i].pA-a,mf.pts[i].pA-a)<1e-8)return;
    mf.pts[mf.count].pA=a;mf.pts[mf.count].pB=b;mf.pts[mf.count].feature=feature;mf.count++;
}
BoxManifold fracPlane(FracHull h,float3 n,float offset,bool flip) {
    BoxManifold mf;mf.count=0;mf.normal=flip?n:-n;
    float lowest=1e30;
    for(uint i=0;i<h.nv;i++)lowest=min(lowest,dot(n,fracVertex(h,i))-offset);
    if(lowest>1e-5)return mf;
    bool used[64];
    for(uint i=0;i<h.nv;i++)used[i]=dot(n,fracVertex(h,i))-offset>lowest+0.002;
    for(uint k=0;k<8u;k++){
        uint pick=64u;float best=-1e30;
        for(uint i=0;i<h.nv;i++){
            if(used[i])continue;float3 p=fracVertex(h,i);float score=-(dot(n,p)-offset);
            if(k>0){score=1e30;for(int j=0;j<mf.count;j++){float3 q=flip?mf.pts[j].pB:mf.pts[j].pA;score=min(score,dot(p-q,p-q));}}
            if(score>best){best=score;pick=i;}
        }
        if(pick==64u)break;used[pick]=true;
        float3 p=fracVertex(h,pick),ground=p-n*(dot(n,p)-offset);
        fracAdd(mf,flip?ground:p,flip?p:ground,pick);
    }
    return mf;
}
void fracSegments(float3 p1,float3 q1,float3 p2,float3 q2,out float3 c1,out float3 c2) {
    float3 d1=q1-p1,d2=q2-p2,r=p1-p2;float a=dot(d1,d1),e=dot(d2,d2),b=dot(d1,d2),c=dot(d1,r),f=dot(d2,r);
    float s=0,t=0,denom=a*e-b*b;
    if(a>1e-12 && e>1e-12){s=denom>1e-12?clamp((b*f-c*e)/denom,0.0,1.0):0.0;t=(b*s+f)/e;
        if(t<0){t=0;s=clamp(-c/a,0.0,1.0);}else if(t>1){t=1;s=clamp((b-c)/a,0.0,1.0);}}
    c1=p1+s*d1;c2=p2+t*d2;
}
BoxManifold fracContacts(FracHull a,FracHull b,FracHit hit){
    BoxManifold mf;mf.count=0;mf.normal=hit.normal.xyz;if(hit.feature.w==0u)return mf;
    float3 n=hit.normal.xyz;uint type=hit.feature.x,ea=hit.feature.y,eb=hit.feature.z;
    bool complete=a.closed&&b.closed;
    mf.normal=n;
    if(type==2u){
        uint4 ai=fracEdge(a,ea),bi=fracEdge(b,eb);
        if(!complete){
            float3 da=normalize(fracVertex(a,ai.y)-fracVertex(a,ai.x)),db=normalize(fracVertex(b,bi.y)-fracVertex(b,bi.x));
            float ascore=-1e30,bscore=1e30;
            for(uint i=0;i<a.ne;i++){uint4 e=fracEdge(a,i);float3 x=fracVertex(a,e.x),y=fracVertex(a,e.y);if(abs(dot(normalize(y-x),da))>.999){float v=dot(n,(x+y)*.5);if(v>ascore){ascore=v;ai=e;}}}
            for(uint i=0;i<b.ne;i++){uint4 e=fracEdge(b,i);float3 x=fracVertex(b,e.x),y=fracVertex(b,e.y);if(abs(dot(normalize(y-x),db))>.999){float v=dot(n,(x+y)*.5);if(v<bscore){bscore=v;bi=e;}}}
        }
        float3 pa,pb;fracSegments(fracVertex(a,ai.x),fracVertex(a,ai.y),fracVertex(b,bi.x),fracVertex(b,bi.y),pa,pb);
        fracAdd(mf,pa,pb,0x800000u+ea*192u+eb);return mf;
    }
    FracHull ref=type==0u?a:b,inc=type==0u?b:a;float3 rn=type==0u?n:-n;
    uint rf=0u,inf=0u;float rbest=-1e30,ibest=1e30;
    for(uint i=0;i<ref.nf;i++){float d=dot(fracNormal(ref,i),rn);if(d>rbest){rbest=d;rf=i;}}
    float lowest=1e30;float3 deepest=float3(0,0,0);
    for(uint i=0;i<inc.nv;i++){float3 v=fracVertex(inc,i);float d=dot(v,rn);if(d<lowest){lowest=d;deepest=v;}}
    // Irregular hulls: choose an incident face touching the deepest vertex.
    for(uint i=0;i<inc.nf;i++){
        bool touches=false;uint nc=fracFaceRange(inc,i).y;
        for(uint j=0;j<nc;j++){float3 d=fracFaceVertex(inc,i,j)-deepest;if(dot(d,d)<1e-10)touches=true;}
        float d=dot(fracNormal(inc,i),rn);if(touches&&d<ibest){ibest=d;inf=i;}
    }
    float3 poly[128];float3 next[128];uint count=min(fracFaceRange(inc,inf).y,64u);
    for(uint i=0;i<count;i++)poly[i]=fracFaceVertex(inc,inf,i);
    uint rc=fracFaceRange(ref,rf).y;float3 refN=fracNormal(ref,rf);
    for(uint edge=0u;edge<rc && count>0u;edge++){
        float3 p=fracFaceVertex(ref,rf,edge),q=fracFaceVertex(ref,rf,(edge+1u)%rc);
        float3 side=normalize(cross(q-p,refN));uint nc=0u;
        for(uint i=0u;i<count;i++){
            float3 x=poly[i],y=poly[(i+1u)%count];float dx=dot(side,x-p),dy=dot(side,y-p);
            if(dx<=1e-6 && nc<128u)next[nc++]=x;
            if((dx<=1e-6)!=(dy<=1e-6) && nc<128u)next[nc++]=lerp(x,y,clamp(dx/(dx-dy),0.0,1.0));
        }
        count=nc;for(uint i=0;i<count;i++)poly[i]=next[i];
    }
    float3 rp=fracFaceVertex(ref,rf,0u);
    // Deepest first, then maximize distance from chosen contacts. Preserve
    // polygon indices in feature IDs, including when there are >8 vertices.
    bool used[128];float depth[128];
    for(uint i=0;i<count;i++){depth[i]=dot(rn,poly[i]-rp);used[i]=depth[i]>1e-4;}
    for(uint k=0;k<8u;k++){
        uint pick=128u;float scoreBest=-1e30;
        for(uint i=0;i<count;i++){
            if(used[i])continue;
            float score=-depth[i];
            if(k>0u){score=1e30;for(int j=0;j<mf.count;j++){
                float3 p=type==0u?mf.pts[j].pB:mf.pts[j].pA;float3 d=poly[i]-p;score=min(score,dot(d,d));}}
            if(score>scoreBest){scoreBest=score;pick=i;}
        }
        if(pick==128u)break;
        used[pick]=true;float3 point=poly[pick],projected=point-rn*depth[pick];
        fracAdd(mf,type==0u?projected:point,type==0u?point:projected,(type<<24u)|(rf<<16u)|(inf<<8u)|pick);
    }
    // Roundoff or an imperfect baked face loop can empty a clipped patch.
    // SAT has already established overlap; retain a supporting contact instead
    // of dropping the collision (same fallback as three-avbd).
    if(mf.count==0){
        float3 pa=fracVertex(a,0),pb=fracVertex(b,0);
        for(uint i=1;i<a.nv;i++){float3 p=fracVertex(a,i);if(dot(n,p)>dot(n,pa))pa=p;}
        for(uint i=1;i<b.nv;i++){float3 p=fracVertex(b,i);if(dot(n,p)<dot(n,pb))pb=p;}
        fracAdd(mf,pa,pb,(type<<24u)|(rf<<16u)|(inf<<8u)|255u);
    }
    return mf;
}

// Closest point on a triangle, including its edge and vertex Voronoi regions.
float3 fracClosestTriangle(float3 p,float3 a,float3 b,float3 c) {
    float3 ab=b-a,ac=c-a,ap=p-a;float d1=dot(ab,ap),d2=dot(ac,ap);
    if(d1<=0 && d2<=0)return a;
    float3 bp=p-b;float d3=dot(ab,bp),d4=dot(ac,bp);
    if(d3>=0 && d4<=d3)return b;
    float vc=d1*d4-d3*d2;if(vc<=0 && d1>=0 && d3<=0)return a+ab*(d1/(d1-d3));
    float3 cp=p-c;float d5=dot(ab,cp),d6=dot(ac,cp);
    if(d6>=0 && d5<=d6)return c;
    float vb=d5*d2-d1*d6;if(vb<=0 && d2>=0 && d6<=0)return a+ac*(d2/(d2-d6));
    float va=d3*d6-d5*d4;if(va<=0 && d4-d3>=0 && d5-d6>=0)return b+(c-b)*((d4-d3)/((d4-d3)+(d5-d6)));
    float denom=va+vb+vc;if(abs(denom)<1e-20)return a;
    return a+ab*(vb/denom)+ac*(vc/denom);
}
BoxManifold fracSphere(FracHull h,float3 center,float radius,bool flip) {
    BoxManifold mf;mf.count=0;mf.normal=float3(0,1,0);
    float best=1e30,maxPlane=-1e30;float3 closest=center,outward=float3(0,1,0);uint feature=0;
    for(uint face=0;face<h.nf;face++){
        float3 a=fracFaceVertex(h,face,0),n=fracNormal(h,face);float plane=dot(n,center-a);
        if(plane>radius)return mf;
        if(plane>maxPlane){maxPlane=plane;outward=n;}
        uint count=fracFaceRange(h,face).y;
        for(uint i=1;i+1<count;i++){
            float3 q=fracClosestTriangle(center,a,fracFaceVertex(h,face,i),fracFaceVertex(h,face,i+1));
            float d=dot(center-q,center-q);if(d<best){best=d;closest=q;feature=face;}
        }
    }
    float3 n;
    if(maxPlane<=0){n=outward;closest=center-n*maxPlane;}
    else {if(best>radius*radius)return mf;n=best>1e-16?(center-closest)*rsqrt(best):outward;}
    float3 spherePoint=center-n*radius;mf.normal=flip?-n:n;
    fracAdd(mf,flip?spherePoint:closest,flip?closest:spherePoint,feature);return mf;
}
BoxManifold fracSpherePlane(float3 center,float radius,float3 n,float offset,bool flip){
    BoxManifold mf;mf.count=0;mf.normal=flip?n:-n;
    float d=dot(center,n)-offset-radius;if(d>0)return mf;
    float3 p=center-n*radius,q=p-n*d;fracAdd(mf,flip?q:p,flip?p:q,0);return mf;
}

// Reuse settled contact patches before scheduling expensive hull work. Records
// stay keyed by collider pair, so compound children cannot alias one another.
[shader("compute")]
[numthreads(64,1,1)]
void fracture_prepare_candidates(uint3 tid:SV_DispatchThreadID){
    uint i=dispatchIndex(tid);if(i>=min(ALOAD(gCounters[0].candidateCount),HOST_S(2)))return;
    FracHit hit;hit.normal=float4(0);hit.feature=uint4(0);
    if((PARAMS.flags&1u)!=0u){
        CandidatePair pair=gCandidates[i];Collider a=gColliders[pair.colliderA],b=gColliders[pair.colliderB];
        if(tryReuseContacts(pair,gBodies[a.body],gBodies[b.body],PARAMS.contactOffset))hit.feature.w=2u;
    }
    gFracHits[i]=hit;
}
// One workgroup per hull pair. Cache each transformed vertex/normal once;
// distribute face/edge axes over 64 lanes. 13.5 KiB shared memory, no private
// clipping arrays. This matters for 64-vertex / 186-edge hulls.
groupshared float4 fracAV[64];
groupshared float4 fracBV[64];
groupshared float4 fracAF[128];
groupshared float4 fracBF[128];
groupshared uint4 fracAE[192];
groupshared uint4 fracBE[192];
groupshared float4 fracScores[64];
groupshared uint fracKeys[64];
groupshared float4 fracFaceWinner;
groupshared uint fracFaceKey;
float2 fracCachedProject(bool a,uint count,float3 n){
    float2 range=float2(1e30,-1e30);
    for(uint v=0;v<count;v++){float p=dot((a?fracAV[v]:fracBV[v]).xyz,n);range.x=min(range.x,p);range.y=max(range.y,p);}
    return range;
}
[shader("compute")]
[numthreads(1,1,1)]
void fracture_write_sat_args(uint3 tid:SV_DispatchThreadID){
    uint n=min(ALOAD(gCounters[0].candidateCount),HOST_S(2));
    // Balance rows: 35k pairs must not launch 65,536 workgroups, almost half
    // of which would only execute the mandatory uniformity barriers.
    uint rows=max(1u,(n+32767u)/32768u),width=(n+rows-1u)/rows;
    gIndirectArgs[69u*3u]=width;gIndirectArgs[69u*3u+1u]=rows;gIndirectArgs[69u*3u+2u]=1u;
}
[shader("compute")]
[numthreads(64,1,1)]
void fracture_sat(uint3 group:SV_GroupID,uint lane:SV_GroupIndex){
    uint count=min(ALOAD(gCounters[0].candidateCount),HOST_S(2));
    uint rows=max(1u,(count+32767u)/32768u),width=(count+rows-1u)/rows;
    uint index=group.x+group.y*width;
    bool valid=index<count;
    if(valid)valid=gFracHits[index].feature.w!=2u;
    FracHull a=(FracHull)0,b=(FracHull)0;
    if(valid){
        CandidatePair pair=gCandidates[index];Collider ca=gColliders[pair.colliderA],cb=gColliders[pair.colliderB];
        valid=ca.shape==SHAPE_CONVEX&&cb.shape==SHAPE_CONVEX;
        if(valid){a=fracHull(ca,gBodies[ca.body]);b=fracHull(cb,gBodies[cb.body]);}
    }
    // No storage-dependent early returns: every lane reaches every barrier,
    // including padding groups and sphere/plane pairs (WebGPU uniformity).
    bool complete=a.closed&&b.closed;
    if(lane<a.nv)fracAV[lane]=float4(fracVertex(a,lane),0);
    if(lane<b.nv)fracBV[lane]=float4(fracVertex(b,lane),0);
    for(uint f=lane;f<a.nf;f+=64u){float3 n=fracNormal(a,f);fracAF[f]=float4(n,dot(n,fracFaceVertex(a,f,0)));}
    for(uint f=lane;f<b.nf;f+=64u){float3 n=fracNormal(b,f);fracBF[f]=float4(n,dot(n,fracFaceVertex(b,f,0)));}
    for(uint e=lane;e<a.ne;e+=64u)fracAE[e]=fracEdge(a,e);
    for(uint e=lane;e<b.ne;e+=64u)fracBE[e]=fracEdge(b,e);
    GroupMemoryBarrierWithGroupSync();
    float4 best=float4(0,1,0,-1e30);uint key=0xFFFFFFFFu;
    for(uint f=lane;f<a.nf+b.nf;f+=64u){
        bool fromA=f<a.nf;uint fi=fromA?f:f-a.nf;float4 face=fromA?fracAF[fi]:fracBF[fi];
        float3 n=fromA?face.xyz:-face.xyz;float sep;
        if(complete){float2 range=fracCachedProject(!fromA,fromA?b.nv:a.nv,face.xyz);sep=range.x-face.w;}
        else{float2 pa=fracCachedProject(true,a.nv,n),pb=fracCachedProject(false,b.nv,n);float forward=pb.x-pa.y,backward=pa.x-pb.y;sep=max(forward,backward);if(backward>forward)n=-n;}
        uint candidate=(fromA?0u:128u)+fi;
        if(sep>best.w||(sep==best.w&&candidate<key)){best=float4(n,sep);key=candidate;}
    }
    fracScores[lane]=best;fracKeys[lane]=key;GroupMemoryBarrierWithGroupSync();
    if(lane==0u){
        for(uint k=1;k<64u;k++)if(fracScores[k].w>best.w||(fracScores[k].w==best.w&&fracKeys[k]<key)){best=fracScores[k];key=fracKeys[k];}
        fracFaceWinner=best;fracFaceKey=key;
    }
    GroupMemoryBarrierWithGroupSync();
    best=float4(0,1,0,-1e30);key=0xFFFFFFFFu;float3 interior=fracInterior(a);
    for(uint e=lane;fracFaceWinner.w<=1e-5&&e<a.ne*b.ne;e+=64u){
        uint ea=e/b.ne,eb=e%b.ne;uint4 ai=fracAE[ea],bi=fracBE[eb];
        if(complete&&!fracMinkowski(fracAF[ai.z].xyz,fracAF[ai.w].xyz,-fracBF[bi.z].xyz,-fracBF[bi.w].xyz))continue;
        float3 pa=fracAV[ai.x].xyz,pb=fracBV[bi.x].xyz,da=fracAV[ai.y].xyz-pa,db=fracBV[bi.y].xyz-pb,n=cross(da,db);
        float lengthN=length(n);if(lengthN<1e-5*length(da)*length(db))continue;
        n/=lengthN;if(dot(n,pa-interior)<0)n=-n;float sep=dot(n,pb-pa);
        if(!complete){float2 ra=fracCachedProject(true,a.nv,n),rb=fracCachedProject(false,b.nv,n);float forward=rb.x-ra.y,backward=ra.x-rb.y;sep=max(forward,backward);if(backward>forward)n=-n;}
        uint candidate=(ea<<16u)|eb;
        if(sep>best.w||(sep==best.w&&candidate<key)){best=float4(n,sep);key=candidate;}
    }
    fracScores[lane]=best;fracKeys[lane]=key;GroupMemoryBarrierWithGroupSync();
    if(lane==0u){
        for(uint k=1;k<64u;k++)if(fracScores[k].w>best.w||(fracScores[k].w==best.w&&fracKeys[k]<key)){best=fracScores[k];key=fracKeys[k];}
        FracHit hit;hit.normal=fracFaceWinner;hit.feature=uint4(fracFaceKey>=128u?1u:0u,0,0,1);
        if(fracFaceWinner.w>1e-5||best.w>1e-5)hit.feature.w=0u;
        else if(best.w>0.95*fracFaceWinner.w+0.01){hit.normal=best;hit.feature=uint4(2u,key>>16u,key&65535u,1u);}
        if(valid)gFracHits[index]=hit;
    }
}

[shader("compute")]
[numthreads(64,1,1)]
void fracture_narrowphase(uint3 tid:SV_DispatchThreadID) {
    uint i=dispatchIndex(tid);if(i>=ALOAD(gCounters[0].candidateCount))return;
    if(gFracHits[i].feature.w==2u)return;
    CandidatePair pair=gCandidates[i];Collider ca=gColliders[pair.colliderA],cb=gColliders[pair.colliderB];
    BodyState ba=gBodies[ca.body],bb=gBodies[cb.body];BoxManifold mf;mf.count=0;mf.normal=float3(0,1,0);
    if(ca.shape==SHAPE_CONVEX && cb.shape==SHAPE_CONVEX)mf=fracContacts(fracHull(ca,ba),fracHull(cb,bb),gFracHits[i]);
    else if(ca.shape==SHAPE_CONVEX && cb.shape==SHAPE_PLANE)mf=fracPlane(fracHull(ca,ba),cb.params.xyz,cb.params.w,false);
    else if(ca.shape==SHAPE_PLANE && cb.shape==SHAPE_CONVEX)mf=fracPlane(fracHull(cb,bb),ca.params.xyz,ca.params.w,true);
    else if(ca.shape==SHAPE_CONVEX && cb.shape==SHAPE_SPHERE)mf=fracSphere(fracHull(ca,ba),transformPoint(bb.position,bb.orientation,cb.localPos),cb.params.x,false);
    else if(ca.shape==SHAPE_SPHERE && cb.shape==SHAPE_CONVEX)mf=fracSphere(fracHull(cb,bb),transformPoint(ba.position,ba.orientation,ca.localPos),ca.params.x,true);
    else if(ca.shape==SHAPE_SPHERE && cb.shape==SHAPE_PLANE)mf=fracSpherePlane(ba.position,ca.params.x,cb.params.xyz,cb.params.w,false);
    else if(ca.shape==SHAPE_PLANE && cb.shape==SHAPE_SPHERE)mf=fracSpherePlane(bb.position,cb.params.x,ca.params.xyz,ca.params.w,true);
    else if(ca.shape==SHAPE_SPHERE && cb.shape==SHAPE_SPHERE){
        float3 d=bb.position-ba.position;float distance=length(d),radius=ca.params.x+cb.params.x;
        if(distance<=radius){mf.normal=distance>1e-8?d/distance:float3(1,0,0);fracAdd(mf,ba.position+mf.normal*ca.params.x,bb.position-mf.normal*cb.params.x,0);}
    }
    if(mf.count>0)emitContactManifold(pair,ba,bb,ca,cb,mf,PARAMS.contactOffset);
}

// Exact per-hull bounds and conservative OBB proxies. All child vertices stay
// in their compound body's frame. Static/sleeping caches are initialized by host.
[shader("compute")]
[numthreads(64,1,1)]
void fracture_build_aabbs(uint3 tid:SV_DispatchThreadID){
    uint i=dispatchIndex(tid);if(i>=gCounters[0].colliderCount)return;
    Collider c=gColliders[i];BodyState b=gBodies[c.body];
    if(c.shape!=SHAPE_CONVEX){build_aabbs(tid);return;}
    if(bodyIsSleeping(b)&&bodyIsDynamic(b)&&HOST_S(17)>0u)return;
    FracHull h=fracHull(c,b);float3 lo=float3(1e30),hi=float3(-1e30);
    for(uint v=0;v<h.nv;v++){float3 p=fracVertex(h,v);lo=min(lo,p);hi=max(hi,p);}
    Aabb bounds;bounds.mn=lo-1e-4;bounds.mx=hi+1e-4;bounds._pad0=c.params.w;bounds._pad1=asfloat(c.body);
    gAabbs[HOST_S(1)+i]=bounds;
    uint meta=fracMeta(h);float3 localLo=gHullVerts[meta+1u],localHi=gHullVerts[meta+2u];
    WorldObbGpu obb;obb.center=float4(transformPoint(h.p,h.q,(localLo+localHi)*0.5),0);
    obb.half=float4((localHi-localLo)*0.5,0);obb.axis0=float4(quatRotate(h.q,float3(1,0,0)),0);
    obb.axis1=float4(quatRotate(h.q,float3(0,1,0)),0);obb.axis2=float4(quatRotate(h.q,float3(0,0,1)),0);
    gWorldObbs[i]=obb;
}

// Geometric size hierarchy, inspired by three-avbd's fine/coarse ownership.
// Each hull is inserted once. Cell width >= its diameter; a fine proxy queries
// its own and every coarser level's 27 cells. No fixed-size large-object list.
uint fracLevel(Collider c){
    float diameter=2.0*(c.shape==SHAPE_SPHERE?c.params.x:c.params.w)+0.0004;
    return uint(max(0.0,ceil(log2(max(diameter/PARAMS.gridCellSize,1.0)))));
}
float fracCellSize(uint level){return PARAMS.gridCellSize*exp2(float(level));}
int3 fracCell(Aabb a,uint level){return int3(floor((a.mn+a.mx)*(0.5/fracCellSize(level))));}
uint fracHash(int3 cell,uint level){return (gridHash(cell.x,cell.y,cell.z)^(level*2654435769u))&(GRID_HASH_SIZE-1u);}

[shader("compute")]
[numthreads(64,1,1)]
void fracture_broadphase_insert(uint3 tid:SV_DispatchThreadID){
    uint i=dispatchIndex(tid);if(i>=gCounters[0].colliderCount)return;
    Collider c=gColliders[i];if(c.shape==SHAPE_PLANE)return;
    uint level=fracLevel(c),prev;
    uint hash=fracHash(fracCell(gAabbs[HOST_S(1)+i],level),level);
    AXCHG3(gConstraintIds[hash],i,prev);gColorLists[i]=prev;
}
bool fracMoving(BodyState b){return bodyIsDynamic(b)&&!bodyIsSleeping(b);}
bool fracOverlap(Aabb a,Aabb b){return all(a.mn<=b.mx)&&all(b.mn<=a.mx);}
bool fracObbSeparated(uint i,uint j){
    Collider a=gColliders[i],b=gColliders[j];
    if(a.shape!=SHAPE_CONVEX||b.shape!=SHAPE_CONVEX)return false;
    BoxObb aa=worldObbFromCache(i),bb=worldObbFromCache(j);
    float3 d=aa.center-bb.center;
    float3 axes[6]={aa.axis0,aa.axis1,aa.axis2,bb.axis0,bb.axis1,bb.axis2};
    for(uint k=0;k<6;k++){
        float3 n=axes[k];
        float ra=dot(aa.half,abs(float3(dot(n,aa.axis0),dot(n,aa.axis1),dot(n,aa.axis2))));
        float rb=dot(bb.half,abs(float3(dot(n,bb.axis0),dot(n,bb.axis1),dot(n,bb.axis2))));
        if(abs(dot(d,n))>ra+rb+0.001)return true;
    }
    return false;
}
[shader("compute")]
[numthreads(64,1,1)]
void fracture_broadphase_query(uint3 tid:SV_DispatchThreadID){
    uint i=dispatchIndex(tid),count=gCounters[0].colliderCount;if(i>=count)return;
    Collider a=gColliders[i];if(a.shape==SHAPE_PLANE)return;
    BodyState ba=gBodies[a.body];bool active=fracMoving(ba);
    Aabb aa=gAabbs[HOST_S(1)+i];uint own=fracLevel(a),levels=HOST_S(18);
    // Plane indices are a small host-provided list in gEvents, outside grid scratch.
    if(active)for(uint p=0;p<HOST_S(19);p++)emitCandidate(i,ALOAD(gEvents[p]));
    // Fixed/fine -> moving/coarse is necessary: coarse movers never query fine.
    for(uint level=own;level<levels;level++){
        if(!active&&level==own)continue;
        int3 center=fracCell(aa,level);
        for(int z=-1;z<=1;z++)for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){
            int3 cell=center+int3(x,y,z);uint j=ALOAD(gConstraintIds[fracHash(cell,level)]),guard=0;
            while(j!=0xFFFFFFFFu&&guard<count){
                guard++;Collider b=gColliders[j];Aabb ab=gAabbs[HOST_S(1)+j];
                if(j!=i&&b.body!=a.body&&fracLevel(b)==level&&all(fracCell(ab,level)==cell)&&fracOverlap(aa,ab)){
                    bool other=fracMoving(gBodies[b.body]);
                    bool owner=level>own||(active&&(!other||i<j));
                    if((active||other)&&owner&&!fracObbSeparated(i,j))emitCandidate(i,j);
                }
                j=gColorLists[j];
            }
            if(j!=0xFFFFFFFFu)AOR2(gCounters[0].overflowFlags,OVERFLOW_BP_CHAIN);
        }
    }
}
