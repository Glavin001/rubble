// Convex hull narrowphase entries (hull/hull, hull/box, hull/plane, hull/sphere,
// hull/capsule). Gauss-map pruning and contact selection adapted from three-avbd
// (MIT), Copyright (c) 2026 Steven Bobyn; see ../THIRD_PARTY_LICENSES.txt.
//
// Schedule, after `narrowphase` has marked hull pairs with CONVEX_DEFER:
//   convex_classify    one thread per candidate (indirect slot 64). Plane, sphere
//                      and capsule pairs emit contacts directly; hull/hull and
//                      hull/box pairs are compacted into the small (front) or
//                      large (back) SAT bucket of gConvexWork.
//   convex_write_args  slots 70 (small groups), 71 (large groups), 72 (pairs).
//   convex_sat_small   16 lanes per pair, 4 pairs per workgroup (V <= 16).
//   convex_sat_large   64 lanes per pair (V <= 64, F <= 128, E <= 192).
//   convex_manifold    one thread per SAT pair (slot 72): clip and emit.
// Each entry stays within WebGPU's per-stage storage-buffer budget, which is why
// this work is split from `narrowphase` instead of branching inside it.
#if !defined(CONVEX_SKIP_WORLD_INCLUDE)
#include "world_step.slang"
#endif

#ifndef CONVEX_FAST_FACE
#define CONVEX_FAST_FACE 1
#endif

struct ConvexWork
{
    float4 normalSep;  // world A->B normal, separation (after SAT)
    uint candidate;    // index into gCandidates
    uint feature;      // type | idx1 << 8 | idx2 << 16 | CONVEX_WORK_FAST
    uint _pad0;        // best edge pair (ea << 8 | eb) | CONVEX_WORK_EDGE_VALID
    uint _pad1;        // best edge separation (float bits)
};
static const uint CONVEX_WORK_FAST = 0x01000000u;
static const uint CONVEX_WORK_EDGE_VALID = 0x80000000u;
static const uint CONVEX_SLOT_SMALL_GROUPS = 74u;
static const uint CONVEX_SLOT_LARGE_GROUPS = 75u;
static const uint CONVEX_SLOT_PAIRS = 76u;

// Appended after every world_step binding (Metal assigns buffer indices in
// declaration order; WGSL uses the explicit binding). Sized maxCandidates: each
// candidate contributes at most one item, so the two buckets never collide.
BIND(29) RWStructuredBuffer<ConvexWork> gConvexWork;

[shader("compute")]
[numthreads(64, 1, 1)]
void convex_classify(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint n = min(ALOAD(gCounters[0].candidateCount), HOST_S(2));
    if (i >= n) return;
    CandidatePair pair = gCandidates[i];
    if ((pair._pad & CONVEX_DEFER) == 0u) return;
    Collider ca = gColliders[pair.colliderA];
    Collider cb = gColliders[pair.colliderB];
    if (shapeHasHull(ca.shape) && shapeHasHull(cb.shape))
    {
        HullRef ha = hullOf(ca);
        HullRef hb = hullOf(cb);
        bool small = ha.nv <= 16u && hb.nv <= 16u && ha.nf <= 32u && hb.nf <= 32u &&
                     ha.ne <= 48u && hb.ne <= 48u;
        uint slot;
        if (small)
            AADD3(gCounters[0].convexSmall, 1, slot);
        else
        {
            AADD3(gCounters[0].convexLarge, 1, slot);
            slot = HOST_S(2) - 1u - slot;
        }
        ConvexWork w;
        w.normalSep = float4(0, 0, 0, 0);
        w.candidate = i;
        w.feature = 0u;
        w._pad0 = 0u;
        w._pad1 = 0u;
        gConvexWork[slot] = w;
        return;
    }
    BodyState ba = gBodies[ca.body];
    BodyState bb = gBodies[cb.body];
    Frame fa = colliderFrame(ca, ba);
    Frame fb = colliderFrame(cb, bb);
    BoxManifold mf;
    mf.count = 0;
    mf.normal = float3(0, 1, 0);
    if (ca.shape == SHAPE_CONVEX && cb.shape == SHAPE_PLANE)
        mf = convexPlaneManifold(hullOf(ca), fa, cb.params.xyz, cb.params.w, true);
    else if (ca.shape == SHAPE_PLANE && cb.shape == SHAPE_CONVEX)
        mf = convexPlaneManifold(hullOf(cb), fb, ca.params.xyz, ca.params.w, false);
    else if (ca.shape == SHAPE_CONVEX && cb.shape == SHAPE_SPHERE)
        mf = convexSphereManifold(hullOf(ca), fa, fb.p, cb.params.x, true);
    else if (ca.shape == SHAPE_SPHERE && cb.shape == SHAPE_CONVEX)
        mf = convexSphereManifold(hullOf(cb), fb, fa.p, ca.params.x, false);
    else if (ca.shape == SHAPE_CONVEX && cb.shape == SHAPE_CAPSULE)
        mf = convexCapsuleManifold(hullOf(ca), fa, fb, cb.params.x, cb.params.y, true);
    else if (ca.shape == SHAPE_CAPSULE && cb.shape == SHAPE_CONVEX)
        mf = convexCapsuleManifold(hullOf(cb), fb, fa, ca.params.x, ca.params.y, false);
    if (mf.count > 0)
        emitContactManifold(pair, ba, bb, ca, cb, mf, PARAMS.contactOffset);
}

void writeGroupArgs(uint slot, uint groups)
{
    uint rows = max(1u, (groups + 32767u) / 32768u);
    gIndirectArgs[slot * 3u] = (groups + rows - 1u) / rows;
    gIndirectArgs[slot * 3u + 1u] = rows;
    gIndirectArgs[slot * 3u + 2u] = 1u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void convex_write_args(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0u) return;
    uint small = ALOAD(gCounters[0].convexSmall);
    uint large = ALOAD(gCounters[0].convexLarge);
    writeGroupArgs(CONVEX_SLOT_SMALL_GROUPS, (small + 3u) / 4u);
    writeGroupArgs(CONVEX_SLOT_LARGE_GROUPS, large);
    writeDispatchArgs(CONVEX_SLOT_PAIRS, small + large);
}

#define CSAT_ENTRY convex_sat_small
#define CSAT_SFX _small
#define CSAT_LANES 16u
#define CSAT_MAXV 16u
#define CSAT_MAXF 32u
#define CSAT_MAXE 48u
#define CSAT_BUCKET 0
#include "convex_sat_body.slang"
#undef CSAT_ENTRY
#undef CSAT_SFX
#undef CSAT_LANES
#undef CSAT_MAXV
#undef CSAT_MAXF
#undef CSAT_MAXE
#undef CSAT_BUCKET

#define CSAT_ENTRY convex_sat_large
#define CSAT_SFX _large
#define CSAT_LANES 64u
#define CSAT_MAXV 64u
#define CSAT_MAXF 128u
#define CSAT_MAXE 192u
#define CSAT_BUCKET 1
#include "convex_sat_body.slang"
#undef CSAT_ENTRY
#undef CSAT_SFX
#undef CSAT_LANES
#undef CSAT_MAXV
#undef CSAT_MAXF
#undef CSAT_MAXE
#undef CSAT_BUCKET


[shader("compute")]
[numthreads(64, 1, 1)]
void convex_manifold(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint small = ALOAD(gCounters[0].convexSmall);
    uint large = ALOAD(gCounters[0].convexLarge);
    if (i >= small + large) return;
    uint slot = i < small ? i : HOST_S(2) - 1u - (i - small);
    ConvexWork w = gConvexWork[slot];
    uint type = w.feature & 255u;
    if (type == 0u) return;
    uint idx1 = (w.feature >> 8) & 255u;
    uint idx2 = (w.feature >> 16) & 255u;
    bool fast = (w.feature & CONVEX_WORK_FAST) != 0u;
    CandidatePair pair = gCandidates[w.candidate];
    Collider ca = gColliders[pair.colliderA];
    Collider cb = gColliders[pair.colliderB];
    BodyState ba = gBodies[ca.body];
    BodyState bb = gBodies[cb.body];
    Frame fa = colliderFrame(ca, ba);
    Frame fb = colliderFrame(cb, bb);
    HullRef ha = hullOf(ca);
    HullRef hb = hullOf(cb);
    float3 n = w.normalSep.xyz;
    BoxManifold mf;
    if (type == CONVEX_FEATURE_EDGE)
        mf = convexEdgeManifold(ha, fa, hb, fb, idx1, idx2, n);
    else if (type == CONVEX_FEATURE_FACE_A)
        mf = convexFaceManifold(ha, fa, hb, fb, idx1, true, n, false);
    else
        mf = convexFaceManifold(hb, fb, ha, fa, idx1, false, n, false);
    if (mf.count == 0 && type != CONVEX_FEATURE_EDGE)
    {
        // Face bias kept a face whose patch clips empty: the overlap is edge-
        // dominated. Use the best edge pair (from SAT, or a scalar query when the
        // cached-face fast path skipped edges). A projected support point is the
        // last resort; it can sit off the reference surface.
        uint ea = 0u, eb = 0u;
        float edgeSep = -1e30;
        bool edgeValid = false;
        if (!fast)
        {
            edgeValid = (w._pad0 & CONVEX_WORK_EDGE_VALID) != 0u;
            ea = (w._pad0 >> 8) & 255u;
            eb = w._pad0 & 255u;
            edgeSep = asfloat(w._pad1);
        }
        else
        {
            float3 unused;
            edgeSep = convexEdgeQueryScalar(ha, fa, hb, fb, ea, eb, unused);
            edgeValid = edgeSep > -1e29;
        }
        if (edgeValid && edgeSep > 0.0) return;
        if (edgeValid)
            mf = convexEdgeManifold(ha, fa, hb, fb, ea, eb, convexEdgeAxis(ha, fa, hb, fb, ea, eb));
        if (mf.count == 0)
        {
            if (type == CONVEX_FEATURE_FACE_A)
                mf = convexFaceManifold(ha, fa, hb, fb, idx1, true, n, true);
            else
                mf = convexFaceManifold(hb, fb, ha, fa, idx1, false, n, true);
        }
    }
    if (mf.count > 0)
        emitContactManifold(pair, ba, bb, ca, cb, mf, PARAMS.contactOffset);
}
