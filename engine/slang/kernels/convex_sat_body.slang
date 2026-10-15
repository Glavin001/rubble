// Cooperative convex SAT, instantiated twice by convex_step.slang (small/large
// buckets). Parameters: CSAT_ENTRY, CSAT_SFX, CSAT_LANES (lanes per pair),
// CSAT_MAXV/MAXF/MAXE (shared capacity per pair), CSAT_BUCKET (0 small, 1 large).
//
// A workgroup of 64 lanes processes 64/CSAT_LANES pairs. Each pair's segment
// caches A in B's local frame (one transform per vertex/plane, not per axis
// test) plus B's local data in shared memory, then strides face axes and A edges
// across its lanes. Closed hulls prune edge pairs with the Gauss map; uncertified
// hulls fall back to exhaustive projections.
//
// The kernel is a sequence of phases separated by workgroup barriers. Each phase
// is a function of (group, lane) plus shared memory, so the GPU entry runs them
// with barriers in uniform control flow (no lane returns early), and the CPU
// conformance entry (ENGINE_CPU_TEST) runs every lane through a phase before the
// next one: the same barrier semantics without hardware barriers.

#ifndef CSAT_PASTE2
#define CSAT_PASTE2(a, b) a##b
#define CSAT_PASTE(a, b) CSAT_PASTE2(a, b)
#endif
#define CS(x) CSAT_PASTE(x, CSAT_SFX)
#define CSAT_PAIRS (64u / CSAT_LANES)

groupshared float4 CS(gsAV)[CSAT_PAIRS * CSAT_MAXV];
groupshared float4 CS(gsBV)[CSAT_PAIRS * CSAT_MAXV];
groupshared float4 CS(gsAP)[CSAT_PAIRS * CSAT_MAXF];
groupshared float4 CS(gsBP)[CSAT_PAIRS * CSAT_MAXF];
groupshared uint CS(gsAE)[CSAT_PAIRS * CSAT_MAXE];
groupshared uint CS(gsBE)[CSAT_PAIRS * CSAT_MAXE];
groupshared float CS(gsScore)[64];
groupshared float CS(gsSep)[64];
groupshared uint CS(gsKey)[64];
groupshared float CS(gsFaceSep)[CSAT_PAIRS];
groupshared uint CS(gsFaceKey)[CSAT_PAIRS];

struct CS(SatLane)
{
    uint lane;   // 0..63 in the workgroup
    uint seg;    // pair within the workgroup
    uint l;      // lane within the pair's segment
    uint slot;   // gConvexWork index
    bool valid;
    uint hint;
    HullRef ha;
    HullRef hb;
    Frame fb;
    float4 qRel; // A -> B local
    float3 tRel;
    float3 interiorA;
    float3 interiorB;
    uint colA;
    uint colB;
};

CS(SatLane) CS(satLane)(uint3 group, uint lane)
{
    CS(SatLane) s;
    uint count = CSAT_BUCKET == 0 ? ALOAD(gCounters[0].convexSmall) : ALOAD(gCounters[0].convexLarge);
    uint groups = (count + CSAT_PAIRS - 1u) / CSAT_PAIRS;
    // Balanced rows (convex_write_args): never launch padding-only workgroups.
    uint rows = max(1u, (groups + 32767u) / 32768u);
    uint width = (groups + rows - 1u) / rows;
    uint g = group.x + group.y * width;
    s.lane = lane;
    s.seg = lane / CSAT_LANES;
    s.l = lane % CSAT_LANES;
    uint item = g * CSAT_PAIRS + s.seg;
    s.valid = item < count;
    s.slot = CSAT_BUCKET == 0 ? item : HOST_S(2) - 1u - item;
    s.ha = (HullRef)0;
    s.hb = (HullRef)0;
    Frame fa = (Frame)0;
    fa.q = quatIdentity();
    s.fb = (Frame)0;
    s.fb.q = quatIdentity();
    s.hint = 0u;
    if (s.valid)
    {
        ConvexWork w = gConvexWork[s.slot];
        CandidatePair pair = gCandidates[w.candidate];
        s.hint = pair._pad & 0x3FFFFu;
        s.colA = pair.colliderA;
        s.colB = pair.colliderB;
        Collider ca = gColliders[pair.colliderA];
        Collider cb = gColliders[pair.colliderB];
        fa = colliderFrame(ca, gBodies[ca.body]);
        s.fb = colliderFrame(cb, gBodies[cb.body]);
        s.ha = hullOf(ca);
        s.hb = hullOf(cb);
    }
    float4 qBi = quatConjugate(s.fb.q);
    s.qRel = quatMul(qBi, fa.q);
    s.tRel = quatRotate(qBi, fa.p - s.fb.p);
    s.interiorA = s.valid ? s.tRel + quatRotate(s.qRel, hullInterior(s.ha)) : float3(0, 0, 0);
    s.interiorB = s.valid ? hullInterior(s.hb) : float3(0, 0, 0);
    return s;
}

uint CS(hintFace)(uint hint)
{
    uint t = hint >> 16;
    return t == CONVEX_FEATURE_FACE_A ? ((hint >> 8) & 255u)
         : t == CONVEX_FEATURE_FACE_B ? 128u + ((hint >> 8) & 255u)
         : 0xFFFFu;
}

float2 CS(projectA)(uint seg, uint nv, float3 n)
{
    float2 r = float2(1e30, -1e30);
    for (uint v = 0u; v < nv; v++)
    {
        float d = dot(CS(gsAV)[seg * CSAT_MAXV + v].xyz, n);
        r = float2(min(r.x, d), max(r.y, d));
    }
    return r;
}

float2 CS(projectB)(uint seg, uint nv, float3 n)
{
    float2 r = float2(1e30, -1e30);
    for (uint v = 0u; v < nv; v++)
    {
        float d = dot(CS(gsBV)[seg * CSAT_MAXV + v].xyz, n);
        r = float2(min(r.x, d), max(r.y, d));
    }
    return r;
}

// Phase 1: cache A (in B's frame) and B into the segment's shared slots.
void CS(phaseLoad)(CS(SatLane) s)
{
    uint vOff = s.seg * CSAT_MAXV;
    uint fOff = s.seg * CSAT_MAXF;
    uint eOff = s.seg * CSAT_MAXE;
    for (uint v = s.l; v < s.ha.nv; v += CSAT_LANES)
        CS(gsAV)[vOff + v] = float4(s.tRel + quatRotate(s.qRel, hullVertex(s.ha, v)), 0.0);
    for (uint v = s.l; v < s.hb.nv; v += CSAT_LANES)
        CS(gsBV)[vOff + v] = float4(hullVertex(s.hb, v), 0.0);
    for (uint f = s.l; f < s.ha.nf; f += CSAT_LANES)
    {
        float4 p = hullPlane(s.ha, f);
        float3 n = quatRotate(s.qRel, p.xyz);
        CS(gsAP)[fOff + f] = float4(n, p.w + dot(n, s.tRel));
    }
    for (uint f = s.l; f < s.hb.nf; f += CSAT_LANES)
        CS(gsBP)[fOff + f] = hullPlane(s.hb, f);
    for (uint e = s.l; e < s.ha.ne; e += CSAT_LANES)
        CS(gsAE)[eOff + e] = hullWord(s.ha, s.ha.edgeWord + e);
    for (uint e = s.l; e < s.hb.ne; e += CSAT_LANES)
        CS(gsBE)[eOff + e] = hullWord(s.hb, s.hb.edgeWord + e);
}

// The pair's cached separating feature is tested before bucketing
// (convex_bucket, satCacheSeparates), so every pair here runs the full SAT;
// phaseFinish records the feature that proves a separation for next step.

// Phase 2: face axes strided over the segment; the previous frame's face gets
// a small hysteresis bonus (stable features and warm starts).
void CS(phaseFaces)(CS(SatLane) s)
{
    uint fOff = s.seg * CSAT_MAXF;
    uint hintFace = CS(hintFace)(s.hint);
    bool closedA = hullClosed(s.ha);
    bool closedB = hullClosed(s.hb);
    float bestScore = -1e30;
    float bestSep = -1e30;
    uint bestKey = 0xFFFFFFFFu;
    uint faces = s.ha.nf + s.hb.nf;
    for (uint f = s.l; f < faces; f += CSAT_LANES)
    {
        bool fromA = f < s.ha.nf;
        uint fi = fromA ? f : f - s.ha.nf;
        float sep;
        if (fromA)
        {
            float4 pl = CS(gsAP)[fOff + fi];
            float hi = closedA ? pl.w : CS(projectA)(s.seg, s.ha.nv, pl.xyz).y;
            sep = CS(projectB)(s.seg, s.hb.nv, pl.xyz).x - hi;
        }
        else
        {
            float4 pl = CS(gsBP)[fOff + fi];
            float hi = closedB ? pl.w : CS(projectB)(s.seg, s.hb.nv, pl.xyz).y;
            sep = CS(projectA)(s.seg, s.ha.nv, pl.xyz).x - hi;
        }
        uint key = fromA ? fi : 128u + fi;
        float score = sep + (key == hintFace ? CONVEX_HYSTERESIS : 0.0);
        if (score > bestScore || (score == bestScore && key < bestKey))
        {
            bestScore = score;
            bestSep = sep;
            bestKey = key;
        }
    }
    CS(gsScore)[s.lane] = bestScore;
    CS(gsSep)[s.lane] = bestSep;
    CS(gsKey)[s.lane] = bestKey;
}

// One tree-reduction level: higher score wins, ties go to the smaller key.
void CS(phaseReduce)(CS(SatLane) s, uint stride)
{
    if (s.l < stride)
    {
        uint o = s.lane + stride;
        float a = CS(gsScore)[s.lane];
        float b = CS(gsScore)[o];
        if (b > a || (b == a && CS(gsKey)[o] < CS(gsKey)[s.lane]))
        {
            CS(gsScore)[s.lane] = b;
            CS(gsSep)[s.lane] = CS(gsSep)[o];
            CS(gsKey)[s.lane] = CS(gsKey)[o];
        }
    }
}

void CS(phasePublishFace)(CS(SatLane) s)
{
    if (s.l == 0u)
    {
        CS(gsFaceSep)[s.seg] = CS(gsSep)[s.lane];
        CS(gsFaceKey)[s.seg] = CS(gsKey)[s.lane];
    }
}

// Temporal coherence: when last frame's reference face still wins the face
// query, skip the O(Ea·Eb) edge query (70%+ of hull SAT time). convex_manifold
// resolves the rare empty patch with a scalar edge query instead.
bool CS(fastFace)(CS(SatLane) s)
{
    return CONVEX_FAST_FACE != 0 && CS(gsFaceKey)[s.seg] == CS(hintFace)(s.hint);
}

// Phase 3: edge pairs, A edges strided over the segment.
void CS(phaseEdges)(CS(SatLane) s)
{
    uint vOff = s.seg * CSAT_MAXV;
    uint fOff = s.seg * CSAT_MAXF;
    uint eOff = s.seg * CSAT_MAXE;
    bool closedBoth = hullClosed(s.ha) && hullClosed(s.hb);
    uint hintEdge = (s.hint >> 16) == CONVEX_FEATURE_EDGE ? (s.hint & 0xFFFFu) : 0xFFFFFFFFu;
    bool doEdges = s.valid && CS(gsFaceSep)[s.seg] <= speculativeMargin() && !CS(fastFace)(s);
    float edgeScore = -1e30;
    float edgeSep = -1e30;
    uint edgeKey = 0xFFFFFFFFu;
    if (doEdges)
    {
        for (uint ea = s.l; ea < s.ha.ne; ea += CSAT_LANES)
        {
            uint e1 = CS(gsAE)[eOff + ea];
            float3 pa = CS(gsAV)[vOff + (e1 & 255u)].xyz;
            float3 da = CS(gsAV)[vOff + ((e1 >> 8) & 255u)].xyz - pa;
            float3 na = float3(0, 0, 0);
            float3 nb = float3(0, 0, 0);
            float3 bxa = float3(0, 0, 0);
            if (closedBoth)
            {
                na = CS(gsAP)[fOff + ((e1 >> 16) & 255u)].xyz;
                nb = CS(gsAP)[fOff + (e1 >> 24)].xyz;
                bxa = cross(nb, na);
            }
            for (uint eb = 0u; eb < s.hb.ne; eb++)
            {
                uint e2 = CS(gsBE)[eOff + eb];
                if (closedBoth)
                {
                    // Gregorius: only Minkowski-face edge pairs can be separating axes.
                    float3 c = -CS(gsBP)[fOff + ((e2 >> 16) & 255u)].xyz;
                    float3 d = -CS(gsBP)[fOff + (e2 >> 24)].xyz;
                    float cba = dot(c, bxa);
                    float dba = dot(d, bxa);
                    if (cba * dba >= 0.0) continue;
                    float3 dxc = cross(d, c);
                    float adc = dot(na, dxc);
                    float bdc = dot(nb, dxc);
                    if (adc * bdc >= 0.0 || cba * bdc <= 0.0) continue;
                }
                float3 pb = CS(gsBV)[vOff + (e2 & 255u)].xyz;
                float3 db = CS(gsBV)[vOff + ((e2 >> 8) & 255u)].xyz - pb;
                float3 n = cross(da, db);
                float len = length(n);
                if (len < 1e-5 * length(da) * length(db) || len < 1e-12) continue;
                n /= len;
                float sep;
                if (closedBoth)
                {
                    if (dot(n, pa - s.interiorA) < 0.0) n = -n;
                    sep = dot(n, pb - pa);
                }
                else
                {
                    if (dot(n, s.interiorB - s.interiorA) < 0.0) n = -n;
                    sep = CS(projectB)(s.seg, s.hb.nv, n).x - CS(projectA)(s.seg, s.ha.nv, n).y;
                }
                uint key = (ea << 8) | eb;
                float score = sep + (key == hintEdge ? CONVEX_HYSTERESIS : 0.0);
                if (score > edgeScore || (score == edgeScore && key < edgeKey))
                {
                    edgeScore = score;
                    edgeSep = sep;
                    edgeKey = key;
                }
            }
        }
    }
    CS(gsScore)[s.lane] = edgeScore;
    CS(gsSep)[s.lane] = edgeSep;
    CS(gsKey)[s.lane] = edgeKey;
}

// Phase 4: the segment's first lane picks face vs edge and writes the result.
void CS(phaseFinish)(CS(SatLane) s)
{
    if (!s.valid || s.l != 0u) return;
    uint vOff = s.seg * CSAT_MAXV;
    uint fOff = s.seg * CSAT_MAXF;
    uint eOff = s.seg * CSAT_MAXE;
    float faceSep = CS(gsFaceSep)[s.seg];
    uint faceKey = CS(gsFaceKey)[s.seg];
    float edgeSep = CS(gsSep)[s.lane];
    uint edgeKey = CS(gsKey)[s.lane];
    bool fast = CS(fastFace)(s);
    ConvexWork w = gConvexWork[s.slot];
    float3 nB = float3(0, 1, 0);
    float sep = faceSep;
    uint type = 0u;
    uint idx1 = 0u;
    uint idx2 = 0u;
    bool edgeHit = edgeKey != 0xFFFFFFFFu;
    float spec = speculativeMargin();
    if (faceSep <= spec && !(edgeHit && edgeSep > spec))
    {
        // A positive gap (speculative contact) is measured on the axis of largest
        // separation; overlapping pairs keep the face bias.
        bool gap = faceSep > 0.0 || (edgeHit && edgeSep > 0.0);
        if (edgeHit && (gap ? edgeSep > faceSep : edgeSep > CONVEX_EDGE_REL * faceSep + CONVEX_EDGE_ABS))
        {
            uint ea = edgeKey >> 8;
            uint eb = edgeKey & 255u;
            uint e1 = CS(gsAE)[eOff + ea];
            uint e2 = CS(gsBE)[eOff + eb];
            float3 pa = CS(gsAV)[vOff + (e1 & 255u)].xyz;
            float3 da = CS(gsAV)[vOff + ((e1 >> 8) & 255u)].xyz - pa;
            float3 pb = CS(gsBV)[vOff + (e2 & 255u)].xyz;
            float3 db = CS(gsBV)[vOff + ((e2 >> 8) & 255u)].xyz - pb;
            nB = normalize(cross(da, db));
            float3 away = hullClosed(s.ha) && hullClosed(s.hb) ? pa - s.interiorA : s.interiorB - s.interiorA;
            if (dot(nB, away) < 0.0) nB = -nB;
            sep = edgeSep;
            type = CONVEX_FEATURE_EDGE;
            idx1 = ea;
            idx2 = eb;
        }
        else if (faceKey < 128u)
        {
            nB = CS(gsAP)[fOff + faceKey].xyz;
            type = CONVEX_FEATURE_FACE_A;
            idx1 = faceKey;
        }
        else
        {
            nB = -CS(gsBP)[fOff + faceKey - 128u].xyz;
            type = CONVEX_FEATURE_FACE_B;
            idx1 = faceKey - 128u;
        }
    }
    if (type == 0u)
    {
        // Separated: remember the feature that proved it for the next step.
        uint feature = 0u;
        if (faceSep > spec)
            feature = faceKey < 128u ? (SAT_CACHE_FACE_A << 16) | (faceKey << 8)
                                     : (SAT_CACHE_FACE_B << 16) | ((faceKey - 128u) << 8);
        else if (edgeHit && edgeSep > spec)
            feature = (SAT_CACHE_EDGE << 16) | edgeKey;
        if (feature != 0u)
            gSatCache[satCacheSlot(s.colA, s.colB)] = uint2(satCacheCheck(s.colA, s.colB), feature);
    }
    w.normalSep = float4(quatRotate(s.fb.q, nB), sep);
    w.feature = type | (idx1 << 8) | (idx2 << 16) | ((fast && type != CONVEX_FEATURE_EDGE) ? CONVEX_WORK_FAST : 0u);
    // Best edge pair as well: an empty face patch falls back to it.
    w._pad0 = edgeHit ? (CONVEX_WORK_EDGE_VALID | edgeKey) : 0u;
    w._pad1 = asuint(edgeSep);
    gConvexWork[s.slot] = w;
}

#if !defined(ENGINE_CPU_TEST)
[shader("compute")]
[numthreads(64, 1, 1)]
void CSAT_ENTRY(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    CS(SatLane) s = CS(satLane)(group, lane);
    CS(phaseLoad)(s);
    GROUP_SYNC();
    CS(phaseFaces)(s);
    GROUP_SYNC();
    [ForceUnroll]
    for (uint stride = CSAT_LANES / 2u; stride > 0u; stride >>= 1u)
    {
        CS(phaseReduce)(s, stride);
        GROUP_SYNC();
    }
    CS(phasePublishFace)(s);
    GROUP_SYNC();
    CS(phaseEdges)(s);
    GROUP_SYNC();
    [ForceUnroll]
    for (uint stride = CSAT_LANES / 2u; stride > 0u; stride >>= 1u)
    {
        CS(phaseReduce)(s, stride);
        GROUP_SYNC();
    }
    CS(phaseFinish)(s);
}
#else
// CPU conformance: one invocation per workgroup; every lane completes a phase
// before any lane starts the next, exactly as the barriers order the GPU entry.
[shader("compute")]
[numthreads(1, 1, 1)]
void CSAT_ENTRY(uint3 group : SV_GroupID)
{
    for (uint lane = 0u; lane < 64u; lane++) CS(phaseLoad)(CS(satLane)(group, lane));
    for (uint lane = 0u; lane < 64u; lane++) CS(phaseFaces)(CS(satLane)(group, lane));
    for (uint stride = CSAT_LANES / 2u; stride > 0u; stride >>= 1u)
        for (uint lane = 0u; lane < 64u; lane++) CS(phaseReduce)(CS(satLane)(group, lane), stride);
    for (uint lane = 0u; lane < 64u; lane++) CS(phasePublishFace)(CS(satLane)(group, lane));
    for (uint lane = 0u; lane < 64u; lane++) CS(phaseEdges)(CS(satLane)(group, lane));
    for (uint stride = CSAT_LANES / 2u; stride > 0u; stride >>= 1u)
        for (uint lane = 0u; lane < 64u; lane++) CS(phaseReduce)(CS(satLane)(group, lane), stride);
    for (uint lane = 0u; lane < 64u; lane++) CS(phaseFinish)(CS(satLane)(group, lane));
}
#endif

#undef CS
#undef CSAT_PAIRS
