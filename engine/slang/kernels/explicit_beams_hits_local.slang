// --- PERF_HITS_LOCAL: selfApply's two-body path from workgroup memory ---
// With PERF_HIT_FB and PERF_HIT_ROLES, a two-body dissipative vehicle's selfApply is:
// hitFbFill; one round (two-body: no hit reads another's force, so round 1 computes
// every open hit); selfResidualRoles' seven phases. Every value those phases pass
// between them is a node's (its beam force, mass, hit count, accumulators A and B) or a
// hit's (its record, its role records), each re-read from device memory in every phase
// (a chain of dependent device accesses at 0.15-0.2 us each, §8.23). Here item q (hit
// q >> 2, role q & 3: 0 the contacter, 1..3 the triangle's n0, n1, n2) on lane q loads
// them once; the phases run on workgroup copies; the accumulators are written back once.
// A node named by several items has one copy (the first item naming it: its slot), so
// the integer adds of every item land on one sum as they did in device memory (integer
// adds: any order gives the same bits). Each float computes the same strict arithmetic
// on the same bits in the same order as selfTwoBodyCore, hitRolePhase and
// hitCombinePhase. The scratch records those phases leave in gNext (the beam forces, the
// hit's force, round, residual, its role records and total) are written as they leave
// them. Workgroup arrays: the list kernels' (gsTested, gsPairs, gsCount), which the hits
// kernel does not otherwise use. Its barriers are workgroup barriers only (sync() adds a
// device-memory barrier, which waits for the phase's outstanding device writes): no lane
// here reads a device word another lane wrote here (the phases exchange values through
// the workgroup copies), and the caller's sync() after it orders its device writes for
// the tyres.
static const uint HITS_LOCAL_CAP = 48u;
// gsTested: per item LQ words at q * LQ (node, slot | immovable << 31, mass, beam force
// xyz, inverse mass of its hit count), then per hit LH words at LH_BASE + i * LH (alpha,
// beta, normal xyz, relative velocity xyz, first-pass force xyz, residual r).
static const uint LQ = 7u;
static const uint LH = 12u;
static const uint LH_BASE = HITS_LOCAL_CAP * 4u * LQ;
// gsPairs: per item its role record (acceleration xyz, inverse mass) at q * 4, then per
// hit its total xyz at LT_BASE + i * 3.
static const uint LT_BASE = HITS_LOCAL_CAP * 4u * 4u;
// gsCount: per slot A xyz at slot * 3 and B xyz at LB_BASE + slot * 3 (as uint: the
// same two's complement adds).
static const uint LB_BASE = HITS_LOCAL_CAP * 4u * 3u;

// The node acceleration of item q's slot copy (resAccNow's arithmetic): beams and A,
// with B when `withB`.
float3 localAccNow(uint q, uint slot, bool withB)
{
    uint s = gsTested[q * LQ + 1u];
    if ((s >> 31) != 0u) return float3(0.0, 0.0, 0.0);
    float3 f = asfloat(uint3(gsTested[q * LQ + 3u], gsTested[q * LQ + 4u], gsTested[q * LQ + 5u]));
    int3 a = int3(asint(gsCount[slot * 3u].load()), asint(gsCount[slot * 3u + 1u].load()),
                  asint(gsCount[slot * 3u + 2u].load()));
    if (withB)
        a += int3(asint(gsCount[LB_BASE + slot * 3u].load()), asint(gsCount[LB_BASE + slot * 3u + 1u].load()),
                  asint(gsCount[LB_BASE + slot * 3u + 2u].load()));
    float3 c = float3(float(a.x), float(a.y), float(a.z)) * (1.0 / FIXED);
    return sdiv3(sadd3(f, c), asfloat(gsTested[q * LQ + 2u]));
}

// Item q's selfAcc (beams over mass; 0 immovable) and own inverse mass from its copy.
float3 localSelfAcc(uint q)
{
    if ((gsTested[q * LQ + 1u] >> 31) != 0u) return float3(0.0, 0.0, 0.0);
    float3 f = asfloat(uint3(gsTested[q * LQ + 3u], gsTested[q * LQ + 4u], gsTested[q * LQ + 5u]));
    return sdiv3(f, asfloat(gsTested[q * LQ + 2u]));
}
float localOwnInvMass(uint q)
{
    if ((gsTested[q * LQ + 1u] >> 31) != 0u) return 0.0;
    return sdiv(1.0, asfloat(gsTested[q * LQ + 2u]));
}

// The round's position step d for item q's node (gsPairs is free until phase 1).
void localShift(uint q, float3 d)
{
    gsPairs[q * 4u] = asuint(d.x);
    gsPairs[q * 4u + 1u] = asuint(d.y);
    gsPairs[q * 4u + 2u] = asuint(d.z);
    gsPairs[q * 4u + 3u] = 1u;
}

void localAdd(uint at, int3 d)
{
    gsCount[at].add(asuint(d.x));
    gsCount[at + 1u].add(asuint(d.y));
    gsCount[at + 2u].add(asuint(d.z));
}

void selfApplyLocal(uint v, uint lane, uint beamOff, uint groupOff, uint groupCount, uint parity, uint mapNodes);
// PERF_HIT_SLOT_MAP (uniform): how many of vehicle v's nodes the slot map holds (all of them,
// at gsCount's first words, or 0: no map, each item scans the items before it).
uint hitSlotMapNodes(uint v)
{
    if ((gP.perfFlags & PERF_HIT_SLOT_MAP) == 0u) return 0u;
    uint n = gStatic[v * 4u].y;
    return n <= GRID_BUCKETS ? n : 0u;
}
// PERF_SELF_HITS_LOCAL (lane 0, the step's hit count final): whether vehicle v's hits of
// this step take selfApplyLocal on the self split path, as the wide hits kernel decides
// for PERF_HITS_LOCAL (a workgroup of at least 256 lanes: 4 · HITS_LOCAL_CAP items and
// REUSE_WORDS kept words, a lane each).
bool selfHitsLocalOn(uint v)
{
    if ((gP.perfFlags & PERF_SELF_HITS_LOCAL) == 0u || LANES < 256u) return false;
    uint hits = uint(gAcc[hitAcc(v)].load());
    return hits > 0u && hits <= HITS_LOCAL_CAP && hitRolesOn(v);
}
// PERF_SELF_HITS_LOCAL, every lane (the caller's choice uniform): selfApplyLocal on the
// self split path. There gsCount keeps the pair list's state past the buckets
// (soloStepLoad's REUSE_WORDS words, which the step reads on), and selfApplyLocal's
// accumulator copies use gsCount's first 4 · HITS_LOCAL_CAP · 6 words: each lane keeps one
// of those words in a register across it and puts it back. gsTested and gsPairs hold
// nothing there yet (the step fills them after). Then the device barrier selfApply's
// callers have after it, and gsUniformHits as selfApply leaves it (1: the caller's
// selfSplitHits set it).
void selfApplyLocalKeep(uint v, uint lane, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    uint kept = lane < REUSE_WORDS ? gsCount[GRID_BUCKETS + lane].load() : 0u;
    GroupMemoryBarrierWithGroupSync();
    selfApplyLocal(v, lane, beamOff, groupOff, groupCount, parity, 0u);
    sync();
    if (lane < REUSE_WORDS) gsCount[GRID_BUCKETS + lane].store(kept);
    if (lane == 0u) gsUniformHits = 1u;
}

// selfApply for vehicle v with PERF_HITS_LOCAL (the caller checked: hitRolesOn(v), 1..
// HITS_LOCAL_CAP hits, every lane here: the barriers are unconditional). mapNodes > 0
// (PERF_HIT_SLOT_MAP, uniform): gsCount's first mapNodes words are zero (the caller cleared
// them before a barrier), a word a node of v; the slots are found there, and each slot's
// copies of A and B are written one phase later (the round reads none of them), when no
// lane reads the map any more.
void selfApplyLocal(uint v, uint lane, uint beamOff, uint groupOff, uint groupCount, uint parity, uint mapNodes)
{
    uint count = uint(gAcc[hitAcc(v)].load());
    uint q = lane;
    bool item = q < count * 4u && count <= HITS_LOCAL_CAP;
    uint i = q >> 2;
    uint role = q & 3u;
    bool lead = item && role == 0u;
    uint at = hitList(v) + i * HIT_WORDS;
    uint x = 0u;
    uint fbAt = 0u;
    bool immovable = false;
    int3 a0 = int3(0, 0, 0);
    int3 b0 = int3(0, 0, 0);
    // The hit's own words (its lead lane keeps them for the round).
    uint h = 0u;
    uint t0 = 0u;
    uint t1 = 0u;
    uint t2 = 0u;
    float alpha = 0.0;
    float beta = 0.0;
    float3 nn = float3(0.0, 0.0, 0.0);
    float3 vv = float3(0.0, 0.0, 0.0);
    float pen = 0.0;
    float friction = 0.0;
    uint marked = 0u;
    float3 f1 = float3(0.0, 0.0, 0.0);
    // Load: hitFbFill's beam force (and its record), the node's mass, flags, hit count
    // and accumulators; the lead lane the hit's record.
    uint mapAt = 0xFFFFFFFFu;
    if (item)
    {
        uint nodeOff = gStatic[v * 4u].x;
        x = role == 0u ? nodeOff + (gNext[at] & 0xFFFFu) : gNext[at + 7u + role];
        // The slot map: the first item naming x keeps the largest 256 - q (q < 256).
        if (x - nodeOff < mapNodes)
        {
            mapAt = x - nodeOff;
            gsCount[mapAt].max(256u - q);
        }
        uint info = gStatic[gP.nodeInfoBase + x].x;
        immovable = (info & NODE_IMMOVABLE) != 0u;
        float mass = gNodes[x * 3u].w;
        float3 fb = beamSum(gNodes[x * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + x], beamOff, groupOff, groupCount, parity,
                            false);
        fbAt = hitFbAt(v, i) + role * 3u;
        gNext[fbAt] = asuint(fb.x);
        gNext[fbAt + 1u] = asuint(fb.y);
        gNext[fbAt + 2u] = asuint(fb.z);
        float imSelf = immovable ? 0.0 : sdiv(float(uint(gAcc[gP.selfSplit + x].load())), mass);
        a0 = int3(gAcc[x * 3u].load(), gAcc[x * 3u + 1u].load(), gAcc[x * 3u + 2u].load());
        uint b = resAt(x);
        b0 = int3(gAcc[b + 1u].load(), gAcc[b + 2u].load(), gAcc[b + 3u].load());
        gsTested[q * LQ] = x;
        gsTested[q * LQ + 2u] = asuint(mass);
        gsTested[q * LQ + 3u] = asuint(fb.x);
        gsTested[q * LQ + 4u] = asuint(fb.y);
        gsTested[q * LQ + 5u] = asuint(fb.z);
        gsTested[q * LQ + 6u] = asuint(imSelf);
        gsPairs[q * 4u + 3u] = 0u;
        if (lead)
        {
            h = x;
            t0 = gNext[at + 8u];
            t1 = gNext[at + 9u];
            t2 = gNext[at + 10u];
            alpha = asfloat(gNext[at + 1u]);
            beta = asfloat(gNext[at + 2u]);
            nn = asfloat(uint3(gNext[at + 12u], gNext[at + 13u], gNext[at + 14u]));
            pen = asfloat(gNext[at + 15u]);
            vv = asfloat(uint3(gNext[at + 16u], gNext[at + 17u], gNext[at + 18u]));
            friction = asfloat(gNext[at + 20u]);
            marked = gNext[at + 6u];
            f1 = asfloat(uint3(gNext[at + 3u], gNext[at + 4u], gNext[at + 5u]));
            uint hb = LH_BASE + i * LH;
            gsTested[hb] = asuint(alpha);
            gsTested[hb + 1u] = asuint(beta);
            gsTested[hb + 2u] = asuint(nn.x);
            gsTested[hb + 3u] = asuint(nn.y);
            gsTested[hb + 4u] = asuint(nn.z);
            gsTested[hb + 5u] = asuint(vv.x);
            gsTested[hb + 6u] = asuint(vv.y);
            gsTested[hb + 7u] = asuint(vv.z);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // Slots: the first item naming the node; its copy of A and B starts as the device's.
    uint slot = q;
    if (item)
    {
        // (A node the map does not hold: every item naming it scans, as without the map.)
        if (mapAt != 0xFFFFFFFFu)
            slot = 256u - gsCount[mapAt].load();
        else
        {
            for (uint k = 0u, kStop = EB_END(99, k, q, 1u); k < kStop; k++)
            {
                if (gsTested[k * LQ] == x)
                {
                    slot = k;
                    break;
                }
            }
        }
        gsTested[q * LQ + 1u] = slot | (immovable ? 0x80000000u : 0u);
        if (slot == q && mapNodes == 0u)
        {
            gsCount[q * 3u].store(asuint(a0.x));
            gsCount[q * 3u + 1u].store(asuint(a0.y));
            gsCount[q * 3u + 2u].store(asuint(a0.z));
            gsCount[LB_BASE + q * 3u].store(asuint(b0.x));
            gsCount[LB_BASE + q * 3u + 1u].store(asuint(b0.y));
            gsCount[LB_BASE + q * 3u + 2u].store(asuint(b0.z));
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // PERF_HIT_SLOT_MAP: the slots' copies of A and B now (the map is no longer read).
    if (item && slot == q && mapNodes > 0u)
    {
        gsCount[q * 3u].store(asuint(a0.x));
        gsCount[q * 3u + 1u].store(asuint(a0.y));
        gsCount[q * 3u + 2u].store(asuint(a0.z));
        gsCount[LB_BASE + q * 3u].store(asuint(b0.x));
        gsCount[LB_BASE + q * 3u + 1u].store(asuint(b0.y));
        gsCount[LB_BASE + q * 3u + 2u].store(asuint(b0.z));
    }
    float gamma = nf(nf(1.0 - alpha) - beta);
    // The round (selfRound's two-body branch, selfTwoBodyCore with shifts): the hit's
    // four items are q, q + 1 (n0 = t0), q + 2 (n1 = t1), q + 3 (n2 = t2).
    if (lead)
    {
        if (marked == 0u)
        {
            float imt = nf(nf(selfTerm(alpha, asfloat(gsTested[(q + 2u) * LQ + 6u]))
                              + selfTerm(beta, asfloat(gsTested[(q + 3u) * LQ + 6u])))
                           + selfTerm(gamma, asfloat(gsTested[(q + 1u) * LQ + 6u])));
            float inv = nf(asfloat(gsTested[q * LQ + 6u]) + imt);
            float m = inv > 0.0 ? sdiv(1.0, inv) : 0.0;
            float3 a3 = sscale(localSelfAcc(q + 2u), alpha);
            a3 = sadd3(a3, sscale(localSelfAcc(q + 3u), beta));
            a3 = sadd3(a3, sscale(localSelfAcc(q + 1u), gamma));
            float3 ah = localSelfAcc(q);
            float3 fh = sscale(ssub3(ah, a3), m);
            uint hv = gStatic[gP.nodeInfoBase + h].w;
            Surface surf = contactSurfaceV(hv, hv);
            float budget;
            float3 f = contactResponse(fh, vv, m, nn, surf, pen, friction, budget);
            if (surf.dissipative != 0u && any(f != float3(0.0, 0.0, 0.0)))
            {
                float delta = shiftDepth(pen, budget, sdot(fh, nn));
                if (delta > 0.0)
                {
                    // The position steps for the items' own lanes to add (shiftAdd's
                    // integer adds: any order).
                    float qd = smul(delta, m);
                    localShift(q, sscale(nn, smul(qd, localOwnInvMass(q))));
                    if (alpha > 0.0) localShift(q + 2u, sscale(nn, -smul(alpha, smul(qd, localOwnInvMass(q + 2u)))));
                    if (beta > 0.0) localShift(q + 3u, sscale(nn, -smul(beta, smul(qd, localOwnInvMass(q + 3u)))));
                    if (gamma > 0.0) localShift(q + 1u, sscale(nn, -smul(gamma, smul(qd, localOwnInvMass(q + 1u)))));
                }
            }
            f1 = f;
            gNext[at + 3u] = asuint(f.x);
            gNext[at + 4u] = asuint(f.y);
            gNext[at + 5u] = asuint(f.z);
            gNext[at + 6u] = 1u;
        }
        uint hb = LH_BASE + i * LH;
        gsTested[hb + 8u] = asuint(f1.x);
        gsTested[hb + 9u] = asuint(f1.y);
        gsTested[hb + 10u] = asuint(f1.z);
    }
    GroupMemoryBarrierWithGroupSync();
    // hitRolePhase's and hitCombinePhase's phases 0..6 on the copies.
    uint share = role == 0u ? 1u : (role == 1u ? 4u : role);
    uint hb = LH_BASE + i * LH;
    float ia = 0.0;
    float ib = 0.0;
    float ig = 0.0;
    float3 if1 = float3(0.0, 0.0, 0.0);
    if (item)
    {
        ia = asfloat(gsTested[hb]);
        ib = asfloat(gsTested[hb + 1u]);
        ig = nf(nf(1.0 - ia) - ib);
        if1 = asfloat(uint3(gsTested[hb + 8u], gsTested[hb + 9u], gsTested[hb + 10u]));
        localAdd(slot * 3u, toFixed3(hitShare(if1, share, ia, ib, ig)));
        if (gsPairs[q * 4u + 3u] != 0u)
            shiftAdd(x, asfloat(uint3(gsPairs[q * 4u], gsPairs[q * 4u + 1u], gsPairs[q * 4u + 2u])));
    }
    GroupMemoryBarrierWithGroupSync();
    if (item)
    {
        float3 acc = localAccNow(q, slot, false);
        gsPairs[q * 4u] = asuint(acc.x);
        gsPairs[q * 4u + 1u] = asuint(acc.y);
        gsPairs[q * 4u + 2u] = asuint(acc.z);
        gsPairs[q * 4u + 3u] = gsTested[q * LQ + 6u];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lead)
    {
        // The roles' records: the contacter, n0, n1, n2.
        float3 ah = asfloat(uint3(gsPairs[q * 4u], gsPairs[q * 4u + 1u], gsPairs[q * 4u + 2u]));
        float imh = asfloat(gsPairs[q * 4u + 3u]);
        float3 r0 = asfloat(uint3(gsPairs[q * 4u + 4u], gsPairs[q * 4u + 5u], gsPairs[q * 4u + 6u]));
        float im0 = asfloat(gsPairs[q * 4u + 7u]);
        float3 r1 = asfloat(uint3(gsPairs[q * 4u + 8u], gsPairs[q * 4u + 9u], gsPairs[q * 4u + 10u]));
        float im1 = asfloat(gsPairs[q * 4u + 11u]);
        float3 r2 = asfloat(uint3(gsPairs[q * 4u + 12u], gsPairs[q * 4u + 13u], gsPairs[q * 4u + 14u]));
        float im2 = asfloat(gsPairs[q * 4u + 15u]);
        float3 aw = sscale(r1, alpha);
        aw = sadd3(aw, sscale(r2, beta));
        aw = sadd3(aw, sscale(r0, gamma));
        float imt = nf(nf(selfTerm(alpha, im1) + selfTerm(beta, im2)) + selfTerm(gamma, im0));
        float a = sdot(ssub3(ah, aw), nn);
        float vn = sdot(vv, nn);
        float xx = nf(vn + max(vn, 0.0));
        float r = 0.0;
        if (vn < 0.0 && nf(smul(a, STEP) + xx) < 0.0)
        {
            float inv = nf(imh + imt);
            if (inv > 0.0)
            {
                float m = sdiv(1.0, inv);
                float rr = nf(-smul(a, m) - sdiv(smul(xx, m), STEP));
                r = rr > 0.0 ? rr : 0.0;
            }
        }
        gNext[at + 21u] = asuint(r);
        gsTested[hb + 11u] = asuint(r);
    }
    GroupMemoryBarrierWithGroupSync();
    if (item)
    {
        float r = asfloat(gsTested[hb + 11u]);
        if (r > 0.0)
        {
            float3 inn = asfloat(uint3(gsTested[hb + 2u], gsTested[hb + 3u], gsTested[hb + 4u]));
            float3 s = hitShare(sscale(inn, r), share, ia, ib, ig);
            localAdd(LB_BASE + slot * 3u, int3(toFixed(s.x), toFixed(s.y), toFixed(s.z)));
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint rec = hitRoleAt(v, i) + role * 4u;
    if (item)
    {
        float3 acc = localAccNow(q, slot, true);
        float im = localOwnInvMass(q);
        gsPairs[q * 4u] = asuint(acc.x);
        gsPairs[q * 4u + 1u] = asuint(acc.y);
        gsPairs[q * 4u + 2u] = asuint(acc.z);
        gsPairs[q * 4u + 3u] = asuint(im);
        gNext[rec] = asuint(acc.x);
        gNext[rec + 1u] = asuint(acc.y);
        gNext[rec + 2u] = asuint(acc.z);
        gNext[rec + 3u] = asuint(im);
    }
    GroupMemoryBarrierWithGroupSync();
    if (lead)
    {
        float3 ah = asfloat(uint3(gsPairs[q * 4u], gsPairs[q * 4u + 1u], gsPairs[q * 4u + 2u]));
        float imh = asfloat(gsPairs[q * 4u + 3u]);
        float3 r0 = asfloat(uint3(gsPairs[q * 4u + 4u], gsPairs[q * 4u + 5u], gsPairs[q * 4u + 6u]));
        float im0 = asfloat(gsPairs[q * 4u + 7u]);
        float3 r1 = asfloat(uint3(gsPairs[q * 4u + 8u], gsPairs[q * 4u + 9u], gsPairs[q * 4u + 10u]));
        float im1 = asfloat(gsPairs[q * 4u + 11u]);
        float3 r2 = asfloat(uint3(gsPairs[q * 4u + 12u], gsPairs[q * 4u + 13u], gsPairs[q * 4u + 14u]));
        float im2 = asfloat(gsPairs[q * 4u + 15u]);
        float3 aw = sscale(r1, alpha);
        aw = sadd3(aw, sscale(r2, beta));
        aw = sadd3(aw, sscale(r0, gamma));
        float imt = nf(nf(selfTerm(alpha, im1) + selfTerm(beta, im2)) + selfTerm(gamma, im0));
        float3 f = f1;
        float r = asfloat(gsTested[hb + 11u]);
        if (r > 0.0) f = sadd3(f, sscale(nn, r));
        if (!all(f == float3(0.0, 0.0, 0.0)))
        {
            float3 a = ssub3(ah, aw);
            float inv = nf(imh + imt);
            float3 rest = ssub3(a, sscale(f, inv));
            float aa = sdot(f, sadd3(sadd3(vv, vv), sscale(rest, STEP)));
            float bb = smul(smul(STEP, sdot(f, f)), inv);
            if (nf(aa + bb) > 0.0 && bb > 0.0)
            {
                float k = aa < 0.0 ? min(sdiv(-aa, bb), 1.0) : 0.0;
                f = sadd3(f, sscale(f, -nf(1.0 - k)));
            }
        }
        uint tr = hitRoleAt(v, i) + 16u;
        gNext[tr] = asuint(f.x);
        gNext[tr + 1u] = asuint(f.y);
        gNext[tr + 2u] = asuint(f.z);
        gsPairs[LT_BASE + i * 3u] = asuint(f.x);
        gsPairs[LT_BASE + i * 3u + 1u] = asuint(f.y);
        gsPairs[LT_BASE + i * 3u + 2u] = asuint(f.z);
    }
    GroupMemoryBarrierWithGroupSync();
    if (item)
    {
        float3 total = asfloat(uint3(gsPairs[LT_BASE + i * 3u], gsPairs[LT_BASE + i * 3u + 1u], gsPairs[LT_BASE + i * 3u + 2u]));
        float3 st = hitShare(total, share, ia, ib, ig);
        int3 d = toFixed3(st) - toFixed3(hitShare(if1, share, ia, ib, ig));
        localAdd(slot * 3u, d);
        cfAdd(x, st);
    }
    GroupMemoryBarrierWithGroupSync();
    // Write back: each slot's A as the device's plus its adds; B cleared (phase 6's
    // stores).
    if (item && slot == q)
    {
        int3 a = int3(asint(gsCount[q * 3u].load()), asint(gsCount[q * 3u + 1u].load()), asint(gsCount[q * 3u + 2u].load()));
        int3 d = a - a0;
        if (d.x != 0) gAcc[x * 3u].add(d.x);
        if (d.y != 0) gAcc[x * 3u + 1u].add(d.y);
        if (d.z != 0) gAcc[x * 3u + 2u].add(d.z);
        uint b = resAt(x);
        gAcc[b + 1u].store(0);
        gAcc[b + 2u].store(0);
        gAcc[b + 3u].store(0);
    }
}

// After the triangles: a workgroup per vehicle resolves its deferred breaks and
// applies its self-collision hits in order (selfApply).
[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_hits(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    uint lane = lid.x;
    // Vehicle sleep: a sleeping vehicle is not stepped.
    if (gP.sleepRec != 0u)
    {
        if (sleepSkip(v, lane)) return ebLoopCapFlush();
    }
    Vehicle x = vehicleAt(v);
    uint parity = vehicleStep(v) & 1u;
    if (lane == 0u)
    {
        driveBegin();
        resolvePending(v, parity);
        uint hits = uint(gAcc[hitAcc(v)].load());
        gsUniformHits = hits > 0u ? 1u : 0u;
        // PERF_HITS_LOCAL: 2, the hits from workgroup memory (selfApplyLocal).
        if ((gP.perfFlags & PERF_HITS_LOCAL) != 0u && hits > 0u && hits <= HITS_LOCAL_CAP && hitRolesOn(v))
            gsUniformHits = 2u;
        gsUniformOpen = 0u;
        // The reused pair list (PERF_WIDE_LIST): a walk's list of this step recorded in the
        // list's saved state as pairReuseTests records a build's (the self split's
        // selfSplitWalkEnd, on the words soloStepSave left: the next step loads them).
        if ((gP.perfFlags & PERF_WIDE_LIST) != 0u)
        {
            uint st = gP.gridCountBase + v * GRID_BUCKETS;
            if (gAcc[st + REUSE_WORDS + 2u].load() != 0)
            {
                int listed = gAcc[st + REUSE_WORDS + 1u].load();
                gAcc[st + RU_VALID].store(uint(listed) <= TRI_PAIR_CAP ? 1 : 0);
                gAcc[st + RU_COUNT].store(listed);
                gAcc[st + RU_STATS].add(1);
                gAcc[st + RU_STATS + 2u].add(listed);
                gAcc[st + RU_STATS + 3u].max(listed);
            }
        }
    }
    // PERF_HIT_SLOT_MAP: the vehicle's node map cleared (selfApplyLocal's slots; nothing
    // here reads gsCount before it), before the barrier below, on a step with hits only
    // (each lane reads the count lane 0 decides from; a step without hits clears nothing).
    uint mapNodes = hitSlotMapNodes(v);
    uint mapClear = mapNodes > 0u && gAcc[hitAcc(v)].load() > 0 ? mapNodes : 0u;
    for (uint k = lane, kStop = EB_END(100, k, mapClear, 256u); k < kStop; k += 256u) gsCount[k].store(0u);
    sync();
    uint on = gsUniformHits;
    uint driveOn = gsUniformDrive;
    // PERF_TYRE_EARLY: the next step's tyres' second part loaded before the hits.
    bool tyreEarly = (gP.perfFlags & (PERF_TYRE_EARLY | PERF_TYRE_AHEAD | PERF_AHEAD_APART | PERF_WIDE_FUSED))
                         == (PERF_TYRE_EARLY | PERF_TYRE_AHEAD | PERF_AHEAD_APART | PERF_WIDE_FUSED)
                     && wideStep() + 1u < gP.steps && driveOn != 0u;
    DrvPostIn tyreIn = (DrvPostIn)0;
    if (tyreEarly) tyreIn = drvTyreLoad(v, 255u - lane, vehicleStep(v) + 1u);
    if (on == 2u)
        selfApplyLocal(v, lane, x.beamOff, x.groupOff, x.groupCount, parity, mapNodes);
    else if (on != 0u)
        selfApply(v, true, lane, 256u, lane, vehicleOrigin(v), x.range, x.beamOff, x.groupOff, x.groupCount, parity);
    // The next step's tyres (PERF_WIDE_FUSED), after this step's hits: their hubs'
    // accumulators are final (its rays and bars ran beside this step's list phases).
    if ((gP.perfFlags & PERF_WIDE_FUSED) != 0u && wideStep() + 1u < gP.steps && driveOn != 0u)
    {
        if (on != 0u) sync();
        // (PERF_TYRE_AHEAD: their first part ran on the list dispatch's coupling and
        // wheels workgroup; the rest here.)
        if (tyreEarly)
        {
            if (tyreIn.ok) drvTyreFinish(tyreIn);
            else driveTyresPost(v, 255u - lane, vehicleStep(v) + 1u);
            return ebLoopCapFlush();
        }
        if ((gP.perfFlags & (PERF_TYRE_AHEAD | PERF_AHEAD_APART)) == (PERF_TYRE_AHEAD | PERF_AHEAD_APART))
        {
            driveTyresPost(v, 255u - lane, vehicleStep(v) + 1u);
            return ebLoopCapFlush();
        }
        DrvCtx drvC;
        bool drvHas;
        bool drvApart;
        driveAheadCtx(v, 255u - lane, drvC, drvHas, drvApart);
        driveAheadPart(drvC, drvHas, drvApart, 255u - lane, 2u, vehicleStep(v) + 1u);
    }
    ebLoopCapFlush();
}

// After the last step: its triangles' throttle and its group breaks.
[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_end(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (gP.steps == 0u) return;
    if (i < gP.totalTris)
    {
        uint v = gStatic[gP.triBase + i].w;
        if (gStatic[v * 4u + 2u].y > 0u && !sleepAsleep(v)) triangleFinish(i);
        return;
    }
    i -= gP.totalTris;
    if (i < gP.totalBeams)
    {
        uint v = gStatic[gP.beamBase + i * 3u].z >> 16;
        Vehicle x = vehicleAt(v);
        if (x.groupCount > 0u && !sleepAsleep(v)) groupDrops(i, x.beamOff, x.groupOff, (vehicleStep(v) - 1u) & 1u, true);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_close(uint3 id : SV_DispatchThreadID)
{
    uint v = id.x;
    if (v >= gP.vehicles) return ebLoopCapFlush();
    // Vehicle sleep: a sleeping vehicle's step count stands; an awake one's frame-end rule
    // (the frame's sleepSettle, on this thread).
    if (gP.sleepRec != 0u)
    {
        if (sleepAsleep(v)) return ebLoopCapFlush();
        sleepSettleOne(v, gP.steps);
    }
    uint4 m = gMisc[v];
    gMisc[v] = uint4(m.xyz, m.w + wideStep());
    ebLoopCapFlush();
}

// The rigid coupling's and the raycast wheels' part of a step in the wide passes: a
// workgroup a vehicle before the step's wide_nodes, coupleStep then driveStep as the
// frame runs them (their phases are a few lanes' short chains: cheap inside a workgroup);
// after the last step (the parameter copy `steps`) the coupling's end. No sphere or
// wide_nodes reads the spheres and lists from the buffers (workgroup copies do not reach it).
[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_wide_aux(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    uint lane = lid.x;
    uint s = wideStep();
    // Vehicle sleep: a sleeping vehicle is not stepped.
    if (gP.sleepRec != 0u)
    {
        if (sleepSkip(v, lane)) return ebLoopCapFlush();
    }
    if (lane == 0u)
    {
        coupleBegin();
        driveBegin();
        // (The spheres' workgroup copies serve this dispatch's own phases, as the frame's.)
        cplGsClear();
        cplGsAdd(0u, v, 0u, 0u);
        if (s == 0u) cplListBegin(v);
        cplFoldMark(v);
    }
    sync();
    uint coupleOn = gsUniformCouple;
    uint driveOn = gsUniformDrive;
    if (s < gP.steps)
    {
        coupleStep(v, lane, s, coupleOn);
        driveStep(v, lane, s, driveOn);
    }
    else
    {
        coupleEnd(v, lane, coupleOn);
    }
    ebLoopCapFlush();
}

#ifdef EB_SELF_SPLIT
#include "explicit_beams_selfsplit_frame.slang"
#endif

