// --- A small group's members stepped side by side (k <= VV_PART members) ---
// The workgroup's lanes split into k partitions of P lanes (a power of two dividing
// GRID_BUCKETS); partition m steps member m exactly as contactMemberStep would,
// with its own shared counters and its self-collision tested list, grid counts and
// offsets in the per-vehicle global regions the wide passes use. Members do not
// read each other's state within their steps (contact reaches them only through
// the accumulators of the contact pass), so every member's step is the same, bit
// for bit, while the barriers are shared: k members cost about one member's step.
// (VV_PART: with VV_MAX_GROUP, above.)
// (The members' step's own: at most VV_SIDE members side by side. A browser's workgroup
// storage is 32,768 bytes a module: sized by VV_PART these passed it.)
groupshared Atomic<uint> gsTestedM[2u * VV_SIDE];
groupshared Atomic<uint> gsPairCountM[VV_PART];
groupshared uint gsBlkM[GRID_BLOCKS * VV_SIDE];
static const uint PAIR_PART = PAIR_LIST / VV_SIDE;

// Lanes per member for k members: LANES >> ceil(log2 k).
uint vvPartLanes(uint k)
{
    uint p = LANES;
    uint pGuard = 0u;
    while (p * k > LANES && p > 1u)
    {
        EB_PASS(155, pGuard);
        p = p >> 1u;
    }
    return p;
}

// triangleGrid with the offsets in gNext at offBase (a member's grid).
void triangleGridAt(uint gt, uint sub, float3 origin, float r, float cell, uint candOff, uint candCount,
                    uint beamOff, uint groupOff, uint groupCount, uint parity, uint offBase)
{
    triangleGridWalk(gt, sub, origin, r, cell, candOff, candCount, beamOff, groupOff, groupCount, parity, true, offBase, 0u);
}

// trianglePairs into member m's slice of the pair list.
void trianglePairsM(uint m, uint t, uint gt, uint sub, uint subs, float3 origin, float r, float cell, uint candOff,
                    uint candCount, uint parity)
{
    if (asint(gMisc[gP.vehicles + gt].x) > 0) return;
    uint4 tn = gStatic[gP.triBase + gt];
    float3 p0 = nf3(origin + gNodes[tn.x * 3u].xyz);
    float3 p1 = nf3(origin + gNodes[tn.y * 3u].xyz);
    float3 p2 = nf3(origin + gNodes[tn.z * 3u].xyz);
    // The box tryContact tests (triangleGeo's: with anti-clip, its reach).
    float reach = triReach(tn, r, parity);
    float3 lo = min(min(p0, p1), p2) - float3(reach, reach, reach);
    float3 hi = max(max(p0, p1), p2) + float3(reach, reach, reach);
    bool every = true;
    int3 c0 = int3(0, 0, 0);
    int3 span = int3(0, 0, 0);
    if (cell > 0.0)
    {
        float3 f0 = floor((lo - origin) / cell);
        float3 fs = floor((hi - origin) / cell) - f0 + float3(1.0, 1.0, 1.0);
        every = !(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS));
        if (!every)
        {
            c0 = int3(f0);
            span = int3(fs);
        }
    }
    uint yz = uint(span.y * span.z);
    uint cells = uint(span.x) * yz;
    for (uint k = sub, kStop = EB_END(156, k, candCount, subs); k < kStop; k += subs)
    {
        uint h = gLists[candOff + k];
        if (h == tn.x || h == tn.y || h == tn.z) continue;
        float3 e = gNodes[h * 3u].xyz;
        if (!every)
        {
            float3 p = origin + e;
            if (any(p < lo) || any(p > hi)) continue;
            int3 c = int3(floor(e / cell));
            if (!(all(c >= c0) && all(c < c0 + span)))
            {
                uint b = gridBucket(c);
                bool covered = false;
                for (uint i = 0u, iStop = EB_END(157, i, cells, 1u); i < iStop; i++)
                    covered = covered
                        || gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z)))) == b;
                if (!covered) continue;
            }
        }
        float3 x = nf3(origin + e);
        if (any(x < lo) || any(x > hi)) continue;
        uint i = gsPairCountM[m].add(1u);
        if (i < PAIR_PART) gsPairs[m * PAIR_PART + i] = (t << 16) | k;
    }
}

// `only` (contact-groups-design.md, the split group path): the one member this workgroup
// steps (its partition's lanes are its own; the other partitions idle), or >= k for
// every member (the group's own frame).
void contactMembersStepAt(uint off, uint k, uint lane, uint s, uint only);
void contactMembersStep(uint off, uint k, uint lane, uint s)
{
    contactMembersStepAt(off, k, lane, s, VV_PART);
}

// (gVvWhole, the split path's members, a workgroup each: member `only` on every lane of
// the workgroup, its workgroup arrays in slot 0, in place of its partition's lanes. The
// same step: the lanes only share its items out.)
static bool gVvWhole = false;
void contactMembersStepAt(uint off, uint k, uint lane, uint s, uint only)
{
    uint P = gVvWhole ? LANES : vvPartLanes(k);
    uint mm = gVvWhole ? only : lane / P;
    uint m = gVvWhole ? 0u : mm;
    uint pl = lane % P;
    bool on = mm < k && (only >= k || mm == only);
    uint v = vvMember(off, on ? mm : 0u);
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 r3 = gStatic[v * 4u + 3u];
    uint nodeOff = r0.x;
    uint nodeCount = on ? r0.y : 0u;
    uint beamOff = r0.z;
    uint beamCount = on ? r0.w : 0u;
    uint triOff = r1.x;
    uint triCount = on ? r1.y : 0u;
    uint boxOff = r1.z;
    uint boxCount = r1.w;
    uint candOff = r2.x;
    uint candCount = on ? r2.y : 0u;
    uint groupOff = r2.z;
    uint groupCount = on ? r2.w : 0u;
    uint flagsV = r3.x;
    float ground = asfloat(r3.y);
    float range = selfRangeOf(v, asfloat(r3.z));
    float cell = asfloat(r3.w);
    bool fast = (flagsV & FLAG_FAST_INV_SQRT) != 0u;
#ifdef EB_AERO
    uint4 aero0 = gStatic[gP.aeroBase + v * 3u];
    uint4 aero1 = gStatic[gP.aeroBase + v * 3u + 1u];
    uint4 aero2 = gStatic[gP.aeroBase + v * 3u + 2u];
    if (!on)
    {
        aero0 = uint4(0u, 0u, 0u, 0u);
        flagsV = flagsV & ~(FLAG_FUSEDRAG | FLAG_SLIDES | FLAG_KEYS);
    }
#endif
    uint4 misc = gMisc[v];
    float3 origin = asfloat(misc.xyz);
    uint step = misc.w + s;
    uint parity = step & 1u;
    bool pending = gMisc[vvRec(v, VV_FLAGS)].w != 0u;
    // The member's grid: counts (gAcc), offsets (gNext), as the wide passes keep them.
    uint countBase = gP.gridCountBase + v * GRID_BUCKETS;
    uint offBase = gP.gridOffBase + v * (GRID_BUCKETS + 1u);
    uint listBase = gP.testedListBase + triOff;
    if (lane == 0u)
    {
#ifndef EB_SOLO_STEP
        // (The split path's members' module: the groups' pass clears them.)
        vvClearCounters();
        vvClearSmall();
#endif
        gsUniformHits = 0u;
        gsUniformOpen = 0u;
    }
    if (on && pl == 0u)
    {
        gsTestedM[2u * m + parity].store(0u);
        gsPairCountM[m].store(0u);
        gAcc[hitAcc(v)].store(0);
        gAcc[hitAcc(v) + 1u].store(0);
        clipVmaxClear(v, parity);
    }
    sync();
    {
        bool centredV = gP.centredDv != 0u && (flagsV & FLAG_CENTRED) != 0u;
        for (uint i = pl, iStop = EB_END(158, i, nodeCount, P); i < iStop; i += P)
        {
            if (centredV && (flagsV & FLAG_CENTRED_OBJECTIVE) != 0u) centredKeep(nodeOff + i);
            float3 v0 = centredV ? gNodes[(nodeOff + i) * 3u + 1u].xyz : float3(0.0, 0.0, 0.0);
            nodeUpdate(nodeOff + i, gStatic[gP.nodeInfoBase + nodeOff + i], v, i, step, origin, flagsV, ground,
                       boxOff, boxCount, beamOff, groupOff, groupCount, true);
            if (centredV) centredDelta(nodeOff + i, v0);
        }
        if (on && pl < 6u) gAcc[vvBoxKey(v, parity ^ 1u) + pl].store(pl < 3u ? VV_INT_MAX : VV_INT_MIN);
        memberBox(v, nodeOff, nodeCount, origin, parity, pl, P);
    }
    if (gP.centredDv != 0u)
    {
        centredPhase(flagsV, nodeOff, nodeCount, beamOff, beamCount, groupOff, groupCount, step, pl, P);
        if (on && (flagsV & FLAG_CENTRED) != 0u) memberBox(v, nodeOff, nodeCount, origin, parity, pl, P);
    }
    for (uint g = pl, gStop = EB_END(159, g, groupCount, P); g < gStop; g += P) gGroups[groupSlot(parity, groupOff, g)].store(NO_GROUP);
    if (candCount > 0u)
    {
        for (uint t = pl, tStop = EB_END(160, t, triCount, P); t < tStop; t += P)
        {
            int w = s > 0u ? triangleFinish(triOff + t) : asint(gMisc[gP.vehicles + triOff + t].x);
            if (s > 0u && pending)
            {
                w = 0;
                gMisc[gP.vehicles + triOff + t].x = 0u;
            }
            if (w <= 0) gNext[listBase + gsTestedM[2u * m + parity].add(1u)] = t;
        }
    }
    if (on && cell > 0.0)
    {
        for (uint b = pl; b < GRID_BUCKETS; b += P) gAcc[countBase + b].store(0);
    }
    sync();
    if (on && pl == 0u) grabApply(nodeOff, nodeCount, origin);
#ifdef EB_AERO
    if (on && pl == 0u) brakeApply(aero2.z, aero2.w);
#endif
    if (on && pl == 0u && s > 0u) gMisc[vvRec(v, VV_FLAGS)].w = 0u;
    // This step's self-collision per member: the grid or (few pairs) the early list.
    uint listed = on ? gsTestedM[2u * m + parity].load() : 0u;
    bool gridM = cell > 0.0 && candCount > 0u && listed * candCount > BRUTE_PAIRS_PER_LANE * P;
    if (lane == 0u)
    {
        uint any = 0u;
        for (uint j = 0u, jStop = EB_END(161, j, k, 1u); j < jStop; j++)
        {
            uint vj = vvMember(off, j);
            uint4 q1 = gStatic[vj * 4u + 1u];
            uint4 q2 = gStatic[vj * 4u + 2u];
            float cj = asfloat(gStatic[vj * 4u + 3u].w);
            uint lj = (only >= k || j == only) ? gsTestedM[2u * (gVvWhole ? 0u : j) + (((gMisc[vj].w + s) & 1u))].load() : 0u;
            if (cj > 0.0 && q2.y > 0u && lj * q2.y > BRUTE_PAIRS_PER_LANE * P) any = any | 1u;
            // Self-collision hits or pending beams possible (selfApply's barriers).
            if (q1.y > 0u) any = any | 2u;
        }
        gsUniformLanes = any;
    }
    if (gridM)
    {
        for (uint c = pl, cStop = EB_END(162, c, candCount, P); c < cStop; c += P)
        {
            float3 p = gNodes[gLists[candOff + c] * 3u].xyz;
            uint b = gridBucket(int3(floor(p / cell)));
            gNext[candOff + c] = b | (uint(gAcc[countBase + b].add(1)) << 10);
        }
    }
#ifdef EB_AERO
    for (uint w = pl, wStop = EB_END(163, w, aero0.y, P); w < wStop; w += P) wingForces(aero0.x + w, origin);
    if (on && pl == P - 1u && (flagsV & FLAG_FUSEDRAG) != 0u) fuseDrag(aero1, origin, v);
    if ((flagsV & FLAG_KEYS) != 0u)
    {
        for (uint c = pl, cStop = EB_END(164, c, aero0.w, P); c < cStop; c += P) commandUpdateKeys(aero0.z + c, v, parity ^ 1u);
    }
    else
    {
        for (uint c = pl, cStop = EB_END(165, c, aero0.w, P); c < cStop; c += P) commandUpdate(aero0.z + c);
    }
    sync();
    {
        uint4 ext = gStatic[gP.vehExtBase + v];
        if (!on) ext.w = 0u;
        for (uint q = pl, qStop = EB_END(166, q, ext.w, P); q < qStop; q += P) wingGather(ext.z + q);
    }
    if ((flagsV & FLAG_KEYS) != 0u)
    {
        for (uint q = pl; q < KEYS; q += P) gNext[keySlot(v, parity ^ 1u, q)] = 0u;
    }
#endif
    for (uint j = pl, jStop = EB_END(167, j, beamCount, P); j < jStop; j += P) beamUpdate(beamOff + j, j, fast, groupOff, parity);
    bool early = candCount > 0u && !gridM;
    if (early)
    {
        uint lanes = clamp(P / max(listed, 1u), 1u, candCount);
        uint stride = (P / lanes) * lanes;
        uint rl = P - 1u - pl;
        for (uint q = rl, qStop = EB_END(168, q, listed * lanes, stride); q < qStop && rl < stride; q += stride)
        {
            uint t = gNext[listBase + q / lanes];
            trianglePairsM(m, t, triOff + t, q % lanes, lanes, origin, range, cell, candOff, candCount, parity);
        }
    }
    sync();
    uint flagsU = gsUniformLanes;
    uint gridAny = flagsU & 1u;
    if (on && pl == 0u) resolvePending(v, parity);
    if (gridAny != 0u)
    {
        uint per = GRID_BUCKETS / P;
        uint block = P / GRID_BLOCKS;
        uint sum = 0u;
        if (gridM)
        {
            for (uint q = 0u, qStop = EB_END(169, q, per, 1u); q < qStop; q++) sum += uint(gAcc[countBase + pl * per + q].load());
        }
        gsPart[lane] = sum;
        sync();
        if (gridM && pl < GRID_BLOCKS)
        {
            uint run = 0u;
            for (uint q = 0u, qStop = EB_END(170, q, block, 1u); q < qStop; q++)
            {
                uint x = gsPart[m * P + pl * block + q];
                gsPart[m * P + pl * block + q] = run;
                run += x;
            }
            gsBlkM[m * GRID_BLOCKS + pl] = run;
        }
        sync();
        if (gridM && pl == 0u)
        {
            uint run = 0u;
            for (uint q = 0u; q < GRID_BLOCKS; q++)
            {
                uint x = gsBlkM[m * GRID_BLOCKS + q];
                gsBlkM[m * GRID_BLOCKS + q] = run;
                run += x;
            }
            gNext[offBase + GRID_BUCKETS] = run;
        }
        sync();
        if (gridM)
        {
            uint run = gsPart[lane] + gsBlkM[m * GRID_BLOCKS + pl / block];
            for (uint q = 0u, qStop = EB_END(171, q, per, 1u); q < qStop; q++)
            {
                gNext[offBase + pl * per + q] = run;
                run += uint(gAcc[countBase + pl * per + q].load());
            }
        }
        sync();
        if (gridM)
        {
            for (uint c = pl, cStop = EB_END(172, c, candCount, P); c < cStop; c += P)
            {
                uint br = gNext[candOff + c];
                uint h = gLists[candOff + c];
                gBeams[gP.gridBase + candOff + gNext[offBase + (br & (GRID_BUCKETS - 1u))] + (br >> 10)] =
                    float4(gNodes[h * 3u].xyz, asfloat(h));
            }
        }
        sync();
    }
    uint pairs = on ? gsPairCountM[m].load() : 0u;
#ifdef EB_ABLATE
    // Measurement only: 1048576 the members' self-collision tests.
    if ((gP.ablate & 1048576u) != 0u) {} else
#endif
    {
        // One of four tests from one loop of triangleCandidatesFrom (one copy of tryContact):
        // cmode 5 the early pairs (a pair: its triangle against its one candidate, mode 0),
        // 3 the grid's walk at the member's offsets, 1 the brute pairs, 0 every candidate.
        uint cmode = 4u, cN = 0u, ck = 1u, cstride = P;
        if (early && pairs <= PAIR_PART)
        {
            cN = pairs;
            cmode = 5u;
        }
        else if (candCount > 0u && cell > 0.0 && gridM)
        {
            cN = listed * GRID_SUBS;
            ck = GRID_SUBS;
            cmode = 3u;
        }
        else if (candCount > 0u)
        {
            ck = clamp(P / max(listed, 1u), 1u, candCount);
            cstride = (P / ck) * ck;
            cN = listed * ck;
            cmode = cell > 0.0 ? 1u : 0u;
        }
        for (uint q = pl, qStop = EB_END(174, q, cN, cstride); q < qStop && pl < cstride; q += cstride)
        {
            uint gt = triOff;
            uint sub = q % ck;
            uint subs = ck;
            if (cmode == 5u)
            {
                uint e = gsPairs[m * PAIR_PART + q];
                gt += e >> 16;
                sub = e & 0xFFFFu;
                subs = candCount;
            }
            else
                gt += gNext[listBase + q / ck];
            triangleCandidatesFrom(gt, sub, subs, origin, range, cell, candOff, candCount, beamOff, groupOff, groupCount,
                                   parity, cmode == 5u ? 0u : cmode, cmode == 3u ? offBase : TC_GS);
        }
    }
    if ((flagsU & 2u) != 0u)
    {
        sync();
        selfApply(v, on, pl, P, lane, origin, range, beamOff, groupOff, groupCount, parity);
    }
#ifdef EB_AERO
    if (on && pl == P - 1u && (flagsV & FLAG_SLIDES) != 0u) slidePass(aero2.x, aero2.y, SLIDE_FRAME, groupOff, parity);
#endif
    sync();
}

// A member's frame start: the wing check and the origin rebase.
void contactMemberBegin(uint v, uint lane)
{
    uint4 r0 = gStatic[v * 4u];
#ifdef EB_AERO
    uint4 aero0 = gStatic[gP.aeroBase + v * 3u];
    {
        float3 o = asfloat(gMisc[v].xyz);
        for (uint w = lane, wStop = EB_END(177, w, aero0.y, LANES); w < wStop; w += LANES) wingCheck(aero0.x + w, o);
    }
#endif
    float3 first = gNodes[r0.x * 3u].xyz;
    sync();
    if (length(first) > REBASE_RADIUS)
    {
        for (uint i = lane, iStop = EB_END(178, i, r0.y, LANES); i < iStop; i += LANES)
        {
            float4 P = gNodes[(r0.x + i) * 3u];
            gNodes[(r0.x + i) * 3u] = float4(P.xyz - first, P.w);
        }
        if (lane == 0u)
        {
            uint4 m = gMisc[v];
            gMisc[v] = uint4(asuint(asfloat(m.xyz) + first), m.w);
        }
        cplRebase(v, lane, first);
    }
    sync();
}

// A member's frame end: the last step's throttle update (and a pending reset), the
// last step's group breaks, the steps run; its box keys cleared for a later frame.
void contactMemberEnd(uint v, uint lane)
{
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 misc = gMisc[v];
    bool pending = gMisc[vvRec(v, VV_FLAGS)].w != 0u;
    if (gP.steps > 0u && r2.y > 0u)
    {
        for (uint t = lane, tStop = EB_END(179, t, r1.y, LANES); t < tStop; t += LANES)
        {
            triangleFinish(r1.x + t);
            if (pending) gMisc[gP.vehicles + r1.x + t].x = 0u;
        }
    }
    if (gP.steps > 0u && r2.w > 0u)
    {
        uint last = (misc.w + gP.steps - 1u) & 1u;
        for (uint j = lane, jStop = EB_END(180, j, r0.w, LANES); j < jStop; j += LANES) groupDrops(r0.z + j, r0.z, r2.z, last, true);
    }
    if (lane < 12u) gAcc[gP.vvAcc + v * 12u + lane].store((lane % 6u) < 3u ? VV_INT_MAX : VV_INT_MIN);
    sync();
    if (lane == 0u)
    {
        gMisc[v] = uint4(misc.xyz, misc.w + gP.steps);
        gMisc[vvRec(v, VV_FLAGS)].w = 0u;
    }
}

// A collision triangle as the contact pass sees it (spec §5).
struct VvTri
{
    uint4 tn;
    float3 p0;
    float3 n;
    float3 ra;
    float3 rb;
    float3 rn;
    bool solvable;
    float3 lo;
    float3 hi;
    // Anti-clip (§36): bit 0 the owner has it, bit 1 the triangle is anti-clip
    // only; `travel` its fastest node's travel over the step (with any anti-clip).
    uint clip;
    float travel;
};

// A contact triangle's frame from its world positions (ContactFrame::new).
void vvFrame(inout VvTri t, float3 p1, float3 p2)
{
    float3 u = ssub3(p1, t.p0);
    float3 w = ssub3(p2, t.p0);
    float3 cr = scross(u, w);
    float cl = ssqrt(sdot(cr, cr));
    t.n = cl > 1e-8 ? sscale(cr, sdiv(1.0, cl)) : cr;
    float det = sdot(u, scross(w, t.n));
    t.solvable = abs(det) > 1e-6;
    float rd = t.solvable ? sdiv(1.0, det) : 0.0;
    t.ra = sscale(scross(w, t.n), rd);
    t.rb = sscale(scross(t.n, u), rd);
    t.rn = sscale(scross(u, w), rd);
}

// Vehicle contact in the owner's frame: vehicle v's origin relative to the owner a's
// (exactly 0 for v = a, and for every vehicle while the origins are 0). The contact's
// geometry (triangles, contact points, votes, anti-clip) is in the owner's frame, its
// selection (frame bounds, boxes, prefilters) in the World's.
// (nf3: a compiler allowed to reassociate, as Metal's is, would otherwise turn
// `vvRel(v, a) + x` into another order in one inlined copy and not in another, and one
// contact point would have two positions: the gather's and the kept pairs' tests of
// vehicle contact differed in the last bit; contact-groups-design.md §8.10.)
float3 vvRel(uint v, uint a) { return nf3(asfloat(gMisc[v].xyz) - asfloat(gMisc[a].xyz)); }

// --- Contact offsets (ExplicitBeams::contact_offset; explicit_plan's W_OFFSET) ---
// gStatic at gP.offsetBase per global node (its outer triangles' first entry in gLists,
// their count, its offset's bits, 0), then per vehicle (its largest offset's bits, ...);
// at gP.offsetBase - 1 (the vectors' first float4 in gBeams). Each node's vector is its
// offset along its vertex normal (the unit sum of its live outer triangles' unit
// normals, explicit_contact/edge.rs contact_offset_vectors), computed at the contact pass's
// start (contactOffsets) and added to a vehicle's triangles (vvTriangle, vvNearBox).
// (EB_NO_OFFSETS: a module that runs only with the fleet-wide schedules, which the host
// leaves off with contact offsets: explicit_plan's PERF_FLEET_FAMILY.)
float3 contactOff(uint n)
{
    if (gP.offsetBase == 0u) return float3(0.0, 0.0, 0.0);
    return gBeams[gStatic[gP.offsetBase - 1u].x + n].xyz;
}
float contactOffMax(uint v) { return gP.offsetBase == 0u ? 0.0 : asfloat(gStatic[gP.offsetBase + gP.totalNodes + v].x); }
// glam's normalize_or_zero.
float3 contactNormOrZero(float3 x)
{
    float rcp = sdiv(1.0, ssqrtCr(sdot(x, x)));
    if (!(rcp > 0.0 && rcp <= 3.4028235e38)) return float3(0.0, 0.0, 0.0);
    return sscale(x, rcp);
}
// Vehicle v's nodes' offset vectors (lane of lanes; the caller's barrier follows).
void contactOffsets(uint v, uint lane, uint lanes)
{
    if (gP.offsetBase == 0u || contactOffMax(v) == 0.0) return;
    uint4 r0 = gStatic[v * 4u];
    uint vecs = gStatic[gP.offsetBase - 1u].x;
    for (uint i = lane, iStop = EB_END(870, i, r0.y, lanes); i < iStop; i += lanes)
    {
        uint n = r0.x + i;
        uint4 rec = gStatic[gP.offsetBase + n];
        float o = asfloat(rec.z);
        float3 sum = float3(0.0, 0.0, 0.0);
        for (uint q = 0u, qStop = EB_END(871, q, o != 0.0 ? rec.y : 0u, 1u); q < qStop; q++)
        {
            uint gt = gLists[rec.x + q];
            if ((gMisc[gP.vehicles + gt].z & TRI_BREAKABLE) != 0u && triangleDead(gt)) continue;
            uint4 tn = gStatic[gP.triBase + gt];
            float3 a = gNodes[tn.x * 3u].xyz;
            sum = sadd3(sum, contactNormOrZero(scross(ssub3(gNodes[tn.y * 3u].xyz, a), ssub3(gNodes[tn.z * 3u].xyz, a))));
        }
        gBeams[vecs + n] = float4(o != 0.0 ? sscale(contactNormOrZero(sum), o) : float3(0.0, 0.0, 0.0), 0.0);
    }
}

VvTri vvTriangle(uint gt, float3 origin, float r)
{
    VvTri t;
    t.tn = gStatic[gP.triBase + gt];
    t.p0 = nf3(origin + gNodes[t.tn.x * 3u].xyz);
    float3 p1 = nf3(origin + gNodes[t.tn.y * 3u].xyz);
    float3 p2 = nf3(origin + gNodes[t.tn.z * 3u].xyz);
#ifndef EB_NO_OFFSETS
    if (gP.offsetBase != 0u)
    {
        t.p0 = sadd3(t.p0, contactOff(t.tn.x));
        p1 = sadd3(p1, contactOff(t.tn.y));
        p2 = sadd3(p2, contactOff(t.tn.z));
    }
#endif
    vvFrame(t, p1, p2);
    float3 r3 = float3(r, r, r);
    t.lo = ssub3(min(min(t.p0, p1), p2), r3);
    t.hi = sadd3(max(max(t.p0, p1), p2), r3);
    uint z = gMisc[gP.vehicles + gt].z;
    t.clip = ((gStatic[t.tn.w * 4u + 3u].x & FLAG_ANTI_CLIP) != 0u ? 1u : 0u) | ((z & TRI_ANTI_ONLY) != 0u ? 2u : 0u);
    t.travel = gP.clipMem != 0u ? smul(clipTriSpeed(t.tn), STEP) : 0.0;
    if (((z & TRI_BREAKABLE) != 0u && triangleDead(gt)) || t.clip == 2u)
    {
        t.lo = float3(3.0e38, 3.0e38, 3.0e38);
        t.hi = -t.lo;
    }
    return t;
}

// Candidate h (global node, world position p, of vehicle vh) against triangle gt of
// the owner: the hit test, the side vote, the contact force into the accumulators
// (h, and the triangle's nodes by their weights). Returns whether it hit.
// Two-body contact (spec §10.3): the pair (owner va, partner vh) opts in.
bool vvTwoBody(uint va, uint vh)
{
    return gP.vvSplit != 0u && ((gStatic[va * 4u + 3u].x | gStatic[vh * 4u + 3u].x) & FLAG_TWO_BODY) != 0u;
}
uint vvSplitTri(uint n) { return gP.vvSplit + n; }
#ifdef EB_VV_FLEET
// The owners' state by vehicle (explicit_beams_fleet.slang): gFl 1 one owner, gFlV, on
// this thread (the tests across workgroups); 2 a group's owners in its own workgroup,
// owner a its member a (the levels); 0 a group's slots, as before. Set by the caller
// around the leaf functions, whose address helpers follow it.
static uint gFl = 0u;
static uint gFlV = 0u;
static uint gFlOff = 0u;
uint flVehOf(uint a) { return gFl == 2u ? gNext[gP.vvMembers + gP.vehicles + gFlOff + a] : gFlV; }
uint flRun(uint v) { return gNext[flPrefixAt(0u, v)]; }
// Owner va's two-body slot at a point of vehicle vh: its rank among vh's two-body
// partners of the step (flOwnerPlan), found among va's pairs; the last one kept.
static uint gFlSlotVa = 0xFFFFFFFFu;
static uint gFlSlotVh = 0xFFFFFFFFu;
static uint gFlSlotVal = 0u;
uint flSlot(uint va, uint vh)
{
    if (gFlSlotVa == va && gFlSlotVh == vh) return gFlSlotVal;
    uint slot = 0u;
    uint e1 = gNext[flAdjFirst(va + 1u)];
    for (uint e = gNext[flAdjFirst(va)], eStop = EB_END(181, e, e1, 1u); e < eStop; e++)
    {
        uint d = gNext[flAdjAt(e)];
        if (flPartner(d) == vh)
        {
            slot = min(gNext[flSlotAt(d)], vvRoom() - 1u);
            break;
        }
    }
    gFlSlotVa = va;
    gFlSlotVh = vh;
    gFlSlotVal = slot;
    return slot;
}
uint vvSplitPoint(uint slot, uint n)
{
    if (gFl != 0u) slot = flSlot(flVehOf(slot), gStatic[gP.nodeInfoBase + n].w);
    return gP.vvSplit + (1u + slot) * gP.totalNodes + n;
}
// An owner's hit list's room and the point slots a full clear zeroes.
uint vvHitRoom(uint a, uint points) { return gFl != 0u ? 2u * uint(gAcc[flOwn(flVehOf(a)) + FL_OFFERED].load()) : 2u * points; }
uint vvSplitSlots(uint k) { return gFl != 0u ? vvRoom() : k; }
#else
uint vvSplitPoint(uint slot, uint n) { return gP.vvSplit + (1u + slot) * gP.totalNodes + n; }
uint vvHitRoom(uint a, uint points) { return 2u * points; }
uint vvSplitSlots(uint k) { return k; }
#endif
// The listed triangles' throttle updates already made by threads over the fleet
// (explicit_beams_fleet.slang, PERF_FLEET_POST): the levels leave them out.
#ifdef EB_VV_FLEET
bool vvPostDone() { return gFl != 0u && (gP.perfFlags & (PERF_FLEET_FRONT | PERF_FLEET_POST)) == (PERF_FLEET_FRONT | PERF_FLEET_POST); }
#else
bool vvPostDone() { return false; }
#endif
// A two-body hit found by a walk: its nodes' hit counts of this owner's turn.
void vvSplitCount(VvTri t, uint h, uint slot)
{
    uint va = gStatic[gP.nodeInfoBase + t.tn.x].w;
    uint vh = gStatic[gP.nodeInfoBase + h].w;
    if (!vvTwoBody(va, vh)) return;
    gAcc[vvSplitTri(t.tn.x)].add(1);
    gAcc[vvSplitTri(t.tn.y)].add(1);
    gAcc[vvSplitTri(t.tn.z)].add(1);
    gAcc[vvSplitPoint(slot, h)].add(1);
}
float vvSplitK(uint at) { return float(max(gAcc[at].load(), 1)); }
bool vvImmovable(uint n) { return (gStatic[gP.nodeInfoBase + n].x & NODE_IMMOVABLE) != 0u; }

// Node n's acceleration (F/m)·w, component by component (explicit_contact.rs's
// order); 0 for an immovable node.
float3 vvAccel(float3 f, uint n, float m, float w)
{
    if (vvImmovable(n)) return float3(0.0, 0.0, 0.0);
    return sscale(float3(sdiv(f.x, m), sdiv(f.y, m), sdiv(f.z, m)), w);
}

// Every member's node force at the contact pass's start (two-body contact).
void vvSnapshot(uint off, uint k, uint parity, uint lane)
{
    for (uint mi = 0u, miStop = EB_END(182, mi, k, 1u); mi < miStop; mi++)
    {
        uint v = vvMember(off, mi);
        uint4 r0 = gStatic[v * 4u];
        for (uint i = lane, iStop = EB_END(183, i, r0.y, LANES); i < iStop; i += LANES)
            gBeams[gP.vvSnap + r0.x + i] = float4(vvPointForce(r0.x + i, v, parity), 0.0);
    }
}

// Zeroes the two-body counts of the group's members' nodes (triangle counts and
// point slots 0..slots − 1), after a turn's hits are applied.
void vvSplitClear(uint off, uint k, uint slots, uint lane)
{
    if (gP.vvSplit == 0u) return;
    for (uint m = 0u, mStop = EB_END(184, m, k, 1u); m < mStop; m++)
    {
        uint4 r0 = gStatic[vvMember(off, m) * 4u];
        for (uint i = lane, iStop = EB_END(185, i, r0.y, LANES); i < iStop; i += LANES)
        {
            gAcc[vvSplitTri(r0.x + i)].store(0);
            for (uint sl = 0u, slStop = EB_END(186, sl, min(slots, EB_MAX_MEMBERS), 1u); sl < slStop; sl++)
            {
                // (The slot's own word, whatever the owners' addressing.)
                if (sl < slots) gAcc[gP.vvSplit + (1u + sl) * gP.totalNodes + r0.x + i].store(0);
            }
        }
    }
}
