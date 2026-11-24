// The contact pass after every member's step (their boxes already in the step's
// keys): each owner in vehicle order, three barriers each. Phase S: the partner
// scan over the other members (and the previous owner's throttle updates); phase
// P: contact throttle resets, the owner's throttle step listing its tested
// triangles, and its candidates compacted; phase T: tested triangles against the
// candidates.
#ifdef EB_EDGE
void edgeResetPrepare(uint k, uint off, uint lane, bool prepare);
void edgeOwnerBig(uint a, uint k, uint off, uint lane);
void edgeApplyOwner(uint va, uint slot, uint lane, uint lanes);
void edgeResidualPhase(uint va, uint slot, uint parity, uint lane, uint lanes, uint phase);
uint edgeHits(uint va);
bool edgeOn();
#endif
void vvPass(uint k, uint off, uint runOff, uint points, uint step, uint lane)
{
    uint parity = step & 1u;
    if (gP.vvSplit != 0u)
    {
        vvSnapshot(off, k, parity, lane);
        sync();
    }
#ifdef EB_EDGE
    // Edge vehicle contact (explicit_beams_vv_edges.slang): every member's edges this step.
    if (edgeOn()) edgeResetPrepare(k, off, lane, true);
#endif
    float3 m3 = float3(VV_BOX_MARGIN, VV_BOX_MARGIN, VV_BOX_MARGIN);
    uint prevTri = 0u;
    uint prevListed = 0u;
    uint prevVa = 0u;
    float prevRange = 0.0;
    for (uint a = 0u, aStop = EB_END_UNIFORM(195, a, k, 1u, EB_MAX_MEMBERS); a < aStop; a++)
    {
        uint q = a & 1u;
        uint va = vvMember(off, a);
        uint key = vvBoxKey(va, parity);
        float3 blo = float3(vvUnkey(gAcc[key].load()), vvUnkey(gAcc[key + 1u].load()), vvUnkey(gAcc[key + 2u].load()));
        float3 bhi = float3(vvUnkey(gAcc[key + 3u].load()), vvUnkey(gAcc[key + 4u].load()), vvUnkey(gAcc[key + 5u].load()));
        float3 alo = ssub3(blo, m3);
        float3 ahi = sadd3(bhi, m3);
        float3 va0 = gNodes[gStatic[va * 4u].x * 3u + 1u].xyz;
        // Phase S (with the previous owner's hits).
        for (uint i = lane, iStop = EB_END(196, i, prevListed, LANES); i < iStop; i += LANES) vvThrottleEnd(prevTri + gNext[vvTestedAt(prevTri + i)]);
#ifdef EB_ABLATE
        if ((gP.ablate & 1024u) == 0u)
#endif
        uint prevHits = a > 0u ? gsVvHits[q ^ 1u].load() : 0u;
        if (a > 0u) vvApplyHits(prevVa, runOff, points, prevHits, prevRange, parity, lane);
#ifdef EB_EDGE
        if (a > 0u && edgeOn()) edgeApplyOwner(prevVa, 0u, lane, LANES);
#endif
        // §37.1: the previous owner's residual pass and work guard (its listed hits).
        if (gP.vvSplit != 0u && a > 0u)
        {
            for (uint ph = 0u; ph < 3u; ph++)
            {
                sync();
                if (prevHits <= 2u * points)
                    vvResidualPhase(prevVa, vvHitAt(runOff, 0u), prevHits, prevRange, parity, lane, LANES, 0u, ph);
#ifdef EB_EDGE
                if (edgeOn()) edgeResidualPhase(prevVa, 0u, parity, lane, LANES, ph);
#endif
            }
        }
        for (uint b = lane, bStop = EB_END(197, b, k, LANES); b < bStop; b += LANES)
        {
            uint vb = vvMember(off, b);
            uint kb = vvBoxKey(vb, parity);
            float3 olo = ssub3(float3(vvUnkey(gAcc[kb].load()), vvUnkey(gAcc[kb + 1u].load()), vvUnkey(gAcc[kb + 2u].load())), m3);
            float3 ohi = sadd3(float3(vvUnkey(gAcc[kb + 3u].load()), vvUnkey(gAcc[kb + 4u].load()), vvUnkey(gAcc[kb + 5u].load())), m3);
            bool partner = b != a && vvOverlap(alo, ahi, olo, ohi);
            bool reset = false;
            if (partner)
            {
                gsVvOffered[q].add(vvCpCount(vb));
                float3 d = ssub3(va0, gNodes[gStatic[vb * 4u].x * 3u + 1u].xyz);
                if (sdot(d, d) > VV_APPROACH_SQ)
                {
                    reset = true;
                    gsVvReset[q] = 1u;
                    gMisc[vvRec(vb, VV_FLAGS)].w = 1u;
                    gMisc[vvRec(va, VV_FLAGS)].w = 1u;
                }
            }
            gNext[vvPartnerAt(off, b)] = partner ? 1u : 0u;
            gNext[vvResetAt(off, b)] = reset ? 1u : 0u;
        }
        sync();
        // Phase P.
        uint offered = gsVvOffered[q].load();
        bool ownerReset = gsVvReset[q] != 0u;
        if (lane == 0u)
        {
            gsVvOffered[q ^ 1u].store(0u);
            gsVvListed[q ^ 1u].store(0u);
            gsVvCands[q ^ 1u].store(0u);
            gsVvHits[q ^ 1u].store(0u);
            gsVvReset[q ^ 1u] = 0u;
        }
        for (uint m = 0u, mStop = EB_END(198, m, k, 1u); m < mStop; m++)
        {
            if (m == a || gNext[vvResetAt(off, m)] == 0u) continue;
            uint4 r1 = gStatic[vvMember(off, m) * 4u + 1u];
            for (uint t = lane, tStop = EB_END(199, t, r1.y, LANES); t < tStop; t += LANES) gMisc[gP.vvTri + r1.x + t].x = 0u;
        }
        // The previous owner's two-body counts (its hits were applied in phase S).
#ifdef EB_EDGE
        if (prevHits > 0u || (a > 0u && edgeOn() && edgeHits(prevVa) > 0u)) vvSplitClear(off, k, 1u, lane);
#else
        if (prevHits > 0u) vvSplitClear(off, k, 1u, lane);
#endif
        uint4 ra1 = gStatic[va * 4u + 1u];
        float range = asfloat(gStatic[va * 4u + 3u].z);
        // The partners' boxes' union (their contact points all lie in it).
        float3 plo = float3(3.0e38, 3.0e38, 3.0e38);
        float3 phi = -plo;
        float vmaxB = 0.0;
        for (uint m = 0u, mStop = EB_END(200, m, k, 1u); m < mStop; m++)
        {
            if (m == a || gNext[vvPartnerAt(off, m)] == 0u) continue;
            uint kb = vvBoxKey(vvMember(off, m), parity);
            plo = min(plo, float3(vvUnkey(gAcc[kb].load()), vvUnkey(gAcc[kb + 1u].load()), vvUnkey(gAcc[kb + 2u].load())));
            phi = max(phi, float3(vvUnkey(gAcc[kb + 3u].load()), vvUnkey(gAcc[kb + 4u].load()), vvUnkey(gAcc[kb + 5u].load())));
            if (gP.clipMem != 0u) vmaxB = max(vmaxB, clipVmax(vvMember(off, m), parity));
        }
        float extra = vvClipExtra(va, vmaxB, parity);
        for (uint t = lane, tStop = EB_END(201, t, ra1.y, LANES); t < tStop; t += LANES)
        {
            uint4 c = gMisc[gP.vvTri + ra1.x + t];
            int cwait = ownerReset ? 0 : asint(c.x);
            int cidle = asint(c.y);
            if (offered == 0u)
            {
                if (ownerReset) gMisc[gP.vvTri + ra1.x + t].x = 0u;
                continue;
            }
            if (cwait > 0)
            {
                gMisc[gP.vvTri + ra1.x + t] = uint4(asuint(cwait - 1), asuint(cidle + 1), 0u, 0u);
                continue;
            }
            cwait = VV_MAX_WAIT < cidle ? VV_MAX_WAIT : cidle;
            // A triangle whose box (grown by the range) meets no partner's box has
            // no candidate: its throttle update at once, no tests.
            if (!vvNearBox(ra1.x + t, va, range, plo, phi, extra))
            {
                gMisc[gP.vvTri + ra1.x + t] = uint4(asuint(min(cwait + 1, throttleCap(ra1.x + t))), 0u, 0u, 0u);
                continue;
            }
            gMisc[gP.vvTri + ra1.x + t] = uint4(asuint(cwait), 0u, 0u, 0u);
            gNext[vvTestedAt(ra1.x + gsVvListed[q].add(1u))] = t;
        }
#ifdef EB_ABLATE
        if (offered > 0u && (gP.ablate & 4096u) == 0u)
#else
        if (offered > 0u)
#endif
        {
            float3 r3 = float3(range, range, range);
            float3 clo = ssub3(blo, r3);
            float3 chi = sadd3(bhi, r3);
            // Anti-clip: the point's own travel (per point) on top of the bound.
            float ownerReach = gP.clipMem != 0u ? CLIP_DEPTH + clipVmax(va, parity) * STEP + 0.001 : 0.0;
            for (uint m = 0u, mStop = EB_END(202, m, k, 1u); m < mStop; m++)
            {
                if (m == a || gNext[vvPartnerAt(off, m)] == 0u) continue;
                uint v = vvMember(off, m);
                uint4 cp = gStatic[gP.vvStatic + v];
                float3 o = asfloat(gMisc[v].xyz);
                float3 orel = vvRel(v, va);
                for (uint i = lane, iStop = EB_END(203, i, cp.y, LANES); i < iStop; i += LANES)
                {
                    uint h = gLists[cp.x + i];
                    float3 x = nf3(o + gNodes[h * 3u].xyz);
                    float ex = 0.0;
                    if (gP.clipMem != 0u)
                    {
                        float3 vh3 = gNodes[h * 3u + 1u].xyz;
                        ex = ownerReach + sqrt(dot(vh3, vh3)) * STEP;
                    }
                    float3 e3 = float3(ex, ex, ex);
                    if (any(x < clo - e3) || any(x > chi + e3)) continue;
                    uint ci = gsVvCands[q].add(1u);
                    gBeams[gP.vvRun + runOff + ci] = float4(nf3(orel + gNodes[h * 3u].xyz), asfloat(h));
                    if ((gP.perfFlags & (PERF_VV_STREAM | PERF_VV_BIG_GRID)) == PERF_VV_STREAM)
                    {
                        // The candidate's part of vvInBox's pad, its vehicle's anti-clip and
                        // its vehicle (vvTriangleCands; the group's slice of the small
                        // groups' sort records, which a larger group does not use).
                        float3 vs = gNodes[h * 3u + 1u].xyz;
                        gBeams[vvSortAt(runOff) + ci] = float4(smul(ssqrt(sdot(vs, vs)), STEP),
                                                              (gStatic[v * 4u + 3u].x & FLAG_ANTI_CLIP) != 0u ? 1.0 : 0.0, asfloat(v), 0.0);
                    }
                }
            }
        }
        sync();
#ifdef EB_EDGE
        // Edge vehicle contact: the owner's hits (after the counts' clear, before phase T).
        if (edgeOn()) edgeOwnerBig(a, k, off, lane);
#endif
        // Phase T.
        uint listed = gsVvListed[q].load();
        uint cands = gsVvCands[q].load();
        // Lanes per triangle: all the lanes over the listed triangles.
        uint subs = clamp(LANES / max(listed, 1u), 1u, 64u);
#ifdef EB_ABLATE
        if ((gP.ablate & 2048u) != 0u) listed = 0u;
#endif
        bool bigGrid = (gP.perfFlags & PERF_VV_BIG_GRID) != 0u;
        if (bigGrid) vvBigGridBuild(va, runOff, cands, lane);
        for (uint p = lane, pStop = EB_END(204, p, listed * subs, LANES); p < pStop; p += LANES)
        {
            uint t = gNext[vvTestedAt(ra1.x + p / subs)];
            if (bigGrid)
                vvTriangleCandsBigGrid(va, ra1.x + t, p % subs, subs, runOff, points, cands, range, parity, q);
            else
                vvTriangleCands(va, ra1.x + t, p % subs, subs, runOff, points, cands, range, parity, q);
        }
        sync();
        prevTri = ra1.x;
        prevListed = listed;
        prevVa = va;
        prevRange = range;
    }
    for (uint i = lane, iStop = EB_END(205, i, prevListed, LANES); i < iStop; i += LANES) vvThrottleEnd(prevTri + gNext[vvTestedAt(prevTri + i)]);
    uint lastHits = k > 0u ? gsVvHits[(k - 1u) & 1u].load() : 0u;
    if (k > 0u) vvApplyHits(prevVa, runOff, points, lastHits, prevRange, parity, lane);
#ifdef EB_EDGE
    if (k > 0u && edgeOn()) edgeApplyOwner(prevVa, 0u, lane, LANES);
#endif
    if (gP.vvSplit != 0u)
    {
        for (uint ph = 0u; ph < 3u; ph++)
        {
            sync();
            if (k > 0u && lastHits <= 2u * points)
                vvResidualPhase(prevVa, vvHitAt(runOff, 0u), lastHits, prevRange, parity, lane, LANES, 0u, ph);
#ifdef EB_EDGE
            if (k > 0u && edgeOn()) edgeResidualPhase(prevVa, 0u, parity, lane, LANES, ph);
#endif
        }
        sync();
#ifdef EB_EDGE
        if (lastHits > 0u || (k > 0u && edgeOn() && edgeHits(prevVa) > 0u)) vvSplitClear(off, k, 1u, lane);
#else
        if (lastHits > 0u) vvSplitClear(off, k, 1u, lane);
#endif
    }
    // (The counters are cleared by the next step's first member step, before its
    // first barrier.)
}

// --- The contact pass of a small group (k <= VV_PART owners), owners side by side ---
// The same results as vvPass's owner-by-owner pass. What depends on the owner order
// is only the forces: a hit reads its candidate's current force. Everything else
// reads positions, which the contact pass does not move, or commutes: (S) every
// owner's partner scan (symmetric: a fast approach marks both), (P) every owner's
// throttle step and candidates, (T) every owner's triangle tests into its own hit
// list, each with one barrier for all owners. The throttle resets need no order
// either: an owner whose partner approaches fast resets its own triangles (as its
// own scan finds the same partner), and a later owner's reset of an earlier owner's
// triangles, which in vvPass comes after that owner's throttle update, is applied
// after it here (gsVvFix). Then the hits are applied in levels: an owner joins the
// level after every earlier owner with hits whose vehicles it may touch (itself and
// its partners: a hit reads and adds to a partner's contact point and adds to the
// owner's triangle nodes), so each owner's hits see exactly the earlier owners'
// forces, as in vvPass. Forces are fixed-point sums, so the order inside a level
// does not matter. An owner whose hits overflow its list re-walks its triangles at
// its level, applying directly (vvPass's overflow path).
groupshared uint gsVvPair[VV_PART * VV_PART];
#ifdef EB_EDGE
// Edge vehicle contact: the member listed edge hits this step (explicit_beams_vv_edges.slang
// edgeTestsSmall; read with gP.edgeBase only).
groupshared uint gsEdgeHas[VV_PART];
#endif
groupshared Atomic<uint> gsVvOff4[VV_PART];
groupshared Atomic<uint> gsVvList4[VV_PART];
groupshared Atomic<uint> gsVvCand4[VV_PART];
groupshared Atomic<uint> gsVvHit4[VV_PART];
groupshared uint gsVvReset4[VV_PART];
groupshared uint gsVvFix[VV_PART];

void vvClearSmall()
{
    for (uint a = 0u; a < VV_PART; a++)
    {
        gsVvOff4[a].store(0u);
        gsVvList4[a].store(0u);
        gsVvCand4[a].store(0u);
        gsVvHit4[a].store(0u);
        gsVvReset4[a] = 0u;
        gsVvFix[a] = 0u;
        for (uint b = 0u; b < VV_PART; b++) gsVvPair[a * VV_PART + b] = 0u;
    }
}

// The tests' counters, the owners' fastest candidates and the grid's offsets: in
// workgroup memory in a group's own pass; in the split path's test dispatches
// (EB_VV_GLOBAL: many workgroups per group) in the group's scratch, which its pass's
// front fills (explicit_beams_split.slang, splitVvAt / splitVvOffAt).
#ifdef EB_VV_GLOBAL
static uint gVvC; // gAcc: per owner its pairs (0..3), hits (4..7), fastest (8..11)
static uint gVvO; // gNext: the grid's offsets
#ifdef EB_VV_FLEET
// (With the owners' state by vehicle, gFl: the owner's record and its own grid's offsets.)
uint vvHitAdd(uint a) { return uint(gAcc[gFl != 0u ? flOwn(flVehOf(a)) + FL_HITS : gVvC + VV_PART + a].add(1)); }
uint vvPairAdd(uint a) { return uint(gAcc[gFl != 0u ? flOwn(flVehOf(a)) + FL_PAIRS : gVvC + a].add(1)); }
float vvSpeedAt(uint a) { return asfloat(uint(gAcc[gFl != 0u ? flOwn(flVehOf(a)) + FL_FAST : gVvC + 2u * VV_PART + a].load())); }
uint vvOffAt(uint b) { return gNext[(gFl != 0u ? gP.vvScratchOff + gFlV * (GRID_BUCKETS + 1u) : gVvO) + b]; }
#else
uint vvHitAdd(uint a) { return uint(gAcc[gVvC + VV_PART + a].add(1)); }
uint vvPairAdd(uint a) { return uint(gAcc[gVvC + a].add(1)); }
float vvSpeedAt(uint a) { return asfloat(uint(gAcc[gVvC + 2u * VV_PART + a].load())); }
uint vvOffAt(uint b) { return gNext[gVvO + b]; }
#endif
#else
uint vvHitAdd(uint a) { return gsVvHit4[a].add(1u); }
uint vvPairAdd(uint a) { return gsPairCountM[a].add(1u); }
float vvSpeedAt(uint a) { return asfloat(gsCount[GRID_BUCKETS + a].load()); }
uint vvOffAt(uint b) { return gsOff[b]; }
#endif

// Owner a's candidate run and hit list (each owner its own, within the group's
// regions, which hold four times the group's points).
#ifdef EB_VV_FLEET
// (With the owners' state by vehicle, gFl: the owner's part of the fleet-wide run and hit
// list, at its prefix of the owners' offered counts.)
uint vvRunSmall(uint runOff, uint points, uint a) { return gFl != 0u ? gP.vvRun + flRun(flVehOf(a)) : gP.vvRun + VV_PART * runOff + a * points; }
uint vvHitSmall(uint runOff, uint points, uint a, uint i)
{
    return gFl != 0u ? gP.vvHits + 2u * (2u * flRun(flVehOf(a)) + i) : gP.vvHits + 2u * (2u * VV_PART * runOff + 2u * a * points + i);
}
#else
uint vvRunSmall(uint runOff, uint points, uint a) { return gP.vvRun + VV_PART * runOff + a * points; }
uint vvHitSmall(uint runOff, uint points, uint a, uint i) { return gP.vvHits + 2u * (2u * VV_PART * runOff + 2u * a * points + i); }
#endif

// vvTriangleCands into owner a's own list (capacity: twice the group's points, as
// vvPass's); on overflow the list keeps its first entries and the owner re-walks at
// its level.
void vvTriangleCandsSmall(uint a, uint va, uint gt, uint sub, uint subs, uint runOff, uint points, uint cands,
                          float r)
{
    VvTri t = vvTriangle(gt, vvRel(va, va), r);
    bool cand = false;
    bool hit = false;
    uint run = vvRunSmall(runOff, points, a);
    vvVisit(sub < cands ? (cands - sub + subs - 1u) / subs : 0u);
    for (uint e = sub, eStop = EB_END(206, e, cands, subs); e < eStop; e += subs)
    {
        float4 E = gBeams[run + e];
        uint h = asuint(E.w);
        uint vh = gStatic[gP.nodeInfoBase + h].w;
        bool anti = vvAnti(t, vh);
        if (!vvInBox(t, E.xyz, h, anti)) continue;
        cand = true;
        if (!vvFind(t, gt, h, E.xyz, vh, r, anti)) continue;
        hit = true;
        vvSplitCount(t, h, a);
        uint i = vvHitAdd(a);
        if (i < 2u * points)
        {
            gNext[vvHitSmall(runOff, points, a, i)] = gt;
            gNext[vvHitSmall(runOff, points, a, i) + 1u] = h;
        }
    }
    vvMark(gt, cand, hit);
}

// The overflow path: owner a's triangle walk again, applying each hit at once.
void vvTriangleCandsApply(uint va, uint gt, uint sub, uint subs, uint run, uint cands, float r, uint parity, uint slot)
{
    VvTri t = vvTriangle(gt, vvRel(va, va), r);
    for (uint e = sub, eStop = EB_END(207, e, cands, subs); e < eStop; e += subs)
    {
        float4 E = gBeams[run + e];
        uint h = asuint(E.w);
        uint vh = gStatic[gP.nodeInfoBase + h].w;
        bool anti = vvAnti(t, vh);
        if (!vvInBox(t, E.xyz, h, anti)) continue;
        if (!vvHitTest(t, gt, h, E.xyz, vh, r, anti)) continue;
        vvTry(t, gt, h, E.xyz, vh, r, parity, slot);
    }
}


// Owner a's level: 1 + the level of every earlier owner with hits whose vehicles
// ({itself} and its partners) meet a's (0: a has no hits).
uint vvLevel(uint a, uint k, uint points)
{
    uint lv[VV_PART];
    uint mine = 0u;
    for (uint b = 0u, bStop = EB_END(208, b, k, 1u); b < bStop; b++)
    {
        lv[b] = 0u;
        if (gsVvHit4[b].load() == 0u) continue;
        uint l = 1u;
        for (uint c = 0u, cStop = EB_END(209, c, b, 1u); c < cStop; c++)
        {
            if (lv[c] == 0u) continue;
            // Touched vehicle sets of b and c meet: b == c's partner, or a common
            // partner (bit 1 of gsVvPair: partners).
            bool meet = (gsVvPair[b * VV_PART + c] & 1u) != 0u;
            for (uint d = 0u, dStop = EB_END(210, d, k, 1u); d < dStop; d++)
                meet = meet || ((gsVvPair[b * VV_PART + d] & 1u) != 0u && (gsVvPair[c * VV_PART + d] & 1u) != 0u);
            if (meet) l = max(l, lv[c] + 1u);
        }
        lv[b] = l;
        if (b == a) mine = l;
    }
    return mine;
}


// Every owner's level once a pass (lane 0; vvLevel's rule: the hit counts and the
// partner bits are final when the levels start), four bits an owner in gsUniformLanes
// (no workgroup variable to spare on Metal); each lane keeps the word (gVvLv).
static uint gVvLv;
uint vvLv(uint a) { return (gVvLv >> (4u * a)) & 15u; }
void vvLevelsSet(uint k, uint lane)
{
    if (lane != 0u) return;
    uint lv[VV_PART];
    uint packed = 0u;
    for (uint b = 0u, bStop = EB_END(211, b, k, 1u); b < bStop; b++)
    {
        uint l = 0u;
#ifdef EB_EDGE
        if (gsVvHit4[b].load() != 0u || (gP.edgeBase != 0u && gsEdgeHas[b] != 0u))
#else
        if (gsVvHit4[b].load() != 0u)
#endif
        {
            l = 1u;
            for (uint c = 0u, cStop = EB_END(212, c, b, 1u); c < cStop; c++)
            {
                if (lv[c] == 0u) continue;
                bool meet = (gsVvPair[b * VV_PART + c] & 1u) != 0u;
                for (uint d = 0u, dStop = EB_END(213, d, k, 1u); d < dStop; d++)
                    meet = meet || ((gsVvPair[b * VV_PART + d] & 1u) != 0u && (gsVvPair[c * VV_PART + d] & 1u) != 0u);
                if (meet) l = max(l, lv[c] + 1u);
            }
        }
        lv[b] = l;
        packed |= l << (4u * b);
    }
    gsUniformLanes = packed;
}

// --- The small groups' candidates in a grid (lossless) ---
// The (T) phase tested every listed triangle against every candidate of its owner (the
// partners' contact points in the owner's box): in a crash both counts are hundreds a
// step. With the grid, after the candidates are listed each owner's candidates are
// counted into buckets of its own cells (gsCount, free in a group's frame: owner a's
// buckets a·256 …; a counting sort, the sorted copy after the runs at vvSortAt), and a
// triangle walks the cells of its box padded by the most any candidate's box test pads it
// (vvInBox with the owner's fastest candidate), each bucket once: every candidate inside
// the box it tests is in a covered cell, so the same candidates pass the box test and the
// same hits are found (applied in the canonical order whatever order they were found in).
// A box over more than GRID_MAX_CELLS cells tests every
// candidate as before.
uint vvTotalCp() { return gP.vvHits - (gP.vvBig + gP.vehicles); }
// The partners' room and the regions' unit: that many times the fleet's contact points
// (the run and its sorted copy in gBeams; in gNext four units of hits, one of candidate
// ranks, four of the hits' order).
uint vvRoom() { return gP.vvRoom != 0u ? gP.vvRoom : VV_PART; }
uint vvRoomCp() { return vvRoom() * vvTotalCp(); }
#ifdef EB_VV_FLEET
// (gFl: the one owner's sorted copy at its part's place, every bucket of its own grid.)
uint vvSortAt(uint runOff) { return gP.vvRun + vvRoomCp() + (gFl != 0u ? flRun(gFlV) : VV_PART * runOff); }
uint vvGridBucket(uint a, int3 c) { return gFl != 0u ? gridBucket(c) : a * (GRID_BUCKETS / VV_PART) + (gridBucket(c) & (GRID_BUCKETS / VV_PART - 1u)); }
#else
uint vvSortAt(uint runOff) { return gP.vvRun + vvRoomCp() + VV_PART * runOff; }
uint vvGridBucket(uint a, int3 c) { return a * (GRID_BUCKETS / VV_PART) + (gridBucket(c) & (GRID_BUCKETS / VV_PART - 1u)); }
#endif
uint vvRankAt(uint runOff, uint points, uint a, uint e) { return gP.vvHits + 4u * vvRoomCp() + VV_PART * runOff + a * points + e; }
// The grid's cell: the owner's self-collision cell, else half a metre (any size gives
// the same candidates; it only sets how many are visited).
float vvGridCell(uint va)
{
    float c = asfloat(gStatic[va * 4u + 3u].w);
    return c > 0.0 ? c : 0.5;
}
void vvVisit(uint n)
{
    if (n > 0u) gAcc[vvCtl(VV_VISITS)].add(int(n));
}

// Every lane (uniform; after the candidates' barrier): the owners' candidates bucketed,
// scanned and copied in bucket order. gsCount[GRID_BUCKETS + a]: owner a's fastest
// candidate's speed (float bits).
void vvGridBuild(uint k, uint off, uint runOff, uint points, uint lane)
{
    for (uint b = lane; b < GRID_BUCKETS; b += LANES) gsCount[b].store(0u);
    if (lane < VV_PART) gsCount[GRID_BUCKETS + lane].store(0u);
    sync();
    for (uint a = 0u, aStop = EB_END(214, a, k, 1u); a < aStop; a++)
    {
        uint va = vvMember(off, a);
        float cell = vvGridCell(va);
        uint cands = gsVvOff4[a].load() != 0u ? gsVvCand4[a].load() : 0u;
        uint run = vvRunSmall(runOff, points, a);
        for (uint e = lane, eStop = EB_END(215, e, cands, LANES); e < eStop; e += LANES)
        {
            float4 E = gBeams[run + e];
            uint h = asuint(E.w);
            uint b = vvGridBucket(a, int3(floor(E.xyz / cell)));
            gNext[vvRankAt(runOff, points, a, e)] = b | (gsCount[b].add(1u) << 10);
            float3 v = gNodes[h * 3u + 1u].xyz;
            uint sp = asuint(ssqrt(sdot(v, v)));
            if (sp > gsCount[GRID_BUCKETS + a].load()) gsCount[GRID_BUCKETS + a].max(sp);
        }
    }
    sync();
    // Offsets: per lane its buckets' sum, a scan over blocks of lanes, the blocks' totals.
    uint per = GRID_BUCKETS / LANES;
    uint block = LANES / GRID_BLOCKS;
    uint sum = 0u;
    for (uint q = 0u, qStop = EB_END(216, q, per, 1u); q < qStop; q++) sum += gsCount[lane * per + q].load();
    gsPart[lane] = sum;
    sync();
    if (lane < GRID_BLOCKS)
    {
        uint run = 0u;
        for (uint q = 0u, qStop = EB_END(217, q, block, 1u); q < qStop; q++)
        {
            uint x = gsPart[lane * block + q];
            gsPart[lane * block + q] = run;
            run += x;
        }
        gsBlk[lane] = run;
    }
    sync();
    if (lane == 0u)
    {
        uint run = 0u;
        for (uint q = 0u; q < GRID_BLOCKS; q++)
        {
            uint x = gsBlk[q];
            gsBlk[q] = run;
            run += x;
        }
        gsOff[GRID_BUCKETS] = run;
    }
    sync();
    {
        uint run = gsPart[lane] + gsBlk[lane / block];
        for (uint q = 0u, qStop = EB_END(218, q, per, 1u); q < qStop; q++)
        {
            gsOff[lane * per + q] = run;
            run += gsCount[lane * per + q].load();
        }
    }
    sync();
    uint sorted = vvSortAt(runOff);
    for (uint a = 0u, aStop = EB_END(219, a, k, 1u); a < aStop; a++)
    {
        uint va = vvMember(off, a);
        float cell = vvGridCell(va);
        uint cands = gsVvOff4[a].load() != 0u ? gsVvCand4[a].load() : 0u;
        uint run = vvRunSmall(runOff, points, a);
        for (uint e = lane, eStop = EB_END(220, e, cands, LANES); e < eStop; e += LANES)
        {
            uint br = gNext[vvRankAt(runOff, points, a, e)];
            gBeams[sorted + gsOff[br & (GRID_BUCKETS - 1u)] + (br >> 10)] = gBeams[run + e];
        }
    }
    sync();
}

// vvTriangleCandsSmall with the grid: the same candidates, from the cells the box covers.
void vvTriangleCandsGrid(uint a, uint va, uint gt, uint sub, uint subs, uint runOff, uint points, uint cands, float r)
{
    float cell = vvGridCell(va);
    VvTri t = vvTriangle(gt, vvRel(va, va), r);
    // The most vvInBox pads this box (its anti-clip pad with the fastest candidate).
    float vmax = vvSpeedAt(a);
    float pad = nf(nf(t.travel + smul(vmax, STEP)) + CLIP_DEPTH);
    float3 lo = ssub3(t.lo, float3(pad, pad, pad));
    float3 hi = sadd3(t.hi, float3(pad, pad, pad));
    if (any(t.lo > t.hi)) return; // a dead triangle's empty box: no candidate in it
    float3 f0 = floor(lo / cell);
    float3 fs = floor(hi / cell) - f0 + float3(1.0, 1.0, 1.0);
    bool every = !(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS))
                 || !(all(abs(f0) < 1.0e8));
    if (every)
    {
        vvTriangleCandsSmall(a, va, gt, sub, subs, runOff, points, cands, r);
        return;
    }
    bool cand = false;
    bool hit = false;
    int3 c0 = int3(f0);
    int3 span = int3(fs);
    uint yz = uint(span.y * span.z);
    uint cells = uint(span.x) * yz;
    uint sorted = vvSortAt(runOff);
    uint seen = 0u;
    for (uint i = sub, iStop = EB_END(221, i, cells, subs); i < iStop; i += subs)
    {
        uint b = vvGridBucket(a, c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z))));
        bool dup = false;
        int3 cj = int3(0, 0, 0);
        for (uint j = 0u, jStop = EB_END(222, j, i, 1u); j < jStop; j++)
        {
            if (vvGridBucket(a, c0 + cj) == b)
            {
                dup = true;
                break;
            }
            cj.z += 1;
            if (cj.z == span.z)
            {
                cj.z = 0;
                cj.y += 1;
                if (cj.y == span.y)
                {
                    cj.y = 0;
                    cj.x += 1;
                }
            }
        }
        if (dup) continue;
        uint e0 = vvOffAt(b);
        uint e1 = vvOffAt(b + 1u);
        seen += e1 - e0;
        for (uint e = e0, eGuard = 0u; e < e1; e++)
        {
            EB_PASS(223, eGuard);
            float4 E = gBeams[sorted + e];
            uint h = asuint(E.w);
            uint vh = gStatic[gP.nodeInfoBase + h].w;
            bool anti = vvAnti(t, vh);
            if (!vvInBox(t, E.xyz, h, anti)) continue;
            cand = true;
            if (!vvFind(t, gt, h, E.xyz, vh, r, anti)) continue;
            hit = true;
            vvSplitCount(t, h, a);
            uint n = vvHitAdd(a);
            if (n < 2u * points)
            {
                gNext[vvHitSmall(runOff, points, a, n)] = gt;
                gNext[vvHitSmall(runOff, points, a, n) + 1u] = h;
            }
        }
    }
    vvVisit(seen);
    vvMark(gt, cand, hit);
}


