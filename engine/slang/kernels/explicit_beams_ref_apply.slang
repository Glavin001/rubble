// Owner va's n listed hits (triangle, node: two words each from `base`) in the
// specification's order (triangles in list order, then contact points in node
// order). A hit reads its contact point's force and adds to it and to the owner's
// triangle nodes; the owner's nodes are never contact points of its own hits, so
// only the hits on one contact point depend on each other: one lane per contact
// point runs its hits in triangle order (each reads the ones before), the points
// side by side.
void vvApplyOrdered(uint va, uint base, uint n, float r, uint parity, uint lane, uint lanes)
{
    float3 oa = asfloat(gMisc[va].xyz);
    for (uint i = lane; i < n; i += lanes)
    {
        uint gt = gNext[base + 2u * i];
        uint h = gNext[base + 2u * i + 1u];
        bool first = true;
        for (uint j = 0u; j < n; j++)
        {
            if (gNext[base + 2u * j + 1u] == h && gNext[base + 2u * j] < gt)
            {
                first = false;
                break;
            }
        }
        if (!first) continue;
        uint vh = gStatic[gP.nodeInfoBase + h].w;
        for (uint it = 0u; it < n; it++)
        {
            VvTri t = vvTriangle(gt, oa, r);
            vvTry(t, gt, h, nf3(asfloat(gMisc[vh].xyz) + gNodes[h * 3u].xyz), vh, r, parity);
            // The point's next hit: the smallest later triangle.
            uint next = 0xFFFFFFFFu;
            for (uint j = 0u; j < n; j++)
            {
                uint gj = gNext[base + 2u * j];
                if (gNext[base + 2u * j + 1u] == h && gj > gt && gj < next) next = gj;
            }
            if (next == 0xFFFFFFFFu) break;
            gt = next;
        }
    }
}

// The listed hits of owner va, in order (vvApplyOrdered).
void vvApplyHits(uint va, uint runOff, uint points, uint hits, float r, uint parity, uint lane)
{
    vvApplyOrdered(va, vvHitAt(runOff, 0u), min(hits, 2u * points), r, parity, lane, LANES);
}


// A tested triangle's throttle after its tests (a hit → 0, no candidate → + 1).
void vvThrottleEnd(uint gt)
{
    uint4 c = gMisc[gP.vvTri + gt];
    int cwait = asint(c.x);
    if (c.w != 0u) cwait = 0;
    else if (c.z == 0u) cwait = cwait + 1;
    gMisc[gP.vvTri + gt] = uint4(asuint(cwait), c.y, 0u, 0u);
}

// Triangle gt of owner va: its box grown by the range overlaps the box `lo..hi`
// (the partners' boxes' union), so it may have candidates.
bool vvNearBox(uint gt, uint va, float r, float3 blo, float3 bhi)
{
    uint4 tn = gStatic[gP.triBase + gt];
    float3 o = asfloat(gMisc[va].xyz);
    float3 p0 = nf3(o + gNodes[tn.x * 3u].xyz);
    float3 p1 = nf3(o + gNodes[tn.y * 3u].xyz);
    float3 p2 = nf3(o + gNodes[tn.z * 3u].xyz);
    float3 r3 = float3(r, r, r);
    return vvOverlap(ssub3(min(min(p0, p1), p2), r3), sadd3(max(max(p0, p1), p2), r3), blo, bhi);
}

void vvClearCounters()
{
    for (uint q = 0u; q < 2u; q++)
    {
        gsVvOffered[q].store(0u);
        gsVvListed[q].store(0u);
        gsVvCands[q].store(0u);
        gsVvHits[q].store(0u);
        gsVvReset[q] = 0u;
    }
}

// The contact pass after every member's step (their boxes already in the step's
// keys): each owner in vehicle order, three barriers each. Phase S: the partner
// scan over the other members (and the previous owner's throttle updates); phase
// P: contact throttle resets, the owner's throttle step listing its tested
// triangles, and its candidates compacted; phase T: tested triangles against the
// candidates.
void vvPass(uint k, uint off, uint runOff, uint points, uint step, uint lane)
{
    uint parity = step & 1u;
    float3 m3 = float3(VV_BOX_MARGIN, VV_BOX_MARGIN, VV_BOX_MARGIN);
    uint prevTri = 0u;
    uint prevListed = 0u;
    uint prevVa = 0u;
    float prevRange = 0.0;
    for (uint a = 0u; a < k; a++)
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
        for (uint i = lane; i < prevListed; i += LANES) vvThrottleEnd(prevTri + gNext[vvTestedAt(prevTri + i)]);
#ifdef EB_ABLATE
        if ((gP.ablate & 1024u) == 0u)
#endif
        if (a > 0u) vvApplyHits(prevVa, runOff, points, gsVvHits[q ^ 1u].load(), prevRange, parity, lane);
        for (uint b = lane; b < k; b += LANES)
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
        for (uint m = 0u; m < k; m++)
        {
            if (m == a || gNext[vvResetAt(off, m)] == 0u) continue;
            uint4 r1 = gStatic[vvMember(off, m) * 4u + 1u];
            for (uint t = lane; t < r1.y; t += LANES) gMisc[gP.vvTri + r1.x + t].x = 0u;
        }
        uint4 ra1 = gStatic[va * 4u + 1u];
        float range = asfloat(gStatic[va * 4u + 3u].z);
        // The partners' boxes' union (their contact points all lie in it).
        float3 plo = float3(3.0e38, 3.0e38, 3.0e38);
        float3 phi = -plo;
        for (uint m = 0u; m < k; m++)
        {
            if (m == a || gNext[vvPartnerAt(off, m)] == 0u) continue;
            uint kb = vvBoxKey(vvMember(off, m), parity);
            plo = min(plo, float3(vvUnkey(gAcc[kb].load()), vvUnkey(gAcc[kb + 1u].load()), vvUnkey(gAcc[kb + 2u].load())));
            phi = max(phi, float3(vvUnkey(gAcc[kb + 3u].load()), vvUnkey(gAcc[kb + 4u].load()), vvUnkey(gAcc[kb + 5u].load())));
        }
        for (uint t = lane; t < ra1.y; t += LANES)
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
            if (!vvNearBox(ra1.x + t, va, range, plo, phi))
            {
                gMisc[gP.vvTri + ra1.x + t] = uint4(asuint(cwait + 1), 0u, 0u, 0u);
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
            for (uint m = 0u; m < k; m++)
            {
                if (m == a || gNext[vvPartnerAt(off, m)] == 0u) continue;
                uint v = vvMember(off, m);
                uint4 cp = gStatic[gP.vvStatic + v];
                float3 o = asfloat(gMisc[v].xyz);
                for (uint i = lane; i < cp.y; i += LANES)
                {
                    uint h = gLists[cp.x + i];
                    float3 x = nf3(o + gNodes[h * 3u].xyz);
                    if (any(x < clo) || any(x > chi)) continue;
                    gBeams[gP.vvRun + runOff + gsVvCands[q].add(1u)] = float4(x, asfloat(h));
                }
            }
        }
        sync();
        // Phase T.
        uint listed = gsVvListed[q].load();
        uint cands = gsVvCands[q].load();
        // Lanes per triangle: all the lanes over the listed triangles.
        uint subs = clamp(LANES / max(listed, 1u), 1u, 64u);
#ifdef EB_ABLATE
        if ((gP.ablate & 2048u) != 0u) listed = 0u;
#endif
        for (uint p = lane; p < listed * subs; p += LANES)
        {
            uint t = gNext[vvTestedAt(ra1.x + p / subs)];
            vvTriangleCands(va, ra1.x + t, p % subs, subs, runOff, points, cands, range, parity, q);
        }
        sync();
        prevTri = ra1.x;
        prevListed = listed;
        prevVa = va;
        prevRange = range;
    }
    for (uint i = lane; i < prevListed; i += LANES) vvThrottleEnd(prevTri + gNext[vvTestedAt(prevTri + i)]);
    if (k > 0u) vvApplyHits(prevVa, runOff, points, gsVvHits[(k - 1u) & 1u].load(), prevRange, parity, lane);
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

// Owner a's candidate run and hit list (each owner its own, within the group's
// regions, which hold four times the group's points).
uint vvRunSmall(uint runOff, uint points, uint a) { return gP.vvRun + 4u * runOff + a * points; }
uint vvHitSmall(uint runOff, uint points, uint a, uint i) { return gP.vvHits + 2u * (8u * runOff + 2u * a * points + i); }

// vvTriangleCands into owner a's own list (capacity: twice the group's points, as
// vvPass's); on overflow the list keeps its first entries and the owner re-walks at
// its level.
void vvTriangleCandsSmall(uint a, uint va, uint gt, uint sub, uint subs, uint runOff, uint points, uint cands,
                          float r)
{
    VvTri t = vvTriangle(gt, asfloat(gMisc[va].xyz), r);
    bool cand = false;
    bool hit = false;
    uint run = vvRunSmall(runOff, points, a);
    for (uint e = sub; e < cands; e += subs)
    {
        float4 E = gBeams[run + e];
        if (any(E.xyz < t.lo) || any(E.xyz > t.hi)) continue;
        uint h = asuint(E.w);
        cand = true;
        if (!vvInBand(t, E.xyz, r)) continue;
        hit = true;
        uint i = gsVvHit4[a].add(1u);
        if (i < 2u * points)
        {
            gNext[vvHitSmall(runOff, points, a, i)] = gt;
            gNext[vvHitSmall(runOff, points, a, i) + 1u] = h;
        }
    }
    if (cand) gMisc[gP.vvTri + gt].z = 1u;
    if (hit) gMisc[gP.vvTri + gt].w = 1u;
}

// The overflow path: owner a's triangle walk again, applying each hit at once.
void vvTriangleCandsApply(uint va, uint gt, uint sub, uint subs, uint run, uint cands, float r, uint parity)
{
    VvTri t = vvTriangle(gt, asfloat(gMisc[va].xyz), r);
    for (uint e = sub; e < cands; e += subs)
    {
        float4 E = gBeams[run + e];
        if (any(E.xyz < t.lo) || any(E.xyz > t.hi)) continue;
        uint h = asuint(E.w);
        if (!vvInBand(t, E.xyz, r)) continue;
        vvTry(t, gt, h, E.xyz, gStatic[gP.nodeInfoBase + h].w, r, parity);
    }
}


// Owner a's level: 1 + the level of every earlier owner with hits whose vehicles
// ({itself} and its partners) meet a's (0: a has no hits).
uint vvLevel(uint a, uint k, uint points)
{
    uint lv[VV_PART];
    uint mine = 0u;
    for (uint b = 0u; b < k; b++)
    {
        lv[b] = 0u;
        if (gsVvHit4[b].load() == 0u) continue;
        uint l = 1u;
        for (uint c = 0u; c < b; c++)
        {
            if (lv[c] == 0u) continue;
            // Touched vehicle sets of b and c meet: b == c's partner, or a common
            // partner (bit 1 of gsVvPair: partners).
            bool meet = (gsVvPair[b * VV_PART + c] & 1u) != 0u;
            for (uint d = 0u; d < k; d++)
                meet = meet || ((gsVvPair[b * VV_PART + d] & 1u) != 0u && (gsVvPair[c * VV_PART + d] & 1u) != 0u);
            if (meet) l = max(l, lv[c] + 1u);
        }
        lv[b] = l;
        if (b == a) mine = l;
    }
    return mine;
}

void vvPassSmall(uint k, uint off, uint runOff, uint points, uint step, uint lane)
{
    uint parity = step & 1u;
    float3 m3 = float3(VV_BOX_MARGIN, VV_BOX_MARGIN, VV_BOX_MARGIN);
    // (S) the partner scans: lane per ordered pair (a, b).
    for (uint p = lane; p < k * k; p += LANES)
    {
        uint a = p / k;
        uint b = p % k;
        if (a == b) continue;
        uint va = vvMember(off, a);
        uint vb = vvMember(off, b);
        uint ka = vvBoxKey(va, parity);
        uint kb = vvBoxKey(vb, parity);
        float3 alo = ssub3(float3(vvUnkey(gAcc[ka].load()), vvUnkey(gAcc[ka + 1u].load()), vvUnkey(gAcc[ka + 2u].load())), m3);
        float3 ahi = sadd3(float3(vvUnkey(gAcc[ka + 3u].load()), vvUnkey(gAcc[ka + 4u].load()), vvUnkey(gAcc[ka + 5u].load())), m3);
        float3 olo = ssub3(float3(vvUnkey(gAcc[kb].load()), vvUnkey(gAcc[kb + 1u].load()), vvUnkey(gAcc[kb + 2u].load())), m3);
        float3 ohi = sadd3(float3(vvUnkey(gAcc[kb + 3u].load()), vvUnkey(gAcc[kb + 4u].load()), vvUnkey(gAcc[kb + 5u].load())), m3);
        if (!vvOverlap(alo, ahi, olo, ohi)) continue;
        uint bits = 1u;
        gsVvOff4[a].add(vvCpCount(vb));
        float3 d = ssub3(gNodes[gStatic[va * 4u].x * 3u + 1u].xyz, gNodes[gStatic[vb * 4u].x * 3u + 1u].xyz);
        if (sdot(d, d) > VV_APPROACH_SQ)
        {
            bits = 3u;
            gsVvReset4[a] = 1u;
            if (b > a) gsVvFix[a] = 1u;
            gMisc[vvRec(vb, VV_FLAGS)].w = 1u;
            gMisc[vvRec(va, VV_FLAGS)].w = 1u;
        }
        gsVvPair[a * VV_PART + b] = bits;
    }
    sync();
    // (P) every owner's throttle step and candidates.
    {
        uint triTotal = 0u;
        for (uint a = 0u; a < k; a++) triTotal += gStatic[vvMember(off, a) * 4u + 1u].y;
        for (uint p = lane; p < triTotal; p += LANES)
        {
            uint a = 0u;
            uint t = p;
            for (uint c = 0u; c + 1u < k; c++)
            {
                uint n = gStatic[vvMember(off, a) * 4u + 1u].y;
                if (t < n) break;
                t -= n;
                a++;
            }
            uint va = vvMember(off, a);
            uint4 ra1 = gStatic[va * 4u + 1u];
            uint gt = ra1.x + t;
            uint offered = gsVvOff4[a].load();
            bool ownerReset = gsVvReset4[a] != 0u;
            bool fix = gsVvFix[a] != 0u;
            uint4 c = gMisc[gP.vvTri + gt];
            int cwait = ownerReset ? 0 : asint(c.x);
            int cidle = asint(c.y);
            if (offered == 0u)
            {
                if (ownerReset || fix) gMisc[gP.vvTri + gt].x = 0u;
                continue;
            }
            if (cwait > 0)
            {
                gMisc[gP.vvTri + gt] = uint4(fix ? 0u : asuint(cwait - 1), asuint(cidle + 1), 0u, 0u);
                continue;
            }
            cwait = VV_MAX_WAIT < cidle ? VV_MAX_WAIT : cidle;
            float range = asfloat(gStatic[va * 4u + 3u].z);
            float3 plo = float3(3.0e38, 3.0e38, 3.0e38);
            float3 phi = -plo;
            for (uint m = 0u; m < k; m++)
            {
                if ((gsVvPair[a * VV_PART + m] & 1u) == 0u) continue;
                uint kb = vvBoxKey(vvMember(off, m), parity);
                plo = min(plo, float3(vvUnkey(gAcc[kb].load()), vvUnkey(gAcc[kb + 1u].load()), vvUnkey(gAcc[kb + 2u].load())));
                phi = max(phi, float3(vvUnkey(gAcc[kb + 3u].load()), vvUnkey(gAcc[kb + 4u].load()), vvUnkey(gAcc[kb + 5u].load())));
            }
            if (!vvNearBox(gt, va, range, plo, phi))
            {
                gMisc[gP.vvTri + gt] = uint4(fix ? 0u : asuint(cwait + 1), 0u, 0u, 0u);
                continue;
            }
            gMisc[gP.vvTri + gt] = uint4(asuint(cwait), 0u, 0u, 0u);
            gNext[vvTestedAt(ra1.x + gsVvList4[a].add(1u))] = t;
        }
        // Candidates: each owner's partners' contact points in its box grown by
        // its range.
        for (uint a = 0u; a < k; a++)
        {
            if (gsVvOff4[a].load() == 0u) continue;
            uint va = vvMember(off, a);
            uint key = vvBoxKey(va, parity);
            float range = asfloat(gStatic[va * 4u + 3u].z);
            float3 r3 = float3(range, range, range);
            float3 clo = ssub3(float3(vvUnkey(gAcc[key].load()), vvUnkey(gAcc[key + 1u].load()), vvUnkey(gAcc[key + 2u].load())), r3);
            float3 chi = sadd3(float3(vvUnkey(gAcc[key + 3u].load()), vvUnkey(gAcc[key + 4u].load()), vvUnkey(gAcc[key + 5u].load())), r3);
            uint run = vvRunSmall(runOff, points, a);
            for (uint m = 0u; m < k; m++)
            {
                if ((gsVvPair[a * VV_PART + m] & 1u) == 0u) continue;
                uint v = vvMember(off, m);
                uint4 cp = gStatic[gP.vvStatic + v];
                float3 o = asfloat(gMisc[v].xyz);
                for (uint i = lane; i < cp.y; i += LANES)
                {
                    uint h = gLists[cp.x + i];
                    float3 x = nf3(o + gNodes[h * 3u].xyz);
                    if (any(x < clo) || any(x > chi)) continue;
                    gBeams[run + gsVvCand4[a].add(1u)] = float4(x, asfloat(h));
                }
            }
        }
    }
    sync();
    // (T) every owner's listed triangles against its candidates.
    {
        uint total = 0u;
        for (uint a = 0u; a < k; a++) total += gsVvList4[a].load();
        uint subs = clamp(LANES / max(total, 1u), 1u, 64u);
        for (uint p = lane; p < total * subs; p += LANES)
        {
            uint q = p / subs;
            uint a = 0u;
            for (uint c = 0u; c + 1u < k; c++)
            {
                uint n = gsVvList4[a].load();
                if (q < n) break;
                q -= n;
                a++;
            }
            uint va = vvMember(off, a);
            uint4 ra1 = gStatic[va * 4u + 1u];
            uint t = gNext[vvTestedAt(ra1.x + q)];
            vvTriangleCandsSmall(a, va, ra1.x + t, p % subs, subs, runOff, points, gsVvCand4[a].load(),
                                 asfloat(gStatic[va * 4u + 3u].z));
        }
    }
    sync();
    // The hits, level by level (at most k levels); with the first, every listed
    // triangle's throttle update (its marks are final) and a later owner's reset.
    for (uint l = 1u; l <= k; l++)
    {
        if (l == 1u)
        {
            uint total = 0u;
            for (uint a = 0u; a < k; a++) total += gsVvList4[a].load();
            for (uint p = lane; p < total; p += LANES)
            {
                uint q = p;
                uint a = 0u;
                for (uint c = 0u; c + 1u < k; c++)
                {
                    uint n = gsVvList4[a].load();
                    if (q < n) break;
                    q -= n;
                    a++;
                }
                uint triOff = gStatic[vvMember(off, a) * 4u + 1u].x;
                uint gt = triOff + gNext[vvTestedAt(triOff + q)];
                vvThrottleEnd(gt);
                if (gsVvFix[a] != 0u) gMisc[gP.vvTri + gt].x = 0u;
            }
        }
        for (uint a = 0u; a < k; a++)
        {
            if (vvLevel(a, k, points) != l) continue;
            uint va = vvMember(off, a);
            float range = asfloat(gStatic[va * 4u + 3u].z);
            uint hits = gsVvHit4[a].load();
            if (hits <= 2u * points)
                vvApplyOrdered(va, vvHitSmall(runOff, points, a, 0u), hits, range, parity, lane, LANES);
            else
            {
                uint4 ra1 = gStatic[va * 4u + 1u];
                uint listed = gsVvList4[a].load();
                uint subs = clamp(LANES / max(listed, 1u), 1u, 64u);
                for (uint p = lane; p < listed * subs; p += LANES)
                {
                    uint t = gNext[vvTestedAt(ra1.x + p / subs)];
                    vvTriangleCandsApply(va, ra1.x + t, p % subs, subs, vvRunSmall(runOff, points, a),
                                         gsVvCand4[a].load(), range, parity);
                }
            }
        }
        sync();
    }
}

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_contact(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    contactGroup(gid.x, lid.x);
}

// Contact group g: its members stepped together with the contact pass.
void contactGroup(uint g, uint lane)
{
    uint4 G = gMisc[gP.vvGroups + g];
    uint off = G.x;
    uint runOff = G.z;
    uint points = G.w;
    if (lane == 0u)
    {
        gsUniformLanes = G.y;
    }
    sync();
    uint k = gsUniformLanes;
    sync();
    // Members in vehicle order (a rank sort), their run prefixes and member
    // indices.
    for (uint i = lane; i < k; i += LANES)
    {
        uint v = gNext[gP.vvMembers + off + i];
        uint rank = 0u;
        for (uint j = 0u; j < k; j++) rank += gNext[gP.vvMembers + off + j] < v ? 1u : 0u;
        gNext[gP.vvMembers + gP.vehicles + off + rank] = v;
        gMisc[vvRec(v, VV_MEMBER)].x = rank;
    }
    sync();
    if (lane == 0u)
    {
        uint run = 0u;
        for (uint m = 0u; m < k; m++)
        {
            gNext[vvPrefixAt(off, m)] = run;
            run += vvCpCount(vvMember(off, m));
        }
        vvClearCounters();
    }
    sync();
    for (uint m = 0u; m < k; m++) contactMemberBegin(vvMember(off, m), lane);
    uint stepBase = gMisc[vvMember(off, 0u)].w;
    for (uint s = 0u; s < gP.steps; s++)
    {
        if (k <= VV_PART)
        {
            contactMembersStep(off, k, lane, s);
        }
        else
        {
            for (uint m = 0u; m < k; m++) contactMemberStep(vvMember(off, m), lane, s, m == 0u);
        }
#ifdef EB_ABLATE
        if ((gP.ablate & 512u) != 0u) {} else
#endif
        if (k <= VV_PART)
        {
            vvPassSmall(k, off, runOff, points, stepBase + s, lane);
        }
        else
        {
            vvPass(k, off, runOff, points, stepBase + s, lane);
        }
    }
    for (uint m = 0u; m < k; m++) contactMemberEnd(vvMember(off, m), lane);
}
