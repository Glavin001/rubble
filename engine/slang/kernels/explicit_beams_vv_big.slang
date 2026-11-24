// --- The larger groups' pass with the grid (PERF_VV_BIG_GRID) ---
// vvPass's owner va (the whole workgroup, one owner at a time): its `cands` candidates
// (the run at gP.vvRun + runOff) counted into every bucket of gsCount (free in a group's
// frame, as vvGridBuild uses it), scanned and copied in bucket order to the group's sort
// records; gsCount[GRID_BUCKETS]: the fastest candidate's speed. Every lane (uniform).
// The larger groups' grid cell: 0.75 m (any size gives the same candidates; a contact
// triangle's padded box, a metre or two across, must stay within GRID_MAX_CELLS cells or
// it visits every candidate, which the self-collision cell's size does not allow).
float vvBigCell(uint va) { return max(vvGridCell(va), 0.75); }
void vvBigGridBuild(uint va, uint runOff, uint cands, uint lane)
{
    for (uint b = lane; b < GRID_BUCKETS; b += LANES) gsCount[b].store(0u);
    if (lane == 0u) gsCount[GRID_BUCKETS].store(0u);
    sync();
    float cell = vvBigCell(va);
    uint rank = vvRankAt(runOff, 0u, 0u, 0u);
    for (uint e = lane, eStop = EB_END(224, e, cands, LANES); e < eStop; e += LANES)
    {
        float4 E = gBeams[gP.vvRun + runOff + e];
        uint h = asuint(E.w);
        uint b = gridBucket(int3(floor(E.xyz / cell)));
        gNext[rank + e] = b | (gsCount[b].add(1u) << 10);
        float3 v = gNodes[h * 3u + 1u].xyz;
        uint sp = asuint(ssqrt(sdot(v, v)));
        if (sp > gsCount[GRID_BUCKETS].load()) gsCount[GRID_BUCKETS].max(sp);
    }
    sync();
    uint per = GRID_BUCKETS / LANES;
    uint block = LANES / GRID_BLOCKS;
    uint sum = 0u;
    for (uint k = 0u, kStop = EB_END(225, k, per, 1u); k < kStop; k++) sum += gsCount[lane * per + k].load();
    gsPart[lane] = sum;
    sync();
    if (lane < GRID_BLOCKS)
    {
        uint run = 0u;
        for (uint k = 0u, kStop = EB_END(226, k, block, 1u); k < kStop; k++)
        {
            uint x = gsPart[lane * block + k];
            gsPart[lane * block + k] = run;
            run += x;
        }
        gsBlk[lane] = run;
    }
    sync();
    if (lane == 0u)
    {
        uint run = 0u;
        for (uint k = 0u; k < GRID_BLOCKS; k++)
        {
            uint x = gsBlk[k];
            gsBlk[k] = run;
            run += x;
        }
        gsOff[GRID_BUCKETS] = run;
    }
    sync();
    {
        uint run = gsPart[lane] + gsBlk[lane / block];
        for (uint k = 0u, kStop = EB_END(227, k, per, 1u); k < kStop; k++)
        {
            gsOff[lane * per + k] = run;
            run += gsCount[lane * per + k].load();
        }
    }
    sync();
    uint sorted = vvSortAt(runOff);
    for (uint e = lane, eStop = EB_END(228, e, cands, LANES); e < eStop; e += LANES)
    {
        uint br = gNext[rank + e];
        gBeams[sorted + gsOff[br & (GRID_BUCKETS - 1u)] + (br >> 10)] = gBeams[gP.vvRun + runOff + e];
    }
    sync();
}

// vvTriangleCands from the grid: the candidates of the cells the triangle's box covers,
// the box padded by the most vvInBox pads it (the owner's fastest candidate), each bucket
// once: every candidate that passes the box test is in a covered cell, so the same
// candidates pass it and the same hits are found (vvApplyHits applies them in the
// canonical order whatever order they were found in). A box over more than GRID_MAX_CELLS
// cells: every candidate, as vvTriangleCands.
void vvTriangleCandsBigGrid(uint va, uint gt, uint sub, uint subs, uint runOff, uint points, uint cands, float r,
                            uint parity, uint q)
{
    float cell = vvBigCell(va);
    VvTri t = vvTriangle(gt, vvRel(va, va), r);
    float vmax = asfloat(gsCount[GRID_BUCKETS].load());
    float pad = nf(nf(t.travel + smul(vmax, STEP)) + CLIP_DEPTH);
    float3 lo = ssub3(t.lo, float3(pad, pad, pad));
    float3 hi = sadd3(t.hi, float3(pad, pad, pad));
    if (any(t.lo > t.hi))
    {
        // (A dead triangle's empty box: no candidate in it; its mark as vvTriangleCands
        // leaves it.)
        vvMark(gt, false, false);
        return;
    }
    float3 f0 = floor(lo / cell);
    float3 fs = floor(hi / cell) - f0 + float3(1.0, 1.0, 1.0);
    bool every = !(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS))
                 || !(all(abs(f0) < 1.0e8));
    if (every)
    {
        // (The run itself, unsorted: the sort records hold the sorted copy.)
        vvTriangleCands(va, gt, sub, subs, runOff, points, cands, r, parity, q);
        return;
    }
    bool cand = false;
    bool hit = false;
    int3 c0 = int3(f0);
    int3 span = int3(fs);
    uint yz = uint(span.y * span.z);
    uint cells = uint(span.x) * yz;
    uint sorted = vvSortAt(runOff);
    for (uint i = sub, iStop = EB_END(229, i, cells, subs); i < iStop; i += subs)
    {
        uint b = gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z))));
        bool dup = false;
        int3 cj = int3(0, 0, 0);
        for (uint j = 0u, jStop = EB_END(230, j, i, 1u); j < jStop; j++)
        {
            if (gridBucket(c0 + cj) == b)
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
        uint e0 = gsOff[b];
        uint e1 = gsOff[b + 1u];
        for (uint e = e0, eGuard = 0u; e < e1; e++)
        {
            EB_PASS(231, eGuard);
            float4 E = gBeams[sorted + e];
            uint h = asuint(E.w);
            uint vh = gStatic[gP.nodeInfoBase + h].w;
            bool anti = vvAnti(t, vh);
            if (!vvInBox(t, E.xyz, h, anti)) continue;
            cand = true;
            if (!vvFind(t, gt, h, E.xyz, vh, r, anti)) continue;
            hit = true;
            vvSplitCount(t, h, 0u);
            uint n = gsVvHits[q].add(1u);
            if (n < 2u * points)
            {
                gNext[vvHitAt(runOff, n)] = gt;
                gNext[vvHitAt(runOff, n) + 1u] = h;
            }
            else
            {
                vvTry(t, gt, h, E.xyz, vh, r, parity, 0u);
            }
        }
    }
    vvMark(gt, cand, hit);
}

// --- The triangle tests as a pair list (lossless) ---
// The triangles' tests ran on the lane that walked the triangle: a few triangles in the
// crush hold most candidates, and the step waited for their lanes. With the pair list
// (a region of its own per owner, vvTriPairsAt: the member's self-collision pair region,
// triPairsAt, stays the member's),
// the walk only lists the (triangle, candidate) pairs its box test passes, and after a
// barrier the lanes test the pairs, one each: the same tests, the same hits (applied in
// the canonical order), the same marks. An owner whose list overflows TRI_PAIR_CAP tests
// inline (vvTriangleCandsGrid) instead.
uint vvTriPairsAt(uint v) { return gP.vvTriPairs + v * 2u * TRI_PAIR_CAP; }
static const uint VV_PAIR_RUN = 0x80000000u; // a candidate's index in its owner's run (else the sorted copy)
void vvTriangleCandsList(uint a, uint va, uint gt, uint sub, uint subs, uint runOff, uint points, uint cands, float r)
{
    float cell = vvGridCell(va);
    VvTri t = vvTriangle(gt, vvRel(va, va), r);
    if (any(t.lo > t.hi)) return;
    float vmax = vvSpeedAt(a);
    float pad = nf(nf(t.travel + smul(vmax, STEP)) + CLIP_DEPTH);
    float3 lo = ssub3(t.lo, float3(pad, pad, pad));
    float3 hi = sadd3(t.hi, float3(pad, pad, pad));
    float3 f0 = floor(lo / cell);
    float3 fs = floor(hi / cell) - f0 + float3(1.0, 1.0, 1.0);
    bool every = !(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS))
                 || !(all(abs(f0) < 1.0e8));
    uint at = vvTriPairsAt(va);
    bool cand = false;
    if (every)
    {
        uint run = vvRunSmall(runOff, points, a);
        vvVisit(sub < cands ? (cands - sub + subs - 1u) / subs : 0u);
        for (uint e = sub, eStop = EB_END(232, e, cands, subs); e < eStop; e += subs)
        {
            float4 E = gBeams[run + e];
            uint h = asuint(E.w);
            if (!vvInBox(t, E.xyz, h, vvAnti(t, gStatic[gP.nodeInfoBase + h].w))) continue;
            cand = true;
            uint n = vvPairAdd(a);
            if (n < TRI_PAIR_CAP)
            {
                gNext[at + 2u * n] = gt;
                gNext[at + 2u * n + 1u] = e | VV_PAIR_RUN;
            }
        }
        vvMark(gt, cand, false);
        return;
    }
    int3 c0 = int3(f0);
    int3 span = int3(fs);
    uint yz = uint(span.y * span.z);
    uint cells = uint(span.x) * yz;
    uint sorted = vvSortAt(runOff);
    uint seen = 0u;
    for (uint i = sub, iStop = EB_END(233, i, cells, subs); i < iStop; i += subs)
    {
        uint b = vvGridBucket(a, c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z))));
        bool dup = false;
        int3 cj = int3(0, 0, 0);
        for (uint j = 0u, jStop = EB_END(234, j, i, 1u); j < jStop; j++)
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
            EB_PASS(235, eGuard);
            float4 E = gBeams[sorted + e];
            uint h = asuint(E.w);
            if (!vvInBox(t, E.xyz, h, vvAnti(t, gStatic[gP.nodeInfoBase + h].w))) continue;
            cand = true;
            uint n = vvPairAdd(a);
            if (n < TRI_PAIR_CAP)
            {
                gNext[at + 2u * n] = gt;
                gNext[at + 2u * n + 1u] = e;
            }
        }
    }
    vvVisit(seen);
    vvMark(gt, cand, false);
}

// A listed pair's test (vvTriangleCandsSmall's after its box test).
void vvPairTest(uint a, uint va, uint gt, uint idx, uint runOff, uint points, float r)
{
    VvTri t = vvTriangle(gt, vvRel(va, va), r);
    float4 E = (idx & VV_PAIR_RUN) != 0u ? gBeams[vvRunSmall(runOff, points, a) + (idx & ~VV_PAIR_RUN)]
                                          : gBeams[vvSortAt(runOff) + idx];
    uint h = asuint(E.w);
    uint vh = gStatic[gP.nodeInfoBase + h].w;
    bool anti = vvAnti(t, vh);
    if (!vvFind(t, gt, h, E.xyz, vh, r, anti)) return;
    vvSplitCount(t, h, a);
    uint n = vvHitAdd(a);
    if (n < 2u * points)
    {
        gNext[vvHitSmall(runOff, points, a, n)] = gt;
        gNext[vvHitSmall(runOff, points, a, n) + 1u] = h;
    }
    vvMark(gt, false, true);
}



// --- A level's listed hits bucketed by contact point (lossless) ---
// vvApplyOrdered found each hit's place in its point's chain by scanning every hit of
// the owner, twice per hit (O(n²) a level). The hits of one contact point are now found
// in their point's bucket (a counting sort of the hit indices by point, owner a's buckets
// a·256 …, in gsCount/gsOff: free in the levels, the grid's last use was the tests): the
// same chain, the same order, the same results.
uint vvOrderAt(uint runOff, uint points, uint a, uint i) { return gP.vvHits + 5u * vvRoomCp() + 4u * VV_PART * runOff + 2u * a * points + i; }
uint vvHitBucket(uint a, uint h) { return a * (GRID_BUCKETS / VV_PART) + (((h * 2654435761u) >> 22) & (GRID_BUCKETS / VV_PART - 1u)); }

// Every lane (uniform): the hit lists of level l's owners bucketed by point.
void vvHitBuckets(uint k, uint l, uint off, uint runOff, uint points, uint lane)
{
    for (uint b = lane; b < GRID_BUCKETS; b += LANES) gsCount[b].store(0u);
    sync();
    for (uint a = 0u, aStop = EB_END(236, a, k, 1u); a < aStop; a++)
    {
        uint n = gsVvHit4[a].load();
        if (vvLv(a) != l || n > vvHitRoom(a, points)) continue;
        for (uint i = lane, iStop = EB_END(237, i, n, LANES); i < iStop; i += LANES)
        {
            uint b = vvHitBucket(a, gNext[vvHitSmall(runOff, points, a, i) + 1u]);
            gNext[vvOrderAt(runOff, points, a, i)] = b | (gsCount[b].add(1u) << 10);
        }
    }
    sync();
    uint per = GRID_BUCKETS / LANES;
    uint block = LANES / GRID_BLOCKS;
    uint sum = 0u;
    for (uint q = 0u, qStop = EB_END(238, q, per, 1u); q < qStop; q++) sum += gsCount[lane * per + q].load();
    gsPart[lane] = sum;
    sync();
    if (lane < GRID_BLOCKS)
    {
        uint run = 0u;
        for (uint q = 0u, qStop = EB_END(239, q, block, 1u); q < qStop; q++)
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
        for (uint q = 0u, qStop = EB_END(240, q, per, 1u); q < qStop; q++)
        {
            gsOff[lane * per + q] = run;
            run += gsCount[lane * per + q].load();
        }
    }
    sync();
    // The sorted hit indices after the ranks (2·points words an owner, four owners).
    for (uint a = 0u, aStop = EB_END(241, a, k, 1u); a < aStop; a++)
    {
        uint n = gsVvHit4[a].load();
        if (vvLv(a) != l || n > vvHitRoom(a, points)) continue;
        for (uint i = lane, iStop = EB_END(242, i, n, LANES); i < iStop; i += LANES)
        {
            uint br = gNext[vvOrderAt(runOff, points, a, i)];
            gNext[vvOrderAt(runOff, 0u, 0u, 0u) + 2u * VV_PART * points + gsOff[br & (GRID_BUCKETS - 1u)] + (br >> 10)] = i;
        }
    }
    sync();
}


// vvPassSmall in three parts (the split group path, contact-groups-design.md, runs the
// tests across workgroups between the first and the last): the scans, candidates and
// their grid (returns 0 when nothing is offered, else bit 0, with bit 1 the grid); the
// triangle tests; the hits' levels.
uint vvPassSmallFront(uint k, uint off, uint runOff, uint points, uint parity, uint lane);
void vvPassSmallTests(uint k, uint off, uint runOff, uint points, uint parity, uint lane, bool vvGrid);
void vvPassSmallLevels(uint k, uint off, uint runOff, uint points, uint parity, uint lane, bool vvGrid);
void vvPassSmall(uint k, uint off, uint runOff, uint points, uint step, uint lane)
{
    uint parity = step & 1u;
    uint front = vvPassSmallFront(k, off, runOff, points, parity, lane);
    if (front == 0u) return;
    bool vvGrid = (front & 2u) != 0u;
    vvPassSmallTests(k, off, runOff, points, parity, lane, vvGrid);
    gVvTag = (gP.perfFlags & PERF_VV_BEAM_SUM) != 0u ? ((step + 1u) | 0x80000000u) : 0u;
    vvPassSmallLevels(k, off, runOff, points, parity, lane, vvGrid);
    gVvTag = 0u;
}

uint vvPassSmallFront(uint k, uint off, uint runOff, uint points, uint parity, uint lane)
{
    float3 m3 = float3(VV_BOX_MARGIN, VV_BOX_MARGIN, VV_BOX_MARGIN);
    // (S) the partner scans: lane per ordered pair (a, b).
    for (uint p = lane, pStop = EB_END(247, p, k * k, LANES); p < pStop; p += LANES)
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
    // No owner offered a partner this step and none resets (no frame boxes
    // overlap: a group whose members have not met yet): the phases below would list
    // nothing, test nothing and hit nothing (every throttle and count untouched), so
    // they are skipped with their barriers. Lane 0 decides after the scans' barrier
    // (gsUniformOpen is free between the member step and the next one).
    if (lane == 0u)
    {
        uint offered = 0u;
        // (An owner's reset or fix still clears its throttles below, offered or not.)
        for (uint a = 0u, aStop = EB_END(248, a, k, 1u); a < aStop; a++)
            offered += gsVvOff4[a].load() + gsVvReset4[a] + gsVvFix[a];
        gsUniformOpen = offered;
    }
    sync();
    if (gsUniformOpen == 0u) return 0u;
    // (The two-body pass's node forces at its start: taken by the levels, for the
    // listed hits' nodes, before any hit is applied: vvPassSmallLevels.)
    // (P) every owner's throttle step and candidates.
    {
        uint triTotal = 0u;
        for (uint a = 0u, aStop = EB_END(249, a, k, 1u); a < aStop; a++) triTotal += gStatic[vvMember(off, a) * 4u + 1u].y;
        for (uint p = lane, pStop = EB_END(250, p, triTotal, LANES); p < pStop; p += LANES)
        {
            uint a = 0u;
            uint t = p;
            for (uint c = 0u, cGuard = 0u; c + 1u < k; c++)
            {
                EB_PASS(251, cGuard);
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
            float vmaxB = 0.0;
            for (uint m = 0u, mStop = EB_END(252, m, k, 1u); m < mStop; m++)
            {
                if ((gsVvPair[a * VV_PART + m] & 1u) == 0u) continue;
                uint kb = vvBoxKey(vvMember(off, m), parity);
                plo = min(plo, float3(vvUnkey(gAcc[kb].load()), vvUnkey(gAcc[kb + 1u].load()), vvUnkey(gAcc[kb + 2u].load())));
                phi = max(phi, float3(vvUnkey(gAcc[kb + 3u].load()), vvUnkey(gAcc[kb + 4u].load()), vvUnkey(gAcc[kb + 5u].load())));
                if (gP.clipMem != 0u) vmaxB = max(vmaxB, clipVmax(vvMember(off, m), parity));
            }
            if (!vvNearBox(gt, va, range, plo, phi, vvClipExtra(va, vmaxB, parity)))
            {
                gMisc[gP.vvTri + gt] = uint4(fix ? 0u : asuint(min(cwait + 1, throttleCap(gt))), 0u, 0u, 0u);
                continue;
            }
            gMisc[gP.vvTri + gt] = uint4(asuint(cwait), 0u, 0u, 0u);
            gNext[vvTestedAt(ra1.x + gsVvList4[a].add(1u))] = t;
        }
        // Candidates: each owner's partners' contact points in its box grown by
        // its range; the owners side by side, vvPartLanes(k) lanes each (the same
        // candidates per owner; their order in its run follows an atomic counter
        // either way, and the hits apply in a canonical order, vvApplyOrdered).
        uint cP = vvPartLanes(k);
        uint cl = lane % cP;
        for (uint a = lane / cP, aStop = EB_END(253, a, min(lane / cP + 1u, k), 1u); a < aStop; a++)
        {
#ifdef EB_ABLATE
            // Measurement only: 262144 the candidates.
            if ((gP.ablate & 262144u) != 0u) break;
#endif
            if (gsVvOff4[a].load() == 0u) continue;
            uint va = vvMember(off, a);
            uint key = vvBoxKey(va, parity);
            float range = asfloat(gStatic[va * 4u + 3u].z);
            float3 r3 = float3(range, range, range);
            float3 clo = ssub3(float3(vvUnkey(gAcc[key].load()), vvUnkey(gAcc[key + 1u].load()), vvUnkey(gAcc[key + 2u].load())), r3);
            float3 chi = sadd3(float3(vvUnkey(gAcc[key + 3u].load()), vvUnkey(gAcc[key + 4u].load()), vvUnkey(gAcc[key + 5u].load())), r3);
            float ownerReach = gP.clipMem != 0u ? CLIP_DEPTH + clipVmax(va, parity) * STEP + 0.001 : 0.0;
            uint run = vvRunSmall(runOff, points, a);
            for (uint m = 0u, mStop = EB_END(254, m, k, 1u); m < mStop; m++)
            {
                if ((gsVvPair[a * VV_PART + m] & 1u) == 0u) continue;
                uint v = vvMember(off, m);
                uint4 cp = gStatic[gP.vvStatic + v];
                float3 o = asfloat(gMisc[v].xyz);
                float3 orel = vvRel(v, va);
                for (uint i = cl, iStop = EB_END(255, i, cp.y, cP); i < iStop; i += cP)
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
                    gBeams[run + gsVvCand4[a].add(1u)] = float4(nf3(orel + gNodes[h * 3u].xyz), asfloat(h));
                }
            }
        }
    }
    sync();
    // The candidates' grid (lane 0's decision through gsUniformOpen, free here).
    if (lane == 0u) gsUniformOpen = gAcc[vvCtl(VV_GRID_OFF)].load() == 0 ? 1u : 0u;
    sync();
    bool vvGrid = gsUniformOpen != 0u;
    if (vvGrid) vvGridBuild(k, off, runOff, points, lane);
    return vvGrid ? 3u : 1u;
}

void vvPassSmallTests(uint k, uint off, uint runOff, uint points, uint parity, uint lane, bool vvGrid)
{
    // The tests as a pair list (with the grid and the members' pair regions).
    bool vvList = vvGrid && gP.vvTriPairs != 0u;
    if (vvList)
    {
        if (lane < VV_PART) gsPairCountM[lane].store(0u);
        sync();
    }
    // (T) every owner's listed triangles against its candidates.
    {
        uint total = 0u;
        for (uint a = 0u, aStop = EB_END(256, a, k, 1u); a < aStop; a++) total += gsVvList4[a].load();
        uint subs = clamp(LANES / max(total, 1u), 1u, 64u);
#ifdef EB_ABLATE
        // Measurement only: 131072 the owners' triangles against their candidates.
        if ((gP.ablate & 131072u) != 0u) total = 0u;
#endif
        for (uint p = lane, pStop = EB_END(257, p, total * subs, LANES); p < pStop; p += LANES)
        {
            uint q = p / subs;
            uint a = 0u;
            for (uint c = 0u, cGuard = 0u; c + 1u < k; c++)
            {
                EB_PASS(258, cGuard);
                uint n = gsVvList4[a].load();
                if (q < n) break;
                q -= n;
                a++;
            }
            uint va = vvMember(off, a);
            uint4 ra1 = gStatic[va * 4u + 1u];
            uint t = gNext[vvTestedAt(ra1.x + q)];
            if (vvList)
                vvTriangleCandsList(a, va, ra1.x + t, p % subs, subs, runOff, points, gsVvCand4[a].load(),
                                    asfloat(gStatic[va * 4u + 3u].z));
            else if (vvGrid)
                vvTriangleCandsGrid(a, va, ra1.x + t, p % subs, subs, runOff, points, gsVvCand4[a].load(),
                                    asfloat(gStatic[va * 4u + 3u].z));
            else
                vvTriangleCandsSmall(a, va, ra1.x + t, p % subs, subs, runOff, points, gsVvCand4[a].load(),
                                     asfloat(gStatic[va * 4u + 3u].z));
        }
    }
    if (vvList)
    {
        sync();
        // The listed pairs, a lane each; an owner past the list's room tests inline.
        uint total = 0u;
        for (uint a = 0u, aStop = EB_END(259, a, k, 1u); a < aStop; a++)
        {
            uint n = gsPairCountM[a].load();
            total += n <= TRI_PAIR_CAP ? n : 0u;
        }
        for (uint p = lane, pStop = EB_END(260, p, total, LANES); p < pStop; p += LANES)
        {
            uint q = p;
            uint a = 0u;
            for (uint c = 0u, cGuard = 0u; c < k; c++)
            {
                EB_PASS(261, cGuard);
                uint n = gsPairCountM[c].load();
                n = n <= TRI_PAIR_CAP ? n : 0u;
                if (q < n)
                {
                    a = c;
                    break;
                }
                q -= n;
            }
            uint va = vvMember(off, a);
            uint at = vvTriPairsAt(va);
            vvPairTest(a, va, gNext[at + 2u * q], gNext[at + 2u * q + 1u], runOff, points, asfloat(gStatic[va * 4u + 3u].z));
        }
        uint over = 0u;
        for (uint a = 0u, aStop = EB_END(262, a, k, 1u); a < aStop; a++)
            over += gsPairCountM[a].load() > TRI_PAIR_CAP ? gsVvList4[a].load() : 0u;
        uint subs = clamp(LANES / max(over, 1u), 1u, 64u);
        for (uint p = lane, pStop = EB_END(263, p, over * subs, LANES); p < pStop; p += LANES)
        {
            uint q = p / subs;
            uint a = 0u;
            for (uint c = 0u, cGuard = 0u; c < k; c++)
            {
                EB_PASS(264, cGuard);
                uint n = gsPairCountM[c].load() > TRI_PAIR_CAP ? gsVvList4[c].load() : 0u;
                if (q < n)
                {
                    a = c;
                    break;
                }
                q -= n;
            }
            uint va = vvMember(off, a);
            uint4 ra1 = gStatic[va * 4u + 1u];
            uint t = gNext[vvTestedAt(ra1.x + q)];
            vvTriangleCandsGrid(a, va, ra1.x + t, p % subs, subs, runOff, points, gsVvCand4[a].load(),
                                asfloat(gStatic[va * 4u + 3u].z));
        }
    }
#ifdef EB_EDGE
    // Edge vehicle contact's hits (explicit_beams_vv_edges.slang; none without it).
    edgeTestsSmall(k, off, parity, lane);
#endif
}

// Node n's force at the pass's start into the snapshot; with a tag, its beams' sum kept
// beside it (vvBeamSumOf) and the snapshot tagged. (Several lanes may take one node: the
// same values.)
void vvSnapNode(uint n, uint vn, uint parity, uint tag)
{
    if (tag == 0u)
    {
        gBeams[gP.vvSnap + n] = float4(vvPointForce(n, vn, parity), 0.0);
        return;
    }
    uint4 info = gStatic[gP.nodeInfoBase + n];
    uint4 rb0 = gStatic[vn * 4u];
    uint4 rb2 = gStatic[vn * 4u + 2u];
    float3 bs = beamSum(gNodes[n * 3u + 2u].xyz, info, rb0.z, rb2.z, rb2.w, parity, false);
    float3 fh = sadd3(bs, accForce(n));
#ifdef EB_AERO
    uint slot1 = info.x >> NODE_SLOT_SHIFT;
    if (slot1 != 0u) fh = sadd3(fh, gBeams[gP.slideAcc + min(slot1 - 1u, max(gP.slotTotal, 1u) - 1u)].xyz);
#endif
    gBeams[gP.vvSnap + gP.totalNodes + n] = float4(bs, 0.0);
    gBeams[gP.vvSnap + n] = float4(fh, asfloat(tag));
}

