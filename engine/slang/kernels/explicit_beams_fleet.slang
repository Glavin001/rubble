// --- Vehicle contact's state by owner vehicle (engine/docs/measurements/
// contact-groups-design.md §8) ---
// The small groups' contact pass (vvPassSmall: the scans, the throttle steps, the
// candidates and their grid, the triangle tests, the levels) addresses an owner's state
// by its slot in its contact group: counters in workgroup arrays of VV_PART, its
// candidates and hits at the group's run offset plus the slot times the group's points,
// its grid an eighth of the workgroup's buckets, its partners a VV_PART² table. Nothing
// in the contact rule needs the group: here the same state is the owner vehicle's own
// (the addresses: explicit_beams.slang fl*):
//   counters                  a record a vehicle in gAcc (flOwn), zero between passes
//   partners this step        a word a side of each of the plan's overlapping pairs
//                             (flBitsAt), read through the vehicle's pairs (flAdjAt)
//   candidates, sorted copy,  fleet-wide regions (vvRun, vvHits), an owner's part at its
//   ranks, hits               prefix of the owners' offered counts (flRun)
//   candidates' grid          the owner's own GRID_BUCKETS counts (flGridCount) and
//                             offsets (vvScratchOff)
//   two-body point counts     the slot of the owner's rank among the point's vehicle's
//                             two-body partners of the step (flSlotAt)
// The leaf rules are vvPassSmall's own functions (the step scan's tests, a triangle's
// throttle step and near-box test, the candidate's box test, vvTriangleCandsList,
// vvPairTest, vvTry, the levels), which reach the state through the address helpers
// (gFl). So the same partners, listed triangles, candidates, pairs and hits, and the
// hits applied in the same order: the same results bit for bit.
// Limits (the contact status' overflow word, explicit_plan::contact_error; never a silent
// drop): the fleet-wide run holds VV_PART times the fleet's contact points (FL_ERR_RUN),
// an owner's hit list twice its offered count (FL_ERR_HITS), a node's two-body point
// counts VV_PART owners (FL_ERR_SLOTS).

void flError(int code) { gAcc[vvCtl(VV_OVERFLOW)].max(code); }

// (S) Pair p of the plan's overlapping pairs: this step's scan (vvPassSmallFront's, for
// both of its sides at once: the box test and the approach test are symmetric). Each
// side's partner bits, each vehicle's offered count, the resets.
void flScanPair(uint p, uint parity)
{
    // (A later copy of a pair the plan's list holds twice: vvAdjBuild.)
    if ((gNext[flBitsAt(2u * p)] & 4u) != 0u) return;
    float3 m3 = float3(VV_BOX_MARGIN, VV_BOX_MARGIN, VV_BOX_MARGIN);
    uint va = gNext[gP.vvPairs + 2u * p];
    uint vb = gNext[gP.vvPairs + 2u * p + 1u];
    uint ka = vvBoxKey(va, parity);
    uint kb = vvBoxKey(vb, parity);
    float3 alo = ssub3(float3(vvUnkey(gAcc[ka].load()), vvUnkey(gAcc[ka + 1u].load()), vvUnkey(gAcc[ka + 2u].load())), m3);
    float3 ahi = sadd3(float3(vvUnkey(gAcc[ka + 3u].load()), vvUnkey(gAcc[ka + 4u].load()), vvUnkey(gAcc[ka + 5u].load())), m3);
    float3 olo = ssub3(float3(vvUnkey(gAcc[kb].load()), vvUnkey(gAcc[kb + 1u].load()), vvUnkey(gAcc[kb + 2u].load())), m3);
    float3 ohi = sadd3(float3(vvUnkey(gAcc[kb + 3u].load()), vvUnkey(gAcc[kb + 4u].load()), vvUnkey(gAcc[kb + 5u].load())), m3);
    uint bits = 0u;
    if (vvOverlap(alo, ahi, olo, ohi))
    {
        bits = 1u;
        gAcc[flOwn(va) + FL_OFFERED].add(int(vvCpCount(vb)));
        gAcc[flOwn(vb) + FL_OFFERED].add(int(vvCpCount(va)));
        float3 d = ssub3(gNodes[gStatic[va * 4u].x * 3u + 1u].xyz, gNodes[gStatic[vb * 4u].x * 3u + 1u].xyz);
        if (sdot(d, d) > VV_APPROACH_SQ)
        {
            bits = 3u;
            gAcc[flOwn(va) + FL_RESET].store(1);
            gAcc[flOwn(vb) + FL_RESET].store(1);
            // (The earlier vehicle's reset by a later one: vvPassSmall's gsVvFix.)
            gAcc[flOwn(min(va, vb)) + FL_FIX].store(1);
            gMisc[vvRec(vb, VV_FLAGS)].w = 1u;
            gMisc[vvRec(va, VV_FLAGS)].w = 1u;
        }
    }
    gNext[flBitsAt(2u * p)] = bits;
    gNext[flBitsAt(2u * p + 1u)] = bits;
}

// After the scans: owner v's records of the step. Its partners' boxes' union and their
// fastest node (the throttle step's near-box test), its own box grown by its range and
// its anti-clip reach (the candidates' box test), and, for each partner with two-body
// contact, that partner's slot at v's points (v's partners in the order of its pairs).
void flOwnerPlan(uint v, uint parity)
{
    float3 plo = float3(3.0e38, 3.0e38, 3.0e38);
    float3 phi = -plo;
    float vmaxB = 0.0;
    uint rank = 0u;
    uint e1 = gNext[flAdjFirst(v + 1u)];
    for (uint e = gNext[flAdjFirst(v)], eStop = EB_END(e, e1, 1u); e < eStop; e++)
    {
        uint d = gNext[flAdjAt(e)];
        if ((gNext[flBitsAt(d)] & 1u) == 0u) continue;
        uint w = flPartner(d);
        uint kb = vvBoxKey(w, parity);
        plo = min(plo, float3(vvUnkey(gAcc[kb].load()), vvUnkey(gAcc[kb + 1u].load()), vvUnkey(gAcc[kb + 2u].load())));
        phi = max(phi, float3(vvUnkey(gAcc[kb + 3u].load()), vvUnkey(gAcc[kb + 4u].load()), vvUnkey(gAcc[kb + 5u].load())));
        if (gP.clipMem != 0u) vmaxB = max(vmaxB, clipVmax(w, parity));
        if (vvTwoBody(w, v))
        {
            gNext[flSlotAt(d ^ 1u)] = rank;
            if (rank >= VV_PART) flError(FL_ERR_SLOTS);
            rank++;
        }
    }
    uint key = vvBoxKey(v, parity);
    float range = asfloat(gStatic[v * 4u + 3u].z);
    float3 r3 = float3(range, range, range);
    float3 clo = ssub3(float3(vvUnkey(gAcc[key].load()), vvUnkey(gAcc[key + 1u].load()), vvUnkey(gAcc[key + 2u].load())), r3);
    float3 chi = sadd3(float3(vvUnkey(gAcc[key + 3u].load()), vvUnkey(gAcc[key + 4u].load()), vvUnkey(gAcc[key + 5u].load())), r3);
    float ownerReach = gP.clipMem != 0u ? CLIP_DEPTH + clipVmax(v, parity) * STEP + 0.001 : 0.0;
    gBeams[flBoxAt(v)] = float4(clo, ownerReach);
    gBeams[flBoxAt(v) + 1u] = float4(chi, 0.0);
    gBeams[flBoxAt(v) + 2u] = float4(plo, vmaxB);
    gBeams[flBoxAt(v) + 3u] = float4(phi, 0.0);
}

// (P) Triangle gt of owner va: its throttle step (vvPassSmallFront's), listed in the
// owner's tested list when it may have candidates.
void flTriangle(uint va, uint gt, uint parity)
{
    uint own = flOwn(va);
    uint offered = uint(gAcc[own + FL_OFFERED].load());
    bool ownerReset = gAcc[own + FL_RESET].load() != 0;
    bool fix = gAcc[own + FL_FIX].load() != 0;
    uint4 c = gMisc[gP.vvTri + gt];
    int cwait = ownerReset ? 0 : asint(c.x);
    int cidle = asint(c.y);
    if (offered == 0u)
    {
        if (ownerReset || fix) gMisc[gP.vvTri + gt].x = 0u;
        return;
    }
    if (cwait > 0)
    {
        gMisc[gP.vvTri + gt] = uint4(fix ? 0u : asuint(cwait - 1), asuint(cidle + 1), 0u, 0u);
        return;
    }
    cwait = VV_MAX_WAIT < cidle ? VV_MAX_WAIT : cidle;
    float range = asfloat(gStatic[va * 4u + 3u].z);
    float4 P0 = gBeams[flBoxAt(va) + 2u];
    float4 P1 = gBeams[flBoxAt(va) + 3u];
    if (!vvNearBox(gt, va, range, P0.xyz, P1.xyz, vvClipExtra(va, P0.w, parity)))
    {
        gMisc[gP.vvTri + gt] = uint4(fix ? 0u : asuint(min(cwait + 1, throttleCap(gt))), 0u, 0u, 0u);
        return;
    }
    gMisc[gP.vvTri + gt] = uint4(asuint(cwait), 0u, 0u, 0u);
    uint triOff = gStatic[va * 4u + 1u].x;
    gNext[vvTestedAt(triOff + uint(gAcc[own + FL_LISTED].add(1)))] = gt - triOff;
}

// An owner's part of the fleet-wide candidate ranks (as its run: at its prefix).
uint flRankAt(uint va) { return gP.vvHits + 4u * VV_PART * vvTotalCp() + flRun(va); }
// Its sorted copy and its own grid's offsets.
uint flSortAt(uint va) { return gP.vvRun + VV_PART * vvTotalCp() + flRun(va); }
uint flOffAt(uint va) { return gP.vvScratchOff + va * (GRID_BUCKETS + 1u); }

// Contact point i of vehicle vb: into the run of each owner it is a partner of this step
// whose box (grown by its range and reach) holds it (vvPassSmallFront's candidates), with
// its bucket in that owner's grid and its rank there (vvGridBuild's counts), and the
// owner's fastest candidate.
void flPoint(uint vb, uint i)
{
    uint4 cp = gStatic[gP.vvStatic + vb];
    uint h = gLists[cp.x + i];
    float3 o = asfloat(gMisc[vb].xyz);
    float3 x = nf3(o + gNodes[h * 3u].xyz);
    float3 vh3 = gNodes[h * 3u + 1u].xyz;
    uint e1 = gNext[flAdjFirst(vb + 1u)];
    for (uint e = gNext[flAdjFirst(vb)], eStop = EB_END(e, e1, 1u); e < eStop; e++)
    {
        uint d = gNext[flAdjAt(e)];
        if ((gNext[flBitsAt(d)] & 1u) == 0u) continue;
        uint va = flPartner(d);
        float4 B0 = gBeams[flBoxAt(va)];
        float4 B1 = gBeams[flBoxAt(va) + 1u];
        float ex = 0.0;
        if (gP.clipMem != 0u) ex = B0.w + sqrt(dot(vh3, vh3)) * STEP;
        float3 e3 = float3(ex, ex, ex);
        if (any(x < B0.xyz - e3) || any(x > B1.xyz + e3)) continue;
        uint own = flOwn(va);
        uint base = flRun(va);
        // (An owner past the run's room has no part: FL_ERR_RUN.)
        if (base == 0xFFFFFFFFu) continue;
        uint ci = uint(gAcc[own + FL_CANDS].add(1));
        float4 E = float4(nf3(vvRel(vb, va) + gNodes[h * 3u].xyz), asfloat(h));
        gBeams[gP.vvRun + base + ci] = E;
        uint b = gridBucket(int3(floor(E.xyz / vvGridCell(va))));
        gNext[flRankAt(va) + ci] = b | (uint(gAcc[flGridCount(va) + b].add(1)) << 10);
        gAcc[own + FL_FAST].max(int(asuint(ssqrt(sdot(vh3, vh3)))));
    }
}

// Owner va's grid: its buckets' counts to offsets (vvGridBuild's scan, over the owner's
// own GRID_BUCKETS buckets), the counts back to zero. Every lane of a workgroup, uniform;
// four barriers, the first before any lane reads a count.
void flOffsets(uint va, uint lane)
{
    uint cnt = flGridCount(va);
    uint off = flOffAt(va);
    uint per = GRID_BUCKETS / LANES;
    uint block = LANES / GRID_BLOCKS;
    sync();
    uint sum = 0u;
    for (uint q = 0u, qStop = EB_END(q, per, 1u); q < qStop; q++) sum += uint(gAcc[cnt + lane * per + q].load());
    gsPart[lane] = sum;
    sync();
    if (lane < GRID_BLOCKS)
    {
        uint run = 0u;
        for (uint q = 0u, qStop = EB_END(q, block, 1u); q < qStop; q++)
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
        gNext[off + GRID_BUCKETS] = run;
    }
    sync();
    {
        uint run = gsPart[lane] + gsBlk[lane / block];
        for (uint q = 0u, qStop = EB_END(q, per, 1u); q < qStop; q++)
        {
            gNext[off + lane * per + q] = run;
            run += uint(gAcc[cnt + lane * per + q].load());
            gAcc[cnt + lane * per + q].store(0);
        }
    }
}

// Candidate e of owner va's run to its place in the sorted copy (after flOffsets).
void flSortCand(uint va, uint e)
{
    uint br = gNext[flRankAt(va) + e];
    gBeams[flSortAt(va) + gNext[flOffAt(va) + (br & (GRID_BUCKETS - 1u))] + (br >> 10)] = gBeams[gP.vvRun + flRun(va) + e];
}

// An owner's candidates: what the gather counted, within its part.
uint flCands(uint va)
{
    uint own = flOwn(va);
    return min(uint(gAcc[own + FL_CANDS].load()), uint(gAcc[own + FL_OFFERED].load()));
}

// (T) Part `sub` of `subs` of owner va's listed triangle q: its box's cells, its pairs
// listed (vvTriangleCandsList).
void flWalk(uint va, uint q, uint sub, uint subs)
{
    gFl = 1u;
    gFlV = va;
    uint triOff = gStatic[va * 4u + 1u].x;
    uint t = gNext[vvTestedAt(triOff + q)];
    vvTriangleCandsList(0u, va, triOff + t, sub, subs, 0u, uint(gAcc[flOwn(va) + FL_OFFERED].load()), flCands(va),
                        asfloat(gStatic[va * 4u + 3u].z));
    gFl = 0u;
}
// The same tested inline (an owner past its pair list's room: vvTriangleCandsGrid).
void flWalkInline(uint va, uint q, uint sub, uint subs)
{
    gFl = 1u;
    gFlV = va;
    uint triOff = gStatic[va * 4u + 1u].x;
    uint t = gNext[vvTestedAt(triOff + q)];
    vvTriangleCandsGrid(0u, va, triOff + t, sub, subs, 0u, uint(gAcc[flOwn(va) + FL_OFFERED].load()), flCands(va),
                        asfloat(gStatic[va * 4u + 3u].z));
    gFl = 0u;
}
// Owner va's listed pair q tested (vvPairTest).
void flPairTest(uint va, uint q)
{
    gFl = 1u;
    gFlV = va;
    uint at = vvTriPairsAt(va);
    vvPairTest(0u, va, gNext[at + 2u * q], gNext[at + 2u * q + 1u], 0u, uint(gAcc[flOwn(va) + FL_OFFERED].load()),
               asfloat(gStatic[va * 4u + 3u].z));
    gFl = 0u;
}

// The pass's end for owner v: an overflowed hit list reported, its last pass's counts
// kept for the host's measurement (contact_group_counts), its record back to zero.
void flOwnerEnd(uint v)
{
    uint own = flOwn(v);
    uint hits = uint(gAcc[own + FL_HITS].load());
    if (hits > 2u * uint(gAcc[own + FL_OFFERED].load())) flError(FL_ERR_HITS);
    gNext[flVehAt(1u, v)] = uint(gAcc[own + FL_PAIRS].load());
    gNext[flVehAt(2u, v)] = hits;
    gNext[flVehAt(3u, v)] = uint(gAcc[own + FL_LISTED].load());
    gNext[flVehAt(4u, v)] = uint(gAcc[own + FL_CANDS].load());
    for (uint w = 0u; w < FL_OWN; w++) gAcc[own + w].store(0);
}

// --- Stage 1: the state by vehicle under the contact groups' own dispatches ---

// Group (k members at `off`, its part of the fleet-wide regions at VV_PART · runOff)'s
// front in its workgroup: vvPassSmallFront with every owner's state its vehicle's.
// Returns 0 when no owner is offered a partner, resets or fixes (uniform), else 3.
uint fleetGroupFront(uint g, uint k, uint off, uint runOff, uint parity, uint lane)
{
    // (S) the plan's pairs of this group's vehicles.
    uint pairs = gNext[flAdjFirst(gP.vehicles)] / 2u;
    for (uint p = lane, pStop = EB_END(p, pairs, LANES); p < pStop; p += LANES)
    {
        uint4 fl = gMisc[vvRec(gNext[gP.vvPairs + 2u * p], VV_FLAGS)];
        if (fl.y == 0u || fl.z != g) continue;
        flScanPair(p, parity);
    }
    sync();
    // Each owner's part of the run (the prefix of the offered counts, from the group's
    // part) and its records of the step.
    if (lane == 0u)
    {
        uint run = VV_PART * runOff;
        uint open = 0u;
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++)
        {
            uint own = flOwn(vvMember(off, m));
            uint offered = uint(gAcc[own + FL_OFFERED].load());
            gNext[flPrefixAt(0u, vvMember(off, m))] = run;
            run += offered;
            open += offered + uint(gAcc[own + FL_RESET].load()) + uint(gAcc[own + FL_FIX].load());
        }
        gsUniformOpen = open;
    }
    for (uint m = lane, mStop = EB_END(m, k, LANES); m < mStop; m += LANES) flOwnerPlan(vvMember(off, m), parity);
    sync();
    if (gsUniformOpen == 0u) return 0u;
    // (P) every owner's throttle step and candidates.
    {
        uint triTotal = 0u;
        for (uint a = 0u, aStop = EB_END(a, k, 1u); a < aStop; a++) triTotal += gStatic[vvMember(off, a) * 4u + 1u].y;
        for (uint p = lane, pStop = EB_END(p, triTotal, LANES); p < pStop; p += LANES)
        {
            uint a = 0u;
            uint t = p;
            for (uint c = 0u, cGuard = 0u; c + 1u < k; c++)
            {
                EB_PASS(cGuard);
                uint n = gStatic[vvMember(off, a) * 4u + 1u].y;
                if (t < n) break;
                t -= n;
                a++;
            }
            uint va = vvMember(off, a);
            flTriangle(va, gStatic[va * 4u + 1u].x + t, parity);
        }
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++)
        {
            uint vb = vvMember(off, m);
            uint n = vvCpCount(vb);
            for (uint i = lane, iStop = EB_END(i, n, LANES); i < iStop; i += LANES) flPoint(vb, i);
        }
    }
    // Each owner's grid, then its candidates' sorted copy.
    for (uint a = 0u, aStop = EB_END_UNIFORM(a, k, 1u, EB_MAX_MEMBERS); a < aStop; a++) flOffsets(vvMember(off, a), lane);
    sync();
    for (uint a = 0u, aStop = EB_END(a, k, 1u); a < aStop; a++)
    {
        uint va = vvMember(off, a);
        uint cands = flCands(va);
        for (uint e = lane, eStop = EB_END(e, cands, LANES); e < eStop; e += LANES) flSortCand(va, e);
    }
    sync();
    return 3u;
}

// The levels' inputs from the owners' records into the group's workgroup arrays (lane 0,
// before a barrier): vvPassSmallLevels then runs as in the group's own pass, its hit lists
// through the address helpers (gFl 2).
void flImport(uint off, uint k)
{
    for (uint a = 0u, aStop = EB_END(a, k, 1u); a < aStop; a++)
    {
        uint v = vvMember(off, a);
        uint own = flOwn(v);
        gsVvOff4[a].store(uint(gAcc[own + FL_OFFERED].load()));
        gsVvList4[a].store(uint(gAcc[own + FL_LISTED].load()));
        gsVvCand4[a].store(flCands(v));
        gsVvHit4[a].store(uint(gAcc[own + FL_HITS].load()));
        gsVvReset4[a] = uint(gAcc[own + FL_RESET].load());
        gsVvFix[a] = uint(gAcc[own + FL_FIX].load());
        uint e1 = gNext[flAdjFirst(v + 1u)];
        for (uint e = gNext[flAdjFirst(v)], eStop = EB_END(e, e1, 1u); e < eStop; e++)
        {
            uint d = gNext[flAdjAt(e)];
            uint m = gMisc[vvRec(flPartner(d), VV_MEMBER)].x;
            uint bits = gNext[flBitsAt(d)];
            if (m < VV_PART && (bits & 4u) == 0u) gsVvPair[a * VV_PART + m] = bits;
        }
    }
}

// The group's levels after the tests' dispatches, then the pass's end for its owners.
void fleetGroupLevels(uint k, uint off, uint runOff, uint points, uint parity, uint lane, uint tag)
{
    uint open = 0u;
    if (lane == 0u)
    {
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++)
        {
            uint own = flOwn(vvMember(off, m));
            open += uint(gAcc[own + FL_OFFERED].load()) + uint(gAcc[own + FL_RESET].load()) + uint(gAcc[own + FL_FIX].load());
        }
    }
    open = splitUniform(lane, open);
    if (open != 0u)
    {
        if (lane == 0u)
        {
            vvClearCounters();
            vvClearSmall();
            flImport(off, k);
            gsUniformHits = 0u;
            gsUniformOpen = 0u;
        }
        sync();
        gFl = 2u;
        gFlOff = off;
        gVvTag = tag;
        vvPassSmallLevels(k, off, runOff, points, parity, lane, true);
        gVvTag = 0u;
        gFl = 0u;
    }
    sync();
    for (uint m = lane, mStop = EB_END(m, k, LANES); m < mStop; m += LANES) flOwnerEnd(vvMember(off, m));
}

#ifdef EB_VV_GLOBAL
// Stage 1's tests across workgroups (splitTests with every owner's state its vehicle's):
// part `part` of group g's walk, or of its listed pairs' tests (SPLIT_LEVELS).
void fleetTests(uint g, uint part, uint lane, uint word)
{
    uint4 G = gMisc[gP.vvGroups + g];
    uint k = G.y;
    if (k == 0u || k > VV_PART) return;
    uint off = G.x;
    uint at0 = part * LANES + lane;
    uint stride = SPLIT_TEST_WGS * LANES;
    if ((word & SPLIT_LEVELS) == 0u)
    {
        uint total = 0u;
        for (uint a = 0u, aStop = EB_END(a, k, 1u); a < aStop; a++) total += uint(gAcc[flOwn(vvMember(off, a)) + FL_LISTED].load());
        uint subs = clamp(stride / max(total, 1u), 1u, 64u);
        for (uint p = at0, pStop = EB_END(p, total * subs, stride); p < pStop; p += stride)
        {
            uint q = p / subs;
            uint a = 0u;
            for (uint c = 0u, cGuard = 0u; c + 1u < k; c++)
            {
                EB_PASS(cGuard);
                uint n = uint(gAcc[flOwn(vvMember(off, a)) + FL_LISTED].load());
                if (q < n) break;
                q -= n;
                a++;
            }
            flWalk(vvMember(off, a), q, p % subs, subs);
        }
        return;
    }
    // The listed pairs, a thread each; an owner past the list's room tests inline.
    uint total = 0u;
    for (uint a = 0u, aStop = EB_END(a, k, 1u); a < aStop; a++)
    {
        uint n = uint(gAcc[flOwn(vvMember(off, a)) + FL_PAIRS].load());
        total += n <= TRI_PAIR_CAP ? n : 0u;
    }
    for (uint p = at0, pStop = EB_END(p, total, stride); p < pStop; p += stride)
    {
        uint q = p;
        uint a = 0u;
        for (uint c = 0u, cGuard = 0u; c < k; c++)
        {
            EB_PASS(cGuard);
            uint n = uint(gAcc[flOwn(vvMember(off, c)) + FL_PAIRS].load());
            n = n <= TRI_PAIR_CAP ? n : 0u;
            if (q < n)
            {
                a = c;
                break;
            }
            q -= n;
        }
        flPairTest(vvMember(off, a), q);
    }
    uint over = 0u;
    for (uint a = 0u, aStop = EB_END(a, k, 1u); a < aStop; a++)
    {
        uint own = flOwn(vvMember(off, a));
        over += uint(gAcc[own + FL_PAIRS].load()) > TRI_PAIR_CAP ? uint(gAcc[own + FL_LISTED].load()) : 0u;
    }
    uint subs = clamp(stride / max(over, 1u), 1u, 64u);
    for (uint p = at0, pStop = EB_END(p, over * subs, stride); p < pStop; p += stride)
    {
        uint q = p / subs;
        uint a = 0u;
        for (uint c = 0u, cGuard = 0u; c < k; c++)
        {
            EB_PASS(cGuard);
            uint own = flOwn(vvMember(off, c));
            uint n = uint(gAcc[own + FL_PAIRS].load()) > TRI_PAIR_CAP ? uint(gAcc[own + FL_LISTED].load()) : 0u;
            if (q < n)
            {
                a = c;
                break;
            }
            q -= n;
        }
        flWalkInline(vvMember(off, a), q, p % subs, subs);
    }
}
#endif
