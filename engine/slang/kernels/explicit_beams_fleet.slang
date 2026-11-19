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

// --- The candidates kept across substeps (PERF_FLEET_KEEP; contact-groups-design.md
// §8.10) ---
// Each substep the front gathers every partner's contact points, grids and sorts them and
// walks every listed triangle's cells to list the (triangle, point) pairs its box test
// passes, for 0 to 20 hits, and almost nothing has moved since the substep before. Here
// the pairs are a Verlet list:
//   the build   every (triangle, contact point) of two vehicles the plan pairs whose
//               point is in the triangle's box grown by its range, CLIP_DEPTH, the
//               triangle's and the point's allowances (FK_CALM, or FK_WILD for a node
//               that is fast relative to its vehicle or hot: fkRefNow) and the allowance
//               between vehicles (fkBetween: the build's choice, from how fast partners
//               move relative to each other), whatever the throttle, the partners' bits
//               or the owner's box say this substep (fkPoint, fkWalk: a grid as the
//               front's, padded); every node's position and class and each vehicle's
//               origin recorded (fkRef, fkOrigin); a point within FK_FAR of a kept
//               pair's box marks its vehicle near the owner (fkNear)
//   each pass   the kept pairs tested (fkPairTry): the triangle listed by its throttle
//               step this pass, the vehicles partners this pass, the point in the owner's
//               box (flPoint's test), in the triangle's box (vvInBox), then vvFind: the
//               tests the walk and the listed pairs' test run today, on today's values
//   the bound   per vehicle and class the farthest a node has moved since the build
//               relative to the vehicle's reference motion (fkReference: its nodes' mean
//               motion, predicted), and the fastest one's travel over a step (anti-clip's
//               pads grow with it) (fkTrack); held to the class's allowance
//               (fkVehicleCheck), and per pair of partners of the pass the reference
//               motions' difference, with the origins' (a rebase cancels) and room for
//               rounding, to the allowance between vehicles, FK_FAR more for two
//               vehicles not near each other (fkPairCheck); past either, the list is
//               built again (fkDecide)
// A pair today's walk would list has its point within the triangle's box grown by the
// range and at most CLIP_DEPTH plus both nodes' travel. Since the build the point has
// moved relative to the triangle's nodes by at most its own motion relative to its
// vehicle's reference, the triangle's nodes' relative to theirs, and the two references'
// relative motion; with the travel that is within the two allowances and the one between
// vehicles, so the point was in the build's box: the kept list holds the pair. (Two
// vehicles not near each other: every point was outside the box grown by FK_FAR too.)
// The pairs tested are therefore today's, the hits and the marks the same. The classes,
// the heat, the reference motion and the allowance between vehicles are choices that
// cannot make a list unsound: the build's boxes and the bound use the same ones.
// What is not kept: a pass with a node too fast for a build's allowances, or a bound not
// finite, takes today's path; a build past an owner's share of the fleet's pair room
// (fkKeepPair: the list void, that pass's tests walk the walk's own boxes over the build's
// candidates) builds again two passes on with shares from its counts, and one past the
// fleet's room or the run's is followed by FK_BACKOFF passes on today's path;
// the list's first two words an owner are a signature, so another path's list in the same
// region (the flag off for a frame) ends the kept one.
// Owner v's room in the build's candidate run: its covered partners' contact points.
uint fkOffered(uint v)
{
    uint n = min(gNext[fkCover(v, 0u)], FK_COVER);
    uint sum = 0u;
    for (uint j = 0u; j < FK_COVER; j++)
    {
        if (j < n) sum += vvCpCount(gNext[fkCover(v, 1u + j)] & ~FK_NEAR);
    }
    return sum;
}
// Its candidates of the build (FL_CANDS is the build's count in a build pass).
uint fkCands(uint v) { return min(uint(gAcc[flOwn(v) + FL_CANDS].load()), gNext[fkRun(v + 1u)] - gNext[fkRun(v)]); }

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
            if (rank >= vvRoom()) flError(FL_ERR_SLOTS);
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
    // (The kept candidates' tests read whether this pass listed the triangle: fkPairTry.)
    bool keep = fkOn();
    if (keep) gNext[fkListed(gt)] = 0u;
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
    if (keep) gNext[fkListed(gt)] = 1u;
}

// An owner's part of the fleet-wide candidate ranks (as its run: at its prefix).
uint flRankAt(uint va) { return gP.vvHits + 4u * vvRoomCp() + flRun(va); }
// Its sorted copy and its own grid's offsets.
uint flSortAt(uint va) { return gP.vvRun + vvRoomCp() + flRun(va); }
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
        // (Past the run's room no owner is offered anything: FL_ERR_RUN, fleetPlan.)
        if (gAcc[own + FL_OFFERED].load() == 0) continue;
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
    flArgsZero(lane);
}

// The fleet-wide dispatches' counts back to zero after a group's levels (PERF_FLEET_FRONT:
// fleetPlan wrote them for this pass; without a group nothing does, and they must run no
// thread). The groups' module alone binds both buffers.
void flArgsZero(uint lane)
{
#ifdef EB_VV_GROUPS
    if (lane == 0u && (gP.perfFlags & PERF_FLEET_FRONT) != 0u)
    {
        gArgs[FL_ARGS_ELEMENTS] = 0u;
        gArgs[FL_ARGS_GRIDS] = 0u;
        gArgs[FL_ARGS_SORT] = 0u;
        gArgs[FL_ARGS_ONE] = 0u;
        gArgs2[FL_ARGS2_WALK] = 0u;
        gArgs2[FL_ARGS2_PAIRS] = 0u;
    }
#endif
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

// --- Stage 3a: groups of any size on the fleet-wide passes (PERF_FLEET_ANY) ---
// With the state by vehicle and the front and tests over the fleet, what still knew a
// group's size was its levels: workgroup arrays of VV_PART owners, a VV_PART² partner
// table, four bits a level in one word, an eighth of the buckets an owner. A group past
// VV_PART members therefore kept the larger groups' pass (vvPass: owner after owner in
// one workgroup, its own scans, candidates and tests, all on one core). fleetLevelsAny is
// the levels for any number of owners, in the group's workgroup still: every count from
// the owners' records, the partners from the pairs' bits, an owner's level a word a
// vehicle (flVehAt 5) found by relaxation (contact-groups-design.md §8.3), each owner's
// hits in the canonical order by scan (vvApplyOrdered, as vvPass applies them). The same
// rule as vvPassSmallLevels, so the same results as vvPass's owner after owner: an owner's
// level is one past every earlier owner with hits whose touched vehicles meet its own.

uint flHits(uint v) { return uint(gAcc[flOwn(v) + FL_HITS].load()); }
// An owner's hits past which a small group keeps its own levels (the hits' order by
// buckets; by scan it is the list's length squared).
static const uint FL_SCAN_MOST = 128u;
uint flLevelAt(uint v) { return flVehAt(5u, v); }

// The levels of a group of k owners (any k up to VV_MAX_GROUP) after the tests'
// dispatches, then the pass's end for its owners. Every lane of the group's workgroup.
void fleetLevelsAny(uint k, uint off, uint parity, uint lane, uint tag)
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
        gFl = 2u;
        gFlOff = off;
        gVvTag = 0u;
        // The owners with hits, listed (gsOff, free in this dispatch; H of them, usually a
        // few of the group's): what follows visits those alone. Every lane walking every
        // owner for each phase was the pass's cost in a large group (32 owners: a
        // millisecond a substep with no hit at all).
        if (lane == 0u) gsCount[0].store(0u);
        sync();
        for (uint m = lane, mStop = EB_END(m, k, LANES); m < mStop; m += LANES)
        {
            if (flHits(vvMember(off, m)) != 0u) gsOff[gsCount[0].add(1u)] = m;
        }
        sync();
        uint H = splitUniform(lane, lane == 0u ? gsCount[0].load() : 0u);
        // The longest of their hit lists: a lane past it has no hit in any phase and
        // walks no owner (every lane, when a list overflowed: that owner walks again).
        uint most = 0u;
        if (lane == 0u)
        {
            for (uint j = 0u, jStop = EB_END(j, H, 1u); j < jStop; j++)
            {
                uint n = flHits(vvMember(off, gsOff[j]));
                most = n > vvHitRoom(gsOff[j], 0u) ? 0xFFFFFFFFu : max(most, n);
            }
        }
        most = splitUniform(lane, most);
        uint Hl = lane < most ? H : 0u;
        // The two-body pass's forces at the pass's start, for the listed hits' nodes
        // (vvPassSmallLevels' snapshot); every member's nodes when a list overflowed.
        // (Measurement only, time: the parameter word's bits as vvPassSmallLevels': 65536
        // the snapshot, 32768 the hits' application, 16384 the residual phases; 262144 the
        // levels' rounds. The results are then meaningless.)
        if (gP.vvSplit != 0u && (gP.ablate & 65536u) == 0u)
        {
            bool every = false;
            for (uint j = 0u, jStop = EB_END(j, H, 1u); j < jStop; j++)
            {
                uint a = gsOff[j];
                if (flHits(vvMember(off, a)) > vvHitRoom(a, 0u)) every = true;
            }
            if (every) vvSnapshot(off, k, parity, lane);
            else
            {
                for (uint j = 0u, jStop = EB_END(j, Hl, 1u); j < jStop; j++)
                {
                    uint a = gsOff[j];
                    uint base = vvHitSmall(0u, 0u, a, 0u);
                    uint n = flHits(vvMember(off, a));
                    for (uint i = lane, iStop = EB_END(i, n, LANES); i < iStop; i += LANES)
                    {
                        uint4 tn = gStatic[gP.triBase + gNext[base + 2u * i]];
                        uint h = gNext[base + 2u * i + 1u];
                        vvSnapNode(tn.x, tn.w, parity, tag);
                        vvSnapNode(tn.y, tn.w, parity, tag);
                        vvSnapNode(tn.z, tn.w, parity, tag);
                        vvSnapNode(h, gStatic[gP.nodeInfoBase + h].w, parity, tag);
                    }
                }
            }
            sync();
            if (!every) gVvTag = tag;
        }
        // The owners' levels. Owner b's is one past the highest of an earlier owner c with
        // hits whose touched vehicles (itself and its partners) meet b's partners
        // (vvLevelsSet's rule: c is b's partner, or they share one). A group has at most
        // 64 members, so an owner's partners are two words of bits by member (gsPart, free
        // here: its partners, then those with itself), and a round is a lane an owner with
        // hits over the others with hits: raised from 1 until none changes (a round's
        // reads may be this round's or the last's; either is at most the level, and a
        // round without a change has every owner's rule met).
        for (uint m = lane, mStop = EB_END(m, k, LANES); m < mStop; m += LANES)
        {
            uint v = vvMember(off, m);
            gNext[flLevelAt(v)] = flHits(v) != 0u ? 1u : 0u;
        }
        for (uint j = lane, jStop = EB_END(j, H, LANES); j < jStop; j += LANES)
        {
            uint m = gsOff[j];
            uint v = vvMember(off, m);
            uint2 p = uint2(0u, 0u);
            uint e1 = gNext[flAdjFirst(v + 1u)];
            for (uint e = gNext[flAdjFirst(v)], eStop = EB_END(e, e1, 1u); e < eStop; e++)
            {
                uint d = gNext[flAdjAt(e)];
                if ((gNext[flBitsAt(d)] & 5u) != 1u) continue;
                uint r = gMisc[vvRec(flPartner(d), VV_MEMBER)].x;
                if (r < 32u) p.x |= 1u << r;
                else if (r < 64u) p.y |= 1u << (r - 32u);
            }
            gsPart[4u * j] = p.x;
            gsPart[4u * j + 1u] = p.y;
            gsPart[4u * j + 2u] = p.x | (m < 32u ? 1u << m : 0u);
            gsPart[4u * j + 3u] = p.y | (m >= 32u ? 1u << (m - 32u) : 0u);
        }
        sync();
        for (uint round = 0u; round < EB_MAX_MEMBERS + 2u; round++)
        {
            if (H == 0u || (gP.ablate & 262144u) != 0u) break;
            if (lane == 0u) gsUniformLanes = 0u;
            sync();
            bool changed = false;
            for (uint j = lane, jStop = EB_END(j, H, LANES); j < jStop; j += LANES)
            {
                uint m = gsOff[j];
                uint v = vvMember(off, m);
                uint l = 1u;
                for (uint c = 0u, cStop = EB_END(c, H, 1u); c < cStop; c++)
                {
                    uint mc = gsOff[c];
                    if (mc >= m) continue;
                    if ((gsPart[4u * j] & gsPart[4u * c + 2u]) == 0u && (gsPart[4u * j + 1u] & gsPart[4u * c + 3u]) == 0u) continue;
                    l = max(l, gNext[flLevelAt(vvMember(off, mc))] + 1u);
                }
                if (l != gNext[flLevelAt(v)])
                {
                    gNext[flLevelAt(v)] = l;
                    changed = true;
                }
            }
            if (changed) gsUniformLanes = 1u;
            sync();
            uint go = gsUniformLanes;
            sync();
            if (go == 0u) break;
        }
        uint top = 1u;
        if (lane == 0u)
        {
            for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++) top = max(top, gNext[flLevelAt(vvMember(off, m))]);
        }
        top = splitUniform(lane, top);
        // The hits, level by level; with the first, every listed triangle's throttle
        // update (its marks are final) and a later owner's reset.
        for (uint l = 1u, lGuard = 0u; l <= top; l++)
        {
            EB_PASS_UNIFORM(lGuard, EB_MAX_MEMBERS + 1u);
            if (l == 1u && !flPostOn())
            {
                for (uint a = 0u, aStop = EB_END(a, k, 1u); a < aStop; a++)
                {
                    uint va = vvMember(off, a);
                    uint own = flOwn(va);
                    uint triOff = gStatic[va * 4u + 1u].x;
                    uint listed = uint(gAcc[own + FL_LISTED].load());
                    bool fix = gAcc[own + FL_FIX].load() != 0;
                    for (uint i = lane, iStop = EB_END(i, listed, LANES); i < iStop; i += LANES)
                    {
                        uint gt = triOff + gNext[vvTestedAt(triOff + i)];
                        vvThrottleEnd(gt);
                        if (fix) gMisc[gP.vvTri + gt].x = 0u;
                    }
                }
            }
            for (uint j = 0u, jStop = EB_END(j, Hl, 1u); j < jStop; j++)
            {
                uint a = gsOff[j];
                uint va = vvMember(off, a);
                if (gNext[flLevelAt(va)] != l || (gP.ablate & 32768u) != 0u) continue;
                float range = asfloat(gStatic[va * 4u + 3u].z);
                uint hits = flHits(va);
                if (hits <= vvHitRoom(a, 0u))
                    vvApplyOrdered(va, vvHitSmall(0u, 0u, a, 0u), hits, range, parity, lane, LANES, a);
                else
                {
                    // (A list past its room, reported by flOwnerEnd: the owner's walk
                    // again, each hit applied at once, as vvPassSmallLevels.)
                    uint4 ra1 = gStatic[va * 4u + 1u];
                    uint listed = uint(gAcc[flOwn(va) + FL_LISTED].load());
                    uint subs = clamp(LANES / max(listed, 1u), 1u, 64u);
                    for (uint p = lane, pStop = EB_END(p, listed * subs, LANES); p < pStop; p += LANES)
                    {
                        uint t = gNext[vvTestedAt(ra1.x + p / subs)];
                        vvTriangleCandsApply(va, ra1.x + t, p % subs, subs, vvRunSmall(0u, 0u, a), flCands(va), range, parity, a);
                    }
                }
            }
            sync();
            if (gP.vvSplit != 0u && (gP.ablate & 16384u) == 0u)
            {
                for (uint ph = 0u; ph < 3u; ph++)
                {
                    for (uint j = 0u, jStop = EB_END(j, Hl, 1u); j < jStop; j++)
                    {
                        uint a = gsOff[j];
                        uint va = vvMember(off, a);
                        if (gNext[flLevelAt(va)] != l) continue;
                        uint hits = flHits(va);
                        if (hits > vvHitRoom(a, 0u)) continue;
                        vvResidualPhase(va, vvHitSmall(0u, 0u, a, 0u), hits, asfloat(gStatic[va * 4u + 3u].z), parity, lane, LANES, a, ph);
                    }
                    sync();
                }
            }
        }
        // The two-body counts the hits wrote, once all levels are applied.
        if (gP.vvSplit != 0u)
        {
            uint anyHits = 0u;
            bool listedAll = true;
            for (uint j = 0u, jStop = EB_END(j, H, 1u); j < jStop; j++)
            {
                uint a = gsOff[j];
                uint n = flHits(vvMember(off, a));
                anyHits += n;
                if (n > vvHitRoom(a, 0u)) listedAll = false;
            }
            if (anyHits > 0u && !listedAll) vvSplitClear(off, k, vvRoom(), lane);
            else if (anyHits > 0u)
            {
                for (uint j = 0u, jStop = EB_END(j, Hl, 1u); j < jStop; j++)
                {
                    uint a = gsOff[j];
                    uint base = vvHitSmall(0u, 0u, a, 0u);
                    uint n = flHits(vvMember(off, a));
                    for (uint i = lane, iStop = EB_END(i, n, LANES); i < iStop; i += LANES)
                    {
                        uint4 tn = gStatic[gP.triBase + gNext[base + 2u * i]];
                        gAcc[vvSplitTri(tn.x)].store(0);
                        gAcc[vvSplitTri(tn.y)].store(0);
                        gAcc[vvSplitTri(tn.z)].store(0);
                        gAcc[vvSplitPoint(a, gNext[base + 2u * i + 1u])].store(0);
                    }
                }
            }
        }
        gVvTag = 0u;
        gFl = 0u;
    }
    sync();
    for (uint m = lane, mStop = EB_END(m, k, LANES); m < mStop; m += LANES) flOwnerEnd(vvMember(off, m));
    flArgsZero(lane);
}

// --- Stage 3b: the levels' work that needs no order, as threads over the fleet
// (PERF_FLEET_POST) ---
// Every listed triangle's throttle update (vvThrottleEnd: its marks are final once the
// tests are done) and a later owner's reset ran in the group's workgroup with its first
// level: 528 triangles a car on one core every substep, hits or not (32 cars: 31 to 68 ms
// a frame in the levels with few hits). They read and write the triangle's own record and
// marks and nothing a hit touches, so here they are a thread a listed triangle, in a
// dispatch after the listed pairs' tests and before the groups' levels, which then leave
// them out (flPostOn). Sized by the walk's dispatch (at least a thread a listed triangle).
// With the kept candidates (PERF_FLEET_KEEP) a pass that reuses them computes no walk, so
// no prefix of the listed triangles: there the dispatch is the elements' (a thread a
// triangle of the fleet, flTriangle's) and a thread updates its triangle when this pass
// listed it (fkListed, which flTriangle wrote in this pass for every triangle of an owner
// on the fleet-wide passes): the same triangles, each once (fkPost).
bool flPostOn() { return (gP.perfFlags & (PERF_FLEET_FRONT | PERF_FLEET_POST)) == (PERF_FLEET_FRONT | PERF_FLEET_POST); }

#ifdef EB_FLEET_POST
// Thread tid of the elements' dispatch with the kept candidates: triangle tid's update
// when its throttle step listed it this pass (a separate path: fleetPost returns into it).
void fkPost(uint tid)
{
    if (tid >= gP.totalTris) return;
    uint va = gStatic[gP.triBase + tid].w;
    if (!flVehicleOn(va) || gNext[fkListed(tid)] == 0u) return;
    vvThrottleEnd(tid);
    if (gAcc[flOwn(va) + FL_FIX].load() != 0) gMisc[gP.vvTri + tid].x = 0u;
}
void fleetPost(uint tid)
{
    uint nv = gP.vehicles;
    if (!flPostOn()) return;
    if (fkOn())
    {
        fkPost(tid);
        return;
    }
    if (tid >= gNext[flPrefixAt(1u, nv)]) return;
    uint va = flFind(1u, tid);
    uint triOff = gStatic[va * 4u + 1u].x;
    uint gt = triOff + gNext[vvTestedAt(triOff + tid - gNext[flPrefixAt(1u, va)])];
    vvThrottleEnd(gt);
    if (gAcc[flOwn(va) + FL_FIX].load() != 0) gMisc[gP.vvTri + gt].x = 0u;
}

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_fleet_post(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    fleetPost(gid.x * LANES + lid.x);
    ebLoopCapFlush();
}
#endif

// --- Stage 2: the front as fleet-wide dispatches (contact-groups-design.md §8.4) ---
// Each a dispatch over every pair, vehicle or element, its phase the step word's (three a
// module: the step alone, SPLIT_TESTS, SPLIT_LEVELS):
//   explicit_beams_fleet_front  the pairs' step scans (a thread a pair);
//                               SPLIT_TESTS: every triangle's throttle step and every
//                               contact point's gather (a thread each);
//                               SPLIT_LEVELS: the candidates' sorted copy (a thread a
//                               place of the fleet-wide run)
//   explicit_beams_fleet_plan   one workgroup each: after the scans the owners' parts of
//                               the run (the prefix of their offered counts) and their
//                               records of the step; SPLIT_TESTS: after the gather the
//                               listed triangles' prefix; SPLIT_LEVELS: after the walk
//                               the listed pairs' prefix
//   explicit_beams_split_tests  a workgroup a vehicle, its grid's offsets; SPLIT_TESTS:
//                               the walk (a thread a part of a listed triangle);
//                               SPLIT_LEVELS: the listed pairs (a thread a pair)
// The dispatches' sizes: the scans' and the first plan's from the contact plan's
// arguments (none without a group). The first plan writes the fleet's first arguments
// (gArgs, binding 10: the triangles and points at 0, the grids at 4, the sorted copy at
// 8, one workgroup for each later plan at 12; all zero when no owner is offered a
// partner, so a group whose members have not met costs two small dispatches a step); the
// later plans, run from those, write the second arguments (gArgs2, binding 11: the walk
// at 0, the pairs at 4). A dispatch never binds the buffer it runs from. The groups'
// levels zero both after the pass (no group: no workgroup anywhere).
[[vk::binding(11, 0)]] RWStructuredBuffer<uint> gArgs2;

// The step's parity for vehicle v's boxes: its group's first member's step count (as the
// group's pass has it), kept at the groups' frame start.
uint flParity(uint v, uint s) { return (gNext[flVehAt(0u, v)] + s) & 1u; }

// The last vehicle whose prefix (row r) is at most x: the owner of place x of a
// fleet-wide list (the vehicles without a part have the next one's start).
// (Rows FK_ROW_RUN and FK_ROW_PAIRS: the kept candidates' own, fkRun and fkPrefix.)
uint flRowAt(uint r, uint v) { return r == FK_ROW_RUN ? fkRun(v) : (r == FK_ROW_PAIRS ? fkPrefix(v) : (r == FK_ROW_ALLOC ? fkAlloc(v) : flPrefixAt(r, v))); }
uint flFind(uint r, uint x)
{
    uint lo = 0u;
    uint hi = gP.vehicles;
    for (uint it = 0u; it < 32u; it++)
    {
        if (hi - lo <= 1u) break;
        uint mid = (lo + hi) >> 1u;
        if (gNext[flRowAt(r, mid)] <= x) lo = mid;
        else hi = mid;
    }
    return lo;
}

// Row r of the prefixes from the owners' counter `word` (a listed pairs' count past the
// list's room counts 0: that owner tests inline). Every lane of one workgroup, uniform:
// a small fleet's by lane 0 (two barriers), a larger one's a scan over the lanes (five).
// The total is the row's last word.
static const uint FL_SERIAL = 64u;
uint flCount(uint v, uint word)
{
    // (The kept candidates' counts: the build's run of an owner, its kept pairs.)
    if (word == FK_W_RUN) return fkOffered(v);
    if (word == FK_W_PAIRS) return min(uint(gAcc[fkAcc(v, FK_PAIRS)].load()), fkShare(v));
    if (word == FK_W_ALLOC) return fkWant(v);
    uint n = uint(gAcc[flOwn(v) + word].load());
    return word == FL_PAIRS && n > TRI_PAIR_CAP ? 0u : n;
}
void flPrefix(uint r, uint word, uint lane)
{
    uint nv = gP.vehicles;
    sync();
    if (nv <= FL_SERIAL)
    {
        if (lane == 0u)
        {
            uint run = 0u;
            for (uint v = 0u, vStop = EB_END(v, nv, 1u); v < vStop; v++)
            {
                gNext[flRowAt(r, v)] = run;
                run += flCount(v, word);
            }
            gNext[flRowAt(r, nv)] = run;
        }
        sync();
        return;
    }
    uint chunk = (nv + LANES - 1u) / LANES;
    uint v0 = min(lane * chunk, nv);
    uint v1 = min(v0 + chunk, nv);
    uint block = LANES / GRID_BLOCKS;
    uint sum = 0u;
    for (uint v = v0, vStop = EB_END(v, v1, 1u); v < vStop; v++) sum += flCount(v, word);
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
        gNext[flRowAt(r, nv)] = run;
    }
    sync();
    {
        uint run = gsPart[lane] + gsBlk[lane / block];
        for (uint v = v0, vStop = EB_END(v, v1, 1u); v < vStop; v++)
        {
            gNext[flRowAt(r, v)] = run;
            run += flCount(v, word);
        }
    }
    sync();
}

// --- The kept candidates' passes (PERF_FLEET_KEEP; the design at this file's head) ---

// The nodes of vehicle v whose motion the mean takes (fkTrackNode: one in eight).
uint fkSampled(uint v)
{
    uint4 r0 = gStatic[v * 4u];
    return ((r0.x + r0.y + 7u) >> 3u) - ((r0.x + 7u) >> 3u);
}
// Node n of vehicle v now, as a build records it: its position and its class, wild when
// it is fast relative to the vehicle (its reference motion at a build is its mean motion
// over the last pass: a spinning wheel, a piece flying off), too fast for a calm node's
// allowance at all, or hot. (Any choice is sound: the class only says which allowance the
// node's pairs are built with and its motion is held to.)
float4 fkRefNow(uint n, uint v)
{
    float3 w = gNodes[n * 3u + 1u].xyz;
    float3 dv = w * STEP - fkReference(v);
    bool calm = dot(dv, dv) <= (FK_WILD_SPEED * STEP) * (FK_WILD_SPEED * STEP) && dot(w, w) * (STEP * STEP) <= 0.25 * FK_CALM * FK_CALM;
    uint heat = asuint(gBeams[fkRef(n)].w) >> 1u;
    if (heat > (gNext[fkCtl(FK_PASS)] >> 8u)) calm = false;
    return float4(gNodes[n * 3u].xyz, asfloat((heat << 1u) | (calm ? 0u : 1u)));
}
#ifdef EB_FLEET_FRONT
// Thread tid of the scans' dispatch: node tid.
void fkTrack(uint tid)
{
    if (tid >= gP.totalNodes || fkTrackInRest()) return;
    // (Passes left on today's path: this pass's bounds decide nothing, the pass is today's
    // whatever they say. The reference motion they would feed is any vector and the heat a
    // class: both choices. With none left they are tracked: a build needs them.)
    if (fkWaitLeft() > 1u) return;
    uint v = gStatic[gP.nodeInfoBase + tid].w;
    if (flVehicleOn(v)) fkTrackNode(tid, v);
}

// The build's gather of contact point i of vehicle vb, into the run of every owner the
// build covers it for (its vehicle's plan partners) whose box, grown by its range
// (flOwnerPlan's), by the most the build's triangle boxes are grown by, FK_FAR and room
// for rounding, holds it (every triangle of the owner is inside the owner's box); a wild
// point also into the box of the owner's wild candidates. (Its class as the node's own
// thread of this dispatch records it: the same function of the same values.)
void fkPoint(uint vb, uint i)
{
    uint h = gLists[gStatic[gP.vvStatic + vb].x + i];
    float4 ref = fkRefNow(h, vb);
    float3 x = nf3(asfloat(gMisc[vb].xyz) + ref.xyz);
    uint n = min(gNext[fkCover(vb, 0u)], FK_COVER);
    for (uint j = 0u; j < FK_COVER; j++)
    {
        if (j >= n) break;
        uint va = gNext[fkCover(vb, 1u + j)] & ~FK_NEAR;
        float4 B0 = gBeams[flBoxAt(va)];
        float4 B1 = gBeams[flBoxAt(va) + 1u];
        float3 m3 = max(max(abs(B0.xyz), abs(B1.xyz)), abs(x));
        float ex = CLIP_DEPTH + 2.0 * FK_WILD + FK_BETWEEN_MOST + FK_FAR + 0.002 + 4.0e-6 * max(m3.x, max(m3.y, m3.z));
        float3 e3 = float3(ex, ex, ex);
        if (any(x < B0.xyz - e3) || any(x > B1.xyz + e3)) continue;
        uint base = gNext[fkRun(va)];
        uint ci = uint(gAcc[flOwn(va) + FL_CANDS].add(1));
        if (ci >= gNext[fkRun(va + 1u)] - base) continue;
        float4 E = float4(nf3(vvRel(vb, va) + ref.xyz), asfloat(h));
        gBeams[gP.vvRun + base + ci] = E;
        // (The owner's fastest candidate, as flPoint keeps it: a build past an owner's room
        // walks its tests in the walk's own boxes, fkWalk.)
        float3 w3 = gNodes[h * 3u + 1u].xyz;
        gAcc[flOwn(va) + FL_FAST].max(int(asuint(ssqrt(sdot(w3, w3)))));
        uint b = gridBucket(int3(floor(E.xyz * FK_PER_CELL)));
        gNext[gP.vvHits + 4u * VV_PART * vvTotalCp() + base + ci] = b | (uint(gAcc[flGridCount(va) + b].add(1)) << 10);
        if (fkWild(ref))
        {
            gAcc[fkAcc(va, FK_WILD_BOX)].min(vvKey(E.x));
            gAcc[fkAcc(va, FK_WILD_BOX + 1u)].min(vvKey(E.y));
            gAcc[fkAcc(va, FK_WILD_BOX + 2u)].min(vvKey(E.z));
            gAcc[fkAcc(va, FK_WILD_BOX + 3u)].max(vvKey(E.x));
            gAcc[fkAcc(va, FK_WILD_BOX + 4u)].max(vvKey(E.y));
            gAcc[fkAcc(va, FK_WILD_BOX + 5u)].max(vvKey(E.z));
        }
    }
}
// Place tid of the build's run to its place in the sorted copy (flSortCand's).
void fkSortCand(uint tid)
{
    uint nv = gP.vehicles;
    if (tid >= gNext[fkRun(nv)]) return;
    uint va = flFind(FK_ROW_RUN, tid);
    uint e = tid - gNext[fkRun(va)];
    if (e < fkCands(va)) fkSortCandOf(va, e);
}
#endif

#ifdef EB_FLEET_PLAN
// Vehicle va's entry of partner vb in the build's cover, 1 added (0: not covered).
uint fkCovered(uint va, uint vb)
{
    uint found = 0u;
    uint n = min(gNext[fkCover(va, 0u)], FK_COVER);
    for (uint j = 0u; j < FK_COVER; j++)
    {
        uint e = gNext[fkCover(va, 1u + j)];
        if (j < n && (e & ~FK_NEAR) == vb) found = (e & FK_NEAR) | 1u;
    }
    return found;
}
// A vehicle's bounds of the pass: 0 its tracked nodes are within their classes'
// allowances (a calm node's motion and travel within FK_CALM, any node's within FK_WILD),
// 1 a wild one is not, 2 a calm one is not (a build), 5 a node is too fast for a build's
// allowances, or a bound is not finite (today's path).
uint fkVehicleCheck(uint v)
{
    uint dc = uint(gAcc[fkAcc(v, FK_DEV)].load());
    uint tc = uint(gAcc[fkAcc(v, FK_TRAV)].load());
    uint dw = uint(gAcc[fkAcc(v, FK_DEV + 2u)].load());
    uint tw = uint(gAcc[fkAcc(v, FK_TRAV + 2u)].load());
    if (max(max(dc, tc), max(dw, tw)) >= 0x7F800000u) return 5u;
    // (Float bits of values that are not negative order as the values.)
    if (max(tc, tw) > asuint(0.5 * FK_WILD)) return 5u;
    if (fkBits(asfloat(dc) + asfloat(tc)) > asuint(FK_CALM)) return 2u;
    return fkBits(asfloat(max(dc, dw)) + asfloat(max(tc, tw))) > asuint(FK_WILD) ? 1u : 0u;
}
// Pair p of the plan this pass: 0 the kept list holds every pair today's walk would list
// for it (or its vehicles are not partners this pass); it may not, a build: 1, 2 its
// nodes having moved (fkVehicleCheck), 3 its vehicles' first nodes having, 4 the build
// did not cover it; 5 today's path (fkVehicleCheck).
uint fkPairCheck(uint p)
{
    uint bits = gNext[flBitsAt(2u * p)];
    if ((bits & 5u) != 1u) return 0u;
    uint va = gNext[gP.vvPairs + 2u * p];
    uint vb = gNext[gP.vvPairs + 2u * p + 1u];
    uint code = max(fkVehicleCheck(va), fkVehicleCheck(vb));
    if (code == 5u) return 5u;
    // Covered by the build, each way; near when a point of either was within FK_FAR of a
    // kept pair's box of the other's triangles at the build (fkWalk).
    uint ab = fkCovered(va, vb);
    uint ba = fkCovered(vb, va);
    if (ab == 0u || ba == 0u) return 4u;
    float limit = ((ab | ba) & FK_NEAR) != 0u ? fkBetween() : fkBetween() + FK_FAR;
    // The two vehicles' relative motion since the build, in the owner's frame as the tests
    // have it (a point of vb at vvRel(vb, va) + its position): their reference motions'
    // difference, with the origins' terms as the tests and the nodes' motions round them,
    // and room for the tests' rounding.
    float3 la = gNodes[gStatic[va * 4u].x * 3u].xyz;
    float3 lb = gNodes[gStatic[vb * 4u].x * 3u].xyz;
    float3 rel = vvRel(vb, va);
    float3 relThen = asfloat(uint3(gNext[fkOrigin(vb)], gNext[fkOrigin(vb) + 1u], gNext[fkOrigin(vb) + 2u]))
                     - asfloat(uint3(gNext[fkOrigin(va)], gNext[fkOrigin(va) + 1u], gNext[fkOrigin(va) + 2u]));
    float3 sa = fkOriginShift(va);
    float3 sb = fkOriginShift(vb);
    float3 ca = fkReference(va);
    float3 cb = fkReference(vb);
    float3 m3 = max(max(max(abs(la), abs(lb)), max(abs(rel), abs(relThen))), max(max(abs(sa), abs(sb)), max(abs(ca), abs(cb))));
    float3 c = (cb - ca) + ((rel - relThen) - (sb - sa));
    // (The rounding: the tests' few operations on these coordinates, each within half a
    // unit in the last place, 6e-8 of them; 2 mm and thirty times that.)
    float between = sqrt(dot(c, c)) + 0.002 + 2.0e-6 * max(m3.x, max(m3.y, m3.z));
    uint bb = asuint(between);
    if (bb >= 0x7F800000u) return 5u;
    // (Measurement: the largest of the pass.)
    gAcc[fkAccCtl(3u)].max(int(bb));
    return bb > asuint(limit) ? 3u : code;
}

// The first plan's decision of the pass (every lane; uniform): the mode, with a build's
// records (each vehicle's covered partners, origin, signature, its part of the build's
// run). Returns the mode. In two parts: the pairs' and vehicles' checks, before a barrier
void fkCheck(uint lane)
{
    uint nv = gP.vehicles;
    uint epoch = gNext[fkCtl(FK_EPOCH)];
    uint code = 0u;
    uint pairs = gNext[flAdjFirst(nv)] / 2u;
    for (uint p = lane, pStop = EB_END(p, pairs, LANES); p < pStop; p += LANES) code = max(code, fkPairCheck(p));
    // (Another path's list in an owner's region: the kept one is gone.)
    for (uint v = lane, vStop = EB_END(v, nv, LANES); v < vStop; v += LANES)
    {
        if (gNext[vvTriPairsAt(v)] != FK_MAGIC || gNext[vvTriPairsAt(v) + 1u] != epoch) code = max(code, 4u);
        // (Measurement: the vehicles' largest bounds of the pass.)
        gAcc[fkAccCtl(1u)].max(max(gAcc[fkAcc(v, FK_DEV)].load(), gAcc[fkAcc(v, FK_DEV + 2u)].load()));
    }
    if (code != 0u) gAcc[fkAccCtl(0u)].max(int(code));
}
// … and after a barrier.
uint fkDecide(uint lane, bool untracked)
{
    uint nv = gP.vehicles;
    uint epoch = gNext[fkCtl(FK_EPOCH)];
    uint pairs = gNext[flAdjFirst(nv)] / 2u;
    uint mode = FK_OFF;
    if (lane == 0u)
    {
        uint c = uint(gAcc[fkAccCtl(0u)].load());
        // (A pass whose nodes were not tracked, fkTrackIdle: its bounds say nothing, so no
        // list is reused in it; a build or today's front.)
        if (untracked) c = max(c, 4u);
        gAcc[fkAccCtl(0u)].store(0);
        uint wait = fkWaitLeft();
        if (gNext[flPrefixAt(0u, nv)] != 0u)
        {
            if (wait != 0u) gNext[fkCtl(FK_WAIT)] -= 1u;
            else if (c == 0u && gNext[fkCtl(FK_VALID)] != 0u) mode = FK_REUSE;
            else if (c < 5u)
            {
                // A list that lasted fewer than FK_SHORT passes (wild nodes, tumbling cars:
                // a build every few passes costs more than today's front): today's path for
                // the next FK_THRASH_WAIT passes instead of another build. (A choice of path
                // alone: today's front is exact; FK_VALID holds the pass of the list's build.)
                uint built = gNext[fkCtl(FK_VALID)];
                if (built != 0u && gNext[fkCtl(FK_PASS)] - built < FK_SHORT) gNext[fkCtl(FK_WAIT)] = (gNext[fkCtl(FK_WAIT)] & 0xFFFF0000u) | FK_THRASH_WAIT;
                else mode = FK_BUILD;
            }
            // (Measurement, of builds that end a list: the largest node bound at one for a
            // node's motion; at one for two vehicles', their bound, in the same word.)
            bool had = gNext[fkCtl(FK_VALID)] != 0u && gNext[fkCtl(FK_REUSED)] != 0u;
            if (had && mode == FK_BUILD && (c == 1u || c == 2u)) gNext[fkCtl(FK_DEV_MOST)] = max(gNext[fkCtl(FK_DEV_MOST)], uint(gAcc[fkAccCtl(1u)].load()));
            if (had && mode == FK_BUILD && c == 3u) gNext[fkCtl(FK_DEV_MOST)] = max(gNext[fkCtl(FK_DEV_MOST)], uint(gAcc[fkAccCtl(3u)].load()));
            if (mode == FK_BUILD && c == 2u) gNext[fkCtl(FK_BY_CALM)] += 1u;
            if (mode == FK_BUILD && c == 3u) gNext[fkCtl(FK_BY_BETWEEN)] += 1u;
            if (mode == FK_BUILD && (c == 4u || c == 0u)) gNext[fkCtl(FK_BY_COVER)] += 1u;
        }
        gAcc[fkAccCtl(1u)].store(0);
        gAcc[fkAccCtl(3u)].store(0);
        gNext[fkCtl(FK_PASS)] += 1u;
    }
    if (lane == 0u) gsUniformLanes = mode;
    sync();
    mode = gsUniformLanes;
    // Each vehicle's reference motion of the next pass: its tracked nodes' mean motion of
    // this one plus its change over this pass; from a build, whose records are this pass's
    // positions, that change alone. The pass's bounds and sums back to zero.
    for (uint v = lane, vStop = EB_END(v, nv, LANES); v < vStop; v += LANES)
    {
        float visits = float(max(fkSampled(v), 1u)) * 4096.0;
        float3 mean = float3(float(gAcc[fkAcc(v, FK_SUM)].load()), float(gAcc[fkAcc(v, FK_SUM + 1u)].load()), float(gAcc[fkAcc(v, FK_SUM + 2u)].load())) / visits;
        float3 delta = mean - asfloat(uint3(gNext[fkMean(v, 3u)], gNext[fkMean(v, 4u)], gNext[fkMean(v, 5u)]));
        float3 next = mode == FK_BUILD ? delta : mean + delta;
        if (mode == FK_BUILD) mean = float3(0.0, 0.0, 0.0);
        uint3 nb = asuint(next);
        uint3 mb = asuint(mean);
        gNext[fkMean(v, 0u)] = nb.x;
        gNext[fkMean(v, 1u)] = nb.y;
        gNext[fkMean(v, 2u)] = nb.z;
        gNext[fkMean(v, 3u)] = mb.x;
        gNext[fkMean(v, 4u)] = mb.y;
        gNext[fkMean(v, 5u)] = mb.z;
        for (uint w = 0u; w < 4u; w++) gAcc[fkAcc(v, w)].store(0);
        for (uint w = 0u; w < 3u; w++) gAcc[fkAcc(v, FK_SUM + w)].store(0);
    }
    if (mode == FK_BUILD)
    {
        // (Each owner's share of the fleet's room for its kept pairs, from its last build's
        // count, before the counts are cleared: fkBase.)
        flPrefix(FK_ROW_ALLOC, FK_W_ALLOC, lane);
        if (lane == 0u) gsUniformOpen = 0u;
        sync();
        bool tooMany = false;
        for (uint v = lane, vStop = EB_END(v, nv, LANES); v < vStop; v += LANES)
        {
            uint n = 0u;
            if (flVehicleOn(v))
            {
                uint e1 = gNext[flAdjFirst(v + 1u)];
                for (uint e = gNext[flAdjFirst(v)], eStop = EB_END(e, e1, 1u); e < eStop; e++)
                {
                    uint d = gNext[flAdjAt(e)];
                    if ((gNext[flBitsAt(d)] & 4u) != 0u) continue;
                    if (n < FK_COVER) gNext[fkCover(v, 1u + n)] = flPartner(d);
                    n++;
                }
            }
            if (n > FK_COVER)
            {
                tooMany = true;
                n = 0u;
            }
            gNext[fkCover(v, 0u)] = n;
            uint3 o = gMisc[v].xyz;
            gNext[fkOrigin(v)] = o.x;
            gNext[fkOrigin(v) + 1u] = o.y;
            gNext[fkOrigin(v) + 2u] = o.z;
            gAcc[fkAcc(v, FK_PAIRS)].store(0);
            gNext[fkShareAt(v)] = fkShare(v);
            for (uint w = 0u; w < 6u; w++) gAcc[fkAcc(v, FK_WILD_BOX + w)].store(w < 3u ? VV_INT_MAX : VV_INT_MIN);
            gNext[vvTriPairsAt(v)] = FK_MAGIC;
            gNext[vvTriPairsAt(v) + 1u] = epoch + 1u;
        }
        if (lane == 0u) gAcc[fkAccCtl(2u)].store(0);
        if (tooMany) gsUniformOpen = 1u;
        // The fastest relative motion of two partners over the last pass (their reference
        // motions at a build are that pass's mean motions): the list's allowance for it.
        uint relMost = 0u;
        for (uint p = lane, pStop = EB_END(p, pairs, LANES); p < pStop; p += LANES)
        {
            if ((gNext[flBitsAt(2u * p)] & 5u) != 1u) continue;
            float3 rv = fkReference(gNext[gP.vvPairs + 2u * p + 1u]) - fkReference(gNext[gP.vvPairs + 2u * p]);
            relMost = max(relMost, fkBits(sqrt(dot(rv, rv))));
        }
        if (relMost != 0u) gAcc[fkAccCtl(3u)].max(int(relMost));
        sync();
        flPrefix(FK_ROW_RUN, FK_W_RUN, lane);
        uint total = splitUniform(lane, gsUniformOpen != 0u ? 0xFFFFFFFFu : gNext[fkRun(nv)]);
        // (Past the fleet-wide run's room, or a vehicle with more partners than a build
        // covers: today's path.)
        if (total > VV_PART * vvTotalCp()) mode = FK_OFF;
        if (lane == 0u)
        {
            gNext[fkCtl(FK_VALID)] = 0u;
            gNext[fkCtl(FK_EPOCH)] = epoch + 1u;
            if (mode == FK_OFF) fkBackoff();
            gNext[fkCtl(FK_BTW)] = asuint(min(FK_BETWEEN_PASSES * asfloat(uint(gAcc[fkAccCtl(3u)].load())) + FK_BETWEEN, FK_BETWEEN_MOST));
            gAcc[fkAccCtl(3u)].store(0);
        }
    }
    if (lane == 0u)
    {
        gNext[fkCtl(FK_MODE)] = mode;
        if (mode == FK_REUSE) gNext[fkCtl(FK_REUSED)] += 1u;
        if (mode == FK_OFF && gNext[flPrefixAt(0u, nv)] != 0u) gNext[fkCtl(FK_FELL)] += 1u;
    }
    return mode;
}
#endif

#ifdef EB_VV_GLOBAL
// A kept pair's tests this pass, as today's front and walk select it and vvPairTest tests
// it: triangle gt of owner va listed by its throttle step, contact point h's vehicle a
// partner, h in the owner's box (flPoint's test) and in the triangle's (vvInBox: the
// triangle's had-candidates mark), then the hit test (vvFind), the hit listed.
void fkPairTry(uint va, uint gt, uint h)
{
    if (gNext[fkListed(gt)] == 0u) return;
    uint own = flOwn(va);
    uint offered = uint(gAcc[own + FL_OFFERED].load());
    if (offered == 0u) return;
    uint vb = gStatic[gP.nodeInfoBase + h].w;
    bool partner = false;
    uint e1 = gNext[flAdjFirst(va + 1u)];
    for (uint e = gNext[flAdjFirst(va)], eStop = EB_END(e, e1, 1u); e < eStop; e++)
    {
        uint d = gNext[flAdjAt(e)];
        if (flPartner(d) == vb && (gNext[flBitsAt(d)] & 1u) != 0u) partner = true;
    }
    if (!partner) return;
    {
        float3 o = asfloat(gMisc[vb].xyz);
        float3 x = nf3(o + gNodes[h * 3u].xyz);
        float3 vh3 = gNodes[h * 3u + 1u].xyz;
        float4 B0 = gBeams[flBoxAt(va)];
        float4 B1 = gBeams[flBoxAt(va) + 1u];
        float ex = 0.0;
        if (gP.clipMem != 0u) ex = B0.w + sqrt(dot(vh3, vh3)) * STEP;
        float3 e3 = float3(ex, ex, ex);
        if (any(x < B0.xyz - e3) || any(x > B1.xyz + e3)) return;
    }
    gFl = 1u;
    gFlV = va;
    float r = asfloat(gStatic[va * 4u + 3u].z);
    VvTri t = vvTriangle(gt, vvRel(va, va), r);
    float3 p = nf3(vvRel(vb, va) + gNodes[h * 3u].xyz);
    bool anti = vvAnti(t, vb);
    if (vvInBox(t, p, h, anti))
    {
        vvMark(gt, true, false);
        if (vvFind(t, gt, h, p, vb, r, anti))
        {
            vvSplitCount(t, h, 0u);
            uint n = vvHitAdd(0u);
            if (n < 2u * offered)
            {
                gNext[vvHitSmall(0u, offered, 0u, n)] = gt;
                gNext[vvHitSmall(0u, offered, 0u, n) + 1u] = h;
            }
            vvMark(gt, false, true);
        }
    }
    gFl = 0u;
}

// Thread tid of the kept pairs' tests.
void fkTest(uint tid)
{
    uint nv = gP.vehicles;
    if (tid >= gNext[fkPrefix(nv)]) return;
    uint va = flFind(FK_ROW_PAIRS, tid);
    uint at = fkSlotAt(fkBase(va) + tid - gNext[fkPrefix(va)]);
    fkPairTry(va, gNext[at], gNext[at + 1u]);
}

// A point of vehicle vb is within FK_FAR of a kept pair's box of owner va's triangle: va's
// entry of vb marked (the same word from every thread). `done`: the partner this thread
// marked last; returns it.
uint fkNear(uint va, uint vb, uint done)
{
    if (vb == done) return done;
    uint n = min(gNext[fkCover(va, 0u)], FK_COVER);
    for (uint j = 0u; j < FK_COVER; j++)
    {
        if (j < n && gNext[fkCover(va, 1u + j)] == vb) gNext[fkCover(va, 1u + j)] = vb | FK_NEAR;
    }
    return vb;
}

// Pair (gt, h) of owner va kept, at its share's next slot (fkBase). Past its share the list
// is void (fkAccCtl(2) 1), but the listing goes on counting, so that the next build's
// shares have every owner's whole count.
bool fkKeepPair(uint va, uint gt, uint h)
{
    uint n = uint(gAcc[fkAcc(va, FK_PAIRS)].add(1));
    if (n >= fkShare(va))
    {
        if (gAcc[fkAccCtl(2u)].load() == 0) gAcc[fkAccCtl(2u)].max(1);
        return true;
    }
    uint w = fkSlotAt(fkBase(va) + n);
    gNext[w] = gt;
    gNext[w + 1u] = h;
    return true;
}

// The build's walk: part `sub` of `subs` of triangle gt (any triangle of an owner with
// candidates, whatever its throttle or its state: the tests decide) against its owner's
// candidates of the build, by the grid's cells its padded box covers
// (vvTriangleCandsList's walk). A candidate inside the triangle's box grown by the range,
// CLIP_DEPTH, the triangle's and the candidate's classes' allowances and FK_BETWEEN is a
// kept pair (`test`: tested here instead, a build past the owner's room). The cells are
// those of the box for a calm candidate unless a wild one may be in reach (the owner's
// wild candidates' box). A bucket holds every cell that hashes to it: a candidate is
// taken from its own cell alone, so once (no search for a bucket met twice).
void fkWalk(uint gt, uint sub, uint subs, bool test)
{
    uint4 tn = gStatic[gP.triBase + gt];
    uint va = tn.w;
    uint cands = fkCands(va);
    if (cands == 0u) return;
    if (test && gNext[fkListed(gt)] == 0u) return;
    float3 oa = vvRel(va, va);
    float3 p0 = nf3(oa + gNodes[tn.x * 3u].xyz);
    float3 p1 = nf3(oa + gNodes[tn.y * 3u].xyz);
    float3 p2 = nf3(oa + gNodes[tn.z * 3u].xyz);
    bool wildTri = fkWild(gBeams[fkRef(tn.x)]) || fkWild(gBeams[fkRef(tn.y)]) || fkWild(gBeams[fkRef(tn.z)]);
    float base = asfloat(gStatic[va * 4u + 3u].z) + CLIP_DEPTH + (wildTri ? FK_WILD : FK_CALM) + fkBetween();
    float3 tlo = min(min(p0, p1), p2);
    float3 thi = max(max(p0, p1), p2);
    float padW = base + FK_WILD;
    float padC = base + FK_CALM;
    float3 wlo = float3(vvUnkey(gAcc[fkAcc(va, FK_WILD_BOX)].load()), vvUnkey(gAcc[fkAcc(va, FK_WILD_BOX + 1u)].load()), vvUnkey(gAcc[fkAcc(va, FK_WILD_BOX + 2u)].load()));
    float3 whi = float3(vvUnkey(gAcc[fkAcc(va, FK_WILD_BOX + 3u)].load()), vvUnkey(gAcc[fkAcc(va, FK_WILD_BOX + 4u)].load()), vvUnkey(gAcc[fkAcc(va, FK_WILD_BOX + 5u)].load()));
    // (No wild candidate: the box's keys are as the build's start left them.) The cells
    // are the kept pair's box's grown by FK_FAR: a candidate there and not kept makes its
    // vehicle near (fkNear).
    float farW = padW + FK_FAR;
    bool wildNear = gAcc[fkAcc(va, FK_WILD_BOX)].load() != VV_INT_MAX
                    && !(any(whi < tlo - float3(farW, farW, farW)) || any(wlo > thi + float3(farW, farW, farW)));
    float pad = (wildNear ? padW : padC) + FK_FAR;
    if (test)
    {
        // A build past an owner's room (FK_OVER): this pass's tests. Their cells are those
        // of the walk's own box (vvTriangleCandsList's: the triangle's box grown by its
        // range, its travel, the fastest candidate's travel and CLIP_DEPTH; room for the
        // two boxes' rounding), not the build's padded one: every pair today's walk lists
        // has its point there, and fkPairTry decides each exactly. (The padded box made
        // this pass several times a pass's cost: the worst step of a run.)
        float vmax = asfloat(uint(gAcc[flOwn(va) + FL_FAST].load()));
        float travel = gP.clipMem != 0u ? smul(clipTriSpeed(tn), STEP) : 0.0;
        float3 m3 = max(abs(tlo), abs(thi));
        pad = asfloat(gStatic[va * 4u + 3u].z) + nf(nf(travel + smul(vmax, STEP)) + CLIP_DEPTH) + 0.002
              + 4.0e-6 * max(m3.x, max(m3.y, m3.z));
    }
    // (A build that skips its listing, fkAccCtl(2) 2: none.)
    else if (gAcc[fkAccCtl(2u)].load() >= 2) return;
    uint nearDone = 0xFFFFFFFFu;
    float3 lo = tlo - float3(pad, pad, pad);
    float3 hi = thi + float3(pad, pad, pad);
    float3 f0 = floor(lo * FK_PER_CELL);
    float3 fs = floor(hi * FK_PER_CELL) - f0 + float3(1.0, 1.0, 1.0);
    bool every = !(all(fs <= float(FK_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(FK_MAX_CELLS))
                 || !(all(abs(f0) < 1.0e8));
    uint at = vvTriPairsAt(va);
    uint run = gP.vvRun + gNext[fkRun(va)];
    if (every)
    {
        for (uint e = sub, eStop = EB_END(e, cands, subs); e < eStop; e += subs)
        {
            float4 E = gBeams[run + e];
            float ph = fkWild(gBeams[fkRef(asuint(E.w))]) ? padW : padC;
            float pf = ph + FK_FAR;
            if (any(E.xyz < tlo - float3(pf, pf, pf)) || any(E.xyz > thi + float3(pf, pf, pf))) continue;
            if (!test) nearDone = fkNear(va, gStatic[gP.nodeInfoBase + asuint(E.w)].w, nearDone);
            if (any(E.xyz < tlo - float3(ph, ph, ph)) || any(E.xyz > thi + float3(ph, ph, ph))) continue;
            if (test)
            {
                fkPairTry(va, gt, asuint(E.w));
                continue;
            }
            if (!fkKeepPair(va, gt, asuint(E.w))) return;
        }
        return;
    }
    int3 c0 = int3(f0);
    int3 span = int3(fs);
    uint yz = uint(span.y * span.z);
    uint cells = uint(span.x) * yz;
    uint sorted = run + VV_PART * vvTotalCp();
    uint off = flOffAt(va);
    for (uint i = sub, iStop = EB_END(i, cells, subs); i < iStop; i += subs)
    {
        int3 cc = c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z)));
        uint b = gridBucket(cc);
        uint e0 = gNext[off + b];
        uint e1 = gNext[off + b + 1u];
        for (uint e = e0, eGuard = 0u; e < e1; e++)
        {
            EB_PASS(eGuard);
            float4 E = gBeams[sorted + e];
            if (any(int3(floor(E.xyz * FK_PER_CELL)) != cc)) continue;
            float ph = fkWild(gBeams[fkRef(asuint(E.w))]) ? padW : padC;
            float pf = ph + FK_FAR;
            if (any(E.xyz < tlo - float3(pf, pf, pf)) || any(E.xyz > thi + float3(pf, pf, pf))) continue;
            if (!test) nearDone = fkNear(va, gStatic[gP.nodeInfoBase + asuint(E.w)].w, nearDone);
            if (any(E.xyz < tlo - float3(ph, ph, ph)) || any(E.xyz > thi + float3(ph, ph, ph))) continue;
            if (test)
            {
                fkPairTry(va, gt, asuint(E.w));
                continue;
            }
            if (!fkKeepPair(va, gt, asuint(E.w))) return;
        }
    }
}
#endif

#ifdef EB_FLEET_FRONT
// Thread `tid` of the fleet-wide front's thread dispatches.
void fleetFront(uint tid, uint word)
{
    uint s = word & SPLIT_STEP;
    if ((word & (SPLIT_TESTS | SPLIT_LEVELS)) == 0u)
    {
        // (S) the plan's pairs. A pair outside the fleet-wide passes (no group, or a
        // larger group's): no partners here, whatever an earlier frame left.
        // (With the kept candidates: also triangle and contact point tid's motion since
        // the list's build.)
        if (fkOn()) fkTrack(tid);
        uint pairs = gNext[flAdjFirst(gP.vehicles)] / 2u;
        if (tid >= pairs) return;
        uint va = gNext[gP.vvPairs + 2u * tid];
        if (!flVehicleOn(va))
        {
            if ((gNext[flBitsAt(2u * tid)] & 4u) == 0u)
            {
                gNext[flBitsAt(2u * tid)] = 0u;
                gNext[flBitsAt(2u * tid + 1u)] = 0u;
            }
            return;
        }
        flScanPair(tid, flParity(va, s));
        return;
    }
    if ((word & SPLIT_TESTS) != 0u)
    {
        // (P) triangle tid's throttle step; contact point tid's gather. (With the kept
        // candidates: no gather in a pass that reuses them; a build's own, padded.)
        uint mode = fkMode();
        if (tid < gP.totalTris)
        {
            uint va = gStatic[gP.triBase + tid].w;
            if (flVehicleOn(va)) flTriangle(va, tid, flParity(va, s));
        }
        // (A build's record of node tid.)
        if (mode == FK_BUILD && tid < gP.totalNodes) gBeams[fkRef(tid)] = fkRefNow(tid, gStatic[gP.nodeInfoBase + tid].w);
        if (tid < gNext[flPrefixAt(4u, gP.vehicles)] && mode != FK_REUSE)
        {
            uint vb = flFind(4u, tid);
            if (mode == FK_BUILD) fkPoint(vb, tid - gNext[flPrefixAt(4u, vb)]);
            else if (flVehicleOn(vb)) flPoint(vb, tid - gNext[flPrefixAt(4u, vb)]);
        }
        return;
    }
    // The sorted copy: place tid of the fleet-wide run (of the build's, when it builds).
    if (fkMode() == FK_BUILD)
    {
        fkSortCand(tid);
        return;
    }
    if (tid >= gNext[flPrefixAt(0u, gP.vehicles)]) return;
    uint va = flFind(0u, tid);
    uint e = tid - flRun(va);
    if (e < flCands(va)) flSortCand(va, e);
}

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_fleet_front(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    fleetFront(gid.x * LANES + lid.x, gP.pad2);
    ebLoopCapFlush();
}
#endif

// The fleet's indirect arguments' words: the first buffer's (gArgs) and the second's.
static const uint FL_ARGS_ELEMENTS = 0u;
static const uint FL_ARGS_GRIDS = 4u;
static const uint FL_ARGS_SORT = 8u;
static const uint FL_ARGS_ONE = 12u;
static const uint FL_ARGS2_WALK = 0u;
static const uint FL_ARGS2_PAIRS = 4u;

#ifdef EB_FLEET_PLAN
// Threads the walk aims at: a listed triangle's cells or candidates in as many parts as
// fit (at most 64), the same pairs listed whatever the count.
static const uint FL_WALK_THREADS = 128u * LANES;
uint flWorkgroups(uint threads) { return (threads + LANES - 1u) / LANES; }

void fleetPlan(uint lane, uint word)
{
    uint s = word & SPLIT_STEP;
    uint nv = gP.vehicles;
    if ((word & (SPLIT_TESTS | SPLIT_LEVELS)) == 0u)
    {
        // After the scans: the owners' parts of the fleet-wide run, the prefix of their
        // offered counts; past the run's room no owner is offered anything this step
        // (FL_ERR_RUN: the pass is cut short, loudly).
        // (Whether the members' rest tracked this pass's nodes: fkTrackIdle reads the last
        // pass's offered total, which the prefix below replaces.)
        bool untracked = fkTrackInRest() && fkTrackIdle();
        flPrefix(0u, FL_OFFERED, lane);
        uint total = splitUniform(lane, gNext[flPrefixAt(0u, nv)]);
        if (total > vvRoomCp())
        {
            for (uint v = lane, vStop = EB_END(v, nv, LANES); v < vStop; v += LANES) gAcc[flOwn(v) + FL_OFFERED].store(0);
            if (lane == 0u) flError(FL_ERR_RUN);
            flPrefix(0u, FL_OFFERED, lane);
        }
        // Each owner's records of the step; whether any owner resets (its triangles'
        // throttles then change even when nothing is offered: gsUniformOpen, free here).
        if (lane == 0u) gsUniformOpen = 0u;
        sync();
        bool resets = false;
        for (uint v = lane, vStop = EB_END(v, nv, LANES); v < vStop; v += LANES)
        {
            if (!flVehicleOn(v)) continue;
            flOwnerPlan(v, flParity(v, s));
            if (gAcc[flOwn(v) + FL_RESET].load() != 0 || gAcc[flOwn(v) + FL_FIX].load() != 0) resets = true;
        }
        if (resets) gsUniformOpen = 1u;
        // (The kept candidates' checks of the pass, beside the owners' records: fkDecide
        // reads their result after this barrier.)
        // (Nothing offered: the pass decides nothing, fkDecide.)
        if (fkOn() && total != 0u) fkCheck(lane);
        sync();
        uint anyReset = gsUniformOpen;
        // The kept candidates' mode of the pass (PERF_FLEET_KEEP). A pass that reuses
        // them runs their tests where the grids' offsets were (the tests' module, the
        // step word alone) and nothing after; a build sizes the grids and the sorted copy
        // by its own run, and its tests are the listed pairs' dispatch.
        uint mode = FK_OFF;
        if (fkOn()) mode = fkDecide(lane, untracked);
        if (lane == 0u)
        {
            uint offered = gNext[flPrefixAt(0u, nv)];
            uint on = offered != 0u ? 1u : 0u;
            uint any = (offered != 0u || anyReset != 0u) ? 1u : 0u;
            uint grids = on * nv;
            uint sort = flWorkgroups(offered);
            if (mode == FK_REUSE)
            {
                grids = flWorkgroups(gNext[fkPrefix(nv)]);
                sort = 0u;
                on = 0u;
            }
            if (mode == FK_BUILD) sort = flWorkgroups(gNext[fkRun(nv)]);
            gArgs[FL_ARGS_ELEMENTS] = any * flWorkgroups(max(max(gP.totalTris, gNext[flPrefixAt(4u, nv)]), mode == FK_BUILD ? gP.totalNodes : 0u));
            gArgs[FL_ARGS_ELEMENTS + 1u] = 1u;
            gArgs[FL_ARGS_ELEMENTS + 2u] = 1u;
            gArgs[FL_ARGS_GRIDS] = grids;
            gArgs[FL_ARGS_GRIDS + 1u] = 1u;
            gArgs[FL_ARGS_GRIDS + 2u] = 1u;
            gArgs[FL_ARGS_SORT] = sort;
            gArgs[FL_ARGS_SORT + 1u] = 1u;
            gArgs[FL_ARGS_SORT + 2u] = 1u;
            gArgs[FL_ARGS_ONE] = on;
            gArgs[FL_ARGS_ONE + 1u] = 1u;
            gArgs[FL_ARGS_ONE + 2u] = 1u;
        }
        return;
    }
    // (The kept candidates' mode, uniform. A build: the walk a thread a part of every
    // triangle; then the kept pairs' prefix, their threads, and whether an owner's list
    // is past its room: then the pairs' dispatch walks again with its tests (the walk's own boxes),
    // a part of a listed triangle a thread, and the list is not kept.)
    uint mode = splitUniform(lane, fkMode());
    if (mode == FK_BUILD && (word & SPLIT_TESTS) != 0u)
    {
        // (The listed triangles' prefix, as today's front has it: a build past an owner's
        // room walks its tests over them.)
        flPrefix(1u, FL_LISTED, lane);
        if (lane == 0u)
        {
            // (No owner with a candidate: no walk.)
            uint any = 0u;
            for (uint v = 0u, vStop = EB_END(v, nv, 1u); v < vStop; v++) any += fkCands(v);
            uint subs = clamp(FL_WALK_THREADS / max(gP.totalTris, 1u), 1u, 64u);
            // A build that would be past an owner's room again: the candidates of the last
            // build past it (FK_SUBS above its 7 bits), and no fewer now. No listing (the
            // list would be void): the pass goes as one past the room, its tests in the
            // walk's own boxes. (A choice of path alone: those tests are exact either way;
            // repeated, a pile's every build attempt listed 32,767 pairs to throw away.)
            uint over = gNext[fkCtl(FK_SUBS)] >> 7u;
            bool skip = over != 0u && any >= over;
            if (skip) gAcc[fkAccCtl(2u)].store(2);
            gNext[fkCtl(FK_SUBS)] = subs | (over << 7u);
            gArgs2[FL_ARGS2_WALK] = any != 0u && !skip ? flWorkgroups(gP.totalTris * subs) : 0u;
            gArgs2[FL_ARGS2_WALK + 1u] = 1u;
            gArgs2[FL_ARGS2_WALK + 2u] = 1u;
        }
        return;
    }
    if (mode == FK_BUILD)
    {
        flPrefix(FK_ROW_PAIRS, FK_W_PAIRS, lane);
        if (lane == 0u)
        {
            uint most = 0u;
            for (uint v = 0u, vStop = EB_END(v, nv, 1u); v < vStop; v++) most = max(most, uint(gAcc[fkAcc(v, FK_PAIRS)].load()));
            uint total = gNext[fkPrefix(nv)];
            gNext[fkCtl(FK_MOST)] = max(gNext[fkCtl(FK_MOST)], most);
            uint cands = 0u;
            for (uint v = 0u, vStop = EB_END(v, nv, 1u); v < vStop; v++) cands += fkCands(v);
            uint subs = gNext[fkCtl(FK_SUBS)] & 127u;
            if (gAcc[fkAccCtl(2u)].load() != 0)
            {
                gNext[fkCtl(FK_MODE)] = FK_OVER;
                // (Its tests: each listed triangle in parts, as today's walk: FK_SUBS's low
                // bits now the parts. The candidates this time above them: the next build
                // with no fewer skips its listing.)
                uint listed = gNext[flPrefixAt(1u, nv)];
                uint parts = clamp(FL_WALK_THREADS / max(listed, 1u), 1u, 64u);
                total = listed * parts;
                // (Only when the fleet's room itself was past: an owner past its share
                // has its count in the next build's shares.)
                uint listedAll = 0u;
                for (uint v = 0u, vStop = EB_END(v, nv, 1u); v < vStop; v++) listedAll += min(uint(gAcc[fkAcc(v, FK_PAIRS)].load()), fkRoom());
                bool pastRoom = listedAll > fkRoom() || gAcc[fkAccCtl(2u)].load() == 2;
                gNext[fkCtl(FK_SUBS)] = parts | (pastRoom ? min(max(cands, 1u), 0x1FFFFFFu) << 7u : 0u);
                // (Past the fleet's room, or the third time in a row: the backoff. Past an
                // owner's share alone: a build again two passes on, with the shares the
                // counts now ask for.)
                uint streak = gNext[fkCtl(FK_WAIT)] >> 16u;
                if (pastRoom || streak >= 3u) fkBackoff();
                else gNext[fkCtl(FK_WAIT)] = ((streak + 1u) << 16u) | 2u;
            }
            else
            {
                gNext[fkCtl(FK_SUBS)] = subs;
                // (Valid, and the pass it was built in: FK_PASS counts from 1 here.)
                gNext[fkCtl(FK_VALID)] = max(gNext[fkCtl(FK_PASS)], 1u);
                gNext[fkCtl(FK_WAIT)] = 0u;
                gNext[fkCtl(FK_BUILDS)] += 1u;
                gNext[fkCtl(FK_KEPT)] += total;
            }
            gArgs2[FL_ARGS2_PAIRS] = flWorkgroups(total);
            gArgs2[FL_ARGS2_PAIRS + 1u] = 1u;
            gArgs2[FL_ARGS2_PAIRS + 2u] = 1u;
        }
        return;
    }
    if ((word & SPLIT_TESTS) != 0u)
    {
        // After the gather: the listed triangles' prefix and the walk's threads.
        flPrefix(1u, FL_LISTED, lane);
        if (lane == 0u)
        {
            uint listed = gNext[flPrefixAt(1u, nv)];
            uint subs = clamp(FL_WALK_THREADS / max(listed, 1u), 1u, 64u);
            gNext[flCtlNext(0u)] = subs;
            gArgs2[FL_ARGS2_WALK] = flWorkgroups(listed * subs);
            gArgs2[FL_ARGS2_WALK + 1u] = 1u;
            gArgs2[FL_ARGS2_WALK + 2u] = 1u;
        }
        return;
    }
    // After the walk: the listed pairs' prefix and the pairs' threads (an owner past its
    // list's room tests its listed triangles inline there, a thread each).
    flPrefix(2u, FL_PAIRS, lane);
    if (lane == 0u)
    {
        uint over = 0u;
        for (uint v = 0u, vStop = EB_END(v, nv, 1u); v < vStop; v++)
        {
            if (uint(gAcc[flOwn(v) + FL_PAIRS].load()) > TRI_PAIR_CAP) over = 1u;
        }
        gNext[flCtlNext(1u)] = over;
        gArgs2[FL_ARGS2_PAIRS] = flWorkgroups(max(gNext[flPrefixAt(2u, nv)], over != 0u ? gNext[flPrefixAt(1u, nv)] : 0u));
        gArgs2[FL_ARGS2_PAIRS + 1u] = 1u;
        gArgs2[FL_ARGS2_PAIRS + 2u] = 1u;
    }
}

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_fleet_plan(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    // (The levels across the GPU's plan: stage 3c below.)
    if ((gP.pad2 & 0xF0000000u) == SPLIT_LV)
    {
        fleetLevelsPlan(lid.x, gP.pad2);
        ebLoopCapFlush();
        return;
    }
    // (PERF_FLEET_FUSE: the pairs' scans in this workgroup first; not with the kept
    // candidates, whose scans' dispatch also tracks every node: fuseScansOn.)
    if (fuseScansOn() && (gP.pad2 & (SPLIT_TESTS | SPLIT_LEVELS)) == 0u) fuseScans(lid.x, gP.pad2 & SPLIT_STEP);
    fleetPlan(lid.x, gP.pad2);
    ebLoopCapFlush();
}
#endif

#ifdef EB_VV_GLOBAL
// The tests' module over every owner (PERF_FLEET_FRONT). The step word alone: workgroup g
// is vehicle g's, its grid's offsets (every lane; uniform). SPLIT_TESTS: thread tid of the
// walk (a part of a listed triangle); SPLIT_LEVELS: of the listed pairs.
void fleetWideTests(uint g, uint lane, uint word)
{
    uint nv = gP.vehicles;
    uint tid = g * LANES + lane;
    if ((word & (SPLIT_TESTS | SPLIT_LEVELS)) == 0u)
    {
        // (The kept candidates, the mode the word's top bits: a pass that reuses them
        // tests them here, a thread a kept pair; a build's grids are of its own run.)
        uint mode = fkMode();
        uint count = 0u;
        if (g < nv) count = mode == FK_BUILD ? gNext[fkRun(g + 1u)] - gNext[fkRun(g)] : uint(gAcc[flOwn(g) + FL_OFFERED].load());
        uint packed = splitUniform(lane, (mode << 30) | min(count, 0x3FFFFFFFu));
        if ((packed >> 30) == FK_REUSE)
        {
            fkTest(tid);
            return;
        }
        if ((packed & 0x3FFFFFFFu) != 0u)
        {
            flOffsets(g, lane);
            // (PERF_FLEET_FUSE: the owner's sorted copy here, explicit_beams_fleet_hitlat.slang;
            // a build's of its own run.)
            if (flFuseOn()) fuseSortCands(g, lane, (packed >> 30) == FK_BUILD);
        }
        return;
    }
    uint listed = gNext[flPrefixAt(1u, nv)];
    {
        uint mode = fkMode();
        if (mode == FK_BUILD || mode == FK_OVER)
        {
            if ((word & SPLIT_LEVELS) == 0u)
            {
                uint subs = gNext[fkCtl(FK_SUBS)] & 127u;
                if (subs != 0u && tid < gP.totalTris * subs) fkWalk(tid / subs, tid % subs, subs, false);
            }
            else if (mode == FK_BUILD) fkTest(tid);
            else
            {
                // (Past an owner's room: listed triangle q's part, as flWalk.)
                uint subs = gNext[fkCtl(FK_SUBS)] & 127u;
                if (subs == 0u || tid >= listed * subs) return;
                uint q = tid / subs;
                uint va = flFind(1u, q);
                uint triOff = gStatic[va * 4u + 1u].x;
                fkWalk(triOff + gNext[vvTestedAt(triOff + q - gNext[flPrefixAt(1u, va)])], tid % subs, subs, true);
            }
            return;
        }
    }
    if ((word & SPLIT_LEVELS) == 0u)
    {
        uint subs = gNext[flCtlNext(0u)];
        if (subs == 0u || tid >= listed * subs) return;
        uint q = tid / subs;
        uint va = flFind(1u, q);
        flWalk(va, q - gNext[flPrefixAt(1u, va)], tid % subs, subs);
        return;
    }
    if (tid < gNext[flPrefixAt(2u, nv)])
    {
        uint va = flFind(2u, tid);
        flPairTest(va, tid - gNext[flPrefixAt(2u, va)]);
    }
    if (gNext[flCtlNext(1u)] != 0u && tid < listed)
    {
        uint va = flFind(1u, tid);
        if (uint(gAcc[flOwn(va) + FL_PAIRS].load()) > TRI_PAIR_CAP) flWalkInline(va, tid - gNext[flPrefixAt(1u, va)], 0u, 1u);
    }
}
#endif

// Stage 3c: the levels across the GPU (PERF_FLEET_LEVELS).
#include "explicit_beams_fleet_levels.slang"
// One hit's latency in the levels; the front's fused dispatches.
#include "explicit_beams_fleet_hitlat.slang"
// A group's levels with each hit worked out once.
#include "explicit_beams_fleet_once.slang"
