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
// §8.8) ---
// Each substep the front gathers every partner's contact points, grids and sorts them and
// walks every listed triangle's cells to list the (triangle, point) pairs its box test
// passes: in a pile over 10,000 pairs from 4,000 triangles for 0 to 20 hits, and almost
// nothing has moved since the substep before. Here the pairs are a Verlet list:
//   the build   every (triangle, contact point) of two vehicles the plan pairs whose
//               point is in the triangle's box grown by its range, CLIP_DEPTH and
//               FK_MARGIN, whatever the throttle, the partners' bits or the owner's
//               box say this substep (fkPoint, fkWalk: the front's own grid, padded);
//               each vehicle's nodes' positions and its origin recorded (fkRef, fkOrigin)
//   each pass   the kept pairs tested (fkPairTry): the triangle listed by its throttle
//               step this pass, the vehicles partners this pass, the point in the owner's
//               box (flPoint's test), in the triangle's box (vvInBox), then vvFind: the
//               tests the walk and the listed pairs' test run today, on today's values
//   the bound   per vehicle the farthest a tracked node (a triangle's, a contact point)
//               has moved since the build relative to its first node, and its fastest
//               (fkTrack); per pair of partners of the pass the two bounds, their first
//               nodes' relative motion with the origins' (the rebase), and both fastest
//               speeds' travel over a step (anti-clip's pads): under FK_MARGIN with
//               room for rounding, or the list is built again (fkDecide)
// A pair today's walk would list has its point within the triangle's box grown by the
// range and at most CLIP_DEPTH plus both nodes' travel; at the build it was nearer than
// that plus the pair's relative motion since, so within the build's box: the kept list
// holds it. The pairs tested are therefore today's, the hits and the marks the same.
// What is not kept: a pass whose nodes are too fast for the margin, a build past an
// owner's room (FK_CAP pairs: its triangles test inline that pass) take today's path for
// FK_BACKOFF passes; the list's first two words an owner are a signature, so another
// path's list in the same region (the flag off for a frame) ends the kept one.
static const float FK_MARGIN = 0.1;
static const uint FK_CAP = TRI_PAIR_CAP - 1u;
static const uint FK_MAGIC = 0xFFFFFFFFu;
static const uint FK_BACKOFF = 136u;
// The pass's mode (fkCtl(FK_MODE), written by the first plan).
static const uint FK_OFF = 0u; // today's front
static const uint FK_REUSE = 1u; // the kept pairs' tests alone
static const uint FK_BUILD = 2u; // the padded front, then the kept pairs' tests
static const uint FK_OVER = 3u; // a build past an owner's room: its tests in the padded walk
// fkCtl's words.
static const uint FK_MODE = 0u;
static const uint FK_VALID = 1u;
static const uint FK_EPOCH = 2u;
static const uint FK_WAIT = 3u;
static const uint FK_SUBS = 4u;
static const uint FK_BUILDS = 5u; // measurement: builds, reused passes, pairs kept (summed over
static const uint FK_REUSED = 6u; // builds), passes on today's path after a build failed, the
static const uint FK_KEPT = 7u; // most pairs an owner kept
static const uint FK_FELL = 8u;
static const uint FK_MOST = 9u;
static const uint FK_BY_MOTION = 10u; // builds because a pair's bound passed the margin,
static const uint FK_BY_COVER = 11u; // … or a pair the build did not cover (or no list)
static const uint FK_DEV_MOST = 12u; // the largest motion bound a vehicle had at such a build (float bits)
static const uint FK_VMAX_MOST = 13u; // the fastest tracked node seen (float bits)
// flPrefix's rows and counts of the kept candidates.
static const uint FK_ROW_RUN = 16u;
static const uint FK_ROW_PAIRS = 17u;
static const uint FK_W_RUN = 100u;
static const uint FK_W_PAIRS = 101u;

bool fkOn() { return (gP.perfFlags & PERF_FLEET_KEEP) != 0u && gP.vvTriPairs != 0u; }
uint fkMode() { return fkOn() ? gNext[fkCtl(FK_MODE)] : FK_OFF; }
// Owner v's room in the build's candidate run: its covered partners' contact points.
uint fkOffered(uint v)
{
    uint n = min(gNext[fkCover(v, 0u)], FK_COVER);
    uint sum = 0u;
    for (uint j = 0u; j < FK_COVER; j++)
    {
        if (j < n) sum += vvCpCount(gNext[fkCover(v, 1u + j)]);
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
#ifdef EB_VV_GROUPS
    // The fleet-wide dispatches' counts back to zero (PERF_FLEET_FRONT: fleetPlan wrote
    // them for this pass; without a group nothing does, and they must run no thread).
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

// A vehicle whose contact is the fleet-wide passes' this frame: a member of a group the
// small groups' levels take (a larger group's pass, vvPass, is its own).
bool flVehicleOn(uint v)
{
    uint4 fl = gMisc[vvRec(v, VV_FLAGS)];
    if (fl.y == 0u) return false;
    uint k = gMisc[gP.vvGroups + fl.z].y;
    return k != 0u && k <= VV_PART;
}
// The step's parity for vehicle v's boxes: its group's first member's step count (as the
// group's pass has it), kept at the groups' frame start.
uint flParity(uint v, uint s) { return (gNext[flVehAt(0u, v)] + s) & 1u; }

// The last vehicle whose prefix (row r) is at most x: the owner of place x of a
// fleet-wide list (the vehicles without a part have the next one's start).
// (Rows FK_ROW_RUN and FK_ROW_PAIRS: the kept candidates' own, fkRun and fkPrefix.)
uint flRowAt(uint r, uint v) { return r == FK_ROW_RUN ? fkRun(v) : (r == FK_ROW_PAIRS ? fkPrefix(v) : flPrefixAt(r, v)); }
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
    if (word == FK_W_PAIRS) return min(uint(gAcc[fkAcc(v, FK_PAIRS)].load()), FK_CAP);
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

// A float's bits as an ordered key of a bound: itself when finite and not negative, else
// +inf's (NaN and negative values: never under a margin).
uint fkBits(float x)
{
    uint b = asuint(x);
    return b >= 0x7F800000u ? 0x7F800000u : b;
}

#ifdef EB_FLEET_FRONT
// Node n of vehicle v: how far it has moved since the build relative to the vehicle's
// first node (their positions in the vehicle's frame: a rebase shifts both alike), and
// its speed, into the vehicle's bounds.
void fkTrackNode(uint n, uint v)
{
    uint r = gStatic[v * 4u].x;
    float3 d = (gNodes[n * 3u].xyz - gBeams[fkRef(n)].xyz) - (gNodes[r * 3u].xyz - gBeams[fkRef(r)].xyz);
    float3 w = gNodes[n * 3u + 1u].xyz;
    gAcc[fkAcc(v, FK_DEV)].max(int(fkBits(sqrt(dot(d, d)))));
    gAcc[fkAcc(v, FK_VMAX)].max(int(fkBits(sqrt(dot(w, w)))));
}
// Thread tid of the scans' dispatch: triangle tid's nodes, contact point tid.
void fkTrack(uint tid)
{
    if (tid < gP.totalTris)
    {
        uint4 tn = gStatic[gP.triBase + tid];
        if (flVehicleOn(tn.w))
        {
            fkTrackNode(tn.x, tn.w);
            fkTrackNode(tn.y, tn.w);
            fkTrackNode(tn.z, tn.w);
        }
    }
    if (tid < gNext[flPrefixAt(4u, gP.vehicles)])
    {
        uint vb = flFind(4u, tid);
        if (flVehicleOn(vb)) fkTrackNode(gLists[gStatic[gP.vvStatic + vb].x + (tid - gNext[flPrefixAt(4u, vb)])], vb);
    }
}

// The build's records of triangle gt: its nodes' positions.
void fkTriRefs(uint gt)
{
    uint4 tn = gStatic[gP.triBase + gt];
    gBeams[fkRef(tn.x)] = float4(gNodes[tn.x * 3u].xyz, 0.0);
    gBeams[fkRef(tn.y)] = float4(gNodes[tn.y * 3u].xyz, 0.0);
    gBeams[fkRef(tn.z)] = float4(gNodes[tn.z * 3u].xyz, 0.0);
}
// The build's gather of contact point i of vehicle vb: its position recorded; into the
// run of every owner the build covers it for (its vehicle's plan partners) whose box,
// grown by its range (flOwnerPlan's), by what the build's triangle boxes are grown by and
// by room for rounding, holds it: every triangle of the owner is inside the owner's box.
void fkPoint(uint vb, uint i)
{
    uint h = gLists[gStatic[gP.vvStatic + vb].x + i];
    float3 loc = gNodes[h * 3u].xyz;
    gBeams[fkRef(h)] = float4(loc, 0.0);
    float3 x = nf3(asfloat(gMisc[vb].xyz) + loc);
    uint n = min(gNext[fkCover(vb, 0u)], FK_COVER);
    for (uint j = 0u; j < FK_COVER; j++)
    {
        if (j >= n) break;
        uint va = gNext[fkCover(vb, 1u + j)];
        float4 B0 = gBeams[flBoxAt(va)];
        float4 B1 = gBeams[flBoxAt(va) + 1u];
        float3 m3 = max(max(abs(B0.xyz), abs(B1.xyz)), abs(x));
        float ex = CLIP_DEPTH + FK_MARGIN + 0.002 + 4.0e-6 * max(m3.x, max(m3.y, m3.z));
        float3 e3 = float3(ex, ex, ex);
        if (any(x < B0.xyz - e3) || any(x > B1.xyz + e3)) continue;
        uint base = gNext[fkRun(va)];
        uint ci = uint(gAcc[flOwn(va) + FL_CANDS].add(1));
        if (ci >= gNext[fkRun(va + 1u)] - base) continue;
        float4 E = float4(nf3(vvRel(vb, va) + loc), asfloat(h));
        gBeams[gP.vvRun + base + ci] = E;
        uint b = gridBucket(int3(floor(E.xyz / vvGridCell(va))));
        gNext[gP.vvHits + 4u * VV_PART * vvTotalCp() + base + ci] = b | (uint(gAcc[flGridCount(va) + b].add(1)) << 10);
    }
}
// Place tid of the build's run to its place in the sorted copy (flSortCand's).
void fkSortCand(uint tid)
{
    uint nv = gP.vehicles;
    if (tid >= gNext[fkRun(nv)]) return;
    uint va = flFind(FK_ROW_RUN, tid);
    uint base = gNext[fkRun(va)];
    uint e = tid - base;
    if (e >= fkCands(va)) return;
    uint br = gNext[gP.vvHits + 4u * VV_PART * vvTotalCp() + base + e];
    gBeams[gP.vvRun + VV_PART * vvTotalCp() + base + gNext[flOffAt(va) + (br & (GRID_BUCKETS - 1u))] + (br >> 10)] = gBeams[gP.vvRun + base + e];
}
#endif

#ifdef EB_FLEET_PLAN
// Pair p of the plan this pass: 0 the kept list holds every pair today's walk would list
// for it (or its vehicles are not partners this pass), 1 it may not, its nodes having
// moved (a build), 2 the build did not cover it (a build), 3 its nodes are too fast for
// any list of this margin (today's path).
uint fkPairCheck(uint p)
{
    uint bits = gNext[flBitsAt(2u * p)];
    if ((bits & 5u) != 1u) return 0u;
    uint va = gNext[gP.vvPairs + 2u * p];
    uint vb = gNext[gP.vvPairs + 2u * p + 1u];
    uint ra = gStatic[va * 4u].x;
    uint rb = gStatic[vb * 4u].x;
    float3 la = gNodes[ra * 3u].xyz;
    float3 lb = gNodes[rb * 3u].xyz;
    float3 rel = vvRel(vb, va);
    float3 m3 = max(max(abs(la), abs(lb)), abs(rel));
    float slop = 0.002 + 1.0e-5 * max(m3.x, max(m3.y, m3.z));
    // Anti-clip's pads at most: both vehicles' fastest tracked nodes' travel over a step.
    float travel = (asfloat(uint(gAcc[fkAcc(va, FK_VMAX)].load())) + asfloat(uint(gAcc[fkAcc(vb, FK_VMAX)].load()))) * STEP + slop;
    if (fkBits(travel) > asuint(FK_MARGIN)) return 3u;
    bool covered = false;
    uint n = min(gNext[fkCover(va, 0u)], FK_COVER);
    for (uint j = 0u; j < FK_COVER; j++)
    {
        if (j < n && gNext[fkCover(va, 1u + j)] == vb) covered = true;
    }
    if (!covered) return 2u;
    // The first nodes' relative motion since the build, in the owner's frame as the tests
    // have it (a point of vb at vvRel(vb, va) + its position).
    float3 relThen = asfloat(uint3(gNext[fkOrigin(vb)], gNext[fkOrigin(vb) + 1u], gNext[fkOrigin(vb) + 2u]))
                     - asfloat(uint3(gNext[fkOrigin(va)], gNext[fkOrigin(va) + 1u], gNext[fkOrigin(va) + 2u]));
    float3 c = ((lb - gBeams[fkRef(rb)].xyz) - (la - gBeams[fkRef(ra)].xyz)) + (rel - relThen);
    float moved = asfloat(uint(gAcc[fkAcc(va, FK_DEV)].load())) + asfloat(uint(gAcc[fkAcc(vb, FK_DEV)].load())) + sqrt(dot(c, c)) + travel;
    return fkBits(moved) > asuint(FK_MARGIN) ? 1u : 0u;
}

// The first plan's decision of the pass (every lane; uniform): the mode, with a build's
// records (each vehicle's covered partners, origin, first node, signature, its part of
// the build's run). Returns the mode.
uint fkDecide(uint lane)
{
    uint nv = gP.vehicles;
    uint epoch = gNext[fkCtl(FK_EPOCH)];
    uint code = 0u;
    uint pairs = gNext[flAdjFirst(nv)] / 2u;
    for (uint p = lane, pStop = EB_END(p, pairs, LANES); p < pStop; p += LANES) code = max(code, fkPairCheck(p));
    // (Another path's list in an owner's region: the kept one is gone.)
    for (uint v = lane, vStop = EB_END(v, nv, LANES); v < vStop; v += LANES)
    {
        if (gNext[vvTriPairsAt(v)] != FK_MAGIC || gNext[vvTriPairsAt(v) + 1u] != epoch) code = max(code, 2u);
        // (Measurement: the vehicles' largest bounds of the pass.)
        gAcc[fkAccCtl(1u)].max(gAcc[fkAcc(v, FK_DEV)].load());
        gAcc[fkAccCtl(2u)].max(gAcc[fkAcc(v, FK_VMAX)].load());
    }
    if (code != 0u) gAcc[fkAccCtl(0u)].max(int(code));
    sync();
    uint mode = FK_OFF;
    if (lane == 0u)
    {
        uint c = uint(gAcc[fkAccCtl(0u)].load());
        gAcc[fkAccCtl(0u)].store(0);
        uint wait = gNext[fkCtl(FK_WAIT)];
        if (gNext[flPrefixAt(0u, nv)] != 0u)
        {
            if (wait != 0u) gNext[fkCtl(FK_WAIT)] = wait - 1u;
            else if (c == 0u && gNext[fkCtl(FK_VALID)] != 0u) mode = FK_REUSE;
            else if (c < 3u) mode = FK_BUILD;
            if (mode == FK_BUILD && c == 1u)
            {
                gNext[fkCtl(FK_BY_MOTION)] += 1u;
                gNext[fkCtl(FK_DEV_MOST)] = max(gNext[fkCtl(FK_DEV_MOST)], uint(gAcc[fkAccCtl(1u)].load()));
            }
            if (mode == FK_BUILD && c == 2u) gNext[fkCtl(FK_BY_COVER)] += 1u;
            gNext[fkCtl(FK_VMAX_MOST)] = max(gNext[fkCtl(FK_VMAX_MOST)], uint(gAcc[fkAccCtl(2u)].load()));
        }
        gAcc[fkAccCtl(1u)].store(0);
        gAcc[fkAccCtl(2u)].store(0);
    }
    mode = splitUniform(lane, mode);
    for (uint v = lane, vStop = EB_END(v, nv, LANES); v < vStop; v += LANES)
    {
        gAcc[fkAcc(v, FK_DEV)].store(0);
        gAcc[fkAcc(v, FK_VMAX)].store(0);
    }
    if (mode == FK_BUILD)
    {
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
            uint r = gStatic[v * 4u].x;
            gBeams[fkRef(r)] = float4(gNodes[r * 3u].xyz, 0.0);
            gAcc[fkAcc(v, FK_PAIRS)].store(0);
            gNext[vvTriPairsAt(v)] = FK_MAGIC;
            gNext[vvTriPairsAt(v) + 1u] = epoch + 1u;
        }
        if (tooMany) gsUniformOpen = 1u;
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
            if (mode == FK_OFF) gNext[fkCtl(FK_WAIT)] = FK_BACKOFF;
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
    uint at = vvTriPairsAt(va) + 2u * (1u + tid - gNext[fkPrefix(va)]);
    fkPairTry(va, gNext[at], gNext[at + 1u]);
}

// The build's walk: part `sub` of `subs` of triangle gt (any triangle of an owner with
// candidates, whatever its throttle or its state: the tests decide) against its owner's
// candidates of the build, by the grid's cells its padded box covers
// (vvTriangleCandsList's walk); each candidate in the box is a kept pair (`test`: tested
// here instead, a build past the owner's room).
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
    float pad = asfloat(gStatic[va * 4u + 3u].z) + CLIP_DEPTH + FK_MARGIN;
    float3 lo = min(min(p0, p1), p2) - float3(pad, pad, pad);
    float3 hi = max(max(p0, p1), p2) + float3(pad, pad, pad);
    float cell = vvGridCell(va);
    float3 f0 = floor(lo / cell);
    float3 fs = floor(hi / cell) - f0 + float3(1.0, 1.0, 1.0);
    bool every = !(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS))
                 || !(all(abs(f0) < 1.0e8));
    uint at = vvTriPairsAt(va);
    uint run = gP.vvRun + gNext[fkRun(va)];
    if (every)
    {
        for (uint e = sub, eStop = EB_END(e, cands, subs); e < eStop; e += subs)
        {
            float4 E = gBeams[run + e];
            if (any(E.xyz < lo) || any(E.xyz > hi)) continue;
            if (test)
            {
                fkPairTry(va, gt, asuint(E.w));
                continue;
            }
            uint n = uint(gAcc[fkAcc(va, FK_PAIRS)].add(1));
            if (n < FK_CAP)
            {
                gNext[at + 2u * (n + 1u)] = gt;
                gNext[at + 2u * (n + 1u) + 1u] = asuint(E.w);
            }
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
        uint b = gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z))));
        bool dup = false;
        int3 cj = int3(0, 0, 0);
        for (uint j = 0u, jStop = EB_END(j, i, 1u); j < jStop; j++)
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
        uint e0 = gNext[off + b];
        uint e1 = gNext[off + b + 1u];
        for (uint e = e0, eGuard = 0u; e < e1; e++)
        {
            EB_PASS(eGuard);
            float4 E = gBeams[sorted + e];
            if (any(E.xyz < lo) || any(E.xyz > hi)) continue;
            if (test)
            {
                fkPairTry(va, gt, asuint(E.w));
                continue;
            }
            uint n = uint(gAcc[fkAcc(va, FK_PAIRS)].add(1));
            if (n < FK_CAP)
            {
                gNext[at + 2u * (n + 1u)] = gt;
                gNext[at + 2u * (n + 1u) + 1u] = asuint(E.w);
            }
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
            if (mode == FK_BUILD) fkTriRefs(tid);
        }
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
        flPrefix(0u, FL_OFFERED, lane);
        uint total = splitUniform(lane, gNext[flPrefixAt(0u, nv)]);
        if (total > VV_PART * vvTotalCp())
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
        sync();
        uint anyReset = gsUniformOpen;
        // The kept candidates' mode of the pass (PERF_FLEET_KEEP). A pass that reuses
        // them runs their tests where the grids' offsets were (the tests' module, the
        // step word alone) and nothing after; a build sizes the grids and the sorted copy
        // by its own run, and its tests are the listed pairs' dispatch.
        uint mode = FK_OFF;
        if (fkOn()) mode = fkDecide(lane);
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
            gArgs[FL_ARGS_ELEMENTS] = any * flWorkgroups(max(gP.totalTris, gNext[flPrefixAt(4u, nv)]));
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
    // is past its room: then the pairs' dispatch is the padded walk again with its tests,
    // a thread a triangle, and the list is not kept.)
    uint mode = splitUniform(lane, fkMode());
    if (mode == FK_BUILD && (word & SPLIT_TESTS) != 0u)
    {
        if (lane == 0u)
        {
            uint subs = clamp(FL_WALK_THREADS / max(gP.totalTris, 1u), 1u, 64u);
            gNext[fkCtl(FK_SUBS)] = subs;
            gArgs2[FL_ARGS2_WALK] = flWorkgroups(gP.totalTris * subs);
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
            if (most > FK_CAP)
            {
                gNext[fkCtl(FK_MODE)] = FK_OVER;
                gNext[fkCtl(FK_WAIT)] = FK_BACKOFF;
                total = gP.totalTris;
            }
            else
            {
                gNext[fkCtl(FK_VALID)] = 1u;
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
        if ((packed & 0x3FFFFFFFu) != 0u) flOffsets(g, lane);
        return;
    }
    uint listed = gNext[flPrefixAt(1u, nv)];
    {
        uint mode = fkMode();
        if (mode == FK_BUILD || mode == FK_OVER)
        {
            if ((word & SPLIT_LEVELS) == 0u)
            {
                uint subs = gNext[fkCtl(FK_SUBS)];
                if (subs != 0u && tid < gP.totalTris * subs) fkWalk(tid / subs, tid % subs, subs, false);
            }
            else if (mode == FK_BUILD) fkTest(tid);
            else if (tid < gP.totalTris) fkWalk(tid, 0u, 1u, true);
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
