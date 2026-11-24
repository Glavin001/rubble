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
    for (uint e = gNext[flAdjFirst(v)], eStop = EB_END(314, e, e1, 1u); e < eStop; e++)
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

// The walk with each candidate's position tested first and its big triangles shared out
// (explicit_plan::PERF_FLEET_WALK_BOX; the host keeps the switch in bit 30 of the word
// after beamOrder: the fleet copies' perfFlags have no free bit).
// vehicle-perf-first-principles.md §8.56.
bool flWalkBox() { return (gP.beamOrderPad0 & 0x40000000u) != 0u; }
// A listed triangle whose box covers more cells than the walk takes by cell tests every
// candidate of its owner (a torn panel's triangle, metres long: a pile of 20 cars has 16
// to 42 a pass, 5,000 candidates each), which was three threads' chains of thousands of
// loads: the dispatch waited for them (20 of a 65 ms frame). With the switch such a
// triangle is listed by the throttles' dispatch (flBigHint: its mark in fkListed's bit 1,
// its place in the fleet's list, flListAt) and the walk's dispatch has FL_BIG_PARTS more
// threads for each, a part of the owner's candidates a thread (flWalkBoxed's `helper`).
// The stored mark alone decides who tests a triangle, never a value two threads work out
// apart: a marked triangle is tested against every candidate of its owner by its own
// threads and skipped by the walk's (whatever the walk's own count of cells, which also
// takes the owner's fastest candidate, would say: the cells only leave out candidates
// that fail the box test), an unmarked one is the walk's as before. The same (triangle,
// candidate) pairs, each once; which triangles are marked is a choice of schedule.
static const uint FL_CTL_BIG = 4u; // gAcc, flCtl: the pass's big triangles so far
static const uint FL_NEXT_BIG = 2u; // gNext, flCtlNext: their count as the walk's plan sized it
static const uint FL_BIG_PARTS = 128u;
bool flBigHint(uint gt, uint va)
{
    uint4 tn = gStatic[gP.triBase + gt];
    float3 oa = vvRel(va, va);
    float3 p0 = nf3(oa + gNodes[tn.x * 3u].xyz);
    float3 p1 = nf3(oa + gNodes[tn.y * 3u].xyz);
    float3 p2 = nf3(oa + gNodes[tn.z * 3u].xyz);
    float pad = asfloat(gStatic[va * 4u + 3u].z) + CLIP_DEPTH + 0.05;
    float cell = vvGridCell(va);
    float3 f0 = floor((min(min(p0, p1), p2) - float3(pad, pad, pad)) / cell);
    float3 fs = floor((max(max(p0, p1), p2) + float3(pad, pad, pad)) / cell) - f0 + float3(1.0, 1.0, 1.0);
    return !(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS)) || !(all(abs(f0) < 1.0e8));
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
    // (And the walk's shared-out triangles: flWalkBox.)
    bool keep = fkOn() || flWalkBox();
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
    for (uint e = gNext[flAdjFirst(vb)], eStop = EB_END(315, e, e1, 1u); e < eStop; e++)
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
    for (uint q = 0u, qStop = EB_END(316, q, per, 1u); q < qStop; q++) sum += uint(gAcc[cnt + lane * per + q].load());
    gsPart[lane] = sum;
    sync();
    if (lane < GRID_BLOCKS)
    {
        uint run = 0u;
        for (uint q = 0u, qStop = EB_END(317, q, block, 1u); q < qStop; q++)
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
        for (uint q = 0u, qStop = EB_END(318, q, per, 1u); q < qStop; q++)
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

// vvTriangleCandsList for an owner of the fleet-wide passes (gFl 1, gFlV va), the same
// pairs listed. A candidate's box test (vvInBox) loads its vehicle and its velocity, two
// loads that depend on the candidate's own; in a pile a triangle's cells' buckets hold
// thousands of candidates of other cells (9.2 million visited a frame of 20 cars for 700
// inside a box). Here a candidate is first tested against the triangle's box grown by the
// largest pad any candidate of the owner can have (the walk's own `pad`, from the owner's
// fastest candidate: vvInBox's pad of a candidate is the same expression of its own speed,
// which is at most that one, and each operation is monotone), so a candidate outside it
// fails vvInBox: the same candidates pass, without the two loads for the others. The
// first test's box is 0.1 mm and a millionth of the coordinates larger still: with fast
// arithmetic the two pads may round apart in their last place (a fused multiply-add in
// one), which that room is a thousand times; vvInBox alone decides inside it.
// `helper`: a thread of a marked triangle's own (part `sub` of FL_BIG_PARTS of the owner's
// candidates); the walk's threads leave a marked triangle to them.
void flWalkBoxed(uint va, uint gt, uint sub, uint subs, uint cands, float r, bool helper)
{
    if (!helper && (gNext[fkListed(gt)] & 2u) != 0u) return;
    float cell = vvGridCell(va);
    VvTri t = vvTriangle(gt, vvRel(va, va), r);
    if (any(t.lo > t.hi)) return;
    float vmax = vvSpeedAt(0u);
    float pad = nf(nf(t.travel + smul(vmax, STEP)) + CLIP_DEPTH);
    float3 lo = ssub3(t.lo, float3(pad, pad, pad));
    float3 hi = sadd3(t.hi, float3(pad, pad, pad));
    float3 room = float3(1.0e-4, 1.0e-4, 1.0e-4) + 1.0e-6 * max(abs(lo), abs(hi));
    float3 blo = lo - room;
    float3 bhi = hi + room;
    float3 f0 = floor(lo / cell);
    float3 fs = floor(hi / cell) - f0 + float3(1.0, 1.0, 1.0);
    bool every = helper || !(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS))
                 || !(all(abs(f0) < 1.0e8));
    uint at = vvTriPairsAt(va);
    bool cand = false;
    if (every)
    {
        uint run = vvRunSmall(0u, 0u, 0u);
        vvVisit(sub < cands ? (cands - sub + subs - 1u) / subs : 0u);
        for (uint e = sub, eStop = EB_END(640, e, cands, subs); e < eStop; e += subs)
        {
            float4 E = gBeams[run + e];
            if (any(E.xyz < blo) || any(E.xyz > bhi)) continue;
            uint h = asuint(E.w);
            if (!vvInBox(t, E.xyz, h, vvAnti(t, gStatic[gP.nodeInfoBase + h].w))) continue;
            cand = true;
            uint n = vvPairAdd(0u);
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
    uint sorted = vvSortAt(0u);
    uint seen = 0u;
    for (uint i = sub, iStop = EB_END(641, i, cells, subs); i < iStop; i += subs)
    {
        uint b = gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z))));
        bool dup = false;
        int3 cj = int3(0, 0, 0);
        for (uint j = 0u, jStop = EB_END(642, j, i, 1u); j < jStop; j++)
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
        uint e0 = vvOffAt(b);
        uint e1 = vvOffAt(b + 1u);
        seen += e1 - e0;
        for (uint e = e0, eGuard = 0u; e < e1; e++)
        {
            EB_PASS(643, eGuard);
            float4 E = gBeams[sorted + e];
            if (any(E.xyz < blo) || any(E.xyz > bhi)) continue;
            uint h = asuint(E.w);
            if (!vvInBox(t, E.xyz, h, vvAnti(t, gStatic[gP.nodeInfoBase + h].w))) continue;
            cand = true;
            uint n = vvPairAdd(0u);
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

// (T) Part `sub` of `subs` of owner va's listed triangle q: its box's cells, its pairs
// listed (vvTriangleCandsList).
void flWalk(uint va, uint q, uint sub, uint subs)
{
    gFl = 1u;
    gFlV = va;
    uint triOff = gStatic[va * 4u + 1u].x;
    uint t = gNext[vvTestedAt(triOff + q)];
    if (flWalkBox()) flWalkBoxed(va, triOff + t, sub, subs, flCands(va), asfloat(gStatic[va * 4u + 3u].z), false);
    else
        vvTriangleCandsList(0u, va, triOff + t, sub, subs, 0u, uint(gAcc[flOwn(va) + FL_OFFERED].load()), flCands(va),
                            asfloat(gStatic[va * 4u + 3u].z));
    gFl = 0u;
}
// Part `part` of big triangle gt's test of every candidate of its owner (flBigHint).
void flWalkBig(uint gt, uint part)
{
    uint va = gStatic[gP.triBase + gt].w;
    gFl = 1u;
    gFlV = va;
    flWalkBoxed(va, gt, part, FL_BIG_PARTS, flCands(va), asfloat(gStatic[va * 4u + 3u].z), true);
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

#include "explicit_beams_fleet_state.slang"
#include "explicit_beams_fleet_keep.slang"
#include "explicit_beams_fleet_plan.slang"
