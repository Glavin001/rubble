// --- Stage 1: the state by vehicle under the contact groups' own dispatches ---

// Group (k members at `off`, its part of the fleet-wide regions at VV_PART · runOff)'s
// front in its workgroup: vvPassSmallFront with every owner's state its vehicle's.
// Returns 0 when no owner is offered a partner, resets or fixes (uniform), else 3.
uint fleetGroupFront(uint g, uint k, uint off, uint runOff, uint parity, uint lane)
{
    // (S) the plan's pairs of this group's vehicles.
    uint pairs = gNext[flAdjFirst(gP.vehicles)] / 2u;
    for (uint p = lane, pStop = EB_END(319, p, pairs, LANES); p < pStop; p += LANES)
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
        for (uint m = 0u, mStop = EB_END(320, m, k, 1u); m < mStop; m++)
        {
            uint own = flOwn(vvMember(off, m));
            uint offered = uint(gAcc[own + FL_OFFERED].load());
            gNext[flPrefixAt(0u, vvMember(off, m))] = run;
            run += offered;
            open += offered + uint(gAcc[own + FL_RESET].load()) + uint(gAcc[own + FL_FIX].load());
        }
        gsUniformOpen = open;
    }
    for (uint m = lane, mStop = EB_END(321, m, k, LANES); m < mStop; m += LANES) flOwnerPlan(vvMember(off, m), parity);
    sync();
    if (gsUniformOpen == 0u) return 0u;
    // (P) every owner's throttle step and candidates.
    {
        uint triTotal = 0u;
        for (uint a = 0u, aStop = EB_END(322, a, k, 1u); a < aStop; a++) triTotal += gStatic[vvMember(off, a) * 4u + 1u].y;
        for (uint p = lane, pStop = EB_END(323, p, triTotal, LANES); p < pStop; p += LANES)
        {
            uint a = 0u;
            uint t = p;
            for (uint c = 0u, cGuard = 0u; c + 1u < k; c++)
            {
                EB_PASS(324, cGuard);
                uint n = gStatic[vvMember(off, a) * 4u + 1u].y;
                if (t < n) break;
                t -= n;
                a++;
            }
            uint va = vvMember(off, a);
            flTriangle(va, gStatic[va * 4u + 1u].x + t, parity);
        }
        for (uint m = 0u, mStop = EB_END(325, m, k, 1u); m < mStop; m++)
        {
            uint vb = vvMember(off, m);
            uint n = vvCpCount(vb);
            for (uint i = lane, iStop = EB_END(326, i, n, LANES); i < iStop; i += LANES) flPoint(vb, i);
        }
    }
    // Each owner's grid, then its candidates' sorted copy.
    for (uint a = 0u, aStop = EB_END_UNIFORM(327, a, k, 1u, EB_MAX_MEMBERS); a < aStop; a++) flOffsets(vvMember(off, a), lane);
    sync();
    for (uint a = 0u, aStop = EB_END(328, a, k, 1u); a < aStop; a++)
    {
        uint va = vvMember(off, a);
        uint cands = flCands(va);
        for (uint e = lane, eStop = EB_END(329, e, cands, LANES); e < eStop; e += LANES) flSortCand(va, e);
    }
    sync();
    return 3u;
}

// The levels' inputs from the owners' records into the group's workgroup arrays (lane 0,
// before a barrier): vvPassSmallLevels then runs as in the group's own pass, its hit lists
// through the address helpers (gFl 2).
void flImport(uint off, uint k)
{
    for (uint a = 0u, aStop = EB_END(330, a, k, 1u); a < aStop; a++)
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
        for (uint e = gNext[flAdjFirst(v)], eStop = EB_END(331, e, e1, 1u); e < eStop; e++)
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
        for (uint m = 0u, mStop = EB_END(332, m, k, 1u); m < mStop; m++)
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
    for (uint m = lane, mStop = EB_END(333, m, k, LANES); m < mStop; m += LANES) flOwnerEnd(vvMember(off, m));
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
        for (uint a = 0u, aStop = EB_END(334, a, k, 1u); a < aStop; a++) total += uint(gAcc[flOwn(vvMember(off, a)) + FL_LISTED].load());
        uint subs = clamp(stride / max(total, 1u), 1u, 64u);
        for (uint p = at0, pStop = EB_END(335, p, total * subs, stride); p < pStop; p += stride)
        {
            uint q = p / subs;
            uint a = 0u;
            for (uint c = 0u, cGuard = 0u; c + 1u < k; c++)
            {
                EB_PASS(336, cGuard);
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
    for (uint a = 0u, aStop = EB_END(337, a, k, 1u); a < aStop; a++)
    {
        uint n = uint(gAcc[flOwn(vvMember(off, a)) + FL_PAIRS].load());
        total += n <= TRI_PAIR_CAP ? n : 0u;
    }
    for (uint p = at0, pStop = EB_END(338, p, total, stride); p < pStop; p += stride)
    {
        uint q = p;
        uint a = 0u;
        for (uint c = 0u, cGuard = 0u; c < k; c++)
        {
            EB_PASS(339, cGuard);
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
    for (uint a = 0u, aStop = EB_END(340, a, k, 1u); a < aStop; a++)
    {
        uint own = flOwn(vvMember(off, a));
        over += uint(gAcc[own + FL_PAIRS].load()) > TRI_PAIR_CAP ? uint(gAcc[own + FL_LISTED].load()) : 0u;
    }
    uint subs = clamp(stride / max(over, 1u), 1u, 64u);
    for (uint p = at0, pStop = EB_END(341, p, over * subs, stride); p < pStop; p += stride)
    {
        uint q = p / subs;
        uint a = 0u;
        for (uint c = 0u, cGuard = 0u; c < k; c++)
        {
            EB_PASS(342, cGuard);
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
        for (uint m = 0u, mStop = EB_END(343, m, k, 1u); m < mStop; m++)
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
        for (uint m = lane, mStop = EB_END(344, m, k, LANES); m < mStop; m += LANES)
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
            for (uint j = 0u, jStop = EB_END(345, j, H, 1u); j < jStop; j++)
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
            for (uint j = 0u, jStop = EB_END(346, j, H, 1u); j < jStop; j++)
            {
                uint a = gsOff[j];
                if (flHits(vvMember(off, a)) > vvHitRoom(a, 0u)) every = true;
            }
            if (every) vvSnapshot(off, k, parity, lane);
            else
            {
                for (uint j = 0u, jStop = EB_END(347, j, Hl, 1u); j < jStop; j++)
                {
                    uint a = gsOff[j];
                    uint base = vvHitSmall(0u, 0u, a, 0u);
                    uint n = flHits(vvMember(off, a));
                    for (uint i = lane, iStop = EB_END(348, i, n, LANES); i < iStop; i += LANES)
                    {
                        uint4 tn = gStatic[gP.triBase + gNext[base + 2u * i]];
                        uint h = gNext[base + 2u * i + 1u];
                        // (The four nodes at one call site, in the same order: tn.x, tn.y, tn.z, h. The
                        // compilers inline every call; opaqueCount keeps the loop from being unrolled back.)
                        for (uint q = 0u, qStop = EB_END(645, q, opaqueCount(4u), 1u); q < qStop; q++)
                            vvSnapNode(q == 0u ? tn.x : (q == 1u ? tn.y : (q == 2u ? tn.z : h)),
                                       q < 3u ? tn.w : gStatic[gP.nodeInfoBase + h].w, parity, tag);
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
        for (uint m = lane, mStop = EB_END(349, m, k, LANES); m < mStop; m += LANES)
        {
            uint v = vvMember(off, m);
            gNext[flLevelAt(v)] = flHits(v) != 0u ? 1u : 0u;
        }
        for (uint j = lane, jStop = EB_END(350, j, H, LANES); j < jStop; j += LANES)
        {
            uint m = gsOff[j];
            uint v = vvMember(off, m);
            uint2 p = uint2(0u, 0u);
            uint e1 = gNext[flAdjFirst(v + 1u)];
            for (uint e = gNext[flAdjFirst(v)], eStop = EB_END(351, e, e1, 1u); e < eStop; e++)
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
            for (uint j = lane, jStop = EB_END(352, j, H, LANES); j < jStop; j += LANES)
            {
                uint m = gsOff[j];
                uint v = vvMember(off, m);
                uint l = 1u;
                for (uint c = 0u, cStop = EB_END(353, c, H, 1u); c < cStop; c++)
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
            for (uint m = 0u, mStop = EB_END(354, m, k, 1u); m < mStop; m++) top = max(top, gNext[flLevelAt(vvMember(off, m))]);
        }
        top = splitUniform(lane, top);
        // The hits, level by level; with the first, every listed triangle's throttle
        // update (its marks are final) and a later owner's reset.
        for (uint l = 1u, lGuard = 0u; l <= top; l++)
        {
            EB_PASS_UNIFORM(355, lGuard, EB_MAX_MEMBERS + 1u);
            if (l == 1u && !flPostOn())
            {
                for (uint a = 0u, aStop = EB_END(356, a, k, 1u); a < aStop; a++)
                {
                    uint va = vvMember(off, a);
                    uint own = flOwn(va);
                    uint triOff = gStatic[va * 4u + 1u].x;
                    uint listed = uint(gAcc[own + FL_LISTED].load());
                    bool fix = gAcc[own + FL_FIX].load() != 0;
                    for (uint i = lane, iStop = EB_END(357, i, listed, LANES); i < iStop; i += LANES)
                    {
                        uint gt = triOff + gNext[vvTestedAt(triOff + i)];
                        vvThrottleEnd(gt);
                        if (fix) gMisc[gP.vvTri + gt].x = 0u;
                    }
                }
            }
            for (uint j = 0u, jStop = EB_END(358, j, Hl, 1u); j < jStop; j++)
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
                    for (uint p = lane, pStop = EB_END(359, p, listed * subs, LANES); p < pStop; p += LANES)
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
                    for (uint j = 0u, jStop = EB_END(360, j, Hl, 1u); j < jStop; j++)
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
            for (uint j = 0u, jStop = EB_END(361, j, H, 1u); j < jStop; j++)
            {
                uint a = gsOff[j];
                uint n = flHits(vvMember(off, a));
                anyHits += n;
                if (n > vvHitRoom(a, 0u)) listedAll = false;
            }
            if (anyHits > 0u && !listedAll) vvSplitClear(off, k, vvRoom(), lane);
            else if (anyHits > 0u)
            {
                for (uint j = 0u, jStop = EB_END(362, j, Hl, 1u); j < jStop; j++)
                {
                    uint a = gsOff[j];
                    uint base = vvHitSmall(0u, 0u, a, 0u);
                    uint n = flHits(vvMember(off, a));
                    for (uint i = lane, iStop = EB_END(363, i, n, LANES); i < iStop; i += LANES)
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
    for (uint m = lane, mStop = EB_END(364, m, k, LANES); m < mStop; m += LANES) flOwnerEnd(vvMember(off, m));
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
            for (uint v = 0u, vStop = EB_END(365, v, nv, 1u); v < vStop; v++)
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
    for (uint v = v0, vStop = EB_END(366, v, v1, 1u); v < vStop; v++) sum += flCount(v, word);
    gsPart[lane] = sum;
    sync();
    if (lane < GRID_BLOCKS)
    {
        uint run = 0u;
        for (uint q = 0u, qStop = EB_END(367, q, block, 1u); q < qStop; q++)
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
        for (uint v = v0, vStop = EB_END(368, v, v1, 1u); v < vStop; v++)
        {
            gNext[flRowAt(r, v)] = run;
            run += flCount(v, word);
        }
    }
    sync();
}

