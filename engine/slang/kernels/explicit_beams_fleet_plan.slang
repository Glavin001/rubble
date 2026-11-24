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
        if (lane == 0u) gAcc[flCtl(FL_CTL_BIG)].store(0);
        flPrefix(0u, FL_OFFERED, lane);
        uint total = splitUniform(lane, gNext[flPrefixAt(0u, nv)]);
        if (total > vvRoomCp())
        {
            for (uint v = lane, vStop = EB_END(379, v, nv, LANES); v < vStop; v += LANES) gAcc[flOwn(v) + FL_OFFERED].store(0);
            if (lane == 0u) flError(FL_ERR_RUN);
            flPrefix(0u, FL_OFFERED, lane);
        }
        // Each owner's records of the step; whether any owner resets (its triangles'
        // throttles then change even when nothing is offered: gsUniformOpen, free here).
        if (lane == 0u) gsUniformOpen = 0u;
        sync();
        bool resets = false;
        for (uint v = lane, vStop = EB_END(380, v, nv, LANES); v < vStop; v += LANES)
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
            for (uint v = 0u, vStop = EB_END(381, v, nv, 1u); v < vStop; v++) any += fkCands(v);
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
            for (uint v = 0u, vStop = EB_END(382, v, nv, 1u); v < vStop; v++) most = max(most, uint(gAcc[fkAcc(v, FK_PAIRS)].load()));
            uint total = gNext[fkPrefix(nv)];
            gNext[fkCtl(FK_MOST)] = max(gNext[fkCtl(FK_MOST)], most);
            uint cands = 0u;
            for (uint v = 0u, vStop = EB_END(383, v, nv, 1u); v < vStop; v++) cands += fkCands(v);
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
                for (uint v = 0u, vStop = EB_END(384, v, nv, 1u); v < vStop; v++) listedAll += min(uint(gAcc[fkAcc(v, FK_PAIRS)].load()), fkRoom());
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
            // (The big triangles' threads after the listed ones': flWalkBig.)
            uint big = flWalkBox() ? min(uint(gAcc[flCtl(FL_CTL_BIG)].load()), gP.totalTris) : 0u;
            gNext[flCtlNext(FL_NEXT_BIG)] = big;
            gArgs2[FL_ARGS2_WALK] = flWorkgroups(listed * subs + big * FL_BIG_PARTS);
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
        for (uint v = 0u, vStop = EB_END(385, v, nv, 1u); v < vStop; v++)
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
        if (subs == 0u) return;
        if (tid >= listed * subs)
        {
            uint j = tid - listed * subs;
            if (j < gNext[flCtlNext(FL_NEXT_BIG)] * FL_BIG_PARTS) flWalkBig(gNext[flListAt(j / FL_BIG_PARTS)], j % FL_BIG_PARTS);
            return;
        }
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
