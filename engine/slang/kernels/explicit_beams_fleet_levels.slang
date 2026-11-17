// --- Stage 3c: the levels across the GPU (PERF_FLEET_LEVELS; engine/docs/measurements/
// contact-groups-design.md §8.7; included at the end of explicit_beams_fleet.slang) ---
// fleetLevelsAny applies a group's hits level by level in the group's one workgroup: every
// hit on one core (32 cars: two-thirds of the frame). What the order needs is only this:
// level l reads what level l - 1 wrote (the levels in turn), and the hits on one contact
// point run in triangle order (vvApplyOrdered: a chain read and written by one thread).
// Everything else a hit writes is an integer sum (accAdd, shiftAdd, cfAdd, the residual
// stages), the same whatever the order, and the owners of one level touch disjoint
// vehicles. So a level's hits, of every group of the fleet, run as threads across the
// GPU, a dispatch a level and phase (the application, the three residual phases), with
// the same leaf functions on the same state as the group's workgroup gives them:
//   the group's workgroup (fleetLevelsDefer)  its owners' levels (fleetLevelsAny's rule
//                        and rounds), each member marked with the step; a group whose
//                        hit list overflowed, or that takes the small groups' levels,
//                        keeps its own levels whole (marked 0)
//   the plan (fleetLevelsPlan, the fleet plan's module, one workgroup)  the marked owners
//                        with hits by level slot (1 .. L, then L + 1 for any higher
//                        level), the hits' prefix in that order, each dispatch's size
//   the snapshot (fleetLevelsWide, LV_SNAP)  every listed hit's nodes, a thread a node of
//                        a hit (one node may be taken by several: the same values)
//   a level l <= L (LV_APPLY, then the residual phases)  a thread a hit of that level
//   the end (fleetLevelsEnd, the groups' module, a workgroup a group)  a group's levels
//                        past L in its workgroup, its two-body counts cleared, its
//                        owners' records ended, the fleet's arguments zeroed, the frame's
//                        end after the last step
// A dispatch's level and phase are its parameter copy's step word (SPLIT_LV | phase << 8
// | level), shared by every step (a level dispatch needs no step: each owner's step is its
// mark). The same leaf functions, values and order of every float sum: the same results
// bit for bit (vehicle-perf-first-principles.md §8.32).
static const uint PERF_FLEET_LEVELS = 524288u;
static const uint SPLIT_LV = 0x10000000u;
static const uint LV_APPLY = 0u; // 1, 2, 3: the residual phases 0, 1, 2
static const uint LV_SNAP = 4u;
static const uint LV_PLAN = 5u;
static const uint LV_END = 6u;
// Level slots at most (the host's FLEET_LEVEL_MAX).
static const uint LV_MAX = 16u;
static const uint LV_MARK = 0x80000000u;
uint lvPhase(uint word) { return (word >> 8u) & 15u; }
uint lvLevel(uint word) { return min(word & 255u, LV_MAX); }
bool flLevelsOn()
{
    uint m = PERF_FLEET_STATE | PERF_FLEET_FRONT | PERF_FLEET_ANY | PERF_FLEET_POST | PERF_FLEET_LEVELS;
    return (gP.perfFlags & m) == m;
}
// The frame's level slots (the host's, in the copies' perf flags past the switches:
// explicit_plan::FLEET_LEVEL_SLOTS_SHIFT). 0: the levels' dispatches are not in this
// frame; the groups find and report their levels (fleetLevelsDefer) and keep them.
static const uint LV_SLOTS_SHIFT = 27u;
uint lvSlots() { return (gP.perfFlags >> LV_SLOTS_SHIFT) & 31u; }
// gAcc's fleet counters 0 and 1 (flCtl): the frame's highest level of a group and its most
// owners with hits past its levels (some level holds two or more: the levels' dispatches
// then run owners side by side), maxima the host reads back and zeroes each frame.
static const uint LV_REPORT = 0u;
// gNext after the fleet's counters (explicit_plan::fleet_next_words): per vehicle its mark
// (LV_MARK | the step: its group's levels are the fleet's this step; 0 not), the marked
// owners with hits in slot order, the hits' prefix in that order (one more), then per slot
// (index 1 .. L + 2; 0 the totals) its first owner's place and its first hit.
uint flLvAt(uint i) { return fkCtl(FL_CTL) + i; }
uint flLvMark(uint v) { return flLvAt(v); }
uint flLvOwner(uint i) { return flLvAt(gP.vehicles + i); }
uint flLvPrefix(uint i) { return flLvAt(2u * gP.vehicles + i); }
uint flLvStart(uint l) { return flLvAt(3u * gP.vehicles + 1u + l); }
uint flLvHit(uint l) { return flLvAt(3u * gP.vehicles + 1u + (LV_MAX + 3u) + l); }
// The levels' arguments in the first buffer (gArgs): the snapshot at FL_ARGS_LV, level l's
// at FL_ARGS_LV + 4 l (l = 1 .. LV_MAX; the host's FLEET_LEVEL_ARGS).
static const uint FL_ARGS_LV = 16u;
// An owner's step's parity and its kept beam sums' tag (as splitGroupStep has them), from
// its group's step count at the frame's start and the step of its mark.
uint lvParity(uint v, uint s) { return (gNext[flVehAt(0u, v)] + s) & 1u; }
uint lvTag(uint v, uint s)
{
    return (gP.perfFlags & PERF_VV_BEAM_SUM) != 0u ? ((gNext[flVehAt(0u, v)] + s + 1u) | 0x80000000u) : 0u;
}

// (The groups' module's: splitGroupStep and its entry reach these in every module that
// includes it.)
// The group's part (every lane; uniform result): its owners' levels as fleetLevelsAny finds
// them, each member marked. True: the levels are the fleet's dispatches' this step and the
// rest of the pass the end dispatch's. False (marked 0): the group's own levels follow.
// `allowed` false: the small groups' levels (a long hit list), not these.
bool fleetLevelsDefer(uint k, uint off, uint lane, uint s, bool allowed)
{
    uint open = 0u;
    if (lane == 0u && allowed && (gP.ablate & (65536u | 32768u | 16384u | 262144u)) == 0u)
    {
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++)
        {
            uint own = flOwn(vvMember(off, m));
            open += uint(gAcc[own + FL_OFFERED].load()) + uint(gAcc[own + FL_RESET].load()) + uint(gAcc[own + FL_FIX].load());
        }
    }
    open = splitUniform(lane, open);
    uint ok = 0u;
    if (open != 0u)
    {
        // The owners with hits (gsOff), and whether every hit list is within its room.
        if (lane == 0u) gsCount[0].store(0u);
        sync();
        for (uint m = lane, mStop = EB_END(m, k, LANES); m < mStop; m += LANES)
        {
            if (flHits(vvMember(off, m)) != 0u) gsOff[gsCount[0].add(1u)] = m;
        }
        sync();
        uint H = splitUniform(lane, lane == 0u ? gsCount[0].load() : 0u);
        uint fits = 1u;
        if (lane == 0u)
        {
            for (uint j = 0u, jStop = EB_END(j, H, 1u); j < jStop; j++)
            {
                uint v = vvMember(off, gsOff[j]);
                if (flHits(v) > 2u * uint(gAcc[flOwn(v) + FL_OFFERED].load())) fits = 0u;
            }
        }
        fits = splitUniform(lane, fits);
        if (H != 0u && fits != 0u)
        {
            // fleetLevelsAny's levels: the same rule, partners and rounds.
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
            // The report: this group's highest level and its owners past it.
            if (lane == 0u)
            {
                uint top = 0u;
                for (uint j = 0u, jStop = EB_END(j, H, 1u); j < jStop; j++) top = max(top, gNext[flLevelAt(vvMember(off, gsOff[j]))]);
                gAcc[flCtl(LV_REPORT)].max(int(top));
                gAcc[flCtl(LV_REPORT + 1u)].max(int(H - min(top, H)));
            }
            ok = lvSlots() != 0u ? 1u : 0u;
        }
    }
    for (uint m = lane, mStop = EB_END(m, k, LANES); m < mStop; m += LANES)
        gNext[flLvMark(vvMember(off, m))] = ok != 0u ? (s | LV_MARK) : 0u;
    sync();
    return ok != 0u;
}

// The end dispatch (the groups' module, step word SPLIT_LV | LV_END << 8 | L), workgroup g:
// the fleet's arguments zeroed; for a marked group its levels past the L slots in turn (as
// fleetLevelsAny's), its two-body counts cleared, its owners' records ended, its marks
// cleared, and after the frame's last step its frame end.
void fleetLevelsEnd(uint g, uint lane, uint word)
{
    uint4 G = gMisc[gP.vvGroups + g];
    uint off = G.x;
    uint k = splitUniform(lane, G.y);
    if (k == 0u) return;
    flArgsZero(lane);
    if (lane <= LV_MAX) gArgs[FL_ARGS_LV + 4u * lane] = 0u;
    uint w = splitUniform(lane, gNext[flLvMark(vvMember(off, 0u))]);
    if ((w & LV_MARK) == 0u) return;
    uint s = w & SPLIT_STEP;
    uint L = lvLevel(word);
    uint v0 = vvMember(off, 0u);
    uint parity = lvParity(v0, s);
    gFl = 2u;
    gFlOff = off;
    gVvTag = gP.vvSplit != 0u ? lvTag(v0, s) : 0u;
    // The owners with hits (gsOff) and the highest level.
    if (lane == 0u) gsCount[0].store(0u);
    sync();
    for (uint m = lane, mStop = EB_END(m, k, LANES); m < mStop; m += LANES)
    {
        if (flHits(vvMember(off, m)) != 0u) gsOff[gsCount[0].add(1u)] = m;
    }
    sync();
    uint H = splitUniform(lane, lane == 0u ? gsCount[0].load() : 0u);
    uint top = 0u;
    if (lane == 0u)
    {
        for (uint j = 0u, jStop = EB_END(j, H, 1u); j < jStop; j++) top = max(top, gNext[flLevelAt(vvMember(off, gsOff[j]))]);
    }
    top = splitUniform(lane, top);
    for (uint l = L + 1u, lGuard = 0u; l <= top; l++)
    {
        EB_PASS_UNIFORM(lGuard, EB_MAX_MEMBERS + 1u);
        for (uint j = 0u, jStop = EB_END(j, H, 1u); j < jStop; j++)
        {
            uint a = gsOff[j];
            uint va = vvMember(off, a);
            if (gNext[flLevelAt(va)] != l) continue;
            vvApplyOrdered(va, vvHitSmall(0u, 0u, a, 0u), flHits(va), asfloat(gStatic[va * 4u + 3u].z), parity, lane, LANES, a);
        }
        sync();
        if (gP.vvSplit != 0u)
        {
            for (uint ph = 0u; ph < 3u; ph++)
            {
                for (uint j = 0u, jStop = EB_END(j, H, 1u); j < jStop; j++)
                {
                    uint a = gsOff[j];
                    uint va = vvMember(off, a);
                    if (gNext[flLevelAt(va)] != l) continue;
                    vvResidualPhase(va, vvHitSmall(0u, 0u, a, 0u), flHits(va), asfloat(gStatic[va * 4u + 3u].z), parity, lane, LANES, a, ph);
                }
                sync();
            }
        }
    }
    // The two-body counts the hits wrote, once all levels are applied (every list fits).
    if (gP.vvSplit != 0u)
    {
        for (uint j = 0u, jStop = EB_END(j, H, 1u); j < jStop; j++)
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
    gVvTag = 0u;
    gFl = 0u;
    sync();
    for (uint m = lane, mStop = EB_END(m, k, LANES); m < mStop; m += LANES)
    {
        uint v = vvMember(off, m);
        flOwnerEnd(v);
        gNext[flLvMark(v)] = 0u;
    }
    splitGroupEnd(off, k, lane, s);
}

// Owner v's slot this step: 0 none (not marked, no hits), 1 .. L its level, L + 1 higher.
uint flLvSlot(uint v, uint L)
{
    if (!flVehicleOn(v) || (gNext[flLvMark(v)] & LV_MARK) == 0u || flHits(v) == 0u) return 0u;
    uint l = gNext[flLevelAt(v)];
    return l == 0u ? 0u : min(l, L + 1u);
}

#ifdef EB_FLEET_PLAN
// The hits' prefix over the n owners in slot order (flLvPrefix 0 .. n; n uniform): flPrefix's
// scan with an owner's place for a vehicle.
void lvScan(uint n, uint lane)
{
    sync();
    if (n <= FL_SERIAL)
    {
        if (lane == 0u)
        {
            uint run = 0u;
            for (uint i = 0u, iStop = EB_END(i, n, 1u); i < iStop; i++)
            {
                gNext[flLvPrefix(i)] = run;
                run += flHits(gNext[flLvOwner(i)]);
            }
            gNext[flLvPrefix(n)] = run;
        }
        sync();
        return;
    }
    uint chunk = (n + LANES - 1u) / LANES;
    uint i0 = min(lane * chunk, n);
    uint i1 = min(i0 + chunk, n);
    uint block = LANES / GRID_BLOCKS;
    uint sum = 0u;
    for (uint i = i0, iStop = EB_END(i, i1, 1u); i < iStop; i++) sum += flHits(gNext[flLvOwner(i)]);
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
        gNext[flLvPrefix(n)] = run;
    }
    sync();
    {
        uint run = gsPart[lane] + gsBlk[lane / block];
        for (uint i = i0, iStop = EB_END(i, i1, 1u); i < iStop; i++)
        {
            gNext[flLvPrefix(i)] = run;
            run += flHits(gNext[flLvOwner(i)]);
        }
    }
    sync();
}

// The levels' plan (one workgroup; step word SPLIT_LV | LV_PLAN << 8 | L), after the groups'
// dispatch: the marked owners with hits by slot (within a slot in any order: nothing in a
// level depends on its owners' order), their hits' prefix, each level dispatch's size.
void fleetLevelsPlan(uint lane, uint word)
{
    uint nv = gP.vehicles;
    uint L = lvLevel(word);
    if (lane < LV_MAX + 3u) gsCount[lane].store(0u);
    sync();
    for (uint v = lane, vStop = EB_END(v, nv, LANES); v < vStop; v += LANES)
    {
        uint sl = flLvSlot(v, L);
        if (sl != 0u) gsCount[sl].add(1u);
    }
    sync();
    if (lane == 0u)
    {
        uint run = 0u;
        for (uint l = 1u, lStop = EB_END(l, L + 2u, 1u); l < lStop; l++)
        {
            uint c = gsCount[l].load();
            gsOff[l] = run;
            gNext[flLvStart(l)] = run;
            run += c;
            gsCount[l].store(0u);
        }
        gNext[flLvStart(L + 2u)] = run;
        gNext[flLvStart(0u)] = run;
        gsUniformOpen = run;
    }
    sync();
    for (uint v = lane, vStop = EB_END(v, nv, LANES); v < vStop; v += LANES)
    {
        uint sl = flLvSlot(v, L);
        if (sl != 0u) gNext[flLvOwner(gsOff[sl] + gsCount[sl].add(1u))] = v;
    }
    uint n = splitUniform(lane, gsUniformOpen);
    lvScan(n, lane);
    if (lane == 0u)
    {
        for (uint l = 1u, lStop = EB_END(l, L + 3u, 1u); l < lStop; l++) gNext[flLvHit(l)] = gNext[flLvPrefix(gNext[flLvStart(l)])];
        uint total = gNext[flLvPrefix(n)];
        gNext[flLvHit(0u)] = total;
        gArgs[FL_ARGS_LV] = flWorkgroups(4u * total);
        gArgs[FL_ARGS_LV + 1u] = 1u;
        gArgs[FL_ARGS_LV + 2u] = 1u;
        for (uint l = 1u, lStop = EB_END(l, L + 1u, 1u); l < lStop; l++)
        {
            gArgs[FL_ARGS_LV + 4u * l] = flWorkgroups(gNext[flLvHit(l + 1u)] - gNext[flLvHit(l)]);
            gArgs[FL_ARGS_LV + 4u * l + 1u] = 1u;
            gArgs[FL_ARGS_LV + 4u * l + 2u] = 1u;
        }
    }
}
#endif

#ifdef EB_FLEET_LEVELS
// The last place p in [lo, hi) whose first hit is at most g.
uint flLvFind(uint lo, uint hi, uint g)
{
    for (uint it = 0u; it < 32u; it++)
    {
        if (hi - lo <= 1u) break;
        uint mid = (lo + hi) >> 1u;
        if (gNext[flLvPrefix(mid)] <= g) lo = mid;
        else hi = mid;
    }
    return lo;
}

// Thread tid of a levels dispatch (step word SPLIT_LV | phase << 8 | level): the snapshot of
// one listed hit's nodes (fleetLevelsAny's), or one hit of level l: its contact point's
// chain when it heads it (vvApplyOrdered with the hit alone on its lane), or its residual
// phase (vvResidualPhase, the same). The owner's step from its mark; its state through the
// address helpers as one owner on this thread (gFl 1).
void fleetLevelsWide(uint tid, uint word)
{
    uint phase = lvPhase(word);
    uint g;
    uint lo;
    uint hi;
    if (phase == LV_SNAP)
    {
        // (Four threads a hit: its triangle's nodes and its contact point.)
        if (gP.vvSplit == 0u || (tid >> 2u) >= gNext[flLvHit(0u)]) return;
        g = tid >> 2u;
        lo = 0u;
        hi = gNext[flLvStart(0u)];
    }
    else
    {
        uint l = lvLevel(word);
        g = gNext[flLvHit(l)] + tid;
        if (g >= gNext[flLvHit(l + 1u)]) return;
        lo = gNext[flLvStart(l)];
        hi = gNext[flLvStart(l + 1u)];
    }
    uint p = flLvFind(lo, hi, g);
    uint va = gNext[flLvOwner(p)];
    uint i = g - gNext[flLvPrefix(p)];
    uint n = flHits(va);
    uint s = gNext[flLvMark(va)] & SPLIT_STEP;
    uint parity = lvParity(va, s);
    uint tag = lvTag(va, s);
    gFl = 1u;
    gFlV = va;
    uint base = vvHitSmall(0u, 0u, 0u, 0u);
    if (phase == LV_SNAP)
    {
        uint4 tn = gStatic[gP.triBase + gNext[base + 2u * i]];
        uint h = gNext[base + 2u * i + 1u];
        uint which = tid & 3u;
        if (which == 3u)
            vvSnapNode(h, gStatic[gP.nodeInfoBase + h].w, parity, tag);
        else
            vvSnapNode(which == 0u ? tn.x : (which == 1u ? tn.y : tn.z), tn.w, parity, tag);
        return;
    }
    gVvTag = gP.vvSplit != 0u ? tag : 0u;
    float range = asfloat(gStatic[va * 4u + 3u].z);
    if (phase == LV_APPLY)
        vvApplyOrdered(va, base, n, range, parity, i, n, 0u);
    else if (gP.vvSplit != 0u)
        lvResidual(va, base, i, range, parity, phase - 1u, g);
}

// gBeams after the owners' boxes (explicit_plan's FLEET_LEVEL_CACHE): four records a hit of
// the step, at its place g in slot order: (alpha, beta, gamma, staged), (nn), (v), (f1).
static const uint LV_CACHE = 4096u;
uint flLvCache(uint g) { return gP.vvFleetBox + 4u * gP.vehicles + gP.totalNodes + 4u * g; }

// vvResidualPhase for hit i of owner va (place g) alone. Its staged geometry and first-pass
// force (vvStagedHit) read only what no phase of the pass writes (positions, velocities,
// the snapshot, the two-body counts, the clip state), so phase 0 keeps them and phases 1
// and 2 load them: the same values (past LV_CACHE hits a step, each phase takes them).
void lvResidual(uint va, uint base, uint i, float r, uint parity, uint phase, uint g)
{
    if (gP.selfSplit == 0u || gP.vvSplit == 0u) return;
    uint gt = gNext[base + 2u * i];
    uint h = gNext[base + 2u * i + 1u];
    uint vh = gStatic[gP.nodeInfoBase + h].w;
    uint4 tn = gStatic[gP.triBase + gt];
    bool kept = gNext[flLvHit(0u)] <= LV_CACHE;
    float alpha;
    float beta;
    float gamma;
    float3 nn;
    float3 v;
    float3 f1;
    if (phase == 0u || !kept)
    {
        VvTri t = vvTriangle(gt, asfloat(gMisc[va].xyz), r);
        bool staged = vvStagedHit(t, gt, h, nf3(asfloat(gMisc[vh].xyz) + gNodes[h * 3u].xyz), vh, r, parity, 0u, alpha, beta,
                                  gamma, nn, v, f1);
        if (kept)
        {
            uint at = flLvCache(g);
            gBeams[at] = float4(alpha, beta, gamma, staged ? 1.0 : 0.0);
            gBeams[at + 1u] = float4(nn, 0.0);
            gBeams[at + 2u] = float4(v, 0.0);
            gBeams[at + 3u] = float4(f1, 0.0);
        }
        if (!staged) return;
    }
    else
    {
        uint at = flLvCache(g);
        float4 a = gBeams[at];
        if (a.w == 0.0) return;
        alpha = a.x;
        beta = a.y;
        gamma = a.z;
        nn = gBeams[at + 1u].xyz;
        v = gBeams[at + 2u].xyz;
        f1 = gBeams[at + 3u].xyz;
    }
    if (phase == 2u)
    {
        resMerge(h, tn.x, tn.y, tn.z);
        return;
    }
    float rr = vvResOf(va, vh, h, tn, alpha, beta, gamma, nn, v, parity, 0u);
    if (phase == 0u)
    {
        if (rr > 0.0) resStage(h, tn.x, tn.y, tn.z, alpha, beta, gamma, sscale(nn, rr));
        return;
    }
    float3 total = f1;
    if (rr > 0.0) total = sadd3(total, sscale(nn, rr));
    float3 gg;
    if (vvGuardOf(va, vh, h, tn, alpha, beta, gamma, v, total, parity, gg)) total = sadd3(total, gg);
    resFinal(h, tn.x, tn.y, tn.z, alpha, beta, gamma, f1, total);
}

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_fleet_levels(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    fleetLevelsWide(gid.x * LANES + lid.x, gP.pad2);
    ebLoopCapFlush();
}
#endif
