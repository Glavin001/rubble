// --- One hit's latency in a group's levels (PERF_FLEET_HITLAT), and the fleet front's
// fused dispatches (PERF_FLEET_FUSE); included at the end of explicit_beams_fleet.slang;
// engine/docs/measurements/contact-groups-design.md §8.9 ---
// The levels are chains: a level is the hits' application then three residual phases in
// turn, each as long as its slowest hit, one thread's dependent loads
// (vehicle-perf-first-principles.md §8.32). fleetLevelsAny takes a level's owners one after
// another on every lane and stages each hit again in every residual phase. fleetLevelsLat
// is fleetLevelsAny with the same leaf functions on the same state, the same rule, rounds
// and order, and three changes of how:
//   - the group's listed hits flattened over the lanes (a lane a hit of the level, whatever
//     its owner): a level's owners side by side, not in turn;
//   - the snapshot a lane a node of a hit (four lanes a hit);
//   - a hit's staged geometry and first-pass force (vvStagedHit) taken once in residual
//     phase 0 and kept in the lane's registers for phases 1 and 2, the lane that staged
//     the hit being the one that finishes it (it reads nothing a phase writes: positions,
//     velocities, the snapshot, the two-body counts, the clip state). A lane with a
//     second hit (past LANES hits a level) takes that one again in each phase.
// Integer sums only between the lanes of a level (accAdd, shiftAdd, cfAdd, the residual
// stages), one contact point's chain on one lane (vvApplyOrdered): the same results bit
// for bit.
static const uint PERF_FLEET_HITLAT = 8388608u;
static const uint PERF_FLEET_FUSE = 16777216u;
bool flHitLatOn() { return (gP.perfFlags & PERF_FLEET_HITLAT) != 0u; }
bool flFuseOn() { return (gP.perfFlags & (PERF_FLEET_FRONT | PERF_FLEET_FUSE)) == (PERF_FLEET_FRONT | PERF_FLEET_FUSE); }

// gsOff past the owners with hits (at most VV_MAX_GROUP of them): their hits' prefix.
static const uint LAT_PRE = 512u;

// The listed owner (its place j among the H) of the group's flattened hit x.
uint latOwner(uint H, uint x)
{
    uint lo = 0u;
    uint hi = H;
    for (uint it = 0u; it < 8u; it++)
    {
        if (hi - lo <= 1u) break;
        uint mid = (lo + hi) >> 1u;
        if (gsOff[LAT_PRE + mid] <= x) lo = mid;
        else hi = mid;
    }
    return lo;
}

// A hit's staged values, kept by its lane from residual phase 0.
struct LatKept
{
    bool staged;
    float alpha;
    float beta;
    float gamma;
    float3 nn;
    float3 v;
    float3 f1;
};

// vvResidualPhase's body for hit i of owner va (slot a), staging it unless `kept` holds it
// (`keep`: store what is staged into `kept`).
void latResidual(uint va, uint base, uint i, float r, uint parity, uint a, uint phase, bool have, bool keep,
                 inout LatKept kept)
{
    uint gt = gNext[base + 2u * i];
    uint h = gNext[base + 2u * i + 1u];
    uint vh = gStatic[gP.nodeInfoBase + h].w;
    uint4 tn = gStatic[gP.triBase + gt];
    if (!have)
    {
        VvTri t = vvTriangle(gt, asfloat(gMisc[va].xyz), r);
        float alpha;
        float beta;
        float gamma;
        float3 nn;
        float3 v;
        float3 f1;
        bool staged = vvStagedHit(t, gt, h, nf3(asfloat(gMisc[vh].xyz) + gNodes[h * 3u].xyz), vh, r, parity, a, alpha, beta,
                                  gamma, nn, v, f1);
        kept.staged = staged;
        kept.alpha = alpha;
        kept.beta = beta;
        kept.gamma = gamma;
        kept.nn = nn;
        kept.v = v;
        kept.f1 = f1;
    }
    if (!kept.staged) return;
    if (phase == 2u)
    {
        resMerge(h, tn.x, tn.y, tn.z);
        return;
    }
    float rr = vvResOf(va, vh, h, tn, kept.alpha, kept.beta, kept.gamma, kept.nn, kept.v, parity, a);
    if (phase == 0u)
    {
        if (rr > 0.0) resStage(h, tn.x, tn.y, tn.z, kept.alpha, kept.beta, kept.gamma, sscale(kept.nn, rr));
        return;
    }
    float3 total = kept.f1;
    if (rr > 0.0) total = sadd3(total, sscale(kept.nn, rr));
    float3 g;
    if (vvGuardOf(va, vh, h, tn, kept.alpha, kept.beta, kept.gamma, kept.v, total, parity, g)) total = sadd3(total, g);
    resFinal(h, tn.x, tn.y, tn.z, kept.alpha, kept.beta, kept.gamma, kept.f1, total);
}

// fleetLevelsAny with the group's hits flattened over the lanes (see above). Every lane of
// the group's workgroup; a hit list past its room: fleetLevelsAny itself.
void fleetLevelsLat(uint k, uint off, uint parity, uint lane, uint tag)
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
    if (open == 0u)
    {
        fleetLevelsAny(k, off, parity, lane, tag);
        return;
    }
    // The owners with hits (gsOff), their hits' prefix (gsOff past LAT_PRE), any list past
    // its room (then fleetLevelsAny, whose levels re-walk such an owner).
    if (lane == 0u) gsCount[0].store(0u);
    sync();
    for (uint m = lane, mStop = EB_END(m, k, LANES); m < mStop; m += LANES)
    {
        if (flHits(vvMember(off, m)) != 0u) gsOff[gsCount[0].add(1u)] = m;
    }
    sync();
    uint over = 0u;
    if (lane == 0u)
    {
        uint H = gsCount[0].load();
        uint run = 0u;
        for (uint j = 0u, jStop = EB_END(j, H, 1u); j < jStop; j++)
        {
            uint v = vvMember(off, gsOff[j]);
            uint n = flHits(v);
            if (n > 2u * uint(gAcc[flOwn(v) + FL_OFFERED].load())) over = 1u;
            gsOff[LAT_PRE + j] = run;
            run += n;
        }
        gsOff[LAT_PRE + H] = run;
    }
    over = splitUniform(lane, over);
    if (over != 0u || (gP.ablate & (65536u | 32768u | 16384u | 262144u)) != 0u)
    {
        fleetLevelsAny(k, off, parity, lane, tag);
        return;
    }
    uint H = splitUniform(lane, lane == 0u ? gsCount[0].load() : 0u);
    uint T = splitUniform(lane, gsOff[LAT_PRE + H]);
    gFl = 2u;
    gFlOff = off;
    gVvTag = 0u;
    // The snapshot: a lane a node of a listed hit.
    if (gP.vvSplit != 0u)
    {
        for (uint x = lane, xStop = EB_END(x, 4u * T, LANES); x < xStop; x += LANES)
        {
            uint g = x >> 2u;
            uint j = latOwner(H, g);
            uint base = vvHitSmall(0u, 0u, gsOff[j], 0u);
            uint i = g - gsOff[LAT_PRE + j];
            uint4 tn = gStatic[gP.triBase + gNext[base + 2u * i]];
            uint h = gNext[base + 2u * i + 1u];
            uint which = x & 3u;
            if (which == 3u)
                vvSnapNode(h, gStatic[gP.nodeInfoBase + h].w, parity, tag);
            else
                vvSnapNode(which == 0u ? tn.x : (which == 1u ? tn.y : tn.z), tn.w, parity, tag);
        }
        sync();
        gVvTag = tag;
    }
    // The owners' levels: fleetLevelsAny's rule, partners and rounds.
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
        if (H == 0u) break;
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
    // The hits, level by level; with the first, every listed triangle's throttle update and
    // a later owner's reset (unless the throttles' own dispatch made them).
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
        for (uint x = lane, xStop = EB_END(x, T, LANES); x < xStop; x += LANES)
        {
            uint j = latOwner(H, x);
            uint a = gsOff[j];
            uint va = vvMember(off, a);
            if (gNext[flLevelAt(va)] != l) continue;
            uint n = flHits(va);
            vvApplyOrdered(va, vvHitSmall(0u, 0u, a, 0u), n, asfloat(gStatic[va * 4u + 3u].z), parity, x - gsOff[LAT_PRE + j], n, a);
        }
        sync();
        if (gP.vvSplit != 0u && gP.selfSplit != 0u)
        {
            LatKept kept;
            kept.staged = false;
            kept.alpha = 0.0;
            kept.beta = 0.0;
            kept.gamma = 0.0;
            kept.nn = float3(0.0, 0.0, 0.0);
            kept.v = float3(0.0, 0.0, 0.0);
            kept.f1 = float3(0.0, 0.0, 0.0);
            for (uint ph = 0u; ph < 3u; ph++)
            {
                for (uint x = lane, xStop = EB_END(x, T, LANES); x < xStop; x += LANES)
                {
                    uint j = latOwner(H, x);
                    uint a = gsOff[j];
                    uint va = vvMember(off, a);
                    if (gNext[flLevelAt(va)] != l) continue;
                    // (The lane's first hit is the one it keeps: x == lane.)
                    bool mine = x == lane;
                    if (mine)
                        latResidual(va, vvHitSmall(0u, 0u, a, 0u), x - gsOff[LAT_PRE + j], asfloat(gStatic[va * 4u + 3u].z), parity, a, ph, ph != 0u, true, kept);
                    else
                    {
                        LatKept other = kept;
                        latResidual(va, vvHitSmall(0u, 0u, a, 0u), x - gsOff[LAT_PRE + j], asfloat(gStatic[va * 4u + 3u].z), parity, a, ph, false, false, other);
                    }
                }
                sync();
            }
        }
    }
    // The two-body counts the hits wrote, once all levels are applied.
    if (gP.vvSplit != 0u)
    {
        for (uint x = lane, xStop = EB_END(x, T, LANES); x < xStop; x += LANES)
        {
            uint j = latOwner(H, x);
            uint a = gsOff[j];
            uint base = vvHitSmall(0u, 0u, a, 0u);
            uint i = x - gsOff[LAT_PRE + j];
            uint4 tn = gStatic[gP.triBase + gNext[base + 2u * i]];
            gAcc[vvSplitTri(tn.x)].store(0);
            gAcc[vvSplitTri(tn.y)].store(0);
            gAcc[vvSplitTri(tn.z)].store(0);
            gAcc[vvSplitPoint(a, gNext[base + 2u * i + 1u])].store(0);
        }
    }
    gVvTag = 0u;
    gFl = 0u;
    sync();
    for (uint m = lane, mStop = EB_END(m, k, LANES); m < mStop; m += LANES) flOwnerEnd(vvMember(off, m));
    flArgsZero(lane);
}

// --- The fleet front's fused dispatches (PERF_FLEET_FUSE) ---
// Two of the front's dispatches are short and on the substep's chain of dependent
// dispatches: the pairs' scans (a thread a pair, then the plan's one workgroup) and the
// candidates' sorted copy (a thread a place of the run, after the grids' workgroups). With
// the switch the plan's workgroup scans the pairs itself (its lanes over them, a barrier,
// then its prefix as before) and each owner's grid workgroup makes its own candidates'
// sorted copy after its offsets: the same writes from the same functions, two dispatches
// fewer a substep (the host leaves them out).
void fuseScans(uint lane, uint s)
{
    uint pairs = gNext[flAdjFirst(gP.vehicles)] / 2u;
    for (uint p = lane, pStop = EB_END(p, pairs, LANES); p < pStop; p += LANES)
    {
        uint va = gNext[gP.vvPairs + 2u * p];
        if (!flVehicleOn(va))
        {
            if ((gNext[flBitsAt(2u * p)] & 4u) == 0u)
            {
                gNext[flBitsAt(2u * p)] = 0u;
                gNext[flBitsAt(2u * p + 1u)] = 0u;
            }
            continue;
        }
        flScanPair(p, flParity(va, s));
    }
}

void fuseSortCands(uint va, uint lane)
{
    sync();
    uint cands = flCands(va);
    for (uint e = lane, eStop = EB_END(e, cands, LANES); e < eStop; e += LANES) flSortCand(va, e);
}
