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
    for (uint p = lane, pStop = EB_END(369, p, pairs, LANES); p < pStop; p += LANES) code = max(code, fkPairCheck(p));
    // (Another path's list in an owner's region: the kept one is gone.)
    for (uint v = lane, vStop = EB_END(370, v, nv, LANES); v < vStop; v += LANES)
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
    for (uint v = lane, vStop = EB_END(371, v, nv, LANES); v < vStop; v += LANES)
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
        for (uint v = lane, vStop = EB_END(372, v, nv, LANES); v < vStop; v += LANES)
        {
            uint n = 0u;
            if (flVehicleOn(v))
            {
                uint e1 = gNext[flAdjFirst(v + 1u)];
                for (uint e = gNext[flAdjFirst(v)], eStop = EB_END(373, e, e1, 1u); e < eStop; e++)
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
        for (uint p = lane, pStop = EB_END(374, p, pairs, LANES); p < pStop; p += LANES)
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
    for (uint e = gNext[flAdjFirst(va)], eStop = EB_END(375, e, e1, 1u); e < eStop; e++)
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
    // (Contact offsets: the raw corners' box grown by the vehicle's largest offset covers
    // its paint corners now and at any later pass, whatever the offsets' directions.)
    float om = contactOffMax(va);
    float base = asfloat(gStatic[va * 4u + 3u].z) + CLIP_DEPTH + (wildTri ? FK_WILD : FK_CALM) + fkBetween() + om;
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
              + 4.0e-6 * max(m3.x, max(m3.y, m3.z)) + om;
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
        for (uint e = sub, eStop = EB_END(376, e, cands, subs); e < eStop; e += subs)
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
    for (uint i = sub, iStop = EB_END(377, i, cells, subs); i < iStop; i += subs)
    {
        int3 cc = c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z)));
        uint b = gridBucket(cc);
        uint e0 = gNext[off + b];
        uint e1 = gNext[off + b + 1u];
        for (uint e = e0, eGuard = 0u; e < e1; e++)
        {
            EB_PASS(378, eGuard);
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
            if (flVehicleOn(va))
            {
                flTriangle(va, tid, flParity(va, s));
                // (A listed triangle over the cells, in a pass that walks: marked and into
                // the fleet's list of big triangles, flWalkBig's threads.)
                if (flWalkBox() && mode == FK_OFF && gNext[fkListed(tid)] != 0u && flBigHint(tid, va))
                {
                    uint k = uint(gAcc[flCtl(FL_CTL_BIG)].add(1));
                    if (k < gP.totalTris)
                    {
                        gNext[flListAt(k)] = tid;
                        gNext[fkListed(tid)] = 3u;
                    }
                }
            }
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

