// --- The contact groups: explicit_beams_contact ---

// Group member m (sorted by vehicle index) and its partner / reset flags, its
// contact points' first run index; per group at its member offset in regions of
// one word per vehicle.
uint vvMember(uint off, uint m) { return gNext[gP.vvMembers + gP.vehicles + off + m]; }
uint vvPrefixAt(uint off, uint m) { return gP.vvMembers + 2u * gP.vehicles + off + m; }
uint vvPartnerAt(uint off, uint m) { return gP.vvMembers + 3u * gP.vehicles + off + m; }
uint vvResetAt(uint off, uint m) { return gP.vvMembers + 4u * gP.vehicles + off + m; }
uint vvTestedAt(uint i) { return gP.vvMembers + 5u * gP.vehicles + i; }

// A member's step: explicit_beams_frame's step for vehicle v (every barrier at the
// top level), with the self-collision accumulators always read (the contact pass
// adds into them), the tested-triangle list restarted per member, and a pending
// reset of the self-collision throttles (a fast approach in the last contact pass)
// applied after their update.
void contactMemberStep(uint v, uint lane, uint s, bool first)
{
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 r3 = gStatic[v * 4u + 3u];
    uint nodeOff = r0.x;
    uint nodeCount = r0.y;
    uint beamOff = r0.z;
    uint beamCount = r0.w;
    uint triOff = r1.x;
    uint triCount = r1.y;
    uint boxOff = r1.z;
    uint boxCount = r1.w;
    uint candOff = r2.x;
    uint candCount = r2.y;
    uint groupOff = r2.z;
    uint groupCount = r2.w;
    uint flagsV = r3.x;
    float ground = asfloat(r3.y);
    float range = asfloat(r3.z);
    float cell = asfloat(r3.w);
    bool fast = (flagsV & FLAG_FAST_INV_SQRT) != 0u;
#ifdef EB_AERO
    uint4 aero0 = gStatic[gP.aeroBase + v * 3u];
    uint4 aero1 = gStatic[gP.aeroBase + v * 3u + 1u];
    uint4 aero2 = gStatic[gP.aeroBase + v * 3u + 2u];
#endif
    uint4 misc = gMisc[v];
    float3 origin = asfloat(misc.xyz);
    uint step = misc.w + s;
    uint parity = step & 1u;
    bool pending = gMisc[vvRec(v, VV_FLAGS)].w != 0u;
#ifdef EB_RACE_PROBE
    probeEpoch = step * 128u + 126u;
#endif
    if (lane == 0u)
    {
        gsTestedCount[parity].store(0u);
        if (first) vvClearCounters();
    }
    sync();
    {
        // The contact pass's box of this step (world positions after integration)
        // into this parity's keys (the other parity's cleared for the next step),
        // and the frame bounds check.
        float3 lo = float3(3.0e38, 3.0e38, 3.0e38);
        float3 hi = -lo;
        bool has = false;
        for (uint i = lane; i < nodeCount; i += LANES)
        {
            nodeUpdate(nodeOff + i, gStatic[gP.nodeInfoBase + nodeOff + i], v, i, step, origin, flagsV, ground,
                       boxOff, boxCount, beamOff, groupOff, groupCount, true);
            float3 x = nf3(origin + gNodes[(nodeOff + i) * 3u].xyz);
            lo = min(lo, x);
            hi = max(hi, x);
            has = true;
        }
        if (lane < 6u) gAcc[vvBoxKey(v, parity ^ 1u) + lane].store(lane < 3u ? VV_INT_MAX : VV_INT_MIN);
        if (has)
        {
            uint key = vvBoxKey(v, parity);
            gAcc[key].min(vvKey(lo.x));
            gAcc[key + 1u].min(vvKey(lo.y));
            gAcc[key + 2u].min(vvKey(lo.z));
            gAcc[key + 3u].max(vvKey(hi.x));
            gAcc[key + 4u].max(vvKey(hi.y));
            gAcc[key + 5u].max(vvKey(hi.z));
            if (any(lo < asfloat(gMisc[vvRec(v, VV_ALLOWED_LO)].xyz)) || any(hi > asfloat(gMisc[vvRec(v, VV_ALLOWED_HI)].xyz)))
                vvReportOutside(v);
        }
    }
    for (uint g = lane; g < groupCount; g += LANES) gGroups[groupSlot(parity, groupOff, g)].store(NO_GROUP);
    if (candCount > 0u)
    {
        for (uint t = lane; t < triCount; t += LANES)
        {
            int w = s > 0u ? triangleFinish(triOff + t) : asint(gMisc[gP.vehicles + triOff + t].x);
            if (s > 0u && pending)
            {
                w = 0;
                gMisc[gP.vehicles + triOff + t].x = 0u;
            }
            if (triCount <= TRI_LIST && w <= 0) gsTested[gsTestedCount[parity].add(1u)] = t;
        }
    }
    for (uint b = lane; b < GRID_BUCKETS; b += LANES) gsCount[b].store(0u);
    sync();
    if (lane == 0u && s > 0u) gMisc[vvRec(v, VV_FLAGS)].w = 0u;
    if (cell > 0.0)
    {
        for (uint c = lane; c < candCount; c += LANES)
        {
            float3 p = gNodes[gLists[candOff + c] * 3u].xyz;
            uint b = gridBucket(int3(floor(p / cell)));
            gNext[candOff + c] = b | (gsCount[b].add(1u) << 10);
        }
    }
#ifdef EB_AERO
    for (uint w = lane; w < aero0.y; w += LANES) wingForces(aero0.x + w, origin);
    if (lane == LANES - 1u && (flagsV & FLAG_FUSEDRAG) != 0u) fuseDrag(aero1, origin, v);
    if ((flagsV & FLAG_KEYS) != 0u)
    {
        for (uint c = lane; c < aero0.w; c += LANES) commandUpdateKeys(aero0.z + c, v, parity ^ 1u);
    }
    else
    {
        for (uint c = lane; c < aero0.w; c += LANES) commandUpdate(aero0.z + c);
    }
    sync();
    if (lane == 0u) wingApply(aero0.x, aero0.y);
    if ((flagsV & FLAG_KEYS) != 0u)
    {
        for (uint k = lane; k < KEYS; k += LANES) gNext[keySlot(v, parity ^ 1u, k)] = 0u;
    }
#endif
    for (uint j = lane; j < beamCount; j += LANES) beamUpdate(beamOff + j, j, fast, groupOff, parity);
    sync();
    if (gP.grid != 0u)
    {
        uint per = GRID_BUCKETS / LANES;
        uint block = LANES / GRID_BLOCKS;
        uint sum = 0u;
        for (uint k = 0u; k < per; k++) sum += gsCount[lane * per + k].load();
        gsPart[lane] = sum;
        sync();
        if (lane < GRID_BLOCKS)
        {
            uint run = 0u;
            for (uint k = 0u; k < block; k++)
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
        uint run = gsPart[lane] + gsBlk[lane / block];
        for (uint k = 0u; k < per; k++)
        {
            gsOff[lane * per + k] = run;
            run += gsCount[lane * per + k].load();
        }
        sync();
        if (cell > 0.0)
        {
            for (uint c = lane; c < candCount; c += LANES)
            {
                uint br = gNext[candOff + c];
                uint h = gLists[candOff + c];
                gBeams[gP.gridBase + candOff + gsOff[br & (GRID_BUCKETS - 1u)] + (br >> 10)] =
                    float4(gNodes[h * 3u].xyz, asfloat(h));
            }
        }
        sync();
    }
#ifdef EB_RACE_PROBE
    probeEpoch = step * 128u + 127u;
#endif
    if (candCount > 0u && cell > 0.0)
    {
        uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
        for (uint p = lane; p < listed * GRID_SUBS; p += LANES)
        {
            uint t = triCount <= TRI_LIST ? gsTested[p / GRID_SUBS] : p / GRID_SUBS;
            triangleGrid(triOff + t, p % GRID_SUBS, origin, range, cell, candOff, candCount, beamOff, groupOff,
                         groupCount, parity);
        }
    }
    else if (candCount > 0u)
    {
        uint lanes = clamp(LANES / max(triCount, 1u), 1u, candCount);
        uint stride = (LANES / lanes) * lanes;
        for (uint p = lane; p < triCount * lanes && lane < stride; p += stride)
            triangleTest(triOff + p / lanes, p % lanes, lanes, origin, range, candOff, candCount, beamOff, groupOff,
                         groupCount, parity);
    }
#ifdef EB_DET_HITS
    sync();
    if (lane == 0u) detApply(v, origin, range, beamOff, groupOff, groupCount, parity);
#endif
#ifdef EB_AERO
    if (lane == LANES - 1u && (flagsV & FLAG_SLIDES) != 0u) slidePass(aero2.x, aero2.y, origin, groupOff, parity);
#endif
    sync();
}

// A member's frame start: the wing check and the origin rebase.
void contactMemberBegin(uint v, uint lane)
{
    uint4 r0 = gStatic[v * 4u];
#ifdef EB_AERO
    uint4 aero0 = gStatic[gP.aeroBase + v * 3u];
    {
        float3 o = asfloat(gMisc[v].xyz);
        for (uint w = lane; w < aero0.y; w += LANES) wingCheck(aero0.x + w, o);
    }
#endif
    float3 first = gNodes[r0.x * 3u].xyz;
    sync();
    if (length(first) > REBASE_RADIUS)
    {
        for (uint i = lane; i < r0.y; i += LANES)
        {
            float4 P = gNodes[(r0.x + i) * 3u];
            gNodes[(r0.x + i) * 3u] = float4(P.xyz - first, P.w);
        }
        if (lane == 0u)
        {
            uint4 m = gMisc[v];
            gMisc[v] = uint4(asuint(asfloat(m.xyz) + first), m.w);
        }
    }
    sync();
}

// A member's frame end: the last step's throttle update (and a pending reset), the
// last step's group breaks, the steps run; its box keys cleared for a later frame.
void contactMemberEnd(uint v, uint lane)
{
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 misc = gMisc[v];
    bool pending = gMisc[vvRec(v, VV_FLAGS)].w != 0u;
    if (gP.steps > 0u && r2.y > 0u)
    {
        for (uint t = lane; t < r1.y; t += LANES)
        {
            triangleFinish(r1.x + t);
            if (pending) gMisc[gP.vehicles + r1.x + t].x = 0u;
        }
    }
    if (gP.steps > 0u && r2.w > 0u)
    {
        uint last = (misc.w + gP.steps - 1u) & 1u;
        for (uint j = lane; j < r0.w; j += LANES) groupDrops(r0.z + j, r0.z, r2.z, last, true);
    }
    if (lane < 12u) gAcc[gP.vvAcc + v * 12u + lane].store((lane % 6u) < 3u ? VV_INT_MAX : VV_INT_MIN);
    sync();
    if (lane == 0u)
    {
        gMisc[v] = uint4(misc.xyz, misc.w + gP.steps);
        gMisc[vvRec(v, VV_FLAGS)].w = 0u;
    }
}

// A collision triangle as the contact pass sees it (spec §5).
struct VvTri
{
    uint4 tn;
    float3 p0;
    float3 n;
    float3 ra;
    float3 rb;
    float3 rn;
    bool solvable;
    float3 lo;
    float3 hi;
};

VvTri vvTriangle(uint gt, float3 origin, float r)
{
    VvTri t;
    t.tn = gStatic[gP.triBase + gt];
    t.p0 = nf3(origin + gNodes[t.tn.x * 3u].xyz);
    float3 p1 = nf3(origin + gNodes[t.tn.y * 3u].xyz);
    float3 p2 = nf3(origin + gNodes[t.tn.z * 3u].xyz);
    float3 u = ssub3(p1, t.p0);
    float3 w = ssub3(p2, t.p0);
    float3 cr = scross(u, w);
    float cl = ssqrt(sdot(cr, cr));
    t.n = cl > 1e-8 ? sscale(cr, sdiv(1.0, cl)) : cr;
    float det = sdot(u, scross(w, t.n));
    t.solvable = abs(det) > 1e-6;
    float rd = t.solvable ? sdiv(1.0, det) : 0.0;
    t.ra = sscale(scross(w, t.n), rd);
    t.rb = sscale(scross(t.n, u), rd);
    t.rn = sscale(scross(u, w), rd);
    float3 r3 = float3(r, r, r);
    t.lo = ssub3(min(min(t.p0, p1), p2), r3);
    t.hi = sadd3(max(max(t.p0, p1), p2), r3);
    return t;
}

// Candidate h (global node, world position p, of vehicle vh) against triangle gt of
// the owner: the hit test, the side vote, the contact force into the accumulators
// (h, and the triangle's nodes by their weights). Returns whether it hit.
bool vvTry(VvTri t, uint gt, uint h, float3 p, uint vh, float r, uint parity)
{
    float3 q = ssub3(p, t.p0);
    float alpha = 0.0;
    float beta = 0.0;
    float delta = 0.0;
    if (t.solvable)
    {
        alpha = sdot(t.ra, q);
        beta = sdot(t.rb, q);
        delta = sdot(t.rn, q);
    }
    float gamma = nf(nf(1.0 - alpha) - beta);
    if (!(alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 && abs(delta) <= r)) return false;
    // The side: a vote of h and its spawn-time beam neighbours (its incidence
    // entries' far ends, in beam order).
    uint4 info = gStatic[gP.nodeInfoBase + h];
    int score = delta >= 0.0 ? 3 : -3;
    if (info.z > 3u)
    {
        float3 ob = asfloat(gMisc[vh].xyz);
        uint base = gP.vvNbr + info.y;
        // Four independent loads at a time (the far ends are a contiguous list).
        uint k = 0u;
        for (; k + 4u <= info.z; k += 4u)
        {
            float3 y0 = gNodes[gLists[base + k] * 3u].xyz;
            float3 y1 = gNodes[gLists[base + k + 1u] * 3u].xyz;
            float3 y2 = gNodes[gLists[base + k + 2u] * 3u].xyz;
            float3 y3 = gNodes[gLists[base + k + 3u] * 3u].xyz;
            score += sdot(t.n, ssub3(nf3(ob + y0), t.p0)) >= 0.0 ? 1 : -1;
            score += sdot(t.n, ssub3(nf3(ob + y1), t.p0)) >= 0.0 ? 1 : -1;
            score += sdot(t.n, ssub3(nf3(ob + y2), t.p0)) >= 0.0 ? 1 : -1;
            score += sdot(t.n, ssub3(nf3(ob + y3), t.p0)) >= 0.0 ? 1 : -1;
        }
        for (; k < info.z; k++)
            score += sdot(t.n, ssub3(nf3(ob + gNodes[gLists[base + k] * 3u].xyz), t.p0)) >= 0.0 ? 1 : -1;
    }
    float3 nn = t.n;
    if (score < 0)
    {
        nn = -t.n;
        delta = -delta;
    }
    float pen = nf(r - delta);
    float4 P = gNodes[h * 3u];
    float4 V = gNodes[h * 3u + 1u];
    float4 P0 = gNodes[t.tn.x * 3u];
    float4 P1 = gNodes[t.tn.y * 3u];
    float4 P2 = gNodes[t.tn.z * 3u];
    float3 vt = sadd3(sadd3(sscale(gNodes[t.tn.y * 3u + 1u].xyz, alpha), sscale(gNodes[t.tn.z * 3u + 1u].xyz, beta)),
                      sscale(gNodes[t.tn.x * 3u + 1u].xyz, gamma));
    float3 v = ssub3(V.xyz, vt);
    float mt = nf(nf(smul(P1.w, alpha) + smul(P2.w, beta)) + smul(P0.w, gamma));
    float m = sdiv(smul(P.w, mt), nf(P.w + mt));
    uint4 rb0 = gStatic[vh * 4u];
    uint4 rb2 = gStatic[vh * 4u + 2u];
    float3 fh = sadd3(beamSum(gNodes[h * 3u + 2u].xyz, info, rb0.z, rb2.z, rb2.w, parity, false), accForce(h));
#ifdef EB_AERO
    uint slot1 = info.x >> NODE_SLOT_SHIFT;
    if (slot1 != 0u) fh = sadd3(fh, gBeams[gP.slideAcc + min(slot1 - 1u, max(gP.slotTotal, 1u) - 1u)].xyz);
#endif
    float3 f = contactForce(fh, v, m, nn, boxSurface(), pen, V.w);
#ifdef EB_RACE_PROBE
    probeTouch(h, true);
    probeTouch(t.tn.x, false);
    probeTouch(t.tn.y, false);
    probeTouch(t.tn.z, false);
#endif
    accAdd(h, f);
    accAdd(t.tn.y, -sscale(f, alpha));
    accAdd(t.tn.z, -sscale(f, beta));
    accAdd(t.tn.x, -sscale(f, gamma));
    return true;
}

// The owner's tested triangle t (list position p of GRID_SUBS lanes, lane `sub`)
// against the owner's candidates (partner contact points inside the owner's box
// grown by its range, compacted at the run offset): the box test, then the hit
// test, vote and force; marks the triangle's record (z: had candidates, w: hit).
// The hit test of vvTry alone.
bool vvInBand(VvTri t, float3 p, float r)
{
    float3 q = ssub3(p, t.p0);
    float alpha = 0.0;
    float beta = 0.0;
    float delta = 0.0;
    if (t.solvable)
    {
        alpha = sdot(t.ra, q);
        beta = sdot(t.rb, q);
        delta = sdot(t.rn, q);
    }
    float gamma = nf(nf(1.0 - alpha) - beta);
    return alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 && abs(delta) <= r;
}

// Hit list entry `i` of the group's list of this owner (q its counter parity).
uint vvHitAt(uint runOff, uint i) { return gP.vvHits + 2u * (2u * runOff + i); }

void vvTriangleCands(uint va, uint gt, uint sub, uint subs, uint runOff, uint points, uint cands, float r, uint parity,
                     uint q)
{
    VvTri t = vvTriangle(gt, asfloat(gMisc[va].xyz), r);
    bool cand = false;
    bool hit = false;
    for (uint e = sub; e < cands; e += subs)
    {
        float4 E = gBeams[gP.vvRun + runOff + e];
        if (any(E.xyz < t.lo) || any(E.xyz > t.hi)) continue;
        uint h = asuint(E.w);
        cand = true;
        if (!vvInBand(t, E.xyz, r)) continue;
        hit = true;
        // The force later, one lane per hit (the vote and the force read many
        // nodes); a full list applies it here.
        uint i = gsVvHits[q].add(1u);
        if (i < 2u * points)
        {
            gNext[vvHitAt(runOff, i)] = gt;
            gNext[vvHitAt(runOff, i) + 1u] = h;
        }
        else
        {
            vvTry(t, gt, h, E.xyz, gStatic[gP.nodeInfoBase + h].w, r, parity);
        }
    }
    if (cand) gMisc[gP.vvTri + gt].z = 1u;
    if (hit) gMisc[gP.vvTri + gt].w = 1u;
}

// The listed hits of owner va: one lane each, the vote and the force.
void vvApplyHits(uint va, uint runOff, uint points, uint hits, float r, uint parity, uint lane)
{
    float3 oa = asfloat(gMisc[va].xyz);
    for (uint i = lane; i < min(hits, 2u * points); i += LANES)
    {
        uint gt = gNext[vvHitAt(runOff, i)];
        uint h = gNext[vvHitAt(runOff, i) + 1u];
        uint vh = gStatic[gP.nodeInfoBase + h].w;
        VvTri t = vvTriangle(gt, oa, r);
        vvTry(t, gt, h, nf3(asfloat(gMisc[vh].xyz) + gNodes[h * 3u].xyz), vh, r, parity);
    }
}

#ifdef EB_DET_HITS
// Tests only: vvApplyHits on one lane, the hits sorted by (triangle, node), so the
// order is fixed and each hit sees every earlier one.
void vvApplyHitsDet(uint va, uint runOff, uint points, uint hits, float r, uint parity, uint lane)
{
    if (lane != 0u) return;
    uint n = min(hits, 2u * points);
    for (uint i = 1u; i < n; i++)
    {
        uint gt = gNext[vvHitAt(runOff, i)];
        uint h = gNext[vvHitAt(runOff, i) + 1u];
        uint j = i;
        while (j > 0u)
        {
            uint pg = gNext[vvHitAt(runOff, j - 1u)];
            uint ph = gNext[vvHitAt(runOff, j - 1u) + 1u];
            if (pg < gt || (pg == gt && ph <= h)) break;
            gNext[vvHitAt(runOff, j)] = pg;
            gNext[vvHitAt(runOff, j) + 1u] = ph;
            j--;
        }
        gNext[vvHitAt(runOff, j)] = gt;
        gNext[vvHitAt(runOff, j) + 1u] = h;
    }
    float3 oa = asfloat(gMisc[va].xyz);
    for (uint i = 0u; i < n; i++)
    {
        uint gt = gNext[vvHitAt(runOff, i)];
        uint h = gNext[vvHitAt(runOff, i) + 1u];
        uint vh = gStatic[gP.nodeInfoBase + h].w;
        VvTri t = vvTriangle(gt, oa, r);
        vvTry(t, gt, h, nf3(asfloat(gMisc[vh].xyz) + gNodes[h * 3u].xyz), vh, r, parity);
    }
}
#endif

// A tested triangle's throttle after its tests (a hit → 0, no candidate → + 1).
void vvThrottleEnd(uint gt)
{
    uint4 c = gMisc[gP.vvTri + gt];
    int cwait = asint(c.x);
    if (c.w != 0u) cwait = 0;
    else if (c.z == 0u) cwait = cwait + 1;
    gMisc[gP.vvTri + gt] = uint4(asuint(cwait), c.y, 0u, 0u);
}

// Triangle gt of owner va: its box grown by the range overlaps the box `lo..hi`
// (the partners' boxes' union), so it may have candidates.
bool vvNearBox(uint gt, uint va, float r, float3 blo, float3 bhi)
{
    uint4 tn = gStatic[gP.triBase + gt];
    float3 o = asfloat(gMisc[va].xyz);
    float3 p0 = nf3(o + gNodes[tn.x * 3u].xyz);
    float3 p1 = nf3(o + gNodes[tn.y * 3u].xyz);
    float3 p2 = nf3(o + gNodes[tn.z * 3u].xyz);
    float3 r3 = float3(r, r, r);
    return vvOverlap(ssub3(min(min(p0, p1), p2), r3), sadd3(max(max(p0, p1), p2), r3), blo, bhi);
}

void vvClearCounters()
{
    for (uint q = 0u; q < 2u; q++)
    {
        gsVvOffered[q].store(0u);
        gsVvListed[q].store(0u);
        gsVvCands[q].store(0u);
        gsVvHits[q].store(0u);
        gsVvReset[q] = 0u;
    }
}

// The contact pass after every member's step (their boxes already in the step's
// keys): each owner in vehicle order, three barriers each. Phase S: the partner
// scan over the other members (and the previous owner's throttle updates); phase
// P: contact throttle resets, the owner's throttle step listing its tested
// triangles, and its candidates compacted; phase T: tested triangles against the
// candidates.
void vvPass(uint k, uint off, uint runOff, uint points, uint step, uint lane)
{
    uint parity = step & 1u;
    float3 m3 = float3(VV_BOX_MARGIN, VV_BOX_MARGIN, VV_BOX_MARGIN);
    uint prevTri = 0u;
    uint prevListed = 0u;
    uint prevVa = 0u;
    float prevRange = 0.0;
    for (uint a = 0u; a < k; a++)
    {
        uint q = a & 1u;
        uint va = vvMember(off, a);
        uint key = vvBoxKey(va, parity);
        float3 blo = float3(vvUnkey(gAcc[key].load()), vvUnkey(gAcc[key + 1u].load()), vvUnkey(gAcc[key + 2u].load()));
        float3 bhi = float3(vvUnkey(gAcc[key + 3u].load()), vvUnkey(gAcc[key + 4u].load()), vvUnkey(gAcc[key + 5u].load()));
        float3 alo = ssub3(blo, m3);
        float3 ahi = sadd3(bhi, m3);
        float3 va0 = gNodes[gStatic[va * 4u].x * 3u + 1u].xyz;
        // Phase S (with the previous owner's hits).
        for (uint i = lane; i < prevListed; i += LANES) vvThrottleEnd(prevTri + gNext[vvTestedAt(prevTri + i)]);
#ifdef EB_RACE_PROBE
        probeEpoch = step * 128u + (a > 0u ? a - 1u : 0u);
#endif
#ifdef EB_DET_HITS
        if (a > 0u) vvApplyHitsDet(prevVa, runOff, points, gsVvHits[q ^ 1u].load(), prevRange, parity, lane);
#else
        if (a > 0u) vvApplyHits(prevVa, runOff, points, gsVvHits[q ^ 1u].load(), prevRange, parity, lane);
#endif
        for (uint b = lane; b < k; b += LANES)
        {
            uint vb = vvMember(off, b);
            uint kb = vvBoxKey(vb, parity);
            float3 olo = ssub3(float3(vvUnkey(gAcc[kb].load()), vvUnkey(gAcc[kb + 1u].load()), vvUnkey(gAcc[kb + 2u].load())), m3);
            float3 ohi = sadd3(float3(vvUnkey(gAcc[kb + 3u].load()), vvUnkey(gAcc[kb + 4u].load()), vvUnkey(gAcc[kb + 5u].load())), m3);
            bool partner = b != a && vvOverlap(alo, ahi, olo, ohi);
            bool reset = false;
            if (partner)
            {
                gsVvOffered[q].add(vvCpCount(vb));
                float3 d = ssub3(va0, gNodes[gStatic[vb * 4u].x * 3u + 1u].xyz);
                if (sdot(d, d) > VV_APPROACH_SQ)
                {
                    reset = true;
                    gsVvReset[q] = 1u;
                    gMisc[vvRec(vb, VV_FLAGS)].w = 1u;
                    gMisc[vvRec(va, VV_FLAGS)].w = 1u;
                }
            }
            gNext[vvPartnerAt(off, b)] = partner ? 1u : 0u;
            gNext[vvResetAt(off, b)] = reset ? 1u : 0u;
        }
        sync();
        // Phase P.
        uint offered = gsVvOffered[q].load();
        bool ownerReset = gsVvReset[q] != 0u;
        if (lane == 0u)
        {
            gsVvOffered[q ^ 1u].store(0u);
            gsVvListed[q ^ 1u].store(0u);
            gsVvCands[q ^ 1u].store(0u);
            gsVvHits[q ^ 1u].store(0u);
            gsVvReset[q ^ 1u] = 0u;
        }
        for (uint m = 0u; m < k; m++)
        {
            if (m == a || gNext[vvResetAt(off, m)] == 0u) continue;
            uint4 r1 = gStatic[vvMember(off, m) * 4u + 1u];
            for (uint t = lane; t < r1.y; t += LANES) gMisc[gP.vvTri + r1.x + t].x = 0u;
        }
        uint4 ra1 = gStatic[va * 4u + 1u];
        float range = asfloat(gStatic[va * 4u + 3u].z);
        // The partners' boxes' union (their contact points all lie in it).
        float3 plo = float3(3.0e38, 3.0e38, 3.0e38);
        float3 phi = -plo;
        for (uint m = 0u; m < k; m++)
        {
            if (m == a || gNext[vvPartnerAt(off, m)] == 0u) continue;
            uint kb = vvBoxKey(vvMember(off, m), parity);
            plo = min(plo, float3(vvUnkey(gAcc[kb].load()), vvUnkey(gAcc[kb + 1u].load()), vvUnkey(gAcc[kb + 2u].load())));
            phi = max(phi, float3(vvUnkey(gAcc[kb + 3u].load()), vvUnkey(gAcc[kb + 4u].load()), vvUnkey(gAcc[kb + 5u].load())));
        }
        for (uint t = lane; t < ra1.y; t += LANES)
        {
            uint4 c = gMisc[gP.vvTri + ra1.x + t];
            int cwait = ownerReset ? 0 : asint(c.x);
            int cidle = asint(c.y);
            if (offered == 0u)
            {
                if (ownerReset) gMisc[gP.vvTri + ra1.x + t].x = 0u;
                continue;
            }
            if (cwait > 0)
            {
                gMisc[gP.vvTri + ra1.x + t] = uint4(asuint(cwait - 1), asuint(cidle + 1), 0u, 0u);
                continue;
            }
            cwait = VV_MAX_WAIT < cidle ? VV_MAX_WAIT : cidle;
            // A triangle whose box (grown by the range) meets no partner's box has
            // no candidate: its throttle update at once, no tests.
            if (!vvNearBox(ra1.x + t, va, range, plo, phi))
            {
                gMisc[gP.vvTri + ra1.x + t] = uint4(asuint(cwait + 1), 0u, 0u, 0u);
                continue;
            }
            gMisc[gP.vvTri + ra1.x + t] = uint4(asuint(cwait), 0u, 0u, 0u);
            gNext[vvTestedAt(ra1.x + gsVvListed[q].add(1u))] = t;
        }
        if (offered > 0u)
        {
            float3 r3 = float3(range, range, range);
            float3 clo = ssub3(blo, r3);
            float3 chi = sadd3(bhi, r3);
            for (uint m = 0u; m < k; m++)
            {
                if (m == a || gNext[vvPartnerAt(off, m)] == 0u) continue;
                uint v = vvMember(off, m);
                uint4 cp = gStatic[gP.vvStatic + v];
                float3 o = asfloat(gMisc[v].xyz);
                for (uint i = lane; i < cp.y; i += LANES)
                {
                    uint h = gLists[cp.x + i];
                    float3 x = nf3(o + gNodes[h * 3u].xyz);
                    if (any(x < clo) || any(x > chi)) continue;
                    gBeams[gP.vvRun + runOff + gsVvCands[q].add(1u)] = float4(x, asfloat(h));
                }
            }
        }
        sync();
        // Phase T.
#ifdef EB_RACE_PROBE
        probeEpoch = step * 128u + a;
#endif
        uint listed = gsVvListed[q].load();
        uint cands = gsVvCands[q].load();
        // Lanes per triangle: all the lanes over the listed triangles.
        uint subs = clamp(LANES / max(listed, 1u), 1u, 64u);
        for (uint p = lane; p < listed * subs; p += LANES)
        {
            uint t = gNext[vvTestedAt(ra1.x + p / subs)];
            vvTriangleCands(va, ra1.x + t, p % subs, subs, runOff, points, cands, range, parity, q);
        }
        sync();
        prevTri = ra1.x;
        prevListed = listed;
        prevVa = va;
        prevRange = range;
    }
    for (uint i = lane; i < prevListed; i += LANES) vvThrottleEnd(prevTri + gNext[vvTestedAt(prevTri + i)]);
#ifdef EB_RACE_PROBE
    probeEpoch = step * 128u + (k > 0u ? k - 1u : 0u);
#endif
#ifdef EB_DET_HITS
    if (k > 0u) vvApplyHitsDet(prevVa, runOff, points, gsVvHits[(k - 1u) & 1u].load(), prevRange, parity, lane);
#else
    if (k > 0u) vvApplyHits(prevVa, runOff, points, gsVvHits[(k - 1u) & 1u].load(), prevRange, parity, lane);
#endif
    // (The counters are cleared by the next step's first member step, before its
    // first barrier.)
}

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_contact(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    contactGroup(gid.x, lid.x);
}

// Contact group g: its members stepped together with the contact pass.
void contactGroup(uint g, uint lane)
{
    uint4 G = gMisc[gP.vvGroups + g];
    uint off = G.x;
    uint runOff = G.z;
    uint points = G.w;
    if (lane == 0u) gsUniformLanes = G.y;
    sync();
    uint k = gsUniformLanes;
    sync();
    // Members in vehicle order (a rank sort), their run prefixes and member
    // indices.
    for (uint i = lane; i < k; i += LANES)
    {
        uint v = gNext[gP.vvMembers + off + i];
        uint rank = 0u;
        for (uint j = 0u; j < k; j++) rank += gNext[gP.vvMembers + off + j] < v ? 1u : 0u;
        gNext[gP.vvMembers + gP.vehicles + off + rank] = v;
        gMisc[vvRec(v, VV_MEMBER)].x = rank;
    }
    sync();
    if (lane == 0u)
    {
        uint run = 0u;
        for (uint m = 0u; m < k; m++)
        {
            gNext[vvPrefixAt(off, m)] = run;
            run += vvCpCount(vvMember(off, m));
        }
        vvClearCounters();
    }
    sync();
    for (uint m = 0u; m < k; m++) contactMemberBegin(vvMember(off, m), lane);
    uint stepBase = gMisc[vvMember(off, 0u)].w;
    for (uint s = 0u; s < gP.steps; s++)
    {
        for (uint m = 0u; m < k; m++) contactMemberStep(vvMember(off, m), lane, s, m == 0u);
        vvPass(k, off, runOff, points, stepBase + s, lane);
    }
    for (uint m = 0u; m < k; m++) contactMemberEnd(vvMember(off, m), lane);
}
