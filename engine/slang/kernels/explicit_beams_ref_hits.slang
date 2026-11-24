// --- Self-collision hits in the specification's order (see tryContact) ---

// Rounds of selfRound before one lane finishes what is left in key order.
static const uint SELF_ROUNDS = 32u;

// Node `h`'s roles in the hit recorded at `at` (key `key`): bit 0 the contacter,
// bits 1..3 the triangle's n1, n2, n0 (a triangle may name a node twice); 0 none.
uint hitRoles(uint at, uint key, uint h, uint nodeOff)
{
    if (nodeOff + (key & 0xFFFFu) == h) return 1u;
    return (gNext[at + 9u] == h ? 2u : 0u) | (gNext[at + 10u] == h ? 4u : 0u) | (gNext[at + 8u] == h ? 8u : 0u);
}

// The accumulator shares (fixed point) node roles `roles` get from the computed hit
// at record `at`, as accAdd adds them.
int3 hitShareFixed(uint at, uint roles)
{
    float alpha = asfloat(gNext[at + 1u]);
    float beta = asfloat(gNext[at + 2u]);
    float gamma = nf(nf(1.0 - alpha) - beta);
    float3 f = asfloat(uint3(gNext[at + 3u], gNext[at + 4u], gNext[at + 5u]));
    int3 sum = int3(0, 0, 0);
    for (uint role = 1u; role <= 4u; role++)
    {
        if ((roles & (1u << (role - 1u))) != 0u) sum += toFixed3(hitShare(f, role, alpha, beta, gamma));
    }
    return sum;
}

// A hit's contact force: contacter (velocity V, position and mass P) against the
// triangle (masses P0..P2, velocities v0..v2) at weights (alpha, beta, gamma),
// distance delta along normal nn; its force is fb (beams) plus a3 (accumulator).
// Strict, in the CPU reference's order (the test-only EB_DET_HITS build of hitForce
// keeps the frozen baseline's expressions instead).
float3 hitContact(float4 V, float4 P, float4 P0, float4 P1, float4 P2, float3 v0, float3 v1, float3 v2, float alpha,
                  float beta, float gamma, float delta, float3 nn, float3 fb, float3 a3, float r)
{
    float3 vv = ssub3(V.xyz, sadd3(sadd3(sscale(v1, alpha), sscale(v2, beta)), sscale(v0, gamma)));
    float mt = nf(nf(smul(alpha, P1.w) + smul(beta, P2.w)) + smul(gamma, P0.w));
    float m = sdiv(smul(P.w, mt), nf(P.w + mt));
    return contactForce(sadd3(fb, a3), vv, m, nn, boxSurface(), nf(r - delta), V.w);
}

// The force of the hit at record `at` (key `key`) of vehicle v: its node's force,
// with `sum` the fixed-point shares of every earlier hit on that node (as the
// accumulator would hold them had the hits been added in order), then the contact
// force from the geometry the finding lane stored; stored with `round`.
void hitForce(uint v, uint at, uint key, int3 sum, uint round, float3 origin, float r, uint beamOff, uint groupOff,
              uint groupCount, uint parity)
{
    uint h = gStatic[v * 4u].x + (key & 0xFFFFu);
    float3 nn = asfloat(uint3(gNext[at + 12u], gNext[at + 13u], gNext[at + 14u]));
    float pen = asfloat(gNext[at + 15u]);
    float3 vv = asfloat(uint3(gNext[at + 16u], gNext[at + 17u], gNext[at + 18u]));
    float m = asfloat(gNext[at + 19u]);
    float friction = asfloat(gNext[at + 20u]);
    // The accumulator as it would read with the earlier hits added (accForce). It
    // holds nothing else in this phase: the node phase took and cleared it, and
    // vehicle contact adds after the vehicles' steps.
    int3 acc = sum;
    float3 fb = beamSum(gNodes[h * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + h], beamOff, groupOff, groupCount, parity,
                        false);
    float3 a3 = float3(float(acc.x), float(acc.y), float(acc.z)) * (1.0 / FIXED);
#ifdef EB_DET_HITS
    float3 f = contactForce(fb + a3, vv, m, nn, boxSurface(), pen, friction);
#else
    float3 f = contactForce(sadd3(fb, a3), vv, m, nn, boxSurface(), pen, friction);
#endif
    gNext[at + 3u] = asuint(f.x);
    gNext[at + 4u] = asuint(f.y);
    gNext[at + 5u] = asuint(f.z);
    gNext[at + 6u] = round;
}

// The computed hit at record `at` adds its shares to the accumulators.
void selfAddOne(uint at, uint key, uint nodeOff)
{
    float alpha = asfloat(gNext[at + 1u]);
    float beta = asfloat(gNext[at + 2u]);
    float gamma = nf(nf(1.0 - alpha) - beta);
    float3 f = asfloat(uint3(gNext[at + 3u], gNext[at + 4u], gNext[at + 5u]));
    accAdd(nodeOff + (key & 0xFFFFu), f);
    accAdd(gNext[at + 9u], hitShare(f, 2u, alpha, beta, gamma));
    accAdd(gNext[at + 10u], hitShare(f, 3u, alpha, beta, gamma));
    accAdd(gNext[at + 8u], hitShare(f, 4u, alpha, beta, gamma));
}

// One round over vehicle v's hits (lane `sub` of `lanes`): every open hit whose
// earlier hits (smaller keys) on its node were computed in an earlier round is
// computed now. `round` 0 takes any hit whose earlier hits on its node are computed
// (the one-lane pass, in key order). Returns the hits left open.
uint selfRound(uint v, uint sub, uint lanes, uint round, float3 origin, float r, uint beamOff, uint groupOff,
               uint groupCount, uint parity)
{
    uint count = uint(gAcc[hitAcc(v)].load());
    if (count > HIT_CAP) return 0u;
    uint base = hitList(v);
    uint nodeOff = gStatic[v * 4u].x;
    uint open = 0u;
    for (uint i = sub; i < count; i += lanes)
    {
        uint at = base + i * HIT_WORDS;
        if (gNext[at + 6u] != 0u) continue;
        uint key = gNext[at];
        uint h = nodeOff + (key & 0xFFFFu);
        int3 sum = int3(0, 0, 0);
        bool ready = true;
        for (uint j = 0u; j < count; j++)
        {
            uint bj = base + j * HIT_WORDS;
            uint kj = gNext[bj];
            if (kj >= key) continue;
            uint role = hitRoles(bj, kj, h, nodeOff);
            if (role == 0u) continue;
            uint rj = gNext[bj + 6u];
            if (rj == 0u || (round != 0u && rj == round))
            {
                ready = false;
                break;
            }
            sum += hitShareFixed(bj, role);
        }
        if (!ready)
        {
            open++;
            continue;
        }
        uint done = round != 0u ? round : SELF_ROUNDS + 1u;
        if (gNext[at + 7u] != 0u && all(sum == int3(0, 0, 0)))
            gNext[at + 6u] = done;
        else
            hitForce(v, at, key, sum, done, origin, r, beamOff, groupOff, groupCount, parity);
        // Its shares now: no later hit reads the accumulators (they sum the shares
        // from the records), so the adds need no order.
        selfAddOne(at, key, nodeOff);
    }
    return open;
}

// Vehicle v's self-collision hits of this step, after the lanes that found them
// (and resolvePending) passed a barrier: rounds of selfRound until no hit of any
// member is open (every lane of the workgroup runs them: barriers), then each hit's
// shares into the accumulators. `on`: this lane's partition has a vehicle. More
// hits than HIT_CAP: one lane runs the whole step's self-collision in order
// (selfSerial) instead.
void selfSerial(uint v, float3 origin, float r, uint beamOff, uint groupOff, uint groupCount, uint parity);

void selfApply(uint v, bool on, uint sub, uint lanes, uint lane, float3 origin, float r, uint beamOff, uint groupOff,
               uint groupCount, uint parity)
{
    // gsUniformHits: set by any lane that found a hit (tryContact), read after the
    // caller's barrier; none: nothing to apply, no further barrier.
    uint any = gsUniformHits;
    if (any == 0u) return;
    // A lane with hits still open after round k sets gsUniformOpen to k: the round
    // numbers only grow within a step, so reading "at least k" after the barrier
    // is the same for every lane even if a fast lane has moved on to round k + 1.
    uint go = 0u;
    for (uint round = 1u; round <= SELF_ROUNDS; round++)
    {
        uint open = on ? selfRound(v, sub, lanes, round, origin, r, beamOff, groupOff, groupCount, parity) : 0u;
        if (open != 0u) gsUniformOpen = round;
        sync();
        uint last = gsUniformOpen;
        go = last >= round ? 1u : 0u;
        if (go == 0u) break;
    }
    if (go != 0u)
    {
        // A dependency chain longer than the rounds: one lane per vehicle finishes
        // the open hits in key order.
        if (on && sub == 0u)
        {
            for (uint it = 0u; it <= HIT_CAP; it++)
            {
                if (selfRound(v, 0u, 1u, 0u, origin, r, beamOff, groupOff, groupCount, parity) == 0u) break;
            }
        }
        sync();
    }
    if (on && sub == 0u && uint(gAcc[hitAcc(v)].load()) > HIT_CAP)
        selfSerial(v, origin, r, beamOff, groupOff, groupCount, parity);
}



// More hits in a step than the hit list holds: one lane runs vehicle v's
// self-collision of this step exactly as the specification orders it (tested
// triangles in file order, every candidate in node order, each hit added at once).
// Slow, and only for a step with more than HIT_CAP hits.
void selfSerial(uint v, float3 origin, float r, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    for (uint t = 0u; t < r1.y; t++)
    {
        uint gt = r1.x + t;
        if (asint(gMisc[gP.vehicles + gt].x) > 0) continue;
        TriGeo T = triangleGeo(gt, origin, r);
        uint4 tn = T.tn;
        for (uint k = 0u; k < r2.y; k++)
        {
            uint h = gLists[r2.x + k];
            if (h == tn.x || h == tn.y || h == tn.z) continue;
            float4 P = gNodes[h * 3u];
            float3 p = nf3(origin + P.xyz);
            if (any(p < T.lo) || any(p > T.hi)) continue;
            float alpha;
            float beta;
            float gamma;
            float delta;
            if (!triangleHit(T, p, r, alpha, beta, gamma, delta)) continue;
            gMisc[T.mi].w = 1u;
            float3 nn = T.n;
            if (delta < 0.0)
            {
                nn = -T.n;
                delta = -delta;
            }
            float4 V = gNodes[h * 3u + 1u];
            float4 P0 = gNodes[tn.x * 3u];
            float4 P1 = gNodes[tn.y * 3u];
            float4 P2 = gNodes[tn.z * 3u];
            float3 fb = beamSum(gNodes[h * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + h], beamOff, groupOff, groupCount,
                                parity, false);
            float3 f = hitContact(V, P, P0, P1, P2, gNodes[tn.x * 3u + 1u].xyz, gNodes[tn.y * 3u + 1u].xyz,
                                  gNodes[tn.z * 3u + 1u].xyz, alpha, beta, gamma, delta, nn, fb, accForce(h), r);
            accAdd(h, f);
            accAdd(tn.y, hitShare(f, 2u, alpha, beta, gamma));
            accAdd(tn.z, hitShare(f, 3u, alpha, beta, gamma));
            accAdd(tn.x, hitShare(f, 4u, alpha, beta, gamma));
        }
    }
}

// Vehicle v's pending beams of this step's beam phase (breakException), on one lane
// after a barrier: each beam's update again, in spawn order, with the exception's
// live-beam counts as the sequential pass sees them.
void resolvePending(uint v, uint parity)
{
    uint n = uint(gAcc[hitAcc(v) + 1u].load());
    if (n == 0u) return;
    uint4 r0 = gStatic[v * 4u];
    uint4 r2 = gStatic[v * 4u + 2u];
    bool fast = (gStatic[v * 4u + 3u].x & FLAG_FAST_INV_SQRT) != 0u;
    gResolving = 1u;
    if (n <= PEND_CAP)
    {
        uint base = pendList(v);
        for (uint i = 1u; i < n; i++)
        {
            uint x = gNext[base + i];
            uint j = i;
            while (j > 0u && gNext[base + j - 1u] > x)
            {
                gNext[base + j] = gNext[base + j - 1u];
                j--;
            }
            gNext[base + j] = x;
        }
        for (uint i = 0u; i < n; i++)
        {
            uint j = gNext[base + i];
            beamUpdate(r0.z + j, j, fast, r2.z, parity);
        }
    }
    else
    {
        for (uint j = 0u; j < r0.w; j++)
        {
            if (gBeams[(r0.z + j) * 2u + 1u].w == 2.0) beamUpdate(r0.z + j, j, fast, r2.z, parity);
        }
    }
    gResolving = 0u;
}



void vvReportOutside(uint v)
{
    gAcc[vvCtl(VV_OUTSIDE)].max(int(v + 1u));
}

#ifdef EB_VV_GROUPS
void contactGroup(uint g, uint lane);
#endif

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_frame(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    uint lane = lid.x;
#ifdef EB_VV_SOLO
#ifdef EB_VV_GROUPS
    // Workgroups past the vehicles step the contact groups (one dispatch: groups
    // and solo vehicles run side by side).
    if (gid.x >= gP.vehicles)
    {
        contactGroup(gid.x - gP.vehicles, lane);
        return;
    }
#endif
    // A vehicle of a contact group this frame is stepped by its group's workgroup.
    if (lane == 0u) gsUniformLanes = gMisc[vvRec(v, VV_FLAGS)].y;
    sync();
    if (gsUniformLanes != 0u) return;
#endif
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
    // The wing break check, once per frame on the positions before the rebase.
    {
        float3 o = asfloat(gMisc[v].xyz);
        for (uint w = lane; w < aero0.y; w += LANES) wingCheck(aero0.x + w, o);
    }
#endif

    // Origin rebase: every lane reads node 0 before any lane moves it.
    float3 first = gNodes[nodeOff * 3u].xyz;
    sync();
    if (length(first) > REBASE_RADIUS)
    {
        for (uint i = lane; i < nodeCount; i += LANES)
        {
            float4 P = gNodes[(nodeOff + i) * 3u];
            gNodes[(nodeOff + i) * 3u] = float4(P.xyz - first, P.w);
        }
        if (lane == 0u)
        {
            uint4 m = gMisc[v];
            gMisc[v] = uint4(asuint(asfloat(m.xyz) + first), m.w);
        }
    }
    sync();
    uint4 misc = gMisc[v];
    float3 origin = asfloat(misc.xyz);
    uint stepBase = misc.w;
    if (lane == 0u)
    {
        gsHit = 1u;
        gsTestedCount[0].store(0u);
        gsTestedCount[1].store(0u);
        // Self-collision hits or pending beams possible (selfApply's barriers).
        gsUniformLanes = triCount > 0u ? 1u : 0u;
    }
    // Static records the lane reads every step, kept in registers: its first node's.
    uint4 myInfo = gStatic[gP.nodeInfoBase + nodeOff + min(lane, max(nodeCount, 1u) - 1u)];
#ifdef EB_VV_SOLO
    if (lane == 0u)
    {
        // Centre and half extents (relative).
        float3 lo = asfloat(gMisc[vvRec(v, VV_ALLOWED_LO)].xyz) - origin;
        float3 hi = asfloat(gMisc[vvRec(v, VV_ALLOWED_HI)].xyz) - origin;
        gsVvBounds[0] = 0.5 * (lo + hi);
        gsVvBounds[1] = 0.5 * (hi - lo);
    }
#endif

    sync();
    uint selfOn = gsUniformLanes;

    for (uint s = 0u; s < gP.steps; s++)
    {
        uint step = stepBase + s;
        uint parity = step & 1u;
        bool hits = gsHit != 0u;
#ifdef EB_VV_SOLO
        for (uint i = lane; i < nodeCount; i += LANES)
            nodeUpdateSolo(nodeOff + i, i == lane ? myInfo : gStatic[gP.nodeInfoBase + nodeOff + i], v, i, step, origin, flagsV, ground, boxOff, boxCount, beamOff, groupOff,
                       groupCount, hits);
#else
#ifdef EB_ABLATE
        uint nodeEnd = (gP.ablate & 2u) != 0u ? 0u : ((gP.ablate & 4u) != 0u ? min(nodeCount, LANES) : nodeCount);
#else
        uint nodeEnd = nodeCount;
#endif
        for (uint i = lane; i < nodeEnd; i += LANES)
            nodeUpdate(nodeOff + i, i == lane ? myInfo : gStatic[gP.nodeInfoBase + nodeOff + i], v, i, step, origin, flagsV, ground, boxOff, boxCount, beamOff, groupOff,
                       groupCount, hits);
#endif

        for (uint g = lane; g < groupCount; g += LANES) gGroups[groupSlot(parity, groupOff, g)].store(NO_GROUP);
        if (candCount > 0u)
        {
            for (uint t = lane; t < triCount; t += LANES)
            {
                int w = s > 0u ? triangleFinish(triOff + t) : asint(gMisc[gP.vehicles + triOff + t].x);
                if (triCount <= TRI_LIST && w <= 0) gsTested[gsTestedCount[parity].add(1u)] = t;
            }
        }
        for (uint b = lane; b < GRID_BUCKETS; b += LANES) gsCount[b].store(0u);
        if (lane == 0u)
        {
            gsPairCount.store(0u);
            gAcc[hitAcc(v)].store(0);
            gAcc[hitAcc(v) + 1u].store(0);
            gsUniformHits = 0u;
            gsUniformOpen = 0u;
        }
        sync();
        if (lane == 0u)
        {
            gsHit = 0u;
            gsTestedCount[parity ^ 1u].store(0u);
            // This step's self-collision: the grid, or (few pairs) every candidate.
            uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
            gsUniformLanes = cell > 0.0 && candCount > 0u && listed * candCount > BRUTE_PAIRS_PER_LANE * LANES ? 1u : 0u;
        }
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
#ifdef EB_ABLATE
        if ((gP.ablate & 16u) == 0u) {
#endif
        // Wings (forces from this step's integrated state), the fuselage drag for
        // the next step, and the commands' rest lengths; then one lane adds the
        // wings' forces in file order while the others start the beams.
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
        {
            uint4 ext = gStatic[gP.vehExtBase + v];
#ifdef EB_ABLATE
            if ((gP.ablate & 8u) != 0u) ext.w = 0u;
#endif
            for (uint k = lane; k < ext.w; k += LANES) wingGather(ext.z + k);
        }
        // The last step's trigger inputs, read by this step's commands: cleared for
        // the next step's beam phase.
        if ((flagsV & FLAG_KEYS) != 0u)
        {
            for (uint k = lane; k < KEYS; k += LANES) gNext[keySlot(v, parity ^ 1u, k)] = 0u;
        }
#ifdef EB_ABLATE
        }
#endif
#endif
#ifdef EB_ABLATE
        uint beamEnd = (gP.ablate & 32u) != 0u ? 0u : beamCount;
#else
        uint beamEnd = beamCount;
#endif
        for (uint j = lane; j < beamEnd; j += LANES) beamUpdate(beamOff + j, j, fast, groupOff, parity);
        // Without the grid this step: list the candidate pairs now (lanes in
        // reverse, so those with one beam fewer start first).
        uint listedB = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
        bool early = candCount > 0u && !(cell > 0.0 && listedB * candCount > BRUTE_PAIRS_PER_LANE * LANES);
        if (early)
        {
            uint lanes = clamp(LANES / max(listedB, 1u), 1u, candCount);
            uint stride = (LANES / lanes) * lanes;
            uint rl = LANES - 1u - lane;
            for (uint p = rl; p < listedB * lanes && rl < stride; p += stride)
            {
                uint t = triCount <= TRI_LIST ? gsTested[p / lanes] : p / lanes;
                trianglePairs(t, triOff + t, p % lanes, lanes, origin, range, cell, candOff, candCount);
            }
        }
        sync();
        uint gridOn = gsUniformLanes;
        // This step's deferred breaks, in spawn order (before selfApply's barrier).
        if (lane == 0u) resolvePending(v, parity);
#ifdef EB_ABLATE
        if (gridOn != 0u && (gP.ablate & 1u) == 0u)
#else
        if (gridOn != 0u)
#endif
        {
            // Offsets: per lane its buckets' sum, a scan over blocks of lanes, then
            // the blocks' totals, then each lane's buckets.
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
        uint pairs = gsPairCount.load();
#ifdef EB_ABLATE
        if ((gP.ablate & 256u) != 0u) {} else
#endif
        if (early && pairs <= PAIR_LIST)
        {
            for (uint i = lane; i < pairs; i += LANES)
            {
                uint e = gsPairs[i];
                tryContact(triangleGeo(triOff + (e >> 16), origin, range), gLists[candOff + (e & 0xFFFFu)], origin,
                           range, beamOff, groupOff, groupCount, parity);
            }
        }
        else if (candCount > 0u && cell > 0.0 && gridOn != 0u)
        {
            uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
            for (uint p = lane; p < listed * GRID_SUBS; p += LANES)
            {
                uint t = triCount <= TRI_LIST ? gsTested[p / GRID_SUBS] : p / GRID_SUBS;
                triangleGrid(triOff + t, p % GRID_SUBS, origin, range, cell, candOff, candCount, beamOff,
                             groupOff, groupCount, parity);
            }
        }
        else if (candCount > 0u && cell > 0.0)
        {
            uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
            uint lanes = clamp(LANES / max(listed, 1u), 1u, candCount);
            uint stride = (LANES / lanes) * lanes;
            for (uint p = lane; p < listed * lanes && lane < stride; p += stride)
            {
                uint t = triCount <= TRI_LIST ? gsTested[p / lanes] : p / lanes;
                triangleGridBrute(triOff + t, p % lanes, lanes, origin, range, cell, candOff, candCount, beamOff,
                                  groupOff, groupCount, parity);
            }
        }
        else if (candCount > 0u)
        {
            // Up to LANES lanes per step: `lanes` per triangle, one per candidate when
            // they fit.
            uint lanes = clamp(LANES / max(triCount, 1u), 1u, candCount);
            uint stride = (LANES / lanes) * lanes;
            for (uint p = lane; p < triCount * lanes && lane < stride; p += stride)
                triangleTest(triOff + p / lanes, p % lanes, lanes, origin, range, candOff, candCount, beamOff,
                             groupOff, groupCount, parity);
        }
        if (selfOn != 0u)
        {
            sync();
            selfApply(v, true, lane, LANES, lane, origin, range, beamOff, groupOff, groupCount, parity);
        }
#ifdef EB_AERO
        // Slide nodes, after this step's beams and self-collision, a lane each.
        if ((flagsV & FLAG_SLIDES) != 0u)
        {
            uint count = min(aero2.y, gP.slideTotal - min(aero2.x, gP.slideTotal));
            for (uint k = lane; k < count; k += LANES) slideNode(aero2.x + k, origin, groupOff, parity);
        }
#endif
        sync();
    }
    if (gP.steps > 0u && candCount > 0u)
    {
        for (uint t = lane; t < triCount; t += LANES) triangleFinish(triOff + t);
    }
#ifdef EB_AERO
    if ((flagsV & FLAG_SLIDES) != 0u)
    {
        uint4 ext = gStatic[gP.vehExtBase + v];
        for (uint k = lane; k < ext.y; k += LANES) slideSettle(ext.x + k);
    }
#endif
    // The last step's group breaks, for a read-back (the next node phase applies
    // them to the forces).
    if (gP.steps > 0u && groupCount > 0u)
    {
        uint last = (stepBase + gP.steps - 1u) & 1u;
        for (uint j = lane; j < beamCount; j += LANES) groupDrops(beamOff + j, beamOff, groupOff, last, true);
    }
    if (lane == 0u) gMisc[v] = uint4(misc.xyz, stepBase + gP.steps);
#ifdef EB_VV_SOLO
    // The last step's positions (each earlier step's were checked at the start of
    // the next step's node phase).
    for (uint i = lane; i < nodeCount; i += LANES)
    {
        float3 p = gNodes[(nodeOff + i) * 3u].xyz;
        if (any(abs(p - gsVvBounds[0]) > gsVvBounds[1])) vvReportOutside(v);
    }
#endif
}

// Bucket slots of the wide passes' grid (a fuller bucket overflows to a list).
static const uint WIDE_SLOTS = 16u;

