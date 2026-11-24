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
    if (lane == 0u)
    {
        gAcc[hitAcc(v)].store(0);
        gAcc[hitAcc(v) + 1u].store(0);
        gsUniformHits = 0u;
        gsUniformOpen = 0u;
    }
    sync();
    if (lane == 0u && s > 0u) gMisc[vvRec(v, VV_FLAGS)].w = 0u;
    if (lane == 0u)
    {
        // Bit 0: the grid this step; bit 1: self-collision hits or pending beams
        // possible (selfApply's barriers).
        uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
        gsUniformLanes = (cell > 0.0 && candCount > 0u && listed * candCount > BRUTE_PAIRS_PER_LANE * LANES ? 1u : 0u)
            | (triCount > 0u ? 2u : 0u);
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
    uint flagsU = gsUniformLanes;
    uint gridOn = flagsU & 1u;
    if (lane == 0u) resolvePending(v, parity);
    if (gridOn != 0u)
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
    if (candCount > 0u && cell > 0.0 && gridOn != 0u)
    {
        uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
        for (uint p = lane; p < listed * GRID_SUBS; p += LANES)
        {
            uint t = triCount <= TRI_LIST ? gsTested[p / GRID_SUBS] : p / GRID_SUBS;
            triangleGrid(triOff + t, p % GRID_SUBS, origin, range, cell, candOff, candCount, beamOff, groupOff,
                         groupCount, parity);
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
        uint lanes = clamp(LANES / max(triCount, 1u), 1u, candCount);
        uint stride = (LANES / lanes) * lanes;
        for (uint p = lane; p < triCount * lanes && lane < stride; p += stride)
            triangleTest(triOff + p / lanes, p % lanes, lanes, origin, range, candOff, candCount, beamOff, groupOff,
                         groupCount, parity);
    }
    if ((flagsU & 2u) != 0u)
    {
        sync();
        selfApply(v, true, lane, LANES, lane, origin, range, beamOff, groupOff, groupCount, parity);
    }
#ifdef EB_AERO
    if (lane == LANES - 1u && (flagsV & FLAG_SLIDES) != 0u) slidePass(aero2.x, aero2.y, origin, groupOff, parity);
#endif
    sync();
}

// --- A small group's members stepped side by side (k <= VV_PART members) ---
// The workgroup's lanes split into k partitions of P lanes (a power of two dividing
// GRID_BUCKETS); partition m steps member m exactly as contactMemberStep would,
// with its own shared counters and its self-collision tested list, grid counts and
// offsets in the per-vehicle global regions the wide passes use. Members do not
// read each other's state within their steps (contact reaches them only through
// the accumulators of the contact pass), so every member's step is the same, bit
// for bit, while the barriers are shared: k members cost about one member's step.
static const uint VV_PART = 4u;
groupshared Atomic<uint> gsTestedM[2u * VV_PART];
groupshared Atomic<uint> gsPairCountM[VV_PART];
groupshared uint gsBlkM[GRID_BLOCKS * VV_PART];
static const uint PAIR_PART = PAIR_LIST / VV_PART;

// Lanes per member for k members: LANES >> ceil(log2 k).
uint vvPartLanes(uint k)
{
    uint p = LANES;
    while (p * k > LANES && p > 1u) p = p >> 1u;
    return p;
}

// triangleGrid with the offsets in gNext at offBase (a member's grid).
void triangleGridAt(uint gt, uint sub, float3 origin, float r, float cell, uint candOff, uint candCount,
                    uint beamOff, uint groupOff, uint groupCount, uint parity, uint offBase)
{
    if (asint(gMisc[gP.vehicles + gt].x) > 0) return;
    TriGeo t = triangleGeo(gt, origin, r);
    float3 f0 = floor((t.lo - origin) / cell);
    float3 fs = floor((t.hi - origin) / cell) - f0 + float3(1.0, 1.0, 1.0);
    if (!(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS)))
    {
        for (uint k = sub; k < candCount; k += GRID_SUBS)
            tryContact(t, gLists[candOff + k], origin, r, beamOff, groupOff, groupCount, parity);
        return;
    }
    int3 c0 = int3(f0);
    int3 span = int3(fs);
    uint yz = uint(span.y * span.z);
    uint cells = uint(span.x) * yz;
    for (uint i = sub; i < cells; i += GRID_SUBS)
    {
        uint b = gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z))));
        bool dup = false;
        for (uint j = 0u; j < i; j++)
            dup = dup || gridBucket(c0 + int3(int(j / yz), int((j / uint(span.z)) % uint(span.y)), int(j % uint(span.z)))) == b;
        if (dup) continue;
        for (uint e = gNext[offBase + b]; e < gNext[offBase + b + 1u]; e++)
        {
            float4 E = gBeams[gP.gridBase + candOff + e];
            float3 p = origin + E.xyz;
            if (any(p < t.lo) || any(p > t.hi)) continue;
            tryContact(t, asuint(E.w), origin, r, beamOff, groupOff, groupCount, parity);
        }
    }
}

// trianglePairs into member m's slice of the pair list.
void trianglePairsM(uint m, uint t, uint gt, uint sub, uint subs, float3 origin, float r, float cell, uint candOff,
                    uint candCount)
{
    if (asint(gMisc[gP.vehicles + gt].x) > 0) return;
    uint4 tn = gStatic[gP.triBase + gt];
    float3 p0 = nf3(origin + gNodes[tn.x * 3u].xyz);
    float3 p1 = nf3(origin + gNodes[tn.y * 3u].xyz);
    float3 p2 = nf3(origin + gNodes[tn.z * 3u].xyz);
    float3 lo = min(min(p0, p1), p2) - float3(r, r, r);
    float3 hi = max(max(p0, p1), p2) + float3(r, r, r);
    bool every = true;
    int3 c0 = int3(0, 0, 0);
    int3 span = int3(0, 0, 0);
    if (cell > 0.0)
    {
        float3 f0 = floor((lo - origin) / cell);
        float3 fs = floor((hi - origin) / cell) - f0 + float3(1.0, 1.0, 1.0);
        every = !(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS));
        if (!every)
        {
            c0 = int3(f0);
            span = int3(fs);
        }
    }
    uint yz = uint(span.y * span.z);
    uint cells = uint(span.x) * yz;
    for (uint k = sub; k < candCount; k += subs)
    {
        uint h = gLists[candOff + k];
        if (h == tn.x || h == tn.y || h == tn.z) continue;
        float3 e = gNodes[h * 3u].xyz;
        if (!every)
        {
            float3 p = origin + e;
            if (any(p < lo) || any(p > hi)) continue;
            int3 c = int3(floor(e / cell));
            if (!(all(c >= c0) && all(c < c0 + span)))
            {
                uint b = gridBucket(c);
                bool covered = false;
                for (uint i = 0u; i < cells; i++)
                    covered = covered
                        || gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z)))) == b;
                if (!covered) continue;
            }
        }
        float3 x = nf3(origin + e);
        if (any(x < lo) || any(x > hi)) continue;
        uint i = gsPairCountM[m].add(1u);
        if (i < PAIR_PART) gsPairs[m * PAIR_PART + i] = (t << 16) | k;
    }
}

void contactMembersStep(uint off, uint k, uint lane, uint s)
{
    uint P = vvPartLanes(k);
    uint m = lane / P;
    uint pl = lane % P;
    bool on = m < k;
    uint v = vvMember(off, on ? m : 0u);
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 r3 = gStatic[v * 4u + 3u];
    uint nodeOff = r0.x;
    uint nodeCount = on ? r0.y : 0u;
    uint beamOff = r0.z;
    uint beamCount = on ? r0.w : 0u;
    uint triOff = r1.x;
    uint triCount = on ? r1.y : 0u;
    uint boxOff = r1.z;
    uint boxCount = r1.w;
    uint candOff = r2.x;
    uint candCount = on ? r2.y : 0u;
    uint groupOff = r2.z;
    uint groupCount = on ? r2.w : 0u;
    uint flagsV = r3.x;
    float ground = asfloat(r3.y);
    float range = asfloat(r3.z);
    float cell = asfloat(r3.w);
    bool fast = (flagsV & FLAG_FAST_INV_SQRT) != 0u;
#ifdef EB_AERO
    uint4 aero0 = gStatic[gP.aeroBase + v * 3u];
    uint4 aero1 = gStatic[gP.aeroBase + v * 3u + 1u];
    uint4 aero2 = gStatic[gP.aeroBase + v * 3u + 2u];
    if (!on)
    {
        aero0 = uint4(0u, 0u, 0u, 0u);
        flagsV = flagsV & ~(FLAG_FUSEDRAG | FLAG_SLIDES | FLAG_KEYS);
    }
#endif
    uint4 misc = gMisc[v];
    float3 origin = asfloat(misc.xyz);
    uint step = misc.w + s;
    uint parity = step & 1u;
    bool pending = gMisc[vvRec(v, VV_FLAGS)].w != 0u;
    // The member's grid: counts (gAcc), offsets (gNext), as the wide passes keep them.
    uint countBase = gP.gridCountBase + v * GRID_BUCKETS;
    uint offBase = gP.gridOffBase + v * (GRID_BUCKETS + 1u);
    uint listBase = gP.testedListBase + triOff;
    if (lane == 0u)
    {
        vvClearCounters();
        vvClearSmall();
        gsUniformHits = 0u;
        gsUniformOpen = 0u;
    }
    if (on && pl == 0u)
    {
        gsTestedM[2u * m + parity].store(0u);
        gsPairCountM[m].store(0u);
        gAcc[hitAcc(v)].store(0);
        gAcc[hitAcc(v) + 1u].store(0);
    }
    sync();
    {
        float3 lo = float3(3.0e38, 3.0e38, 3.0e38);
        float3 hi = -lo;
        bool has = false;
        for (uint i = pl; i < nodeCount; i += P)
        {
            nodeUpdate(nodeOff + i, gStatic[gP.nodeInfoBase + nodeOff + i], v, i, step, origin, flagsV, ground,
                       boxOff, boxCount, beamOff, groupOff, groupCount, true);
            float3 x = nf3(origin + gNodes[(nodeOff + i) * 3u].xyz);
            lo = min(lo, x);
            hi = max(hi, x);
            has = true;
        }
        if (on && pl < 6u) gAcc[vvBoxKey(v, parity ^ 1u) + pl].store(pl < 3u ? VV_INT_MAX : VV_INT_MIN);
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
    for (uint g = pl; g < groupCount; g += P) gGroups[groupSlot(parity, groupOff, g)].store(NO_GROUP);
    if (candCount > 0u)
    {
        for (uint t = pl; t < triCount; t += P)
        {
            int w = s > 0u ? triangleFinish(triOff + t) : asint(gMisc[gP.vehicles + triOff + t].x);
            if (s > 0u && pending)
            {
                w = 0;
                gMisc[gP.vehicles + triOff + t].x = 0u;
            }
            if (w <= 0) gNext[listBase + gsTestedM[2u * m + parity].add(1u)] = t;
        }
    }
    if (on && cell > 0.0)
    {
        for (uint b = pl; b < GRID_BUCKETS; b += P) gAcc[countBase + b].store(0);
    }
    sync();
    if (on && pl == 0u && s > 0u) gMisc[vvRec(v, VV_FLAGS)].w = 0u;
    // This step's self-collision per member: the grid or (few pairs) the early list.
    uint listed = on ? gsTestedM[2u * m + parity].load() : 0u;
    bool gridM = cell > 0.0 && candCount > 0u && listed * candCount > BRUTE_PAIRS_PER_LANE * P;
    if (lane == 0u)
    {
        uint any = 0u;
        for (uint j = 0u; j < k; j++)
        {
            uint vj = vvMember(off, j);
            uint4 q1 = gStatic[vj * 4u + 1u];
            uint4 q2 = gStatic[vj * 4u + 2u];
            float cj = asfloat(gStatic[vj * 4u + 3u].w);
            uint lj = gsTestedM[2u * j + (((gMisc[vj].w + s) & 1u))].load();
            if (cj > 0.0 && q2.y > 0u && lj * q2.y > BRUTE_PAIRS_PER_LANE * P) any = any | 1u;
            // Self-collision hits or pending beams possible (selfApply's barriers).
            if (q1.y > 0u) any = any | 2u;
        }
        gsUniformLanes = any;
    }
    if (gridM)
    {
        for (uint c = pl; c < candCount; c += P)
        {
            float3 p = gNodes[gLists[candOff + c] * 3u].xyz;
            uint b = gridBucket(int3(floor(p / cell)));
            gNext[candOff + c] = b | (uint(gAcc[countBase + b].add(1)) << 10);
        }
    }
#ifdef EB_AERO
    for (uint w = pl; w < aero0.y; w += P) wingForces(aero0.x + w, origin);
    if (on && pl == P - 1u && (flagsV & FLAG_FUSEDRAG) != 0u) fuseDrag(aero1, origin, v);
    if ((flagsV & FLAG_KEYS) != 0u)
    {
        for (uint c = pl; c < aero0.w; c += P) commandUpdateKeys(aero0.z + c, v, parity ^ 1u);
    }
    else
    {
        for (uint c = pl; c < aero0.w; c += P) commandUpdate(aero0.z + c);
    }
    sync();
    {
        uint4 ext = gStatic[gP.vehExtBase + v];
        if (!on) ext.w = 0u;
        for (uint q = pl; q < ext.w; q += P) wingGather(ext.z + q);
    }
    if ((flagsV & FLAG_KEYS) != 0u)
    {
        for (uint q = pl; q < KEYS; q += P) gNext[keySlot(v, parity ^ 1u, q)] = 0u;
    }
#endif
    for (uint j = pl; j < beamCount; j += P) beamUpdate(beamOff + j, j, fast, groupOff, parity);
    bool early = candCount > 0u && !gridM;
    if (early)
    {
        uint lanes = clamp(P / max(listed, 1u), 1u, candCount);
        uint stride = (P / lanes) * lanes;
        uint rl = P - 1u - pl;
        for (uint q = rl; q < listed * lanes && rl < stride; q += stride)
        {
            uint t = gNext[listBase + q / lanes];
            trianglePairsM(m, t, triOff + t, q % lanes, lanes, origin, range, cell, candOff, candCount);
        }
    }
    sync();
    uint flagsU = gsUniformLanes;
    uint gridAny = flagsU & 1u;
    if (on && pl == 0u) resolvePending(v, parity);
    if (gridAny != 0u)
    {
        uint per = GRID_BUCKETS / P;
        uint block = P / GRID_BLOCKS;
        uint sum = 0u;
        if (gridM)
        {
            for (uint q = 0u; q < per; q++) sum += uint(gAcc[countBase + pl * per + q].load());
        }
        gsPart[lane] = sum;
        sync();
        if (gridM && pl < GRID_BLOCKS)
        {
            uint run = 0u;
            for (uint q = 0u; q < block; q++)
            {
                uint x = gsPart[m * P + pl * block + q];
                gsPart[m * P + pl * block + q] = run;
                run += x;
            }
            gsBlkM[m * GRID_BLOCKS + pl] = run;
        }
        sync();
        if (gridM && pl == 0u)
        {
            uint run = 0u;
            for (uint q = 0u; q < GRID_BLOCKS; q++)
            {
                uint x = gsBlkM[m * GRID_BLOCKS + q];
                gsBlkM[m * GRID_BLOCKS + q] = run;
                run += x;
            }
            gNext[offBase + GRID_BUCKETS] = run;
        }
        sync();
        if (gridM)
        {
            uint run = gsPart[lane] + gsBlkM[m * GRID_BLOCKS + pl / block];
            for (uint q = 0u; q < per; q++)
            {
                gNext[offBase + pl * per + q] = run;
                run += uint(gAcc[countBase + pl * per + q].load());
            }
        }
        sync();
        if (gridM)
        {
            for (uint c = pl; c < candCount; c += P)
            {
                uint br = gNext[candOff + c];
                uint h = gLists[candOff + c];
                gBeams[gP.gridBase + candOff + gNext[offBase + (br & (GRID_BUCKETS - 1u))] + (br >> 10)] =
                    float4(gNodes[h * 3u].xyz, asfloat(h));
            }
        }
        sync();
    }
    uint pairs = on ? gsPairCountM[m].load() : 0u;
    if (early && pairs <= PAIR_PART)
    {
        for (uint i = pl; i < pairs; i += P)
        {
            uint e = gsPairs[m * PAIR_PART + i];
            tryContact(triangleGeo(triOff + (e >> 16), origin, range), gLists[candOff + (e & 0xFFFFu)], origin, range,
                       beamOff, groupOff, groupCount, parity);
        }
    }
    else if (candCount > 0u && cell > 0.0 && gridM)
    {
        for (uint q = pl; q < listed * GRID_SUBS; q += P)
        {
            uint t = gNext[listBase + q / GRID_SUBS];
            triangleGridAt(triOff + t, q % GRID_SUBS, origin, range, cell, candOff, candCount, beamOff, groupOff,
                           groupCount, parity, offBase);
        }
    }
    else if (candCount > 0u && cell > 0.0)
    {
        uint lanes = clamp(P / max(listed, 1u), 1u, candCount);
        uint stride = (P / lanes) * lanes;
        for (uint q = pl; q < listed * lanes && pl < stride; q += stride)
        {
            uint t = gNext[listBase + q / lanes];
            triangleGridBrute(triOff + t, q % lanes, lanes, origin, range, cell, candOff, candCount, beamOff,
                              groupOff, groupCount, parity);
        }
    }
    else if (candCount > 0u)
    {
        uint lanes = clamp(P / max(listed, 1u), 1u, candCount);
        uint stride = (P / lanes) * lanes;
        for (uint q = pl; q < listed * lanes && pl < stride; q += stride)
        {
            uint t = gNext[listBase + q / lanes];
            triangleTest(triOff + t, q % lanes, lanes, origin, range, candOff, candCount, beamOff, groupOff,
                         groupCount, parity);
        }
    }
    if ((flagsU & 2u) != 0u)
    {
        sync();
        selfApply(v, on, pl, P, lane, origin, range, beamOff, groupOff, groupCount, parity);
    }
#ifdef EB_AERO
    if (on && pl == P - 1u && (flagsV & FLAG_SLIDES) != 0u) slidePass(aero2.x, aero2.y, origin, groupOff, parity);
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

