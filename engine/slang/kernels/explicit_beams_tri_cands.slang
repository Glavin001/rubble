// Included by explicit_beams.slang after pairFar.

// A listed pair's box test before its contact test (pairReuseTest's; false: not tested).
bool pairReuseNear(uint gt, uint h, float3 origin, float cell)
{
    if (asint(gMisc[gP.vehicles + gt].x) > 0) return false;
    // The box from triGeoFill's records before its liveness (a dead triangle's empty box
    // fails tryContact's test, as in the walk).
    uint at = gP.triGeo + gt * TRI_GEO_RECS;
    float4 r4 = gBeams[at + 4u];
    float4 r5 = gBeams[at + 5u];
    float3 lo = r4.xyz;
    float3 hi = float3(r4.w, r5.xy);
    float3 f0 = floor((lo - origin) / cell);
    float3 fs = floor((hi - origin) / cell) - f0 + float3(1.0, 1.0, 1.0);
    bool every = !(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS));
    if (!every)
    {
        float3 e = gNodes[h * 3u].xyz;
        float3 p = origin + e;
        if (any(p < lo) || any(p > hi)) return false;
        int3 c0 = int3(f0);
        int3 span = int3(fs);
        int3 c = int3(floor(e / cell));
        if (!(all(c >= c0) && all(c < c0 + span)))
        {
            // Outside the covered cells: the walk meets it only in a covered cell's bucket.
            uint yz = uint(span.y * span.z);
            uint cells = uint(span.x) * yz;
            uint b = gridBucket(c);
            bool covered = false;
            for (uint i = 0u, iStop = EB_END(34, i, cells, 1u); i < iStop; i++)
                covered = covered
                    || gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z)))) == b;
            if (!covered) return false;
        }
    }
#ifdef EB_ABLATE
    // Measurement only (time): 2^26 the listed pairs' contact tests (their box tests kept).
    if ((gP.ablate & 67108864u) != 0u) return false;
#endif
    return true;
}
void pairReuseTest(uint gt, uint h, float3 origin, float r, float cell, uint beamOff, uint groupOff,
                   uint groupCount, uint parity)
{
    if (pairReuseNear(gt, h, origin, cell)) tryContact(triGeoLoad(gt), h, origin, r, beamOff, groupOff, groupCount, parity);
}

// Triangle gt against its candidates: mode 0 every one (triangleTest); 1 those the grid's
// walk would visit (triangleGridBrute: inside the triangle's box and in a covered cell or a
// covered cell's bucket; every one past GRID_MAX_CELLS cells); 2 a volatile piece
// triangle's (piecesTestTriangle: the same filter with the box of triGeoFill's records, the
// geometry as triGeoLoad gives it, a pair pairFar rules out skipped); 3 the grid's walk
// (triangleGrid's testing walk: the covered cells' buckets' entries inside the box, every
// candidate past GRID_MAX_CELLS cells). Candidates k = sub, sub + subs, ... (cells for
// the walk). One function, so that a step that runs one of the four has one copy of
// tryContact for them (the compilers inline every call): the same tests in the same order
// as each had.
void triangleCandidates(uint gt, uint sub, uint subs, float3 origin, float r, float cell, uint candOff, uint candCount,
                        uint beamOff, uint groupOff, uint groupCount, uint parity, uint mode)
{
    if (asint(gMisc[gP.vehicles + gt].x) > 0) return;
    TriGeo t;
    float3 lo;
    float3 hi;
    if (mode == 2u)
    {
        uint at = gP.triGeo + gt * TRI_GEO_RECS;
        float4 r4 = gBeams[at + 4u];
        float4 r5 = gBeams[at + 5u];
        lo = r4.xyz;
        hi = float3(r4.w, r5.xy);
        t = triGeoLoad(gt);
    }
    else
    {
        t = mode == 3u && gP.triGeo != 0u ? triGeoLoad(gt) : triangleGeo(gt, origin, r, parity);
        lo = t.lo;
        hi = t.hi;
    }
#ifdef EB_ABLATE
    // Measurement only: 4194304 the walk stops after the triangle's geometry.
    if (mode == 3u && (gP.ablate & 4194304u) != 0u) return;
#endif
    bool every = true;
    int3 c0 = int3(0, 0, 0);
    int3 span = int3(1, 1, 1);
    uint yz = 1u;
    uint cells = 0u;
    if (mode != 0u)
    {
        // (The box's cells counted in float: a stretched or non-finite triangle takes
        // the every-candidate path, never an unbounded cell loop.)
        float3 f0 = floor((lo - origin) / cell);
        float3 fs = floor((hi - origin) / cell) - f0 + float3(1.0, 1.0, 1.0);
        every = !(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS));
        if (!every)
        {
            c0 = int3(f0);
            span = int3(fs);
            yz = uint(span.y * span.z);
            cells = uint(span.x) * yz;
        }
    }
    bool walk = mode == 3u && !every;
    // The loop budget as each of the four drew it: the outer loop budgeted, and the walk's
    // every bucket entry a pass more (ebSpend below; the entry loop capped, unbudgeted).
    for (uint i = sub, iStop = EB_END(17, i, walk ? cells : candCount, subs); i < iStop; i += subs)
    {
        uint e0 = 0u;
        uint e1 = 1u;
        if (walk)
        {
            uint b = gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z))));
            // Skip a bucket an earlier cell of this triangle already covers.
            bool dup = false;
#ifdef EB_ABLATE
            // Measurement only: 8388608 no duplicate-bucket check.
            if ((gP.ablate & 8388608u) == 0u)
#endif
            {
                // Cell j's offset stepped from j = 0 (z fastest), as triangleGridWalk.
                int3 cj = int3(0, 0, 0);
                for (uint j = 0u, jStop = EB_END(649, j, i, 1u); j < jStop; j++)
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
            }
            if (dup) continue;
            e0 = gsOff[b];
            e1 = gsOff[b + 1u];
        }
        for (uint e = e0, eStop = EB_END_UNIFORM(644, e, e1, 1u, EB_LOOP_CAP); e < eStop; e++)
        {
            if (mode == 3u && !ebSpend())
            {
                ebLoopCapHit(652);
                break;
            }
            uint h;
            if (walk)
            {
                float4 E = gBeams[gP.gridBase + candOff + e];
                float3 p = origin + E.xyz;
                if (any(p < t.lo) || any(p > t.hi)) continue;
                h = asuint(E.w);
            }
            else
            {
                h = gLists[candOff + i];
                if (!every)
                {
                    float3 ep = gNodes[h * 3u].xyz;
                    float3 p = origin + ep;
                    if (any(p < lo) || any(p > hi)) continue;
                    int3 c = int3(floor(ep / cell));
                    if (!(all(c >= c0) && all(c < c0 + span)))
                    {
                        // Outside the covered cells: tested only when its bucket is a covered
                        // cell's (a hash collision the grid walk would visit).
                        uint b = gridBucket(c);
                        bool covered = false;
                        for (uint ci = 0u, ciStop = EB_END(18, ci, cells, 1u); ci < ciStop; ci++)
                            covered = covered
                                || gridBucket(c0 + int3(int(ci / yz), int((ci / uint(span.z)) % uint(span.y)), int(ci % uint(span.z)))) == b;
                        if (!covered) continue;
                    }
                }
            }
            if (mode == 2u && pairFar(t, h, origin, r)) continue;
            tryContact(t, h, origin, r, beamOff, groupOff, groupCount, parity);
        }
    }
}

// triangleGridWalk's listing walk alone (its pairs into the list at listAt, for the test
// pass after it): no test here, so a frame whose testing walk runs in its one candidate
// loop (triangleCandidates) keeps no copy of tryContact for the listing.
void triangleGridList(uint gt, uint sub, float3 origin, float r, float cell, uint candOff, uint candCount, uint parity,
                      uint listAt)
{
    if (asint(gMisc[gP.vehicles + gt].x) > 0) return;
    // The frame's walk: the step's geometry from the beam phase (triGeoFill).
    TriGeo t = triGeoBox(gt);
    uint subs = LIST_SUBS;
#ifdef EB_ABLATE
    // Measurement only: 4194304 the walk stops after the triangle's geometry.
    if ((gP.ablate & 4194304u) != 0u) return;
#endif
    // The box's cells, counted in float: a stretched or non-finite triangle (a
    // flung node) must take the every-candidate path, not overflow an integer
    // product into an unbounded cell loop (a GPU hang).
    float3 f0 = floor((t.lo - origin) / cell);
    float3 fs = floor((t.hi - origin) / cell) - f0 + float3(1.0, 1.0, 1.0);
    bool every = !(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS));
    int3 c0 = int3(0, 0, 0);
    int3 span = int3(1, 1, 1);
    uint yz = 1u;
    uint cells = 0u;
    if (!every)
    {
        c0 = int3(f0);
        span = int3(fs);
        yz = uint(span.y * span.z);
        cells = uint(span.x) * yz;
    }
    uint outer = every ? candCount : cells;
    for (uint i = sub, iStop = EB_END(608, i, outer, subs); i < iStop; i += subs)
    {
        uint e0 = 0u;
        uint e1 = 1u;
        if (!every)
        {
            uint b = gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z))));
            // Skip a bucket an earlier cell of this triangle already covers.
            bool dup = false;
#ifdef EB_ABLATE
            // Measurement only: 8388608 no duplicate-bucket check.
            if ((gP.ablate & 8388608u) == 0u)
#endif
            {
                // Cell j's offset (j / yz, j / span.z % span.y, j % span.z) stepped
                // from j = 0 (z fastest) instead of divided out for every j.
                int3 cj = int3(0, 0, 0);
                for (uint j = 0u, jStop = EB_END(609, j, i, 1u); j < jStop; j++)
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
            }
            if (dup) continue;
            e0 = gsOff[b];
            e1 = gsOff[b + 1u];
        }
        for (uint e = e0, eGuard = 0u; e < e1; e++)
        {
            EB_PASS(610, eGuard);
            uint h;
            if (every)
                h = gLists[candOff + i];
            else
            {
                float4 E = gBeams[gP.gridBase + candOff + e];
                float3 p = origin + E.xyz;
                if (any(p < t.lo) || any(p > t.hi)) continue;
                h = asuint(E.w);
            }
            // Listed for the test pass: a bucket's entry as the testing walk passes it on
            // (its box test above); every candidate (too many cells) only when inside the box
            // the test checks first, with its expression.
            if (every)
            {
                float3 q = nf3(origin + gNodes[h * 3u].xyz);
                if (any(q < t.lo) || any(q > t.hi)) continue;
            }
            uint n = triPairAdd();
            if (n < TRI_PAIR_CAP)
            {
                gNext[listAt + 2u * n] = gt;
                gNext[listAt + 2u * n + 1u] = h;
            }
        }
    }
}
