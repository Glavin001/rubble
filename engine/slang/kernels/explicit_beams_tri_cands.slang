// Included by explicit_beams.slang after pairFar.

// Triangle gt against its candidates: mode 0 every one (triangleTest); 1 those the grid's
// walk would visit (triangleGridBrute: inside the triangle's box and in a covered cell or a
// covered cell's bucket; every one past GRID_MAX_CELLS cells); 2 a volatile piece
// triangle's (piecesTestTriangle: the same filter with the box of triGeoFill's records, the
// geometry as triGeoLoad gives it, a pair pairFar rules out skipped); 3 the grid's walk
// (triangleGrid's testing walk: the covered cells' buckets' entries inside the box, every
// candidate past GRID_MAX_CELLS cells). Candidates k = sub, sub + subs, ... (cells for
// the walk). One function, so that a step that runs one of the four has one copy of
// tryContact for them (the compilers inline every call): the same tests in the same order
// as each had, the walk's entry loop budgeted as before.
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
                for (uint j = 0u, jStop = EB_END(602, j, i, 1u); j < jStop; j++)
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
            if (mode == 3u)
                EB_PASS(603, eGuard);
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
