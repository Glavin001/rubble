// Included by explicit_beams.slang after pairFar.

// Triangle gt against its candidates k = sub, sub + subs, ... (all of them, candidate
// lists at candOff): mode 0 every one (triangleTest); 1 those the grid's walk would visit
// (triangleGridBrute: inside the triangle's box and in a covered cell or a covered cell's
// bucket; every one past GRID_MAX_CELLS cells); 2 a volatile piece triangle's
// (piecesTestTriangle: the same filter with the box of triGeoFill's records, the geometry
// as triGeoLoad gives it, a pair pairFar rules out skipped). One function, so that a step
// that runs one of the three has one copy of tryContact for them (the compilers inline
// every call): the same tests in the same order as the three functions each had.
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
        t = triangleGeo(gt, origin, r, parity);
        lo = t.lo;
        hi = t.hi;
    }
    bool every = true;
    int3 c0 = int3(0, 0, 0);
    int3 span = int3(1, 1, 1);
    uint yz = 1u;
    uint cells = 0u;
    if (mode != 0u)
    {
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
    for (uint k = sub, kStop = EB_END(17, k, candCount, subs); k < kStop; k += subs)
    {
        uint h = gLists[candOff + k];
        if (!every)
        {
            float3 e = gNodes[h * 3u].xyz;
            float3 p = origin + e;
            if (any(p < lo) || any(p > hi)) continue;
            int3 c = int3(floor(e / cell));
            if (!(all(c >= c0) && all(c < c0 + span)))
            {
                // Outside the covered cells: tested only when its bucket is a covered
                // cell's (a hash collision the grid walk would visit).
                uint b = gridBucket(c);
                bool covered = false;
                for (uint i = 0u, iStop = EB_END(18, i, cells, 1u); i < iStop; i++)
                    covered = covered
                        || gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z)))) == b;
                if (!covered) continue;
            }
        }
        if (mode == 2u && pairFar(t, h, origin, r)) continue;
        tryContact(t, h, origin, r, beamOff, groupOff, groupCount, parity);
    }
}
