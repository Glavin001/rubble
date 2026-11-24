// --- cpl_query: the World broadphase's candidates (mode 1) ---

uint candAppend(uint v, uint c)
{
    uint at = gWork[wCandCount(v)].add(1u);
    if (at < gP.candCap) gWork[wCands(v) + at].store(c);
    else gWork[1].or(ST_CANDS);
    return at;
}

// A candidate is listed only when it can reach one of the vehicle's group boxes
// (pieceAccept, as cpl_plan keeps it): the same proxies, and a vehicle whose parts lie far
// apart lists the colliders near them, not every one in the box between them.
void candTry(uint v, uint c, float3 lo, float3 hi)
{
    if (c < gP.colliders && cplAccept(c, lo, hi) && pieceAccept(v, c, sweptReach(v))) candAppend(v, c);
}

int3 cellOf(float3 p, float cs) { return int3(floor(p / cs)); }
uint gridHashW(int3 c) { return ((uint)c.x * 73856093u ^ (uint)c.y * 19349663u ^ (uint)c.z * 83492791u) & (gP.hashSize - 1u); }
uint hierHashW(int3 c, uint level) { return (gridHashW(c) ^ (level * 2654435769u)) & (gP.hashSize - 1u); }
uint hierLevelW(float radius)
{
    float diameter = 2.0 * radius + 0.0004;
    return uint(max(0.0, ceil(log2(max(diameter / gP.cell, 1.0)))));
}
float3 aabbCentre(uint c) { return 0.5 * (gAabbs[2u * (gP.aabbBase + c)].xyz + gAabbs[2u * (gP.aabbBase + c) + 1u].xyz); }
// The cell an entry was inserted at, by the inserter's own arithmetic (world_step.slang
// hierCell, uniformCell).
int3 entryCell(uint j, float cs, bool hier)
{
    float3 mn = gAabbs[2u * (gP.aabbBase + j)].xyz;
    float3 mx = gAabbs[2u * (gP.aabbBase + j) + 1u].xyz;
    if (hier) return int3(floor((mn + mx) * (0.5 / cs)));
    return int3(floor((0.5 * (mn + mx)) / cs));
}
uint restHeadsW() { return gP.restBase + 32u; }
uint restLinksW() { return gP.restBase + 32u + 2u * gP.hashSize; }

// One resting chain (head h): every entry whose true cell (and level) is `cell` tested.
void restChain(uint v, uint h, int3 cell, uint level, float cs, bool hier, float3 lo, float3 hi)
{
    uint j = gIds[restHeadsW() + h];
    uint guard = 0u;
    uint jGuard = 0u;
    while (j != 0xFFFFFFFFu && guard < gP.colliders + 1u)
    {
        EB_PASS(562, jGuard);
        guard++;
        uint next = gIds[restLinksW() + j];
        bool mine = hier ? (hierLevelW(gAabbs[2u * (gP.aabbBase + j)].w) == level && all(entryCell(j, cs, true) == cell))
                         : all(entryCell(j, cs, false) == cell);
        if (mine) candTry(v, j, lo, hi);
        j = next;
    }
}

// One resting chain (head h): every entry (of `level`, hierarchical) whose cell lies in
// a..b tested.
void restRange(uint v, uint h, int3 a, int3 b, uint level, float cs, bool hier, float3 lo, float3 hi)
{
    uint j = gIds[restHeadsW() + h];
    uint guard = 0u;
    uint jGuard = 0u;
    while (j != 0xFFFFFFFFu && guard < gP.colliders + 1u)
    {
        EB_PASS(563, jGuard);
        guard++;
        uint next = gIds[restLinksW() + j];
        int3 c = entryCell(j, cs, hier);
        bool mine = all(c >= a) && all(c <= b) && (!hier || hierLevelW(gAabbs[2u * (gP.aabbBase + j)].w) == level);
        if (mine) candTry(v, j, lo, hi);
        j = next;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void cpl_query(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    uint lane = lid.x;
    if (gP.mode == 0u) return ebLoopCapFlush();
    if (lane == 0u) gsUniformCpl = vehicleAt(v);
    GroupMemoryBarrierWithGroupSync();
    uint4 vt = gsUniformCpl;
    if ((vt.w & PROXY_OUT) != 0u) return ebLoopCapFlush();
    float3 lo;
    float3 hi;
    sweptBox(v, lo, hi, false);
    uint hdr = gP.restBase;
    if (gP.levels > 0u)
    {
        // Hierarchical: a level-L entry's centre lies in its cell and its sphere within a
        // cell of it (its bound radius is at most half a cell), and 5 cm for a sleeper's
        // bounds from the start of its last moving step.
        uint mask = gIds[hdr];
        for (uint level = 0u; level < 32u; level++)
        {
            if (((mask >> level) & 1u) == 0u) continue;
            float cs = gP.cell * exp2(float(level));
            int3 a = cellOf(lo - (cs + 0.05), cs);
            int3 b = cellOf(hi + (cs + 0.05), cs);
            int3 span = b - a + int3(1, 1, 1);
            uint cells = uint(span.x) * uint(span.y) * uint(span.z);
            if (cells > gP.hashSize)
            {
                // More cells than heads (a box far larger than the level's cells): every
                // head once, its entries of this level whose cell is in the range.
                for (uint h = lane, hStop = EB_END(564, h, gP.hashSize, 256u); h < hStop; h += 256u) restRange(v, h, a, b, level, cs, true, lo, hi);
                continue;
            }
            for (uint k = lane, kStop = EB_END(565, k, cells, 256u); k < kStop; k += 256u)
            {
                int3 cell = a + int3(int(k % uint(span.x)), int((k / uint(span.x)) % uint(span.y)), int(k / (uint(span.x) * uint(span.y))));
                restChain(v, hierHashW(cell, level), cell, level, cs, true, lo, hi);
            }
        }
        // Oversized statics (header 11..20).
        uint large = min(gIds[hdr + 10u], 10u);
        for (uint k = lane, kStop = EB_END(566, k, large, 256u); k < kStop; k += 256u) candTry(v, gIds[hdr + 11u + k], lo, hi);
    }
    else
    {
        // Uniform: an entry's centre in its cell, its sphere within two cells of it
        // (statics of bound radius past half a cell, dynamics past a cell, are listed).
        float cs = max(gP.cell, 0.5);
        float m = 2.0 * cs + asfloat(gIds[hdr + 29u]) + 0.05;
        int3 a = cellOf(lo - m, cs);
        int3 b = cellOf(hi + m, cs);
        int3 span = b - a + int3(1, 1, 1);
        uint cells = uint(span.x) * uint(span.y) * uint(span.z);
        if (cells > gP.hashSize)
        {
            for (uint h = lane, hStop = EB_END(567, h, gP.hashSize, 256u); h < hStop; h += 256u) restRange(v, h, a, b, 0u, cs, false, lo, hi);
        }
        else
        {
            for (uint k = lane, kStop = EB_END(568, k, cells, 256u); k < kStop; k += 256u)
            {
                int3 cell = a + int3(int(k % uint(span.x)), int((k / uint(span.x)) % uint(span.y)), int(k / (uint(span.x) * uint(span.y))));
                restChain(v, gridHashW(cell), cell, 0u, cs, false, lo, hi);
            }
        }
        uint large = min(gIds[hdr + 10u], 16u);
        for (uint k = lane, kStop = EB_END(569, k, large, 256u); k < kStop; k += 256u) candTry(v, gIds[hdr + 11u + k], lo, hi);
        uint statics = min(gIds[hdr + 28u], 1024u);
        for (uint k = lane, kStop = EB_END(570, k, statics, 256u); k < kStop; k += 256u) candTry(v, gScratch[gP.largeStaticBase + k].load(), lo, hi);
    }
    // The step's moving colliders (awake at its start), and every collider of a body
    // whose pose or sleep state changed in it (pose_pack).
    uint nm = gScratch[6].load();
    for (uint i = lane, iStop = EB_END(571, i, nm, 256u); i < iStop; i += 256u) candTry(v, gIds[restLinksW() + gP.colliders + i], lo, hi);
    uint nb = gMovedHeader[0];
    for (uint i = lane, iStop = EB_END(572, i, nb, 256u); i < iStop; i += 256u)
    {
        uint body = gMoved[2u * i].w & 0x0FFFFFFFu;
        uint l = 0u;
        uint h = gP.colliders;
        uint lGuard = 0u;
        while (l < h)
        {
            EB_PASS(573, lGuard);
            uint mid = (l + h) >> 1;
            if (gColliders[4u * mid].x < body) l = mid + 1u; else h = mid;
        }
        for (uint c = l, cStop = EB_END(574, c, gP.colliders, 1u); c < cStop && gColliders[4u * c].x == body; c++) candTry(v, c, lo, hi);
    }
    ebLoopCapFlush();
}

// A hull proxy's records in the planes room: its planes, and with the surface contact
// (hullGeom) its vertices (scaled) and edges (a | b << 8, four to a record) after them
// (explicit_beams_surface.slang fqHull*).
uint hullRecords(uint base)
{
    uint head = gHulls[4u * base];
    uint nv = head & 255u;
    uint nf = (head >> 8) & 255u;
    uint ne = (head >> 16) & 255u;
    return nf + (gP.hullGeom != 0u ? nv + (ne + 3u) / 4u : 0u);
}

// The vertices and edges of hull `base` at planes-room record `at` (hullGeom): its
// counts nv | ne << 8 for the shape record, 0 without.
uint hullGeometry(uint v, uint base, float scale, uint at)
{
    if (gP.hullGeom == 0u) return 0u;
    uint head = gHulls[4u * base];
    uint nv = head & 255u;
    uint ne = (head >> 16) & 255u;
    for (uint i = 0u, iStop = EB_END(575, i, nv, 1u); i < iStop; i++)
    {
        uint w = 4u * (base + 4u + i);
        float3 p = asfloat(uint3(gHulls[w], gHulls[w + 1u], gHulls[w + 2u])) * scale;
        gMisc[prec(v, at + i)] = asuint(float4(p, 0.0));
    }
    uint edgeWord = gHulls[4u * base + 2u];
    for (uint r = 0u, rStop = EB_END(576, r, (ne + 3u) / 4u, 1u); r < rStop; r++)
    {
        uint4 rec = uint4(0u, 0u, 0u, 0u);
        for (uint m = 0u; m < 4u; m++)
        {
            uint e = 4u * r + m;
            uint x = e < ne ? (gHulls[4u * base + edgeWord + e] & 0xFFFFu) : 0u;
            if (m == 0u) rec.x = x;
            else if (m == 1u) rec.y = x;
            else if (m == 2u) rec.z = x;
            else rec.w = x;
        }
        gMisc[prec(v, at + nv + r)] = rec;
    }
    return nv | (ne << 8);
}

// cpl_plan's records a lane a shape (the n shapes in gsSorted, their bodies and faces
// in gsBody and gsFaces, every body and plane within its room): the same records the
// serial walk writes. Shape k's body slot is its body's number in order of first
// appearance, its planes follow the earlier shapes' faces, its body's range is the
// body's first and last shape.
void cplPlanShapes(uint v, uint lane, uint n)
{
    uint nb = 0u;
    uint planes = 0u;
    for (uint k = lane, kStop = EB_END(577, k, n, 256u); k < kStop; k += 256u)
    {
        uint body = gsBody[k];
        // Its body's first shape, the bodies first seen before it, the planes before it.
        uint f = k;
        uint slot = 0u;
        uint po = 0u;
        uint last = k;
        for (uint j = 0u, jStop = EB_END(578, j, n, 1u); j < jStop; j++)
        {
            uint bj = gsBody[j];
            if (j < k) po += gsFaces[j] & 0xFFFFu;
            if (bj == body)
            {
                f = min(f, j);
                last = j;
            }
        }
        for (uint j = 0u, jStop = EB_END(579, j, f, 1u); j < jStop; j++) slot += gsFaces[j] >> 31;
        uint c = gsSorted[k];
        uint b2, kind;
        float4 sph = colliderSphere(c, b2, kind);
        uint j = slot;
        if (f == k)
        {
            uint b = brec(v, j);
            float4 p0 = gBodies[7u * body];
            float4 p2 = gBodies[7u * body + 2u];
            float4 p3 = gBodies[7u * body + 3u];
            float4 p4 = gBodies[7u * body + 4u];
            gMisc[b] = asuint(float4(p0.xyz - asfloat(gMisc[v].xyz), p0.w));
            gMisc[b + 1u] = asuint(gBodies[7u * body + 1u]);
            gMisc[b + 2u] = asuint(float4(p3.xyz, 1.0));
            gMisc[b + 3u] = uint4(asuint(p4.xyz), body);
            // (1/I, w): w the body's shape range, first + 1 | (last + 1) << 16.
            uint4 r4 = asuint(float4(p0.w > 0.0 ? p2.xyz : float3(0.0), 0.0));
            gMisc[b + 4u] = uint4(r4.xyz, (f + 1u) | ((last + 1u) << 16));
            gWork[wShare() + body].add(1u);
        }
        float4 lp = asfloat(gColliders[4u * c + 1u]);
        uint4 lq = gColliders[4u * c + 2u];
        float4 pr = asfloat(gColliders[4u * c + 3u]);
        uint at = srec(v, k);
        uint span = 0u;
        uint code = 0u;
        uint hullCounts = 0u;
        float3 prm = float3(0.0);
        if (kind == 2u) { code = 0u; prm = pr.xyz; }
        else if (kind == 0u) { code = 1u; prm = float3(pr.x, 0.0, 0.0); }
        else if (kind == 1u) { code = 2u; prm = float3(pr.x, pr.y, 0.0); }
        else
        {
            code = 3u;
            uint base = asuint(pr.x) & HULL_MAX_BASE;
            float scale = pr.y;
            uint head = gHulls[4u * base];
            uint nvh = head & 255u;
            uint nf = (head >> 8) & 255u;
            for (uint fi = 0u, fiStop = EB_END(580, fi, nf, 1u); fi < fiStop; fi++)
            {
                uint w = 4u * (base + 4u + nvh + fi);
                float3 nn = asfloat(uint3(gHulls[w], gHulls[w + 1u], gHulls[w + 2u]));
                float d = asfloat(gHulls[w + 3u]) * scale;
                gMisc[prec(v, po + fi)] = asuint(float4(nn, d));
            }
            hullCounts = hullGeometry(v, base, scale, po + nf);
            span = po | (nf << 16);
        }
        gMisc[at] = uint4(code, j, c, span);
        gMisc[at + 1u] = asuint(float4(lp.xyz, prm.x));
        gMisc[at + 2u] = lq;
        gMisc[at + 3u] = uint4(asuint(prm.y), asuint(prm.z), asuint(sph.w + 0.01), hullCounts);
        gMisc[at + 4u] = uint4(0u);
        gMisc[at + 5u] = uint4(0u);
        gMisc[at + 6u] = asuint(float4(sph.xyz - asfloat(gMisc[v].xyz), 0.0));
    }
    if (lane == 0u)
    {
        gMisc[vrec(v)] = uint4(gsBodies.load(), n, gsPlanes.load(), 0u);
        listAppend(v, n);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void cpl_plan(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    uint lane = lid.x;
    if (lane == 0u)
    {
        gsCount.store(0u);
        gsWake.store(0u);
        gsUniformCpl = vehicleAt(v);
    }
    GroupMemoryBarrierWithGroupSync();
    if ((gsUniformCpl.w & PROXY_OUT) != 0u)
    {
        // Out of the broadphase: no proxies.
        if (lane == 0u) gMisc[vrec(v)] = uint4(0u, 0u, 0u, 0u);
        return ebLoopCapFlush();
    }
    float3 lo;
    float3 hi;
    sweptBox(v, lo, hi, false);
    if (gP.mode == 0u)
    {
        float pr = sweptReach(v);
        for (uint c = lane, cStop = EB_END(581, c, gP.colliders, 256u); c < cStop; c += 256u)
        {
            if (!cplAccept(c, lo, hi)) continue;
            if (!pieceAccept(v, c, pr)) continue;
            if (gP.sleepBase != 0u && bodyMoving(gColliders[4u * c].x)) gsWake.store(1u);
            uint at = gsCount.add(1u);
            if (at < LIST_CAP) gsList[at] = c;
        }
    }
    else
    {
        // cpl_query's candidates (each passed the exact test), one copy of each: the
        // resting entry and the moving or moved list can both hold a collider.
        uint nc = min(gWork[wCandCount(v)].load(), gP.candCap);
        for (uint i = lane, iStop = EB_END(582, i, nc, 256u); i < iStop; i += 256u)
        {
            uint c = gWork[wCands(v) + i].load();
            bool dup = false;
            for (uint j = 0u, jStop = EB_END(583, j, i, 1u); j < jStop; j++)
            {
                if (gWork[wCands(v) + j].load() == c) { dup = true; break; }
            }
            if (dup) continue;
            if (!pieceAccept(v, c, sweptReach(v))) continue;
            if (gP.sleepBase != 0u && bodyMoving(gColliders[4u * c].x)) gsWake.store(1u);
            uint at = gsCount.add(1u);
            if (at < LIST_CAP) gsList[at] = c;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u && gP.sleepBase != 0u && gsWake.load() != 0u) gMisc[gP.sleepBase + v] = uint4(0u, 0u, 0u, 0u);
    // The listed colliders in collider order: each one's rank among them (the indices
    // are distinct), the list's first n (all of it within its room; a list past it is
    // an error the status reports, as before).
    uint total = gsCount.load();
    uint n = min(total, min(gP.S, LIST_CAP));
    if (lane == 0u && (total > gP.S || total > LIST_CAP)) gWork[1].or(ST_SHAPES);
    for (uint i = lane, iStop = EB_END(584, i, n, 256u); i < iStop; i += 256u)
    {
        uint x = gsList[i];
        uint rank = 0u;
        for (uint j = 0u, jStop = EB_END(585, j, n, 1u); j < jStop; j++) rank += gsList[j] < x ? 1u : 0u;
        gsSorted[rank] = x;
    }
    if (lane == 0u)
    {
        gsBodies.store(0u);
        gsPlanes.store(0u);
    }
    GroupMemoryBarrierWithGroupSync();
    // Each shape's engine body and hull faces; its body's first appearance counted.
    for (uint k = lane, kStop = EB_END(586, k, n, 256u); k < kStop; k += 256u)
    {
        uint body, kind;
        colliderSphere(gsSorted[k], body, kind);
        uint nf = 0u;
        if (kind != 0u && kind != 1u && kind != 2u)
        {
            uint4 pr = gColliders[4u * gsSorted[k] + 3u];
            uint base = pr.x & HULL_MAX_BASE;
            nf = hullRecords(base);
        }
        gsBody[k] = body;
        gsFaces[k] = nf;
        gsPlanes.add(nf);
    }
    GroupMemoryBarrierWithGroupSync();
    // (gsFaces bit 31: the shape is its body's first.)
    for (uint k = lane, kStop = EB_END(587, k, n, 256u); k < kStop; k += 256u)
    {
        bool first = true;
        for (uint j = 0u, jStop = EB_END(588, j, k, 1u); j < jStop; j++)
        {
            if (gsBody[j] == gsBody[k]) { first = false; break; }
        }
        if (first)
        {
            gsBodies.add(1u);
            gsFaces[k] = gsFaces[k] | 0x80000000u;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // Within every room (no shape skipped, as in the serial walk when nothing
    // overflows): every record written a lane a shape, as that walk writes it.
    // Otherwise lane 0 walks the shapes one by one as before (the status then fails
    // the step).
    if (gsBodies.load() <= gP.B && gsPlanes.load() <= gP.planes && gP.serialPlan == 0u)
    {
        cplPlanShapes(v, lane, n);
        return ebLoopCapFlush();
    }
    if (lane != 0u) return ebLoopCapFlush();
    // Collider order: an insertion sort (the lists are short).
    for (uint a = 1u, aStop = EB_END(589, a, n, 1u); a < aStop; a++)
    {
        uint x = gsSorted[a];
        uint b = a;
        uint bGuard = 0u;
        while (b > 0u && gsSorted[b - 1u] > x)
        {
            EB_PASS(590, bGuard);
            gsSorted[b] = gsSorted[b - 1u];
            b--;
        }
        gsSorted[b] = x;
    }
    uint nb = 0u;
    uint planes = 0u;
    uint ns = 0u;
    for (uint k = 0u, kStop = EB_END(591, k, n, 1u); k < kStop; k++)
    {
        uint c = gsSorted[k];
        uint body, kind;
        float4 sph = colliderSphere(c, body, kind);
        // Its body's slot (first appearance).
        uint j = NONE;
        for (uint q = 0u, qStop = EB_END(592, q, nb, 1u); q < qStop; q++)
        {
            if (gMisc[brec(v, q) + 3u].w == body) { j = q; break; }
        }
        if (j == NONE)
        {
            if (nb >= gP.B) { gWork[1].or(ST_BODIES); continue; }
            j = nb;
            nb++;
            uint b = brec(v, j);
            float4 p0 = gBodies[7u * body];
            float4 p2 = gBodies[7u * body + 2u];
            float4 p3 = gBodies[7u * body + 3u];
            float4 p4 = gBodies[7u * body + 4u];
            // Asleep or not, its motion; static and kinematic: 1/M = 0 (infinite mass).
            gMisc[b] = asuint(float4(p0.xyz - asfloat(gMisc[v].xyz), p0.w));
            gMisc[b + 1u] = asuint(gBodies[7u * body + 1u]);
            gMisc[b + 2u] = asuint(float4(p3.xyz, 1.0));
            gMisc[b + 3u] = uint4(asuint(p4.xyz), body);
            gMisc[b + 4u] = asuint(float4(p0.w > 0.0 ? p2.xyz : float3(0.0), 0.0));
            gWork[wShare() + body].add(1u);
        }
        float4 lp = asfloat(gColliders[4u * c + 1u]);
        uint4 lq = gColliders[4u * c + 2u];
        float4 pr = asfloat(gColliders[4u * c + 3u]);
        uint at = srec(v, ns);
        uint span = 0u;
        uint code = 0u;
        uint hullCounts = 0u;
        float3 prm = float3(0.0);
        if (kind == 2u) { code = 0u; prm = pr.xyz; }
        else if (kind == 0u) { code = 1u; prm = float3(pr.x, 0.0, 0.0); }
        else if (kind == 1u) { code = 2u; prm = float3(pr.x, pr.y, 0.0); }
        else
        {
            code = 3u;
            uint base = asuint(pr.x) & HULL_MAX_BASE;
            float scale = pr.y;
            uint head = gHulls[4u * base];
            uint nv = head & 255u;
            uint nf = (head >> 8) & 255u;
            uint recs = hullRecords(base);
            if (planes + recs > gP.planes) { gWork[1].or(ST_PLANES); continue; }
            for (uint f = 0u, fStop = EB_END(593, f, nf, 1u); f < fStop; f++)
            {
                uint w = 4u * (base + 4u + nv + f);
                float3 nn = asfloat(uint3(gHulls[w], gHulls[w + 1u], gHulls[w + 2u]));
                float d = asfloat(gHulls[w + 3u]) * scale;
                gMisc[prec(v, planes + f)] = asuint(float4(nn, d));
            }
            hullCounts = hullGeometry(v, base, scale, planes + nf);
            span = planes | (nf << 16);
            planes += recs;
        }
        gMisc[at] = uint4(code, j, c, span);
        gMisc[at + 1u] = asuint(float4(lp.xyz, prm.x));
        gMisc[at + 2u] = lq;
        gMisc[at + 3u] = uint4(asuint(prm.y), asuint(prm.z), asuint(sph.w + 0.01), hullCounts);
        gMisc[at + 4u] = uint4(0u);
        gMisc[at + 5u] = uint4(0u);
        gMisc[at + 6u] = asuint(float4(sph.xyz - asfloat(gMisc[v].xyz), 0.0));
        // The body's shapes lie in [first, last] (its record's (1/I, w) w: first + 1 |
        // (last + 1) << 16; 0, as a host writes it, means every shape): its step sums
        // (explicit_beams_rigid.slang cplIntegrateBody) scan that range only.
        uint bw = brec(v, j) + 4u;
        uint4 r4 = gMisc[bw];
        uint first = (r4.w & 0xFFFFu) == 0u ? ns + 1u : (r4.w & 0xFFFFu);
        gMisc[bw] = uint4(r4.xyz, first | ((ns + 1u) << 16));
        ns++;
    }
    gMisc[vrec(v)] = uint4(nb, ns, planes, 0u);
    listAppend(v, ns);
    ebLoopCapFlush();
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cpl_share(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = listed(gid.x);
    if (v == NONE) return ebLoopCapFlush();
    for (uint j = lid.x, jStop = EB_END(594, j, gMisc[vrec(v)].x, 64u); j < jStop; j += 64u)
    {
        uint b = brec(v, j);
        uint body = gMisc[b + 3u].w;
        float4 r2 = asfloat(gMisc[b + 2u]);
        gMisc[b + 2u] = asuint(float4(r2.xyz, float(max(gWork[wShare() + body].load(), 1u))));
    }
    ebLoopCapFlush();
}

float3 to_zup(float3 v) { return float3(v.x, -v.z, v.y); }

// The point on a shape impulse's line of action nearest its reference.
float3 actionPoint(float3 ref, float3 j, float3 m)
{
    float jj = dot(j, j);
    return jj > 0.0 ? ref + cross(j, m) / jj : ref;
}

// cpl_loads for shape k of vehicle v.
void shapeLoad(uint v, uint k)
{
    uint at = srec(v, k);
    uint c = gMisc[at].z;
    uint4 info = gChunkInfo[2u * c];
    float3 j = asfloat(gMisc[at + 4u].xyz);
    if (info.x == NONE || all(j == float3(0.0))) return;
    float3 m = asfloat(gMisc[at + 5u].xyz);
    float3 ref = asfloat(gMisc[at + 6u].xyz);
    uint row = gWork[0].add(1u);
    if (row >= gP.rowCap) { gWork[1].or(ST_ROWS); return; }
    uint base = 2u * (gP.ingestSlot * gP.ingestRows + row);
    float3 jz = to_zup(j);
    float3 p = to_zup(actionPoint(ref, j, m) + asfloat(gMisc[v].xyz));
    // coupling.slang Record: ids (chunk, other body: none, slot, valid), point (|J|),
    // impulse (closing speed 0: a sustained load, J / dt), other (mass, modulus, radius),
    // key (substep, this collider, other: the vehicle, feature).
    float mass = asfloat(vehicleAt(v).z);
    gRecords[5u * base] = uint4(info.x, NONE, base, 1u);
    gRecords[5u * base + 1u] = asuint(float4(p, length(jz)));
    gRecords[5u * base + 2u] = asuint(float4(jz, 0.0));
    gRecords[5u * base + 3u] = asuint(float4(mass, 2.0e11, 1.0, 0.0));
    gRecords[5u * base + 4u] = uint4(gP.ingestSlot, c, 0x80000000u | v, 2u * k);
    gRecords[5u * (base + 1u)] = uint4(NONE, NONE, base + 1u, 0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cpl_loads(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = listed(gid.x);
    if (v == NONE) return ebLoopCapFlush();
    if (gP.ingestRows == 0u) return ebLoopCapFlush();
    for (uint k = lid.x, kStop = EB_END(595, k, gMisc[vrec(v)].y, 64u); k < kStop; k += 64u) shapeLoad(v, k);
    ebLoopCapFlush();
}

[shader("compute")]
[numthreads(1, 1, 1)]
void cpl_loads_end(uint3 tid : SV_DispatchThreadID)
{
    if (gP.ingestRows == 0u) return ebLoopCapFlush();
    gRowCounts[gP.ingestSlot] = min(gWork[0].load(), gP.rowCap);
    ebLoopCapFlush();
}

// A proxy body's window impulse and its angular impulse about point c.
void proxyImpulse(uint v, uint j, float3 c, out float3 J, out float3 L)
{
    J = float3(0.0);
    L = float3(0.0);
    uint ns = gMisc[vrec(v)].y;
    for (uint q = 0u, qStop = EB_END(596, q, ns, 1u); q < qStop; q++)
    {
        uint aq = srec(v, q);
        if (gMisc[aq].y != j) continue;
        float3 i = asfloat(gMisc[aq + 4u].xyz);
        J += i;
        L += asfloat(gMisc[aq + 5u].xyz) + cross(asfloat(gMisc[aq + 6u].xyz) - c, i);
    }
}

// The impulse records of vehicle v's window, one per (proxy body, body its shapes belong
// to now). A proxy body of infinite mass in the window (static, anchored or kinematic: the
// vehicle met an immovable body) sends nothing: its reaction went into the ground, as the
// vehicle felt it. A proxy body still whole sends its impulse. One that split (or whose
// shapes moved to other bodies) after the frame: each piece takes the change of motion
// the whole body's impulse gives at the piece's centre of mass, m (dv + dw x r) and I dw,
// as the pieces of a split move with their parent's motion. Momentum closes; no piece
// moves faster than the whole body would have.
void emitPair(uint v, uint j, uint body, uint key)
{
    uint b = brec(v, j);
    float4 c0 = asfloat(gMisc[b]);
    if (!(c0.w > 0.0)) return;
    float4 cur = gBodies[7u * body];
    if (!(cur.w > 0.0)) return;
    float3 J;
    float3 L;
    bool whole = gMisc[b + 3u].w == body && asuint(cur.w) == asuint(c0.w);
    if (whole)
    {
        proxyImpulse(v, j, cur.xyz - asfloat(gMisc[v].xyz), J, L);
    }
    else
    {
        float3 Jp;
        float3 Lp;
        proxyImpulse(v, j, c0.xyz, Jp, Lp);
        float4 q = asfloat(gMisc[b + 1u]);
        float3 ii = asfloat(gMisc[b + 4u].xyz);
        float3 dv = Jp * c0.w;
        float3 dw = qrot(q, ii * qrot(float4(-q.xyz, q.w), Lp));
        J = (dv + cross(dw, (cur.xyz - asfloat(gMisc[v].xyz)) - c0.xyz)) / cur.w;
        float4 rot = gBodies[7u * body + 1u];
        float3 ci = gBodies[7u * body + 2u].xyz;
        float3 lw = qrot(float4(-rot.xyz, rot.w), dw);
        float3 li = float3(ci.x > 0.0 ? lw.x / ci.x : 0.0, ci.y > 0.0 ? lw.y / ci.y : 0.0, ci.z > 0.0 ? lw.z / ci.z : 0.0);
        L = qrot(rot, li);
    }
    if (all(J == float3(0.0)) && all(L == float3(0.0))) return;
    uint r = gScratch[gP.impulseCount].add(1u);
    if (r >= gP.impulseCap) { gWork[1].or(ST_IMPULSES); return; }
    uint w = gP.impulseBase + 8u * r;
    gScratch[w].store(body);
    gScratch[w + 1u].store(((gP.keyBase + v * 1024u + key) << 8) | IMPULSE_LINEAR_ANGULAR);
    gScratch[w + 2u].store(asuint(J.x));
    gScratch[w + 3u].store(asuint(J.y));
    gScratch[w + 4u].store(asuint(J.z));
    gScratch[w + 5u].store(asuint(L.x));
    gScratch[w + 6u].store(asuint(L.y));
    gScratch[w + 7u].store(asuint(L.z));
}

// A lane a shape (64 lanes, shape k by lane k % 64): each shape that is its pair's first
// (in shape order) emits that pair's record, its sums in shape order as one lane wrote
// them before. The records' order in the list is the appends', which the rigid step
// never reads: it sums a body's records by key (ccd_step.slang apply_impulses).
[shader("compute")]
[numthreads(64, 1, 1)]
void cpl_emit(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = listed(gid.x);
    if (v == NONE) return ebLoopCapFlush();
    uint ns = gMisc[vrec(v)].y;
    // (Tests, gP.serialPlan: lane 0 alone, the shapes one after another.)
    uint lanes = gP.serialPlan != 0u ? 1u : 64u;
    if (lid.x >= lanes) return ebLoopCapFlush();
    for (uint k = lid.x, kStop = EB_END(597, k, ns, lanes); k < kStop; k += lanes)
    {
        uint at = srec(v, k);
        uint j = gMisc[at].y;
        uint body = gColliders[4u * gMisc[at].z].x;
        bool first = true;
        for (uint q = 0u, qStop = EB_END(598, q, k, 1u); q < qStop; q++)
        {
            uint aq = srec(v, q);
            if (gMisc[aq].y == j && gColliders[4u * gMisc[aq].z].x == body) { first = false; break; }
        }
        if (first) emitPair(v, j, body, k);
    }
    ebLoopCapFlush();
}

[shader("compute")]
[numthreads(1, 1, 1)]
void cpl_emit_end(uint3 tid : SV_DispatchThreadID)
{
    gScratch[gP.impulseFirst].store(min(gScratch[gP.impulseCount].load(), gP.impulseCap));
    ebLoopCapFlush();
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cpl_sleep_mask(uint3 tid : SV_DispatchThreadID)
{
    uint t = tid.x;
    if (gP.sleepBase == 0u || 32u * t >= gP.vehicles) return ebLoopCapFlush();
    uint word = 0u;
    for (uint k = 0u; k < 32u && 32u * t + k < gP.vehicles; k++)
        word |= (gMisc[gP.sleepBase + 32u * t + k].y & 1u) << k;
    gWork[2u + gP.maxBodies + 12u * gP.vehicles + t].store(word);
    ebLoopCapFlush();
}
