// The share of a hit's force f that node role r of the hit gets (1 the contacter,
// 2..4 the triangle's n1, n2, n0), as the accumulators add it.
float3 hitShare(float3 f, uint role, float alpha, float beta, float gamma)
{
    if (role == 1u) return f;
    if (role == 2u) return -alpha * f;
    if (role == 3u) return -beta * f;
    return -gamma * f;
}

int3 toFixed3(float3 f) { return int3(toFixed(f.x), toFixed(f.y), toFixed(f.z)); }

// Triangle gt's geometry; mi < vehicles (none) when it waits this step.
// A breakable triangle (spec §18) is dead once any of its four listed beams was
// broken before this beam phase (.z set, .w not this phase's own break).
bool triangleDead(uint gt)
{
    uint4 l = gStatic[gP.triBreak + gt];
    for (uint k = 0u; k < 4u; k++)
    {
        uint j = l[k];
        if (j == 0xFFFFFFFFu) continue;
        float4 s1 = gBeams[j * 2u + 1u];
        if (s1.z != 0.0 && s1.w != 1.0) return true;
    }
    return false;
}

// A triangle's frame from its three world positions (TriFrame::new).
void triangleFrame(inout TriGeo t, float3 p1, float3 p2)
{
    float3 u = nf3(p1 - t.p0);
    float3 w = nf3(p2 - t.p0);
    float3 cr = scross(u, w);
    float cl = ssqrt(sdot(cr, cr));
    t.n = cl > 0.0 ? float3(sdiv(cr.x, cl), sdiv(cr.y, cl), sdiv(cr.z, cl)) : cr;
    float det = sdot(u, scross(w, t.n));
    t.solvable = abs(det) > 1e-6;
    float rd = t.solvable ? sdiv(1.0, det) : 0.0;
    t.ra = sscale(scross(w, t.n), rd);
    t.rb = sscale(scross(t.n, u), rd);
    t.rn = sscale(scross(u, w), rd);
}

// A triangle's geometry before its liveness (triangleGeo without triangleKill).
TriGeo triangleGeoLive(uint gt, float3 origin, float r, uint parity)
{
    TriGeo t;
    t.mi = gP.vehicles + gt;
    t.tn = gStatic[gP.triBase + gt];
    t.p0 = nf3(origin + gNodes[t.tn.x * 3u].xyz);
    float3 p1 = nf3(origin + gNodes[t.tn.y * 3u].xyz);
    float3 p2 = nf3(origin + gNodes[t.tn.z * 3u].xyz);
    triangleFrame(t, p1, p2);
    uint z = gMisc[t.mi].z;
    t.clip = ((gStatic[t.tn.w * 4u + 3u].x & FLAG_ANTI_CLIP) != 0u ? 1u : 0u) | ((z & TRI_ANTI_ONLY) != 0u ? 2u : 0u);
    float reach = (t.clip & 1u) != 0u ? triReach(t.tn, r, parity) : r;
    t.lo = min(min(t.p0, p1), p2) - float3(reach, reach, reach);
    t.hi = max(max(t.p0, p1), p2) + float3(reach, reach, reach);
    t.qs = 0u;
    return t;
}

// A broken triangle (spec §18), or an anti-clip-only one without anti-clip: none.
void triangleKill(inout TriGeo t, uint gt)
{
    if (((gMisc[t.mi].z & TRI_BREAKABLE) != 0u && triangleDead(gt)) || t.clip == 2u)
    {
        t.lo = float3(3.0e38, 3.0e38, 3.0e38);
        t.hi = -t.lo;
    }
}

TriGeo triangleGeo(uint gt, float3 origin, float r, uint parity)
{
    TriGeo t = triangleGeoLive(gt, origin, r, parity);
    triangleKill(t, gt);
    return t;
}

// Anti-clip's frame of triangle t a step back (selfClip's crossing test).
TriGeo trianglePrev(TriGeo t, float3 origin)
{
    TriGeo pf;
    pf.p0 = ssub3(t.p0, sscale(gNodes[t.tn.x * 3u + 1u].xyz, STEP));
    float3 b1 = ssub3(nf3(origin + gNodes[t.tn.y * 3u].xyz), sscale(gNodes[t.tn.y * 3u + 1u].xyz, STEP));
    float3 b2 = ssub3(nf3(origin + gNodes[t.tn.z * 3u].xyz), sscale(gNodes[t.tn.z * 3u + 1u].xyz, STEP));
    triangleFrame(pf, b1, b2);
    return pf;
}

// --- The triangles' step geometry, once a step (lossless) ---
// The grid walk spreads a triangle over GRID_SUBS lanes, and each computed the
// triangle's geometry (and, with anti-clip, for every contacter in its box, its frame
// a step back). After the node phase nothing a triangle's geometry reads changes until
// the triangle phase ends (positions, velocities, the fastest node's speed, the
// throttle words), so the beam phase writes each tested triangle's once (triGeoFill,
// the same functions on the same inputs) and the walk loads it (triGeoLoad); its
// liveness (triangleKill: beam states the beam phase writes) is applied at the load,
// where triangleGeo applied it. gBeams at gP.triGeo, TRI_GEO_RECS records per
// triangle: (p0, n.x), (n.yz, ra.xy), (ra.z, rb), (rn, flags), (lo, hi.x),
// (hi.yz, q0.xy), (q0.z, qra), (qrb, qrn.x), (qrn.yz, 0, 0); flags (a whole float)
// solvable | clip << 1 | qsolv << 3 | qs << 4.
static const uint TRI_GEO_RECS = 9u;
void triGeoFill(uint gt, float3 origin, float r, uint parity)
{
    TriGeo t = triangleGeoLive(gt, origin, r, parity);
    TriGeo pf;
    pf.p0 = float3(0.0, 0.0, 0.0);
    pf.ra = float3(0.0, 0.0, 0.0);
    pf.rb = float3(0.0, 0.0, 0.0);
    pf.rn = float3(0.0, 0.0, 0.0);
    pf.solvable = false;
    uint qs = 0u;
    if ((t.clip & 1u) != 0u)
    {
        pf = trianglePrev(t, origin);
        qs = 1u;
    }
    uint flags = (t.solvable ? 1u : 0u) | (t.clip << 1) | (pf.solvable ? 8u : 0u) | (qs << 4);
    uint at = gP.triGeo + gt * TRI_GEO_RECS;
    gBeams[at] = float4(t.p0, t.n.x);
    gBeams[at + 1u] = float4(t.n.yz, t.ra.xy);
    gBeams[at + 2u] = float4(t.ra.z, t.rb);
    gBeams[at + 3u] = float4(t.rn, float(flags));
    gBeams[at + 4u] = float4(t.lo, t.hi.x);
    gBeams[at + 5u] = float4(t.hi.yz, pf.p0.xy);
    gBeams[at + 6u] = float4(pf.p0.z, pf.ra);
    gBeams[at + 7u] = float4(pf.rb, pf.rn.x);
    gBeams[at + 8u] = float4(pf.rn.yz, 0.0, 0.0);
}

// The walk's list mode needs only the box (and the liveness that empties it): triangle
// gt's lo, hi and clip from triGeoFill's records, the rest left unset.
TriGeo triGeoBox(uint gt)
{
    TriGeo t;
    t.mi = gP.vehicles + gt;
    uint at = gP.triGeo + gt * TRI_GEO_RECS;
    float4 r3 = gBeams[at + 3u];
    float4 r4 = gBeams[at + 4u];
    float4 r5 = gBeams[at + 5u];
    t.clip = (uint(r3.w) >> 1) & 3u;
    t.lo = r4.xyz;
    t.hi = float3(r4.w, r5.xy);
    t.qs = 0u;
    triangleKill(t, gt);
    return t;
}

// Triangle gt's geometry as triangleGeo computes it, from triGeoFill's records.
TriGeo triGeoLoad(uint gt)
{
    TriGeo t;
    t.mi = gP.vehicles + gt;
    t.tn = gStatic[gP.triBase + gt];
    uint at = gP.triGeo + gt * TRI_GEO_RECS;
    float4 r0 = gBeams[at];
    float4 r1 = gBeams[at + 1u];
    float4 r2 = gBeams[at + 2u];
    float4 r3 = gBeams[at + 3u];
    float4 r4 = gBeams[at + 4u];
    float4 r5 = gBeams[at + 5u];
    float4 r6 = gBeams[at + 6u];
    float4 r7 = gBeams[at + 7u];
    float4 r8 = gBeams[at + 8u];
    uint flags = uint(r3.w);
    t.p0 = r0.xyz;
    t.n = float3(r0.w, r1.xy);
    t.ra = float3(r1.zw, r2.x);
    t.rb = r2.yzw;
    t.rn = r3.xyz;
    t.solvable = (flags & 1u) != 0u;
    t.clip = (flags >> 1) & 3u;
    t.lo = r4.xyz;
    t.hi = float3(r4.w, r5.xy);
    t.qs = (flags >> 4) & 1u;
    t.q0 = float3(r5.zw, r6.x);
    t.qra = r6.yzw;
    t.qrb = r7.xyz;
    t.qrn = float3(r7.w, r8.xy);
    t.qsolv = (flags & 8u) != 0u;
    triangleKill(t, gt);
    return t;
}

// Lane `sub` of the `lanes` lanes of triangle gt tests candidates sub, sub + lanes, …
// Forces go to fixed-point atomics, so the lanes' order does not change the sums.
void triangleCandidates(uint gt, uint sub, uint subs, float3 origin, float r, float cell, uint candOff, uint candCount,
                        uint beamOff, uint groupOff, uint groupCount, uint parity, uint mode);
void triangleTest(uint gt, uint sub, uint lanes, float3 origin, float r, uint candOff, uint candCount,
                  uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    triangleCandidates(gt, sub, lanes, origin, r, 1.0, candOff, candCount, beamOff, groupOff, groupCount, parity, 0u);
}

// The grid's bucket of a cell.
uint gridBucket(int3 c)
{
    return (uint(c.x) * 73856093u ^ uint(c.y) * 19349663u ^ uint(c.z) * 83492791u) & (GRID_BUCKETS - 1u);
}

// Triangle gt against the contacters in the grid cells its box (padded by r)
// covers: each covered bucket's run once (two cells can share a bucket). A box
// over more than GRID_MAX_CELLS cells tests every candidate instead. The runs
// hold every contacter, so the contacts found are the all-pairs test's. GRID_SUBS
// lanes share a triangle: lane `sub` takes the cells sub, sub + GRID_SUBS, … (and
// the same share of the candidates when it falls back), so a triangle's walk is
// spread rather than serial on one lane.
static const uint GRID_SUBS = 8u;

// The triangles tested this step (wait <= 0 after the throttle update), listed by
// the node phase for a vehicle of up to TRI_LIST triangles; the triangle phase then
// walks only them. Counters by step parity: the beam phase clears the next step's.
static const uint TRI_LIST = 2048u;
groupshared uint gsTested[TRI_LIST];
groupshared Atomic<uint> gsTestedCount[2];

void triangleGrid(uint gt, uint sub, float3 origin, float r, float cell, uint candOff, uint candCount,
                  uint beamOff, uint groupOff, uint groupCount, uint parity, uint listAt)
{
    triangleGridWalk(gt, sub, origin, r, cell, candOff, candCount, beamOff, groupOff, groupCount, parity, false, 0u,
                     listAt);
}

// --- The grid walk's pairs, then their tests (lossless) ---
// The walk's lanes take (triangle, cell) shares of very different sizes, and each
// tested its contacters as it found them, so the step waited for the lane with the
// most tests. With gP.triPairs, the frame's walk lists the (triangle, contacter) pairs
// that pass the box test tryContact applies first (with its expression), in gNext at
// triPairs + vehicle · 2 · TRI_PAIR_CAP (2 words a pair: global triangle, node; count in
// gsPairCount, unused by a grid step), and after a barrier the lanes test the list,
// a pair each. The tests and their results are the walk's: the hits are applied in
// the specification's order whatever order they were found in (selfApply). A list past
// TRI_PAIR_CAP runs the walk again with its tests (the list has no effects).
static const uint TRI_PAIR_CAP = 32768u;
uint triPairsAt(uint v) { return gP.triPairs + v * 2u * TRI_PAIR_CAP; }
uint triPairAdd();
// Lanes a triangle's walk is spread over when it lists (its cells sub, sub + LIST_SUBS, ...):
// fewer than the testing walk's GRID_SUBS, so less of a triangle's fixed cost repeats.
static const uint LIST_SUBS = 2u;

// triangleGrid's walk, the bucket offsets in workgroup memory (gsOff) or, `at`, in
// gNext at offBase (a contact group member's grid, triangleGridAt). One call site of
// tryContact for both paths (the compilers inline every call): too many cells, the
// outer loop walks the candidates (one each); else the covered cells, each bucket's
// run once. The same tests in the same order as two separate loops.
void triangleGridWalk(uint gt, uint sub, float3 origin, float r, float cell, uint candOff, uint candCount,
                      uint beamOff, uint groupOff, uint groupCount, uint parity, bool at, uint offBase, uint listAt)
{
    if (asint(gMisc[gP.vehicles + gt].x) > 0) return;
    // The frame's walk: the step's geometry from the beam phase (triGeoFill).
    TriGeo t = listAt != 0u ? triGeoBox(gt) : (!at && gP.triGeo != 0u ? triGeoLoad(gt) : triangleGeo(gt, origin, r, parity));
    uint subs = listAt != 0u ? LIST_SUBS : GRID_SUBS;
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
    for (uint i = sub, iStop = EB_END(14, i, outer, subs); i < iStop; i += subs)
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
                for (uint j = 0u, jStop = EB_END(15, j, i, 1u); j < jStop; j++)
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
            e0 = at ? gNext[offBase + b] : gsOff[b];
            e1 = at ? gNext[offBase + b + 1u] : gsOff[b + 1u];
        }
        for (uint e = e0, eGuard = 0u; e < e1; e++)
        {
            EB_PASS(16, eGuard);
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
            if (listAt != 0u)
            {
                // Listed for the test pass: a bucket's entry as the testing walk passes
                // it to tryContact (its box test above); every candidate (too many
                // cells) only when inside the box tryContact tests first, with its
                // expression (outside, tryContact returns before any effect).
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
                continue;
            }
            tryContact(t, h, origin, r, beamOff, groupOff, groupCount, parity);
        }
    }
}

// A step with few (triangle, contacter) pairs skips the grid's counting sort (its
// scan costs five barriers) and walks every candidate instead, keeping exactly the
// ones the grid walk would test: inside the triangle's box and in a bucket its
// cells cover (or every candidate when the box covers too many cells), with the
// grid's own expressions. So the hits are the grid's, bit for bit.
// 0: a vehicle with a grid always uses it. The skip (16 pairs a lane) gave the same
// hits but resolves the documented same-contacter race differently (the lanes of a
// wave read before any adds), which moved the E36 drive ensemble against RoR from
// 115-150 mm to 310-520 mm (fail; 3 runs each): reverted for accuracy.
static const uint BRUTE_PAIRS_PER_LANE = 16u;
void triangleGridBrute(uint gt, uint sub, uint subs, float3 origin, float r, float cell, uint candOff,
                       uint candCount, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    triangleCandidates(gt, sub, subs, origin, r, cell, candOff, candCount, beamOff, groupOff, groupCount, parity, 1u);
}

// Whether contacter h is provably too far from triangle t's plane for tryContact to do
// anything: a contact (tryContact's triangleHit, or anti-clip's selfClip, held or
// crossed) needs the signed distance delta = rn · (p - p0) of triangleHit within
// max(r, CLIP_DEPTH); computed here as triangleHit computes it, with an allowance for
// rounding. tryContact has no effect before it returns for such a pair.
bool pairFar(TriGeo t, uint h, float3 origin, float r)
{
    if (!t.solvable) return false;
    float3 p = nf3(origin + gNodes[h * 3u].xyz);
    float3 q = nf3(p - t.p0);
    float delta = sdot(t.rn, q);
    float reach = max(r, CLIP_DEPTH);
    return abs(delta) > reach * 1.001 + 1.0e-4;
}

#include "explicit_beams_tri_cands.slang"

// A step without the grid finds its self-collision candidates early: the beam
// phase (positions are final after the node phase) lists the (triangle, candidate)
// pairs that pass the tests before tryContact's hit test (the box, and for a grid
// vehicle the grid's covered buckets, with the same expressions), on the lanes the
// beams leave idle; the triangle phase then runs tryContact on the listed pairs
// only (usually none). A fuller list than PAIR_LIST runs the step's usual walk.
static const uint PAIR_LIST = 1024u;
groupshared uint gsPairs[PAIR_LIST];
groupshared Atomic<uint> gsPairCount;
uint triPairAdd() { return gsPairCount.add(1u); }

void trianglePairs(uint t, uint gt, uint sub, uint subs, float3 origin, float r, float cell, uint candOff,
                   uint candCount, uint parity)
{
    if (asint(gMisc[gP.vehicles + gt].x) > 0) return;
    uint4 tn = gStatic[gP.triBase + gt];
    float3 p0 = nf3(origin + gNodes[tn.x * 3u].xyz);
    float3 p1 = nf3(origin + gNodes[tn.y * 3u].xyz);
    float3 p2 = nf3(origin + gNodes[tn.z * 3u].xyz);
    // The box tryContact tests (triangleGeo's: with anti-clip, its reach).
    float reach = triReach(tn, r, parity);
    float3 lo = min(min(p0, p1), p2) - float3(reach, reach, reach);
    float3 hi = max(max(p0, p1), p2) + float3(reach, reach, reach);
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
    for (uint k = sub, kStop = EB_END(19, k, candCount, subs); k < kStop; k += subs)
    {
        uint h = gLists[candOff + k];
        if (h == tn.x || h == tn.y || h == tn.z) continue;
        float3 e = gNodes[h * 3u].xyz;
        if (!every)
        {
            // The grid walk's tests (triangleGridBrute).
            float3 p = origin + e;
            if (any(p < lo) || any(p > hi)) continue;
            int3 c = int3(floor(e / cell));
            if (!(all(c >= c0) && all(c < c0 + span)))
            {
                uint b = gridBucket(c);
                bool covered = false;
                for (uint i = 0u, iStop = EB_END(20, i, cells, 1u); i < iStop; i++)
                    covered = covered
                        || gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z)))) == b;
                if (!covered) continue;
            }
        }
        // tryContact's box test.
        float3 x = nf3(origin + e);
        if (any(x < lo) || any(x > hi)) continue;
        uint i = gsPairCount.add(1u);
        if (i < PAIR_LIST) gsPairs[i] = (t << 16) | k;
    }
}

// --- The grid walk's pair list, built once and reused over steps (lossless) ---
// The solo frame's walk listed its (triangle, contacter) pairs every step (the grid's
// counting sort, five barriers, then every triangle's cells). With gP.pairReuse (a
// margin M, float bits; 0 off) and gP.pairBuild, a step that builds lists every
// triangle (waiting or not) against the box padded by M more, and later steps test
// that list again while it provably holds every pair the walk would test:
// * A pair absent from the list was outside the padded box at the build. With every
//   tracked node's displacement since the build (the contacters and the triangles'
//   nodes, the only nodes a box test reads), per axis, within [m, m + R] (R the
//   spread of the displacements: the common translation cancels in the box test), the
//   contacter has moved at most R relative to any of the triangle's nodes, and the
//   box's reach (triReach: the triangle's fastest node and the vehicle's fastest) has
//   grown at most h · (2 G + dV) (G the largest growth of a node's speed, dV the
//   vehicle's fastest node's): the pair cannot be inside its box while
//   R + h · (2 G + dV) + eps <= M (eps: rounding, from the coordinates' size). Else
//   the step builds again (and at a frame's first step).
// * Each listed pair is tested as the walk would test it (pairReuseTest: the waiting
//   triangle skipped, the walk's cell, bucket and box tests on this step's geometry,
//   then tryContact), and tryContact itself has no effect outside its box; the hits are
//   applied in the specification's order whatever order they were found in (selfApply).
// So the hits are the walk's, bit for bit. A build past TRI_PAIR_CAP walks with its
// tests that step (as before) and builds again the next.
// gBeams at pairBuild: per node (position, speed) at the build; then per vehicle the
// statistics (builds, reused steps, pairs listed, most listed; bits, accumulated). The
// tracked nodes carry NODE_PAIR_TRACKED in their node info (the host's: contacters and
// collision triangles' nodes).
// This step's path rides in gsUniformLanes (the grid flag, bit 0, workgroup-uniform):
// bits 1..2 the list's mode, 0 the walk, 1 build the list, 2 reuse it. The list's state
// is in gsCount past its buckets (gsReuse): 0..7 order-preserving
// keys (pairReuseKey) of the displacements' largest x, y, z and largest negated x, y, z,
// the largest speed growth, the largest coordinate's size; then the list valid, its
// pairs, the fastest node's speed at the build (bits), the statistics (4 words).
static const uint RU_VALID = 8u;
static const uint RU_COUNT = 9u;
static const uint RU_VMAX = 10u;
static const uint RU_STATS = 11u;
static const uint RU_STREAK = 15u; // steps the current list was reused, then (after a
// list reused on fewer than gP.pairStreak steps) the grid steps left on the walk, as
// REUSE_BACKOFF_FLAG | n
static const uint REUSE_BACKOFF = 8u;
static const uint REUSE_BACKOFF_FLAG = 0x80000000u;

uint pairReuseKey(float f)
{
    uint b = asuint(f);
    return (b & 0x80000000u) != 0u ? ~b : (b | 0x80000000u);
}
float pairReuseVal(uint k) { return asfloat((k & 0x80000000u) != 0u ? (k & 0x7FFFFFFFu) : ~k); }
bool pairReuseOn() { return gP.pairReuse != 0u && gP.pairBuild != 0u && gP.triPairs != 0u && gP.triGeo != 0u; }

// Lane 0, the frame's start: no list yet.
void pairReuseBegin()
{
    if (!pairReuseOn()) return;
    for (uint k = 0u; k < 16u; k++) gsCount[GRID_BUCKETS + k].store(0u);
    if (gP.pieceBase != 0u)
    {
        for (uint k = 16u; k < REUSE_WORDS; k++) gsCount[GRID_BUCKETS + k].store(0u);
    }
}

static const uint NODE_PAIR_TRACKED = 128u;

void pairReuseList(uint gt, uint sub, float3 origin, float r, float cell, uint candOff, uint candCount,
                   uint parity, uint listAt, float margin);
void pairReuseTest(uint gt, uint h, float3 origin, float r, float cell, uint beamOff, uint groupOff,
                   uint groupCount, uint parity);
void piecesClearBuild();
void piecesEnd(uint v);
bool pairFar(TriGeo t, uint h, float3 origin, float r);
void piecesTestTriangle(uint gt, uint sub, uint subs, float3 origin, float r, float cell, uint candOff, uint candCount,
                        uint beamOff, uint groupOff, uint groupCount, uint parity);

