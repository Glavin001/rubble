// --- The reused pair list's pieces: parts torn off the vehicle (lossless) ---
// With one bound for the whole vehicle (the spread R of every tracked node's displacement)
// a detached piece in free fall invalidates the list every step: it moves metres a frame
// relative to the car. The host labels each node with a piece (gNext at gP.pieceBase: 0
// the main body, 1..PIECE_LOOSE - 1 a piece with collision triangles of its own,
// PIECE_LOOSE the rest; a vehicle's word after the nodes nonzero: labels in use). Any
// labelling is sound, the labels only decide how much is reused: the list is kept while
// * main-main pairs: the main body's spread R0 + h · (2 G + dV) + eps <= M (as before);
// * a piece's own pairs (its triangle, its contacter): its spread likewise;
// * a triangle whose nodes carry different labels, or the PIECE_LOOSE label, is volatile:
//   never listed, its candidates tested every step as the walk tests them;
// * cross pairs (triangle of group a, contacter h of another group): at the build, h was
//   s = max(gap(h, E_a) - Pb, M) from every box of a (E_a: the box of a's tracked nodes,
//   Pb: a bound on any triangle's padding; M: unlisted), and h has since moved relative to every node of a
//   by less than s less the boxes' growth: per axis d_h - s < lo_a - g and
//   -d_h - s < -hi_a - g (lo_a, hi_a: a's displacements; contacters of a piece, a key per
//   axis), or for the main body's contacters (one group) the largest relative
//   displacement of the two groups below the least s.
// gsCount past its buckets: RU_D group g's (1..PIECE_LOOSE - 1) displacement keys (as
// words 0..5); RU_E group g's (0..PIECE_LOOSE) position keys (largest x, y, z, negated);
// RU_K group a's (0..PIECE_LOOSE - 1) contacter keys (d - s, -d - s per axis); RU_S per
// group a the least s of a main contacter (pairReuseKey of -s); RU_VALL the fastest
// tracked node's speed; RU_VOL the volatile triangles listed. gNext after the labels and
// the vehicles' words: per node PIECE_GROUPS separations (float bits; +inf none), then per
// triangle the volatile list (a vehicle's at its triOff).
static const uint PIECE_GROUPS = 8u;
static const uint PIECE_LOOSE = 7u;
static const uint RU_D = 16u;
static const uint RU_E = RU_D + 6u * (PIECE_LOOSE - 1u);
static const uint RU_K = RU_E + 6u * PIECE_GROUPS;
static const uint RU_S = RU_K + 6u * PIECE_LOOSE;
static const uint RU_VALL = RU_S + PIECE_GROUPS;
static const uint RU_VOL = RU_VALL + 1u;
static const uint RU_VSUM = RU_VOL + 1u; // volatile triangles tested, summed over the frame's steps
static const uint PIECE_SUBS = 8u;

uint pieceLabel(uint gi) { return min(gNext[gP.pieceBase + gi] & 0xFFu, PIECE_LOOSE); }
bool piecesOn(uint v)
{
    return gP.pieceBase != 0u && gNext[gP.pieceBase + gP.totalNodes + v] != 0u;
}
uint pieceSigmaAt(uint gi, uint a) { return gP.pieceBase + gP.totalNodes + gP.vehicles + gi * PIECE_GROUPS + a; }
uint pieceVolAt(uint t) { return gP.pieceBase + gP.totalNodes * (1u + PIECE_GROUPS) + gP.vehicles + t; }
void pieceKeyMax(uint w, float f)
{
    uint k = pairReuseKey(f);
    if (k > gsCount[GRID_BUCKETS + w].load()) gsCount[GRID_BUCKETS + w].max(k);
}
// A group's six keys (largest x, y, z, then of the negated) present.
bool pieceKeysSet(uint w) { return gsCount[GRID_BUCKETS + w].load() != 0u; }
float pieceKey(uint w) { return pairReuseVal(gsCount[GRID_BUCKETS + w].load()); }
// A triangle with a retired node (nodeRetired): tryContact rejects every pair.
bool pieceTriRetired(uint t)
{
    uint4 tn = gStatic[gP.triBase + t];
    return nodeRetired(tn.x) || nodeRetired(tn.y) || nodeRetired(tn.z);
}
// Triangle t's nodes in different groups, or in the loose group: tested every step.
bool pieceVolatile(uint t)
{
    uint4 tn = gStatic[gP.triBase + t];
    uint a = pieceLabel(tn.x);
    return a == PIECE_LOOSE || pieceLabel(tn.y) != a || pieceLabel(tn.z) != a;
}

// pairReuseTrack with labels in use (true: done): the lane's tracked nodes, each into its
// group's keys; the main body's displacements into words 0..5 as before.
bool piecesTrack(uint nodeOff, uint nodeCount, uint lane)
{
    if (gP.pieceBase == 0u || !pairReuseOn() || nodeCount == 0u) return false;
    // Retired debris (no contact reads it) stays out of the bounds: this path skips it.
    if (!piecesOn(gStatic[gP.nodeInfoBase + nodeOff].w) && gP.retireBase == 0u) return false;
    if (lane >= nodeCount) return true;
    bool valid = gsCount[GRID_BUCKETS + RU_VALID].load() != 0u;
    float3 dHi = float3(-3.0e38, -3.0e38, -3.0e38);
    float3 dLo = dHi;
    float grow = -3.0e38;
    float size = 0.0;
    float vall = 0.0;
    uint bad = 0u;
    for (uint i = lane, iStop = EB_END(21, i, nodeCount, LANES); i < iStop; i += LANES)
    {
        if ((gStatic[gP.nodeInfoBase + nodeOff + i].x & NODE_PAIR_TRACKED) == 0u) continue;
        uint gi = nodeOff + i;
        if (gP.retireBase != 0u && nodeRetired(gi)) continue;
        uint g = pieceLabel(gi);
        float3 P = gNodes[gi * 3u].xyz;
        float3 V = gNodes[gi * 3u + 1u].xyz;
        float sp = ssqrt(sdot(V, V));
        vall = max(vall, sp);
        for (uint x = 0u; x < 3u; x++)
        {
            pieceKeyMax(RU_E + g * 6u + x, P[x]);
            pieceKeyMax(RU_E + g * 6u + 3u + x, -P[x]);
        }
        if ((asuint(sp) & 0x7F800000u) == 0x7F800000u) bad = 1u;
        if (!valid) continue;
        float4 B = gBeams[gP.pairBuild + gi];
        float3 d = P - B.xyz;
        float gr = sp - B.w;
        uint3 db = asuint(d) & 0x7F800000u;
        uint3 pb = asuint(P) & 0x7F800000u;
        if (any(db == 0x7F800000u) || any(pb == 0x7F800000u) || (asuint(gr) & 0x7F800000u) == 0x7F800000u) bad = 1u;
        grow = max(grow, gr);
        size = max(size, max(max(abs(P.x), abs(P.y)), abs(P.z)));
        if (g == 0u)
        {
            dHi = max(dHi, d);
            dLo = max(dLo, -d);
            continue;
        }
        if (g < PIECE_LOOSE)
        {
            for (uint x = 0u; x < 3u; x++)
            {
                pieceKeyMax(RU_D + (g - 1u) * 6u + x, d[x]);
                pieceKeyMax(RU_D + (g - 1u) * 6u + 3u + x, -d[x]);
            }
        }
        // A piece's contacter against the other groups' triangles.
        for (uint a = 0u; a < PIECE_LOOSE; a++)
        {
            if (a == g) continue;
            float s = asfloat(gNext[pieceSigmaAt(gi, a)]);
            if (!(s < 3.0e38)) continue;
            for (uint x = 0u; x < 3u; x++)
            {
                pieceKeyMax(RU_K + a * 6u + x, d[x] - s);
                pieceKeyMax(RU_K + a * 6u + 3u + x, -d[x] - s);
            }
        }
    }
    pieceKeyMax(RU_VALL, vall);
    if (!valid) return true;
    uint keys[8] = { pairReuseKey(dHi.x), pairReuseKey(dHi.y), pairReuseKey(dHi.z), pairReuseKey(dLo.x),
                     pairReuseKey(dLo.y), pairReuseKey(dLo.z), pairReuseKey(grow), pairReuseKey(size) };
    if (bad != 0u) keys[7] = 0xFFFFFFFFu;
    for (uint k = 0u; k < 8u; k++)
    {
        if (keys[k] > gsCount[GRID_BUCKETS + k].load()) gsCount[GRID_BUCKETS + k].max(keys[k]);
    }
    return true;
}

// pairReuseDecide's test with labels in use (lane 0; true when off): the pieces' own
// spreads and the cross pairs' separations hold. grow: the boxes' growth h · (2 G + dV).
bool piecesAllow(uint v, float grow, float eps)
{
    if (!piecesOn(v)) return true;
    float M = asfloat(gP.pairReuse);
    bool ok = true;
    // A group's displacement range (lo, hi) per axis: the main body's words 0..5.
    for (uint a = 0u; a < PIECE_LOOSE; a++)
    {
        uint w = a == 0u ? 0u : RU_D + (a - 1u) * 6u;
        bool hasD = pieceKeysSet(w);
        float3 hi = hasD ? float3(pieceKey(w), pieceKey(w + 1u), pieceKey(w + 2u)) : float3(0.0, 0.0, 0.0);
        float3 lo = hasD ? -float3(pieceKey(w + 3u), pieceKey(w + 4u), pieceKey(w + 5u)) : float3(0.0, 0.0, 0.0);
        if (a > 0u && hasD)
        {
            float3 r = hi - lo;
            ok = ok && max(max(r.x, r.y), r.z) + grow + eps <= M;
        }
        // Other groups' contacters against a's triangles.
        if (pieceKeysSet(RU_K + a * 6u))
        {
            ok = ok && hasD;
            for (uint x = 0u; x < 3u; x++)
            {
                ok = ok && pieceKey(RU_K + a * 6u + x) + grow + 2.0 * eps < lo[x];
                ok = ok && pieceKey(RU_K + a * 6u + 3u + x) + grow + 2.0 * eps < -hi[x];
            }
        }
        // The main body's contacters against a piece's triangles.
        if (a > 0u && gsCount[GRID_BUCKETS + RU_S + a].load() != 0u)
        {
            float s = -pairReuseVal(gsCount[GRID_BUCKETS + RU_S + a].load());
            bool has0 = pieceKeysSet(0u);
            ok = ok && hasD && has0;
            float3 hi0 = float3(pieceKey(0u), pieceKey(1u), pieceKey(2u));
            float3 lo0 = -float3(pieceKey(3u), pieceKey(4u), pieceKey(5u));
            float3 m = max(hi0 - lo, hi - lo0);
            ok = ok && max(max(m.x, m.y), m.z) + grow + 2.0 * eps < s;
        }
    }
    return ok;
}

// pairReuseDecide's clears with labels in use (lane 0): the step's keys; the build's
// inputs unless this step builds (then its outputs).
void piecesClear(uint v, uint mode)
{
    // Nothing written this frame without labels or retirement (pairReuseBegin cleared it).
    if (gP.pieceBase == 0u || (!piecesOn(v) && gP.retireBase == 0u)) return;
    for (uint k = RU_D; k < RU_E; k++) gsCount[GRID_BUCKETS + k].store(0u);
    for (uint k = RU_K; k < RU_S; k++) gsCount[GRID_BUCKETS + k].store(0u);
    if (mode == 1u)
    {
        for (uint k = RU_S; k < RU_VALL; k++) gsCount[GRID_BUCKETS + k].store(0u);
        gsCount[GRID_BUCKETS + RU_VOL].store(0u);
    }
    else
        piecesClearBuild();
}
void piecesClearBuild()
{
    for (uint k = RU_E; k < RU_K; k++) gsCount[GRID_BUCKETS + k].store(0u);
    gsCount[GRID_BUCKETS + RU_VALL].store(0u);
}

// The Chebyshev gap from p to the box of group a's keys (+inf: no such group).
float pieceGap(float3 p, uint a)
{
    uint w = RU_E + a * 6u;
    if (!pieceKeysSet(w)) return 3.4e38;
    float3 hi = float3(pieceKey(w), pieceKey(w + 1u), pieceKey(w + 2u));
    float3 lo = -float3(pieceKey(w + 3u), pieceKey(w + 4u), pieceKey(w + 5u));
    float3 out = max(lo - p, p - hi);
    return max(max(out.x, out.y), out.z);
}

// pairReuseBuild with labels in use (true: done): the volatile triangles listed instead of
// their pairs; each contacter's separations from the other groups (a piece's per node, the
// main body's least per group); then the build records as pairReuseBuild writes them.
bool piecesBuild(uint v, uint lane, uint triOff, uint triCount, uint nodeOff, uint nodeCount, float3 origin, float r,
                 float cell, uint candOff, uint candCount, uint parity)
{
    if (!piecesOn(v)) return false;
    float margin = asfloat(gP.pairReuse);
    uint at = triPairsAt(v);
    for (uint p = lane, pStop = EB_END(22, p, triCount * LIST_SUBS, LANES); p < pStop; p += LANES)
    {
        uint t = triOff + p / LIST_SUBS;
        if (pieceVolatile(t))
        {
            // A dead triangle (a break beam broken before this beam phase: dead from now
            // on, tryContact's box empty) or one with a retired node has no contact to
            // test: not listed.
            bool gone = ((gMisc[gP.vehicles + t].z & TRI_BREAKABLE) != 0u && triangleDead(t))
                || (gP.retireBase != 0u && pieceTriRetired(t));
            if (p % LIST_SUBS == 0u && !gone) gNext[pieceVolAt(triOff + gsCount[GRID_BUCKETS + RU_VOL].add(1u))] = t;
            continue;
        }
        pairReuseList(t, p % LIST_SUBS, origin, r, cell, candOff, candCount, parity, at, margin);
    }
    // A bound on every triangle's box padding (triReach; rounding allowance).
    bool anti = (gStatic[v * 4u + 3u].x & FLAG_ANTI_CLIP) != 0u;
    float vall = gsCount[GRID_BUCKETS + RU_VALL].load() != 0u ? pieceKey(RU_VALL) : 3.4e38;
    float pb = anti ? r + (vall + clipVmax(v, parity)) * STEP + CLIP_DEPTH : r;
    pb = pb + 1.0e-4 + 1.0e-5 * pb;
    for (uint c = lane, cStop = EB_END(23, c, candCount, LANES); c < cStop; c += LANES)
    {
        uint h = gLists[candOff + c];
        uint g = pieceLabel(h);
        float3 p = gNodes[h * 3u].xyz;
        for (uint a = 0u; a < PIECE_LOOSE; a++)
        {
            if (a == g) continue;
            // Absent from the list, a pair was also at least M outside its box.
            float s = pieceGap(p, a);
            s = s < 3.0e38 ? max(s - pb, margin) : 3.4e38;
            if (g == 0u)
            {
                if (s < 3.0e38) pieceKeyMax(RU_S + a, -s);
            }
            else
                gNext[pieceSigmaAt(h, a)] = asuint(s);
        }
    }
    for (uint i = lane, iStop = EB_END(24, i, nodeCount, LANES); i < iStop; i += LANES)
    {
        float3 P = gNodes[(nodeOff + i) * 3u].xyz;
        float3 V = gNodes[(nodeOff + i) * 3u + 1u].xyz;
        gBeams[gP.pairBuild + nodeOff + i] = float4(P, ssqrt(sdot(V, V)));
    }
    if (lane == 0u)
        gsCount[GRID_BUCKETS + RU_VMAX].store(asuint(anti ? clipVmax(v, parity) : 0.0));
    return true;
}

// pairReuseTests' volatile triangles (labels in use): every candidate of each, as the walk
// tests it (pairReuseTest), PIECE_SUBS lanes a triangle.
// The volatile triangles' tests' count (0: none) and lanes a triangle (piecesTests's,
// for a caller with its own loop of triangleCandidates).
uint piecesBegin(uint v, uint lane, out uint subs)
{
    subs = PIECE_SUBS;
    if (!piecesOn(v)) return 0u;
#ifdef EB_ABLATE
    // Measurement only (time): 2^28 the volatile triangles' tests (also the surface
    // pass's, which this scene does not run).
    if ((gP.ablate & 268435456u) != 0u) return 0u;
#endif
    uint vol = gsCount[GRID_BUCKETS + RU_VOL].load();
    if (lane == 0u) gsCount[GRID_BUCKETS + RU_VSUM].add(vol);
    // Few triangles: more lanes each (the same tests, spread wider).
    subs = max(PIECE_SUBS, min(64u, LANES / max(vol, 1u)));
    return vol;
}
void piecesTests(uint v, uint lane, uint triOff, float3 origin, float r, float cell, uint candOff, uint candCount,
                 uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    uint subs;
    uint vol = piecesBegin(v, lane, subs);
    for (uint p = lane, pStop = EB_END(25, p, vol * subs, LANES); p < pStop; p += LANES)
        piecesTestTriangle(gNext[pieceVolAt(triOff + p / subs)], p % subs, subs, origin, r, cell, candOff, candCount,
                           beamOff, groupOff, groupCount, parity);
}

// pairReuseTest for triangle gt against candidates sub, sub + subs, …: its records loaded
// once, then each candidate as pairReuseTest takes it (the walk's box and cells).
void piecesTestTriangle(uint gt, uint sub, uint subs, float3 origin, float r, float cell, uint candOff, uint candCount,
                        uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    triangleCandidates(gt, sub, subs, origin, r, cell, candOff, candCount, beamOff, groupOff, groupCount, parity, 2u);
}

// After the node phase (its last writes to positions and velocities: the lane's own
// nodes, the centred correction's on the same lanes), before its barrier: the lane's
// tracked nodes against their build records.
void pairReuseTrack(uint nodeOff, uint nodeCount, uint lane)
{
    if (piecesTrack(nodeOff, nodeCount, lane)) return;
    if (!pairReuseOn() || gsCount[GRID_BUCKETS + RU_VALID].load() == 0u || lane >= nodeCount) return;
    float3 dHi = float3(-3.0e38, -3.0e38, -3.0e38);
    float3 dLo = dHi;
    float grow = -3.0e38;
    float size = 0.0;
    uint bad = 0u;
    for (uint i = lane, iStop = EB_END(28, i, nodeCount, LANES); i < iStop; i += LANES)
    {
        if ((gStatic[gP.nodeInfoBase + nodeOff + i].x & NODE_PAIR_TRACKED) == 0u) continue;
        float4 B = gBeams[gP.pairBuild + nodeOff + i];
        float3 P = gNodes[(nodeOff + i) * 3u].xyz;
        float3 V = gNodes[(nodeOff + i) * 3u + 1u].xyz;
        float3 d = P - B.xyz;
        float g = ssqrt(sdot(V, V)) - B.w;
        // Non-finite (bits, whatever the compiler assumes of NaN): build again.
        uint3 db = asuint(d) & 0x7F800000u;
        uint3 pb = asuint(P) & 0x7F800000u;
        if (any(db == 0x7F800000u) || any(pb == 0x7F800000u) || (asuint(g) & 0x7F800000u) == 0x7F800000u) bad = 1u;
        dHi = max(dHi, d);
        dLo = max(dLo, -d);
        grow = max(grow, g);
        size = max(size, max(max(abs(P.x), abs(P.y)), abs(P.z)));
    }
    uint keys[8] = { pairReuseKey(dHi.x), pairReuseKey(dHi.y), pairReuseKey(dHi.z), pairReuseKey(dLo.x),
                     pairReuseKey(dLo.y), pairReuseKey(dLo.z), pairReuseKey(grow), pairReuseKey(size) };
    if (bad != 0u) keys[7] = 0xFFFFFFFFu;
    for (uint k = 0u; k < 8u; k++)
    {
        if (keys[k] > gsCount[GRID_BUCKETS + k].load()) gsCount[GRID_BUCKETS + k].max(keys[k]);
    }
}

#ifdef EB_ABLATE
// Measurement only (2^29, nothing skipped): why a grid step did not reuse the list, into
// two records per vehicle after the pieces' statistics: (no list, main spread, main
// spread with the growth, the pieces), (backoff walks, non-finite, reused, spread sum in
// 0.1 mm). first: 1 no list, 2 non-finite; R the main spread with eps, need the whole.
void reuseCause(uint v, uint mode, float R, float need, uint first)
{
    if ((gP.ablate & 536870912u) == 0u) return;
    uint at = gP.pairBuild + gP.totalNodes + 2u * gP.vehicles + 2u * v;
    uint4 a = asuint(gBeams[at]);
    uint4 b = asuint(gBeams[at + 1u]);
    float M = asfloat(gP.pairReuse);
    if (first == 1u) a.x += 1u;
    else if (first == 2u) b.y += 1u;
    else if (mode == 2u) b.z += 1u;
    else if (R > M) a.y += 1u;
    else if (need > M) a.z += 1u;
    else a.w += 1u;
    if (first == 0u) b.w += uint(min(R, 100.0) * 1.0e4);
    gBeams[at] = asfloat(a);
    gBeams[at + 1u] = asfloat(b);
}
void reuseCauseWalk(uint v)
{
    if ((gP.ablate & 536870912u) == 0u) return;
    uint at = gP.pairBuild + gP.totalNodes + 2u * gP.vehicles + 2u * v + 1u;
    uint4 b = asuint(gBeams[at]);
    b.x += 1u;
    gBeams[at] = asfloat(b);
}
#endif

// Lane 0, after the node phase's barrier, the grid decided (grid): this step's path, the
// list's mode (0 the walk, 1 build, 2 reuse) for gsUniformLanes' bits 1..2.
uint pairReuseDecide(uint v, uint grid, uint parity, float3 origin)
{
    if (!pairReuseOn()) return 0u;
    uint mode = 0u;
    uint streak = gsCount[GRID_BUCKETS + RU_STREAK].load();
    if (grid != 0u && (streak & REUSE_BACKOFF_FLAG) != 0u)
    {
        // Backing off (a list was invalidated before its first reuse, as when falling
        // debris spreads the tracked nodes every step): the walk (mode 0, the same
        // results) for REUSE_BACKOFF grid steps, then a list again.
        uint left = (streak & ~REUSE_BACKOFF_FLAG) - 1u;
#ifdef EB_ABLATE
        reuseCauseWalk(v);
#endif
        gsCount[GRID_BUCKETS + RU_STREAK].store(left > 0u ? (REUSE_BACKOFF_FLAG | left) : 0u);
    }
    else if (grid != 0u)
    {
        mode = 1u;
        bool finite = gsCount[GRID_BUCKETS + 7].load() < 0xFF800000u;
        for (uint k = 0u; k < 7u; k++) finite = finite && gsCount[GRID_BUCKETS + k].load() < 0xFF800000u;
        if (gsCount[GRID_BUCKETS + RU_VALID].load() != 0u && finite)
        {
            float rx = pairReuseVal(gsCount[GRID_BUCKETS + 0].load()) + pairReuseVal(gsCount[GRID_BUCKETS + 3].load());
            float ry = pairReuseVal(gsCount[GRID_BUCKETS + 1].load()) + pairReuseVal(gsCount[GRID_BUCKETS + 4].load());
            float rz = pairReuseVal(gsCount[GRID_BUCKETS + 2].load()) + pairReuseVal(gsCount[GRID_BUCKETS + 5].load());
            float grow = max(pairReuseVal(gsCount[GRID_BUCKETS + 6].load()), 0.0);
            bool anti = (gStatic[v * 4u + 3u].x & FLAG_ANTI_CLIP) != 0u;
            float dv = anti ? max(clipVmax(v, parity) - asfloat(gsCount[GRID_BUCKETS + RU_VMAX].load()), 0.0) : 0.0;
            float size = max(max(abs(origin.x), abs(origin.y)), abs(origin.z)) + pairReuseVal(gsCount[GRID_BUCKETS + 7].load());
            float eps = 1.0e-4 + 4.0e-6 * (size + 1.0);
            float need = max(max(rx, ry), rz) + STEP * (2.0 * grow + dv) + eps;
            if (need <= asfloat(gP.pairReuse) && piecesAllow(v, STEP * (2.0 * grow + dv), eps)) mode = 2u;
#ifdef EB_ABLATE
            reuseCause(v, mode, max(max(rx, ry), rz) + eps, need, 0u);
#endif
        }
#ifdef EB_ABLATE
        else
            reuseCause(v, 1u, 0.0, 0.0, gsCount[GRID_BUCKETS + RU_VALID].load() != 0u ? 2u : 1u);
#endif
        if (mode == 2u)
            gsCount[GRID_BUCKETS + RU_STREAK].add(1u);
        else if (gsCount[GRID_BUCKETS + RU_VALID].load() != 0u && streak < gP.pairStreak)
        {
            // The list was reused on too few steps to pay for its build: back off.
            mode = 0u;
            gsCount[GRID_BUCKETS + RU_VALID].store(0u);
            gsCount[GRID_BUCKETS + RU_STREAK].store(REUSE_BACKOFF_FLAG | REUSE_BACKOFF);
        }
        else
            gsCount[GRID_BUCKETS + RU_STREAK].store(0u);
    }
    for (uint k = 0u; k < 8u; k++) gsCount[GRID_BUCKETS + k].store(0u);
    piecesClear(v, mode);
    if (mode == 2u) gsCount[GRID_BUCKETS + RU_STATS + 1u].add(1u);
    return mode;
}

// Triangle gt (every one, waiting or not) against the contacters in the cells of its box
// padded by `margin` more, the walk's list mode (triangleGridWalk) on that box: the
// pairs inside it into the list at listAt. With FLAG_OWN_TRAVEL each contacter must also
// be inside the box ownTravelOut tests, padded by `margin` more.
void pairReuseList(uint gt, uint sub, float3 origin, float r, float cell, uint candOff, uint candCount,
                   uint parity, uint listAt, float margin)
{
    uint4 tn = gStatic[gP.triBase + gt];
    float3 p0 = nf3(origin + gNodes[tn.x * 3u].xyz);
    float3 p1 = nf3(origin + gNodes[tn.y * 3u].xyz);
    float3 p2 = nf3(origin + gNodes[tn.z * 3u].xyz);
    uint flags = gStatic[tn.w * 4u + 3u].x;
    bool anti = (flags & FLAG_ANTI_CLIP) != 0u;
    bool own = anti && (flags & FLAG_OWN_TRAVEL) != 0u;
    float pad = (anti ? triReach(tn, r, parity) : r) + margin;
    float3 mn = min(min(p0, p1), p2);
    float3 mx = max(max(p0, p1), p2);
    float3 lo = mn - float3(pad, pad, pad);
    float3 hi = mx + float3(pad, pad, pad);
    float tv = own ? clipTriSpeed(tn) : 0.0;
    float vm = own ? clipVmax(tn.w, parity) : 0.0;
    float3 f0 = floor((lo - origin) / cell);
    float3 fs = floor((hi - origin) / cell) - f0 + float3(1.0, 1.0, 1.0);
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
    for (uint i = sub, iStop = EB_END(29, i, outer, LIST_SUBS); i < iStop; i += LIST_SUBS)
    {
        uint e0 = 0u;
        uint e1 = 1u;
        if (!every)
        {
            uint b = gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z))));
            bool dup = false;
            int3 cj = int3(0, 0, 0);
            for (uint j = 0u, jStop = EB_END(30, j, i, 1u); j < jStop; j++)
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
            if (dup) continue;
            e0 = gsOff[b];
            e1 = gsOff[b + 1u];
        }
        for (uint e = e0, eGuard = 0u; e < e1; e++)
        {
            EB_PASS(31, eGuard);
            uint h;
            float3 p;
            if (every)
            {
                h = gLists[candOff + i];
                p = nf3(origin + gNodes[h * 3u].xyz);
            }
            else
            {
                float4 E = gBeams[gP.gridBase + candOff + e];
                p = origin + E.xyz;
                h = asuint(E.w);
            }
            if (h == tn.x || h == tn.y || h == tn.z) continue;
            if (any(p < lo) || any(p > hi)) continue;
            if (own)
            {
                float3 V = gNodes[h * 3u + 1u].xyz;
                float vh = ssqrt(sdot(V, V));
                float sp = vh < vm ? vh : vm;
                float q = nf(nf(r + smul(nf(tv + sp), STEP)) + CLIP_DEPTH) + margin;
                if (any(p < mn - float3(q, q, q)) || any(p > mx + float3(q, q, q))) continue;
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

// A building step's triangle phase (uniform): every triangle's list, then every node's
// build record (positions and velocities are the step's: the node phase is over).
void pairReuseBuild(uint v, uint lane, uint triOff, uint triCount, uint nodeOff, uint nodeCount, float3 origin,
                    float r, float cell, uint candOff, uint candCount, uint parity)
{
    if (piecesBuild(v, lane, triOff, triCount, nodeOff, nodeCount, origin, r, cell, candOff, candCount, parity)) return;
    float margin = asfloat(gP.pairReuse);
    uint at = triPairsAt(v);
    for (uint p = lane, pStop = EB_END(32, p, triCount * LIST_SUBS, LANES); p < pStop; p += LANES)
        pairReuseList(triOff + p / LIST_SUBS, p % LIST_SUBS, origin, r, cell, candOff, candCount, parity, at, margin);
    for (uint i = lane, iStop = EB_END(33, i, nodeCount, LANES); i < iStop; i += LANES)
    {
        float3 P = gNodes[(nodeOff + i) * 3u].xyz;
        float3 V = gNodes[(nodeOff + i) * 3u + 1u].xyz;
        gBeams[gP.pairBuild + nodeOff + i] = float4(P, ssqrt(sdot(V, V)));
    }
    if (lane == 0u)
    {
        bool anti = (gStatic[v * 4u + 3u].x & FLAG_ANTI_CLIP) != 0u;
        gsCount[GRID_BUCKETS + RU_VMAX].store(asuint(anti ? clipVmax(v, parity) : 0.0));
    }
}

// Listed pair (gt, h) as the walk tests it this step: a waiting triangle skips it; the
// walk's cells and buckets of the triangle's box (or every candidate past
// GRID_MAX_CELLS) and its box test; then tryContact (whose own box test, on the live
// triangle's geometry, comes first).


// The step's tests of the list (uniform; after the build or in its place): a pair a
// lane; a build past the list's room walks with its tests instead.
#include "explicit_beams_pair_reuse_tests.slang"

// Lane 0, the frame's end: the statistics into the vehicle's record.
void pairReuseEnd(uint v, uint lane)
{
    if (!pairReuseOn() || lane != 0u) return;
    uint at = gP.pairBuild + gP.totalNodes + v;
    uint4 s = asuint(gBeams[at]);
    gBeams[at] = asfloat(uint4(s.x + gsCount[GRID_BUCKETS + RU_STATS].load(), s.y + gsCount[GRID_BUCKETS + RU_STATS + 1u].load(),
                               s.z + gsCount[GRID_BUCKETS + RU_STATS + 2u].load(), max(s.w, gsCount[GRID_BUCKETS + RU_STATS + 3u].load())));
    piecesEnd(v);
}
// The volatile triangles tested this frame into the vehicle's second statistics record.
void piecesEnd(uint v)
{
    if (gP.pieceBase == 0u) return;
    uint at = gP.pairBuild + gP.totalNodes + gP.vehicles + v;
    uint4 s = asuint(gBeams[at]);
    gBeams[at] = asfloat(uint4(s.x + gsCount[GRID_BUCKETS + RU_VSUM].load(), s.yzw));
}


// --- Aerodynamics and commands with no input (EB_AERO modules) ---
// engine/docs/explicit-beams-aero-commands-spec.md; CPU reference
// engine_softbody::explicit_aero. The specification's double-precision
// sub-expressions run in strict single precision here (WGSL has no f64), so the GPU
// may differ from the CPU by an ulp in them; the angle of attack's acos is the
// device's.
//
//   gStatic at aeroBase per vehicle: (wingOff, wingCount, cmdOff, cmdCount) (wing and
//            command indices global), (front node global, width bits, node count,
//            0), (first slide node global, slide node count, 0, 0); at wingBase per wing: (FLD, FRD, FLU, FRU), (BLD, BRD, BLU, BRU)
//            (global nodes), (efficacy, break reference, table record, induced side
//            0 none 1 left 2 right), (chain span, chain area, 0, 0) as bits; at
//            cmdBase per command: (beam global, contract first, centre bits, 0 for a
//            self-centering command with no input, else contract key | extend key
//            << 8 | self-centering << 16 | 1 << 17 for a vehicle with triggers),
//            (shorten rate, lengthen rate, max contraction, max extension) bits;
//            a table record holds 3601 (cl, cd, cm, 0) as bits.
//   gBeams   at aeroState per vehicle (D, 0); then per wing (broken, 0, 0, 0),
//            induced drag, front force, back force; then per command (mode, 0, 0, 0).
#ifdef EB_AERO
#include "explicit_beams_aero.slang"
#endif

// --- Vehicle–vehicle contact: records the frame entry uses (the rest of the
// section is at the end of the file) ---

// Per vehicle VV_RECS records in gMisc at vvVeh: its padded frame box (lo, hi),
// the bounds its nodes must stay in this frame (lo, hi), flags (x: too big for
// the vehicle grid, y: grouped this frame, z: its group, w: its self-collision
// throttles reset at the next throttle update), member (x: its index in its
// group's sorted members).
static const uint VV_RECS = 6u;
static const uint VV_P_LO = 0u;
static const uint VV_P_HI = 1u;
static const uint VV_ALLOWED_LO = 2u;
static const uint VV_ALLOWED_HI = 3u;
static const uint VV_FLAGS = 4u;
static const uint VV_MEMBER = 5u;
uint vvRec(uint v, uint r) { return gP.vvVeh + v * VV_RECS + r; }

// Counters in gAcc after the box keys: overlapping pairs, big vehicles, solo
// vehicles, groups, a vehicle outside its bounds (index + 1), pair overflow, no
// convergence of the grouping.
static const uint VV_PAIRS = 0u;
static const uint VV_BIGS = 1u;
static const uint VV_SOLOS = 2u;
static const uint VV_GROUPS = 3u;
static const uint VV_OUTSIDE = 4u;
static const uint VV_OVERFLOW = 5u;
static const uint VV_UNCONVERGED = 6u;
static const uint VV_TOO_BIG = 7u;
// Measurement: candidates the small groups' triangle tests visited (vvVisit), and the
// host's switch of the candidates' grid (vvGridOn: nonzero off).
static const uint VV_VISITS = 8u;
static const uint VV_GRID_OFF = 9u;
static const uint VV_CTL_WORDS = 10u;
uint vvCtl(uint i) { return gP.vvAcc + gP.vehicles * 12u + i; }
void vvVisit(uint n);

// --- Vehicle contact's state by owner vehicle: addresses (contact-groups-design.md §8;
// the passes are explicit_beams_fleet.slang's) ---
// gAcc at vvFleet: per vehicle FL_OWN counters of the pass (zero between passes), per
// vehicle its count of the plan's overlapping pairs, per vehicle GRID_BUCKETS counts of
// its candidates' grid (zero between passes), FL_CTL counters of the fleet.
static const uint FL_OWN = 8u;
static const uint FL_OFFERED = 0u; // its partners' contact points this step
static const uint FL_LISTED = 1u; // its listed triangles (vvTestedAt)
static const uint FL_CANDS = 2u; // its candidates (its part of the run)
static const uint FL_PAIRS = 3u; // its listed (triangle, candidate) pairs (vvTriPairsAt)
static const uint FL_HITS = 4u; // its hits (its part of the hit list)
static const uint FL_RESET = 5u; // a partner approaches fast: its throttles reset
static const uint FL_FIX = 6u; // … and that partner is a later vehicle (vvPassSmall's gsVvFix)
static const uint FL_FAST = 7u; // its fastest candidate's speed (float bits)
static const uint FL_CTL = 16u;
// The overflow word's values past the pair list's 1 (explicit_plan::contact_error).
static const int FL_ERR_RUN = 2;
static const int FL_ERR_HITS = 3;
static const int FL_ERR_SLOTS = 4;
uint flOwn(uint v) { return gP.vvFleet + v * FL_OWN; }
uint flAdjCount(uint v) { return gP.vvFleet + gP.vehicles * FL_OWN + v; }
uint flGridCount(uint v) { return gP.vvFleet + gP.vehicles * (FL_OWN + 1u) + v * 1024u; }
uint flCtl(uint i) { return gP.vvFleet + gP.vehicles * (FL_OWN + 1u + 1024u) + i; }
// gNext at vvFleetNext: the plan's adjacency (per vehicle its first entry, one more for
// the end; then the entries: d = 2 · pair + side, the vehicle that side of the pair, its
// partner the other, d ^ 1), then per d the step's bits (1 partners, 2 a fast approach;
// 4, from the plan: a later copy of a pair the list holds twice, never a partner)
// and the owner's two-body slot at the partner's points (its rank among that vehicle's
// two-body partners of the step); then five words a vehicle plus one (0: its part of the
// fleet-wide run, the prefix of the owners' offered counts; 1..3 the prefixes of the
// listed triangles, the listed pairs and the hits; 4 the prefix of the vehicles' contact
// points, static, from the host); then six words a vehicle (0: its
// group's step count at the frame's start; 1..4: its last pass's pairs, hits, listed
// triangles and candidates, a measurement; 5: its level); the fleet's listed triangles;
// FL_CTL counters.
uint flAdjFirst(uint v) { return gP.vvFleetNext + v; }
uint flAdjAt(uint e) { return gP.vvFleetNext + gP.vehicles + 1u + e; }
uint flBitsAt(uint d) { return flAdjAt(0u) + 2u * gP.vvPairCap + d; }
uint flSlotAt(uint d) { return flAdjAt(0u) + 4u * gP.vvPairCap + d; }
uint flPrefixAt(uint r, uint v) { return flAdjAt(0u) + 6u * gP.vvPairCap + r * (gP.vehicles + 1u) + v; }
uint flVehAt(uint r, uint v) { return flPrefixAt(5u, 0u) + r * gP.vehicles + v; }
uint flListAt(uint i) { return flVehAt(6u, 0u) + i; }
uint flCtlNext(uint i) { return flListAt(gP.totalTris) + i; }
// A pair's side d: its vehicle and the other.
uint flOwner(uint d) { return gNext[gP.vvPairs + d]; }
uint flPartner(uint d) { return gNext[gP.vvPairs + (d ^ 1u)]; }
// gBeams at vvFleetBox: per owner vehicle 4 records of the step.
uint flBoxAt(uint v) { return gP.vvFleetBox + 4u * v; }

