// Node gi of vehicle v at x against every proxy shape (surfPoint's).
uint surfPointProxies(uint v, uint gi, float3 x, float r, uint boxCount, uint proxies, uint parity, bool apply,
                      uint mask, uint found)
{
    float3 tlo = ssub3(x, float3(r, r, r));
    float3 thi = sadd3(x, float3(r, r, r));
    // The top bit of `proxies`: the fast proxies only (the pair path meets the rest).
    bool fastOnly = (proxies >> 31) != 0u;
    uint count = proxies & 0x7FFFFFFFu;
    for (uint k = 0u, kStop = EB_END(525, k, count, 1u); k < kStop; k++)
    {
        uint b = boxCount + k;
        if (apply && (mask & surfMarkBit(b)) == 0u) continue;
        float3 vb;
        bool fast = surfProxyFast(v, k, vb);
        if (!fast && (fastOnly || !surfProxyMay(v, k, tlo, thi))) continue;
#ifdef EB_ABLATE
        if ((gP.ablate & 536870912u) != 0u) continue;
#endif
        uint j;
        FqShape s = surfProxyShape(v, k, j);
        float3 lo;
        float3 hi;
        fqBounds(s, lo, hi);
#ifdef EB_ABLATE
        if ((gP.ablate & 268435456u) != 0u) continue;
#endif
        float3 nn;
        float g;
        bool hit = fqSegNear(x, x, r, lo, hi) && fqPointContact(s, x, r, nn, g);
        float3 shift = float3(0.0, 0.0, 0.0);
        if (!hit && fast)
        {
            FqShape sm;
            hit = surfCastUse(v, j, parity, s, x, x, x, 2u, sscale(ssub3(vb, gNodes[gi * 3u + 1u].xyz), STEP), r, sm, shift)
                  && fqPointContact(sm, x, r, nn, g);
            if (!hit) shift = float3(0.0, 0.0, 0.0);
        }
        if (!hit) continue;
        if (!apply)
        {
            found |= surfMarkBit(b);
            gAcc[gP.surfSplit + gi].add(1);
            surfBodyHit(v, j);
        }
        else
            surfApplyTo(v, uint4(gi, gi, gi, 0u), float3(1.0, 0.0, 0.0), nn, g, r, k, j, ssub3(x, shift));
    }
    return found;
}

// Node gi of vehicle v (point contact in the band) against every box slot, then every
// proxy shape: apply 0 counts its hits, 1 applies them.
void surfPoint(uint v, uint gi, uint mk, float3 origin, float r, uint boxOff, uint boxCount, uint proxies, uint parity,
               bool apply)
{
    uint mask = surfMaskIn(mk, apply);
    if (gP.retireBase != 0u && nodeRetired(gi))
    {
        surfMaskOut(mk, apply, 0u);
        return;
    }
    if ((apply && mask == 0u) || (gStatic[gP.nodeInfoBase + gi].x & NODE_NO_GROUND) != 0u)
    {
        surfMaskOut(mk, apply, 0u);
        return;
    }
    float3 xw = nf3(origin + gNodes[gi * 3u].xyz);
    // Proxy records are vehicle-local (relative to the origin).
    float3 xl = nf3(gNodes[gi * 3u].xyz);
    uint found = 0u;
    for (uint b = 0u, bStop = EB_END(526, b, boxCount, 1u); b < bStop; b++)
    {
        float3 x = xw;
        if (apply && (mask & surfMarkBit(b)) == 0u) continue;
        float3 lo;
        float3 hi;
        FqShape s = surfSlot(boxOff, b, lo, hi);
        if (!fqSegNear(x, x, r, lo, hi)) continue;
        float3 nn;
        float g;
        if (!fqPointContact(s, x, r, nn, g)) continue;
        if (!apply)
        {
            found |= surfMarkBit(b);
            gAcc[gP.surfSplit + gi].add(1);
        }
        else
            surfApply(v, uint4(gi, gi, gi, 0u), float3(1.0, 0.0, 0.0), nn, g, r);
    }
#ifndef EB_NO_COUPLING
    if (proxies != 0u) found = surfPointProxies(v, gi, xl, r, boxCount, proxies, parity, apply, mask, found);
#endif
    surfMaskOut(mk, apply, found);
}

// The casts' gathering (CPU Casts, its first pass), in the snapshot phase: every element
// against every fast proxy it has no discrete contact with, its time of impact into its
// body's earliest. Triangle gt, segment beam gj, node gi.
void surfGatherTriangle(uint v, uint gt, float3 origin, float r, uint proxies, uint parity)
{
    if (gP.retireBase != 0u && surfTriRetired(gt)) return;
    if ((gP.triBreak != 0u && triangleDead(gt)) || (gMisc[gP.vehicles + gt].z & TRI_ANTI_ONLY) != 0u) return;
    uint4 tn = gStatic[gP.triBase + gt];
    float3 x0 = nf3(gNodes[tn.x * 3u].xyz);
    float3 x1 = nf3(gNodes[tn.y * 3u].xyz);
    float3 x2 = nf3(gNodes[tn.z * 3u].xyz);
    for (uint k = 0u, kStop = EB_END(527, k, proxies, 1u); k < kStop; k++)
    {
        float3 vb;
        if (!surfProxyFast(v, k, vb)) continue;
        uint j;
        FqShape s = surfProxyShape(v, k, j);
        float3 lo;
        float3 hi;
        fqBounds(s, lo, hi);
        float3 w;
        float3 nn;
        float g;
        float3 q;
        if (fqNear(x0, x1, x2, r, lo, hi) && !fqSeparated(s, x0, x1, x2, r) && fqContact(s, x0, x1, x2, r, w, nn, g, q)) continue;
        float3 ve = sdiv3(sadd3(sadd3(gNodes[tn.x * 3u + 1u].xyz, gNodes[tn.y * 3u + 1u].xyz), gNodes[tn.z * 3u + 1u].xyz), 3.0);
        surfGatherAt(v, j, parity, surfCastToi(s, x0, x1, x2, 0u, sscale(ssub3(vb, ve), STEP), r));
    }
}

void surfGatherSegment(uint v, uint gj, float3 origin, float r, uint proxies, uint groupOff, uint parity)
{
    if (gP.retireBase != 0u && surfSegRetired(gj)) return;
    if (surfBroken(gj, groupOff, parity)) return;
    uint4 bn = gStatic[gP.beamBase + gj * 3u];
    float3 x0 = nf3(gNodes[bn.x * 3u].xyz);
    float3 x1 = nf3(gNodes[bn.y * 3u].xyz);
    for (uint k = 0u, kStop = EB_END(528, k, proxies, 1u); k < kStop; k++)
    {
        float3 vb;
        if (!surfProxyFast(v, k, vb)) continue;
        uint j;
        FqShape s = surfProxyShape(v, k, j);
        float3 lo;
        float3 hi;
        fqBounds(s, lo, hi);
        float3 w;
        float3 nn;
        float g;
        float3 q;
        bool two;
        float3 w2;
        float3 q2;
        if (fqSegNear(x0, x1, r, lo, hi) && !fqSeparated(s, x0, x1, x1, r) && fqSegContact(s, x0, x1, r, w, nn, g, q, two, w2, q2)) continue;
        float3 ve = sscale(sadd3(gNodes[bn.x * 3u + 1u].xyz, gNodes[bn.y * 3u + 1u].xyz), 0.5);
        surfGatherAt(v, j, parity, surfCastToi(s, x0, x1, x1, 1u, sscale(ssub3(vb, ve), STEP), r));
    }
}

void surfGatherPoint(uint v, uint gi, float3 origin, float r, uint proxies, uint parity)
{
    if (gP.retireBase != 0u && nodeRetired(gi)) return;
    if ((gStatic[gP.nodeInfoBase + gi].x & NODE_NO_GROUND) != 0u) return;
    float3 x = nf3(gNodes[gi * 3u].xyz);
    for (uint k = 0u, kStop = EB_END(529, k, proxies, 1u); k < kStop; k++)
    {
        float3 vb;
        if (!surfProxyFast(v, k, vb)) continue;
        uint j;
        FqShape s = surfProxyShape(v, k, j);
        float3 lo;
        float3 hi;
        fqBounds(s, lo, hi);
        float3 nn;
        float g;
        if (fqSegNear(x, x, r, lo, hi) && fqPointContact(s, x, r, nn, g)) continue;
        surfGatherAt(v, j, parity, surfCastToi(s, x, x, x, 2u, sscale(ssub3(vb, gNodes[gi * 3u + 1u].xyz), STEP), r));
    }
}

// Whether vehicle v has a fast proxy this step (every lane reads its proxies).
bool surfAnyFast(uint v, uint proxies)
{
    for (uint k = 0u, kStop = EB_END(530, k, proxies, 1u); k < kStop; k++)
    {
        float3 vb;
        if (surfProxyFast(v, k, vb)) return true;
    }
    return false;
}

// --- The pair path (lossless): the proxy part of the find and the apply over a list of
// (element, proxy shape) pairs whose boxes meet, spread over the lanes, each hit kept
// for the apply. The same hits as the per-element loops (a pair is listed exactly when
// the loops' fqNear holds; their sphere cull only skips pairs fqNear rejects); every sum
// it feeds is fixed point or an integer count, so their order is free. Used when the
// vehicle has proxies, for those not fast this step (a fast proxy's casts keep the
// per-element loops), when its list holds every pair (else the per-element loops meet
// every proxy, that step).
//   gNext at gMisc[coupleRec() + 1].w (0: no pair path): per vehicle SURF_PAIR_CAP
//   pair words (kind << 30 | shape k << 20 | the element's index in the vehicle), then
//   per vehicle SURF_HIT_CAP hits of SURF_HIT_WORDS words (the pair word; two; the body
//   slot; w, n, g, q, w2, q2), then per vehicle and shape its bounds (lo, hi) this step.
//   gAcc after the casts' earliest times: per vehicle the pairs listed, then per vehicle
//   the hits found this step. More hits than the region holds: the apply recomputes
//   every pair's.
static const uint SURF_PAIR_CAP = 2048u;
static const uint SURF_HIT_CAP = 256u;
static const uint SURF_HIT_WORDS = 19u;
static const uint SURF_TRI = 0u;
static const uint SURF_SEG = 1u;
static const uint SURF_PT = 2u;

uint surfPairBase() { return gMisc[coupleRec() + 1u].w; }
// The coupling header's bit 16: the pair path off (tests and measurement).
bool surfPairsOn() { return surfPairBase() != 0u && (gMisc[coupleRec()].x & 16u) == 0u; }
uint surfPairAt(uint v, uint p) { return surfPairBase() + v * SURF_PAIR_CAP + p; }
uint surfHitAt(uint v, uint h)
{
    return surfPairBase() + gP.vehicles * SURF_PAIR_CAP + (v * SURF_HIT_CAP + h) * SURF_HIT_WORDS;
}
uint surfBoundsAt(uint v, uint k)
{
    return surfPairBase() + gP.vehicles * (SURF_PAIR_CAP + SURF_HIT_CAP * SURF_HIT_WORDS) + (v * cplS() + k) * 6u;
}
uint surfPairCount(uint v) { return surfBodyBase() + 3u * cplB() * gP.vehicles + v; }
uint surfHitCount(uint v) { return surfPairCount(v) + gP.vehicles; }

float3 surfLoad3(uint at) { return asfloat(uint3(gNext[at], gNext[at + 1u], gNext[at + 2u])); }
void surfStore3(uint at, float3 x)
{
    uint3 u = asuint(x);
    gNext[at] = u.x;
    gNext[at + 1u] = u.y;
    gNext[at + 2u] = u.z;
}

// Every lane, the snapshot phase: each proxy shape's bounds this step (as the loops'
// fqBounds of surfProxyShape), and the counts cleared.
void surfPairBounds(uint v, uint lane, uint proxies)
{
    for (uint k = lane, kStop = EB_END(531, k, proxies, LANES); k < kStop; k += LANES)
    {
        uint j;
        FqShape s = surfProxyShape(v, k, j);
        float3 lo;
        float3 hi;
        fqBounds(s, lo, hi);
        // A fast shape (its casts) stays with the per-element loops: bounds no box meets.
        float3 vb;
        if (surfProxyFast(v, k, vb))
        {
            lo = float3(3.0e38, 3.0e38, 3.0e38);
            hi = -lo;
        }
        uint at = surfBoundsAt(v, k);
        surfStore3(at, lo);
        surfStore3(at + 3u, hi);
    }
    if (lane == 0u)
    {
        gAcc[surfPairCount(v)].store(0);
        gAcc[surfHitCount(v)].store(0);
    }
}

// Element (kind, index i in the vehicle) with box tlo..thi (grown by the band) against
// every proxy shape's bounds: each that it meets listed (fqNear's test).
bool surfPairSeparated(uint v, uint k, float3 x0, float3 x1, float3 x2, float r)
{
    uint j;
    FqShape s = surfProxyShape(v, k, j);
    return fqSeparated(s, x0, x1, x2, r);
}

// A triangle's or segment's points (a segment passes x1 twice; a node is not culled,
// sep false) not listed against a shape they are separated from (fqSeparated, lossless:
// its find would find nothing), so a slab on a slope, whose bounds hold the whole
// vehicle, lists the elements along its face only and the list keeps within its room.
void surfPairList(uint v, uint kind, uint i, float3 tlo, float3 thi, uint proxies, bool sep, float3 x0, float3 x1,
                  float3 x2, float r)
{
    for (uint k = 0u, kStop = EB_END(532, k, proxies, 1u); k < kStop; k++)
    {
        uint at = surfBoundsAt(v, k);
        float3 lo = surfLoad3(at);
        float3 hi = surfLoad3(at + 3u);
        if (!(all(tlo <= hi) && all(thi >= lo))) continue;
        if (sep && surfPairSeparated(v, k, x0, x1, x2, r)) continue;
        uint p = uint(gAcc[surfPairCount(v)].add(1));
        if (p < SURF_PAIR_CAP) gNext[surfPairAt(v, p)] = (kind << 30) | (k << 20) | i;
    }
}

// The listing phase: every live element of vehicle v against the shapes' bounds.
void surfPairCull(uint v, uint lane, float3 origin, float r, uint4 r0, uint4 r1, uint4 sr, uint groupOff,
                  uint parity, uint triCount, uint segCount, uint nodeCount, uint proxies)
{
    for (uint t = lane, tStop = EB_END(533, t, triCount, LANES); t < tStop; t += LANES)
    {
        uint gt = r1.x + t;
        if (gP.retireBase != 0u && surfTriRetired(gt)) continue;
        if ((gP.triBreak != 0u && triangleDead(gt)) || (gMisc[gP.vehicles + gt].z & TRI_ANTI_ONLY) != 0u) continue;
        uint4 tn = gStatic[gP.triBase + gt];
        float3 x0 = nf3(gNodes[tn.x * 3u].xyz);
        float3 x1 = nf3(gNodes[tn.y * 3u].xyz);
        float3 x2 = nf3(gNodes[tn.z * 3u].xyz);
        float3 tlo = ssub3(min(min(x0, x1), x2), float3(r, r, r));
        float3 thi = sadd3(max(max(x0, x1), x2), float3(r, r, r));
        surfPairList(v, SURF_TRI, t, tlo, thi, proxies, true, x0, x1, x2, r);
    }
    for (uint j = lane, jStop = EB_END(534, j, segCount, LANES); j < jStop; j += LANES)
    {
        uint gj = gLists[sr.x + j];
        if (gP.retireBase != 0u && surfSegRetired(gj)) continue;
        if (surfBroken(gj, groupOff, parity)) continue;
        uint4 bn = gStatic[gP.beamBase + gj * 3u];
        float3 x0 = nf3(gNodes[bn.x * 3u].xyz);
        float3 x1 = nf3(gNodes[bn.y * 3u].xyz);
        float3 tlo = ssub3(min(x0, x1), float3(r, r, r));
        float3 thi = sadd3(max(x0, x1), float3(r, r, r));
        surfPairList(v, SURF_SEG, j, tlo, thi, proxies, true, x0, x1, x1, r);
    }
    for (uint i = lane, iStop = EB_END(535, i, nodeCount, LANES); i < iStop; i += LANES)
    {
        uint gi = r0.x + i;
        if (gP.retireBase != 0u && nodeRetired(gi)) continue;
        if ((gStatic[gP.nodeInfoBase + gi].x & NODE_NO_GROUND) != 0u) continue;
        float3 x = nf3(gNodes[gi * 3u].xyz);
        float3 tlo = ssub3(min(x, x), float3(r, r, r));
        float3 thi = sadd3(max(x, x), float3(r, r, r));
        surfPairList(v, SURF_PT, i, tlo, thi, proxies, false, x, x, x, r);
    }
}

// A pair word's element as nodes (a triangle's three; a segment's (a, b, a); a node's
// (x, x, x)).
uint4 surfPairNodes(uint code, uint4 r0, uint4 r1, uint4 sr)
{
    uint kind = code >> 30;
    uint i = code & 0xFFFFFu;
    if (kind == SURF_TRI) return gStatic[gP.triBase + r1.x + i];
    if (kind == SURF_SEG)
    {
        uint4 bn = gStatic[gP.beamBase + gLists[sr.x + i] * 3u];
        return uint4(bn.x, bn.y, bn.x, 0u);
    }
    uint gi = r0.x + i;
    return uint4(gi, gi, gi, 0u);
}

// Pair `code`'s narrow phase (the loops' own queries): a hit with its body slot j, its
// weights, normal, gap and point, and a segment's second.
bool surfPairContact(uint v, uint code, float3 origin, float r, uint4 tn, out uint j, out float3 w, out float3 nn,
                     out float g, out float3 q, out bool two, out float3 w2, out float3 q2)
{
    uint kind = code >> 30;
    uint k = (code >> 20) & 1023u;
    float3 x0 = nf3(gNodes[tn.x * 3u].xyz);
    float3 x1 = nf3(gNodes[tn.y * 3u].xyz);
    float3 x2 = nf3(gNodes[tn.z * 3u].xyz);
    FqShape s = surfProxyShape(v, k, j);
    w = float3(1.0, 0.0, 0.0);
    nn = float3(0.0, 0.0, 0.0);
    g = 0.0;
    q = x0;
    two = false;
    w2 = float3(0.0, 0.0, 0.0);
    q2 = float3(0.0, 0.0, 0.0);
    // The separation cull (lossless): the narrow phase would find nothing.
    if (kind != SURF_PT && fqSeparated(s, x0, x1, kind == SURF_TRI ? x2 : x1, r)) return false;
    if (kind == SURF_TRI) return fqContact(s, x0, x1, x2, r, w, nn, g, q);
    if (kind == SURF_SEG) return fqSegContact(s, x0, x1, r, w, nn, g, q, two, w2, q2);
    return fqPointContact(s, x0, r, nn, g);
}

// The find of pair p: its narrow phase, the hit counted (as the loops count it) and kept.
void surfPairFind(uint v, uint p, float3 origin, float r, uint4 r0, uint4 r1, uint4 sr)
{
    uint code = gNext[surfPairAt(v, p)];
    uint4 tn = surfPairNodes(code, r0, r1, sr);
    uint j;
    float3 w;
    float3 nn;
    float g;
    float3 q;
    bool two;
    float3 w2;
    float3 q2;
    if (!surfPairContact(v, code, origin, r, tn, j, w, nn, g, q, two, w2, q2)) return;
    surfPairKeep(v, code, tn, j, w, nn, g, q, two, w2, q2);
}

// A pair's hit counted (as the loops count it) and kept.
void surfPairKeep(uint v, uint code, uint4 tn, uint j, float3 w, float3 nn, float g, float3 q, bool two, float3 w2,
                  float3 q2)
{
    uint kind = code >> 30;
    if (kind == SURF_TRI)
    {
        if (w.x > 0.0) gAcc[gP.surfSplit + tn.x].add(1);
        if (w.y > 0.0) gAcc[gP.surfSplit + tn.y].add(1);
        if (w.z > 0.0) gAcc[gP.surfSplit + tn.z].add(1);
    }
    else if (kind == SURF_SEG)
    {
        int hits = two ? 2 : 1;
        gAcc[gP.surfSplit + tn.x].add(hits);
        gAcc[gP.surfSplit + tn.y].add(hits);
    }
    else
        gAcc[gP.surfSplit + tn.x].add(1);
    surfBodyHit(v, j);
    if (two) surfBodyHit(v, j);
    uint h = uint(gAcc[surfHitCount(v)].add(1));
    if (h >= SURF_HIT_CAP) return;
    uint at = surfHitAt(v, h);
    gNext[at] = code;
    gNext[at + 1u] = two ? 1u : 0u;
    gNext[at + 2u] = j;
    surfStore3(at + 3u, w);
    surfStore3(at + 6u, nn);
    gNext[at + 9u] = asuint(g);
    surfStore3(at + 10u, q);
    surfStore3(at + 13u, w2);
    surfStore3(at + 16u, q2);
}

// The apply of kept hit h, as the loops apply it (a segment's second as well).
void surfHitApply(uint v, uint h, float r, uint4 r0, uint4 r1, uint4 sr)
{
    uint at = surfHitAt(v, h);
    uint code = gNext[at];
    uint4 tn = surfPairNodes(code, r0, r1, sr);
    uint k = (code >> 20) & 1023u;
    uint j = gNext[at + 2u];
    float3 nn = surfLoad3(at + 6u);
    float g = asfloat(gNext[at + 9u]);
    surfApplyTo(v, tn, surfLoad3(at + 3u), nn, g, r, k, j, surfLoad3(at + 10u));
    if (gNext[at + 1u] != 0u) surfApplyTo(v, tn, surfLoad3(at + 13u), nn, g, r, k, j, surfLoad3(at + 16u));
}

// More hits than the region holds: pair p's narrow phase again, and its hit applied.
void surfPairApply(uint v, uint p, float3 origin, float r, uint4 r0, uint4 r1, uint4 sr)
{
    uint code = gNext[surfPairAt(v, p)];
    uint4 tn = surfPairNodes(code, r0, r1, sr);
    uint j;
    float3 w;
    float3 nn;
    float g;
    float3 q;
    bool two;
    float3 w2;
    float3 q2;
    if (!surfPairContact(v, code, origin, r, tn, j, w, nn, g, q, two, w2, q2)) return;
    uint k = (code >> 20) & 1023u;
    surfApplyTo(v, tn, w, nn, g, r, k, j, q);
    if (two) surfApplyTo(v, tn, w2, nn, g, r, k, j, q2);
}

// Whether any vehicle of the upload has surface contact (uniform: a parameter).
// A module built with EB_NO_SURFACE (the host picks it when no vehicle of the upload
// has surface contact, so gP.surfSplit is 0 and the pass would return at once) has
// none of the pass: the browser compiles the kernel without it.
#ifdef EB_NO_SURFACE
uint surfaceOn() { return 0u; }
#else
uint surfaceOn() { return gP.surfSplit != 0u ? 1u : 0u; }
#endif

// Whether the host runs the surface pass as its own pipelines between this module's
// dispatches (explicit_beams_surface_pipes.slang; uniform: a parameter): a self split
// module (built without the pass, EB_NO_SURFACE) run for a fleet with surface contact,
// which a host does only with the pipelines (engine_exec explicit_plan
// FramePlanner::surface_piped). A solo vehicle's step then leaves its self-collision hits
// to a dispatch before the pipelines (explicit_beams_selfsplit_frame.slang
// explicit_beams_solo_rest), and its next step's dispatch skips them (the rest at the
// start of explicitBeamsFrameStep: the hit flag it loaded is that dispatch's and the
// pipelines' prep's).
#if defined(EB_NO_SURFACE) && defined(EB_SELF_SPLIT)
bool surfPiped() { return gP.surfSplit != 0u; }
#else
bool surfPiped() { return false; }
#endif

#if defined(EB_FRAME_SEG) || defined(EB_SURF_PIPE) || defined(EB_SOLO_STEP)
// A frame in segments with the surface pass as its own pipelines
// (explicit_beams_surface_pipes.slang): per vehicle one gMisc record, the wide passes'
// (unused while the frame runs per vehicle, and the wide passes have no surface contact).
uint segRec(uint v) { return gP.wideMisc + 1u + v; }
#endif

// Step's surface pass of vehicle v (uniform `on`), after its self-collision and
// before its slide nodes (parity: the step's): the barrier, the snapshot (and the casts'
// gathering when a proxy is fast, and the proxies' bounds on the pair path), the listing
// of the pair path (its own barrier), the find, the apply. A vehicle without the flag
// reaches the barriers only (all decisions below are uniform). The caller's barrier
// follows. Each heavy function has one call site (the find and the apply share
// them, phase by phase): the Metal and Chrome compilers inline every call.
groupshared uint gsUniformSurf;
void surfaceStep(uint v, uint lane, uint on, uint parity)
{
#if defined(EB_NO_SURFACE) || defined(EB_FRAME_SEG)
    // Compiled out (EB_NO_SURFACE: on is 0 in this module's uploads, the pass returns
    // here; EB_FRAME_SEG: the pass runs as its own pipelines between the segments).
    return;
#else
#ifdef EB_ABLATE
    // Measurement only (time): 2^27 the whole pass and its barriers.
    if ((gP.ablate & 134217728u) != 0u) return;
#endif
    if (on == 0u) return;
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 r3 = gStatic[v * 4u + 3u];
    bool mine = (r3.x & FLAG_SURFACE) != 0u;
    uint nodeCount = mine ? r0.y : 0u;
    uint triCount = mine ? r1.y : 0u;
    float3 origin = asfloat(gMisc[v].xyz);
    float r = selfRangeOf(v, asfloat(r3.z));
    uint4 sr = gStatic[gP.surfSeg + v];
    uint segCount = mine ? sr.y : 0u;
    uint proxies = mine ? surfProxies(v) : 0u;
#ifdef EB_NO_COUPLING
    // No proxies in this module's uploads: no bodies, no casts, no pair path.
    uint bodies = 0u;
    bool anyFast = false;
    bool pairs = false;
#else
    uint bodies = proxies != 0u ? cplVeh(v).x : 0u;
    bool anyFast = proxies != 0u && surfAnyFast(v, proxies);
    // The pair path when the vehicle meets proxies and has its region (measurement:
    // EB_ABLATE's 2^30 turns it off).
    bool pairs = proxies != 0u && surfPairsOn();
#ifdef EB_ABLATE
    if ((gP.ablate & 1073741824u) != 0u) pairs = false;
#endif
#endif
    // Barriers sit under `if (pairs)`: lane 0's flag through workgroup memory
    // (gsUniformSurf, read as workgroupUniformLoad), uniform for Chrome.
    if (lane == 0u) gsUniformSurf = pairs ? 1u : 0u;
    sync();
    pairs = gsUniformSurf != 0u;
    for (uint i = lane, iStop = EB_END(536, i, nodeCount, LANES); i < iStop; i += LANES)
    {
        uint gi = r0.x + i;
        float3 f = beamSum(gNodes[gi * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + gi], r0.z, r2.z, r2.w, parity, false);
        gBeams[gP.surfSnap + gi] = float4(sadd3(f, accForce(gi)), 0.0);
        gAcc[gP.surfSplit + gi].store(0);
    }
#ifndef EB_NO_COUPLING
    for (uint j = lane, jStop = EB_END(537, j, bodies, LANES); j < jStop; j += LANES)
    {
        gAcc[surfBodyBase() + v * cplB() + j].store(0);
        gAcc[surfTminAt(v, j, parity + 1u)].store(0);
    }
    if (anyFast)
    {
        for (uint t = lane, tStop = EB_END(538, t, triCount, LANES); t < tStop; t += LANES)
            surfGatherTriangle(v, r1.x + t, origin, r, proxies, parity);
        for (uint j = lane, jStop = EB_END(539, j, segCount, LANES); j < jStop; j += LANES)
            surfGatherSegment(v, gLists[sr.x + j], origin, r, proxies, r2.z, parity);
        for (uint i = lane, iStop = EB_END(540, i, nodeCount, LANES); i < iStop; i += LANES)
            surfGatherPoint(v, r0.x + i, origin, r, proxies, parity);
    }
    if (pairs) surfPairBounds(v, lane, proxies);
#endif
    sync();
    // The proxies the per-element loops meet: every one, the fast ones only (the top
    // bit) beside the pairs, or none.
    uint px = proxies;
    uint listed = 0u;
#ifndef EB_NO_COUPLING
    if (pairs)
    {
        surfPairCull(v, lane, origin, r, r0, r1, sr, r2.z, parity, triCount, segCount, nodeCount, proxies);
        sync();
        listed = uint(gAcc[surfPairCount(v)].load());
        // A list longer than the region: the per-element loops meet every proxy this step.
        if (listed <= SURF_PAIR_CAP) px = anyFast ? (proxies | 0x80000000u) : 0u;
        else pairs = false;
    }
#endif
    // Without static slots nor proxies for them the per-element loops find nothing.
    bool loops = r1.w != 0u || px != 0u;
    // The find (phase 0), the barrier, the apply (phase 1): the same calls per phase as
    // two passes in a row, from one call site each (the compilers inline every call).
    // The pair path's find and apply stay separate functions (Metal's fast math compiles
    // a function by its whole body: one shared body would shift the find's rounding).
    for (uint phase = 0u; phase < 2u; phase++)
    {
        bool apply = phase != 0u;
        if (loops)
        {
            for (uint t = lane, tStop = EB_END(541, t, triCount, LANES); t < tStop; t += LANES)
                surfTriangle(v, r1.x + t, origin, r, r1.z, r1.w, px, parity, apply);
            for (uint j = lane, jStop = EB_END(542, j, segCount, LANES); j < jStop; j += LANES)
                surfSegment(v, gLists[sr.x + j], sr.z + j, origin, r, r1.z, r1.w, px, r2.z, parity, apply);
            for (uint i = lane, iStop = EB_END(543, i, nodeCount, LANES); i < iStop; i += LANES)
                surfPoint(v, r0.x + i, sr.w + i, origin, r, r1.z, r1.w, px, parity, apply);
        }
#ifndef EB_NO_COUPLING
        if (pairs && !apply)
        {
            for (uint p = lane, pStop = EB_END(544, p, listed, LANES); p < pStop; p += LANES)
                surfPairFind(v, p, origin, r, r0, r1, sr);
        }
        if (pairs && apply)
        {
            uint found = uint(gAcc[surfHitCount(v)].load());
            if (found <= SURF_HIT_CAP)
            {
                for (uint h = lane, hStop = EB_END(545, h, found, LANES); h < hStop; h += LANES)
                    surfHitApply(v, h, r, r0, r1, sr);
            }
            else
            {
                for (uint p = lane, pStop = EB_END(546, p, listed, LANES); p < pStop; p += LANES)
                    surfPairApply(v, p, origin, r, r0, r1, sr);
            }
        }
#endif
        if (!apply) sync();
    }
    if (lane == 0u && mine) gsHit = 1u;
#endif
}
