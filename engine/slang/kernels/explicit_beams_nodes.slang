// --- Debris retired by the host (engine_softbody::explicit_debris::DebrisTtl) ---
// A small piece off the vehicle's main body for its time to live is retired: its nodes
// (a bit each in gMisc at gP.retireBase, set by the host) are no longer simulated. The
// node phase leaves them where they were retired, at rest (nodeRetire), and no contact
// reads them: no self-collision as contacter or triangle node (tryContact), no surface
// contact (the surface pass's elements), no part in the reused pair list's bounds, the
// fastest node's speed or the coupling's bounds (vehicle_coupling.slang cpl_bounds).
// Their own beams are still evaluated (they cannot move them). Off (0): nothing here runs.
bool nodeRetired(uint gi)
{
    uint4 w = gMisc[gP.retireBase + (gi >> 7u)];
    uint k = (gi >> 5u) & 3u;
    uint x = k == 0u ? w.x : (k == 1u ? w.y : (k == 2u ? w.z : w.w));
    return ((x >> (gi & 31u)) & 1u) != 0u;
}
// The node phase's work for a retired node: at rest where it is, no contact sums held.
void nodeRetire(uint gi)
{
    float4 V = gNodes[gi * 3u + 1u];
    gNodes[gi * 3u + 1u] = float4(0.0, 0.0, 0.0, V.w);
    gAcc[gi * 3u].store(0);
    gAcc[gi * 3u + 1u].store(0);
    gAcc[gi * 3u + 2u].store(0);
}
// A contact between a triangle and a node either of which is retired: none.
bool contactRetired(uint4 tn, uint h)
{
    return nodeRetired(h) || nodeRetired(tn.x) || nodeRetired(tn.y) || nodeRetired(tn.z);
}

uint nodeUpdateShapes(uint gi, uint4 info, uint vehicle, uint local, uint step, float3 origin, uint flagsV,
                      float ground, uint boxOff, uint boxCount, uint beamOff, uint groupOff, uint groupCount, bool hits,
                      bool solo);
// Explicit vehicles against rigid bodies (explicit_beams_rigid.slang).
bool cplHas(uint v);
uint nodeUpdateCoupled(uint gi, uint4 info, uint vehicle, uint local, uint step, float3 origin, uint flagsV,
                       float ground, uint boxOff, uint boxCount, uint beamOff, uint groupOff, uint groupCount, bool hits,
                       bool solo);

void nodeUpdate(uint gi, uint4 info, uint vehicle, uint local, uint step, float3 origin, uint flagsV, float ground,
                uint boxOff, uint boxCount, uint beamOff, uint groupOff, uint groupCount, bool hits)
{
    if (gP.retireBase != 0u && nodeRetired(gi))
    {
        nodeRetire(gi);
        return;
    }
    if (cplHas(vehicle))
    {
        nodeUpdateCoupled(gi, info, vehicle, local, step, origin, flagsV, ground, boxOff, boxCount, beamOff, groupOff,
                          groupCount, hits, false);
        return;
    }
    if ((flagsV & (FLAG_SHAPES | FLAG_CFORCE | FLAG_TREAD | FLAG_TWO_BODY | FLAG_CLIP_MEM | FLAG_SHIFT_MEM)) != 0u)
    {
        nodeUpdateShapes(gi, info, vehicle, local, step, origin, flagsV, ground, boxOff, boxCount, beamOff, groupOff,
                         groupCount, hits, false);
        return;
    }
#ifdef EB_AERO
    if ((flagsV & FLAG_FUSEDRAG) != 0u)
    {
        nodeUpdateFuse(gi, info, vehicle, origin, ground, boxOff, boxCount, beamOff, groupOff, groupCount,
                       (step + 1u) & 1u, hits);
        return;
    }
#endif
    float4 P = gNodes[gi * 3u];
    float4 V = gNodes[gi * 3u + 1u];
    // Last step's beams (with its group breaks) and self-collision.
    float selfGroup = gNodes[gi * 3u + 2u].w; // JBeam self-collision group (kept)
    float3 f = beamSum(gNodes[gi * 3u + 2u].xyz, info, beamOff, groupOff, groupCount, (step + 1u) & 1u, true);
    if (hits)
    {
        f = f + accForce(gi);
        gAcc[gi * 3u].store(0);
        gAcc[gi * 3u + 1u].store(0);
        gAcc[gi * 3u + 2u].store(0);
    }
#ifdef EB_AERO
    if ((info.x >> NODE_SLOT_SHIFT) != 0u) f = slideTake(f, info.x >> NODE_SLOT_SHIFT);
#endif
    uint flags = info.x;
    float3 pos = P.xyz;
    float3 vel = V.xyz;
    float mass = P.w;
    if ((flags & NODE_NO_GROUND) == 0u)
    {
        float3 x = origin + pos;
        f = groundBoxContacts(f, vel, mass, x, info, vehicle, ground, boxOff, boxCount, V.w);
    }
    if ((flags & NODE_IMMOVABLE) == 0u)
    {
        vel = sadd3(vel, sscale(sdiv3(f, mass), STEP));
        pos = sadd3(pos, sscale(vel, STEP));
    }
    f = float3(0.0, mass * GRAVITY, 0.0);
    if ((flagsV & FLAG_DRAG) != 0u)
    {
        float s = approxSqrt(dot(vel, vel));
        f += -DRAG * s * vel;
        if ((flagsV & FLAG_TURBULENCE) != 0u)
        {
            float t = DRAG * s * s * TURBULENCE;
            f += t * float3(noise(vehicle, step, local, 0u), noise(vehicle, step, local, 1u),
                            noise(vehicle, step, local, 2u));
        }
    }
    gNodes[gi * 3u] = float4(pos, mass);
    gNodes[gi * 3u + 1u] = float4(vel, V.w);
    gNodes[gi * 3u + 2u] = float4(f, selfGroup);
}
#ifdef EB_VV_SOLO
#ifdef EB_AERO
uint nodeUpdateFuseSolo(uint gi, uint4 info, uint vehicle, float3 origin, float ground, uint boxOff, uint boxCount,
                        uint beamOff, uint groupOff, uint groupCount, uint parity, bool hits);
#endif
// nodeUpdate for the solo frame: also whether the node left the frame bounds.
uint nodeUpdateSolo(uint gi, uint4 info, uint vehicle, uint local, uint step, float3 origin, uint flagsV, float ground,
                uint boxOff, uint boxCount, uint beamOff, uint groupOff, uint groupCount, bool hits)
{
    if (gP.retireBase != 0u && nodeRetired(gi))
    {
        nodeRetire(gi);
        return 0u;
    }
    if (cplHas(vehicle))
    {
        return nodeUpdateCoupled(gi, info, vehicle, local, step, origin, flagsV, ground, boxOff, boxCount, beamOff,
                                 groupOff, groupCount, hits, true);
    }
    if ((flagsV & (FLAG_SHAPES | FLAG_CFORCE | FLAG_TREAD | FLAG_TWO_BODY | FLAG_CLIP_MEM | FLAG_SHIFT_MEM)) != 0u)
    {
        return nodeUpdateShapes(gi, info, vehicle, local, step, origin, flagsV, ground, boxOff, boxCount, beamOff,
                                groupOff, groupCount, hits, true);
    }
#ifdef EB_AERO
    if ((flagsV & FLAG_FUSEDRAG) != 0u)
    {
        return nodeUpdateFuseSolo(gi, info, vehicle, origin, ground, boxOff, boxCount, beamOff, groupOff, groupCount,
                       (step + 1u) & 1u, hits);
    }
#endif
    float4 P = gNodes[gi * 3u];
    // The position the last step left (each step's checked once: the frame's
    // last at its end).
    if (any(abs(P.xyz - gsVvBounds[0]) > gsVvBounds[1])) vvReportOutside(vehicle);
    float4 V = gNodes[gi * 3u + 1u];
    // Last step's beams (with its group breaks) and self-collision.
    float selfGroup = gNodes[gi * 3u + 2u].w; // JBeam self-collision group (kept)
    float3 f = beamSum(gNodes[gi * 3u + 2u].xyz, info, beamOff, groupOff, groupCount, (step + 1u) & 1u, true);
    if (hits)
    {
        f = f + accForce(gi);
        gAcc[gi * 3u].store(0);
        gAcc[gi * 3u + 1u].store(0);
        gAcc[gi * 3u + 2u].store(0);
    }
#ifdef EB_AERO
    if ((info.x >> NODE_SLOT_SHIFT) != 0u) f = slideTake(f, info.x >> NODE_SLOT_SHIFT);
#endif
    uint flags = info.x;
    float3 pos = P.xyz;
    float3 vel = V.xyz;
    float mass = P.w;
    if ((flags & NODE_NO_GROUND) == 0u)
    {
        float3 x = origin + pos;
        f = groundBoxContacts(f, vel, mass, x, info, vehicle, ground, boxOff, boxCount, V.w);
    }
    if ((flags & NODE_IMMOVABLE) == 0u)
    {
        vel = sadd3(vel, sscale(sdiv3(f, mass), STEP));
        pos = sadd3(pos, sscale(vel, STEP));
    }
    f = float3(0.0, mass * GRAVITY, 0.0);
    if ((flagsV & FLAG_DRAG) != 0u)
    {
        float s = approxSqrt(dot(vel, vel));
        f += -DRAG * s * vel;
        if ((flagsV & FLAG_TURBULENCE) != 0u)
        {
            float t = DRAG * s * s * TURBULENCE;
            f += t * float3(noise(vehicle, step, local, 0u), noise(vehicle, step, local, 1u),
                            noise(vehicle, step, local, 2u));
        }
    }
    gNodes[gi * 3u] = float4(pos, mass);
    gNodes[gi * 3u + 1u] = float4(vel, V.w);
    gNodes[gi * 3u + 2u] = float4(f, selfGroup);
    return 0u;
}
#endif

// --- Node grab (Rubble only; CPU reference ExplicitBeams::grab_pass): an external
// spring–damper on one node, set by the host per frame (ExplicitBeamsGpu::set_grab).
// gMisc at grabRec(): (on, global node, k bits, c bits), then (target world xyz
// bits, 0). After a step's node phase (its barrier), one lane of the grabbed
// node's vehicle adds k·(target − x) − c·v to the node's force slot (the restart),
// before anything else adds to it or reads it; strict arithmetic in the CPU's
// order. Off (on = 0) it returns at once and nothing changes.
uint grabRec() { return gP.wideMisc + 1u + gP.vehicles; }
void grabApply(uint nodeOff, uint nodeCount, float3 origin)
{
    uint4 g = gMisc[grabRec()];
    if (g.x == 0u) return;
    uint gi = g.y;
    if (gi < nodeOff || gi - nodeOff >= nodeCount) return;
    float3 target = asfloat(gMisc[grabRec() + 1u].xyz);
    float k = asfloat(g.z);
    float c = asfloat(g.w);
    float3 x = sadd3(origin, gNodes[gi * 3u].xyz);
    float3 vel = gNodes[gi * 3u + 1u].xyz;
    float4 F = gNodes[gi * 3u + 2u];
    float3 pull = ssub3(sscale(ssub3(target, x), k), sscale(vel, c));
    gNodes[gi * 3u + 2u] = float4(sadd3(F.xyz, pull), F.w);
}

// --- static shapes (Rubble only): oriented boxes, vertical cylinders, capsules,
// spheres. A vehicle with any (FLAG_SHAPES) steps its nodes here instead of
// nodeUpdate (whose code stays as it was for every other vehicle). Its box slots
// are its axis-aligned boxes then its shapes: per slot (lo, kind) and (hi, at);
// kind 0 an axis-aligned box (lo, hi), else lo, hi the shape's culling box and
// `at` its parameter records (from gP.boxBase): an oriented box (centre, half.x),
// (axis x, half.y), (axis y, half.z), (axis z, 0); a cylinder (centre, radius),
// (half height, 0, 0, 0); a capsule (a, radius), (b, 0); a sphere (centre, radius).
// Strict arithmetic in the CPU reference's order (`shape_face`).

static const uint SHAPE_OBOX = 1u;
static const uint SHAPE_CYLINDER = 2u;
static const uint SHAPE_CAPSULE = 3u;

// Pushed out along d (+y at the centre) when strictly inside the radius.
float4 roundNormal(float3 d, float radius)
{
    float l = ssqrt(sdot(d, d));
    if (!(l < radius)) return float4(0.0, 0.0, 0.0, 0.0);
    if (l > 0.0) return float4(sdiv3(d, l), 1.0);
    return float4(0.0, 1.0, 0.0, 1.0);
}

// The outward normal of the nearest surface of the shape at records `at` (kind
// `kind`) for x strictly inside it (.w 1), else .w 0.
float4 shapeNormal(uint kind, uint at, float3 x)
{
    float4 p0 = asfloat(gStatic[at]);
    float4 p1 = asfloat(gStatic[at + 1u]);
    if (kind == SHAPE_OBOX)
    {
        float4 p2 = asfloat(gStatic[at + 2u]);
        float4 p3 = asfloat(gStatic[at + 3u]);
        float3 h = float3(p0.w, p1.w, p2.w);
        float3 d = ssub3(x, p0.xyz);
        float3 l = float3(sdot(p1.xyz, d), sdot(p2.xyz, d), sdot(p3.xyz, d));
        if (!(l.x > -h.x && l.x < h.x && l.y > -h.y && l.y < h.y && l.z > -h.z && l.z < h.z))
            return float4(0.0, 0.0, 0.0, 0.0);
        float3 n = -p3.xyz;
        float dist = nf(l.z + h.z);
        float t = nf(h.z - l.z);
        if (t < dist) { dist = t; n = p3.xyz; }
        t = nf(l.x + h.x);
        if (t < dist) { dist = t; n = -p1.xyz; }
        t = nf(h.x - l.x);
        if (t < dist) { dist = t; n = p1.xyz; }
        t = nf(l.y + h.y);
        if (t < dist) { dist = t; n = -p2.xyz; }
        t = nf(h.y - l.y);
        if (t < dist) { dist = t; n = p2.xyz; }
        return float4(n, 1.0);
    }
    if (kind == SHAPE_CYLINDER)
    {
        float radius = p0.w;
        float hh = p1.x;
        float3 d = ssub3(x, p0.xyz);
        float r = ssqrt(nf(smul(d.x, d.x) + smul(d.z, d.z)));
        if (!(r < radius && d.y > -hh && d.y < hh)) return float4(0.0, 0.0, 0.0, 0.0);
        float3 n = float3(1.0, 0.0, 0.0);
        if (r > 0.0) n = float3(sdiv(d.x, r), 0.0, sdiv(d.z, r));
        float dist = nf(radius - r);
        float t = nf(hh - d.y);
        if (t < dist) { dist = t; n = float3(0.0, 1.0, 0.0); }
        t = nf(d.y + hh);
        if (t < dist) n = float3(0.0, -1.0, 0.0);
        return float4(n, 1.0);
    }
    if (kind == SHAPE_CAPSULE)
    {
        float3 ab = ssub3(p1.xyz, p0.xyz);
        float l2 = sdot(ab, ab);
        float t = 0.0;
        if (l2 > 0.0) t = min(max(sdiv(sdot(ssub3(x, p0.xyz), ab), l2), 0.0), 1.0);
        return roundNormal(ssub3(x, sadd3(p0.xyz, sscale(ab, t))), p0.w);
    }
    return roundNormal(ssub3(x, p0.xyz), p0.w);
}

// nodeUpdate (and, `solo`, nodeUpdateSolo; with a fuselage drag, nodeUpdateFuse)
// for a vehicle with static shapes: its box slots tested in order as above.
uint nodeUpdateShapes(uint gi, uint4 info, uint vehicle, uint local, uint step, float3 origin, uint flagsV,
                      float ground, uint boxOff, uint boxCount, uint beamOff, uint groupOff, uint groupCount, bool hits,
                      bool solo)
{
    float4 P = gNodes[gi * 3u];
#ifdef EB_VV_SOLO
    if (solo && any(abs(P.xyz - gsVvBounds[0]) > gsVvBounds[1])) vvReportOutside(vehicle);
#endif
    float4 V = gNodes[gi * 3u + 1u];
    cfRoll(gi, groupOff, flagsV);
    clipRoll(gi, flagsV);
    if (gP.selfSplit != 0u && (flagsV & FLAG_TWO_BODY) != 0u) gAcc[gP.selfSplit + gi].store(0);
    float3 shift = shiftTake(gi, flagsV);
    float lift = 0.0; // §37's ground position step
    float selfGroup = gNodes[gi * 3u + 2u].w; // JBeam self-collision group (kept)
    float3 f = beamSum(gNodes[gi * 3u + 2u].xyz, info, beamOff, groupOff, groupCount, (step + 1u) & 1u, true);
    if (hits)
    {
        f = f + accForce(gi);
        gAcc[gi * 3u].store(0);
        gAcc[gi * 3u + 1u].store(0);
        gAcc[gi * 3u + 2u].store(0);
    }
#ifdef EB_AERO
    if ((info.x >> NODE_SLOT_SHIFT) != 0u) f = slideTake(f, info.x >> NODE_SLOT_SHIFT);
#endif
    uint flags = info.x;
    float3 pos = P.xyz;
    float3 vel = V.xyz;
    float mass = P.w;
    if ((flags & NODE_NO_GROUND) == 0u)
    {
        float3 x = origin + pos;
        if ((flags & NODE_TREAD) != 0u)
        {
            float3 c = treadGround(gi, f, vel, mass, V.w, ground - x.y, ground > x.y);
            if (ground > x.y && any(c != float3(0.0, 0.0, 0.0)))
            {
                cfAdd(gi, c);
                f += c;
            }
        }
        else if (ground > x.y)
        {
            float3 c = groundContact(f, vel, mass, vehicle, ground - x.y, V.w, lift);
            if ((info.x & NODE_FRICTION_LIMIT) != 0u) c = frictionLimit(c, f, vel, mass, float3(0.0, 1.0, 0.0), ground - x.y);
            cfAdd(gi, c);
            f += c;
        }
        for (uint b = 0u, bStop = EB_END(5, b, boxCount, 1u); b < bStop; b++)
        {
            uint4 lo4 = gStatic[gP.boxBase + (boxOff + b) * 2u];
            uint4 hi4 = gStatic[gP.boxBase + (boxOff + b) * 2u + 1u];
            float3 lo = asfloat(lo4.xyz);
            float3 hi = asfloat(hi4.xyz);
            if (x.x > lo.x && x.x < hi.x && x.y > lo.y && x.y < hi.y && x.z > lo.z && x.z < hi.z)
            {
                float4 n = float4(0.0, 0.0, -1.0, 1.0);
                if (lo4.w == 0u)
                {
                    float d = x.z - lo.z;
                    if (hi.z - x.z < d) { d = hi.z - x.z; n.xyz = float3(0.0, 0.0, 1.0); }
                    if (x.x - lo.x < d) { d = x.x - lo.x; n.xyz = float3(-1.0, 0.0, 0.0); }
                    if (hi.x - x.x < d) { d = hi.x - x.x; n.xyz = float3(1.0, 0.0, 0.0); }
                    if (x.y - lo.y < d) { d = x.y - lo.y; n.xyz = float3(0.0, -1.0, 0.0); }
                    if (hi.y - x.y < d) { d = hi.y - x.y; n.xyz = float3(0.0, 1.0, 0.0); }
                }
                else
                    n = shapeNormal(lo4.w, gP.boxBase + hi4.w, x);
                if (n.w != 0.0)
                {
                    float3 c = contactForce(f, vel, mass, n.xyz, surfaceV(boxSurface(), vehicle), 0.0, V.w);
                    cfAdd(gi, c);
                    f += c;
                }
            }
        }
    }
    if ((flags & NODE_IMMOVABLE) == 0u)
    {
        vel = sadd3(vel, sscale(sdiv3(f, mass), STEP));
        pos = sadd3(pos, sscale(vel, STEP));
        pos = takeSteps(pos, shift, lift);
    }
    clipSpeed(vehicle, step, flagsV, vel);
#ifdef EB_AERO
    if ((flagsV & FLAG_FUSEDRAG) != 0u)
    {
        f = sadd3(float3(0.0, smul(mass, GRAVITY), 0.0), gBeams[gP.aeroState + vehicle].xyz);
        gNodes[gi * 3u] = float4(pos, mass);
        gNodes[gi * 3u + 1u] = float4(vel, V.w);
        gNodes[gi * 3u + 2u] = float4(f, selfGroup);
        return 0u;
    }
#endif
    f = float3(0.0, mass * GRAVITY, 0.0);
    if ((flagsV & FLAG_DRAG) != 0u)
    {
        float s = approxSqrt(dot(vel, vel));
        f += -DRAG * s * vel;
        if ((flagsV & FLAG_TURBULENCE) != 0u)
        {
            float t = DRAG * s * s * TURBULENCE;
            f += t * float3(noise(vehicle, step, local, 0u), noise(vehicle, step, local, 1u),
                            noise(vehicle, step, local, 2u));
        }
    }
    gNodes[gi * 3u] = float4(pos, mass);
    gNodes[gi * 3u + 1u] = float4(vel, V.w);
    gNodes[gi * 3u + 2u] = float4(f, selfGroup);
    return 0u;
}

#include "explicit_beams_rigid.slang"

// Self-collision hits are found in parallel and applied in the specification's
// order (triangles in file order, then candidates in node order): a hit's contact
// force reads its node's force with every earlier hit's share added, as the CPU
// reference does, whatever order the lanes find them in. The finding lanes record
// each hit's key (triangle << 16 | node, both vehicle-local) in the vehicle's hit
// list; selfApply then computes the forces in rounds (a hit waits until every
// earlier hit touching its node has its force) and adds them to the fixed-point
// accumulators. Per vehicle (gStatic at vvStatic, .z and .w): HIT_CAP records of
// HIT_WORDS words in gNext (key, alpha, beta, force xyz, round, 0, the triangle's
// n0, n1, n2, 0, then from the finding lane the contact's normal xyz, penetration,
// relative velocity xyz, reduced mass, friction) then PEND_CAP
// pending beams; two gAcc counters (hits, pending beams), cleared by the node
// phase.
static const uint HIT_CAP = 256u;
static const uint HIT_WORDS = 24u;
static const uint PEND_CAP = 64u;
uint hitList(uint v) { return gStatic[gP.vvStatic + v].z; }
uint hitAcc(uint v) { return gStatic[gP.vvStatic + v].w; }
uint pendList(uint v) { return hitList(v) + HIT_CAP * HIT_WORDS; }
// selfApply's node forces (perfFlags bit 1): per listed hit and role (0 the contacter, 1..3
// the triangle's n0, n1, n2) the node's beam force (beamSum), 3 words, after the pending
// beams; filled once a step (hitFbFill) for the round and the residual pass's phases,
// which each summed it again. 0: no record (the sum from the incidence entries).
static const uint PERF_HIT_FB = 1u;
// perfFlags bit 2: the raycast wheels' rays and bars beside the step before's beam phase,
// their tyres beside the coupling's first phase (driveAheadPart): no phase of their own.
static const uint PERF_DRIVE_AHEAD = 2u;
// perfFlags bit 16, set by the host on a wide frame's parameter copies when it runs the
// reused pair list (explicit_plan::PERF_WIDE_LIST): the wide passes leave the triangles'
// throttles, the grid and its counts' words to explicit_beams_wide_list.
static const uint PERF_WIDE_LIST = 16u;
// perfFlags bit 32, with it (explicit_plan::PERF_WIDE_FUSED): no aux dispatch for steps
// 1..: step s + 1's coupling phases and its wheels' rays and bars run on the list kernel's
// lanes beside step s's list phases, its tyres at the end of step s's wide_hits (all of
// them read what step s's node phase and hits left, as the aux dispatch before step
// s + 1's nodes did).
static const uint PERF_WIDE_FUSED = 32u;
// explicit_plan::PERF_AHEAD_APART and PERF_TYRE_AHEAD (wideAheadApart, driveTyresPost).
static const uint PERF_AHEAD_APART = 131072u;
static const uint PERF_TYRE_AHEAD = 262144u;
// perfFlags bit 64 (explicit_plan::PERF_VV_STREAM): the larger groups' contact pass keeps,
// beside each gathered candidate, its travel over the step, its vehicle and whether that
// vehicle has anti-clip (written where the candidate is gathered), so a triangle's
// candidate loop reads two streamed records a candidate in place of the node's record,
// its vehicle's flags and its velocity: the same box test on the same values.
static const uint PERF_VV_STREAM = 64u;
// perfFlags bit 128 (explicit_plan::PERF_VV_BIG_GRID): the larger groups' contact pass
// sorts each owner's candidates into a grid (every bucket, the whole workgroup: one owner
// at a time) and a triangle visits the cells of its padded box, as the small groups'
// pass does (vvGridBuild, vvTriangleCandsGrid), in place of every candidate.
static const uint PERF_VV_BIG_GRID = 128u;
// perfFlags bit 256 (explicit_plan::PERF_MEMBER_LIST): on the split group path a contact
// group's member keeps its self-collision pair list over the frame's steps as a solo
// vehicle does (pairReuse*: built at a step, tested again while it provably holds every
// pair the walk would test), its state saved between the steps' dispatches
// (memberListSave), in place of the grid's sort and walk every step (contactMemberStep).
// With self-collision split (EB_SELF_SPLIT) the list's walk and its pairs' tests run in
// the walk and pairs dispatches across the GPU, beside the solo vehicles', and the
// member's step ends at its tests: its hits are applied in a dispatch of their own
// before the groups' pass (explicit_beams_split.slang memberRest; the host adds it).
static const uint PERF_MEMBER_LIST = 256u;
// perfFlags bit 4096 (explicit_plan::PERF_FLEET_STATE; contact-groups-design.md §8 stage
// 1): on the split group path with its tests across workgroups, vehicle contact's state
// by owner vehicle (explicit_beams_fleet.slang) in place of a contact group's slots.
static const uint PERF_FLEET_STATE = 4096u;
// perfFlags bit 8192 (explicit_plan::PERF_FLEET_FRONT; §8 stage 2), with it: the pass's
// front and tests as fleet-wide dispatches (explicit_beams_fleet.slang fleetFront,
// fleetPlan, fleetWideTests).
static const uint PERF_FLEET_FRONT = 8192u;
// perfFlags bit 16384 (explicit_plan::PERF_FLEET_ANY; §8 stage 3), with both: a contact
// group of any size on the fleet-wide front and tests, its levels for any number of
// owners (fleetLevelsAny), in place of the larger groups' pass (vvPass) past VV_PART.
static const uint PERF_FLEET_ANY = 16384u;
// perfFlags bit 65536 (explicit_plan::PERF_FLEET_POST), with the front fleet-wide: the
// listed triangles' throttle updates a thread each in a dispatch of their own
// (fleetPost), in place of the group's workgroup with its first level.
static const uint PERF_FLEET_POST = 65536u;
// perfFlags bit 32768 (explicit_plan::PERF_FLEET_KEEP; contact-groups-design.md §8.10),
// with PERF_FLEET_FRONT: vehicle contact's (triangle, contact point) candidates kept
// from one substep to the next (explicit_beams_fleet.slang fk*), built with a margin and
// rebuilt when the vehicles have moved past it, in place of the gather, grids, sorted
// copy and walk every substep. The exact tests are the same on the same pairs.
static const uint PERF_FLEET_KEEP = 32768u;
// perfFlags bit 8 (explicit_plan::PERF_KEEP_TRACK_REST; set by the host, never by a
// caller): with PERF_FLEET_KEEP, the fused scans (PERF_FLEET_FUSE) and the members' rest
// dispatches, the kept candidates' tracking of every node in the members' rest
// (explicit_beams_fleet_track.slang fkTrackVehicle), in place of the scans' own dispatch.
static const uint PERF_KEEP_TRACK_REST = 8u;
uint hitFbAt(uint v, uint i) { return pendList(v) + PEND_CAP + i * 12u; }
// perfFlags bit 1024 (explicit_plan::PERF_HIT_ROLES, with PERF_HIT_FB): a two-body
// dissipative vehicle's hits applied a (hit, role) a lane. A hit's first-pass adds, its
// residual's and its total's shares, its four nodes' accelerations and the merge are
// each four independent node updates (fixed-point atomics, order-free); selfApply ran
// them in series on the hit's lane (about 130 atomics a hit over its phases). Here each
// phase that touches a node runs on the (hit, role)'s own lane and each phase that
// combines the four runs on the hit's lane, the values between them in the hit's
// record after the beam forces (hitRoleAt: per role the acceleration xyz and an
// inverse mass, then the total xyz; HIT_ROLE_WORDS words a hit): the same sums of the
// same bits in the same order, more barriers and shorter chains (selfResidualRoles).
static const uint PERF_HIT_ROLES = 1024u;
// perfFlags bit 1048576 (explicit_plan::PERF_HITS_LOCAL, with PERF_HIT_ROLES and
// PERF_HIT_FB): in the wide hits kernel, a two-body dissipative vehicle's step with
// 1..HITS_LOCAL_CAP hits applied from workgroup memory (selfApplyLocal): each (hit,
// role)'s node record and accumulators loaded once, the round and the residual pass's
// seven phases on those copies, written back once. The same arithmetic in the same
// order on the same bits; only where the values wait between phases changes.
static const uint PERF_HITS_LOCAL = 1048576u;
// perfFlags bit 33554432 (explicit_plan::PERF_SELF_HITS_LOCAL, with PERF_HIT_ROLES and
// PERF_HIT_FB): the same on the self-collision split path, where a car's own step is the
// step dispatch's (a solo car: step k - 1's rest at the next step dispatch's start; a
// contact group's member: memberRest): selfApplyLocal in place of selfApply for a
// two-body dissipative car with 1..HITS_LOCAL_CAP hits (selfApplyLocalKeep).
static const uint PERF_SELF_HITS_LOCAL = 33554432u;
// perfFlags bit 134217728 (explicit_plan::PERF_HIT_SLOT_MAP, with PERF_HITS_LOCAL): in the
// wide hits kernel, each node's slot (the first item naming it) found through a map of the
// vehicle's nodes in workgroup memory (an atomic maximum of 256 - q at the node's word)
// instead of each item scanning every item before it: the same slots, so the same sums of
// the same bits in the same order. vehicle-perf-first-principles.md §8.46.
static const uint PERF_HIT_SLOT_MAP = 134217728u;
// perfFlags bit 67108864 (explicit_plan::PERF_SPLIT_WHEELS_AHEAD, with PERF_MEMBER_LIST and
// self-collision split): a contact group member's raycast wheels' rays and bars of step
// s + 1 on a workgroup of their own in step s's rest dispatch (explicit_beams_split.slang
// splitWheelsAhead), its tyres at step s + 1's start as before (splitDrive with `ahead`).
static const uint PERF_SPLIT_WHEELS_AHEAD = 67108864u;
// perfFlags bit 2097152 (explicit_plan::PERF_TYRE_EARLY, with PERF_TYRE_AHEAD): the
// wheels' second part loaded before the hits (explicit_beams_drive.slang drvTyreLoad).
static const uint PERF_TYRE_EARLY = 2097152u;
// perfFlags bit 4194304 (explicit_plan::PERF_WHEELS_APART, with PERF_AHEAD_APART): the
// coupling and the wheels of the list dispatch on workgroups of their own
// (wideCoupleApart, wideWheelsApart).
static const uint PERF_WHEELS_APART = 4194304u;
static const uint HIT_ROLE_WORDS = 20u;
uint hitRoleAt(uint v, uint i) { return hitFbAt(v, HIT_CAP) + i * HIT_ROLE_WORDS; }

// The break exception (spec §2, step 4: an end on a collision triangle with fewer
// than 3 live plain beams saves the beam) reads the beams' state in spawn order: a
// beam sees the breaks of earlier beams in its step and none of later ones. The
// beam phase runs in parallel, so a beam whose test reaches the exception is
// deferred (marked pending: its record's .w = 2, nothing else written) and one lane
// then resolves the vehicle's pending beams in spawn order (resolvePending), each
// counting live beams as the sequential pass would see them. A beam that breaks in a
// beam phase carries .w = 1 until its next beam phase clears it.
static uint gResolving = 0u;

// Live plain beams at `node` as beam `j` (vehicle-local) of this step's beam phase
// sees them: broken before this step, by an earlier beam of this step, or by an
// earlier master's group break of this step (group slot of this parity) is dead.
uint liveAt(uint node, uint j, uint beamOff, uint groupOff, uint parity)
{
    uint4 info = gStatic[gP.nodeInfoBase + node];
    uint live = 0u;
    for (uint k = 0u, kStop = EB_END(6, k, info.z, 1u); k < kStop; k++)
    {
        uint b = gLists[info.y + k] >> 1;
        uint4 bn = gStatic[gP.beamBase + b * 3u];
        if ((bn.z & 0xFFu) != KIND_PLAIN) continue;
        float4 s1 = gBeams[b * 2u + 1u];
        bool dead = s1.z != 0.0 && !(s1.w == 1.0 && b - beamOff > j);
        int g = asint(bn.w);
        if (!dead && g != 0)
        {
            uint master = gGroups[groupSlot(parity, groupOff, uint(abs(g)))].load();
            dead = master != NO_GROUP && master < j;
        }
        if (!dead) live++;
    }
    return live;
}

// The break exception of beam gj (vehicle-local index `local`) whose force passed its
// strength: 0 it breaks, 1 an end saves it, 2 deferred (beam phase).
uint breakException(uint gj, uint4 bn, uint local, uint groupOff, uint parity)
{
    bool ta = (gStatic[gP.nodeInfoBase + bn.x].x & NODE_ON_TRIANGLE) != 0u;
    bool tb = (gStatic[gP.nodeInfoBase + bn.y].x & NODE_ON_TRIANGLE) != 0u;
    if (!ta && !tb) return 0u;
    uint v = bn.z >> 16;
    if (gResolving == 0u)
    {
        uint i = uint(gAcc[hitAcc(v) + 1u].add(1));
        if (i < PEND_CAP) gNext[pendList(v) + i] = local;
        float4 s1 = gBeams[gj * 2u + 1u];
        gBeams[gj * 2u + 1u] = float4(s1.xyz, 2.0);
        return 2u;
    }
    uint beamOff = gStatic[v * 4u].z;
    if (ta && liveAt(bn.x, local, beamOff, groupOff, parity) < 3u) return 1u;
    if (tb && liveAt(bn.y, local, beamOff, groupOff, parity) < 3u) return 1u;
    return 0u;
}

#ifdef EB_AERO
void shock2Update(uint gj, uint local, bool fast, uint groupOff, uint parity, uint4 bn, float4 s1);
void triggerUpdate(uint gj, bool fast, uint parity, uint4 bn);
#endif

// A JBeam torsion element (KIND_JTORSION, the CPU reference's torsion_force): the
// torque -k wrap(phi - phi0) - c dphi/dt on the dihedral angle phi of nodes 1-2-3-4
// about the 2->3 axis, as forces torque * dphi/dx_i on the four nodes (this record's
// incidence entries for nodes 1 and 2, the next record's for 3 and 4). State: phi0 in
// the rest slot, the deform torque in Tc. Static: (k, c, -, -), the watched break
// groups (c2.xyz): it breaks the step after one of them breaks (the last step's
// group slots, parity ^ 1).
// atan2 by the CPU's explicit sequence (explicit_beams.rs exact_atan2,
// jbeam-semantics-spec.md §35): no hardware atan2, bit for bit.
float sAtan2(float y, float x)
{
    float ax = abs(x);
    float ay = abs(y);
    bool swap = ay > ax;
    float lo = swap ? ax : ay;
    float hi = swap ? ay : ax;
    if (hi == 0.0) return 0.0;
    float a = sdiv(lo, hi);
    float t = a;
    float base = 0.0;
    if (a > asfloat(0x3ed413cdu))
    {
        t = sdiv(nf(a - 1.0), nf(a + 1.0));
        base = asfloat(0x3f490fdbu);
    }
    float t2 = smul(t, t);
    // The series coefficients (-1)^k / (2k + 1), k = 11 down to 0 (the CPU's ATAN_C).
    float p = asfloat(0xbd321643u);
    p = nf(smul(p, t2) + asfloat(0x3d430c31u));
    p = nf(smul(p, t2) + asfloat(0xbd579436u));
    p = nf(smul(p, t2) + asfloat(0x3d70f0f1u));
    p = nf(smul(p, t2) + asfloat(0xbd888889u));
    p = nf(smul(p, t2) + asfloat(0x3d9d89d9u));
    p = nf(smul(p, t2) + asfloat(0xbdba2e8cu));
    p = nf(smul(p, t2) + asfloat(0x3de38e39u));
    p = nf(smul(p, t2) + asfloat(0xbe124925u));
    p = nf(smul(p, t2) + asfloat(0x3e4ccccdu));
    p = nf(smul(p, t2) + asfloat(0xbeaaaaabu));
    p = nf(smul(p, t2) + asfloat(0x3f800000u));
    float r = nf(base + smul(t, p));
    if (swap) r = nf(asfloat(0x3fc90fdbu) - r);
    if (x < 0.0) r = nf(asfloat(0x40490fdbu) - r);
    if (y < 0.0) r = -r;
    return r;
}

float wrapAngle(float a)
{
    if (a > 3.14159274) return nf(a - 6.28318548);
    if (a <= -3.14159274) return nf(a + 6.28318548);
    return a;
}

// Beam bn's vehicle has centred damping (FLAG_CENTRED, §39).
bool centredOn(uint4 bn) { return gP.centredDv != 0u && (gStatic[(bn.z >> 16) * 4u + 3u].x & FLAG_CENTRED) != 0u; }
void jtorsionUpdate(uint gj, uint local, uint groupOff, uint parity, uint4 bn, float4 s1, float4 c1, uint4 r2,
                    float3 x1, float3 x2, float3 v1, float3 v2)
{
    uint4 bn2 = gStatic[gP.beamBase + (gj + 1u) * 3u];
    float4 s0 = gBeams[gj * 2u];
    int group = asint(bn.w);
    float3 x3 = gNodes[bn2.x * 3u].xyz;
    float3 x4 = gNodes[bn2.y * 3u].xyz;
    float3 v3 = gNodes[bn2.x * 3u + 1u].xyz;
    float3 v4 = gNodes[bn2.y * 3u + 1u].xyz;
    bool broken = false;
    if (r2.x != 0u && gGroups[groupSlot(parity ^ 1u, groupOff, r2.x)].load() != NO_GROUP) broken = true;
    if (r2.y != 0u && gGroups[groupSlot(parity ^ 1u, groupOff, r2.y)].load() != NO_GROUP) broken = true;
    if (r2.z != 0u && gGroups[groupSlot(parity ^ 1u, groupOff, r2.z)].load() != NO_GROUP) broken = true;
    float rest = s0.x;
    float t = 0.0;
    float3 g1 = float3(0.0, 0.0, 0.0);
    float3 g2 = g1;
    float3 g3 = g1;
    float3 g4 = g1;
    if (!broken)
    {
        float3 b1 = ssub3(x2, x1);
        float3 b2 = ssub3(x3, x2);
        float3 b3 = ssub3(x4, x3);
        float3 n1 = scross(b1, b2);
        float3 n2 = scross(b2, b3);
        float l1 = sdot(n1, n1);
        float l2 = sdot(n2, n2);
        float ll = sdot(b2, b2);
        float lb = ssqrt(ll);
        float phi = sAtan2(smul(lb, sdot(b1, n2)), sdot(n1, n2));
        if (!(l1 < 1e-12 || l2 < 1e-12 || ll < 1e-12))
        {
            g1 = sscale(n1, sdiv(-lb, l1));
            g4 = sscale(n2, sdiv(lb, l2));
            float p = sdiv(-sdot(b1, b2), ll);
            float q = sdiv(-sdot(b3, b2), ll);
            g2 = ssub3(sscale(g1, nf(p - 1.0)), sscale(g4, q));
            g3 = ssub3(sscale(g4, nf(q - 1.0)), sscale(g1, p));
        }
        float rate = nf(nf(nf(sdot(g1, v1) + sdot(g2, v2)) + sdot(g3, v3)) + sdot(g4, v4));
        float d = wrapAngle(nf(phi - rest));
        if (smul(c1.x, abs(d)) > s0.y)
        {
            float m = sdiv(s0.y, c1.x);
            d = d < 0.0 ? -m : m;
            rest = wrapAngle(nf(phi - d));
        }
        t = nf(smul(-c1.x, d) - smul(c1.y, rate));
        if (abs(t) > s1.x) broken = true;
    }
    if (broken)
    {
        t = 0.0;
        if (group > 0)
        {
            gGroups[groupSlot(parity, groupOff, uint(group))].min(local);
            gGroups[groupSlot(parity, groupOff, 0u)].min(local);
        }
        gBeams[gj * 2u + 1u] = float4(s1.x, 0.0, 1.0, 1.0);
        float4 p1 = gBeams[(gj + 1u) * 2u + 1u];
        gBeams[(gj + 1u) * 2u + 1u] = float4(p1.x, 0.0, 1.0, 1.0);
    }
    else if (s1.w != 0.0)
        gBeams[gj * 2u + 1u] = float4(s1.xyz, 0.0);
    if (rest != s0.x) gBeams[gj * 2u] = float4(rest, s0.yzw);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(sscale(g1, t), !broken && centredOn(bn) ? c1.y : 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(sscale(g2, t), 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u]] = float4(sscale(g3, t), 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u + 1u]] = float4(sscale(g4, t), 0.0);
}

// Beam bn's vehicle keeps strength under plastic stretch (FLAG_KEEP_STRENGTH).
bool keepStrength(uint4 bn) { return (gStatic[(bn.z >> 16) * 4u + 3u].x & FLAG_KEEP_STRENGTH) != 0u; }

