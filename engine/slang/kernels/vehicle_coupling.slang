// Explicit vehicles in the World (engine/docs/explicit-vehicles-in-world.md): the
// coupling's plan, its impulses and its chunk loads, between the World's buffers and the
// explicit solver's (ExplicitBeamsGpu built with coupling room; its records:
// explicit_beams_rigid.slang). One standalone WGSL module; host engine-world
// `explicit_vehicles.rs`.
//
//   cpl_bounds     one group per vehicle: its nodes' world box and fastest speed;
//   cpl_plan       one group per vehicle: every eligible collider whose bounding sphere,
//                  swept over the window, meets the vehicle's swept box, sorted by
//                  collider; its bodies in order of first appearance; the proxy records
//                  (window-start pose, velocity, inverse mass and inertia; shapes, hull
//                  planes); each engine body's vehicle count (mass splitting);
//   cpl_share      one group per vehicle: each proxy body's vehicle count;
//   cpl_loads      one group per vehicle, after the frame: each shape's window impulse on
//                  a stress chunk as a contact record of the threshold ingest's slot 1;
//   cpl_loads_end  one thread: that slot's row count;
//   cpl_emit       one group per vehicle, at the next step's start (after the fracture's
//                  split): per proxy body its window impulse, on the bodies its shapes
//                  belong to now (emitPair: whole, or split among its pieces as the
//                  whole body's change of motion), as impulse records after the host's;
//   cpl_emit_end   one thread: IMPULSE_FIRST (the records are applied once).
//
// Limits are loud: a vehicle with more shapes, bodies or planes than its room, more
// impulse records than IMPULSE_CAP or more chunk records than the ingest's rows sets a
// status bit and the host fails the step.

struct Params
{
    uint vehicles;
    uint colliders;
    uint maxBodies;      // the World's body capacity (share counts)
    uint coupleRec;      // gMisc record of the explicit coupling header
    uint B;              // proxy bodies, shapes, planes per vehicle
    uint S;
    uint planes;
    uint recs;           // records per vehicle
    uint miscBase;       // gMisc record of vehicle 0's region
    float window;        // the window's length (s)
    float margin;        // extra reach (m)
    uint keyBase;        // impulse record keys: keyBase + vehicle * 1024 + shape (< 2^24)
    uint impulseBase;    // engine-schema IMPULSE_BASE, IMPULSE_COUNT, IMPULSE_FIRST, IMPULSE_CAP
    uint impulseCount;
    uint impulseFirst;
    uint impulseCap;
    uint ingestRows;     // the threshold ingest's rows per slot (0: no ingest)
    uint ingestSlot;     // its slot for the vehicles' records (1)
    uint step;
    uint filter;         // the vehicles' collision filter: categories | mask << 16
    uint rowCap;         // chunk records at most (the ingest merges this many of slot 1)
    uint pad0, pad1, pad2;
};

static const uint BODY_RECS = 5u;
static const uint SHAPE_RECS = 7u;
static const uint NONE = 0xFFFFFFFFu;
static const uint IMPULSE_LINEAR_ANGULAR = 2u;
static const uint HULL_MAX_BASE = 0x1FFFFFFFu;
// Status bits (work[1]).
static const uint ST_SHAPES = 1u;
static const uint ST_BODIES = 2u;
static const uint ST_PLANES = 4u;
static const uint ST_IMPULSES = 8u;
static const uint ST_ROWS = 16u;
// work: [0] chunk rows used, [1] status, [2..2+maxBodies) share counts, then per vehicle
// 8 words of bounds (ordered bits: lo xyz, hi xyz, speed, 0) (all cleared each window),
// then per vehicle 4 words written once by the host: node offset, nodes, mass bits, 0.
uint wShare() { return 2u; }
uint wBounds(uint v) { return 2u + gP.maxBodies + 8u * v; }
uint4 vehicleAt(uint v)
{
    uint w = 2u + gP.maxBodies + 8u * gP.vehicles + 4u * v;
    return uint4(gWork[w].load(), gWork[w + 1u].load(), gWork[w + 2u].load(), 0u);
}

[[vk::binding(0, 0)]] ConstantBuffer<Params> gP;
[[vk::binding(1, 0)]] StructuredBuffer<float4> gBodies;     // engine-schema BodyState, 7 each
[[vk::binding(2, 0)]] StructuredBuffer<uint4> gColliders;   // engine-schema Collider, 4 each
[[vk::binding(3, 0)]] StructuredBuffer<uint> gHulls;        // the hull arena's words
[[vk::binding(4, 0)]] StructuredBuffer<float4> gNodes;      // explicit nodes, 3 each
[[vk::binding(5, 0)]] RWStructuredBuffer<uint4> gMisc;      // explicit gMisc
[[vk::binding(8, 0)]] RWStructuredBuffer<Atomic<uint>> gScratch; // the World's scratch atomics
[[vk::binding(9, 0)]] RWStructuredBuffer<Atomic<uint>> gWork;
[[vk::binding(10, 0)]] StructuredBuffer<uint4> gChunkInfo;  // the ingest's collider table, 2 each
[[vk::binding(11, 0)]] RWStructuredBuffer<uint4> gRecords;  // the ingest's records, 5 each
[[vk::binding(12, 0)]] RWStructuredBuffer<uint> gRowCounts; // the ingest's per-slot rows

uint ordered(float f)
{
    uint b = asuint(f);
    return (b & 0x80000000u) != 0u ? ~b : (b | 0x80000000u);
}
float unordered(uint u)
{
    return asfloat((u & 0x80000000u) != 0u ? (u & 0x7FFFFFFFu) : ~u);
}

float3 qrot(float4 q, float3 v)
{
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

uint vrec(uint v) { return gP.miscBase + v * gP.recs; }
uint brec(uint v, uint j) { return vrec(v) + 1u + j * BODY_RECS; }
uint srec(uint v, uint k) { return vrec(v) + 1u + gP.B * BODY_RECS + k * SHAPE_RECS; }
uint prec(uint v, uint i) { return vrec(v) + 1u + gP.B * BODY_RECS + gP.S * SHAPE_RECS + i; }

groupshared Atomic<uint> gsBox[7];
groupshared Atomic<uint> gsCount;
static const uint LIST_CAP = 1024u;
groupshared uint gsList[LIST_CAP];

[shader("compute")]
[numthreads(256, 1, 1)]
void cpl_bounds(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    uint lane = lid.x;
    if (lane < 3u) gsBox[lane].store(0xFFFFFFFFu);
    else if (lane < 7u) gsBox[lane].store(0u);
    GroupMemoryBarrierWithGroupSync();
    uint4 vt = vehicleAt(v);
    float3 origin = asfloat(gMisc[v].xyz);
    for (uint i = lane; i < vt.y; i += 256u)
    {
        float3 x = origin + gNodes[(vt.x + i) * 3u].xyz;
        float3 u = gNodes[(vt.x + i) * 3u + 1u].xyz;
        gsBox[0].min(ordered(x.x));
        gsBox[1].min(ordered(x.y));
        gsBox[2].min(ordered(x.z));
        gsBox[3].max(ordered(x.x));
        gsBox[4].max(ordered(x.y));
        gsBox[5].max(ordered(x.z));
        gsBox[6].max(asuint(length(u)));
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < 7u) gWork[wBounds(v) + lane].store(gsBox[lane].load());
}

// A collider's world centre and bounding radius (its kind supported), else radius < 0.
float4 colliderSphere(uint c, out uint body, out uint kind)
{
    uint4 h = gColliders[4u * c];
    body = h.x;
    kind = h.y;
    float4 lp = asfloat(gColliders[4u * c + 1u]);
    float4 lq = asfloat(gColliders[4u * c + 2u]);
    float4 pr = asfloat(gColliders[4u * c + 3u]);
    float4 pos = gBodies[7u * body];
    float4 rot = gBodies[7u * body + 1u];
    float3 p = pos.xyz + qrot(rot, lp.xyz);
    float r = -1.0;
    if (kind == 0u) r = pr.x;                         // sphere
    else if (kind == 1u) r = pr.x + pr.y;             // capsule
    else if (kind == 2u) r = length(pr.xyz);          // box
    else if (kind == 3u) r = pr.w;                    // hull: its bounding radius (scaled)
    return float4(p, r);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void cpl_plan(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    uint lane = lid.x;
    if (lane == 0u) gsCount.store(0u);
    GroupMemoryBarrierWithGroupSync();
    uint wb = wBounds(v);
    float vmax = asfloat(gWork[wb + 6u].load());
    float reach = vmax * gP.window + gP.margin;
    float3 lo = float3(unordered(gWork[wb].load()), unordered(gWork[wb + 1u].load()), unordered(gWork[wb + 2u].load())) - reach;
    float3 hi = float3(unordered(gWork[wb + 3u].load()), unordered(gWork[wb + 4u].load()), unordered(gWork[wb + 5u].load())) + reach;
    for (uint c = lane; c < gP.colliders; c += 256u)
    {
        // Collision filters (engine-schema CollisionFilter::apply: the collider's
        // generation word, 0 for the default).
        uint g = gColliders[4u * c].w;
        uint cat = g == 0u ? 0xFFFFu : (g & 0xFFFFu);
        uint mask = g == 0u ? 0xFFFFu : (g >> 16);
        if ((cat & (gP.filter >> 16)) == 0u || ((gP.filter & 0xFFFFu) & mask) == 0u) continue;
        uint body, kind;
        float4 s = colliderSphere(c, body, kind);
        if (s.w < 0.0) continue;
        float3 bv = gBodies[7u * body + 3u].xyz;
        float3 bw = gBodies[7u * body + 4u].xyz;
        float r = s.w + (length(bv) + length(bw) * s.w) * gP.window;
        float3 q = clamp(s.xyz, lo, hi);
        float3 d = s.xyz - q;
        if (dot(d, d) > r * r) continue;
        uint at = gsCount.add(1u);
        if (at < LIST_CAP) gsList[at] = c;
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane != 0u) return;
    uint n = gsCount.load();
    if (n > gP.S || n > LIST_CAP)
    {
        gWork[1].or(ST_SHAPES);
        n = min(n, min(gP.S, LIST_CAP));
    }
    // Collider order: an insertion sort (the lists are short).
    for (uint a = 1u; a < n; a++)
    {
        uint x = gsList[a];
        uint b = a;
        while (b > 0u && gsList[b - 1u] > x)
        {
            gsList[b] = gsList[b - 1u];
            b--;
        }
        gsList[b] = x;
    }
    uint nb = 0u;
    uint planes = 0u;
    uint ns = 0u;
    for (uint k = 0u; k < n; k++)
    {
        uint c = gsList[k];
        uint body, kind;
        float4 sph = colliderSphere(c, body, kind);
        // Its body's slot (first appearance).
        uint j = NONE;
        for (uint q = 0u; q < nb; q++)
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
            gMisc[b] = asuint(p0);
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
            if (planes + nf > gP.planes) { gWork[1].or(ST_PLANES); continue; }
            for (uint f = 0u; f < nf; f++)
            {
                uint w = 4u * (base + 4u + nv + f);
                float3 nn = asfloat(uint3(gHulls[w], gHulls[w + 1u], gHulls[w + 2u]));
                float d = asfloat(gHulls[w + 3u]) * scale;
                gMisc[prec(v, planes + f)] = asuint(float4(nn, d));
            }
            span = planes | (nf << 16);
            planes += nf;
        }
        gMisc[at] = uint4(code, j, c, span);
        gMisc[at + 1u] = asuint(float4(lp.xyz, prm.x));
        gMisc[at + 2u] = lq;
        gMisc[at + 3u] = asuint(float4(prm.y, prm.z, sph.w + 0.01, 0.0));
        gMisc[at + 4u] = uint4(0u);
        gMisc[at + 5u] = uint4(0u);
        gMisc[at + 6u] = asuint(float4(sph.xyz, 0.0));
        ns++;
    }
    gMisc[vrec(v)] = uint4(nb, ns, planes, 0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cpl_share(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    for (uint j = lid.x; j < gMisc[vrec(v)].x; j += 64u)
    {
        uint b = brec(v, j);
        uint body = gMisc[b + 3u].w;
        float4 r2 = asfloat(gMisc[b + 2u]);
        gMisc[b + 2u] = asuint(float4(r2.xyz, float(max(gWork[wShare() + body].load(), 1u))));
    }
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
    float3 p = to_zup(actionPoint(ref, j, m));
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
    uint v = gid.x;
    if (gP.ingestRows == 0u) return;
    for (uint k = lid.x; k < gMisc[vrec(v)].y; k += 64u) shapeLoad(v, k);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void cpl_loads_end(uint3 tid : SV_DispatchThreadID)
{
    if (gP.ingestRows == 0u) return;
    gRowCounts[gP.ingestSlot] = min(gWork[0].load(), gP.rowCap);
}

// A proxy body's window impulse and its angular impulse about point c.
void proxyImpulse(uint v, uint j, float3 c, out float3 J, out float3 L)
{
    J = float3(0.0);
    L = float3(0.0);
    uint ns = gMisc[vrec(v)].y;
    for (uint q = 0u; q < ns; q++)
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
        proxyImpulse(v, j, cur.xyz, J, L);
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
        J = (dv + cross(dw, cur.xyz - c0.xyz)) / cur.w;
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

[shader("compute")]
[numthreads(1, 1, 1)]
void cpl_emit(uint3 gid : SV_GroupID)
{
    uint v = gid.x;
    uint ns = gMisc[vrec(v)].y;
    for (uint k = 0u; k < ns; k++)
    {
        uint at = srec(v, k);
        uint j = gMisc[at].y;
        uint body = gColliders[4u * gMisc[at].z].x;
        bool first = true;
        for (uint q = 0u; q < k; q++)
        {
            uint aq = srec(v, q);
            if (gMisc[aq].y == j && gColliders[4u * gMisc[aq].z].x == body) { first = false; break; }
        }
        if (first) emitPair(v, j, body, k);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void cpl_emit_end(uint3 tid : SV_DispatchThreadID)
{
    gScratch[gP.impulseFirst].store(min(gScratch[gP.impulseCount].load(), gP.impulseCap));
}
