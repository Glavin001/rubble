// Explicit vehicles in the World (engine/docs/explicit-vehicles-in-world.md): the
// coupling's plan, its impulses and its chunk loads, between the World's buffers and the
// explicit solver's (ExplicitBeamsGpu built with coupling room; its records:
// explicit_beams_rigid.slang). One standalone WGSL module; host engine-world
// `explicit_vehicles.rs`.
//
//   cpl_bounds     one group per vehicle: its nodes' world box and fastest speed, and
//                  its broadphase proxy (a fat box, refattened only when the swept box
//                  leaves it; not while the vehicle sleeps or is out of the broadphase);
//   cpl_query      one group per vehicle (the World's resting table kept by the last
//                  step, gP.mode 1): the candidates near its swept box, from the World
//                  broadphase's persistent resting table (statics and sleepers at their
//                  cells: O(nearby)) and the step's moving and moved colliders
//                  (O(awake)), each kept when it passes cpl_plan's exact test;
//   cpl_plan       one group per vehicle: every eligible collider whose bounding sphere,
//                  swept over the window, meets the vehicle's swept box (mode 0: a scan
//                  of every collider; mode 1: cpl_query's candidates), sorted by
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
//   cpl_sleep_mask one thread per 32 vehicles, after the frame: the vehicles' asleep bits
//                  (vehicle sleep, explicit_beams_sleep.slang) for the World's readback.
//
// Proxied list (gP.listBase nonzero): cpl_plan appends each vehicle with a proxy shape to
// gWork[listBase + 4 ...] (its count at listBase, the x of the indirect arguments
// (count, 1, 1) the host dispatches cpl_share, cpl_loads and cpl_emit with): a vehicle
// with no proxy is not launched (those passes do nothing for it). The list stays until
// the next window's plan clears its count (cpl_emit, at the next step's start, reads it).
// cpl_bounds keeps a sleeping vehicle's bounds (its nodes have not moved since they
// were taken; the bounds are not in the cleared region).
//
// Vehicle sleep (sleepBase nonzero: gMisc record of vehicle 0's sleep record): cpl_plan
// wakes a vehicle (writes its record (0, 0, 0, 0): awake, still count 0, no anchor) when
// a collider among its candidates belongs to a moving body: a dynamic body awake in the
// World, or any body with a velocity (a kinematic one moving). Static, anchored and
// sleeping bodies do not wake it. Run every window, for awake vehicles too, so a vehicle
// cannot fall asleep beside a moving body.
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
    uint serialPlan;     // tests: cpl_plan's lane-0 walk even within every room; cpl_emit on lane 0
    uint sleepBase;      // vehicle sleep: gMisc record of vehicle 0's sleep record (0 off)
    uint listBase;       // gWork word of the proxied list's count (0: every vehicle, gid.x)
    uint hullGeom;       // nonzero: hull proxies carry their vertices and edges (surface contact)
    uint mode;           // 0: cpl_plan scans every collider; 1: cpl_query's candidates
    // The World's resting table as the last step left it (WgpuSim::rest_view; mode 1).
    uint restBase;       // binding 14 word of its header
    uint hashSize;       // heads per table
    uint aabbBase;       // gAabbs record of collider 0 (2 float4 each)
    uint levels;         // hierarchical levels (0: the uniform table)
    float cell;          // the grid's base cell (m)
    uint largeStaticBase; // scratch word of the uniform table's oversized statics
    float slack;         // a proxy's fat box: the swept box grown by this (m) when it leaves
    uint candCap;        // candidates per vehicle
    uint extraBase;      // gWork word of the broadphase region (candidate counts, proxies, candidates)
    uint loopBudget;     // passes one invocation may take (explicit_loop_cap.slang; 0 none)
    uint retireBase;     // gMisc record of the explicit nodes' retired bits (explicit_beams.slang nodeRetired; 0 none)
    uint labelBase;      // gMisc record of the nodes' piece labels, 4 bits each (0: one window per vehicle)
    uint pieceBoxBase;   // gMisc: per vehicle PIECE_BOX_RECS records, its pieces' node boxes, then its labels-in-use word
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
static const uint ST_LOOP = 32u; // a loop held to its cap or past the pass budget
static const uint ST_CANDS = 64u; // cpl_query's candidates (duplicates included) past their room
// work: [0] chunk rows used, [1] status, [2..2+maxBodies) share counts (all cleared each
// window), then per vehicle 8 words of bounds (ordered bits: lo xyz, hi xyz, speed, 0;
// rewritten by cpl_bounds for every awake vehicle), then per vehicle 4 words written once
// by the host: node offset, nodes, mass bits, proxy flags (bit 1 out of the broadphase);
// then the asleep bits (vehicle sleep), a word per 32 vehicles; then the proxied list
// (listBase: count, 1, 1, 0, the vehicles); then at extraBase per vehicle its candidate
// count (cpl_bounds clears it), per vehicle its proxy (8 words: fat lo xyz, fat hi xyz
// as float bits, 0, refattens), per vehicle candCap candidates.
uint wShare() { return 2u; }
uint wBounds(uint v) { return 2u + gP.maxBodies + 8u * v; }
uint4 vehicleAt(uint v)
{
    uint w = 2u + gP.maxBodies + 8u * gP.vehicles + 4u * v;
    return uint4(gWork[w].load(), gWork[w + 1u].load(), gWork[w + 2u].load(), gWork[w + 3u].load());
}
static const uint PROXY_OUT = 2u;
uint wCandCount(uint v) { return gP.extraBase + v; }
uint wProxy(uint v) { return gP.extraBase + gP.vehicles + 8u * v; }
uint wCands(uint v) { return gP.extraBase + 9u * gP.vehicles + gP.candCap * v; }

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
[[vk::binding(13, 0)]] StructuredBuffer<uint> gIds;         // the World's binding 14 (resting table)
[[vk::binding(14, 0)]] StructuredBuffer<float4> gAabbs;     // the World's collider bounds (mn, r), (mx, link)
[[vk::binding(15, 0)]] StructuredBuffer<uint> gMovedHeader; // pose_pack: [0] bodies moved this step
[[vk::binding(16, 0)]] StructuredBuffer<uint4> gMoved;      // pose_pack: 2 records each, .w of the first the body

// Loop caps and the pass budget (explicit_loop_cap.slang): a loop held to its cap or
// past the budget sets ST_LOOP in the status word (work[1]), which the host reports.
#define EB_LOOP_BUDGET gP.loopBudget
#define EB_LOOP_CAP_STORE(hit) gWork[1u].or(ST_LOOP)
#include "explicit_loop_cap.slang"

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

// Workgroup g's vehicle: the g-th proxied one (listBase), or g; NONE past the list.
uint listed(uint g)
{
    if (gP.listBase == 0u) return g;
    return g < gWork[gP.listBase].load() ? gWork[gP.listBase + 4u + g].load() : NONE;
}
// cpl_plan: vehicle v has proxies this window.
void listAppend(uint v, uint shapes)
{
    if (gP.listBase == 0u || shapes == 0u) return;
    uint at = gWork[gP.listBase].add(1u);
    gWork[gP.listBase + 4u + at].store(v);
}
uint brec(uint v, uint j) { return vrec(v) + 1u + j * BODY_RECS; }
uint srec(uint v, uint k) { return vrec(v) + 1u + gP.B * BODY_RECS + k * SHAPE_RECS; }
uint prec(uint v, uint i) { return vrec(v) + 1u + gP.B * BODY_RECS + gP.S * SHAPE_RECS + i; }

// The vehicle's node box and fastest speed (0..6), then with piece labels each piece's
// node box (7 + 6 g: min x, y, z keys, max x, y, z keys; one variable: Metal's slots).
// 16 groups: the pair list's piece labels use 0..7, a host may group a vehicle's nodes for
// the coupling alone into up to 16 (ExplicitBeamsGpu::set_coupling_groups).
static const uint PIECES = 16u;
static const uint PIECE_BOX_RECS = 25u;
groupshared Atomic<uint> gsBox[7u + 6u * PIECES];
groupshared Atomic<uint> gsCount;
static const uint LIST_CAP = 1024u;
groupshared uint gsList[LIST_CAP];
groupshared uint gsSorted[LIST_CAP];
groupshared uint gsBody[LIST_CAP];
groupshared uint gsFaces[LIST_CAP];
groupshared Atomic<uint> gsBodies;
groupshared Atomic<uint> gsPlanes;
groupshared Atomic<uint> gsWake;
// A vehicle's record and sleep word broadcast to its workgroup: barriers follow
// branches on them, and Chrome's uniformity analysis needs workgroupUniformLoad
// (engine_webgpu::uniformize_workgroup_lanes rewrites the reads).
groupshared uint4 gsUniformCpl;
groupshared uint gsUniformCplKeep;

// Vehicle sleep: a body that moves (wakes a vehicle it may touch).
bool bodyMoving(uint body)
{
    uint flags = asuint(gBodies[7u * body + 2u].w);
    float3 bv = gBodies[7u * body + 3u].xyz;
    float3 bw = gBodies[7u * body + 4u].xyz;
    bool awakeDynamic = (flags & 1u) != 0u && (flags & 4u) == 0u;
    return awakeDynamic || any(bv != float3(0.0)) || any(bw != float3(0.0));
}

void sweptBox(uint v, out float3 lo, out float3 hi, bool fresh);
bool cplAccept(uint c, float3 lo, float3 hi);

// --- A window per piece (lossless) ---
// With piece labels (the host's, engine_softbody::explicit_debris; written by
// ExplicitBeamsGpu::set_piece_labels, or its groups by set_coupling_groups: 4 bits a node
// at labelBase; any grouping is sound, a group's box is taken from its nodes each window)
// cpl_bounds also takes
// each piece's node box, and cpl_plan keeps a collider only if its swept sphere meets some
// piece's box grown by the vehicle's reach (its fastest node's travel over the window and
// the margin), not just the whole vehicle's: a piece fallen far from the car no longer
// puts every body between them in the window. A node meets only shapes near its own
// piece's box under the same bound the vehicle's box relies on, so the shapes left out
// meet no node: the same contacts, with fewer proxies to test.
uint pieceLabelOf(uint gi)
{
    uint4 w = gMisc[gP.labelBase + (gi >> 5u)];
    uint k = (gi >> 3u) & 3u;
    uint x = k == 0u ? w.x : (k == 1u ? w.y : (k == 2u ? w.z : w.w));
    return (x >> ((gi & 7u) * 4u)) & 15u;
}
void pieceBoxAdd(uint gi, float3 x)
{
    uint b = 7u + 6u * pieceLabelOf(gi);
    gsBox[b].min(ordered(x.x));
    gsBox[b + 1u].min(ordered(x.y));
    gsBox[b + 2u].min(ordered(x.z));
    gsBox[b + 3u].max(ordered(x.x));
    gsBox[b + 4u].max(ordered(x.y));
    gsBox[b + 5u].max(ordered(x.z));
}
// Whether collider c's swept sphere meets some piece's box grown by `reach` (true when the
// vehicle's labels are not in use).
bool pieceAccept(uint v, uint c, float reach)
{
    if (gP.labelBase == 0u) return true;
    uint at = gP.pieceBoxBase + v * PIECE_BOX_RECS;
    if (gMisc[at + 24u].x == 0u) return true;
    for (uint g = 0u; g < PIECES; g++)
    {
        uint w = 6u * g;
        uint k[6];
        for (uint j = 0u; j < 6u; j++)
        {
            uint4 r = gMisc[at + (w + j) / 4u];
            uint m = (w + j) % 4u;
            k[j] = m == 0u ? r.x : (m == 1u ? r.y : (m == 2u ? r.z : r.w));
        }
        if (k[0] == 0xFFFFFFFFu) continue;
        float3 lo = float3(unordered(k[0]), unordered(k[1]), unordered(k[2])) - reach;
        float3 hi = float3(unordered(k[3]), unordered(k[4]), unordered(k[5])) + reach;
        if (cplAccept(c, lo, hi)) return true;
    }
    return false;
}
// The vehicle's reach (sweptBox's growth).
float sweptReach(uint v)
{
    return asfloat(gWork[wBounds(v) + 6u].load()) * gP.window + gP.margin;
}

// explicit_beams.slang nodeRetired: node gi's retired bit.
bool cplRetired(uint gi)
{
    uint4 w = gMisc[gP.retireBase + (gi >> 7u)];
    uint k = (gi >> 5u) & 3u;
    uint x = k == 0u ? w.x : (k == 1u ? w.y : (k == 2u ? w.z : w.w));
    return ((x >> (gi & 31u)) & 1u) != 0u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void cpl_bounds(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    uint lane = lid.x;
    if (lane < 3u) gsBox[lane].store(0xFFFFFFFFu);
    else if (lane < 7u) gsBox[lane].store(0u);
    else if (lane < 7u + 6u * PIECES) gsBox[lane].store(((lane - 7u) % 6u) < 3u ? 0xFFFFFFFFu : 0u);
    if (lane == 0u)
    {
        gsUniformCpl = vehicleAt(v);
        gsUniformCplKeep = (gP.sleepBase != 0u && (gMisc[gP.sleepBase + v].y & 1u) != 0u) ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    uint4 vt = gsUniformCpl;
    // Vehicle sleep: a sleeping vehicle's nodes have not moved since its bounds were
    // taken; they stand (every lane takes the barriers).
    bool keep = gsUniformCplKeep != 0u;
    // Out of the broadphase: no update either (no proxies).
    keep = keep || (vt.w & PROXY_OUT) != 0u;
    if (lane == 0u && gP.extraBase != 0u) gWork[wCandCount(v)].store(0u);
    uint nodes = keep ? 0u : vt.y;
    float3 origin = asfloat(gMisc[v].xyz);
    for (uint i = lane, iStop = EB_END(561, i, nodes, 256u); i < iStop; i += 256u)
    {
        // Retired debris is not where the vehicle is (it meets nothing).
        if (gP.retireBase != 0u && cplRetired(vt.x + i)) continue;
        float3 x = origin + gNodes[(vt.x + i) * 3u].xyz;
        float3 u = gNodes[(vt.x + i) * 3u + 1u].xyz;
        gsBox[0].min(ordered(x.x));
        gsBox[1].min(ordered(x.y));
        gsBox[2].min(ordered(x.z));
        gsBox[3].max(ordered(x.x));
        gsBox[4].max(ordered(x.y));
        gsBox[5].max(ordered(x.z));
        gsBox[6].max(asuint(length(u)));
        if (gP.labelBase != 0u) pieceBoxAdd(vt.x + i, x);
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < 7u && !keep) gWork[wBounds(v) + lane].store(gsBox[lane].load());
    // Each piece's box (a window per piece in cpl_plan; pieceAccept): 24 records.
    if (gP.labelBase != 0u && lane < 24u && !keep)
    {
        uint b = 7u + 4u * lane;
        gMisc[gP.pieceBoxBase + v * PIECE_BOX_RECS + lane] =
            uint4(gsBox[b].load(), gsBox[b + 1u].load(), gsBox[b + 2u].load(), gsBox[b + 3u].load());
    }
    // The broadphase proxy: kept while the swept box stays inside it.
    if (lane == 0u && !keep && vt.y > 0u && gP.extraBase != 0u)
    {
        float3 lo;
        float3 hi;
        sweptBox(v, lo, hi, true);
        uint pw = wProxy(v);
        float3 flo = asfloat(uint3(gWork[pw].load(), gWork[pw + 1u].load(), gWork[pw + 2u].load()));
        float3 fhi = asfloat(uint3(gWork[pw + 3u].load(), gWork[pw + 4u].load(), gWork[pw + 5u].load()));
        uint moves = gWork[pw + 7u].load();
        if (moves == 0u || any(lo < flo) || any(hi > fhi))
        {
            float3 s = float3(gP.slack, gP.slack, gP.slack);
            flo = lo - s;
            fhi = hi + s;
            gWork[pw].store(asuint(flo.x));
            gWork[pw + 1u].store(asuint(flo.y));
            gWork[pw + 2u].store(asuint(flo.z));
            gWork[pw + 3u].store(asuint(fhi.x));
            gWork[pw + 4u].store(asuint(fhi.y));
            gWork[pw + 5u].store(asuint(fhi.z));
            gWork[pw + 7u].store(moves + 1u);
        }
    }
    ebLoopCapFlush();
}

// Vehicle v's swept box this window: its nodes' box (cpl_bounds) grown by the fastest
// node's travel over the window and the margin. `fresh`: within cpl_bounds (workgroup
// memory), else from gWork.
void sweptBox(uint v, out float3 lo, out float3 hi, bool fresh)
{
    uint wb = wBounds(v);
    uint b[7];
    for (uint k = 0u; k < 7u; k++) b[k] = fresh ? gsBox[k].load() : gWork[wb + k].load();
    float vmax = asfloat(b[6]);
    float reach = vmax * gP.window + gP.margin;
    lo = float3(unordered(b[0]), unordered(b[1]), unordered(b[2])) - reach;
    hi = float3(unordered(b[3]), unordered(b[4]), unordered(b[5])) + reach;
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

// The exact test: collider c (its filter accepting the vehicles', a supported kind)
// whose bounding sphere, grown by its body's travel over the window, meets the swept
// box lo..hi.
bool cplAccept(uint c, float3 lo, float3 hi)
{
    // Collision filters (engine-schema CollisionFilter::apply: the collider's
    // generation word, 0 for the default).
    uint g = gColliders[4u * c].w;
    uint cat = g == 0u ? 0xFFFFu : (g & 0xFFFFu);
    uint mask = g == 0u ? 0xFFFFu : (g >> 16);
    if ((cat & (gP.filter >> 16)) == 0u || ((gP.filter & 0xFFFFu) & mask) == 0u) return false;
    uint body, kind;
    float4 s = colliderSphere(c, body, kind);
    if (s.w < 0.0) return false;
    float3 bv = gBodies[7u * body + 3u].xyz;
    float3 bw = gBodies[7u * body + 4u].xyz;
    float r = s.w + (length(bv) + length(bw) * s.w) * gP.window;
    float3 q = clamp(s.xyz, lo, hi);
    float3 d = s.xyz - q;
    return !(dot(d, d) > r * r);
}

#include "vehicle_coupling_query.slang"
