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
    uint pad0;
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

groupshared Atomic<uint> gsBox[7];
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

[shader("compute")]
[numthreads(256, 1, 1)]
void cpl_bounds(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    uint lane = lid.x;
    if (lane < 3u) gsBox[lane].store(0xFFFFFFFFu);
    else if (lane < 7u) gsBox[lane].store(0u);
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
    for (uint i = lane; i < nodes; i += 256u)
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
    if (lane < 7u && !keep) gWork[wBounds(v) + lane].store(gsBox[lane].load());
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

// --- cpl_query: the World broadphase's candidates (mode 1) ---

uint candAppend(uint v, uint c)
{
    uint at = gWork[wCandCount(v)].add(1u);
    if (at < gP.candCap) gWork[wCands(v) + at].store(c);
    else gWork[1].or(ST_SHAPES);
    return at;
}

void candTry(uint v, uint c, float3 lo, float3 hi)
{
    if (c < gP.colliders && cplAccept(c, lo, hi)) candAppend(v, c);
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
    while (j != 0xFFFFFFFFu && guard < gP.colliders + 1u)
    {
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
    while (j != 0xFFFFFFFFu && guard < gP.colliders + 1u)
    {
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
    if (gP.mode == 0u) return;
    if (lane == 0u) gsUniformCpl = vehicleAt(v);
    GroupMemoryBarrierWithGroupSync();
    uint4 vt = gsUniformCpl;
    if ((vt.w & PROXY_OUT) != 0u) return;
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
                for (uint h = lane; h < gP.hashSize; h += 256u) restRange(v, h, a, b, level, cs, true, lo, hi);
                continue;
            }
            for (uint k = lane; k < cells; k += 256u)
            {
                int3 cell = a + int3(int(k % uint(span.x)), int((k / uint(span.x)) % uint(span.y)), int(k / (uint(span.x) * uint(span.y))));
                restChain(v, hierHashW(cell, level), cell, level, cs, true, lo, hi);
            }
        }
        // Oversized statics (header 11..20).
        uint large = min(gIds[hdr + 10u], 10u);
        for (uint k = lane; k < large; k += 256u) candTry(v, gIds[hdr + 11u + k], lo, hi);
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
            for (uint h = lane; h < gP.hashSize; h += 256u) restRange(v, h, a, b, 0u, cs, false, lo, hi);
        }
        else
        {
            for (uint k = lane; k < cells; k += 256u)
            {
                int3 cell = a + int3(int(k % uint(span.x)), int((k / uint(span.x)) % uint(span.y)), int(k / (uint(span.x) * uint(span.y))));
                restChain(v, gridHashW(cell), cell, 0u, cs, false, lo, hi);
            }
        }
        uint large = min(gIds[hdr + 10u], 16u);
        for (uint k = lane; k < large; k += 256u) candTry(v, gIds[hdr + 11u + k], lo, hi);
        uint statics = min(gIds[hdr + 28u], 1024u);
        for (uint k = lane; k < statics; k += 256u) candTry(v, gScratch[gP.largeStaticBase + k].load(), lo, hi);
    }
    // The step's moving colliders (awake at its start), and every collider of a body
    // whose pose or sleep state changed in it (pose_pack).
    uint nm = gScratch[6].load();
    for (uint i = lane; i < nm; i += 256u) candTry(v, gIds[restLinksW() + gP.colliders + i], lo, hi);
    uint nb = gMovedHeader[0];
    for (uint i = lane; i < nb; i += 256u)
    {
        uint body = gMoved[2u * i].w & 0x0FFFFFFFu;
        uint l = 0u;
        uint h = gP.colliders;
        while (l < h)
        {
            uint mid = (l + h) >> 1;
            if (gColliders[4u * mid].x < body) l = mid + 1u; else h = mid;
        }
        for (uint c = l; c < gP.colliders && gColliders[4u * c].x == body; c++) candTry(v, c, lo, hi);
    }
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
    for (uint i = 0u; i < nv; i++)
    {
        uint w = 4u * (base + 4u + i);
        float3 p = asfloat(uint3(gHulls[w], gHulls[w + 1u], gHulls[w + 2u])) * scale;
        gMisc[prec(v, at + i)] = asuint(float4(p, 0.0));
    }
    uint edgeWord = gHulls[4u * base + 2u];
    for (uint r = 0u; r < (ne + 3u) / 4u; r++)
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
    for (uint k = lane; k < n; k += 256u)
    {
        uint body = gsBody[k];
        // Its body's first shape, the bodies first seen before it, the planes before it.
        uint f = k;
        uint slot = 0u;
        uint po = 0u;
        uint last = k;
        for (uint j = 0u; j < n; j++)
        {
            uint bj = gsBody[j];
            if (j < k) po += gsFaces[j] & 0xFFFFu;
            if (bj == body)
            {
                f = min(f, j);
                last = j;
            }
        }
        for (uint j = 0u; j < f; j++) slot += gsFaces[j] >> 31;
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
            for (uint fi = 0u; fi < nf; fi++)
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
        return;
    }
    float3 lo;
    float3 hi;
    sweptBox(v, lo, hi, false);
    if (gP.mode == 0u)
    {
        for (uint c = lane; c < gP.colliders; c += 256u)
        {
            if (!cplAccept(c, lo, hi)) continue;
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
        for (uint i = lane; i < nc; i += 256u)
        {
            uint c = gWork[wCands(v) + i].load();
            bool dup = false;
            for (uint j = 0u; j < i; j++)
            {
                if (gWork[wCands(v) + j].load() == c) { dup = true; break; }
            }
            if (dup) continue;
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
    for (uint i = lane; i < n; i += 256u)
    {
        uint x = gsList[i];
        uint rank = 0u;
        for (uint j = 0u; j < n; j++) rank += gsList[j] < x ? 1u : 0u;
        gsSorted[rank] = x;
    }
    if (lane == 0u)
    {
        gsBodies.store(0u);
        gsPlanes.store(0u);
    }
    GroupMemoryBarrierWithGroupSync();
    // Each shape's engine body and hull faces; its body's first appearance counted.
    for (uint k = lane; k < n; k += 256u)
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
    for (uint k = lane; k < n; k += 256u)
    {
        bool first = true;
        for (uint j = 0u; j < k; j++)
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
        return;
    }
    if (lane != 0u) return;
    // Collider order: an insertion sort (the lists are short).
    for (uint a = 1u; a < n; a++)
    {
        uint x = gsSorted[a];
        uint b = a;
        while (b > 0u && gsSorted[b - 1u] > x)
        {
            gsSorted[b] = gsSorted[b - 1u];
            b--;
        }
        gsSorted[b] = x;
    }
    uint nb = 0u;
    uint planes = 0u;
    uint ns = 0u;
    for (uint k = 0u; k < n; k++)
    {
        uint c = gsSorted[k];
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
            for (uint f = 0u; f < nf; f++)
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
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cpl_share(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = listed(gid.x);
    if (v == NONE) return;
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
    if (v == NONE) return;
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
    if (v == NONE) return;
    uint ns = gMisc[vrec(v)].y;
    // (Tests, gP.serialPlan: lane 0 alone, the shapes one after another.)
    uint lanes = gP.serialPlan != 0u ? 1u : 64u;
    if (lid.x >= lanes) return;
    for (uint k = lid.x; k < ns; k += lanes)
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

[shader("compute")]
[numthreads(64, 1, 1)]
void cpl_sleep_mask(uint3 tid : SV_DispatchThreadID)
{
    uint t = tid.x;
    if (gP.sleepBase == 0u || 32u * t >= gP.vehicles) return;
    uint word = 0u;
    for (uint k = 0u; k < 32u && 32u * t + k < gP.vehicles; k++)
        word |= (gMisc[gP.sleepBase + 32u * t + k].y & 1u) << k;
    gWork[2u + gP.maxBodies + 12u * gP.vehicles + t].store(word);
}
