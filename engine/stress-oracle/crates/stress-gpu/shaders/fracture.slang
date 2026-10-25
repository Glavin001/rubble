// GPU-resident bond-graph fracture over a rigid engine with fixed slots (the threshold
// profile's frame, engine/docs/stress-threshold-gpu.md). Every chunk keeps its colliders'
// engine slots for good, and the engine has a body slot for as many bodies as there are
// chunks, so a split allocates nothing: it rewrites body records and re-parents colliders.
// The host reads nothing back to run a frame.
//
//   fr_loads        one thread per chunk: its engine load over the frame (impact.slang
//                   `chunk_engine_load`, from the ingest's manifolds), into `loads`;
//   fr_breaks       one thread per bond: an intact bond breaks when the load on either of
//                   its chunks beats its tensile capacity (threshold.slang `threshold_breaks`);
//                   its body is marked dirty;
//   fr_gate         one thread: the indirect sizes of the split kernels (all zero when no
//                   bond broke this frame, so a quiet frame costs nothing more);
//   fr_uf_init, fr_body_init, fr_uf_union, fr_uf_flatten
//                   connected components of every dirty body's chunks over its intact bonds:
//                   lock-free union-find, each root hooked under the smaller (so a piece's
//                   root is its lowest chunk, whatever order the threads ran in), and each
//                   piece's chunk count, anchoring and mass sums (64-bit fixed point: integer
//                   sums do not depend on the order of the atomics);
//   fr_best_count, fr_best_root
//                   each dirty body's largest piece (ties: the lowest root), which keeps the
//                   body (the minimum edit, as blast-stress-solver's `plan_split_migration`
//                   keeps the best-overlapping body: its slot, its colliders' slots);
//   fr_mark, scans, fr_freelist, fr_assign
//                   every other piece takes a free slot, the k-th new piece (by root) the
//                   k-th free slot (ascending): deterministic;
//   fr_piece, fr_commit
//                   each piece's body as stress-rubble `cook_with` makes it (mass, centre,
//                   principal axes from its chunks' parts), in its parent's frame, moving
//                   with its parent's rigid motion; written in both pose regions;
//   fr_reparent     one thread per chunk of a dirty body: its colliders on its piece's body,
//                   at `to_body (origin - com)` turned `to_body rotation` (spawn.slang
//                   `member_colliders`), and the ingest's collider table (body code, mass);
//   fr_body_clear   the dirty marks cleared.
//
// Frames: chunk and part data are in their structure's frame (the stress solver's, Z-up);
// each body slot keeps where its centre of mass and principal axes sit in that frame
// (`frames`). The engine is Y-up (rigid_frame.slang).

import common;
import impact;
import rigid_frame;

struct FractureParams
{
    uint chunk_count;
    uint bond_count;
    uint body_count;     // engine body slots the fracture may use (`free`'s length)
    uint max_bodies;     // the engine's capacity (its previous-pose region starts there)
    uint step;           // the engine's next step: a re-parented body's generation
    uint pool_cap;       // carried impact pulses each half of the pool holds
    uint impact_peak;
    uint parity;         // the half of the pool this frame reads (the frame's number mod 2)
    float dt;
    float band;          // |g| dt: closing speeds within it are not impacts
    float scale_m;       // fixed-point scales of the mass sums (2^k)
    float scale_s;
    float scale_j;
    float strike_speed;  // a slower closing is no strike (world_coupling.rs `tiered_impact_speed`)
    uint shot_count;     // the shots (`shots`), parked ones included
    float pad3;
};

struct ScanParams
{
    uint n;
    uint in_base;   // `work` words
    uint out_base;
    uint block_base;
};

// Per chunk 4 float4: (centre (structure frame), mass); inertia about the centre
// (xx, yy, zz, xy); (xz, yz, first part as bits, parts as bits); (Young's modulus,
// Poisson's ratio, cbrt(volume), smallest half extent). `info` per chunk: (structure,
// flags: 1 support).
[[vk::binding(0, 0)]] ConstantBuffer<FractureParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<float4> chunks;
// Per bond: (chunk a, chunk b, tensile capacity as bits, -).
[[vk::binding(2, 0)]] StructuredBuffer<uint4> bonds;
// Per part 2 float4: (origin (structure frame), engine collider as bits); rotation.
[[vk::binding(3, 0)]] StructuredBuffer<float4> parts;
// Plain words (bases from `Layout`, fracture.rs): per chunk its body, per bond whether it
// is intact, per chunk its piece root, per root its body slot, scan inputs and outputs.
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> work;
// Words the kernels update with atomics: union-find parents, piece counts, anchoring and
// mass sums, dirty marks, each dirty body's largest piece, counters.
[[vk::binding(5, 0)]] RWStructuredBuffer<Atomic<uint>> atom;
// Per engine body 2 float4: (centre of mass (structure frame), structure as bits);
// principal rotation (structure frame). Then per root (from `stage_base`) its body's 7
// float4 and its 2 frame float4, staged by `fr_piece` for `fr_commit`.
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> frames;
// The engine's bodies (rubble `BodyState`, 7 float4) and colliders (`Collider`, 4 float4).
[[vk::binding(7, 0)]] RWStructuredBuffer<float4> bodies;
[[vk::binding(8, 0)]] RWStructuredBuffer<float4> colliders;
// The ingest's collider table (`ColliderInfo`: uint4 a, float4 b as bits).
[[vk::binding(9, 0)]] RWStructuredBuffer<uint4> collider_info;
[[vk::binding(10, 0)]] StructuredBuffer<Record> manifolds;
[[vk::binding(11, 0)]] StructuredBuffer<uint> sort;
// Per chunk its load (x: |force|), then its carried pulses (impact.slang).
[[vk::binding(12, 0)]] RWStructuredBuffer<float4> loads;
// The split kernels' indirect sizes (3 words each, `ARG_*`), copied to the buffer the
// dispatches read.
[[vk::binding(13, 0)]] RWStructuredBuffer<uint> args;
[[vk::binding(14, 0)]] ConstantBuffer<ScanParams> scan;
[[vk::binding(15, 0)]] StructuredBuffer<uint> info;

static const uint NONE = 0xffffffffu;
static const uint BODY_WORDS = 7;
static const uint FLAG_DYNAMIC = 1;
static const uint FLAG_AWAKE = 8;
static const uint INFO_SUPPORT = 1;
static const uint SUMS = 20;      // per root: mass, first moment (3), second moment (6), 64-bit each

// `work` layout (fracture.rs `Layout` writes the same).
uint w_body() { return 0; }
uint w_alive() { return params.chunk_count; }
uint w_label() { return w_alive() + params.bond_count; }
uint w_slot() { return w_label() + params.chunk_count; }
uint w_new() { return w_slot() + params.chunk_count; }
uint w_rank() { return w_new() + params.chunk_count; }
uint w_free() { return w_rank() + params.chunk_count; }
uint w_free_rank() { return w_free() + params.body_count; }
uint w_free_list() { return w_free_rank() + params.body_count; }
// Per chunk, for each half of the pool: where its carried pulses start and how many.
uint w_pulses() { return w_free_list() + params.body_count + 2 * 1025; }

// `atom` layout.
uint a_parent() { return 0; }
uint a_count() { return params.chunk_count; }
uint a_anchor() { return 2 * params.chunk_count; }
uint a_sums() { return 3 * params.chunk_count; }
uint a_dirty() { return a_sums() + SUMS * params.chunk_count; }
uint a_best_count() { return a_dirty() + params.body_count; }
uint a_best_root() { return a_best_count() + params.body_count; }
uint a_counters() { return a_best_root() + params.body_count; }
// Per body slot 9 words for the strike: its bounds (ordered floats: low xyz, high xyz), the
// entry time (float bits), the entry chunk's distance to the path (float bits), the entry
// chunk.
uint a_strike() { return a_counters() + 16; }

// Counters (`a_counters` + i).
static const uint C_FRAME_BROKEN = 0; // bonds broken this frame (fr_gate clears it)
static const uint C_BROKEN = 1;       // bonds broken in all
static const uint C_PIECES = 2;       // new bodies in all
static const uint C_ERROR = 3;        // bits: 1 no free slot, 2 a mass sum out of range, 4 a union did not settle, 8 a pulse found no room
static const uint C_SPLIT_FRAMES = 4;
static const uint C_POOL_TOP = 5;     // + the half: pulses claimed in it this frame

// `loads` layout: per chunk 1 float4 (|its load|, its load (world)), then the two halves of the pulse pool
// (`pool_cap` pulses of 3 float4 each). A frame reads its chunks' carried pulses from one
// half and writes the ones that last into the other: each chunk claims a run there with
// one atomic add, where it lands does not matter (each chunk reads only its own, in
// order), and the pool's size follows the engine's contact records (fracture.rs), not a
// per-chunk count.
uint l_pool(uint half) { return params.chunk_count + 3 * params.pool_cap * half; }

// Indirect sizes (`args`, 3 words each).
static const uint ARG_CHUNKS = 0;
static const uint ARG_BONDS = 1;
static const uint ARG_BODIES = 2;
static const uint ARG_SCAN_CHUNKS = 3;
static const uint ARG_SCAN_BODIES = 4;
static const uint ARG_ONE = 5;

uint body_of(uint c) { return work[w_body() + c]; }
bool dirty(uint body) { return atom[a_dirty() + body].load() != 0; }
uint groups_of(uint n, uint size) { return (n + size - 1) / size; }

// A body slot's structure-frame pose in the world (Z-up): the rotation and origin of its
// structure's frame, from the engine's record (`bodies` at word `b`) and the slot's frame.
void structure_pose(uint slot, uint b, out Quat q, out float3 origin)
{
    Quat p_inv = conj(quat_of(frames[2 * slot + 1]));
    float3 com = frames[2 * slot].xyz;
    q = normalized(quat_mul(rot_from_rubble(quat_of(bodies[b + 1])), p_inv));
    origin = from_rubble(bodies[b].xyz) - rotate(q, com);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_loads(uint3 id: SV_DispatchThreadID)
{
    uint k = id.x;
    if (k >= params.chunk_count)
        return;
    uint slot = body_of(k);
    uint b = BODY_WORDS * slot;
    Quat q;
    float3 origin;
    structure_pose(slot, b, q, origin);
    float4 c0 = chunks[4 * k];
    float3 centre = origin + rotate(q, c0.xyz);
    float inv_mass = bodies[b].w;
    float m_body = inv_mass > 0.0 ? 1.0 / inv_mass : float_bits(0x7f800000u);
    float3 f_total;
    float3 t_total;
    bool overflow;
    uint kept;
    uint from = params.parity;
    uint to = 1 - from;
    uint off = work[w_pulses() + 4 * k + 2 * from];
    uint carried = work[w_pulses() + 4 * k + 2 * from + 1];
    // Room for every carried pulse and one per manifold (more than can be kept).
    uint room = carried + sort[params.chunk_count + 1 + k];
    uint base = 0;
    if (room != 0)
    {
        base = atom[a_counters() + C_POOL_TOP + to].add(room);
        if (base + room > params.pool_cap)
        {
            // The pool is sized from the engine's contact records: a bug, reported.
            atom[a_counters() + C_ERROR].or(8u);
            room = base < params.pool_cap ? params.pool_cap - base : 0;
        }
    }
    chunk_engine_load(manifolds, sort, loads, params.chunk_count, k, m_body, c0.w, centre, chunks[4 * k + 3], params.band, params.dt, params.impact_peak != 0, l_pool(from) + 3 * off, carried, l_pool(to) + 3 * base, room, f_total, t_total, kept, overflow);
    if (overflow)
        atom[a_counters() + C_ERROR].or(8u);
    work[w_pulses() + 4 * k + 2 * to] = base;
    work[w_pulses() + 4 * k + 2 * to + 1] = kept;
    // The magnitude (fr_breaks), and the force itself (blast_fracture.slang `blast_loads`).
    loads[k] = float4(length(f_total), f_total);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_breaks(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.bond_count || work[w_alive() + i] == 0)
        return;
    uint4 bond = bonds[i];
    float demand = max(loads[bond.x].x, loads[bond.y].x);
    if (demand > asfloat(bond.z))
    {
        work[w_alive() + i] = 0;
        atom[a_dirty() + body_of(bond.x)].or(1u);
        atom[a_counters() + C_FRAME_BROKEN].add(1u);
        atom[a_counters() + C_BROKEN].add(1u);
    }
}

void set_args(uint at, uint x)
{
    args[3 * at] = x;
    args[3 * at + 1] = 1;
    args[3 * at + 2] = 1;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void fr_gate(uint3 id: SV_DispatchThreadID)
{
    bool run = atom[a_counters() + C_FRAME_BROKEN].load() != 0;
    atom[a_counters() + C_FRAME_BROKEN].store(0);
    // The half this frame read is the next frame's to write.
    atom[a_counters() + C_POOL_TOP + params.parity].store(0);
    if (run)
        atom[a_counters() + C_SPLIT_FRAMES].add(1);
    set_args(ARG_CHUNKS, run ? groups_of(params.chunk_count, 64) : 0);
    set_args(ARG_BONDS, run ? groups_of(params.bond_count, 64) : 0);
    set_args(ARG_BODIES, run ? groups_of(params.body_count, 64) : 0);
    set_args(ARG_SCAN_CHUNKS, run ? groups_of(params.chunk_count, 1024) : 0);
    set_args(ARG_SCAN_BODIES, run ? groups_of(params.body_count, 1024) : 0);
    set_args(ARG_ONE, run ? 1 : 0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_uf_init(uint3 id: SV_DispatchThreadID)
{
    uint c = id.x;
    if (c >= params.chunk_count || !dirty(body_of(c)))
        return;
    atom[a_parent() + c].store(c);
    atom[a_count() + c].store(0);
    atom[a_anchor() + c].store(0);
    for (uint w = 0; w < SUMS; w++)
        atom[a_sums() + SUMS * c + w].store(0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_body_init(uint3 id: SV_DispatchThreadID)
{
    uint s = id.x;
    if (s >= params.body_count || !dirty(s))
        return;
    atom[a_best_count() + s].store(0);
    atom[a_best_root() + s].store(NONE);
}

// The root of `x`'s tree, halving the path as it goes (a benign race: a parent only ever
// moves to an ancestor).
uint find(uint x)
{
    uint p = atom[a_parent() + x].load();
    for (uint guard = 0; guard < (1u << 20) && p != x; guard++)
    {
        uint gp = atom[a_parent() + p].load();
        if (gp != p)
        {
            uint ignored;
            ignored = atom[a_parent() + x].compareExchange(p, gp);
        }
        x = p;
        p = gp;
    }
    return x;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_uf_union(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.bond_count || work[w_alive() + i] == 0)
        return;
    uint4 bond = bonds[i];
    if (!dirty(body_of(bond.x)))
        return;
    uint a = bond.x;
    uint b = bond.y;
    for (uint guard = 0; guard < 4096; guard++)
    {
        uint ra = find(a);
        uint rb = find(b);
        if (ra == rb)
            return;
        // Hook the larger root under the smaller: a root never gets a larger parent, so
        // every tree's root is its lowest chunk.
        uint hi = max(ra, rb);
        uint lo = min(ra, rb);
        uint was;
        was = atom[a_parent() + hi].compareExchange(hi, lo);
        if (was == hi)
            return;
        a = hi;
        b = lo;
    }
    atom[a_counters() + C_ERROR].or(4u);
}

// A 64-bit fixed-point sum of 32-bit terms: (lo, hi) at `at`, carries counted by the add
// that wrapped (integer sums: the same whatever order the threads add in).
void add64(uint at, float value, float scale)
{
    float x = round(value * scale);
    if (!(abs(x) < 1073741824.0))
    {
        atom[a_counters() + C_ERROR].or(2u);
        x = 0.0;
    }
    int v = int(x);
    uint lo = asuint(v);
    uint old;
    old = atom[at].add(lo);
    uint carry = old + lo < old ? 1u : 0u;
    uint hi = (v < 0 ? 0xffffffffu : 0u) + carry;
    if (hi != 0)
        atom[at + 1].add(hi);
}

float read64(uint at, float scale)
{
    int hi = asint(atom[at + 1].load());
    uint lo = atom[at].load();
    return (float(hi) * 4294967296.0 + float(lo)) / scale;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_uf_flatten(uint3 id: SV_DispatchThreadID)
{
    uint c = id.x;
    if (c >= params.chunk_count || !dirty(body_of(c)))
        return;
    uint r = find(c);
    work[w_label() + c] = r;
    atom[a_count() + r].add(1u);
    if ((info[c] & INFO_SUPPORT) != 0)
        atom[a_anchor() + r].or(1u);
    // The chunk's mass, first and second moments about its piece's root chunk's centre
    // (structure frame; the root is in the piece, so the offsets stay small).
    float4 c0 = chunks[4 * c];
    float4 c1 = chunks[4 * c + 1];
    float4 c2 = chunks[4 * c + 2];
    float m = c0.w;
    float3 d = c0.xyz - chunks[4 * r].xyz;
    float dd = dot(d, d);
    uint at = a_sums() + SUMS * r;
    add64(at, m, params.scale_m);
    add64(at + 2, m * d.x, params.scale_s);
    add64(at + 4, m * d.y, params.scale_s);
    add64(at + 6, m * d.z, params.scale_s);
    add64(at + 8, c1.x + m * (dd - d.x * d.x), params.scale_j);
    add64(at + 10, c1.y + m * (dd - d.y * d.y), params.scale_j);
    add64(at + 12, c1.z + m * (dd - d.z * d.z), params.scale_j);
    add64(at + 14, c1.w - m * d.x * d.y, params.scale_j);
    add64(at + 16, c2.x - m * d.x * d.z, params.scale_j);
    add64(at + 18, c2.y - m * d.y * d.z, params.scale_j);
}

bool is_root(uint c)
{
    return c < params.chunk_count && dirty(body_of(c)) && work[w_label() + c] == c;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_best_count(uint3 id: SV_DispatchThreadID)
{
    uint c = id.x;
    if (!is_root(c))
        return;
    atom[a_best_count() + body_of(c)].max(atom[a_count() + c].load());
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_best_root(uint3 id: SV_DispatchThreadID)
{
    uint c = id.x;
    if (!is_root(c))
        return;
    uint s = body_of(c);
    if (atom[a_count() + c].load() == atom[a_best_count() + s].load())
        atom[a_best_root() + s].min(c);
}

// Scan inputs: per chunk whether it roots a new piece; per body slot whether it is free.
[shader("compute")]
[numthreads(64, 1, 1)]
void fr_mark(uint3 id: SV_DispatchThreadID)
{
    uint c = id.x;
    if (c >= params.chunk_count)
        return;
    work[w_new() + c] = is_root(c) && atom[a_best_root() + body_of(c)].load() != c ? 1u : 0u;
}

groupshared uint g_scan[1024];

// Exclusive scan of `scan.n` words from `scan.in_base` into `scan.out_base`, each group
// 1024 of them (4 a thread), its total at `scan.block_base` + its group.
[shader("compute")]
[numthreads(256, 1, 1)]
void fr_scan_blocks(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint t = thread.x;
    uint base = 1024 * group.x + 4 * t;
    uint v[4];
    uint sum = 0;
    for (uint j = 0; j < 4; j++)
    {
        v[j] = base + j < scan.n ? work[scan.in_base + base + j] : 0;
        sum += v[j];
    }
    g_scan[t] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint step = 1; step < 256; step <<= 1)
    {
        uint add = t >= step ? g_scan[t - step] : 0;
        GroupMemoryBarrierWithGroupSync();
        g_scan[t] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint run = g_scan[t] - sum;
    for (uint j = 0; j < 4; j++)
    {
        if (base + j < scan.n)
            work[scan.out_base + base + j] = run;
        run += v[j];
    }
    if (t == 255)
        work[scan.block_base + group.x] = g_scan[255];
}

// The block totals scanned (exclusive), the grand total after them: one group, at most
// 1024 blocks (2^20 words; fracture.rs checks).
[shader("compute")]
[numthreads(1024, 1, 1)]
void fr_scan_top(uint3 thread: SV_GroupThreadID)
{
    uint t = thread.x;
    uint blocks = (scan.n + 1023) / 1024;
    uint v = t < blocks ? work[scan.block_base + t] : 0;
    g_scan[t] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint step = 1; step < 1024; step <<= 1)
    {
        uint add = t >= step ? g_scan[t - step] : 0;
        GroupMemoryBarrierWithGroupSync();
        g_scan[t] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    if (t < blocks)
        work[scan.block_base + t] = g_scan[t] - v;
    if (t == 1023)
        work[scan.block_base + blocks] = g_scan[1023];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_scan_add(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= scan.n)
        return;
    work[scan.out_base + i] += work[scan.block_base + i / 1024];
}

// The free slots in ascending order (`w_free_list`), from the scan of the free marks.
[shader("compute")]
[numthreads(64, 1, 1)]
void fr_freelist(uint3 id: SV_DispatchThreadID)
{
    uint s = id.x;
    if (s >= params.body_count || work[w_free() + s] == 0)
        return;
    work[w_free_list() + work[w_free_rank() + s]] = s;
}

uint free_total()
{
    // `fr_scan_top` leaves the free scan's total after its blocks.
    uint blocks = (params.body_count + 1023) / 1024;
    return work[scan_free_blocks() + blocks];
}

// Where each scan keeps its block totals (after `w_free_list`).
uint scan_new_blocks() { return w_free_list() + params.body_count; }
uint scan_free_blocks() { return scan_new_blocks() + 1025; }

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_assign(uint3 id: SV_DispatchThreadID)
{
    uint c = id.x;
    if (!is_root(c))
        return;
    uint s = body_of(c);
    uint slot = s;
    if (atom[a_best_root() + s].load() != c)
    {
        uint k = work[w_rank() + c];
        if (k < free_total())
        {
            slot = work[w_free_list() + k];
            work[w_free() + slot] = 0;
            atom[a_counters() + C_PIECES].add(1u);
        }
        else
        {
            // No slot (fracture.rs sizes them, so a bug): the piece stays on its parent.
            atom[a_counters() + C_ERROR].or(1u);
        }
    }
    work[w_slot() + c] = slot;
}

uint stage(uint c) { return 2 * params.body_count + 9 * c; }
// Per body slot 3 float4 (the strike): (the shot's centre at the frame's start, the
// shot + 1, 0 for none: a NaN constant, asfloat(NONE), is not valid WGSL); (its velocity relative to the body, t0); (contact point, force).
uint strike_at(uint b) { return 2 * params.body_count + 9 * params.chunk_count + 3 * b; }

// The piece's body: `cook_with`'s mass, centre and principal axes of its chunks, posed in
// its parent's structure frame, moving with the parent's rigid motion.
[shader("compute")]
[numthreads(64, 1, 1)]
void fr_piece(uint3 id: SV_DispatchThreadID)
{
    uint c = id.x;
    if (!is_root(c))
        return;
    uint parent = body_of(c);
    uint pb = BODY_WORDS * parent;
    uint at = a_sums() + SUMS * c;
    float mass = read64(at, params.scale_m);
    float3 s = float3(read64(at + 2, params.scale_s), read64(at + 4, params.scale_s), read64(at + 6, params.scale_s));
    float3 d = s / max(mass, 1e-30);
    float jxx = read64(at + 8, params.scale_j);
    float jyy = read64(at + 10, params.scale_j);
    float jzz = read64(at + 12, params.scale_j);
    float jxy = read64(at + 14, params.scale_j);
    float jxz = read64(at + 16, params.scale_j);
    float jyz = read64(at + 18, params.scale_j);
    // About the piece's centre of mass (the root's centre + d).
    float dd = dot(d, d);
    float3 i0 = float3(jxx - mass * (dd - d.x * d.x), jxy + mass * d.x * d.y, jxz + mass * d.x * d.z);
    float3 i1 = float3(i0.y, jyy - mass * (dd - d.y * d.y), jyz + mass * d.y * d.z);
    float3 i2 = float3(i0.z, i1.z, jzz - mass * (dd - d.z * d.z));
    float3 com = chunks[4 * c].xyz + d;
    Quat principal;
    float3 moments;
    principal_axes(i0, i1, i2, principal, moments);
    bool anchored = atom[a_anchor() + c].load() != 0;
    // The parent's structure frame in the world, and its motion.
    Quat q;
    float3 origin;
    structure_pose(parent, pb, q, origin);
    float3 p_parent = bodies[pb].xyz;
    float3 v_parent = bodies[pb + 3].xyz;
    float3 w_parent = bodies[pb + 4].xyz;
    float3 pos = to_rubble(origin + rotate(q, com));
    Quat orientation = rot_to_rubble(normalized(quat_mul(q, principal)));
    float inv_mass = anchored || !(mass > 0.0) ? 0.0 : 1.0 / mass;
    float3 inv_inertia = float3(0.0);
    if (!anchored)
        inv_inertia = float3(moments.x > 0.0 ? 1.0 / moments.x : 0.0, moments.y > 0.0 ? 1.0 / moments.y : 0.0, moments.z > 0.0 ? 1.0 / moments.z : 0.0);
    float3 velocity = anchored ? float3(0.0) : v_parent + cross(w_parent, pos - p_parent);
    float3 spin = anchored ? float3(0.0) : w_parent;
    uint flags = anchored ? 0u : (FLAG_DYNAMIC | FLAG_AWAKE);
    uint st = stage(c);
    frames[st] = float4(pos, inv_mass);
    frames[st + 1] = quat_vec(orientation);
    frames[st + 2] = float4(inv_inertia, asfloat(flags));
    frames[st + 3] = float4(velocity, 0.0);
    // Its generation: the engine's next step, so no contact cached before the re-parent
    // warm-starts it (world_step.slang `generated < max(generation)`).
    frames[st + 4] = float4(spin, asfloat(params.step));
    frames[st + 5] = float4(0.0);
    frames[st + 6] = float4(0.0, 0.0, 0.0, 1.0);
    frames[st + 7] = float4(com, frames[2 * parent].w);
    frames[st + 8] = quat_vec(principal);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_commit(uint3 id: SV_DispatchThreadID)
{
    uint c = id.x;
    if (!is_root(c))
        return;
    uint slot = work[w_slot() + c];
    uint st = stage(c);
    for (uint region = 0; region < 2; region++)
        for (uint w = 0; w < BODY_WORDS; w++)
            bodies[BODY_WORDS * (slot + region * params.max_bodies) + w] = frames[st + w];
    frames[2 * slot] = frames[st + 7];
    frames[2 * slot + 1] = frames[st + 8];
}

// The chunk's colliders on its piece's body (`member_colliders`), and the ingest's view
// of them: the body's code (its slot) and mass (infinite when anchored).
[shader("compute")]
[numthreads(64, 1, 1)]
void fr_reparent(uint3 id: SV_DispatchThreadID)
{
    uint c = id.x;
    if (c >= params.chunk_count || !dirty(body_of(c)))
        return;
    uint r = work[w_label() + c];
    uint slot = work[w_slot() + r];
    work[w_body() + c] = slot;
    float3 com = frames[2 * slot].xyz;
    Quat to_body = conj(quat_of(frames[2 * slot + 1]));
    float inv_mass = bodies[BODY_WORDS * slot].w;
    float mass = inv_mass > 0.0 ? 1.0 / inv_mass : float_bits(0x7f800000u);
    float4 c2 = chunks[4 * c + 2];
    uint first = asuint(c2.z);
    uint count = asuint(c2.w);
    // The piece's first chunk (its root) gives the ingest its modulus and radius.
    float4 root = chunks[4 * r + 3];
    for (uint p = first; p < first + count; p++)
    {
        float4 o = parts[2 * p];
        uint col = asuint(o.w);
        Quat rot = quat_of(parts[2 * p + 1]);
        float4 head = colliders[4 * col];
        colliders[4 * col] = float4(asfloat(slot), head.y, head.z, head.w);
        colliders[4 * col + 1] = float4(rotate(to_body, o.xyz - com), colliders[4 * col + 1].w);
        colliders[4 * col + 2] = quat_vec(normalized(quat_mul(to_body, rot)));
        uint4 a = collider_info[2 * col];
        collider_info[2 * col] = uint4(a.x, slot, a.z, a.w);
        collider_info[2 * col + 1] = uint4(asuint(mass), asuint(root.x), asuint(root.w), 0u);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_body_clear(uint3 id: SV_DispatchThreadID)
{
    uint s = id.x;
    if (s >= params.body_count)
        return;
    atom[a_dirty() + s].store(0);
}

// ---- The view (stress-rubble render.rs objects and bond lines, written where they are) ----
//
//   fr_view_chunks  one thread per chunk: its object's rows (render.rs `Object`: rotation
//                   rows and translation, the stress frame) from its body, and its colour by
//                   `view.mode`: 0 its material (a free body's mixed with its slot's colour),
//                   1 its body's state (moving orange, at rest blue, static grey), 2 its body
//                   slot's colour (stable: a split's largest piece keeps its slot);
//   fr_view_shots   one thread per shot: its sphere at its body (hidden while parked);
//   fr_view_bonds   one thread per bond: its line between its chunks' centres, or nothing
//                   once it broke (both ends at one point).

struct ViewParams
{
    uint mode;
    uint shot_count;
    uint impactor_base;  // the renderer's first shot object
    uint pad0;
    float rest_speed;    // render.rs `REST_SPEED`
    float pad1;
    float pad2;
    float pad3;
};

[[vk::binding(16, 0)]] ConstantBuffer<ViewParams> view;
// Per chunk 3 float4: (object as bits, render centre (structure frame)); its rotation
// (structure frame); (material colour, -).
[[vk::binding(17, 0)]] StructuredBuffer<float4> view_chunks;
// The renderer's objects (4 float4: 3 rows, colour) and bond-line vertices (7 floats).
[[vk::binding(18, 0)]] RWStructuredBuffer<float4> objects;
[[vk::binding(19, 0)]] RWStructuredBuffer<float> lines;
// The shots' engine bodies.
[[vk::binding(20, 0)]] StructuredBuffer<uint> shot_bodies;
// Per chunk 3 float4 (the strike's): (box centre (structure frame), bounding radius); (half
// extents, plane-strain modulus E'); box rotation (structure frame).
[[vk::binding(21, 0)]] StructuredBuffer<float4> geometry;
// Per shot 2 float4: (engine body as bits, bounding radius, face radius, mass); (E', -, -, -).
[[vk::binding(22, 0)]] StructuredBuffer<float4> shots;

static const float3 MOVING = float3(1.0, 0.45, 0.12);
static const float3 AT_REST = float3(0.25, 0.45, 0.85);
static const float3 STATIC = float3(0.45, 0.45, 0.48);
static const float4 BOND_LINE = float4(1.0, 0.86, 0.25, 1.0);

// render.rs `slot_color`: hues a golden-ratio step apart.
float3 slot_color(uint slot)
{
    float h = frac(float(slot) * 0.6180339887) * 6.0;
    float s = 0.62, v = 0.92, f = frac(h);
    float p = v * (1.0 - s), q = v * (1.0 - s * f), t = v * (1.0 - s * (1.0 - f));
    uint k = uint(h);
    return k == 0u ? float3(v, t, p) : k == 1u ? float3(q, v, p) : k == 2u ? float3(p, v, t)
         : k == 3u ? float3(p, q, v) : k == 4u ? float3(t, p, v) : float3(v, p, q);
}

void write_object(uint o, Quat q, float3 t, float4 color)
{
    float3 x = rotate(q, float3(1.0, 0.0, 0.0));
    float3 y = rotate(q, float3(0.0, 1.0, 0.0));
    float3 z = rotate(q, float3(0.0, 0.0, 1.0));
    objects[4 * o] = float4(x.x, y.x, z.x, t.x);
    objects[4 * o + 1] = float4(x.y, y.y, z.y, t.y);
    objects[4 * o + 2] = float4(x.z, y.z, z.z, t.z);
    objects[4 * o + 3] = color;
}

// A chunk's centre in the world (the stress frame).
float3 chunk_world(uint k)
{
    uint slot = body_of(k);
    Quat q;
    float3 origin;
    structure_pose(slot, BODY_WORDS * slot, q, origin);
    return origin + rotate(q, view_chunks[3 * k].xyz);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_view_chunks(uint3 id: SV_DispatchThreadID)
{
    uint k = id.x;
    if (k >= params.chunk_count)
        return;
    uint slot = body_of(k);
    uint b = BODY_WORDS * slot;
    Quat q;
    float3 origin;
    structure_pose(slot, b, q, origin);
    float4 v0 = view_chunks[3 * k];
    float3 base = view_chunks[3 * k + 2].xyz;
    bool fixed = bodies[b].w == 0.0;
    float3 color;
    if (view.mode == 1)
    {
        float speed = max(length(bodies[b + 3].xyz), length(bodies[b + 4].xyz));
        color = fixed ? STATIC : (speed > view.rest_speed ? MOVING : AT_REST);
    }
    else if (view.mode == 2)
        color = slot_color(slot);
    else
        color = fixed ? base : 0.45 * base + 0.55 * slot_color(slot);
    write_object(asuint(v0.w), normalized(quat_mul(q, quat_of(view_chunks[3 * k + 1]))), origin + rotate(q, v0.xyz), float4(color, 1.0));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_view_shots(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= view.shot_count)
        return;
    uint b = BODY_WORDS * shot_bodies[i];
    uint o = view.impactor_base + i;
    if (bodies[b].w == 0.0)
    {
        for (uint w = 0; w < 4; w++)
            objects[4 * o + w] = float4(0.0);
        return;
    }
    write_object(o, rot_from_rubble(quat_of(bodies[b + 1])), from_rubble(bodies[b].xyz), float4(0.18, 0.19, 0.22, 1.0));
}

void write_line_vertex(uint v, float3 p)
{
    lines[7 * v] = p.x;
    lines[7 * v + 1] = p.y;
    lines[7 * v + 2] = p.z;
    lines[7 * v + 3] = BOND_LINE.x;
    lines[7 * v + 4] = BOND_LINE.y;
    lines[7 * v + 5] = BOND_LINE.z;
    lines[7 * v + 6] = BOND_LINE.w;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_view_bonds(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.bond_count)
        return;
    uint4 bond = bonds[i];
    float3 a = chunk_world(bond.x);
    float3 b = work[w_alive() + i] != 0 ? chunk_world(bond.y) : a;
    write_line_vertex(2 * i, a);
    write_line_vertex(2 * i + 1, b);
}


// ---- Strike prediction (threshold.slang `threshold_predict`, strike.slang `find_strike`) ----
//
// A shot that will reach a body within the next frame, closing faster than
// `strike_speed`, breaks every intact bond of it whose share of the strike's Hertz force
// (a point load on a half-space: F min(1, A / (2 pi d^2)) at distance d) beats its tensile
// capacity, before the engine steps: forward only, the step meets the pieces.
//
//   fr_strike_reset    one thread per body: no bounds, no strike;
//   fr_strike_bounds   one thread per chunk: its body's bounds (chunk centres +- bounding
//                      radius);
//   fr_strike_find     one thread per body: the first shot (in order) whose sphere reaches
//                      its bounds within the frame;
//   fr_strike_entry    one thread per chunk of a struck body: the first time the shot meets
//                      its box, and its distance to the shot's path;
//   fr_strike_nearest  the struck chunk: the nearest to the path (ties: the lowest);
//   fr_strike_force    one thread per struck body: the contact point and the Hertz force;
//   fr_strike_breaks   one thread per bond.

uint ordered(float f)
{
    uint u = asuint(f);
    return (u & 0x80000000u) != 0 ? ~u : (u | 0x80000000u);
}

float unordered(uint u)
{
    return asfloat((u & 0x80000000u) != 0 ? (u & 0x7fffffffu) : ~u);
}

// A chunk's box centre in the world (the stress frame), and its body's rotation.
float3 chunk_box_world(uint k, out Quat q)
{
    uint slot = body_of(k);
    float3 origin;
    structure_pose(slot, BODY_WORDS * slot, q, origin);
    return origin + rotate(q, geometry[3 * k].xyz);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_strike_reset(uint3 id: SV_DispatchThreadID)
{
    uint b = id.x;
    if (b >= params.body_count)
        return;
    uint at = a_strike() + 9 * b;
    for (uint w = 0; w < 3; w++)
    {
        atom[at + w].store(0xffffffffu);
        atom[at + 3 + w].store(0u);
    }
    atom[at + 6].store(0x7f800000u);
    atom[at + 7].store(0x7f800000u);
    atom[at + 8].store(NONE);
    frames[strike_at(b)].w = 0.0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_strike_bounds(uint3 id: SV_DispatchThreadID)
{
    uint k = id.x;
    if (k >= params.chunk_count || params.shot_count == 0)
        return;
    Quat q;
    float3 c = chunk_box_world(k, q);
    float r = geometry[3 * k].w;
    uint at = a_strike() + 9 * body_of(k);
    for (uint ax = 0; ax < 3; ax++)
    {
        atom[at + ax].min(ordered(c[ax] - r));
        atom[at + 3 + ax].max(ordered(c[ax] + r));
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_strike_find(uint3 id: SV_DispatchThreadID)
{
    uint b = id.x;
    if (b >= params.body_count || params.shot_count == 0)
        return;
    uint at = a_strike() + 9 * b;
    if (atom[at].load() == 0xffffffffu)
        return; // no chunk on this body
    float3 lo = float3(unordered(atom[at].load()), unordered(atom[at + 1].load()), unordered(atom[at + 2].load()));
    float3 hi = float3(unordered(atom[at + 3].load()), unordered(atom[at + 4].load()), unordered(atom[at + 5].load()));
    float3 v = from_rubble(bodies[BODY_WORDS * b + 3].xyz);
    float inf = float_bits(0x7f800000u);
    for (uint k = 0; k < params.shot_count; k++)
    {
        float4 info = shots[2 * k];
        uint sb = BODY_WORDS * asuint(info.x);
        if (bodies[sb].w == 0.0)
            continue; // parked
        float3 c = from_rubble(bodies[sb].xyz);
        float3 rel = from_rubble(bodies[sb + 3].xyz) - v;
        float sp = length(rel);
        if (!(sp > params.strike_speed))
            continue;
        float r = info.y;
        float3 elo = lo - float3(r);
        float3 ehi = hi + float3(r);
        float t0 = 0.0;
        float t1 = params.dt;
        for (uint ax = 0; ax < 3; ax++)
        {
            if (abs(rel[ax]) < 1e-12)
            {
                if (c[ax] < elo[ax] || c[ax] > ehi[ax])
                    t0 = inf;
                continue;
            }
            float ta = (elo[ax] - c[ax]) / rel[ax];
            float tb = (ehi[ax] - c[ax]) / rel[ax];
            t0 = max(t0, min(ta, tb));
            t1 = min(t1, max(ta, tb));
        }
        if (t0 <= t1)
        {
            uint st = strike_at(b);
            frames[st] = float4(c, float(k + 1));
            frames[st + 1] = float4(rel, t0);
            return;
        }
    }
}

bool struck(uint b, out uint shot)
{
    uint k = uint(frames[strike_at(b)].w);
    shot = k - 1;
    return k != 0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_strike_entry(uint3 id: SV_DispatchThreadID)
{
    uint k = id.x;
    if (k >= params.chunk_count || params.shot_count == 0)
        return;
    uint b = body_of(k);
    uint shot;
    if (!struck(b, shot))
        return;
    float4 s0 = frames[strike_at(b)];
    float4 s1 = frames[strike_at(b) + 1];
    float3 from = s0.xyz;
    float3 rel = s1.xyz;
    float3 d = rel / max(length(rel), 1e-12);
    float radius = shots[2 * shot].y;
    Quat q;
    float3 cc = chunk_box_world(k, q);
    Quat box = normalized(quat_mul(q, quat_of(geometry[3 * k + 2])));
    float3 half = geometry[3 * k + 1].xyz;
    float inf = float_bits(0x7f800000u);
    float t_in = 0.0;
    float t_out = params.dt;
    for (uint ax = 0; ax < 3; ax++)
    {
        float3 e = float3(0.0);
        e[ax] = 1.0;
        float3 axis = rotate(box, e);
        float o = dot(from - cc, axis);
        float dv = dot(rel, axis);
        float ext = half[ax] + radius;
        if (abs(dv) < 1e-12)
        {
            if (abs(o) > ext)
                t_in = inf;
            continue;
        }
        float ta = (-ext - o) / dv;
        float tb = (ext - o) / dv;
        t_in = max(t_in, min(ta, tb));
        t_out = min(t_out, max(ta, tb));
    }
    uint at = a_strike() + 9 * b;
    if (t_in <= t_out)
        atom[at + 6].min(asuint(t_in));
    // The struck chunk: the nearest to the path from where it meets the bounds.
    float3 hit = from + rel * s1.w;
    float3 r = cc - hit;
    float along = dot(r, d);
    float dist = length(r - d * along) + abs(along) * 1e-3;
    atom[at + 7].min(asuint(dist));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_strike_nearest(uint3 id: SV_DispatchThreadID)
{
    uint k = id.x;
    if (k >= params.chunk_count || params.shot_count == 0)
        return;
    uint b = body_of(k);
    uint shot;
    if (!struck(b, shot))
        return;
    float4 s0 = frames[strike_at(b)];
    float4 s1 = frames[strike_at(b) + 1];
    float3 d = s1.xyz / max(length(s1.xyz), 1e-12);
    Quat q;
    float3 r = chunk_box_world(k, q) - (s0.xyz + s1.xyz * s1.w);
    float along = dot(r, d);
    float dist = length(r - d * along) + abs(along) * 1e-3;
    uint at = a_strike() + 9 * b;
    if (asuint(dist) == atom[at + 7].load())
        atom[at + 8].min(k);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_strike_force(uint3 id: SV_DispatchThreadID)
{
    uint b = id.x;
    if (b >= params.body_count || params.shot_count == 0)
        return;
    uint shot;
    if (!struck(b, shot))
        return;
    uint at = a_strike() + 9 * b;
    float t_in = asfloat(atom[at + 6].load());
    uint entry = atom[at + 8].load();
    uint st = strike_at(b);
    if (!(t_in <= params.dt) || entry == NONE)
    {
        frames[st].w = 0.0; // the bounds were reached, no box: no strike
        return;
    }
    float4 s0 = frames[st];
    float4 s1 = frames[st + 1];
    float3 rel = s1.xyz;
    float speed = length(rel);
    float3 d = rel / max(speed, 1e-12);
    float4 info = shots[2 * shot];
    float radius = info.y;
    float face = max(info.z, 1e-3);
    float m_imp = info.w;
    float e_imp = shots[2 * shot + 1].x;
    float3 contact = s0.xyz + rel * t_in + d * radius;
    float inv_mass = bodies[BODY_WORDS * b].w;
    bool anchored = inv_mass == 0.0;
    float m_body = anchored ? 0.0 : 1.0 / inv_mass;
    float m_eff = anchored ? m_imp : m_imp * m_body / (m_imp + m_body);
    float modulus = 1.0 / (1.0 / e_imp + 1.0 / geometry[3 * entry + 1].w);
    // api.rs `hertz_duration`, as `punch_islands` takes it.
    float tau = 2.87 * pow(m_eff * m_eff / (face * modulus * modulus * max(speed, 1e-3)), 0.2);
    frames[st + 2] = float4(contact, m_eff * speed / tau);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_strike_breaks(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.bond_count || params.shot_count == 0 || work[w_alive() + i] == 0)
        return;
    uint4 bond = bonds[i];
    uint b = body_of(bond.x);
    uint shot;
    if (!struck(b, shot))
        return;
    float4 f = frames[strike_at(b) + 2];
    Quat q;
    float3 pa = chunk_box_world(bond.x, q);
    float3 pb = chunk_box_world(bond.y, q);
    float dist = length(0.5 * (pa + pb) - f.xyz);
    float area = asfloat(bond.w);
    float reach = f.w * min(1.0, area / (6.28318530718 * max(dist * dist, 1e-12)));
    if (reach > asfloat(bond.z))
    {
        work[w_alive() + i] = 0;
        atom[a_dirty() + b].or(1u);
        atom[a_counters() + C_FRAME_BROKEN].add(1u);
        atom[a_counters() + C_BROKEN].add(1u);
    }
}
