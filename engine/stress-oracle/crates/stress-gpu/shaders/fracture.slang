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
    uint pulse_cap;
    uint impact_peak;
    uint pad0;
    float dt;
    float band;          // |g| dt: closing speeds within it are not impacts
    float scale_m;       // fixed-point scales of the mass sums (2^k)
    float scale_s;
    float scale_j;
    float pad1;
    float pad2;
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
// (the scans' block sums after it)

// `atom` layout.
uint a_parent() { return 0; }
uint a_count() { return params.chunk_count; }
uint a_anchor() { return 2 * params.chunk_count; }
uint a_sums() { return 3 * params.chunk_count; }
uint a_dirty() { return a_sums() + SUMS * params.chunk_count; }
uint a_best_count() { return a_dirty() + params.body_count; }
uint a_best_root() { return a_best_count() + params.body_count; }
uint a_counters() { return a_best_root() + params.body_count; }

// Counters (`a_counters` + i).
static const uint C_FRAME_BROKEN = 0; // bonds broken this frame (fr_gate clears it)
static const uint C_BROKEN = 1;       // bonds broken in all
static const uint C_PIECES = 2;       // new bodies in all
static const uint C_ERROR = 3;        // bits: 1 no free slot, 2 a mass sum out of range, 4 a union did not settle, 8 a pulse found no room
static const uint C_SPLIT_FRAMES = 4;

// `loads` layout: per chunk 1 float4 (its load), then the pulses.
uint l_pulse_base() { return params.chunk_count; }
uint l_pulse_count_base() { return params.chunk_count + 3 * params.pulse_cap * params.chunk_count; }

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
    chunk_engine_load(manifolds, sort, loads, params.chunk_count, k, m_body, c0.w, centre, chunks[4 * k + 3], params.band, params.dt, params.impact_peak != 0, l_pulse_base(), l_pulse_count_base(), params.pulse_cap, f_total, t_total, overflow);
    if (overflow)
        atom[a_counters() + C_ERROR].or(8u);
    loads[k] = float4(length(f_total), 0.0, 0.0, 0.0);
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
