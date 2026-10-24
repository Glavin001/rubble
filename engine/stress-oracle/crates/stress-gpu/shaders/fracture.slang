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
import joint;
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
    // The load and strike models (stress-ref `capability::LoadModel`, `StrikeModel`).
    uint load_model;     // 0 threshold (`fr_breaks`), 1 load path (`fr_lp_*`)
    uint strike_model;   // 0 Hertz (`fr_strike_breaks`), 1 analytic (`fr_il_*`)
    uint fracture_work;  // the analytic model's: a perforating shot gives up the ballistic limit's energy
    uint sweeps;         // the load path's sweeps a frame (even)
    float gx;            // gravity (the stress frame)
    float gy;
    float gz;
    float lp_c1;         // the load path's recurrence (stress-ref load_path.rs `momentum_coefficients`)
    float lp_c2;
    uint ext_chunks;     // `ext` offsets (float4s): the chunks' records, their bonds' list, the materials
    uint ext_adj;
    uint ext_mat;
    float lp_scale;      // fixed-point scale of the load path's per-body sums
    uint cs_sweeps;      // the in-step contact fracture: on (> 0), the load path's sweeps (even)
    uint cs_row_cap;     // the engine's contact rows, at most
    float cs_c1;         // its recurrence (load_path.rs `momentum_coefficients(IMPACT_LOWER)`)
    float cs_c2;
    uint cs_row_offset;  // this step's first contact row (rubble's contacts' parity)
    float pad7;
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
// Per chunk its load (x: |force|), then its carried pulses (impact.slang), then (the load
// path's) per chunk its force and moment (world, about its centre) as 2 float4.
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
// Per chunk: the launch (`shots`) whose analytic strike broke it out (plug, scab, crater),
// 0 for none (`fr_il_mark`).
uint w_marks() { return w_pulses() + 4 * params.chunk_count; }
// Per chunk: the struck body slot + 1 whose plug it came out in this frame, 0 for none
// (`fr_il_mark` sets it, `fr_il_carry` clears it).
uint w_carry() { return w_marks() + params.chunk_count; }

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
static const uint C_ERROR = 3;        // bits: 1 no free slot, 2 a mass sum out of range, 4 a union did not settle, 8 a pulse found no room,
                                      // 32 a load-path block was not positive definite (16 is no longer set)
static const uint C_SPLIT_FRAMES = 4;
static const uint C_POOL_TOP = 5;     // + the half: pulses claimed in it this frame (5, 6)
// The analytic strikes' outcomes, in all (`fr_il_strike`), and the load path's largest
// failure index since the last read (float bits, `fr_lp_breaks`).
static const uint C_IL_BOUNCE = 7;
static const uint C_IL_CRATER = 8;
static const uint C_IL_SPALL = 9;
static const uint C_IL_PERFORATE = 10;
static const uint C_LP_PEAK = 11;
// The in-step contact fracture: strikes it solved and bonds they broke, in all; whether
// this step's struck structures and their chunks listed.
static const uint C_CS_STRIKES = 12;
static const uint C_CS_BREAKS = 13;
static const uint C_CS_BODIES = 14; // this step's struck structures (`fr_cs_record`)
static const uint C_CS_CURSOR = 15; // their chunks listed so far (`fr_cs_base`)

// `loads` layout: per chunk 1 float4 (its load), then the two halves of the pulse pool
// (`pool_cap` pulses of 3 float4 each). A frame reads its chunks' carried pulses from one
// half and writes the ones that last into the other: each chunk claims a run there with
// one atomic add, where it lands does not matter (each chunk reads only its own, in
// order), and the pool's size follows the engine's contact records (fracture.rs), not a
// per-chunk count.
uint l_pool(uint half) { return params.chunk_count + 3 * params.pool_cap * half; }
uint l_vec() { return params.chunk_count + 6 * params.pool_cap; }

// Indirect sizes (`args`, 3 words each).
static const uint ARG_CHUNKS = 0;
static const uint ARG_BONDS = 1;
static const uint ARG_BODIES = 2;
static const uint ARG_SCAN_CHUNKS = 3;
static const uint ARG_SCAN_BODIES = 4;
static const uint ARG_ONE = 5;
// The in-step contact fracture's (`fr_cs_begin`, `fr_cs_gate`, `fr_cs_gate2`): without a
// strike every kernel past the detection dispatches no group.
static const uint ARG_CS_BODIES = 6;  // the load path's solve: a group per struck structure
static const uint ARG_CS_ROWS = 7;    // the detection: the step's contact rows
static const uint ARG_CS_PICK = 8;    // the step's contact rows when a structure is struck
static const uint ARG_CS_LIST = 9;    // the struck structures
static const uint ARG_CS_CHUNKS = 10; // every chunk when a structure is struck
static const uint ARG_CS_BONDS = 11;  // every bond when a structure is struck
static const uint ARG_CS_SLOTS = 12;  // every body slot when a structure is struck
static const uint ARG_CS_SAVE = 13;   // the step's contact rows when a bond broke

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
    loads[k] = float4(length(f_total), 0.0, 0.0, 0.0);
    if (params.load_model == 1)
    {
        loads[l_vec() + 2 * k] = float4(f_total, 0.0);
        loads[l_vec() + 2 * k + 1] = float4(t_total, 0.0);
    }
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
// Per shot 2 float4: (engine body as bits, bounding radius, face radius, mass); (E', its
// launch as bits (1, 2, ...: which shot in flight it is, `fr_il_mark`), -, -).
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

// Whether `bond` breaks under a strike of Hertz force `force` at `contact` (the world): its
// share, a point load on a half-space at the distance of its chunks' midpoint, `F min(1, A /
// (2 pi d^2))`, beats its tensile capacity.
bool strike_share_breaks(uint4 bond, float3 contact, float force)
{
    Quat q;
    float3 pa = chunk_box_world(bond.x, q);
    float3 pb = chunk_box_world(bond.y, q);
    float dist = length(0.5 * (pa + pb) - contact);
    float area = asfloat(bond.w);
    float reach = force * min(1.0, area / (6.28318530718 * max(dist * dist, 1e-12)));
    return reach > asfloat(bond.z);
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
    if (strike_share_breaks(bond, f.xyz, f.w))
    {
        work[w_alive() + i] = 0;
        atom[a_dirty() + b].or(1u);
        atom[a_counters() + C_FRAME_BROKEN].add(1u);
        atom[a_counters() + C_BROKEN].add(1u);
    }
}


// ---- Load path (stress-ref load_path.rs, `capability::LoadModel::LoadPath`) ----
//
// The bond graph's quasi-static equilibrium `K x = f`, each chunk's small displacement
// `x = (u, theta)` (structure frame) carried from frame to frame, solved by `sweeps` sweeps
// a frame of block Jacobi accelerated by Chebyshev's recurrence at its limit (`d = c1 d +
// c2 D^-1 r`, `x += d`). The loads: gravity, the engine's contact forces and moments
// (`fr_loads`), and on an unsupported dynamic body the d'Alembert loads of its rigid
// acceleration (inertia relief: its loads balance). Supports hold still; so does an
// unsupported body's lowest chunk. Then every intact bond's force is checked against its
// strengths (joint.slang `failure_indices`): tension, shear, compression, buckling.
//
//   fr_lp_clear        one thread per body: its sums;
//   fr_lp_sum          one thread per chunk: its load (structure frame), its body's sums
//                      (force and moment about its centre of mass, 64-bit fixed point), its
//                      body's lowest chunk and whether it is supported;
//   fr_lp_relief       one thread per chunk: the relieved load, whether it holds still;
//   fr_lp_block        one thread per chunk: its diagonal block's Cholesky factor;
//   fr_lp_sweep_even, fr_lp_sweep_odd
//                      one thread per chunk: a sweep from one copy of `x` to the other;
//   fr_lp_breaks       one thread per bond.
//
// `ext` (read only; `fracture.rs` `ExtTables`): per bond 10 float4 (centroid, normal, t1,
// then its JointBond's 7), then from `ext_chunks` per chunk 4 float4 ((first, count) of its
// bonds in `ext_adj` as bits; the analytic strike's material: (density, tensile,
// compressive, shear); (DIF reference rate, exponent, transition rate, high exponent);
// (DIF max (0: none), Poisson's ratio, Young's modulus, -)), the bonds' list from
// `ext_adj` (4 to a float4, as bits), the JointMaterials from `ext_mat` (5 float4 each).
[[vk::binding(23, 0)]] StructuredBuffer<float4> ext;
// Per chunk 22 float4: the load path's state (`x` twice (2 each), the step `d` (2), its
// load (2: force and fixed flag, moment)), its block's factor (6: 21 floats), then the
// in-step contact fracture's increment (`fr_cs_*`: its `x` twice, `d`, load, as the load
// path's); then per body slot 7 float4: the analytic strike's record (`fr_il_strike`, 2),
// the in-step contact strike's (`fr_cs_record`, 5).
[[vk::binding(24, 0)]] RWStructuredBuffer<float4> lp;
// The engine's step (the in-step contact fracture, encoded within rubble's step where its
// contacts are made and kept): its contact rows (rubble's `Contact`, 8 float4 each: the
// colliders, feature and alive flag as bits; normal (side 0 to side 1), penetration; anchor
// a (body frame), c0 n; anchor b, friction; tangent 1, lambda n; tangent 2, lambda t1; kN,
// kT, lambda t2, c0 t1; c0 t2, stick, step, -; this step's from `cs_row_offset`) and its
// counters (`StepCounters`: word 2 the contact rows' count).
[[vk::binding(25, 0)]] RWStructuredBuffer<float4> engine_contacts;
[[vk::binding(26, 0)]] StructuredBuffer<uint> engine_counters;
// Per contact row 2 float4: where each side on a chunk of a broken structure touches, in
// the world (w 1; `fr_cs_save`), for `fr_cs_fix` after the split.
[[vk::binding(27, 0)]] RWStructuredBuffer<float4> cs_anchors;
// Per entry of `ext_adj` (a chunk's bond, in its chunk's order) 5 float4, what a sweep
// takes of the bond (`fracture.rs` `sides`): (the bond's centroid less its first chunk's
// centre, the bond as bits); (less its second chunk's, the other chunk as bits); (t1, ks);
// (normal, kn); (kb_t1, kb_t2, kt, 1 when this chunk is the bond's first). The same values
// `lp_bond` and `st_force` make, read in one place. (Declared in binding order: Slang's
// Metal output numbers its buffers in the order of declaration.)
[[vk::binding(28, 0)]] StructuredBuffer<float4> lp_sides;

// An engine collider of a body (the ingest's `ColliderInfo` kind, coupling.slang `KIND_BODY`).
static const uint KIND_BODY_COLLIDER = 2;

static const uint LP_CHUNK = 22;
// A state's place in a chunk's record: the load path's, and the contact fracture's
// increment.
static const uint LP_STATE = 0;
static const uint CS_STATE = 14;
uint st_x(uint k, uint state, uint copy) { return LP_CHUNK * k + state + 2 * copy; }
uint st_d(uint k, uint state) { return LP_CHUNK * k + state + 4; }
uint st_f(uint k, uint state) { return LP_CHUNK * k + state + 6; }
uint lp_x(uint k, uint copy) { return st_x(k, LP_STATE, copy); }
uint lp_d(uint k) { return st_d(k, LP_STATE); }
uint lp_f(uint k) { return st_f(k, LP_STATE); }
uint lp_l(uint k) { return LP_CHUNK * k + 8; }
uint lp_body(uint b) { return LP_CHUNK * params.chunk_count + 7 * b; }

// `atom`: per body slot 14 words of the load path's sums (force, moment: 3 x 64 bits each;
// lowest chunk; supported), then per body slot 6 words of the analytic strike: its run's
// end along the path, its tube's count, its plug's mass (64-bit fixed point, `scale_m`),
// its tube's start in the list and fill (`fr_il_tube_*`).
uint a_lp(uint b) { return a_strike() + 9 * params.body_count + 14 * b; }
uint a_il(uint b) { return a_strike() + 23 * params.body_count + 6 * b; }
// Per shot (`MAX_SHOTS`): the earliest strike time of this frame's strikes (float bits).
uint a_first(uint shot) { return a_strike() + 29 * params.body_count + shot; }
// The tubes listed so far (`fr_il_tube_base`), after the shots' words.
uint a_il_cursor() { return a_first(0) + 64; }
// `work`: the struck bodies' tubes (2 words per chunk: its interval along the path), listed
// body by body, after the in-step lists.
uint w_il_tube() { return w_carry() + 2 * params.chunk_count + params.body_count; }

// `add64` for terms up to 2^62 (a load's): the magnitude split into two words exactly (it
// has 24 significant bits), negated as a 64-bit integer.
void add64w(uint at, float value, float scale)
{
    float x = round(value * scale);
    if (!(abs(x) < 4.0e18))
    {
        atom[a_counters() + C_ERROR].or(2u);
        x = 0.0;
    }
    float u = abs(x);
    float h = floor(u * (1.0 / 4294967296.0));
    uint lo = uint(u - h * 4294967296.0);
    uint hi = uint(h);
    if (x < 0.0)
    {
        lo = ~lo + 1u;
        hi = ~hi + (lo == 0u ? 1u : 0u);
    }
    uint old;
    old = atom[at].add(lo);
    uint carry = old + lo < old ? 1u : 0u;
    hi += carry;
    if (hi != 0)
        atom[at + 1].add(hi);
}

float3 read64w3(uint at, float scale)
{
    return float3(read64(at, scale), read64(at + 2, scale), read64(at + 4, scale));
}

struct LpBond
{
    float3 centroid;
    float3 n;
    float3 t1;
    float3 t2;
    JointBond law;
};

LpBond lp_bond(uint i)
{
    uint o = 10 * i;
    LpBond b;
    b.centroid = ext[o].xyz;
    b.n = ext[o + 1].xyz;
    b.t1 = ext[o + 2].xyz;
    b.t2 = cross(b.n, b.t1);
    b.law.geom0 = ext[o + 3];
    b.law.geom1 = ext[o + 4];
    b.law.stiff0 = ext[o + 5];
    b.law.stiff1 = ext[o + 6];
    b.law.rebar0 = ext[o + 7];
    b.law.rebar1 = ext[o + 8];
    b.law.ids = asuint(ext[o + 9]);
    return b;
}

JointMaterial lp_material(uint m)
{
    uint o = params.ext_mat + 5 * m;
    JointMaterial j;
    j.strength = ext[o];
    j.energy = ext[o + 1];
    j.dif = ext[o + 2];
    j.misc = ext[o + 3];
    j.kind_flags = asuint(ext[o + 4]);
    return j;
}

uint adj_bond(uint at)
{
    return asuint(ext[params.ext_adj + at / 4][at % 4]);
}

// Bond `i`'s generalised force (local: shear t1, t2, axial; bending t1, t2, torsion),
// tension positive, from the `x` copy `c` (load_path.rs `Network::force`).
void lp_force(LpBond b, uint ka, uint kb, uint c, out float3 q_lin, out float3 q_ang)
{
    st_force(b, ka, kb, LP_STATE, c, q_lin, q_ang);
}

// As `lp_force`, from state `state`'s copy `c`.
void st_force(LpBond b, uint ka, uint kb, uint state, uint c, out float3 q_lin, out float3 q_ang)
{
    float3 ra = b.centroid - chunks[4 * ka].xyz;
    float3 rb = b.centroid - chunks[4 * kb].xyz;
    float4 xa0 = lp[st_x(ka, state, c)];
    float4 xa1 = lp[st_x(ka, state, c) + 1];
    float4 xb0 = lp[st_x(kb, state, c)];
    float4 xb1 = lp[st_x(kb, state, c) + 1];
    float3 delta = (xb0.xyz + cross(xb1.xyz, rb)) - (xa0.xyz + cross(xa1.xyz, ra));
    float3 phi = xb1.xyz - xa1.xyz;
    float4 k0 = b.law.stiff0; // kn, ks, kb_t1, kb_t2
    float kt = b.law.stiff1.x;
    q_lin = float3(k0.y * dot(delta, b.t1), k0.y * dot(delta, b.t2), k0.x * dot(delta, b.n));
    q_ang = float3(k0.z * dot(phi, b.t1), k0.w * dot(phi, b.t2), kt * dot(phi, b.n));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_lp_clear(uint3 id: SV_DispatchThreadID)
{
    uint b = id.x;
    if (b >= params.body_count)
        return;
    uint at = a_lp(b);
    for (uint w = 0; w < 12; w++)
        atom[at + w].store(0u);
    atom[at + 12].store(NONE);
    atom[at + 13].store(0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_lp_sum(uint3 id: SV_DispatchThreadID)
{
    uint k = id.x;
    if (k >= params.chunk_count)
        return;
    uint slot = body_of(k);
    Quat q;
    float3 origin;
    structure_pose(slot, BODY_WORDS * slot, q, origin);
    Quat qi = conj(q);
    float4 c0 = chunks[4 * k];
    float3 f = rotate(qi, loads[l_vec() + 2 * k].xyz) + c0.w * rotate(qi, float3(params.gx, params.gy, params.gz));
    float3 t = rotate(qi, loads[l_vec() + 2 * k + 1].xyz);
    lp[lp_f(k)] = float4(f, 0.0);
    lp[lp_f(k) + 1] = float4(t, 0.0);
    float3 r = c0.xyz - frames[2 * slot].xyz;
    float3 m = t + cross(r, f);
    uint at = a_lp(slot);
    for (uint a = 0; a < 3; a++)
    {
        add64w(at + 2 * a, f[a], params.lp_scale);
        add64w(at + 6 + 2 * a, m[a], params.lp_scale);
    }
    atom[at + 12].min(k);
    if ((info[k] & INFO_SUPPORT) != 0)
        atom[at + 13].store(1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_lp_relief(uint3 id: SV_DispatchThreadID)
{
    uint k = id.x;
    if (k >= params.chunk_count)
        return;
    uint slot = body_of(k);
    uint at = a_lp(slot);
    bool supported = atom[at + 13].load() != 0;
    bool fixed = (info[k] & INFO_SUPPORT) != 0 || (!supported && atom[at + 12].load() == k);
    float4 f = lp[lp_f(k)];
    float4 t = lp[lp_f(k) + 1];
    uint b = BODY_WORDS * slot;
    float inv_mass = bodies[b].w;
    if (!supported && inv_mass > 0.0)
    {
        float3 force = read64w3(at, params.lp_scale);
        float3 moment = read64w3(at + 6, params.lp_scale);
        // The body's inverse inertia (principal, its record's) in the structure frame.
        Quat p = quat_of(frames[2 * slot + 1]);
        float3 alpha = rotate(p, bodies[b + 2].xyz * rotate(conj(p), moment));
        float3 acc = force * inv_mass;
        float4 c0 = chunks[4 * k];
        float4 c1 = chunks[4 * k + 1];
        float4 c2 = chunks[4 * k + 2];
        float3 r = c0.xyz - frames[2 * slot].xyz;
        f.xyz -= c0.w * (acc + cross(alpha, r));
        // The chunk's own inertia about its centre: (xx, yy, zz, xy), (xz, yz).
        float3 ia = float3(c1.x * alpha.x + c1.w * alpha.y + c2.x * alpha.z, c1.w * alpha.x + c1.y * alpha.y + c2.y * alpha.z, c2.x * alpha.x + c2.y * alpha.y + c1.z * alpha.z);
        t.xyz -= ia;
    }
    f.w = fixed ? 1.0 : 0.0;
    lp[lp_f(k)] = f;
    lp[lp_f(k) + 1] = t;
}

// Packed lower-triangular index.
uint tri(uint i, uint j) { return i * (i + 1) / 2 + j; }

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_lp_block(uint3 id: SV_DispatchThreadID)
{
    uint k = id.x;
    if (k >= params.chunk_count || lp[lp_f(k)].w != 0.0)
        return;
    if (!lp_factor(k))
    {
        // Holds still (as a chunk with no bonds would), reported.
        lp[lp_f(k)].w = 1.0;
        atom[a_counters() + C_ERROR].or(32u);
    }
}

// Chunk `k`'s intact bonds, counted.
uint lp_intact(uint k)
{
    float4 range = ext[params.ext_chunks + 4 * k];
    uint first = asuint(range.x);
    uint count = asuint(range.y);
    uint intact = 0;
    for (uint e = 0; e < count; e++)
        intact += work[w_alive() + adj_bond(first + e)] != 0 ? 1u : 0u;
    return intact;
}

// Chunk `k`'s diagonal block over its intact bonds, Cholesky-factored into `lp_l(k)`;
// false when not positive definite. The factor keeps the count of the intact bonds it was
// made over (+ 1, in its record's last word): a bond never heals, so while the count is
// the same the bonds are, and so is the factor (made again, it would be the same bits).
bool lp_factor(uint k)
{
    uint intact = lp_intact(k);
    if (asuint(lp[lp_l(k) + 5].w) == intact + 1)
        return true;
    float d[6][6];
    for (uint p = 0; p < 6; p++)
        for (uint q = 0; q < 6; q++)
            d[p][q] = 0.0;
    float4 range = ext[params.ext_chunks + 4 * k];
    uint first = asuint(range.x);
    uint count = asuint(range.y);
    for (uint e = 0; e < count; e++)
    {
        uint i = adj_bond(first + e);
        if (work[w_alive() + i] == 0)
            continue;
        LpBond b = lp_bond(i);
        float3 r = b.centroid - chunks[4 * k].xyz;
        float kl[6] = { b.law.stiff0.y, b.law.stiff0.y, b.law.stiff0.x, b.law.stiff0.z, b.law.stiff0.w, b.law.stiff1.x };
        float3 rows[3] = { b.t1, b.t2, b.n };
        float j[6][6];
        for (uint a = 0; a < 3; a++)
        {
            float3 c = cross(r, rows[a]);
            j[a][0] = rows[a].x; j[a][1] = rows[a].y; j[a][2] = rows[a].z;
            j[a][3] = c.x; j[a][4] = c.y; j[a][5] = c.z;
            j[3 + a][0] = 0.0; j[3 + a][1] = 0.0; j[3 + a][2] = 0.0;
            j[3 + a][3] = rows[a].x; j[3 + a][4] = rows[a].y; j[3 + a][5] = rows[a].z;
        }
        for (uint p2 = 0; p2 < 6; p2++)
            for (uint q2 = 0; q2 < 6; q2++)
            {
                float s = 0.0;
                for (uint a2 = 0; a2 < 6; a2++)
                    s += j[a2][p2] * kl[a2] * j[a2][q2];
                d[p2][q2] += s;
            }
    }
    float l[21];
    for (uint z0 = 0; z0 < 21; z0++)
        l[z0] = 0.0;
    bool ok = true;
    for (uint i2 = 0; i2 < 6; i2++)
        for (uint j2 = 0; j2 <= i2; j2++)
        {
            float s = d[i2][j2];
            for (uint p3 = 0; p3 < j2; p3++)
                s -= l[tri(i2, p3)] * l[tri(j2, p3)];
            if (i2 == j2)
            {
                ok = ok && s > 0.0;
                l[tri(i2, i2)] = sqrt(max(s, 1e-30));
            }
            else
                l[tri(i2, j2)] = s / l[tri(j2, j2)];
        }
    if (!ok)
        return false;
    for (uint w = 0; w < 6; w++)
        lp[lp_l(k) + w] = float4(l[4 * w], 4 * w + 1 < 21 ? l[4 * w + 1] : 0.0, 4 * w + 2 < 21 ? l[4 * w + 2] : 0.0, w == 5 ? asfloat(intact + 1) : l[4 * w + 3]);
    return true;
}

void lp_sweep(uint k, uint from)
{
    st_sweep(k, LP_STATE, from, params.lp_c1, params.lp_c2);
}

// A sweep of state `state` (its load and fixed flag, its `x` from copy `from` to the
// other, its step `d = c1 d + c2 D^-1 r`), with chunk `k`'s factored block.
void st_sweep(uint k, uint state, uint from, float c1, float c2)
{
    uint to = 1 - from;
    float4 x0 = lp[st_x(k, state, from)];
    float4 x1 = lp[st_x(k, state, from) + 1];
    float4 f = lp[st_f(k, state)];
    if (f.w != 0.0)
    {
        lp[st_x(k, state, to)] = x0;
        lp[st_x(k, state, to) + 1] = x1;
        lp[st_d(k, state)] = float4(0.0);
        lp[st_d(k, state) + 1] = float4(0.0);
        return;
    }
    float r[6] = { f.x, f.y, f.z, 0.0, 0.0, 0.0 };
    float3 t = lp[st_f(k, state) + 1].xyz;
    r[3] = t.x; r[4] = t.y; r[5] = t.z;
    float4 range = ext[params.ext_chunks + 4 * k];
    uint first = asuint(range.x);
    uint count = asuint(range.y);
    for (uint e = 0; e < count; e++)
    {
        // The bond as `st_force` takes it, from its side's record (`lp_sides`).
        float4 s0 = lp_sides[5 * (first + e)];
        if (work[w_alive() + asuint(s0.w)] == 0)
            continue;
        float4 s1 = lp_sides[5 * (first + e) + 1];
        float4 s2 = lp_sides[5 * (first + e) + 2];
        float4 s3 = lp_sides[5 * (first + e) + 3];
        float4 s4 = lp_sides[5 * (first + e) + 4];
        bool first_side = s4.w != 0.0;
        uint ka = first_side ? k : asuint(s1.w);
        uint kb = first_side ? asuint(s1.w) : k;
        float3 ra = s0.xyz;
        float3 rb = s1.xyz;
        float3 t1 = s2.xyz;
        float3 n = s3.xyz;
        float3 t2 = cross(n, t1);
        float4 xa0 = lp[st_x(ka, state, from)];
        float4 xa1 = lp[st_x(ka, state, from) + 1];
        float4 xb0 = lp[st_x(kb, state, from)];
        float4 xb1 = lp[st_x(kb, state, from) + 1];
        float3 delta = (xb0.xyz + cross(xb1.xyz, rb)) - (xa0.xyz + cross(xa1.xyz, ra));
        float3 phi = xb1.xyz - xa1.xyz;
        float3 q_lin = float3(s2.w * dot(delta, t1), s2.w * dot(delta, t2), s3.w * dot(delta, n));
        float3 q_ang = float3(s4.x * dot(phi, t1), s4.y * dot(phi, t2), s4.z * dot(phi, n));
        float3 fb = t1 * q_lin.x + t2 * q_lin.y + n * q_lin.z;
        float3 mb = t1 * q_ang.x + t2 * q_ang.y + n * q_ang.z;
        float3 arm = first_side ? ra : rb;
        float3 lf = first_side ? fb : -fb;
        float3 lm = (first_side ? mb : -mb) + cross(arm, lf);
        r[0] += lf.x; r[1] += lf.y; r[2] += lf.z;
        r[3] += lm.x; r[4] += lm.y; r[5] += lm.z;
    }
    float l[21];
    for (uint z0 = 0; z0 < 21; z0++)
        l[z0] = 0.0;
    for (uint w = 0; w < 6; w++)
    {
        float4 v = lp[lp_l(k) + w];
        l[4 * w] = v.x;
        if (4 * w + 1 < 21) l[4 * w + 1] = v.y;
        if (4 * w + 2 < 21) l[4 * w + 2] = v.z;
        if (4 * w + 3 < 21) l[4 * w + 3] = v.w;
    }
    float y[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    for (uint i2 = 0; i2 < 6; i2++)
    {
        float s = r[i2];
        for (uint p = 0; p < i2; p++)
            s -= l[tri(i2, p)] * y[p];
        y[i2] = s / l[tri(i2, i2)];
    }
    float z[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    for (int i3 = 5; i3 >= 0; i3--)
    {
        float s = y[i3];
        for (int p2 = i3 + 1; p2 < 6; p2++)
            s -= l[tri(uint(p2), uint(i3))] * z[p2];
        z[i3] = s / l[tri(uint(i3), uint(i3))];
    }
    float4 d0 = lp[st_d(k, state)];
    float4 d1 = lp[st_d(k, state) + 1];
    d0.xyz = c1 * d0.xyz + c2 * float3(z[0], z[1], z[2]);
    d1.xyz = c1 * d1.xyz + c2 * float3(z[3], z[4], z[5]);
    lp[st_d(k, state)] = d0;
    lp[st_d(k, state) + 1] = d1;
    lp[st_x(k, state, to)] = float4(x0.xyz + d0.xyz, 0.0);
    lp[st_x(k, state, to) + 1] = float4(x1.xyz + d1.xyz, 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_lp_sweep_even(uint3 id: SV_DispatchThreadID)
{
    if (id.x < params.chunk_count)
        lp_sweep(id.x, 0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_lp_sweep_odd(uint3 id: SV_DispatchThreadID)
{
    if (id.x < params.chunk_count)
        lp_sweep(id.x, 1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_lp_breaks(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.bond_count || work[w_alive() + i] == 0)
        return;
    uint4 bond = bonds[i];
    LpBond b = lp_bond(i);
    float3 q_lin;
    float3 q_ang;
    lp_force(b, bond.x, bond.y, 0, q_lin, q_ang);
    Measures m = stress_measures(b.law, q_lin, q_ang);
    float4 idx = failure_indices(lp_material(b.law.ids.x), b.law, m, b.law.geom1.w);
    float worst = max(max(idx.x, idx.y), max(idx.z, idx.w));
    atom[a_counters() + C_LP_PEAK].max(asuint(max(worst, 0.0)));
    if (worst >= 1.0)
    {
        work[w_alive() + i] = 0;
        atom[a_dirty() + body_of(bond.x)].or(1u);
        atom[a_counters() + C_FRAME_BROKEN].add(1u);
        atom[a_counters() + C_BROKEN].add(1u);
    }
}

// ---- Local impact (stress-ref impact_local.rs, `capability::StrikeModel::Analytic`) ----
//
// The analytic strike, on either source of strikes (`SRC`): 0 the end-of-frame strike
// prediction's (`fr_strike_*`, a shot's), in place of `fr_strike_breaks` after
// `fr_strike_force`; 1 the in-step contact fracture's (`fr_cs_record`, a dynamic body's
// contact in this step), the `fr_csl_*` entries, encoded within rubble's step:
//   fr_il_first_reset, fr_il_first
//                      (prediction only) each shot's earliest strike this frame: only the
//                      first body on its path takes a strike in a frame;
//   fr_il_begin        one thread per struck body: the thickness run starts at the entry
//                      chunk's far side (along the path, its box widened by the striker's
//                      radius);
//   fr_il_tube_count, fr_il_tube_base, fr_il_tube_collect
//                      one thread per chunk (per struck body): the tube, the struck body's
//                      chunks whose interval along the path (their box widened by the
//                      striker's radius) ends past the run's start, listed body by body
//                      (where an interval lands in the list depends on the threads' order;
//                      the run does not);
//   fr_il_run          one group per struck body: the run grows over every tube interval
//                      that starts within it until none does (no cap: each round reaches a
//                      further interval's end, at most the tube's count of rounds);
//   fr_il_strike       one thread per struck body: the closed forms (impact_local.rs
//                      `strike`), the outcome counted and recorded; a perforating striker's
//                      fracture work, or the contact of one that does not get through
//                      (inelastic: it comes to rest on the face), as impulses between the
//                      striker and the wall;
//   fr_il_mark         one thread per chunk of a struck body: (prediction) the launch's mark
//                      on what the outcome breaks out (a later strike of that launch on it
//                      is the engine's: the shot pushes its own plug, it does not strike it
//                      again); the plug's chunks for the carry;
//   fr_il_breaks       one thread per bond of a struck body: it breaks when it crosses the
//                      plug's or the scab's boundary or touches the crater
//                      (impact_local.rs `breaks`);
//   fr_il_carry        after the split, one thread per chunk of a plug: the striker and its
//                      plug leave together, their momentum along the path kept.

static const float IL_CONE_TAN = 1.0;
static const uint IL_DAMAGE = 1;
static const uint IL_CRATER = 2;
static const uint IL_SPALL = 4;
static const uint IL_PERFORATE = 8;

// ---- The in-step contact strike's record (`fr_cs_record`), per body slot in `lp` ----
//
// (contact point (stress frame), the load path's peak force `pi J / (2 tau)`); (the
// direction the striker pushes in (stress frame), its closing speed); (the strike's
// effective mass along it, the striker's face radius, its Young's modulus, its engine body
// as bits); (the struck chunk as bits, the Hertz rule's force `J / tau`, the striker's mass,
// when within the step they meet); (the striker's collider, its launch (a shot's, else 0),
// as bits).
// `atom` per body slot 12 words, all 0 for a body not struck (as the buffer starts, and as
// `fr_cs_clean` leaves a struck one): the largest strike impulse (float bits), the struck
// chunk + 1, the complement of the strike's collider (`cs_pick`); the complement of the
// lowest chunk; whether it is supported; its chunks: their count, their list's start, its
// fill; whether it is listed; the complements of the strike's other collider and of its
// feature and side; its row and side (`2 row + side + 1`).
static const uint CS_WORDS = 12;
uint a_cs(uint b) { return a_il_cursor() + 1 + CS_WORDS * b; }

bool cs_struck(uint b)
{
    return b < params.body_count && atom[a_cs(b) + 1].load() != 0;
}

// What an analytic strike needs of its striker.
struct IlStriker
{
    uint body;     // its engine body
    float bound;   // its bounding radius (prediction: the path's marks)
    float radius;  // its face (Hertz)
    float mass;
    float m_eff;   // the strike's effective mass
    float modulus; // its modulus
    uint launch;   // prediction: the launch whose debris it does not strike again (0 none)
};

bool il_struck<let SRC : uint>(uint b)
{
    if (SRC == 1)
        return cs_struck(b);
    uint shot;
    return params.shot_count > 0 && b < params.body_count && struck(b, shot);
}

uint il_entry<let SRC : uint>(uint b)
{
    return SRC == 1 ? atom[a_cs(b) + 1].load() - 1 : atom[a_strike() + 9 * b + 8].load();
}

// The strike's time within the step (in-step: when the contact's two points meet).
float il_t_in<let SRC : uint>(uint b)
{
    return SRC == 1 ? lp[lp_body(b) + 5].w : asfloat(atom[a_strike() + 9 * b + 6].load());
}

// The struck body's path: from the contact point along the closing direction.
void il_path<let SRC : uint>(uint b, out float3 contact, out float3 d, out float speed)
{
    if (SRC == 1)
    {
        contact = lp[lp_body(b) + 2].xyz;
        float4 r1 = lp[lp_body(b) + 3];
        d = r1.xyz;
        speed = r1.w;
        return;
    }
    uint st = strike_at(b);
    float3 rel = frames[st + 1].xyz;
    speed = length(rel);
    d = rel / max(speed, 1e-12);
    contact = frames[st + 2].xyz;
}

IlStriker il_striker<let SRC : uint>(uint b)
{
    IlStriker s;
    if (SRC == 1)
    {
        float4 r2 = lp[lp_body(b) + 4];
        float4 r3 = lp[lp_body(b) + 5];
        s.body = asuint(r2.w);
        // The punch list's sphere (world_coupling.rs `predict_punch`), as the shots' table.
        s.bound = 1.01 * r2.y + 1e-3;
        s.radius = r2.y;
        s.mass = r3.z;
        s.m_eff = r2.x;
        s.modulus = r2.z;
        s.launch = asuint(lp[lp_body(b) + 6].y);
        return s;
    }
    uint shot = uint(frames[strike_at(b)].w) - 1;
    float4 info = shots[2 * shot];
    s.body = asuint(info.x);
    s.bound = info.y;
    s.radius = info.z;
    s.mass = info.w;
    float inv_mass = bodies[BODY_WORDS * b].w;
    s.m_eff = inv_mass == 0.0 ? s.mass : s.mass / (1.0 + s.mass * inv_mass);
    s.modulus = shots[2 * shot + 1].x;
    s.launch = asuint(shots[2 * shot + 1].y);
    return s;
}

// The line `p + d t` through chunk `k`'s box widened by `pad`: (t0, t1), t0 > t1 if it
// misses.
float2 il_interval(uint k, float3 p, float3 d, float pad)
{
    Quat q;
    float3 cc = chunk_box_world(k, q);
    Quat box = normalized(quat_mul(q, quat_of(geometry[3 * k + 2])));
    float3 half = geometry[3 * k + 1].xyz;
    float inf = float_bits(0x7f800000u);
    float t0 = -inf;
    float t1 = inf;
    for (uint ax = 0; ax < 3; ax++)
    {
        float3 e = float3(0.0);
        e[ax] = 1.0;
        float3 axis = rotate(box, e);
        float o = dot(p - cc, axis);
        float dv = dot(d, axis);
        float ext_ = half[ax] + pad;
        if (abs(dv) < 1e-9)
        {
            if (abs(o) > ext_)
                return float2(1.0, 0.0);
            continue;
        }
        float ta = (-ext_ - o) / dv;
        float tb = (ext_ - o) / dv;
        t0 = max(t0, min(ta, tb));
        t1 = min(t1, max(ta, tb));
    }
    return float2(t0, t1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_il_first_reset(uint3 id: SV_DispatchThreadID)
{
    if (id.x == 0)
        atom[a_il_cursor()].store(0u);
    if (id.x < params.shot_count)
        atom[a_first(id.x)].store(0x7f800000u);
}

// One thread per struck body: its strike time into its shot's earliest (a time is not
// negative: its bits order as the floats do).
[shader("compute")]
[numthreads(64, 1, 1)]
void fr_il_first(uint3 id: SV_DispatchThreadID)
{
    uint b = id.x;
    if (b >= params.body_count || params.shot_count == 0)
        return;
    uint shot;
    if (!struck(b, shot))
        return;
    atom[a_first(shot)].min(atom[a_strike() + 9 * b + 6].load());
}

void il_begin<let SRC : uint>(uint b)
{
    if (b >= params.body_count || !il_struck<SRC>(b))
        return;
    // A shot strikes the first body on its path in a frame; one further on waits for the
    // next frame, and the speed the first strike leaves it (no strike charged twice).
    if (SRC == 0)
    {
        uint shot;
        struck(b, shot);
        if (atom[a_strike() + 9 * b + 6].load() != atom[a_first(shot)].load())
        {
            frames[strike_at(b)].w = 0.0;
            return;
        }
    }
    float3 contact;
    float3 d;
    float speed;
    il_path<SRC>(b, contact, d, speed);
    float2 t = il_interval(il_entry<SRC>(b), contact, d, il_striker<SRC>(b).radius);
    atom[a_il(b)].store(asuint(max(t.y, 0.0)));
    for (uint i = 1; i < 6; i++)
        atom[a_il(b) + i].store(0u);
}

// Chunk `k`'s interval along its struck body's path (`b`), when it is in the body's tube:
// it ends past the run's start (`il_begin`).
bool il_tube<let SRC : uint>(uint k, out uint b, out float2 t)
{
    b = 0;
    t = float2(1.0, 0.0);
    if (k >= params.chunk_count)
        return false;
    b = body_of(k);
    if (!il_struck<SRC>(b))
        return false;
    float3 contact;
    float3 d;
    float speed;
    il_path<SRC>(b, contact, d, speed);
    t = il_interval(k, contact, d, il_striker<SRC>(b).radius);
    return t.x <= t.y && t.y > asfloat(atom[a_il(b)].load());
}

void il_tube_count<let SRC : uint>(uint k)
{
    uint b;
    float2 t;
    if (il_tube<SRC>(k, b, t))
        atom[a_il(b) + 1].add(1u);
}

void il_tube_base<let SRC : uint>(uint b)
{
    if (b < params.body_count && il_struck<SRC>(b))
        atom[a_il(b) + 4].store(atom[a_il_cursor()].add(atom[a_il(b) + 1].load()));
}

void il_tube_collect<let SRC : uint>(uint k)
{
    uint b;
    float2 t;
    if (!il_tube<SRC>(k, b, t))
        return;
    uint at = w_il_tube() + 2 * (atom[a_il(b) + 4].load() + atom[a_il(b) + 5].add(1u));
    work[at] = asuint(t.x);
    work[at + 1] = asuint(t.y);
}

static const uint IL_GROUP = 64;
groupshared float g_il_best[IL_GROUP];

// Struck body `b`'s run (alike in the group, thread `i`): from the entry chunk's far side,
// each round reaches the furthest end of the tube's intervals that start within it
// (touching within a millimetre and a 50th of the striker's radius: chunks of one wall
// share faces; a gap wider than that is another wall), until a round reaches no further.
// Every round passes a further interval's end, so the run settles within the tube's count
// of rounds, however many chunks it crosses; a maximum, it does not depend on the list's
// order. The exits are group-uniform (`group_uniform`): the rounds have barriers.
void il_run<let SRC : uint>(uint b, uint i)
{
    if (group_uniform(il_struck<SRC>(b) ? 1u : 0u, i) == 0)
        return;
    float pad = il_striker<SRC>(b).radius;
    float touch = 1e-3 + 0.02 * pad;
    uint base = atom[a_il(b) + 4].load();
    uint count = atom[a_il(b) + 1].load();
    float end = group_uniform_float(asfloat(atom[a_il(b)].load()), i);
    for (;;)
    {
        float best = end;
        for (uint j = i; j < count; j += IL_GROUP)
        {
            uint at = w_il_tube() + 2 * (base + j);
            if (asfloat(work[at]) <= end + touch)
                best = max(best, asfloat(work[at + 1]));
        }
        g_il_best[i] = best;
        GroupMemoryBarrierWithGroupSync();
        for (uint step = IL_GROUP / 2; step > 0; step >>= 1)
        {
            if (i < step)
                g_il_best[i] = max(g_il_best[i], g_il_best[i + step]);
            GroupMemoryBarrierWithGroupSync();
        }
        float next = group_uniform_float(g_il_best[0], i);
        GroupMemoryBarrierWithGroupSync();
        if (!(next > end))
            break;
        end = next;
    }
    if (i == 0)
        atom[a_il(b)].store(asuint(end));
}

// An impulse `j` along the path (the stress frame's `d`) from the striker (body `sb`, of
// mass `m_shot`) onto body `b` at `contact`: the striker slows by `j / m_shot`, the body
// takes it at the contact (linear and angular, through its inverse inertia; nothing for an
// anchored body). Both pose regions.
void il_impulse(uint b, uint sb, float m_shot, float3 contact, float3 d, float j)
{
    float3 jr = to_rubble(d) * j;
    uint at = BODY_WORDS * b;
    float inv_mass = bodies[at].w;
    float3 dv = jr * inv_mass;
    float3 dw = float3(0.0);
    if (inv_mass > 0.0)
    {
        Quat q = quat_of(bodies[at + 1]);
        float3 r = to_rubble(contact) - bodies[at].xyz;
        dw = rotate(q, bodies[at + 2].xyz * rotate(conj(q), cross(r, jr)));
    }
    for (uint region = 0; region < 2; region++)
    {
        uint s_at = BODY_WORDS * (sb + region * params.max_bodies) + 3;
        bodies[s_at].xyz -= jr / m_shot;
        uint w_at = BODY_WORDS * (b + region * params.max_bodies);
        bodies[w_at + 3].xyz += dv;
        bodies[w_at + 4].xyz += dw;
    }
}

void il_strike<let SRC : uint>(uint b)
{
    if (b >= params.body_count || !il_struck<SRC>(b))
        return;
    float t_in = il_t_in<SRC>(b);
    uint entry = il_entry<SRC>(b);
    float3 contact;
    float3 d;
    float speed;
    il_path<SRC>(b, contact, d, speed);
    IlStriker sk = il_striker<SRC>(b);
    float r = max(sk.radius, 1e-3);
    float m_imp = sk.mass;
    float e_imp = sk.modulus;
    float m = sk.m_eff;
    // The launch's own debris (its plug, scab or crater, `fr_il_mark`): the shot pushes what
    // it broke out, the engine's contact; struck once.
    if (sk.launch != 0 && work[w_marks() + entry] == sk.launch)
    {
        lp[lp_body(b)] = float4(0.0);
        lp[lp_body(b) + 1] = float4(0.0);
        return;
    }
    float h = max(asfloat(atom[a_il(b)].load()) - r, 1e-6);
    // Prediction: already touching at the frame's start, the strike was met before (one
    // strike per contact; the engine's contact carries on from it). In the step, a contact
    // closing faster than `strike_speed` is the strike itself, touching or not.
    if (SRC == 0 && !(t_in > 0.0))
    {
        lp[lp_body(b)] = float4(h, 0.0, 0.0, 0.0);
        lp[lp_body(b) + 1] = float4(0.0);
        return;
    }
    // The entry chunk's material (impact_local.rs `Target`).
    uint o = params.ext_chunks + 4 * entry;
    float4 mat0 = ext[o + 1]; // density, tensile, compressive, shear
    JointMaterial dm;
    dm.strength = float4(0.0);
    dm.energy = float4(0.0);
    dm.dif = ext[o + 2];
    float4 mat3 = ext[o + 3]; // DIF max, Poisson's ratio, Young's modulus
    dm.misc = float4(mat3.x, 0.0, 0.0, 0.0);
    dm.kind_flags = uint4(0u);
    float nu = mat3.y;
    float e_target = mat3.z / (1.0 - nu * nu);
    float e_star = 1.0 / (1.0 / e_imp + 1.0 / e_target);
    float v = max(speed, 1e-3);
    float pi = 3.14159265358979;
    // Hertz.
    float h0 = pow(15.0 * m * v * v / (16.0 * e_star * sqrt(r)), 0.4);
    float t_c = 2.94 * h0 / v;
    float force = 4.0 / 3.0 * e_star * sqrt(r) * pow(h0, 1.5);
    float a = sqrt(r * h0);
    float p0 = 3.0 * force / (2.0 * pi * a * a);
    float ring = (1.0 - 2.0 * nu) * p0 / 3.0;
    // Rate.
    float k = mat3.x > 0.0 ? dif_factor(dm, v / h) : 1.0;
    float ft = k * mat0.y;
    float fc = k * mat0.z;
    float tau = k * mat0.w;
    bool damage = p0 > fc || ring > ft;
    // Ballistic limit.
    float area = pi * r * r;
    float e_bl = min(fc * area * h, 0.5 * pi * (2.0 * r + IL_CONE_TAN * h) * h * h * tau);
    float v_bl = sqrt(2.0 * e_bl / m);
    bool perforate = damage && v >= v_bl;
    float r2 = r + IL_CONE_TAN * h;
    float plug = mat0.x * pi * h / 3.0 * (r * r + r * r2 + r2 * r2);
    float v_res = perforate ? m_imp / (m_imp + plug) * sqrt(max(v * v - v_bl * v_bl, 0.0)) : 0.0;
    // Crater.
    float depth = damage && !perforate ? min(0.5 * m * v * v / (fc * area), h) : 0.0;
    bool crater = depth > a;
    // Spall.
    float c = sqrt(e_target / mat0.x);
    float z_t = mat0.x * c;
    float rho_p = m_imp / (4.0 / 3.0 * pi * r * r * r);
    float z_p = rho_p * sqrt(e_imp / rho_p);
    float mean_pressure = force / (pi * a * a);
    float source = min(min(z_t * v * z_p / (z_p + z_t), mean_pressure), fc);
    float source_radius = mean_pressure > fc ? max(a, r) : a;
    float pulse = c * t_c;
    float back = source * min(source_radius / h, 1.0) * min(2.0 * h / pulse, 1.0);
    bool spall = damage && !perforate && back > ft;
    float scab = spall ? 0.5 * min(pulse, h) : 0.0;
    uint outcome = (damage ? IL_DAMAGE : 0u) | (crater ? IL_CRATER : 0u) | (spall ? IL_SPALL : 0u) | (perforate ? IL_PERFORATE : 0u);
    lp[lp_body(b)] = float4(h, v_bl, v_res, asfloat(outcome));
    lp[lp_body(b) + 1] = float4(crater ? depth : 0.0, scab, a, e_bl);
    if (perforate)
        atom[a_counters() + C_IL_PERFORATE].add(1u);
    else if (spall)
        atom[a_counters() + C_IL_SPALL].add(1u);
    else if (crater)
        atom[a_counters() + C_IL_CRATER].add(1u);
    else
        atom[a_counters() + C_IL_BOUNCE].add(1u);
    uint sb = sk.body;
    if (perforate)
    {
        // Fracture work: the closing speed after the plug's ballistic limit's energy is
        // spent, `1/2 m (v^2 - v'^2) = E_bl`, by an impulse between the striker and the wall
        // (the striker and its plug leave together after the split: `fr_il_carry`).
        if (params.fracture_work != 0)
            il_impulse(b, sb, m_imp, contact, d, m * (v - sqrt(max(v * v - 2.0 * e_bl / m, 0.0))));
    }
    else if (damage)
    {
        // Not through: the strike is resolved here, inelastically (what crushed, spalled or
        // cracked took the energy): the striker comes to rest against the wall's face, and
        // its momentum along the path goes into the wall. It is moved on to the face (where
        // it is at `t_in`), so a frame's travel longer than the wall cannot pass through it.
        il_impulse(b, sb, m_imp, contact, d, m * v);
        float3 step = to_rubble(d) * (v * clamp(t_in, 0.0, params.dt));
        for (uint region = 0; region < 2; region++)
            bodies[BODY_WORDS * (sb + region * params.max_bodies)].xyz += step;
    }
}

struct IlPlace
{
    float along;
    float radial;
    bool entry;
    bool crossed;
};

// Where chunk `k`'s centre sits relative to a strike at `contact` along `d`; and whether the
// striker (a sphere of radius `r`, touching the face at `contact`) overlaps its box on its
// way through the wall's thickness `h` (`il_crossed`).
IlPlace il_place(uint k, uint entry, float3 contact, float3 d, float r, float h)
{
    uint slot = body_of(k);
    Quat q;
    float3 origin;
    structure_pose(slot, BODY_WORDS * slot, q, origin);
    float3 c = origin + rotate(q, chunks[4 * k].xyz);
    float3 rel = c - contact;
    IlPlace p;
    p.along = dot(rel, d);
    p.radial = length(rel - d * p.along);
    p.entry = k == entry;
    p.crossed = il_crossed(k, contact - d * r, d, h + 2.0 * r, r);
    return p;
}

// Whether a sphere of radius `r` whose centre runs from `a` along `d` for `length` overlaps
// chunk `k`'s box (the segment comes nearer the box than `r`, by more than a part in 10^4:
// a box it only touches is not crossed). The squared distance from a point of the segment
// to the box is convex along it: its least value by a ternary search (fixed passes).
bool il_crossed(uint k, float3 a, float3 d, float length, float r)
{
    Quat q;
    float3 cc = chunk_box_world(k, q);
    Quat box = normalized(quat_mul(q, quat_of(geometry[3 * k + 2])));
    float3 half = geometry[3 * k + 1].xyz;
    float3 al = rotate(conj(box), a - cc);
    float3 ul = rotate(conj(box), d);
    float lo = 0.0;
    float hi = length;
    for (uint pass = 0; pass < 40; pass++)
    {
        float m1 = lo + (hi - lo) / 3.0;
        float m2 = hi - (hi - lo) / 3.0;
        float3 e1 = max(abs(al + ul * m1) - half, float3(0.0));
        float3 e2 = max(abs(al + ul * m2) - half, float3(0.0));
        if (dot(e1, e1) <= dot(e2, e2))
            hi = m2;
        else
            lo = m1;
    }
    float3 e = max(abs(al + ul * (0.5 * (lo + hi))) - half, float3(0.0));
    return dot(e, e) < 0.9998 * r * r;
}

// The plug: the cone (impact_local.rs `in_plug`), and every chunk the striker crosses on its
// way through (a perforation goes through what is in its path, whole chunks: one whose
// centre is outside the cone but whose box the striker overlaps would stand in the hole).
bool il_plug(uint outcome, float r, float h, IlPlace p)
{
    return (outcome & IL_PERFORATE) != 0 && (p.entry || p.crossed || (p.along >= 0.0 && p.along <= h && p.radial <= r + IL_CONE_TAN * p.along));
}

bool il_scab(uint outcome, float r, float h, float scab, IlPlace p)
{
    return (outcome & IL_SPALL) != 0 && p.along >= h - scab && p.along <= h && p.radial <= r + IL_CONE_TAN * h;
}

bool il_crater(uint outcome, float r, float depth, float a, IlPlace p)
{
    return (outcome & IL_CRATER) != 0 && p.along >= 0.0 && p.along <= depth && p.radial <= max(r, a);
}

// One thread per chunk of a struck body: a shot's launch's mark on every chunk the outcome
// breaks out (in the plug, the scab or the crater) or its path crosses; the plug's chunks
// for the carry.
void il_mark<let SRC : uint>(uint k)
{
    if (k >= params.chunk_count)
        return;
    uint b = body_of(k);
    if (!il_struck<SRC>(b))
        return;
    float4 s0 = lp[lp_body(b)];
    float4 s1 = lp[lp_body(b) + 1];
    uint outcome = asuint(s0.w);
    if ((outcome & (IL_CRATER | IL_SPALL | IL_PERFORATE)) == 0)
        return;
    float3 contact;
    float3 d;
    float speed;
    il_path<SRC>(b, contact, d, speed);
    uint entry = il_entry<SRC>(b);
    IlStriker sk = il_striker<SRC>(b);
    float r = max(sk.radius, 1e-3);
    IlPlace p = il_place(k, entry, contact, d, r, s0.x);
    bool plug = il_plug(outcome, r, s0.x, p);
    if (sk.launch != 0)
    {
        // The chunks its path crosses within the wall (their boxes widened by the shot's
        // bounding radius, as `fr_strike_entry` takes them): the strike has met them; a
        // graze of the hole's rim is the engine's contact, not another strike.
        float bound = sk.bound;
        float2 t = il_interval(k, contact - d * bound, d, bound);
        bool path = t.x <= t.y && t.x <= s0.x + 2.0 * bound && t.y >= 0.0;
        if (plug || path || il_scab(outcome, r, s0.x, s1.y, p) || il_crater(outcome, r, s1.x, s1.z, p))
            work[w_marks() + k] = sk.launch;
    }
    if (plug)
    {
        work[w_carry() + k] = b + 1;
        add64(a_il(b) + 2, chunks[4 * k].w, params.scale_m);
    }
}

// After the split, one thread per chunk that came out in a plug: the striker and the plug
// leave together (Recht and Ipson's premise), at the velocity that keeps their momentum
// along the path, `m v' / (m + M_plug)` relative to the wall (`v'` the striker's closing
// speed after the fracture work; `M_plug` the plug's chunks' mass). The exchange happens at
// the strike's time `t_in` within the step, not at its start: the striker, closing at the
// strike's speed `v` until then, is moved on by `(v - v_r) t_in` along the path, where it
// would be at the step's end. Every thread of a plug writes the same values.
void il_carry<let SRC : uint>(uint k)
{
    if (k >= params.chunk_count)
        return;
    uint c = work[w_carry() + k];
    if (c == 0)
        return;
    work[w_carry() + k] = 0;
    uint b0 = c - 1;
    if (!il_struck<SRC>(b0))
        return;
    float3 contact;
    float3 d;
    float speed;
    il_path<SRC>(b0, contact, d, speed);
    float3 dr = to_rubble(d);
    IlStriker sk = il_striker<SRC>(b0);
    uint sb = sk.body;
    float m_s = sk.mass;
    float plug_mass = read64(a_il(b0) + 2, params.scale_m);
    float3 v_wall = bodies[BODY_WORDS * b0 + 3].xyz;
    float3 v_shot = bodies[BODY_WORDS * sb + 3].xyz;
    float closing = dot(v_shot - v_wall, dr);
    float v_r = m_s * closing / (m_s + plug_mass);
    float3 shot_v = v_shot + dr * (v_r - closing);
    float3 plug_v = v_wall + dr * v_r;
    // Before the strike the striker closed at the strike's speed (`il_path`), not at the
    // speed the fracture work left it.
    float t_in = clamp(il_t_in<SRC>(b0), 0.0, params.dt);
    float3 shot_p = bodies[BODY_WORDS * sb].xyz + dr * ((speed - v_r) * t_in);
    uint pb = body_of(k);
    for (uint region = 0; region < 2; region++)
    {
        uint s_at = BODY_WORDS * (sb + region * params.max_bodies);
        bodies[s_at].xyz = shot_p;
        bodies[s_at + 3].xyz = shot_v;
        bodies[BODY_WORDS * (pb + region * params.max_bodies) + 3].xyz = plug_v;
    }
}

void il_breaks<let SRC : uint>(uint i)
{
    if (i >= params.bond_count || work[w_alive() + i] == 0)
        return;
    uint4 bond = bonds[i];
    uint b = body_of(bond.x);
    if (!il_struck<SRC>(b))
        return;
    float4 s0 = lp[lp_body(b)];
    float4 s1 = lp[lp_body(b) + 1];
    uint outcome = asuint(s0.w);
    if ((outcome & (IL_CRATER | IL_SPALL | IL_PERFORATE)) == 0)
        return;
    float3 contact;
    float3 d;
    float speed;
    il_path<SRC>(b, contact, d, speed);
    uint entry = il_entry<SRC>(b);
    float r = max(il_striker<SRC>(b).radius, 1e-3);
    float h = s0.x;
    IlPlace pa = il_place(bond.x, entry, contact, d, r, h);
    IlPlace pb = il_place(bond.y, entry, contact, d, r, h);
    bool cut = il_plug(outcome, r, h, pa) != il_plug(outcome, r, h, pb)
        || il_scab(outcome, r, h, s1.y, pa) != il_scab(outcome, r, h, s1.y, pb)
        || il_crater(outcome, r, s1.x, s1.z, pa) || il_crater(outcome, r, s1.x, s1.z, pb);
    if (cut)
    {
        work[w_alive() + i] = 0;
        atom[a_dirty() + b].or(1u);
        atom[a_counters() + C_FRAME_BROKEN].add(1u);
        atom[a_counters() + C_BROKEN].add(1u);
        if (SRC == 1)
            atom[a_counters() + C_CS_BREAKS].add(1u);
    }
}

// The prediction's entries (shots: `params.shot_count` of them).
[shader("compute")]
[numthreads(64, 1, 1)]
void fr_il_begin(uint3 id: SV_DispatchThreadID)
{
    if (params.shot_count > 0)
        il_begin<0>(id.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_il_tube_count(uint3 id: SV_DispatchThreadID)
{
    if (params.shot_count > 0)
        il_tube_count<0>(id.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_il_tube_base(uint3 id: SV_DispatchThreadID)
{
    if (params.shot_count > 0)
        il_tube_base<0>(id.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_il_tube_collect(uint3 id: SV_DispatchThreadID)
{
    if (params.shot_count > 0)
        il_tube_collect<0>(id.x);
}

// A group per body slot (most leave at once: not struck).
[shader("compute")]
[numthreads(64, 1, 1)]
void fr_il_run(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    if (params.shot_count > 0)
        il_run<0>(group.x, thread.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_il_strike(uint3 id: SV_DispatchThreadID)
{
    if (params.shot_count > 0)
        il_strike<0>(id.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_il_mark(uint3 id: SV_DispatchThreadID)
{
    if (params.shot_count > 0)
        il_mark<0>(id.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_il_carry(uint3 id: SV_DispatchThreadID)
{
    if (params.shot_count > 0)
        il_carry<0>(id.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_il_breaks(uint3 id: SV_DispatchThreadID)
{
    if (params.shot_count > 0)
        il_breaks<0>(id.x);
}

// The in-step contact fracture's entries (`fr_cs_record`'s strikes).
[shader("compute")]
[numthreads(64, 1, 1)]
void fr_csl_begin(uint3 id: SV_DispatchThreadID)
{
    il_begin<1>(id.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_csl_tube_count(uint3 id: SV_DispatchThreadID)
{
    il_tube_count<1>(id.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_csl_tube_base(uint3 id: SV_DispatchThreadID)
{
    il_tube_base<1>(id.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_csl_tube_collect(uint3 id: SV_DispatchThreadID)
{
    il_tube_collect<1>(id.x);
}

// A group per struck structure (`fr_cs_record`'s list).
[shader("compute")]
[numthreads(64, 1, 1)]
void fr_csl_run(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    il_run<1>(work[w_cs_bodies() + group.x], thread.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_csl_strike(uint3 id: SV_DispatchThreadID)
{
    il_strike<1>(id.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_csl_mark(uint3 id: SV_DispatchThreadID)
{
    il_mark<1>(id.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_csl_carry(uint3 id: SV_DispatchThreadID)
{
    il_carry<1>(id.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_csl_breaks(uint3 id: SV_DispatchThreadID)
{
    il_breaks<1>(id.x);
}

// ---- In-step contact fracture (`capability::Profile::contact_fracture`) ----
//
// Encoded within rubble's step where its contacts are made and kept (the plan's "warm"
// phase: after the narrowphase, the time-of-impact contacts and the warm start; before the
// constraint graph and the solve). This step's contacts are the strikes: a row with a
// structure's chunk on one side and a dynamic body of its own on the other, closing faster
// than `strike_speed` at the contact (its material points' velocities, spin included). Each
// struck structure takes its largest strike (the impulse that stops the closing, `J = m_eff
// v`, `m_eff` along the contact normal with both bodies' rotation; ties: the lowest row and
// side), and the profile's own stress solver judges it, before the step solves anything:
//   the threshold rule (`StrikeModel::Hertz`): a bond breaks when its share of the strike's
//                      Hertz force beats its tensile capacity (`fr_strike_breaks`' rule);
//   the load path (`LoadModel::LoadPath`): the strike's load solved through the structure,
//                      a bond broken by its own strengths (`fr_cs_solve`);
//   the analytic strike (`StrikeModel::Analytic`): the local impact's outcome (`fr_csl_*`,
//                      the `fr_il_*` kernels on these strikes), the striker's momentum with
//                      it.
// What broke splits (`fr_gate`, the split); the rows of the moved colliders are re-anchored
// to their pieces and start cold; the step solves its contacts with the pieces. Nothing
// breaks: the whole structure takes the contact and the striker rebounds. Forward only: no
// rollback, no second solve, no detection of its own (rubble's contacts are the strikes).
//
//   fr_cs_begin        one thread: the step's sizes (the detection's: its contact rows);
//   fr_cs_contacts     one thread per contact row: its strikes' impulses (the largest per
//                      structure kept), each struck structure listed once;
//   fr_cs_gate         one thread: the sizes of everything after (none without a strike);
//   fr_cs_pick_a … _d  one thread per contact row, a pass each: of each structure's largest
//                      strikes, the one of the least identity (colliders, feature, side), not
//                      its row (rubble's rows are in no fixed order); then its row;
//   fr_cs_record       one thread per struck structure: the strike's record;
//   fr_cs_hertz        (threshold rule) one thread per bond;
//   fr_cs_count, fr_cs_base, fr_cs_collect, fr_cs_solve
//                      (load path) each struck structure's chunks listed, a group per struck
//                      structure solves the strike's load and breaks its bonds;
//   fr_cs_gate2        one thread: the contact rows' sizes when a bond broke (else none);
//   fr_cs_save         one thread per contact row: where its sides on a broken structure's
//                      chunks touch, in the world (before the split moves their bodies);
//   fr_cs_fix          after the split, one thread per contact row: those sides' anchors on
//                      their pieces' bodies, the row's warm start dropped;
//   fr_cs_clean        one thread per struck structure: its words back to 0.

// The contact rows this step has (rubble's `StepCounters::contactCount`).
uint cs_rows()
{
    return min(engine_counters[2], params.cs_row_cap);
}

uint cs_row(uint r)
{
    return 8 * (params.cs_row_offset + r);
}

// `(r x u) . I^-1 (r x u)` of engine body `b` with lever arm `r` along `u` (rubble's frame;
// its world inverse inertia `R diag(I^-1) R^T`).
float cs_angular(uint b, float3 r, float3 u)
{
    uint at = BODY_WORDS * b;
    float3 l = rotate(conj(quat_of(bodies[at + 1])), cross(r, u));
    return dot(l * bodies[at + 2].xyz, l);
}

// Side `side` of this step's contact row `r` as a strike on a structure: its chunk `k` (the
// structure's body `s`), the other side's body `ob` and collider `co`, the contact point `p`
// (this side's anchor) and the direction `u` the other side pushes in (rubble's frame), the
// closing speed `v` of the two material points there (spin included), and the strike's
// effective mass `m_eff` along `u` (both bodies' rotation included). False when it is none: no chunk on this side, the other side not a
// dynamic body of its own, or not closing faster than `strike_speed`.
bool cs_side(uint r, uint side, out uint k, out uint s, out uint ob, out uint co, out float3 p, out float3 u, out float v, out float m_eff, out float t_in)
{
    t_in = 0.0;
    k = NONE;
    s = NONE;
    ob = NONE;
    co = NONE;
    p = float3(0.0);
    u = float3(0.0);
    v = 0.0;
    m_eff = 0.0;
    uint row = cs_row(r);
    uint4 ids = asuint(engine_contacts[row]);
    if (ids.w == 0)
        return false;
    uint cm = side == 0 ? ids.x : ids.y;
    co = side == 0 ? ids.y : ids.x;
    k = collider_info[2 * cm].x;
    if (k == NONE || k >= params.chunk_count || collider_info[2 * co].z != KIND_BODY_COLLIDER)
        return false;
    s = body_of(k);
    ob = asuint(colliders[4 * co].x);
    if (ob == s || ob >= params.max_bodies)
        return false;
    // A shot meeting what its own analytic strike broke out or went through (`il_mark`): its
    // contact is the engine's, not another strike (it pushes its plug, it grazes its hole).
    uint launch = collider_info[2 * co + 1].w;
    if (launch != 0 && work[w_marks() + k] == launch)
        return false;
    uint bs = BODY_WORDS * s;
    uint bo = BODY_WORDS * ob;
    float inv_o = bodies[bo].w;
    if (!(inv_o > 0.0))
        return false;
    // The normal points from side 0 to side 1: side 1 pushes side 0 along -n.
    float3 n = engine_contacts[row + 1].xyz;
    u = side == 0 ? -n : n;
    // The lever arms: each side's anchor turned to the world (a time-of-impact row's
    // anchors are its bodies' at the time of impact: rubble `emitToiManifold`).
    float3 rs = rotate(quat_of(bodies[bs + 1]), side == 0 ? engine_contacts[row + 2].xyz : engine_contacts[row + 3].xyz);
    float3 ro = rotate(quat_of(bodies[bo + 1]), side == 0 ? engine_contacts[row + 3].xyz : engine_contacts[row + 2].xyz);
    p = bodies[bs].xyz + rs;
    // When within the step they meet: the gap along `u` between the two contact points at
    // the step's start over the closing speed (0 for a row already touching).
    float gap = dot(p - (bodies[bo].xyz + ro), u);
    float3 vs = bodies[bs + 3].xyz + cross(bodies[bs + 4].xyz, rs);
    float3 vo = bodies[bo + 3].xyz + cross(bodies[bo + 4].xyz, ro);
    v = dot(vo - vs, u);
    if (!(v > params.strike_speed))
        return false;
    float w = bodies[bs].w + cs_angular(s, rs, u) + inv_o + cs_angular(ob, ro, u);
    if (!(w > 0.0))
        return false;
    m_eff = 1.0 / w;
    t_in = clamp(gap / v, 0.0, params.dt);
    return true;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void fr_cs_begin(uint3 id: SV_DispatchThreadID)
{
    atom[a_counters() + C_CS_BODIES].store(0u);
    atom[a_counters() + C_CS_CURSOR].store(0u);
    atom[a_il_cursor()].store(0u);
    // The detection sized by the step's contact rows, not the engine's capacity.
    set_args(ARG_CS_ROWS, groups_of(cs_rows(), 64));
}

// One thread per contact row: each side's strike impulse into its structure's largest (an
// impulse is positive: its bits order as the floats do); a struck structure listed once
// (where in the list depends on the threads' order; nothing computed from it does).
[shader("compute")]
[numthreads(64, 1, 1)]
void fr_cs_contacts(uint3 id: SV_DispatchThreadID)
{
    uint r = id.x;
    if (r >= cs_rows())
        return;
    for (uint side = 0; side < 2; side++)
    {
        uint k, s, ob, co;
        float3 p, u;
        float v, m_eff, t_in;
        if (cs_side(r, side, k, s, ob, co, p, u, v, m_eff, t_in))
        {
            atom[a_cs(s)].max(asuint(m_eff * v));
            if (atom[a_cs(s) + 8].exchange(1u) == 0u)
                work[w_cs_bodies() + atom[a_counters() + C_CS_BODIES].add(1u)] = s;
        }
    }
}

// One thread: everything after the detection sized by it. Without a strike (most steps)
// none of it dispatches a group.
[shader("compute")]
[numthreads(1, 1, 1)]
void fr_cs_gate(uint3 id: SV_DispatchThreadID)
{
    uint struck = atom[a_counters() + C_CS_BODIES].load();
    bool any = struck > 0;
    set_args(ARG_CS_PICK, any ? groups_of(cs_rows(), 64) : 0);
    set_args(ARG_CS_LIST, groups_of(struck, 64));
    set_args(ARG_CS_BODIES, struck);
    set_args(ARG_CS_CHUNKS, any ? groups_of(params.chunk_count, 64) : 0);
    set_args(ARG_CS_BONDS, any ? groups_of(params.bond_count, 64) : 0);
    set_args(ARG_CS_SLOTS, any ? groups_of(params.body_count, 64) : 0);
}

// A strike row's identity, as rubble's warm start keys a contact (its order does not depend
// on where the engine's atomic appends put the row): this side's collider, the other's, the
// feature and the side (`2 feature + side`).
uint3 cs_identity(uint r, uint side)
{
    uint4 ids = asuint(engine_contacts[cs_row(r)]);
    return side == 0 ? uint3(ids.x, ids.y, 2 * ids.z) : uint3(ids.y, ids.x, 2 * ids.z + 1);
}

// Which of each structure's largest strikes is its strike: the least identity (ties in the
// impulse are common: a face's points of one manifold, alike by symmetry), compared one word
// a pass (`STAGE` 0, 1, 2; the complements' maxima), then (`STAGE` 3) its row.
void cs_pick<let STAGE : uint>(uint r)
{
    if (r >= cs_rows())
        return;
    for (uint side = 0; side < 2; side++)
    {
        uint k, s, ob, co;
        float3 p, u;
        float v, m_eff, t_in;
        if (!cs_side(r, side, k, s, ob, co, p, u, v, m_eff, t_in) || asuint(m_eff * v) != atom[a_cs(s)].load())
            continue;
        uint3 id = cs_identity(r, side);
        uint at = a_cs(s);
        if (STAGE >= 1 && ~atom[at + 2].load() != id.x)
            continue;
        if (STAGE >= 2 && ~atom[at + 9].load() != id.y)
            continue;
        if (STAGE >= 3 && ~atom[at + 10].load() != id.z)
            continue;
        if (STAGE == 0)
            atom[at + 2].max(~id.x);
        else if (STAGE == 1)
            atom[at + 9].max(~id.y);
        else if (STAGE == 2)
            atom[at + 10].max(~id.z);
        else
            atom[at + 11].store(2 * r + side + 1);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_cs_pick_a(uint3 id: SV_DispatchThreadID)
{
    cs_pick<0>(id.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_cs_pick_b(uint3 id: SV_DispatchThreadID)
{
    cs_pick<1>(id.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_cs_pick_c(uint3 id: SV_DispatchThreadID)
{
    cs_pick<2>(id.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_cs_pick_d(uint3 id: SV_DispatchThreadID)
{
    cs_pick<3>(id.x);
}

// One thread per struck structure: its strike record (`lp`, `a_cs` above). The Hertz
// pulse (api.rs `hertz_duration`) of the impulse `J` between the striker's face (the
// ingest's table: a shot's radius, a chunk's smallest half extent) and the struck chunk
// (their moduli: the striker's as the table has it, the chunk's plane-strain one).
[shader("compute")]
[numthreads(64, 1, 1)]
void fr_cs_record(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= atom[a_counters() + C_CS_BODIES].load())
        return;
    uint s = work[w_cs_bodies() + id.x];
    uint key = atom[a_cs(s) + 11].load() - 1;
    uint k, s2, ob, co;
    float3 p, u;
    float v, m_eff, t_in;
    if (!cs_side(key / 2, key % 2, k, s2, ob, co, p, u, v, m_eff, t_in))
        return;
    uint4 other = collider_info[2 * co + 1];
    float e_o = asfloat(other.y);
    float face = max(asfloat(other.z), 1e-3);
    float e_star = 1.0 / (1.0 / max(e_o, 1.0) + 1.0 / geometry[3 * k + 1].w);
    float j = m_eff * v;
    float tau = hertz_duration(m_eff, face, e_star, v);
    lp[lp_body(s) + 2] = float4(from_rubble(p), 3.14159265358979 * j / (2.0 * tau));
    lp[lp_body(s) + 3] = float4(from_rubble(u), v);
    lp[lp_body(s) + 4] = float4(m_eff, face, e_o, asfloat(ob));
    lp[lp_body(s) + 5] = float4(asfloat(k), j / tau, 1.0 / bodies[BODY_WORDS * ob].w, t_in);
    lp[lp_body(s) + 6] = float4(asfloat(co), asfloat(other.w), 0.0, 0.0);
    atom[a_cs(s) + 1].store(k + 1);
    atom[a_counters() + C_CS_STRIKES].add(1u);
}

// The threshold rule on an in-step strike: as `fr_strike_breaks`.
[shader("compute")]
[numthreads(64, 1, 1)]
void fr_cs_hertz(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.bond_count || work[w_alive() + i] == 0)
        return;
    uint4 bond = bonds[i];
    uint b = body_of(bond.x);
    if (!cs_struck(b))
        return;
    if (strike_share_breaks(bond, lp[lp_body(b) + 2].xyz, lp[lp_body(b) + 5].y))
    {
        work[w_alive() + i] = 0;
        atom[a_dirty() + b].or(1u);
        atom[a_counters() + C_FRAME_BROKEN].add(1u);
        atom[a_counters() + C_BROKEN].add(1u);
        atom[a_counters() + C_CS_BREAKS].add(1u);
    }
}

// One thread: the contact rows' sizes for `fr_cs_save` and `fr_cs_fix` when a bond broke in
// the step (else none).
[shader("compute")]
[numthreads(1, 1, 1)]
void fr_cs_gate2(uint3 id: SV_DispatchThreadID)
{
    set_args(ARG_CS_SAVE, atom[a_counters() + C_FRAME_BROKEN].load() > 0 ? groups_of(cs_rows(), 64) : 0);
}

// One thread per contact row: each side on a chunk of a structure that broke this step, where
// it touches in the world (w 1), from its body as it is before the split; else w 0.
[shader("compute")]
[numthreads(64, 1, 1)]
void fr_cs_save(uint3 id: SV_DispatchThreadID)
{
    uint r = id.x;
    if (r >= cs_rows())
        return;
    uint row = cs_row(r);
    uint4 ids = asuint(engine_contacts[row]);
    for (uint side = 0; side < 2; side++)
    {
        float4 saved = float4(0.0);
        uint col = side == 0 ? ids.x : ids.y;
        uint k = ids.w != 0 ? collider_info[2 * col].x : NONE;
        if (k != NONE && k < params.chunk_count && dirty(body_of(k)))
        {
            uint at = BODY_WORDS * body_of(k);
            float3 anchor = side == 0 ? engine_contacts[row + 2].xyz : engine_contacts[row + 3].xyz;
            saved = float4(bodies[at].xyz + rotate(quat_of(bodies[at + 1]), anchor), 1.0);
        }
        cs_anchors[2 * r + side] = saved;
    }
}

// One thread per struck structure, the step's last: its words back to 0 (as for a body not
// struck), so a step without a strike touches no body slot's.
[shader("compute")]
[numthreads(64, 1, 1)]
void fr_cs_clean(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= atom[a_counters() + C_CS_BODIES].load())
        return;
    uint at = a_cs(work[w_cs_bodies() + id.x]);
    for (uint w = 0; w < CS_WORDS; w++)
        atom[at + w].store(0u);
}

// After the split, one thread per contact row: a saved side's anchor on its collider's body
// now (its piece), the same world point; the row starts cold (rubble's `makeContactRow`:
// no multipliers, unit penalties, not sticking).
[shader("compute")]
[numthreads(64, 1, 1)]
void fr_cs_fix(uint3 id: SV_DispatchThreadID)
{
    uint r = id.x;
    if (r >= cs_rows())
        return;
    uint row = cs_row(r);
    uint4 ids = asuint(engine_contacts[row]);
    bool moved = false;
    for (uint side = 0; side < 2; side++)
    {
        float4 saved = cs_anchors[2 * r + side];
        if (saved.w == 0.0)
            continue;
        uint col = side == 0 ? ids.x : ids.y;
        uint at = BODY_WORDS * asuint(colliders[4 * col].x);
        float3 anchor = rotate(conj(quat_of(bodies[at + 1])), saved.xyz - bodies[at].xyz);
        engine_contacts[row + 2 + side].xyz = anchor;
        moved = true;
    }
    if (moved)
    {
        engine_contacts[row + 4].w = 0.0; // lambda n
        engine_contacts[row + 5].w = 0.0; // lambda t1
        float4 w6 = engine_contacts[row + 6];
        engine_contacts[row + 6] = float4(1.0, 1.0, 0.0, w6.w); // kN, kT, lambda t2, c0 t1
        engine_contacts[row + 7].y = 0.0; // stick
    }
}

// `work`: the struck structures' chunks, listed structure by structure (`fr_cs_collect`),
// then the struck structures (`fr_cs_record`).
uint w_cs_list() { return w_carry() + params.chunk_count; }
uint w_cs_bodies() { return w_cs_list() + params.chunk_count; }

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_cs_count(uint3 id: SV_DispatchThreadID)
{
    uint k = id.x;
    if (k >= params.chunk_count)
        return;
    uint s = body_of(k);
    if (!cs_struck(s))
        return;
    atom[a_cs(s) + 3].max(~k);
    if ((info[k] & INFO_SUPPORT) != 0)
        atom[a_cs(s) + 4].store(1u);
    atom[a_cs(s) + 5].add(1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fr_cs_base(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= atom[a_counters() + C_CS_BODIES].load())
        return;
    uint s = work[w_cs_bodies() + id.x];
    atom[a_cs(s) + 6].store(atom[a_counters() + C_CS_CURSOR].add(atom[a_cs(s) + 5].load()));
}

// (Where a chunk lands in its structure's list depends on the threads' order; nothing
// computed from the list does: each chunk's work is its own.)
[shader("compute")]
[numthreads(64, 1, 1)]
void fr_cs_collect(uint3 id: SV_DispatchThreadID)
{
    uint k = id.x;
    if (k >= params.chunk_count)
        return;
    uint s = body_of(k);
    if (!cs_struck(s))
        return;
    work[w_cs_list() + atom[a_cs(s) + 6].load() + atom[a_cs(s) + 7].add(1u)] = k;
}

// Chunk `k`'s increment's load (of struck structure `s`): the strike's force and moment on
// its chunk, less (a free structure) the d'Alembert loads of the rigid acceleration it
// gives the whole body; its state from rest.
void cs_load(uint k, uint s)
{
    bool supported = atom[a_cs(s) + 4].load() != 0;
    bool fixed = (info[k] & INFO_SUPPORT) != 0 || (!supported && ~atom[a_cs(s) + 3].load() == k);
    float4 r0 = lp[lp_body(s) + 2];
    float4 r1 = lp[lp_body(s) + 3];
    Quat q;
    float3 origin;
    structure_pose(s, BODY_WORDS * s, q, origin);
    Quat qi = conj(q);
    float3 force = rotate(qi, r1.xyz * r0.w);
    float3 contact = rotate(qi, r0.xyz - origin);
    float4 c0 = chunks[4 * k];
    float3 f = float3(0.0);
    float3 t = float3(0.0);
    if (atom[a_cs(s) + 1].load() == k + 1)
    {
        f = force;
        t = cross(contact - c0.xyz, force);
    }
    uint b = BODY_WORDS * s;
    float inv_mass = bodies[b].w;
    if (!supported && inv_mass > 0.0)
    {
        float3 com = frames[2 * s].xyz;
        Quat p = quat_of(frames[2 * s + 1]);
        float3 alpha = rotate(p, bodies[b + 2].xyz * rotate(conj(p), cross(contact - com, force)));
        float3 r = c0.xyz - com;
        f -= c0.w * (force * inv_mass + cross(alpha, r));
        float4 c1 = chunks[4 * k + 1];
        float4 c2 = chunks[4 * k + 2];
        t -= float3(c1.x * alpha.x + c1.w * alpha.y + c2.x * alpha.z, c1.w * alpha.x + c1.y * alpha.y + c2.y * alpha.z, c2.x * alpha.x + c2.y * alpha.y + c1.z * alpha.z);
    }
    lp[st_f(k, CS_STATE)] = float4(f, fixed ? 1.0 : 0.0);
    lp[st_f(k, CS_STATE) + 1] = float4(t, 0.0);
    for (uint c = 0; c < 2; c++)
    {
        lp[st_x(k, CS_STATE, c)] = float4(0.0);
        lp[st_x(k, CS_STATE, c) + 1] = float4(0.0);
    }
    lp[st_d(k, CS_STATE)] = float4(0.0);
    lp[st_d(k, CS_STATE) + 1] = float4(0.0);
}

// Bond `i` of struck structure `s` under the structure's own state (the load path's, where
// it runs) and the strike's increment: broken when a failure index reaches 1.
void cs_break(uint i, uint s)
{
    uint4 bond = bonds[i];
    LpBond b = lp_bond(i);
    float3 q_lin;
    float3 q_ang;
    st_force(b, bond.x, bond.y, CS_STATE, 0, q_lin, q_ang);
    if (params.load_model == 1)
    {
        float3 b_lin;
        float3 b_ang;
        lp_force(b, bond.x, bond.y, 0, b_lin, b_ang);
        q_lin += b_lin;
        q_ang += b_ang;
    }
    Measures m = stress_measures(b.law, q_lin, q_ang);
    float4 idx = failure_indices(lp_material(b.law.ids.x), b.law, m, b.law.geom1.w);
    if (max(max(idx.x, idx.y), max(idx.z, idx.w)) >= 1.0)
    {
        work[w_alive() + i] = 0;
        atom[a_dirty() + s].or(1u);
        atom[a_counters() + C_FRAME_BROKEN].add(1u);
        atom[a_counters() + C_BROKEN].add(1u);
        atom[a_counters() + C_CS_BREAKS].add(1u);
    }
}

static const uint CS_GROUP = 256;

[shader("compute")]
[numthreads(256, 1, 1)]
void fr_cs_solve(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint s = work[w_cs_bodies() + group.x];
    uint t = thread.x;
    uint base = atom[a_cs(s) + 6].load();
    uint count = atom[a_cs(s) + 5].load();
    for (uint j = t; j < count; j += CS_GROUP)
        cs_load(work[w_cs_list() + base + j], s);
    device_group_barrier();
    for (uint j2 = t; j2 < count; j2 += CS_GROUP)
    {
        uint k = work[w_cs_list() + base + j2];
        if (lp[st_f(k, CS_STATE)].w == 0.0 && !lp_factor(k))
        {
            lp[st_f(k, CS_STATE)].w = 1.0;
            atom[a_counters() + C_ERROR].or(32u);
        }
    }
    device_group_barrier();
    for (uint sweep = 0; sweep < params.cs_sweeps; sweep++)
    {
        for (uint j3 = t; j3 < count; j3 += CS_GROUP)
            st_sweep(work[w_cs_list() + base + j3], CS_STATE, sweep % 2, params.cs_c1, params.cs_c2);
        device_group_barrier();
    }
    // The sweeps end in copy 0 (an even count). Each bond by its first chunk's thread.
    for (uint j4 = t; j4 < count; j4 += CS_GROUP)
    {
        uint k = work[w_cs_list() + base + j4];
        float4 range = ext[params.ext_chunks + 4 * k];
        uint first = asuint(range.x);
        uint n = asuint(range.y);
        for (uint e = 0; e < n; e++)
        {
            uint i = adj_bond(first + e);
            if (bonds[i].x == k && work[w_alive() + i] != 0)
                cs_break(i, s);
        }
    }
}
