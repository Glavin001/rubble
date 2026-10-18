// The GPU-resident coupling's contact ingest (docs/stress-fracture-gpu-resident.md,
// stages 2-3, phase 2): an engine's solved contact rows, read where the engine left them
// on the GPU, as the stress solver's contact manifolds per (chunk, other body), with no
// host round trip.
//
//   ingest_rows     one thread per contact row of an engine substep: the impulse on each
//                   side's chunk over the substep, the contact point (the anchors' mean
//                   at the substep's end), the closing speed of the material points at
//                   the frame's start, the other body's mass, modulus and radius, as
//                   stress-rubble `contacts_of_step` and world_coupling.rs
//                   `engine_contacts_to_impulses` compute them; records at fixed slots
//                   (substep, row, side), so their order is the host's;
//   ingest_count, ingest_scan, ingest_scatter   the frame's records grouped by chunk
//                   (a counting sort; atomics only on integer counts);
//   ingest_merge    one thread per chunk: its records in slot order, merged per other
//                   body in order of first appearance (coupled.rs `manifolds`): summed
//                   impulse, impulse-weighted point and closing speed.
//
// Frames: the engine is Y-up, the stress solver Z-up: (x, y, z) -> (x, -z, y).

struct IngestParams
{
    uint row_offset;    // the substep's first row in `contacts`
    uint max_rows;      // rows per substep slot (the engine's contact capacity)
    uint substep;       // this substep's slot
    uint chunk_count;   // GPU chunks
    uint record_count;  // records in the frame (substeps x 2 x max_rows)
    uint max_bodies;    // the engine's body capacity (its previous rows follow)
    float h;            // the substep's duration (s)
    uint pad;
};

// An engine body (engine-schema `BodyState`, 112 B).
struct EngineBody
{
    float4 pos_inv_mass;
    float4 rot;               // x, y, z, w
    float4 inv_inertia_flags;
    float4 vel_sleep;
    float4 ang_vel_gen;
    float4 initial_pos;
    float4 initial_rot;
};

// An engine collider (engine-schema `Collider`, 64 B): ids.x its body.
struct EngineCollider
{
    uint4 ids;
    float4 local_pos;
    float4 local_rot;
    float4 params;
};

// An engine contact row (engine-schema `Contact`, 128 B).
struct EngineContactRow
{
    uint4 ids;               // collider a, collider b, feature, alive
    float4 normal_pen;
    float4 anchor_a;
    float4 anchor_b;
    float4 t1_lambda_n;
    float4 t2_lambda_t1;
    float4 kn_kt_lt2_c0t1;   // z: lambda_t2
    float4 tail;
};

// What the coupling knows of a collider: `a` = (GPU chunk or NONE, body code, kind,
// chunk side taken), `b` = (body mass (inf: anchored or static), Young's modulus,
// radius, -) as the other side of a contact sees it.
struct ColliderInfo
{
    uint4 a;
    float4 b;
};

static const uint NONE = 0xffffffffu;
static const uint KIND_SKIP = 0;   // retired, or not the coupling's
static const uint KIND_STATIC = 1; // ground or static geometry (no body code)
static const uint KIND_BODY = 2;

// A chunk-side record (64 B): `ids` = (GPU chunk, other body code, slot, valid);
// `point` (w: |impulse|), `impulse` (w: closing speed), `other` (mass, modulus, radius).
struct Record
{
    uint4 ids;
    float4 point;
    float4 impulse;
    float4 other;
    // The contact's identity, for an order that does not depend on where the engine's
    // atomic appends put its row: substep, this side's collider, the other side's, and
    // the engine's feature id * 2 + side.
    uint4 key;
};



[[vk::binding(0, 0)]] ConstantBuffer<IngestParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<EngineContactRow> contacts;
[[vk::binding(2, 0)]] StructuredBuffer<EngineCollider> colliders;
[[vk::binding(3, 0)]] StructuredBuffer<EngineBody> bodies;       // at the substep's end
[[vk::binding(4, 0)]] StructuredBuffer<EngineBody> start_bodies; // at the frame's start
[[vk::binding(5, 0)]] StructuredBuffer<uint> counters;           // engine-schema `StepCounters`
[[vk::binding(6, 0)]] StructuredBuffer<ColliderInfo> collider_info;
[[vk::binding(7, 0)]] RWStructuredBuffer<Record> records;
// Per chunk: [0, n + 1) segment starts, then the sorted record slots; [n + 1, 2n + 1)
// each chunk's manifold count.
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> sort;
// Per record slot (in sorted position): the manifold, as a record (point, impulse with
// closing speed, other).
[[vk::binding(9, 0)]] RWStructuredBuffer<Record> manifolds;
// Per chunk: [0, n) record counts, [n, 2n) fill cursors.
[[vk::binding(10, 0)]] RWStructuredBuffer<Atomic<uint>> counts;

// Whether record a comes before record b in a chunk's merge order: by key, then by slot.
bool record_before(uint a, uint b)
{
    uint4 ka = records[a].key;
    uint4 kb = records[b].key;
    for (uint q = 0; q < 4; q++)
    {
        if (ka[q] != kb[q])
            return ka[q] < kb[q];
    }
    return a < b;
}

float3 rotate_q(float4 q, float3 v)
{
    float3 u = q.xyz;
    float3 t = 2.0 * cross(u, v);
    return v + q.w * t + cross(u, t);
}

float3 to_zup(float3 v)
{
    return float3(v.x, -v.z, v.y);
}

// Velocity at the frame's start of a body's material point `p` (engine frame).
float3 point_velocity(ColliderInfo info, uint body, float3 p)
{
    if (info.a.z != KIND_BODY)
        return float3(0.0);
    EngineBody s = start_bodies[body];
    return s.vel_sleep.xyz + cross(s.ang_vel_gen.xyz, p - s.pos_inv_mass.xyz);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ingest_rows(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.max_rows)
        return;
    uint base = 2 * (params.substep * params.max_rows + i);
    Record none;
    none.ids = uint4(NONE, NONE, base, 0);
    none.point = float4(0.0);
    none.impulse = float4(0.0);
    none.other = float4(0.0);
    none.key = uint4(0);
    records[base] = none;
    none.ids.z = base + 1;
    records[base + 1] = none;
    if (i >= counters[2])
        return;
    EngineContactRow c = contacts[params.row_offset + i];
    if (c.ids.w == 0)
        return;
    float3 n = c.normal_pen.xyz;
    float3 f = n * c.t1_lambda_n.w + c.t1_lambda_n.xyz * c.t2_lambda_t1.w + c.t2_lambda_t1.xyz * c.kn_kt_lt2_c0t1.z;
    // Impulse on B's body over the substep (A's: the opposite).
    float3 jb = -f * params.h;
    if (all(jb == float3(0.0)) || any(isnan(jb)) || any(isinf(jb)))
        return;
    EngineCollider ca = colliders[c.ids.x];
    EngineCollider cb = colliders[c.ids.y];
    EngineBody ba = bodies[ca.ids.x];
    EngineBody bb = bodies[cb.ids.x];
    float3 pa = ba.pos_inv_mass.xyz + rotate_q(ba.rot, c.anchor_a.xyz);
    float3 pb = bb.pos_inv_mass.xyz + rotate_q(bb.rot, c.anchor_b.xyz);
    float3 point = (pa + pb) * 0.5;
    ColliderInfo ia = collider_info[c.ids.x];
    ColliderInfo ib = collider_info[c.ids.y];
    for (uint side = 0; side < 2; side++)
    {
        ColliderInfo mine = side == 0 ? ia : ib;
        ColliderInfo other = side == 0 ? ib : ia;
        uint mine_body = side == 0 ? ca.ids.x : cb.ids.x;
        uint other_body = side == 0 ? cb.ids.x : ca.ids.x;
        if (mine.a.x == NONE || mine.a.w == 0 || other.a.z == KIND_SKIP)
            continue;
        float3 j = side == 0 ? -jb : jb;
        float3 push = side == 0 ? -n : n;
        float closing = max(dot(point_velocity(other, other_body, point) - point_velocity(mine, mine_body, point), push), 0.0);
        Record r;
        r.ids = uint4(mine.a.x, other.a.z == KIND_BODY ? other.a.y : NONE, base + side, 1);
        float3 jz = to_zup(j);
        r.point = float4(to_zup(point), length(jz));
        r.impulse = float4(jz, closing);
        r.other = other.b;
        r.key = uint4(params.substep, side == 0 ? c.ids.x : c.ids.y, side == 0 ? c.ids.y : c.ids.x, 2 * c.ids.z + side);
        records[base + side] = r;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ingest_clear(uint3 id: SV_DispatchThreadID)
{
    if (id.x < 2 * params.chunk_count)
        counts[id.x].store(0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ingest_count(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= params.record_count)
        return;
    Record r = records[id.x];
    if (r.ids.w != 0)
        counts[r.ids.x].add(1u);
}

// Segment starts: an exclusive scan of the counts by one group (chunk_count / 1024 rounds).
groupshared uint g_scan[1024];

[shader("compute")]
[numthreads(1024, 1, 1)]
void ingest_scan(uint3 thread: SV_GroupThreadID)
{
    uint tid = thread.x;
    uint n = params.chunk_count;
    uint carry = 0;
    for (uint base = 0; base < n; base += 1024)
    {
        uint k = base + tid;
        uint v = k < n ? counts[k].load() : 0;
        g_scan[tid] = v;
        GroupMemoryBarrierWithGroupSync();
        for (uint step = 1; step < 1024; step <<= 1)
        {
            uint add = tid >= step ? g_scan[tid - step] : 0;
            GroupMemoryBarrierWithGroupSync();
            g_scan[tid] += add;
            GroupMemoryBarrierWithGroupSync();
        }
        if (k < n)
            sort[k] = carry + g_scan[tid] - v;
        uint total = g_scan[1023];
        GroupMemoryBarrierWithGroupSync();
        carry += total;
    }
    if (tid == 0)
        sort[n] = carry;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ingest_scatter(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= params.record_count)
        return;
    Record r = records[id.x];
    if (r.ids.w == 0)
        return;
    uint n = params.chunk_count;
    uint at = counts[n + r.ids.x].add(1u);
    sort[2 * n + 1 + sort[r.ids.x] + at] = id.x;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ingest_merge(uint3 id: SV_DispatchThreadID)
{
    uint k = id.x;
    uint n = params.chunk_count;
    if (k >= n)
        return;
    uint begin = sort[k];
    uint end = sort[k + 1];
    uint slots = 2 * n + 1;
    // The chunk's records in contact order (`record_before`: by substep and contact
    // identity, not by the slot the engine's atomic append gave the row, nor by the
    // scatter's order): insertion sort, the segments are short.
    for (uint a = begin + 1; a < end; a++)
    {
        uint v = sort[slots + a];
        uint b = a;
        while (b > begin && record_before(v, sort[slots + b - 1]))
        {
            sort[slots + b] = sort[slots + b - 1];
            b--;
        }
        sort[slots + b] = v;
    }
    // Manifolds per other body, in order of first appearance.
    uint count = 0;
    for (uint a = begin; a < end; a++)
    {
        Record r = records[sort[slots + a]];
        uint m = NONE;
        for (uint q = 0; q < count; q++)
        {
            if (manifolds[begin + q].ids.y == r.ids.y)
            {
                m = q;
                break;
            }
        }
        if (m == NONE)
        {
            manifolds[begin + count] = r;
            count++;
            continue;
        }
        Record acc = manifolds[begin + m];
        float w0 = acc.point.w;
        float w = r.point.w;
        float total = w0 + w;
        if (total > 0.0)
        {
            acc.point.xyz = (acc.point.xyz * w0 + r.point.xyz * w) / total;
            acc.impulse.w = (acc.impulse.w * w0 + r.impulse.w * w) / total;
        }
        acc.impulse.xyz += r.impulse.xyz;
        acc.point.w = total;
        manifolds[begin + m] = acc;
    }
    sort[n + 1 + k] = count;
}
