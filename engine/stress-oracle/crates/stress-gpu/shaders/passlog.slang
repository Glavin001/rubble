// A frame's split log (engine/docs/stress-baseline-goal.md, M4): after each split pass of a
// frame run in one submission, what the host needs to follow the split (its mirror's
// `split_cluster_into`, the engine's `adopt_spawned`) appended to `log`, which the host
// reads with the frame's status.
//
// log[0]: the next free word (entries start at 16); log[1]: entries; log[2]: 1 when an
// entry did not fit (the host then reads nothing of it and splits those parents itself).
// An entry (uint words, at `at`):
//   at + 0  its words           at + 1  parents P        at + 2  children C
//   at + 3  aborted             at + 4  the cascade pass that disconnected
//   at + 5  first body          at + 6  first collider
//   at + 7  bodies after        at + 8  colliders after
//   at + 9  live islands before at + 10 the next cluster id before
//   at + 11 child slots (the parents' chunks, `total`)
//   then P x PARENT (island, children, cluster id, structure); then C x CHILD_HEAD (member
//   count, anchored, body, first collider, colliders, valid, centre of mass and mass (float
//   bits), principal rotation (float bits)); then the members (`total`, each its chunk
//   within its structure, in parent then child order); then the new bodies (7 float4 each)
//   and colliders (4 float4 each). Everything from the tables as they were before the
//   split, so the host can read an entry without having followed the ones before it.

import common;

struct LogParams
{
    uint log_words; // the log's capacity
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::binding(0, 0)]] ConstantBuffer<LogParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<uint> control;
[[vk::binding(2, 0)]] StructuredBuffer<uint4> parents;
[[vk::binding(3, 0)]] StructuredBuffer<uint> work;
[[vk::binding(4, 0)]] StructuredBuffer<float4> children;
[[vk::binding(5, 0)]] StructuredBuffer<uint4> slots;
[[vk::binding(6, 0)]] StructuredBuffer<float4> extras;
[[vk::binding(7, 0)]] StructuredBuffer<float4> bodies;
[[vk::binding(8, 0)]] StructuredBuffer<float4> colliders;
[[vk::binding(9, 0)]] RWStructuredBuffer<uint> log;
[[vk::binding(10, 0)]] StructuredBuffer<uint> status;
// The split pass's copies of the islands and chunk statics as they were (gpu_pass.rs `old`).
[[vk::binding(11, 0)]] StructuredBuffer<float4> old_islands;
[[vk::binding(12, 0)]] StructuredBuffer<float4> old_chunks;

static const uint CHILD_WORDS = 8;
static const uint CHILD_HEAD = 14;
static const uint HEADER = 16;
static const uint PARENT = 4;
static const uint ISLAND_WORDS = 25;
static const uint CHUNK_WORDS = 16;

groupshared uint g_at;
groupshared uint g_size;
groupshared uint g_children;

[shader("compute")]
[numthreads(256, 1, 1)]
void pass_log(uint3 thread: SV_GroupThreadID)
{
    uint tid = thread.x;
    uint n = control[CTL_PARENTS];
    if (n == 0)
        return;
    uint total = control[CTL_TOTAL];
    bool aborted = control[CTL_ABORT] != 0;
    uint first_body = control[CTL_FIRST_BODY];
    uint first_collider = control[CTL_FIRST_COLLIDER];
    uint nb = aborted ? 0u : control[CTL_BODIES_AFTER] - first_body;
    uint nc = aborted ? 0u : control[CTL_COLLIDERS_AFTER] - first_collider;
    if (tid == 0)
    {
        uint c = 0;
        for (uint p = 0; p < n; p++)
            c += parents[p].z;
        uint size = HEADER + PARENT * n + CHILD_HEAD * c + total + 28 * nb + 16 * nc;
        uint at = max(log[0], HEADER);
        g_children = c;
        if (at + size > params.log_words)
        {
            log[2] = 1;
            g_size = 0;
        }
        else
        {
            g_size = size;
        }
        g_at = at;
    }
    GroupMemoryBarrierWithGroupSync();
    uint at = g_at;
    uint size = g_size;
    uint c = g_children;
    if (size == 0)
        return;
    // The header, the parents and the children's heads (one thread: a fixed order).
    if (tid == 0)
    {
        log[at + 0] = size;
        log[at + 1] = n;
        log[at + 2] = c;
        log[at + 3] = aborted ? 1u : 0u;
        log[at + 4] = status[1] - 1;
        log[at + 5] = first_body;
        log[at + 6] = first_collider;
        log[at + 7] = first_body + nb;
        log[at + 8] = first_collider + nc;
        log[at + 9] = control[CTL_ISLANDS];
        log[at + 10] = control[CTL_NEXT_ID];
        log[at + 11] = total;
        for (uint w = 12; w < HEADER; w++)
            log[at + w] = 0;
        uint h = at + HEADER + PARENT * n;
        for (uint p = 0; p < n; p++)
        {
            uint4 parent = parents[p];
            uint4 range = asuint(old_islands[ISLAND_WORDS * parent.x]);
            uint4 info = asuint(old_chunks[CHUNK_WORDS * range.x + 8]);
            log[at + HEADER + PARENT * p] = parent.x;
            log[at + HEADER + PARENT * p + 1] = parent.z;
            log[at + HEADER + PARENT * p + 2] = asuint(old_islands[ISLAND_WORDS * parent.x + 19]).w; // done_pad: the cluster id
            log[at + HEADER + PARENT * p + 3] = info.z;
            for (uint k = 0; k < parent.z; k++)
            {
                uint slot = parent.y + k;
                float4 head = children[CHILD_WORDS * slot];
                uint4 s = aborted ? uint4(0) : slots[slot];
                float4 com = extras[2 * slot];
                float4 rot = extras[2 * slot + 1];
                log[h + 0] = asuint(head.z);
                log[h + 1] = asuint(head.w);
                log[h + 2] = s.x;
                log[h + 3] = s.y;
                log[h + 4] = s.z;
                log[h + 5] = s.w;
                log[h + 6] = asuint(com.x);
                log[h + 7] = asuint(com.y);
                log[h + 8] = asuint(com.z);
                log[h + 9] = asuint(com.w);
                log[h + 10] = asuint(rot.x);
                log[h + 11] = asuint(rot.y);
                log[h + 12] = asuint(rot.z);
                log[h + 13] = asuint(rot.w);
                h += CHILD_HEAD;
            }
        }
    }
    // The members, already in parent then child order (split.slang), and the new bodies
    // and colliders, contiguous in child order (spawn.slang).
    uint m = at + HEADER + PARENT * n + CHILD_HEAD * c;
    for (uint j = tid; j < total; j += 256)
    {
        // Member j of the parent whose base (`parents[p].y`) holds it: its chunk in its structure.
        uint p = 0;
        while (p + 1 < n && parents[p + 1].y <= j)
            p++;
        uint first = asuint(old_islands[ISLAND_WORDS * parents[p].x]).x;
        log[m + j] = asuint(old_chunks[CHUNK_WORDS * (first + work[total + j]) + 8]).w;
    }
    uint b = m + total;
    for (uint j = tid; j < 7 * nb; j += 256)
    {
        float4 v = bodies[7 * first_body + j];
        log[b + 4 * j] = asuint(v.x);
        log[b + 4 * j + 1] = asuint(v.y);
        log[b + 4 * j + 2] = asuint(v.z);
        log[b + 4 * j + 3] = asuint(v.w);
    }
    uint k0 = b + 28 * nb;
    for (uint j = tid; j < 4 * nc; j += 256)
    {
        float4 v = colliders[4 * first_collider + j];
        log[k0 + 4 * j] = asuint(v.x);
        log[k0 + 4 * j + 1] = asuint(v.y);
        log[k0 + 4 * j + 2] = asuint(v.z);
        log[k0 + 4 * j + 3] = asuint(v.w);
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0)
    {
        log[0] = at + size;
        log[1] = log[1] + 1;
    }
}
