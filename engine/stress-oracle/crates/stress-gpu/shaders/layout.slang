// A split's tables re-laid out on the GPU (engine/docs/stress-baseline-goal.md, M4): what
// `GpuSolver::split_resident` does on the host, from the GPU's own split (split.slang):
//   layout_plan     one thread: the islands' new order as the host's `split_cluster`
//                   leaves it (each split parent, from the last, swap-removed, its children
//                   appended), each child's island index and cluster id;
//   layout_parent   one group per split parent: its chunks, chunk states, bonds and bond
//                   records re-laid in its ranges (each child's chunks in order, then its
//                   bonds; the cut bonds after, in no island), its chunks' incidence (each
//                   keeps its slots, padded with a cut bond), the resident tables' chunk
//                   entries moved with their chunks, the static-solve slots cleared, and
//                   its children's island records;
//   layout_moved    one thread per island (before layout_parent: a child may take a moved
//                   island's old index): an island moved into a parent's index takes its
//                   record there and its chunks and bonds name it; the sentinel after the
//                   islands is blanked.
// It reads the tables as they were (`old_*`, copies made before it) and writes the live
// ones. Storage tables are packed (old_f4 / live_f4 / uint_pack) for WebGPU ≤10. Only the resident quasi-static frame calls it (island flags as its clusters have
// them: driven, settled; anchored by the child).

import common;

struct LayoutParams
{
    uint parent_count;
    uint island_count;   // before the split
    uint next_id;        // the host's next cluster id (children take the next ones)
    uint substeps;       // the solver's substep count (island info.w)
    uint chunk_words;    // float4 per chunk static (16)
    uint bond_words;     // float4 per bond static (15)
    uint dyn_words;      // float4 per bond record
    uint island_words;   // float4 per island (25)
    uint chunk_count;    // layout chunks
    uint statics_base;   // scratch: static-solve chunk slots (STATICS_PER_CHUNK each)
    uint statics_bonds;  // scratch: then 2 per bond (secant stiffness)
    uint resident;       // 1: the resident tables (ctab) follow
    uint props_base;     // ctab: 1 float4 per chunk
    uint eload_base;     // ctab: 2 float4 per chunk
    uint pulse_base;     // ctab: 3 * pulse_cap float4 per chunk
    uint pulse_count_base; // ctab: 1 per chunk
    uint pulse_cap;
    uint engine_base;    // ctab: 3 float4 per island
    uint total;          // the split's child slots (split.slang `parents[n].y`)
    uint gate_flag_base; // gate: per island flag
    uint bond_count;     // layout bonds
    uint driven;         // 1: every cluster's rigid motion is an engine's (`integrate_rigid` off)
    // Packed buffer bases (float4 or uint element indices). Chrome ≤10 storage bindings.
    uint base_old_chunks;
    uint base_old_state;
    uint base_old_bonds;
    uint base_old_dyn;
    uint base_old_islands;
    uint base_old_ctab;
    uint base_old_copy;
    uint base_chunks;
    uint base_state;
    uint base_bonds;
    uint base_dyn;
    uint base_islands;
    uint base_scratch;
    uint base_ctab;
    uint base_gate;
    uint base_island_copy;
    uint base_plan;      // uint_pack
    uint base_index;
    uint base_control;
    uint base_old_index;
};

// 9 storage (+ 1 uniform): parents, work, children, frames, old_f4, live_f4, uint_pack,
// spawn_slots, spawn_extras. old_* float4 / live float4 / plan+index+control+old_index packed.
[[vk::binding(0, 0)]] ConstantBuffer<LayoutParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> parents;   // split.slang
[[vk::binding(2, 0)]] StructuredBuffer<uint> work;       // split.slang (labels: child numbers; members)
[[vk::binding(3, 0)]] StructuredBuffer<float4> children; // split.slang child records
[[vk::binding(4, 0)]] StructuredBuffer<float4> frames;   // split.slang frames (5 per parent)
[[vk::binding(5, 0)]] StructuredBuffer<float4> old_f4;   // packed old_* float4 tables
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> live_f4; // packed live float4 tables
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> uint_pack; // plan, index, control, old_index
// Per child slot (spawn.slang `slots`): its engine body (x); per child slot its centre of
// mass and principal rotation (`extras`), for the resident tables' engine entries.
[[vk::binding(8, 0)]] StructuredBuffer<uint4> spawn_slots;
[[vk::binding(9, 0)]] StructuredBuffer<float4> spawn_extras;


float4 old_chunks_at(uint i) { return old_f4[params.base_old_chunks + i]; }
float4 old_state_at(uint i) { return old_f4[params.base_old_state + i]; }
float4 old_bonds_at(uint i) { return old_f4[params.base_old_bonds + i]; }
float4 old_dyn_at(uint i) { return old_f4[params.base_old_dyn + i]; }
float4 old_islands_at(uint i) { return old_f4[params.base_old_islands + i]; }
float4 old_ctab_at(uint i) { return old_f4[params.base_old_ctab + i]; }
float4 old_copy_at(uint i) { return old_f4[params.base_old_copy + i]; }

float4 chunks_at(uint i) { return live_f4[params.base_chunks + i]; }
void chunks_set(uint i, float4 v) { live_f4[params.base_chunks + i] = v; }
float4 state_at(uint i) { return live_f4[params.base_state + i]; }
void state_set(uint i, float4 v) { live_f4[params.base_state + i] = v; }
float4 bonds_at(uint i) { return live_f4[params.base_bonds + i]; }
void bonds_set(uint i, float4 v) { live_f4[params.base_bonds + i] = v; }
float4 dyn_at(uint i) { return live_f4[params.base_dyn + i]; }
void dyn_set(uint i, float4 v) { live_f4[params.base_dyn + i] = v; }
float4 islands_at(uint i) { return live_f4[params.base_islands + i]; }
void islands_set(uint i, float4 v) { live_f4[params.base_islands + i] = v; }
float4 scratch_at(uint i) { return live_f4[params.base_scratch + i]; }
void scratch_set(uint i, float4 v) { live_f4[params.base_scratch + i] = v; }
float4 ctab_at(uint i) { return live_f4[params.base_ctab + i]; }
void ctab_set(uint i, float4 v) { live_f4[params.base_ctab + i] = v; }
float4 gate_at(uint i) { return live_f4[params.base_gate + i]; }
void gate_set(uint i, float4 v) { live_f4[params.base_gate + i] = v; }
float4 island_copy_at(uint i) { return live_f4[params.base_island_copy + i]; }
void island_copy_set(uint i, float4 v) { live_f4[params.base_island_copy + i] = v; }

uint plan_at(uint i) { return uint_pack[params.base_plan + i]; }
void plan_set(uint i, uint v) { uint_pack[params.base_plan + i] = v; }
uint index_at(uint i) { return uint_pack[params.base_index + i]; }
void index_set(uint i, uint v) { uint_pack[params.base_index + i] = v; }
uint control_at(uint i) { return uint_pack[params.base_control + i]; }
void control_set(uint i, uint v) { uint_pack[params.base_control + i] = v; }
uint old_index_at(uint i) { return uint_pack[params.base_old_index + i]; }

static const uint NONE = 0xffffffffu;
static const uint CHILD_WORDS = 8;
static const uint STATICS_PER_CHUNK = 23;
static const uint ISLAND_FLAG_ANCHORED = 1;
static const uint ISLAND_FLAG_DRIVEN = 2;
static const uint ISLAND_FLAG_SETTLED = 16;

// plan: [0] the new island count; from 16 the regions below (`*_at`).
uint cap() { return control_at(CTL_ISLANDS) + control_at(CTL_TOTAL); }
// The new order: per island an element, an old island or `island_count + child slot`.
uint order_at(uint i) { return 16 + i; }
// Per child slot: its island, its cluster id.
uint child_island_at(uint slot) { return 16 + cap() + slot; }
uint child_id_at(uint slot) { return 16 + cap() + control_at(CTL_TOTAL) + slot; }
// Per old island: its new index (NONE: split).
uint old_new_at(uint j) { return 16 + cap() + 2 * control_at(CTL_TOTAL) + j; }
// Per parent chunk (at its parent's base): its new position in the parent's range.
uint new_chunk_at(uint slot) { return 16 + cap() + 2 * control_at(CTL_TOTAL) + control_at(CTL_ISLANDS) + slot; }
// Per child slot: its bond range (first, end).
uint child_bonds_at(uint slot) { return 16 + cap() + 3 * control_at(CTL_TOTAL) + control_at(CTL_ISLANDS) + 2 * slot; }
// Per bond (by its old index): its new index (NONE: cut).
uint bond_new_at(uint b) { return 16 + cap() + 5 * control_at(CTL_TOTAL) + control_at(CTL_ISLANDS) + b; }
// Per parent (at its base): its children in the order of their islands (child numbers).
uint child_order_at(uint slot) { return 16 + cap() + 5 * control_at(CTL_TOTAL) + control_at(CTL_ISLANDS) + params.bond_count + slot; }
// Per child slot: its first new chunk position in its parent's range (island order).
uint child_start_at(uint slot) { return 16 + cap() + 6 * control_at(CTL_TOTAL) + control_at(CTL_ISLANDS) + params.bond_count + slot; }
// Per parent chunk (at its parent's base, by new position): its degree (incidence entries).
uint degree_at(uint slot) { return 16 + cap() + 7 * control_at(CTL_TOTAL) + control_at(CTL_ISLANDS) + params.bond_count + slot; }

[shader("compute")]
[numthreads(1, 1, 1)]
void layout_plan(uint3 id: SV_DispatchThreadID)
{
    if (control_at(0) != 0 || control_at(CTL_PARENTS) == 0)
        return;
    uint n = control_at(CTL_ISLANDS);
    for (uint i = 0; i < n; i++)
        plan_set(order_at(i), i);
    uint len = n;
    uint next = control_at(CTL_NEXT_ID);
    // The host splits from the last parent (`out.disconnected` is ascending).
    for (uint q = control_at(CTL_PARENTS); q > 0; q--)
    {
        uint p = q - 1;
        uint4 parent = parents[p];
        if (parent.z < 2)
            continue;
        // `swap_remove`: the last element into the parent's index.
        len--;
        if (parent.x < len)
            plan_set(order_at(parent.x), plan_at(order_at(len)));
        for (uint k = 0; k < parent.z; k++)
        {
            plan_set(order_at(len), n + parent.y + k);
            len++;
            plan_set(child_id_at(parent.y + k), next);
            next++;
        }
    }
    plan_set(0, len);
    control_set(CTL_NEW_ISLANDS, len);
    control_set(CTL_NEXT_ID_AFTER, next);
    for (uint j = 0; j < n; j++)
        plan_set(old_new_at(j), NONE);
    for (uint i = 0; i < len; i++)
    {
        uint e = plan_at(order_at(i));
        if (e < n)
            plan_set(old_new_at(e), i);
        else
            plan_set(child_island_at(e - n), i);
    }
}

uint chunk_word(uint c, uint w) { return params.chunk_words * c + w; }
uint bond_word(uint b, uint w) { return params.bond_words * b + w; }
uint island_word(uint i, uint w) { return params.island_words * i + w; }

// A chunk static's island (`info.y`, word 8).
void set_chunk_island(uint c, uint island)
{
    float4 info = chunks_at(chunk_word(c, 8));
    info.y = asfloat(island);
    chunks_set(chunk_word(c, 8), info);
}

void inverse3(float3 r0, float3 r1, float3 r2, out float3 i0, out float3 i1, out float3 i2)
{
    float3 c0 = cross(r1, r2);
    float3 c1 = cross(r2, r0);
    float3 c2 = cross(r0, r1);
    float det = dot(r0, c0);
    float inv = det != 0.0 ? 1.0 / det : 0.0;
    i0 = float3(c0.x, c1.x, c2.x) * inv;
    i1 = float3(c0.y, c1.y, c2.y) * inv;
    i2 = float3(c0.z, c1.z, c2.z) * inv;
}

groupshared uint g_cut_first;
groupshared uint g_cut_next;
static const uint LT = 256; // layout_parent's threads
groupshared uint g_class[LT];
groupshared uint g_scan64[LT];
static const uint CUT_CLASS = 0xfffffffeu;

// The parent's children in the order of their islands (the host lays them out as its
// clusters stand after the split): the child number of the `r`-th (`child_order_at`,
// sorted by `sort_children`).
uint child_by_island(uint4 parent, uint r)
{
    return plan_at(child_order_at(parent.y + r));
}

// Thread 0: a parent's children sorted by island (insertion sort: they are appended in
// order, so at most a swap-removed one is out of place).
void sort_children(uint4 parent)
{
    for (uint k = 0; k < parent.z; k++)
    {
        uint ik = plan_at(child_island_at(parent.y + k));
        uint j = k;
        while (j > 0 && plan_at(child_island_at(parent.y + plan_at(child_order_at(parent.y + j - 1)))) > ik)
        {
            plan_set(child_order_at(parent.y + j), plan_at(child_order_at(parent.y + j - 1)));
            j--;
        }
        plan_set(child_order_at(parent.y + j), k);
    }
}

// Bonds of a parent in blocks of the group's threads, in bond order: each one's rank among
// its child's (or the cut ones) in that order. `place`: its new index, its child's next
// (`child_bonds_at` + 1) or the cut cursor plus its rank, the cursors advanced past the
// block; otherwise each child's count advanced (`child_bonds_at`). The whole group calls it.
void bond_ranks(uint4 parent, uint base, uint c0, uint b0, uint b1, uint tid, bool place)
{
    for (uint block = b0; block < b1; block += LT)
    {
        uint b = block + tid;
        uint cls = NONE;
        if (b < b1)
        {
            uint4 ids = asuint(old_bonds_at(bond_word(b, 14)));
            uint ka = work[base + ids.y - c0];
            uint kb = work[base + ids.z - c0];
            cls = ka == kb ? ka : CUT_CLASS;
        }
        g_class[tid] = cls;
        GroupMemoryBarrierWithGroupSync();
        uint rank = 0;
        bool last = cls != NONE;
        for (uint t = 0; t < LT; t++)
        {
            if (t < tid && g_class[t] == cls)
                rank++;
            if (t > tid && g_class[t] == cls)
                last = false;
        }
        if (place && cls != NONE)
        {
            uint first = cls == CUT_CLASS ? g_cut_next : plan_at(child_bonds_at(base + cls) + 1);
            plan_set(bond_new_at(b), first + rank);
        }
        device_group_barrier();
        if (last)
        {
            if (place)
            {
                if (cls == CUT_CLASS)
                    g_cut_next += rank + 1;
                else
                    plan_set(child_bonds_at(base + cls) + 1, plan_at(child_bonds_at(base + cls) + 1) + rank + 1);
            }
            else if (cls != CUT_CLASS)
                plan_set(child_bonds_at(base + cls), plan_at(child_bonds_at(base + cls)) + rank + 1);
        }
        device_group_barrier();
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void layout_parent(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint tid = thread.x;
    uint p = group.x;
    // The exits and the loop bounds as uniform values: barriers follow.
    if (group_uniform(p >= control_at(CTL_PARENTS) || control_at(0) != 0 ? 1u : 0u, tid) != 0)
        return;
    uint4 parent = parents[p];
    if (group_uniform(parent.z < 2 ? 1u : 0u, tid) != 0)
        return;
    uint ci = parent.x;
    float4 r = old_islands_at(island_word(ci, 0));
    uint c0 = group_uniform(asuint(r.x), tid), c1 = group_uniform(asuint(r.y), tid), b0 = group_uniform(asuint(r.z), tid), b1 = group_uniform(asuint(r.w), tid);
    uint n = c1 - c0;
    uint base = parent.y;
    uint total = control_at(CTL_TOTAL);
    // 1. New chunk positions: each child's start in island order (thread 0, over the
    // children), then every member in parallel: child k's members (`offset`, in child order)
    // go to its start plus their place among them, as one pass in island order would.
    if (tid == 0)
    {
        sort_children(parent);
        uint at = 0;
        for (uint r = 0; r < parent.z; r++)
        {
            uint k = child_by_island(parent, r);
            plan_set(child_start_at(base + k), at);
            at += asuint(children[CHILD_WORDS * (base + k)].z);
        }
    }
    device_group_barrier();
    for (uint i = tid; i < n; i += LT)
    {
        // The child whose members hold slot i: offsets rise with the child number.
        uint lo = 0, hi = parent.z - 1;
        while (lo < hi)
        {
            uint mid = (lo + hi + 1) / 2;
            if (asuint(children[CHILD_WORDS * (base + mid)].y) <= i)
                lo = mid;
            else
                hi = mid - 1;
        }
        uint j = i - asuint(children[CHILD_WORDS * (base + lo)].y);
        plan_set(new_chunk_at(base + work[total + base + i]), plan_at(child_start_at(base + lo)) + j);
    }
    device_group_barrier();
    // Chunk statics, states, the resident tables' chunk entries; the static-solve slots
    // cleared (a new island's solve starts from rest, as after the host's split).
    for (uint m = tid; m < n; m += LT)
    {
        uint nk = plan_at(new_chunk_at(base + m));
        uint k = work[base + m];
        uint island = plan_at(child_island_at(base + k));
        for (uint w = 0; w < params.chunk_words; w++)
            chunks_set(chunk_word(c0 + nk, w), old_chunks_at(chunk_word(c0 + m, w)));
        set_chunk_island(c0 + nk, island);
        for (uint w = 0; w < 4; w++)
            state_set(4 * (c0 + nk) + w, old_state_at(4 * (c0 + m) + w));
        for (uint w = 0; w < STATICS_PER_CHUNK; w++)
            scratch_set(params.statics_base + STATICS_PER_CHUNK * (c0 + nk) + w, float4(0.0));
        if (params.resident != 0)
        {
            uint o = c0 + m, d = c0 + nk;
            ctab_set(params.props_base + d, old_ctab_at(params.props_base + o));
            ctab_set(params.eload_base + 2 * d, old_ctab_at(params.eload_base + 2 * o));
            ctab_set(params.eload_base + 2 * d + 1, old_ctab_at(params.eload_base + 2 * o + 1));
            ctab_set(params.pulse_count_base + d, old_ctab_at(params.pulse_count_base + o));
            for (uint w = 0; w < 3 * params.pulse_cap; w++)
                ctab_set(params.pulse_base + 3 * params.pulse_cap * d + w, old_ctab_at(params.pulse_base + 3 * params.pulse_cap * o + w));
        }
    }
    device_group_barrier();
    // 2. Bonds: each child's, in the parent's order, then the cut ones. Counts per child
    // (counted as they are ranked below), starts in island order
    // (thread 0, over the children), then each bond's place among its child's in bond
    // order, in blocks of the group's threads: as one pass in bond order would give.
    for (uint k = tid; k < parent.z; k += LT)
        plan_set(child_bonds_at(base + k), 0);
    device_group_barrier();
    bond_ranks(parent, base, c0, b0, b1, tid, false);
    device_group_barrier();
    if (tid == 0)
    {
        uint at = 0;
        for (uint r = 0; r < parent.z; r++)
        {
            uint k = child_by_island(parent, r);
            uint count = plan_at(child_bonds_at(base + k));
            plan_set(child_bonds_at(base + k), b0 + at);
            plan_set(child_bonds_at(base + k) + 1, b0 + at);
            at += count;
        }
        g_cut_first = b0 + at;
        g_cut_next = b0 + at;
    }
    device_group_barrier();
    bond_ranks(parent, base, c0, b0, b1, tid, true);
    // Copies: a child's bond with its chunks renamed and its island; a cut bond to its slot
    // after the children's, in no island, its loads and static stiffness zeroed.
    for (uint b = b0 + tid; b < b1; b += LT)
    {
        uint4 ids = asuint(old_bonds_at(bond_word(b, 14)));
        uint ka = work[base + ids.y - c0];
        uint kb = work[base + ids.z - c0];
        uint nb;
        uint4 new_ids;
        if (ka == kb)
        {
            nb = plan_at(bond_new_at(b));
            new_ids = uint4(ids.x, c0 + plan_at(new_chunk_at(base + ids.y - c0)), c0 + plan_at(new_chunk_at(base + ids.z - c0)), plan_at(child_island_at(base + ka)));
        }
        else
        {
            nb = plan_at(bond_new_at(b));
            new_ids = uint4(ids.x, ids.y, ids.z, NONE);
            for (uint w = 0; w < 3; w++)
                scratch_set(3 * nb + w, float4(0.0));
            scratch_set(params.statics_base + STATICS_PER_CHUNK * params.chunk_count + 2 * nb, float4(0.0));
            scratch_set(params.statics_base + STATICS_PER_CHUNK * params.chunk_count + 2 * nb + 1, float4(0.0));
        }
        for (uint w = 0; w < params.bond_words; w++)
            bonds_set(bond_word(nb, w), old_bonds_at(bond_word(b, w)));
        bonds_set(bond_word(nb, 14), asfloat(new_ids));
        for (uint w = 0; w < params.dyn_words; w++)
            dyn_set(params.dyn_words * nb + w, old_dyn_at(params.dyn_words * b + w));
    }
    // The children's bond ranges (first, end): `child_bonds_at` holds (first, next) after
    // placing, so `next` is the end.
    device_group_barrier();
    // 3. Incidence: each chunk keeps its slots (its old degree), its entries its
    // remaining bonds in the new order, padded with a cut bond of this split (or the
    // padding an earlier split left: an entry outside the parent's bonds).
    uint pad = NONE;
    if (g_cut_first < b1)
        pad = 2 * g_cut_first;
    else
    {
        for (uint e = old_index_at(c0); e < old_index_at(c1) && pad == NONE; e++)
        {
            uint entry = old_index_at(e);
            uint bond = entry >> 1;
            if (bond < b0 || bond >= b1)
                pad = entry;
        }
    }
    // Offsets in the new chunk order, each chunk its old degree: the degrees by new
    // position, then an exclusive scan over them (blocks of the group's threads).
    for (uint m = tid; m < n; m += LT)
        plan_set(degree_at(base + plan_at(new_chunk_at(base + m))), old_index_at(c0 + m + 1) - old_index_at(c0 + m));
    device_group_barrier();
    uint carry = old_index_at(c0);
    for (uint block = 0; block < n; block += LT)
    {
        uint i = block + tid;
        uint v = i < n ? plan_at(degree_at(base + i)) : 0u;
        g_scan64[tid] = v;
        GroupMemoryBarrierWithGroupSync();
        for (uint step = 1; step < LT; step <<= 1)
        {
            uint add = tid >= step ? g_scan64[tid - step] : 0u;
            GroupMemoryBarrierWithGroupSync();
            g_scan64[tid] += add;
            GroupMemoryBarrierWithGroupSync();
        }
        if (i < n)
            index_set(c0 + i, carry + g_scan64[tid] - v);
        uint block_total = g_scan64[LT - 1];
        GroupMemoryBarrierWithGroupSync();
        carry += block_total;
    }
    device_group_barrier();
    for (uint m = tid; m < n; m += LT)
    {
        uint nk = plan_at(new_chunk_at(base + m));
        uint at = index_at(c0 + nk);
        uint d = old_index_at(c0 + m + 1) - old_index_at(c0 + m);
        uint q = 0;
        for (uint e = old_index_at(c0 + m); e < old_index_at(c0 + m + 1); e++)
        {
            uint entry = old_index_at(e);
            uint bond = entry >> 1;
            if (bond < b0 || bond >= b1)
                continue;
            uint nb = plan_at(bond_new_at(bond));
            if (nb >= g_cut_first)
                continue;
            index_set(at + q, 2 * nb + (entry & 1));
            q++;
        }
        for (; q < d; q++)
            index_set(at + q, pad);
    }
    device_group_barrier();
    // 4. The children's island records (`island_record`, a resident quasi-static
    // frame's clusters: driven and settled), with the parent's rigid frame.
    for (uint k = tid; k < parent.z; k += LT)
    {
        uint slot = base + k;
        uint island = plan_at(child_island_at(slot));
        float4 head = children[CHILD_WORDS * slot];
        uint offset = asuint(head.y), count = asuint(head.z);
        bool anchored = asuint(head.w) != 0;
        float4 com = children[CHILD_WORDS * slot + 1];
        float3 i0 = children[CHILD_WORDS * slot + 2].xyz;
        float3 i1 = children[CHILD_WORDS * slot + 3].xyz;
        float3 i2 = children[CHILD_WORDS * slot + 4].xyz;
        float3 v = children[CHILD_WORDS * slot + 5].xyz;
        float3 w = children[CHILD_WORDS * slot + 6].xyz;
        // The first new chunk of this child, and the drift-removal weights (inertia
        // scales: `scale.x` of each chunk).
        uint first_new = NONE;
        float wmass = 0.0;
        float3 wmoment = float3(0.0);
        float su_mass = 0.0;
        float3 su = float3(0.0);
        bool scaled = false;
        for (uint j = 0; j < count; j++)
        {
            uint m = work[total + base + offset + j];
            uint nk = plan_at(new_chunk_at(base + m));
            first_new = min(first_new, nk);
            float4 ctr = old_chunks_at(chunk_word(c0 + m, 0));
            float mu = old_chunks_at(chunk_word(c0 + m, 7)).x;
            scaled = scaled || mu != 1.0;
            wmass += ctr.w * mu;
            wmoment += ctr.xyz * ctr.w * mu;
            su_mass += ctr.w;
            su += old_state_at(4 * (c0 + m)).xyz * ctr.w;
        }
        float3 wcom = scaled ? wmoment / wmass : com.xyz;
        float3 w0 = i0, w1 = i1, w2 = i2;
        if (scaled)
        {
            w0 = float3(0.0); w1 = float3(0.0); w2 = float3(0.0);
            for (uint j = 0; j < count; j++)
            {
                uint m = work[total + base + offset + j];
                float4 ctr = old_chunks_at(chunk_word(c0 + m, 0));
                float mu = old_chunks_at(chunk_word(c0 + m, 7)).x;
                float3 rr = ctr.xyz - wcom;
                float dd = dot(rr, rr);
                w0 += (old_chunks_at(chunk_word(c0 + m, 1)).xyz + (float3(dd, 0, 0) - rr.x * rr) * ctr.w) * mu;
                w1 += (old_chunks_at(chunk_word(c0 + m, 2)).xyz + (float3(0, dd, 0) - rr.y * rr) * ctr.w) * mu;
                w2 += (old_chunks_at(chunk_word(c0 + m, 3)).xyz + (float3(0, 0, dd) - rr.z * rr) * ctr.w) * mu;
            }
        }
        float3 v0, v1, v2, x0, x1, x2;
        inverse3(i0, i1, i2, v0, v1, v2);
        inverse3(w0, w1, w2, x0, x1, x2);
        Quat q = quat_of(frames[5 * p]);
        float3 pos = frames[5 * p + 4].xyz;
        if (!anchored)
            pos += rotate(q, su / su_mass);
        // World inertia times the angular velocity: R I R^T w.
        float3 iw = rotate(q, float3(dot(i0, inverse_rotate(q, w)), dot(i1, inverse_rotate(q, w)), dot(i2, inverse_rotate(q, w))));
        // `island_record`: a new cluster is driven only when no cluster integrates its own
        // rigid motion (the resident frame drives them all from its next frame).
        uint flags = (anchored ? ISLAND_FLAG_ANCHORED : 0u) | (params.driven != 0 ? ISLAND_FLAG_DRIVEN : 0u) | ISLAND_FLAG_SETTLED;
        uint bfirst = plan_at(child_bonds_at(slot));
        uint bend = plan_at(child_bonds_at(slot) + 1);
        float4 rec[25];
        for (uint t = 0; t < 25; t++)
            rec[t] = float4(0.0);
        rec[0] = asfloat(uint4(c0 + first_new, c0 + first_new + count, bfirst, bend));
        rec[1] = asfloat(uint4(flags, 0, 0, params.substeps));
        rec[2] = com;
        rec[3] = float4(i0, 0.0); rec[4] = float4(i1, 0.0); rec[5] = float4(i2, 0.0);
        rec[6] = float4(v0, 0.0); rec[7] = float4(v1, 0.0); rec[8] = float4(v2, 0.0);
        rec[9] = float4(wcom, scaled ? wmass : com.w);
        rec[10] = float4(x0, 0.0); rec[11] = float4(x1, 0.0); rec[12] = float4(x2, 0.0);
        rec[13] = quat_vec(q);
        rec[14] = float4(pos, 0.0);
        rec[16] = float4(anchored ? float3(0.0) : v, 0.0);
        rec[18] = float4(anchored ? float3(0.0) : w, 0.0);
        rec[19] = asfloat(uint4(0, 0, 0, plan_at(child_id_at(slot)))); // done_pad: the cluster id
        rec[21] = float4(0.0, 0.0, 0.0, float_bits(0x7f800000u)); // settled: load norm 0, never wakes
        rec[23] = float4(anchored ? float3(0.0) : iw, 0.0);
        for (uint t = 0; t < 25; t++)
            islands_set(island_word(island, t), rec[t]);
        if (params.resident != 0)
        {
            uint4 sl = spawn_slots[slot];
            float4 pr = spawn_extras[2 * slot + 1];
            uint at = params.engine_base + 3 * island;
            // y-w: its engine body (a fixed one too) and collider range (pass.slang retires it).
            float3 body_range = sl.w != 0 ? asfloat(uint3(sl.x, sl.y, sl.y + sl.z)) : float3(float_bits(NONE), 0.0, 0.0);
            if (anchored || sl.w == 0)
            {
                ctab_set(at, float4(float_bits(NONE), body_range));
                ctab_set(at + 1, float4(0.0));
                ctab_set(at + 2, float4(0.0));
            }
            else
            {
                ctab_set(at, float4(asfloat(sl.x), body_range));
                ctab_set(at + 1, float4(-pr.x, -pr.y, -pr.z, pr.w));
                ctab_set(at + 2, float4(spawn_extras[2 * slot].xyz, 0.0));
            }
        }
        gate_set(params.gate_flag_base + island, float4(0.0));
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void layout_moved(uint3 id: SV_DispatchThreadID)
{
    if (control_at(0) != 0 || control_at(CTL_PARENTS) == 0)
        return;
    uint i = id.x;
    // The sentinel island after the islands (batch control): a blank record.
    if (i == plan_at(0))
    {
        for (uint t = 0; t < params.island_words; t++)
            islands_set(island_word(i, t), float4(0.0));
        return;
    }
    if (i >= plan_at(0))
        return;
    uint e = plan_at(order_at(i));
    if (e >= control_at(CTL_ISLANDS) || e == i)
        return;
    for (uint t = 0; t < params.island_words; t++)
        islands_set(island_word(i, t), old_islands_at(island_word(e, t)));
    float4 r = old_islands_at(island_word(e, 0));
    uint c0 = asuint(r.x), c1 = asuint(r.y), b0 = asuint(r.z), b1 = asuint(r.w);
    for (uint c = c0; c < c1; c++)
        set_chunk_island(c, i);
    for (uint b = b0; b < b1; b++)
    {
        float4 ids = bonds_at(bond_word(b, 14));
        ids.w = asfloat(i);
        bonds_set(bond_word(b, 14), ids);
    }
    if (params.resident != 0)
        for (uint t = 0; t < 3; t++)
            ctab_set(params.engine_base + 3 * i + t, old_ctab_at(params.engine_base + 3 * e + t));
    gate_set(params.gate_flag_base + i, gate_at(params.gate_flag_base + e));
}

// After `layout_parent`: the cascade's copy of the island records in the new order (as the
// host's `split_resident` leaves its layout's islands): a kept or moved island's record from
// the copy as it was, a child's as `layout_parent` wrote it; the sentinel blank.
[shader("compute")]
[numthreads(64, 1, 1)]
void layout_copy(uint3 id: SV_DispatchThreadID)
{
    if (control_at(0) != 0 || control_at(CTL_PARENTS) == 0)
        return;
    uint i = id.x;
    uint n = plan_at(0);
    if (i > n)
        return;
    if (i == n)
    {
        for (uint t = 0; t < params.island_words; t++)
            island_copy_set(island_word(i, t), float4(0.0));
        return;
    }
    uint e = plan_at(order_at(i));
    if (e < control_at(CTL_ISLANDS))
    {
        for (uint t = 0; t < params.island_words; t++)
            island_copy_set(island_word(i, t), old_copy_at(island_word(e, t)));
    }
    else
    {
        for (uint t = 0; t < params.island_words; t++)
            island_copy_set(island_word(i, t), islands_at(island_word(i, t)));
    }
}
