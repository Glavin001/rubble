// Blast's islands on the GPU from the bonds' health (stressgpu `computeIslands`, which the host
// path runs on the CPU at every topology change): the dynamic nodes' connected components over
// live bonds, cut at supports, numbered by their lowest node, each as a list of its nodes and
// of its bonds in ascending order (blast_cgls.slang's `islands` and `lists`). Its f64
// reference is stress-ref blast_bond.rs `Islands::compute`. Every result is a function of the
// health alone: a union hooks each root under the smaller, counts are integer atomics, and the
// lists are sorted, so the order the threads ran in shows nowhere.
//
//   bt_gate                 one thread: unless a bond broke since the islands were built
//                           (`topo_dirty`, which blast_material.slang and the host set), every
//                           size below is zero, so a frame without a break rebuilds nothing;
//   bt_reset                one thread per node: its own root, its counts cleared;
//   bt_union                one thread per live bond between two dynamic nodes: lock-free union;
//   bt_flatten              one thread per node: its root (`label`);
//   bt_count_nodes, bt_count_bonds
//                           each root's dynamic nodes, and bonds (by the dynamic end's root);
//   bt_scan_blocks, bt_scan_top, bt_scan_add
//                           the roots with a live bond are the islands: exclusive scans of
//                           (is island, nodes, bonds) give each its id and list ranges, and
//                           the solve's dispatch size (`args`);
//   bt_islands              one thread per root: its island record, its report cleared;
//   bt_scatter_nodes, bt_scatter_bonds
//                           every node and bond into its island's list;
//   bt_sort                 one group per island: its lists sorted ascending (an all-ascending
//                           bitonic network in place: a comparator whose upper end lies past the
//                           list would move nothing, so lists of any length sort in place).

import common;
import blast_common;

[[vk::binding(0, 0)]] ConstantBuffer<BlastParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<BlastNode> nodes;
[[vk::binding(2, 0)]] StructuredBuffer<BlastBond> bonds;
[[vk::binding(3, 0)]] RWStructuredBuffer<Atomic<uint>> topo;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint4> islands;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> lists;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint4> report;
// Indirect sizes, 3 words each (A_*), and the island count (word 3).
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> args;

static const uint A_SOLVE = 0;   // (islands, 1, 1): blast_cgls.slang
static const uint A_COUNT = 3;
static const uint A_NODES = 4;   // the rebuild's kernels, zero on a frame without a break
static const uint A_BONDS = 7;
static const uint A_SCAN = 10;
static const uint A_ONE = 13;
static const uint A_SORT = 16;

static const uint SCAN_BLOCK = 1024;

// `topo` layout (blast.rs `topo_words` sizes it).
uint t_parent() { return 0; }
uint t_label() { return params.node_count; }
uint t_nodes() { return 2 * params.node_count; }
uint t_bonds() { return 3 * params.node_count; }
uint t_node_cursor() { return 4 * params.node_count; }
uint t_bond_cursor() { return 5 * params.node_count; }
uint t_scan() { return 6 * params.node_count; }        // 3 per node: island, nodes, bonds
uint t_island_of() { return 9 * params.node_count; }
uint t_blocks() { return 10 * params.node_count; }     // 3 per scan block
uint t_total() { return topo_total(params.node_count); } // islands, nodes, bonds, error, dirty

bool is_dynamic(uint i)
{
    return nodes[i].w_lin > 0.0;
}

// A bond that couples: live, and not between two supports.
bool couples(BlastBond b)
{
    return b.health > 0.0 && (is_dynamic(b.n0) || is_dynamic(b.n1));
}

uint find(uint i)
{
    uint p = topo[t_parent() + i].load();
    while (p != i)
    {
        uint g = topo[t_parent() + p].load();
        // Path halving: harmless under races (a parent only ever moves to a smaller root).
        if (g != p)
            topo[t_parent() + i].compareExchange(p, g);
        i = p;
        p = topo[t_parent() + i].load();
    }
    return i;
}

void set_args(uint at, uint x)
{
    args[at] = x;
    args[at + 1] = 1;
    args[at + 2] = 1;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void bt_gate(uint3 id: SV_DispatchThreadID)
{
    bool run = topo[topo_dirty(params.node_count)].exchange(0) != 0;
    set_args(A_NODES, run ? (params.node_count + 63) / 64 : 0);
    set_args(A_BONDS, run ? (params.bond_count + 63) / 64 : 0);
    set_args(A_SCAN, run ? (params.node_count + SCAN_BLOCK - 1) / SCAN_BLOCK : 0);
    set_args(A_ONE, run ? 1 : 0);
    // bt_scan_top sizes the sort when the rebuild runs.
    set_args(A_SORT, 0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bt_reset(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.node_count)
        return;
    topo[t_parent() + i].store(i);
    topo[t_nodes() + i].store(0);
    topo[t_bonds() + i].store(0);
    topo[t_node_cursor() + i].store(0);
    topo[t_bond_cursor() + i].store(0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bt_union(uint3 id: SV_DispatchThreadID)
{
    uint j = id.x;
    if (j >= params.bond_count)
        return;
    BlastBond b = bonds[j];
    if (!(b.health > 0.0) || !is_dynamic(b.n0) || !is_dynamic(b.n1))
        return;
    uint a = b.n0;
    uint c = b.n1;
    // Each failed exchange means another union moved a root to a smaller one: progress. The
    // bound only stops a runaway; reaching it is reported (`t_total() + 3`).
    for (uint guard = 0; guard < 4096; guard++)
    {
        a = find(a);
        c = find(c);
        if (a == c)
            return;
        // Hook the larger root under the smaller: a component's root is its lowest node.
        uint hi = max(a, c);
        uint lo = min(a, c);
        if (topo[t_parent() + hi].compareExchange(hi, lo) == hi)
            return;
        a = hi;
        c = lo;
    }
    topo[t_total() + 3].store(1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bt_flatten(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.node_count)
        return;
    topo[t_label() + i].store(is_dynamic(i) ? find(i) : NO_NODE);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bt_count_nodes(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.node_count || !is_dynamic(i))
        return;
    topo[t_nodes() + topo[t_label() + i].load()].add(1);
}

uint bond_root(BlastBond b)
{
    return topo[t_label() + (is_dynamic(b.n0) ? b.n0 : b.n1)].load();
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bt_count_bonds(uint3 id: SV_DispatchThreadID)
{
    uint j = id.x;
    if (j >= params.bond_count)
        return;
    BlastBond b = bonds[j];
    if (couples(b))
        topo[t_bonds() + bond_root(b)].add(1);
}

// An island: a root with a coupling bond (a lone dynamic node is in none).
uint3 scan_input(uint e)
{
    if (e >= params.node_count || topo[t_label() + e].load() != e)
        return uint3(0);
    uint nb = topo[t_bonds() + e].load();
    return nb > 0 ? uint3(1, topo[t_nodes() + e].load(), nb) : uint3(0);
}

groupshared uint3 g_scan[SCAN_BLOCK];

// Exclusive scan of `g_scan` over the group (SCAN_BLOCK threads, Hillis-Steele); returns the
// thread's exclusive prefix and leaves the block total in `total`.
uint3 group_exclusive_scan(uint tid, uint3 v, out uint3 total)
{
    g_scan[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint d = 1; d < SCAN_BLOCK; d <<= 1)
    {
        uint3 add = tid >= d ? g_scan[tid - d] : uint3(0);
        GroupMemoryBarrierWithGroupSync();
        g_scan[tid] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    total = g_scan[SCAN_BLOCK - 1];
    uint3 inclusive = g_scan[tid];
    GroupMemoryBarrierWithGroupSync();
    return inclusive - v;
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void bt_scan_blocks(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint tid = thread.x;
    uint e = group.x * SCAN_BLOCK + tid;
    uint3 v = scan_input(e);
    uint3 total;
    uint3 ex = group_exclusive_scan(tid, v, total);
    if (e < params.node_count)
    {
        topo[t_scan() + 3 * e].store(ex.x);
        topo[t_scan() + 3 * e + 1].store(ex.y);
        topo[t_scan() + 3 * e + 2].store(ex.z);
    }
    if (tid == 0)
    {
        topo[t_blocks() + 3 * group.x].store(total.x);
        topo[t_blocks() + 3 * group.x + 1].store(total.y);
        topo[t_blocks() + 3 * group.x + 2].store(total.z);
    }
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void bt_scan_top(uint3 thread: SV_GroupThreadID)
{
    uint tid = thread.x;
    uint blocks = (params.node_count + SCAN_BLOCK - 1) / SCAN_BLOCK;
    uint3 v = tid < blocks ? uint3(topo[t_blocks() + 3 * tid].load(), topo[t_blocks() + 3 * tid + 1].load(), topo[t_blocks() + 3 * tid + 2].load()) : uint3(0);
    uint3 total;
    uint3 ex = group_exclusive_scan(tid, v, total);
    if (tid < blocks)
    {
        topo[t_blocks() + 3 * tid].store(ex.x);
        topo[t_blocks() + 3 * tid + 1].store(ex.y);
        topo[t_blocks() + 3 * tid + 2].store(ex.z);
    }
    if (tid == 0)
    {
        topo[t_total()].store(total.x);
        topo[t_total() + 1].store(total.y);
        topo[t_total() + 2].store(total.z);
        set_args(A_SOLVE, total.x);
        args[A_COUNT] = total.x;
        set_args(A_SORT, total.x);
    }
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void bt_scan_add(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint e = group.x * SCAN_BLOCK + thread.x;
    if (e >= params.node_count)
        return;
    for (uint k = 0; k < 3; k++)
        topo[t_scan() + 3 * e + k].add(topo[t_blocks() + 3 * group.x + k].load());
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bt_islands(uint3 id: SV_DispatchThreadID)
{
    uint e = id.x;
    if (e >= params.node_count)
        return;
    uint3 v = scan_input(e);
    if (v.x == 0)
        return;
    uint k = topo[t_scan() + 3 * e].load();
    uint n0 = topo[t_scan() + 3 * e + 1].load();
    uint b0 = topo[t_total() + 1].load() + topo[t_scan() + 3 * e + 2].load();
    islands[k] = uint4(n0, n0 + v.y, b0, b0 + v.z);
    topo[t_island_of() + e].store(k);
    // Island ids change with the topology: no island is settled across it (the warm-started
    // solution stays, by bond).
    report[k] = uint4(0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bt_scatter_nodes(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.node_count || !is_dynamic(i))
        return;
    uint r = topo[t_label() + i].load();
    if (topo[t_bonds() + r].load() == 0)
        return;
    uint k = topo[t_island_of() + r].load();
    lists[islands[k].x + topo[t_node_cursor() + r].add(1)] = i;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bt_scatter_bonds(uint3 id: SV_DispatchThreadID)
{
    uint j = id.x;
    if (j >= params.bond_count)
        return;
    BlastBond b = bonds[j];
    if (!couples(b))
        return;
    uint r = bond_root(b);
    uint k = topo[t_island_of() + r].load();
    lists[islands[k].z + topo[t_bond_cursor() + r].add(1)] = j;
}

// Sort lists[begin, begin + n) ascending, by the group: an all-ascending bitonic network over
// the next power of two, each comparator skipped when its upper end lies past the list.
void sort_list(uint tid, uint begin, uint n)
{
    uint span = 1;
    while (span < n)
        span <<= 1;
    for (uint k = 2; k <= span; k <<= 1)
    {
        for (uint j = k >> 1; j > 0; j >>= 1)
        {
            for (uint t = tid; t < span; t += BLAST_THREADS)
            {
                // The first step of each merge mirrors its block, the rest are half-cleaners.
                uint l = j == (k >> 1) ? t ^ (k - 1) : t ^ j;
                if (l > t && l < n)
                {
                    uint a = lists[begin + t];
                    uint b = lists[begin + l];
                    if (a > b)
                    {
                        lists[begin + t] = b;
                        lists[begin + l] = a;
                    }
                }
            }
            device_group_barrier();
        }
    }
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void bt_sort(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint tid = thread.x;
    uint4 r = islands[group.x];
    uint4 u = group_uniform4(uint4(r.x, r.y - r.x, r.z, r.w - r.z), tid);
    sort_list(tid, u.x, u.y);
    sort_list(tid, u.z, u.w);
}
