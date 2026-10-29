// Shared by the Blast-model kernels (blast_cgls.slang, blast_material.slang): the solver's
// data layout and group reductions. The model is NVIDIA Blast's bond-impulse stress solver
// as the PhysX fork's GPU backend runs it (stressgpu/NvBlastExtStressGpu.cu); its f64
// reference is stress-ref blast_bond.rs, whose module docs give the math.
//
// Layout (src/blast.rs): nodes and bonds keep their indices for good (the structure's chunks
// and bonds; a support is a node whose weights in D are 0). Islands are Blast's (dynamic nodes
// over live bonds, cut at supports), each a list of its nodes and of its bonds, ascending
// (blast_topology.slang). Vectors are float4 with w unused, two per node or bond (angular,
// then linear).

module blast_common;

public static const uint BLAST_THREADS = 1024;
public static const uint NO_NODE = 0xFFFFFFFFu;
// A CSR ref's top bit: the node is the bond's second endpoint.
public static const uint REF_SECOND = 0x80000000u;

// Island report flags.
public static const uint REPORT_CONVERGED = 1u;
public static const uint REPORT_SKIPPED = 2u;
// The island's last solve converged or was skipped: its input may be skipped next time.
public static const uint REPORT_SETTLED = 4u;

public struct BlastParams
{
    public uint island_count;
    public uint max_iterations;
    public uint warm;            // keep the solution of the last solve (else zero it)
    public uint skip_settled;    // skip an island whose input repeats a settled solve's
    public float tolerance;
    public float recip_length;   // 1 / length scale
    public uint node_count;      // node slots
    public uint bond_count;      // bond slots
    // blast_material.slang: physical impulse = x s (length mass, length^2 mass).
    public float linear_scale;
    public float angular_scale;
    public float bend_gain_max;
    public uint fibres;
    public float dt;
    public float damage_rate;
    public uint apply_damage;
    // blast_native.slang: also stop when a step moves the forces by at most this fraction (0 off).
    public float force_tolerance;
    // blast_fracture.slang: gravity (world, Z-up), and where fracture.slang's `work` and `atom`
    // keep the bonds' alive flags, the bodies' dirty marks and the counters.
    public float4 gravity;
    public uint w_alive;
    public uint a_dirty;
    public uint a_counters;
    // blast_native_wide.slang: islands over wide_min nodes are solved by many groups; it holds up
    // to wide_rows of them, each over at most wide_blocks blocks.
    public uint wide_min;
    public uint wide_rows;
    public uint wide_blocks;
    public uint pad2;
};

// `topo` (blast_topology.slang): the totals after the per-node and per-block words: islands,
// nodes, bonds, a union that did not settle, and whether a bond broke since the islands were
// built.
public uint topo_total(uint node_count)
{
    return 10 * node_count + 3 * ((node_count + 1023) / 1024);
}
public uint topo_dirty(uint node_count)
{
    return topo_total(node_count) + 4;
}

// A dynamic node: its weights in D and its CSR range in `refs`.
public struct BlastNode
{
    public float w_ang;
    public float w_lin;
    public uint ref_begin;
    public uint ref_end;
    // Its position over the length scale (the native solve's rigid modes).
    public float4 position;
};

// A bond: its endpoints, compliance column scale, health (remaining area; 0 broken), offsets
// in units of the length scale.
public struct BlastBond
{
    public uint n0;
    public uint n1;
    public float s;
    public float health;
    public float4 o0;
    public float4 o1;
};

// Scratch: per node (rho, pi, q, mu, last input) x (angular, linear); then per bond
// t = S C^T D rho, angular and linear.
public static const uint NODE_VECTORS = 10;
public uint node_slot(uint node, uint vector)
{
    return node * NODE_VECTORS + vector;
}
public uint bond_slot(uint node_count, uint bond)
{
    return node_count * NODE_VECTORS + bond * 2;
}
public static const uint RHO = 0;
public static const uint PI = 2;
public static const uint Q = 4;
public static const uint MU = 6;
public static const uint LAST = 8;

groupshared float2 g_blast_red[BLAST_THREADS];
groupshared float2 g_blast_part[32];
groupshared float2 g_blast_sum;

// Sum of a float2 over the group, the same on every thread and every run (no float atomics):
// on Metal and CUDA SIMD-group sums then their partials in order; elsewhere (WGSL) a fixed
// tree, read back with `workgroupUniformLoad` so it may steer the group's barriers.
public float2 blast_group_sum(uint tid, float2 v)
{
    __target_switch
    {
    case metal:
    case cuda:
        float2 w = WaveActiveSum(v);
        if (tid % 32 == 0)
            g_blast_part[tid / 32] = w;
        GroupMemoryBarrierWithGroupSync();
        float2 s = g_blast_part[0];
        for (uint k = 1; k < BLAST_THREADS / 32; k++)
            s += g_blast_part[k];
        GroupMemoryBarrierWithGroupSync();
        return s;
    default:
        g_blast_red[tid] = v;
        GroupMemoryBarrierWithGroupSync();
        for (uint k = BLAST_THREADS / 2; k > 0; k >>= 1)
        {
            if (tid < k)
                g_blast_red[tid] += g_blast_red[tid + k];
            GroupMemoryBarrierWithGroupSync();
        }
        if (tid == 0)
            g_blast_sum = g_blast_red[0];
        float2 s = workgroupUniformLoad(g_blast_sum);
        GroupMemoryBarrierWithGroupSync();
        return s;
    }
}

public float norm2(float3 a, float3 b)
{
    return dot(a, a) + dot(b, b);
}
