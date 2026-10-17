// Explicit substeps of the stress solver (stress-ref `ReferenceSolver::substep`,
// explicit mode, intact bonds, anchored clusters). Each substep has two phases:
//
//   bond phase   per bond: generalised deformation d and rate from the chunks' hidden
//                displacements (u, theta) and velocities (v, w) at the start of the
//                substep, force q = K d + C rate (local frame t1, t2, n), and the loads
//                it puts on both chunks (body frame).
//   chunk phase  per chunk: the loads of its bonds gathered in a fixed order (CSR, no
//                atomics), plus its external load, then the central-difference update
//                (velocities at half steps) by support kind.
//
// The arithmetic lives in pure functions (`bond_math`, `chunk_math`) shared by every
// kernel shape, so all shapes compute identical values:
//
//   bond_forces, chunk_integrate  one dispatch per phase per substep (all chunks):
//                                 any island size, uses the whole GPU per substep;
//   island_substeps               one threadgroup per island, all substeps of a
//                                 segment in one dispatch, state in device memory;
//   island_shared_substeps        the same with the island's state and bond loads in
//                                 threadgroup memory (islands up to SHARED_CHUNKS
//                                 chunks and SHARED_BONDS bonds).
//
// Mirrors bond.rs (`kinematics`, `chunk_loads`), joint.rs (intact force and dashpots)
// and solver.rs (`integrate_cluster`) in f32.

static const uint SUPPORT_NONE = 0;
static const uint SUPPORT_FIXED = 1;
static const uint SUPPORT_PINNED = 2;

struct StepParams
{
    uint chunk_count;
    uint bond_count;
    float dt;
    uint substeps; // island kernels: substeps per dispatch
};

// Static data of a bond (body frame of its cluster).
struct Bond
{
    uint4 chunks;    // x: a, y: b
    float4 t1;       // local frame: tangents and normal
    float4 t2;
    float4 normal;
    float4 ra;       // centroid - centre of a
    float4 rb;       // centroid - centre of b
    float4 k_lin;    // stiffness: ks, ks, kn
    float4 k_ang;    // kb_t1, kb_t2, kt
    float4 c_lin;    // dashpots, same components
    float4 c_ang;
    float4 section;  // area, s_t1, s_t2 (section moduli)
};

// Static data of a chunk (body frame of its cluster).
struct Chunk
{
    float4 inv_inertia0; // rows of the inverse rotational inertia
    float4 inv_inertia1;
    float4 inv_inertia2;
    float4 force;        // external force (gravity, body frame); w: 1 / (mass * inertia scale)
    float4 moment;       // external moment (body frame); w: 1 / inertia scale
    uint4 support;       // x: SUPPORT_*
};

[[vk::binding(0, 0)]] ConstantBuffer<StepParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<Bond> bonds;
[[vk::binding(2, 0)]] StructuredBuffer<Chunk> chunks;
[[vk::binding(3, 0)]] StructuredBuffer<uint> chunk_bond_start; // chunk_count + 1 offsets
[[vk::binding(4, 0)]] StructuredBuffer<uint> chunk_bonds;      // 2 * bond + side (0: a, 1: b)
// Hidden state, 4 per chunk: u (w: largest bond stress, Pa), theta, v, w.
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> state;
// 3 per bond: f on a (w: peak normal stress, Pa), moment on a, moment on b (force on b: -f).
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> bond_loads;
// Per island: chunk range [x, y), bond range [z, w) (chunks and bonds stored by island).
[[vk::binding(7, 0)]] StructuredBuffer<uint4> islands;

static const uint U = 0;
static const uint THETA = 1;
static const uint V = 2;
static const uint W = 3;

// A chunk's hidden state.
struct ChunkState
{
    float3 u;
    float3 theta;
    float3 v;
    float3 w;
};

// The loads one bond puts on its chunks.
struct BondLoads
{
    float4 fa; // w: peak normal stress
    float3 ma;
    float3 mb;
};

float3 to_local(Bond b, float3 x)
{
    return float3(dot(x, b.t1.xyz), dot(x, b.t2.xyz), dot(x, b.normal.xyz));
}

float3 to_body(Bond b, float3 l)
{
    return b.t1.xyz * l.x + b.t2.xyz * l.y + b.normal.xyz * l.z;
}

BondLoads bond_math(Bond b, ChunkState sa, ChunkState sb)
{
    float3 ra = b.ra.xyz;
    float3 rb = b.rb.xyz;
    // Kinematics: relative displacement of the bond centroid and relative rotation.
    float3 delta = (sb.u + cross(sb.theta, rb)) - (sa.u + cross(sa.theta, ra));
    float3 d_lin = to_local(b, delta);
    float3 d_ang = to_local(b, sb.theta - sa.theta);
    float3 rate_delta = (sb.v + cross(sb.w, rb)) - (sa.v + cross(sa.w, ra));
    float3 r_lin = to_local(b, rate_delta);
    float3 r_ang = to_local(b, sb.w - sa.w);
    // Intact joint: elastic force plus stiffness-proportional dashpots.
    float3 q_lin = b.k_lin.xyz * d_lin + b.c_lin.xyz * r_lin;
    float3 q_ang = b.k_ang.xyz * d_ang + b.c_ang.xyz * r_ang;
    // Peak normal stress of the elastic part: axial plus both bending moments.
    float3 e_lin = b.k_lin.xyz * d_lin;
    float3 e_ang = b.k_ang.xyz * d_ang;
    float stress = abs(e_lin.z) / b.section.x + abs(e_ang.x) / b.section.y + abs(e_ang.y) / b.section.z;
    // Loads on the chunks (body frame): equal and opposite, moments about each centre.
    float3 f = to_body(b, q_lin);
    float3 m = to_body(b, q_ang);
    BondLoads o;
    o.fa = float4(f, stress);
    o.ma = m + cross(ra, f);
    o.mb = -m + cross(rb, -f);
    return o;
}

// Central-difference update of one chunk under internal loads (fi, mi), summed from
// zero in bond order, plus its external load, as solver.rs does (`f_ext + fi`).
ChunkState chunk_math(Chunk ch, ChunkState st, float3 fi, float3 mi, float dt)
{
    float3 f = ch.force.xyz + fi;
    float3 m = ch.moment.xyz + mi;
    float3 alpha = float3(dot(ch.inv_inertia0.xyz, m), dot(ch.inv_inertia1.xyz, m), dot(ch.inv_inertia2.xyz, m)) * ch.moment.w;
    uint support = ch.support.x;
    if (support == SUPPORT_FIXED)
    {
        st.v = float3(0.0);
        st.w = float3(0.0);
    }
    else
    {
        st.w += alpha * dt;
        st.theta += st.w * dt;
        if (support == SUPPORT_PINNED)
        {
            st.v = float3(0.0);
        }
        else
        {
            st.v += f * (dt * ch.force.w);
            st.u += st.v * dt;
        }
    }
    return st;
}

// ---------------------------------------------------------------- device memory

ChunkState load_state(uint c)
{
    ChunkState s;
    s.u = state[4 * c + U].xyz;
    s.theta = state[4 * c + THETA].xyz;
    s.v = state[4 * c + V].xyz;
    s.w = state[4 * c + W].xyz;
    return s;
}

void store_state(uint c, ChunkState s, float stress)
{
    state[4 * c + U] = float4(s.u, stress);
    state[4 * c + THETA] = float4(s.theta, 0.0);
    state[4 * c + V] = float4(s.v, 0.0);
    state[4 * c + W] = float4(s.w, 0.0);
}

void bond_phase(uint i)
{
    Bond b = bonds[i];
    BondLoads o = bond_math(b, load_state(b.chunks.x), load_state(b.chunks.y));
    bond_loads[3 * i + 0] = o.fa;
    bond_loads[3 * i + 1] = float4(o.ma, 0.0);
    bond_loads[3 * i + 2] = float4(o.mb, 0.0);
}

void chunk_phase(uint c, float dt)
{
    float3 fi = float3(0.0);
    float3 mi = float3(0.0);
    float stress = 0.0;
    for (uint k = chunk_bond_start[c]; k < chunk_bond_start[c + 1]; k++)
    {
        uint e = chunk_bonds[k];
        uint bond = e >> 1;
        float4 fa = bond_loads[3 * bond];
        if ((e & 1) == 0)
        {
            fi += fa.xyz;
            mi += bond_loads[3 * bond + 1].xyz;
        }
        else
        {
            fi += -fa.xyz;
            mi += bond_loads[3 * bond + 2].xyz;
        }
        stress = max(stress, fa.w);
    }
    store_state(c, chunk_math(chunks[c], load_state(c), fi, mi, dt), stress);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bond_forces(uint3 id: SV_DispatchThreadID)
{
    if (id.x < params.bond_count)
        bond_phase(id.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void chunk_integrate(uint3 id: SV_DispatchThreadID)
{
    if (id.x < params.chunk_count)
        chunk_phase(id.x, params.dt);
}

static const uint ISLAND_THREADS = 256;

[shader("compute")]
[numthreads(256, 1, 1)]
void island_substeps(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint4 island = islands[group.x];
    float dt = params.dt;
    for (uint s = 0; s < params.substeps; s++)
    {
        for (uint i = island.z + thread.x; i < island.w; i += ISLAND_THREADS)
            bond_phase(i);
        // Bond loads written to device memory must be visible to the whole group.
        AllMemoryBarrierWithGroupSync();
        for (uint c = island.x + thread.x; c < island.y; c += ISLAND_THREADS)
            chunk_phase(c, dt);
        AllMemoryBarrierWithGroupSync();
    }
}

// ------------------------------------------------------------ threadgroup memory

static const uint SHARED_CHUNKS = 160;
static const uint SHARED_BONDS = 400;

groupshared float3 g_u[SHARED_CHUNKS];
groupshared float3 g_theta[SHARED_CHUNKS];
groupshared float3 g_v[SHARED_CHUNKS];
groupshared float3 g_w[SHARED_CHUNKS];
groupshared float4 g_fa[SHARED_BONDS];
groupshared float3 g_ma[SHARED_BONDS];
groupshared float3 g_mb[SHARED_BONDS];

ChunkState shared_state(uint local)
{
    ChunkState s;
    s.u = g_u[local];
    s.theta = g_theta[local];
    s.v = g_v[local];
    s.w = g_w[local];
    return s;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void island_shared_substeps(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint4 island = islands[group.x];
    uint chunk0 = island.x;
    uint chunk_count = island.y - island.x;
    uint bond0 = island.z;
    uint bond_count = island.w - island.z;
    float dt = params.dt;
    for (uint c = thread.x; c < chunk_count; c += ISLAND_THREADS)
    {
        ChunkState s = load_state(chunk0 + c);
        g_u[c] = s.u;
        g_theta[c] = s.theta;
        g_v[c] = s.v;
        g_w[c] = s.w;
    }
    GroupMemoryBarrierWithGroupSync();
    // SHARED_CHUNKS <= ISLAND_THREADS: each thread owns at most one chunk.
    float peak_stress = 0.0;
    for (uint s = 0; s < params.substeps; s++)
    {
        for (uint i = thread.x; i < bond_count; i += ISLAND_THREADS)
        {
            Bond b = bonds[bond0 + i];
            BondLoads o = bond_math(b, shared_state(b.chunks.x - chunk0), shared_state(b.chunks.y - chunk0));
            g_fa[i] = o.fa;
            g_ma[i] = o.ma;
            g_mb[i] = o.mb;
        }
        GroupMemoryBarrierWithGroupSync();
        uint c = thread.x;
        if (c < chunk_count)
        {
            float3 fi = float3(0.0);
            float3 mi = float3(0.0);
            float peak = 0.0;
            for (uint k = chunk_bond_start[chunk0 + c]; k < chunk_bond_start[chunk0 + c + 1]; k++)
            {
                uint e = chunk_bonds[k];
                uint bond = (e >> 1) - bond0;
                float4 fa = g_fa[bond];
                if ((e & 1) == 0)
                {
                    fi += fa.xyz;
                    mi += g_ma[bond];
                }
                else
                {
                    fi += -fa.xyz;
                    mi += g_mb[bond];
                }
                peak = max(peak, fa.w);
            }
            ChunkState next = chunk_math(chunks[chunk0 + c], shared_state(c), fi, mi, dt);
            g_u[c] = next.u;
            g_theta[c] = next.theta;
            g_v[c] = next.v;
            g_w[c] = next.w;
            peak_stress = peak;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    uint c = thread.x;
    if (c < chunk_count)
        store_state(chunk0 + c, shared_state(c), params.substeps > 0 ? peak_stress : state[4 * (chunk0 + c)].w);
}
