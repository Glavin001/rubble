// Blast's bond-impulse stress solve, the host path of the PhysX fork's GPU backend
// (stressgpu/NvBlastExtStressGpu.cu built without PHYSX_RESIDENT_DESTRUCTION: what
// ExtStressSolver runs with gpuStressSolver on, vibe-land's `blast` backend). Its f64
// reference is stress-ref blast_bond.rs `BlastSolver::solve`.
//
// Node-space CGLS (StressNodeOperator.cuh): with B = D C S, solve min |B x - b|, b = -D^-1 a
// (linear part over the length scale), by x = x0 + B^T mu and L = B B^T:
//
//     rho = b - B x0;  pi = q = mu = 0
//     each iteration: t = B^T rho;  z = |t|^2 (each bond once)
//                     stop when z <= tol^2 |b|^2 (the island's own target)
//                     beta = z / z_prev (0 on the first);  pi = rho + beta pi;  q = B t + beta q
//                     stop (unconverged) unless |q|^2 is positive and finite
//                     alpha = z / |q|^2;  mu += alpha pi;  rho -= alpha q
//     then x = x0 + B^T mu.
//
// The device solves every island at once, one node or bond per thread with per-island slot
// atomics; here one group solves one island with fixed-order group sums instead, the same
// recurrence with the same scalars (no float atomics, so a run repeats bit for bit). The
// direction restarts every solve (Blast's warmth 1): only x carries over.
//
// Nodes and bonds keep their indices for good (the scene's chunks and bonds); an island is a
// list of each (`islands`, `lists`), so a break rewrites lists and nothing moves.
//
// One group per island (`islands`; `args` sizes the dispatch).

import common;
import blast_common;

[[vk::binding(0, 0)]] ConstantBuffer<BlastParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> islands;     // `lists` ranges: nodes [x, y), bonds [z, w)
[[vk::binding(2, 0)]] StructuredBuffer<BlastNode> nodes;
[[vk::binding(3, 0)]] StructuredBuffer<uint> refs;         // bond slot | REF_SECOND
[[vk::binding(4, 0)]] RWStructuredBuffer<BlastBond> bonds;
[[vk::binding(5, 0)]] StructuredBuffer<float4> accel;      // per node slot: angular, linear
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> x;        // per bond slot: angular, linear
[[vk::binding(7, 0)]] RWStructuredBuffer<float4> scratch;
// Per island: iterations, flags (REPORT_*), the last test's |t|^2 and the target (bits).
[[vk::binding(8, 0)]] RWStructuredBuffer<uint4> report;
// Each island's nodes then its bonds, by index, ascending (blast_topology.slang builds them).
[[vk::binding(9, 0)]] StructuredBuffer<uint> lists;

// t_j = S C^T D v for a bond, v the node vector at scratch slot `vec` (B^T v).
void bond_term(BlastBond b, uint vec, out float3 ang, out float3 lin)
{
    // A support's weights are 0 (its scratch is never written, and stays 0).
    BlastNode m0 = nodes[b.n0];
    BlastNode m1 = nodes[b.n1];
    float3 a0 = scratch[node_slot(b.n0, vec)].xyz * m0.w_ang;
    float3 l0 = scratch[node_slot(b.n0, vec) + 1].xyz * m0.w_lin;
    float3 a1 = scratch[node_slot(b.n1, vec)].xyz * m1.w_ang;
    float3 l1 = scratch[node_slot(b.n1, vec) + 1].xyz * m1.w_lin;
    ang = (a0 - a1) * b.s;
    lin = ((l0 - l1) + (cross(b.o0.xyz, a0) - cross(b.o1.xyz, a1))) * b.s;
}

// (D C S y)_node for bond vectors y: the bond terms in `x` (from_x) or the scratch's t.
void gather(uint node, bool from_x, out float3 ang, out float3 lin)
{
    BlastNode n = nodes[node];
    float3 acc_a = float3(0.0);
    float3 acc_l = float3(0.0);
    for (uint r = n.ref_begin; r < n.ref_end; r++)
    {
        uint ref = refs[r];
        uint j = ref & ~REF_SECOND;
        BlastBond b = bonds[j];
        if (!(b.health > 0.0))
            continue;
        float3 ya;
        float3 yl;
        if (from_x)
        {
            ya = x[j * 2].xyz * b.s;
            yl = x[j * 2 + 1].xyz * b.s;
        }
        else
        {
            uint t = bond_slot(params.node_count, j);
            ya = scratch[t].xyz * b.s;
            yl = scratch[t + 1].xyz * b.s;
        }
        if ((ref & REF_SECOND) != 0)
        {
            acc_a += cross(b.o1.xyz, yl) - ya;
            acc_l -= yl;
        }
        else
        {
            acc_a += ya - cross(b.o0.xyz, yl);
            acc_l += yl;
        }
    }
    ang = acc_a * n.w_ang;
    lin = acc_l * n.w_lin;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void blast_solve(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    uint k = group.x;
    if (k >= params.island_count)
        return;
    uint4 range = islands[k];
    uint n0 = range.x;
    uint n1 = range.y;
    uint b0 = range.z;
    uint b1 = range.w;
    uint4 last = report[k];
    uint last_flags = group_uniform(last.y, tid);
    bool warm = params.warm != 0;

    // A settled island: its last solve converged and its input repeats bit for bit. Its
    // bonds keep their impulses (the device's islandSkip mask).
    if (params.skip_settled != 0 && warm && (last_flags & REPORT_SETTLED) != 0)
    {
        float differ = 0.0;
        for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
        {
            uint i = lists[p];
            if (any(accel[i * 2] != scratch[node_slot(i, LAST)]) || any(accel[i * 2 + 1] != scratch[node_slot(i, LAST) + 1]))
                differ = 1.0;
        }
        if (blast_group_sum(tid, float2(differ, 0.0)).x == 0.0)
        {
            if (tid == 0)
                report[k] = uint4(0, REPORT_SKIPPED | REPORT_SETTLED, last.z, last.w);
            return;
        }
    }

    // b = -D^-1 a; rho = b - B x0; the solve's vectors start at zero.
    float target = 0.0;
    for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
    {
        uint i = lists[p];
        BlastNode n = nodes[i];
        float4 a_ang = accel[i * 2];
        float4 a_lin = accel[i * 2 + 1];
        float ia = n.w_ang > 0.0 ? 1.0 / n.w_ang : 1.0;
        float il = n.w_lin > 0.0 ? params.recip_length / n.w_lin : params.recip_length;
        float3 b_ang = -a_ang.xyz * ia;
        float3 b_lin = -a_lin.xyz * il;
        target += norm2(b_ang, b_lin);
        scratch[node_slot(i, RHO)] = float4(b_ang, 0.0);
        scratch[node_slot(i, RHO) + 1] = float4(b_lin, 0.0);
        scratch[node_slot(i, LAST)] = a_ang;
        scratch[node_slot(i, LAST) + 1] = a_lin;
        for (uint v = PI; v < LAST; v++)
            scratch[node_slot(i, v)] = float4(0.0);
    }
    if (!warm)
    {
        for (uint q = b0 + tid; q < b1; q += BLAST_THREADS)
        {
            uint j = lists[q];
            x[j * 2] = float4(0.0);
            x[j * 2 + 1] = float4(0.0);
        }
    }
    device_group_barrier();
    if (warm)
    {
        for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
        {
            uint i = lists[p];
            float3 ya;
            float3 yl;
            gather(i, true, ya, yl);
            scratch[node_slot(i, RHO)] = float4(scratch[node_slot(i, RHO)].xyz - ya, 0.0);
            scratch[node_slot(i, RHO) + 1] = float4(scratch[node_slot(i, RHO) + 1].xyz - yl, 0.0);
        }
    }
    float target2 = blast_group_sum(tid, float2(target, 0.0)).x * params.tolerance * params.tolerance;
    device_group_barrier();

    uint iterations = 0;
    bool converged = false;
    float z = 0.0;
    float z_prev = 0.0;
    for (uint it = 0; it < params.max_iterations; it++)
    {
        // t = B^T rho, and z = |t|^2 over the island's bonds (each bond once).
        float zt = 0.0;
        for (uint q = b0 + tid; q < b1; q += BLAST_THREADS)
        {
            uint j = lists[q];
            BlastBond b = bonds[j];
            float3 ta = float3(0.0);
            float3 tl = float3(0.0);
            if (b.health > 0.0)
                bond_term(b, RHO, ta, tl);
            zt += norm2(ta, tl);
            uint t = bond_slot(params.node_count, j);
            scratch[t] = float4(ta, 0.0);
            scratch[t + 1] = float4(tl, 0.0);
        }
        z = blast_group_sum(tid, float2(zt, 0.0)).x;
        if (z <= target2)
        {
            converged = true;
            break;
        }
        device_group_barrier();
        float beta = it != 0 && z_prev > 0.0 ? z / z_prev : 0.0;
        // pi = rho + beta pi;  q = B t + beta q.
        float qq = 0.0;
        for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
        {
            uint i = lists[p];
            float3 wa;
            float3 wl;
            gather(i, false, wa, wl);
            float3 pa = scratch[node_slot(i, RHO)].xyz + scratch[node_slot(i, PI)].xyz * beta;
            float3 pl = scratch[node_slot(i, RHO) + 1].xyz + scratch[node_slot(i, PI) + 1].xyz * beta;
            float3 qa = wa + scratch[node_slot(i, Q)].xyz * beta;
            float3 ql = wl + scratch[node_slot(i, Q) + 1].xyz * beta;
            scratch[node_slot(i, PI)] = float4(pa, 0.0);
            scratch[node_slot(i, PI) + 1] = float4(pl, 0.0);
            scratch[node_slot(i, Q)] = float4(qa, 0.0);
            scratch[node_slot(i, Q) + 1] = float4(ql, 0.0);
            qq += norm2(qa, ql);
        }
        float q2 = blast_group_sum(tid, float2(qq, 0.0)).x;
        if (!(q2 > 0.0) || !isfinite(q2))
            break;
        z_prev = z;
        float alpha = z / q2;
        // mu += alpha pi;  rho -= alpha q (each thread its own nodes, as above).
        for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
        {
            uint i = lists[p];
            scratch[node_slot(i, MU)] = float4(scratch[node_slot(i, MU)].xyz + scratch[node_slot(i, PI)].xyz * alpha, 0.0);
            scratch[node_slot(i, MU) + 1] = float4(scratch[node_slot(i, MU) + 1].xyz + scratch[node_slot(i, PI) + 1].xyz * alpha, 0.0);
            scratch[node_slot(i, RHO)] = float4(scratch[node_slot(i, RHO)].xyz - scratch[node_slot(i, Q)].xyz * alpha, 0.0);
            scratch[node_slot(i, RHO) + 1] = float4(scratch[node_slot(i, RHO) + 1].xyz - scratch[node_slot(i, Q) + 1].xyz * alpha, 0.0);
        }
        iterations++;
        device_group_barrier();
    }
    device_group_barrier();

    // x = x0 + B^T mu over the island's live bonds.
    for (uint q = b0 + tid; q < b1; q += BLAST_THREADS)
    {
        uint j = lists[q];
        BlastBond b = bonds[j];
        if (!(b.health > 0.0))
        {
            x[j * 2] = float4(0.0);
            x[j * 2 + 1] = float4(0.0);
            continue;
        }
        float3 ta;
        float3 tl;
        bond_term(b, MU, ta, tl);
        x[j * 2] = float4(x[j * 2].xyz + ta, 0.0);
        x[j * 2 + 1] = float4(x[j * 2 + 1].xyz + tl, 0.0);
    }
    if (tid == 0)
        report[k] = uint4(iterations, (converged ? REPORT_CONVERGED | REPORT_SETTLED : 0u), asuint(z), asuint(target2));
}
