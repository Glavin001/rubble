// The resident (native) Blast solve of PhysX's GPU destruction stage, one group per island
// (stressgpu/detail/StressComponentIteration.cuh `componentStressSolve`, without rotational or
// shear stiffness; vibe-land's `native` backend). Its f64/f32 reference, with the full recurrence
// in its docs, is stress-ref blast_bond/native.rs.
//
// Projected PCG in node space on L = B B^T (B = D C S, D the native mass weighting):
//   r = b - B x0 (warm) or b;  delta^2 = tol^2 |b|^2;  a zero load retires with zero forces;
//   each iteration: r projected off a free island's rigid modes; num = |B^T r|^2 with the
//   relative residual |r|^2 <= delta^2 also required; passing after an update or on a warm
//   start, the true residual rebuilt (reliable residual) and both tests taken again; 512
//   iterations without a 1% gain stop; converged; the force test (optional); g = r on iteration
//   0, else the two-step Chebyshev block-Jacobi polynomial, projected and scaled by its largest
//   component; gamma = r.g > 0; beta = gamma / gamma_prev from iteration 2; p = g + beta p;
//   q = L p, den = |B^T p|^2 > 0; u += alpha p, r -= alpha q;  then x = x0 + B^T u.
//
// Each node's 6x6 block of L is [[A, -[k]x], [[k]x, c I]] (StressNativeRigidInverse.cuh):
// A = dA^2 sum s^2 ((1 + |o|^2) I - o o^T), k = dA dL sum s^2 o, c = dL^2 sum s^2 over its live
// bonds (o its own offset); its inverse by the Schur complement S = A + [k]x^2 / c.
//
// The island's lists as blast_cgls.slang's (blast_topology.slang builds them); one group per
// island (`args` sizes the dispatch).

#include "blast_native_ops.slangh"

groupshared float3 g_omega;
groupshared float3 g_t;

// Project node vector `v` of island k's nodes [n0, n1) off its rigid modes (`rotations` and
// `free` uniform; blast_motion.slang built the frame).
void project(uint tid, uint k, uint rotations, uint n0, uint n1, uint v, bool free)
{
    if (!free)
        return;
    float s[6];
    for (uint z = 0; z < 6; z++)
        s[z] = 0.0;
    for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
        frame_terms(k, rotations, lists[p], v, s);
    float2 a = blast_group_sum(tid, float2(s[0], s[1]));
    float2 b = blast_group_sum(tid, float2(s[2], s[3]));
    float2 c = blast_group_sum(tid, float2(s[4], s[5]));
    if (tid == 0)
    {
        float sum[6] = { a.x, a.y, b.x, b.y, c.x, c.y };
        float3 omega;
        float3 t;
        frame_solve(k, rotations, sum, omega, t);
        g_omega = omega;
        g_t = t;
    }
    GroupMemoryBarrierWithGroupSync();
    float3 omega = g_omega;
    float3 t = g_t;
    for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
        frame_apply(lists[p], v, omega, t);
    device_group_barrier();
}

// t into the bond scratch for node vector v over the island's bonds, and sum |t|^2.
float bond_pass(uint tid, uint b0, uint b1, uint v)
{
    float z = 0.0;
    for (uint q = b0 + tid; q < b1; q += BLAST_THREADS)
    {
        uint j = lists[q];
        BlastBond b = bonds[j];
        float3 ta = float3(0.0);
        float3 tl = float3(0.0);
        if (b.health > 0.0)
            bond_term(b, j, v, ta, tl);
        z += norm2(ta, tl);
        scratch[bs(j)] = float4(ta, 0.0);
        scratch[bs(j) + 1] = float4(tl, 0.0);
    }
    return blast_group_sum(tid, float2(z, 0.0)).x;
}

// out = L v over the island's nodes (bond_pass for v first, then a barrier), and |out|^2.
float node_pass(uint tid, uint n0, uint n1, uint out_v)
{
    float s = 0.0;
    for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
    {
        uint i = lists[p];
        float3 a;
        float3 l;
        gather(i, true, false, a, l);
        set_v(i, out_v, a, l);
        s += norm2(a, l);
    }
    return s;
}

// The true residual b - D C S (x0 + (with_u ? B^T u : 0)) into R, and |r|^2.
float rebuild(uint tid, uint n0, uint n1, bool with_u)
{
    float s = 0.0;
    for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
    {
        uint i = lists[p];
        float3 a;
        float3 l;
        gather(i, false, with_u, a, l);
        float3 ra = va(i, N_B) - a;
        float3 rl = vl(i, N_B) - l;
        set_v(i, N_R, ra, rl);
        s += norm2(ra, rl);
    }
    device_group_barrier();
    return blast_group_sum(tid, float2(s, 0.0)).x;
}

float residual_norm(uint tid, uint n0, uint n1)
{
    float s = 0.0;
    for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
    {
        uint i = lists[p];
        s += norm2(va(i, N_R), vl(i, N_R));
    }
    return blast_group_sum(tid, float2(s, 0.0)).x;
}

groupshared float g_max[BLAST_THREADS];
groupshared float g_max_out;

float group_max(uint tid, float v)
{
    g_max[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint k = BLAST_THREADS / 2; k > 0; k >>= 1)
    {
        if (tid < k)
            g_max[tid] = max(g_max[tid], g_max[tid + k]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0)
        g_max_out = g_max[0];
    // As a uniform value on WGSL (it steers the group's barriers).
    float m;
    __target_switch
    {
    case wgsl:
        m = workgroupUniformLoad(g_max_out);
    default:
        GroupMemoryBarrierWithGroupSync();
        m = g_max_out;
    }
    GroupMemoryBarrierWithGroupSync();
    return m;
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void blast_native(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
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
    // A large island is blast_native_wide.slang's.
    if (group_uniform(n1 - n0 > params.wide_min ? 1u : 0u, tid) != 0)
        return;
    uint4 last = report[k];
    uint last_flags = group_uniform(last.y, tid);
    bool warm = params.warm != 0;

    // The settled certificate (beginNativeSettledReuse): the last solve verified its stored
    // output and the input repeats bit for bit.
    if (warm && (last_flags & REPORT_SETTLED) != 0)
    {
        float differ = 0.0;
        for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
        {
            uint i = lists[p];
            if (any(accel[i * 2] != scratch[ns(i, N_LAST)]) || any(accel[i * 2 + 1] != scratch[ns(i, N_LAST) + 1]))
                differ = 1.0;
        }
        if (blast_group_sum(tid, float2(differ, 0.0)).x == 0.0)
        {
            if (tid == 0)
                report[k] = uint4(0, REPORT_SKIPPED | REPORT_SETTLED | REPORT_CONVERGED, last.z, last.w);
            return;
        }
    }

    // Krylov carry: the last solve's recurrence, if it may continue (cleared here; set again
    // by an unconverged solve).
    uint3 kept = group_uniform4(uint4(carry_of(k), 0u), tid).xyz;
    bool carried = carry_kept(kept, n1 - n0);
    // Every thread has read it (group_uniform4 synchronises only on WGSL) before it is cleared.
    device_group_barrier();
    if (tid == 0 && params.carry != 0)
        set_carry(k, 0u, 0u, 0.0);

    // b = -D^-1 a; the blocks and their inverses; whether the island is anchored.
    float target = 0.0;
    float singular = 0.0;
    float change = 0.0;
    for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
    {
        uint i = lists[p];
        BlastNode n = nodes[i];
        float4 a_ang = accel[i * 2];
        float4 a_lin = accel[i * 2 + 1];
        float3 ba = -a_ang.xyz * (n.w_ang > 0.0 ? 1.0 / n.w_ang : 1.0);
        float3 bl = -a_lin.xyz * (n.w_lin > 0.0 ? params.recip_length / n.w_lin : params.recip_length);
        target += norm2(ba, bl);
        // The load's change since the last solve (N_B still holds its b).
        if (carried)
            change += norm2(ba - va(i, N_B), bl - vl(i, N_B));
        set_v(i, N_B, ba, bl);
        set_v(i, N_R, ba, bl);
        scratch[ns(i, N_LAST)] = a_ang;
        scratch[ns(i, N_LAST) + 1] = a_lin;
        set_v(i, N_U, float3(0.0), float3(0.0));
        // A restart writes p = g outright; only a carried direction is ever read.
        if (params.carry == 0)
            set_v(i, N_P, float3(0.0), float3(0.0));
        if (!build_block(i))
            singular = 1.0;
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
    float2 tsum = blast_group_sum(tid, float2(target, singular));
    float delta2 = tsum.x * params.tolerance * params.tolerance;
    // A load that moved by more than the solve resolves restarts: ||b - b_prev|| <= tol ||b||.
    float moved = blast_group_sum(tid, float2(change, 0.0)).x;
    carried = carried && moved <= params.tolerance * params.tolerance * tsum.x;
    // The rigid-mode frame (blast_motion.slang): anchored, or free with its rotations; a frame
    // that could not be built fails the solve.
    uint3 frame = group_uniform4(uint4(frame_word(k, F_ANCHORED), frame_word(k, F_ROTATIONS), frame_word(k, F_ERROR), 0u), tid).xyz;
    bool failed = tsum.y > 0.0 || frame.z != 0;
    bool free = frame.x == 0;
    uint rotations = frame.y;
    device_group_barrier();

    // A zero load: zero forces at once (retireHomogeneousTreeComponent).
    if (!(delta2 > 0.0))
    {
        for (uint q = b0 + tid; q < b1; q += BLAST_THREADS)
        {
            uint j = lists[q];
            x[j * 2] = float4(0.0);
            x[j * 2 + 1] = float4(0.0);
        }
        if (tid == 0)
            report[k] = uint4(0, REPORT_CONVERGED | (warm ? REPORT_SETTLED : 0u), asuint(0.0), asuint(delta2));
        return;
    }
    if (warm)
        rebuild(tid, n0, n1, false);

    float diagonal = LOW_WEIGHT + HIGH_WEIGHT - LOW_WEIGHT * HIGH_WEIGHT;
    float coupling = LOW_WEIGHT * HIGH_WEIGHT;
    float previous = carried ? asfloat(kept.z) : 0.0;
    float best = float_bits(0x7f800000u);
    uint best_iteration = 0;
    bool converged = false;
    uint iteration = 0;
    float num = 0.0;
    float force_start = 0.0;
    float travel = 0.0;
    float step = float_bits(0x7f800000u);
    if (params.force_tolerance > 0.0)
    {
        float f = 0.0;
        for (uint q = b0 + tid; q < b1; q += BLAST_THREADS)
        {
            uint j = lists[q];
            f += norm2(x[j * 2].xyz, x[j * 2 + 1].xyz);
        }
        force_start = sqrt(blast_group_sum(tid, float2(f, 0.0)).x);
    }
    while (!failed && iteration < params.max_iterations)
    {
        project(tid, k, rotations, n0, n1, N_R, free);
        num = bond_pass(tid, b0, b1, N_R);
        bool hold = num <= delta2 && !(residual_norm(tid, n0, n1) <= delta2);
        if ((iteration > 0 || warm) && num <= delta2 && !hold)
        {
            rebuild(tid, n0, n1, true);
            previous = 0.0;
            project(tid, k, rotations, n0, n1, N_R, free);
            num = bond_pass(tid, b0, b1, N_R);
            hold = num <= delta2 && !(residual_norm(tid, n0, n1) <= delta2);
        }
        if (num < best * 0.99)
        {
            best = num;
            best_iteration = iteration;
        }
        else if (iteration - best_iteration >= STAGNATION)
            break;
        if (num <= delta2 && !hold)
        {
            converged = true;
            break;
        }
        if (params.force_tolerance > 0.0 && iteration > 0 && step <= params.force_tolerance * (force_start + travel))
        {
            float f = 0.0;
            for (uint q = b0 + tid; q < b1; q += BLAST_THREADS)
            {
                uint j = lists[q];
                BlastBond b = bonds[j];
                float3 ta;
                float3 tl;
                bond_term(b, j, N_U, ta, tl);
                f += norm2(x[j * 2].xyz + ta, x[j * 2 + 1].xyz + tl);
            }
            if (step <= params.force_tolerance * sqrt(blast_group_sum(tid, float2(f, 0.0)).x))
            {
                converged = true;
                break;
            }
        }
        device_group_barrier();

        // g: r on iteration 0 (unless the polynomial runs first), else P r = (a+b-ab) M r - ab M E M r,
        // with E = L - the blocks.
        if (iteration == 0 && params.first_polynomial == 0 && !carried)
        {
            for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
            {
                uint i = lists[p];
                set_v(i, N_G, va(i, N_R), vl(i, N_R));
            }
        }
        else
        {
            for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
            {
                uint i = lists[p];
                float3 a;
                float3 l;
                inverse_apply(i, va(i, N_R), vl(i, N_R), a, l);
                set_v(i, N_LOC, a, l);
            }
            device_group_barrier();
            bond_pass(tid, b0, b1, N_LOC);
            device_group_barrier();
            node_pass(tid, n0, n1, N_OFF);
            for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
            {
                uint i = lists[p];
                float3 sa;
                float3 sl;
                block_apply(i, va(i, N_LOC), vl(i, N_LOC), sa, sl);
                float3 ma;
                float3 ml;
                inverse_apply(i, va(i, N_OFF) - sa, vl(i, N_OFF) - sl, ma, ml);
                set_v(i, N_G, diagonal * va(i, N_LOC) - coupling * ma, diagonal * vl(i, N_LOC) - coupling * ml);
            }
        }
        device_group_barrier();
        project(tid, k, rotations, n0, n1, N_G, free);
        float mloc = 0.0;
        for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
        {
            uint i = lists[p];
            float3 a = abs(va(i, N_G));
            float3 l = abs(vl(i, N_G));
            mloc = max(mloc, max(max(max(a.x, a.y), max(a.z, l.x)), max(l.y, l.z)));
        }
        float mag = group_max(tid, mloc);
        if (!(mag > 0.0) || !isfinite(mag))
        {
            failed = true;
            break;
        }
        float inv = 1.0 / mag;
        float gl = 0.0;
        for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
        {
            uint i = lists[p];
            float3 ga = va(i, N_G) * inv;
            float3 gn = vl(i, N_G) * inv;
            set_v(i, N_G, ga, gn);
            gl += dot(va(i, N_R), ga) + dot(vl(i, N_R), gn);
        }
        float gamma = blast_group_sum(tid, float2(gl, 0.0)).x;
        if (!(gamma > 0.0) || !isfinite(gamma))
        {
            failed = true;
            break;
        }
        float beta = (iteration > 1 || (params.first_polynomial != 0 && iteration > 0) || carried) && previous > 0.0 ? gamma / previous : 0.0;
        // A carried first step whose gamma grew 16-fold (a new load) restarts.
        if (carried && iteration == 0 && !(beta <= 16.0))
            beta = 0.0;
        for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
        {
            uint i = lists[p];
            if (beta == 0.0)
                set_v(i, N_P, va(i, N_G), vl(i, N_G));
            else
                set_v(i, N_P, va(i, N_G) + va(i, N_P) * beta, vl(i, N_G) + vl(i, N_P) * beta);
        }
        device_group_barrier();
        float den = bond_pass(tid, b0, b1, N_P);
        device_group_barrier();
        node_pass(tid, n0, n1, N_Q);
        if (params.force_tolerance > 0.0)
        {
            step = den > 0.0 && gamma > 0.0 ? gamma / sqrt(den) : float_bits(0x7f800000u);
            if (isfinite(step))
                travel += step;
            // Only a preconditioned step is judged: not iteration 0's steepest-descent step.
            if (!(iteration > 0 || params.first_polynomial != 0))
                step = float_bits(0x7f800000u);
        }
        previous = gamma;
        iteration++;
        if (!(den > 0.0) || !isfinite(den))
            break;
        float alpha = gamma / den;
        for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
        {
            uint i = lists[p];
            set_v(i, N_U, va(i, N_U) + va(i, N_P) * alpha, vl(i, N_U) + vl(i, N_P) * alpha);
            set_v(i, N_R, va(i, N_R) - va(i, N_Q) * alpha, vl(i, N_R) - vl(i, N_Q) * alpha);
        }
        device_group_barrier();
    }
    device_group_barrier();

    // x = x0 + B^T u over the island's live bonds.
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
        bond_term(b, j, N_U, ta, tl);
        x[j * 2] = float4(x[j * 2].xyz + ta, 0.0);
        x[j * 2 + 1] = float4(x[j * 2 + 1].xyz + tl, 0.0);
    }
    bool ok = converged && !failed;
    // The certificate (commitNativeSettledReuse): a warm solve that converged with no update.
    uint flags = (ok ? REPORT_CONVERGED : 0u) | (ok && warm && iteration == 0 ? REPORT_SETTLED : 0u);
    if (tid == 0)
    {
        report[k] = uint4(iteration, flags, asuint(num), asuint(delta2));
        // Leave an unconverged solve's recurrence for the next solve of this island.
        if (params.carry != 0 && !failed && !converged)
            set_carry(k, n1 - n0, (carried ? kept.y : 0u) + iteration, previous);
    }
}
