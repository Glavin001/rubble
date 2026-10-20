#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
float silu_0(float x_0)
{
    return x_0 / (1.0f + exp(- x_0));
}

struct Params_0
{
    uint width_0;
    uint nodes_0;
    uint grid_x_0;
    uint weights_0;
    uint edges_0;
    uint mode_0;
    uint in_width_0;
    uint max_degree_0;
};

struct KernelContext_0
{
    Params_0 constant* params_0;
    uint device* offsets_0;
    uint device* src_0;
    float device* h_in_0;
    float device* h_out_0;
    float device* wts_0;
    float device* e_in_0;
    float device* e_out_0;
    array<array<float, int(384)>, int(4)> threadgroup* xs_0;
    array<float, int(128)> threadgroup* hid_0;
    array<float, int(128)> threadgroup* red_0;
    array<array<float, int(64)>, int(8)> threadgroup* tx_0;
    array<array<float, int(128)>, int(8)> threadgroup* tw_0;
    array<float, int(2048)> threadgroup* th_0;
    array<array<float, int(64)>, int(2)> threadgroup* tstat_0;
};

float slot_sum_0(float v_0, uint tid_0, uint lane_0, uint w_0, KernelContext_0 thread* kernelContext_0)
{
    (*kernelContext_0->red_0)[tid_0] = v_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s_0 = w_0 / 2U;
    for(;;)
    {
        if(s_0 > 0U)
        {
        }
        else
        {
            break;
        }
        if(lane_0 < s_0)
        {
            (*kernelContext_0->red_0)[tid_0] = (*kernelContext_0->red_0)[tid_0] + (*kernelContext_0->red_0)[tid_0 + s_0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_0->red_0)[tid_0 - lane_0];
}

float mlp_0(uint base_0, uint k_in_0, uint tid_1, uint slot_0, uint lane_1, uint w_1, bool live_0, KernelContext_0 thread* kernelContext_1)
{
    uint k_0;
    float a_0;
    if(live_0)
    {
        float _S1 = kernelContext_1->wts_0[base_0 + k_in_0 * w_1 + lane_1];
        k_0 = 0U;
        a_0 = _S1;
        for(;;)
        {
            if(k_0 < k_in_0)
            {
            }
            else
            {
                break;
            }
            float a_1 = a_0 + kernelContext_1->wts_0[base_0 + k_0 * w_1 + lane_1] * (*kernelContext_1->xs_0)[slot_0][k_0];
            k_0 = k_0 + 1U;
            a_0 = a_1;
        }
        a_0 = silu_0(a_0);
    }
    else
    {
        a_0 = 0.0f;
    }
    (*kernelContext_1->hid_0)[tid_1] = a_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint b2_0 = base_0 + k_in_0 * w_1 + w_1;
    float acc_0;
    if(live_0)
    {
        float _S2 = kernelContext_1->wts_0[b2_0 + w_1 * w_1 + lane_1];
        k_0 = 0U;
        acc_0 = _S2;
        for(;;)
        {
            if(k_0 < w_1)
            {
            }
            else
            {
                break;
            }
            float acc_1 = acc_0 + kernelContext_1->wts_0[b2_0 + k_0 * w_1 + lane_1] * (*kernelContext_1->hid_0)[tid_1 - lane_1 + k_0];
            k_0 = k_0 + 1U;
            acc_0 = acc_1;
        }
    }
    else
    {
        acc_0 = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float _S3 = slot_sum_0(acc_0, tid_1, lane_1, w_1, kernelContext_1);
    float _S4 = float(w_1);
    float d_0 = acc_0 - _S3 / _S4;
    float _S5 = slot_sum_0(d_0 * d_0, tid_1, lane_1, w_1, kernelContext_1);
    uint g_0 = b2_0 + w_1 * w_1 + w_1;
    return kernelContext_1->wts_0[g_0 + lane_1] * d_0 * rsqrt(_S5 / _S4 + 0.00000999999974738f) + kernelContext_1->wts_0[g_0 + w_1 + lane_1];
}

uint mlp_size_0(uint k_in_1, uint w_2)
{
    return k_in_1 * w_2 + w_2 + w_2 * w_2 + w_2 + 2U * w_2;
}

[[kernel]] void gnn_round(uint3 gid_0 [[threadgroup_position_in_grid]], uint3 lid_0 [[thread_position_in_threadgroup]], Params_0 constant* params_1 [[buffer(0)]], uint device* offsets_1 [[buffer(6)]], uint device* src_1 [[buffer(7)]], float device* h_in_1 [[buffer(2)]], float device* h_out_1 [[buffer(3)]], float device* wts_1 [[buffer(1)]], float device* e_in_1 [[buffer(4)]], float device* e_out_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_2;
    (&kernelContext_2)->params_0 = params_1;
    (&kernelContext_2)->offsets_0 = offsets_1;
    (&kernelContext_2)->src_0 = src_1;
    (&kernelContext_2)->h_in_0 = h_in_1;
    (&kernelContext_2)->h_out_0 = h_out_1;
    (&kernelContext_2)->wts_0 = wts_1;
    (&kernelContext_2)->e_in_0 = e_in_1;
    (&kernelContext_2)->e_out_0 = e_out_1;
    threadgroup array<array<float, int(384)>, int(4)> xs_1;
    (&kernelContext_2)->xs_0 = &xs_1;
    threadgroup array<float, int(128)> hid_1;
    (&kernelContext_2)->hid_0 = &hid_1;
    threadgroup array<float, int(128)> red_1;
    (&kernelContext_2)->red_0 = &red_1;
    threadgroup array<array<float, int(64)>, int(8)> tx_1;
    (&kernelContext_2)->tx_0 = &tx_1;
    threadgroup array<array<float, int(128)>, int(8)> tw_1;
    (&kernelContext_2)->tw_0 = &tw_1;
    threadgroup array<float, int(2048)> th_1;
    (&kernelContext_2)->th_0 = &th_1;
    threadgroup array<array<float, int(64)>, int(2)> tstat_1;
    (&kernelContext_2)->tstat_0 = &tstat_1;
    uint w_3 = params_1->width_0;
    uint tid_2 = lid_0.x;
    uint slot_1 = tid_2 / params_1->width_0;
    uint lane_2 = tid_2 % w_3;
    uint group_0 = gid_0.y * (&kernelContext_2)->params_0->grid_x_0 + gid_0.x;
    uint _S6 = 128U / w_3;
    uint node_0 = group_0 * _S6 + slot_1;
    bool live_1 = node_0 < ((&kernelContext_2)->params_0->nodes_0);
    bool on_0;
    if(((&kernelContext_2)->params_0->mode_0) == 3U)
    {
        on_0 = true;
    }
    else
    {
        on_0 = ((&kernelContext_2)->params_0->mode_0) == 4U;
    }
    uint i_0;
    float s_1;
    if(on_0)
    {
        if(live_1)
        {
            on_0 = ((&kernelContext_2)->params_0->mode_0) == 3U;
        }
        else
        {
            on_0 = false;
        }
        if(on_0)
        {
            uint a_2 = (&kernelContext_2)->offsets_0[node_0];
            uint b_0 = (&kernelContext_2)->offsets_0[node_0 + 1U];
            i_0 = a_2;
            s_1 = 0.0f;
            for(;;)
            {
                if(i_0 < b_0)
                {
                }
                else
                {
                    break;
                }
                float s_2 = s_1 + (&kernelContext_2)->h_in_0[(&kernelContext_2)->src_0[i_0] * w_3 + lane_2];
                i_0 = i_0 + 1U;
                s_1 = s_2;
            }
            *((&kernelContext_2)->h_out_0+(node_0 * w_3 + lane_2)) = s_1 / float(max(b_0 - a_2, 1U));
        }
        else
        {
            if(live_1)
            {
                uint _S7 = node_0 * w_3 + lane_2;
                *((&kernelContext_2)->h_out_0+_S7) = *((&kernelContext_2)->h_out_0+_S7) + (&kernelContext_2)->h_in_0[(&kernelContext_2)->src_0[node_0] * w_3 + lane_2];
            }
        }
        return;
    }
    if(((&kernelContext_2)->params_0->mode_0) == 1U)
    {
        on_0 = true;
    }
    else
    {
        on_0 = ((&kernelContext_2)->params_0->mode_0) == 2U;
    }
    if(on_0)
    {
        if(((&kernelContext_2)->params_0->mode_0) == 1U)
        {
            i_0 = (&kernelContext_2)->params_0->in_width_0;
        }
        else
        {
            i_0 = w_3;
        }
        if(live_1)
        {
            on_0 = lane_2 < i_0;
        }
        else
        {
            on_0 = false;
        }
        if(on_0)
        {
            (*(&kernelContext_2)->xs_0)[slot_1][lane_2] = (&kernelContext_2)->h_in_0[node_0 * i_0 + lane_2];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _S8 = mlp_0((&kernelContext_2)->params_0->weights_0, i_0, tid_2, slot_1, lane_2, w_3, live_1, &kernelContext_2);
        if(live_1)
        {
            if(((&kernelContext_2)->params_0->mode_0) == 1U)
            {
                on_0 = true;
            }
            else
            {
                on_0 = lane_2 < ((&kernelContext_2)->params_0->in_width_0);
            }
        }
        else
        {
            on_0 = false;
        }
        if(on_0)
        {
            if(((&kernelContext_2)->params_0->mode_0) == 1U)
            {
                i_0 = w_3;
            }
            else
            {
                i_0 = (&kernelContext_2)->params_0->in_width_0;
            }
            *((&kernelContext_2)->h_out_0+(node_0 * i_0 + lane_2)) = _S8;
        }
        return;
    }
    if(((&kernelContext_2)->params_0->mode_0) == 5U)
    {
        bool on_1 = node_0 < ((&kernelContext_2)->params_0->edges_0);
        if(on_1)
        {
            (*(&kernelContext_2)->xs_0)[slot_1][lane_2] = (&kernelContext_2)->e_in_0[node_0 * w_3 + lane_2];
            (*(&kernelContext_2)->xs_0)[slot_1][w_3 + lane_2] = (&kernelContext_2)->h_in_0[(&kernelContext_2)->src_0[node_0] * w_3 + lane_2];
            (*(&kernelContext_2)->xs_0)[slot_1][2U * w_3 + lane_2] = (&kernelContext_2)->h_in_0[(&kernelContext_2)->offsets_0[node_0] * w_3 + lane_2];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _S9 = mlp_0((&kernelContext_2)->params_0->weights_0, 3U * w_3, tid_2, slot_1, lane_2, w_3, on_1, &kernelContext_2);
        if(on_1)
        {
            *((&kernelContext_2)->e_out_0+(node_0 * w_3 + lane_2)) = (*(&kernelContext_2)->xs_0)[slot_1][lane_2] + _S9;
        }
        return;
    }
    float agg_0;
    if(((&kernelContext_2)->params_0->mode_0) == 6U)
    {
        if(live_1)
        {
            float hd_0 = (&kernelContext_2)->h_in_0[node_0 * w_3 + lane_2];
            i_0 = (&kernelContext_2)->offsets_0[node_0];
            agg_0 = 0.0f;
            for(;;)
            {
                if(i_0 < ((&kernelContext_2)->offsets_0)[node_0 + 1U])
                {
                }
                else
                {
                    break;
                }
                float agg_1 = agg_0 + (&kernelContext_2)->e_in_0[i_0 * w_3 + lane_2];
                i_0 = i_0 + 1U;
                agg_0 = agg_1;
            }
            (*(&kernelContext_2)->xs_0)[slot_1][lane_2] = hd_0;
            (*(&kernelContext_2)->xs_0)[slot_1][w_3 + lane_2] = agg_0;
            s_1 = hd_0;
        }
        else
        {
            s_1 = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _S10 = mlp_0((&kernelContext_2)->params_0->weights_0 + mlp_size_0(3U * w_3, w_3), 2U * w_3, tid_2, slot_1, lane_2, w_3, live_1, &kernelContext_2);
        if(live_1)
        {
            *((&kernelContext_2)->h_out_0+(node_0 * w_3 + lane_2)) = s_1 + _S10;
        }
        return;
    }
    uint edge_w_0 = (&kernelContext_2)->params_0->weights_0;
    uint _S11 = 3U * w_3;
    uint node_w_0 = (&kernelContext_2)->params_0->weights_0 + mlp_size_0(_S11, w_3);
    if(live_1)
    {
        i_0 = (&kernelContext_2)->offsets_0[node_0];
    }
    else
    {
        i_0 = 0U;
    }
    uint _S12;
    if(live_1)
    {
        _S12 = (&kernelContext_2)->offsets_0[node_0 + 1U];
    }
    else
    {
        _S12 = 0U;
    }
    uint _S13 = (&kernelContext_2)->params_0->max_degree_0;
    if(live_1)
    {
        s_1 = (&kernelContext_2)->h_in_0[node_0 * w_3 + lane_2];
    }
    else
    {
        s_1 = 0.0f;
    }
    uint j_0 = 0U;
    agg_0 = 0.0f;
    for(;;)
    {
        if(j_0 < _S13)
        {
        }
        else
        {
            break;
        }
        uint e_0 = i_0 + j_0;
        if(live_1)
        {
            on_0 = e_0 < _S12;
        }
        else
        {
            on_0 = false;
        }
        if(on_0)
        {
            uint s_3 = (&kernelContext_2)->src_0[e_0];
            (*(&kernelContext_2)->xs_0)[slot_1][lane_2] = (&kernelContext_2)->e_in_0[e_0 * w_3 + lane_2];
            (*(&kernelContext_2)->xs_0)[slot_1][w_3 + lane_2] = (&kernelContext_2)->h_in_0[s_3 * w_3 + lane_2];
            (*(&kernelContext_2)->xs_0)[slot_1][2U * w_3 + lane_2] = s_1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _S14 = mlp_0(edge_w_0, _S11, tid_2, slot_1, lane_2, w_3, on_0, &kernelContext_2);
        if(on_0)
        {
            float en_0 = (*(&kernelContext_2)->xs_0)[slot_1][lane_2] + _S14;
            *((&kernelContext_2)->e_out_0+(e_0 * w_3 + lane_2)) = en_0;
            agg_0 = agg_0 + en_0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        j_0 = j_0 + 1U;
    }
    if(live_1)
    {
        (*(&kernelContext_2)->xs_0)[slot_1][lane_2] = s_1;
        (*(&kernelContext_2)->xs_0)[slot_1][w_3 + lane_2] = agg_0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float _S15 = mlp_0(node_w_0, 2U * w_3, tid_2, slot_1, lane_2, w_3, live_1, &kernelContext_2);
    if(live_1)
    {
        *((&kernelContext_2)->h_out_0+(node_0 * w_3 + lane_2)) = s_1 + _S15;
    }
    return;
}

float gather_0(uint row_0, uint k_1, uint w_4, uint mode_1, KernelContext_0 thread* kernelContext_3)
{
    uint seg_0 = k_1 / w_4;
    uint c_0 = k_1 - seg_0 * w_4;
    if(seg_0 == 0U)
    {
        return kernelContext_3->h_in_0[row_0 * w_4 + c_0];
    }
    if(mode_1 == 8U)
    {
        return kernelContext_3->e_in_0[row_0 * w_4 + c_0];
    }
    uint n_0;
    if(seg_0 == 1U)
    {
        n_0 = kernelContext_3->src_0[row_0];
    }
    else
    {
        n_0 = kernelContext_3->offsets_0[row_0];
    }
    return kernelContext_3->e_in_0[n_0 * w_4 + c_0];
}

[[kernel]] void gnn_mlp_tiled(uint3 gid_1 [[threadgroup_position_in_grid]], uint3 lid_1 [[thread_position_in_threadgroup]], Params_0 constant* params_2 [[buffer(0)]], uint device* offsets_2 [[buffer(6)]], uint device* src_2 [[buffer(7)]], float device* h_in_2 [[buffer(2)]], float device* h_out_2 [[buffer(3)]], float device* wts_2 [[buffer(1)]], float device* e_in_2 [[buffer(4)]], float device* e_out_2 [[buffer(5)]])
{
    uint q_0;
    float m_0;
    thread KernelContext_0 kernelContext_4;
    (&kernelContext_4)->params_0 = params_2;
    (&kernelContext_4)->offsets_0 = offsets_2;
    (&kernelContext_4)->src_0 = src_2;
    (&kernelContext_4)->h_in_0 = h_in_2;
    (&kernelContext_4)->h_out_0 = h_out_2;
    (&kernelContext_4)->wts_0 = wts_2;
    (&kernelContext_4)->e_in_0 = e_in_2;
    (&kernelContext_4)->e_out_0 = e_out_2;
    threadgroup array<array<float, int(384)>, int(4)> xs_2;
    (&kernelContext_4)->xs_0 = &xs_2;
    threadgroup array<float, int(128)> hid_2;
    (&kernelContext_4)->hid_0 = &hid_2;
    threadgroup array<float, int(128)> red_2;
    (&kernelContext_4)->red_0 = &red_2;
    threadgroup array<array<float, int(64)>, int(8)> tx_2;
    (&kernelContext_4)->tx_0 = &tx_2;
    threadgroup array<array<float, int(128)>, int(8)> tw_2;
    (&kernelContext_4)->tw_0 = &tw_2;
    threadgroup array<float, int(2048)> th_2;
    (&kernelContext_4)->th_0 = &th_2;
    threadgroup array<array<float, int(64)>, int(2)> tstat_2;
    (&kernelContext_4)->tstat_0 = &tstat_2;
    uint w_5 = params_2->width_0;
    uint tr_0 = 2048U / params_2->width_0;
    uint t_0 = lid_1.x;
    uint cg_0 = w_5 / 4U;
    uint _S16 = t_0 / cg_0;
    uint _S17 = _S16 * 4U;
    uint _S18 = t_0 % cg_0;
    uint cb_0 = _S18 * 4U;
    uint _S19 = (gid_1.y * (&kernelContext_4)->params_0->grid_x_0 + gid_1.x) * tr_0;
    uint _S20 = (&kernelContext_4)->params_0->nodes_0;
    int _S21;
    if(((&kernelContext_4)->params_0->mode_0) == 7U)
    {
        _S21 = int(3);
    }
    else
    {
        _S21 = int(2);
    }
    uint k_in_2 = uint(_S21) * w_5;
    uint _S22 = (&kernelContext_4)->params_0->weights_0;
    uint k0_0;
    if(((&kernelContext_4)->params_0->mode_0) == 8U)
    {
        k0_0 = mlp_size_0(3U * w_5, w_5);
    }
    else
    {
        k0_0 = 0U;
    }
    uint base_1 = _S22 + k0_0;
    uint _S23 = base_1 + k_in_2 * w_5;
    uint _S24 = _S23 + cb_0;
    float4 bias_0 = float4((&kernelContext_4)->wts_0[_S24], (&kernelContext_4)->wts_0[_S24 + 1U], (&kernelContext_4)->wts_0[_S24 + 2U], (&kernelContext_4)->wts_0[_S24 + 3U]);
    k0_0 = 0U;
    float4 r0_0 = bias_0;
    float4 r1_0 = bias_0;
    float4 r2_0 = bias_0;
    float4 r3_0 = bias_0;
    for(;;)
    {
        if(k0_0 < k_in_2)
        {
        }
        else
        {
            break;
        }
        q_0 = t_0;
        for(;;)
        {
            if(q_0 < (8U * tr_0))
            {
            }
            else
            {
                break;
            }
            uint kk_0 = q_0 / tr_0;
            uint r_0 = q_0 - kk_0 * tr_0;
            uint row_1 = _S19 + r_0;
            if(row_1 < _S20)
            {
                float _S25 = gather_0(row_1, k0_0 + kk_0, w_5, (&kernelContext_4)->params_0->mode_0, &kernelContext_4);
                m_0 = _S25;
            }
            else
            {
                m_0 = 0.0f;
            }
            (*(&kernelContext_4)->tx_0)[kk_0][r_0] = m_0;
            q_0 = q_0 + 128U;
        }
        uint q_1 = t_0;
        for(;;)
        {
            if(q_1 < (8U * w_5))
            {
            }
            else
            {
                break;
            }
            uint kk_1 = q_1 / w_5;
            uint c_1 = q_1 - kk_1 * w_5;
            (*(&kernelContext_4)->tw_0)[kk_1][c_1] = (&kernelContext_4)->wts_0[base_1 + (k0_0 + kk_1) * w_5 + c_1];
            q_1 = q_1 + 128U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _S26 = cb_0 + 1U;
        uint _S27 = cb_0 + 2U;
        uint _S28 = cb_0 + 3U;
        float4 b_1 = float4((*(&kernelContext_4)->tw_0)[0U][cb_0], (*(&kernelContext_4)->tw_0)[0U][_S26], (*(&kernelContext_4)->tw_0)[0U][_S27], (*(&kernelContext_4)->tw_0)[0U][_S28]);
        uint _S29 = _S17 + 1U;
        uint _S30 = _S17 + 2U;
        uint _S31 = _S17 + 3U;
        float4 b_2 = float4((*(&kernelContext_4)->tw_0)[1U][cb_0], (*(&kernelContext_4)->tw_0)[1U][_S26], (*(&kernelContext_4)->tw_0)[1U][_S27], (*(&kernelContext_4)->tw_0)[1U][_S28]);
        float4 b_3 = float4((*(&kernelContext_4)->tw_0)[2U][cb_0], (*(&kernelContext_4)->tw_0)[2U][_S26], (*(&kernelContext_4)->tw_0)[2U][_S27], (*(&kernelContext_4)->tw_0)[2U][_S28]);
        float4 b_4 = float4((*(&kernelContext_4)->tw_0)[3U][cb_0], (*(&kernelContext_4)->tw_0)[3U][_S26], (*(&kernelContext_4)->tw_0)[3U][_S27], (*(&kernelContext_4)->tw_0)[3U][_S28]);
        float4 b_5 = float4((*(&kernelContext_4)->tw_0)[4U][cb_0], (*(&kernelContext_4)->tw_0)[4U][_S26], (*(&kernelContext_4)->tw_0)[4U][_S27], (*(&kernelContext_4)->tw_0)[4U][_S28]);
        float4 b_6 = float4((*(&kernelContext_4)->tw_0)[5U][cb_0], (*(&kernelContext_4)->tw_0)[5U][_S26], (*(&kernelContext_4)->tw_0)[5U][_S27], (*(&kernelContext_4)->tw_0)[5U][_S28]);
        float4 b_7 = float4((*(&kernelContext_4)->tw_0)[6U][cb_0], (*(&kernelContext_4)->tw_0)[6U][_S26], (*(&kernelContext_4)->tw_0)[6U][_S27], (*(&kernelContext_4)->tw_0)[6U][_S28]);
        float4 b_8 = float4((*(&kernelContext_4)->tw_0)[7U][cb_0], (*(&kernelContext_4)->tw_0)[7U][_S26], (*(&kernelContext_4)->tw_0)[7U][_S27], (*(&kernelContext_4)->tw_0)[7U][_S28]);
        float4 r0_1 = r0_0 + float4((*(&kernelContext_4)->tx_0)[0U][_S17])  * b_1 + float4((*(&kernelContext_4)->tx_0)[1U][_S17])  * b_2 + float4((*(&kernelContext_4)->tx_0)[2U][_S17])  * b_3 + float4((*(&kernelContext_4)->tx_0)[3U][_S17])  * b_4 + float4((*(&kernelContext_4)->tx_0)[4U][_S17])  * b_5 + float4((*(&kernelContext_4)->tx_0)[5U][_S17])  * b_6 + float4((*(&kernelContext_4)->tx_0)[6U][_S17])  * b_7 + float4((*(&kernelContext_4)->tx_0)[7U][_S17])  * b_8;
        float4 r1_1 = r1_0 + float4((*(&kernelContext_4)->tx_0)[0U][_S29])  * b_1 + float4((*(&kernelContext_4)->tx_0)[1U][_S29])  * b_2 + float4((*(&kernelContext_4)->tx_0)[2U][_S29])  * b_3 + float4((*(&kernelContext_4)->tx_0)[3U][_S29])  * b_4 + float4((*(&kernelContext_4)->tx_0)[4U][_S29])  * b_5 + float4((*(&kernelContext_4)->tx_0)[5U][_S29])  * b_6 + float4((*(&kernelContext_4)->tx_0)[6U][_S29])  * b_7 + float4((*(&kernelContext_4)->tx_0)[7U][_S29])  * b_8;
        float4 r2_1 = r2_0 + float4((*(&kernelContext_4)->tx_0)[0U][_S30])  * b_1 + float4((*(&kernelContext_4)->tx_0)[1U][_S30])  * b_2 + float4((*(&kernelContext_4)->tx_0)[2U][_S30])  * b_3 + float4((*(&kernelContext_4)->tx_0)[3U][_S30])  * b_4 + float4((*(&kernelContext_4)->tx_0)[4U][_S30])  * b_5 + float4((*(&kernelContext_4)->tx_0)[5U][_S30])  * b_6 + float4((*(&kernelContext_4)->tx_0)[6U][_S30])  * b_7 + float4((*(&kernelContext_4)->tx_0)[7U][_S30])  * b_8;
        float4 r3_1 = r3_0 + float4((*(&kernelContext_4)->tx_0)[0U][_S31])  * b_1 + float4((*(&kernelContext_4)->tx_0)[1U][_S31])  * b_2 + float4((*(&kernelContext_4)->tx_0)[2U][_S31])  * b_3 + float4((*(&kernelContext_4)->tx_0)[3U][_S31])  * b_4 + float4((*(&kernelContext_4)->tx_0)[4U][_S31])  * b_5 + float4((*(&kernelContext_4)->tx_0)[5U][_S31])  * b_6 + float4((*(&kernelContext_4)->tx_0)[6U][_S31])  * b_7 + float4((*(&kernelContext_4)->tx_0)[7U][_S31])  * b_8;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k0_0 = k0_0 + 8U;
        r0_0 = r0_1;
        r1_0 = r1_1;
        r2_0 = r2_1;
        r3_0 = r3_1;
    }
    float _S32 = silu_0(r0_0.x);
    float _S33 = silu_0(r0_0.y);
    float _S34 = silu_0(r0_0.z);
    float _S35 = silu_0(r0_0.w);
    float _S36 = silu_0(r1_0.x);
    float _S37 = silu_0(r1_0.y);
    float _S38 = silu_0(r1_0.z);
    float _S39 = silu_0(r1_0.w);
    float _S40 = silu_0(r2_0.x);
    float _S41 = silu_0(r2_0.y);
    float _S42 = silu_0(r2_0.z);
    float _S43 = silu_0(r2_0.w);
    float _S44 = silu_0(r3_0.x);
    float _S45 = silu_0(r3_0.y);
    float _S46 = silu_0(r3_0.z);
    float _S47 = silu_0(r3_0.w);
    uint _S48 = _S17 * w_5;
    uint _S49 = _S48 + cb_0;
    (*(&kernelContext_4)->th_0)[_S49] = _S32;
    uint _S50 = _S17 + 1U;
    uint _S51 = _S50 * w_5;
    uint _S52 = _S51 + cb_0;
    (*(&kernelContext_4)->th_0)[_S52] = _S36;
    uint _S53 = _S17 + 2U;
    uint _S54 = _S53 * w_5;
    uint _S55 = _S54 + cb_0;
    (*(&kernelContext_4)->th_0)[_S55] = _S40;
    uint _S56 = _S17 + 3U;
    uint _S57 = _S56 * w_5;
    uint _S58 = _S57 + cb_0;
    (*(&kernelContext_4)->th_0)[_S58] = _S44;
    (*(&kernelContext_4)->th_0)[_S49 + 1U] = _S33;
    (*(&kernelContext_4)->th_0)[_S52 + 1U] = _S37;
    (*(&kernelContext_4)->th_0)[_S55 + 1U] = _S41;
    (*(&kernelContext_4)->th_0)[_S58 + 1U] = _S45;
    (*(&kernelContext_4)->th_0)[_S49 + 2U] = _S34;
    (*(&kernelContext_4)->th_0)[_S52 + 2U] = _S38;
    (*(&kernelContext_4)->th_0)[_S55 + 2U] = _S42;
    (*(&kernelContext_4)->th_0)[_S58 + 2U] = _S46;
    (*(&kernelContext_4)->th_0)[_S49 + 3U] = _S35;
    (*(&kernelContext_4)->th_0)[_S52 + 3U] = _S39;
    (*(&kernelContext_4)->th_0)[_S55 + 3U] = _S43;
    (*(&kernelContext_4)->th_0)[_S58 + 3U] = _S47;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint b2_1 = _S23 + w_5;
    uint _S59 = b2_1 + w_5 * w_5;
    uint _S60 = _S59 + cb_0;
    float4 bias_1 = float4((&kernelContext_4)->wts_0[_S60], (&kernelContext_4)->wts_0[_S60 + 1U], (&kernelContext_4)->wts_0[_S60 + 2U], (&kernelContext_4)->wts_0[_S60 + 3U]);
    k0_0 = 0U;
    r0_0 = bias_1;
    r1_0 = bias_1;
    r2_0 = bias_1;
    r3_0 = bias_1;
    for(;;)
    {
        if(k0_0 < w_5)
        {
        }
        else
        {
            break;
        }
        q_0 = t_0;
        for(;;)
        {
            if(q_0 < (8U * w_5))
            {
            }
            else
            {
                break;
            }
            uint kk_2 = q_0 / w_5;
            uint c_2 = q_0 - kk_2 * w_5;
            (*(&kernelContext_4)->tw_0)[kk_2][c_2] = (&kernelContext_4)->wts_0[b2_1 + (k0_0 + kk_2) * w_5 + c_2];
            q_0 = q_0 + 128U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _S61 = cb_0 + 1U;
        uint _S62 = cb_0 + 2U;
        uint _S63 = cb_0 + 3U;
        float4 b_9 = float4((*(&kernelContext_4)->tw_0)[0U][cb_0], (*(&kernelContext_4)->tw_0)[0U][_S61], (*(&kernelContext_4)->tw_0)[0U][_S62], (*(&kernelContext_4)->tw_0)[0U][_S63]);
        uint _S64 = _S48 + k0_0;
        uint _S65 = _S51 + k0_0;
        uint _S66 = _S54 + k0_0;
        uint _S67 = _S57 + k0_0;
        float4 b_10 = float4((*(&kernelContext_4)->tw_0)[1U][cb_0], (*(&kernelContext_4)->tw_0)[1U][_S61], (*(&kernelContext_4)->tw_0)[1U][_S62], (*(&kernelContext_4)->tw_0)[1U][_S63]);
        float4 b_11 = float4((*(&kernelContext_4)->tw_0)[2U][cb_0], (*(&kernelContext_4)->tw_0)[2U][_S61], (*(&kernelContext_4)->tw_0)[2U][_S62], (*(&kernelContext_4)->tw_0)[2U][_S63]);
        float4 b_12 = float4((*(&kernelContext_4)->tw_0)[3U][cb_0], (*(&kernelContext_4)->tw_0)[3U][_S61], (*(&kernelContext_4)->tw_0)[3U][_S62], (*(&kernelContext_4)->tw_0)[3U][_S63]);
        float4 b_13 = float4((*(&kernelContext_4)->tw_0)[4U][cb_0], (*(&kernelContext_4)->tw_0)[4U][_S61], (*(&kernelContext_4)->tw_0)[4U][_S62], (*(&kernelContext_4)->tw_0)[4U][_S63]);
        float4 b_14 = float4((*(&kernelContext_4)->tw_0)[5U][cb_0], (*(&kernelContext_4)->tw_0)[5U][_S61], (*(&kernelContext_4)->tw_0)[5U][_S62], (*(&kernelContext_4)->tw_0)[5U][_S63]);
        float4 b_15 = float4((*(&kernelContext_4)->tw_0)[6U][cb_0], (*(&kernelContext_4)->tw_0)[6U][_S61], (*(&kernelContext_4)->tw_0)[6U][_S62], (*(&kernelContext_4)->tw_0)[6U][_S63]);
        float4 b_16 = float4((*(&kernelContext_4)->tw_0)[7U][cb_0], (*(&kernelContext_4)->tw_0)[7U][_S61], (*(&kernelContext_4)->tw_0)[7U][_S62], (*(&kernelContext_4)->tw_0)[7U][_S63]);
        float4 r0_2 = r0_0 + float4((*(&kernelContext_4)->th_0)[_S64])  * b_9 + float4((*(&kernelContext_4)->th_0)[_S64 + 1U])  * b_10 + float4((*(&kernelContext_4)->th_0)[_S64 + 2U])  * b_11 + float4((*(&kernelContext_4)->th_0)[_S64 + 3U])  * b_12 + float4((*(&kernelContext_4)->th_0)[_S64 + 4U])  * b_13 + float4((*(&kernelContext_4)->th_0)[_S64 + 5U])  * b_14 + float4((*(&kernelContext_4)->th_0)[_S64 + 6U])  * b_15 + float4((*(&kernelContext_4)->th_0)[_S64 + 7U])  * b_16;
        float4 r1_2 = r1_0 + float4((*(&kernelContext_4)->th_0)[_S65])  * b_9 + float4((*(&kernelContext_4)->th_0)[_S65 + 1U])  * b_10 + float4((*(&kernelContext_4)->th_0)[_S65 + 2U])  * b_11 + float4((*(&kernelContext_4)->th_0)[_S65 + 3U])  * b_12 + float4((*(&kernelContext_4)->th_0)[_S65 + 4U])  * b_13 + float4((*(&kernelContext_4)->th_0)[_S65 + 5U])  * b_14 + float4((*(&kernelContext_4)->th_0)[_S65 + 6U])  * b_15 + float4((*(&kernelContext_4)->th_0)[_S65 + 7U])  * b_16;
        float4 r2_2 = r2_0 + float4((*(&kernelContext_4)->th_0)[_S66])  * b_9 + float4((*(&kernelContext_4)->th_0)[_S66 + 1U])  * b_10 + float4((*(&kernelContext_4)->th_0)[_S66 + 2U])  * b_11 + float4((*(&kernelContext_4)->th_0)[_S66 + 3U])  * b_12 + float4((*(&kernelContext_4)->th_0)[_S66 + 4U])  * b_13 + float4((*(&kernelContext_4)->th_0)[_S66 + 5U])  * b_14 + float4((*(&kernelContext_4)->th_0)[_S66 + 6U])  * b_15 + float4((*(&kernelContext_4)->th_0)[_S66 + 7U])  * b_16;
        float4 r3_2 = r3_0 + float4((*(&kernelContext_4)->th_0)[_S67])  * b_9 + float4((*(&kernelContext_4)->th_0)[_S67 + 1U])  * b_10 + float4((*(&kernelContext_4)->th_0)[_S67 + 2U])  * b_11 + float4((*(&kernelContext_4)->th_0)[_S67 + 3U])  * b_12 + float4((*(&kernelContext_4)->th_0)[_S67 + 4U])  * b_13 + float4((*(&kernelContext_4)->th_0)[_S67 + 5U])  * b_14 + float4((*(&kernelContext_4)->th_0)[_S67 + 6U])  * b_15 + float4((*(&kernelContext_4)->th_0)[_S67 + 7U])  * b_16;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k0_0 = k0_0 + 8U;
        r0_0 = r0_2;
        r1_0 = r1_2;
        r2_0 = r2_2;
        r3_0 = r3_2;
    }
    float4 _S68 = r0_0;
    (*(&kernelContext_4)->th_0)[_S49] = r0_0[0U];
    float4 _S69 = r1_0;
    (*(&kernelContext_4)->th_0)[_S52] = r1_0[0U];
    float4 _S70 = r2_0;
    (*(&kernelContext_4)->th_0)[_S55] = r2_0[0U];
    float4 _S71 = r3_0;
    (*(&kernelContext_4)->th_0)[_S58] = r3_0[0U];
    float4 _S72 = r0_0;
    (*(&kernelContext_4)->th_0)[_S49 + 1U] = r0_0[1U];
    float4 _S73 = r1_0;
    (*(&kernelContext_4)->th_0)[_S52 + 1U] = r1_0[1U];
    float4 _S74 = r2_0;
    (*(&kernelContext_4)->th_0)[_S55 + 1U] = r2_0[1U];
    float4 _S75 = r3_0;
    (*(&kernelContext_4)->th_0)[_S58 + 1U] = r3_0[1U];
    float4 _S76 = r0_0;
    (*(&kernelContext_4)->th_0)[_S49 + 2U] = r0_0[2U];
    float4 _S77 = r1_0;
    (*(&kernelContext_4)->th_0)[_S52 + 2U] = r1_0[2U];
    float4 _S78 = r2_0;
    (*(&kernelContext_4)->th_0)[_S55 + 2U] = r2_0[2U];
    float4 _S79 = r3_0;
    (*(&kernelContext_4)->th_0)[_S58 + 2U] = r3_0[2U];
    float4 _S80 = r0_0;
    (*(&kernelContext_4)->th_0)[_S49 + 3U] = r0_0[3U];
    float4 _S81 = r1_0;
    (*(&kernelContext_4)->th_0)[_S52 + 3U] = r1_0[3U];
    float4 _S82 = r2_0;
    (*(&kernelContext_4)->th_0)[_S55 + 3U] = r2_0[3U];
    float4 _S83 = r3_0;
    (*(&kernelContext_4)->th_0)[_S58 + 3U] = r3_0[3U];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t_0 < tr_0)
    {
        q_0 = 0U;
        m_0 = 0.0f;
        for(;;)
        {
            if(q_0 < w_5)
            {
            }
            else
            {
                break;
            }
            float m_1 = m_0 + (*(&kernelContext_4)->th_0)[t_0 * w_5 + q_0];
            q_0 = q_0 + 1U;
            m_0 = m_1;
        }
        float _S84 = float(w_5);
        float m_2 = m_0 / _S84;
        q_0 = 0U;
        float v_1 = 0.0f;
        for(;;)
        {
            if(q_0 < w_5)
            {
            }
            else
            {
                break;
            }
            float d_1 = (*(&kernelContext_4)->th_0)[t_0 * w_5 + q_0] - m_2;
            float v_2 = v_1 + d_1 * d_1;
            q_0 = q_0 + 1U;
            v_1 = v_2;
        }
        (*(&kernelContext_4)->tstat_0)[int(0)][t_0] = m_2;
        (*(&kernelContext_4)->tstat_0)[int(1)][t_0] = rsqrt(v_1 / _S84 + 0.00000999999974738f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S85 = _S59 + w_5;
    uint _S86 = _S19 + _S17;
    if(_S86 < _S20)
    {
        uint _S87 = _S85 + w_5;
        uint _S88 = _S86 * w_5;
        uint _S89 = _S88 + cb_0;
        *((&kernelContext_4)->h_out_0+_S89) = (&kernelContext_4)->h_in_0[_S89] + ((&kernelContext_4)->wts_0[_S85 + cb_0] * (_S68[0U] - (*(&kernelContext_4)->tstat_0)[int(0)][_S17]) * (*(&kernelContext_4)->tstat_0)[int(1)][_S17] + (&kernelContext_4)->wts_0[_S87 + cb_0]);
        uint c_3 = cb_0 + 1U;
        uint _S90 = _S88 + c_3;
        *((&kernelContext_4)->h_out_0+_S90) = (&kernelContext_4)->h_in_0[_S90] + ((&kernelContext_4)->wts_0[_S85 + c_3] * (_S72[1U] - (*(&kernelContext_4)->tstat_0)[int(0)][_S17]) * (*(&kernelContext_4)->tstat_0)[int(1)][_S17] + (&kernelContext_4)->wts_0[_S87 + c_3]);
        uint c_4 = cb_0 + 2U;
        uint _S91 = _S88 + c_4;
        *((&kernelContext_4)->h_out_0+_S91) = (&kernelContext_4)->h_in_0[_S91] + ((&kernelContext_4)->wts_0[_S85 + c_4] * (_S76[2U] - (*(&kernelContext_4)->tstat_0)[int(0)][_S17]) * (*(&kernelContext_4)->tstat_0)[int(1)][_S17] + (&kernelContext_4)->wts_0[_S87 + c_4]);
        uint c_5 = cb_0 + 3U;
        uint _S92 = _S88 + c_5;
        *((&kernelContext_4)->h_out_0+_S92) = (&kernelContext_4)->h_in_0[_S92] + ((&kernelContext_4)->wts_0[_S85 + c_5] * (_S80[3U] - (*(&kernelContext_4)->tstat_0)[int(0)][_S17]) * (*(&kernelContext_4)->tstat_0)[int(1)][_S17] + (&kernelContext_4)->wts_0[_S87 + c_5]);
    }
    uint row_2 = _S86 + 1U;
    if(row_2 < _S20)
    {
        uint _S93 = _S85 + w_5;
        uint _S94 = row_2 * w_5;
        uint _S95 = _S94 + cb_0;
        *((&kernelContext_4)->h_out_0+_S95) = (&kernelContext_4)->h_in_0[_S95] + ((&kernelContext_4)->wts_0[_S85 + cb_0] * (_S69[0U] - (*(&kernelContext_4)->tstat_0)[int(0)][_S50]) * (*(&kernelContext_4)->tstat_0)[int(1)][_S50] + (&kernelContext_4)->wts_0[_S93 + cb_0]);
        uint c_6 = cb_0 + 1U;
        uint _S96 = _S94 + c_6;
        *((&kernelContext_4)->h_out_0+_S96) = (&kernelContext_4)->h_in_0[_S96] + ((&kernelContext_4)->wts_0[_S85 + c_6] * (_S73[1U] - (*(&kernelContext_4)->tstat_0)[int(0)][_S50]) * (*(&kernelContext_4)->tstat_0)[int(1)][_S50] + (&kernelContext_4)->wts_0[_S93 + c_6]);
        uint c_7 = cb_0 + 2U;
        uint _S97 = _S94 + c_7;
        *((&kernelContext_4)->h_out_0+_S97) = (&kernelContext_4)->h_in_0[_S97] + ((&kernelContext_4)->wts_0[_S85 + c_7] * (_S77[2U] - (*(&kernelContext_4)->tstat_0)[int(0)][_S50]) * (*(&kernelContext_4)->tstat_0)[int(1)][_S50] + (&kernelContext_4)->wts_0[_S93 + c_7]);
        uint c_8 = cb_0 + 3U;
        uint _S98 = _S94 + c_8;
        *((&kernelContext_4)->h_out_0+_S98) = (&kernelContext_4)->h_in_0[_S98] + ((&kernelContext_4)->wts_0[_S85 + c_8] * (_S81[3U] - (*(&kernelContext_4)->tstat_0)[int(0)][_S50]) * (*(&kernelContext_4)->tstat_0)[int(1)][_S50] + (&kernelContext_4)->wts_0[_S93 + c_8]);
    }
    uint row_3 = _S86 + 2U;
    if(row_3 < _S20)
    {
        uint _S99 = _S85 + w_5;
        uint _S100 = row_3 * w_5;
        uint _S101 = _S100 + cb_0;
        *((&kernelContext_4)->h_out_0+_S101) = (&kernelContext_4)->h_in_0[_S101] + ((&kernelContext_4)->wts_0[_S85 + cb_0] * (_S70[0U] - (*(&kernelContext_4)->tstat_0)[int(0)][_S53]) * (*(&kernelContext_4)->tstat_0)[int(1)][_S53] + (&kernelContext_4)->wts_0[_S99 + cb_0]);
        uint c_9 = cb_0 + 1U;
        uint _S102 = _S100 + c_9;
        *((&kernelContext_4)->h_out_0+_S102) = (&kernelContext_4)->h_in_0[_S102] + ((&kernelContext_4)->wts_0[_S85 + c_9] * (_S74[1U] - (*(&kernelContext_4)->tstat_0)[int(0)][_S53]) * (*(&kernelContext_4)->tstat_0)[int(1)][_S53] + (&kernelContext_4)->wts_0[_S99 + c_9]);
        uint c_10 = cb_0 + 2U;
        uint _S103 = _S100 + c_10;
        *((&kernelContext_4)->h_out_0+_S103) = (&kernelContext_4)->h_in_0[_S103] + ((&kernelContext_4)->wts_0[_S85 + c_10] * (_S78[2U] - (*(&kernelContext_4)->tstat_0)[int(0)][_S53]) * (*(&kernelContext_4)->tstat_0)[int(1)][_S53] + (&kernelContext_4)->wts_0[_S99 + c_10]);
        uint c_11 = cb_0 + 3U;
        uint _S104 = _S100 + c_11;
        *((&kernelContext_4)->h_out_0+_S104) = (&kernelContext_4)->h_in_0[_S104] + ((&kernelContext_4)->wts_0[_S85 + c_11] * (_S82[3U] - (*(&kernelContext_4)->tstat_0)[int(0)][_S53]) * (*(&kernelContext_4)->tstat_0)[int(1)][_S53] + (&kernelContext_4)->wts_0[_S99 + c_11]);
    }
    uint row_4 = _S86 + 3U;
    if(row_4 < _S20)
    {
        uint _S105 = _S85 + w_5;
        uint _S106 = row_4 * w_5;
        uint _S107 = _S106 + cb_0;
        *((&kernelContext_4)->h_out_0+_S107) = (&kernelContext_4)->h_in_0[_S107] + ((&kernelContext_4)->wts_0[_S85 + cb_0] * (_S71[0U] - (*(&kernelContext_4)->tstat_0)[int(0)][_S56]) * (*(&kernelContext_4)->tstat_0)[int(1)][_S56] + (&kernelContext_4)->wts_0[_S105 + cb_0]);
        uint c_12 = cb_0 + 1U;
        uint _S108 = _S106 + c_12;
        *((&kernelContext_4)->h_out_0+_S108) = (&kernelContext_4)->h_in_0[_S108] + ((&kernelContext_4)->wts_0[_S85 + c_12] * (_S75[1U] - (*(&kernelContext_4)->tstat_0)[int(0)][_S56]) * (*(&kernelContext_4)->tstat_0)[int(1)][_S56] + (&kernelContext_4)->wts_0[_S105 + c_12]);
        uint c_13 = cb_0 + 2U;
        uint _S109 = _S106 + c_13;
        *((&kernelContext_4)->h_out_0+_S109) = (&kernelContext_4)->h_in_0[_S109] + ((&kernelContext_4)->wts_0[_S85 + c_13] * (_S79[2U] - (*(&kernelContext_4)->tstat_0)[int(0)][_S56]) * (*(&kernelContext_4)->tstat_0)[int(1)][_S56] + (&kernelContext_4)->wts_0[_S105 + c_13]);
        uint c_14 = cb_0 + 3U;
        uint _S110 = _S106 + c_14;
        *((&kernelContext_4)->h_out_0+_S110) = (&kernelContext_4)->h_in_0[_S110] + ((&kernelContext_4)->wts_0[_S85 + c_14] * (_S83[3U] - (*(&kernelContext_4)->tstat_0)[int(0)][_S56]) * (*(&kernelContext_4)->tstat_0)[int(1)][_S56] + (&kernelContext_4)->wts_0[_S105 + c_14]);
    }
    return;
}

[[kernel]] void gnn_csr_sum(uint3 gid_2 [[threadgroup_position_in_grid]], uint3 lid_2 [[thread_position_in_threadgroup]], Params_0 constant* params_3 [[buffer(0)]], uint device* offsets_3 [[buffer(6)]], uint device* src_3 [[buffer(7)]], float device* h_in_3 [[buffer(2)]], float device* h_out_3 [[buffer(3)]], float device* wts_3 [[buffer(1)]], float device* e_in_3 [[buffer(4)]], float device* e_out_3 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->params_0 = params_3;
    (&kernelContext_5)->offsets_0 = offsets_3;
    (&kernelContext_5)->src_0 = src_3;
    (&kernelContext_5)->h_in_0 = h_in_3;
    (&kernelContext_5)->h_out_0 = h_out_3;
    (&kernelContext_5)->wts_0 = wts_3;
    (&kernelContext_5)->e_in_0 = e_in_3;
    (&kernelContext_5)->e_out_0 = e_out_3;
    threadgroup array<array<float, int(384)>, int(4)> xs_3;
    (&kernelContext_5)->xs_0 = &xs_3;
    threadgroup array<float, int(128)> hid_3;
    (&kernelContext_5)->hid_0 = &hid_3;
    threadgroup array<float, int(128)> red_3;
    (&kernelContext_5)->red_0 = &red_3;
    threadgroup array<array<float, int(64)>, int(8)> tx_3;
    (&kernelContext_5)->tx_0 = &tx_3;
    threadgroup array<array<float, int(128)>, int(8)> tw_3;
    (&kernelContext_5)->tw_0 = &tw_3;
    threadgroup array<float, int(2048)> th_3;
    (&kernelContext_5)->th_0 = &th_3;
    threadgroup array<array<float, int(64)>, int(2)> tstat_3;
    (&kernelContext_5)->tstat_0 = &tstat_3;
    uint w_6 = params_3->width_0;
    uint _S111 = lid_2.x;
    uint slot_2 = _S111 / params_3->width_0;
    uint lane_3 = _S111 % w_6;
    uint _S112 = gid_2.y * (&kernelContext_5)->params_0->grid_x_0 + gid_2.x;
    uint _S113 = 128U / w_6;
    uint node_1 = _S112 * _S113 + slot_2;
    if(node_1 >= ((&kernelContext_5)->params_0->nodes_0))
    {
        return;
    }
    uint i_1 = (&kernelContext_5)->offsets_0[node_1];
    float s_4 = 0.0f;
    for(;;)
    {
        if(i_1 < ((&kernelContext_5)->offsets_0)[node_1 + 1U])
        {
        }
        else
        {
            break;
        }
        float s_5 = s_4 + (&kernelContext_5)->e_in_0[i_1 * w_6 + lane_3];
        i_1 = i_1 + 1U;
        s_4 = s_5;
    }
    *((&kernelContext_5)->h_out_0+(node_1 * w_6 + lane_3)) = s_4;
    return;
}

