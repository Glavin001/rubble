#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct Params_0
{
    array<uint4, int(16)> p_0;
};

struct KernelContext_0
{
    Params_0 constant* params_0;
    uint device* table_0;
    float device* arena_0;
    float device* wts_0;
    array<array<float, int(384)>, int(4)> threadgroup* xs_0;
    array<float, int(128)> threadgroup* hid_0;
    array<float, int(128)> threadgroup* red_0;
    array<array<float, int(64)>, int(8)> threadgroup* tx_0;
    array<array<float, int(64)>, int(8)> threadgroup* tw_0;
    array<float, int(2048)> threadgroup* th_0;
    array<array<float, int(64)>, int(2)> threadgroup* tstat_0;
};

uint P_0(uint i_0, KernelContext_0 thread* kernelContext_0)
{
    return kernelContext_0->params_0->p_0[i_0 >> 2U][i_0 & 3U];
}

float silu_0(float x_0)
{
    return x_0 / (1.0f + exp(- x_0));
}

float slot_sum_0(float v_0, uint tid_0, uint lane_0, uint lanes_0, KernelContext_0 thread* kernelContext_1)
{
    (*kernelContext_1->red_0)[tid_0] = v_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s_0 = lanes_0 / 2U;
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
            (*kernelContext_1->red_0)[tid_0] = (*kernelContext_1->red_0)[tid_0] + (*kernelContext_1->red_0)[tid_0 + s_0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_1->red_0)[tid_0 - lane_0];
}

[[kernel]] void gnn_mlp(uint3 gid_0 [[threadgroup_position_in_grid]], uint3 lid_0 [[thread_position_in_threadgroup]], Params_0 constant* params_1 [[buffer(0)]], uint device* table_1 [[buffer(2)]], float device* arena_1 [[buffer(3)]], float device* wts_1 [[buffer(1)]])
{
    uint r_0;
    float pre_0;
    float v_1;
    bool _S1;
    thread KernelContext_0 kernelContext_2;
    (&kernelContext_2)->params_0 = params_1;
    (&kernelContext_2)->table_0 = table_1;
    (&kernelContext_2)->arena_0 = arena_1;
    (&kernelContext_2)->wts_0 = wts_1;
    threadgroup array<array<float, int(384)>, int(4)> xs_1;
    (&kernelContext_2)->xs_0 = &xs_1;
    threadgroup array<float, int(128)> hid_1;
    (&kernelContext_2)->hid_0 = &hid_1;
    threadgroup array<float, int(128)> red_1;
    (&kernelContext_2)->red_0 = &red_1;
    threadgroup array<array<float, int(64)>, int(8)> tx_1;
    (&kernelContext_2)->tx_0 = &tx_1;
    threadgroup array<array<float, int(64)>, int(8)> tw_1;
    (&kernelContext_2)->tw_0 = &tw_1;
    threadgroup array<float, int(2048)> th_1;
    (&kernelContext_2)->th_0 = &th_1;
    threadgroup array<array<float, int(64)>, int(2)> tstat_1;
    (&kernelContext_2)->tstat_0 = &tstat_1;
    uint _S2 = P_0(3U, &kernelContext_2);
    uint tid_1 = lid_0.x;
    uint slot_0 = tid_1 / _S2;
    uint lane_1 = tid_1 % _S2;
    uint _S3 = gid_0.y;
    uint _S4 = P_0(2U, &kernelContext_2);
    uint _S5 = _S3 * _S4 + gid_0.x;
    uint _S6 = 128U / _S2;
    uint row_0 = _S5 * _S6 + slot_0;
    uint _S7 = P_0(1U, &kernelContext_2);
    bool live_0 = row_0 < _S7;
    uint _S8 = P_0(42U, &kernelContext_2);
    uint _S9 = P_0(43U, &kernelContext_2);
    uint _S10 = P_0(4U, &kernelContext_2);
    uint s_1 = 0U;
    float pre_1 = 0.0f;
    uint k_0 = 0U;
    for(;;)
    {
        if(s_1 < _S10)
        {
        }
        else
        {
            break;
        }
        uint o_0 = 8U + 5U * s_1;
        uint _S11 = P_0(o_0, &kernelContext_2);
        uint _S12 = P_0(o_0 + 1U, &kernelContext_2);
        uint _S13 = P_0(o_0 + 2U, &kernelContext_2);
        uint _S14 = P_0(o_0 + 3U, &kernelContext_2);
        uint _S15 = P_0(o_0 + 4U, &kernelContext_2);
        if(live_0)
        {
            if(_S14 == 1U)
            {
                _S1 = true;
            }
            else
            {
                _S1 = _S14 == 3U;
            }
            if(_S1)
            {
                r_0 = (&kernelContext_2)->table_0[_S12 + row_0];
            }
            else
            {
                if(_S14 == 2U)
                {
                    r_0 = _S12;
                }
                else
                {
                    r_0 = row_0;
                }
            }
            if(_S14 >= 3U)
            {
                if(lane_1 < _S8)
                {
                    pre_0 = pre_1 + *((&kernelContext_2)->arena_0+(_S11 + r_0 * _S13 + lane_1));
                }
                else
                {
                    pre_0 = pre_1;
                }
            }
            else
            {
                uint c_0 = lane_1;
                for(;;)
                {
                    if(c_0 < _S13)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float v_2 = *((&kernelContext_2)->arena_0+(_S11 + r_0 * _S13 + c_0));
                    if(_S15 != 4294967295U)
                    {
                        v_1 = (v_2 - (&kernelContext_2)->wts_0[_S15 + c_0]) / (&kernelContext_2)->wts_0[_S15 + _S13 + c_0];
                    }
                    else
                    {
                        v_1 = v_2;
                    }
                    (*(&kernelContext_2)->xs_0)[slot_0][k_0 + c_0] = v_1;
                    c_0 = c_0 + _S2;
                }
                pre_0 = pre_1;
            }
            pre_1 = pre_0;
        }
        if(_S14 < 3U)
        {
            k_0 = k_0 + _S13;
        }
        s_1 = s_1 + 1U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S16 = P_0(41U, &kernelContext_2);
    uint _S17 = P_0(40U, &kernelContext_2);
    if(live_0)
    {
        _S1 = lane_1 < _S8;
    }
    else
    {
        _S1 = false;
    }
    if(_S1)
    {
        uint _S18 = P_0(52U, &kernelContext_2);
        if(_S18 == 4294967295U)
        {
            pre_0 = 0.0f;
        }
        else
        {
            pre_0 = (&kernelContext_2)->wts_0[_S18 + lane_1];
        }
        r_0 = 0U;
        for(;;)
        {
            if(r_0 < _S16)
            {
            }
            else
            {
                break;
            }
            float a_0 = pre_0 + (&kernelContext_2)->wts_0[_S17 + r_0 * _S8 + lane_1] * (*(&kernelContext_2)->xs_0)[slot_0][r_0];
            r_0 = r_0 + 1U;
            pre_0 = a_0;
        }
        pre_1 = pre_1 + pre_0;
    }
    uint _S19 = P_0(44U, &kernelContext_2);
    if(_S19 == 4294967295U)
    {
        pre_0 = pre_1;
    }
    else
    {
        if(live_0)
        {
            _S1 = lane_1 < _S8;
        }
        else
        {
            _S1 = false;
        }
        if(_S1)
        {
            pre_1 = silu_0(pre_1);
        }
        else
        {
            pre_1 = 0.0f;
        }
        (*(&kernelContext_2)->hid_0)[tid_1] = pre_1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(live_0)
        {
            _S1 = lane_1 < _S9;
        }
        else
        {
            _S1 = false;
        }
        if(_S1)
        {
            uint _S20 = P_0(53U, &kernelContext_2);
            float _S21 = (&kernelContext_2)->wts_0[_S20 + lane_1];
            r_0 = 0U;
            pre_0 = _S21;
            for(;;)
            {
                if(r_0 < _S8)
                {
                }
                else
                {
                    break;
                }
                float a_1 = pre_0 + (&kernelContext_2)->wts_0[_S19 + r_0 * _S9 + lane_1] * (*(&kernelContext_2)->hid_0)[tid_1 - lane_1 + r_0];
                r_0 = r_0 + 1U;
                pre_0 = a_1;
            }
        }
        else
        {
            pre_0 = 0.0f;
        }
        uint _S22 = P_0(45U, &kernelContext_2);
        if(_S22 != 4294967295U)
        {
            float _S23 = slot_sum_0(pre_0, tid_1, lane_1, _S2, &kernelContext_2);
            float _S24 = float(_S9);
            float d_0 = pre_0 - _S23 / _S24;
            float _S25 = slot_sum_0(d_0 * d_0, tid_1, lane_1, _S2, &kernelContext_2);
            pre_0 = d_0 * (1.0f / sqrt(_S25 / _S24 + 0.00000999999974738f)) * (&kernelContext_2)->wts_0[_S22 + lane_1] + (&kernelContext_2)->wts_0[_S22 + _S9 + lane_1];
        }
    }
    if(live_0)
    {
        _S1 = lane_1 < _S9;
    }
    else
    {
        _S1 = false;
    }
    if(_S1)
    {
        uint _S26 = P_0(51U, &kernelContext_2);
        float v_3 = (as_type<float>((_S26))) * pre_0;
        uint _S27 = P_0(48U, &kernelContext_2);
        if(_S27 != 4294967295U)
        {
            uint _S28 = P_0(49U, &kernelContext_2);
            v_1 = v_3 + *((&kernelContext_2)->arena_0+(_S27 + row_0 * _S28 + lane_1));
        }
        else
        {
            v_1 = v_3;
        }
        uint _S29 = P_0(46U, &kernelContext_2);
        uint _S30 = P_0(47U, &kernelContext_2);
        uint o_1 = _S29 + row_0 * _S30 + lane_1;
        uint _S31 = P_0(50U, &kernelContext_2);
        if((_S31 & 1U) != 0U)
        {
            *((&kernelContext_2)->arena_0+o_1) = *((&kernelContext_2)->arena_0+o_1) + v_1;
        }
        else
        {
            *((&kernelContext_2)->arena_0+o_1) = v_1;
        }
    }
    return;
}

[[kernel]] void gnn_csr(uint3 gid_1 [[threadgroup_position_in_grid]], uint3 lid_1 [[thread_position_in_threadgroup]], Params_0 constant* params_2 [[buffer(0)]], uint device* table_2 [[buffer(2)]], float device* arena_2 [[buffer(3)]], float device* wts_2 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->params_0 = params_2;
    (&kernelContext_3)->table_0 = table_2;
    (&kernelContext_3)->arena_0 = arena_2;
    (&kernelContext_3)->wts_0 = wts_2;
    threadgroup array<array<float, int(384)>, int(4)> xs_2;
    (&kernelContext_3)->xs_0 = &xs_2;
    threadgroup array<float, int(128)> hid_2;
    (&kernelContext_3)->hid_0 = &hid_2;
    threadgroup array<float, int(128)> red_2;
    (&kernelContext_3)->red_0 = &red_2;
    threadgroup array<array<float, int(64)>, int(8)> tx_2;
    (&kernelContext_3)->tx_0 = &tx_2;
    threadgroup array<array<float, int(64)>, int(8)> tw_2;
    (&kernelContext_3)->tw_0 = &tw_2;
    threadgroup array<float, int(2048)> th_2;
    (&kernelContext_3)->th_0 = &th_2;
    threadgroup array<array<float, int(64)>, int(2)> tstat_2;
    (&kernelContext_3)->tstat_0 = &tstat_2;
    uint _S32 = P_0(3U, &kernelContext_3);
    uint _S33 = lid_1.x;
    uint slot_1 = _S33 / _S32;
    uint lane_2 = _S33 % _S32;
    uint _S34 = gid_1.y;
    uint _S35 = P_0(2U, &kernelContext_3);
    uint _S36 = _S34 * _S35 + gid_1.x;
    uint _S37 = 128U / _S32;
    uint row_1 = _S36 * _S37 + slot_1;
    uint _S38 = P_0(1U, &kernelContext_3);
    bool _S39;
    if(row_1 >= _S38)
    {
        _S39 = true;
    }
    else
    {
        uint _S40 = P_0(4U, &kernelContext_3);
        _S39 = lane_2 >= _S40;
    }
    if(_S39)
    {
        return;
    }
    uint _S41 = P_0(5U, &kernelContext_3);
    uint _S42 = _S41 + row_1;
    uint e0_0 = (&kernelContext_3)->table_0[_S42];
    uint _S43 = (&kernelContext_3)->table_0[_S42 + 1U];
    uint _S44 = P_0(6U, &kernelContext_3);
    uint e_0 = e0_0;
    float s_2 = 0.0f;
    for(;;)
    {
        if(e_0 < _S43)
        {
        }
        else
        {
            break;
        }
        uint _S45 = _S44 + 2U * e_0;
        float s_3 = s_2 + *((&kernelContext_3)->arena_0+((&kernelContext_3)->table_0[_S45] + lane_2)) * (as_type<float>(((&kernelContext_3)->table_0[_S45 + 1U])));
        e_0 = e_0 + 1U;
        s_2 = s_3;
    }
    uint _S46 = P_0(7U, &kernelContext_3);
    uint _S47 = P_0(8U, &kernelContext_3);
    uint o_2 = _S46 + row_1 * _S47 + lane_2;
    uint _S48 = P_0(9U, &kernelContext_3);
    if((_S48 & 1U) != 0U)
    {
        *((&kernelContext_3)->arena_0+o_2) = *((&kernelContext_3)->arena_0+o_2) + s_2;
    }
    else
    {
        *((&kernelContext_3)->arena_0+o_2) = s_2;
    }
    return;
}

[[kernel]] void gnn_fibre(uint3 gid_2 [[threadgroup_position_in_grid]], uint3 lid_2 [[thread_position_in_threadgroup]], Params_0 constant* params_3 [[buffer(0)]], uint device* table_3 [[buffer(2)]], float device* arena_3 [[buffer(3)]], float device* wts_3 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_4;
    (&kernelContext_4)->params_0 = params_3;
    (&kernelContext_4)->table_0 = table_3;
    (&kernelContext_4)->arena_0 = arena_3;
    (&kernelContext_4)->wts_0 = wts_3;
    threadgroup array<array<float, int(384)>, int(4)> xs_3;
    (&kernelContext_4)->xs_0 = &xs_3;
    threadgroup array<float, int(128)> hid_3;
    (&kernelContext_4)->hid_0 = &hid_3;
    threadgroup array<float, int(128)> red_3;
    (&kernelContext_4)->red_0 = &red_3;
    threadgroup array<array<float, int(64)>, int(8)> tx_3;
    (&kernelContext_4)->tx_0 = &tx_3;
    threadgroup array<array<float, int(64)>, int(8)> tw_3;
    (&kernelContext_4)->tw_0 = &tw_3;
    threadgroup array<float, int(2048)> th_3;
    (&kernelContext_4)->th_0 = &th_3;
    threadgroup array<array<float, int(64)>, int(2)> tstat_3;
    (&kernelContext_4)->tstat_0 = &tstat_3;
    uint _S49 = gid_2.y;
    uint _S50 = P_0(2U, &kernelContext_4);
    uint r_1 = (_S49 * _S50 + gid_2.x) * 128U + lid_2.x;
    uint _S51 = P_0(1U, &kernelContext_4);
    if(r_1 >= _S51)
    {
        return;
    }
    uint _S52 = P_0(5U, &kernelContext_4);
    uint _S53 = 4U * r_1;
    uint t_0 = _S52 + _S53;
    uint i_1 = (&kernelContext_4)->table_0[t_0];
    uint j_0 = (&kernelContext_4)->table_0[t_0 + 1U];
    uint c_1 = (&kernelContext_4)->table_0[t_0 + 2U];
    uint _S54 = P_0(4U, &kernelContext_4);
    uint _S55 = _S54 + 3U * i_1;
    uint _S56 = _S54 + 3U * j_0;
    float3 d_1 = float3(*((&kernelContext_4)->arena_0+_S55) - *((&kernelContext_4)->arena_0+_S56), *((&kernelContext_4)->arena_0+(_S55 + 1U)) - *((&kernelContext_4)->arena_0+(_S56 + 1U)), *((&kernelContext_4)->arena_0+(_S55 + 2U)) - *((&kernelContext_4)->arena_0+(_S56 + 2U)));
    uint _S57 = P_0(6U, &kernelContext_4);
    uint qb_0 = _S57 + 4U * c_1;
    float3 u_0 = - float3(*((&kernelContext_4)->arena_0+(qb_0 + 1U)), *((&kernelContext_4)->arena_0+(qb_0 + 2U)), *((&kernelContext_4)->arena_0+(qb_0 + 3U)));
    float3 tt_0 = float3(2.0f)  * cross(u_0, d_1);
    float3 rr_0 = (d_1 + float3(*((&kernelContext_4)->arena_0+qb_0))  * tt_0 + cross(u_0, tt_0)) / float3((as_type<float>(((&kernelContext_4)->table_0[t_0 + 3U])))) ;
    uint _S58 = P_0(7U, &kernelContext_4);
    uint o_3 = _S58 + _S53;
    float _S59 = rr_0.x;
    *((&kernelContext_4)->arena_0+o_3) = _S59;
    float _S60 = rr_0.y;
    *((&kernelContext_4)->arena_0+(o_3 + 1U)) = _S60;
    float _S61 = rr_0.z;
    *((&kernelContext_4)->arena_0+(o_3 + 2U)) = _S61;
    *((&kernelContext_4)->arena_0+(o_3 + 3U)) = sqrt(_S59 * _S59 + _S60 * _S60 + _S61 * _S61);
    return;
}

float seg_value_0(uint base_0, uint idx_0, uint width_0, uint mode_0, uint norm_0, uint row_2, uint c_2, KernelContext_0 thread* kernelContext_5)
{
    if(c_2 >= width_0)
    {
        return 0.0f;
    }
    uint r_2;
    if(mode_0 == 1U)
    {
        r_2 = kernelContext_5->table_0[idx_0 + row_2];
    }
    else
    {
        if(mode_0 == 2U)
        {
            r_2 = idx_0;
        }
        else
        {
            r_2 = row_2;
        }
    }
    float v_4 = *(kernelContext_5->arena_0+(base_0 + r_2 * width_0 + c_2));
    float v_5;
    if(norm_0 != 4294967295U)
    {
        v_5 = (v_4 - kernelContext_5->wts_0[norm_0 + c_2]) / kernelContext_5->wts_0[norm_0 + width_0 + c_2];
    }
    else
    {
        v_5 = v_4;
    }
    return v_5;
}

[[kernel]] void gnn_mlp_tiled(uint3 gid_3 [[threadgroup_position_in_grid]], uint3 lid_3 [[thread_position_in_threadgroup]], Params_0 constant* params_4 [[buffer(0)]], uint device* table_4 [[buffer(2)]], float device* arena_4 [[buffer(3)]], float device* wts_4 [[buffer(1)]])
{
    float4 r0_0;
    float4 r1_0;
    float4 r2_0;
    float4 r3_0;
    uint k0_0;
    uint q_0;
    float m_0;
    thread KernelContext_0 kernelContext_6;
    (&kernelContext_6)->params_0 = params_4;
    (&kernelContext_6)->table_0 = table_4;
    (&kernelContext_6)->arena_0 = arena_4;
    (&kernelContext_6)->wts_0 = wts_4;
    threadgroup array<array<float, int(384)>, int(4)> xs_4;
    (&kernelContext_6)->xs_0 = &xs_4;
    threadgroup array<float, int(128)> hid_4;
    (&kernelContext_6)->hid_0 = &hid_4;
    threadgroup array<float, int(128)> red_4;
    (&kernelContext_6)->red_0 = &red_4;
    threadgroup array<array<float, int(64)>, int(8)> tx_4;
    (&kernelContext_6)->tx_0 = &tx_4;
    threadgroup array<array<float, int(64)>, int(8)> tw_4;
    (&kernelContext_6)->tw_0 = &tw_4;
    threadgroup array<float, int(2048)> th_4;
    (&kernelContext_6)->th_0 = &th_4;
    threadgroup array<array<float, int(64)>, int(2)> tstat_4;
    (&kernelContext_6)->tstat_0 = &tstat_4;
    uint _S62 = P_0(3U, &kernelContext_6);
    uint tr_0 = 2048U / _S62;
    uint t_1 = lid_3.x;
    uint cg_0 = _S62 / 4U;
    uint _S63 = t_1 / cg_0;
    uint _S64 = _S63 * 4U;
    uint _S65 = t_1 % cg_0;
    uint cb_0 = _S65 * 4U;
    uint _S66 = gid_3.y;
    uint _S67 = P_0(2U, &kernelContext_6);
    uint _S68 = (_S66 * _S67 + gid_3.x) * tr_0;
    uint _S69 = P_0(1U, &kernelContext_6);
    uint _S70 = P_0(42U, &kernelContext_6);
    uint _S71 = P_0(43U, &kernelContext_6);
    uint _S72 = P_0(40U, &kernelContext_6);
    float4 _S73 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    uint _S74 = P_0(52U, &kernelContext_6);
    float4 bias_0;
    if(_S74 != 4294967295U)
    {
        uint _S75 = _S74 + cb_0;
        bias_0 = float4((&kernelContext_6)->wts_0[_S75], (&kernelContext_6)->wts_0[_S75 + 1U], (&kernelContext_6)->wts_0[_S75 + 2U], (&kernelContext_6)->wts_0[_S75 + 3U]);
    }
    else
    {
        bias_0 = _S73;
    }
    uint _S76 = P_0(4U, &kernelContext_6);
    uint s_4 = 0U;
    uint koff_0 = 0U;
    float4 r0_1 = bias_0;
    float4 r1_1 = bias_0;
    float4 r2_1 = bias_0;
    float4 r3_1 = bias_0;
    for(;;)
    {
        if(s_4 < _S76)
        {
        }
        else
        {
            break;
        }
        uint o_4 = 8U + 5U * s_4;
        uint _S77 = P_0(o_4, &kernelContext_6);
        uint _S78 = P_0(o_4 + 1U, &kernelContext_6);
        uint _S79 = P_0(o_4 + 2U, &kernelContext_6);
        uint _S80 = P_0(o_4 + 3U, &kernelContext_6);
        uint _S81 = P_0(o_4 + 4U, &kernelContext_6);
        if(_S80 >= 3U)
        {
            s_4 = s_4 + 1U;
            continue;
        }
        k0_0 = 0U;
        r0_0 = r0_1;
        r1_0 = r1_1;
        r2_0 = r2_1;
        r3_0 = r3_1;
        for(;;)
        {
            if(k0_0 < _S79)
            {
            }
            else
            {
                break;
            }
            q_0 = t_1;
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
                uint r_3 = q_0 - kk_0 * tr_0;
                uint row_3 = _S68 + r_3;
                if(row_3 < _S69)
                {
                    float _S82 = seg_value_0(_S77, _S78, _S79, _S80, _S81, row_3, k0_0 + kk_0, &kernelContext_6);
                    m_0 = _S82;
                }
                else
                {
                    m_0 = 0.0f;
                }
                (*(&kernelContext_6)->tx_0)[kk_0][r_3] = m_0;
                q_0 = q_0 + 128U;
            }
            uint q_1 = t_1;
            for(;;)
            {
                if(q_1 < (8U * _S62))
                {
                }
                else
                {
                    break;
                }
                uint kk_1 = q_1 / _S62;
                uint c_3 = q_1 - kk_1 * _S62;
                if((k0_0 + kk_1) < _S79)
                {
                    m_0 = (&kernelContext_6)->wts_0[_S72 + (koff_0 + k0_0 + kk_1) * _S70 + c_3];
                }
                else
                {
                    m_0 = 0.0f;
                }
                (*(&kernelContext_6)->tw_0)[kk_1][c_3] = m_0;
                q_1 = q_1 + 128U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint _S83 = cb_0 + 1U;
            uint _S84 = cb_0 + 2U;
            uint _S85 = cb_0 + 3U;
            float4 b_0 = float4((*(&kernelContext_6)->tw_0)[0U][cb_0], (*(&kernelContext_6)->tw_0)[0U][_S83], (*(&kernelContext_6)->tw_0)[0U][_S84], (*(&kernelContext_6)->tw_0)[0U][_S85]);
            uint _S86 = _S64 + 1U;
            uint _S87 = _S64 + 2U;
            uint _S88 = _S64 + 3U;
            float4 b_1 = float4((*(&kernelContext_6)->tw_0)[1U][cb_0], (*(&kernelContext_6)->tw_0)[1U][_S83], (*(&kernelContext_6)->tw_0)[1U][_S84], (*(&kernelContext_6)->tw_0)[1U][_S85]);
            float4 b_2 = float4((*(&kernelContext_6)->tw_0)[2U][cb_0], (*(&kernelContext_6)->tw_0)[2U][_S83], (*(&kernelContext_6)->tw_0)[2U][_S84], (*(&kernelContext_6)->tw_0)[2U][_S85]);
            float4 b_3 = float4((*(&kernelContext_6)->tw_0)[3U][cb_0], (*(&kernelContext_6)->tw_0)[3U][_S83], (*(&kernelContext_6)->tw_0)[3U][_S84], (*(&kernelContext_6)->tw_0)[3U][_S85]);
            float4 b_4 = float4((*(&kernelContext_6)->tw_0)[4U][cb_0], (*(&kernelContext_6)->tw_0)[4U][_S83], (*(&kernelContext_6)->tw_0)[4U][_S84], (*(&kernelContext_6)->tw_0)[4U][_S85]);
            float4 b_5 = float4((*(&kernelContext_6)->tw_0)[5U][cb_0], (*(&kernelContext_6)->tw_0)[5U][_S83], (*(&kernelContext_6)->tw_0)[5U][_S84], (*(&kernelContext_6)->tw_0)[5U][_S85]);
            float4 b_6 = float4((*(&kernelContext_6)->tw_0)[6U][cb_0], (*(&kernelContext_6)->tw_0)[6U][_S83], (*(&kernelContext_6)->tw_0)[6U][_S84], (*(&kernelContext_6)->tw_0)[6U][_S85]);
            float4 b_7 = float4((*(&kernelContext_6)->tw_0)[7U][cb_0], (*(&kernelContext_6)->tw_0)[7U][_S83], (*(&kernelContext_6)->tw_0)[7U][_S84], (*(&kernelContext_6)->tw_0)[7U][_S85]);
            float4 r0_2 = r0_0 + float4((*(&kernelContext_6)->tx_0)[0U][_S64])  * b_0 + float4((*(&kernelContext_6)->tx_0)[1U][_S64])  * b_1 + float4((*(&kernelContext_6)->tx_0)[2U][_S64])  * b_2 + float4((*(&kernelContext_6)->tx_0)[3U][_S64])  * b_3 + float4((*(&kernelContext_6)->tx_0)[4U][_S64])  * b_4 + float4((*(&kernelContext_6)->tx_0)[5U][_S64])  * b_5 + float4((*(&kernelContext_6)->tx_0)[6U][_S64])  * b_6 + float4((*(&kernelContext_6)->tx_0)[7U][_S64])  * b_7;
            float4 r1_2 = r1_0 + float4((*(&kernelContext_6)->tx_0)[0U][_S86])  * b_0 + float4((*(&kernelContext_6)->tx_0)[1U][_S86])  * b_1 + float4((*(&kernelContext_6)->tx_0)[2U][_S86])  * b_2 + float4((*(&kernelContext_6)->tx_0)[3U][_S86])  * b_3 + float4((*(&kernelContext_6)->tx_0)[4U][_S86])  * b_4 + float4((*(&kernelContext_6)->tx_0)[5U][_S86])  * b_5 + float4((*(&kernelContext_6)->tx_0)[6U][_S86])  * b_6 + float4((*(&kernelContext_6)->tx_0)[7U][_S86])  * b_7;
            float4 r2_2 = r2_0 + float4((*(&kernelContext_6)->tx_0)[0U][_S87])  * b_0 + float4((*(&kernelContext_6)->tx_0)[1U][_S87])  * b_1 + float4((*(&kernelContext_6)->tx_0)[2U][_S87])  * b_2 + float4((*(&kernelContext_6)->tx_0)[3U][_S87])  * b_3 + float4((*(&kernelContext_6)->tx_0)[4U][_S87])  * b_4 + float4((*(&kernelContext_6)->tx_0)[5U][_S87])  * b_5 + float4((*(&kernelContext_6)->tx_0)[6U][_S87])  * b_6 + float4((*(&kernelContext_6)->tx_0)[7U][_S87])  * b_7;
            float4 r3_2 = r3_0 + float4((*(&kernelContext_6)->tx_0)[0U][_S88])  * b_0 + float4((*(&kernelContext_6)->tx_0)[1U][_S88])  * b_1 + float4((*(&kernelContext_6)->tx_0)[2U][_S88])  * b_2 + float4((*(&kernelContext_6)->tx_0)[3U][_S88])  * b_3 + float4((*(&kernelContext_6)->tx_0)[4U][_S88])  * b_4 + float4((*(&kernelContext_6)->tx_0)[5U][_S88])  * b_5 + float4((*(&kernelContext_6)->tx_0)[6U][_S88])  * b_6 + float4((*(&kernelContext_6)->tx_0)[7U][_S88])  * b_7;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            k0_0 = k0_0 + 8U;
            r0_0 = r0_2;
            r1_0 = r1_2;
            r2_0 = r2_2;
            r3_0 = r3_2;
        }
        koff_0 = koff_0 + _S79;
        r0_1 = r0_0;
        r1_1 = r1_0;
        r2_1 = r2_0;
        r3_1 = r3_0;
        s_4 = s_4 + 1U;
    }
    s_4 = 0U;
    for(;;)
    {
        if(s_4 < _S76)
        {
        }
        else
        {
            break;
        }
        uint o_5 = 8U + 5U * s_4;
        uint _S89 = P_0(o_5, &kernelContext_6);
        uint _S90 = P_0(o_5 + 1U, &kernelContext_6);
        uint _S91 = P_0(o_5 + 3U, &kernelContext_6);
        if(_S91 < 3U)
        {
            s_4 = s_4 + 1U;
            continue;
        }
        uint _S92 = _S68 + _S64;
        if(_S92 < _S69)
        {
            if(_S91 == 3U)
            {
                k0_0 = (&kernelContext_6)->table_0[_S90 + _S92];
            }
            else
            {
                k0_0 = _S92;
            }
            uint _S93 = _S89 + k0_0 * _S70 + cb_0;
            float4 pv_0 = float4(*((&kernelContext_6)->arena_0+_S93), *((&kernelContext_6)->arena_0+(_S93 + 1U)), *((&kernelContext_6)->arena_0+(_S93 + 2U)), *((&kernelContext_6)->arena_0+(_S93 + 3U)));
            r0_0 = r0_1 + pv_0;
        }
        else
        {
            r0_0 = r0_1;
        }
        uint row_4 = _S92 + 1U;
        if(row_4 < _S69)
        {
            if(_S91 == 3U)
            {
                k0_0 = (&kernelContext_6)->table_0[_S90 + row_4];
            }
            else
            {
                k0_0 = row_4;
            }
            uint _S94 = _S89 + k0_0 * _S70 + cb_0;
            float4 pv_1 = float4(*((&kernelContext_6)->arena_0+_S94), *((&kernelContext_6)->arena_0+(_S94 + 1U)), *((&kernelContext_6)->arena_0+(_S94 + 2U)), *((&kernelContext_6)->arena_0+(_S94 + 3U)));
            r1_0 = r1_1 + pv_1;
        }
        else
        {
            r1_0 = r1_1;
        }
        uint row_5 = _S92 + 2U;
        if(row_5 < _S69)
        {
            if(_S91 == 3U)
            {
                k0_0 = (&kernelContext_6)->table_0[_S90 + row_5];
            }
            else
            {
                k0_0 = row_5;
            }
            uint _S95 = _S89 + k0_0 * _S70 + cb_0;
            float4 pv_2 = float4(*((&kernelContext_6)->arena_0+_S95), *((&kernelContext_6)->arena_0+(_S95 + 1U)), *((&kernelContext_6)->arena_0+(_S95 + 2U)), *((&kernelContext_6)->arena_0+(_S95 + 3U)));
            r2_0 = r2_1 + pv_2;
        }
        else
        {
            r2_0 = r2_1;
        }
        uint row_6 = _S92 + 3U;
        if(row_6 < _S69)
        {
            if(_S91 == 3U)
            {
                k0_0 = (&kernelContext_6)->table_0[_S90 + row_6];
            }
            else
            {
                k0_0 = row_6;
            }
            uint _S96 = _S89 + k0_0 * _S70 + cb_0;
            float4 pv_3 = float4(*((&kernelContext_6)->arena_0+_S96), *((&kernelContext_6)->arena_0+(_S96 + 1U)), *((&kernelContext_6)->arena_0+(_S96 + 2U)), *((&kernelContext_6)->arena_0+(_S96 + 3U)));
            r3_0 = r3_1 + pv_3;
        }
        else
        {
            r3_0 = r3_1;
        }
        r0_1 = r0_0;
        r1_1 = r1_0;
        r2_1 = r2_0;
        r3_1 = r3_0;
        s_4 = s_4 + 1U;
    }
    thread float4 y0_0 = r0_1;
    thread float4 y1_0 = r1_1;
    thread float4 y2_0 = r2_1;
    thread float4 y3_0 = r3_1;
    uint _S97 = P_0(44U, &kernelContext_6);
    float v_6;
    if(_S97 != 4294967295U)
    {
        uint _S98 = _S64 * _S62;
        uint _S99 = _S98 + cb_0;
        (*(&kernelContext_6)->th_0)[_S99] = silu_0(r0_1[0U]);
        uint _S100 = _S64 + 1U;
        uint _S101 = _S100 * _S62;
        uint _S102 = _S101 + cb_0;
        (*(&kernelContext_6)->th_0)[_S102] = silu_0(r1_1[0U]);
        uint _S103 = _S64 + 2U;
        uint _S104 = _S103 * _S62;
        uint _S105 = _S104 + cb_0;
        (*(&kernelContext_6)->th_0)[_S105] = silu_0(r2_1[0U]);
        uint _S106 = _S64 + 3U;
        uint _S107 = _S106 * _S62;
        uint _S108 = _S107 + cb_0;
        (*(&kernelContext_6)->th_0)[_S108] = silu_0(r3_1[0U]);
        (*(&kernelContext_6)->th_0)[_S99 + 1U] = silu_0(r0_1[1U]);
        (*(&kernelContext_6)->th_0)[_S102 + 1U] = silu_0(r1_1[1U]);
        (*(&kernelContext_6)->th_0)[_S105 + 1U] = silu_0(r2_1[1U]);
        (*(&kernelContext_6)->th_0)[_S108 + 1U] = silu_0(r3_1[1U]);
        (*(&kernelContext_6)->th_0)[_S99 + 2U] = silu_0(r0_1[2U]);
        (*(&kernelContext_6)->th_0)[_S102 + 2U] = silu_0(r1_1[2U]);
        (*(&kernelContext_6)->th_0)[_S105 + 2U] = silu_0(r2_1[2U]);
        (*(&kernelContext_6)->th_0)[_S108 + 2U] = silu_0(r3_1[2U]);
        (*(&kernelContext_6)->th_0)[_S99 + 3U] = silu_0(r0_1[3U]);
        (*(&kernelContext_6)->th_0)[_S102 + 3U] = silu_0(r1_1[3U]);
        (*(&kernelContext_6)->th_0)[_S105 + 3U] = silu_0(r2_1[3U]);
        (*(&kernelContext_6)->th_0)[_S108 + 3U] = silu_0(r3_1[3U]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        thread float4 b2_0 = _S73;
        if(cb_0 < _S71)
        {
            uint _S109 = P_0(53U, &kernelContext_6);
            m_0 = (&kernelContext_6)->wts_0[_S109 + cb_0];
        }
        else
        {
            m_0 = 0.0f;
        }
        b2_0.x = m_0;
        uint _S110 = cb_0 + 1U;
        if(_S110 < _S71)
        {
            uint _S111 = P_0(53U, &kernelContext_6);
            m_0 = (&kernelContext_6)->wts_0[_S111 + cb_0 + 1U];
        }
        else
        {
            m_0 = 0.0f;
        }
        b2_0.y = m_0;
        uint _S112 = cb_0 + 2U;
        if(_S112 < _S71)
        {
            uint _S113 = P_0(53U, &kernelContext_6);
            m_0 = (&kernelContext_6)->wts_0[_S113 + cb_0 + 2U];
        }
        else
        {
            m_0 = 0.0f;
        }
        b2_0.z = m_0;
        uint _S114 = cb_0 + 3U;
        if(_S114 < _S71)
        {
            uint _S115 = P_0(53U, &kernelContext_6);
            m_0 = (&kernelContext_6)->wts_0[_S115 + cb_0 + 3U];
        }
        else
        {
            m_0 = 0.0f;
        }
        b2_0.w = m_0;
        y0_0 = b2_0;
        y1_0 = b2_0;
        y2_0 = b2_0;
        y3_0 = b2_0;
        k0_0 = 0U;
        for(;;)
        {
            if(k0_0 < _S70)
            {
            }
            else
            {
                break;
            }
            q_0 = t_1;
            for(;;)
            {
                if(q_0 < (8U * _S62))
                {
                }
                else
                {
                    break;
                }
                uint kk_2 = q_0 / _S62;
                uint c_4 = q_0 - kk_2 * _S62;
                if(c_4 < _S71)
                {
                    m_0 = (&kernelContext_6)->wts_0[_S97 + (k0_0 + kk_2) * _S71 + c_4];
                }
                else
                {
                    m_0 = 0.0f;
                }
                (*(&kernelContext_6)->tw_0)[kk_2][c_4] = m_0;
                q_0 = q_0 + 128U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float4 b_8 = float4((*(&kernelContext_6)->tw_0)[0U][cb_0], (*(&kernelContext_6)->tw_0)[0U][_S110], (*(&kernelContext_6)->tw_0)[0U][_S112], (*(&kernelContext_6)->tw_0)[0U][_S114]);
            uint _S116 = _S98 + k0_0;
            float4 _S117 = y0_0 + float4((*(&kernelContext_6)->th_0)[_S116])  * b_8;
            y0_0 = _S117;
            uint _S118 = _S101 + k0_0;
            float4 _S119 = y1_0 + float4((*(&kernelContext_6)->th_0)[_S118])  * b_8;
            y1_0 = _S119;
            uint _S120 = _S104 + k0_0;
            float4 _S121 = y2_0 + float4((*(&kernelContext_6)->th_0)[_S120])  * b_8;
            y2_0 = _S121;
            uint _S122 = _S107 + k0_0;
            float4 _S123 = y3_0 + float4((*(&kernelContext_6)->th_0)[_S122])  * b_8;
            y3_0 = _S123;
            float4 b_9 = float4((*(&kernelContext_6)->tw_0)[1U][cb_0], (*(&kernelContext_6)->tw_0)[1U][_S110], (*(&kernelContext_6)->tw_0)[1U][_S112], (*(&kernelContext_6)->tw_0)[1U][_S114]);
            float4 _S124 = _S117 + float4((*(&kernelContext_6)->th_0)[_S116 + 1U])  * b_9;
            y0_0 = _S124;
            float4 _S125 = _S119 + float4((*(&kernelContext_6)->th_0)[_S118 + 1U])  * b_9;
            y1_0 = _S125;
            float4 _S126 = _S121 + float4((*(&kernelContext_6)->th_0)[_S120 + 1U])  * b_9;
            y2_0 = _S126;
            float4 _S127 = _S123 + float4((*(&kernelContext_6)->th_0)[_S122 + 1U])  * b_9;
            y3_0 = _S127;
            float4 b_10 = float4((*(&kernelContext_6)->tw_0)[2U][cb_0], (*(&kernelContext_6)->tw_0)[2U][_S110], (*(&kernelContext_6)->tw_0)[2U][_S112], (*(&kernelContext_6)->tw_0)[2U][_S114]);
            float4 _S128 = _S124 + float4((*(&kernelContext_6)->th_0)[_S116 + 2U])  * b_10;
            y0_0 = _S128;
            float4 _S129 = _S125 + float4((*(&kernelContext_6)->th_0)[_S118 + 2U])  * b_10;
            y1_0 = _S129;
            float4 _S130 = _S126 + float4((*(&kernelContext_6)->th_0)[_S120 + 2U])  * b_10;
            y2_0 = _S130;
            float4 _S131 = _S127 + float4((*(&kernelContext_6)->th_0)[_S122 + 2U])  * b_10;
            y3_0 = _S131;
            float4 b_11 = float4((*(&kernelContext_6)->tw_0)[3U][cb_0], (*(&kernelContext_6)->tw_0)[3U][_S110], (*(&kernelContext_6)->tw_0)[3U][_S112], (*(&kernelContext_6)->tw_0)[3U][_S114]);
            float4 _S132 = _S128 + float4((*(&kernelContext_6)->th_0)[_S116 + 3U])  * b_11;
            y0_0 = _S132;
            float4 _S133 = _S129 + float4((*(&kernelContext_6)->th_0)[_S118 + 3U])  * b_11;
            y1_0 = _S133;
            float4 _S134 = _S130 + float4((*(&kernelContext_6)->th_0)[_S120 + 3U])  * b_11;
            y2_0 = _S134;
            float4 _S135 = _S131 + float4((*(&kernelContext_6)->th_0)[_S122 + 3U])  * b_11;
            y3_0 = _S135;
            float4 b_12 = float4((*(&kernelContext_6)->tw_0)[4U][cb_0], (*(&kernelContext_6)->tw_0)[4U][_S110], (*(&kernelContext_6)->tw_0)[4U][_S112], (*(&kernelContext_6)->tw_0)[4U][_S114]);
            float4 _S136 = _S132 + float4((*(&kernelContext_6)->th_0)[_S116 + 4U])  * b_12;
            y0_0 = _S136;
            float4 _S137 = _S133 + float4((*(&kernelContext_6)->th_0)[_S118 + 4U])  * b_12;
            y1_0 = _S137;
            float4 _S138 = _S134 + float4((*(&kernelContext_6)->th_0)[_S120 + 4U])  * b_12;
            y2_0 = _S138;
            float4 _S139 = _S135 + float4((*(&kernelContext_6)->th_0)[_S122 + 4U])  * b_12;
            y3_0 = _S139;
            float4 b_13 = float4((*(&kernelContext_6)->tw_0)[5U][cb_0], (*(&kernelContext_6)->tw_0)[5U][_S110], (*(&kernelContext_6)->tw_0)[5U][_S112], (*(&kernelContext_6)->tw_0)[5U][_S114]);
            float4 _S140 = _S136 + float4((*(&kernelContext_6)->th_0)[_S116 + 5U])  * b_13;
            y0_0 = _S140;
            float4 _S141 = _S137 + float4((*(&kernelContext_6)->th_0)[_S118 + 5U])  * b_13;
            y1_0 = _S141;
            float4 _S142 = _S138 + float4((*(&kernelContext_6)->th_0)[_S120 + 5U])  * b_13;
            y2_0 = _S142;
            float4 _S143 = _S139 + float4((*(&kernelContext_6)->th_0)[_S122 + 5U])  * b_13;
            y3_0 = _S143;
            float4 b_14 = float4((*(&kernelContext_6)->tw_0)[6U][cb_0], (*(&kernelContext_6)->tw_0)[6U][_S110], (*(&kernelContext_6)->tw_0)[6U][_S112], (*(&kernelContext_6)->tw_0)[6U][_S114]);
            float4 _S144 = _S140 + float4((*(&kernelContext_6)->th_0)[_S116 + 6U])  * b_14;
            y0_0 = _S144;
            float4 _S145 = _S141 + float4((*(&kernelContext_6)->th_0)[_S118 + 6U])  * b_14;
            y1_0 = _S145;
            float4 _S146 = _S142 + float4((*(&kernelContext_6)->th_0)[_S120 + 6U])  * b_14;
            y2_0 = _S146;
            float4 _S147 = _S143 + float4((*(&kernelContext_6)->th_0)[_S122 + 6U])  * b_14;
            y3_0 = _S147;
            float4 b_15 = float4((*(&kernelContext_6)->tw_0)[7U][cb_0], (*(&kernelContext_6)->tw_0)[7U][_S110], (*(&kernelContext_6)->tw_0)[7U][_S112], (*(&kernelContext_6)->tw_0)[7U][_S114]);
            y0_0 = _S144 + float4((*(&kernelContext_6)->th_0)[_S116 + 7U])  * b_15;
            y1_0 = _S145 + float4((*(&kernelContext_6)->th_0)[_S118 + 7U])  * b_15;
            y2_0 = _S146 + float4((*(&kernelContext_6)->th_0)[_S120 + 7U])  * b_15;
            y3_0 = _S147 + float4((*(&kernelContext_6)->th_0)[_S122 + 7U])  * b_15;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            k0_0 = k0_0 + 8U;
        }
        uint _S148 = P_0(45U, &kernelContext_6);
        if(_S148 != 4294967295U)
        {
            (*(&kernelContext_6)->th_0)[_S99] = y0_0.x;
            (*(&kernelContext_6)->th_0)[_S102] = y1_0.x;
            (*(&kernelContext_6)->th_0)[_S105] = y2_0.x;
            (*(&kernelContext_6)->th_0)[_S108] = y3_0.x;
            (*(&kernelContext_6)->th_0)[_S99 + 1U] = y0_0.y;
            (*(&kernelContext_6)->th_0)[_S102 + 1U] = y1_0.y;
            (*(&kernelContext_6)->th_0)[_S105 + 1U] = y2_0.y;
            (*(&kernelContext_6)->th_0)[_S108 + 1U] = y3_0.y;
            (*(&kernelContext_6)->th_0)[_S99 + 2U] = y0_0.z;
            (*(&kernelContext_6)->th_0)[_S102 + 2U] = y1_0.z;
            (*(&kernelContext_6)->th_0)[_S105 + 2U] = y2_0.z;
            (*(&kernelContext_6)->th_0)[_S108 + 2U] = y3_0.z;
            (*(&kernelContext_6)->th_0)[_S99 + 3U] = y0_0.w;
            (*(&kernelContext_6)->th_0)[_S102 + 3U] = y1_0.w;
            (*(&kernelContext_6)->th_0)[_S105 + 3U] = y2_0.w;
            (*(&kernelContext_6)->th_0)[_S108 + 3U] = y3_0.w;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(t_1 < tr_0)
            {
                q_0 = 0U;
                m_0 = 0.0f;
                for(;;)
                {
                    if(q_0 < _S71)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float m_1 = m_0 + (*(&kernelContext_6)->th_0)[t_1 * _S62 + q_0];
                    q_0 = q_0 + 1U;
                    m_0 = m_1;
                }
                float _S149 = float(_S71);
                float m_2 = m_0 / _S149;
                q_0 = 0U;
                v_6 = 0.0f;
                for(;;)
                {
                    if(q_0 < _S71)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float d_2 = (*(&kernelContext_6)->th_0)[t_1 * _S62 + q_0] - m_2;
                    float v_7 = v_6 + d_2 * d_2;
                    q_0 = q_0 + 1U;
                    v_6 = v_7;
                }
                (*(&kernelContext_6)->tstat_0)[int(0)][t_1] = m_2;
                (*(&kernelContext_6)->tstat_0)[int(1)][t_1] = 1.0f / sqrt(v_6 / _S149 + 0.00000999999974738f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint _S150 = _S148 + cb_0;
            uint _S151 = _S148 + _S71;
            uint _S152 = _S151 + cb_0;
            y0_0.x = (y0_0.x - (*(&kernelContext_6)->tstat_0)[int(0)][_S64]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S64] * (&kernelContext_6)->wts_0[_S150] + (&kernelContext_6)->wts_0[_S152];
            y1_0.x = (y1_0.x - (*(&kernelContext_6)->tstat_0)[int(0)][_S100]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S100] * (&kernelContext_6)->wts_0[_S150] + (&kernelContext_6)->wts_0[_S152];
            y2_0.x = (y2_0.x - (*(&kernelContext_6)->tstat_0)[int(0)][_S103]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S103] * (&kernelContext_6)->wts_0[_S150] + (&kernelContext_6)->wts_0[_S152];
            y3_0.x = (y3_0.x - (*(&kernelContext_6)->tstat_0)[int(0)][_S106]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S106] * (&kernelContext_6)->wts_0[_S150] + (&kernelContext_6)->wts_0[_S152];
            uint _S153 = _S148 + _S110;
            uint _S154 = _S151 + _S110;
            y0_0.y = (y0_0.y - (*(&kernelContext_6)->tstat_0)[int(0)][_S64]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S64] * (&kernelContext_6)->wts_0[_S153] + (&kernelContext_6)->wts_0[_S154];
            y1_0.y = (y1_0.y - (*(&kernelContext_6)->tstat_0)[int(0)][_S100]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S100] * (&kernelContext_6)->wts_0[_S153] + (&kernelContext_6)->wts_0[_S154];
            y2_0.y = (y2_0.y - (*(&kernelContext_6)->tstat_0)[int(0)][_S103]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S103] * (&kernelContext_6)->wts_0[_S153] + (&kernelContext_6)->wts_0[_S154];
            y3_0.y = (y3_0.y - (*(&kernelContext_6)->tstat_0)[int(0)][_S106]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S106] * (&kernelContext_6)->wts_0[_S153] + (&kernelContext_6)->wts_0[_S154];
            uint _S155 = _S148 + _S112;
            uint _S156 = _S151 + _S112;
            y0_0.z = (y0_0.z - (*(&kernelContext_6)->tstat_0)[int(0)][_S64]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S64] * (&kernelContext_6)->wts_0[_S155] + (&kernelContext_6)->wts_0[_S156];
            y1_0.z = (y1_0.z - (*(&kernelContext_6)->tstat_0)[int(0)][_S100]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S100] * (&kernelContext_6)->wts_0[_S155] + (&kernelContext_6)->wts_0[_S156];
            y2_0.z = (y2_0.z - (*(&kernelContext_6)->tstat_0)[int(0)][_S103]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S103] * (&kernelContext_6)->wts_0[_S155] + (&kernelContext_6)->wts_0[_S156];
            y3_0.z = (y3_0.z - (*(&kernelContext_6)->tstat_0)[int(0)][_S106]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S106] * (&kernelContext_6)->wts_0[_S155] + (&kernelContext_6)->wts_0[_S156];
            uint _S157 = _S148 + _S114;
            uint _S158 = _S151 + _S114;
            y0_0.w = (y0_0.w - (*(&kernelContext_6)->tstat_0)[int(0)][_S64]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S64] * (&kernelContext_6)->wts_0[_S157] + (&kernelContext_6)->wts_0[_S158];
            y1_0.w = (y1_0.w - (*(&kernelContext_6)->tstat_0)[int(0)][_S100]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S100] * (&kernelContext_6)->wts_0[_S157] + (&kernelContext_6)->wts_0[_S158];
            y2_0.w = (y2_0.w - (*(&kernelContext_6)->tstat_0)[int(0)][_S103]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S103] * (&kernelContext_6)->wts_0[_S157] + (&kernelContext_6)->wts_0[_S158];
            y3_0.w = (y3_0.w - (*(&kernelContext_6)->tstat_0)[int(0)][_S106]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S106] * (&kernelContext_6)->wts_0[_S157] + (&kernelContext_6)->wts_0[_S158];
        }
    }
    uint _S159 = P_0(51U, &kernelContext_6);
    float _S160 = (as_type<float>((_S159)));
    uint _S161 = P_0(48U, &kernelContext_6);
    uint _S162 = _S68 + _S64;
    float4 _S163 = y0_0;
    if(_S162 < _S69)
    {
        if(cb_0 < _S71)
        {
            float v_8 = _S160 * _S163[0U];
            if(_S161 != 4294967295U)
            {
                uint _S164 = P_0(49U, &kernelContext_6);
                v_6 = v_8 + *((&kernelContext_6)->arena_0+(_S161 + _S162 * _S164 + cb_0));
            }
            else
            {
                v_6 = v_8;
            }
            uint _S165 = P_0(46U, &kernelContext_6);
            uint _S166 = P_0(47U, &kernelContext_6);
            uint o_6 = _S165 + _S162 * _S166 + cb_0;
            uint _S167 = P_0(50U, &kernelContext_6);
            if((_S167 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_6) = *((&kernelContext_6)->arena_0+o_6) + v_6;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_6) = v_6;
            }
        }
        uint c_5 = cb_0 + 1U;
        if(c_5 < _S71)
        {
            float v_9 = _S160 * _S163[1U];
            if(_S161 != 4294967295U)
            {
                uint _S168 = P_0(49U, &kernelContext_6);
                v_6 = v_9 + *((&kernelContext_6)->arena_0+(_S161 + _S162 * _S168 + c_5));
            }
            else
            {
                v_6 = v_9;
            }
            uint _S169 = P_0(46U, &kernelContext_6);
            uint _S170 = P_0(47U, &kernelContext_6);
            uint o_7 = _S169 + _S162 * _S170 + c_5;
            uint _S171 = P_0(50U, &kernelContext_6);
            if((_S171 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_7) = *((&kernelContext_6)->arena_0+o_7) + v_6;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_7) = v_6;
            }
        }
        uint c_6 = cb_0 + 2U;
        if(c_6 < _S71)
        {
            float v_10 = _S160 * _S163[2U];
            if(_S161 != 4294967295U)
            {
                uint _S172 = P_0(49U, &kernelContext_6);
                v_6 = v_10 + *((&kernelContext_6)->arena_0+(_S161 + _S162 * _S172 + c_6));
            }
            else
            {
                v_6 = v_10;
            }
            uint _S173 = P_0(46U, &kernelContext_6);
            uint _S174 = P_0(47U, &kernelContext_6);
            uint o_8 = _S173 + _S162 * _S174 + c_6;
            uint _S175 = P_0(50U, &kernelContext_6);
            if((_S175 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_8) = *((&kernelContext_6)->arena_0+o_8) + v_6;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_8) = v_6;
            }
        }
        uint c_7 = cb_0 + 3U;
        if(c_7 < _S71)
        {
            float v_11 = _S160 * _S163[3U];
            if(_S161 != 4294967295U)
            {
                uint _S176 = P_0(49U, &kernelContext_6);
                v_6 = v_11 + *((&kernelContext_6)->arena_0+(_S161 + _S162 * _S176 + c_7));
            }
            else
            {
                v_6 = v_11;
            }
            uint _S177 = P_0(46U, &kernelContext_6);
            uint _S178 = P_0(47U, &kernelContext_6);
            uint o_9 = _S177 + _S162 * _S178 + c_7;
            uint _S179 = P_0(50U, &kernelContext_6);
            if((_S179 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_9) = *((&kernelContext_6)->arena_0+o_9) + v_6;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_9) = v_6;
            }
        }
    }
    uint row_7 = _S162 + 1U;
    float4 _S180 = y1_0;
    if(row_7 < _S69)
    {
        if(cb_0 < _S71)
        {
            float v_12 = _S160 * _S180[0U];
            if(_S161 != 4294967295U)
            {
                uint _S181 = P_0(49U, &kernelContext_6);
                v_6 = v_12 + *((&kernelContext_6)->arena_0+(_S161 + row_7 * _S181 + cb_0));
            }
            else
            {
                v_6 = v_12;
            }
            uint _S182 = P_0(46U, &kernelContext_6);
            uint _S183 = P_0(47U, &kernelContext_6);
            uint o_10 = _S182 + row_7 * _S183 + cb_0;
            uint _S184 = P_0(50U, &kernelContext_6);
            if((_S184 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_10) = *((&kernelContext_6)->arena_0+o_10) + v_6;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_10) = v_6;
            }
        }
        uint c_8 = cb_0 + 1U;
        if(c_8 < _S71)
        {
            float v_13 = _S160 * _S180[1U];
            if(_S161 != 4294967295U)
            {
                uint _S185 = P_0(49U, &kernelContext_6);
                v_6 = v_13 + *((&kernelContext_6)->arena_0+(_S161 + row_7 * _S185 + c_8));
            }
            else
            {
                v_6 = v_13;
            }
            uint _S186 = P_0(46U, &kernelContext_6);
            uint _S187 = P_0(47U, &kernelContext_6);
            uint o_11 = _S186 + row_7 * _S187 + c_8;
            uint _S188 = P_0(50U, &kernelContext_6);
            if((_S188 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_11) = *((&kernelContext_6)->arena_0+o_11) + v_6;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_11) = v_6;
            }
        }
        uint c_9 = cb_0 + 2U;
        if(c_9 < _S71)
        {
            float v_14 = _S160 * _S180[2U];
            if(_S161 != 4294967295U)
            {
                uint _S189 = P_0(49U, &kernelContext_6);
                v_6 = v_14 + *((&kernelContext_6)->arena_0+(_S161 + row_7 * _S189 + c_9));
            }
            else
            {
                v_6 = v_14;
            }
            uint _S190 = P_0(46U, &kernelContext_6);
            uint _S191 = P_0(47U, &kernelContext_6);
            uint o_12 = _S190 + row_7 * _S191 + c_9;
            uint _S192 = P_0(50U, &kernelContext_6);
            if((_S192 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_12) = *((&kernelContext_6)->arena_0+o_12) + v_6;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_12) = v_6;
            }
        }
        uint c_10 = cb_0 + 3U;
        if(c_10 < _S71)
        {
            float v_15 = _S160 * _S180[3U];
            if(_S161 != 4294967295U)
            {
                uint _S193 = P_0(49U, &kernelContext_6);
                v_6 = v_15 + *((&kernelContext_6)->arena_0+(_S161 + row_7 * _S193 + c_10));
            }
            else
            {
                v_6 = v_15;
            }
            uint _S194 = P_0(46U, &kernelContext_6);
            uint _S195 = P_0(47U, &kernelContext_6);
            uint o_13 = _S194 + row_7 * _S195 + c_10;
            uint _S196 = P_0(50U, &kernelContext_6);
            if((_S196 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_13) = *((&kernelContext_6)->arena_0+o_13) + v_6;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_13) = v_6;
            }
        }
    }
    uint row_8 = _S162 + 2U;
    float4 _S197 = y2_0;
    if(row_8 < _S69)
    {
        if(cb_0 < _S71)
        {
            float v_16 = _S160 * _S197[0U];
            if(_S161 != 4294967295U)
            {
                uint _S198 = P_0(49U, &kernelContext_6);
                v_6 = v_16 + *((&kernelContext_6)->arena_0+(_S161 + row_8 * _S198 + cb_0));
            }
            else
            {
                v_6 = v_16;
            }
            uint _S199 = P_0(46U, &kernelContext_6);
            uint _S200 = P_0(47U, &kernelContext_6);
            uint o_14 = _S199 + row_8 * _S200 + cb_0;
            uint _S201 = P_0(50U, &kernelContext_6);
            if((_S201 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_14) = *((&kernelContext_6)->arena_0+o_14) + v_6;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_14) = v_6;
            }
        }
        uint c_11 = cb_0 + 1U;
        if(c_11 < _S71)
        {
            float v_17 = _S160 * _S197[1U];
            if(_S161 != 4294967295U)
            {
                uint _S202 = P_0(49U, &kernelContext_6);
                v_6 = v_17 + *((&kernelContext_6)->arena_0+(_S161 + row_8 * _S202 + c_11));
            }
            else
            {
                v_6 = v_17;
            }
            uint _S203 = P_0(46U, &kernelContext_6);
            uint _S204 = P_0(47U, &kernelContext_6);
            uint o_15 = _S203 + row_8 * _S204 + c_11;
            uint _S205 = P_0(50U, &kernelContext_6);
            if((_S205 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_15) = *((&kernelContext_6)->arena_0+o_15) + v_6;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_15) = v_6;
            }
        }
        uint c_12 = cb_0 + 2U;
        if(c_12 < _S71)
        {
            float v_18 = _S160 * _S197[2U];
            if(_S161 != 4294967295U)
            {
                uint _S206 = P_0(49U, &kernelContext_6);
                v_6 = v_18 + *((&kernelContext_6)->arena_0+(_S161 + row_8 * _S206 + c_12));
            }
            else
            {
                v_6 = v_18;
            }
            uint _S207 = P_0(46U, &kernelContext_6);
            uint _S208 = P_0(47U, &kernelContext_6);
            uint o_16 = _S207 + row_8 * _S208 + c_12;
            uint _S209 = P_0(50U, &kernelContext_6);
            if((_S209 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_16) = *((&kernelContext_6)->arena_0+o_16) + v_6;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_16) = v_6;
            }
        }
        uint c_13 = cb_0 + 3U;
        if(c_13 < _S71)
        {
            float v_19 = _S160 * _S197[3U];
            if(_S161 != 4294967295U)
            {
                uint _S210 = P_0(49U, &kernelContext_6);
                v_6 = v_19 + *((&kernelContext_6)->arena_0+(_S161 + row_8 * _S210 + c_13));
            }
            else
            {
                v_6 = v_19;
            }
            uint _S211 = P_0(46U, &kernelContext_6);
            uint _S212 = P_0(47U, &kernelContext_6);
            uint o_17 = _S211 + row_8 * _S212 + c_13;
            uint _S213 = P_0(50U, &kernelContext_6);
            if((_S213 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_17) = *((&kernelContext_6)->arena_0+o_17) + v_6;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_17) = v_6;
            }
        }
    }
    uint row_9 = _S162 + 3U;
    float4 _S214 = y3_0;
    if(row_9 < _S69)
    {
        if(cb_0 < _S71)
        {
            float v_20 = _S160 * _S214[0U];
            if(_S161 != 4294967295U)
            {
                uint _S215 = P_0(49U, &kernelContext_6);
                v_6 = v_20 + *((&kernelContext_6)->arena_0+(_S161 + row_9 * _S215 + cb_0));
            }
            else
            {
                v_6 = v_20;
            }
            uint _S216 = P_0(46U, &kernelContext_6);
            uint _S217 = P_0(47U, &kernelContext_6);
            uint o_18 = _S216 + row_9 * _S217 + cb_0;
            uint _S218 = P_0(50U, &kernelContext_6);
            if((_S218 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_18) = *((&kernelContext_6)->arena_0+o_18) + v_6;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_18) = v_6;
            }
        }
        uint c_14 = cb_0 + 1U;
        if(c_14 < _S71)
        {
            float v_21 = _S160 * _S214[1U];
            if(_S161 != 4294967295U)
            {
                uint _S219 = P_0(49U, &kernelContext_6);
                v_6 = v_21 + *((&kernelContext_6)->arena_0+(_S161 + row_9 * _S219 + c_14));
            }
            else
            {
                v_6 = v_21;
            }
            uint _S220 = P_0(46U, &kernelContext_6);
            uint _S221 = P_0(47U, &kernelContext_6);
            uint o_19 = _S220 + row_9 * _S221 + c_14;
            uint _S222 = P_0(50U, &kernelContext_6);
            if((_S222 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_19) = *((&kernelContext_6)->arena_0+o_19) + v_6;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_19) = v_6;
            }
        }
        uint c_15 = cb_0 + 2U;
        if(c_15 < _S71)
        {
            float v_22 = _S160 * _S214[2U];
            if(_S161 != 4294967295U)
            {
                uint _S223 = P_0(49U, &kernelContext_6);
                v_6 = v_22 + *((&kernelContext_6)->arena_0+(_S161 + row_9 * _S223 + c_15));
            }
            else
            {
                v_6 = v_22;
            }
            uint _S224 = P_0(46U, &kernelContext_6);
            uint _S225 = P_0(47U, &kernelContext_6);
            uint o_20 = _S224 + row_9 * _S225 + c_15;
            uint _S226 = P_0(50U, &kernelContext_6);
            if((_S226 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_20) = *((&kernelContext_6)->arena_0+o_20) + v_6;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_20) = v_6;
            }
        }
        uint c_16 = cb_0 + 3U;
        if(c_16 < _S71)
        {
            float v_23 = _S160 * _S214[3U];
            if(_S161 != 4294967295U)
            {
                uint _S227 = P_0(49U, &kernelContext_6);
                v_6 = v_23 + *((&kernelContext_6)->arena_0+(_S161 + row_9 * _S227 + c_16));
            }
            else
            {
                v_6 = v_23;
            }
            uint _S228 = P_0(46U, &kernelContext_6);
            uint _S229 = P_0(47U, &kernelContext_6);
            uint o_21 = _S228 + row_9 * _S229 + c_16;
            uint _S230 = P_0(50U, &kernelContext_6);
            if((_S230 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_21) = *((&kernelContext_6)->arena_0+o_21) + v_6;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_21) = v_6;
            }
        }
    }
    return;
}

