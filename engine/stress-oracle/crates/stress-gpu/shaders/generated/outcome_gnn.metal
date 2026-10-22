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
                _S1 = _S14 == 4U;
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
            if(_S14 >= 4U)
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
                uint c_0;
                if(_S14 == 3U)
                {
                    uint _S16 = _S12 + row_0;
                    uint _S17 = (&kernelContext_2)->table_0[_S16];
                    uint _S18 = (&kernelContext_2)->table_0[_S16 + 1U];
                    c_0 = lane_1;
                    for(;;)
                    {
                        if(c_0 < _S13)
                        {
                        }
                        else
                        {
                            break;
                        }
                        uint e_0 = _S17;
                        v_1 = 0.0f;
                        for(;;)
                        {
                            if(e_0 < _S18)
                            {
                            }
                            else
                            {
                                break;
                            }
                            uint _S19 = _S15 + 2U * e_0;
                            float v_2 = v_1 + *((&kernelContext_2)->arena_0+((&kernelContext_2)->table_0[_S19] + c_0)) * (as_type<float>(((&kernelContext_2)->table_0[_S19 + 1U])));
                            e_0 = e_0 + 1U;
                            v_1 = v_2;
                        }
                        (*(&kernelContext_2)->xs_0)[slot_0][k_0 + c_0] = v_1;
                        c_0 = c_0 + _S2;
                    }
                }
                else
                {
                    c_0 = lane_1;
                    for(;;)
                    {
                        if(c_0 < _S13)
                        {
                        }
                        else
                        {
                            break;
                        }
                        float v_3 = *((&kernelContext_2)->arena_0+(_S11 + r_0 * _S13 + c_0));
                        if(_S15 != 4294967295U)
                        {
                            v_1 = (v_3 - (&kernelContext_2)->wts_0[_S15 + c_0]) / (&kernelContext_2)->wts_0[_S15 + _S13 + c_0];
                        }
                        else
                        {
                            v_1 = v_3;
                        }
                        (*(&kernelContext_2)->xs_0)[slot_0][k_0 + c_0] = v_1;
                        c_0 = c_0 + _S2;
                    }
                }
                pre_0 = pre_1;
            }
            pre_1 = pre_0;
        }
        if(_S14 < 4U)
        {
            k_0 = k_0 + _S13;
        }
        s_1 = s_1 + 1U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S20 = P_0(41U, &kernelContext_2);
    uint _S21 = P_0(40U, &kernelContext_2);
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
        uint _S22 = P_0(52U, &kernelContext_2);
        if(_S22 == 4294967295U)
        {
            pre_0 = 0.0f;
        }
        else
        {
            pre_0 = (&kernelContext_2)->wts_0[_S22 + lane_1];
        }
        r_0 = 0U;
        for(;;)
        {
            if(r_0 < _S20)
            {
            }
            else
            {
                break;
            }
            float a_0 = pre_0 + (&kernelContext_2)->wts_0[_S21 + r_0 * _S8 + lane_1] * (*(&kernelContext_2)->xs_0)[slot_0][r_0];
            r_0 = r_0 + 1U;
            pre_0 = a_0;
        }
        pre_1 = pre_1 + pre_0;
    }
    uint _S23 = P_0(44U, &kernelContext_2);
    if(_S23 == 4294967295U)
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
            uint _S24 = P_0(53U, &kernelContext_2);
            float _S25 = (&kernelContext_2)->wts_0[_S24 + lane_1];
            r_0 = 0U;
            pre_0 = _S25;
            for(;;)
            {
                if(r_0 < _S8)
                {
                }
                else
                {
                    break;
                }
                float a_1 = pre_0 + (&kernelContext_2)->wts_0[_S23 + r_0 * _S9 + lane_1] * (*(&kernelContext_2)->hid_0)[tid_1 - lane_1 + r_0];
                r_0 = r_0 + 1U;
                pre_0 = a_1;
            }
        }
        else
        {
            pre_0 = 0.0f;
        }
        uint _S26 = P_0(45U, &kernelContext_2);
        if(_S26 != 4294967295U)
        {
            float _S27 = slot_sum_0(pre_0, tid_1, lane_1, _S2, &kernelContext_2);
            float _S28 = float(_S9);
            float d_0 = pre_0 - _S27 / _S28;
            float _S29 = slot_sum_0(d_0 * d_0, tid_1, lane_1, _S2, &kernelContext_2);
            pre_0 = d_0 * (1.0f / sqrt(_S29 / _S28 + 0.00000999999974738f)) * (&kernelContext_2)->wts_0[_S26 + lane_1] + (&kernelContext_2)->wts_0[_S26 + _S9 + lane_1];
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
        uint _S30 = P_0(51U, &kernelContext_2);
        float v_4 = (as_type<float>((_S30))) * pre_0;
        uint _S31 = P_0(48U, &kernelContext_2);
        if(_S31 != 4294967295U)
        {
            uint _S32 = P_0(49U, &kernelContext_2);
            v_1 = v_4 + *((&kernelContext_2)->arena_0+(_S31 + row_0 * _S32 + lane_1));
        }
        else
        {
            v_1 = v_4;
        }
        uint _S33 = P_0(46U, &kernelContext_2);
        uint _S34 = P_0(47U, &kernelContext_2);
        uint o_1 = _S33 + row_0 * _S34 + lane_1;
        uint _S35 = P_0(50U, &kernelContext_2);
        if((_S35 & 1U) != 0U)
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
    uint _S36 = P_0(3U, &kernelContext_3);
    uint _S37 = lid_1.x;
    uint slot_1 = _S37 / _S36;
    uint lane_2 = _S37 % _S36;
    uint _S38 = gid_1.y;
    uint _S39 = P_0(2U, &kernelContext_3);
    uint _S40 = _S38 * _S39 + gid_1.x;
    uint _S41 = 128U / _S36;
    uint row_1 = _S40 * _S41 + slot_1;
    uint _S42 = P_0(1U, &kernelContext_3);
    bool _S43;
    if(row_1 >= _S42)
    {
        _S43 = true;
    }
    else
    {
        uint _S44 = P_0(4U, &kernelContext_3);
        _S43 = lane_2 >= _S44;
    }
    if(_S43)
    {
        return;
    }
    uint _S45 = P_0(5U, &kernelContext_3);
    uint _S46 = _S45 + row_1;
    uint e0_0 = (&kernelContext_3)->table_0[_S46];
    uint _S47 = (&kernelContext_3)->table_0[_S46 + 1U];
    uint _S48 = P_0(6U, &kernelContext_3);
    uint e_1 = e0_0;
    float s_2 = 0.0f;
    for(;;)
    {
        if(e_1 < _S47)
        {
        }
        else
        {
            break;
        }
        uint _S49 = _S48 + 2U * e_1;
        float s_3 = s_2 + *((&kernelContext_3)->arena_0+((&kernelContext_3)->table_0[_S49] + lane_2)) * (as_type<float>(((&kernelContext_3)->table_0[_S49 + 1U])));
        e_1 = e_1 + 1U;
        s_2 = s_3;
    }
    uint _S50 = P_0(7U, &kernelContext_3);
    uint _S51 = P_0(8U, &kernelContext_3);
    uint o_2 = _S50 + row_1 * _S51 + lane_2;
    uint _S52 = P_0(9U, &kernelContext_3);
    if((_S52 & 1U) != 0U)
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
    uint _S53 = gid_2.y;
    uint _S54 = P_0(2U, &kernelContext_4);
    uint r_1 = (_S53 * _S54 + gid_2.x) * 128U + lid_2.x;
    uint _S55 = P_0(1U, &kernelContext_4);
    if(r_1 >= _S55)
    {
        return;
    }
    uint _S56 = P_0(5U, &kernelContext_4);
    uint _S57 = 4U * r_1;
    uint t_0 = _S56 + _S57;
    uint i_1 = (&kernelContext_4)->table_0[t_0];
    uint j_0 = (&kernelContext_4)->table_0[t_0 + 1U];
    uint c_1 = (&kernelContext_4)->table_0[t_0 + 2U];
    uint _S58 = P_0(4U, &kernelContext_4);
    uint _S59 = _S58 + 3U * i_1;
    uint _S60 = _S58 + 3U * j_0;
    float3 d_1 = float3(*((&kernelContext_4)->arena_0+_S59) - *((&kernelContext_4)->arena_0+_S60), *((&kernelContext_4)->arena_0+(_S59 + 1U)) - *((&kernelContext_4)->arena_0+(_S60 + 1U)), *((&kernelContext_4)->arena_0+(_S59 + 2U)) - *((&kernelContext_4)->arena_0+(_S60 + 2U)));
    uint _S61 = P_0(6U, &kernelContext_4);
    uint qb_0 = _S61 + 4U * c_1;
    float3 u_0 = - float3(*((&kernelContext_4)->arena_0+(qb_0 + 1U)), *((&kernelContext_4)->arena_0+(qb_0 + 2U)), *((&kernelContext_4)->arena_0+(qb_0 + 3U)));
    float3 tt_0 = float3(2.0f)  * cross(u_0, d_1);
    float3 rr_0 = (d_1 + float3(*((&kernelContext_4)->arena_0+qb_0))  * tt_0 + cross(u_0, tt_0)) / float3((as_type<float>(((&kernelContext_4)->table_0[t_0 + 3U])))) ;
    uint _S62 = P_0(7U, &kernelContext_4);
    uint o_3 = _S62 + _S57;
    float _S63 = rr_0.x;
    *((&kernelContext_4)->arena_0+o_3) = _S63;
    float _S64 = rr_0.y;
    *((&kernelContext_4)->arena_0+(o_3 + 1U)) = _S64;
    float _S65 = rr_0.z;
    *((&kernelContext_4)->arena_0+(o_3 + 2U)) = _S65;
    *((&kernelContext_4)->arena_0+(o_3 + 3U)) = sqrt(_S63 * _S63 + _S64 * _S64 + _S65 * _S65);
    return;
}

float seg_value_0(uint base_0, uint idx_0, uint width_0, uint mode_0, uint norm_0, uint row_2, uint c_2, KernelContext_0 thread* kernelContext_5)
{
    if(c_2 >= width_0)
    {
        return 0.0f;
    }
    uint e_2;
    float v_5;
    if(mode_0 == 3U)
    {
        uint _S66 = idx_0 + row_2;
        uint _S67 = kernelContext_5->table_0[_S66 + 1U];
        e_2 = kernelContext_5->table_0[_S66];
        v_5 = 0.0f;
        for(;;)
        {
            if(e_2 < _S67)
            {
            }
            else
            {
                break;
            }
            uint _S68 = norm_0 + 2U * e_2;
            float v_6 = v_5 + *(kernelContext_5->arena_0+(kernelContext_5->table_0[_S68] + c_2)) * (as_type<float>((kernelContext_5->table_0[_S68 + 1U])));
            e_2 = e_2 + 1U;
            v_5 = v_6;
        }
        return v_5;
    }
    if(mode_0 == 1U)
    {
        e_2 = kernelContext_5->table_0[idx_0 + row_2];
    }
    else
    {
        if(mode_0 == 2U)
        {
            e_2 = idx_0;
        }
        else
        {
            e_2 = row_2;
        }
    }
    float v_7 = *(kernelContext_5->arena_0+(base_0 + e_2 * width_0 + c_2));
    if(norm_0 != 4294967295U)
    {
        v_5 = (v_7 - kernelContext_5->wts_0[norm_0 + c_2]) / kernelContext_5->wts_0[norm_0 + width_0 + c_2];
    }
    else
    {
        v_5 = v_7;
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
    uint _S69 = P_0(3U, &kernelContext_6);
    uint tr_0 = 2048U / _S69;
    uint t_1 = lid_3.x;
    uint cg_0 = _S69 / 4U;
    uint _S70 = t_1 / cg_0;
    uint _S71 = _S70 * 4U;
    uint _S72 = t_1 % cg_0;
    uint cb_0 = _S72 * 4U;
    uint _S73 = gid_3.y;
    uint _S74 = P_0(2U, &kernelContext_6);
    uint _S75 = (_S73 * _S74 + gid_3.x) * tr_0;
    uint _S76 = P_0(1U, &kernelContext_6);
    uint _S77 = P_0(42U, &kernelContext_6);
    uint _S78 = P_0(43U, &kernelContext_6);
    uint _S79 = P_0(40U, &kernelContext_6);
    float4 _S80 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    uint _S81 = P_0(52U, &kernelContext_6);
    float4 bias_0;
    if(_S81 != 4294967295U)
    {
        uint _S82 = _S81 + cb_0;
        bias_0 = float4((&kernelContext_6)->wts_0[_S82], (&kernelContext_6)->wts_0[_S82 + 1U], (&kernelContext_6)->wts_0[_S82 + 2U], (&kernelContext_6)->wts_0[_S82 + 3U]);
    }
    else
    {
        bias_0 = _S80;
    }
    uint _S83 = P_0(4U, &kernelContext_6);
    uint s_4 = 0U;
    uint koff_0 = 0U;
    float4 r0_1 = bias_0;
    float4 r1_1 = bias_0;
    float4 r2_1 = bias_0;
    float4 r3_1 = bias_0;
    for(;;)
    {
        if(s_4 < _S83)
        {
        }
        else
        {
            break;
        }
        uint o_4 = 8U + 5U * s_4;
        uint _S84 = P_0(o_4, &kernelContext_6);
        uint _S85 = P_0(o_4 + 1U, &kernelContext_6);
        uint _S86 = P_0(o_4 + 2U, &kernelContext_6);
        uint _S87 = P_0(o_4 + 3U, &kernelContext_6);
        uint _S88 = P_0(o_4 + 4U, &kernelContext_6);
        if(_S87 >= 4U)
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
            if(k0_0 < _S86)
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
                uint r_2 = q_0 - kk_0 * tr_0;
                uint row_3 = _S75 + r_2;
                if(row_3 < _S76)
                {
                    float _S89 = seg_value_0(_S84, _S85, _S86, _S87, _S88, row_3, k0_0 + kk_0, &kernelContext_6);
                    m_0 = _S89;
                }
                else
                {
                    m_0 = 0.0f;
                }
                (*(&kernelContext_6)->tx_0)[kk_0][r_2] = m_0;
                q_0 = q_0 + 128U;
            }
            uint q_1 = t_1;
            for(;;)
            {
                if(q_1 < (8U * _S69))
                {
                }
                else
                {
                    break;
                }
                uint kk_1 = q_1 / _S69;
                uint c_3 = q_1 - kk_1 * _S69;
                if((k0_0 + kk_1) < _S86)
                {
                    m_0 = (&kernelContext_6)->wts_0[_S79 + (koff_0 + k0_0 + kk_1) * _S77 + c_3];
                }
                else
                {
                    m_0 = 0.0f;
                }
                (*(&kernelContext_6)->tw_0)[kk_1][c_3] = m_0;
                q_1 = q_1 + 128U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint _S90 = cb_0 + 1U;
            uint _S91 = cb_0 + 2U;
            uint _S92 = cb_0 + 3U;
            float4 b_0 = float4((*(&kernelContext_6)->tw_0)[0U][cb_0], (*(&kernelContext_6)->tw_0)[0U][_S90], (*(&kernelContext_6)->tw_0)[0U][_S91], (*(&kernelContext_6)->tw_0)[0U][_S92]);
            uint _S93 = _S71 + 1U;
            uint _S94 = _S71 + 2U;
            uint _S95 = _S71 + 3U;
            float4 b_1 = float4((*(&kernelContext_6)->tw_0)[1U][cb_0], (*(&kernelContext_6)->tw_0)[1U][_S90], (*(&kernelContext_6)->tw_0)[1U][_S91], (*(&kernelContext_6)->tw_0)[1U][_S92]);
            float4 b_2 = float4((*(&kernelContext_6)->tw_0)[2U][cb_0], (*(&kernelContext_6)->tw_0)[2U][_S90], (*(&kernelContext_6)->tw_0)[2U][_S91], (*(&kernelContext_6)->tw_0)[2U][_S92]);
            float4 b_3 = float4((*(&kernelContext_6)->tw_0)[3U][cb_0], (*(&kernelContext_6)->tw_0)[3U][_S90], (*(&kernelContext_6)->tw_0)[3U][_S91], (*(&kernelContext_6)->tw_0)[3U][_S92]);
            float4 b_4 = float4((*(&kernelContext_6)->tw_0)[4U][cb_0], (*(&kernelContext_6)->tw_0)[4U][_S90], (*(&kernelContext_6)->tw_0)[4U][_S91], (*(&kernelContext_6)->tw_0)[4U][_S92]);
            float4 b_5 = float4((*(&kernelContext_6)->tw_0)[5U][cb_0], (*(&kernelContext_6)->tw_0)[5U][_S90], (*(&kernelContext_6)->tw_0)[5U][_S91], (*(&kernelContext_6)->tw_0)[5U][_S92]);
            float4 b_6 = float4((*(&kernelContext_6)->tw_0)[6U][cb_0], (*(&kernelContext_6)->tw_0)[6U][_S90], (*(&kernelContext_6)->tw_0)[6U][_S91], (*(&kernelContext_6)->tw_0)[6U][_S92]);
            float4 b_7 = float4((*(&kernelContext_6)->tw_0)[7U][cb_0], (*(&kernelContext_6)->tw_0)[7U][_S90], (*(&kernelContext_6)->tw_0)[7U][_S91], (*(&kernelContext_6)->tw_0)[7U][_S92]);
            float4 r0_2 = r0_0 + float4((*(&kernelContext_6)->tx_0)[0U][_S71])  * b_0 + float4((*(&kernelContext_6)->tx_0)[1U][_S71])  * b_1 + float4((*(&kernelContext_6)->tx_0)[2U][_S71])  * b_2 + float4((*(&kernelContext_6)->tx_0)[3U][_S71])  * b_3 + float4((*(&kernelContext_6)->tx_0)[4U][_S71])  * b_4 + float4((*(&kernelContext_6)->tx_0)[5U][_S71])  * b_5 + float4((*(&kernelContext_6)->tx_0)[6U][_S71])  * b_6 + float4((*(&kernelContext_6)->tx_0)[7U][_S71])  * b_7;
            float4 r1_2 = r1_0 + float4((*(&kernelContext_6)->tx_0)[0U][_S93])  * b_0 + float4((*(&kernelContext_6)->tx_0)[1U][_S93])  * b_1 + float4((*(&kernelContext_6)->tx_0)[2U][_S93])  * b_2 + float4((*(&kernelContext_6)->tx_0)[3U][_S93])  * b_3 + float4((*(&kernelContext_6)->tx_0)[4U][_S93])  * b_4 + float4((*(&kernelContext_6)->tx_0)[5U][_S93])  * b_5 + float4((*(&kernelContext_6)->tx_0)[6U][_S93])  * b_6 + float4((*(&kernelContext_6)->tx_0)[7U][_S93])  * b_7;
            float4 r2_2 = r2_0 + float4((*(&kernelContext_6)->tx_0)[0U][_S94])  * b_0 + float4((*(&kernelContext_6)->tx_0)[1U][_S94])  * b_1 + float4((*(&kernelContext_6)->tx_0)[2U][_S94])  * b_2 + float4((*(&kernelContext_6)->tx_0)[3U][_S94])  * b_3 + float4((*(&kernelContext_6)->tx_0)[4U][_S94])  * b_4 + float4((*(&kernelContext_6)->tx_0)[5U][_S94])  * b_5 + float4((*(&kernelContext_6)->tx_0)[6U][_S94])  * b_6 + float4((*(&kernelContext_6)->tx_0)[7U][_S94])  * b_7;
            float4 r3_2 = r3_0 + float4((*(&kernelContext_6)->tx_0)[0U][_S95])  * b_0 + float4((*(&kernelContext_6)->tx_0)[1U][_S95])  * b_1 + float4((*(&kernelContext_6)->tx_0)[2U][_S95])  * b_2 + float4((*(&kernelContext_6)->tx_0)[3U][_S95])  * b_3 + float4((*(&kernelContext_6)->tx_0)[4U][_S95])  * b_4 + float4((*(&kernelContext_6)->tx_0)[5U][_S95])  * b_5 + float4((*(&kernelContext_6)->tx_0)[6U][_S95])  * b_6 + float4((*(&kernelContext_6)->tx_0)[7U][_S95])  * b_7;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            k0_0 = k0_0 + 8U;
            r0_0 = r0_2;
            r1_0 = r1_2;
            r2_0 = r2_2;
            r3_0 = r3_2;
        }
        koff_0 = koff_0 + _S86;
        r0_1 = r0_0;
        r1_1 = r1_0;
        r2_1 = r2_0;
        r3_1 = r3_0;
        s_4 = s_4 + 1U;
    }
    s_4 = 0U;
    for(;;)
    {
        if(s_4 < _S83)
        {
        }
        else
        {
            break;
        }
        uint o_5 = 8U + 5U * s_4;
        uint _S96 = P_0(o_5, &kernelContext_6);
        uint _S97 = P_0(o_5 + 1U, &kernelContext_6);
        uint _S98 = P_0(o_5 + 3U, &kernelContext_6);
        if(_S98 < 4U)
        {
            s_4 = s_4 + 1U;
            continue;
        }
        uint _S99 = _S75 + _S71;
        if(_S99 < _S76)
        {
            if(_S98 == 4U)
            {
                k0_0 = (&kernelContext_6)->table_0[_S97 + _S99];
            }
            else
            {
                k0_0 = _S99;
            }
            uint _S100 = _S96 + k0_0 * _S77 + cb_0;
            float4 pv_0 = float4(*((&kernelContext_6)->arena_0+_S100), *((&kernelContext_6)->arena_0+(_S100 + 1U)), *((&kernelContext_6)->arena_0+(_S100 + 2U)), *((&kernelContext_6)->arena_0+(_S100 + 3U)));
            r0_0 = r0_1 + pv_0;
        }
        else
        {
            r0_0 = r0_1;
        }
        uint row_4 = _S99 + 1U;
        if(row_4 < _S76)
        {
            if(_S98 == 4U)
            {
                k0_0 = (&kernelContext_6)->table_0[_S97 + row_4];
            }
            else
            {
                k0_0 = row_4;
            }
            uint _S101 = _S96 + k0_0 * _S77 + cb_0;
            float4 pv_1 = float4(*((&kernelContext_6)->arena_0+_S101), *((&kernelContext_6)->arena_0+(_S101 + 1U)), *((&kernelContext_6)->arena_0+(_S101 + 2U)), *((&kernelContext_6)->arena_0+(_S101 + 3U)));
            r1_0 = r1_1 + pv_1;
        }
        else
        {
            r1_0 = r1_1;
        }
        uint row_5 = _S99 + 2U;
        if(row_5 < _S76)
        {
            if(_S98 == 4U)
            {
                k0_0 = (&kernelContext_6)->table_0[_S97 + row_5];
            }
            else
            {
                k0_0 = row_5;
            }
            uint _S102 = _S96 + k0_0 * _S77 + cb_0;
            float4 pv_2 = float4(*((&kernelContext_6)->arena_0+_S102), *((&kernelContext_6)->arena_0+(_S102 + 1U)), *((&kernelContext_6)->arena_0+(_S102 + 2U)), *((&kernelContext_6)->arena_0+(_S102 + 3U)));
            r2_0 = r2_1 + pv_2;
        }
        else
        {
            r2_0 = r2_1;
        }
        uint row_6 = _S99 + 3U;
        if(row_6 < _S76)
        {
            if(_S98 == 4U)
            {
                k0_0 = (&kernelContext_6)->table_0[_S97 + row_6];
            }
            else
            {
                k0_0 = row_6;
            }
            uint _S103 = _S96 + k0_0 * _S77 + cb_0;
            float4 pv_3 = float4(*((&kernelContext_6)->arena_0+_S103), *((&kernelContext_6)->arena_0+(_S103 + 1U)), *((&kernelContext_6)->arena_0+(_S103 + 2U)), *((&kernelContext_6)->arena_0+(_S103 + 3U)));
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
    uint _S104 = P_0(44U, &kernelContext_6);
    float v_8;
    if(_S104 != 4294967295U)
    {
        uint _S105 = _S71 * _S69;
        uint _S106 = _S105 + cb_0;
        (*(&kernelContext_6)->th_0)[_S106] = silu_0(r0_1[0U]);
        uint _S107 = _S71 + 1U;
        uint _S108 = _S107 * _S69;
        uint _S109 = _S108 + cb_0;
        (*(&kernelContext_6)->th_0)[_S109] = silu_0(r1_1[0U]);
        uint _S110 = _S71 + 2U;
        uint _S111 = _S110 * _S69;
        uint _S112 = _S111 + cb_0;
        (*(&kernelContext_6)->th_0)[_S112] = silu_0(r2_1[0U]);
        uint _S113 = _S71 + 3U;
        uint _S114 = _S113 * _S69;
        uint _S115 = _S114 + cb_0;
        (*(&kernelContext_6)->th_0)[_S115] = silu_0(r3_1[0U]);
        (*(&kernelContext_6)->th_0)[_S106 + 1U] = silu_0(r0_1[1U]);
        (*(&kernelContext_6)->th_0)[_S109 + 1U] = silu_0(r1_1[1U]);
        (*(&kernelContext_6)->th_0)[_S112 + 1U] = silu_0(r2_1[1U]);
        (*(&kernelContext_6)->th_0)[_S115 + 1U] = silu_0(r3_1[1U]);
        (*(&kernelContext_6)->th_0)[_S106 + 2U] = silu_0(r0_1[2U]);
        (*(&kernelContext_6)->th_0)[_S109 + 2U] = silu_0(r1_1[2U]);
        (*(&kernelContext_6)->th_0)[_S112 + 2U] = silu_0(r2_1[2U]);
        (*(&kernelContext_6)->th_0)[_S115 + 2U] = silu_0(r3_1[2U]);
        (*(&kernelContext_6)->th_0)[_S106 + 3U] = silu_0(r0_1[3U]);
        (*(&kernelContext_6)->th_0)[_S109 + 3U] = silu_0(r1_1[3U]);
        (*(&kernelContext_6)->th_0)[_S112 + 3U] = silu_0(r2_1[3U]);
        (*(&kernelContext_6)->th_0)[_S115 + 3U] = silu_0(r3_1[3U]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        thread float4 b2_0 = _S80;
        if(cb_0 < _S78)
        {
            uint _S116 = P_0(53U, &kernelContext_6);
            m_0 = (&kernelContext_6)->wts_0[_S116 + cb_0];
        }
        else
        {
            m_0 = 0.0f;
        }
        b2_0.x = m_0;
        uint _S117 = cb_0 + 1U;
        if(_S117 < _S78)
        {
            uint _S118 = P_0(53U, &kernelContext_6);
            m_0 = (&kernelContext_6)->wts_0[_S118 + cb_0 + 1U];
        }
        else
        {
            m_0 = 0.0f;
        }
        b2_0.y = m_0;
        uint _S119 = cb_0 + 2U;
        if(_S119 < _S78)
        {
            uint _S120 = P_0(53U, &kernelContext_6);
            m_0 = (&kernelContext_6)->wts_0[_S120 + cb_0 + 2U];
        }
        else
        {
            m_0 = 0.0f;
        }
        b2_0.z = m_0;
        uint _S121 = cb_0 + 3U;
        if(_S121 < _S78)
        {
            uint _S122 = P_0(53U, &kernelContext_6);
            m_0 = (&kernelContext_6)->wts_0[_S122 + cb_0 + 3U];
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
            if(k0_0 < _S77)
            {
            }
            else
            {
                break;
            }
            q_0 = t_1;
            for(;;)
            {
                if(q_0 < (8U * _S69))
                {
                }
                else
                {
                    break;
                }
                uint kk_2 = q_0 / _S69;
                uint c_4 = q_0 - kk_2 * _S69;
                if(c_4 < _S78)
                {
                    m_0 = (&kernelContext_6)->wts_0[_S104 + (k0_0 + kk_2) * _S78 + c_4];
                }
                else
                {
                    m_0 = 0.0f;
                }
                (*(&kernelContext_6)->tw_0)[kk_2][c_4] = m_0;
                q_0 = q_0 + 128U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float4 b_8 = float4((*(&kernelContext_6)->tw_0)[0U][cb_0], (*(&kernelContext_6)->tw_0)[0U][_S117], (*(&kernelContext_6)->tw_0)[0U][_S119], (*(&kernelContext_6)->tw_0)[0U][_S121]);
            uint _S123 = _S105 + k0_0;
            float4 _S124 = y0_0 + float4((*(&kernelContext_6)->th_0)[_S123])  * b_8;
            y0_0 = _S124;
            uint _S125 = _S108 + k0_0;
            float4 _S126 = y1_0 + float4((*(&kernelContext_6)->th_0)[_S125])  * b_8;
            y1_0 = _S126;
            uint _S127 = _S111 + k0_0;
            float4 _S128 = y2_0 + float4((*(&kernelContext_6)->th_0)[_S127])  * b_8;
            y2_0 = _S128;
            uint _S129 = _S114 + k0_0;
            float4 _S130 = y3_0 + float4((*(&kernelContext_6)->th_0)[_S129])  * b_8;
            y3_0 = _S130;
            float4 b_9 = float4((*(&kernelContext_6)->tw_0)[1U][cb_0], (*(&kernelContext_6)->tw_0)[1U][_S117], (*(&kernelContext_6)->tw_0)[1U][_S119], (*(&kernelContext_6)->tw_0)[1U][_S121]);
            float4 _S131 = _S124 + float4((*(&kernelContext_6)->th_0)[_S123 + 1U])  * b_9;
            y0_0 = _S131;
            float4 _S132 = _S126 + float4((*(&kernelContext_6)->th_0)[_S125 + 1U])  * b_9;
            y1_0 = _S132;
            float4 _S133 = _S128 + float4((*(&kernelContext_6)->th_0)[_S127 + 1U])  * b_9;
            y2_0 = _S133;
            float4 _S134 = _S130 + float4((*(&kernelContext_6)->th_0)[_S129 + 1U])  * b_9;
            y3_0 = _S134;
            float4 b_10 = float4((*(&kernelContext_6)->tw_0)[2U][cb_0], (*(&kernelContext_6)->tw_0)[2U][_S117], (*(&kernelContext_6)->tw_0)[2U][_S119], (*(&kernelContext_6)->tw_0)[2U][_S121]);
            float4 _S135 = _S131 + float4((*(&kernelContext_6)->th_0)[_S123 + 2U])  * b_10;
            y0_0 = _S135;
            float4 _S136 = _S132 + float4((*(&kernelContext_6)->th_0)[_S125 + 2U])  * b_10;
            y1_0 = _S136;
            float4 _S137 = _S133 + float4((*(&kernelContext_6)->th_0)[_S127 + 2U])  * b_10;
            y2_0 = _S137;
            float4 _S138 = _S134 + float4((*(&kernelContext_6)->th_0)[_S129 + 2U])  * b_10;
            y3_0 = _S138;
            float4 b_11 = float4((*(&kernelContext_6)->tw_0)[3U][cb_0], (*(&kernelContext_6)->tw_0)[3U][_S117], (*(&kernelContext_6)->tw_0)[3U][_S119], (*(&kernelContext_6)->tw_0)[3U][_S121]);
            float4 _S139 = _S135 + float4((*(&kernelContext_6)->th_0)[_S123 + 3U])  * b_11;
            y0_0 = _S139;
            float4 _S140 = _S136 + float4((*(&kernelContext_6)->th_0)[_S125 + 3U])  * b_11;
            y1_0 = _S140;
            float4 _S141 = _S137 + float4((*(&kernelContext_6)->th_0)[_S127 + 3U])  * b_11;
            y2_0 = _S141;
            float4 _S142 = _S138 + float4((*(&kernelContext_6)->th_0)[_S129 + 3U])  * b_11;
            y3_0 = _S142;
            float4 b_12 = float4((*(&kernelContext_6)->tw_0)[4U][cb_0], (*(&kernelContext_6)->tw_0)[4U][_S117], (*(&kernelContext_6)->tw_0)[4U][_S119], (*(&kernelContext_6)->tw_0)[4U][_S121]);
            float4 _S143 = _S139 + float4((*(&kernelContext_6)->th_0)[_S123 + 4U])  * b_12;
            y0_0 = _S143;
            float4 _S144 = _S140 + float4((*(&kernelContext_6)->th_0)[_S125 + 4U])  * b_12;
            y1_0 = _S144;
            float4 _S145 = _S141 + float4((*(&kernelContext_6)->th_0)[_S127 + 4U])  * b_12;
            y2_0 = _S145;
            float4 _S146 = _S142 + float4((*(&kernelContext_6)->th_0)[_S129 + 4U])  * b_12;
            y3_0 = _S146;
            float4 b_13 = float4((*(&kernelContext_6)->tw_0)[5U][cb_0], (*(&kernelContext_6)->tw_0)[5U][_S117], (*(&kernelContext_6)->tw_0)[5U][_S119], (*(&kernelContext_6)->tw_0)[5U][_S121]);
            float4 _S147 = _S143 + float4((*(&kernelContext_6)->th_0)[_S123 + 5U])  * b_13;
            y0_0 = _S147;
            float4 _S148 = _S144 + float4((*(&kernelContext_6)->th_0)[_S125 + 5U])  * b_13;
            y1_0 = _S148;
            float4 _S149 = _S145 + float4((*(&kernelContext_6)->th_0)[_S127 + 5U])  * b_13;
            y2_0 = _S149;
            float4 _S150 = _S146 + float4((*(&kernelContext_6)->th_0)[_S129 + 5U])  * b_13;
            y3_0 = _S150;
            float4 b_14 = float4((*(&kernelContext_6)->tw_0)[6U][cb_0], (*(&kernelContext_6)->tw_0)[6U][_S117], (*(&kernelContext_6)->tw_0)[6U][_S119], (*(&kernelContext_6)->tw_0)[6U][_S121]);
            float4 _S151 = _S147 + float4((*(&kernelContext_6)->th_0)[_S123 + 6U])  * b_14;
            y0_0 = _S151;
            float4 _S152 = _S148 + float4((*(&kernelContext_6)->th_0)[_S125 + 6U])  * b_14;
            y1_0 = _S152;
            float4 _S153 = _S149 + float4((*(&kernelContext_6)->th_0)[_S127 + 6U])  * b_14;
            y2_0 = _S153;
            float4 _S154 = _S150 + float4((*(&kernelContext_6)->th_0)[_S129 + 6U])  * b_14;
            y3_0 = _S154;
            float4 b_15 = float4((*(&kernelContext_6)->tw_0)[7U][cb_0], (*(&kernelContext_6)->tw_0)[7U][_S117], (*(&kernelContext_6)->tw_0)[7U][_S119], (*(&kernelContext_6)->tw_0)[7U][_S121]);
            y0_0 = _S151 + float4((*(&kernelContext_6)->th_0)[_S123 + 7U])  * b_15;
            y1_0 = _S152 + float4((*(&kernelContext_6)->th_0)[_S125 + 7U])  * b_15;
            y2_0 = _S153 + float4((*(&kernelContext_6)->th_0)[_S127 + 7U])  * b_15;
            y3_0 = _S154 + float4((*(&kernelContext_6)->th_0)[_S129 + 7U])  * b_15;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            k0_0 = k0_0 + 8U;
        }
        uint _S155 = P_0(45U, &kernelContext_6);
        if(_S155 != 4294967295U)
        {
            (*(&kernelContext_6)->th_0)[_S106] = y0_0.x;
            (*(&kernelContext_6)->th_0)[_S109] = y1_0.x;
            (*(&kernelContext_6)->th_0)[_S112] = y2_0.x;
            (*(&kernelContext_6)->th_0)[_S115] = y3_0.x;
            (*(&kernelContext_6)->th_0)[_S106 + 1U] = y0_0.y;
            (*(&kernelContext_6)->th_0)[_S109 + 1U] = y1_0.y;
            (*(&kernelContext_6)->th_0)[_S112 + 1U] = y2_0.y;
            (*(&kernelContext_6)->th_0)[_S115 + 1U] = y3_0.y;
            (*(&kernelContext_6)->th_0)[_S106 + 2U] = y0_0.z;
            (*(&kernelContext_6)->th_0)[_S109 + 2U] = y1_0.z;
            (*(&kernelContext_6)->th_0)[_S112 + 2U] = y2_0.z;
            (*(&kernelContext_6)->th_0)[_S115 + 2U] = y3_0.z;
            (*(&kernelContext_6)->th_0)[_S106 + 3U] = y0_0.w;
            (*(&kernelContext_6)->th_0)[_S109 + 3U] = y1_0.w;
            (*(&kernelContext_6)->th_0)[_S112 + 3U] = y2_0.w;
            (*(&kernelContext_6)->th_0)[_S115 + 3U] = y3_0.w;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(t_1 < tr_0)
            {
                q_0 = 0U;
                m_0 = 0.0f;
                for(;;)
                {
                    if(q_0 < _S78)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float m_1 = m_0 + (*(&kernelContext_6)->th_0)[t_1 * _S69 + q_0];
                    q_0 = q_0 + 1U;
                    m_0 = m_1;
                }
                float _S156 = float(_S78);
                float m_2 = m_0 / _S156;
                q_0 = 0U;
                v_8 = 0.0f;
                for(;;)
                {
                    if(q_0 < _S78)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float d_2 = (*(&kernelContext_6)->th_0)[t_1 * _S69 + q_0] - m_2;
                    float v_9 = v_8 + d_2 * d_2;
                    q_0 = q_0 + 1U;
                    v_8 = v_9;
                }
                (*(&kernelContext_6)->tstat_0)[int(0)][t_1] = m_2;
                (*(&kernelContext_6)->tstat_0)[int(1)][t_1] = 1.0f / sqrt(v_8 / _S156 + 0.00000999999974738f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint _S157 = _S155 + cb_0;
            uint _S158 = _S155 + _S78;
            uint _S159 = _S158 + cb_0;
            y0_0.x = (y0_0.x - (*(&kernelContext_6)->tstat_0)[int(0)][_S71]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S71] * (&kernelContext_6)->wts_0[_S157] + (&kernelContext_6)->wts_0[_S159];
            y1_0.x = (y1_0.x - (*(&kernelContext_6)->tstat_0)[int(0)][_S107]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S107] * (&kernelContext_6)->wts_0[_S157] + (&kernelContext_6)->wts_0[_S159];
            y2_0.x = (y2_0.x - (*(&kernelContext_6)->tstat_0)[int(0)][_S110]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S110] * (&kernelContext_6)->wts_0[_S157] + (&kernelContext_6)->wts_0[_S159];
            y3_0.x = (y3_0.x - (*(&kernelContext_6)->tstat_0)[int(0)][_S113]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S113] * (&kernelContext_6)->wts_0[_S157] + (&kernelContext_6)->wts_0[_S159];
            uint _S160 = _S155 + _S117;
            uint _S161 = _S158 + _S117;
            y0_0.y = (y0_0.y - (*(&kernelContext_6)->tstat_0)[int(0)][_S71]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S71] * (&kernelContext_6)->wts_0[_S160] + (&kernelContext_6)->wts_0[_S161];
            y1_0.y = (y1_0.y - (*(&kernelContext_6)->tstat_0)[int(0)][_S107]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S107] * (&kernelContext_6)->wts_0[_S160] + (&kernelContext_6)->wts_0[_S161];
            y2_0.y = (y2_0.y - (*(&kernelContext_6)->tstat_0)[int(0)][_S110]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S110] * (&kernelContext_6)->wts_0[_S160] + (&kernelContext_6)->wts_0[_S161];
            y3_0.y = (y3_0.y - (*(&kernelContext_6)->tstat_0)[int(0)][_S113]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S113] * (&kernelContext_6)->wts_0[_S160] + (&kernelContext_6)->wts_0[_S161];
            uint _S162 = _S155 + _S119;
            uint _S163 = _S158 + _S119;
            y0_0.z = (y0_0.z - (*(&kernelContext_6)->tstat_0)[int(0)][_S71]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S71] * (&kernelContext_6)->wts_0[_S162] + (&kernelContext_6)->wts_0[_S163];
            y1_0.z = (y1_0.z - (*(&kernelContext_6)->tstat_0)[int(0)][_S107]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S107] * (&kernelContext_6)->wts_0[_S162] + (&kernelContext_6)->wts_0[_S163];
            y2_0.z = (y2_0.z - (*(&kernelContext_6)->tstat_0)[int(0)][_S110]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S110] * (&kernelContext_6)->wts_0[_S162] + (&kernelContext_6)->wts_0[_S163];
            y3_0.z = (y3_0.z - (*(&kernelContext_6)->tstat_0)[int(0)][_S113]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S113] * (&kernelContext_6)->wts_0[_S162] + (&kernelContext_6)->wts_0[_S163];
            uint _S164 = _S155 + _S121;
            uint _S165 = _S158 + _S121;
            y0_0.w = (y0_0.w - (*(&kernelContext_6)->tstat_0)[int(0)][_S71]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S71] * (&kernelContext_6)->wts_0[_S164] + (&kernelContext_6)->wts_0[_S165];
            y1_0.w = (y1_0.w - (*(&kernelContext_6)->tstat_0)[int(0)][_S107]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S107] * (&kernelContext_6)->wts_0[_S164] + (&kernelContext_6)->wts_0[_S165];
            y2_0.w = (y2_0.w - (*(&kernelContext_6)->tstat_0)[int(0)][_S110]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S110] * (&kernelContext_6)->wts_0[_S164] + (&kernelContext_6)->wts_0[_S165];
            y3_0.w = (y3_0.w - (*(&kernelContext_6)->tstat_0)[int(0)][_S113]) * (*(&kernelContext_6)->tstat_0)[int(1)][_S113] * (&kernelContext_6)->wts_0[_S164] + (&kernelContext_6)->wts_0[_S165];
        }
    }
    uint _S166 = P_0(51U, &kernelContext_6);
    float _S167 = (as_type<float>((_S166)));
    uint _S168 = P_0(48U, &kernelContext_6);
    uint _S169 = _S75 + _S71;
    float4 _S170 = y0_0;
    if(_S169 < _S76)
    {
        if(cb_0 < _S78)
        {
            float v_10 = _S167 * _S170[0U];
            if(_S168 != 4294967295U)
            {
                uint _S171 = P_0(49U, &kernelContext_6);
                v_8 = v_10 + *((&kernelContext_6)->arena_0+(_S168 + _S169 * _S171 + cb_0));
            }
            else
            {
                v_8 = v_10;
            }
            uint _S172 = P_0(46U, &kernelContext_6);
            uint _S173 = P_0(47U, &kernelContext_6);
            uint o_6 = _S172 + _S169 * _S173 + cb_0;
            uint _S174 = P_0(50U, &kernelContext_6);
            if((_S174 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_6) = *((&kernelContext_6)->arena_0+o_6) + v_8;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_6) = v_8;
            }
        }
        uint c_5 = cb_0 + 1U;
        if(c_5 < _S78)
        {
            float v_11 = _S167 * _S170[1U];
            if(_S168 != 4294967295U)
            {
                uint _S175 = P_0(49U, &kernelContext_6);
                v_8 = v_11 + *((&kernelContext_6)->arena_0+(_S168 + _S169 * _S175 + c_5));
            }
            else
            {
                v_8 = v_11;
            }
            uint _S176 = P_0(46U, &kernelContext_6);
            uint _S177 = P_0(47U, &kernelContext_6);
            uint o_7 = _S176 + _S169 * _S177 + c_5;
            uint _S178 = P_0(50U, &kernelContext_6);
            if((_S178 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_7) = *((&kernelContext_6)->arena_0+o_7) + v_8;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_7) = v_8;
            }
        }
        uint c_6 = cb_0 + 2U;
        if(c_6 < _S78)
        {
            float v_12 = _S167 * _S170[2U];
            if(_S168 != 4294967295U)
            {
                uint _S179 = P_0(49U, &kernelContext_6);
                v_8 = v_12 + *((&kernelContext_6)->arena_0+(_S168 + _S169 * _S179 + c_6));
            }
            else
            {
                v_8 = v_12;
            }
            uint _S180 = P_0(46U, &kernelContext_6);
            uint _S181 = P_0(47U, &kernelContext_6);
            uint o_8 = _S180 + _S169 * _S181 + c_6;
            uint _S182 = P_0(50U, &kernelContext_6);
            if((_S182 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_8) = *((&kernelContext_6)->arena_0+o_8) + v_8;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_8) = v_8;
            }
        }
        uint c_7 = cb_0 + 3U;
        if(c_7 < _S78)
        {
            float v_13 = _S167 * _S170[3U];
            if(_S168 != 4294967295U)
            {
                uint _S183 = P_0(49U, &kernelContext_6);
                v_8 = v_13 + *((&kernelContext_6)->arena_0+(_S168 + _S169 * _S183 + c_7));
            }
            else
            {
                v_8 = v_13;
            }
            uint _S184 = P_0(46U, &kernelContext_6);
            uint _S185 = P_0(47U, &kernelContext_6);
            uint o_9 = _S184 + _S169 * _S185 + c_7;
            uint _S186 = P_0(50U, &kernelContext_6);
            if((_S186 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_9) = *((&kernelContext_6)->arena_0+o_9) + v_8;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_9) = v_8;
            }
        }
    }
    uint row_7 = _S169 + 1U;
    float4 _S187 = y1_0;
    if(row_7 < _S76)
    {
        if(cb_0 < _S78)
        {
            float v_14 = _S167 * _S187[0U];
            if(_S168 != 4294967295U)
            {
                uint _S188 = P_0(49U, &kernelContext_6);
                v_8 = v_14 + *((&kernelContext_6)->arena_0+(_S168 + row_7 * _S188 + cb_0));
            }
            else
            {
                v_8 = v_14;
            }
            uint _S189 = P_0(46U, &kernelContext_6);
            uint _S190 = P_0(47U, &kernelContext_6);
            uint o_10 = _S189 + row_7 * _S190 + cb_0;
            uint _S191 = P_0(50U, &kernelContext_6);
            if((_S191 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_10) = *((&kernelContext_6)->arena_0+o_10) + v_8;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_10) = v_8;
            }
        }
        uint c_8 = cb_0 + 1U;
        if(c_8 < _S78)
        {
            float v_15 = _S167 * _S187[1U];
            if(_S168 != 4294967295U)
            {
                uint _S192 = P_0(49U, &kernelContext_6);
                v_8 = v_15 + *((&kernelContext_6)->arena_0+(_S168 + row_7 * _S192 + c_8));
            }
            else
            {
                v_8 = v_15;
            }
            uint _S193 = P_0(46U, &kernelContext_6);
            uint _S194 = P_0(47U, &kernelContext_6);
            uint o_11 = _S193 + row_7 * _S194 + c_8;
            uint _S195 = P_0(50U, &kernelContext_6);
            if((_S195 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_11) = *((&kernelContext_6)->arena_0+o_11) + v_8;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_11) = v_8;
            }
        }
        uint c_9 = cb_0 + 2U;
        if(c_9 < _S78)
        {
            float v_16 = _S167 * _S187[2U];
            if(_S168 != 4294967295U)
            {
                uint _S196 = P_0(49U, &kernelContext_6);
                v_8 = v_16 + *((&kernelContext_6)->arena_0+(_S168 + row_7 * _S196 + c_9));
            }
            else
            {
                v_8 = v_16;
            }
            uint _S197 = P_0(46U, &kernelContext_6);
            uint _S198 = P_0(47U, &kernelContext_6);
            uint o_12 = _S197 + row_7 * _S198 + c_9;
            uint _S199 = P_0(50U, &kernelContext_6);
            if((_S199 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_12) = *((&kernelContext_6)->arena_0+o_12) + v_8;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_12) = v_8;
            }
        }
        uint c_10 = cb_0 + 3U;
        if(c_10 < _S78)
        {
            float v_17 = _S167 * _S187[3U];
            if(_S168 != 4294967295U)
            {
                uint _S200 = P_0(49U, &kernelContext_6);
                v_8 = v_17 + *((&kernelContext_6)->arena_0+(_S168 + row_7 * _S200 + c_10));
            }
            else
            {
                v_8 = v_17;
            }
            uint _S201 = P_0(46U, &kernelContext_6);
            uint _S202 = P_0(47U, &kernelContext_6);
            uint o_13 = _S201 + row_7 * _S202 + c_10;
            uint _S203 = P_0(50U, &kernelContext_6);
            if((_S203 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_13) = *((&kernelContext_6)->arena_0+o_13) + v_8;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_13) = v_8;
            }
        }
    }
    uint row_8 = _S169 + 2U;
    float4 _S204 = y2_0;
    if(row_8 < _S76)
    {
        if(cb_0 < _S78)
        {
            float v_18 = _S167 * _S204[0U];
            if(_S168 != 4294967295U)
            {
                uint _S205 = P_0(49U, &kernelContext_6);
                v_8 = v_18 + *((&kernelContext_6)->arena_0+(_S168 + row_8 * _S205 + cb_0));
            }
            else
            {
                v_8 = v_18;
            }
            uint _S206 = P_0(46U, &kernelContext_6);
            uint _S207 = P_0(47U, &kernelContext_6);
            uint o_14 = _S206 + row_8 * _S207 + cb_0;
            uint _S208 = P_0(50U, &kernelContext_6);
            if((_S208 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_14) = *((&kernelContext_6)->arena_0+o_14) + v_8;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_14) = v_8;
            }
        }
        uint c_11 = cb_0 + 1U;
        if(c_11 < _S78)
        {
            float v_19 = _S167 * _S204[1U];
            if(_S168 != 4294967295U)
            {
                uint _S209 = P_0(49U, &kernelContext_6);
                v_8 = v_19 + *((&kernelContext_6)->arena_0+(_S168 + row_8 * _S209 + c_11));
            }
            else
            {
                v_8 = v_19;
            }
            uint _S210 = P_0(46U, &kernelContext_6);
            uint _S211 = P_0(47U, &kernelContext_6);
            uint o_15 = _S210 + row_8 * _S211 + c_11;
            uint _S212 = P_0(50U, &kernelContext_6);
            if((_S212 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_15) = *((&kernelContext_6)->arena_0+o_15) + v_8;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_15) = v_8;
            }
        }
        uint c_12 = cb_0 + 2U;
        if(c_12 < _S78)
        {
            float v_20 = _S167 * _S204[2U];
            if(_S168 != 4294967295U)
            {
                uint _S213 = P_0(49U, &kernelContext_6);
                v_8 = v_20 + *((&kernelContext_6)->arena_0+(_S168 + row_8 * _S213 + c_12));
            }
            else
            {
                v_8 = v_20;
            }
            uint _S214 = P_0(46U, &kernelContext_6);
            uint _S215 = P_0(47U, &kernelContext_6);
            uint o_16 = _S214 + row_8 * _S215 + c_12;
            uint _S216 = P_0(50U, &kernelContext_6);
            if((_S216 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_16) = *((&kernelContext_6)->arena_0+o_16) + v_8;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_16) = v_8;
            }
        }
        uint c_13 = cb_0 + 3U;
        if(c_13 < _S78)
        {
            float v_21 = _S167 * _S204[3U];
            if(_S168 != 4294967295U)
            {
                uint _S217 = P_0(49U, &kernelContext_6);
                v_8 = v_21 + *((&kernelContext_6)->arena_0+(_S168 + row_8 * _S217 + c_13));
            }
            else
            {
                v_8 = v_21;
            }
            uint _S218 = P_0(46U, &kernelContext_6);
            uint _S219 = P_0(47U, &kernelContext_6);
            uint o_17 = _S218 + row_8 * _S219 + c_13;
            uint _S220 = P_0(50U, &kernelContext_6);
            if((_S220 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_17) = *((&kernelContext_6)->arena_0+o_17) + v_8;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_17) = v_8;
            }
        }
    }
    uint row_9 = _S169 + 3U;
    float4 _S221 = y3_0;
    if(row_9 < _S76)
    {
        if(cb_0 < _S78)
        {
            float v_22 = _S167 * _S221[0U];
            if(_S168 != 4294967295U)
            {
                uint _S222 = P_0(49U, &kernelContext_6);
                v_8 = v_22 + *((&kernelContext_6)->arena_0+(_S168 + row_9 * _S222 + cb_0));
            }
            else
            {
                v_8 = v_22;
            }
            uint _S223 = P_0(46U, &kernelContext_6);
            uint _S224 = P_0(47U, &kernelContext_6);
            uint o_18 = _S223 + row_9 * _S224 + cb_0;
            uint _S225 = P_0(50U, &kernelContext_6);
            if((_S225 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_18) = *((&kernelContext_6)->arena_0+o_18) + v_8;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_18) = v_8;
            }
        }
        uint c_14 = cb_0 + 1U;
        if(c_14 < _S78)
        {
            float v_23 = _S167 * _S221[1U];
            if(_S168 != 4294967295U)
            {
                uint _S226 = P_0(49U, &kernelContext_6);
                v_8 = v_23 + *((&kernelContext_6)->arena_0+(_S168 + row_9 * _S226 + c_14));
            }
            else
            {
                v_8 = v_23;
            }
            uint _S227 = P_0(46U, &kernelContext_6);
            uint _S228 = P_0(47U, &kernelContext_6);
            uint o_19 = _S227 + row_9 * _S228 + c_14;
            uint _S229 = P_0(50U, &kernelContext_6);
            if((_S229 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_19) = *((&kernelContext_6)->arena_0+o_19) + v_8;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_19) = v_8;
            }
        }
        uint c_15 = cb_0 + 2U;
        if(c_15 < _S78)
        {
            float v_24 = _S167 * _S221[2U];
            if(_S168 != 4294967295U)
            {
                uint _S230 = P_0(49U, &kernelContext_6);
                v_8 = v_24 + *((&kernelContext_6)->arena_0+(_S168 + row_9 * _S230 + c_15));
            }
            else
            {
                v_8 = v_24;
            }
            uint _S231 = P_0(46U, &kernelContext_6);
            uint _S232 = P_0(47U, &kernelContext_6);
            uint o_20 = _S231 + row_9 * _S232 + c_15;
            uint _S233 = P_0(50U, &kernelContext_6);
            if((_S233 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_20) = *((&kernelContext_6)->arena_0+o_20) + v_8;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_20) = v_8;
            }
        }
        uint c_16 = cb_0 + 3U;
        if(c_16 < _S78)
        {
            float v_25 = _S167 * _S221[3U];
            if(_S168 != 4294967295U)
            {
                uint _S234 = P_0(49U, &kernelContext_6);
                v_8 = v_25 + *((&kernelContext_6)->arena_0+(_S168 + row_9 * _S234 + c_16));
            }
            else
            {
                v_8 = v_25;
            }
            uint _S235 = P_0(46U, &kernelContext_6);
            uint _S236 = P_0(47U, &kernelContext_6);
            uint o_21 = _S235 + row_9 * _S236 + c_16;
            uint _S237 = P_0(50U, &kernelContext_6);
            if((_S237 & 1U) != 0U)
            {
                *((&kernelContext_6)->arena_0+o_21) = *((&kernelContext_6)->arena_0+o_21) + v_8;
            }
            else
            {
                *((&kernelContext_6)->arena_0+o_21) = v_8;
            }
        }
    }
    return;
}

