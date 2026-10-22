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

