#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct Quat_0
{
    float w_0;
    float x_0;
    float y_0;
    float z_0;
};

Quat_0 quat_of_0(float4 q_0)
{
    thread Quat_0 r_0;
    (&r_0)->x_0 = q_0.x;
    (&r_0)->y_0 = q_0.y;
    (&r_0)->z_0 = q_0.z;
    (&r_0)->w_0 = q_0.w;
    return r_0;
}

float3 rotate_0(const Quat_0 thread* q_1, float3 v_0)
{
    float3 qv_0 = float3(q_1->x_0, q_1->y_0, q_1->z_0);
    float3 t_0 = cross(qv_0, v_0) * float3(2.0f) ;
    return v_0 + t_0 * float3(q_1->w_0)  + cross(qv_0, t_0);
}

struct PlanParams_0
{
    uint chunk_count_0;
    uint island_count_0;
    uint job_count_0;
    uint impactor_count_0;
    float duration_0;
    float g_0;
    float gz_0;
    float ground_height_0;
    uint has_ground_0;
    uint reach_base_0;
    uint cp_base_0;
    uint job_base_0;
    uint count_base_0;
    uint pair_base_0;
    uint flag_base_0;
    uint owned_base_0;
};

struct ChunkStatic_natural_0
{
    packed_float4 center_0;
    packed_float4 inertia0_0;
    packed_float4 inertia1_0;
    packed_float4 inertia2_0;
    packed_float4 inv0_0;
    packed_float4 inv1_0;
    packed_float4 inv2_0;
    packed_float4 scale_0;
    packed_uint4 info_0;
    packed_uint4 load_range_0;
    packed_float4 half_0;
    packed_float4 crot0_0;
    packed_float4 crot1_0;
    packed_float4 crot2_0;
    packed_float4 cmat_0;
    packed_uint4 cinfo_0;
};

struct Island_natural_0
{
    packed_uint4 range_0;
    packed_uint4 info_1;
    packed_float4 com_0;
    packed_float4 inertia0_1;
    packed_float4 inertia1_1;
    packed_float4 inertia2_1;
    packed_float4 inv0_1;
    packed_float4 inv1_1;
    packed_float4 inv2_1;
    packed_float4 wcom_0;
    packed_float4 winv0_0;
    packed_float4 winv1_0;
    packed_float4 winv2_0;
    packed_float4 rotation_0;
    packed_float4 position_0;
    packed_float4 position_err_0;
    packed_float4 velocity_0;
    packed_float4 velocity_err_0;
    packed_float4 angular_velocity_0;
    uint done_0;
    uint coverage_0;
    uint wide_groups_0;
    uint done_pad_0;
    packed_uint4 probes_0;
    packed_float4 energy_0;
    packed_float4 rotation_err_0;
    packed_float4 momentum_0;
    packed_float4 momentum_err_0;
};

struct Impactor_natural_0
{
    packed_float4 position_1;
    packed_float4 position_err_1;
    packed_float4 velocity_1;
    packed_float4 velocity_err_1;
    packed_float4 angular_velocity_1;
    packed_float4 rotation_1;
    packed_float4 inertia0_2;
    packed_float4 inertia1_2;
    packed_float4 inertia2_2;
    packed_float4 inv0_2;
    packed_float4 inv1_2;
    packed_float4 inv2_2;
    packed_float4 shape_0;
    packed_float4 half_1;
    packed_float4 mat_0;
    packed_float4 crush_0;
    packed_float4 geom_0;
    packed_float4 rotation_err_1;
    packed_float4 ledger_0;
    packed_uint4 cand_0;
    packed_float4 momentum_1;
    packed_float4 momentum_err_1;
};

struct KernelContext_0
{
    PlanParams_0 constant* params_0;
    ChunkStatic_natural_0 device* chunks_0;
    Island_natural_0 device* islands_0;
    packed_float4 device* state_0;
    packed_float4 device* pm_0;
    uint device* pu_0;
    Impactor_natural_0 device* impactors_0;
    array<float, int(64)> threadgroup* g_spread_0;
    array<float, int(64)> threadgroup* g_rb_0;
    array<float, int(64)> threadgroup* g_v_0;
    array<uint, int(1024)> threadgroup* g_scan_0;
};

[[kernel]] void plan_motion(uint3 id_0 [[thread_position_in_grid]], PlanParams_0 constant* params_1 [[buffer(0)]], ChunkStatic_natural_0 device* chunks_1 [[buffer(1)]], Island_natural_0 device* islands_1 [[buffer(2)]], packed_float4 device* state_1 [[buffer(3)]], packed_float4 device* pm_1 [[buffer(5)]], uint device* pu_1 [[buffer(6)]], Impactor_natural_0 device* impactors_1 [[buffer(4)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->chunks_0 = chunks_1;
    (&kernelContext_0)->islands_0 = islands_1;
    (&kernelContext_0)->state_0 = state_1;
    (&kernelContext_0)->pm_0 = pm_1;
    (&kernelContext_0)->pu_0 = pu_1;
    (&kernelContext_0)->impactors_0 = impactors_1;
    threadgroup array<float, int(64)> g_spread_1;
    (&kernelContext_0)->g_spread_0 = &g_spread_1;
    threadgroup array<float, int(64)> g_rb_1;
    (&kernelContext_0)->g_rb_0 = &g_rb_1;
    threadgroup array<float, int(64)> g_v_1;
    (&kernelContext_0)->g_v_0 = &g_v_1;
    threadgroup array<uint, int(1024)> g_scan_1;
    (&kernelContext_0)->g_scan_0 = &g_scan_1;
    uint k_0 = id_0.x;
    if(k_0 >= (params_1->chunk_count_0))
    {
        return;
    }
    ChunkStatic_natural_0 device* _S1 = (&kernelContext_0)->chunks_0+k_0;
    Island_natural_0 device* _S2 = (&kernelContext_0)->islands_0+(uint4(_S1->info_0) ).y;
    Quat_0 q_2 = quat_of_0(float4(_S2->rotation_0) );
    uint _S3 = 4U * k_0;
    float3 local_0 = (float4(_S1->center_0) ).xyz + (float4(*((&kernelContext_0)->state_0+_S3)) ).xyz;
    float3 _S4 = (float4(_S2->position_0) ).xyz + (float4(_S2->position_err_0) ).xyz;
    thread Quat_0 _S5 = q_2;
    float3 _S6 = rotate_0(&_S5, local_0);
    float3 pos_0 = _S4 + _S6;
    float3 _S7 = (float4(_S2->com_0) ).xyz;
    float3 _S8 = local_0 - _S7;
    thread Quat_0 _S9 = q_2;
    float3 _S10 = rotate_0(&_S9, _S8);
    float3 _S11 = (float4(_S2->angular_velocity_0) ).xyz;
    float3 _S12 = (float4(_S2->velocity_0) ).xyz + (float4(_S2->velocity_err_0) ).xyz + cross(_S11, _S10);
    float3 _S13 = (float4(*((&kernelContext_0)->state_0+(_S3 + 2U))) ).xyz;
    thread Quat_0 _S14 = q_2;
    float3 _S15 = rotate_0(&_S14, _S13);
    float3 v_1 = _S12 + _S15;
    float3 _S16 = (float4(*((&kernelContext_0)->state_0+(_S3 + 3U))) ).xyz;
    thread Quat_0 _S17 = q_2;
    float3 _S18 = rotate_0(&_S17, _S16);
    float3 w_1 = _S11 + _S18;
    thread Quat_0 _S19 = q_2;
    float3 _S20 = rotate_0(&_S19, _S7);
    float radius_0 = length((float4(_S1->half_0) ).xyz);
    float budget_0 = 2.0f * length(w_1) * (length(pos_0 - (_S4 + _S20)) + radius_0) * (&kernelContext_0)->params_0->duration_0 + (&kernelContext_0)->params_0->g_0 * (&kernelContext_0)->params_0->duration_0 * (&kernelContext_0)->params_0->duration_0 + 0.05000000074505806f * radius_0 + 0.00100000004749745f;
    bool falls_0 = (((uint4(_S2->info_1) ).x) & 1U) == 0U;
    uint _S21 = 3U * k_0;
    *((&kernelContext_0)->pm_0+_S21) = packed_float4(float4(pos_0, radius_0)) ;
    *((&kernelContext_0)->pm_0+(_S21 + 1U)) = packed_float4(float4(v_1, budget_0)) ;
    packed_float4 device* _S22 = (&kernelContext_0)->pm_0+(_S21 + 2U);
    float _S23;
    if(falls_0)
    {
        _S23 = 1.0f;
    }
    else
    {
        _S23 = 0.0f;
    }
    *_S22 = packed_float4(float4(w_1, _S23)) ;
    return;
}

[[kernel]] void plan_reach(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], PlanParams_0 constant* params_2 [[buffer(0)]], ChunkStatic_natural_0 device* chunks_2 [[buffer(1)]], Island_natural_0 device* islands_2 [[buffer(2)]], packed_float4 device* state_2 [[buffer(3)]], packed_float4 device* pm_2 [[buffer(5)]], uint device* pu_2 [[buffer(6)]], Impactor_natural_0 device* impactors_2 [[buffer(4)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->params_0 = params_2;
    (&kernelContext_1)->chunks_0 = chunks_2;
    (&kernelContext_1)->islands_0 = islands_2;
    (&kernelContext_1)->state_0 = state_2;
    (&kernelContext_1)->pm_0 = pm_2;
    (&kernelContext_1)->pu_0 = pu_2;
    (&kernelContext_1)->impactors_0 = impactors_2;
    threadgroup array<float, int(64)> g_spread_2;
    (&kernelContext_1)->g_spread_0 = &g_spread_2;
    threadgroup array<float, int(64)> g_rb_2;
    (&kernelContext_1)->g_rb_0 = &g_rb_2;
    threadgroup array<float, int(64)> g_v_2;
    (&kernelContext_1)->g_v_0 = &g_v_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_1)->g_scan_0 = &g_scan_2;
    uint i_0 = group_0.x;
    uint tid_0 = thread_0.x;
    if(i_0 >= (params_2->island_count_0))
    {
        return;
    }
    uint4 _S24 = uint4(((&kernelContext_1)->islands_0+i_0)->range_0) ;
    uint _S25 = _S24.y;
    uint _S26 = _S24.x;
    float3 center_1;
    if(_S25 > _S26)
    {
        center_1 = (float4(*((&kernelContext_1)->pm_0+3U * _S26)) ).xyz;
    }
    else
    {
        center_1 = float3(0.0f) ;
    }
    uint _S27 = _S26 + tid_0;
    float spread_0 = 0.0f;
    float rb_0 = 0.0f;
    float vmax_0 = 0.0f;
    uint k_1 = _S27;
    for(;;)
    {
        if(k_1 < _S25)
        {
        }
        else
        {
            break;
        }
        uint _S28 = 3U * k_1;
        float4 _S29 = float4(*((&kernelContext_1)->pm_0+_S28)) ;
        float4 _S30 = float4(*((&kernelContext_1)->pm_0+(_S28 + 1U))) ;
        float _S31 = max(spread_0, length(_S29.xyz - center_1));
        float _S32 = max(rb_0, _S29.w + _S30.w);
        float _S33 = max(vmax_0, length(_S30.xyz));
        uint k_2 = k_1 + 64U;
        spread_0 = _S31;
        rb_0 = _S32;
        vmax_0 = _S33;
        k_1 = k_2;
    }
    (*(&kernelContext_1)->g_spread_0)[tid_0] = spread_0;
    (*(&kernelContext_1)->g_rb_0)[tid_0] = rb_0;
    (*(&kernelContext_1)->g_v_0)[tid_0] = vmax_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s_0 = 32U;
    for(;;)
    {
        if(s_0 > 0U)
        {
        }
        else
        {
            break;
        }
        if(tid_0 < s_0)
        {
            uint _S34 = tid_0 + s_0;
            (*(&kernelContext_1)->g_spread_0)[tid_0] = max((*(&kernelContext_1)->g_spread_0)[tid_0], (*(&kernelContext_1)->g_spread_0)[_S34]);
            (*(&kernelContext_1)->g_rb_0)[tid_0] = max((*(&kernelContext_1)->g_rb_0)[tid_0], (*(&kernelContext_1)->g_rb_0)[_S34]);
            (*(&kernelContext_1)->g_v_0)[tid_0] = max((*(&kernelContext_1)->g_v_0)[tid_0], (*(&kernelContext_1)->g_v_0)[_S34]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    if(tid_0 == 0U)
    {
        uint _S35 = 2U * i_0;
        *((&kernelContext_1)->pm_0+((&kernelContext_1)->params_0->reach_base_0 + _S35)) = packed_float4(float4(center_1, (*(&kernelContext_1)->g_spread_0)[int(0)])) ;
        *((&kernelContext_1)->pm_0+((&kernelContext_1)->params_0->reach_base_0 + _S35 + 1U)) = packed_float4(float4((*(&kernelContext_1)->g_rb_0)[int(0)], (*(&kernelContext_1)->g_v_0)[int(0)], 0.0f, 0.0f)) ;
    }
    return;
}

bool clusters_may_touch_0(uint ia_0, uint ib_0, KernelContext_0 thread* kernelContext_2)
{
    uint _S36 = 2U * ia_0;
    float4 _S37 = float4(*(kernelContext_2->pm_0+(kernelContext_2->params_0->reach_base_0 + _S36))) ;
    float4 _S38 = float4(*(kernelContext_2->pm_0+(kernelContext_2->params_0->reach_base_0 + _S36 + 1U))) ;
    uint _S39 = 2U * ib_0;
    float4 _S40 = float4(*(kernelContext_2->pm_0+(kernelContext_2->params_0->reach_base_0 + _S39))) ;
    float4 _S41 = float4(*(kernelContext_2->pm_0+(kernelContext_2->params_0->reach_base_0 + _S39 + 1U))) ;
    float d_0 = kernelContext_2->params_0->duration_0;
    float dg_0;
    if(((((uint4((kernelContext_2->islands_0+ia_0)->info_1) ).x) & 1U) != 0U) == ((((uint4((kernelContext_2->islands_0+ib_0)->info_1) ).x) & 1U) != 0U))
    {
        dg_0 = 0.0f;
    }
    else
    {
        dg_0 = kernelContext_2->params_0->g_0;
    }
    float _S42 = _S37.w;
    float _S43 = _S40.w;
    float _S44 = 0.5f * ((_S38.y + _S41.y) * d_0);
    float near_0 = length(_S40.xyz - _S37.xyz) - _S42 - _S43 - _S44;
    float limit_0 = _S38.x + _S41.x + _S44 + 0.5f * dg_0 * d_0 * d_0;
    return near_0 <= (limit_0 + 0.00000999999974738f * (abs(near_0) + limit_0 + _S42 + _S43 + 1.0f));
}

float3 safe_normalize_0(float3 v_2)
{
    float n_0 = length(v_2);
    float3 _S45;
    if(n_0 > 1.00000000317107685e-30f)
    {
        _S45 = v_2 / float3(n_0) ;
    }
    else
    {
        _S45 = float3(0.0f) ;
    }
    return _S45;
}

Quat_0 from_axis_angle_0(float3 axis_0, float angle_0)
{
    float3 a_0 = safe_normalize_0(axis_0);
    float _S46 = 0.5f * angle_0;
    float s_1 = sin(_S46);
    thread Quat_0 q_3;
    (&q_3)->w_0 = cos(_S46);
    (&q_3)->x_0 = a_0.x * s_1;
    (&q_3)->y_0 = a_0.y * s_1;
    (&q_3)->z_0 = a_0.z * s_1;
    return q_3;
}

struct Box_0
{
    float3 center_2;
    float3 axis0_0;
    float3 axis1_0;
    float3 axis2_0;
    float3 half_2;
};

Box_0 grown_box_0(uint c_0, float3 center_3, float grow_0, KernelContext_0 thread* kernelContext_3)
{
    ChunkStatic_natural_0 device* _S47 = kernelContext_3->chunks_0+c_0;
    Quat_0 q_4 = quat_of_0(float4((kernelContext_3->islands_0+(uint4(_S47->info_0) ).y)->rotation_0) );
    float3 th_0 = (float4(*(kernelContext_3->state_0+(4U * c_0 + 1U))) ).xyz;
    Quat_0 hidden_0 = from_axis_angle_0(th_0, length(th_0));
    thread Box_0 b_0;
    (&b_0)->center_2 = center_3;
    float4 _S48 = float4(_S47->crot0_0) ;
    float4 _S49 = float4(_S47->crot1_0) ;
    float4 _S50 = float4(_S47->crot2_0) ;
    float3 _S51 = float3(_S48.x, _S49.x, _S50.x);
    thread Quat_0 _S52 = hidden_0;
    float3 _S53 = rotate_0(&_S52, _S51);
    thread Quat_0 _S54 = q_4;
    float3 _S55 = rotate_0(&_S54, _S53);
    (&b_0)->axis0_0 = _S55;
    float3 _S56 = float3(_S48.y, _S49.y, _S50.y);
    thread Quat_0 _S57 = hidden_0;
    float3 _S58 = rotate_0(&_S57, _S56);
    thread Quat_0 _S59 = q_4;
    float3 _S60 = rotate_0(&_S59, _S58);
    (&b_0)->axis1_0 = _S60;
    float3 _S61 = float3(_S48.z, _S49.z, _S50.z);
    thread Quat_0 _S62 = hidden_0;
    float3 _S63 = rotate_0(&_S62, _S61);
    thread Quat_0 _S64 = q_4;
    float3 _S65 = rotate_0(&_S64, _S63);
    (&b_0)->axis2_0 = _S65;
    (&b_0)->half_2 = (float4(_S47->half_0) ).xyz + float3(grow_0) ;
    return b_0;
}

bool may_overlap_0(const Box_0 thread* a_1, const Box_0 thread* b_1)
{
    float3 _S66 = b_1->center_2 - a_1->center_2;
    float3 _S67 = a_1->half_2;
    float3 _S68 = b_1->half_2;
    float _S69 = 0.00000999999974738f * (length(a_1->half_2) + length(b_1->half_2));
    float3 _S70 = a_1->axis0_0;
    float3 _S71 = a_1->axis1_0;
    float3 _S72 = a_1->axis2_0;
    float3 _S73 = b_1->axis0_0;
    float3 _S74 = b_1->axis1_0;
    float3 _S75 = b_1->axis2_0;
    array<float3, int(6)> _S76 = { { a_1->axis0_0, a_1->axis1_0, a_1->axis2_0, b_1->axis0_0, b_1->axis1_0, b_1->axis2_0 } };
    uint i_1 = 0U;
    for(;;)
    {
        if(i_1 < 15U)
        {
        }
        else
        {
            break;
        }
        float3 l_0;
        if(i_1 < 6U)
        {
            l_0 = _S76[i_1];
        }
        else
        {
            uint _S77 = i_1 - 6U;
            l_0 = cross(_S76[_S77 / 3U], _S76[3U + _S77 % 3U]);
        }
        float len_0 = length(l_0);
        if(len_0 <= 9.99999997475242708e-07f)
        {
            i_1 = i_1 + 1U;
            continue;
        }
        if((abs(dot(_S66, l_0))) > (_S67.x * abs(dot(_S70, l_0)) + _S67.y * abs(dot(_S71, l_0)) + _S67.z * abs(dot(_S72, l_0)) + (_S68.x * abs(dot(_S73, l_0)) + _S68.y * abs(dot(_S74, l_0)) + _S68.z * abs(dot(_S75, l_0))) + _S69 * len_0))
        {
            return false;
        }
        i_1 = i_1 + 1U;
    }
    return true;
}

bool pair_passes_0(uint a_2, uint b_2, bool thread* swept_0, KernelContext_0 thread* kernelContext_4)
{
    uint _S78 = 3U * a_2;
    float4 _S79 = float4(*(kernelContext_4->pm_0+_S78)) ;
    float4 _S80 = float4(*(kernelContext_4->pm_0+(_S78 + 1U))) ;
    uint _S81 = 3U * b_2;
    float4 _S82 = float4(*(kernelContext_4->pm_0+_S81)) ;
    float4 _S83 = float4(*(kernelContext_4->pm_0+(_S81 + 1U))) ;
    float d_1 = kernelContext_4->params_0->duration_0;
    float3 dv_0 = _S83.xyz - _S80.xyz;
    float dg_1;
    if(((float4(*(kernelContext_4->pm_0+(_S78 + 2U))) ).w) == ((float4(*(kernelContext_4->pm_0+(_S81 + 2U))) ).w))
    {
        dg_1 = 0.0f;
    }
    else
    {
        dg_1 = kernelContext_4->params_0->g_0;
    }
    float3 _S84 = _S79.xyz;
    float3 middle_0 = _S82.xyz - _S84 + dv_0 * float3((0.5f * d_1)) ;
    float sweep_0 = 0.5f * length(dv_0) * d_1 + 0.5f * dg_1 * d_1 * d_1;
    float _S85 = _S80.w;
    float _S86 = _S83.w;
    bool _S87 = (length(middle_0)) <= (_S79.w + _S82.w + _S85 + _S86 + sweep_0);
    *swept_0 = _S87;
    if(!_S87)
    {
        return false;
    }
    Box_0 _S88 = grown_box_0(a_2, _S84, _S85, kernelContext_4);
    Box_0 _S89 = grown_box_0(b_2, _S84 + middle_0, _S86 + sweep_0, kernelContext_4);
    thread Box_0 _S90 = _S88;
    thread Box_0 _S91 = _S89;
    bool _S92 = may_overlap_0(&_S90, &_S91);
    return _S92;
}

[[kernel]] void plan_pair_count(uint3 id_1 [[thread_position_in_grid]], PlanParams_0 constant* params_3 [[buffer(0)]], ChunkStatic_natural_0 device* chunks_3 [[buffer(1)]], Island_natural_0 device* islands_3 [[buffer(2)]], packed_float4 device* state_3 [[buffer(3)]], packed_float4 device* pm_3 [[buffer(5)]], uint device* pu_3 [[buffer(6)]], Impactor_natural_0 device* impactors_3 [[buffer(4)]])
{
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->params_0 = params_3;
    (&kernelContext_5)->chunks_0 = chunks_3;
    (&kernelContext_5)->islands_0 = islands_3;
    (&kernelContext_5)->state_0 = state_3;
    (&kernelContext_5)->pm_0 = pm_3;
    (&kernelContext_5)->pu_0 = pu_3;
    (&kernelContext_5)->impactors_0 = impactors_3;
    threadgroup array<float, int(64)> g_spread_3;
    (&kernelContext_5)->g_spread_0 = &g_spread_3;
    threadgroup array<float, int(64)> g_rb_3;
    (&kernelContext_5)->g_rb_0 = &g_rb_3;
    threadgroup array<float, int(64)> g_v_3;
    (&kernelContext_5)->g_v_0 = &g_v_3;
    threadgroup array<uint, int(1024)> g_scan_3;
    (&kernelContext_5)->g_scan_0 = &g_scan_3;
    uint j_0 = id_1.x;
    if(j_0 >= (params_3->job_count_0))
    {
        return;
    }
    uint _S93 = 2U * j_0;
    uint _S94 = *((&kernelContext_5)->pu_0+((&kernelContext_5)->params_0->job_base_0 + _S93 + 1U));
    uint _S95 = 2U * *((&kernelContext_5)->pu_0+((&kernelContext_5)->params_0->job_base_0 + _S93));
    uint device* _S96 = (&kernelContext_5)->pu_0+((&kernelContext_5)->params_0->cp_base_0 + _S95 + 1U);
    uint ib_1 = *_S96;
    bool _S97 = clusters_may_touch_0(*((&kernelContext_5)->pu_0+((&kernelContext_5)->params_0->cp_base_0 + _S95)), *_S96, &kernelContext_5);
    uint count_0;
    if(_S97)
    {
        uint4 _S98 = uint4(((&kernelContext_5)->islands_0+ib_1)->range_0) ;
        uint b_3 = _S98.x;
        count_0 = 0U;
        for(;;)
        {
            if(b_3 < (_S98.y))
            {
            }
            else
            {
                break;
            }
            thread bool swept_1;
            bool _S99 = pair_passes_0(_S94, b_3, &swept_1, &kernelContext_5);
            if(swept_1)
            {
                uint device* _S100 = (&kernelContext_5)->pu_0+((&kernelContext_5)->params_0->flag_base_0 + _S94);
                *_S100 = (*_S100) | 1U;
                uint device* _S101 = (&kernelContext_5)->pu_0+((&kernelContext_5)->params_0->flag_base_0 + b_3);
                *_S101 = (*_S101) | 1U;
            }
            if(_S99)
            {
                count_0 = count_0 + 1U;
            }
            b_3 = b_3 + 1U;
        }
    }
    else
    {
        count_0 = 0U;
    }
    *((&kernelContext_5)->pu_0+((&kernelContext_5)->params_0->count_base_0 + j_0)) = count_0;
    return;
}

[[kernel]] void plan_scan(uint3 thread_1 [[thread_position_in_threadgroup]], PlanParams_0 constant* params_4 [[buffer(0)]], ChunkStatic_natural_0 device* chunks_4 [[buffer(1)]], Island_natural_0 device* islands_4 [[buffer(2)]], packed_float4 device* state_4 [[buffer(3)]], packed_float4 device* pm_4 [[buffer(5)]], uint device* pu_4 [[buffer(6)]], Impactor_natural_0 device* impactors_4 [[buffer(4)]])
{
    thread KernelContext_0 kernelContext_6;
    (&kernelContext_6)->params_0 = params_4;
    (&kernelContext_6)->chunks_0 = chunks_4;
    (&kernelContext_6)->islands_0 = islands_4;
    (&kernelContext_6)->state_0 = state_4;
    (&kernelContext_6)->pm_0 = pm_4;
    (&kernelContext_6)->pu_0 = pu_4;
    (&kernelContext_6)->impactors_0 = impactors_4;
    threadgroup array<float, int(64)> g_spread_4;
    (&kernelContext_6)->g_spread_0 = &g_spread_4;
    threadgroup array<float, int(64)> g_rb_4;
    (&kernelContext_6)->g_rb_0 = &g_rb_4;
    threadgroup array<float, int(64)> g_v_4;
    (&kernelContext_6)->g_v_0 = &g_v_4;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_6)->g_scan_0 = &g_scan_4;
    uint tid_1 = thread_1.x;
    uint n_1 = params_4->job_count_0;
    uint base_0 = 0U;
    uint carry_0 = 0U;
    for(;;)
    {
        if(base_0 < n_1)
        {
        }
        else
        {
            break;
        }
        uint k_3 = base_0 + tid_1;
        bool _S102 = k_3 < n_1;
        uint v_3;
        if(_S102)
        {
            v_3 = *((&kernelContext_6)->pu_0+((&kernelContext_6)->params_0->count_base_0 + k_3));
        }
        else
        {
            v_3 = 0U;
        }
        (*(&kernelContext_6)->g_scan_0)[tid_1] = v_3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint step_0 = 1U;
        for(;;)
        {
            if(step_0 < 1024U)
            {
            }
            else
            {
                break;
            }
            uint add_0;
            if(tid_1 >= step_0)
            {
                add_0 = (*(&kernelContext_6)->g_scan_0)[tid_1 - step_0];
            }
            else
            {
                add_0 = 0U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            (*(&kernelContext_6)->g_scan_0)[tid_1] = (*(&kernelContext_6)->g_scan_0)[tid_1] + add_0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            step_0 = step_0 << 1U;
        }
        if(_S102)
        {
            *((&kernelContext_6)->pu_0+((&kernelContext_6)->params_0->count_base_0 + n_1 + 1U + k_3)) = carry_0 + (*(&kernelContext_6)->g_scan_0)[tid_1] - v_3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint carry_1 = carry_0 + (*(&kernelContext_6)->g_scan_0)[int(1023)];
        base_0 = base_0 + 1024U;
        carry_0 = carry_1;
    }
    if(tid_1 == 0U)
    {
        *((&kernelContext_6)->pu_0+((&kernelContext_6)->params_0->count_base_0 + n_1)) = carry_0;
    }
    return;
}

[[kernel]] void plan_pair_write(uint3 id_2 [[thread_position_in_grid]], PlanParams_0 constant* params_5 [[buffer(0)]], ChunkStatic_natural_0 device* chunks_5 [[buffer(1)]], Island_natural_0 device* islands_5 [[buffer(2)]], packed_float4 device* state_5 [[buffer(3)]], packed_float4 device* pm_5 [[buffer(5)]], uint device* pu_5 [[buffer(6)]], Impactor_natural_0 device* impactors_5 [[buffer(4)]])
{
    thread KernelContext_0 kernelContext_7;
    (&kernelContext_7)->params_0 = params_5;
    (&kernelContext_7)->chunks_0 = chunks_5;
    (&kernelContext_7)->islands_0 = islands_5;
    (&kernelContext_7)->state_0 = state_5;
    (&kernelContext_7)->pm_0 = pm_5;
    (&kernelContext_7)->pu_0 = pu_5;
    (&kernelContext_7)->impactors_0 = impactors_5;
    threadgroup array<float, int(64)> g_spread_5;
    (&kernelContext_7)->g_spread_0 = &g_spread_5;
    threadgroup array<float, int(64)> g_rb_5;
    (&kernelContext_7)->g_rb_0 = &g_rb_5;
    threadgroup array<float, int(64)> g_v_5;
    (&kernelContext_7)->g_v_0 = &g_v_5;
    threadgroup array<uint, int(1024)> g_scan_5;
    (&kernelContext_7)->g_scan_0 = &g_scan_5;
    uint j_1 = id_2.x;
    uint n_2 = params_5->job_count_0;
    bool _S103;
    if(j_1 >= (params_5->job_count_0))
    {
        _S103 = true;
    }
    else
    {
        _S103 = (*((&kernelContext_7)->pu_0+((&kernelContext_7)->params_0->count_base_0 + j_1))) == 0U;
    }
    if(_S103)
    {
        return;
    }
    uint _S104 = 2U * j_1;
    uint _S105 = *((&kernelContext_7)->pu_0+((&kernelContext_7)->params_0->job_base_0 + _S104 + 1U));
    uint device* _S106 = (&kernelContext_7)->pu_0+((&kernelContext_7)->params_0->count_base_0 + n_2 + 1U + j_1);
    uint4 _S107 = uint4(((&kernelContext_7)->islands_0+*((&kernelContext_7)->pu_0+((&kernelContext_7)->params_0->cp_base_0 + 2U * *((&kernelContext_7)->pu_0+((&kernelContext_7)->params_0->job_base_0 + _S104)) + 1U)))->range_0) ;
    uint b_4 = _S107.x;
    uint at_0 = *_S106;
    for(;;)
    {
        if(b_4 < (_S107.y))
        {
        }
        else
        {
            break;
        }
        thread bool swept_2;
        bool _S108 = pair_passes_0(_S105, b_4, &swept_2, &kernelContext_7);
        if(_S108)
        {
            uint _S109 = 2U * at_0;
            *((&kernelContext_7)->pu_0+((&kernelContext_7)->params_0->pair_base_0 + _S109)) = _S105;
            *((&kernelContext_7)->pu_0+((&kernelContext_7)->params_0->pair_base_0 + _S109 + 1U)) = b_4;
            at_0 = at_0 + 1U;
        }
        b_4 = b_4 + 1U;
    }
    return;
}

[[kernel]] void plan_side(uint3 id_3 [[thread_position_in_grid]], PlanParams_0 constant* params_6 [[buffer(0)]], ChunkStatic_natural_0 device* chunks_6 [[buffer(1)]], Island_natural_0 device* islands_6 [[buffer(2)]], packed_float4 device* state_6 [[buffer(3)]], packed_float4 device* pm_6 [[buffer(5)]], uint device* pu_6 [[buffer(6)]], Impactor_natural_0 device* impactors_6 [[buffer(4)]])
{
    thread KernelContext_0 kernelContext_8;
    (&kernelContext_8)->params_0 = params_6;
    (&kernelContext_8)->chunks_0 = chunks_6;
    (&kernelContext_8)->islands_0 = islands_6;
    (&kernelContext_8)->state_0 = state_6;
    (&kernelContext_8)->pm_0 = pm_6;
    (&kernelContext_8)->pu_0 = pu_6;
    (&kernelContext_8)->impactors_0 = impactors_6;
    threadgroup array<float, int(64)> g_spread_6;
    (&kernelContext_8)->g_spread_0 = &g_spread_6;
    threadgroup array<float, int(64)> g_rb_6;
    (&kernelContext_8)->g_rb_0 = &g_rb_6;
    threadgroup array<float, int(64)> g_v_6;
    (&kernelContext_8)->g_v_0 = &g_v_6;
    threadgroup array<uint, int(1024)> g_scan_6;
    (&kernelContext_8)->g_scan_0 = &g_scan_6;
    uint k_4 = id_3.x;
    if(k_4 >= (params_6->chunk_count_0))
    {
        return;
    }
    uint _S110 = (uint4(((&kernelContext_8)->chunks_0+k_4)->info_0) ).y;
    bool anchored_0 = (((uint4(((&kernelContext_8)->islands_0+_S110)->info_1) ).x) & 1U) != 0U;
    if((*((&kernelContext_8)->pu_0+((&kernelContext_8)->params_0->owned_base_0 + _S110))) == 0U)
    {
        return;
    }
    uint _S111 = 3U * k_4;
    float4 _S112 = float4(*((&kernelContext_8)->pm_0+_S111)) ;
    float4 _S113 = float4(*((&kernelContext_8)->pm_0+(_S111 + 1U))) ;
    bool falls_1 = ((float4(*((&kernelContext_8)->pm_0+(_S111 + 2U))) ).w) != 0.0f;
    float d_2 = (&kernelContext_8)->params_0->duration_0;
    bool _S114;
    if(((&kernelContext_8)->params_0->has_ground_0) != 0U)
    {
        _S114 = !anchored_0;
    }
    else
    {
        _S114 = false;
    }
    float reach_0;
    uint flags_0;
    if(_S114)
    {
        float z0_0 = _S112.z;
        float vz_0 = _S113.z;
        if(falls_1)
        {
            reach_0 = (&kernelContext_8)->params_0->gz_0;
        }
        else
        {
            reach_0 = 0.0f;
        }
        float t_low_0;
        if(reach_0 != 0.0f)
        {
            t_low_0 = clamp(- vz_0 / reach_0, 0.0f, d_2);
        }
        else
        {
            t_low_0 = 0.0f;
        }
        float _S115 = 0.5f * reach_0;
        if((min(min(z0_0, z0_0 + vz_0 * d_2 + _S115 * d_2 * d_2), z0_0 + vz_0 * t_low_0 + _S115 * t_low_0 * t_low_0) - _S112.w - _S113.w) <= ((&kernelContext_8)->params_0->ground_height_0))
        {
            flags_0 = 2U;
        }
        else
        {
            flags_0 = 0U;
        }
    }
    else
    {
        flags_0 = 0U;
    }
    float _S116 = length(_S113.xyz) * d_2;
    if(falls_1)
    {
        reach_0 = 0.5f * (&kernelContext_8)->params_0->g_0 * d_2 * d_2;
    }
    else
    {
        reach_0 = 0.0f;
    }
    float _S117 = _S116 + reach_0 + _S113.w;
    uint i_2 = 0U;
    for(;;)
    {
        if(i_2 < ((&kernelContext_8)->params_0->impactor_count_0))
        {
            _S114 = i_2 < 30U;
        }
        else
        {
            _S114 = false;
        }
        if(_S114)
        {
        }
        else
        {
            break;
        }
        Impactor_natural_0 device* _S118 = (&kernelContext_8)->impactors_0+i_2;
        if(((uint4(_S118->cand_0) ).z) != 0U)
        {
            i_2 = i_2 + 1U;
            continue;
        }
        float4 _S119 = float4(_S118->shape_0) ;
        if((_S119.x) == 0.0f)
        {
            reach_0 = _S119.y * 1.73205077648162842f;
        }
        else
        {
            reach_0 = length((float4(_S118->half_1) ).xyz);
        }
        uint flags_1;
        if((length(_S112.xyz - ((float4(_S118->position_1) ).xyz + (float4(_S118->position_err_1) ).xyz))) <= (reach_0 + _S112.w + _S117 + (2.0f * (length((float4(_S118->velocity_1) ).xyz) + length((float4(_S118->angular_velocity_1) ).xyz) * reach_0) * d_2 + (&kernelContext_8)->params_0->g_0 * d_2 * d_2 + 0.00100000004749745f)))
        {
            flags_1 = flags_0 | (4U << i_2);
        }
        else
        {
            flags_1 = flags_0;
        }
        flags_0 = flags_1;
        i_2 = i_2 + 1U;
    }
    uint device* _S120 = (&kernelContext_8)->pu_0+((&kernelContext_8)->params_0->flag_base_0 + k_4);
    *_S120 = (*_S120) | flags_0;
    return;
}

