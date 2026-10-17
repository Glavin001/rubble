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

float3 safe_normalize_0(float3 v_1)
{
    float n_0 = length(v_1);
    float3 _S1;
    if(n_0 > 1.00000000317107685e-30f)
    {
        _S1 = v_1 / float3(n_0) ;
    }
    else
    {
        _S1 = float3(0.0f) ;
    }
    return _S1;
}

Quat_0 from_axis_angle_0(float3 axis_0, float angle_0)
{
    float3 a_0 = safe_normalize_0(axis_0);
    float _S2 = 0.5f * angle_0;
    float s_0 = sin(_S2);
    thread Quat_0 q_2;
    (&q_2)->w_0 = cos(_S2);
    (&q_2)->x_0 = a_0.x * s_0;
    (&q_2)->y_0 = a_0.y * s_0;
    (&q_2)->z_0 = a_0.z * s_0;
    return q_2;
}

Quat_0 quat_mul_0(const Quat_0 thread* a_1, const Quat_0 thread* o_0)
{
    thread Quat_0 r_1;
    (&r_1)->w_0 = a_1->w_0 * o_0->w_0 - a_1->x_0 * o_0->x_0 - a_1->y_0 * o_0->y_0 - a_1->z_0 * o_0->z_0;
    (&r_1)->x_0 = a_1->w_0 * o_0->x_0 + a_1->x_0 * o_0->w_0 + a_1->y_0 * o_0->z_0 - a_1->z_0 * o_0->y_0;
    (&r_1)->y_0 = a_1->w_0 * o_0->y_0 - a_1->x_0 * o_0->z_0 + a_1->y_0 * o_0->w_0 + a_1->z_0 * o_0->x_0;
    (&r_1)->z_0 = a_1->w_0 * o_0->z_0 + a_1->x_0 * o_0->y_0 - a_1->y_0 * o_0->x_0 + a_1->z_0 * o_0->w_0;
    return r_1;
}

float4 quat_vec_0(const Quat_0 thread* q_3)
{
    return float4(q_3->x_0, q_3->y_0, q_3->z_0, q_3->w_0);
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
    ChunkStatic_natural_0 device* _S3 = (&kernelContext_0)->chunks_0+k_0;
    Island_natural_0 device* _S4 = (&kernelContext_0)->islands_0+(uint4(_S3->info_0) ).y;
    Quat_0 q_4 = quat_of_0(float4(_S4->rotation_0) );
    uint _S5 = 4U * k_0;
    float3 local_0 = (float4(_S3->center_0) ).xyz + (float4(*((&kernelContext_0)->state_0+_S5)) ).xyz;
    thread Quat_0 _S6 = q_4;
    float3 _S7 = rotate_0(&_S6, local_0);
    float3 _S8 = (float4(_S4->position_0) ).xyz;
    float3 _S9 = (float4(_S4->position_err_0) ).xyz;
    float3 _S10 = _S8 + _S9;
    float3 pos_0 = _S10 + _S7;
    float3 s1_0 = _S8 + _S7;
    float3 bb_0 = s1_0 - _S8;
    float3 _S11 = _S8 - (s1_0 - bb_0) + (_S7 - bb_0) + _S9;
    float3 hi_0 = s1_0 + _S11;
    float3 lo_0 = _S11 - (hi_0 - s1_0);
    float3 _S12 = (float4(_S4->com_0) ).xyz;
    float3 _S13 = local_0 - _S12;
    thread Quat_0 _S14 = q_4;
    float3 _S15 = rotate_0(&_S14, _S13);
    float3 _S16 = (float4(_S4->angular_velocity_0) ).xyz;
    float3 _S17 = (float4(_S4->velocity_0) ).xyz + (float4(_S4->velocity_err_0) ).xyz + cross(_S16, _S15);
    float3 _S18 = (float4(*((&kernelContext_0)->state_0+(_S5 + 2U))) ).xyz;
    thread Quat_0 _S19 = q_4;
    float3 _S20 = rotate_0(&_S19, _S18);
    float3 v_2 = _S17 + _S20;
    float3 _S21 = (float4(*((&kernelContext_0)->state_0+(_S5 + 3U))) ).xyz;
    thread Quat_0 _S22 = q_4;
    float3 _S23 = rotate_0(&_S22, _S21);
    float3 w_1 = _S16 + _S23;
    thread Quat_0 _S24 = q_4;
    float3 _S25 = rotate_0(&_S24, _S12);
    float radius_0 = length((float4(_S3->half_0) ).xyz);
    float budget_0 = 2.0f * length(w_1) * (length(pos_0 - (_S10 + _S25)) + radius_0) * (&kernelContext_0)->params_0->duration_0 + (&kernelContext_0)->params_0->g_0 * (&kernelContext_0)->params_0->duration_0 * (&kernelContext_0)->params_0->duration_0 + 0.05000000074505806f * radius_0 + 0.00100000004749745f;
    bool falls_0 = (((uint4(_S4->info_1) ).x) & 1U) == 0U;
    uint _S26 = 5U * k_0;
    *((&kernelContext_0)->pm_0+_S26) = packed_float4(float4(pos_0, radius_0)) ;
    *((&kernelContext_0)->pm_0+(_S26 + 1U)) = packed_float4(float4(v_2, budget_0)) ;
    packed_float4 device* _S27 = (&kernelContext_0)->pm_0+(_S26 + 2U);
    float _S28;
    if(falls_0)
    {
        _S28 = 1.0f;
    }
    else
    {
        _S28 = 0.0f;
    }
    *_S27 = packed_float4(float4(w_1, _S28)) ;
    float3 th_0 = (float4(*((&kernelContext_0)->state_0+(_S5 + 1U))) ).xyz;
    packed_float4 device* _S29 = (&kernelContext_0)->pm_0+(_S26 + 3U);
    Quat_0 _S30 = from_axis_angle_0(th_0, length(th_0));
    thread Quat_0 _S31 = q_4;
    thread Quat_0 _S32 = _S30;
    Quat_0 _S33 = quat_mul_0(&_S31, &_S32);
    thread Quat_0 _S34 = _S33;
    float4 _S35 = quat_vec_0(&_S34);
    *_S29 = packed_float4(_S35) ;
    *((&kernelContext_0)->pm_0+(_S26 + 4U)) = packed_float4(float4(hi_0, 0.0f)) ;
    *((&kernelContext_0)->pm_0+((&kernelContext_0)->params_0->reach_base_0 + 2U * (&kernelContext_0)->params_0->island_count_0 + k_0)) = packed_float4(float4(lo_0, 0.0f)) ;
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
    uint4 _S36 = uint4(((&kernelContext_1)->islands_0+i_0)->range_0) ;
    uint _S37 = _S36.y;
    uint _S38 = _S36.x;
    float3 center_1;
    if(_S37 > _S38)
    {
        center_1 = (float4(*((&kernelContext_1)->pm_0+5U * _S38)) ).xyz;
    }
    else
    {
        center_1 = float3(0.0f) ;
    }
    uint _S39 = _S38 + tid_0;
    float spread_0 = 0.0f;
    float rb_0 = 0.0f;
    float vmax_0 = 0.0f;
    uint k_1 = _S39;
    for(;;)
    {
        if(k_1 < _S37)
        {
        }
        else
        {
            break;
        }
        uint _S40 = 5U * k_1;
        float4 _S41 = float4(*((&kernelContext_1)->pm_0+_S40)) ;
        float4 _S42 = float4(*((&kernelContext_1)->pm_0+(_S40 + 1U))) ;
        float _S43 = max(spread_0, length(_S41.xyz - center_1));
        float _S44 = max(rb_0, _S41.w + _S42.w);
        float _S45 = max(vmax_0, length(_S42.xyz));
        uint k_2 = k_1 + 64U;
        spread_0 = _S43;
        rb_0 = _S44;
        vmax_0 = _S45;
        k_1 = k_2;
    }
    (*(&kernelContext_1)->g_spread_0)[tid_0] = spread_0;
    (*(&kernelContext_1)->g_rb_0)[tid_0] = rb_0;
    (*(&kernelContext_1)->g_v_0)[tid_0] = vmax_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s_1 = 32U;
    for(;;)
    {
        if(s_1 > 0U)
        {
        }
        else
        {
            break;
        }
        if(tid_0 < s_1)
        {
            uint _S46 = tid_0 + s_1;
            (*(&kernelContext_1)->g_spread_0)[tid_0] = max((*(&kernelContext_1)->g_spread_0)[tid_0], (*(&kernelContext_1)->g_spread_0)[_S46]);
            (*(&kernelContext_1)->g_rb_0)[tid_0] = max((*(&kernelContext_1)->g_rb_0)[tid_0], (*(&kernelContext_1)->g_rb_0)[_S46]);
            (*(&kernelContext_1)->g_v_0)[tid_0] = max((*(&kernelContext_1)->g_v_0)[tid_0], (*(&kernelContext_1)->g_v_0)[_S46]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_1 = s_1 >> 1U;
    }
    if(tid_0 == 0U)
    {
        uint _S47 = 2U * i_0;
        *((&kernelContext_1)->pm_0+((&kernelContext_1)->params_0->reach_base_0 + _S47)) = packed_float4(float4(center_1, (*(&kernelContext_1)->g_spread_0)[int(0)])) ;
        *((&kernelContext_1)->pm_0+((&kernelContext_1)->params_0->reach_base_0 + _S47 + 1U)) = packed_float4(float4((*(&kernelContext_1)->g_rb_0)[int(0)], (*(&kernelContext_1)->g_v_0)[int(0)], 0.0f, 0.0f)) ;
    }
    return;
}

bool clusters_may_touch_0(uint ia_0, uint ib_0, KernelContext_0 thread* kernelContext_2)
{
    uint _S48 = 2U * ia_0;
    float4 _S49 = float4(*(kernelContext_2->pm_0+(kernelContext_2->params_0->reach_base_0 + _S48))) ;
    float4 _S50 = float4(*(kernelContext_2->pm_0+(kernelContext_2->params_0->reach_base_0 + _S48 + 1U))) ;
    uint _S51 = 2U * ib_0;
    float4 _S52 = float4(*(kernelContext_2->pm_0+(kernelContext_2->params_0->reach_base_0 + _S51))) ;
    float4 _S53 = float4(*(kernelContext_2->pm_0+(kernelContext_2->params_0->reach_base_0 + _S51 + 1U))) ;
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
    float _S54 = _S49.w;
    float _S55 = _S52.w;
    float _S56 = 0.5f * ((_S50.y + _S53.y) * d_0);
    float near_0 = length(_S52.xyz - _S49.xyz) - _S54 - _S55 - _S56;
    float limit_0 = _S50.x + _S53.x + _S56 + 0.5f * dg_0 * d_0 * d_0;
    return near_0 <= (limit_0 + 0.00000999999974738f * (abs(near_0) + limit_0 + _S54 + _S55 + 1.0f));
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
    ChunkStatic_natural_0 device* _S57 = kernelContext_3->chunks_0+c_0;
    Quat_0 q_5 = quat_of_0(float4((kernelContext_3->islands_0+(uint4(_S57->info_0) ).y)->rotation_0) );
    float3 th_1 = (float4(*(kernelContext_3->state_0+(4U * c_0 + 1U))) ).xyz;
    Quat_0 hidden_0 = from_axis_angle_0(th_1, length(th_1));
    thread Box_0 b_0;
    (&b_0)->center_2 = center_3;
    float4 _S58 = float4(_S57->crot0_0) ;
    float4 _S59 = float4(_S57->crot1_0) ;
    float4 _S60 = float4(_S57->crot2_0) ;
    float3 _S61 = float3(_S58.x, _S59.x, _S60.x);
    thread Quat_0 _S62 = hidden_0;
    float3 _S63 = rotate_0(&_S62, _S61);
    thread Quat_0 _S64 = q_5;
    float3 _S65 = rotate_0(&_S64, _S63);
    (&b_0)->axis0_0 = _S65;
    float3 _S66 = float3(_S58.y, _S59.y, _S60.y);
    thread Quat_0 _S67 = hidden_0;
    float3 _S68 = rotate_0(&_S67, _S66);
    thread Quat_0 _S69 = q_5;
    float3 _S70 = rotate_0(&_S69, _S68);
    (&b_0)->axis1_0 = _S70;
    float3 _S71 = float3(_S58.z, _S59.z, _S60.z);
    thread Quat_0 _S72 = hidden_0;
    float3 _S73 = rotate_0(&_S72, _S71);
    thread Quat_0 _S74 = q_5;
    float3 _S75 = rotate_0(&_S74, _S73);
    (&b_0)->axis2_0 = _S75;
    (&b_0)->half_2 = (float4(_S57->half_0) ).xyz + float3(grow_0) ;
    return b_0;
}

bool may_overlap_0(const Box_0 thread* a_2, const Box_0 thread* b_1)
{
    float3 _S76 = b_1->center_2 - a_2->center_2;
    float3 _S77 = a_2->half_2;
    float3 _S78 = b_1->half_2;
    float _S79 = 0.00000999999974738f * (length(a_2->half_2) + length(b_1->half_2));
    float3 _S80 = a_2->axis0_0;
    float3 _S81 = a_2->axis1_0;
    float3 _S82 = a_2->axis2_0;
    float3 _S83 = b_1->axis0_0;
    float3 _S84 = b_1->axis1_0;
    float3 _S85 = b_1->axis2_0;
    array<float3, int(6)> _S86 = { { a_2->axis0_0, a_2->axis1_0, a_2->axis2_0, b_1->axis0_0, b_1->axis1_0, b_1->axis2_0 } };
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
            l_0 = _S86[i_1];
        }
        else
        {
            uint _S87 = i_1 - 6U;
            l_0 = cross(_S86[_S87 / 3U], _S86[3U + _S87 % 3U]);
        }
        float len_0 = length(l_0);
        if(len_0 <= 9.99999997475242708e-07f)
        {
            i_1 = i_1 + 1U;
            continue;
        }
        if((abs(dot(_S76, l_0))) > (_S77.x * abs(dot(_S80, l_0)) + _S77.y * abs(dot(_S81, l_0)) + _S77.z * abs(dot(_S82, l_0)) + (_S78.x * abs(dot(_S83, l_0)) + _S78.y * abs(dot(_S84, l_0)) + _S78.z * abs(dot(_S85, l_0))) + _S79 * len_0))
        {
            return false;
        }
        i_1 = i_1 + 1U;
    }
    return true;
}

bool pair_passes_0(uint a_3, uint b_2, bool thread* swept_0, KernelContext_0 thread* kernelContext_4)
{
    uint _S88 = 5U * a_3;
    float4 _S89 = float4(*(kernelContext_4->pm_0+_S88)) ;
    float4 _S90 = float4(*(kernelContext_4->pm_0+(_S88 + 1U))) ;
    uint _S91 = 5U * b_2;
    float4 _S92 = float4(*(kernelContext_4->pm_0+_S91)) ;
    float4 _S93 = float4(*(kernelContext_4->pm_0+(_S91 + 1U))) ;
    float d_1 = kernelContext_4->params_0->duration_0;
    float3 dv_0 = _S93.xyz - _S90.xyz;
    float dg_1;
    if(((float4(*(kernelContext_4->pm_0+(_S88 + 2U))) ).w) == ((float4(*(kernelContext_4->pm_0+(_S91 + 2U))) ).w))
    {
        dg_1 = 0.0f;
    }
    else
    {
        dg_1 = kernelContext_4->params_0->g_0;
    }
    float3 _S94 = _S89.xyz;
    float3 middle_0 = _S92.xyz - _S94 + dv_0 * float3((0.5f * d_1)) ;
    float sweep_0 = 0.5f * length(dv_0) * d_1 + 0.5f * dg_1 * d_1 * d_1;
    float _S95 = _S90.w;
    float _S96 = _S93.w;
    bool _S97 = (length(middle_0)) <= (_S89.w + _S92.w + _S95 + _S96 + sweep_0);
    *swept_0 = _S97;
    if(!_S97)
    {
        return false;
    }
    Box_0 _S98 = grown_box_0(a_3, _S94, _S95, kernelContext_4);
    Box_0 _S99 = grown_box_0(b_2, _S94 + middle_0, _S96 + sweep_0, kernelContext_4);
    thread Box_0 _S100 = _S98;
    thread Box_0 _S101 = _S99;
    bool _S102 = may_overlap_0(&_S100, &_S101);
    return _S102;
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
    uint _S103 = 2U * j_0;
    uint _S104 = *((&kernelContext_5)->pu_0+((&kernelContext_5)->params_0->job_base_0 + _S103 + 1U));
    uint _S105 = 2U * *((&kernelContext_5)->pu_0+((&kernelContext_5)->params_0->job_base_0 + _S103));
    uint device* _S106 = (&kernelContext_5)->pu_0+((&kernelContext_5)->params_0->cp_base_0 + _S105 + 1U);
    uint ib_1 = *_S106;
    bool _S107 = clusters_may_touch_0(*((&kernelContext_5)->pu_0+((&kernelContext_5)->params_0->cp_base_0 + _S105)), *_S106, &kernelContext_5);
    uint count_0;
    if(_S107)
    {
        uint4 _S108 = uint4(((&kernelContext_5)->islands_0+ib_1)->range_0) ;
        uint b_3 = _S108.x;
        count_0 = 0U;
        for(;;)
        {
            if(b_3 < (_S108.y))
            {
            }
            else
            {
                break;
            }
            thread bool swept_1;
            bool _S109 = pair_passes_0(_S104, b_3, &swept_1, &kernelContext_5);
            if(swept_1)
            {
                uint device* _S110 = (&kernelContext_5)->pu_0+((&kernelContext_5)->params_0->flag_base_0 + _S104);
                *_S110 = (*_S110) | 1U;
                uint device* _S111 = (&kernelContext_5)->pu_0+((&kernelContext_5)->params_0->flag_base_0 + b_3);
                *_S111 = (*_S111) | 1U;
            }
            if(_S109)
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
        bool _S112 = k_3 < n_1;
        uint v_3;
        if(_S112)
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
        if(_S112)
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
    bool _S113;
    if(j_1 >= (params_5->job_count_0))
    {
        _S113 = true;
    }
    else
    {
        _S113 = (*((&kernelContext_7)->pu_0+((&kernelContext_7)->params_0->count_base_0 + j_1))) == 0U;
    }
    if(_S113)
    {
        return;
    }
    uint _S114 = 2U * j_1;
    uint _S115 = *((&kernelContext_7)->pu_0+((&kernelContext_7)->params_0->job_base_0 + _S114 + 1U));
    uint device* _S116 = (&kernelContext_7)->pu_0+((&kernelContext_7)->params_0->count_base_0 + n_2 + 1U + j_1);
    uint4 _S117 = uint4(((&kernelContext_7)->islands_0+*((&kernelContext_7)->pu_0+((&kernelContext_7)->params_0->cp_base_0 + 2U * *((&kernelContext_7)->pu_0+((&kernelContext_7)->params_0->job_base_0 + _S114)) + 1U)))->range_0) ;
    uint b_4 = _S117.x;
    uint at_0 = *_S116;
    for(;;)
    {
        if(b_4 < (_S117.y))
        {
        }
        else
        {
            break;
        }
        thread bool swept_2;
        bool _S118 = pair_passes_0(_S115, b_4, &swept_2, &kernelContext_7);
        if(_S118)
        {
            uint _S119 = 2U * at_0;
            *((&kernelContext_7)->pu_0+((&kernelContext_7)->params_0->pair_base_0 + _S119)) = _S115;
            *((&kernelContext_7)->pu_0+((&kernelContext_7)->params_0->pair_base_0 + _S119 + 1U)) = b_4;
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
    uint _S120 = (uint4(((&kernelContext_8)->chunks_0+k_4)->info_0) ).y;
    bool anchored_0 = (((uint4(((&kernelContext_8)->islands_0+_S120)->info_1) ).x) & 1U) != 0U;
    if((*((&kernelContext_8)->pu_0+((&kernelContext_8)->params_0->owned_base_0 + _S120))) == 0U)
    {
        return;
    }
    uint _S121 = 5U * k_4;
    float4 _S122 = float4(*((&kernelContext_8)->pm_0+_S121)) ;
    float4 _S123 = float4(*((&kernelContext_8)->pm_0+(_S121 + 1U))) ;
    bool falls_1 = ((float4(*((&kernelContext_8)->pm_0+(_S121 + 2U))) ).w) != 0.0f;
    float d_2 = (&kernelContext_8)->params_0->duration_0;
    bool _S124;
    if(((&kernelContext_8)->params_0->has_ground_0) != 0U)
    {
        _S124 = !anchored_0;
    }
    else
    {
        _S124 = false;
    }
    float reach_0;
    uint flags_0;
    if(_S124)
    {
        float z0_0 = _S122.z;
        float vz_0 = _S123.z;
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
        float _S125 = 0.5f * reach_0;
        if((min(min(z0_0, z0_0 + vz_0 * d_2 + _S125 * d_2 * d_2), z0_0 + vz_0 * t_low_0 + _S125 * t_low_0 * t_low_0) - _S122.w - _S123.w) <= ((&kernelContext_8)->params_0->ground_height_0))
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
    float _S126 = length(_S123.xyz) * d_2;
    if(falls_1)
    {
        reach_0 = 0.5f * (&kernelContext_8)->params_0->g_0 * d_2 * d_2;
    }
    else
    {
        reach_0 = 0.0f;
    }
    float _S127 = _S126 + reach_0 + _S123.w;
    uint i_2 = 0U;
    for(;;)
    {
        if(i_2 < ((&kernelContext_8)->params_0->impactor_count_0))
        {
            _S124 = i_2 < 30U;
        }
        else
        {
            _S124 = false;
        }
        if(_S124)
        {
        }
        else
        {
            break;
        }
        Impactor_natural_0 device* _S128 = (&kernelContext_8)->impactors_0+i_2;
        if(((uint4(_S128->cand_0) ).z) != 0U)
        {
            i_2 = i_2 + 1U;
            continue;
        }
        float4 _S129 = float4(_S128->shape_0) ;
        if((_S129.x) == 0.0f)
        {
            reach_0 = _S129.y * 1.73205077648162842f;
        }
        else
        {
            reach_0 = length((float4(_S128->half_1) ).xyz);
        }
        uint flags_1;
        if((length(_S122.xyz - ((float4(_S128->position_1) ).xyz + (float4(_S128->position_err_1) ).xyz))) <= (reach_0 + _S122.w + _S127 + (2.0f * (length((float4(_S128->velocity_1) ).xyz) + length((float4(_S128->angular_velocity_1) ).xyz) * reach_0) * d_2 + (&kernelContext_8)->params_0->g_0 * d_2 * d_2 + 0.00100000004749745f)))
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
    uint device* _S130 = (&kernelContext_8)->pu_0+((&kernelContext_8)->params_0->flag_base_0 + k_4);
    *_S130 = (*_S130) | flags_0;
    return;
}

