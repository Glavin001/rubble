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

Quat_0 quat_mul_0(const Quat_0 thread* a_0, const Quat_0 thread* o_0)
{
    thread Quat_0 r_1;
    (&r_1)->w_0 = a_0->w_0 * o_0->w_0 - a_0->x_0 * o_0->x_0 - a_0->y_0 * o_0->y_0 - a_0->z_0 * o_0->z_0;
    (&r_1)->x_0 = a_0->w_0 * o_0->x_0 + a_0->x_0 * o_0->w_0 + a_0->y_0 * o_0->z_0 - a_0->z_0 * o_0->y_0;
    (&r_1)->y_0 = a_0->w_0 * o_0->y_0 - a_0->x_0 * o_0->z_0 + a_0->y_0 * o_0->w_0 + a_0->z_0 * o_0->x_0;
    (&r_1)->z_0 = a_0->w_0 * o_0->z_0 + a_0->x_0 * o_0->y_0 - a_0->y_0 * o_0->x_0 + a_0->z_0 * o_0->w_0;
    return r_1;
}

Quat_0 normalized_0(const Quat_0 thread* q_2)
{
    float n_0 = sqrt(q_2->w_0 * q_2->w_0 + q_2->x_0 * q_2->x_0 + q_2->y_0 * q_2->y_0 + q_2->z_0 * q_2->z_0);
    thread Quat_0 r_2;
    (&r_2)->w_0 = q_2->w_0 / n_0;
    (&r_2)->x_0 = q_2->x_0 / n_0;
    (&r_2)->y_0 = q_2->y_0 / n_0;
    (&r_2)->z_0 = q_2->z_0 / n_0;
    return r_2;
}

float3 to_zup_0(float3 v_1)
{
    return float3(v_1.x, - v_1.z, v_1.y);
}

struct RouteParams_0
{
    uint island_count_0;
    uint impactor_count_0;
    uint engine_base_0;
    uint pad0_0;
    float dt_0;
    float speed_0;
    float margin_0;
    float pad1_0;
};

struct Island_natural_0
{
    packed_uint4 range_0;
    packed_uint4 info_0;
    packed_float4 com_0;
    packed_float4 inertia0_0;
    packed_float4 inertia1_0;
    packed_float4 inertia2_0;
    packed_float4 inv0_0;
    packed_float4 inv1_0;
    packed_float4 inv2_0;
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

struct EngineBody_natural_0
{
    packed_float4 pos_inv_mass_0;
    packed_float4 rot_0;
    packed_float4 inv_inertia_flags_0;
    packed_float4 vel_sleep_0;
    packed_float4 ang_vel_gen_0;
    packed_float4 initial_pos_0;
    packed_float4 initial_rot_0;
};

struct ChunkStatic_natural_0
{
    packed_float4 center_0;
    packed_float4 inertia0_1;
    packed_float4 inertia1_1;
    packed_float4 inertia2_1;
    packed_float4 inv0_1;
    packed_float4 inv1_1;
    packed_float4 inv2_1;
    packed_float4 scale_0;
    packed_uint4 info_1;
    packed_uint4 load_range_0;
    packed_float4 half_0;
    packed_float4 crot0_0;
    packed_float4 crot1_0;
    packed_float4 crot2_0;
    packed_float4 cmat_0;
    packed_uint4 cinfo_0;
};

struct KernelContext_0
{
    RouteParams_0 constant* params_0;
    Island_natural_0 device* islands_0;
    packed_float4 device* ctab_0;
    EngineBody_natural_0 device* bodies_0;
    ChunkStatic_natural_0 device* chunks_0;
    packed_float4 device* state_0;
    packed_float4 device* impactors_0;
    uint device* route_0;
    array<float3, int(64)> threadgroup* g_lo_0;
    array<float3, int(64)> threadgroup* g_hi_0;
};

[[kernel]] void route_islands(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], RouteParams_0 constant* params_1 [[buffer(0)]], Island_natural_0 device* islands_1 [[buffer(1)]], packed_float4 device* ctab_1 [[buffer(4)]], EngineBody_natural_0 device* bodies_1 [[buffer(5)]], ChunkStatic_natural_0 device* chunks_1 [[buffer(2)]], packed_float4 device* state_1 [[buffer(3)]], packed_float4 device* impactors_1 [[buffer(6)]], uint device* route_1 [[buffer(7)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->islands_0 = islands_1;
    (&kernelContext_0)->ctab_0 = ctab_1;
    (&kernelContext_0)->bodies_0 = bodies_1;
    (&kernelContext_0)->chunks_0 = chunks_1;
    (&kernelContext_0)->state_0 = state_1;
    (&kernelContext_0)->impactors_0 = impactors_1;
    (&kernelContext_0)->route_0 = route_1;
    threadgroup array<float3, int(64)> g_lo_1;
    (&kernelContext_0)->g_lo_0 = &g_lo_1;
    threadgroup array<float3, int(64)> g_hi_1;
    (&kernelContext_0)->g_hi_0 = &g_hi_1;
    uint i_0 = group_0.x;
    uint t_1 = thread_0.x;
    bool overlap_0;
    if(i_0 >= (params_1->island_count_0))
    {
        overlap_0 = true;
    }
    else
    {
        overlap_0 = ((&kernelContext_0)->params_0->impactor_count_0) == 0U;
    }
    if(overlap_0)
    {
        return;
    }
    Island_natural_0 device* _S1 = (&kernelContext_0)->islands_0+i_0;
    uint4 _S2 = uint4(_S1->range_0) ;
    if((_S2.z) >= (_S2.w))
    {
        return;
    }
    uint _S3 = 3U * i_0;
    uint body_0 = (as_type<uint>(((float4(*((&kernelContext_0)->ctab_0+((&kernelContext_0)->params_0->engine_base_0 + _S3))) ).x)));
    Quat_0 q_3 = quat_of_0(float4(_S1->rotation_0) );
    float3 origin_0 = (float4(_S1->position_0) ).xyz;
    float3 _S4 = (float4(_S1->com_0) ).xyz;
    thread Quat_0 _S5 = q_3;
    float3 _S6 = rotate_0(&_S5, _S4);
    float3 _S7 = origin_0 + _S6;
    float3 _S8 = (float4(_S1->velocity_0) ).xyz;
    float3 _S9 = (float4(_S1->angular_velocity_0) ).xyz;
    float3 origin_1;
    float3 centre_0;
    float3 v_2;
    float3 w_1;
    Quat_0 q_4;
    if(body_0 != 4294967295U)
    {
        Quat_0 p_inv_0 = quat_of_0(float4(*((&kernelContext_0)->ctab_0+((&kernelContext_0)->params_0->engine_base_0 + _S3 + 1U))) );
        float3 com_1 = (float4(*((&kernelContext_0)->ctab_0+((&kernelContext_0)->params_0->engine_base_0 + _S3 + 2U))) ).xyz;
        EngineBody_natural_0 b_0 = (&kernelContext_0)->bodies_0[body_0];
        thread Quat_0 a_inv_0;
        (&a_inv_0)->w_0 = 0.70710676908493042f;
        (&a_inv_0)->x_0 = 0.70710676908493042f;
        (&a_inv_0)->y_0 = 0.0f;
        (&a_inv_0)->z_0 = 0.0f;
        Quat_0 _S10 = quat_of_0(float4(b_0.rot_0) );
        thread Quat_0 _S11 = a_inv_0;
        thread Quat_0 _S12 = _S10;
        Quat_0 _S13 = quat_mul_0(&_S11, &_S12);
        thread Quat_0 _S14 = _S13;
        thread Quat_0 _S15 = p_inv_0;
        Quat_0 _S16 = quat_mul_0(&_S14, &_S15);
        thread Quat_0 _S17 = _S16;
        Quat_0 _S18 = normalized_0(&_S17);
        float3 centre_1 = to_zup_0((float4(b_0.pos_inv_mass_0) ).xyz);
        thread Quat_0 _S19 = _S18;
        float3 _S20 = rotate_0(&_S19, com_1);
        float3 _S21 = to_zup_0((float4(b_0.vel_sleep_0) ).xyz);
        float3 _S22 = to_zup_0((float4(b_0.ang_vel_gen_0) ).xyz);
        origin_1 = centre_1 - _S20;
        q_4 = _S18;
        centre_0 = centre_1;
        v_2 = _S21;
        w_1 = _S22;
    }
    else
    {
        origin_1 = origin_0;
        q_4 = q_3;
        centre_0 = _S7;
        v_2 = _S8;
        w_1 = _S9;
    }
    float inf_0 = (as_type<float>((2139095040U)));
    float3 _S23 = float3(- inf_0) ;
    uint _S24 = _S2.x + t_1;
    float3 lo_0 = float3(inf_0) ;
    float3 hi_0 = _S23;
    uint c_0 = _S24;
    for(;;)
    {
        if(c_0 < (_S2.y))
        {
        }
        else
        {
            break;
        }
        float3 _S25 = (float4(((&kernelContext_0)->chunks_0+c_0)->center_0) ).xyz + (float4(*((&kernelContext_0)->state_0+4U * c_0)) ).xyz;
        thread Quat_0 _S26 = q_4;
        float3 _S27 = rotate_0(&_S26, _S25);
        float3 p_0 = origin_1 + _S27;
        float3 _S28 = float3((float4(((&kernelContext_0)->chunks_0+c_0)->half_0) ).w) ;
        float3 _S29 = min(lo_0, p_0 - _S28);
        float3 _S30 = max(hi_0, p_0 + _S28);
        uint c_1 = c_0 + 64U;
        lo_0 = _S29;
        hi_0 = _S30;
        c_0 = c_1;
    }
    (*(&kernelContext_0)->g_lo_0)[t_1] = lo_0;
    (*(&kernelContext_0)->g_hi_0)[t_1] = hi_0;
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
        if(t_1 < s_0)
        {
            uint _S31 = t_1 + s_0;
            (*(&kernelContext_0)->g_lo_0)[t_1] = min((*(&kernelContext_0)->g_lo_0)[t_1], (*(&kernelContext_0)->g_lo_0)[_S31]);
            (*(&kernelContext_0)->g_hi_0)[t_1] = max((*(&kernelContext_0)->g_hi_0)[t_1], (*(&kernelContext_0)->g_hi_0)[_S31]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    if(t_1 != 0U)
    {
        return;
    }
    float dt_1 = (&kernelContext_0)->params_0->dt_0;
    float3 _S32 = float3((&kernelContext_0)->params_0->dt_0) ;
    float _S33 = length(w_1) * 2.0f * length(max(abs((*(&kernelContext_0)->g_hi_0)[int(0)] - centre_0), abs(centre_0 - (*(&kernelContext_0)->g_lo_0)[int(0)])));
    float3 _S34 = float3((&kernelContext_0)->params_0->margin_0) ;
    float3 sweep_0 = abs(v_2) * _S32 + float3((_S33 * (&kernelContext_0)->params_0->dt_0))  + _S34;
    float3 _S35 = (*(&kernelContext_0)->g_lo_0)[int(0)] - sweep_0;
    float3 _S36 = (*(&kernelContext_0)->g_hi_0)[int(0)] + sweep_0;
    uint k_0 = 0U;
    uint found_0 = 0U;
    for(;;)
    {
        if(k_0 < ((&kernelContext_0)->params_0->impactor_count_0))
        {
        }
        else
        {
            break;
        }
        float4 _S37 = float4(*((&kernelContext_0)->impactors_0+k_0)) ;
        EngineBody_natural_0 b_1 = (&kernelContext_0)->bodies_0[(as_type<uint>((_S37.x)))];
        float3 c_2 = to_zup_0((float4(b_1.pos_inv_mass_0) ).xyz);
        float3 vk_0 = to_zup_0((float4(b_1.vel_sleep_0) ).xyz);
        float r_3 = _S37.y;
        float _S38 = length(to_zup_0((float4(b_1.ang_vel_gen_0) ).xyz)) * r_3;
        float3 ext_0 = float3(r_3)  + abs(vk_0) * _S32 + float3((_S38 * dt_1))  + _S34;
        if(all((c_2 - ext_0) <= _S36))
        {
            overlap_0 = all(_S35 <= (c_2 + ext_0));
        }
        else
        {
            overlap_0 = false;
        }
        float closing_0 = length(vk_0 - v_2) + _S38 + _S33;
        bool _S39;
        if(overlap_0)
        {
            _S39 = closing_0 > ((&kernelContext_0)->params_0->speed_0);
        }
        else
        {
            _S39 = false;
        }
        if(_S39)
        {
            found_0 = found_0 + 1U;
        }
        k_0 = k_0 + 1U;
    }
    if(found_0 > 0U)
    {
        *((&kernelContext_0)->route_0+int(0)) = 1U;
    }
    return;
}

