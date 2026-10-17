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

Quat_0 quat_mul_0(const Quat_0 thread* a_0, const Quat_0 thread* o_0)
{
    thread Quat_0 r_1;
    (&r_1)->w_0 = a_0->w_0 * o_0->w_0 - a_0->x_0 * o_0->x_0 - a_0->y_0 * o_0->y_0 - a_0->z_0 * o_0->z_0;
    (&r_1)->x_0 = a_0->w_0 * o_0->x_0 + a_0->x_0 * o_0->w_0 + a_0->y_0 * o_0->z_0 - a_0->z_0 * o_0->y_0;
    (&r_1)->y_0 = a_0->w_0 * o_0->y_0 - a_0->x_0 * o_0->z_0 + a_0->y_0 * o_0->w_0 + a_0->z_0 * o_0->x_0;
    (&r_1)->z_0 = a_0->w_0 * o_0->z_0 + a_0->x_0 * o_0->y_0 - a_0->y_0 * o_0->x_0 + a_0->z_0 * o_0->w_0;
    return r_1;
}

Quat_0 normalized_0(const Quat_0 thread* q_1)
{
    float n_0 = sqrt(q_1->w_0 * q_1->w_0 + q_1->x_0 * q_1->x_0 + q_1->y_0 * q_1->y_0 + q_1->z_0 * q_1->z_0);
    thread Quat_0 r_2;
    (&r_2)->w_0 = q_1->w_0 / n_0;
    (&r_2)->x_0 = q_1->x_0 / n_0;
    (&r_2)->y_0 = q_1->y_0 / n_0;
    (&r_2)->z_0 = q_1->z_0 / n_0;
    return r_2;
}

float3 to_zup_0(float3 v_0)
{
    return float3(v_0.x, - v_0.z, v_0.y);
}

float4 quat_vec_0(const Quat_0 thread* q_2)
{
    return float4(q_2->x_0, q_2->y_0, q_2->z_0, q_2->w_0);
}

float3 rotate_0(const Quat_0 thread* q_3, float3 v_1)
{
    float3 qv_0 = float3(q_3->x_0, q_3->y_0, q_3->z_0);
    float3 t_0 = cross(qv_0, v_1) * float3(2.0f) ;
    return v_1 + t_0 * float3(q_3->w_0)  + cross(qv_0, t_0);
}

float3 rotate_1(const Quat_0 thread* q_4, float3 v_2)
{
    float3 qv_1 = float3(q_4->x_0, q_4->y_0, q_4->z_0);
    float3 t_1 = cross(qv_1, v_2) * float3(2.0f) ;
    return v_2 + t_1 * float3(q_4->w_0)  + cross(qv_1, t_1);
}

float3 inverse_rotate_0(const Quat_0 thread* q_5, float3 v_3)
{
    thread Quat_0 c_0;
    (&c_0)->w_0 = q_5->w_0;
    (&c_0)->x_0 = - q_5->x_0;
    (&c_0)->y_0 = - q_5->y_0;
    (&c_0)->z_0 = - q_5->z_0;
    thread Quat_0 _S1 = c_0;
    float3 _S2 = rotate_0(&_S1, v_3);
    return _S2;
}

float3 inverse_rotate_1(const Quat_0 thread* q_6, float3 v_4)
{
    thread Quat_0 c_1;
    (&c_1)->w_0 = q_6->w_0;
    (&c_1)->x_0 = - q_6->x_0;
    (&c_1)->y_0 = - q_6->y_0;
    (&c_1)->z_0 = - q_6->z_0;
    thread Quat_0 _S3 = c_1;
    float3 _S4 = rotate_0(&_S3, v_4);
    return _S4;
}

float3 mat_mul_0(float4 r0_0, float4 r1_0, float4 r2_0, float3 v_5)
{
    return float3(dot(r0_0.xyz, v_5), dot(r1_0.xyz, v_5), dot(r2_0.xyz, v_5));
}

float3 world_mul_0(const Quat_0 thread* q_7, float4 r0_1, float4 r1_1, float4 r2_1, float3 v_6)
{
    float3 _S5 = inverse_rotate_1(q_7, v_6);
    float3 _S6 = rotate_1(q_7, mat_mul_0(r0_1, r1_1, r2_1, _S5));
    return _S6;
}

struct ResidentParams_0
{
    uint chunk_count_0;
    uint island_count_0;
    uint sload_base_0;
    uint rml_0;
    float4 gravity_0;
    float dt_0;
    uint props_base_0;
    uint engine_base_0;
    uint eload_base_0;
    uint pulse_base_0;
    uint pulse_count_base_0;
    uint pulse_cap_0;
    uint status_base_0;
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

struct Record_natural_0
{
    packed_uint4 ids_0;
    packed_float4 point_0;
    packed_float4 impulse_0;
    packed_float4 other_0;
};

struct KernelContext_0
{
    ResidentParams_0 constant* params_0;
    packed_float4 device* ctab_0;
    EngineBody_natural_0 device* start_bodies_0;
    Island_natural_0 device* islands_0;
    ChunkStatic_natural_0 device* chunks_0;
    packed_float4 device* state_0;
    uint device* sort_0;
    Record_natural_0 device* manifolds_0;
    packed_float4 device* scratch_0;
    array<float3, int(256)> threadgroup* g_f_0;
    array<float3, int(256)> threadgroup* g_t_0;
};

struct Island_0
{
    uint4 range_0;
    uint4 info_0;
    float4 com_0;
    float4 inertia0_0;
    float4 inertia1_0;
    float4 inertia2_0;
    float4 inv0_0;
    float4 inv1_0;
    float4 inv2_0;
    float4 wcom_0;
    float4 winv0_0;
    float4 winv1_0;
    float4 winv2_0;
    float4 rotation_0;
    float4 position_0;
    float4 position_err_0;
    float4 velocity_0;
    float4 velocity_err_0;
    float4 angular_velocity_0;
    uint done_0;
    uint coverage_0;
    uint wide_groups_0;
    uint done_pad_0;
    uint4 probes_0;
    float4 energy_0;
    float4 rotation_err_0;
    float4 momentum_0;
    float4 momentum_err_0;
};

[[kernel]] void resident_poses(uint3 id_0 [[thread_position_in_grid]], ResidentParams_0 constant* params_1 [[buffer(0)]], packed_float4 device* ctab_1 [[buffer(7)]], EngineBody_natural_0 device* start_bodies_1 [[buffer(8)]], Island_natural_0 device* islands_1 [[buffer(3)]], ChunkStatic_natural_0 device* chunks_1 [[buffer(4)]], packed_float4 device* state_1 [[buffer(5)]], uint device* sort_1 [[buffer(2)]], Record_natural_0 device* manifolds_1 [[buffer(1)]], packed_float4 device* scratch_1 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->ctab_0 = ctab_1;
    (&kernelContext_0)->start_bodies_0 = start_bodies_1;
    (&kernelContext_0)->islands_0 = islands_1;
    (&kernelContext_0)->chunks_0 = chunks_1;
    (&kernelContext_0)->state_0 = state_1;
    (&kernelContext_0)->sort_0 = sort_1;
    (&kernelContext_0)->manifolds_0 = manifolds_1;
    (&kernelContext_0)->scratch_0 = scratch_1;
    threadgroup array<float3, int(256)> g_f_1;
    (&kernelContext_0)->g_f_0 = &g_f_1;
    threadgroup array<float3, int(256)> g_t_1;
    (&kernelContext_0)->g_t_0 = &g_t_1;
    uint i_0 = id_0.x;
    if(i_0 >= (params_1->island_count_0))
    {
        return;
    }
    uint _S7 = 3U * i_0;
    uint body_0 = (as_type<uint>(((float4(*((&kernelContext_0)->ctab_0+((&kernelContext_0)->params_0->engine_base_0 + _S7))) ).x)));
    if(body_0 == 4294967295U)
    {
        return;
    }
    Quat_0 p_inv_0 = quat_of_0(float4(*((&kernelContext_0)->ctab_0+((&kernelContext_0)->params_0->engine_base_0 + _S7 + 1U))) );
    float3 com_1 = (float4(*((&kernelContext_0)->ctab_0+((&kernelContext_0)->params_0->engine_base_0 + _S7 + 2U))) ).xyz;
    EngineBody_natural_0 b_0 = (&kernelContext_0)->start_bodies_0[body_0];
    thread Quat_0 a_inv_0;
    (&a_inv_0)->w_0 = 0.70710676908493042f;
    (&a_inv_0)->x_0 = 0.70710676908493042f;
    (&a_inv_0)->y_0 = 0.0f;
    (&a_inv_0)->z_0 = 0.0f;
    Quat_0 _S8 = quat_of_0(float4(b_0.rot_0) );
    thread Quat_0 _S9 = a_inv_0;
    thread Quat_0 _S10 = _S8;
    Quat_0 _S11 = quat_mul_0(&_S9, &_S10);
    thread Quat_0 _S12 = _S11;
    thread Quat_0 _S13 = p_inv_0;
    Quat_0 _S14 = quat_mul_0(&_S12, &_S13);
    thread Quat_0 _S15 = _S14;
    Quat_0 _S16 = normalized_0(&_S15);
    float3 com_world_0 = to_zup_0((float4(b_0.pos_inv_mass_0) ).xyz);
    Island_natural_0 device* _S17 = (&kernelContext_0)->islands_0+i_0;
    uint4 _S18 = uint4((*_S17).info_0) ;
    float4 _S19 = float4((*_S17).com_0) ;
    float4 _S20 = float4((*_S17).inertia0_0) ;
    float4 _S21 = float4((*_S17).inertia1_0) ;
    float4 _S22 = float4((*_S17).inertia2_0) ;
    float4 _S23 = float4((*_S17).inv0_0) ;
    float4 _S24 = float4((*_S17).inv1_0) ;
    float4 _S25 = float4((*_S17).inv2_0) ;
    float4 _S26 = float4((*_S17).wcom_0) ;
    float4 _S27 = float4((*_S17).winv0_0) ;
    float4 _S28 = float4((*_S17).winv1_0) ;
    float4 _S29 = float4((*_S17).winv2_0) ;
    float4 _S30 = float4((*_S17).rotation_0) ;
    float4 _S31 = float4((*_S17).position_0) ;
    float4 _S32 = float4((*_S17).position_err_0) ;
    float4 _S33 = float4((*_S17).velocity_0) ;
    float4 _S34 = float4((*_S17).velocity_err_0) ;
    float4 _S35 = float4((*_S17).angular_velocity_0) ;
    uint4 _S36 = uint4((*_S17).probes_0) ;
    float4 _S37 = float4((*_S17).energy_0) ;
    float4 _S38 = float4((*_S17).rotation_err_0) ;
    float4 _S39 = float4((*_S17).momentum_0) ;
    float4 _S40 = float4((*_S17).momentum_err_0) ;
    thread Island_0 isl_0;
    (&isl_0)->range_0 = uint4((*_S17).range_0) ;
    (&isl_0)->info_0 = _S18;
    (&isl_0)->com_0 = _S19;
    (&isl_0)->inertia0_0 = _S20;
    (&isl_0)->inertia1_0 = _S21;
    (&isl_0)->inertia2_0 = _S22;
    (&isl_0)->inv0_0 = _S23;
    (&isl_0)->inv1_0 = _S24;
    (&isl_0)->inv2_0 = _S25;
    (&isl_0)->wcom_0 = _S26;
    (&isl_0)->winv0_0 = _S27;
    (&isl_0)->winv1_0 = _S28;
    (&isl_0)->winv2_0 = _S29;
    (&isl_0)->rotation_0 = _S30;
    (&isl_0)->position_0 = _S31;
    (&isl_0)->position_err_0 = _S32;
    (&isl_0)->velocity_0 = _S33;
    (&isl_0)->velocity_err_0 = _S34;
    (&isl_0)->angular_velocity_0 = _S35;
    (&isl_0)->done_0 = (*_S17).done_0;
    (&isl_0)->coverage_0 = (*_S17).coverage_0;
    (&isl_0)->wide_groups_0 = (*_S17).wide_groups_0;
    (&isl_0)->done_pad_0 = (*_S17).done_pad_0;
    (&isl_0)->probes_0 = _S36;
    (&isl_0)->energy_0 = _S37;
    (&isl_0)->rotation_err_0 = _S38;
    (&isl_0)->momentum_0 = _S39;
    (&isl_0)->momentum_err_0 = _S40;
    thread Quat_0 _S41 = _S16;
    float4 _S42 = quat_vec_0(&_S41);
    (&isl_0)->rotation_0 = _S42;
    float4 _S43 = float4(0.0f) ;
    (&isl_0)->rotation_err_0 = _S43;
    thread Quat_0 _S44 = _S16;
    float3 _S45 = rotate_0(&_S44, com_1);
    (&isl_0)->position_0 = float4(com_world_0 - _S45, 0.0f);
    (&isl_0)->position_err_0 = _S43;
    (&isl_0)->velocity_0 = float4(to_zup_0((float4(b_0.vel_sleep_0) ).xyz), 0.0f);
    (&isl_0)->velocity_err_0 = _S43;
    float3 w_1 = to_zup_0((float4(b_0.ang_vel_gen_0) ).xyz);
    (&isl_0)->angular_velocity_0 = float4(w_1, 0.0f);
    thread Quat_0 _S46 = _S16;
    float3 _S47 = world_mul_0(&_S46, (&isl_0)->inertia0_0, (&isl_0)->inertia1_0, (&isl_0)->inertia2_0, w_1);
    (&isl_0)->momentum_0 = float4(_S47, 0.0f);
    (&isl_0)->momentum_err_0 = _S43;
    Island_natural_0 device* _S48 = (&kernelContext_0)->islands_0+i_0;
    _S48->range_0 = packed_uint4(isl_0.range_0) ;
    _S48->info_0 = packed_uint4(isl_0.info_0) ;
    _S48->com_0 = packed_float4(isl_0.com_0) ;
    _S48->inertia0_0 = packed_float4(isl_0.inertia0_0) ;
    _S48->inertia1_0 = packed_float4(isl_0.inertia1_0) ;
    _S48->inertia2_0 = packed_float4(isl_0.inertia2_0) ;
    _S48->inv0_0 = packed_float4(isl_0.inv0_0) ;
    _S48->inv1_0 = packed_float4(isl_0.inv1_0) ;
    _S48->inv2_0 = packed_float4(isl_0.inv2_0) ;
    _S48->wcom_0 = packed_float4(isl_0.wcom_0) ;
    _S48->winv0_0 = packed_float4(isl_0.winv0_0) ;
    _S48->winv1_0 = packed_float4(isl_0.winv1_0) ;
    _S48->winv2_0 = packed_float4(isl_0.winv2_0) ;
    _S48->rotation_0 = packed_float4(isl_0.rotation_0) ;
    _S48->position_0 = packed_float4(isl_0.position_0) ;
    _S48->position_err_0 = packed_float4(isl_0.position_err_0) ;
    _S48->velocity_0 = packed_float4(isl_0.velocity_0) ;
    _S48->velocity_err_0 = packed_float4(isl_0.velocity_err_0) ;
    _S48->angular_velocity_0 = packed_float4(isl_0.angular_velocity_0) ;
    _S48->done_0 = isl_0.done_0;
    _S48->coverage_0 = isl_0.coverage_0;
    _S48->wide_groups_0 = isl_0.wide_groups_0;
    _S48->done_pad_0 = isl_0.done_pad_0;
    _S48->probes_0 = packed_uint4(isl_0.probes_0) ;
    _S48->energy_0 = packed_float4(isl_0.energy_0) ;
    _S48->rotation_err_0 = packed_float4(isl_0.rotation_err_0) ;
    _S48->momentum_0 = packed_float4(isl_0.momentum_0) ;
    _S48->momentum_err_0 = packed_float4(isl_0.momentum_err_0) ;
    return;
}

float pulse_mean_0(float start_0, float plateau_0, float plateau_time_0, float peak_0, float duration_0, float dt_1)
{
    float _S49 = start_0 + plateau_time_0;
    float sum_0 = plateau_0 * max(min(_S49, dt_1) - max(start_0, 0.0f), 0.0f);
    float _S50 = max(0.0f, _S49);
    float _S51 = min(dt_1, _S49 + duration_0);
    bool _S52;
    if(_S51 > _S50)
    {
        _S52 = duration_0 > 0.0f;
    }
    else
    {
        _S52 = false;
    }
    float sum_1;
    if(_S52)
    {
        float w_2 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_2 * (cos(w_2 * (_S50 - _S49)) - cos(w_2 * (_S51 - _S49)));
    }
    else
    {
        sum_1 = sum_0;
    }
    return sum_1 / dt_1;
}

float hertz_duration_0(float m_0, float r_3, float e_0, float v_7)
{
    return 2.86999988555908203f * pow(m_0 * m_0 / (r_3 * e_0 * e_0 * max(abs(v_7), 0.00100000004749745f)), 0.20000000298023224f);
}

Quat_0 island_rotation_0(uint _S53, KernelContext_0 thread* kernelContext_1)
{
    return quat_of_0(float4((kernelContext_1->islands_0+_S53)->rotation_0) );
}

[[kernel]] void resident_filter(uint3 id_1 [[thread_position_in_grid]], ResidentParams_0 constant* params_2 [[buffer(0)]], packed_float4 device* ctab_2 [[buffer(7)]], EngineBody_natural_0 device* start_bodies_2 [[buffer(8)]], Island_natural_0 device* islands_2 [[buffer(3)]], ChunkStatic_natural_0 device* chunks_2 [[buffer(4)]], packed_float4 device* state_2 [[buffer(5)]], uint device* sort_2 [[buffer(2)]], Record_natural_0 device* manifolds_2 [[buffer(1)]], packed_float4 device* scratch_2 [[buffer(6)]])
{
    float m_1;
    bool overflow_0;
    bool overflow_1;
    uint p_0;
    uint kept_0;
    uint q3_0;
    float3 f_total_0;
    float3 t_total_0;
    thread KernelContext_0 kernelContext_2;
    (&kernelContext_2)->params_0 = params_2;
    (&kernelContext_2)->ctab_0 = ctab_2;
    (&kernelContext_2)->start_bodies_0 = start_bodies_2;
    (&kernelContext_2)->islands_0 = islands_2;
    (&kernelContext_2)->chunks_0 = chunks_2;
    (&kernelContext_2)->state_0 = state_2;
    (&kernelContext_2)->sort_0 = sort_2;
    (&kernelContext_2)->manifolds_0 = manifolds_2;
    (&kernelContext_2)->scratch_0 = scratch_2;
    threadgroup array<float3, int(256)> g_f_2;
    (&kernelContext_2)->g_f_0 = &g_f_2;
    threadgroup array<float3, int(256)> g_t_2;
    (&kernelContext_2)->g_t_0 = &g_t_2;
    uint k_0 = id_1.x;
    uint n_1 = params_2->chunk_count_0;
    if(k_0 >= (params_2->chunk_count_0))
    {
        return;
    }
    ChunkStatic_natural_0 device* _S54 = (&kernelContext_2)->chunks_0+k_0;
    uint _S55 = (uint4(_S54->info_1) ).y;
    Island_natural_0 device* _S56 = (&kernelContext_2)->islands_0+_S55;
    Island_natural_0 isl_1 = *_S56;
    bool anchored_0 = (((uint4((*_S56).info_0) ).x) & 1U) != 0U;
    Quat_0 _S57 = island_rotation_0(_S55, &kernelContext_2);
    float3 _S58 = (float4((*_S56).position_0) ).xyz;
    float4 _S59 = float4(_S54->center_0) ;
    float3 _S60 = _S59.xyz + (float4(*((&kernelContext_2)->state_0+4U * k_0)) ).xyz;
    thread Quat_0 _S61 = _S57;
    float3 _S62 = rotate_0(&_S61, _S60);
    float3 _S63 = _S58 + _S62;
    float4 _S64 = float4(*((&kernelContext_2)->ctab_0+((&kernelContext_2)->params_0->props_base_0 + k_0))) ;
    float _S65;
    if(anchored_0)
    {
        _S65 = (as_type<float>((2139095040U)));
    }
    else
    {
        _S65 = (float4(isl_1.com_0) ).w;
    }
    float _S66 = (&kernelContext_2)->params_0->dt_0;
    float3 _S67 = float3(0.0f) ;
    uint _S68 = (&kernelContext_2)->sort_0[k_0];
    uint _S69 = (&kernelContext_2)->sort_0[n_1 + 1U + k_0];
    bool first_0 = true;
    uint last_0 = 0U;
    uint step_0 = 0U;
    float3 f_total_1 = _S67;
    float3 t_total_1 = _S67;
    for(;;)
    {
        if(step_0 < _S69)
        {
        }
        else
        {
            break;
        }
        p_0 = 4294967295U;
        kept_0 = 0U;
        q3_0 = 0U;
        for(;;)
        {
            if(q3_0 < _S69)
            {
            }
            else
            {
                break;
            }
            uint key_0 = (uint4((&kernelContext_2)->manifolds_0[_S68 + q3_0].ids_0) ).y + 1U;
            if(first_0)
            {
                overflow_0 = true;
            }
            else
            {
                overflow_0 = key_0 > last_0;
            }
            if(overflow_0)
            {
                if(p_0 == 4294967295U)
                {
                    overflow_1 = true;
                }
                else
                {
                    overflow_1 = key_0 < kept_0;
                }
            }
            else
            {
                overflow_1 = false;
            }
            if(overflow_1)
            {
                p_0 = q3_0;
                kept_0 = key_0;
            }
            q3_0 = q3_0 + 1U;
        }
        if(p_0 == 4294967295U)
        {
            break;
        }
        Record_natural_0 r_4 = (&kernelContext_2)->manifolds_0[_S68 + p_0];
        float4 _S70 = float4(r_4.impulse_0) ;
        float3 _S71 = _S70.xyz;
        float j_0 = length(_S71);
        if(j_0 == 0.0f)
        {
            uint _S72 = step_0 + 1U;
            first_0 = false;
            last_0 = kept_0;
            step_0 = _S72;
            continue;
        }
        float m_other_0 = (float4(r_4.other_0) ).x;
        if(isinf(_S65))
        {
            if(isinf(m_other_0))
            {
                m_1 = _S59.w;
            }
            else
            {
                m_1 = m_other_0;
            }
        }
        else
        {
            if(isinf(m_other_0))
            {
                m_1 = _S65;
            }
            else
            {
                m_1 = _S65 * m_other_0 / (_S65 + m_other_0);
            }
        }
        float _S73 = min(j_0, 2.0f * m_1 * max(_S70.w - (&kernelContext_2)->params_0->gravity_0.w, 0.0f));
        if(_S73 < j_0)
        {
            float3 f_0 = _S71 / float3(j_0)  * float3(((j_0 - _S73) / _S66)) ;
            float3 t_total_2 = t_total_1 + cross((float4(r_4.point_0) ).xyz - _S63, f_0);
            f_total_0 = f_total_1 + f_0;
            t_total_0 = t_total_2;
        }
        else
        {
            f_total_0 = f_total_1;
            t_total_0 = t_total_1;
        }
        f_total_1 = f_total_0;
        t_total_1 = t_total_0;
        uint _S72 = step_0 + 1U;
        first_0 = false;
        last_0 = kept_0;
        step_0 = _S72;
    }
    uint _S74 = (&kernelContext_2)->params_0->pulse_base_0 + 3U * (&kernelContext_2)->params_0->pulse_cap_0 * k_0;
    uint _S75 = (as_type<uint>(((float4(*((&kernelContext_2)->ctab_0+((&kernelContext_2)->params_0->pulse_count_base_0 + k_0))) ).x)));
    p_0 = 0U;
    kept_0 = 0U;
    for(;;)
    {
        if(p_0 < _S75)
        {
        }
        else
        {
            break;
        }
        uint _S76 = _S74 + 3U * p_0;
        float4 _S77 = float4(*((&kernelContext_2)->ctab_0+_S76)) ;
        thread float4 a_1 = _S77;
        float4 _S78 = float4(*((&kernelContext_2)->ctab_0+(_S76 + 1U))) ;
        float4 _S79 = float4(*((&kernelContext_2)->ctab_0+(_S76 + 2U))) ;
        float _S80 = _S78.w;
        float mean_0 = pulse_mean_0(_S77.x, _S77.y, _S77.z, _S77.w, _S80, _S66);
        if(mean_0 != 0.0f)
        {
            float3 f_1 = _S79.xyz * float3(mean_0) ;
            float3 t_total_3 = t_total_1 + cross(_S78.xyz - _S63, f_1);
            f_total_1 = f_total_1 + f_1;
            t_total_1 = t_total_3;
        }
        if((a_1.x + a_1.z + _S80) > _S66)
        {
            a_1.x = a_1.x - _S66;
            uint _S81 = _S74 + 3U * kept_0;
            *((&kernelContext_2)->ctab_0+_S81) = packed_float4(a_1) ;
            *((&kernelContext_2)->ctab_0+(_S81 + 1U)) = packed_float4(_S78) ;
            *((&kernelContext_2)->ctab_0+(_S81 + 2U)) = packed_float4(_S79) ;
            kept_0 = kept_0 + 1U;
        }
        p_0 = p_0 + 1U;
    }
    overflow_0 = false;
    q3_0 = 0U;
    for(;;)
    {
        if(q3_0 < _S69)
        {
        }
        else
        {
            break;
        }
        Record_natural_0 r_5 = (&kernelContext_2)->manifolds_0[_S68 + q3_0];
        float4 _S82 = float4(r_5.impulse_0) ;
        float3 _S83 = _S82.xyz;
        float j_1 = length(_S83);
        if(j_1 == 0.0f)
        {
            overflow_1 = overflow_0;
            uint _S84 = q3_0 + 1U;
            overflow_0 = overflow_1;
            q3_0 = _S84;
            continue;
        }
        float4 _S85 = float4(r_5.other_0) ;
        float m_other_1 = _S85.x;
        if(isinf(_S65))
        {
            if(isinf(m_other_1))
            {
                m_1 = _S59.w;
            }
            else
            {
                m_1 = m_other_1;
            }
        }
        else
        {
            if(isinf(m_other_1))
            {
                m_1 = _S65;
            }
            else
            {
                m_1 = _S65 * m_other_1 / (_S65 + m_other_1);
            }
        }
        float _S86 = _S82.w;
        float _S87 = min(j_1, 2.0f * m_1 * max(_S86 - (&kernelContext_2)->params_0->gravity_0.w, 0.0f));
        if(_S87 <= 0.0f)
        {
            overflow_1 = overflow_0;
            uint _S84 = q3_0 + 1U;
            overflow_0 = overflow_1;
            q3_0 = _S84;
            continue;
        }
        float _S88 = _S64.y;
        float duration_1 = hertz_duration_0(m_1, min(_S85.z, _S64.z), 1.0f / ((1.0f - _S88 * _S88) / _S64.x + 1.0f / _S85.y), _S86);
        float peak_1 = 3.14159274101257324f * _S87 / (2.0f * duration_1);
        float3 dir_0 = _S83 / float3(j_1) ;
        float mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S66);
        if(mean_1 != 0.0f)
        {
            float3 f_2 = dir_0 * float3(mean_1) ;
            float3 t_total_4 = t_total_1 + cross((float4(r_5.point_0) ).xyz - _S63, f_2);
            f_total_0 = f_total_1 + f_2;
            t_total_0 = t_total_4;
        }
        else
        {
            f_total_0 = f_total_1;
            t_total_0 = t_total_1;
        }
        uint kept_1;
        if(duration_1 > _S66)
        {
            if(kept_0 < ((&kernelContext_2)->params_0->pulse_cap_0))
            {
                uint _S89 = _S74 + 3U * kept_0;
                *((&kernelContext_2)->ctab_0+_S89) = packed_float4(float4(- _S66, 0.0f, 0.0f, peak_1)) ;
                *((&kernelContext_2)->ctab_0+(_S89 + 1U)) = packed_float4(float4((float4(r_5.point_0) ).xyz, duration_1)) ;
                *((&kernelContext_2)->ctab_0+(_S89 + 2U)) = packed_float4(float4(dir_0, 0.0f)) ;
                uint _S90 = kept_0 + 1U;
                overflow_1 = overflow_0;
                kept_1 = _S90;
            }
            else
            {
                overflow_1 = true;
                kept_1 = kept_0;
            }
        }
        else
        {
            overflow_1 = overflow_0;
            kept_1 = kept_0;
        }
        f_total_1 = f_total_0;
        t_total_1 = t_total_0;
        kept_0 = kept_1;
        uint _S84 = q3_0 + 1U;
        overflow_0 = overflow_1;
        q3_0 = _S84;
    }
    *((&kernelContext_2)->ctab_0+((&kernelContext_2)->params_0->pulse_count_base_0 + k_0)) = packed_float4(float4((as_type<float>((kept_0))), 0.0f, 0.0f, 0.0f)) ;
    if(overflow_0)
    {
        *((&kernelContext_2)->ctab_0+(&kernelContext_2)->params_0->status_base_0) = packed_float4(float4((as_type<float>((1U))), 0.0f, 0.0f, 0.0f)) ;
    }
    uint _S91 = 2U * k_0;
    *((&kernelContext_2)->ctab_0+((&kernelContext_2)->params_0->eload_base_0 + _S91)) = packed_float4(float4(f_total_1, 0.0f)) ;
    *((&kernelContext_2)->ctab_0+((&kernelContext_2)->params_0->eload_base_0 + _S91 + 1U)) = packed_float4(float4(t_total_1, 0.0f)) ;
    return;
}

[[kernel]] void resident_static_loads(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], ResidentParams_0 constant* params_3 [[buffer(0)]], packed_float4 device* ctab_3 [[buffer(7)]], EngineBody_natural_0 device* start_bodies_3 [[buffer(8)]], Island_natural_0 device* islands_3 [[buffer(3)]], ChunkStatic_natural_0 device* chunks_3 [[buffer(4)]], packed_float4 device* state_3 [[buffer(5)]], uint device* sort_3 [[buffer(2)]], Record_natural_0 device* manifolds_3 [[buffer(1)]], packed_float4 device* scratch_3 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->params_0 = params_3;
    (&kernelContext_3)->ctab_0 = ctab_3;
    (&kernelContext_3)->start_bodies_0 = start_bodies_3;
    (&kernelContext_3)->islands_0 = islands_3;
    (&kernelContext_3)->chunks_0 = chunks_3;
    (&kernelContext_3)->state_0 = state_3;
    (&kernelContext_3)->sort_0 = sort_3;
    (&kernelContext_3)->manifolds_0 = manifolds_3;
    (&kernelContext_3)->scratch_0 = scratch_3;
    threadgroup array<float3, int(256)> g_f_3;
    (&kernelContext_3)->g_f_0 = &g_f_3;
    threadgroup array<float3, int(256)> g_t_3;
    (&kernelContext_3)->g_t_0 = &g_t_3;
    uint i_1 = group_0.x;
    uint tid_0 = thread_0.x;
    if(i_1 >= (params_3->island_count_0))
    {
        return;
    }
    Island_natural_0 device* _S92 = (&kernelContext_3)->islands_0+i_1;
    Island_natural_0 isl_2 = *_S92;
    uint4 _S93 = uint4((*_S92).info_0) ;
    if(((_S93.z) & 8U) == 0U)
    {
        return;
    }
    bool anchored_1 = ((_S93.x) & 1U) != 0U;
    Quat_0 _S94 = island_rotation_0(i_1, &kernelContext_3);
    float3 w_3 = (float4(isl_2.angular_velocity_0) ).xyz;
    float3 _S95 = (&kernelContext_3)->params_0->gravity_0.xyz;
    float3 _S96 = (float4(isl_2.position_0) ).xyz;
    float4 _S97 = float4(isl_2.com_0) ;
    float3 _S98 = _S97.xyz;
    thread Quat_0 _S99 = _S94;
    float3 _S100 = rotate_0(&_S99, _S98);
    float3 _S101 = _S96 + _S100;
    float3 _S102 = float3(0.0f) ;
    bool _S103 = !anchored_1;
    uint s_0;
    float3 f_3;
    float3 t_2;
    if(_S103)
    {
        uint4 _S104 = uint4(isl_2.range_0) ;
        s_0 = _S104.x + tid_0;
        f_3 = _S102;
        t_2 = _S102;
        for(;;)
        {
            if(s_0 < (_S104.y))
            {
            }
            else
            {
                break;
            }
            uint _S105 = 2U * s_0;
            float4 _S106 = float4(((&kernelContext_3)->chunks_0+s_0)->center_0) ;
            float3 fc_0 = (float4(*((&kernelContext_3)->ctab_0+((&kernelContext_3)->params_0->eload_base_0 + _S105))) ).xyz + _S95 * float3(_S106.w) ;
            float3 _S107 = _S106.xyz + (float4(*((&kernelContext_3)->state_0+4U * s_0)) ).xyz;
            thread Quat_0 _S108 = _S94;
            float3 _S109 = rotate_0(&_S108, _S107);
            float3 f_4 = f_3 + fc_0;
            float3 t_3 = t_2 + (cross(_S96 + _S109 - _S101, fc_0) + (float4(*((&kernelContext_3)->ctab_0+((&kernelContext_3)->params_0->eload_base_0 + _S105 + 1U))) ).xyz);
            s_0 = s_0 + 256U;
            f_3 = f_4;
            t_2 = t_3;
        }
    }
    else
    {
        f_3 = _S102;
        t_2 = _S102;
    }
    (*(&kernelContext_3)->g_f_0)[tid_0] = f_3;
    (*(&kernelContext_3)->g_t_0)[tid_0] = t_2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s_0 = 128U;
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
            uint _S110 = tid_0 + s_0;
            (*(&kernelContext_3)->g_f_0)[tid_0] = (*(&kernelContext_3)->g_f_0)[tid_0] + (*(&kernelContext_3)->g_f_0)[_S110];
            (*(&kernelContext_3)->g_t_0)[tid_0] = (*(&kernelContext_3)->g_t_0)[tid_0] + (*(&kernelContext_3)->g_t_0)[_S110];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    float3 a_2;
    float3 alpha_0;
    if(_S103)
    {
        float4 _S111 = float4(isl_2.inertia0_0) ;
        float4 _S112 = float4(isl_2.inertia1_0) ;
        float4 _S113 = float4(isl_2.inertia2_0) ;
        thread Quat_0 _S114 = _S94;
        float3 _S115 = world_mul_0(&_S114, _S111, _S112, _S113, w_3);
        float4 _S116 = float4(isl_2.inv0_0) ;
        float4 _S117 = float4(isl_2.inv1_0) ;
        float4 _S118 = float4(isl_2.inv2_0) ;
        float3 _S119 = (*(&kernelContext_3)->g_t_0)[int(0)] - cross(w_3, _S115);
        thread Quat_0 _S120 = _S94;
        float3 _S121 = world_mul_0(&_S120, _S116, _S117, _S118, _S119);
        a_2 = (*(&kernelContext_3)->g_f_0)[int(0)] / float3(_S97.w) ;
        alpha_0 = _S121;
    }
    else
    {
        a_2 = _S102;
        alpha_0 = _S102;
    }
    bool _S122 = ((&kernelContext_3)->params_0->rml_0) != 0U;
    thread Quat_0 _S123 = _S94;
    float3 _S124 = inverse_rotate_0(&_S123, w_3);
    uint4 _S125 = uint4(isl_2.range_0) ;
    uint c_2 = _S125.x + tid_0;
    for(;;)
    {
        if(c_2 < (_S125.y))
        {
        }
        else
        {
            break;
        }
        ChunkStatic_natural_0 device* _S126 = (&kernelContext_3)->chunks_0+c_2;
        float4 _S127 = float4(_S126->center_0) ;
        float mass_0 = _S127.w;
        uint _S128 = 4U * c_2;
        float3 _S129 = _S127.xyz + (float4(*((&kernelContext_3)->state_0+_S128)) ).xyz - _S98;
        thread Quat_0 _S130 = _S94;
        float3 _S131 = rotate_0(&_S130, _S129);
        uint _S132 = 2U * c_2;
        float3 _S133 = float3(mass_0) ;
        float3 f_world_0 = (float4(*((&kernelContext_3)->ctab_0+((&kernelContext_3)->params_0->eload_base_0 + _S132))) ).xyz + _S95 * _S133;
        float3 t_world_0 = (float4(*((&kernelContext_3)->ctab_0+((&kernelContext_3)->params_0->eload_base_0 + _S132 + 1U))) ).xyz;
        float3 f_world_1;
        float3 t_world_1;
        if(_S122)
        {
            float3 f_world_2 = f_world_0 - (a_2 + cross(alpha_0, _S131) + cross(w_3, cross(w_3, _S131))) * _S133;
            float4 _S134 = float4(_S126->inertia0_1) ;
            float4 _S135 = float4(_S126->inertia1_1) ;
            float4 _S136 = float4(_S126->inertia2_1) ;
            thread Quat_0 _S137 = _S94;
            float3 _S138 = world_mul_0(&_S137, _S134, _S135, _S136, alpha_0);
            thread Quat_0 _S139 = _S94;
            float3 _S140 = world_mul_0(&_S139, _S134, _S135, _S136, w_3);
            float3 t_world_2 = t_world_0 - (_S138 + cross(w_3, _S140));
            f_world_1 = f_world_2;
            t_world_1 = t_world_2;
        }
        else
        {
            f_world_1 = f_world_0;
            t_world_1 = t_world_0;
        }
        thread Quat_0 _S141 = _S94;
        float3 _S142 = inverse_rotate_0(&_S141, f_world_1);
        thread Quat_0 _S143 = _S94;
        float3 _S144 = inverse_rotate_0(&_S143, t_world_1);
        float3 fb_0;
        float3 mo_0;
        if(_S122)
        {
            float3 sw_0 = (float4(*((&kernelContext_3)->state_0+(_S128 + 3U))) ).xyz;
            float4 _S145 = float4(_S126->inertia0_1) ;
            float4 _S146 = float4(_S126->inertia1_1) ;
            float4 _S147 = float4(_S126->inertia2_1) ;
            float3 _S148 = mat_mul_0(_S145, _S146, _S147, sw_0);
            float3 mo_1 = _S144 - (cross(_S124, _S148) + cross(sw_0, mat_mul_0(_S145, _S146, _S147, _S124)) + cross(sw_0, _S148));
            fb_0 = _S142 - cross(_S124, (float4(*((&kernelContext_3)->state_0+(_S128 + 2U))) ).xyz) * float3((2.0f * mass_0)) ;
            mo_0 = mo_1;
        }
        else
        {
            fb_0 = _S142;
            mo_0 = _S144;
        }
        *((&kernelContext_3)->scratch_0+((&kernelContext_3)->params_0->sload_base_0 + _S132)) = packed_float4(float4(fb_0, 0.0f)) ;
        *((&kernelContext_3)->scratch_0+((&kernelContext_3)->params_0->sload_base_0 + _S132 + 1U)) = packed_float4(float4(mo_0, 0.0f)) ;
        c_2 = c_2 + 256U;
    }
    return;
}

