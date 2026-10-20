#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
uint group_uniform_0(uint value_0, uint thread_0)
{
    return value_0;
}

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

float3 rotate_0(const Quat_0 thread* q_2, float3 v_1)
{
    float3 qv_0 = float3(q_2->x_0, q_2->y_0, q_2->z_0);
    float3 t_0 = cross(qv_0, v_1) * float3(2.0f) ;
    return v_1 + t_0 * float3(q_2->w_0)  + cross(qv_0, t_0);
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

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

struct Strike_0
{
    Island_0 isl_0;
    uint n_1;
    float3 origin_0;
    Quat_0 q_3;
    float3 hit_0;
    float3 d_0;
    float speed_0;
    float face_0;
    float m_imp_0;
    float e_imp_0;
    uint entry_0;
    uint entry_k_0;
    float entry_depth_0;
    float3 from_0;
    float3 rel_0;
    float radius_0;
};

struct PunchParams_0
{
    uint island_count_0;
    uint impactor_count_0;
    uint engine_base_0;
    uint pad0_0;
    float dt_0;
    float speed_1;
    float pad1_0;
    float pad2_0;
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

struct JointBond_natural_0
{
    packed_float4 geom0_0;
    packed_float4 geom1_0;
    packed_float4 stiff0_0;
    packed_float4 stiff1_0;
    packed_float4 rebar0_0;
    packed_float4 rebar1_0;
    packed_uint4 ids_0;
};

struct BondStatic_natural_0
{
    packed_float4 t1_0;
    packed_float4 t2_0;
    packed_float4 normal_0;
    packed_float4 ra_0;
    packed_float4 rb_0;
    packed_float4 centroid_0;
    packed_float4 c_lin_0;
    packed_float4 c_ang_0;
    JointBond_natural_0 law_0;
};

struct JointState_0
{
    float damage_0;
    float crush_0;
    float kappa_0;
    float kappa_c_0;
    float ductility_0;
    float ductility_c_0;
    float life_0;
    float plastic_x_0;
    float plastic_y_0;
    float plastic_t_0;
    float rebar_plastic_0;
    float rebar_slip0_0;
    float rebar_slip1_0;
    float rebar_work_0;
    float rebar_broken_0;
    float strain_rate_0;
    float governing_stress_0;
    float dissipated_0;
    float utilization_0;
    uint mode_0;
};

struct BondDyn_natural_0
{
    JointState_0 js_0;
    packed_float4 force_lin_0;
    packed_float4 force_ang_0;
    packed_float4 sums_0;
    packed_float4 comps_0;
    packed_uint4 events_0;
};

struct JointMaterial_0
{
    float4 strength_0;
    float4 energy_1;
    float4 dif_0;
    float4 misc_0;
    uint4 kind_flags_0;
};

struct MaterialTable_0
{
    array<JointMaterial_0, int(64)> m_0;
};

struct KernelContext_0
{
    PunchParams_0 constant* params_0;
    packed_float4 device* fast_0;
    uint device* status_0;
    Island_natural_0 device* islands_0;
    packed_float4 device* ctab_0;
    EngineBody_natural_0 device* bodies_0;
    ChunkStatic_natural_0 device* chunks_0;
    packed_float4 device* state_0;
    packed_float4 device* impactors_0;
    BondStatic_natural_0 device* bond_static_0;
    BondDyn_natural_0 device* bond_dyn_0;
    MaterialTable_0 constant* materials_0;
    uint device* route_0;
    array<float3, int(256)> threadgroup* g_lo_0;
    array<float3, int(256)> threadgroup* g_hi_0;
    array<float, int(256)> threadgroup* g_a_0;
    array<uint, int(256)> threadgroup* g_idx_0;
    array<uint, int(4096)> threadgroup* g_plug_0;
    uint threadgroup* g_changed_0;
    array<float, int(256)> threadgroup* g_b_0;
    uint threadgroup* g_go_0;
    array<uint, int(1024)> threadgroup* g_scan_0;
};

bool find_strike_0(uint i_0, uint t_1, uint max_chunks_0, Strike_0 thread* st_0, KernelContext_0 thread* kernelContext_0)
{
    uint ax_0;
    float r_3;
    float t0_0;
    Island_natural_0 _S1 = kernelContext_0->islands_0[i_0];
    uint4 _S2 = uint4(_S1.info_0) ;
    float4 _S3 = float4(_S1.com_0) ;
    float4 _S4 = float4(_S1.inertia0_0) ;
    float4 _S5 = float4(_S1.inertia1_0) ;
    float4 _S6 = float4(_S1.inertia2_0) ;
    float4 _S7 = float4(_S1.inv0_0) ;
    float4 _S8 = float4(_S1.inv1_0) ;
    float4 _S9 = float4(_S1.inv2_0) ;
    float4 _S10 = float4(_S1.wcom_0) ;
    float4 _S11 = float4(_S1.winv0_0) ;
    float4 _S12 = float4(_S1.winv1_0) ;
    float4 _S13 = float4(_S1.winv2_0) ;
    float4 _S14 = float4(_S1.rotation_0) ;
    float4 _S15 = float4(_S1.position_0) ;
    float4 _S16 = float4(_S1.position_err_0) ;
    float4 _S17 = float4(_S1.velocity_0) ;
    float4 _S18 = float4(_S1.velocity_err_0) ;
    float4 _S19 = float4(_S1.angular_velocity_0) ;
    uint4 _S20 = uint4(_S1.probes_0) ;
    float4 _S21 = float4(_S1.energy_0) ;
    float4 _S22 = float4(_S1.rotation_err_0) ;
    float4 _S23 = float4(_S1.momentum_0) ;
    float4 _S24 = float4(_S1.momentum_err_0) ;
    (&st_0->isl_0)->range_0 = uint4(_S1.range_0) ;
    (&st_0->isl_0)->info_0 = _S2;
    (&st_0->isl_0)->com_0 = _S3;
    (&st_0->isl_0)->inertia0_0 = _S4;
    (&st_0->isl_0)->inertia1_0 = _S5;
    (&st_0->isl_0)->inertia2_0 = _S6;
    (&st_0->isl_0)->inv0_0 = _S7;
    (&st_0->isl_0)->inv1_0 = _S8;
    (&st_0->isl_0)->inv2_0 = _S9;
    (&st_0->isl_0)->wcom_0 = _S10;
    (&st_0->isl_0)->winv0_0 = _S11;
    (&st_0->isl_0)->winv1_0 = _S12;
    (&st_0->isl_0)->winv2_0 = _S13;
    (&st_0->isl_0)->rotation_0 = _S14;
    (&st_0->isl_0)->position_0 = _S15;
    (&st_0->isl_0)->position_err_0 = _S16;
    (&st_0->isl_0)->velocity_0 = _S17;
    (&st_0->isl_0)->velocity_err_0 = _S18;
    (&st_0->isl_0)->angular_velocity_0 = _S19;
    (&st_0->isl_0)->done_0 = _S1.done_0;
    (&st_0->isl_0)->coverage_0 = _S1.coverage_0;
    (&st_0->isl_0)->wide_groups_0 = _S1.wide_groups_0;
    (&st_0->isl_0)->done_pad_0 = _S1.done_pad_0;
    (&st_0->isl_0)->probes_0 = _S20;
    (&st_0->isl_0)->energy_0 = _S21;
    (&st_0->isl_0)->rotation_err_0 = _S22;
    (&st_0->isl_0)->momentum_0 = _S23;
    (&st_0->isl_0)->momentum_err_0 = _S24;
    st_0->n_1 = 0U;
    float3 _S25 = float3(0.0f) ;
    st_0->origin_0 = _S25;
    st_0->q_3 = quat_of_0(float4(0.0f, 0.0f, 0.0f, 1.0f));
    st_0->hit_0 = _S25;
    st_0->d_0 = _S25;
    st_0->speed_0 = 0.0f;
    st_0->face_0 = 0.0f;
    st_0->m_imp_0 = 0.0f;
    st_0->e_imp_0 = 0.0f;
    st_0->entry_0 = 4294967295U;
    st_0->entry_k_0 = 4294967295U;
    st_0->entry_depth_0 = 0.0f;
    st_0->from_0 = _S25;
    st_0->rel_0 = _S25;
    st_0->radius_0 = 0.0f;
    Island_natural_0 isl_1 = kernelContext_0->islands_0[i_0];
    uint4 _S26 = uint4(isl_1.range_0) ;
    uint4 _S27 = uint4(isl_1.info_0) ;
    float4 _S28 = float4(isl_1.com_0) ;
    float4 _S29 = float4(isl_1.inertia0_0) ;
    float4 _S30 = float4(isl_1.inertia1_0) ;
    float4 _S31 = float4(isl_1.inertia2_0) ;
    float4 _S32 = float4(isl_1.inv0_0) ;
    float4 _S33 = float4(isl_1.inv1_0) ;
    float4 _S34 = float4(isl_1.inv2_0) ;
    float4 _S35 = float4(isl_1.wcom_0) ;
    float4 _S36 = float4(isl_1.winv0_0) ;
    float4 _S37 = float4(isl_1.winv1_0) ;
    float4 _S38 = float4(isl_1.winv2_0) ;
    float4 _S39 = float4(isl_1.rotation_0) ;
    float4 _S40 = float4(isl_1.position_0) ;
    float4 _S41 = float4(isl_1.position_err_0) ;
    float4 _S42 = float4(isl_1.velocity_0) ;
    float4 _S43 = float4(isl_1.velocity_err_0) ;
    float4 _S44 = float4(isl_1.angular_velocity_0) ;
    uint4 _S45 = uint4(isl_1.probes_0) ;
    float4 _S46 = float4(isl_1.energy_0) ;
    float4 _S47 = float4(isl_1.rotation_err_0) ;
    float4 _S48 = float4(isl_1.momentum_0) ;
    float4 _S49 = float4(isl_1.momentum_err_0) ;
    uint _S50 = _S26.x;
    uint n_2 = group_uniform_0(_S26.y - _S50, t_1);
    bool _S51;
    if((_S26.z) >= (_S26.w))
    {
        _S51 = true;
    }
    else
    {
        _S51 = n_2 > max_chunks_0;
    }
    if(_S51)
    {
        _S51 = true;
    }
    else
    {
        _S51 = (kernelContext_0->params_0->impactor_count_0) == 0U;
    }
    uint c_0;
    if(_S51)
    {
        c_0 = 1U;
    }
    else
    {
        c_0 = 0U;
    }
    if((group_uniform_0(c_0, t_1)) != 0U)
    {
        return false;
    }
    uint _S52 = 3U * i_0;
    uint body_0 = (as_type<uint>(((float4(*(kernelContext_0->ctab_0+(kernelContext_0->params_0->engine_base_0 + _S52))) ).x)));
    Quat_0 _S53 = quat_of_0(_S39);
    float3 _S54 = _S40.xyz;
    float3 _S55 = _S42.xyz;
    float3 origin_1;
    float3 v_2;
    Quat_0 q_4;
    if(body_0 != 4294967295U)
    {
        Quat_0 p_inv_0 = quat_of_0(float4(*(kernelContext_0->ctab_0+(kernelContext_0->params_0->engine_base_0 + _S52 + 1U))) );
        float3 com_1 = (float4(*(kernelContext_0->ctab_0+(kernelContext_0->params_0->engine_base_0 + _S52 + 2U))) ).xyz;
        EngineBody_natural_0 b_0 = kernelContext_0->bodies_0[body_0];
        thread Quat_0 a_inv_0;
        (&a_inv_0)->w_0 = 0.70710676908493042f;
        (&a_inv_0)->x_0 = 0.70710676908493042f;
        (&a_inv_0)->y_0 = 0.0f;
        (&a_inv_0)->z_0 = 0.0f;
        Quat_0 _S56 = quat_of_0(float4(b_0.rot_0) );
        thread Quat_0 _S57 = a_inv_0;
        thread Quat_0 _S58 = _S56;
        Quat_0 _S59 = quat_mul_0(&_S57, &_S58);
        thread Quat_0 _S60 = _S59;
        thread Quat_0 _S61 = p_inv_0;
        Quat_0 _S62 = quat_mul_0(&_S60, &_S61);
        thread Quat_0 _S63 = _S62;
        Quat_0 _S64 = normalized_0(&_S63);
        float3 _S65 = to_zup_0((float4(b_0.pos_inv_mass_0) ).xyz);
        thread Quat_0 _S66 = _S64;
        float3 _S67 = rotate_0(&_S66, com_1);
        float3 _S68 = to_zup_0((float4(b_0.vel_sleep_0) ).xyz);
        origin_1 = _S65 - _S67;
        q_4 = _S64;
        v_2 = _S68;
    }
    else
    {
        origin_1 = _S54;
        q_4 = _S53;
        v_2 = _S55;
    }
    float inf_0 = float_bits_0(2139095040U);
    float3 _S69 = float3(- inf_0) ;
    float3 lo_0 = float3(inf_0) ;
    float3 hi_0 = _S69;
    c_0 = t_1;
    for(;;)
    {
        if(c_0 < n_2)
        {
        }
        else
        {
            break;
        }
        uint k_0 = _S50 + c_0;
        float3 _S70 = (float4((kernelContext_0->chunks_0+k_0)->center_0) ).xyz + (float4(*(kernelContext_0->state_0+4U * k_0)) ).xyz;
        thread Quat_0 _S71 = q_4;
        float3 _S72 = rotate_0(&_S71, _S70);
        float3 p_0 = origin_1 + _S72;
        float3 _S73 = min(lo_0, p_0 - float3((float4((kernelContext_0->chunks_0+k_0)->half_0) ).w) );
        float3 _S74 = max(hi_0, p_0 + float3((float4((kernelContext_0->chunks_0+k_0)->half_0) ).w) );
        uint c_1 = c_0 + 256U;
        lo_0 = _S73;
        hi_0 = _S74;
        c_0 = c_1;
    }
    (*kernelContext_0->g_lo_0)[t_1] = lo_0;
    (*kernelContext_0->g_hi_0)[t_1] = hi_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s_0 = 128U;
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
            uint _S75 = t_1 + s_0;
            (*kernelContext_0->g_lo_0)[t_1] = min((*kernelContext_0->g_lo_0)[t_1], (*kernelContext_0->g_lo_0)[_S75]);
            (*kernelContext_0->g_hi_0)[t_1] = max((*kernelContext_0->g_hi_0)[t_1], (*kernelContext_0->g_hi_0)[_S75]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    float3 _S76 = (*kernelContext_0->g_lo_0)[int(0)];
    float3 _S77 = (*kernelContext_0->g_hi_0)[int(0)];
    uint _S78 = (as_type<uint>(((float4(*(kernelContext_0->fast_0+int(0))) ).x)));
    uint hit_k_0 = 4294967295U;
    float3 hit_1 = _S25;
    float3 d_1 = _S25;
    float speed_2 = 0.0f;
    float face_1 = 0.0f;
    float m_imp_1 = 0.0f;
    float e_imp_1 = 0.0f;
    float3 hit_c_0 = _S25;
    float3 hit_rel_0 = _S25;
    float p_radius_0 = 0.0f;
    uint k_1 = 0U;
    for(;;)
    {
        if(k_1 < (kernelContext_0->params_0->impactor_count_0 + _S78))
        {
            _S51 = hit_k_0 == 4294967295U;
        }
        else
        {
            _S51 = false;
        }
        if(_S51)
        {
        }
        else
        {
            break;
        }
        float3 rel_1;
        float3 c_2;
        float4 info_2;
        if(k_1 < (kernelContext_0->params_0->impactor_count_0))
        {
            uint _S79 = 2U * k_1;
            float4 _S80 = float4(*(kernelContext_0->impactors_0+_S79)) ;
            EngineBody_natural_0 b_1 = kernelContext_0->bodies_0[(as_type<uint>((_S80.x)))];
            float3 _S81 = to_zup_0((float4(b_1.pos_inv_mass_0) ).xyz);
            float _S82 = _S80.y;
            float4 _S83 = float4(_S80.z, _S80.w, (float4(*(kernelContext_0->impactors_0+(_S79 + 1U))) ).x, 0.0f);
            rel_1 = to_zup_0((float4(b_1.vel_sleep_0) ).xyz) - v_2;
            r_3 = _S82;
            c_2 = _S81;
            info_2 = _S83;
        }
        else
        {
            uint f_0 = 1U + 3U * (k_1 - kernelContext_0->params_0->impactor_count_0);
            uint _S84 = f_0 + 1U;
            if((as_type<uint>(((float4(*(kernelContext_0->fast_0+_S84)) ).w))) == i_0)
            {
                k_1 = k_1 + 1U;
                continue;
            }
            float3 _S85 = (float4(*(kernelContext_0->fast_0+f_0)) ).xyz;
            float _S86 = (float4(*(kernelContext_0->fast_0+f_0)) ).w;
            float4 _S87 = float4(*(kernelContext_0->fast_0+(f_0 + 2U))) ;
            rel_1 = (float4(*(kernelContext_0->fast_0+_S84)) ).xyz - v_2;
            r_3 = _S86;
            c_2 = _S85;
            info_2 = _S87;
        }
        float sp_0 = length(rel_1);
        if(!(sp_0 > (kernelContext_0->params_0->speed_1)))
        {
            k_1 = k_1 + 1U;
            continue;
        }
        float3 _S88 = float3(r_3) ;
        float3 _S89 = _S76 - _S88;
        float3 _S90 = _S77 + _S88;
        float _S91 = kernelContext_0->params_0->dt_0;
        float t0_1 = 0.0f;
        float t1_1 = _S91;
        ax_0 = 0U;
        for(;;)
        {
            if(ax_0 < 3U)
            {
            }
            else
            {
                break;
            }
            float3 _S92 = rel_1;
            uint _S93 = ax_0;
            if((abs(rel_1[ax_0])) < 9.999999960041972e-13f)
            {
                float3 _S94 = c_2;
                uint _S95 = ax_0;
                bool _S96;
                if((c_2[ax_0]) < (_S89[ax_0]))
                {
                    _S96 = true;
                }
                else
                {
                    _S96 = (_S94[_S95]) > (_S90[ax_0]);
                }
                if(_S96)
                {
                    t0_0 = inf_0;
                }
                else
                {
                    t0_0 = t0_1;
                }
                t0_1 = t0_0;
                ax_0 = ax_0 + 1U;
                continue;
            }
            float ta_0 = (_S89[ax_0] - c_2[ax_0]) / _S92[_S93];
            float tb_0 = (_S90[ax_0] - c_2[ax_0]) / _S92[_S93];
            float _S97 = min(t1_1, max(ta_0, tb_0));
            t0_1 = max(t0_1, min(ta_0, tb_0));
            t1_1 = _S97;
            ax_0 = ax_0 + 1U;
        }
        uint hit_k_1;
        float3 hit_2;
        float3 d_2;
        float3 hit_c_1;
        float3 hit_rel_1;
        float face_2;
        float m_imp_2;
        float e_imp_2;
        float p_radius_1;
        if(t0_1 <= t1_1)
        {
            float3 _S98 = c_2 + rel_1 * float3(t0_1) ;
            float3 _S99 = rel_1 / float3(sp_0) ;
            float _S100 = info_2.x;
            float _S101 = info_2.y;
            float _S102 = info_2.z;
            hit_k_1 = k_1;
            hit_2 = _S98;
            d_2 = _S99;
            t0_0 = sp_0;
            face_2 = _S100;
            m_imp_2 = _S101;
            e_imp_2 = _S102;
            hit_c_1 = c_2;
            hit_rel_1 = rel_1;
            p_radius_1 = r_3;
        }
        else
        {
            hit_k_1 = hit_k_0;
            hit_2 = hit_1;
            d_2 = d_1;
            t0_0 = speed_2;
            face_2 = face_1;
            m_imp_2 = m_imp_1;
            e_imp_2 = e_imp_1;
            hit_c_1 = hit_c_0;
            hit_rel_1 = hit_rel_0;
            p_radius_1 = p_radius_0;
        }
        hit_k_0 = hit_k_1;
        hit_1 = hit_2;
        d_1 = d_2;
        speed_2 = t0_0;
        face_1 = face_2;
        m_imp_1 = m_imp_2;
        e_imp_1 = e_imp_2;
        hit_c_0 = hit_c_1;
        hit_rel_0 = hit_rel_1;
        p_radius_0 = p_radius_1;
        k_1 = k_1 + 1U;
    }
    if(hit_k_0 == 4294967295U)
    {
        c_0 = 1U;
    }
    else
    {
        c_0 = 0U;
    }
    if((group_uniform_0(c_0, t_1)) != 0U)
    {
        return false;
    }
    r_3 = inf_0;
    ax_0 = 4294967295U;
    c_0 = t_1;
    for(;;)
    {
        if(c_0 < n_2)
        {
        }
        else
        {
            break;
        }
        uint k_2 = _S50 + c_0;
        float3 _S103 = (float4((kernelContext_0->chunks_0+k_2)->center_0) ).xyz + (float4(*(kernelContext_0->state_0+4U * k_2)) ).xyz;
        thread Quat_0 _S104 = q_4;
        float3 _S105 = rotate_0(&_S104, _S103);
        float3 r_4 = origin_1 + _S105 - hit_1;
        float along_0 = dot(r_4, d_1);
        float dist_0 = length(r_4 - d_1 * float3(along_0) ) + abs(along_0) * 0.00100000004749745f;
        if(dist_0 < r_3)
        {
            r_3 = dist_0;
            ax_0 = c_0;
        }
        c_0 = c_0 + 256U;
    }
    (*kernelContext_0->g_a_0)[t_1] = r_3;
    (*kernelContext_0->g_idx_0)[t_1] = ax_0;
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
        if(t_1 < s_0)
        {
            uint _S106 = t_1 + s_0;
            if(((*kernelContext_0->g_a_0)[_S106]) < (*kernelContext_0->g_a_0)[t_1])
            {
                _S51 = true;
            }
            else
            {
                if(((*kernelContext_0->g_a_0)[_S106]) == (*kernelContext_0->g_a_0)[t_1])
                {
                    _S51 = ((*kernelContext_0->g_idx_0)[_S106]) < (*kernelContext_0->g_idx_0)[t_1];
                }
                else
                {
                    _S51 = false;
                }
            }
        }
        else
        {
            _S51 = false;
        }
        if(_S51)
        {
            uint _S107 = t_1 + s_0;
            (*kernelContext_0->g_a_0)[t_1] = (*kernelContext_0->g_a_0)[_S107];
            (*kernelContext_0->g_idx_0)[t_1] = (*kernelContext_0->g_idx_0)[_S107];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    uint entry_1 = (*kernelContext_0->g_idx_0)[int(0)];
    uint entry_k_1 = _S50 + (*kernelContext_0->g_idx_0)[int(0)];
    float3 _S108 = (float4((kernelContext_0->chunks_0+entry_k_1)->center_0) ).xyz + (float4(*(kernelContext_0->state_0+4U * entry_k_1)) ).xyz;
    thread Quat_0 _S109 = q_4;
    float3 _S110 = rotate_0(&_S109, _S108);
    float entry_depth_1 = dot(origin_1 + _S110 - hit_1, d_1);
    (&st_0->isl_0)->range_0 = _S26;
    (&st_0->isl_0)->info_0 = _S27;
    (&st_0->isl_0)->com_0 = _S28;
    (&st_0->isl_0)->inertia0_0 = _S29;
    (&st_0->isl_0)->inertia1_0 = _S30;
    (&st_0->isl_0)->inertia2_0 = _S31;
    (&st_0->isl_0)->inv0_0 = _S32;
    (&st_0->isl_0)->inv1_0 = _S33;
    (&st_0->isl_0)->inv2_0 = _S34;
    (&st_0->isl_0)->wcom_0 = _S35;
    (&st_0->isl_0)->winv0_0 = _S36;
    (&st_0->isl_0)->winv1_0 = _S37;
    (&st_0->isl_0)->winv2_0 = _S38;
    (&st_0->isl_0)->rotation_0 = _S39;
    (&st_0->isl_0)->position_0 = _S40;
    (&st_0->isl_0)->position_err_0 = _S41;
    (&st_0->isl_0)->velocity_0 = _S42;
    (&st_0->isl_0)->velocity_err_0 = _S43;
    (&st_0->isl_0)->angular_velocity_0 = _S44;
    (&st_0->isl_0)->done_0 = isl_1.done_0;
    (&st_0->isl_0)->coverage_0 = isl_1.coverage_0;
    (&st_0->isl_0)->wide_groups_0 = isl_1.wide_groups_0;
    (&st_0->isl_0)->done_pad_0 = isl_1.done_pad_0;
    (&st_0->isl_0)->probes_0 = _S45;
    (&st_0->isl_0)->energy_0 = _S46;
    (&st_0->isl_0)->rotation_err_0 = _S47;
    (&st_0->isl_0)->momentum_0 = _S48;
    (&st_0->isl_0)->momentum_err_0 = _S49;
    st_0->n_1 = n_2;
    st_0->origin_0 = origin_1;
    st_0->q_3 = q_4;
    st_0->hit_0 = hit_1;
    st_0->d_0 = d_1;
    st_0->speed_0 = speed_2;
    st_0->face_0 = face_1;
    st_0->m_imp_0 = m_imp_1;
    st_0->e_imp_0 = e_imp_1;
    st_0->entry_0 = entry_1;
    st_0->entry_k_0 = entry_k_1;
    st_0->entry_depth_0 = entry_depth_1;
    st_0->from_0 = hit_c_0;
    st_0->rel_0 = hit_rel_0;
    st_0->radius_0 = p_radius_0;
    return true;
}

bool connected_0(const JointState_0 thread* st_1, bool has_rebar_0)
{
    bool _S111;
    if((st_1->damage_0) < 1.0f)
    {
        _S111 = true;
    }
    else
    {
        if(has_rebar_0)
        {
            _S111 = (st_1->rebar_broken_0) == 0.0f;
        }
        else
        {
            _S111 = false;
        }
    }
    return _S111;
}

struct BondDyn_0
{
    JointState_0 js_0;
    float4 force_lin_0;
    float4 force_ang_0;
    float4 sums_0;
    float4 comps_0;
    uint4 events_0;
};

[[kernel]] void punch_islands(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], PunchParams_0 constant* params_1 [[buffer(0)]], packed_float4 device* fast_1 [[buffer(11)]], uint device* status_1 [[buffer(10)]], Island_natural_0 device* islands_1 [[buffer(1)]], packed_float4 device* ctab_1 [[buffer(4)]], EngineBody_natural_0 device* bodies_1 [[buffer(5)]], ChunkStatic_natural_0 device* chunks_1 [[buffer(2)]], packed_float4 device* state_1 [[buffer(3)]], packed_float4 device* impactors_1 [[buffer(6)]], BondStatic_natural_0 device* bond_static_1 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_1 [[buffer(8)]], MaterialTable_0 constant* materials_1 [[buffer(9)]], uint device* route_1 [[buffer(12)]])
{
    uint iter_0;
    uint bi_0;
    uint s_1;
    bool out_0;
    uint _S112;
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->params_0 = params_1;
    (&kernelContext_1)->fast_0 = fast_1;
    (&kernelContext_1)->status_0 = status_1;
    (&kernelContext_1)->islands_0 = islands_1;
    (&kernelContext_1)->ctab_0 = ctab_1;
    (&kernelContext_1)->bodies_0 = bodies_1;
    (&kernelContext_1)->chunks_0 = chunks_1;
    (&kernelContext_1)->state_0 = state_1;
    (&kernelContext_1)->impactors_0 = impactors_1;
    (&kernelContext_1)->bond_static_0 = bond_static_1;
    (&kernelContext_1)->bond_dyn_0 = bond_dyn_1;
    (&kernelContext_1)->materials_0 = materials_1;
    (&kernelContext_1)->route_0 = route_1;
    threadgroup array<float3, int(256)> g_lo_1;
    (&kernelContext_1)->g_lo_0 = &g_lo_1;
    threadgroup array<float3, int(256)> g_hi_1;
    (&kernelContext_1)->g_hi_0 = &g_hi_1;
    threadgroup array<float, int(256)> g_a_1;
    (&kernelContext_1)->g_a_0 = &g_a_1;
    threadgroup array<uint, int(256)> g_idx_1;
    (&kernelContext_1)->g_idx_0 = &g_idx_1;
    threadgroup array<uint, int(4096)> g_plug_1;
    (&kernelContext_1)->g_plug_0 = &g_plug_1;
    threadgroup uint g_changed_1;
    (&kernelContext_1)->g_changed_0 = &g_changed_1;
    threadgroup array<float, int(256)> g_b_1;
    (&kernelContext_1)->g_b_0 = &g_b_1;
    threadgroup uint g_go_1;
    (&kernelContext_1)->g_go_0 = &g_go_1;
    threadgroup array<uint, int(1024)> g_scan_1;
    (&kernelContext_1)->g_scan_0 = &g_scan_1;
    uint i_1 = group_0.x;
    uint t_2 = thread_1.x;
    if(i_1 >= (params_1->island_count_0))
    {
        return;
    }
    if((group_uniform_0((as_type<uint>(((float4(*((&kernelContext_1)->fast_0+int(0))) ).y))), t_2)) == 0U)
    {
        if(t_2 == 0U)
        {
            uint device* _S113 = (&kernelContext_1)->status_0+(16U + i_1);
            *_S113 = (*_S113) & 4294967287U;
        }
        return;
    }
    thread Strike_0 st_2;
    bool _S114 = find_strike_0(i_1, t_2, 4096U, &st_2, &kernelContext_1);
    uint c_3;
    if(_S114)
    {
        c_3 = 1U;
    }
    else
    {
        c_3 = 0U;
    }
    if((group_uniform_0(c_3, t_2)) == 0U)
    {
        if(t_2 == 0U)
        {
            uint device* _S115 = (&kernelContext_1)->status_0+(16U + i_1);
            *_S115 = (*_S115) & 4294967287U;
        }
        return;
    }
    Island_0 isl_2 = (&st_2)->isl_0;
    uint n_3 = group_uniform_0((&st_2)->n_1, t_2);
    float3 _S116 = (&st_2)->origin_0;
    Quat_0 _S117 = (&st_2)->q_3;
    float3 _S118 = (&st_2)->hit_0;
    float3 _S119 = (&st_2)->d_0;
    float speed_3 = (&st_2)->speed_0;
    float face_3 = (&st_2)->face_0;
    float m_imp_3 = (&st_2)->m_imp_0;
    float e_imp_3 = (&st_2)->e_imp_0;
    uint _S120 = (&st_2)->entry_0;
    uint entry_k_2 = (&st_2)->entry_k_0;
    float _S121 = (&st_2)->entry_depth_0;
    c_3 = t_2;
    for(;;)
    {
        if(c_3 < n_3)
        {
        }
        else
        {
            break;
        }
        if(c_3 == _S120)
        {
            iter_0 = 1U;
        }
        else
        {
            iter_0 = 0U;
        }
        (*(&kernelContext_1)->g_plug_0)[c_3] = iter_0;
        c_3 = c_3 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    iter_0 = 0U;
    for(;;)
    {
        if(iter_0 < n_3)
        {
        }
        else
        {
            break;
        }
        if(t_2 == 0U)
        {
            *(&kernelContext_1)->g_changed_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bi_0 = isl_2.range_0.z + t_2;
        for(;;)
        {
            if(bi_0 < (isl_2.range_0.w))
            {
            }
            else
            {
                break;
            }
            JointBond_natural_0 law_1 = ((&kernelContext_1)->bond_static_0+bi_0)->law_0;
            bool _S122 = ((float4(((&kernelContext_1)->bond_static_0+bi_0)->law_0.stiff1_0) ).w) != 0.0f;
            thread JointState_0 _S123 = ((&kernelContext_1)->bond_dyn_0+bi_0)->js_0;
            bool _S124 = connected_0(&_S123, _S122);
            if(!_S124)
            {
                bi_0 = bi_0 + 256U;
                continue;
            }
            uint4 _S125 = uint4(law_1.ids_0) ;
            uint _S126 = isl_2.range_0.x;
            uint ca_0 = _S125.y - _S126;
            uint cb_0 = _S125.z - _S126;
            if(((*(&kernelContext_1)->g_plug_0)[ca_0]) == (*(&kernelContext_1)->g_plug_0)[cb_0])
            {
                bi_0 = bi_0 + 256U;
                continue;
            }
            if(((*(&kernelContext_1)->g_plug_0)[ca_0]) != 0U)
            {
                s_1 = cb_0;
            }
            else
            {
                s_1 = ca_0;
            }
            uint ok_0 = _S126 + s_1;
            float3 _S127 = (float4(((&kernelContext_1)->chunks_0+ok_0)->center_0) ).xyz + (float4(*((&kernelContext_1)->state_0+4U * ok_0)) ).xyz;
            thread Quat_0 _S128 = _S117;
            float3 _S129 = rotate_0(&_S128, _S127);
            float3 r_5 = _S116 + _S129 - _S118;
            float along_1 = dot(r_5, _S119);
            float lateral_0 = length(r_5 - _S119 * float3(along_1) );
            if(along_1 >= (_S121 - (float4(((&kernelContext_1)->chunks_0+ok_0)->half_0) ).w))
            {
                out_0 = lateral_0 <= (face_3 + max(along_1 - _S121, 0.0f));
            }
            else
            {
                out_0 = false;
            }
            if(out_0)
            {
                (*(&kernelContext_1)->g_plug_0)[s_1] = 1U;
                *(&kernelContext_1)->g_changed_0 = 1U;
            }
            bi_0 = bi_0 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if((group_uniform_0(*(&kernelContext_1)->g_changed_0, t_2)) == 0U)
        {
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        iter_0 = iter_0 + 1U;
    }
    c_3 = t_2;
    float mass_0 = 0.0f;
    float count_0 = 0.0f;
    for(;;)
    {
        if(c_3 < n_3)
        {
        }
        else
        {
            break;
        }
        if(((*(&kernelContext_1)->g_plug_0)[c_3]) != 0U)
        {
            float count_1 = count_0 + 1.0f;
            mass_0 = mass_0 + (float4(((&kernelContext_1)->chunks_0+(isl_2.range_0.x + c_3))->center_0) ).w;
            count_0 = count_1;
        }
        c_3 = c_3 + 256U;
    }
    uint _S130 = isl_2.range_0.z + t_2;
    bi_0 = _S130;
    float capacity_0 = 0.0f;
    for(;;)
    {
        uint _S131 = isl_2.range_0.w;
        _S112 = _S131;
        if(bi_0 < _S131)
        {
        }
        else
        {
            break;
        }
        JointBond_natural_0 law_2 = ((&kernelContext_1)->bond_static_0+bi_0)->law_0;
        bool _S132 = ((float4(((&kernelContext_1)->bond_static_0+bi_0)->law_0.stiff1_0) ).w) != 0.0f;
        thread JointState_0 _S133 = ((&kernelContext_1)->bond_dyn_0+bi_0)->js_0;
        bool _S134 = connected_0(&_S133, _S132);
        if(!_S134)
        {
            bi_0 = bi_0 + 256U;
            continue;
        }
        uint4 _S135 = uint4(law_2.ids_0) ;
        uint _S136 = isl_2.range_0.x;
        float capacity_1;
        if(((*(&kernelContext_1)->g_plug_0)[_S135.y - _S136]) != (*(&kernelContext_1)->g_plug_0)[_S135.z - _S136])
        {
            capacity_1 = capacity_0 + (&(&kernelContext_1)->materials_0->m_0[_S135.x])->strength_0.x * (float4(law_2.geom0_0) ).x * (float4(law_2.geom1_0) ).w;
        }
        else
        {
            capacity_1 = capacity_0;
        }
        capacity_0 = capacity_1;
        bi_0 = bi_0 + 256U;
    }
    (*(&kernelContext_1)->g_a_0)[t_2] = mass_0 + 0.0f * count_0;
    (*(&kernelContext_1)->g_b_0)[t_2] = capacity_0;
    (*(&kernelContext_1)->g_lo_0)[t_2] = float3(count_0, 0.0f, 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s_1 = 128U;
    for(;;)
    {
        if(s_1 > 0U)
        {
        }
        else
        {
            break;
        }
        if(t_2 < s_1)
        {
            uint _S137 = t_2 + s_1;
            (*(&kernelContext_1)->g_a_0)[t_2] = (*(&kernelContext_1)->g_a_0)[t_2] + (*(&kernelContext_1)->g_a_0)[_S137];
            (*(&kernelContext_1)->g_b_0)[t_2] = (*(&kernelContext_1)->g_b_0)[t_2] + (*(&kernelContext_1)->g_b_0)[_S137];
            (*(&kernelContext_1)->g_lo_0)[t_2] = (*(&kernelContext_1)->g_lo_0)[t_2] + (*(&kernelContext_1)->g_lo_0)[_S137];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_1 = s_1 >> 1U;
    }
    if(t_2 == 0U)
    {
        float m_eff_0 = (*(&kernelContext_1)->g_a_0)[int(0)] * m_imp_3 / ((*(&kernelContext_1)->g_a_0)[int(0)] + m_imp_3);
        float impulse_0 = m_eff_0 * speed_3;
        float modulus_0 = 1.0f / (1.0f / e_imp_3 + 1.0f / (float4(((&kernelContext_1)->chunks_0+entry_k_2)->cmat_0) ).x);
        float tau_0 = 2.86999988555908203f * pow(m_eff_0 * m_eff_0 / (face_3 * modulus_0 * modulus_0 * max(speed_3, 0.00100000004749745f)), 0.20000000298023224f);
        if(((*(&kernelContext_1)->g_lo_0)[int(0)].x) < float(n_3))
        {
            out_0 = ((*(&kernelContext_1)->g_b_0)[int(0)]) > 0.0f;
        }
        else
        {
            out_0 = false;
        }
        if(out_0)
        {
            out_0 = (impulse_0 / tau_0) > (*(&kernelContext_1)->g_b_0)[int(0)];
        }
        else
        {
            out_0 = false;
        }
        if(out_0)
        {
            c_3 = 1U;
        }
        else
        {
            c_3 = 0U;
        }
        *(&kernelContext_1)->g_go_0 = c_3;
        if(out_0)
        {
            uint device* _S138 = (&kernelContext_1)->status_0+(16U + i_1);
            *_S138 = (*_S138) | 8U;
            *((&kernelContext_1)->status_0+1U) = 1U;
        }
        else
        {
            uint device* _S139 = (&kernelContext_1)->status_0+(16U + i_1);
            *_S139 = (*_S139) & 4294967287U;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if((group_uniform_0(*(&kernelContext_1)->g_go_0, t_2)) == 0U)
    {
        return;
    }
    bi_0 = _S130;
    for(;;)
    {
        if(bi_0 < _S112)
        {
        }
        else
        {
            break;
        }
        JointBond_natural_0 law_3 = ((&kernelContext_1)->bond_static_0+bi_0)->law_0;
        BondDyn_natural_0 device* _S140 = (&kernelContext_1)->bond_dyn_0+bi_0;
        float4 _S141 = float4((*_S140).force_lin_0) ;
        float4 _S142 = float4((*_S140).force_ang_0) ;
        float4 _S143 = float4((*_S140).sums_0) ;
        float4 _S144 = float4((*_S140).comps_0) ;
        uint4 _S145 = uint4((*_S140).events_0) ;
        thread BondDyn_0 dyn_0;
        (&dyn_0)->js_0 = (*_S140).js_0;
        (&dyn_0)->force_lin_0 = _S141;
        (&dyn_0)->force_ang_0 = _S142;
        (&dyn_0)->sums_0 = _S143;
        (&dyn_0)->comps_0 = _S144;
        (&dyn_0)->events_0 = _S145;
        bool _S146 = ((float4(((&kernelContext_1)->bond_static_0+bi_0)->law_0.stiff1_0) ).w) != 0.0f;
        thread JointState_0 _S147 = (&dyn_0)->js_0;
        bool _S148 = connected_0(&_S147, _S146);
        if(!_S148)
        {
            bi_0 = bi_0 + 256U;
            continue;
        }
        uint4 _S149 = uint4(law_3.ids_0) ;
        uint _S150 = isl_2.range_0.x;
        if(((*(&kernelContext_1)->g_plug_0)[_S149.y - _S150]) != (*(&kernelContext_1)->g_plug_0)[_S149.z - _S150])
        {
            (&(&dyn_0)->js_0)->damage_0 = 1.0f;
            (&(&dyn_0)->js_0)->rebar_broken_0 = 1.0f;
            (&dyn_0)->events_0.z = max((&dyn_0)->events_0.z, 1U);
            BondDyn_natural_0 device* _S151 = (&kernelContext_1)->bond_dyn_0+bi_0;
            _S151->js_0 = dyn_0.js_0;
            _S151->force_lin_0 = packed_float4(dyn_0.force_lin_0) ;
            _S151->force_ang_0 = packed_float4(dyn_0.force_ang_0) ;
            _S151->sums_0 = packed_float4(dyn_0.sums_0) ;
            _S151->comps_0 = packed_float4(dyn_0.comps_0) ;
            _S151->events_0 = packed_uint4(dyn_0.events_0) ;
        }
        bi_0 = bi_0 + 256U;
    }
    return;
}

[[kernel]] void punch_fast(uint3 thread_2 [[thread_position_in_threadgroup]], PunchParams_0 constant* params_2 [[buffer(0)]], packed_float4 device* fast_2 [[buffer(11)]], uint device* status_2 [[buffer(10)]], Island_natural_0 device* islands_2 [[buffer(1)]], packed_float4 device* ctab_2 [[buffer(4)]], EngineBody_natural_0 device* bodies_2 [[buffer(5)]], ChunkStatic_natural_0 device* chunks_2 [[buffer(2)]], packed_float4 device* state_2 [[buffer(3)]], packed_float4 device* impactors_2 [[buffer(6)]], BondStatic_natural_0 device* bond_static_2 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_2 [[buffer(8)]], MaterialTable_0 constant* materials_2 [[buffer(9)]], uint device* route_2 [[buffer(12)]])
{
    uint add_0;
    thread KernelContext_0 kernelContext_2;
    (&kernelContext_2)->params_0 = params_2;
    (&kernelContext_2)->fast_0 = fast_2;
    (&kernelContext_2)->status_0 = status_2;
    (&kernelContext_2)->islands_0 = islands_2;
    (&kernelContext_2)->ctab_0 = ctab_2;
    (&kernelContext_2)->bodies_0 = bodies_2;
    (&kernelContext_2)->chunks_0 = chunks_2;
    (&kernelContext_2)->state_0 = state_2;
    (&kernelContext_2)->impactors_0 = impactors_2;
    (&kernelContext_2)->bond_static_0 = bond_static_2;
    (&kernelContext_2)->bond_dyn_0 = bond_dyn_2;
    (&kernelContext_2)->materials_0 = materials_2;
    (&kernelContext_2)->route_0 = route_2;
    threadgroup array<float3, int(256)> g_lo_2;
    (&kernelContext_2)->g_lo_0 = &g_lo_2;
    threadgroup array<float3, int(256)> g_hi_2;
    (&kernelContext_2)->g_hi_0 = &g_hi_2;
    threadgroup array<float, int(256)> g_a_2;
    (&kernelContext_2)->g_a_0 = &g_a_2;
    threadgroup array<uint, int(256)> g_idx_2;
    (&kernelContext_2)->g_idx_0 = &g_idx_2;
    threadgroup array<uint, int(4096)> g_plug_2;
    (&kernelContext_2)->g_plug_0 = &g_plug_2;
    threadgroup uint g_changed_2;
    (&kernelContext_2)->g_changed_0 = &g_changed_2;
    threadgroup array<float, int(256)> g_b_2;
    (&kernelContext_2)->g_b_0 = &g_b_2;
    threadgroup uint g_go_2;
    (&kernelContext_2)->g_go_0 = &g_go_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_2)->g_scan_0 = &g_scan_2;
    uint t_3 = thread_2.x;
    uint base_0 = 0U;
    uint carry_0 = 0U;
    for(;;)
    {
        if(base_0 < ((&kernelContext_2)->params_0->island_count_0))
        {
        }
        else
        {
            break;
        }
        uint i_2 = base_0 + t_3;
        float3 _S152 = float3(0.0f) ;
        bool _S153;
        if(i_2 < ((&kernelContext_2)->params_0->island_count_0))
        {
            _S153 = (&kernelContext_2)->route_0[int(0)] != 0U;
        }
        else
        {
            _S153 = false;
        }
        uint k_3;
        bool _S154;
        bool fast_i_0;
        float radius_1;
        float face_r_0;
        float3 c_4;
        float3 vel_0;
        Island_0 isl_3;
        if(_S153)
        {
            Island_natural_0 isl_4 = (&kernelContext_2)->islands_0[i_2];
            uint4 _S155 = uint4(isl_4.range_0) ;
            uint4 _S156 = uint4(isl_4.info_0) ;
            float4 _S157 = float4(isl_4.com_0) ;
            float4 _S158 = float4(isl_4.inertia0_0) ;
            float4 _S159 = float4(isl_4.inertia1_0) ;
            float4 _S160 = float4(isl_4.inertia2_0) ;
            float4 _S161 = float4(isl_4.inv0_0) ;
            float4 _S162 = float4(isl_4.inv1_0) ;
            float4 _S163 = float4(isl_4.inv2_0) ;
            float4 _S164 = float4(isl_4.wcom_0) ;
            float4 _S165 = float4(isl_4.winv0_0) ;
            float4 _S166 = float4(isl_4.winv1_0) ;
            float4 _S167 = float4(isl_4.winv2_0) ;
            float4 _S168 = float4(isl_4.rotation_0) ;
            float4 _S169 = float4(isl_4.position_0) ;
            float4 _S170 = float4(isl_4.position_err_0) ;
            float4 _S171 = float4(isl_4.velocity_0) ;
            float4 _S172 = float4(isl_4.velocity_err_0) ;
            float4 _S173 = float4(isl_4.angular_velocity_0) ;
            uint4 _S174 = uint4(isl_4.probes_0) ;
            float4 _S175 = float4(isl_4.energy_0) ;
            float4 _S176 = float4(isl_4.rotation_err_0) ;
            float4 _S177 = float4(isl_4.momentum_0) ;
            float4 _S178 = float4(isl_4.momentum_err_0) ;
            uint _S179 = 3U * i_2;
            uint body_1 = (as_type<uint>(((float4(*((&kernelContext_2)->ctab_0+((&kernelContext_2)->params_0->engine_base_0 + _S179))) ).x)));
            uint _S180 = _S155.y;
            uint _S181 = _S155.x;
            uint n_4 = _S180 - _S181;
            if(n_4 > 0U)
            {
                _S154 = n_4 <= 64U;
            }
            else
            {
                _S154 = false;
            }
            bool _S182;
            if(_S154)
            {
                _S182 = body_1 != 4294967295U;
            }
            else
            {
                _S182 = false;
            }
            bool _S183;
            if(_S182)
            {
                _S183 = ((_S156.x) & 1U) == 0U;
            }
            else
            {
                _S183 = false;
            }
            if(_S183)
            {
                EngineBody_natural_0 b_2 = (&kernelContext_2)->bodies_0[body_1];
                float3 vel_1 = to_zup_0((float4(b_2.vel_sleep_0) ).xyz);
                if((length(vel_1)) > ((&kernelContext_2)->params_0->speed_1))
                {
                    float3 c_5 = to_zup_0((float4(b_2.pos_inv_mass_0) ).xyz);
                    Quat_0 p_inv_1 = quat_of_0(float4(*((&kernelContext_2)->ctab_0+((&kernelContext_2)->params_0->engine_base_0 + _S179 + 1U))) );
                    float3 com_2 = (float4(*((&kernelContext_2)->ctab_0+((&kernelContext_2)->params_0->engine_base_0 + _S179 + 2U))) ).xyz;
                    thread Quat_0 a_inv_1;
                    (&a_inv_1)->w_0 = 0.70710676908493042f;
                    (&a_inv_1)->x_0 = 0.70710676908493042f;
                    (&a_inv_1)->y_0 = 0.0f;
                    (&a_inv_1)->z_0 = 0.0f;
                    Quat_0 _S184 = quat_of_0(float4(b_2.rot_0) );
                    thread Quat_0 _S185 = a_inv_1;
                    thread Quat_0 _S186 = _S184;
                    Quat_0 _S187 = quat_mul_0(&_S185, &_S186);
                    thread Quat_0 _S188 = _S187;
                    thread Quat_0 _S189 = p_inv_1;
                    Quat_0 _S190 = quat_mul_0(&_S188, &_S189);
                    thread Quat_0 _S191 = _S190;
                    Quat_0 _S192 = normalized_0(&_S191);
                    thread Quat_0 _S193 = _S192;
                    float3 _S194 = rotate_0(&_S193, com_2);
                    float3 _S195 = c_5 - _S194;
                    float3 _S196 = normalize(vel_1);
                    radius_1 = 0.0f;
                    face_r_0 = 0.0f;
                    k_3 = _S181;
                    for(;;)
                    {
                        if(k_3 < _S180)
                        {
                        }
                        else
                        {
                            break;
                        }
                        float3 _S197 = (float4(((&kernelContext_2)->chunks_0+k_3)->center_0) ).xyz + (float4(*((&kernelContext_2)->state_0+4U * k_3)) ).xyz;
                        thread Quat_0 _S198 = _S192;
                        float3 _S199 = rotate_0(&_S198, _S197);
                        float3 r_6 = _S195 + _S199 - c_5;
                        float _S200 = max(radius_1, length(r_6) + (float4(((&kernelContext_2)->chunks_0+k_3)->half_0) ).w);
                        float _S201 = max(face_r_0, length(r_6 - _S196 * float3(dot(r_6, _S196)) ) + (float4(((&kernelContext_2)->chunks_0+k_3)->half_0) ).w);
                        uint _S202 = k_3 + 1U;
                        radius_1 = _S200;
                        face_r_0 = _S201;
                        k_3 = _S202;
                    }
                    fast_i_0 = true;
                    c_4 = c_5;
                }
                else
                {
                    fast_i_0 = false;
                    c_4 = _S152;
                    radius_1 = 0.0f;
                    face_r_0 = 0.0f;
                }
                vel_0 = vel_1;
            }
            else
            {
                fast_i_0 = false;
                c_4 = _S152;
                radius_1 = 0.0f;
                vel_0 = _S152;
                face_r_0 = 0.0f;
            }
            (&isl_3)->range_0 = _S155;
            (&isl_3)->info_0 = _S156;
            (&isl_3)->com_0 = _S157;
            (&isl_3)->inertia0_0 = _S158;
            (&isl_3)->inertia1_0 = _S159;
            (&isl_3)->inertia2_0 = _S160;
            (&isl_3)->inv0_0 = _S161;
            (&isl_3)->inv1_0 = _S162;
            (&isl_3)->inv2_0 = _S163;
            (&isl_3)->wcom_0 = _S164;
            (&isl_3)->winv0_0 = _S165;
            (&isl_3)->winv1_0 = _S166;
            (&isl_3)->winv2_0 = _S167;
            (&isl_3)->rotation_0 = _S168;
            (&isl_3)->position_0 = _S169;
            (&isl_3)->position_err_0 = _S170;
            (&isl_3)->velocity_0 = _S171;
            (&isl_3)->velocity_err_0 = _S172;
            (&isl_3)->angular_velocity_0 = _S173;
            (&isl_3)->done_0 = isl_4.done_0;
            (&isl_3)->coverage_0 = isl_4.coverage_0;
            (&isl_3)->wide_groups_0 = isl_4.wide_groups_0;
            (&isl_3)->done_pad_0 = isl_4.done_pad_0;
            (&isl_3)->probes_0 = _S174;
            (&isl_3)->energy_0 = _S175;
            (&isl_3)->rotation_err_0 = _S176;
            (&isl_3)->momentum_0 = _S177;
            (&isl_3)->momentum_err_0 = _S178;
        }
        else
        {
            fast_i_0 = false;
            c_4 = _S152;
            radius_1 = 0.0f;
            vel_0 = _S152;
            face_r_0 = 0.0f;
        }
        if(fast_i_0)
        {
            k_3 = 1U;
        }
        else
        {
            k_3 = 0U;
        }
        (*(&kernelContext_2)->g_scan_0)[t_3] = k_3;
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
            if(t_3 >= step_0)
            {
                add_0 = (*(&kernelContext_2)->g_scan_0)[t_3 - step_0];
            }
            else
            {
                add_0 = 0U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            (*(&kernelContext_2)->g_scan_0)[t_3] = (*(&kernelContext_2)->g_scan_0)[t_3] + add_0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            step_0 = step_0 << 1U;
        }
        uint _S203 = carry_0 + (*(&kernelContext_2)->g_scan_0)[t_3];
        if(fast_i_0)
        {
            add_0 = 1U;
        }
        else
        {
            add_0 = 0U;
        }
        uint at_0 = _S203 - add_0;
        if(fast_i_0)
        {
            _S154 = at_0 < 128U;
        }
        else
        {
            _S154 = false;
        }
        if(_S154)
        {
            uint _S204 = 3U * at_0;
            *((&kernelContext_2)->fast_0+(1U + _S204)) = packed_float4(float4(c_4, radius_1)) ;
            *((&kernelContext_2)->fast_0+(2U + _S204)) = packed_float4(float4(vel_0, (as_type<float>((i_2))))) ;
            *((&kernelContext_2)->fast_0+(3U + _S204)) = packed_float4(float4(face_r_0, isl_3.com_0.w, (float4(((&kernelContext_2)->chunks_0+isl_3.range_0.x)->cmat_0) ).x, 0.0f)) ;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint carry_1 = carry_0 + (*(&kernelContext_2)->g_scan_0)[int(1023)];
        base_0 = base_0 + 1024U;
        carry_0 = carry_1;
    }
    if(t_3 == 0U)
    {
        *((&kernelContext_2)->fast_0+int(0)) = packed_float4(float4((as_type<float>((min(carry_0, 128U)))), (as_type<float>(((&kernelContext_2)->route_0[int(0)]))), 0.0f, 0.0f)) ;
    }
    return;
}

