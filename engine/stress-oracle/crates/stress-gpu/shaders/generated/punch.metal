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

bool connected_0(const JointState_0 thread* st_0, bool has_rebar_0)
{
    bool _S1;
    if((st_0->damage_0) < 1.0f)
    {
        _S1 = true;
    }
    else
    {
        if(has_rebar_0)
        {
            _S1 = (st_0->rebar_broken_0) == 0.0f;
        }
        else
        {
            _S1 = false;
        }
    }
    return _S1;
}

struct PunchParams_0
{
    uint island_count_0;
    uint impactor_count_0;
    uint engine_base_0;
    uint pad0_0;
    float dt_0;
    float speed_0;
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
    uint device* route_0;
    uint device* status_0;
    Island_natural_0 device* islands_0;
    packed_float4 device* ctab_0;
    EngineBody_natural_0 device* bodies_0;
    ChunkStatic_natural_0 device* chunks_0;
    packed_float4 device* state_0;
    packed_float4 device* fast_0;
    packed_float4 device* impactors_0;
    BondStatic_natural_0 device* bond_static_0;
    BondDyn_natural_0 device* bond_dyn_0;
    MaterialTable_0 constant* materials_0;
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

struct BondDyn_0
{
    JointState_0 js_0;
    float4 force_lin_0;
    float4 force_ang_0;
    float4 sums_0;
    float4 comps_0;
    uint4 events_0;
};

[[kernel]] void punch_islands(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], PunchParams_0 constant* params_1 [[buffer(0)]], uint device* route_1 [[buffer(12)]], uint device* status_1 [[buffer(10)]], Island_natural_0 device* islands_1 [[buffer(1)]], packed_float4 device* ctab_1 [[buffer(4)]], EngineBody_natural_0 device* bodies_1 [[buffer(5)]], ChunkStatic_natural_0 device* chunks_1 [[buffer(2)]], packed_float4 device* state_1 [[buffer(3)]], packed_float4 device* fast_1 [[buffer(11)]], packed_float4 device* impactors_1 [[buffer(6)]], BondStatic_natural_0 device* bond_static_1 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_1 [[buffer(8)]], MaterialTable_0 constant* materials_1 [[buffer(9)]])
{
    uint ax_0;
    uint hit_k_0;
    float r_3;
    float t0_0;
    float t1_1;
    float t0_1;
    float m_imp_0;
    uint bi_0;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->route_0 = route_1;
    (&kernelContext_0)->status_0 = status_1;
    (&kernelContext_0)->islands_0 = islands_1;
    (&kernelContext_0)->ctab_0 = ctab_1;
    (&kernelContext_0)->bodies_0 = bodies_1;
    (&kernelContext_0)->chunks_0 = chunks_1;
    (&kernelContext_0)->state_0 = state_1;
    (&kernelContext_0)->fast_0 = fast_1;
    (&kernelContext_0)->impactors_0 = impactors_1;
    (&kernelContext_0)->bond_static_0 = bond_static_1;
    (&kernelContext_0)->bond_dyn_0 = bond_dyn_1;
    (&kernelContext_0)->materials_0 = materials_1;
    threadgroup array<float3, int(256)> g_lo_1;
    (&kernelContext_0)->g_lo_0 = &g_lo_1;
    threadgroup array<float3, int(256)> g_hi_1;
    (&kernelContext_0)->g_hi_0 = &g_hi_1;
    threadgroup array<float, int(256)> g_a_1;
    (&kernelContext_0)->g_a_0 = &g_a_1;
    threadgroup array<uint, int(256)> g_idx_1;
    (&kernelContext_0)->g_idx_0 = &g_idx_1;
    threadgroup array<uint, int(4096)> g_plug_1;
    (&kernelContext_0)->g_plug_0 = &g_plug_1;
    threadgroup uint g_changed_1;
    (&kernelContext_0)->g_changed_0 = &g_changed_1;
    threadgroup array<float, int(256)> g_b_1;
    (&kernelContext_0)->g_b_0 = &g_b_1;
    threadgroup uint g_go_1;
    (&kernelContext_0)->g_go_0 = &g_go_1;
    threadgroup array<uint, int(1024)> g_scan_1;
    (&kernelContext_0)->g_scan_0 = &g_scan_1;
    uint i_0 = group_0.x;
    uint t_1 = thread_1.x;
    if(i_0 >= (params_1->island_count_0))
    {
        return;
    }
    if((group_uniform_0((&kernelContext_0)->route_0[int(0)], t_1)) == 0U)
    {
        if(t_1 == 0U)
        {
            uint device* _S2 = (&kernelContext_0)->status_0+(16U + i_0);
            *_S2 = (*_S2) & 4294967287U;
        }
        return;
    }
    Island_natural_0 device* _S3 = (&kernelContext_0)->islands_0+i_0;
    uint4 _S4 = uint4(_S3->range_0) ;
    uint _S5 = _S4.x;
    uint n_1 = group_uniform_0(_S4.y - _S5, t_1);
    uint _S6 = _S4.z;
    uint _S7 = _S4.w;
    bool out_0;
    if(_S6 >= _S7)
    {
        out_0 = true;
    }
    else
    {
        out_0 = n_1 > 4096U;
    }
    if(out_0)
    {
        out_0 = true;
    }
    else
    {
        out_0 = ((&kernelContext_0)->params_0->impactor_count_0) == 0U;
    }
    uint c_0;
    if(out_0)
    {
        c_0 = 1U;
    }
    else
    {
        c_0 = 0U;
    }
    if((group_uniform_0(c_0, t_1)) != 0U)
    {
        if(t_1 == 0U)
        {
            uint device* _S8 = (&kernelContext_0)->status_0+(16U + i_0);
            *_S8 = (*_S8) & 4294967287U;
        }
        return;
    }
    uint _S9 = 3U * i_0;
    uint body_0 = (as_type<uint>(((float4(*((&kernelContext_0)->ctab_0+((&kernelContext_0)->params_0->engine_base_0 + _S9))) ).x)));
    Quat_0 _S10 = quat_of_0(float4(_S3->rotation_0) );
    float3 _S11 = (float4(_S3->position_0) ).xyz;
    float3 _S12 = (float4(_S3->velocity_0) ).xyz;
    float3 origin_0;
    float3 v_2;
    Quat_0 q_3;
    if(body_0 != 4294967295U)
    {
        Quat_0 p_inv_0 = quat_of_0(float4(*((&kernelContext_0)->ctab_0+((&kernelContext_0)->params_0->engine_base_0 + _S9 + 1U))) );
        float3 com_1 = (float4(*((&kernelContext_0)->ctab_0+((&kernelContext_0)->params_0->engine_base_0 + _S9 + 2U))) ).xyz;
        EngineBody_natural_0 b_0 = (&kernelContext_0)->bodies_0[body_0];
        thread Quat_0 a_inv_0;
        (&a_inv_0)->w_0 = 0.70710676908493042f;
        (&a_inv_0)->x_0 = 0.70710676908493042f;
        (&a_inv_0)->y_0 = 0.0f;
        (&a_inv_0)->z_0 = 0.0f;
        Quat_0 _S13 = quat_of_0(float4(b_0.rot_0) );
        thread Quat_0 _S14 = a_inv_0;
        thread Quat_0 _S15 = _S13;
        Quat_0 _S16 = quat_mul_0(&_S14, &_S15);
        thread Quat_0 _S17 = _S16;
        thread Quat_0 _S18 = p_inv_0;
        Quat_0 _S19 = quat_mul_0(&_S17, &_S18);
        thread Quat_0 _S20 = _S19;
        Quat_0 _S21 = normalized_0(&_S20);
        float3 _S22 = to_zup_0((float4(b_0.pos_inv_mass_0) ).xyz);
        thread Quat_0 _S23 = _S21;
        float3 _S24 = rotate_0(&_S23, com_1);
        float3 _S25 = to_zup_0((float4(b_0.vel_sleep_0) ).xyz);
        origin_0 = _S22 - _S24;
        q_3 = _S21;
        v_2 = _S25;
    }
    else
    {
        origin_0 = _S11;
        q_3 = _S10;
        v_2 = _S12;
    }
    float inf_0 = float_bits_0(2139095040U);
    float3 _S26 = float3(- inf_0) ;
    float3 lo_0 = float3(inf_0) ;
    float3 hi_0 = _S26;
    c_0 = t_1;
    for(;;)
    {
        if(c_0 < n_1)
        {
        }
        else
        {
            break;
        }
        uint k_0 = _S5 + c_0;
        float3 _S27 = (float4(((&kernelContext_0)->chunks_0+k_0)->center_0) ).xyz + (float4(*((&kernelContext_0)->state_0+4U * k_0)) ).xyz;
        thread Quat_0 _S28 = q_3;
        float3 _S29 = rotate_0(&_S28, _S27);
        float3 p_0 = origin_0 + _S29;
        float3 _S30 = min(lo_0, p_0 - float3((float4(((&kernelContext_0)->chunks_0+k_0)->half_0) ).w) );
        float3 _S31 = max(hi_0, p_0 + float3((float4(((&kernelContext_0)->chunks_0+k_0)->half_0) ).w) );
        uint c_1 = c_0 + 256U;
        lo_0 = _S30;
        hi_0 = _S31;
        c_0 = c_1;
    }
    (*(&kernelContext_0)->g_lo_0)[t_1] = lo_0;
    (*(&kernelContext_0)->g_hi_0)[t_1] = hi_0;
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
            uint _S32 = t_1 + s_0;
            (*(&kernelContext_0)->g_lo_0)[t_1] = min((*(&kernelContext_0)->g_lo_0)[t_1], (*(&kernelContext_0)->g_lo_0)[_S32]);
            (*(&kernelContext_0)->g_hi_0)[t_1] = max((*(&kernelContext_0)->g_hi_0)[t_1], (*(&kernelContext_0)->g_hi_0)[_S32]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    float3 _S33 = (*(&kernelContext_0)->g_lo_0)[int(0)];
    float3 _S34 = (*(&kernelContext_0)->g_hi_0)[int(0)];
    float3 _S35 = float3(0.0f) ;
    uint _S36 = (as_type<uint>(((float4(*((&kernelContext_0)->fast_0+int(0))) ).x)));
    uint hit_k_1 = 4294967295U;
    float3 hit_0 = _S35;
    float3 d_0 = _S35;
    float face_0 = 0.0f;
    float m_imp_1 = 0.0f;
    float speed_1 = 0.0f;
    float e_imp_0 = 0.0f;
    uint k_1 = 0U;
    for(;;)
    {
        if(k_1 < ((&kernelContext_0)->params_0->impactor_count_0 + _S36))
        {
            out_0 = hit_k_1 == 4294967295U;
        }
        else
        {
            out_0 = false;
        }
        if(out_0)
        {
        }
        else
        {
            break;
        }
        float3 rel_0;
        float3 c_2;
        float4 info_2;
        if(k_1 < ((&kernelContext_0)->params_0->impactor_count_0))
        {
            uint _S37 = 2U * k_1;
            float4 _S38 = float4(*((&kernelContext_0)->impactors_0+_S37)) ;
            EngineBody_natural_0 b_1 = (&kernelContext_0)->bodies_0[(as_type<uint>((_S38.x)))];
            float3 _S39 = to_zup_0((float4(b_1.pos_inv_mass_0) ).xyz);
            float _S40 = _S38.y;
            float4 _S41 = float4(_S38.z, _S38.w, (float4(*((&kernelContext_0)->impactors_0+(_S37 + 1U))) ).x, 0.0f);
            rel_0 = to_zup_0((float4(b_1.vel_sleep_0) ).xyz) - v_2;
            r_3 = _S40;
            c_2 = _S39;
            info_2 = _S41;
        }
        else
        {
            uint f_0 = 1U + 3U * (k_1 - (&kernelContext_0)->params_0->impactor_count_0);
            uint _S42 = f_0 + 1U;
            if((as_type<uint>(((float4(*((&kernelContext_0)->fast_0+_S42)) ).w))) == i_0)
            {
                k_1 = k_1 + 1U;
                continue;
            }
            float3 _S43 = (float4(*((&kernelContext_0)->fast_0+f_0)) ).xyz;
            float _S44 = (float4(*((&kernelContext_0)->fast_0+f_0)) ).w;
            float4 _S45 = float4(*((&kernelContext_0)->fast_0+(f_0 + 2U))) ;
            rel_0 = (float4(*((&kernelContext_0)->fast_0+_S42)) ).xyz - v_2;
            r_3 = _S44;
            c_2 = _S43;
            info_2 = _S45;
        }
        float sp_0 = length(rel_0);
        if(!(sp_0 > ((&kernelContext_0)->params_0->speed_0)))
        {
            k_1 = k_1 + 1U;
            continue;
        }
        float3 _S46 = float3(r_3) ;
        float3 _S47 = _S33 - _S46;
        float3 _S48 = _S34 + _S46;
        float _S49 = (&kernelContext_0)->params_0->dt_0;
        t0_0 = 0.0f;
        t1_1 = _S49;
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
            float3 _S50 = rel_0;
            uint _S51 = ax_0;
            if((abs(rel_0[ax_0])) < 9.999999960041972e-13f)
            {
                float3 _S52 = c_2;
                uint _S53 = ax_0;
                bool _S54;
                if((c_2[ax_0]) < (_S47[ax_0]))
                {
                    _S54 = true;
                }
                else
                {
                    _S54 = (_S52[_S53]) > (_S48[ax_0]);
                }
                if(_S54)
                {
                    t0_1 = inf_0;
                }
                else
                {
                    t0_1 = t0_0;
                }
                t0_0 = t0_1;
                ax_0 = ax_0 + 1U;
                continue;
            }
            float ta_0 = (_S47[ax_0] - c_2[ax_0]) / _S50[_S51];
            float tb_0 = (_S48[ax_0] - c_2[ax_0]) / _S50[_S51];
            float _S55 = min(t1_1, max(ta_0, tb_0));
            t0_0 = max(t0_0, min(ta_0, tb_0));
            t1_1 = _S55;
            ax_0 = ax_0 + 1U;
        }
        float3 hit_1;
        float3 d_1;
        float speed_2;
        float e_imp_1;
        if(t0_0 <= t1_1)
        {
            float3 _S56 = c_2 + rel_0 * float3(t0_0) ;
            float3 _S57 = rel_0 / float3(sp_0) ;
            float _S58 = info_2.x;
            float _S59 = info_2.y;
            float _S60 = info_2.z;
            hit_k_0 = k_1;
            hit_1 = _S56;
            d_1 = _S57;
            t0_1 = _S58;
            m_imp_0 = _S59;
            speed_2 = sp_0;
            e_imp_1 = _S60;
        }
        else
        {
            hit_k_0 = hit_k_1;
            hit_1 = hit_0;
            d_1 = d_0;
            t0_1 = face_0;
            m_imp_0 = m_imp_1;
            speed_2 = speed_1;
            e_imp_1 = e_imp_0;
        }
        hit_k_1 = hit_k_0;
        hit_0 = hit_1;
        d_0 = d_1;
        face_0 = t0_1;
        m_imp_1 = m_imp_0;
        speed_1 = speed_2;
        e_imp_0 = e_imp_1;
        k_1 = k_1 + 1U;
    }
    if(hit_k_1 == 4294967295U)
    {
        c_0 = 1U;
    }
    else
    {
        c_0 = 0U;
    }
    if((group_uniform_0(c_0, t_1)) != 0U)
    {
        if(t_1 == 0U)
        {
            uint device* _S61 = (&kernelContext_0)->status_0+(16U + i_0);
            *_S61 = (*_S61) & 4294967287U;
        }
        return;
    }
    r_3 = inf_0;
    ax_0 = 4294967295U;
    c_0 = t_1;
    for(;;)
    {
        if(c_0 < n_1)
        {
        }
        else
        {
            break;
        }
        uint k_2 = _S5 + c_0;
        float3 _S62 = (float4(((&kernelContext_0)->chunks_0+k_2)->center_0) ).xyz + (float4(*((&kernelContext_0)->state_0+4U * k_2)) ).xyz;
        thread Quat_0 _S63 = q_3;
        float3 _S64 = rotate_0(&_S63, _S62);
        float3 r_4 = origin_0 + _S64 - hit_0;
        float along_0 = dot(r_4, d_0);
        float dist_0 = length(r_4 - d_0 * float3(along_0) ) + abs(along_0) * 0.00100000004749745f;
        if(dist_0 < r_3)
        {
            r_3 = dist_0;
            ax_0 = c_0;
        }
        c_0 = c_0 + 256U;
    }
    (*(&kernelContext_0)->g_a_0)[t_1] = r_3;
    (*(&kernelContext_0)->g_idx_0)[t_1] = ax_0;
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
            uint _S65 = t_1 + s_0;
            if(((*(&kernelContext_0)->g_a_0)[_S65]) < (*(&kernelContext_0)->g_a_0)[t_1])
            {
                out_0 = true;
            }
            else
            {
                if(((*(&kernelContext_0)->g_a_0)[_S65]) == (*(&kernelContext_0)->g_a_0)[t_1])
                {
                    out_0 = ((*(&kernelContext_0)->g_idx_0)[_S65]) < (*(&kernelContext_0)->g_idx_0)[t_1];
                }
                else
                {
                    out_0 = false;
                }
            }
        }
        else
        {
            out_0 = false;
        }
        if(out_0)
        {
            uint _S66 = t_1 + s_0;
            (*(&kernelContext_0)->g_a_0)[t_1] = (*(&kernelContext_0)->g_a_0)[_S66];
            (*(&kernelContext_0)->g_idx_0)[t_1] = (*(&kernelContext_0)->g_idx_0)[_S66];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    uint entry_0 = (*(&kernelContext_0)->g_idx_0)[int(0)];
    uint entry_k_0 = _S5 + (*(&kernelContext_0)->g_idx_0)[int(0)];
    float3 _S67 = (float4(((&kernelContext_0)->chunks_0+entry_k_0)->center_0) ).xyz + (float4(*((&kernelContext_0)->state_0+4U * entry_k_0)) ).xyz;
    thread Quat_0 _S68 = q_3;
    float3 _S69 = rotate_0(&_S68, _S67);
    float _S70 = dot(origin_0 + _S69 - hit_0, d_0);
    c_0 = t_1;
    for(;;)
    {
        if(c_0 < n_1)
        {
        }
        else
        {
            break;
        }
        if(c_0 == entry_0)
        {
            s_0 = 1U;
        }
        else
        {
            s_0 = 0U;
        }
        (*(&kernelContext_0)->g_plug_0)[c_0] = s_0;
        c_0 = c_0 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hit_k_0 = 0U;
    for(;;)
    {
        if(hit_k_0 < n_1)
        {
        }
        else
        {
            break;
        }
        if(t_1 == 0U)
        {
            *(&kernelContext_0)->g_changed_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bi_0 = _S6 + t_1;
        for(;;)
        {
            if(bi_0 < _S7)
            {
            }
            else
            {
                break;
            }
            JointBond_natural_0 law_1 = ((&kernelContext_0)->bond_static_0+bi_0)->law_0;
            bool _S71 = ((float4(((&kernelContext_0)->bond_static_0+bi_0)->law_0.stiff1_0) ).w) != 0.0f;
            thread JointState_0 _S72 = ((&kernelContext_0)->bond_dyn_0+bi_0)->js_0;
            bool _S73 = connected_0(&_S72, _S71);
            if(!_S73)
            {
                bi_0 = bi_0 + 256U;
                continue;
            }
            uint4 _S74 = uint4(law_1.ids_0) ;
            uint ca_0 = _S74.y - _S5;
            uint cb_0 = _S74.z - _S5;
            if(((*(&kernelContext_0)->g_plug_0)[ca_0]) == (*(&kernelContext_0)->g_plug_0)[cb_0])
            {
                bi_0 = bi_0 + 256U;
                continue;
            }
            uint o_1;
            if(((*(&kernelContext_0)->g_plug_0)[ca_0]) != 0U)
            {
                o_1 = cb_0;
            }
            else
            {
                o_1 = ca_0;
            }
            uint ok_0 = _S5 + o_1;
            float3 _S75 = (float4(((&kernelContext_0)->chunks_0+ok_0)->center_0) ).xyz + (float4(*((&kernelContext_0)->state_0+4U * ok_0)) ).xyz;
            thread Quat_0 _S76 = q_3;
            float3 _S77 = rotate_0(&_S76, _S75);
            float3 r_5 = origin_0 + _S77 - hit_0;
            float along_1 = dot(r_5, d_0);
            float lateral_0 = length(r_5 - d_0 * float3(along_1) );
            if(along_1 >= (_S70 - (float4(((&kernelContext_0)->chunks_0+ok_0)->half_0) ).w))
            {
                out_0 = lateral_0 <= (face_0 + max(along_1 - _S70, 0.0f));
            }
            else
            {
                out_0 = false;
            }
            if(out_0)
            {
                (*(&kernelContext_0)->g_plug_0)[o_1] = 1U;
                *(&kernelContext_0)->g_changed_0 = 1U;
            }
            bi_0 = bi_0 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if((group_uniform_0(*(&kernelContext_0)->g_changed_0, t_1)) == 0U)
        {
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        hit_k_0 = hit_k_0 + 1U;
    }
    c_0 = t_1;
    t0_0 = 0.0f;
    t1_1 = 0.0f;
    for(;;)
    {
        if(c_0 < n_1)
        {
        }
        else
        {
            break;
        }
        if(((*(&kernelContext_0)->g_plug_0)[c_0]) != 0U)
        {
            float count_0 = t1_1 + 1.0f;
            t0_0 = t0_0 + (float4(((&kernelContext_0)->chunks_0+(_S5 + c_0))->center_0) ).w;
            t1_1 = count_0;
        }
        c_0 = c_0 + 256U;
    }
    uint _S78 = _S6 + t_1;
    bi_0 = _S78;
    t0_1 = 0.0f;
    for(;;)
    {
        if(bi_0 < _S7)
        {
        }
        else
        {
            break;
        }
        JointBond_natural_0 law_2 = ((&kernelContext_0)->bond_static_0+bi_0)->law_0;
        bool _S79 = ((float4(((&kernelContext_0)->bond_static_0+bi_0)->law_0.stiff1_0) ).w) != 0.0f;
        thread JointState_0 _S80 = ((&kernelContext_0)->bond_dyn_0+bi_0)->js_0;
        bool _S81 = connected_0(&_S80, _S79);
        if(!_S81)
        {
            bi_0 = bi_0 + 256U;
            continue;
        }
        uint4 _S82 = uint4(law_2.ids_0) ;
        if(((*(&kernelContext_0)->g_plug_0)[_S82.y - _S5]) != (*(&kernelContext_0)->g_plug_0)[_S82.z - _S5])
        {
            m_imp_0 = t0_1 + (&(&kernelContext_0)->materials_0->m_0[_S82.x])->strength_0.x * (float4(law_2.geom0_0) ).x * (float4(law_2.geom1_0) ).w;
        }
        else
        {
            m_imp_0 = t0_1;
        }
        t0_1 = m_imp_0;
        bi_0 = bi_0 + 256U;
    }
    (*(&kernelContext_0)->g_a_0)[t_1] = t0_0 + 0.0f * t1_1;
    (*(&kernelContext_0)->g_b_0)[t_1] = t0_1;
    (*(&kernelContext_0)->g_lo_0)[t_1] = float3(t1_1, 0.0f, 0.0f);
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
            uint _S83 = t_1 + s_0;
            (*(&kernelContext_0)->g_a_0)[t_1] = (*(&kernelContext_0)->g_a_0)[t_1] + (*(&kernelContext_0)->g_a_0)[_S83];
            (*(&kernelContext_0)->g_b_0)[t_1] = (*(&kernelContext_0)->g_b_0)[t_1] + (*(&kernelContext_0)->g_b_0)[_S83];
            (*(&kernelContext_0)->g_lo_0)[t_1] = (*(&kernelContext_0)->g_lo_0)[t_1] + (*(&kernelContext_0)->g_lo_0)[_S83];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    if(t_1 == 0U)
    {
        float m_eff_0 = (*(&kernelContext_0)->g_a_0)[int(0)] * m_imp_1 / ((*(&kernelContext_0)->g_a_0)[int(0)] + m_imp_1);
        float impulse_0 = m_eff_0 * speed_1;
        float modulus_0 = 1.0f / (1.0f / e_imp_0 + 1.0f / (float4(((&kernelContext_0)->chunks_0+entry_k_0)->cmat_0) ).x);
        float tau_0 = 2.86999988555908203f * pow(m_eff_0 * m_eff_0 / (face_0 * modulus_0 * modulus_0 * max(speed_1, 0.00100000004749745f)), 0.20000000298023224f);
        if(((*(&kernelContext_0)->g_lo_0)[int(0)].x) < float(n_1))
        {
            out_0 = ((*(&kernelContext_0)->g_b_0)[int(0)]) > 0.0f;
        }
        else
        {
            out_0 = false;
        }
        if(out_0)
        {
            out_0 = (impulse_0 / tau_0) > (*(&kernelContext_0)->g_b_0)[int(0)];
        }
        else
        {
            out_0 = false;
        }
        if(out_0)
        {
            c_0 = 1U;
        }
        else
        {
            c_0 = 0U;
        }
        *(&kernelContext_0)->g_go_0 = c_0;
        if(out_0)
        {
            uint device* _S84 = (&kernelContext_0)->status_0+(16U + i_0);
            *_S84 = (*_S84) | 8U;
            *((&kernelContext_0)->status_0+1U) = 1U;
        }
        else
        {
            uint device* _S85 = (&kernelContext_0)->status_0+(16U + i_0);
            *_S85 = (*_S85) & 4294967287U;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if((group_uniform_0(*(&kernelContext_0)->g_go_0, t_1)) == 0U)
    {
        return;
    }
    bi_0 = _S78;
    for(;;)
    {
        if(bi_0 < _S7)
        {
        }
        else
        {
            break;
        }
        JointBond_natural_0 law_3 = ((&kernelContext_0)->bond_static_0+bi_0)->law_0;
        BondDyn_natural_0 device* _S86 = (&kernelContext_0)->bond_dyn_0+bi_0;
        float4 _S87 = float4((*_S86).force_lin_0) ;
        float4 _S88 = float4((*_S86).force_ang_0) ;
        float4 _S89 = float4((*_S86).sums_0) ;
        float4 _S90 = float4((*_S86).comps_0) ;
        uint4 _S91 = uint4((*_S86).events_0) ;
        thread BondDyn_0 dyn_0;
        (&dyn_0)->js_0 = (*_S86).js_0;
        (&dyn_0)->force_lin_0 = _S87;
        (&dyn_0)->force_ang_0 = _S88;
        (&dyn_0)->sums_0 = _S89;
        (&dyn_0)->comps_0 = _S90;
        (&dyn_0)->events_0 = _S91;
        bool _S92 = ((float4(((&kernelContext_0)->bond_static_0+bi_0)->law_0.stiff1_0) ).w) != 0.0f;
        thread JointState_0 _S93 = (&dyn_0)->js_0;
        bool _S94 = connected_0(&_S93, _S92);
        if(!_S94)
        {
            bi_0 = bi_0 + 256U;
            continue;
        }
        uint4 _S95 = uint4(law_3.ids_0) ;
        if(((*(&kernelContext_0)->g_plug_0)[_S95.y - _S5]) != (*(&kernelContext_0)->g_plug_0)[_S95.z - _S5])
        {
            (&(&dyn_0)->js_0)->damage_0 = 1.0f;
            (&(&dyn_0)->js_0)->rebar_broken_0 = 1.0f;
            (&dyn_0)->events_0.z = max((&dyn_0)->events_0.z, 1U);
            BondDyn_natural_0 device* _S96 = (&kernelContext_0)->bond_dyn_0+bi_0;
            _S96->js_0 = dyn_0.js_0;
            _S96->force_lin_0 = packed_float4(dyn_0.force_lin_0) ;
            _S96->force_ang_0 = packed_float4(dyn_0.force_ang_0) ;
            _S96->sums_0 = packed_float4(dyn_0.sums_0) ;
            _S96->comps_0 = packed_float4(dyn_0.comps_0) ;
            _S96->events_0 = packed_uint4(dyn_0.events_0) ;
        }
        bi_0 = bi_0 + 256U;
    }
    return;
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

[[kernel]] void punch_fast(uint3 thread_2 [[thread_position_in_threadgroup]], PunchParams_0 constant* params_2 [[buffer(0)]], uint device* route_2 [[buffer(12)]], uint device* status_2 [[buffer(10)]], Island_natural_0 device* islands_2 [[buffer(1)]], packed_float4 device* ctab_2 [[buffer(4)]], EngineBody_natural_0 device* bodies_2 [[buffer(5)]], ChunkStatic_natural_0 device* chunks_2 [[buffer(2)]], packed_float4 device* state_2 [[buffer(3)]], packed_float4 device* fast_2 [[buffer(11)]], packed_float4 device* impactors_2 [[buffer(6)]], BondStatic_natural_0 device* bond_static_2 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_2 [[buffer(8)]], MaterialTable_0 constant* materials_2 [[buffer(9)]])
{
    uint add_0;
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->params_0 = params_2;
    (&kernelContext_1)->route_0 = route_2;
    (&kernelContext_1)->status_0 = status_2;
    (&kernelContext_1)->islands_0 = islands_2;
    (&kernelContext_1)->ctab_0 = ctab_2;
    (&kernelContext_1)->bodies_0 = bodies_2;
    (&kernelContext_1)->chunks_0 = chunks_2;
    (&kernelContext_1)->state_0 = state_2;
    (&kernelContext_1)->fast_0 = fast_2;
    (&kernelContext_1)->impactors_0 = impactors_2;
    (&kernelContext_1)->bond_static_0 = bond_static_2;
    (&kernelContext_1)->bond_dyn_0 = bond_dyn_2;
    (&kernelContext_1)->materials_0 = materials_2;
    threadgroup array<float3, int(256)> g_lo_2;
    (&kernelContext_1)->g_lo_0 = &g_lo_2;
    threadgroup array<float3, int(256)> g_hi_2;
    (&kernelContext_1)->g_hi_0 = &g_hi_2;
    threadgroup array<float, int(256)> g_a_2;
    (&kernelContext_1)->g_a_0 = &g_a_2;
    threadgroup array<uint, int(256)> g_idx_2;
    (&kernelContext_1)->g_idx_0 = &g_idx_2;
    threadgroup array<uint, int(4096)> g_plug_2;
    (&kernelContext_1)->g_plug_0 = &g_plug_2;
    threadgroup uint g_changed_2;
    (&kernelContext_1)->g_changed_0 = &g_changed_2;
    threadgroup array<float, int(256)> g_b_2;
    (&kernelContext_1)->g_b_0 = &g_b_2;
    threadgroup uint g_go_2;
    (&kernelContext_1)->g_go_0 = &g_go_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_1)->g_scan_0 = &g_scan_2;
    uint t_2 = thread_2.x;
    uint base_0 = 0U;
    uint carry_0 = 0U;
    for(;;)
    {
        if(base_0 < ((&kernelContext_1)->params_0->island_count_0))
        {
        }
        else
        {
            break;
        }
        uint i_1 = base_0 + t_2;
        float3 _S97 = float3(0.0f) ;
        bool _S98;
        if(i_1 < ((&kernelContext_1)->params_0->island_count_0))
        {
            _S98 = (&kernelContext_1)->route_0[int(0)] != 0U;
        }
        else
        {
            _S98 = false;
        }
        uint k_3;
        bool _S99;
        bool fast_i_0;
        float radius_0;
        float face_r_0;
        float3 c_3;
        float3 vel_0;
        Island_0 isl_0;
        if(_S98)
        {
            Island_natural_0 isl_1 = (&kernelContext_1)->islands_0[i_1];
            uint4 _S100 = uint4(isl_1.range_0) ;
            uint4 _S101 = uint4(isl_1.info_0) ;
            float4 _S102 = float4(isl_1.com_0) ;
            float4 _S103 = float4(isl_1.inertia0_0) ;
            float4 _S104 = float4(isl_1.inertia1_0) ;
            float4 _S105 = float4(isl_1.inertia2_0) ;
            float4 _S106 = float4(isl_1.inv0_0) ;
            float4 _S107 = float4(isl_1.inv1_0) ;
            float4 _S108 = float4(isl_1.inv2_0) ;
            float4 _S109 = float4(isl_1.wcom_0) ;
            float4 _S110 = float4(isl_1.winv0_0) ;
            float4 _S111 = float4(isl_1.winv1_0) ;
            float4 _S112 = float4(isl_1.winv2_0) ;
            float4 _S113 = float4(isl_1.rotation_0) ;
            float4 _S114 = float4(isl_1.position_0) ;
            float4 _S115 = float4(isl_1.position_err_0) ;
            float4 _S116 = float4(isl_1.velocity_0) ;
            float4 _S117 = float4(isl_1.velocity_err_0) ;
            float4 _S118 = float4(isl_1.angular_velocity_0) ;
            uint4 _S119 = uint4(isl_1.probes_0) ;
            float4 _S120 = float4(isl_1.energy_0) ;
            float4 _S121 = float4(isl_1.rotation_err_0) ;
            float4 _S122 = float4(isl_1.momentum_0) ;
            float4 _S123 = float4(isl_1.momentum_err_0) ;
            uint _S124 = 3U * i_1;
            uint body_1 = (as_type<uint>(((float4(*((&kernelContext_1)->ctab_0+((&kernelContext_1)->params_0->engine_base_0 + _S124))) ).x)));
            uint _S125 = _S100.y;
            uint _S126 = _S100.x;
            uint n_2 = _S125 - _S126;
            if(n_2 > 0U)
            {
                _S99 = n_2 <= 64U;
            }
            else
            {
                _S99 = false;
            }
            bool _S127;
            if(_S99)
            {
                _S127 = body_1 != 4294967295U;
            }
            else
            {
                _S127 = false;
            }
            bool _S128;
            if(_S127)
            {
                _S128 = ((_S101.x) & 1U) == 0U;
            }
            else
            {
                _S128 = false;
            }
            if(_S128)
            {
                EngineBody_natural_0 b_2 = (&kernelContext_1)->bodies_0[body_1];
                float3 vel_1 = to_zup_0((float4(b_2.vel_sleep_0) ).xyz);
                if((length(vel_1)) > ((&kernelContext_1)->params_0->speed_0))
                {
                    float3 c_4 = to_zup_0((float4(b_2.pos_inv_mass_0) ).xyz);
                    Quat_0 p_inv_1 = quat_of_0(float4(*((&kernelContext_1)->ctab_0+((&kernelContext_1)->params_0->engine_base_0 + _S124 + 1U))) );
                    float3 com_2 = (float4(*((&kernelContext_1)->ctab_0+((&kernelContext_1)->params_0->engine_base_0 + _S124 + 2U))) ).xyz;
                    thread Quat_0 a_inv_1;
                    (&a_inv_1)->w_0 = 0.70710676908493042f;
                    (&a_inv_1)->x_0 = 0.70710676908493042f;
                    (&a_inv_1)->y_0 = 0.0f;
                    (&a_inv_1)->z_0 = 0.0f;
                    Quat_0 _S129 = quat_of_0(float4(b_2.rot_0) );
                    thread Quat_0 _S130 = a_inv_1;
                    thread Quat_0 _S131 = _S129;
                    Quat_0 _S132 = quat_mul_0(&_S130, &_S131);
                    thread Quat_0 _S133 = _S132;
                    thread Quat_0 _S134 = p_inv_1;
                    Quat_0 _S135 = quat_mul_0(&_S133, &_S134);
                    thread Quat_0 _S136 = _S135;
                    Quat_0 _S137 = normalized_0(&_S136);
                    thread Quat_0 _S138 = _S137;
                    float3 _S139 = rotate_0(&_S138, com_2);
                    float3 _S140 = c_4 - _S139;
                    float3 _S141 = normalize(vel_1);
                    radius_0 = 0.0f;
                    face_r_0 = 0.0f;
                    k_3 = _S126;
                    for(;;)
                    {
                        if(k_3 < _S125)
                        {
                        }
                        else
                        {
                            break;
                        }
                        float3 _S142 = (float4(((&kernelContext_1)->chunks_0+k_3)->center_0) ).xyz + (float4(*((&kernelContext_1)->state_0+4U * k_3)) ).xyz;
                        thread Quat_0 _S143 = _S137;
                        float3 _S144 = rotate_0(&_S143, _S142);
                        float3 r_6 = _S140 + _S144 - c_4;
                        float _S145 = max(radius_0, length(r_6) + (float4(((&kernelContext_1)->chunks_0+k_3)->half_0) ).w);
                        float _S146 = max(face_r_0, length(r_6 - _S141 * float3(dot(r_6, _S141)) ) + (float4(((&kernelContext_1)->chunks_0+k_3)->half_0) ).w);
                        uint _S147 = k_3 + 1U;
                        radius_0 = _S145;
                        face_r_0 = _S146;
                        k_3 = _S147;
                    }
                    fast_i_0 = true;
                    c_3 = c_4;
                }
                else
                {
                    fast_i_0 = false;
                    c_3 = _S97;
                    radius_0 = 0.0f;
                    face_r_0 = 0.0f;
                }
                vel_0 = vel_1;
            }
            else
            {
                fast_i_0 = false;
                c_3 = _S97;
                radius_0 = 0.0f;
                vel_0 = _S97;
                face_r_0 = 0.0f;
            }
            (&isl_0)->range_0 = _S100;
            (&isl_0)->info_0 = _S101;
            (&isl_0)->com_0 = _S102;
            (&isl_0)->inertia0_0 = _S103;
            (&isl_0)->inertia1_0 = _S104;
            (&isl_0)->inertia2_0 = _S105;
            (&isl_0)->inv0_0 = _S106;
            (&isl_0)->inv1_0 = _S107;
            (&isl_0)->inv2_0 = _S108;
            (&isl_0)->wcom_0 = _S109;
            (&isl_0)->winv0_0 = _S110;
            (&isl_0)->winv1_0 = _S111;
            (&isl_0)->winv2_0 = _S112;
            (&isl_0)->rotation_0 = _S113;
            (&isl_0)->position_0 = _S114;
            (&isl_0)->position_err_0 = _S115;
            (&isl_0)->velocity_0 = _S116;
            (&isl_0)->velocity_err_0 = _S117;
            (&isl_0)->angular_velocity_0 = _S118;
            (&isl_0)->done_0 = isl_1.done_0;
            (&isl_0)->coverage_0 = isl_1.coverage_0;
            (&isl_0)->wide_groups_0 = isl_1.wide_groups_0;
            (&isl_0)->done_pad_0 = isl_1.done_pad_0;
            (&isl_0)->probes_0 = _S119;
            (&isl_0)->energy_0 = _S120;
            (&isl_0)->rotation_err_0 = _S121;
            (&isl_0)->momentum_0 = _S122;
            (&isl_0)->momentum_err_0 = _S123;
        }
        else
        {
            fast_i_0 = false;
            c_3 = _S97;
            radius_0 = 0.0f;
            vel_0 = _S97;
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
        (*(&kernelContext_1)->g_scan_0)[t_2] = k_3;
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
            if(t_2 >= step_0)
            {
                add_0 = (*(&kernelContext_1)->g_scan_0)[t_2 - step_0];
            }
            else
            {
                add_0 = 0U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            (*(&kernelContext_1)->g_scan_0)[t_2] = (*(&kernelContext_1)->g_scan_0)[t_2] + add_0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            step_0 = step_0 << 1U;
        }
        uint _S148 = carry_0 + (*(&kernelContext_1)->g_scan_0)[t_2];
        if(fast_i_0)
        {
            add_0 = 1U;
        }
        else
        {
            add_0 = 0U;
        }
        uint at_0 = _S148 - add_0;
        if(fast_i_0)
        {
            _S99 = at_0 < 128U;
        }
        else
        {
            _S99 = false;
        }
        if(_S99)
        {
            uint _S149 = 3U * at_0;
            *((&kernelContext_1)->fast_0+(1U + _S149)) = packed_float4(float4(c_3, radius_0)) ;
            *((&kernelContext_1)->fast_0+(2U + _S149)) = packed_float4(float4(vel_0, (as_type<float>((i_1))))) ;
            *((&kernelContext_1)->fast_0+(3U + _S149)) = packed_float4(float4(face_r_0, isl_0.com_0.w, (float4(((&kernelContext_1)->chunks_0+isl_0.range_0.x)->cmat_0) ).x, 0.0f)) ;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint carry_1 = carry_0 + (*(&kernelContext_1)->g_scan_0)[int(1023)];
        base_0 = base_0 + 1024U;
        carry_0 = carry_1;
    }
    if(t_2 == 0U)
    {
        *((&kernelContext_1)->fast_0+int(0)) = packed_float4(float4((as_type<float>((min(carry_0, 128U)))), 0.0f, 0.0f, 0.0f)) ;
    }
    return;
}

