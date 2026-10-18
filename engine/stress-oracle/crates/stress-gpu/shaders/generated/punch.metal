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

float3 rotate_0(const Quat_0 thread* q_2, float3 v_1)
{
    float3 qv_0 = float3(q_2->x_0, q_2->y_0, q_2->z_0);
    float3 t_0 = cross(qv_0, v_1) * float3(2.0f) ;
    return v_1 + t_0 * float3(q_2->w_0)  + cross(qv_0, t_0);
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
    Island_natural_0 device* islands_0;
    uint device* status_0;
    packed_float4 device* ctab_0;
    EngineBody_natural_0 device* bodies_0;
    ChunkStatic_natural_0 device* chunks_0;
    packed_float4 device* state_0;
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

[[kernel]] void punch_islands(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], PunchParams_0 constant* params_1 [[buffer(0)]], Island_natural_0 device* islands_1 [[buffer(1)]], uint device* status_1 [[buffer(10)]], packed_float4 device* ctab_1 [[buffer(4)]], EngineBody_natural_0 device* bodies_1 [[buffer(5)]], ChunkStatic_natural_0 device* chunks_1 [[buffer(2)]], packed_float4 device* state_1 [[buffer(3)]], packed_float4 device* impactors_1 [[buffer(6)]], BondStatic_natural_0 device* bond_static_1 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_1 [[buffer(8)]], MaterialTable_0 constant* materials_1 [[buffer(9)]])
{
    uint ax_0;
    uint hit_k_0;
    float t0_0;
    float t1_1;
    float t0_1;
    uint bi_0;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->islands_0 = islands_1;
    (&kernelContext_0)->status_0 = status_1;
    (&kernelContext_0)->ctab_0 = ctab_1;
    (&kernelContext_0)->bodies_0 = bodies_1;
    (&kernelContext_0)->chunks_0 = chunks_1;
    (&kernelContext_0)->state_0 = state_1;
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
    uint i_0 = group_0.x;
    uint t_1 = thread_0.x;
    if(i_0 >= (params_1->island_count_0))
    {
        return;
    }
    Island_natural_0 device* _S2 = (&kernelContext_0)->islands_0+i_0;
    uint4 _S3 = uint4(_S2->range_0) ;
    uint _S4 = _S3.x;
    uint n_1 = _S3.y - _S4;
    uint _S5 = _S3.z;
    uint _S6 = _S3.w;
    bool out_0;
    if(_S5 >= _S6)
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
    if(out_0)
    {
        if(t_1 == 0U)
        {
            uint device* _S7 = (&kernelContext_0)->status_0+(16U + i_0);
            *_S7 = (*_S7) & 4294967287U;
        }
        return;
    }
    uint _S8 = 3U * i_0;
    uint body_0 = (as_type<uint>(((float4(*((&kernelContext_0)->ctab_0+((&kernelContext_0)->params_0->engine_base_0 + _S8))) ).x)));
    Quat_0 _S9 = quat_of_0(float4(_S2->rotation_0) );
    float3 _S10 = (float4(_S2->position_0) ).xyz;
    float3 _S11 = (float4(_S2->velocity_0) ).xyz;
    float3 origin_0;
    float3 v_2;
    Quat_0 q_3;
    if(body_0 != 4294967295U)
    {
        Quat_0 p_inv_0 = quat_of_0(float4(*((&kernelContext_0)->ctab_0+((&kernelContext_0)->params_0->engine_base_0 + _S8 + 1U))) );
        float3 com_1 = (float4(*((&kernelContext_0)->ctab_0+((&kernelContext_0)->params_0->engine_base_0 + _S8 + 2U))) ).xyz;
        EngineBody_natural_0 b_0 = (&kernelContext_0)->bodies_0[body_0];
        thread Quat_0 a_inv_0;
        (&a_inv_0)->w_0 = 0.70710676908493042f;
        (&a_inv_0)->x_0 = 0.70710676908493042f;
        (&a_inv_0)->y_0 = 0.0f;
        (&a_inv_0)->z_0 = 0.0f;
        Quat_0 _S12 = quat_of_0(float4(b_0.rot_0) );
        thread Quat_0 _S13 = a_inv_0;
        thread Quat_0 _S14 = _S12;
        Quat_0 _S15 = quat_mul_0(&_S13, &_S14);
        thread Quat_0 _S16 = _S15;
        thread Quat_0 _S17 = p_inv_0;
        Quat_0 _S18 = quat_mul_0(&_S16, &_S17);
        thread Quat_0 _S19 = _S18;
        Quat_0 _S20 = normalized_0(&_S19);
        float3 _S21 = to_zup_0((float4(b_0.pos_inv_mass_0) ).xyz);
        thread Quat_0 _S22 = _S20;
        float3 _S23 = rotate_0(&_S22, com_1);
        float3 _S24 = to_zup_0((float4(b_0.vel_sleep_0) ).xyz);
        origin_0 = _S21 - _S23;
        q_3 = _S20;
        v_2 = _S24;
    }
    else
    {
        origin_0 = _S10;
        q_3 = _S9;
        v_2 = _S11;
    }
    float inf_0 = (as_type<float>((2139095040U)));
    float3 _S25 = float3(- inf_0) ;
    float3 lo_0 = float3(inf_0) ;
    float3 hi_0 = _S25;
    uint c_0 = t_1;
    for(;;)
    {
        if(c_0 < n_1)
        {
        }
        else
        {
            break;
        }
        uint k_0 = _S4 + c_0;
        float3 _S26 = (float4(((&kernelContext_0)->chunks_0+k_0)->center_0) ).xyz + (float4(*((&kernelContext_0)->state_0+4U * k_0)) ).xyz;
        thread Quat_0 _S27 = q_3;
        float3 _S28 = rotate_0(&_S27, _S26);
        float3 p_0 = origin_0 + _S28;
        float3 _S29 = min(lo_0, p_0 - float3((float4(((&kernelContext_0)->chunks_0+k_0)->half_0) ).w) );
        float3 _S30 = max(hi_0, p_0 + float3((float4(((&kernelContext_0)->chunks_0+k_0)->half_0) ).w) );
        uint c_1 = c_0 + 256U;
        lo_0 = _S29;
        hi_0 = _S30;
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
            uint _S31 = t_1 + s_0;
            (*(&kernelContext_0)->g_lo_0)[t_1] = min((*(&kernelContext_0)->g_lo_0)[t_1], (*(&kernelContext_0)->g_lo_0)[_S31]);
            (*(&kernelContext_0)->g_hi_0)[t_1] = max((*(&kernelContext_0)->g_hi_0)[t_1], (*(&kernelContext_0)->g_hi_0)[_S31]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    float3 _S32 = (*(&kernelContext_0)->g_lo_0)[int(0)];
    float3 _S33 = (*(&kernelContext_0)->g_hi_0)[int(0)];
    float3 _S34 = float3(0.0f) ;
    uint hit_k_1 = 4294967295U;
    float3 hit_0 = _S34;
    float3 d_0 = _S34;
    float speed_1 = 0.0f;
    uint k_1 = 0U;
    for(;;)
    {
        if(k_1 < ((&kernelContext_0)->params_0->impactor_count_0))
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
        float4 _S35 = float4(*((&kernelContext_0)->impactors_0+2U * k_1)) ;
        EngineBody_natural_0 b_1 = (&kernelContext_0)->bodies_0[(as_type<uint>((_S35.x)))];
        float3 c_2 = to_zup_0((float4(b_1.pos_inv_mass_0) ).xyz);
        float3 rel_0 = to_zup_0((float4(b_1.vel_sleep_0) ).xyz) - v_2;
        float sp_0 = length(rel_0);
        if(!(sp_0 > ((&kernelContext_0)->params_0->speed_0)))
        {
            k_1 = k_1 + 1U;
            continue;
        }
        float3 _S36 = float3(_S35.y) ;
        float3 _S37 = _S32 - _S36;
        float3 _S38 = _S33 + _S36;
        float _S39 = (&kernelContext_0)->params_0->dt_0;
        t0_0 = 0.0f;
        t1_1 = _S39;
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
            uint _S40 = ax_0;
            if((abs(rel_0[ax_0])) < 9.999999960041972e-13f)
            {
                uint _S41 = ax_0;
                bool _S42;
                if((c_2[ax_0]) < (_S37[ax_0]))
                {
                    _S42 = true;
                }
                else
                {
                    _S42 = (c_2[_S41]) > (_S38[ax_0]);
                }
                if(_S42)
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
            float ta_0 = (_S37[ax_0] - c_2[ax_0]) / rel_0[_S40];
            float tb_0 = (_S38[ax_0] - c_2[ax_0]) / rel_0[_S40];
            float _S43 = min(t1_1, max(ta_0, tb_0));
            t0_0 = max(t0_0, min(ta_0, tb_0));
            t1_1 = _S43;
            ax_0 = ax_0 + 1U;
        }
        float3 hit_1;
        float3 d_1;
        if(t0_0 <= t1_1)
        {
            float3 _S44 = c_2 + rel_0 * float3(t0_0) ;
            float3 _S45 = rel_0 / float3(sp_0) ;
            hit_k_0 = k_1;
            hit_1 = _S44;
            d_1 = _S45;
            t0_1 = sp_0;
        }
        else
        {
            hit_k_0 = hit_k_1;
            hit_1 = hit_0;
            d_1 = d_0;
            t0_1 = speed_1;
        }
        hit_k_1 = hit_k_0;
        hit_0 = hit_1;
        d_0 = d_1;
        speed_1 = t0_1;
        k_1 = k_1 + 1U;
    }
    if(hit_k_1 == 4294967295U)
    {
        if(t_1 == 0U)
        {
            uint device* _S46 = (&kernelContext_0)->status_0+(16U + i_0);
            *_S46 = (*_S46) & 4294967287U;
        }
        return;
    }
    uint _S47 = 2U * hit_k_1;
    float4 _S48 = float4(*((&kernelContext_0)->impactors_0+_S47)) ;
    float4 _S49 = float4(*((&kernelContext_0)->impactors_0+(_S47 + 1U))) ;
    float face_0 = _S48.z;
    t0_0 = inf_0;
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
        uint k_2 = _S4 + c_0;
        float3 _S50 = (float4(((&kernelContext_0)->chunks_0+k_2)->center_0) ).xyz + (float4(*((&kernelContext_0)->state_0+4U * k_2)) ).xyz;
        thread Quat_0 _S51 = q_3;
        float3 _S52 = rotate_0(&_S51, _S50);
        float3 r_3 = origin_0 + _S52 - hit_0;
        float along_0 = dot(r_3, d_0);
        float dist_0 = length(r_3 - d_0 * float3(along_0) ) + abs(along_0) * 0.00100000004749745f;
        if(dist_0 < t0_0)
        {
            t0_0 = dist_0;
            ax_0 = c_0;
        }
        c_0 = c_0 + 256U;
    }
    (*(&kernelContext_0)->g_a_0)[t_1] = t0_0;
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
            uint _S53 = t_1 + s_0;
            if(((*(&kernelContext_0)->g_a_0)[_S53]) < (*(&kernelContext_0)->g_a_0)[t_1])
            {
                out_0 = true;
            }
            else
            {
                if(((*(&kernelContext_0)->g_a_0)[_S53]) == (*(&kernelContext_0)->g_a_0)[t_1])
                {
                    out_0 = ((*(&kernelContext_0)->g_idx_0)[_S53]) < (*(&kernelContext_0)->g_idx_0)[t_1];
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
            uint _S54 = t_1 + s_0;
            (*(&kernelContext_0)->g_a_0)[t_1] = (*(&kernelContext_0)->g_a_0)[_S54];
            (*(&kernelContext_0)->g_idx_0)[t_1] = (*(&kernelContext_0)->g_idx_0)[_S54];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    uint entry_0 = (*(&kernelContext_0)->g_idx_0)[int(0)];
    uint entry_k_0 = _S4 + (*(&kernelContext_0)->g_idx_0)[int(0)];
    float3 _S55 = (float4(((&kernelContext_0)->chunks_0+entry_k_0)->center_0) ).xyz + (float4(*((&kernelContext_0)->state_0+4U * entry_k_0)) ).xyz;
    thread Quat_0 _S56 = q_3;
    float3 _S57 = rotate_0(&_S56, _S55);
    float _S58 = dot(origin_0 + _S57 - hit_0, d_0);
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
        bi_0 = _S5 + t_1;
        for(;;)
        {
            if(bi_0 < _S6)
            {
            }
            else
            {
                break;
            }
            JointBond_natural_0 law_1 = ((&kernelContext_0)->bond_static_0+bi_0)->law_0;
            bool _S59 = ((float4(((&kernelContext_0)->bond_static_0+bi_0)->law_0.stiff1_0) ).w) != 0.0f;
            thread JointState_0 _S60 = ((&kernelContext_0)->bond_dyn_0+bi_0)->js_0;
            bool _S61 = connected_0(&_S60, _S59);
            if(!_S61)
            {
                bi_0 = bi_0 + 256U;
                continue;
            }
            uint4 _S62 = uint4(law_1.ids_0) ;
            uint ca_0 = _S62.y - _S4;
            uint cb_0 = _S62.z - _S4;
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
            uint ok_0 = _S4 + o_1;
            float3 _S63 = (float4(((&kernelContext_0)->chunks_0+ok_0)->center_0) ).xyz + (float4(*((&kernelContext_0)->state_0+4U * ok_0)) ).xyz;
            thread Quat_0 _S64 = q_3;
            float3 _S65 = rotate_0(&_S64, _S63);
            float3 r_4 = origin_0 + _S65 - hit_0;
            float along_1 = dot(r_4, d_0);
            float lateral_0 = length(r_4 - d_0 * float3(along_1) );
            if(along_1 >= (_S58 - (float4(((&kernelContext_0)->chunks_0+ok_0)->half_0) ).w))
            {
                out_0 = lateral_0 <= (face_0 + max(along_1 - _S58, 0.0f));
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
        if((*(&kernelContext_0)->g_changed_0) == 0U)
        {
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        hit_k_0 = hit_k_0 + 1U;
    }
    c_0 = t_1;
    t1_1 = 0.0f;
    t0_1 = 0.0f;
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
            float count_0 = t0_1 + 1.0f;
            t1_1 = t1_1 + (float4(((&kernelContext_0)->chunks_0+(_S4 + c_0))->center_0) ).w;
            t0_1 = count_0;
        }
        c_0 = c_0 + 256U;
    }
    uint _S66 = _S5 + t_1;
    bi_0 = _S66;
    float capacity_0 = 0.0f;
    for(;;)
    {
        if(bi_0 < _S6)
        {
        }
        else
        {
            break;
        }
        JointBond_natural_0 law_2 = ((&kernelContext_0)->bond_static_0+bi_0)->law_0;
        bool _S67 = ((float4(((&kernelContext_0)->bond_static_0+bi_0)->law_0.stiff1_0) ).w) != 0.0f;
        thread JointState_0 _S68 = ((&kernelContext_0)->bond_dyn_0+bi_0)->js_0;
        bool _S69 = connected_0(&_S68, _S67);
        if(!_S69)
        {
            bi_0 = bi_0 + 256U;
            continue;
        }
        uint4 _S70 = uint4(law_2.ids_0) ;
        float capacity_1;
        if(((*(&kernelContext_0)->g_plug_0)[_S70.y - _S4]) != (*(&kernelContext_0)->g_plug_0)[_S70.z - _S4])
        {
            capacity_1 = capacity_0 + (&(&kernelContext_0)->materials_0->m_0[_S70.x])->strength_0.x * (float4(law_2.geom0_0) ).x * (float4(law_2.geom1_0) ).w;
        }
        else
        {
            capacity_1 = capacity_0;
        }
        capacity_0 = capacity_1;
        bi_0 = bi_0 + 256U;
    }
    (*(&kernelContext_0)->g_a_0)[t_1] = t1_1 + 0.0f * t0_1;
    (*(&kernelContext_0)->g_b_0)[t_1] = capacity_0;
    (*(&kernelContext_0)->g_lo_0)[t_1] = float3(t0_1, 0.0f, 0.0f);
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
            uint _S71 = t_1 + s_0;
            (*(&kernelContext_0)->g_a_0)[t_1] = (*(&kernelContext_0)->g_a_0)[t_1] + (*(&kernelContext_0)->g_a_0)[_S71];
            (*(&kernelContext_0)->g_b_0)[t_1] = (*(&kernelContext_0)->g_b_0)[t_1] + (*(&kernelContext_0)->g_b_0)[_S71];
            (*(&kernelContext_0)->g_lo_0)[t_1] = (*(&kernelContext_0)->g_lo_0)[t_1] + (*(&kernelContext_0)->g_lo_0)[_S71];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    if(t_1 == 0U)
    {
        float m_imp_0 = _S48.w;
        float m_eff_0 = (*(&kernelContext_0)->g_a_0)[int(0)] * m_imp_0 / ((*(&kernelContext_0)->g_a_0)[int(0)] + m_imp_0);
        float impulse_0 = m_eff_0 * speed_1;
        float modulus_0 = 1.0f / (1.0f / _S49.x + 1.0f / (float4(((&kernelContext_0)->chunks_0+entry_k_0)->cmat_0) ).x);
        float tau_0 = 2.86999988555908203f * pow(m_eff_0 * m_eff_0 / (face_0 * modulus_0 * modulus_0 * max(speed_1, 0.00100000004749745f)), 0.20000000298023224f);
        if(((*(&kernelContext_0)->g_lo_0)[int(0)].x) < float(n_1))
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
            uint device* _S72 = (&kernelContext_0)->status_0+(16U + i_0);
            *_S72 = (*_S72) | 8U;
            *((&kernelContext_0)->status_0+1U) = 1U;
        }
        else
        {
            uint device* _S73 = (&kernelContext_0)->status_0+(16U + i_0);
            *_S73 = (*_S73) & 4294967287U;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if((*(&kernelContext_0)->g_go_0) == 0U)
    {
        return;
    }
    bi_0 = _S66;
    for(;;)
    {
        if(bi_0 < _S6)
        {
        }
        else
        {
            break;
        }
        JointBond_natural_0 law_3 = ((&kernelContext_0)->bond_static_0+bi_0)->law_0;
        BondDyn_natural_0 device* _S74 = (&kernelContext_0)->bond_dyn_0+bi_0;
        float4 _S75 = float4((*_S74).force_lin_0) ;
        float4 _S76 = float4((*_S74).force_ang_0) ;
        float4 _S77 = float4((*_S74).sums_0) ;
        float4 _S78 = float4((*_S74).comps_0) ;
        uint4 _S79 = uint4((*_S74).events_0) ;
        thread BondDyn_0 dyn_0;
        (&dyn_0)->js_0 = (*_S74).js_0;
        (&dyn_0)->force_lin_0 = _S75;
        (&dyn_0)->force_ang_0 = _S76;
        (&dyn_0)->sums_0 = _S77;
        (&dyn_0)->comps_0 = _S78;
        (&dyn_0)->events_0 = _S79;
        bool _S80 = ((float4(((&kernelContext_0)->bond_static_0+bi_0)->law_0.stiff1_0) ).w) != 0.0f;
        thread JointState_0 _S81 = (&dyn_0)->js_0;
        bool _S82 = connected_0(&_S81, _S80);
        if(!_S82)
        {
            bi_0 = bi_0 + 256U;
            continue;
        }
        uint4 _S83 = uint4(law_3.ids_0) ;
        if(((*(&kernelContext_0)->g_plug_0)[_S83.y - _S4]) != (*(&kernelContext_0)->g_plug_0)[_S83.z - _S4])
        {
            (&(&dyn_0)->js_0)->damage_0 = 1.0f;
            (&(&dyn_0)->js_0)->rebar_broken_0 = 1.0f;
            (&dyn_0)->events_0.z = max((&dyn_0)->events_0.z, 1U);
            BondDyn_natural_0 device* _S84 = (&kernelContext_0)->bond_dyn_0+bi_0;
            _S84->js_0 = dyn_0.js_0;
            _S84->force_lin_0 = packed_float4(dyn_0.force_lin_0) ;
            _S84->force_ang_0 = packed_float4(dyn_0.force_ang_0) ;
            _S84->sums_0 = packed_float4(dyn_0.sums_0) ;
            _S84->comps_0 = packed_float4(dyn_0.comps_0) ;
            _S84->events_0 = packed_uint4(dyn_0.events_0) ;
        }
        bi_0 = bi_0 + 256U;
    }
    return;
}

