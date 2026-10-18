#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct SplitParams_0
{
    uint parent_count_0;
    uint out_base_0;
    uint pad0_0;
    uint pad1_0;
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
    SplitParams_0 constant* params_0;
    packed_uint4 device* parents_0;
    Island_natural_0 device* islands_0;
    uint device* work_0;
    uint device* index_0;
    BondDyn_natural_0 device* bond_dyn_0;
    BondStatic_natural_0 device* bonds_0;
    packed_float4 device* out_0;
    packed_float4 device* frames_0;
    ChunkStatic_natural_0 device* chunks_0;
    packed_float4 device* state_0;
    packed_float4 device* parent_bodies_0;
    packed_float4 device* engine_bodies_0;
    uint threadgroup* g_changed_0;
};

uint total_chunks_0(KernelContext_0 thread* kernelContext_0)
{
    return (uint4(*(kernelContext_0->parents_0+kernelContext_0->params_0->parent_count_0)) ).y;
}

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

[[kernel]] void split_components(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], SplitParams_0 constant* params_1 [[buffer(0)]], packed_uint4 device* parents_1 [[buffer(7)]], Island_natural_0 device* islands_1 [[buffer(5)]], uint device* work_1 [[buffer(8)]], uint device* index_1 [[buffer(2)]], BondDyn_natural_0 device* bond_dyn_1 [[buffer(4)]], BondStatic_natural_0 device* bonds_1 [[buffer(3)]], packed_float4 device* out_1 [[buffer(9)]], packed_float4 device* frames_1 [[buffer(10)]], ChunkStatic_natural_0 device* chunks_1 [[buffer(1)]], packed_float4 device* state_1 [[buffer(6)]], packed_float4 device* parent_bodies_1 [[buffer(12)]], packed_float4 device* engine_bodies_1 [[buffer(11)]])
{
    uint children_0;
    uint k_0;
    uint at_0;
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->params_0 = params_1;
    (&kernelContext_1)->parents_0 = parents_1;
    (&kernelContext_1)->islands_0 = islands_1;
    (&kernelContext_1)->work_0 = work_1;
    (&kernelContext_1)->index_0 = index_1;
    (&kernelContext_1)->bond_dyn_0 = bond_dyn_1;
    (&kernelContext_1)->bonds_0 = bonds_1;
    (&kernelContext_1)->out_0 = out_1;
    (&kernelContext_1)->frames_0 = frames_1;
    (&kernelContext_1)->chunks_0 = chunks_1;
    (&kernelContext_1)->state_0 = state_1;
    (&kernelContext_1)->parent_bodies_0 = parent_bodies_1;
    (&kernelContext_1)->engine_bodies_0 = engine_bodies_1;
    threadgroup uint g_changed_1;
    (&kernelContext_1)->g_changed_0 = &g_changed_1;
    uint tid_0 = thread_0.x;
    uint p_0 = group_0.x;
    if(p_0 >= (params_1->parent_count_0))
    {
        return;
    }
    uint4 _S2 = uint4(*((&kernelContext_1)->parents_0+p_0)) ;
    uint _S3 = _S2.x;
    uint4 _S4 = uint4(((&kernelContext_1)->islands_0+_S3)->range_0) ;
    uint _S5 = _S4.x;
    uint _S6 = _S4.y - _S5;
    uint _S7 = _S2.y;
    uint _S8 = total_chunks_0(&kernelContext_1);
    uint c_0 = tid_0;
    for(;;)
    {
        if(c_0 < _S6)
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_1)->work_0+(_S7 + c_0)) = c_0;
        c_0 = c_0 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint pass_0 = 0U;
    for(;;)
    {
        if(pass_0 <= _S6)
        {
        }
        else
        {
            break;
        }
        if(tid_0 == 0U)
        {
            *(&kernelContext_1)->g_changed_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        c_0 = tid_0;
        for(;;)
        {
            if(c_0 < _S6)
            {
            }
            else
            {
                break;
            }
            uint _S9 = _S7 + c_0;
            uint gc_0 = _S5 + c_0;
            uint _S10 = (&kernelContext_1)->index_0[gc_0];
            children_0 = *((&kernelContext_1)->work_0+_S9);
            k_0 = _S10;
            for(;;)
            {
                if(k_0 < ((&kernelContext_1)->index_0)[gc_0 + 1U])
                {
                }
                else
                {
                    break;
                }
                uint entry_0 = (&kernelContext_1)->index_0[k_0];
                uint bond_0 = entry_0 >> 1U;
                bool _S11 = ((float4(((&kernelContext_1)->bonds_0+bond_0)->law_0.stiff1_0) ).w) != 0.0f;
                thread JointState_0 _S12 = ((&kernelContext_1)->bond_dyn_0+bond_0)->js_0;
                bool _S13 = connected_0(&_S12, _S11);
                if(!_S13)
                {
                    k_0 = k_0 + 1U;
                    continue;
                }
                if((entry_0 & 1U) == 0U)
                {
                    at_0 = (uint4(((&kernelContext_1)->bonds_0+bond_0)->law_0.ids_0) ).z;
                }
                else
                {
                    at_0 = (uint4(((&kernelContext_1)->bonds_0+bond_0)->law_0.ids_0) ).y;
                }
                bool _S14;
                if(at_0 < _S5)
                {
                    _S14 = true;
                }
                else
                {
                    _S14 = at_0 >= (_S5 + _S6);
                }
                if(_S14)
                {
                    k_0 = k_0 + 1U;
                    continue;
                }
                children_0 = min(children_0, *((&kernelContext_1)->work_0+(_S7 + at_0 - _S5)));
                k_0 = k_0 + 1U;
            }
            uint _S15 = min(children_0, *((&kernelContext_1)->work_0+(_S7 + children_0)));
            if(_S15 < (*((&kernelContext_1)->work_0+_S9)))
            {
                *((&kernelContext_1)->work_0+_S9) = _S15;
                *(&kernelContext_1)->g_changed_0 = 1U;
            }
            c_0 = c_0 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        bool done_1 = (*(&kernelContext_1)->g_changed_0) == 0U;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(done_1)
        {
            break;
        }
        pass_0 = pass_0 + 1U;
    }
    if(tid_0 != 0U)
    {
        return;
    }
    c_0 = 0U;
    children_0 = 0U;
    for(;;)
    {
        if(c_0 < _S6)
        {
        }
        else
        {
            break;
        }
        if((*((&kernelContext_1)->work_0+(_S7 + c_0))) == c_0)
        {
            *((&kernelContext_1)->work_0+(_S8 + _S7 + c_0)) = children_0;
            children_0 = children_0 + 1U;
        }
        c_0 = c_0 + 1U;
    }
    k_0 = 0U;
    for(;;)
    {
        if(k_0 < children_0)
        {
        }
        else
        {
            break;
        }
        float _S16 = (as_type<float>((0U)));
        *((&kernelContext_1)->out_0+((&kernelContext_1)->params_0->out_base_0 + 8U * (_S7 + k_0))) = packed_float4(float4((as_type<float>((p_0))), _S16, _S16, _S16)) ;
        k_0 = k_0 + 1U;
    }
    c_0 = 0U;
    for(;;)
    {
        if(c_0 < _S6)
        {
        }
        else
        {
            break;
        }
        uint _S17 = 8U * (_S7 + *((&kernelContext_1)->work_0+(_S8 + _S7 + *((&kernelContext_1)->work_0+(_S7 + c_0)))));
        float4 _S18 = float4(*((&kernelContext_1)->out_0+((&kernelContext_1)->params_0->out_base_0 + _S17))) ;
        thread float4 rec_0 = _S18;
        rec_0.z = (as_type<float>(((as_type<uint>((_S18.z))) + 1U)));
        *((&kernelContext_1)->out_0+((&kernelContext_1)->params_0->out_base_0 + _S17)) = packed_float4(rec_0) ;
        c_0 = c_0 + 1U;
    }
    k_0 = 0U;
    at_0 = 0U;
    for(;;)
    {
        if(k_0 < children_0)
        {
        }
        else
        {
            break;
        }
        uint _S19 = 8U * (_S7 + k_0);
        thread float4 rec_1 = float4(*((&kernelContext_1)->out_0+((&kernelContext_1)->params_0->out_base_0 + _S19))) ;
        rec_1.y = (as_type<float>((at_0)));
        uint at_1 = at_0 + (as_type<uint>((rec_1.z)));
        rec_1.z = (as_type<float>((0U)));
        *((&kernelContext_1)->out_0+((&kernelContext_1)->params_0->out_base_0 + _S19)) = packed_float4(rec_1) ;
        k_0 = k_0 + 1U;
        at_0 = at_1;
    }
    c_0 = 0U;
    for(;;)
    {
        if(c_0 < _S6)
        {
        }
        else
        {
            break;
        }
        uint _S20 = _S7 + c_0;
        *((&kernelContext_1)->work_0+_S20) = *((&kernelContext_1)->work_0+(_S8 + _S7 + *((&kernelContext_1)->work_0+_S20)));
        c_0 = c_0 + 1U;
    }
    c_0 = 0U;
    for(;;)
    {
        if(c_0 < _S6)
        {
        }
        else
        {
            break;
        }
        uint _S21 = 8U * (_S7 + *((&kernelContext_1)->work_0+(_S7 + c_0)));
        float4 _S22 = float4(*((&kernelContext_1)->out_0+((&kernelContext_1)->params_0->out_base_0 + _S21))) ;
        thread float4 rec_2 = _S22;
        uint _S23 = (as_type<uint>((_S22.z)));
        *((&kernelContext_1)->work_0+(_S8 + _S7 + ((as_type<uint>((_S22.y))) + _S23))) = c_0;
        rec_2.z = (as_type<float>((_S23 + 1U)));
        *((&kernelContext_1)->out_0+((&kernelContext_1)->params_0->out_base_0 + _S21)) = packed_float4(rec_2) ;
        c_0 = c_0 + 1U;
    }
    *((&kernelContext_1)->parents_0+p_0) = packed_uint4(uint4(_S3, _S7, children_0, 0U)) ;
    return;
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

float3 rotate_0(const Quat_0 thread* q_1, float3 v_0)
{
    float3 qv_0 = float3(q_1->x_0, q_1->y_0, q_1->z_0);
    float3 t_0 = cross(qv_0, v_0) * float3(2.0f) ;
    return v_0 + t_0 * float3(q_1->w_0)  + cross(qv_0, t_0);
}

float3 rotate_1(const Quat_0 thread* q_2, float3 v_1)
{
    float3 qv_1 = float3(q_2->x_0, q_2->y_0, q_2->z_0);
    float3 t_1 = cross(qv_1, v_1) * float3(2.0f) ;
    return v_1 + t_1 * float3(q_2->w_0)  + cross(qv_1, t_1);
}

float3 inverse_rotate_0(const Quat_0 thread* q_3, float3 v_2)
{
    thread Quat_0 c_1;
    (&c_1)->w_0 = q_3->w_0;
    (&c_1)->x_0 = - q_3->x_0;
    (&c_1)->y_0 = - q_3->y_0;
    (&c_1)->z_0 = - q_3->z_0;
    thread Quat_0 _S24 = c_1;
    float3 _S25 = rotate_0(&_S24, v_2);
    return _S25;
}

float3 mat_mul_0(float3 r0_0, float3 r1_0, float3 r2_0, float3 v_3)
{
    return float3(dot(r0_0, v_3), dot(r1_0, v_3), dot(r2_0, v_3));
}

void rotate_matrix_0(const Quat_0 thread* q_4, float3 m0_0, float3 m1_0, float3 m2_0, float3 thread* o0_0, float3 thread* o1_0, float3 thread* o2_0)
{
    float3 _S26 = inverse_rotate_0(q_4, float3(1.0f, 0.0f, 0.0f));
    float3 _S27 = rotate_1(q_4, mat_mul_0(m0_0, m1_0, m2_0, _S26));
    float3 _S28 = inverse_rotate_0(q_4, float3(0.0f, 1.0f, 0.0f));
    float3 _S29 = rotate_1(q_4, mat_mul_0(m0_0, m1_0, m2_0, _S28));
    float3 _S30 = inverse_rotate_0(q_4, float3(0.0f, 0.0f, 1.0f));
    float3 _S31 = rotate_1(q_4, mat_mul_0(m0_0, m1_0, m2_0, _S30));
    *o0_0 = float3(_S27.x, _S29.x, _S31.x);
    *o1_0 = float3(_S27.y, _S29.y, _S31.y);
    *o2_0 = float3(_S27.z, _S29.z, _S31.z);
    return;
}

void inverse3_0(float3 r0_1, float3 r1_1, float3 r2_1, float3 thread* i0_0, float3 thread* i1_0, float3 thread* i2_0)
{
    float3 c0_0 = cross(r1_1, r2_1);
    float3 c1_0 = cross(r2_1, r0_1);
    float3 c2_0 = cross(r0_1, r1_1);
    float det_0 = dot(r0_1, c0_0);
    float inv_0;
    if(det_0 != 0.0f)
    {
        inv_0 = 1.0f / det_0;
    }
    else
    {
        inv_0 = 0.0f;
    }
    *i0_0 = float3(c0_0.x, c1_0.x, c2_0.x) * float3(inv_0) ;
    *i1_0 = float3(c0_0.y, c1_0.y, c2_0.y) * float3(inv_0) ;
    *i2_0 = float3(c0_0.z, c1_0.z, c2_0.z) * float3(inv_0) ;
    return;
}

[[kernel]] void split_children(uint3 id_0 [[thread_position_in_grid]], SplitParams_0 constant* params_2 [[buffer(0)]], packed_uint4 device* parents_2 [[buffer(7)]], Island_natural_0 device* islands_2 [[buffer(5)]], uint device* work_2 [[buffer(8)]], uint device* index_2 [[buffer(2)]], BondDyn_natural_0 device* bond_dyn_2 [[buffer(4)]], BondStatic_natural_0 device* bonds_2 [[buffer(3)]], packed_float4 device* out_2 [[buffer(9)]], packed_float4 device* frames_2 [[buffer(10)]], ChunkStatic_natural_0 device* chunks_2 [[buffer(1)]], packed_float4 device* state_2 [[buffer(6)]], packed_float4 device* parent_bodies_2 [[buffer(12)]], packed_float4 device* engine_bodies_2 [[buffer(11)]])
{
    bool anchored_0;
    thread KernelContext_0 kernelContext_2;
    (&kernelContext_2)->params_0 = params_2;
    (&kernelContext_2)->parents_0 = parents_2;
    (&kernelContext_2)->islands_0 = islands_2;
    (&kernelContext_2)->work_0 = work_2;
    (&kernelContext_2)->index_0 = index_2;
    (&kernelContext_2)->bond_dyn_0 = bond_dyn_2;
    (&kernelContext_2)->bonds_0 = bonds_2;
    (&kernelContext_2)->out_0 = out_2;
    (&kernelContext_2)->frames_0 = frames_2;
    (&kernelContext_2)->chunks_0 = chunks_2;
    (&kernelContext_2)->state_0 = state_2;
    (&kernelContext_2)->parent_bodies_0 = parent_bodies_2;
    (&kernelContext_2)->engine_bodies_0 = engine_bodies_2;
    threadgroup uint g_changed_2;
    (&kernelContext_2)->g_changed_0 = &g_changed_2;
    uint slot_0 = id_0.x;
    uint _S32 = total_chunks_0(&kernelContext_2);
    if(slot_0 >= _S32)
    {
        return;
    }
    uint p_1 = 0U;
    for(;;)
    {
        uint _S33 = p_1 + 1U;
        if(_S33 < ((&kernelContext_2)->params_0->parent_count_0))
        {
            anchored_0 = ((uint4(*((&kernelContext_2)->parents_0+_S33)) ).y) <= slot_0;
        }
        else
        {
            anchored_0 = false;
        }
        if(anchored_0)
        {
        }
        else
        {
            break;
        }
        p_1 = _S33;
    }
    uint4 _S34 = uint4(*((&kernelContext_2)->parents_0+p_1)) ;
    uint _S35 = _S34.y;
    if((slot_0 - _S35) >= (_S34.z))
    {
        return;
    }
    uint at_2 = (&kernelContext_2)->params_0->out_base_0 + 8U * slot_0;
    float4 _S36 = float4(*((&kernelContext_2)->out_0+at_2)) ;
    uint offset_0 = (as_type<uint>((_S36.y)));
    uint count_0 = (as_type<uint>((_S36.z)));
    uint _S37 = (uint4(((&kernelContext_2)->islands_0+_S34.x)->range_0) ).x;
    uint _S38 = 5U * p_1;
    Quat_0 q_5 = quat_of_0(float4(*((&kernelContext_2)->frames_0+_S38)) );
    float3 _S39 = (float4(*((&kernelContext_2)->frames_0+(_S38 + 1U))) ).xyz;
    float3 _S40 = (float4(*((&kernelContext_2)->frames_0+(_S38 + 2U))) ).xyz;
    float3 _S41 = (float4(*((&kernelContext_2)->frames_0+(_S38 + 3U))) ).xyz;
    float3 _S42 = float3(0.0f) ;
    anchored_0 = false;
    uint j_0 = 0U;
    float mass_0 = 0.0f;
    float3 moment_0 = _S42;
    for(;;)
    {
        if(j_0 < count_0)
        {
        }
        else
        {
            break;
        }
        ChunkStatic_natural_0 device* _S43 = (&kernelContext_2)->chunks_0+(_S37 + *((&kernelContext_2)->work_0+(_S32 + _S35 + offset_0 + j_0)));
        float4 _S44 = float4(_S43->center_0) ;
        float _S45 = _S44.w;
        float mass_1 = mass_0 + _S45;
        float3 moment_1 = moment_0 + _S44.xyz * float3(_S45) ;
        if(anchored_0)
        {
            anchored_0 = true;
        }
        else
        {
            anchored_0 = ((uint4(_S43->info_1) ).x) != 0U;
        }
        j_0 = j_0 + 1U;
        mass_0 = mass_1;
        moment_0 = moment_1;
    }
    float3 com_1 = moment_0 / float3(mass_0) ;
    j_0 = 0U;
    float3 i0_1 = _S42;
    float3 i1_1 = _S42;
    float3 i2_1 = _S42;
    float3 pm_0 = _S42;
    float3 l_0 = _S42;
    for(;;)
    {
        if(j_0 < count_0)
        {
        }
        else
        {
            break;
        }
        uint c_2 = _S37 + *((&kernelContext_2)->work_0+(_S32 + _S35 + offset_0 + j_0));
        ChunkStatic_natural_0 device* _S46 = (&kernelContext_2)->chunks_0+c_2;
        float4 _S47 = float4(_S46->center_0) ;
        float3 _S48 = _S47.xyz;
        float3 r_1 = _S48 - com_1;
        float rr_0 = dot(r_1, r_1);
        float3 _S49 = (float4(_S46->inertia0_1) ).xyz;
        float3 _S50 = float3(_S47.w) ;
        float3 i0_2 = i0_1 + (_S49 + (float3(rr_0, 0.0f, 0.0f) - float3(r_1.x)  * r_1) * _S50);
        float3 _S51 = (float4(_S46->inertia1_1) ).xyz;
        float3 i1_2 = i1_1 + (_S51 + (float3(0.0f, rr_0, 0.0f) - float3(r_1.y)  * r_1) * _S50);
        float3 _S52 = (float4(_S46->inertia2_1) ).xyz;
        float3 i2_2 = i2_1 + (_S52 + (float3(0.0f, 0.0f, rr_0) - float3(r_1.z)  * r_1) * _S50);
        uint _S53 = 4U * c_2;
        float3 _S54 = _S48 + (float4(*((&kernelContext_2)->state_0+_S53)) ).xyz - _S41;
        thread Quat_0 _S55 = q_5;
        float3 _S56 = rotate_0(&_S55, _S54);
        float3 _S57 = _S39 + cross(_S40, _S56);
        float3 _S58 = (float4(*((&kernelContext_2)->state_0+(_S53 + 2U))) ).xyz;
        thread Quat_0 _S59 = q_5;
        float3 _S60 = rotate_0(&_S59, _S58);
        float3 v_4 = _S57 + _S60;
        float3 _S61 = (float4(*((&kernelContext_2)->state_0+(_S53 + 3U))) ).xyz;
        thread Quat_0 _S62 = q_5;
        float3 _S63 = rotate_0(&_S62, _S61);
        float3 w_1 = _S40 + _S63;
        float3 pm_1 = pm_0 + v_4 * _S50;
        thread Quat_0 _S64 = q_5;
        float3 _S65 = rotate_0(&_S64, r_1);
        thread Quat_0 _S66 = q_5;
        thread float3 ci0_0;
        thread float3 ci1_0;
        thread float3 ci2_0;
        rotate_matrix_0(&_S66, _S49, _S51, _S52, &ci0_0, &ci1_0, &ci2_0);
        float3 l_1 = l_0 + (cross(_S65, v_4) * _S50 + mat_mul_0(ci0_0, ci1_0, ci2_0, w_1));
        j_0 = j_0 + 1U;
        i0_1 = i0_2;
        i1_1 = i1_2;
        i2_1 = i2_2;
        pm_0 = pm_1;
        l_0 = l_1;
    }
    float3 cv_0;
    float3 cw_0;
    if(!anchored_0)
    {
        thread Quat_0 _S67 = q_5;
        thread float3 w0_0;
        thread float3 w1_0;
        thread float3 w2_0;
        rotate_matrix_0(&_S67, i0_1, i1_1, i2_1, &w0_0, &w1_0, &w2_0);
        thread float3 v0_0;
        thread float3 v1_0;
        thread float3 v2_0;
        inverse3_0(w0_0, w1_0, w2_0, &v0_0, &v1_0, &v2_0);
        float3 _S68 = mat_mul_0(v0_0, v1_0, v2_0, l_0);
        cv_0 = pm_0 / float3(mass_0) ;
        cw_0 = _S68;
    }
    else
    {
        cv_0 = _S42;
        cw_0 = _S42;
    }
    packed_float4 device* _S69 = (&kernelContext_2)->out_0+at_2;
    float _S70 = (as_type<float>((p_1)));
    float _S71 = (as_type<float>((offset_0)));
    float _S72 = (as_type<float>((count_0)));
    if(anchored_0)
    {
        p_1 = 1U;
    }
    else
    {
        p_1 = 0U;
    }
    *_S69 = packed_float4(float4(_S70, _S71, _S72, (as_type<float>((p_1))))) ;
    *((&kernelContext_2)->out_0+(at_2 + 1U)) = packed_float4(float4(com_1, mass_0)) ;
    *((&kernelContext_2)->out_0+(at_2 + 2U)) = packed_float4(float4(i0_1, 0.0f)) ;
    *((&kernelContext_2)->out_0+(at_2 + 3U)) = packed_float4(float4(i1_1, 0.0f)) ;
    *((&kernelContext_2)->out_0+(at_2 + 4U)) = packed_float4(float4(i2_1, 0.0f)) ;
    *((&kernelContext_2)->out_0+(at_2 + 5U)) = packed_float4(float4(cv_0, 0.0f)) ;
    *((&kernelContext_2)->out_0+(at_2 + 6U)) = packed_float4(float4(cw_0, 0.0f)) ;
    *((&kernelContext_2)->out_0+(at_2 + 7U)) = packed_float4(float4(0.0f) ) ;
    return;
}

Quat_0 quat_mul_0(const Quat_0 thread* a_0, const Quat_0 thread* o_0)
{
    thread Quat_0 r_2;
    (&r_2)->w_0 = a_0->w_0 * o_0->w_0 - a_0->x_0 * o_0->x_0 - a_0->y_0 * o_0->y_0 - a_0->z_0 * o_0->z_0;
    (&r_2)->x_0 = a_0->w_0 * o_0->x_0 + a_0->x_0 * o_0->w_0 + a_0->y_0 * o_0->z_0 - a_0->z_0 * o_0->y_0;
    (&r_2)->y_0 = a_0->w_0 * o_0->y_0 - a_0->x_0 * o_0->z_0 + a_0->y_0 * o_0->w_0 + a_0->z_0 * o_0->x_0;
    (&r_2)->z_0 = a_0->w_0 * o_0->z_0 + a_0->x_0 * o_0->y_0 - a_0->y_0 * o_0->x_0 + a_0->z_0 * o_0->w_0;
    return r_2;
}

Quat_0 normalized_0(const Quat_0 thread* q_6)
{
    float n_0 = sqrt(q_6->w_0 * q_6->w_0 + q_6->x_0 * q_6->x_0 + q_6->y_0 * q_6->y_0 + q_6->z_0 * q_6->z_0);
    thread Quat_0 r_3;
    (&r_3)->w_0 = q_6->w_0 / n_0;
    (&r_3)->x_0 = q_6->x_0 / n_0;
    (&r_3)->y_0 = q_6->y_0 / n_0;
    (&r_3)->z_0 = q_6->z_0 / n_0;
    return r_3;
}

float4 quat_vec_0(const Quat_0 thread* q_7)
{
    return float4(q_7->x_0, q_7->y_0, q_7->z_0, q_7->w_0);
}

[[kernel]] void split_frames(uint3 id_1 [[thread_position_in_grid]], SplitParams_0 constant* params_3 [[buffer(0)]], packed_uint4 device* parents_3 [[buffer(7)]], Island_natural_0 device* islands_3 [[buffer(5)]], uint device* work_3 [[buffer(8)]], uint device* index_3 [[buffer(2)]], BondDyn_natural_0 device* bond_dyn_3 [[buffer(4)]], BondStatic_natural_0 device* bonds_3 [[buffer(3)]], packed_float4 device* out_3 [[buffer(9)]], packed_float4 device* frames_3 [[buffer(10)]], ChunkStatic_natural_0 device* chunks_3 [[buffer(1)]], packed_float4 device* state_3 [[buffer(6)]], packed_float4 device* parent_bodies_3 [[buffer(12)]], packed_float4 device* engine_bodies_3 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->params_0 = params_3;
    (&kernelContext_3)->parents_0 = parents_3;
    (&kernelContext_3)->islands_0 = islands_3;
    (&kernelContext_3)->work_0 = work_3;
    (&kernelContext_3)->index_0 = index_3;
    (&kernelContext_3)->bond_dyn_0 = bond_dyn_3;
    (&kernelContext_3)->bonds_0 = bonds_3;
    (&kernelContext_3)->out_0 = out_3;
    (&kernelContext_3)->frames_0 = frames_3;
    (&kernelContext_3)->chunks_0 = chunks_3;
    (&kernelContext_3)->state_0 = state_3;
    (&kernelContext_3)->parent_bodies_0 = parent_bodies_3;
    (&kernelContext_3)->engine_bodies_0 = engine_bodies_3;
    threadgroup uint g_changed_3;
    (&kernelContext_3)->g_changed_0 = &g_changed_3;
    uint p_2 = id_1.x;
    if(p_2 >= (params_3->parent_count_0))
    {
        return;
    }
    uint _S73 = 2U * p_2;
    float4 _S74 = float4(*((&kernelContext_3)->parent_bodies_0+_S73)) ;
    uint body_0 = (as_type<uint>((_S74.x)));
    if(body_0 == 4294967295U)
    {
        return;
    }
    float3 com_2 = _S74.yzw;
    Quat_0 principal_0 = quat_of_0(float4(*((&kernelContext_3)->parent_bodies_0+(_S73 + 1U))) );
    uint _S75 = 7U * body_0;
    float4 _S76 = float4(*((&kernelContext_3)->engine_bodies_0+_S75)) ;
    Quat_0 orientation_0 = quat_of_0(float4(*((&kernelContext_3)->engine_bodies_0+(_S75 + 1U))) );
    float3 lin_0 = (float4(*((&kernelContext_3)->engine_bodies_0+(_S75 + 3U))) ).xyz;
    float3 ang_0 = (float4(*((&kernelContext_3)->engine_bodies_0+(_S75 + 4U))) ).xyz;
    thread Quat_0 axes_inv_0;
    (&axes_inv_0)->w_0 = 0.70710676908493042f;
    (&axes_inv_0)->x_0 = 0.70710676908493042f;
    (&axes_inv_0)->y_0 = 0.0f;
    (&axes_inv_0)->z_0 = 0.0f;
    thread Quat_0 p_inv_0;
    (&p_inv_0)->w_0 = principal_0.w_0;
    (&p_inv_0)->x_0 = - principal_0.x_0;
    (&p_inv_0)->y_0 = - principal_0.y_0;
    (&p_inv_0)->z_0 = - principal_0.z_0;
    thread Quat_0 _S77 = axes_inv_0;
    thread Quat_0 _S78 = orientation_0;
    Quat_0 _S79 = quat_mul_0(&_S77, &_S78);
    thread Quat_0 _S80 = _S79;
    Quat_0 _S81 = normalized_0(&_S80);
    thread Quat_0 _S82 = _S81;
    thread Quat_0 _S83 = p_inv_0;
    Quat_0 _S84 = quat_mul_0(&_S82, &_S83);
    thread Quat_0 _S85 = _S84;
    Quat_0 _S86 = normalized_0(&_S85);
    float3 com_world_0 = float3(_S76.x, - _S76.z, _S76.y);
    uint _S87 = 5U * p_2;
    packed_float4 device* _S88 = (&kernelContext_3)->frames_0+_S87;
    thread Quat_0 _S89 = _S86;
    float4 _S90 = quat_vec_0(&_S89);
    *_S88 = packed_float4(_S90) ;
    *((&kernelContext_3)->frames_0+(_S87 + 1U)) = packed_float4(float4(lin_0.x, - lin_0.z, lin_0.y, 0.0f)) ;
    *((&kernelContext_3)->frames_0+(_S87 + 2U)) = packed_float4(float4(ang_0.x, - ang_0.z, ang_0.y, 0.0f)) ;
    *((&kernelContext_3)->frames_0+(_S87 + 3U)) = packed_float4(float4(com_2, 0.0f)) ;
    packed_float4 device* _S91 = (&kernelContext_3)->frames_0+(_S87 + 4U);
    thread Quat_0 _S92 = _S86;
    float3 _S93 = rotate_0(&_S92, com_2);
    *_S91 = packed_float4(float4(com_world_0 - _S93, 0.0f)) ;
    return;
}

