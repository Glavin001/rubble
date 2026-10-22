#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct RouteParams_0
{
    uint island_cap_0;
    uint impactor_count_0;
    uint engine_base_0;
    uint n_pad_0;
    float dt_0;
    float speed_0;
    uint tiered_0;
    uint fast_joins_0;
    uint debris_0;
    uint ground_0;
    float ground_height_0;
    float ground_modulus_0;
    uint edge_cap_0;
    uint always_0;
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
    uint device* counters_0;
    RouteParams_0 constant* params_0;
    uint device* route_0;
    Island_natural_0 device* islands_0;
    packed_float4 device* ctab_0;
    packed_float4 device* bodies_0;
    ChunkStatic_natural_0 device* chunks_0;
    packed_float4 device* state_0;
    uint device* rtab_0;
    BondStatic_natural_0 device* bond_static_0;
    MaterialTable_0 constant* materials_0;
    packed_float4 device* parts_0;
    packed_float4 device* imps_0;
    atomic<uint> device* actl_0;
    packed_uint2 device* sorted_0;
    atomic<uint> device* lab_0;
    uint device* seed_0;
    packed_uint4 device* pass_0;
    packed_uint2 device* edges_0;
    packed_uint4 device* isl_0;
    uint device* ctl_0;
    array<float3, int(64)> threadgroup* g_lo_0;
    array<float3, int(64)> threadgroup* g_hi_0;
    array<float4, int(64)> threadgroup* g_a_0;
    array<float, int(64)> threadgroup* g_cap_0;
};

uint live_islands_0(KernelContext_0 thread* kernelContext_0)
{
    return kernelContext_0->counters_0[int(0)];
}

bool gate_0(KernelContext_0 thread* kernelContext_1)
{
    bool _S1;
    if((kernelContext_1->params_0->always_0) != 0U)
    {
        _S1 = true;
    }
    else
    {
        _S1 = kernelContext_1->route_0[int(0)] != 0U;
    }
    return _S1;
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

float4 quat_vec_0(const Quat_0 thread* q_1)
{
    return float4(q_1->x_0, q_1->y_0, q_1->z_0, q_1->w_0);
}

float4 qmul_0(float4 a_1, float4 b_0)
{
    Quat_0 _S2 = quat_of_0(b_0);
    thread Quat_0 _S3 = quat_of_0(a_1);
    thread Quat_0 _S4 = _S2;
    Quat_0 _S5 = quat_mul_0(&_S3, &_S4);
    thread Quat_0 _S6 = _S5;
    float4 _S7 = quat_vec_0(&_S6);
    return _S7;
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

struct Frame_0
{
    float4 q_3;
    float3 origin_0;
    float3 com_1;
    float3 v_2;
    float3 w_1;
};

Frame_0 island_frame_0(uint i_0, KernelContext_0 thread* kernelContext_2)
{
    Island_natural_0 device* _S8 = kernelContext_2->islands_0+i_0;
    uint _S9 = 3U * i_0;
    uint body_0 = (as_type<uint>(((float4(*(kernelContext_2->ctab_0+(kernelContext_2->params_0->engine_base_0 + _S9))) ).x)));
    thread Frame_0 f_0;
    (&f_0)->q_3 = float4(_S8->rotation_0) ;
    (&f_0)->origin_0 = (float4(_S8->position_0) ).xyz;
    (&f_0)->v_2 = (float4(_S8->velocity_0) ).xyz;
    (&f_0)->w_1 = (float4(_S8->angular_velocity_0) ).xyz;
    if(body_0 != 4294967295U)
    {
        float3 com_2 = (float4(*(kernelContext_2->ctab_0+(kernelContext_2->params_0->engine_base_0 + _S9 + 2U))) ).xyz;
        uint _S10 = 7U * body_0;
        float4 _S11 = normalize(qmul_0(qmul_0(float4(0.70710676908493042f, 0.0f, 0.0f, 0.70710676908493042f), float4(*(kernelContext_2->bodies_0+(_S10 + 1U))) ), float4(*(kernelContext_2->ctab_0+(kernelContext_2->params_0->engine_base_0 + _S9 + 1U))) ));
        (&f_0)->q_3 = _S11;
        float3 _S12 = to_zup_0((float4(*(kernelContext_2->bodies_0+_S10)) ).xyz);
        (&f_0)->com_1 = _S12;
        thread Quat_0 _S13 = quat_of_0(_S11);
        float3 _S14 = rotate_0(&_S13, com_2);
        (&f_0)->origin_0 = _S12 - _S14;
        (&f_0)->v_2 = to_zup_0((float4(*(kernelContext_2->bodies_0+(_S10 + 3U))) ).xyz);
        (&f_0)->w_1 = to_zup_0((float4(*(kernelContext_2->bodies_0+(_S10 + 4U))) ).xyz);
    }
    else
    {
        float3 _S15 = (float4(_S8->com_0) ).xyz;
        thread Quat_0 _S16 = quat_of_0((&f_0)->q_3);
        float3 _S17 = rotate_0(&_S16, _S15);
        (&f_0)->com_1 = (&f_0)->origin_0 + _S17;
    }
    return f_0;
}

float float_inf_0()
{
    return (as_type<float>((2139095040U)));
}

struct Part_0
{
    float3 lo_0;
    float3 hi_0;
    float3 box_lo_0;
    float3 box_hi_0;
    float3 center_1;
    float3 v_3;
    float3 w_2;
    float reach_0;
    float mass_0;
    float modulus_0;
    float radius_0;
    float round_trip_0;
    float capacity_0;
    uint flags_0;
};

void store_part_0(uint i_1, const Part_0 thread* p_0, KernelContext_0 thread* kernelContext_3)
{
    uint _S18 = 8U * i_1;
    *(kernelContext_3->parts_0+_S18) = packed_float4(float4(p_0->lo_0, (as_type<float>((p_0->flags_0))))) ;
    *(kernelContext_3->parts_0+(_S18 + 1U)) = packed_float4(float4(p_0->hi_0, p_0->mass_0)) ;
    *(kernelContext_3->parts_0+(_S18 + 2U)) = packed_float4(float4(p_0->box_lo_0, p_0->reach_0)) ;
    *(kernelContext_3->parts_0+(_S18 + 3U)) = packed_float4(float4(p_0->box_hi_0, p_0->modulus_0)) ;
    *(kernelContext_3->parts_0+(_S18 + 4U)) = packed_float4(float4(p_0->center_1, p_0->radius_0)) ;
    *(kernelContext_3->parts_0+(_S18 + 5U)) = packed_float4(float4(p_0->v_3, p_0->round_trip_0)) ;
    *(kernelContext_3->parts_0+(_S18 + 6U)) = packed_float4(float4(p_0->w_2, p_0->capacity_0)) ;
    return;
}

[[kernel]] void lr_islands(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], uint device* counters_1 [[buffer(10)]], RouteParams_0 constant* params_1 [[buffer(0)]], uint device* route_1 [[buffer(11)]], Island_natural_0 device* islands_1 [[buffer(1)]], packed_float4 device* ctab_1 [[buffer(4)]], packed_float4 device* bodies_1 [[buffer(5)]], ChunkStatic_natural_0 device* chunks_1 [[buffer(2)]], packed_float4 device* state_1 [[buffer(3)]], uint device* rtab_1 [[buffer(8)]], BondStatic_natural_0 device* bond_static_1 [[buffer(6)]], MaterialTable_0 constant* materials_1 [[buffer(7)]], packed_float4 device* parts_1 [[buffer(12)]], packed_float4 device* imps_1 [[buffer(9)]], atomic<uint> device* actl_1 [[buffer(16)]], packed_uint2 device* sorted_1 [[buffer(13)]], atomic<uint> device* lab_1 [[buffer(15)]], uint device* seed_1 [[buffer(17)]], packed_uint4 device* pass_1 [[buffer(19)]], packed_uint2 device* edges_1 [[buffer(14)]], packed_uint4 device* isl_1 [[buffer(18)]], uint device* ctl_1 [[buffer(20)]])
{
    uint _S19;
    thread KernelContext_0 kernelContext_4;
    (&kernelContext_4)->counters_0 = counters_1;
    (&kernelContext_4)->params_0 = params_1;
    (&kernelContext_4)->route_0 = route_1;
    (&kernelContext_4)->islands_0 = islands_1;
    (&kernelContext_4)->ctab_0 = ctab_1;
    (&kernelContext_4)->bodies_0 = bodies_1;
    (&kernelContext_4)->chunks_0 = chunks_1;
    (&kernelContext_4)->state_0 = state_1;
    (&kernelContext_4)->rtab_0 = rtab_1;
    (&kernelContext_4)->bond_static_0 = bond_static_1;
    (&kernelContext_4)->materials_0 = materials_1;
    (&kernelContext_4)->parts_0 = parts_1;
    (&kernelContext_4)->imps_0 = imps_1;
    (&kernelContext_4)->actl_0 = actl_1;
    (&kernelContext_4)->sorted_0 = sorted_1;
    (&kernelContext_4)->lab_0 = lab_1;
    (&kernelContext_4)->seed_0 = seed_1;
    (&kernelContext_4)->pass_0 = pass_1;
    (&kernelContext_4)->edges_0 = edges_1;
    (&kernelContext_4)->isl_0 = isl_1;
    (&kernelContext_4)->ctl_0 = ctl_1;
    threadgroup array<float3, int(64)> g_lo_1;
    (&kernelContext_4)->g_lo_0 = &g_lo_1;
    threadgroup array<float3, int(64)> g_hi_1;
    (&kernelContext_4)->g_hi_0 = &g_hi_1;
    threadgroup array<float4, int(64)> g_a_1;
    (&kernelContext_4)->g_a_0 = &g_a_1;
    threadgroup array<float, int(64)> g_cap_1;
    (&kernelContext_4)->g_cap_0 = &g_cap_1;
    uint i_2 = group_0.x;
    uint t_1 = thread_0.x;
    uint _S20 = live_islands_0(&kernelContext_4);
    bool _S21;
    if(i_2 >= _S20)
    {
        _S21 = true;
    }
    else
    {
        bool _S22 = gate_0(&kernelContext_4);
        _S21 = !_S22;
    }
    if(_S21)
    {
        return;
    }
    Island_natural_0 device* _S23 = (&kernelContext_4)->islands_0+i_2;
    Frame_0 _S24 = island_frame_0(i_2, &kernelContext_4);
    uint4 _S25 = uint4(_S23->range_0) ;
    uint _S26 = _S25.x;
    uint n_0 = _S25.y - _S26;
    float inf_0 = float_inf_0();
    float3 _S27 = float3(inf_0) ;
    float3 _S28 = float3(- inf_0) ;
    thread float4 a_2 = float4(0.0f, inf_0, inf_0, inf_0);
    float3 lo_1 = _S27;
    float3 hi_1 = _S28;
    uint c_0 = t_1;
    for(;;)
    {
        if(c_0 < n_0)
        {
        }
        else
        {
            break;
        }
        uint k_0 = _S26 + c_0;
        ChunkStatic_natural_0 device* _S29 = (&kernelContext_4)->chunks_0+k_0;
        float3 _S30 = (float4(_S29->center_0) ).xyz + (float4(*((&kernelContext_4)->state_0+4U * k_0)) ).xyz;
        thread Quat_0 _S31 = quat_of_0(_S24.q_3);
        float3 _S32 = rotate_0(&_S31, _S30);
        float3 p_1 = _S24.origin_0 + _S32;
        float4 _S33 = float4(_S29->half_0) ;
        float r_2 = _S33.w;
        float3 _S34 = float3(r_2) ;
        float3 _S35 = min(lo_1, p_1 - _S34);
        float3 _S36 = max(hi_1, p_1 + _S34);
        a_2.x = max(a_2.x, length(p_1 - _S24.com_1) + r_2);
        a_2.y = min(a_2.y, (float4(_S29->cmat_0) ).x);
        uint4 _S37 = uint4(_S29->info_1) ;
        a_2.z = min(a_2.z, (as_type<float>(((&kernelContext_4)->rtab_0[2U + (&kernelContext_4)->rtab_0[int(0)] + (&kernelContext_4)->rtab_0[2U + _S37.z] + _S37.w]))));
        a_2.w = min(a_2.w, min(_S33.x, min(_S33.y, _S33.z)));
        uint c_1 = c_0 + 64U;
        lo_1 = _S35;
        hi_1 = _S36;
        c_0 = c_1;
    }
    uint _S38 = _S25.z;
    uint _S39 = _S38 + t_1;
    float cap_0 = inf_0;
    uint b_1 = _S39;
    for(;;)
    {
        uint _S40 = _S25.w;
        _S19 = _S40;
        if(b_1 < _S40)
        {
        }
        else
        {
            break;
        }
        float _S41 = min(cap_0, (&(&kernelContext_4)->materials_0->m_0[(uint4(((&kernelContext_4)->bond_static_0+b_1)->law_0.ids_0) ).x])->strength_0.x * (float4(((&kernelContext_4)->bond_static_0+b_1)->law_0.geom0_0) ).x * (float4(((&kernelContext_4)->bond_static_0+b_1)->law_0.geom1_0) ).w);
        uint b_2 = b_1 + 64U;
        cap_0 = _S41;
        b_1 = b_2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    (*(&kernelContext_4)->g_lo_0)[t_1] = lo_1;
    (*(&kernelContext_4)->g_hi_0)[t_1] = hi_1;
    (*(&kernelContext_4)->g_a_0)[t_1] = a_2;
    (*(&kernelContext_4)->g_cap_0)[t_1] = cap_0;
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
            uint _S42 = t_1 + s_0;
            (*(&kernelContext_4)->g_lo_0)[t_1] = min((*(&kernelContext_4)->g_lo_0)[t_1], (*(&kernelContext_4)->g_lo_0)[_S42]);
            (*(&kernelContext_4)->g_hi_0)[t_1] = max((*(&kernelContext_4)->g_hi_0)[t_1], (*(&kernelContext_4)->g_hi_0)[_S42]);
            (*(&kernelContext_4)->g_a_0)[t_1] = float4(max((*(&kernelContext_4)->g_a_0)[t_1].x, (*(&kernelContext_4)->g_a_0)[_S42].x), min((*(&kernelContext_4)->g_a_0)[t_1].yzw, (*(&kernelContext_4)->g_a_0)[_S42].yzw));
            (*(&kernelContext_4)->g_cap_0)[t_1] = min((*(&kernelContext_4)->g_cap_0)[t_1], (*(&kernelContext_4)->g_cap_0)[_S42]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    if(t_1 != 0U)
    {
        return;
    }
    thread Part_0 p_2;
    (&p_2)->flags_0 = 0U;
    if(n_0 == 0U)
    {
        thread Part_0 _S43 = p_2;
        store_part_0(i_2, &_S43, &kernelContext_4);
        return;
    }
    float3 lo_2 = (*(&kernelContext_4)->g_lo_0)[int(0)];
    float3 hi_2 = (*(&kernelContext_4)->g_hi_0)[int(0)];
    float4 _S44 = (*(&kernelContext_4)->g_a_0)[int(0)];
    a_2 = (*(&kernelContext_4)->g_a_0)[int(0)];
    bool anchored_0 = (((uint4(_S23->info_0) ).x) & 1U) != 0U;
    float _S45 = max(hi_2.x - lo_2.x, max(hi_2.y - lo_2.y, hi_2.z - lo_2.z));
    float3 sweep_0 = abs(_S24.v_2) * float3((&kernelContext_4)->params_0->dt_0)  + float3((length(_S24.w_1) * _S44.x * (&kernelContext_4)->params_0->dt_0)) ;
    if(_S19 > _S38)
    {
        c_0 = 4U;
    }
    else
    {
        c_0 = 0U;
    }
    uint _S46 = 1U | c_0;
    if(anchored_0)
    {
        c_0 = 16U;
    }
    else
    {
        c_0 = 0U;
    }
    (&p_2)->flags_0 = _S46 | c_0;
    (&p_2)->lo_0 = lo_2 - sweep_0;
    (&p_2)->hi_0 = hi_2 + sweep_0;
    (&p_2)->box_lo_0 = lo_2;
    (&p_2)->box_hi_0 = hi_2;
    (&p_2)->center_1 = _S24.com_1;
    (&p_2)->v_3 = _S24.v_2;
    (&p_2)->w_2 = _S24.w_1;
    (&p_2)->reach_0 = a_2.x;
    if(anchored_0)
    {
        cap_0 = inf_0;
    }
    else
    {
        cap_0 = (float4(_S23->com_0) ).w;
    }
    (&p_2)->mass_0 = cap_0;
    (&p_2)->modulus_0 = a_2.y;
    (&p_2)->radius_0 = a_2.w;
    (&p_2)->round_trip_0 = 2.0f * _S45 / a_2.z;
    (&p_2)->capacity_0 = (*(&kernelContext_4)->g_cap_0)[int(0)];
    thread Part_0 _S47 = p_2;
    store_part_0(i_2, &_S47, &kernelContext_4);
    return;
}

[[kernel]] void lr_impactors(uint3 id_0 [[thread_position_in_grid]], uint device* counters_2 [[buffer(10)]], RouteParams_0 constant* params_2 [[buffer(0)]], uint device* route_2 [[buffer(11)]], Island_natural_0 device* islands_2 [[buffer(1)]], packed_float4 device* ctab_2 [[buffer(4)]], packed_float4 device* bodies_2 [[buffer(5)]], ChunkStatic_natural_0 device* chunks_2 [[buffer(2)]], packed_float4 device* state_2 [[buffer(3)]], uint device* rtab_2 [[buffer(8)]], BondStatic_natural_0 device* bond_static_2 [[buffer(6)]], MaterialTable_0 constant* materials_2 [[buffer(7)]], packed_float4 device* parts_2 [[buffer(12)]], packed_float4 device* imps_2 [[buffer(9)]], atomic<uint> device* actl_2 [[buffer(16)]], packed_uint2 device* sorted_2 [[buffer(13)]], atomic<uint> device* lab_2 [[buffer(15)]], uint device* seed_2 [[buffer(17)]], packed_uint4 device* pass_2 [[buffer(19)]], packed_uint2 device* edges_2 [[buffer(14)]], packed_uint4 device* isl_2 [[buffer(18)]], uint device* ctl_2 [[buffer(20)]])
{
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->counters_0 = counters_2;
    (&kernelContext_5)->params_0 = params_2;
    (&kernelContext_5)->route_0 = route_2;
    (&kernelContext_5)->islands_0 = islands_2;
    (&kernelContext_5)->ctab_0 = ctab_2;
    (&kernelContext_5)->bodies_0 = bodies_2;
    (&kernelContext_5)->chunks_0 = chunks_2;
    (&kernelContext_5)->state_0 = state_2;
    (&kernelContext_5)->rtab_0 = rtab_2;
    (&kernelContext_5)->bond_static_0 = bond_static_2;
    (&kernelContext_5)->materials_0 = materials_2;
    (&kernelContext_5)->parts_0 = parts_2;
    (&kernelContext_5)->imps_0 = imps_2;
    (&kernelContext_5)->actl_0 = actl_2;
    (&kernelContext_5)->sorted_0 = sorted_2;
    (&kernelContext_5)->lab_0 = lab_2;
    (&kernelContext_5)->seed_0 = seed_2;
    (&kernelContext_5)->pass_0 = pass_2;
    (&kernelContext_5)->edges_0 = edges_2;
    (&kernelContext_5)->isl_0 = isl_2;
    (&kernelContext_5)->ctl_0 = ctl_2;
    threadgroup array<float3, int(64)> g_lo_2;
    (&kernelContext_5)->g_lo_0 = &g_lo_2;
    threadgroup array<float3, int(64)> g_hi_2;
    (&kernelContext_5)->g_hi_0 = &g_hi_2;
    threadgroup array<float4, int(64)> g_a_2;
    (&kernelContext_5)->g_a_0 = &g_a_2;
    threadgroup array<float, int(64)> g_cap_2;
    (&kernelContext_5)->g_cap_0 = &g_cap_2;
    uint k_1 = id_0.x;
    bool _S48;
    if(k_1 >= (min(params_2->impactor_count_0, 32U)))
    {
        _S48 = true;
    }
    else
    {
        bool _S49 = gate_0(&kernelContext_5);
        _S48 = !_S49;
    }
    if(_S48)
    {
        return;
    }
    uint _S50 = 4U * k_1;
    float4 _S51 = float4(*((&kernelContext_5)->imps_0+_S50)) ;
    float4 _S52 = float4(*((&kernelContext_5)->imps_0+(_S50 + 1U))) ;
    float r_3 = _S51.y;
    uint _S53 = 7U * (as_type<uint>((_S51.x)));
    float3 c_2 = to_zup_0((float4(*((&kernelContext_5)->bodies_0+_S53)) ).xyz);
    float3 v_4 = to_zup_0((float4(*((&kernelContext_5)->bodies_0+(_S53 + 3U))) ).xyz);
    float3 w_3 = to_zup_0((float4(*((&kernelContext_5)->bodies_0+(_S53 + 4U))) ).xyz);
    float3 sweep_1 = abs(v_4) * float3((&kernelContext_5)->params_0->dt_0)  + float3((length(w_3) * r_3 * (&kernelContext_5)->params_0->dt_0)) ;
    thread Part_0 p_3;
    (&p_3)->flags_0 = 3U;
    float3 _S54 = float3(r_3) ;
    float3 _S55 = c_2 - _S54;
    (&p_3)->lo_0 = _S55 - sweep_1;
    float3 _S56 = c_2 + _S54;
    (&p_3)->hi_0 = _S56 + sweep_1;
    (&p_3)->box_lo_0 = _S55;
    (&p_3)->box_hi_0 = _S56;
    (&p_3)->center_1 = c_2;
    (&p_3)->v_3 = v_4;
    (&p_3)->w_2 = w_3;
    (&p_3)->reach_0 = r_3;
    (&p_3)->mass_0 = _S51.w;
    (&p_3)->modulus_0 = _S52.w;
    (&p_3)->radius_0 = _S51.z;
    (&p_3)->round_trip_0 = 0.0f;
    (&p_3)->capacity_0 = 0.0f;
    uint _S57 = live_islands_0(&kernelContext_5);
    uint _S58 = _S57 + k_1;
    thread Part_0 _S59 = p_3;
    store_part_0(_S58, &_S59, &kernelContext_5);
    return;
}

Part_0 load_part_0(uint i_3, KernelContext_0 thread* kernelContext_6)
{
    uint _S60 = 8U * i_3;
    float4 _S61 = float4(*(kernelContext_6->parts_0+_S60)) ;
    float4 _S62 = float4(*(kernelContext_6->parts_0+(_S60 + 1U))) ;
    float4 _S63 = float4(*(kernelContext_6->parts_0+(_S60 + 2U))) ;
    float4 _S64 = float4(*(kernelContext_6->parts_0+(_S60 + 3U))) ;
    float4 _S65 = float4(*(kernelContext_6->parts_0+(_S60 + 4U))) ;
    float4 _S66 = float4(*(kernelContext_6->parts_0+(_S60 + 5U))) ;
    float4 _S67 = float4(*(kernelContext_6->parts_0+(_S60 + 6U))) ;
    thread Part_0 p_4;
    (&p_4)->lo_0 = _S61.xyz;
    (&p_4)->flags_0 = (as_type<uint>((_S61.w)));
    (&p_4)->hi_0 = _S62.xyz;
    (&p_4)->mass_0 = _S62.w;
    (&p_4)->box_lo_0 = _S63.xyz;
    (&p_4)->reach_0 = _S63.w;
    (&p_4)->box_hi_0 = _S64.xyz;
    (&p_4)->modulus_0 = _S64.w;
    (&p_4)->center_1 = _S65.xyz;
    (&p_4)->radius_0 = _S65.w;
    (&p_4)->v_3 = _S66.xyz;
    (&p_4)->round_trip_0 = _S66.w;
    (&p_4)->w_2 = _S67.xyz;
    (&p_4)->capacity_0 = _S67.w;
    return p_4;
}

bool overlaps_0(const Part_0 thread* a_3, const Part_0 thread* b_3)
{
    bool _S68;
    if(all((a_3->lo_0) <= (b_3->hi_0)))
    {
        _S68 = all((b_3->lo_0) <= (a_3->hi_0));
    }
    else
    {
        _S68 = false;
    }
    return _S68;
}

float closing_speed_0(const Part_0 thread* a_4, const Part_0 thread* b_4)
{
    float spin_0 = length(a_4->w_2) * a_4->reach_0 + length(b_4->w_2) * b_4->reach_0;
    float3 rel_0 = a_4->v_3 - b_4->v_3;
    float3 n0_0 = clamp(a_4->center_1, b_4->box_lo_0, b_4->box_hi_0) - a_4->center_1;
    float3 n1_0 = clamp(b_4->center_1, a_4->box_lo_0, a_4->box_hi_0) - b_4->center_1;
    float l0_0 = length(n0_0);
    float l1_0 = length(n1_0);
    bool _S69;
    if(!(l0_0 > 0.0f))
    {
        _S69 = true;
    }
    else
    {
        _S69 = !(l1_0 > 0.0f);
    }
    if(_S69)
    {
        return length(rel_0) + spin_0;
    }
    return max(dot(rel_0, n0_0 / float3(l0_0) ), dot(rel_0, - n1_0 / float3(l1_0) )) + spin_0;
}

[[kernel]] void lr_debris(uint3 id_1 [[thread_position_in_grid]], uint device* counters_3 [[buffer(10)]], RouteParams_0 constant* params_3 [[buffer(0)]], uint device* route_3 [[buffer(11)]], Island_natural_0 device* islands_3 [[buffer(1)]], packed_float4 device* ctab_3 [[buffer(4)]], packed_float4 device* bodies_3 [[buffer(5)]], ChunkStatic_natural_0 device* chunks_3 [[buffer(2)]], packed_float4 device* state_3 [[buffer(3)]], uint device* rtab_3 [[buffer(8)]], BondStatic_natural_0 device* bond_static_3 [[buffer(6)]], MaterialTable_0 constant* materials_3 [[buffer(7)]], packed_float4 device* parts_3 [[buffer(12)]], packed_float4 device* imps_3 [[buffer(9)]], atomic<uint> device* actl_3 [[buffer(16)]], packed_uint2 device* sorted_3 [[buffer(13)]], atomic<uint> device* lab_3 [[buffer(15)]], uint device* seed_3 [[buffer(17)]], packed_uint4 device* pass_3 [[buffer(19)]], packed_uint2 device* edges_3 [[buffer(14)]], packed_uint4 device* isl_3 [[buffer(18)]], uint device* ctl_3 [[buffer(20)]])
{
    thread KernelContext_0 kernelContext_7;
    (&kernelContext_7)->counters_0 = counters_3;
    (&kernelContext_7)->params_0 = params_3;
    (&kernelContext_7)->route_0 = route_3;
    (&kernelContext_7)->islands_0 = islands_3;
    (&kernelContext_7)->ctab_0 = ctab_3;
    (&kernelContext_7)->bodies_0 = bodies_3;
    (&kernelContext_7)->chunks_0 = chunks_3;
    (&kernelContext_7)->state_0 = state_3;
    (&kernelContext_7)->rtab_0 = rtab_3;
    (&kernelContext_7)->bond_static_0 = bond_static_3;
    (&kernelContext_7)->materials_0 = materials_3;
    (&kernelContext_7)->parts_0 = parts_3;
    (&kernelContext_7)->imps_0 = imps_3;
    (&kernelContext_7)->actl_0 = actl_3;
    (&kernelContext_7)->sorted_0 = sorted_3;
    (&kernelContext_7)->lab_0 = lab_3;
    (&kernelContext_7)->seed_0 = seed_3;
    (&kernelContext_7)->pass_0 = pass_3;
    (&kernelContext_7)->edges_0 = edges_3;
    (&kernelContext_7)->isl_0 = isl_3;
    (&kernelContext_7)->ctl_0 = ctl_3;
    threadgroup array<float3, int(64)> g_lo_3;
    (&kernelContext_7)->g_lo_0 = &g_lo_3;
    threadgroup array<float3, int(64)> g_hi_3;
    (&kernelContext_7)->g_hi_0 = &g_hi_3;
    threadgroup array<float4, int(64)> g_a_3;
    (&kernelContext_7)->g_a_0 = &g_a_3;
    threadgroup array<float, int(64)> g_cap_3;
    (&kernelContext_7)->g_cap_0 = &g_cap_3;
    uint c_3 = id_1.x;
    bool _S70;
    if((params_3->debris_0) == 0U)
    {
        _S70 = true;
    }
    else
    {
        uint _S71 = live_islands_0(&kernelContext_7);
        _S70 = c_3 >= _S71;
    }
    if(_S70)
    {
        _S70 = true;
    }
    else
    {
        bool _S72 = gate_0(&kernelContext_7);
        _S70 = !_S72;
    }
    if(_S70)
    {
        return;
    }
    Part_0 _S73 = load_part_0(c_3, &kernelContext_7);
    if(((_S73.flags_0) & 1U) == 0U)
    {
        _S70 = true;
    }
    else
    {
        _S70 = ((_S73.flags_0) & 4U) == 0U;
    }
    if(_S70)
    {
        return;
    }
    float3 _S74 = float3(0.0f) ;
    uint _S75 = live_islands_0(&kernelContext_7);
    float3 g0_0 = _S74;
    float3 g1_0 = _S74;
    uint k_2 = 0U;
    for(;;)
    {
        if(k_2 < (min((&kernelContext_7)->params_0->impactor_count_0, 32U)))
        {
        }
        else
        {
            break;
        }
        Part_0 _S76 = load_part_0(_S75 + k_2, &kernelContext_7);
        thread Part_0 _S77 = _S76;
        thread Part_0 _S78 = _S73;
        bool _S79 = overlaps_0(&_S77, &_S78);
        if(!_S79)
        {
            _S70 = true;
        }
        else
        {
            thread Part_0 _S80 = _S76;
            thread Part_0 _S81 = _S73;
            float _S82 = closing_speed_0(&_S80, &_S81);
            _S70 = !(_S82 > ((&kernelContext_7)->params_0->speed_0));
        }
        if(_S70)
        {
            k_2 = k_2 + 1U;
            continue;
        }
        float3 d_0 = (_S76.v_3 - _S73.v_3) * float3((2.0f * (&kernelContext_7)->params_0->dt_0)) ;
        float3 _S83 = max(g1_0, d_0);
        g0_0 = min(g0_0, d_0);
        g1_0 = _S83;
        k_2 = k_2 + 1U;
    }
    if(any(g0_0 != _S74))
    {
        _S70 = true;
    }
    else
    {
        _S70 = any(g1_0 != _S74);
    }
    if(_S70)
    {
        uint _S84 = 8U * c_3;
        *((&kernelContext_7)->parts_0+_S84) = packed_float4(float4(_S73.lo_0 + g0_0, (as_type<float>(((_S73.flags_0) | 8U))))) ;
        *((&kernelContext_7)->parts_0+(_S84 + 1U)) = packed_float4(float4(_S73.hi_0 + g1_0, _S73.mass_0)) ;
    }
    return;
}

uint part_count_0(KernelContext_0 thread* kernelContext_8)
{
    uint _S85 = live_islands_0(kernelContext_8);
    return _S85 + min(kernelContext_8->params_0->impactor_count_0, 32U);
}

uint sortable_0(float x_1)
{
    uint u_0 = (as_type<uint>((x_1)));
    uint _S86;
    if((u_0 & 2147483648U) != 0U)
    {
        _S86 = ~u_0;
    }
    else
    {
        _S86 = u_0 | 2147483648U;
    }
    return _S86;
}

[[kernel]] void lr_keys(uint3 id_2 [[thread_position_in_grid]], uint device* counters_4 [[buffer(10)]], RouteParams_0 constant* params_4 [[buffer(0)]], uint device* route_4 [[buffer(11)]], Island_natural_0 device* islands_4 [[buffer(1)]], packed_float4 device* ctab_4 [[buffer(4)]], packed_float4 device* bodies_4 [[buffer(5)]], ChunkStatic_natural_0 device* chunks_4 [[buffer(2)]], packed_float4 device* state_4 [[buffer(3)]], uint device* rtab_4 [[buffer(8)]], BondStatic_natural_0 device* bond_static_4 [[buffer(6)]], MaterialTable_0 constant* materials_4 [[buffer(7)]], packed_float4 device* parts_4 [[buffer(12)]], packed_float4 device* imps_4 [[buffer(9)]], atomic<uint> device* actl_4 [[buffer(16)]], packed_uint2 device* sorted_4 [[buffer(13)]], atomic<uint> device* lab_4 [[buffer(15)]], uint device* seed_4 [[buffer(17)]], packed_uint4 device* pass_4 [[buffer(19)]], packed_uint2 device* edges_4 [[buffer(14)]], packed_uint4 device* isl_4 [[buffer(18)]], uint device* ctl_4 [[buffer(20)]])
{
    thread KernelContext_0 kernelContext_9;
    (&kernelContext_9)->counters_0 = counters_4;
    (&kernelContext_9)->params_0 = params_4;
    (&kernelContext_9)->route_0 = route_4;
    (&kernelContext_9)->islands_0 = islands_4;
    (&kernelContext_9)->ctab_0 = ctab_4;
    (&kernelContext_9)->bodies_0 = bodies_4;
    (&kernelContext_9)->chunks_0 = chunks_4;
    (&kernelContext_9)->state_0 = state_4;
    (&kernelContext_9)->rtab_0 = rtab_4;
    (&kernelContext_9)->bond_static_0 = bond_static_4;
    (&kernelContext_9)->materials_0 = materials_4;
    (&kernelContext_9)->parts_0 = parts_4;
    (&kernelContext_9)->imps_0 = imps_4;
    (&kernelContext_9)->actl_0 = actl_4;
    (&kernelContext_9)->sorted_0 = sorted_4;
    (&kernelContext_9)->lab_0 = lab_4;
    (&kernelContext_9)->seed_0 = seed_4;
    (&kernelContext_9)->pass_0 = pass_4;
    (&kernelContext_9)->edges_0 = edges_4;
    (&kernelContext_9)->isl_0 = isl_4;
    (&kernelContext_9)->ctl_0 = ctl_4;
    threadgroup array<float3, int(64)> g_lo_4;
    (&kernelContext_9)->g_lo_0 = &g_lo_4;
    threadgroup array<float3, int(64)> g_hi_4;
    (&kernelContext_9)->g_hi_0 = &g_hi_4;
    threadgroup array<float4, int(64)> g_a_4;
    (&kernelContext_9)->g_a_0 = &g_a_4;
    threadgroup array<float, int(64)> g_cap_4;
    (&kernelContext_9)->g_cap_0 = &g_cap_4;
    uint s_1 = id_2.x;
    if(s_1 >= (params_4->n_pad_0))
    {
        return;
    }
    if(s_1 < 56U)
    {
        atomic_store_explicit((&kernelContext_9)->actl_0+s_1, 0U, memory_order_relaxed);
    }
    uint _S87 = part_count_0(&kernelContext_9);
    bool _S88 = s_1 < _S87;
    bool used_0;
    if(_S88)
    {
        bool _S89 = gate_0(&kernelContext_9);
        used_0 = _S89;
    }
    else
    {
        used_0 = false;
    }
    if(used_0)
    {
        used_0 = ((as_type<uint>(((float4(*((&kernelContext_9)->parts_0+8U * s_1)) ).w))) & 1U) != 0U;
    }
    else
    {
        used_0 = false;
    }
    packed_uint2 device* _S90 = (&kernelContext_9)->sorted_0+s_1;
    uint _S91;
    if(used_0)
    {
        _S91 = sortable_0((float4(*((&kernelContext_9)->parts_0+8U * s_1)) ).x);
    }
    else
    {
        _S91 = 4294967295U;
    }
    uint _S92;
    if(used_0)
    {
        _S92 = s_1;
    }
    else
    {
        _S92 = 4294967295U;
    }
    *_S90 = packed_uint2(uint2(_S91, _S92)) ;
    if(_S88)
    {
        atomic_store_explicit((&kernelContext_9)->lab_0+s_1, s_1, memory_order_relaxed);
        *((&kernelContext_9)->seed_0+s_1) = 0U;
    }
    return;
}

[[kernel]] void lr_bitonic(uint3 id_3 [[thread_position_in_grid]], uint device* counters_5 [[buffer(10)]], RouteParams_0 constant* params_5 [[buffer(0)]], uint device* route_5 [[buffer(11)]], Island_natural_0 device* islands_5 [[buffer(1)]], packed_float4 device* ctab_5 [[buffer(4)]], packed_float4 device* bodies_5 [[buffer(5)]], ChunkStatic_natural_0 device* chunks_5 [[buffer(2)]], packed_float4 device* state_5 [[buffer(3)]], uint device* rtab_5 [[buffer(8)]], BondStatic_natural_0 device* bond_static_5 [[buffer(6)]], MaterialTable_0 constant* materials_5 [[buffer(7)]], packed_float4 device* parts_5 [[buffer(12)]], packed_float4 device* imps_5 [[buffer(9)]], atomic<uint> device* actl_5 [[buffer(16)]], packed_uint2 device* sorted_5 [[buffer(13)]], atomic<uint> device* lab_5 [[buffer(15)]], uint device* seed_5 [[buffer(17)]], packed_uint4 device* pass_5 [[buffer(19)]], packed_uint2 device* edges_5 [[buffer(14)]], packed_uint4 device* isl_5 [[buffer(18)]], uint device* ctl_5 [[buffer(20)]])
{
    thread KernelContext_0 kernelContext_10;
    (&kernelContext_10)->counters_0 = counters_5;
    (&kernelContext_10)->params_0 = params_5;
    (&kernelContext_10)->route_0 = route_5;
    (&kernelContext_10)->islands_0 = islands_5;
    (&kernelContext_10)->ctab_0 = ctab_5;
    (&kernelContext_10)->bodies_0 = bodies_5;
    (&kernelContext_10)->chunks_0 = chunks_5;
    (&kernelContext_10)->state_0 = state_5;
    (&kernelContext_10)->rtab_0 = rtab_5;
    (&kernelContext_10)->bond_static_0 = bond_static_5;
    (&kernelContext_10)->materials_0 = materials_5;
    (&kernelContext_10)->parts_0 = parts_5;
    (&kernelContext_10)->imps_0 = imps_5;
    (&kernelContext_10)->actl_0 = actl_5;
    (&kernelContext_10)->sorted_0 = sorted_5;
    (&kernelContext_10)->lab_0 = lab_5;
    (&kernelContext_10)->seed_0 = seed_5;
    (&kernelContext_10)->pass_0 = pass_5;
    (&kernelContext_10)->edges_0 = edges_5;
    (&kernelContext_10)->isl_0 = isl_5;
    (&kernelContext_10)->ctl_0 = ctl_5;
    threadgroup array<float3, int(64)> g_lo_5;
    (&kernelContext_10)->g_lo_0 = &g_lo_5;
    threadgroup array<float3, int(64)> g_hi_5;
    (&kernelContext_10)->g_hi_0 = &g_hi_5;
    threadgroup array<float4, int(64)> g_a_5;
    (&kernelContext_10)->g_a_0 = &g_a_5;
    threadgroup array<float, int(64)> g_cap_5;
    (&kernelContext_10)->g_cap_0 = &g_cap_5;
    uint4 _S93 = uint4(*(pass_5+int(0))) ;
    uint k_3 = _S93.x;
    uint j_0 = _S93.y;
    uint t_2 = id_3.x;
    if(t_2 >= (params_5->n_pad_0 / 2U))
    {
        return;
    }
    uint _S94 = j_0 - 1U;
    uint lo_3 = ((t_2 & (~_S94)) << 1U) | (t_2 & _S94);
    uint hi_3 = lo_3 | j_0;
    uint2 _S95 = uint2(*((&kernelContext_10)->sorted_0+lo_3)) ;
    uint2 _S96 = uint2(*((&kernelContext_10)->sorted_0+hi_3)) ;
    bool swap_0;
    if((lo_3 & k_3) == 0U)
    {
        uint _S97 = _S95.x;
        uint _S98 = _S96.x;
        if(_S97 > _S98)
        {
            swap_0 = true;
        }
        else
        {
            if(_S97 == _S98)
            {
                swap_0 = (_S95.y) > (_S96.y);
            }
            else
            {
                swap_0 = false;
            }
        }
    }
    else
    {
        uint _S99 = _S95.x;
        uint _S100 = _S96.x;
        if(_S99 < _S100)
        {
            swap_0 = true;
        }
        else
        {
            if(_S99 == _S100)
            {
                swap_0 = (_S95.y) < (_S96.y);
            }
            else
            {
                swap_0 = false;
            }
        }
    }
    if(swap_0)
    {
        *((&kernelContext_10)->sorted_0+lo_3) = packed_uint2(_S96) ;
        *((&kernelContext_10)->sorted_0+hi_3) = packed_uint2(_S95) ;
    }
    return;
}

float hertz_duration_0(float m_1, float r_4, float e_0, float v_5)
{
    return 2.86999988555908203f * pow(m_1 * m_1 / (r_4 * e_0 * e_0 * max(abs(v_5), 0.00100000004749745f)), 0.20000000298023224f);
}

bool rigid_fails_0(const Part_0 thread* a_5, const Part_0 thread* b_5, float speed_1)
{
    uint _S101 = b_5->flags_0;
    if(((b_5->flags_0) & 4U) == 0U)
    {
        return false;
    }
    bool fa_0 = ((a_5->flags_0) & 16U) == 0U;
    bool fb_0 = (_S101 & 16U) == 0U;
    bool _S102;
    if(fa_0)
    {
        _S102 = fb_0;
    }
    else
    {
        _S102 = false;
    }
    float m_2;
    if(_S102)
    {
        m_2 = a_5->mass_0 * b_5->mass_0 / (a_5->mass_0 + b_5->mass_0);
    }
    else
    {
        if(fa_0)
        {
            m_2 = a_5->mass_0;
        }
        else
        {
            float _S103 = 0.0f;
            if(fb_0)
            {
                _S103 = b_5->mass_0;
            }
            else
            {
                return false;
            }
            m_2 = _S103;
        }
    }
    if(!(speed_1 > 0.0f))
    {
        return false;
    }
    float duration_0 = hertz_duration_0(m_2, min(a_5->radius_0, b_5->radius_0), 1.0f / (1.0f / a_5->modulus_0 + 1.0f / b_5->modulus_0), speed_1);
    float peak_0 = 3.14159274101257324f * (2.0f * m_2 * speed_1) / (2.0f * duration_0);
    if(duration_0 < (b_5->round_trip_0))
    {
        _S102 = true;
    }
    else
    {
        _S102 = peak_0 > (b_5->capacity_0);
    }
    return _S102;
}

bool ground_owned_0(const Part_0 thread* p_5, KernelContext_0 thread* kernelContext_11)
{
    float speed_2 = p_5->v_3.z * -1.0f + length(p_5->w_2) * p_5->reach_0;
    thread Part_0 g_0;
    (&g_0)->flags_0 = 17U;
    float _S104 = float_inf_0();
    (&g_0)->mass_0 = _S104;
    (&g_0)->modulus_0 = kernelContext_11->params_0->ground_modulus_0;
    (&g_0)->radius_0 = _S104;
    thread Part_0 _S105 = g_0;
    bool _S106 = rigid_fails_0(&_S105, p_5, speed_2);
    return _S106;
}

[[kernel]] void lr_sweep(uint3 id_4 [[thread_position_in_grid]], uint device* counters_6 [[buffer(10)]], RouteParams_0 constant* params_6 [[buffer(0)]], uint device* route_6 [[buffer(11)]], Island_natural_0 device* islands_6 [[buffer(1)]], packed_float4 device* ctab_6 [[buffer(4)]], packed_float4 device* bodies_6 [[buffer(5)]], ChunkStatic_natural_0 device* chunks_6 [[buffer(2)]], packed_float4 device* state_6 [[buffer(3)]], uint device* rtab_6 [[buffer(8)]], BondStatic_natural_0 device* bond_static_6 [[buffer(6)]], MaterialTable_0 constant* materials_6 [[buffer(7)]], packed_float4 device* parts_6 [[buffer(12)]], packed_float4 device* imps_6 [[buffer(9)]], atomic<uint> device* actl_6 [[buffer(16)]], packed_uint2 device* sorted_6 [[buffer(13)]], atomic<uint> device* lab_6 [[buffer(15)]], uint device* seed_6 [[buffer(17)]], packed_uint4 device* pass_6 [[buffer(19)]], packed_uint2 device* edges_6 [[buffer(14)]], packed_uint4 device* isl_6 [[buffer(18)]], uint device* ctl_6 [[buffer(20)]])
{
    thread KernelContext_0 kernelContext_12;
    (&kernelContext_12)->counters_0 = counters_6;
    (&kernelContext_12)->params_0 = params_6;
    (&kernelContext_12)->route_0 = route_6;
    (&kernelContext_12)->islands_0 = islands_6;
    (&kernelContext_12)->ctab_0 = ctab_6;
    (&kernelContext_12)->bodies_0 = bodies_6;
    (&kernelContext_12)->chunks_0 = chunks_6;
    (&kernelContext_12)->state_0 = state_6;
    (&kernelContext_12)->rtab_0 = rtab_6;
    (&kernelContext_12)->bond_static_0 = bond_static_6;
    (&kernelContext_12)->materials_0 = materials_6;
    (&kernelContext_12)->parts_0 = parts_6;
    (&kernelContext_12)->imps_0 = imps_6;
    (&kernelContext_12)->actl_0 = actl_6;
    (&kernelContext_12)->sorted_0 = sorted_6;
    (&kernelContext_12)->lab_0 = lab_6;
    (&kernelContext_12)->seed_0 = seed_6;
    (&kernelContext_12)->pass_0 = pass_6;
    (&kernelContext_12)->edges_0 = edges_6;
    (&kernelContext_12)->isl_0 = isl_6;
    (&kernelContext_12)->ctl_0 = ctl_6;
    threadgroup array<float3, int(64)> g_lo_6;
    (&kernelContext_12)->g_lo_0 = &g_lo_6;
    threadgroup array<float3, int(64)> g_hi_6;
    (&kernelContext_12)->g_hi_0 = &g_hi_6;
    threadgroup array<float4, int(64)> g_a_6;
    (&kernelContext_12)->g_a_0 = &g_a_6;
    threadgroup array<float, int(64)> g_cap_6;
    (&kernelContext_12)->g_cap_0 = &g_cap_6;
    uint s_2 = id_4.x;
    bool fast_0;
    if(s_2 >= (params_6->n_pad_0))
    {
        fast_0 = true;
    }
    else
    {
        bool _S107 = gate_0(&kernelContext_12);
        fast_0 = !_S107;
    }
    if(fast_0)
    {
        return;
    }
    uint ia_0 = (uint2(*((&kernelContext_12)->sorted_0+s_2)) ).y;
    if(ia_0 == 4294967295U)
    {
        return;
    }
    Part_0 _S108 = load_part_0(ia_0, &kernelContext_12);
    bool tiered_1 = ((&kernelContext_12)->params_0->tiered_0) != 0U;
    if(((&kernelContext_12)->params_0->ground_0) != 0U)
    {
        fast_0 = (_S108.lo_0.z) <= ((&kernelContext_12)->params_0->ground_height_0);
    }
    else
    {
        fast_0 = false;
    }
    if(fast_0)
    {
        if(!tiered_1)
        {
            thread Part_0 _S109 = _S108;
            bool _S110 = ground_owned_0(&_S109, &kernelContext_12);
            fast_0 = _S110;
        }
        else
        {
            fast_0 = false;
        }
        if(fast_0)
        {
            *((&kernelContext_12)->seed_0+ia_0) = 1U;
        }
    }
    uint q_4 = s_2 + 1U;
    for(;;)
    {
        if(q_4 < (params_6->n_pad_0))
        {
        }
        else
        {
            break;
        }
        uint ib_0 = (uint2(*((&kernelContext_12)->sorted_0+q_4)) ).y;
        if(ib_0 == 4294967295U)
        {
            break;
        }
        Part_0 _S111 = load_part_0(ib_0, &kernelContext_12);
        if((_S111.lo_0.x) > (_S108.hi_0.x))
        {
            break;
        }
        thread Part_0 _S112 = _S108;
        thread Part_0 _S113 = _S111;
        bool _S114 = overlaps_0(&_S112, &_S113);
        if(!_S114)
        {
            q_4 = q_4 + 1U;
            continue;
        }
        uint _S115 = (_S108.flags_0) | (_S111.flags_0);
        bool impactor_0 = (_S115 & 2U) != 0U;
        thread Part_0 _S116 = _S108;
        thread Part_0 _S117 = _S111;
        float _S118 = closing_speed_0(&_S116, &_S117);
        thread Part_0 _S119 = _S111;
        thread Part_0 _S120 = _S108;
        float _S121 = closing_speed_0(&_S119, &_S120);
        bool _S122 = !tiered_1;
        if(_S122)
        {
            fast_0 = true;
        }
        else
        {
            fast_0 = (max(_S118, _S121)) > ((&kernelContext_12)->params_0->speed_0);
        }
        bool _S123;
        if(_S122)
        {
            _S123 = true;
        }
        else
        {
            _S123 = impactor_0;
        }
        bool _S124;
        if(_S123)
        {
            _S124 = fast_0;
        }
        else
        {
            _S124 = false;
        }
        bool owned_0;
        if(_S124)
        {
            thread Part_0 _S125 = _S108;
            thread Part_0 _S126 = _S111;
            bool _S127 = rigid_fails_0(&_S125, &_S126, _S118);
            if(_S127)
            {
                owned_0 = true;
            }
            else
            {
                thread Part_0 _S128 = _S111;
                thread Part_0 _S129 = _S108;
                bool _S130 = rigid_fails_0(&_S128, &_S129, _S121);
                owned_0 = _S130;
            }
        }
        else
        {
            owned_0 = false;
        }
        bool both_immovable_0;
        if(((_S108.flags_0) & 16U) != 0U)
        {
            both_immovable_0 = ((_S111.flags_0) & 16U) != 0U;
        }
        else
        {
            both_immovable_0 = false;
        }
        if(both_immovable_0)
        {
            q_4 = q_4 + 1U;
            continue;
        }
        bool _S131;
        if(tiered_1)
        {
            _S131 = ((&kernelContext_12)->params_0->fast_joins_0) != 0U;
        }
        else
        {
            _S131 = false;
        }
        bool _S132;
        if(_S131)
        {
            _S132 = !impactor_0;
        }
        else
        {
            _S132 = false;
        }
        bool _S133;
        if(_S132)
        {
            _S133 = !fast_0;
        }
        else
        {
            _S133 = false;
        }
        bool _S134;
        if(_S133)
        {
            _S134 = (_S115 & 8U) == 0U;
        }
        else
        {
            _S134 = false;
        }
        if(_S134)
        {
            q_4 = q_4 + 1U;
            continue;
        }
        uint e_1 = atomic_fetch_add_explicit((&kernelContext_12)->actl_0+0U, 1U, memory_order_relaxed);
        if(e_1 < ((&kernelContext_12)->params_0->edge_cap_0))
        {
            *((&kernelContext_12)->edges_0+e_1) = packed_uint2(uint2(ia_0, ib_0)) ;
        }
        else
        {
            atomic_store_explicit((&kernelContext_12)->actl_0+1U, 1U, memory_order_relaxed);
        }
        if(owned_0)
        {
            *((&kernelContext_12)->seed_0+ia_0) = 1U;
            *((&kernelContext_12)->seed_0+ib_0) = 1U;
        }
        q_4 = q_4 + 1U;
    }
    return;
}

uint edge_count_0(KernelContext_0 thread* kernelContext_13)
{
    uint _S135 = atomic_load_explicit(kernelContext_13->actl_0+0U, memory_order_relaxed);
    return min(_S135, kernelContext_13->params_0->edge_cap_0);
}

[[kernel]] void lr_hook(uint3 id_5 [[thread_position_in_grid]], uint device* counters_7 [[buffer(10)]], RouteParams_0 constant* params_7 [[buffer(0)]], uint device* route_7 [[buffer(11)]], Island_natural_0 device* islands_7 [[buffer(1)]], packed_float4 device* ctab_7 [[buffer(4)]], packed_float4 device* bodies_7 [[buffer(5)]], ChunkStatic_natural_0 device* chunks_7 [[buffer(2)]], packed_float4 device* state_7 [[buffer(3)]], uint device* rtab_7 [[buffer(8)]], BondStatic_natural_0 device* bond_static_7 [[buffer(6)]], MaterialTable_0 constant* materials_7 [[buffer(7)]], packed_float4 device* parts_7 [[buffer(12)]], packed_float4 device* imps_7 [[buffer(9)]], atomic<uint> device* actl_7 [[buffer(16)]], packed_uint2 device* sorted_7 [[buffer(13)]], atomic<uint> device* lab_7 [[buffer(15)]], uint device* seed_7 [[buffer(17)]], packed_uint4 device* pass_7 [[buffer(19)]], packed_uint2 device* edges_7 [[buffer(14)]], packed_uint4 device* isl_7 [[buffer(18)]], uint device* ctl_7 [[buffer(20)]])
{
    thread KernelContext_0 kernelContext_14;
    (&kernelContext_14)->counters_0 = counters_7;
    (&kernelContext_14)->params_0 = params_7;
    (&kernelContext_14)->route_0 = route_7;
    (&kernelContext_14)->islands_0 = islands_7;
    (&kernelContext_14)->ctab_0 = ctab_7;
    (&kernelContext_14)->bodies_0 = bodies_7;
    (&kernelContext_14)->chunks_0 = chunks_7;
    (&kernelContext_14)->state_0 = state_7;
    (&kernelContext_14)->rtab_0 = rtab_7;
    (&kernelContext_14)->bond_static_0 = bond_static_7;
    (&kernelContext_14)->materials_0 = materials_7;
    (&kernelContext_14)->parts_0 = parts_7;
    (&kernelContext_14)->imps_0 = imps_7;
    (&kernelContext_14)->actl_0 = actl_7;
    (&kernelContext_14)->sorted_0 = sorted_7;
    (&kernelContext_14)->lab_0 = lab_7;
    (&kernelContext_14)->seed_0 = seed_7;
    (&kernelContext_14)->pass_0 = pass_7;
    (&kernelContext_14)->edges_0 = edges_7;
    (&kernelContext_14)->isl_0 = isl_7;
    (&kernelContext_14)->ctl_0 = ctl_7;
    threadgroup array<float3, int(64)> g_lo_7;
    (&kernelContext_14)->g_lo_0 = &g_lo_7;
    threadgroup array<float3, int(64)> g_hi_7;
    (&kernelContext_14)->g_hi_0 = &g_hi_7;
    threadgroup array<float4, int(64)> g_a_7;
    (&kernelContext_14)->g_a_0 = &g_a_7;
    threadgroup array<float, int(64)> g_cap_7;
    (&kernelContext_14)->g_cap_0 = &g_cap_7;
    uint r_5 = (uint4(*(pass_7+int(0))) ).z;
    uint e_2 = id_5.x;
    bool _S136 = gate_0(&kernelContext_14);
    bool _S137;
    if(!_S136)
    {
        _S137 = true;
    }
    else
    {
        if(r_5 > 0U)
        {
            uint _S138 = atomic_load_explicit((&kernelContext_14)->actl_0+(8U + r_5 - 1U), memory_order_relaxed);
            _S137 = _S138 == 0U;
        }
        else
        {
            _S137 = false;
        }
    }
    if(_S137)
    {
        _S137 = true;
    }
    else
    {
        uint _S139 = edge_count_0(&kernelContext_14);
        _S137 = e_2 >= _S139;
    }
    if(_S137)
    {
        return;
    }
    uint2 _S140 = uint2(*((&kernelContext_14)->edges_0+e_2)) ;
    uint _S141 = _S140.x;
    uint la_0 = atomic_load_explicit((&kernelContext_14)->lab_0+_S141, memory_order_relaxed);
    uint _S142 = _S140.y;
    uint lb_0 = atomic_load_explicit((&kernelContext_14)->lab_0+_S142, memory_order_relaxed);
    if(la_0 == lb_0)
    {
        return;
    }
    uint _S143 = min(la_0, lb_0);
    uint _S144 = atomic_fetch_min_explicit((&kernelContext_14)->lab_0+_S141, _S143, memory_order_relaxed);
    uint _S145 = atomic_fetch_min_explicit((&kernelContext_14)->lab_0+_S142, _S143, memory_order_relaxed);
    uint _S146 = atomic_fetch_min_explicit((&kernelContext_14)->lab_0+max(la_0, lb_0), _S143, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_14)->actl_0+(8U + r_5), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void lr_jump(uint3 id_6 [[thread_position_in_grid]], uint device* counters_8 [[buffer(10)]], RouteParams_0 constant* params_8 [[buffer(0)]], uint device* route_8 [[buffer(11)]], Island_natural_0 device* islands_8 [[buffer(1)]], packed_float4 device* ctab_8 [[buffer(4)]], packed_float4 device* bodies_8 [[buffer(5)]], ChunkStatic_natural_0 device* chunks_8 [[buffer(2)]], packed_float4 device* state_8 [[buffer(3)]], uint device* rtab_8 [[buffer(8)]], BondStatic_natural_0 device* bond_static_8 [[buffer(6)]], MaterialTable_0 constant* materials_8 [[buffer(7)]], packed_float4 device* parts_8 [[buffer(12)]], packed_float4 device* imps_8 [[buffer(9)]], atomic<uint> device* actl_8 [[buffer(16)]], packed_uint2 device* sorted_8 [[buffer(13)]], atomic<uint> device* lab_8 [[buffer(15)]], uint device* seed_8 [[buffer(17)]], packed_uint4 device* pass_8 [[buffer(19)]], packed_uint2 device* edges_8 [[buffer(14)]], packed_uint4 device* isl_8 [[buffer(18)]], uint device* ctl_8 [[buffer(20)]])
{
    thread KernelContext_0 kernelContext_15;
    (&kernelContext_15)->counters_0 = counters_8;
    (&kernelContext_15)->params_0 = params_8;
    (&kernelContext_15)->route_0 = route_8;
    (&kernelContext_15)->islands_0 = islands_8;
    (&kernelContext_15)->ctab_0 = ctab_8;
    (&kernelContext_15)->bodies_0 = bodies_8;
    (&kernelContext_15)->chunks_0 = chunks_8;
    (&kernelContext_15)->state_0 = state_8;
    (&kernelContext_15)->rtab_0 = rtab_8;
    (&kernelContext_15)->bond_static_0 = bond_static_8;
    (&kernelContext_15)->materials_0 = materials_8;
    (&kernelContext_15)->parts_0 = parts_8;
    (&kernelContext_15)->imps_0 = imps_8;
    (&kernelContext_15)->actl_0 = actl_8;
    (&kernelContext_15)->sorted_0 = sorted_8;
    (&kernelContext_15)->lab_0 = lab_8;
    (&kernelContext_15)->seed_0 = seed_8;
    (&kernelContext_15)->pass_0 = pass_8;
    (&kernelContext_15)->edges_0 = edges_8;
    (&kernelContext_15)->isl_0 = isl_8;
    (&kernelContext_15)->ctl_0 = ctl_8;
    threadgroup array<float3, int(64)> g_lo_8;
    (&kernelContext_15)->g_lo_0 = &g_lo_8;
    threadgroup array<float3, int(64)> g_hi_8;
    (&kernelContext_15)->g_hi_0 = &g_hi_8;
    threadgroup array<float4, int(64)> g_a_8;
    (&kernelContext_15)->g_a_0 = &g_a_8;
    threadgroup array<float, int(64)> g_cap_8;
    (&kernelContext_15)->g_cap_0 = &g_cap_8;
    uint r_6 = (uint4(*(pass_8+int(0))) ).z;
    uint i_4 = id_6.x;
    bool _S147 = gate_0(&kernelContext_15);
    bool _S148;
    if(!_S147)
    {
        _S148 = true;
    }
    else
    {
        if(r_6 > 0U)
        {
            uint _S149 = atomic_load_explicit((&kernelContext_15)->actl_0+(8U + r_6 - 1U), memory_order_relaxed);
            _S148 = _S149 == 0U;
        }
        else
        {
            _S148 = false;
        }
    }
    if(_S148)
    {
        _S148 = true;
    }
    else
    {
        uint _S150 = part_count_0(&kernelContext_15);
        _S148 = i_4 >= _S150;
    }
    if(_S148)
    {
        return;
    }
    uint l_0 = atomic_load_explicit((&kernelContext_15)->lab_0+i_4, memory_order_relaxed);
    uint ll_0 = atomic_load_explicit((&kernelContext_15)->lab_0+l_0, memory_order_relaxed);
    if(ll_0 < l_0)
    {
        uint _S151 = atomic_fetch_min_explicit((&kernelContext_15)->lab_0+i_4, ll_0, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_15)->actl_0+(8U + r_6), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void lr_roots(uint3 id_7 [[thread_position_in_grid]], uint device* counters_9 [[buffer(10)]], RouteParams_0 constant* params_9 [[buffer(0)]], uint device* route_9 [[buffer(11)]], Island_natural_0 device* islands_9 [[buffer(1)]], packed_float4 device* ctab_9 [[buffer(4)]], packed_float4 device* bodies_9 [[buffer(5)]], ChunkStatic_natural_0 device* chunks_9 [[buffer(2)]], packed_float4 device* state_9 [[buffer(3)]], uint device* rtab_9 [[buffer(8)]], BondStatic_natural_0 device* bond_static_9 [[buffer(6)]], MaterialTable_0 constant* materials_9 [[buffer(7)]], packed_float4 device* parts_9 [[buffer(12)]], packed_float4 device* imps_9 [[buffer(9)]], atomic<uint> device* actl_9 [[buffer(16)]], packed_uint2 device* sorted_9 [[buffer(13)]], atomic<uint> device* lab_9 [[buffer(15)]], uint device* seed_9 [[buffer(17)]], packed_uint4 device* pass_9 [[buffer(19)]], packed_uint2 device* edges_9 [[buffer(14)]], packed_uint4 device* isl_9 [[buffer(18)]], uint device* ctl_9 [[buffer(20)]])
{
    thread KernelContext_0 kernelContext_16;
    (&kernelContext_16)->counters_0 = counters_9;
    (&kernelContext_16)->params_0 = params_9;
    (&kernelContext_16)->route_0 = route_9;
    (&kernelContext_16)->islands_0 = islands_9;
    (&kernelContext_16)->ctab_0 = ctab_9;
    (&kernelContext_16)->bodies_0 = bodies_9;
    (&kernelContext_16)->chunks_0 = chunks_9;
    (&kernelContext_16)->state_0 = state_9;
    (&kernelContext_16)->rtab_0 = rtab_9;
    (&kernelContext_16)->bond_static_0 = bond_static_9;
    (&kernelContext_16)->materials_0 = materials_9;
    (&kernelContext_16)->parts_0 = parts_9;
    (&kernelContext_16)->imps_0 = imps_9;
    (&kernelContext_16)->actl_0 = actl_9;
    (&kernelContext_16)->sorted_0 = sorted_9;
    (&kernelContext_16)->lab_0 = lab_9;
    (&kernelContext_16)->seed_0 = seed_9;
    (&kernelContext_16)->pass_0 = pass_9;
    (&kernelContext_16)->edges_0 = edges_9;
    (&kernelContext_16)->isl_0 = isl_9;
    (&kernelContext_16)->ctl_0 = ctl_9;
    threadgroup array<float3, int(64)> g_lo_9;
    (&kernelContext_16)->g_lo_0 = &g_lo_9;
    threadgroup array<float3, int(64)> g_hi_9;
    (&kernelContext_16)->g_hi_0 = &g_hi_9;
    threadgroup array<float4, int(64)> g_a_9;
    (&kernelContext_16)->g_a_0 = &g_a_9;
    threadgroup array<float, int(64)> g_cap_9;
    (&kernelContext_16)->g_cap_0 = &g_cap_9;
    uint i_5 = id_7.x;
    bool _S152 = gate_0(&kernelContext_16);
    bool _S153;
    if(!_S152)
    {
        _S153 = true;
    }
    else
    {
        uint _S154 = part_count_0(&kernelContext_16);
        _S153 = i_5 >= _S154;
    }
    if(_S153)
    {
        return;
    }
    if(((*((&kernelContext_16)->seed_0+i_5)) & 1U) != 0U)
    {
        uint _S155 = atomic_load_explicit((&kernelContext_16)->lab_0+i_5, memory_order_relaxed);
        uint device* _S156 = (&kernelContext_16)->seed_0+_S155;
        *_S156 = (*_S156) | 2U;
    }
    return;
}

[[kernel]] void lr_finish(uint3 id_8 [[thread_position_in_grid]], uint device* counters_10 [[buffer(10)]], RouteParams_0 constant* params_10 [[buffer(0)]], uint device* route_10 [[buffer(11)]], Island_natural_0 device* islands_10 [[buffer(1)]], packed_float4 device* ctab_10 [[buffer(4)]], packed_float4 device* bodies_10 [[buffer(5)]], ChunkStatic_natural_0 device* chunks_10 [[buffer(2)]], packed_float4 device* state_10 [[buffer(3)]], uint device* rtab_10 [[buffer(8)]], BondStatic_natural_0 device* bond_static_10 [[buffer(6)]], MaterialTable_0 constant* materials_10 [[buffer(7)]], packed_float4 device* parts_10 [[buffer(12)]], packed_float4 device* imps_10 [[buffer(9)]], atomic<uint> device* actl_10 [[buffer(16)]], packed_uint2 device* sorted_10 [[buffer(13)]], atomic<uint> device* lab_10 [[buffer(15)]], uint device* seed_10 [[buffer(17)]], packed_uint4 device* pass_10 [[buffer(19)]], packed_uint2 device* edges_10 [[buffer(14)]], packed_uint4 device* isl_10 [[buffer(18)]], uint device* ctl_10 [[buffer(20)]])
{
    thread KernelContext_0 kernelContext_17;
    (&kernelContext_17)->counters_0 = counters_10;
    (&kernelContext_17)->params_0 = params_10;
    (&kernelContext_17)->route_0 = route_10;
    (&kernelContext_17)->islands_0 = islands_10;
    (&kernelContext_17)->ctab_0 = ctab_10;
    (&kernelContext_17)->bodies_0 = bodies_10;
    (&kernelContext_17)->chunks_0 = chunks_10;
    (&kernelContext_17)->state_0 = state_10;
    (&kernelContext_17)->rtab_0 = rtab_10;
    (&kernelContext_17)->bond_static_0 = bond_static_10;
    (&kernelContext_17)->materials_0 = materials_10;
    (&kernelContext_17)->parts_0 = parts_10;
    (&kernelContext_17)->imps_0 = imps_10;
    (&kernelContext_17)->actl_0 = actl_10;
    (&kernelContext_17)->sorted_0 = sorted_10;
    (&kernelContext_17)->lab_0 = lab_10;
    (&kernelContext_17)->seed_0 = seed_10;
    (&kernelContext_17)->pass_0 = pass_10;
    (&kernelContext_17)->edges_0 = edges_10;
    (&kernelContext_17)->isl_0 = isl_10;
    (&kernelContext_17)->ctl_0 = ctl_10;
    threadgroup array<float3, int(64)> g_lo_10;
    (&kernelContext_17)->g_lo_0 = &g_lo_10;
    threadgroup array<float3, int(64)> g_hi_10;
    (&kernelContext_17)->g_hi_0 = &g_hi_10;
    threadgroup array<float4, int(64)> g_a_10;
    (&kernelContext_17)->g_a_0 = &g_a_10;
    threadgroup array<float, int(64)> g_cap_10;
    (&kernelContext_17)->g_cap_0 = &g_cap_10;
    uint i_6 = id_8.x;
    uint _S157 = live_islands_0(&kernelContext_17);
    uint _S158 = part_count_0(&kernelContext_17);
    if(i_6 >= _S158)
    {
        return;
    }
    bool _S159 = gate_0(&kernelContext_17);
    bool valid_0;
    if(_S159)
    {
        valid_0 = ((as_type<uint>(((float4(*((&kernelContext_17)->parts_0+8U * i_6)) ).w))) & 1U) != 0U;
    }
    else
    {
        valid_0 = false;
    }
    bool routed_0;
    if(valid_0)
    {
        uint _S160 = atomic_load_explicit((&kernelContext_17)->lab_0+i_6, memory_order_relaxed);
        routed_0 = ((*((&kernelContext_17)->seed_0+_S160)) & 2U) != 0U;
    }
    else
    {
        routed_0 = false;
    }
    uint _S161;
    if(i_6 < _S157)
    {
        uint _S162 = 5U * i_6;
        packed_uint4 device* _S163 = (&kernelContext_17)->isl_0+_S162;
        if(routed_0)
        {
            _S161 = 2147483648U;
        }
        else
        {
            _S161 = 0U;
        }
        *_S163 = packed_uint4(uint4(_S161, 4294967295U, 0U, 0U)) ;
        packed_uint4 _S164 = packed_uint4(uint4(int4(int(0)) )) ;
        *((&kernelContext_17)->isl_0+(_S162 + 3U)) = _S164;
        *((&kernelContext_17)->isl_0+(_S162 + 4U)) = _S164;
    }
    else
    {
        if(routed_0)
        {
            uint _S165 = atomic_fetch_or_explicit((&kernelContext_17)->actl_0+2U, 1U << (i_6 - _S157), memory_order_relaxed);
        }
    }
    if(i_6 == 0U)
    {
        uint _S166 = atomic_load_explicit((&kernelContext_17)->actl_0+1U, memory_order_relaxed);
        uint _S167 = atomic_load_explicit((&kernelContext_17)->actl_0+55U, memory_order_relaxed);
        if(_S167 != 0U)
        {
            _S161 = 2U;
        }
        else
        {
            _S161 = 0U;
        }
        uint err_0 = _S166 | _S161;
        uint device* _S168 = (&kernelContext_17)->ctl_0+20U;
        bool _S169 = gate_0(&kernelContext_17);
        if(_S169)
        {
            _S161 = err_0;
        }
        else
        {
            _S161 = 0U;
        }
        *_S168 = _S161;
        uint device* _S170 = (&kernelContext_17)->ctl_0+21U;
        bool _S171 = gate_0(&kernelContext_17);
        if(_S171)
        {
            uint _S172 = atomic_load_explicit((&kernelContext_17)->actl_0+0U, memory_order_relaxed);
            _S161 = _S172;
        }
        else
        {
            _S161 = 0U;
        }
        *_S170 = _S161;
    }
    return;
}

[[kernel]] void lr_mark(uint3 id_9 [[thread_position_in_grid]], uint device* counters_11 [[buffer(10)]], RouteParams_0 constant* params_11 [[buffer(0)]], uint device* route_11 [[buffer(11)]], Island_natural_0 device* islands_11 [[buffer(1)]], packed_float4 device* ctab_11 [[buffer(4)]], packed_float4 device* bodies_11 [[buffer(5)]], ChunkStatic_natural_0 device* chunks_11 [[buffer(2)]], packed_float4 device* state_11 [[buffer(3)]], uint device* rtab_11 [[buffer(8)]], BondStatic_natural_0 device* bond_static_11 [[buffer(6)]], MaterialTable_0 constant* materials_11 [[buffer(7)]], packed_float4 device* parts_11 [[buffer(12)]], packed_float4 device* imps_11 [[buffer(9)]], atomic<uint> device* actl_11 [[buffer(16)]], packed_uint2 device* sorted_11 [[buffer(13)]], atomic<uint> device* lab_11 [[buffer(15)]], uint device* seed_11 [[buffer(17)]], packed_uint4 device* pass_11 [[buffer(19)]], packed_uint2 device* edges_11 [[buffer(14)]], packed_uint4 device* isl_11 [[buffer(18)]], uint device* ctl_11 [[buffer(20)]])
{
    thread KernelContext_0 kernelContext_18;
    (&kernelContext_18)->counters_0 = counters_11;
    (&kernelContext_18)->params_0 = params_11;
    (&kernelContext_18)->route_0 = route_11;
    (&kernelContext_18)->islands_0 = islands_11;
    (&kernelContext_18)->ctab_0 = ctab_11;
    (&kernelContext_18)->bodies_0 = bodies_11;
    (&kernelContext_18)->chunks_0 = chunks_11;
    (&kernelContext_18)->state_0 = state_11;
    (&kernelContext_18)->rtab_0 = rtab_11;
    (&kernelContext_18)->bond_static_0 = bond_static_11;
    (&kernelContext_18)->materials_0 = materials_11;
    (&kernelContext_18)->parts_0 = parts_11;
    (&kernelContext_18)->imps_0 = imps_11;
    (&kernelContext_18)->actl_0 = actl_11;
    (&kernelContext_18)->sorted_0 = sorted_11;
    (&kernelContext_18)->lab_0 = lab_11;
    (&kernelContext_18)->seed_0 = seed_11;
    (&kernelContext_18)->pass_0 = pass_11;
    (&kernelContext_18)->edges_0 = edges_11;
    (&kernelContext_18)->isl_0 = isl_11;
    (&kernelContext_18)->ctl_0 = ctl_11;
    threadgroup array<float3, int(64)> g_lo_11;
    (&kernelContext_18)->g_lo_0 = &g_lo_11;
    threadgroup array<float3, int(64)> g_hi_11;
    (&kernelContext_18)->g_hi_0 = &g_hi_11;
    threadgroup array<float4, int(64)> g_a_11;
    (&kernelContext_18)->g_a_0 = &g_a_11;
    threadgroup array<float, int(64)> g_cap_11;
    (&kernelContext_18)->g_cap_0 = &g_cap_11;
    uint i_7 = id_9.x;
    uint _S173 = live_islands_0(&kernelContext_18);
    bool _S174;
    if(i_7 >= _S173)
    {
        _S174 = true;
    }
    else
    {
        bool _S175 = gate_0(&kernelContext_18);
        _S174 = !_S175;
    }
    if(_S174)
    {
        return;
    }
    uint _S176 = 5U * i_7;
    uint4 _S177 = uint4(*((&kernelContext_18)->isl_0+_S176)) ;
    if((_S177.x) != 0U)
    {
        packed_uint4 device* _S178 = (&kernelContext_18)->isl_0+_S176;
        uint _S179 = atomic_load_explicit((&kernelContext_18)->actl_0+2U, memory_order_relaxed);
        *_S178 = packed_uint4(uint4(2147483648U | (_S179 & 2147483647U), _S177.yzw)) ;
    }
    return;
}

