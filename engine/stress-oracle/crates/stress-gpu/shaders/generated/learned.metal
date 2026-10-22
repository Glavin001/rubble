#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

template<typename T, int A, int B>
matrix<T,A,B> _slang_matrixCompMult(matrix<T,A,B> m1, matrix<T,A,B> m2)
{
    matrix<T,A,B> result;
    for (int i = 0; i < A; i++)
        result[i] = m1[i] * m2[i];
    return result;
}

uint firstbitlow_0(uint value_0)
{
    uint _S1;
    if(value_0 == 0U)
    {
        _S1 = 4294967295U;
    }
    else
    {
        uint _S2 = ctz(value_0);
        _S1 = _S2;
    }
    return _S1;
}

uint4 group_uniform4_0(uint4 value_1, uint thread_0)
{
    return value_1;
}

struct LearnedParams_0
{
    uint impactor_count_0;
    uint engine_base_0;
    uint island_cap_0;
    uint stub_0;
    float dt_0;
    float speed_0;
    float log_tau_0;
    float tolerance_0;
    float4 gravity_0;
    uint max_calls_0;
    uint max_nodes_0;
    uint max_bonds_0;
    uint max_call_chunks_0;
    uint debris_0;
    uint max_cisl_0;
    uint chunk_count_0;
    uint pad0_0;
    uint4 out_bases_0;
    uint4 out_bases2_0;
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

struct KernelContext_0
{
    uint device* ctl_0;
    packed_float4 device* calls_0;
    packed_float4 device* nodes_0;
    LearnedParams_0 constant* params_0;
    packed_float4 device* bonds_0;
    uint device* chunk_node_0;
    uint device* counters_0;
    uint device* route_0;
    packed_uint4 device* isl_0;
    packed_float4 device* islf_0;
    Island_natural_0 device* islands_0;
    packed_float4 device* ctab_0;
    packed_float4 device* bodies_0;
    ChunkStatic_natural_0 device* chunks_0;
    packed_float4 device* state_0;
    packed_float4 device* imps_0;
    uint device* status_0;
    packed_uint4 device* largs_0;
    packed_uint4 device* cisl_0;
    BondDyn_natural_0 device* bond_dyn_0;
    BondStatic_natural_0 device* bond_static_0;
    packed_float4 device* frozen_0;
    uint device* split_work_0;
    uint device* split_control_0;
    packed_uint4 device* split_parents_0;
    packed_uint4 device* split_slots_0;
    packed_float4 device* split_out_0;
    packed_uint4 device* split_firsts_0;
    float device* lout_0;
    array<array<float4, int(256)>, int(4)> threadgroup* g_red_0;
    array<float, int(6)> threadgroup* g_lambda_0;
    array<float3, int(256)> threadgroup* g_lo_0;
    array<float3, int(256)> threadgroup* g_hi_0;
    array<uint, int(32)> threadgroup* g_label_0;
    uint threadgroup* g_used_0;
    array<uint, int(32)> threadgroup* g_call_of_0;
    array<uint, int(64)> threadgroup* g_mask_of_0;
    array<uint, int(64)> threadgroup* g_cnt_isl_0;
    array<uint, int(64)> threadgroup* g_cnt_chunk_0;
    array<uint, int(64)> threadgroup* g_cnt_bond_0;
    array<uint, int(64)> threadgroup* g_cnt_imp_0;
    array<uint, int(256)> threadgroup* g_scan_0;
    array<float, int(4096)> threadgroup* g_sort_0;
    uint threadgroup* g_stale_0;
    uint threadgroup* g_broke_0;
};

float4 node_0(uint n_0, uint k_0, KernelContext_0 thread* kernelContext_0)
{
    return float4(*(kernelContext_0->nodes_0+(14U * n_0 + k_0))) ;
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

void reduce_store_0(uint t_1, float4 a_0, float4 b_0, float4 c_0, float4 d_0, KernelContext_0 thread* kernelContext_1)
{
    (*kernelContext_1->g_red_0)[int(0)][t_1] = a_0;
    (*kernelContext_1->g_red_0)[int(1)][t_1] = b_0;
    (*kernelContext_1->g_red_0)[int(2)][t_1] = c_0;
    (*kernelContext_1->g_red_0)[int(3)][t_1] = d_0;
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
            uint _S3 = t_1 + s_0;
            (*kernelContext_1->g_red_0)[int(0)][t_1] = (*kernelContext_1->g_red_0)[int(0)][t_1] + (*kernelContext_1->g_red_0)[int(0)][_S3];
            (*kernelContext_1->g_red_0)[int(1)][t_1] = (*kernelContext_1->g_red_0)[int(1)][t_1] + (*kernelContext_1->g_red_0)[int(1)][_S3];
            (*kernelContext_1->g_red_0)[int(2)][t_1] = (*kernelContext_1->g_red_0)[int(2)][t_1] + (*kernelContext_1->g_red_0)[int(2)][_S3];
            (*kernelContext_1->g_red_0)[int(3)][t_1] = (*kernelContext_1->g_red_0)[int(3)][t_1] + (*kernelContext_1->g_red_0)[int(3)][_S3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    return;
}

void reduce_read_0(float4 thread* a_1, float4 thread* b_1, float4 thread* c_1, float4 thread* d_1, KernelContext_0 thread* kernelContext_2)
{
    *a_1 = (*kernelContext_2->g_red_0)[int(0)][int(0)];
    *b_1 = (*kernelContext_2->g_red_0)[int(1)][int(0)];
    *c_1 = (*kernelContext_2->g_red_0)[int(2)][int(0)];
    *d_1 = (*kernelContext_2->g_red_0)[int(3)][int(0)];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return;
}

matrix<float,int(3),int(3)>  quat_matrix_0(float4 q_2)
{
    float w_1 = q_2.w;
    float x_1 = q_2.x;
    float y_1 = q_2.y;
    float z_1 = q_2.z;
    float _S4 = y_1 * y_1;
    float _S5 = z_1 * z_1;
    float _S6 = x_1 * y_1;
    float _S7 = w_1 * z_1;
    float _S8 = x_1 * z_1;
    float _S9 = w_1 * y_1;
    float _S10 = x_1 * x_1;
    float _S11 = y_1 * z_1;
    float _S12 = w_1 * x_1;
    return matrix<float,int(3),int(3)> (float3(1.0f - 2.0f * (_S4 + _S5), 2.0f * (_S6 - _S7), 2.0f * (_S8 + _S9)), float3(2.0f * (_S6 + _S7), 1.0f - 2.0f * (_S10 + _S5), 2.0f * (_S11 - _S12)), float3(2.0f * (_S8 - _S9), 2.0f * (_S11 + _S12), 1.0f - 2.0f * (_S10 + _S4)));
}

matrix<float,int(3),int(3)>  diag3_0(float3 d_2)
{
    return matrix<float,int(3),int(3)> (float3(d_2.x, 0.0f, 0.0f), float3(0.0f, d_2.y, 0.0f), float3(0.0f, 0.0f, d_2.z));
}

matrix<float,int(3),int(3)>  world_inertia_0(float4 q_3, float3 principal_0)
{
    matrix<float,int(3),int(3)>  r_1 = quat_matrix_0(q_3);
    return (((transpose(r_1)) * ((((diag3_0(principal_0)) * (r_1))))));
}

matrix<float,int(3),int(3)>  skew_0(float3 r_2)
{
    float _S13 = r_2.z;
    float _S14 = r_2.y;
    float _S15 = r_2.x;
    return matrix<float,int(3),int(3)> (float3(0.0f, - _S13, _S14), float3(_S13, 0.0f, - _S15), float3(- _S14, _S15, 0.0f));
}

matrix<float,int(3),int(3)>  rotvec_matrix_0(float3 r_3)
{
    float th_0 = length(r_3);
    matrix<float,int(3),int(3)>  eye_0 = matrix<float,int(3),int(3)> (float3(1.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), float3(0.0f, 0.0f, 1.0f));
    if(th_0 < 9.99999997475242708e-07f)
    {
        return eye_0 + skew_0(r_3);
    }
    matrix<float,int(3),int(3)>  k_1 = skew_0(r_3 / float3(th_0) );
    return eye_0 + _slang_matrixCompMult(matrix<float,int(3),int(3)> (sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0)), k_1) + _slang_matrixCompMult(matrix<float,int(3),int(3)> (1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0)), (((k_1) * (k_1))));
}

matrix<float,int(3),int(3)>  outer3_0(float3 a_2, float3 b_2)
{
    return matrix<float,int(3),int(3)> (a_2 * float3(b_2.x) , a_2 * float3(b_2.y) , a_2 * float3(b_2.z) );
}

uint group_uniform_0(uint value_2, uint thread_1)
{
    return value_2;
}

float3 solve3_0(matrix<float,int(3),int(3)>  a_3, float3 b_3)
{
    float3 c0_0 = float3(a_3[int(0)][int(0)], a_3[int(1)][int(0)], a_3[int(2)][int(0)]);
    float3 c1_0 = float3(a_3[int(0)][int(1)], a_3[int(1)][int(1)], a_3[int(2)][int(1)]);
    float3 c2_0 = float3(a_3[int(0)][int(2)], a_3[int(1)][int(2)], a_3[int(2)][int(2)]);
    float3 _S16 = cross(c1_0, c2_0);
    return float3(dot(b_3, _S16), dot(c0_0, cross(b_3, c2_0)), dot(c0_0, cross(c1_0, b_3))) / float3(dot(c0_0, _S16)) ;
}

[[kernel]] void learned_decode(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], uint device* ctl_1 [[buffer(1)]], packed_float4 device* calls_1 [[buffer(2)]], packed_float4 device* nodes_1 [[buffer(3)]], LearnedParams_0 constant* params_1 [[buffer(0)]], packed_float4 device* bonds_1 [[buffer(4)]], uint device* chunk_node_1 [[buffer(8)]], uint device* counters_1 [[buffer(19)]], uint device* route_1 [[buffer(25)]], packed_uint4 device* isl_1 [[buffer(5)]], packed_float4 device* islf_1 [[buffer(6)]], Island_natural_0 device* islands_1 [[buffer(11)]], packed_float4 device* ctab_1 [[buffer(14)]], packed_float4 device* bodies_1 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_1 [[buffer(12)]], packed_float4 device* state_1 [[buffer(13)]], packed_float4 device* imps_1 [[buffer(10)]], uint device* status_1 [[buffer(18)]], packed_uint4 device* largs_1 [[buffer(27)]], packed_uint4 device* cisl_1 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_1 [[buffer(17)]], BondStatic_natural_0 device* bond_static_1 [[buffer(16)]], packed_float4 device* frozen_1 [[buffer(9)]], uint device* split_work_1 [[buffer(21)]], uint device* split_control_1 [[buffer(23)]], packed_uint4 device* split_parents_1 [[buffer(20)]], packed_uint4 device* split_slots_1 [[buffer(22)]], packed_float4 device* split_out_1 [[buffer(26)]], packed_uint4 device* split_firsts_1 [[buffer(24)]], float device* lout_1 [[buffer(28)]])
{
    float3 v_end_0;
    float3 w_end_0;
    uint q_4;
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->ctl_0 = ctl_1;
    (&kernelContext_3)->calls_0 = calls_1;
    (&kernelContext_3)->nodes_0 = nodes_1;
    (&kernelContext_3)->params_0 = params_1;
    (&kernelContext_3)->bonds_0 = bonds_1;
    (&kernelContext_3)->chunk_node_0 = chunk_node_1;
    (&kernelContext_3)->counters_0 = counters_1;
    (&kernelContext_3)->route_0 = route_1;
    (&kernelContext_3)->isl_0 = isl_1;
    (&kernelContext_3)->islf_0 = islf_1;
    (&kernelContext_3)->islands_0 = islands_1;
    (&kernelContext_3)->ctab_0 = ctab_1;
    (&kernelContext_3)->bodies_0 = bodies_1;
    (&kernelContext_3)->chunks_0 = chunks_1;
    (&kernelContext_3)->state_0 = state_1;
    (&kernelContext_3)->imps_0 = imps_1;
    (&kernelContext_3)->status_0 = status_1;
    (&kernelContext_3)->largs_0 = largs_1;
    (&kernelContext_3)->cisl_0 = cisl_1;
    (&kernelContext_3)->bond_dyn_0 = bond_dyn_1;
    (&kernelContext_3)->bond_static_0 = bond_static_1;
    (&kernelContext_3)->frozen_0 = frozen_1;
    (&kernelContext_3)->split_work_0 = split_work_1;
    (&kernelContext_3)->split_control_0 = split_control_1;
    (&kernelContext_3)->split_parents_0 = split_parents_1;
    (&kernelContext_3)->split_slots_0 = split_slots_1;
    (&kernelContext_3)->split_out_0 = split_out_1;
    (&kernelContext_3)->split_firsts_0 = split_firsts_1;
    (&kernelContext_3)->lout_0 = lout_1;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_1;
    (&kernelContext_3)->g_red_0 = &g_red_1;
    threadgroup array<float, int(6)> g_lambda_1;
    (&kernelContext_3)->g_lambda_0 = &g_lambda_1;
    threadgroup array<float3, int(256)> g_lo_1;
    (&kernelContext_3)->g_lo_0 = &g_lo_1;
    threadgroup array<float3, int(256)> g_hi_1;
    (&kernelContext_3)->g_hi_0 = &g_hi_1;
    threadgroup array<uint, int(32)> g_label_1;
    (&kernelContext_3)->g_label_0 = &g_label_1;
    threadgroup uint g_used_1;
    (&kernelContext_3)->g_used_0 = &g_used_1;
    threadgroup array<uint, int(32)> g_call_of_1;
    (&kernelContext_3)->g_call_of_0 = &g_call_of_1;
    threadgroup array<uint, int(64)> g_mask_of_1;
    (&kernelContext_3)->g_mask_of_0 = &g_mask_of_1;
    threadgroup array<uint, int(64)> g_cnt_isl_1;
    (&kernelContext_3)->g_cnt_isl_0 = &g_cnt_isl_1;
    threadgroup array<uint, int(64)> g_cnt_chunk_1;
    (&kernelContext_3)->g_cnt_chunk_0 = &g_cnt_chunk_1;
    threadgroup array<uint, int(64)> g_cnt_bond_1;
    (&kernelContext_3)->g_cnt_bond_0 = &g_cnt_bond_1;
    threadgroup array<uint, int(64)> g_cnt_imp_1;
    (&kernelContext_3)->g_cnt_imp_0 = &g_cnt_imp_1;
    threadgroup array<uint, int(256)> g_scan_1;
    (&kernelContext_3)->g_scan_0 = &g_scan_1;
    threadgroup array<float, int(4096)> g_sort_1;
    (&kernelContext_3)->g_sort_0 = &g_sort_1;
    threadgroup uint g_stale_1;
    (&kernelContext_3)->g_stale_0 = &g_stale_1;
    threadgroup uint g_broke_1;
    (&kernelContext_3)->g_broke_0 = &g_broke_1;
    uint g_0 = group_0.x;
    uint t_2 = thread_2.x;
    if(g_0 >= (*(ctl_1+1U)))
    {
        return;
    }
    uint _S17 = 8U * g_0;
    float4 _S18 = float4(*((&kernelContext_3)->calls_0+_S17)) ;
    float4 _S19 = float4(*((&kernelContext_3)->calls_0+(_S17 + 2U))) ;
    uint _S20 = _S17 + 4U;
    float4 _S21 = float4(*((&kernelContext_3)->calls_0+_S20)) ;
    float _S22 = _S18.x;
    float _S23 = _S18.z;
    float dt_1 = (float4(*((&kernelContext_3)->calls_0+(_S17 + 1U))) ).w;
    float e_in_0 = _S19.x;
    float _S24 = _S21.y;
    uint4 u_0 = group_uniform4_0(uint4((as_type<uint>((_S19.y))), (as_type<uint>((_S19.z))), (as_type<uint>((_S19.w))), (as_type<uint>((_S24)))), t_2);
    uint flags_0 = u_0.x;
    uint _S25 = u_0.y;
    uint _S26 = u_0.z;
    uint _S27 = u_0.w;
    float _S28 = _S21.x;
    uint _S29 = (as_type<uint>((_S28)));
    float3 grav_0 = (float4(*((&kernelContext_3)->calls_0+(_S17 + 3U))) ).xyz;
    bool free_0 = (flags_0 & 1U) != 0U;
    uint j_0 = t_2;
    for(;;)
    {
        if(j_0 < _S26)
        {
        }
        else
        {
            break;
        }
        uint k_2 = _S25 + j_0;
        float4 _S30 = node_0(k_2, 1U, &kernelContext_3);
        float4 _S31 = node_0(k_2, 3U, &kernelContext_3);
        float3 v_1 = _S31.xyz;
        float4 _S32 = node_0(k_2, 4U, &kernelContext_3);
        float3 w_2 = _S32.xyz;
        float4 _S33 = node_0(k_2, 5U, &kernelContext_3);
        float4 _S34 = node_0(k_2, 6U, &kernelContext_3);
        float4 _S35 = node_0(k_2, 7U, &kernelContext_3);
        Quat_0 qq_0 = quat_of_0(_S30);
        float3 _S36 = _S33.xyz;
        thread Quat_0 _S37 = qq_0;
        float3 _S38 = rotate_0(&_S37, _S36);
        float3 _S39 = float3(_S23) ;
        float3 _S40 = v_1 + _S38 * _S39;
        float3 _S41 = float3(_S33.w, _S34.x, _S34.y);
        thread Quat_0 _S42 = qq_0;
        float3 _S43 = rotate_0(&_S42, _S41);
        float3 _S44 = float3(_S22) ;
        float3 _S45 = w_2 + _S43 * _S39 / _S44;
        float3 _S46 = float3(dt_1) ;
        float3 _S47 = v_1 * _S46 + float3(0.5f)  * grav_0 * _S46 * _S46;
        float3 _S48 = float3(_S34.z, _S34.w, _S35.x);
        thread Quat_0 _S49 = qq_0;
        float3 _S50 = rotate_0(&_S49, _S48);
        float3 _S51 = _S47 + _S50 * _S39 * _S46;
        float3 _S52 = w_2 * _S46;
        float3 _S53 = _S35.yzw;
        thread Quat_0 _S54 = qq_0;
        float3 _S55 = rotate_0(&_S54, _S53);
        float3 _S56 = _S52 + _S55 * _S39 * _S46 / _S44;
        float4 _S57 = node_0(k_2, 2U, &kernelContext_3);
        float3 dx_0;
        float3 dr_0;
        if((_S57.w) != 0.0f)
        {
            float3 _S58 = float3(0.0f) ;
            v_end_0 = _S58;
            w_end_0 = _S58;
            dx_0 = _S58;
            dr_0 = _S58;
        }
        else
        {
            v_end_0 = _S40;
            w_end_0 = _S45;
            dx_0 = _S51;
            dr_0 = _S56;
        }
        uint _S59 = 14U * k_2;
        *((&kernelContext_3)->nodes_0+(_S59 + 8U)) = packed_float4(float4(v_end_0, 0.0f)) ;
        *((&kernelContext_3)->nodes_0+(_S59 + 9U)) = packed_float4(float4(w_end_0, 0.0f)) ;
        *((&kernelContext_3)->nodes_0+(_S59 + 10U)) = packed_float4(float4(dx_0, 0.0f)) ;
        *((&kernelContext_3)->nodes_0+(_S59 + 11U)) = packed_float4(float4(dr_0, 0.0f)) ;
        j_0 = j_0 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    thread float4 a_4;
    thread float4 b_4;
    thread float4 c_2;
    thread float4 d_3;
    float4 _S60 = float4(0.0f) ;
    j_0 = t_2;
    float4 sa_0 = _S60;
    float4 sb_0 = _S60;
    float4 sc_0 = _S60;
    for(;;)
    {
        if(j_0 < _S26)
        {
        }
        else
        {
            break;
        }
        uint k_3 = _S25 + j_0;
        float4 _S61 = node_0(k_3, 0U, &kernelContext_3);
        float m_0 = _S61.w;
        float3 _S62 = float3(m_0) ;
        float4 sa_1 = sa_0 + float4(_S61.xyz * _S62, m_0);
        float4 _S63 = node_0(k_3, 3U, &kernelContext_3);
        float4 sb_1 = sb_0 + float4(_S63.xyz * _S62, 0.0f);
        float4 _S64 = node_0(k_3, 10U, &kernelContext_3);
        float4 sc_1 = sc_0 + float4(_S64.xyz * _S62, 0.0f);
        j_0 = j_0 + 256U;
        sa_0 = sa_1;
        sb_0 = sb_1;
        sc_0 = sc_1;
    }
    reduce_store_0(t_2, sa_0, sb_0, sc_0, _S60, &kernelContext_3);
    reduce_read_0(&a_4, &b_4, &c_2, &d_3, &kernelContext_3);
    float mt_0 = a_4.w;
    float3 _S65 = float3(mt_0) ;
    float3 _S66 = a_4.xyz / _S65;
    float3 p0_0 = b_4.xyz;
    float3 have_0 = c_2.xyz;
    float3 _S67 = float3(0.0f) ;
    if(free_0)
    {
        float3 _S68 = float3(dt_1) ;
        v_end_0 = (p0_0 * _S68 + float3((0.5f * mt_0))  * grav_0 * _S68 * _S68 - have_0) / _S65;
    }
    else
    {
        v_end_0 = _S67;
    }
    j_0 = t_2;
    sa_0 = _S60;
    sb_0 = _S60;
    for(;;)
    {
        if(j_0 < _S26)
        {
        }
        else
        {
            break;
        }
        uint k_4 = _S25 + j_0;
        float4 _S69 = node_0(k_4, 0U, &kernelContext_3);
        float m_1 = _S69.w;
        float4 _S70 = node_0(k_4, 10U, &kernelContext_3);
        float3 dx_1 = _S70.xyz;
        if(free_0)
        {
            w_end_0 = dx_1 + v_end_0;
        }
        else
        {
            w_end_0 = dx_1;
        }
        float3 _S71 = _S69.xyz;
        float3 _S72 = float3(m_1) ;
        float4 sa_2 = sa_0 + float4((_S71 + w_end_0) * _S72, 0.0f);
        float3 r0_0 = _S71 - _S66;
        float4 _S73 = node_0(k_4, 1U, &kernelContext_3);
        float4 _S74 = node_0(k_4, 2U, &kernelContext_3);
        matrix<float,int(3),int(3)>  i0_0 = world_inertia_0(_S73, _S74.xyz);
        float4 _S75 = node_0(k_4, 3U, &kernelContext_3);
        float3 _S76 = cross(r0_0, _S72 * _S75.xyz);
        float4 _S77 = node_0(k_4, 4U, &kernelContext_3);
        float4 sb_2 = sb_0 + float4(_S76 + (((_S77.xyz) * (i0_0))), 0.0f);
        j_0 = j_0 + 256U;
        sa_0 = sa_2;
        sb_0 = sb_2;
    }
    reduce_store_0(t_2, sa_0, sb_0, _S60, _S60, &kernelContext_3);
    reduce_read_0(&a_4, &b_4, &c_2, &d_3, &kernelContext_3);
    float3 _S78 = a_4.xyz / _S65;
    float3 l0_0 = b_4.xyz;
    uint dec_0;
    uint bad_0;
    float4 ta_0;
    float4 tb_0;
    float4 tc_0;
    if(free_0)
    {
        j_0 = t_2;
        for(;;)
        {
            if(j_0 < _S26)
            {
            }
            else
            {
                break;
            }
            uint k_5 = _S25 + j_0;
            packed_float4 device* _S79 = (&kernelContext_3)->nodes_0+(14U * k_5 + 10U);
            float4 _S80 = node_0(k_5, 10U, &kernelContext_3);
            *_S79 = packed_float4(float4(_S80.xyz + v_end_0, 0.0f)) ;
            j_0 = j_0 + 256U;
        }
        j_0 = t_2;
        sa_0 = _S60;
        sb_0 = _S60;
        sc_0 = _S60;
        ta_0 = _S60;
        tb_0 = _S60;
        tc_0 = _S60;
        for(;;)
        {
            if(j_0 < _S26)
            {
            }
            else
            {
                break;
            }
            uint k_6 = _S25 + j_0;
            float4 _S81 = node_0(k_6, 0U, &kernelContext_3);
            float m_2 = _S81.w;
            float3 _S82 = _S81.xyz;
            float4 _S83 = node_0(k_6, 10U, &kernelContext_3);
            float3 r_4 = _S82 + _S83.xyz - _S78;
            float4 _S84 = node_0(k_6, 11U, &kernelContext_3);
            matrix<float,int(3),int(3)>  rd_0 = rotvec_matrix_0(_S84.xyz);
            float4 _S85 = node_0(k_6, 1U, &kernelContext_3);
            float4 _S86 = node_0(k_6, 2U, &kernelContext_3);
            matrix<float,int(3),int(3)>  i1_0 = (((transpose(rd_0)) * ((((world_inertia_0(_S85, _S86.xyz)) * (rd_0))))));
            float4 _S87 = node_0(k_6, 8U, &kernelContext_3);
            float3 v_2 = _S87.xyz;
            float4 _S88 = node_0(k_6, 9U, &kernelContext_3);
            float3 _S89 = float3(m_2) ;
            float3 _S90 = _S89 * v_2;
            float4 sa_3 = sa_0 + float4(_S90, 0.0f);
            float4 sb_3 = sb_0 + float4(cross(r_4, _S90) + (((_S88.xyz) * (i1_0))), 0.0f);
            float4 sc_2 = sc_0 + float4(_S89 * r_4, 0.0f);
            matrix<float,int(3),int(3)>  jj_0 = _slang_matrixCompMult(matrix<float,int(3),int(3)> (m_2, m_2, m_2, m_2, m_2, m_2, m_2, m_2, m_2), _slang_matrixCompMult(matrix<float,int(3),int(3)> (dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4)), matrix<float,int(3),int(3)> (float3(1.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), float3(0.0f, 0.0f, 1.0f))) - outer3_0(r_4, r_4)) + i1_0;
            float4 sd_0 = ta_0 + float4(jj_0[int(0)], 0.0f);
            float4 se_0 = tb_0 + float4(jj_0[int(1)], 0.0f);
            float4 sf_0 = tc_0 + float4(jj_0[int(2)], 0.0f);
            j_0 = j_0 + 256U;
            sa_0 = sa_3;
            sb_0 = sb_3;
            sc_0 = sc_2;
            ta_0 = sd_0;
            tb_0 = se_0;
            tc_0 = sf_0;
        }
        reduce_store_0(t_2, sa_0, sb_0, sc_0, ta_0, &kernelContext_3);
        thread float4 ax_p_0;
        thread float4 ax_l_0;
        thread float4 smr_0;
        thread float4 j0_0;
        reduce_read_0(&ax_p_0, &ax_l_0, &smr_0, &j0_0, &kernelContext_3);
        reduce_store_0(t_2, tb_0, tc_0, _S60, _S60, &kernelContext_3);
        thread float4 j1_0;
        thread float4 j2_0;
        thread float4 unused0_0;
        thread float4 unused1_0;
        reduce_read_0(&j1_0, &j2_0, &unused0_0, &unused1_0, &kernelContext_3);
        if(t_2 == 0U)
        {
            float3 _S91 = p0_0 + _S65 * grav_0 * float3(dt_1)  - ax_p_0.xyz;
            float3 _S92 = l0_0 - ax_l_0.xyz;
            matrix<float,int(3),int(3)>  _S93 = skew_0(smr_0.xyz);
            thread array<array<float, int(7)>, int(6)> m6_0;
            dec_0 = 0U;
            for(;;)
            {
                if(dec_0 < 6U)
                {
                }
                else
                {
                    break;
                }
                q_4 = 0U;
                for(;;)
                {
                    if(q_4 < 7U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    m6_0[dec_0][q_4] = 0.0f;
                    q_4 = q_4 + 1U;
                }
                dec_0 = dec_0 + 1U;
            }
            array<float3, int(3)> _S94 = { { j0_0.xyz, j1_0.xyz, j2_0.xyz } };
            dec_0 = 0U;
            for(;;)
            {
                if(dec_0 < 3U)
                {
                }
                else
                {
                    break;
                }
                m6_0[dec_0][dec_0] = mt_0;
                q_4 = 0U;
                for(;;)
                {
                    if(q_4 < 3U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    uint _S95 = 3U + q_4;
                    m6_0[dec_0][_S95] = - _S93[dec_0][q_4];
                    m6_0[3U + dec_0][q_4] = _S93[dec_0][q_4];
                    m6_0[3U + dec_0][_S95] = _S94[dec_0][q_4];
                    q_4 = q_4 + 1U;
                }
                m6_0[dec_0][int(6)] = _S91[dec_0];
                m6_0[3U + dec_0][int(6)] = _S92[dec_0];
                dec_0 = dec_0 + 1U;
            }
            bad_0 = 0U;
            for(;;)
            {
                if(bad_0 < 6U)
                {
                }
                else
                {
                    break;
                }
                uint _S96 = bad_0 + 1U;
                q_4 = bad_0;
                dec_0 = _S96;
                for(;;)
                {
                    if(dec_0 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    if((abs(m6_0[dec_0][bad_0])) > (abs(m6_0[q_4][bad_0])))
                    {
                        q_4 = dec_0;
                    }
                    dec_0 = dec_0 + 1U;
                }
                uint q_5 = 0U;
                for(;;)
                {
                    if(q_5 < 7U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float tmp_0 = m6_0[bad_0][q_5];
                    m6_0[bad_0][q_5] = m6_0[q_4][q_5];
                    m6_0[q_4][q_5] = tmp_0;
                    q_5 = q_5 + 1U;
                }
                uint r_5 = _S96;
                for(;;)
                {
                    if(r_5 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float _S97 = m6_0[r_5][bad_0] / m6_0[bad_0][bad_0];
                    uint q_6 = bad_0;
                    for(;;)
                    {
                        if(q_6 < 7U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        m6_0[r_5][q_6] = m6_0[r_5][q_6] - _S97 * m6_0[bad_0][q_6];
                        q_6 = q_6 + 1U;
                    }
                    r_5 = r_5 + 1U;
                }
                bad_0 = _S96;
            }
            thread array<float, int(6)> x6_0;
            x6_0[int(0)] = 0.0f;
            x6_0[int(1)] = 0.0f;
            x6_0[int(2)] = 0.0f;
            x6_0[int(3)] = 0.0f;
            x6_0[int(4)] = 0.0f;
            x6_0[int(5)] = 0.0f;
            int r_6 = int(5);
            for(;;)
            {
                if(r_6 >= int(0))
                {
                }
                else
                {
                    break;
                }
                q_4 = uint(r_6) + 1U;
                float acc_0 = m6_0[r_6][int(6)];
                for(;;)
                {
                    if(q_4 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float acc_1 = acc_0 - m6_0[r_6][q_4] * x6_0[q_4];
                    q_4 = q_4 + 1U;
                    acc_0 = acc_1;
                }
                x6_0[r_6] = acc_0 / m6_0[r_6][r_6];
                r_6 = r_6 - int(1);
            }
            dec_0 = 0U;
            for(;;)
            {
                if(dec_0 < 6U)
                {
                }
                else
                {
                    break;
                }
                (*(&kernelContext_3)->g_lambda_0)[dec_0] = x6_0[dec_0];
                dec_0 = dec_0 + 1U;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float3 _S98 = float3((*(&kernelContext_3)->g_lambda_0)[int(0)], (*(&kernelContext_3)->g_lambda_0)[int(1)], (*(&kernelContext_3)->g_lambda_0)[int(2)]);
        float3 _S99 = float3((*(&kernelContext_3)->g_lambda_0)[int(3)], (*(&kernelContext_3)->g_lambda_0)[int(4)], (*(&kernelContext_3)->g_lambda_0)[int(5)]);
        j_0 = t_2;
        for(;;)
        {
            if(j_0 < _S26)
            {
            }
            else
            {
                break;
            }
            uint k_7 = _S25 + j_0;
            float4 _S100 = node_0(k_7, 0U, &kernelContext_3);
            float3 _S101 = _S100.xyz;
            float4 _S102 = node_0(k_7, 10U, &kernelContext_3);
            float3 r_7 = _S101 + _S102.xyz - _S78;
            uint _S103 = 14U * k_7;
            packed_float4 device* _S104 = (&kernelContext_3)->nodes_0+(_S103 + 8U);
            float4 _S105 = node_0(k_7, 8U, &kernelContext_3);
            *_S104 = packed_float4(float4(_S105.xyz + _S98 + cross(_S99, r_7), 0.0f)) ;
            packed_float4 device* _S106 = (&kernelContext_3)->nodes_0+(_S103 + 9U);
            float4 _S107 = node_0(k_7, 9U, &kernelContext_3);
            *_S106 = packed_float4(float4(_S107.xyz + _S99, 0.0f)) ;
            j_0 = j_0 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    }
    if((flags_0 & 2U) == 0U)
    {
        j_0 = t_2;
        sa_0 = _S60;
        sb_0 = _S60;
        for(;;)
        {
            if(j_0 < _S26)
            {
            }
            else
            {
                break;
            }
            uint k_8 = _S25 + j_0;
            float4 _S108 = node_0(k_8, 0U, &kernelContext_3);
            float m_3 = _S108.w;
            float4 _S109 = node_0(k_8, 1U, &kernelContext_3);
            float4 _S110 = node_0(k_8, 2U, &kernelContext_3);
            matrix<float,int(3),int(3)>  i0_1 = world_inertia_0(_S109, _S110.xyz);
            float4 _S111 = node_0(k_8, 11U, &kernelContext_3);
            matrix<float,int(3),int(3)>  rd_1 = rotvec_matrix_0(_S111.xyz);
            matrix<float,int(3),int(3)>  i1_1 = (((transpose(rd_1)) * ((((i0_1) * (rd_1))))));
            float4 _S112 = node_0(k_8, 3U, &kernelContext_3);
            float3 v0_0 = _S112.xyz;
            float4 _S113 = node_0(k_8, 4U, &kernelContext_3);
            float3 w0_0 = _S113.xyz;
            float4 _S114 = node_0(k_8, 8U, &kernelContext_3);
            float3 v_3 = _S114.xyz;
            float4 _S115 = node_0(k_8, 9U, &kernelContext_3);
            float3 w_3 = _S115.xyz;
            float ke_in_0 = 0.5f * (m_3 * dot(v0_0, v0_0) + dot(w0_0, (((w0_0) * (i0_1)))));
            float4 _S116 = node_0(k_8, 10U, &kernelContext_3);
            float4 sa_4 = sa_0 + float4(ke_in_0, m_3 * dot(grav_0, _S116.xyz), 0.5f * (m_3 * dot(v_3, v_3) + dot(w_3, (((w_3) * (i1_1))))), 0.0f);
            float4 sb_4 = sb_0 + float4(float3(m_3)  * v_3, 0.0f);
            j_0 = j_0 + 256U;
            sa_0 = sa_4;
            sb_0 = sb_4;
        }
        reduce_store_0(t_2, sa_0, sb_0, _S60, _S60, &kernelContext_3);
        reduce_read_0(&a_4, &b_4, &c_2, &d_3, &kernelContext_3);
        float bound_0 = a_4.x + a_4.y + (&kernelContext_3)->params_0->tolerance_0 * e_in_0;
        float ke_0 = a_4.z;
        if(ke_0 > bound_0)
        {
            j_0 = 1U;
        }
        else
        {
            j_0 = 0U;
        }
        if((group_uniform_0(j_0, t_2)) != 0U)
        {
            if(free_0)
            {
                float3 vc_0 = b_4.xyz / _S65;
                j_0 = t_2;
                ta_0 = _S60;
                tb_0 = _S60;
                tc_0 = _S60;
                float4 td_0 = _S60;
                for(;;)
                {
                    if(j_0 < _S26)
                    {
                    }
                    else
                    {
                        break;
                    }
                    uint k_9 = _S25 + j_0;
                    float4 _S117 = node_0(k_9, 0U, &kernelContext_3);
                    float m_4 = _S117.w;
                    float4 _S118 = node_0(k_9, 0U, &kernelContext_3);
                    float3 _S119 = _S118.xyz;
                    float4 _S120 = node_0(k_9, 10U, &kernelContext_3);
                    float3 r_8 = _S119 + _S120.xyz - _S78;
                    float4 _S121 = node_0(k_9, 11U, &kernelContext_3);
                    matrix<float,int(3),int(3)>  rd_2 = rotvec_matrix_0(_S121.xyz);
                    float4 _S122 = node_0(k_9, 1U, &kernelContext_3);
                    float4 _S123 = node_0(k_9, 2U, &kernelContext_3);
                    matrix<float,int(3),int(3)>  i1_2 = (((transpose(rd_2)) * ((((world_inertia_0(_S122, _S123.xyz)) * (rd_2))))));
                    matrix<float,int(3),int(3)>  jj_1 = _slang_matrixCompMult(matrix<float,int(3),int(3)> (m_4, m_4, m_4, m_4, m_4, m_4, m_4, m_4, m_4), _slang_matrixCompMult(matrix<float,int(3),int(3)> (dot(r_8, r_8), dot(r_8, r_8), dot(r_8, r_8), dot(r_8, r_8), dot(r_8, r_8), dot(r_8, r_8), dot(r_8, r_8), dot(r_8, r_8), dot(r_8, r_8)), matrix<float,int(3),int(3)> (float3(1.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), float3(0.0f, 0.0f, 1.0f))) - outer3_0(r_8, r_8)) + i1_2;
                    float4 _S124 = node_0(k_9, 8U, &kernelContext_3);
                    float4 ta_1 = ta_0 + float4(jj_1[int(0)], 0.0f);
                    float4 tb_1 = tb_0 + float4(jj_1[int(1)], 0.0f);
                    float4 tc_1 = tc_0 + float4(jj_1[int(2)], 0.0f);
                    float3 _S125 = cross(r_8, float3(m_4)  * (_S124.xyz - vc_0));
                    float4 _S126 = node_0(k_9, 9U, &kernelContext_3);
                    float4 td_1 = td_0 + float4(_S125 + (((_S126.xyz) * (i1_2))), 0.0f);
                    j_0 = j_0 + 256U;
                    ta_0 = ta_1;
                    tb_0 = tb_1;
                    tc_0 = tc_1;
                    td_0 = td_1;
                }
                reduce_store_0(t_2, ta_0, tb_0, tc_0, td_0, &kernelContext_3);
                reduce_read_0(&a_4, &b_4, &c_2, &d_3, &kernelContext_3);
                float3 lrel_0 = d_3.xyz;
                float3 omega_0 = solve3_0(matrix<float,int(3),int(3)> (a_4.xyz, b_4.xyz, c_2.xyz), lrel_0);
                float ke_base_0 = 0.5f * (mt_0 * dot(vc_0, vc_0) + dot(omega_0, lrel_0));
                thread float4 ra_1 = _S60;
                j_0 = t_2;
                for(;;)
                {
                    if(j_0 < _S26)
                    {
                    }
                    else
                    {
                        break;
                    }
                    uint k_10 = _S25 + j_0;
                    float4 _S127 = node_0(k_10, 0U, &kernelContext_3);
                    float m_5 = _S127.w;
                    float4 _S128 = node_0(k_10, 0U, &kernelContext_3);
                    float3 _S129 = _S128.xyz;
                    float4 _S130 = node_0(k_10, 10U, &kernelContext_3);
                    float3 r_9 = _S129 + _S130.xyz - _S78;
                    float4 _S131 = node_0(k_10, 11U, &kernelContext_3);
                    matrix<float,int(3),int(3)>  rd_3 = rotvec_matrix_0(_S131.xyz);
                    float4 _S132 = node_0(k_10, 1U, &kernelContext_3);
                    float4 _S133 = node_0(k_10, 2U, &kernelContext_3);
                    matrix<float,int(3),int(3)>  i1_3 = (((transpose(rd_3)) * ((((world_inertia_0(_S132, _S133.xyz)) * (rd_3))))));
                    float4 _S134 = node_0(k_10, 8U, &kernelContext_3);
                    float3 uu_0 = _S134.xyz - vc_0 - cross(omega_0, r_9);
                    float4 _S135 = node_0(k_10, 9U, &kernelContext_3);
                    float3 wr_0 = _S135.xyz - omega_0;
                    ra_1.x = ra_1.x + 0.5f * (m_5 * dot(uu_0, uu_0) + dot(wr_0, (((wr_0) * (i1_3)))));
                    j_0 = j_0 + 256U;
                }
                reduce_store_0(t_2, ra_1, _S60, _S60, _S60, &kernelContext_3);
                reduce_read_0(&a_4, &b_4, &c_2, &d_3, &kernelContext_3);
                float _S136 = sqrt(clamp((bound_0 - ke_base_0) / max(a_4.x, 1.00000000317107685e-30f), 0.0f, 1.0f));
                if(ke_base_0 > bound_0)
                {
                    dec_0 = 3U;
                }
                else
                {
                    dec_0 = 1U;
                }
                j_0 = t_2;
                for(;;)
                {
                    if(j_0 < _S26)
                    {
                    }
                    else
                    {
                        break;
                    }
                    uint k_11 = _S25 + j_0;
                    float4 _S137 = node_0(k_11, 0U, &kernelContext_3);
                    float3 _S138 = _S137.xyz;
                    float4 _S139 = node_0(k_11, 10U, &kernelContext_3);
                    float3 r_10 = _S138 + _S139.xyz - _S78;
                    float4 _S140 = node_0(k_11, 8U, &kernelContext_3);
                    float3 _S141 = cross(omega_0, r_10);
                    float3 uu_1 = _S140.xyz - vc_0 - _S141;
                    float4 _S142 = node_0(k_11, 9U, &kernelContext_3);
                    float3 wr_1 = _S142.xyz - omega_0;
                    uint _S143 = 14U * k_11;
                    float3 _S144 = float3(_S136) ;
                    *((&kernelContext_3)->nodes_0+(_S143 + 8U)) = packed_float4(float4(vc_0 + _S141 + _S144 * uu_1, 0.0f)) ;
                    *((&kernelContext_3)->nodes_0+(_S143 + 9U)) = packed_float4(float4(omega_0 + _S144 * wr_1, 0.0f)) ;
                    j_0 = j_0 + 256U;
                }
            }
            else
            {
                float _S145 = sqrt(min(max(bound_0, 0.0f) / max(ke_0, 1.00000000317107685e-30f), 1.0f));
                j_0 = t_2;
                for(;;)
                {
                    if(j_0 < _S26)
                    {
                    }
                    else
                    {
                        break;
                    }
                    uint k_12 = _S25 + j_0;
                    uint _S146 = 14U * k_12;
                    packed_float4 device* _S147 = (&kernelContext_3)->nodes_0+(_S146 + 8U);
                    float4 _S148 = node_0(k_12, 8U, &kernelContext_3);
                    float3 _S149 = float3(_S145) ;
                    *_S147 = packed_float4(float4(_S148.xyz * _S149, 0.0f)) ;
                    packed_float4 device* _S150 = (&kernelContext_3)->nodes_0+(_S146 + 9U);
                    float4 _S151 = node_0(k_12, 9U, &kernelContext_3);
                    *_S150 = packed_float4(float4(_S151.xyz * _S149, 0.0f)) ;
                    j_0 = j_0 + 256U;
                }
                dec_0 = 1U;
            }
        }
        else
        {
            dec_0 = 0U;
        }
    }
    else
    {
        dec_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    j_0 = t_2;
    for(;;)
    {
        if(j_0 < _S27)
        {
        }
        else
        {
            break;
        }
        uint _S152 = 2U * (_S29 + j_0);
        float4 _S153 = float4(*((&kernelContext_3)->bonds_0+_S152)) ;
        uint _S154 = _S152 + 1U;
        float4 _S155 = float4(*((&kernelContext_3)->bonds_0+_S154)) ;
        float _S156 = min(_S153.x + _S153.z, 0.99999898672103882f);
        packed_float4 device* _S157 = (&kernelContext_3)->bonds_0+_S154;
        if((_S153.y) >= ((&kernelContext_3)->params_0->log_tau_0))
        {
            bad_0 = 1U;
        }
        else
        {
            bad_0 = 0U;
        }
        *_S157 = packed_float4(float4(_S156, (as_type<float>((bad_0))), _S155.z, _S155.w)) ;
        j_0 = j_0 + 256U;
    }
    bad_0 = 0U;
    j_0 = t_2;
    for(;;)
    {
        if(j_0 < _S26)
        {
        }
        else
        {
            break;
        }
        uint k_13 = _S25 + j_0;
        float4 _S158 = node_0(k_13, 8U, &kernelContext_3);
        float4 _S159 = float4(1.0f) ;
        float _S160 = dot(_S158, _S159);
        float4 _S161 = node_0(k_13, 9U, &kernelContext_3);
        float _S162 = _S160 + dot(_S161, _S159);
        float4 _S163 = node_0(k_13, 10U, &kernelContext_3);
        float _S164 = _S162 + dot(_S163, _S159);
        float4 _S165 = node_0(k_13, 11U, &kernelContext_3);
        if(!((abs(_S164 + dot(_S165, _S159))) < 3.00000000549775576e+38f))
        {
            bad_0 = 1U;
        }
        j_0 = j_0 + 256U;
    }
    reduce_store_0(t_2, float4(float(bad_0), 0.0f, 0.0f, 0.0f), _S60, _S60, _S60, &kernelContext_3);
    reduce_read_0(&a_4, &b_4, &c_2, &d_3, &kernelContext_3);
    if((a_4.x) > 0.0f)
    {
        dec_0 = dec_0 | 4U;
    }
    if(t_2 == 0U)
    {
        *((&kernelContext_3)->calls_0+_S20) = packed_float4(float4(_S28, _S24, (as_type<float>((dec_0))), _S21.w)) ;
    }
    return;
}

[[kernel]] void learned_clear(uint3 thread_3 [[thread_position_in_threadgroup]], uint device* ctl_2 [[buffer(1)]], packed_float4 device* calls_2 [[buffer(2)]], packed_float4 device* nodes_2 [[buffer(3)]], LearnedParams_0 constant* params_2 [[buffer(0)]], packed_float4 device* bonds_2 [[buffer(4)]], uint device* chunk_node_2 [[buffer(8)]], uint device* counters_2 [[buffer(19)]], uint device* route_2 [[buffer(25)]], packed_uint4 device* isl_2 [[buffer(5)]], packed_float4 device* islf_2 [[buffer(6)]], Island_natural_0 device* islands_2 [[buffer(11)]], packed_float4 device* ctab_2 [[buffer(14)]], packed_float4 device* bodies_2 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_2 [[buffer(12)]], packed_float4 device* state_2 [[buffer(13)]], packed_float4 device* imps_2 [[buffer(10)]], uint device* status_2 [[buffer(18)]], packed_uint4 device* largs_2 [[buffer(27)]], packed_uint4 device* cisl_2 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_2 [[buffer(17)]], BondStatic_natural_0 device* bond_static_2 [[buffer(16)]], packed_float4 device* frozen_2 [[buffer(9)]], uint device* split_work_2 [[buffer(21)]], uint device* split_control_2 [[buffer(23)]], packed_uint4 device* split_parents_2 [[buffer(20)]], packed_uint4 device* split_slots_2 [[buffer(22)]], packed_float4 device* split_out_2 [[buffer(26)]], packed_uint4 device* split_firsts_2 [[buffer(24)]], float device* lout_2 [[buffer(28)]])
{
    thread KernelContext_0 kernelContext_4;
    (&kernelContext_4)->ctl_0 = ctl_2;
    (&kernelContext_4)->calls_0 = calls_2;
    (&kernelContext_4)->nodes_0 = nodes_2;
    (&kernelContext_4)->params_0 = params_2;
    (&kernelContext_4)->bonds_0 = bonds_2;
    (&kernelContext_4)->chunk_node_0 = chunk_node_2;
    (&kernelContext_4)->counters_0 = counters_2;
    (&kernelContext_4)->route_0 = route_2;
    (&kernelContext_4)->isl_0 = isl_2;
    (&kernelContext_4)->islf_0 = islf_2;
    (&kernelContext_4)->islands_0 = islands_2;
    (&kernelContext_4)->ctab_0 = ctab_2;
    (&kernelContext_4)->bodies_0 = bodies_2;
    (&kernelContext_4)->chunks_0 = chunks_2;
    (&kernelContext_4)->state_0 = state_2;
    (&kernelContext_4)->imps_0 = imps_2;
    (&kernelContext_4)->status_0 = status_2;
    (&kernelContext_4)->largs_0 = largs_2;
    (&kernelContext_4)->cisl_0 = cisl_2;
    (&kernelContext_4)->bond_dyn_0 = bond_dyn_2;
    (&kernelContext_4)->bond_static_0 = bond_static_2;
    (&kernelContext_4)->frozen_0 = frozen_2;
    (&kernelContext_4)->split_work_0 = split_work_2;
    (&kernelContext_4)->split_control_0 = split_control_2;
    (&kernelContext_4)->split_parents_0 = split_parents_2;
    (&kernelContext_4)->split_slots_0 = split_slots_2;
    (&kernelContext_4)->split_out_0 = split_out_2;
    (&kernelContext_4)->split_firsts_0 = split_firsts_2;
    (&kernelContext_4)->lout_0 = lout_2;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_2;
    (&kernelContext_4)->g_red_0 = &g_red_2;
    threadgroup array<float, int(6)> g_lambda_2;
    (&kernelContext_4)->g_lambda_0 = &g_lambda_2;
    threadgroup array<float3, int(256)> g_lo_2;
    (&kernelContext_4)->g_lo_0 = &g_lo_2;
    threadgroup array<float3, int(256)> g_hi_2;
    (&kernelContext_4)->g_hi_0 = &g_hi_2;
    threadgroup array<uint, int(32)> g_label_2;
    (&kernelContext_4)->g_label_0 = &g_label_2;
    threadgroup uint g_used_2;
    (&kernelContext_4)->g_used_0 = &g_used_2;
    threadgroup array<uint, int(32)> g_call_of_2;
    (&kernelContext_4)->g_call_of_0 = &g_call_of_2;
    threadgroup array<uint, int(64)> g_mask_of_2;
    (&kernelContext_4)->g_mask_of_0 = &g_mask_of_2;
    threadgroup array<uint, int(64)> g_cnt_isl_2;
    (&kernelContext_4)->g_cnt_isl_0 = &g_cnt_isl_2;
    threadgroup array<uint, int(64)> g_cnt_chunk_2;
    (&kernelContext_4)->g_cnt_chunk_0 = &g_cnt_chunk_2;
    threadgroup array<uint, int(64)> g_cnt_bond_2;
    (&kernelContext_4)->g_cnt_bond_0 = &g_cnt_bond_2;
    threadgroup array<uint, int(64)> g_cnt_imp_2;
    (&kernelContext_4)->g_cnt_imp_0 = &g_cnt_imp_2;
    threadgroup array<uint, int(256)> g_scan_2;
    (&kernelContext_4)->g_scan_0 = &g_scan_2;
    threadgroup array<float, int(4096)> g_sort_2;
    (&kernelContext_4)->g_sort_0 = &g_sort_2;
    threadgroup uint g_stale_2;
    (&kernelContext_4)->g_stale_0 = &g_stale_2;
    threadgroup uint g_broke_2;
    (&kernelContext_4)->g_broke_0 = &g_broke_2;
    uint t_3 = thread_3.x;
    uint _S166 = *(ctl_2+2U);
    uint j_1 = t_3;
    for(;;)
    {
        if(j_1 < _S166)
        {
        }
        else
        {
            break;
        }
        uint _S167 = 14U * j_1;
        float4 _S168 = float4(*((&kernelContext_4)->nodes_0+(_S167 + 12U))) ;
        bool _S169;
        if((as_type<uint>(((float4(*((&kernelContext_4)->nodes_0+(_S167 + 4U))) ).w))) == 0U)
        {
            _S169 = (as_type<uint>((_S168.x))) < ((&kernelContext_4)->params_0->chunk_count_0);
        }
        else
        {
            _S169 = false;
        }
        if(_S169)
        {
            *((&kernelContext_4)->chunk_node_0+(as_type<uint>((_S168.x)))) = 0U;
        }
        j_1 = j_1 + 256U;
    }
    if(t_3 == 0U)
    {
        *((&kernelContext_4)->ctl_0+0U) = 0U;
        *((&kernelContext_4)->ctl_0+18U) = 0U;
    }
    return;
}

Quat_0 quat_mul_0(const Quat_0 thread* a_5, const Quat_0 thread* o_0)
{
    thread Quat_0 r_11;
    (&r_11)->w_0 = a_5->w_0 * o_0->w_0 - a_5->x_0 * o_0->x_0 - a_5->y_0 * o_0->y_0 - a_5->z_0 * o_0->z_0;
    (&r_11)->x_0 = a_5->w_0 * o_0->x_0 + a_5->x_0 * o_0->w_0 + a_5->y_0 * o_0->z_0 - a_5->z_0 * o_0->y_0;
    (&r_11)->y_0 = a_5->w_0 * o_0->y_0 - a_5->x_0 * o_0->z_0 + a_5->y_0 * o_0->w_0 + a_5->z_0 * o_0->x_0;
    (&r_11)->z_0 = a_5->w_0 * o_0->z_0 + a_5->x_0 * o_0->y_0 - a_5->y_0 * o_0->x_0 + a_5->z_0 * o_0->w_0;
    return r_11;
}

float4 quat_vec_0(const Quat_0 thread* q_7)
{
    return float4(q_7->x_0, q_7->y_0, q_7->z_0, q_7->w_0);
}

float4 qmul_0(float4 a_6, float4 b_5)
{
    Quat_0 _S170 = quat_of_0(b_5);
    thread Quat_0 _S171 = quat_of_0(a_6);
    thread Quat_0 _S172 = _S170;
    Quat_0 _S173 = quat_mul_0(&_S171, &_S172);
    thread Quat_0 _S174 = _S173;
    float4 _S175 = quat_vec_0(&_S174);
    return _S175;
}

float4 body_rotation_zup_0(float4 q_rubble_0, float4 p_inv_0)
{
    return normalize(qmul_0(qmul_0(float4(0.70710676908493042f, 0.0f, 0.0f, 0.70710676908493042f), q_rubble_0), p_inv_0));
}

float3 to_zup_0(float3 v_4)
{
    return float3(v_4.x, - v_4.z, v_4.y);
}

struct Frame_0
{
    float4 q_8;
    float3 origin_0;
    float3 com_1;
    float3 v_5;
    float3 w_4;
    uint body_0;
};

Frame_0 island_frame_0(uint i_0, KernelContext_0 thread* kernelContext_5)
{
    Island_natural_0 device* _S176 = kernelContext_5->islands_0+i_0;
    thread Frame_0 f_0;
    uint _S177 = 3U * i_0;
    uint _S178 = (as_type<uint>(((float4(*(kernelContext_5->ctab_0+(kernelContext_5->params_0->engine_base_0 + _S177))) ).x)));
    (&f_0)->body_0 = _S178;
    (&f_0)->q_8 = float4(_S176->rotation_0) ;
    (&f_0)->origin_0 = (float4(_S176->position_0) ).xyz;
    (&f_0)->v_5 = (float4(_S176->velocity_0) ).xyz;
    (&f_0)->w_4 = (float4(_S176->angular_velocity_0) ).xyz;
    if(_S178 != 4294967295U)
    {
        float3 com_2 = (float4(*(kernelContext_5->ctab_0+(kernelContext_5->params_0->engine_base_0 + _S177 + 2U))) ).xyz;
        float4 _S179 = body_rotation_zup_0(float4(*(kernelContext_5->bodies_0+(7U * (&f_0)->body_0 + 1U))) , float4(*(kernelContext_5->ctab_0+(kernelContext_5->params_0->engine_base_0 + _S177 + 1U))) );
        (&f_0)->q_8 = _S179;
        float3 _S180 = to_zup_0((float4(*(kernelContext_5->bodies_0+7U * (&f_0)->body_0)) ).xyz);
        (&f_0)->com_1 = _S180;
        thread Quat_0 _S181 = quat_of_0(_S179);
        float3 _S182 = rotate_0(&_S181, com_2);
        (&f_0)->origin_0 = _S180 - _S182;
        (&f_0)->v_5 = to_zup_0((float4(*(kernelContext_5->bodies_0+(7U * (&f_0)->body_0 + 3U))) ).xyz);
        (&f_0)->w_4 = to_zup_0((float4(*(kernelContext_5->bodies_0+(7U * (&f_0)->body_0 + 4U))) ).xyz);
    }
    else
    {
        float3 _S183 = (float4(_S176->com_0) ).xyz;
        thread Quat_0 _S184 = quat_of_0((&f_0)->q_8);
        float3 _S185 = rotate_0(&_S184, _S183);
        (&f_0)->com_1 = (&f_0)->origin_0 + _S185;
    }
    return f_0;
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

float3 chunk_world_0(const Frame_0 thread* f_1, uint k_14, KernelContext_0 thread* kernelContext_6)
{
    float3 _S186 = (float4((kernelContext_6->chunks_0+k_14)->center_0) ).xyz + (float4(*(kernelContext_6->state_0+4U * k_14)) ).xyz;
    thread Quat_0 _S187 = quat_of_0(f_1->q_8);
    float3 _S188 = rotate_0(&_S187, _S186);
    return f_1->origin_0 + _S188;
}

[[kernel]] void learned_select(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], uint device* ctl_3 [[buffer(1)]], packed_float4 device* calls_3 [[buffer(2)]], packed_float4 device* nodes_3 [[buffer(3)]], LearnedParams_0 constant* params_3 [[buffer(0)]], packed_float4 device* bonds_3 [[buffer(4)]], uint device* chunk_node_3 [[buffer(8)]], uint device* counters_3 [[buffer(19)]], uint device* route_3 [[buffer(25)]], packed_uint4 device* isl_3 [[buffer(5)]], packed_float4 device* islf_3 [[buffer(6)]], Island_natural_0 device* islands_3 [[buffer(11)]], packed_float4 device* ctab_3 [[buffer(14)]], packed_float4 device* bodies_3 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_3 [[buffer(12)]], packed_float4 device* state_3 [[buffer(13)]], packed_float4 device* imps_3 [[buffer(10)]], uint device* status_3 [[buffer(18)]], packed_uint4 device* largs_3 [[buffer(27)]], packed_uint4 device* cisl_3 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_3 [[buffer(17)]], BondStatic_natural_0 device* bond_static_3 [[buffer(16)]], packed_float4 device* frozen_3 [[buffer(9)]], uint device* split_work_3 [[buffer(21)]], uint device* split_control_3 [[buffer(23)]], packed_uint4 device* split_parents_3 [[buffer(20)]], packed_uint4 device* split_slots_3 [[buffer(22)]], packed_float4 device* split_out_3 [[buffer(26)]], packed_uint4 device* split_firsts_3 [[buffer(24)]], float device* lout_3 [[buffer(28)]])
{
    float t0_0;
    bool _S189;
    thread KernelContext_0 kernelContext_7;
    (&kernelContext_7)->ctl_0 = ctl_3;
    (&kernelContext_7)->calls_0 = calls_3;
    (&kernelContext_7)->nodes_0 = nodes_3;
    (&kernelContext_7)->params_0 = params_3;
    (&kernelContext_7)->bonds_0 = bonds_3;
    (&kernelContext_7)->chunk_node_0 = chunk_node_3;
    (&kernelContext_7)->counters_0 = counters_3;
    (&kernelContext_7)->route_0 = route_3;
    (&kernelContext_7)->isl_0 = isl_3;
    (&kernelContext_7)->islf_0 = islf_3;
    (&kernelContext_7)->islands_0 = islands_3;
    (&kernelContext_7)->ctab_0 = ctab_3;
    (&kernelContext_7)->bodies_0 = bodies_3;
    (&kernelContext_7)->chunks_0 = chunks_3;
    (&kernelContext_7)->state_0 = state_3;
    (&kernelContext_7)->imps_0 = imps_3;
    (&kernelContext_7)->status_0 = status_3;
    (&kernelContext_7)->largs_0 = largs_3;
    (&kernelContext_7)->cisl_0 = cisl_3;
    (&kernelContext_7)->bond_dyn_0 = bond_dyn_3;
    (&kernelContext_7)->bond_static_0 = bond_static_3;
    (&kernelContext_7)->frozen_0 = frozen_3;
    (&kernelContext_7)->split_work_0 = split_work_3;
    (&kernelContext_7)->split_control_0 = split_control_3;
    (&kernelContext_7)->split_parents_0 = split_parents_3;
    (&kernelContext_7)->split_slots_0 = split_slots_3;
    (&kernelContext_7)->split_out_0 = split_out_3;
    (&kernelContext_7)->split_firsts_0 = split_firsts_3;
    (&kernelContext_7)->lout_0 = lout_3;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_3;
    (&kernelContext_7)->g_red_0 = &g_red_3;
    threadgroup array<float, int(6)> g_lambda_3;
    (&kernelContext_7)->g_lambda_0 = &g_lambda_3;
    threadgroup array<float3, int(256)> g_lo_3;
    (&kernelContext_7)->g_lo_0 = &g_lo_3;
    threadgroup array<float3, int(256)> g_hi_3;
    (&kernelContext_7)->g_hi_0 = &g_hi_3;
    threadgroup array<uint, int(32)> g_label_3;
    (&kernelContext_7)->g_label_0 = &g_label_3;
    threadgroup uint g_used_3;
    (&kernelContext_7)->g_used_0 = &g_used_3;
    threadgroup array<uint, int(32)> g_call_of_3;
    (&kernelContext_7)->g_call_of_0 = &g_call_of_3;
    threadgroup array<uint, int(64)> g_mask_of_3;
    (&kernelContext_7)->g_mask_of_0 = &g_mask_of_3;
    threadgroup array<uint, int(64)> g_cnt_isl_3;
    (&kernelContext_7)->g_cnt_isl_0 = &g_cnt_isl_3;
    threadgroup array<uint, int(64)> g_cnt_chunk_3;
    (&kernelContext_7)->g_cnt_chunk_0 = &g_cnt_chunk_3;
    threadgroup array<uint, int(64)> g_cnt_bond_3;
    (&kernelContext_7)->g_cnt_bond_0 = &g_cnt_bond_3;
    threadgroup array<uint, int(64)> g_cnt_imp_3;
    (&kernelContext_7)->g_cnt_imp_0 = &g_cnt_imp_3;
    threadgroup array<uint, int(256)> g_scan_3;
    (&kernelContext_7)->g_scan_0 = &g_scan_3;
    threadgroup array<float, int(4096)> g_sort_3;
    (&kernelContext_7)->g_sort_0 = &g_sort_3;
    threadgroup uint g_stale_3;
    (&kernelContext_7)->g_stale_0 = &g_stale_3;
    threadgroup uint g_broke_3;
    (&kernelContext_7)->g_broke_0 = &g_broke_3;
    uint i_1 = group_1.x;
    uint t_4 = thread_4.x;
    if(i_1 >= (group_uniform_0(counters_3[int(0)], t_4)))
    {
        return;
    }
    if((group_uniform_0((&kernelContext_7)->route_0[int(0)], t_4)) == 0U)
    {
        if(t_4 == 0U)
        {
            *((&kernelContext_7)->isl_0+i_1) = packed_uint4(uint4(0U, 4294967295U, 0U, 0U)) ;
            *((&kernelContext_7)->islf_0+(3U * i_1 + 2U)) = packed_float4(float4(0.0f) ) ;
        }
        return;
    }
    Island_natural_0 device* _S190 = (&kernelContext_7)->islands_0+i_1;
    uint4 _S191 = uint4(_S190->range_0) ;
    uint _S192 = _S191.x;
    uint _S193 = group_uniform_0(_S191.y - _S192, t_4);
    Frame_0 _S194 = island_frame_0(i_1, &kernelContext_7);
    float inf_0 = float_bits_0(2139095040U);
    float3 _S195 = float3(- inf_0) ;
    float3 lo_0 = float3(inf_0) ;
    float3 hi_0 = _S195;
    uint c_3 = t_4;
    for(;;)
    {
        if(c_3 < _S193)
        {
        }
        else
        {
            break;
        }
        uint k_15 = _S192 + c_3;
        thread Frame_0 _S196 = _S194;
        float3 _S197 = chunk_world_0(&_S196, k_15, &kernelContext_7);
        float3 _S198 = min(lo_0, _S197 - float3((float4(((&kernelContext_7)->chunks_0+k_15)->half_0) ).w) );
        float3 _S199 = max(hi_0, _S197 + float3((float4(((&kernelContext_7)->chunks_0+k_15)->half_0) ).w) );
        uint c_4 = c_3 + 256U;
        lo_0 = _S198;
        hi_0 = _S199;
        c_3 = c_4;
    }
    (*(&kernelContext_7)->g_lo_0)[t_4] = lo_0;
    (*(&kernelContext_7)->g_hi_0)[t_4] = hi_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s_1 = 128U;
    for(;;)
    {
        if(s_1 > 0U)
        {
        }
        else
        {
            break;
        }
        if(t_4 < s_1)
        {
            uint _S200 = t_4 + s_1;
            (*(&kernelContext_7)->g_lo_0)[t_4] = min((*(&kernelContext_7)->g_lo_0)[t_4], (*(&kernelContext_7)->g_lo_0)[_S200]);
            (*(&kernelContext_7)->g_hi_0)[t_4] = max((*(&kernelContext_7)->g_hi_0)[t_4], (*(&kernelContext_7)->g_hi_0)[_S200]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_1 = s_1 >> 1U;
    }
    if(t_4 != 0U)
    {
        return;
    }
    float3 lo_1 = (*(&kernelContext_7)->g_lo_0)[int(0)];
    float3 hi_1 = (*(&kernelContext_7)->g_hi_0)[int(0)];
    float3 _S201 = float3(0.0f) ;
    bool launched_0 = false;
    float3 launch_0 = _S201;
    uint k_16 = 0U;
    uint mask_0 = 0U;
    for(;;)
    {
        if(k_16 < (min((&kernelContext_7)->params_0->impactor_count_0, 32U)))
        {
        }
        else
        {
            break;
        }
        float4 _S202 = float4(*((&kernelContext_7)->imps_0+3U * k_16)) ;
        uint _S203 = 7U * (as_type<uint>((_S202.x)));
        float3 _S204 = to_zup_0((float4(*((&kernelContext_7)->bodies_0+_S203)) ).xyz);
        float3 rel_0 = to_zup_0((float4(*((&kernelContext_7)->bodies_0+(_S203 + 3U))) ).xyz) - _S194.v_5;
        float r_12 = _S202.y;
        bool _S205;
        if(!((length(rel_0)) > ((&kernelContext_7)->params_0->speed_0)))
        {
            _S205 = true;
        }
        else
        {
            _S205 = _S193 == 0U;
        }
        float3 launch_1;
        bool launched_1;
        if(_S205)
        {
            launched_1 = launched_0;
            launch_1 = launch_0;
            uint _S206 = k_16 + 1U;
            launched_0 = launched_1;
            launch_0 = launch_1;
            k_16 = _S206;
            continue;
        }
        float3 _S207 = float3(r_12) ;
        float3 _S208 = lo_1 - _S207;
        float3 _S209 = hi_1 + _S207;
        float _S210 = (&kernelContext_7)->params_0->dt_0;
        t0_0 = 0.0f;
        float t1_1 = _S210;
        uint ax_0 = 0U;
        for(;;)
        {
            if(ax_0 < 3U)
            {
            }
            else
            {
                break;
            }
            uint _S211 = ax_0;
            if((abs(rel_0[ax_0])) < 9.999999960041972e-13f)
            {
                uint _S212 = ax_0;
                if((_S204[ax_0]) < (_S208[ax_0]))
                {
                    _S189 = true;
                }
                else
                {
                    _S189 = (_S204[_S212]) > (_S209[ax_0]);
                }
                float t0_1;
                if(_S189)
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
            float ta_2 = (_S208[ax_0] - _S204[ax_0]) / rel_0[_S211];
            float tb_2 = (_S209[ax_0] - _S204[ax_0]) / rel_0[_S211];
            float _S213 = min(t1_1, max(ta_2, tb_2));
            t0_0 = max(t0_0, min(ta_2, tb_2));
            t1_1 = _S213;
            ax_0 = ax_0 + 1U;
        }
        uint mask_1;
        if(t0_0 <= t1_1)
        {
            uint mask_2 = mask_0 | (1U << k_16);
            if(!launched_0)
            {
                _S189 = (((uint4(_S190->info_0) ).x) & 1U) == 0U;
            }
            else
            {
                _S189 = false;
            }
            if(_S189)
            {
                float3 _S214 = float3(2.0f)  * rel_0 * float3((&kernelContext_7)->params_0->dt_0) ;
                launched_1 = true;
                launch_1 = _S214;
            }
            else
            {
                launched_1 = launched_0;
                launch_1 = launch_0;
            }
            mask_1 = mask_2;
        }
        else
        {
            launched_1 = launched_0;
            launch_1 = launch_0;
            mask_1 = mask_0;
        }
        mask_0 = mask_1;
        uint _S206 = k_16 + 1U;
        launched_0 = launched_1;
        launch_0 = launch_1;
        k_16 = _S206;
    }
    *((&kernelContext_7)->isl_0+i_1) = packed_uint4(uint4(mask_0, 4294967295U, 0U, 0U)) ;
    uint _S215 = 3U * i_1;
    *((&kernelContext_7)->islf_0+_S215) = packed_float4(float4(lo_1, 0.0f)) ;
    *((&kernelContext_7)->islf_0+(_S215 + 1U)) = packed_float4(float4(hi_1, 0.0f)) ;
    packed_float4 device* _S216 = (&kernelContext_7)->islf_0+(_S215 + 2U);
    if(launched_0)
    {
        t0_0 = 1.0f;
    }
    else
    {
        t0_0 = 0.0f;
    }
    *_S216 = packed_float4(float4(launch_0, t0_0)) ;
    return;
}

[[kernel]] void learned_debris(uint3 id_0 [[thread_position_in_grid]], uint device* ctl_4 [[buffer(1)]], packed_float4 device* calls_4 [[buffer(2)]], packed_float4 device* nodes_4 [[buffer(3)]], LearnedParams_0 constant* params_4 [[buffer(0)]], packed_float4 device* bonds_4 [[buffer(4)]], uint device* chunk_node_4 [[buffer(8)]], uint device* counters_4 [[buffer(19)]], uint device* route_4 [[buffer(25)]], packed_uint4 device* isl_4 [[buffer(5)]], packed_float4 device* islf_4 [[buffer(6)]], Island_natural_0 device* islands_4 [[buffer(11)]], packed_float4 device* ctab_4 [[buffer(14)]], packed_float4 device* bodies_4 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_4 [[buffer(12)]], packed_float4 device* state_4 [[buffer(13)]], packed_float4 device* imps_4 [[buffer(10)]], uint device* status_4 [[buffer(18)]], packed_uint4 device* largs_4 [[buffer(27)]], packed_uint4 device* cisl_4 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_4 [[buffer(17)]], BondStatic_natural_0 device* bond_static_4 [[buffer(16)]], packed_float4 device* frozen_4 [[buffer(9)]], uint device* split_work_4 [[buffer(21)]], uint device* split_control_4 [[buffer(23)]], packed_uint4 device* split_parents_4 [[buffer(20)]], packed_uint4 device* split_slots_4 [[buffer(22)]], packed_float4 device* split_out_4 [[buffer(26)]], packed_uint4 device* split_firsts_4 [[buffer(24)]], float device* lout_4 [[buffer(28)]])
{
    thread KernelContext_0 kernelContext_8;
    (&kernelContext_8)->ctl_0 = ctl_4;
    (&kernelContext_8)->calls_0 = calls_4;
    (&kernelContext_8)->nodes_0 = nodes_4;
    (&kernelContext_8)->params_0 = params_4;
    (&kernelContext_8)->bonds_0 = bonds_4;
    (&kernelContext_8)->chunk_node_0 = chunk_node_4;
    (&kernelContext_8)->counters_0 = counters_4;
    (&kernelContext_8)->route_0 = route_4;
    (&kernelContext_8)->isl_0 = isl_4;
    (&kernelContext_8)->islf_0 = islf_4;
    (&kernelContext_8)->islands_0 = islands_4;
    (&kernelContext_8)->ctab_0 = ctab_4;
    (&kernelContext_8)->bodies_0 = bodies_4;
    (&kernelContext_8)->chunks_0 = chunks_4;
    (&kernelContext_8)->state_0 = state_4;
    (&kernelContext_8)->imps_0 = imps_4;
    (&kernelContext_8)->status_0 = status_4;
    (&kernelContext_8)->largs_0 = largs_4;
    (&kernelContext_8)->cisl_0 = cisl_4;
    (&kernelContext_8)->bond_dyn_0 = bond_dyn_4;
    (&kernelContext_8)->bond_static_0 = bond_static_4;
    (&kernelContext_8)->frozen_0 = frozen_4;
    (&kernelContext_8)->split_work_0 = split_work_4;
    (&kernelContext_8)->split_control_0 = split_control_4;
    (&kernelContext_8)->split_parents_0 = split_parents_4;
    (&kernelContext_8)->split_slots_0 = split_slots_4;
    (&kernelContext_8)->split_out_0 = split_out_4;
    (&kernelContext_8)->split_firsts_0 = split_firsts_4;
    (&kernelContext_8)->lout_0 = lout_4;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_4;
    (&kernelContext_8)->g_red_0 = &g_red_4;
    threadgroup array<float, int(6)> g_lambda_4;
    (&kernelContext_8)->g_lambda_0 = &g_lambda_4;
    threadgroup array<float3, int(256)> g_lo_4;
    (&kernelContext_8)->g_lo_0 = &g_lo_4;
    threadgroup array<float3, int(256)> g_hi_4;
    (&kernelContext_8)->g_hi_0 = &g_hi_4;
    threadgroup array<uint, int(32)> g_label_4;
    (&kernelContext_8)->g_label_0 = &g_label_4;
    threadgroup uint g_used_4;
    (&kernelContext_8)->g_used_0 = &g_used_4;
    threadgroup array<uint, int(32)> g_call_of_4;
    (&kernelContext_8)->g_call_of_0 = &g_call_of_4;
    threadgroup array<uint, int(64)> g_mask_of_4;
    (&kernelContext_8)->g_mask_of_0 = &g_mask_of_4;
    threadgroup array<uint, int(64)> g_cnt_isl_4;
    (&kernelContext_8)->g_cnt_isl_0 = &g_cnt_isl_4;
    threadgroup array<uint, int(64)> g_cnt_chunk_4;
    (&kernelContext_8)->g_cnt_chunk_0 = &g_cnt_chunk_4;
    threadgroup array<uint, int(64)> g_cnt_bond_4;
    (&kernelContext_8)->g_cnt_bond_0 = &g_cnt_bond_4;
    threadgroup array<uint, int(64)> g_cnt_imp_4;
    (&kernelContext_8)->g_cnt_imp_0 = &g_cnt_imp_4;
    threadgroup array<uint, int(256)> g_scan_4;
    (&kernelContext_8)->g_scan_0 = &g_scan_4;
    threadgroup array<float, int(4096)> g_sort_4;
    (&kernelContext_8)->g_sort_0 = &g_sort_4;
    threadgroup uint g_stale_4;
    (&kernelContext_8)->g_stale_0 = &g_stale_4;
    threadgroup uint g_broke_4;
    (&kernelContext_8)->g_broke_0 = &g_broke_4;
    uint j_2 = id_0.x;
    uint live_0 = counters_4[int(0)];
    bool _S217;
    if((params_4->debris_0) == 0U)
    {
        _S217 = true;
    }
    else
    {
        _S217 = j_2 >= live_0;
    }
    if(_S217)
    {
        _S217 = true;
    }
    else
    {
        _S217 = (&kernelContext_8)->route_0[int(0)] == 0U;
    }
    if(_S217)
    {
        _S217 = true;
    }
    else
    {
        _S217 = ((uint4(*((&kernelContext_8)->isl_0+j_2)) ).x) != 0U;
    }
    if(_S217)
    {
        return;
    }
    uint _S218 = 3U * j_2;
    float3 lo_2 = (float4(*((&kernelContext_8)->islf_0+_S218)) ).xyz;
    float3 hi_2 = (float4(*((&kernelContext_8)->islf_0+(_S218 + 1U))) ).xyz;
    if(!((lo_2.x) <= (hi_2.x)))
    {
        return;
    }
    uint i_2 = 0U;
    uint mask_3 = 0U;
    for(;;)
    {
        if(i_2 < live_0)
        {
        }
        else
        {
            break;
        }
        uint _S219 = 3U * i_2;
        float4 _S220 = float4(*((&kernelContext_8)->islf_0+(_S219 + 2U))) ;
        if((_S220.w) == 0.0f)
        {
            _S217 = true;
        }
        else
        {
            _S217 = i_2 == j_2;
        }
        if(_S217)
        {
            i_2 = i_2 + 1U;
            continue;
        }
        float3 a_7 = (float4(*((&kernelContext_8)->islf_0+_S219)) ).xyz;
        float3 b_6 = (float4(*((&kernelContext_8)->islf_0+(_S219 + 1U))) ).xyz;
        float3 _S221 = _S220.xyz;
        float3 _S222 = float3(0.00100000004749745f) ;
        float3 shi_0 = max(b_6, b_6 + _S221) + _S222;
        bool _S223;
        if(all((min(a_7, a_7 + _S221) - _S222) <= hi_2))
        {
            _S223 = all(lo_2 <= shi_0);
        }
        else
        {
            _S223 = false;
        }
        uint mask_4;
        if(_S223)
        {
            mask_4 = mask_3 | ((uint4(*((&kernelContext_8)->isl_0+i_2)) ).x);
        }
        else
        {
            mask_4 = mask_3;
        }
        mask_3 = mask_4;
        i_2 = i_2 + 1U;
    }
    if(mask_3 != 0U)
    {
        *((&kernelContext_8)->isl_0+j_2) = packed_uint4(uint4(mask_3, 4294967295U, 0U, 0U)) ;
    }
    return;
}

uint lowest_bit_0(uint m_6)
{
    return firstbitlow_0(m_6);
}

[[kernel]] void learned_compact(uint3 thread_5 [[thread_position_in_threadgroup]], uint device* ctl_5 [[buffer(1)]], packed_float4 device* calls_5 [[buffer(2)]], packed_float4 device* nodes_5 [[buffer(3)]], LearnedParams_0 constant* params_5 [[buffer(0)]], packed_float4 device* bonds_5 [[buffer(4)]], uint device* chunk_node_5 [[buffer(8)]], uint device* counters_5 [[buffer(19)]], uint device* route_5 [[buffer(25)]], packed_uint4 device* isl_5 [[buffer(5)]], packed_float4 device* islf_5 [[buffer(6)]], Island_natural_0 device* islands_5 [[buffer(11)]], packed_float4 device* ctab_5 [[buffer(14)]], packed_float4 device* bodies_5 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_5 [[buffer(12)]], packed_float4 device* state_5 [[buffer(13)]], packed_float4 device* imps_5 [[buffer(10)]], uint device* status_5 [[buffer(18)]], packed_uint4 device* largs_5 [[buffer(27)]], packed_uint4 device* cisl_5 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_5 [[buffer(17)]], BondStatic_natural_0 device* bond_static_5 [[buffer(16)]], packed_float4 device* frozen_5 [[buffer(9)]], uint device* split_work_5 [[buffer(21)]], uint device* split_control_5 [[buffer(23)]], packed_uint4 device* split_parents_5 [[buffer(20)]], packed_uint4 device* split_slots_5 [[buffer(22)]], packed_float4 device* split_out_5 [[buffer(26)]], packed_uint4 device* split_firsts_5 [[buffer(24)]], float device* lout_5 [[buffer(28)]])
{
    uint _S224;
    thread KernelContext_0 kernelContext_9;
    (&kernelContext_9)->ctl_0 = ctl_5;
    (&kernelContext_9)->calls_0 = calls_5;
    (&kernelContext_9)->nodes_0 = nodes_5;
    (&kernelContext_9)->params_0 = params_5;
    (&kernelContext_9)->bonds_0 = bonds_5;
    (&kernelContext_9)->chunk_node_0 = chunk_node_5;
    (&kernelContext_9)->counters_0 = counters_5;
    (&kernelContext_9)->route_0 = route_5;
    (&kernelContext_9)->isl_0 = isl_5;
    (&kernelContext_9)->islf_0 = islf_5;
    (&kernelContext_9)->islands_0 = islands_5;
    (&kernelContext_9)->ctab_0 = ctab_5;
    (&kernelContext_9)->bodies_0 = bodies_5;
    (&kernelContext_9)->chunks_0 = chunks_5;
    (&kernelContext_9)->state_0 = state_5;
    (&kernelContext_9)->imps_0 = imps_5;
    (&kernelContext_9)->status_0 = status_5;
    (&kernelContext_9)->largs_0 = largs_5;
    (&kernelContext_9)->cisl_0 = cisl_5;
    (&kernelContext_9)->bond_dyn_0 = bond_dyn_5;
    (&kernelContext_9)->bond_static_0 = bond_static_5;
    (&kernelContext_9)->frozen_0 = frozen_5;
    (&kernelContext_9)->split_work_0 = split_work_5;
    (&kernelContext_9)->split_control_0 = split_control_5;
    (&kernelContext_9)->split_parents_0 = split_parents_5;
    (&kernelContext_9)->split_slots_0 = split_slots_5;
    (&kernelContext_9)->split_out_0 = split_out_5;
    (&kernelContext_9)->split_firsts_0 = split_firsts_5;
    (&kernelContext_9)->lout_0 = lout_5;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_5;
    (&kernelContext_9)->g_red_0 = &g_red_5;
    threadgroup array<float, int(6)> g_lambda_5;
    (&kernelContext_9)->g_lambda_0 = &g_lambda_5;
    threadgroup array<float3, int(256)> g_lo_5;
    (&kernelContext_9)->g_lo_0 = &g_lo_5;
    threadgroup array<float3, int(256)> g_hi_5;
    (&kernelContext_9)->g_hi_0 = &g_hi_5;
    threadgroup array<uint, int(32)> g_label_5;
    (&kernelContext_9)->g_label_0 = &g_label_5;
    threadgroup uint g_used_5;
    (&kernelContext_9)->g_used_0 = &g_used_5;
    threadgroup array<uint, int(32)> g_call_of_5;
    (&kernelContext_9)->g_call_of_0 = &g_call_of_5;
    threadgroup array<uint, int(64)> g_mask_of_5;
    (&kernelContext_9)->g_mask_of_0 = &g_mask_of_5;
    threadgroup array<uint, int(64)> g_cnt_isl_5;
    (&kernelContext_9)->g_cnt_isl_0 = &g_cnt_isl_5;
    threadgroup array<uint, int(64)> g_cnt_chunk_5;
    (&kernelContext_9)->g_cnt_chunk_0 = &g_cnt_chunk_5;
    threadgroup array<uint, int(64)> g_cnt_bond_5;
    (&kernelContext_9)->g_cnt_bond_0 = &g_cnt_bond_5;
    threadgroup array<uint, int(64)> g_cnt_imp_5;
    (&kernelContext_9)->g_cnt_imp_0 = &g_cnt_imp_5;
    threadgroup array<uint, int(256)> g_scan_5;
    (&kernelContext_9)->g_scan_0 = &g_scan_5;
    threadgroup array<float, int(4096)> g_sort_5;
    (&kernelContext_9)->g_sort_0 = &g_sort_5;
    threadgroup uint g_stale_5;
    (&kernelContext_9)->g_stale_0 = &g_stale_5;
    threadgroup uint g_broke_5;
    (&kernelContext_9)->g_broke_0 = &g_broke_5;
    uint t_5 = thread_5.x;
    uint _S225 = group_uniform_0(counters_5[int(0)], t_5);
    uint _S226 = min(params_5->impactor_count_0, 32U);
    uint i_3 = t_5;
    for(;;)
    {
        if(i_3 < _S225)
        {
        }
        else
        {
            break;
        }
        uint device* _S227 = (&kernelContext_9)->status_0+(16U + i_3);
        *_S227 = (*_S227) & 4294963199U;
        i_3 = i_3 + 1024U;
    }
    if(t_5 < 32U)
    {
        (*(&kernelContext_9)->g_label_0)[t_5] = t_5;
    }
    bool _S228 = t_5 == 0U;
    if(_S228)
    {
        *(&kernelContext_9)->g_used_0 = 0U;
        *((&kernelContext_9)->ctl_0+1U) = 0U;
        *((&kernelContext_9)->ctl_0+2U) = 0U;
        *((&kernelContext_9)->ctl_0+3U) = 0U;
        *((&kernelContext_9)->ctl_0+9U) = 0U;
        *((&kernelContext_9)->ctl_0+8U) = 0U;
        *((&kernelContext_9)->ctl_0+5U) = 0U;
        *((&kernelContext_9)->ctl_0+6U) = 0U;
        *((&kernelContext_9)->ctl_0+7U) = 0U;
        *((&kernelContext_9)->ctl_0+14U) = 0U;
        *((&kernelContext_9)->ctl_0+15U) = 0U;
        *((&kernelContext_9)->ctl_0+16U) = 0U;
        *((&kernelContext_9)->ctl_0+17U) = 0U;
    }
    if(_S228)
    {
        packed_uint4 _S229 = packed_uint4(uint4(0U, 1U, 1U, 0U)) ;
        *((&kernelContext_9)->largs_0+int(0)) = _S229;
        *((&kernelContext_9)->largs_0+int(1)) = _S229;
        *((&kernelContext_9)->largs_0+int(2)) = _S229;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if((group_uniform_0((&kernelContext_9)->route_0[int(0)], t_5)) == 0U)
    {
        return;
    }
    uint over_0;
    uint routed_0;
    uint g_1;
    uint node_at_0;
    uint bond_at_0;
    bool changed_0;
    if(_S228)
    {
        i_3 = 0U;
        over_0 = 0U;
        for(;;)
        {
            if(i_3 < _S225)
            {
            }
            else
            {
                break;
            }
            uint used_0 = over_0 | ((uint4(*((&kernelContext_9)->isl_0+i_3)) ).x);
            i_3 = i_3 + 1U;
            over_0 = used_0;
        }
        *(&kernelContext_9)->g_used_0 = over_0;
        routed_0 = 0U;
        for(;;)
        {
            if(routed_0 < 32U)
            {
            }
            else
            {
                break;
            }
            changed_0 = false;
            i_3 = 0U;
            for(;;)
            {
                if(i_3 < _S225)
                {
                }
                else
                {
                    break;
                }
                uint m_7 = (uint4(*((&kernelContext_9)->isl_0+i_3)) ).x;
                bool changed_1;
                if(((uint)popcount((m_7))) < 2U)
                {
                    changed_1 = changed_0;
                    uint _S230 = i_3 + 1U;
                    changed_0 = changed_1;
                    i_3 = _S230;
                    continue;
                }
                g_1 = 32U;
                node_at_0 = m_7;
                for(;;)
                {
                    if(node_at_0 != 0U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    uint _S231 = min(g_1, (*(&kernelContext_9)->g_label_0)[lowest_bit_0(node_at_0)]);
                    uint mm_0 = node_at_0 & (node_at_0 - 1U);
                    g_1 = _S231;
                    node_at_0 = mm_0;
                }
                changed_1 = changed_0;
                bond_at_0 = m_7;
                for(;;)
                {
                    if(bond_at_0 != 0U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    if(((*(&kernelContext_9)->g_label_0)[lowest_bit_0(bond_at_0)]) > g_1)
                    {
                        (*(&kernelContext_9)->g_label_0)[lowest_bit_0(bond_at_0)] = g_1;
                        changed_1 = true;
                    }
                    bond_at_0 = bond_at_0 & (bond_at_0 - 1U);
                }
                uint _S230 = i_3 + 1U;
                changed_0 = changed_1;
                i_3 = _S230;
            }
            if(!changed_0)
            {
                break;
            }
            routed_0 = routed_0 + 1U;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S228)
    {
        uint _S232 = *(&kernelContext_9)->g_used_0;
        over_0 = 0U;
        routed_0 = 0U;
        for(;;)
        {
            if(over_0 < _S226)
            {
            }
            else
            {
                break;
            }
            (*(&kernelContext_9)->g_call_of_0)[over_0] = 4294967295U;
            if(((*(&kernelContext_9)->g_label_0)[over_0]) != over_0)
            {
                over_0 = over_0 + 1U;
                continue;
            }
            g_1 = 0U;
            node_at_0 = 0U;
            for(;;)
            {
                if(g_1 < _S226)
                {
                }
                else
                {
                    break;
                }
                if(((*(&kernelContext_9)->g_label_0)[g_1]) == over_0)
                {
                    node_at_0 = node_at_0 | (1U << g_1);
                }
                g_1 = g_1 + 1U;
            }
            uint _S233 = node_at_0 & _S232;
            if(_S233 == 0U)
            {
                over_0 = over_0 + 1U;
                continue;
            }
            if(routed_0 < 64U)
            {
                (*(&kernelContext_9)->g_mask_of_0)[routed_0] = _S233;
                (*(&kernelContext_9)->g_cnt_isl_0)[routed_0] = 0U;
                (*(&kernelContext_9)->g_cnt_chunk_0)[routed_0] = 0U;
                (*(&kernelContext_9)->g_cnt_bond_0)[routed_0] = 0U;
                (*(&kernelContext_9)->g_cnt_imp_0)[routed_0] = ((uint)popcount((_S233)));
            }
            (*(&kernelContext_9)->g_call_of_0)[over_0] = routed_0;
            routed_0 = routed_0 + 1U;
            over_0 = over_0 + 1U;
        }
        *((&kernelContext_9)->ctl_0+1U) = routed_0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint ncalls_0 = group_uniform_0(*((&kernelContext_9)->ctl_0+1U), t_5);
    if(_S228)
    {
        i_3 = 0U;
        for(;;)
        {
            if(i_3 < _S225)
            {
            }
            else
            {
                break;
            }
            uint4 _S234 = uint4(*((&kernelContext_9)->isl_0+i_3)) ;
            uint _S235 = _S234.x;
            if(_S235 != 0U)
            {
                *((&kernelContext_9)->isl_0+i_3) = packed_uint4(uint4(_S235, (*(&kernelContext_9)->g_call_of_0)[(*(&kernelContext_9)->g_label_0)[lowest_bit_0(_S235)]], _S234.z, 0U)) ;
            }
            i_3 = i_3 + 1U;
        }
    }
    if(t_5 != 0U)
    {
        return;
    }
    if(ncalls_0 > (min((&kernelContext_9)->params_0->max_calls_0, 64U)))
    {
        over_0 = 1U;
    }
    else
    {
        over_0 = 0U;
    }
    i_3 = 0U;
    routed_0 = 0U;
    for(;;)
    {
        if(i_3 < _S225)
        {
        }
        else
        {
            break;
        }
        uint4 _S236 = uint4(*((&kernelContext_9)->isl_0+i_3)) ;
        if((_S236.x) == 0U)
        {
            changed_0 = true;
        }
        else
        {
            changed_0 = (_S236.y) >= 64U;
        }
        if(changed_0)
        {
            i_3 = i_3 + 1U;
            continue;
        }
        Island_natural_0 device* _S237 = (&kernelContext_9)->islands_0+i_3;
        uint _S238 = _S236.y;
        (*(&kernelContext_9)->g_cnt_isl_0)[_S238] = (*(&kernelContext_9)->g_cnt_isl_0)[_S238] + 1U;
        uint4 _S239 = uint4(_S237->range_0) ;
        (*(&kernelContext_9)->g_cnt_chunk_0)[_S238] = (*(&kernelContext_9)->g_cnt_chunk_0)[_S238] + (_S239.y - _S239.x);
        (*(&kernelContext_9)->g_cnt_bond_0)[_S238] = (*(&kernelContext_9)->g_cnt_bond_0)[_S238] + _S236.z;
        routed_0 = routed_0 + 1U;
        i_3 = i_3 + 1U;
    }
    g_1 = 0U;
    node_at_0 = 0U;
    for(;;)
    {
        uint _S240 = min(ncalls_0, 64U);
        _S224 = _S240;
        if(g_1 < _S240)
        {
        }
        else
        {
            break;
        }
        uint chunks_n_0 = (*(&kernelContext_9)->g_cnt_chunk_0)[g_1];
        uint nodes_n_0 = (*(&kernelContext_9)->g_cnt_chunk_0)[g_1] + (*(&kernelContext_9)->g_cnt_imp_0)[g_1];
        if(((*(&kernelContext_9)->g_cnt_chunk_0)[g_1]) > ((&kernelContext_9)->params_0->max_call_chunks_0))
        {
            over_0 = over_0 | 16U;
        }
        uint _S241 = 8U * g_1;
        *((&kernelContext_9)->calls_0+(_S241 + 2U)) = packed_float4(float4(0.0f, 0.0f, (as_type<float>((node_at_0))), (as_type<float>((nodes_n_0))))) ;
        *((&kernelContext_9)->calls_0+(_S241 + 3U)) = packed_float4(float4((&kernelContext_9)->params_0->gravity_0.xyz, (as_type<float>((chunks_n_0))))) ;
        float _S242 = (as_type<float>((0U)));
        *((&kernelContext_9)->calls_0+(_S241 + 4U)) = packed_float4(float4(_S242, (as_type<float>(((*(&kernelContext_9)->g_cnt_bond_0)[g_1]))), 0.0f, 0.0f)) ;
        *((&kernelContext_9)->calls_0+(_S241 + 5U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, (as_type<float>(((*(&kernelContext_9)->g_mask_of_0)[g_1]))))) ;
        *((&kernelContext_9)->calls_0+(_S241 + 6U)) = packed_float4(float4(_S242, (as_type<float>(((*(&kernelContext_9)->g_cnt_isl_0)[g_1]))), 0.0f, 0.0f)) ;
        (*(&kernelContext_9)->g_cnt_chunk_0)[g_1] = node_at_0;
        (*(&kernelContext_9)->g_cnt_bond_0)[g_1] = 0U;
        (*(&kernelContext_9)->g_cnt_isl_0)[g_1] = 0U;
        uint node_at_1 = node_at_0 + nodes_n_0;
        g_1 = g_1 + 1U;
        node_at_0 = node_at_1;
    }
    g_1 = 0U;
    bond_at_0 = 0U;
    uint cisl_at_0 = 0U;
    uint chunk_base_0 = 0U;
    uint imp_base_0 = 0U;
    for(;;)
    {
        if(g_1 < _S224)
        {
        }
        else
        {
            break;
        }
        uint _S243 = 8U * g_1;
        uint _S244 = _S243 + 4U;
        (*((&kernelContext_9)->calls_0+_S244))[int(0)] = (as_type<float>((bond_at_0)));
        uint _S245 = _S243 + 6U;
        *((&kernelContext_9)->calls_0+_S245) = packed_float4(float4((as_type<float>((cisl_at_0))), (float4(*((&kernelContext_9)->calls_0+_S245)) ).y, (as_type<float>((chunk_base_0))), (as_type<float>((imp_base_0))))) ;
        uint chunk_base_1 = chunk_base_0 + (as_type<uint>(((float4(*((&kernelContext_9)->calls_0+(_S243 + 3U))) ).w)));
        uint imp_base_1 = imp_base_0 + ((uint)popcount(((as_type<uint>(((float4(*((&kernelContext_9)->calls_0+(_S243 + 5U))) ).w))))));
        (*(&kernelContext_9)->g_cnt_bond_0)[g_1] = bond_at_0;
        (*(&kernelContext_9)->g_cnt_isl_0)[g_1] = cisl_at_0;
        uint bond_at_1 = bond_at_0 + (as_type<uint>(((float4(*((&kernelContext_9)->calls_0+_S244)) ).y)));
        uint cisl_at_1 = cisl_at_0 + (as_type<uint>(((float4(*((&kernelContext_9)->calls_0+_S245)) ).y)));
        g_1 = g_1 + 1U;
        bond_at_0 = bond_at_1;
        cisl_at_0 = cisl_at_1;
        chunk_base_0 = chunk_base_1;
        imp_base_0 = imp_base_1;
    }
    if(node_at_0 > ((&kernelContext_9)->params_0->max_nodes_0))
    {
        over_0 = over_0 | 2U;
    }
    if(bond_at_0 > ((&kernelContext_9)->params_0->max_bonds_0))
    {
        over_0 = over_0 | 4U;
    }
    if(cisl_at_0 > ((&kernelContext_9)->params_0->max_cisl_0))
    {
        over_0 = over_0 | 8U;
    }
    if(over_0 != 0U)
    {
        *((&kernelContext_9)->ctl_0+8U) = over_0;
        *((&kernelContext_9)->ctl_0+1U) = 0U;
        return;
    }
    i_3 = 0U;
    for(;;)
    {
        if(i_3 < _S225)
        {
        }
        else
        {
            break;
        }
        uint4 _S246 = uint4(*((&kernelContext_9)->isl_0+i_3)) ;
        if((_S246.x) == 0U)
        {
            changed_0 = true;
        }
        else
        {
            changed_0 = (_S246.y) >= 64U;
        }
        if(changed_0)
        {
            i_3 = i_3 + 1U;
            continue;
        }
        Island_natural_0 device* _S247 = (&kernelContext_9)->islands_0+i_3;
        uint g_2 = _S246.y;
        uint at_0 = (*(&kernelContext_9)->g_cnt_isl_0)[g_2];
        uint4 _S248 = uint4(_S247->range_0) ;
        uint nchunks_0 = _S248.y - _S248.x;
        uint _S249 = 2U * (*(&kernelContext_9)->g_cnt_isl_0)[g_2];
        *((&kernelContext_9)->cisl_0+_S249) = packed_uint4(uint4(i_3, (*(&kernelContext_9)->g_cnt_chunk_0)[g_2], nchunks_0, g_2)) ;
        packed_uint4 device* _S250 = (&kernelContext_9)->cisl_0+(_S249 + 1U);
        uint _S251 = (as_type<uint>(((float4(*((&kernelContext_9)->ctab_0+((&kernelContext_9)->params_0->engine_base_0 + 3U * i_3))) ).x)));
        uint _S252 = (*(&kernelContext_9)->g_cnt_bond_0)[g_2];
        uint _S253 = _S246.z;
        if((((uint4(_S247->info_0) ).x) & 1U) != 0U)
        {
            over_0 = 1U;
        }
        else
        {
            over_0 = 0U;
        }
        *_S250 = packed_uint4(uint4(_S251, _S252, _S253, over_0)) ;
        (*(&kernelContext_9)->g_cnt_isl_0)[g_2] = at_0 + 1U;
        (*(&kernelContext_9)->g_cnt_chunk_0)[g_2] = (*(&kernelContext_9)->g_cnt_chunk_0)[g_2] + nchunks_0;
        (*(&kernelContext_9)->g_cnt_bond_0)[g_2] = (*(&kernelContext_9)->g_cnt_bond_0)[g_2] + _S253;
        i_3 = i_3 + 1U;
    }
    *((&kernelContext_9)->ctl_0+2U) = node_at_0;
    *((&kernelContext_9)->ctl_0+3U) = bond_at_0;
    *((&kernelContext_9)->ctl_0+9U) = cisl_at_0;
    *((&kernelContext_9)->ctl_0+17U) = routed_0;
    *((&kernelContext_9)->largs_0+int(0)) = packed_uint4(uint4(ncalls_0, 1U, 1U, 0U)) ;
    *((&kernelContext_9)->largs_0+int(1)) = packed_uint4(uint4(cisl_at_0, 1U, 1U, 0U)) ;
    *((&kernelContext_9)->largs_0+int(2)) = packed_uint4(uint4((cisl_at_0 + 63U) / 64U, 1U, 1U, 0U)) ;
    return;
}

bool connected_0(const JointState_0 thread* st_0, bool has_rebar_0)
{
    bool _S254;
    if((st_0->damage_0) < 1.0f)
    {
        _S254 = true;
    }
    else
    {
        if(has_rebar_0)
        {
            _S254 = (st_0->rebar_broken_0) == 0.0f;
        }
        else
        {
            _S254 = false;
        }
    }
    return _S254;
}

[[kernel]] void learned_count_bonds(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_6 [[thread_position_in_threadgroup]], uint device* ctl_6 [[buffer(1)]], packed_float4 device* calls_6 [[buffer(2)]], packed_float4 device* nodes_6 [[buffer(3)]], LearnedParams_0 constant* params_6 [[buffer(0)]], packed_float4 device* bonds_6 [[buffer(4)]], uint device* chunk_node_6 [[buffer(8)]], uint device* counters_6 [[buffer(19)]], uint device* route_6 [[buffer(25)]], packed_uint4 device* isl_6 [[buffer(5)]], packed_float4 device* islf_6 [[buffer(6)]], Island_natural_0 device* islands_6 [[buffer(11)]], packed_float4 device* ctab_6 [[buffer(14)]], packed_float4 device* bodies_6 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_6 [[buffer(12)]], packed_float4 device* state_6 [[buffer(13)]], packed_float4 device* imps_6 [[buffer(10)]], uint device* status_6 [[buffer(18)]], packed_uint4 device* largs_6 [[buffer(27)]], packed_uint4 device* cisl_6 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_6 [[buffer(17)]], BondStatic_natural_0 device* bond_static_6 [[buffer(16)]], packed_float4 device* frozen_6 [[buffer(9)]], uint device* split_work_6 [[buffer(21)]], uint device* split_control_6 [[buffer(23)]], packed_uint4 device* split_parents_6 [[buffer(20)]], packed_uint4 device* split_slots_6 [[buffer(22)]], packed_float4 device* split_out_6 [[buffer(26)]], packed_uint4 device* split_firsts_6 [[buffer(24)]], float device* lout_6 [[buffer(28)]])
{
    thread KernelContext_0 kernelContext_10;
    (&kernelContext_10)->ctl_0 = ctl_6;
    (&kernelContext_10)->calls_0 = calls_6;
    (&kernelContext_10)->nodes_0 = nodes_6;
    (&kernelContext_10)->params_0 = params_6;
    (&kernelContext_10)->bonds_0 = bonds_6;
    (&kernelContext_10)->chunk_node_0 = chunk_node_6;
    (&kernelContext_10)->counters_0 = counters_6;
    (&kernelContext_10)->route_0 = route_6;
    (&kernelContext_10)->isl_0 = isl_6;
    (&kernelContext_10)->islf_0 = islf_6;
    (&kernelContext_10)->islands_0 = islands_6;
    (&kernelContext_10)->ctab_0 = ctab_6;
    (&kernelContext_10)->bodies_0 = bodies_6;
    (&kernelContext_10)->chunks_0 = chunks_6;
    (&kernelContext_10)->state_0 = state_6;
    (&kernelContext_10)->imps_0 = imps_6;
    (&kernelContext_10)->status_0 = status_6;
    (&kernelContext_10)->largs_0 = largs_6;
    (&kernelContext_10)->cisl_0 = cisl_6;
    (&kernelContext_10)->bond_dyn_0 = bond_dyn_6;
    (&kernelContext_10)->bond_static_0 = bond_static_6;
    (&kernelContext_10)->frozen_0 = frozen_6;
    (&kernelContext_10)->split_work_0 = split_work_6;
    (&kernelContext_10)->split_control_0 = split_control_6;
    (&kernelContext_10)->split_parents_0 = split_parents_6;
    (&kernelContext_10)->split_slots_0 = split_slots_6;
    (&kernelContext_10)->split_out_0 = split_out_6;
    (&kernelContext_10)->split_firsts_0 = split_firsts_6;
    (&kernelContext_10)->lout_0 = lout_6;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_6;
    (&kernelContext_10)->g_red_0 = &g_red_6;
    threadgroup array<float, int(6)> g_lambda_6;
    (&kernelContext_10)->g_lambda_0 = &g_lambda_6;
    threadgroup array<float3, int(256)> g_lo_6;
    (&kernelContext_10)->g_lo_0 = &g_lo_6;
    threadgroup array<float3, int(256)> g_hi_6;
    (&kernelContext_10)->g_hi_0 = &g_hi_6;
    threadgroup array<uint, int(32)> g_label_6;
    (&kernelContext_10)->g_label_0 = &g_label_6;
    threadgroup uint g_used_6;
    (&kernelContext_10)->g_used_0 = &g_used_6;
    threadgroup array<uint, int(32)> g_call_of_6;
    (&kernelContext_10)->g_call_of_0 = &g_call_of_6;
    threadgroup array<uint, int(64)> g_mask_of_6;
    (&kernelContext_10)->g_mask_of_0 = &g_mask_of_6;
    threadgroup array<uint, int(64)> g_cnt_isl_6;
    (&kernelContext_10)->g_cnt_isl_0 = &g_cnt_isl_6;
    threadgroup array<uint, int(64)> g_cnt_chunk_6;
    (&kernelContext_10)->g_cnt_chunk_0 = &g_cnt_chunk_6;
    threadgroup array<uint, int(64)> g_cnt_bond_6;
    (&kernelContext_10)->g_cnt_bond_0 = &g_cnt_bond_6;
    threadgroup array<uint, int(64)> g_cnt_imp_6;
    (&kernelContext_10)->g_cnt_imp_0 = &g_cnt_imp_6;
    threadgroup array<uint, int(256)> g_scan_6;
    (&kernelContext_10)->g_scan_0 = &g_scan_6;
    threadgroup array<float, int(4096)> g_sort_6;
    (&kernelContext_10)->g_sort_0 = &g_sort_6;
    threadgroup uint g_stale_6;
    (&kernelContext_10)->g_stale_0 = &g_stale_6;
    threadgroup uint g_broke_6;
    (&kernelContext_10)->g_broke_0 = &g_broke_6;
    uint i_4 = group_2.x;
    uint t_6 = thread_6.x;
    if(i_4 >= (group_uniform_0(counters_6[int(0)], t_6)))
    {
        return;
    }
    uint4 _S255 = uint4(*((&kernelContext_10)->isl_0+i_4)) ;
    bool _S256;
    if((&kernelContext_10)->route_0[int(0)] == 0U)
    {
        _S256 = true;
    }
    else
    {
        _S256 = (_S255.x) == 0U;
    }
    uint bi_0;
    if(_S256)
    {
        bi_0 = 1U;
    }
    else
    {
        bi_0 = 0U;
    }
    if((group_uniform_0(bi_0, t_6)) != 0U)
    {
        return;
    }
    uint4 _S257 = uint4(((&kernelContext_10)->islands_0+i_4)->range_0) ;
    bi_0 = _S257.z + t_6;
    uint nb_0 = 0U;
    for(;;)
    {
        if(bi_0 < (_S257.w))
        {
        }
        else
        {
            break;
        }
        bool _S258 = ((float4(((&kernelContext_10)->bond_static_0+bi_0)->law_0.stiff1_0) ).w) != 0.0f;
        thread JointState_0 _S259 = ((&kernelContext_10)->bond_dyn_0+bi_0)->js_0;
        bool _S260 = connected_0(&_S259, _S258);
        if(_S260)
        {
            nb_0 = nb_0 + 1U;
        }
        bi_0 = bi_0 + 256U;
    }
    (*(&kernelContext_10)->g_scan_0)[t_6] = nb_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s_2 = 128U;
    for(;;)
    {
        if(s_2 > 0U)
        {
        }
        else
        {
            break;
        }
        if(t_6 < s_2)
        {
            (*(&kernelContext_10)->g_scan_0)[t_6] = (*(&kernelContext_10)->g_scan_0)[t_6] + (*(&kernelContext_10)->g_scan_0)[t_6 + s_2];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_2 = s_2 >> 1U;
    }
    if(t_6 == 0U)
    {
        *((&kernelContext_10)->isl_0+i_4) = packed_uint4(uint4(_S255.x, _S255.y, (*(&kernelContext_10)->g_scan_0)[int(0)], _S255.w)) ;
    }
    return;
}

float4 matrix_quat_0(matrix<float,int(3),int(3)>  m_8)
{
    float tr_0 = m_8[int(0)][int(0)] + m_8[int(1)][int(1)] + m_8[int(2)][int(2)];
    float4 q_9;
    if(tr_0 > 0.0f)
    {
        float s_3 = sqrt(tr_0 + 1.0f) * 2.0f;
        q_9 = float4((m_8[int(2)][int(1)] - m_8[int(1)][int(2)]) / s_3, (m_8[int(0)][int(2)] - m_8[int(2)][int(0)]) / s_3, (m_8[int(1)][int(0)] - m_8[int(0)][int(1)]) / s_3, 0.25f * s_3);
    }
    else
    {
        bool _S261;
        if((m_8[int(0)][int(0)]) > (m_8[int(1)][int(1)]))
        {
            _S261 = (m_8[int(0)][int(0)]) > (m_8[int(2)][int(2)]);
        }
        else
        {
            _S261 = false;
        }
        if(_S261)
        {
            float s_4 = sqrt(1.0f + m_8[int(0)][int(0)] - m_8[int(1)][int(1)] - m_8[int(2)][int(2)]) * 2.0f;
            q_9 = float4(0.25f * s_4, (m_8[int(0)][int(1)] + m_8[int(1)][int(0)]) / s_4, (m_8[int(0)][int(2)] + m_8[int(2)][int(0)]) / s_4, (m_8[int(2)][int(1)] - m_8[int(1)][int(2)]) / s_4);
        }
        else
        {
            if((m_8[int(1)][int(1)]) > (m_8[int(2)][int(2)]))
            {
                float s_5 = sqrt(1.0f + m_8[int(1)][int(1)] - m_8[int(0)][int(0)] - m_8[int(2)][int(2)]) * 2.0f;
                q_9 = float4((m_8[int(0)][int(1)] + m_8[int(1)][int(0)]) / s_5, 0.25f * s_5, (m_8[int(1)][int(2)] + m_8[int(2)][int(1)]) / s_5, (m_8[int(0)][int(2)] - m_8[int(2)][int(0)]) / s_5);
            }
            else
            {
                float s_6 = sqrt(1.0f + m_8[int(2)][int(2)] - m_8[int(0)][int(0)] - m_8[int(1)][int(1)]) * 2.0f;
                q_9 = float4((m_8[int(0)][int(2)] + m_8[int(2)][int(0)]) / s_6, (m_8[int(1)][int(2)] + m_8[int(2)][int(1)]) / s_6, 0.25f * s_6, (m_8[int(1)][int(0)] - m_8[int(0)][int(1)]) / s_6);
            }
        }
    }
    return normalize(q_9);
}

[[kernel]] void learned_gather_chunks(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_7 [[thread_position_in_threadgroup]], uint device* ctl_7 [[buffer(1)]], packed_float4 device* calls_7 [[buffer(2)]], packed_float4 device* nodes_7 [[buffer(3)]], LearnedParams_0 constant* params_7 [[buffer(0)]], packed_float4 device* bonds_7 [[buffer(4)]], uint device* chunk_node_7 [[buffer(8)]], uint device* counters_7 [[buffer(19)]], uint device* route_7 [[buffer(25)]], packed_uint4 device* isl_7 [[buffer(5)]], packed_float4 device* islf_7 [[buffer(6)]], Island_natural_0 device* islands_7 [[buffer(11)]], packed_float4 device* ctab_7 [[buffer(14)]], packed_float4 device* bodies_7 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_7 [[buffer(12)]], packed_float4 device* state_7 [[buffer(13)]], packed_float4 device* imps_7 [[buffer(10)]], uint device* status_7 [[buffer(18)]], packed_uint4 device* largs_7 [[buffer(27)]], packed_uint4 device* cisl_7 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_7 [[buffer(17)]], BondStatic_natural_0 device* bond_static_7 [[buffer(16)]], packed_float4 device* frozen_7 [[buffer(9)]], uint device* split_work_7 [[buffer(21)]], uint device* split_control_7 [[buffer(23)]], packed_uint4 device* split_parents_7 [[buffer(20)]], packed_uint4 device* split_slots_7 [[buffer(22)]], packed_float4 device* split_out_7 [[buffer(26)]], packed_uint4 device* split_firsts_7 [[buffer(24)]], float device* lout_7 [[buffer(28)]])
{
    thread KernelContext_0 kernelContext_11;
    (&kernelContext_11)->ctl_0 = ctl_7;
    (&kernelContext_11)->calls_0 = calls_7;
    (&kernelContext_11)->nodes_0 = nodes_7;
    (&kernelContext_11)->params_0 = params_7;
    (&kernelContext_11)->bonds_0 = bonds_7;
    (&kernelContext_11)->chunk_node_0 = chunk_node_7;
    (&kernelContext_11)->counters_0 = counters_7;
    (&kernelContext_11)->route_0 = route_7;
    (&kernelContext_11)->isl_0 = isl_7;
    (&kernelContext_11)->islf_0 = islf_7;
    (&kernelContext_11)->islands_0 = islands_7;
    (&kernelContext_11)->ctab_0 = ctab_7;
    (&kernelContext_11)->bodies_0 = bodies_7;
    (&kernelContext_11)->chunks_0 = chunks_7;
    (&kernelContext_11)->state_0 = state_7;
    (&kernelContext_11)->imps_0 = imps_7;
    (&kernelContext_11)->status_0 = status_7;
    (&kernelContext_11)->largs_0 = largs_7;
    (&kernelContext_11)->cisl_0 = cisl_7;
    (&kernelContext_11)->bond_dyn_0 = bond_dyn_7;
    (&kernelContext_11)->bond_static_0 = bond_static_7;
    (&kernelContext_11)->frozen_0 = frozen_7;
    (&kernelContext_11)->split_work_0 = split_work_7;
    (&kernelContext_11)->split_control_0 = split_control_7;
    (&kernelContext_11)->split_parents_0 = split_parents_7;
    (&kernelContext_11)->split_slots_0 = split_slots_7;
    (&kernelContext_11)->split_out_0 = split_out_7;
    (&kernelContext_11)->split_firsts_0 = split_firsts_7;
    (&kernelContext_11)->lout_0 = lout_7;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_7;
    (&kernelContext_11)->g_red_0 = &g_red_7;
    threadgroup array<float, int(6)> g_lambda_7;
    (&kernelContext_11)->g_lambda_0 = &g_lambda_7;
    threadgroup array<float3, int(256)> g_lo_7;
    (&kernelContext_11)->g_lo_0 = &g_lo_7;
    threadgroup array<float3, int(256)> g_hi_7;
    (&kernelContext_11)->g_hi_0 = &g_hi_7;
    threadgroup array<uint, int(32)> g_label_7;
    (&kernelContext_11)->g_label_0 = &g_label_7;
    threadgroup uint g_used_7;
    (&kernelContext_11)->g_used_0 = &g_used_7;
    threadgroup array<uint, int(32)> g_call_of_7;
    (&kernelContext_11)->g_call_of_0 = &g_call_of_7;
    threadgroup array<uint, int(64)> g_mask_of_7;
    (&kernelContext_11)->g_mask_of_0 = &g_mask_of_7;
    threadgroup array<uint, int(64)> g_cnt_isl_7;
    (&kernelContext_11)->g_cnt_isl_0 = &g_cnt_isl_7;
    threadgroup array<uint, int(64)> g_cnt_chunk_7;
    (&kernelContext_11)->g_cnt_chunk_0 = &g_cnt_chunk_7;
    threadgroup array<uint, int(64)> g_cnt_bond_7;
    (&kernelContext_11)->g_cnt_bond_0 = &g_cnt_bond_7;
    threadgroup array<uint, int(64)> g_cnt_imp_7;
    (&kernelContext_11)->g_cnt_imp_0 = &g_cnt_imp_7;
    threadgroup array<uint, int(256)> g_scan_7;
    (&kernelContext_11)->g_scan_0 = &g_scan_7;
    threadgroup array<float, int(4096)> g_sort_7;
    (&kernelContext_11)->g_sort_0 = &g_sort_7;
    threadgroup uint g_stale_7;
    (&kernelContext_11)->g_stale_0 = &g_stale_7;
    threadgroup uint g_broke_7;
    (&kernelContext_11)->g_broke_0 = &g_broke_7;
    uint e_0 = group_3.x;
    uint t_7 = thread_7.x;
    if(e_0 >= (*(ctl_7+9U)))
    {
        return;
    }
    uint _S262 = 2U * e_0;
    uint4 _S263 = uint4(*((&kernelContext_11)->cisl_0+_S262)) ;
    uint4 _S264 = uint4(*((&kernelContext_11)->cisl_0+(_S262 + 1U))) ;
    uint i_5 = _S263.x;
    Island_natural_0 device* _S265 = (&kernelContext_11)->islands_0+i_5;
    Frame_0 _S266 = island_frame_0(i_5, &kernelContext_11);
    uint c_5 = t_7;
    for(;;)
    {
        if(c_5 < (_S263.z))
        {
        }
        else
        {
            break;
        }
        uint k_17 = (uint4(_S265->range_0) ).x + c_5;
        uint n_1 = _S263.y + c_5;
        ChunkStatic_natural_0 device* _S267 = (&kernelContext_11)->chunks_0+k_17;
        thread Frame_0 _S268 = _S266;
        float3 _S269 = chunk_world_0(&_S268, k_17, &kernelContext_11);
        matrix<float,int(3),int(3)>  rest_0 = transpose(matrix<float,int(3),int(3)> ((float4(_S267->crot0_0) ).xyz, (float4(_S267->crot1_0) ).xyz, (float4(_S267->crot2_0) ).xyz));
        matrix<float,int(3),int(3)>  ic_0 = (((rest_0) * ((((matrix<float,int(3),int(3)> ((float4(_S267->inertia0_1) ).xyz, (float4(_S267->inertia1_1) ).xyz, (float4(_S267->inertia2_1) ).xyz)) * (transpose(rest_0)))))));
        float4 q_10 = normalize(qmul_0(_S266.q_8, matrix_quat_0(rest_0)));
        float3 v_6 = _S266.v_5 + cross(_S266.w_4, _S269 - _S266.com_1);
        uint4 _S270 = uint4(_S267->info_1) ;
        bool anchored_0 = (_S270.x) != 0U;
        uint _S271 = 14U * n_1;
        *((&kernelContext_11)->nodes_0+_S271) = packed_float4(float4(_S269, (float4(_S267->center_0) ).w)) ;
        *((&kernelContext_11)->nodes_0+(_S271 + 1U)) = packed_float4(q_10) ;
        packed_float4 device* _S272 = (&kernelContext_11)->nodes_0+(_S271 + 2U);
        float _S273;
        if(anchored_0)
        {
            _S273 = 1.0f;
        }
        else
        {
            _S273 = 0.0f;
        }
        *_S272 = packed_float4(float4(ic_0[int(0)][int(0)], ic_0[int(1)][int(1)], ic_0[int(2)][int(2)], _S273)) ;
        *((&kernelContext_11)->nodes_0+(_S271 + 3U)) = packed_float4(float4(v_6, (as_type<float>((_S263.w))))) ;
        *((&kernelContext_11)->nodes_0+(_S271 + 4U)) = packed_float4(float4(_S266.w_4, (as_type<float>((0U))))) ;
        *((&kernelContext_11)->nodes_0+(_S271 + 12U)) = packed_float4(float4((as_type<float>((k_17))), (as_type<float>((_S264.x))), (as_type<float>((_S270.z))), (as_type<float>((_S270.w))))) ;
        *((&kernelContext_11)->chunk_node_0+k_17) = n_1 + 1U;
        c_5 = c_5 + 256U;
    }
    return;
}

[[kernel]] void learned_gather_bonds(uint3 group_4 [[threadgroup_position_in_grid]], uint3 thread_8 [[thread_position_in_threadgroup]], uint device* ctl_8 [[buffer(1)]], packed_float4 device* calls_8 [[buffer(2)]], packed_float4 device* nodes_8 [[buffer(3)]], LearnedParams_0 constant* params_8 [[buffer(0)]], packed_float4 device* bonds_8 [[buffer(4)]], uint device* chunk_node_8 [[buffer(8)]], uint device* counters_8 [[buffer(19)]], uint device* route_8 [[buffer(25)]], packed_uint4 device* isl_8 [[buffer(5)]], packed_float4 device* islf_8 [[buffer(6)]], Island_natural_0 device* islands_8 [[buffer(11)]], packed_float4 device* ctab_8 [[buffer(14)]], packed_float4 device* bodies_8 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_8 [[buffer(12)]], packed_float4 device* state_8 [[buffer(13)]], packed_float4 device* imps_8 [[buffer(10)]], uint device* status_8 [[buffer(18)]], packed_uint4 device* largs_8 [[buffer(27)]], packed_uint4 device* cisl_8 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_8 [[buffer(17)]], BondStatic_natural_0 device* bond_static_8 [[buffer(16)]], packed_float4 device* frozen_8 [[buffer(9)]], uint device* split_work_8 [[buffer(21)]], uint device* split_control_8 [[buffer(23)]], packed_uint4 device* split_parents_8 [[buffer(20)]], packed_uint4 device* split_slots_8 [[buffer(22)]], packed_float4 device* split_out_8 [[buffer(26)]], packed_uint4 device* split_firsts_8 [[buffer(24)]], float device* lout_8 [[buffer(28)]])
{
    thread KernelContext_0 kernelContext_12;
    (&kernelContext_12)->ctl_0 = ctl_8;
    (&kernelContext_12)->calls_0 = calls_8;
    (&kernelContext_12)->nodes_0 = nodes_8;
    (&kernelContext_12)->params_0 = params_8;
    (&kernelContext_12)->bonds_0 = bonds_8;
    (&kernelContext_12)->chunk_node_0 = chunk_node_8;
    (&kernelContext_12)->counters_0 = counters_8;
    (&kernelContext_12)->route_0 = route_8;
    (&kernelContext_12)->isl_0 = isl_8;
    (&kernelContext_12)->islf_0 = islf_8;
    (&kernelContext_12)->islands_0 = islands_8;
    (&kernelContext_12)->ctab_0 = ctab_8;
    (&kernelContext_12)->bodies_0 = bodies_8;
    (&kernelContext_12)->chunks_0 = chunks_8;
    (&kernelContext_12)->state_0 = state_8;
    (&kernelContext_12)->imps_0 = imps_8;
    (&kernelContext_12)->status_0 = status_8;
    (&kernelContext_12)->largs_0 = largs_8;
    (&kernelContext_12)->cisl_0 = cisl_8;
    (&kernelContext_12)->bond_dyn_0 = bond_dyn_8;
    (&kernelContext_12)->bond_static_0 = bond_static_8;
    (&kernelContext_12)->frozen_0 = frozen_8;
    (&kernelContext_12)->split_work_0 = split_work_8;
    (&kernelContext_12)->split_control_0 = split_control_8;
    (&kernelContext_12)->split_parents_0 = split_parents_8;
    (&kernelContext_12)->split_slots_0 = split_slots_8;
    (&kernelContext_12)->split_out_0 = split_out_8;
    (&kernelContext_12)->split_firsts_0 = split_firsts_8;
    (&kernelContext_12)->lout_0 = lout_8;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_8;
    (&kernelContext_12)->g_red_0 = &g_red_8;
    threadgroup array<float, int(6)> g_lambda_8;
    (&kernelContext_12)->g_lambda_0 = &g_lambda_8;
    threadgroup array<float3, int(256)> g_lo_8;
    (&kernelContext_12)->g_lo_0 = &g_lo_8;
    threadgroup array<float3, int(256)> g_hi_8;
    (&kernelContext_12)->g_hi_0 = &g_hi_8;
    threadgroup array<uint, int(32)> g_label_8;
    (&kernelContext_12)->g_label_0 = &g_label_8;
    threadgroup uint g_used_8;
    (&kernelContext_12)->g_used_0 = &g_used_8;
    threadgroup array<uint, int(32)> g_call_of_8;
    (&kernelContext_12)->g_call_of_0 = &g_call_of_8;
    threadgroup array<uint, int(64)> g_mask_of_8;
    (&kernelContext_12)->g_mask_of_0 = &g_mask_of_8;
    threadgroup array<uint, int(64)> g_cnt_isl_8;
    (&kernelContext_12)->g_cnt_isl_0 = &g_cnt_isl_8;
    threadgroup array<uint, int(64)> g_cnt_chunk_8;
    (&kernelContext_12)->g_cnt_chunk_0 = &g_cnt_chunk_8;
    threadgroup array<uint, int(64)> g_cnt_bond_8;
    (&kernelContext_12)->g_cnt_bond_0 = &g_cnt_bond_8;
    threadgroup array<uint, int(64)> g_cnt_imp_8;
    (&kernelContext_12)->g_cnt_imp_0 = &g_cnt_imp_8;
    threadgroup array<uint, int(256)> g_scan_8;
    (&kernelContext_12)->g_scan_0 = &g_scan_8;
    threadgroup array<float, int(4096)> g_sort_8;
    (&kernelContext_12)->g_sort_0 = &g_sort_8;
    threadgroup uint g_stale_8;
    (&kernelContext_12)->g_stale_0 = &g_stale_8;
    threadgroup uint g_broke_8;
    (&kernelContext_12)->g_broke_0 = &g_broke_8;
    uint e_1 = group_4.x;
    uint t_8 = thread_8.x;
    uint base_0;
    if(e_1 >= (*(ctl_8+9U)))
    {
        base_0 = 1U;
    }
    else
    {
        base_0 = 0U;
    }
    if((group_uniform_0(base_0, t_8)) != 0U)
    {
        return;
    }
    uint _S274 = 2U * e_1;
    uint4 _S275 = uint4(*((&kernelContext_12)->cisl_0+_S274)) ;
    uint4 _S276 = uint4(*((&kernelContext_12)->cisl_0+(_S274 + 1U))) ;
    uint4 _S277 = uint4(((&kernelContext_12)->islands_0+_S275.x)->range_0) ;
    uint _S278 = group_uniform_0(_S277.w, t_8);
    base_0 = group_uniform_0(_S277.z, t_8);
    uint carry_0 = 0U;
    for(;;)
    {
        if(base_0 < _S278)
        {
        }
        else
        {
            break;
        }
        uint bi_1 = base_0 + t_8;
        bool live_1;
        if(bi_1 < _S278)
        {
            bool _S279 = ((float4(((&kernelContext_12)->bond_static_0+bi_1)->law_0.stiff1_0) ).w) != 0.0f;
            thread JointState_0 _S280 = ((&kernelContext_12)->bond_dyn_0+bi_1)->js_0;
            bool _S281 = connected_0(&_S280, _S279);
            live_1 = _S281;
        }
        else
        {
            live_1 = false;
        }
        uint _S282;
        if(live_1)
        {
            _S282 = 1U;
        }
        else
        {
            _S282 = 0U;
        }
        (*(&kernelContext_12)->g_scan_0)[t_8] = _S282;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint step_0 = 1U;
        for(;;)
        {
            if(step_0 < 256U)
            {
            }
            else
            {
                break;
            }
            uint add_0;
            if(t_8 >= step_0)
            {
                add_0 = (*(&kernelContext_12)->g_scan_0)[t_8 - step_0];
            }
            else
            {
                add_0 = 0U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            (*(&kernelContext_12)->g_scan_0)[t_8] = (*(&kernelContext_12)->g_scan_0)[t_8] + add_0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            step_0 = step_0 << 1U;
        }
        if(live_1)
        {
            uint _S283 = 2U * (_S276.y + carry_0 + (*(&kernelContext_12)->g_scan_0)[t_8] - 1U);
            *((&kernelContext_12)->bonds_0+_S283) = packed_float4(float4((&((&kernelContext_12)->bond_dyn_0+bi_1)->js_0)->damage_0, -30.0f, 0.0f, (as_type<float>((1U))))) ;
            *((&kernelContext_12)->bonds_0+(_S283 + 1U)) = packed_float4(float4(0.0f, (as_type<float>((0U))), (as_type<float>((bi_1))), (as_type<float>((_S275.w))))) ;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint carry_1 = carry_0 + (*(&kernelContext_12)->g_scan_0)[255U];
        base_0 = base_0 + 256U;
        carry_0 = carry_1;
    }
    return;
}

[[kernel]] void learned_gather_impactors(uint3 group_5 [[threadgroup_position_in_grid]], uint3 thread_9 [[thread_position_in_threadgroup]], uint device* ctl_9 [[buffer(1)]], packed_float4 device* calls_9 [[buffer(2)]], packed_float4 device* nodes_9 [[buffer(3)]], LearnedParams_0 constant* params_9 [[buffer(0)]], packed_float4 device* bonds_9 [[buffer(4)]], uint device* chunk_node_9 [[buffer(8)]], uint device* counters_9 [[buffer(19)]], uint device* route_9 [[buffer(25)]], packed_uint4 device* isl_9 [[buffer(5)]], packed_float4 device* islf_9 [[buffer(6)]], Island_natural_0 device* islands_9 [[buffer(11)]], packed_float4 device* ctab_9 [[buffer(14)]], packed_float4 device* bodies_9 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_9 [[buffer(12)]], packed_float4 device* state_9 [[buffer(13)]], packed_float4 device* imps_9 [[buffer(10)]], uint device* status_9 [[buffer(18)]], packed_uint4 device* largs_9 [[buffer(27)]], packed_uint4 device* cisl_9 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_9 [[buffer(17)]], BondStatic_natural_0 device* bond_static_9 [[buffer(16)]], packed_float4 device* frozen_9 [[buffer(9)]], uint device* split_work_9 [[buffer(21)]], uint device* split_control_9 [[buffer(23)]], packed_uint4 device* split_parents_9 [[buffer(20)]], packed_uint4 device* split_slots_9 [[buffer(22)]], packed_float4 device* split_out_9 [[buffer(26)]], packed_uint4 device* split_firsts_9 [[buffer(24)]], float device* lout_9 [[buffer(28)]])
{
    thread KernelContext_0 kernelContext_13;
    (&kernelContext_13)->ctl_0 = ctl_9;
    (&kernelContext_13)->calls_0 = calls_9;
    (&kernelContext_13)->nodes_0 = nodes_9;
    (&kernelContext_13)->params_0 = params_9;
    (&kernelContext_13)->bonds_0 = bonds_9;
    (&kernelContext_13)->chunk_node_0 = chunk_node_9;
    (&kernelContext_13)->counters_0 = counters_9;
    (&kernelContext_13)->route_0 = route_9;
    (&kernelContext_13)->isl_0 = isl_9;
    (&kernelContext_13)->islf_0 = islf_9;
    (&kernelContext_13)->islands_0 = islands_9;
    (&kernelContext_13)->ctab_0 = ctab_9;
    (&kernelContext_13)->bodies_0 = bodies_9;
    (&kernelContext_13)->chunks_0 = chunks_9;
    (&kernelContext_13)->state_0 = state_9;
    (&kernelContext_13)->imps_0 = imps_9;
    (&kernelContext_13)->status_0 = status_9;
    (&kernelContext_13)->largs_0 = largs_9;
    (&kernelContext_13)->cisl_0 = cisl_9;
    (&kernelContext_13)->bond_dyn_0 = bond_dyn_9;
    (&kernelContext_13)->bond_static_0 = bond_static_9;
    (&kernelContext_13)->frozen_0 = frozen_9;
    (&kernelContext_13)->split_work_0 = split_work_9;
    (&kernelContext_13)->split_control_0 = split_control_9;
    (&kernelContext_13)->split_parents_0 = split_parents_9;
    (&kernelContext_13)->split_slots_0 = split_slots_9;
    (&kernelContext_13)->split_out_0 = split_out_9;
    (&kernelContext_13)->split_firsts_0 = split_firsts_9;
    (&kernelContext_13)->lout_0 = lout_9;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_9;
    (&kernelContext_13)->g_red_0 = &g_red_9;
    threadgroup array<float, int(6)> g_lambda_9;
    (&kernelContext_13)->g_lambda_0 = &g_lambda_9;
    threadgroup array<float3, int(256)> g_lo_9;
    (&kernelContext_13)->g_lo_0 = &g_lo_9;
    threadgroup array<float3, int(256)> g_hi_9;
    (&kernelContext_13)->g_hi_0 = &g_hi_9;
    threadgroup array<uint, int(32)> g_label_9;
    (&kernelContext_13)->g_label_0 = &g_label_9;
    threadgroup uint g_used_9;
    (&kernelContext_13)->g_used_0 = &g_used_9;
    threadgroup array<uint, int(32)> g_call_of_9;
    (&kernelContext_13)->g_call_of_0 = &g_call_of_9;
    threadgroup array<uint, int(64)> g_mask_of_9;
    (&kernelContext_13)->g_mask_of_0 = &g_mask_of_9;
    threadgroup array<uint, int(64)> g_cnt_isl_9;
    (&kernelContext_13)->g_cnt_isl_0 = &g_cnt_isl_9;
    threadgroup array<uint, int(64)> g_cnt_chunk_9;
    (&kernelContext_13)->g_cnt_chunk_0 = &g_cnt_chunk_9;
    threadgroup array<uint, int(64)> g_cnt_bond_9;
    (&kernelContext_13)->g_cnt_bond_0 = &g_cnt_bond_9;
    threadgroup array<uint, int(64)> g_cnt_imp_9;
    (&kernelContext_13)->g_cnt_imp_0 = &g_cnt_imp_9;
    threadgroup array<uint, int(256)> g_scan_9;
    (&kernelContext_13)->g_scan_0 = &g_scan_9;
    threadgroup array<float, int(4096)> g_sort_9;
    (&kernelContext_13)->g_sort_0 = &g_sort_9;
    threadgroup uint g_stale_9;
    (&kernelContext_13)->g_stale_0 = &g_stale_9;
    threadgroup uint g_broke_9;
    (&kernelContext_13)->g_broke_0 = &g_broke_9;
    uint g_3 = group_5.x;
    uint k_18 = thread_9.x;
    if(g_3 >= (*(ctl_9+1U)))
    {
        return;
    }
    uint _S284 = 8U * g_3;
    uint mask_5 = (as_type<uint>(((float4(*((&kernelContext_13)->calls_0+(_S284 + 5U))) ).w)));
    uint _S285 = 1U << k_18;
    if((mask_5 & _S285) == 0U)
    {
        return;
    }
    uint _S286 = 3U * k_18;
    float4 _S287 = float4(*((&kernelContext_13)->imps_0+_S286)) ;
    float4 _S288 = float4(*((&kernelContext_13)->imps_0+(_S286 + 1U))) ;
    uint body_1 = (as_type<uint>((_S287.x)));
    uint _S289 = 7U * body_1;
    float4 q_11 = body_rotation_zup_0(float4(*((&kernelContext_13)->bodies_0+(_S289 + 1U))) , float4(0.0f, 0.0f, 0.0f, 1.0f));
    uint _S290 = 14U * ((as_type<uint>(((float4(*((&kernelContext_13)->calls_0+(_S284 + 2U))) ).z))) + (as_type<uint>(((float4(*((&kernelContext_13)->calls_0+(_S284 + 3U))) ).w))) + ((uint)popcount((mask_5 & (_S285 - 1U)))));
    *((&kernelContext_13)->nodes_0+_S290) = packed_float4(float4(to_zup_0((float4(*((&kernelContext_13)->bodies_0+_S289)) ).xyz), _S287.w)) ;
    *((&kernelContext_13)->nodes_0+(_S290 + 1U)) = packed_float4(q_11) ;
    *((&kernelContext_13)->nodes_0+(_S290 + 2U)) = packed_float4(float4(_S288.xyz, 0.0f)) ;
    *((&kernelContext_13)->nodes_0+(_S290 + 3U)) = packed_float4(float4(to_zup_0((float4(*((&kernelContext_13)->bodies_0+(_S289 + 3U))) ).xyz), (as_type<float>((g_3))))) ;
    *((&kernelContext_13)->nodes_0+(_S290 + 4U)) = packed_float4(float4(to_zup_0((float4(*((&kernelContext_13)->bodies_0+(_S289 + 4U))) ).xyz), (as_type<float>((1U))))) ;
    float _S291 = (as_type<float>((4294967295U)));
    *((&kernelContext_13)->nodes_0+(_S290 + 12U)) = packed_float4(float4((as_type<float>((k_18))), (as_type<float>((body_1))), _S291, _S291)) ;
    return;
}

void sort_group_0(uint t_9, uint n_2, KernelContext_0 thread* kernelContext_14)
{
    uint m_9 = 1U;
    for(;;)
    {
        if(m_9 < n_2)
        {
        }
        else
        {
            break;
        }
        m_9 = m_9 << 1U;
    }
    float _S292 = float_bits_0(2139095040U);
    uint j_3 = n_2 + t_9;
    for(;;)
    {
        if(j_3 < m_9)
        {
        }
        else
        {
            break;
        }
        (*kernelContext_14->g_sort_0)[j_3] = _S292;
        j_3 = j_3 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint size_0 = 2U;
    for(;;)
    {
        if(size_0 <= m_9)
        {
        }
        else
        {
            break;
        }
        uint stride_0 = size_0 >> 1U;
        for(;;)
        {
            if(stride_0 > 0U)
            {
            }
            else
            {
                break;
            }
            j_3 = t_9;
            for(;;)
            {
                if(j_3 < m_9)
                {
                }
                else
                {
                    break;
                }
                uint p_0 = j_3 ^ stride_0;
                if(p_0 > j_3)
                {
                    float a_8 = (*kernelContext_14->g_sort_0)[j_3];
                    float b_7 = (*kernelContext_14->g_sort_0)[p_0];
                    if((((*kernelContext_14->g_sort_0)[j_3]) > (*kernelContext_14->g_sort_0)[p_0]) == ((j_3 & size_0) == 0U))
                    {
                        (*kernelContext_14->g_sort_0)[j_3] = b_7;
                        (*kernelContext_14->g_sort_0)[p_0] = a_8;
                    }
                }
                j_3 = j_3 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            stride_0 = stride_0 >> 1U;
        }
        size_0 = size_0 << 1U;
    }
    return;
}

float median_sorted_0(uint n_3, KernelContext_0 thread* kernelContext_15)
{
    uint k_19 = n_3 / 2U;
    float _S293;
    if((n_3 % 2U) == 1U)
    {
        _S293 = (*kernelContext_15->g_sort_0)[k_19];
    }
    else
    {
        _S293 = 0.5f * ((*kernelContext_15->g_sort_0)[k_19 - 1U] + (*kernelContext_15->g_sort_0)[k_19]);
    }
    return _S293;
}

[[kernel]] void learned_finish(uint3 group_6 [[threadgroup_position_in_grid]], uint3 thread_10 [[thread_position_in_threadgroup]], uint device* ctl_10 [[buffer(1)]], packed_float4 device* calls_10 [[buffer(2)]], packed_float4 device* nodes_10 [[buffer(3)]], LearnedParams_0 constant* params_10 [[buffer(0)]], packed_float4 device* bonds_10 [[buffer(4)]], uint device* chunk_node_10 [[buffer(8)]], uint device* counters_10 [[buffer(19)]], uint device* route_10 [[buffer(25)]], packed_uint4 device* isl_10 [[buffer(5)]], packed_float4 device* islf_10 [[buffer(6)]], Island_natural_0 device* islands_10 [[buffer(11)]], packed_float4 device* ctab_10 [[buffer(14)]], packed_float4 device* bodies_10 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_10 [[buffer(12)]], packed_float4 device* state_10 [[buffer(13)]], packed_float4 device* imps_10 [[buffer(10)]], uint device* status_10 [[buffer(18)]], packed_uint4 device* largs_10 [[buffer(27)]], packed_uint4 device* cisl_10 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_10 [[buffer(17)]], BondStatic_natural_0 device* bond_static_10 [[buffer(16)]], packed_float4 device* frozen_10 [[buffer(9)]], uint device* split_work_10 [[buffer(21)]], uint device* split_control_10 [[buffer(23)]], packed_uint4 device* split_parents_10 [[buffer(20)]], packed_uint4 device* split_slots_10 [[buffer(22)]], packed_float4 device* split_out_10 [[buffer(26)]], packed_uint4 device* split_firsts_10 [[buffer(24)]], float device* lout_10 [[buffer(28)]])
{
    uint jc_0;
    thread KernelContext_0 kernelContext_16;
    (&kernelContext_16)->ctl_0 = ctl_10;
    (&kernelContext_16)->calls_0 = calls_10;
    (&kernelContext_16)->nodes_0 = nodes_10;
    (&kernelContext_16)->params_0 = params_10;
    (&kernelContext_16)->bonds_0 = bonds_10;
    (&kernelContext_16)->chunk_node_0 = chunk_node_10;
    (&kernelContext_16)->counters_0 = counters_10;
    (&kernelContext_16)->route_0 = route_10;
    (&kernelContext_16)->isl_0 = isl_10;
    (&kernelContext_16)->islf_0 = islf_10;
    (&kernelContext_16)->islands_0 = islands_10;
    (&kernelContext_16)->ctab_0 = ctab_10;
    (&kernelContext_16)->bodies_0 = bodies_10;
    (&kernelContext_16)->chunks_0 = chunks_10;
    (&kernelContext_16)->state_0 = state_10;
    (&kernelContext_16)->imps_0 = imps_10;
    (&kernelContext_16)->status_0 = status_10;
    (&kernelContext_16)->largs_0 = largs_10;
    (&kernelContext_16)->cisl_0 = cisl_10;
    (&kernelContext_16)->bond_dyn_0 = bond_dyn_10;
    (&kernelContext_16)->bond_static_0 = bond_static_10;
    (&kernelContext_16)->frozen_0 = frozen_10;
    (&kernelContext_16)->split_work_0 = split_work_10;
    (&kernelContext_16)->split_control_0 = split_control_10;
    (&kernelContext_16)->split_parents_0 = split_parents_10;
    (&kernelContext_16)->split_slots_0 = split_slots_10;
    (&kernelContext_16)->split_out_0 = split_out_10;
    (&kernelContext_16)->split_firsts_0 = split_firsts_10;
    (&kernelContext_16)->lout_0 = lout_10;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_10;
    (&kernelContext_16)->g_red_0 = &g_red_10;
    threadgroup array<float, int(6)> g_lambda_10;
    (&kernelContext_16)->g_lambda_0 = &g_lambda_10;
    threadgroup array<float3, int(256)> g_lo_10;
    (&kernelContext_16)->g_lo_0 = &g_lo_10;
    threadgroup array<float3, int(256)> g_hi_10;
    (&kernelContext_16)->g_hi_0 = &g_hi_10;
    threadgroup array<uint, int(32)> g_label_10;
    (&kernelContext_16)->g_label_0 = &g_label_10;
    threadgroup uint g_used_10;
    (&kernelContext_16)->g_used_0 = &g_used_10;
    threadgroup array<uint, int(32)> g_call_of_10;
    (&kernelContext_16)->g_call_of_0 = &g_call_of_10;
    threadgroup array<uint, int(64)> g_mask_of_10;
    (&kernelContext_16)->g_mask_of_0 = &g_mask_of_10;
    threadgroup array<uint, int(64)> g_cnt_isl_10;
    (&kernelContext_16)->g_cnt_isl_0 = &g_cnt_isl_10;
    threadgroup array<uint, int(64)> g_cnt_chunk_10;
    (&kernelContext_16)->g_cnt_chunk_0 = &g_cnt_chunk_10;
    threadgroup array<uint, int(64)> g_cnt_bond_10;
    (&kernelContext_16)->g_cnt_bond_0 = &g_cnt_bond_10;
    threadgroup array<uint, int(64)> g_cnt_imp_10;
    (&kernelContext_16)->g_cnt_imp_0 = &g_cnt_imp_10;
    threadgroup array<uint, int(256)> g_scan_10;
    (&kernelContext_16)->g_scan_0 = &g_scan_10;
    threadgroup array<float, int(4096)> g_sort_10;
    (&kernelContext_16)->g_sort_0 = &g_sort_10;
    threadgroup uint g_stale_10;
    (&kernelContext_16)->g_stale_0 = &g_stale_10;
    threadgroup uint g_broke_10;
    (&kernelContext_16)->g_broke_0 = &g_broke_10;
    uint g_4 = group_6.x;
    uint t_10 = thread_10.x;
    uint j_4;
    if(g_4 >= (*(ctl_10+1U)))
    {
        j_4 = 1U;
    }
    else
    {
        j_4 = 0U;
    }
    if((group_uniform_0(j_4, t_10)) != 0U)
    {
        return;
    }
    uint _S294 = 8U * g_4;
    uint _S295 = _S294 + 2U;
    uint _S296 = (as_type<uint>(((float4(*((&kernelContext_16)->calls_0+_S295)) ).z)));
    uint _S297 = group_uniform_0((as_type<uint>(((float4(*((&kernelContext_16)->calls_0+_S295)) ).w))), t_10);
    uint nc_0 = group_uniform_0((as_type<uint>(((float4(*((&kernelContext_16)->calls_0+(_S294 + 3U))) ).w))), t_10);
    thread float4 a_9;
    thread float4 b_8;
    thread float4 c_6;
    thread float4 d_4;
    float4 _S298 = float4(0.0f) ;
    j_4 = t_10;
    float4 sa_5 = _S298;
    float4 sb_5 = _S298;
    for(;;)
    {
        if(j_4 < _S297)
        {
        }
        else
        {
            break;
        }
        uint k_20 = _S296 + j_4;
        float4 _S299 = node_0(k_20, 0U, &kernelContext_16);
        float _S300 = _S299.w;
        float3 _S301 = float3(_S300) ;
        float4 sa_6 = sa_5 + float4(_S299.xyz * _S301, _S300);
        float4 _S302 = node_0(k_20, 3U, &kernelContext_16);
        float3 _S303 = _S302.xyz * _S301;
        float4 _S304 = node_0(k_20, 2U, &kernelContext_16);
        float4 sb_6 = sb_5 + float4(_S303, _S304.w);
        j_4 = j_4 + 256U;
        sa_5 = sa_6;
        sb_5 = sb_6;
    }
    reduce_store_0(t_10, sa_5, sb_5, _S298, _S298, &kernelContext_16);
    reduce_read_0(&a_9, &b_8, &c_6, &d_4, &kernelContext_16);
    float3 _S305 = float3(a_9.w) ;
    float3 com_3 = a_9.xyz / _S305;
    float3 _S306 = b_8.xyz / _S305;
    bool anchored_1 = (b_8.w) > 0.0f;
    float vmax_0 = (&kernelContext_16)->params_0->speed_0;
    j_4 = t_10;
    float ke_in_1 = 0.0f;
    float wg_0 = 0.0f;
    for(;;)
    {
        if(j_4 < _S297)
        {
        }
        else
        {
            break;
        }
        uint k_21 = _S296 + j_4;
        float4 _S307 = node_0(k_21, 0U, &kernelContext_16);
        float4 _S308 = node_0(k_21, 3U, &kernelContext_16);
        float3 v_7 = _S308.xyz;
        float4 _S309 = node_0(k_21, 4U, &kernelContext_16);
        float3 w_5 = _S309.xyz;
        float4 _S310 = node_0(k_21, 4U, &kernelContext_16);
        bool imp_0 = (as_type<uint>((_S310.w))) != 0U;
        float4 _S311 = node_0(k_21, 12U, &kernelContext_16);
        uint id_1 = (as_type<uint>((_S311.x)));
        float rim_0;
        if(imp_0)
        {
            rim_0 = (float4(*((&kernelContext_16)->imps_0+(3U * id_1 + 2U))) ).x;
        }
        else
        {
            rim_0 = length((float4(((&kernelContext_16)->chunks_0+id_1)->half_0) ).xyz);
        }
        float _S312 = max(vmax_0, max(length(v_7 - _S306), length(w_5) * rim_0));
        if(imp_0)
        {
            float vmax_1 = _S312;
            jc_0 = 0U;
            for(;;)
            {
                if(jc_0 < nc_0)
                {
                }
                else
                {
                    break;
                }
                uint _S313 = _S296 + jc_0;
                float4 _S314 = node_0(_S313, 3U, &kernelContext_16);
                float3 _S315 = v_7 - _S314.xyz;
                float4 _S316 = node_0(_S313, 0U, &kernelContext_16);
                float _S317 = max(vmax_1, dot(_S315, normalize(_S316.xyz - _S307.xyz)));
                uint _S318 = jc_0 + 1U;
                vmax_1 = _S317;
                jc_0 = _S318;
            }
            vmax_0 = vmax_1;
        }
        else
        {
            vmax_0 = _S312;
        }
        float4 _S319 = node_0(k_21, 1U, &kernelContext_16);
        float4 _S320 = node_0(k_21, 2U, &kernelContext_16);
        float _S321 = _S307.w;
        float ke_in_2 = ke_in_1 + 0.5f * (_S321 * dot(v_7, v_7) + dot(w_5, (((w_5) * (world_inertia_0(_S319, _S320.xyz))))));
        float wg_1 = wg_0 + _S321 * dot(v_7 * float3((&kernelContext_16)->params_0->dt_0)  + (&kernelContext_16)->params_0->gravity_0.xyz * float3((0.5f * (&kernelContext_16)->params_0->dt_0 * (&kernelContext_16)->params_0->dt_0)) , (&kernelContext_16)->params_0->gravity_0.xyz);
        j_4 = j_4 + 256U;
        ke_in_1 = ke_in_2;
        wg_0 = wg_1;
    }
    (*(&kernelContext_16)->g_sort_0)[t_10] = vmax_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    jc_0 = 128U;
    for(;;)
    {
        if(jc_0 > 0U)
        {
        }
        else
        {
            break;
        }
        if(t_10 < jc_0)
        {
            (*(&kernelContext_16)->g_sort_0)[t_10] = max((*(&kernelContext_16)->g_sort_0)[t_10], (*(&kernelContext_16)->g_sort_0)[t_10 + jc_0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        jc_0 = jc_0 >> 1U;
    }
    float vmax_2 = (*(&kernelContext_16)->g_sort_0)[int(0)];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    reduce_store_0(t_10, float4(ke_in_1, wg_0, 0.0f, 0.0f), _S298, _S298, _S298, &kernelContext_16);
    reduce_read_0(&a_9, &b_8, &c_6, &d_4, &kernelContext_16);
    float ke_in_3 = a_9.x;
    float wg_2 = a_9.y;
    uint _S322 = min(nc_0, 4096U);
    j_4 = t_10;
    for(;;)
    {
        if(j_4 < _S322)
        {
        }
        else
        {
            break;
        }
        float4 _S323 = node_0(_S296 + j_4, 12U, &kernelContext_16);
        float3 h_0 = (float4(((&kernelContext_16)->chunks_0+(as_type<uint>((_S323.x))))->half_0) ).xyz;
        (*(&kernelContext_16)->g_sort_0)[j_4] = 8.0f * h_0.x * h_0.y * h_0.z;
        j_4 = j_4 + 256U;
    }
    sort_group_0(t_10, _S322, &kernelContext_16);
    float _S324 = median_sorted_0(_S322, &kernelContext_16);
    float L_0 = pow(_S324, 0.3333333432674408f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    j_4 = t_10;
    for(;;)
    {
        if(j_4 < _S322)
        {
        }
        else
        {
            break;
        }
        float4 _S325 = node_0(_S296 + j_4, 0U, &kernelContext_16);
        (*(&kernelContext_16)->g_sort_0)[j_4] = _S325.w;
        j_4 = j_4 + 256U;
    }
    sort_group_0(t_10, _S322, &kernelContext_16);
    float _S326 = median_sorted_0(_S322, &kernelContext_16);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    j_4 = t_10;
    for(;;)
    {
        if(j_4 < _S297)
        {
        }
        else
        {
            break;
        }
        uint k_22 = _S296 + j_4;
        float4 _S327 = node_0(k_22, 0U, &kernelContext_16);
        uint _S328 = 14U * k_22;
        float3 _S329 = _S327.xyz;
        *((&kernelContext_16)->nodes_0+(_S328 + 13U)) = packed_float4(float4(_S329, 0.0f)) ;
        *((&kernelContext_16)->nodes_0+_S328) = packed_float4(float4(_S329 - com_3, _S327.w)) ;
        j_4 = j_4 + 256U;
    }
    if(t_10 == 0U)
    {
        float _S330 = max(ke_in_3 + wg_2, 0.5f * _S326 * vmax_2 * vmax_2);
        uint flags_1;
        if(anchored_1)
        {
            flags_1 = 4U;
        }
        else
        {
            flags_1 = 0U;
        }
        bool _S331;
        if(!anchored_1)
        {
            _S331 = ((&kernelContext_16)->params_0->gravity_0.w) == 0.0f;
        }
        else
        {
            _S331 = false;
        }
        if(_S331)
        {
            flags_1 = flags_1 | 1U;
        }
        float4 _S332 = float4(*((&kernelContext_16)->calls_0+_S295)) ;
        *((&kernelContext_16)->calls_0+_S294) = packed_float4(float4(L_0, _S326, vmax_2, 0.0f)) ;
        *((&kernelContext_16)->calls_0+(_S294 + 1U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, (&kernelContext_16)->params_0->dt_0)) ;
        *((&kernelContext_16)->calls_0+_S295) = packed_float4(float4(_S330, (as_type<float>((flags_1))), _S332.z, _S332.w)) ;
        uint _S333 = _S294 + 5U;
        *((&kernelContext_16)->calls_0+_S333) = packed_float4(float4(com_3, (float4(*((&kernelContext_16)->calls_0+_S333)) ).w)) ;
    }
    return;
}

float3 inverse_rotate_0(const Quat_0 thread* q_12, float3 v_8)
{
    thread Quat_0 c_7;
    (&c_7)->w_0 = q_12->w_0;
    (&c_7)->x_0 = - q_12->x_0;
    (&c_7)->y_0 = - q_12->y_0;
    (&c_7)->z_0 = - q_12->z_0;
    thread Quat_0 _S334 = c_7;
    float3 _S335 = rotate_0(&_S334, v_8);
    return _S335;
}

[[kernel]] void learned_stub(uint3 group_7 [[threadgroup_position_in_grid]], uint3 thread_11 [[thread_position_in_threadgroup]], uint device* ctl_11 [[buffer(1)]], packed_float4 device* calls_11 [[buffer(2)]], packed_float4 device* nodes_11 [[buffer(3)]], LearnedParams_0 constant* params_11 [[buffer(0)]], packed_float4 device* bonds_11 [[buffer(4)]], uint device* chunk_node_11 [[buffer(8)]], uint device* counters_11 [[buffer(19)]], uint device* route_11 [[buffer(25)]], packed_uint4 device* isl_11 [[buffer(5)]], packed_float4 device* islf_11 [[buffer(6)]], Island_natural_0 device* islands_11 [[buffer(11)]], packed_float4 device* ctab_11 [[buffer(14)]], packed_float4 device* bodies_11 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_11 [[buffer(12)]], packed_float4 device* state_11 [[buffer(13)]], packed_float4 device* imps_11 [[buffer(10)]], uint device* status_11 [[buffer(18)]], packed_uint4 device* largs_11 [[buffer(27)]], packed_uint4 device* cisl_11 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_11 [[buffer(17)]], BondStatic_natural_0 device* bond_static_11 [[buffer(16)]], packed_float4 device* frozen_11 [[buffer(9)]], uint device* split_work_11 [[buffer(21)]], uint device* split_control_11 [[buffer(23)]], packed_uint4 device* split_parents_11 [[buffer(20)]], packed_uint4 device* split_slots_11 [[buffer(22)]], packed_float4 device* split_out_11 [[buffer(26)]], packed_uint4 device* split_firsts_11 [[buffer(24)]], float device* lout_11 [[buffer(28)]])
{
    bool plug_0;
    thread KernelContext_0 kernelContext_17;
    (&kernelContext_17)->ctl_0 = ctl_11;
    (&kernelContext_17)->calls_0 = calls_11;
    (&kernelContext_17)->nodes_0 = nodes_11;
    (&kernelContext_17)->params_0 = params_11;
    (&kernelContext_17)->bonds_0 = bonds_11;
    (&kernelContext_17)->chunk_node_0 = chunk_node_11;
    (&kernelContext_17)->counters_0 = counters_11;
    (&kernelContext_17)->route_0 = route_11;
    (&kernelContext_17)->isl_0 = isl_11;
    (&kernelContext_17)->islf_0 = islf_11;
    (&kernelContext_17)->islands_0 = islands_11;
    (&kernelContext_17)->ctab_0 = ctab_11;
    (&kernelContext_17)->bodies_0 = bodies_11;
    (&kernelContext_17)->chunks_0 = chunks_11;
    (&kernelContext_17)->state_0 = state_11;
    (&kernelContext_17)->imps_0 = imps_11;
    (&kernelContext_17)->status_0 = status_11;
    (&kernelContext_17)->largs_0 = largs_11;
    (&kernelContext_17)->cisl_0 = cisl_11;
    (&kernelContext_17)->bond_dyn_0 = bond_dyn_11;
    (&kernelContext_17)->bond_static_0 = bond_static_11;
    (&kernelContext_17)->frozen_0 = frozen_11;
    (&kernelContext_17)->split_work_0 = split_work_11;
    (&kernelContext_17)->split_control_0 = split_control_11;
    (&kernelContext_17)->split_parents_0 = split_parents_11;
    (&kernelContext_17)->split_slots_0 = split_slots_11;
    (&kernelContext_17)->split_out_0 = split_out_11;
    (&kernelContext_17)->split_firsts_0 = split_firsts_11;
    (&kernelContext_17)->lout_0 = lout_11;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_11;
    (&kernelContext_17)->g_red_0 = &g_red_11;
    threadgroup array<float, int(6)> g_lambda_11;
    (&kernelContext_17)->g_lambda_0 = &g_lambda_11;
    threadgroup array<float3, int(256)> g_lo_11;
    (&kernelContext_17)->g_lo_0 = &g_lo_11;
    threadgroup array<float3, int(256)> g_hi_11;
    (&kernelContext_17)->g_hi_0 = &g_hi_11;
    threadgroup array<uint, int(32)> g_label_11;
    (&kernelContext_17)->g_label_0 = &g_label_11;
    threadgroup uint g_used_11;
    (&kernelContext_17)->g_used_0 = &g_used_11;
    threadgroup array<uint, int(32)> g_call_of_11;
    (&kernelContext_17)->g_call_of_0 = &g_call_of_11;
    threadgroup array<uint, int(64)> g_mask_of_11;
    (&kernelContext_17)->g_mask_of_0 = &g_mask_of_11;
    threadgroup array<uint, int(64)> g_cnt_isl_11;
    (&kernelContext_17)->g_cnt_isl_0 = &g_cnt_isl_11;
    threadgroup array<uint, int(64)> g_cnt_chunk_11;
    (&kernelContext_17)->g_cnt_chunk_0 = &g_cnt_chunk_11;
    threadgroup array<uint, int(64)> g_cnt_bond_11;
    (&kernelContext_17)->g_cnt_bond_0 = &g_cnt_bond_11;
    threadgroup array<uint, int(64)> g_cnt_imp_11;
    (&kernelContext_17)->g_cnt_imp_0 = &g_cnt_imp_11;
    threadgroup array<uint, int(256)> g_scan_11;
    (&kernelContext_17)->g_scan_0 = &g_scan_11;
    threadgroup array<float, int(4096)> g_sort_11;
    (&kernelContext_17)->g_sort_0 = &g_sort_11;
    threadgroup uint g_stale_11;
    (&kernelContext_17)->g_stale_0 = &g_stale_11;
    threadgroup uint g_broke_11;
    (&kernelContext_17)->g_broke_0 = &g_broke_11;
    uint g_5 = group_7.x;
    uint t_11 = thread_11.x;
    bool pa_0;
    if(g_5 >= (*(ctl_11+1U)))
    {
        pa_0 = true;
    }
    else
    {
        pa_0 = ((&kernelContext_17)->params_0->stub_0) == 0U;
    }
    uint j_5;
    if(pa_0)
    {
        j_5 = 1U;
    }
    else
    {
        j_5 = 0U;
    }
    if((group_uniform_0(j_5, t_11)) != 0U)
    {
        return;
    }
    uint _S336 = 8U * g_5;
    uint _S337 = _S336 + 2U;
    uint first_0 = (as_type<uint>(((float4(*((&kernelContext_17)->calls_0+_S337)) ).z)));
    uint n_4 = group_uniform_0((as_type<uint>(((float4(*((&kernelContext_17)->calls_0+_S337)) ).w))), t_11);
    uint nc_1 = group_uniform_0((as_type<uint>(((float4(*((&kernelContext_17)->calls_0+(_S336 + 3U))) ).w))), t_11);
    uint _S338 = _S336 + 4U;
    uint _S339 = (as_type<uint>(((float4(*((&kernelContext_17)->calls_0+_S338)) ).x)));
    uint _S340 = (as_type<uint>(((float4(*((&kernelContext_17)->calls_0+_S338)) ).y)));
    float vref_0 = (float4(*((&kernelContext_17)->calls_0+_S336)) ).z;
    j_5 = t_11;
    for(;;)
    {
        if(j_5 < n_4)
        {
        }
        else
        {
            break;
        }
        uint _S341 = 14U * (first_0 + j_5);
        packed_float4 _S342 = packed_float4(float4(0.0f) ) ;
        *((&kernelContext_17)->nodes_0+(_S341 + 5U)) = _S342;
        *((&kernelContext_17)->nodes_0+(_S341 + 6U)) = _S342;
        *((&kernelContext_17)->nodes_0+(_S341 + 7U)) = _S342;
        j_5 = j_5 + 256U;
    }
    j_5 = t_11;
    for(;;)
    {
        if(j_5 < _S340)
        {
        }
        else
        {
            break;
        }
        uint _S343 = 2U * (_S339 + j_5);
        *((&kernelContext_17)->bonds_0+_S343) = packed_float4(float4((float4(*((&kernelContext_17)->bonds_0+_S343)) ).x, -30.0f, 0.0f, (as_type<float>((1U))))) ;
        j_5 = j_5 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    if(((&kernelContext_17)->params_0->stub_0) == 3U)
    {
        j_5 = 1U;
    }
    else
    {
        j_5 = 0U;
    }
    if((group_uniform_0(j_5, t_11)) != 0U)
    {
        float _S344 = float_bits_0(2143289344U);
        j_5 = t_11;
        for(;;)
        {
            if(j_5 < n_4)
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_17)->nodes_0+(14U * (first_0 + j_5) + 5U)) = packed_float4(float4(_S344) ) ;
            j_5 = j_5 + 256U;
        }
        return;
    }
    if(((&kernelContext_17)->params_0->stub_0) == 2U)
    {
        j_5 = 1U;
    }
    else
    {
        j_5 = 0U;
    }
    if((group_uniform_0(j_5, t_11)) == 0U)
    {
        return;
    }
    uint ki_0 = first_0 + nc_1;
    if(n_4 > nc_1)
    {
        j_5 = 1U;
    }
    else
    {
        j_5 = 0U;
    }
    if((group_uniform_0(j_5, t_11)) == 0U)
    {
        return;
    }
    float4 _S345 = node_0(ki_0, 0U, &kernelContext_17);
    float3 _S346 = _S345.xyz;
    float4 _S347 = node_0(ki_0, 3U, &kernelContext_17);
    float3 vi_0 = _S347.xyz;
    float4 _S348 = node_0(ki_0, 0U, &kernelContext_17);
    float mi_0 = _S348.w;
    float4 _S349 = node_0(ki_0, 12U, &kernelContext_17);
    float _S350 = (float4(*((&kernelContext_17)->imps_0+3U * (as_type<uint>((_S349.x))))) ).z;
    float3 _S351 = normalize(vi_0 - (float4(*((&kernelContext_17)->calls_0+(_S336 + 5U))) ).xyz * float3(0.0f) );
    float inf_1 = float_bits_0(2139095040U);
    float best_0 = inf_1;
    j_5 = t_11;
    for(;;)
    {
        if(j_5 < nc_1)
        {
        }
        else
        {
            break;
        }
        uint k_23 = first_0 + j_5;
        float4 _S352 = node_0(k_23, 0U, &kernelContext_17);
        float3 r_13 = _S352.xyz - _S346;
        float along_0 = dot(r_13, _S351);
        float4 _S353 = node_0(k_23, 12U, &kernelContext_17);
        if((length(r_13 - _S351 * float3(along_0) )) <= (_S350 + (float4(((&kernelContext_17)->chunks_0+(as_type<uint>((_S353.x))))->half_0) ).w))
        {
            pa_0 = along_0 > 0.0f;
        }
        else
        {
            pa_0 = false;
        }
        if(pa_0)
        {
            best_0 = min(best_0, along_0);
        }
        j_5 = j_5 + 256U;
    }
    (*(&kernelContext_17)->g_sort_0)[t_11] = best_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s_7 = 128U;
    for(;;)
    {
        if(s_7 > 0U)
        {
        }
        else
        {
            break;
        }
        if(t_11 < s_7)
        {
            (*(&kernelContext_17)->g_sort_0)[t_11] = min((*(&kernelContext_17)->g_sort_0)[t_11], (*(&kernelContext_17)->g_sort_0)[t_11 + s_7]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_7 = s_7 >> 1U;
    }
    float _S354 = (*(&kernelContext_17)->g_sort_0)[int(0)];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 _S355 = float4(0.0f) ;
    j_5 = t_11;
    float4 sa_7 = _S355;
    for(;;)
    {
        if(j_5 < nc_1)
        {
        }
        else
        {
            break;
        }
        uint k_24 = first_0 + j_5;
        float4 _S356 = node_0(k_24, 0U, &kernelContext_17);
        float3 r_14 = _S356.xyz - _S346;
        float along_1 = dot(r_14, _S351);
        float4 _S357 = node_0(k_24, 12U, &kernelContext_17);
        float h_1 = (float4(((&kernelContext_17)->chunks_0+(as_type<uint>((_S357.x))))->half_0) ).w;
        if(_S354 < inf_1)
        {
            pa_0 = along_1 >= (_S354 - h_1);
        }
        else
        {
            pa_0 = false;
        }
        if(pa_0)
        {
            plug_0 = (length(r_14 - _S351 * float3(along_1) )) <= _S350;
        }
        else
        {
            plug_0 = false;
        }
        bool plug_1;
        if(plug_0)
        {
            float4 _S358 = node_0(k_24, 2U, &kernelContext_17);
            plug_1 = (_S358.w) == 0.0f;
        }
        else
        {
            plug_1 = false;
        }
        float4 _S359 = node_0(k_24, 12U, &kernelContext_17);
        (*((&kernelContext_17)->nodes_0+(14U * k_24 + 12U)))[int(1)] = _S359.y;
        if(plug_1)
        {
            float4 _S360 = node_0(k_24, 3U, &kernelContext_17);
            float3 _S361 = _S360.xyz;
            float4 _S362 = node_0(k_24, 0U, &kernelContext_17);
            float3 _S363 = _S361 * float3(_S362.w) ;
            float4 _S364 = node_0(k_24, 0U, &kernelContext_17);
            sa_7 = sa_7 + float4(_S363, _S364.w);
        }
        (*(&kernelContext_17)->g_scan_0)[t_11] = 0U;
        j_5 = j_5 + 256U;
    }
    reduce_store_0(t_11, sa_7, _S355, _S355, _S355, &kernelContext_17);
    thread float4 a_10;
    thread float4 b_9;
    thread float4 c_8;
    thread float4 d_5;
    reduce_read_0(&a_10, &b_9, &c_8, &d_5, &kernelContext_17);
    float mp_0 = a_10.w;
    if(mp_0 > 0.0f)
    {
        j_5 = 1U;
    }
    else
    {
        j_5 = 0U;
    }
    if((group_uniform_0(j_5, t_11)) == 0U)
    {
        return;
    }
    float3 vc_1 = (a_10.xyz + float3(mi_0)  * vi_0) / float3((mp_0 + mi_0)) ;
    j_5 = t_11;
    for(;;)
    {
        if(j_5 < nc_1)
        {
        }
        else
        {
            break;
        }
        uint k_25 = first_0 + j_5;
        float4 _S365 = node_0(k_25, 0U, &kernelContext_17);
        float3 r_15 = _S365.xyz - _S346;
        float along_2 = dot(r_15, _S351);
        float4 _S366 = node_0(k_25, 12U, &kernelContext_17);
        if(along_2 >= (_S354 - (float4(((&kernelContext_17)->chunks_0+(as_type<uint>((_S366.x))))->half_0) ).w))
        {
            pa_0 = (length(r_15 - _S351 * float3(along_2) )) <= _S350;
        }
        else
        {
            pa_0 = false;
        }
        if(pa_0)
        {
            float4 _S367 = node_0(k_25, 2U, &kernelContext_17);
            plug_0 = (_S367.w) == 0.0f;
        }
        else
        {
            plug_0 = false;
        }
        if(plug_0)
        {
            float4 _S368 = node_0(k_25, 1U, &kernelContext_17);
            Quat_0 q_13 = quat_of_0(_S368);
            float4 _S369 = node_0(k_25, 3U, &kernelContext_17);
            float3 _S370 = vc_1 - _S369.xyz;
            thread Quat_0 _S371 = q_13;
            float3 _S372 = inverse_rotate_0(&_S371, _S370);
            float3 o_1 = _S372 / float3(vref_0) ;
            uint _S373 = 14U * k_25;
            *((&kernelContext_17)->nodes_0+(_S373 + 5U)) = packed_float4(float4(o_1, 0.0f)) ;
            *((&kernelContext_17)->nodes_0+(_S373 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.5f * o_1.x, 0.5f * o_1.y)) ;
            *((&kernelContext_17)->nodes_0+(_S373 + 7U)) = packed_float4(float4(0.5f * o_1.z, 0.0f, 0.0f, 0.0f)) ;
            packed_float4 device* _S374 = (&kernelContext_17)->nodes_0+(_S373 + 4U);
            float4 _S375 = node_0(k_25, 4U, &kernelContext_17);
            *_S374 = packed_float4(float4(_S375.xyz, (as_type<float>((2U))))) ;
        }
        j_5 = j_5 + 256U;
    }
    if(t_11 == 0U)
    {
        float4 _S376 = node_0(ki_0, 1U, &kernelContext_17);
        float3 _S377 = vc_1 - vi_0;
        thread Quat_0 _S378 = quat_of_0(_S376);
        float3 _S379 = inverse_rotate_0(&_S378, _S377);
        float3 o_2 = _S379 / float3(vref_0) ;
        uint _S380 = 14U * ki_0;
        *((&kernelContext_17)->nodes_0+(_S380 + 5U)) = packed_float4(float4(o_2, 0.0f)) ;
        *((&kernelContext_17)->nodes_0+(_S380 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.5f * o_2.x, 0.5f * o_2.y)) ;
        *((&kernelContext_17)->nodes_0+(_S380 + 7U)) = packed_float4(float4(0.5f * o_2.z, 0.0f, 0.0f, 0.0f)) ;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    j_5 = t_11;
    for(;;)
    {
        if(j_5 < _S340)
        {
        }
        else
        {
            break;
        }
        uint _S381 = 2U * (_S339 + j_5);
        uint4 _S382 = uint4(((&kernelContext_17)->bond_static_0+(as_type<uint>(((float4(*((&kernelContext_17)->bonds_0+(_S381 + 1U))) ).z))))->law_0.ids_0) ;
        uint device* _S383 = (&kernelContext_17)->chunk_node_0+_S382.y;
        uint na_0 = *_S383;
        uint nb2_0 = *((&kernelContext_17)->chunk_node_0+_S382.z);
        if((*_S383) != 0U)
        {
            pa_0 = ((as_type<uint>(((float4(*((&kernelContext_17)->nodes_0+(14U * (na_0 - 1U) + 4U))) ).w))) & 2U) != 0U;
        }
        else
        {
            pa_0 = false;
        }
        if(nb2_0 != 0U)
        {
            plug_0 = ((as_type<uint>(((float4(*((&kernelContext_17)->nodes_0+(14U * (nb2_0 - 1U) + 4U))) ).w))) & 2U) != 0U;
        }
        else
        {
            plug_0 = false;
        }
        if(pa_0 != plug_0)
        {
            *((&kernelContext_17)->bonds_0+_S381) = packed_float4(float4((float4(*((&kernelContext_17)->bonds_0+_S381)) ).x, 1.0f, 0.0f, (as_type<float>((1U))))) ;
        }
        j_5 = j_5 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    j_5 = t_11;
    for(;;)
    {
        if(j_5 < nc_1)
        {
        }
        else
        {
            break;
        }
        uint k_26 = first_0 + j_5;
        packed_float4 device* _S384 = (&kernelContext_17)->nodes_0+(14U * k_26 + 4U);
        float4 _S385 = node_0(k_26, 4U, &kernelContext_17);
        *_S384 = packed_float4(float4(_S385.xyz, (as_type<float>((0U))))) ;
        j_5 = j_5 + 256U;
    }
    return;
}

[[kernel]] void learned_gate(uint3 group_8 [[threadgroup_position_in_grid]], uint3 thread_12 [[thread_position_in_threadgroup]], uint device* ctl_12 [[buffer(1)]], packed_float4 device* calls_12 [[buffer(2)]], packed_float4 device* nodes_12 [[buffer(3)]], LearnedParams_0 constant* params_12 [[buffer(0)]], packed_float4 device* bonds_12 [[buffer(4)]], uint device* chunk_node_12 [[buffer(8)]], uint device* counters_12 [[buffer(19)]], uint device* route_12 [[buffer(25)]], packed_uint4 device* isl_12 [[buffer(5)]], packed_float4 device* islf_12 [[buffer(6)]], Island_natural_0 device* islands_12 [[buffer(11)]], packed_float4 device* ctab_12 [[buffer(14)]], packed_float4 device* bodies_12 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_12 [[buffer(12)]], packed_float4 device* state_12 [[buffer(13)]], packed_float4 device* imps_12 [[buffer(10)]], uint device* status_12 [[buffer(18)]], packed_uint4 device* largs_12 [[buffer(27)]], packed_uint4 device* cisl_12 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_12 [[buffer(17)]], BondStatic_natural_0 device* bond_static_12 [[buffer(16)]], packed_float4 device* frozen_12 [[buffer(9)]], uint device* split_work_12 [[buffer(21)]], uint device* split_control_12 [[buffer(23)]], packed_uint4 device* split_parents_12 [[buffer(20)]], packed_uint4 device* split_slots_12 [[buffer(22)]], packed_float4 device* split_out_12 [[buffer(26)]], packed_uint4 device* split_firsts_12 [[buffer(24)]], float device* lout_12 [[buffer(28)]])
{
    bool _S386;
    thread KernelContext_0 kernelContext_18;
    (&kernelContext_18)->ctl_0 = ctl_12;
    (&kernelContext_18)->calls_0 = calls_12;
    (&kernelContext_18)->nodes_0 = nodes_12;
    (&kernelContext_18)->params_0 = params_12;
    (&kernelContext_18)->bonds_0 = bonds_12;
    (&kernelContext_18)->chunk_node_0 = chunk_node_12;
    (&kernelContext_18)->counters_0 = counters_12;
    (&kernelContext_18)->route_0 = route_12;
    (&kernelContext_18)->isl_0 = isl_12;
    (&kernelContext_18)->islf_0 = islf_12;
    (&kernelContext_18)->islands_0 = islands_12;
    (&kernelContext_18)->ctab_0 = ctab_12;
    (&kernelContext_18)->bodies_0 = bodies_12;
    (&kernelContext_18)->chunks_0 = chunks_12;
    (&kernelContext_18)->state_0 = state_12;
    (&kernelContext_18)->imps_0 = imps_12;
    (&kernelContext_18)->status_0 = status_12;
    (&kernelContext_18)->largs_0 = largs_12;
    (&kernelContext_18)->cisl_0 = cisl_12;
    (&kernelContext_18)->bond_dyn_0 = bond_dyn_12;
    (&kernelContext_18)->bond_static_0 = bond_static_12;
    (&kernelContext_18)->frozen_0 = frozen_12;
    (&kernelContext_18)->split_work_0 = split_work_12;
    (&kernelContext_18)->split_control_0 = split_control_12;
    (&kernelContext_18)->split_parents_0 = split_parents_12;
    (&kernelContext_18)->split_slots_0 = split_slots_12;
    (&kernelContext_18)->split_out_0 = split_out_12;
    (&kernelContext_18)->split_firsts_0 = split_firsts_12;
    (&kernelContext_18)->lout_0 = lout_12;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_12;
    (&kernelContext_18)->g_red_0 = &g_red_12;
    threadgroup array<float, int(6)> g_lambda_12;
    (&kernelContext_18)->g_lambda_0 = &g_lambda_12;
    threadgroup array<float3, int(256)> g_lo_12;
    (&kernelContext_18)->g_lo_0 = &g_lo_12;
    threadgroup array<float3, int(256)> g_hi_12;
    (&kernelContext_18)->g_hi_0 = &g_hi_12;
    threadgroup array<uint, int(32)> g_label_12;
    (&kernelContext_18)->g_label_0 = &g_label_12;
    threadgroup uint g_used_12;
    (&kernelContext_18)->g_used_0 = &g_used_12;
    threadgroup array<uint, int(32)> g_call_of_12;
    (&kernelContext_18)->g_call_of_0 = &g_call_of_12;
    threadgroup array<uint, int(64)> g_mask_of_12;
    (&kernelContext_18)->g_mask_of_0 = &g_mask_of_12;
    threadgroup array<uint, int(64)> g_cnt_isl_12;
    (&kernelContext_18)->g_cnt_isl_0 = &g_cnt_isl_12;
    threadgroup array<uint, int(64)> g_cnt_chunk_12;
    (&kernelContext_18)->g_cnt_chunk_0 = &g_cnt_chunk_12;
    threadgroup array<uint, int(64)> g_cnt_bond_12;
    (&kernelContext_18)->g_cnt_bond_0 = &g_cnt_bond_12;
    threadgroup array<uint, int(64)> g_cnt_imp_12;
    (&kernelContext_18)->g_cnt_imp_0 = &g_cnt_imp_12;
    threadgroup array<uint, int(256)> g_scan_12;
    (&kernelContext_18)->g_scan_0 = &g_scan_12;
    threadgroup array<float, int(4096)> g_sort_12;
    (&kernelContext_18)->g_sort_0 = &g_sort_12;
    threadgroup uint g_stale_12;
    (&kernelContext_18)->g_stale_0 = &g_stale_12;
    threadgroup uint g_broke_12;
    (&kernelContext_18)->g_broke_0 = &g_broke_12;
    uint g_6 = group_8.x;
    uint t_12 = thread_12.x;
    if(g_6 >= (*(ctl_12+1U)))
    {
        return;
    }
    uint _S387 = 8U * g_6;
    uint _S388 = _S387 + 4U;
    float4 _S389 = float4(*((&kernelContext_18)->calls_0+_S388)) ;
    float _S390 = _S389.z;
    uint dec_1 = (as_type<uint>((_S390)));
    uint fb_0;
    if((dec_1 & 4U) != 0U)
    {
        fb_0 = 1U;
    }
    else
    {
        fb_0 = 0U;
    }
    if((dec_1 & 2U) != 0U)
    {
        fb_0 = fb_0 | 2U;
    }
    uint _S391 = _S387 + 6U;
    uint e0_0 = (as_type<uint>(((float4(*((&kernelContext_18)->calls_0+_S391)) ).x)));
    uint _S392 = (as_type<uint>(((float4(*((&kernelContext_18)->calls_0+_S391)) ).y)));
    uint mask_6 = (as_type<uint>(((float4(*((&kernelContext_18)->calls_0+(_S387 + 5U))) ).w)));
    if(t_12 == 0U)
    {
        *((&kernelContext_18)->calls_0+_S388) = packed_float4(float4(_S389.x, _S389.y, _S390, (as_type<float>((fb_0))))) ;
    }
    uint j_6;
    if(fb_0 != 0U)
    {
        uint _S393 = (as_type<uint>(((float4(*((&kernelContext_18)->calls_0+(_S387 + 2U))) ).z)));
        uint _S394 = (as_type<uint>(((float4(*((&kernelContext_18)->calls_0+(_S387 + 3U))) ).w)));
        j_6 = t_12;
        for(;;)
        {
            if(j_6 < _S394)
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_18)->chunk_node_0+(as_type<uint>(((float4(*((&kernelContext_18)->nodes_0+(14U * (_S393 + j_6) + 12U))) ).x)))) = 0U;
            j_6 = j_6 + 64U;
        }
        return;
    }
    j_6 = e0_0 + t_12;
    for(;;)
    {
        if(j_6 < (e0_0 + _S392))
        {
        }
        else
        {
            break;
        }
        uint _S395 = 2U * j_6;
        uint4 _S396 = uint4(*((&kernelContext_18)->cisl_0+(_S395 + 1U))) ;
        uint device* _S397 = (&kernelContext_18)->status_0+(16U + (uint4(*((&kernelContext_18)->cisl_0+_S395)) ).x);
        *_S397 = (*_S397) | 4096U;
        uint body_2 = _S396.x;
        if(body_2 == 4294967295U)
        {
            _S386 = true;
        }
        else
        {
            _S386 = (_S396.w) != 0U;
        }
        if(_S386)
        {
            *((&kernelContext_18)->frozen_0+4U * j_6) = packed_float4(float4((as_type<float>((4294967295U))), 0.0f, 0.0f, 0.0f)) ;
            j_6 = j_6 + 64U;
            continue;
        }
        uint at_1 = 7U * body_2;
        uint _S398 = 4U * j_6;
        *((&kernelContext_18)->frozen_0+_S398) = packed_float4(float4((as_type<float>((body_2))), (float4(*((&kernelContext_18)->bodies_0+at_1)) ).w, 0.0f, 0.0f)) ;
        uint _S399 = at_1 + 2U;
        *((&kernelContext_18)->frozen_0+(_S398 + 1U)) = packed_float4(float4(*((&kernelContext_18)->bodies_0+_S399)) ) ;
        uint _S400 = at_1 + 3U;
        *((&kernelContext_18)->frozen_0+(_S398 + 2U)) = packed_float4(float4(*((&kernelContext_18)->bodies_0+_S400)) ) ;
        uint _S401 = at_1 + 4U;
        *((&kernelContext_18)->frozen_0+(_S398 + 3U)) = packed_float4(float4(*((&kernelContext_18)->bodies_0+_S401)) ) ;
        *((&kernelContext_18)->bodies_0+at_1) = packed_float4(float4((float4(*((&kernelContext_18)->bodies_0+at_1)) ).xyz, 0.0f)) ;
        *((&kernelContext_18)->bodies_0+_S399) = packed_float4(float4(0.0f, 0.0f, 0.0f, (float4(*((&kernelContext_18)->bodies_0+_S399)) ).w)) ;
        *((&kernelContext_18)->bodies_0+_S400) = packed_float4(float4(0.0f, 0.0f, 0.0f, (float4(*((&kernelContext_18)->bodies_0+_S400)) ).w)) ;
        *((&kernelContext_18)->bodies_0+_S401) = packed_float4(float4(0.0f, 0.0f, 0.0f, (float4(*((&kernelContext_18)->bodies_0+_S401)) ).w)) ;
        j_6 = j_6 + 64U;
    }
    if(t_12 < 32U)
    {
        _S386 = (mask_6 & (1U << t_12)) != 0U;
    }
    else
    {
        _S386 = false;
    }
    if(_S386)
    {
        uint body_3 = (as_type<uint>(((float4(*((&kernelContext_18)->imps_0+3U * t_12)) ).x)));
        uint at_2 = 7U * body_3;
        uint _S402 = 4U * ((&kernelContext_18)->params_0->max_cisl_0 + t_12);
        *((&kernelContext_18)->frozen_0+_S402) = packed_float4(float4((as_type<float>((body_3))), (float4(*((&kernelContext_18)->bodies_0+at_2)) ).w, 0.0f, 0.0f)) ;
        uint _S403 = at_2 + 2U;
        *((&kernelContext_18)->frozen_0+(_S402 + 1U)) = packed_float4(float4(*((&kernelContext_18)->bodies_0+_S403)) ) ;
        uint _S404 = at_2 + 3U;
        *((&kernelContext_18)->frozen_0+(_S402 + 2U)) = packed_float4(float4(*((&kernelContext_18)->bodies_0+_S404)) ) ;
        uint _S405 = at_2 + 4U;
        *((&kernelContext_18)->frozen_0+(_S402 + 3U)) = packed_float4(float4(*((&kernelContext_18)->bodies_0+_S405)) ) ;
        *((&kernelContext_18)->bodies_0+at_2) = packed_float4(float4((float4(*((&kernelContext_18)->bodies_0+at_2)) ).xyz, 0.0f)) ;
        *((&kernelContext_18)->bodies_0+_S403) = packed_float4(float4(0.0f, 0.0f, 0.0f, (float4(*((&kernelContext_18)->bodies_0+_S403)) ).w)) ;
        *((&kernelContext_18)->bodies_0+_S404) = packed_float4(float4(0.0f, 0.0f, 0.0f, (float4(*((&kernelContext_18)->bodies_0+_S404)) ).w)) ;
        *((&kernelContext_18)->bodies_0+_S405) = packed_float4(float4(0.0f, 0.0f, 0.0f, (float4(*((&kernelContext_18)->bodies_0+_S405)) ).w)) ;
    }
    return;
}

bool island_current_0(uint e_2, KernelContext_0 thread* kernelContext_19)
{
    uint _S406;
    uint4 _S407 = uint4(*(kernelContext_19->cisl_0+2U * e_2)) ;
    uint _S408 = _S407.y;
    uint island_0 = 4294967295U;
    uint j_7 = 0U;
    for(;;)
    {
        uint _S409 = _S407.z;
        _S406 = _S409;
        if(j_7 < _S409)
        {
        }
        else
        {
            break;
        }
        float4 _S410 = float4(*(kernelContext_19->nodes_0+(14U * (_S408 + j_7) + 12U))) ;
        uint k_27 = (as_type<uint>((_S410.x)));
        if(k_27 >= (kernelContext_19->params_0->chunk_count_0))
        {
            return false;
        }
        uint4 _S411 = uint4((kernelContext_19->chunks_0+k_27)->info_1) ;
        bool _S412;
        if((_S411.z) != (as_type<uint>((_S410.z))))
        {
            _S412 = true;
        }
        else
        {
            _S412 = (_S411.w) != (as_type<uint>((_S410.w)));
        }
        if(_S412)
        {
            return false;
        }
        if(j_7 == 0U)
        {
            island_0 = _S411.y;
        }
        else
        {
            if((_S411.y) != island_0)
            {
                return false;
            }
        }
        j_7 = j_7 + 1U;
    }
    if(_S406 == 0U)
    {
        return false;
    }
    uint4 _S413 = uint4((kernelContext_19->islands_0+island_0)->range_0) ;
    return (_S413.y - _S413.x) == _S406;
}

[[kernel]] void learned_check(uint3 group_9 [[threadgroup_position_in_grid]], uint3 thread_13 [[thread_position_in_threadgroup]], uint device* ctl_13 [[buffer(1)]], packed_float4 device* calls_13 [[buffer(2)]], packed_float4 device* nodes_13 [[buffer(3)]], LearnedParams_0 constant* params_13 [[buffer(0)]], packed_float4 device* bonds_13 [[buffer(4)]], uint device* chunk_node_13 [[buffer(8)]], uint device* counters_13 [[buffer(19)]], uint device* route_13 [[buffer(25)]], packed_uint4 device* isl_13 [[buffer(5)]], packed_float4 device* islf_13 [[buffer(6)]], Island_natural_0 device* islands_13 [[buffer(11)]], packed_float4 device* ctab_13 [[buffer(14)]], packed_float4 device* bodies_13 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_13 [[buffer(12)]], packed_float4 device* state_13 [[buffer(13)]], packed_float4 device* imps_13 [[buffer(10)]], uint device* status_13 [[buffer(18)]], packed_uint4 device* largs_13 [[buffer(27)]], packed_uint4 device* cisl_13 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_13 [[buffer(17)]], BondStatic_natural_0 device* bond_static_13 [[buffer(16)]], packed_float4 device* frozen_13 [[buffer(9)]], uint device* split_work_13 [[buffer(21)]], uint device* split_control_13 [[buffer(23)]], packed_uint4 device* split_parents_13 [[buffer(20)]], packed_uint4 device* split_slots_13 [[buffer(22)]], packed_float4 device* split_out_13 [[buffer(26)]], packed_uint4 device* split_firsts_13 [[buffer(24)]], float device* lout_13 [[buffer(28)]])
{
    thread KernelContext_0 kernelContext_20;
    (&kernelContext_20)->ctl_0 = ctl_13;
    (&kernelContext_20)->calls_0 = calls_13;
    (&kernelContext_20)->nodes_0 = nodes_13;
    (&kernelContext_20)->params_0 = params_13;
    (&kernelContext_20)->bonds_0 = bonds_13;
    (&kernelContext_20)->chunk_node_0 = chunk_node_13;
    (&kernelContext_20)->counters_0 = counters_13;
    (&kernelContext_20)->route_0 = route_13;
    (&kernelContext_20)->isl_0 = isl_13;
    (&kernelContext_20)->islf_0 = islf_13;
    (&kernelContext_20)->islands_0 = islands_13;
    (&kernelContext_20)->ctab_0 = ctab_13;
    (&kernelContext_20)->bodies_0 = bodies_13;
    (&kernelContext_20)->chunks_0 = chunks_13;
    (&kernelContext_20)->state_0 = state_13;
    (&kernelContext_20)->imps_0 = imps_13;
    (&kernelContext_20)->status_0 = status_13;
    (&kernelContext_20)->largs_0 = largs_13;
    (&kernelContext_20)->cisl_0 = cisl_13;
    (&kernelContext_20)->bond_dyn_0 = bond_dyn_13;
    (&kernelContext_20)->bond_static_0 = bond_static_13;
    (&kernelContext_20)->frozen_0 = frozen_13;
    (&kernelContext_20)->split_work_0 = split_work_13;
    (&kernelContext_20)->split_control_0 = split_control_13;
    (&kernelContext_20)->split_parents_0 = split_parents_13;
    (&kernelContext_20)->split_slots_0 = split_slots_13;
    (&kernelContext_20)->split_out_0 = split_out_13;
    (&kernelContext_20)->split_firsts_0 = split_firsts_13;
    (&kernelContext_20)->lout_0 = lout_13;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_13;
    (&kernelContext_20)->g_red_0 = &g_red_13;
    threadgroup array<float, int(6)> g_lambda_13;
    (&kernelContext_20)->g_lambda_0 = &g_lambda_13;
    threadgroup array<float3, int(256)> g_lo_13;
    (&kernelContext_20)->g_lo_0 = &g_lo_13;
    threadgroup array<float3, int(256)> g_hi_13;
    (&kernelContext_20)->g_hi_0 = &g_hi_13;
    threadgroup array<uint, int(32)> g_label_13;
    (&kernelContext_20)->g_label_0 = &g_label_13;
    threadgroup uint g_used_13;
    (&kernelContext_20)->g_used_0 = &g_used_13;
    threadgroup array<uint, int(32)> g_call_of_13;
    (&kernelContext_20)->g_call_of_0 = &g_call_of_13;
    threadgroup array<uint, int(64)> g_mask_of_13;
    (&kernelContext_20)->g_mask_of_0 = &g_mask_of_13;
    threadgroup array<uint, int(64)> g_cnt_isl_13;
    (&kernelContext_20)->g_cnt_isl_0 = &g_cnt_isl_13;
    threadgroup array<uint, int(64)> g_cnt_chunk_13;
    (&kernelContext_20)->g_cnt_chunk_0 = &g_cnt_chunk_13;
    threadgroup array<uint, int(64)> g_cnt_bond_13;
    (&kernelContext_20)->g_cnt_bond_0 = &g_cnt_bond_13;
    threadgroup array<uint, int(64)> g_cnt_imp_13;
    (&kernelContext_20)->g_cnt_imp_0 = &g_cnt_imp_13;
    threadgroup array<uint, int(256)> g_scan_13;
    (&kernelContext_20)->g_scan_0 = &g_scan_13;
    threadgroup array<float, int(4096)> g_sort_13;
    (&kernelContext_20)->g_sort_0 = &g_sort_13;
    threadgroup uint g_stale_13;
    (&kernelContext_20)->g_stale_0 = &g_stale_13;
    threadgroup uint g_broke_13;
    (&kernelContext_20)->g_broke_0 = &g_broke_13;
    uint g_7 = group_9.x;
    uint t_13 = thread_13.x;
    bool _S414;
    if((*(ctl_13+0U)) == 0U)
    {
        _S414 = true;
    }
    else
    {
        _S414 = g_7 >= (*((&kernelContext_20)->ctl_0+1U));
    }
    uint e_3;
    if(_S414)
    {
        e_3 = 1U;
    }
    else
    {
        e_3 = 0U;
    }
    if((group_uniform_0(e_3, t_13)) != 0U)
    {
        return;
    }
    uint _S415 = 8U * g_7;
    uint _S416 = _S415 + 4U;
    float4 _S417 = float4(*((&kernelContext_20)->calls_0+_S416)) ;
    if((group_uniform_0((as_type<uint>((_S417.w))), t_13)) != 0U)
    {
        return;
    }
    uint _S418 = _S415 + 6U;
    uint e0_1 = (as_type<uint>(((float4(*((&kernelContext_20)->calls_0+_S418)) ).x)));
    uint _S419 = (as_type<uint>(((float4(*((&kernelContext_20)->calls_0+_S418)) ).y)));
    bool _S420 = t_13 == 0U;
    if(_S420)
    {
        *(&kernelContext_20)->g_stale_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    e_3 = e0_1 + t_13;
    for(;;)
    {
        if(e_3 < (e0_1 + _S419))
        {
        }
        else
        {
            break;
        }
        bool _S421 = island_current_0(e_3, &kernelContext_20);
        if(!_S421)
        {
            *(&kernelContext_20)->g_stale_0 = 1U;
        }
        e_3 = e_3 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if((group_uniform_0(*(&kernelContext_20)->g_stale_0, t_13)) == 0U)
    {
        return;
    }
    if(_S420)
    {
        *((&kernelContext_20)->calls_0+_S416) = packed_float4(float4(_S417.x, _S417.y, _S417.z, (as_type<float>((8U))))) ;
        float _S422 = (as_type<float>((0U)));
        *((&kernelContext_20)->calls_0+(_S415 + 7U)) = packed_float4(float4((as_type<float>((1U))), _S422, _S422, 0.0f)) ;
    }
    uint _S423 = (as_type<uint>(((float4(*((&kernelContext_20)->calls_0+(_S415 + 2U))) ).z)));
    uint _S424 = (as_type<uint>(((float4(*((&kernelContext_20)->calls_0+(_S415 + 3U))) ).w)));
    uint j_8 = t_13;
    for(;;)
    {
        if(j_8 < _S424)
        {
        }
        else
        {
            break;
        }
        uint k_28 = (as_type<uint>(((float4(*((&kernelContext_20)->nodes_0+(14U * (_S423 + j_8) + 12U))) ).x)));
        if(k_28 < ((&kernelContext_20)->params_0->chunk_count_0))
        {
            *((&kernelContext_20)->chunk_node_0+k_28) = 0U;
        }
        j_8 = j_8 + 256U;
    }
    return;
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

[[kernel]] void learned_apply_bonds(uint3 group_10 [[threadgroup_position_in_grid]], uint3 thread_14 [[thread_position_in_threadgroup]], uint device* ctl_14 [[buffer(1)]], packed_float4 device* calls_14 [[buffer(2)]], packed_float4 device* nodes_14 [[buffer(3)]], LearnedParams_0 constant* params_14 [[buffer(0)]], packed_float4 device* bonds_14 [[buffer(4)]], uint device* chunk_node_14 [[buffer(8)]], uint device* counters_14 [[buffer(19)]], uint device* route_14 [[buffer(25)]], packed_uint4 device* isl_14 [[buffer(5)]], packed_float4 device* islf_14 [[buffer(6)]], Island_natural_0 device* islands_14 [[buffer(11)]], packed_float4 device* ctab_14 [[buffer(14)]], packed_float4 device* bodies_14 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_14 [[buffer(12)]], packed_float4 device* state_14 [[buffer(13)]], packed_float4 device* imps_14 [[buffer(10)]], uint device* status_14 [[buffer(18)]], packed_uint4 device* largs_14 [[buffer(27)]], packed_uint4 device* cisl_14 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_14 [[buffer(17)]], BondStatic_natural_0 device* bond_static_14 [[buffer(16)]], packed_float4 device* frozen_14 [[buffer(9)]], uint device* split_work_14 [[buffer(21)]], uint device* split_control_14 [[buffer(23)]], packed_uint4 device* split_parents_14 [[buffer(20)]], packed_uint4 device* split_slots_14 [[buffer(22)]], packed_float4 device* split_out_14 [[buffer(26)]], packed_uint4 device* split_firsts_14 [[buffer(24)]], float device* lout_14 [[buffer(28)]])
{
    uint count_0;
    thread KernelContext_0 kernelContext_21;
    (&kernelContext_21)->ctl_0 = ctl_14;
    (&kernelContext_21)->calls_0 = calls_14;
    (&kernelContext_21)->nodes_0 = nodes_14;
    (&kernelContext_21)->params_0 = params_14;
    (&kernelContext_21)->bonds_0 = bonds_14;
    (&kernelContext_21)->chunk_node_0 = chunk_node_14;
    (&kernelContext_21)->counters_0 = counters_14;
    (&kernelContext_21)->route_0 = route_14;
    (&kernelContext_21)->isl_0 = isl_14;
    (&kernelContext_21)->islf_0 = islf_14;
    (&kernelContext_21)->islands_0 = islands_14;
    (&kernelContext_21)->ctab_0 = ctab_14;
    (&kernelContext_21)->bodies_0 = bodies_14;
    (&kernelContext_21)->chunks_0 = chunks_14;
    (&kernelContext_21)->state_0 = state_14;
    (&kernelContext_21)->imps_0 = imps_14;
    (&kernelContext_21)->status_0 = status_14;
    (&kernelContext_21)->largs_0 = largs_14;
    (&kernelContext_21)->cisl_0 = cisl_14;
    (&kernelContext_21)->bond_dyn_0 = bond_dyn_14;
    (&kernelContext_21)->bond_static_0 = bond_static_14;
    (&kernelContext_21)->frozen_0 = frozen_14;
    (&kernelContext_21)->split_work_0 = split_work_14;
    (&kernelContext_21)->split_control_0 = split_control_14;
    (&kernelContext_21)->split_parents_0 = split_parents_14;
    (&kernelContext_21)->split_slots_0 = split_slots_14;
    (&kernelContext_21)->split_out_0 = split_out_14;
    (&kernelContext_21)->split_firsts_0 = split_firsts_14;
    (&kernelContext_21)->lout_0 = lout_14;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_14;
    (&kernelContext_21)->g_red_0 = &g_red_14;
    threadgroup array<float, int(6)> g_lambda_14;
    (&kernelContext_21)->g_lambda_0 = &g_lambda_14;
    threadgroup array<float3, int(256)> g_lo_14;
    (&kernelContext_21)->g_lo_0 = &g_lo_14;
    threadgroup array<float3, int(256)> g_hi_14;
    (&kernelContext_21)->g_hi_0 = &g_hi_14;
    threadgroup array<uint, int(32)> g_label_14;
    (&kernelContext_21)->g_label_0 = &g_label_14;
    threadgroup uint g_used_14;
    (&kernelContext_21)->g_used_0 = &g_used_14;
    threadgroup array<uint, int(32)> g_call_of_14;
    (&kernelContext_21)->g_call_of_0 = &g_call_of_14;
    threadgroup array<uint, int(64)> g_mask_of_14;
    (&kernelContext_21)->g_mask_of_0 = &g_mask_of_14;
    threadgroup array<uint, int(64)> g_cnt_isl_14;
    (&kernelContext_21)->g_cnt_isl_0 = &g_cnt_isl_14;
    threadgroup array<uint, int(64)> g_cnt_chunk_14;
    (&kernelContext_21)->g_cnt_chunk_0 = &g_cnt_chunk_14;
    threadgroup array<uint, int(64)> g_cnt_bond_14;
    (&kernelContext_21)->g_cnt_bond_0 = &g_cnt_bond_14;
    threadgroup array<uint, int(64)> g_cnt_imp_14;
    (&kernelContext_21)->g_cnt_imp_0 = &g_cnt_imp_14;
    threadgroup array<uint, int(256)> g_scan_14;
    (&kernelContext_21)->g_scan_0 = &g_scan_14;
    threadgroup array<float, int(4096)> g_sort_14;
    (&kernelContext_21)->g_sort_0 = &g_sort_14;
    threadgroup uint g_stale_14;
    (&kernelContext_21)->g_stale_0 = &g_stale_14;
    threadgroup uint g_broke_14;
    (&kernelContext_21)->g_broke_0 = &g_broke_14;
    uint g_8 = group_10.x;
    uint t_14 = thread_14.x;
    bool _S425;
    if((*(ctl_14+0U)) == 0U)
    {
        _S425 = true;
    }
    else
    {
        _S425 = g_8 >= (*((&kernelContext_21)->ctl_0+1U));
    }
    uint j_9;
    if(_S425)
    {
        j_9 = 1U;
    }
    else
    {
        j_9 = 0U;
    }
    if((group_uniform_0(j_9, t_14)) != 0U)
    {
        return;
    }
    uint _S426 = 8U * g_8;
    float4 _S427 = float4(*((&kernelContext_21)->calls_0+(_S426 + 4U))) ;
    if((group_uniform_0((as_type<uint>((_S427.w))), t_14)) != 0U)
    {
        return;
    }
    bool _S428 = t_14 == 0U;
    if(_S428)
    {
        *(&kernelContext_21)->g_broke_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S429 = (as_type<uint>((_S427.x)));
    uint _S430 = (as_type<uint>((_S427.y)));
    j_9 = t_14;
    uint count_1 = 0U;
    for(;;)
    {
        if(j_9 < _S430)
        {
        }
        else
        {
            break;
        }
        float4 _S431 = float4(*((&kernelContext_21)->bonds_0+(2U * (_S429 + j_9) + 1U))) ;
        uint bi_2 = (as_type<uint>((_S431.z)));
        BondDyn_natural_0 device* _S432 = (&kernelContext_21)->bond_dyn_0+bi_2;
        float4 _S433 = float4((*_S432).force_lin_0) ;
        float4 _S434 = float4((*_S432).force_ang_0) ;
        float4 _S435 = float4((*_S432).sums_0) ;
        float4 _S436 = float4((*_S432).comps_0) ;
        uint4 _S437 = uint4((*_S432).events_0) ;
        thread BondDyn_0 dyn_0;
        (&dyn_0)->js_0 = (*_S432).js_0;
        (&dyn_0)->force_lin_0 = _S433;
        (&dyn_0)->force_ang_0 = _S434;
        (&dyn_0)->sums_0 = _S435;
        (&dyn_0)->comps_0 = _S436;
        (&dyn_0)->events_0 = _S437;
        if((as_type<uint>((_S431.y))) != 0U)
        {
            bool _S438 = ((float4(((&kernelContext_21)->bond_static_0+bi_2)->law_0.stiff1_0) ).w) != 0.0f;
            thread JointState_0 _S439 = (&dyn_0)->js_0;
            bool _S440 = connected_0(&_S439, _S438);
            if(_S440)
            {
                (&(&dyn_0)->js_0)->damage_0 = 1.0f;
                (&(&dyn_0)->js_0)->rebar_broken_0 = 1.0f;
                (&dyn_0)->events_0.z = max((&dyn_0)->events_0.z, 1U);
                BondDyn_natural_0 device* _S441 = (&kernelContext_21)->bond_dyn_0+bi_2;
                _S441->js_0 = dyn_0.js_0;
                _S441->force_lin_0 = packed_float4(dyn_0.force_lin_0) ;
                _S441->force_ang_0 = packed_float4(dyn_0.force_ang_0) ;
                _S441->sums_0 = packed_float4(dyn_0.sums_0) ;
                _S441->comps_0 = packed_float4(dyn_0.comps_0) ;
                _S441->events_0 = packed_uint4(dyn_0.events_0) ;
                uint _S442 = count_1 + 1U;
                uint device* _S443 = (&kernelContext_21)->status_0+(16U + (uint4(((&kernelContext_21)->chunks_0+(uint4(((&kernelContext_21)->bond_static_0+bi_2)->law_0.ids_0) ).y)->info_1) ).y);
                *_S443 = (*_S443) | 8U;
                *(&kernelContext_21)->g_broke_0 = 1U;
                count_0 = _S442;
            }
            else
            {
                count_0 = count_1;
            }
            count_1 = count_0;
        }
        else
        {
            (&(&dyn_0)->js_0)->damage_0 = max((&(&dyn_0)->js_0)->damage_0, min(_S431.x, 0.99999898672103882f));
            BondDyn_natural_0 device* _S444 = (&kernelContext_21)->bond_dyn_0+bi_2;
            _S444->js_0 = dyn_0.js_0;
            _S444->force_lin_0 = packed_float4(dyn_0.force_lin_0) ;
            _S444->force_ang_0 = packed_float4(dyn_0.force_ang_0) ;
            _S444->sums_0 = packed_float4(dyn_0.sums_0) ;
            _S444->comps_0 = packed_float4(dyn_0.comps_0) ;
            _S444->events_0 = packed_uint4(dyn_0.events_0) ;
        }
        j_9 = j_9 + 256U;
    }
    (*(&kernelContext_21)->g_scan_0)[t_14] = count_1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    count_0 = 128U;
    for(;;)
    {
        if(count_0 > 0U)
        {
        }
        else
        {
            break;
        }
        if(t_14 < count_0)
        {
            (*(&kernelContext_21)->g_scan_0)[t_14] = (*(&kernelContext_21)->g_scan_0)[t_14] + (*(&kernelContext_21)->g_scan_0)[t_14 + count_0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        count_0 = count_0 >> 1U;
    }
    if(_S428)
    {
        *((&kernelContext_21)->calls_0+(_S426 + 7U)) = packed_float4(float4((as_type<float>((0U))), (as_type<float>(((*(&kernelContext_21)->g_scan_0)[int(0)]))), (as_type<float>((1U))), 0.0f)) ;
        if((*(&kernelContext_21)->g_broke_0) != 0U)
        {
            *((&kernelContext_21)->status_0+1U) = 1U;
        }
    }
    return;
}

float4 rotvec_quat_0(float3 r_16)
{
    float th_1 = length(r_16);
    if(th_1 < 9.999999960041972e-13f)
    {
        return normalize(float4(float3(0.5f)  * r_16, 1.0f));
    }
    float _S445 = 0.5f * th_1;
    return float4(r_16 / float3(th_1)  * float3(sin(_S445)) , cos(_S445));
}

struct Fit_0
{
    float3 com0_0;
    float3 com1_0;
    float4 turn_0;
    float3 v_9;
    float3 w_6;
};

Fit_0 fit_nodes_0(uint first_1, uint n_5, bool listed_0, uint list_base_0, uint list_first_0, KernelContext_0 thread* kernelContext_22)
{
    uint k_29;
    float3 _S446 = float3(0.0f) ;
    uint j_10 = 0U;
    float m_10 = 0.0f;
    float3 x0_0 = _S446;
    float3 x1_0 = _S446;
    float3 rot_0 = _S446;
    float3 p_1 = _S446;
    for(;;)
    {
        if(j_10 < n_5)
        {
        }
        else
        {
            break;
        }
        if(listed_0)
        {
            k_29 = *(kernelContext_22->chunk_node_0+(list_first_0 + kernelContext_22->split_work_0[list_base_0 + j_10])) - 1U;
        }
        else
        {
            k_29 = first_1 + j_10;
        }
        float4 _S447 = node_0(k_29, 0U, kernelContext_22);
        float mc_0 = _S447.w;
        float4 _S448 = node_0(k_29, 13U, kernelContext_22);
        float3 xs_0 = _S448.xyz;
        float m_11 = m_10 + mc_0;
        float3 _S449 = float3(mc_0) ;
        float3 x0_1 = x0_0 + xs_0 * _S449;
        float4 _S450 = node_0(k_29, 10U, kernelContext_22);
        float3 x1_1 = x1_0 + (xs_0 + _S450.xyz) * _S449;
        float4 _S451 = node_0(k_29, 11U, kernelContext_22);
        float3 rot_1 = rot_0 + _S451.xyz * _S449;
        float4 _S452 = node_0(k_29, 8U, kernelContext_22);
        float3 p_2 = p_1 + _S452.xyz * _S449;
        j_10 = j_10 + 1U;
        m_10 = m_11;
        x0_0 = x0_1;
        x1_0 = x1_1;
        rot_0 = rot_1;
        p_1 = p_2;
    }
    thread Fit_0 f_2;
    (&f_2)->com0_0 = x0_0 / float3(m_10) ;
    (&f_2)->com1_0 = x1_0 / float3(m_10) ;
    float4 _S453 = rotvec_quat_0(rot_0 / float3(m_10) );
    (&f_2)->turn_0 = _S453;
    (&f_2)->v_9 = p_1 / float3(m_10) ;
    Quat_0 _S454 = quat_of_0(_S453);
    matrix<float,int(3),int(3)>  _S455 = quat_matrix_0(_S453);
    matrix<float,int(3),int(3)>  _S456 = matrix<float,int(3),int(3)> (_S446, _S446, _S446);
    j_10 = 0U;
    float3 l_0 = _S446;
    matrix<float,int(3),int(3)>  itot_0 = _S456;
    for(;;)
    {
        if(j_10 < n_5)
        {
        }
        else
        {
            break;
        }
        if(listed_0)
        {
            k_29 = *(kernelContext_22->chunk_node_0+(list_first_0 + kernelContext_22->split_work_0[list_base_0 + j_10])) - 1U;
        }
        else
        {
            k_29 = first_1 + j_10;
        }
        float4 _S457 = node_0(k_29, 0U, kernelContext_22);
        float mc_1 = _S457.w;
        float4 _S458 = node_0(k_29, 13U, kernelContext_22);
        float3 _S459 = _S458.xyz - (&f_2)->com0_0;
        thread Quat_0 _S460 = _S454;
        float3 _S461 = rotate_0(&_S460, _S459);
        float4 _S462 = node_0(k_29, 1U, kernelContext_22);
        float4 _S463 = node_0(k_29, 2U, kernelContext_22);
        matrix<float,int(3),int(3)>  ic_1 = (((transpose(_S455)) * ((((world_inertia_0(_S462, _S463.xyz)) * (_S455))))));
        float4 _S464 = node_0(k_29, 8U, kernelContext_22);
        float3 _S465 = cross(_S461, float3(mc_1)  * _S464.xyz);
        float4 _S466 = node_0(k_29, 9U, kernelContext_22);
        float3 l_1 = l_0 + (_S465 + (((_S466.xyz) * (ic_1))));
        matrix<float,int(3),int(3)>  _S467 = itot_0 + (ic_1 + _slang_matrixCompMult(matrix<float,int(3),int(3)> (mc_1, mc_1, mc_1, mc_1, mc_1, mc_1, mc_1, mc_1, mc_1), _slang_matrixCompMult(matrix<float,int(3),int(3)> (dot(_S461, _S461), dot(_S461, _S461), dot(_S461, _S461), dot(_S461, _S461), dot(_S461, _S461), dot(_S461, _S461), dot(_S461, _S461), dot(_S461, _S461), dot(_S461, _S461)), matrix<float,int(3),int(3)> (float3(1.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), float3(0.0f, 0.0f, 1.0f))) - outer3_0(_S461, _S461)));
        j_10 = j_10 + 1U;
        l_0 = l_1;
        itot_0 = _S467;
    }
    (&f_2)->w_6 = solve3_0(itot_0, l_0);
    return f_2;
}

float3 to_yup_0(float3 v_10)
{
    return float3(v_10.x, v_10.z, - v_10.y);
}

float4 turn_rubble_0(float4 q_rubble_1, float3 rotvec_zup_0)
{
    return normalize(qmul_0(rotvec_quat_0(to_yup_0(rotvec_zup_0)), q_rubble_1));
}

void place_body_0(uint body_4, const Fit_0 thread* f_3, float4 held_mass_0, float4 held_inertia_0, KernelContext_0 thread* kernelContext_23)
{
    uint at_3 = 7U * body_4;
    float4 _S468 = f_3->turn_0;
    float3 _S469 = to_zup_0((float4(*(kernelContext_23->bodies_0+at_3)) ).xyz) - f_3->com0_0;
    thread Quat_0 _S470 = quat_of_0(f_3->turn_0);
    float3 _S471 = rotate_0(&_S470, _S469);
    *(kernelContext_23->bodies_0+at_3) = packed_float4(float4(to_yup_0(f_3->com1_0 + _S471), held_mass_0.y)) ;
    uint _S472 = at_3 + 1U;
    packed_float4 device* _S473 = kernelContext_23->bodies_0+_S472;
    float4 _S474 = float4(*(kernelContext_23->bodies_0+_S472)) ;
    float3 _S475 = _S468.xyz;
    float _S476 = length(_S475);
    float3 _S477;
    if(_S476 > 0.0f)
    {
        _S477 = float3((2.0f * atan2(_S476, _S468.w)))  * normalize(_S475);
    }
    else
    {
        _S477 = float3(0.0f) ;
    }
    *_S473 = packed_float4(turn_rubble_0(_S474, _S477)) ;
    *(kernelContext_23->bodies_0+(at_3 + 2U)) = packed_float4(held_inertia_0) ;
    *(kernelContext_23->bodies_0+(at_3 + 3U)) = packed_float4(float4(to_yup_0(f_3->v_9), 0.0f)) ;
    uint _S478 = at_3 + 4U;
    *(kernelContext_23->bodies_0+_S478) = packed_float4(float4(to_yup_0(f_3->w_6), (float4(*(kernelContext_23->bodies_0+_S478)) ).w)) ;
    return;
}

[[kernel]] void learned_apply_bodies(uint3 id_2 [[thread_position_in_grid]], uint device* ctl_15 [[buffer(1)]], packed_float4 device* calls_15 [[buffer(2)]], packed_float4 device* nodes_15 [[buffer(3)]], LearnedParams_0 constant* params_15 [[buffer(0)]], packed_float4 device* bonds_15 [[buffer(4)]], uint device* chunk_node_15 [[buffer(8)]], uint device* counters_15 [[buffer(19)]], uint device* route_15 [[buffer(25)]], packed_uint4 device* isl_15 [[buffer(5)]], packed_float4 device* islf_15 [[buffer(6)]], Island_natural_0 device* islands_15 [[buffer(11)]], packed_float4 device* ctab_15 [[buffer(14)]], packed_float4 device* bodies_15 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_15 [[buffer(12)]], packed_float4 device* state_15 [[buffer(13)]], packed_float4 device* imps_15 [[buffer(10)]], uint device* status_15 [[buffer(18)]], packed_uint4 device* largs_15 [[buffer(27)]], packed_uint4 device* cisl_15 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_15 [[buffer(17)]], BondStatic_natural_0 device* bond_static_15 [[buffer(16)]], packed_float4 device* frozen_15 [[buffer(9)]], uint device* split_work_15 [[buffer(21)]], uint device* split_control_15 [[buffer(23)]], packed_uint4 device* split_parents_15 [[buffer(20)]], packed_uint4 device* split_slots_15 [[buffer(22)]], packed_float4 device* split_out_15 [[buffer(26)]], packed_uint4 device* split_firsts_15 [[buffer(24)]], float device* lout_15 [[buffer(28)]])
{
    thread KernelContext_0 kernelContext_24;
    (&kernelContext_24)->ctl_0 = ctl_15;
    (&kernelContext_24)->calls_0 = calls_15;
    (&kernelContext_24)->nodes_0 = nodes_15;
    (&kernelContext_24)->params_0 = params_15;
    (&kernelContext_24)->bonds_0 = bonds_15;
    (&kernelContext_24)->chunk_node_0 = chunk_node_15;
    (&kernelContext_24)->counters_0 = counters_15;
    (&kernelContext_24)->route_0 = route_15;
    (&kernelContext_24)->isl_0 = isl_15;
    (&kernelContext_24)->islf_0 = islf_15;
    (&kernelContext_24)->islands_0 = islands_15;
    (&kernelContext_24)->ctab_0 = ctab_15;
    (&kernelContext_24)->bodies_0 = bodies_15;
    (&kernelContext_24)->chunks_0 = chunks_15;
    (&kernelContext_24)->state_0 = state_15;
    (&kernelContext_24)->imps_0 = imps_15;
    (&kernelContext_24)->status_0 = status_15;
    (&kernelContext_24)->largs_0 = largs_15;
    (&kernelContext_24)->cisl_0 = cisl_15;
    (&kernelContext_24)->bond_dyn_0 = bond_dyn_15;
    (&kernelContext_24)->bond_static_0 = bond_static_15;
    (&kernelContext_24)->frozen_0 = frozen_15;
    (&kernelContext_24)->split_work_0 = split_work_15;
    (&kernelContext_24)->split_control_0 = split_control_15;
    (&kernelContext_24)->split_parents_0 = split_parents_15;
    (&kernelContext_24)->split_slots_0 = split_slots_15;
    (&kernelContext_24)->split_out_0 = split_out_15;
    (&kernelContext_24)->split_firsts_0 = split_firsts_15;
    (&kernelContext_24)->lout_0 = lout_15;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_15;
    (&kernelContext_24)->g_red_0 = &g_red_15;
    threadgroup array<float, int(6)> g_lambda_15;
    (&kernelContext_24)->g_lambda_0 = &g_lambda_15;
    threadgroup array<float3, int(256)> g_lo_15;
    (&kernelContext_24)->g_lo_0 = &g_lo_15;
    threadgroup array<float3, int(256)> g_hi_15;
    (&kernelContext_24)->g_hi_0 = &g_hi_15;
    threadgroup array<uint, int(32)> g_label_15;
    (&kernelContext_24)->g_label_0 = &g_label_15;
    threadgroup uint g_used_15;
    (&kernelContext_24)->g_used_0 = &g_used_15;
    threadgroup array<uint, int(32)> g_call_of_15;
    (&kernelContext_24)->g_call_of_0 = &g_call_of_15;
    threadgroup array<uint, int(64)> g_mask_of_15;
    (&kernelContext_24)->g_mask_of_0 = &g_mask_of_15;
    threadgroup array<uint, int(64)> g_cnt_isl_15;
    (&kernelContext_24)->g_cnt_isl_0 = &g_cnt_isl_15;
    threadgroup array<uint, int(64)> g_cnt_chunk_15;
    (&kernelContext_24)->g_cnt_chunk_0 = &g_cnt_chunk_15;
    threadgroup array<uint, int(64)> g_cnt_bond_15;
    (&kernelContext_24)->g_cnt_bond_0 = &g_cnt_bond_15;
    threadgroup array<uint, int(64)> g_cnt_imp_15;
    (&kernelContext_24)->g_cnt_imp_0 = &g_cnt_imp_15;
    threadgroup array<uint, int(256)> g_scan_15;
    (&kernelContext_24)->g_scan_0 = &g_scan_15;
    threadgroup array<float, int(4096)> g_sort_15;
    (&kernelContext_24)->g_sort_0 = &g_sort_15;
    threadgroup uint g_stale_15;
    (&kernelContext_24)->g_stale_0 = &g_stale_15;
    threadgroup uint g_broke_15;
    (&kernelContext_24)->g_broke_0 = &g_broke_15;
    uint e_4 = id_2.x;
    bool _S479;
    if((*(ctl_15+0U)) == 0U)
    {
        _S479 = true;
    }
    else
    {
        _S479 = e_4 >= (*((&kernelContext_24)->ctl_0+9U));
    }
    if(_S479)
    {
        return;
    }
    uint4 _S480 = uint4(*((&kernelContext_24)->cisl_0+2U * e_4)) ;
    uint _S481 = 4U * e_4;
    float4 _S482 = float4(*((&kernelContext_24)->frozen_0+_S481)) ;
    uint body_5 = (as_type<uint>((_S482.x)));
    uint _S483 = 8U * _S480.w + 4U;
    bool stood_0 = (as_type<uint>(((float4(*((&kernelContext_24)->calls_0+_S483)) ).w))) == 0U;
    if(body_5 == 4294967295U)
    {
        return;
    }
    uint at_4 = 7U * body_5;
    if(!stood_0)
    {
        if(((as_type<uint>(((float4(*((&kernelContext_24)->calls_0+_S483)) ).w))) & 8U) != 0U)
        {
            *((&kernelContext_24)->bodies_0+at_4) = packed_float4(float4((float4(*((&kernelContext_24)->bodies_0+at_4)) ).xyz, _S482.y)) ;
            *((&kernelContext_24)->bodies_0+(at_4 + 2U)) = packed_float4(float4(*((&kernelContext_24)->frozen_0+(_S481 + 1U))) ) ;
            *((&kernelContext_24)->bodies_0+(at_4 + 3U)) = packed_float4(float4(*((&kernelContext_24)->frozen_0+(_S481 + 2U))) ) ;
            *((&kernelContext_24)->bodies_0+(at_4 + 4U)) = packed_float4(float4(*((&kernelContext_24)->frozen_0+(_S481 + 3U))) ) ;
        }
        return;
    }
    uint _S484 = _S480.y;
    if(((*((&kernelContext_24)->status_0+(16U + (uint4(((&kernelContext_24)->chunks_0+(as_type<uint>(((float4(*((&kernelContext_24)->nodes_0+(14U * _S484 + 12U))) ).x))))->info_1) ).y))) & 8U) != 0U)
    {
        *((&kernelContext_24)->bodies_0+at_4) = packed_float4(float4((float4(*((&kernelContext_24)->bodies_0+at_4)) ).xyz, _S482.y)) ;
        *((&kernelContext_24)->bodies_0+(at_4 + 2U)) = packed_float4(float4(*((&kernelContext_24)->frozen_0+(_S481 + 1U))) ) ;
        *((&kernelContext_24)->bodies_0+(at_4 + 3U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 0.0f)) ;
        return;
    }
    Fit_0 _S485 = fit_nodes_0(_S484, _S480.z, false, 0U, 0U, &kernelContext_24);
    float4 _S486 = float4(*((&kernelContext_24)->frozen_0+(_S481 + 1U))) ;
    thread Fit_0 _S487 = _S485;
    place_body_0(body_5, &_S487, _S482, _S486, &kernelContext_24);
    return;
}

[[kernel]] void learned_apply_impactors(uint3 group_11 [[threadgroup_position_in_grid]], uint3 thread_15 [[thread_position_in_threadgroup]], uint device* ctl_16 [[buffer(1)]], packed_float4 device* calls_16 [[buffer(2)]], packed_float4 device* nodes_16 [[buffer(3)]], LearnedParams_0 constant* params_16 [[buffer(0)]], packed_float4 device* bonds_16 [[buffer(4)]], uint device* chunk_node_16 [[buffer(8)]], uint device* counters_16 [[buffer(19)]], uint device* route_16 [[buffer(25)]], packed_uint4 device* isl_16 [[buffer(5)]], packed_float4 device* islf_16 [[buffer(6)]], Island_natural_0 device* islands_16 [[buffer(11)]], packed_float4 device* ctab_16 [[buffer(14)]], packed_float4 device* bodies_16 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_16 [[buffer(12)]], packed_float4 device* state_16 [[buffer(13)]], packed_float4 device* imps_16 [[buffer(10)]], uint device* status_16 [[buffer(18)]], packed_uint4 device* largs_16 [[buffer(27)]], packed_uint4 device* cisl_16 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_16 [[buffer(17)]], BondStatic_natural_0 device* bond_static_16 [[buffer(16)]], packed_float4 device* frozen_16 [[buffer(9)]], uint device* split_work_16 [[buffer(21)]], uint device* split_control_16 [[buffer(23)]], packed_uint4 device* split_parents_16 [[buffer(20)]], packed_uint4 device* split_slots_16 [[buffer(22)]], packed_float4 device* split_out_16 [[buffer(26)]], packed_uint4 device* split_firsts_16 [[buffer(24)]], float device* lout_16 [[buffer(28)]])
{
    thread KernelContext_0 kernelContext_25;
    (&kernelContext_25)->ctl_0 = ctl_16;
    (&kernelContext_25)->calls_0 = calls_16;
    (&kernelContext_25)->nodes_0 = nodes_16;
    (&kernelContext_25)->params_0 = params_16;
    (&kernelContext_25)->bonds_0 = bonds_16;
    (&kernelContext_25)->chunk_node_0 = chunk_node_16;
    (&kernelContext_25)->counters_0 = counters_16;
    (&kernelContext_25)->route_0 = route_16;
    (&kernelContext_25)->isl_0 = isl_16;
    (&kernelContext_25)->islf_0 = islf_16;
    (&kernelContext_25)->islands_0 = islands_16;
    (&kernelContext_25)->ctab_0 = ctab_16;
    (&kernelContext_25)->bodies_0 = bodies_16;
    (&kernelContext_25)->chunks_0 = chunks_16;
    (&kernelContext_25)->state_0 = state_16;
    (&kernelContext_25)->imps_0 = imps_16;
    (&kernelContext_25)->status_0 = status_16;
    (&kernelContext_25)->largs_0 = largs_16;
    (&kernelContext_25)->cisl_0 = cisl_16;
    (&kernelContext_25)->bond_dyn_0 = bond_dyn_16;
    (&kernelContext_25)->bond_static_0 = bond_static_16;
    (&kernelContext_25)->frozen_0 = frozen_16;
    (&kernelContext_25)->split_work_0 = split_work_16;
    (&kernelContext_25)->split_control_0 = split_control_16;
    (&kernelContext_25)->split_parents_0 = split_parents_16;
    (&kernelContext_25)->split_slots_0 = split_slots_16;
    (&kernelContext_25)->split_out_0 = split_out_16;
    (&kernelContext_25)->split_firsts_0 = split_firsts_16;
    (&kernelContext_25)->lout_0 = lout_16;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_16;
    (&kernelContext_25)->g_red_0 = &g_red_16;
    threadgroup array<float, int(6)> g_lambda_16;
    (&kernelContext_25)->g_lambda_0 = &g_lambda_16;
    threadgroup array<float3, int(256)> g_lo_16;
    (&kernelContext_25)->g_lo_0 = &g_lo_16;
    threadgroup array<float3, int(256)> g_hi_16;
    (&kernelContext_25)->g_hi_0 = &g_hi_16;
    threadgroup array<uint, int(32)> g_label_16;
    (&kernelContext_25)->g_label_0 = &g_label_16;
    threadgroup uint g_used_16;
    (&kernelContext_25)->g_used_0 = &g_used_16;
    threadgroup array<uint, int(32)> g_call_of_16;
    (&kernelContext_25)->g_call_of_0 = &g_call_of_16;
    threadgroup array<uint, int(64)> g_mask_of_16;
    (&kernelContext_25)->g_mask_of_0 = &g_mask_of_16;
    threadgroup array<uint, int(64)> g_cnt_isl_16;
    (&kernelContext_25)->g_cnt_isl_0 = &g_cnt_isl_16;
    threadgroup array<uint, int(64)> g_cnt_chunk_16;
    (&kernelContext_25)->g_cnt_chunk_0 = &g_cnt_chunk_16;
    threadgroup array<uint, int(64)> g_cnt_bond_16;
    (&kernelContext_25)->g_cnt_bond_0 = &g_cnt_bond_16;
    threadgroup array<uint, int(64)> g_cnt_imp_16;
    (&kernelContext_25)->g_cnt_imp_0 = &g_cnt_imp_16;
    threadgroup array<uint, int(256)> g_scan_16;
    (&kernelContext_25)->g_scan_0 = &g_scan_16;
    threadgroup array<float, int(4096)> g_sort_16;
    (&kernelContext_25)->g_sort_0 = &g_sort_16;
    threadgroup uint g_stale_16;
    (&kernelContext_25)->g_stale_0 = &g_stale_16;
    threadgroup uint g_broke_16;
    (&kernelContext_25)->g_broke_0 = &g_broke_16;
    uint g_9 = group_11.x;
    uint k_30 = thread_15.x;
    bool _S488;
    if((*(ctl_16+0U)) == 0U)
    {
        _S488 = true;
    }
    else
    {
        _S488 = g_9 >= (*((&kernelContext_25)->ctl_0+1U));
    }
    if(_S488)
    {
        return;
    }
    uint _S489 = 8U * g_9;
    uint mask_7 = (as_type<uint>(((float4(*((&kernelContext_25)->calls_0+(_S489 + 5U))) ).w)));
    uint fb_1 = (as_type<uint>(((float4(*((&kernelContext_25)->calls_0+(_S489 + 4U))) ).w)));
    uint _S490 = 1U << k_30;
    if((mask_7 & _S490) == 0U)
    {
        _S488 = true;
    }
    else
    {
        if(fb_1 != 0U)
        {
            _S488 = (fb_1 & 8U) == 0U;
        }
        else
        {
            _S488 = false;
        }
    }
    if(_S488)
    {
        return;
    }
    uint _S491 = 4U * ((&kernelContext_25)->params_0->max_cisl_0 + k_30);
    float4 _S492 = float4(*((&kernelContext_25)->frozen_0+_S491)) ;
    uint at_5 = 7U * (as_type<uint>((_S492.x)));
    float _S493 = _S492.y;
    *((&kernelContext_25)->bodies_0+at_5) = packed_float4(float4((float4(*((&kernelContext_25)->bodies_0+at_5)) ).xyz, _S493)) ;
    *((&kernelContext_25)->bodies_0+(at_5 + 2U)) = packed_float4(float4(*((&kernelContext_25)->frozen_0+(_S491 + 1U))) ) ;
    if(fb_1 != 0U)
    {
        *((&kernelContext_25)->bodies_0+(at_5 + 3U)) = packed_float4(float4(*((&kernelContext_25)->frozen_0+(_S491 + 2U))) ) ;
        *((&kernelContext_25)->bodies_0+(at_5 + 4U)) = packed_float4(float4(*((&kernelContext_25)->frozen_0+(_S491 + 3U))) ) ;
        return;
    }
    uint n_6 = (as_type<uint>(((float4(*((&kernelContext_25)->calls_0+(_S489 + 2U))) ).z))) + (as_type<uint>(((float4(*((&kernelContext_25)->calls_0+(_S489 + 3U))) ).w))) + ((uint)popcount((mask_7 & (_S490 - 1U))));
    float4 _S494 = node_0(n_6, 10U, &kernelContext_25);
    float3 dx_2 = _S494.xyz;
    float4 _S495 = node_0(n_6, 11U, &kernelContext_25);
    float3 dr_1 = _S495.xyz;
    *((&kernelContext_25)->bodies_0+at_5) = packed_float4(float4((float4(*((&kernelContext_25)->bodies_0+at_5)) ).xyz + to_yup_0(dx_2), _S493)) ;
    uint _S496 = at_5 + 1U;
    *((&kernelContext_25)->bodies_0+_S496) = packed_float4(turn_rubble_0(float4(*((&kernelContext_25)->bodies_0+_S496)) , dr_1)) ;
    packed_float4 device* _S497 = (&kernelContext_25)->bodies_0+(at_5 + 3U);
    float4 _S498 = node_0(n_6, 8U, &kernelContext_25);
    *_S497 = packed_float4(float4(to_yup_0(_S498.xyz), 0.0f)) ;
    uint _S499 = at_5 + 4U;
    packed_float4 device* _S500 = (&kernelContext_25)->bodies_0+_S499;
    float4 _S501 = node_0(n_6, 9U, &kernelContext_25);
    *_S500 = packed_float4(float4(to_yup_0(_S501.xyz), (float4(*((&kernelContext_25)->bodies_0+_S499)) ).w)) ;
    return;
}

[[kernel]] void learned_place(uint3 id_3 [[thread_position_in_grid]], uint device* ctl_17 [[buffer(1)]], packed_float4 device* calls_17 [[buffer(2)]], packed_float4 device* nodes_17 [[buffer(3)]], LearnedParams_0 constant* params_17 [[buffer(0)]], packed_float4 device* bonds_17 [[buffer(4)]], uint device* chunk_node_17 [[buffer(8)]], uint device* counters_17 [[buffer(19)]], uint device* route_17 [[buffer(25)]], packed_uint4 device* isl_17 [[buffer(5)]], packed_float4 device* islf_17 [[buffer(6)]], Island_natural_0 device* islands_17 [[buffer(11)]], packed_float4 device* ctab_17 [[buffer(14)]], packed_float4 device* bodies_17 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_17 [[buffer(12)]], packed_float4 device* state_17 [[buffer(13)]], packed_float4 device* imps_17 [[buffer(10)]], uint device* status_17 [[buffer(18)]], packed_uint4 device* largs_17 [[buffer(27)]], packed_uint4 device* cisl_17 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_17 [[buffer(17)]], BondStatic_natural_0 device* bond_static_17 [[buffer(16)]], packed_float4 device* frozen_17 [[buffer(9)]], uint device* split_work_17 [[buffer(21)]], uint device* split_control_17 [[buffer(23)]], packed_uint4 device* split_parents_17 [[buffer(20)]], packed_uint4 device* split_slots_17 [[buffer(22)]], packed_float4 device* split_out_17 [[buffer(26)]], packed_uint4 device* split_firsts_17 [[buffer(24)]], float device* lout_17 [[buffer(28)]])
{
    thread KernelContext_0 kernelContext_26;
    (&kernelContext_26)->ctl_0 = ctl_17;
    (&kernelContext_26)->calls_0 = calls_17;
    (&kernelContext_26)->nodes_0 = nodes_17;
    (&kernelContext_26)->params_0 = params_17;
    (&kernelContext_26)->bonds_0 = bonds_17;
    (&kernelContext_26)->chunk_node_0 = chunk_node_17;
    (&kernelContext_26)->counters_0 = counters_17;
    (&kernelContext_26)->route_0 = route_17;
    (&kernelContext_26)->isl_0 = isl_17;
    (&kernelContext_26)->islf_0 = islf_17;
    (&kernelContext_26)->islands_0 = islands_17;
    (&kernelContext_26)->ctab_0 = ctab_17;
    (&kernelContext_26)->bodies_0 = bodies_17;
    (&kernelContext_26)->chunks_0 = chunks_17;
    (&kernelContext_26)->state_0 = state_17;
    (&kernelContext_26)->imps_0 = imps_17;
    (&kernelContext_26)->status_0 = status_17;
    (&kernelContext_26)->largs_0 = largs_17;
    (&kernelContext_26)->cisl_0 = cisl_17;
    (&kernelContext_26)->bond_dyn_0 = bond_dyn_17;
    (&kernelContext_26)->bond_static_0 = bond_static_17;
    (&kernelContext_26)->frozen_0 = frozen_17;
    (&kernelContext_26)->split_work_0 = split_work_17;
    (&kernelContext_26)->split_control_0 = split_control_17;
    (&kernelContext_26)->split_parents_0 = split_parents_17;
    (&kernelContext_26)->split_slots_0 = split_slots_17;
    (&kernelContext_26)->split_out_0 = split_out_17;
    (&kernelContext_26)->split_firsts_0 = split_firsts_17;
    (&kernelContext_26)->lout_0 = lout_17;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_17;
    (&kernelContext_26)->g_red_0 = &g_red_17;
    threadgroup array<float, int(6)> g_lambda_17;
    (&kernelContext_26)->g_lambda_0 = &g_lambda_17;
    threadgroup array<float3, int(256)> g_lo_17;
    (&kernelContext_26)->g_lo_0 = &g_lo_17;
    threadgroup array<float3, int(256)> g_hi_17;
    (&kernelContext_26)->g_hi_0 = &g_hi_17;
    threadgroup array<uint, int(32)> g_label_17;
    (&kernelContext_26)->g_label_0 = &g_label_17;
    threadgroup uint g_used_17;
    (&kernelContext_26)->g_used_0 = &g_used_17;
    threadgroup array<uint, int(32)> g_call_of_17;
    (&kernelContext_26)->g_call_of_0 = &g_call_of_17;
    threadgroup array<uint, int(64)> g_mask_of_17;
    (&kernelContext_26)->g_mask_of_0 = &g_mask_of_17;
    threadgroup array<uint, int(64)> g_cnt_isl_17;
    (&kernelContext_26)->g_cnt_isl_0 = &g_cnt_isl_17;
    threadgroup array<uint, int(64)> g_cnt_chunk_17;
    (&kernelContext_26)->g_cnt_chunk_0 = &g_cnt_chunk_17;
    threadgroup array<uint, int(64)> g_cnt_bond_17;
    (&kernelContext_26)->g_cnt_bond_0 = &g_cnt_bond_17;
    threadgroup array<uint, int(64)> g_cnt_imp_17;
    (&kernelContext_26)->g_cnt_imp_0 = &g_cnt_imp_17;
    threadgroup array<uint, int(256)> g_scan_17;
    (&kernelContext_26)->g_scan_0 = &g_scan_17;
    threadgroup array<float, int(4096)> g_sort_17;
    (&kernelContext_26)->g_sort_0 = &g_sort_17;
    threadgroup uint g_stale_17;
    (&kernelContext_26)->g_stale_0 = &g_stale_17;
    threadgroup uint g_broke_17;
    (&kernelContext_26)->g_broke_0 = &g_broke_17;
    uint slot_0 = id_3.x;
    bool _S502;
    if((*(ctl_17+18U)) == 0U)
    {
        _S502 = true;
    }
    else
    {
        _S502 = (&kernelContext_26)->split_control_0[0U] != 0U;
    }
    if(_S502)
    {
        return;
    }
    uint np_0 = (&kernelContext_26)->split_control_0[1U];
    if(np_0 == 0U)
    {
        return;
    }
    uint total_0 = (uint4(*((&kernelContext_26)->split_parents_0+np_0)) ).y;
    if(slot_0 >= total_0)
    {
        return;
    }
    uint _S503 = np_0 - 1U;
    uint lo_3 = 0U;
    uint hi_3 = _S503;
    for(;;)
    {
        if(lo_3 < hi_3)
        {
        }
        else
        {
            break;
        }
        uint mid_0 = (lo_3 + hi_3 + 1U) / 2U;
        if(((uint4(*((&kernelContext_26)->split_parents_0+mid_0)) ).y) <= slot_0)
        {
            lo_3 = mid_0;
        }
        else
        {
            hi_3 = mid_0 - 1U;
        }
    }
    uint4 _S504 = uint4(*((&kernelContext_26)->split_parents_0+lo_3)) ;
    uint _S505 = _S504.y;
    if((slot_0 - _S505) >= (_S504.z))
    {
        return;
    }
    uint4 _S506 = uint4(*((&kernelContext_26)->split_slots_0+slot_0)) ;
    if((_S506.w) == 0U)
    {
        return;
    }
    float4 _S507 = float4(*((&kernelContext_26)->split_out_0+8U * slot_0)) ;
    if((as_type<uint>((_S507.w))) != 0U)
    {
        return;
    }
    uint count_2 = (as_type<uint>((_S507.z)));
    uint base_1 = total_0 + _S505 + (as_type<uint>((_S507.y)));
    uint first_2 = (uint4(*((&kernelContext_26)->split_firsts_0+lo_3)) ).x;
    if(count_2 == 0U)
    {
        return;
    }
    uint j_11 = 0U;
    for(;;)
    {
        if(j_11 < count_2)
        {
        }
        else
        {
            break;
        }
        if((*((&kernelContext_26)->chunk_node_0+(first_2 + (&kernelContext_26)->split_work_0[base_1 + j_11]))) == 0U)
        {
            return;
        }
        j_11 = j_11 + 1U;
    }
    Fit_0 _S508 = fit_nodes_0(0U, count_2, true, base_1, first_2, &kernelContext_26);
    uint _S509 = _S506.x;
    uint at_6 = 7U * _S509;
    float4 _S510 = float4(0.0f, (float4(*((&kernelContext_26)->bodies_0+at_6)) ).w, 0.0f, 0.0f);
    float4 _S511 = float4(*((&kernelContext_26)->bodies_0+(at_6 + 2U))) ;
    thread Fit_0 _S512 = _S508;
    place_body_0(_S509, &_S512, _S510, _S511, &kernelContext_26);
    return;
}

[[kernel]] void learned_tally_apply(uint device* ctl_18 [[buffer(1)]], packed_float4 device* calls_18 [[buffer(2)]], packed_float4 device* nodes_18 [[buffer(3)]], LearnedParams_0 constant* params_18 [[buffer(0)]], packed_float4 device* bonds_18 [[buffer(4)]], uint device* chunk_node_18 [[buffer(8)]], uint device* counters_18 [[buffer(19)]], uint device* route_18 [[buffer(25)]], packed_uint4 device* isl_18 [[buffer(5)]], packed_float4 device* islf_18 [[buffer(6)]], Island_natural_0 device* islands_18 [[buffer(11)]], packed_float4 device* ctab_18 [[buffer(14)]], packed_float4 device* bodies_18 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_18 [[buffer(12)]], packed_float4 device* state_18 [[buffer(13)]], packed_float4 device* imps_18 [[buffer(10)]], uint device* status_18 [[buffer(18)]], packed_uint4 device* largs_18 [[buffer(27)]], packed_uint4 device* cisl_18 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_18 [[buffer(17)]], BondStatic_natural_0 device* bond_static_18 [[buffer(16)]], packed_float4 device* frozen_18 [[buffer(9)]], uint device* split_work_18 [[buffer(21)]], uint device* split_control_18 [[buffer(23)]], packed_uint4 device* split_parents_18 [[buffer(20)]], packed_uint4 device* split_slots_18 [[buffer(22)]], packed_float4 device* split_out_18 [[buffer(26)]], packed_uint4 device* split_firsts_18 [[buffer(24)]], float device* lout_18 [[buffer(28)]])
{
    thread KernelContext_0 kernelContext_27;
    (&kernelContext_27)->ctl_0 = ctl_18;
    (&kernelContext_27)->calls_0 = calls_18;
    (&kernelContext_27)->nodes_0 = nodes_18;
    (&kernelContext_27)->params_0 = params_18;
    (&kernelContext_27)->bonds_0 = bonds_18;
    (&kernelContext_27)->chunk_node_0 = chunk_node_18;
    (&kernelContext_27)->counters_0 = counters_18;
    (&kernelContext_27)->route_0 = route_18;
    (&kernelContext_27)->isl_0 = isl_18;
    (&kernelContext_27)->islf_0 = islf_18;
    (&kernelContext_27)->islands_0 = islands_18;
    (&kernelContext_27)->ctab_0 = ctab_18;
    (&kernelContext_27)->bodies_0 = bodies_18;
    (&kernelContext_27)->chunks_0 = chunks_18;
    (&kernelContext_27)->state_0 = state_18;
    (&kernelContext_27)->imps_0 = imps_18;
    (&kernelContext_27)->status_0 = status_18;
    (&kernelContext_27)->largs_0 = largs_18;
    (&kernelContext_27)->cisl_0 = cisl_18;
    (&kernelContext_27)->bond_dyn_0 = bond_dyn_18;
    (&kernelContext_27)->bond_static_0 = bond_static_18;
    (&kernelContext_27)->frozen_0 = frozen_18;
    (&kernelContext_27)->split_work_0 = split_work_18;
    (&kernelContext_27)->split_control_0 = split_control_18;
    (&kernelContext_27)->split_parents_0 = split_parents_18;
    (&kernelContext_27)->split_slots_0 = split_slots_18;
    (&kernelContext_27)->split_out_0 = split_out_18;
    (&kernelContext_27)->split_firsts_0 = split_firsts_18;
    (&kernelContext_27)->lout_0 = lout_18;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_18;
    (&kernelContext_27)->g_red_0 = &g_red_18;
    threadgroup array<float, int(6)> g_lambda_18;
    (&kernelContext_27)->g_lambda_0 = &g_lambda_18;
    threadgroup array<float3, int(256)> g_lo_18;
    (&kernelContext_27)->g_lo_0 = &g_lo_18;
    threadgroup array<float3, int(256)> g_hi_18;
    (&kernelContext_27)->g_hi_0 = &g_hi_18;
    threadgroup array<uint, int(32)> g_label_18;
    (&kernelContext_27)->g_label_0 = &g_label_18;
    threadgroup uint g_used_18;
    (&kernelContext_27)->g_used_0 = &g_used_18;
    threadgroup array<uint, int(32)> g_call_of_18;
    (&kernelContext_27)->g_call_of_0 = &g_call_of_18;
    threadgroup array<uint, int(64)> g_mask_of_18;
    (&kernelContext_27)->g_mask_of_0 = &g_mask_of_18;
    threadgroup array<uint, int(64)> g_cnt_isl_18;
    (&kernelContext_27)->g_cnt_isl_0 = &g_cnt_isl_18;
    threadgroup array<uint, int(64)> g_cnt_chunk_18;
    (&kernelContext_27)->g_cnt_chunk_0 = &g_cnt_chunk_18;
    threadgroup array<uint, int(64)> g_cnt_bond_18;
    (&kernelContext_27)->g_cnt_bond_0 = &g_cnt_bond_18;
    threadgroup array<uint, int(64)> g_cnt_imp_18;
    (&kernelContext_27)->g_cnt_imp_0 = &g_cnt_imp_18;
    threadgroup array<uint, int(256)> g_scan_18;
    (&kernelContext_27)->g_scan_0 = &g_scan_18;
    threadgroup array<float, int(4096)> g_sort_18;
    (&kernelContext_27)->g_sort_0 = &g_sort_18;
    threadgroup uint g_stale_18;
    (&kernelContext_27)->g_stale_0 = &g_stale_18;
    threadgroup uint g_broke_18;
    (&kernelContext_27)->g_broke_0 = &g_broke_18;
    if((*(ctl_18+0U)) == 0U)
    {
        return;
    }
    uint g_10 = 0U;
    uint stale_0 = 0U;
    uint broken_0 = 0U;
    uint applied_0 = 0U;
    for(;;)
    {
        if(g_10 < (*((&kernelContext_27)->ctl_0+1U)))
        {
        }
        else
        {
            break;
        }
        uint _S513 = 8U * g_10;
        uint _S514 = _S513 + 4U;
        bool _S515;
        if((as_type<uint>(((float4(*((&kernelContext_27)->calls_0+_S514)) ).w))) != 0U)
        {
            _S515 = (as_type<uint>(((float4(*((&kernelContext_27)->calls_0+_S514)) ).w))) != 8U;
        }
        else
        {
            _S515 = false;
        }
        if(_S515)
        {
            g_10 = g_10 + 1U;
            continue;
        }
        float4 _S516 = float4(*((&kernelContext_27)->calls_0+(_S513 + 7U))) ;
        uint broken_1 = broken_0 + (as_type<uint>((_S516.y)));
        uint applied_1 = applied_0 + (as_type<uint>((_S516.z)));
        stale_0 = stale_0 + (as_type<uint>((_S516.x)));
        broken_0 = broken_1;
        applied_0 = applied_1;
        g_10 = g_10 + 1U;
    }
    *((&kernelContext_27)->ctl_0+18U) = 1U;
    *((&kernelContext_27)->ctl_0+10U) = stale_0;
    *((&kernelContext_27)->ctl_0+12U) = broken_0;
    *((&kernelContext_27)->ctl_0+11U) = applied_0;
    return;
}

[[kernel]] void learned_tally(uint device* ctl_19 [[buffer(1)]], packed_float4 device* calls_19 [[buffer(2)]], packed_float4 device* nodes_19 [[buffer(3)]], LearnedParams_0 constant* params_19 [[buffer(0)]], packed_float4 device* bonds_19 [[buffer(4)]], uint device* chunk_node_19 [[buffer(8)]], uint device* counters_19 [[buffer(19)]], uint device* route_19 [[buffer(25)]], packed_uint4 device* isl_19 [[buffer(5)]], packed_float4 device* islf_19 [[buffer(6)]], Island_natural_0 device* islands_19 [[buffer(11)]], packed_float4 device* ctab_19 [[buffer(14)]], packed_float4 device* bodies_19 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_19 [[buffer(12)]], packed_float4 device* state_19 [[buffer(13)]], packed_float4 device* imps_19 [[buffer(10)]], uint device* status_19 [[buffer(18)]], packed_uint4 device* largs_19 [[buffer(27)]], packed_uint4 device* cisl_19 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_19 [[buffer(17)]], BondStatic_natural_0 device* bond_static_19 [[buffer(16)]], packed_float4 device* frozen_19 [[buffer(9)]], uint device* split_work_19 [[buffer(21)]], uint device* split_control_19 [[buffer(23)]], packed_uint4 device* split_parents_19 [[buffer(20)]], packed_uint4 device* split_slots_19 [[buffer(22)]], packed_float4 device* split_out_19 [[buffer(26)]], packed_uint4 device* split_firsts_19 [[buffer(24)]], float device* lout_19 [[buffer(28)]])
{
    thread KernelContext_0 kernelContext_28;
    (&kernelContext_28)->ctl_0 = ctl_19;
    (&kernelContext_28)->calls_0 = calls_19;
    (&kernelContext_28)->nodes_0 = nodes_19;
    (&kernelContext_28)->params_0 = params_19;
    (&kernelContext_28)->bonds_0 = bonds_19;
    (&kernelContext_28)->chunk_node_0 = chunk_node_19;
    (&kernelContext_28)->counters_0 = counters_19;
    (&kernelContext_28)->route_0 = route_19;
    (&kernelContext_28)->isl_0 = isl_19;
    (&kernelContext_28)->islf_0 = islf_19;
    (&kernelContext_28)->islands_0 = islands_19;
    (&kernelContext_28)->ctab_0 = ctab_19;
    (&kernelContext_28)->bodies_0 = bodies_19;
    (&kernelContext_28)->chunks_0 = chunks_19;
    (&kernelContext_28)->state_0 = state_19;
    (&kernelContext_28)->imps_0 = imps_19;
    (&kernelContext_28)->status_0 = status_19;
    (&kernelContext_28)->largs_0 = largs_19;
    (&kernelContext_28)->cisl_0 = cisl_19;
    (&kernelContext_28)->bond_dyn_0 = bond_dyn_19;
    (&kernelContext_28)->bond_static_0 = bond_static_19;
    (&kernelContext_28)->frozen_0 = frozen_19;
    (&kernelContext_28)->split_work_0 = split_work_19;
    (&kernelContext_28)->split_control_0 = split_control_19;
    (&kernelContext_28)->split_parents_0 = split_parents_19;
    (&kernelContext_28)->split_slots_0 = split_slots_19;
    (&kernelContext_28)->split_out_0 = split_out_19;
    (&kernelContext_28)->split_firsts_0 = split_firsts_19;
    (&kernelContext_28)->lout_0 = lout_19;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_19;
    (&kernelContext_28)->g_red_0 = &g_red_19;
    threadgroup array<float, int(6)> g_lambda_19;
    (&kernelContext_28)->g_lambda_0 = &g_lambda_19;
    threadgroup array<float3, int(256)> g_lo_19;
    (&kernelContext_28)->g_lo_0 = &g_lo_19;
    threadgroup array<float3, int(256)> g_hi_19;
    (&kernelContext_28)->g_hi_0 = &g_hi_19;
    threadgroup array<uint, int(32)> g_label_19;
    (&kernelContext_28)->g_label_0 = &g_label_19;
    threadgroup uint g_used_19;
    (&kernelContext_28)->g_used_0 = &g_used_19;
    threadgroup array<uint, int(32)> g_call_of_19;
    (&kernelContext_28)->g_call_of_0 = &g_call_of_19;
    threadgroup array<uint, int(64)> g_mask_of_19;
    (&kernelContext_28)->g_mask_of_0 = &g_mask_of_19;
    threadgroup array<uint, int(64)> g_cnt_isl_19;
    (&kernelContext_28)->g_cnt_isl_0 = &g_cnt_isl_19;
    threadgroup array<uint, int(64)> g_cnt_chunk_19;
    (&kernelContext_28)->g_cnt_chunk_0 = &g_cnt_chunk_19;
    threadgroup array<uint, int(64)> g_cnt_bond_19;
    (&kernelContext_28)->g_cnt_bond_0 = &g_cnt_bond_19;
    threadgroup array<uint, int(64)> g_cnt_imp_19;
    (&kernelContext_28)->g_cnt_imp_0 = &g_cnt_imp_19;
    threadgroup array<uint, int(256)> g_scan_19;
    (&kernelContext_28)->g_scan_0 = &g_scan_19;
    threadgroup array<float, int(4096)> g_sort_19;
    (&kernelContext_28)->g_sort_0 = &g_sort_19;
    threadgroup uint g_stale_19;
    (&kernelContext_28)->g_stale_0 = &g_stale_19;
    threadgroup uint g_broke_19;
    (&kernelContext_28)->g_broke_0 = &g_broke_19;
    uint _S517 = *(ctl_19+1U);
    uint g_11 = 0U;
    uint bound_1 = 0U;
    uint infeasible_0 = 0U;
    uint fallbacks_0 = 0U;
    uint nonfinite_0 = 0U;
    uint fb_inf_0 = 0U;
    uint range_1 = 0U;
    uint stood_1 = 0U;
    for(;;)
    {
        if(g_11 < _S517)
        {
        }
        else
        {
            break;
        }
        uint _S518 = 8U * g_11;
        uint _S519 = _S518 + 4U;
        uint dec_2 = (as_type<uint>(((float4(*((&kernelContext_28)->calls_0+_S519)) ).z)));
        uint fb_2 = (as_type<uint>(((float4(*((&kernelContext_28)->calls_0+_S519)) ).w)));
        uint _S520;
        if((dec_2 & 1U) != 0U)
        {
            _S520 = 1U;
        }
        else
        {
            _S520 = 0U;
        }
        uint bound_2 = bound_1 + _S520;
        uint _S521;
        if((dec_2 & 2U) != 0U)
        {
            _S521 = 1U;
        }
        else
        {
            _S521 = 0U;
        }
        uint infeasible_1 = infeasible_0 + _S521;
        if(fb_2 != 0U)
        {
            uint _S522 = fallbacks_0 + 1U;
            uint _S523;
            if((fb_2 & 1U) != 0U)
            {
                _S523 = 1U;
            }
            else
            {
                _S523 = 0U;
            }
            uint nonfinite_1 = nonfinite_0 + _S523;
            uint _S524;
            if((fb_2 & 2U) != 0U)
            {
                _S524 = 1U;
            }
            else
            {
                _S524 = 0U;
            }
            uint fb_inf_1 = fb_inf_0 + _S524;
            uint _S525;
            if((fb_2 & 4U) != 0U)
            {
                _S525 = 1U;
            }
            else
            {
                _S525 = 0U;
            }
            uint range_2 = range_1 + _S525;
            fallbacks_0 = _S522;
            nonfinite_0 = nonfinite_1;
            fb_inf_0 = fb_inf_1;
            range_1 = range_2;
        }
        else
        {
            stood_1 = stood_1 + 1U;
        }
        *((&kernelContext_28)->calls_0+(_S518 + 7U)) = packed_float4(float4(0.0f) ) ;
        g_11 = g_11 + 1U;
        bound_1 = bound_2;
        infeasible_0 = infeasible_1;
    }
    *((&kernelContext_28)->ctl_0+5U) = fallbacks_0;
    *((&kernelContext_28)->ctl_0+6U) = bound_1;
    *((&kernelContext_28)->ctl_0+7U) = infeasible_0;
    *((&kernelContext_28)->ctl_0+14U) = nonfinite_0;
    *((&kernelContext_28)->ctl_0+15U) = fb_inf_0;
    *((&kernelContext_28)->ctl_0+16U) = range_1;
    uint device* _S526 = (&kernelContext_28)->ctl_0+0U;
    if(stood_1 > 0U)
    {
        g_11 = 1U;
    }
    else
    {
        g_11 = 0U;
    }
    *_S526 = g_11;
    return;
}

[[kernel]] void learned_adopt(uint3 group_12 [[threadgroup_position_in_grid]], uint3 thread_16 [[thread_position_in_threadgroup]], uint device* ctl_20 [[buffer(1)]], packed_float4 device* calls_20 [[buffer(2)]], packed_float4 device* nodes_20 [[buffer(3)]], LearnedParams_0 constant* params_20 [[buffer(0)]], packed_float4 device* bonds_20 [[buffer(4)]], uint device* chunk_node_20 [[buffer(8)]], uint device* counters_20 [[buffer(19)]], uint device* route_20 [[buffer(25)]], packed_uint4 device* isl_20 [[buffer(5)]], packed_float4 device* islf_20 [[buffer(6)]], Island_natural_0 device* islands_20 [[buffer(11)]], packed_float4 device* ctab_20 [[buffer(14)]], packed_float4 device* bodies_20 [[buffer(15)]], ChunkStatic_natural_0 device* chunks_20 [[buffer(12)]], packed_float4 device* state_20 [[buffer(13)]], packed_float4 device* imps_20 [[buffer(10)]], uint device* status_20 [[buffer(18)]], packed_uint4 device* largs_20 [[buffer(27)]], packed_uint4 device* cisl_20 [[buffer(7)]], BondDyn_natural_0 device* bond_dyn_20 [[buffer(17)]], BondStatic_natural_0 device* bond_static_20 [[buffer(16)]], packed_float4 device* frozen_20 [[buffer(9)]], uint device* split_work_20 [[buffer(21)]], uint device* split_control_20 [[buffer(23)]], packed_uint4 device* split_parents_20 [[buffer(20)]], packed_uint4 device* split_slots_20 [[buffer(22)]], packed_float4 device* split_out_20 [[buffer(26)]], packed_uint4 device* split_firsts_20 [[buffer(24)]], float device* lout_20 [[buffer(28)]])
{
    uint at_7;
    thread KernelContext_0 kernelContext_29;
    (&kernelContext_29)->ctl_0 = ctl_20;
    (&kernelContext_29)->calls_0 = calls_20;
    (&kernelContext_29)->nodes_0 = nodes_20;
    (&kernelContext_29)->params_0 = params_20;
    (&kernelContext_29)->bonds_0 = bonds_20;
    (&kernelContext_29)->chunk_node_0 = chunk_node_20;
    (&kernelContext_29)->counters_0 = counters_20;
    (&kernelContext_29)->route_0 = route_20;
    (&kernelContext_29)->isl_0 = isl_20;
    (&kernelContext_29)->islf_0 = islf_20;
    (&kernelContext_29)->islands_0 = islands_20;
    (&kernelContext_29)->ctab_0 = ctab_20;
    (&kernelContext_29)->bodies_0 = bodies_20;
    (&kernelContext_29)->chunks_0 = chunks_20;
    (&kernelContext_29)->state_0 = state_20;
    (&kernelContext_29)->imps_0 = imps_20;
    (&kernelContext_29)->status_0 = status_20;
    (&kernelContext_29)->largs_0 = largs_20;
    (&kernelContext_29)->cisl_0 = cisl_20;
    (&kernelContext_29)->bond_dyn_0 = bond_dyn_20;
    (&kernelContext_29)->bond_static_0 = bond_static_20;
    (&kernelContext_29)->frozen_0 = frozen_20;
    (&kernelContext_29)->split_work_0 = split_work_20;
    (&kernelContext_29)->split_control_0 = split_control_20;
    (&kernelContext_29)->split_parents_0 = split_parents_20;
    (&kernelContext_29)->split_slots_0 = split_slots_20;
    (&kernelContext_29)->split_out_0 = split_out_20;
    (&kernelContext_29)->split_firsts_0 = split_firsts_20;
    (&kernelContext_29)->lout_0 = lout_20;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_20;
    (&kernelContext_29)->g_red_0 = &g_red_20;
    threadgroup array<float, int(6)> g_lambda_20;
    (&kernelContext_29)->g_lambda_0 = &g_lambda_20;
    threadgroup array<float3, int(256)> g_lo_20;
    (&kernelContext_29)->g_lo_0 = &g_lo_20;
    threadgroup array<float3, int(256)> g_hi_20;
    (&kernelContext_29)->g_hi_0 = &g_hi_20;
    threadgroup array<uint, int(32)> g_label_20;
    (&kernelContext_29)->g_label_0 = &g_label_20;
    threadgroup uint g_used_20;
    (&kernelContext_29)->g_used_0 = &g_used_20;
    threadgroup array<uint, int(32)> g_call_of_20;
    (&kernelContext_29)->g_call_of_0 = &g_call_of_20;
    threadgroup array<uint, int(64)> g_mask_of_20;
    (&kernelContext_29)->g_mask_of_0 = &g_mask_of_20;
    threadgroup array<uint, int(64)> g_cnt_isl_20;
    (&kernelContext_29)->g_cnt_isl_0 = &g_cnt_isl_20;
    threadgroup array<uint, int(64)> g_cnt_chunk_20;
    (&kernelContext_29)->g_cnt_chunk_0 = &g_cnt_chunk_20;
    threadgroup array<uint, int(64)> g_cnt_bond_20;
    (&kernelContext_29)->g_cnt_bond_0 = &g_cnt_bond_20;
    threadgroup array<uint, int(64)> g_cnt_imp_20;
    (&kernelContext_29)->g_cnt_imp_0 = &g_cnt_imp_20;
    threadgroup array<uint, int(256)> g_scan_20;
    (&kernelContext_29)->g_scan_0 = &g_scan_20;
    threadgroup array<float, int(4096)> g_sort_20;
    (&kernelContext_29)->g_sort_0 = &g_sort_20;
    threadgroup uint g_stale_20;
    (&kernelContext_29)->g_stale_0 = &g_stale_20;
    threadgroup uint g_broke_20;
    (&kernelContext_29)->g_broke_0 = &g_broke_20;
    uint g_12 = group_12.x;
    uint t_15 = thread_16.x;
    if(g_12 >= (*(ctl_20+1U)))
    {
        return;
    }
    uint _S527 = 8U * g_12;
    uint _S528 = _S527 + 2U;
    uint _S529 = (as_type<uint>(((float4(*((&kernelContext_29)->calls_0+_S528)) ).z)));
    uint _S530 = (as_type<uint>(((float4(*((&kernelContext_29)->calls_0+_S528)) ).w)));
    uint _S531 = (as_type<uint>(((float4(*((&kernelContext_29)->calls_0+(_S527 + 3U))) ).w)));
    uint _S532 = _S527 + 4U;
    uint _S533 = (as_type<uint>(((float4(*((&kernelContext_29)->calls_0+_S532)) ).x)));
    uint _S534 = (as_type<uint>(((float4(*((&kernelContext_29)->calls_0+_S532)) ).y)));
    uint _S535 = _S527 + 6U;
    uint _S536 = (as_type<uint>(((float4(*((&kernelContext_29)->calls_0+_S535)) ).z)));
    uint _S537 = (as_type<uint>(((float4(*((&kernelContext_29)->calls_0+_S535)) ).w)));
    uint j_12 = t_15;
    for(;;)
    {
        if(j_12 < _S530)
        {
        }
        else
        {
            break;
        }
        if(j_12 < _S531)
        {
            at_7 = (&kernelContext_29)->params_0->out_bases_0.x + 12U * (_S536 + j_12);
        }
        else
        {
            at_7 = (&kernelContext_29)->params_0->out_bases_0.y + 12U * (_S537 + j_12 - _S531);
        }
        uint _S538 = 14U * (_S529 + j_12);
        *((&kernelContext_29)->nodes_0+(_S538 + 5U)) = packed_float4(float4((&kernelContext_29)->lout_0[at_7], (&kernelContext_29)->lout_0[at_7 + 1U], (&kernelContext_29)->lout_0[at_7 + 2U], (&kernelContext_29)->lout_0[at_7 + 3U])) ;
        *((&kernelContext_29)->nodes_0+(_S538 + 6U)) = packed_float4(float4((&kernelContext_29)->lout_0[at_7 + 4U], (&kernelContext_29)->lout_0[at_7 + 5U], (&kernelContext_29)->lout_0[at_7 + 6U], (&kernelContext_29)->lout_0[at_7 + 7U])) ;
        *((&kernelContext_29)->nodes_0+(_S538 + 7U)) = packed_float4(float4((&kernelContext_29)->lout_0[at_7 + 8U], (&kernelContext_29)->lout_0[at_7 + 9U], (&kernelContext_29)->lout_0[at_7 + 10U], (&kernelContext_29)->lout_0[at_7 + 11U])) ;
        j_12 = j_12 + 256U;
    }
    j_12 = t_15;
    for(;;)
    {
        if(j_12 < _S534)
        {
        }
        else
        {
            break;
        }
        uint b_10 = _S533 + j_12;
        uint _S539 = (&kernelContext_29)->params_0->out_bases2_0.x + 8U * b_10;
        at_7 = 0U;
        uint i_6 = 1U;
        for(;;)
        {
            if(i_6 < 8U)
            {
            }
            else
            {
                break;
            }
            if((&kernelContext_29)->lout_0[_S539 + i_6] > ((&kernelContext_29)->lout_0)[_S539 + at_7])
            {
                at_7 = i_6;
            }
            i_6 = i_6 + 1U;
        }
        uint _S540 = 2U * b_10;
        *((&kernelContext_29)->bonds_0+_S540) = packed_float4(float4((float4(*((&kernelContext_29)->bonds_0+_S540)) ).x, (&kernelContext_29)->lout_0[(&kernelContext_29)->params_0->out_bases_0.z + b_10], (&kernelContext_29)->lout_0[(&kernelContext_29)->params_0->out_bases_0.w + b_10], (as_type<float>((at_7 + 1U))))) ;
        j_12 = j_12 + 256U;
    }
    return;
}

