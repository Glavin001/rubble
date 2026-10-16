#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct ChunkState_0
{
    float3 u_0;
    float3 theta_0;
    float3 v_0;
    float3 w_0;
};

struct StepParams_0
{
    uint chunk_count_0;
    uint bond_count_0;
    float dt_0;
    uint substeps_0;
};

struct Bond_natural_0
{
    packed_uint4 chunks_0;
    packed_float4 t1_0;
    packed_float4 t2_0;
    packed_float4 normal_0;
    packed_float4 ra_0;
    packed_float4 rb_0;
    packed_float4 k_lin_0;
    packed_float4 k_ang_0;
    packed_float4 c_lin_0;
    packed_float4 c_ang_0;
    packed_float4 section_0;
};

struct Chunk_natural_0
{
    packed_float4 inv_inertia0_0;
    packed_float4 inv_inertia1_0;
    packed_float4 inv_inertia2_0;
    packed_float4 force_0;
    packed_float4 moment_0;
    packed_uint4 support_0;
};

struct KernelContext_0
{
    StepParams_0 constant* params_0;
    Bond_natural_0 device* bonds_0;
    packed_float4 device* state_0;
    packed_float4 device* bond_loads_0;
    uint device* chunk_bond_start_0;
    uint device* chunk_bonds_0;
    Chunk_natural_0 device* chunks_1;
    packed_uint4 device* islands_0;
    array<float3, int(160)> threadgroup* g_u_0;
    array<float3, int(160)> threadgroup* g_theta_0;
    array<float3, int(160)> threadgroup* g_v_0;
    array<float3, int(160)> threadgroup* g_w_0;
    array<float4, int(400)> threadgroup* g_fa_0;
    array<float3, int(400)> threadgroup* g_ma_0;
    array<float3, int(400)> threadgroup* g_mb_0;
};

ChunkState_0 load_state_0(uint c_0, KernelContext_0 thread* kernelContext_0)
{
    thread ChunkState_0 s_0;
    uint _S1 = 4U * c_0;
    (&s_0)->u_0 = (float4(*(kernelContext_0->state_0+_S1)) ).xyz;
    (&s_0)->theta_0 = (float4(*(kernelContext_0->state_0+(_S1 + 1U))) ).xyz;
    (&s_0)->v_0 = (float4(*(kernelContext_0->state_0+(_S1 + 2U))) ).xyz;
    (&s_0)->w_0 = (float4(*(kernelContext_0->state_0+(_S1 + 3U))) ).xyz;
    return s_0;
}

struct BondLoads_0
{
    float4 fa_0;
    float3 ma_0;
    float3 mb_0;
};

float3 to_local_0(uint _S2, float3 _S3, KernelContext_0 thread* kernelContext_1)
{
    Bond_natural_0 device* _S4 = kernelContext_1->bonds_0+_S2;
    return float3(dot(_S3, (float4(_S4->t1_0) ).xyz), dot(_S3, (float4(_S4->t2_0) ).xyz), dot(_S3, (float4(_S4->normal_0) ).xyz));
}

float3 to_body_0(uint _S5, float3 _S6, KernelContext_0 thread* kernelContext_2)
{
    Bond_natural_0 device* _S7 = kernelContext_2->bonds_0+_S5;
    return (float4(_S7->t1_0) ).xyz * float3(_S6.x)  + (float4(_S7->t2_0) ).xyz * float3(_S6.y)  + (float4(_S7->normal_0) ).xyz * float3(_S6.z) ;
}

BondLoads_0 bond_math_0(uint _S8, const ChunkState_0 thread* _S9, const ChunkState_0 thread* _S10, KernelContext_0 thread* kernelContext_3)
{
    Bond_natural_0 device* _S11 = kernelContext_3->bonds_0+_S8;
    float3 ra_1 = (float4(_S11->ra_0) ).xyz;
    float3 rb_1 = (float4(_S11->rb_0) ).xyz;
    float3 _S12 = _S10->theta_0;
    float3 _S13 = _S9->theta_0;
    float3 _S14 = to_local_0(_S8, _S10->u_0 + cross(_S10->theta_0, rb_1) - (_S9->u_0 + cross(_S9->theta_0, ra_1)), kernelContext_3);
    float3 _S15 = to_local_0(_S8, _S12 - _S13, kernelContext_3);
    float3 _S16 = _S10->w_0;
    float3 _S17 = _S9->w_0;
    float3 _S18 = to_local_0(_S8, _S10->v_0 + cross(_S10->w_0, rb_1) - (_S9->v_0 + cross(_S9->w_0, ra_1)), kernelContext_3);
    float3 _S19 = to_local_0(_S8, _S16 - _S17, kernelContext_3);
    float3 _S20 = (float4(_S11->k_lin_0) ).xyz * _S14;
    float3 _S21 = (float4(_S11->k_ang_0) ).xyz * _S15;
    float3 q_ang_0 = _S21 + (float4(_S11->c_ang_0) ).xyz * _S19;
    float4 _S22 = float4(_S11->section_0) ;
    float stress_0 = abs(_S20.z) / _S22.x + abs(_S21.x) / _S22.y + abs(_S21.y) / _S22.z;
    float3 _S23 = to_body_0(_S8, _S20 + (float4(_S11->c_lin_0) ).xyz * _S18, kernelContext_3);
    float3 _S24 = to_body_0(_S8, q_ang_0, kernelContext_3);
    thread BondLoads_0 o_0;
    (&o_0)->fa_0 = float4(_S23, stress_0);
    (&o_0)->ma_0 = _S24 + cross(ra_1, _S23);
    (&o_0)->mb_0 = - _S24 + cross(rb_1, - _S23);
    return o_0;
}

void bond_phase_0(uint i_0, KernelContext_0 thread* kernelContext_4)
{
    uint4 _S25 = uint4((kernelContext_4->bonds_0+i_0)->chunks_0) ;
    ChunkState_0 _S26 = load_state_0(_S25.x, kernelContext_4);
    ChunkState_0 _S27 = load_state_0(_S25.y, kernelContext_4);
    thread ChunkState_0 _S28 = _S26;
    thread ChunkState_0 _S29 = _S27;
    BondLoads_0 _S30 = bond_math_0(i_0, &_S28, &_S29, kernelContext_4);
    uint _S31 = 3U * i_0;
    *(kernelContext_4->bond_loads_0+_S31) = packed_float4(_S30.fa_0) ;
    *(kernelContext_4->bond_loads_0+(_S31 + 1U)) = packed_float4(float4(_S30.ma_0, 0.0f)) ;
    *(kernelContext_4->bond_loads_0+(_S31 + 2U)) = packed_float4(float4(_S30.mb_0, 0.0f)) ;
    return;
}

[[kernel]] void bond_forces(uint3 id_0 [[thread_position_in_grid]], StepParams_0 constant* params_1 [[buffer(0)]], Bond_natural_0 device* bonds_1 [[buffer(1)]], packed_float4 device* state_1 [[buffer(5)]], packed_float4 device* bond_loads_1 [[buffer(6)]], uint device* chunk_bond_start_1 [[buffer(3)]], uint device* chunk_bonds_1 [[buffer(4)]], Chunk_natural_0 device* chunks_2 [[buffer(2)]], packed_uint4 device* islands_1 [[buffer(7)]])
{
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->params_0 = params_1;
    (&kernelContext_5)->bonds_0 = bonds_1;
    (&kernelContext_5)->state_0 = state_1;
    (&kernelContext_5)->bond_loads_0 = bond_loads_1;
    (&kernelContext_5)->chunk_bond_start_0 = chunk_bond_start_1;
    (&kernelContext_5)->chunk_bonds_0 = chunk_bonds_1;
    (&kernelContext_5)->chunks_1 = chunks_2;
    (&kernelContext_5)->islands_0 = islands_1;
    threadgroup array<float3, int(160)> g_u_1;
    (&kernelContext_5)->g_u_0 = &g_u_1;
    threadgroup array<float3, int(160)> g_theta_1;
    (&kernelContext_5)->g_theta_0 = &g_theta_1;
    threadgroup array<float3, int(160)> g_v_1;
    (&kernelContext_5)->g_v_0 = &g_v_1;
    threadgroup array<float3, int(160)> g_w_1;
    (&kernelContext_5)->g_w_0 = &g_w_1;
    threadgroup array<float4, int(400)> g_fa_1;
    (&kernelContext_5)->g_fa_0 = &g_fa_1;
    threadgroup array<float3, int(400)> g_ma_1;
    (&kernelContext_5)->g_ma_0 = &g_ma_1;
    threadgroup array<float3, int(400)> g_mb_1;
    (&kernelContext_5)->g_mb_0 = &g_mb_1;
    uint _S32 = id_0.x;
    if(_S32 < (params_1->bond_count_0))
    {
        bond_phase_0(_S32, &kernelContext_5);
    }
    return;
}

ChunkState_0 chunk_math_0(const Chunk_natural_0 thread* ch_0, const ChunkState_0 thread* st_0, float3 fi_0, float3 mi_0, float dt_1)
{
    thread ChunkState_0 _S33 = *st_0;
    float4 _S34 = float4(ch_0->force_0) ;
    float3 f_0 = _S34.xyz + fi_0;
    float4 _S35 = float4(ch_0->moment_0) ;
    float3 m_0 = _S35.xyz + mi_0;
    float3 alpha_0 = float3(dot((float4(ch_0->inv_inertia0_0) ).xyz, m_0), dot((float4(ch_0->inv_inertia1_0) ).xyz, m_0), dot((float4(ch_0->inv_inertia2_0) ).xyz, m_0)) * float3(_S35.w) ;
    uint support_1 = (uint4(ch_0->support_0) ).x;
    if(support_1 == 1U)
    {
        float3 _S36 = float3(0.0f) ;
        (&_S33)->v_0 = _S36;
        (&_S33)->w_0 = _S36;
    }
    else
    {
        float3 _S37 = float3(dt_1) ;
        float3 _S38 = (&_S33)->w_0 + alpha_0 * _S37;
        (&_S33)->w_0 = _S38;
        (&_S33)->theta_0 = (&_S33)->theta_0 + _S38 * _S37;
        if(support_1 == 2U)
        {
            (&_S33)->v_0 = float3(0.0f) ;
        }
        else
        {
            float3 _S39 = (&_S33)->v_0 + f_0 * float3((dt_1 * _S34.w)) ;
            (&_S33)->v_0 = _S39;
            (&_S33)->u_0 = (&_S33)->u_0 + _S39 * _S37;
        }
    }
    return _S33;
}

void store_state_0(uint c_1, const ChunkState_0 thread* s_1, float stress_1, KernelContext_0 thread* kernelContext_6)
{
    uint _S40 = 4U * c_1;
    *(kernelContext_6->state_0+_S40) = packed_float4(float4(s_1->u_0, stress_1)) ;
    *(kernelContext_6->state_0+(_S40 + 1U)) = packed_float4(float4(s_1->theta_0, 0.0f)) ;
    *(kernelContext_6->state_0+(_S40 + 2U)) = packed_float4(float4(s_1->v_0, 0.0f)) ;
    *(kernelContext_6->state_0+(_S40 + 3U)) = packed_float4(float4(s_1->w_0, 0.0f)) ;
    return;
}

void chunk_phase_0(uint c_2, float dt_2, KernelContext_0 thread* kernelContext_7)
{
    float3 _S41 = float3(0.0f) ;
    uint _S42 = kernelContext_7->chunk_bond_start_0[c_2];
    float stress_2 = 0.0f;
    uint k_0 = _S42;
    float3 fi_1 = _S41;
    float3 mi_1 = _S41;
    for(;;)
    {
        if(k_0 < (kernelContext_7->chunk_bond_start_0)[c_2 + 1U])
        {
        }
        else
        {
            break;
        }
        uint e_0 = kernelContext_7->chunk_bonds_0[k_0];
        uint _S43 = 3U * (e_0 >> 1U);
        float4 _S44 = float4(*(kernelContext_7->bond_loads_0+_S43)) ;
        if((e_0 & 1U) == 0U)
        {
            float3 mi_2 = mi_1 + (float4(*(kernelContext_7->bond_loads_0+(_S43 + 1U))) ).xyz;
            fi_1 = fi_1 + _S44.xyz;
            mi_1 = mi_2;
        }
        else
        {
            float3 mi_3 = mi_1 + (float4(*(kernelContext_7->bond_loads_0+(_S43 + 2U))) ).xyz;
            fi_1 = fi_1 + - _S44.xyz;
            mi_1 = mi_3;
        }
        float _S45 = max(stress_2, _S44.w);
        uint _S46 = k_0 + 1U;
        stress_2 = _S45;
        k_0 = _S46;
    }
    Chunk_natural_0 _S47 = kernelContext_7->chunks_1[c_2];
    ChunkState_0 _S48 = load_state_0(c_2, kernelContext_7);
    thread Chunk_natural_0 _S49 = _S47;
    thread ChunkState_0 _S50 = _S48;
    ChunkState_0 _S51 = chunk_math_0(&_S49, &_S50, fi_1, mi_1, dt_2);
    thread ChunkState_0 _S52 = _S51;
    store_state_0(c_2, &_S52, stress_2, kernelContext_7);
    return;
}

[[kernel]] void chunk_integrate(uint3 id_1 [[thread_position_in_grid]], StepParams_0 constant* params_2 [[buffer(0)]], Bond_natural_0 device* bonds_2 [[buffer(1)]], packed_float4 device* state_2 [[buffer(5)]], packed_float4 device* bond_loads_2 [[buffer(6)]], uint device* chunk_bond_start_2 [[buffer(3)]], uint device* chunk_bonds_2 [[buffer(4)]], Chunk_natural_0 device* chunks_3 [[buffer(2)]], packed_uint4 device* islands_2 [[buffer(7)]])
{
    thread KernelContext_0 kernelContext_8;
    (&kernelContext_8)->params_0 = params_2;
    (&kernelContext_8)->bonds_0 = bonds_2;
    (&kernelContext_8)->state_0 = state_2;
    (&kernelContext_8)->bond_loads_0 = bond_loads_2;
    (&kernelContext_8)->chunk_bond_start_0 = chunk_bond_start_2;
    (&kernelContext_8)->chunk_bonds_0 = chunk_bonds_2;
    (&kernelContext_8)->chunks_1 = chunks_3;
    (&kernelContext_8)->islands_0 = islands_2;
    threadgroup array<float3, int(160)> g_u_2;
    (&kernelContext_8)->g_u_0 = &g_u_2;
    threadgroup array<float3, int(160)> g_theta_2;
    (&kernelContext_8)->g_theta_0 = &g_theta_2;
    threadgroup array<float3, int(160)> g_v_2;
    (&kernelContext_8)->g_v_0 = &g_v_2;
    threadgroup array<float3, int(160)> g_w_2;
    (&kernelContext_8)->g_w_0 = &g_w_2;
    threadgroup array<float4, int(400)> g_fa_2;
    (&kernelContext_8)->g_fa_0 = &g_fa_2;
    threadgroup array<float3, int(400)> g_ma_2;
    (&kernelContext_8)->g_ma_0 = &g_ma_2;
    threadgroup array<float3, int(400)> g_mb_2;
    (&kernelContext_8)->g_mb_0 = &g_mb_2;
    uint _S53 = id_1.x;
    if(_S53 < (params_2->chunk_count_0))
    {
        chunk_phase_0(_S53, (&kernelContext_8)->params_0->dt_0, &kernelContext_8);
    }
    return;
}

[[kernel]] void island_substeps(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], StepParams_0 constant* params_3 [[buffer(0)]], Bond_natural_0 device* bonds_3 [[buffer(1)]], packed_float4 device* state_3 [[buffer(5)]], packed_float4 device* bond_loads_3 [[buffer(6)]], uint device* chunk_bond_start_3 [[buffer(3)]], uint device* chunk_bonds_3 [[buffer(4)]], Chunk_natural_0 device* chunks_4 [[buffer(2)]], packed_uint4 device* islands_3 [[buffer(7)]])
{
    thread KernelContext_0 kernelContext_9;
    (&kernelContext_9)->params_0 = params_3;
    (&kernelContext_9)->bonds_0 = bonds_3;
    (&kernelContext_9)->state_0 = state_3;
    (&kernelContext_9)->bond_loads_0 = bond_loads_3;
    (&kernelContext_9)->chunk_bond_start_0 = chunk_bond_start_3;
    (&kernelContext_9)->chunk_bonds_0 = chunk_bonds_3;
    (&kernelContext_9)->chunks_1 = chunks_4;
    (&kernelContext_9)->islands_0 = islands_3;
    threadgroup array<float3, int(160)> g_u_3;
    (&kernelContext_9)->g_u_0 = &g_u_3;
    threadgroup array<float3, int(160)> g_theta_3;
    (&kernelContext_9)->g_theta_0 = &g_theta_3;
    threadgroup array<float3, int(160)> g_v_3;
    (&kernelContext_9)->g_v_0 = &g_v_3;
    threadgroup array<float3, int(160)> g_w_3;
    (&kernelContext_9)->g_w_0 = &g_w_3;
    threadgroup array<float4, int(400)> g_fa_3;
    (&kernelContext_9)->g_fa_0 = &g_fa_3;
    threadgroup array<float3, int(400)> g_ma_3;
    (&kernelContext_9)->g_ma_0 = &g_ma_3;
    threadgroup array<float3, int(400)> g_mb_3;
    (&kernelContext_9)->g_mb_0 = &g_mb_3;
    uint4 _S54 = uint4(*(islands_3+group_0.x)) ;
    float _S55 = params_3->dt_0;
    uint s_2 = 0U;
    for(;;)
    {
        if(s_2 < ((&kernelContext_9)->params_0->substeps_0))
        {
        }
        else
        {
            break;
        }
        uint _S56 = thread_0.x;
        uint i_1 = _S54.z + _S56;
        for(;;)
        {
            if(i_1 < (_S54.w))
            {
            }
            else
            {
                break;
            }
            bond_phase_0(i_1, &kernelContext_9);
            i_1 = i_1 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        uint c_3 = _S54.x + _S56;
        for(;;)
        {
            if(c_3 < (_S54.y))
            {
            }
            else
            {
                break;
            }
            chunk_phase_0(c_3, _S55, &kernelContext_9);
            c_3 = c_3 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        s_2 = s_2 + 1U;
    }
    return;
}

ChunkState_0 shared_state_0(uint local_0, KernelContext_0 thread* kernelContext_10)
{
    thread ChunkState_0 s_3;
    (&s_3)->u_0 = (*kernelContext_10->g_u_0)[local_0];
    (&s_3)->theta_0 = (*kernelContext_10->g_theta_0)[local_0];
    (&s_3)->v_0 = (*kernelContext_10->g_v_0)[local_0];
    (&s_3)->w_0 = (*kernelContext_10->g_w_0)[local_0];
    return s_3;
}

[[kernel]] void island_shared_substeps(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], StepParams_0 constant* params_4 [[buffer(0)]], Bond_natural_0 device* bonds_4 [[buffer(1)]], packed_float4 device* state_4 [[buffer(5)]], packed_float4 device* bond_loads_4 [[buffer(6)]], uint device* chunk_bond_start_4 [[buffer(3)]], uint device* chunk_bonds_4 [[buffer(4)]], Chunk_natural_0 device* chunks_5 [[buffer(2)]], packed_uint4 device* islands_4 [[buffer(7)]])
{
    thread KernelContext_0 kernelContext_11;
    (&kernelContext_11)->params_0 = params_4;
    (&kernelContext_11)->bonds_0 = bonds_4;
    (&kernelContext_11)->state_0 = state_4;
    (&kernelContext_11)->bond_loads_0 = bond_loads_4;
    (&kernelContext_11)->chunk_bond_start_0 = chunk_bond_start_4;
    (&kernelContext_11)->chunk_bonds_0 = chunk_bonds_4;
    (&kernelContext_11)->chunks_1 = chunks_5;
    (&kernelContext_11)->islands_0 = islands_4;
    threadgroup array<float3, int(160)> g_u_4;
    (&kernelContext_11)->g_u_0 = &g_u_4;
    threadgroup array<float3, int(160)> g_theta_4;
    (&kernelContext_11)->g_theta_0 = &g_theta_4;
    threadgroup array<float3, int(160)> g_v_4;
    (&kernelContext_11)->g_v_0 = &g_v_4;
    threadgroup array<float3, int(160)> g_w_4;
    (&kernelContext_11)->g_w_0 = &g_w_4;
    threadgroup array<float4, int(400)> g_fa_4;
    (&kernelContext_11)->g_fa_0 = &g_fa_4;
    threadgroup array<float3, int(400)> g_ma_4;
    (&kernelContext_11)->g_ma_0 = &g_ma_4;
    threadgroup array<float3, int(400)> g_mb_4;
    (&kernelContext_11)->g_mb_0 = &g_mb_4;
    uint4 _S57 = uint4(*(islands_4+group_1.x)) ;
    uint chunk0_0 = _S57.x;
    uint chunk_count_1 = _S57.y - chunk0_0;
    uint _S58 = _S57.z;
    uint _S59 = _S57.w - _S58;
    float _S60 = params_4->dt_0;
    uint _S61 = thread_1.x;
    uint c_4 = _S61;
    for(;;)
    {
        if(c_4 < chunk_count_1)
        {
        }
        else
        {
            break;
        }
        ChunkState_0 _S62 = load_state_0(chunk0_0 + c_4, &kernelContext_11);
        (*(&kernelContext_11)->g_u_0)[c_4] = _S62.u_0;
        (*(&kernelContext_11)->g_theta_0)[c_4] = _S62.theta_0;
        (*(&kernelContext_11)->g_v_0)[c_4] = _S62.v_0;
        (*(&kernelContext_11)->g_w_0)[c_4] = _S62.w_0;
        c_4 = c_4 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float peak_stress_0 = 0.0f;
    uint s_4 = 0U;
    for(;;)
    {
        if(s_4 < ((&kernelContext_11)->params_0->substeps_0))
        {
        }
        else
        {
            break;
        }
        uint i_2 = _S61;
        for(;;)
        {
            if(i_2 < _S59)
            {
            }
            else
            {
                break;
            }
            uint _S63 = _S58 + i_2;
            uint4 _S64 = uint4(((&kernelContext_11)->bonds_0+_S63)->chunks_0) ;
            ChunkState_0 _S65 = shared_state_0(_S64.x - chunk0_0, &kernelContext_11);
            ChunkState_0 _S66 = shared_state_0(_S64.y - chunk0_0, &kernelContext_11);
            thread ChunkState_0 _S67 = _S65;
            thread ChunkState_0 _S68 = _S66;
            BondLoads_0 _S69 = bond_math_0(_S63, &_S67, &_S68, &kernelContext_11);
            (*(&kernelContext_11)->g_fa_0)[i_2] = _S69.fa_0;
            (*(&kernelContext_11)->g_ma_0)[i_2] = _S69.ma_0;
            (*(&kernelContext_11)->g_mb_0)[i_2] = _S69.mb_0;
            i_2 = i_2 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(_S61 < chunk_count_1)
        {
            float3 _S70 = float3(0.0f) ;
            uint _S71 = chunk0_0 + _S61;
            uint _S72 = (&kernelContext_11)->chunk_bond_start_0[_S71];
            float peak_0 = 0.0f;
            uint k_1 = _S72;
            float3 fi_2 = _S70;
            float3 mi_4 = _S70;
            for(;;)
            {
                if(k_1 < ((&kernelContext_11)->chunk_bond_start_0)[_S71 + 1U])
                {
                }
                else
                {
                    break;
                }
                uint e_1 = (&kernelContext_11)->chunk_bonds_0[k_1];
                uint bond_0 = (e_1 >> 1U) - _S58;
                float4 fa_1 = (*(&kernelContext_11)->g_fa_0)[bond_0];
                if((e_1 & 1U) == 0U)
                {
                    float3 mi_5 = mi_4 + (*(&kernelContext_11)->g_ma_0)[bond_0];
                    fi_2 = fi_2 + fa_1.xyz;
                    mi_4 = mi_5;
                }
                else
                {
                    float3 mi_6 = mi_4 + (*(&kernelContext_11)->g_mb_0)[bond_0];
                    fi_2 = fi_2 + - fa_1.xyz;
                    mi_4 = mi_6;
                }
                float _S73 = max(peak_0, fa_1.w);
                uint _S74 = k_1 + 1U;
                peak_0 = _S73;
                k_1 = _S74;
            }
            Chunk_natural_0 _S75 = (&kernelContext_11)->chunks_1[_S71];
            ChunkState_0 _S76 = shared_state_0(_S61, &kernelContext_11);
            thread Chunk_natural_0 _S77 = _S75;
            thread ChunkState_0 _S78 = _S76;
            ChunkState_0 _S79 = chunk_math_0(&_S77, &_S78, fi_2, mi_4, _S60);
            (*(&kernelContext_11)->g_u_0)[_S61] = _S79.u_0;
            (*(&kernelContext_11)->g_theta_0)[_S61] = _S79.theta_0;
            (*(&kernelContext_11)->g_v_0)[_S61] = _S79.v_0;
            (*(&kernelContext_11)->g_w_0)[_S61] = _S79.w_0;
            peak_stress_0 = peak_0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_4 = s_4 + 1U;
    }
    if(_S61 < chunk_count_1)
    {
        uint _S80 = chunk0_0 + _S61;
        ChunkState_0 _S81 = shared_state_0(_S61, &kernelContext_11);
        if(((&kernelContext_11)->params_0->substeps_0) > 0U)
        {
        }
        else
        {
            peak_stress_0 = (float4(*((&kernelContext_11)->state_0+4U * _S80)) ).w;
        }
        thread ChunkState_0 _S82 = _S81;
        store_state_0(_S80, &_S82, peak_stress_0, &kernelContext_11);
    }
    return;
}

