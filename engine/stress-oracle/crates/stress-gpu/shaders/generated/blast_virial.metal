#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct BlastParams_0
{
    uint island_count_0;
    uint max_iterations_0;
    uint warm_0;
    uint skip_settled_0;
    float tolerance_0;
    float recip_length_0;
    uint node_count_0;
    uint bond_count_0;
    float linear_scale_0;
    float angular_scale_0;
    float bend_gain_max_0;
    uint fibres_0;
    float dt_0;
    float damage_rate_0;
    uint apply_damage_0;
    float force_tolerance_0;
    float4 gravity_0;
    uint w_alive_0;
    uint a_dirty_0;
    uint a_counters_0;
    uint wide_min_0;
    uint wide_rows_0;
    uint wide_blocks_0;
    uint opaque_zero_0;
    uint first_polynomial_0;
    uint rotation_0;
    uint node_vectors_0;
    uint carry_0;
    uint section_bending_0;
    uint chunk_loads_0;
    uint r_virial_0;
};

struct BlastNode_natural_0
{
    float w_ang_0;
    float w_lin_0;
    uint ref_begin_0;
    uint ref_end_0;
    packed_float4 position_0;
};

struct BlastBond_natural_0
{
    uint n0_0;
    uint n1_0;
    float s_0;
    float health_0;
    packed_float4 o0_0;
    packed_float4 o1_0;
};

struct KernelContext_0
{
    BlastParams_0 constant* params_0;
    packed_float4 device* node_frame_0;
    BlastNode_natural_0 device* nodes_0;
    uint device* refs_0;
    BlastBond_natural_0 device* bonds_0;
    packed_float4 device* x_0;
    packed_float4 device* stiffness_0;
    packed_float4 device* centroids_0;
    packed_float4 device* crush_rw_0;
};

float3 linear_block_0(uint j_0, float3 v_0, KernelContext_0 thread* kernelContext_0)
{
    thread array<float, int(6)> m_0;
    uint k_0 = 0U;
    for(;;)
    {
        if(k_0 < 6U)
        {
        }
        else
        {
            break;
        }
        uint _S1 = 12U + k_0;
        m_0[k_0] = (float4(*(kernelContext_0->stiffness_0+(j_0 * 6U + _S1 / 4U))) )[_S1 % 4U];
        k_0 = k_0 + 1U;
    }
    float _S2 = v_0.x;
    float _S3 = v_0.y;
    float _S4 = v_0.z;
    return float3(m_0[int(0)] * _S2 + m_0[int(3)] * _S3 + m_0[int(4)] * _S4, m_0[int(3)] * _S2 + m_0[int(1)] * _S3 + m_0[int(5)] * _S4, m_0[int(4)] * _S2 + m_0[int(5)] * _S3 + m_0[int(2)] * _S4);
}

[[kernel]] void blast_virial(uint3 id_0 [[thread_position_in_grid]], BlastParams_0 constant* params_1 [[buffer(0)]], packed_float4 device* node_frame_1 [[buffer(7)]], BlastNode_natural_0 device* nodes_1 [[buffer(1)]], uint device* refs_1 [[buffer(2)]], BlastBond_natural_0 device* bonds_1 [[buffer(3)]], packed_float4 device* x_1 [[buffer(4)]], packed_float4 device* stiffness_1 [[buffer(5)]], packed_float4 device* centroids_1 [[buffer(6)]], packed_float4 device* crush_rw_1 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->params_0 = params_1;
    (&kernelContext_1)->node_frame_0 = node_frame_1;
    (&kernelContext_1)->nodes_0 = nodes_1;
    (&kernelContext_1)->refs_0 = refs_1;
    (&kernelContext_1)->bonds_0 = bonds_1;
    (&kernelContext_1)->x_0 = x_1;
    (&kernelContext_1)->stiffness_0 = stiffness_1;
    (&kernelContext_1)->centroids_0 = centroids_1;
    (&kernelContext_1)->crush_rw_0 = crush_rw_1;
    uint k_1 = id_0.x;
    if(k_1 >= (params_1->node_count_0))
    {
        return;
    }
    float3 _S5 = float3(0.0f) ;
    float4 _S6 = float4(*((&kernelContext_1)->node_frame_0+6U * k_1)) ;
    BlastNode_natural_0 n_0 = (&kernelContext_1)->nodes_0[k_1];
    float3 d_0;
    float3 o_0;
    if((_S6.w) > 0.0f)
    {
        uint r_0 = n_0.ref_begin_0;
        d_0 = _S5;
        o_0 = _S5;
        for(;;)
        {
            if(r_0 < (n_0.ref_end_0))
            {
            }
            else
            {
                break;
            }
            uint ref_0 = (&kernelContext_1)->refs_0[r_0];
            uint j_1 = ref_0 & 2147483647U;
            BlastBond_natural_0 b_0 = (&kernelContext_1)->bonds_0[j_1];
            if(!((b_0.health_0) > 0.0f))
            {
                r_0 = r_0 + 1U;
                continue;
            }
            float3 xl_0 = (float4(*((&kernelContext_1)->x_0+(j_1 * 2U + 1U))) ).xyz;
            float3 xl_1;
            if(((&kernelContext_1)->params_0->rotation_0) == 2U)
            {
                float3 _S7 = linear_block_0(j_1, xl_0, &kernelContext_1);
                xl_1 = _S7;
            }
            else
            {
                xl_1 = xl_0;
            }
            float3 f_0 = xl_1 * float3((b_0.s_0 * (&kernelContext_1)->params_0->linear_scale_0)) ;
            float3 f_1;
            if((ref_0 & 2147483648U) != 0U)
            {
                f_1 = - f_0;
            }
            else
            {
                f_1 = f_0;
            }
            float3 a_0 = (float4(*((&kernelContext_1)->centroids_0+j_1)) ).xyz - _S6.xyz;
            float _S8 = a_0.x;
            float _S9 = f_1.y;
            float _S10 = a_0.y;
            float _S11 = f_1.x;
            float _S12 = f_1.z;
            float _S13 = a_0.z;
            float3 o_1 = o_0 + float3(0.5f)  * float3(_S8 * _S9 + _S10 * _S11, _S8 * _S12 + _S13 * _S11, _S10 * _S12 + _S13 * _S9);
            d_0 = d_0 + a_0 * f_1;
            o_0 = o_1;
            r_0 = r_0 + 1U;
        }
    }
    else
    {
        d_0 = _S5;
        o_0 = _S5;
    }
    uint _S14 = 2U * k_1;
    *((&kernelContext_1)->crush_rw_0+((&kernelContext_1)->params_0->r_virial_0 + _S14)) = packed_float4(float4(d_0, 0.0f)) ;
    *((&kernelContext_1)->crush_rw_0+((&kernelContext_1)->params_0->r_virial_0 + _S14 + 1U)) = packed_float4(float4(o_0, 0.0f)) ;
    return;
}

