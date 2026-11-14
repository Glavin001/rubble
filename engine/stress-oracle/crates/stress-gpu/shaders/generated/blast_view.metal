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
    uint a_events_0;
    uint ev_broken_cap_0;
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
    atomic<uint> device* view_0;
    BlastBond_natural_0 device* bonds_0;
    packed_float4 device* out_0;
    packed_float4 device* geometry_0;
    packed_float4 device* materials_0;
};

[[kernel]] void blast_view_clear(uint3 id_0 [[thread_position_in_grid]], BlastParams_0 constant* params_1 [[buffer(0)]], atomic<uint> device* view_1 [[buffer(5)]], BlastBond_natural_0 device* bonds_1 [[buffer(1)]], packed_float4 device* out_1 [[buffer(2)]], packed_float4 device* geometry_1 [[buffer(3)]], packed_float4 device* materials_1 [[buffer(4)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->view_0 = view_1;
    (&kernelContext_0)->bonds_0 = bonds_1;
    (&kernelContext_0)->out_0 = out_1;
    (&kernelContext_0)->geometry_0 = geometry_1;
    (&kernelContext_0)->materials_0 = materials_1;
    uint _S1 = id_0.x;
    if(_S1 < (params_1->node_count_0))
    {
        atomic_store_explicit((&kernelContext_0)->view_0+_S1, 0U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void blast_view(uint3 id_1 [[thread_position_in_grid]], BlastParams_0 constant* params_2 [[buffer(0)]], atomic<uint> device* view_2 [[buffer(5)]], BlastBond_natural_0 device* bonds_2 [[buffer(1)]], packed_float4 device* out_2 [[buffer(2)]], packed_float4 device* geometry_2 [[buffer(3)]], packed_float4 device* materials_2 [[buffer(4)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->params_0 = params_2;
    (&kernelContext_1)->view_0 = view_2;
    (&kernelContext_1)->bonds_0 = bonds_2;
    (&kernelContext_1)->out_0 = out_2;
    (&kernelContext_1)->geometry_0 = geometry_2;
    (&kernelContext_1)->materials_0 = materials_2;
    uint j_0 = id_1.x;
    if(j_0 >= (params_2->bond_count_0))
    {
        return;
    }
    BlastBond_natural_0 bond_0 = (&kernelContext_1)->bonds_0[j_0];
    if(!((bond_0.health_0) > 0.0f))
    {
        return;
    }
    uint _S2 = j_0 * 4U;
    float4 _S3 = float4(*((&kernelContext_1)->out_0+(_S2 + 2U))) ;
    float tension_0 = (float4(*((&kernelContext_1)->out_0+(_S2 + 3U))) ).x;
    uint _S4 = (as_type<uint>(((float4(*((&kernelContext_1)->geometry_0+(j_0 * 2U + 1U))) ).y))) * 3U;
    float4 _S5 = float4(*((&kernelContext_1)->materials_0+_S4)) ;
    float4 _S6 = float4(*((&kernelContext_1)->materials_0+(_S4 + 1U))) ;
    float _S7 = _S5.y;
    float u_0;
    if(_S7 > 0.0f)
    {
        u_0 = max(0.0f, _S3.w / _S7);
    }
    else
    {
        u_0 = 0.0f;
    }
    float _S8 = _S5.w;
    if(_S8 > 0.0f)
    {
        u_0 = max(u_0, tension_0 / _S8);
    }
    float _S9 = _S6.y;
    if(_S9 > 0.0f)
    {
        u_0 = max(u_0, _S3.y / _S9);
    }
    uint bits_0 = (as_type<uint>((max(u_0, 0.0f))));
    uint _S10 = atomic_fetch_max_explicit((&kernelContext_1)->view_0+bond_0.n0_0, bits_0, memory_order_relaxed);
    uint _S11 = atomic_fetch_max_explicit((&kernelContext_1)->view_0+bond_0.n1_0, bits_0, memory_order_relaxed);
    return;
}

