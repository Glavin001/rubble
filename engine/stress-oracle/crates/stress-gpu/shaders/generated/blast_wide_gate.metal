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

struct KernelContext_0
{
    uint device* meta_0;
    BlastParams_0 constant* params_0;
    packed_float4 device* wide_0;
    uint device* grids_0;
};

[[kernel]] void w_gate(uint3 id_0 [[thread_position_in_grid]], uint device* meta_1 [[buffer(2)]], BlastParams_0 constant* params_1 [[buffer(0)]], packed_float4 device* wide_1 [[buffer(1)]], uint device* grids_1 [[buffer(3)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->meta_0 = meta_1;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->wide_0 = wide_1;
    (&kernelContext_0)->grids_0 = grids_1;
    uint _S1 = min(meta_1[int(0)], params_1->wide_rows_0);
    uint row_0 = 0U;
    for(;;)
    {
        if(row_0 < _S1)
        {
        }
        else
        {
            break;
        }
        if(((as_type<uint>(((float4(*((&kernelContext_0)->wide_0+(row_0 * 16U + 2U))) ).w))) & 1U) != 0U)
        {
            return;
        }
        row_0 = row_0 + 1U;
    }
    *((&kernelContext_0)->grids_0+int(0)) = 0U;
    *((&kernelContext_0)->grids_0+int(4)) = 0U;
    *((&kernelContext_0)->grids_0+int(8)) = 0U;
    return;
}

