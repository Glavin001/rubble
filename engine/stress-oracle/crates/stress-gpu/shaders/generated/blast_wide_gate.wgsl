@binding(2) @group(0) var<storage, read> meta_0 : array<u32>;

struct BlastParams_std140_0
{
    @align(16) island_count_0 : u32,
    @align(4) max_iterations_0 : u32,
    @align(8) warm_0 : u32,
    @align(4) skip_settled_0 : u32,
    @align(16) tolerance_0 : f32,
    @align(4) recip_length_0 : f32,
    @align(8) node_count_0 : u32,
    @align(4) bond_count_0 : u32,
    @align(16) linear_scale_0 : f32,
    @align(4) angular_scale_0 : f32,
    @align(8) bend_gain_max_0 : f32,
    @align(4) fibres_0 : u32,
    @align(16) dt_0 : f32,
    @align(4) damage_rate_0 : f32,
    @align(8) apply_damage_0 : u32,
    @align(4) force_tolerance_0 : f32,
    @align(16) gravity_0 : vec4<f32>,
    @align(16) w_alive_0 : u32,
    @align(4) a_dirty_0 : u32,
    @align(8) a_counters_0 : u32,
    @align(4) wide_min_0 : u32,
    @align(16) wide_rows_0 : u32,
    @align(4) wide_blocks_0 : u32,
    @align(8) opaque_zero_0 : u32,
    @align(4) first_polynomial_0 : u32,
    @align(16) rotation_0 : u32,
    @align(4) node_vectors_0 : u32,
    @align(8) carry_0 : u32,
    @align(4) section_bending_0 : u32,
    @align(16) chunk_loads_0 : u32,
    @align(4) r_virial_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : BlastParams_std140_0;
@binding(1) @group(0) var<storage, read> wide_0 : array<vec4<f32>>;

@binding(3) @group(0) var<storage, read_write> grids_0 : array<u32>;

@compute
@workgroup_size(1, 1, 1)
fn w_gate(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : u32 = min(meta_0[i32(0)], params_0.wide_rows_0);
    var row_0 : u32 = u32(0);
    loop
    {
        if(row_0 < _S1)
        {
        }
        else
        {
            break;
        }
        if((((bitcast<u32>((wide_0[row_0 * u32(16) + u32(2)].w))) & (u32(1)))) != u32(0))
        {
            return;
        }
        row_0 = row_0 + u32(1);
    }
    grids_0[i32(0)] = u32(0);
    grids_0[i32(4)] = u32(0);
    grids_0[i32(8)] = u32(0);
    return;
}

