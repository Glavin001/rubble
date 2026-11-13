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
    @align(8) a_events_0 : u32,
    @align(4) ev_broken_cap_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : BlastParams_std140_0;
@binding(5) @group(0) var<storage, read_write> view_0 : array<atomic<u32>>;

struct BlastBond_std430_0
{
    @align(16) n0_0 : u32,
    @align(4) n1_0 : u32,
    @align(8) s_0 : f32,
    @align(4) health_0 : f32,
    @align(16) o0_0 : vec4<f32>,
    @align(16) o1_0 : vec4<f32>,
};

@binding(1) @group(0) var<storage, read> bonds_0 : array<BlastBond_std430_0>;

@binding(2) @group(0) var<storage, read> out_0 : array<vec4<f32>>;

@binding(3) @group(0) var<storage, read> geometry_0 : array<vec4<f32>>;

@binding(4) @group(0) var<storage, read> materials_0 : array<vec4<f32>>;

@compute
@workgroup_size(64, 1, 1)
fn blast_view_clear(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : u32 = id_0.x;
    if(_S1 < (params_0.node_count_0))
    {
        atomicStore(&(view_0[_S1]), u32(0));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn blast_view(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var j_0 : u32 = id_1.x;
    if(j_0 >= (params_0.bond_count_0))
    {
        return;
    }
    var bond_0 : BlastBond_std430_0 = bonds_0[j_0];
    if(!((bond_0.health_0) > 0.0f))
    {
        return;
    }
    var _S2 : u32 = j_0 * u32(4);
    var s_1 : vec4<f32> = out_0[_S2 + u32(2)];
    var tension_0 : f32 = out_0[_S2 + u32(3)].x;
    var _S3 : u32 = (bitcast<u32>((geometry_0[j_0 * u32(2) + u32(1)].y))) * u32(3);
    var m0_0 : vec4<f32> = materials_0[_S3];
    var m1_0 : vec4<f32> = materials_0[_S3 + u32(1)];
    var _S4 : f32 = m0_0.y;
    var u_0 : f32;
    if(_S4 > 0.0f)
    {
        u_0 = max(0.0f, s_1.w / _S4);
    }
    else
    {
        u_0 = 0.0f;
    }
    var _S5 : f32 = m0_0.w;
    if(_S5 > 0.0f)
    {
        u_0 = max(u_0, tension_0 / _S5);
    }
    var _S6 : f32 = m1_0.y;
    if(_S6 > 0.0f)
    {
        u_0 = max(u_0, s_1.y / _S6);
    }
    var bits_0 : u32 = (bitcast<u32>((max(u_0, 0.0f))));
    var _S7 : u32 = atomicMax(&(view_0[bond_0.n0_0]), bits_0);
    var _S8 : u32 = atomicMax(&(view_0[bond_0.n1_0]), bits_0);
    return;
}

