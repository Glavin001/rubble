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
@binding(7) @group(0) var<storage, read> node_frame_0 : array<vec4<f32>>;

struct BlastNode_std430_0
{
    @align(16) w_ang_0 : f32,
    @align(4) w_lin_0 : f32,
    @align(8) ref_begin_0 : u32,
    @align(4) ref_end_0 : u32,
    @align(16) position_0 : vec4<f32>,
};

@binding(1) @group(0) var<storage, read> nodes_0 : array<BlastNode_std430_0>;

@binding(2) @group(0) var<storage, read> refs_0 : array<u32>;

struct BlastBond_std430_0
{
    @align(16) n0_0 : u32,
    @align(4) n1_0 : u32,
    @align(8) s_0 : f32,
    @align(4) health_0 : f32,
    @align(16) o0_0 : vec4<f32>,
    @align(16) o1_0 : vec4<f32>,
};

@binding(3) @group(0) var<storage, read> bonds_0 : array<BlastBond_std430_0>;

@binding(4) @group(0) var<storage, read> x_0 : array<vec4<f32>>;

@binding(5) @group(0) var<storage, read> stiffness_0 : array<vec4<f32>>;

@binding(6) @group(0) var<storage, read> centroids_0 : array<vec4<f32>>;

@binding(8) @group(0) var<storage, read_write> crush_rw_0 : array<vec4<f32>>;

fn linear_block_0( j_0 : u32,  v_0 : vec3<f32>) -> vec3<f32>
{
    var m_0 : array<f32, i32(6)>;
    var k_0 : u32 = u32(0);
    loop
    {
        if(k_0 < u32(6))
        {
        }
        else
        {
            break;
        }
        var _S1 : u32 = u32(12) + k_0;
        m_0[k_0] = stiffness_0[j_0 * u32(6) + _S1 / u32(4)][_S1 % u32(4)];
        k_0 = k_0 + u32(1);
    }
    var _S2 : f32 = v_0.x;
    var _S3 : f32 = v_0.y;
    var _S4 : f32 = v_0.z;
    return vec3<f32>(m_0[i32(0)] * _S2 + m_0[i32(3)] * _S3 + m_0[i32(4)] * _S4, m_0[i32(3)] * _S2 + m_0[i32(1)] * _S3 + m_0[i32(5)] * _S4, m_0[i32(4)] * _S2 + m_0[i32(5)] * _S3 + m_0[i32(2)] * _S4);
}

@compute
@workgroup_size(64, 1, 1)
fn blast_virial(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var k_1 : u32 = id_0.x;
    if(k_1 >= (params_0.node_count_0))
    {
        return;
    }
    var _S5 : vec3<f32> = vec3<f32>(0.0f);
    var node_0 : vec4<f32> = node_frame_0[u32(6) * k_1];
    var n_0 : BlastNode_std430_0 = nodes_0[k_1];
    var d_0 : vec3<f32>;
    var o_0 : vec3<f32>;
    if((node_0.w) > 0.0f)
    {
        var r_0 : u32 = n_0.ref_begin_0;
        d_0 = _S5;
        o_0 = _S5;
        loop
        {
            if(r_0 < (n_0.ref_end_0))
            {
            }
            else
            {
                break;
            }
            var ref_0 : u32 = refs_0[r_0];
            var j_1 : u32 = (ref_0 & (u32(2147483647)));
            var b_0 : BlastBond_std430_0 = bonds_0[j_1];
            if(!((b_0.health_0) > 0.0f))
            {
                r_0 = r_0 + u32(1);
                continue;
            }
            var xl_0 : vec3<f32> = x_0[j_1 * u32(2) + u32(1)].xyz;
            var xl_1 : vec3<f32>;
            if((params_0.rotation_0) == u32(2))
            {
                xl_1 = linear_block_0(j_1, xl_0);
            }
            else
            {
                xl_1 = xl_0;
            }
            var f_0 : vec3<f32> = xl_1 * vec3<f32>((b_0.s_0 * params_0.linear_scale_0));
            var f_1 : vec3<f32>;
            if(((ref_0 & (u32(2147483648)))) != u32(0))
            {
                f_1 = (vec3<f32>(0) - f_0);
            }
            else
            {
                f_1 = f_0;
            }
            var a_0 : vec3<f32> = centroids_0[j_1].xyz - node_0.xyz;
            var _S6 : f32 = a_0.x;
            var _S7 : f32 = f_1.y;
            var _S8 : f32 = a_0.y;
            var _S9 : f32 = f_1.x;
            var _S10 : f32 = f_1.z;
            var _S11 : f32 = a_0.z;
            var o_1 : vec3<f32> = o_0 + vec3<f32>(0.5f) * vec3<f32>(_S6 * _S7 + _S8 * _S9, _S6 * _S10 + _S11 * _S9, _S8 * _S10 + _S11 * _S7);
            d_0 = d_0 + a_0 * f_1;
            o_0 = o_1;
            r_0 = r_0 + u32(1);
        }
    }
    else
    {
        d_0 = _S5;
        o_0 = _S5;
    }
    var _S12 : u32 = u32(2) * k_1;
    crush_rw_0[params_0.r_virial_0 + _S12] = vec4<f32>(d_0, 0.0f);
    crush_rw_0[params_0.r_virial_0 + _S12 + u32(1)] = vec4<f32>(o_0, 0.0f);
    return;
}

