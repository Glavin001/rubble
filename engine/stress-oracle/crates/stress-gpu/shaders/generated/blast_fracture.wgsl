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
@binding(1) @group(0) var<storage, read_write> work_0 : array<u32>;

@binding(3) @group(0) var<storage, read> frames_0 : array<vec4<f32>>;

@binding(4) @group(0) var<storage, read> bodies_0 : array<vec4<f32>>;

@binding(6) @group(0) var<storage, read> node_frame_0 : array<vec4<f32>>;

@binding(5) @group(0) var<storage, read> loads_0 : array<vec4<f32>>;

@binding(7) @group(0) var<storage, read_write> accel_0 : array<vec4<f32>>;

struct BlastBond_std430_0
{
    @align(16) n0_0 : u32,
    @align(4) n1_0 : u32,
    @align(8) s_0 : f32,
    @align(4) health_0 : f32,
    @align(16) o0_0 : vec4<f32>,
    @align(16) o1_0 : vec4<f32>,
};

@binding(8) @group(0) var<storage, read_write> bonds_0 : array<BlastBond_std430_0>;

@binding(10) @group(0) var<storage, read_write> x_0 : array<vec4<f32>>;

@binding(9) @group(0) var<storage, read_write> topo_0 : array<atomic<u32>>;

@binding(2) @group(0) var<storage, read_write> atom_0 : array<atomic<u32>>;

struct Quat_0
{
     w_0 : f32,
     x_1 : f32,
     y_0 : f32,
     z_0 : f32,
};

fn quat_of_0( q_0 : vec4<f32>) -> Quat_0
{
    var r_0 : Quat_0;
    r_0.x_1 = q_0.x;
    r_0.y_0 = q_0.y;
    r_0.z_0 = q_0.z;
    r_0.w_0 = q_0.w;
    return r_0;
}

fn conj_0( q_1 : Quat_0) -> Quat_0
{
    var c_0 : Quat_0;
    c_0.w_0 = q_1.w_0;
    c_0.x_1 = - q_1.x_1;
    c_0.y_0 = - q_1.y_0;
    c_0.z_0 = - q_1.z_0;
    return c_0;
}

fn quat_mul_0( a_0 : Quat_0,  o_0 : Quat_0) -> Quat_0
{
    var r_1 : Quat_0;
    r_1.w_0 = a_0.w_0 * o_0.w_0 - a_0.x_1 * o_0.x_1 - a_0.y_0 * o_0.y_0 - a_0.z_0 * o_0.z_0;
    r_1.x_1 = a_0.w_0 * o_0.x_1 + a_0.x_1 * o_0.w_0 + a_0.y_0 * o_0.z_0 - a_0.z_0 * o_0.y_0;
    r_1.y_0 = a_0.w_0 * o_0.y_0 - a_0.x_1 * o_0.z_0 + a_0.y_0 * o_0.w_0 + a_0.z_0 * o_0.x_1;
    r_1.z_0 = a_0.w_0 * o_0.z_0 + a_0.x_1 * o_0.y_0 - a_0.y_0 * o_0.x_1 + a_0.z_0 * o_0.w_0;
    return r_1;
}

fn normalized_0( q_2 : Quat_0) -> Quat_0
{
    var _S1 : f32 = q_2.w_0;
    var _S2 : f32 = q_2.x_1;
    var _S3 : f32 = q_2.y_0;
    var _S4 : f32 = q_2.z_0;
    var n_0 : f32 = sqrt(_S1 * _S1 + _S2 * _S2 + _S3 * _S3 + _S4 * _S4);
    var r_2 : Quat_0;
    r_2.w_0 = q_2.w_0 / n_0;
    r_2.x_1 = q_2.x_1 / n_0;
    r_2.y_0 = q_2.y_0 / n_0;
    r_2.z_0 = q_2.z_0 / n_0;
    return r_2;
}

fn rot_from_rubble_0( q_3 : Quat_0) -> Quat_0
{
    var axes_0 : Quat_0;
    axes_0.w_0 = 0.70710676908493042f;
    axes_0.x_1 = 0.70710676908493042f;
    axes_0.y_0 = 0.0f;
    axes_0.z_0 = 0.0f;
    return normalized_0(quat_mul_0(axes_0, q_3));
}

fn rotate_0( q_4 : Quat_0,  v_0 : vec3<f32>) -> vec3<f32>
{
    var qv_0 : vec3<f32> = vec3<f32>(q_4.x_1, q_4.y_0, q_4.z_0);
    var t_0 : vec3<f32> = cross(qv_0, v_0) * vec3<f32>(2.0f);
    return v_0 + t_0 * vec3<f32>(q_4.w_0) + cross(qv_0, t_0);
}

fn inverse_rotate_0( q_5 : Quat_0,  v_1 : vec3<f32>) -> vec3<f32>
{
    var c_1 : Quat_0;
    c_1.w_0 = q_5.w_0;
    c_1.x_1 = - q_5.x_1;
    c_1.y_0 = - q_5.y_0;
    c_1.z_0 = - q_5.z_0;
    return rotate_0(c_1, v_1);
}

fn from_rubble_0( v_2 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(v_2.x, - v_2.z, v_2.y);
}

@compute
@workgroup_size(64, 1, 1)
fn blast_loads(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var k_0 : u32 = id_0.x;
    if(k_0 >= (params_0.node_count_0))
    {
        return;
    }
    var b_0 : u32 = u32(7) * work_0[k_0];
    var _S5 : u32 = u32(2) * work_0[k_0];
    var q_6 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_0 + u32(1)])), conj_0(quat_of_0(frames_0[_S5 + u32(1)]))));
    var _S6 : u32 = u32(6) * k_0;
    var node_0 : vec4<f32> = node_frame_0[_S6];
    var f_0 : vec3<f32> = loads_0[k_0].yzw;
    var _S7 : vec3<f32> = params_0.gravity_0.xyz;
    var _S8 : f32 = node_0.w;
    var _S9 : bool = _S8 > 0.0f;
    var alpha_0 : vec3<f32>;
    if(_S9)
    {
        alpha_0 = f_0 / vec3<f32>(_S8);
    }
    else
    {
        alpha_0 = vec3<f32>(0.0f);
    }
    var a_1 : vec3<f32> = _S7 + alpha_0;
    var _S10 : vec3<f32> = vec3<f32>(0.0f);
    var _S11 : bool;
    if((params_0.chunk_loads_0) != u32(0))
    {
        _S11 = _S9;
    }
    else
    {
        _S11 = false;
    }
    var a_2 : vec3<f32>;
    if(_S11)
    {
        var torque_0 : vec3<f32> = node_frame_0[_S6 + u32(4)].xyz + node_frame_0[_S6 + u32(5)].xyz / vec3<f32>(params_0.dt_0);
        var a_3 : vec3<f32> = a_1 + (node_frame_0[_S6 + u32(2)].xyz + node_frame_0[_S6 + u32(3)].xyz / vec3<f32>(params_0.dt_0)) / vec3<f32>(_S8);
        var inertia_0 : f32 = node_frame_0[_S6 + u32(1)].x;
        if(inertia_0 > 0.0f)
        {
            alpha_0 = (vec3<f32>(0) - torque_0) / vec3<f32>(inertia_0);
        }
        else
        {
            alpha_0 = _S10;
        }
        a_2 = a_3;
    }
    else
    {
        alpha_0 = _S10;
        a_2 = a_1;
    }
    var w_1 : vec3<f32> = inverse_rotate_0(q_6, from_rubble_0(bodies_0[b_0 + u32(4)].xyz));
    var a_local_0 : vec3<f32> = inverse_rotate_0(q_6, a_2) - cross(w_1, cross(w_1, node_0.xyz - frames_0[_S5].xyz));
    var _S12 : u32 = u32(2) * k_0;
    accel_0[_S12] = vec4<f32>(inverse_rotate_0(q_6, alpha_0), 0.0f);
    accel_0[_S12 + u32(1)] = vec4<f32>(a_local_0, 0.0f);
    return;
}

fn topo_total_0( node_count_1 : u32) -> u32
{
    return u32(10) * node_count_1 + u32(3) * ((node_count_1 + u32(1023)) / u32(1024));
}

fn topo_dirty_0( node_count_2 : u32) -> u32
{
    return topo_total_0(node_count_2) + u32(4);
}

@compute
@workgroup_size(64, 1, 1)
fn blast_sync(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var j_0 : u32 = id_1.x;
    var _S13 : bool;
    if(j_0 >= (params_0.bond_count_0))
    {
        _S13 = true;
    }
    else
    {
        _S13 = work_0[params_0.w_alive_0 + j_0] != u32(0);
    }
    if(_S13)
    {
        return;
    }
    if(!((bonds_0[j_0].health_0) > 0.0f))
    {
        return;
    }
    bonds_0[j_0].health_0 = 0.0f;
    var _S14 : u32 = j_0 * u32(2);
    var _S15 : vec4<f32> = vec4<f32>(0.0f);
    x_0[_S14] = _S15;
    x_0[_S14 + u32(1)] = _S15;
    atomicStore(&(topo_0[topo_dirty_0(params_0.node_count_0)]), u32(1));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn blast_breaks(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var j_1 : u32 = id_2.x;
    var _S16 : bool;
    if(j_1 >= (params_0.bond_count_0))
    {
        _S16 = true;
    }
    else
    {
        _S16 = work_0[params_0.w_alive_0 + j_1] == u32(0);
    }
    if(_S16)
    {
        return;
    }
    var bond_0 : BlastBond_std430_0 = bonds_0[j_1];
    if((bonds_0[j_1].health_0) > 0.0f)
    {
        return;
    }
    work_0[params_0.w_alive_0 + j_1] = u32(0);
    var _S17 : u32 = atomicOr(&(atom_0[params_0.a_dirty_0 + work_0[bond_0.n0_0]]), u32(1));
    var _S18 : u32 = atomicAdd(&(atom_0[params_0.a_counters_0]), u32(1));
    var _S19 : u32 = atomicAdd(&(atom_0[params_0.a_counters_0 + u32(1)]), u32(1));
    return;
}

