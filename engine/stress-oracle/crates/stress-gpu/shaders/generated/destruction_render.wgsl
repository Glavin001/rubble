struct ChunkStatic_std430_0
{
    @align(16) center_0 : vec4<f32>,
    @align(16) inertia0_0 : vec4<f32>,
    @align(16) inertia1_0 : vec4<f32>,
    @align(16) inertia2_0 : vec4<f32>,
    @align(16) inv0_0 : vec4<f32>,
    @align(16) inv1_0 : vec4<f32>,
    @align(16) inv2_0 : vec4<f32>,
    @align(16) scale_0 : vec4<f32>,
    @align(16) info_0 : vec4<u32>,
    @align(16) load_range_0 : vec4<u32>,
    @align(16) half_0 : vec4<f32>,
    @align(16) crot0_0 : vec4<f32>,
    @align(16) crot1_0 : vec4<f32>,
    @align(16) crot2_0 : vec4<f32>,
    @align(16) cmat_0 : vec4<f32>,
    @align(16) cinfo_0 : vec4<u32>,
};

@binding(1) @group(0) var<storage, read> chunks_0 : array<ChunkStatic_std430_0>;

struct Island_std430_0
{
    @align(16) range_0 : vec4<u32>,
    @align(16) info_1 : vec4<u32>,
    @align(16) com_0 : vec4<f32>,
    @align(16) inertia0_1 : vec4<f32>,
    @align(16) inertia1_1 : vec4<f32>,
    @align(16) inertia2_1 : vec4<f32>,
    @align(16) inv0_1 : vec4<f32>,
    @align(16) inv1_1 : vec4<f32>,
    @align(16) inv2_1 : vec4<f32>,
    @align(16) wcom_0 : vec4<f32>,
    @align(16) winv0_0 : vec4<f32>,
    @align(16) winv1_0 : vec4<f32>,
    @align(16) winv2_0 : vec4<f32>,
    @align(16) rotation_0 : vec4<f32>,
    @align(16) position_0 : vec4<f32>,
    @align(16) position_err_0 : vec4<f32>,
    @align(16) velocity_0 : vec4<f32>,
    @align(16) velocity_err_0 : vec4<f32>,
    @align(16) angular_velocity_0 : vec4<f32>,
    @align(16) done_0 : u32,
    @align(4) coverage_0 : u32,
    @align(8) wide_groups_0 : u32,
    @align(4) done_pad_0 : u32,
    @align(16) probes_0 : vec4<u32>,
    @align(16) energy_0 : vec4<f32>,
    @align(16) rotation_err_0 : vec4<f32>,
    @align(16) momentum_0 : vec4<f32>,
    @align(16) momentum_err_0 : vec4<f32>,
};

@binding(2) @group(0) var<storage, read> islands_0 : array<Island_std430_0>;

@binding(3) @group(0) var<storage, read> state_0 : array<vec4<f32>>;

@binding(4) @group(0) var<storage, read> index_0 : array<u32>;

struct JointState_std430_0
{
    @align(4) damage_0 : f32,
    @align(4) crush_0 : f32,
    @align(4) kappa_0 : f32,
    @align(4) kappa_c_0 : f32,
    @align(4) ductility_0 : f32,
    @align(4) ductility_c_0 : f32,
    @align(4) life_0 : f32,
    @align(4) plastic_x_0 : f32,
    @align(4) plastic_y_0 : f32,
    @align(4) plastic_t_0 : f32,
    @align(4) rebar_plastic_0 : f32,
    @align(4) rebar_slip0_0 : f32,
    @align(4) rebar_slip1_0 : f32,
    @align(4) rebar_work_0 : f32,
    @align(4) rebar_broken_0 : f32,
    @align(4) strain_rate_0 : f32,
    @align(4) governing_stress_0 : f32,
    @align(4) dissipated_0 : f32,
    @align(4) utilization_0 : f32,
    @align(4) mode_0 : u32,
};

struct BondDyn_std430_0
{
    @align(16) js_0 : JointState_std430_0,
    @align(16) force_lin_0 : vec4<f32>,
    @align(16) force_ang_0 : vec4<f32>,
    @align(16) sums_0 : vec4<f32>,
    @align(16) comps_0 : vec4<f32>,
    @align(16) events_0 : vec4<u32>,
};

@binding(5) @group(0) var<storage, read> bond_dyn_0 : array<BondDyn_std430_0>;

struct Camera_std140_0
{
    @align(16) view_proj_0 : array<vec4<f32>, i32(4)>,
    @align(16) light_0 : vec4<f32>,
    @align(16) ground_0 : vec4<f32>,
};

@binding(0) @group(0) var<uniform> camera_0 : Camera_std140_0;
struct HullVertex_std430_0
{
    @align(16) pos_0 : vec4<f32>,
    @align(16) normal_0 : vec4<f32>,
};

@binding(8) @group(0) var<storage, read> hull_verts_0 : array<HullVertex_std430_0>;

struct HostChunk_std430_0
{
    @align(16) center_1 : vec4<f32>,
    @align(16) axis0_0 : vec4<f32>,
    @align(16) axis1_0 : vec4<f32>,
    @align(16) axis2_0 : vec4<f32>,
};

@binding(7) @group(0) var<storage, read> host_chunks_0 : array<HostChunk_std430_0>;

@binding(9) @group(0) var<storage, read> host_hull_verts_0 : array<HullVertex_std430_0>;

struct Impactor_std430_0
{
    @align(16) position_1 : vec4<f32>,
    @align(16) position_err_1 : vec4<f32>,
    @align(16) velocity_1 : vec4<f32>,
    @align(16) velocity_err_1 : vec4<f32>,
    @align(16) angular_velocity_1 : vec4<f32>,
    @align(16) rotation_1 : vec4<f32>,
    @align(16) inertia0_2 : vec4<f32>,
    @align(16) inertia1_2 : vec4<f32>,
    @align(16) inertia2_2 : vec4<f32>,
    @align(16) inv0_2 : vec4<f32>,
    @align(16) inv1_2 : vec4<f32>,
    @align(16) inv2_2 : vec4<f32>,
    @align(16) shape_0 : vec4<f32>,
    @align(16) half_1 : vec4<f32>,
    @align(16) mat_0 : vec4<f32>,
    @align(16) crush_1 : vec4<f32>,
    @align(16) geom_0 : vec4<f32>,
    @align(16) rotation_err_1 : vec4<f32>,
    @align(16) ledger_0 : vec4<f32>,
    @align(16) cand_0 : vec4<u32>,
    @align(16) momentum_1 : vec4<f32>,
    @align(16) momentum_err_1 : vec4<f32>,
};

@binding(6) @group(0) var<storage, read> impactors_0 : array<Impactor_std430_0>;

fn box_corner_0( vertex_0 : u32,  p_0 : ptr<function, vec3<f32>>,  n_0 : ptr<function, vec3<f32>>)
{
    var face_0 : u32 = vertex_0 / u32(6);
    var corner_0 : u32 = vertex_0 % u32(6);
    var axis_0 : u32 = face_0 / u32(2);
    var sign_0 : f32;
    if((face_0 % u32(2)) == u32(0))
    {
        sign_0 = 1.0f;
    }
    else
    {
        sign_0 = -1.0f;
    }
    const _S1 : vec2<f32> = vec2<f32>(-1.0f, -1.0f);
    const _S2 : vec2<f32> = vec2<f32>(1.0f, 1.0f);
    var quad_0 : array<vec2<f32>, i32(6)> = array<vec2<f32>, i32(6)>( _S1, vec2<f32>(1.0f, -1.0f), _S2, _S1, _S2, vec2<f32>(-1.0f, 1.0f) );
    var st_0 : vec2<f32> = quad_0[corner_0];
    if(sign_0 < 0.0f)
    {
        st_0[i32(0)] = - st_0.x;
    }
    if(axis_0 == u32(0))
    {
        (*n_0) = vec3<f32>(sign_0, 0.0f, 0.0f);
        (*p_0) = vec3<f32>(sign_0, st_0.x, st_0.y);
    }
    else
    {
        if(axis_0 == u32(1))
        {
            (*n_0) = vec3<f32>(0.0f, sign_0, 0.0f);
            (*p_0) = vec3<f32>(st_0.y, sign_0, st_0.x);
        }
        else
        {
            (*n_0) = vec3<f32>(0.0f, 0.0f, sign_0);
            (*p_0) = vec3<f32>(st_0.x, st_0.y, sign_0);
        }
    }
    return;
}

struct Quat_0
{
     w_0 : f32,
     x_0 : f32,
     y_0 : f32,
     z_0 : f32,
};

fn quat_of_0( q_0 : vec4<f32>) -> Quat_0
{
    var r_0 : Quat_0;
    r_0.x_0 = q_0.x;
    r_0.y_0 = q_0.y;
    r_0.z_0 = q_0.z;
    r_0.w_0 = q_0.w;
    return r_0;
}

fn normalized_0( q_1 : Quat_0) -> Quat_0
{
    var _S3 : f32 = q_1.w_0;
    var _S4 : f32 = q_1.x_0;
    var _S5 : f32 = q_1.y_0;
    var _S6 : f32 = q_1.z_0;
    var n_1 : f32 = sqrt(_S3 * _S3 + _S4 * _S4 + _S5 * _S5 + _S6 * _S6);
    var r_1 : Quat_0;
    r_1.w_0 = q_1.w_0 / n_1;
    r_1.x_0 = q_1.x_0 / n_1;
    r_1.y_0 = q_1.y_0 / n_1;
    r_1.z_0 = q_1.z_0 / n_1;
    return r_1;
}

fn rotation_of_0( hi_0 : vec4<f32>,  lo_0 : vec4<f32>) -> Quat_0
{
    return normalized_0(quat_of_0(hi_0 + lo_0));
}

fn safe_normalize_0( v_0 : vec3<f32>) -> vec3<f32>
{
    var n_2 : f32 = length(v_0);
    var _S7 : vec3<f32>;
    if(n_2 > 1.00000000317107685e-30f)
    {
        _S7 = v_0 / vec3<f32>(n_2);
    }
    else
    {
        _S7 = vec3<f32>(0.0f);
    }
    return _S7;
}

fn from_axis_angle_0( axis_1 : vec3<f32>,  angle_0 : f32) -> Quat_0
{
    var a_0 : vec3<f32> = safe_normalize_0(axis_1);
    var _S8 : f32 = 0.5f * angle_0;
    var s_0 : f32 = sin(_S8);
    var q_2 : Quat_0;
    q_2.w_0 = cos(_S8);
    q_2.x_0 = a_0.x * s_0;
    q_2.y_0 = a_0.y * s_0;
    q_2.z_0 = a_0.z * s_0;
    return q_2;
}

fn rotate_0( q_3 : Quat_0,  v_1 : vec3<f32>) -> vec3<f32>
{
    var qv_0 : vec3<f32> = vec3<f32>(q_3.x_0, q_3.y_0, q_3.z_0);
    var t_0 : vec3<f32> = cross(qv_0, v_1) * vec3<f32>(2.0f);
    return v_1 + t_0 * vec3<f32>(q_3.w_0) + cross(qv_0, t_0);
}

fn clip_of_0( world_0 : vec3<f32>) -> vec4<f32>
{
    return camera_0.view_proj_0[i32(0)] * vec4<f32>(world_0.x) + camera_0.view_proj_0[i32(1)] * vec4<f32>(world_0.y) + camera_0.view_proj_0[i32(2)] * vec4<f32>(world_0.z) + camera_0.view_proj_0[i32(3)];
}

fn ramp_0( t_1 : f32) -> vec3<f32>
{
    const c0_0 : vec3<f32> = vec3<f32>(0.15000000596046448f, 0.25f, 0.85000002384185791f);
    const c1_0 : vec3<f32> = vec3<f32>(0.10000000149011612f, 0.75f, 0.85000002384185791f);
    const c2_0 : vec3<f32> = vec3<f32>(0.25f, 0.80000001192092896f, 0.30000001192092896f);
    const c3_0 : vec3<f32> = vec3<f32>(0.94999998807907104f, 0.85000002384185791f, 0.20000000298023224f);
    const c4_0 : vec3<f32> = vec3<f32>(0.89999997615814209f, 0.20000000298023224f, 0.15000000596046448f);
    var s_1 : f32 = saturate(t_1) * 4.0f;
    if(s_1 < 1.0f)
    {
        return mix(c0_0, c1_0, vec3<f32>(s_1));
    }
    if(s_1 < 2.0f)
    {
        return mix(c1_0, c2_0, vec3<f32>((s_1 - 1.0f)));
    }
    if(s_1 < 3.0f)
    {
        return mix(c2_0, c3_0, vec3<f32>((s_1 - 2.0f)));
    }
    return mix(c3_0, c4_0, vec3<f32>((s_1 - 3.0f)));
}

struct VertexOut_0
{
    @builtin(position) position_2 : vec4<f32>,
    @location(0) normal_1 : vec3<f32>,
    @location(1) color_0 : vec3<f32>,
    @location(2) world_1 : vec3<f32>,
};

fn chunk_vertex_0( c_0 : u32,  p_1 : vec3<f32>,  n_3 : vec3<f32>) -> VertexOut_0
{
    var q_4 : Quat_0 = rotation_of_0(islands_0[chunks_0[c_0].info_0.y].rotation_0, islands_0[chunks_0[c_0].info_0.y].rotation_err_0);
    var _S9 : u32 = u32(4) * c_0;
    var th_0 : vec3<f32> = state_0[_S9 + u32(1)].xyz;
    var hidden_0 : Quat_0 = from_axis_angle_0(th_0, length(th_0));
    var _S10 : vec3<f32> = chunks_0[c_0].crot0_0.xyz;
    var _S11 : vec3<f32> = chunks_0[c_0].crot1_0.xyz;
    var _S12 : vec3<f32> = chunks_0[c_0].crot2_0.xyz;
    var local_n_0 : vec3<f32> = vec3<f32>(dot(_S10, n_3), dot(_S11, n_3), dot(_S12, n_3));
    var world_2 : vec3<f32> = islands_0[chunks_0[c_0].info_0.y].position_0.xyz + islands_0[chunks_0[c_0].info_0.y].position_err_0.xyz + rotate_0(q_4, chunks_0[c_0].center_0.xyz + state_0[_S9].xyz + rotate_0(hidden_0, vec3<f32>(dot(_S10, p_1), dot(_S11, p_1), dot(_S12, p_1))));
    var _S13 : u32 = index_0[c_0];
    var util_0 : f32 = 0.0f;
    var e_0 : u32 = _S13;
    var intact_0 : u32 = u32(0);
    loop
    {
        if(e_0 < index_0[c_0 + u32(1)])
        {
        }
        else
        {
            break;
        }
        if((bond_dyn_0[(index_0[e_0] >> (u32(1)))].events_0.z) != u32(0))
        {
            e_0 = e_0 + u32(1);
            continue;
        }
        var _S14 : u32 = intact_0 + u32(1);
        util_0 = max(util_0, bond_dyn_0[(index_0[e_0] >> (u32(1)))].js_0.utilization_0);
        intact_0 = _S14;
        e_0 = e_0 + u32(1);
    }
    var o_0 : VertexOut_0;
    o_0.position_2 = clip_of_0(world_2);
    o_0.normal_1 = rotate_0(q_4, rotate_0(hidden_0, local_n_0));
    var _S15 : vec3<f32>;
    if(intact_0 > u32(0))
    {
        _S15 = ramp_0(util_0);
    }
    else
    {
        _S15 = vec3<f32>(0.55000001192092896f, 0.51999998092651367f, 0.47999998927116394f);
    }
    o_0.color_0 = _S15;
    o_0.world_1 = world_2;
    return o_0;
}

@vertex
fn chunk_vs(@builtin(vertex_index) vertex_1 : u32, @builtin(instance_index) instance_0 : u32) -> VertexOut_0
{
    var p_2 : vec3<f32>;
    var n_4 : vec3<f32>;
    box_corner_0(vertex_1, &(p_2), &(n_4));
    return chunk_vertex_0(instance_0, p_2 * chunks_0[instance_0].half_0.xyz, n_4);
}

@vertex
fn chunk_hull_vs(@builtin(vertex_index) vertex_2 : u32) -> VertexOut_0
{
    var v_2 : HullVertex_std430_0 = hull_verts_0[vertex_2];
    return chunk_vertex_0(u32(v_2.pos_0.w), v_2.pos_0.xyz, v_2.normal_0.xyz);
}

fn host_vertex_0( hc_0 : ptr<function, HostChunk_std430_0>,  p_3 : vec3<f32>,  n_5 : vec3<f32>) -> VertexOut_0
{
    var _S16 : vec3<f32> = (*hc_0).axis0_0.xyz;
    var _S17 : vec3<f32> = (*hc_0).axis1_0.xyz;
    var _S18 : vec3<f32> = (*hc_0).axis2_0.xyz;
    var world_3 : vec3<f32> = (*hc_0).center_1.xyz + _S16 * vec3<f32>(p_3.x) + _S17 * vec3<f32>(p_3.y) + _S18 * vec3<f32>(p_3.z);
    var o_1 : VertexOut_0;
    o_1.position_2 = clip_of_0(world_3);
    o_1.normal_1 = _S16 * vec3<f32>(n_5.x) + _S17 * vec3<f32>(n_5.y) + _S18 * vec3<f32>(n_5.z);
    var _S19 : f32 = (*hc_0).center_1.w;
    var _S20 : vec3<f32>;
    if(_S19 >= 0.0f)
    {
        _S20 = ramp_0(_S19);
    }
    else
    {
        _S20 = vec3<f32>(0.55000001192092896f, 0.51999998092651367f, 0.47999998927116394f);
    }
    o_1.color_0 = _S20;
    o_1.world_1 = world_3;
    return o_1;
}

@vertex
fn host_chunk_vs(@builtin(vertex_index) vertex_3 : u32, @builtin(instance_index) instance_1 : u32) -> VertexOut_0
{
    var p_4 : vec3<f32>;
    var n_6 : vec3<f32>;
    box_corner_0(vertex_3, &(p_4), &(n_6));
    var _S21 : HostChunk_std430_0 = host_chunks_0[instance_1];
    var _S22 : VertexOut_0 = host_vertex_0(&(_S21), p_4 * vec3<f32>(_S21.axis0_0.w, _S21.axis1_0.w, _S21.axis2_0.w), n_6);
    return _S22;
}

@vertex
fn host_hull_vs(@builtin(vertex_index) vertex_4 : u32) -> VertexOut_0
{
    var v_3 : HullVertex_std430_0 = host_hull_verts_0[vertex_4];
    var _S23 : HostChunk_std430_0 = host_chunks_0[u32(v_3.pos_0.w)];
    var _S24 : VertexOut_0 = host_vertex_0(&(_S23), v_3.pos_0.xyz, v_3.normal_0.xyz);
    return _S24;
}

@vertex
fn impactor_vs(@builtin(vertex_index) vertex_5 : u32, @builtin(instance_index) instance_2 : u32) -> VertexOut_0
{
    var q_5 : Quat_0 = rotation_of_0(impactors_0[instance_2].rotation_1, impactors_0[instance_2].rotation_err_1);
    var center_2 : vec3<f32> = impactors_0[instance_2].position_1.xyz + impactors_0[instance_2].position_err_1.xyz;
    var p_5 : vec3<f32>;
    var n_7 : vec3<f32>;
    var o_2 : VertexOut_0;
    var _S25 : vec4<f32> = impactors_0[instance_2].shape_0;
    if((impactors_0[instance_2].shape_0.x) > 0.5f)
    {
        if(vertex_5 >= u32(36))
        {
            o_2.position_2 = vec4<f32>(0.0f, 0.0f, 2.0f, 1.0f);
            o_2.normal_1 = vec3<f32>(0.0f, 0.0f, 1.0f);
            var _S26 : vec3<f32> = vec3<f32>(0.0f);
            o_2.color_0 = _S26;
            o_2.world_1 = _S26;
            return o_2;
        }
        box_corner_0(vertex_5, &(p_5), &(n_7));
        p_5 = p_5 * impactors_0[instance_2].half_1.xyz;
    }
    else
    {
        var quad_1 : u32 = vertex_5 / u32(6);
        var corner_1 : u32 = vertex_5 % u32(6);
        const _S27 : vec2<u32> = vec2<u32>(u32(0), u32(0));
        const _S28 : vec2<u32> = vec2<u32>(u32(1), u32(1));
        var offs_0 : array<vec2<u32>, i32(6)> = array<vec2<u32>, i32(6)>( _S27, vec2<u32>(u32(1), u32(0)), _S28, _S27, _S28, vec2<u32>(u32(0), u32(1)) );
        var theta_0 : f32 = 3.14159274101257324f * f32(quad_1 / u32(16) + offs_0[corner_1].y) / 10.0f;
        var phi_0 : f32 = 6.28318548202514648f * f32(quad_1 % u32(16) + offs_0[corner_1].x) / 16.0f;
        var _S29 : f32 = sin(theta_0);
        var _S30 : vec3<f32> = vec3<f32>(_S29 * cos(phi_0), _S29 * sin(phi_0), cos(theta_0));
        n_7 = _S30;
        p_5 = _S30 * vec3<f32>(_S25.y);
    }
    var world_4 : vec3<f32> = center_2 + rotate_0(q_5, p_5);
    o_2.position_2 = clip_of_0(world_4);
    o_2.normal_1 = rotate_0(q_5, n_7);
    o_2.color_0 = vec3<f32>(0.77999997138977051f, 0.80000001192092896f, 0.8399999737739563f);
    o_2.world_1 = world_4;
    return o_2;
}

@vertex
fn ground_vs(@builtin(vertex_index) vertex_6 : u32) -> VertexOut_0
{
    const _S31 : vec2<f32> = vec2<f32>(-1.0f, -1.0f);
    const _S32 : vec2<f32> = vec2<f32>(1.0f, 1.0f);
    var quad_2 : array<vec2<f32>, i32(6)> = array<vec2<f32>, i32(6)>( _S31, vec2<f32>(1.0f, -1.0f), _S32, _S31, _S32, vec2<f32>(-1.0f, 1.0f) );
    var world_5 : vec3<f32> = vec3<f32>(camera_0.ground_0.zw + quad_2[vertex_6] * vec2<f32>(camera_0.ground_0.y), camera_0.ground_0.x);
    var o_3 : VertexOut_0;
    o_3.position_2 = clip_of_0(world_5);
    o_3.normal_1 = vec3<f32>(0.0f, 0.0f, 1.0f);
    o_3.color_0 = vec3<f32>(0.0f);
    o_3.world_1 = world_5;
    return o_3;
}

struct pixelOutput_0
{
    @location(0) output_0 : vec4<f32>,
};

struct pixelInput_0
{
    @location(0) normal_2 : vec3<f32>,
    @location(1) color_1 : vec3<f32>,
    @location(2) world_6 : vec3<f32>,
};

@fragment
fn lit_fs( _S33 : pixelInput_0, @builtin(position) position_3 : vec4<f32>) -> pixelOutput_0
{
    var _S34 : pixelOutput_0 = pixelOutput_0( vec4<f32>(_S33.color_1 * vec3<f32>((0.34999999403953552f + 0.64999997615814209f * max(dot(normalize(_S33.normal_2), normalize(camera_0.light_0.xyz)), 0.0f))), 1.0f) );
    return _S34;
}

struct pixelOutput_1
{
    @location(0) output_1 : vec4<f32>,
};

struct pixelInput_1
{
    @location(0) normal_3 : vec3<f32>,
    @location(1) color_2 : vec3<f32>,
    @location(2) world_7 : vec3<f32>,
};

@fragment
fn ground_fs( _S35 : pixelInput_1, @builtin(position) position_4 : vec4<f32>) -> pixelOutput_1
{
    var _S36 : vec2<f32> = _S35.world_7.xy;
    var _S37 : vec2<i32> = vec2<i32>(floor(_S36));
    var checker_0 : f32;
    if((((_S37.x + _S37.y) & (i32(1)))) == i32(0))
    {
        checker_0 = 1.0f;
    }
    else
    {
        checker_0 = 0.0f;
    }
    var edge_0 : f32 = saturate(length(_S36 - camera_0.ground_0.zw) / camera_0.ground_0.y);
    var _S38 : pixelOutput_1 = pixelOutput_1( vec4<f32>(mix(mix(vec3<f32>(0.15999999642372131f, 0.17000000178813934f, 0.18999999761581421f), vec3<f32>(0.20000000298023224f, 0.20999999344348907f, 0.23000000417232513f), vec3<f32>(checker_0)), vec3<f32>(0.05000000074505806f, 0.05999999865889549f, 0.07999999821186066f), vec3<f32>((edge_0 * edge_0))), 1.0f) );
    return _S38;
}

