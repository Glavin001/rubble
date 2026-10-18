struct PunchParams_std140_0
{
    @align(16) island_count_0 : u32,
    @align(4) impactor_count_0 : u32,
    @align(8) engine_base_0 : u32,
    @align(4) pad0_0 : u32,
    @align(16) dt_0 : f32,
    @align(4) speed_0 : f32,
    @align(8) pad1_0 : f32,
    @align(4) pad2_0 : f32,
};

@binding(0) @group(0) var<uniform> params_0 : PunchParams_std140_0;
struct Island_std430_0
{
    @align(16) range_0 : vec4<u32>,
    @align(16) info_0 : vec4<u32>,
    @align(16) com_0 : vec4<f32>,
    @align(16) inertia0_0 : vec4<f32>,
    @align(16) inertia1_0 : vec4<f32>,
    @align(16) inertia2_0 : vec4<f32>,
    @align(16) inv0_0 : vec4<f32>,
    @align(16) inv1_0 : vec4<f32>,
    @align(16) inv2_0 : vec4<f32>,
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

@binding(1) @group(0) var<storage, read> islands_0 : array<Island_std430_0>;

@binding(10) @group(0) var<storage, read_write> status_0 : array<u32>;

@binding(4) @group(0) var<storage, read> ctab_0 : array<vec4<f32>>;

struct EngineBody_std430_0
{
    @align(16) pos_inv_mass_0 : vec4<f32>,
    @align(16) rot_0 : vec4<f32>,
    @align(16) inv_inertia_flags_0 : vec4<f32>,
    @align(16) vel_sleep_0 : vec4<f32>,
    @align(16) ang_vel_gen_0 : vec4<f32>,
    @align(16) initial_pos_0 : vec4<f32>,
    @align(16) initial_rot_0 : vec4<f32>,
};

@binding(5) @group(0) var<storage, read> bodies_0 : array<EngineBody_std430_0>;

struct ChunkStatic_std430_0
{
    @align(16) center_0 : vec4<f32>,
    @align(16) inertia0_1 : vec4<f32>,
    @align(16) inertia1_1 : vec4<f32>,
    @align(16) inertia2_1 : vec4<f32>,
    @align(16) inv0_1 : vec4<f32>,
    @align(16) inv1_1 : vec4<f32>,
    @align(16) inv2_1 : vec4<f32>,
    @align(16) scale_0 : vec4<f32>,
    @align(16) info_1 : vec4<u32>,
    @align(16) load_range_0 : vec4<u32>,
    @align(16) half_0 : vec4<f32>,
    @align(16) crot0_0 : vec4<f32>,
    @align(16) crot1_0 : vec4<f32>,
    @align(16) crot2_0 : vec4<f32>,
    @align(16) cmat_0 : vec4<f32>,
    @align(16) cinfo_0 : vec4<u32>,
};

@binding(2) @group(0) var<storage, read> chunks_0 : array<ChunkStatic_std430_0>;

@binding(3) @group(0) var<storage, read> state_0 : array<vec4<f32>>;

@binding(6) @group(0) var<storage, read> impactors_0 : array<vec4<f32>>;

struct JointBond_std430_0
{
    @align(16) geom0_0 : vec4<f32>,
    @align(16) geom1_0 : vec4<f32>,
    @align(16) stiff0_0 : vec4<f32>,
    @align(16) stiff1_0 : vec4<f32>,
    @align(16) rebar0_0 : vec4<f32>,
    @align(16) rebar1_0 : vec4<f32>,
    @align(16) ids_0 : vec4<u32>,
};

struct BondStatic_std430_0
{
    @align(16) t1_0 : vec4<f32>,
    @align(16) t2_0 : vec4<f32>,
    @align(16) normal_0 : vec4<f32>,
    @align(16) ra_0 : vec4<f32>,
    @align(16) rb_0 : vec4<f32>,
    @align(16) centroid_0 : vec4<f32>,
    @align(16) c_lin_0 : vec4<f32>,
    @align(16) c_ang_0 : vec4<f32>,
    @align(16) law_0 : JointBond_std430_0,
};

@binding(7) @group(0) var<storage, read> bond_static_0 : array<BondStatic_std430_0>;

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

@binding(8) @group(0) var<storage, read_write> bond_dyn_0 : array<BondDyn_std430_0>;

struct JointMaterial_std140_0
{
    @align(16) strength_0 : vec4<f32>,
    @align(16) energy_1 : vec4<f32>,
    @align(16) dif_0 : vec4<f32>,
    @align(16) misc_0 : vec4<f32>,
    @align(16) kind_flags_0 : vec4<u32>,
};

struct _Array_std140_JointMaterial64_0
{
    @align(16) data_0 : array<JointMaterial_std140_0, i32(64)>,
};

struct MaterialTable_std140_0
{
    @align(16) m_0 : _Array_std140_JointMaterial64_0,
};

@binding(9) @group(0) var<uniform> materials_0 : MaterialTable_std140_0;
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

fn quat_mul_0( a_0 : Quat_0,  o_0 : Quat_0) -> Quat_0
{
    var r_1 : Quat_0;
    r_1.w_0 = a_0.w_0 * o_0.w_0 - a_0.x_0 * o_0.x_0 - a_0.y_0 * o_0.y_0 - a_0.z_0 * o_0.z_0;
    r_1.x_0 = a_0.w_0 * o_0.x_0 + a_0.x_0 * o_0.w_0 + a_0.y_0 * o_0.z_0 - a_0.z_0 * o_0.y_0;
    r_1.y_0 = a_0.w_0 * o_0.y_0 - a_0.x_0 * o_0.z_0 + a_0.y_0 * o_0.w_0 + a_0.z_0 * o_0.x_0;
    r_1.z_0 = a_0.w_0 * o_0.z_0 + a_0.x_0 * o_0.y_0 - a_0.y_0 * o_0.x_0 + a_0.z_0 * o_0.w_0;
    return r_1;
}

fn normalized_0( q_1 : Quat_0) -> Quat_0
{
    var _S1 : f32 = q_1.w_0;
    var _S2 : f32 = q_1.x_0;
    var _S3 : f32 = q_1.y_0;
    var _S4 : f32 = q_1.z_0;
    var n_0 : f32 = sqrt(_S1 * _S1 + _S2 * _S2 + _S3 * _S3 + _S4 * _S4);
    var r_2 : Quat_0;
    r_2.w_0 = q_1.w_0 / n_0;
    r_2.x_0 = q_1.x_0 / n_0;
    r_2.y_0 = q_1.y_0 / n_0;
    r_2.z_0 = q_1.z_0 / n_0;
    return r_2;
}

fn to_zup_0( v_0 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(v_0.x, - v_0.z, v_0.y);
}

fn rotate_0( q_2 : Quat_0,  v_1 : vec3<f32>) -> vec3<f32>
{
    var qv_0 : vec3<f32> = vec3<f32>(q_2.x_0, q_2.y_0, q_2.z_0);
    var t_0 : vec3<f32> = cross(qv_0, v_1) * vec3<f32>(2.0f);
    return v_1 + t_0 * vec3<f32>(q_2.w_0) + cross(qv_0, t_0);
}

var<workgroup> g_lo_0 : array<vec3<f32>, i32(256)>;

var<workgroup> g_hi_0 : array<vec3<f32>, i32(256)>;

var<workgroup> g_a_0 : array<f32, i32(256)>;

var<workgroup> g_idx_0 : array<u32, i32(256)>;

var<workgroup> g_plug_0 : array<u32, i32(4096)>;

var<workgroup> g_changed_0 : u32;

fn connected_0( st_0 : ptr<function, JointState_std430_0>,  has_rebar_0 : bool) -> bool
{
    var _S5 : bool;
    if(((*st_0).damage_0) < 1.0f)
    {
        _S5 = true;
    }
    else
    {
        if(has_rebar_0)
        {
            _S5 = ((*st_0).rebar_broken_0) == 0.0f;
        }
        else
        {
            _S5 = false;
        }
    }
    return _S5;
}

struct JointState_0
{
     damage_0 : f32,
     crush_0 : f32,
     kappa_0 : f32,
     kappa_c_0 : f32,
     ductility_0 : f32,
     ductility_c_0 : f32,
     life_0 : f32,
     plastic_x_0 : f32,
     plastic_y_0 : f32,
     plastic_t_0 : f32,
     rebar_plastic_0 : f32,
     rebar_slip0_0 : f32,
     rebar_slip1_0 : f32,
     rebar_work_0 : f32,
     rebar_broken_0 : f32,
     strain_rate_0 : f32,
     governing_stress_0 : f32,
     dissipated_0 : f32,
     utilization_0 : f32,
     mode_0 : u32,
};

fn connected_1( st_1 : JointState_0,  has_rebar_1 : bool) -> bool
{
    var _S6 : bool;
    if((st_1.damage_0) < 1.0f)
    {
        _S6 = true;
    }
    else
    {
        if(has_rebar_1)
        {
            _S6 = (st_1.rebar_broken_0) == 0.0f;
        }
        else
        {
            _S6 = false;
        }
    }
    return _S6;
}

var<workgroup> g_b_0 : array<f32, i32(256)>;

var<workgroup> g_go_0 : u32;

struct BondDyn_0
{
     js_0 : JointState_0,
     force_lin_0 : vec4<f32>,
     force_ang_0 : vec4<f32>,
     sums_0 : vec4<f32>,
     comps_0 : vec4<f32>,
     events_0 : vec4<u32>,
};

@compute
@workgroup_size(256, 1, 1)
fn punch_islands(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var ax_0 : u32;
    var hit_k_0 : u32;
    var t0_0 : f32;
    var t1_1 : f32;
    var t0_1 : f32;
    var bi_0 : u32;
    var i_0 : u32 = group_0.x;
    var t_1 : u32 = thread_0.x;
    if(i_0 >= (params_0.island_count_0))
    {
        return;
    }
    var _S7 : u32 = islands_0[i_0].range_0.x;
    var n_1 : u32 = islands_0[i_0].range_0.y - _S7;
    var _S8 : u32 = islands_0[i_0].range_0.z;
    var _S9 : u32 = islands_0[i_0].range_0.w;
    var out_0 : bool;
    if(_S8 >= _S9)
    {
        out_0 = true;
    }
    else
    {
        out_0 = n_1 > u32(4096);
    }
    if(out_0)
    {
        out_0 = true;
    }
    else
    {
        out_0 = (params_0.impactor_count_0) == u32(0);
    }
    if(out_0)
    {
        if(t_1 == u32(0))
        {
            status_0[u32(16) + i_0] = (status_0[u32(16) + i_0] & (u32(4294967287)));
        }
        return;
    }
    var _S10 : u32 = u32(3) * i_0;
    var body_0 : u32 = (bitcast<u32>((ctab_0[params_0.engine_base_0 + _S10].x)));
    var _S11 : Quat_0 = quat_of_0(islands_0[i_0].rotation_0);
    var _S12 : vec3<f32> = islands_0[i_0].position_0.xyz;
    var _S13 : vec3<f32> = islands_0[i_0].velocity_0.xyz;
    var origin_0 : vec3<f32>;
    var v_2 : vec3<f32>;
    var q_3 : Quat_0;
    if(body_0 != u32(4294967295))
    {
        var p_inv_0 : Quat_0 = quat_of_0(ctab_0[params_0.engine_base_0 + _S10 + u32(1)]);
        var com_1 : vec3<f32> = ctab_0[params_0.engine_base_0 + _S10 + u32(2)].xyz;
        var b_0 : EngineBody_std430_0 = bodies_0[body_0];
        var a_inv_0 : Quat_0;
        a_inv_0.w_0 = 0.70710676908493042f;
        a_inv_0.x_0 = 0.70710676908493042f;
        a_inv_0.y_0 = 0.0f;
        a_inv_0.z_0 = 0.0f;
        var q_4 : Quat_0 = normalized_0(quat_mul_0(quat_mul_0(a_inv_0, quat_of_0(b_0.rot_0)), p_inv_0));
        var _S14 : vec3<f32> = to_zup_0(b_0.vel_sleep_0.xyz);
        origin_0 = to_zup_0(b_0.pos_inv_mass_0.xyz) - rotate_0(q_4, com_1);
        q_3 = q_4;
        v_2 = _S14;
    }
    else
    {
        origin_0 = _S12;
        q_3 = _S11;
        v_2 = _S13;
    }
    var inf_0 : f32 = (bitcast<f32>((u32(2139095040))));
    var _S15 : vec3<f32> = vec3<f32>(- inf_0);
    var lo_0 : vec3<f32> = vec3<f32>(inf_0);
    var hi_0 : vec3<f32> = _S15;
    var c_0 : u32 = t_1;
    loop
    {
        if(c_0 < n_1)
        {
        }
        else
        {
            break;
        }
        var k_0 : u32 = _S7 + c_0;
        var p_0 : vec3<f32> = origin_0 + rotate_0(q_3, chunks_0[k_0].center_0.xyz + state_0[u32(4) * k_0].xyz);
        var _S16 : vec3<f32> = min(lo_0, p_0 - vec3<f32>(chunks_0[k_0].half_0.w));
        var _S17 : vec3<f32> = max(hi_0, p_0 + vec3<f32>(chunks_0[k_0].half_0.w));
        var c_1 : u32 = c_0 + u32(256);
        lo_0 = _S16;
        hi_0 = _S17;
        c_0 = c_1;
    }
    g_lo_0[t_1] = lo_0;
    g_hi_0[t_1] = hi_0;
    workgroupBarrier();
    var s_0 : u32 = u32(128);
    loop
    {
        if(s_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(t_1 < s_0)
        {
            var _S18 : u32 = t_1 + s_0;
            g_lo_0[t_1] = min(g_lo_0[t_1], g_lo_0[_S18]);
            g_hi_0[t_1] = max(g_hi_0[t_1], g_hi_0[_S18]);
        }
        workgroupBarrier();
        s_0 = (s_0 >> (u32(1)));
    }
    var _S19 : vec3<f32> = g_lo_0[i32(0)];
    var _S20 : vec3<f32> = g_hi_0[i32(0)];
    var _S21 : vec3<f32> = vec3<f32>(0.0f);
    var hit_k_1 : u32 = u32(4294967295);
    var hit_0 : vec3<f32> = _S21;
    var d_0 : vec3<f32> = _S21;
    var speed_1 : f32 = 0.0f;
    var k_1 : u32 = u32(0);
    loop
    {
        if(k_1 < (params_0.impactor_count_0))
        {
            out_0 = hit_k_1 == u32(4294967295);
        }
        else
        {
            out_0 = false;
        }
        if(out_0)
        {
        }
        else
        {
            break;
        }
        var a_1 : vec4<f32> = impactors_0[u32(2) * k_1];
        var b_1 : EngineBody_std430_0 = bodies_0[(bitcast<u32>((a_1.x)))];
        var c_2 : vec3<f32> = to_zup_0(b_1.pos_inv_mass_0.xyz);
        var rel_0 : vec3<f32> = to_zup_0(b_1.vel_sleep_0.xyz) - v_2;
        var sp_0 : f32 = length(rel_0);
        if(!(sp_0 > (params_0.speed_0)))
        {
            k_1 = k_1 + u32(1);
            continue;
        }
        var _S22 : vec3<f32> = vec3<f32>(a_1.y);
        var _S23 : vec3<f32> = _S19 - _S22;
        var _S24 : vec3<f32> = _S20 + _S22;
        t0_0 = 0.0f;
        t1_1 = params_0.dt_0;
        ax_0 = u32(0);
        loop
        {
            if(ax_0 < u32(3))
            {
            }
            else
            {
                break;
            }
            var _S25 : u32 = ax_0;
            if((abs(rel_0[ax_0])) < 9.999999960041972e-13f)
            {
                var _S26 : u32 = ax_0;
                var _S27 : bool;
                if((c_2[ax_0]) < (_S23[ax_0]))
                {
                    _S27 = true;
                }
                else
                {
                    _S27 = (c_2[_S26]) > (_S24[ax_0]);
                }
                if(_S27)
                {
                    t0_1 = inf_0;
                }
                else
                {
                    t0_1 = t0_0;
                }
                t0_0 = t0_1;
                ax_0 = ax_0 + u32(1);
                continue;
            }
            var ta_0 : f32 = (_S23[ax_0] - c_2[ax_0]) / rel_0[_S25];
            var tb_0 : f32 = (_S24[ax_0] - c_2[ax_0]) / rel_0[_S25];
            var _S28 : f32 = min(t1_1, max(ta_0, tb_0));
            t0_0 = max(t0_0, min(ta_0, tb_0));
            t1_1 = _S28;
            ax_0 = ax_0 + u32(1);
        }
        var hit_1 : vec3<f32>;
        var d_1 : vec3<f32>;
        if(t0_0 <= t1_1)
        {
            var _S29 : vec3<f32> = c_2 + rel_0 * vec3<f32>(t0_0);
            var _S30 : vec3<f32> = rel_0 / vec3<f32>(sp_0);
            hit_k_0 = k_1;
            hit_1 = _S29;
            d_1 = _S30;
            t0_1 = sp_0;
        }
        else
        {
            hit_k_0 = hit_k_1;
            hit_1 = hit_0;
            d_1 = d_0;
            t0_1 = speed_1;
        }
        hit_k_1 = hit_k_0;
        hit_0 = hit_1;
        d_0 = d_1;
        speed_1 = t0_1;
        k_1 = k_1 + u32(1);
    }
    if(hit_k_1 == u32(4294967295))
    {
        if(t_1 == u32(0))
        {
            status_0[u32(16) + i_0] = (status_0[u32(16) + i_0] & (u32(4294967287)));
        }
        return;
    }
    var _S31 : u32 = u32(2) * hit_k_1;
    var imp_a_0 : vec4<f32> = impactors_0[_S31];
    var imp_b_0 : vec4<f32> = impactors_0[_S31 + u32(1)];
    var face_0 : f32 = imp_a_0.z;
    t0_0 = inf_0;
    ax_0 = u32(4294967295);
    c_0 = t_1;
    loop
    {
        if(c_0 < n_1)
        {
        }
        else
        {
            break;
        }
        var k_2 : u32 = _S7 + c_0;
        var r_3 : vec3<f32> = origin_0 + rotate_0(q_3, chunks_0[k_2].center_0.xyz + state_0[u32(4) * k_2].xyz) - hit_0;
        var along_0 : f32 = dot(r_3, d_0);
        var dist_0 : f32 = length(r_3 - d_0 * vec3<f32>(along_0)) + abs(along_0) * 0.00100000004749745f;
        if(dist_0 < t0_0)
        {
            t0_0 = dist_0;
            ax_0 = c_0;
        }
        c_0 = c_0 + u32(256);
    }
    g_a_0[t_1] = t0_0;
    g_idx_0[t_1] = ax_0;
    workgroupBarrier();
    s_0 = u32(128);
    loop
    {
        if(s_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(t_1 < s_0)
        {
            var _S32 : u32 = t_1 + s_0;
            if((g_a_0[_S32]) < (g_a_0[t_1]))
            {
                out_0 = true;
            }
            else
            {
                if((g_a_0[_S32]) == (g_a_0[t_1]))
                {
                    out_0 = (g_idx_0[_S32]) < (g_idx_0[t_1]);
                }
                else
                {
                    out_0 = false;
                }
            }
        }
        else
        {
            out_0 = false;
        }
        if(out_0)
        {
            var _S33 : u32 = t_1 + s_0;
            g_a_0[t_1] = g_a_0[_S33];
            g_idx_0[t_1] = g_idx_0[_S33];
        }
        workgroupBarrier();
        s_0 = (s_0 >> (u32(1)));
    }
    var entry_0 : u32 = g_idx_0[i32(0)];
    var entry_k_0 : u32 = _S7 + g_idx_0[i32(0)];
    var _S34 : f32 = dot(origin_0 + rotate_0(q_3, chunks_0[entry_k_0].center_0.xyz + state_0[u32(4) * entry_k_0].xyz) - hit_0, d_0);
    c_0 = t_1;
    loop
    {
        if(c_0 < n_1)
        {
        }
        else
        {
            break;
        }
        if(c_0 == entry_0)
        {
            s_0 = u32(1);
        }
        else
        {
            s_0 = u32(0);
        }
        g_plug_0[c_0] = s_0;
        c_0 = c_0 + u32(256);
    }
    workgroupBarrier();
    hit_k_0 = u32(0);
    loop
    {
        if(hit_k_0 < n_1)
        {
        }
        else
        {
            break;
        }
        if(t_1 == u32(0))
        {
            g_changed_0 = u32(0);
        }
        workgroupBarrier();
        bi_0 = _S8 + t_1;
        loop
        {
            if(bi_0 < _S9)
            {
            }
            else
            {
                break;
            }
            var law_1 : JointBond_std430_0 = bond_static_0[bi_0].law_0;
            var _S35 : JointState_std430_0 = bond_dyn_0[bi_0].js_0;
            var _S36 : bool = connected_0(&(_S35), (bond_static_0[bi_0].law_0.stiff1_0.w) != 0.0f);
            if(!_S36)
            {
                bi_0 = bi_0 + u32(256);
                continue;
            }
            var ca_0 : u32 = law_1.ids_0.y - _S7;
            var cb_0 : u32 = law_1.ids_0.z - _S7;
            if((g_plug_0[ca_0]) == (g_plug_0[cb_0]))
            {
                bi_0 = bi_0 + u32(256);
                continue;
            }
            var o_1 : u32;
            if((g_plug_0[ca_0]) != u32(0))
            {
                o_1 = cb_0;
            }
            else
            {
                o_1 = ca_0;
            }
            var ok_0 : u32 = _S7 + o_1;
            var r_4 : vec3<f32> = origin_0 + rotate_0(q_3, chunks_0[ok_0].center_0.xyz + state_0[u32(4) * ok_0].xyz) - hit_0;
            var along_1 : f32 = dot(r_4, d_0);
            var lateral_0 : f32 = length(r_4 - d_0 * vec3<f32>(along_1));
            if(along_1 >= (_S34 - chunks_0[ok_0].half_0.w))
            {
                out_0 = lateral_0 <= (face_0 + max(along_1 - _S34, 0.0f));
            }
            else
            {
                out_0 = false;
            }
            if(out_0)
            {
                g_plug_0[o_1] = u32(1);
                g_changed_0 = u32(1);
            }
            bi_0 = bi_0 + u32(256);
        }
        workgroupBarrier();
        if(g_changed_0 == u32(0))
        {
            break;
        }
        workgroupBarrier();
        hit_k_0 = hit_k_0 + u32(1);
    }
    c_0 = t_1;
    t1_1 = 0.0f;
    t0_1 = 0.0f;
    loop
    {
        if(c_0 < n_1)
        {
        }
        else
        {
            break;
        }
        if((g_plug_0[c_0]) != u32(0))
        {
            var count_0 : f32 = t0_1 + 1.0f;
            t1_1 = t1_1 + chunks_0[_S7 + c_0].center_0.w;
            t0_1 = count_0;
        }
        c_0 = c_0 + u32(256);
    }
    var _S37 : u32 = _S8 + t_1;
    bi_0 = _S37;
    var capacity_0 : f32 = 0.0f;
    loop
    {
        if(bi_0 < _S9)
        {
        }
        else
        {
            break;
        }
        var law_2 : JointBond_std430_0 = bond_static_0[bi_0].law_0;
        var _S38 : JointState_std430_0 = bond_dyn_0[bi_0].js_0;
        var _S39 : bool = connected_0(&(_S38), (bond_static_0[bi_0].law_0.stiff1_0.w) != 0.0f);
        if(!_S39)
        {
            bi_0 = bi_0 + u32(256);
            continue;
        }
        var capacity_1 : f32;
        if((g_plug_0[law_2.ids_0.y - _S7]) != (g_plug_0[law_2.ids_0.z - _S7]))
        {
            capacity_1 = capacity_0 + materials_0.m_0.data_0[law_2.ids_0.x].strength_0.x * law_2.geom0_0.x * law_2.geom1_0.w;
        }
        else
        {
            capacity_1 = capacity_0;
        }
        capacity_0 = capacity_1;
        bi_0 = bi_0 + u32(256);
    }
    g_a_0[t_1] = t1_1 + 0.0f * t0_1;
    g_b_0[t_1] = capacity_0;
    g_lo_0[t_1] = vec3<f32>(t0_1, 0.0f, 0.0f);
    workgroupBarrier();
    s_0 = u32(128);
    loop
    {
        if(s_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(t_1 < s_0)
        {
            var _S40 : u32 = t_1 + s_0;
            g_a_0[t_1] = g_a_0[t_1] + g_a_0[_S40];
            g_b_0[t_1] = g_b_0[t_1] + g_b_0[_S40];
            g_lo_0[t_1] = g_lo_0[t_1] + g_lo_0[_S40];
        }
        workgroupBarrier();
        s_0 = (s_0 >> (u32(1)));
    }
    if(t_1 == u32(0))
    {
        var m_imp_0 : f32 = imp_a_0.w;
        var m_eff_0 : f32 = g_a_0[i32(0)] * m_imp_0 / (g_a_0[i32(0)] + m_imp_0);
        var impulse_0 : f32 = m_eff_0 * speed_1;
        var modulus_0 : f32 = 1.0f / (1.0f / imp_b_0.x + 1.0f / chunks_0[entry_k_0].cmat_0.x);
        var tau_0 : f32 = 2.86999988555908203f * pow(m_eff_0 * m_eff_0 / (face_0 * modulus_0 * modulus_0 * max(speed_1, 0.00100000004749745f)), 0.20000000298023224f);
        if((g_lo_0[i32(0)].x) < f32(n_1))
        {
            out_0 = (impulse_0 / tau_0) > (g_b_0[i32(0)]);
        }
        else
        {
            out_0 = false;
        }
        if(out_0)
        {
            c_0 = u32(1);
        }
        else
        {
            c_0 = u32(0);
        }
        g_go_0 = c_0;
        if(out_0)
        {
            status_0[u32(16) + i_0] = (status_0[u32(16) + i_0] | (u32(8)));
            status_0[u32(1)] = u32(1);
        }
        else
        {
            status_0[u32(16) + i_0] = (status_0[u32(16) + i_0] & (u32(4294967287)));
        }
    }
    workgroupBarrier();
    if(g_go_0 == u32(0))
    {
        return;
    }
    bi_0 = _S37;
    loop
    {
        if(bi_0 < _S9)
        {
        }
        else
        {
            break;
        }
        var law_3 : JointBond_std430_0 = bond_static_0[bi_0].law_0;
        var _S41 : JointState_0 = JointState_0( bond_dyn_0[bi_0].js_0.damage_0, bond_dyn_0[bi_0].js_0.crush_0, bond_dyn_0[bi_0].js_0.kappa_0, bond_dyn_0[bi_0].js_0.kappa_c_0, bond_dyn_0[bi_0].js_0.ductility_0, bond_dyn_0[bi_0].js_0.ductility_c_0, bond_dyn_0[bi_0].js_0.life_0, bond_dyn_0[bi_0].js_0.plastic_x_0, bond_dyn_0[bi_0].js_0.plastic_y_0, bond_dyn_0[bi_0].js_0.plastic_t_0, bond_dyn_0[bi_0].js_0.rebar_plastic_0, bond_dyn_0[bi_0].js_0.rebar_slip0_0, bond_dyn_0[bi_0].js_0.rebar_slip1_0, bond_dyn_0[bi_0].js_0.rebar_work_0, bond_dyn_0[bi_0].js_0.rebar_broken_0, bond_dyn_0[bi_0].js_0.strain_rate_0, bond_dyn_0[bi_0].js_0.governing_stress_0, bond_dyn_0[bi_0].js_0.dissipated_0, bond_dyn_0[bi_0].js_0.utilization_0, bond_dyn_0[bi_0].js_0.mode_0 );
        var dyn_0 : BondDyn_0;
        dyn_0.js_0 = _S41;
        dyn_0.force_lin_0 = bond_dyn_0[bi_0].force_lin_0;
        dyn_0.force_ang_0 = bond_dyn_0[bi_0].force_ang_0;
        dyn_0.sums_0 = bond_dyn_0[bi_0].sums_0;
        dyn_0.comps_0 = bond_dyn_0[bi_0].comps_0;
        dyn_0.events_0 = bond_dyn_0[bi_0].events_0;
        if(!connected_1(dyn_0.js_0, (bond_static_0[bi_0].law_0.stiff1_0.w) != 0.0f))
        {
            bi_0 = bi_0 + u32(256);
            continue;
        }
        if((g_plug_0[law_3.ids_0.y - _S7]) != (g_plug_0[law_3.ids_0.z - _S7]))
        {
            dyn_0.js_0.damage_0 = 1.0f;
            dyn_0.js_0.rebar_broken_0 = 1.0f;
            dyn_0.events_0[i32(2)] = max(dyn_0.events_0.z, u32(1));
            bond_dyn_0[bi_0].js_0.damage_0 = dyn_0.js_0.damage_0;
            bond_dyn_0[bi_0].js_0.crush_0 = dyn_0.js_0.crush_0;
            bond_dyn_0[bi_0].js_0.kappa_0 = dyn_0.js_0.kappa_0;
            bond_dyn_0[bi_0].js_0.kappa_c_0 = dyn_0.js_0.kappa_c_0;
            bond_dyn_0[bi_0].js_0.ductility_0 = dyn_0.js_0.ductility_0;
            bond_dyn_0[bi_0].js_0.ductility_c_0 = dyn_0.js_0.ductility_c_0;
            bond_dyn_0[bi_0].js_0.life_0 = dyn_0.js_0.life_0;
            bond_dyn_0[bi_0].js_0.plastic_x_0 = dyn_0.js_0.plastic_x_0;
            bond_dyn_0[bi_0].js_0.plastic_y_0 = dyn_0.js_0.plastic_y_0;
            bond_dyn_0[bi_0].js_0.plastic_t_0 = dyn_0.js_0.plastic_t_0;
            bond_dyn_0[bi_0].js_0.rebar_plastic_0 = dyn_0.js_0.rebar_plastic_0;
            bond_dyn_0[bi_0].js_0.rebar_slip0_0 = dyn_0.js_0.rebar_slip0_0;
            bond_dyn_0[bi_0].js_0.rebar_slip1_0 = dyn_0.js_0.rebar_slip1_0;
            bond_dyn_0[bi_0].js_0.rebar_work_0 = dyn_0.js_0.rebar_work_0;
            bond_dyn_0[bi_0].js_0.rebar_broken_0 = dyn_0.js_0.rebar_broken_0;
            bond_dyn_0[bi_0].js_0.strain_rate_0 = dyn_0.js_0.strain_rate_0;
            bond_dyn_0[bi_0].js_0.governing_stress_0 = dyn_0.js_0.governing_stress_0;
            bond_dyn_0[bi_0].js_0.dissipated_0 = dyn_0.js_0.dissipated_0;
            bond_dyn_0[bi_0].js_0.utilization_0 = dyn_0.js_0.utilization_0;
            bond_dyn_0[bi_0].js_0.mode_0 = dyn_0.js_0.mode_0;
            bond_dyn_0[bi_0].force_lin_0 = dyn_0.force_lin_0;
            bond_dyn_0[bi_0].force_ang_0 = dyn_0.force_ang_0;
            bond_dyn_0[bi_0].sums_0 = dyn_0.sums_0;
            bond_dyn_0[bi_0].comps_0 = dyn_0.comps_0;
            bond_dyn_0[bi_0].events_0 = dyn_0.events_0;
        }
        bi_0 = bi_0 + u32(256);
    }
    return;
}

