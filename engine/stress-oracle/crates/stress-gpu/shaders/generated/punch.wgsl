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
@binding(12) @group(0) var<storage, read> route_0 : array<u32>;

@binding(10) @group(0) var<storage, read_write> status_0 : array<u32>;

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

@binding(11) @group(0) var<storage, read_write> fast_0 : array<vec4<f32>>;

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
    var r_3 : f32;
    var t0_0 : f32;
    var t1_1 : f32;
    var t0_1 : f32;
    var m_imp_0 : f32;
    var bi_0 : u32;
    var i_0 : u32 = group_0.x;
    var t_1 : u32 = thread_0.x;
    if(i_0 >= (params_0.island_count_0))
    {
        return;
    }
    if(route_0[i32(0)] == u32(0))
    {
        if(t_1 == u32(0))
        {
            status_0[u32(16) + i_0] = (status_0[u32(16) + i_0] & (u32(4294967287)));
        }
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
    var _S22 : u32 = (bitcast<u32>((fast_0[i32(0)].x)));
    var hit_k_1 : u32 = u32(4294967295);
    var hit_0 : vec3<f32> = _S21;
    var d_0 : vec3<f32> = _S21;
    var face_0 : f32 = 0.0f;
    var m_imp_1 : f32 = 0.0f;
    var speed_1 : f32 = 0.0f;
    var e_imp_0 : f32 = 0.0f;
    var k_1 : u32 = u32(0);
    loop
    {
        if(k_1 < (params_0.impactor_count_0 + _S22))
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
        var rel_0 : vec3<f32>;
        var c_2 : vec3<f32>;
        var info_2 : vec4<f32>;
        if(k_1 < (params_0.impactor_count_0))
        {
            var _S23 : u32 = u32(2) * k_1;
            var a_1 : vec4<f32> = impactors_0[_S23];
            var b_1 : EngineBody_std430_0 = bodies_0[(bitcast<u32>((a_1.x)))];
            var _S24 : vec3<f32> = to_zup_0(b_1.pos_inv_mass_0.xyz);
            var _S25 : f32 = a_1.y;
            var _S26 : vec4<f32> = vec4<f32>(a_1.z, a_1.w, impactors_0[_S23 + u32(1)].x, 0.0f);
            rel_0 = to_zup_0(b_1.vel_sleep_0.xyz) - v_2;
            r_3 = _S25;
            c_2 = _S24;
            info_2 = _S26;
        }
        else
        {
            var f_0 : u32 = u32(1) + u32(3) * (k_1 - params_0.impactor_count_0);
            var _S27 : u32 = f_0 + u32(1);
            if((bitcast<u32>((fast_0[_S27].w))) == i_0)
            {
                k_1 = k_1 + u32(1);
                continue;
            }
            var _S28 : vec3<f32> = fast_0[f_0].xyz;
            var _S29 : f32 = fast_0[f_0].w;
            rel_0 = fast_0[_S27].xyz - v_2;
            r_3 = _S29;
            c_2 = _S28;
            info_2 = fast_0[f_0 + u32(2)];
        }
        var sp_0 : f32 = length(rel_0);
        if(!(sp_0 > (params_0.speed_0)))
        {
            k_1 = k_1 + u32(1);
            continue;
        }
        var _S30 : vec3<f32> = vec3<f32>(r_3);
        var _S31 : vec3<f32> = _S19 - _S30;
        var _S32 : vec3<f32> = _S20 + _S30;
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
            var _S33 : vec3<f32> = rel_0;
            var _S34 : u32 = ax_0;
            if((abs(rel_0[ax_0])) < 9.999999960041972e-13f)
            {
                var _S35 : vec3<f32> = c_2;
                var _S36 : u32 = ax_0;
                var _S37 : bool;
                if((c_2[ax_0]) < (_S31[ax_0]))
                {
                    _S37 = true;
                }
                else
                {
                    _S37 = (_S35[_S36]) > (_S32[ax_0]);
                }
                if(_S37)
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
            var ta_0 : f32 = (_S31[ax_0] - c_2[ax_0]) / _S33[_S34];
            var tb_0 : f32 = (_S32[ax_0] - c_2[ax_0]) / _S33[_S34];
            var _S38 : f32 = min(t1_1, max(ta_0, tb_0));
            t0_0 = max(t0_0, min(ta_0, tb_0));
            t1_1 = _S38;
            ax_0 = ax_0 + u32(1);
        }
        var hit_1 : vec3<f32>;
        var d_1 : vec3<f32>;
        var speed_2 : f32;
        var e_imp_1 : f32;
        if(t0_0 <= t1_1)
        {
            var _S39 : vec3<f32> = c_2 + rel_0 * vec3<f32>(t0_0);
            var _S40 : vec3<f32> = rel_0 / vec3<f32>(sp_0);
            var _S41 : f32 = info_2.x;
            var _S42 : f32 = info_2.y;
            var _S43 : f32 = info_2.z;
            hit_k_0 = k_1;
            hit_1 = _S39;
            d_1 = _S40;
            t0_1 = _S41;
            m_imp_0 = _S42;
            speed_2 = sp_0;
            e_imp_1 = _S43;
        }
        else
        {
            hit_k_0 = hit_k_1;
            hit_1 = hit_0;
            d_1 = d_0;
            t0_1 = face_0;
            m_imp_0 = m_imp_1;
            speed_2 = speed_1;
            e_imp_1 = e_imp_0;
        }
        hit_k_1 = hit_k_0;
        hit_0 = hit_1;
        d_0 = d_1;
        face_0 = t0_1;
        m_imp_1 = m_imp_0;
        speed_1 = speed_2;
        e_imp_0 = e_imp_1;
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
    r_3 = inf_0;
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
        var r_4 : vec3<f32> = origin_0 + rotate_0(q_3, chunks_0[k_2].center_0.xyz + state_0[u32(4) * k_2].xyz) - hit_0;
        var along_0 : f32 = dot(r_4, d_0);
        var dist_0 : f32 = length(r_4 - d_0 * vec3<f32>(along_0)) + abs(along_0) * 0.00100000004749745f;
        if(dist_0 < r_3)
        {
            r_3 = dist_0;
            ax_0 = c_0;
        }
        c_0 = c_0 + u32(256);
    }
    g_a_0[t_1] = r_3;
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
            var _S44 : u32 = t_1 + s_0;
            if((g_a_0[_S44]) < (g_a_0[t_1]))
            {
                out_0 = true;
            }
            else
            {
                if((g_a_0[_S44]) == (g_a_0[t_1]))
                {
                    out_0 = (g_idx_0[_S44]) < (g_idx_0[t_1]);
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
            var _S45 : u32 = t_1 + s_0;
            g_a_0[t_1] = g_a_0[_S45];
            g_idx_0[t_1] = g_idx_0[_S45];
        }
        workgroupBarrier();
        s_0 = (s_0 >> (u32(1)));
    }
    var entry_0 : u32 = g_idx_0[i32(0)];
    var entry_k_0 : u32 = _S7 + g_idx_0[i32(0)];
    var _S46 : f32 = dot(origin_0 + rotate_0(q_3, chunks_0[entry_k_0].center_0.xyz + state_0[u32(4) * entry_k_0].xyz) - hit_0, d_0);
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
            var _S47 : JointState_std430_0 = bond_dyn_0[bi_0].js_0;
            var _S48 : bool = connected_0(&(_S47), (bond_static_0[bi_0].law_0.stiff1_0.w) != 0.0f);
            if(!_S48)
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
            var r_5 : vec3<f32> = origin_0 + rotate_0(q_3, chunks_0[ok_0].center_0.xyz + state_0[u32(4) * ok_0].xyz) - hit_0;
            var along_1 : f32 = dot(r_5, d_0);
            var lateral_0 : f32 = length(r_5 - d_0 * vec3<f32>(along_1));
            if(along_1 >= (_S46 - chunks_0[ok_0].half_0.w))
            {
                out_0 = lateral_0 <= (face_0 + max(along_1 - _S46, 0.0f));
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
    t0_0 = 0.0f;
    t1_1 = 0.0f;
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
            var count_0 : f32 = t1_1 + 1.0f;
            t0_0 = t0_0 + chunks_0[_S7 + c_0].center_0.w;
            t1_1 = count_0;
        }
        c_0 = c_0 + u32(256);
    }
    var _S49 : u32 = _S8 + t_1;
    bi_0 = _S49;
    t0_1 = 0.0f;
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
        var _S50 : JointState_std430_0 = bond_dyn_0[bi_0].js_0;
        var _S51 : bool = connected_0(&(_S50), (bond_static_0[bi_0].law_0.stiff1_0.w) != 0.0f);
        if(!_S51)
        {
            bi_0 = bi_0 + u32(256);
            continue;
        }
        if((g_plug_0[law_2.ids_0.y - _S7]) != (g_plug_0[law_2.ids_0.z - _S7]))
        {
            m_imp_0 = t0_1 + materials_0.m_0.data_0[law_2.ids_0.x].strength_0.x * law_2.geom0_0.x * law_2.geom1_0.w;
        }
        else
        {
            m_imp_0 = t0_1;
        }
        t0_1 = m_imp_0;
        bi_0 = bi_0 + u32(256);
    }
    g_a_0[t_1] = t0_0 + 0.0f * t1_1;
    g_b_0[t_1] = t0_1;
    g_lo_0[t_1] = vec3<f32>(t1_1, 0.0f, 0.0f);
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
            var _S52 : u32 = t_1 + s_0;
            g_a_0[t_1] = g_a_0[t_1] + g_a_0[_S52];
            g_b_0[t_1] = g_b_0[t_1] + g_b_0[_S52];
            g_lo_0[t_1] = g_lo_0[t_1] + g_lo_0[_S52];
        }
        workgroupBarrier();
        s_0 = (s_0 >> (u32(1)));
    }
    if(t_1 == u32(0))
    {
        var m_eff_0 : f32 = g_a_0[i32(0)] * m_imp_1 / (g_a_0[i32(0)] + m_imp_1);
        var impulse_0 : f32 = m_eff_0 * speed_1;
        var modulus_0 : f32 = 1.0f / (1.0f / e_imp_0 + 1.0f / chunks_0[entry_k_0].cmat_0.x);
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
    bi_0 = _S49;
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
        var _S53 : JointState_0 = JointState_0( bond_dyn_0[bi_0].js_0.damage_0, bond_dyn_0[bi_0].js_0.crush_0, bond_dyn_0[bi_0].js_0.kappa_0, bond_dyn_0[bi_0].js_0.kappa_c_0, bond_dyn_0[bi_0].js_0.ductility_0, bond_dyn_0[bi_0].js_0.ductility_c_0, bond_dyn_0[bi_0].js_0.life_0, bond_dyn_0[bi_0].js_0.plastic_x_0, bond_dyn_0[bi_0].js_0.plastic_y_0, bond_dyn_0[bi_0].js_0.plastic_t_0, bond_dyn_0[bi_0].js_0.rebar_plastic_0, bond_dyn_0[bi_0].js_0.rebar_slip0_0, bond_dyn_0[bi_0].js_0.rebar_slip1_0, bond_dyn_0[bi_0].js_0.rebar_work_0, bond_dyn_0[bi_0].js_0.rebar_broken_0, bond_dyn_0[bi_0].js_0.strain_rate_0, bond_dyn_0[bi_0].js_0.governing_stress_0, bond_dyn_0[bi_0].js_0.dissipated_0, bond_dyn_0[bi_0].js_0.utilization_0, bond_dyn_0[bi_0].js_0.mode_0 );
        var dyn_0 : BondDyn_0;
        dyn_0.js_0 = _S53;
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

var<workgroup> g_scan_0 : array<u32, i32(1024)>;

struct Island_0
{
     range_0 : vec4<u32>,
     info_0 : vec4<u32>,
     com_0 : vec4<f32>,
     inertia0_0 : vec4<f32>,
     inertia1_0 : vec4<f32>,
     inertia2_0 : vec4<f32>,
     inv0_0 : vec4<f32>,
     inv1_0 : vec4<f32>,
     inv2_0 : vec4<f32>,
     wcom_0 : vec4<f32>,
     winv0_0 : vec4<f32>,
     winv1_0 : vec4<f32>,
     winv2_0 : vec4<f32>,
     rotation_0 : vec4<f32>,
     position_0 : vec4<f32>,
     position_err_0 : vec4<f32>,
     velocity_0 : vec4<f32>,
     velocity_err_0 : vec4<f32>,
     angular_velocity_0 : vec4<f32>,
     done_0 : u32,
     coverage_0 : u32,
     wide_groups_0 : u32,
     done_pad_0 : u32,
     probes_0 : vec4<u32>,
     energy_0 : vec4<f32>,
     rotation_err_0 : vec4<f32>,
     momentum_0 : vec4<f32>,
     momentum_err_0 : vec4<f32>,
};

@compute
@workgroup_size(1024, 1, 1)
fn punch_fast(@builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var add_0 : u32;
    var t_2 : u32 = thread_1.x;
    var base_0 : u32 = u32(0);
    var carry_0 : u32 = u32(0);
    loop
    {
        if(base_0 < (params_0.island_count_0))
        {
        }
        else
        {
            break;
        }
        var i_1 : u32 = base_0 + t_2;
        var _S54 : vec3<f32> = vec3<f32>(0.0f);
        var _S55 : bool;
        if(i_1 < (params_0.island_count_0))
        {
            _S55 = route_0[i32(0)] != u32(0);
        }
        else
        {
            _S55 = false;
        }
        var k_3 : u32;
        var _S56 : bool;
        var fast_i_0 : bool;
        var radius_0 : f32;
        var face_r_0 : f32;
        var c_3 : vec3<f32>;
        var vel_0 : vec3<f32>;
        var isl_0 : Island_0;
        if(_S55)
        {
            var isl_1 : Island_std430_0 = islands_0[i_1];
            var _S57 : u32 = u32(3) * i_1;
            var body_1 : u32 = (bitcast<u32>((ctab_0[params_0.engine_base_0 + _S57].x)));
            var _S58 : u32 = isl_1.range_0.y;
            var _S59 : u32 = isl_1.range_0.x;
            var n_2 : u32 = _S58 - _S59;
            if(n_2 > u32(0))
            {
                _S56 = n_2 <= u32(64);
            }
            else
            {
                _S56 = false;
            }
            var _S60 : bool;
            if(_S56)
            {
                _S60 = body_1 != u32(4294967295);
            }
            else
            {
                _S60 = false;
            }
            var _S61 : bool;
            if(_S60)
            {
                _S61 = (((isl_1.info_0.x) & (u32(1)))) == u32(0);
            }
            else
            {
                _S61 = false;
            }
            if(_S61)
            {
                var b_2 : EngineBody_std430_0 = bodies_0[body_1];
                var vel_1 : vec3<f32> = to_zup_0(b_2.vel_sleep_0.xyz);
                if((length(vel_1)) > (params_0.speed_0))
                {
                    var c_4 : vec3<f32> = to_zup_0(b_2.pos_inv_mass_0.xyz);
                    var p_inv_1 : Quat_0 = quat_of_0(ctab_0[params_0.engine_base_0 + _S57 + u32(1)]);
                    var com_2 : vec3<f32> = ctab_0[params_0.engine_base_0 + _S57 + u32(2)].xyz;
                    var a_inv_1 : Quat_0;
                    a_inv_1.w_0 = 0.70710676908493042f;
                    a_inv_1.x_0 = 0.70710676908493042f;
                    a_inv_1.y_0 = 0.0f;
                    a_inv_1.z_0 = 0.0f;
                    var q_5 : Quat_0 = normalized_0(quat_mul_0(quat_mul_0(a_inv_1, quat_of_0(b_2.rot_0)), p_inv_1));
                    var _S62 : vec3<f32> = c_4 - rotate_0(q_5, com_2);
                    var _S63 : vec3<f32> = normalize(vel_1);
                    radius_0 = 0.0f;
                    face_r_0 = 0.0f;
                    k_3 = _S59;
                    loop
                    {
                        if(k_3 < _S58)
                        {
                        }
                        else
                        {
                            break;
                        }
                        var r_6 : vec3<f32> = _S62 + rotate_0(q_5, chunks_0[k_3].center_0.xyz + state_0[u32(4) * k_3].xyz) - c_4;
                        var _S64 : f32 = max(radius_0, length(r_6) + chunks_0[k_3].half_0.w);
                        var _S65 : f32 = max(face_r_0, length(r_6 - _S63 * vec3<f32>(dot(r_6, _S63))) + chunks_0[k_3].half_0.w);
                        var _S66 : u32 = k_3 + u32(1);
                        radius_0 = _S64;
                        face_r_0 = _S65;
                        k_3 = _S66;
                    }
                    fast_i_0 = true;
                    c_3 = c_4;
                }
                else
                {
                    fast_i_0 = false;
                    c_3 = _S54;
                    radius_0 = 0.0f;
                    face_r_0 = 0.0f;
                }
                vel_0 = vel_1;
            }
            else
            {
                fast_i_0 = false;
                c_3 = _S54;
                radius_0 = 0.0f;
                vel_0 = _S54;
                face_r_0 = 0.0f;
            }
            isl_0.range_0 = isl_1.range_0;
            isl_0.info_0 = isl_1.info_0;
            isl_0.com_0 = isl_1.com_0;
            isl_0.inertia0_0 = isl_1.inertia0_0;
            isl_0.inertia1_0 = isl_1.inertia1_0;
            isl_0.inertia2_0 = isl_1.inertia2_0;
            isl_0.inv0_0 = isl_1.inv0_0;
            isl_0.inv1_0 = isl_1.inv1_0;
            isl_0.inv2_0 = isl_1.inv2_0;
            isl_0.wcom_0 = isl_1.wcom_0;
            isl_0.winv0_0 = isl_1.winv0_0;
            isl_0.winv1_0 = isl_1.winv1_0;
            isl_0.winv2_0 = isl_1.winv2_0;
            isl_0.rotation_0 = isl_1.rotation_0;
            isl_0.position_0 = isl_1.position_0;
            isl_0.position_err_0 = isl_1.position_err_0;
            isl_0.velocity_0 = isl_1.velocity_0;
            isl_0.velocity_err_0 = isl_1.velocity_err_0;
            isl_0.angular_velocity_0 = isl_1.angular_velocity_0;
            isl_0.done_0 = isl_1.done_0;
            isl_0.coverage_0 = isl_1.coverage_0;
            isl_0.wide_groups_0 = isl_1.wide_groups_0;
            isl_0.done_pad_0 = isl_1.done_pad_0;
            isl_0.probes_0 = isl_1.probes_0;
            isl_0.energy_0 = isl_1.energy_0;
            isl_0.rotation_err_0 = isl_1.rotation_err_0;
            isl_0.momentum_0 = isl_1.momentum_0;
            isl_0.momentum_err_0 = isl_1.momentum_err_0;
        }
        else
        {
            fast_i_0 = false;
            c_3 = _S54;
            radius_0 = 0.0f;
            vel_0 = _S54;
            face_r_0 = 0.0f;
        }
        if(fast_i_0)
        {
            k_3 = u32(1);
        }
        else
        {
            k_3 = u32(0);
        }
        g_scan_0[t_2] = k_3;
        workgroupBarrier();
        var step_0 : u32 = u32(1);
        loop
        {
            if(step_0 < u32(1024))
            {
            }
            else
            {
                break;
            }
            if(t_2 >= step_0)
            {
                add_0 = g_scan_0[t_2 - step_0];
            }
            else
            {
                add_0 = u32(0);
            }
            workgroupBarrier();
            g_scan_0[t_2] = g_scan_0[t_2] + add_0;
            workgroupBarrier();
            step_0 = (step_0 << (u32(1)));
        }
        var _S67 : u32 = carry_0 + g_scan_0[t_2];
        if(fast_i_0)
        {
            add_0 = u32(1);
        }
        else
        {
            add_0 = u32(0);
        }
        var at_0 : u32 = _S67 - add_0;
        if(fast_i_0)
        {
            _S56 = at_0 < u32(128);
        }
        else
        {
            _S56 = false;
        }
        if(_S56)
        {
            var _S68 : u32 = u32(3) * at_0;
            fast_0[u32(1) + _S68] = vec4<f32>(c_3, radius_0);
            fast_0[u32(2) + _S68] = vec4<f32>(vel_0, (bitcast<f32>((i_1))));
            fast_0[u32(3) + _S68] = vec4<f32>(face_r_0, isl_0.com_0.w, chunks_0[isl_0.range_0.x].cmat_0.x, 0.0f);
        }
        var total_0 : u32 = g_scan_0[i32(1023)];
        workgroupBarrier();
        var carry_1 : u32 = carry_0 + total_0;
        base_0 = base_0 + u32(1024);
        carry_0 = carry_1;
    }
    if(t_2 == u32(0))
    {
        fast_0[i32(0)] = vec4<f32>((bitcast<f32>((min(carry_0, u32(128))))), 0.0f, 0.0f, 0.0f);
    }
    return;
}

