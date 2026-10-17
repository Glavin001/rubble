struct Params_std140_0
{
    @align(16) gravity_0 : vec4<f32>,
    @align(16) dt_0 : f32,
    @align(4) fracture_0 : u32,
    @align(8) rigid_motion_loads_0 : u32,
    @align(4) step_start_0 : u32,
    @align(16) t_hi_0 : f32,
    @align(4) t_lo_0 : f32,
    @align(8) max_steps_0 : u32,
    @align(4) contact_mode_0 : u32,
    @align(16) halt_index_0 : u32,
    @align(4) chunk_count_0 : u32,
    @align(8) pair_count_0 : u32,
    @align(4) impactor_count_0 : u32,
    @align(16) cand_count_0 : u32,
    @align(4) probe_base_0 : u32,
    @align(8) probe_stride_0 : u32,
    @align(4) slot_base_0 : u32,
    @align(16) ledger_base_0 : u32,
    @align(4) record_base_0 : u32,
    @align(8) record_stride_0 : u32,
    @align(4) cand_base_0 : u32,
    @align(16) pair_index_0 : u32,
    @align(4) cand_index_0 : u32,
    @align(8) zeta_0 : f32,
    @align(4) pair_friction_0 : f32,
    @align(16) ground_hi_0 : f32,
    @align(4) ground_lo_0 : f32,
    @align(8) ground_friction_0 : f32,
    @align(4) ground_modulus_0 : f32,
    @align(16) has_ground_0 : u32,
    @align(4) wide_bond_groups_0 : u32,
    @align(8) wide_bond_table_0 : u32,
    @align(4) wide_chunk_table_0 : u32,
    @align(16) wide_base_0 : u32,
    @align(4) seg_index_0 : u32,
    @align(8) seg_count_0 : u32,
    @align(4) seg_base_0 : u32,
    @align(16) solve_mode_0 : u32,
    @align(4) cload_base_0 : u32,
    @align(8) cframe_base_0 : u32,
    @align(4) statics_base_0 : u32,
    @align(16) statics_bonds_0 : u32,
    @align(4) statics_newton_0 : u32,
    @align(8) statics_cg_0 : u32,
    @align(4) statics_tol_0 : f32,
    @align(16) pair_lanes_0 : u32,
    @align(4) device_connect_0 : u32,
    @align(8) cplan_base_0 : u32,
    @align(4) iplan_base_0 : u32,
    @align(16) contact_bond_groups_0 : u32,
    @align(4) contact_bond_table_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : Params_std140_0;
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

@binding(9) @group(0) var<storage, read_write> islands_0 : array<Island_std430_0>;

@binding(4) @group(0) var<storage, read> index_0 : array<u32>;

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

@binding(3) @group(0) var<storage, read> chunks_0 : array<ChunkStatic_std430_0>;

@binding(6) @group(0) var<storage, read_write> state_0 : array<vec4<f32>>;

@binding(8) @group(0) var<storage, read_write> scratch_0 : array<vec4<f32>>;

@binding(11) @group(0) var<storage, read_write> contact_state_0 : array<vec4<f32>>;

@binding(5) @group(0) var<storage, read> loads_0 : array<vec4<f32>>;

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
    @align(16) crush_0 : vec4<f32>,
    @align(16) geom_0 : vec4<f32>,
    @align(16) rotation_err_1 : vec4<f32>,
    @align(16) ledger_0 : vec4<f32>,
    @align(16) cand_0 : vec4<u32>,
    @align(16) momentum_1 : vec4<f32>,
    @align(16) momentum_err_1 : vec4<f32>,
};

@binding(10) @group(0) var<storage, read_write> impactors_0 : array<Impactor_std430_0>;

struct JointState_std430_0
{
    @align(4) damage_0 : f32,
    @align(4) crush_1 : f32,
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

@binding(7) @group(0) var<storage, read_write> bond_dyn_0 : array<BondDyn_std430_0>;

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

@binding(2) @group(0) var<storage, read> bonds_0 : array<BondStatic_std430_0>;

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

@binding(1) @group(0) var<uniform> materials_0 : MaterialTable_std140_0;
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

var<workgroup> g_seen_0 : array<u32, i32(5)>;

fn seen_clear_0( tid_0 : u32)
{
    if(tid_0 < u32(5))
    {
        g_seen_0[tid_0] = u32(0);
    }
    return;
}

fn contact_stopped_0( isl_0 : Island_0) -> bool
{
    var _S1 : vec4<u32> = islands_0[params_0.halt_index_0].info_0;
    var _S2 : bool;
    if((((islands_0[params_0.halt_index_0].info_0.z) & (u32(1)))) != u32(0))
    {
        _S2 = true;
    }
    else
    {
        var _S3 : u32 = _S1.y;
        if(_S3 != u32(0))
        {
            _S2 = _S3 <= (isl_0.info_0.w);
        }
        else
        {
            _S2 = false;
        }
    }
    if(_S2)
    {
        _S2 = true;
    }
    else
    {
        var _S4 : u32 = islands_0[params_0.halt_index_0].range_0.x;
        if(_S4 != u32(0))
        {
            _S2 = _S4 <= (isl_0.info_0.w + u32(1));
        }
        else
        {
            _S2 = false;
        }
    }
    return _S2;
}

var<workgroup> g_run_0 : u32;

var<workgroup> g_halt_0 : u32;

fn stop_genuine_0( _S5 : u32,  _S6 : u32) -> bool
{
    var at_0 : u32 = islands_0[_S5].range_0.x;
    var _S7 : bool;
    if(at_0 != u32(0))
    {
        _S7 = at_0 <= _S6;
    }
    else
    {
        _S7 = false;
    }
    if(_S7)
    {
        var _S8 : u32 = islands_0[_S5].info_0.y;
        if(_S8 == u32(0))
        {
            _S7 = true;
        }
        else
        {
            _S7 = at_0 <= _S8;
        }
    }
    else
    {
        _S7 = false;
    }
    return _S7;
}

fn cov_or_0( island_0 : u32,  bits_0 : u32)
{
    islands_0[island_0].coverage_0 = ((islands_0[island_0].coverage_0) | (bits_0));
    return;
}

fn cov_mark_0( island_1 : u32,  seen_0 : u32,  bits_1 : u32)
{
    if(((bits_1 & ((~seen_0)))) != u32(0))
    {
        cov_or_0(island_1, bits_1);
    }
    return;
}

fn sentinel_mark_0( bits_2 : u32)
{
    cov_mark_0(params_0.halt_index_0, islands_0[params_0.halt_index_0].coverage_0, bits_2);
    return;
}

fn stop_seen_0( _S9 : u32,  _S10 : u32)
{
    if(stop_genuine_0(_S9, _S10))
    {
        islands_0[params_0.halt_index_0].info_0[i32(2)] = ((islands_0[params_0.halt_index_0].info_0.z) | (u32(1)));
        sentinel_mark_0(u32(1048576));
    }
    return;
}

var<workgroup> g_sums_0 : u32;

struct Quat_0
{
     w_0 : f32,
     x_0 : f32,
     y_0 : f32,
     z_0 : f32,
};

struct Rigid_0
{
     rot_0 : Quat_0,
     rot_err_0 : vec4<f32>,
     pos_0 : vec3<f32>,
     pos_err_0 : vec3<f32>,
     vel_0 : vec3<f32>,
     vel_err_0 : vec3<f32>,
     w_1 : vec3<f32>,
     l_0 : vec3<f32>,
     l_err_0 : vec3<f32>,
     torque_0 : vec3<f32>,
     a_0 : vec3<f32>,
     alpha_0 : vec3<f32>,
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

fn rigid_of_0( isl_1 : Island_0) -> Rigid_0
{
    var rg_0 : Rigid_0;
    rg_0.rot_0 = quat_of_0(isl_1.rotation_0);
    rg_0.rot_err_0 = isl_1.rotation_err_0;
    rg_0.pos_0 = isl_1.position_0.xyz;
    rg_0.pos_err_0 = isl_1.position_err_0.xyz;
    rg_0.vel_0 = isl_1.velocity_0.xyz;
    rg_0.vel_err_0 = isl_1.velocity_err_0.xyz;
    rg_0.w_1 = isl_1.angular_velocity_0.xyz;
    rg_0.l_0 = isl_1.momentum_0.xyz;
    rg_0.l_err_0 = isl_1.momentum_err_0.xyz;
    var _S11 : vec3<f32> = vec3<f32>(0.0f);
    rg_0.torque_0 = _S11;
    rg_0.a_0 = _S11;
    rg_0.alpha_0 = _S11;
    return rg_0;
}

struct WorldPoint_0
{
     hi_0 : vec3<f32>,
     lo_0 : vec3<f32>,
     rel_0 : vec3<f32>,
};

fn rotate_0( q_1 : Quat_0,  v_0 : vec3<f32>) -> vec3<f32>
{
    var qv_0 : vec3<f32> = vec3<f32>(q_1.x_0, q_1.y_0, q_1.z_0);
    var t_0 : vec3<f32> = cross(qv_0, v_0) * vec3<f32>(2.0f);
    return v_0 + t_0 * vec3<f32>(q_1.w_0) + cross(qv_0, t_0);
}

fn chunk_world_0( c_0 : u32) -> WorldPoint_0
{
    var q_2 : Quat_0 = quat_of_0(islands_0[chunks_0[c_0].info_1.y].rotation_0);
    var w_2 : WorldPoint_0;
    w_2.hi_0 = islands_0[chunks_0[c_0].info_1.y].position_0.xyz;
    w_2.lo_0 = islands_0[chunks_0[c_0].info_1.y].position_err_0.xyz;
    w_2.rel_0 = rotate_0(q_2, chunks_0[c_0].center_0.xyz + state_0[u32(4) * c_0].xyz);
    return w_2;
}

struct Box_0
{
     center_1 : vec3<f32>,
     axis0_0 : vec3<f32>,
     axis1_0 : vec3<f32>,
     axis2_0 : vec3<f32>,
     half_2 : vec3<f32>,
     hull_at_0 : u32,
     hull_v_0 : u32,
     hull_f_0 : u32,
};

fn safe_normalize_0( v_1 : vec3<f32>) -> vec3<f32>
{
    var n_0 : f32 = length(v_1);
    var _S12 : vec3<f32>;
    if(n_0 > 1.00000000317107685e-30f)
    {
        _S12 = v_1 / vec3<f32>(n_0);
    }
    else
    {
        _S12 = vec3<f32>(0.0f);
    }
    return _S12;
}

fn from_axis_angle_0( axis_0 : vec3<f32>,  angle_0 : f32) -> Quat_0
{
    var a_1 : vec3<f32> = safe_normalize_0(axis_0);
    var _S13 : f32 = 0.5f * angle_0;
    var s_0 : f32 = sin(_S13);
    var q_3 : Quat_0;
    q_3.w_0 = cos(_S13);
    q_3.x_0 = a_1.x * s_0;
    q_3.y_0 = a_1.y * s_0;
    q_3.z_0 = a_1.z * s_0;
    return q_3;
}

fn chunk_box_0( c_1 : u32,  center_2 : vec3<f32>) -> Box_0
{
    var q_4 : Quat_0 = quat_of_0(islands_0[chunks_0[c_1].info_1.y].rotation_0);
    var th_0 : vec3<f32> = state_0[u32(4) * c_1 + u32(1)].xyz;
    var hidden_0 : Quat_0 = from_axis_angle_0(th_0, length(th_0));
    var b_0 : Box_0;
    b_0.center_1 = center_2;
    b_0.axis0_0 = rotate_0(q_4, rotate_0(hidden_0, vec3<f32>(chunks_0[c_1].crot0_0.x, chunks_0[c_1].crot1_0.x, chunks_0[c_1].crot2_0.x)));
    b_0.axis1_0 = rotate_0(q_4, rotate_0(hidden_0, vec3<f32>(chunks_0[c_1].crot0_0.y, chunks_0[c_1].crot1_0.y, chunks_0[c_1].crot2_0.y)));
    b_0.axis2_0 = rotate_0(q_4, rotate_0(hidden_0, vec3<f32>(chunks_0[c_1].crot0_0.z, chunks_0[c_1].crot1_0.z, chunks_0[c_1].crot2_0.z)));
    b_0.half_2 = chunks_0[c_1].half_0.xyz;
    b_0.hull_at_0 = (bitcast<u32>((chunks_0[c_1].cmat_0.z)));
    var _S14 : u32 = (bitcast<u32>((chunks_0[c_1].cmat_0.w)));
    b_0.hull_v_0 = (_S14 & (u32(255)));
    b_0.hull_f_0 = (_S14 >> (u32(8)));
    return b_0;
}

fn box_axis_0( b_1 : Box_0,  k_0 : u32) -> vec3<f32>
{
    var _S15 : vec3<f32>;
    if(k_0 == u32(0))
    {
        _S15 = b_1.axis0_0;
    }
    else
    {
        if(k_0 == u32(1))
        {
            _S15 = b_1.axis1_0;
        }
        else
        {
            _S15 = b_1.axis2_0;
        }
    }
    return _S15;
}

fn comp3_0( v_2 : vec3<f32>,  k_1 : u32) -> f32
{
    var _S16 : f32;
    if(k_1 == u32(0))
    {
        _S16 = v_2.x;
    }
    else
    {
        if(k_1 == u32(1))
        {
            _S16 = v_2.y;
        }
        else
        {
            _S16 = v_2.z;
        }
    }
    return _S16;
}

fn half_thickness_and_area_0( b_2 : Box_0,  d_0 : vec3<f32>) -> vec2<f32>
{
    var k_2 : u32 = u32(0);
    var h_0 : f32 = 0.0f;
    var area_0 : f32 = 0.0f;
    loop
    {
        if(k_2 < u32(3))
        {
        }
        else
        {
            break;
        }
        var c_2 : f32 = abs(dot(d_0, box_axis_0(b_2, k_2)));
        var h_1 : f32 = h_0 + c_2 * comp3_0(b_2.half_2, k_2);
        var _S17 : u32 = k_2 + u32(1);
        var area_1 : f32 = area_0 + c_2 * 4.0f * comp3_0(b_2.half_2, _S17 % u32(3)) * comp3_0(b_2.half_2, (k_2 + u32(2)) % u32(3));
        k_2 = _S17;
        h_0 = h_1;
        area_0 = area_1;
    }
    return vec2<f32>(h_0, area_0);
}

fn contact_stiffness_0( ea_0 : f32,  a_2 : Box_0,  eb_0 : f32,  b_3 : Box_0,  dir_0 : vec3<f32>) -> f32
{
    var d_1 : vec3<f32> = safe_normalize_0(dir_0);
    var ta_0 : vec2<f32> = half_thickness_and_area_0(a_2, d_1);
    var tb_0 : vec2<f32> = half_thickness_and_area_0(b_3, d_1);
    return min(ta_0.y, tb_0.y) / (ta_0.x / ea_0 + tb_0.x / eb_0);
}

fn sample_count_0( b_4 : Box_0) -> u32
{
    var _S18 : u32;
    if((b_4.hull_v_0) == u32(0))
    {
        _S18 = u32(14);
    }
    else
    {
        _S18 = b_4.hull_v_0 + b_4.hull_f_0;
    }
    return _S18;
}

fn box_to_world_0( b_5 : Box_0,  local_0 : vec3<f32>) -> vec3<f32>
{
    return b_5.axis0_0 * vec3<f32>(local_0.x) + b_5.axis1_0 * vec3<f32>(local_0.y) + b_5.axis2_0 * vec3<f32>(local_0.z);
}

fn sample_point_0( b_6 : Box_0,  i_0 : u32) -> vec3<f32>
{
    if((b_6.hull_v_0) != u32(0))
    {
        var local_1 : vec3<f32>;
        if(i_0 < (b_6.hull_v_0))
        {
            local_1 = loads_0[b_6.hull_at_0 + i_0].xyz * vec3<f32>(0.89999997615814209f);
        }
        else
        {
            local_1 = loads_0[b_6.hull_at_0 + b_6.hull_v_0 + b_6.hull_f_0 + (i_0 - b_6.hull_v_0)].xyz;
        }
        return b_6.center_1 + box_to_world_0(b_6, local_1);
    }
    var sign_0 : f32;
    if(i_0 < u32(8))
    {
        var h_2 : vec3<f32> = b_6.half_2 * vec3<f32>(0.89999997615814209f);
        if(((i_0 & (u32(1)))) == u32(0))
        {
            sign_0 = - h_2.x;
        }
        else
        {
            sign_0 = h_2.x;
        }
        var _S19 : f32;
        if(((i_0 & (u32(2)))) == u32(0))
        {
            _S19 = - h_2.y;
        }
        else
        {
            _S19 = h_2.y;
        }
        var _S20 : f32;
        if(((i_0 & (u32(4)))) == u32(0))
        {
            _S20 = - h_2.z;
        }
        else
        {
            _S20 = h_2.z;
        }
        return b_6.center_1 + b_6.axis0_0 * vec3<f32>(sign_0) + b_6.axis1_0 * vec3<f32>(_S19) + b_6.axis2_0 * vec3<f32>(_S20);
    }
    var _S21 : u32 = i_0 - u32(8);
    var axis_1 : u32 = _S21 / u32(2);
    if((_S21 % u32(2)) == u32(0))
    {
        sign_0 = -1.0f;
    }
    else
    {
        sign_0 = 1.0f;
    }
    return b_6.center_1 + box_axis_0(b_6, axis_1) * vec3<f32>((sign_0 * comp3_0(b_6.half_2, axis_1)));
}

fn chunk_velocity_0( c_3 : u32,  v_3 : ptr<function, vec3<f32>>,  w_3 : ptr<function, vec3<f32>>)
{
    var q_5 : Quat_0 = quat_of_0(islands_0[chunks_0[c_3].info_1.y].rotation_0);
    var _S22 : u32 = u32(4) * c_3;
    var _S23 : vec3<f32> = islands_0[chunks_0[c_3].info_1.y].angular_velocity_0.xyz;
    (*v_3) = islands_0[chunks_0[c_3].info_1.y].velocity_0.xyz + islands_0[chunks_0[c_3].info_1.y].velocity_err_0.xyz + cross(_S23, rotate_0(q_5, chunks_0[c_3].center_0.xyz + state_0[_S22].xyz - islands_0[chunks_0[c_3].info_1.y].com_0.xyz)) + rotate_0(q_5, state_0[_S22 + u32(2)].xyz);
    (*w_3) = _S23 + rotate_0(q_5, state_0[_S22 + u32(3)].xyz);
    return;
}

fn penalty_force_0( k_3 : f32,  m_red_0 : f32,  friction_0 : f32,  depth_0 : f32,  normal_1 : vec3<f32>,  rel_velocity_0 : vec3<f32>,  dt_1 : f32,  points_0 : u32,  stored_0 : ptr<function, f32>,  dissipated_1 : ptr<function, f32>) -> vec3<f32>
{
    var c_max_0 : f32 = 1.0f / max(f32(points_0), 10.0f) * m_red_0 / dt_1;
    var vn_0 : f32 = dot(rel_velocity_0, normal_1);
    var _S24 : f32 = k_3 * depth_0;
    var _S25 : f32 = min(2.0f * params_0.zeta_0 * sqrt(k_3 * m_red_0), c_max_0) * vn_0;
    var _S26 : f32 = _S24 - _S25;
    var _S27 : f32 = max(_S26, 0.0f);
    var vt_0 : vec3<f32> = rel_velocity_0 - normal_1 * vec3<f32>(vn_0);
    var vt_mag_0 : f32 = length(vt_0);
    var _S28 : f32 = friction_0 * _S27;
    var _S29 : f32 = min(_S28, min(c_max_0, _S28 / 0.00100000004749745f) * vt_mag_0);
    var ft_0 : vec3<f32>;
    if(vt_mag_0 > 0.0f)
    {
        ft_0 = (vec3<f32>(0) - vt_0) * vec3<f32>((_S29 / vt_mag_0));
    }
    else
    {
        ft_0 = vec3<f32>(0.0f);
    }
    (*stored_0) = 0.5f * k_3 * depth_0 * depth_0;
    var damping_power_0 : f32;
    if(_S26 > 0.0f)
    {
        damping_power_0 = _S25 * vn_0;
    }
    else
    {
        damping_power_0 = _S24 * max(vn_0, 0.0f);
    }
    (*dissipated_1) = (damping_power_0 + length(ft_0) * vt_mag_0) * dt_1;
    return normal_1 * vec3<f32>(_S27) + ft_0;
}

fn comp_add1_0( sum_0 : ptr<function, f32>,  err_0 : ptr<function, f32>,  x_1 : f32)
{
    var t_1 : f32 = (*sum_0) + x_1;
    if((abs((*sum_0))) >= (abs(x_1)))
    {
        (*err_0) = (*err_0) + ((*sum_0) - t_1 + x_1);
    }
    else
    {
        (*err_0) = (*err_0) + (x_1 - t_1 + (*sum_0));
    }
    (*sum_0) = t_1;
    return;
}

fn ground_contact_0( c_4 : u32,  account_0 : bool,  f_0 : ptr<function, vec3<f32>>,  t_2 : ptr<function, vec3<f32>>)
{
    var wp_0 : WorldPoint_0 = chunk_world_0(c_4);
    var above_0 : f32 = wp_0.hi_0.z - params_0.ground_hi_0 + (wp_0.lo_0.z - params_0.ground_lo_0) + wp_0.rel_0.z;
    if((above_0 - chunks_0[c_4].half_0.w) > 0.0f)
    {
        return;
    }
    var b_7 : Box_0 = chunk_box_0(c_4, vec3<f32>(0.0f));
    const _S30 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 1.0f);
    var _S31 : f32 = contact_stiffness_0(params_0.ground_modulus_0, b_7, chunks_0[c_4].cmat_0.x, b_7, _S30);
    var _S32 : u32 = sample_count_0(b_7);
    var s_1 : u32 = u32(0);
    var n_1 : u32 = u32(0);
    loop
    {
        if(s_1 < _S32)
        {
        }
        else
        {
            break;
        }
        if((above_0 + sample_point_0(b_7, s_1).z) < 0.0f)
        {
            n_1 = n_1 + u32(1);
        }
        s_1 = s_1 + u32(1);
    }
    if(n_1 == u32(0))
    {
        return;
    }
    sentinel_mark_0(u32(4));
    var vc_0 : vec3<f32>;
    var wc_0 : vec3<f32>;
    chunk_velocity_0(c_4, &(vc_0), &(wc_0));
    var ledger_1 : vec4<f32> = scratch_0[params_0.ledger_base_0 + params_0.pair_count_0 + c_4];
    s_1 = u32(0);
    loop
    {
        if(s_1 < _S32)
        {
        }
        else
        {
            break;
        }
        var p_0 : vec3<f32> = sample_point_0(b_7, s_1);
        var _S33 : f32 = above_0 + p_0.z;
        if(!(_S33 < 0.0f))
        {
            s_1 = s_1 + u32(1);
            continue;
        }
        var stored_1 : f32;
        var diss_0 : f32;
        var g_0 : vec3<f32> = penalty_force_0(_S31 / f32(max(n_1, u32(5))), chunks_0[c_4].center_0.w, params_0.ground_friction_0, - _S33, _S30, vc_0 + cross(wc_0, p_0), params_0.dt_0, n_1, &(stored_1), &(diss_0));
        (*f_0) = (*f_0) + g_0;
        (*t_2) = (*t_2) + cross(p_0, g_0);
        var _S34 : f32 = ledger_1[i32(1)];
        var _S35 : f32 = ledger_1[i32(2)];
        comp_add1_0(&(_S34), &(_S35), diss_0);
        ledger_1[i32(1)] = _S34;
        ledger_1[i32(2)] = _S35;
        s_1 = s_1 + u32(1);
    }
    if(account_0)
    {
        scratch_0[params_0.ledger_base_0 + params_0.pair_count_0 + c_4] = ledger_1;
    }
    return;
}

fn contact_segment_0( g_1 : u32)
{
    var _S36 : u32 = u32(3) * g_1;
    var _S37 : u32 = index_0[params_0.seg_index_0 + _S36];
    var begin_0 : u32 = index_0[params_0.seg_index_0 + _S36 + u32(1)];
    var _S38 : u32 = index_0[params_0.seg_index_0 + _S36 + u32(2)];
    var _S39 : vec3<f32> = vec3<f32>(0.0f);
    var f_1 : vec3<f32> = _S39;
    var t_3 : vec3<f32> = _S39;
    var e_0 : u32 = begin_0;
    loop
    {
        if(e_0 < _S38)
        {
        }
        else
        {
            break;
        }
        var entry_0 : u32 = index_0[e_0];
        if(entry_0 == u32(2147483648))
        {
            ground_contact_0(_S37, true, &(f_1), &(t_3));
            e_0 = e_0 + u32(1);
            continue;
        }
        var _S40 : u32 = u32(2) * entry_0;
        f_1 = f_1 + scratch_0[params_0.slot_base_0 + _S40].xyz;
        t_3 = t_3 + scratch_0[params_0.slot_base_0 + _S40 + u32(1)].xyz;
        e_0 = e_0 + u32(1);
    }
    var _S41 : u32 = u32(2) * g_1;
    scratch_0[params_0.seg_base_0 + _S41] = vec4<f32>(f_1, 0.0f);
    scratch_0[params_0.seg_base_0 + _S41 + u32(1)] = vec4<f32>(t_3, 0.0f);
    return;
}

fn time_since_0( origin_0 : vec4<f32>,  k_4 : u32,  dt_2 : f32) -> f32
{
    return params_0.t_hi_0 - origin_0.x + (params_0.t_lo_0 - origin_0.y) + f32(k_4) * dt_2;
}

fn table_eval_0( offset_0 : u32,  count_0 : u32,  tau_0 : f32) -> f32
{
    var first_0 : vec4<f32> = loads_0[offset_0];
    if(tau_0 <= (first_0.x))
    {
        return first_0.y;
    }
    var i_1 : u32 = u32(1);
    loop
    {
        if(i_1 < count_0)
        {
        }
        else
        {
            break;
        }
        var _S42 : u32 = offset_0 + i_1;
        var b_8 : vec4<f32> = loads_0[_S42];
        var _S43 : f32 = b_8.x;
        if(tau_0 <= _S43)
        {
            var a_3 : vec4<f32> = loads_0[_S42 - u32(1)];
            var _S44 : f32 = a_3.x;
            var _S45 : f32 = a_3.y;
            return _S45 + (tau_0 - _S44) / max(_S43 - _S44, 1.00000000317107685e-30f) * (b_8.y - _S45);
        }
        i_1 = i_1 + u32(1);
    }
    return loads_0[offset_0 + count_0 - u32(1)].y;
}

fn eval_function_0( term_0 : u32,  k_5 : u32,  dt_3 : f32,  shift_0 : f32) -> f32
{
    var _S46 : u32 = u32(5) * term_0;
    var info_2 : vec4<u32> = (bitcast<vec4<u32>>((loads_0[_S46])));
    var origin_1 : vec4<f32> = loads_0[_S46 + u32(3)];
    var p_1 : vec4<f32> = loads_0[_S46 + u32(4)];
    var kind_0 : u32 = info_2.z;
    if(kind_0 == u32(0))
    {
        return origin_1.z;
    }
    var tau_1 : f32 = time_since_0(origin_1, k_5, dt_3) + shift_0;
    var shape_1 : f32;
    if(kind_0 == u32(1))
    {
        if(tau_1 <= 0.0f)
        {
            shape_1 = 0.0f;
        }
        else
        {
            var _S47 : f32 = p_1.x;
            if(tau_1 >= _S47)
            {
                shape_1 = p_1.y;
            }
            else
            {
                shape_1 = p_1.y * tau_1 / _S47;
            }
        }
        return shape_1;
    }
    var _S48 : bool;
    if(kind_0 == u32(2))
    {
        if(tau_1 < 0.0f)
        {
            _S48 = true;
        }
        else
        {
            _S48 = tau_1 > (p_1.x);
        }
        if(_S48)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = p_1.y * sin(3.14159274101257324f * tau_1 / p_1.x);
        }
        return shape_1;
    }
    if(kind_0 == u32(3))
    {
        var sn_0 : f32 = tau_1 / p_1.y;
        if(sn_0 < 0.0f)
        {
            _S48 = true;
        }
        else
        {
            _S48 = sn_0 > 1.0f;
        }
        if(_S48)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = p_1.x * (1.0f - sn_0) * exp(- p_1.z * sn_0);
        }
        return shape_1;
    }
    if(kind_0 == u32(4))
    {
        return table_eval_0(info_2.w, (bitcast<u32>((p_1.x))), tau_1);
    }
    if(kind_0 == u32(5))
    {
        g_seen_0[u32(1)] = u32(1);
        if(tau_1 < 0.0f)
        {
            return 0.0f;
        }
        var sn_1 : f32 = tau_1 / p_1.x;
        if(sn_1 < 0.0f)
        {
            _S48 = true;
        }
        else
        {
            _S48 = sn_1 > 1.0f;
        }
        if(_S48)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = (1.0f - sn_1) * exp(- p_1.y * sn_1);
        }
        var clearing_0 : f32 = origin_1.w;
        var relax_0 : f32;
        if(clearing_0 > 0.0f)
        {
            relax_0 = max(1.0f - tau_1 / clearing_0, 0.0f);
        }
        else
        {
            relax_0 = 0.0f;
        }
        var _S49 : f32 = p_1.w;
        return (_S49 + (p_1.z - _S49) * relax_0) * shape_1;
    }
    if(kind_0 == u32(7))
    {
        g_seen_0[u32(2)] = u32(1);
        if(tau_1 < 0.0f)
        {
            return 0.0f;
        }
        var _S50 : f32 = p_1.y;
        if(tau_1 < _S50)
        {
            return p_1.x;
        }
        var s_2 : f32 = tau_1 - _S50;
        var _S51 : f32 = p_1.w;
        if(s_2 > _S51)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = p_1.z * sin(3.14159274101257324f * s_2 / _S51);
        }
        return shape_1;
    }
    var _S52 : f32 = p_1.x;
    if(_S52 <= 0.0f)
    {
        return 0.0f;
    }
    return clamp(1.0f - tau_1 / _S52, 0.0f, 1.0f);
}

fn chunk_loads_0( _S53 : u32,  _S54 : u32,  _S55 : Quat_0,  _S56 : u32,  _S57 : f32,  _S58 : bool,  _S59 : f32,  _S60 : ptr<function, vec3<f32>>,  _S61 : ptr<function, vec3<f32>>,  _S62 : ptr<function, vec3<f32>>,  _S63 : ptr<function, vec3<f32>>)
{
    var _S64 : bool;
    var _S65 : vec3<f32> = vec3<f32>(0.0f);
    (*_S60) = _S65;
    (*_S61) = _S65;
    (*_S62) = _S65;
    (*_S63) = _S65;
    var _S66 : vec4<u32> = chunks_0[_S54].load_range_0;
    var term_1 : u32 = chunks_0[_S54].load_range_0.x;
    loop
    {
        if(term_1 < (_S66.y))
        {
        }
        else
        {
            break;
        }
        g_seen_0[u32(0)] = u32(1);
        var _S67 : u32 = u32(5) * term_1;
        var _S68 : u32 = (bitcast<vec4<u32>>((loads_0[_S67]))).y;
        var replacement_0 : bool = _S68 == u32(2);
        var dir_1 : vec4<f32> = loads_0[_S67 + u32(1)];
        var arm_0 : vec4<f32> = loads_0[_S67 + u32(2)];
        var _S69 : f32;
        if(replacement_0)
        {
            _S69 = _S59;
        }
        else
        {
            _S69 = 0.0f;
        }
        var value_0 : f32 = eval_function_0(term_1, _S56, _S57, _S69);
        if(replacement_0)
        {
            var _S70 : vec3<f32> = vec3<f32>(value_0);
            (*_S62) = (*_S62) + dir_1.xyz * _S70;
            (*_S63) = (*_S63) + arm_0.xyz * _S70;
            term_1 = term_1 + u32(1);
            continue;
        }
        if(_S68 == u32(0))
        {
            _S64 = true;
        }
        else
        {
            _S64 = _S68 == u32(3);
        }
        var fw_0 : vec3<f32>;
        if(_S64)
        {
            fw_0 = dir_1.xyz * vec3<f32>(value_0);
        }
        else
        {
            fw_0 = rotate_0(_S55, dir_1.xyz) * vec3<f32>((- value_0 * dir_1.w));
        }
        var lever_0 : vec3<f32>;
        if(_S68 == u32(3))
        {
            lever_0 = arm_0.xyz - state_0[u32(4) * _S53].xyz;
        }
        else
        {
            lever_0 = arm_0.xyz;
        }
        (*_S60) = (*_S60) + fw_0;
        (*_S61) = (*_S61) + cross(rotate_0(_S55, lever_0), fw_0);
        term_1 = term_1 + u32(1);
    }
    if(_S58)
    {
        _S64 = (chunks_0[_S54].cinfo_0.z) != u32(0);
    }
    else
    {
        _S64 = false;
    }
    if(_S64)
    {
        var _S71 : vec4<u32> = chunks_0[_S54].cinfo_0;
        var g_2 : u32 = chunks_0[_S54].cinfo_0.x;
        loop
        {
            if(g_2 < (_S71.y))
            {
            }
            else
            {
                break;
            }
            var _S72 : u32 = u32(2) * g_2;
            (*_S60) = (*_S60) + scratch_0[params_0.seg_base_0 + _S72].xyz;
            (*_S61) = (*_S61) + scratch_0[params_0.seg_base_0 + _S72 + u32(1)].xyz;
            g_2 = g_2 + u32(1);
        }
    }
    return;
}

fn chunk_external_0( _S73 : u32,  _S74 : u32,  _S75 : Quat_0,  _S76 : u32,  _S77 : f32,  _S78 : bool,  _S79 : ptr<function, vec3<f32>>,  _S80 : ptr<function, vec3<f32>>)
{
    var rf_0 : vec3<f32>;
    var rm_0 : vec3<f32>;
    chunk_loads_0(_S73, _S74, _S75, _S76, _S77, _S78, 0.0f, &((*_S79)), &((*_S80)), &(rf_0), &(rm_0));
    return;
}

fn record_chunk_load_0( c_5 : u32,  f_2 : vec3<f32>,  t_4 : vec3<f32>)
{
    if((params_0.solve_mode_0) == u32(0))
    {
        return;
    }
    var _S81 : u32 = u32(2) * c_5;
    scratch_0[params_0.cload_base_0 + _S81] = vec4<f32>(f_2, 0.0f);
    scratch_0[params_0.cload_base_0 + _S81 + u32(1)] = vec4<f32>(t_4, 0.0f);
    scratch_0[params_0.cframe_base_0 + _S81] = vec4<f32>(scratch_0[params_0.cframe_base_0 + _S81].xyz + f_2, 0.0f);
    scratch_0[params_0.cframe_base_0 + _S81 + u32(1)] = vec4<f32>(scratch_0[params_0.cframe_base_0 + _S81 + u32(1)].xyz + t_4, 0.0f);
    return;
}

fn settled_chunk_load_0( c_6 : u32,  rot_1 : Quat_0,  k_6 : u32,  dt_4 : f32,  contact_0 : bool) -> f32
{
    var f_3 : vec3<f32>;
    var t_5 : vec3<f32>;
    chunk_external_0(c_6, c_6, rot_1, k_6, dt_4, contact_0, &(f_3), &(t_5));
    record_chunk_load_0(c_6, f_3, t_5);
    return length(f_3);
}

var<workgroup> g_red_a_0 : array<vec4<f32>, i32(256)>;

var<workgroup> g_red_b_0 : array<vec4<f32>, i32(256)>;

fn group_sum2_0( tid_1 : u32,  a_4 : ptr<function, vec4<f32>>,  b_9 : ptr<function, vec4<f32>>)
{
    if(tid_1 < u32(256))
    {
        g_red_a_0[tid_1] = (*a_4);
        g_red_b_0[tid_1] = (*b_9);
    }
    workgroupBarrier();
    var s_3 : u32 = u32(128);
    loop
    {
        if(s_3 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_1 < s_3)
        {
            var _S82 : u32 = tid_1 + s_3;
            g_red_a_0[tid_1] = g_red_a_0[tid_1] + g_red_a_0[_S82];
            g_red_b_0[tid_1] = g_red_b_0[tid_1] + g_red_b_0[_S82];
        }
        workgroupBarrier();
        s_3 = (s_3 >> (u32(1)));
    }
    (*a_4) = g_red_a_0[i32(0)];
    (*b_9) = g_red_b_0[i32(0)];
    workgroupBarrier();
    return;
}

fn group_sum3_0( tid_2 : u32,  a_5 : ptr<function, vec3<f32>>,  b_10 : ptr<function, vec3<f32>>)
{
    var x_2 : vec4<f32> = vec4<f32>((*a_5), 0.0f);
    var y_1 : vec4<f32> = vec4<f32>((*b_10), 0.0f);
    group_sum2_0(tid_2, &(x_2), &(y_1));
    (*a_5) = x_2.xyz;
    (*b_10) = y_1.xyz;
    return;
}

fn chunk_loads_1( _S83 : u32,  _S84 : u32,  _S85 : Quat_0,  _S86 : u32,  _S87 : f32,  _S88 : bool,  _S89 : f32,  _S90 : ptr<function, vec3<f32>>,  _S91 : ptr<function, vec3<f32>>,  _S92 : ptr<function, vec3<f32>>,  _S93 : ptr<function, vec3<f32>>)
{
    var _S94 : bool;
    var _S95 : vec3<f32> = vec3<f32>(0.0f);
    (*_S90) = _S95;
    (*_S91) = _S95;
    (*_S92) = _S95;
    (*_S93) = _S95;
    var _S96 : vec4<u32> = chunks_0[_S84].load_range_0;
    var term_2 : u32 = chunks_0[_S84].load_range_0.x;
    loop
    {
        if(term_2 < (_S96.y))
        {
        }
        else
        {
            break;
        }
        g_seen_0[u32(0)] = u32(1);
        var _S97 : u32 = u32(5) * term_2;
        var _S98 : u32 = (bitcast<vec4<u32>>((loads_0[_S97]))).y;
        var replacement_1 : bool = _S98 == u32(2);
        var dir_2 : vec4<f32> = loads_0[_S97 + u32(1)];
        var arm_1 : vec4<f32> = loads_0[_S97 + u32(2)];
        var _S99 : f32;
        if(replacement_1)
        {
            _S99 = _S89;
        }
        else
        {
            _S99 = 0.0f;
        }
        var value_1 : f32 = eval_function_0(term_2, _S86, _S87, _S99);
        if(replacement_1)
        {
            var _S100 : vec3<f32> = vec3<f32>(value_1);
            (*_S92) = (*_S92) + dir_2.xyz * _S100;
            (*_S93) = (*_S93) + arm_1.xyz * _S100;
            term_2 = term_2 + u32(1);
            continue;
        }
        if(_S98 == u32(0))
        {
            _S94 = true;
        }
        else
        {
            _S94 = _S98 == u32(3);
        }
        var fw_1 : vec3<f32>;
        if(_S94)
        {
            fw_1 = dir_2.xyz * vec3<f32>(value_1);
        }
        else
        {
            fw_1 = rotate_0(_S85, dir_2.xyz) * vec3<f32>((- value_1 * dir_2.w));
        }
        var lever_1 : vec3<f32>;
        if(_S98 == u32(3))
        {
            lever_1 = arm_1.xyz - state_0[u32(4) * _S83].xyz;
        }
        else
        {
            lever_1 = arm_1.xyz;
        }
        (*_S90) = (*_S90) + fw_1;
        (*_S91) = (*_S91) + cross(rotate_0(_S85, lever_1), fw_1);
        term_2 = term_2 + u32(1);
    }
    if(_S88)
    {
        _S94 = (chunks_0[_S84].cinfo_0.z) != u32(0);
    }
    else
    {
        _S94 = false;
    }
    if(_S94)
    {
        var _S101 : vec4<u32> = chunks_0[_S84].cinfo_0;
        var g_3 : u32 = chunks_0[_S84].cinfo_0.x;
        loop
        {
            if(g_3 < (_S101.y))
            {
            }
            else
            {
                break;
            }
            var _S102 : u32 = u32(2) * g_3;
            (*_S90) = (*_S90) + scratch_0[params_0.seg_base_0 + _S102].xyz;
            (*_S91) = (*_S91) + scratch_0[params_0.seg_base_0 + _S102 + u32(1)].xyz;
            g_3 = g_3 + u32(1);
        }
    }
    return;
}

fn net_load_0( c_7 : u32,  isl_2 : Island_0,  rg_1 : Rigid_0,  k_7 : u32,  dt_5 : f32,  contact_1 : bool,  f_4 : ptr<function, vec3<f32>>,  t_6 : ptr<function, vec3<f32>>)
{
    var fl_0 : vec3<f32>;
    var tl_0 : vec3<f32>;
    var rep_f_0 : vec3<f32>;
    var rep_m_0 : vec3<f32>;
    chunk_loads_1(c_7, c_7, rg_1.rot_0, k_7, dt_5, contact_1, 0.0f, &(fl_0), &(tl_0), &(rep_f_0), &(rep_m_0));
    var fc_0 : vec3<f32> = fl_0 + params_0.gravity_0.xyz * vec3<f32>(chunks_0[c_7].center_0.w);
    var _S103 : vec3<f32> = chunks_0[c_7].center_0.xyz;
    var _S104 : vec3<f32> = isl_2.com_0.xyz;
    var r_1 : vec3<f32> = rotate_0(rg_1.rot_0, _S103 + state_0[u32(4) * c_7].xyz - _S104);
    (*f_4) = (*f_4) + fc_0;
    (*t_6) = (*t_6) + (cross(r_1, fc_0) + tl_0);
    if((chunks_0[c_7].load_range_0.y) > (chunks_0[c_7].load_range_0.x))
    {
        var fw_2 : vec3<f32> = rotate_0(rg_1.rot_0, rep_f_0);
        (*f_4) = (*f_4) + fw_2;
        (*t_6) = (*t_6) + (cross(rotate_0(rg_1.rot_0, _S103 - _S104), fw_2) + rotate_0(rg_1.rot_0, rep_m_0));
    }
    return;
}

fn inverse_rotate_0( q_6 : Quat_0,  v_4 : vec3<f32>) -> vec3<f32>
{
    var c_8 : Quat_0;
    c_8.w_0 = q_6.w_0;
    c_8.x_0 = - q_6.x_0;
    c_8.y_0 = - q_6.y_0;
    c_8.z_0 = - q_6.z_0;
    return rotate_0(c_8, v_4);
}

fn rows_mul_0( r0_0 : vec4<f32>,  r1_0 : vec4<f32>,  r2_0 : vec4<f32>,  v_5 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(dot(r0_0.xyz, v_5), dot(r1_0.xyz, v_5), dot(r2_0.xyz, v_5));
}

fn world_mul_0( q_7 : Quat_0,  r0_1 : vec4<f32>,  r1_1 : vec4<f32>,  r2_1 : vec4<f32>,  v_6 : vec3<f32>) -> vec3<f32>
{
    return rotate_0(q_7, rows_mul_0(r0_1, r1_1, r2_1, inverse_rotate_0(q_7, v_6)));
}

fn rigid_acceleration_0( isl_3 : Island_0,  rg_2 : ptr<function, Rigid_0>,  f_5 : vec3<f32>,  t_7 : vec3<f32>)
{
    var iw_w_0 : vec3<f32> = world_mul_0((*rg_2).rot_0, isl_3.inertia0_0, isl_3.inertia1_0, isl_3.inertia2_0, (*rg_2).w_1);
    (*rg_2).a_0 = f_5 / vec3<f32>(isl_3.com_0.w);
    (*rg_2).alpha_0 = world_mul_0((*rg_2).rot_0, isl_3.inv0_0, isl_3.inv1_0, isl_3.inv2_0, t_7 - cross((*rg_2).w_1, iw_w_0));
    (*rg_2).torque_0 = t_7;
    return;
}

fn chunk_update_0( c_9 : u32,  isl_4 : Island_0,  rg_3 : Rigid_0,  dt_6 : f32,  rml_0 : bool,  step_0 : u32,  contact_2 : bool,  work_0 : ptr<function, f32>,  work_err_0 : ptr<function, f32>)
{
    var _S105 : vec3<f32> = vec3<f32>(0.0f);
    var _S106 : u32 = index_0[c_9];
    var peak_0 : f32 = 0.0f;
    var e_1 : u32 = _S106;
    var fi_0 : vec3<f32> = _S105;
    var mi_0 : vec3<f32> = _S105;
    loop
    {
        if(e_1 < index_0[c_9 + u32(1)])
        {
        }
        else
        {
            break;
        }
        var entry_1 : u32 = index_0[e_1];
        var _S107 : u32 = u32(3) * ((entry_1 >> (u32(1))));
        var fa_0 : vec4<f32> = scratch_0[_S107];
        if(((entry_1 & (u32(1)))) == u32(0))
        {
            var mi_1 : vec3<f32> = mi_0 + scratch_0[_S107 + u32(1)].xyz;
            fi_0 = fi_0 + fa_0.xyz;
            mi_0 = mi_1;
        }
        else
        {
            var mi_2 : vec3<f32> = mi_0 + scratch_0[_S107 + u32(2)].xyz;
            fi_0 = fi_0 + (vec3<f32>(0) - fa_0.xyz);
            mi_0 = mi_2;
        }
        var _S108 : f32 = max(peak_0, fa_0.w);
        var _S109 : u32 = e_1 + u32(1);
        peak_0 = _S108;
        e_1 = _S109;
    }
    var _S110 : u32 = u32(4) * c_9;
    var u_0 : vec3<f32> = state_0[_S110].xyz;
    var _S111 : u32 = _S110 + u32(1);
    var th_1 : vec3<f32> = state_0[_S111].xyz;
    var _S112 : u32 = _S110 + u32(2);
    var v_7 : vec3<f32> = state_0[_S112].xyz;
    var _S113 : u32 = _S110 + u32(3);
    var w_4 : vec3<f32> = state_0[_S113].xyz;
    var mass_0 : f32 = chunks_0[c_9].center_0.w;
    var _S114 : vec3<f32> = chunks_0[c_9].center_0.xyz;
    var _S115 : vec3<f32> = isl_4.com_0.xyz;
    var r_world_0 : vec3<f32> = rotate_0(rg_3.rot_0, _S114 + u_0 - _S115);
    var f_load_0 : vec3<f32>;
    var t_load_0 : vec3<f32>;
    var rep_f_1 : vec3<f32>;
    var rep_m_1 : vec3<f32>;
    chunk_loads_1(c_9, c_9, rg_3.rot_0, step_0, dt_6, contact_2, dt_6, &(f_load_0), &(t_load_0), &(rep_f_1), &(rep_m_1));
    record_chunk_load_0(c_9, f_load_0, t_load_0);
    var _S116 : vec3<f32> = vec3<f32>(mass_0);
    var f_world_0 : vec3<f32> = f_load_0 + params_0.gravity_0.xyz * _S116;
    var t_world_0 : vec3<f32> = t_load_0;
    var f_world_1 : vec3<f32>;
    var t_world_1 : vec3<f32>;
    if(rml_0)
    {
        var t_world_2 : vec3<f32> = t_world_0 - (world_mul_0(rg_3.rot_0, chunks_0[c_9].inertia0_1, chunks_0[c_9].inertia1_1, chunks_0[c_9].inertia2_1, rg_3.alpha_0) + cross(rg_3.w_1, world_mul_0(rg_3.rot_0, chunks_0[c_9].inertia0_1, chunks_0[c_9].inertia1_1, chunks_0[c_9].inertia2_1, rg_3.w_1)));
        f_world_1 = f_world_0 - (rg_3.a_0 + cross(rg_3.alpha_0, r_world_0) + cross(rg_3.w_1, cross(rg_3.w_1, r_world_0))) * _S116;
        t_world_1 = t_world_2;
    }
    else
    {
        f_world_1 = f_world_0;
        t_world_1 = t_world_0;
    }
    var f_ext_0 : vec3<f32> = inverse_rotate_0(rg_3.rot_0, f_world_1);
    var m_ext_0 : vec3<f32> = inverse_rotate_0(rg_3.rot_0, t_world_1);
    var f_ext_1 : vec3<f32>;
    var m_ext_1 : vec3<f32>;
    if(rml_0)
    {
        var wb_0 : vec3<f32> = inverse_rotate_0(rg_3.rot_0, rg_3.w_1);
        var i_w_0 : vec3<f32> = rows_mul_0(chunks_0[c_9].inertia0_1, chunks_0[c_9].inertia1_1, chunks_0[c_9].inertia2_1, w_4);
        var m_ext_2 : vec3<f32> = m_ext_0 - (cross(wb_0, i_w_0) + cross(w_4, rows_mul_0(chunks_0[c_9].inertia0_1, chunks_0[c_9].inertia1_1, chunks_0[c_9].inertia2_1, wb_0)) + cross(w_4, i_w_0));
        f_ext_1 = f_ext_0 - cross(wb_0, v_7) * vec3<f32>((2.0f * mass_0));
        m_ext_1 = m_ext_2;
    }
    else
    {
        f_ext_1 = f_ext_0;
        m_ext_1 = m_ext_0;
    }
    var f_6 : vec3<f32> = f_ext_1 + rep_f_1 + fi_0;
    var m_1 : vec3<f32> = m_ext_1 + rep_m_1 + mi_0;
    var support_0 : u32 = chunks_0[c_9].info_1.x;
    var _S117 : vec3<f32> = vec3<f32>(state_0[_S111].w, state_0[_S112].w, state_0[_S113].w);
    var reaction_0 : vec3<f32>;
    var u_1 : vec3<f32>;
    var th_2 : vec3<f32>;
    var v_8 : vec3<f32>;
    var w_5 : vec3<f32>;
    if(support_0 == u32(1))
    {
        reaction_0 = (vec3<f32>(0) - f_6);
        u_1 = u_0;
        th_2 = th_1;
        v_8 = _S105;
        w_5 = _S105;
    }
    else
    {
        var _S118 : vec4<f32> = chunks_0[c_9].scale_0;
        var w_6 : vec3<f32> = w_4 + rows_mul_0(chunks_0[c_9].inv0_1, chunks_0[c_9].inv1_1, chunks_0[c_9].inv2_1, m_1) * vec3<f32>((dt_6 * chunks_0[c_9].scale_0.z));
        var _S119 : vec3<f32> = vec3<f32>(dt_6);
        var th_3 : vec3<f32> = th_1 + w_6 * _S119;
        if(support_0 == u32(2))
        {
            reaction_0 = (vec3<f32>(0) - f_6);
            u_1 = u_0;
            th_2 = _S105;
        }
        else
        {
            var v_9 : vec3<f32> = v_7 + f_6 * vec3<f32>((dt_6 * _S118.y));
            var u_2 : vec3<f32> = u_0 + v_9 * _S119;
            reaction_0 = _S117;
            u_1 = u_2;
            th_2 = v_9;
        }
        var _S120 : vec3<f32> = th_2;
        th_2 = th_3;
        v_8 = _S120;
        w_5 = w_6;
    }
    state_0[_S110] = vec4<f32>(u_1, peak_0);
    state_0[_S111] = vec4<f32>(th_2, reaction_0.x);
    state_0[_S112] = vec4<f32>(v_8, reaction_0.y);
    state_0[_S113] = vec4<f32>(w_5, reaction_0.z);
    comp_add1_0(&((*work_0)), &((*work_err_0)), (dot(f_load_0, rg_3.vel_0 + rg_3.vel_err_0 + cross(rg_3.w_1, rotate_0(rg_3.rot_0, _S114 + u_1 - _S115)) + rotate_0(rg_3.rot_0, v_8)) + dot(t_load_0, rg_3.w_1 + rotate_0(rg_3.rot_0, w_5))) * dt_6);
    return;
}

fn comp_add_0( sum_1 : ptr<function, vec3<f32>>,  err_1 : ptr<function, vec3<f32>>,  x_3 : vec3<f32>)
{
    var t_8 : vec3<f32> = (*sum_1) + x_3;
    var _S121 : vec3<f32> = abs(x_3);
    (*err_1) = (*err_1) + (select(x_3, (*sum_1), (abs((*sum_1))) >= _S121) - t_8 + select((*sum_1), x_3, (abs((*sum_1))) >= _S121));
    (*sum_1) = t_8;
    return;
}

fn turn_difference_0( omega_0 : vec3<f32>,  dt_7 : f32,  v_10 : vec3<f32>) -> vec3<f32>
{
    var _S122 : f32 = length(omega_0);
    var angle_1 : f32 = _S122 * dt_7;
    if(angle_1 < 1.00000000317107685e-30f)
    {
        return vec3<f32>(0.0f);
    }
    var a_6 : vec3<f32> = omega_0 / vec3<f32>(_S122);
    var s_4 : f32 = sin(0.5f * angle_1);
    var av_0 : vec3<f32> = cross(a_6, v_10);
    return av_0 * vec3<f32>(sin(angle_1)) + cross(a_6, av_0) * vec3<f32>((2.0f * s_4 * s_4));
}

fn turn_minus_one_0( axis_2 : vec3<f32>,  angle_2 : f32) -> vec4<f32>
{
    var s_5 : f32 = sin(0.25f * angle_2);
    return vec4<f32>(axis_2 * vec3<f32>(sin(0.5f * angle_2)), -2.0f * s_5 * s_5);
}

fn quat_mul_0( a_7 : Quat_0,  o_0 : Quat_0) -> Quat_0
{
    var r_2 : Quat_0;
    r_2.w_0 = a_7.w_0 * o_0.w_0 - a_7.x_0 * o_0.x_0 - a_7.y_0 * o_0.y_0 - a_7.z_0 * o_0.z_0;
    r_2.x_0 = a_7.w_0 * o_0.x_0 + a_7.x_0 * o_0.w_0 + a_7.y_0 * o_0.z_0 - a_7.z_0 * o_0.y_0;
    r_2.y_0 = a_7.w_0 * o_0.y_0 - a_7.x_0 * o_0.z_0 + a_7.y_0 * o_0.w_0 + a_7.z_0 * o_0.x_0;
    r_2.z_0 = a_7.w_0 * o_0.z_0 + a_7.x_0 * o_0.y_0 - a_7.y_0 * o_0.x_0 + a_7.z_0 * o_0.w_0;
    return r_2;
}

fn quat_vec_0( q_8 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_8.x_0, q_8.y_0, q_8.z_0, q_8.w_0);
}

fn comp_add4_0( sum_2 : ptr<function, vec4<f32>>,  err_2 : ptr<function, vec4<f32>>,  x_4 : vec4<f32>)
{
    var t_9 : vec4<f32> = (*sum_2) + x_4;
    var _S123 : vec4<f32> = abs(x_4);
    (*err_2) = (*err_2) + (select(x_4, (*sum_2), (abs((*sum_2))) >= _S123) - t_9 + select((*sum_2), x_4, (abs((*sum_2))) >= _S123));
    (*sum_2) = t_9;
    return;
}

fn quat_accumulate_0( hi_1 : ptr<function, Quat_0>,  lo_1 : ptr<function, vec4<f32>>,  x_5 : vec4<f32>)
{
    var sum_3 : vec4<f32> = quat_vec_0((*hi_1));
    comp_add4_0(&(sum_3), &((*lo_1)), x_5);
    var t_10 : vec4<f32> = sum_3 + (*lo_1);
    (*lo_1) = (*lo_1) - (t_10 - sum_3);
    (*hi_1) = quat_of_0(t_10);
    return;
}

fn turn_left_0( hi_2 : ptr<function, Quat_0>,  lo_2 : ptr<function, vec4<f32>>,  omega_1 : vec3<f32>,  dt_8 : f32)
{
    var _S124 : f32 = length(omega_1);
    var angle_3 : f32 = _S124 * dt_8;
    if(angle_3 < 1.00000000317107685e-30f)
    {
        return;
    }
    var d_2 : vec4<f32> = turn_minus_one_0(omega_1 / vec3<f32>(_S124), angle_3);
    var dq_0 : Quat_0;
    dq_0.x_0 = d_2.x;
    dq_0.y_0 = d_2.y;
    dq_0.z_0 = d_2.z;
    dq_0.w_0 = d_2.w;
    quat_accumulate_0(&((*hi_2)), &((*lo_2)), quat_vec_0(quat_mul_0(dq_0, (*hi_2))));
    return;
}

fn integrate_rigid_0( isl_5 : Island_0,  rg_4 : ptr<function, Rigid_0>,  dt_9 : f32)
{
    var _S125 : vec3<f32> = vec3<f32>(dt_9);
    var _S126 : vec3<f32> = (*rg_4).torque_0 * _S125;
    var _S127 : vec3<f32> = (*rg_4).l_0;
    var _S128 : vec3<f32> = (*rg_4).l_err_0;
    comp_add_0(&(_S127), &(_S128), _S126);
    (*rg_4).l_0 = _S127;
    (*rg_4).l_err_0 = _S128;
    var l_1 : vec3<f32> = _S127 + _S128;
    var _S129 : vec3<f32> = (*rg_4).a_0 * _S125;
    var _S130 : vec3<f32> = (*rg_4).vel_0;
    var _S131 : vec3<f32> = (*rg_4).vel_err_0;
    comp_add_0(&(_S130), &(_S131), _S129);
    (*rg_4).vel_0 = _S130;
    (*rg_4).vel_err_0 = _S131;
    var w_mid_0 : vec3<f32> = world_mul_0((*rg_4).rot_0, isl_5.inv0_0, isl_5.inv1_0, isl_5.inv2_0, l_1);
    var delta_0 : vec3<f32> = (_S130 + _S131) * _S125 - turn_difference_0(w_mid_0, dt_9, rotate_0((*rg_4).rot_0, isl_5.com_0.xyz));
    var _S132 : vec3<f32> = (*rg_4).pos_0;
    var _S133 : vec3<f32> = (*rg_4).pos_err_0;
    comp_add_0(&(_S132), &(_S133), delta_0);
    (*rg_4).pos_0 = _S132;
    (*rg_4).pos_err_0 = _S133;
    var _S134 : Quat_0 = (*rg_4).rot_0;
    var _S135 : vec4<f32> = (*rg_4).rot_err_0;
    turn_left_0(&(_S134), &(_S135), w_mid_0, dt_9);
    (*rg_4).rot_0 = _S134;
    (*rg_4).rot_err_0 = _S135;
    (*rg_4).w_1 = world_mul_0(_S134, isl_5.inv0_0, isl_5.inv1_0, isl_5.inv2_0, l_1);
    return;
}

fn drift_moments_0( c_10 : u32,  tu_0 : ptr<function, vec3<f32>>,  pv_0 : ptr<function, vec3<f32>>)
{
    var _S136 : u32 = u32(4) * c_10;
    var _S137 : vec3<f32> = vec3<f32>((chunks_0[c_10].center_0.w * chunks_0[c_10].scale_0.x));
    (*tu_0) = (*tu_0) + state_0[_S136].xyz * _S137;
    (*pv_0) = (*pv_0) + state_0[_S136 + u32(2)].xyz * _S137;
    return;
}

fn drift_angular_0( c_11 : u32,  wcom_1 : vec3<f32>,  tr_0 : vec3<f32>,  dv_0 : vec3<f32>,  lu_0 : ptr<function, vec3<f32>>,  lv_0 : ptr<function, vec3<f32>>)
{
    var r_3 : vec3<f32> = chunks_0[c_11].center_0.xyz - wcom_1;
    var _S138 : u32 = u32(4) * c_11;
    var _S139 : vec3<f32> = vec3<f32>(chunks_0[c_11].center_0.w);
    var _S140 : vec4<f32> = chunks_0[c_11].inertia0_1;
    var _S141 : vec4<f32> = chunks_0[c_11].inertia1_1;
    var _S142 : vec4<f32> = chunks_0[c_11].inertia2_1;
    var _S143 : vec3<f32> = vec3<f32>(chunks_0[c_11].scale_0.x);
    (*lu_0) = (*lu_0) + (cross(r_3, state_0[_S138].xyz - tr_0) * _S139 + rows_mul_0(chunks_0[c_11].inertia0_1, chunks_0[c_11].inertia1_1, chunks_0[c_11].inertia2_1, state_0[_S138 + u32(1)].xyz)) * _S143;
    (*lv_0) = (*lv_0) + (cross(r_3, state_0[_S138 + u32(2)].xyz - dv_0) * _S139 + rows_mul_0(_S140, _S141, _S142, state_0[_S138 + u32(3)].xyz)) * _S143;
    return;
}

fn drift_apply_0( c_12 : u32,  wcom_2 : vec3<f32>,  tr_1 : vec3<f32>,  phi_0 : vec3<f32>,  dv_1 : vec3<f32>,  dw_0 : vec3<f32>)
{
    var r_4 : vec3<f32> = chunks_0[c_12].center_0.xyz - wcom_2;
    var _S144 : u32 = u32(4) * c_12;
    state_0[_S144] = vec4<f32>(state_0[_S144].xyz - (tr_1 + cross(phi_0, r_4)), state_0[_S144].w);
    var _S145 : u32 = _S144 + u32(1);
    state_0[_S145] = vec4<f32>(state_0[_S145].xyz - phi_0, state_0[_S145].w);
    var _S146 : u32 = _S144 + u32(2);
    state_0[_S146] = vec4<f32>(state_0[_S146].xyz - (dv_1 + cross(dw_0, r_4)), state_0[_S146].w);
    var _S147 : u32 = _S144 + u32(3);
    state_0[_S147] = vec4<f32>(state_0[_S147].xyz - dw_0, state_0[_S147].w);
    return;
}

fn turn_right_0( hi_3 : ptr<function, Quat_0>,  lo_3 : ptr<function, vec4<f32>>,  phi_1 : vec3<f32>)
{
    var angle_4 : f32 = length(phi_1);
    if(angle_4 < 1.00000000317107685e-30f)
    {
        return;
    }
    var d_3 : vec4<f32> = turn_minus_one_0(phi_1 / vec3<f32>(angle_4), angle_4);
    var dq_1 : Quat_0;
    dq_1.x_0 = d_3.x;
    dq_1.y_0 = d_3.y;
    dq_1.z_0 = d_3.z;
    dq_1.w_0 = d_3.w;
    quat_accumulate_0(&((*hi_3)), &((*lo_3)), quat_vec_0(quat_mul_0((*hi_3), dq_1)));
    return;
}

fn drift_rigid_0( isl_6 : Island_0,  rg_5 : ptr<function, Rigid_0>,  tr_2 : vec3<f32>,  phi_2 : vec3<f32>,  dv_2 : vec3<f32>,  dw_1 : vec3<f32>)
{
    var wcom_3 : vec3<f32> = isl_6.wcom_0.xyz;
    var rot_2 : Quat_0 = (*rg_5).rot_0;
    var _S148 : vec3<f32> = rotate_0((*rg_5).rot_0, tr_2 - turn_difference_0(phi_2, 1.0f, wcom_3));
    var _S149 : vec3<f32> = (*rg_5).pos_0;
    var _S150 : vec3<f32> = (*rg_5).pos_err_0;
    comp_add_0(&(_S149), &(_S150), _S148);
    (*rg_5).pos_0 = _S149;
    (*rg_5).pos_err_0 = _S150;
    var b_11 : vec3<f32> = inverse_rotate_0((*rg_5).rot_0, (*rg_5).w_1);
    var dl_turn_0 : vec3<f32> = rotate_0((*rg_5).rot_0, cross(phi_2, rows_mul_0(isl_6.inertia0_0, isl_6.inertia1_0, isl_6.inertia2_0, b_11)) - rows_mul_0(isl_6.inertia0_0, isl_6.inertia1_0, isl_6.inertia2_0, cross(phi_2, b_11)));
    var _S151 : Quat_0 = (*rg_5).rot_0;
    var _S152 : vec4<f32> = (*rg_5).rot_err_0;
    turn_right_0(&(_S151), &(_S152), phi_2);
    (*rg_5).rot_0 = _S151;
    (*rg_5).rot_err_0 = _S152;
    var _S153 : vec3<f32> = rotate_0(rot_2, dv_2 + cross(dw_1, isl_6.com_0.xyz - wcom_3));
    var _S154 : vec3<f32> = (*rg_5).vel_0;
    var _S155 : vec3<f32> = (*rg_5).vel_err_0;
    comp_add_0(&(_S154), &(_S155), _S153);
    (*rg_5).vel_0 = _S154;
    (*rg_5).vel_err_0 = _S155;
    var _S156 : vec3<f32> = rotate_0(rot_2, dw_1);
    (*rg_5).w_1 = (*rg_5).w_1 + _S156;
    var _S157 : vec3<f32> = dl_turn_0 + world_mul_0(_S151, isl_6.inertia0_0, isl_6.inertia1_0, isl_6.inertia2_0, _S156);
    var _S158 : vec3<f32> = (*rg_5).l_0;
    var _S159 : vec3<f32> = (*rg_5).l_err_0;
    comp_add_0(&(_S158), &(_S159), _S157);
    (*rg_5).l_0 = _S158;
    (*rg_5).l_err_0 = _S159;
    return;
}

fn write_probe_0( slot_0 : u32,  k_8 : u32,  value_2 : f32)
{
    var at_1 : u32 = params_0.probe_base_0 * u32(4) + slot_0 * params_0.probe_stride_0 + k_8;
    var v_11 : vec4<f32> = scratch_0[at_1 / u32(4)];
    v_11[at_1 % u32(4)] = value_2;
    scratch_0[at_1 / u32(4)] = v_11;
    return;
}

fn record_probes_0( isl_7 : Island_0,  rg_6 : Rigid_0,  k_9 : u32)
{
    var at_2 : u32 = isl_7.probes_0.x;
    loop
    {
        if(at_2 < (isl_7.probes_0.y))
        {
        }
        else
        {
            break;
        }
        var info_3 : vec4<u32> = (bitcast<vec4<u32>>((loads_0[at_2])));
        var a_8 : vec4<f32> = loads_0[at_2 + u32(1)];
        var b_12 : vec4<f32> = loads_0[at_2 + u32(2)];
        var c4_0 : vec4<f32> = loads_0[at_2 + u32(3)];
        var kind_1 : u32 = info_3.x;
        var i_2 : u32 = info_3.y;
        var value_3 : f32;
        if(kind_1 == u32(0))
        {
            value_3 = dot(rg_6.pos_0 - b_12.xyz + rotate_0(rg_6.rot_0, chunks_0[i_2].center_0.xyz) + (rg_6.pos_err_0 - c4_0.xyz + rotate_0(rg_6.rot_0, state_0[u32(4) * i_2].xyz + vec3<f32>(a_8.w, b_12.w, c4_0.w))), a_8.xyz);
        }
        else
        {
            if(kind_1 == u32(1))
            {
                var _S160 : u32 = u32(4) * i_2;
                value_3 = dot(rg_6.vel_0 + rg_6.vel_err_0 + cross(rg_6.w_1, rotate_0(rg_6.rot_0, chunks_0[i_2].center_0.xyz + state_0[_S160].xyz - isl_7.com_0.xyz)) + rotate_0(rg_6.rot_0, state_0[_S160 + u32(2)].xyz), a_8.xyz);
            }
            else
            {
                if(kind_1 == u32(2))
                {
                    var _S161 : u32 = u32(3) * i_2;
                    var f_7 : vec3<f32> = scratch_0[_S161].xyz;
                    var _S162 : bool = (info_3.z) == u32(0);
                    var mc_0 : vec3<f32>;
                    if(_S162)
                    {
                        mc_0 = scratch_0[_S161 + u32(1)].xyz;
                    }
                    else
                    {
                        mc_0 = scratch_0[_S161 + u32(2)].xyz;
                    }
                    var fc_1 : vec3<f32>;
                    if(_S162)
                    {
                        fc_1 = f_7;
                    }
                    else
                    {
                        fc_1 = (vec3<f32>(0) - f_7);
                    }
                    value_3 = dot(fc_1, a_8.xyz) + dot(mc_0, b_12.xyz);
                }
                else
                {
                    var _S163 : u32 = u32(4) * i_2;
                    value_3 = dot(rotate_0(rg_6.rot_0, vec3<f32>(state_0[_S163 + u32(1)].w, state_0[_S163 + u32(2)].w, state_0[_S163 + u32(3)].w)), a_8.xyz);
                }
            }
        }
        write_probe_0(info_3.w, k_9, value_3);
        at_2 = at_2 + u32(4);
    }
    return;
}

var<workgroup> g_label_narrow_0 : array<u32, i32(512)>;

var<workgroup> g_changed_narrow_0 : u32;

fn connected_0( st_0 : ptr<function, JointState_std430_0>,  has_rebar_0 : bool) -> bool
{
    var _S164 : bool;
    if(((*st_0).damage_0) < 1.0f)
    {
        _S164 = true;
    }
    else
    {
        if(has_rebar_0)
        {
            _S164 = ((*st_0).rebar_broken_0) == 0.0f;
        }
        else
        {
            _S164 = false;
        }
    }
    return _S164;
}

fn island_connected_narrow_0( tid_3 : u32,  ct_0 : u32,  isl_8 : Island_0) -> bool
{
    var _S165 : u32 = isl_8.range_0.x;
    var n_2 : u32 = isl_8.range_0.y - _S165;
    var converged_0 : bool;
    if(n_2 > u32(512))
    {
        converged_0 = true;
    }
    else
    {
        converged_0 = (params_0.device_connect_0) == u32(0);
    }
    if(converged_0)
    {
        return false;
    }
    var c_13 : u32 = ct_0;
    loop
    {
        if(c_13 < n_2)
        {
        }
        else
        {
            break;
        }
        g_label_narrow_0[c_13] = c_13;
        c_13 = c_13 + u32(256);
    }
    workgroupBarrier();
    converged_0 = false;
    var pass_0 : u32 = u32(0);
    loop
    {
        if(pass_0 <= n_2)
        {
        }
        else
        {
            break;
        }
        if(tid_3 == u32(0))
        {
            g_changed_narrow_0 = u32(0);
        }
        workgroupBarrier();
        c_13 = ct_0;
        loop
        {
            if(c_13 < n_2)
            {
            }
            else
            {
                break;
            }
            var gc_0 : u32 = _S165 + c_13;
            var _S166 : u32 = index_0[gc_0];
            var l_2 : u32 = g_label_narrow_0[c_13];
            var e_2 : u32 = _S166;
            loop
            {
                if(e_2 < index_0[gc_0 + u32(1)])
                {
                }
                else
                {
                    break;
                }
                var entry_2 : u32 = index_0[e_2];
                var bond_0 : u32 = (entry_2 >> (u32(1)));
                var _S167 : JointState_std430_0 = bond_dyn_0[bond_0].js_0;
                var _S168 : bool = connected_0(&(_S167), (bonds_0[bond_0].law_0.stiff1_0.w) != 0.0f);
                if(!_S168)
                {
                    e_2 = e_2 + u32(1);
                    continue;
                }
                var other_0 : u32;
                if(((entry_2 & (u32(1)))) == u32(0))
                {
                    other_0 = bonds_0[bond_0].law_0.ids_0.z;
                }
                else
                {
                    other_0 = bonds_0[bond_0].law_0.ids_0.y;
                }
                l_2 = min(l_2, g_label_narrow_0[other_0 - _S165]);
                e_2 = e_2 + u32(1);
            }
            var _S169 : u32 = min(l_2, g_label_narrow_0[l_2]);
            if(_S169 < (g_label_narrow_0[c_13]))
            {
                g_label_narrow_0[c_13] = _S169;
                g_changed_narrow_0 = u32(1);
            }
            c_13 = c_13 + u32(256);
        }
        workgroupBarrier();
        var converged_1 : bool = g_changed_narrow_0 == u32(0);
        workgroupBarrier();
        if(converged_1)
        {
            converged_0 = converged_1;
            break;
        }
        var _S170 : u32 = pass_0 + u32(1);
        converged_0 = converged_1;
        pass_0 = _S170;
    }
    if(!converged_0)
    {
        return false;
    }
    if(tid_3 == u32(0))
    {
        g_changed_narrow_0 = u32(0);
    }
    workgroupBarrier();
    c_13 = ct_0;
    loop
    {
        if(c_13 < n_2)
        {
        }
        else
        {
            break;
        }
        if((g_label_narrow_0[c_13]) != u32(0))
        {
            g_changed_narrow_0 = u32(1);
        }
        c_13 = c_13 + u32(256);
    }
    workgroupBarrier();
    return g_changed_narrow_0 == u32(0);
}

fn contact_split_at_0( at_3 : u32)
{
    var previous_0 : u32 = islands_0[params_0.halt_index_0].info_0.y;
    var _S171 : u32;
    if(previous_0 == u32(0))
    {
        _S171 = at_3;
    }
    else
    {
        _S171 = min(previous_0, at_3);
    }
    islands_0[params_0.halt_index_0].info_0[i32(1)] = _S171;
    return;
}

fn comp_add1_1( sum_4 : ptr<function, f32>,  err_3 : ptr<function, f32>,  x_6 : f32)
{
    var t_11 : f32 = (*sum_4) + x_6;
    if((abs((*sum_4))) >= (abs(x_6)))
    {
        (*err_3) = (*err_3) + ((*sum_4) - t_11 + x_6);
    }
    else
    {
        (*err_3) = (*err_3) + (x_6 - t_11 + (*sum_4));
    }
    (*sum_4) = t_11;
    return;
}

fn seen_bits_0() -> u32
{
    var _S172 : u32;
    if((g_seen_0[u32(0)]) != u32(0))
    {
        _S172 = u32(32768);
    }
    else
    {
        _S172 = u32(0);
    }
    var _S173 : u32;
    if((g_seen_0[u32(1)]) != u32(0))
    {
        _S173 = u32(65536);
    }
    else
    {
        _S173 = u32(0);
    }
    var _S174 : u32 = (_S172 | (_S173));
    if((g_seen_0[u32(2)]) != u32(0))
    {
        _S172 = u32(131072);
    }
    else
    {
        _S172 = u32(0);
    }
    var _S175 : u32 = (_S174 | (_S172));
    if((g_seen_0[u32(3)]) != u32(0))
    {
        _S172 = u32(32);
    }
    else
    {
        _S172 = u32(0);
    }
    var _S176 : u32 = (_S175 | (_S172));
    if((g_seen_0[u32(4)]) != u32(0))
    {
        _S172 = u32(64);
    }
    else
    {
        _S172 = u32(0);
    }
    return (_S176 | (_S172));
}

fn travel_stop_0( substep_0 : u32,  fused_0 : bool)
{
    if(fused_0)
    {
        islands_0[params_0.halt_index_0].range_0[i32(0)] = substep_0;
        return;
    }
    islands_0[params_0.halt_index_0].info_0[i32(2)] = ((islands_0[params_0.halt_index_0].info_0.z) | (u32(1)));
    sentinel_mark_0(u32(1048576));
    return;
}

fn travel_check_0( c_14 : u32,  fused_1 : bool)
{
    var _S177 : vec4<u32> = chunks_0[c_14].cinfo_0;
    if((chunks_0[c_14].cinfo_0.z) == u32(0))
    {
        return;
    }
    var _S178 : u32 = u32(4) * c_14;
    var at_4 : u32 = params_0.cplan_base_0 + _S178;
    var start_hi_0 : vec4<f32> = scratch_0[at_4];
    var start_rot_0 : vec4<f32> = scratch_0[at_4 + u32(2)];
    var wp_1 : WorldPoint_0 = chunk_world_0(c_14);
    var moved_0 : vec3<f32> = wp_1.hi_0 - scratch_0[at_4].xyz + (wp_1.lo_0 - scratch_0[at_4 + u32(1)].xyz) + wp_1.rel_0;
    var _S179 : u32 = chunks_0[c_14].info_1.y;
    var _S180 : vec3<f32> = vec3<f32>((params_0.t_hi_0 - scratch_0[at_4 + u32(3)].w + (params_0.t_lo_0 - scratch_0[at_4 + u32(1)].w) + f32(islands_0[_S179].info_0.w - params_0.step_start_0) * params_0.dt_0));
    var _S181 : vec3<f32> = scratch_0[at_4 + u32(3)].xyz * _S180;
    var _S182 : u32 = _S177.w;
    var _S183 : vec3<f32>;
    if(((_S182 & (u32(2)))) != u32(0))
    {
        _S183 = vec3<f32>(0.5f) * params_0.gravity_0.xyz * _S180 * _S180;
    }
    else
    {
        _S183 = vec3<f32>(0.0f);
    }
    var travel_0 : f32 = length(moved_0 - (_S181 + _S183));
    var travel_1 : f32;
    if(((_S182 & (u32(1)))) != u32(0))
    {
        var th_4 : vec3<f32> = state_0[_S178 + u32(1)].xyz;
        var turn_0 : Quat_0 = quat_mul_0(quat_of_0(vec4<f32>((vec3<f32>(0) - start_rot_0.xyz), start_rot_0.w)), quat_mul_0(quat_of_0(islands_0[_S179].rotation_0), from_axis_angle_0(th_4, length(th_4))));
        travel_1 = travel_0 + 2.0f * atan2(length(vec3<f32>(turn_0.x_0, turn_0.y_0, turn_0.z_0)), abs(turn_0.w_0)) * chunks_0[c_14].half_0.w;
    }
    else
    {
        travel_1 = travel_0;
    }
    if(travel_1 > (start_hi_0.w))
    {
        travel_stop_0(islands_0[_S179].info_0.w + u32(1), fused_1);
    }
    return;
}

fn island_frame_at_0( gi_0 : u32,  tid_4 : u32)
{
    var c_15 : u32;
    var woke_0 : bool;
    var ct_1 : u32;
    if(tid_4 < u32(256))
    {
        ct_1 = tid_4;
    }
    else
    {
        ct_1 = u32(1073741824);
    }
    var isl_9 : Island_0;
    isl_9.range_0 = islands_0[gi_0].range_0;
    isl_9.info_0 = islands_0[gi_0].info_0;
    isl_9.com_0 = islands_0[gi_0].com_0;
    isl_9.inertia0_0 = islands_0[gi_0].inertia0_0;
    isl_9.inertia1_0 = islands_0[gi_0].inertia1_0;
    isl_9.inertia2_0 = islands_0[gi_0].inertia2_0;
    isl_9.inv0_0 = islands_0[gi_0].inv0_0;
    isl_9.inv1_0 = islands_0[gi_0].inv1_0;
    isl_9.inv2_0 = islands_0[gi_0].inv2_0;
    isl_9.wcom_0 = islands_0[gi_0].wcom_0;
    isl_9.winv0_0 = islands_0[gi_0].winv0_0;
    isl_9.winv1_0 = islands_0[gi_0].winv1_0;
    isl_9.winv2_0 = islands_0[gi_0].winv2_0;
    isl_9.rotation_0 = islands_0[gi_0].rotation_0;
    isl_9.position_0 = islands_0[gi_0].position_0;
    isl_9.position_err_0 = islands_0[gi_0].position_err_0;
    isl_9.velocity_0 = islands_0[gi_0].velocity_0;
    isl_9.velocity_err_0 = islands_0[gi_0].velocity_err_0;
    isl_9.angular_velocity_0 = islands_0[gi_0].angular_velocity_0;
    isl_9.done_0 = islands_0[gi_0].done_0;
    isl_9.coverage_0 = islands_0[gi_0].coverage_0;
    isl_9.wide_groups_0 = islands_0[gi_0].wide_groups_0;
    isl_9.done_pad_0 = islands_0[gi_0].done_pad_0;
    isl_9.probes_0 = islands_0[gi_0].probes_0;
    isl_9.energy_0 = islands_0[gi_0].energy_0;
    isl_9.rotation_err_0 = islands_0[gi_0].rotation_err_0;
    isl_9.momentum_0 = islands_0[gi_0].momentum_0;
    isl_9.momentum_err_0 = islands_0[gi_0].momentum_err_0;
    var driven_0 : bool = (((isl_9.info_0.x) & (u32(2)))) != u32(0);
    var _S184 : bool = !((((isl_9.info_0.x) & (u32(1)))) != u32(0));
    var _S185 : bool;
    if(_S184)
    {
        _S185 = !driven_0;
    }
    else
    {
        _S185 = false;
    }
    var contact_island_0 : bool = (((isl_9.info_0.x) & (u32(4)))) != u32(0);
    var _S186 : bool = (((isl_9.info_0.x) & (u32(16)))) != u32(0);
    seen_clear_0(tid_4);
    var _S187 : bool = tid_4 == u32(0);
    var cov_0 : u32;
    var settled_0 : bool;
    if(_S187)
    {
        if(contact_island_0 != ((params_0.contact_mode_0) == u32(1)))
        {
            settled_0 = true;
        }
        else
        {
            settled_0 = (((isl_9.info_0.x) & (u32(8)))) != u32(0);
        }
        if(settled_0)
        {
            cov_0 = u32(0);
        }
        else
        {
            cov_0 = u32(1);
        }
        if(contact_island_0)
        {
            settled_0 = contact_stopped_0(isl_9);
        }
        else
        {
            settled_0 = false;
        }
        if(settled_0)
        {
            cov_0 = u32(0);
        }
        g_run_0 = cov_0;
        g_halt_0 = u32(0);
        if(contact_island_0)
        {
            stop_seen_0(params_0.halt_index_0, isl_9.info_0.w + u32(1));
            var _S188 : u32 = islands_0[params_0.halt_index_0].info_0.y;
            if(_S188 != u32(0))
            {
                settled_0 = _S188 <= (isl_9.info_0.w);
            }
            else
            {
                settled_0 = false;
            }
            var _S189 : i32;
            if(settled_0)
            {
                _S189 = i32(0);
            }
            else
            {
                _S189 = i32(1);
            }
            g_sums_0 = u32(_S189);
        }
    }
    workgroupBarrier();
    if((((isl_9.info_0.z) & (u32(1)))) != u32(0))
    {
        settled_0 = true;
    }
    else
    {
        settled_0 = g_run_0 == u32(0);
    }
    if(settled_0)
    {
        cov_0 = u32(0);
    }
    else
    {
        cov_0 = min(isl_9.info_0.y, params_0.max_steps_0);
    }
    var _S190 : f32 = params_0.dt_0;
    var _S191 : bool = (params_0.rigid_motion_loads_0) != u32(0);
    var rg_7 : Rigid_0 = rigid_of_0(isl_9);
    var work_1 : f32 = 0.0f;
    var work_err_1 : f32 = 0.0f;
    settled_0 = _S186;
    var done_1 : u32 = u32(0);
    var woke_1 : bool = false;
    var resumed_0 : bool = false;
    loop
    {
        if(done_1 < cov_0)
        {
        }
        else
        {
            woke_0 = woke_1;
            break;
        }
        var k_10 : u32 = isl_9.info_0.w + done_1 + u32(1) - u32(1) - params_0.step_start_0;
        var quiet_0 : bool = (((isl_9.info_0.x) & (u32(32)))) != u32(0);
        var _S192 : bool;
        if(contact_island_0)
        {
            _S192 = g_sums_0 != u32(0);
        }
        else
        {
            _S192 = false;
        }
        if(_S192)
        {
            var _S193 : u32 = chunks_0[isl_9.range_0.y - u32(1)].cinfo_0.y;
            c_15 = chunks_0[isl_9.range_0.x].cinfo_0.x + ct_1;
            loop
            {
                if(c_15 < _S193)
                {
                }
                else
                {
                    break;
                }
                contact_segment_0(c_15);
                c_15 = c_15 + u32(256);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
        }
        var _S194 : bool;
        if(settled_0)
        {
            _S194 = !quiet_0;
        }
        else
        {
            _S194 = false;
        }
        var _S195 : bool;
        var explicit_step_0 : bool;
        if(_S194)
        {
            var _S196 : vec3<f32> = vec3<f32>(0.0f);
            var norm_0 : vec3<f32> = _S196;
            var unused0_0 : vec3<f32> = _S196;
            c_15 = isl_9.range_0.x + ct_1;
            loop
            {
                if(c_15 < (isl_9.range_0.y))
                {
                }
                else
                {
                    break;
                }
                var _S197 : f32 = settled_chunk_load_0(c_15, rg_7.rot_0, k_10, _S190, contact_island_0);
                norm_0[i32(0)] = norm_0[i32(0)] + _S197;
                c_15 = c_15 + u32(256);
            }
            group_sum3_0(tid_4, &(norm_0), &(unused0_0));
            if((params_0.solve_mode_0) == u32(1))
            {
                _S195 = (abs(norm_0.x - isl_9.energy_0.z)) > (isl_9.energy_0.w);
            }
            else
            {
                _S195 = false;
            }
            if(_S195)
            {
                explicit_step_0 = false;
                woke_0 = true;
            }
            else
            {
                explicit_step_0 = settled_0;
                woke_0 = woke_1;
            }
            settled_0 = explicit_step_0;
        }
        else
        {
            woke_0 = woke_1;
        }
        if(_S184)
        {
            _S195 = !quiet_0;
        }
        else
        {
            _S195 = false;
        }
        if(_S195)
        {
            var _S198 : vec3<f32> = vec3<f32>(0.0f);
            var f_8 : vec3<f32> = _S198;
            var t_12 : vec3<f32> = _S198;
            c_15 = isl_9.range_0.x + ct_1;
            loop
            {
                if(c_15 < (isl_9.range_0.y))
                {
                }
                else
                {
                    break;
                }
                net_load_0(c_15, isl_9, rg_7, k_10, _S190, contact_island_0, &(f_8), &(t_12));
                c_15 = c_15 + u32(256);
            }
            group_sum3_0(tid_4, &(f_8), &(t_12));
            rigid_acceleration_0(isl_9, &(rg_7), f_8, t_12);
        }
        if(!settled_0)
        {
            explicit_step_0 = !quiet_0;
        }
        else
        {
            explicit_step_0 = false;
        }
        var _S199 : bool;
        if(explicit_step_0)
        {
            if(_S187)
            {
                _S199 = (((islands_0[gi_0].info_0.z) & (u32(2)))) != u32(0);
            }
            else
            {
                _S199 = false;
            }
            if(_S199)
            {
                g_halt_0 = u32(1);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
            c_15 = isl_9.range_0.x + ct_1;
            loop
            {
                if(c_15 < (isl_9.range_0.y))
                {
                }
                else
                {
                    break;
                }
                chunk_update_0(c_15, isl_9, rg_7, _S190, _S191, k_10, contact_island_0, &(work_1), &(work_err_1));
                c_15 = c_15 + u32(256);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
        }
        if(_S185)
        {
            _S199 = !quiet_0;
        }
        else
        {
            _S199 = false;
        }
        if(_S199)
        {
            integrate_rigid_0(isl_9, &(rg_7), _S190);
        }
        var _S200 : bool;
        if(_S184)
        {
            _S200 = explicit_step_0;
        }
        else
        {
            _S200 = false;
        }
        if(_S200)
        {
            var _S201 : vec3<f32> = isl_9.wcom_0.xyz;
            var _S202 : vec3<f32> = vec3<f32>(0.0f);
            var tu_1 : vec3<f32> = _S202;
            var pv_1 : vec3<f32> = _S202;
            c_15 = isl_9.range_0.x + ct_1;
            loop
            {
                if(c_15 < (isl_9.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_moments_0(c_15, &(tu_1), &(pv_1));
                c_15 = c_15 + u32(256);
            }
            group_sum3_0(tid_4, &(tu_1), &(pv_1));
            var tr_3 : vec3<f32> = tu_1 / vec3<f32>(isl_9.wcom_0.w);
            var dv_3 : vec3<f32> = pv_1 / vec3<f32>(isl_9.wcom_0.w);
            var lu_1 : vec3<f32> = _S202;
            var lv_1 : vec3<f32> = _S202;
            var c_16 : u32 = isl_9.range_0.x + ct_1;
            loop
            {
                if(c_16 < (isl_9.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_angular_0(c_16, _S201, tr_3, dv_3, &(lu_1), &(lv_1));
                c_16 = c_16 + u32(256);
            }
            group_sum3_0(tid_4, &(lu_1), &(lv_1));
            var phi_3 : vec3<f32> = rows_mul_0(isl_9.winv0_0, isl_9.winv1_0, isl_9.winv2_0, lu_1);
            var dw_2 : vec3<f32> = rows_mul_0(isl_9.winv0_0, isl_9.winv1_0, isl_9.winv2_0, lv_1);
            var c_17 : u32 = isl_9.range_0.x + ct_1;
            loop
            {
                if(c_17 < (isl_9.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_17, _S201, tr_3, phi_3, dv_3, dw_2);
                c_17 = c_17 + u32(256);
            }
            if(!driven_0)
            {
                drift_rigid_0(isl_9, &(rg_7), tr_3, phi_3, dv_3, dw_2);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
        }
        var _S203 : bool;
        if(_S187)
        {
            _S203 = (isl_9.probes_0.y) > (isl_9.probes_0.x);
        }
        else
        {
            _S203 = false;
        }
        if(_S203)
        {
            record_probes_0(isl_9, rg_7, k_10);
        }
        var _S204 : u32 = done_1 + u32(1);
        var _S205 : bool;
        if(g_halt_0 != u32(0))
        {
            var _S206 : bool = island_connected_narrow_0(tid_4, ct_1, isl_9);
            _S205 = _S206;
        }
        else
        {
            _S205 = false;
        }
        var resumed_1 : bool;
        if(_S205)
        {
            if(_S187)
            {
                g_halt_0 = u32(0);
            }
            workgroupBarrier();
            resumed_1 = true;
        }
        else
        {
            resumed_1 = resumed_0;
        }
        if(g_halt_0 != u32(0))
        {
            done_1 = _S204;
            resumed_0 = resumed_1;
            break;
        }
        done_1 = _S204;
        woke_1 = woke_0;
        resumed_0 = resumed_1;
    }
    var wsum_0 : vec3<f32> = vec3<f32>(work_1, work_err_1, 0.0f);
    var unused_0 : vec3<f32> = vec3<f32>(0.0f);
    group_sum3_0(tid_4, &(wsum_0), &(unused_0));
    if(_S187)
    {
        isl_9.rotation_0 = quat_vec_0(rg_7.rot_0);
        isl_9.rotation_err_0 = rg_7.rot_err_0;
        isl_9.position_0 = vec4<f32>(rg_7.pos_0, 0.0f);
        isl_9.position_err_0 = vec4<f32>(rg_7.pos_err_0, 0.0f);
        isl_9.velocity_0 = vec4<f32>(rg_7.vel_0, 0.0f);
        isl_9.velocity_err_0 = vec4<f32>(rg_7.vel_err_0, 0.0f);
        isl_9.angular_velocity_0 = vec4<f32>(rg_7.w_1, 0.0f);
        isl_9.momentum_0 = vec4<f32>(rg_7.l_0, 0.0f);
        isl_9.momentum_err_0 = vec4<f32>(rg_7.l_err_0, 0.0f);
        isl_9.done_0 = done_1;
        isl_9.info_0[i32(1)] = isl_9.info_0[i32(1)] - done_1;
        if(g_halt_0 != u32(0))
        {
            _S185 = contact_island_0;
        }
        else
        {
            _S185 = false;
        }
        if(_S185)
        {
            contact_split_at_0(isl_9.info_0.w + done_1);
        }
        var _S207 : f32 = wsum_0.x;
        var _S208 : f32 = isl_9.energy_0[i32(0)];
        var _S209 : f32 = isl_9.energy_0[i32(1)];
        comp_add1_1(&(_S208), &(_S209), _S207);
        isl_9.energy_0[i32(0)] = _S208;
        isl_9.energy_0[i32(1)] = _S209 + wsum_0.y;
        isl_9.info_0[i32(3)] = isl_9.info_0[i32(3)] + done_1;
        if((((isl_9.info_0.x) & (u32(8)))) == u32(0))
        {
            isl_9.info_0[i32(2)] = ((isl_9.info_0[i32(2)]) & (u32(4294967293)));
        }
        if(g_halt_0 != u32(0))
        {
            isl_9.info_0[i32(2)] = ((isl_9.info_0[i32(2)]) | (u32(1)));
        }
        var _S210 : u32 = seen_bits_0();
        if(g_halt_0 != u32(0))
        {
            cov_0 = u32(128);
        }
        else
        {
            cov_0 = u32(0);
        }
        var _S211 : u32 = (_S210 | (cov_0));
        if(woke_0)
        {
            cov_0 = u32(2048);
        }
        else
        {
            cov_0 = u32(0);
        }
        var _S212 : u32 = (_S211 | (cov_0));
        if(resumed_0)
        {
            cov_0 = u32(2097280);
        }
        else
        {
            cov_0 = u32(0);
        }
        var cov_1 : u32 = (_S212 | (cov_0));
        if(done_1 > u32(0))
        {
            var quiet_1 : bool = (((isl_9.info_0.x) & (u32(32)))) != u32(0);
            if(quiet_1)
            {
                cov_0 = u32(4096);
            }
            else
            {
                cov_0 = u32(0);
            }
            var cov_2 : u32 = (cov_1 | (cov_0));
            if((((isl_9.info_0.x) & (u32(16)))) != u32(0))
            {
                _S185 = !quiet_1;
            }
            else
            {
                _S185 = false;
            }
            if(_S185)
            {
                cov_0 = u32(512);
            }
            else
            {
                cov_0 = u32(0);
            }
            var cov_3 : u32 = (cov_2 | (cov_0));
            if((params_0.contact_mode_0) == u32(1))
            {
                cov_0 = u32(524288);
            }
            else
            {
                cov_0 = u32(0);
            }
            var cov_4 : u32 = (cov_3 | (cov_0));
            if((isl_9.probes_0.y) > (isl_9.probes_0.x))
            {
                cov_0 = u32(262144);
            }
            else
            {
                cov_0 = u32(0);
            }
            cov_0 = (cov_4 | (cov_0));
        }
        else
        {
            cov_0 = cov_1;
        }
        isl_9.coverage_0 = ((isl_9.coverage_0) | (cov_0));
        if(woke_0)
        {
            isl_9.info_0[i32(0)] = ((isl_9.info_0[i32(0)]) & (u32(4294967279)));
            isl_9.info_0[i32(2)] = ((isl_9.info_0[i32(2)]) | (u32(4)));
        }
        islands_0[gi_0].range_0 = isl_9.range_0;
        islands_0[gi_0].info_0 = isl_9.info_0;
        islands_0[gi_0].com_0 = isl_9.com_0;
        islands_0[gi_0].inertia0_0 = isl_9.inertia0_0;
        islands_0[gi_0].inertia1_0 = isl_9.inertia1_0;
        islands_0[gi_0].inertia2_0 = isl_9.inertia2_0;
        islands_0[gi_0].inv0_0 = isl_9.inv0_0;
        islands_0[gi_0].inv1_0 = isl_9.inv1_0;
        islands_0[gi_0].inv2_0 = isl_9.inv2_0;
        islands_0[gi_0].wcom_0 = isl_9.wcom_0;
        islands_0[gi_0].winv0_0 = isl_9.winv0_0;
        islands_0[gi_0].winv1_0 = isl_9.winv1_0;
        islands_0[gi_0].winv2_0 = isl_9.winv2_0;
        islands_0[gi_0].rotation_0 = isl_9.rotation_0;
        islands_0[gi_0].position_0 = isl_9.position_0;
        islands_0[gi_0].position_err_0 = isl_9.position_err_0;
        islands_0[gi_0].velocity_0 = isl_9.velocity_0;
        islands_0[gi_0].velocity_err_0 = isl_9.velocity_err_0;
        islands_0[gi_0].angular_velocity_0 = isl_9.angular_velocity_0;
        islands_0[gi_0].done_0 = isl_9.done_0;
        islands_0[gi_0].coverage_0 = isl_9.coverage_0;
        islands_0[gi_0].wide_groups_0 = isl_9.wide_groups_0;
        islands_0[gi_0].done_pad_0 = isl_9.done_pad_0;
        islands_0[gi_0].probes_0 = isl_9.probes_0;
        islands_0[gi_0].energy_0 = isl_9.energy_0;
        islands_0[gi_0].rotation_err_0 = isl_9.rotation_err_0;
        islands_0[gi_0].momentum_0 = isl_9.momentum_0;
        islands_0[gi_0].momentum_err_0 = isl_9.momentum_err_0;
    }
    if(contact_island_0)
    {
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        if(done_1 > u32(0))
        {
            c_15 = isl_9.range_0.x + ct_1;
            loop
            {
                if(c_15 < (isl_9.range_0.y))
                {
                }
                else
                {
                    break;
                }
                travel_check_0(c_15, true);
                c_15 = c_15 + u32(256);
            }
        }
    }
    return;
}

fn group_sum3_1( tid_5 : u32,  a_9 : ptr<function, vec3<f32>>,  b_13 : ptr<function, vec3<f32>>)
{
    var x_7 : vec4<f32> = vec4<f32>((*a_9), 0.0f);
    var y_2 : vec4<f32> = vec4<f32>((*b_13), 0.0f);
    group_sum2_0(tid_5, &(x_7), &(y_2));
    (*a_9) = x_7.xyz;
    (*b_13) = y_2.xyz;
    return;
}

fn net_load_1( c_18 : u32,  isl_10 : Island_0,  rg_8 : Rigid_0,  k_11 : u32,  dt_10 : f32,  contact_3 : bool,  f_9 : ptr<function, vec3<f32>>,  t_13 : ptr<function, vec3<f32>>)
{
    var fl_1 : vec3<f32>;
    var tl_1 : vec3<f32>;
    var rep_f_2 : vec3<f32>;
    var rep_m_2 : vec3<f32>;
    chunk_loads_1(c_18, c_18, rg_8.rot_0, k_11, dt_10, contact_3, 0.0f, &(fl_1), &(tl_1), &(rep_f_2), &(rep_m_2));
    var fc_2 : vec3<f32> = fl_1 + params_0.gravity_0.xyz * vec3<f32>(chunks_0[c_18].center_0.w);
    var _S213 : vec3<f32> = chunks_0[c_18].center_0.xyz;
    var _S214 : vec3<f32> = isl_10.com_0.xyz;
    var r_5 : vec3<f32> = rotate_0(rg_8.rot_0, _S213 + state_0[u32(4) * c_18].xyz - _S214);
    (*f_9) = (*f_9) + fc_2;
    (*t_13) = (*t_13) + (cross(r_5, fc_2) + tl_1);
    if((chunks_0[c_18].load_range_0.y) > (chunks_0[c_18].load_range_0.x))
    {
        var fw_3 : vec3<f32> = rotate_0(rg_8.rot_0, rep_f_2);
        (*f_9) = (*f_9) + fw_3;
        (*t_13) = (*t_13) + (cross(rotate_0(rg_8.rot_0, _S213 - _S214), fw_3) + rotate_0(rg_8.rot_0, rep_m_2));
    }
    return;
}

fn rigid_acceleration_1( isl_11 : Island_0,  rg_9 : ptr<function, Rigid_0>,  f_10 : vec3<f32>,  t_14 : vec3<f32>)
{
    var iw_w_1 : vec3<f32> = world_mul_0((*rg_9).rot_0, isl_11.inertia0_0, isl_11.inertia1_0, isl_11.inertia2_0, (*rg_9).w_1);
    (*rg_9).a_0 = f_10 / vec3<f32>(isl_11.com_0.w);
    (*rg_9).alpha_0 = world_mul_0((*rg_9).rot_0, isl_11.inv0_0, isl_11.inv1_0, isl_11.inv2_0, t_14 - cross((*rg_9).w_1, iw_w_1));
    (*rg_9).torque_0 = t_14;
    return;
}

fn chunk_update_1( c_19 : u32,  isl_12 : Island_0,  rg_10 : Rigid_0,  dt_11 : f32,  rml_1 : bool,  step_1 : u32,  contact_4 : bool,  work_2 : ptr<function, f32>,  work_err_2 : ptr<function, f32>)
{
    var _S215 : vec3<f32> = vec3<f32>(0.0f);
    var _S216 : u32 = index_0[c_19];
    var peak_1 : f32 = 0.0f;
    var e_3 : u32 = _S216;
    var fi_1 : vec3<f32> = _S215;
    var mi_3 : vec3<f32> = _S215;
    loop
    {
        if(e_3 < index_0[c_19 + u32(1)])
        {
        }
        else
        {
            break;
        }
        var entry_3 : u32 = index_0[e_3];
        var _S217 : u32 = u32(3) * ((entry_3 >> (u32(1))));
        var fa_1 : vec4<f32> = scratch_0[_S217];
        if(((entry_3 & (u32(1)))) == u32(0))
        {
            var mi_4 : vec3<f32> = mi_3 + scratch_0[_S217 + u32(1)].xyz;
            fi_1 = fi_1 + fa_1.xyz;
            mi_3 = mi_4;
        }
        else
        {
            var mi_5 : vec3<f32> = mi_3 + scratch_0[_S217 + u32(2)].xyz;
            fi_1 = fi_1 + (vec3<f32>(0) - fa_1.xyz);
            mi_3 = mi_5;
        }
        var _S218 : f32 = max(peak_1, fa_1.w);
        var _S219 : u32 = e_3 + u32(1);
        peak_1 = _S218;
        e_3 = _S219;
    }
    var _S220 : u32 = u32(4) * c_19;
    var u_3 : vec3<f32> = state_0[_S220].xyz;
    var _S221 : u32 = _S220 + u32(1);
    var th_5 : vec3<f32> = state_0[_S221].xyz;
    var _S222 : u32 = _S220 + u32(2);
    var v_12 : vec3<f32> = state_0[_S222].xyz;
    var _S223 : u32 = _S220 + u32(3);
    var w_7 : vec3<f32> = state_0[_S223].xyz;
    var mass_1 : f32 = chunks_0[c_19].center_0.w;
    var _S224 : vec3<f32> = chunks_0[c_19].center_0.xyz;
    var _S225 : vec3<f32> = isl_12.com_0.xyz;
    var r_world_1 : vec3<f32> = rotate_0(rg_10.rot_0, _S224 + u_3 - _S225);
    var f_load_1 : vec3<f32>;
    var t_load_1 : vec3<f32>;
    var rep_f_3 : vec3<f32>;
    var rep_m_3 : vec3<f32>;
    chunk_loads_1(c_19, c_19, rg_10.rot_0, step_1, dt_11, contact_4, dt_11, &(f_load_1), &(t_load_1), &(rep_f_3), &(rep_m_3));
    record_chunk_load_0(c_19, f_load_1, t_load_1);
    var _S226 : vec3<f32> = vec3<f32>(mass_1);
    var f_world_2 : vec3<f32> = f_load_1 + params_0.gravity_0.xyz * _S226;
    var t_world_3 : vec3<f32> = t_load_1;
    var f_world_3 : vec3<f32>;
    var t_world_4 : vec3<f32>;
    if(rml_1)
    {
        var t_world_5 : vec3<f32> = t_world_3 - (world_mul_0(rg_10.rot_0, chunks_0[c_19].inertia0_1, chunks_0[c_19].inertia1_1, chunks_0[c_19].inertia2_1, rg_10.alpha_0) + cross(rg_10.w_1, world_mul_0(rg_10.rot_0, chunks_0[c_19].inertia0_1, chunks_0[c_19].inertia1_1, chunks_0[c_19].inertia2_1, rg_10.w_1)));
        f_world_3 = f_world_2 - (rg_10.a_0 + cross(rg_10.alpha_0, r_world_1) + cross(rg_10.w_1, cross(rg_10.w_1, r_world_1))) * _S226;
        t_world_4 = t_world_5;
    }
    else
    {
        f_world_3 = f_world_2;
        t_world_4 = t_world_3;
    }
    var f_ext_2 : vec3<f32> = inverse_rotate_0(rg_10.rot_0, f_world_3);
    var m_ext_3 : vec3<f32> = inverse_rotate_0(rg_10.rot_0, t_world_4);
    var f_ext_3 : vec3<f32>;
    var m_ext_4 : vec3<f32>;
    if(rml_1)
    {
        var wb_1 : vec3<f32> = inverse_rotate_0(rg_10.rot_0, rg_10.w_1);
        var i_w_1 : vec3<f32> = rows_mul_0(chunks_0[c_19].inertia0_1, chunks_0[c_19].inertia1_1, chunks_0[c_19].inertia2_1, w_7);
        var m_ext_5 : vec3<f32> = m_ext_3 - (cross(wb_1, i_w_1) + cross(w_7, rows_mul_0(chunks_0[c_19].inertia0_1, chunks_0[c_19].inertia1_1, chunks_0[c_19].inertia2_1, wb_1)) + cross(w_7, i_w_1));
        f_ext_3 = f_ext_2 - cross(wb_1, v_12) * vec3<f32>((2.0f * mass_1));
        m_ext_4 = m_ext_5;
    }
    else
    {
        f_ext_3 = f_ext_2;
        m_ext_4 = m_ext_3;
    }
    var f_11 : vec3<f32> = f_ext_3 + rep_f_3 + fi_1;
    var m_2 : vec3<f32> = m_ext_4 + rep_m_3 + mi_3;
    var support_1 : u32 = chunks_0[c_19].info_1.x;
    var _S227 : vec3<f32> = vec3<f32>(state_0[_S221].w, state_0[_S222].w, state_0[_S223].w);
    var reaction_1 : vec3<f32>;
    var u_4 : vec3<f32>;
    var th_6 : vec3<f32>;
    var v_13 : vec3<f32>;
    var w_8 : vec3<f32>;
    if(support_1 == u32(1))
    {
        reaction_1 = (vec3<f32>(0) - f_11);
        u_4 = u_3;
        th_6 = th_5;
        v_13 = _S215;
        w_8 = _S215;
    }
    else
    {
        var _S228 : vec4<f32> = chunks_0[c_19].scale_0;
        var w_9 : vec3<f32> = w_7 + rows_mul_0(chunks_0[c_19].inv0_1, chunks_0[c_19].inv1_1, chunks_0[c_19].inv2_1, m_2) * vec3<f32>((dt_11 * chunks_0[c_19].scale_0.z));
        var _S229 : vec3<f32> = vec3<f32>(dt_11);
        var th_7 : vec3<f32> = th_5 + w_9 * _S229;
        if(support_1 == u32(2))
        {
            reaction_1 = (vec3<f32>(0) - f_11);
            u_4 = u_3;
            th_6 = _S215;
        }
        else
        {
            var v_14 : vec3<f32> = v_12 + f_11 * vec3<f32>((dt_11 * _S228.y));
            var u_5 : vec3<f32> = u_3 + v_14 * _S229;
            reaction_1 = _S227;
            u_4 = u_5;
            th_6 = v_14;
        }
        var _S230 : vec3<f32> = th_6;
        th_6 = th_7;
        v_13 = _S230;
        w_8 = w_9;
    }
    state_0[_S220] = vec4<f32>(u_4, peak_1);
    state_0[_S221] = vec4<f32>(th_6, reaction_1.x);
    state_0[_S222] = vec4<f32>(v_13, reaction_1.y);
    state_0[_S223] = vec4<f32>(w_8, reaction_1.z);
    comp_add1_0(&((*work_2)), &((*work_err_2)), (dot(f_load_1, rg_10.vel_0 + rg_10.vel_err_0 + cross(rg_10.w_1, rotate_0(rg_10.rot_0, _S224 + u_4 - _S225)) + rotate_0(rg_10.rot_0, v_13)) + dot(t_load_1, rg_10.w_1 + rotate_0(rg_10.rot_0, w_8))) * dt_11);
    return;
}

fn integrate_rigid_1( isl_13 : Island_0,  rg_11 : ptr<function, Rigid_0>,  dt_12 : f32)
{
    var _S231 : vec3<f32> = vec3<f32>(dt_12);
    var _S232 : vec3<f32> = (*rg_11).torque_0 * _S231;
    var _S233 : vec3<f32> = (*rg_11).l_0;
    var _S234 : vec3<f32> = (*rg_11).l_err_0;
    comp_add_0(&(_S233), &(_S234), _S232);
    (*rg_11).l_0 = _S233;
    (*rg_11).l_err_0 = _S234;
    var l_3 : vec3<f32> = _S233 + _S234;
    var _S235 : vec3<f32> = (*rg_11).a_0 * _S231;
    var _S236 : vec3<f32> = (*rg_11).vel_0;
    var _S237 : vec3<f32> = (*rg_11).vel_err_0;
    comp_add_0(&(_S236), &(_S237), _S235);
    (*rg_11).vel_0 = _S236;
    (*rg_11).vel_err_0 = _S237;
    var w_mid_1 : vec3<f32> = world_mul_0((*rg_11).rot_0, isl_13.inv0_0, isl_13.inv1_0, isl_13.inv2_0, l_3);
    var delta_1 : vec3<f32> = (_S236 + _S237) * _S231 - turn_difference_0(w_mid_1, dt_12, rotate_0((*rg_11).rot_0, isl_13.com_0.xyz));
    var _S238 : vec3<f32> = (*rg_11).pos_0;
    var _S239 : vec3<f32> = (*rg_11).pos_err_0;
    comp_add_0(&(_S238), &(_S239), delta_1);
    (*rg_11).pos_0 = _S238;
    (*rg_11).pos_err_0 = _S239;
    var _S240 : Quat_0 = (*rg_11).rot_0;
    var _S241 : vec4<f32> = (*rg_11).rot_err_0;
    turn_left_0(&(_S240), &(_S241), w_mid_1, dt_12);
    (*rg_11).rot_0 = _S240;
    (*rg_11).rot_err_0 = _S241;
    (*rg_11).w_1 = world_mul_0(_S240, isl_13.inv0_0, isl_13.inv1_0, isl_13.inv2_0, l_3);
    return;
}

fn drift_moments_1( c_20 : u32,  tu_2 : ptr<function, vec3<f32>>,  pv_2 : ptr<function, vec3<f32>>)
{
    var _S242 : u32 = u32(4) * c_20;
    var _S243 : vec3<f32> = vec3<f32>((chunks_0[c_20].center_0.w * chunks_0[c_20].scale_0.x));
    (*tu_2) = (*tu_2) + state_0[_S242].xyz * _S243;
    (*pv_2) = (*pv_2) + state_0[_S242 + u32(2)].xyz * _S243;
    return;
}

fn drift_angular_1( c_21 : u32,  wcom_4 : vec3<f32>,  tr_4 : vec3<f32>,  dv_4 : vec3<f32>,  lu_2 : ptr<function, vec3<f32>>,  lv_2 : ptr<function, vec3<f32>>)
{
    var r_6 : vec3<f32> = chunks_0[c_21].center_0.xyz - wcom_4;
    var _S244 : u32 = u32(4) * c_21;
    var _S245 : vec3<f32> = vec3<f32>(chunks_0[c_21].center_0.w);
    var _S246 : vec4<f32> = chunks_0[c_21].inertia0_1;
    var _S247 : vec4<f32> = chunks_0[c_21].inertia1_1;
    var _S248 : vec4<f32> = chunks_0[c_21].inertia2_1;
    var _S249 : vec3<f32> = vec3<f32>(chunks_0[c_21].scale_0.x);
    (*lu_2) = (*lu_2) + (cross(r_6, state_0[_S244].xyz - tr_4) * _S245 + rows_mul_0(chunks_0[c_21].inertia0_1, chunks_0[c_21].inertia1_1, chunks_0[c_21].inertia2_1, state_0[_S244 + u32(1)].xyz)) * _S249;
    (*lv_2) = (*lv_2) + (cross(r_6, state_0[_S244 + u32(2)].xyz - dv_4) * _S245 + rows_mul_0(_S246, _S247, _S248, state_0[_S244 + u32(3)].xyz)) * _S249;
    return;
}

fn drift_rigid_1( isl_14 : Island_0,  rg_12 : ptr<function, Rigid_0>,  tr_5 : vec3<f32>,  phi_4 : vec3<f32>,  dv_5 : vec3<f32>,  dw_3 : vec3<f32>)
{
    var wcom_5 : vec3<f32> = isl_14.wcom_0.xyz;
    var rot_3 : Quat_0 = (*rg_12).rot_0;
    var _S250 : vec3<f32> = rotate_0((*rg_12).rot_0, tr_5 - turn_difference_0(phi_4, 1.0f, wcom_5));
    var _S251 : vec3<f32> = (*rg_12).pos_0;
    var _S252 : vec3<f32> = (*rg_12).pos_err_0;
    comp_add_0(&(_S251), &(_S252), _S250);
    (*rg_12).pos_0 = _S251;
    (*rg_12).pos_err_0 = _S252;
    var b_14 : vec3<f32> = inverse_rotate_0((*rg_12).rot_0, (*rg_12).w_1);
    var dl_turn_1 : vec3<f32> = rotate_0((*rg_12).rot_0, cross(phi_4, rows_mul_0(isl_14.inertia0_0, isl_14.inertia1_0, isl_14.inertia2_0, b_14)) - rows_mul_0(isl_14.inertia0_0, isl_14.inertia1_0, isl_14.inertia2_0, cross(phi_4, b_14)));
    var _S253 : Quat_0 = (*rg_12).rot_0;
    var _S254 : vec4<f32> = (*rg_12).rot_err_0;
    turn_right_0(&(_S253), &(_S254), phi_4);
    (*rg_12).rot_0 = _S253;
    (*rg_12).rot_err_0 = _S254;
    var _S255 : vec3<f32> = rotate_0(rot_3, dv_5 + cross(dw_3, isl_14.com_0.xyz - wcom_5));
    var _S256 : vec3<f32> = (*rg_12).vel_0;
    var _S257 : vec3<f32> = (*rg_12).vel_err_0;
    comp_add_0(&(_S256), &(_S257), _S255);
    (*rg_12).vel_0 = _S256;
    (*rg_12).vel_err_0 = _S257;
    var _S258 : vec3<f32> = rotate_0(rot_3, dw_3);
    (*rg_12).w_1 = (*rg_12).w_1 + _S258;
    var _S259 : vec3<f32> = dl_turn_1 + world_mul_0(_S253, isl_14.inertia0_0, isl_14.inertia1_0, isl_14.inertia2_0, _S258);
    var _S260 : vec3<f32> = (*rg_12).l_0;
    var _S261 : vec3<f32> = (*rg_12).l_err_0;
    comp_add_0(&(_S260), &(_S261), _S259);
    (*rg_12).l_0 = _S260;
    (*rg_12).l_err_0 = _S261;
    return;
}

fn comp_add1_2( sum_5 : ptr<function, f32>,  err_4 : ptr<function, f32>,  x_8 : f32)
{
    var t_15 : f32 = (*sum_5) + x_8;
    if((abs((*sum_5))) >= (abs(x_8)))
    {
        (*err_4) = (*err_4) + ((*sum_5) - t_15 + x_8);
    }
    else
    {
        (*err_4) = (*err_4) + (x_8 - t_15 + (*sum_5));
    }
    (*sum_5) = t_15;
    return;
}

fn island_frame_at_1( gi_1 : u32,  tid_6 : u32)
{
    var woke_2 : bool;
    var ct_2 : u32;
    if(tid_6 < u32(256))
    {
        ct_2 = tid_6;
    }
    else
    {
        ct_2 = u32(1073741824);
    }
    var isl_15 : Island_0;
    isl_15.range_0 = islands_0[gi_1].range_0;
    isl_15.info_0 = islands_0[gi_1].info_0;
    isl_15.com_0 = islands_0[gi_1].com_0;
    isl_15.inertia0_0 = islands_0[gi_1].inertia0_0;
    isl_15.inertia1_0 = islands_0[gi_1].inertia1_0;
    isl_15.inertia2_0 = islands_0[gi_1].inertia2_0;
    isl_15.inv0_0 = islands_0[gi_1].inv0_0;
    isl_15.inv1_0 = islands_0[gi_1].inv1_0;
    isl_15.inv2_0 = islands_0[gi_1].inv2_0;
    isl_15.wcom_0 = islands_0[gi_1].wcom_0;
    isl_15.winv0_0 = islands_0[gi_1].winv0_0;
    isl_15.winv1_0 = islands_0[gi_1].winv1_0;
    isl_15.winv2_0 = islands_0[gi_1].winv2_0;
    isl_15.rotation_0 = islands_0[gi_1].rotation_0;
    isl_15.position_0 = islands_0[gi_1].position_0;
    isl_15.position_err_0 = islands_0[gi_1].position_err_0;
    isl_15.velocity_0 = islands_0[gi_1].velocity_0;
    isl_15.velocity_err_0 = islands_0[gi_1].velocity_err_0;
    isl_15.angular_velocity_0 = islands_0[gi_1].angular_velocity_0;
    isl_15.done_0 = islands_0[gi_1].done_0;
    isl_15.coverage_0 = islands_0[gi_1].coverage_0;
    isl_15.wide_groups_0 = islands_0[gi_1].wide_groups_0;
    isl_15.done_pad_0 = islands_0[gi_1].done_pad_0;
    isl_15.probes_0 = islands_0[gi_1].probes_0;
    isl_15.energy_0 = islands_0[gi_1].energy_0;
    isl_15.rotation_err_0 = islands_0[gi_1].rotation_err_0;
    isl_15.momentum_0 = islands_0[gi_1].momentum_0;
    isl_15.momentum_err_0 = islands_0[gi_1].momentum_err_0;
    var driven_1 : bool = (((isl_15.info_0.x) & (u32(2)))) != u32(0);
    var _S262 : bool = !((((isl_15.info_0.x) & (u32(1)))) != u32(0));
    var _S263 : bool;
    if(_S262)
    {
        _S263 = !driven_1;
    }
    else
    {
        _S263 = false;
    }
    var contact_island_1 : bool = (((isl_15.info_0.x) & (u32(4)))) != u32(0);
    var _S264 : bool = (((isl_15.info_0.x) & (u32(16)))) != u32(0);
    seen_clear_0(tid_6);
    var _S265 : bool = tid_6 == u32(0);
    var cov_5 : u32;
    var settled_1 : bool;
    if(_S265)
    {
        if(contact_island_1 != ((params_0.contact_mode_0) == u32(1)))
        {
            settled_1 = true;
        }
        else
        {
            settled_1 = (((isl_15.info_0.x) & (u32(8)))) != u32(0);
        }
        if(settled_1)
        {
            cov_5 = u32(0);
        }
        else
        {
            cov_5 = u32(1);
        }
        if(contact_island_1)
        {
            settled_1 = contact_stopped_0(isl_15);
        }
        else
        {
            settled_1 = false;
        }
        if(settled_1)
        {
            cov_5 = u32(0);
        }
        g_run_0 = cov_5;
        g_halt_0 = u32(0);
    }
    workgroupBarrier();
    if((((isl_15.info_0.z) & (u32(1)))) != u32(0))
    {
        settled_1 = true;
    }
    else
    {
        settled_1 = g_run_0 == u32(0);
    }
    if(settled_1)
    {
        cov_5 = u32(0);
    }
    else
    {
        cov_5 = min(isl_15.info_0.y, params_0.max_steps_0);
    }
    var _S266 : f32 = params_0.dt_0;
    var _S267 : bool = (params_0.rigid_motion_loads_0) != u32(0);
    var rg_13 : Rigid_0 = rigid_of_0(isl_15);
    var work_3 : f32 = 0.0f;
    var work_err_3 : f32 = 0.0f;
    settled_1 = _S264;
    var done_2 : u32 = u32(0);
    var woke_3 : bool = false;
    var resumed_2 : bool = false;
    loop
    {
        if(done_2 < cov_5)
        {
        }
        else
        {
            woke_2 = woke_3;
            break;
        }
        var k_12 : u32 = isl_15.info_0.w + done_2 + u32(1) - u32(1) - params_0.step_start_0;
        var quiet_2 : bool = (((isl_15.info_0.x) & (u32(32)))) != u32(0);
        var _S268 : bool;
        if(settled_1)
        {
            _S268 = !quiet_2;
        }
        else
        {
            _S268 = false;
        }
        var c_22 : u32;
        var _S269 : bool;
        var explicit_step_1 : bool;
        if(_S268)
        {
            var _S270 : vec3<f32> = vec3<f32>(0.0f);
            var norm_1 : vec3<f32> = _S270;
            var unused0_1 : vec3<f32> = _S270;
            c_22 = isl_15.range_0.x + ct_2;
            loop
            {
                if(c_22 < (isl_15.range_0.y))
                {
                }
                else
                {
                    break;
                }
                var _S271 : f32 = settled_chunk_load_0(c_22, rg_13.rot_0, k_12, _S266, contact_island_1);
                norm_1[i32(0)] = norm_1[i32(0)] + _S271;
                c_22 = c_22 + u32(256);
            }
            group_sum3_1(tid_6, &(norm_1), &(unused0_1));
            if((params_0.solve_mode_0) == u32(1))
            {
                _S269 = (abs(norm_1.x - isl_15.energy_0.z)) > (isl_15.energy_0.w);
            }
            else
            {
                _S269 = false;
            }
            if(_S269)
            {
                explicit_step_1 = false;
                woke_2 = true;
            }
            else
            {
                explicit_step_1 = settled_1;
                woke_2 = woke_3;
            }
            settled_1 = explicit_step_1;
        }
        else
        {
            woke_2 = woke_3;
        }
        if(_S262)
        {
            _S269 = !quiet_2;
        }
        else
        {
            _S269 = false;
        }
        if(_S269)
        {
            var _S272 : vec3<f32> = vec3<f32>(0.0f);
            var f_12 : vec3<f32> = _S272;
            var t_16 : vec3<f32> = _S272;
            c_22 = isl_15.range_0.x + ct_2;
            loop
            {
                if(c_22 < (isl_15.range_0.y))
                {
                }
                else
                {
                    break;
                }
                net_load_1(c_22, isl_15, rg_13, k_12, _S266, contact_island_1, &(f_12), &(t_16));
                c_22 = c_22 + u32(256);
            }
            group_sum3_1(tid_6, &(f_12), &(t_16));
            rigid_acceleration_1(isl_15, &(rg_13), f_12, t_16);
        }
        if(!settled_1)
        {
            explicit_step_1 = !quiet_2;
        }
        else
        {
            explicit_step_1 = false;
        }
        var _S273 : bool;
        if(explicit_step_1)
        {
            if(_S265)
            {
                _S273 = (((islands_0[gi_1].info_0.z) & (u32(2)))) != u32(0);
            }
            else
            {
                _S273 = false;
            }
            if(_S273)
            {
                g_halt_0 = u32(1);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
            c_22 = isl_15.range_0.x + ct_2;
            loop
            {
                if(c_22 < (isl_15.range_0.y))
                {
                }
                else
                {
                    break;
                }
                chunk_update_1(c_22, isl_15, rg_13, _S266, _S267, k_12, contact_island_1, &(work_3), &(work_err_3));
                c_22 = c_22 + u32(256);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
        }
        if(_S263)
        {
            _S273 = !quiet_2;
        }
        else
        {
            _S273 = false;
        }
        if(_S273)
        {
            integrate_rigid_1(isl_15, &(rg_13), _S266);
        }
        var _S274 : bool;
        if(_S262)
        {
            _S274 = explicit_step_1;
        }
        else
        {
            _S274 = false;
        }
        if(_S274)
        {
            var _S275 : vec3<f32> = isl_15.wcom_0.xyz;
            var _S276 : vec3<f32> = vec3<f32>(0.0f);
            var tu_3 : vec3<f32> = _S276;
            var pv_3 : vec3<f32> = _S276;
            c_22 = isl_15.range_0.x + ct_2;
            loop
            {
                if(c_22 < (isl_15.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_moments_1(c_22, &(tu_3), &(pv_3));
                c_22 = c_22 + u32(256);
            }
            group_sum3_1(tid_6, &(tu_3), &(pv_3));
            var tr_6 : vec3<f32> = tu_3 / vec3<f32>(isl_15.wcom_0.w);
            var dv_6 : vec3<f32> = pv_3 / vec3<f32>(isl_15.wcom_0.w);
            var lu_3 : vec3<f32> = _S276;
            var lv_3 : vec3<f32> = _S276;
            var c_23 : u32 = isl_15.range_0.x + ct_2;
            loop
            {
                if(c_23 < (isl_15.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_angular_1(c_23, _S275, tr_6, dv_6, &(lu_3), &(lv_3));
                c_23 = c_23 + u32(256);
            }
            group_sum3_1(tid_6, &(lu_3), &(lv_3));
            var phi_5 : vec3<f32> = rows_mul_0(isl_15.winv0_0, isl_15.winv1_0, isl_15.winv2_0, lu_3);
            var dw_4 : vec3<f32> = rows_mul_0(isl_15.winv0_0, isl_15.winv1_0, isl_15.winv2_0, lv_3);
            var c_24 : u32 = isl_15.range_0.x + ct_2;
            loop
            {
                if(c_24 < (isl_15.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_24, _S275, tr_6, phi_5, dv_6, dw_4);
                c_24 = c_24 + u32(256);
            }
            if(!driven_1)
            {
                drift_rigid_1(isl_15, &(rg_13), tr_6, phi_5, dv_6, dw_4);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
        }
        var _S277 : bool;
        if(_S265)
        {
            _S277 = (isl_15.probes_0.y) > (isl_15.probes_0.x);
        }
        else
        {
            _S277 = false;
        }
        if(_S277)
        {
            record_probes_0(isl_15, rg_13, k_12);
        }
        var _S278 : u32 = done_2 + u32(1);
        var _S279 : bool;
        if(g_halt_0 != u32(0))
        {
            var _S280 : bool = island_connected_narrow_0(tid_6, ct_2, isl_15);
            _S279 = _S280;
        }
        else
        {
            _S279 = false;
        }
        var resumed_3 : bool;
        if(_S279)
        {
            if(_S265)
            {
                g_halt_0 = u32(0);
            }
            workgroupBarrier();
            resumed_3 = true;
        }
        else
        {
            resumed_3 = resumed_2;
        }
        if(g_halt_0 != u32(0))
        {
            done_2 = _S278;
            resumed_2 = resumed_3;
            break;
        }
        done_2 = _S278;
        woke_3 = woke_2;
        resumed_2 = resumed_3;
    }
    var wsum_1 : vec3<f32> = vec3<f32>(work_3, work_err_3, 0.0f);
    var unused_1 : vec3<f32> = vec3<f32>(0.0f);
    group_sum3_1(tid_6, &(wsum_1), &(unused_1));
    if(_S265)
    {
        isl_15.rotation_0 = quat_vec_0(rg_13.rot_0);
        isl_15.rotation_err_0 = rg_13.rot_err_0;
        isl_15.position_0 = vec4<f32>(rg_13.pos_0, 0.0f);
        isl_15.position_err_0 = vec4<f32>(rg_13.pos_err_0, 0.0f);
        isl_15.velocity_0 = vec4<f32>(rg_13.vel_0, 0.0f);
        isl_15.velocity_err_0 = vec4<f32>(rg_13.vel_err_0, 0.0f);
        isl_15.angular_velocity_0 = vec4<f32>(rg_13.w_1, 0.0f);
        isl_15.momentum_0 = vec4<f32>(rg_13.l_0, 0.0f);
        isl_15.momentum_err_0 = vec4<f32>(rg_13.l_err_0, 0.0f);
        isl_15.done_0 = done_2;
        isl_15.info_0[i32(1)] = isl_15.info_0[i32(1)] - done_2;
        if(g_halt_0 != u32(0))
        {
            _S263 = contact_island_1;
        }
        else
        {
            _S263 = false;
        }
        if(_S263)
        {
            contact_split_at_0(isl_15.info_0.w + done_2);
        }
        var _S281 : f32 = wsum_1.x;
        var _S282 : f32 = isl_15.energy_0[i32(0)];
        var _S283 : f32 = isl_15.energy_0[i32(1)];
        comp_add1_2(&(_S282), &(_S283), _S281);
        isl_15.energy_0[i32(0)] = _S282;
        isl_15.energy_0[i32(1)] = _S283 + wsum_1.y;
        isl_15.info_0[i32(3)] = isl_15.info_0[i32(3)] + done_2;
        if((((isl_15.info_0.x) & (u32(8)))) == u32(0))
        {
            isl_15.info_0[i32(2)] = ((isl_15.info_0[i32(2)]) & (u32(4294967293)));
        }
        if(g_halt_0 != u32(0))
        {
            isl_15.info_0[i32(2)] = ((isl_15.info_0[i32(2)]) | (u32(1)));
        }
        var _S284 : u32 = seen_bits_0();
        if(g_halt_0 != u32(0))
        {
            ct_2 = u32(128);
        }
        else
        {
            ct_2 = u32(0);
        }
        var _S285 : u32 = (_S284 | (ct_2));
        if(woke_2)
        {
            ct_2 = u32(2048);
        }
        else
        {
            ct_2 = u32(0);
        }
        var _S286 : u32 = (_S285 | (ct_2));
        if(resumed_2)
        {
            ct_2 = u32(2097280);
        }
        else
        {
            ct_2 = u32(0);
        }
        var cov_6 : u32 = (_S286 | (ct_2));
        if(done_2 > u32(0))
        {
            var quiet_3 : bool = (((isl_15.info_0.x) & (u32(32)))) != u32(0);
            if(quiet_3)
            {
                ct_2 = u32(4096);
            }
            else
            {
                ct_2 = u32(0);
            }
            var cov_7 : u32 = (cov_6 | (ct_2));
            if((((isl_15.info_0.x) & (u32(16)))) != u32(0))
            {
                _S263 = !quiet_3;
            }
            else
            {
                _S263 = false;
            }
            if(_S263)
            {
                ct_2 = u32(512);
            }
            else
            {
                ct_2 = u32(0);
            }
            var cov_8 : u32 = (cov_7 | (ct_2));
            if((params_0.contact_mode_0) == u32(1))
            {
                ct_2 = u32(524288);
            }
            else
            {
                ct_2 = u32(0);
            }
            var cov_9 : u32 = (cov_8 | (ct_2));
            if((isl_15.probes_0.y) > (isl_15.probes_0.x))
            {
                ct_2 = u32(262144);
            }
            else
            {
                ct_2 = u32(0);
            }
            cov_5 = (cov_9 | (ct_2));
        }
        else
        {
            cov_5 = cov_6;
        }
        isl_15.coverage_0 = ((isl_15.coverage_0) | (cov_5));
        if(woke_2)
        {
            isl_15.info_0[i32(0)] = ((isl_15.info_0[i32(0)]) & (u32(4294967279)));
            isl_15.info_0[i32(2)] = ((isl_15.info_0[i32(2)]) | (u32(4)));
        }
        islands_0[gi_1].range_0 = isl_15.range_0;
        islands_0[gi_1].info_0 = isl_15.info_0;
        islands_0[gi_1].com_0 = isl_15.com_0;
        islands_0[gi_1].inertia0_0 = isl_15.inertia0_0;
        islands_0[gi_1].inertia1_0 = isl_15.inertia1_0;
        islands_0[gi_1].inertia2_0 = isl_15.inertia2_0;
        islands_0[gi_1].inv0_0 = isl_15.inv0_0;
        islands_0[gi_1].inv1_0 = isl_15.inv1_0;
        islands_0[gi_1].inv2_0 = isl_15.inv2_0;
        islands_0[gi_1].wcom_0 = isl_15.wcom_0;
        islands_0[gi_1].winv0_0 = isl_15.winv0_0;
        islands_0[gi_1].winv1_0 = isl_15.winv1_0;
        islands_0[gi_1].winv2_0 = isl_15.winv2_0;
        islands_0[gi_1].rotation_0 = isl_15.rotation_0;
        islands_0[gi_1].position_0 = isl_15.position_0;
        islands_0[gi_1].position_err_0 = isl_15.position_err_0;
        islands_0[gi_1].velocity_0 = isl_15.velocity_0;
        islands_0[gi_1].velocity_err_0 = isl_15.velocity_err_0;
        islands_0[gi_1].angular_velocity_0 = isl_15.angular_velocity_0;
        islands_0[gi_1].done_0 = isl_15.done_0;
        islands_0[gi_1].coverage_0 = isl_15.coverage_0;
        islands_0[gi_1].wide_groups_0 = isl_15.wide_groups_0;
        islands_0[gi_1].done_pad_0 = isl_15.done_pad_0;
        islands_0[gi_1].probes_0 = isl_15.probes_0;
        islands_0[gi_1].energy_0 = isl_15.energy_0;
        islands_0[gi_1].rotation_err_0 = isl_15.rotation_err_0;
        islands_0[gi_1].momentum_0 = isl_15.momentum_0;
        islands_0[gi_1].momentum_err_0 = isl_15.momentum_err_0;
    }
    return;
}

fn to_local_0( _S287 : u32,  _S288 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(dot(_S288, bonds_0[_S287].t1_0.xyz), dot(_S288, bonds_0[_S287].t2_0.xyz), dot(_S288, bonds_0[_S287].normal_0.xyz));
}

fn is_damaged_0( st_1 : ptr<function, JointState_std430_0>) -> bool
{
    var _S289 : bool;
    if(((*st_1).damage_0) > 0.0f)
    {
        _S289 = true;
    }
    else
    {
        _S289 = ((*st_1).crush_1) > 0.0f;
    }
    return _S289;
}

struct JointState_0
{
     damage_0 : f32,
     crush_1 : f32,
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

struct Measures_0
{
     tension_0 : f32,
     shear_0 : f32,
     normal_compression_0 : f32,
     compression_0 : f32,
     compressive_force_0 : f32,
};

struct JointResponse_0
{
     force_lin_1 : vec3<f32>,
     force_ang_1 : vec3<f32>,
     state_1 : JointState_0,
     dissipated_2 : f32,
     overshoot_0 : f32,
     stored_2 : f32,
     disconnected_0 : bool,
     fatigue_before_0 : f32,
     measures_0 : Measures_0,
};

fn connected_1( st_2 : ptr<function, JointState_std430_0>,  has_rebar_1 : bool) -> bool
{
    var _S290 : bool;
    if(((*st_2).damage_0) < 1.0f)
    {
        _S290 = true;
    }
    else
    {
        if(has_rebar_1)
        {
            _S290 = ((*st_2).rebar_broken_0) == 0.0f;
        }
        else
        {
            _S290 = false;
        }
    }
    return _S290;
}

fn fdiv_0( a_10 : f32,  b_15 : f32) -> f32
{
    return a_10 / b_15;
}

fn fsqrt_0( a_11 : f32) -> f32
{
    return sqrt(a_11);
}

fn stress_measures_0( b_16 : ptr<function, JointBond_std430_0>,  q_lin_0 : vec3<f32>,  q_ang_0 : vec3<f32>) -> Measures_0
{
    var area_2 : f32 = (*b_16).geom0_0.x;
    var _S291 : f32 = q_lin_0.z;
    var axial_0 : f32 = fdiv_0(_S291, area_2);
    var bending_0 : f32 = fdiv_0(abs(q_ang_0.x), (*b_16).geom1_0.x) + fdiv_0(abs(q_ang_0.y), (*b_16).geom1_0.y);
    var _S292 : f32 = q_lin_0.x;
    var _S293 : f32 = q_lin_0.y;
    var shear_1 : f32 = fdiv_0(fsqrt_0(_S292 * _S292 + _S293 * _S293), area_2) + fdiv_0(abs(q_ang_0.z), (*b_16).geom0_0.w);
    var m_3 : Measures_0;
    m_3.tension_0 = axial_0 + bending_0;
    m_3.shear_0 = shear_1;
    var _S294 : f32 = - axial_0;
    m_3.normal_compression_0 = max(_S294, 0.0f);
    m_3.compression_0 = _S294 + bending_0;
    m_3.compressive_force_0 = max(- _S291, 0.0f);
    return m_3;
}

fn expm1_accurate_0( x_9 : f32) -> f32
{
    if((abs(x_9)) < 0.00100000004749745f)
    {
        return x_9 * (1.0f + x_9 * (0.5f + x_9 * 0.1666666716337204f));
    }
    return exp(x_9) - 1.0f;
}

fn fpow_0( a_12 : f32,  b_17 : f32) -> f32
{
    return pow(a_12, b_17);
}

fn dif_factor_0( mat_1 : ptr<function, JointMaterial_std140_0>,  strain_rate_1 : f32) -> f32
{
    var r_7 : f32 = abs(strain_rate_1);
    var _S295 : vec4<f32> = (*mat_1).dif_0;
    var ref_0 : f32 = (*mat_1).dif_0.x;
    if(r_7 <= ref_0)
    {
        return 1.0f;
    }
    var _S296 : f32 = _S295.z;
    var f_13 : f32;
    if(r_7 <= _S296)
    {
        f_13 = fpow_0(fdiv_0(r_7, ref_0), _S295.y);
    }
    else
    {
        f_13 = fpow_0(fdiv_0(_S296, ref_0), _S295.y) * fpow_0(fdiv_0(r_7, _S296), _S295.w);
    }
    return clamp(f_13, 1.0f, (*mat_1).misc_0.x);
}

fn fatigue_factor_0( mat_2 : ptr<function, JointMaterial_std140_0>,  life_1 : f32) -> f32
{
    if(((((*mat_2).kind_flags_0.y) & (u32(64)))) == u32(0))
    {
        return 1.0f;
    }
    return fpow_0(clamp(life_1, 0.0f, 1.0f), fdiv_0(1.0f, (*mat_2).misc_0.y - 2.0f));
}

fn infinity_0() -> f32
{
    return (bitcast<f32>((u32(2139095040))));
}

fn failure_indices_0( mat_3 : ptr<function, JointMaterial_std140_0>,  b_18 : ptr<function, JointBond_std430_0>,  m_4 : Measures_0,  multiplier_0 : f32) -> vec4<f32>
{
    var fc_3 : f32 = (*mat_3).strength_0.y * multiplier_0;
    var _S297 : f32 = min((*mat_3).strength_0.z * multiplier_0 + (*mat_3).strength_0.w * m_4.normal_compression_0, (*mat_3).energy_1.x * multiplier_0);
    var idx_0 : vec4<f32>;
    idx_0[i32(0)] = max(fdiv_0(m_4.tension_0, (*mat_3).strength_0.x * multiplier_0), 0.0f);
    var _S298 : f32;
    if(_S297 > 0.0f)
    {
        _S298 = fdiv_0(m_4.shear_0, _S297);
    }
    else
    {
        _S298 = infinity_0();
    }
    idx_0[i32(1)] = _S298;
    idx_0[i32(2)] = max(fdiv_0(m_4.compression_0, fc_3), 0.0f);
    var _S299 : f32 = (*b_18).stiff1_0.y;
    if(_S299 > 0.0f)
    {
        _S298 = fdiv_0(m_4.compressive_force_0, _S299);
    }
    else
    {
        _S298 = 0.0f;
    }
    idx_0[i32(3)] = _S298;
    return idx_0;
}

fn sq_0( x_10 : f32) -> f32
{
    return x_10 * x_10;
}

fn compressed_region_0( w0_0 : f32,  w1_0 : f32,  dz_0 : f32,  ax_0 : f32,  ay_0 : f32,  region_0 : ptr<function, array<f32, i32(6)>>)
{
    var count_1 : u32;
    var h0_0 : f32 = 0.5f * w0_0;
    var h1_0 : f32 = 0.5f * w1_0;
    var _S300 : f32 = - h0_0;
    var _S301 : f32 = - h1_0;
    var _S302 : array<vec2<f32>, i32(4)> = array<vec2<f32>, i32(4)>( vec2<f32>(_S300, _S301), vec2<f32>(h0_0, _S301), vec2<f32>(h0_0, h1_0), vec2<f32>(_S300, h1_0) );
    var poly_0 : array<vec2<f32>, i32(8)>;
    var i_3 : u32 = u32(0);
    var count_2 : u32 = u32(0);
    loop
    {
        if(i_3 < u32(4))
        {
        }
        else
        {
            break;
        }
        var _S303 : u32 = i_3;
        var _S304 : u32 = i_3 + u32(1);
        var _S305 : u32 = _S304 % u32(4);
        var _S306 : f32 = _S302[i_3].y;
        var _S307 : f32 = _S302[i_3].x;
        var fp_0 : f32 = dz_0 + ax_0 * _S306 - ay_0 * _S307;
        var _S308 : f32 = _S302[_S305].y;
        var _S309 : f32 = _S302[_S305].x;
        var fq_0 : f32 = dz_0 + ax_0 * _S308 - ay_0 * _S309;
        var _S310 : bool = fp_0 < 0.0f;
        if(_S310)
        {
            var _S311 : u32 = count_2 + u32(1);
            poly_0[count_2] = _S302[_S303];
            count_1 = _S311;
        }
        else
        {
            count_1 = count_2;
        }
        if(_S310 != (fq_0 < 0.0f))
        {
            var t_17 : f32 = fp_0 / (fp_0 - fq_0);
            var _S312 : u32 = count_1 + u32(1);
            poly_0[count_1] = vec2<f32>(_S307 + t_17 * (_S309 - _S307), _S306 + t_17 * (_S308 - _S306));
            count_2 = _S312;
        }
        else
        {
            count_2 = count_1;
        }
        i_3 = _S304;
    }
    count_1 = u32(0);
    loop
    {
        if(count_1 < u32(6))
        {
        }
        else
        {
            break;
        }
        (*region_0)[count_1] = 0.0f;
        count_1 = count_1 + u32(1);
    }
    if(count_2 < u32(3))
    {
        return;
    }
    var o_1 : vec2<f32> = poly_0[i32(0)];
    i_3 = u32(0);
    var a_13 : f32 = 0.0f;
    var sx_0 : f32 = 0.0f;
    var sy_0 : f32 = 0.0f;
    var ixx_0 : f32 = 0.0f;
    var iyy_0 : f32 = 0.0f;
    var ixy_0 : f32 = 0.0f;
    loop
    {
        if(i_3 < count_2)
        {
        }
        else
        {
            break;
        }
        var _S313 : f32 = o_1.x;
        var x0_0 : f32 = poly_0[i_3].x - _S313;
        var _S314 : f32 = o_1.y;
        var y0_0 : f32 = poly_0[i_3].y - _S314;
        var _S315 : u32 = i_3 + u32(1);
        var _S316 : u32 = _S315 % count_2;
        var x1_0 : f32 = poly_0[_S316].x - _S313;
        var y1_0 : f32 = poly_0[_S316].y - _S314;
        var _S317 : f32 = x0_0 * y1_0;
        var _S318 : f32 = x1_0 * y0_0;
        var cr_0 : f32 = _S317 - _S318;
        var a_14 : f32 = a_13 + cr_0 / 2.0f;
        var sx_1 : f32 = sx_0 + (x0_0 + x1_0) * cr_0 / 6.0f;
        var sy_1 : f32 = sy_0 + (y0_0 + y1_0) * cr_0 / 6.0f;
        var ixx_1 : f32 = ixx_0 + (x0_0 * x0_0 + x0_0 * x1_0 + x1_0 * x1_0) * cr_0 / 12.0f;
        var iyy_1 : f32 = iyy_0 + (y0_0 * y0_0 + y0_0 * y1_0 + y1_0 * y1_0) * cr_0 / 12.0f;
        var ixy_1 : f32 = ixy_0 + (_S317 + 2.0f * x0_0 * y0_0 + 2.0f * x1_0 * y1_0 + _S318) * cr_0 / 24.0f;
        i_3 = _S315;
        a_13 = a_14;
        sx_0 = sx_1;
        sy_0 = sy_1;
        ixx_0 = ixx_1;
        iyy_0 = iyy_1;
        ixy_0 = ixy_1;
    }
    if(a_13 <= 0.0f)
    {
        return;
    }
    var cx_0 : f32 = sx_0 / a_13;
    var cy_0 : f32 = sy_0 / a_13;
    (*region_0)[i32(0)] = a_13;
    (*region_0)[i32(1)] = o_1.x + cx_0;
    (*region_0)[i32(2)] = o_1.y + cy_0;
    var _S319 : f32 = a_13 * cx_0;
    (*region_0)[i32(3)] = ixx_0 - _S319 * cx_0;
    (*region_0)[i32(4)] = iyy_0 - a_13 * cy_0 * cy_0;
    (*region_0)[i32(5)] = ixy_0 - _S319 * cy_0;
    return;
}

fn no_tension_patch_0( kn_0 : f32,  w0_1 : f32,  w1_1 : f32,  dz_1 : f32,  ax_1 : f32,  ay_1 : f32) -> vec4<f32>
{
    var r_8 : array<f32, i32(6)>;
    compressed_region_0(w0_1, w1_1, dz_1, ax_1, ay_1, &(r_8));
    var a_15 : f32 = r_8[i32(0)];
    if((r_8[i32(0)]) == 0.0f)
    {
        return vec4<f32>(0.0f);
    }
    var k_13 : f32 = kn_0 / (w0_1 * w1_1);
    var fc_4 : f32 = dz_1 + ax_1 * r_8[i32(2)] - ay_1 * r_8[i32(1)];
    var _S320 : f32 = a_15 * fc_4;
    var _S321 : f32 = - ay_1;
    return vec4<f32>(k_13 * a_15 * fc_4, k_13 * (_S320 * r_8[i32(2)] + (_S321 * r_8[i32(5)] + ax_1 * r_8[i32(4)])), - k_13 * (_S320 * r_8[i32(1)] + (_S321 * r_8[i32(3)] + ax_1 * r_8[i32(5)])), 0.5f * k_13 * (_S320 * fc_4 + ay_1 * ay_1 * r_8[i32(3)] + ax_1 * ax_1 * r_8[i32(4)] - 2.0f * ax_1 * ay_1 * r_8[i32(5)]));
}

var<private> SPRING_AT_0 : array<f32, i32(6)> = array<f32, i32(6)>( -0.4166666567325592f, -0.25f, -0.0833333358168602f, 0.0833333358168602f, 0.25f, 0.4166666567325592f );
fn contact_normal_0( mat_4 : ptr<function, JointMaterial_std140_0>,  b_19 : ptr<function, JointBond_std430_0>,  crush_2 : f32,  d_lin_0 : vec3<f32>,  d_ang_0 : vec3<f32>) -> vec4<f32>
{
    var kn_1 : f32 = (*b_19).stiff0_0.x;
    var w0_2 : f32 = (*b_19).geom0_0.y;
    var w1_2 : f32 = (*b_19).geom0_0.z;
    var nc_sum_0 : f32;
    var m1_0 : f32;
    var m2_0 : f32;
    var energy_2 : f32;
    if(((((*mat_4).kind_flags_0.y) & (u32(4)))) != u32(0))
    {
        var p_2 : vec4<f32> = no_tension_patch_0(kn_1 * (1.0f - crush_2), w0_2, w1_2, d_lin_0.z, d_ang_0.x, d_ang_0.y);
        var _S322 : f32 = p_2.y;
        var _S323 : f32 = p_2.z;
        var _S324 : f32 = p_2.w;
        nc_sum_0 = p_2.x;
        m1_0 = _S322;
        m2_0 = _S323;
        energy_2 = _S324;
    }
    else
    {
        var ki_0 : f32 = kn_1 * (1.0f - crush_2) / 36.0f;
        var _S325 : f32 = d_ang_0.x;
        var _S326 : f32 = d_ang_0.y;
        var spread_0 : f32 = abs(_S325) * 0.4166666567325592f * w1_2 + abs(_S326) * 0.4166666567325592f * w0_2;
        var _S327 : f32 = d_lin_0.z;
        var slack_0 : f32 = 9.99999997475242708e-07f * (abs(_S327) + spread_0);
        if((_S327 - spread_0) > slack_0)
        {
            nc_sum_0 = 0.0f;
            m1_0 = 0.0f;
            m2_0 = 0.0f;
            energy_2 = 0.0f;
        }
        else
        {
            if((_S327 + spread_0) < (- slack_0))
            {
                var i1_0 : f32 = 2.91666650772094727f * w0_2 * w0_2;
                var i2_0 : f32 = 2.91666650772094727f * w1_2 * w1_2;
                var _S328 : f32 = ki_0 * _S325 * i2_0;
                var _S329 : f32 = ki_0 * _S326 * i1_0;
                var _S330 : f32 = 0.5f * ki_0 * (36.0f * _S327 * _S327 + _S325 * _S325 * i2_0 + _S326 * _S326 * i1_0);
                nc_sum_0 = ki_0 * 36.0f * _S327;
                m1_0 = _S328;
                m2_0 = _S329;
                energy_2 = _S330;
            }
            else
            {
                var i_4 : u32 = u32(0);
                var nc_sum_1 : f32 = 0.0f;
                var m1_1 : f32 = 0.0f;
                var m2_1 : f32 = 0.0f;
                var energy_3 : f32 = 0.0f;
                loop
                {
                    if(i_4 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var _S331 : f32 = SPRING_AT_0[i_4] * w0_2;
                    var j_0 : u32 = u32(0);
                    nc_sum_0 = nc_sum_1;
                    m1_0 = m1_1;
                    m2_0 = m2_1;
                    energy_2 = energy_3;
                    loop
                    {
                        if(j_0 < u32(6))
                        {
                        }
                        else
                        {
                            break;
                        }
                        var s2_0 : f32 = SPRING_AT_0[j_0] * w1_2;
                        var di_0 : f32 = _S327 + _S325 * s2_0 - _S326 * _S331;
                        if(di_0 < 0.0f)
                        {
                            var f_14 : f32 = ki_0 * di_0;
                            var m1_2 : f32 = m1_0 + f_14 * s2_0;
                            var m2_2 : f32 = m2_0 - f_14 * _S331;
                            var energy_4 : f32 = energy_2 + 0.5f * ki_0 * di_0 * di_0;
                            nc_sum_0 = nc_sum_0 + f_14;
                            m1_0 = m1_2;
                            m2_0 = m2_2;
                            energy_2 = energy_4;
                        }
                        j_0 = j_0 + u32(1);
                    }
                    i_4 = i_4 + u32(1);
                    nc_sum_1 = nc_sum_0;
                    m1_1 = m1_0;
                    m2_1 = m2_0;
                    energy_3 = energy_2;
                }
                nc_sum_0 = nc_sum_1;
                m1_0 = m1_1;
                m2_0 = m2_1;
                energy_2 = energy_3;
            }
        }
    }
    return vec4<f32>(nc_sum_0, m1_0, m2_0, energy_2);
}

fn damage_law_0( kind_2 : u32,  kappa_1 : f32,  r_9 : f32) -> f32
{
    if(kappa_1 <= 1.0f)
    {
        return 0.0f;
    }
    if(kind_2 == u32(0))
    {
        if(r_9 <= 1.0f)
        {
            return 1.0f;
        }
        return min(fdiv_0(r_9 * (kappa_1 - 1.0f), kappa_1 * (r_9 - 1.0f)), 1.0f);
    }
    if(kappa_1 >= (0.5f * (r_9 + 1.0f)))
    {
        return 1.0f;
    }
    return 1.0f - fdiv_0(1.0f, kappa_1);
}

fn damage_increment_0( kind_3 : u32,  kappa_old_0 : f32,  lambda_0 : f32,  r_10 : f32,  d_old_0 : f32,  psi_0 : f32) -> vec2<f32>
{
    var _S332 : f32 = max(damage_law_0(kind_3, lambda_0, r_10), d_old_0);
    var _S333 : bool;
    if(_S332 <= d_old_0)
    {
        _S333 = true;
    }
    else
    {
        _S333 = d_old_0 >= 1.0f;
    }
    if(_S333)
    {
        return vec2<f32>(d_old_0, 0.0f);
    }
    var u0_0 : f32 = fdiv_0(psi_0, lambda_0 * lambda_0);
    var _S334 : f32 = max(kappa_old_0, 1.0f);
    if(kind_3 == u32(0))
    {
        if(r_10 > 1.0f)
        {
            return vec2<f32>(_S332, fdiv_0(u0_0 * r_10, r_10 - 1.0f) * max(min(lambda_0, r_10) - min(_S334, r_10), 0.0f));
        }
        return vec2<f32>(_S332, (1.0f - d_old_0) * psi_0);
    }
    var ku_0 : f32 = 0.5f * (r_10 + 1.0f);
    var plateau_0 : f32 = u0_0 * max(min(lambda_0, ku_0) - min(_S334, ku_0), 0.0f);
    var snap_0 : f32;
    if(_S332 >= 1.0f)
    {
        snap_0 = u0_0 * ku_0;
    }
    else
    {
        snap_0 = 0.0f;
    }
    return vec2<f32>(_S332, plateau_0 + snap_0);
}

struct Contact_0
{
     q_lin_1 : vec3<f32>,
     q_ang_1 : vec3<f32>,
     energy_5 : f32,
     diss_1 : f32,
     plastic_0 : vec3<f32>,
};

fn signum_0( x_11 : f32) -> f32
{
    var _S335 : f32;
    if((((bitcast<u32>((x_11))) & (u32(2147483648)))) != u32(0))
    {
        _S335 = -1.0f;
    }
    else
    {
        _S335 = 1.0f;
    }
    return _S335;
}

fn return_map_0( k_14 : f32,  total_0 : f32,  plastic_1 : f32,  cap_0 : f32) -> vec2<f32>
{
    var trial_0 : f32 = k_14 * (total_0 - plastic_1);
    if((abs(trial_0)) <= cap_0)
    {
        return vec2<f32>(trial_0, 0.0f);
    }
    var f_15 : f32 = cap_0 * signum_0(trial_0);
    return vec2<f32>(f_15, fdiv_0(trial_0 - f_15, k_14));
}

fn contact_part_0( mat_5 : ptr<function, JointMaterial_std140_0>,  b_20 : ptr<function, JointBond_std430_0>,  crush_3 : f32,  plastic_2 : vec3<f32>,  d_lin_1 : vec3<f32>,  d_ang_1 : vec3<f32>) -> Contact_0
{
    var c_25 : Contact_0;
    var _S336 : vec3<f32> = vec3<f32>(0.0f);
    c_25.q_lin_1 = _S336;
    c_25.q_ang_1 = _S336;
    c_25.energy_5 = 0.0f;
    c_25.diss_1 = 0.0f;
    c_25.plastic_0 = plastic_2;
    if(((((*mat_5).kind_flags_0.y) & (u32(2)))) == u32(0))
    {
        return c_25;
    }
    var ks_0 : f32 = (*b_20).stiff0_0.y;
    var kt_0 : f32 = (*b_20).stiff1_0.x;
    var _S337 : vec4<f32> = contact_normal_0(&((*mat_5)), &((*b_20)), crush_3, d_lin_1, d_ang_1);
    var nc_sum_2 : f32 = _S337.x;
    var m1_3 : f32 = _S337.y;
    var m2_3 : f32 = _S337.z;
    var energy_6 : f32 = _S337.w;
    var nc_0 : f32 = - nc_sum_2;
    var p_3 : vec3<f32> = plastic_2;
    c_25.q_lin_1 = vec3<f32>(0.0f, 0.0f, nc_sum_2);
    c_25.q_ang_1 = vec3<f32>(m1_3, m2_3, 0.0f);
    var slide_cap_0 : f32 = (*mat_5).strength_0.w * nc_0;
    var _S338 : f32 = ks_0 * (d_lin_1.x - plastic_2.x);
    var _S339 : f32 = ks_0 * (d_lin_1.y - plastic_2.y);
    var tn_0 : f32 = fsqrt_0(_S338 * _S338 + _S339 * _S339);
    var _S340 : bool;
    if(tn_0 > slide_cap_0)
    {
        _S340 = tn_0 > 0.0f;
    }
    else
    {
        _S340 = false;
    }
    var diss_2 : f32;
    if(_S340)
    {
        var _S341 : f32 = fdiv_0(_S338, tn_0);
        var _S342 : f32 = fdiv_0(_S339, tn_0);
        var dslip_0 : f32 = fdiv_0(tn_0 - slide_cap_0, ks_0);
        p_3[i32(0)] = p_3[i32(0)] + _S341 * dslip_0;
        p_3[i32(1)] = p_3[i32(1)] + _S342 * dslip_0;
        c_25.q_lin_1[i32(0)] = _S341 * slide_cap_0;
        c_25.q_lin_1[i32(1)] = _S342 * slide_cap_0;
        diss_2 = slide_cap_0 * dslip_0;
    }
    else
    {
        c_25.q_lin_1[i32(0)] = _S338;
        c_25.q_lin_1[i32(1)] = _S339;
        diss_2 = 0.0f;
    }
    var tq_0 : vec2<f32> = return_map_0(kt_0, d_ang_1.z, p_3.z, slide_cap_0 * (*b_20).geom1_0.z);
    var _S343 : f32 = tq_0.x;
    var _S344 : f32 = tq_0.y;
    var diss_3 : f32 = diss_2 + abs(_S343) * abs(_S344);
    p_3[i32(2)] = p_3[i32(2)] + _S344;
    c_25.q_ang_1[i32(2)] = _S343;
    c_25.energy_5 = energy_6 + 0.5f * (fdiv_0(sq_0(c_25.q_lin_1.x), ks_0) + fdiv_0(sq_0(c_25.q_lin_1.y), ks_0) + fdiv_0(sq_0(_S343), kt_0));
    c_25.diss_1 = diss_3;
    c_25.plastic_0 = p_3;
    return c_25;
}

fn contact_offsets_0( mat_6 : ptr<function, JointMaterial_std140_0>,  b_21 : ptr<function, JointBond_std430_0>,  crush_4 : f32,  plastic_3 : vec3<f32>,  d_lin_2 : vec3<f32>,  d_ang_2 : vec3<f32>) -> vec3<f32>
{
    var _S345 : u32 = (*mat_6).kind_flags_0.y;
    if(((_S345 & (u32(2)))) == u32(0))
    {
        return plastic_3;
    }
    var kn_2 : f32 = (*b_21).stiff0_0.x;
    var ks_1 : f32 = (*b_21).stiff0_0.y;
    var kt_1 : f32 = (*b_21).stiff1_0.x;
    var w0_3 : f32 = (*b_21).geom0_0.y;
    var w1_3 : f32 = (*b_21).geom0_0.z;
    var nc_sum_3 : f32;
    if(((_S345 & (u32(4)))) != u32(0))
    {
        var _S346 : vec4<f32> = no_tension_patch_0(kn_2 * (1.0f - crush_4), w0_3, w1_3, d_lin_2.z, d_ang_2.x, d_ang_2.y);
        nc_sum_3 = _S346.x;
    }
    else
    {
        var ki_1 : f32 = kn_2 * (1.0f - crush_4) / 36.0f;
        var _S347 : f32 = d_ang_2.x;
        var _S348 : f32 = d_ang_2.y;
        var spread_1 : f32 = abs(_S347) * 0.4166666567325592f * w1_3 + abs(_S348) * 0.4166666567325592f * w0_3;
        var _S349 : f32 = d_lin_2.z;
        var slack_1 : f32 = 9.99999997475242708e-07f * (abs(_S349) + spread_1);
        if((_S349 - spread_1) > slack_1)
        {
            nc_sum_3 = 0.0f;
        }
        else
        {
            if((_S349 + spread_1) < (- slack_1))
            {
                nc_sum_3 = ki_1 * 36.0f * _S349;
            }
            else
            {
                var i_5 : u32 = u32(0);
                var nc_sum_4 : f32 = 0.0f;
                loop
                {
                    if(i_5 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var _S350 : f32 = SPRING_AT_0[i_5] * w0_3;
                    var j_1 : u32 = u32(0);
                    nc_sum_3 = nc_sum_4;
                    loop
                    {
                        if(j_1 < u32(6))
                        {
                        }
                        else
                        {
                            break;
                        }
                        var di_1 : f32 = _S349 + _S347 * (SPRING_AT_0[j_1] * w1_3) - _S348 * _S350;
                        if(di_1 < 0.0f)
                        {
                            nc_sum_3 = nc_sum_3 + ki_1 * di_1;
                        }
                        j_1 = j_1 + u32(1);
                    }
                    i_5 = i_5 + u32(1);
                    nc_sum_4 = nc_sum_3;
                }
                nc_sum_3 = nc_sum_4;
            }
        }
    }
    var nc_1 : f32 = - nc_sum_3;
    var p_4 : vec3<f32> = plastic_3;
    var slide_cap_1 : f32 = (*mat_6).strength_0.w * nc_1;
    var _S351 : f32 = ks_1 * (d_lin_2.x - plastic_3.x);
    var _S352 : f32 = ks_1 * (d_lin_2.y - plastic_3.y);
    var tn_1 : f32 = fsqrt_0(_S351 * _S351 + _S352 * _S352);
    var _S353 : bool;
    if(tn_1 > slide_cap_1)
    {
        _S353 = tn_1 > 0.0f;
    }
    else
    {
        _S353 = false;
    }
    if(_S353)
    {
        var _S354 : f32 = fdiv_0(_S352, tn_1);
        var dslip_1 : f32 = fdiv_0(tn_1 - slide_cap_1, ks_1);
        p_4[i32(0)] = p_4[i32(0)] + fdiv_0(_S351, tn_1) * dslip_1;
        p_4[i32(1)] = p_4[i32(1)] + _S354 * dslip_1;
    }
    p_4[i32(2)] = p_4[i32(2)] + return_map_0(kt_1, d_ang_2.z, p_4.z, slide_cap_1 * (*b_21).geom1_0.z).y;
    return p_4;
}

fn life_rate_0( mat_7 : ptr<function, JointMaterial_std140_0>,  s_6 : f32) -> f32
{
    if(s_6 <= 0.0f)
    {
        return 0.0f;
    }
    var _S355 : f32 = (*mat_7).misc_0.y;
    return fdiv_0((_S355 + 1.0f) * fpow_0(s_6, _S355), (*mat_7).misc_0.z);
}

fn connected_2( st_3 : JointState_0,  has_rebar_2 : bool) -> bool
{
    var _S356 : bool;
    if((st_3.damage_0) < 1.0f)
    {
        _S356 = true;
    }
    else
    {
        if(has_rebar_2)
        {
            _S356 = (st_3.rebar_broken_0) == 0.0f;
        }
        else
        {
            _S356 = false;
        }
    }
    return _S356;
}

fn joint_evaluate_0( mat_8 : ptr<function, JointMaterial_std140_0>,  b_22 : ptr<function, JointBond_std430_0>,  state_2 : ptr<function, JointState_std430_0>,  d_lin_3 : vec3<f32>,  d_ang_3 : vec3<f32>,  dt_13 : f32,  fracture_1 : bool) -> JointResponse_0
{
    var kn_3 : f32 = (*b_22).stiff0_0.x;
    var ks_2 : f32 = (*b_22).stiff0_0.y;
    var kb1_0 : f32 = (*b_22).stiff0_0.z;
    var kb2_0 : f32 = (*b_22).stiff0_0.w;
    var _S357 : vec4<f32> = (*b_22).stiff1_0;
    var kt_2 : f32 = (*b_22).stiff1_0.x;
    var has_rebar_3 : bool = ((*b_22).stiff1_0.w) != 0.0f;
    var kind_4 : u32 = (*mat_8).kind_flags_0.x;
    var flags_0 : u32 = (*mat_8).kind_flags_0.y;
    var softening_0 : bool = ((flags_0 & (u32(1)))) != u32(0);
    var st_4 : JointState_0;
    st_4.damage_0 = (*state_2).damage_0;
    st_4.crush_1 = (*state_2).crush_1;
    st_4.kappa_0 = (*state_2).kappa_0;
    st_4.kappa_c_0 = (*state_2).kappa_c_0;
    st_4.ductility_0 = (*state_2).ductility_0;
    st_4.ductility_c_0 = (*state_2).ductility_c_0;
    st_4.life_0 = (*state_2).life_0;
    st_4.plastic_x_0 = (*state_2).plastic_x_0;
    st_4.plastic_y_0 = (*state_2).plastic_y_0;
    st_4.plastic_t_0 = (*state_2).plastic_t_0;
    st_4.rebar_plastic_0 = (*state_2).rebar_plastic_0;
    st_4.rebar_slip0_0 = (*state_2).rebar_slip0_0;
    st_4.rebar_slip1_0 = (*state_2).rebar_slip1_0;
    st_4.rebar_work_0 = (*state_2).rebar_work_0;
    st_4.rebar_broken_0 = (*state_2).rebar_broken_0;
    st_4.strain_rate_0 = (*state_2).strain_rate_0;
    st_4.governing_stress_0 = (*state_2).governing_stress_0;
    st_4.dissipated_0 = (*state_2).dissipated_0;
    st_4.utilization_0 = (*state_2).utilization_0;
    st_4.mode_0 = (*state_2).mode_0;
    var _S358 : bool = connected_1(&((*state_2)), has_rebar_3);
    var qe_lin_0 : vec3<f32> = d_lin_3 * vec3<f32>(ks_2, ks_2, kn_3);
    var qe_ang_0 : vec3<f32> = d_ang_3 * vec3<f32>(kb1_0, kb2_0, kt_2);
    var _S359 : Measures_0 = stress_measures_0(&((*b_22)), qe_lin_0, qe_ang_0);
    var _S360 : f32 = max(max(_S359.tension_0, _S359.shear_0), _S359.compression_0);
    var _S361 : bool = dt_13 > 0.0f;
    var dif_1 : f32;
    if(_S361)
    {
        var raw_0 : f32 = fdiv_0(max(fdiv_0(_S360 - st_4.governing_stress_0, dt_13), 0.0f), (*mat_8).misc_0.w);
        var tau_2 : f32 = _S357.z;
        if(((flags_0 & (u32(16)))) != u32(0))
        {
            dif_1 = - expm1_accurate_0(- fdiv_0(dt_13, tau_2));
        }
        else
        {
            dif_1 = min(fdiv_0(dt_13, tau_2), 1.0f);
        }
        st_4.strain_rate_0 = st_4.strain_rate_0 + (raw_0 - st_4.strain_rate_0) * dif_1;
        st_4.governing_stress_0 = _S360;
    }
    if(((flags_0 & (u32(32)))) != u32(0))
    {
        var _S362 : f32 = dif_factor_0(&((*mat_8)), st_4.strain_rate_0);
        dif_1 = _S362;
    }
    else
    {
        dif_1 = 1.0f;
    }
    var weibull_0 : f32 = (*b_22).geom1_0.w;
    var _S363 : f32 = fatigue_factor_0(&((*mat_8)), st_4.life_0);
    var multiplier_1 : f32 = weibull_0 * dif_1 * _S363;
    var _S364 : vec4<f32> = failure_indices_0(&((*mat_8)), &((*b_22)), _S359, multiplier_1);
    var _S365 : f32 = _S364.x;
    var _S366 : f32 = _S364.y;
    st_4.utilization_0 = max(max(_S365, _S366), max(_S364.z, _S364.w));
    var _S367 : f32 = d_lin_3.x;
    var _S368 : f32 = d_lin_3.y;
    var _S369 : f32 = d_ang_3.z;
    var _S370 : f32 = ks_2 * (sq_0(_S367) + sq_0(_S368)) + kb1_0 * sq_0(d_ang_3.x) + kb2_0 * sq_0(d_ang_3.y) + kt_2 * sq_0(_S369);
    var _S371 : f32 = d_lin_3.z;
    var _S372 : bool = _S371 > 0.0f;
    if(_S372)
    {
        dif_1 = kn_3 * sq_0(_S371);
    }
    else
    {
        dif_1 = 0.0f;
    }
    var psi_ts_0 : f32 = 0.5f * (_S370 + dif_1);
    var psi_c_0 : f32;
    if(_S371 < 0.0f)
    {
        psi_c_0 = 0.5f * kn_3 * sq_0(_S371);
    }
    else
    {
        psi_c_0 = 0.0f;
    }
    var plastic_4 : vec3<f32> = vec3<f32>(st_4.plastic_x_0, st_4.plastic_y_0, st_4.plastic_t_0);
    var changed_0 : bool;
    if(((flags_0 & (u32(2)))) != u32(0))
    {
        changed_0 = ((*state_2).damage_0) != 0.0f;
    }
    else
    {
        changed_0 = false;
    }
    var patch_0 : f32;
    if(changed_0)
    {
        var _S373 : vec4<f32> = contact_normal_0(&((*mat_8)), &((*b_22)), (*state_2).crush_1, d_lin_3, d_ang_3);
        patch_0 = _S373.w + 0.5f * (ks_2 * (sq_0(_S367 - (*state_2).plastic_x_0) + sq_0(_S368 - (*state_2).plastic_y_0)) + kt_2 * sq_0(_S369 - (*state_2).plastic_t_0));
    }
    else
    {
        patch_0 = 0.0f;
    }
    if(has_rebar_3)
    {
        changed_0 = ((*state_2).rebar_broken_0) == 0.0f;
    }
    else
    {
        changed_0 = false;
    }
    var rebar_0 : f32;
    if(changed_0)
    {
        rebar_0 = 0.5f * ((*b_22).rebar0_0.x * sq_0(_S371 - (*state_2).rebar_plastic_0) + (*b_22).rebar0_0.y * (sq_0(_S367 - (*state_2).rebar_slip0_0) + sq_0(_S368 - (*state_2).rebar_slip1_0)));
    }
    else
    {
        rebar_0 = 0.0f;
    }
    var _S374 : f32 = (*state_2).damage_0;
    var _S375 : f32 = 1.0f - (*state_2).damage_0;
    var _S376 : f32 = (*state_2).crush_1;
    var frozen_0 : f32 = _S375 * (psi_ts_0 + (1.0f - (*state_2).crush_1) * psi_c_0) + (*state_2).damage_0 * patch_0 + rebar_0;
    var diss_contact_0 : f32;
    var psi_contact_0 : f32;
    var intact_normal_0 : f32;
    var dissipated_3 : f32;
    var qc_lin_0 : vec3<f32>;
    if(fracture_1)
    {
        var _S377 : bool = _S365 >= _S366;
        if(_S377)
        {
            diss_contact_0 = _S365;
        }
        else
        {
            diss_contact_0 = _S366;
        }
        var mode_ts_0 : u32;
        if(_S377)
        {
            mode_ts_0 = u32(1);
        }
        else
        {
            mode_ts_0 = u32(2);
        }
        if(diss_contact_0 > (st_4.kappa_0))
        {
            changed_0 = diss_contact_0 > 1.0f;
        }
        else
        {
            changed_0 = false;
        }
        if(changed_0)
        {
            changed_0 = psi_ts_0 > 0.0f;
        }
        else
        {
            changed_0 = false;
        }
        var mode_c_0 : u32;
        if(changed_0)
        {
            if(mode_ts_0 == u32(1))
            {
                psi_contact_0 = (*mat_8).energy_1.y;
            }
            else
            {
                psi_contact_0 = (*mat_8).energy_1.z;
            }
            if(softening_0)
            {
                intact_normal_0 = fdiv_0(psi_contact_0 * (*b_22).geom0_0.x * diss_contact_0 * diss_contact_0, psi_ts_0);
            }
            else
            {
                intact_normal_0 = 0.0f;
            }
            st_4.ductility_0 = intact_normal_0;
            if(softening_0)
            {
                mode_c_0 = kind_4;
            }
            else
            {
                mode_c_0 = u32(0);
            }
            var inc_0 : vec2<f32> = damage_increment_0(mode_c_0, st_4.kappa_0, diss_contact_0, intact_normal_0, st_4.damage_0, psi_ts_0);
            var _S378 : f32 = inc_0.x;
            if(_S378 > (st_4.damage_0))
            {
                var _S379 : Contact_0 = contact_part_0(&((*mat_8)), &((*b_22)), st_4.crush_1, plastic_4, d_lin_3, d_ang_3);
                var _S380 : f32 = max(inc_0.y - max(_S379.energy_5 - (1.0f - st_4.crush_1) * psi_c_0, 0.0f) * (_S378 - st_4.damage_0), 0.0f);
                st_4.damage_0 = _S378;
                st_4.mode_0 = mode_ts_0;
                dissipated_3 = _S380;
            }
            else
            {
                dissipated_3 = 0.0f;
            }
        }
        else
        {
            dissipated_3 = 0.0f;
        }
        st_4.kappa_0 = max(st_4.kappa_0, diss_contact_0);
        if(_S374 > 0.0f)
        {
            var _S381 : Contact_0 = contact_part_0(&((*mat_8)), &((*b_22)), _S376, vec3<f32>((*state_2).plastic_x_0, (*state_2).plastic_y_0, (*state_2).plastic_t_0), d_lin_3, d_ang_3);
            qc_lin_0 = qe_ang_0 * vec3<f32>(_S375) + _S381.q_ang_1 * vec3<f32>(_S374);
        }
        else
        {
            qc_lin_0 = qe_ang_0;
        }
        var _S382 : Measures_0 = stress_measures_0(&((*b_22)), vec3<f32>(0.0f, 0.0f, min(qe_lin_0.z, 0.0f)), qc_lin_0);
        var _S383 : vec4<f32> = failure_indices_0(&((*mat_8)), &((*b_22)), _S382, multiplier_1);
        var _S384 : f32 = _S383.z;
        var _S385 : f32 = _S383.w;
        var _S386 : bool = _S384 >= _S385;
        if(_S386)
        {
            psi_contact_0 = _S384;
        }
        else
        {
            psi_contact_0 = _S385;
        }
        if(_S386)
        {
            mode_c_0 = u32(3);
        }
        else
        {
            mode_c_0 = u32(4);
        }
        if(psi_contact_0 > (st_4.kappa_c_0))
        {
            changed_0 = psi_contact_0 > 1.0f;
        }
        else
        {
            changed_0 = false;
        }
        if(changed_0)
        {
            changed_0 = psi_c_0 > 0.0f;
        }
        else
        {
            changed_0 = false;
        }
        if(changed_0)
        {
            if(softening_0)
            {
                intact_normal_0 = fdiv_0((*mat_8).energy_1.w * (*b_22).geom0_0.x * psi_contact_0 * psi_contact_0, psi_c_0);
            }
            else
            {
                intact_normal_0 = 0.0f;
            }
            st_4.ductility_c_0 = intact_normal_0;
            var law_1 : u32;
            if(!softening_0)
            {
                law_1 = u32(0);
            }
            else
            {
                if(mode_c_0 == u32(4))
                {
                    mode_ts_0 = u32(1);
                }
                else
                {
                    mode_ts_0 = kind_4;
                }
                law_1 = mode_ts_0;
            }
            var inc_1 : vec2<f32> = damage_increment_0(law_1, st_4.kappa_c_0, psi_contact_0, intact_normal_0, st_4.crush_1, psi_c_0);
            var _S387 : f32 = inc_1.x;
            if(_S387 > (st_4.crush_1))
            {
                var dissipated_4 : f32 = dissipated_3 + inc_1.y;
                st_4.crush_1 = _S387;
                st_4.mode_0 = mode_c_0;
                if(_S387 >= 1.0f)
                {
                    changed_0 = (st_4.damage_0) < 1.0f;
                }
                else
                {
                    changed_0 = false;
                }
                if(changed_0)
                {
                    var dissipated_5 : f32 = dissipated_4 + psi_ts_0 * (1.0f - st_4.damage_0);
                    st_4.damage_0 = 1.0f;
                    dissipated_3 = dissipated_5;
                }
                else
                {
                    dissipated_3 = dissipated_4;
                }
            }
        }
        st_4.kappa_c_0 = max(st_4.kappa_c_0, psi_contact_0);
    }
    else
    {
        dissipated_3 = 0.0f;
    }
    var dmg_0 : f32 = st_4.damage_0;
    var _S388 : vec3<f32> = vec3<f32>(0.0f);
    var qc_ang_0 : vec3<f32>;
    if((st_4.damage_0) == 0.0f)
    {
        if(((flags_0 & (u32(8)))) == u32(0))
        {
            var _S389 : vec3<f32> = contact_offsets_0(&((*mat_8)), &((*b_22)), st_4.crush_1, plastic_4, d_lin_3, d_ang_3);
            st_4.plastic_x_0 = _S389.x;
            st_4.plastic_y_0 = _S389.y;
            st_4.plastic_t_0 = _S389.z;
        }
        diss_contact_0 = 0.0f;
        qc_lin_0 = _S388;
        qc_ang_0 = _S388;
        psi_contact_0 = 0.0f;
    }
    else
    {
        var _S390 : Contact_0 = contact_part_0(&((*mat_8)), &((*b_22)), st_4.crush_1, plastic_4, d_lin_3, d_ang_3);
        st_4.plastic_x_0 = _S390.plastic_0.x;
        st_4.plastic_y_0 = _S390.plastic_0.y;
        st_4.plastic_t_0 = _S390.plastic_0.z;
        diss_contact_0 = _S390.diss_1;
        qc_lin_0 = _S390.q_lin_1;
        qc_ang_0 = _S390.q_ang_1;
        psi_contact_0 = _S390.energy_5;
    }
    var dissipated_6 : f32 = dissipated_3 + dmg_0 * diss_contact_0;
    if(_S372)
    {
        intact_normal_0 = kn_3 * _S371;
    }
    else
    {
        intact_normal_0 = (1.0f - st_4.crush_1) * kn_3 * _S371;
    }
    var _S391 : f32 = 1.0f - dmg_0;
    var force_lin_2 : vec3<f32> = vec3<f32>(_S391 * qe_lin_0.x + dmg_0 * qc_lin_0.x, _S391 * qe_lin_0.y + dmg_0 * qc_lin_0.y, _S391 * intact_normal_0 + dmg_0 * qc_lin_0.z);
    var force_ang_2 : vec3<f32> = qe_ang_0 * vec3<f32>(_S391) + qc_ang_0 * vec3<f32>(dmg_0);
    var stored_3 : f32 = _S391 * (psi_ts_0 + (1.0f - st_4.crush_1) * psi_c_0) + dmg_0 * psi_contact_0;
    if(has_rebar_3)
    {
        changed_0 = (st_4.rebar_broken_0) == 0.0f;
    }
    else
    {
        changed_0 = false;
    }
    var stored_4 : f32;
    var force_lin_3 : vec3<f32>;
    if(changed_0)
    {
        var k_axial_0 : f32 = (*b_22).rebar0_0.x;
        var k_dowel_0 : f32 = (*b_22).rebar0_0.y;
        var yield_force_0 : f32 = (*b_22).rebar0_0.z;
        var dowel_capacity_0 : f32 = (*b_22).rebar0_0.w;
        var nr_0 : vec2<f32> = return_map_0(k_axial_0, _S371, st_4.rebar_plastic_0, yield_force_0);
        var v1_0 : vec2<f32> = return_map_0(k_dowel_0, _S367, st_4.rebar_slip0_0, dowel_capacity_0);
        var v2_0 : vec2<f32> = return_map_0(k_dowel_0, _S368, st_4.rebar_slip1_0, dowel_capacity_0);
        var _S392 : f32 = nr_0.y;
        var _S393 : f32 = v1_0.y;
        var _S394 : f32 = v2_0.y;
        var work_4 : f32 = yield_force_0 * abs(_S392) + dowel_capacity_0 * (abs(_S393) + abs(_S394));
        st_4.rebar_plastic_0 = st_4.rebar_plastic_0 + _S392;
        st_4.rebar_slip0_0 = st_4.rebar_slip0_0 + _S393;
        st_4.rebar_slip1_0 = st_4.rebar_slip1_0 + _S394;
        st_4.rebar_work_0 = st_4.rebar_work_0 + work_4;
        var dissipated_7 : f32 = dissipated_6 + work_4;
        var _S395 : f32 = nr_0.x;
        var _S396 : f32 = v1_0.x;
        var _S397 : f32 = v2_0.x;
        var elastic_0 : f32 = 0.5f * (fdiv_0(sq_0(_S395), k_axial_0) + fdiv_0(sq_0(_S396) + sq_0(_S397), k_dowel_0));
        if(fracture_1)
        {
            changed_0 = (st_4.rebar_work_0) >= ((*b_22).rebar1_0.x);
        }
        else
        {
            changed_0 = false;
        }
        if(changed_0)
        {
            st_4.rebar_broken_0 = 1.0f;
            var dissipated_8 : f32 = dissipated_7 + elastic_0;
            force_lin_3 = force_lin_2;
            stored_4 = stored_3;
            dissipated_3 = dissipated_8;
        }
        else
        {
            var stored_5 : f32 = stored_3 + elastic_0;
            force_lin_3 = force_lin_2 + vec3<f32>(_S396, _S397, _S395);
            stored_4 = stored_5;
            dissipated_3 = dissipated_7;
        }
    }
    else
    {
        force_lin_3 = force_lin_2;
        stored_4 = stored_3;
        dissipated_3 = dissipated_6;
    }
    if(fracture_1)
    {
        changed_0 = _S361;
    }
    else
    {
        changed_0 = false;
    }
    if(changed_0)
    {
        changed_0 = ((flags_0 & (u32(64)))) != u32(0);
    }
    else
    {
        changed_0 = false;
    }
    if(changed_0)
    {
        var _S398 : Measures_0 = stress_measures_0(&((*b_22)), force_lin_3, force_ang_2);
        var _S399 : vec4<f32> = failure_indices_0(&((*mat_8)), &((*b_22)), _S398, weibull_0);
        var _S400 : f32 = life_rate_0(&((*mat_8)), max(max(_S399.x, _S399.y), _S399.z));
        st_4.life_0 = max(st_4.life_0 - _S400 * dt_13, 0.0f);
    }
    if((st_4.damage_0) != _S374)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_4.crush_1) != _S376;
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_4.plastic_x_0) != ((*state_2).plastic_x_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_4.plastic_y_0) != ((*state_2).plastic_y_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_4.plastic_t_0) != ((*state_2).plastic_t_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_4.rebar_plastic_0) != ((*state_2).rebar_plastic_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_4.rebar_slip0_0) != ((*state_2).rebar_slip0_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_4.rebar_slip1_0) != ((*state_2).rebar_slip1_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_4.rebar_broken_0) != ((*state_2).rebar_broken_0);
    }
    var released_0 : f32;
    if(changed_0)
    {
        released_0 = frozen_0 - stored_4;
    }
    else
    {
        released_0 = 0.0f;
    }
    var _S401 : f32 = max(min(dissipated_3, released_0), 0.0f);
    var overshoot_1 : f32 = released_0 - _S401;
    st_4.dissipated_0 = st_4.dissipated_0 + _S401;
    var resp_0 : JointResponse_0;
    resp_0.force_lin_1 = force_lin_3;
    resp_0.force_ang_1 = force_ang_2;
    resp_0.state_1 = st_4;
    resp_0.dissipated_2 = _S401;
    resp_0.overshoot_0 = overshoot_1;
    resp_0.stored_2 = stored_4;
    if(_S358)
    {
        changed_0 = !connected_2(st_4, has_rebar_3);
    }
    else
    {
        changed_0 = false;
    }
    resp_0.disconnected_0 = changed_0;
    resp_0.fatigue_before_0 = _S363;
    resp_0.measures_0 = _S359;
    return resp_0;
}

struct BondDyn_0
{
     js_0 : JointState_0,
     force_lin_0 : vec4<f32>,
     force_ang_0 : vec4<f32>,
     sums_0 : vec4<f32>,
     comps_0 : vec4<f32>,
     events_0 : vec4<u32>,
};

fn secant_factors_0( b_23 : ptr<function, JointBond_std430_0>,  st_5 : JointState_0,  d_lin_4 : vec3<f32>,  f_lin_0 : ptr<function, vec3<f32>>,  f_ang_0 : ptr<function, vec3<f32>>)
{
    var compressed_0 : bool = (d_lin_4.z) < 0.0f;
    var contact_5 : f32;
    if(compressed_0)
    {
        contact_5 = st_5.damage_0;
    }
    else
    {
        contact_5 = 0.0f;
    }
    var _S402 : f32 = 1.0f - st_5.damage_0;
    var _S403 : f32 = max(_S402 + contact_5, 9.99999997475242708e-07f);
    var normal_2 : f32;
    if(compressed_0)
    {
        normal_2 = max(1.0f - st_5.crush_1, 9.99999997475242708e-07f);
    }
    else
    {
        normal_2 = max(_S402, 9.99999997475242708e-07f);
    }
    (*f_lin_0) = vec3<f32>(_S403, _S403, normal_2);
    (*f_ang_0) = vec3<f32>(_S403);
    var _S404 : bool;
    if(((*b_23).stiff1_0.w) != 0.0f)
    {
        _S404 = (st_5.rebar_broken_0) == 0.0f;
    }
    else
    {
        _S404 = false;
    }
    if(_S404)
    {
        var _S405 : vec4<f32> = (*b_23).rebar0_0;
        var _S406 : vec4<f32> = (*b_23).stiff0_0;
        (*f_lin_0)[i32(2)] = (*f_lin_0)[i32(2)] + fdiv_0((*b_23).rebar0_0.x, (*b_23).stiff0_0.x);
        var _S407 : f32 = fdiv_0(_S405.y, _S406.y);
        (*f_lin_0)[i32(0)] = (*f_lin_0)[i32(0)] + _S407;
        (*f_lin_0)[i32(1)] = (*f_lin_0)[i32(1)] + _S407;
    }
    return;
}

fn to_body_0( _S408 : u32,  _S409 : vec3<f32>) -> vec3<f32>
{
    return bonds_0[_S408].t1_0.xyz * vec3<f32>(_S409.x) + bonds_0[_S408].t2_0.xyz * vec3<f32>(_S409.y) + bonds_0[_S408].normal_0.xyz * vec3<f32>(_S409.z);
}

fn comp_add1_3( sum_6 : ptr<function, f32>,  err_5 : ptr<function, f32>,  x_12 : f32)
{
    var t_18 : f32 = (*sum_6) + x_12;
    if((abs((*sum_6))) >= (abs(x_12)))
    {
        (*err_5) = (*err_5) + ((*sum_6) - t_18 + x_12);
    }
    else
    {
        (*err_5) = (*err_5) + (x_12 - t_18 + (*sum_6));
    }
    (*sum_6) = t_18;
    return;
}

fn is_damaged_1( st_6 : JointState_0) -> bool
{
    var _S410 : bool;
    if((st_6.damage_0) > 0.0f)
    {
        _S410 = true;
    }
    else
    {
        _S410 = (st_6.crush_1) > 0.0f;
    }
    return _S410;
}

fn fatigue_factor_1( mat_9 : ptr<function, JointMaterial_std140_0>,  life_2 : f32) -> f32
{
    if(((((*mat_9).kind_flags_0.y) & (u32(64)))) == u32(0))
    {
        return 1.0f;
    }
    return fpow_0(clamp(life_2, 0.0f, 1.0f), fdiv_0(1.0f, (*mat_9).misc_0.y - 2.0f));
}

fn bond_update_0( i_6 : u32,  dt_14 : f32,  fracture_2 : bool,  abs_step_0 : u32,  wide_0 : bool) -> bool
{
    var ra_1 : vec3<f32> = bonds_0[i_6].ra_0.xyz;
    var rb_1 : vec3<f32> = bonds_0[i_6].rb_0.xyz;
    var _S411 : u32 = u32(4) * bonds_0[i_6].law_0.ids_0.y;
    var ta_1 : vec3<f32> = state_0[_S411 + u32(1)].xyz;
    var wa_0 : vec3<f32> = state_0[_S411 + u32(3)].xyz;
    var _S412 : u32 = u32(4) * bonds_0[i_6].law_0.ids_0.z;
    var tb_1 : vec3<f32> = state_0[_S412 + u32(1)].xyz;
    var wb_2 : vec3<f32> = state_0[_S412 + u32(3)].xyz;
    var _S413 : vec3<f32> = to_local_0(i_6, state_0[_S412].xyz + cross(tb_1, rb_1) - (state_0[_S411].xyz + cross(ta_1, ra_1)));
    var _S414 : vec3<f32> = to_local_0(i_6, tb_1 - ta_1);
    var _S415 : vec3<f32> = to_local_0(i_6, state_0[_S412 + u32(2)].xyz + cross(wb_2, rb_1) - (state_0[_S411 + u32(2)].xyz + cross(wa_0, ra_1)));
    var _S416 : vec3<f32> = to_local_0(i_6, wb_2 - wa_0);
    var _S417 : JointMaterial_std140_0 = materials_0.m_0.data_0[bonds_0[i_6].law_0.ids_0.x];
    var _S418 : JointState_std430_0 = bond_dyn_0[i_6].js_0;
    var _S419 : bool = is_damaged_0(&(_S418));
    var _S420 : f32 = _S418.life_0;
    var _S421 : JointBond_std430_0 = bonds_0[i_6].law_0;
    var _S422 : JointResponse_0 = joint_evaluate_0(&(_S417), &(_S421), &(_S418), _S413, _S414, dt_14, fracture_2);
    var _S423 : JointState_0 = JointState_0( bond_dyn_0[i_6].js_0.damage_0, bond_dyn_0[i_6].js_0.crush_1, bond_dyn_0[i_6].js_0.kappa_0, bond_dyn_0[i_6].js_0.kappa_c_0, bond_dyn_0[i_6].js_0.ductility_0, bond_dyn_0[i_6].js_0.ductility_c_0, bond_dyn_0[i_6].js_0.life_0, bond_dyn_0[i_6].js_0.plastic_x_0, bond_dyn_0[i_6].js_0.plastic_y_0, bond_dyn_0[i_6].js_0.plastic_t_0, bond_dyn_0[i_6].js_0.rebar_plastic_0, bond_dyn_0[i_6].js_0.rebar_slip0_0, bond_dyn_0[i_6].js_0.rebar_slip1_0, bond_dyn_0[i_6].js_0.rebar_work_0, bond_dyn_0[i_6].js_0.rebar_broken_0, bond_dyn_0[i_6].js_0.strain_rate_0, bond_dyn_0[i_6].js_0.governing_stress_0, bond_dyn_0[i_6].js_0.dissipated_0, bond_dyn_0[i_6].js_0.utilization_0, bond_dyn_0[i_6].js_0.mode_0 );
    var bd_0 : BondDyn_0;
    bd_0.js_0 = _S423;
    bd_0.force_lin_0 = bond_dyn_0[i_6].force_lin_0;
    bd_0.force_ang_0 = bond_dyn_0[i_6].force_ang_0;
    bd_0.sums_0 = bond_dyn_0[i_6].sums_0;
    bd_0.comps_0 = bond_dyn_0[i_6].comps_0;
    bd_0.events_0 = bond_dyn_0[i_6].events_0;
    var ra_2 : vec3<f32> = bonds_0[i_6].ra_0.xyz;
    var rb_2 : vec3<f32> = bonds_0[i_6].rb_0.xyz;
    var _S424 : JointBond_std430_0 = bonds_0[i_6].law_0;
    var f_lin_1 : vec3<f32>;
    var f_ang_1 : vec3<f32>;
    secant_factors_0(&(_S424), _S422.state_1, _S413, &(f_lin_1), &(f_ang_1));
    var qd_lin_0 : vec3<f32> = _S415 * bonds_0[i_6].c_lin_0.xyz * f_lin_1;
    var qd_ang_0 : vec3<f32> = _S416 * bonds_0[i_6].c_ang_0.xyz * f_ang_1;
    var q_lin_2 : vec3<f32> = _S422.force_lin_1 + qd_lin_0;
    var q_ang_2 : vec3<f32> = _S422.force_ang_1 + qd_ang_0;
    var damped_0 : f32 = (dot(qd_lin_0, _S415) + dot(qd_ang_0, _S416)) * dt_14;
    var _S425 : vec3<f32> = to_body_0(i_6, q_lin_2);
    var _S426 : vec3<f32> = to_body_0(i_6, q_ang_2);
    var _S427 : u32 = u32(3) * i_6;
    scratch_0[_S427] = vec4<f32>(_S425, max(_S422.measures_0.tension_0, _S422.measures_0.compression_0));
    scratch_0[_S427 + u32(1)] = vec4<f32>(_S426 + cross(ra_2, _S425), 0.0f);
    scratch_0[_S427 + u32(2)] = vec4<f32>((vec3<f32>(0) - _S426) + cross(rb_2, (vec3<f32>(0) - _S425)), 0.0f);
    var _S428 : f32 = bd_0.sums_0[i32(0)];
    var _S429 : f32 = bd_0.comps_0[i32(0)];
    comp_add1_3(&(_S428), &(_S429), _S422.dissipated_2);
    bd_0.sums_0[i32(0)] = _S428;
    bd_0.comps_0[i32(0)] = _S429;
    var _S430 : f32 = bd_0.sums_0[i32(1)];
    var _S431 : f32 = bd_0.comps_0[i32(1)];
    comp_add1_3(&(_S430), &(_S431), _S422.overshoot_0);
    bd_0.sums_0[i32(1)] = _S430;
    bd_0.comps_0[i32(1)] = _S431;
    var _S432 : f32 = bd_0.sums_0[i32(2)];
    var _S433 : f32 = bd_0.comps_0[i32(2)];
    comp_add1_3(&(_S432), &(_S433), damped_0);
    bd_0.sums_0[i32(2)] = _S432;
    bd_0.comps_0[i32(2)] = _S433;
    bd_0.force_lin_0 = vec4<f32>(q_lin_2, _S422.stored_2);
    bd_0.force_ang_0 = vec4<f32>(q_ang_2, max(bd_0.force_ang_0.w, _S422.state_1.utilization_0));
    var _S434 : bool;
    if(!_S419)
    {
        _S434 = is_damaged_1(_S422.state_1);
    }
    else
    {
        _S434 = false;
    }
    if(_S434)
    {
        _S434 = (bd_0.events_0.x) == u32(0);
    }
    else
    {
        _S434 = false;
    }
    if(_S434)
    {
        bd_0.events_0[i32(0)] = abs_step_0;
        bd_0.events_0[i32(3)] = _S422.state_1.mode_0;
        if(wide_0)
        {
            cov_or_0(_S424.ids_0.w, u32(32));
        }
        else
        {
            g_seen_0[u32(3)] = u32(1);
        }
    }
    if((bd_0.events_0.y) == u32(0))
    {
        _S434 = (_S422.state_1.life_0) != _S420;
    }
    else
    {
        _S434 = false;
    }
    if(_S434)
    {
        _S434 = (_S422.fatigue_before_0) > 0.99000000953674316f;
    }
    else
    {
        _S434 = false;
    }
    if(_S434)
    {
        var _S435 : f32 = fatigue_factor_1(&(_S417), _S422.state_1.life_0);
        _S434 = _S435 <= 0.99000000953674316f;
    }
    else
    {
        _S434 = false;
    }
    if(_S434)
    {
        bd_0.events_0[i32(1)] = abs_step_0;
        if(wide_0)
        {
            cov_or_0(_S424.ids_0.w, u32(64));
        }
        else
        {
            g_seen_0[u32(4)] = u32(1);
        }
    }
    if(_S422.disconnected_0)
    {
        bd_0.events_0[i32(2)] = abs_step_0;
    }
    bd_0.js_0 = _S422.state_1;
    bond_dyn_0[i_6].js_0.damage_0 = bd_0.js_0.damage_0;
    bond_dyn_0[i_6].js_0.crush_1 = bd_0.js_0.crush_1;
    bond_dyn_0[i_6].js_0.kappa_0 = bd_0.js_0.kappa_0;
    bond_dyn_0[i_6].js_0.kappa_c_0 = bd_0.js_0.kappa_c_0;
    bond_dyn_0[i_6].js_0.ductility_0 = bd_0.js_0.ductility_0;
    bond_dyn_0[i_6].js_0.ductility_c_0 = bd_0.js_0.ductility_c_0;
    bond_dyn_0[i_6].js_0.life_0 = bd_0.js_0.life_0;
    bond_dyn_0[i_6].js_0.plastic_x_0 = bd_0.js_0.plastic_x_0;
    bond_dyn_0[i_6].js_0.plastic_y_0 = bd_0.js_0.plastic_y_0;
    bond_dyn_0[i_6].js_0.plastic_t_0 = bd_0.js_0.plastic_t_0;
    bond_dyn_0[i_6].js_0.rebar_plastic_0 = bd_0.js_0.rebar_plastic_0;
    bond_dyn_0[i_6].js_0.rebar_slip0_0 = bd_0.js_0.rebar_slip0_0;
    bond_dyn_0[i_6].js_0.rebar_slip1_0 = bd_0.js_0.rebar_slip1_0;
    bond_dyn_0[i_6].js_0.rebar_work_0 = bd_0.js_0.rebar_work_0;
    bond_dyn_0[i_6].js_0.rebar_broken_0 = bd_0.js_0.rebar_broken_0;
    bond_dyn_0[i_6].js_0.strain_rate_0 = bd_0.js_0.strain_rate_0;
    bond_dyn_0[i_6].js_0.governing_stress_0 = bd_0.js_0.governing_stress_0;
    bond_dyn_0[i_6].js_0.dissipated_0 = bd_0.js_0.dissipated_0;
    bond_dyn_0[i_6].js_0.utilization_0 = bd_0.js_0.utilization_0;
    bond_dyn_0[i_6].js_0.mode_0 = bd_0.js_0.mode_0;
    bond_dyn_0[i_6].force_lin_0 = bd_0.force_lin_0;
    bond_dyn_0[i_6].force_ang_0 = bd_0.force_ang_0;
    bond_dyn_0[i_6].sums_0 = bd_0.sums_0;
    bond_dyn_0[i_6].comps_0 = bd_0.comps_0;
    bond_dyn_0[i_6].events_0 = bd_0.events_0;
    return _S422.disconnected_0;
}

fn island_frame_at_2( gi_2 : u32,  tid_7 : u32)
{
    var woke_4 : bool;
    var ct_3 : u32;
    if(tid_7 < u32(256))
    {
        ct_3 = tid_7;
    }
    else
    {
        ct_3 = u32(1073741824);
    }
    var isl_16 : Island_0;
    isl_16.range_0 = islands_0[gi_2].range_0;
    isl_16.info_0 = islands_0[gi_2].info_0;
    isl_16.com_0 = islands_0[gi_2].com_0;
    isl_16.inertia0_0 = islands_0[gi_2].inertia0_0;
    isl_16.inertia1_0 = islands_0[gi_2].inertia1_0;
    isl_16.inertia2_0 = islands_0[gi_2].inertia2_0;
    isl_16.inv0_0 = islands_0[gi_2].inv0_0;
    isl_16.inv1_0 = islands_0[gi_2].inv1_0;
    isl_16.inv2_0 = islands_0[gi_2].inv2_0;
    isl_16.wcom_0 = islands_0[gi_2].wcom_0;
    isl_16.winv0_0 = islands_0[gi_2].winv0_0;
    isl_16.winv1_0 = islands_0[gi_2].winv1_0;
    isl_16.winv2_0 = islands_0[gi_2].winv2_0;
    isl_16.rotation_0 = islands_0[gi_2].rotation_0;
    isl_16.position_0 = islands_0[gi_2].position_0;
    isl_16.position_err_0 = islands_0[gi_2].position_err_0;
    isl_16.velocity_0 = islands_0[gi_2].velocity_0;
    isl_16.velocity_err_0 = islands_0[gi_2].velocity_err_0;
    isl_16.angular_velocity_0 = islands_0[gi_2].angular_velocity_0;
    isl_16.done_0 = islands_0[gi_2].done_0;
    isl_16.coverage_0 = islands_0[gi_2].coverage_0;
    isl_16.wide_groups_0 = islands_0[gi_2].wide_groups_0;
    isl_16.done_pad_0 = islands_0[gi_2].done_pad_0;
    isl_16.probes_0 = islands_0[gi_2].probes_0;
    isl_16.energy_0 = islands_0[gi_2].energy_0;
    isl_16.rotation_err_0 = islands_0[gi_2].rotation_err_0;
    isl_16.momentum_0 = islands_0[gi_2].momentum_0;
    isl_16.momentum_err_0 = islands_0[gi_2].momentum_err_0;
    var driven_2 : bool = (((isl_16.info_0.x) & (u32(2)))) != u32(0);
    var _S436 : bool = !((((isl_16.info_0.x) & (u32(1)))) != u32(0));
    var _S437 : bool;
    if(_S436)
    {
        _S437 = !driven_2;
    }
    else
    {
        _S437 = false;
    }
    var contact_island_2 : bool = (((isl_16.info_0.x) & (u32(4)))) != u32(0);
    var _S438 : bool = (((isl_16.info_0.x) & (u32(16)))) != u32(0);
    seen_clear_0(tid_7);
    var _S439 : bool = tid_7 == u32(0);
    var cov_10 : u32;
    var settled_2 : bool;
    if(_S439)
    {
        if(contact_island_2 != ((params_0.contact_mode_0) == u32(1)))
        {
            settled_2 = true;
        }
        else
        {
            settled_2 = (((isl_16.info_0.x) & (u32(8)))) != u32(0);
        }
        if(settled_2)
        {
            cov_10 = u32(0);
        }
        else
        {
            cov_10 = u32(1);
        }
        if(contact_island_2)
        {
            settled_2 = contact_stopped_0(isl_16);
        }
        else
        {
            settled_2 = false;
        }
        if(settled_2)
        {
            cov_10 = u32(0);
        }
        g_run_0 = cov_10;
        g_halt_0 = u32(0);
    }
    workgroupBarrier();
    if((((isl_16.info_0.z) & (u32(1)))) != u32(0))
    {
        settled_2 = true;
    }
    else
    {
        settled_2 = g_run_0 == u32(0);
    }
    if(settled_2)
    {
        cov_10 = u32(0);
    }
    else
    {
        cov_10 = min(isl_16.info_0.y, params_0.max_steps_0);
    }
    var _S440 : f32 = params_0.dt_0;
    var _S441 : bool = (params_0.fracture_0) != u32(0);
    var _S442 : bool = (params_0.rigid_motion_loads_0) != u32(0);
    var rg_14 : Rigid_0 = rigid_of_0(isl_16);
    var work_5 : f32 = 0.0f;
    var work_err_4 : f32 = 0.0f;
    settled_2 = _S438;
    var done_3 : u32 = u32(0);
    var woke_5 : bool = false;
    var resumed_4 : bool = false;
    loop
    {
        if(done_3 < cov_10)
        {
        }
        else
        {
            woke_4 = woke_5;
            break;
        }
        var abs_step_1 : u32 = isl_16.info_0.w + done_3 + u32(1);
        var k_15 : u32 = abs_step_1 - u32(1) - params_0.step_start_0;
        var quiet_4 : bool = (((isl_16.info_0.x) & (u32(32)))) != u32(0);
        var _S443 : bool;
        if(settled_2)
        {
            _S443 = !quiet_4;
        }
        else
        {
            _S443 = false;
        }
        var c_26 : u32;
        var _S444 : bool;
        var explicit_step_2 : bool;
        if(_S443)
        {
            var _S445 : vec3<f32> = vec3<f32>(0.0f);
            var norm_2 : vec3<f32> = _S445;
            var unused0_2 : vec3<f32> = _S445;
            c_26 = isl_16.range_0.x + ct_3;
            loop
            {
                if(c_26 < (isl_16.range_0.y))
                {
                }
                else
                {
                    break;
                }
                var _S446 : f32 = settled_chunk_load_0(c_26, rg_14.rot_0, k_15, _S440, contact_island_2);
                norm_2[i32(0)] = norm_2[i32(0)] + _S446;
                c_26 = c_26 + u32(256);
            }
            group_sum3_0(tid_7, &(norm_2), &(unused0_2));
            if((params_0.solve_mode_0) == u32(1))
            {
                _S444 = (abs(norm_2.x - isl_16.energy_0.z)) > (isl_16.energy_0.w);
            }
            else
            {
                _S444 = false;
            }
            if(_S444)
            {
                explicit_step_2 = false;
                woke_4 = true;
            }
            else
            {
                explicit_step_2 = settled_2;
                woke_4 = woke_5;
            }
            settled_2 = explicit_step_2;
        }
        else
        {
            woke_4 = woke_5;
        }
        if(_S436)
        {
            _S444 = !quiet_4;
        }
        else
        {
            _S444 = false;
        }
        if(_S444)
        {
            var _S447 : vec3<f32> = vec3<f32>(0.0f);
            var f_16 : vec3<f32> = _S447;
            var t_19 : vec3<f32> = _S447;
            c_26 = isl_16.range_0.x + ct_3;
            loop
            {
                if(c_26 < (isl_16.range_0.y))
                {
                }
                else
                {
                    break;
                }
                net_load_0(c_26, isl_16, rg_14, k_15, _S440, contact_island_2, &(f_16), &(t_19));
                c_26 = c_26 + u32(256);
            }
            group_sum3_0(tid_7, &(f_16), &(t_19));
            rigid_acceleration_0(isl_16, &(rg_14), f_16, t_19);
        }
        if(!settled_2)
        {
            explicit_step_2 = !quiet_4;
        }
        else
        {
            explicit_step_2 = false;
        }
        var c_27 : u32;
        if(explicit_step_2)
        {
            c_26 = isl_16.range_0.z + tid_7;
            loop
            {
                if(c_26 < (isl_16.range_0.w))
                {
                }
                else
                {
                    break;
                }
                var _S448 : bool = bond_update_0(c_26, _S440, _S441, abs_step_1, false);
                if(_S448)
                {
                    g_halt_0 = u32(1);
                }
                c_26 = c_26 + u32(512);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
            c_27 = isl_16.range_0.x + ct_3;
            loop
            {
                if(c_27 < (isl_16.range_0.y))
                {
                }
                else
                {
                    break;
                }
                chunk_update_0(c_27, isl_16, rg_14, _S440, _S442, k_15, contact_island_2, &(work_5), &(work_err_4));
                c_27 = c_27 + u32(256);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
        }
        var _S449 : bool;
        if(_S437)
        {
            _S449 = !quiet_4;
        }
        else
        {
            _S449 = false;
        }
        if(_S449)
        {
            integrate_rigid_0(isl_16, &(rg_14), _S440);
        }
        var _S450 : bool;
        if(_S436)
        {
            _S450 = explicit_step_2;
        }
        else
        {
            _S450 = false;
        }
        if(_S450)
        {
            var _S451 : vec3<f32> = isl_16.wcom_0.xyz;
            var _S452 : vec3<f32> = vec3<f32>(0.0f);
            var tu_4 : vec3<f32> = _S452;
            var pv_4 : vec3<f32> = _S452;
            c_26 = isl_16.range_0.x + ct_3;
            loop
            {
                if(c_26 < (isl_16.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_moments_0(c_26, &(tu_4), &(pv_4));
                c_26 = c_26 + u32(256);
            }
            group_sum3_0(tid_7, &(tu_4), &(pv_4));
            var tr_7 : vec3<f32> = tu_4 / vec3<f32>(isl_16.wcom_0.w);
            var dv_7 : vec3<f32> = pv_4 / vec3<f32>(isl_16.wcom_0.w);
            var lu_4 : vec3<f32> = _S452;
            var lv_4 : vec3<f32> = _S452;
            c_27 = isl_16.range_0.x + ct_3;
            loop
            {
                if(c_27 < (isl_16.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_angular_0(c_27, _S451, tr_7, dv_7, &(lu_4), &(lv_4));
                c_27 = c_27 + u32(256);
            }
            group_sum3_0(tid_7, &(lu_4), &(lv_4));
            var phi_6 : vec3<f32> = rows_mul_0(isl_16.winv0_0, isl_16.winv1_0, isl_16.winv2_0, lu_4);
            var dw_5 : vec3<f32> = rows_mul_0(isl_16.winv0_0, isl_16.winv1_0, isl_16.winv2_0, lv_4);
            var c_28 : u32 = isl_16.range_0.x + ct_3;
            loop
            {
                if(c_28 < (isl_16.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_28, _S451, tr_7, phi_6, dv_7, dw_5);
                c_28 = c_28 + u32(256);
            }
            if(!driven_2)
            {
                drift_rigid_0(isl_16, &(rg_14), tr_7, phi_6, dv_7, dw_5);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
        }
        var _S453 : bool;
        if(_S439)
        {
            _S453 = (isl_16.probes_0.y) > (isl_16.probes_0.x);
        }
        else
        {
            _S453 = false;
        }
        if(_S453)
        {
            record_probes_0(isl_16, rg_14, k_15);
        }
        var _S454 : u32 = done_3 + u32(1);
        var _S455 : bool;
        if(g_halt_0 != u32(0))
        {
            var _S456 : bool = island_connected_narrow_0(tid_7, ct_3, isl_16);
            _S455 = _S456;
        }
        else
        {
            _S455 = false;
        }
        var resumed_5 : bool;
        if(_S455)
        {
            if(_S439)
            {
                g_halt_0 = u32(0);
            }
            workgroupBarrier();
            resumed_5 = true;
        }
        else
        {
            resumed_5 = resumed_4;
        }
        if(g_halt_0 != u32(0))
        {
            done_3 = _S454;
            resumed_4 = resumed_5;
            break;
        }
        done_3 = _S454;
        woke_5 = woke_4;
        resumed_4 = resumed_5;
    }
    var wsum_2 : vec3<f32> = vec3<f32>(work_5, work_err_4, 0.0f);
    var unused_2 : vec3<f32> = vec3<f32>(0.0f);
    group_sum3_0(tid_7, &(wsum_2), &(unused_2));
    if(_S439)
    {
        isl_16.rotation_0 = quat_vec_0(rg_14.rot_0);
        isl_16.rotation_err_0 = rg_14.rot_err_0;
        isl_16.position_0 = vec4<f32>(rg_14.pos_0, 0.0f);
        isl_16.position_err_0 = vec4<f32>(rg_14.pos_err_0, 0.0f);
        isl_16.velocity_0 = vec4<f32>(rg_14.vel_0, 0.0f);
        isl_16.velocity_err_0 = vec4<f32>(rg_14.vel_err_0, 0.0f);
        isl_16.angular_velocity_0 = vec4<f32>(rg_14.w_1, 0.0f);
        isl_16.momentum_0 = vec4<f32>(rg_14.l_0, 0.0f);
        isl_16.momentum_err_0 = vec4<f32>(rg_14.l_err_0, 0.0f);
        isl_16.done_0 = done_3;
        isl_16.info_0[i32(1)] = isl_16.info_0[i32(1)] - done_3;
        if(g_halt_0 != u32(0))
        {
            _S437 = contact_island_2;
        }
        else
        {
            _S437 = false;
        }
        if(_S437)
        {
            contact_split_at_0(isl_16.info_0.w + done_3);
        }
        var _S457 : f32 = wsum_2.x;
        var _S458 : f32 = isl_16.energy_0[i32(0)];
        var _S459 : f32 = isl_16.energy_0[i32(1)];
        comp_add1_1(&(_S458), &(_S459), _S457);
        isl_16.energy_0[i32(0)] = _S458;
        isl_16.energy_0[i32(1)] = _S459 + wsum_2.y;
        isl_16.info_0[i32(3)] = isl_16.info_0[i32(3)] + done_3;
        if((((isl_16.info_0.x) & (u32(8)))) == u32(0))
        {
            isl_16.info_0[i32(2)] = ((isl_16.info_0[i32(2)]) & (u32(4294967293)));
        }
        if(g_halt_0 != u32(0))
        {
            isl_16.info_0[i32(2)] = ((isl_16.info_0[i32(2)]) | (u32(1)));
        }
        var _S460 : u32 = seen_bits_0();
        if(g_halt_0 != u32(0))
        {
            ct_3 = u32(128);
        }
        else
        {
            ct_3 = u32(0);
        }
        var _S461 : u32 = (_S460 | (ct_3));
        if(woke_4)
        {
            ct_3 = u32(2048);
        }
        else
        {
            ct_3 = u32(0);
        }
        var _S462 : u32 = (_S461 | (ct_3));
        if(resumed_4)
        {
            ct_3 = u32(2097280);
        }
        else
        {
            ct_3 = u32(0);
        }
        var cov_11 : u32 = (_S462 | (ct_3));
        if(done_3 > u32(0))
        {
            var quiet_5 : bool = (((isl_16.info_0.x) & (u32(32)))) != u32(0);
            if(quiet_5)
            {
                ct_3 = u32(4096);
            }
            else
            {
                ct_3 = u32(0);
            }
            var cov_12 : u32 = (cov_11 | (ct_3));
            if((((isl_16.info_0.x) & (u32(16)))) != u32(0))
            {
                _S437 = !quiet_5;
            }
            else
            {
                _S437 = false;
            }
            if(_S437)
            {
                ct_3 = u32(512);
            }
            else
            {
                ct_3 = u32(0);
            }
            var cov_13 : u32 = (cov_12 | (ct_3));
            if((params_0.contact_mode_0) == u32(1))
            {
                ct_3 = u32(524288);
            }
            else
            {
                ct_3 = u32(0);
            }
            var cov_14 : u32 = (cov_13 | (ct_3));
            if((isl_16.probes_0.y) > (isl_16.probes_0.x))
            {
                ct_3 = u32(262144);
            }
            else
            {
                ct_3 = u32(0);
            }
            cov_10 = (cov_14 | (ct_3));
        }
        else
        {
            cov_10 = cov_11;
        }
        isl_16.coverage_0 = ((isl_16.coverage_0) | (cov_10));
        if(woke_4)
        {
            isl_16.info_0[i32(0)] = ((isl_16.info_0[i32(0)]) & (u32(4294967279)));
            isl_16.info_0[i32(2)] = ((isl_16.info_0[i32(2)]) | (u32(4)));
        }
        islands_0[gi_2].range_0 = isl_16.range_0;
        islands_0[gi_2].info_0 = isl_16.info_0;
        islands_0[gi_2].com_0 = isl_16.com_0;
        islands_0[gi_2].inertia0_0 = isl_16.inertia0_0;
        islands_0[gi_2].inertia1_0 = isl_16.inertia1_0;
        islands_0[gi_2].inertia2_0 = isl_16.inertia2_0;
        islands_0[gi_2].inv0_0 = isl_16.inv0_0;
        islands_0[gi_2].inv1_0 = isl_16.inv1_0;
        islands_0[gi_2].inv2_0 = isl_16.inv2_0;
        islands_0[gi_2].wcom_0 = isl_16.wcom_0;
        islands_0[gi_2].winv0_0 = isl_16.winv0_0;
        islands_0[gi_2].winv1_0 = isl_16.winv1_0;
        islands_0[gi_2].winv2_0 = isl_16.winv2_0;
        islands_0[gi_2].rotation_0 = isl_16.rotation_0;
        islands_0[gi_2].position_0 = isl_16.position_0;
        islands_0[gi_2].position_err_0 = isl_16.position_err_0;
        islands_0[gi_2].velocity_0 = isl_16.velocity_0;
        islands_0[gi_2].velocity_err_0 = isl_16.velocity_err_0;
        islands_0[gi_2].angular_velocity_0 = isl_16.angular_velocity_0;
        islands_0[gi_2].done_0 = isl_16.done_0;
        islands_0[gi_2].coverage_0 = isl_16.coverage_0;
        islands_0[gi_2].wide_groups_0 = isl_16.wide_groups_0;
        islands_0[gi_2].done_pad_0 = isl_16.done_pad_0;
        islands_0[gi_2].probes_0 = isl_16.probes_0;
        islands_0[gi_2].energy_0 = isl_16.energy_0;
        islands_0[gi_2].rotation_err_0 = isl_16.rotation_err_0;
        islands_0[gi_2].momentum_0 = isl_16.momentum_0;
        islands_0[gi_2].momentum_err_0 = isl_16.momentum_err_0;
    }
    return;
}

fn island_frame_at_3( gi_3 : u32,  tid_8 : u32)
{
    var woke_6 : bool;
    var ct_4 : u32;
    if(tid_8 < u32(256))
    {
        ct_4 = tid_8;
    }
    else
    {
        ct_4 = u32(1073741824);
    }
    var isl_17 : Island_0;
    isl_17.range_0 = islands_0[gi_3].range_0;
    isl_17.info_0 = islands_0[gi_3].info_0;
    isl_17.com_0 = islands_0[gi_3].com_0;
    isl_17.inertia0_0 = islands_0[gi_3].inertia0_0;
    isl_17.inertia1_0 = islands_0[gi_3].inertia1_0;
    isl_17.inertia2_0 = islands_0[gi_3].inertia2_0;
    isl_17.inv0_0 = islands_0[gi_3].inv0_0;
    isl_17.inv1_0 = islands_0[gi_3].inv1_0;
    isl_17.inv2_0 = islands_0[gi_3].inv2_0;
    isl_17.wcom_0 = islands_0[gi_3].wcom_0;
    isl_17.winv0_0 = islands_0[gi_3].winv0_0;
    isl_17.winv1_0 = islands_0[gi_3].winv1_0;
    isl_17.winv2_0 = islands_0[gi_3].winv2_0;
    isl_17.rotation_0 = islands_0[gi_3].rotation_0;
    isl_17.position_0 = islands_0[gi_3].position_0;
    isl_17.position_err_0 = islands_0[gi_3].position_err_0;
    isl_17.velocity_0 = islands_0[gi_3].velocity_0;
    isl_17.velocity_err_0 = islands_0[gi_3].velocity_err_0;
    isl_17.angular_velocity_0 = islands_0[gi_3].angular_velocity_0;
    isl_17.done_0 = islands_0[gi_3].done_0;
    isl_17.coverage_0 = islands_0[gi_3].coverage_0;
    isl_17.wide_groups_0 = islands_0[gi_3].wide_groups_0;
    isl_17.done_pad_0 = islands_0[gi_3].done_pad_0;
    isl_17.probes_0 = islands_0[gi_3].probes_0;
    isl_17.energy_0 = islands_0[gi_3].energy_0;
    isl_17.rotation_err_0 = islands_0[gi_3].rotation_err_0;
    isl_17.momentum_0 = islands_0[gi_3].momentum_0;
    isl_17.momentum_err_0 = islands_0[gi_3].momentum_err_0;
    var driven_3 : bool = (((isl_17.info_0.x) & (u32(2)))) != u32(0);
    var _S463 : bool = !((((isl_17.info_0.x) & (u32(1)))) != u32(0));
    var _S464 : bool;
    if(_S463)
    {
        _S464 = !driven_3;
    }
    else
    {
        _S464 = false;
    }
    var contact_island_3 : bool = (((isl_17.info_0.x) & (u32(4)))) != u32(0);
    var _S465 : bool = (((isl_17.info_0.x) & (u32(16)))) != u32(0);
    seen_clear_0(tid_8);
    var _S466 : bool = tid_8 == u32(0);
    var cov_15 : u32;
    var settled_3 : bool;
    if(_S466)
    {
        if(contact_island_3 != ((params_0.contact_mode_0) == u32(1)))
        {
            settled_3 = true;
        }
        else
        {
            settled_3 = (((isl_17.info_0.x) & (u32(8)))) != u32(0);
        }
        if(settled_3)
        {
            cov_15 = u32(0);
        }
        else
        {
            cov_15 = u32(1);
        }
        if(contact_island_3)
        {
            settled_3 = contact_stopped_0(isl_17);
        }
        else
        {
            settled_3 = false;
        }
        if(settled_3)
        {
            cov_15 = u32(0);
        }
        g_run_0 = cov_15;
        g_halt_0 = u32(0);
    }
    workgroupBarrier();
    if((((isl_17.info_0.z) & (u32(1)))) != u32(0))
    {
        settled_3 = true;
    }
    else
    {
        settled_3 = g_run_0 == u32(0);
    }
    if(settled_3)
    {
        cov_15 = u32(0);
    }
    else
    {
        cov_15 = min(isl_17.info_0.y, params_0.max_steps_0);
    }
    var _S467 : f32 = params_0.dt_0;
    var _S468 : bool = (params_0.fracture_0) != u32(0);
    var _S469 : bool = (params_0.rigid_motion_loads_0) != u32(0);
    var rg_15 : Rigid_0 = rigid_of_0(isl_17);
    var work_6 : f32 = 0.0f;
    var work_err_5 : f32 = 0.0f;
    settled_3 = _S465;
    var done_4 : u32 = u32(0);
    var woke_7 : bool = false;
    var resumed_6 : bool = false;
    loop
    {
        if(done_4 < cov_15)
        {
        }
        else
        {
            woke_6 = woke_7;
            break;
        }
        var abs_step_2 : u32 = isl_17.info_0.w + done_4 + u32(1);
        var k_16 : u32 = abs_step_2 - u32(1) - params_0.step_start_0;
        var quiet_6 : bool = (((isl_17.info_0.x) & (u32(32)))) != u32(0);
        var _S470 : bool;
        if(settled_3)
        {
            _S470 = !quiet_6;
        }
        else
        {
            _S470 = false;
        }
        var c_29 : u32;
        var _S471 : bool;
        var explicit_step_3 : bool;
        if(_S470)
        {
            var _S472 : vec3<f32> = vec3<f32>(0.0f);
            var norm_3 : vec3<f32> = _S472;
            var unused0_3 : vec3<f32> = _S472;
            c_29 = isl_17.range_0.x + ct_4;
            loop
            {
                if(c_29 < (isl_17.range_0.y))
                {
                }
                else
                {
                    break;
                }
                var _S473 : f32 = settled_chunk_load_0(c_29, rg_15.rot_0, k_16, _S467, contact_island_3);
                norm_3[i32(0)] = norm_3[i32(0)] + _S473;
                c_29 = c_29 + u32(256);
            }
            group_sum3_1(tid_8, &(norm_3), &(unused0_3));
            if((params_0.solve_mode_0) == u32(1))
            {
                _S471 = (abs(norm_3.x - isl_17.energy_0.z)) > (isl_17.energy_0.w);
            }
            else
            {
                _S471 = false;
            }
            if(_S471)
            {
                explicit_step_3 = false;
                woke_6 = true;
            }
            else
            {
                explicit_step_3 = settled_3;
                woke_6 = woke_7;
            }
            settled_3 = explicit_step_3;
        }
        else
        {
            woke_6 = woke_7;
        }
        if(_S463)
        {
            _S471 = !quiet_6;
        }
        else
        {
            _S471 = false;
        }
        if(_S471)
        {
            var _S474 : vec3<f32> = vec3<f32>(0.0f);
            var f_17 : vec3<f32> = _S474;
            var t_20 : vec3<f32> = _S474;
            c_29 = isl_17.range_0.x + ct_4;
            loop
            {
                if(c_29 < (isl_17.range_0.y))
                {
                }
                else
                {
                    break;
                }
                net_load_1(c_29, isl_17, rg_15, k_16, _S467, contact_island_3, &(f_17), &(t_20));
                c_29 = c_29 + u32(256);
            }
            group_sum3_1(tid_8, &(f_17), &(t_20));
            rigid_acceleration_1(isl_17, &(rg_15), f_17, t_20);
        }
        if(!settled_3)
        {
            explicit_step_3 = !quiet_6;
        }
        else
        {
            explicit_step_3 = false;
        }
        var c_30 : u32;
        if(explicit_step_3)
        {
            c_29 = isl_17.range_0.z + tid_8;
            loop
            {
                if(c_29 < (isl_17.range_0.w))
                {
                }
                else
                {
                    break;
                }
                var _S475 : bool = bond_update_0(c_29, _S467, _S468, abs_step_2, false);
                if(_S475)
                {
                    g_halt_0 = u32(1);
                }
                c_29 = c_29 + u32(256);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
            c_30 = isl_17.range_0.x + ct_4;
            loop
            {
                if(c_30 < (isl_17.range_0.y))
                {
                }
                else
                {
                    break;
                }
                chunk_update_1(c_30, isl_17, rg_15, _S467, _S469, k_16, contact_island_3, &(work_6), &(work_err_5));
                c_30 = c_30 + u32(256);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
        }
        var _S476 : bool;
        if(_S464)
        {
            _S476 = !quiet_6;
        }
        else
        {
            _S476 = false;
        }
        if(_S476)
        {
            integrate_rigid_1(isl_17, &(rg_15), _S467);
        }
        var _S477 : bool;
        if(_S463)
        {
            _S477 = explicit_step_3;
        }
        else
        {
            _S477 = false;
        }
        if(_S477)
        {
            var _S478 : vec3<f32> = isl_17.wcom_0.xyz;
            var _S479 : vec3<f32> = vec3<f32>(0.0f);
            var tu_5 : vec3<f32> = _S479;
            var pv_5 : vec3<f32> = _S479;
            c_29 = isl_17.range_0.x + ct_4;
            loop
            {
                if(c_29 < (isl_17.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_moments_1(c_29, &(tu_5), &(pv_5));
                c_29 = c_29 + u32(256);
            }
            group_sum3_1(tid_8, &(tu_5), &(pv_5));
            var tr_8 : vec3<f32> = tu_5 / vec3<f32>(isl_17.wcom_0.w);
            var dv_8 : vec3<f32> = pv_5 / vec3<f32>(isl_17.wcom_0.w);
            var lu_5 : vec3<f32> = _S479;
            var lv_5 : vec3<f32> = _S479;
            c_30 = isl_17.range_0.x + ct_4;
            loop
            {
                if(c_30 < (isl_17.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_angular_1(c_30, _S478, tr_8, dv_8, &(lu_5), &(lv_5));
                c_30 = c_30 + u32(256);
            }
            group_sum3_1(tid_8, &(lu_5), &(lv_5));
            var phi_7 : vec3<f32> = rows_mul_0(isl_17.winv0_0, isl_17.winv1_0, isl_17.winv2_0, lu_5);
            var dw_6 : vec3<f32> = rows_mul_0(isl_17.winv0_0, isl_17.winv1_0, isl_17.winv2_0, lv_5);
            var c_31 : u32 = isl_17.range_0.x + ct_4;
            loop
            {
                if(c_31 < (isl_17.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_31, _S478, tr_8, phi_7, dv_8, dw_6);
                c_31 = c_31 + u32(256);
            }
            if(!driven_3)
            {
                drift_rigid_1(isl_17, &(rg_15), tr_8, phi_7, dv_8, dw_6);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
        }
        var _S480 : bool;
        if(_S466)
        {
            _S480 = (isl_17.probes_0.y) > (isl_17.probes_0.x);
        }
        else
        {
            _S480 = false;
        }
        if(_S480)
        {
            record_probes_0(isl_17, rg_15, k_16);
        }
        var _S481 : u32 = done_4 + u32(1);
        var _S482 : bool;
        if(g_halt_0 != u32(0))
        {
            var _S483 : bool = island_connected_narrow_0(tid_8, ct_4, isl_17);
            _S482 = _S483;
        }
        else
        {
            _S482 = false;
        }
        var resumed_7 : bool;
        if(_S482)
        {
            if(_S466)
            {
                g_halt_0 = u32(0);
            }
            workgroupBarrier();
            resumed_7 = true;
        }
        else
        {
            resumed_7 = resumed_6;
        }
        if(g_halt_0 != u32(0))
        {
            done_4 = _S481;
            resumed_6 = resumed_7;
            break;
        }
        done_4 = _S481;
        woke_7 = woke_6;
        resumed_6 = resumed_7;
    }
    var wsum_3 : vec3<f32> = vec3<f32>(work_6, work_err_5, 0.0f);
    var unused_3 : vec3<f32> = vec3<f32>(0.0f);
    group_sum3_1(tid_8, &(wsum_3), &(unused_3));
    if(_S466)
    {
        isl_17.rotation_0 = quat_vec_0(rg_15.rot_0);
        isl_17.rotation_err_0 = rg_15.rot_err_0;
        isl_17.position_0 = vec4<f32>(rg_15.pos_0, 0.0f);
        isl_17.position_err_0 = vec4<f32>(rg_15.pos_err_0, 0.0f);
        isl_17.velocity_0 = vec4<f32>(rg_15.vel_0, 0.0f);
        isl_17.velocity_err_0 = vec4<f32>(rg_15.vel_err_0, 0.0f);
        isl_17.angular_velocity_0 = vec4<f32>(rg_15.w_1, 0.0f);
        isl_17.momentum_0 = vec4<f32>(rg_15.l_0, 0.0f);
        isl_17.momentum_err_0 = vec4<f32>(rg_15.l_err_0, 0.0f);
        isl_17.done_0 = done_4;
        isl_17.info_0[i32(1)] = isl_17.info_0[i32(1)] - done_4;
        if(g_halt_0 != u32(0))
        {
            _S464 = contact_island_3;
        }
        else
        {
            _S464 = false;
        }
        if(_S464)
        {
            contact_split_at_0(isl_17.info_0.w + done_4);
        }
        var _S484 : f32 = wsum_3.x;
        var _S485 : f32 = isl_17.energy_0[i32(0)];
        var _S486 : f32 = isl_17.energy_0[i32(1)];
        comp_add1_2(&(_S485), &(_S486), _S484);
        isl_17.energy_0[i32(0)] = _S485;
        isl_17.energy_0[i32(1)] = _S486 + wsum_3.y;
        isl_17.info_0[i32(3)] = isl_17.info_0[i32(3)] + done_4;
        if((((isl_17.info_0.x) & (u32(8)))) == u32(0))
        {
            isl_17.info_0[i32(2)] = ((isl_17.info_0[i32(2)]) & (u32(4294967293)));
        }
        if(g_halt_0 != u32(0))
        {
            isl_17.info_0[i32(2)] = ((isl_17.info_0[i32(2)]) | (u32(1)));
        }
        var _S487 : u32 = seen_bits_0();
        if(g_halt_0 != u32(0))
        {
            ct_4 = u32(128);
        }
        else
        {
            ct_4 = u32(0);
        }
        var _S488 : u32 = (_S487 | (ct_4));
        if(woke_6)
        {
            ct_4 = u32(2048);
        }
        else
        {
            ct_4 = u32(0);
        }
        var _S489 : u32 = (_S488 | (ct_4));
        if(resumed_6)
        {
            ct_4 = u32(2097280);
        }
        else
        {
            ct_4 = u32(0);
        }
        var cov_16 : u32 = (_S489 | (ct_4));
        if(done_4 > u32(0))
        {
            var quiet_7 : bool = (((isl_17.info_0.x) & (u32(32)))) != u32(0);
            if(quiet_7)
            {
                ct_4 = u32(4096);
            }
            else
            {
                ct_4 = u32(0);
            }
            var cov_17 : u32 = (cov_16 | (ct_4));
            if((((isl_17.info_0.x) & (u32(16)))) != u32(0))
            {
                _S464 = !quiet_7;
            }
            else
            {
                _S464 = false;
            }
            if(_S464)
            {
                ct_4 = u32(512);
            }
            else
            {
                ct_4 = u32(0);
            }
            var cov_18 : u32 = (cov_17 | (ct_4));
            if((params_0.contact_mode_0) == u32(1))
            {
                ct_4 = u32(524288);
            }
            else
            {
                ct_4 = u32(0);
            }
            var cov_19 : u32 = (cov_18 | (ct_4));
            if((isl_17.probes_0.y) > (isl_17.probes_0.x))
            {
                ct_4 = u32(262144);
            }
            else
            {
                ct_4 = u32(0);
            }
            cov_15 = (cov_19 | (ct_4));
        }
        else
        {
            cov_15 = cov_16;
        }
        isl_17.coverage_0 = ((isl_17.coverage_0) | (cov_15));
        if(woke_6)
        {
            isl_17.info_0[i32(0)] = ((isl_17.info_0[i32(0)]) & (u32(4294967279)));
            isl_17.info_0[i32(2)] = ((isl_17.info_0[i32(2)]) | (u32(4)));
        }
        islands_0[gi_3].range_0 = isl_17.range_0;
        islands_0[gi_3].info_0 = isl_17.info_0;
        islands_0[gi_3].com_0 = isl_17.com_0;
        islands_0[gi_3].inertia0_0 = isl_17.inertia0_0;
        islands_0[gi_3].inertia1_0 = isl_17.inertia1_0;
        islands_0[gi_3].inertia2_0 = isl_17.inertia2_0;
        islands_0[gi_3].inv0_0 = isl_17.inv0_0;
        islands_0[gi_3].inv1_0 = isl_17.inv1_0;
        islands_0[gi_3].inv2_0 = isl_17.inv2_0;
        islands_0[gi_3].wcom_0 = isl_17.wcom_0;
        islands_0[gi_3].winv0_0 = isl_17.winv0_0;
        islands_0[gi_3].winv1_0 = isl_17.winv1_0;
        islands_0[gi_3].winv2_0 = isl_17.winv2_0;
        islands_0[gi_3].rotation_0 = isl_17.rotation_0;
        islands_0[gi_3].position_0 = isl_17.position_0;
        islands_0[gi_3].position_err_0 = isl_17.position_err_0;
        islands_0[gi_3].velocity_0 = isl_17.velocity_0;
        islands_0[gi_3].velocity_err_0 = isl_17.velocity_err_0;
        islands_0[gi_3].angular_velocity_0 = isl_17.angular_velocity_0;
        islands_0[gi_3].done_0 = isl_17.done_0;
        islands_0[gi_3].coverage_0 = isl_17.coverage_0;
        islands_0[gi_3].wide_groups_0 = isl_17.wide_groups_0;
        islands_0[gi_3].done_pad_0 = isl_17.done_pad_0;
        islands_0[gi_3].probes_0 = isl_17.probes_0;
        islands_0[gi_3].energy_0 = isl_17.energy_0;
        islands_0[gi_3].rotation_err_0 = isl_17.rotation_err_0;
        islands_0[gi_3].momentum_0 = isl_17.momentum_0;
        islands_0[gi_3].momentum_err_0 = isl_17.momentum_err_0;
    }
    return;
}

fn group_sum3_2( tid_9 : u32,  a_16 : ptr<function, vec3<f32>>,  b_24 : ptr<function, vec3<f32>>)
{
    var x_13 : vec4<f32> = vec4<f32>((*a_16), 0.0f);
    var y_3 : vec4<f32> = vec4<f32>((*b_24), 0.0f);
    group_sum2_0(tid_9, &(x_13), &(y_3));
    (*a_16) = x_13.xyz;
    (*b_24) = y_3.xyz;
    return;
}

fn group_sum2_1( tid_10 : u32,  a_17 : ptr<function, vec4<f32>>,  b_25 : ptr<function, vec4<f32>>)
{
    if(tid_10 < u32(256))
    {
        g_red_a_0[tid_10] = (*a_17);
        g_red_b_0[tid_10] = (*b_25);
    }
    workgroupBarrier();
    var s_7 : u32 = u32(128);
    loop
    {
        if(s_7 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_10 < s_7)
        {
            var _S490 : u32 = tid_10 + s_7;
            g_red_a_0[tid_10] = g_red_a_0[tid_10] + g_red_a_0[_S490];
            g_red_b_0[tid_10] = g_red_b_0[tid_10] + g_red_b_0[_S490];
        }
        workgroupBarrier();
        s_7 = (s_7 >> (u32(1)));
    }
    (*a_17) = g_red_a_0[i32(0)];
    (*b_25) = g_red_b_0[i32(0)];
    workgroupBarrier();
    return;
}

fn group_sum3_3( tid_11 : u32,  a_18 : ptr<function, vec3<f32>>,  b_26 : ptr<function, vec3<f32>>)
{
    var x_14 : vec4<f32> = vec4<f32>((*a_18), 0.0f);
    var y_4 : vec4<f32> = vec4<f32>((*b_26), 0.0f);
    group_sum2_1(tid_11, &(x_14), &(y_4));
    (*a_18) = x_14.xyz;
    (*b_26) = y_4.xyz;
    return;
}

fn group_sum2_2( tid_12 : u32,  a_19 : ptr<function, vec4<f32>>,  b_27 : ptr<function, vec4<f32>>)
{
    if(tid_12 < u32(256))
    {
        g_red_a_0[tid_12] = (*a_19);
        g_red_b_0[tid_12] = (*b_27);
    }
    workgroupBarrier();
    var s_8 : u32 = u32(128);
    loop
    {
        if(s_8 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_12 < s_8)
        {
            var _S491 : u32 = tid_12 + s_8;
            g_red_a_0[tid_12] = g_red_a_0[tid_12] + g_red_a_0[_S491];
            g_red_b_0[tid_12] = g_red_b_0[tid_12] + g_red_b_0[_S491];
        }
        workgroupBarrier();
        s_8 = (s_8 >> (u32(1)));
    }
    (*a_19) = g_red_a_0[i32(0)];
    (*b_27) = g_red_b_0[i32(0)];
    workgroupBarrier();
    return;
}

fn group_sum2_3( tid_13 : u32,  a_20 : ptr<function, vec4<f32>>,  b_28 : ptr<function, vec4<f32>>)
{
    if(tid_13 < u32(256))
    {
        g_red_a_0[tid_13] = (*a_20);
        g_red_b_0[tid_13] = (*b_28);
    }
    workgroupBarrier();
    var s_9 : u32 = u32(128);
    loop
    {
        if(s_9 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_13 < s_9)
        {
            var _S492 : u32 = tid_13 + s_9;
            g_red_a_0[tid_13] = g_red_a_0[tid_13] + g_red_a_0[_S492];
            g_red_b_0[tid_13] = g_red_b_0[tid_13] + g_red_b_0[_S492];
        }
        workgroupBarrier();
        s_9 = (s_9 >> (u32(1)));
    }
    (*a_20) = g_red_a_0[i32(0)];
    (*b_28) = g_red_b_0[i32(0)];
    workgroupBarrier();
    return;
}

fn pair_lane_count_0() -> u32
{
    return u32(1);
}

fn stopped_0() -> bool
{
    var _S493 : vec4<u32> = islands_0[params_0.halt_index_0].info_0;
    var _S494 : bool;
    if((((islands_0[params_0.halt_index_0].info_0.z) & (u32(1)))) != u32(0))
    {
        _S494 = true;
    }
    else
    {
        _S494 = (_S493.y) != u32(0);
    }
    return _S494;
}

fn pair_any_0( x_15 : bool,  lanes_0 : u32) -> bool
{
    return x_15;
}

fn world_diff_0( a_21 : WorldPoint_0,  b_29 : WorldPoint_0) -> vec3<f32>
{
    return a_21.hi_0 - b_29.hi_0 + (a_21.lo_0 - b_29.lo_0) + (a_21.rel_0 - b_29.rel_0);
}

fn may_overlap_lanes_0( a_22 : Box_0,  b_30 : Box_0,  lane_0 : u32,  lanes_1 : u32) -> bool
{
    var separated_0 : bool = false;
    var i_7 : u32 = lane_0;
    loop
    {
        if(i_7 < u32(15))
        {
        }
        else
        {
            break;
        }
        var d_4 : vec3<f32> = b_30.center_1 - a_22.center_1;
        var margin_0 : f32 = 0.00000999999974738f * (length(a_22.half_2) + length(b_30.half_2));
        var axes_0 : array<vec3<f32>, i32(6)> = array<vec3<f32>, i32(6)>( a_22.axis0_0, a_22.axis1_0, a_22.axis2_0, b_30.axis0_0, b_30.axis1_0, b_30.axis2_0 );
        var l_4 : vec3<f32>;
        if(i_7 < u32(6))
        {
            l_4 = axes_0[i_7];
        }
        else
        {
            var _S495 : u32 = i_7 - u32(6);
            l_4 = cross(axes_0[_S495 / u32(3)], axes_0[u32(3) + _S495 % u32(3)]);
        }
        var len_0 : f32 = length(l_4);
        if(len_0 <= 9.99999997475242708e-07f)
        {
            i_7 = i_7 + lanes_1;
            continue;
        }
        var separated_1 : bool;
        if((abs(dot(d_4, l_4))) > (a_22.half_2.x * abs(dot(a_22.axis0_0, l_4)) + a_22.half_2.y * abs(dot(a_22.axis1_0, l_4)) + a_22.half_2.z * abs(dot(a_22.axis2_0, l_4)) + (b_30.half_2.x * abs(dot(b_30.axis0_0, l_4)) + b_30.half_2.y * abs(dot(b_30.axis1_0, l_4)) + b_30.half_2.z * abs(dot(b_30.axis2_0, l_4))) + margin_0 * len_0))
        {
            separated_1 = true;
        }
        else
        {
            separated_1 = separated_0;
        }
        separated_0 = separated_1;
        i_7 = i_7 + lanes_1;
    }
    return !pair_any_0(separated_0, lanes_1);
}

fn may_overlap_0( a_23 : Box_0,  b_31 : Box_0) -> bool
{
    var _S496 : vec3<f32> = b_31.center_1 - a_23.center_1;
    var _S497 : f32 = 0.00000999999974738f * (length(a_23.half_2) + length(b_31.half_2));
    var _S498 : array<vec3<f32>, i32(6)> = array<vec3<f32>, i32(6)>( a_23.axis0_0, a_23.axis1_0, a_23.axis2_0, b_31.axis0_0, b_31.axis1_0, b_31.axis2_0 );
    var i_8 : u32 = u32(0);
    loop
    {
        if(i_8 < u32(15))
        {
        }
        else
        {
            break;
        }
        var l_5 : vec3<f32>;
        if(i_8 < u32(6))
        {
            l_5 = _S498[i_8];
        }
        else
        {
            var _S499 : u32 = i_8 - u32(6);
            l_5 = cross(_S498[_S499 / u32(3)], _S498[u32(3) + _S499 % u32(3)]);
        }
        var len_1 : f32 = length(l_5);
        if(len_1 <= 9.99999997475242708e-07f)
        {
            i_8 = i_8 + u32(1);
            continue;
        }
        if((abs(dot(_S496, l_5))) > (a_23.half_2.x * abs(dot(a_23.axis0_0, l_5)) + a_23.half_2.y * abs(dot(a_23.axis1_0, l_5)) + a_23.half_2.z * abs(dot(a_23.axis2_0, l_5)) + (b_31.half_2.x * abs(dot(b_31.axis0_0, l_5)) + b_31.half_2.y * abs(dot(b_31.axis1_0, l_5)) + b_31.half_2.z * abs(dot(b_31.axis2_0, l_5))) + _S497 * len_1))
        {
            return false;
        }
        i_8 = i_8 + u32(1);
    }
    return true;
}

fn chunk_velocity_1( c_32 : u32,  v_15 : ptr<function, vec3<f32>>,  w_10 : ptr<function, vec3<f32>>)
{
    var q_9 : Quat_0 = quat_of_0(islands_0[chunks_0[c_32].info_1.y].rotation_0);
    var _S500 : u32 = u32(4) * c_32;
    var _S501 : vec3<f32> = islands_0[chunks_0[c_32].info_1.y].angular_velocity_0.xyz;
    (*v_15) = islands_0[chunks_0[c_32].info_1.y].velocity_0.xyz + islands_0[chunks_0[c_32].info_1.y].velocity_err_0.xyz + cross(_S501, rotate_0(q_9, chunks_0[c_32].center_0.xyz + state_0[_S500].xyz - islands_0[chunks_0[c_32].info_1.y].com_0.xyz)) + rotate_0(q_9, state_0[_S500 + u32(2)].xyz);
    (*w_10) = _S501 + rotate_0(q_9, state_0[_S500 + u32(3)].xyz);
    return;
}

fn box_to_local_0( b_32 : Box_0,  r_11 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(dot(r_11, b_32.axis0_0), dot(r_11, b_32.axis1_0), dot(r_11, b_32.axis2_0));
}

fn hull_signed_distance_0( b_33 : Box_0,  local_2 : vec3<f32>,  face_0 : ptr<function, u32>) -> f32
{
    (*face_0) = u32(0);
    var best_0 : f32 = -1.00000001504746622e+30f;
    var f_18 : u32 = u32(0);
    loop
    {
        if(f_18 < (b_33.hull_f_0))
        {
        }
        else
        {
            break;
        }
        var plane_0 : vec4<f32> = loads_0[b_33.hull_at_0 + b_33.hull_v_0 + f_18];
        var d_5 : f32 = dot(plane_0.xyz, local_2) - plane_0.w;
        if(d_5 > best_0)
        {
            (*face_0) = f_18;
            best_0 = d_5;
        }
        f_18 = f_18 + u32(1);
    }
    return best_0;
}

fn hull_signed_distance_1( b_34 : Box_0,  local_3 : vec3<f32>,  face_1 : ptr<function, u32>) -> f32
{
    (*face_1) = u32(0);
    var best_1 : f32 = -1.00000001504746622e+30f;
    var f_19 : u32 = u32(0);
    loop
    {
        if(f_19 < (b_34.hull_f_0))
        {
        }
        else
        {
            break;
        }
        var plane_1 : vec4<f32> = loads_0[b_34.hull_at_0 + b_34.hull_v_0 + f_19];
        var d_6 : f32 = dot(plane_1.xyz, local_3) - plane_1.w;
        if(d_6 > best_1)
        {
            (*face_1) = f_19;
            best_1 = d_6;
        }
        f_19 = f_19 + u32(1);
    }
    return best_1;
}

fn penetration_0( b_35 : Box_0,  p_5 : vec3<f32>,  depth_1 : ptr<function, f32>,  normal_3 : ptr<function, vec3<f32>>) -> bool
{
    (*depth_1) = 0.0f;
    (*normal_3) = vec3<f32>(0.0f);
    var r_12 : vec3<f32> = p_5 - b_35.center_1;
    var _S502 : vec3<f32> = b_35.half_2;
    if((dot(r_12, r_12)) > (dot(_S502, _S502) * 1.00001001358032227f))
    {
        return false;
    }
    if((b_35.hull_v_0) != u32(0))
    {
        var face_2 : u32;
        var d_7 : f32 = hull_signed_distance_0(b_35, box_to_local_0(b_35, r_12), &(face_2));
        if(!(d_7 < 0.0f))
        {
            return false;
        }
        (*depth_1) = - d_7;
        (*normal_3) = box_to_world_0(b_35, loads_0[b_35.hull_at_0 + b_35.hull_v_0 + face_2].xyz);
        return true;
    }
    var best_2 : f32 = 1.00000001504746622e+30f;
    var axis_3 : u32 = u32(0);
    var side_0 : f32 = 1.0f;
    var k_17 : u32 = u32(0);
    loop
    {
        if(k_17 < u32(3))
        {
        }
        else
        {
            break;
        }
        var local_4 : f32 = dot(r_12, box_axis_0(b_35, k_17));
        var d_8 : f32 = comp3_0(b_35.half_2, k_17) - abs(local_4);
        if(d_8 <= 0.0f)
        {
            return false;
        }
        if(d_8 < best_2)
        {
            var _S503 : f32;
            if(local_4 >= 0.0f)
            {
                _S503 = 1.0f;
            }
            else
            {
                _S503 = -1.0f;
            }
            best_2 = d_8;
            axis_3 = k_17;
            side_0 = _S503;
        }
        k_17 = k_17 + u32(1);
    }
    (*depth_1) = best_2;
    (*normal_3) = box_axis_0(b_35, axis_3) * vec3<f32>(side_0);
    return true;
}

fn penetration_1( b_36 : Box_0,  p_6 : vec3<f32>,  depth_2 : ptr<function, f32>,  normal_4 : ptr<function, vec3<f32>>) -> bool
{
    (*depth_2) = 0.0f;
    (*normal_4) = vec3<f32>(0.0f);
    var r_13 : vec3<f32> = p_6 - b_36.center_1;
    var _S504 : vec3<f32> = b_36.half_2;
    if((dot(r_13, r_13)) > (dot(_S504, _S504) * 1.00001001358032227f))
    {
        return false;
    }
    if((b_36.hull_v_0) != u32(0))
    {
        var face_3 : u32;
        var d_9 : f32 = hull_signed_distance_1(b_36, box_to_local_0(b_36, r_13), &(face_3));
        if(!(d_9 < 0.0f))
        {
            return false;
        }
        (*depth_2) = - d_9;
        (*normal_4) = box_to_world_0(b_36, loads_0[b_36.hull_at_0 + b_36.hull_v_0 + face_3].xyz);
        return true;
    }
    var best_3 : f32 = 1.00000001504746622e+30f;
    var axis_4 : u32 = u32(0);
    var side_1 : f32 = 1.0f;
    var k_18 : u32 = u32(0);
    loop
    {
        if(k_18 < u32(3))
        {
        }
        else
        {
            break;
        }
        var local_5 : f32 = dot(r_13, box_axis_0(b_36, k_18));
        var d_10 : f32 = comp3_0(b_36.half_2, k_18) - abs(local_5);
        if(d_10 <= 0.0f)
        {
            return false;
        }
        if(d_10 < best_3)
        {
            var _S505 : f32;
            if(local_5 >= 0.0f)
            {
                _S505 = 1.0f;
            }
            else
            {
                _S505 = -1.0f;
            }
            best_3 = d_10;
            axis_4 = k_18;
            side_1 = _S505;
        }
        k_18 = k_18 + u32(1);
    }
    (*depth_2) = best_3;
    (*normal_4) = box_axis_0(b_36, axis_4) * vec3<f32>(side_1);
    return true;
}

fn penetration_2( b_37 : Box_0,  p_7 : vec3<f32>,  depth_3 : ptr<function, f32>,  normal_5 : ptr<function, vec3<f32>>) -> bool
{
    (*depth_3) = 0.0f;
    (*normal_5) = vec3<f32>(0.0f);
    var r_14 : vec3<f32> = p_7 - b_37.center_1;
    var _S506 : vec3<f32> = b_37.half_2;
    if((dot(r_14, r_14)) > (dot(_S506, _S506) * 1.00001001358032227f))
    {
        return false;
    }
    if((b_37.hull_v_0) != u32(0))
    {
        var face_4 : u32;
        var d_11 : f32 = hull_signed_distance_1(b_37, box_to_local_0(b_37, r_14), &(face_4));
        if(!(d_11 < 0.0f))
        {
            return false;
        }
        (*depth_3) = - d_11;
        (*normal_5) = box_to_world_0(b_37, loads_0[b_37.hull_at_0 + b_37.hull_v_0 + face_4].xyz);
        return true;
    }
    var best_4 : f32 = 1.00000001504746622e+30f;
    var axis_5 : u32 = u32(0);
    var side_2 : f32 = 1.0f;
    var k_19 : u32 = u32(0);
    loop
    {
        if(k_19 < u32(3))
        {
        }
        else
        {
            break;
        }
        var local_6 : f32 = dot(r_14, box_axis_0(b_37, k_19));
        var d_12 : f32 = comp3_0(b_37.half_2, k_19) - abs(local_6);
        if(d_12 <= 0.0f)
        {
            return false;
        }
        if(d_12 < best_4)
        {
            var _S507 : f32;
            if(local_6 >= 0.0f)
            {
                _S507 = 1.0f;
            }
            else
            {
                _S507 = -1.0f;
            }
            best_4 = d_12;
            axis_5 = k_19;
            side_2 = _S507;
        }
        k_19 = k_19 + u32(1);
    }
    (*depth_3) = best_4;
    (*normal_5) = box_axis_0(b_37, axis_5) * vec3<f32>(side_2);
    return true;
}

fn pair_point_0( ba_0 : Box_0,  bb_0 : Box_0,  na_0 : u32,  e_4 : u32,  p_8 : ptr<function, vec3<f32>>,  n_3 : ptr<function, vec3<f32>>,  d_13 : ptr<function, f32>) -> bool
{
    if(e_4 < na_0)
    {
        var _S508 : vec3<f32> = sample_point_0(ba_0, e_4);
        (*p_8) = _S508;
        var _S509 : bool = penetration_2(bb_0, _S508, &((*d_13)), &((*n_3)));
        return _S509;
    }
    var _S510 : vec3<f32> = sample_point_0(bb_0, e_4 - na_0);
    (*p_8) = _S510;
    var _S511 : bool = penetration_2(ba_0, _S510, &((*d_13)), &((*n_3)));
    if(!_S511)
    {
        return false;
    }
    (*n_3) = (vec3<f32>(0) - (*n_3));
    return true;
}

fn is_nan_0( x_16 : f32) -> bool
{
    return (((bitcast<u32>((x_16))) & (u32(2147483647)))) > u32(2139095040);
}

fn pair_count_sum_0( x_17 : u32,  lanes_2 : u32) -> u32
{
    return x_17;
}

fn penalty_force_1( k_20 : f32,  m_red_1 : f32,  friction_1 : f32,  depth_4 : f32,  normal_6 : vec3<f32>,  rel_velocity_1 : vec3<f32>,  dt_15 : f32,  points_1 : u32,  stored_6 : ptr<function, f32>,  dissipated_9 : ptr<function, f32>) -> vec3<f32>
{
    var c_max_1 : f32 = 1.0f / max(f32(points_1), 10.0f) * m_red_1 / dt_15;
    var vn_1 : f32 = dot(rel_velocity_1, normal_6);
    var _S512 : f32 = k_20 * depth_4;
    var _S513 : f32 = min(2.0f * params_0.zeta_0 * sqrt(k_20 * m_red_1), c_max_1) * vn_1;
    var _S514 : f32 = _S512 - _S513;
    var _S515 : f32 = max(_S514, 0.0f);
    var vt_1 : vec3<f32> = rel_velocity_1 - normal_6 * vec3<f32>(vn_1);
    var vt_mag_1 : f32 = length(vt_1);
    var _S516 : f32 = friction_1 * _S515;
    var _S517 : f32 = min(_S516, min(c_max_1, _S516 / 0.00100000004749745f) * vt_mag_1);
    var ft_1 : vec3<f32>;
    if(vt_mag_1 > 0.0f)
    {
        ft_1 = (vec3<f32>(0) - vt_1) * vec3<f32>((_S517 / vt_mag_1));
    }
    else
    {
        ft_1 = vec3<f32>(0.0f);
    }
    (*stored_6) = 0.5f * k_20 * depth_4 * depth_4;
    var damping_power_1 : f32;
    if(_S514 > 0.0f)
    {
        damping_power_1 = _S513 * vn_1;
    }
    else
    {
        damping_power_1 = _S512 * max(vn_1, 0.0f);
    }
    (*dissipated_9) = (damping_power_1 + length(ft_1) * vt_mag_1) * dt_15;
    return normal_6 * vec3<f32>(_S515) + ft_1;
}

fn penalty_force_2( k_21 : f32,  m_red_2 : f32,  friction_2 : f32,  depth_5 : f32,  normal_7 : vec3<f32>,  rel_velocity_2 : vec3<f32>,  dt_16 : f32,  points_2 : u32,  stored_7 : ptr<function, f32>,  dissipated_10 : ptr<function, f32>) -> vec3<f32>
{
    var c_max_2 : f32 = 1.0f / max(f32(points_2), 10.0f) * m_red_2 / dt_16;
    var vn_2 : f32 = dot(rel_velocity_2, normal_7);
    var _S518 : f32 = k_21 * depth_5;
    var _S519 : f32 = min(2.0f * params_0.zeta_0 * sqrt(k_21 * m_red_2), c_max_2) * vn_2;
    var _S520 : f32 = _S518 - _S519;
    var _S521 : f32 = max(_S520, 0.0f);
    var vt_2 : vec3<f32> = rel_velocity_2 - normal_7 * vec3<f32>(vn_2);
    var vt_mag_2 : f32 = length(vt_2);
    var _S522 : f32 = friction_2 * _S521;
    var _S523 : f32 = min(_S522, min(c_max_2, _S522 / 0.00100000004749745f) * vt_mag_2);
    var ft_2 : vec3<f32>;
    if(vt_mag_2 > 0.0f)
    {
        ft_2 = (vec3<f32>(0) - vt_2) * vec3<f32>((_S523 / vt_mag_2));
    }
    else
    {
        ft_2 = vec3<f32>(0.0f);
    }
    (*stored_7) = 0.5f * k_21 * depth_5 * depth_5;
    var damping_power_2 : f32;
    if(_S520 > 0.0f)
    {
        damping_power_2 = _S519 * vn_2;
    }
    else
    {
        damping_power_2 = _S518 * max(vn_2, 0.0f);
    }
    (*dissipated_10) = (damping_power_2 + length(ft_2) * vt_mag_2) * dt_16;
    return normal_7 * vec3<f32>(_S521) + ft_2;
}

fn pair_sum3_0( x_18 : vec3<f32>,  lanes_3 : u32) -> vec3<f32>
{
    return x_18;
}

fn pair_sum_0( x_19 : f32,  lanes_4 : u32) -> f32
{
    return x_19;
}

fn comp_add1_4( sum_7 : ptr<function, f32>,  err_6 : ptr<function, f32>,  x_20 : f32)
{
    var t_21 : f32 = (*sum_7) + x_20;
    if((abs((*sum_7))) >= (abs(x_20)))
    {
        (*err_6) = (*err_6) + ((*sum_7) - t_21 + x_20);
    }
    else
    {
        (*err_6) = (*err_6) + (x_20 - t_21 + (*sum_7));
    }
    (*sum_7) = t_21;
    return;
}

fn comp_add1_5( sum_8 : ptr<function, f32>,  err_7 : ptr<function, f32>,  x_21 : f32)
{
    var t_22 : f32 = (*sum_8) + x_21;
    if((abs((*sum_8))) >= (abs(x_21)))
    {
        (*err_7) = (*err_7) + ((*sum_8) - t_22 + x_21);
    }
    else
    {
        (*err_7) = (*err_7) + (x_21 - t_22 + (*sum_8));
    }
    (*sum_8) = t_22;
    return;
}

fn pair_contact_0( i_9 : u32,  lane_1 : u32,  lanes_5 : u32)
{
    var at_5 : u32 = params_0.pair_index_0 + u32(6) * i_9;
    var ca_0 : u32 = index_0[at_5];
    var cb_0 : u32 = index_0[at_5 + u32(1)];
    var _S524 : u32 = index_0[at_5 + u32(3)];
    var _S525 : f32 = (bitcast<f32>((index_0[at_5 + u32(4)])));
    var _S526 : f32 = (bitcast<f32>((index_0[at_5 + u32(5)])));
    var _S527 : f32 = params_0.dt_0;
    var out_0 : u32 = params_0.slot_base_0 + u32(2) * index_0[at_5 + u32(2)];
    var cb_rel_0 : vec3<f32> = world_diff_0(chunk_world_0(cb_0), chunk_world_0(ca_0));
    var touching_0 : bool = !((length(cb_rel_0)) > (chunks_0[ca_0].half_0.w + chunks_0[cb_0].half_0.w));
    var ledger_2 : vec4<f32> = scratch_0[params_0.ledger_base_0 + i_9];
    var flags_1 : u32 = (bitcast<u32>((scratch_0[params_0.ledger_base_0 + i_9].w)));
    var _S528 : vec3<f32> = vec3<f32>(0.0f);
    var ba_1 : Box_0 = chunk_box_0(ca_0, _S528);
    var bb_1 : Box_0 = chunk_box_0(cb_0, cb_rel_0);
    var _S529 : u32 = sample_count_0(ba_1);
    var _S530 : u32 = _S529 + sample_count_0(bb_1);
    var has_state_0 : bool;
    var e_5 : u32;
    if(!touching_0)
    {
        if(((flags_1 & (u32(1)))) != u32(0))
        {
            e_5 = lane_1;
            loop
            {
                if(e_5 < _S530)
                {
                }
                else
                {
                    break;
                }
                contact_state_0[_S524 + e_5] = vec4<f32>((bitcast<f32>((u32(2143289344)))), 0.0f, 0.0f, 0.0f);
                e_5 = e_5 + lanes_5;
            }
        }
        if(lane_1 == u32(0))
        {
            if(((flags_1 & (u32(2)))) != u32(0))
            {
                e_5 = u32(0);
                loop
                {
                    if(e_5 < u32(4))
                    {
                    }
                    else
                    {
                        break;
                    }
                    scratch_0[out_0 + e_5] = vec4<f32>(0.0f);
                    e_5 = e_5 + u32(1);
                }
            }
            if(flags_1 != u32(0))
            {
                has_state_0 = true;
            }
            else
            {
                has_state_0 = (ledger_2.x) != 0.0f;
            }
            if(has_state_0)
            {
                ledger_2[i32(0)] = 0.0f;
                ledger_2[i32(3)] = (bitcast<f32>((u32(0))));
                scratch_0[params_0.ledger_base_0 + i_9] = ledger_2;
            }
        }
        return;
    }
    if(lanes_5 > u32(1))
    {
        has_state_0 = may_overlap_lanes_0(ba_1, bb_1, lane_1, lanes_5);
    }
    else
    {
        has_state_0 = may_overlap_0(ba_1, bb_1);
    }
    var count_3 : u32;
    var fa_2 : vec3<f32>;
    var ta_2 : vec3<f32>;
    var fb_0 : vec3<f32>;
    var tb_2 : vec3<f32>;
    var stored_sum_0 : f32;
    var diss_sum_0 : f32;
    if(has_state_0)
    {
        var va0_0 : vec3<f32>;
        var wa0_0 : vec3<f32>;
        chunk_velocity_1(ca_0, &(va0_0), &(wa0_0));
        var vb0_0 : vec3<f32>;
        var wb0_0 : vec3<f32>;
        chunk_velocity_1(cb_0, &(vb0_0), &(wb0_0));
        e_5 = lane_1;
        count_3 = u32(0);
        var engaged_0 : u32 = u32(0);
        loop
        {
            if(e_5 < _S530)
            {
            }
            else
            {
                break;
            }
            var p_9 : vec3<f32>;
            var n_4 : vec3<f32>;
            var d_14 : f32;
            var _S531 : bool = pair_point_0(ba_1, bb_1, _S529, e_5, &(p_9), &(n_4), &(d_14));
            if(!_S531)
            {
                var _S532 : u32 = _S524 + e_5;
                if(!is_nan_0(contact_state_0[_S532].x))
                {
                    contact_state_0[_S532] = vec4<f32>((bitcast<f32>((u32(2143289344)))), 0.0f, 0.0f, 0.0f);
                }
                e_5 = e_5 + lanes_5;
                continue;
            }
            var _S533 : u32 = count_3 + u32(1);
            var _S534 : u32 = _S524 + e_5;
            var entry_4 : vec4<f32> = contact_state_0[_S534];
            if(is_nan_0(contact_state_0[_S534].x))
            {
                has_state_0 = true;
            }
            else
            {
                has_state_0 = (dot(entry_4.yzw, n_4)) < 0.99000000953674316f;
            }
            if(has_state_0)
            {
                if(d_14 > (2.0f * abs(dot(va0_0 + cross(wa0_0, p_9) - (vb0_0 + cross(wb0_0, p_9 - bb_1.center_1)), n_4)) * _S527 + 9.99999971718068537e-10f))
                {
                    stored_sum_0 = d_14;
                }
                else
                {
                    stored_sum_0 = 0.0f;
                }
                entry_4 = vec4<f32>(stored_sum_0, n_4);
            }
            entry_4[i32(0)] = min(entry_4.x, d_14);
            contact_state_0[_S534] = entry_4;
            var engaged_1 : u32;
            if((d_14 - entry_4.x) > 0.0f)
            {
                engaged_1 = engaged_0 + u32(1);
            }
            else
            {
                engaged_1 = engaged_0;
            }
            count_3 = _S533;
            engaged_0 = engaged_1;
            e_5 = e_5 + lanes_5;
        }
        var engaged_2 : u32 = pair_count_sum_0(engaged_0, lanes_5);
        if((pair_count_sum_0(count_3, lanes_5)) > u32(0))
        {
            var _S535 : f32 = contact_stiffness_0(chunks_0[ca_0].cmat_0.x, ba_1, chunks_0[cb_0].cmat_0.x, bb_1, bb_1.center_1 - ba_1.center_1) / max(f32(engaged_2), 10.0f);
            e_5 = lane_1;
            fa_2 = _S528;
            ta_2 = _S528;
            fb_0 = _S528;
            tb_2 = _S528;
            stored_sum_0 = 0.0f;
            diss_sum_0 = 0.0f;
            loop
            {
                if(e_5 < _S530)
                {
                }
                else
                {
                    break;
                }
                var p_10 : vec3<f32>;
                var n_5 : vec3<f32>;
                var d_15 : f32;
                var _S536 : bool = pair_point_0(ba_1, bb_1, _S529, e_5, &(p_10), &(n_5), &(d_15));
                if(!_S536)
                {
                    e_5 = e_5 + lanes_5;
                    continue;
                }
                var eff_0 : f32 = d_15 - contact_state_0[_S524 + e_5].x;
                if(eff_0 <= 0.0f)
                {
                    e_5 = e_5 + lanes_5;
                    continue;
                }
                var stored_8 : f32;
                var diss_4 : f32;
                var f_20 : vec3<f32> = penalty_force_2(_S535, _S525, _S526, eff_0, n_5, va0_0 + cross(wa0_0, p_10) - (vb0_0 + cross(wb0_0, p_10 - bb_1.center_1)), _S527, engaged_2, &(stored_8), &(diss_4));
                var ta_3 : vec3<f32> = ta_2 + cross(p_10, f_20);
                var _S537 : vec3<f32> = (vec3<f32>(0) - f_20);
                var fb_1 : vec3<f32> = fb_0 + _S537;
                var tb_3 : vec3<f32> = tb_2 + cross(p_10 - bb_1.center_1, _S537);
                var stored_sum_1 : f32 = stored_sum_0 + stored_8;
                var diss_sum_1 : f32 = diss_sum_0 + diss_4;
                fa_2 = fa_2 + f_20;
                ta_2 = ta_3;
                fb_0 = fb_1;
                tb_2 = tb_3;
                stored_sum_0 = stored_sum_1;
                diss_sum_0 = diss_sum_1;
                e_5 = e_5 + lanes_5;
            }
            var _S538 : vec3<f32> = pair_sum3_0(fa_2, lanes_5);
            var _S539 : vec3<f32> = pair_sum3_0(ta_2, lanes_5);
            var _S540 : vec3<f32> = pair_sum3_0(fb_0, lanes_5);
            var _S541 : vec3<f32> = pair_sum3_0(tb_2, lanes_5);
            var _S542 : f32 = pair_sum_0(stored_sum_0, lanes_5);
            var _S543 : f32 = pair_sum_0(diss_sum_0, lanes_5);
            has_state_0 = true;
            diss_sum_0 = _S543;
            fa_2 = _S538;
            ta_2 = _S539;
            fb_0 = _S540;
            tb_2 = _S541;
            stored_sum_0 = _S542;
        }
        else
        {
            has_state_0 = false;
            diss_sum_0 = 0.0f;
            fa_2 = _S528;
            ta_2 = _S528;
            fb_0 = _S528;
            tb_2 = _S528;
            stored_sum_0 = 0.0f;
        }
    }
    else
    {
        has_state_0 = false;
        diss_sum_0 = 0.0f;
        fa_2 = _S528;
        ta_2 = _S528;
        fb_0 = _S528;
        tb_2 = _S528;
        stored_sum_0 = 0.0f;
    }
    var loaded_0 : bool;
    if(!has_state_0)
    {
        loaded_0 = ((flags_1 & (u32(1)))) != u32(0);
    }
    else
    {
        loaded_0 = false;
    }
    if(loaded_0)
    {
        e_5 = lane_1;
        loop
        {
            if(e_5 < _S530)
            {
            }
            else
            {
                break;
            }
            contact_state_0[_S524 + e_5] = vec4<f32>((bitcast<f32>((u32(2143289344)))), 0.0f, 0.0f, 0.0f);
            e_5 = e_5 + lanes_5;
        }
    }
    if(lane_1 != u32(0))
    {
        return;
    }
    if(has_state_0)
    {
        var _S544 : f32 = ledger_2[i32(1)];
        var _S545 : f32 = ledger_2[i32(2)];
        comp_add1_5(&(_S544), &(_S545), diss_sum_0);
        ledger_2[i32(1)] = _S544;
        ledger_2[i32(2)] = _S545;
    }
    var _S546 : vec3<f32> = vec3<f32>(0.0f);
    if((any((fa_2 != _S546))))
    {
        loaded_0 = true;
    }
    else
    {
        loaded_0 = (any((ta_2 != _S546)));
    }
    if(loaded_0)
    {
        loaded_0 = true;
    }
    else
    {
        loaded_0 = (any((fb_0 != _S546)));
    }
    if(loaded_0)
    {
        loaded_0 = true;
    }
    else
    {
        loaded_0 = (any((tb_2 != _S546)));
    }
    var _S547 : bool;
    if(loaded_0)
    {
        _S547 = true;
    }
    else
    {
        _S547 = ((flags_1 & (u32(2)))) != u32(0);
    }
    if(_S547)
    {
        scratch_0[out_0] = vec4<f32>(fa_2, 0.0f);
        scratch_0[out_0 + u32(1)] = vec4<f32>(ta_2, 0.0f);
        scratch_0[out_0 + u32(2)] = vec4<f32>(fb_0, 0.0f);
        scratch_0[out_0 + u32(3)] = vec4<f32>(tb_2, 0.0f);
    }
    ledger_2[i32(0)] = stored_sum_0;
    if(has_state_0)
    {
        e_5 = u32(1);
    }
    else
    {
        e_5 = u32(0);
    }
    if(loaded_0)
    {
        count_3 = u32(2);
    }
    else
    {
        count_3 = u32(0);
    }
    ledger_2[i32(3)] = (bitcast<f32>(((e_5 | (count_3)))));
    scratch_0[params_0.ledger_base_0 + i_9] = ledger_2;
    if(loaded_0)
    {
        sentinel_mark_0(u32(1));
    }
    return;
}

struct Impactor_0
{
     position_1 : vec4<f32>,
     position_err_1 : vec4<f32>,
     velocity_1 : vec4<f32>,
     velocity_err_1 : vec4<f32>,
     angular_velocity_1 : vec4<f32>,
     rotation_1 : vec4<f32>,
     inertia0_2 : vec4<f32>,
     inertia1_2 : vec4<f32>,
     inertia2_2 : vec4<f32>,
     inv0_2 : vec4<f32>,
     inv1_2 : vec4<f32>,
     inv2_2 : vec4<f32>,
     shape_0 : vec4<f32>,
     half_1 : vec4<f32>,
     mat_0 : vec4<f32>,
     crush_0 : vec4<f32>,
     geom_0 : vec4<f32>,
     rotation_err_1 : vec4<f32>,
     ledger_0 : vec4<f32>,
     cand_0 : vec4<u32>,
     momentum_1 : vec4<f32>,
     momentum_err_1 : vec4<f32>,
};

fn impactor_box_0( imp_0 : Impactor_0,  center_3 : vec3<f32>,  half_3 : vec3<f32>) -> Box_0
{
    var q_10 : Quat_0 = quat_of_0(imp_0.rotation_1);
    var b_38 : Box_0;
    b_38.center_1 = center_3;
    b_38.axis0_0 = rotate_0(q_10, vec3<f32>(1.0f, 0.0f, 0.0f));
    b_38.axis1_0 = rotate_0(q_10, vec3<f32>(0.0f, 1.0f, 0.0f));
    b_38.axis2_0 = rotate_0(q_10, vec3<f32>(0.0f, 0.0f, 1.0f));
    b_38.half_2 = half_3;
    b_38.hull_at_0 = u32(0);
    b_38.hull_v_0 = u32(0);
    b_38.hull_f_0 = u32(0);
    return b_38;
}

fn impactor_slots_0( imp_1 : ptr<function, Impactor_std430_0>,  b_39 : Box_0) -> u32
{
    var _S548 : u32;
    if(((*imp_1).shape_0.x) == 0.0f)
    {
        _S548 = u32(1);
    }
    else
    {
        _S548 = sample_count_0(b_39) + u32(14);
    }
    return _S548;
}

fn impactor_slots_1( imp_2 : ptr<function, Impactor_std430_0>,  b_40 : Box_0) -> u32
{
    var _S549 : u32;
    if(((*imp_2).shape_0.x) == 0.0f)
    {
        _S549 = u32(1);
    }
    else
    {
        _S549 = sample_count_0(b_40) + u32(14);
    }
    return _S549;
}

fn impactor_slots_2( imp_3 : ptr<function, Impactor_std430_0>,  b_41 : Box_0) -> u32
{
    var _S550 : u32;
    if(((*imp_3).shape_0.x) == 0.0f)
    {
        _S550 = u32(1);
    }
    else
    {
        _S550 = sample_count_0(b_41) + u32(14);
    }
    return _S550;
}

fn sphere_contact_0( b_42 : Box_0,  center_4 : vec3<f32>,  radius_0 : f32,  point_0 : ptr<function, vec3<f32>>,  normal_8 : ptr<function, vec3<f32>>,  depth_6 : ptr<function, f32>) -> bool
{
    var _S551 : vec3<f32> = vec3<f32>(0.0f);
    (*point_0) = _S551;
    (*normal_8) = _S551;
    (*depth_6) = 0.0f;
    var r_15 : vec3<f32> = center_4 - b_42.center_1;
    var local_7 : vec3<f32> = vec3<f32>(dot(r_15, b_42.axis0_0), dot(r_15, b_42.axis1_0), dot(r_15, b_42.axis2_0));
    if((b_42.hull_v_0) != u32(0))
    {
        var face_5 : u32;
        var d_16 : f32 = hull_signed_distance_0(b_42, local_7, &(face_5));
        if(d_16 >= radius_0)
        {
            return false;
        }
        var _S552 : vec3<f32> = box_to_world_0(b_42, loads_0[b_42.hull_at_0 + b_42.hull_v_0 + face_5].xyz);
        (*normal_8) = _S552;
        (*point_0) = center_4 - _S552 * vec3<f32>(max(d_16, 0.0f));
        (*depth_6) = radius_0 - d_16;
        return true;
    }
    var q_11 : vec3<f32> = clamp(local_7, (vec3<f32>(0) - b_42.half_2), b_42.half_2);
    var d_17 : vec3<f32> = local_7 - q_11;
    var dist_0 : f32 = length(d_17);
    if(dist_0 > 9.999999960041972e-13f)
    {
        if(dist_0 >= radius_0)
        {
            return false;
        }
        var dn_0 : vec3<f32> = d_17 / vec3<f32>(dist_0);
        (*normal_8) = b_42.axis0_0 * vec3<f32>(dn_0.x) + b_42.axis1_0 * vec3<f32>(dn_0.y) + b_42.axis2_0 * vec3<f32>(dn_0.z);
        (*point_0) = b_42.center_1 + b_42.axis0_0 * vec3<f32>(q_11.x) + b_42.axis1_0 * vec3<f32>(q_11.y) + b_42.axis2_0 * vec3<f32>(q_11.z);
        (*depth_6) = radius_0 - dist_0;
        return true;
    }
    var inside_0 : f32;
    var n_6 : vec3<f32>;
    var _S553 : bool = penetration_0(b_42, center_4, &(inside_0), &(n_6));
    if(!_S553)
    {
        return false;
    }
    (*normal_8) = n_6;
    (*point_0) = center_4 - n_6 * vec3<f32>(min(radius_0, inside_0));
    (*depth_6) = radius_0 + inside_0;
    return true;
}

fn sphere_contact_1( b_43 : Box_0,  center_5 : vec3<f32>,  radius_1 : f32,  point_1 : ptr<function, vec3<f32>>,  normal_9 : ptr<function, vec3<f32>>,  depth_7 : ptr<function, f32>) -> bool
{
    var _S554 : vec3<f32> = vec3<f32>(0.0f);
    (*point_1) = _S554;
    (*normal_9) = _S554;
    (*depth_7) = 0.0f;
    var r_16 : vec3<f32> = center_5 - b_43.center_1;
    var local_8 : vec3<f32> = vec3<f32>(dot(r_16, b_43.axis0_0), dot(r_16, b_43.axis1_0), dot(r_16, b_43.axis2_0));
    if((b_43.hull_v_0) != u32(0))
    {
        var face_6 : u32;
        var d_18 : f32 = hull_signed_distance_1(b_43, local_8, &(face_6));
        if(d_18 >= radius_1)
        {
            return false;
        }
        var _S555 : vec3<f32> = box_to_world_0(b_43, loads_0[b_43.hull_at_0 + b_43.hull_v_0 + face_6].xyz);
        (*normal_9) = _S555;
        (*point_1) = center_5 - _S555 * vec3<f32>(max(d_18, 0.0f));
        (*depth_7) = radius_1 - d_18;
        return true;
    }
    var q_12 : vec3<f32> = clamp(local_8, (vec3<f32>(0) - b_43.half_2), b_43.half_2);
    var d_19 : vec3<f32> = local_8 - q_12;
    var dist_1 : f32 = length(d_19);
    if(dist_1 > 9.999999960041972e-13f)
    {
        if(dist_1 >= radius_1)
        {
            return false;
        }
        var dn_1 : vec3<f32> = d_19 / vec3<f32>(dist_1);
        (*normal_9) = b_43.axis0_0 * vec3<f32>(dn_1.x) + b_43.axis1_0 * vec3<f32>(dn_1.y) + b_43.axis2_0 * vec3<f32>(dn_1.z);
        (*point_1) = b_43.center_1 + b_43.axis0_0 * vec3<f32>(q_12.x) + b_43.axis1_0 * vec3<f32>(q_12.y) + b_43.axis2_0 * vec3<f32>(q_12.z);
        (*depth_7) = radius_1 - dist_1;
        return true;
    }
    var inside_1 : f32;
    var n_7 : vec3<f32>;
    var _S556 : bool = penetration_1(b_43, center_5, &(inside_1), &(n_7));
    if(!_S556)
    {
        return false;
    }
    (*normal_9) = n_7;
    (*point_1) = center_5 - n_7 * vec3<f32>(min(radius_1, inside_1));
    (*depth_7) = radius_1 + inside_1;
    return true;
}

fn impactor_contact_0( imp_4 : ptr<function, Impactor_std430_0>,  crush_depth_0 : f32,  shrunk_0 : Box_0,  b_44 : Box_0,  j_2 : u32,  p_11 : ptr<function, vec3<f32>>,  n_8 : ptr<function, vec3<f32>>,  d_20 : ptr<function, f32>) -> bool
{
    var _S557 : vec3<f32> = vec3<f32>(0.0f);
    (*p_11) = _S557;
    (*n_8) = _S557;
    (*d_20) = 0.0f;
    var _S558 : vec4<f32> = (*imp_4).shape_0;
    if(((*imp_4).shape_0.x) == 0.0f)
    {
        var _S559 : bool = sphere_contact_0(b_44, _S557, _S558.y - crush_depth_0, &((*p_11)), &((*n_8)), &((*d_20)));
        if(!_S559)
        {
            return false;
        }
        (*n_8) = (vec3<f32>(0) - (*n_8));
        return true;
    }
    var cb_1 : u32 = sample_count_0(b_44);
    if(j_2 < cb_1)
    {
        var _S560 : vec3<f32> = sample_point_0(b_44, j_2);
        (*p_11) = _S560;
        var _S561 : bool = penetration_0(shrunk_0, _S560, &((*d_20)), &((*n_8)));
        return _S561;
    }
    var _S562 : vec3<f32> = sample_point_0(shrunk_0, j_2 - cb_1);
    (*p_11) = _S562;
    var _S563 : bool = penetration_0(b_44, _S562, &((*d_20)), &((*n_8)));
    if(!_S563)
    {
        return false;
    }
    (*n_8) = (vec3<f32>(0) - (*n_8));
    return true;
}

fn impactor_contact_1( imp_5 : ptr<function, Impactor_std430_0>,  crush_depth_1 : f32,  shrunk_1 : Box_0,  b_45 : Box_0,  j_3 : u32,  p_12 : ptr<function, vec3<f32>>,  n_9 : ptr<function, vec3<f32>>,  d_21 : ptr<function, f32>) -> bool
{
    var _S564 : vec3<f32> = vec3<f32>(0.0f);
    (*p_12) = _S564;
    (*n_9) = _S564;
    (*d_21) = 0.0f;
    var _S565 : vec4<f32> = (*imp_5).shape_0;
    if(((*imp_5).shape_0.x) == 0.0f)
    {
        var _S566 : bool = sphere_contact_1(b_45, _S564, _S565.y - crush_depth_1, &((*p_12)), &((*n_9)), &((*d_21)));
        if(!_S566)
        {
            return false;
        }
        (*n_9) = (vec3<f32>(0) - (*n_9));
        return true;
    }
    var cb_2 : u32 = sample_count_0(b_45);
    if(j_3 < cb_2)
    {
        var _S567 : vec3<f32> = sample_point_0(b_45, j_3);
        (*p_12) = _S567;
        var _S568 : bool = penetration_1(shrunk_1, _S567, &((*d_21)), &((*n_9)));
        return _S568;
    }
    var _S569 : vec3<f32> = sample_point_0(shrunk_1, j_3 - cb_2);
    (*p_12) = _S569;
    var _S570 : bool = penetration_1(b_45, _S569, &((*d_21)), &((*n_9)));
    if(!_S570)
    {
        return false;
    }
    (*n_9) = (vec3<f32>(0) - (*n_9));
    return true;
}

fn impactor_point_0( _S571 : u32) -> WorldPoint_0
{
    var wi_0 : WorldPoint_0;
    wi_0.hi_0 = impactors_0[_S571].position_1.xyz;
    wi_0.lo_0 = impactors_0[_S571].position_err_1.xyz;
    wi_0.rel_0 = vec3<f32>(0.0f);
    return wi_0;
}

fn impactor_box_1( _S572 : u32,  _S573 : vec3<f32>,  _S574 : vec3<f32>) -> Box_0
{
    var q_13 : Quat_0 = quat_of_0(impactors_0[_S572].rotation_1);
    var b_46 : Box_0;
    b_46.center_1 = _S573;
    b_46.axis0_0 = rotate_0(q_13, vec3<f32>(1.0f, 0.0f, 0.0f));
    b_46.axis1_0 = rotate_0(q_13, vec3<f32>(0.0f, 1.0f, 0.0f));
    b_46.axis2_0 = rotate_0(q_13, vec3<f32>(0.0f, 0.0f, 1.0f));
    b_46.half_2 = _S574;
    b_46.hull_at_0 = u32(0);
    b_46.hull_v_0 = u32(0);
    b_46.hull_f_0 = u32(0);
    return b_46;
}

fn impactor_shrunk_0( _S575 : u32,  _S576 : f32,  _S577 : Box_0) -> Box_0
{
    var shrunk_2 : Box_0 = _S577;
    shrunk_2.half_2 = _S577.half_2 - min(vec3<f32>(_S576), _S577.half_2 * vec3<f32>(0.5f));
    return shrunk_2;
}

fn impactor_contact_2( _S578 : u32,  _S579 : f32,  _S580 : Box_0,  _S581 : Box_0,  _S582 : u32,  _S583 : ptr<function, vec3<f32>>,  _S584 : ptr<function, vec3<f32>>,  _S585 : ptr<function, f32>) -> bool
{
    var _S586 : Impactor_std430_0 = impactors_0[_S578];
    var _S587 : vec3<f32> = vec3<f32>(0.0f);
    (*_S583) = _S587;
    (*_S584) = _S587;
    (*_S585) = 0.0f;
    if((_S586.shape_0.x) == 0.0f)
    {
        var _S588 : bool = sphere_contact_1(_S581, _S587, _S586.shape_0.y - _S579, &((*_S583)), &((*_S584)), &((*_S585)));
        if(!_S588)
        {
            return false;
        }
        (*_S584) = (vec3<f32>(0) - (*_S584));
        return true;
    }
    var cb_3 : u32 = sample_count_0(_S581);
    if(_S582 < cb_3)
    {
        var _S589 : vec3<f32> = sample_point_0(_S581, _S582);
        (*_S583) = _S589;
        var _S590 : bool = penetration_1(_S580, _S589, &((*_S585)), &((*_S584)));
        return _S590;
    }
    var _S591 : vec3<f32> = sample_point_0(_S580, _S582 - cb_3);
    (*_S583) = _S591;
    var _S592 : bool = penetration_1(_S581, _S591, &((*_S585)), &((*_S584)));
    if(!_S592)
    {
        return false;
    }
    (*_S584) = (vec3<f32>(0) - (*_S584));
    return true;
}

fn impactor_candidate_forces_0( k_22 : u32,  lane_2 : u32,  lanes_6 : u32)
{
    var _S593 : u32;
    var _S594 : bool;
    var _S595 : u32 = u32(3) * k_22;
    var at_6 : u32 = params_0.cand_index_0 + _S595;
    var c_33 : u32 = index_0[at_6];
    var slot_1 : u32 = index_0[at_6 + u32(1)];
    var _S596 : u32 = index_0[at_6 + u32(2)];
    var _S597 : Impactor_std430_0 = impactors_0[_S596];
    var _S598 : f32 = params_0.dt_0;
    var _S599 : vec3<f32> = vec3<f32>(0.0f);
    var data_1 : vec4<f32> = scratch_0[params_0.cand_base_0 + _S595];
    var hit_0 : bool;
    var f_sum_0 : vec3<f32>;
    var t_sum_0 : vec3<f32>;
    var imp_f_0 : vec3<f32>;
    var imp_t_0 : vec3<f32>;
    if((_S597.cand_0.z) == u32(0))
    {
        var rel_1 : vec3<f32> = world_diff_0(chunk_world_0(c_33), impactor_point_0(_S596));
        var _S600 : vec4<f32> = _S597.half_1;
        if(!((length(rel_1)) > (_S597.half_1.w + chunks_0[c_33].half_0.w)))
        {
            var _S601 : Box_0 = impactor_box_1(_S596, _S599, _S600.xyz);
            var b_47 : Box_0 = chunk_box_0(c_33, rel_1);
            var _S602 : vec4<f32> = _S597.mat_0;
            var kc_0 : f32 = contact_stiffness_0(_S597.mat_0.x, _S601, chunks_0[c_33].cmat_0.x, b_47, b_47.center_1 - _S601.center_1);
            var _S603 : vec4<f32> = _S597.geom_0;
            var _S604 : f32 = _S597.geom_0.y;
            var _S605 : Box_0 = impactor_shrunk_0(_S596, _S604, _S601);
            var j_4 : u32 = lane_2;
            var n_10 : u32 = u32(0);
            loop
            {
                var _S606 : u32 = impactor_slots_0(&(_S597), b_47);
                _S593 = _S606;
                if(j_4 < _S606)
                {
                }
                else
                {
                    break;
                }
                var p_13 : vec3<f32>;
                var nrm_0 : vec3<f32>;
                var dep_0 : f32;
                if(impactor_contact_2(_S596, _S604, _S605, b_47, j_4, &(p_13), &(nrm_0), &(dep_0)))
                {
                    n_10 = n_10 + u32(1);
                }
                j_4 = j_4 + lanes_6;
            }
            var n_11 : u32 = pair_count_sum_0(n_10, lanes_6);
            var _S607 : bool = (_S597.shape_0.x) == 0.0f;
            var _S608 : f32;
            if(_S607)
            {
                _S608 = kc_0;
            }
            else
            {
                _S608 = kc_0 / max(f32(n_11), 10.0f);
            }
            if(_S607)
            {
                j_4 = u32(1);
            }
            else
            {
                j_4 = n_11;
            }
            var m_5 : f32 = chunks_0[c_33].center_0.w;
            var _S609 : f32 = _S602.z;
            var _S610 : f32 = m_5 * _S609 / (m_5 + _S609);
            var _S611 : f32;
            if((params_0.pair_friction_0) >= 0.0f)
            {
                _S611 = params_0.pair_friction_0;
            }
            else
            {
                _S611 = min(_S602.y, chunks_0[c_33].cmat_0.y);
            }
            var _S612 : vec3<f32> = _S597.velocity_1.xyz + _S597.velocity_err_1.xyz;
            var vc_1 : vec3<f32>;
            var wc_1 : vec3<f32>;
            chunk_velocity_1(c_33, &(vc_1), &(wc_1));
            hit_0 = false;
            var j_5 : u32 = lane_2;
            f_sum_0 = _S599;
            t_sum_0 = _S599;
            imp_f_0 = _S599;
            imp_t_0 = _S599;
            var diss_sum_2 : f32 = 0.0f;
            loop
            {
                var _S613 : bool = n_11 > u32(0);
                _S594 = _S613;
                var _S614 : bool;
                if(_S613)
                {
                    _S614 = j_5 < _S593;
                }
                else
                {
                    _S614 = false;
                }
                if(_S614)
                {
                }
                else
                {
                    break;
                }
                var p_14 : vec3<f32>;
                var nrm_1 : vec3<f32>;
                var dep_1 : f32;
                var _S615 : bool = impactor_contact_0(&(_S597), _S604, _S605, b_47, j_5, &(p_14), &(nrm_1), &(dep_1));
                if(!_S615)
                {
                    j_5 = j_5 + lanes_6;
                    continue;
                }
                var stored_9 : f32;
                var diss_5 : f32;
                var f_21 : vec3<f32> = penalty_force_2(_S608, _S610, _S611, dep_1 * _S603.x, nrm_1, vc_1 + cross(wc_1, p_14 - b_47.center_1) - (_S612 + cross(_S597.angular_velocity_1.xyz, p_14)), _S598, j_4, &(stored_9), &(diss_5));
                var f_sum_1 : vec3<f32> = f_sum_0 + f_21;
                var t_sum_1 : vec3<f32> = t_sum_0 + cross(p_14 - b_47.center_1, f_21);
                var imp_f_1 : vec3<f32> = imp_f_0 - f_21;
                var imp_t_1 : vec3<f32> = imp_t_0 - cross(p_14, f_21);
                var diss_sum_3 : f32 = diss_sum_2 + diss_5;
                hit_0 = true;
                f_sum_0 = f_sum_1;
                t_sum_0 = t_sum_1;
                imp_f_0 = imp_f_1;
                imp_t_0 = imp_t_1;
                diss_sum_2 = diss_sum_3;
                j_5 = j_5 + lanes_6;
            }
            var _S616 : vec3<f32> = pair_sum3_0(f_sum_0, lanes_6);
            var _S617 : vec3<f32> = pair_sum3_0(t_sum_0, lanes_6);
            var _S618 : vec3<f32> = pair_sum3_0(imp_f_0, lanes_6);
            var _S619 : vec3<f32> = pair_sum3_0(imp_t_0, lanes_6);
            var diss_sum_4 : f32 = pair_sum_0(diss_sum_2, lanes_6);
            var _S620 : bool = pair_any_0(hit_0, lanes_6);
            if(_S594)
            {
                var _S621 : f32 = data_1[i32(2)];
                var _S622 : f32 = data_1[i32(3)];
                comp_add1_5(&(_S621), &(_S622), diss_sum_4);
                data_1[i32(2)] = _S621;
                data_1[i32(3)] = _S622;
            }
            hit_0 = _S620;
            f_sum_0 = _S616;
            t_sum_0 = _S617;
            imp_f_0 = _S618;
            imp_t_0 = _S619;
        }
        else
        {
            hit_0 = false;
            f_sum_0 = _S599;
            t_sum_0 = _S599;
            imp_f_0 = _S599;
            imp_t_0 = _S599;
        }
    }
    else
    {
        hit_0 = false;
        f_sum_0 = _S599;
        t_sum_0 = _S599;
        imp_f_0 = _S599;
        imp_t_0 = _S599;
    }
    if(lane_2 != u32(0))
    {
        return;
    }
    var _S623 : u32 = u32(2) * slot_1;
    scratch_0[params_0.slot_base_0 + _S623] = vec4<f32>(f_sum_0, 0.0f);
    scratch_0[params_0.slot_base_0 + _S623 + u32(1)] = vec4<f32>(t_sum_0, 0.0f);
    scratch_0[params_0.cand_base_0 + _S595] = data_1;
    scratch_0[params_0.cand_base_0 + _S595 + u32(1)] = vec4<f32>(imp_f_0, 0.0f);
    scratch_0[params_0.cand_base_0 + _S595 + u32(2)] = vec4<f32>(imp_t_0, 0.0f);
    if(hit_0)
    {
        sentinel_mark_0(u32(2));
    }
    return;
}

fn impactor_travel_check_0( ii_0 : u32,  fused_2 : bool)
{
    var at_7 : u32 = params_0.iplan_base_0 + u32(4) * ii_0;
    var start_hi_1 : vec4<f32> = scratch_0[at_7];
    var _S624 : vec4<u32> = impactors_0[ii_0].cand_0;
    var _S625 : bool;
    if((impactors_0[ii_0].cand_0.z) != u32(0))
    {
        _S625 = true;
    }
    else
    {
        _S625 = (start_hi_1.w) <= 0.0f;
    }
    if(_S625)
    {
        return;
    }
    var _S626 : u32 = _S624.w;
    var _S627 : vec3<f32> = vec3<f32>((params_0.t_hi_0 - scratch_0[at_7 + u32(3)].w + (params_0.t_lo_0 - scratch_0[at_7 + u32(1)].w) + f32(_S626 - params_0.step_start_0) * params_0.dt_0));
    var turn_1 : Quat_0 = quat_mul_0(quat_of_0(vec4<f32>((vec3<f32>(0) - scratch_0[at_7 + u32(2)].xyz), scratch_0[at_7 + u32(2)].w)), quat_of_0(impactors_0[ii_0].rotation_1));
    if((length(impactors_0[ii_0].position_1.xyz - start_hi_1.xyz + (impactors_0[ii_0].position_err_1.xyz - scratch_0[at_7 + u32(1)].xyz) - (scratch_0[at_7 + u32(3)].xyz * _S627 + vec3<f32>(0.5f) * params_0.gravity_0.xyz * _S627 * _S627)) + 2.0f * atan2(length(vec3<f32>(turn_1.x_0, turn_1.y_0, turn_1.z_0)), abs(turn_1.w_0)) * impactors_0[ii_0].half_1.w) > (start_hi_1.w))
    {
        travel_stop_0(_S626 + u32(1), fused_2);
    }
    return;
}

fn contact_forces_at_0( i_10 : u32,  fused_3 : bool)
{
    var lanes_7 : u32 = pair_lane_count_0();
    var pair_threads_0 : u32 = params_0.pair_count_0 * lanes_7;
    if(i_10 < pair_threads_0)
    {
        if(!pair_any_0(stopped_0(), lanes_7))
        {
            var _S628 : u32 = i_10 / lanes_7;
            var _S629 : u32 = i_10 % lanes_7;
            pair_contact_0(_S628, _S629, lanes_7);
        }
        return;
    }
    var j_6 : u32 = i_10 - pair_threads_0;
    var cand_threads_0 : u32 = params_0.cand_count_0 * lanes_7;
    if(j_6 < cand_threads_0)
    {
        if(!pair_any_0(stopped_0(), lanes_7))
        {
            var _S630 : u32 = j_6 / lanes_7;
            var _S631 : u32 = j_6 % lanes_7;
            impactor_candidate_forces_0(_S630, _S631, lanes_7);
        }
        return;
    }
    if(fused_3)
    {
        return;
    }
    var _S632 : u32 = j_6 - cand_threads_0;
    var _S633 : bool;
    if(_S632 < (params_0.chunk_count_0))
    {
        _S633 = !stopped_0();
    }
    else
    {
        _S633 = false;
    }
    if(_S633)
    {
        travel_check_0(_S632, false);
    }
    else
    {
        if((_S632 - params_0.chunk_count_0) < (params_0.impactor_count_0))
        {
            _S633 = !stopped_0();
        }
        else
        {
            _S633 = false;
        }
        if(_S633)
        {
            impactor_travel_check_0(_S632 - params_0.chunk_count_0, false);
        }
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn contact_forces(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    contact_forces_at_0(id_0.x, false);
    return;
}

fn impactor_contact_3( _S634 : u32,  _S635 : f32,  _S636 : Box_0,  _S637 : Box_0,  _S638 : u32,  _S639 : ptr<function, vec3<f32>>,  _S640 : ptr<function, vec3<f32>>,  _S641 : ptr<function, f32>) -> bool
{
    var _S642 : Impactor_std430_0 = impactors_0[_S634];
    var _S643 : vec3<f32> = vec3<f32>(0.0f);
    (*_S639) = _S643;
    (*_S640) = _S643;
    (*_S641) = 0.0f;
    if((_S642.shape_0.x) == 0.0f)
    {
        var _S644 : bool = sphere_contact_1(_S637, _S643, _S642.shape_0.y - _S635, &((*_S639)), &((*_S640)), &((*_S641)));
        if(!_S644)
        {
            return false;
        }
        (*_S640) = (vec3<f32>(0) - (*_S640));
        return true;
    }
    var cb_4 : u32 = sample_count_0(_S637);
    if(_S638 < cb_4)
    {
        var _S645 : vec3<f32> = sample_point_0(_S637, _S638);
        (*_S639) = _S645;
        var _S646 : bool = penetration_1(_S636, _S645, &((*_S641)), &((*_S640)));
        return _S646;
    }
    var _S647 : vec3<f32> = sample_point_0(_S636, _S638 - cb_4);
    (*_S639) = _S647;
    var _S648 : bool = penetration_1(_S637, _S647, &((*_S641)), &((*_S640)));
    if(!_S648)
    {
        return false;
    }
    (*_S640) = (vec3<f32>(0) - (*_S640));
    return true;
}

fn impactor_contact_count_0( _S649 : u32,  _S650 : f32,  _S651 : Box_0,  _S652 : Box_0) -> u32
{
    var _S653 : Impactor_std430_0 = impactors_0[_S649];
    var j_7 : u32 = u32(0);
    var count_4 : u32 = u32(0);
    loop
    {
        var _S654 : u32 = impactor_slots_1(&(_S653), _S652);
        if(j_7 < _S654)
        {
        }
        else
        {
            break;
        }
        var p_15 : vec3<f32>;
        var n_12 : vec3<f32>;
        var d_22 : f32;
        if(impactor_contact_3(_S649, _S650, _S651, _S652, j_7, &(p_15), &(n_12), &(d_22)))
        {
            count_4 = count_4 + u32(1);
        }
        j_7 = j_7 + u32(1);
    }
    return count_4;
}

@compute
@workgroup_size(64, 1, 1)
fn impactor_shares(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var k_23 : u32 = id_1.x;
    var _S655 : bool;
    if(k_23 >= (params_0.cand_count_0))
    {
        _S655 = true;
    }
    else
    {
        _S655 = stopped_0();
    }
    if(_S655)
    {
        return;
    }
    var _S656 : u32 = u32(3) * k_23;
    var at_8 : u32 = params_0.cand_index_0 + _S656;
    var c_34 : u32 = index_0[at_8];
    var _S657 : u32 = index_0[at_8 + u32(2)];
    var _S658 : Impactor_std430_0 = impactors_0[_S657];
    if((_S658.cand_0.z) == u32(0))
    {
        _S655 = (_S658.crush_0.x) > 0.0f;
    }
    else
    {
        _S655 = false;
    }
    var total_1 : f32;
    var ksum_0 : f32;
    if(_S655)
    {
        var rel_2 : vec3<f32> = world_diff_0(chunk_world_0(c_34), impactor_point_0(_S657));
        var _S659 : vec4<f32> = _S658.half_1;
        if(!((length(rel_2)) > (_S658.half_1.w + chunks_0[c_34].half_0.w)))
        {
            var _S660 : Box_0 = impactor_box_1(_S657, vec3<f32>(0.0f), _S659.xyz);
            var b_48 : Box_0 = chunk_box_0(c_34, rel_2);
            var kc_1 : f32 = contact_stiffness_0(_S658.mat_0.x, _S660, chunks_0[c_34].cmat_0.x, b_48, b_48.center_1 - _S660.center_1);
            var _S661 : f32 = _S658.crush_0.w;
            var _S662 : Box_0 = impactor_shrunk_0(_S657, _S661, _S660);
            var _S663 : u32 = impactor_contact_count_0(_S657, _S661, _S662, b_48);
            var _S664 : f32;
            if((_S658.shape_0.x) == 0.0f)
            {
                _S664 = kc_1;
            }
            else
            {
                _S664 = kc_1 / max(f32(_S663), 10.0f);
            }
            var j_8 : u32 = u32(0);
            total_1 = 0.0f;
            ksum_0 = 0.0f;
            loop
            {
                if(_S663 > u32(0))
                {
                    var _S665 : u32 = impactor_slots_2(&(_S658), b_48);
                    _S655 = j_8 < _S665;
                }
                else
                {
                    _S655 = false;
                }
                if(_S655)
                {
                }
                else
                {
                    break;
                }
                var p_16 : vec3<f32>;
                var nrm_2 : vec3<f32>;
                var dep_2 : f32;
                var _S666 : bool = impactor_contact_1(&(_S658), _S661, _S662, b_48, j_8, &(p_16), &(nrm_2), &(dep_2));
                if(!_S666)
                {
                    j_8 = j_8 + u32(1);
                    continue;
                }
                var ksum_1 : f32 = ksum_0 + _S664;
                total_1 = total_1 + _S664 * dep_2;
                ksum_0 = ksum_1;
                j_8 = j_8 + u32(1);
            }
        }
        else
        {
            total_1 = 0.0f;
            ksum_0 = 0.0f;
        }
    }
    else
    {
        total_1 = 0.0f;
        ksum_0 = 0.0f;
    }
    scratch_0[params_0.cand_base_0 + _S656] = vec4<f32>(total_1, ksum_0, scratch_0[params_0.cand_base_0 + _S656].z, scratch_0[params_0.cand_base_0 + _S656].w);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn impactor_crush(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var ii_1 : u32 = group_0.x;
    var tid_14 : u32 = thread_0.x;
    var _S667 : bool;
    if(ii_1 >= (params_0.impactor_count_0))
    {
        _S667 = true;
    }
    else
    {
        _S667 = stopped_0();
    }
    if(_S667)
    {
        return;
    }
    var imp_6 : Impactor_0;
    imp_6.position_1 = impactors_0[ii_1].position_1;
    imp_6.position_err_1 = impactors_0[ii_1].position_err_1;
    imp_6.velocity_1 = impactors_0[ii_1].velocity_1;
    imp_6.velocity_err_1 = impactors_0[ii_1].velocity_err_1;
    imp_6.angular_velocity_1 = impactors_0[ii_1].angular_velocity_1;
    imp_6.rotation_1 = impactors_0[ii_1].rotation_1;
    imp_6.inertia0_2 = impactors_0[ii_1].inertia0_2;
    imp_6.inertia1_2 = impactors_0[ii_1].inertia1_2;
    imp_6.inertia2_2 = impactors_0[ii_1].inertia2_2;
    imp_6.inv0_2 = impactors_0[ii_1].inv0_2;
    imp_6.inv1_2 = impactors_0[ii_1].inv1_2;
    imp_6.inv2_2 = impactors_0[ii_1].inv2_2;
    imp_6.shape_0 = impactors_0[ii_1].shape_0;
    imp_6.half_1 = impactors_0[ii_1].half_1;
    imp_6.mat_0 = impactors_0[ii_1].mat_0;
    imp_6.crush_0 = impactors_0[ii_1].crush_0;
    imp_6.geom_0 = impactors_0[ii_1].geom_0;
    imp_6.rotation_err_1 = impactors_0[ii_1].rotation_err_1;
    imp_6.ledger_0 = impactors_0[ii_1].ledger_0;
    imp_6.cand_0 = impactors_0[ii_1].cand_0;
    imp_6.momentum_1 = impactors_0[ii_1].momentum_1;
    imp_6.momentum_err_1 = impactors_0[ii_1].momentum_err_1;
    var _S668 : vec4<f32> = vec4<f32>(0.0f);
    var shares_0 : vec4<f32> = _S668;
    var unused_4 : vec4<f32> = _S668;
    var k_24 : u32 = imp_6.cand_0.x + tid_14;
    loop
    {
        if(k_24 < (imp_6.cand_0.y))
        {
        }
        else
        {
            break;
        }
        shares_0 = shares_0 + scratch_0[params_0.cand_base_0 + u32(3) * k_24];
        k_24 = k_24 + u32(256);
    }
    group_sum2_1(tid_14, &(shares_0), &(unused_4));
    if(tid_14 != u32(0))
    {
        _S667 = true;
    }
    else
    {
        _S667 = (imp_6.cand_0.z) != u32(0);
    }
    if(_S667)
    {
        return;
    }
    imp_6.geom_0 = vec4<f32>(1.0f, imp_6.crush_0.w, 0.0f, 0.0f);
    var total_2 : f32 = shares_0.x;
    if((imp_6.crush_0.x) > 0.0f)
    {
        _S667 = (imp_6.crush_0.z) < (imp_6.crush_0.y);
    }
    else
    {
        _S667 = false;
    }
    if(_S667)
    {
        _S667 = total_2 > (imp_6.crush_0.x);
    }
    else
    {
        _S667 = false;
    }
    if(_S667)
    {
        var extra_0 : f32 = (total_2 - imp_6.crush_0.x) / shares_0.y;
        imp_6.crush_0[i32(3)] = imp_6.crush_0[i32(3)] + extra_0;
        imp_6.crush_0[i32(2)] = imp_6.crush_0[i32(2)] + imp_6.crush_0.x * extra_0;
        var _S669 : f32 = imp_6.crush_0.x * extra_0;
        var _S670 : f32 = imp_6.ledger_0[i32(2)];
        var _S671 : f32 = imp_6.ledger_0[i32(3)];
        comp_add1_0(&(_S670), &(_S671), _S669);
        imp_6.ledger_0[i32(2)] = _S670;
        imp_6.ledger_0[i32(3)] = _S671;
        var _S672 : f32 = imp_6.crush_0.x * extra_0;
        var _S673 : f32 = imp_6.ledger_0[i32(0)];
        var _S674 : f32 = imp_6.ledger_0[i32(1)];
        comp_add1_0(&(_S673), &(_S674), _S672);
        imp_6.ledger_0[i32(0)] = _S673;
        imp_6.ledger_0[i32(1)] = _S674;
        imp_6.geom_0[i32(0)] = imp_6.crush_0.x / total_2;
    }
    impactors_0[ii_1].position_1 = imp_6.position_1;
    impactors_0[ii_1].position_err_1 = imp_6.position_err_1;
    impactors_0[ii_1].velocity_1 = imp_6.velocity_1;
    impactors_0[ii_1].velocity_err_1 = imp_6.velocity_err_1;
    impactors_0[ii_1].angular_velocity_1 = imp_6.angular_velocity_1;
    impactors_0[ii_1].rotation_1 = imp_6.rotation_1;
    impactors_0[ii_1].inertia0_2 = imp_6.inertia0_2;
    impactors_0[ii_1].inertia1_2 = imp_6.inertia1_2;
    impactors_0[ii_1].inertia2_2 = imp_6.inertia2_2;
    impactors_0[ii_1].inv0_2 = imp_6.inv0_2;
    impactors_0[ii_1].inv1_2 = imp_6.inv1_2;
    impactors_0[ii_1].inv2_2 = imp_6.inv2_2;
    impactors_0[ii_1].shape_0 = imp_6.shape_0;
    impactors_0[ii_1].half_1 = imp_6.half_1;
    impactors_0[ii_1].mat_0 = imp_6.mat_0;
    impactors_0[ii_1].crush_0 = imp_6.crush_0;
    impactors_0[ii_1].geom_0 = imp_6.geom_0;
    impactors_0[ii_1].rotation_err_1 = imp_6.rotation_err_1;
    impactors_0[ii_1].ledger_0 = imp_6.ledger_0;
    impactors_0[ii_1].cand_0 = imp_6.cand_0;
    impactors_0[ii_1].momentum_1 = imp_6.momentum_1;
    impactors_0[ii_1].momentum_err_1 = imp_6.momentum_err_1;
    return;
}

struct WideGroup_0
{
     island_2 : u32,
     begin_1 : u32,
     end_0 : u32,
     first_1 : u32,
};

fn wide_group_0( table_0 : u32,  g_4 : u32) -> WideGroup_0
{
    var w_11 : WideGroup_0;
    var _S675 : u32 = table_0 + u32(4) * g_4;
    w_11.island_2 = index_0[_S675];
    w_11.begin_1 = index_0[_S675 + u32(1)];
    w_11.end_0 = index_0[_S675 + u32(2)];
    w_11.first_1 = index_0[_S675 + u32(3)];
    return w_11;
}

fn contact_stopped_1( isl_18 : ptr<function, Island_std430_0>) -> bool
{
    var _S676 : vec4<u32> = islands_0[params_0.halt_index_0].info_0;
    var _S677 : bool;
    if((((islands_0[params_0.halt_index_0].info_0.z) & (u32(1)))) != u32(0))
    {
        _S677 = true;
    }
    else
    {
        var _S678 : u32 = _S676.y;
        if(_S678 != u32(0))
        {
            _S677 = _S678 <= ((*isl_18).info_0.w);
        }
        else
        {
            _S677 = false;
        }
    }
    if(_S677)
    {
        _S677 = true;
    }
    else
    {
        var _S679 : u32 = islands_0[params_0.halt_index_0].range_0.x;
        if(_S679 != u32(0))
        {
            _S677 = _S679 <= ((*isl_18).info_0.w + u32(1));
        }
        else
        {
            _S677 = false;
        }
    }
    return _S677;
}

fn contact_stopped_2( isl_19 : ptr<function, Island_std430_0>) -> bool
{
    var _S680 : vec4<u32> = islands_0[params_0.halt_index_0].info_0;
    var _S681 : bool;
    if((((islands_0[params_0.halt_index_0].info_0.z) & (u32(1)))) != u32(0))
    {
        _S681 = true;
    }
    else
    {
        var _S682 : u32 = _S680.y;
        if(_S682 != u32(0))
        {
            _S681 = _S682 <= ((*isl_19).info_0.w);
        }
        else
        {
            _S681 = false;
        }
    }
    if(_S681)
    {
        _S681 = true;
    }
    else
    {
        var _S683 : u32 = islands_0[params_0.halt_index_0].range_0.x;
        if(_S683 != u32(0))
        {
            _S681 = _S683 <= ((*isl_19).info_0.w + u32(1));
        }
        else
        {
            _S681 = false;
        }
    }
    return _S681;
}

fn wide_runs_0( isl_20 : ptr<function, Island_std430_0>) -> bool
{
    var _S684 : vec4<u32> = (*isl_20).info_0;
    var _S685 : bool;
    if(((((*isl_20).info_0.z) & (u32(1)))) != u32(0))
    {
        _S685 = true;
    }
    else
    {
        _S685 = (_S684.y) == u32(0);
    }
    if(_S685)
    {
        return false;
    }
    if((((_S684.x) & (u32(4)))) == u32(0))
    {
        _S685 = true;
    }
    else
    {
        var _S686 : bool = contact_stopped_1(&((*isl_20)));
        _S685 = !_S686;
    }
    return _S685;
}

fn wide_runs_1( isl_21 : ptr<function, Island_std430_0>) -> bool
{
    var _S687 : vec4<u32> = (*isl_21).info_0;
    var _S688 : bool;
    if(((((*isl_21).info_0.z) & (u32(1)))) != u32(0))
    {
        _S688 = true;
    }
    else
    {
        _S688 = (_S687.y) == u32(0);
    }
    if(_S688)
    {
        return false;
    }
    if((((_S687.x) & (u32(4)))) == u32(0))
    {
        _S688 = true;
    }
    else
    {
        var _S689 : bool = contact_stopped_2(&((*isl_21)));
        _S688 = !_S689;
    }
    return _S688;
}

var<workgroup> g_wide_run_0 : u32;

fn wide_enter_0( tid_15 : u32,  isl_22 : ptr<function, Island_std430_0>) -> bool
{
    seen_clear_0(tid_15);
    if(tid_15 == u32(0))
    {
        var _S690 : bool = wide_runs_1(&((*isl_22)));
        var _S691 : i32;
        if(_S690)
        {
            _S691 = i32(1);
        }
        else
        {
            _S691 = i32(0);
        }
        g_wide_run_0 = u32(_S691);
    }
    workgroupBarrier();
    return g_wide_run_0 != u32(0);
}

fn wide_step_0( isl_23 : ptr<function, Island_std430_0>) -> u32
{
    return (*isl_23).info_0.w - params_0.step_start_0;
}

fn wide_step_1( isl_24 : ptr<function, Island_std430_0>) -> u32
{
    return (*isl_24).info_0.w - params_0.step_start_0;
}

fn wide_step_2( isl_25 : ptr<function, Island_std430_0>) -> u32
{
    return (*isl_25).info_0.w - params_0.step_start_0;
}

fn wide_step_3( isl_26 : Island_0) -> u32
{
    return isl_26.info_0.w - params_0.step_start_0;
}

fn wide_enter_1( _S692 : u32,  _S693 : u32) -> bool
{
    var _S694 : Island_std430_0 = islands_0[_S693];
    seen_clear_0(_S692);
    if(_S692 == u32(0))
    {
        var _S695 : bool = wide_runs_0(&(_S694));
        var _S696 : i32;
        if(_S695)
        {
            _S696 = i32(1);
        }
        else
        {
            _S696 = i32(0);
        }
        g_wide_run_0 = u32(_S696);
    }
    workgroupBarrier();
    return g_wide_run_0 != u32(0);
}

@compute
@workgroup_size(256, 1, 1)
fn wide_wake(@builtin(workgroup_id) group_1 : vec3<u32>, @builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var tid_16 : u32 = thread_1.x;
    var _S697 : u32 = group_1.x;
    var wg_0 : WideGroup_0 = wide_group_0(params_0.wide_chunk_table_0, _S697);
    if(_S697 != (wg_0.first_1))
    {
        return;
    }
    var _S698 : Island_std430_0 = islands_0[wg_0.island_2];
    var _S699 : vec4<u32> = _S698.info_0;
    var _S700 : u32 = _S698.info_0.x;
    var _S701 : bool;
    if(((_S700 & (u32(16)))) == u32(0))
    {
        _S701 = true;
    }
    else
    {
        var _S702 : bool = wide_enter_1(tid_16, wg_0.island_2);
        _S701 = !_S702;
    }
    if(_S701)
    {
        return;
    }
    var _S703 : Quat_0 = quat_of_0(_S698.rotation_0);
    var _S704 : bool = ((_S700 & (u32(4)))) != u32(0);
    var _S705 : vec3<f32> = vec3<f32>(0.0f);
    var norm_4 : vec3<f32> = _S705;
    var unused_5 : vec3<f32> = _S705;
    var _S706 : vec4<u32> = _S698.range_0;
    var c_35 : u32 = _S698.range_0.x + tid_16;
    loop
    {
        if(c_35 < (_S706.y))
        {
        }
        else
        {
            break;
        }
        var _S707 : u32 = wide_step_2(&(_S698));
        var _S708 : f32 = settled_chunk_load_0(c_35, _S703, _S707, params_0.dt_0, _S704);
        norm_4[i32(0)] = norm_4[i32(0)] + _S708;
        c_35 = c_35 + u32(256);
    }
    group_sum3_3(tid_16, &(norm_4), &(unused_5));
    if(tid_16 != u32(0))
    {
        return;
    }
    var cov_20 : u32 = (u32(1024) | ((seen_bits_0())));
    if((params_0.solve_mode_0) == u32(1))
    {
        _S701 = (abs(norm_4.x - _S698.energy_0.z)) > (_S698.energy_0.w);
    }
    else
    {
        _S701 = false;
    }
    var cov_21 : u32;
    if(_S701)
    {
        islands_0[wg_0.island_2].info_0[i32(0)] = (_S700 & (u32(4294967279)));
        islands_0[wg_0.island_2].info_0[i32(2)] = ((_S699.z) | (u32(4)));
        cov_21 = (cov_20 | (u32(2048)));
    }
    else
    {
        cov_21 = cov_20;
    }
    cov_mark_0(wg_0.island_2, _S698.coverage_0, cov_21);
    return;
}

fn wide_partials_0( tid_17 : u32,  first_2 : u32,  count_5 : u32,  p_17 : u32,  a_24 : ptr<function, vec3<f32>>,  b_49 : ptr<function, vec3<f32>>)
{
    var _S709 : vec4<f32> = vec4<f32>(0.0f);
    var x_22 : vec4<f32> = _S709;
    var y_5 : vec4<f32> = _S709;
    var s_10 : u32 = tid_17;
    loop
    {
        if(s_10 < count_5)
        {
        }
        else
        {
            break;
        }
        var _S710 : u32 = u32(8) * (first_2 + s_10);
        x_22 = x_22 + scratch_0[params_0.wide_base_0 + _S710 + p_17];
        y_5 = y_5 + scratch_0[params_0.wide_base_0 + _S710 + p_17 + u32(1)];
        s_10 = s_10 + u32(256);
    }
    group_sum2_0(tid_17, &(x_22), &(y_5));
    (*a_24) = x_22.xyz;
    (*b_49) = y_5.xyz;
    return;
}

fn wide_partials_1( tid_18 : u32,  first_3 : u32,  count_6 : u32,  p_18 : u32,  a_25 : ptr<function, vec3<f32>>,  b_50 : ptr<function, vec3<f32>>)
{
    var _S711 : vec4<f32> = vec4<f32>(0.0f);
    var x_23 : vec4<f32> = _S711;
    var y_6 : vec4<f32> = _S711;
    var s_11 : u32 = tid_18;
    loop
    {
        if(s_11 < count_6)
        {
        }
        else
        {
            break;
        }
        var _S712 : u32 = u32(8) * (first_3 + s_11);
        x_23 = x_23 + scratch_0[params_0.wide_base_0 + _S712 + p_18];
        y_6 = y_6 + scratch_0[params_0.wide_base_0 + _S712 + p_18 + u32(1)];
        s_11 = s_11 + u32(256);
    }
    group_sum2_1(tid_18, &(x_23), &(y_6));
    (*a_25) = x_23.xyz;
    (*b_50) = y_6.xyz;
    return;
}

fn drift_angular_2( c_36 : u32,  wcom_6 : vec3<f32>,  tr_9 : vec3<f32>,  dv_9 : vec3<f32>,  lu_6 : ptr<function, vec3<f32>>,  lv_6 : ptr<function, vec3<f32>>)
{
    var r_17 : vec3<f32> = chunks_0[c_36].center_0.xyz - wcom_6;
    var _S713 : u32 = u32(4) * c_36;
    var _S714 : vec3<f32> = vec3<f32>(chunks_0[c_36].center_0.w);
    var _S715 : vec4<f32> = chunks_0[c_36].inertia0_1;
    var _S716 : vec4<f32> = chunks_0[c_36].inertia1_1;
    var _S717 : vec4<f32> = chunks_0[c_36].inertia2_1;
    var _S718 : vec3<f32> = vec3<f32>(chunks_0[c_36].scale_0.x);
    (*lu_6) = (*lu_6) + (cross(r_17, state_0[_S713].xyz - tr_9) * _S714 + rows_mul_0(chunks_0[c_36].inertia0_1, chunks_0[c_36].inertia1_1, chunks_0[c_36].inertia2_1, state_0[_S713 + u32(1)].xyz)) * _S718;
    (*lv_6) = (*lv_6) + (cross(r_17, state_0[_S713 + u32(2)].xyz - dv_9) * _S714 + rows_mul_0(_S715, _S716, _S717, state_0[_S713 + u32(3)].xyz)) * _S718;
    return;
}

fn wide_store_0( slot_2 : u32,  p_19 : u32,  a_26 : vec3<f32>,  b_51 : vec3<f32>)
{
    var _S719 : u32 = u32(8) * slot_2;
    scratch_0[params_0.wide_base_0 + _S719 + p_19] = vec4<f32>(a_26, 0.0f);
    scratch_0[params_0.wide_base_0 + _S719 + p_19 + u32(1)] = vec4<f32>(b_51, 0.0f);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn wide_drift(@builtin(workgroup_id) group_2 : vec3<u32>, @builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var tid_19 : u32 = thread_2.x;
    var _S720 : u32 = group_2.x;
    var wg_1 : WideGroup_0 = wide_group_0(params_0.wide_chunk_table_0, _S720);
    var isl_27 : Island_std430_0 = islands_0[wg_1.island_2];
    var _S721 : bool;
    if((((islands_0[wg_1.island_2].info_0.x) & (u32(17)))) != u32(0))
    {
        _S721 = true;
    }
    else
    {
        var _S722 : bool = wide_enter_1(tid_19, wg_1.island_2);
        _S721 = !_S722;
    }
    if(_S721)
    {
        return;
    }
    var tu_6 : vec3<f32>;
    var pv_6 : vec3<f32>;
    wide_partials_1(tid_19, wg_1.first_1, isl_27.wide_groups_0, u32(2), &(tu_6), &(pv_6));
    var _S723 : vec3<f32> = vec3<f32>(isl_27.wcom_0.w);
    var tr_10 : vec3<f32> = tu_6 / _S723;
    var dv_10 : vec3<f32> = pv_6 / _S723;
    var _S724 : vec3<f32> = vec3<f32>(0.0f);
    var lu_7 : vec3<f32> = _S724;
    var lv_7 : vec3<f32> = _S724;
    var c_37 : u32 = wg_1.begin_1 + tid_19;
    if(c_37 < (wg_1.end_0))
    {
        drift_angular_2(c_37, isl_27.wcom_0.xyz, tr_10, dv_10, &(lu_7), &(lv_7));
    }
    group_sum3_3(tid_19, &(lu_7), &(lv_7));
    if(tid_19 == u32(0))
    {
        wide_store_0(_S720, u32(4), lu_7, lv_7);
    }
    return;
}

fn rigid_of_1( isl_28 : ptr<function, Island_std430_0>) -> Rigid_0
{
    var rg_16 : Rigid_0;
    rg_16.rot_0 = quat_of_0((*isl_28).rotation_0);
    rg_16.rot_err_0 = (*isl_28).rotation_err_0;
    rg_16.pos_0 = (*isl_28).position_0.xyz;
    rg_16.pos_err_0 = (*isl_28).position_err_0.xyz;
    rg_16.vel_0 = (*isl_28).velocity_0.xyz;
    rg_16.vel_err_0 = (*isl_28).velocity_err_0.xyz;
    rg_16.w_1 = (*isl_28).angular_velocity_0.xyz;
    rg_16.l_0 = (*isl_28).momentum_0.xyz;
    rg_16.l_err_0 = (*isl_28).momentum_err_0.xyz;
    var _S725 : vec3<f32> = vec3<f32>(0.0f);
    rg_16.torque_0 = _S725;
    rg_16.a_0 = _S725;
    rg_16.alpha_0 = _S725;
    return rg_16;
}

fn rigid_of_2( isl_29 : ptr<function, Island_std430_0>) -> Rigid_0
{
    var rg_17 : Rigid_0;
    rg_17.rot_0 = quat_of_0((*isl_29).rotation_0);
    rg_17.rot_err_0 = (*isl_29).rotation_err_0;
    rg_17.pos_0 = (*isl_29).position_0.xyz;
    rg_17.pos_err_0 = (*isl_29).position_err_0.xyz;
    rg_17.vel_0 = (*isl_29).velocity_0.xyz;
    rg_17.vel_err_0 = (*isl_29).velocity_err_0.xyz;
    rg_17.w_1 = (*isl_29).angular_velocity_0.xyz;
    rg_17.l_0 = (*isl_29).momentum_0.xyz;
    rg_17.l_err_0 = (*isl_29).momentum_err_0.xyz;
    var _S726 : vec3<f32> = vec3<f32>(0.0f);
    rg_17.torque_0 = _S726;
    rg_17.a_0 = _S726;
    rg_17.alpha_0 = _S726;
    return rg_17;
}

fn rigid_of_3( isl_30 : ptr<function, Island_std430_0>) -> Rigid_0
{
    var rg_18 : Rigid_0;
    rg_18.rot_0 = quat_of_0((*isl_30).rotation_0);
    rg_18.rot_err_0 = (*isl_30).rotation_err_0;
    rg_18.pos_0 = (*isl_30).position_0.xyz;
    rg_18.pos_err_0 = (*isl_30).position_err_0.xyz;
    rg_18.vel_0 = (*isl_30).velocity_0.xyz;
    rg_18.vel_err_0 = (*isl_30).velocity_err_0.xyz;
    rg_18.w_1 = (*isl_30).angular_velocity_0.xyz;
    rg_18.l_0 = (*isl_30).momentum_0.xyz;
    rg_18.l_err_0 = (*isl_30).momentum_err_0.xyz;
    var _S727 : vec3<f32> = vec3<f32>(0.0f);
    rg_18.torque_0 = _S727;
    rg_18.a_0 = _S727;
    rg_18.alpha_0 = _S727;
    return rg_18;
}

fn rigid_acceleration_2( isl_31 : ptr<function, Island_std430_0>,  rg_19 : ptr<function, Rigid_0>,  f_22 : vec3<f32>,  t_23 : vec3<f32>)
{
    var iw_w_2 : vec3<f32> = world_mul_0((*rg_19).rot_0, (*isl_31).inertia0_0, (*isl_31).inertia1_0, (*isl_31).inertia2_0, (*rg_19).w_1);
    (*rg_19).a_0 = f_22 / vec3<f32>((*isl_31).com_0.w);
    (*rg_19).alpha_0 = world_mul_0((*rg_19).rot_0, (*isl_31).inv0_0, (*isl_31).inv1_0, (*isl_31).inv2_0, t_23 - cross((*rg_19).w_1, iw_w_2));
    (*rg_19).torque_0 = t_23;
    return;
}

fn rigid_acceleration_3( isl_32 : ptr<function, Island_std430_0>,  rg_20 : ptr<function, Rigid_0>,  f_23 : vec3<f32>,  t_24 : vec3<f32>)
{
    var iw_w_3 : vec3<f32> = world_mul_0((*rg_20).rot_0, (*isl_32).inertia0_0, (*isl_32).inertia1_0, (*isl_32).inertia2_0, (*rg_20).w_1);
    (*rg_20).a_0 = f_23 / vec3<f32>((*isl_32).com_0.w);
    (*rg_20).alpha_0 = world_mul_0((*rg_20).rot_0, (*isl_32).inv0_0, (*isl_32).inv1_0, (*isl_32).inv2_0, t_24 - cross((*rg_20).w_1, iw_w_3));
    (*rg_20).torque_0 = t_24;
    return;
}

fn wide_rigid_frame_0( tid_20 : u32,  isl_33 : ptr<function, Island_std430_0>,  wg_2 : WideGroup_0) -> Rigid_0
{
    var _S728 : Rigid_0 = rigid_of_1(&((*isl_33)));
    var rg_21 : Rigid_0 = _S728;
    if(((((*isl_33).info_0.x) & (u32(1)))) == u32(0))
    {
        var f_24 : vec3<f32>;
        var t_25 : vec3<f32>;
        wide_partials_0(tid_20, wg_2.first_1, (*isl_33).wide_groups_0, u32(0), &(f_24), &(t_25));
        rigid_acceleration_2(&((*isl_33)), &(rg_21), f_24, t_25);
    }
    return rg_21;
}

fn wide_rigid_frame_1( tid_21 : u32,  isl_34 : ptr<function, Island_std430_0>,  wg_3 : WideGroup_0) -> Rigid_0
{
    var _S729 : Rigid_0 = rigid_of_3(&((*isl_34)));
    var rg_22 : Rigid_0 = _S729;
    if(((((*isl_34).info_0.x) & (u32(1)))) == u32(0))
    {
        var f_25 : vec3<f32>;
        var t_26 : vec3<f32>;
        wide_partials_1(tid_21, wg_3.first_1, (*isl_34).wide_groups_0, u32(0), &(f_25), &(t_26));
        rigid_acceleration_3(&((*isl_34)), &(rg_22), f_25, t_26);
    }
    return rg_22;
}

fn comp_add_1( sum_9 : ptr<function, vec3<f32>>,  err_8 : ptr<function, vec3<f32>>,  x_24 : vec3<f32>)
{
    var t_27 : vec3<f32> = (*sum_9) + x_24;
    var _S730 : vec3<f32> = abs(x_24);
    (*err_8) = (*err_8) + (select(x_24, (*sum_9), (abs((*sum_9))) >= _S730) - t_27 + select((*sum_9), x_24, (abs((*sum_9))) >= _S730));
    (*sum_9) = t_27;
    return;
}

fn turn_left_1( hi_4 : ptr<function, Quat_0>,  lo_4 : ptr<function, vec4<f32>>,  omega_2 : vec3<f32>,  dt_17 : f32)
{
    var _S731 : f32 = length(omega_2);
    var angle_5 : f32 = _S731 * dt_17;
    if(angle_5 < 1.00000000317107685e-30f)
    {
        return;
    }
    var d_23 : vec4<f32> = turn_minus_one_0(omega_2 / vec3<f32>(_S731), angle_5);
    var dq_2 : Quat_0;
    dq_2.x_0 = d_23.x;
    dq_2.y_0 = d_23.y;
    dq_2.z_0 = d_23.z;
    dq_2.w_0 = d_23.w;
    quat_accumulate_0(&((*hi_4)), &((*lo_4)), quat_vec_0(quat_mul_0(dq_2, (*hi_4))));
    return;
}

fn integrate_rigid_2( isl_35 : ptr<function, Island_std430_0>,  rg_23 : ptr<function, Rigid_0>,  dt_18 : f32)
{
    var _S732 : vec3<f32> = vec3<f32>(dt_18);
    var _S733 : vec3<f32> = (*rg_23).torque_0 * _S732;
    var _S734 : vec3<f32> = (*rg_23).l_0;
    var _S735 : vec3<f32> = (*rg_23).l_err_0;
    comp_add_0(&(_S734), &(_S735), _S733);
    (*rg_23).l_0 = _S734;
    (*rg_23).l_err_0 = _S735;
    var l_6 : vec3<f32> = _S734 + _S735;
    var _S736 : vec3<f32> = (*rg_23).a_0 * _S732;
    var _S737 : vec3<f32> = (*rg_23).vel_0;
    var _S738 : vec3<f32> = (*rg_23).vel_err_0;
    comp_add_0(&(_S737), &(_S738), _S736);
    (*rg_23).vel_0 = _S737;
    (*rg_23).vel_err_0 = _S738;
    var _S739 : vec4<f32> = (*isl_35).inv0_0;
    var _S740 : vec4<f32> = (*isl_35).inv1_0;
    var _S741 : vec4<f32> = (*isl_35).inv2_0;
    var w_mid_2 : vec3<f32> = world_mul_0((*rg_23).rot_0, (*isl_35).inv0_0, (*isl_35).inv1_0, (*isl_35).inv2_0, l_6);
    var delta_2 : vec3<f32> = (_S737 + _S738) * _S732 - turn_difference_0(w_mid_2, dt_18, rotate_0((*rg_23).rot_0, (*isl_35).com_0.xyz));
    var _S742 : vec3<f32> = (*rg_23).pos_0;
    var _S743 : vec3<f32> = (*rg_23).pos_err_0;
    comp_add_0(&(_S742), &(_S743), delta_2);
    (*rg_23).pos_0 = _S742;
    (*rg_23).pos_err_0 = _S743;
    var _S744 : Quat_0 = (*rg_23).rot_0;
    var _S745 : vec4<f32> = (*rg_23).rot_err_0;
    turn_left_0(&(_S744), &(_S745), w_mid_2, dt_18);
    (*rg_23).rot_0 = _S744;
    (*rg_23).rot_err_0 = _S745;
    (*rg_23).w_1 = world_mul_0(_S744, _S739, _S740, _S741, l_6);
    return;
}

fn drift_rigid_2( isl_36 : ptr<function, Island_std430_0>,  rg_24 : ptr<function, Rigid_0>,  tr_11 : vec3<f32>,  phi_8 : vec3<f32>,  dv_11 : vec3<f32>,  dw_7 : vec3<f32>)
{
    var wcom_7 : vec3<f32> = (*isl_36).wcom_0.xyz;
    var rot_4 : Quat_0 = (*rg_24).rot_0;
    var _S746 : vec3<f32> = rotate_0((*rg_24).rot_0, tr_11 - turn_difference_0(phi_8, 1.0f, wcom_7));
    var _S747 : vec3<f32> = (*rg_24).pos_0;
    var _S748 : vec3<f32> = (*rg_24).pos_err_0;
    comp_add_0(&(_S747), &(_S748), _S746);
    (*rg_24).pos_0 = _S747;
    (*rg_24).pos_err_0 = _S748;
    var b_52 : vec3<f32> = inverse_rotate_0((*rg_24).rot_0, (*rg_24).w_1);
    var _S749 : vec4<f32> = (*isl_36).inertia0_0;
    var _S750 : vec4<f32> = (*isl_36).inertia1_0;
    var _S751 : vec4<f32> = (*isl_36).inertia2_0;
    var dl_turn_2 : vec3<f32> = rotate_0((*rg_24).rot_0, cross(phi_8, rows_mul_0((*isl_36).inertia0_0, (*isl_36).inertia1_0, (*isl_36).inertia2_0, b_52)) - rows_mul_0((*isl_36).inertia0_0, (*isl_36).inertia1_0, (*isl_36).inertia2_0, cross(phi_8, b_52)));
    var _S752 : Quat_0 = (*rg_24).rot_0;
    var _S753 : vec4<f32> = (*rg_24).rot_err_0;
    turn_right_0(&(_S752), &(_S753), phi_8);
    (*rg_24).rot_0 = _S752;
    (*rg_24).rot_err_0 = _S753;
    var _S754 : vec3<f32> = rotate_0(rot_4, dv_11 + cross(dw_7, (*isl_36).com_0.xyz - wcom_7));
    var _S755 : vec3<f32> = (*rg_24).vel_0;
    var _S756 : vec3<f32> = (*rg_24).vel_err_0;
    comp_add_0(&(_S755), &(_S756), _S754);
    (*rg_24).vel_0 = _S755;
    (*rg_24).vel_err_0 = _S756;
    var _S757 : vec3<f32> = rotate_0(rot_4, dw_7);
    (*rg_24).w_1 = (*rg_24).w_1 + _S757;
    var _S758 : vec3<f32> = dl_turn_2 + world_mul_0(_S752, _S749, _S750, _S751, _S757);
    var _S759 : vec3<f32> = (*rg_24).l_0;
    var _S760 : vec3<f32> = (*rg_24).l_err_0;
    comp_add_0(&(_S759), &(_S760), _S758);
    (*rg_24).l_0 = _S759;
    (*rg_24).l_err_0 = _S760;
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn wide_rigid(@builtin(workgroup_id) group_3 : vec3<u32>, @builtin(local_invocation_id) thread_3 : vec3<u32>)
{
    var tid_22 : u32 = thread_3.x;
    var _S761 : u32 = group_3.x;
    var wg_4 : WideGroup_0 = wide_group_0(params_0.wide_chunk_table_0, _S761);
    var _S762 : Island_std430_0 = islands_0[wg_4.island_2];
    var _S763 : u32 = _S762.info_0.x;
    var _S764 : bool;
    if(((_S763 & (u32(1)))) != u32(0))
    {
        _S764 = true;
    }
    else
    {
        var _S765 : bool = wide_enter_1(tid_22, wg_4.island_2);
        _S764 = !_S765;
    }
    if(_S764)
    {
        return;
    }
    if(((_S763 & (u32(16)))) != u32(0))
    {
        if(_S761 != (wg_4.first_1))
        {
            return;
        }
        var _S766 : Rigid_0 = wide_rigid_frame_1(tid_22, &(_S762), wg_4);
        var rs_0 : Rigid_0 = _S766;
        if(tid_22 != u32(0))
        {
            _S764 = true;
        }
        else
        {
            _S764 = ((_S763 & (u32(2)))) != u32(0);
        }
        if(_S764)
        {
            return;
        }
        integrate_rigid_2(&(_S762), &(rs_0), params_0.dt_0);
        islands_0[wg_4.island_2].rotation_0 = quat_vec_0(rs_0.rot_0);
        islands_0[wg_4.island_2].rotation_err_0 = rs_0.rot_err_0;
        islands_0[wg_4.island_2].position_0 = vec4<f32>(rs_0.pos_0, 0.0f);
        islands_0[wg_4.island_2].position_err_0 = vec4<f32>(rs_0.pos_err_0, 0.0f);
        islands_0[wg_4.island_2].velocity_0 = vec4<f32>(rs_0.vel_0, 0.0f);
        islands_0[wg_4.island_2].velocity_err_0 = vec4<f32>(rs_0.vel_err_0, 0.0f);
        islands_0[wg_4.island_2].angular_velocity_0 = vec4<f32>(rs_0.w_1, 0.0f);
        islands_0[wg_4.island_2].momentum_0 = vec4<f32>(rs_0.l_0, 0.0f);
        islands_0[wg_4.island_2].momentum_err_0 = vec4<f32>(rs_0.l_err_0, 0.0f);
        return;
    }
    var tu_7 : vec3<f32>;
    var pv_7 : vec3<f32>;
    wide_partials_1(tid_22, wg_4.first_1, _S762.wide_groups_0, u32(2), &(tu_7), &(pv_7));
    var lu_8 : vec3<f32>;
    var lv_8 : vec3<f32>;
    wide_partials_1(tid_22, wg_4.first_1, _S762.wide_groups_0, u32(4), &(lu_8), &(lv_8));
    var _S767 : vec4<f32> = _S762.wcom_0;
    var _S768 : vec3<f32> = vec3<f32>(_S762.wcom_0.w);
    var tr_12 : vec3<f32> = tu_7 / _S768;
    var dv_12 : vec3<f32> = pv_7 / _S768;
    var phi_9 : vec3<f32> = rows_mul_0(_S762.winv0_0, _S762.winv1_0, _S762.winv2_0, lu_8);
    var dw_8 : vec3<f32> = rows_mul_0(_S762.winv0_0, _S762.winv1_0, _S762.winv2_0, lv_8);
    var c_38 : u32 = wg_4.begin_1 + tid_22;
    if(c_38 < (wg_4.end_0))
    {
        drift_apply_0(c_38, _S767.xyz, tr_12, phi_9, dv_12, dw_8);
    }
    if(_S761 != (wg_4.first_1))
    {
        return;
    }
    var _S769 : Rigid_0 = wide_rigid_frame_1(tid_22, &(_S762), wg_4);
    var rg_25 : Rigid_0 = _S769;
    if(tid_22 != u32(0))
    {
        return;
    }
    if(!(((_S763 & (u32(2)))) != u32(0)))
    {
        integrate_rigid_2(&(_S762), &(rg_25), params_0.dt_0);
        drift_rigid_2(&(_S762), &(rg_25), tr_12, phi_9, dv_12, dw_8);
    }
    islands_0[wg_4.island_2].rotation_0 = quat_vec_0(rg_25.rot_0);
    islands_0[wg_4.island_2].rotation_err_0 = rg_25.rot_err_0;
    islands_0[wg_4.island_2].position_0 = vec4<f32>(rg_25.pos_0, 0.0f);
    islands_0[wg_4.island_2].position_err_0 = vec4<f32>(rg_25.pos_err_0, 0.0f);
    islands_0[wg_4.island_2].velocity_0 = vec4<f32>(rg_25.vel_0, 0.0f);
    islands_0[wg_4.island_2].velocity_err_0 = vec4<f32>(rg_25.vel_err_0, 0.0f);
    islands_0[wg_4.island_2].angular_velocity_0 = vec4<f32>(rg_25.w_1, 0.0f);
    islands_0[wg_4.island_2].momentum_0 = vec4<f32>(rg_25.l_0, 0.0f);
    islands_0[wg_4.island_2].momentum_err_0 = vec4<f32>(rg_25.l_err_0, 0.0f);
    return;
}

var<workgroup> g_label_0 : array<u32, i32(4096)>;

var<workgroup> g_changed_0 : u32;

fn connected_3( st_7 : ptr<function, JointState_std430_0>,  has_rebar_4 : bool) -> bool
{
    var _S770 : bool;
    if(((*st_7).damage_0) < 1.0f)
    {
        _S770 = true;
    }
    else
    {
        if(has_rebar_4)
        {
            _S770 = ((*st_7).rebar_broken_0) == 0.0f;
        }
        else
        {
            _S770 = false;
        }
    }
    return _S770;
}

fn connected_4( st_8 : ptr<function, JointState_std430_0>,  has_rebar_5 : bool) -> bool
{
    var _S771 : bool;
    if(((*st_8).damage_0) < 1.0f)
    {
        _S771 = true;
    }
    else
    {
        if(has_rebar_5)
        {
            _S771 = ((*st_8).rebar_broken_0) == 0.0f;
        }
        else
        {
            _S771 = false;
        }
    }
    return _S771;
}

fn island_connected_0( tid_23 : u32,  isl_37 : Island_0) -> bool
{
    var _S772 : u32 = isl_37.range_0.x;
    var n_13 : u32 = isl_37.range_0.y - _S772;
    var converged_2 : bool;
    if(n_13 > u32(4096))
    {
        converged_2 = true;
    }
    else
    {
        converged_2 = (params_0.device_connect_0) == u32(0);
    }
    if(converged_2)
    {
        return false;
    }
    var c_39 : u32 = tid_23;
    loop
    {
        if(c_39 < n_13)
        {
        }
        else
        {
            break;
        }
        g_label_0[c_39] = c_39;
        c_39 = c_39 + u32(256);
    }
    workgroupBarrier();
    converged_2 = false;
    var pass_1 : u32 = u32(0);
    loop
    {
        if(pass_1 <= n_13)
        {
        }
        else
        {
            break;
        }
        if(tid_23 == u32(0))
        {
            g_changed_0 = u32(0);
        }
        workgroupBarrier();
        c_39 = tid_23;
        loop
        {
            if(c_39 < n_13)
            {
            }
            else
            {
                break;
            }
            var gc_1 : u32 = _S772 + c_39;
            var _S773 : u32 = index_0[gc_1];
            var l_7 : u32 = g_label_0[c_39];
            var e_6 : u32 = _S773;
            loop
            {
                if(e_6 < index_0[gc_1 + u32(1)])
                {
                }
                else
                {
                    break;
                }
                var entry_5 : u32 = index_0[e_6];
                var bond_1 : u32 = (entry_5 >> (u32(1)));
                var _S774 : JointState_std430_0 = bond_dyn_0[bond_1].js_0;
                var _S775 : bool = connected_3(&(_S774), (bonds_0[bond_1].law_0.stiff1_0.w) != 0.0f);
                if(!_S775)
                {
                    e_6 = e_6 + u32(1);
                    continue;
                }
                var other_1 : u32;
                if(((entry_5 & (u32(1)))) == u32(0))
                {
                    other_1 = bonds_0[bond_1].law_0.ids_0.z;
                }
                else
                {
                    other_1 = bonds_0[bond_1].law_0.ids_0.y;
                }
                l_7 = min(l_7, g_label_0[other_1 - _S772]);
                e_6 = e_6 + u32(1);
            }
            var _S776 : u32 = min(l_7, g_label_0[l_7]);
            if(_S776 < (g_label_0[c_39]))
            {
                g_label_0[c_39] = _S776;
                g_changed_0 = u32(1);
            }
            c_39 = c_39 + u32(256);
        }
        workgroupBarrier();
        var converged_3 : bool = g_changed_0 == u32(0);
        workgroupBarrier();
        if(converged_3)
        {
            converged_2 = converged_3;
            break;
        }
        var _S777 : u32 = pass_1 + u32(1);
        converged_2 = converged_3;
        pass_1 = _S777;
    }
    if(!converged_2)
    {
        return false;
    }
    if(tid_23 == u32(0))
    {
        g_changed_0 = u32(0);
    }
    workgroupBarrier();
    c_39 = tid_23;
    loop
    {
        if(c_39 < n_13)
        {
        }
        else
        {
            break;
        }
        if((g_label_0[c_39]) != u32(0))
        {
            g_changed_0 = u32(1);
        }
        c_39 = c_39 + u32(256);
    }
    workgroupBarrier();
    return g_changed_0 == u32(0);
}

@compute
@workgroup_size(256, 1, 1)
fn wide_end(@builtin(workgroup_id) group_4 : vec3<u32>, @builtin(local_invocation_id) thread_4 : vec3<u32>)
{
    var tid_24 : u32 = thread_4.x;
    var _S778 : u32 = group_4.x;
    var wg_5 : WideGroup_0 = wide_group_0(params_0.wide_chunk_table_0, _S778);
    if(_S778 != (wg_5.first_1))
    {
        return;
    }
    var _S779 : Island_std430_0 = islands_0[wg_5.island_2];
    var isl_38 : Island_0;
    isl_38.range_0 = _S779.range_0;
    isl_38.info_0 = _S779.info_0;
    isl_38.com_0 = _S779.com_0;
    isl_38.inertia0_0 = _S779.inertia0_0;
    isl_38.inertia1_0 = _S779.inertia1_0;
    isl_38.inertia2_0 = _S779.inertia2_0;
    isl_38.inv0_0 = _S779.inv0_0;
    isl_38.inv1_0 = _S779.inv1_0;
    isl_38.inv2_0 = _S779.inv2_0;
    isl_38.wcom_0 = _S779.wcom_0;
    isl_38.winv0_0 = _S779.winv0_0;
    isl_38.winv1_0 = _S779.winv1_0;
    isl_38.winv2_0 = _S779.winv2_0;
    isl_38.rotation_0 = _S779.rotation_0;
    isl_38.position_0 = _S779.position_0;
    isl_38.position_err_0 = _S779.position_err_0;
    isl_38.velocity_0 = _S779.velocity_0;
    isl_38.velocity_err_0 = _S779.velocity_err_0;
    isl_38.angular_velocity_0 = _S779.angular_velocity_0;
    isl_38.done_0 = _S779.done_0;
    isl_38.coverage_0 = _S779.coverage_0;
    isl_38.wide_groups_0 = _S779.wide_groups_0;
    isl_38.done_pad_0 = _S779.done_pad_0;
    isl_38.probes_0 = _S779.probes_0;
    isl_38.energy_0 = _S779.energy_0;
    isl_38.rotation_err_0 = _S779.rotation_err_0;
    isl_38.momentum_0 = _S779.momentum_0;
    isl_38.momentum_err_0 = _S779.momentum_err_0;
    var _S780 : bool = wide_enter_0(tid_24, &(_S779));
    if(!_S780)
    {
        return;
    }
    var work_7 : vec3<f32>;
    var flags_2 : vec3<f32>;
    wide_partials_1(tid_24, wg_5.first_1, isl_38.wide_groups_0, u32(6), &(work_7), &(flags_2));
    var halt_0 : bool = (((isl_38.info_0.z) & (u32(2)))) != u32(0);
    var whole_0 : bool;
    if(halt_0)
    {
        var _S781 : bool = island_connected_0(tid_24, isl_38);
        whole_0 = _S781;
    }
    else
    {
        whole_0 = false;
    }
    if(tid_24 != u32(0))
    {
        return;
    }
    var halt_1 : bool;
    if(whole_0)
    {
        isl_38.info_0[i32(2)] = ((isl_38.info_0[i32(2)]) & (u32(4294967293)));
        isl_38.coverage_0 = ((isl_38.coverage_0) | (u32(2097280)));
        halt_1 = false;
    }
    else
    {
        halt_1 = halt_0;
    }
    var k_25 : u32 = wide_step_3(isl_38);
    if((isl_38.probes_0.y) > (isl_38.probes_0.x))
    {
        record_probes_0(isl_38, rigid_of_0(isl_38), k_25);
    }
    if(halt_1)
    {
        whole_0 = (((isl_38.info_0.x) & (u32(4)))) != u32(0);
    }
    else
    {
        whole_0 = false;
    }
    if(whole_0)
    {
        contact_split_at_0(isl_38.info_0.w + u32(1));
    }
    var _S782 : f32 = work_7.x;
    var _S783 : f32 = isl_38.energy_0[i32(0)];
    var _S784 : f32 = isl_38.energy_0[i32(1)];
    comp_add1_0(&(_S783), &(_S784), _S782);
    isl_38.energy_0[i32(0)] = _S783;
    isl_38.energy_0[i32(1)] = _S784 + work_7.y;
    isl_38.done_0 = isl_38.done_0 + u32(1);
    isl_38.info_0[i32(1)] = isl_38.info_0[i32(1)] - u32(1);
    isl_38.info_0[i32(3)] = isl_38.info_0[i32(3)] + u32(1);
    if(halt_1)
    {
        isl_38.info_0[i32(2)] = ((((isl_38.info_0.z) & (u32(4294967293)))) | (u32(1)));
    }
    var _S785 : u32;
    if(halt_1)
    {
        _S785 = u32(128);
    }
    else
    {
        _S785 = u32(0);
    }
    var _S786 : u32 = (u32(256) | (_S785));
    if((isl_38.probes_0.y) > (isl_38.probes_0.x))
    {
        _S785 = u32(262144);
    }
    else
    {
        _S785 = u32(0);
    }
    var _S787 : u32 = (_S786 | (_S785));
    if((((isl_38.info_0.x) & (u32(4)))) != u32(0))
    {
        _S785 = u32(524288);
    }
    else
    {
        _S785 = u32(0);
    }
    var _S788 : u32 = (_S787 | (_S785));
    if((flags_2.x) > 0.0f)
    {
        _S785 = u32(32768);
    }
    else
    {
        _S785 = u32(0);
    }
    var _S789 : u32 = (_S788 | (_S785));
    if((flags_2.y) > 0.0f)
    {
        _S785 = u32(65536);
    }
    else
    {
        _S785 = u32(0);
    }
    var _S790 : u32 = (_S789 | (_S785));
    if((flags_2.z) > 0.0f)
    {
        _S785 = u32(131072);
    }
    else
    {
        _S785 = u32(0);
    }
    isl_38.coverage_0 = ((isl_38.coverage_0) | (((_S790 | (_S785)))));
    islands_0[wg_5.island_2].range_0 = isl_38.range_0;
    islands_0[wg_5.island_2].info_0 = isl_38.info_0;
    islands_0[wg_5.island_2].com_0 = isl_38.com_0;
    islands_0[wg_5.island_2].inertia0_0 = isl_38.inertia0_0;
    islands_0[wg_5.island_2].inertia1_0 = isl_38.inertia1_0;
    islands_0[wg_5.island_2].inertia2_0 = isl_38.inertia2_0;
    islands_0[wg_5.island_2].inv0_0 = isl_38.inv0_0;
    islands_0[wg_5.island_2].inv1_0 = isl_38.inv1_0;
    islands_0[wg_5.island_2].inv2_0 = isl_38.inv2_0;
    islands_0[wg_5.island_2].wcom_0 = isl_38.wcom_0;
    islands_0[wg_5.island_2].winv0_0 = isl_38.winv0_0;
    islands_0[wg_5.island_2].winv1_0 = isl_38.winv1_0;
    islands_0[wg_5.island_2].winv2_0 = isl_38.winv2_0;
    islands_0[wg_5.island_2].rotation_0 = isl_38.rotation_0;
    islands_0[wg_5.island_2].position_0 = isl_38.position_0;
    islands_0[wg_5.island_2].position_err_0 = isl_38.position_err_0;
    islands_0[wg_5.island_2].velocity_0 = isl_38.velocity_0;
    islands_0[wg_5.island_2].velocity_err_0 = isl_38.velocity_err_0;
    islands_0[wg_5.island_2].angular_velocity_0 = isl_38.angular_velocity_0;
    islands_0[wg_5.island_2].done_0 = isl_38.done_0;
    islands_0[wg_5.island_2].coverage_0 = isl_38.coverage_0;
    islands_0[wg_5.island_2].wide_groups_0 = isl_38.wide_groups_0;
    islands_0[wg_5.island_2].done_pad_0 = isl_38.done_pad_0;
    islands_0[wg_5.island_2].probes_0 = isl_38.probes_0;
    islands_0[wg_5.island_2].energy_0 = isl_38.energy_0;
    islands_0[wg_5.island_2].rotation_err_0 = isl_38.rotation_err_0;
    islands_0[wg_5.island_2].momentum_0 = isl_38.momentum_0;
    islands_0[wg_5.island_2].momentum_err_0 = isl_38.momentum_err_0;
    return;
}

fn sv_0( c_40 : u32,  slot_3 : u32) -> u32
{
    return params_0.statics_base_0 + u32(23) * c_40 + slot_3;
}

fn project_load_slot_0( tid_25 : u32,  isl_39 : ptr<function, Island_std430_0>,  slot_4 : u32)
{
    var _S791 : u32;
    var _S792 : vec3<f32> = vec3<f32>(0.0f);
    var net_f_0 : vec3<f32> = _S792;
    var net_m_0 : vec3<f32> = _S792;
    var _S793 : vec4<u32> = (*isl_39).range_0;
    var _S794 : u32 = (*isl_39).range_0.x + tid_25;
    var c_41 : u32 = _S794;
    loop
    {
        var _S795 : u32 = _S793.y;
        _S791 = _S795;
        if(c_41 < _S795)
        {
        }
        else
        {
            break;
        }
        var fi_2 : vec3<f32> = scratch_0[sv_0(c_41, slot_4)].xyz;
        net_f_0 = net_f_0 + fi_2;
        net_m_0 = net_m_0 + (cross(chunks_0[c_41].center_0.xyz - (*isl_39).com_0.xyz, fi_2) + scratch_0[sv_0(c_41, slot_4 + u32(1))].xyz);
        c_41 = c_41 + u32(256);
    }
    group_sum3_3(tid_25, &(net_f_0), &(net_m_0));
    var _S796 : vec4<f32> = (*isl_39).com_0;
    var _S797 : vec3<f32> = net_f_0 / vec3<f32>((*isl_39).com_0.w);
    var _S798 : vec3<f32> = rows_mul_0((*isl_39).inv0_0, (*isl_39).inv1_0, (*isl_39).inv2_0, net_m_0);
    c_41 = _S794;
    loop
    {
        if(c_41 < _S791)
        {
        }
        else
        {
            break;
        }
        var _S799 : u32 = sv_0(c_41, slot_4);
        scratch_0[_S799] = vec4<f32>(scratch_0[_S799].xyz - (_S797 + cross(_S798, chunks_0[c_41].center_0.xyz - _S796.xyz)) * vec3<f32>(chunks_0[c_41].center_0.w), 0.0f);
        var _S800 : u32 = sv_0(c_41, slot_4 + u32(1));
        scratch_0[_S800] = vec4<f32>(scratch_0[_S800].xyz - rows_mul_0(chunks_0[c_41].inertia0_1, chunks_0[c_41].inertia1_1, chunks_0[c_41].inertia2_1, _S798), 0.0f);
        c_41 = c_41 + u32(256);
    }
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    return;
}

fn island_dot_0( tid_26 : u32,  c0_0 : u32,  c1_0 : u32,  sa_0 : u32,  sb_0 : u32) -> f32
{
    var _S801 : vec4<f32> = vec4<f32>(0.0f);
    var acc_0 : vec4<f32> = _S801;
    var unused_6 : vec4<f32> = _S801;
    var c_42 : u32 = c0_0 + tid_26;
    loop
    {
        if(c_42 < c1_0)
        {
        }
        else
        {
            break;
        }
        acc_0[i32(0)] = acc_0[i32(0)] + (dot(scratch_0[sv_0(c_42, sa_0)].xyz, scratch_0[sv_0(c_42, sb_0)].xyz) + dot(scratch_0[sv_0(c_42, sa_0 + u32(1))].xyz, scratch_0[sv_0(c_42, sb_0 + u32(1))].xyz));
        c_42 = c_42 + u32(256);
    }
    group_sum2_2(tid_26, &(acc_0), &(unused_6));
    return acc_0.x;
}

fn stress_measures_1( b_53 : ptr<function, JointBond_std430_0>,  q_lin_3 : vec3<f32>,  q_ang_3 : vec3<f32>) -> Measures_0
{
    var area_3 : f32 = (*b_53).geom0_0.x;
    var _S802 : f32 = q_lin_3.z;
    var axial_1 : f32 = fdiv_0(_S802, area_3);
    var bending_1 : f32 = fdiv_0(abs(q_ang_3.x), (*b_53).geom1_0.x) + fdiv_0(abs(q_ang_3.y), (*b_53).geom1_0.y);
    var _S803 : f32 = q_lin_3.x;
    var _S804 : f32 = q_lin_3.y;
    var shear_2 : f32 = fdiv_0(fsqrt_0(_S803 * _S803 + _S804 * _S804), area_3) + fdiv_0(abs(q_ang_3.z), (*b_53).geom0_0.w);
    var m_6 : Measures_0;
    m_6.tension_0 = axial_1 + bending_1;
    m_6.shear_0 = shear_2;
    var _S805 : f32 = - axial_1;
    m_6.normal_compression_0 = max(_S805, 0.0f);
    m_6.compression_0 = _S805 + bending_1;
    m_6.compressive_force_0 = max(- _S802, 0.0f);
    return m_6;
}

fn dif_factor_1( mat_10 : ptr<function, JointMaterial_std140_0>,  strain_rate_2 : f32) -> f32
{
    var r_18 : f32 = abs(strain_rate_2);
    var _S806 : vec4<f32> = (*mat_10).dif_0;
    var ref_1 : f32 = (*mat_10).dif_0.x;
    if(r_18 <= ref_1)
    {
        return 1.0f;
    }
    var _S807 : f32 = _S806.z;
    var f_26 : f32;
    if(r_18 <= _S807)
    {
        f_26 = fpow_0(fdiv_0(r_18, ref_1), _S806.y);
    }
    else
    {
        f_26 = fpow_0(fdiv_0(_S807, ref_1), _S806.y) * fpow_0(fdiv_0(r_18, _S807), _S806.w);
    }
    return clamp(f_26, 1.0f, (*mat_10).misc_0.x);
}

fn fatigue_factor_2( mat_11 : ptr<function, JointMaterial_std140_0>,  life_3 : f32) -> f32
{
    if(((((*mat_11).kind_flags_0.y) & (u32(64)))) == u32(0))
    {
        return 1.0f;
    }
    return fpow_0(clamp(life_3, 0.0f, 1.0f), fdiv_0(1.0f, (*mat_11).misc_0.y - 2.0f));
}

fn failure_indices_1( mat_12 : ptr<function, JointMaterial_std140_0>,  b_54 : ptr<function, JointBond_std430_0>,  m_7 : Measures_0,  multiplier_2 : f32) -> vec4<f32>
{
    var fc_5 : f32 = (*mat_12).strength_0.y * multiplier_2;
    var _S808 : f32 = min((*mat_12).strength_0.z * multiplier_2 + (*mat_12).strength_0.w * m_7.normal_compression_0, (*mat_12).energy_1.x * multiplier_2);
    var idx_1 : vec4<f32>;
    idx_1[i32(0)] = max(fdiv_0(m_7.tension_0, (*mat_12).strength_0.x * multiplier_2), 0.0f);
    var _S809 : f32;
    if(_S808 > 0.0f)
    {
        _S809 = fdiv_0(m_7.shear_0, _S808);
    }
    else
    {
        _S809 = infinity_0();
    }
    idx_1[i32(1)] = _S809;
    idx_1[i32(2)] = max(fdiv_0(m_7.compression_0, fc_5), 0.0f);
    var _S810 : f32 = (*b_54).stiff1_0.y;
    if(_S810 > 0.0f)
    {
        _S809 = fdiv_0(m_7.compressive_force_0, _S810);
    }
    else
    {
        _S809 = 0.0f;
    }
    idx_1[i32(3)] = _S809;
    return idx_1;
}

fn contact_normal_1( mat_13 : ptr<function, JointMaterial_std140_0>,  b_55 : ptr<function, JointBond_std430_0>,  crush_5 : f32,  d_lin_5 : vec3<f32>,  d_ang_4 : vec3<f32>) -> vec4<f32>
{
    var kn_4 : f32 = (*b_55).stiff0_0.x;
    var w0_4 : f32 = (*b_55).geom0_0.y;
    var w1_4 : f32 = (*b_55).geom0_0.z;
    var nc_sum_5 : f32;
    var m1_4 : f32;
    var m2_4 : f32;
    var energy_7 : f32;
    if(((((*mat_13).kind_flags_0.y) & (u32(4)))) != u32(0))
    {
        var p_20 : vec4<f32> = no_tension_patch_0(kn_4 * (1.0f - crush_5), w0_4, w1_4, d_lin_5.z, d_ang_4.x, d_ang_4.y);
        var _S811 : f32 = p_20.y;
        var _S812 : f32 = p_20.z;
        var _S813 : f32 = p_20.w;
        nc_sum_5 = p_20.x;
        m1_4 = _S811;
        m2_4 = _S812;
        energy_7 = _S813;
    }
    else
    {
        var ki_2 : f32 = kn_4 * (1.0f - crush_5) / 36.0f;
        var _S814 : f32 = d_ang_4.x;
        var _S815 : f32 = d_ang_4.y;
        var spread_2 : f32 = abs(_S814) * 0.4166666567325592f * w1_4 + abs(_S815) * 0.4166666567325592f * w0_4;
        var _S816 : f32 = d_lin_5.z;
        var slack_2 : f32 = 9.99999997475242708e-07f * (abs(_S816) + spread_2);
        if((_S816 - spread_2) > slack_2)
        {
            nc_sum_5 = 0.0f;
            m1_4 = 0.0f;
            m2_4 = 0.0f;
            energy_7 = 0.0f;
        }
        else
        {
            if((_S816 + spread_2) < (- slack_2))
            {
                var i1_1 : f32 = 2.91666650772094727f * w0_4 * w0_4;
                var i2_1 : f32 = 2.91666650772094727f * w1_4 * w1_4;
                var _S817 : f32 = ki_2 * _S814 * i2_1;
                var _S818 : f32 = ki_2 * _S815 * i1_1;
                var _S819 : f32 = 0.5f * ki_2 * (36.0f * _S816 * _S816 + _S814 * _S814 * i2_1 + _S815 * _S815 * i1_1);
                nc_sum_5 = ki_2 * 36.0f * _S816;
                m1_4 = _S817;
                m2_4 = _S818;
                energy_7 = _S819;
            }
            else
            {
                var i_11 : u32 = u32(0);
                var nc_sum_6 : f32 = 0.0f;
                var m1_5 : f32 = 0.0f;
                var m2_5 : f32 = 0.0f;
                var energy_8 : f32 = 0.0f;
                loop
                {
                    if(i_11 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var _S820 : f32 = SPRING_AT_0[i_11] * w0_4;
                    var j_9 : u32 = u32(0);
                    nc_sum_5 = nc_sum_6;
                    m1_4 = m1_5;
                    m2_4 = m2_5;
                    energy_7 = energy_8;
                    loop
                    {
                        if(j_9 < u32(6))
                        {
                        }
                        else
                        {
                            break;
                        }
                        var s2_1 : f32 = SPRING_AT_0[j_9] * w1_4;
                        var di_2 : f32 = _S816 + _S814 * s2_1 - _S815 * _S820;
                        if(di_2 < 0.0f)
                        {
                            var f_27 : f32 = ki_2 * di_2;
                            var m1_6 : f32 = m1_4 + f_27 * s2_1;
                            var m2_6 : f32 = m2_4 - f_27 * _S820;
                            var energy_9 : f32 = energy_7 + 0.5f * ki_2 * di_2 * di_2;
                            nc_sum_5 = nc_sum_5 + f_27;
                            m1_4 = m1_6;
                            m2_4 = m2_6;
                            energy_7 = energy_9;
                        }
                        j_9 = j_9 + u32(1);
                    }
                    i_11 = i_11 + u32(1);
                    nc_sum_6 = nc_sum_5;
                    m1_5 = m1_4;
                    m2_5 = m2_4;
                    energy_8 = energy_7;
                }
                nc_sum_5 = nc_sum_6;
                m1_4 = m1_5;
                m2_4 = m2_5;
                energy_7 = energy_8;
            }
        }
    }
    return vec4<f32>(nc_sum_5, m1_4, m2_4, energy_7);
}

fn contact_part_1( mat_14 : ptr<function, JointMaterial_std140_0>,  b_56 : ptr<function, JointBond_std430_0>,  crush_6 : f32,  plastic_5 : vec3<f32>,  d_lin_6 : vec3<f32>,  d_ang_5 : vec3<f32>) -> Contact_0
{
    var c_43 : Contact_0;
    var _S821 : vec3<f32> = vec3<f32>(0.0f);
    c_43.q_lin_1 = _S821;
    c_43.q_ang_1 = _S821;
    c_43.energy_5 = 0.0f;
    c_43.diss_1 = 0.0f;
    c_43.plastic_0 = plastic_5;
    if(((((*mat_14).kind_flags_0.y) & (u32(2)))) == u32(0))
    {
        return c_43;
    }
    var ks_3 : f32 = (*b_56).stiff0_0.y;
    var kt_3 : f32 = (*b_56).stiff1_0.x;
    var _S822 : vec4<f32> = contact_normal_1(&((*mat_14)), &((*b_56)), crush_6, d_lin_6, d_ang_5);
    var nc_sum_7 : f32 = _S822.x;
    var m1_7 : f32 = _S822.y;
    var m2_7 : f32 = _S822.z;
    var energy_10 : f32 = _S822.w;
    var nc_2 : f32 = - nc_sum_7;
    var p_21 : vec3<f32> = plastic_5;
    c_43.q_lin_1 = vec3<f32>(0.0f, 0.0f, nc_sum_7);
    c_43.q_ang_1 = vec3<f32>(m1_7, m2_7, 0.0f);
    var slide_cap_2 : f32 = (*mat_14).strength_0.w * nc_2;
    var _S823 : f32 = ks_3 * (d_lin_6.x - plastic_5.x);
    var _S824 : f32 = ks_3 * (d_lin_6.y - plastic_5.y);
    var tn_2 : f32 = fsqrt_0(_S823 * _S823 + _S824 * _S824);
    var _S825 : bool;
    if(tn_2 > slide_cap_2)
    {
        _S825 = tn_2 > 0.0f;
    }
    else
    {
        _S825 = false;
    }
    var diss_6 : f32;
    if(_S825)
    {
        var _S826 : f32 = fdiv_0(_S823, tn_2);
        var _S827 : f32 = fdiv_0(_S824, tn_2);
        var dslip_2 : f32 = fdiv_0(tn_2 - slide_cap_2, ks_3);
        p_21[i32(0)] = p_21[i32(0)] + _S826 * dslip_2;
        p_21[i32(1)] = p_21[i32(1)] + _S827 * dslip_2;
        c_43.q_lin_1[i32(0)] = _S826 * slide_cap_2;
        c_43.q_lin_1[i32(1)] = _S827 * slide_cap_2;
        diss_6 = slide_cap_2 * dslip_2;
    }
    else
    {
        c_43.q_lin_1[i32(0)] = _S823;
        c_43.q_lin_1[i32(1)] = _S824;
        diss_6 = 0.0f;
    }
    var tq_1 : vec2<f32> = return_map_0(kt_3, d_ang_5.z, p_21.z, slide_cap_2 * (*b_56).geom1_0.z);
    var _S828 : f32 = tq_1.x;
    var _S829 : f32 = tq_1.y;
    var diss_7 : f32 = diss_6 + abs(_S828) * abs(_S829);
    p_21[i32(2)] = p_21[i32(2)] + _S829;
    c_43.q_ang_1[i32(2)] = _S828;
    c_43.energy_5 = energy_10 + 0.5f * (fdiv_0(sq_0(c_43.q_lin_1.x), ks_3) + fdiv_0(sq_0(c_43.q_lin_1.y), ks_3) + fdiv_0(sq_0(_S828), kt_3));
    c_43.diss_1 = diss_7;
    c_43.plastic_0 = p_21;
    return c_43;
}

fn contact_offsets_1( mat_15 : ptr<function, JointMaterial_std140_0>,  b_57 : ptr<function, JointBond_std430_0>,  crush_7 : f32,  plastic_6 : vec3<f32>,  d_lin_7 : vec3<f32>,  d_ang_6 : vec3<f32>) -> vec3<f32>
{
    var _S830 : u32 = (*mat_15).kind_flags_0.y;
    if(((_S830 & (u32(2)))) == u32(0))
    {
        return plastic_6;
    }
    var kn_5 : f32 = (*b_57).stiff0_0.x;
    var ks_4 : f32 = (*b_57).stiff0_0.y;
    var kt_4 : f32 = (*b_57).stiff1_0.x;
    var w0_5 : f32 = (*b_57).geom0_0.y;
    var w1_5 : f32 = (*b_57).geom0_0.z;
    var nc_sum_8 : f32;
    if(((_S830 & (u32(4)))) != u32(0))
    {
        var _S831 : vec4<f32> = no_tension_patch_0(kn_5 * (1.0f - crush_7), w0_5, w1_5, d_lin_7.z, d_ang_6.x, d_ang_6.y);
        nc_sum_8 = _S831.x;
    }
    else
    {
        var ki_3 : f32 = kn_5 * (1.0f - crush_7) / 36.0f;
        var _S832 : f32 = d_ang_6.x;
        var _S833 : f32 = d_ang_6.y;
        var spread_3 : f32 = abs(_S832) * 0.4166666567325592f * w1_5 + abs(_S833) * 0.4166666567325592f * w0_5;
        var _S834 : f32 = d_lin_7.z;
        var slack_3 : f32 = 9.99999997475242708e-07f * (abs(_S834) + spread_3);
        if((_S834 - spread_3) > slack_3)
        {
            nc_sum_8 = 0.0f;
        }
        else
        {
            if((_S834 + spread_3) < (- slack_3))
            {
                nc_sum_8 = ki_3 * 36.0f * _S834;
            }
            else
            {
                var i_12 : u32 = u32(0);
                var nc_sum_9 : f32 = 0.0f;
                loop
                {
                    if(i_12 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var _S835 : f32 = SPRING_AT_0[i_12] * w0_5;
                    var j_10 : u32 = u32(0);
                    nc_sum_8 = nc_sum_9;
                    loop
                    {
                        if(j_10 < u32(6))
                        {
                        }
                        else
                        {
                            break;
                        }
                        var di_3 : f32 = _S834 + _S832 * (SPRING_AT_0[j_10] * w1_5) - _S833 * _S835;
                        if(di_3 < 0.0f)
                        {
                            nc_sum_8 = nc_sum_8 + ki_3 * di_3;
                        }
                        j_10 = j_10 + u32(1);
                    }
                    i_12 = i_12 + u32(1);
                    nc_sum_9 = nc_sum_8;
                }
                nc_sum_8 = nc_sum_9;
            }
        }
    }
    var nc_3 : f32 = - nc_sum_8;
    var p_22 : vec3<f32> = plastic_6;
    var slide_cap_3 : f32 = (*mat_15).strength_0.w * nc_3;
    var _S836 : f32 = ks_4 * (d_lin_7.x - plastic_6.x);
    var _S837 : f32 = ks_4 * (d_lin_7.y - plastic_6.y);
    var tn_3 : f32 = fsqrt_0(_S836 * _S836 + _S837 * _S837);
    var _S838 : bool;
    if(tn_3 > slide_cap_3)
    {
        _S838 = tn_3 > 0.0f;
    }
    else
    {
        _S838 = false;
    }
    if(_S838)
    {
        var _S839 : f32 = fdiv_0(_S837, tn_3);
        var dslip_3 : f32 = fdiv_0(tn_3 - slide_cap_3, ks_4);
        p_22[i32(0)] = p_22[i32(0)] + fdiv_0(_S836, tn_3) * dslip_3;
        p_22[i32(1)] = p_22[i32(1)] + _S839 * dslip_3;
    }
    p_22[i32(2)] = p_22[i32(2)] + return_map_0(kt_4, d_ang_6.z, p_22.z, slide_cap_3 * (*b_57).geom1_0.z).y;
    return p_22;
}

fn life_rate_1( mat_16 : ptr<function, JointMaterial_std140_0>,  s_12 : f32) -> f32
{
    if(s_12 <= 0.0f)
    {
        return 0.0f;
    }
    var _S840 : f32 = (*mat_16).misc_0.y;
    return fdiv_0((_S840 + 1.0f) * fpow_0(s_12, _S840), (*mat_16).misc_0.z);
}

fn joint_evaluate_1( mat_17 : ptr<function, JointMaterial_std140_0>,  b_58 : ptr<function, JointBond_std430_0>,  state_3 : ptr<function, JointState_std430_0>,  d_lin_8 : vec3<f32>,  d_ang_7 : vec3<f32>,  dt_19 : f32,  fracture_3 : bool) -> JointResponse_0
{
    var kn_6 : f32 = (*b_58).stiff0_0.x;
    var ks_5 : f32 = (*b_58).stiff0_0.y;
    var kb1_1 : f32 = (*b_58).stiff0_0.z;
    var kb2_1 : f32 = (*b_58).stiff0_0.w;
    var _S841 : vec4<f32> = (*b_58).stiff1_0;
    var kt_5 : f32 = (*b_58).stiff1_0.x;
    var has_rebar_6 : bool = ((*b_58).stiff1_0.w) != 0.0f;
    var kind_5 : u32 = (*mat_17).kind_flags_0.x;
    var flags_3 : u32 = (*mat_17).kind_flags_0.y;
    var softening_1 : bool = ((flags_3 & (u32(1)))) != u32(0);
    var st_9 : JointState_0;
    st_9.damage_0 = (*state_3).damage_0;
    st_9.crush_1 = (*state_3).crush_1;
    st_9.kappa_0 = (*state_3).kappa_0;
    st_9.kappa_c_0 = (*state_3).kappa_c_0;
    st_9.ductility_0 = (*state_3).ductility_0;
    st_9.ductility_c_0 = (*state_3).ductility_c_0;
    st_9.life_0 = (*state_3).life_0;
    st_9.plastic_x_0 = (*state_3).plastic_x_0;
    st_9.plastic_y_0 = (*state_3).plastic_y_0;
    st_9.plastic_t_0 = (*state_3).plastic_t_0;
    st_9.rebar_plastic_0 = (*state_3).rebar_plastic_0;
    st_9.rebar_slip0_0 = (*state_3).rebar_slip0_0;
    st_9.rebar_slip1_0 = (*state_3).rebar_slip1_0;
    st_9.rebar_work_0 = (*state_3).rebar_work_0;
    st_9.rebar_broken_0 = (*state_3).rebar_broken_0;
    st_9.strain_rate_0 = (*state_3).strain_rate_0;
    st_9.governing_stress_0 = (*state_3).governing_stress_0;
    st_9.dissipated_0 = (*state_3).dissipated_0;
    st_9.utilization_0 = (*state_3).utilization_0;
    st_9.mode_0 = (*state_3).mode_0;
    var _S842 : bool = connected_4(&((*state_3)), has_rebar_6);
    var qe_lin_1 : vec3<f32> = d_lin_8 * vec3<f32>(ks_5, ks_5, kn_6);
    var qe_ang_1 : vec3<f32> = d_ang_7 * vec3<f32>(kb1_1, kb2_1, kt_5);
    var _S843 : Measures_0 = stress_measures_1(&((*b_58)), qe_lin_1, qe_ang_1);
    var _S844 : f32 = max(max(_S843.tension_0, _S843.shear_0), _S843.compression_0);
    var _S845 : bool = dt_19 > 0.0f;
    var dif_2 : f32;
    if(_S845)
    {
        var raw_1 : f32 = fdiv_0(max(fdiv_0(_S844 - st_9.governing_stress_0, dt_19), 0.0f), (*mat_17).misc_0.w);
        var tau_3 : f32 = _S841.z;
        if(((flags_3 & (u32(16)))) != u32(0))
        {
            dif_2 = - expm1_accurate_0(- fdiv_0(dt_19, tau_3));
        }
        else
        {
            dif_2 = min(fdiv_0(dt_19, tau_3), 1.0f);
        }
        st_9.strain_rate_0 = st_9.strain_rate_0 + (raw_1 - st_9.strain_rate_0) * dif_2;
        st_9.governing_stress_0 = _S844;
    }
    if(((flags_3 & (u32(32)))) != u32(0))
    {
        var _S846 : f32 = dif_factor_1(&((*mat_17)), st_9.strain_rate_0);
        dif_2 = _S846;
    }
    else
    {
        dif_2 = 1.0f;
    }
    var weibull_1 : f32 = (*b_58).geom1_0.w;
    var _S847 : f32 = fatigue_factor_2(&((*mat_17)), st_9.life_0);
    var multiplier_3 : f32 = weibull_1 * dif_2 * _S847;
    var _S848 : vec4<f32> = failure_indices_1(&((*mat_17)), &((*b_58)), _S843, multiplier_3);
    var _S849 : f32 = _S848.x;
    var _S850 : f32 = _S848.y;
    st_9.utilization_0 = max(max(_S849, _S850), max(_S848.z, _S848.w));
    var _S851 : f32 = d_lin_8.x;
    var _S852 : f32 = d_lin_8.y;
    var _S853 : f32 = d_ang_7.z;
    var _S854 : f32 = ks_5 * (sq_0(_S851) + sq_0(_S852)) + kb1_1 * sq_0(d_ang_7.x) + kb2_1 * sq_0(d_ang_7.y) + kt_5 * sq_0(_S853);
    var _S855 : f32 = d_lin_8.z;
    var _S856 : bool = _S855 > 0.0f;
    if(_S856)
    {
        dif_2 = kn_6 * sq_0(_S855);
    }
    else
    {
        dif_2 = 0.0f;
    }
    var psi_ts_1 : f32 = 0.5f * (_S854 + dif_2);
    var psi_c_1 : f32;
    if(_S855 < 0.0f)
    {
        psi_c_1 = 0.5f * kn_6 * sq_0(_S855);
    }
    else
    {
        psi_c_1 = 0.0f;
    }
    var plastic_7 : vec3<f32> = vec3<f32>(st_9.plastic_x_0, st_9.plastic_y_0, st_9.plastic_t_0);
    var changed_1 : bool;
    if(((flags_3 & (u32(2)))) != u32(0))
    {
        changed_1 = ((*state_3).damage_0) != 0.0f;
    }
    else
    {
        changed_1 = false;
    }
    var patch_1 : f32;
    if(changed_1)
    {
        var _S857 : vec4<f32> = contact_normal_1(&((*mat_17)), &((*b_58)), (*state_3).crush_1, d_lin_8, d_ang_7);
        patch_1 = _S857.w + 0.5f * (ks_5 * (sq_0(_S851 - (*state_3).plastic_x_0) + sq_0(_S852 - (*state_3).plastic_y_0)) + kt_5 * sq_0(_S853 - (*state_3).plastic_t_0));
    }
    else
    {
        patch_1 = 0.0f;
    }
    if(has_rebar_6)
    {
        changed_1 = ((*state_3).rebar_broken_0) == 0.0f;
    }
    else
    {
        changed_1 = false;
    }
    var rebar_1 : f32;
    if(changed_1)
    {
        rebar_1 = 0.5f * ((*b_58).rebar0_0.x * sq_0(_S855 - (*state_3).rebar_plastic_0) + (*b_58).rebar0_0.y * (sq_0(_S851 - (*state_3).rebar_slip0_0) + sq_0(_S852 - (*state_3).rebar_slip1_0)));
    }
    else
    {
        rebar_1 = 0.0f;
    }
    var _S858 : f32 = (*state_3).damage_0;
    var _S859 : f32 = 1.0f - (*state_3).damage_0;
    var _S860 : f32 = (*state_3).crush_1;
    var frozen_1 : f32 = _S859 * (psi_ts_1 + (1.0f - (*state_3).crush_1) * psi_c_1) + (*state_3).damage_0 * patch_1 + rebar_1;
    var diss_contact_1 : f32;
    var psi_contact_1 : f32;
    var intact_normal_1 : f32;
    var dissipated_11 : f32;
    var qc_lin_1 : vec3<f32>;
    if(fracture_3)
    {
        var _S861 : bool = _S849 >= _S850;
        if(_S861)
        {
            diss_contact_1 = _S849;
        }
        else
        {
            diss_contact_1 = _S850;
        }
        var mode_ts_1 : u32;
        if(_S861)
        {
            mode_ts_1 = u32(1);
        }
        else
        {
            mode_ts_1 = u32(2);
        }
        if(diss_contact_1 > (st_9.kappa_0))
        {
            changed_1 = diss_contact_1 > 1.0f;
        }
        else
        {
            changed_1 = false;
        }
        if(changed_1)
        {
            changed_1 = psi_ts_1 > 0.0f;
        }
        else
        {
            changed_1 = false;
        }
        var mode_c_1 : u32;
        if(changed_1)
        {
            if(mode_ts_1 == u32(1))
            {
                psi_contact_1 = (*mat_17).energy_1.y;
            }
            else
            {
                psi_contact_1 = (*mat_17).energy_1.z;
            }
            if(softening_1)
            {
                intact_normal_1 = fdiv_0(psi_contact_1 * (*b_58).geom0_0.x * diss_contact_1 * diss_contact_1, psi_ts_1);
            }
            else
            {
                intact_normal_1 = 0.0f;
            }
            st_9.ductility_0 = intact_normal_1;
            if(softening_1)
            {
                mode_c_1 = kind_5;
            }
            else
            {
                mode_c_1 = u32(0);
            }
            var inc_2 : vec2<f32> = damage_increment_0(mode_c_1, st_9.kappa_0, diss_contact_1, intact_normal_1, st_9.damage_0, psi_ts_1);
            var _S862 : f32 = inc_2.x;
            if(_S862 > (st_9.damage_0))
            {
                var _S863 : Contact_0 = contact_part_1(&((*mat_17)), &((*b_58)), st_9.crush_1, plastic_7, d_lin_8, d_ang_7);
                var _S864 : f32 = max(inc_2.y - max(_S863.energy_5 - (1.0f - st_9.crush_1) * psi_c_1, 0.0f) * (_S862 - st_9.damage_0), 0.0f);
                st_9.damage_0 = _S862;
                st_9.mode_0 = mode_ts_1;
                dissipated_11 = _S864;
            }
            else
            {
                dissipated_11 = 0.0f;
            }
        }
        else
        {
            dissipated_11 = 0.0f;
        }
        st_9.kappa_0 = max(st_9.kappa_0, diss_contact_1);
        if(_S858 > 0.0f)
        {
            var _S865 : Contact_0 = contact_part_1(&((*mat_17)), &((*b_58)), _S860, vec3<f32>((*state_3).plastic_x_0, (*state_3).plastic_y_0, (*state_3).plastic_t_0), d_lin_8, d_ang_7);
            qc_lin_1 = qe_ang_1 * vec3<f32>(_S859) + _S865.q_ang_1 * vec3<f32>(_S858);
        }
        else
        {
            qc_lin_1 = qe_ang_1;
        }
        var _S866 : Measures_0 = stress_measures_1(&((*b_58)), vec3<f32>(0.0f, 0.0f, min(qe_lin_1.z, 0.0f)), qc_lin_1);
        var _S867 : vec4<f32> = failure_indices_1(&((*mat_17)), &((*b_58)), _S866, multiplier_3);
        var _S868 : f32 = _S867.z;
        var _S869 : f32 = _S867.w;
        var _S870 : bool = _S868 >= _S869;
        if(_S870)
        {
            psi_contact_1 = _S868;
        }
        else
        {
            psi_contact_1 = _S869;
        }
        if(_S870)
        {
            mode_c_1 = u32(3);
        }
        else
        {
            mode_c_1 = u32(4);
        }
        if(psi_contact_1 > (st_9.kappa_c_0))
        {
            changed_1 = psi_contact_1 > 1.0f;
        }
        else
        {
            changed_1 = false;
        }
        if(changed_1)
        {
            changed_1 = psi_c_1 > 0.0f;
        }
        else
        {
            changed_1 = false;
        }
        if(changed_1)
        {
            if(softening_1)
            {
                intact_normal_1 = fdiv_0((*mat_17).energy_1.w * (*b_58).geom0_0.x * psi_contact_1 * psi_contact_1, psi_c_1);
            }
            else
            {
                intact_normal_1 = 0.0f;
            }
            st_9.ductility_c_0 = intact_normal_1;
            var law_2 : u32;
            if(!softening_1)
            {
                law_2 = u32(0);
            }
            else
            {
                if(mode_c_1 == u32(4))
                {
                    mode_ts_1 = u32(1);
                }
                else
                {
                    mode_ts_1 = kind_5;
                }
                law_2 = mode_ts_1;
            }
            var inc_3 : vec2<f32> = damage_increment_0(law_2, st_9.kappa_c_0, psi_contact_1, intact_normal_1, st_9.crush_1, psi_c_1);
            var _S871 : f32 = inc_3.x;
            if(_S871 > (st_9.crush_1))
            {
                var dissipated_12 : f32 = dissipated_11 + inc_3.y;
                st_9.crush_1 = _S871;
                st_9.mode_0 = mode_c_1;
                if(_S871 >= 1.0f)
                {
                    changed_1 = (st_9.damage_0) < 1.0f;
                }
                else
                {
                    changed_1 = false;
                }
                if(changed_1)
                {
                    var dissipated_13 : f32 = dissipated_12 + psi_ts_1 * (1.0f - st_9.damage_0);
                    st_9.damage_0 = 1.0f;
                    dissipated_11 = dissipated_13;
                }
                else
                {
                    dissipated_11 = dissipated_12;
                }
            }
        }
        st_9.kappa_c_0 = max(st_9.kappa_c_0, psi_contact_1);
    }
    else
    {
        dissipated_11 = 0.0f;
    }
    var dmg_1 : f32 = st_9.damage_0;
    var _S872 : vec3<f32> = vec3<f32>(0.0f);
    var qc_ang_1 : vec3<f32>;
    if((st_9.damage_0) == 0.0f)
    {
        if(((flags_3 & (u32(8)))) == u32(0))
        {
            var _S873 : vec3<f32> = contact_offsets_1(&((*mat_17)), &((*b_58)), st_9.crush_1, plastic_7, d_lin_8, d_ang_7);
            st_9.plastic_x_0 = _S873.x;
            st_9.plastic_y_0 = _S873.y;
            st_9.plastic_t_0 = _S873.z;
        }
        diss_contact_1 = 0.0f;
        qc_lin_1 = _S872;
        qc_ang_1 = _S872;
        psi_contact_1 = 0.0f;
    }
    else
    {
        var _S874 : Contact_0 = contact_part_1(&((*mat_17)), &((*b_58)), st_9.crush_1, plastic_7, d_lin_8, d_ang_7);
        st_9.plastic_x_0 = _S874.plastic_0.x;
        st_9.plastic_y_0 = _S874.plastic_0.y;
        st_9.plastic_t_0 = _S874.plastic_0.z;
        diss_contact_1 = _S874.diss_1;
        qc_lin_1 = _S874.q_lin_1;
        qc_ang_1 = _S874.q_ang_1;
        psi_contact_1 = _S874.energy_5;
    }
    var dissipated_14 : f32 = dissipated_11 + dmg_1 * diss_contact_1;
    if(_S856)
    {
        intact_normal_1 = kn_6 * _S855;
    }
    else
    {
        intact_normal_1 = (1.0f - st_9.crush_1) * kn_6 * _S855;
    }
    var _S875 : f32 = 1.0f - dmg_1;
    var force_lin_4 : vec3<f32> = vec3<f32>(_S875 * qe_lin_1.x + dmg_1 * qc_lin_1.x, _S875 * qe_lin_1.y + dmg_1 * qc_lin_1.y, _S875 * intact_normal_1 + dmg_1 * qc_lin_1.z);
    var force_ang_3 : vec3<f32> = qe_ang_1 * vec3<f32>(_S875) + qc_ang_1 * vec3<f32>(dmg_1);
    var stored_10 : f32 = _S875 * (psi_ts_1 + (1.0f - st_9.crush_1) * psi_c_1) + dmg_1 * psi_contact_1;
    if(has_rebar_6)
    {
        changed_1 = (st_9.rebar_broken_0) == 0.0f;
    }
    else
    {
        changed_1 = false;
    }
    var stored_11 : f32;
    var force_lin_5 : vec3<f32>;
    if(changed_1)
    {
        var k_axial_1 : f32 = (*b_58).rebar0_0.x;
        var k_dowel_1 : f32 = (*b_58).rebar0_0.y;
        var yield_force_1 : f32 = (*b_58).rebar0_0.z;
        var dowel_capacity_1 : f32 = (*b_58).rebar0_0.w;
        var nr_1 : vec2<f32> = return_map_0(k_axial_1, _S855, st_9.rebar_plastic_0, yield_force_1);
        var v1_1 : vec2<f32> = return_map_0(k_dowel_1, _S851, st_9.rebar_slip0_0, dowel_capacity_1);
        var v2_1 : vec2<f32> = return_map_0(k_dowel_1, _S852, st_9.rebar_slip1_0, dowel_capacity_1);
        var _S876 : f32 = nr_1.y;
        var _S877 : f32 = v1_1.y;
        var _S878 : f32 = v2_1.y;
        var work_8 : f32 = yield_force_1 * abs(_S876) + dowel_capacity_1 * (abs(_S877) + abs(_S878));
        st_9.rebar_plastic_0 = st_9.rebar_plastic_0 + _S876;
        st_9.rebar_slip0_0 = st_9.rebar_slip0_0 + _S877;
        st_9.rebar_slip1_0 = st_9.rebar_slip1_0 + _S878;
        st_9.rebar_work_0 = st_9.rebar_work_0 + work_8;
        var dissipated_15 : f32 = dissipated_14 + work_8;
        var _S879 : f32 = nr_1.x;
        var _S880 : f32 = v1_1.x;
        var _S881 : f32 = v2_1.x;
        var elastic_1 : f32 = 0.5f * (fdiv_0(sq_0(_S879), k_axial_1) + fdiv_0(sq_0(_S880) + sq_0(_S881), k_dowel_1));
        if(fracture_3)
        {
            changed_1 = (st_9.rebar_work_0) >= ((*b_58).rebar1_0.x);
        }
        else
        {
            changed_1 = false;
        }
        if(changed_1)
        {
            st_9.rebar_broken_0 = 1.0f;
            var dissipated_16 : f32 = dissipated_15 + elastic_1;
            force_lin_5 = force_lin_4;
            stored_11 = stored_10;
            dissipated_11 = dissipated_16;
        }
        else
        {
            var stored_12 : f32 = stored_10 + elastic_1;
            force_lin_5 = force_lin_4 + vec3<f32>(_S880, _S881, _S879);
            stored_11 = stored_12;
            dissipated_11 = dissipated_15;
        }
    }
    else
    {
        force_lin_5 = force_lin_4;
        stored_11 = stored_10;
        dissipated_11 = dissipated_14;
    }
    if(fracture_3)
    {
        changed_1 = _S845;
    }
    else
    {
        changed_1 = false;
    }
    if(changed_1)
    {
        changed_1 = ((flags_3 & (u32(64)))) != u32(0);
    }
    else
    {
        changed_1 = false;
    }
    if(changed_1)
    {
        var _S882 : Measures_0 = stress_measures_1(&((*b_58)), force_lin_5, force_ang_3);
        var _S883 : vec4<f32> = failure_indices_1(&((*mat_17)), &((*b_58)), _S882, weibull_1);
        var _S884 : f32 = life_rate_1(&((*mat_17)), max(max(_S883.x, _S883.y), _S883.z));
        st_9.life_0 = max(st_9.life_0 - _S884 * dt_19, 0.0f);
    }
    if((st_9.damage_0) != _S858)
    {
        changed_1 = true;
    }
    else
    {
        changed_1 = (st_9.crush_1) != _S860;
    }
    if(changed_1)
    {
        changed_1 = true;
    }
    else
    {
        changed_1 = (st_9.plastic_x_0) != ((*state_3).plastic_x_0);
    }
    if(changed_1)
    {
        changed_1 = true;
    }
    else
    {
        changed_1 = (st_9.plastic_y_0) != ((*state_3).plastic_y_0);
    }
    if(changed_1)
    {
        changed_1 = true;
    }
    else
    {
        changed_1 = (st_9.plastic_t_0) != ((*state_3).plastic_t_0);
    }
    if(changed_1)
    {
        changed_1 = true;
    }
    else
    {
        changed_1 = (st_9.rebar_plastic_0) != ((*state_3).rebar_plastic_0);
    }
    if(changed_1)
    {
        changed_1 = true;
    }
    else
    {
        changed_1 = (st_9.rebar_slip0_0) != ((*state_3).rebar_slip0_0);
    }
    if(changed_1)
    {
        changed_1 = true;
    }
    else
    {
        changed_1 = (st_9.rebar_slip1_0) != ((*state_3).rebar_slip1_0);
    }
    if(changed_1)
    {
        changed_1 = true;
    }
    else
    {
        changed_1 = (st_9.rebar_broken_0) != ((*state_3).rebar_broken_0);
    }
    var released_1 : f32;
    if(changed_1)
    {
        released_1 = frozen_1 - stored_11;
    }
    else
    {
        released_1 = 0.0f;
    }
    var _S885 : f32 = max(min(dissipated_11, released_1), 0.0f);
    var overshoot_2 : f32 = released_1 - _S885;
    st_9.dissipated_0 = st_9.dissipated_0 + _S885;
    var resp_1 : JointResponse_0;
    resp_1.force_lin_1 = force_lin_5;
    resp_1.force_ang_1 = force_ang_3;
    resp_1.state_1 = st_9;
    resp_1.dissipated_2 = _S885;
    resp_1.overshoot_0 = overshoot_2;
    resp_1.stored_2 = stored_11;
    if(_S842)
    {
        changed_1 = !connected_2(st_9, has_rebar_6);
    }
    else
    {
        changed_1 = false;
    }
    resp_1.disconnected_0 = changed_1;
    resp_1.fatigue_before_0 = _S847;
    resp_1.measures_0 = _S843;
    return resp_1;
}

fn joint_evaluate_2( mat_18 : ptr<function, JointMaterial_std140_0>,  b_59 : ptr<function, JointBond_std430_0>,  state_4 : JointState_0,  d_lin_9 : vec3<f32>,  d_ang_8 : vec3<f32>,  dt_20 : f32,  fracture_4 : bool) -> JointResponse_0
{
    var kn_7 : f32 = (*b_59).stiff0_0.x;
    var ks_6 : f32 = (*b_59).stiff0_0.y;
    var kb1_2 : f32 = (*b_59).stiff0_0.z;
    var kb2_2 : f32 = (*b_59).stiff0_0.w;
    var _S886 : vec4<f32> = (*b_59).stiff1_0;
    var kt_6 : f32 = (*b_59).stiff1_0.x;
    var has_rebar_7 : bool = ((*b_59).stiff1_0.w) != 0.0f;
    var kind_6 : u32 = (*mat_18).kind_flags_0.x;
    var flags_4 : u32 = (*mat_18).kind_flags_0.y;
    var softening_2 : bool = ((flags_4 & (u32(1)))) != u32(0);
    var st_10 : JointState_0 = state_4;
    var was_connected_0 : bool = connected_2(state_4, has_rebar_7);
    var qe_lin_2 : vec3<f32> = d_lin_9 * vec3<f32>(ks_6, ks_6, kn_7);
    var qe_ang_2 : vec3<f32> = d_ang_8 * vec3<f32>(kb1_2, kb2_2, kt_6);
    var _S887 : Measures_0 = stress_measures_1(&((*b_59)), qe_lin_2, qe_ang_2);
    var _S888 : f32 = max(max(_S887.tension_0, _S887.shear_0), _S887.compression_0);
    var _S889 : bool = dt_20 > 0.0f;
    var dif_3 : f32;
    if(_S889)
    {
        var raw_2 : f32 = fdiv_0(max(fdiv_0(_S888 - st_10.governing_stress_0, dt_20), 0.0f), (*mat_18).misc_0.w);
        var tau_4 : f32 = _S886.z;
        if(((flags_4 & (u32(16)))) != u32(0))
        {
            dif_3 = - expm1_accurate_0(- fdiv_0(dt_20, tau_4));
        }
        else
        {
            dif_3 = min(fdiv_0(dt_20, tau_4), 1.0f);
        }
        st_10.strain_rate_0 = st_10.strain_rate_0 + (raw_2 - st_10.strain_rate_0) * dif_3;
        st_10.governing_stress_0 = _S888;
    }
    if(((flags_4 & (u32(32)))) != u32(0))
    {
        var _S890 : f32 = dif_factor_1(&((*mat_18)), st_10.strain_rate_0);
        dif_3 = _S890;
    }
    else
    {
        dif_3 = 1.0f;
    }
    var weibull_2 : f32 = (*b_59).geom1_0.w;
    var _S891 : f32 = fatigue_factor_2(&((*mat_18)), st_10.life_0);
    var multiplier_4 : f32 = weibull_2 * dif_3 * _S891;
    var _S892 : vec4<f32> = failure_indices_1(&((*mat_18)), &((*b_59)), _S887, multiplier_4);
    var _S893 : f32 = _S892.x;
    var _S894 : f32 = _S892.y;
    st_10.utilization_0 = max(max(_S893, _S894), max(_S892.z, _S892.w));
    var _S895 : f32 = d_lin_9.x;
    var _S896 : f32 = d_lin_9.y;
    var _S897 : f32 = d_ang_8.z;
    var _S898 : f32 = ks_6 * (sq_0(_S895) + sq_0(_S896)) + kb1_2 * sq_0(d_ang_8.x) + kb2_2 * sq_0(d_ang_8.y) + kt_6 * sq_0(_S897);
    var _S899 : f32 = d_lin_9.z;
    var _S900 : bool = _S899 > 0.0f;
    if(_S900)
    {
        dif_3 = kn_7 * sq_0(_S899);
    }
    else
    {
        dif_3 = 0.0f;
    }
    var psi_ts_2 : f32 = 0.5f * (_S898 + dif_3);
    var psi_c_2 : f32;
    if(_S899 < 0.0f)
    {
        psi_c_2 = 0.5f * kn_7 * sq_0(_S899);
    }
    else
    {
        psi_c_2 = 0.0f;
    }
    var plastic_8 : vec3<f32> = vec3<f32>(st_10.plastic_x_0, st_10.plastic_y_0, st_10.plastic_t_0);
    var changed_2 : bool;
    if(((flags_4 & (u32(2)))) != u32(0))
    {
        changed_2 = (state_4.damage_0) != 0.0f;
    }
    else
    {
        changed_2 = false;
    }
    var patch_2 : f32;
    if(changed_2)
    {
        var _S901 : vec4<f32> = contact_normal_1(&((*mat_18)), &((*b_59)), state_4.crush_1, d_lin_9, d_ang_8);
        patch_2 = _S901.w + 0.5f * (ks_6 * (sq_0(_S895 - state_4.plastic_x_0) + sq_0(_S896 - state_4.plastic_y_0)) + kt_6 * sq_0(_S897 - state_4.plastic_t_0));
    }
    else
    {
        patch_2 = 0.0f;
    }
    if(has_rebar_7)
    {
        changed_2 = (state_4.rebar_broken_0) == 0.0f;
    }
    else
    {
        changed_2 = false;
    }
    var rebar_2 : f32;
    if(changed_2)
    {
        rebar_2 = 0.5f * ((*b_59).rebar0_0.x * sq_0(_S899 - state_4.rebar_plastic_0) + (*b_59).rebar0_0.y * (sq_0(_S895 - state_4.rebar_slip0_0) + sq_0(_S896 - state_4.rebar_slip1_0)));
    }
    else
    {
        rebar_2 = 0.0f;
    }
    var _S902 : f32 = 1.0f - state_4.damage_0;
    var frozen_2 : f32 = _S902 * (psi_ts_2 + (1.0f - state_4.crush_1) * psi_c_2) + state_4.damage_0 * patch_2 + rebar_2;
    var diss_contact_2 : f32;
    var psi_contact_2 : f32;
    var intact_normal_2 : f32;
    var dissipated_17 : f32;
    var qc_lin_2 : vec3<f32>;
    if(fracture_4)
    {
        var _S903 : bool = _S893 >= _S894;
        if(_S903)
        {
            diss_contact_2 = _S893;
        }
        else
        {
            diss_contact_2 = _S894;
        }
        var mode_ts_2 : u32;
        if(_S903)
        {
            mode_ts_2 = u32(1);
        }
        else
        {
            mode_ts_2 = u32(2);
        }
        if(diss_contact_2 > (st_10.kappa_0))
        {
            changed_2 = diss_contact_2 > 1.0f;
        }
        else
        {
            changed_2 = false;
        }
        if(changed_2)
        {
            changed_2 = psi_ts_2 > 0.0f;
        }
        else
        {
            changed_2 = false;
        }
        var mode_c_2 : u32;
        if(changed_2)
        {
            if(mode_ts_2 == u32(1))
            {
                psi_contact_2 = (*mat_18).energy_1.y;
            }
            else
            {
                psi_contact_2 = (*mat_18).energy_1.z;
            }
            if(softening_2)
            {
                intact_normal_2 = fdiv_0(psi_contact_2 * (*b_59).geom0_0.x * diss_contact_2 * diss_contact_2, psi_ts_2);
            }
            else
            {
                intact_normal_2 = 0.0f;
            }
            st_10.ductility_0 = intact_normal_2;
            if(softening_2)
            {
                mode_c_2 = kind_6;
            }
            else
            {
                mode_c_2 = u32(0);
            }
            var inc_4 : vec2<f32> = damage_increment_0(mode_c_2, st_10.kappa_0, diss_contact_2, intact_normal_2, st_10.damage_0, psi_ts_2);
            var _S904 : f32 = inc_4.x;
            if(_S904 > (st_10.damage_0))
            {
                var _S905 : Contact_0 = contact_part_1(&((*mat_18)), &((*b_59)), st_10.crush_1, plastic_8, d_lin_9, d_ang_8);
                var _S906 : f32 = max(inc_4.y - max(_S905.energy_5 - (1.0f - st_10.crush_1) * psi_c_2, 0.0f) * (_S904 - st_10.damage_0), 0.0f);
                st_10.damage_0 = _S904;
                st_10.mode_0 = mode_ts_2;
                dissipated_17 = _S906;
            }
            else
            {
                dissipated_17 = 0.0f;
            }
        }
        else
        {
            dissipated_17 = 0.0f;
        }
        st_10.kappa_0 = max(st_10.kappa_0, diss_contact_2);
        if((state_4.damage_0) > 0.0f)
        {
            var _S907 : Contact_0 = contact_part_1(&((*mat_18)), &((*b_59)), state_4.crush_1, vec3<f32>(state_4.plastic_x_0, state_4.plastic_y_0, state_4.plastic_t_0), d_lin_9, d_ang_8);
            qc_lin_2 = qe_ang_2 * vec3<f32>(_S902) + _S907.q_ang_1 * vec3<f32>(state_4.damage_0);
        }
        else
        {
            qc_lin_2 = qe_ang_2;
        }
        var _S908 : Measures_0 = stress_measures_1(&((*b_59)), vec3<f32>(0.0f, 0.0f, min(qe_lin_2.z, 0.0f)), qc_lin_2);
        var _S909 : vec4<f32> = failure_indices_1(&((*mat_18)), &((*b_59)), _S908, multiplier_4);
        var _S910 : f32 = _S909.z;
        var _S911 : f32 = _S909.w;
        var _S912 : bool = _S910 >= _S911;
        if(_S912)
        {
            psi_contact_2 = _S910;
        }
        else
        {
            psi_contact_2 = _S911;
        }
        if(_S912)
        {
            mode_c_2 = u32(3);
        }
        else
        {
            mode_c_2 = u32(4);
        }
        if(psi_contact_2 > (st_10.kappa_c_0))
        {
            changed_2 = psi_contact_2 > 1.0f;
        }
        else
        {
            changed_2 = false;
        }
        if(changed_2)
        {
            changed_2 = psi_c_2 > 0.0f;
        }
        else
        {
            changed_2 = false;
        }
        if(changed_2)
        {
            if(softening_2)
            {
                intact_normal_2 = fdiv_0((*mat_18).energy_1.w * (*b_59).geom0_0.x * psi_contact_2 * psi_contact_2, psi_c_2);
            }
            else
            {
                intact_normal_2 = 0.0f;
            }
            st_10.ductility_c_0 = intact_normal_2;
            var law_3 : u32;
            if(!softening_2)
            {
                law_3 = u32(0);
            }
            else
            {
                if(mode_c_2 == u32(4))
                {
                    mode_ts_2 = u32(1);
                }
                else
                {
                    mode_ts_2 = kind_6;
                }
                law_3 = mode_ts_2;
            }
            var inc_5 : vec2<f32> = damage_increment_0(law_3, st_10.kappa_c_0, psi_contact_2, intact_normal_2, st_10.crush_1, psi_c_2);
            var _S913 : f32 = inc_5.x;
            if(_S913 > (st_10.crush_1))
            {
                var dissipated_18 : f32 = dissipated_17 + inc_5.y;
                st_10.crush_1 = _S913;
                st_10.mode_0 = mode_c_2;
                if(_S913 >= 1.0f)
                {
                    changed_2 = (st_10.damage_0) < 1.0f;
                }
                else
                {
                    changed_2 = false;
                }
                if(changed_2)
                {
                    var dissipated_19 : f32 = dissipated_18 + psi_ts_2 * (1.0f - st_10.damage_0);
                    st_10.damage_0 = 1.0f;
                    dissipated_17 = dissipated_19;
                }
                else
                {
                    dissipated_17 = dissipated_18;
                }
            }
        }
        st_10.kappa_c_0 = max(st_10.kappa_c_0, psi_contact_2);
    }
    else
    {
        dissipated_17 = 0.0f;
    }
    var dmg_2 : f32 = st_10.damage_0;
    var _S914 : vec3<f32> = vec3<f32>(0.0f);
    var qc_ang_2 : vec3<f32>;
    if((st_10.damage_0) == 0.0f)
    {
        if(((flags_4 & (u32(8)))) == u32(0))
        {
            var _S915 : vec3<f32> = contact_offsets_1(&((*mat_18)), &((*b_59)), st_10.crush_1, plastic_8, d_lin_9, d_ang_8);
            st_10.plastic_x_0 = _S915.x;
            st_10.plastic_y_0 = _S915.y;
            st_10.plastic_t_0 = _S915.z;
        }
        diss_contact_2 = 0.0f;
        qc_lin_2 = _S914;
        qc_ang_2 = _S914;
        psi_contact_2 = 0.0f;
    }
    else
    {
        var _S916 : Contact_0 = contact_part_1(&((*mat_18)), &((*b_59)), st_10.crush_1, plastic_8, d_lin_9, d_ang_8);
        st_10.plastic_x_0 = _S916.plastic_0.x;
        st_10.plastic_y_0 = _S916.plastic_0.y;
        st_10.plastic_t_0 = _S916.plastic_0.z;
        diss_contact_2 = _S916.diss_1;
        qc_lin_2 = _S916.q_lin_1;
        qc_ang_2 = _S916.q_ang_1;
        psi_contact_2 = _S916.energy_5;
    }
    var dissipated_20 : f32 = dissipated_17 + dmg_2 * diss_contact_2;
    if(_S900)
    {
        intact_normal_2 = kn_7 * _S899;
    }
    else
    {
        intact_normal_2 = (1.0f - st_10.crush_1) * kn_7 * _S899;
    }
    var _S917 : f32 = 1.0f - dmg_2;
    var force_lin_6 : vec3<f32> = vec3<f32>(_S917 * qe_lin_2.x + dmg_2 * qc_lin_2.x, _S917 * qe_lin_2.y + dmg_2 * qc_lin_2.y, _S917 * intact_normal_2 + dmg_2 * qc_lin_2.z);
    var force_ang_4 : vec3<f32> = qe_ang_2 * vec3<f32>(_S917) + qc_ang_2 * vec3<f32>(dmg_2);
    var stored_13 : f32 = _S917 * (psi_ts_2 + (1.0f - st_10.crush_1) * psi_c_2) + dmg_2 * psi_contact_2;
    if(has_rebar_7)
    {
        changed_2 = (st_10.rebar_broken_0) == 0.0f;
    }
    else
    {
        changed_2 = false;
    }
    var stored_14 : f32;
    var force_lin_7 : vec3<f32>;
    if(changed_2)
    {
        var k_axial_2 : f32 = (*b_59).rebar0_0.x;
        var k_dowel_2 : f32 = (*b_59).rebar0_0.y;
        var yield_force_2 : f32 = (*b_59).rebar0_0.z;
        var dowel_capacity_2 : f32 = (*b_59).rebar0_0.w;
        var nr_2 : vec2<f32> = return_map_0(k_axial_2, _S899, st_10.rebar_plastic_0, yield_force_2);
        var v1_2 : vec2<f32> = return_map_0(k_dowel_2, _S895, st_10.rebar_slip0_0, dowel_capacity_2);
        var v2_2 : vec2<f32> = return_map_0(k_dowel_2, _S896, st_10.rebar_slip1_0, dowel_capacity_2);
        var _S918 : f32 = nr_2.y;
        var _S919 : f32 = v1_2.y;
        var _S920 : f32 = v2_2.y;
        var work_9 : f32 = yield_force_2 * abs(_S918) + dowel_capacity_2 * (abs(_S919) + abs(_S920));
        st_10.rebar_plastic_0 = st_10.rebar_plastic_0 + _S918;
        st_10.rebar_slip0_0 = st_10.rebar_slip0_0 + _S919;
        st_10.rebar_slip1_0 = st_10.rebar_slip1_0 + _S920;
        st_10.rebar_work_0 = st_10.rebar_work_0 + work_9;
        var dissipated_21 : f32 = dissipated_20 + work_9;
        var _S921 : f32 = nr_2.x;
        var _S922 : f32 = v1_2.x;
        var _S923 : f32 = v2_2.x;
        var elastic_2 : f32 = 0.5f * (fdiv_0(sq_0(_S921), k_axial_2) + fdiv_0(sq_0(_S922) + sq_0(_S923), k_dowel_2));
        if(fracture_4)
        {
            changed_2 = (st_10.rebar_work_0) >= ((*b_59).rebar1_0.x);
        }
        else
        {
            changed_2 = false;
        }
        if(changed_2)
        {
            st_10.rebar_broken_0 = 1.0f;
            var dissipated_22 : f32 = dissipated_21 + elastic_2;
            force_lin_7 = force_lin_6;
            stored_14 = stored_13;
            dissipated_17 = dissipated_22;
        }
        else
        {
            var stored_15 : f32 = stored_13 + elastic_2;
            force_lin_7 = force_lin_6 + vec3<f32>(_S922, _S923, _S921);
            stored_14 = stored_15;
            dissipated_17 = dissipated_21;
        }
    }
    else
    {
        force_lin_7 = force_lin_6;
        stored_14 = stored_13;
        dissipated_17 = dissipated_20;
    }
    if(fracture_4)
    {
        changed_2 = _S889;
    }
    else
    {
        changed_2 = false;
    }
    if(changed_2)
    {
        changed_2 = ((flags_4 & (u32(64)))) != u32(0);
    }
    else
    {
        changed_2 = false;
    }
    if(changed_2)
    {
        var _S924 : Measures_0 = stress_measures_1(&((*b_59)), force_lin_7, force_ang_4);
        var _S925 : vec4<f32> = failure_indices_1(&((*mat_18)), &((*b_59)), _S924, weibull_2);
        var _S926 : f32 = life_rate_1(&((*mat_18)), max(max(_S925.x, _S925.y), _S925.z));
        st_10.life_0 = max(st_10.life_0 - _S926 * dt_20, 0.0f);
    }
    if((st_10.damage_0) != (state_4.damage_0))
    {
        changed_2 = true;
    }
    else
    {
        changed_2 = (st_10.crush_1) != (state_4.crush_1);
    }
    if(changed_2)
    {
        changed_2 = true;
    }
    else
    {
        changed_2 = (st_10.plastic_x_0) != (state_4.plastic_x_0);
    }
    if(changed_2)
    {
        changed_2 = true;
    }
    else
    {
        changed_2 = (st_10.plastic_y_0) != (state_4.plastic_y_0);
    }
    if(changed_2)
    {
        changed_2 = true;
    }
    else
    {
        changed_2 = (st_10.plastic_t_0) != (state_4.plastic_t_0);
    }
    if(changed_2)
    {
        changed_2 = true;
    }
    else
    {
        changed_2 = (st_10.rebar_plastic_0) != (state_4.rebar_plastic_0);
    }
    if(changed_2)
    {
        changed_2 = true;
    }
    else
    {
        changed_2 = (st_10.rebar_slip0_0) != (state_4.rebar_slip0_0);
    }
    if(changed_2)
    {
        changed_2 = true;
    }
    else
    {
        changed_2 = (st_10.rebar_slip1_0) != (state_4.rebar_slip1_0);
    }
    if(changed_2)
    {
        changed_2 = true;
    }
    else
    {
        changed_2 = (st_10.rebar_broken_0) != (state_4.rebar_broken_0);
    }
    var released_2 : f32;
    if(changed_2)
    {
        released_2 = frozen_2 - stored_14;
    }
    else
    {
        released_2 = 0.0f;
    }
    var _S927 : f32 = max(min(dissipated_17, released_2), 0.0f);
    var overshoot_3 : f32 = released_2 - _S927;
    st_10.dissipated_0 = st_10.dissipated_0 + _S927;
    var resp_2 : JointResponse_0;
    resp_2.force_lin_1 = force_lin_7;
    resp_2.force_ang_1 = force_ang_4;
    resp_2.state_1 = st_10;
    resp_2.dissipated_2 = _S927;
    resp_2.overshoot_0 = overshoot_3;
    resp_2.stored_2 = stored_14;
    if(was_connected_0)
    {
        changed_2 = !connected_2(st_10, has_rebar_7);
    }
    else
    {
        changed_2 = false;
    }
    resp_2.disconnected_0 = changed_2;
    resp_2.fatigue_before_0 = _S891;
    resp_2.measures_0 = _S887;
    return resp_2;
}

fn static_kinematics_0( _S928 : u32,  _S929 : ptr<function, vec3<f32>>,  _S930 : ptr<function, vec3<f32>>)
{
    var ca_1 : u32 = bonds_0[_S928].law_0.ids_0.y;
    var cb_5 : u32 = bonds_0[_S928].law_0.ids_0.z;
    var _S931 : u32 = u32(4) * cb_5;
    var _S932 : u32 = u32(4) * ca_1;
    var _S933 : u32 = _S931 + u32(1);
    var _S934 : u32 = _S932 + u32(1);
    var _S935 : u32 = sv_0(cb_5, u32(22));
    var _S936 : u32 = sv_0(ca_1, u32(22));
    var dth_0 : vec3<f32> = state_0[_S933].xyz - state_0[_S934].xyz + (scratch_0[_S935].xyz - scratch_0[_S936].xyz);
    (*_S929) = to_local_0(_S928, state_0[_S931].xyz - state_0[_S932].xyz + (scratch_0[sv_0(cb_5, u32(21))].xyz - scratch_0[sv_0(ca_1, u32(21))].xyz) + (cross(state_0[_S933].xyz + scratch_0[_S935].xyz, bonds_0[_S928].rb_0.xyz) - cross(state_0[_S934].xyz + scratch_0[_S936].xyz, bonds_0[_S928].ra_0.xyz)));
    (*_S930) = to_local_0(_S928, dth_0);
    return;
}

fn static_response_0( i_13 : u32) -> JointResponse_0
{
    var d_lin_10 : vec3<f32>;
    var d_ang_9 : vec3<f32>;
    static_kinematics_0(i_13, &(d_lin_10), &(d_ang_9));
    var _S937 : JointBond_std430_0 = bonds_0[i_13].law_0;
    var _S938 : JointMaterial_std140_0 = materials_0.m_0.data_0[_S937.ids_0.x];
    var _S939 : JointState_std430_0 = bond_dyn_0[i_13].js_0;
    var _S940 : JointResponse_0 = joint_evaluate_1(&(_S938), &(_S937), &(_S939), d_lin_10, d_ang_9, 0.0f, false);
    return _S940;
}

fn gather_loads_0( c_44 : u32,  fi_3 : ptr<function, vec3<f32>>,  mi_6 : ptr<function, vec3<f32>>)
{
    var _S941 : vec3<f32> = vec3<f32>(0.0f);
    (*fi_3) = _S941;
    (*mi_6) = _S941;
    var e_7 : u32 = index_0[c_44];
    loop
    {
        if(e_7 < index_0[c_44 + u32(1)])
        {
        }
        else
        {
            break;
        }
        var entry_6 : u32 = index_0[e_7];
        var bond_2 : u32 = (entry_6 >> (u32(1)));
        if(((entry_6 & (u32(1)))) == u32(0))
        {
            var _S942 : u32 = u32(3) * bond_2;
            (*fi_3) = (*fi_3) + scratch_0[_S942].xyz;
            (*mi_6) = (*mi_6) + scratch_0[_S942 + u32(1)].xyz;
        }
        else
        {
            var _S943 : u32 = u32(3) * bond_2;
            (*fi_3) = (*fi_3) - scratch_0[_S943].xyz;
            (*mi_6) = (*mi_6) + scratch_0[_S943 + u32(2)].xyz;
        }
        e_7 = e_7 + u32(1);
    }
    return;
}

fn bond_load_magnitude2_0( c_45 : u32) -> f32
{
    var e_8 : u32 = index_0[c_45];
    var m_8 : f32 = 0.0f;
    loop
    {
        if(e_8 < index_0[c_45 + u32(1)])
        {
        }
        else
        {
            break;
        }
        var entry_7 : u32 = index_0[e_8];
        var _S944 : u32 = u32(3) * ((entry_7 >> (u32(1))));
        var f_28 : vec3<f32> = scratch_0[_S944].xyz;
        var t_28 : vec3<f32>;
        if(((entry_7 & (u32(1)))) == u32(0))
        {
            t_28 = scratch_0[_S944 + u32(1)].xyz;
        }
        else
        {
            t_28 = scratch_0[_S944 + u32(2)].xyz;
        }
        var m_9 : f32 = m_8 + (dot(f_28, f_28) + dot(t_28, t_28));
        e_8 = e_8 + u32(1);
        m_8 = m_9;
    }
    return m_8;
}

fn fixed_mask_0( c_46 : u32) -> u32
{
    var support_2 : u32 = chunks_0[c_46].info_1.x;
    var _S945 : u32;
    if(support_2 == u32(1))
    {
        _S945 = u32(63);
    }
    else
    {
        if(support_2 == u32(2))
        {
            _S945 = u32(7);
        }
        else
        {
            _S945 = u32(0);
        }
    }
    return _S945;
}

fn hold_0( mask_0 : u32,  lin_0 : ptr<function, vec4<f32>>,  ang_0 : ptr<function, vec4<f32>>,  keep_lin_0 : vec4<f32>,  keep_ang_0 : vec4<f32>)
{
    var d_24 : u32 = u32(0);
    loop
    {
        if(d_24 < u32(3))
        {
        }
        else
        {
            break;
        }
        if(((mask_0 & (((u32(1) << (d_24)))))) != u32(0))
        {
            (*lin_0)[d_24] = keep_lin_0[d_24];
        }
        if(((mask_0 & (((u32(1) << ((d_24 + u32(3)))))))) != u32(0))
        {
            (*ang_0)[d_24] = keep_ang_0[d_24];
        }
        d_24 = d_24 + u32(1);
    }
    return;
}

fn is_finite_0( x_25 : f32) -> bool
{
    return (((bitcast<u32>((x_25))) & (u32(2147483647)))) < u32(2139095040);
}

fn secant_factors_1( b_60 : ptr<function, JointBond_std430_0>,  st_11 : ptr<function, JointState_std430_0>,  d_lin_11 : vec3<f32>,  f_lin_2 : ptr<function, vec3<f32>>,  f_ang_2 : ptr<function, vec3<f32>>)
{
    var _S946 : f32 = (*st_11).damage_0;
    var compressed_1 : bool = (d_lin_11.z) < 0.0f;
    var contact_6 : f32;
    if(compressed_1)
    {
        contact_6 = _S946;
    }
    else
    {
        contact_6 = 0.0f;
    }
    var _S947 : f32 = 1.0f - _S946;
    var _S948 : f32 = max(_S947 + contact_6, 9.99999997475242708e-07f);
    var normal_10 : f32;
    if(compressed_1)
    {
        normal_10 = max(1.0f - (*st_11).crush_1, 9.99999997475242708e-07f);
    }
    else
    {
        normal_10 = max(_S947, 9.99999997475242708e-07f);
    }
    (*f_lin_2) = vec3<f32>(_S948, _S948, normal_10);
    (*f_ang_2) = vec3<f32>(_S948);
    var _S949 : bool;
    if(((*b_60).stiff1_0.w) != 0.0f)
    {
        _S949 = ((*st_11).rebar_broken_0) == 0.0f;
    }
    else
    {
        _S949 = false;
    }
    if(_S949)
    {
        var _S950 : vec4<f32> = (*b_60).rebar0_0;
        var _S951 : vec4<f32> = (*b_60).stiff0_0;
        (*f_lin_2)[i32(2)] = (*f_lin_2)[i32(2)] + fdiv_0((*b_60).rebar0_0.x, (*b_60).stiff0_0.x);
        var _S952 : f32 = fdiv_0(_S950.y, _S951.y);
        (*f_lin_2)[i32(0)] = (*f_lin_2)[i32(0)] + _S952;
        (*f_lin_2)[i32(1)] = (*f_lin_2)[i32(1)] + _S952;
    }
    return;
}

fn statics_bond_slot_0( i_14 : u32) -> u32
{
    return params_0.statics_base_0 + u32(23) * params_0.chunk_count_0 + u32(2) * i_14;
}

fn store_inverse_0( c_47 : u32,  a_27 : array<f32, i32(36)>)
{
    var j_11 : u32;
    var sum_10 : f32;
    var l_8 : array<f32, i32(36)>;
    var k_26 : u32 = u32(0);
    loop
    {
        if(k_26 < u32(36))
        {
        }
        else
        {
            break;
        }
        l_8[k_26] = 0.0f;
        k_26 = k_26 + u32(1);
    }
    var spd_0 : bool = true;
    var i_15 : u32 = u32(0);
    loop
    {
        var _S953 : bool;
        if(i_15 < u32(6))
        {
            _S953 = spd_0;
        }
        else
        {
            _S953 = false;
        }
        if(_S953)
        {
        }
        else
        {
            break;
        }
        j_11 = u32(0);
        loop
        {
            if(j_11 <= i_15)
            {
            }
            else
            {
                break;
            }
            var _S954 : u32 = i_15 * u32(6);
            var _S955 : u32 = _S954 + j_11;
            k_26 = u32(0);
            sum_10 = a_27[_S955];
            loop
            {
                if(k_26 < j_11)
                {
                }
                else
                {
                    break;
                }
                var sum_11 : f32 = sum_10 - l_8[_S954 + k_26] * l_8[j_11 * u32(6) + k_26];
                k_26 = k_26 + u32(1);
                sum_10 = sum_11;
            }
            if(i_15 == j_11)
            {
                if(sum_10 <= 0.0f)
                {
                    spd_0 = false;
                    break;
                }
                l_8[_S954 + i_15] = sqrt(sum_10);
            }
            else
            {
                l_8[_S955] = sum_10 / l_8[j_11 * u32(6) + j_11];
            }
            j_11 = j_11 + u32(1);
        }
        i_15 = i_15 + u32(1);
    }
    var inv_0 : array<f32, i32(36)>;
    if(!spd_0)
    {
        k_26 = u32(0);
        loop
        {
            if(k_26 < u32(36))
            {
            }
            else
            {
                break;
            }
            inv_0[k_26] = 0.0f;
            k_26 = k_26 + u32(1);
        }
        k_26 = u32(0);
        loop
        {
            if(k_26 < u32(6))
            {
            }
            else
            {
                break;
            }
            var _S956 : u32 = k_26 * u32(6) + k_26;
            if((a_27[_S956]) > 0.0f)
            {
                sum_10 = 1.0f / a_27[_S956];
            }
            else
            {
                sum_10 = 0.0f;
            }
            inv_0[_S956] = sum_10;
            k_26 = k_26 + u32(1);
        }
    }
    else
    {
        j_11 = u32(0);
        loop
        {
            if(j_11 < u32(6))
            {
            }
            else
            {
                break;
            }
            var y_7 : array<f32, i32(6)>;
            y_7[i32(0)] = 0.0f;
            y_7[i32(1)] = 0.0f;
            y_7[i32(2)] = 0.0f;
            y_7[i32(3)] = 0.0f;
            y_7[i32(4)] = 0.0f;
            y_7[i32(5)] = 0.0f;
            i_15 = u32(0);
            loop
            {
                if(i_15 < u32(6))
                {
                }
                else
                {
                    break;
                }
                if(i_15 == j_11)
                {
                    sum_10 = 1.0f;
                }
                else
                {
                    sum_10 = 0.0f;
                }
                k_26 = u32(0);
                var s_13 : f32 = sum_10;
                loop
                {
                    if(k_26 < i_15)
                    {
                    }
                    else
                    {
                        break;
                    }
                    var s_14 : f32 = s_13 - l_8[i_15 * u32(6) + k_26] * y_7[k_26];
                    k_26 = k_26 + u32(1);
                    s_13 = s_14;
                }
                y_7[i_15] = s_13 / l_8[i_15 * u32(6) + i_15];
                i_15 = i_15 + u32(1);
            }
            var x_26 : array<f32, i32(6)>;
            x_26[i32(0)] = 0.0f;
            x_26[i32(1)] = 0.0f;
            x_26[i32(2)] = 0.0f;
            x_26[i32(3)] = 0.0f;
            x_26[i32(4)] = 0.0f;
            x_26[i32(5)] = 0.0f;
            var ii_2 : u32 = u32(0);
            loop
            {
                if(ii_2 < u32(6))
                {
                }
                else
                {
                    break;
                }
                var i_16 : u32 = u32(5) - ii_2;
                k_26 = i_16 + u32(1);
                sum_10 = y_7[i_16];
                loop
                {
                    if(k_26 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var s_15 : f32 = sum_10 - l_8[k_26 * u32(6) + i_16] * x_26[k_26];
                    k_26 = k_26 + u32(1);
                    sum_10 = s_15;
                }
                x_26[i_16] = sum_10 / l_8[i_16 * u32(6) + i_16];
                ii_2 = ii_2 + u32(1);
            }
            var i_17 : u32 = u32(0);
            loop
            {
                if(i_17 < u32(6))
                {
                }
                else
                {
                    break;
                }
                inv_0[i_17 * u32(6) + j_11] = x_26[i_17];
                i_17 = i_17 + u32(1);
            }
            j_11 = j_11 + u32(1);
        }
    }
    j_11 = u32(0);
    loop
    {
        if(j_11 < u32(9))
        {
        }
        else
        {
            break;
        }
        var _S957 : u32 = u32(4) * j_11;
        scratch_0[sv_0(c_47, u32(12) + j_11)] = vec4<f32>(inv_0[_S957], inv_0[_S957 + u32(1)], inv_0[_S957 + u32(2)], inv_0[_S957 + u32(3)]);
        j_11 = j_11 + u32(1);
    }
    return;
}

fn assemble_block_0( c_48 : u32)
{
    var p_23 : u32;
    var r_19 : u32;
    var a_28 : array<f32, i32(36)>;
    var k_27 : u32 = u32(0);
    loop
    {
        if(k_27 < u32(36))
        {
        }
        else
        {
            break;
        }
        a_28[k_27] = 0.0f;
        k_27 = k_27 + u32(1);
    }
    var e_9 : u32 = index_0[c_48];
    loop
    {
        if(e_9 < index_0[c_48 + u32(1)])
        {
        }
        else
        {
            break;
        }
        var entry_8 : u32 = index_0[e_9];
        var i_18 : u32 = (entry_8 >> (u32(1)));
        var _S958 : bool = ((entry_8 & (u32(1)))) != u32(0);
        var _S959 : u32 = statics_bond_slot_0(i_18);
        var _S960 : vec4<f32> = scratch_0[_S959];
        var _S961 : vec4<f32> = scratch_0[_S959 + u32(1)];
        p_23 = u32(0);
        loop
        {
            if(p_23 < u32(6))
            {
            }
            else
            {
                break;
            }
            var _S962 : u32 = p_23 % u32(3);
            var t_29 : vec3<f32>;
            if(_S962 == u32(0))
            {
                t_29 = bonds_0[i_18].t1_0.xyz;
            }
            else
            {
                if(_S962 == u32(1))
                {
                    t_29 = bonds_0[i_18].t2_0.xyz;
                }
                else
                {
                    t_29 = bonds_0[i_18].normal_0.xyz;
                }
            }
            var _S963 : bool = p_23 < u32(3);
            var row_u_0 : vec3<f32>;
            var row_t_0 : vec3<f32>;
            if(_S963)
            {
                if(_S958)
                {
                    row_u_0 = t_29;
                }
                else
                {
                    row_u_0 = (vec3<f32>(0) - t_29);
                }
                if(_S958)
                {
                    row_t_0 = cross(bonds_0[i_18].rb_0.xyz, t_29);
                }
                else
                {
                    row_t_0 = (vec3<f32>(0) - cross(bonds_0[i_18].ra_0.xyz, t_29));
                }
            }
            else
            {
                var _S964 : vec3<f32> = vec3<f32>(0.0f);
                if(_S958)
                {
                    row_u_0 = t_29;
                }
                else
                {
                    row_u_0 = (vec3<f32>(0) - t_29);
                }
                var _S965 : vec3<f32> = row_u_0;
                row_u_0 = _S964;
                row_t_0 = _S965;
            }
            var kp_0 : f32;
            if(_S963)
            {
                kp_0 = _S960[p_23];
            }
            else
            {
                kp_0 = _S961[p_23 - u32(3)];
            }
            if(kp_0 == 0.0f)
            {
                p_23 = p_23 + u32(1);
                continue;
            }
            var _S966 : array<f32, i32(6)> = array<f32, i32(6)>( row_u_0.x, row_u_0.y, row_u_0.z, row_t_0.x, row_t_0.y, row_t_0.z );
            r_19 = u32(0);
            loop
            {
                if(r_19 < u32(6))
                {
                }
                else
                {
                    break;
                }
                var q_14 : u32 = u32(0);
                loop
                {
                    if(q_14 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    a_28[r_19 * u32(6) + q_14] = a_28[r_19 * u32(6) + q_14] + kp_0 * _S966[r_19] * _S966[q_14];
                    q_14 = q_14 + u32(1);
                }
                r_19 = r_19 + u32(1);
            }
            p_23 = p_23 + u32(1);
        }
        e_9 = e_9 + u32(1);
    }
    var _S967 : u32 = fixed_mask_0(c_48);
    p_23 = u32(0);
    loop
    {
        if(p_23 < u32(6))
        {
        }
        else
        {
            break;
        }
        if(((_S967 & (((u32(1) << (p_23)))))) != u32(0))
        {
            r_19 = u32(0);
            loop
            {
                if(r_19 < u32(6))
                {
                }
                else
                {
                    break;
                }
                a_28[p_23 * u32(6) + r_19] = 0.0f;
                a_28[r_19 * u32(6) + p_23] = 0.0f;
                r_19 = r_19 + u32(1);
            }
            a_28[p_23 * u32(6) + p_23] = 1.0f;
        }
        p_23 = p_23 + u32(1);
    }
    p_23 = u32(0);
    loop
    {
        if(p_23 < u32(6))
        {
        }
        else
        {
            break;
        }
        if((a_28[p_23 * u32(6) + p_23]) == 0.0f)
        {
            a_28[p_23 * u32(6) + p_23] = 1.0f;
        }
        p_23 = p_23 + u32(1);
    }
    store_inverse_0(c_48, a_28);
    return;
}

fn block_get_0( c_49 : u32,  i_19 : u32,  j_12 : u32) -> f32
{
    var k_28 : u32 = i_19 * u32(6) + j_12;
    return scratch_0[sv_0(c_49, u32(12) + k_28 / u32(4))][k_28 % u32(4)];
}

fn precondition_0( c_50 : u32)
{
    var _S968 : array<f32, i32(6)> = array<f32, i32(6)>( scratch_0[sv_0(c_50, u32(4))].x, scratch_0[sv_0(c_50, u32(4))].y, scratch_0[sv_0(c_50, u32(4))].z, scratch_0[sv_0(c_50, u32(5))].x, scratch_0[sv_0(c_50, u32(5))].y, scratch_0[sv_0(c_50, u32(5))].z );
    var z_1 : array<f32, i32(6)>;
    var i_20 : u32 = u32(0);
    loop
    {
        if(i_20 < u32(6))
        {
        }
        else
        {
            break;
        }
        var j_13 : u32 = u32(0);
        var s_16 : f32 = 0.0f;
        loop
        {
            if(j_13 < u32(6))
            {
            }
            else
            {
                break;
            }
            var s_17 : f32 = s_16 + block_get_0(c_50, i_20, j_13) * _S968[j_13];
            j_13 = j_13 + u32(1);
            s_16 = s_17;
        }
        z_1[i_20] = s_16;
        i_20 = i_20 + u32(1);
    }
    scratch_0[sv_0(c_50, u32(6))] = vec4<f32>(z_1[i32(0)], z_1[i32(1)], z_1[i32(2)], 0.0f);
    scratch_0[sv_0(c_50, u32(7))] = vec4<f32>(z_1[i32(3)], z_1[i32(4)], z_1[i32(5)], 0.0f);
    return;
}

fn project_displacement_slot_0( tid_27 : u32,  isl_40 : ptr<function, Island_std430_0>,  slot_5 : u32)
{
    var _S969 : u32;
    var _S970 : vec3<f32> = vec3<f32>(0.0f);
    var p_24 : vec3<f32> = _S970;
    var l_9 : vec3<f32> = _S970;
    var _S971 : vec4<u32> = (*isl_40).range_0;
    var _S972 : u32 = (*isl_40).range_0.x + tid_27;
    var c_51 : u32 = _S972;
    loop
    {
        var _S973 : u32 = _S971.y;
        _S969 = _S973;
        if(c_51 < _S973)
        {
        }
        else
        {
            break;
        }
        var u_6 : vec3<f32> = scratch_0[sv_0(c_51, slot_5)].xyz;
        var th_8 : vec3<f32> = scratch_0[sv_0(c_51, slot_5 + u32(1))].xyz;
        var r_20 : vec3<f32> = chunks_0[c_51].center_0.xyz - (*isl_40).com_0.xyz;
        var _S974 : vec3<f32> = vec3<f32>(chunks_0[c_51].center_0.w);
        p_24 = p_24 + u_6 * _S974;
        l_9 = l_9 + (cross(r_20, u_6) * _S974 + rows_mul_0(chunks_0[c_51].inertia0_1, chunks_0[c_51].inertia1_1, chunks_0[c_51].inertia2_1, th_8));
        c_51 = c_51 + u32(256);
    }
    group_sum3_3(tid_27, &(p_24), &(l_9));
    var _S975 : vec4<f32> = (*isl_40).com_0;
    var _S976 : vec3<f32> = p_24 / vec3<f32>((*isl_40).com_0.w);
    var _S977 : vec3<f32> = rows_mul_0((*isl_40).inv0_0, (*isl_40).inv1_0, (*isl_40).inv2_0, l_9);
    c_51 = _S972;
    loop
    {
        if(c_51 < _S969)
        {
        }
        else
        {
            break;
        }
        var _S978 : u32 = sv_0(c_51, slot_5);
        scratch_0[_S978] = vec4<f32>(scratch_0[_S978].xyz - _S976 - cross(_S977, chunks_0[c_51].center_0.xyz - _S975.xyz), 0.0f);
        var _S979 : u32 = sv_0(c_51, slot_5 + u32(1));
        scratch_0[_S979] = vec4<f32>(scratch_0[_S979].xyz - _S977, 0.0f);
        c_51 = c_51 + u32(256);
    }
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    return;
}

fn statics_result_slot_0( island_3 : u32) -> u32
{
    return params_0.statics_base_0 + u32(23) * params_0.chunk_count_0 + u32(2) * params_0.statics_bonds_0 + island_3;
}

fn write_bond_loads_0( _S980 : u32,  _S981 : u32,  _S982 : vec3<f32>,  _S983 : vec3<f32>,  _S984 : f32)
{
    var _S985 : vec3<f32> = to_body_0(_S981, _S982);
    var _S986 : vec3<f32> = to_body_0(_S981, _S983);
    var _S987 : u32 = u32(3) * _S980;
    scratch_0[_S987] = vec4<f32>(_S985, _S984);
    scratch_0[_S987 + u32(1)] = vec4<f32>(_S986 + cross(bonds_0[_S981].ra_0.xyz, _S985), 0.0f);
    scratch_0[_S987 + u32(2)] = vec4<f32>((vec3<f32>(0) - _S986) + cross(bonds_0[_S981].rb_0.xyz, (vec3<f32>(0) - _S985)), 0.0f);
    return;
}

fn static_kinematics_1( _S988 : u32,  _S989 : ptr<function, vec3<f32>>,  _S990 : ptr<function, vec3<f32>>)
{
    var ca_2 : u32 = bonds_0[_S988].law_0.ids_0.y;
    var cb_6 : u32 = bonds_0[_S988].law_0.ids_0.z;
    var _S991 : u32 = u32(4) * cb_6;
    var _S992 : u32 = u32(4) * ca_2;
    var _S993 : u32 = _S991 + u32(1);
    var _S994 : u32 = _S992 + u32(1);
    var _S995 : u32 = sv_0(cb_6, u32(22));
    var _S996 : u32 = sv_0(ca_2, u32(22));
    var dth_1 : vec3<f32> = state_0[_S993].xyz - state_0[_S994].xyz + (scratch_0[_S995].xyz - scratch_0[_S996].xyz);
    (*_S989) = to_local_0(_S988, state_0[_S991].xyz - state_0[_S992].xyz + (scratch_0[sv_0(cb_6, u32(21))].xyz - scratch_0[sv_0(ca_2, u32(21))].xyz) + (cross(state_0[_S993].xyz + scratch_0[_S995].xyz, bonds_0[_S988].rb_0.xyz) - cross(state_0[_S994].xyz + scratch_0[_S996].xyz, bonds_0[_S988].ra_0.xyz)));
    (*_S990) = to_local_0(_S988, dth_1);
    return;
}

fn bond_kinematics_0( _S997 : u32,  _S998 : vec3<f32>,  _S999 : vec3<f32>,  _S1000 : vec3<f32>,  _S1001 : vec3<f32>,  _S1002 : ptr<function, vec3<f32>>,  _S1003 : ptr<function, vec3<f32>>)
{
    (*_S1002) = to_local_0(_S997, _S1000 + cross(_S1001, bonds_0[_S997].rb_0.xyz) - (_S998 + cross(_S999, bonds_0[_S997].ra_0.xyz)));
    (*_S1003) = to_local_0(_S997, _S1001 - _S999);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn island_statics(@builtin(workgroup_id) group_5 : vec3<u32>, @builtin(local_invocation_id) thread_5 : vec3<u32>)
{
    var i_21 : u32;
    var converged_4 : bool;
    var c_52 : u32;
    var tid_28 : u32 = thread_5.x;
    var _S1004 : u32 = group_5.x;
    var _S1005 : Island_std430_0 = islands_0[_S1004];
    var _S1006 : vec4<u32> = _S1005.info_0;
    var _S1007 : u32 = _S1005.info_0.z;
    if(((_S1007 & (u32(8)))) == u32(0))
    {
        return;
    }
    var free_0 : bool = (((_S1006.x) & (u32(1)))) == u32(0);
    var c0_1 : u32 = _S1005.range_0.x;
    var c1_1 : u32 = _S1005.range_0.y;
    var b0_0 : u32 = _S1005.range_0.z;
    var _S1008 : u32 = _S1005.range_0.w;
    var _S1009 : u32 = c0_1 + tid_28;
    var c_53 : u32 = _S1009;
    loop
    {
        if(c_53 < c1_1)
        {
        }
        else
        {
            break;
        }
        var _S1010 : vec4<f32> = vec4<f32>(0.0f);
        scratch_0[sv_0(c_53, u32(21))] = _S1010;
        scratch_0[sv_0(c_53, u32(22))] = _S1010;
        c_53 = c_53 + u32(256);
    }
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    if(free_0)
    {
        project_load_slot_0(tid_28, &(_S1005), u32(0));
    }
    var _S1011 : f32 = island_dot_0(tid_28, c0_1, c1_1, u32(0), u32(0));
    var _S1012 : f32 = max(sqrt(_S1011), 1.00000000317107685e-30f);
    var _S1013 : f32 = params_0.statics_tol_0;
    var _S1014 : u32 = min(params_0.statics_cg_0, u32(20) * (c1_1 - c0_1) * u32(6) + u32(200));
    var previous_1 : f32 = 1.00000001504746622e+30f;
    var start_0 : f32 = 1.00000001504746622e+30f;
    var residual_0 : f32 = 0.0f;
    var newton_0 : u32 = u32(0);
    var cg_total_0 : u32 = u32(0);
    loop
    {
        if(newton_0 < (params_0.statics_newton_0))
        {
        }
        else
        {
            converged_4 = false;
            break;
        }
        var _S1015 : u32 = b0_0 + tid_28;
        i_21 = _S1015;
        loop
        {
            if(i_21 < _S1008)
            {
            }
            else
            {
                break;
            }
            var resp_3 : JointResponse_0 = static_response_0(i_21);
            write_bond_loads_0(i_21, i_21, resp_3.force_lin_1, resp_3.force_ang_1, 0.0f);
            i_21 = i_21 + u32(256);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        var _S1016 : vec4<f32> = vec4<f32>(0.0f);
        var magnitude_0 : vec4<f32> = _S1016;
        var unused_m_0 : vec4<f32> = _S1016;
        c_53 = _S1009;
        loop
        {
            if(c_53 < c1_1)
            {
            }
            else
            {
                break;
            }
            var fi_4 : vec3<f32>;
            var mi_7 : vec3<f32>;
            gather_loads_0(c_53, &(fi_4), &(mi_7));
            magnitude_0[i32(0)] = magnitude_0[i32(0)] + bond_load_magnitude2_0(c_53);
            var r_lin_0 : vec4<f32> = vec4<f32>(scratch_0[sv_0(c_53, u32(0))].xyz + fi_4, 0.0f);
            var r_ang_0 : vec4<f32> = vec4<f32>(scratch_0[sv_0(c_53, u32(1))].xyz + mi_7, 0.0f);
            hold_0(fixed_mask_0(c_53), &(r_lin_0), &(r_ang_0), _S1016, _S1016);
            scratch_0[sv_0(c_53, u32(4))] = r_lin_0;
            scratch_0[sv_0(c_53, u32(5))] = r_ang_0;
            c_53 = c_53 + u32(256);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        group_sum2_1(tid_28, &(magnitude_0), &(unused_m_0));
        if(free_0)
        {
            project_load_slot_0(tid_28, &(_S1005), u32(4));
        }
        var _S1017 : f32 = island_dot_0(tid_28, c0_1, c1_1, u32(4), u32(4));
        var residual_1 : f32 = sqrt(_S1017) / _S1012;
        var _S1018 : f32 = max(0.00100000004749745f, 3.83999986297567375e-06f * sqrt(magnitude_0.x) / _S1012);
        if(newton_0 == u32(0))
        {
            start_0 = residual_1;
        }
        var _S1019 : bool = is_finite_0(residual_1);
        var _S1020 : bool;
        if(_S1019)
        {
            if(residual_1 <= _S1013)
            {
                converged_4 = true;
            }
            else
            {
                if(residual_1 <= _S1018)
                {
                    converged_4 = residual_1 > (0.5f * previous_1);
                }
                else
                {
                    converged_4 = false;
                }
                if(converged_4)
                {
                    _S1020 = residual_1 <= start_0;
                }
                else
                {
                    _S1020 = false;
                }
                converged_4 = _S1020;
            }
        }
        else
        {
            converged_4 = false;
        }
        if(converged_4)
        {
            residual_0 = residual_1;
            converged_4 = true;
            break;
        }
        if(!_S1019)
        {
            residual_0 = residual_1;
            converged_4 = false;
            break;
        }
        var i_22 : u32 = _S1015;
        loop
        {
            if(i_22 < _S1008)
            {
            }
            else
            {
                break;
            }
            var d_lin_12 : vec3<f32>;
            var d_ang_10 : vec3<f32>;
            static_kinematics_1(i_22, &(d_lin_12), &(d_ang_10));
            var _S1021 : JointBond_std430_0 = bonds_0[i_22].law_0;
            var _S1022 : JointState_std430_0 = bond_dyn_0[i_22].js_0;
            var f_lin_3 : vec3<f32>;
            var f_ang_3 : vec3<f32>;
            secant_factors_1(&(_S1021), &(_S1022), d_lin_12, &(f_lin_3), &(f_ang_3));
            var _S1023 : u32 = statics_bond_slot_0(i_22);
            var _S1024 : f32 = _S1021.stiff0_0.y;
            scratch_0[_S1023] = vec4<f32>(_S1024 * f_lin_3.x, _S1024 * f_lin_3.y, _S1021.stiff0_0.x * f_lin_3.z, 0.0f);
            scratch_0[_S1023 + u32(1)] = vec4<f32>(_S1021.stiff0_0.z * f_ang_3.x, _S1021.stiff0_0.w * f_ang_3.y, _S1021.stiff1_0.x * f_ang_3.z, 0.0f);
            i_22 = i_22 + u32(256);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        var c_54 : u32 = _S1009;
        loop
        {
            if(c_54 < c1_1)
            {
            }
            else
            {
                break;
            }
            assemble_block_0(c_54);
            scratch_0[sv_0(c_54, u32(2))] = _S1016;
            scratch_0[sv_0(c_54, u32(3))] = _S1016;
            c_54 = c_54 + u32(256);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        var c_55 : u32 = _S1009;
        loop
        {
            if(c_55 < c1_1)
            {
            }
            else
            {
                break;
            }
            precondition_0(c_55);
            c_55 = c_55 + u32(256);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        if(free_0)
        {
            project_displacement_slot_0(tid_28, &(_S1005), u32(6));
        }
        var c_56 : u32 = _S1009;
        loop
        {
            if(c_56 < c1_1)
            {
            }
            else
            {
                break;
            }
            scratch_0[sv_0(c_56, u32(8))] = scratch_0[sv_0(c_56, u32(6))];
            scratch_0[sv_0(c_56, u32(9))] = scratch_0[sv_0(c_56, u32(7))];
            c_56 = c_56 + u32(256);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        var _S1025 : f32 = island_dot_0(tid_28, c0_1, c1_1, u32(4), u32(6));
        var _S1026 : f32 = island_dot_0(tid_28, c0_1, c1_1, u32(4), u32(4));
        var _S1027 : f32 = sqrt(_S1026);
        var rz_0 : f32 = _S1025;
        var k_29 : u32 = u32(0);
        var cg_total_1 : u32 = cg_total_0;
        loop
        {
            if(k_29 < _S1014)
            {
                _S1020 = _S1027 > 0.0f;
            }
            else
            {
                _S1020 = false;
            }
            if(_S1020)
            {
            }
            else
            {
                cg_total_0 = cg_total_1;
                break;
            }
            var i_23 : u32 = _S1015;
            loop
            {
                if(i_23 < _S1008)
                {
                }
                else
                {
                    break;
                }
                var ca_3 : u32 = bonds_0[i_23].law_0.ids_0.y;
                var cb_7 : u32 = bonds_0[i_23].law_0.ids_0.z;
                var d_lin_13 : vec3<f32>;
                var d_ang_11 : vec3<f32>;
                bond_kinematics_0(i_23, scratch_0[sv_0(ca_3, u32(8))].xyz, scratch_0[sv_0(ca_3, u32(9))].xyz, scratch_0[sv_0(cb_7, u32(8))].xyz, scratch_0[sv_0(cb_7, u32(9))].xyz, &(d_lin_13), &(d_ang_11));
                var _S1028 : u32 = statics_bond_slot_0(i_23);
                write_bond_loads_0(i_23, i_23, d_lin_13 * scratch_0[_S1028].xyz, d_ang_11 * scratch_0[_S1028 + u32(1)].xyz, 0.0f);
                i_23 = i_23 + u32(256);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
            c_52 = _S1009;
            loop
            {
                if(c_52 < c1_1)
                {
                }
                else
                {
                    break;
                }
                var fi_5 : vec3<f32>;
                var mi_8 : vec3<f32>;
                gather_loads_0(c_52, &(fi_5), &(mi_8));
                var ap_lin_0 : vec4<f32> = vec4<f32>((vec3<f32>(0) - fi_5), 0.0f);
                var ap_ang_0 : vec4<f32> = vec4<f32>((vec3<f32>(0) - mi_8), 0.0f);
                hold_0(fixed_mask_0(c_52), &(ap_lin_0), &(ap_ang_0), scratch_0[sv_0(c_52, u32(8))], scratch_0[sv_0(c_52, u32(9))]);
                scratch_0[sv_0(c_52, u32(10))] = ap_lin_0;
                scratch_0[sv_0(c_52, u32(11))] = ap_ang_0;
                c_52 = c_52 + u32(256);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
            var pap_0 : f32 = island_dot_0(tid_28, c0_1, c1_1, u32(8), u32(10));
            var _S1029 : u32 = cg_total_1 + u32(1);
            if(pap_0 <= 0.0f)
            {
                cg_total_0 = _S1029;
                break;
            }
            var _S1030 : f32 = rz_0 / pap_0;
            var c_57 : u32 = _S1009;
            loop
            {
                if(c_57 < c1_1)
                {
                }
                else
                {
                    break;
                }
                var _S1031 : u32 = sv_0(c_57, u32(2));
                var _S1032 : vec3<f32> = vec3<f32>(_S1030);
                scratch_0[_S1031] = vec4<f32>(scratch_0[_S1031].xyz + _S1032 * scratch_0[sv_0(c_57, u32(8))].xyz, 0.0f);
                var _S1033 : u32 = sv_0(c_57, u32(3));
                scratch_0[_S1033] = vec4<f32>(scratch_0[_S1033].xyz + _S1032 * scratch_0[sv_0(c_57, u32(9))].xyz, 0.0f);
                var _S1034 : u32 = sv_0(c_57, u32(4));
                scratch_0[_S1034] = vec4<f32>(scratch_0[_S1034].xyz - _S1032 * scratch_0[sv_0(c_57, u32(10))].xyz, 0.0f);
                var _S1035 : u32 = sv_0(c_57, u32(5));
                scratch_0[_S1035] = vec4<f32>(scratch_0[_S1035].xyz - _S1032 * scratch_0[sv_0(c_57, u32(11))].xyz, 0.0f);
                c_57 = c_57 + u32(256);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
            var _S1036 : f32 = island_dot_0(tid_28, c0_1, c1_1, u32(4), u32(4));
            if((sqrt(_S1036)) <= (0.00009999999747379f * _S1027))
            {
                cg_total_0 = _S1029;
                break;
            }
            var c_58 : u32 = _S1009;
            loop
            {
                if(c_58 < c1_1)
                {
                }
                else
                {
                    break;
                }
                precondition_0(c_58);
                c_58 = c_58 + u32(256);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
            if(free_0)
            {
                project_displacement_slot_0(tid_28, &(_S1005), u32(6));
            }
            var rz_new_0 : f32 = island_dot_0(tid_28, c0_1, c1_1, u32(4), u32(6));
            var _S1037 : f32 = rz_new_0 / rz_0;
            var c_59 : u32 = _S1009;
            loop
            {
                if(c_59 < c1_1)
                {
                }
                else
                {
                    break;
                }
                var _S1038 : u32 = sv_0(c_59, u32(8));
                var _S1039 : vec3<f32> = vec3<f32>(_S1037);
                scratch_0[_S1038] = vec4<f32>(scratch_0[sv_0(c_59, u32(6))].xyz + _S1039 * scratch_0[_S1038].xyz, 0.0f);
                var _S1040 : u32 = sv_0(c_59, u32(9));
                scratch_0[_S1040] = vec4<f32>(scratch_0[sv_0(c_59, u32(7))].xyz + _S1039 * scratch_0[_S1040].xyz, 0.0f);
                c_59 = c_59 + u32(256);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
            var _S1041 : u32 = k_29 + u32(1);
            rz_0 = rz_new_0;
            k_29 = _S1041;
            cg_total_1 = _S1029;
        }
        c_52 = _S1009;
        loop
        {
            if(c_52 < c1_1)
            {
            }
            else
            {
                break;
            }
            var _S1042 : u32 = u32(4) * c_52;
            var u_7 : vec3<f32> = state_0[_S1042].xyz;
            var _S1043 : u32 = sv_0(c_52, u32(21));
            var u_lo_0 : vec3<f32> = scratch_0[_S1043].xyz;
            var _S1044 : u32 = _S1042 + u32(1);
            var th_9 : vec3<f32> = state_0[_S1044].xyz;
            var _S1045 : u32 = sv_0(c_52, u32(22));
            var th_lo_0 : vec3<f32> = scratch_0[_S1045].xyz;
            comp_add_0(&(u_7), &(u_lo_0), scratch_0[sv_0(c_52, u32(2))].xyz);
            comp_add_0(&(th_9), &(th_lo_0), scratch_0[sv_0(c_52, u32(3))].xyz);
            state_0[_S1042] = vec4<f32>(u_7, state_0[_S1042].w);
            state_0[_S1044] = vec4<f32>(th_9, state_0[_S1044].w);
            scratch_0[_S1043] = vec4<f32>(u_lo_0, 0.0f);
            scratch_0[_S1045] = vec4<f32>(th_lo_0, 0.0f);
            c_52 = c_52 + u32(256);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        var _S1046 : u32 = newton_0 + u32(1);
        previous_1 = residual_1;
        residual_0 = residual_1;
        newton_0 = _S1046;
    }
    i_21 = b0_0 + tid_28;
    loop
    {
        if(i_21 < _S1008)
        {
        }
        else
        {
            break;
        }
        var resp_4 : JointResponse_0 = static_response_0(i_21);
        var _S1047 : JointState_0 = JointState_0( bond_dyn_0[i_21].js_0.damage_0, bond_dyn_0[i_21].js_0.crush_1, bond_dyn_0[i_21].js_0.kappa_0, bond_dyn_0[i_21].js_0.kappa_c_0, bond_dyn_0[i_21].js_0.ductility_0, bond_dyn_0[i_21].js_0.ductility_c_0, bond_dyn_0[i_21].js_0.life_0, bond_dyn_0[i_21].js_0.plastic_x_0, bond_dyn_0[i_21].js_0.plastic_y_0, bond_dyn_0[i_21].js_0.plastic_t_0, bond_dyn_0[i_21].js_0.rebar_plastic_0, bond_dyn_0[i_21].js_0.rebar_slip0_0, bond_dyn_0[i_21].js_0.rebar_slip1_0, bond_dyn_0[i_21].js_0.rebar_work_0, bond_dyn_0[i_21].js_0.rebar_broken_0, bond_dyn_0[i_21].js_0.strain_rate_0, bond_dyn_0[i_21].js_0.governing_stress_0, bond_dyn_0[i_21].js_0.dissipated_0, bond_dyn_0[i_21].js_0.utilization_0, bond_dyn_0[i_21].js_0.mode_0 );
        var bd_1 : BondDyn_0;
        bd_1.js_0 = _S1047;
        bd_1.force_lin_0 = bond_dyn_0[i_21].force_lin_0;
        bd_1.force_ang_0 = bond_dyn_0[i_21].force_ang_0;
        bd_1.sums_0 = bond_dyn_0[i_21].sums_0;
        bd_1.comps_0 = bond_dyn_0[i_21].comps_0;
        bd_1.events_0 = bond_dyn_0[i_21].events_0;
        bd_1.force_lin_0 = vec4<f32>(resp_4.force_lin_1, resp_4.stored_2);
        bd_1.force_ang_0 = vec4<f32>(resp_4.force_ang_1, bd_1.force_ang_0.w);
        bond_dyn_0[i_21].js_0.damage_0 = bd_1.js_0.damage_0;
        bond_dyn_0[i_21].js_0.crush_1 = bd_1.js_0.crush_1;
        bond_dyn_0[i_21].js_0.kappa_0 = bd_1.js_0.kappa_0;
        bond_dyn_0[i_21].js_0.kappa_c_0 = bd_1.js_0.kappa_c_0;
        bond_dyn_0[i_21].js_0.ductility_0 = bd_1.js_0.ductility_0;
        bond_dyn_0[i_21].js_0.ductility_c_0 = bd_1.js_0.ductility_c_0;
        bond_dyn_0[i_21].js_0.life_0 = bd_1.js_0.life_0;
        bond_dyn_0[i_21].js_0.plastic_x_0 = bd_1.js_0.plastic_x_0;
        bond_dyn_0[i_21].js_0.plastic_y_0 = bd_1.js_0.plastic_y_0;
        bond_dyn_0[i_21].js_0.plastic_t_0 = bd_1.js_0.plastic_t_0;
        bond_dyn_0[i_21].js_0.rebar_plastic_0 = bd_1.js_0.rebar_plastic_0;
        bond_dyn_0[i_21].js_0.rebar_slip0_0 = bd_1.js_0.rebar_slip0_0;
        bond_dyn_0[i_21].js_0.rebar_slip1_0 = bd_1.js_0.rebar_slip1_0;
        bond_dyn_0[i_21].js_0.rebar_work_0 = bd_1.js_0.rebar_work_0;
        bond_dyn_0[i_21].js_0.rebar_broken_0 = bd_1.js_0.rebar_broken_0;
        bond_dyn_0[i_21].js_0.strain_rate_0 = bd_1.js_0.strain_rate_0;
        bond_dyn_0[i_21].js_0.governing_stress_0 = bd_1.js_0.governing_stress_0;
        bond_dyn_0[i_21].js_0.dissipated_0 = bd_1.js_0.dissipated_0;
        bond_dyn_0[i_21].js_0.utilization_0 = bd_1.js_0.utilization_0;
        bond_dyn_0[i_21].js_0.mode_0 = bd_1.js_0.mode_0;
        bond_dyn_0[i_21].force_lin_0 = bd_1.force_lin_0;
        bond_dyn_0[i_21].force_ang_0 = bd_1.force_ang_0;
        bond_dyn_0[i_21].sums_0 = bd_1.sums_0;
        bond_dyn_0[i_21].comps_0 = bd_1.comps_0;
        bond_dyn_0[i_21].events_0 = bd_1.events_0;
        write_bond_loads_0(i_21, i_21, resp_4.force_lin_1, resp_4.force_ang_1, max(resp_4.measures_0.tension_0, resp_4.measures_0.compression_0));
        i_21 = i_21 + u32(256);
    }
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    c_53 = _S1009;
    loop
    {
        if(c_53 < c1_1)
        {
        }
        else
        {
            break;
        }
        var fi_6 : vec3<f32>;
        var mi_9 : vec3<f32>;
        gather_loads_0(c_53, &(fi_6), &(mi_9));
        var reaction_2 : vec3<f32>;
        if((fixed_mask_0(c_53)) != u32(0))
        {
            reaction_2 = (vec3<f32>(0) - (scratch_0[sv_0(c_53, u32(0))].xyz + fi_6));
        }
        else
        {
            var _S1048 : u32 = u32(4) * c_53;
            reaction_2 = vec3<f32>(state_0[_S1048 + u32(1)].w, state_0[_S1048 + u32(2)].w, state_0[_S1048 + u32(3)].w);
        }
        var _S1049 : u32 = u32(4) * c_53;
        var _S1050 : u32 = _S1049 + u32(1);
        state_0[_S1050] = vec4<f32>(state_0[_S1050].xyz, reaction_2.x);
        state_0[_S1049 + u32(2)] = vec4<f32>(0.0f, 0.0f, 0.0f, reaction_2.y);
        state_0[_S1049 + u32(3)] = vec4<f32>(0.0f, 0.0f, 0.0f, reaction_2.z);
        c_53 = c_53 + u32(256);
    }
    if(tid_28 == u32(0))
    {
        var _S1051 : f32 = (bitcast<f32>((newton_0)));
        var _S1052 : f32 = (bitcast<f32>((cg_total_0)));
        if(converged_4)
        {
            previous_1 = 1.0f;
        }
        else
        {
            previous_1 = 0.0f;
        }
        scratch_0[statics_result_slot_0(_S1004)] = vec4<f32>(residual_0, _S1051, _S1052, previous_1);
        islands_0[_S1004].info_0[i32(2)] = (_S1007 & (u32(4294967287)));
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn settled_fatigue(@builtin(workgroup_id) group_6 : vec3<u32>, @builtin(local_invocation_id) thread_6 : vec3<u32>)
{
    var tid_29 : u32 = thread_6.x;
    var _S1053 : u32 = group_6.x;
    var isl_41 : Island_std430_0 = islands_0[_S1053];
    var _S1054 : bool;
    if((((islands_0[_S1053].info_0.x) & (u32(16)))) == u32(0))
    {
        _S1054 = true;
    }
    else
    {
        _S1054 = (isl_41.range_0.w) == (isl_41.range_0.z);
    }
    if(_S1054)
    {
        return;
    }
    seen_clear_0(tid_29);
    if(tid_29 == u32(0))
    {
        g_halt_0 = u32(0);
        g_run_0 = u32(0);
    }
    workgroupBarrier();
    var _S1055 : u32 = isl_41.info_0.w;
    var i_24 : u32 = isl_41.range_0.z + tid_29;
    loop
    {
        if(i_24 < (isl_41.range_0.w))
        {
        }
        else
        {
            break;
        }
        var _S1056 : JointState_0 = JointState_0( bond_dyn_0[i_24].js_0.damage_0, bond_dyn_0[i_24].js_0.crush_1, bond_dyn_0[i_24].js_0.kappa_0, bond_dyn_0[i_24].js_0.kappa_c_0, bond_dyn_0[i_24].js_0.ductility_0, bond_dyn_0[i_24].js_0.ductility_c_0, bond_dyn_0[i_24].js_0.life_0, bond_dyn_0[i_24].js_0.plastic_x_0, bond_dyn_0[i_24].js_0.plastic_y_0, bond_dyn_0[i_24].js_0.plastic_t_0, bond_dyn_0[i_24].js_0.rebar_plastic_0, bond_dyn_0[i_24].js_0.rebar_slip0_0, bond_dyn_0[i_24].js_0.rebar_slip1_0, bond_dyn_0[i_24].js_0.rebar_work_0, bond_dyn_0[i_24].js_0.rebar_broken_0, bond_dyn_0[i_24].js_0.strain_rate_0, bond_dyn_0[i_24].js_0.governing_stress_0, bond_dyn_0[i_24].js_0.dissipated_0, bond_dyn_0[i_24].js_0.utilization_0, bond_dyn_0[i_24].js_0.mode_0 );
        var bd_2 : BondDyn_0;
        bd_2.js_0 = _S1056;
        bd_2.force_lin_0 = bond_dyn_0[i_24].force_lin_0;
        bd_2.force_ang_0 = bond_dyn_0[i_24].force_ang_0;
        bd_2.sums_0 = bond_dyn_0[i_24].sums_0;
        bd_2.comps_0 = bond_dyn_0[i_24].comps_0;
        bd_2.events_0 = bond_dyn_0[i_24].events_0;
        var _S1057 : JointBond_std430_0 = bonds_0[i_24].law_0;
        var _S1058 : u32 = u32(4) * _S1057.ids_0.y;
        var _S1059 : u32 = u32(4) * _S1057.ids_0.z;
        var d_lin_14 : vec3<f32>;
        var d_ang_12 : vec3<f32>;
        bond_kinematics_0(i_24, state_0[_S1058].xyz, state_0[_S1058 + u32(1)].xyz, state_0[_S1059].xyz, state_0[_S1059 + u32(1)].xyz, &(d_lin_14), &(d_ang_12));
        var _S1060 : JointMaterial_std140_0 = materials_0.m_0.data_0[_S1057.ids_0.x];
        var previous_2 : JointState_0 = bd_2.js_0;
        var _S1061 : JointResponse_0 = joint_evaluate_2(&(_S1060), &(_S1057), bd_2.js_0, d_lin_14, d_ang_12, params_0.dt_0, (params_0.fracture_0) != u32(0));
        if((_S1061.state_1.damage_0) > (bd_2.js_0.damage_0 + 9.99999971718068537e-10f))
        {
            _S1054 = true;
        }
        else
        {
            _S1054 = (_S1061.state_1.crush_1) > (previous_2.crush_1 + 9.99999971718068537e-10f);
        }
        var flags_5 : u32;
        if(_S1054)
        {
            flags_5 = u32(16);
        }
        else
        {
            flags_5 = u32(0);
        }
        var _S1062 : f32 = bd_2.sums_0[i32(0)];
        var _S1063 : f32 = bd_2.comps_0[i32(0)];
        comp_add1_0(&(_S1062), &(_S1063), _S1061.dissipated_2);
        bd_2.sums_0[i32(0)] = _S1062;
        bd_2.comps_0[i32(0)] = _S1063;
        var _S1064 : f32 = bd_2.sums_0[i32(1)];
        var _S1065 : f32 = bd_2.comps_0[i32(1)];
        comp_add1_0(&(_S1064), &(_S1065), _S1061.overshoot_0);
        bd_2.sums_0[i32(1)] = _S1064;
        bd_2.comps_0[i32(1)] = _S1065;
        bd_2.force_lin_0 = vec4<f32>(_S1061.force_lin_1, _S1061.stored_2);
        bd_2.force_ang_0 = vec4<f32>(_S1061.force_ang_1, max(bd_2.force_ang_0.w, _S1061.state_1.utilization_0));
        var _S1066 : bool;
        if(!is_damaged_1(previous_2))
        {
            _S1066 = is_damaged_1(_S1061.state_1);
        }
        else
        {
            _S1066 = false;
        }
        var _S1067 : bool;
        if(_S1066)
        {
            _S1067 = (bd_2.events_0.x) == u32(0);
        }
        else
        {
            _S1067 = false;
        }
        if(_S1067)
        {
            bd_2.events_0[i32(0)] = _S1055;
            bd_2.events_0[i32(3)] = _S1061.state_1.mode_0;
            g_seen_0[u32(3)] = u32(1);
        }
        var _S1068 : bool;
        if((bd_2.events_0.y) == u32(0))
        {
            _S1068 = (_S1061.state_1.life_0) != (previous_2.life_0);
        }
        else
        {
            _S1068 = false;
        }
        var _S1069 : bool;
        if(_S1068)
        {
            _S1069 = (_S1061.fatigue_before_0) > 0.99000000953674316f;
        }
        else
        {
            _S1069 = false;
        }
        var _S1070 : bool;
        if(_S1069)
        {
            var _S1071 : f32 = fatigue_factor_2(&(_S1060), _S1061.state_1.life_0);
            _S1070 = _S1071 <= 0.99000000953674316f;
        }
        else
        {
            _S1070 = false;
        }
        if(_S1070)
        {
            bd_2.events_0[i32(1)] = _S1055;
            g_seen_0[u32(4)] = u32(1);
        }
        var flags_6 : u32;
        if(_S1061.disconnected_0)
        {
            bd_2.events_0[i32(2)] = _S1055;
            flags_6 = (flags_5 | (u32(32)));
        }
        else
        {
            flags_6 = flags_5;
        }
        bd_2.js_0 = _S1061.state_1;
        bond_dyn_0[i_24].js_0.damage_0 = bd_2.js_0.damage_0;
        bond_dyn_0[i_24].js_0.crush_1 = bd_2.js_0.crush_1;
        bond_dyn_0[i_24].js_0.kappa_0 = bd_2.js_0.kappa_0;
        bond_dyn_0[i_24].js_0.kappa_c_0 = bd_2.js_0.kappa_c_0;
        bond_dyn_0[i_24].js_0.ductility_0 = bd_2.js_0.ductility_0;
        bond_dyn_0[i_24].js_0.ductility_c_0 = bd_2.js_0.ductility_c_0;
        bond_dyn_0[i_24].js_0.life_0 = bd_2.js_0.life_0;
        bond_dyn_0[i_24].js_0.plastic_x_0 = bd_2.js_0.plastic_x_0;
        bond_dyn_0[i_24].js_0.plastic_y_0 = bd_2.js_0.plastic_y_0;
        bond_dyn_0[i_24].js_0.plastic_t_0 = bd_2.js_0.plastic_t_0;
        bond_dyn_0[i_24].js_0.rebar_plastic_0 = bd_2.js_0.rebar_plastic_0;
        bond_dyn_0[i_24].js_0.rebar_slip0_0 = bd_2.js_0.rebar_slip0_0;
        bond_dyn_0[i_24].js_0.rebar_slip1_0 = bd_2.js_0.rebar_slip1_0;
        bond_dyn_0[i_24].js_0.rebar_work_0 = bd_2.js_0.rebar_work_0;
        bond_dyn_0[i_24].js_0.rebar_broken_0 = bd_2.js_0.rebar_broken_0;
        bond_dyn_0[i_24].js_0.strain_rate_0 = bd_2.js_0.strain_rate_0;
        bond_dyn_0[i_24].js_0.governing_stress_0 = bd_2.js_0.governing_stress_0;
        bond_dyn_0[i_24].js_0.dissipated_0 = bd_2.js_0.dissipated_0;
        bond_dyn_0[i_24].js_0.utilization_0 = bd_2.js_0.utilization_0;
        bond_dyn_0[i_24].js_0.mode_0 = bd_2.js_0.mode_0;
        bond_dyn_0[i_24].force_lin_0 = bd_2.force_lin_0;
        bond_dyn_0[i_24].force_ang_0 = bd_2.force_ang_0;
        bond_dyn_0[i_24].sums_0 = bd_2.sums_0;
        bond_dyn_0[i_24].comps_0 = bd_2.comps_0;
        bond_dyn_0[i_24].events_0 = bd_2.events_0;
        write_bond_loads_0(i_24, i_24, _S1061.force_lin_1, _S1061.force_ang_1, max(_S1061.measures_0.tension_0, _S1061.measures_0.compression_0));
        if(((flags_6 & (u32(16)))) != u32(0))
        {
            g_halt_0 = u32(1);
        }
        if(((flags_6 & (u32(32)))) != u32(0))
        {
            g_run_0 = u32(1);
        }
        i_24 = i_24 + u32(256);
    }
    workgroupBarrier();
    if(tid_29 != u32(0))
    {
        return;
    }
    if(((g_halt_0 | (g_run_0))) != u32(0))
    {
        var _S1072 : u32 = isl_41.info_0.z;
        if(g_halt_0 != u32(0))
        {
            i_24 = u32(16);
        }
        else
        {
            i_24 = u32(0);
        }
        var _S1073 : u32 = (_S1072 | (i_24));
        if(g_run_0 != u32(0))
        {
            i_24 = u32(32);
        }
        else
        {
            i_24 = u32(0);
        }
        islands_0[_S1053].info_0[i32(2)] = (_S1073 | (i_24));
    }
    var _S1074 : u32 = ((((isl_41.coverage_0) | (u32(16384)))) | ((seen_bits_0())));
    if(g_run_0 != u32(0))
    {
        i_24 = u32(128);
    }
    else
    {
        i_24 = u32(0);
    }
    islands_0[_S1053].coverage_0 = (_S1074 | (i_24));
    return;
}

fn impactor_integrate_at_0( ii_3 : u32,  tid_30 : u32,  fused_4 : bool)
{
    var p_25 : vec3<f32>;
    if(ii_3 >= (params_0.impactor_count_0))
    {
        return;
    }
    var _S1075 : u32 = params_0.halt_index_0;
    var imp_7 : Impactor_0;
    imp_7.position_1 = impactors_0[ii_3].position_1;
    imp_7.position_err_1 = impactors_0[ii_3].position_err_1;
    imp_7.velocity_1 = impactors_0[ii_3].velocity_1;
    imp_7.velocity_err_1 = impactors_0[ii_3].velocity_err_1;
    imp_7.angular_velocity_1 = impactors_0[ii_3].angular_velocity_1;
    imp_7.rotation_1 = impactors_0[ii_3].rotation_1;
    imp_7.inertia0_2 = impactors_0[ii_3].inertia0_2;
    imp_7.inertia1_2 = impactors_0[ii_3].inertia1_2;
    imp_7.inertia2_2 = impactors_0[ii_3].inertia2_2;
    imp_7.inv0_2 = impactors_0[ii_3].inv0_2;
    imp_7.inv1_2 = impactors_0[ii_3].inv1_2;
    imp_7.inv2_2 = impactors_0[ii_3].inv2_2;
    imp_7.shape_0 = impactors_0[ii_3].shape_0;
    imp_7.half_1 = impactors_0[ii_3].half_1;
    imp_7.mat_0 = impactors_0[ii_3].mat_0;
    imp_7.crush_0 = impactors_0[ii_3].crush_0;
    imp_7.geom_0 = impactors_0[ii_3].geom_0;
    imp_7.rotation_err_1 = impactors_0[ii_3].rotation_err_1;
    imp_7.ledger_0 = impactors_0[ii_3].ledger_0;
    imp_7.cand_0 = impactors_0[ii_3].cand_0;
    imp_7.momentum_1 = impactors_0[ii_3].momentum_1;
    imp_7.momentum_err_1 = impactors_0[ii_3].momentum_err_1;
    var _S1076 : vec4<u32> = islands_0[params_0.halt_index_0].info_0;
    var stop_0 : bool;
    if((((islands_0[params_0.halt_index_0].info_0.z) & (u32(1)))) != u32(0))
    {
        stop_0 = true;
    }
    else
    {
        var _S1077 : u32 = _S1076.y;
        if(_S1077 != u32(0))
        {
            stop_0 = (imp_7.cand_0.w) >= _S1077;
        }
        else
        {
            stop_0 = false;
        }
    }
    if(fused_4)
    {
        if(stop_0)
        {
            stop_0 = true;
        }
        else
        {
            var _S1078 : u32 = islands_0[params_0.halt_index_0].range_0.x;
            if(_S1078 != u32(0))
            {
                stop_0 = _S1078 <= (imp_7.cand_0.w + u32(1));
            }
            else
            {
                stop_0 = false;
            }
        }
        if(tid_30 == u32(0))
        {
            stop_seen_0(_S1075, imp_7.cand_0.w + u32(1));
        }
    }
    if(stop_0)
    {
        return;
    }
    if((imp_7.cand_0.z) != u32(0))
    {
        if(tid_30 == u32(0))
        {
            impactors_0[ii_3].cand_0[i32(3)] = imp_7.cand_0.w + u32(1);
        }
        return;
    }
    var _S1079 : vec4<f32> = vec4<f32>(0.0f);
    var rf_1 : vec4<f32> = _S1079;
    var rt_0 : vec4<f32> = _S1079;
    var k_30 : u32 = imp_7.cand_0.x + tid_30;
    loop
    {
        if(k_30 < (imp_7.cand_0.y))
        {
        }
        else
        {
            break;
        }
        var _S1080 : u32 = u32(3) * k_30;
        rf_1 = rf_1 + scratch_0[params_0.cand_base_0 + _S1080 + u32(1)];
        rt_0 = rt_0 + scratch_0[params_0.cand_base_0 + _S1080 + u32(2)];
        k_30 = k_30 + u32(256);
    }
    group_sum2_3(tid_30, &(rf_1), &(rt_0));
    if(tid_30 != u32(0))
    {
        return;
    }
    var dt_21 : f32 = params_0.dt_0;
    var _S1081 : vec3<f32> = vec3<f32>(0.0f);
    var load_f_0 : vec3<f32>;
    var load_t_0 : vec3<f32>;
    if((params_0.has_ground_0) != u32(0))
    {
        var ib_0 : Box_0 = impactor_box_0(imp_7, _S1081, imp_7.half_1.xyz);
        var _S1082 : vec3<f32> = imp_7.velocity_1.xyz + imp_7.velocity_err_1.xyz;
        const _S1083 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 1.0f);
        var kc_2 : f32 = contact_stiffness_0(params_0.ground_modulus_0, ib_0, imp_7.mat_0.x, ib_0, _S1083);
        var _S1084 : f32 = imp_7.position_1.z - params_0.ground_hi_0 + (imp_7.position_err_1.z - params_0.ground_lo_0);
        var total_points_0 : u32;
        if((imp_7.shape_0.x) == 0.0f)
        {
            total_points_0 = u32(1);
        }
        else
        {
            total_points_0 = u32(14);
        }
        var _S1085 : f32 = kc_2 / f32(min(total_points_0, u32(5)));
        var s_18 : u32 = u32(0);
        var below_0 : u32 = u32(0);
        loop
        {
            if(s_18 < total_points_0)
            {
            }
            else
            {
                break;
            }
            if((imp_7.shape_0.x) == 0.0f)
            {
                p_25 = vec3<f32>(0.0f, 0.0f, - imp_7.shape_0.y);
            }
            else
            {
                p_25 = sample_point_0(ib_0, s_18);
            }
            if((_S1084 + p_25.z) < 0.0f)
            {
                below_0 = below_0 + u32(1);
            }
            s_18 = s_18 + u32(1);
        }
        if(below_0 > u32(0))
        {
            sentinel_mark_0(u32(8));
        }
        s_18 = u32(0);
        load_f_0 = _S1081;
        load_t_0 = _S1081;
        loop
        {
            if(s_18 < total_points_0)
            {
            }
            else
            {
                break;
            }
            if((imp_7.shape_0.x) == 0.0f)
            {
                p_25 = vec3<f32>(0.0f, 0.0f, - imp_7.shape_0.y);
            }
            else
            {
                p_25 = sample_point_0(ib_0, s_18);
            }
            var depth_8 : f32 = - (_S1084 + p_25.z);
            if(depth_8 <= 0.0f)
            {
                s_18 = s_18 + u32(1);
                continue;
            }
            var stored_16 : f32;
            var diss_8 : f32;
            var f_29 : vec3<f32> = penalty_force_1(_S1085, imp_7.mat_0.z, params_0.ground_friction_0, depth_8, _S1083, _S1082 + cross(imp_7.angular_velocity_1.xyz, p_25), dt_21, below_0, &(stored_16), &(diss_8));
            var load_f_1 : vec3<f32> = load_f_0 + f_29;
            var load_t_1 : vec3<f32> = load_t_0 + cross(p_25, f_29);
            var _S1086 : f32 = imp_7.ledger_0[i32(0)];
            var _S1087 : f32 = imp_7.ledger_0[i32(1)];
            comp_add1_4(&(_S1086), &(_S1087), diss_8);
            imp_7.ledger_0[i32(0)] = _S1086;
            imp_7.ledger_0[i32(1)] = _S1087;
            load_f_0 = load_f_1;
            load_t_0 = load_t_1;
            s_18 = s_18 + u32(1);
        }
    }
    else
    {
        load_f_0 = _S1081;
        load_t_0 = _S1081;
    }
    var load_f_2 : vec3<f32> = rf_1.xyz + load_f_0;
    var load_t_2 : vec3<f32> = rt_0.xyz + load_t_0;
    var m_10 : f32 = imp_7.mat_0.z;
    var vel_1 : vec3<f32> = imp_7.velocity_1.xyz;
    var vel_err_1 : vec3<f32> = imp_7.velocity_err_1.xyz;
    var _S1088 : vec3<f32> = vec3<f32>(dt_21);
    comp_add_1(&(vel_1), &(vel_err_1), (load_f_2 / vec3<f32>(m_10) + params_0.gravity_0.xyz) * _S1088);
    var _S1089 : Quat_0 = quat_of_0(imp_7.rotation_1);
    var q_15 : Quat_0 = _S1089;
    var q_err_0 : vec4<f32> = imp_7.rotation_err_1;
    var l_hi_0 : vec3<f32> = imp_7.momentum_1.xyz;
    var l_err_1 : vec3<f32> = imp_7.momentum_err_1.xyz;
    comp_add_1(&(l_hi_0), &(l_err_1), load_t_2 * _S1088);
    var l_10 : vec3<f32> = l_hi_0 + l_err_1;
    var w_mid_3 : vec3<f32> = world_mul_0(_S1089, imp_7.inv0_2, imp_7.inv1_2, imp_7.inv2_2, l_10);
    var pos_1 : vec3<f32> = imp_7.position_1.xyz;
    var pos_err_1 : vec3<f32> = imp_7.position_err_1.xyz;
    comp_add_1(&(pos_1), &(pos_err_1), (vel_1 + vel_err_1) * _S1088);
    turn_left_1(&(q_15), &(q_err_0), w_mid_3, dt_21);
    imp_7.angular_velocity_1 = vec4<f32>(world_mul_0(q_15, imp_7.inv0_2, imp_7.inv1_2, imp_7.inv2_2, l_10), 0.0f);
    imp_7.rotation_1 = quat_vec_0(q_15);
    imp_7.rotation_err_1 = q_err_0;
    imp_7.momentum_1 = vec4<f32>(l_hi_0, 0.0f);
    imp_7.momentum_err_1 = vec4<f32>(l_err_1, 0.0f);
    imp_7.position_1 = vec4<f32>(pos_1, 0.0f);
    imp_7.position_err_1 = vec4<f32>(pos_err_1, 0.0f);
    imp_7.velocity_1 = vec4<f32>(vel_1, 0.0f);
    imp_7.velocity_err_1 = vec4<f32>(vel_err_1, 0.0f);
    imp_7.cand_0[i32(3)] = imp_7.cand_0[i32(3)] + u32(1);
    imp_7.geom_0 = vec4<f32>(1.0f, imp_7.crush_0.w, 0.0f, 0.0f);
    impactors_0[ii_3].position_1 = imp_7.position_1;
    impactors_0[ii_3].position_err_1 = imp_7.position_err_1;
    impactors_0[ii_3].velocity_1 = imp_7.velocity_1;
    impactors_0[ii_3].velocity_err_1 = imp_7.velocity_err_1;
    impactors_0[ii_3].angular_velocity_1 = imp_7.angular_velocity_1;
    impactors_0[ii_3].rotation_1 = imp_7.rotation_1;
    impactors_0[ii_3].inertia0_2 = imp_7.inertia0_2;
    impactors_0[ii_3].inertia1_2 = imp_7.inertia1_2;
    impactors_0[ii_3].inertia2_2 = imp_7.inertia2_2;
    impactors_0[ii_3].inv0_2 = imp_7.inv0_2;
    impactors_0[ii_3].inv1_2 = imp_7.inv1_2;
    impactors_0[ii_3].inv2_2 = imp_7.inv2_2;
    impactors_0[ii_3].shape_0 = imp_7.shape_0;
    impactors_0[ii_3].half_1 = imp_7.half_1;
    impactors_0[ii_3].mat_0 = imp_7.mat_0;
    impactors_0[ii_3].crush_0 = imp_7.crush_0;
    impactors_0[ii_3].geom_0 = imp_7.geom_0;
    impactors_0[ii_3].rotation_err_1 = imp_7.rotation_err_1;
    impactors_0[ii_3].ledger_0 = imp_7.ledger_0;
    impactors_0[ii_3].cand_0 = imp_7.cand_0;
    impactors_0[ii_3].momentum_1 = imp_7.momentum_1;
    impactors_0[ii_3].momentum_err_1 = imp_7.momentum_err_1;
    var k_31 : u32 = imp_7.cand_0.w - u32(1) - params_0.step_start_0;
    if(k_31 < (params_0.record_stride_0))
    {
        var at_9 : u32 = params_0.record_base_0 + u32(2) * (ii_3 * params_0.record_stride_0 + k_31);
        scratch_0[at_9] = vec4<f32>(vel_1 + vel_err_1, 0.0f);
        scratch_0[at_9 + u32(1)] = vec4<f32>(pos_1 + pos_err_1, 0.0f);
    }
    if(fused_4)
    {
        impactor_travel_check_0(ii_3, true);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn impactor_integrate(@builtin(workgroup_id) group_7 : vec3<u32>, @builtin(local_invocation_id) thread_7 : vec3<u32>)
{
    impactor_integrate_at_0(group_7.x, thread_7.x, false);
    return;
}

fn contact_sum_at_0( g_5 : u32)
{
    var _S1090 : bool;
    if(g_5 >= (params_0.seg_count_0))
    {
        _S1090 = true;
    }
    else
    {
        _S1090 = stopped_0();
    }
    if(_S1090)
    {
        return;
    }
    contact_segment_0(g_5);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn contact_sums(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    contact_sum_at_0(id_2.x);
    return;
}

fn net_load_2( c_60 : u32,  isl_42 : ptr<function, Island_std430_0>,  rg_26 : Rigid_0,  k_32 : u32,  dt_22 : f32,  contact_7 : bool,  f_30 : ptr<function, vec3<f32>>,  t_30 : ptr<function, vec3<f32>>)
{
    var fl_2 : vec3<f32>;
    var tl_2 : vec3<f32>;
    var rep_f_4 : vec3<f32>;
    var rep_m_4 : vec3<f32>;
    chunk_loads_1(c_60, c_60, rg_26.rot_0, k_32, dt_22, contact_7, 0.0f, &(fl_2), &(tl_2), &(rep_f_4), &(rep_m_4));
    var fc_6 : vec3<f32> = fl_2 + params_0.gravity_0.xyz * vec3<f32>(chunks_0[c_60].center_0.w);
    var _S1091 : vec3<f32> = chunks_0[c_60].center_0.xyz;
    var _S1092 : vec3<f32> = (*isl_42).com_0.xyz;
    var r_21 : vec3<f32> = rotate_0(rg_26.rot_0, _S1091 + state_0[u32(4) * c_60].xyz - _S1092);
    (*f_30) = (*f_30) + fc_6;
    (*t_30) = (*t_30) + (cross(r_21, fc_6) + tl_2);
    if((chunks_0[c_60].load_range_0.y) > (chunks_0[c_60].load_range_0.x))
    {
        var fw_4 : vec3<f32> = rotate_0(rg_26.rot_0, rep_f_4);
        (*f_30) = (*f_30) + fw_4;
        (*t_30) = (*t_30) + (cross(rotate_0(rg_26.rot_0, _S1091 - _S1092), fw_4) + rotate_0(rg_26.rot_0, rep_m_4));
    }
    return;
}

fn chunk_update_2( c_61 : u32,  isl_43 : ptr<function, Island_std430_0>,  rg_27 : Rigid_0,  dt_23 : f32,  rml_2 : bool,  step_2 : u32,  contact_8 : bool,  work_10 : ptr<function, f32>,  work_err_6 : ptr<function, f32>)
{
    var _S1093 : vec3<f32> = vec3<f32>(0.0f);
    var _S1094 : u32 = index_0[c_61];
    var peak_2 : f32 = 0.0f;
    var e_10 : u32 = _S1094;
    var fi_7 : vec3<f32> = _S1093;
    var mi_10 : vec3<f32> = _S1093;
    loop
    {
        if(e_10 < index_0[c_61 + u32(1)])
        {
        }
        else
        {
            break;
        }
        var entry_9 : u32 = index_0[e_10];
        var _S1095 : u32 = u32(3) * ((entry_9 >> (u32(1))));
        var fa_3 : vec4<f32> = scratch_0[_S1095];
        if(((entry_9 & (u32(1)))) == u32(0))
        {
            var mi_11 : vec3<f32> = mi_10 + scratch_0[_S1095 + u32(1)].xyz;
            fi_7 = fi_7 + fa_3.xyz;
            mi_10 = mi_11;
        }
        else
        {
            var mi_12 : vec3<f32> = mi_10 + scratch_0[_S1095 + u32(2)].xyz;
            fi_7 = fi_7 + (vec3<f32>(0) - fa_3.xyz);
            mi_10 = mi_12;
        }
        var _S1096 : f32 = max(peak_2, fa_3.w);
        var _S1097 : u32 = e_10 + u32(1);
        peak_2 = _S1096;
        e_10 = _S1097;
    }
    var _S1098 : u32 = u32(4) * c_61;
    var u_8 : vec3<f32> = state_0[_S1098].xyz;
    var _S1099 : u32 = _S1098 + u32(1);
    var th_10 : vec3<f32> = state_0[_S1099].xyz;
    var _S1100 : u32 = _S1098 + u32(2);
    var v_16 : vec3<f32> = state_0[_S1100].xyz;
    var _S1101 : u32 = _S1098 + u32(3);
    var w_12 : vec3<f32> = state_0[_S1101].xyz;
    var mass_2 : f32 = chunks_0[c_61].center_0.w;
    var _S1102 : vec3<f32> = chunks_0[c_61].center_0.xyz;
    var _S1103 : vec3<f32> = (*isl_43).com_0.xyz;
    var r_world_2 : vec3<f32> = rotate_0(rg_27.rot_0, _S1102 + u_8 - _S1103);
    var f_load_2 : vec3<f32>;
    var t_load_2 : vec3<f32>;
    var rep_f_5 : vec3<f32>;
    var rep_m_5 : vec3<f32>;
    chunk_loads_1(c_61, c_61, rg_27.rot_0, step_2, dt_23, contact_8, dt_23, &(f_load_2), &(t_load_2), &(rep_f_5), &(rep_m_5));
    record_chunk_load_0(c_61, f_load_2, t_load_2);
    var _S1104 : vec3<f32> = vec3<f32>(mass_2);
    var f_world_4 : vec3<f32> = f_load_2 + params_0.gravity_0.xyz * _S1104;
    var t_world_6 : vec3<f32> = t_load_2;
    var f_world_5 : vec3<f32>;
    var t_world_7 : vec3<f32>;
    if(rml_2)
    {
        var t_world_8 : vec3<f32> = t_world_6 - (world_mul_0(rg_27.rot_0, chunks_0[c_61].inertia0_1, chunks_0[c_61].inertia1_1, chunks_0[c_61].inertia2_1, rg_27.alpha_0) + cross(rg_27.w_1, world_mul_0(rg_27.rot_0, chunks_0[c_61].inertia0_1, chunks_0[c_61].inertia1_1, chunks_0[c_61].inertia2_1, rg_27.w_1)));
        f_world_5 = f_world_4 - (rg_27.a_0 + cross(rg_27.alpha_0, r_world_2) + cross(rg_27.w_1, cross(rg_27.w_1, r_world_2))) * _S1104;
        t_world_7 = t_world_8;
    }
    else
    {
        f_world_5 = f_world_4;
        t_world_7 = t_world_6;
    }
    var f_ext_4 : vec3<f32> = inverse_rotate_0(rg_27.rot_0, f_world_5);
    var m_ext_6 : vec3<f32> = inverse_rotate_0(rg_27.rot_0, t_world_7);
    var f_ext_5 : vec3<f32>;
    var m_ext_7 : vec3<f32>;
    if(rml_2)
    {
        var wb_3 : vec3<f32> = inverse_rotate_0(rg_27.rot_0, rg_27.w_1);
        var i_w_2 : vec3<f32> = rows_mul_0(chunks_0[c_61].inertia0_1, chunks_0[c_61].inertia1_1, chunks_0[c_61].inertia2_1, w_12);
        var m_ext_8 : vec3<f32> = m_ext_6 - (cross(wb_3, i_w_2) + cross(w_12, rows_mul_0(chunks_0[c_61].inertia0_1, chunks_0[c_61].inertia1_1, chunks_0[c_61].inertia2_1, wb_3)) + cross(w_12, i_w_2));
        f_ext_5 = f_ext_4 - cross(wb_3, v_16) * vec3<f32>((2.0f * mass_2));
        m_ext_7 = m_ext_8;
    }
    else
    {
        f_ext_5 = f_ext_4;
        m_ext_7 = m_ext_6;
    }
    var f_31 : vec3<f32> = f_ext_5 + rep_f_5 + fi_7;
    var m_11 : vec3<f32> = m_ext_7 + rep_m_5 + mi_10;
    var support_3 : u32 = chunks_0[c_61].info_1.x;
    var _S1105 : vec3<f32> = vec3<f32>(state_0[_S1099].w, state_0[_S1100].w, state_0[_S1101].w);
    var reaction_3 : vec3<f32>;
    var u_9 : vec3<f32>;
    var th_11 : vec3<f32>;
    var v_17 : vec3<f32>;
    var w_13 : vec3<f32>;
    if(support_3 == u32(1))
    {
        reaction_3 = (vec3<f32>(0) - f_31);
        u_9 = u_8;
        th_11 = th_10;
        v_17 = _S1093;
        w_13 = _S1093;
    }
    else
    {
        var _S1106 : vec4<f32> = chunks_0[c_61].scale_0;
        var w_14 : vec3<f32> = w_12 + rows_mul_0(chunks_0[c_61].inv0_1, chunks_0[c_61].inv1_1, chunks_0[c_61].inv2_1, m_11) * vec3<f32>((dt_23 * chunks_0[c_61].scale_0.z));
        var _S1107 : vec3<f32> = vec3<f32>(dt_23);
        var th_12 : vec3<f32> = th_10 + w_14 * _S1107;
        if(support_3 == u32(2))
        {
            reaction_3 = (vec3<f32>(0) - f_31);
            u_9 = u_8;
            th_11 = _S1093;
        }
        else
        {
            var v_18 : vec3<f32> = v_16 + f_31 * vec3<f32>((dt_23 * _S1106.y));
            var u_10 : vec3<f32> = u_8 + v_18 * _S1107;
            reaction_3 = _S1105;
            u_9 = u_10;
            th_11 = v_18;
        }
        var _S1108 : vec3<f32> = th_11;
        th_11 = th_12;
        v_17 = _S1108;
        w_13 = w_14;
    }
    state_0[_S1098] = vec4<f32>(u_9, peak_2);
    state_0[_S1099] = vec4<f32>(th_11, reaction_3.x);
    state_0[_S1100] = vec4<f32>(v_17, reaction_3.y);
    state_0[_S1101] = vec4<f32>(w_13, reaction_3.z);
    comp_add1_0(&((*work_10)), &((*work_err_6)), (dot(f_load_2, rg_27.vel_0 + rg_27.vel_err_0 + cross(rg_27.w_1, rotate_0(rg_27.rot_0, _S1102 + u_9 - _S1103)) + rotate_0(rg_27.rot_0, v_17)) + dot(t_load_2, rg_27.w_1 + rotate_0(rg_27.rot_0, w_13))) * dt_23);
    return;
}

fn drift_moments_2( c_62 : u32,  tu_8 : ptr<function, vec3<f32>>,  pv_8 : ptr<function, vec3<f32>>)
{
    var _S1109 : u32 = u32(4) * c_62;
    var _S1110 : vec3<f32> = vec3<f32>((chunks_0[c_62].center_0.w * chunks_0[c_62].scale_0.x));
    (*tu_8) = (*tu_8) + state_0[_S1109].xyz * _S1110;
    (*pv_8) = (*pv_8) + state_0[_S1109 + u32(2)].xyz * _S1110;
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn island_frame(@builtin(workgroup_id) group_8 : vec3<u32>, @builtin(local_invocation_id) thread_8 : vec3<u32>)
{
    island_frame_at_3(group_8.x, thread_8.x);
    return;
}

@compute
@workgroup_size(512, 1, 1)
fn island_frame_bonds_wide(@builtin(workgroup_id) group_9 : vec3<u32>, @builtin(local_invocation_id) thread_9 : vec3<u32>)
{
    island_frame_at_2(group_9.x, thread_9.x);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn contact_islands(@builtin(workgroup_id) group_10 : vec3<u32>, @builtin(local_invocation_id) thread_10 : vec3<u32>)
{
    island_frame_at_1(group_10.x, thread_10.x);
    return;
}

fn wide_chunks_at_0( gx_0 : u32,  tid_31 : u32)
{
    var wg_6 : WideGroup_0 = wide_group_0(params_0.wide_chunk_table_0, gx_0);
    var _S1111 : Island_std430_0 = islands_0[wg_6.island_2];
    var _S1112 : bool = wide_enter_1(tid_31, wg_6.island_2);
    if(!_S1112)
    {
        return;
    }
    var _S1113 : u32 = _S1111.info_0.x;
    var anchored_0 : bool = ((_S1113 & (u32(1)))) != u32(0);
    if(((_S1113 & (u32(16)))) != u32(0))
    {
        if(tid_31 == u32(0))
        {
            scratch_0[params_0.wide_base_0 + u32(8) * gx_0 + u32(6)] = vec4<f32>(0.0f);
        }
        return;
    }
    var _S1114 : Rigid_0 = wide_rigid_frame_0(tid_31, &(_S1111), wg_6);
    var work_11 : f32 = 0.0f;
    var work_err_7 : f32 = 0.0f;
    var _S1115 : vec3<f32> = vec3<f32>(0.0f);
    var tu_9 : vec3<f32> = _S1115;
    var pv_9 : vec3<f32> = _S1115;
    var c_63 : u32 = wg_6.begin_1 + tid_31;
    if(c_63 < (wg_6.end_0))
    {
        var _S1116 : bool = (params_0.rigid_motion_loads_0) != u32(0);
        var _S1117 : u32 = wide_step_0(&(_S1111));
        chunk_update_2(c_63, &(_S1111), _S1114, params_0.dt_0, _S1116, _S1117, ((_S1113 & (u32(4)))) != u32(0), &(work_11), &(work_err_7));
        if(!anchored_0)
        {
            drift_moments_2(c_63, &(tu_9), &(pv_9));
        }
    }
    var wsum_4 : vec3<f32> = vec3<f32>(work_11, work_err_7, 0.0f);
    var unused_7 : vec3<f32> = _S1115;
    group_sum3_2(tid_31, &(wsum_4), &(unused_7));
    var _S1118 : bool = !anchored_0;
    if(_S1118)
    {
        group_sum3_2(tid_31, &(tu_9), &(pv_9));
    }
    if(tid_31 == u32(0))
    {
        var _S1119 : u32 = u32(8) * gx_0;
        scratch_0[params_0.wide_base_0 + _S1119 + u32(6)] = vec4<f32>(wsum_4, 0.0f);
        if(_S1118)
        {
            wide_store_0(gx_0, u32(2), tu_9, pv_9);
        }
        scratch_0[params_0.wide_base_0 + _S1119 + u32(7)] = vec4<f32>(f32(g_seen_0[u32(0)]), f32(g_seen_0[u32(1)]), f32(g_seen_0[u32(2)]), 0.0f);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn wide_chunks(@builtin(workgroup_id) group_11 : vec3<u32>, @builtin(local_invocation_id) thread_11 : vec3<u32>)
{
    wide_chunks_at_0(group_11.x, thread_11.x);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn sums_and_impactors(@builtin(workgroup_id) group_12 : vec3<u32>, @builtin(local_invocation_id) thread_12 : vec3<u32>)
{
    var sums_groups_0 : u32 = (params_0.seg_count_0 + u32(255)) / u32(256);
    var _S1120 : u32 = group_12.x;
    if(_S1120 < sums_groups_0)
    {
        contact_sum_at_0(_S1120 * u32(256) + thread_12.x);
    }
    else
    {
        impactor_integrate_at_0(_S1120 - sums_groups_0, thread_12.x, false);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn islands_and_wide_chunks(@builtin(workgroup_id) group_13 : vec3<u32>, @builtin(local_invocation_id) thread_13 : vec3<u32>)
{
    var _S1121 : u32 = group_13.x;
    if(_S1121 < (params_0.halt_index_0))
    {
        island_frame_at_3(_S1121, thread_13.x);
    }
    else
    {
        wide_chunks_at_0(_S1121 - params_0.halt_index_0, thread_13.x);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn contact_islands_and_wide_chunks(@builtin(workgroup_id) group_14 : vec3<u32>, @builtin(local_invocation_id) thread_14 : vec3<u32>)
{
    var _S1122 : u32 = group_14.x;
    if(_S1122 < (params_0.halt_index_0))
    {
        island_frame_at_1(_S1122, thread_14.x);
    }
    else
    {
        wide_chunks_at_0(_S1122 - params_0.halt_index_0, thread_14.x);
    }
    return;
}

fn wide_bonds_at_0( gx_1 : u32,  tid_32 : u32)
{
    var bond_group_0 : bool = gx_1 < (params_0.wide_bond_groups_0);
    var wg_7 : WideGroup_0;
    if(bond_group_0)
    {
        wg_7 = wide_group_0(params_0.wide_bond_table_0, gx_1);
    }
    else
    {
        wg_7 = wide_group_0(params_0.wide_chunk_table_0, gx_1 - params_0.wide_bond_groups_0);
    }
    var _S1123 : WideGroup_0 = wg_7;
    var _S1124 : Island_std430_0 = islands_0[wg_7.island_2];
    var _S1125 : bool = wide_enter_1(tid_32, wg_7.island_2);
    if(!_S1125)
    {
        return;
    }
    var _S1126 : u32 = wide_step_1(&(_S1124));
    if(bond_group_0)
    {
        var _S1127 : vec4<u32> = _S1124.info_0;
        if((((_S1124.info_0.x) & (u32(16)))) != u32(0))
        {
            return;
        }
        var i_25 : u32 = wg_7.begin_1 + tid_32;
        var _S1128 : bool;
        if(i_25 < (wg_7.end_0))
        {
            var _S1129 : bool = bond_update_0(i_25, params_0.dt_0, (params_0.fracture_0) != u32(0), _S1127.w + u32(1), true);
            _S1128 = _S1129;
        }
        else
        {
            _S1128 = false;
        }
        if(_S1128)
        {
            islands_0[_S1123.island_2].info_0[i32(2)] = ((_S1127.z) | (u32(2)));
        }
        return;
    }
    var _S1130 : u32 = _S1124.info_0.x;
    if(((_S1130 & (u32(1)))) != u32(0))
    {
        return;
    }
    var _S1131 : vec3<f32> = vec3<f32>(0.0f);
    var f_32 : vec3<f32> = _S1131;
    var t_31 : vec3<f32> = _S1131;
    var c_64 : u32 = wg_7.begin_1 + tid_32;
    if(c_64 < (wg_7.end_0))
    {
        var _S1132 : Rigid_0 = rigid_of_2(&(_S1124));
        net_load_2(c_64, &(_S1124), _S1132, _S1126, params_0.dt_0, ((_S1130 & (u32(4)))) != u32(0), &(f_32), &(t_31));
    }
    group_sum3_0(tid_32, &(f_32), &(t_31));
    if(tid_32 == u32(0))
    {
        wide_store_0(gx_1 - params_0.wide_bond_groups_0, u32(0), f_32, t_31);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn wide_bonds(@builtin(workgroup_id) group_15 : vec3<u32>, @builtin(local_invocation_id) thread_15 : vec3<u32>)
{
    wide_bonds_at_0(group_15.x, thread_15.x);
    return;
}

fn bond_group_at_0( table_1 : u32,  gx_2 : u32,  tid_33 : u32)
{
    var wg_8 : WideGroup_0 = wide_group_0(table_1, gx_2);
    var isl_44 : Island_std430_0 = islands_0[wg_8.island_2];
    var _S1133 : bool = wide_enter_1(tid_33, wg_8.island_2);
    var _S1134 : bool;
    if(!_S1133)
    {
        _S1134 = true;
    }
    else
    {
        _S1134 = (((isl_44.info_0.x) & (u32(16)))) != u32(0);
    }
    if(_S1134)
    {
        return;
    }
    var i_26 : u32 = wg_8.begin_1 + tid_33;
    if(i_26 < (wg_8.end_0))
    {
        var _S1135 : bool = bond_update_0(i_26, params_0.dt_0, (params_0.fracture_0) != u32(0), isl_44.info_0.w + u32(1), true);
        _S1134 = _S1135;
    }
    else
    {
        _S1134 = false;
    }
    if(_S1134)
    {
        islands_0[wg_8.island_2].info_0[i32(2)] = ((isl_44.info_0.z) | (u32(2)));
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn sums_impactors_bonds(@builtin(workgroup_id) group_16 : vec3<u32>, @builtin(local_invocation_id) thread_16 : vec3<u32>)
{
    var sums_groups_1 : u32 = (params_0.seg_count_0 + u32(255)) / u32(256);
    var _S1136 : u32 = max(params_0.impactor_count_0, u32(1));
    var _S1137 : u32 = group_16.x;
    if(_S1137 < sums_groups_1)
    {
        contact_sum_at_0(_S1137 * u32(256) + thread_16.x);
    }
    else
    {
        if(_S1137 < (sums_groups_1 + _S1136))
        {
            impactor_integrate_at_0(_S1137 - sums_groups_1, thread_16.x, false);
        }
        else
        {
            var gb_0 : u32 = _S1137 - sums_groups_1 - _S1136;
            var narrow_0 : bool = gb_0 < (params_0.contact_bond_groups_0);
            var _S1138 : u32;
            if(narrow_0)
            {
                _S1138 = params_0.contact_bond_table_0;
            }
            else
            {
                _S1138 = params_0.wide_bond_table_0;
            }
            var _S1139 : u32;
            if(narrow_0)
            {
                _S1139 = gb_0;
            }
            else
            {
                _S1139 = gb_0 - params_0.contact_bond_groups_0;
            }
            bond_group_at_0(_S1138, _S1139, thread_16.x);
        }
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn forces_and_bonds(@builtin(workgroup_id) group_17 : vec3<u32>, @builtin(local_invocation_id) thread_17 : vec3<u32>)
{
    var force_groups_0 : u32 = ((params_0.pair_count_0 + params_0.cand_count_0) * pair_lane_count_0() + u32(255)) / u32(256);
    var _S1140 : u32 = group_17.x;
    if(_S1140 < force_groups_0)
    {
        contact_forces_at_0(_S1140 * u32(256) + thread_17.x, true);
    }
    else
    {
        bond_group_at_0(params_0.contact_bond_table_0, _S1140 - force_groups_0, thread_17.x);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn islands_sums_impactors(@builtin(workgroup_id) group_18 : vec3<u32>, @builtin(local_invocation_id) thread_18 : vec3<u32>)
{
    var _S1141 : u32 = group_18.x;
    if(_S1141 < (params_0.halt_index_0))
    {
        island_frame_at_0(_S1141, thread_18.x);
    }
    else
    {
        impactor_integrate_at_0(_S1141 - params_0.halt_index_0, thread_18.x, true);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn travel_start(@builtin(global_invocation_id) id_3 : vec3<u32>)
{
    var i_27 : u32 = id_3.x;
    if(i_27 < (params_0.chunk_count_0))
    {
        travel_check_0(i_27, true);
    }
    else
    {
        if((i_27 - params_0.chunk_count_0) < (params_0.impactor_count_0))
        {
            impactor_travel_check_0(i_27 - params_0.chunk_count_0, true);
        }
    }
    return;
}

