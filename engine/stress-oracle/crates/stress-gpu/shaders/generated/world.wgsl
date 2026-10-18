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
    @align(8) sload_base_0 : u32,
    @align(4) implicit_contact_0 : u32,
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

fn device_group_barrier_0()
{
    storageBarrier(); textureBarrier(); workgroupBarrier();;
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

fn to_local_0( _S105 : u32,  _S106 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(dot(_S106, bonds_0[_S105].t1_0.xyz), dot(_S106, bonds_0[_S105].t2_0.xyz), dot(_S106, bonds_0[_S105].normal_0.xyz));
}

fn is_damaged_0( st_0 : ptr<function, JointState_std430_0>) -> bool
{
    var _S107 : bool;
    if(((*st_0).damage_0) > 0.0f)
    {
        _S107 = true;
    }
    else
    {
        _S107 = ((*st_0).crush_1) > 0.0f;
    }
    return _S107;
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

fn connected_0( st_1 : ptr<function, JointState_std430_0>,  has_rebar_0 : bool) -> bool
{
    var _S108 : bool;
    if(((*st_1).damage_0) < 1.0f)
    {
        _S108 = true;
    }
    else
    {
        if(has_rebar_0)
        {
            _S108 = ((*st_1).rebar_broken_0) == 0.0f;
        }
        else
        {
            _S108 = false;
        }
    }
    return _S108;
}

fn fdiv_0( a_6 : f32,  b_11 : f32) -> f32
{
    return a_6 / b_11;
}

fn fsqrt_0( a_7 : f32) -> f32
{
    return sqrt(a_7);
}

fn stress_measures_0( b_12 : ptr<function, JointBond_std430_0>,  q_lin_0 : vec3<f32>,  q_ang_0 : vec3<f32>) -> Measures_0
{
    var area_2 : f32 = (*b_12).geom0_0.x;
    var _S109 : f32 = q_lin_0.z;
    var axial_0 : f32 = fdiv_0(_S109, area_2);
    var bending_0 : f32 = fdiv_0(abs(q_ang_0.x), (*b_12).geom1_0.x) + fdiv_0(abs(q_ang_0.y), (*b_12).geom1_0.y);
    var _S110 : f32 = q_lin_0.x;
    var _S111 : f32 = q_lin_0.y;
    var shear_1 : f32 = fdiv_0(fsqrt_0(_S110 * _S110 + _S111 * _S111), area_2) + fdiv_0(abs(q_ang_0.z), (*b_12).geom0_0.w);
    var m_1 : Measures_0;
    m_1.tension_0 = axial_0 + bending_0;
    m_1.shear_0 = shear_1;
    var _S112 : f32 = - axial_0;
    m_1.normal_compression_0 = max(_S112, 0.0f);
    m_1.compression_0 = _S112 + bending_0;
    m_1.compressive_force_0 = max(- _S109, 0.0f);
    return m_1;
}

fn expm1_accurate_0( x_3 : f32) -> f32
{
    if((abs(x_3)) < 0.00100000004749745f)
    {
        return x_3 * (1.0f + x_3 * (0.5f + x_3 * 0.1666666716337204f));
    }
    return exp(x_3) - 1.0f;
}

fn fpow_0( a_8 : f32,  b_13 : f32) -> f32
{
    return pow(a_8, b_13);
}

fn dif_factor_0( mat_1 : ptr<function, JointMaterial_std140_0>,  strain_rate_1 : f32) -> f32
{
    var r_2 : f32 = abs(strain_rate_1);
    var _S113 : vec4<f32> = (*mat_1).dif_0;
    var ref_0 : f32 = (*mat_1).dif_0.x;
    if(r_2 <= ref_0)
    {
        return 1.0f;
    }
    var _S114 : f32 = _S113.z;
    var f_6 : f32;
    if(r_2 <= _S114)
    {
        f_6 = fpow_0(fdiv_0(r_2, ref_0), _S113.y);
    }
    else
    {
        f_6 = fpow_0(fdiv_0(_S114, ref_0), _S113.y) * fpow_0(fdiv_0(r_2, _S114), _S113.w);
    }
    return clamp(f_6, 1.0f, (*mat_1).misc_0.x);
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

fn failure_indices_0( mat_3 : ptr<function, JointMaterial_std140_0>,  b_14 : ptr<function, JointBond_std430_0>,  m_2 : Measures_0,  multiplier_0 : f32) -> vec4<f32>
{
    var fc_1 : f32 = (*mat_3).strength_0.y * multiplier_0;
    var _S115 : f32 = min((*mat_3).strength_0.z * multiplier_0 + (*mat_3).strength_0.w * m_2.normal_compression_0, (*mat_3).energy_1.x * multiplier_0);
    var idx_0 : vec4<f32>;
    idx_0[i32(0)] = max(fdiv_0(m_2.tension_0, (*mat_3).strength_0.x * multiplier_0), 0.0f);
    var _S116 : f32;
    if(_S115 > 0.0f)
    {
        _S116 = fdiv_0(m_2.shear_0, _S115);
    }
    else
    {
        _S116 = infinity_0();
    }
    idx_0[i32(1)] = _S116;
    idx_0[i32(2)] = max(fdiv_0(m_2.compression_0, fc_1), 0.0f);
    var _S117 : f32 = (*b_14).stiff1_0.y;
    if(_S117 > 0.0f)
    {
        _S116 = fdiv_0(m_2.compressive_force_0, _S117);
    }
    else
    {
        _S116 = 0.0f;
    }
    idx_0[i32(3)] = _S116;
    return idx_0;
}

fn sq_0( x_4 : f32) -> f32
{
    return x_4 * x_4;
}

fn compressed_region_0( w0_0 : f32,  w1_0 : f32,  dz_0 : f32,  ax_0 : f32,  ay_0 : f32,  region_0 : ptr<function, array<f32, i32(6)>>)
{
    var count_1 : u32;
    var h0_0 : f32 = 0.5f * w0_0;
    var h1_0 : f32 = 0.5f * w1_0;
    var _S118 : f32 = - h0_0;
    var _S119 : f32 = - h1_0;
    var _S120 : array<vec2<f32>, i32(4)> = array<vec2<f32>, i32(4)>( vec2<f32>(_S118, _S119), vec2<f32>(h0_0, _S119), vec2<f32>(h0_0, h1_0), vec2<f32>(_S118, h1_0) );
    var poly_0 : array<vec2<f32>, i32(8)>;
    var i_2 : u32 = u32(0);
    var count_2 : u32 = u32(0);
    loop
    {
        if(i_2 < u32(4))
        {
        }
        else
        {
            break;
        }
        var _S121 : u32 = i_2;
        var _S122 : u32 = i_2 + u32(1);
        var _S123 : u32 = _S122 % u32(4);
        var _S124 : f32 = _S120[i_2].y;
        var _S125 : f32 = _S120[i_2].x;
        var fp_0 : f32 = dz_0 + ax_0 * _S124 - ay_0 * _S125;
        var _S126 : f32 = _S120[_S123].y;
        var _S127 : f32 = _S120[_S123].x;
        var fq_0 : f32 = dz_0 + ax_0 * _S126 - ay_0 * _S127;
        var _S128 : bool = fp_0 < 0.0f;
        if(_S128)
        {
            var _S129 : u32 = count_2 + u32(1);
            poly_0[count_2] = _S120[_S121];
            count_1 = _S129;
        }
        else
        {
            count_1 = count_2;
        }
        if(_S128 != (fq_0 < 0.0f))
        {
            var t_8 : f32 = fp_0 / (fp_0 - fq_0);
            var _S130 : u32 = count_1 + u32(1);
            poly_0[count_1] = vec2<f32>(_S125 + t_8 * (_S127 - _S125), _S124 + t_8 * (_S126 - _S124));
            count_2 = _S130;
        }
        else
        {
            count_2 = count_1;
        }
        i_2 = _S122;
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
    var o_0 : vec2<f32> = poly_0[i32(0)];
    i_2 = u32(0);
    var a_9 : f32 = 0.0f;
    var sx_0 : f32 = 0.0f;
    var sy_0 : f32 = 0.0f;
    var ixx_0 : f32 = 0.0f;
    var iyy_0 : f32 = 0.0f;
    var ixy_0 : f32 = 0.0f;
    loop
    {
        if(i_2 < count_2)
        {
        }
        else
        {
            break;
        }
        var _S131 : f32 = o_0.x;
        var x0_0 : f32 = poly_0[i_2].x - _S131;
        var _S132 : f32 = o_0.y;
        var y0_0 : f32 = poly_0[i_2].y - _S132;
        var _S133 : u32 = i_2 + u32(1);
        var _S134 : u32 = _S133 % count_2;
        var x1_0 : f32 = poly_0[_S134].x - _S131;
        var y1_0 : f32 = poly_0[_S134].y - _S132;
        var _S135 : f32 = x0_0 * y1_0;
        var _S136 : f32 = x1_0 * y0_0;
        var cr_0 : f32 = _S135 - _S136;
        var a_10 : f32 = a_9 + cr_0 / 2.0f;
        var sx_1 : f32 = sx_0 + (x0_0 + x1_0) * cr_0 / 6.0f;
        var sy_1 : f32 = sy_0 + (y0_0 + y1_0) * cr_0 / 6.0f;
        var ixx_1 : f32 = ixx_0 + (x0_0 * x0_0 + x0_0 * x1_0 + x1_0 * x1_0) * cr_0 / 12.0f;
        var iyy_1 : f32 = iyy_0 + (y0_0 * y0_0 + y0_0 * y1_0 + y1_0 * y1_0) * cr_0 / 12.0f;
        var ixy_1 : f32 = ixy_0 + (_S135 + 2.0f * x0_0 * y0_0 + 2.0f * x1_0 * y1_0 + _S136) * cr_0 / 24.0f;
        i_2 = _S133;
        a_9 = a_10;
        sx_0 = sx_1;
        sy_0 = sy_1;
        ixx_0 = ixx_1;
        iyy_0 = iyy_1;
        ixy_0 = ixy_1;
    }
    if(a_9 <= 0.0f)
    {
        return;
    }
    var cx_0 : f32 = sx_0 / a_9;
    var cy_0 : f32 = sy_0 / a_9;
    (*region_0)[i32(0)] = a_9;
    (*region_0)[i32(1)] = o_0.x + cx_0;
    (*region_0)[i32(2)] = o_0.y + cy_0;
    var _S137 : f32 = a_9 * cx_0;
    (*region_0)[i32(3)] = ixx_0 - _S137 * cx_0;
    (*region_0)[i32(4)] = iyy_0 - a_9 * cy_0 * cy_0;
    (*region_0)[i32(5)] = ixy_0 - _S137 * cy_0;
    return;
}

fn no_tension_patch_0( kn_0 : f32,  w0_1 : f32,  w1_1 : f32,  dz_1 : f32,  ax_1 : f32,  ay_1 : f32) -> vec4<f32>
{
    var r_3 : array<f32, i32(6)>;
    compressed_region_0(w0_1, w1_1, dz_1, ax_1, ay_1, &(r_3));
    var a_11 : f32 = r_3[i32(0)];
    if((r_3[i32(0)]) == 0.0f)
    {
        return vec4<f32>(0.0f);
    }
    var k_8 : f32 = kn_0 / (w0_1 * w1_1);
    var fc_2 : f32 = dz_1 + ax_1 * r_3[i32(2)] - ay_1 * r_3[i32(1)];
    var _S138 : f32 = a_11 * fc_2;
    var _S139 : f32 = - ay_1;
    return vec4<f32>(k_8 * a_11 * fc_2, k_8 * (_S138 * r_3[i32(2)] + (_S139 * r_3[i32(5)] + ax_1 * r_3[i32(4)])), - k_8 * (_S138 * r_3[i32(1)] + (_S139 * r_3[i32(3)] + ax_1 * r_3[i32(5)])), 0.5f * k_8 * (_S138 * fc_2 + ay_1 * ay_1 * r_3[i32(3)] + ax_1 * ax_1 * r_3[i32(4)] - 2.0f * ax_1 * ay_1 * r_3[i32(5)]));
}

var<private> SPRING_AT_0 : array<f32, i32(6)> = array<f32, i32(6)>( -0.4166666567325592f, -0.25f, -0.0833333358168602f, 0.0833333358168602f, 0.25f, 0.4166666567325592f );
fn contact_normal_0( mat_4 : ptr<function, JointMaterial_std140_0>,  b_15 : ptr<function, JointBond_std430_0>,  crush_2 : f32,  d_lin_0 : vec3<f32>,  d_ang_0 : vec3<f32>) -> vec4<f32>
{
    var kn_1 : f32 = (*b_15).stiff0_0.x;
    var w0_2 : f32 = (*b_15).geom0_0.y;
    var w1_2 : f32 = (*b_15).geom0_0.z;
    var nc_sum_0 : f32;
    var m1_0 : f32;
    var m2_0 : f32;
    var energy_2 : f32;
    if(((((*mat_4).kind_flags_0.y) & (u32(4)))) != u32(0))
    {
        var p_2 : vec4<f32> = no_tension_patch_0(kn_1 * (1.0f - crush_2), w0_2, w1_2, d_lin_0.z, d_ang_0.x, d_ang_0.y);
        var _S140 : f32 = p_2.y;
        var _S141 : f32 = p_2.z;
        var _S142 : f32 = p_2.w;
        nc_sum_0 = p_2.x;
        m1_0 = _S140;
        m2_0 = _S141;
        energy_2 = _S142;
    }
    else
    {
        var ki_0 : f32 = kn_1 * (1.0f - crush_2) / 36.0f;
        var _S143 : f32 = d_ang_0.x;
        var _S144 : f32 = d_ang_0.y;
        var spread_0 : f32 = abs(_S143) * 0.4166666567325592f * w1_2 + abs(_S144) * 0.4166666567325592f * w0_2;
        var _S145 : f32 = d_lin_0.z;
        var slack_0 : f32 = 9.99999997475242708e-07f * (abs(_S145) + spread_0);
        if((_S145 - spread_0) > slack_0)
        {
            nc_sum_0 = 0.0f;
            m1_0 = 0.0f;
            m2_0 = 0.0f;
            energy_2 = 0.0f;
        }
        else
        {
            if((_S145 + spread_0) < (- slack_0))
            {
                var i1_0 : f32 = 2.91666650772094727f * w0_2 * w0_2;
                var i2_0 : f32 = 2.91666650772094727f * w1_2 * w1_2;
                var _S146 : f32 = ki_0 * _S143 * i2_0;
                var _S147 : f32 = ki_0 * _S144 * i1_0;
                var _S148 : f32 = 0.5f * ki_0 * (36.0f * _S145 * _S145 + _S143 * _S143 * i2_0 + _S144 * _S144 * i1_0);
                nc_sum_0 = ki_0 * 36.0f * _S145;
                m1_0 = _S146;
                m2_0 = _S147;
                energy_2 = _S148;
            }
            else
            {
                var i_3 : u32 = u32(0);
                var nc_sum_1 : f32 = 0.0f;
                var m1_1 : f32 = 0.0f;
                var m2_1 : f32 = 0.0f;
                var energy_3 : f32 = 0.0f;
                loop
                {
                    if(i_3 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var _S149 : f32 = SPRING_AT_0[i_3] * w0_2;
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
                        var di_0 : f32 = _S145 + _S143 * s2_0 - _S144 * _S149;
                        if(di_0 < 0.0f)
                        {
                            var f_7 : f32 = ki_0 * di_0;
                            var m1_2 : f32 = m1_0 + f_7 * s2_0;
                            var m2_2 : f32 = m2_0 - f_7 * _S149;
                            var energy_4 : f32 = energy_2 + 0.5f * ki_0 * di_0 * di_0;
                            nc_sum_0 = nc_sum_0 + f_7;
                            m1_0 = m1_2;
                            m2_0 = m2_2;
                            energy_2 = energy_4;
                        }
                        j_0 = j_0 + u32(1);
                    }
                    i_3 = i_3 + u32(1);
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

fn damage_law_0( kind_1 : u32,  kappa_1 : f32,  r_4 : f32) -> f32
{
    if(kappa_1 <= 1.0f)
    {
        return 0.0f;
    }
    if(kind_1 == u32(0))
    {
        if(r_4 <= 1.0f)
        {
            return 1.0f;
        }
        return min(fdiv_0(r_4 * (kappa_1 - 1.0f), kappa_1 * (r_4 - 1.0f)), 1.0f);
    }
    if(kappa_1 >= (0.5f * (r_4 + 1.0f)))
    {
        return 1.0f;
    }
    return 1.0f - fdiv_0(1.0f, kappa_1);
}

fn damage_increment_0( kind_2 : u32,  kappa_old_0 : f32,  lambda_0 : f32,  r_5 : f32,  d_old_0 : f32,  psi_0 : f32) -> vec2<f32>
{
    var _S150 : f32 = max(damage_law_0(kind_2, lambda_0, r_5), d_old_0);
    var _S151 : bool;
    if(_S150 <= d_old_0)
    {
        _S151 = true;
    }
    else
    {
        _S151 = d_old_0 >= 1.0f;
    }
    if(_S151)
    {
        return vec2<f32>(d_old_0, 0.0f);
    }
    var u0_0 : f32 = fdiv_0(psi_0, lambda_0 * lambda_0);
    var _S152 : f32 = max(kappa_old_0, 1.0f);
    if(kind_2 == u32(0))
    {
        if(r_5 > 1.0f)
        {
            return vec2<f32>(_S150, fdiv_0(u0_0 * r_5, r_5 - 1.0f) * max(min(lambda_0, r_5) - min(_S152, r_5), 0.0f));
        }
        return vec2<f32>(_S150, (1.0f - d_old_0) * psi_0);
    }
    var ku_0 : f32 = 0.5f * (r_5 + 1.0f);
    var plateau_0 : f32 = u0_0 * max(min(lambda_0, ku_0) - min(_S152, ku_0), 0.0f);
    var snap_0 : f32;
    if(_S150 >= 1.0f)
    {
        snap_0 = u0_0 * ku_0;
    }
    else
    {
        snap_0 = 0.0f;
    }
    return vec2<f32>(_S150, plateau_0 + snap_0);
}

struct Contact_0
{
     q_lin_1 : vec3<f32>,
     q_ang_1 : vec3<f32>,
     energy_5 : f32,
     diss_1 : f32,
     plastic_0 : vec3<f32>,
};

fn signum_0( x_5 : f32) -> f32
{
    var _S153 : f32;
    if((((bitcast<u32>((x_5))) & (u32(2147483648)))) != u32(0))
    {
        _S153 = -1.0f;
    }
    else
    {
        _S153 = 1.0f;
    }
    return _S153;
}

fn return_map_0( k_9 : f32,  total_0 : f32,  plastic_1 : f32,  cap_0 : f32) -> vec2<f32>
{
    var trial_0 : f32 = k_9 * (total_0 - plastic_1);
    if((abs(trial_0)) <= cap_0)
    {
        return vec2<f32>(trial_0, 0.0f);
    }
    var f_8 : f32 = cap_0 * signum_0(trial_0);
    return vec2<f32>(f_8, fdiv_0(trial_0 - f_8, k_9));
}

fn contact_part_0( mat_5 : ptr<function, JointMaterial_std140_0>,  b_16 : ptr<function, JointBond_std430_0>,  crush_3 : f32,  plastic_2 : vec3<f32>,  d_lin_1 : vec3<f32>,  d_ang_1 : vec3<f32>) -> Contact_0
{
    var c_9 : Contact_0;
    var _S154 : vec3<f32> = vec3<f32>(0.0f);
    c_9.q_lin_1 = _S154;
    c_9.q_ang_1 = _S154;
    c_9.energy_5 = 0.0f;
    c_9.diss_1 = 0.0f;
    c_9.plastic_0 = plastic_2;
    if(((((*mat_5).kind_flags_0.y) & (u32(2)))) == u32(0))
    {
        return c_9;
    }
    var ks_0 : f32 = (*b_16).stiff0_0.y;
    var kt_0 : f32 = (*b_16).stiff1_0.x;
    var _S155 : vec4<f32> = contact_normal_0(&((*mat_5)), &((*b_16)), crush_3, d_lin_1, d_ang_1);
    var nc_sum_2 : f32 = _S155.x;
    var m1_3 : f32 = _S155.y;
    var m2_3 : f32 = _S155.z;
    var energy_6 : f32 = _S155.w;
    var nc_0 : f32 = - nc_sum_2;
    var p_3 : vec3<f32> = plastic_2;
    c_9.q_lin_1 = vec3<f32>(0.0f, 0.0f, nc_sum_2);
    c_9.q_ang_1 = vec3<f32>(m1_3, m2_3, 0.0f);
    var slide_cap_0 : f32 = (*mat_5).strength_0.w * nc_0;
    var _S156 : f32 = ks_0 * (d_lin_1.x - plastic_2.x);
    var _S157 : f32 = ks_0 * (d_lin_1.y - plastic_2.y);
    var tn_0 : f32 = fsqrt_0(_S156 * _S156 + _S157 * _S157);
    var _S158 : bool;
    if(tn_0 > slide_cap_0)
    {
        _S158 = tn_0 > 0.0f;
    }
    else
    {
        _S158 = false;
    }
    var diss_2 : f32;
    if(_S158)
    {
        var _S159 : f32 = fdiv_0(_S156, tn_0);
        var _S160 : f32 = fdiv_0(_S157, tn_0);
        var dslip_0 : f32 = fdiv_0(tn_0 - slide_cap_0, ks_0);
        p_3[i32(0)] = p_3[i32(0)] + _S159 * dslip_0;
        p_3[i32(1)] = p_3[i32(1)] + _S160 * dslip_0;
        c_9.q_lin_1[i32(0)] = _S159 * slide_cap_0;
        c_9.q_lin_1[i32(1)] = _S160 * slide_cap_0;
        diss_2 = slide_cap_0 * dslip_0;
    }
    else
    {
        c_9.q_lin_1[i32(0)] = _S156;
        c_9.q_lin_1[i32(1)] = _S157;
        diss_2 = 0.0f;
    }
    var tq_0 : vec2<f32> = return_map_0(kt_0, d_ang_1.z, p_3.z, slide_cap_0 * (*b_16).geom1_0.z);
    var _S161 : f32 = tq_0.x;
    var _S162 : f32 = tq_0.y;
    var diss_3 : f32 = diss_2 + abs(_S161) * abs(_S162);
    p_3[i32(2)] = p_3[i32(2)] + _S162;
    c_9.q_ang_1[i32(2)] = _S161;
    c_9.energy_5 = energy_6 + 0.5f * (fdiv_0(sq_0(c_9.q_lin_1.x), ks_0) + fdiv_0(sq_0(c_9.q_lin_1.y), ks_0) + fdiv_0(sq_0(_S161), kt_0));
    c_9.diss_1 = diss_3;
    c_9.plastic_0 = p_3;
    return c_9;
}

fn contact_offsets_0( mat_6 : ptr<function, JointMaterial_std140_0>,  b_17 : ptr<function, JointBond_std430_0>,  crush_4 : f32,  plastic_3 : vec3<f32>,  d_lin_2 : vec3<f32>,  d_ang_2 : vec3<f32>) -> vec3<f32>
{
    var _S163 : u32 = (*mat_6).kind_flags_0.y;
    if(((_S163 & (u32(2)))) == u32(0))
    {
        return plastic_3;
    }
    var kn_2 : f32 = (*b_17).stiff0_0.x;
    var ks_1 : f32 = (*b_17).stiff0_0.y;
    var kt_1 : f32 = (*b_17).stiff1_0.x;
    var w0_3 : f32 = (*b_17).geom0_0.y;
    var w1_3 : f32 = (*b_17).geom0_0.z;
    var nc_sum_3 : f32;
    if(((_S163 & (u32(4)))) != u32(0))
    {
        var _S164 : vec4<f32> = no_tension_patch_0(kn_2 * (1.0f - crush_4), w0_3, w1_3, d_lin_2.z, d_ang_2.x, d_ang_2.y);
        nc_sum_3 = _S164.x;
    }
    else
    {
        var ki_1 : f32 = kn_2 * (1.0f - crush_4) / 36.0f;
        var _S165 : f32 = d_ang_2.x;
        var _S166 : f32 = d_ang_2.y;
        var spread_1 : f32 = abs(_S165) * 0.4166666567325592f * w1_3 + abs(_S166) * 0.4166666567325592f * w0_3;
        var _S167 : f32 = d_lin_2.z;
        var slack_1 : f32 = 9.99999997475242708e-07f * (abs(_S167) + spread_1);
        if((_S167 - spread_1) > slack_1)
        {
            nc_sum_3 = 0.0f;
        }
        else
        {
            if((_S167 + spread_1) < (- slack_1))
            {
                nc_sum_3 = ki_1 * 36.0f * _S167;
            }
            else
            {
                var i_4 : u32 = u32(0);
                var nc_sum_4 : f32 = 0.0f;
                loop
                {
                    if(i_4 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var _S168 : f32 = SPRING_AT_0[i_4] * w0_3;
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
                        var di_1 : f32 = _S167 + _S165 * (SPRING_AT_0[j_1] * w1_3) - _S166 * _S168;
                        if(di_1 < 0.0f)
                        {
                            nc_sum_3 = nc_sum_3 + ki_1 * di_1;
                        }
                        j_1 = j_1 + u32(1);
                    }
                    i_4 = i_4 + u32(1);
                    nc_sum_4 = nc_sum_3;
                }
                nc_sum_3 = nc_sum_4;
            }
        }
    }
    var nc_1 : f32 = - nc_sum_3;
    var p_4 : vec3<f32> = plastic_3;
    var slide_cap_1 : f32 = (*mat_6).strength_0.w * nc_1;
    var _S169 : f32 = ks_1 * (d_lin_2.x - plastic_3.x);
    var _S170 : f32 = ks_1 * (d_lin_2.y - plastic_3.y);
    var tn_1 : f32 = fsqrt_0(_S169 * _S169 + _S170 * _S170);
    var _S171 : bool;
    if(tn_1 > slide_cap_1)
    {
        _S171 = tn_1 > 0.0f;
    }
    else
    {
        _S171 = false;
    }
    if(_S171)
    {
        var _S172 : f32 = fdiv_0(_S170, tn_1);
        var dslip_1 : f32 = fdiv_0(tn_1 - slide_cap_1, ks_1);
        p_4[i32(0)] = p_4[i32(0)] + fdiv_0(_S169, tn_1) * dslip_1;
        p_4[i32(1)] = p_4[i32(1)] + _S172 * dslip_1;
    }
    p_4[i32(2)] = p_4[i32(2)] + return_map_0(kt_1, d_ang_2.z, p_4.z, slide_cap_1 * (*b_17).geom1_0.z).y;
    return p_4;
}

fn life_rate_0( mat_7 : ptr<function, JointMaterial_std140_0>,  s_4 : f32) -> f32
{
    if(s_4 <= 0.0f)
    {
        return 0.0f;
    }
    var _S173 : f32 = (*mat_7).misc_0.y;
    return fdiv_0((_S173 + 1.0f) * fpow_0(s_4, _S173), (*mat_7).misc_0.z);
}

fn connected_1( st_2 : JointState_0,  has_rebar_1 : bool) -> bool
{
    var _S174 : bool;
    if((st_2.damage_0) < 1.0f)
    {
        _S174 = true;
    }
    else
    {
        if(has_rebar_1)
        {
            _S174 = (st_2.rebar_broken_0) == 0.0f;
        }
        else
        {
            _S174 = false;
        }
    }
    return _S174;
}

fn joint_evaluate_0( mat_8 : ptr<function, JointMaterial_std140_0>,  b_18 : ptr<function, JointBond_std430_0>,  state_2 : ptr<function, JointState_std430_0>,  d_lin_3 : vec3<f32>,  d_ang_3 : vec3<f32>,  dt_6 : f32,  fracture_1 : bool) -> JointResponse_0
{
    var kn_3 : f32 = (*b_18).stiff0_0.x;
    var ks_2 : f32 = (*b_18).stiff0_0.y;
    var kb1_0 : f32 = (*b_18).stiff0_0.z;
    var kb2_0 : f32 = (*b_18).stiff0_0.w;
    var _S175 : vec4<f32> = (*b_18).stiff1_0;
    var kt_2 : f32 = (*b_18).stiff1_0.x;
    var has_rebar_2 : bool = ((*b_18).stiff1_0.w) != 0.0f;
    var kind_3 : u32 = (*mat_8).kind_flags_0.x;
    var flags_0 : u32 = (*mat_8).kind_flags_0.y;
    var softening_0 : bool = ((flags_0 & (u32(1)))) != u32(0);
    var st_3 : JointState_0;
    st_3.damage_0 = (*state_2).damage_0;
    st_3.crush_1 = (*state_2).crush_1;
    st_3.kappa_0 = (*state_2).kappa_0;
    st_3.kappa_c_0 = (*state_2).kappa_c_0;
    st_3.ductility_0 = (*state_2).ductility_0;
    st_3.ductility_c_0 = (*state_2).ductility_c_0;
    st_3.life_0 = (*state_2).life_0;
    st_3.plastic_x_0 = (*state_2).plastic_x_0;
    st_3.plastic_y_0 = (*state_2).plastic_y_0;
    st_3.plastic_t_0 = (*state_2).plastic_t_0;
    st_3.rebar_plastic_0 = (*state_2).rebar_plastic_0;
    st_3.rebar_slip0_0 = (*state_2).rebar_slip0_0;
    st_3.rebar_slip1_0 = (*state_2).rebar_slip1_0;
    st_3.rebar_work_0 = (*state_2).rebar_work_0;
    st_3.rebar_broken_0 = (*state_2).rebar_broken_0;
    st_3.strain_rate_0 = (*state_2).strain_rate_0;
    st_3.governing_stress_0 = (*state_2).governing_stress_0;
    st_3.dissipated_0 = (*state_2).dissipated_0;
    st_3.utilization_0 = (*state_2).utilization_0;
    st_3.mode_0 = (*state_2).mode_0;
    var _S176 : bool = connected_0(&((*state_2)), has_rebar_2);
    var qe_lin_0 : vec3<f32> = d_lin_3 * vec3<f32>(ks_2, ks_2, kn_3);
    var qe_ang_0 : vec3<f32> = d_ang_3 * vec3<f32>(kb1_0, kb2_0, kt_2);
    var _S177 : Measures_0 = stress_measures_0(&((*b_18)), qe_lin_0, qe_ang_0);
    var _S178 : f32 = max(max(_S177.tension_0, _S177.shear_0), _S177.compression_0);
    var _S179 : bool = dt_6 > 0.0f;
    var dif_1 : f32;
    if(_S179)
    {
        var raw_0 : f32 = fdiv_0(max(fdiv_0(_S178 - st_3.governing_stress_0, dt_6), 0.0f), (*mat_8).misc_0.w);
        var tau_2 : f32 = _S175.z;
        if(((flags_0 & (u32(16)))) != u32(0))
        {
            dif_1 = - expm1_accurate_0(- fdiv_0(dt_6, tau_2));
        }
        else
        {
            dif_1 = min(fdiv_0(dt_6, tau_2), 1.0f);
        }
        st_3.strain_rate_0 = st_3.strain_rate_0 + (raw_0 - st_3.strain_rate_0) * dif_1;
        st_3.governing_stress_0 = _S178;
    }
    if(((flags_0 & (u32(32)))) != u32(0))
    {
        var _S180 : f32 = dif_factor_0(&((*mat_8)), st_3.strain_rate_0);
        dif_1 = _S180;
    }
    else
    {
        dif_1 = 1.0f;
    }
    var weibull_0 : f32 = (*b_18).geom1_0.w;
    var _S181 : f32 = fatigue_factor_0(&((*mat_8)), st_3.life_0);
    var multiplier_1 : f32 = weibull_0 * dif_1 * _S181;
    var _S182 : vec4<f32> = failure_indices_0(&((*mat_8)), &((*b_18)), _S177, multiplier_1);
    var _S183 : f32 = _S182.x;
    var _S184 : f32 = _S182.y;
    st_3.utilization_0 = max(max(_S183, _S184), max(_S182.z, _S182.w));
    var _S185 : f32 = d_lin_3.x;
    var _S186 : f32 = d_lin_3.y;
    var _S187 : f32 = d_ang_3.z;
    var _S188 : f32 = ks_2 * (sq_0(_S185) + sq_0(_S186)) + kb1_0 * sq_0(d_ang_3.x) + kb2_0 * sq_0(d_ang_3.y) + kt_2 * sq_0(_S187);
    var _S189 : f32 = d_lin_3.z;
    var _S190 : bool = _S189 > 0.0f;
    if(_S190)
    {
        dif_1 = kn_3 * sq_0(_S189);
    }
    else
    {
        dif_1 = 0.0f;
    }
    var psi_ts_0 : f32 = 0.5f * (_S188 + dif_1);
    var psi_c_0 : f32;
    if(_S189 < 0.0f)
    {
        psi_c_0 = 0.5f * kn_3 * sq_0(_S189);
    }
    else
    {
        psi_c_0 = 0.0f;
    }
    var plastic_4 : vec3<f32> = vec3<f32>(st_3.plastic_x_0, st_3.plastic_y_0, st_3.plastic_t_0);
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
        var _S191 : vec4<f32> = contact_normal_0(&((*mat_8)), &((*b_18)), (*state_2).crush_1, d_lin_3, d_ang_3);
        patch_0 = _S191.w + 0.5f * (ks_2 * (sq_0(_S185 - (*state_2).plastic_x_0) + sq_0(_S186 - (*state_2).plastic_y_0)) + kt_2 * sq_0(_S187 - (*state_2).plastic_t_0));
    }
    else
    {
        patch_0 = 0.0f;
    }
    if(has_rebar_2)
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
        rebar_0 = 0.5f * ((*b_18).rebar0_0.x * sq_0(_S189 - (*state_2).rebar_plastic_0) + (*b_18).rebar0_0.y * (sq_0(_S185 - (*state_2).rebar_slip0_0) + sq_0(_S186 - (*state_2).rebar_slip1_0)));
    }
    else
    {
        rebar_0 = 0.0f;
    }
    var _S192 : f32 = (*state_2).damage_0;
    var _S193 : f32 = 1.0f - (*state_2).damage_0;
    var _S194 : f32 = (*state_2).crush_1;
    var frozen_0 : f32 = _S193 * (psi_ts_0 + (1.0f - (*state_2).crush_1) * psi_c_0) + (*state_2).damage_0 * patch_0 + rebar_0;
    var diss_contact_0 : f32;
    var psi_contact_0 : f32;
    var intact_normal_0 : f32;
    var dissipated_3 : f32;
    var qc_lin_0 : vec3<f32>;
    if(fracture_1)
    {
        var _S195 : bool = _S183 >= _S184;
        if(_S195)
        {
            diss_contact_0 = _S183;
        }
        else
        {
            diss_contact_0 = _S184;
        }
        var mode_ts_0 : u32;
        if(_S195)
        {
            mode_ts_0 = u32(1);
        }
        else
        {
            mode_ts_0 = u32(2);
        }
        if(diss_contact_0 > (st_3.kappa_0))
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
                intact_normal_0 = fdiv_0(psi_contact_0 * (*b_18).geom0_0.x * diss_contact_0 * diss_contact_0, psi_ts_0);
            }
            else
            {
                intact_normal_0 = 0.0f;
            }
            st_3.ductility_0 = intact_normal_0;
            if(softening_0)
            {
                mode_c_0 = kind_3;
            }
            else
            {
                mode_c_0 = u32(0);
            }
            var inc_0 : vec2<f32> = damage_increment_0(mode_c_0, st_3.kappa_0, diss_contact_0, intact_normal_0, st_3.damage_0, psi_ts_0);
            var _S196 : f32 = inc_0.x;
            if(_S196 > (st_3.damage_0))
            {
                var _S197 : Contact_0 = contact_part_0(&((*mat_8)), &((*b_18)), st_3.crush_1, plastic_4, d_lin_3, d_ang_3);
                var _S198 : f32 = max(inc_0.y - max(_S197.energy_5 - (1.0f - st_3.crush_1) * psi_c_0, 0.0f) * (_S196 - st_3.damage_0), 0.0f);
                st_3.damage_0 = _S196;
                st_3.mode_0 = mode_ts_0;
                dissipated_3 = _S198;
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
        st_3.kappa_0 = max(st_3.kappa_0, diss_contact_0);
        if(_S192 > 0.0f)
        {
            var _S199 : Contact_0 = contact_part_0(&((*mat_8)), &((*b_18)), _S194, vec3<f32>((*state_2).plastic_x_0, (*state_2).plastic_y_0, (*state_2).plastic_t_0), d_lin_3, d_ang_3);
            qc_lin_0 = qe_ang_0 * vec3<f32>(_S193) + _S199.q_ang_1 * vec3<f32>(_S192);
        }
        else
        {
            qc_lin_0 = qe_ang_0;
        }
        var _S200 : Measures_0 = stress_measures_0(&((*b_18)), vec3<f32>(0.0f, 0.0f, min(qe_lin_0.z, 0.0f)), qc_lin_0);
        var _S201 : vec4<f32> = failure_indices_0(&((*mat_8)), &((*b_18)), _S200, multiplier_1);
        var _S202 : f32 = _S201.z;
        var _S203 : f32 = _S201.w;
        var _S204 : bool = _S202 >= _S203;
        if(_S204)
        {
            psi_contact_0 = _S202;
        }
        else
        {
            psi_contact_0 = _S203;
        }
        if(_S204)
        {
            mode_c_0 = u32(3);
        }
        else
        {
            mode_c_0 = u32(4);
        }
        if(psi_contact_0 > (st_3.kappa_c_0))
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
                intact_normal_0 = fdiv_0((*mat_8).energy_1.w * (*b_18).geom0_0.x * psi_contact_0 * psi_contact_0, psi_c_0);
            }
            else
            {
                intact_normal_0 = 0.0f;
            }
            st_3.ductility_c_0 = intact_normal_0;
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
                    mode_ts_0 = kind_3;
                }
                law_1 = mode_ts_0;
            }
            var inc_1 : vec2<f32> = damage_increment_0(law_1, st_3.kappa_c_0, psi_contact_0, intact_normal_0, st_3.crush_1, psi_c_0);
            var _S205 : f32 = inc_1.x;
            if(_S205 > (st_3.crush_1))
            {
                var dissipated_4 : f32 = dissipated_3 + inc_1.y;
                st_3.crush_1 = _S205;
                st_3.mode_0 = mode_c_0;
                if(_S205 >= 1.0f)
                {
                    changed_0 = (st_3.damage_0) < 1.0f;
                }
                else
                {
                    changed_0 = false;
                }
                if(changed_0)
                {
                    var dissipated_5 : f32 = dissipated_4 + psi_ts_0 * (1.0f - st_3.damage_0);
                    st_3.damage_0 = 1.0f;
                    dissipated_3 = dissipated_5;
                }
                else
                {
                    dissipated_3 = dissipated_4;
                }
            }
        }
        st_3.kappa_c_0 = max(st_3.kappa_c_0, psi_contact_0);
    }
    else
    {
        dissipated_3 = 0.0f;
    }
    var dmg_0 : f32 = st_3.damage_0;
    var _S206 : vec3<f32> = vec3<f32>(0.0f);
    var qc_ang_0 : vec3<f32>;
    if((st_3.damage_0) == 0.0f)
    {
        if(((flags_0 & (u32(8)))) == u32(0))
        {
            var _S207 : vec3<f32> = contact_offsets_0(&((*mat_8)), &((*b_18)), st_3.crush_1, plastic_4, d_lin_3, d_ang_3);
            st_3.plastic_x_0 = _S207.x;
            st_3.plastic_y_0 = _S207.y;
            st_3.plastic_t_0 = _S207.z;
        }
        diss_contact_0 = 0.0f;
        qc_lin_0 = _S206;
        qc_ang_0 = _S206;
        psi_contact_0 = 0.0f;
    }
    else
    {
        var _S208 : Contact_0 = contact_part_0(&((*mat_8)), &((*b_18)), st_3.crush_1, plastic_4, d_lin_3, d_ang_3);
        st_3.plastic_x_0 = _S208.plastic_0.x;
        st_3.plastic_y_0 = _S208.plastic_0.y;
        st_3.plastic_t_0 = _S208.plastic_0.z;
        diss_contact_0 = _S208.diss_1;
        qc_lin_0 = _S208.q_lin_1;
        qc_ang_0 = _S208.q_ang_1;
        psi_contact_0 = _S208.energy_5;
    }
    var dissipated_6 : f32 = dissipated_3 + dmg_0 * diss_contact_0;
    if(_S190)
    {
        intact_normal_0 = kn_3 * _S189;
    }
    else
    {
        intact_normal_0 = (1.0f - st_3.crush_1) * kn_3 * _S189;
    }
    var _S209 : f32 = 1.0f - dmg_0;
    var force_lin_2 : vec3<f32> = vec3<f32>(_S209 * qe_lin_0.x + dmg_0 * qc_lin_0.x, _S209 * qe_lin_0.y + dmg_0 * qc_lin_0.y, _S209 * intact_normal_0 + dmg_0 * qc_lin_0.z);
    var force_ang_2 : vec3<f32> = qe_ang_0 * vec3<f32>(_S209) + qc_ang_0 * vec3<f32>(dmg_0);
    var stored_3 : f32 = _S209 * (psi_ts_0 + (1.0f - st_3.crush_1) * psi_c_0) + dmg_0 * psi_contact_0;
    if(has_rebar_2)
    {
        changed_0 = (st_3.rebar_broken_0) == 0.0f;
    }
    else
    {
        changed_0 = false;
    }
    var stored_4 : f32;
    var force_lin_3 : vec3<f32>;
    if(changed_0)
    {
        var k_axial_0 : f32 = (*b_18).rebar0_0.x;
        var k_dowel_0 : f32 = (*b_18).rebar0_0.y;
        var yield_force_0 : f32 = (*b_18).rebar0_0.z;
        var dowel_capacity_0 : f32 = (*b_18).rebar0_0.w;
        var nr_0 : vec2<f32> = return_map_0(k_axial_0, _S189, st_3.rebar_plastic_0, yield_force_0);
        var v1_0 : vec2<f32> = return_map_0(k_dowel_0, _S185, st_3.rebar_slip0_0, dowel_capacity_0);
        var v2_0 : vec2<f32> = return_map_0(k_dowel_0, _S186, st_3.rebar_slip1_0, dowel_capacity_0);
        var _S210 : f32 = nr_0.y;
        var _S211 : f32 = v1_0.y;
        var _S212 : f32 = v2_0.y;
        var work_0 : f32 = yield_force_0 * abs(_S210) + dowel_capacity_0 * (abs(_S211) + abs(_S212));
        st_3.rebar_plastic_0 = st_3.rebar_plastic_0 + _S210;
        st_3.rebar_slip0_0 = st_3.rebar_slip0_0 + _S211;
        st_3.rebar_slip1_0 = st_3.rebar_slip1_0 + _S212;
        st_3.rebar_work_0 = st_3.rebar_work_0 + work_0;
        var dissipated_7 : f32 = dissipated_6 + work_0;
        var _S213 : f32 = nr_0.x;
        var _S214 : f32 = v1_0.x;
        var _S215 : f32 = v2_0.x;
        var elastic_0 : f32 = 0.5f * (fdiv_0(sq_0(_S213), k_axial_0) + fdiv_0(sq_0(_S214) + sq_0(_S215), k_dowel_0));
        if(fracture_1)
        {
            changed_0 = (st_3.rebar_work_0) >= ((*b_18).rebar1_0.x);
        }
        else
        {
            changed_0 = false;
        }
        if(changed_0)
        {
            st_3.rebar_broken_0 = 1.0f;
            var dissipated_8 : f32 = dissipated_7 + elastic_0;
            force_lin_3 = force_lin_2;
            stored_4 = stored_3;
            dissipated_3 = dissipated_8;
        }
        else
        {
            var stored_5 : f32 = stored_3 + elastic_0;
            force_lin_3 = force_lin_2 + vec3<f32>(_S214, _S215, _S213);
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
        changed_0 = _S179;
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
        var _S216 : Measures_0 = stress_measures_0(&((*b_18)), force_lin_3, force_ang_2);
        var _S217 : vec4<f32> = failure_indices_0(&((*mat_8)), &((*b_18)), _S216, weibull_0);
        var _S218 : f32 = life_rate_0(&((*mat_8)), max(max(_S217.x, _S217.y), _S217.z));
        st_3.life_0 = max(st_3.life_0 - _S218 * dt_6, 0.0f);
    }
    if((st_3.damage_0) != _S192)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_3.crush_1) != _S194;
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_3.plastic_x_0) != ((*state_2).plastic_x_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_3.plastic_y_0) != ((*state_2).plastic_y_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_3.plastic_t_0) != ((*state_2).plastic_t_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_3.rebar_plastic_0) != ((*state_2).rebar_plastic_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_3.rebar_slip0_0) != ((*state_2).rebar_slip0_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_3.rebar_slip1_0) != ((*state_2).rebar_slip1_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_3.rebar_broken_0) != ((*state_2).rebar_broken_0);
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
    var _S219 : f32 = max(min(dissipated_3, released_0), 0.0f);
    var overshoot_1 : f32 = released_0 - _S219;
    st_3.dissipated_0 = st_3.dissipated_0 + _S219;
    var resp_0 : JointResponse_0;
    resp_0.force_lin_1 = force_lin_3;
    resp_0.force_ang_1 = force_ang_2;
    resp_0.state_1 = st_3;
    resp_0.dissipated_2 = _S219;
    resp_0.overshoot_0 = overshoot_1;
    resp_0.stored_2 = stored_4;
    if(_S176)
    {
        changed_0 = !connected_1(st_3, has_rebar_2);
    }
    else
    {
        changed_0 = false;
    }
    resp_0.disconnected_0 = changed_0;
    resp_0.fatigue_before_0 = _S181;
    resp_0.measures_0 = _S177;
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

fn secant_factors_0( b_19 : ptr<function, JointBond_std430_0>,  st_4 : JointState_0,  d_lin_4 : vec3<f32>,  f_lin_0 : ptr<function, vec3<f32>>,  f_ang_0 : ptr<function, vec3<f32>>)
{
    var compressed_0 : bool = (d_lin_4.z) < 0.0f;
    var contact_2 : f32;
    if(compressed_0)
    {
        contact_2 = st_4.damage_0;
    }
    else
    {
        contact_2 = 0.0f;
    }
    var _S220 : f32 = 1.0f - st_4.damage_0;
    var _S221 : f32 = max(_S220 + contact_2, 9.99999997475242708e-07f);
    var normal_2 : f32;
    if(compressed_0)
    {
        normal_2 = max(1.0f - st_4.crush_1, 9.99999997475242708e-07f);
    }
    else
    {
        normal_2 = max(_S220, 9.99999997475242708e-07f);
    }
    (*f_lin_0) = vec3<f32>(_S221, _S221, normal_2);
    (*f_ang_0) = vec3<f32>(_S221);
    var _S222 : bool;
    if(((*b_19).stiff1_0.w) != 0.0f)
    {
        _S222 = (st_4.rebar_broken_0) == 0.0f;
    }
    else
    {
        _S222 = false;
    }
    if(_S222)
    {
        var _S223 : vec4<f32> = (*b_19).rebar0_0;
        var _S224 : vec4<f32> = (*b_19).stiff0_0;
        (*f_lin_0)[i32(2)] = (*f_lin_0)[i32(2)] + fdiv_0((*b_19).rebar0_0.x, (*b_19).stiff0_0.x);
        var _S225 : f32 = fdiv_0(_S223.y, _S224.y);
        (*f_lin_0)[i32(0)] = (*f_lin_0)[i32(0)] + _S225;
        (*f_lin_0)[i32(1)] = (*f_lin_0)[i32(1)] + _S225;
    }
    return;
}

fn to_body_0( _S226 : u32,  _S227 : vec3<f32>) -> vec3<f32>
{
    return bonds_0[_S226].t1_0.xyz * vec3<f32>(_S227.x) + bonds_0[_S226].t2_0.xyz * vec3<f32>(_S227.y) + bonds_0[_S226].normal_0.xyz * vec3<f32>(_S227.z);
}

fn comp_add1_1( sum_1 : ptr<function, f32>,  err_1 : ptr<function, f32>,  x_6 : f32)
{
    var t_9 : f32 = (*sum_1) + x_6;
    if((abs((*sum_1))) >= (abs(x_6)))
    {
        (*err_1) = (*err_1) + ((*sum_1) - t_9 + x_6);
    }
    else
    {
        (*err_1) = (*err_1) + (x_6 - t_9 + (*sum_1));
    }
    (*sum_1) = t_9;
    return;
}

fn is_damaged_1( st_5 : JointState_0) -> bool
{
    var _S228 : bool;
    if((st_5.damage_0) > 0.0f)
    {
        _S228 = true;
    }
    else
    {
        _S228 = (st_5.crush_1) > 0.0f;
    }
    return _S228;
}

fn fatigue_factor_1( mat_9 : ptr<function, JointMaterial_std140_0>,  life_2 : f32) -> f32
{
    if(((((*mat_9).kind_flags_0.y) & (u32(64)))) == u32(0))
    {
        return 1.0f;
    }
    return fpow_0(clamp(life_2, 0.0f, 1.0f), fdiv_0(1.0f, (*mat_9).misc_0.y - 2.0f));
}

fn bond_update_0( i_5 : u32,  dt_7 : f32,  fracture_2 : bool,  abs_step_0 : u32,  wide_0 : bool) -> bool
{
    var ra_1 : vec3<f32> = bonds_0[i_5].ra_0.xyz;
    var rb_1 : vec3<f32> = bonds_0[i_5].rb_0.xyz;
    var _S229 : u32 = u32(4) * bonds_0[i_5].law_0.ids_0.y;
    var ta_1 : vec3<f32> = state_0[_S229 + u32(1)].xyz;
    var wa_0 : vec3<f32> = state_0[_S229 + u32(3)].xyz;
    var _S230 : u32 = u32(4) * bonds_0[i_5].law_0.ids_0.z;
    var tb_1 : vec3<f32> = state_0[_S230 + u32(1)].xyz;
    var wb_0 : vec3<f32> = state_0[_S230 + u32(3)].xyz;
    var _S231 : vec3<f32> = to_local_0(i_5, state_0[_S230].xyz + cross(tb_1, rb_1) - (state_0[_S229].xyz + cross(ta_1, ra_1)));
    var _S232 : vec3<f32> = to_local_0(i_5, tb_1 - ta_1);
    var _S233 : vec3<f32> = to_local_0(i_5, state_0[_S230 + u32(2)].xyz + cross(wb_0, rb_1) - (state_0[_S229 + u32(2)].xyz + cross(wa_0, ra_1)));
    var _S234 : vec3<f32> = to_local_0(i_5, wb_0 - wa_0);
    var _S235 : JointMaterial_std140_0 = materials_0.m_0.data_0[bonds_0[i_5].law_0.ids_0.x];
    var _S236 : JointState_std430_0 = bond_dyn_0[i_5].js_0;
    var _S237 : bool = is_damaged_0(&(_S236));
    var _S238 : f32 = _S236.life_0;
    var _S239 : JointBond_std430_0 = bonds_0[i_5].law_0;
    var _S240 : JointResponse_0 = joint_evaluate_0(&(_S235), &(_S239), &(_S236), _S231, _S232, dt_7, fracture_2);
    var _S241 : JointState_0 = JointState_0( bond_dyn_0[i_5].js_0.damage_0, bond_dyn_0[i_5].js_0.crush_1, bond_dyn_0[i_5].js_0.kappa_0, bond_dyn_0[i_5].js_0.kappa_c_0, bond_dyn_0[i_5].js_0.ductility_0, bond_dyn_0[i_5].js_0.ductility_c_0, bond_dyn_0[i_5].js_0.life_0, bond_dyn_0[i_5].js_0.plastic_x_0, bond_dyn_0[i_5].js_0.plastic_y_0, bond_dyn_0[i_5].js_0.plastic_t_0, bond_dyn_0[i_5].js_0.rebar_plastic_0, bond_dyn_0[i_5].js_0.rebar_slip0_0, bond_dyn_0[i_5].js_0.rebar_slip1_0, bond_dyn_0[i_5].js_0.rebar_work_0, bond_dyn_0[i_5].js_0.rebar_broken_0, bond_dyn_0[i_5].js_0.strain_rate_0, bond_dyn_0[i_5].js_0.governing_stress_0, bond_dyn_0[i_5].js_0.dissipated_0, bond_dyn_0[i_5].js_0.utilization_0, bond_dyn_0[i_5].js_0.mode_0 );
    var bd_0 : BondDyn_0;
    bd_0.js_0 = _S241;
    bd_0.force_lin_0 = bond_dyn_0[i_5].force_lin_0;
    bd_0.force_ang_0 = bond_dyn_0[i_5].force_ang_0;
    bd_0.sums_0 = bond_dyn_0[i_5].sums_0;
    bd_0.comps_0 = bond_dyn_0[i_5].comps_0;
    bd_0.events_0 = bond_dyn_0[i_5].events_0;
    var ra_2 : vec3<f32> = bonds_0[i_5].ra_0.xyz;
    var rb_2 : vec3<f32> = bonds_0[i_5].rb_0.xyz;
    var _S242 : JointBond_std430_0 = bonds_0[i_5].law_0;
    var f_lin_1 : vec3<f32>;
    var f_ang_1 : vec3<f32>;
    secant_factors_0(&(_S242), _S240.state_1, _S231, &(f_lin_1), &(f_ang_1));
    var qd_lin_0 : vec3<f32> = _S233 * bonds_0[i_5].c_lin_0.xyz * f_lin_1;
    var qd_ang_0 : vec3<f32> = _S234 * bonds_0[i_5].c_ang_0.xyz * f_ang_1;
    var q_lin_2 : vec3<f32> = _S240.force_lin_1 + qd_lin_0;
    var q_ang_2 : vec3<f32> = _S240.force_ang_1 + qd_ang_0;
    var damped_0 : f32 = (dot(qd_lin_0, _S233) + dot(qd_ang_0, _S234)) * dt_7;
    var _S243 : vec3<f32> = to_body_0(i_5, q_lin_2);
    var _S244 : vec3<f32> = to_body_0(i_5, q_ang_2);
    var _S245 : u32 = u32(3) * i_5;
    scratch_0[_S245] = vec4<f32>(_S243, max(_S240.measures_0.tension_0, _S240.measures_0.compression_0));
    scratch_0[_S245 + u32(1)] = vec4<f32>(_S244 + cross(ra_2, _S243), 0.0f);
    scratch_0[_S245 + u32(2)] = vec4<f32>((vec3<f32>(0) - _S244) + cross(rb_2, (vec3<f32>(0) - _S243)), 0.0f);
    var _S246 : f32 = bd_0.sums_0[i32(0)];
    var _S247 : f32 = bd_0.comps_0[i32(0)];
    comp_add1_1(&(_S246), &(_S247), _S240.dissipated_2);
    bd_0.sums_0[i32(0)] = _S246;
    bd_0.comps_0[i32(0)] = _S247;
    var _S248 : f32 = bd_0.sums_0[i32(1)];
    var _S249 : f32 = bd_0.comps_0[i32(1)];
    comp_add1_1(&(_S248), &(_S249), _S240.overshoot_0);
    bd_0.sums_0[i32(1)] = _S248;
    bd_0.comps_0[i32(1)] = _S249;
    var _S250 : f32 = bd_0.sums_0[i32(2)];
    var _S251 : f32 = bd_0.comps_0[i32(2)];
    comp_add1_1(&(_S250), &(_S251), damped_0);
    bd_0.sums_0[i32(2)] = _S250;
    bd_0.comps_0[i32(2)] = _S251;
    bd_0.force_lin_0 = vec4<f32>(q_lin_2, _S240.stored_2);
    bd_0.force_ang_0 = vec4<f32>(q_ang_2, max(bd_0.force_ang_0.w, _S240.state_1.utilization_0));
    var _S252 : bool;
    if(!_S237)
    {
        _S252 = is_damaged_1(_S240.state_1);
    }
    else
    {
        _S252 = false;
    }
    if(_S252)
    {
        _S252 = (bd_0.events_0.x) == u32(0);
    }
    else
    {
        _S252 = false;
    }
    if(_S252)
    {
        bd_0.events_0[i32(0)] = abs_step_0;
        bd_0.events_0[i32(3)] = _S240.state_1.mode_0;
        if(wide_0)
        {
            cov_or_0(_S242.ids_0.w, u32(32));
        }
        else
        {
            g_seen_0[u32(3)] = u32(1);
        }
    }
    if((bd_0.events_0.y) == u32(0))
    {
        _S252 = (_S240.state_1.life_0) != _S238;
    }
    else
    {
        _S252 = false;
    }
    if(_S252)
    {
        _S252 = (_S240.fatigue_before_0) > 0.99000000953674316f;
    }
    else
    {
        _S252 = false;
    }
    if(_S252)
    {
        var _S253 : f32 = fatigue_factor_1(&(_S235), _S240.state_1.life_0);
        _S252 = _S253 <= 0.99000000953674316f;
    }
    else
    {
        _S252 = false;
    }
    if(_S252)
    {
        bd_0.events_0[i32(1)] = abs_step_0;
        if(wide_0)
        {
            cov_or_0(_S242.ids_0.w, u32(64));
        }
        else
        {
            g_seen_0[u32(4)] = u32(1);
        }
    }
    if(_S240.disconnected_0)
    {
        bd_0.events_0[i32(2)] = abs_step_0;
    }
    bd_0.js_0 = _S240.state_1;
    bond_dyn_0[i_5].js_0.damage_0 = bd_0.js_0.damage_0;
    bond_dyn_0[i_5].js_0.crush_1 = bd_0.js_0.crush_1;
    bond_dyn_0[i_5].js_0.kappa_0 = bd_0.js_0.kappa_0;
    bond_dyn_0[i_5].js_0.kappa_c_0 = bd_0.js_0.kappa_c_0;
    bond_dyn_0[i_5].js_0.ductility_0 = bd_0.js_0.ductility_0;
    bond_dyn_0[i_5].js_0.ductility_c_0 = bd_0.js_0.ductility_c_0;
    bond_dyn_0[i_5].js_0.life_0 = bd_0.js_0.life_0;
    bond_dyn_0[i_5].js_0.plastic_x_0 = bd_0.js_0.plastic_x_0;
    bond_dyn_0[i_5].js_0.plastic_y_0 = bd_0.js_0.plastic_y_0;
    bond_dyn_0[i_5].js_0.plastic_t_0 = bd_0.js_0.plastic_t_0;
    bond_dyn_0[i_5].js_0.rebar_plastic_0 = bd_0.js_0.rebar_plastic_0;
    bond_dyn_0[i_5].js_0.rebar_slip0_0 = bd_0.js_0.rebar_slip0_0;
    bond_dyn_0[i_5].js_0.rebar_slip1_0 = bd_0.js_0.rebar_slip1_0;
    bond_dyn_0[i_5].js_0.rebar_work_0 = bd_0.js_0.rebar_work_0;
    bond_dyn_0[i_5].js_0.rebar_broken_0 = bd_0.js_0.rebar_broken_0;
    bond_dyn_0[i_5].js_0.strain_rate_0 = bd_0.js_0.strain_rate_0;
    bond_dyn_0[i_5].js_0.governing_stress_0 = bd_0.js_0.governing_stress_0;
    bond_dyn_0[i_5].js_0.dissipated_0 = bd_0.js_0.dissipated_0;
    bond_dyn_0[i_5].js_0.utilization_0 = bd_0.js_0.utilization_0;
    bond_dyn_0[i_5].js_0.mode_0 = bd_0.js_0.mode_0;
    bond_dyn_0[i_5].force_lin_0 = bd_0.force_lin_0;
    bond_dyn_0[i_5].force_ang_0 = bd_0.force_ang_0;
    bond_dyn_0[i_5].sums_0 = bd_0.sums_0;
    bond_dyn_0[i_5].comps_0 = bd_0.comps_0;
    bond_dyn_0[i_5].events_0 = bd_0.events_0;
    return _S240.disconnected_0;
}

fn chunk_update_0( c_10 : u32,  isl_4 : Island_0,  rg_3 : Rigid_0,  dt_8 : f32,  rml_0 : bool,  step_0 : u32,  contact_3 : bool,  work_1 : ptr<function, f32>,  work_err_0 : ptr<function, f32>)
{
    var _S254 : vec3<f32> = vec3<f32>(0.0f);
    var _S255 : u32 = index_0[c_10];
    var peak_0 : f32 = 0.0f;
    var e_1 : u32 = _S255;
    var fi_0 : vec3<f32> = _S254;
    var mi_0 : vec3<f32> = _S254;
    loop
    {
        if(e_1 < index_0[c_10 + u32(1)])
        {
        }
        else
        {
            break;
        }
        var entry_1 : u32 = index_0[e_1];
        var _S256 : u32 = u32(3) * ((entry_1 >> (u32(1))));
        var fa_0 : vec4<f32> = scratch_0[_S256];
        if(((entry_1 & (u32(1)))) == u32(0))
        {
            var mi_1 : vec3<f32> = mi_0 + scratch_0[_S256 + u32(1)].xyz;
            fi_0 = fi_0 + fa_0.xyz;
            mi_0 = mi_1;
        }
        else
        {
            var mi_2 : vec3<f32> = mi_0 + scratch_0[_S256 + u32(2)].xyz;
            fi_0 = fi_0 + (vec3<f32>(0) - fa_0.xyz);
            mi_0 = mi_2;
        }
        var _S257 : f32 = max(peak_0, fa_0.w);
        var _S258 : u32 = e_1 + u32(1);
        peak_0 = _S257;
        e_1 = _S258;
    }
    var _S259 : u32 = u32(4) * c_10;
    var u_0 : vec3<f32> = state_0[_S259].xyz;
    var _S260 : u32 = _S259 + u32(1);
    var th_1 : vec3<f32> = state_0[_S260].xyz;
    var _S261 : u32 = _S259 + u32(2);
    var v_7 : vec3<f32> = state_0[_S261].xyz;
    var _S262 : u32 = _S259 + u32(3);
    var w_4 : vec3<f32> = state_0[_S262].xyz;
    var mass_0 : f32 = chunks_0[c_10].center_0.w;
    var _S263 : vec3<f32> = chunks_0[c_10].center_0.xyz;
    var _S264 : vec3<f32> = isl_4.com_0.xyz;
    var r_world_0 : vec3<f32> = rotate_0(rg_3.rot_0, _S263 + u_0 - _S264);
    var f_load_0 : vec3<f32>;
    var t_load_0 : vec3<f32>;
    var rep_f_1 : vec3<f32>;
    var rep_m_1 : vec3<f32>;
    chunk_loads_1(c_10, c_10, rg_3.rot_0, step_0, dt_8, contact_3, dt_8, &(f_load_0), &(t_load_0), &(rep_f_1), &(rep_m_1));
    record_chunk_load_0(c_10, f_load_0, t_load_0);
    var _S265 : vec3<f32> = vec3<f32>(mass_0);
    var f_world_0 : vec3<f32> = f_load_0 + params_0.gravity_0.xyz * _S265;
    var t_world_0 : vec3<f32> = t_load_0;
    var f_world_1 : vec3<f32>;
    var t_world_1 : vec3<f32>;
    if(rml_0)
    {
        var t_world_2 : vec3<f32> = t_world_0 - (world_mul_0(rg_3.rot_0, chunks_0[c_10].inertia0_1, chunks_0[c_10].inertia1_1, chunks_0[c_10].inertia2_1, rg_3.alpha_0) + cross(rg_3.w_1, world_mul_0(rg_3.rot_0, chunks_0[c_10].inertia0_1, chunks_0[c_10].inertia1_1, chunks_0[c_10].inertia2_1, rg_3.w_1)));
        f_world_1 = f_world_0 - (rg_3.a_0 + cross(rg_3.alpha_0, r_world_0) + cross(rg_3.w_1, cross(rg_3.w_1, r_world_0))) * _S265;
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
        var wb_1 : vec3<f32> = inverse_rotate_0(rg_3.rot_0, rg_3.w_1);
        var i_w_0 : vec3<f32> = rows_mul_0(chunks_0[c_10].inertia0_1, chunks_0[c_10].inertia1_1, chunks_0[c_10].inertia2_1, w_4);
        var m_ext_2 : vec3<f32> = m_ext_0 - (cross(wb_1, i_w_0) + cross(w_4, rows_mul_0(chunks_0[c_10].inertia0_1, chunks_0[c_10].inertia1_1, chunks_0[c_10].inertia2_1, wb_1)) + cross(w_4, i_w_0));
        f_ext_1 = f_ext_0 - cross(wb_1, v_7) * vec3<f32>((2.0f * mass_0));
        m_ext_1 = m_ext_2;
    }
    else
    {
        f_ext_1 = f_ext_0;
        m_ext_1 = m_ext_0;
    }
    var f_9 : vec3<f32> = f_ext_1 + rep_f_1 + fi_0;
    var m_3 : vec3<f32> = m_ext_1 + rep_m_1 + mi_0;
    var support_0 : u32 = chunks_0[c_10].info_1.x;
    var _S266 : vec3<f32> = vec3<f32>(state_0[_S260].w, state_0[_S261].w, state_0[_S262].w);
    var reaction_0 : vec3<f32>;
    var u_1 : vec3<f32>;
    var th_2 : vec3<f32>;
    var v_8 : vec3<f32>;
    var w_5 : vec3<f32>;
    if(support_0 == u32(1))
    {
        reaction_0 = (vec3<f32>(0) - f_9);
        u_1 = u_0;
        th_2 = th_1;
        v_8 = _S254;
        w_5 = _S254;
    }
    else
    {
        var _S267 : vec4<f32> = chunks_0[c_10].scale_0;
        var w_6 : vec3<f32> = w_4 + rows_mul_0(chunks_0[c_10].inv0_1, chunks_0[c_10].inv1_1, chunks_0[c_10].inv2_1, m_3) * vec3<f32>((dt_8 * chunks_0[c_10].scale_0.z));
        var _S268 : vec3<f32> = vec3<f32>(dt_8);
        var th_3 : vec3<f32> = th_1 + w_6 * _S268;
        if(support_0 == u32(2))
        {
            reaction_0 = (vec3<f32>(0) - f_9);
            u_1 = u_0;
            th_2 = _S254;
        }
        else
        {
            var v_9 : vec3<f32> = v_7 + f_9 * vec3<f32>((dt_8 * _S267.y));
            var u_2 : vec3<f32> = u_0 + v_9 * _S268;
            reaction_0 = _S266;
            u_1 = u_2;
            th_2 = v_9;
        }
        var _S269 : vec3<f32> = th_2;
        th_2 = th_3;
        v_8 = _S269;
        w_5 = w_6;
    }
    state_0[_S259] = vec4<f32>(u_1, peak_0);
    state_0[_S260] = vec4<f32>(th_2, reaction_0.x);
    state_0[_S261] = vec4<f32>(v_8, reaction_0.y);
    state_0[_S262] = vec4<f32>(w_5, reaction_0.z);
    comp_add1_0(&((*work_1)), &((*work_err_0)), (dot(f_load_0, rg_3.vel_0 + rg_3.vel_err_0 + cross(rg_3.w_1, rotate_0(rg_3.rot_0, _S263 + u_1 - _S264)) + rotate_0(rg_3.rot_0, v_8)) + dot(t_load_0, rg_3.w_1 + rotate_0(rg_3.rot_0, w_5))) * dt_8);
    return;
}

fn comp_add_0( sum_2 : ptr<function, vec3<f32>>,  err_2 : ptr<function, vec3<f32>>,  x_7 : vec3<f32>)
{
    var t_10 : vec3<f32> = (*sum_2) + x_7;
    var _S270 : vec3<f32> = abs(x_7);
    (*err_2) = (*err_2) + (select(x_7, (*sum_2), (abs((*sum_2))) >= _S270) - t_10 + select((*sum_2), x_7, (abs((*sum_2))) >= _S270));
    (*sum_2) = t_10;
    return;
}

fn turn_difference_0( omega_0 : vec3<f32>,  dt_9 : f32,  v_10 : vec3<f32>) -> vec3<f32>
{
    var _S271 : f32 = length(omega_0);
    var angle_1 : f32 = _S271 * dt_9;
    if(angle_1 < 1.00000000317107685e-30f)
    {
        return vec3<f32>(0.0f);
    }
    var a_12 : vec3<f32> = omega_0 / vec3<f32>(_S271);
    var s_5 : f32 = sin(0.5f * angle_1);
    var av_0 : vec3<f32> = cross(a_12, v_10);
    return av_0 * vec3<f32>(sin(angle_1)) + cross(a_12, av_0) * vec3<f32>((2.0f * s_5 * s_5));
}

fn turn_minus_one_0( axis_2 : vec3<f32>,  angle_2 : f32) -> vec4<f32>
{
    var s_6 : f32 = sin(0.25f * angle_2);
    return vec4<f32>(axis_2 * vec3<f32>(sin(0.5f * angle_2)), -2.0f * s_6 * s_6);
}

fn quat_mul_0( a_13 : Quat_0,  o_1 : Quat_0) -> Quat_0
{
    var r_6 : Quat_0;
    r_6.w_0 = a_13.w_0 * o_1.w_0 - a_13.x_0 * o_1.x_0 - a_13.y_0 * o_1.y_0 - a_13.z_0 * o_1.z_0;
    r_6.x_0 = a_13.w_0 * o_1.x_0 + a_13.x_0 * o_1.w_0 + a_13.y_0 * o_1.z_0 - a_13.z_0 * o_1.y_0;
    r_6.y_0 = a_13.w_0 * o_1.y_0 - a_13.x_0 * o_1.z_0 + a_13.y_0 * o_1.w_0 + a_13.z_0 * o_1.x_0;
    r_6.z_0 = a_13.w_0 * o_1.z_0 + a_13.x_0 * o_1.y_0 - a_13.y_0 * o_1.x_0 + a_13.z_0 * o_1.w_0;
    return r_6;
}

fn quat_vec_0( q_8 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_8.x_0, q_8.y_0, q_8.z_0, q_8.w_0);
}

fn comp_add4_0( sum_3 : ptr<function, vec4<f32>>,  err_3 : ptr<function, vec4<f32>>,  x_8 : vec4<f32>)
{
    var t_11 : vec4<f32> = (*sum_3) + x_8;
    var _S272 : vec4<f32> = abs(x_8);
    (*err_3) = (*err_3) + (select(x_8, (*sum_3), (abs((*sum_3))) >= _S272) - t_11 + select((*sum_3), x_8, (abs((*sum_3))) >= _S272));
    (*sum_3) = t_11;
    return;
}

fn quat_accumulate_0( hi_1 : ptr<function, Quat_0>,  lo_1 : ptr<function, vec4<f32>>,  x_9 : vec4<f32>)
{
    var sum_4 : vec4<f32> = quat_vec_0((*hi_1));
    comp_add4_0(&(sum_4), &((*lo_1)), x_9);
    var t_12 : vec4<f32> = sum_4 + (*lo_1);
    (*lo_1) = (*lo_1) - (t_12 - sum_4);
    (*hi_1) = quat_of_0(t_12);
    return;
}

fn turn_left_0( hi_2 : ptr<function, Quat_0>,  lo_2 : ptr<function, vec4<f32>>,  omega_1 : vec3<f32>,  dt_10 : f32)
{
    var _S273 : f32 = length(omega_1);
    var angle_3 : f32 = _S273 * dt_10;
    if(angle_3 < 1.00000000317107685e-30f)
    {
        return;
    }
    var d_2 : vec4<f32> = turn_minus_one_0(omega_1 / vec3<f32>(_S273), angle_3);
    var dq_0 : Quat_0;
    dq_0.x_0 = d_2.x;
    dq_0.y_0 = d_2.y;
    dq_0.z_0 = d_2.z;
    dq_0.w_0 = d_2.w;
    quat_accumulate_0(&((*hi_2)), &((*lo_2)), quat_vec_0(quat_mul_0(dq_0, (*hi_2))));
    return;
}

fn integrate_rigid_0( isl_5 : Island_0,  rg_4 : ptr<function, Rigid_0>,  dt_11 : f32)
{
    var _S274 : vec3<f32> = vec3<f32>(dt_11);
    var _S275 : vec3<f32> = (*rg_4).torque_0 * _S274;
    var _S276 : vec3<f32> = (*rg_4).l_0;
    var _S277 : vec3<f32> = (*rg_4).l_err_0;
    comp_add_0(&(_S276), &(_S277), _S275);
    (*rg_4).l_0 = _S276;
    (*rg_4).l_err_0 = _S277;
    var l_1 : vec3<f32> = _S276 + _S277;
    var _S278 : vec3<f32> = (*rg_4).a_0 * _S274;
    var _S279 : vec3<f32> = (*rg_4).vel_0;
    var _S280 : vec3<f32> = (*rg_4).vel_err_0;
    comp_add_0(&(_S279), &(_S280), _S278);
    (*rg_4).vel_0 = _S279;
    (*rg_4).vel_err_0 = _S280;
    var w_mid_0 : vec3<f32> = world_mul_0((*rg_4).rot_0, isl_5.inv0_0, isl_5.inv1_0, isl_5.inv2_0, l_1);
    var delta_0 : vec3<f32> = (_S279 + _S280) * _S274 - turn_difference_0(w_mid_0, dt_11, rotate_0((*rg_4).rot_0, isl_5.com_0.xyz));
    var _S281 : vec3<f32> = (*rg_4).pos_0;
    var _S282 : vec3<f32> = (*rg_4).pos_err_0;
    comp_add_0(&(_S281), &(_S282), delta_0);
    (*rg_4).pos_0 = _S281;
    (*rg_4).pos_err_0 = _S282;
    var _S283 : Quat_0 = (*rg_4).rot_0;
    var _S284 : vec4<f32> = (*rg_4).rot_err_0;
    turn_left_0(&(_S283), &(_S284), w_mid_0, dt_11);
    (*rg_4).rot_0 = _S283;
    (*rg_4).rot_err_0 = _S284;
    (*rg_4).w_1 = world_mul_0(_S283, isl_5.inv0_0, isl_5.inv1_0, isl_5.inv2_0, l_1);
    return;
}

fn drift_moments_0( c_11 : u32,  tu_0 : ptr<function, vec3<f32>>,  pv_0 : ptr<function, vec3<f32>>)
{
    var _S285 : u32 = u32(4) * c_11;
    var _S286 : vec3<f32> = vec3<f32>((chunks_0[c_11].center_0.w * chunks_0[c_11].scale_0.x));
    (*tu_0) = (*tu_0) + state_0[_S285].xyz * _S286;
    (*pv_0) = (*pv_0) + state_0[_S285 + u32(2)].xyz * _S286;
    return;
}

fn drift_angular_0( c_12 : u32,  wcom_1 : vec3<f32>,  tr_0 : vec3<f32>,  dv_0 : vec3<f32>,  lu_0 : ptr<function, vec3<f32>>,  lv_0 : ptr<function, vec3<f32>>)
{
    var r_7 : vec3<f32> = chunks_0[c_12].center_0.xyz - wcom_1;
    var _S287 : u32 = u32(4) * c_12;
    var _S288 : vec3<f32> = vec3<f32>(chunks_0[c_12].center_0.w);
    var _S289 : vec4<f32> = chunks_0[c_12].inertia0_1;
    var _S290 : vec4<f32> = chunks_0[c_12].inertia1_1;
    var _S291 : vec4<f32> = chunks_0[c_12].inertia2_1;
    var _S292 : vec3<f32> = vec3<f32>(chunks_0[c_12].scale_0.x);
    (*lu_0) = (*lu_0) + (cross(r_7, state_0[_S287].xyz - tr_0) * _S288 + rows_mul_0(chunks_0[c_12].inertia0_1, chunks_0[c_12].inertia1_1, chunks_0[c_12].inertia2_1, state_0[_S287 + u32(1)].xyz)) * _S292;
    (*lv_0) = (*lv_0) + (cross(r_7, state_0[_S287 + u32(2)].xyz - dv_0) * _S288 + rows_mul_0(_S289, _S290, _S291, state_0[_S287 + u32(3)].xyz)) * _S292;
    return;
}

fn drift_apply_0( c_13 : u32,  wcom_2 : vec3<f32>,  tr_1 : vec3<f32>,  phi_0 : vec3<f32>,  dv_1 : vec3<f32>,  dw_0 : vec3<f32>)
{
    var r_8 : vec3<f32> = chunks_0[c_13].center_0.xyz - wcom_2;
    var _S293 : u32 = u32(4) * c_13;
    state_0[_S293] = vec4<f32>(state_0[_S293].xyz - (tr_1 + cross(phi_0, r_8)), state_0[_S293].w);
    var _S294 : u32 = _S293 + u32(1);
    state_0[_S294] = vec4<f32>(state_0[_S294].xyz - phi_0, state_0[_S294].w);
    var _S295 : u32 = _S293 + u32(2);
    state_0[_S295] = vec4<f32>(state_0[_S295].xyz - (dv_1 + cross(dw_0, r_8)), state_0[_S295].w);
    var _S296 : u32 = _S293 + u32(3);
    state_0[_S296] = vec4<f32>(state_0[_S296].xyz - dw_0, state_0[_S296].w);
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
    var _S297 : vec3<f32> = rotate_0((*rg_5).rot_0, tr_2 - turn_difference_0(phi_2, 1.0f, wcom_3));
    var _S298 : vec3<f32> = (*rg_5).pos_0;
    var _S299 : vec3<f32> = (*rg_5).pos_err_0;
    comp_add_0(&(_S298), &(_S299), _S297);
    (*rg_5).pos_0 = _S298;
    (*rg_5).pos_err_0 = _S299;
    var b_20 : vec3<f32> = inverse_rotate_0((*rg_5).rot_0, (*rg_5).w_1);
    var dl_turn_0 : vec3<f32> = rotate_0((*rg_5).rot_0, cross(phi_2, rows_mul_0(isl_6.inertia0_0, isl_6.inertia1_0, isl_6.inertia2_0, b_20)) - rows_mul_0(isl_6.inertia0_0, isl_6.inertia1_0, isl_6.inertia2_0, cross(phi_2, b_20)));
    var _S300 : Quat_0 = (*rg_5).rot_0;
    var _S301 : vec4<f32> = (*rg_5).rot_err_0;
    turn_right_0(&(_S300), &(_S301), phi_2);
    (*rg_5).rot_0 = _S300;
    (*rg_5).rot_err_0 = _S301;
    var _S302 : vec3<f32> = rotate_0(rot_2, dv_2 + cross(dw_1, isl_6.com_0.xyz - wcom_3));
    var _S303 : vec3<f32> = (*rg_5).vel_0;
    var _S304 : vec3<f32> = (*rg_5).vel_err_0;
    comp_add_0(&(_S303), &(_S304), _S302);
    (*rg_5).vel_0 = _S303;
    (*rg_5).vel_err_0 = _S304;
    var _S305 : vec3<f32> = rotate_0(rot_2, dw_1);
    (*rg_5).w_1 = (*rg_5).w_1 + _S305;
    var _S306 : vec3<f32> = dl_turn_0 + world_mul_0(_S300, isl_6.inertia0_0, isl_6.inertia1_0, isl_6.inertia2_0, _S305);
    var _S307 : vec3<f32> = (*rg_5).l_0;
    var _S308 : vec3<f32> = (*rg_5).l_err_0;
    comp_add_0(&(_S307), &(_S308), _S306);
    (*rg_5).l_0 = _S307;
    (*rg_5).l_err_0 = _S308;
    return;
}

fn write_probe_0( slot_0 : u32,  k_10 : u32,  value_2 : f32)
{
    var at_1 : u32 = params_0.probe_base_0 * u32(4) + slot_0 * params_0.probe_stride_0 + k_10;
    var v_11 : vec4<f32> = scratch_0[at_1 / u32(4)];
    v_11[at_1 % u32(4)] = value_2;
    scratch_0[at_1 / u32(4)] = v_11;
    return;
}

fn record_probes_0( isl_7 : Island_0,  rg_6 : Rigid_0,  k_11 : u32)
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
        var a_14 : vec4<f32> = loads_0[at_2 + u32(1)];
        var b_21 : vec4<f32> = loads_0[at_2 + u32(2)];
        var c4_0 : vec4<f32> = loads_0[at_2 + u32(3)];
        var kind_4 : u32 = info_3.x;
        var i_6 : u32 = info_3.y;
        var value_3 : f32;
        if(kind_4 == u32(0))
        {
            value_3 = dot(rg_6.pos_0 - b_21.xyz + rotate_0(rg_6.rot_0, chunks_0[i_6].center_0.xyz) + (rg_6.pos_err_0 - c4_0.xyz + rotate_0(rg_6.rot_0, state_0[u32(4) * i_6].xyz + vec3<f32>(a_14.w, b_21.w, c4_0.w))), a_14.xyz);
        }
        else
        {
            if(kind_4 == u32(1))
            {
                var _S309 : u32 = u32(4) * i_6;
                value_3 = dot(rg_6.vel_0 + rg_6.vel_err_0 + cross(rg_6.w_1, rotate_0(rg_6.rot_0, chunks_0[i_6].center_0.xyz + state_0[_S309].xyz - isl_7.com_0.xyz)) + rotate_0(rg_6.rot_0, state_0[_S309 + u32(2)].xyz), a_14.xyz);
            }
            else
            {
                if(kind_4 == u32(2))
                {
                    var _S310 : u32 = u32(3) * i_6;
                    var f_10 : vec3<f32> = scratch_0[_S310].xyz;
                    var _S311 : bool = (info_3.z) == u32(0);
                    var mc_0 : vec3<f32>;
                    if(_S311)
                    {
                        mc_0 = scratch_0[_S310 + u32(1)].xyz;
                    }
                    else
                    {
                        mc_0 = scratch_0[_S310 + u32(2)].xyz;
                    }
                    var fc_3 : vec3<f32>;
                    if(_S311)
                    {
                        fc_3 = f_10;
                    }
                    else
                    {
                        fc_3 = (vec3<f32>(0) - f_10);
                    }
                    value_3 = dot(fc_3, a_14.xyz) + dot(mc_0, b_21.xyz);
                }
                else
                {
                    var _S312 : u32 = u32(4) * i_6;
                    value_3 = dot(rotate_0(rg_6.rot_0, vec3<f32>(state_0[_S312 + u32(1)].w, state_0[_S312 + u32(2)].w, state_0[_S312 + u32(3)].w)), a_14.xyz);
                }
            }
        }
        write_probe_0(info_3.w, k_11, value_3);
        at_2 = at_2 + u32(4);
    }
    return;
}

var<workgroup> g_label_narrow_0 : array<u32, i32(512)>;

var<workgroup> g_changed_narrow_0 : u32;

fn connected_2( st_6 : ptr<function, JointState_std430_0>,  has_rebar_3 : bool) -> bool
{
    var _S313 : bool;
    if(((*st_6).damage_0) < 1.0f)
    {
        _S313 = true;
    }
    else
    {
        if(has_rebar_3)
        {
            _S313 = ((*st_6).rebar_broken_0) == 0.0f;
        }
        else
        {
            _S313 = false;
        }
    }
    return _S313;
}

fn island_connected_narrow_0( tid_3 : u32,  ct_0 : u32,  isl_8 : Island_0) -> bool
{
    var _S314 : u32 = isl_8.range_0.x;
    var n_2 : u32 = isl_8.range_0.y - _S314;
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
    var c_14 : u32 = ct_0;
    loop
    {
        if(c_14 < n_2)
        {
        }
        else
        {
            break;
        }
        g_label_narrow_0[c_14] = c_14;
        c_14 = c_14 + u32(256);
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
        c_14 = ct_0;
        loop
        {
            if(c_14 < n_2)
            {
            }
            else
            {
                break;
            }
            var gc_0 : u32 = _S314 + c_14;
            var _S315 : u32 = index_0[gc_0];
            var l_2 : u32 = g_label_narrow_0[c_14];
            var e_2 : u32 = _S315;
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
                var _S316 : JointState_std430_0 = bond_dyn_0[bond_0].js_0;
                var _S317 : bool = connected_2(&(_S316), (bonds_0[bond_0].law_0.stiff1_0.w) != 0.0f);
                if(!_S317)
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
                l_2 = min(l_2, g_label_narrow_0[other_0 - _S314]);
                e_2 = e_2 + u32(1);
            }
            var _S318 : u32 = min(l_2, g_label_narrow_0[l_2]);
            if(_S318 < (g_label_narrow_0[c_14]))
            {
                g_label_narrow_0[c_14] = _S318;
                g_changed_narrow_0 = u32(1);
            }
            c_14 = c_14 + u32(256);
        }
        workgroupBarrier();
        var converged_1 : bool = g_changed_narrow_0 == u32(0);
        workgroupBarrier();
        if(converged_1)
        {
            converged_0 = converged_1;
            break;
        }
        var _S319 : u32 = pass_0 + u32(1);
        converged_0 = converged_1;
        pass_0 = _S319;
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
    c_14 = ct_0;
    loop
    {
        if(c_14 < n_2)
        {
        }
        else
        {
            break;
        }
        if((g_label_narrow_0[c_14]) != u32(0))
        {
            g_changed_narrow_0 = u32(1);
        }
        c_14 = c_14 + u32(256);
    }
    workgroupBarrier();
    return g_changed_narrow_0 == u32(0);
}

fn contact_split_at_0( at_3 : u32)
{
    var previous_0 : u32 = islands_0[params_0.halt_index_0].info_0.y;
    var _S320 : u32;
    if(previous_0 == u32(0))
    {
        _S320 = at_3;
    }
    else
    {
        _S320 = min(previous_0, at_3);
    }
    islands_0[params_0.halt_index_0].info_0[i32(1)] = _S320;
    return;
}

fn comp_add1_2( sum_5 : ptr<function, f32>,  err_4 : ptr<function, f32>,  x_10 : f32)
{
    var t_13 : f32 = (*sum_5) + x_10;
    if((abs((*sum_5))) >= (abs(x_10)))
    {
        (*err_4) = (*err_4) + ((*sum_5) - t_13 + x_10);
    }
    else
    {
        (*err_4) = (*err_4) + (x_10 - t_13 + (*sum_5));
    }
    (*sum_5) = t_13;
    return;
}

fn seen_bits_0() -> u32
{
    var _S321 : u32;
    if((g_seen_0[u32(0)]) != u32(0))
    {
        _S321 = u32(32768);
    }
    else
    {
        _S321 = u32(0);
    }
    var _S322 : u32;
    if((g_seen_0[u32(1)]) != u32(0))
    {
        _S322 = u32(65536);
    }
    else
    {
        _S322 = u32(0);
    }
    var _S323 : u32 = (_S321 | (_S322));
    if((g_seen_0[u32(2)]) != u32(0))
    {
        _S321 = u32(131072);
    }
    else
    {
        _S321 = u32(0);
    }
    var _S324 : u32 = (_S323 | (_S321));
    if((g_seen_0[u32(3)]) != u32(0))
    {
        _S321 = u32(32);
    }
    else
    {
        _S321 = u32(0);
    }
    var _S325 : u32 = (_S324 | (_S321));
    if((g_seen_0[u32(4)]) != u32(0))
    {
        _S321 = u32(64);
    }
    else
    {
        _S321 = u32(0);
    }
    return (_S325 | (_S321));
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

fn travel_check_0( c_15 : u32,  fused_1 : bool)
{
    var _S326 : vec4<u32> = chunks_0[c_15].cinfo_0;
    if((chunks_0[c_15].cinfo_0.z) == u32(0))
    {
        return;
    }
    var _S327 : u32 = u32(4) * c_15;
    var at_4 : u32 = params_0.cplan_base_0 + _S327;
    var start_hi_0 : vec4<f32> = scratch_0[at_4];
    var start_rot_0 : vec4<f32> = scratch_0[at_4 + u32(2)];
    var wp_1 : WorldPoint_0 = chunk_world_0(c_15);
    var moved_0 : vec3<f32> = wp_1.hi_0 - scratch_0[at_4].xyz + (wp_1.lo_0 - scratch_0[at_4 + u32(1)].xyz) + wp_1.rel_0;
    var _S328 : u32 = chunks_0[c_15].info_1.y;
    var _S329 : vec3<f32> = vec3<f32>((params_0.t_hi_0 - scratch_0[at_4 + u32(3)].w + (params_0.t_lo_0 - scratch_0[at_4 + u32(1)].w) + f32(islands_0[_S328].info_0.w - params_0.step_start_0) * params_0.dt_0));
    var _S330 : vec3<f32> = scratch_0[at_4 + u32(3)].xyz * _S329;
    var _S331 : u32 = _S326.w;
    var _S332 : vec3<f32>;
    if(((_S331 & (u32(2)))) != u32(0))
    {
        _S332 = vec3<f32>(0.5f) * params_0.gravity_0.xyz * _S329 * _S329;
    }
    else
    {
        _S332 = vec3<f32>(0.0f);
    }
    var travel_0 : f32 = length(moved_0 - (_S330 + _S332));
    var travel_1 : f32;
    if(((_S331 & (u32(1)))) != u32(0))
    {
        var th_4 : vec3<f32> = state_0[_S327 + u32(1)].xyz;
        var turn_0 : Quat_0 = quat_mul_0(quat_of_0(vec4<f32>((vec3<f32>(0) - start_rot_0.xyz), start_rot_0.w)), quat_mul_0(quat_of_0(islands_0[_S328].rotation_0), from_axis_angle_0(th_4, length(th_4))));
        travel_1 = travel_0 + 2.0f * atan2(length(vec3<f32>(turn_0.x_0, turn_0.y_0, turn_0.z_0)), abs(turn_0.w_0)) * chunks_0[c_15].half_0.w;
    }
    else
    {
        travel_1 = travel_0;
    }
    if(travel_1 > (start_hi_0.w))
    {
        travel_stop_0(islands_0[_S328].info_0.w + u32(1), fused_1);
    }
    return;
}

fn island_frame_at_0( gi_0 : u32,  tid_4 : u32)
{
    var c_16 : u32;
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
    var _S333 : bool = !((((isl_9.info_0.x) & (u32(1)))) != u32(0));
    var _S334 : bool;
    if(_S333)
    {
        _S334 = !driven_0;
    }
    else
    {
        _S334 = false;
    }
    var contact_island_0 : bool = (((isl_9.info_0.x) & (u32(4)))) != u32(0);
    var _S335 : bool = (((isl_9.info_0.x) & (u32(16)))) != u32(0);
    seen_clear_0(tid_4);
    var _S336 : bool = tid_4 == u32(0);
    var cov_0 : u32;
    var settled_0 : bool;
    if(_S336)
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
            var _S337 : u32 = islands_0[params_0.halt_index_0].info_0.y;
            if(_S337 != u32(0))
            {
                settled_0 = _S337 <= (isl_9.info_0.w);
            }
            else
            {
                settled_0 = false;
            }
            var _S338 : i32;
            if(settled_0)
            {
                _S338 = i32(0);
            }
            else
            {
                _S338 = i32(1);
            }
            g_sums_0 = u32(_S338);
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
    var _S339 : f32 = params_0.dt_0;
    var _S340 : bool = (params_0.fracture_0) != u32(0);
    var _S341 : bool = (params_0.rigid_motion_loads_0) != u32(0);
    var rg_7 : Rigid_0 = rigid_of_0(isl_9);
    var work_2 : f32 = 0.0f;
    var work_err_1 : f32 = 0.0f;
    settled_0 = _S335;
    var woke_1 : bool = false;
    var done_1 : u32 = u32(0);
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
        var abs_step_1 : u32 = isl_9.info_0.w + done_1 + u32(1);
        var k_12 : u32 = abs_step_1 - u32(1) - params_0.step_start_0;
        var quiet_0 : bool = (((isl_9.info_0.x) & (u32(32)))) != u32(0);
        var _S342 : bool;
        if(contact_island_0)
        {
            _S342 = g_sums_0 != u32(0);
        }
        else
        {
            _S342 = false;
        }
        if(_S342)
        {
            var _S343 : u32 = chunks_0[isl_9.range_0.y - u32(1)].cinfo_0.y;
            c_16 = chunks_0[isl_9.range_0.x].cinfo_0.x + ct_1;
            loop
            {
                if(c_16 < _S343)
                {
                }
                else
                {
                    break;
                }
                contact_segment_0(c_16);
                c_16 = c_16 + u32(256);
            }
            device_group_barrier_0();
        }
        var _S344 : bool;
        if(settled_0)
        {
            _S344 = !quiet_0;
        }
        else
        {
            _S344 = false;
        }
        var _S345 : bool;
        var explicit_step_0 : bool;
        if(_S344)
        {
            var _S346 : vec3<f32> = vec3<f32>(0.0f);
            var norm_0 : vec3<f32> = _S346;
            var unused0_0 : vec3<f32> = _S346;
            c_16 = isl_9.range_0.x + ct_1;
            loop
            {
                if(c_16 < (isl_9.range_0.y))
                {
                }
                else
                {
                    break;
                }
                var _S347 : f32 = settled_chunk_load_0(c_16, rg_7.rot_0, k_12, _S339, contact_island_0);
                norm_0[i32(0)] = norm_0[i32(0)] + _S347;
                c_16 = c_16 + u32(256);
            }
            group_sum3_0(tid_4, &(norm_0), &(unused0_0));
            if((params_0.solve_mode_0) == u32(1))
            {
                _S345 = (abs(norm_0.x - isl_9.energy_0.z)) > (isl_9.energy_0.w);
            }
            else
            {
                _S345 = false;
            }
            if(_S345)
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
        if(_S333)
        {
            _S345 = !quiet_0;
        }
        else
        {
            _S345 = false;
        }
        if(_S345)
        {
            var _S348 : vec3<f32> = vec3<f32>(0.0f);
            var f_11 : vec3<f32> = _S348;
            var t_14 : vec3<f32> = _S348;
            c_16 = isl_9.range_0.x + ct_1;
            loop
            {
                if(c_16 < (isl_9.range_0.y))
                {
                }
                else
                {
                    break;
                }
                net_load_0(c_16, isl_9, rg_7, k_12, _S339, contact_island_0, &(f_11), &(t_14));
                c_16 = c_16 + u32(256);
            }
            group_sum3_0(tid_4, &(f_11), &(t_14));
            rigid_acceleration_0(isl_9, &(rg_7), f_11, t_14);
        }
        if(!settled_0)
        {
            explicit_step_0 = !quiet_0;
        }
        else
        {
            explicit_step_0 = false;
        }
        var _S349 : bool;
        if(explicit_step_0)
        {
            if(woke_0)
            {
                c_16 = isl_9.range_0.z + tid_4;
                loop
                {
                    if(c_16 < (isl_9.range_0.w))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var _S350 : bool = bond_update_0(c_16, _S339, _S340, abs_step_1, false);
                    if(_S350)
                    {
                        g_halt_0 = u32(1);
                    }
                    c_16 = c_16 + u32(256);
                }
            }
            else
            {
                if(_S336)
                {
                    _S349 = (((islands_0[gi_0].info_0.z) & (u32(2)))) != u32(0);
                }
                else
                {
                    _S349 = false;
                }
                if(_S349)
                {
                    g_halt_0 = u32(1);
                }
            }
            device_group_barrier_0();
            c_16 = isl_9.range_0.x + ct_1;
            loop
            {
                if(c_16 < (isl_9.range_0.y))
                {
                }
                else
                {
                    break;
                }
                chunk_update_0(c_16, isl_9, rg_7, _S339, _S341, k_12, contact_island_0, &(work_2), &(work_err_1));
                c_16 = c_16 + u32(256);
            }
            device_group_barrier_0();
        }
        if(_S334)
        {
            _S349 = !quiet_0;
        }
        else
        {
            _S349 = false;
        }
        if(_S349)
        {
            integrate_rigid_0(isl_9, &(rg_7), _S339);
        }
        var _S351 : bool;
        if(_S333)
        {
            _S351 = explicit_step_0;
        }
        else
        {
            _S351 = false;
        }
        if(_S351)
        {
            var _S352 : vec3<f32> = isl_9.wcom_0.xyz;
            var _S353 : vec3<f32> = vec3<f32>(0.0f);
            var tu_1 : vec3<f32> = _S353;
            var pv_1 : vec3<f32> = _S353;
            c_16 = isl_9.range_0.x + ct_1;
            loop
            {
                if(c_16 < (isl_9.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_moments_0(c_16, &(tu_1), &(pv_1));
                c_16 = c_16 + u32(256);
            }
            group_sum3_0(tid_4, &(tu_1), &(pv_1));
            var tr_3 : vec3<f32> = tu_1 / vec3<f32>(isl_9.wcom_0.w);
            var dv_3 : vec3<f32> = pv_1 / vec3<f32>(isl_9.wcom_0.w);
            var lu_1 : vec3<f32> = _S353;
            var lv_1 : vec3<f32> = _S353;
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
                drift_angular_0(c_17, _S352, tr_3, dv_3, &(lu_1), &(lv_1));
                c_17 = c_17 + u32(256);
            }
            group_sum3_0(tid_4, &(lu_1), &(lv_1));
            var phi_3 : vec3<f32> = rows_mul_0(isl_9.winv0_0, isl_9.winv1_0, isl_9.winv2_0, lu_1);
            var dw_2 : vec3<f32> = rows_mul_0(isl_9.winv0_0, isl_9.winv1_0, isl_9.winv2_0, lv_1);
            var c_18 : u32 = isl_9.range_0.x + ct_1;
            loop
            {
                if(c_18 < (isl_9.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_18, _S352, tr_3, phi_3, dv_3, dw_2);
                c_18 = c_18 + u32(256);
            }
            if(!driven_0)
            {
                drift_rigid_0(isl_9, &(rg_7), tr_3, phi_3, dv_3, dw_2);
            }
            device_group_barrier_0();
        }
        var _S354 : bool;
        if(_S336)
        {
            _S354 = (isl_9.probes_0.y) > (isl_9.probes_0.x);
        }
        else
        {
            _S354 = false;
        }
        if(_S354)
        {
            record_probes_0(isl_9, rg_7, k_12);
        }
        var _S355 : u32 = done_1 + u32(1);
        var _S356 : bool;
        if(g_halt_0 != u32(0))
        {
            var _S357 : bool = island_connected_narrow_0(tid_4, ct_1, isl_9);
            _S356 = _S357;
        }
        else
        {
            _S356 = false;
        }
        var resumed_1 : bool;
        if(_S356)
        {
            if(_S336)
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
            done_1 = _S355;
            resumed_0 = resumed_1;
            break;
        }
        woke_1 = woke_0;
        done_1 = _S355;
        resumed_0 = resumed_1;
    }
    var wsum_0 : vec3<f32> = vec3<f32>(work_2, work_err_1, 0.0f);
    var unused_0 : vec3<f32> = vec3<f32>(0.0f);
    group_sum3_0(tid_4, &(wsum_0), &(unused_0));
    if(_S336)
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
            _S334 = contact_island_0;
        }
        else
        {
            _S334 = false;
        }
        if(_S334)
        {
            contact_split_at_0(isl_9.info_0.w + done_1);
        }
        var _S358 : f32 = wsum_0.x;
        var _S359 : f32 = isl_9.energy_0[i32(0)];
        var _S360 : f32 = isl_9.energy_0[i32(1)];
        comp_add1_2(&(_S359), &(_S360), _S358);
        isl_9.energy_0[i32(0)] = _S359;
        isl_9.energy_0[i32(1)] = _S360 + wsum_0.y;
        isl_9.info_0[i32(3)] = isl_9.info_0[i32(3)] + done_1;
        if((((isl_9.info_0.x) & (u32(8)))) == u32(0))
        {
            isl_9.info_0[i32(2)] = ((isl_9.info_0[i32(2)]) & (u32(4294967293)));
        }
        if(g_halt_0 != u32(0))
        {
            isl_9.info_0[i32(2)] = ((isl_9.info_0[i32(2)]) | (u32(1)));
        }
        var _S361 : u32 = seen_bits_0();
        if(g_halt_0 != u32(0))
        {
            cov_0 = u32(128);
        }
        else
        {
            cov_0 = u32(0);
        }
        var _S362 : u32 = (_S361 | (cov_0));
        if(woke_0)
        {
            cov_0 = u32(2048);
        }
        else
        {
            cov_0 = u32(0);
        }
        var _S363 : u32 = (_S362 | (cov_0));
        if(resumed_0)
        {
            cov_0 = u32(2097280);
        }
        else
        {
            cov_0 = u32(0);
        }
        var cov_1 : u32 = (_S363 | (cov_0));
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
                _S334 = !quiet_1;
            }
            else
            {
                _S334 = false;
            }
            if(_S334)
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
        device_group_barrier_0();
        if(done_1 > u32(0))
        {
            c_16 = isl_9.range_0.x + ct_1;
            loop
            {
                if(c_16 < (isl_9.range_0.y))
                {
                }
                else
                {
                    break;
                }
                travel_check_0(c_16, true);
                c_16 = c_16 + u32(256);
            }
        }
    }
    return;
}

fn island_frame_at_1( gi_1 : u32,  tid_5 : u32)
{
    var c_19 : u32;
    var woke_2 : bool;
    var ct_2 : u32;
    if(tid_5 < u32(256))
    {
        ct_2 = tid_5;
    }
    else
    {
        ct_2 = u32(1073741824);
    }
    var isl_10 : Island_0;
    isl_10.range_0 = islands_0[gi_1].range_0;
    isl_10.info_0 = islands_0[gi_1].info_0;
    isl_10.com_0 = islands_0[gi_1].com_0;
    isl_10.inertia0_0 = islands_0[gi_1].inertia0_0;
    isl_10.inertia1_0 = islands_0[gi_1].inertia1_0;
    isl_10.inertia2_0 = islands_0[gi_1].inertia2_0;
    isl_10.inv0_0 = islands_0[gi_1].inv0_0;
    isl_10.inv1_0 = islands_0[gi_1].inv1_0;
    isl_10.inv2_0 = islands_0[gi_1].inv2_0;
    isl_10.wcom_0 = islands_0[gi_1].wcom_0;
    isl_10.winv0_0 = islands_0[gi_1].winv0_0;
    isl_10.winv1_0 = islands_0[gi_1].winv1_0;
    isl_10.winv2_0 = islands_0[gi_1].winv2_0;
    isl_10.rotation_0 = islands_0[gi_1].rotation_0;
    isl_10.position_0 = islands_0[gi_1].position_0;
    isl_10.position_err_0 = islands_0[gi_1].position_err_0;
    isl_10.velocity_0 = islands_0[gi_1].velocity_0;
    isl_10.velocity_err_0 = islands_0[gi_1].velocity_err_0;
    isl_10.angular_velocity_0 = islands_0[gi_1].angular_velocity_0;
    isl_10.done_0 = islands_0[gi_1].done_0;
    isl_10.coverage_0 = islands_0[gi_1].coverage_0;
    isl_10.wide_groups_0 = islands_0[gi_1].wide_groups_0;
    isl_10.done_pad_0 = islands_0[gi_1].done_pad_0;
    isl_10.probes_0 = islands_0[gi_1].probes_0;
    isl_10.energy_0 = islands_0[gi_1].energy_0;
    isl_10.rotation_err_0 = islands_0[gi_1].rotation_err_0;
    isl_10.momentum_0 = islands_0[gi_1].momentum_0;
    isl_10.momentum_err_0 = islands_0[gi_1].momentum_err_0;
    var driven_1 : bool = (((isl_10.info_0.x) & (u32(2)))) != u32(0);
    var _S364 : bool = !((((isl_10.info_0.x) & (u32(1)))) != u32(0));
    var _S365 : bool;
    if(_S364)
    {
        _S365 = !driven_1;
    }
    else
    {
        _S365 = false;
    }
    var contact_island_1 : bool = (((isl_10.info_0.x) & (u32(4)))) != u32(0);
    var _S366 : bool = (((isl_10.info_0.x) & (u32(16)))) != u32(0);
    seen_clear_0(tid_5);
    var _S367 : bool = tid_5 == u32(0);
    var cov_5 : u32;
    var settled_1 : bool;
    if(_S367)
    {
        if(contact_island_1 != ((params_0.contact_mode_0) == u32(1)))
        {
            settled_1 = true;
        }
        else
        {
            settled_1 = (((isl_10.info_0.x) & (u32(8)))) != u32(0);
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
            settled_1 = contact_stopped_0(isl_10);
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
        if(contact_island_1)
        {
            stop_seen_0(params_0.halt_index_0, isl_10.info_0.w + u32(1));
            var _S368 : u32 = islands_0[params_0.halt_index_0].info_0.y;
            if(_S368 != u32(0))
            {
                settled_1 = _S368 <= (isl_10.info_0.w);
            }
            else
            {
                settled_1 = false;
            }
            var _S369 : i32;
            if(settled_1)
            {
                _S369 = i32(0);
            }
            else
            {
                _S369 = i32(1);
            }
            g_sums_0 = u32(_S369);
        }
    }
    workgroupBarrier();
    if((((isl_10.info_0.z) & (u32(1)))) != u32(0))
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
        cov_5 = min(isl_10.info_0.y, params_0.max_steps_0);
    }
    var _S370 : f32 = params_0.dt_0;
    var _S371 : bool = (params_0.rigid_motion_loads_0) != u32(0);
    var rg_8 : Rigid_0 = rigid_of_0(isl_10);
    var work_3 : f32 = 0.0f;
    var work_err_2 : f32 = 0.0f;
    settled_1 = _S366;
    var woke_3 : bool = false;
    var done_2 : u32 = u32(0);
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
        var k_13 : u32 = isl_10.info_0.w + done_2 + u32(1) - u32(1) - params_0.step_start_0;
        var quiet_2 : bool = (((isl_10.info_0.x) & (u32(32)))) != u32(0);
        var _S372 : bool;
        if(contact_island_1)
        {
            _S372 = g_sums_0 != u32(0);
        }
        else
        {
            _S372 = false;
        }
        if(_S372)
        {
            var _S373 : u32 = chunks_0[isl_10.range_0.y - u32(1)].cinfo_0.y;
            c_19 = chunks_0[isl_10.range_0.x].cinfo_0.x + ct_2;
            loop
            {
                if(c_19 < _S373)
                {
                }
                else
                {
                    break;
                }
                contact_segment_0(c_19);
                c_19 = c_19 + u32(256);
            }
            device_group_barrier_0();
        }
        var _S374 : bool;
        if(settled_1)
        {
            _S374 = !quiet_2;
        }
        else
        {
            _S374 = false;
        }
        var _S375 : bool;
        var explicit_step_1 : bool;
        if(_S374)
        {
            var _S376 : vec3<f32> = vec3<f32>(0.0f);
            var norm_1 : vec3<f32> = _S376;
            var unused0_1 : vec3<f32> = _S376;
            c_19 = isl_10.range_0.x + ct_2;
            loop
            {
                if(c_19 < (isl_10.range_0.y))
                {
                }
                else
                {
                    break;
                }
                var _S377 : f32 = settled_chunk_load_0(c_19, rg_8.rot_0, k_13, _S370, contact_island_1);
                norm_1[i32(0)] = norm_1[i32(0)] + _S377;
                c_19 = c_19 + u32(256);
            }
            group_sum3_0(tid_5, &(norm_1), &(unused0_1));
            if((params_0.solve_mode_0) == u32(1))
            {
                _S375 = (abs(norm_1.x - isl_10.energy_0.z)) > (isl_10.energy_0.w);
            }
            else
            {
                _S375 = false;
            }
            if(_S375)
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
        if(_S364)
        {
            _S375 = !quiet_2;
        }
        else
        {
            _S375 = false;
        }
        if(_S375)
        {
            var _S378 : vec3<f32> = vec3<f32>(0.0f);
            var f_12 : vec3<f32> = _S378;
            var t_15 : vec3<f32> = _S378;
            c_19 = isl_10.range_0.x + ct_2;
            loop
            {
                if(c_19 < (isl_10.range_0.y))
                {
                }
                else
                {
                    break;
                }
                net_load_0(c_19, isl_10, rg_8, k_13, _S370, contact_island_1, &(f_12), &(t_15));
                c_19 = c_19 + u32(256);
            }
            group_sum3_0(tid_5, &(f_12), &(t_15));
            rigid_acceleration_0(isl_10, &(rg_8), f_12, t_15);
        }
        if(!settled_1)
        {
            explicit_step_1 = !quiet_2;
        }
        else
        {
            explicit_step_1 = false;
        }
        var _S379 : bool;
        if(explicit_step_1)
        {
            if(_S367)
            {
                _S379 = (((islands_0[gi_1].info_0.z) & (u32(2)))) != u32(0);
            }
            else
            {
                _S379 = false;
            }
            if(_S379)
            {
                g_halt_0 = u32(1);
            }
            device_group_barrier_0();
            c_19 = isl_10.range_0.x + ct_2;
            loop
            {
                if(c_19 < (isl_10.range_0.y))
                {
                }
                else
                {
                    break;
                }
                chunk_update_0(c_19, isl_10, rg_8, _S370, _S371, k_13, contact_island_1, &(work_3), &(work_err_2));
                c_19 = c_19 + u32(256);
            }
            device_group_barrier_0();
        }
        if(_S365)
        {
            _S379 = !quiet_2;
        }
        else
        {
            _S379 = false;
        }
        if(_S379)
        {
            integrate_rigid_0(isl_10, &(rg_8), _S370);
        }
        var _S380 : bool;
        if(_S364)
        {
            _S380 = explicit_step_1;
        }
        else
        {
            _S380 = false;
        }
        if(_S380)
        {
            var _S381 : vec3<f32> = isl_10.wcom_0.xyz;
            var _S382 : vec3<f32> = vec3<f32>(0.0f);
            var tu_2 : vec3<f32> = _S382;
            var pv_2 : vec3<f32> = _S382;
            c_19 = isl_10.range_0.x + ct_2;
            loop
            {
                if(c_19 < (isl_10.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_moments_0(c_19, &(tu_2), &(pv_2));
                c_19 = c_19 + u32(256);
            }
            group_sum3_0(tid_5, &(tu_2), &(pv_2));
            var tr_4 : vec3<f32> = tu_2 / vec3<f32>(isl_10.wcom_0.w);
            var dv_4 : vec3<f32> = pv_2 / vec3<f32>(isl_10.wcom_0.w);
            var lu_2 : vec3<f32> = _S382;
            var lv_2 : vec3<f32> = _S382;
            var c_20 : u32 = isl_10.range_0.x + ct_2;
            loop
            {
                if(c_20 < (isl_10.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_angular_0(c_20, _S381, tr_4, dv_4, &(lu_2), &(lv_2));
                c_20 = c_20 + u32(256);
            }
            group_sum3_0(tid_5, &(lu_2), &(lv_2));
            var phi_4 : vec3<f32> = rows_mul_0(isl_10.winv0_0, isl_10.winv1_0, isl_10.winv2_0, lu_2);
            var dw_3 : vec3<f32> = rows_mul_0(isl_10.winv0_0, isl_10.winv1_0, isl_10.winv2_0, lv_2);
            var c_21 : u32 = isl_10.range_0.x + ct_2;
            loop
            {
                if(c_21 < (isl_10.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_21, _S381, tr_4, phi_4, dv_4, dw_3);
                c_21 = c_21 + u32(256);
            }
            if(!driven_1)
            {
                drift_rigid_0(isl_10, &(rg_8), tr_4, phi_4, dv_4, dw_3);
            }
            device_group_barrier_0();
        }
        var _S383 : bool;
        if(_S367)
        {
            _S383 = (isl_10.probes_0.y) > (isl_10.probes_0.x);
        }
        else
        {
            _S383 = false;
        }
        if(_S383)
        {
            record_probes_0(isl_10, rg_8, k_13);
        }
        var _S384 : u32 = done_2 + u32(1);
        var _S385 : bool;
        if(g_halt_0 != u32(0))
        {
            var _S386 : bool = island_connected_narrow_0(tid_5, ct_2, isl_10);
            _S385 = _S386;
        }
        else
        {
            _S385 = false;
        }
        var resumed_3 : bool;
        if(_S385)
        {
            if(_S367)
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
            done_2 = _S384;
            resumed_2 = resumed_3;
            break;
        }
        woke_3 = woke_2;
        done_2 = _S384;
        resumed_2 = resumed_3;
    }
    var wsum_1 : vec3<f32> = vec3<f32>(work_3, work_err_2, 0.0f);
    var unused_1 : vec3<f32> = vec3<f32>(0.0f);
    group_sum3_0(tid_5, &(wsum_1), &(unused_1));
    if(_S367)
    {
        isl_10.rotation_0 = quat_vec_0(rg_8.rot_0);
        isl_10.rotation_err_0 = rg_8.rot_err_0;
        isl_10.position_0 = vec4<f32>(rg_8.pos_0, 0.0f);
        isl_10.position_err_0 = vec4<f32>(rg_8.pos_err_0, 0.0f);
        isl_10.velocity_0 = vec4<f32>(rg_8.vel_0, 0.0f);
        isl_10.velocity_err_0 = vec4<f32>(rg_8.vel_err_0, 0.0f);
        isl_10.angular_velocity_0 = vec4<f32>(rg_8.w_1, 0.0f);
        isl_10.momentum_0 = vec4<f32>(rg_8.l_0, 0.0f);
        isl_10.momentum_err_0 = vec4<f32>(rg_8.l_err_0, 0.0f);
        isl_10.done_0 = done_2;
        isl_10.info_0[i32(1)] = isl_10.info_0[i32(1)] - done_2;
        if(g_halt_0 != u32(0))
        {
            _S365 = contact_island_1;
        }
        else
        {
            _S365 = false;
        }
        if(_S365)
        {
            contact_split_at_0(isl_10.info_0.w + done_2);
        }
        var _S387 : f32 = wsum_1.x;
        var _S388 : f32 = isl_10.energy_0[i32(0)];
        var _S389 : f32 = isl_10.energy_0[i32(1)];
        comp_add1_2(&(_S388), &(_S389), _S387);
        isl_10.energy_0[i32(0)] = _S388;
        isl_10.energy_0[i32(1)] = _S389 + wsum_1.y;
        isl_10.info_0[i32(3)] = isl_10.info_0[i32(3)] + done_2;
        if((((isl_10.info_0.x) & (u32(8)))) == u32(0))
        {
            isl_10.info_0[i32(2)] = ((isl_10.info_0[i32(2)]) & (u32(4294967293)));
        }
        if(g_halt_0 != u32(0))
        {
            isl_10.info_0[i32(2)] = ((isl_10.info_0[i32(2)]) | (u32(1)));
        }
        var _S390 : u32 = seen_bits_0();
        if(g_halt_0 != u32(0))
        {
            cov_5 = u32(128);
        }
        else
        {
            cov_5 = u32(0);
        }
        var _S391 : u32 = (_S390 | (cov_5));
        if(woke_2)
        {
            cov_5 = u32(2048);
        }
        else
        {
            cov_5 = u32(0);
        }
        var _S392 : u32 = (_S391 | (cov_5));
        if(resumed_2)
        {
            cov_5 = u32(2097280);
        }
        else
        {
            cov_5 = u32(0);
        }
        var cov_6 : u32 = (_S392 | (cov_5));
        if(done_2 > u32(0))
        {
            var quiet_3 : bool = (((isl_10.info_0.x) & (u32(32)))) != u32(0);
            if(quiet_3)
            {
                cov_5 = u32(4096);
            }
            else
            {
                cov_5 = u32(0);
            }
            var cov_7 : u32 = (cov_6 | (cov_5));
            if((((isl_10.info_0.x) & (u32(16)))) != u32(0))
            {
                _S365 = !quiet_3;
            }
            else
            {
                _S365 = false;
            }
            if(_S365)
            {
                cov_5 = u32(512);
            }
            else
            {
                cov_5 = u32(0);
            }
            var cov_8 : u32 = (cov_7 | (cov_5));
            if((params_0.contact_mode_0) == u32(1))
            {
                cov_5 = u32(524288);
            }
            else
            {
                cov_5 = u32(0);
            }
            var cov_9 : u32 = (cov_8 | (cov_5));
            if((isl_10.probes_0.y) > (isl_10.probes_0.x))
            {
                cov_5 = u32(262144);
            }
            else
            {
                cov_5 = u32(0);
            }
            cov_5 = (cov_9 | (cov_5));
        }
        else
        {
            cov_5 = cov_6;
        }
        isl_10.coverage_0 = ((isl_10.coverage_0) | (cov_5));
        if(woke_2)
        {
            isl_10.info_0[i32(0)] = ((isl_10.info_0[i32(0)]) & (u32(4294967279)));
            isl_10.info_0[i32(2)] = ((isl_10.info_0[i32(2)]) | (u32(4)));
        }
        islands_0[gi_1].range_0 = isl_10.range_0;
        islands_0[gi_1].info_0 = isl_10.info_0;
        islands_0[gi_1].com_0 = isl_10.com_0;
        islands_0[gi_1].inertia0_0 = isl_10.inertia0_0;
        islands_0[gi_1].inertia1_0 = isl_10.inertia1_0;
        islands_0[gi_1].inertia2_0 = isl_10.inertia2_0;
        islands_0[gi_1].inv0_0 = isl_10.inv0_0;
        islands_0[gi_1].inv1_0 = isl_10.inv1_0;
        islands_0[gi_1].inv2_0 = isl_10.inv2_0;
        islands_0[gi_1].wcom_0 = isl_10.wcom_0;
        islands_0[gi_1].winv0_0 = isl_10.winv0_0;
        islands_0[gi_1].winv1_0 = isl_10.winv1_0;
        islands_0[gi_1].winv2_0 = isl_10.winv2_0;
        islands_0[gi_1].rotation_0 = isl_10.rotation_0;
        islands_0[gi_1].position_0 = isl_10.position_0;
        islands_0[gi_1].position_err_0 = isl_10.position_err_0;
        islands_0[gi_1].velocity_0 = isl_10.velocity_0;
        islands_0[gi_1].velocity_err_0 = isl_10.velocity_err_0;
        islands_0[gi_1].angular_velocity_0 = isl_10.angular_velocity_0;
        islands_0[gi_1].done_0 = isl_10.done_0;
        islands_0[gi_1].coverage_0 = isl_10.coverage_0;
        islands_0[gi_1].wide_groups_0 = isl_10.wide_groups_0;
        islands_0[gi_1].done_pad_0 = isl_10.done_pad_0;
        islands_0[gi_1].probes_0 = isl_10.probes_0;
        islands_0[gi_1].energy_0 = isl_10.energy_0;
        islands_0[gi_1].rotation_err_0 = isl_10.rotation_err_0;
        islands_0[gi_1].momentum_0 = isl_10.momentum_0;
        islands_0[gi_1].momentum_err_0 = isl_10.momentum_err_0;
    }
    if(contact_island_1)
    {
        device_group_barrier_0();
        if(done_2 > u32(0))
        {
            c_19 = isl_10.range_0.x + ct_2;
            loop
            {
                if(c_19 < (isl_10.range_0.y))
                {
                }
                else
                {
                    break;
                }
                travel_check_0(c_19, true);
                c_19 = c_19 + u32(256);
            }
        }
    }
    return;
}

fn group_sum3_1( tid_6 : u32,  a_15 : ptr<function, vec3<f32>>,  b_22 : ptr<function, vec3<f32>>)
{
    var x_11 : vec4<f32> = vec4<f32>((*a_15), 0.0f);
    var y_2 : vec4<f32> = vec4<f32>((*b_22), 0.0f);
    group_sum2_0(tid_6, &(x_11), &(y_2));
    (*a_15) = x_11.xyz;
    (*b_22) = y_2.xyz;
    return;
}

fn net_load_1( c_22 : u32,  isl_11 : Island_0,  rg_9 : Rigid_0,  k_14 : u32,  dt_12 : f32,  contact_4 : bool,  f_13 : ptr<function, vec3<f32>>,  t_16 : ptr<function, vec3<f32>>)
{
    var fl_1 : vec3<f32>;
    var tl_1 : vec3<f32>;
    var rep_f_2 : vec3<f32>;
    var rep_m_2 : vec3<f32>;
    chunk_loads_1(c_22, c_22, rg_9.rot_0, k_14, dt_12, contact_4, 0.0f, &(fl_1), &(tl_1), &(rep_f_2), &(rep_m_2));
    var fc_4 : vec3<f32> = fl_1 + params_0.gravity_0.xyz * vec3<f32>(chunks_0[c_22].center_0.w);
    var _S393 : vec3<f32> = chunks_0[c_22].center_0.xyz;
    var _S394 : vec3<f32> = isl_11.com_0.xyz;
    var r_9 : vec3<f32> = rotate_0(rg_9.rot_0, _S393 + state_0[u32(4) * c_22].xyz - _S394);
    (*f_13) = (*f_13) + fc_4;
    (*t_16) = (*t_16) + (cross(r_9, fc_4) + tl_1);
    if((chunks_0[c_22].load_range_0.y) > (chunks_0[c_22].load_range_0.x))
    {
        var fw_3 : vec3<f32> = rotate_0(rg_9.rot_0, rep_f_2);
        (*f_13) = (*f_13) + fw_3;
        (*t_16) = (*t_16) + (cross(rotate_0(rg_9.rot_0, _S393 - _S394), fw_3) + rotate_0(rg_9.rot_0, rep_m_2));
    }
    return;
}

fn rigid_acceleration_1( isl_12 : Island_0,  rg_10 : ptr<function, Rigid_0>,  f_14 : vec3<f32>,  t_17 : vec3<f32>)
{
    var iw_w_1 : vec3<f32> = world_mul_0((*rg_10).rot_0, isl_12.inertia0_0, isl_12.inertia1_0, isl_12.inertia2_0, (*rg_10).w_1);
    (*rg_10).a_0 = f_14 / vec3<f32>(isl_12.com_0.w);
    (*rg_10).alpha_0 = world_mul_0((*rg_10).rot_0, isl_12.inv0_0, isl_12.inv1_0, isl_12.inv2_0, t_17 - cross((*rg_10).w_1, iw_w_1));
    (*rg_10).torque_0 = t_17;
    return;
}

fn chunk_update_1( c_23 : u32,  isl_13 : Island_0,  rg_11 : Rigid_0,  dt_13 : f32,  rml_1 : bool,  step_1 : u32,  contact_5 : bool,  work_4 : ptr<function, f32>,  work_err_3 : ptr<function, f32>)
{
    var _S395 : vec3<f32> = vec3<f32>(0.0f);
    var _S396 : u32 = index_0[c_23];
    var peak_1 : f32 = 0.0f;
    var e_3 : u32 = _S396;
    var fi_1 : vec3<f32> = _S395;
    var mi_3 : vec3<f32> = _S395;
    loop
    {
        if(e_3 < index_0[c_23 + u32(1)])
        {
        }
        else
        {
            break;
        }
        var entry_3 : u32 = index_0[e_3];
        var _S397 : u32 = u32(3) * ((entry_3 >> (u32(1))));
        var fa_1 : vec4<f32> = scratch_0[_S397];
        if(((entry_3 & (u32(1)))) == u32(0))
        {
            var mi_4 : vec3<f32> = mi_3 + scratch_0[_S397 + u32(1)].xyz;
            fi_1 = fi_1 + fa_1.xyz;
            mi_3 = mi_4;
        }
        else
        {
            var mi_5 : vec3<f32> = mi_3 + scratch_0[_S397 + u32(2)].xyz;
            fi_1 = fi_1 + (vec3<f32>(0) - fa_1.xyz);
            mi_3 = mi_5;
        }
        var _S398 : f32 = max(peak_1, fa_1.w);
        var _S399 : u32 = e_3 + u32(1);
        peak_1 = _S398;
        e_3 = _S399;
    }
    var _S400 : u32 = u32(4) * c_23;
    var u_3 : vec3<f32> = state_0[_S400].xyz;
    var _S401 : u32 = _S400 + u32(1);
    var th_5 : vec3<f32> = state_0[_S401].xyz;
    var _S402 : u32 = _S400 + u32(2);
    var v_12 : vec3<f32> = state_0[_S402].xyz;
    var _S403 : u32 = _S400 + u32(3);
    var w_7 : vec3<f32> = state_0[_S403].xyz;
    var mass_1 : f32 = chunks_0[c_23].center_0.w;
    var _S404 : vec3<f32> = chunks_0[c_23].center_0.xyz;
    var _S405 : vec3<f32> = isl_13.com_0.xyz;
    var r_world_1 : vec3<f32> = rotate_0(rg_11.rot_0, _S404 + u_3 - _S405);
    var f_load_1 : vec3<f32>;
    var t_load_1 : vec3<f32>;
    var rep_f_3 : vec3<f32>;
    var rep_m_3 : vec3<f32>;
    chunk_loads_1(c_23, c_23, rg_11.rot_0, step_1, dt_13, contact_5, dt_13, &(f_load_1), &(t_load_1), &(rep_f_3), &(rep_m_3));
    record_chunk_load_0(c_23, f_load_1, t_load_1);
    var _S406 : vec3<f32> = vec3<f32>(mass_1);
    var f_world_2 : vec3<f32> = f_load_1 + params_0.gravity_0.xyz * _S406;
    var t_world_3 : vec3<f32> = t_load_1;
    var f_world_3 : vec3<f32>;
    var t_world_4 : vec3<f32>;
    if(rml_1)
    {
        var t_world_5 : vec3<f32> = t_world_3 - (world_mul_0(rg_11.rot_0, chunks_0[c_23].inertia0_1, chunks_0[c_23].inertia1_1, chunks_0[c_23].inertia2_1, rg_11.alpha_0) + cross(rg_11.w_1, world_mul_0(rg_11.rot_0, chunks_0[c_23].inertia0_1, chunks_0[c_23].inertia1_1, chunks_0[c_23].inertia2_1, rg_11.w_1)));
        f_world_3 = f_world_2 - (rg_11.a_0 + cross(rg_11.alpha_0, r_world_1) + cross(rg_11.w_1, cross(rg_11.w_1, r_world_1))) * _S406;
        t_world_4 = t_world_5;
    }
    else
    {
        f_world_3 = f_world_2;
        t_world_4 = t_world_3;
    }
    var f_ext_2 : vec3<f32> = inverse_rotate_0(rg_11.rot_0, f_world_3);
    var m_ext_3 : vec3<f32> = inverse_rotate_0(rg_11.rot_0, t_world_4);
    var f_ext_3 : vec3<f32>;
    var m_ext_4 : vec3<f32>;
    if(rml_1)
    {
        var wb_2 : vec3<f32> = inverse_rotate_0(rg_11.rot_0, rg_11.w_1);
        var i_w_1 : vec3<f32> = rows_mul_0(chunks_0[c_23].inertia0_1, chunks_0[c_23].inertia1_1, chunks_0[c_23].inertia2_1, w_7);
        var m_ext_5 : vec3<f32> = m_ext_3 - (cross(wb_2, i_w_1) + cross(w_7, rows_mul_0(chunks_0[c_23].inertia0_1, chunks_0[c_23].inertia1_1, chunks_0[c_23].inertia2_1, wb_2)) + cross(w_7, i_w_1));
        f_ext_3 = f_ext_2 - cross(wb_2, v_12) * vec3<f32>((2.0f * mass_1));
        m_ext_4 = m_ext_5;
    }
    else
    {
        f_ext_3 = f_ext_2;
        m_ext_4 = m_ext_3;
    }
    var f_15 : vec3<f32> = f_ext_3 + rep_f_3 + fi_1;
    var m_4 : vec3<f32> = m_ext_4 + rep_m_3 + mi_3;
    var support_1 : u32 = chunks_0[c_23].info_1.x;
    var _S407 : vec3<f32> = vec3<f32>(state_0[_S401].w, state_0[_S402].w, state_0[_S403].w);
    var reaction_1 : vec3<f32>;
    var u_4 : vec3<f32>;
    var th_6 : vec3<f32>;
    var v_13 : vec3<f32>;
    var w_8 : vec3<f32>;
    if(support_1 == u32(1))
    {
        reaction_1 = (vec3<f32>(0) - f_15);
        u_4 = u_3;
        th_6 = th_5;
        v_13 = _S395;
        w_8 = _S395;
    }
    else
    {
        var _S408 : vec4<f32> = chunks_0[c_23].scale_0;
        var w_9 : vec3<f32> = w_7 + rows_mul_0(chunks_0[c_23].inv0_1, chunks_0[c_23].inv1_1, chunks_0[c_23].inv2_1, m_4) * vec3<f32>((dt_13 * chunks_0[c_23].scale_0.z));
        var _S409 : vec3<f32> = vec3<f32>(dt_13);
        var th_7 : vec3<f32> = th_5 + w_9 * _S409;
        if(support_1 == u32(2))
        {
            reaction_1 = (vec3<f32>(0) - f_15);
            u_4 = u_3;
            th_6 = _S395;
        }
        else
        {
            var v_14 : vec3<f32> = v_12 + f_15 * vec3<f32>((dt_13 * _S408.y));
            var u_5 : vec3<f32> = u_3 + v_14 * _S409;
            reaction_1 = _S407;
            u_4 = u_5;
            th_6 = v_14;
        }
        var _S410 : vec3<f32> = th_6;
        th_6 = th_7;
        v_13 = _S410;
        w_8 = w_9;
    }
    state_0[_S400] = vec4<f32>(u_4, peak_1);
    state_0[_S401] = vec4<f32>(th_6, reaction_1.x);
    state_0[_S402] = vec4<f32>(v_13, reaction_1.y);
    state_0[_S403] = vec4<f32>(w_8, reaction_1.z);
    comp_add1_0(&((*work_4)), &((*work_err_3)), (dot(f_load_1, rg_11.vel_0 + rg_11.vel_err_0 + cross(rg_11.w_1, rotate_0(rg_11.rot_0, _S404 + u_4 - _S405)) + rotate_0(rg_11.rot_0, v_13)) + dot(t_load_1, rg_11.w_1 + rotate_0(rg_11.rot_0, w_8))) * dt_13);
    return;
}

fn integrate_rigid_1( isl_14 : Island_0,  rg_12 : ptr<function, Rigid_0>,  dt_14 : f32)
{
    var _S411 : vec3<f32> = vec3<f32>(dt_14);
    var _S412 : vec3<f32> = (*rg_12).torque_0 * _S411;
    var _S413 : vec3<f32> = (*rg_12).l_0;
    var _S414 : vec3<f32> = (*rg_12).l_err_0;
    comp_add_0(&(_S413), &(_S414), _S412);
    (*rg_12).l_0 = _S413;
    (*rg_12).l_err_0 = _S414;
    var l_3 : vec3<f32> = _S413 + _S414;
    var _S415 : vec3<f32> = (*rg_12).a_0 * _S411;
    var _S416 : vec3<f32> = (*rg_12).vel_0;
    var _S417 : vec3<f32> = (*rg_12).vel_err_0;
    comp_add_0(&(_S416), &(_S417), _S415);
    (*rg_12).vel_0 = _S416;
    (*rg_12).vel_err_0 = _S417;
    var w_mid_1 : vec3<f32> = world_mul_0((*rg_12).rot_0, isl_14.inv0_0, isl_14.inv1_0, isl_14.inv2_0, l_3);
    var delta_1 : vec3<f32> = (_S416 + _S417) * _S411 - turn_difference_0(w_mid_1, dt_14, rotate_0((*rg_12).rot_0, isl_14.com_0.xyz));
    var _S418 : vec3<f32> = (*rg_12).pos_0;
    var _S419 : vec3<f32> = (*rg_12).pos_err_0;
    comp_add_0(&(_S418), &(_S419), delta_1);
    (*rg_12).pos_0 = _S418;
    (*rg_12).pos_err_0 = _S419;
    var _S420 : Quat_0 = (*rg_12).rot_0;
    var _S421 : vec4<f32> = (*rg_12).rot_err_0;
    turn_left_0(&(_S420), &(_S421), w_mid_1, dt_14);
    (*rg_12).rot_0 = _S420;
    (*rg_12).rot_err_0 = _S421;
    (*rg_12).w_1 = world_mul_0(_S420, isl_14.inv0_0, isl_14.inv1_0, isl_14.inv2_0, l_3);
    return;
}

fn drift_moments_1( c_24 : u32,  tu_3 : ptr<function, vec3<f32>>,  pv_3 : ptr<function, vec3<f32>>)
{
    var _S422 : u32 = u32(4) * c_24;
    var _S423 : vec3<f32> = vec3<f32>((chunks_0[c_24].center_0.w * chunks_0[c_24].scale_0.x));
    (*tu_3) = (*tu_3) + state_0[_S422].xyz * _S423;
    (*pv_3) = (*pv_3) + state_0[_S422 + u32(2)].xyz * _S423;
    return;
}

fn drift_angular_1( c_25 : u32,  wcom_4 : vec3<f32>,  tr_5 : vec3<f32>,  dv_5 : vec3<f32>,  lu_3 : ptr<function, vec3<f32>>,  lv_3 : ptr<function, vec3<f32>>)
{
    var r_10 : vec3<f32> = chunks_0[c_25].center_0.xyz - wcom_4;
    var _S424 : u32 = u32(4) * c_25;
    var _S425 : vec3<f32> = vec3<f32>(chunks_0[c_25].center_0.w);
    var _S426 : vec4<f32> = chunks_0[c_25].inertia0_1;
    var _S427 : vec4<f32> = chunks_0[c_25].inertia1_1;
    var _S428 : vec4<f32> = chunks_0[c_25].inertia2_1;
    var _S429 : vec3<f32> = vec3<f32>(chunks_0[c_25].scale_0.x);
    (*lu_3) = (*lu_3) + (cross(r_10, state_0[_S424].xyz - tr_5) * _S425 + rows_mul_0(chunks_0[c_25].inertia0_1, chunks_0[c_25].inertia1_1, chunks_0[c_25].inertia2_1, state_0[_S424 + u32(1)].xyz)) * _S429;
    (*lv_3) = (*lv_3) + (cross(r_10, state_0[_S424 + u32(2)].xyz - dv_5) * _S425 + rows_mul_0(_S426, _S427, _S428, state_0[_S424 + u32(3)].xyz)) * _S429;
    return;
}

fn drift_rigid_1( isl_15 : Island_0,  rg_13 : ptr<function, Rigid_0>,  tr_6 : vec3<f32>,  phi_5 : vec3<f32>,  dv_6 : vec3<f32>,  dw_4 : vec3<f32>)
{
    var wcom_5 : vec3<f32> = isl_15.wcom_0.xyz;
    var rot_3 : Quat_0 = (*rg_13).rot_0;
    var _S430 : vec3<f32> = rotate_0((*rg_13).rot_0, tr_6 - turn_difference_0(phi_5, 1.0f, wcom_5));
    var _S431 : vec3<f32> = (*rg_13).pos_0;
    var _S432 : vec3<f32> = (*rg_13).pos_err_0;
    comp_add_0(&(_S431), &(_S432), _S430);
    (*rg_13).pos_0 = _S431;
    (*rg_13).pos_err_0 = _S432;
    var b_23 : vec3<f32> = inverse_rotate_0((*rg_13).rot_0, (*rg_13).w_1);
    var dl_turn_1 : vec3<f32> = rotate_0((*rg_13).rot_0, cross(phi_5, rows_mul_0(isl_15.inertia0_0, isl_15.inertia1_0, isl_15.inertia2_0, b_23)) - rows_mul_0(isl_15.inertia0_0, isl_15.inertia1_0, isl_15.inertia2_0, cross(phi_5, b_23)));
    var _S433 : Quat_0 = (*rg_13).rot_0;
    var _S434 : vec4<f32> = (*rg_13).rot_err_0;
    turn_right_0(&(_S433), &(_S434), phi_5);
    (*rg_13).rot_0 = _S433;
    (*rg_13).rot_err_0 = _S434;
    var _S435 : vec3<f32> = rotate_0(rot_3, dv_6 + cross(dw_4, isl_15.com_0.xyz - wcom_5));
    var _S436 : vec3<f32> = (*rg_13).vel_0;
    var _S437 : vec3<f32> = (*rg_13).vel_err_0;
    comp_add_0(&(_S436), &(_S437), _S435);
    (*rg_13).vel_0 = _S436;
    (*rg_13).vel_err_0 = _S437;
    var _S438 : vec3<f32> = rotate_0(rot_3, dw_4);
    (*rg_13).w_1 = (*rg_13).w_1 + _S438;
    var _S439 : vec3<f32> = dl_turn_1 + world_mul_0(_S433, isl_15.inertia0_0, isl_15.inertia1_0, isl_15.inertia2_0, _S438);
    var _S440 : vec3<f32> = (*rg_13).l_0;
    var _S441 : vec3<f32> = (*rg_13).l_err_0;
    comp_add_0(&(_S440), &(_S441), _S439);
    (*rg_13).l_0 = _S440;
    (*rg_13).l_err_0 = _S441;
    return;
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
    var _S442 : bool = !((((isl_16.info_0.x) & (u32(1)))) != u32(0));
    var _S443 : bool;
    if(_S442)
    {
        _S443 = !driven_2;
    }
    else
    {
        _S443 = false;
    }
    var contact_island_2 : bool = (((isl_16.info_0.x) & (u32(4)))) != u32(0);
    var _S444 : bool = (((isl_16.info_0.x) & (u32(16)))) != u32(0);
    seen_clear_0(tid_7);
    var _S445 : bool = tid_7 == u32(0);
    var cov_10 : u32;
    var settled_2 : bool;
    if(_S445)
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
    var _S446 : f32 = params_0.dt_0;
    var _S447 : bool = (params_0.rigid_motion_loads_0) != u32(0);
    var rg_14 : Rigid_0 = rigid_of_0(isl_16);
    var work_5 : f32 = 0.0f;
    var work_err_4 : f32 = 0.0f;
    settled_2 = _S444;
    var woke_5 : bool = false;
    var done_3 : u32 = u32(0);
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
        var k_15 : u32 = isl_16.info_0.w + done_3 + u32(1) - u32(1) - params_0.step_start_0;
        var quiet_4 : bool = (((isl_16.info_0.x) & (u32(32)))) != u32(0);
        var _S448 : bool;
        if(settled_2)
        {
            _S448 = !quiet_4;
        }
        else
        {
            _S448 = false;
        }
        var c_26 : u32;
        var _S449 : bool;
        var explicit_step_2 : bool;
        if(_S448)
        {
            var _S450 : vec3<f32> = vec3<f32>(0.0f);
            var norm_2 : vec3<f32> = _S450;
            var unused0_2 : vec3<f32> = _S450;
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
                var _S451 : f32 = settled_chunk_load_0(c_26, rg_14.rot_0, k_15, _S446, contact_island_2);
                norm_2[i32(0)] = norm_2[i32(0)] + _S451;
                c_26 = c_26 + u32(256);
            }
            group_sum3_1(tid_7, &(norm_2), &(unused0_2));
            if((params_0.solve_mode_0) == u32(1))
            {
                _S449 = (abs(norm_2.x - isl_16.energy_0.z)) > (isl_16.energy_0.w);
            }
            else
            {
                _S449 = false;
            }
            if(_S449)
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
        if(_S442)
        {
            _S449 = !quiet_4;
        }
        else
        {
            _S449 = false;
        }
        if(_S449)
        {
            var _S452 : vec3<f32> = vec3<f32>(0.0f);
            var f_16 : vec3<f32> = _S452;
            var t_19 : vec3<f32> = _S452;
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
                net_load_1(c_26, isl_16, rg_14, k_15, _S446, contact_island_2, &(f_16), &(t_19));
                c_26 = c_26 + u32(256);
            }
            group_sum3_1(tid_7, &(f_16), &(t_19));
            rigid_acceleration_1(isl_16, &(rg_14), f_16, t_19);
        }
        if(!settled_2)
        {
            explicit_step_2 = !quiet_4;
        }
        else
        {
            explicit_step_2 = false;
        }
        var _S453 : bool;
        if(explicit_step_2)
        {
            if(_S445)
            {
                _S453 = (((islands_0[gi_2].info_0.z) & (u32(2)))) != u32(0);
            }
            else
            {
                _S453 = false;
            }
            if(_S453)
            {
                g_halt_0 = u32(1);
            }
            device_group_barrier_0();
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
                chunk_update_1(c_26, isl_16, rg_14, _S446, _S447, k_15, contact_island_2, &(work_5), &(work_err_4));
                c_26 = c_26 + u32(256);
            }
            device_group_barrier_0();
        }
        if(_S443)
        {
            _S453 = !quiet_4;
        }
        else
        {
            _S453 = false;
        }
        if(_S453)
        {
            integrate_rigid_1(isl_16, &(rg_14), _S446);
        }
        var _S454 : bool;
        if(_S442)
        {
            _S454 = explicit_step_2;
        }
        else
        {
            _S454 = false;
        }
        if(_S454)
        {
            var _S455 : vec3<f32> = isl_16.wcom_0.xyz;
            var _S456 : vec3<f32> = vec3<f32>(0.0f);
            var tu_4 : vec3<f32> = _S456;
            var pv_4 : vec3<f32> = _S456;
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
                drift_moments_1(c_26, &(tu_4), &(pv_4));
                c_26 = c_26 + u32(256);
            }
            group_sum3_1(tid_7, &(tu_4), &(pv_4));
            var tr_7 : vec3<f32> = tu_4 / vec3<f32>(isl_16.wcom_0.w);
            var dv_7 : vec3<f32> = pv_4 / vec3<f32>(isl_16.wcom_0.w);
            var lu_4 : vec3<f32> = _S456;
            var lv_4 : vec3<f32> = _S456;
            var c_27 : u32 = isl_16.range_0.x + ct_3;
            loop
            {
                if(c_27 < (isl_16.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_angular_1(c_27, _S455, tr_7, dv_7, &(lu_4), &(lv_4));
                c_27 = c_27 + u32(256);
            }
            group_sum3_1(tid_7, &(lu_4), &(lv_4));
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
                drift_apply_0(c_28, _S455, tr_7, phi_6, dv_7, dw_5);
                c_28 = c_28 + u32(256);
            }
            if(!driven_2)
            {
                drift_rigid_1(isl_16, &(rg_14), tr_7, phi_6, dv_7, dw_5);
            }
            device_group_barrier_0();
        }
        var _S457 : bool;
        if(_S445)
        {
            _S457 = (isl_16.probes_0.y) > (isl_16.probes_0.x);
        }
        else
        {
            _S457 = false;
        }
        if(_S457)
        {
            record_probes_0(isl_16, rg_14, k_15);
        }
        var _S458 : u32 = done_3 + u32(1);
        var _S459 : bool;
        if(g_halt_0 != u32(0))
        {
            var _S460 : bool = island_connected_narrow_0(tid_7, ct_3, isl_16);
            _S459 = _S460;
        }
        else
        {
            _S459 = false;
        }
        var resumed_5 : bool;
        if(_S459)
        {
            if(_S445)
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
            done_3 = _S458;
            resumed_4 = resumed_5;
            break;
        }
        woke_5 = woke_4;
        done_3 = _S458;
        resumed_4 = resumed_5;
    }
    var wsum_2 : vec3<f32> = vec3<f32>(work_5, work_err_4, 0.0f);
    var unused_2 : vec3<f32> = vec3<f32>(0.0f);
    group_sum3_1(tid_7, &(wsum_2), &(unused_2));
    if(_S445)
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
            _S443 = contact_island_2;
        }
        else
        {
            _S443 = false;
        }
        if(_S443)
        {
            contact_split_at_0(isl_16.info_0.w + done_3);
        }
        var _S461 : f32 = wsum_2.x;
        var _S462 : f32 = isl_16.energy_0[i32(0)];
        var _S463 : f32 = isl_16.energy_0[i32(1)];
        comp_add1_3(&(_S462), &(_S463), _S461);
        isl_16.energy_0[i32(0)] = _S462;
        isl_16.energy_0[i32(1)] = _S463 + wsum_2.y;
        isl_16.info_0[i32(3)] = isl_16.info_0[i32(3)] + done_3;
        if((((isl_16.info_0.x) & (u32(8)))) == u32(0))
        {
            isl_16.info_0[i32(2)] = ((isl_16.info_0[i32(2)]) & (u32(4294967293)));
        }
        if(g_halt_0 != u32(0))
        {
            isl_16.info_0[i32(2)] = ((isl_16.info_0[i32(2)]) | (u32(1)));
        }
        var _S464 : u32 = seen_bits_0();
        if(g_halt_0 != u32(0))
        {
            ct_3 = u32(128);
        }
        else
        {
            ct_3 = u32(0);
        }
        var _S465 : u32 = (_S464 | (ct_3));
        if(woke_4)
        {
            ct_3 = u32(2048);
        }
        else
        {
            ct_3 = u32(0);
        }
        var _S466 : u32 = (_S465 | (ct_3));
        if(resumed_4)
        {
            ct_3 = u32(2097280);
        }
        else
        {
            ct_3 = u32(0);
        }
        var cov_11 : u32 = (_S466 | (ct_3));
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
                _S443 = !quiet_5;
            }
            else
            {
                _S443 = false;
            }
            if(_S443)
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
    var _S467 : bool = !((((isl_17.info_0.x) & (u32(1)))) != u32(0));
    var _S468 : bool;
    if(_S467)
    {
        _S468 = !driven_3;
    }
    else
    {
        _S468 = false;
    }
    var contact_island_3 : bool = (((isl_17.info_0.x) & (u32(4)))) != u32(0);
    var _S469 : bool = (((isl_17.info_0.x) & (u32(16)))) != u32(0);
    seen_clear_0(tid_8);
    var _S470 : bool = tid_8 == u32(0);
    var cov_15 : u32;
    var settled_3 : bool;
    if(_S470)
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
    var _S471 : f32 = params_0.dt_0;
    var _S472 : bool = (params_0.fracture_0) != u32(0);
    var _S473 : bool = (params_0.rigid_motion_loads_0) != u32(0);
    var rg_15 : Rigid_0 = rigid_of_0(isl_17);
    var work_6 : f32 = 0.0f;
    var work_err_5 : f32 = 0.0f;
    settled_3 = _S469;
    var woke_7 : bool = false;
    var done_4 : u32 = u32(0);
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
        var _S474 : bool;
        if(settled_3)
        {
            _S474 = !quiet_6;
        }
        else
        {
            _S474 = false;
        }
        var c_29 : u32;
        var _S475 : bool;
        var explicit_step_3 : bool;
        if(_S474)
        {
            var _S476 : vec3<f32> = vec3<f32>(0.0f);
            var norm_3 : vec3<f32> = _S476;
            var unused0_3 : vec3<f32> = _S476;
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
                var _S477 : f32 = settled_chunk_load_0(c_29, rg_15.rot_0, k_16, _S471, contact_island_3);
                norm_3[i32(0)] = norm_3[i32(0)] + _S477;
                c_29 = c_29 + u32(256);
            }
            group_sum3_0(tid_8, &(norm_3), &(unused0_3));
            if((params_0.solve_mode_0) == u32(1))
            {
                _S475 = (abs(norm_3.x - isl_17.energy_0.z)) > (isl_17.energy_0.w);
            }
            else
            {
                _S475 = false;
            }
            if(_S475)
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
        if(_S467)
        {
            _S475 = !quiet_6;
        }
        else
        {
            _S475 = false;
        }
        if(_S475)
        {
            var _S478 : vec3<f32> = vec3<f32>(0.0f);
            var f_17 : vec3<f32> = _S478;
            var t_20 : vec3<f32> = _S478;
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
                net_load_0(c_29, isl_17, rg_15, k_16, _S471, contact_island_3, &(f_17), &(t_20));
                c_29 = c_29 + u32(256);
            }
            group_sum3_0(tid_8, &(f_17), &(t_20));
            rigid_acceleration_0(isl_17, &(rg_15), f_17, t_20);
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
                var _S479 : bool = bond_update_0(c_29, _S471, _S472, abs_step_2, false);
                if(_S479)
                {
                    g_halt_0 = u32(1);
                }
                c_29 = c_29 + u32(512);
            }
            device_group_barrier_0();
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
                chunk_update_0(c_30, isl_17, rg_15, _S471, _S473, k_16, contact_island_3, &(work_6), &(work_err_5));
                c_30 = c_30 + u32(256);
            }
            device_group_barrier_0();
        }
        var _S480 : bool;
        if(_S468)
        {
            _S480 = !quiet_6;
        }
        else
        {
            _S480 = false;
        }
        if(_S480)
        {
            integrate_rigid_0(isl_17, &(rg_15), _S471);
        }
        var _S481 : bool;
        if(_S467)
        {
            _S481 = explicit_step_3;
        }
        else
        {
            _S481 = false;
        }
        if(_S481)
        {
            var _S482 : vec3<f32> = isl_17.wcom_0.xyz;
            var _S483 : vec3<f32> = vec3<f32>(0.0f);
            var tu_5 : vec3<f32> = _S483;
            var pv_5 : vec3<f32> = _S483;
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
                drift_moments_0(c_29, &(tu_5), &(pv_5));
                c_29 = c_29 + u32(256);
            }
            group_sum3_0(tid_8, &(tu_5), &(pv_5));
            var tr_8 : vec3<f32> = tu_5 / vec3<f32>(isl_17.wcom_0.w);
            var dv_8 : vec3<f32> = pv_5 / vec3<f32>(isl_17.wcom_0.w);
            var lu_5 : vec3<f32> = _S483;
            var lv_5 : vec3<f32> = _S483;
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
                drift_angular_0(c_30, _S482, tr_8, dv_8, &(lu_5), &(lv_5));
                c_30 = c_30 + u32(256);
            }
            group_sum3_0(tid_8, &(lu_5), &(lv_5));
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
                drift_apply_0(c_31, _S482, tr_8, phi_7, dv_8, dw_6);
                c_31 = c_31 + u32(256);
            }
            if(!driven_3)
            {
                drift_rigid_0(isl_17, &(rg_15), tr_8, phi_7, dv_8, dw_6);
            }
            device_group_barrier_0();
        }
        var _S484 : bool;
        if(_S470)
        {
            _S484 = (isl_17.probes_0.y) > (isl_17.probes_0.x);
        }
        else
        {
            _S484 = false;
        }
        if(_S484)
        {
            record_probes_0(isl_17, rg_15, k_16);
        }
        var _S485 : u32 = done_4 + u32(1);
        var _S486 : bool;
        if(g_halt_0 != u32(0))
        {
            var _S487 : bool = island_connected_narrow_0(tid_8, ct_4, isl_17);
            _S486 = _S487;
        }
        else
        {
            _S486 = false;
        }
        var resumed_7 : bool;
        if(_S486)
        {
            if(_S470)
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
            done_4 = _S485;
            resumed_6 = resumed_7;
            break;
        }
        woke_7 = woke_6;
        done_4 = _S485;
        resumed_6 = resumed_7;
    }
    var wsum_3 : vec3<f32> = vec3<f32>(work_6, work_err_5, 0.0f);
    var unused_3 : vec3<f32> = vec3<f32>(0.0f);
    group_sum3_0(tid_8, &(wsum_3), &(unused_3));
    if(_S470)
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
            _S468 = contact_island_3;
        }
        else
        {
            _S468 = false;
        }
        if(_S468)
        {
            contact_split_at_0(isl_17.info_0.w + done_4);
        }
        var _S488 : f32 = wsum_3.x;
        var _S489 : f32 = isl_17.energy_0[i32(0)];
        var _S490 : f32 = isl_17.energy_0[i32(1)];
        comp_add1_2(&(_S489), &(_S490), _S488);
        isl_17.energy_0[i32(0)] = _S489;
        isl_17.energy_0[i32(1)] = _S490 + wsum_3.y;
        isl_17.info_0[i32(3)] = isl_17.info_0[i32(3)] + done_4;
        if((((isl_17.info_0.x) & (u32(8)))) == u32(0))
        {
            isl_17.info_0[i32(2)] = ((isl_17.info_0[i32(2)]) & (u32(4294967293)));
        }
        if(g_halt_0 != u32(0))
        {
            isl_17.info_0[i32(2)] = ((isl_17.info_0[i32(2)]) | (u32(1)));
        }
        var _S491 : u32 = seen_bits_0();
        if(g_halt_0 != u32(0))
        {
            ct_4 = u32(128);
        }
        else
        {
            ct_4 = u32(0);
        }
        var _S492 : u32 = (_S491 | (ct_4));
        if(woke_6)
        {
            ct_4 = u32(2048);
        }
        else
        {
            ct_4 = u32(0);
        }
        var _S493 : u32 = (_S492 | (ct_4));
        if(resumed_6)
        {
            ct_4 = u32(2097280);
        }
        else
        {
            ct_4 = u32(0);
        }
        var cov_16 : u32 = (_S493 | (ct_4));
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
                _S468 = !quiet_7;
            }
            else
            {
                _S468 = false;
            }
            if(_S468)
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

fn island_frame_at_4( gi_4 : u32,  tid_9 : u32)
{
    var woke_8 : bool;
    var ct_5 : u32;
    if(tid_9 < u32(256))
    {
        ct_5 = tid_9;
    }
    else
    {
        ct_5 = u32(1073741824);
    }
    var isl_18 : Island_0;
    isl_18.range_0 = islands_0[gi_4].range_0;
    isl_18.info_0 = islands_0[gi_4].info_0;
    isl_18.com_0 = islands_0[gi_4].com_0;
    isl_18.inertia0_0 = islands_0[gi_4].inertia0_0;
    isl_18.inertia1_0 = islands_0[gi_4].inertia1_0;
    isl_18.inertia2_0 = islands_0[gi_4].inertia2_0;
    isl_18.inv0_0 = islands_0[gi_4].inv0_0;
    isl_18.inv1_0 = islands_0[gi_4].inv1_0;
    isl_18.inv2_0 = islands_0[gi_4].inv2_0;
    isl_18.wcom_0 = islands_0[gi_4].wcom_0;
    isl_18.winv0_0 = islands_0[gi_4].winv0_0;
    isl_18.winv1_0 = islands_0[gi_4].winv1_0;
    isl_18.winv2_0 = islands_0[gi_4].winv2_0;
    isl_18.rotation_0 = islands_0[gi_4].rotation_0;
    isl_18.position_0 = islands_0[gi_4].position_0;
    isl_18.position_err_0 = islands_0[gi_4].position_err_0;
    isl_18.velocity_0 = islands_0[gi_4].velocity_0;
    isl_18.velocity_err_0 = islands_0[gi_4].velocity_err_0;
    isl_18.angular_velocity_0 = islands_0[gi_4].angular_velocity_0;
    isl_18.done_0 = islands_0[gi_4].done_0;
    isl_18.coverage_0 = islands_0[gi_4].coverage_0;
    isl_18.wide_groups_0 = islands_0[gi_4].wide_groups_0;
    isl_18.done_pad_0 = islands_0[gi_4].done_pad_0;
    isl_18.probes_0 = islands_0[gi_4].probes_0;
    isl_18.energy_0 = islands_0[gi_4].energy_0;
    isl_18.rotation_err_0 = islands_0[gi_4].rotation_err_0;
    isl_18.momentum_0 = islands_0[gi_4].momentum_0;
    isl_18.momentum_err_0 = islands_0[gi_4].momentum_err_0;
    var driven_4 : bool = (((isl_18.info_0.x) & (u32(2)))) != u32(0);
    var _S494 : bool = !((((isl_18.info_0.x) & (u32(1)))) != u32(0));
    var _S495 : bool;
    if(_S494)
    {
        _S495 = !driven_4;
    }
    else
    {
        _S495 = false;
    }
    var contact_island_4 : bool = (((isl_18.info_0.x) & (u32(4)))) != u32(0);
    var _S496 : bool = (((isl_18.info_0.x) & (u32(16)))) != u32(0);
    seen_clear_0(tid_9);
    var _S497 : bool = tid_9 == u32(0);
    var cov_20 : u32;
    var settled_4 : bool;
    if(_S497)
    {
        if(contact_island_4 != ((params_0.contact_mode_0) == u32(1)))
        {
            settled_4 = true;
        }
        else
        {
            settled_4 = (((isl_18.info_0.x) & (u32(8)))) != u32(0);
        }
        if(settled_4)
        {
            cov_20 = u32(0);
        }
        else
        {
            cov_20 = u32(1);
        }
        if(contact_island_4)
        {
            settled_4 = contact_stopped_0(isl_18);
        }
        else
        {
            settled_4 = false;
        }
        if(settled_4)
        {
            cov_20 = u32(0);
        }
        g_run_0 = cov_20;
        g_halt_0 = u32(0);
    }
    workgroupBarrier();
    if((((isl_18.info_0.z) & (u32(1)))) != u32(0))
    {
        settled_4 = true;
    }
    else
    {
        settled_4 = g_run_0 == u32(0);
    }
    if(settled_4)
    {
        cov_20 = u32(0);
    }
    else
    {
        cov_20 = min(isl_18.info_0.y, params_0.max_steps_0);
    }
    var _S498 : f32 = params_0.dt_0;
    var _S499 : bool = (params_0.fracture_0) != u32(0);
    var _S500 : bool = (params_0.rigid_motion_loads_0) != u32(0);
    var rg_16 : Rigid_0 = rigid_of_0(isl_18);
    var work_7 : f32 = 0.0f;
    var work_err_6 : f32 = 0.0f;
    settled_4 = _S496;
    var woke_9 : bool = false;
    var done_5 : u32 = u32(0);
    var resumed_8 : bool = false;
    loop
    {
        if(done_5 < cov_20)
        {
        }
        else
        {
            woke_8 = woke_9;
            break;
        }
        var abs_step_3 : u32 = isl_18.info_0.w + done_5 + u32(1);
        var k_17 : u32 = abs_step_3 - u32(1) - params_0.step_start_0;
        var quiet_8 : bool = (((isl_18.info_0.x) & (u32(32)))) != u32(0);
        var _S501 : bool;
        if(settled_4)
        {
            _S501 = !quiet_8;
        }
        else
        {
            _S501 = false;
        }
        var c_32 : u32;
        var _S502 : bool;
        var explicit_step_4 : bool;
        if(_S501)
        {
            var _S503 : vec3<f32> = vec3<f32>(0.0f);
            var norm_4 : vec3<f32> = _S503;
            var unused0_4 : vec3<f32> = _S503;
            c_32 = isl_18.range_0.x + ct_5;
            loop
            {
                if(c_32 < (isl_18.range_0.y))
                {
                }
                else
                {
                    break;
                }
                var _S504 : f32 = settled_chunk_load_0(c_32, rg_16.rot_0, k_17, _S498, contact_island_4);
                norm_4[i32(0)] = norm_4[i32(0)] + _S504;
                c_32 = c_32 + u32(256);
            }
            group_sum3_1(tid_9, &(norm_4), &(unused0_4));
            if((params_0.solve_mode_0) == u32(1))
            {
                _S502 = (abs(norm_4.x - isl_18.energy_0.z)) > (isl_18.energy_0.w);
            }
            else
            {
                _S502 = false;
            }
            if(_S502)
            {
                explicit_step_4 = false;
                woke_8 = true;
            }
            else
            {
                explicit_step_4 = settled_4;
                woke_8 = woke_9;
            }
            settled_4 = explicit_step_4;
        }
        else
        {
            woke_8 = woke_9;
        }
        if(_S494)
        {
            _S502 = !quiet_8;
        }
        else
        {
            _S502 = false;
        }
        if(_S502)
        {
            var _S505 : vec3<f32> = vec3<f32>(0.0f);
            var f_18 : vec3<f32> = _S505;
            var t_21 : vec3<f32> = _S505;
            c_32 = isl_18.range_0.x + ct_5;
            loop
            {
                if(c_32 < (isl_18.range_0.y))
                {
                }
                else
                {
                    break;
                }
                net_load_1(c_32, isl_18, rg_16, k_17, _S498, contact_island_4, &(f_18), &(t_21));
                c_32 = c_32 + u32(256);
            }
            group_sum3_1(tid_9, &(f_18), &(t_21));
            rigid_acceleration_1(isl_18, &(rg_16), f_18, t_21);
        }
        if(!settled_4)
        {
            explicit_step_4 = !quiet_8;
        }
        else
        {
            explicit_step_4 = false;
        }
        var c_33 : u32;
        if(explicit_step_4)
        {
            c_32 = isl_18.range_0.z + tid_9;
            loop
            {
                if(c_32 < (isl_18.range_0.w))
                {
                }
                else
                {
                    break;
                }
                var _S506 : bool = bond_update_0(c_32, _S498, _S499, abs_step_3, false);
                if(_S506)
                {
                    g_halt_0 = u32(1);
                }
                c_32 = c_32 + u32(256);
            }
            device_group_barrier_0();
            c_33 = isl_18.range_0.x + ct_5;
            loop
            {
                if(c_33 < (isl_18.range_0.y))
                {
                }
                else
                {
                    break;
                }
                chunk_update_1(c_33, isl_18, rg_16, _S498, _S500, k_17, contact_island_4, &(work_7), &(work_err_6));
                c_33 = c_33 + u32(256);
            }
            device_group_barrier_0();
        }
        var _S507 : bool;
        if(_S495)
        {
            _S507 = !quiet_8;
        }
        else
        {
            _S507 = false;
        }
        if(_S507)
        {
            integrate_rigid_1(isl_18, &(rg_16), _S498);
        }
        var _S508 : bool;
        if(_S494)
        {
            _S508 = explicit_step_4;
        }
        else
        {
            _S508 = false;
        }
        if(_S508)
        {
            var _S509 : vec3<f32> = isl_18.wcom_0.xyz;
            var _S510 : vec3<f32> = vec3<f32>(0.0f);
            var tu_6 : vec3<f32> = _S510;
            var pv_6 : vec3<f32> = _S510;
            c_32 = isl_18.range_0.x + ct_5;
            loop
            {
                if(c_32 < (isl_18.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_moments_1(c_32, &(tu_6), &(pv_6));
                c_32 = c_32 + u32(256);
            }
            group_sum3_1(tid_9, &(tu_6), &(pv_6));
            var tr_9 : vec3<f32> = tu_6 / vec3<f32>(isl_18.wcom_0.w);
            var dv_9 : vec3<f32> = pv_6 / vec3<f32>(isl_18.wcom_0.w);
            var lu_6 : vec3<f32> = _S510;
            var lv_6 : vec3<f32> = _S510;
            c_33 = isl_18.range_0.x + ct_5;
            loop
            {
                if(c_33 < (isl_18.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_angular_1(c_33, _S509, tr_9, dv_9, &(lu_6), &(lv_6));
                c_33 = c_33 + u32(256);
            }
            group_sum3_1(tid_9, &(lu_6), &(lv_6));
            var phi_8 : vec3<f32> = rows_mul_0(isl_18.winv0_0, isl_18.winv1_0, isl_18.winv2_0, lu_6);
            var dw_7 : vec3<f32> = rows_mul_0(isl_18.winv0_0, isl_18.winv1_0, isl_18.winv2_0, lv_6);
            var c_34 : u32 = isl_18.range_0.x + ct_5;
            loop
            {
                if(c_34 < (isl_18.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_34, _S509, tr_9, phi_8, dv_9, dw_7);
                c_34 = c_34 + u32(256);
            }
            if(!driven_4)
            {
                drift_rigid_1(isl_18, &(rg_16), tr_9, phi_8, dv_9, dw_7);
            }
            device_group_barrier_0();
        }
        var _S511 : bool;
        if(_S497)
        {
            _S511 = (isl_18.probes_0.y) > (isl_18.probes_0.x);
        }
        else
        {
            _S511 = false;
        }
        if(_S511)
        {
            record_probes_0(isl_18, rg_16, k_17);
        }
        var _S512 : u32 = done_5 + u32(1);
        var _S513 : bool;
        if(g_halt_0 != u32(0))
        {
            var _S514 : bool = island_connected_narrow_0(tid_9, ct_5, isl_18);
            _S513 = _S514;
        }
        else
        {
            _S513 = false;
        }
        var resumed_9 : bool;
        if(_S513)
        {
            if(_S497)
            {
                g_halt_0 = u32(0);
            }
            workgroupBarrier();
            resumed_9 = true;
        }
        else
        {
            resumed_9 = resumed_8;
        }
        if(g_halt_0 != u32(0))
        {
            done_5 = _S512;
            resumed_8 = resumed_9;
            break;
        }
        woke_9 = woke_8;
        done_5 = _S512;
        resumed_8 = resumed_9;
    }
    var wsum_4 : vec3<f32> = vec3<f32>(work_7, work_err_6, 0.0f);
    var unused_4 : vec3<f32> = vec3<f32>(0.0f);
    group_sum3_1(tid_9, &(wsum_4), &(unused_4));
    if(_S497)
    {
        isl_18.rotation_0 = quat_vec_0(rg_16.rot_0);
        isl_18.rotation_err_0 = rg_16.rot_err_0;
        isl_18.position_0 = vec4<f32>(rg_16.pos_0, 0.0f);
        isl_18.position_err_0 = vec4<f32>(rg_16.pos_err_0, 0.0f);
        isl_18.velocity_0 = vec4<f32>(rg_16.vel_0, 0.0f);
        isl_18.velocity_err_0 = vec4<f32>(rg_16.vel_err_0, 0.0f);
        isl_18.angular_velocity_0 = vec4<f32>(rg_16.w_1, 0.0f);
        isl_18.momentum_0 = vec4<f32>(rg_16.l_0, 0.0f);
        isl_18.momentum_err_0 = vec4<f32>(rg_16.l_err_0, 0.0f);
        isl_18.done_0 = done_5;
        isl_18.info_0[i32(1)] = isl_18.info_0[i32(1)] - done_5;
        if(g_halt_0 != u32(0))
        {
            _S495 = contact_island_4;
        }
        else
        {
            _S495 = false;
        }
        if(_S495)
        {
            contact_split_at_0(isl_18.info_0.w + done_5);
        }
        var _S515 : f32 = wsum_4.x;
        var _S516 : f32 = isl_18.energy_0[i32(0)];
        var _S517 : f32 = isl_18.energy_0[i32(1)];
        comp_add1_3(&(_S516), &(_S517), _S515);
        isl_18.energy_0[i32(0)] = _S516;
        isl_18.energy_0[i32(1)] = _S517 + wsum_4.y;
        isl_18.info_0[i32(3)] = isl_18.info_0[i32(3)] + done_5;
        if((((isl_18.info_0.x) & (u32(8)))) == u32(0))
        {
            isl_18.info_0[i32(2)] = ((isl_18.info_0[i32(2)]) & (u32(4294967293)));
        }
        if(g_halt_0 != u32(0))
        {
            isl_18.info_0[i32(2)] = ((isl_18.info_0[i32(2)]) | (u32(1)));
        }
        var _S518 : u32 = seen_bits_0();
        if(g_halt_0 != u32(0))
        {
            ct_5 = u32(128);
        }
        else
        {
            ct_5 = u32(0);
        }
        var _S519 : u32 = (_S518 | (ct_5));
        if(woke_8)
        {
            ct_5 = u32(2048);
        }
        else
        {
            ct_5 = u32(0);
        }
        var _S520 : u32 = (_S519 | (ct_5));
        if(resumed_8)
        {
            ct_5 = u32(2097280);
        }
        else
        {
            ct_5 = u32(0);
        }
        var cov_21 : u32 = (_S520 | (ct_5));
        if(done_5 > u32(0))
        {
            var quiet_9 : bool = (((isl_18.info_0.x) & (u32(32)))) != u32(0);
            if(quiet_9)
            {
                ct_5 = u32(4096);
            }
            else
            {
                ct_5 = u32(0);
            }
            var cov_22 : u32 = (cov_21 | (ct_5));
            if((((isl_18.info_0.x) & (u32(16)))) != u32(0))
            {
                _S495 = !quiet_9;
            }
            else
            {
                _S495 = false;
            }
            if(_S495)
            {
                ct_5 = u32(512);
            }
            else
            {
                ct_5 = u32(0);
            }
            var cov_23 : u32 = (cov_22 | (ct_5));
            if((params_0.contact_mode_0) == u32(1))
            {
                ct_5 = u32(524288);
            }
            else
            {
                ct_5 = u32(0);
            }
            var cov_24 : u32 = (cov_23 | (ct_5));
            if((isl_18.probes_0.y) > (isl_18.probes_0.x))
            {
                ct_5 = u32(262144);
            }
            else
            {
                ct_5 = u32(0);
            }
            cov_20 = (cov_24 | (ct_5));
        }
        else
        {
            cov_20 = cov_21;
        }
        isl_18.coverage_0 = ((isl_18.coverage_0) | (cov_20));
        if(woke_8)
        {
            isl_18.info_0[i32(0)] = ((isl_18.info_0[i32(0)]) & (u32(4294967279)));
            isl_18.info_0[i32(2)] = ((isl_18.info_0[i32(2)]) | (u32(4)));
        }
        islands_0[gi_4].range_0 = isl_18.range_0;
        islands_0[gi_4].info_0 = isl_18.info_0;
        islands_0[gi_4].com_0 = isl_18.com_0;
        islands_0[gi_4].inertia0_0 = isl_18.inertia0_0;
        islands_0[gi_4].inertia1_0 = isl_18.inertia1_0;
        islands_0[gi_4].inertia2_0 = isl_18.inertia2_0;
        islands_0[gi_4].inv0_0 = isl_18.inv0_0;
        islands_0[gi_4].inv1_0 = isl_18.inv1_0;
        islands_0[gi_4].inv2_0 = isl_18.inv2_0;
        islands_0[gi_4].wcom_0 = isl_18.wcom_0;
        islands_0[gi_4].winv0_0 = isl_18.winv0_0;
        islands_0[gi_4].winv1_0 = isl_18.winv1_0;
        islands_0[gi_4].winv2_0 = isl_18.winv2_0;
        islands_0[gi_4].rotation_0 = isl_18.rotation_0;
        islands_0[gi_4].position_0 = isl_18.position_0;
        islands_0[gi_4].position_err_0 = isl_18.position_err_0;
        islands_0[gi_4].velocity_0 = isl_18.velocity_0;
        islands_0[gi_4].velocity_err_0 = isl_18.velocity_err_0;
        islands_0[gi_4].angular_velocity_0 = isl_18.angular_velocity_0;
        islands_0[gi_4].done_0 = isl_18.done_0;
        islands_0[gi_4].coverage_0 = isl_18.coverage_0;
        islands_0[gi_4].wide_groups_0 = isl_18.wide_groups_0;
        islands_0[gi_4].done_pad_0 = isl_18.done_pad_0;
        islands_0[gi_4].probes_0 = isl_18.probes_0;
        islands_0[gi_4].energy_0 = isl_18.energy_0;
        islands_0[gi_4].rotation_err_0 = isl_18.rotation_err_0;
        islands_0[gi_4].momentum_0 = isl_18.momentum_0;
        islands_0[gi_4].momentum_err_0 = isl_18.momentum_err_0;
    }
    return;
}

fn group_sum3_2( tid_10 : u32,  a_16 : ptr<function, vec3<f32>>,  b_24 : ptr<function, vec3<f32>>)
{
    var x_13 : vec4<f32> = vec4<f32>((*a_16), 0.0f);
    var y_3 : vec4<f32> = vec4<f32>((*b_24), 0.0f);
    group_sum2_0(tid_10, &(x_13), &(y_3));
    (*a_16) = x_13.xyz;
    (*b_24) = y_3.xyz;
    return;
}

fn group_sum2_1( tid_11 : u32,  a_17 : ptr<function, vec4<f32>>,  b_25 : ptr<function, vec4<f32>>)
{
    if(tid_11 < u32(256))
    {
        g_red_a_0[tid_11] = (*a_17);
        g_red_b_0[tid_11] = (*b_25);
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
        if(tid_11 < s_7)
        {
            var _S521 : u32 = tid_11 + s_7;
            g_red_a_0[tid_11] = g_red_a_0[tid_11] + g_red_a_0[_S521];
            g_red_b_0[tid_11] = g_red_b_0[tid_11] + g_red_b_0[_S521];
        }
        workgroupBarrier();
        s_7 = (s_7 >> (u32(1)));
    }
    (*a_17) = g_red_a_0[i32(0)];
    (*b_25) = g_red_b_0[i32(0)];
    workgroupBarrier();
    return;
}

fn group_sum3_3( tid_12 : u32,  a_18 : ptr<function, vec3<f32>>,  b_26 : ptr<function, vec3<f32>>)
{
    var x_14 : vec4<f32> = vec4<f32>((*a_18), 0.0f);
    var y_4 : vec4<f32> = vec4<f32>((*b_26), 0.0f);
    group_sum2_1(tid_12, &(x_14), &(y_4));
    (*a_18) = x_14.xyz;
    (*b_26) = y_4.xyz;
    return;
}

fn group_sum2_2( tid_13 : u32,  a_19 : ptr<function, vec4<f32>>,  b_27 : ptr<function, vec4<f32>>)
{
    if(tid_13 < u32(256))
    {
        g_red_a_0[tid_13] = (*a_19);
        g_red_b_0[tid_13] = (*b_27);
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
        if(tid_13 < s_8)
        {
            var _S522 : u32 = tid_13 + s_8;
            g_red_a_0[tid_13] = g_red_a_0[tid_13] + g_red_a_0[_S522];
            g_red_b_0[tid_13] = g_red_b_0[tid_13] + g_red_b_0[_S522];
        }
        workgroupBarrier();
        s_8 = (s_8 >> (u32(1)));
    }
    (*a_19) = g_red_a_0[i32(0)];
    (*b_27) = g_red_b_0[i32(0)];
    workgroupBarrier();
    return;
}

fn group_sum2_3( tid_14 : u32,  a_20 : ptr<function, vec4<f32>>,  b_28 : ptr<function, vec4<f32>>)
{
    if(tid_14 < u32(256))
    {
        g_red_a_0[tid_14] = (*a_20);
        g_red_b_0[tid_14] = (*b_28);
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
        if(tid_14 < s_9)
        {
            var _S523 : u32 = tid_14 + s_9;
            g_red_a_0[tid_14] = g_red_a_0[tid_14] + g_red_a_0[_S523];
            g_red_b_0[tid_14] = g_red_b_0[tid_14] + g_red_b_0[_S523];
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
    var _S524 : vec4<u32> = islands_0[params_0.halt_index_0].info_0;
    var _S525 : bool;
    if((((islands_0[params_0.halt_index_0].info_0.z) & (u32(1)))) != u32(0))
    {
        _S525 = true;
    }
    else
    {
        _S525 = (_S524.y) != u32(0);
    }
    return _S525;
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
            var _S526 : u32 = i_7 - u32(6);
            l_4 = cross(axes_0[_S526 / u32(3)], axes_0[u32(3) + _S526 % u32(3)]);
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
    var _S527 : vec3<f32> = b_31.center_1 - a_23.center_1;
    var _S528 : f32 = 0.00000999999974738f * (length(a_23.half_2) + length(b_31.half_2));
    var _S529 : array<vec3<f32>, i32(6)> = array<vec3<f32>, i32(6)>( a_23.axis0_0, a_23.axis1_0, a_23.axis2_0, b_31.axis0_0, b_31.axis1_0, b_31.axis2_0 );
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
            l_5 = _S529[i_8];
        }
        else
        {
            var _S530 : u32 = i_8 - u32(6);
            l_5 = cross(_S529[_S530 / u32(3)], _S529[u32(3) + _S530 % u32(3)]);
        }
        var len_1 : f32 = length(l_5);
        if(len_1 <= 9.99999997475242708e-07f)
        {
            i_8 = i_8 + u32(1);
            continue;
        }
        if((abs(dot(_S527, l_5))) > (a_23.half_2.x * abs(dot(a_23.axis0_0, l_5)) + a_23.half_2.y * abs(dot(a_23.axis1_0, l_5)) + a_23.half_2.z * abs(dot(a_23.axis2_0, l_5)) + (b_31.half_2.x * abs(dot(b_31.axis0_0, l_5)) + b_31.half_2.y * abs(dot(b_31.axis1_0, l_5)) + b_31.half_2.z * abs(dot(b_31.axis2_0, l_5))) + _S528 * len_1))
        {
            return false;
        }
        i_8 = i_8 + u32(1);
    }
    return true;
}

fn chunk_velocity_1( c_35 : u32,  v_15 : ptr<function, vec3<f32>>,  w_10 : ptr<function, vec3<f32>>)
{
    var q_9 : Quat_0 = quat_of_0(islands_0[chunks_0[c_35].info_1.y].rotation_0);
    var _S531 : u32 = u32(4) * c_35;
    var _S532 : vec3<f32> = islands_0[chunks_0[c_35].info_1.y].angular_velocity_0.xyz;
    (*v_15) = islands_0[chunks_0[c_35].info_1.y].velocity_0.xyz + islands_0[chunks_0[c_35].info_1.y].velocity_err_0.xyz + cross(_S532, rotate_0(q_9, chunks_0[c_35].center_0.xyz + state_0[_S531].xyz - islands_0[chunks_0[c_35].info_1.y].com_0.xyz)) + rotate_0(q_9, state_0[_S531 + u32(2)].xyz);
    (*w_10) = _S532 + rotate_0(q_9, state_0[_S531 + u32(3)].xyz);
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
    var f_19 : u32 = u32(0);
    loop
    {
        if(f_19 < (b_33.hull_f_0))
        {
        }
        else
        {
            break;
        }
        var plane_0 : vec4<f32> = loads_0[b_33.hull_at_0 + b_33.hull_v_0 + f_19];
        var d_5 : f32 = dot(plane_0.xyz, local_2) - plane_0.w;
        if(d_5 > best_0)
        {
            (*face_0) = f_19;
            best_0 = d_5;
        }
        f_19 = f_19 + u32(1);
    }
    return best_0;
}

fn hull_signed_distance_1( b_34 : Box_0,  local_3 : vec3<f32>,  face_1 : ptr<function, u32>) -> f32
{
    (*face_1) = u32(0);
    var best_1 : f32 = -1.00000001504746622e+30f;
    var f_20 : u32 = u32(0);
    loop
    {
        if(f_20 < (b_34.hull_f_0))
        {
        }
        else
        {
            break;
        }
        var plane_1 : vec4<f32> = loads_0[b_34.hull_at_0 + b_34.hull_v_0 + f_20];
        var d_6 : f32 = dot(plane_1.xyz, local_3) - plane_1.w;
        if(d_6 > best_1)
        {
            (*face_1) = f_20;
            best_1 = d_6;
        }
        f_20 = f_20 + u32(1);
    }
    return best_1;
}

fn penetration_0( b_35 : Box_0,  p_5 : vec3<f32>,  depth_1 : ptr<function, f32>,  normal_3 : ptr<function, vec3<f32>>) -> bool
{
    (*depth_1) = 0.0f;
    (*normal_3) = vec3<f32>(0.0f);
    var r_12 : vec3<f32> = p_5 - b_35.center_1;
    var _S533 : vec3<f32> = b_35.half_2;
    if((dot(r_12, r_12)) > (dot(_S533, _S533) * 1.00001001358032227f))
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
        var local_4 : f32 = dot(r_12, box_axis_0(b_35, k_18));
        var d_8 : f32 = comp3_0(b_35.half_2, k_18) - abs(local_4);
        if(d_8 <= 0.0f)
        {
            return false;
        }
        if(d_8 < best_2)
        {
            var _S534 : f32;
            if(local_4 >= 0.0f)
            {
                _S534 = 1.0f;
            }
            else
            {
                _S534 = -1.0f;
            }
            best_2 = d_8;
            axis_3 = k_18;
            side_0 = _S534;
        }
        k_18 = k_18 + u32(1);
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
    var _S535 : vec3<f32> = b_36.half_2;
    if((dot(r_13, r_13)) > (dot(_S535, _S535) * 1.00001001358032227f))
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
        var local_5 : f32 = dot(r_13, box_axis_0(b_36, k_19));
        var d_10 : f32 = comp3_0(b_36.half_2, k_19) - abs(local_5);
        if(d_10 <= 0.0f)
        {
            return false;
        }
        if(d_10 < best_3)
        {
            var _S536 : f32;
            if(local_5 >= 0.0f)
            {
                _S536 = 1.0f;
            }
            else
            {
                _S536 = -1.0f;
            }
            best_3 = d_10;
            axis_4 = k_19;
            side_1 = _S536;
        }
        k_19 = k_19 + u32(1);
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
    var _S537 : vec3<f32> = b_37.half_2;
    if((dot(r_14, r_14)) > (dot(_S537, _S537) * 1.00001001358032227f))
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
    var k_20 : u32 = u32(0);
    loop
    {
        if(k_20 < u32(3))
        {
        }
        else
        {
            break;
        }
        var local_6 : f32 = dot(r_14, box_axis_0(b_37, k_20));
        var d_12 : f32 = comp3_0(b_37.half_2, k_20) - abs(local_6);
        if(d_12 <= 0.0f)
        {
            return false;
        }
        if(d_12 < best_4)
        {
            var _S538 : f32;
            if(local_6 >= 0.0f)
            {
                _S538 = 1.0f;
            }
            else
            {
                _S538 = -1.0f;
            }
            best_4 = d_12;
            axis_5 = k_20;
            side_2 = _S538;
        }
        k_20 = k_20 + u32(1);
    }
    (*depth_3) = best_4;
    (*normal_5) = box_axis_0(b_37, axis_5) * vec3<f32>(side_2);
    return true;
}

fn pair_point_0( ba_0 : Box_0,  bb_0 : Box_0,  na_0 : u32,  e_4 : u32,  p_8 : ptr<function, vec3<f32>>,  n_3 : ptr<function, vec3<f32>>,  d_13 : ptr<function, f32>) -> bool
{
    var from_a_0 : bool = e_4 < na_0;
    var _S539 : Box_0;
    if(from_a_0)
    {
        _S539 = ba_0;
    }
    else
    {
        _S539 = bb_0;
    }
    var _S540 : u32;
    if(from_a_0)
    {
        _S540 = e_4;
    }
    else
    {
        _S540 = e_4 - na_0;
    }
    (*p_8) = sample_point_0(_S539, _S540);
    if(from_a_0)
    {
        _S539 = bb_0;
    }
    else
    {
        _S539 = ba_0;
    }
    var _S541 : bool = penetration_2(_S539, (*p_8), &((*d_13)), &((*n_3)));
    if(!_S541)
    {
        return false;
    }
    if(!from_a_0)
    {
        (*n_3) = (vec3<f32>(0) - (*n_3));
    }
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

fn penalty_force_1( k_21 : f32,  m_red_1 : f32,  friction_1 : f32,  depth_4 : f32,  normal_6 : vec3<f32>,  rel_velocity_1 : vec3<f32>,  dt_15 : f32,  points_1 : u32,  stored_6 : ptr<function, f32>,  dissipated_9 : ptr<function, f32>) -> vec3<f32>
{
    var c_max_1 : f32 = 1.0f / max(f32(points_1), 10.0f) * m_red_1 / dt_15;
    var vn_1 : f32 = dot(rel_velocity_1, normal_6);
    var _S542 : f32 = k_21 * depth_4;
    var _S543 : f32 = min(2.0f * params_0.zeta_0 * sqrt(k_21 * m_red_1), c_max_1) * vn_1;
    var _S544 : f32 = _S542 - _S543;
    var _S545 : f32 = max(_S544, 0.0f);
    var vt_1 : vec3<f32> = rel_velocity_1 - normal_6 * vec3<f32>(vn_1);
    var vt_mag_1 : f32 = length(vt_1);
    var _S546 : f32 = friction_1 * _S545;
    var _S547 : f32 = min(_S546, min(c_max_1, _S546 / 0.00100000004749745f) * vt_mag_1);
    var ft_1 : vec3<f32>;
    if(vt_mag_1 > 0.0f)
    {
        ft_1 = (vec3<f32>(0) - vt_1) * vec3<f32>((_S547 / vt_mag_1));
    }
    else
    {
        ft_1 = vec3<f32>(0.0f);
    }
    (*stored_6) = 0.5f * k_21 * depth_4 * depth_4;
    var damping_power_1 : f32;
    if(_S544 > 0.0f)
    {
        damping_power_1 = _S543 * vn_1;
    }
    else
    {
        damping_power_1 = _S542 * max(vn_1, 0.0f);
    }
    (*dissipated_9) = (damping_power_1 + length(ft_1) * vt_mag_1) * dt_15;
    return normal_6 * vec3<f32>(_S545) + ft_1;
}

fn penalty_force_2( k_22 : f32,  m_red_2 : f32,  friction_2 : f32,  depth_5 : f32,  normal_7 : vec3<f32>,  rel_velocity_2 : vec3<f32>,  dt_16 : f32,  points_2 : u32,  stored_7 : ptr<function, f32>,  dissipated_10 : ptr<function, f32>) -> vec3<f32>
{
    var c_max_2 : f32 = 1.0f / max(f32(points_2), 10.0f) * m_red_2 / dt_16;
    var vn_2 : f32 = dot(rel_velocity_2, normal_7);
    var _S548 : f32 = k_22 * depth_5;
    var _S549 : f32 = min(2.0f * params_0.zeta_0 * sqrt(k_22 * m_red_2), c_max_2) * vn_2;
    var _S550 : f32 = _S548 - _S549;
    var _S551 : f32 = max(_S550, 0.0f);
    var vt_2 : vec3<f32> = rel_velocity_2 - normal_7 * vec3<f32>(vn_2);
    var vt_mag_2 : f32 = length(vt_2);
    var _S552 : f32 = friction_2 * _S551;
    var _S553 : f32 = min(_S552, min(c_max_2, _S552 / 0.00100000004749745f) * vt_mag_2);
    var ft_2 : vec3<f32>;
    if(vt_mag_2 > 0.0f)
    {
        ft_2 = (vec3<f32>(0) - vt_2) * vec3<f32>((_S553 / vt_mag_2));
    }
    else
    {
        ft_2 = vec3<f32>(0.0f);
    }
    (*stored_7) = 0.5f * k_22 * depth_5 * depth_5;
    var damping_power_2 : f32;
    if(_S550 > 0.0f)
    {
        damping_power_2 = _S549 * vn_2;
    }
    else
    {
        damping_power_2 = _S548 * max(vn_2, 0.0f);
    }
    (*dissipated_10) = (damping_power_2 + length(ft_2) * vt_mag_2) * dt_16;
    return normal_7 * vec3<f32>(_S551) + ft_2;
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
    var t_22 : f32 = (*sum_7) + x_20;
    if((abs((*sum_7))) >= (abs(x_20)))
    {
        (*err_6) = (*err_6) + ((*sum_7) - t_22 + x_20);
    }
    else
    {
        (*err_6) = (*err_6) + (x_20 - t_22 + (*sum_7));
    }
    (*sum_7) = t_22;
    return;
}

fn comp_add1_5( sum_8 : ptr<function, f32>,  err_7 : ptr<function, f32>,  x_21 : f32)
{
    var t_23 : f32 = (*sum_8) + x_21;
    if((abs((*sum_8))) >= (abs(x_21)))
    {
        (*err_7) = (*err_7) + ((*sum_8) - t_23 + x_21);
    }
    else
    {
        (*err_7) = (*err_7) + (x_21 - t_23 + (*sum_8));
    }
    (*sum_8) = t_23;
    return;
}

fn pair_contact_0( i_9 : u32,  lane_1 : u32,  lanes_5 : u32)
{
    var at_5 : u32 = params_0.pair_index_0 + u32(6) * i_9;
    var ca_0 : u32 = index_0[at_5];
    var cb_0 : u32 = index_0[at_5 + u32(1)];
    var _S554 : u32 = index_0[at_5 + u32(3)];
    var _S555 : f32 = (bitcast<f32>((index_0[at_5 + u32(4)])));
    var _S556 : f32 = (bitcast<f32>((index_0[at_5 + u32(5)])));
    var _S557 : f32 = params_0.dt_0;
    var out_0 : u32 = params_0.slot_base_0 + u32(2) * index_0[at_5 + u32(2)];
    var cb_rel_0 : vec3<f32> = world_diff_0(chunk_world_0(cb_0), chunk_world_0(ca_0));
    var touching_0 : bool = !((length(cb_rel_0)) > (chunks_0[ca_0].half_0.w + chunks_0[cb_0].half_0.w));
    var ledger_2 : vec4<f32> = scratch_0[params_0.ledger_base_0 + i_9];
    var flags_1 : u32 = (bitcast<u32>((scratch_0[params_0.ledger_base_0 + i_9].w)));
    var _S558 : vec3<f32> = vec3<f32>(0.0f);
    var ba_1 : Box_0 = chunk_box_0(ca_0, _S558);
    var bb_1 : Box_0 = chunk_box_0(cb_0, cb_rel_0);
    var _S559 : u32 = sample_count_0(ba_1);
    var _S560 : u32 = _S559 + sample_count_0(bb_1);
    var has_state_0 : bool;
    var e_5 : u32;
    if(!touching_0)
    {
        if(((flags_1 & (u32(1)))) != u32(0))
        {
            e_5 = lane_1;
            loop
            {
                if(e_5 < _S560)
                {
                }
                else
                {
                    break;
                }
                contact_state_0[_S554 + e_5] = vec4<f32>((bitcast<f32>((u32(2143289344)))), 0.0f, 0.0f, 0.0f);
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
            if(e_5 < _S560)
            {
            }
            else
            {
                break;
            }
            var p_9 : vec3<f32>;
            var n_4 : vec3<f32>;
            var d_14 : f32;
            var _S561 : bool = pair_point_0(ba_1, bb_1, _S559, e_5, &(p_9), &(n_4), &(d_14));
            if(!_S561)
            {
                var _S562 : u32 = _S554 + e_5;
                if(!is_nan_0(contact_state_0[_S562].x))
                {
                    contact_state_0[_S562] = vec4<f32>((bitcast<f32>((u32(2143289344)))), 0.0f, 0.0f, 0.0f);
                }
                e_5 = e_5 + lanes_5;
                continue;
            }
            var _S563 : u32 = count_3 + u32(1);
            var _S564 : u32 = _S554 + e_5;
            var entry_4 : vec4<f32> = contact_state_0[_S564];
            if(is_nan_0(contact_state_0[_S564].x))
            {
                has_state_0 = true;
            }
            else
            {
                has_state_0 = (dot(entry_4.yzw, n_4)) < 0.99000000953674316f;
            }
            if(has_state_0)
            {
                if(d_14 > (2.0f * abs(dot(va0_0 + cross(wa0_0, p_9) - (vb0_0 + cross(wb0_0, p_9 - bb_1.center_1)), n_4)) * _S557 + 9.99999971718068537e-10f))
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
            contact_state_0[_S564] = entry_4;
            var engaged_1 : u32;
            if((d_14 - entry_4.x) > 0.0f)
            {
                engaged_1 = engaged_0 + u32(1);
            }
            else
            {
                engaged_1 = engaged_0;
            }
            count_3 = _S563;
            engaged_0 = engaged_1;
            e_5 = e_5 + lanes_5;
        }
        var engaged_2 : u32 = pair_count_sum_0(engaged_0, lanes_5);
        if((pair_count_sum_0(count_3, lanes_5)) > u32(0))
        {
            var _S565 : f32 = contact_stiffness_0(chunks_0[ca_0].cmat_0.x, ba_1, chunks_0[cb_0].cmat_0.x, bb_1, bb_1.center_1 - ba_1.center_1) / max(f32(engaged_2), 10.0f);
            e_5 = lane_1;
            fa_2 = _S558;
            ta_2 = _S558;
            fb_0 = _S558;
            tb_2 = _S558;
            stored_sum_0 = 0.0f;
            diss_sum_0 = 0.0f;
            loop
            {
                if(e_5 < _S560)
                {
                }
                else
                {
                    break;
                }
                var p_10 : vec3<f32>;
                var n_5 : vec3<f32>;
                var d_15 : f32;
                var _S566 : bool = pair_point_0(ba_1, bb_1, _S559, e_5, &(p_10), &(n_5), &(d_15));
                if(!_S566)
                {
                    e_5 = e_5 + lanes_5;
                    continue;
                }
                var eff_0 : f32 = d_15 - contact_state_0[_S554 + e_5].x;
                if(eff_0 <= 0.0f)
                {
                    e_5 = e_5 + lanes_5;
                    continue;
                }
                var stored_8 : f32;
                var diss_4 : f32;
                var f_21 : vec3<f32> = penalty_force_2(_S565, _S555, _S556, eff_0, n_5, va0_0 + cross(wa0_0, p_10) - (vb0_0 + cross(wb0_0, p_10 - bb_1.center_1)), _S557, engaged_2, &(stored_8), &(diss_4));
                var ta_3 : vec3<f32> = ta_2 + cross(p_10, f_21);
                var _S567 : vec3<f32> = (vec3<f32>(0) - f_21);
                var fb_1 : vec3<f32> = fb_0 + _S567;
                var tb_3 : vec3<f32> = tb_2 + cross(p_10 - bb_1.center_1, _S567);
                var stored_sum_1 : f32 = stored_sum_0 + stored_8;
                var diss_sum_1 : f32 = diss_sum_0 + diss_4;
                fa_2 = fa_2 + f_21;
                ta_2 = ta_3;
                fb_0 = fb_1;
                tb_2 = tb_3;
                stored_sum_0 = stored_sum_1;
                diss_sum_0 = diss_sum_1;
                e_5 = e_5 + lanes_5;
            }
            var _S568 : vec3<f32> = pair_sum3_0(fa_2, lanes_5);
            var _S569 : vec3<f32> = pair_sum3_0(ta_2, lanes_5);
            var _S570 : vec3<f32> = pair_sum3_0(fb_0, lanes_5);
            var _S571 : vec3<f32> = pair_sum3_0(tb_2, lanes_5);
            var _S572 : f32 = pair_sum_0(stored_sum_0, lanes_5);
            var _S573 : f32 = pair_sum_0(diss_sum_0, lanes_5);
            has_state_0 = true;
            diss_sum_0 = _S573;
            fa_2 = _S568;
            ta_2 = _S569;
            fb_0 = _S570;
            tb_2 = _S571;
            stored_sum_0 = _S572;
        }
        else
        {
            has_state_0 = false;
            diss_sum_0 = 0.0f;
            fa_2 = _S558;
            ta_2 = _S558;
            fb_0 = _S558;
            tb_2 = _S558;
            stored_sum_0 = 0.0f;
        }
    }
    else
    {
        has_state_0 = false;
        diss_sum_0 = 0.0f;
        fa_2 = _S558;
        ta_2 = _S558;
        fb_0 = _S558;
        tb_2 = _S558;
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
            if(e_5 < _S560)
            {
            }
            else
            {
                break;
            }
            contact_state_0[_S554 + e_5] = vec4<f32>((bitcast<f32>((u32(2143289344)))), 0.0f, 0.0f, 0.0f);
            e_5 = e_5 + lanes_5;
        }
    }
    if(lane_1 != u32(0))
    {
        return;
    }
    if(has_state_0)
    {
        var _S574 : f32 = ledger_2[i32(1)];
        var _S575 : f32 = ledger_2[i32(2)];
        comp_add1_5(&(_S574), &(_S575), diss_sum_0);
        ledger_2[i32(1)] = _S574;
        ledger_2[i32(2)] = _S575;
    }
    var _S576 : vec3<f32> = vec3<f32>(0.0f);
    if((any((fa_2 != _S576))))
    {
        loaded_0 = true;
    }
    else
    {
        loaded_0 = (any((ta_2 != _S576)));
    }
    if(loaded_0)
    {
        loaded_0 = true;
    }
    else
    {
        loaded_0 = (any((fb_0 != _S576)));
    }
    if(loaded_0)
    {
        loaded_0 = true;
    }
    else
    {
        loaded_0 = (any((tb_2 != _S576)));
    }
    var _S577 : bool;
    if(loaded_0)
    {
        _S577 = true;
    }
    else
    {
        _S577 = ((flags_1 & (u32(2)))) != u32(0);
    }
    if(_S577)
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
    var _S578 : u32;
    if(((*imp_1).shape_0.x) == 0.0f)
    {
        _S578 = u32(1);
    }
    else
    {
        _S578 = sample_count_0(b_39) + u32(14);
    }
    return _S578;
}

fn impactor_slots_1( imp_2 : ptr<function, Impactor_std430_0>,  b_40 : Box_0) -> u32
{
    var _S579 : u32;
    if(((*imp_2).shape_0.x) == 0.0f)
    {
        _S579 = u32(1);
    }
    else
    {
        _S579 = sample_count_0(b_40) + u32(14);
    }
    return _S579;
}

fn impactor_slots_2( imp_3 : ptr<function, Impactor_std430_0>,  b_41 : Box_0) -> u32
{
    var _S580 : u32;
    if(((*imp_3).shape_0.x) == 0.0f)
    {
        _S580 = u32(1);
    }
    else
    {
        _S580 = sample_count_0(b_41) + u32(14);
    }
    return _S580;
}

fn sphere_contact_0( b_42 : Box_0,  center_4 : vec3<f32>,  radius_0 : f32,  point_0 : ptr<function, vec3<f32>>,  normal_8 : ptr<function, vec3<f32>>,  depth_6 : ptr<function, f32>) -> bool
{
    var _S581 : vec3<f32> = vec3<f32>(0.0f);
    (*point_0) = _S581;
    (*normal_8) = _S581;
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
        var _S582 : vec3<f32> = box_to_world_0(b_42, loads_0[b_42.hull_at_0 + b_42.hull_v_0 + face_5].xyz);
        (*normal_8) = _S582;
        (*point_0) = center_4 - _S582 * vec3<f32>(max(d_16, 0.0f));
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
    var _S583 : bool = penetration_0(b_42, center_4, &(inside_0), &(n_6));
    if(!_S583)
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
    var _S584 : vec3<f32> = vec3<f32>(0.0f);
    (*point_1) = _S584;
    (*normal_9) = _S584;
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
        var _S585 : vec3<f32> = box_to_world_0(b_43, loads_0[b_43.hull_at_0 + b_43.hull_v_0 + face_6].xyz);
        (*normal_9) = _S585;
        (*point_1) = center_5 - _S585 * vec3<f32>(max(d_18, 0.0f));
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
    var _S586 : bool = penetration_1(b_43, center_5, &(inside_1), &(n_7));
    if(!_S586)
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
    var _S587 : vec3<f32> = vec3<f32>(0.0f);
    (*p_11) = _S587;
    (*n_8) = _S587;
    (*d_20) = 0.0f;
    var _S588 : vec4<f32> = (*imp_4).shape_0;
    if(((*imp_4).shape_0.x) == 0.0f)
    {
        var _S589 : bool = sphere_contact_0(b_44, _S587, _S588.y - crush_depth_0, &((*p_11)), &((*n_8)), &((*d_20)));
        if(!_S589)
        {
            return false;
        }
        (*n_8) = (vec3<f32>(0) - (*n_8));
        return true;
    }
    var cb_1 : u32 = sample_count_0(b_44);
    if(j_2 < cb_1)
    {
        var _S590 : vec3<f32> = sample_point_0(b_44, j_2);
        (*p_11) = _S590;
        var _S591 : bool = penetration_0(shrunk_0, _S590, &((*d_20)), &((*n_8)));
        return _S591;
    }
    var _S592 : vec3<f32> = sample_point_0(shrunk_0, j_2 - cb_1);
    (*p_11) = _S592;
    var _S593 : bool = penetration_0(b_44, _S592, &((*d_20)), &((*n_8)));
    if(!_S593)
    {
        return false;
    }
    (*n_8) = (vec3<f32>(0) - (*n_8));
    return true;
}

fn impactor_contact_1( imp_5 : ptr<function, Impactor_std430_0>,  crush_depth_1 : f32,  shrunk_1 : Box_0,  b_45 : Box_0,  j_3 : u32,  p_12 : ptr<function, vec3<f32>>,  n_9 : ptr<function, vec3<f32>>,  d_21 : ptr<function, f32>) -> bool
{
    var _S594 : vec3<f32> = vec3<f32>(0.0f);
    (*p_12) = _S594;
    (*n_9) = _S594;
    (*d_21) = 0.0f;
    var _S595 : vec4<f32> = (*imp_5).shape_0;
    if(((*imp_5).shape_0.x) == 0.0f)
    {
        var _S596 : bool = sphere_contact_1(b_45, _S594, _S595.y - crush_depth_1, &((*p_12)), &((*n_9)), &((*d_21)));
        if(!_S596)
        {
            return false;
        }
        (*n_9) = (vec3<f32>(0) - (*n_9));
        return true;
    }
    var cb_2 : u32 = sample_count_0(b_45);
    if(j_3 < cb_2)
    {
        var _S597 : vec3<f32> = sample_point_0(b_45, j_3);
        (*p_12) = _S597;
        var _S598 : bool = penetration_1(shrunk_1, _S597, &((*d_21)), &((*n_9)));
        return _S598;
    }
    var _S599 : vec3<f32> = sample_point_0(shrunk_1, j_3 - cb_2);
    (*p_12) = _S599;
    var _S600 : bool = penetration_1(b_45, _S599, &((*d_21)), &((*n_9)));
    if(!_S600)
    {
        return false;
    }
    (*n_9) = (vec3<f32>(0) - (*n_9));
    return true;
}

fn impactor_point_0( _S601 : u32) -> WorldPoint_0
{
    var wi_0 : WorldPoint_0;
    wi_0.hi_0 = impactors_0[_S601].position_1.xyz;
    wi_0.lo_0 = impactors_0[_S601].position_err_1.xyz;
    wi_0.rel_0 = vec3<f32>(0.0f);
    return wi_0;
}

fn impactor_box_1( _S602 : u32,  _S603 : vec3<f32>,  _S604 : vec3<f32>) -> Box_0
{
    var q_13 : Quat_0 = quat_of_0(impactors_0[_S602].rotation_1);
    var b_46 : Box_0;
    b_46.center_1 = _S603;
    b_46.axis0_0 = rotate_0(q_13, vec3<f32>(1.0f, 0.0f, 0.0f));
    b_46.axis1_0 = rotate_0(q_13, vec3<f32>(0.0f, 1.0f, 0.0f));
    b_46.axis2_0 = rotate_0(q_13, vec3<f32>(0.0f, 0.0f, 1.0f));
    b_46.half_2 = _S604;
    b_46.hull_at_0 = u32(0);
    b_46.hull_v_0 = u32(0);
    b_46.hull_f_0 = u32(0);
    return b_46;
}

fn impactor_shrunk_0( _S605 : u32,  _S606 : f32,  _S607 : Box_0) -> Box_0
{
    var shrunk_2 : Box_0 = _S607;
    shrunk_2.half_2 = _S607.half_2 - min(vec3<f32>(_S606), _S607.half_2 * vec3<f32>(0.5f));
    return shrunk_2;
}

fn impactor_contact_2( _S608 : u32,  _S609 : f32,  _S610 : Box_0,  _S611 : Box_0,  _S612 : u32,  _S613 : ptr<function, vec3<f32>>,  _S614 : ptr<function, vec3<f32>>,  _S615 : ptr<function, f32>) -> bool
{
    var _S616 : Impactor_std430_0 = impactors_0[_S608];
    var _S617 : vec3<f32> = vec3<f32>(0.0f);
    (*_S613) = _S617;
    (*_S614) = _S617;
    (*_S615) = 0.0f;
    if((_S616.shape_0.x) == 0.0f)
    {
        var _S618 : bool = sphere_contact_1(_S611, _S617, _S616.shape_0.y - _S609, &((*_S613)), &((*_S614)), &((*_S615)));
        if(!_S618)
        {
            return false;
        }
        (*_S614) = (vec3<f32>(0) - (*_S614));
        return true;
    }
    var cb_3 : u32 = sample_count_0(_S611);
    if(_S612 < cb_3)
    {
        var _S619 : vec3<f32> = sample_point_0(_S611, _S612);
        (*_S613) = _S619;
        var _S620 : bool = penetration_1(_S610, _S619, &((*_S615)), &((*_S614)));
        return _S620;
    }
    var _S621 : vec3<f32> = sample_point_0(_S610, _S612 - cb_3);
    (*_S613) = _S621;
    var _S622 : bool = penetration_1(_S611, _S621, &((*_S615)), &((*_S614)));
    if(!_S622)
    {
        return false;
    }
    (*_S614) = (vec3<f32>(0) - (*_S614));
    return true;
}

fn impactor_candidate_forces_0( k_23 : u32,  lane_2 : u32,  lanes_6 : u32)
{
    var _S623 : u32;
    var _S624 : bool;
    var _S625 : u32 = u32(3) * k_23;
    var at_6 : u32 = params_0.cand_index_0 + _S625;
    var c_36 : u32 = index_0[at_6];
    var slot_1 : u32 = index_0[at_6 + u32(1)];
    var _S626 : u32 = index_0[at_6 + u32(2)];
    var _S627 : Impactor_std430_0 = impactors_0[_S626];
    var _S628 : f32 = params_0.dt_0;
    var _S629 : vec3<f32> = vec3<f32>(0.0f);
    var data_1 : vec4<f32> = scratch_0[params_0.cand_base_0 + _S625];
    var hit_0 : bool;
    var f_sum_0 : vec3<f32>;
    var t_sum_0 : vec3<f32>;
    var imp_f_0 : vec3<f32>;
    var imp_t_0 : vec3<f32>;
    if((_S627.cand_0.z) == u32(0))
    {
        var rel_1 : vec3<f32> = world_diff_0(chunk_world_0(c_36), impactor_point_0(_S626));
        var _S630 : vec4<f32> = _S627.half_1;
        if(!((length(rel_1)) > (_S627.half_1.w + chunks_0[c_36].half_0.w)))
        {
            var _S631 : Box_0 = impactor_box_1(_S626, _S629, _S630.xyz);
            var b_47 : Box_0 = chunk_box_0(c_36, rel_1);
            var _S632 : vec4<f32> = _S627.mat_0;
            var kc_0 : f32 = contact_stiffness_0(_S627.mat_0.x, _S631, chunks_0[c_36].cmat_0.x, b_47, b_47.center_1 - _S631.center_1);
            var _S633 : vec4<f32> = _S627.geom_0;
            var _S634 : f32 = _S627.geom_0.y;
            var _S635 : Box_0 = impactor_shrunk_0(_S626, _S634, _S631);
            var j_4 : u32 = lane_2;
            var n_10 : u32 = u32(0);
            loop
            {
                var _S636 : u32 = impactor_slots_0(&(_S627), b_47);
                _S623 = _S636;
                if(j_4 < _S636)
                {
                }
                else
                {
                    break;
                }
                var p_13 : vec3<f32>;
                var nrm_0 : vec3<f32>;
                var dep_0 : f32;
                if(impactor_contact_2(_S626, _S634, _S635, b_47, j_4, &(p_13), &(nrm_0), &(dep_0)))
                {
                    n_10 = n_10 + u32(1);
                }
                j_4 = j_4 + lanes_6;
            }
            var n_11 : u32 = pair_count_sum_0(n_10, lanes_6);
            var _S637 : bool = (_S627.shape_0.x) == 0.0f;
            var _S638 : f32;
            if(_S637)
            {
                _S638 = kc_0;
            }
            else
            {
                _S638 = kc_0 / max(f32(n_11), 10.0f);
            }
            if(_S637)
            {
                j_4 = u32(1);
            }
            else
            {
                j_4 = n_11;
            }
            var m_5 : f32 = chunks_0[c_36].center_0.w;
            var _S639 : f32 = _S632.z;
            var _S640 : f32 = m_5 * _S639 / (m_5 + _S639);
            var _S641 : f32;
            if((params_0.pair_friction_0) >= 0.0f)
            {
                _S641 = params_0.pair_friction_0;
            }
            else
            {
                _S641 = min(_S632.y, chunks_0[c_36].cmat_0.y);
            }
            var _S642 : vec3<f32> = _S627.velocity_1.xyz + _S627.velocity_err_1.xyz;
            var vc_1 : vec3<f32>;
            var wc_1 : vec3<f32>;
            chunk_velocity_1(c_36, &(vc_1), &(wc_1));
            hit_0 = false;
            var j_5 : u32 = lane_2;
            f_sum_0 = _S629;
            t_sum_0 = _S629;
            imp_f_0 = _S629;
            imp_t_0 = _S629;
            var diss_sum_2 : f32 = 0.0f;
            loop
            {
                var _S643 : bool = n_11 > u32(0);
                _S624 = _S643;
                var _S644 : bool;
                if(_S643)
                {
                    _S644 = j_5 < _S623;
                }
                else
                {
                    _S644 = false;
                }
                if(_S644)
                {
                }
                else
                {
                    break;
                }
                var p_14 : vec3<f32>;
                var nrm_1 : vec3<f32>;
                var dep_1 : f32;
                var _S645 : bool = impactor_contact_0(&(_S627), _S634, _S635, b_47, j_5, &(p_14), &(nrm_1), &(dep_1));
                if(!_S645)
                {
                    j_5 = j_5 + lanes_6;
                    continue;
                }
                var stored_9 : f32;
                var diss_5 : f32;
                var f_22 : vec3<f32> = penalty_force_2(_S638, _S640, _S641, dep_1 * _S633.x, nrm_1, vc_1 + cross(wc_1, p_14 - b_47.center_1) - (_S642 + cross(_S627.angular_velocity_1.xyz, p_14)), _S628, j_4, &(stored_9), &(diss_5));
                var f_sum_1 : vec3<f32> = f_sum_0 + f_22;
                var t_sum_1 : vec3<f32> = t_sum_0 + cross(p_14 - b_47.center_1, f_22);
                var imp_f_1 : vec3<f32> = imp_f_0 - f_22;
                var imp_t_1 : vec3<f32> = imp_t_0 - cross(p_14, f_22);
                var diss_sum_3 : f32 = diss_sum_2 + diss_5;
                hit_0 = true;
                f_sum_0 = f_sum_1;
                t_sum_0 = t_sum_1;
                imp_f_0 = imp_f_1;
                imp_t_0 = imp_t_1;
                diss_sum_2 = diss_sum_3;
                j_5 = j_5 + lanes_6;
            }
            var _S646 : vec3<f32> = pair_sum3_0(f_sum_0, lanes_6);
            var _S647 : vec3<f32> = pair_sum3_0(t_sum_0, lanes_6);
            var _S648 : vec3<f32> = pair_sum3_0(imp_f_0, lanes_6);
            var _S649 : vec3<f32> = pair_sum3_0(imp_t_0, lanes_6);
            var diss_sum_4 : f32 = pair_sum_0(diss_sum_2, lanes_6);
            var _S650 : bool = pair_any_0(hit_0, lanes_6);
            if(_S624)
            {
                var _S651 : f32 = data_1[i32(2)];
                var _S652 : f32 = data_1[i32(3)];
                comp_add1_5(&(_S651), &(_S652), diss_sum_4);
                data_1[i32(2)] = _S651;
                data_1[i32(3)] = _S652;
            }
            hit_0 = _S650;
            f_sum_0 = _S646;
            t_sum_0 = _S647;
            imp_f_0 = _S648;
            imp_t_0 = _S649;
        }
        else
        {
            hit_0 = false;
            f_sum_0 = _S629;
            t_sum_0 = _S629;
            imp_f_0 = _S629;
            imp_t_0 = _S629;
        }
    }
    else
    {
        hit_0 = false;
        f_sum_0 = _S629;
        t_sum_0 = _S629;
        imp_f_0 = _S629;
        imp_t_0 = _S629;
    }
    if(lane_2 != u32(0))
    {
        return;
    }
    var _S653 : u32 = u32(2) * slot_1;
    scratch_0[params_0.slot_base_0 + _S653] = vec4<f32>(f_sum_0, 0.0f);
    scratch_0[params_0.slot_base_0 + _S653 + u32(1)] = vec4<f32>(t_sum_0, 0.0f);
    scratch_0[params_0.cand_base_0 + _S625] = data_1;
    scratch_0[params_0.cand_base_0 + _S625 + u32(1)] = vec4<f32>(imp_f_0, 0.0f);
    scratch_0[params_0.cand_base_0 + _S625 + u32(2)] = vec4<f32>(imp_t_0, 0.0f);
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
    var _S654 : vec4<u32> = impactors_0[ii_0].cand_0;
    var _S655 : bool;
    if((impactors_0[ii_0].cand_0.z) != u32(0))
    {
        _S655 = true;
    }
    else
    {
        _S655 = (start_hi_1.w) <= 0.0f;
    }
    if(_S655)
    {
        return;
    }
    var _S656 : u32 = _S654.w;
    var _S657 : vec3<f32> = vec3<f32>((params_0.t_hi_0 - scratch_0[at_7 + u32(3)].w + (params_0.t_lo_0 - scratch_0[at_7 + u32(1)].w) + f32(_S656 - params_0.step_start_0) * params_0.dt_0));
    var turn_1 : Quat_0 = quat_mul_0(quat_of_0(vec4<f32>((vec3<f32>(0) - scratch_0[at_7 + u32(2)].xyz), scratch_0[at_7 + u32(2)].w)), quat_of_0(impactors_0[ii_0].rotation_1));
    if((length(impactors_0[ii_0].position_1.xyz - start_hi_1.xyz + (impactors_0[ii_0].position_err_1.xyz - scratch_0[at_7 + u32(1)].xyz) - (scratch_0[at_7 + u32(3)].xyz * _S657 + vec3<f32>(0.5f) * params_0.gravity_0.xyz * _S657 * _S657)) + 2.0f * atan2(length(vec3<f32>(turn_1.x_0, turn_1.y_0, turn_1.z_0)), abs(turn_1.w_0)) * impactors_0[ii_0].half_1.w) > (start_hi_1.w))
    {
        travel_stop_0(_S656 + u32(1), fused_2);
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
            var _S658 : u32 = i_10 / lanes_7;
            var _S659 : u32 = i_10 % lanes_7;
            pair_contact_0(_S658, _S659, lanes_7);
        }
        return;
    }
    var j_6 : u32 = i_10 - pair_threads_0;
    var cand_threads_0 : u32 = params_0.cand_count_0 * lanes_7;
    if(j_6 < cand_threads_0)
    {
        if(!pair_any_0(stopped_0(), lanes_7))
        {
            var _S660 : u32 = j_6 / lanes_7;
            var _S661 : u32 = j_6 % lanes_7;
            impactor_candidate_forces_0(_S660, _S661, lanes_7);
        }
        return;
    }
    if(fused_3)
    {
        return;
    }
    var _S662 : u32 = j_6 - cand_threads_0;
    var _S663 : bool;
    if(_S662 < (params_0.chunk_count_0))
    {
        _S663 = !stopped_0();
    }
    else
    {
        _S663 = false;
    }
    if(_S663)
    {
        travel_check_0(_S662, false);
    }
    else
    {
        if((_S662 - params_0.chunk_count_0) < (params_0.impactor_count_0))
        {
            _S663 = !stopped_0();
        }
        else
        {
            _S663 = false;
        }
        if(_S663)
        {
            impactor_travel_check_0(_S662 - params_0.chunk_count_0, false);
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

fn impactor_contact_3( _S664 : u32,  _S665 : f32,  _S666 : Box_0,  _S667 : Box_0,  _S668 : u32,  _S669 : ptr<function, vec3<f32>>,  _S670 : ptr<function, vec3<f32>>,  _S671 : ptr<function, f32>) -> bool
{
    var _S672 : Impactor_std430_0 = impactors_0[_S664];
    var _S673 : vec3<f32> = vec3<f32>(0.0f);
    (*_S669) = _S673;
    (*_S670) = _S673;
    (*_S671) = 0.0f;
    if((_S672.shape_0.x) == 0.0f)
    {
        var _S674 : bool = sphere_contact_1(_S667, _S673, _S672.shape_0.y - _S665, &((*_S669)), &((*_S670)), &((*_S671)));
        if(!_S674)
        {
            return false;
        }
        (*_S670) = (vec3<f32>(0) - (*_S670));
        return true;
    }
    var cb_4 : u32 = sample_count_0(_S667);
    if(_S668 < cb_4)
    {
        var _S675 : vec3<f32> = sample_point_0(_S667, _S668);
        (*_S669) = _S675;
        var _S676 : bool = penetration_1(_S666, _S675, &((*_S671)), &((*_S670)));
        return _S676;
    }
    var _S677 : vec3<f32> = sample_point_0(_S666, _S668 - cb_4);
    (*_S669) = _S677;
    var _S678 : bool = penetration_1(_S667, _S677, &((*_S671)), &((*_S670)));
    if(!_S678)
    {
        return false;
    }
    (*_S670) = (vec3<f32>(0) - (*_S670));
    return true;
}

fn impactor_contact_count_0( _S679 : u32,  _S680 : f32,  _S681 : Box_0,  _S682 : Box_0) -> u32
{
    var _S683 : Impactor_std430_0 = impactors_0[_S679];
    var j_7 : u32 = u32(0);
    var count_4 : u32 = u32(0);
    loop
    {
        var _S684 : u32 = impactor_slots_1(&(_S683), _S682);
        if(j_7 < _S684)
        {
        }
        else
        {
            break;
        }
        var p_15 : vec3<f32>;
        var n_12 : vec3<f32>;
        var d_22 : f32;
        if(impactor_contact_3(_S679, _S680, _S681, _S682, j_7, &(p_15), &(n_12), &(d_22)))
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
    var k_24 : u32 = id_1.x;
    var _S685 : bool;
    if(k_24 >= (params_0.cand_count_0))
    {
        _S685 = true;
    }
    else
    {
        _S685 = stopped_0();
    }
    if(_S685)
    {
        return;
    }
    var _S686 : u32 = u32(3) * k_24;
    var at_8 : u32 = params_0.cand_index_0 + _S686;
    var c_37 : u32 = index_0[at_8];
    var _S687 : u32 = index_0[at_8 + u32(2)];
    var _S688 : Impactor_std430_0 = impactors_0[_S687];
    if((_S688.cand_0.z) == u32(0))
    {
        _S685 = (_S688.crush_0.x) > 0.0f;
    }
    else
    {
        _S685 = false;
    }
    var total_1 : f32;
    var ksum_0 : f32;
    if(_S685)
    {
        var rel_2 : vec3<f32> = world_diff_0(chunk_world_0(c_37), impactor_point_0(_S687));
        var _S689 : vec4<f32> = _S688.half_1;
        if(!((length(rel_2)) > (_S688.half_1.w + chunks_0[c_37].half_0.w)))
        {
            var _S690 : Box_0 = impactor_box_1(_S687, vec3<f32>(0.0f), _S689.xyz);
            var b_48 : Box_0 = chunk_box_0(c_37, rel_2);
            var kc_1 : f32 = contact_stiffness_0(_S688.mat_0.x, _S690, chunks_0[c_37].cmat_0.x, b_48, b_48.center_1 - _S690.center_1);
            var _S691 : f32 = _S688.crush_0.w;
            var _S692 : Box_0 = impactor_shrunk_0(_S687, _S691, _S690);
            var _S693 : u32 = impactor_contact_count_0(_S687, _S691, _S692, b_48);
            var _S694 : f32;
            if((_S688.shape_0.x) == 0.0f)
            {
                _S694 = kc_1;
            }
            else
            {
                _S694 = kc_1 / max(f32(_S693), 10.0f);
            }
            var j_8 : u32 = u32(0);
            total_1 = 0.0f;
            ksum_0 = 0.0f;
            loop
            {
                if(_S693 > u32(0))
                {
                    var _S695 : u32 = impactor_slots_2(&(_S688), b_48);
                    _S685 = j_8 < _S695;
                }
                else
                {
                    _S685 = false;
                }
                if(_S685)
                {
                }
                else
                {
                    break;
                }
                var p_16 : vec3<f32>;
                var nrm_2 : vec3<f32>;
                var dep_2 : f32;
                var _S696 : bool = impactor_contact_1(&(_S688), _S691, _S692, b_48, j_8, &(p_16), &(nrm_2), &(dep_2));
                if(!_S696)
                {
                    j_8 = j_8 + u32(1);
                    continue;
                }
                var ksum_1 : f32 = ksum_0 + _S694;
                total_1 = total_1 + _S694 * dep_2;
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
    scratch_0[params_0.cand_base_0 + _S686] = vec4<f32>(total_1, ksum_0, scratch_0[params_0.cand_base_0 + _S686].z, scratch_0[params_0.cand_base_0 + _S686].w);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn impactor_crush(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var ii_1 : u32 = group_0.x;
    var tid_15 : u32 = thread_0.x;
    var _S697 : bool;
    if(ii_1 >= (params_0.impactor_count_0))
    {
        _S697 = true;
    }
    else
    {
        _S697 = stopped_0();
    }
    if(_S697)
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
    var _S698 : vec4<f32> = vec4<f32>(0.0f);
    var shares_0 : vec4<f32> = _S698;
    var unused_5 : vec4<f32> = _S698;
    var k_25 : u32 = imp_6.cand_0.x + tid_15;
    loop
    {
        if(k_25 < (imp_6.cand_0.y))
        {
        }
        else
        {
            break;
        }
        shares_0 = shares_0 + scratch_0[params_0.cand_base_0 + u32(3) * k_25];
        k_25 = k_25 + u32(256);
    }
    group_sum2_1(tid_15, &(shares_0), &(unused_5));
    if(tid_15 != u32(0))
    {
        _S697 = true;
    }
    else
    {
        _S697 = (imp_6.cand_0.z) != u32(0);
    }
    if(_S697)
    {
        return;
    }
    imp_6.geom_0 = vec4<f32>(1.0f, imp_6.crush_0.w, 0.0f, 0.0f);
    var total_2 : f32 = shares_0.x;
    if((imp_6.crush_0.x) > 0.0f)
    {
        _S697 = (imp_6.crush_0.z) < (imp_6.crush_0.y);
    }
    else
    {
        _S697 = false;
    }
    if(_S697)
    {
        _S697 = total_2 > (imp_6.crush_0.x);
    }
    else
    {
        _S697 = false;
    }
    if(_S697)
    {
        var extra_0 : f32 = (total_2 - imp_6.crush_0.x) / shares_0.y;
        imp_6.crush_0[i32(3)] = imp_6.crush_0[i32(3)] + extra_0;
        imp_6.crush_0[i32(2)] = imp_6.crush_0[i32(2)] + imp_6.crush_0.x * extra_0;
        var _S699 : f32 = imp_6.crush_0.x * extra_0;
        var _S700 : f32 = imp_6.ledger_0[i32(2)];
        var _S701 : f32 = imp_6.ledger_0[i32(3)];
        comp_add1_0(&(_S700), &(_S701), _S699);
        imp_6.ledger_0[i32(2)] = _S700;
        imp_6.ledger_0[i32(3)] = _S701;
        var _S702 : f32 = imp_6.crush_0.x * extra_0;
        var _S703 : f32 = imp_6.ledger_0[i32(0)];
        var _S704 : f32 = imp_6.ledger_0[i32(1)];
        comp_add1_0(&(_S703), &(_S704), _S702);
        imp_6.ledger_0[i32(0)] = _S703;
        imp_6.ledger_0[i32(1)] = _S704;
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
    var _S705 : u32 = table_0 + u32(4) * g_4;
    w_11.island_2 = index_0[_S705];
    w_11.begin_1 = index_0[_S705 + u32(1)];
    w_11.end_0 = index_0[_S705 + u32(2)];
    w_11.first_1 = index_0[_S705 + u32(3)];
    return w_11;
}

fn contact_stopped_1( isl_19 : ptr<function, Island_std430_0>) -> bool
{
    var _S706 : vec4<u32> = islands_0[params_0.halt_index_0].info_0;
    var _S707 : bool;
    if((((islands_0[params_0.halt_index_0].info_0.z) & (u32(1)))) != u32(0))
    {
        _S707 = true;
    }
    else
    {
        var _S708 : u32 = _S706.y;
        if(_S708 != u32(0))
        {
            _S707 = _S708 <= ((*isl_19).info_0.w);
        }
        else
        {
            _S707 = false;
        }
    }
    if(_S707)
    {
        _S707 = true;
    }
    else
    {
        var _S709 : u32 = islands_0[params_0.halt_index_0].range_0.x;
        if(_S709 != u32(0))
        {
            _S707 = _S709 <= ((*isl_19).info_0.w + u32(1));
        }
        else
        {
            _S707 = false;
        }
    }
    return _S707;
}

fn contact_stopped_2( isl_20 : ptr<function, Island_std430_0>) -> bool
{
    var _S710 : vec4<u32> = islands_0[params_0.halt_index_0].info_0;
    var _S711 : bool;
    if((((islands_0[params_0.halt_index_0].info_0.z) & (u32(1)))) != u32(0))
    {
        _S711 = true;
    }
    else
    {
        var _S712 : u32 = _S710.y;
        if(_S712 != u32(0))
        {
            _S711 = _S712 <= ((*isl_20).info_0.w);
        }
        else
        {
            _S711 = false;
        }
    }
    if(_S711)
    {
        _S711 = true;
    }
    else
    {
        var _S713 : u32 = islands_0[params_0.halt_index_0].range_0.x;
        if(_S713 != u32(0))
        {
            _S711 = _S713 <= ((*isl_20).info_0.w + u32(1));
        }
        else
        {
            _S711 = false;
        }
    }
    return _S711;
}

fn wide_runs_0( isl_21 : ptr<function, Island_std430_0>) -> bool
{
    var _S714 : vec4<u32> = (*isl_21).info_0;
    var _S715 : bool;
    if(((((*isl_21).info_0.z) & (u32(1)))) != u32(0))
    {
        _S715 = true;
    }
    else
    {
        _S715 = (_S714.y) == u32(0);
    }
    if(_S715)
    {
        return false;
    }
    if((((_S714.x) & (u32(4)))) == u32(0))
    {
        _S715 = true;
    }
    else
    {
        var _S716 : bool = contact_stopped_1(&((*isl_21)));
        _S715 = !_S716;
    }
    return _S715;
}

fn wide_runs_1( isl_22 : ptr<function, Island_std430_0>) -> bool
{
    var _S717 : vec4<u32> = (*isl_22).info_0;
    var _S718 : bool;
    if(((((*isl_22).info_0.z) & (u32(1)))) != u32(0))
    {
        _S718 = true;
    }
    else
    {
        _S718 = (_S717.y) == u32(0);
    }
    if(_S718)
    {
        return false;
    }
    if((((_S717.x) & (u32(4)))) == u32(0))
    {
        _S718 = true;
    }
    else
    {
        var _S719 : bool = contact_stopped_2(&((*isl_22)));
        _S718 = !_S719;
    }
    return _S718;
}

var<workgroup> g_wide_run_0 : u32;

fn wide_enter_0( tid_16 : u32,  isl_23 : ptr<function, Island_std430_0>) -> bool
{
    seen_clear_0(tid_16);
    if(tid_16 == u32(0))
    {
        var _S720 : bool = wide_runs_1(&((*isl_23)));
        var _S721 : i32;
        if(_S720)
        {
            _S721 = i32(1);
        }
        else
        {
            _S721 = i32(0);
        }
        g_wide_run_0 = u32(_S721);
    }
    workgroupBarrier();
    return g_wide_run_0 != u32(0);
}

fn wide_step_0( isl_24 : ptr<function, Island_std430_0>) -> u32
{
    return (*isl_24).info_0.w - params_0.step_start_0;
}

fn wide_step_1( isl_25 : ptr<function, Island_std430_0>) -> u32
{
    return (*isl_25).info_0.w - params_0.step_start_0;
}

fn wide_step_2( isl_26 : ptr<function, Island_std430_0>) -> u32
{
    return (*isl_26).info_0.w - params_0.step_start_0;
}

fn wide_step_3( isl_27 : Island_0) -> u32
{
    return isl_27.info_0.w - params_0.step_start_0;
}

fn wide_enter_1( _S722 : u32,  _S723 : u32) -> bool
{
    var _S724 : Island_std430_0 = islands_0[_S723];
    seen_clear_0(_S722);
    if(_S722 == u32(0))
    {
        var _S725 : bool = wide_runs_0(&(_S724));
        var _S726 : i32;
        if(_S725)
        {
            _S726 = i32(1);
        }
        else
        {
            _S726 = i32(0);
        }
        g_wide_run_0 = u32(_S726);
    }
    workgroupBarrier();
    return g_wide_run_0 != u32(0);
}

@compute
@workgroup_size(256, 1, 1)
fn wide_wake(@builtin(workgroup_id) group_1 : vec3<u32>, @builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var tid_17 : u32 = thread_1.x;
    var _S727 : u32 = group_1.x;
    var wg_0 : WideGroup_0 = wide_group_0(params_0.wide_chunk_table_0, _S727);
    if(_S727 != (wg_0.first_1))
    {
        return;
    }
    var _S728 : Island_std430_0 = islands_0[wg_0.island_2];
    var _S729 : vec4<u32> = _S728.info_0;
    var _S730 : u32 = _S728.info_0.x;
    var _S731 : bool;
    if(((_S730 & (u32(16)))) == u32(0))
    {
        _S731 = true;
    }
    else
    {
        var _S732 : bool = wide_enter_1(tid_17, wg_0.island_2);
        _S731 = !_S732;
    }
    if(_S731)
    {
        return;
    }
    var _S733 : Quat_0 = quat_of_0(_S728.rotation_0);
    var _S734 : bool = ((_S730 & (u32(4)))) != u32(0);
    var _S735 : vec3<f32> = vec3<f32>(0.0f);
    var norm_5 : vec3<f32> = _S735;
    var unused_6 : vec3<f32> = _S735;
    var _S736 : vec4<u32> = _S728.range_0;
    var c_38 : u32 = _S728.range_0.x + tid_17;
    loop
    {
        if(c_38 < (_S736.y))
        {
        }
        else
        {
            break;
        }
        var _S737 : u32 = wide_step_2(&(_S728));
        var _S738 : f32 = settled_chunk_load_0(c_38, _S733, _S737, params_0.dt_0, _S734);
        norm_5[i32(0)] = norm_5[i32(0)] + _S738;
        c_38 = c_38 + u32(256);
    }
    group_sum3_3(tid_17, &(norm_5), &(unused_6));
    if(tid_17 != u32(0))
    {
        return;
    }
    var cov_25 : u32 = (u32(1024) | ((seen_bits_0())));
    if((params_0.solve_mode_0) == u32(1))
    {
        _S731 = (abs(norm_5.x - _S728.energy_0.z)) > (_S728.energy_0.w);
    }
    else
    {
        _S731 = false;
    }
    var cov_26 : u32;
    if(_S731)
    {
        islands_0[wg_0.island_2].info_0[i32(0)] = (_S730 & (u32(4294967279)));
        islands_0[wg_0.island_2].info_0[i32(2)] = ((_S729.z) | (u32(4)));
        cov_26 = (cov_25 | (u32(2048)));
    }
    else
    {
        cov_26 = cov_25;
    }
    cov_mark_0(wg_0.island_2, _S728.coverage_0, cov_26);
    return;
}

fn wide_partials_0( tid_18 : u32,  first_2 : u32,  count_5 : u32,  p_17 : u32,  a_24 : ptr<function, vec3<f32>>,  b_49 : ptr<function, vec3<f32>>)
{
    var _S739 : vec4<f32> = vec4<f32>(0.0f);
    var x_22 : vec4<f32> = _S739;
    var y_5 : vec4<f32> = _S739;
    var s_10 : u32 = tid_18;
    loop
    {
        if(s_10 < count_5)
        {
        }
        else
        {
            break;
        }
        var _S740 : u32 = u32(8) * (first_2 + s_10);
        x_22 = x_22 + scratch_0[params_0.wide_base_0 + _S740 + p_17];
        y_5 = y_5 + scratch_0[params_0.wide_base_0 + _S740 + p_17 + u32(1)];
        s_10 = s_10 + u32(256);
    }
    group_sum2_0(tid_18, &(x_22), &(y_5));
    (*a_24) = x_22.xyz;
    (*b_49) = y_5.xyz;
    return;
}

fn wide_partials_1( tid_19 : u32,  first_3 : u32,  count_6 : u32,  p_18 : u32,  a_25 : ptr<function, vec3<f32>>,  b_50 : ptr<function, vec3<f32>>)
{
    var _S741 : vec4<f32> = vec4<f32>(0.0f);
    var x_23 : vec4<f32> = _S741;
    var y_6 : vec4<f32> = _S741;
    var s_11 : u32 = tid_19;
    loop
    {
        if(s_11 < count_6)
        {
        }
        else
        {
            break;
        }
        var _S742 : u32 = u32(8) * (first_3 + s_11);
        x_23 = x_23 + scratch_0[params_0.wide_base_0 + _S742 + p_18];
        y_6 = y_6 + scratch_0[params_0.wide_base_0 + _S742 + p_18 + u32(1)];
        s_11 = s_11 + u32(256);
    }
    group_sum2_1(tid_19, &(x_23), &(y_6));
    (*a_25) = x_23.xyz;
    (*b_50) = y_6.xyz;
    return;
}

fn drift_angular_2( c_39 : u32,  wcom_6 : vec3<f32>,  tr_10 : vec3<f32>,  dv_10 : vec3<f32>,  lu_7 : ptr<function, vec3<f32>>,  lv_7 : ptr<function, vec3<f32>>)
{
    var r_17 : vec3<f32> = chunks_0[c_39].center_0.xyz - wcom_6;
    var _S743 : u32 = u32(4) * c_39;
    var _S744 : vec3<f32> = vec3<f32>(chunks_0[c_39].center_0.w);
    var _S745 : vec4<f32> = chunks_0[c_39].inertia0_1;
    var _S746 : vec4<f32> = chunks_0[c_39].inertia1_1;
    var _S747 : vec4<f32> = chunks_0[c_39].inertia2_1;
    var _S748 : vec3<f32> = vec3<f32>(chunks_0[c_39].scale_0.x);
    (*lu_7) = (*lu_7) + (cross(r_17, state_0[_S743].xyz - tr_10) * _S744 + rows_mul_0(chunks_0[c_39].inertia0_1, chunks_0[c_39].inertia1_1, chunks_0[c_39].inertia2_1, state_0[_S743 + u32(1)].xyz)) * _S748;
    (*lv_7) = (*lv_7) + (cross(r_17, state_0[_S743 + u32(2)].xyz - dv_10) * _S744 + rows_mul_0(_S745, _S746, _S747, state_0[_S743 + u32(3)].xyz)) * _S748;
    return;
}

fn wide_store_0( slot_2 : u32,  p_19 : u32,  a_26 : vec3<f32>,  b_51 : vec3<f32>)
{
    var _S749 : u32 = u32(8) * slot_2;
    scratch_0[params_0.wide_base_0 + _S749 + p_19] = vec4<f32>(a_26, 0.0f);
    scratch_0[params_0.wide_base_0 + _S749 + p_19 + u32(1)] = vec4<f32>(b_51, 0.0f);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn wide_drift(@builtin(workgroup_id) group_2 : vec3<u32>, @builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var tid_20 : u32 = thread_2.x;
    var _S750 : u32 = group_2.x;
    var wg_1 : WideGroup_0 = wide_group_0(params_0.wide_chunk_table_0, _S750);
    var isl_28 : Island_std430_0 = islands_0[wg_1.island_2];
    var _S751 : bool;
    if((((islands_0[wg_1.island_2].info_0.x) & (u32(17)))) != u32(0))
    {
        _S751 = true;
    }
    else
    {
        var _S752 : bool = wide_enter_1(tid_20, wg_1.island_2);
        _S751 = !_S752;
    }
    if(_S751)
    {
        return;
    }
    var tu_7 : vec3<f32>;
    var pv_7 : vec3<f32>;
    wide_partials_1(tid_20, wg_1.first_1, isl_28.wide_groups_0, u32(2), &(tu_7), &(pv_7));
    var _S753 : vec3<f32> = vec3<f32>(isl_28.wcom_0.w);
    var tr_11 : vec3<f32> = tu_7 / _S753;
    var dv_11 : vec3<f32> = pv_7 / _S753;
    var _S754 : vec3<f32> = vec3<f32>(0.0f);
    var lu_8 : vec3<f32> = _S754;
    var lv_8 : vec3<f32> = _S754;
    var c_40 : u32 = wg_1.begin_1 + tid_20;
    if(c_40 < (wg_1.end_0))
    {
        drift_angular_2(c_40, isl_28.wcom_0.xyz, tr_11, dv_11, &(lu_8), &(lv_8));
    }
    group_sum3_3(tid_20, &(lu_8), &(lv_8));
    if(tid_20 == u32(0))
    {
        wide_store_0(_S750, u32(4), lu_8, lv_8);
    }
    return;
}

fn rigid_of_1( isl_29 : ptr<function, Island_std430_0>) -> Rigid_0
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
    var _S755 : vec3<f32> = vec3<f32>(0.0f);
    rg_17.torque_0 = _S755;
    rg_17.a_0 = _S755;
    rg_17.alpha_0 = _S755;
    return rg_17;
}

fn rigid_of_2( isl_30 : ptr<function, Island_std430_0>) -> Rigid_0
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
    var _S756 : vec3<f32> = vec3<f32>(0.0f);
    rg_18.torque_0 = _S756;
    rg_18.a_0 = _S756;
    rg_18.alpha_0 = _S756;
    return rg_18;
}

fn rigid_of_3( isl_31 : ptr<function, Island_std430_0>) -> Rigid_0
{
    var rg_19 : Rigid_0;
    rg_19.rot_0 = quat_of_0((*isl_31).rotation_0);
    rg_19.rot_err_0 = (*isl_31).rotation_err_0;
    rg_19.pos_0 = (*isl_31).position_0.xyz;
    rg_19.pos_err_0 = (*isl_31).position_err_0.xyz;
    rg_19.vel_0 = (*isl_31).velocity_0.xyz;
    rg_19.vel_err_0 = (*isl_31).velocity_err_0.xyz;
    rg_19.w_1 = (*isl_31).angular_velocity_0.xyz;
    rg_19.l_0 = (*isl_31).momentum_0.xyz;
    rg_19.l_err_0 = (*isl_31).momentum_err_0.xyz;
    var _S757 : vec3<f32> = vec3<f32>(0.0f);
    rg_19.torque_0 = _S757;
    rg_19.a_0 = _S757;
    rg_19.alpha_0 = _S757;
    return rg_19;
}

fn rigid_acceleration_2( isl_32 : ptr<function, Island_std430_0>,  rg_20 : ptr<function, Rigid_0>,  f_23 : vec3<f32>,  t_24 : vec3<f32>)
{
    var iw_w_2 : vec3<f32> = world_mul_0((*rg_20).rot_0, (*isl_32).inertia0_0, (*isl_32).inertia1_0, (*isl_32).inertia2_0, (*rg_20).w_1);
    (*rg_20).a_0 = f_23 / vec3<f32>((*isl_32).com_0.w);
    (*rg_20).alpha_0 = world_mul_0((*rg_20).rot_0, (*isl_32).inv0_0, (*isl_32).inv1_0, (*isl_32).inv2_0, t_24 - cross((*rg_20).w_1, iw_w_2));
    (*rg_20).torque_0 = t_24;
    return;
}

fn rigid_acceleration_3( isl_33 : ptr<function, Island_std430_0>,  rg_21 : ptr<function, Rigid_0>,  f_24 : vec3<f32>,  t_25 : vec3<f32>)
{
    var iw_w_3 : vec3<f32> = world_mul_0((*rg_21).rot_0, (*isl_33).inertia0_0, (*isl_33).inertia1_0, (*isl_33).inertia2_0, (*rg_21).w_1);
    (*rg_21).a_0 = f_24 / vec3<f32>((*isl_33).com_0.w);
    (*rg_21).alpha_0 = world_mul_0((*rg_21).rot_0, (*isl_33).inv0_0, (*isl_33).inv1_0, (*isl_33).inv2_0, t_25 - cross((*rg_21).w_1, iw_w_3));
    (*rg_21).torque_0 = t_25;
    return;
}

fn wide_rigid_frame_0( tid_21 : u32,  isl_34 : ptr<function, Island_std430_0>,  wg_2 : WideGroup_0) -> Rigid_0
{
    var _S758 : Rigid_0 = rigid_of_1(&((*isl_34)));
    var rg_22 : Rigid_0 = _S758;
    if(((((*isl_34).info_0.x) & (u32(1)))) == u32(0))
    {
        var f_25 : vec3<f32>;
        var t_26 : vec3<f32>;
        wide_partials_0(tid_21, wg_2.first_1, (*isl_34).wide_groups_0, u32(0), &(f_25), &(t_26));
        rigid_acceleration_2(&((*isl_34)), &(rg_22), f_25, t_26);
    }
    return rg_22;
}

fn wide_rigid_frame_1( tid_22 : u32,  isl_35 : ptr<function, Island_std430_0>,  wg_3 : WideGroup_0) -> Rigid_0
{
    var _S759 : Rigid_0 = rigid_of_3(&((*isl_35)));
    var rg_23 : Rigid_0 = _S759;
    if(((((*isl_35).info_0.x) & (u32(1)))) == u32(0))
    {
        var f_26 : vec3<f32>;
        var t_27 : vec3<f32>;
        wide_partials_1(tid_22, wg_3.first_1, (*isl_35).wide_groups_0, u32(0), &(f_26), &(t_27));
        rigid_acceleration_3(&((*isl_35)), &(rg_23), f_26, t_27);
    }
    return rg_23;
}

fn comp_add_1( sum_9 : ptr<function, vec3<f32>>,  err_8 : ptr<function, vec3<f32>>,  x_24 : vec3<f32>)
{
    var t_28 : vec3<f32> = (*sum_9) + x_24;
    var _S760 : vec3<f32> = abs(x_24);
    (*err_8) = (*err_8) + (select(x_24, (*sum_9), (abs((*sum_9))) >= _S760) - t_28 + select((*sum_9), x_24, (abs((*sum_9))) >= _S760));
    (*sum_9) = t_28;
    return;
}

fn turn_left_1( hi_4 : ptr<function, Quat_0>,  lo_4 : ptr<function, vec4<f32>>,  omega_2 : vec3<f32>,  dt_17 : f32)
{
    var _S761 : f32 = length(omega_2);
    var angle_5 : f32 = _S761 * dt_17;
    if(angle_5 < 1.00000000317107685e-30f)
    {
        return;
    }
    var d_23 : vec4<f32> = turn_minus_one_0(omega_2 / vec3<f32>(_S761), angle_5);
    var dq_2 : Quat_0;
    dq_2.x_0 = d_23.x;
    dq_2.y_0 = d_23.y;
    dq_2.z_0 = d_23.z;
    dq_2.w_0 = d_23.w;
    quat_accumulate_0(&((*hi_4)), &((*lo_4)), quat_vec_0(quat_mul_0(dq_2, (*hi_4))));
    return;
}

fn integrate_rigid_2( isl_36 : ptr<function, Island_std430_0>,  rg_24 : ptr<function, Rigid_0>,  dt_18 : f32)
{
    var _S762 : vec3<f32> = vec3<f32>(dt_18);
    var _S763 : vec3<f32> = (*rg_24).torque_0 * _S762;
    var _S764 : vec3<f32> = (*rg_24).l_0;
    var _S765 : vec3<f32> = (*rg_24).l_err_0;
    comp_add_0(&(_S764), &(_S765), _S763);
    (*rg_24).l_0 = _S764;
    (*rg_24).l_err_0 = _S765;
    var l_6 : vec3<f32> = _S764 + _S765;
    var _S766 : vec3<f32> = (*rg_24).a_0 * _S762;
    var _S767 : vec3<f32> = (*rg_24).vel_0;
    var _S768 : vec3<f32> = (*rg_24).vel_err_0;
    comp_add_0(&(_S767), &(_S768), _S766);
    (*rg_24).vel_0 = _S767;
    (*rg_24).vel_err_0 = _S768;
    var _S769 : vec4<f32> = (*isl_36).inv0_0;
    var _S770 : vec4<f32> = (*isl_36).inv1_0;
    var _S771 : vec4<f32> = (*isl_36).inv2_0;
    var w_mid_2 : vec3<f32> = world_mul_0((*rg_24).rot_0, (*isl_36).inv0_0, (*isl_36).inv1_0, (*isl_36).inv2_0, l_6);
    var delta_2 : vec3<f32> = (_S767 + _S768) * _S762 - turn_difference_0(w_mid_2, dt_18, rotate_0((*rg_24).rot_0, (*isl_36).com_0.xyz));
    var _S772 : vec3<f32> = (*rg_24).pos_0;
    var _S773 : vec3<f32> = (*rg_24).pos_err_0;
    comp_add_0(&(_S772), &(_S773), delta_2);
    (*rg_24).pos_0 = _S772;
    (*rg_24).pos_err_0 = _S773;
    var _S774 : Quat_0 = (*rg_24).rot_0;
    var _S775 : vec4<f32> = (*rg_24).rot_err_0;
    turn_left_0(&(_S774), &(_S775), w_mid_2, dt_18);
    (*rg_24).rot_0 = _S774;
    (*rg_24).rot_err_0 = _S775;
    (*rg_24).w_1 = world_mul_0(_S774, _S769, _S770, _S771, l_6);
    return;
}

fn drift_rigid_2( isl_37 : ptr<function, Island_std430_0>,  rg_25 : ptr<function, Rigid_0>,  tr_12 : vec3<f32>,  phi_9 : vec3<f32>,  dv_12 : vec3<f32>,  dw_8 : vec3<f32>)
{
    var wcom_7 : vec3<f32> = (*isl_37).wcom_0.xyz;
    var rot_4 : Quat_0 = (*rg_25).rot_0;
    var _S776 : vec3<f32> = rotate_0((*rg_25).rot_0, tr_12 - turn_difference_0(phi_9, 1.0f, wcom_7));
    var _S777 : vec3<f32> = (*rg_25).pos_0;
    var _S778 : vec3<f32> = (*rg_25).pos_err_0;
    comp_add_0(&(_S777), &(_S778), _S776);
    (*rg_25).pos_0 = _S777;
    (*rg_25).pos_err_0 = _S778;
    var b_52 : vec3<f32> = inverse_rotate_0((*rg_25).rot_0, (*rg_25).w_1);
    var _S779 : vec4<f32> = (*isl_37).inertia0_0;
    var _S780 : vec4<f32> = (*isl_37).inertia1_0;
    var _S781 : vec4<f32> = (*isl_37).inertia2_0;
    var dl_turn_2 : vec3<f32> = rotate_0((*rg_25).rot_0, cross(phi_9, rows_mul_0((*isl_37).inertia0_0, (*isl_37).inertia1_0, (*isl_37).inertia2_0, b_52)) - rows_mul_0((*isl_37).inertia0_0, (*isl_37).inertia1_0, (*isl_37).inertia2_0, cross(phi_9, b_52)));
    var _S782 : Quat_0 = (*rg_25).rot_0;
    var _S783 : vec4<f32> = (*rg_25).rot_err_0;
    turn_right_0(&(_S782), &(_S783), phi_9);
    (*rg_25).rot_0 = _S782;
    (*rg_25).rot_err_0 = _S783;
    var _S784 : vec3<f32> = rotate_0(rot_4, dv_12 + cross(dw_8, (*isl_37).com_0.xyz - wcom_7));
    var _S785 : vec3<f32> = (*rg_25).vel_0;
    var _S786 : vec3<f32> = (*rg_25).vel_err_0;
    comp_add_0(&(_S785), &(_S786), _S784);
    (*rg_25).vel_0 = _S785;
    (*rg_25).vel_err_0 = _S786;
    var _S787 : vec3<f32> = rotate_0(rot_4, dw_8);
    (*rg_25).w_1 = (*rg_25).w_1 + _S787;
    var _S788 : vec3<f32> = dl_turn_2 + world_mul_0(_S782, _S779, _S780, _S781, _S787);
    var _S789 : vec3<f32> = (*rg_25).l_0;
    var _S790 : vec3<f32> = (*rg_25).l_err_0;
    comp_add_0(&(_S789), &(_S790), _S788);
    (*rg_25).l_0 = _S789;
    (*rg_25).l_err_0 = _S790;
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn wide_rigid(@builtin(workgroup_id) group_3 : vec3<u32>, @builtin(local_invocation_id) thread_3 : vec3<u32>)
{
    var tid_23 : u32 = thread_3.x;
    var _S791 : u32 = group_3.x;
    var wg_4 : WideGroup_0 = wide_group_0(params_0.wide_chunk_table_0, _S791);
    var _S792 : Island_std430_0 = islands_0[wg_4.island_2];
    var _S793 : u32 = _S792.info_0.x;
    var _S794 : bool;
    if(((_S793 & (u32(1)))) != u32(0))
    {
        _S794 = true;
    }
    else
    {
        var _S795 : bool = wide_enter_1(tid_23, wg_4.island_2);
        _S794 = !_S795;
    }
    if(_S794)
    {
        return;
    }
    if(((_S793 & (u32(16)))) != u32(0))
    {
        if(_S791 != (wg_4.first_1))
        {
            return;
        }
        var _S796 : Rigid_0 = wide_rigid_frame_1(tid_23, &(_S792), wg_4);
        var rs_0 : Rigid_0 = _S796;
        if(tid_23 != u32(0))
        {
            _S794 = true;
        }
        else
        {
            _S794 = ((_S793 & (u32(2)))) != u32(0);
        }
        if(_S794)
        {
            return;
        }
        integrate_rigid_2(&(_S792), &(rs_0), params_0.dt_0);
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
    var tu_8 : vec3<f32>;
    var pv_8 : vec3<f32>;
    wide_partials_1(tid_23, wg_4.first_1, _S792.wide_groups_0, u32(2), &(tu_8), &(pv_8));
    var lu_9 : vec3<f32>;
    var lv_9 : vec3<f32>;
    wide_partials_1(tid_23, wg_4.first_1, _S792.wide_groups_0, u32(4), &(lu_9), &(lv_9));
    var _S797 : vec4<f32> = _S792.wcom_0;
    var _S798 : vec3<f32> = vec3<f32>(_S792.wcom_0.w);
    var tr_13 : vec3<f32> = tu_8 / _S798;
    var dv_13 : vec3<f32> = pv_8 / _S798;
    var phi_10 : vec3<f32> = rows_mul_0(_S792.winv0_0, _S792.winv1_0, _S792.winv2_0, lu_9);
    var dw_9 : vec3<f32> = rows_mul_0(_S792.winv0_0, _S792.winv1_0, _S792.winv2_0, lv_9);
    var c_41 : u32 = wg_4.begin_1 + tid_23;
    if(c_41 < (wg_4.end_0))
    {
        drift_apply_0(c_41, _S797.xyz, tr_13, phi_10, dv_13, dw_9);
    }
    if(_S791 != (wg_4.first_1))
    {
        return;
    }
    var _S799 : Rigid_0 = wide_rigid_frame_1(tid_23, &(_S792), wg_4);
    var rg_26 : Rigid_0 = _S799;
    if(tid_23 != u32(0))
    {
        return;
    }
    if(!(((_S793 & (u32(2)))) != u32(0)))
    {
        integrate_rigid_2(&(_S792), &(rg_26), params_0.dt_0);
        drift_rigid_2(&(_S792), &(rg_26), tr_13, phi_10, dv_13, dw_9);
    }
    islands_0[wg_4.island_2].rotation_0 = quat_vec_0(rg_26.rot_0);
    islands_0[wg_4.island_2].rotation_err_0 = rg_26.rot_err_0;
    islands_0[wg_4.island_2].position_0 = vec4<f32>(rg_26.pos_0, 0.0f);
    islands_0[wg_4.island_2].position_err_0 = vec4<f32>(rg_26.pos_err_0, 0.0f);
    islands_0[wg_4.island_2].velocity_0 = vec4<f32>(rg_26.vel_0, 0.0f);
    islands_0[wg_4.island_2].velocity_err_0 = vec4<f32>(rg_26.vel_err_0, 0.0f);
    islands_0[wg_4.island_2].angular_velocity_0 = vec4<f32>(rg_26.w_1, 0.0f);
    islands_0[wg_4.island_2].momentum_0 = vec4<f32>(rg_26.l_0, 0.0f);
    islands_0[wg_4.island_2].momentum_err_0 = vec4<f32>(rg_26.l_err_0, 0.0f);
    return;
}

var<workgroup> g_label_0 : array<u32, i32(4096)>;

var<workgroup> g_changed_0 : u32;

fn connected_3( st_7 : ptr<function, JointState_std430_0>,  has_rebar_4 : bool) -> bool
{
    var _S800 : bool;
    if(((*st_7).damage_0) < 1.0f)
    {
        _S800 = true;
    }
    else
    {
        if(has_rebar_4)
        {
            _S800 = ((*st_7).rebar_broken_0) == 0.0f;
        }
        else
        {
            _S800 = false;
        }
    }
    return _S800;
}

fn connected_4( st_8 : ptr<function, JointState_std430_0>,  has_rebar_5 : bool) -> bool
{
    var _S801 : bool;
    if(((*st_8).damage_0) < 1.0f)
    {
        _S801 = true;
    }
    else
    {
        if(has_rebar_5)
        {
            _S801 = ((*st_8).rebar_broken_0) == 0.0f;
        }
        else
        {
            _S801 = false;
        }
    }
    return _S801;
}

fn island_connected_0( tid_24 : u32,  isl_38 : Island_0) -> bool
{
    var _S802 : u32 = isl_38.range_0.x;
    var n_13 : u32 = isl_38.range_0.y - _S802;
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
    var c_42 : u32 = tid_24;
    loop
    {
        if(c_42 < n_13)
        {
        }
        else
        {
            break;
        }
        g_label_0[c_42] = c_42;
        c_42 = c_42 + u32(256);
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
        if(tid_24 == u32(0))
        {
            g_changed_0 = u32(0);
        }
        workgroupBarrier();
        c_42 = tid_24;
        loop
        {
            if(c_42 < n_13)
            {
            }
            else
            {
                break;
            }
            var gc_1 : u32 = _S802 + c_42;
            var _S803 : u32 = index_0[gc_1];
            var l_7 : u32 = g_label_0[c_42];
            var e_6 : u32 = _S803;
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
                var _S804 : JointState_std430_0 = bond_dyn_0[bond_1].js_0;
                var _S805 : bool = connected_3(&(_S804), (bonds_0[bond_1].law_0.stiff1_0.w) != 0.0f);
                if(!_S805)
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
                l_7 = min(l_7, g_label_0[other_1 - _S802]);
                e_6 = e_6 + u32(1);
            }
            var _S806 : u32 = min(l_7, g_label_0[l_7]);
            if(_S806 < (g_label_0[c_42]))
            {
                g_label_0[c_42] = _S806;
                g_changed_0 = u32(1);
            }
            c_42 = c_42 + u32(256);
        }
        workgroupBarrier();
        var converged_3 : bool = g_changed_0 == u32(0);
        workgroupBarrier();
        if(converged_3)
        {
            converged_2 = converged_3;
            break;
        }
        var _S807 : u32 = pass_1 + u32(1);
        converged_2 = converged_3;
        pass_1 = _S807;
    }
    if(!converged_2)
    {
        return false;
    }
    if(tid_24 == u32(0))
    {
        g_changed_0 = u32(0);
    }
    workgroupBarrier();
    c_42 = tid_24;
    loop
    {
        if(c_42 < n_13)
        {
        }
        else
        {
            break;
        }
        if((g_label_0[c_42]) != u32(0))
        {
            g_changed_0 = u32(1);
        }
        c_42 = c_42 + u32(256);
    }
    workgroupBarrier();
    return g_changed_0 == u32(0);
}

@compute
@workgroup_size(256, 1, 1)
fn wide_end(@builtin(workgroup_id) group_4 : vec3<u32>, @builtin(local_invocation_id) thread_4 : vec3<u32>)
{
    var tid_25 : u32 = thread_4.x;
    var _S808 : u32 = group_4.x;
    var wg_5 : WideGroup_0 = wide_group_0(params_0.wide_chunk_table_0, _S808);
    if(_S808 != (wg_5.first_1))
    {
        return;
    }
    var _S809 : Island_std430_0 = islands_0[wg_5.island_2];
    var isl_39 : Island_0;
    isl_39.range_0 = _S809.range_0;
    isl_39.info_0 = _S809.info_0;
    isl_39.com_0 = _S809.com_0;
    isl_39.inertia0_0 = _S809.inertia0_0;
    isl_39.inertia1_0 = _S809.inertia1_0;
    isl_39.inertia2_0 = _S809.inertia2_0;
    isl_39.inv0_0 = _S809.inv0_0;
    isl_39.inv1_0 = _S809.inv1_0;
    isl_39.inv2_0 = _S809.inv2_0;
    isl_39.wcom_0 = _S809.wcom_0;
    isl_39.winv0_0 = _S809.winv0_0;
    isl_39.winv1_0 = _S809.winv1_0;
    isl_39.winv2_0 = _S809.winv2_0;
    isl_39.rotation_0 = _S809.rotation_0;
    isl_39.position_0 = _S809.position_0;
    isl_39.position_err_0 = _S809.position_err_0;
    isl_39.velocity_0 = _S809.velocity_0;
    isl_39.velocity_err_0 = _S809.velocity_err_0;
    isl_39.angular_velocity_0 = _S809.angular_velocity_0;
    isl_39.done_0 = _S809.done_0;
    isl_39.coverage_0 = _S809.coverage_0;
    isl_39.wide_groups_0 = _S809.wide_groups_0;
    isl_39.done_pad_0 = _S809.done_pad_0;
    isl_39.probes_0 = _S809.probes_0;
    isl_39.energy_0 = _S809.energy_0;
    isl_39.rotation_err_0 = _S809.rotation_err_0;
    isl_39.momentum_0 = _S809.momentum_0;
    isl_39.momentum_err_0 = _S809.momentum_err_0;
    var _S810 : bool = wide_enter_0(tid_25, &(_S809));
    if(!_S810)
    {
        return;
    }
    var work_8 : vec3<f32>;
    var flags_2 : vec3<f32>;
    wide_partials_1(tid_25, wg_5.first_1, isl_39.wide_groups_0, u32(6), &(work_8), &(flags_2));
    var halt_0 : bool = (((isl_39.info_0.z) & (u32(2)))) != u32(0);
    var whole_0 : bool;
    if(halt_0)
    {
        var _S811 : bool = island_connected_0(tid_25, isl_39);
        whole_0 = _S811;
    }
    else
    {
        whole_0 = false;
    }
    if(tid_25 != u32(0))
    {
        return;
    }
    var halt_1 : bool;
    if(whole_0)
    {
        isl_39.info_0[i32(2)] = ((isl_39.info_0[i32(2)]) & (u32(4294967293)));
        isl_39.coverage_0 = ((isl_39.coverage_0) | (u32(2097280)));
        halt_1 = false;
    }
    else
    {
        halt_1 = halt_0;
    }
    var k_26 : u32 = wide_step_3(isl_39);
    if((isl_39.probes_0.y) > (isl_39.probes_0.x))
    {
        record_probes_0(isl_39, rigid_of_0(isl_39), k_26);
    }
    if(halt_1)
    {
        whole_0 = (((isl_39.info_0.x) & (u32(4)))) != u32(0);
    }
    else
    {
        whole_0 = false;
    }
    if(whole_0)
    {
        contact_split_at_0(isl_39.info_0.w + u32(1));
    }
    var _S812 : f32 = work_8.x;
    var _S813 : f32 = isl_39.energy_0[i32(0)];
    var _S814 : f32 = isl_39.energy_0[i32(1)];
    comp_add1_0(&(_S813), &(_S814), _S812);
    isl_39.energy_0[i32(0)] = _S813;
    isl_39.energy_0[i32(1)] = _S814 + work_8.y;
    isl_39.done_0 = isl_39.done_0 + u32(1);
    isl_39.info_0[i32(1)] = isl_39.info_0[i32(1)] - u32(1);
    isl_39.info_0[i32(3)] = isl_39.info_0[i32(3)] + u32(1);
    if(halt_1)
    {
        isl_39.info_0[i32(2)] = ((((isl_39.info_0.z) & (u32(4294967293)))) | (u32(1)));
    }
    var _S815 : u32;
    if(halt_1)
    {
        _S815 = u32(128);
    }
    else
    {
        _S815 = u32(0);
    }
    var _S816 : u32 = (u32(256) | (_S815));
    if((isl_39.probes_0.y) > (isl_39.probes_0.x))
    {
        _S815 = u32(262144);
    }
    else
    {
        _S815 = u32(0);
    }
    var _S817 : u32 = (_S816 | (_S815));
    if((((isl_39.info_0.x) & (u32(4)))) != u32(0))
    {
        _S815 = u32(524288);
    }
    else
    {
        _S815 = u32(0);
    }
    var _S818 : u32 = (_S817 | (_S815));
    if((flags_2.x) > 0.0f)
    {
        _S815 = u32(32768);
    }
    else
    {
        _S815 = u32(0);
    }
    var _S819 : u32 = (_S818 | (_S815));
    if((flags_2.y) > 0.0f)
    {
        _S815 = u32(65536);
    }
    else
    {
        _S815 = u32(0);
    }
    var _S820 : u32 = (_S819 | (_S815));
    if((flags_2.z) > 0.0f)
    {
        _S815 = u32(131072);
    }
    else
    {
        _S815 = u32(0);
    }
    isl_39.coverage_0 = ((isl_39.coverage_0) | (((_S820 | (_S815)))));
    islands_0[wg_5.island_2].range_0 = isl_39.range_0;
    islands_0[wg_5.island_2].info_0 = isl_39.info_0;
    islands_0[wg_5.island_2].com_0 = isl_39.com_0;
    islands_0[wg_5.island_2].inertia0_0 = isl_39.inertia0_0;
    islands_0[wg_5.island_2].inertia1_0 = isl_39.inertia1_0;
    islands_0[wg_5.island_2].inertia2_0 = isl_39.inertia2_0;
    islands_0[wg_5.island_2].inv0_0 = isl_39.inv0_0;
    islands_0[wg_5.island_2].inv1_0 = isl_39.inv1_0;
    islands_0[wg_5.island_2].inv2_0 = isl_39.inv2_0;
    islands_0[wg_5.island_2].wcom_0 = isl_39.wcom_0;
    islands_0[wg_5.island_2].winv0_0 = isl_39.winv0_0;
    islands_0[wg_5.island_2].winv1_0 = isl_39.winv1_0;
    islands_0[wg_5.island_2].winv2_0 = isl_39.winv2_0;
    islands_0[wg_5.island_2].rotation_0 = isl_39.rotation_0;
    islands_0[wg_5.island_2].position_0 = isl_39.position_0;
    islands_0[wg_5.island_2].position_err_0 = isl_39.position_err_0;
    islands_0[wg_5.island_2].velocity_0 = isl_39.velocity_0;
    islands_0[wg_5.island_2].velocity_err_0 = isl_39.velocity_err_0;
    islands_0[wg_5.island_2].angular_velocity_0 = isl_39.angular_velocity_0;
    islands_0[wg_5.island_2].done_0 = isl_39.done_0;
    islands_0[wg_5.island_2].coverage_0 = isl_39.coverage_0;
    islands_0[wg_5.island_2].wide_groups_0 = isl_39.wide_groups_0;
    islands_0[wg_5.island_2].done_pad_0 = isl_39.done_pad_0;
    islands_0[wg_5.island_2].probes_0 = isl_39.probes_0;
    islands_0[wg_5.island_2].energy_0 = isl_39.energy_0;
    islands_0[wg_5.island_2].rotation_err_0 = isl_39.rotation_err_0;
    islands_0[wg_5.island_2].momentum_0 = isl_39.momentum_0;
    islands_0[wg_5.island_2].momentum_err_0 = isl_39.momentum_err_0;
    return;
}

fn sv_0( c_43 : u32,  slot_3 : u32) -> u32
{
    return params_0.statics_base_0 + u32(23) * c_43 + slot_3;
}

fn project_load_slot_0( tid_26 : u32,  isl_40 : ptr<function, Island_std430_0>,  slot_4 : u32)
{
    var _S821 : u32;
    var _S822 : vec3<f32> = vec3<f32>(0.0f);
    var net_f_0 : vec3<f32> = _S822;
    var net_m_0 : vec3<f32> = _S822;
    var _S823 : vec4<u32> = (*isl_40).range_0;
    var _S824 : u32 = (*isl_40).range_0.x + tid_26;
    var c_44 : u32 = _S824;
    loop
    {
        var _S825 : u32 = _S823.y;
        _S821 = _S825;
        if(c_44 < _S825)
        {
        }
        else
        {
            break;
        }
        var fi_2 : vec3<f32> = scratch_0[sv_0(c_44, slot_4)].xyz;
        net_f_0 = net_f_0 + fi_2;
        net_m_0 = net_m_0 + (cross(chunks_0[c_44].center_0.xyz - (*isl_40).com_0.xyz, fi_2) + scratch_0[sv_0(c_44, slot_4 + u32(1))].xyz);
        c_44 = c_44 + u32(256);
    }
    group_sum3_3(tid_26, &(net_f_0), &(net_m_0));
    var _S826 : vec4<f32> = (*isl_40).com_0;
    var _S827 : vec3<f32> = net_f_0 / vec3<f32>((*isl_40).com_0.w);
    var _S828 : vec3<f32> = rows_mul_0((*isl_40).inv0_0, (*isl_40).inv1_0, (*isl_40).inv2_0, net_m_0);
    c_44 = _S824;
    loop
    {
        if(c_44 < _S821)
        {
        }
        else
        {
            break;
        }
        var _S829 : u32 = sv_0(c_44, slot_4);
        scratch_0[_S829] = vec4<f32>(scratch_0[_S829].xyz - (_S827 + cross(_S828, chunks_0[c_44].center_0.xyz - _S826.xyz)) * vec3<f32>(chunks_0[c_44].center_0.w), 0.0f);
        var _S830 : u32 = sv_0(c_44, slot_4 + u32(1));
        scratch_0[_S830] = vec4<f32>(scratch_0[_S830].xyz - rows_mul_0(chunks_0[c_44].inertia0_1, chunks_0[c_44].inertia1_1, chunks_0[c_44].inertia2_1, _S828), 0.0f);
        c_44 = c_44 + u32(256);
    }
    device_group_barrier_0();
    return;
}

fn island_dot_0( tid_27 : u32,  c0_0 : u32,  c1_0 : u32,  sa_0 : u32,  sb_0 : u32) -> f32
{
    var _S831 : vec4<f32> = vec4<f32>(0.0f);
    var acc_0 : vec4<f32> = _S831;
    var unused_7 : vec4<f32> = _S831;
    var c_45 : u32 = c0_0 + tid_27;
    loop
    {
        if(c_45 < c1_0)
        {
        }
        else
        {
            break;
        }
        acc_0[i32(0)] = acc_0[i32(0)] + (dot(scratch_0[sv_0(c_45, sa_0)].xyz, scratch_0[sv_0(c_45, sb_0)].xyz) + dot(scratch_0[sv_0(c_45, sa_0 + u32(1))].xyz, scratch_0[sv_0(c_45, sb_0 + u32(1))].xyz));
        c_45 = c_45 + u32(256);
    }
    group_sum2_2(tid_27, &(acc_0), &(unused_7));
    return acc_0.x;
}

fn stress_measures_1( b_53 : ptr<function, JointBond_std430_0>,  q_lin_3 : vec3<f32>,  q_ang_3 : vec3<f32>) -> Measures_0
{
    var area_3 : f32 = (*b_53).geom0_0.x;
    var _S832 : f32 = q_lin_3.z;
    var axial_1 : f32 = fdiv_0(_S832, area_3);
    var bending_1 : f32 = fdiv_0(abs(q_ang_3.x), (*b_53).geom1_0.x) + fdiv_0(abs(q_ang_3.y), (*b_53).geom1_0.y);
    var _S833 : f32 = q_lin_3.x;
    var _S834 : f32 = q_lin_3.y;
    var shear_2 : f32 = fdiv_0(fsqrt_0(_S833 * _S833 + _S834 * _S834), area_3) + fdiv_0(abs(q_ang_3.z), (*b_53).geom0_0.w);
    var m_6 : Measures_0;
    m_6.tension_0 = axial_1 + bending_1;
    m_6.shear_0 = shear_2;
    var _S835 : f32 = - axial_1;
    m_6.normal_compression_0 = max(_S835, 0.0f);
    m_6.compression_0 = _S835 + bending_1;
    m_6.compressive_force_0 = max(- _S832, 0.0f);
    return m_6;
}

fn dif_factor_1( mat_10 : ptr<function, JointMaterial_std140_0>,  strain_rate_2 : f32) -> f32
{
    var r_18 : f32 = abs(strain_rate_2);
    var _S836 : vec4<f32> = (*mat_10).dif_0;
    var ref_1 : f32 = (*mat_10).dif_0.x;
    if(r_18 <= ref_1)
    {
        return 1.0f;
    }
    var _S837 : f32 = _S836.z;
    var f_27 : f32;
    if(r_18 <= _S837)
    {
        f_27 = fpow_0(fdiv_0(r_18, ref_1), _S836.y);
    }
    else
    {
        f_27 = fpow_0(fdiv_0(_S837, ref_1), _S836.y) * fpow_0(fdiv_0(r_18, _S837), _S836.w);
    }
    return clamp(f_27, 1.0f, (*mat_10).misc_0.x);
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
    var _S838 : f32 = min((*mat_12).strength_0.z * multiplier_2 + (*mat_12).strength_0.w * m_7.normal_compression_0, (*mat_12).energy_1.x * multiplier_2);
    var idx_1 : vec4<f32>;
    idx_1[i32(0)] = max(fdiv_0(m_7.tension_0, (*mat_12).strength_0.x * multiplier_2), 0.0f);
    var _S839 : f32;
    if(_S838 > 0.0f)
    {
        _S839 = fdiv_0(m_7.shear_0, _S838);
    }
    else
    {
        _S839 = infinity_0();
    }
    idx_1[i32(1)] = _S839;
    idx_1[i32(2)] = max(fdiv_0(m_7.compression_0, fc_5), 0.0f);
    var _S840 : f32 = (*b_54).stiff1_0.y;
    if(_S840 > 0.0f)
    {
        _S839 = fdiv_0(m_7.compressive_force_0, _S840);
    }
    else
    {
        _S839 = 0.0f;
    }
    idx_1[i32(3)] = _S839;
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
        var _S841 : f32 = p_20.y;
        var _S842 : f32 = p_20.z;
        var _S843 : f32 = p_20.w;
        nc_sum_5 = p_20.x;
        m1_4 = _S841;
        m2_4 = _S842;
        energy_7 = _S843;
    }
    else
    {
        var ki_2 : f32 = kn_4 * (1.0f - crush_5) / 36.0f;
        var _S844 : f32 = d_ang_4.x;
        var _S845 : f32 = d_ang_4.y;
        var spread_2 : f32 = abs(_S844) * 0.4166666567325592f * w1_4 + abs(_S845) * 0.4166666567325592f * w0_4;
        var _S846 : f32 = d_lin_5.z;
        var slack_2 : f32 = 9.99999997475242708e-07f * (abs(_S846) + spread_2);
        if((_S846 - spread_2) > slack_2)
        {
            nc_sum_5 = 0.0f;
            m1_4 = 0.0f;
            m2_4 = 0.0f;
            energy_7 = 0.0f;
        }
        else
        {
            if((_S846 + spread_2) < (- slack_2))
            {
                var i1_1 : f32 = 2.91666650772094727f * w0_4 * w0_4;
                var i2_1 : f32 = 2.91666650772094727f * w1_4 * w1_4;
                var _S847 : f32 = ki_2 * _S844 * i2_1;
                var _S848 : f32 = ki_2 * _S845 * i1_1;
                var _S849 : f32 = 0.5f * ki_2 * (36.0f * _S846 * _S846 + _S844 * _S844 * i2_1 + _S845 * _S845 * i1_1);
                nc_sum_5 = ki_2 * 36.0f * _S846;
                m1_4 = _S847;
                m2_4 = _S848;
                energy_7 = _S849;
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
                    var _S850 : f32 = SPRING_AT_0[i_11] * w0_4;
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
                        var di_2 : f32 = _S846 + _S844 * s2_1 - _S845 * _S850;
                        if(di_2 < 0.0f)
                        {
                            var f_28 : f32 = ki_2 * di_2;
                            var m1_6 : f32 = m1_4 + f_28 * s2_1;
                            var m2_6 : f32 = m2_4 - f_28 * _S850;
                            var energy_9 : f32 = energy_7 + 0.5f * ki_2 * di_2 * di_2;
                            nc_sum_5 = nc_sum_5 + f_28;
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
    var c_46 : Contact_0;
    var _S851 : vec3<f32> = vec3<f32>(0.0f);
    c_46.q_lin_1 = _S851;
    c_46.q_ang_1 = _S851;
    c_46.energy_5 = 0.0f;
    c_46.diss_1 = 0.0f;
    c_46.plastic_0 = plastic_5;
    if(((((*mat_14).kind_flags_0.y) & (u32(2)))) == u32(0))
    {
        return c_46;
    }
    var ks_3 : f32 = (*b_56).stiff0_0.y;
    var kt_3 : f32 = (*b_56).stiff1_0.x;
    var _S852 : vec4<f32> = contact_normal_1(&((*mat_14)), &((*b_56)), crush_6, d_lin_6, d_ang_5);
    var nc_sum_7 : f32 = _S852.x;
    var m1_7 : f32 = _S852.y;
    var m2_7 : f32 = _S852.z;
    var energy_10 : f32 = _S852.w;
    var nc_2 : f32 = - nc_sum_7;
    var p_21 : vec3<f32> = plastic_5;
    c_46.q_lin_1 = vec3<f32>(0.0f, 0.0f, nc_sum_7);
    c_46.q_ang_1 = vec3<f32>(m1_7, m2_7, 0.0f);
    var slide_cap_2 : f32 = (*mat_14).strength_0.w * nc_2;
    var _S853 : f32 = ks_3 * (d_lin_6.x - plastic_5.x);
    var _S854 : f32 = ks_3 * (d_lin_6.y - plastic_5.y);
    var tn_2 : f32 = fsqrt_0(_S853 * _S853 + _S854 * _S854);
    var _S855 : bool;
    if(tn_2 > slide_cap_2)
    {
        _S855 = tn_2 > 0.0f;
    }
    else
    {
        _S855 = false;
    }
    var diss_6 : f32;
    if(_S855)
    {
        var _S856 : f32 = fdiv_0(_S853, tn_2);
        var _S857 : f32 = fdiv_0(_S854, tn_2);
        var dslip_2 : f32 = fdiv_0(tn_2 - slide_cap_2, ks_3);
        p_21[i32(0)] = p_21[i32(0)] + _S856 * dslip_2;
        p_21[i32(1)] = p_21[i32(1)] + _S857 * dslip_2;
        c_46.q_lin_1[i32(0)] = _S856 * slide_cap_2;
        c_46.q_lin_1[i32(1)] = _S857 * slide_cap_2;
        diss_6 = slide_cap_2 * dslip_2;
    }
    else
    {
        c_46.q_lin_1[i32(0)] = _S853;
        c_46.q_lin_1[i32(1)] = _S854;
        diss_6 = 0.0f;
    }
    var tq_1 : vec2<f32> = return_map_0(kt_3, d_ang_5.z, p_21.z, slide_cap_2 * (*b_56).geom1_0.z);
    var _S858 : f32 = tq_1.x;
    var _S859 : f32 = tq_1.y;
    var diss_7 : f32 = diss_6 + abs(_S858) * abs(_S859);
    p_21[i32(2)] = p_21[i32(2)] + _S859;
    c_46.q_ang_1[i32(2)] = _S858;
    c_46.energy_5 = energy_10 + 0.5f * (fdiv_0(sq_0(c_46.q_lin_1.x), ks_3) + fdiv_0(sq_0(c_46.q_lin_1.y), ks_3) + fdiv_0(sq_0(_S858), kt_3));
    c_46.diss_1 = diss_7;
    c_46.plastic_0 = p_21;
    return c_46;
}

fn contact_offsets_1( mat_15 : ptr<function, JointMaterial_std140_0>,  b_57 : ptr<function, JointBond_std430_0>,  crush_7 : f32,  plastic_6 : vec3<f32>,  d_lin_7 : vec3<f32>,  d_ang_6 : vec3<f32>) -> vec3<f32>
{
    var _S860 : u32 = (*mat_15).kind_flags_0.y;
    if(((_S860 & (u32(2)))) == u32(0))
    {
        return plastic_6;
    }
    var kn_5 : f32 = (*b_57).stiff0_0.x;
    var ks_4 : f32 = (*b_57).stiff0_0.y;
    var kt_4 : f32 = (*b_57).stiff1_0.x;
    var w0_5 : f32 = (*b_57).geom0_0.y;
    var w1_5 : f32 = (*b_57).geom0_0.z;
    var nc_sum_8 : f32;
    if(((_S860 & (u32(4)))) != u32(0))
    {
        var _S861 : vec4<f32> = no_tension_patch_0(kn_5 * (1.0f - crush_7), w0_5, w1_5, d_lin_7.z, d_ang_6.x, d_ang_6.y);
        nc_sum_8 = _S861.x;
    }
    else
    {
        var ki_3 : f32 = kn_5 * (1.0f - crush_7) / 36.0f;
        var _S862 : f32 = d_ang_6.x;
        var _S863 : f32 = d_ang_6.y;
        var spread_3 : f32 = abs(_S862) * 0.4166666567325592f * w1_5 + abs(_S863) * 0.4166666567325592f * w0_5;
        var _S864 : f32 = d_lin_7.z;
        var slack_3 : f32 = 9.99999997475242708e-07f * (abs(_S864) + spread_3);
        if((_S864 - spread_3) > slack_3)
        {
            nc_sum_8 = 0.0f;
        }
        else
        {
            if((_S864 + spread_3) < (- slack_3))
            {
                nc_sum_8 = ki_3 * 36.0f * _S864;
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
                    var _S865 : f32 = SPRING_AT_0[i_12] * w0_5;
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
                        var di_3 : f32 = _S864 + _S862 * (SPRING_AT_0[j_10] * w1_5) - _S863 * _S865;
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
    var _S866 : f32 = ks_4 * (d_lin_7.x - plastic_6.x);
    var _S867 : f32 = ks_4 * (d_lin_7.y - plastic_6.y);
    var tn_3 : f32 = fsqrt_0(_S866 * _S866 + _S867 * _S867);
    var _S868 : bool;
    if(tn_3 > slide_cap_3)
    {
        _S868 = tn_3 > 0.0f;
    }
    else
    {
        _S868 = false;
    }
    if(_S868)
    {
        var _S869 : f32 = fdiv_0(_S867, tn_3);
        var dslip_3 : f32 = fdiv_0(tn_3 - slide_cap_3, ks_4);
        p_22[i32(0)] = p_22[i32(0)] + fdiv_0(_S866, tn_3) * dslip_3;
        p_22[i32(1)] = p_22[i32(1)] + _S869 * dslip_3;
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
    var _S870 : f32 = (*mat_16).misc_0.y;
    return fdiv_0((_S870 + 1.0f) * fpow_0(s_12, _S870), (*mat_16).misc_0.z);
}

fn joint_evaluate_1( mat_17 : ptr<function, JointMaterial_std140_0>,  b_58 : ptr<function, JointBond_std430_0>,  state_3 : ptr<function, JointState_std430_0>,  d_lin_8 : vec3<f32>,  d_ang_7 : vec3<f32>,  dt_19 : f32,  fracture_3 : bool) -> JointResponse_0
{
    var kn_6 : f32 = (*b_58).stiff0_0.x;
    var ks_5 : f32 = (*b_58).stiff0_0.y;
    var kb1_1 : f32 = (*b_58).stiff0_0.z;
    var kb2_1 : f32 = (*b_58).stiff0_0.w;
    var _S871 : vec4<f32> = (*b_58).stiff1_0;
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
    var _S872 : bool = connected_4(&((*state_3)), has_rebar_6);
    var qe_lin_1 : vec3<f32> = d_lin_8 * vec3<f32>(ks_5, ks_5, kn_6);
    var qe_ang_1 : vec3<f32> = d_ang_7 * vec3<f32>(kb1_1, kb2_1, kt_5);
    var _S873 : Measures_0 = stress_measures_1(&((*b_58)), qe_lin_1, qe_ang_1);
    var _S874 : f32 = max(max(_S873.tension_0, _S873.shear_0), _S873.compression_0);
    var _S875 : bool = dt_19 > 0.0f;
    var dif_2 : f32;
    if(_S875)
    {
        var raw_1 : f32 = fdiv_0(max(fdiv_0(_S874 - st_9.governing_stress_0, dt_19), 0.0f), (*mat_17).misc_0.w);
        var tau_3 : f32 = _S871.z;
        if(((flags_3 & (u32(16)))) != u32(0))
        {
            dif_2 = - expm1_accurate_0(- fdiv_0(dt_19, tau_3));
        }
        else
        {
            dif_2 = min(fdiv_0(dt_19, tau_3), 1.0f);
        }
        st_9.strain_rate_0 = st_9.strain_rate_0 + (raw_1 - st_9.strain_rate_0) * dif_2;
        st_9.governing_stress_0 = _S874;
    }
    if(((flags_3 & (u32(32)))) != u32(0))
    {
        var _S876 : f32 = dif_factor_1(&((*mat_17)), st_9.strain_rate_0);
        dif_2 = _S876;
    }
    else
    {
        dif_2 = 1.0f;
    }
    var weibull_1 : f32 = (*b_58).geom1_0.w;
    var _S877 : f32 = fatigue_factor_2(&((*mat_17)), st_9.life_0);
    var multiplier_3 : f32 = weibull_1 * dif_2 * _S877;
    var _S878 : vec4<f32> = failure_indices_1(&((*mat_17)), &((*b_58)), _S873, multiplier_3);
    var _S879 : f32 = _S878.x;
    var _S880 : f32 = _S878.y;
    st_9.utilization_0 = max(max(_S879, _S880), max(_S878.z, _S878.w));
    var _S881 : f32 = d_lin_8.x;
    var _S882 : f32 = d_lin_8.y;
    var _S883 : f32 = d_ang_7.z;
    var _S884 : f32 = ks_5 * (sq_0(_S881) + sq_0(_S882)) + kb1_1 * sq_0(d_ang_7.x) + kb2_1 * sq_0(d_ang_7.y) + kt_5 * sq_0(_S883);
    var _S885 : f32 = d_lin_8.z;
    var _S886 : bool = _S885 > 0.0f;
    if(_S886)
    {
        dif_2 = kn_6 * sq_0(_S885);
    }
    else
    {
        dif_2 = 0.0f;
    }
    var psi_ts_1 : f32 = 0.5f * (_S884 + dif_2);
    var psi_c_1 : f32;
    if(_S885 < 0.0f)
    {
        psi_c_1 = 0.5f * kn_6 * sq_0(_S885);
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
        var _S887 : vec4<f32> = contact_normal_1(&((*mat_17)), &((*b_58)), (*state_3).crush_1, d_lin_8, d_ang_7);
        patch_1 = _S887.w + 0.5f * (ks_5 * (sq_0(_S881 - (*state_3).plastic_x_0) + sq_0(_S882 - (*state_3).plastic_y_0)) + kt_5 * sq_0(_S883 - (*state_3).plastic_t_0));
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
        rebar_1 = 0.5f * ((*b_58).rebar0_0.x * sq_0(_S885 - (*state_3).rebar_plastic_0) + (*b_58).rebar0_0.y * (sq_0(_S881 - (*state_3).rebar_slip0_0) + sq_0(_S882 - (*state_3).rebar_slip1_0)));
    }
    else
    {
        rebar_1 = 0.0f;
    }
    var _S888 : f32 = (*state_3).damage_0;
    var _S889 : f32 = 1.0f - (*state_3).damage_0;
    var _S890 : f32 = (*state_3).crush_1;
    var frozen_1 : f32 = _S889 * (psi_ts_1 + (1.0f - (*state_3).crush_1) * psi_c_1) + (*state_3).damage_0 * patch_1 + rebar_1;
    var diss_contact_1 : f32;
    var psi_contact_1 : f32;
    var intact_normal_1 : f32;
    var dissipated_11 : f32;
    var qc_lin_1 : vec3<f32>;
    if(fracture_3)
    {
        var _S891 : bool = _S879 >= _S880;
        if(_S891)
        {
            diss_contact_1 = _S879;
        }
        else
        {
            diss_contact_1 = _S880;
        }
        var mode_ts_1 : u32;
        if(_S891)
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
            var _S892 : f32 = inc_2.x;
            if(_S892 > (st_9.damage_0))
            {
                var _S893 : Contact_0 = contact_part_1(&((*mat_17)), &((*b_58)), st_9.crush_1, plastic_7, d_lin_8, d_ang_7);
                var _S894 : f32 = max(inc_2.y - max(_S893.energy_5 - (1.0f - st_9.crush_1) * psi_c_1, 0.0f) * (_S892 - st_9.damage_0), 0.0f);
                st_9.damage_0 = _S892;
                st_9.mode_0 = mode_ts_1;
                dissipated_11 = _S894;
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
        if(_S888 > 0.0f)
        {
            var _S895 : Contact_0 = contact_part_1(&((*mat_17)), &((*b_58)), _S890, vec3<f32>((*state_3).plastic_x_0, (*state_3).plastic_y_0, (*state_3).plastic_t_0), d_lin_8, d_ang_7);
            qc_lin_1 = qe_ang_1 * vec3<f32>(_S889) + _S895.q_ang_1 * vec3<f32>(_S888);
        }
        else
        {
            qc_lin_1 = qe_ang_1;
        }
        var _S896 : Measures_0 = stress_measures_1(&((*b_58)), vec3<f32>(0.0f, 0.0f, min(qe_lin_1.z, 0.0f)), qc_lin_1);
        var _S897 : vec4<f32> = failure_indices_1(&((*mat_17)), &((*b_58)), _S896, multiplier_3);
        var _S898 : f32 = _S897.z;
        var _S899 : f32 = _S897.w;
        var _S900 : bool = _S898 >= _S899;
        if(_S900)
        {
            psi_contact_1 = _S898;
        }
        else
        {
            psi_contact_1 = _S899;
        }
        if(_S900)
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
            var _S901 : f32 = inc_3.x;
            if(_S901 > (st_9.crush_1))
            {
                var dissipated_12 : f32 = dissipated_11 + inc_3.y;
                st_9.crush_1 = _S901;
                st_9.mode_0 = mode_c_1;
                if(_S901 >= 1.0f)
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
    var _S902 : vec3<f32> = vec3<f32>(0.0f);
    var qc_ang_1 : vec3<f32>;
    if((st_9.damage_0) == 0.0f)
    {
        if(((flags_3 & (u32(8)))) == u32(0))
        {
            var _S903 : vec3<f32> = contact_offsets_1(&((*mat_17)), &((*b_58)), st_9.crush_1, plastic_7, d_lin_8, d_ang_7);
            st_9.plastic_x_0 = _S903.x;
            st_9.plastic_y_0 = _S903.y;
            st_9.plastic_t_0 = _S903.z;
        }
        diss_contact_1 = 0.0f;
        qc_lin_1 = _S902;
        qc_ang_1 = _S902;
        psi_contact_1 = 0.0f;
    }
    else
    {
        var _S904 : Contact_0 = contact_part_1(&((*mat_17)), &((*b_58)), st_9.crush_1, plastic_7, d_lin_8, d_ang_7);
        st_9.plastic_x_0 = _S904.plastic_0.x;
        st_9.plastic_y_0 = _S904.plastic_0.y;
        st_9.plastic_t_0 = _S904.plastic_0.z;
        diss_contact_1 = _S904.diss_1;
        qc_lin_1 = _S904.q_lin_1;
        qc_ang_1 = _S904.q_ang_1;
        psi_contact_1 = _S904.energy_5;
    }
    var dissipated_14 : f32 = dissipated_11 + dmg_1 * diss_contact_1;
    if(_S886)
    {
        intact_normal_1 = kn_6 * _S885;
    }
    else
    {
        intact_normal_1 = (1.0f - st_9.crush_1) * kn_6 * _S885;
    }
    var _S905 : f32 = 1.0f - dmg_1;
    var force_lin_4 : vec3<f32> = vec3<f32>(_S905 * qe_lin_1.x + dmg_1 * qc_lin_1.x, _S905 * qe_lin_1.y + dmg_1 * qc_lin_1.y, _S905 * intact_normal_1 + dmg_1 * qc_lin_1.z);
    var force_ang_3 : vec3<f32> = qe_ang_1 * vec3<f32>(_S905) + qc_ang_1 * vec3<f32>(dmg_1);
    var stored_10 : f32 = _S905 * (psi_ts_1 + (1.0f - st_9.crush_1) * psi_c_1) + dmg_1 * psi_contact_1;
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
        var nr_1 : vec2<f32> = return_map_0(k_axial_1, _S885, st_9.rebar_plastic_0, yield_force_1);
        var v1_1 : vec2<f32> = return_map_0(k_dowel_1, _S881, st_9.rebar_slip0_0, dowel_capacity_1);
        var v2_1 : vec2<f32> = return_map_0(k_dowel_1, _S882, st_9.rebar_slip1_0, dowel_capacity_1);
        var _S906 : f32 = nr_1.y;
        var _S907 : f32 = v1_1.y;
        var _S908 : f32 = v2_1.y;
        var work_9 : f32 = yield_force_1 * abs(_S906) + dowel_capacity_1 * (abs(_S907) + abs(_S908));
        st_9.rebar_plastic_0 = st_9.rebar_plastic_0 + _S906;
        st_9.rebar_slip0_0 = st_9.rebar_slip0_0 + _S907;
        st_9.rebar_slip1_0 = st_9.rebar_slip1_0 + _S908;
        st_9.rebar_work_0 = st_9.rebar_work_0 + work_9;
        var dissipated_15 : f32 = dissipated_14 + work_9;
        var _S909 : f32 = nr_1.x;
        var _S910 : f32 = v1_1.x;
        var _S911 : f32 = v2_1.x;
        var elastic_1 : f32 = 0.5f * (fdiv_0(sq_0(_S909), k_axial_1) + fdiv_0(sq_0(_S910) + sq_0(_S911), k_dowel_1));
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
            force_lin_5 = force_lin_4 + vec3<f32>(_S910, _S911, _S909);
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
        changed_1 = _S875;
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
        var _S912 : Measures_0 = stress_measures_1(&((*b_58)), force_lin_5, force_ang_3);
        var _S913 : vec4<f32> = failure_indices_1(&((*mat_17)), &((*b_58)), _S912, weibull_1);
        var _S914 : f32 = life_rate_1(&((*mat_17)), max(max(_S913.x, _S913.y), _S913.z));
        st_9.life_0 = max(st_9.life_0 - _S914 * dt_19, 0.0f);
    }
    if((st_9.damage_0) != _S888)
    {
        changed_1 = true;
    }
    else
    {
        changed_1 = (st_9.crush_1) != _S890;
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
    var _S915 : f32 = max(min(dissipated_11, released_1), 0.0f);
    var overshoot_2 : f32 = released_1 - _S915;
    st_9.dissipated_0 = st_9.dissipated_0 + _S915;
    var resp_1 : JointResponse_0;
    resp_1.force_lin_1 = force_lin_5;
    resp_1.force_ang_1 = force_ang_3;
    resp_1.state_1 = st_9;
    resp_1.dissipated_2 = _S915;
    resp_1.overshoot_0 = overshoot_2;
    resp_1.stored_2 = stored_11;
    if(_S872)
    {
        changed_1 = !connected_1(st_9, has_rebar_6);
    }
    else
    {
        changed_1 = false;
    }
    resp_1.disconnected_0 = changed_1;
    resp_1.fatigue_before_0 = _S877;
    resp_1.measures_0 = _S873;
    return resp_1;
}

fn joint_evaluate_2( mat_18 : ptr<function, JointMaterial_std140_0>,  b_59 : ptr<function, JointBond_std430_0>,  state_4 : JointState_0,  d_lin_9 : vec3<f32>,  d_ang_8 : vec3<f32>,  dt_20 : f32,  fracture_4 : bool) -> JointResponse_0
{
    var kn_7 : f32 = (*b_59).stiff0_0.x;
    var ks_6 : f32 = (*b_59).stiff0_0.y;
    var kb1_2 : f32 = (*b_59).stiff0_0.z;
    var kb2_2 : f32 = (*b_59).stiff0_0.w;
    var _S916 : vec4<f32> = (*b_59).stiff1_0;
    var kt_6 : f32 = (*b_59).stiff1_0.x;
    var has_rebar_7 : bool = ((*b_59).stiff1_0.w) != 0.0f;
    var kind_6 : u32 = (*mat_18).kind_flags_0.x;
    var flags_4 : u32 = (*mat_18).kind_flags_0.y;
    var softening_2 : bool = ((flags_4 & (u32(1)))) != u32(0);
    var st_10 : JointState_0 = state_4;
    var was_connected_0 : bool = connected_1(state_4, has_rebar_7);
    var qe_lin_2 : vec3<f32> = d_lin_9 * vec3<f32>(ks_6, ks_6, kn_7);
    var qe_ang_2 : vec3<f32> = d_ang_8 * vec3<f32>(kb1_2, kb2_2, kt_6);
    var _S917 : Measures_0 = stress_measures_1(&((*b_59)), qe_lin_2, qe_ang_2);
    var _S918 : f32 = max(max(_S917.tension_0, _S917.shear_0), _S917.compression_0);
    var _S919 : bool = dt_20 > 0.0f;
    var dif_3 : f32;
    if(_S919)
    {
        var raw_2 : f32 = fdiv_0(max(fdiv_0(_S918 - st_10.governing_stress_0, dt_20), 0.0f), (*mat_18).misc_0.w);
        var tau_4 : f32 = _S916.z;
        if(((flags_4 & (u32(16)))) != u32(0))
        {
            dif_3 = - expm1_accurate_0(- fdiv_0(dt_20, tau_4));
        }
        else
        {
            dif_3 = min(fdiv_0(dt_20, tau_4), 1.0f);
        }
        st_10.strain_rate_0 = st_10.strain_rate_0 + (raw_2 - st_10.strain_rate_0) * dif_3;
        st_10.governing_stress_0 = _S918;
    }
    if(((flags_4 & (u32(32)))) != u32(0))
    {
        var _S920 : f32 = dif_factor_1(&((*mat_18)), st_10.strain_rate_0);
        dif_3 = _S920;
    }
    else
    {
        dif_3 = 1.0f;
    }
    var weibull_2 : f32 = (*b_59).geom1_0.w;
    var _S921 : f32 = fatigue_factor_2(&((*mat_18)), st_10.life_0);
    var multiplier_4 : f32 = weibull_2 * dif_3 * _S921;
    var _S922 : vec4<f32> = failure_indices_1(&((*mat_18)), &((*b_59)), _S917, multiplier_4);
    var _S923 : f32 = _S922.x;
    var _S924 : f32 = _S922.y;
    st_10.utilization_0 = max(max(_S923, _S924), max(_S922.z, _S922.w));
    var _S925 : f32 = d_lin_9.x;
    var _S926 : f32 = d_lin_9.y;
    var _S927 : f32 = d_ang_8.z;
    var _S928 : f32 = ks_6 * (sq_0(_S925) + sq_0(_S926)) + kb1_2 * sq_0(d_ang_8.x) + kb2_2 * sq_0(d_ang_8.y) + kt_6 * sq_0(_S927);
    var _S929 : f32 = d_lin_9.z;
    var _S930 : bool = _S929 > 0.0f;
    if(_S930)
    {
        dif_3 = kn_7 * sq_0(_S929);
    }
    else
    {
        dif_3 = 0.0f;
    }
    var psi_ts_2 : f32 = 0.5f * (_S928 + dif_3);
    var psi_c_2 : f32;
    if(_S929 < 0.0f)
    {
        psi_c_2 = 0.5f * kn_7 * sq_0(_S929);
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
        var _S931 : vec4<f32> = contact_normal_1(&((*mat_18)), &((*b_59)), state_4.crush_1, d_lin_9, d_ang_8);
        patch_2 = _S931.w + 0.5f * (ks_6 * (sq_0(_S925 - state_4.plastic_x_0) + sq_0(_S926 - state_4.plastic_y_0)) + kt_6 * sq_0(_S927 - state_4.plastic_t_0));
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
        rebar_2 = 0.5f * ((*b_59).rebar0_0.x * sq_0(_S929 - state_4.rebar_plastic_0) + (*b_59).rebar0_0.y * (sq_0(_S925 - state_4.rebar_slip0_0) + sq_0(_S926 - state_4.rebar_slip1_0)));
    }
    else
    {
        rebar_2 = 0.0f;
    }
    var _S932 : f32 = 1.0f - state_4.damage_0;
    var frozen_2 : f32 = _S932 * (psi_ts_2 + (1.0f - state_4.crush_1) * psi_c_2) + state_4.damage_0 * patch_2 + rebar_2;
    var diss_contact_2 : f32;
    var psi_contact_2 : f32;
    var intact_normal_2 : f32;
    var dissipated_17 : f32;
    var qc_lin_2 : vec3<f32>;
    if(fracture_4)
    {
        var _S933 : bool = _S923 >= _S924;
        if(_S933)
        {
            diss_contact_2 = _S923;
        }
        else
        {
            diss_contact_2 = _S924;
        }
        var mode_ts_2 : u32;
        if(_S933)
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
            var _S934 : f32 = inc_4.x;
            if(_S934 > (st_10.damage_0))
            {
                var _S935 : Contact_0 = contact_part_1(&((*mat_18)), &((*b_59)), st_10.crush_1, plastic_8, d_lin_9, d_ang_8);
                var _S936 : f32 = max(inc_4.y - max(_S935.energy_5 - (1.0f - st_10.crush_1) * psi_c_2, 0.0f) * (_S934 - st_10.damage_0), 0.0f);
                st_10.damage_0 = _S934;
                st_10.mode_0 = mode_ts_2;
                dissipated_17 = _S936;
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
            var _S937 : Contact_0 = contact_part_1(&((*mat_18)), &((*b_59)), state_4.crush_1, vec3<f32>(state_4.plastic_x_0, state_4.plastic_y_0, state_4.plastic_t_0), d_lin_9, d_ang_8);
            qc_lin_2 = qe_ang_2 * vec3<f32>(_S932) + _S937.q_ang_1 * vec3<f32>(state_4.damage_0);
        }
        else
        {
            qc_lin_2 = qe_ang_2;
        }
        var _S938 : Measures_0 = stress_measures_1(&((*b_59)), vec3<f32>(0.0f, 0.0f, min(qe_lin_2.z, 0.0f)), qc_lin_2);
        var _S939 : vec4<f32> = failure_indices_1(&((*mat_18)), &((*b_59)), _S938, multiplier_4);
        var _S940 : f32 = _S939.z;
        var _S941 : f32 = _S939.w;
        var _S942 : bool = _S940 >= _S941;
        if(_S942)
        {
            psi_contact_2 = _S940;
        }
        else
        {
            psi_contact_2 = _S941;
        }
        if(_S942)
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
            var _S943 : f32 = inc_5.x;
            if(_S943 > (st_10.crush_1))
            {
                var dissipated_18 : f32 = dissipated_17 + inc_5.y;
                st_10.crush_1 = _S943;
                st_10.mode_0 = mode_c_2;
                if(_S943 >= 1.0f)
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
    var _S944 : vec3<f32> = vec3<f32>(0.0f);
    var qc_ang_2 : vec3<f32>;
    if((st_10.damage_0) == 0.0f)
    {
        if(((flags_4 & (u32(8)))) == u32(0))
        {
            var _S945 : vec3<f32> = contact_offsets_1(&((*mat_18)), &((*b_59)), st_10.crush_1, plastic_8, d_lin_9, d_ang_8);
            st_10.plastic_x_0 = _S945.x;
            st_10.plastic_y_0 = _S945.y;
            st_10.plastic_t_0 = _S945.z;
        }
        diss_contact_2 = 0.0f;
        qc_lin_2 = _S944;
        qc_ang_2 = _S944;
        psi_contact_2 = 0.0f;
    }
    else
    {
        var _S946 : Contact_0 = contact_part_1(&((*mat_18)), &((*b_59)), st_10.crush_1, plastic_8, d_lin_9, d_ang_8);
        st_10.plastic_x_0 = _S946.plastic_0.x;
        st_10.plastic_y_0 = _S946.plastic_0.y;
        st_10.plastic_t_0 = _S946.plastic_0.z;
        diss_contact_2 = _S946.diss_1;
        qc_lin_2 = _S946.q_lin_1;
        qc_ang_2 = _S946.q_ang_1;
        psi_contact_2 = _S946.energy_5;
    }
    var dissipated_20 : f32 = dissipated_17 + dmg_2 * diss_contact_2;
    if(_S930)
    {
        intact_normal_2 = kn_7 * _S929;
    }
    else
    {
        intact_normal_2 = (1.0f - st_10.crush_1) * kn_7 * _S929;
    }
    var _S947 : f32 = 1.0f - dmg_2;
    var force_lin_6 : vec3<f32> = vec3<f32>(_S947 * qe_lin_2.x + dmg_2 * qc_lin_2.x, _S947 * qe_lin_2.y + dmg_2 * qc_lin_2.y, _S947 * intact_normal_2 + dmg_2 * qc_lin_2.z);
    var force_ang_4 : vec3<f32> = qe_ang_2 * vec3<f32>(_S947) + qc_ang_2 * vec3<f32>(dmg_2);
    var stored_13 : f32 = _S947 * (psi_ts_2 + (1.0f - st_10.crush_1) * psi_c_2) + dmg_2 * psi_contact_2;
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
        var nr_2 : vec2<f32> = return_map_0(k_axial_2, _S929, st_10.rebar_plastic_0, yield_force_2);
        var v1_2 : vec2<f32> = return_map_0(k_dowel_2, _S925, st_10.rebar_slip0_0, dowel_capacity_2);
        var v2_2 : vec2<f32> = return_map_0(k_dowel_2, _S926, st_10.rebar_slip1_0, dowel_capacity_2);
        var _S948 : f32 = nr_2.y;
        var _S949 : f32 = v1_2.y;
        var _S950 : f32 = v2_2.y;
        var work_10 : f32 = yield_force_2 * abs(_S948) + dowel_capacity_2 * (abs(_S949) + abs(_S950));
        st_10.rebar_plastic_0 = st_10.rebar_plastic_0 + _S948;
        st_10.rebar_slip0_0 = st_10.rebar_slip0_0 + _S949;
        st_10.rebar_slip1_0 = st_10.rebar_slip1_0 + _S950;
        st_10.rebar_work_0 = st_10.rebar_work_0 + work_10;
        var dissipated_21 : f32 = dissipated_20 + work_10;
        var _S951 : f32 = nr_2.x;
        var _S952 : f32 = v1_2.x;
        var _S953 : f32 = v2_2.x;
        var elastic_2 : f32 = 0.5f * (fdiv_0(sq_0(_S951), k_axial_2) + fdiv_0(sq_0(_S952) + sq_0(_S953), k_dowel_2));
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
            force_lin_7 = force_lin_6 + vec3<f32>(_S952, _S953, _S951);
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
        changed_2 = _S919;
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
        var _S954 : Measures_0 = stress_measures_1(&((*b_59)), force_lin_7, force_ang_4);
        var _S955 : vec4<f32> = failure_indices_1(&((*mat_18)), &((*b_59)), _S954, weibull_2);
        var _S956 : f32 = life_rate_1(&((*mat_18)), max(max(_S955.x, _S955.y), _S955.z));
        st_10.life_0 = max(st_10.life_0 - _S956 * dt_20, 0.0f);
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
    var _S957 : f32 = max(min(dissipated_17, released_2), 0.0f);
    var overshoot_3 : f32 = released_2 - _S957;
    st_10.dissipated_0 = st_10.dissipated_0 + _S957;
    var resp_2 : JointResponse_0;
    resp_2.force_lin_1 = force_lin_7;
    resp_2.force_ang_1 = force_ang_4;
    resp_2.state_1 = st_10;
    resp_2.dissipated_2 = _S957;
    resp_2.overshoot_0 = overshoot_3;
    resp_2.stored_2 = stored_14;
    if(was_connected_0)
    {
        changed_2 = !connected_1(st_10, has_rebar_7);
    }
    else
    {
        changed_2 = false;
    }
    resp_2.disconnected_0 = changed_2;
    resp_2.fatigue_before_0 = _S921;
    resp_2.measures_0 = _S917;
    return resp_2;
}

fn static_kinematics_0( _S958 : u32,  _S959 : ptr<function, vec3<f32>>,  _S960 : ptr<function, vec3<f32>>)
{
    var ca_1 : u32 = bonds_0[_S958].law_0.ids_0.y;
    var cb_5 : u32 = bonds_0[_S958].law_0.ids_0.z;
    var _S961 : u32 = u32(4) * cb_5;
    var _S962 : u32 = u32(4) * ca_1;
    var _S963 : u32 = _S961 + u32(1);
    var _S964 : u32 = _S962 + u32(1);
    var _S965 : u32 = sv_0(cb_5, u32(22));
    var _S966 : u32 = sv_0(ca_1, u32(22));
    var dth_0 : vec3<f32> = state_0[_S963].xyz - state_0[_S964].xyz + (scratch_0[_S965].xyz - scratch_0[_S966].xyz);
    (*_S959) = to_local_0(_S958, state_0[_S961].xyz - state_0[_S962].xyz + (scratch_0[sv_0(cb_5, u32(21))].xyz - scratch_0[sv_0(ca_1, u32(21))].xyz) + (cross(state_0[_S963].xyz + scratch_0[_S965].xyz, bonds_0[_S958].rb_0.xyz) - cross(state_0[_S964].xyz + scratch_0[_S966].xyz, bonds_0[_S958].ra_0.xyz)));
    (*_S960) = to_local_0(_S958, dth_0);
    return;
}

fn static_response_0( i_13 : u32) -> JointResponse_0
{
    var d_lin_10 : vec3<f32>;
    var d_ang_9 : vec3<f32>;
    static_kinematics_0(i_13, &(d_lin_10), &(d_ang_9));
    var _S967 : JointBond_std430_0 = bonds_0[i_13].law_0;
    var _S968 : JointMaterial_std140_0 = materials_0.m_0.data_0[_S967.ids_0.x];
    var _S969 : JointState_std430_0 = bond_dyn_0[i_13].js_0;
    var _S970 : JointResponse_0 = joint_evaluate_1(&(_S968), &(_S967), &(_S969), d_lin_10, d_ang_9, 0.0f, false);
    return _S970;
}

fn gather_loads_0( c_47 : u32,  fi_3 : ptr<function, vec3<f32>>,  mi_6 : ptr<function, vec3<f32>>)
{
    var _S971 : vec3<f32> = vec3<f32>(0.0f);
    (*fi_3) = _S971;
    (*mi_6) = _S971;
    var e_7 : u32 = index_0[c_47];
    loop
    {
        if(e_7 < index_0[c_47 + u32(1)])
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
            var _S972 : u32 = u32(3) * bond_2;
            (*fi_3) = (*fi_3) + scratch_0[_S972].xyz;
            (*mi_6) = (*mi_6) + scratch_0[_S972 + u32(1)].xyz;
        }
        else
        {
            var _S973 : u32 = u32(3) * bond_2;
            (*fi_3) = (*fi_3) - scratch_0[_S973].xyz;
            (*mi_6) = (*mi_6) + scratch_0[_S973 + u32(2)].xyz;
        }
        e_7 = e_7 + u32(1);
    }
    return;
}

fn bond_load_magnitude2_0( c_48 : u32) -> f32
{
    var e_8 : u32 = index_0[c_48];
    var m_8 : f32 = 0.0f;
    loop
    {
        if(e_8 < index_0[c_48 + u32(1)])
        {
        }
        else
        {
            break;
        }
        var entry_7 : u32 = index_0[e_8];
        var _S974 : u32 = u32(3) * ((entry_7 >> (u32(1))));
        var f_29 : vec3<f32> = scratch_0[_S974].xyz;
        var t_29 : vec3<f32>;
        if(((entry_7 & (u32(1)))) == u32(0))
        {
            t_29 = scratch_0[_S974 + u32(1)].xyz;
        }
        else
        {
            t_29 = scratch_0[_S974 + u32(2)].xyz;
        }
        var m_9 : f32 = m_8 + (dot(f_29, f_29) + dot(t_29, t_29));
        e_8 = e_8 + u32(1);
        m_8 = m_9;
    }
    return m_8;
}

fn fixed_mask_0( c_49 : u32) -> u32
{
    var support_2 : u32 = chunks_0[c_49].info_1.x;
    var _S975 : u32;
    if(support_2 == u32(1))
    {
        _S975 = u32(63);
    }
    else
    {
        if(support_2 == u32(2))
        {
            _S975 = u32(7);
        }
        else
        {
            _S975 = u32(0);
        }
    }
    return _S975;
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
    var _S976 : f32 = (*st_11).damage_0;
    var compressed_1 : bool = (d_lin_11.z) < 0.0f;
    var contact_6 : f32;
    if(compressed_1)
    {
        contact_6 = _S976;
    }
    else
    {
        contact_6 = 0.0f;
    }
    var _S977 : f32 = 1.0f - _S976;
    var _S978 : f32 = max(_S977 + contact_6, 9.99999997475242708e-07f);
    var normal_10 : f32;
    if(compressed_1)
    {
        normal_10 = max(1.0f - (*st_11).crush_1, 9.99999997475242708e-07f);
    }
    else
    {
        normal_10 = max(_S977, 9.99999997475242708e-07f);
    }
    (*f_lin_2) = vec3<f32>(_S978, _S978, normal_10);
    (*f_ang_2) = vec3<f32>(_S978);
    var _S979 : bool;
    if(((*b_60).stiff1_0.w) != 0.0f)
    {
        _S979 = ((*st_11).rebar_broken_0) == 0.0f;
    }
    else
    {
        _S979 = false;
    }
    if(_S979)
    {
        var _S980 : vec4<f32> = (*b_60).rebar0_0;
        var _S981 : vec4<f32> = (*b_60).stiff0_0;
        (*f_lin_2)[i32(2)] = (*f_lin_2)[i32(2)] + fdiv_0((*b_60).rebar0_0.x, (*b_60).stiff0_0.x);
        var _S982 : f32 = fdiv_0(_S980.y, _S981.y);
        (*f_lin_2)[i32(0)] = (*f_lin_2)[i32(0)] + _S982;
        (*f_lin_2)[i32(1)] = (*f_lin_2)[i32(1)] + _S982;
    }
    return;
}

fn statics_bond_slot_0( i_14 : u32) -> u32
{
    return params_0.statics_base_0 + u32(23) * params_0.chunk_count_0 + u32(2) * i_14;
}

fn store_inverse_0( c_50 : u32,  a_27 : array<f32, i32(36)>)
{
    var j_11 : u32;
    var sum_10 : f32;
    var l_8 : array<f32, i32(36)>;
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
        l_8[k_27] = 0.0f;
        k_27 = k_27 + u32(1);
    }
    var spd_0 : bool = true;
    var i_15 : u32 = u32(0);
    loop
    {
        var _S983 : bool;
        if(i_15 < u32(6))
        {
            _S983 = spd_0;
        }
        else
        {
            _S983 = false;
        }
        if(_S983)
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
            var _S984 : u32 = i_15 * u32(6);
            var _S985 : u32 = _S984 + j_11;
            k_27 = u32(0);
            sum_10 = a_27[_S985];
            loop
            {
                if(k_27 < j_11)
                {
                }
                else
                {
                    break;
                }
                var sum_11 : f32 = sum_10 - l_8[_S984 + k_27] * l_8[j_11 * u32(6) + k_27];
                k_27 = k_27 + u32(1);
                sum_10 = sum_11;
            }
            if(i_15 == j_11)
            {
                if(sum_10 <= 0.0f)
                {
                    spd_0 = false;
                    break;
                }
                l_8[_S984 + i_15] = sqrt(sum_10);
            }
            else
            {
                l_8[_S985] = sum_10 / l_8[j_11 * u32(6) + j_11];
            }
            j_11 = j_11 + u32(1);
        }
        i_15 = i_15 + u32(1);
    }
    var inv_0 : array<f32, i32(36)>;
    if(!spd_0)
    {
        k_27 = u32(0);
        loop
        {
            if(k_27 < u32(36))
            {
            }
            else
            {
                break;
            }
            inv_0[k_27] = 0.0f;
            k_27 = k_27 + u32(1);
        }
        k_27 = u32(0);
        loop
        {
            if(k_27 < u32(6))
            {
            }
            else
            {
                break;
            }
            var _S986 : u32 = k_27 * u32(6) + k_27;
            if((a_27[_S986]) > 0.0f)
            {
                sum_10 = 1.0f / a_27[_S986];
            }
            else
            {
                sum_10 = 0.0f;
            }
            inv_0[_S986] = sum_10;
            k_27 = k_27 + u32(1);
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
                k_27 = u32(0);
                var s_13 : f32 = sum_10;
                loop
                {
                    if(k_27 < i_15)
                    {
                    }
                    else
                    {
                        break;
                    }
                    var s_14 : f32 = s_13 - l_8[i_15 * u32(6) + k_27] * y_7[k_27];
                    k_27 = k_27 + u32(1);
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
                k_27 = i_16 + u32(1);
                sum_10 = y_7[i_16];
                loop
                {
                    if(k_27 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var s_15 : f32 = sum_10 - l_8[k_27 * u32(6) + i_16] * x_26[k_27];
                    k_27 = k_27 + u32(1);
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
        var _S987 : u32 = u32(4) * j_11;
        scratch_0[sv_0(c_50, u32(12) + j_11)] = vec4<f32>(inv_0[_S987], inv_0[_S987 + u32(1)], inv_0[_S987 + u32(2)], inv_0[_S987 + u32(3)]);
        j_11 = j_11 + u32(1);
    }
    return;
}

fn assemble_block_0( c_51 : u32)
{
    var p_23 : u32;
    var r_19 : u32;
    var a_28 : array<f32, i32(36)>;
    var k_28 : u32 = u32(0);
    loop
    {
        if(k_28 < u32(36))
        {
        }
        else
        {
            break;
        }
        a_28[k_28] = 0.0f;
        k_28 = k_28 + u32(1);
    }
    var e_9 : u32 = index_0[c_51];
    loop
    {
        if(e_9 < index_0[c_51 + u32(1)])
        {
        }
        else
        {
            break;
        }
        var entry_8 : u32 = index_0[e_9];
        var i_18 : u32 = (entry_8 >> (u32(1)));
        var _S988 : bool = ((entry_8 & (u32(1)))) != u32(0);
        var _S989 : u32 = statics_bond_slot_0(i_18);
        var _S990 : vec4<f32> = scratch_0[_S989];
        var _S991 : vec4<f32> = scratch_0[_S989 + u32(1)];
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
            var _S992 : u32 = p_23 % u32(3);
            var t_30 : vec3<f32>;
            if(_S992 == u32(0))
            {
                t_30 = bonds_0[i_18].t1_0.xyz;
            }
            else
            {
                if(_S992 == u32(1))
                {
                    t_30 = bonds_0[i_18].t2_0.xyz;
                }
                else
                {
                    t_30 = bonds_0[i_18].normal_0.xyz;
                }
            }
            var _S993 : bool = p_23 < u32(3);
            var row_u_0 : vec3<f32>;
            var row_t_0 : vec3<f32>;
            if(_S993)
            {
                if(_S988)
                {
                    row_u_0 = t_30;
                }
                else
                {
                    row_u_0 = (vec3<f32>(0) - t_30);
                }
                if(_S988)
                {
                    row_t_0 = cross(bonds_0[i_18].rb_0.xyz, t_30);
                }
                else
                {
                    row_t_0 = (vec3<f32>(0) - cross(bonds_0[i_18].ra_0.xyz, t_30));
                }
            }
            else
            {
                var _S994 : vec3<f32> = vec3<f32>(0.0f);
                if(_S988)
                {
                    row_u_0 = t_30;
                }
                else
                {
                    row_u_0 = (vec3<f32>(0) - t_30);
                }
                var _S995 : vec3<f32> = row_u_0;
                row_u_0 = _S994;
                row_t_0 = _S995;
            }
            var kp_0 : f32;
            if(_S993)
            {
                kp_0 = _S990[p_23];
            }
            else
            {
                kp_0 = _S991[p_23 - u32(3)];
            }
            if(kp_0 == 0.0f)
            {
                p_23 = p_23 + u32(1);
                continue;
            }
            var _S996 : array<f32, i32(6)> = array<f32, i32(6)>( row_u_0.x, row_u_0.y, row_u_0.z, row_t_0.x, row_t_0.y, row_t_0.z );
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
                    a_28[r_19 * u32(6) + q_14] = a_28[r_19 * u32(6) + q_14] + kp_0 * _S996[r_19] * _S996[q_14];
                    q_14 = q_14 + u32(1);
                }
                r_19 = r_19 + u32(1);
            }
            p_23 = p_23 + u32(1);
        }
        e_9 = e_9 + u32(1);
    }
    var _S997 : u32 = fixed_mask_0(c_51);
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
        if(((_S997 & (((u32(1) << (p_23)))))) != u32(0))
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
    store_inverse_0(c_51, a_28);
    return;
}

fn block_get_0( c_52 : u32,  i_19 : u32,  j_12 : u32) -> f32
{
    var k_29 : u32 = i_19 * u32(6) + j_12;
    return scratch_0[sv_0(c_52, u32(12) + k_29 / u32(4))][k_29 % u32(4)];
}

fn precondition_0( c_53 : u32)
{
    var _S998 : array<f32, i32(6)> = array<f32, i32(6)>( scratch_0[sv_0(c_53, u32(4))].x, scratch_0[sv_0(c_53, u32(4))].y, scratch_0[sv_0(c_53, u32(4))].z, scratch_0[sv_0(c_53, u32(5))].x, scratch_0[sv_0(c_53, u32(5))].y, scratch_0[sv_0(c_53, u32(5))].z );
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
            var s_17 : f32 = s_16 + block_get_0(c_53, i_20, j_13) * _S998[j_13];
            j_13 = j_13 + u32(1);
            s_16 = s_17;
        }
        z_1[i_20] = s_16;
        i_20 = i_20 + u32(1);
    }
    scratch_0[sv_0(c_53, u32(6))] = vec4<f32>(z_1[i32(0)], z_1[i32(1)], z_1[i32(2)], 0.0f);
    scratch_0[sv_0(c_53, u32(7))] = vec4<f32>(z_1[i32(3)], z_1[i32(4)], z_1[i32(5)], 0.0f);
    return;
}

fn project_displacement_slot_0( tid_28 : u32,  isl_41 : ptr<function, Island_std430_0>,  slot_5 : u32)
{
    var _S999 : u32;
    var _S1000 : vec3<f32> = vec3<f32>(0.0f);
    var p_24 : vec3<f32> = _S1000;
    var l_9 : vec3<f32> = _S1000;
    var _S1001 : vec4<u32> = (*isl_41).range_0;
    var _S1002 : u32 = (*isl_41).range_0.x + tid_28;
    var c_54 : u32 = _S1002;
    loop
    {
        var _S1003 : u32 = _S1001.y;
        _S999 = _S1003;
        if(c_54 < _S1003)
        {
        }
        else
        {
            break;
        }
        var u_6 : vec3<f32> = scratch_0[sv_0(c_54, slot_5)].xyz;
        var th_8 : vec3<f32> = scratch_0[sv_0(c_54, slot_5 + u32(1))].xyz;
        var r_20 : vec3<f32> = chunks_0[c_54].center_0.xyz - (*isl_41).com_0.xyz;
        var _S1004 : vec3<f32> = vec3<f32>(chunks_0[c_54].center_0.w);
        p_24 = p_24 + u_6 * _S1004;
        l_9 = l_9 + (cross(r_20, u_6) * _S1004 + rows_mul_0(chunks_0[c_54].inertia0_1, chunks_0[c_54].inertia1_1, chunks_0[c_54].inertia2_1, th_8));
        c_54 = c_54 + u32(256);
    }
    group_sum3_3(tid_28, &(p_24), &(l_9));
    var _S1005 : vec4<f32> = (*isl_41).com_0;
    var _S1006 : vec3<f32> = p_24 / vec3<f32>((*isl_41).com_0.w);
    var _S1007 : vec3<f32> = rows_mul_0((*isl_41).inv0_0, (*isl_41).inv1_0, (*isl_41).inv2_0, l_9);
    c_54 = _S1002;
    loop
    {
        if(c_54 < _S999)
        {
        }
        else
        {
            break;
        }
        var _S1008 : u32 = sv_0(c_54, slot_5);
        scratch_0[_S1008] = vec4<f32>(scratch_0[_S1008].xyz - _S1006 - cross(_S1007, chunks_0[c_54].center_0.xyz - _S1005.xyz), 0.0f);
        var _S1009 : u32 = sv_0(c_54, slot_5 + u32(1));
        scratch_0[_S1009] = vec4<f32>(scratch_0[_S1009].xyz - _S1007, 0.0f);
        c_54 = c_54 + u32(256);
    }
    device_group_barrier_0();
    return;
}

fn statics_result_slot_0( island_3 : u32) -> u32
{
    return params_0.statics_base_0 + u32(23) * params_0.chunk_count_0 + u32(2) * params_0.statics_bonds_0 + island_3;
}

fn write_bond_loads_0( _S1010 : u32,  _S1011 : u32,  _S1012 : vec3<f32>,  _S1013 : vec3<f32>,  _S1014 : f32)
{
    var _S1015 : vec3<f32> = to_body_0(_S1011, _S1012);
    var _S1016 : vec3<f32> = to_body_0(_S1011, _S1013);
    var _S1017 : u32 = u32(3) * _S1010;
    scratch_0[_S1017] = vec4<f32>(_S1015, _S1014);
    scratch_0[_S1017 + u32(1)] = vec4<f32>(_S1016 + cross(bonds_0[_S1011].ra_0.xyz, _S1015), 0.0f);
    scratch_0[_S1017 + u32(2)] = vec4<f32>((vec3<f32>(0) - _S1016) + cross(bonds_0[_S1011].rb_0.xyz, (vec3<f32>(0) - _S1015)), 0.0f);
    return;
}

fn static_kinematics_1( _S1018 : u32,  _S1019 : ptr<function, vec3<f32>>,  _S1020 : ptr<function, vec3<f32>>)
{
    var ca_2 : u32 = bonds_0[_S1018].law_0.ids_0.y;
    var cb_6 : u32 = bonds_0[_S1018].law_0.ids_0.z;
    var _S1021 : u32 = u32(4) * cb_6;
    var _S1022 : u32 = u32(4) * ca_2;
    var _S1023 : u32 = _S1021 + u32(1);
    var _S1024 : u32 = _S1022 + u32(1);
    var _S1025 : u32 = sv_0(cb_6, u32(22));
    var _S1026 : u32 = sv_0(ca_2, u32(22));
    var dth_1 : vec3<f32> = state_0[_S1023].xyz - state_0[_S1024].xyz + (scratch_0[_S1025].xyz - scratch_0[_S1026].xyz);
    (*_S1019) = to_local_0(_S1018, state_0[_S1021].xyz - state_0[_S1022].xyz + (scratch_0[sv_0(cb_6, u32(21))].xyz - scratch_0[sv_0(ca_2, u32(21))].xyz) + (cross(state_0[_S1023].xyz + scratch_0[_S1025].xyz, bonds_0[_S1018].rb_0.xyz) - cross(state_0[_S1024].xyz + scratch_0[_S1026].xyz, bonds_0[_S1018].ra_0.xyz)));
    (*_S1020) = to_local_0(_S1018, dth_1);
    return;
}

fn bond_kinematics_0( _S1027 : u32,  _S1028 : vec3<f32>,  _S1029 : vec3<f32>,  _S1030 : vec3<f32>,  _S1031 : vec3<f32>,  _S1032 : ptr<function, vec3<f32>>,  _S1033 : ptr<function, vec3<f32>>)
{
    (*_S1032) = to_local_0(_S1027, _S1030 + cross(_S1031, bonds_0[_S1027].rb_0.xyz) - (_S1028 + cross(_S1029, bonds_0[_S1027].ra_0.xyz)));
    (*_S1033) = to_local_0(_S1027, _S1031 - _S1029);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn island_statics(@builtin(workgroup_id) group_5 : vec3<u32>, @builtin(local_invocation_id) thread_5 : vec3<u32>)
{
    var i_21 : u32;
    var converged_4 : bool;
    var c_55 : u32;
    var tid_29 : u32 = thread_5.x;
    var _S1034 : u32 = group_5.x;
    var _S1035 : Island_std430_0 = islands_0[_S1034];
    var _S1036 : vec4<u32> = _S1035.info_0;
    var _S1037 : u32 = _S1035.info_0.z;
    if(((_S1037 & (u32(8)))) == u32(0))
    {
        return;
    }
    var free_0 : bool = (((_S1036.x) & (u32(1)))) == u32(0);
    var c0_1 : u32 = _S1035.range_0.x;
    var c1_1 : u32 = _S1035.range_0.y;
    var b0_0 : u32 = _S1035.range_0.z;
    var _S1038 : u32 = _S1035.range_0.w;
    var _S1039 : u32 = c0_1 + tid_29;
    var c_56 : u32 = _S1039;
    loop
    {
        if(c_56 < c1_1)
        {
        }
        else
        {
            break;
        }
        var _S1040 : u32 = u32(2) * c_56;
        scratch_0[sv_0(c_56, u32(0))] = scratch_0[params_0.sload_base_0 + _S1040];
        scratch_0[sv_0(c_56, u32(1))] = scratch_0[params_0.sload_base_0 + _S1040 + u32(1)];
        var _S1041 : vec4<f32> = vec4<f32>(0.0f);
        scratch_0[sv_0(c_56, u32(21))] = _S1041;
        scratch_0[sv_0(c_56, u32(22))] = _S1041;
        c_56 = c_56 + u32(256);
    }
    device_group_barrier_0();
    if(free_0)
    {
        project_load_slot_0(tid_29, &(_S1035), u32(0));
    }
    var _S1042 : f32 = island_dot_0(tid_29, c0_1, c1_1, u32(0), u32(0));
    var _S1043 : f32 = max(sqrt(_S1042), 1.00000000317107685e-30f);
    var _S1044 : f32 = params_0.statics_tol_0;
    var _S1045 : u32 = min(params_0.statics_cg_0, u32(20) * (c1_1 - c0_1) * u32(6) + u32(200));
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
        var _S1046 : u32 = b0_0 + tid_29;
        i_21 = _S1046;
        loop
        {
            if(i_21 < _S1038)
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
        device_group_barrier_0();
        var _S1047 : vec4<f32> = vec4<f32>(0.0f);
        var magnitude_0 : vec4<f32> = _S1047;
        var unused_m_0 : vec4<f32> = _S1047;
        c_56 = _S1039;
        loop
        {
            if(c_56 < c1_1)
            {
            }
            else
            {
                break;
            }
            var fi_4 : vec3<f32>;
            var mi_7 : vec3<f32>;
            gather_loads_0(c_56, &(fi_4), &(mi_7));
            magnitude_0[i32(0)] = magnitude_0[i32(0)] + bond_load_magnitude2_0(c_56);
            var r_lin_0 : vec4<f32> = vec4<f32>(scratch_0[sv_0(c_56, u32(0))].xyz + fi_4, 0.0f);
            var r_ang_0 : vec4<f32> = vec4<f32>(scratch_0[sv_0(c_56, u32(1))].xyz + mi_7, 0.0f);
            hold_0(fixed_mask_0(c_56), &(r_lin_0), &(r_ang_0), _S1047, _S1047);
            scratch_0[sv_0(c_56, u32(4))] = r_lin_0;
            scratch_0[sv_0(c_56, u32(5))] = r_ang_0;
            c_56 = c_56 + u32(256);
        }
        device_group_barrier_0();
        group_sum2_1(tid_29, &(magnitude_0), &(unused_m_0));
        if(free_0)
        {
            project_load_slot_0(tid_29, &(_S1035), u32(4));
        }
        var _S1048 : f32 = island_dot_0(tid_29, c0_1, c1_1, u32(4), u32(4));
        var residual_1 : f32 = sqrt(_S1048) / _S1043;
        var _S1049 : f32 = max(0.00100000004749745f, 3.83999986297567375e-06f * sqrt(magnitude_0.x) / _S1043);
        if(newton_0 == u32(0))
        {
            start_0 = residual_1;
        }
        var _S1050 : bool = is_finite_0(residual_1);
        var _S1051 : bool;
        if(_S1050)
        {
            if(residual_1 <= _S1044)
            {
                converged_4 = true;
            }
            else
            {
                if(residual_1 <= _S1049)
                {
                    converged_4 = residual_1 > (0.5f * previous_1);
                }
                else
                {
                    converged_4 = false;
                }
                if(converged_4)
                {
                    _S1051 = residual_1 <= start_0;
                }
                else
                {
                    _S1051 = false;
                }
                converged_4 = _S1051;
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
        if(!_S1050)
        {
            residual_0 = residual_1;
            converged_4 = false;
            break;
        }
        var i_22 : u32 = _S1046;
        loop
        {
            if(i_22 < _S1038)
            {
            }
            else
            {
                break;
            }
            var d_lin_12 : vec3<f32>;
            var d_ang_10 : vec3<f32>;
            static_kinematics_1(i_22, &(d_lin_12), &(d_ang_10));
            var _S1052 : JointBond_std430_0 = bonds_0[i_22].law_0;
            var _S1053 : JointState_std430_0 = bond_dyn_0[i_22].js_0;
            var f_lin_3 : vec3<f32>;
            var f_ang_3 : vec3<f32>;
            secant_factors_1(&(_S1052), &(_S1053), d_lin_12, &(f_lin_3), &(f_ang_3));
            var _S1054 : u32 = statics_bond_slot_0(i_22);
            var _S1055 : f32 = _S1052.stiff0_0.y;
            scratch_0[_S1054] = vec4<f32>(_S1055 * f_lin_3.x, _S1055 * f_lin_3.y, _S1052.stiff0_0.x * f_lin_3.z, 0.0f);
            scratch_0[_S1054 + u32(1)] = vec4<f32>(_S1052.stiff0_0.z * f_ang_3.x, _S1052.stiff0_0.w * f_ang_3.y, _S1052.stiff1_0.x * f_ang_3.z, 0.0f);
            i_22 = i_22 + u32(256);
        }
        device_group_barrier_0();
        var c_57 : u32 = _S1039;
        loop
        {
            if(c_57 < c1_1)
            {
            }
            else
            {
                break;
            }
            assemble_block_0(c_57);
            scratch_0[sv_0(c_57, u32(2))] = _S1047;
            scratch_0[sv_0(c_57, u32(3))] = _S1047;
            c_57 = c_57 + u32(256);
        }
        device_group_barrier_0();
        var c_58 : u32 = _S1039;
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
        device_group_barrier_0();
        if(free_0)
        {
            project_displacement_slot_0(tid_29, &(_S1035), u32(6));
        }
        var c_59 : u32 = _S1039;
        loop
        {
            if(c_59 < c1_1)
            {
            }
            else
            {
                break;
            }
            scratch_0[sv_0(c_59, u32(8))] = scratch_0[sv_0(c_59, u32(6))];
            scratch_0[sv_0(c_59, u32(9))] = scratch_0[sv_0(c_59, u32(7))];
            c_59 = c_59 + u32(256);
        }
        device_group_barrier_0();
        var _S1056 : f32 = island_dot_0(tid_29, c0_1, c1_1, u32(4), u32(6));
        var _S1057 : f32 = island_dot_0(tid_29, c0_1, c1_1, u32(4), u32(4));
        var _S1058 : f32 = sqrt(_S1057);
        var rz_0 : f32 = _S1056;
        var k_30 : u32 = u32(0);
        var cg_total_1 : u32 = cg_total_0;
        loop
        {
            if(k_30 < _S1045)
            {
                _S1051 = _S1058 > 0.0f;
            }
            else
            {
                _S1051 = false;
            }
            if(_S1051)
            {
            }
            else
            {
                cg_total_0 = cg_total_1;
                break;
            }
            var i_23 : u32 = _S1046;
            loop
            {
                if(i_23 < _S1038)
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
                var _S1059 : u32 = statics_bond_slot_0(i_23);
                write_bond_loads_0(i_23, i_23, d_lin_13 * scratch_0[_S1059].xyz, d_ang_11 * scratch_0[_S1059 + u32(1)].xyz, 0.0f);
                i_23 = i_23 + u32(256);
            }
            device_group_barrier_0();
            c_55 = _S1039;
            loop
            {
                if(c_55 < c1_1)
                {
                }
                else
                {
                    break;
                }
                var fi_5 : vec3<f32>;
                var mi_8 : vec3<f32>;
                gather_loads_0(c_55, &(fi_5), &(mi_8));
                var ap_lin_0 : vec4<f32> = vec4<f32>((vec3<f32>(0) - fi_5), 0.0f);
                var ap_ang_0 : vec4<f32> = vec4<f32>((vec3<f32>(0) - mi_8), 0.0f);
                hold_0(fixed_mask_0(c_55), &(ap_lin_0), &(ap_ang_0), scratch_0[sv_0(c_55, u32(8))], scratch_0[sv_0(c_55, u32(9))]);
                scratch_0[sv_0(c_55, u32(10))] = ap_lin_0;
                scratch_0[sv_0(c_55, u32(11))] = ap_ang_0;
                c_55 = c_55 + u32(256);
            }
            var pap_0 : f32 = island_dot_0(tid_29, c0_1, c1_1, u32(8), u32(10));
            var _S1060 : u32 = cg_total_1 + u32(1);
            if(pap_0 <= 0.0f)
            {
                cg_total_0 = _S1060;
                break;
            }
            var _S1061 : f32 = rz_0 / pap_0;
            var c_60 : u32 = _S1039;
            loop
            {
                if(c_60 < c1_1)
                {
                }
                else
                {
                    break;
                }
                var _S1062 : u32 = sv_0(c_60, u32(2));
                var _S1063 : vec3<f32> = vec3<f32>(_S1061);
                scratch_0[_S1062] = vec4<f32>(scratch_0[_S1062].xyz + _S1063 * scratch_0[sv_0(c_60, u32(8))].xyz, 0.0f);
                var _S1064 : u32 = sv_0(c_60, u32(3));
                scratch_0[_S1064] = vec4<f32>(scratch_0[_S1064].xyz + _S1063 * scratch_0[sv_0(c_60, u32(9))].xyz, 0.0f);
                var _S1065 : u32 = sv_0(c_60, u32(4));
                scratch_0[_S1065] = vec4<f32>(scratch_0[_S1065].xyz - _S1063 * scratch_0[sv_0(c_60, u32(10))].xyz, 0.0f);
                var _S1066 : u32 = sv_0(c_60, u32(5));
                scratch_0[_S1066] = vec4<f32>(scratch_0[_S1066].xyz - _S1063 * scratch_0[sv_0(c_60, u32(11))].xyz, 0.0f);
                c_60 = c_60 + u32(256);
            }
            var _S1067 : f32 = island_dot_0(tid_29, c0_1, c1_1, u32(4), u32(4));
            if((sqrt(_S1067)) <= (0.00009999999747379f * _S1058))
            {
                cg_total_0 = _S1060;
                break;
            }
            var c_61 : u32 = _S1039;
            loop
            {
                if(c_61 < c1_1)
                {
                }
                else
                {
                    break;
                }
                precondition_0(c_61);
                c_61 = c_61 + u32(256);
            }
            if(free_0)
            {
                device_group_barrier_0();
                project_displacement_slot_0(tid_29, &(_S1035), u32(6));
            }
            var rz_new_0 : f32 = island_dot_0(tid_29, c0_1, c1_1, u32(4), u32(6));
            var _S1068 : f32 = rz_new_0 / rz_0;
            var c_62 : u32 = _S1039;
            loop
            {
                if(c_62 < c1_1)
                {
                }
                else
                {
                    break;
                }
                var _S1069 : u32 = sv_0(c_62, u32(8));
                var _S1070 : vec3<f32> = vec3<f32>(_S1068);
                scratch_0[_S1069] = vec4<f32>(scratch_0[sv_0(c_62, u32(6))].xyz + _S1070 * scratch_0[_S1069].xyz, 0.0f);
                var _S1071 : u32 = sv_0(c_62, u32(9));
                scratch_0[_S1071] = vec4<f32>(scratch_0[sv_0(c_62, u32(7))].xyz + _S1070 * scratch_0[_S1071].xyz, 0.0f);
                c_62 = c_62 + u32(256);
            }
            device_group_barrier_0();
            var _S1072 : u32 = k_30 + u32(1);
            rz_0 = rz_new_0;
            k_30 = _S1072;
            cg_total_1 = _S1060;
        }
        c_55 = _S1039;
        loop
        {
            if(c_55 < c1_1)
            {
            }
            else
            {
                break;
            }
            var _S1073 : u32 = u32(4) * c_55;
            var u_7 : vec3<f32> = state_0[_S1073].xyz;
            var _S1074 : u32 = sv_0(c_55, u32(21));
            var u_lo_0 : vec3<f32> = scratch_0[_S1074].xyz;
            var _S1075 : u32 = _S1073 + u32(1);
            var th_9 : vec3<f32> = state_0[_S1075].xyz;
            var _S1076 : u32 = sv_0(c_55, u32(22));
            var th_lo_0 : vec3<f32> = scratch_0[_S1076].xyz;
            comp_add_0(&(u_7), &(u_lo_0), scratch_0[sv_0(c_55, u32(2))].xyz);
            comp_add_0(&(th_9), &(th_lo_0), scratch_0[sv_0(c_55, u32(3))].xyz);
            state_0[_S1073] = vec4<f32>(u_7, state_0[_S1073].w);
            state_0[_S1075] = vec4<f32>(th_9, state_0[_S1075].w);
            scratch_0[_S1074] = vec4<f32>(u_lo_0, 0.0f);
            scratch_0[_S1076] = vec4<f32>(th_lo_0, 0.0f);
            c_55 = c_55 + u32(256);
        }
        device_group_barrier_0();
        var _S1077 : u32 = newton_0 + u32(1);
        previous_1 = residual_1;
        residual_0 = residual_1;
        newton_0 = _S1077;
    }
    i_21 = b0_0 + tid_29;
    loop
    {
        if(i_21 < _S1038)
        {
        }
        else
        {
            break;
        }
        var resp_4 : JointResponse_0 = static_response_0(i_21);
        var _S1078 : JointState_0 = JointState_0( bond_dyn_0[i_21].js_0.damage_0, bond_dyn_0[i_21].js_0.crush_1, bond_dyn_0[i_21].js_0.kappa_0, bond_dyn_0[i_21].js_0.kappa_c_0, bond_dyn_0[i_21].js_0.ductility_0, bond_dyn_0[i_21].js_0.ductility_c_0, bond_dyn_0[i_21].js_0.life_0, bond_dyn_0[i_21].js_0.plastic_x_0, bond_dyn_0[i_21].js_0.plastic_y_0, bond_dyn_0[i_21].js_0.plastic_t_0, bond_dyn_0[i_21].js_0.rebar_plastic_0, bond_dyn_0[i_21].js_0.rebar_slip0_0, bond_dyn_0[i_21].js_0.rebar_slip1_0, bond_dyn_0[i_21].js_0.rebar_work_0, bond_dyn_0[i_21].js_0.rebar_broken_0, bond_dyn_0[i_21].js_0.strain_rate_0, bond_dyn_0[i_21].js_0.governing_stress_0, bond_dyn_0[i_21].js_0.dissipated_0, bond_dyn_0[i_21].js_0.utilization_0, bond_dyn_0[i_21].js_0.mode_0 );
        var bd_1 : BondDyn_0;
        bd_1.js_0 = _S1078;
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
    device_group_barrier_0();
    c_56 = _S1039;
    loop
    {
        if(c_56 < c1_1)
        {
        }
        else
        {
            break;
        }
        var fi_6 : vec3<f32>;
        var mi_9 : vec3<f32>;
        gather_loads_0(c_56, &(fi_6), &(mi_9));
        var reaction_2 : vec3<f32>;
        if((fixed_mask_0(c_56)) != u32(0))
        {
            reaction_2 = (vec3<f32>(0) - (scratch_0[sv_0(c_56, u32(0))].xyz + fi_6));
        }
        else
        {
            var _S1079 : u32 = u32(4) * c_56;
            reaction_2 = vec3<f32>(state_0[_S1079 + u32(1)].w, state_0[_S1079 + u32(2)].w, state_0[_S1079 + u32(3)].w);
        }
        var _S1080 : u32 = u32(4) * c_56;
        var _S1081 : u32 = _S1080 + u32(1);
        state_0[_S1081] = vec4<f32>(state_0[_S1081].xyz, reaction_2.x);
        state_0[_S1080 + u32(2)] = vec4<f32>(0.0f, 0.0f, 0.0f, reaction_2.y);
        state_0[_S1080 + u32(3)] = vec4<f32>(0.0f, 0.0f, 0.0f, reaction_2.z);
        c_56 = c_56 + u32(256);
    }
    if(tid_29 == u32(0))
    {
        var _S1082 : f32 = (bitcast<f32>((newton_0)));
        var _S1083 : f32 = (bitcast<f32>((cg_total_0)));
        if(converged_4)
        {
            previous_1 = 1.0f;
        }
        else
        {
            previous_1 = 0.0f;
        }
        scratch_0[statics_result_slot_0(_S1034)] = vec4<f32>(residual_0, _S1082, _S1083, previous_1);
        islands_0[_S1034].info_0[i32(2)] = (_S1037 & (u32(4294967287)));
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn settled_fatigue(@builtin(workgroup_id) group_6 : vec3<u32>, @builtin(local_invocation_id) thread_6 : vec3<u32>)
{
    var tid_30 : u32 = thread_6.x;
    var _S1084 : u32 = group_6.x;
    var isl_42 : Island_std430_0 = islands_0[_S1084];
    var _S1085 : bool;
    if((((islands_0[_S1084].info_0.x) & (u32(16)))) == u32(0))
    {
        _S1085 = true;
    }
    else
    {
        _S1085 = (isl_42.range_0.w) == (isl_42.range_0.z);
    }
    if(_S1085)
    {
        return;
    }
    seen_clear_0(tid_30);
    if(tid_30 == u32(0))
    {
        g_halt_0 = u32(0);
        g_run_0 = u32(0);
    }
    workgroupBarrier();
    var _S1086 : u32 = isl_42.info_0.w;
    var i_24 : u32 = isl_42.range_0.z + tid_30;
    loop
    {
        if(i_24 < (isl_42.range_0.w))
        {
        }
        else
        {
            break;
        }
        var _S1087 : JointState_0 = JointState_0( bond_dyn_0[i_24].js_0.damage_0, bond_dyn_0[i_24].js_0.crush_1, bond_dyn_0[i_24].js_0.kappa_0, bond_dyn_0[i_24].js_0.kappa_c_0, bond_dyn_0[i_24].js_0.ductility_0, bond_dyn_0[i_24].js_0.ductility_c_0, bond_dyn_0[i_24].js_0.life_0, bond_dyn_0[i_24].js_0.plastic_x_0, bond_dyn_0[i_24].js_0.plastic_y_0, bond_dyn_0[i_24].js_0.plastic_t_0, bond_dyn_0[i_24].js_0.rebar_plastic_0, bond_dyn_0[i_24].js_0.rebar_slip0_0, bond_dyn_0[i_24].js_0.rebar_slip1_0, bond_dyn_0[i_24].js_0.rebar_work_0, bond_dyn_0[i_24].js_0.rebar_broken_0, bond_dyn_0[i_24].js_0.strain_rate_0, bond_dyn_0[i_24].js_0.governing_stress_0, bond_dyn_0[i_24].js_0.dissipated_0, bond_dyn_0[i_24].js_0.utilization_0, bond_dyn_0[i_24].js_0.mode_0 );
        var bd_2 : BondDyn_0;
        bd_2.js_0 = _S1087;
        bd_2.force_lin_0 = bond_dyn_0[i_24].force_lin_0;
        bd_2.force_ang_0 = bond_dyn_0[i_24].force_ang_0;
        bd_2.sums_0 = bond_dyn_0[i_24].sums_0;
        bd_2.comps_0 = bond_dyn_0[i_24].comps_0;
        bd_2.events_0 = bond_dyn_0[i_24].events_0;
        var _S1088 : JointBond_std430_0 = bonds_0[i_24].law_0;
        var _S1089 : u32 = u32(4) * _S1088.ids_0.y;
        var _S1090 : u32 = u32(4) * _S1088.ids_0.z;
        var d_lin_14 : vec3<f32>;
        var d_ang_12 : vec3<f32>;
        bond_kinematics_0(i_24, state_0[_S1089].xyz, state_0[_S1089 + u32(1)].xyz, state_0[_S1090].xyz, state_0[_S1090 + u32(1)].xyz, &(d_lin_14), &(d_ang_12));
        var _S1091 : JointMaterial_std140_0 = materials_0.m_0.data_0[_S1088.ids_0.x];
        var previous_2 : JointState_0 = bd_2.js_0;
        var _S1092 : JointResponse_0 = joint_evaluate_2(&(_S1091), &(_S1088), bd_2.js_0, d_lin_14, d_ang_12, params_0.dt_0, (params_0.fracture_0) != u32(0));
        if((_S1092.state_1.damage_0) > (bd_2.js_0.damage_0 + 9.99999971718068537e-10f))
        {
            _S1085 = true;
        }
        else
        {
            _S1085 = (_S1092.state_1.crush_1) > (previous_2.crush_1 + 9.99999971718068537e-10f);
        }
        var flags_5 : u32;
        if(_S1085)
        {
            flags_5 = u32(16);
        }
        else
        {
            flags_5 = u32(0);
        }
        var _S1093 : f32 = bd_2.sums_0[i32(0)];
        var _S1094 : f32 = bd_2.comps_0[i32(0)];
        comp_add1_0(&(_S1093), &(_S1094), _S1092.dissipated_2);
        bd_2.sums_0[i32(0)] = _S1093;
        bd_2.comps_0[i32(0)] = _S1094;
        var _S1095 : f32 = bd_2.sums_0[i32(1)];
        var _S1096 : f32 = bd_2.comps_0[i32(1)];
        comp_add1_0(&(_S1095), &(_S1096), _S1092.overshoot_0);
        bd_2.sums_0[i32(1)] = _S1095;
        bd_2.comps_0[i32(1)] = _S1096;
        bd_2.force_lin_0 = vec4<f32>(_S1092.force_lin_1, _S1092.stored_2);
        bd_2.force_ang_0 = vec4<f32>(_S1092.force_ang_1, max(bd_2.force_ang_0.w, _S1092.state_1.utilization_0));
        var _S1097 : bool;
        if(!is_damaged_1(previous_2))
        {
            _S1097 = is_damaged_1(_S1092.state_1);
        }
        else
        {
            _S1097 = false;
        }
        var _S1098 : bool;
        if(_S1097)
        {
            _S1098 = (bd_2.events_0.x) == u32(0);
        }
        else
        {
            _S1098 = false;
        }
        if(_S1098)
        {
            bd_2.events_0[i32(0)] = _S1086;
            bd_2.events_0[i32(3)] = _S1092.state_1.mode_0;
            g_seen_0[u32(3)] = u32(1);
        }
        var _S1099 : bool;
        if((bd_2.events_0.y) == u32(0))
        {
            _S1099 = (_S1092.state_1.life_0) != (previous_2.life_0);
        }
        else
        {
            _S1099 = false;
        }
        var _S1100 : bool;
        if(_S1099)
        {
            _S1100 = (_S1092.fatigue_before_0) > 0.99000000953674316f;
        }
        else
        {
            _S1100 = false;
        }
        var _S1101 : bool;
        if(_S1100)
        {
            var _S1102 : f32 = fatigue_factor_2(&(_S1091), _S1092.state_1.life_0);
            _S1101 = _S1102 <= 0.99000000953674316f;
        }
        else
        {
            _S1101 = false;
        }
        if(_S1101)
        {
            bd_2.events_0[i32(1)] = _S1086;
            g_seen_0[u32(4)] = u32(1);
        }
        var flags_6 : u32;
        if(_S1092.disconnected_0)
        {
            bd_2.events_0[i32(2)] = _S1086;
            flags_6 = (flags_5 | (u32(32)));
        }
        else
        {
            flags_6 = flags_5;
        }
        bd_2.js_0 = _S1092.state_1;
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
        write_bond_loads_0(i_24, i_24, _S1092.force_lin_1, _S1092.force_ang_1, max(_S1092.measures_0.tension_0, _S1092.measures_0.compression_0));
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
    if(tid_30 != u32(0))
    {
        return;
    }
    if(((g_halt_0 | (g_run_0))) != u32(0))
    {
        var _S1103 : u32 = isl_42.info_0.z;
        if(g_halt_0 != u32(0))
        {
            i_24 = u32(16);
        }
        else
        {
            i_24 = u32(0);
        }
        var _S1104 : u32 = (_S1103 | (i_24));
        if(g_run_0 != u32(0))
        {
            i_24 = u32(32);
        }
        else
        {
            i_24 = u32(0);
        }
        islands_0[_S1084].info_0[i32(2)] = (_S1104 | (i_24));
    }
    var _S1105 : u32 = ((((isl_42.coverage_0) | (u32(16384)))) | ((seen_bits_0())));
    if(g_run_0 != u32(0))
    {
        i_24 = u32(128);
    }
    else
    {
        i_24 = u32(0);
    }
    islands_0[_S1084].coverage_0 = (_S1105 | (i_24));
    return;
}

fn impactor_integrate_at_0( ii_3 : u32,  tid_31 : u32,  fused_4 : bool)
{
    var p_25 : vec3<f32>;
    if(ii_3 >= (params_0.impactor_count_0))
    {
        return;
    }
    var _S1106 : u32 = params_0.halt_index_0;
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
    var _S1107 : vec4<u32> = islands_0[params_0.halt_index_0].info_0;
    var stop_0 : bool;
    if((((islands_0[params_0.halt_index_0].info_0.z) & (u32(1)))) != u32(0))
    {
        stop_0 = true;
    }
    else
    {
        var _S1108 : u32 = _S1107.y;
        if(_S1108 != u32(0))
        {
            stop_0 = (imp_7.cand_0.w) >= _S1108;
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
            var _S1109 : u32 = islands_0[params_0.halt_index_0].range_0.x;
            if(_S1109 != u32(0))
            {
                stop_0 = _S1109 <= (imp_7.cand_0.w + u32(1));
            }
            else
            {
                stop_0 = false;
            }
        }
        if(tid_31 == u32(0))
        {
            stop_seen_0(_S1106, imp_7.cand_0.w + u32(1));
        }
    }
    if(stop_0)
    {
        return;
    }
    if((imp_7.cand_0.z) != u32(0))
    {
        if(tid_31 == u32(0))
        {
            impactors_0[ii_3].cand_0[i32(3)] = imp_7.cand_0.w + u32(1);
        }
        return;
    }
    var _S1110 : vec4<f32> = vec4<f32>(0.0f);
    var rf_1 : vec4<f32> = _S1110;
    var rt_0 : vec4<f32> = _S1110;
    var k_31 : u32 = imp_7.cand_0.x + tid_31;
    loop
    {
        if(k_31 < (imp_7.cand_0.y))
        {
        }
        else
        {
            break;
        }
        var _S1111 : u32 = u32(3) * k_31;
        rf_1 = rf_1 + scratch_0[params_0.cand_base_0 + _S1111 + u32(1)];
        rt_0 = rt_0 + scratch_0[params_0.cand_base_0 + _S1111 + u32(2)];
        k_31 = k_31 + u32(256);
    }
    group_sum2_3(tid_31, &(rf_1), &(rt_0));
    if(tid_31 != u32(0))
    {
        return;
    }
    var dt_21 : f32 = params_0.dt_0;
    var _S1112 : vec3<f32> = vec3<f32>(0.0f);
    var load_f_0 : vec3<f32>;
    var load_t_0 : vec3<f32>;
    if((params_0.has_ground_0) != u32(0))
    {
        var ib_0 : Box_0 = impactor_box_0(imp_7, _S1112, imp_7.half_1.xyz);
        var _S1113 : vec3<f32> = imp_7.velocity_1.xyz + imp_7.velocity_err_1.xyz;
        const _S1114 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 1.0f);
        var kc_2 : f32 = contact_stiffness_0(params_0.ground_modulus_0, ib_0, imp_7.mat_0.x, ib_0, _S1114);
        var _S1115 : f32 = imp_7.position_1.z - params_0.ground_hi_0 + (imp_7.position_err_1.z - params_0.ground_lo_0);
        var total_points_0 : u32;
        if((imp_7.shape_0.x) == 0.0f)
        {
            total_points_0 = u32(1);
        }
        else
        {
            total_points_0 = u32(14);
        }
        var _S1116 : f32 = kc_2 / f32(min(total_points_0, u32(5)));
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
            if((_S1115 + p_25.z) < 0.0f)
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
        load_f_0 = _S1112;
        load_t_0 = _S1112;
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
            var depth_8 : f32 = - (_S1115 + p_25.z);
            if(depth_8 <= 0.0f)
            {
                s_18 = s_18 + u32(1);
                continue;
            }
            var stored_16 : f32;
            var diss_8 : f32;
            var f_30 : vec3<f32> = penalty_force_1(_S1116, imp_7.mat_0.z, params_0.ground_friction_0, depth_8, _S1114, _S1113 + cross(imp_7.angular_velocity_1.xyz, p_25), dt_21, below_0, &(stored_16), &(diss_8));
            var load_f_1 : vec3<f32> = load_f_0 + f_30;
            var load_t_1 : vec3<f32> = load_t_0 + cross(p_25, f_30);
            var _S1117 : f32 = imp_7.ledger_0[i32(0)];
            var _S1118 : f32 = imp_7.ledger_0[i32(1)];
            comp_add1_4(&(_S1117), &(_S1118), diss_8);
            imp_7.ledger_0[i32(0)] = _S1117;
            imp_7.ledger_0[i32(1)] = _S1118;
            load_f_0 = load_f_1;
            load_t_0 = load_t_1;
            s_18 = s_18 + u32(1);
        }
    }
    else
    {
        load_f_0 = _S1112;
        load_t_0 = _S1112;
    }
    var load_f_2 : vec3<f32> = rf_1.xyz + load_f_0;
    var load_t_2 : vec3<f32> = rt_0.xyz + load_t_0;
    var m_10 : f32 = imp_7.mat_0.z;
    var vel_1 : vec3<f32> = imp_7.velocity_1.xyz;
    var vel_err_1 : vec3<f32> = imp_7.velocity_err_1.xyz;
    var _S1119 : vec3<f32> = vec3<f32>(dt_21);
    comp_add_1(&(vel_1), &(vel_err_1), (load_f_2 / vec3<f32>(m_10) + params_0.gravity_0.xyz) * _S1119);
    var _S1120 : Quat_0 = quat_of_0(imp_7.rotation_1);
    var q_15 : Quat_0 = _S1120;
    var q_err_0 : vec4<f32> = imp_7.rotation_err_1;
    var l_hi_0 : vec3<f32> = imp_7.momentum_1.xyz;
    var l_err_1 : vec3<f32> = imp_7.momentum_err_1.xyz;
    comp_add_1(&(l_hi_0), &(l_err_1), load_t_2 * _S1119);
    var l_10 : vec3<f32> = l_hi_0 + l_err_1;
    var w_mid_3 : vec3<f32> = world_mul_0(_S1120, imp_7.inv0_2, imp_7.inv1_2, imp_7.inv2_2, l_10);
    var pos_1 : vec3<f32> = imp_7.position_1.xyz;
    var pos_err_1 : vec3<f32> = imp_7.position_err_1.xyz;
    comp_add_1(&(pos_1), &(pos_err_1), (vel_1 + vel_err_1) * _S1119);
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
    var k_32 : u32 = imp_7.cand_0.w - u32(1) - params_0.step_start_0;
    if(k_32 < (params_0.record_stride_0))
    {
        var at_9 : u32 = params_0.record_base_0 + u32(2) * (ii_3 * params_0.record_stride_0 + k_32);
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
    var _S1121 : bool;
    if(g_5 >= (params_0.seg_count_0))
    {
        _S1121 = true;
    }
    else
    {
        _S1121 = stopped_0();
    }
    if(_S1121)
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

fn net_load_2( c_63 : u32,  isl_43 : ptr<function, Island_std430_0>,  rg_27 : Rigid_0,  k_33 : u32,  dt_22 : f32,  contact_7 : bool,  f_31 : ptr<function, vec3<f32>>,  t_31 : ptr<function, vec3<f32>>)
{
    var fl_2 : vec3<f32>;
    var tl_2 : vec3<f32>;
    var rep_f_4 : vec3<f32>;
    var rep_m_4 : vec3<f32>;
    chunk_loads_1(c_63, c_63, rg_27.rot_0, k_33, dt_22, contact_7, 0.0f, &(fl_2), &(tl_2), &(rep_f_4), &(rep_m_4));
    var fc_6 : vec3<f32> = fl_2 + params_0.gravity_0.xyz * vec3<f32>(chunks_0[c_63].center_0.w);
    var _S1122 : vec3<f32> = chunks_0[c_63].center_0.xyz;
    var _S1123 : vec3<f32> = (*isl_43).com_0.xyz;
    var r_21 : vec3<f32> = rotate_0(rg_27.rot_0, _S1122 + state_0[u32(4) * c_63].xyz - _S1123);
    (*f_31) = (*f_31) + fc_6;
    (*t_31) = (*t_31) + (cross(r_21, fc_6) + tl_2);
    if((chunks_0[c_63].load_range_0.y) > (chunks_0[c_63].load_range_0.x))
    {
        var fw_4 : vec3<f32> = rotate_0(rg_27.rot_0, rep_f_4);
        (*f_31) = (*f_31) + fw_4;
        (*t_31) = (*t_31) + (cross(rotate_0(rg_27.rot_0, _S1122 - _S1123), fw_4) + rotate_0(rg_27.rot_0, rep_m_4));
    }
    return;
}

fn chunk_update_2( c_64 : u32,  isl_44 : ptr<function, Island_std430_0>,  rg_28 : Rigid_0,  dt_23 : f32,  rml_2 : bool,  step_2 : u32,  contact_8 : bool,  work_11 : ptr<function, f32>,  work_err_7 : ptr<function, f32>)
{
    var _S1124 : vec3<f32> = vec3<f32>(0.0f);
    var _S1125 : u32 = index_0[c_64];
    var peak_2 : f32 = 0.0f;
    var e_10 : u32 = _S1125;
    var fi_7 : vec3<f32> = _S1124;
    var mi_10 : vec3<f32> = _S1124;
    loop
    {
        if(e_10 < index_0[c_64 + u32(1)])
        {
        }
        else
        {
            break;
        }
        var entry_9 : u32 = index_0[e_10];
        var _S1126 : u32 = u32(3) * ((entry_9 >> (u32(1))));
        var fa_3 : vec4<f32> = scratch_0[_S1126];
        if(((entry_9 & (u32(1)))) == u32(0))
        {
            var mi_11 : vec3<f32> = mi_10 + scratch_0[_S1126 + u32(1)].xyz;
            fi_7 = fi_7 + fa_3.xyz;
            mi_10 = mi_11;
        }
        else
        {
            var mi_12 : vec3<f32> = mi_10 + scratch_0[_S1126 + u32(2)].xyz;
            fi_7 = fi_7 + (vec3<f32>(0) - fa_3.xyz);
            mi_10 = mi_12;
        }
        var _S1127 : f32 = max(peak_2, fa_3.w);
        var _S1128 : u32 = e_10 + u32(1);
        peak_2 = _S1127;
        e_10 = _S1128;
    }
    var _S1129 : u32 = u32(4) * c_64;
    var u_8 : vec3<f32> = state_0[_S1129].xyz;
    var _S1130 : u32 = _S1129 + u32(1);
    var th_10 : vec3<f32> = state_0[_S1130].xyz;
    var _S1131 : u32 = _S1129 + u32(2);
    var v_16 : vec3<f32> = state_0[_S1131].xyz;
    var _S1132 : u32 = _S1129 + u32(3);
    var w_12 : vec3<f32> = state_0[_S1132].xyz;
    var mass_2 : f32 = chunks_0[c_64].center_0.w;
    var _S1133 : vec3<f32> = chunks_0[c_64].center_0.xyz;
    var _S1134 : vec3<f32> = (*isl_44).com_0.xyz;
    var r_world_2 : vec3<f32> = rotate_0(rg_28.rot_0, _S1133 + u_8 - _S1134);
    var f_load_2 : vec3<f32>;
    var t_load_2 : vec3<f32>;
    var rep_f_5 : vec3<f32>;
    var rep_m_5 : vec3<f32>;
    chunk_loads_1(c_64, c_64, rg_28.rot_0, step_2, dt_23, contact_8, dt_23, &(f_load_2), &(t_load_2), &(rep_f_5), &(rep_m_5));
    record_chunk_load_0(c_64, f_load_2, t_load_2);
    var _S1135 : vec3<f32> = vec3<f32>(mass_2);
    var f_world_4 : vec3<f32> = f_load_2 + params_0.gravity_0.xyz * _S1135;
    var t_world_6 : vec3<f32> = t_load_2;
    var f_world_5 : vec3<f32>;
    var t_world_7 : vec3<f32>;
    if(rml_2)
    {
        var t_world_8 : vec3<f32> = t_world_6 - (world_mul_0(rg_28.rot_0, chunks_0[c_64].inertia0_1, chunks_0[c_64].inertia1_1, chunks_0[c_64].inertia2_1, rg_28.alpha_0) + cross(rg_28.w_1, world_mul_0(rg_28.rot_0, chunks_0[c_64].inertia0_1, chunks_0[c_64].inertia1_1, chunks_0[c_64].inertia2_1, rg_28.w_1)));
        f_world_5 = f_world_4 - (rg_28.a_0 + cross(rg_28.alpha_0, r_world_2) + cross(rg_28.w_1, cross(rg_28.w_1, r_world_2))) * _S1135;
        t_world_7 = t_world_8;
    }
    else
    {
        f_world_5 = f_world_4;
        t_world_7 = t_world_6;
    }
    var f_ext_4 : vec3<f32> = inverse_rotate_0(rg_28.rot_0, f_world_5);
    var m_ext_6 : vec3<f32> = inverse_rotate_0(rg_28.rot_0, t_world_7);
    var f_ext_5 : vec3<f32>;
    var m_ext_7 : vec3<f32>;
    if(rml_2)
    {
        var wb_3 : vec3<f32> = inverse_rotate_0(rg_28.rot_0, rg_28.w_1);
        var i_w_2 : vec3<f32> = rows_mul_0(chunks_0[c_64].inertia0_1, chunks_0[c_64].inertia1_1, chunks_0[c_64].inertia2_1, w_12);
        var m_ext_8 : vec3<f32> = m_ext_6 - (cross(wb_3, i_w_2) + cross(w_12, rows_mul_0(chunks_0[c_64].inertia0_1, chunks_0[c_64].inertia1_1, chunks_0[c_64].inertia2_1, wb_3)) + cross(w_12, i_w_2));
        f_ext_5 = f_ext_4 - cross(wb_3, v_16) * vec3<f32>((2.0f * mass_2));
        m_ext_7 = m_ext_8;
    }
    else
    {
        f_ext_5 = f_ext_4;
        m_ext_7 = m_ext_6;
    }
    var f_32 : vec3<f32> = f_ext_5 + rep_f_5 + fi_7;
    var m_11 : vec3<f32> = m_ext_7 + rep_m_5 + mi_10;
    var support_3 : u32 = chunks_0[c_64].info_1.x;
    var _S1136 : vec3<f32> = vec3<f32>(state_0[_S1130].w, state_0[_S1131].w, state_0[_S1132].w);
    var reaction_3 : vec3<f32>;
    var u_9 : vec3<f32>;
    var th_11 : vec3<f32>;
    var v_17 : vec3<f32>;
    var w_13 : vec3<f32>;
    if(support_3 == u32(1))
    {
        reaction_3 = (vec3<f32>(0) - f_32);
        u_9 = u_8;
        th_11 = th_10;
        v_17 = _S1124;
        w_13 = _S1124;
    }
    else
    {
        var _S1137 : vec4<f32> = chunks_0[c_64].scale_0;
        var w_14 : vec3<f32> = w_12 + rows_mul_0(chunks_0[c_64].inv0_1, chunks_0[c_64].inv1_1, chunks_0[c_64].inv2_1, m_11) * vec3<f32>((dt_23 * chunks_0[c_64].scale_0.z));
        var _S1138 : vec3<f32> = vec3<f32>(dt_23);
        var th_12 : vec3<f32> = th_10 + w_14 * _S1138;
        if(support_3 == u32(2))
        {
            reaction_3 = (vec3<f32>(0) - f_32);
            u_9 = u_8;
            th_11 = _S1124;
        }
        else
        {
            var v_18 : vec3<f32> = v_16 + f_32 * vec3<f32>((dt_23 * _S1137.y));
            var u_10 : vec3<f32> = u_8 + v_18 * _S1138;
            reaction_3 = _S1136;
            u_9 = u_10;
            th_11 = v_18;
        }
        var _S1139 : vec3<f32> = th_11;
        th_11 = th_12;
        v_17 = _S1139;
        w_13 = w_14;
    }
    state_0[_S1129] = vec4<f32>(u_9, peak_2);
    state_0[_S1130] = vec4<f32>(th_11, reaction_3.x);
    state_0[_S1131] = vec4<f32>(v_17, reaction_3.y);
    state_0[_S1132] = vec4<f32>(w_13, reaction_3.z);
    comp_add1_0(&((*work_11)), &((*work_err_7)), (dot(f_load_2, rg_28.vel_0 + rg_28.vel_err_0 + cross(rg_28.w_1, rotate_0(rg_28.rot_0, _S1133 + u_9 - _S1134)) + rotate_0(rg_28.rot_0, v_17)) + dot(t_load_2, rg_28.w_1 + rotate_0(rg_28.rot_0, w_13))) * dt_23);
    return;
}

fn drift_moments_2( c_65 : u32,  tu_9 : ptr<function, vec3<f32>>,  pv_9 : ptr<function, vec3<f32>>)
{
    var _S1140 : u32 = u32(4) * c_65;
    var _S1141 : vec3<f32> = vec3<f32>((chunks_0[c_65].center_0.w * chunks_0[c_65].scale_0.x));
    (*tu_9) = (*tu_9) + state_0[_S1140].xyz * _S1141;
    (*pv_9) = (*pv_9) + state_0[_S1140 + u32(2)].xyz * _S1141;
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn island_frame(@builtin(workgroup_id) group_8 : vec3<u32>, @builtin(local_invocation_id) thread_8 : vec3<u32>)
{
    island_frame_at_4(group_8.x, thread_8.x);
    return;
}

@compute
@workgroup_size(512, 1, 1)
fn island_frame_bonds_wide(@builtin(workgroup_id) group_9 : vec3<u32>, @builtin(local_invocation_id) thread_9 : vec3<u32>)
{
    island_frame_at_3(group_9.x, thread_9.x);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn contact_islands(@builtin(workgroup_id) group_10 : vec3<u32>, @builtin(local_invocation_id) thread_10 : vec3<u32>)
{
    island_frame_at_2(group_10.x, thread_10.x);
    return;
}

fn wide_chunks_at_0( gx_0 : u32,  tid_32 : u32)
{
    var wg_6 : WideGroup_0 = wide_group_0(params_0.wide_chunk_table_0, gx_0);
    var _S1142 : Island_std430_0 = islands_0[wg_6.island_2];
    var _S1143 : bool = wide_enter_1(tid_32, wg_6.island_2);
    if(!_S1143)
    {
        return;
    }
    var _S1144 : u32 = _S1142.info_0.x;
    var anchored_0 : bool = ((_S1144 & (u32(1)))) != u32(0);
    if(((_S1144 & (u32(16)))) != u32(0))
    {
        if(tid_32 == u32(0))
        {
            scratch_0[params_0.wide_base_0 + u32(8) * gx_0 + u32(6)] = vec4<f32>(0.0f);
        }
        return;
    }
    var _S1145 : Rigid_0 = wide_rigid_frame_0(tid_32, &(_S1142), wg_6);
    var work_12 : f32 = 0.0f;
    var work_err_8 : f32 = 0.0f;
    var _S1146 : vec3<f32> = vec3<f32>(0.0f);
    var tu_10 : vec3<f32> = _S1146;
    var pv_10 : vec3<f32> = _S1146;
    var c_66 : u32 = wg_6.begin_1 + tid_32;
    if(c_66 < (wg_6.end_0))
    {
        var _S1147 : bool = (params_0.rigid_motion_loads_0) != u32(0);
        var _S1148 : u32 = wide_step_0(&(_S1142));
        chunk_update_2(c_66, &(_S1142), _S1145, params_0.dt_0, _S1147, _S1148, ((_S1144 & (u32(4)))) != u32(0), &(work_12), &(work_err_8));
        if(!anchored_0)
        {
            drift_moments_2(c_66, &(tu_10), &(pv_10));
        }
    }
    var wsum_5 : vec3<f32> = vec3<f32>(work_12, work_err_8, 0.0f);
    var unused_8 : vec3<f32> = _S1146;
    group_sum3_2(tid_32, &(wsum_5), &(unused_8));
    var _S1149 : bool = !anchored_0;
    if(_S1149)
    {
        group_sum3_2(tid_32, &(tu_10), &(pv_10));
    }
    if(tid_32 == u32(0))
    {
        var _S1150 : u32 = u32(8) * gx_0;
        scratch_0[params_0.wide_base_0 + _S1150 + u32(6)] = vec4<f32>(wsum_5, 0.0f);
        if(_S1149)
        {
            wide_store_0(gx_0, u32(2), tu_10, pv_10);
        }
        scratch_0[params_0.wide_base_0 + _S1150 + u32(7)] = vec4<f32>(f32(g_seen_0[u32(0)]), f32(g_seen_0[u32(1)]), f32(g_seen_0[u32(2)]), 0.0f);
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
    var _S1151 : u32 = group_12.x;
    if(_S1151 < sums_groups_0)
    {
        contact_sum_at_0(_S1151 * u32(256) + thread_12.x);
    }
    else
    {
        impactor_integrate_at_0(_S1151 - sums_groups_0, thread_12.x, false);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn islands_and_wide_chunks(@builtin(workgroup_id) group_13 : vec3<u32>, @builtin(local_invocation_id) thread_13 : vec3<u32>)
{
    var _S1152 : u32 = group_13.x;
    if(_S1152 < (params_0.halt_index_0))
    {
        island_frame_at_4(_S1152, thread_13.x);
    }
    else
    {
        wide_chunks_at_0(_S1152 - params_0.halt_index_0, thread_13.x);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn contact_islands_and_wide_chunks(@builtin(workgroup_id) group_14 : vec3<u32>, @builtin(local_invocation_id) thread_14 : vec3<u32>)
{
    var _S1153 : u32 = group_14.x;
    if(_S1153 < (params_0.halt_index_0))
    {
        island_frame_at_2(_S1153, thread_14.x);
    }
    else
    {
        wide_chunks_at_0(_S1153 - params_0.halt_index_0, thread_14.x);
    }
    return;
}

fn wide_bonds_at_0( gx_1 : u32,  tid_33 : u32)
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
    var _S1154 : WideGroup_0 = wg_7;
    var _S1155 : Island_std430_0 = islands_0[wg_7.island_2];
    var _S1156 : bool = wide_enter_1(tid_33, wg_7.island_2);
    if(!_S1156)
    {
        return;
    }
    var _S1157 : u32 = wide_step_1(&(_S1155));
    if(bond_group_0)
    {
        var _S1158 : vec4<u32> = _S1155.info_0;
        if((((_S1155.info_0.x) & (u32(16)))) != u32(0))
        {
            return;
        }
        var i_25 : u32 = wg_7.begin_1 + tid_33;
        var _S1159 : bool;
        if(i_25 < (wg_7.end_0))
        {
            var _S1160 : bool = bond_update_0(i_25, params_0.dt_0, (params_0.fracture_0) != u32(0), _S1158.w + u32(1), true);
            _S1159 = _S1160;
        }
        else
        {
            _S1159 = false;
        }
        if(_S1159)
        {
            islands_0[_S1154.island_2].info_0[i32(2)] = ((_S1158.z) | (u32(2)));
        }
        return;
    }
    var _S1161 : u32 = _S1155.info_0.x;
    if(((_S1161 & (u32(1)))) != u32(0))
    {
        return;
    }
    var _S1162 : vec3<f32> = vec3<f32>(0.0f);
    var f_33 : vec3<f32> = _S1162;
    var t_32 : vec3<f32> = _S1162;
    var c_67 : u32 = wg_7.begin_1 + tid_33;
    if(c_67 < (wg_7.end_0))
    {
        var _S1163 : Rigid_0 = rigid_of_2(&(_S1155));
        net_load_2(c_67, &(_S1155), _S1163, _S1157, params_0.dt_0, ((_S1161 & (u32(4)))) != u32(0), &(f_33), &(t_32));
    }
    group_sum3_0(tid_33, &(f_33), &(t_32));
    if(tid_33 == u32(0))
    {
        wide_store_0(gx_1 - params_0.wide_bond_groups_0, u32(0), f_33, t_32);
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

fn bond_group_at_0( table_1 : u32,  gx_2 : u32,  tid_34 : u32)
{
    var wg_8 : WideGroup_0 = wide_group_0(table_1, gx_2);
    var isl_45 : Island_std430_0 = islands_0[wg_8.island_2];
    var _S1164 : bool = wide_enter_1(tid_34, wg_8.island_2);
    var _S1165 : bool;
    if(!_S1164)
    {
        _S1165 = true;
    }
    else
    {
        _S1165 = (((isl_45.info_0.x) & (u32(16)))) != u32(0);
    }
    if(_S1165)
    {
        return;
    }
    var i_26 : u32 = wg_8.begin_1 + tid_34;
    if(i_26 < (wg_8.end_0))
    {
        var _S1166 : bool = bond_update_0(i_26, params_0.dt_0, (params_0.fracture_0) != u32(0), isl_45.info_0.w + u32(1), true);
        _S1165 = _S1166;
    }
    else
    {
        _S1165 = false;
    }
    if(_S1165)
    {
        islands_0[wg_8.island_2].info_0[i32(2)] = ((isl_45.info_0.z) | (u32(2)));
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn sums_impactors_bonds(@builtin(workgroup_id) group_16 : vec3<u32>, @builtin(local_invocation_id) thread_16 : vec3<u32>)
{
    var sums_groups_1 : u32 = (params_0.seg_count_0 + u32(255)) / u32(256);
    var _S1167 : u32 = max(params_0.impactor_count_0, u32(1));
    var _S1168 : u32 = group_16.x;
    if(_S1168 < sums_groups_1)
    {
        contact_sum_at_0(_S1168 * u32(256) + thread_16.x);
    }
    else
    {
        if(_S1168 < (sums_groups_1 + _S1167))
        {
            impactor_integrate_at_0(_S1168 - sums_groups_1, thread_16.x, false);
        }
        else
        {
            var gb_0 : u32 = _S1168 - sums_groups_1 - _S1167;
            var narrow_0 : bool = gb_0 < (params_0.contact_bond_groups_0);
            var _S1169 : u32;
            if(narrow_0)
            {
                _S1169 = params_0.contact_bond_table_0;
            }
            else
            {
                _S1169 = params_0.wide_bond_table_0;
            }
            var _S1170 : u32;
            if(narrow_0)
            {
                _S1170 = gb_0;
            }
            else
            {
                _S1170 = gb_0 - params_0.contact_bond_groups_0;
            }
            bond_group_at_0(_S1169, _S1170, thread_16.x);
        }
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn forces_and_bonds(@builtin(workgroup_id) group_17 : vec3<u32>, @builtin(local_invocation_id) thread_17 : vec3<u32>)
{
    var force_groups_0 : u32 = ((params_0.pair_count_0 + params_0.cand_count_0) * pair_lane_count_0() + u32(255)) / u32(256);
    var _S1171 : u32 = group_17.x;
    if(_S1171 < force_groups_0)
    {
        contact_forces_at_0(_S1171 * u32(256) + thread_17.x, true);
    }
    else
    {
        bond_group_at_0(params_0.contact_bond_table_0, _S1171 - force_groups_0, thread_17.x);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn islands_sums_impactors(@builtin(workgroup_id) group_18 : vec3<u32>, @builtin(local_invocation_id) thread_18 : vec3<u32>)
{
    var _S1172 : u32 = group_18.x;
    if(_S1172 < (params_0.halt_index_0))
    {
        island_frame_at_1(_S1172, thread_18.x);
    }
    else
    {
        impactor_integrate_at_0(_S1172 - params_0.halt_index_0, thread_18.x, true);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn islands_sums_impactors_wake(@builtin(workgroup_id) group_19 : vec3<u32>, @builtin(local_invocation_id) thread_19 : vec3<u32>)
{
    var _S1173 : u32 = group_19.x;
    if(_S1173 < (params_0.halt_index_0))
    {
        island_frame_at_0(_S1173, thread_19.x);
    }
    else
    {
        impactor_integrate_at_0(_S1173 - params_0.halt_index_0, thread_19.x, true);
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

