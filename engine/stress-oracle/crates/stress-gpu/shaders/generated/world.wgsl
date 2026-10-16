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
    return _S2;
}

var<workgroup> g_run_0 : u32;

var<workgroup> g_halt_0 : u32;

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
    var _S4 : vec3<f32> = vec3<f32>(0.0f);
    rg_0.torque_0 = _S4;
    rg_0.a_0 = _S4;
    rg_0.alpha_0 = _S4;
    return rg_0;
}

fn time_since_0( origin_0 : vec4<f32>,  k_0 : u32,  dt_1 : f32) -> f32
{
    return params_0.t_hi_0 - origin_0.x + (params_0.t_lo_0 - origin_0.y) + f32(k_0) * dt_1;
}

fn table_eval_0( offset_0 : u32,  count_0 : u32,  tau_0 : f32) -> f32
{
    var first_0 : vec4<f32> = loads_0[offset_0];
    if(tau_0 <= (first_0.x))
    {
        return first_0.y;
    }
    var i_0 : u32 = u32(1);
    loop
    {
        if(i_0 < count_0)
        {
        }
        else
        {
            break;
        }
        var _S5 : u32 = offset_0 + i_0;
        var b_0 : vec4<f32> = loads_0[_S5];
        var _S6 : f32 = b_0.x;
        if(tau_0 <= _S6)
        {
            var a_1 : vec4<f32> = loads_0[_S5 - u32(1)];
            var _S7 : f32 = a_1.x;
            var _S8 : f32 = a_1.y;
            return _S8 + (tau_0 - _S7) / max(_S6 - _S7, 1.00000000317107685e-30f) * (b_0.y - _S8);
        }
        i_0 = i_0 + u32(1);
    }
    return loads_0[offset_0 + count_0 - u32(1)].y;
}

fn eval_function_0( term_0 : u32,  k_1 : u32,  dt_2 : f32,  shift_0 : f32) -> f32
{
    var _S9 : u32 = u32(5) * term_0;
    var info_2 : vec4<u32> = (bitcast<vec4<u32>>((loads_0[_S9])));
    var origin_1 : vec4<f32> = loads_0[_S9 + u32(3)];
    var p_0 : vec4<f32> = loads_0[_S9 + u32(4)];
    var kind_0 : u32 = info_2.z;
    if(kind_0 == u32(0))
    {
        return origin_1.z;
    }
    var tau_1 : f32 = time_since_0(origin_1, k_1, dt_2) + shift_0;
    var shape_1 : f32;
    if(kind_0 == u32(1))
    {
        if(tau_1 <= 0.0f)
        {
            shape_1 = 0.0f;
        }
        else
        {
            var _S10 : f32 = p_0.x;
            if(tau_1 >= _S10)
            {
                shape_1 = p_0.y;
            }
            else
            {
                shape_1 = p_0.y * tau_1 / _S10;
            }
        }
        return shape_1;
    }
    var _S11 : bool;
    if(kind_0 == u32(2))
    {
        if(tau_1 < 0.0f)
        {
            _S11 = true;
        }
        else
        {
            _S11 = tau_1 > (p_0.x);
        }
        if(_S11)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = p_0.y * sin(3.14159274101257324f * tau_1 / p_0.x);
        }
        return shape_1;
    }
    if(kind_0 == u32(3))
    {
        var sn_0 : f32 = tau_1 / p_0.y;
        if(sn_0 < 0.0f)
        {
            _S11 = true;
        }
        else
        {
            _S11 = sn_0 > 1.0f;
        }
        if(_S11)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = p_0.x * (1.0f - sn_0) * exp(- p_0.z * sn_0);
        }
        return shape_1;
    }
    if(kind_0 == u32(4))
    {
        return table_eval_0(info_2.w, (bitcast<u32>((p_0.x))), tau_1);
    }
    if(kind_0 == u32(5))
    {
        g_seen_0[u32(1)] = u32(1);
        if(tau_1 < 0.0f)
        {
            return 0.0f;
        }
        var sn_1 : f32 = tau_1 / p_0.x;
        if(sn_1 < 0.0f)
        {
            _S11 = true;
        }
        else
        {
            _S11 = sn_1 > 1.0f;
        }
        if(_S11)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = (1.0f - sn_1) * exp(- p_0.y * sn_1);
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
        var _S12 : f32 = p_0.w;
        return (_S12 + (p_0.z - _S12) * relax_0) * shape_1;
    }
    if(kind_0 == u32(7))
    {
        g_seen_0[u32(2)] = u32(1);
        if(tau_1 < 0.0f)
        {
            return 0.0f;
        }
        var _S13 : f32 = p_0.y;
        if(tau_1 < _S13)
        {
            return p_0.x;
        }
        var s_0 : f32 = tau_1 - _S13;
        var _S14 : f32 = p_0.w;
        if(s_0 > _S14)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = p_0.z * sin(3.14159274101257324f * s_0 / _S14);
        }
        return shape_1;
    }
    var _S15 : f32 = p_0.x;
    if(_S15 <= 0.0f)
    {
        return 0.0f;
    }
    return clamp(1.0f - tau_1 / _S15, 0.0f, 1.0f);
}

fn rotate_0( q_1 : Quat_0,  v_0 : vec3<f32>) -> vec3<f32>
{
    var qv_0 : vec3<f32> = vec3<f32>(q_1.x_0, q_1.y_0, q_1.z_0);
    var t_0 : vec3<f32> = cross(qv_0, v_0) * vec3<f32>(2.0f);
    return v_0 + t_0 * vec3<f32>(q_1.w_0) + cross(qv_0, t_0);
}

fn chunk_loads_0( _S16 : u32,  _S17 : u32,  _S18 : Quat_0,  _S19 : u32,  _S20 : f32,  _S21 : bool,  _S22 : f32,  _S23 : ptr<function, vec3<f32>>,  _S24 : ptr<function, vec3<f32>>,  _S25 : ptr<function, vec3<f32>>,  _S26 : ptr<function, vec3<f32>>)
{
    var _S27 : bool;
    var _S28 : vec3<f32> = vec3<f32>(0.0f);
    (*_S23) = _S28;
    (*_S24) = _S28;
    (*_S25) = _S28;
    (*_S26) = _S28;
    var _S29 : vec4<u32> = chunks_0[_S17].load_range_0;
    var term_1 : u32 = chunks_0[_S17].load_range_0.x;
    loop
    {
        if(term_1 < (_S29.y))
        {
        }
        else
        {
            break;
        }
        g_seen_0[u32(0)] = u32(1);
        var _S30 : u32 = u32(5) * term_1;
        var _S31 : u32 = (bitcast<vec4<u32>>((loads_0[_S30]))).y;
        var replacement_0 : bool = _S31 == u32(2);
        var dir_0 : vec4<f32> = loads_0[_S30 + u32(1)];
        var arm_0 : vec4<f32> = loads_0[_S30 + u32(2)];
        var _S32 : f32;
        if(replacement_0)
        {
            _S32 = _S22;
        }
        else
        {
            _S32 = 0.0f;
        }
        var value_0 : f32 = eval_function_0(term_1, _S19, _S20, _S32);
        if(replacement_0)
        {
            var _S33 : vec3<f32> = vec3<f32>(value_0);
            (*_S25) = (*_S25) + dir_0.xyz * _S33;
            (*_S26) = (*_S26) + arm_0.xyz * _S33;
            term_1 = term_1 + u32(1);
            continue;
        }
        if(_S31 == u32(0))
        {
            _S27 = true;
        }
        else
        {
            _S27 = _S31 == u32(3);
        }
        var fw_0 : vec3<f32>;
        if(_S27)
        {
            fw_0 = dir_0.xyz * vec3<f32>(value_0);
        }
        else
        {
            fw_0 = rotate_0(_S18, dir_0.xyz) * vec3<f32>((- value_0 * dir_0.w));
        }
        var lever_0 : vec3<f32>;
        if(_S31 == u32(3))
        {
            lever_0 = arm_0.xyz - state_0[u32(4) * _S16].xyz;
        }
        else
        {
            lever_0 = arm_0.xyz;
        }
        (*_S23) = (*_S23) + fw_0;
        (*_S24) = (*_S24) + cross(rotate_0(_S18, lever_0), fw_0);
        term_1 = term_1 + u32(1);
    }
    if(_S21)
    {
        _S27 = (chunks_0[_S17].cinfo_0.z) != u32(0);
    }
    else
    {
        _S27 = false;
    }
    if(_S27)
    {
        var _S34 : vec4<u32> = chunks_0[_S17].cinfo_0;
        var g_0 : u32 = chunks_0[_S17].cinfo_0.x;
        loop
        {
            if(g_0 < (_S34.y))
            {
            }
            else
            {
                break;
            }
            var _S35 : u32 = u32(2) * g_0;
            (*_S23) = (*_S23) + scratch_0[params_0.seg_base_0 + _S35].xyz;
            (*_S24) = (*_S24) + scratch_0[params_0.seg_base_0 + _S35 + u32(1)].xyz;
            g_0 = g_0 + u32(1);
        }
    }
    return;
}

fn chunk_external_0( _S36 : u32,  _S37 : u32,  _S38 : Quat_0,  _S39 : u32,  _S40 : f32,  _S41 : bool,  _S42 : ptr<function, vec3<f32>>,  _S43 : ptr<function, vec3<f32>>)
{
    var rf_0 : vec3<f32>;
    var rm_0 : vec3<f32>;
    chunk_loads_0(_S36, _S37, _S38, _S39, _S40, _S41, 0.0f, &((*_S42)), &((*_S43)), &(rf_0), &(rm_0));
    return;
}

fn record_chunk_load_0( c_0 : u32,  f_0 : vec3<f32>,  t_1 : vec3<f32>)
{
    if((params_0.solve_mode_0) == u32(0))
    {
        return;
    }
    var _S44 : u32 = u32(2) * c_0;
    scratch_0[params_0.cload_base_0 + _S44] = vec4<f32>(f_0, 0.0f);
    scratch_0[params_0.cload_base_0 + _S44 + u32(1)] = vec4<f32>(t_1, 0.0f);
    scratch_0[params_0.cframe_base_0 + _S44] = vec4<f32>(scratch_0[params_0.cframe_base_0 + _S44].xyz + f_0, 0.0f);
    scratch_0[params_0.cframe_base_0 + _S44 + u32(1)] = vec4<f32>(scratch_0[params_0.cframe_base_0 + _S44 + u32(1)].xyz + t_1, 0.0f);
    return;
}

fn settled_chunk_load_0( c_1 : u32,  rot_1 : Quat_0,  k_2 : u32,  dt_3 : f32,  contact_0 : bool) -> f32
{
    var f_1 : vec3<f32>;
    var t_2 : vec3<f32>;
    chunk_external_0(c_1, c_1, rot_1, k_2, dt_3, contact_0, &(f_1), &(t_2));
    record_chunk_load_0(c_1, f_1, t_2);
    return length(f_1);
}

var<workgroup> g_red_a_0 : array<vec4<f32>, i32(256)>;

var<workgroup> g_red_b_0 : array<vec4<f32>, i32(256)>;

fn group_sum2_0( tid_1 : u32,  a_2 : ptr<function, vec4<f32>>,  b_1 : ptr<function, vec4<f32>>)
{
    g_red_a_0[tid_1] = (*a_2);
    g_red_b_0[tid_1] = (*b_1);
    workgroupBarrier();
    var s_1 : u32 = u32(128);
    loop
    {
        if(s_1 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_1 < s_1)
        {
            var _S45 : u32 = tid_1 + s_1;
            g_red_a_0[tid_1] = g_red_a_0[tid_1] + g_red_a_0[_S45];
            g_red_b_0[tid_1] = g_red_b_0[tid_1] + g_red_b_0[_S45];
        }
        workgroupBarrier();
        s_1 = (s_1 >> (u32(1)));
    }
    (*a_2) = g_red_a_0[i32(0)];
    (*b_1) = g_red_b_0[i32(0)];
    workgroupBarrier();
    return;
}

fn group_sum3_0( tid_2 : u32,  a_3 : ptr<function, vec3<f32>>,  b_2 : ptr<function, vec3<f32>>)
{
    var x_1 : vec4<f32> = vec4<f32>((*a_3), 0.0f);
    var y_1 : vec4<f32> = vec4<f32>((*b_2), 0.0f);
    group_sum2_0(tid_2, &(x_1), &(y_1));
    (*a_3) = x_1.xyz;
    (*b_2) = y_1.xyz;
    return;
}

fn chunk_loads_1( _S46 : u32,  _S47 : u32,  _S48 : Quat_0,  _S49 : u32,  _S50 : f32,  _S51 : bool,  _S52 : f32,  _S53 : ptr<function, vec3<f32>>,  _S54 : ptr<function, vec3<f32>>,  _S55 : ptr<function, vec3<f32>>,  _S56 : ptr<function, vec3<f32>>)
{
    var _S57 : bool;
    var _S58 : vec3<f32> = vec3<f32>(0.0f);
    (*_S53) = _S58;
    (*_S54) = _S58;
    (*_S55) = _S58;
    (*_S56) = _S58;
    var _S59 : vec4<u32> = chunks_0[_S47].load_range_0;
    var term_2 : u32 = chunks_0[_S47].load_range_0.x;
    loop
    {
        if(term_2 < (_S59.y))
        {
        }
        else
        {
            break;
        }
        g_seen_0[u32(0)] = u32(1);
        var _S60 : u32 = u32(5) * term_2;
        var _S61 : u32 = (bitcast<vec4<u32>>((loads_0[_S60]))).y;
        var replacement_1 : bool = _S61 == u32(2);
        var dir_1 : vec4<f32> = loads_0[_S60 + u32(1)];
        var arm_1 : vec4<f32> = loads_0[_S60 + u32(2)];
        var _S62 : f32;
        if(replacement_1)
        {
            _S62 = _S52;
        }
        else
        {
            _S62 = 0.0f;
        }
        var value_1 : f32 = eval_function_0(term_2, _S49, _S50, _S62);
        if(replacement_1)
        {
            var _S63 : vec3<f32> = vec3<f32>(value_1);
            (*_S55) = (*_S55) + dir_1.xyz * _S63;
            (*_S56) = (*_S56) + arm_1.xyz * _S63;
            term_2 = term_2 + u32(1);
            continue;
        }
        if(_S61 == u32(0))
        {
            _S57 = true;
        }
        else
        {
            _S57 = _S61 == u32(3);
        }
        var fw_1 : vec3<f32>;
        if(_S57)
        {
            fw_1 = dir_1.xyz * vec3<f32>(value_1);
        }
        else
        {
            fw_1 = rotate_0(_S48, dir_1.xyz) * vec3<f32>((- value_1 * dir_1.w));
        }
        var lever_1 : vec3<f32>;
        if(_S61 == u32(3))
        {
            lever_1 = arm_1.xyz - state_0[u32(4) * _S46].xyz;
        }
        else
        {
            lever_1 = arm_1.xyz;
        }
        (*_S53) = (*_S53) + fw_1;
        (*_S54) = (*_S54) + cross(rotate_0(_S48, lever_1), fw_1);
        term_2 = term_2 + u32(1);
    }
    if(_S51)
    {
        _S57 = (chunks_0[_S47].cinfo_0.z) != u32(0);
    }
    else
    {
        _S57 = false;
    }
    if(_S57)
    {
        var _S64 : vec4<u32> = chunks_0[_S47].cinfo_0;
        var g_1 : u32 = chunks_0[_S47].cinfo_0.x;
        loop
        {
            if(g_1 < (_S64.y))
            {
            }
            else
            {
                break;
            }
            var _S65 : u32 = u32(2) * g_1;
            (*_S53) = (*_S53) + scratch_0[params_0.seg_base_0 + _S65].xyz;
            (*_S54) = (*_S54) + scratch_0[params_0.seg_base_0 + _S65 + u32(1)].xyz;
            g_1 = g_1 + u32(1);
        }
    }
    return;
}

fn net_load_0( c_2 : u32,  isl_2 : Island_0,  rg_1 : Rigid_0,  k_3 : u32,  dt_4 : f32,  contact_1 : bool,  f_2 : ptr<function, vec3<f32>>,  t_3 : ptr<function, vec3<f32>>)
{
    var fl_0 : vec3<f32>;
    var tl_0 : vec3<f32>;
    var rep_f_0 : vec3<f32>;
    var rep_m_0 : vec3<f32>;
    chunk_loads_1(c_2, c_2, rg_1.rot_0, k_3, dt_4, contact_1, 0.0f, &(fl_0), &(tl_0), &(rep_f_0), &(rep_m_0));
    var fc_0 : vec3<f32> = fl_0 + params_0.gravity_0.xyz * vec3<f32>(chunks_0[c_2].center_0.w);
    var _S66 : vec3<f32> = chunks_0[c_2].center_0.xyz;
    var _S67 : vec3<f32> = isl_2.com_0.xyz;
    var r_1 : vec3<f32> = rotate_0(rg_1.rot_0, _S66 + state_0[u32(4) * c_2].xyz - _S67);
    (*f_2) = (*f_2) + fc_0;
    (*t_3) = (*t_3) + (cross(r_1, fc_0) + tl_0);
    if((chunks_0[c_2].load_range_0.y) > (chunks_0[c_2].load_range_0.x))
    {
        var fw_2 : vec3<f32> = rotate_0(rg_1.rot_0, rep_f_0);
        (*f_2) = (*f_2) + fw_2;
        (*t_3) = (*t_3) + (cross(rotate_0(rg_1.rot_0, _S66 - _S67), fw_2) + rotate_0(rg_1.rot_0, rep_m_0));
    }
    return;
}

fn inverse_rotate_0( q_2 : Quat_0,  v_1 : vec3<f32>) -> vec3<f32>
{
    var c_3 : Quat_0;
    c_3.w_0 = q_2.w_0;
    c_3.x_0 = - q_2.x_0;
    c_3.y_0 = - q_2.y_0;
    c_3.z_0 = - q_2.z_0;
    return rotate_0(c_3, v_1);
}

fn rows_mul_0( r0_0 : vec4<f32>,  r1_0 : vec4<f32>,  r2_0 : vec4<f32>,  v_2 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(dot(r0_0.xyz, v_2), dot(r1_0.xyz, v_2), dot(r2_0.xyz, v_2));
}

fn world_mul_0( q_3 : Quat_0,  r0_1 : vec4<f32>,  r1_1 : vec4<f32>,  r2_1 : vec4<f32>,  v_3 : vec3<f32>) -> vec3<f32>
{
    return rotate_0(q_3, rows_mul_0(r0_1, r1_1, r2_1, inverse_rotate_0(q_3, v_3)));
}

fn rigid_acceleration_0( isl_3 : Island_0,  rg_2 : ptr<function, Rigid_0>,  f_3 : vec3<f32>,  t_4 : vec3<f32>)
{
    var iw_w_0 : vec3<f32> = world_mul_0((*rg_2).rot_0, isl_3.inertia0_0, isl_3.inertia1_0, isl_3.inertia2_0, (*rg_2).w_1);
    (*rg_2).a_0 = f_3 / vec3<f32>(isl_3.com_0.w);
    (*rg_2).alpha_0 = world_mul_0((*rg_2).rot_0, isl_3.inv0_0, isl_3.inv1_0, isl_3.inv2_0, t_4 - cross((*rg_2).w_1, iw_w_0));
    (*rg_2).torque_0 = t_4;
    return;
}

fn comp_add1_0( sum_0 : ptr<function, f32>,  err_0 : ptr<function, f32>,  x_2 : f32)
{
    var t_5 : f32 = (*sum_0) + x_2;
    if((abs((*sum_0))) >= (abs(x_2)))
    {
        (*err_0) = (*err_0) + ((*sum_0) - t_5 + x_2);
    }
    else
    {
        (*err_0) = (*err_0) + (x_2 - t_5 + (*sum_0));
    }
    (*sum_0) = t_5;
    return;
}

fn chunk_update_0( c_4 : u32,  isl_4 : Island_0,  rg_3 : Rigid_0,  dt_5 : f32,  rml_0 : bool,  step_0 : u32,  contact_2 : bool,  work_0 : ptr<function, f32>,  work_err_0 : ptr<function, f32>)
{
    var _S68 : vec3<f32> = vec3<f32>(0.0f);
    var _S69 : u32 = index_0[c_4];
    var peak_0 : f32 = 0.0f;
    var e_0 : u32 = _S69;
    var fi_0 : vec3<f32> = _S68;
    var mi_0 : vec3<f32> = _S68;
    loop
    {
        if(e_0 < index_0[c_4 + u32(1)])
        {
        }
        else
        {
            break;
        }
        var entry_0 : u32 = index_0[e_0];
        var _S70 : u32 = u32(3) * ((entry_0 >> (u32(1))));
        var fa_0 : vec4<f32> = scratch_0[_S70];
        if(((entry_0 & (u32(1)))) == u32(0))
        {
            var mi_1 : vec3<f32> = mi_0 + scratch_0[_S70 + u32(1)].xyz;
            fi_0 = fi_0 + fa_0.xyz;
            mi_0 = mi_1;
        }
        else
        {
            var mi_2 : vec3<f32> = mi_0 + scratch_0[_S70 + u32(2)].xyz;
            fi_0 = fi_0 + (vec3<f32>(0) - fa_0.xyz);
            mi_0 = mi_2;
        }
        var _S71 : f32 = max(peak_0, fa_0.w);
        var _S72 : u32 = e_0 + u32(1);
        peak_0 = _S71;
        e_0 = _S72;
    }
    var _S73 : u32 = u32(4) * c_4;
    var u_0 : vec3<f32> = state_0[_S73].xyz;
    var _S74 : u32 = _S73 + u32(1);
    var th_0 : vec3<f32> = state_0[_S74].xyz;
    var _S75 : u32 = _S73 + u32(2);
    var v_4 : vec3<f32> = state_0[_S75].xyz;
    var _S76 : u32 = _S73 + u32(3);
    var w_2 : vec3<f32> = state_0[_S76].xyz;
    var mass_0 : f32 = chunks_0[c_4].center_0.w;
    var _S77 : vec3<f32> = chunks_0[c_4].center_0.xyz;
    var _S78 : vec3<f32> = isl_4.com_0.xyz;
    var r_world_0 : vec3<f32> = rotate_0(rg_3.rot_0, _S77 + u_0 - _S78);
    var f_load_0 : vec3<f32>;
    var t_load_0 : vec3<f32>;
    var rep_f_1 : vec3<f32>;
    var rep_m_1 : vec3<f32>;
    chunk_loads_1(c_4, c_4, rg_3.rot_0, step_0, dt_5, contact_2, dt_5, &(f_load_0), &(t_load_0), &(rep_f_1), &(rep_m_1));
    record_chunk_load_0(c_4, f_load_0, t_load_0);
    var _S79 : vec3<f32> = vec3<f32>(mass_0);
    var f_world_0 : vec3<f32> = f_load_0 + params_0.gravity_0.xyz * _S79;
    var t_world_0 : vec3<f32> = t_load_0;
    var f_world_1 : vec3<f32>;
    var t_world_1 : vec3<f32>;
    if(rml_0)
    {
        var t_world_2 : vec3<f32> = t_world_0 - (world_mul_0(rg_3.rot_0, chunks_0[c_4].inertia0_1, chunks_0[c_4].inertia1_1, chunks_0[c_4].inertia2_1, rg_3.alpha_0) + cross(rg_3.w_1, world_mul_0(rg_3.rot_0, chunks_0[c_4].inertia0_1, chunks_0[c_4].inertia1_1, chunks_0[c_4].inertia2_1, rg_3.w_1)));
        f_world_1 = f_world_0 - (rg_3.a_0 + cross(rg_3.alpha_0, r_world_0) + cross(rg_3.w_1, cross(rg_3.w_1, r_world_0))) * _S79;
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
        var i_w_0 : vec3<f32> = rows_mul_0(chunks_0[c_4].inertia0_1, chunks_0[c_4].inertia1_1, chunks_0[c_4].inertia2_1, w_2);
        var m_ext_2 : vec3<f32> = m_ext_0 - (cross(wb_0, i_w_0) + cross(w_2, rows_mul_0(chunks_0[c_4].inertia0_1, chunks_0[c_4].inertia1_1, chunks_0[c_4].inertia2_1, wb_0)) + cross(w_2, i_w_0));
        f_ext_1 = f_ext_0 - cross(wb_0, v_4) * vec3<f32>((2.0f * mass_0));
        m_ext_1 = m_ext_2;
    }
    else
    {
        f_ext_1 = f_ext_0;
        m_ext_1 = m_ext_0;
    }
    var f_4 : vec3<f32> = f_ext_1 + rep_f_1 + fi_0;
    var m_1 : vec3<f32> = m_ext_1 + rep_m_1 + mi_0;
    var support_0 : u32 = chunks_0[c_4].info_1.x;
    var _S80 : vec3<f32> = vec3<f32>(state_0[_S74].w, state_0[_S75].w, state_0[_S76].w);
    var reaction_0 : vec3<f32>;
    var u_1 : vec3<f32>;
    var th_1 : vec3<f32>;
    var v_5 : vec3<f32>;
    var w_3 : vec3<f32>;
    if(support_0 == u32(1))
    {
        reaction_0 = (vec3<f32>(0) - f_4);
        u_1 = u_0;
        th_1 = th_0;
        v_5 = _S68;
        w_3 = _S68;
    }
    else
    {
        var _S81 : vec4<f32> = chunks_0[c_4].scale_0;
        var w_4 : vec3<f32> = w_2 + rows_mul_0(chunks_0[c_4].inv0_1, chunks_0[c_4].inv1_1, chunks_0[c_4].inv2_1, m_1) * vec3<f32>((dt_5 * chunks_0[c_4].scale_0.z));
        var _S82 : vec3<f32> = vec3<f32>(dt_5);
        var th_2 : vec3<f32> = th_0 + w_4 * _S82;
        if(support_0 == u32(2))
        {
            reaction_0 = (vec3<f32>(0) - f_4);
            u_1 = u_0;
            th_1 = _S68;
        }
        else
        {
            var v_6 : vec3<f32> = v_4 + f_4 * vec3<f32>((dt_5 * _S81.y));
            var u_2 : vec3<f32> = u_0 + v_6 * _S82;
            reaction_0 = _S80;
            u_1 = u_2;
            th_1 = v_6;
        }
        var _S83 : vec3<f32> = th_1;
        th_1 = th_2;
        v_5 = _S83;
        w_3 = w_4;
    }
    state_0[_S73] = vec4<f32>(u_1, peak_0);
    state_0[_S74] = vec4<f32>(th_1, reaction_0.x);
    state_0[_S75] = vec4<f32>(v_5, reaction_0.y);
    state_0[_S76] = vec4<f32>(w_3, reaction_0.z);
    comp_add1_0(&((*work_0)), &((*work_err_0)), (dot(f_load_0, rg_3.vel_0 + rg_3.vel_err_0 + cross(rg_3.w_1, rotate_0(rg_3.rot_0, _S77 + u_1 - _S78)) + rotate_0(rg_3.rot_0, v_5)) + dot(t_load_0, rg_3.w_1 + rotate_0(rg_3.rot_0, w_3))) * dt_5);
    return;
}

fn comp_add_0( sum_1 : ptr<function, vec3<f32>>,  err_1 : ptr<function, vec3<f32>>,  x_3 : vec3<f32>)
{
    var t_6 : vec3<f32> = (*sum_1) + x_3;
    var _S84 : vec3<f32> = abs(x_3);
    (*err_1) = (*err_1) + (select(x_3, (*sum_1), (abs((*sum_1))) >= _S84) - t_6 + select((*sum_1), x_3, (abs((*sum_1))) >= _S84));
    (*sum_1) = t_6;
    return;
}

fn turn_difference_0( omega_0 : vec3<f32>,  dt_6 : f32,  v_7 : vec3<f32>) -> vec3<f32>
{
    var _S85 : f32 = length(omega_0);
    var angle_0 : f32 = _S85 * dt_6;
    if(angle_0 < 1.00000000317107685e-30f)
    {
        return vec3<f32>(0.0f);
    }
    var a_4 : vec3<f32> = omega_0 / vec3<f32>(_S85);
    var s_2 : f32 = sin(0.5f * angle_0);
    var av_0 : vec3<f32> = cross(a_4, v_7);
    return av_0 * vec3<f32>(sin(angle_0)) + cross(a_4, av_0) * vec3<f32>((2.0f * s_2 * s_2));
}

fn turn_minus_one_0( axis_0 : vec3<f32>,  angle_1 : f32) -> vec4<f32>
{
    var s_3 : f32 = sin(0.25f * angle_1);
    return vec4<f32>(axis_0 * vec3<f32>(sin(0.5f * angle_1)), -2.0f * s_3 * s_3);
}

fn quat_mul_0( a_5 : Quat_0,  o_0 : Quat_0) -> Quat_0
{
    var r_2 : Quat_0;
    r_2.w_0 = a_5.w_0 * o_0.w_0 - a_5.x_0 * o_0.x_0 - a_5.y_0 * o_0.y_0 - a_5.z_0 * o_0.z_0;
    r_2.x_0 = a_5.w_0 * o_0.x_0 + a_5.x_0 * o_0.w_0 + a_5.y_0 * o_0.z_0 - a_5.z_0 * o_0.y_0;
    r_2.y_0 = a_5.w_0 * o_0.y_0 - a_5.x_0 * o_0.z_0 + a_5.y_0 * o_0.w_0 + a_5.z_0 * o_0.x_0;
    r_2.z_0 = a_5.w_0 * o_0.z_0 + a_5.x_0 * o_0.y_0 - a_5.y_0 * o_0.x_0 + a_5.z_0 * o_0.w_0;
    return r_2;
}

fn quat_vec_0( q_4 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_4.x_0, q_4.y_0, q_4.z_0, q_4.w_0);
}

fn comp_add4_0( sum_2 : ptr<function, vec4<f32>>,  err_2 : ptr<function, vec4<f32>>,  x_4 : vec4<f32>)
{
    var t_7 : vec4<f32> = (*sum_2) + x_4;
    var _S86 : vec4<f32> = abs(x_4);
    (*err_2) = (*err_2) + (select(x_4, (*sum_2), (abs((*sum_2))) >= _S86) - t_7 + select((*sum_2), x_4, (abs((*sum_2))) >= _S86));
    (*sum_2) = t_7;
    return;
}

fn quat_accumulate_0( hi_0 : ptr<function, Quat_0>,  lo_0 : ptr<function, vec4<f32>>,  x_5 : vec4<f32>)
{
    var sum_3 : vec4<f32> = quat_vec_0((*hi_0));
    comp_add4_0(&(sum_3), &((*lo_0)), x_5);
    var t_8 : vec4<f32> = sum_3 + (*lo_0);
    (*lo_0) = (*lo_0) - (t_8 - sum_3);
    (*hi_0) = quat_of_0(t_8);
    return;
}

fn turn_left_0( hi_1 : ptr<function, Quat_0>,  lo_1 : ptr<function, vec4<f32>>,  omega_1 : vec3<f32>,  dt_7 : f32)
{
    var _S87 : f32 = length(omega_1);
    var angle_2 : f32 = _S87 * dt_7;
    if(angle_2 < 1.00000000317107685e-30f)
    {
        return;
    }
    var d_0 : vec4<f32> = turn_minus_one_0(omega_1 / vec3<f32>(_S87), angle_2);
    var dq_0 : Quat_0;
    dq_0.x_0 = d_0.x;
    dq_0.y_0 = d_0.y;
    dq_0.z_0 = d_0.z;
    dq_0.w_0 = d_0.w;
    quat_accumulate_0(&((*hi_1)), &((*lo_1)), quat_vec_0(quat_mul_0(dq_0, (*hi_1))));
    return;
}

fn integrate_rigid_0( isl_5 : Island_0,  rg_4 : ptr<function, Rigid_0>,  dt_8 : f32)
{
    var _S88 : vec3<f32> = vec3<f32>(dt_8);
    var _S89 : vec3<f32> = (*rg_4).torque_0 * _S88;
    var _S90 : vec3<f32> = (*rg_4).l_0;
    var _S91 : vec3<f32> = (*rg_4).l_err_0;
    comp_add_0(&(_S90), &(_S91), _S89);
    (*rg_4).l_0 = _S90;
    (*rg_4).l_err_0 = _S91;
    var l_1 : vec3<f32> = _S90 + _S91;
    var _S92 : vec3<f32> = (*rg_4).a_0 * _S88;
    var _S93 : vec3<f32> = (*rg_4).vel_0;
    var _S94 : vec3<f32> = (*rg_4).vel_err_0;
    comp_add_0(&(_S93), &(_S94), _S92);
    (*rg_4).vel_0 = _S93;
    (*rg_4).vel_err_0 = _S94;
    var w_mid_0 : vec3<f32> = world_mul_0((*rg_4).rot_0, isl_5.inv0_0, isl_5.inv1_0, isl_5.inv2_0, l_1);
    var delta_0 : vec3<f32> = (_S93 + _S94) * _S88 - turn_difference_0(w_mid_0, dt_8, rotate_0((*rg_4).rot_0, isl_5.com_0.xyz));
    var _S95 : vec3<f32> = (*rg_4).pos_0;
    var _S96 : vec3<f32> = (*rg_4).pos_err_0;
    comp_add_0(&(_S95), &(_S96), delta_0);
    (*rg_4).pos_0 = _S95;
    (*rg_4).pos_err_0 = _S96;
    var _S97 : Quat_0 = (*rg_4).rot_0;
    var _S98 : vec4<f32> = (*rg_4).rot_err_0;
    turn_left_0(&(_S97), &(_S98), w_mid_0, dt_8);
    (*rg_4).rot_0 = _S97;
    (*rg_4).rot_err_0 = _S98;
    (*rg_4).w_1 = world_mul_0(_S97, isl_5.inv0_0, isl_5.inv1_0, isl_5.inv2_0, l_1);
    return;
}

fn drift_moments_0( c_5 : u32,  tu_0 : ptr<function, vec3<f32>>,  pv_0 : ptr<function, vec3<f32>>)
{
    var _S99 : u32 = u32(4) * c_5;
    var _S100 : vec3<f32> = vec3<f32>((chunks_0[c_5].center_0.w * chunks_0[c_5].scale_0.x));
    (*tu_0) = (*tu_0) + state_0[_S99].xyz * _S100;
    (*pv_0) = (*pv_0) + state_0[_S99 + u32(2)].xyz * _S100;
    return;
}

fn drift_angular_0( c_6 : u32,  wcom_1 : vec3<f32>,  tr_0 : vec3<f32>,  dv_0 : vec3<f32>,  lu_0 : ptr<function, vec3<f32>>,  lv_0 : ptr<function, vec3<f32>>)
{
    var r_3 : vec3<f32> = chunks_0[c_6].center_0.xyz - wcom_1;
    var _S101 : u32 = u32(4) * c_6;
    var _S102 : vec3<f32> = vec3<f32>(chunks_0[c_6].center_0.w);
    var _S103 : vec4<f32> = chunks_0[c_6].inertia0_1;
    var _S104 : vec4<f32> = chunks_0[c_6].inertia1_1;
    var _S105 : vec4<f32> = chunks_0[c_6].inertia2_1;
    var _S106 : vec3<f32> = vec3<f32>(chunks_0[c_6].scale_0.x);
    (*lu_0) = (*lu_0) + (cross(r_3, state_0[_S101].xyz - tr_0) * _S102 + rows_mul_0(chunks_0[c_6].inertia0_1, chunks_0[c_6].inertia1_1, chunks_0[c_6].inertia2_1, state_0[_S101 + u32(1)].xyz)) * _S106;
    (*lv_0) = (*lv_0) + (cross(r_3, state_0[_S101 + u32(2)].xyz - dv_0) * _S102 + rows_mul_0(_S103, _S104, _S105, state_0[_S101 + u32(3)].xyz)) * _S106;
    return;
}

fn drift_apply_0( c_7 : u32,  wcom_2 : vec3<f32>,  tr_1 : vec3<f32>,  phi_0 : vec3<f32>,  dv_1 : vec3<f32>,  dw_0 : vec3<f32>)
{
    var r_4 : vec3<f32> = chunks_0[c_7].center_0.xyz - wcom_2;
    var _S107 : u32 = u32(4) * c_7;
    state_0[_S107] = vec4<f32>(state_0[_S107].xyz - (tr_1 + cross(phi_0, r_4)), state_0[_S107].w);
    var _S108 : u32 = _S107 + u32(1);
    state_0[_S108] = vec4<f32>(state_0[_S108].xyz - phi_0, state_0[_S108].w);
    var _S109 : u32 = _S107 + u32(2);
    state_0[_S109] = vec4<f32>(state_0[_S109].xyz - (dv_1 + cross(dw_0, r_4)), state_0[_S109].w);
    var _S110 : u32 = _S107 + u32(3);
    state_0[_S110] = vec4<f32>(state_0[_S110].xyz - dw_0, state_0[_S110].w);
    return;
}

fn turn_right_0( hi_2 : ptr<function, Quat_0>,  lo_2 : ptr<function, vec4<f32>>,  phi_1 : vec3<f32>)
{
    var angle_3 : f32 = length(phi_1);
    if(angle_3 < 1.00000000317107685e-30f)
    {
        return;
    }
    var d_1 : vec4<f32> = turn_minus_one_0(phi_1 / vec3<f32>(angle_3), angle_3);
    var dq_1 : Quat_0;
    dq_1.x_0 = d_1.x;
    dq_1.y_0 = d_1.y;
    dq_1.z_0 = d_1.z;
    dq_1.w_0 = d_1.w;
    quat_accumulate_0(&((*hi_2)), &((*lo_2)), quat_vec_0(quat_mul_0((*hi_2), dq_1)));
    return;
}

fn drift_rigid_0( isl_6 : Island_0,  rg_5 : ptr<function, Rigid_0>,  tr_2 : vec3<f32>,  phi_2 : vec3<f32>,  dv_2 : vec3<f32>,  dw_1 : vec3<f32>)
{
    var wcom_3 : vec3<f32> = isl_6.wcom_0.xyz;
    var rot_2 : Quat_0 = (*rg_5).rot_0;
    var _S111 : vec3<f32> = rotate_0((*rg_5).rot_0, tr_2 - turn_difference_0(phi_2, 1.0f, wcom_3));
    var _S112 : vec3<f32> = (*rg_5).pos_0;
    var _S113 : vec3<f32> = (*rg_5).pos_err_0;
    comp_add_0(&(_S112), &(_S113), _S111);
    (*rg_5).pos_0 = _S112;
    (*rg_5).pos_err_0 = _S113;
    var b_3 : vec3<f32> = inverse_rotate_0((*rg_5).rot_0, (*rg_5).w_1);
    var dl_turn_0 : vec3<f32> = rotate_0((*rg_5).rot_0, cross(phi_2, rows_mul_0(isl_6.inertia0_0, isl_6.inertia1_0, isl_6.inertia2_0, b_3)) - rows_mul_0(isl_6.inertia0_0, isl_6.inertia1_0, isl_6.inertia2_0, cross(phi_2, b_3)));
    var _S114 : Quat_0 = (*rg_5).rot_0;
    var _S115 : vec4<f32> = (*rg_5).rot_err_0;
    turn_right_0(&(_S114), &(_S115), phi_2);
    (*rg_5).rot_0 = _S114;
    (*rg_5).rot_err_0 = _S115;
    var _S116 : vec3<f32> = rotate_0(rot_2, dv_2 + cross(dw_1, isl_6.com_0.xyz - wcom_3));
    var _S117 : vec3<f32> = (*rg_5).vel_0;
    var _S118 : vec3<f32> = (*rg_5).vel_err_0;
    comp_add_0(&(_S117), &(_S118), _S116);
    (*rg_5).vel_0 = _S117;
    (*rg_5).vel_err_0 = _S118;
    var _S119 : vec3<f32> = rotate_0(rot_2, dw_1);
    (*rg_5).w_1 = (*rg_5).w_1 + _S119;
    var _S120 : vec3<f32> = dl_turn_0 + world_mul_0(_S114, isl_6.inertia0_0, isl_6.inertia1_0, isl_6.inertia2_0, _S119);
    var _S121 : vec3<f32> = (*rg_5).l_0;
    var _S122 : vec3<f32> = (*rg_5).l_err_0;
    comp_add_0(&(_S121), &(_S122), _S120);
    (*rg_5).l_0 = _S121;
    (*rg_5).l_err_0 = _S122;
    return;
}

fn write_probe_0( slot_0 : u32,  k_4 : u32,  value_2 : f32)
{
    var at_0 : u32 = params_0.probe_base_0 * u32(4) + slot_0 * params_0.probe_stride_0 + k_4;
    var v_8 : vec4<f32> = scratch_0[at_0 / u32(4)];
    v_8[at_0 % u32(4)] = value_2;
    scratch_0[at_0 / u32(4)] = v_8;
    return;
}

fn record_probes_0( isl_7 : Island_0,  rg_6 : Rigid_0,  k_5 : u32)
{
    var at_1 : u32 = isl_7.probes_0.x;
    loop
    {
        if(at_1 < (isl_7.probes_0.y))
        {
        }
        else
        {
            break;
        }
        var info_3 : vec4<u32> = (bitcast<vec4<u32>>((loads_0[at_1])));
        var a_6 : vec4<f32> = loads_0[at_1 + u32(1)];
        var b_4 : vec4<f32> = loads_0[at_1 + u32(2)];
        var c4_0 : vec4<f32> = loads_0[at_1 + u32(3)];
        var kind_1 : u32 = info_3.x;
        var i_1 : u32 = info_3.y;
        var value_3 : f32;
        if(kind_1 == u32(0))
        {
            value_3 = dot(rg_6.pos_0 - b_4.xyz + rotate_0(rg_6.rot_0, chunks_0[i_1].center_0.xyz) + (rg_6.pos_err_0 - c4_0.xyz + rotate_0(rg_6.rot_0, state_0[u32(4) * i_1].xyz + vec3<f32>(a_6.w, b_4.w, c4_0.w))), a_6.xyz);
        }
        else
        {
            if(kind_1 == u32(1))
            {
                var _S123 : u32 = u32(4) * i_1;
                value_3 = dot(rg_6.vel_0 + rg_6.vel_err_0 + cross(rg_6.w_1, rotate_0(rg_6.rot_0, chunks_0[i_1].center_0.xyz + state_0[_S123].xyz - isl_7.com_0.xyz)) + rotate_0(rg_6.rot_0, state_0[_S123 + u32(2)].xyz), a_6.xyz);
            }
            else
            {
                if(kind_1 == u32(2))
                {
                    var _S124 : u32 = u32(3) * i_1;
                    var f_5 : vec3<f32> = scratch_0[_S124].xyz;
                    var _S125 : bool = (info_3.z) == u32(0);
                    var mc_0 : vec3<f32>;
                    if(_S125)
                    {
                        mc_0 = scratch_0[_S124 + u32(1)].xyz;
                    }
                    else
                    {
                        mc_0 = scratch_0[_S124 + u32(2)].xyz;
                    }
                    var fc_1 : vec3<f32>;
                    if(_S125)
                    {
                        fc_1 = f_5;
                    }
                    else
                    {
                        fc_1 = (vec3<f32>(0) - f_5);
                    }
                    value_3 = dot(fc_1, a_6.xyz) + dot(mc_0, b_4.xyz);
                }
                else
                {
                    var _S126 : u32 = u32(4) * i_1;
                    value_3 = dot(rotate_0(rg_6.rot_0, vec3<f32>(state_0[_S126 + u32(1)].w, state_0[_S126 + u32(2)].w, state_0[_S126 + u32(3)].w)), a_6.xyz);
                }
            }
        }
        write_probe_0(info_3.w, k_5, value_3);
        at_1 = at_1 + u32(4);
    }
    return;
}

var<workgroup> g_label_narrow_0 : array<u32, i32(512)>;

var<workgroup> g_changed_narrow_0 : u32;

fn connected_0( st_0 : ptr<function, JointState_std430_0>,  has_rebar_0 : bool) -> bool
{
    var _S127 : bool;
    if(((*st_0).damage_0) < 1.0f)
    {
        _S127 = true;
    }
    else
    {
        if(has_rebar_0)
        {
            _S127 = ((*st_0).rebar_broken_0) == 0.0f;
        }
        else
        {
            _S127 = false;
        }
    }
    return _S127;
}

fn island_connected_narrow_0( tid_3 : u32,  isl_8 : Island_0) -> bool
{
    var _S128 : u32 = isl_8.range_0.x;
    var n_0 : u32 = isl_8.range_0.y - _S128;
    var converged_0 : bool;
    if(n_0 > u32(512))
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
    var c_8 : u32 = tid_3;
    loop
    {
        if(c_8 < n_0)
        {
        }
        else
        {
            break;
        }
        g_label_narrow_0[c_8] = c_8;
        c_8 = c_8 + u32(256);
    }
    workgroupBarrier();
    converged_0 = false;
    var pass_0 : u32 = u32(0);
    loop
    {
        if(pass_0 <= n_0)
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
        c_8 = tid_3;
        loop
        {
            if(c_8 < n_0)
            {
            }
            else
            {
                break;
            }
            var gc_0 : u32 = _S128 + c_8;
            var _S129 : u32 = index_0[gc_0];
            var l_2 : u32 = g_label_narrow_0[c_8];
            var e_1 : u32 = _S129;
            loop
            {
                if(e_1 < index_0[gc_0 + u32(1)])
                {
                }
                else
                {
                    break;
                }
                var entry_1 : u32 = index_0[e_1];
                var bond_0 : u32 = (entry_1 >> (u32(1)));
                var _S130 : JointState_std430_0 = bond_dyn_0[bond_0].js_0;
                var _S131 : bool = connected_0(&(_S130), (bonds_0[bond_0].law_0.stiff1_0.w) != 0.0f);
                if(!_S131)
                {
                    e_1 = e_1 + u32(1);
                    continue;
                }
                var other_0 : u32;
                if(((entry_1 & (u32(1)))) == u32(0))
                {
                    other_0 = bonds_0[bond_0].law_0.ids_0.z;
                }
                else
                {
                    other_0 = bonds_0[bond_0].law_0.ids_0.y;
                }
                l_2 = min(l_2, g_label_narrow_0[other_0 - _S128]);
                e_1 = e_1 + u32(1);
            }
            var _S132 : u32 = min(l_2, g_label_narrow_0[l_2]);
            if(_S132 < (g_label_narrow_0[c_8]))
            {
                g_label_narrow_0[c_8] = _S132;
                g_changed_narrow_0 = u32(1);
            }
            c_8 = c_8 + u32(256);
        }
        workgroupBarrier();
        var converged_1 : bool = g_changed_narrow_0 == u32(0);
        workgroupBarrier();
        if(converged_1)
        {
            converged_0 = converged_1;
            break;
        }
        var _S133 : u32 = pass_0 + u32(1);
        converged_0 = converged_1;
        pass_0 = _S133;
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
    c_8 = tid_3;
    loop
    {
        if(c_8 < n_0)
        {
        }
        else
        {
            break;
        }
        if((g_label_narrow_0[c_8]) != u32(0))
        {
            g_changed_narrow_0 = u32(1);
        }
        c_8 = c_8 + u32(256);
    }
    workgroupBarrier();
    return g_changed_narrow_0 == u32(0);
}

fn contact_split_at_0( at_2 : u32)
{
    var previous_0 : u32 = islands_0[params_0.halt_index_0].info_0.y;
    var _S134 : u32;
    if(previous_0 == u32(0))
    {
        _S134 = at_2;
    }
    else
    {
        _S134 = min(previous_0, at_2);
    }
    islands_0[params_0.halt_index_0].info_0[i32(1)] = _S134;
    return;
}

fn comp_add1_1( sum_4 : ptr<function, f32>,  err_3 : ptr<function, f32>,  x_6 : f32)
{
    var t_9 : f32 = (*sum_4) + x_6;
    if((abs((*sum_4))) >= (abs(x_6)))
    {
        (*err_3) = (*err_3) + ((*sum_4) - t_9 + x_6);
    }
    else
    {
        (*err_3) = (*err_3) + (x_6 - t_9 + (*sum_4));
    }
    (*sum_4) = t_9;
    return;
}

fn seen_bits_0() -> u32
{
    var _S135 : u32;
    if((g_seen_0[u32(0)]) != u32(0))
    {
        _S135 = u32(32768);
    }
    else
    {
        _S135 = u32(0);
    }
    var _S136 : u32;
    if((g_seen_0[u32(1)]) != u32(0))
    {
        _S136 = u32(65536);
    }
    else
    {
        _S136 = u32(0);
    }
    var _S137 : u32 = (_S135 | (_S136));
    if((g_seen_0[u32(2)]) != u32(0))
    {
        _S135 = u32(131072);
    }
    else
    {
        _S135 = u32(0);
    }
    var _S138 : u32 = (_S137 | (_S135));
    if((g_seen_0[u32(3)]) != u32(0))
    {
        _S135 = u32(32);
    }
    else
    {
        _S135 = u32(0);
    }
    var _S139 : u32 = (_S138 | (_S135));
    if((g_seen_0[u32(4)]) != u32(0))
    {
        _S135 = u32(64);
    }
    else
    {
        _S135 = u32(0);
    }
    return (_S139 | (_S135));
}

fn island_frame_at_0( gi_0 : u32,  tid_4 : u32)
{
    var woke_0 : bool;
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
    var _S140 : bool = !((((isl_9.info_0.x) & (u32(1)))) != u32(0));
    var _S141 : bool;
    if(_S140)
    {
        _S141 = !driven_0;
    }
    else
    {
        _S141 = false;
    }
    var contact_island_0 : bool = (((isl_9.info_0.x) & (u32(4)))) != u32(0);
    var _S142 : bool = (((isl_9.info_0.x) & (u32(16)))) != u32(0);
    seen_clear_0(tid_4);
    var _S143 : bool = tid_4 == u32(0);
    var settled_0 : bool;
    var cov_0 : u32;
    if(_S143)
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
    var _S144 : f32 = params_0.dt_0;
    var _S145 : bool = (params_0.rigid_motion_loads_0) != u32(0);
    var rg_7 : Rigid_0 = rigid_of_0(isl_9);
    var work_1 : f32 = 0.0f;
    var work_err_1 : f32 = 0.0f;
    settled_0 = _S142;
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
        var k_6 : u32 = isl_9.info_0.w + done_1 + u32(1) - u32(1) - params_0.step_start_0;
        var quiet_0 : bool = (((isl_9.info_0.x) & (u32(32)))) != u32(0);
        var _S146 : bool;
        if(settled_0)
        {
            _S146 = !quiet_0;
        }
        else
        {
            _S146 = false;
        }
        var _S147 : bool;
        var explicit_step_0 : bool;
        var c_9 : u32;
        if(_S146)
        {
            var _S148 : vec3<f32> = vec3<f32>(0.0f);
            var norm_0 : vec3<f32> = _S148;
            var unused0_0 : vec3<f32> = _S148;
            c_9 = isl_9.range_0.x + tid_4;
            loop
            {
                if(c_9 < (isl_9.range_0.y))
                {
                }
                else
                {
                    break;
                }
                var _S149 : f32 = settled_chunk_load_0(c_9, rg_7.rot_0, k_6, _S144, contact_island_0);
                norm_0[i32(0)] = norm_0[i32(0)] + _S149;
                c_9 = c_9 + u32(256);
            }
            group_sum3_0(tid_4, &(norm_0), &(unused0_0));
            if((params_0.solve_mode_0) == u32(1))
            {
                _S147 = (abs(norm_0.x - isl_9.energy_0.z)) > (isl_9.energy_0.w);
            }
            else
            {
                _S147 = false;
            }
            if(_S147)
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
        if(_S140)
        {
            _S147 = !quiet_0;
        }
        else
        {
            _S147 = false;
        }
        if(_S147)
        {
            var _S150 : vec3<f32> = vec3<f32>(0.0f);
            var f_6 : vec3<f32> = _S150;
            var t_10 : vec3<f32> = _S150;
            c_9 = isl_9.range_0.x + tid_4;
            loop
            {
                if(c_9 < (isl_9.range_0.y))
                {
                }
                else
                {
                    break;
                }
                net_load_0(c_9, isl_9, rg_7, k_6, _S144, contact_island_0, &(f_6), &(t_10));
                c_9 = c_9 + u32(256);
            }
            group_sum3_0(tid_4, &(f_6), &(t_10));
            rigid_acceleration_0(isl_9, &(rg_7), f_6, t_10);
        }
        if(!settled_0)
        {
            explicit_step_0 = !quiet_0;
        }
        else
        {
            explicit_step_0 = false;
        }
        var _S151 : bool;
        if(explicit_step_0)
        {
            if(_S143)
            {
                _S151 = (((islands_0[gi_0].info_0.z) & (u32(2)))) != u32(0);
            }
            else
            {
                _S151 = false;
            }
            if(_S151)
            {
                g_halt_0 = u32(1);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
            c_9 = isl_9.range_0.x + tid_4;
            loop
            {
                if(c_9 < (isl_9.range_0.y))
                {
                }
                else
                {
                    break;
                }
                chunk_update_0(c_9, isl_9, rg_7, _S144, _S145, k_6, contact_island_0, &(work_1), &(work_err_1));
                c_9 = c_9 + u32(256);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
        }
        if(_S141)
        {
            _S151 = !quiet_0;
        }
        else
        {
            _S151 = false;
        }
        if(_S151)
        {
            integrate_rigid_0(isl_9, &(rg_7), _S144);
        }
        var _S152 : bool;
        if(_S140)
        {
            _S152 = explicit_step_0;
        }
        else
        {
            _S152 = false;
        }
        if(_S152)
        {
            var _S153 : vec3<f32> = isl_9.wcom_0.xyz;
            var _S154 : vec3<f32> = vec3<f32>(0.0f);
            var tu_1 : vec3<f32> = _S154;
            var pv_1 : vec3<f32> = _S154;
            c_9 = isl_9.range_0.x + tid_4;
            loop
            {
                if(c_9 < (isl_9.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_moments_0(c_9, &(tu_1), &(pv_1));
                c_9 = c_9 + u32(256);
            }
            group_sum3_0(tid_4, &(tu_1), &(pv_1));
            var tr_3 : vec3<f32> = tu_1 / vec3<f32>(isl_9.wcom_0.w);
            var dv_3 : vec3<f32> = pv_1 / vec3<f32>(isl_9.wcom_0.w);
            var lu_1 : vec3<f32> = _S154;
            var lv_1 : vec3<f32> = _S154;
            var c_10 : u32 = isl_9.range_0.x + tid_4;
            loop
            {
                if(c_10 < (isl_9.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_angular_0(c_10, _S153, tr_3, dv_3, &(lu_1), &(lv_1));
                c_10 = c_10 + u32(256);
            }
            group_sum3_0(tid_4, &(lu_1), &(lv_1));
            var phi_3 : vec3<f32> = rows_mul_0(isl_9.winv0_0, isl_9.winv1_0, isl_9.winv2_0, lu_1);
            var dw_2 : vec3<f32> = rows_mul_0(isl_9.winv0_0, isl_9.winv1_0, isl_9.winv2_0, lv_1);
            var c_11 : u32 = isl_9.range_0.x + tid_4;
            loop
            {
                if(c_11 < (isl_9.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_11, _S153, tr_3, phi_3, dv_3, dw_2);
                c_11 = c_11 + u32(256);
            }
            if(!driven_0)
            {
                drift_rigid_0(isl_9, &(rg_7), tr_3, phi_3, dv_3, dw_2);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
        }
        var _S155 : bool;
        if(_S143)
        {
            _S155 = (isl_9.probes_0.y) > (isl_9.probes_0.x);
        }
        else
        {
            _S155 = false;
        }
        if(_S155)
        {
            record_probes_0(isl_9, rg_7, k_6);
        }
        var _S156 : u32 = done_1 + u32(1);
        var _S157 : bool;
        if(g_halt_0 != u32(0))
        {
            var _S158 : bool = island_connected_narrow_0(tid_4, isl_9);
            _S157 = _S158;
        }
        else
        {
            _S157 = false;
        }
        var resumed_1 : bool;
        if(_S157)
        {
            if(_S143)
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
            done_1 = _S156;
            resumed_0 = resumed_1;
            break;
        }
        done_1 = _S156;
        woke_1 = woke_0;
        resumed_0 = resumed_1;
    }
    var wsum_0 : vec3<f32> = vec3<f32>(work_1, work_err_1, 0.0f);
    var unused_0 : vec3<f32> = vec3<f32>(0.0f);
    group_sum3_0(tid_4, &(wsum_0), &(unused_0));
    if(_S143)
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
            _S141 = contact_island_0;
        }
        else
        {
            _S141 = false;
        }
        if(_S141)
        {
            contact_split_at_0(isl_9.info_0.w + done_1);
        }
        var _S159 : f32 = wsum_0.x;
        var _S160 : f32 = isl_9.energy_0[i32(0)];
        var _S161 : f32 = isl_9.energy_0[i32(1)];
        comp_add1_1(&(_S160), &(_S161), _S159);
        isl_9.energy_0[i32(0)] = _S160;
        isl_9.energy_0[i32(1)] = _S161 + wsum_0.y;
        isl_9.info_0[i32(3)] = isl_9.info_0[i32(3)] + done_1;
        if((((isl_9.info_0.x) & (u32(8)))) == u32(0))
        {
            isl_9.info_0[i32(2)] = ((isl_9.info_0[i32(2)]) & (u32(4294967293)));
        }
        if(g_halt_0 != u32(0))
        {
            isl_9.info_0[i32(2)] = ((isl_9.info_0[i32(2)]) | (u32(1)));
        }
        var _S162 : u32 = seen_bits_0();
        if(g_halt_0 != u32(0))
        {
            cov_0 = u32(128);
        }
        else
        {
            cov_0 = u32(0);
        }
        var _S163 : u32 = (_S162 | (cov_0));
        if(woke_0)
        {
            cov_0 = u32(2048);
        }
        else
        {
            cov_0 = u32(0);
        }
        var _S164 : u32 = (_S163 | (cov_0));
        if(resumed_0)
        {
            cov_0 = u32(2097280);
        }
        else
        {
            cov_0 = u32(0);
        }
        var cov_1 : u32 = (_S164 | (cov_0));
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
                _S141 = !quiet_1;
            }
            else
            {
                _S141 = false;
            }
            if(_S141)
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
    return;
}

fn to_local_0( _S165 : u32,  _S166 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(dot(_S166, bonds_0[_S165].t1_0.xyz), dot(_S166, bonds_0[_S165].t2_0.xyz), dot(_S166, bonds_0[_S165].normal_0.xyz));
}

fn is_damaged_0( st_1 : ptr<function, JointState_std430_0>) -> bool
{
    var _S167 : bool;
    if(((*st_1).damage_0) > 0.0f)
    {
        _S167 = true;
    }
    else
    {
        _S167 = ((*st_1).crush_1) > 0.0f;
    }
    return _S167;
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
     dissipated_1 : f32,
     overshoot_0 : f32,
     stored_0 : f32,
     disconnected_0 : bool,
     fatigue_before_0 : f32,
     measures_0 : Measures_0,
};

fn connected_1( st_2 : ptr<function, JointState_std430_0>,  has_rebar_1 : bool) -> bool
{
    var _S168 : bool;
    if(((*st_2).damage_0) < 1.0f)
    {
        _S168 = true;
    }
    else
    {
        if(has_rebar_1)
        {
            _S168 = ((*st_2).rebar_broken_0) == 0.0f;
        }
        else
        {
            _S168 = false;
        }
    }
    return _S168;
}

fn fdiv_0( a_7 : f32,  b_5 : f32) -> f32
{
    return a_7 / b_5;
}

fn fsqrt_0( a_8 : f32) -> f32
{
    return sqrt(a_8);
}

fn stress_measures_0( b_6 : ptr<function, JointBond_std430_0>,  q_lin_0 : vec3<f32>,  q_ang_0 : vec3<f32>) -> Measures_0
{
    var area_0 : f32 = (*b_6).geom0_0.x;
    var _S169 : f32 = q_lin_0.z;
    var axial_0 : f32 = fdiv_0(_S169, area_0);
    var bending_0 : f32 = fdiv_0(abs(q_ang_0.x), (*b_6).geom1_0.x) + fdiv_0(abs(q_ang_0.y), (*b_6).geom1_0.y);
    var _S170 : f32 = q_lin_0.x;
    var _S171 : f32 = q_lin_0.y;
    var shear_1 : f32 = fdiv_0(fsqrt_0(_S170 * _S170 + _S171 * _S171), area_0) + fdiv_0(abs(q_ang_0.z), (*b_6).geom0_0.w);
    var m_2 : Measures_0;
    m_2.tension_0 = axial_0 + bending_0;
    m_2.shear_0 = shear_1;
    var _S172 : f32 = - axial_0;
    m_2.normal_compression_0 = max(_S172, 0.0f);
    m_2.compression_0 = _S172 + bending_0;
    m_2.compressive_force_0 = max(- _S169, 0.0f);
    return m_2;
}

fn expm1_accurate_0( x_7 : f32) -> f32
{
    if((abs(x_7)) < 0.00100000004749745f)
    {
        return x_7 * (1.0f + x_7 * (0.5f + x_7 * 0.1666666716337204f));
    }
    return exp(x_7) - 1.0f;
}

fn fpow_0( a_9 : f32,  b_7 : f32) -> f32
{
    return pow(a_9, b_7);
}

fn dif_factor_0( mat_1 : ptr<function, JointMaterial_std140_0>,  strain_rate_1 : f32) -> f32
{
    var r_5 : f32 = abs(strain_rate_1);
    var _S173 : vec4<f32> = (*mat_1).dif_0;
    var ref_0 : f32 = (*mat_1).dif_0.x;
    if(r_5 <= ref_0)
    {
        return 1.0f;
    }
    var _S174 : f32 = _S173.z;
    var f_7 : f32;
    if(r_5 <= _S174)
    {
        f_7 = fpow_0(fdiv_0(r_5, ref_0), _S173.y);
    }
    else
    {
        f_7 = fpow_0(fdiv_0(_S174, ref_0), _S173.y) * fpow_0(fdiv_0(r_5, _S174), _S173.w);
    }
    return clamp(f_7, 1.0f, (*mat_1).misc_0.x);
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

fn failure_indices_0( mat_3 : ptr<function, JointMaterial_std140_0>,  b_8 : ptr<function, JointBond_std430_0>,  m_3 : Measures_0,  multiplier_0 : f32) -> vec4<f32>
{
    var fc_2 : f32 = (*mat_3).strength_0.y * multiplier_0;
    var _S175 : f32 = min((*mat_3).strength_0.z * multiplier_0 + (*mat_3).strength_0.w * m_3.normal_compression_0, (*mat_3).energy_1.x * multiplier_0);
    var idx_0 : vec4<f32>;
    idx_0[i32(0)] = max(fdiv_0(m_3.tension_0, (*mat_3).strength_0.x * multiplier_0), 0.0f);
    var _S176 : f32;
    if(_S175 > 0.0f)
    {
        _S176 = fdiv_0(m_3.shear_0, _S175);
    }
    else
    {
        _S176 = infinity_0();
    }
    idx_0[i32(1)] = _S176;
    idx_0[i32(2)] = max(fdiv_0(m_3.compression_0, fc_2), 0.0f);
    var _S177 : f32 = (*b_8).stiff1_0.y;
    if(_S177 > 0.0f)
    {
        _S176 = fdiv_0(m_3.compressive_force_0, _S177);
    }
    else
    {
        _S176 = 0.0f;
    }
    idx_0[i32(3)] = _S176;
    return idx_0;
}

fn sq_0( x_8 : f32) -> f32
{
    return x_8 * x_8;
}

fn compressed_region_0( w0_0 : f32,  w1_0 : f32,  dz_0 : f32,  ax_0 : f32,  ay_0 : f32,  region_0 : ptr<function, array<f32, i32(6)>>)
{
    var count_1 : u32;
    var h0_0 : f32 = 0.5f * w0_0;
    var h1_0 : f32 = 0.5f * w1_0;
    var _S178 : f32 = - h0_0;
    var _S179 : f32 = - h1_0;
    var _S180 : array<vec2<f32>, i32(4)> = array<vec2<f32>, i32(4)>( vec2<f32>(_S178, _S179), vec2<f32>(h0_0, _S179), vec2<f32>(h0_0, h1_0), vec2<f32>(_S178, h1_0) );
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
        var _S181 : u32 = i_2;
        var _S182 : u32 = i_2 + u32(1);
        var _S183 : u32 = _S182 % u32(4);
        var _S184 : f32 = _S180[i_2].y;
        var _S185 : f32 = _S180[i_2].x;
        var fp_0 : f32 = dz_0 + ax_0 * _S184 - ay_0 * _S185;
        var _S186 : f32 = _S180[_S183].y;
        var _S187 : f32 = _S180[_S183].x;
        var fq_0 : f32 = dz_0 + ax_0 * _S186 - ay_0 * _S187;
        var _S188 : bool = fp_0 < 0.0f;
        if(_S188)
        {
            var _S189 : u32 = count_2 + u32(1);
            poly_0[count_2] = _S180[_S181];
            count_1 = _S189;
        }
        else
        {
            count_1 = count_2;
        }
        if(_S188 != (fq_0 < 0.0f))
        {
            var t_11 : f32 = fp_0 / (fp_0 - fq_0);
            var _S190 : u32 = count_1 + u32(1);
            poly_0[count_1] = vec2<f32>(_S185 + t_11 * (_S187 - _S185), _S184 + t_11 * (_S186 - _S184));
            count_2 = _S190;
        }
        else
        {
            count_2 = count_1;
        }
        i_2 = _S182;
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
    i_2 = u32(0);
    var a_10 : f32 = 0.0f;
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
        var _S191 : f32 = o_1.x;
        var x0_0 : f32 = poly_0[i_2].x - _S191;
        var _S192 : f32 = o_1.y;
        var y0_0 : f32 = poly_0[i_2].y - _S192;
        var _S193 : u32 = i_2 + u32(1);
        var _S194 : u32 = _S193 % count_2;
        var x1_0 : f32 = poly_0[_S194].x - _S191;
        var y1_0 : f32 = poly_0[_S194].y - _S192;
        var _S195 : f32 = x0_0 * y1_0;
        var _S196 : f32 = x1_0 * y0_0;
        var cr_0 : f32 = _S195 - _S196;
        var a_11 : f32 = a_10 + cr_0 / 2.0f;
        var sx_1 : f32 = sx_0 + (x0_0 + x1_0) * cr_0 / 6.0f;
        var sy_1 : f32 = sy_0 + (y0_0 + y1_0) * cr_0 / 6.0f;
        var ixx_1 : f32 = ixx_0 + (x0_0 * x0_0 + x0_0 * x1_0 + x1_0 * x1_0) * cr_0 / 12.0f;
        var iyy_1 : f32 = iyy_0 + (y0_0 * y0_0 + y0_0 * y1_0 + y1_0 * y1_0) * cr_0 / 12.0f;
        var ixy_1 : f32 = ixy_0 + (_S195 + 2.0f * x0_0 * y0_0 + 2.0f * x1_0 * y1_0 + _S196) * cr_0 / 24.0f;
        i_2 = _S193;
        a_10 = a_11;
        sx_0 = sx_1;
        sy_0 = sy_1;
        ixx_0 = ixx_1;
        iyy_0 = iyy_1;
        ixy_0 = ixy_1;
    }
    if(a_10 <= 0.0f)
    {
        return;
    }
    var cx_0 : f32 = sx_0 / a_10;
    var cy_0 : f32 = sy_0 / a_10;
    (*region_0)[i32(0)] = a_10;
    (*region_0)[i32(1)] = o_1.x + cx_0;
    (*region_0)[i32(2)] = o_1.y + cy_0;
    var _S197 : f32 = a_10 * cx_0;
    (*region_0)[i32(3)] = ixx_0 - _S197 * cx_0;
    (*region_0)[i32(4)] = iyy_0 - a_10 * cy_0 * cy_0;
    (*region_0)[i32(5)] = ixy_0 - _S197 * cy_0;
    return;
}

fn no_tension_patch_0( kn_0 : f32,  w0_1 : f32,  w1_1 : f32,  dz_1 : f32,  ax_1 : f32,  ay_1 : f32) -> vec4<f32>
{
    var r_6 : array<f32, i32(6)>;
    compressed_region_0(w0_1, w1_1, dz_1, ax_1, ay_1, &(r_6));
    var a_12 : f32 = r_6[i32(0)];
    if((r_6[i32(0)]) == 0.0f)
    {
        return vec4<f32>(0.0f);
    }
    var k_7 : f32 = kn_0 / (w0_1 * w1_1);
    var fc_3 : f32 = dz_1 + ax_1 * r_6[i32(2)] - ay_1 * r_6[i32(1)];
    var _S198 : f32 = a_12 * fc_3;
    var _S199 : f32 = - ay_1;
    return vec4<f32>(k_7 * a_12 * fc_3, k_7 * (_S198 * r_6[i32(2)] + (_S199 * r_6[i32(5)] + ax_1 * r_6[i32(4)])), - k_7 * (_S198 * r_6[i32(1)] + (_S199 * r_6[i32(3)] + ax_1 * r_6[i32(5)])), 0.5f * k_7 * (_S198 * fc_3 + ay_1 * ay_1 * r_6[i32(3)] + ax_1 * ax_1 * r_6[i32(4)] - 2.0f * ax_1 * ay_1 * r_6[i32(5)]));
}

var<private> SPRING_AT_0 : array<f32, i32(6)> = array<f32, i32(6)>( -0.4166666567325592f, -0.25f, -0.0833333358168602f, 0.0833333358168602f, 0.25f, 0.4166666567325592f );
fn contact_normal_0( mat_4 : ptr<function, JointMaterial_std140_0>,  b_9 : ptr<function, JointBond_std430_0>,  crush_2 : f32,  d_lin_0 : vec3<f32>,  d_ang_0 : vec3<f32>) -> vec4<f32>
{
    var kn_1 : f32 = (*b_9).stiff0_0.x;
    var w0_2 : f32 = (*b_9).geom0_0.y;
    var w1_2 : f32 = (*b_9).geom0_0.z;
    var nc_sum_0 : f32;
    var m1_0 : f32;
    var m2_0 : f32;
    var energy_2 : f32;
    if(((((*mat_4).kind_flags_0.y) & (u32(4)))) != u32(0))
    {
        var p_1 : vec4<f32> = no_tension_patch_0(kn_1 * (1.0f - crush_2), w0_2, w1_2, d_lin_0.z, d_ang_0.x, d_ang_0.y);
        var _S200 : f32 = p_1.y;
        var _S201 : f32 = p_1.z;
        var _S202 : f32 = p_1.w;
        nc_sum_0 = p_1.x;
        m1_0 = _S200;
        m2_0 = _S201;
        energy_2 = _S202;
    }
    else
    {
        var ki_0 : f32 = kn_1 * (1.0f - crush_2) / 36.0f;
        var _S203 : f32 = d_ang_0.x;
        var _S204 : f32 = d_ang_0.y;
        var spread_0 : f32 = abs(_S203) * 0.4166666567325592f * w1_2 + abs(_S204) * 0.4166666567325592f * w0_2;
        var _S205 : f32 = d_lin_0.z;
        var slack_0 : f32 = 9.99999997475242708e-07f * (abs(_S205) + spread_0);
        if((_S205 - spread_0) > slack_0)
        {
            nc_sum_0 = 0.0f;
            m1_0 = 0.0f;
            m2_0 = 0.0f;
            energy_2 = 0.0f;
        }
        else
        {
            if((_S205 + spread_0) < (- slack_0))
            {
                var i1_0 : f32 = 2.91666650772094727f * w0_2 * w0_2;
                var i2_0 : f32 = 2.91666650772094727f * w1_2 * w1_2;
                var _S206 : f32 = ki_0 * _S203 * i2_0;
                var _S207 : f32 = ki_0 * _S204 * i1_0;
                var _S208 : f32 = 0.5f * ki_0 * (36.0f * _S205 * _S205 + _S203 * _S203 * i2_0 + _S204 * _S204 * i1_0);
                nc_sum_0 = ki_0 * 36.0f * _S205;
                m1_0 = _S206;
                m2_0 = _S207;
                energy_2 = _S208;
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
                    var _S209 : f32 = SPRING_AT_0[i_3] * w0_2;
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
                        var di_0 : f32 = _S205 + _S203 * s2_0 - _S204 * _S209;
                        if(di_0 < 0.0f)
                        {
                            var f_8 : f32 = ki_0 * di_0;
                            var m1_2 : f32 = m1_0 + f_8 * s2_0;
                            var m2_2 : f32 = m2_0 - f_8 * _S209;
                            var energy_4 : f32 = energy_2 + 0.5f * ki_0 * di_0 * di_0;
                            nc_sum_0 = nc_sum_0 + f_8;
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

fn damage_law_0( kind_2 : u32,  kappa_1 : f32,  r_7 : f32) -> f32
{
    if(kappa_1 <= 1.0f)
    {
        return 0.0f;
    }
    if(kind_2 == u32(0))
    {
        if(r_7 <= 1.0f)
        {
            return 1.0f;
        }
        return min(fdiv_0(r_7 * (kappa_1 - 1.0f), kappa_1 * (r_7 - 1.0f)), 1.0f);
    }
    if(kappa_1 >= (0.5f * (r_7 + 1.0f)))
    {
        return 1.0f;
    }
    return 1.0f - fdiv_0(1.0f, kappa_1);
}

fn damage_increment_0( kind_3 : u32,  kappa_old_0 : f32,  lambda_0 : f32,  r_8 : f32,  d_old_0 : f32,  psi_0 : f32) -> vec2<f32>
{
    var _S210 : f32 = max(damage_law_0(kind_3, lambda_0, r_8), d_old_0);
    var _S211 : bool;
    if(_S210 <= d_old_0)
    {
        _S211 = true;
    }
    else
    {
        _S211 = d_old_0 >= 1.0f;
    }
    if(_S211)
    {
        return vec2<f32>(d_old_0, 0.0f);
    }
    var u0_0 : f32 = fdiv_0(psi_0, lambda_0 * lambda_0);
    var _S212 : f32 = max(kappa_old_0, 1.0f);
    if(kind_3 == u32(0))
    {
        if(r_8 > 1.0f)
        {
            return vec2<f32>(_S210, fdiv_0(u0_0 * r_8, r_8 - 1.0f) * max(min(lambda_0, r_8) - min(_S212, r_8), 0.0f));
        }
        return vec2<f32>(_S210, (1.0f - d_old_0) * psi_0);
    }
    var ku_0 : f32 = 0.5f * (r_8 + 1.0f);
    var plateau_0 : f32 = u0_0 * max(min(lambda_0, ku_0) - min(_S212, ku_0), 0.0f);
    var snap_0 : f32;
    if(_S210 >= 1.0f)
    {
        snap_0 = u0_0 * ku_0;
    }
    else
    {
        snap_0 = 0.0f;
    }
    return vec2<f32>(_S210, plateau_0 + snap_0);
}

struct Contact_0
{
     q_lin_1 : vec3<f32>,
     q_ang_1 : vec3<f32>,
     energy_5 : f32,
     diss_0 : f32,
     plastic_0 : vec3<f32>,
};

fn signum_0( x_9 : f32) -> f32
{
    var _S213 : f32;
    if((((bitcast<u32>((x_9))) & (u32(2147483648)))) != u32(0))
    {
        _S213 = -1.0f;
    }
    else
    {
        _S213 = 1.0f;
    }
    return _S213;
}

fn return_map_0( k_8 : f32,  total_0 : f32,  plastic_1 : f32,  cap_0 : f32) -> vec2<f32>
{
    var trial_0 : f32 = k_8 * (total_0 - plastic_1);
    if((abs(trial_0)) <= cap_0)
    {
        return vec2<f32>(trial_0, 0.0f);
    }
    var f_9 : f32 = cap_0 * signum_0(trial_0);
    return vec2<f32>(f_9, fdiv_0(trial_0 - f_9, k_8));
}

fn contact_part_0( mat_5 : ptr<function, JointMaterial_std140_0>,  b_10 : ptr<function, JointBond_std430_0>,  crush_3 : f32,  plastic_2 : vec3<f32>,  d_lin_1 : vec3<f32>,  d_ang_1 : vec3<f32>) -> Contact_0
{
    var c_12 : Contact_0;
    var _S214 : vec3<f32> = vec3<f32>(0.0f);
    c_12.q_lin_1 = _S214;
    c_12.q_ang_1 = _S214;
    c_12.energy_5 = 0.0f;
    c_12.diss_0 = 0.0f;
    c_12.plastic_0 = plastic_2;
    if(((((*mat_5).kind_flags_0.y) & (u32(2)))) == u32(0))
    {
        return c_12;
    }
    var ks_0 : f32 = (*b_10).stiff0_0.y;
    var kt_0 : f32 = (*b_10).stiff1_0.x;
    var _S215 : vec4<f32> = contact_normal_0(&((*mat_5)), &((*b_10)), crush_3, d_lin_1, d_ang_1);
    var nc_sum_2 : f32 = _S215.x;
    var m1_3 : f32 = _S215.y;
    var m2_3 : f32 = _S215.z;
    var energy_6 : f32 = _S215.w;
    var nc_0 : f32 = - nc_sum_2;
    var p_2 : vec3<f32> = plastic_2;
    c_12.q_lin_1 = vec3<f32>(0.0f, 0.0f, nc_sum_2);
    c_12.q_ang_1 = vec3<f32>(m1_3, m2_3, 0.0f);
    var slide_cap_0 : f32 = (*mat_5).strength_0.w * nc_0;
    var _S216 : f32 = ks_0 * (d_lin_1.x - plastic_2.x);
    var _S217 : f32 = ks_0 * (d_lin_1.y - plastic_2.y);
    var tn_0 : f32 = fsqrt_0(_S216 * _S216 + _S217 * _S217);
    var _S218 : bool;
    if(tn_0 > slide_cap_0)
    {
        _S218 = tn_0 > 0.0f;
    }
    else
    {
        _S218 = false;
    }
    var diss_1 : f32;
    if(_S218)
    {
        var _S219 : f32 = fdiv_0(_S216, tn_0);
        var _S220 : f32 = fdiv_0(_S217, tn_0);
        var dslip_0 : f32 = fdiv_0(tn_0 - slide_cap_0, ks_0);
        p_2[i32(0)] = p_2[i32(0)] + _S219 * dslip_0;
        p_2[i32(1)] = p_2[i32(1)] + _S220 * dslip_0;
        c_12.q_lin_1[i32(0)] = _S219 * slide_cap_0;
        c_12.q_lin_1[i32(1)] = _S220 * slide_cap_0;
        diss_1 = slide_cap_0 * dslip_0;
    }
    else
    {
        c_12.q_lin_1[i32(0)] = _S216;
        c_12.q_lin_1[i32(1)] = _S217;
        diss_1 = 0.0f;
    }
    var tq_0 : vec2<f32> = return_map_0(kt_0, d_ang_1.z, p_2.z, slide_cap_0 * (*b_10).geom1_0.z);
    var _S221 : f32 = tq_0.x;
    var _S222 : f32 = tq_0.y;
    var diss_2 : f32 = diss_1 + abs(_S221) * abs(_S222);
    p_2[i32(2)] = p_2[i32(2)] + _S222;
    c_12.q_ang_1[i32(2)] = _S221;
    c_12.energy_5 = energy_6 + 0.5f * (fdiv_0(sq_0(c_12.q_lin_1.x), ks_0) + fdiv_0(sq_0(c_12.q_lin_1.y), ks_0) + fdiv_0(sq_0(_S221), kt_0));
    c_12.diss_0 = diss_2;
    c_12.plastic_0 = p_2;
    return c_12;
}

fn contact_offsets_0( mat_6 : ptr<function, JointMaterial_std140_0>,  b_11 : ptr<function, JointBond_std430_0>,  crush_4 : f32,  plastic_3 : vec3<f32>,  d_lin_2 : vec3<f32>,  d_ang_2 : vec3<f32>) -> vec3<f32>
{
    var _S223 : u32 = (*mat_6).kind_flags_0.y;
    if(((_S223 & (u32(2)))) == u32(0))
    {
        return plastic_3;
    }
    var kn_2 : f32 = (*b_11).stiff0_0.x;
    var ks_1 : f32 = (*b_11).stiff0_0.y;
    var kt_1 : f32 = (*b_11).stiff1_0.x;
    var w0_3 : f32 = (*b_11).geom0_0.y;
    var w1_3 : f32 = (*b_11).geom0_0.z;
    var nc_sum_3 : f32;
    if(((_S223 & (u32(4)))) != u32(0))
    {
        var _S224 : vec4<f32> = no_tension_patch_0(kn_2 * (1.0f - crush_4), w0_3, w1_3, d_lin_2.z, d_ang_2.x, d_ang_2.y);
        nc_sum_3 = _S224.x;
    }
    else
    {
        var ki_1 : f32 = kn_2 * (1.0f - crush_4) / 36.0f;
        var _S225 : f32 = d_ang_2.x;
        var _S226 : f32 = d_ang_2.y;
        var spread_1 : f32 = abs(_S225) * 0.4166666567325592f * w1_3 + abs(_S226) * 0.4166666567325592f * w0_3;
        var _S227 : f32 = d_lin_2.z;
        var slack_1 : f32 = 9.99999997475242708e-07f * (abs(_S227) + spread_1);
        if((_S227 - spread_1) > slack_1)
        {
            nc_sum_3 = 0.0f;
        }
        else
        {
            if((_S227 + spread_1) < (- slack_1))
            {
                nc_sum_3 = ki_1 * 36.0f * _S227;
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
                    var _S228 : f32 = SPRING_AT_0[i_4] * w0_3;
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
                        var di_1 : f32 = _S227 + _S225 * (SPRING_AT_0[j_1] * w1_3) - _S226 * _S228;
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
    var p_3 : vec3<f32> = plastic_3;
    var slide_cap_1 : f32 = (*mat_6).strength_0.w * nc_1;
    var _S229 : f32 = ks_1 * (d_lin_2.x - plastic_3.x);
    var _S230 : f32 = ks_1 * (d_lin_2.y - plastic_3.y);
    var tn_1 : f32 = fsqrt_0(_S229 * _S229 + _S230 * _S230);
    var _S231 : bool;
    if(tn_1 > slide_cap_1)
    {
        _S231 = tn_1 > 0.0f;
    }
    else
    {
        _S231 = false;
    }
    if(_S231)
    {
        var _S232 : f32 = fdiv_0(_S230, tn_1);
        var dslip_1 : f32 = fdiv_0(tn_1 - slide_cap_1, ks_1);
        p_3[i32(0)] = p_3[i32(0)] + fdiv_0(_S229, tn_1) * dslip_1;
        p_3[i32(1)] = p_3[i32(1)] + _S232 * dslip_1;
    }
    p_3[i32(2)] = p_3[i32(2)] + return_map_0(kt_1, d_ang_2.z, p_3.z, slide_cap_1 * (*b_11).geom1_0.z).y;
    return p_3;
}

fn life_rate_0( mat_7 : ptr<function, JointMaterial_std140_0>,  s_4 : f32) -> f32
{
    if(s_4 <= 0.0f)
    {
        return 0.0f;
    }
    var _S233 : f32 = (*mat_7).misc_0.y;
    return fdiv_0((_S233 + 1.0f) * fpow_0(s_4, _S233), (*mat_7).misc_0.z);
}

fn connected_2( st_3 : JointState_0,  has_rebar_2 : bool) -> bool
{
    var _S234 : bool;
    if((st_3.damage_0) < 1.0f)
    {
        _S234 = true;
    }
    else
    {
        if(has_rebar_2)
        {
            _S234 = (st_3.rebar_broken_0) == 0.0f;
        }
        else
        {
            _S234 = false;
        }
    }
    return _S234;
}

fn joint_evaluate_0( mat_8 : ptr<function, JointMaterial_std140_0>,  b_12 : ptr<function, JointBond_std430_0>,  state_2 : ptr<function, JointState_std430_0>,  d_lin_3 : vec3<f32>,  d_ang_3 : vec3<f32>,  dt_9 : f32,  fracture_1 : bool) -> JointResponse_0
{
    var kn_3 : f32 = (*b_12).stiff0_0.x;
    var ks_2 : f32 = (*b_12).stiff0_0.y;
    var kb1_0 : f32 = (*b_12).stiff0_0.z;
    var kb2_0 : f32 = (*b_12).stiff0_0.w;
    var _S235 : vec4<f32> = (*b_12).stiff1_0;
    var kt_2 : f32 = (*b_12).stiff1_0.x;
    var has_rebar_3 : bool = ((*b_12).stiff1_0.w) != 0.0f;
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
    var _S236 : bool = connected_1(&((*state_2)), has_rebar_3);
    var qe_lin_0 : vec3<f32> = d_lin_3 * vec3<f32>(ks_2, ks_2, kn_3);
    var qe_ang_0 : vec3<f32> = d_ang_3 * vec3<f32>(kb1_0, kb2_0, kt_2);
    var _S237 : Measures_0 = stress_measures_0(&((*b_12)), qe_lin_0, qe_ang_0);
    var _S238 : f32 = max(max(_S237.tension_0, _S237.shear_0), _S237.compression_0);
    var _S239 : bool = dt_9 > 0.0f;
    var dif_1 : f32;
    if(_S239)
    {
        var raw_0 : f32 = fdiv_0(max(fdiv_0(_S238 - st_4.governing_stress_0, dt_9), 0.0f), (*mat_8).misc_0.w);
        var tau_2 : f32 = _S235.z;
        if(((flags_0 & (u32(16)))) != u32(0))
        {
            dif_1 = - expm1_accurate_0(- fdiv_0(dt_9, tau_2));
        }
        else
        {
            dif_1 = min(fdiv_0(dt_9, tau_2), 1.0f);
        }
        st_4.strain_rate_0 = st_4.strain_rate_0 + (raw_0 - st_4.strain_rate_0) * dif_1;
        st_4.governing_stress_0 = _S238;
    }
    if(((flags_0 & (u32(32)))) != u32(0))
    {
        var _S240 : f32 = dif_factor_0(&((*mat_8)), st_4.strain_rate_0);
        dif_1 = _S240;
    }
    else
    {
        dif_1 = 1.0f;
    }
    var weibull_0 : f32 = (*b_12).geom1_0.w;
    var _S241 : f32 = fatigue_factor_0(&((*mat_8)), st_4.life_0);
    var multiplier_1 : f32 = weibull_0 * dif_1 * _S241;
    var _S242 : vec4<f32> = failure_indices_0(&((*mat_8)), &((*b_12)), _S237, multiplier_1);
    var _S243 : f32 = _S242.x;
    var _S244 : f32 = _S242.y;
    st_4.utilization_0 = max(max(_S243, _S244), max(_S242.z, _S242.w));
    var _S245 : f32 = d_lin_3.x;
    var _S246 : f32 = d_lin_3.y;
    var _S247 : f32 = d_ang_3.z;
    var _S248 : f32 = ks_2 * (sq_0(_S245) + sq_0(_S246)) + kb1_0 * sq_0(d_ang_3.x) + kb2_0 * sq_0(d_ang_3.y) + kt_2 * sq_0(_S247);
    var _S249 : f32 = d_lin_3.z;
    var _S250 : bool = _S249 > 0.0f;
    if(_S250)
    {
        dif_1 = kn_3 * sq_0(_S249);
    }
    else
    {
        dif_1 = 0.0f;
    }
    var psi_ts_0 : f32 = 0.5f * (_S248 + dif_1);
    var psi_c_0 : f32;
    if(_S249 < 0.0f)
    {
        psi_c_0 = 0.5f * kn_3 * sq_0(_S249);
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
        var _S251 : vec4<f32> = contact_normal_0(&((*mat_8)), &((*b_12)), (*state_2).crush_1, d_lin_3, d_ang_3);
        patch_0 = _S251.w + 0.5f * (ks_2 * (sq_0(_S245 - (*state_2).plastic_x_0) + sq_0(_S246 - (*state_2).plastic_y_0)) + kt_2 * sq_0(_S247 - (*state_2).plastic_t_0));
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
        rebar_0 = 0.5f * ((*b_12).rebar0_0.x * sq_0(_S249 - (*state_2).rebar_plastic_0) + (*b_12).rebar0_0.y * (sq_0(_S245 - (*state_2).rebar_slip0_0) + sq_0(_S246 - (*state_2).rebar_slip1_0)));
    }
    else
    {
        rebar_0 = 0.0f;
    }
    var _S252 : f32 = (*state_2).damage_0;
    var _S253 : f32 = 1.0f - (*state_2).damage_0;
    var _S254 : f32 = (*state_2).crush_1;
    var frozen_0 : f32 = _S253 * (psi_ts_0 + (1.0f - (*state_2).crush_1) * psi_c_0) + (*state_2).damage_0 * patch_0 + rebar_0;
    var diss_contact_0 : f32;
    var psi_contact_0 : f32;
    var intact_normal_0 : f32;
    var dissipated_2 : f32;
    var qc_lin_0 : vec3<f32>;
    if(fracture_1)
    {
        var _S255 : bool = _S243 >= _S244;
        if(_S255)
        {
            diss_contact_0 = _S243;
        }
        else
        {
            diss_contact_0 = _S244;
        }
        var mode_ts_0 : u32;
        if(_S255)
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
                intact_normal_0 = fdiv_0(psi_contact_0 * (*b_12).geom0_0.x * diss_contact_0 * diss_contact_0, psi_ts_0);
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
            var _S256 : f32 = inc_0.x;
            if(_S256 > (st_4.damage_0))
            {
                var _S257 : Contact_0 = contact_part_0(&((*mat_8)), &((*b_12)), st_4.crush_1, plastic_4, d_lin_3, d_ang_3);
                var _S258 : f32 = max(inc_0.y - max(_S257.energy_5 - (1.0f - st_4.crush_1) * psi_c_0, 0.0f) * (_S256 - st_4.damage_0), 0.0f);
                st_4.damage_0 = _S256;
                st_4.mode_0 = mode_ts_0;
                dissipated_2 = _S258;
            }
            else
            {
                dissipated_2 = 0.0f;
            }
        }
        else
        {
            dissipated_2 = 0.0f;
        }
        st_4.kappa_0 = max(st_4.kappa_0, diss_contact_0);
        if(_S252 > 0.0f)
        {
            var _S259 : Contact_0 = contact_part_0(&((*mat_8)), &((*b_12)), _S254, vec3<f32>((*state_2).plastic_x_0, (*state_2).plastic_y_0, (*state_2).plastic_t_0), d_lin_3, d_ang_3);
            qc_lin_0 = qe_ang_0 * vec3<f32>(_S253) + _S259.q_ang_1 * vec3<f32>(_S252);
        }
        else
        {
            qc_lin_0 = qe_ang_0;
        }
        var _S260 : Measures_0 = stress_measures_0(&((*b_12)), vec3<f32>(0.0f, 0.0f, min(qe_lin_0.z, 0.0f)), qc_lin_0);
        var _S261 : vec4<f32> = failure_indices_0(&((*mat_8)), &((*b_12)), _S260, multiplier_1);
        var _S262 : f32 = _S261.z;
        var _S263 : f32 = _S261.w;
        var _S264 : bool = _S262 >= _S263;
        if(_S264)
        {
            psi_contact_0 = _S262;
        }
        else
        {
            psi_contact_0 = _S263;
        }
        if(_S264)
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
                intact_normal_0 = fdiv_0((*mat_8).energy_1.w * (*b_12).geom0_0.x * psi_contact_0 * psi_contact_0, psi_c_0);
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
            var _S265 : f32 = inc_1.x;
            if(_S265 > (st_4.crush_1))
            {
                var dissipated_3 : f32 = dissipated_2 + inc_1.y;
                st_4.crush_1 = _S265;
                st_4.mode_0 = mode_c_0;
                if(_S265 >= 1.0f)
                {
                    changed_0 = (st_4.damage_0) < 1.0f;
                }
                else
                {
                    changed_0 = false;
                }
                if(changed_0)
                {
                    var dissipated_4 : f32 = dissipated_3 + psi_ts_0 * (1.0f - st_4.damage_0);
                    st_4.damage_0 = 1.0f;
                    dissipated_2 = dissipated_4;
                }
                else
                {
                    dissipated_2 = dissipated_3;
                }
            }
        }
        st_4.kappa_c_0 = max(st_4.kappa_c_0, psi_contact_0);
    }
    else
    {
        dissipated_2 = 0.0f;
    }
    var dmg_0 : f32 = st_4.damage_0;
    var _S266 : vec3<f32> = vec3<f32>(0.0f);
    var qc_ang_0 : vec3<f32>;
    if((st_4.damage_0) == 0.0f)
    {
        if(((flags_0 & (u32(8)))) == u32(0))
        {
            var _S267 : vec3<f32> = contact_offsets_0(&((*mat_8)), &((*b_12)), st_4.crush_1, plastic_4, d_lin_3, d_ang_3);
            st_4.plastic_x_0 = _S267.x;
            st_4.plastic_y_0 = _S267.y;
            st_4.plastic_t_0 = _S267.z;
        }
        diss_contact_0 = 0.0f;
        qc_lin_0 = _S266;
        qc_ang_0 = _S266;
        psi_contact_0 = 0.0f;
    }
    else
    {
        var _S268 : Contact_0 = contact_part_0(&((*mat_8)), &((*b_12)), st_4.crush_1, plastic_4, d_lin_3, d_ang_3);
        st_4.plastic_x_0 = _S268.plastic_0.x;
        st_4.plastic_y_0 = _S268.plastic_0.y;
        st_4.plastic_t_0 = _S268.plastic_0.z;
        diss_contact_0 = _S268.diss_0;
        qc_lin_0 = _S268.q_lin_1;
        qc_ang_0 = _S268.q_ang_1;
        psi_contact_0 = _S268.energy_5;
    }
    var dissipated_5 : f32 = dissipated_2 + dmg_0 * diss_contact_0;
    if(_S250)
    {
        intact_normal_0 = kn_3 * _S249;
    }
    else
    {
        intact_normal_0 = (1.0f - st_4.crush_1) * kn_3 * _S249;
    }
    var _S269 : f32 = 1.0f - dmg_0;
    var force_lin_2 : vec3<f32> = vec3<f32>(_S269 * qe_lin_0.x + dmg_0 * qc_lin_0.x, _S269 * qe_lin_0.y + dmg_0 * qc_lin_0.y, _S269 * intact_normal_0 + dmg_0 * qc_lin_0.z);
    var force_ang_2 : vec3<f32> = qe_ang_0 * vec3<f32>(_S269) + qc_ang_0 * vec3<f32>(dmg_0);
    var stored_1 : f32 = _S269 * (psi_ts_0 + (1.0f - st_4.crush_1) * psi_c_0) + dmg_0 * psi_contact_0;
    if(has_rebar_3)
    {
        changed_0 = (st_4.rebar_broken_0) == 0.0f;
    }
    else
    {
        changed_0 = false;
    }
    var stored_2 : f32;
    var force_lin_3 : vec3<f32>;
    if(changed_0)
    {
        var k_axial_0 : f32 = (*b_12).rebar0_0.x;
        var k_dowel_0 : f32 = (*b_12).rebar0_0.y;
        var yield_force_0 : f32 = (*b_12).rebar0_0.z;
        var dowel_capacity_0 : f32 = (*b_12).rebar0_0.w;
        var nr_0 : vec2<f32> = return_map_0(k_axial_0, _S249, st_4.rebar_plastic_0, yield_force_0);
        var v1_0 : vec2<f32> = return_map_0(k_dowel_0, _S245, st_4.rebar_slip0_0, dowel_capacity_0);
        var v2_0 : vec2<f32> = return_map_0(k_dowel_0, _S246, st_4.rebar_slip1_0, dowel_capacity_0);
        var _S270 : f32 = nr_0.y;
        var _S271 : f32 = v1_0.y;
        var _S272 : f32 = v2_0.y;
        var work_2 : f32 = yield_force_0 * abs(_S270) + dowel_capacity_0 * (abs(_S271) + abs(_S272));
        st_4.rebar_plastic_0 = st_4.rebar_plastic_0 + _S270;
        st_4.rebar_slip0_0 = st_4.rebar_slip0_0 + _S271;
        st_4.rebar_slip1_0 = st_4.rebar_slip1_0 + _S272;
        st_4.rebar_work_0 = st_4.rebar_work_0 + work_2;
        var dissipated_6 : f32 = dissipated_5 + work_2;
        var _S273 : f32 = nr_0.x;
        var _S274 : f32 = v1_0.x;
        var _S275 : f32 = v2_0.x;
        var elastic_0 : f32 = 0.5f * (fdiv_0(sq_0(_S273), k_axial_0) + fdiv_0(sq_0(_S274) + sq_0(_S275), k_dowel_0));
        if(fracture_1)
        {
            changed_0 = (st_4.rebar_work_0) >= ((*b_12).rebar1_0.x);
        }
        else
        {
            changed_0 = false;
        }
        if(changed_0)
        {
            st_4.rebar_broken_0 = 1.0f;
            var dissipated_7 : f32 = dissipated_6 + elastic_0;
            force_lin_3 = force_lin_2;
            stored_2 = stored_1;
            dissipated_2 = dissipated_7;
        }
        else
        {
            var stored_3 : f32 = stored_1 + elastic_0;
            force_lin_3 = force_lin_2 + vec3<f32>(_S274, _S275, _S273);
            stored_2 = stored_3;
            dissipated_2 = dissipated_6;
        }
    }
    else
    {
        force_lin_3 = force_lin_2;
        stored_2 = stored_1;
        dissipated_2 = dissipated_5;
    }
    if(fracture_1)
    {
        changed_0 = _S239;
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
        var _S276 : Measures_0 = stress_measures_0(&((*b_12)), force_lin_3, force_ang_2);
        var _S277 : vec4<f32> = failure_indices_0(&((*mat_8)), &((*b_12)), _S276, weibull_0);
        var _S278 : f32 = life_rate_0(&((*mat_8)), max(max(_S277.x, _S277.y), _S277.z));
        st_4.life_0 = max(st_4.life_0 - _S278 * dt_9, 0.0f);
    }
    if((st_4.damage_0) != _S252)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_4.crush_1) != _S254;
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
        released_0 = frozen_0 - stored_2;
    }
    else
    {
        released_0 = 0.0f;
    }
    var _S279 : f32 = max(min(dissipated_2, released_0), 0.0f);
    var overshoot_1 : f32 = released_0 - _S279;
    st_4.dissipated_0 = st_4.dissipated_0 + _S279;
    var resp_0 : JointResponse_0;
    resp_0.force_lin_1 = force_lin_3;
    resp_0.force_ang_1 = force_ang_2;
    resp_0.state_1 = st_4;
    resp_0.dissipated_1 = _S279;
    resp_0.overshoot_0 = overshoot_1;
    resp_0.stored_0 = stored_2;
    if(_S236)
    {
        changed_0 = !connected_2(st_4, has_rebar_3);
    }
    else
    {
        changed_0 = false;
    }
    resp_0.disconnected_0 = changed_0;
    resp_0.fatigue_before_0 = _S241;
    resp_0.measures_0 = _S237;
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

fn secant_factors_0( b_13 : ptr<function, JointBond_std430_0>,  st_5 : JointState_0,  d_lin_4 : vec3<f32>,  f_lin_0 : ptr<function, vec3<f32>>,  f_ang_0 : ptr<function, vec3<f32>>)
{
    var compressed_0 : bool = (d_lin_4.z) < 0.0f;
    var contact_3 : f32;
    if(compressed_0)
    {
        contact_3 = st_5.damage_0;
    }
    else
    {
        contact_3 = 0.0f;
    }
    var _S280 : f32 = 1.0f - st_5.damage_0;
    var _S281 : f32 = max(_S280 + contact_3, 9.99999997475242708e-07f);
    var normal_1 : f32;
    if(compressed_0)
    {
        normal_1 = max(1.0f - st_5.crush_1, 9.99999997475242708e-07f);
    }
    else
    {
        normal_1 = max(_S280, 9.99999997475242708e-07f);
    }
    (*f_lin_0) = vec3<f32>(_S281, _S281, normal_1);
    (*f_ang_0) = vec3<f32>(_S281);
    var _S282 : bool;
    if(((*b_13).stiff1_0.w) != 0.0f)
    {
        _S282 = (st_5.rebar_broken_0) == 0.0f;
    }
    else
    {
        _S282 = false;
    }
    if(_S282)
    {
        var _S283 : vec4<f32> = (*b_13).rebar0_0;
        var _S284 : vec4<f32> = (*b_13).stiff0_0;
        (*f_lin_0)[i32(2)] = (*f_lin_0)[i32(2)] + fdiv_0((*b_13).rebar0_0.x, (*b_13).stiff0_0.x);
        var _S285 : f32 = fdiv_0(_S283.y, _S284.y);
        (*f_lin_0)[i32(0)] = (*f_lin_0)[i32(0)] + _S285;
        (*f_lin_0)[i32(1)] = (*f_lin_0)[i32(1)] + _S285;
    }
    return;
}

fn to_body_0( _S286 : u32,  _S287 : vec3<f32>) -> vec3<f32>
{
    return bonds_0[_S286].t1_0.xyz * vec3<f32>(_S287.x) + bonds_0[_S286].t2_0.xyz * vec3<f32>(_S287.y) + bonds_0[_S286].normal_0.xyz * vec3<f32>(_S287.z);
}

fn comp_add1_2( sum_5 : ptr<function, f32>,  err_4 : ptr<function, f32>,  x_10 : f32)
{
    var t_12 : f32 = (*sum_5) + x_10;
    if((abs((*sum_5))) >= (abs(x_10)))
    {
        (*err_4) = (*err_4) + ((*sum_5) - t_12 + x_10);
    }
    else
    {
        (*err_4) = (*err_4) + (x_10 - t_12 + (*sum_5));
    }
    (*sum_5) = t_12;
    return;
}

fn is_damaged_1( st_6 : JointState_0) -> bool
{
    var _S288 : bool;
    if((st_6.damage_0) > 0.0f)
    {
        _S288 = true;
    }
    else
    {
        _S288 = (st_6.crush_1) > 0.0f;
    }
    return _S288;
}

fn cov_or_0( island_0 : u32,  bits_0 : u32)
{
    islands_0[island_0].coverage_0 = ((islands_0[island_0].coverage_0) | (bits_0));
    return;
}

fn fatigue_factor_1( mat_9 : ptr<function, JointMaterial_std140_0>,  life_2 : f32) -> f32
{
    if(((((*mat_9).kind_flags_0.y) & (u32(64)))) == u32(0))
    {
        return 1.0f;
    }
    return fpow_0(clamp(life_2, 0.0f, 1.0f), fdiv_0(1.0f, (*mat_9).misc_0.y - 2.0f));
}

fn bond_update_0( i_5 : u32,  dt_10 : f32,  fracture_2 : bool,  abs_step_0 : u32,  wide_0 : bool) -> bool
{
    var ra_1 : vec3<f32> = bonds_0[i_5].ra_0.xyz;
    var rb_1 : vec3<f32> = bonds_0[i_5].rb_0.xyz;
    var _S289 : u32 = u32(4) * bonds_0[i_5].law_0.ids_0.y;
    var ta_0 : vec3<f32> = state_0[_S289 + u32(1)].xyz;
    var wa_0 : vec3<f32> = state_0[_S289 + u32(3)].xyz;
    var _S290 : u32 = u32(4) * bonds_0[i_5].law_0.ids_0.z;
    var tb_0 : vec3<f32> = state_0[_S290 + u32(1)].xyz;
    var wb_1 : vec3<f32> = state_0[_S290 + u32(3)].xyz;
    var _S291 : vec3<f32> = to_local_0(i_5, state_0[_S290].xyz + cross(tb_0, rb_1) - (state_0[_S289].xyz + cross(ta_0, ra_1)));
    var _S292 : vec3<f32> = to_local_0(i_5, tb_0 - ta_0);
    var _S293 : vec3<f32> = to_local_0(i_5, state_0[_S290 + u32(2)].xyz + cross(wb_1, rb_1) - (state_0[_S289 + u32(2)].xyz + cross(wa_0, ra_1)));
    var _S294 : vec3<f32> = to_local_0(i_5, wb_1 - wa_0);
    var _S295 : JointMaterial_std140_0 = materials_0.m_0.data_0[bonds_0[i_5].law_0.ids_0.x];
    var _S296 : JointState_std430_0 = bond_dyn_0[i_5].js_0;
    var _S297 : bool = is_damaged_0(&(_S296));
    var _S298 : f32 = _S296.life_0;
    var _S299 : JointBond_std430_0 = bonds_0[i_5].law_0;
    var _S300 : JointResponse_0 = joint_evaluate_0(&(_S295), &(_S299), &(_S296), _S291, _S292, dt_10, fracture_2);
    var _S301 : JointState_0 = JointState_0( bond_dyn_0[i_5].js_0.damage_0, bond_dyn_0[i_5].js_0.crush_1, bond_dyn_0[i_5].js_0.kappa_0, bond_dyn_0[i_5].js_0.kappa_c_0, bond_dyn_0[i_5].js_0.ductility_0, bond_dyn_0[i_5].js_0.ductility_c_0, bond_dyn_0[i_5].js_0.life_0, bond_dyn_0[i_5].js_0.plastic_x_0, bond_dyn_0[i_5].js_0.plastic_y_0, bond_dyn_0[i_5].js_0.plastic_t_0, bond_dyn_0[i_5].js_0.rebar_plastic_0, bond_dyn_0[i_5].js_0.rebar_slip0_0, bond_dyn_0[i_5].js_0.rebar_slip1_0, bond_dyn_0[i_5].js_0.rebar_work_0, bond_dyn_0[i_5].js_0.rebar_broken_0, bond_dyn_0[i_5].js_0.strain_rate_0, bond_dyn_0[i_5].js_0.governing_stress_0, bond_dyn_0[i_5].js_0.dissipated_0, bond_dyn_0[i_5].js_0.utilization_0, bond_dyn_0[i_5].js_0.mode_0 );
    var bd_0 : BondDyn_0;
    bd_0.js_0 = _S301;
    bd_0.force_lin_0 = bond_dyn_0[i_5].force_lin_0;
    bd_0.force_ang_0 = bond_dyn_0[i_5].force_ang_0;
    bd_0.sums_0 = bond_dyn_0[i_5].sums_0;
    bd_0.comps_0 = bond_dyn_0[i_5].comps_0;
    bd_0.events_0 = bond_dyn_0[i_5].events_0;
    var ra_2 : vec3<f32> = bonds_0[i_5].ra_0.xyz;
    var rb_2 : vec3<f32> = bonds_0[i_5].rb_0.xyz;
    var _S302 : JointBond_std430_0 = bonds_0[i_5].law_0;
    var f_lin_1 : vec3<f32>;
    var f_ang_1 : vec3<f32>;
    secant_factors_0(&(_S302), _S300.state_1, _S291, &(f_lin_1), &(f_ang_1));
    var qd_lin_0 : vec3<f32> = _S293 * bonds_0[i_5].c_lin_0.xyz * f_lin_1;
    var qd_ang_0 : vec3<f32> = _S294 * bonds_0[i_5].c_ang_0.xyz * f_ang_1;
    var q_lin_2 : vec3<f32> = _S300.force_lin_1 + qd_lin_0;
    var q_ang_2 : vec3<f32> = _S300.force_ang_1 + qd_ang_0;
    var damped_0 : f32 = (dot(qd_lin_0, _S293) + dot(qd_ang_0, _S294)) * dt_10;
    var _S303 : vec3<f32> = to_body_0(i_5, q_lin_2);
    var _S304 : vec3<f32> = to_body_0(i_5, q_ang_2);
    var _S305 : u32 = u32(3) * i_5;
    scratch_0[_S305] = vec4<f32>(_S303, max(_S300.measures_0.tension_0, _S300.measures_0.compression_0));
    scratch_0[_S305 + u32(1)] = vec4<f32>(_S304 + cross(ra_2, _S303), 0.0f);
    scratch_0[_S305 + u32(2)] = vec4<f32>((vec3<f32>(0) - _S304) + cross(rb_2, (vec3<f32>(0) - _S303)), 0.0f);
    var _S306 : f32 = bd_0.sums_0[i32(0)];
    var _S307 : f32 = bd_0.comps_0[i32(0)];
    comp_add1_2(&(_S306), &(_S307), _S300.dissipated_1);
    bd_0.sums_0[i32(0)] = _S306;
    bd_0.comps_0[i32(0)] = _S307;
    var _S308 : f32 = bd_0.sums_0[i32(1)];
    var _S309 : f32 = bd_0.comps_0[i32(1)];
    comp_add1_2(&(_S308), &(_S309), _S300.overshoot_0);
    bd_0.sums_0[i32(1)] = _S308;
    bd_0.comps_0[i32(1)] = _S309;
    var _S310 : f32 = bd_0.sums_0[i32(2)];
    var _S311 : f32 = bd_0.comps_0[i32(2)];
    comp_add1_2(&(_S310), &(_S311), damped_0);
    bd_0.sums_0[i32(2)] = _S310;
    bd_0.comps_0[i32(2)] = _S311;
    bd_0.force_lin_0 = vec4<f32>(q_lin_2, _S300.stored_0);
    bd_0.force_ang_0 = vec4<f32>(q_ang_2, max(bd_0.force_ang_0.w, _S300.state_1.utilization_0));
    var _S312 : bool;
    if(!_S297)
    {
        _S312 = is_damaged_1(_S300.state_1);
    }
    else
    {
        _S312 = false;
    }
    if(_S312)
    {
        _S312 = (bd_0.events_0.x) == u32(0);
    }
    else
    {
        _S312 = false;
    }
    if(_S312)
    {
        bd_0.events_0[i32(0)] = abs_step_0;
        bd_0.events_0[i32(3)] = _S300.state_1.mode_0;
        if(wide_0)
        {
            cov_or_0(_S302.ids_0.w, u32(32));
        }
        else
        {
            g_seen_0[u32(3)] = u32(1);
        }
    }
    if((bd_0.events_0.y) == u32(0))
    {
        _S312 = (_S300.state_1.life_0) != _S298;
    }
    else
    {
        _S312 = false;
    }
    if(_S312)
    {
        _S312 = (_S300.fatigue_before_0) > 0.99000000953674316f;
    }
    else
    {
        _S312 = false;
    }
    if(_S312)
    {
        var _S313 : f32 = fatigue_factor_1(&(_S295), _S300.state_1.life_0);
        _S312 = _S313 <= 0.99000000953674316f;
    }
    else
    {
        _S312 = false;
    }
    if(_S312)
    {
        bd_0.events_0[i32(1)] = abs_step_0;
        if(wide_0)
        {
            cov_or_0(_S302.ids_0.w, u32(64));
        }
        else
        {
            g_seen_0[u32(4)] = u32(1);
        }
    }
    if(_S300.disconnected_0)
    {
        bd_0.events_0[i32(2)] = abs_step_0;
    }
    bd_0.js_0 = _S300.state_1;
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
    return _S300.disconnected_0;
}

fn island_frame_at_1( gi_1 : u32,  tid_5 : u32)
{
    var woke_2 : bool;
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
    var _S314 : bool = !((((isl_10.info_0.x) & (u32(1)))) != u32(0));
    var _S315 : bool;
    if(_S314)
    {
        _S315 = !driven_1;
    }
    else
    {
        _S315 = false;
    }
    var contact_island_1 : bool = (((isl_10.info_0.x) & (u32(4)))) != u32(0);
    var _S316 : bool = (((isl_10.info_0.x) & (u32(16)))) != u32(0);
    seen_clear_0(tid_5);
    var _S317 : bool = tid_5 == u32(0);
    var settled_1 : bool;
    var cov_5 : u32;
    if(_S317)
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
    var _S318 : f32 = params_0.dt_0;
    var _S319 : bool = (params_0.fracture_0) != u32(0);
    var _S320 : bool = (params_0.rigid_motion_loads_0) != u32(0);
    var rg_8 : Rigid_0 = rigid_of_0(isl_10);
    var work_3 : f32 = 0.0f;
    var work_err_2 : f32 = 0.0f;
    settled_1 = _S316;
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
        var abs_step_1 : u32 = isl_10.info_0.w + done_2 + u32(1);
        var k_9 : u32 = abs_step_1 - u32(1) - params_0.step_start_0;
        var quiet_2 : bool = (((isl_10.info_0.x) & (u32(32)))) != u32(0);
        var _S321 : bool;
        if(settled_1)
        {
            _S321 = !quiet_2;
        }
        else
        {
            _S321 = false;
        }
        var _S322 : bool;
        var explicit_step_1 : bool;
        var c_13 : u32;
        if(_S321)
        {
            var _S323 : vec3<f32> = vec3<f32>(0.0f);
            var norm_1 : vec3<f32> = _S323;
            var unused0_1 : vec3<f32> = _S323;
            c_13 = isl_10.range_0.x + tid_5;
            loop
            {
                if(c_13 < (isl_10.range_0.y))
                {
                }
                else
                {
                    break;
                }
                var _S324 : f32 = settled_chunk_load_0(c_13, rg_8.rot_0, k_9, _S318, contact_island_1);
                norm_1[i32(0)] = norm_1[i32(0)] + _S324;
                c_13 = c_13 + u32(256);
            }
            group_sum3_0(tid_5, &(norm_1), &(unused0_1));
            if((params_0.solve_mode_0) == u32(1))
            {
                _S322 = (abs(norm_1.x - isl_10.energy_0.z)) > (isl_10.energy_0.w);
            }
            else
            {
                _S322 = false;
            }
            if(_S322)
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
        if(_S314)
        {
            _S322 = !quiet_2;
        }
        else
        {
            _S322 = false;
        }
        if(_S322)
        {
            var _S325 : vec3<f32> = vec3<f32>(0.0f);
            var f_10 : vec3<f32> = _S325;
            var t_13 : vec3<f32> = _S325;
            c_13 = isl_10.range_0.x + tid_5;
            loop
            {
                if(c_13 < (isl_10.range_0.y))
                {
                }
                else
                {
                    break;
                }
                net_load_0(c_13, isl_10, rg_8, k_9, _S318, contact_island_1, &(f_10), &(t_13));
                c_13 = c_13 + u32(256);
            }
            group_sum3_0(tid_5, &(f_10), &(t_13));
            rigid_acceleration_0(isl_10, &(rg_8), f_10, t_13);
        }
        if(!settled_1)
        {
            explicit_step_1 = !quiet_2;
        }
        else
        {
            explicit_step_1 = false;
        }
        var c_14 : u32;
        if(explicit_step_1)
        {
            c_13 = isl_10.range_0.z + tid_5;
            loop
            {
                if(c_13 < (isl_10.range_0.w))
                {
                }
                else
                {
                    break;
                }
                var _S326 : bool = bond_update_0(c_13, _S318, _S319, abs_step_1, false);
                if(_S326)
                {
                    g_halt_0 = u32(1);
                }
                c_13 = c_13 + u32(256);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
            c_14 = isl_10.range_0.x + tid_5;
            loop
            {
                if(c_14 < (isl_10.range_0.y))
                {
                }
                else
                {
                    break;
                }
                chunk_update_0(c_14, isl_10, rg_8, _S318, _S320, k_9, contact_island_1, &(work_3), &(work_err_2));
                c_14 = c_14 + u32(256);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
        }
        var _S327 : bool;
        if(_S315)
        {
            _S327 = !quiet_2;
        }
        else
        {
            _S327 = false;
        }
        if(_S327)
        {
            integrate_rigid_0(isl_10, &(rg_8), _S318);
        }
        var _S328 : bool;
        if(_S314)
        {
            _S328 = explicit_step_1;
        }
        else
        {
            _S328 = false;
        }
        if(_S328)
        {
            var _S329 : vec3<f32> = isl_10.wcom_0.xyz;
            var _S330 : vec3<f32> = vec3<f32>(0.0f);
            var tu_2 : vec3<f32> = _S330;
            var pv_2 : vec3<f32> = _S330;
            c_13 = isl_10.range_0.x + tid_5;
            loop
            {
                if(c_13 < (isl_10.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_moments_0(c_13, &(tu_2), &(pv_2));
                c_13 = c_13 + u32(256);
            }
            group_sum3_0(tid_5, &(tu_2), &(pv_2));
            var tr_4 : vec3<f32> = tu_2 / vec3<f32>(isl_10.wcom_0.w);
            var dv_4 : vec3<f32> = pv_2 / vec3<f32>(isl_10.wcom_0.w);
            var lu_2 : vec3<f32> = _S330;
            var lv_2 : vec3<f32> = _S330;
            c_14 = isl_10.range_0.x + tid_5;
            loop
            {
                if(c_14 < (isl_10.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_angular_0(c_14, _S329, tr_4, dv_4, &(lu_2), &(lv_2));
                c_14 = c_14 + u32(256);
            }
            group_sum3_0(tid_5, &(lu_2), &(lv_2));
            var phi_4 : vec3<f32> = rows_mul_0(isl_10.winv0_0, isl_10.winv1_0, isl_10.winv2_0, lu_2);
            var dw_3 : vec3<f32> = rows_mul_0(isl_10.winv0_0, isl_10.winv1_0, isl_10.winv2_0, lv_2);
            var c_15 : u32 = isl_10.range_0.x + tid_5;
            loop
            {
                if(c_15 < (isl_10.range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_15, _S329, tr_4, phi_4, dv_4, dw_3);
                c_15 = c_15 + u32(256);
            }
            if(!driven_1)
            {
                drift_rigid_0(isl_10, &(rg_8), tr_4, phi_4, dv_4, dw_3);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
        }
        var _S331 : bool;
        if(_S317)
        {
            _S331 = (isl_10.probes_0.y) > (isl_10.probes_0.x);
        }
        else
        {
            _S331 = false;
        }
        if(_S331)
        {
            record_probes_0(isl_10, rg_8, k_9);
        }
        var _S332 : u32 = done_2 + u32(1);
        var _S333 : bool;
        if(g_halt_0 != u32(0))
        {
            var _S334 : bool = island_connected_narrow_0(tid_5, isl_10);
            _S333 = _S334;
        }
        else
        {
            _S333 = false;
        }
        var resumed_3 : bool;
        if(_S333)
        {
            if(_S317)
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
            done_2 = _S332;
            resumed_2 = resumed_3;
            break;
        }
        done_2 = _S332;
        woke_3 = woke_2;
        resumed_2 = resumed_3;
    }
    var wsum_1 : vec3<f32> = vec3<f32>(work_3, work_err_2, 0.0f);
    var unused_1 : vec3<f32> = vec3<f32>(0.0f);
    group_sum3_0(tid_5, &(wsum_1), &(unused_1));
    if(_S317)
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
            _S315 = contact_island_1;
        }
        else
        {
            _S315 = false;
        }
        if(_S315)
        {
            contact_split_at_0(isl_10.info_0.w + done_2);
        }
        var _S335 : f32 = wsum_1.x;
        var _S336 : f32 = isl_10.energy_0[i32(0)];
        var _S337 : f32 = isl_10.energy_0[i32(1)];
        comp_add1_1(&(_S336), &(_S337), _S335);
        isl_10.energy_0[i32(0)] = _S336;
        isl_10.energy_0[i32(1)] = _S337 + wsum_1.y;
        isl_10.info_0[i32(3)] = isl_10.info_0[i32(3)] + done_2;
        if((((isl_10.info_0.x) & (u32(8)))) == u32(0))
        {
            isl_10.info_0[i32(2)] = ((isl_10.info_0[i32(2)]) & (u32(4294967293)));
        }
        if(g_halt_0 != u32(0))
        {
            isl_10.info_0[i32(2)] = ((isl_10.info_0[i32(2)]) | (u32(1)));
        }
        var _S338 : u32 = seen_bits_0();
        if(g_halt_0 != u32(0))
        {
            cov_5 = u32(128);
        }
        else
        {
            cov_5 = u32(0);
        }
        var _S339 : u32 = (_S338 | (cov_5));
        if(woke_2)
        {
            cov_5 = u32(2048);
        }
        else
        {
            cov_5 = u32(0);
        }
        var _S340 : u32 = (_S339 | (cov_5));
        if(resumed_2)
        {
            cov_5 = u32(2097280);
        }
        else
        {
            cov_5 = u32(0);
        }
        var cov_6 : u32 = (_S340 | (cov_5));
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
                _S315 = !quiet_3;
            }
            else
            {
                _S315 = false;
            }
            if(_S315)
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
    return;
}

fn group_sum3_1( tid_6 : u32,  a_13 : ptr<function, vec3<f32>>,  b_14 : ptr<function, vec3<f32>>)
{
    var x_11 : vec4<f32> = vec4<f32>((*a_13), 0.0f);
    var y_2 : vec4<f32> = vec4<f32>((*b_14), 0.0f);
    group_sum2_0(tid_6, &(x_11), &(y_2));
    (*a_13) = x_11.xyz;
    (*b_14) = y_2.xyz;
    return;
}

fn group_sum3_2( tid_7 : u32,  a_14 : ptr<function, vec3<f32>>,  b_15 : ptr<function, vec3<f32>>)
{
    var x_12 : vec4<f32> = vec4<f32>((*a_14), 0.0f);
    var y_3 : vec4<f32> = vec4<f32>((*b_15), 0.0f);
    group_sum2_0(tid_7, &(x_12), &(y_3));
    (*a_14) = x_12.xyz;
    (*b_15) = y_3.xyz;
    return;
}

fn group_sum2_1( tid_8 : u32,  a_15 : ptr<function, vec4<f32>>,  b_16 : ptr<function, vec4<f32>>)
{
    g_red_a_0[tid_8] = (*a_15);
    g_red_b_0[tid_8] = (*b_16);
    workgroupBarrier();
    var s_5 : u32 = u32(128);
    loop
    {
        if(s_5 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_8 < s_5)
        {
            var _S341 : u32 = tid_8 + s_5;
            g_red_a_0[tid_8] = g_red_a_0[tid_8] + g_red_a_0[_S341];
            g_red_b_0[tid_8] = g_red_b_0[tid_8] + g_red_b_0[_S341];
        }
        workgroupBarrier();
        s_5 = (s_5 >> (u32(1)));
    }
    (*a_15) = g_red_a_0[i32(0)];
    (*b_16) = g_red_b_0[i32(0)];
    workgroupBarrier();
    return;
}

fn group_sum3_3( tid_9 : u32,  a_16 : ptr<function, vec3<f32>>,  b_17 : ptr<function, vec3<f32>>)
{
    var x_13 : vec4<f32> = vec4<f32>((*a_16), 0.0f);
    var y_4 : vec4<f32> = vec4<f32>((*b_17), 0.0f);
    group_sum2_1(tid_9, &(x_13), &(y_4));
    (*a_16) = x_13.xyz;
    (*b_17) = y_4.xyz;
    return;
}

fn group_sum2_2( tid_10 : u32,  a_17 : ptr<function, vec4<f32>>,  b_18 : ptr<function, vec4<f32>>)
{
    g_red_a_0[tid_10] = (*a_17);
    g_red_b_0[tid_10] = (*b_18);
    workgroupBarrier();
    var s_6 : u32 = u32(128);
    loop
    {
        if(s_6 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_10 < s_6)
        {
            var _S342 : u32 = tid_10 + s_6;
            g_red_a_0[tid_10] = g_red_a_0[tid_10] + g_red_a_0[_S342];
            g_red_b_0[tid_10] = g_red_b_0[tid_10] + g_red_b_0[_S342];
        }
        workgroupBarrier();
        s_6 = (s_6 >> (u32(1)));
    }
    (*a_17) = g_red_a_0[i32(0)];
    (*b_18) = g_red_b_0[i32(0)];
    workgroupBarrier();
    return;
}

fn group_sum2_3( tid_11 : u32,  a_18 : ptr<function, vec4<f32>>,  b_19 : ptr<function, vec4<f32>>)
{
    g_red_a_0[tid_11] = (*a_18);
    g_red_b_0[tid_11] = (*b_19);
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
            var _S343 : u32 = tid_11 + s_7;
            g_red_a_0[tid_11] = g_red_a_0[tid_11] + g_red_a_0[_S343];
            g_red_b_0[tid_11] = g_red_b_0[tid_11] + g_red_b_0[_S343];
        }
        workgroupBarrier();
        s_7 = (s_7 >> (u32(1)));
    }
    (*a_18) = g_red_a_0[i32(0)];
    (*b_19) = g_red_b_0[i32(0)];
    workgroupBarrier();
    return;
}

fn pair_lane_count_0() -> u32
{
    return u32(1);
}

fn stopped_0() -> bool
{
    var _S344 : vec4<u32> = islands_0[params_0.halt_index_0].info_0;
    var _S345 : bool;
    if((((islands_0[params_0.halt_index_0].info_0.z) & (u32(1)))) != u32(0))
    {
        _S345 = true;
    }
    else
    {
        _S345 = (_S344.y) != u32(0);
    }
    return _S345;
}

fn pair_any_0( x_14 : bool,  lanes_0 : u32) -> bool
{
    return x_14;
}

struct WorldPoint_0
{
     hi_3 : vec3<f32>,
     lo_3 : vec3<f32>,
     rel_0 : vec3<f32>,
};

fn chunk_world_0( c_16 : u32) -> WorldPoint_0
{
    var q_5 : Quat_0 = quat_of_0(islands_0[chunks_0[c_16].info_1.y].rotation_0);
    var w_5 : WorldPoint_0;
    w_5.hi_3 = islands_0[chunks_0[c_16].info_1.y].position_0.xyz;
    w_5.lo_3 = islands_0[chunks_0[c_16].info_1.y].position_err_0.xyz;
    w_5.rel_0 = rotate_0(q_5, chunks_0[c_16].center_0.xyz + state_0[u32(4) * c_16].xyz);
    return w_5;
}

fn world_diff_0( a_19 : WorldPoint_0,  b_20 : WorldPoint_0) -> vec3<f32>
{
    return a_19.hi_3 - b_20.hi_3 + (a_19.lo_3 - b_20.lo_3) + (a_19.rel_0 - b_20.rel_0);
}

fn safe_normalize_0( v_9 : vec3<f32>) -> vec3<f32>
{
    var n_1 : f32 = length(v_9);
    var _S346 : vec3<f32>;
    if(n_1 > 1.00000000317107685e-30f)
    {
        _S346 = v_9 / vec3<f32>(n_1);
    }
    else
    {
        _S346 = vec3<f32>(0.0f);
    }
    return _S346;
}

fn from_axis_angle_0( axis_1 : vec3<f32>,  angle_4 : f32) -> Quat_0
{
    var a_20 : vec3<f32> = safe_normalize_0(axis_1);
    var _S347 : f32 = 0.5f * angle_4;
    var s_8 : f32 = sin(_S347);
    var q_6 : Quat_0;
    q_6.w_0 = cos(_S347);
    q_6.x_0 = a_20.x * s_8;
    q_6.y_0 = a_20.y * s_8;
    q_6.z_0 = a_20.z * s_8;
    return q_6;
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

fn chunk_box_0( c_17 : u32,  center_2 : vec3<f32>) -> Box_0
{
    var q_7 : Quat_0 = quat_of_0(islands_0[chunks_0[c_17].info_1.y].rotation_0);
    var th_3 : vec3<f32> = state_0[u32(4) * c_17 + u32(1)].xyz;
    var hidden_0 : Quat_0 = from_axis_angle_0(th_3, length(th_3));
    var b_21 : Box_0;
    b_21.center_1 = center_2;
    b_21.axis0_0 = rotate_0(q_7, rotate_0(hidden_0, vec3<f32>(chunks_0[c_17].crot0_0.x, chunks_0[c_17].crot1_0.x, chunks_0[c_17].crot2_0.x)));
    b_21.axis1_0 = rotate_0(q_7, rotate_0(hidden_0, vec3<f32>(chunks_0[c_17].crot0_0.y, chunks_0[c_17].crot1_0.y, chunks_0[c_17].crot2_0.y)));
    b_21.axis2_0 = rotate_0(q_7, rotate_0(hidden_0, vec3<f32>(chunks_0[c_17].crot0_0.z, chunks_0[c_17].crot1_0.z, chunks_0[c_17].crot2_0.z)));
    b_21.half_2 = chunks_0[c_17].half_0.xyz;
    b_21.hull_at_0 = (bitcast<u32>((chunks_0[c_17].cmat_0.z)));
    var _S348 : u32 = (bitcast<u32>((chunks_0[c_17].cmat_0.w)));
    b_21.hull_v_0 = (_S348 & (u32(255)));
    b_21.hull_f_0 = (_S348 >> (u32(8)));
    return b_21;
}

fn sample_count_0( b_22 : Box_0) -> u32
{
    var _S349 : u32;
    if((b_22.hull_v_0) == u32(0))
    {
        _S349 = u32(14);
    }
    else
    {
        _S349 = b_22.hull_v_0 + b_22.hull_f_0;
    }
    return _S349;
}

fn may_overlap_lanes_0( a_21 : Box_0,  b_23 : Box_0,  lane_0 : u32,  lanes_1 : u32) -> bool
{
    var separated_0 : bool = false;
    var i_6 : u32 = lane_0;
    loop
    {
        if(i_6 < u32(15))
        {
        }
        else
        {
            break;
        }
        var d_2 : vec3<f32> = b_23.center_1 - a_21.center_1;
        var margin_0 : f32 = 0.00000999999974738f * (length(a_21.half_2) + length(b_23.half_2));
        var axes_0 : array<vec3<f32>, i32(6)> = array<vec3<f32>, i32(6)>( a_21.axis0_0, a_21.axis1_0, a_21.axis2_0, b_23.axis0_0, b_23.axis1_0, b_23.axis2_0 );
        var l_3 : vec3<f32>;
        if(i_6 < u32(6))
        {
            l_3 = axes_0[i_6];
        }
        else
        {
            var _S350 : u32 = i_6 - u32(6);
            l_3 = cross(axes_0[_S350 / u32(3)], axes_0[u32(3) + _S350 % u32(3)]);
        }
        var len_0 : f32 = length(l_3);
        if(len_0 <= 9.99999997475242708e-07f)
        {
            i_6 = i_6 + lanes_1;
            continue;
        }
        var separated_1 : bool;
        if((abs(dot(d_2, l_3))) > (a_21.half_2.x * abs(dot(a_21.axis0_0, l_3)) + a_21.half_2.y * abs(dot(a_21.axis1_0, l_3)) + a_21.half_2.z * abs(dot(a_21.axis2_0, l_3)) + (b_23.half_2.x * abs(dot(b_23.axis0_0, l_3)) + b_23.half_2.y * abs(dot(b_23.axis1_0, l_3)) + b_23.half_2.z * abs(dot(b_23.axis2_0, l_3))) + margin_0 * len_0))
        {
            separated_1 = true;
        }
        else
        {
            separated_1 = separated_0;
        }
        separated_0 = separated_1;
        i_6 = i_6 + lanes_1;
    }
    return !pair_any_0(separated_0, lanes_1);
}

fn may_overlap_0( a_22 : Box_0,  b_24 : Box_0) -> bool
{
    var _S351 : vec3<f32> = b_24.center_1 - a_22.center_1;
    var _S352 : f32 = 0.00000999999974738f * (length(a_22.half_2) + length(b_24.half_2));
    var _S353 : array<vec3<f32>, i32(6)> = array<vec3<f32>, i32(6)>( a_22.axis0_0, a_22.axis1_0, a_22.axis2_0, b_24.axis0_0, b_24.axis1_0, b_24.axis2_0 );
    var i_7 : u32 = u32(0);
    loop
    {
        if(i_7 < u32(15))
        {
        }
        else
        {
            break;
        }
        var l_4 : vec3<f32>;
        if(i_7 < u32(6))
        {
            l_4 = _S353[i_7];
        }
        else
        {
            var _S354 : u32 = i_7 - u32(6);
            l_4 = cross(_S353[_S354 / u32(3)], _S353[u32(3) + _S354 % u32(3)]);
        }
        var len_1 : f32 = length(l_4);
        if(len_1 <= 9.99999997475242708e-07f)
        {
            i_7 = i_7 + u32(1);
            continue;
        }
        if((abs(dot(_S351, l_4))) > (a_22.half_2.x * abs(dot(a_22.axis0_0, l_4)) + a_22.half_2.y * abs(dot(a_22.axis1_0, l_4)) + a_22.half_2.z * abs(dot(a_22.axis2_0, l_4)) + (b_24.half_2.x * abs(dot(b_24.axis0_0, l_4)) + b_24.half_2.y * abs(dot(b_24.axis1_0, l_4)) + b_24.half_2.z * abs(dot(b_24.axis2_0, l_4))) + _S352 * len_1))
        {
            return false;
        }
        i_7 = i_7 + u32(1);
    }
    return true;
}

fn chunk_velocity_0( c_18 : u32,  v_10 : ptr<function, vec3<f32>>,  w_6 : ptr<function, vec3<f32>>)
{
    var q_8 : Quat_0 = quat_of_0(islands_0[chunks_0[c_18].info_1.y].rotation_0);
    var _S355 : u32 = u32(4) * c_18;
    var _S356 : vec3<f32> = islands_0[chunks_0[c_18].info_1.y].angular_velocity_0.xyz;
    (*v_10) = islands_0[chunks_0[c_18].info_1.y].velocity_0.xyz + islands_0[chunks_0[c_18].info_1.y].velocity_err_0.xyz + cross(_S356, rotate_0(q_8, chunks_0[c_18].center_0.xyz + state_0[_S355].xyz - islands_0[chunks_0[c_18].info_1.y].com_0.xyz)) + rotate_0(q_8, state_0[_S355 + u32(2)].xyz);
    (*w_6) = _S356 + rotate_0(q_8, state_0[_S355 + u32(3)].xyz);
    return;
}

fn chunk_velocity_1( c_19 : u32,  v_11 : ptr<function, vec3<f32>>,  w_7 : ptr<function, vec3<f32>>)
{
    var q_9 : Quat_0 = quat_of_0(islands_0[chunks_0[c_19].info_1.y].rotation_0);
    var _S357 : u32 = u32(4) * c_19;
    var _S358 : vec3<f32> = islands_0[chunks_0[c_19].info_1.y].angular_velocity_0.xyz;
    (*v_11) = islands_0[chunks_0[c_19].info_1.y].velocity_0.xyz + islands_0[chunks_0[c_19].info_1.y].velocity_err_0.xyz + cross(_S358, rotate_0(q_9, chunks_0[c_19].center_0.xyz + state_0[_S357].xyz - islands_0[chunks_0[c_19].info_1.y].com_0.xyz)) + rotate_0(q_9, state_0[_S357 + u32(2)].xyz);
    (*w_7) = _S358 + rotate_0(q_9, state_0[_S357 + u32(3)].xyz);
    return;
}

fn box_to_world_0( b_25 : Box_0,  local_0 : vec3<f32>) -> vec3<f32>
{
    return b_25.axis0_0 * vec3<f32>(local_0.x) + b_25.axis1_0 * vec3<f32>(local_0.y) + b_25.axis2_0 * vec3<f32>(local_0.z);
}

fn box_axis_0( b_26 : Box_0,  k_10 : u32) -> vec3<f32>
{
    var _S359 : vec3<f32>;
    if(k_10 == u32(0))
    {
        _S359 = b_26.axis0_0;
    }
    else
    {
        if(k_10 == u32(1))
        {
            _S359 = b_26.axis1_0;
        }
        else
        {
            _S359 = b_26.axis2_0;
        }
    }
    return _S359;
}

fn comp3_0( v_12 : vec3<f32>,  k_11 : u32) -> f32
{
    var _S360 : f32;
    if(k_11 == u32(0))
    {
        _S360 = v_12.x;
    }
    else
    {
        if(k_11 == u32(1))
        {
            _S360 = v_12.y;
        }
        else
        {
            _S360 = v_12.z;
        }
    }
    return _S360;
}

fn sample_point_0( b_27 : Box_0,  i_8 : u32) -> vec3<f32>
{
    if((b_27.hull_v_0) != u32(0))
    {
        var local_1 : vec3<f32>;
        if(i_8 < (b_27.hull_v_0))
        {
            local_1 = loads_0[b_27.hull_at_0 + i_8].xyz * vec3<f32>(0.89999997615814209f);
        }
        else
        {
            local_1 = loads_0[b_27.hull_at_0 + b_27.hull_v_0 + b_27.hull_f_0 + (i_8 - b_27.hull_v_0)].xyz;
        }
        return b_27.center_1 + box_to_world_0(b_27, local_1);
    }
    var sign_0 : f32;
    if(i_8 < u32(8))
    {
        var h_0 : vec3<f32> = b_27.half_2 * vec3<f32>(0.89999997615814209f);
        if(((i_8 & (u32(1)))) == u32(0))
        {
            sign_0 = - h_0.x;
        }
        else
        {
            sign_0 = h_0.x;
        }
        var _S361 : f32;
        if(((i_8 & (u32(2)))) == u32(0))
        {
            _S361 = - h_0.y;
        }
        else
        {
            _S361 = h_0.y;
        }
        var _S362 : f32;
        if(((i_8 & (u32(4)))) == u32(0))
        {
            _S362 = - h_0.z;
        }
        else
        {
            _S362 = h_0.z;
        }
        return b_27.center_1 + b_27.axis0_0 * vec3<f32>(sign_0) + b_27.axis1_0 * vec3<f32>(_S361) + b_27.axis2_0 * vec3<f32>(_S362);
    }
    var _S363 : u32 = i_8 - u32(8);
    var axis_2 : u32 = _S363 / u32(2);
    if((_S363 % u32(2)) == u32(0))
    {
        sign_0 = -1.0f;
    }
    else
    {
        sign_0 = 1.0f;
    }
    return b_27.center_1 + box_axis_0(b_27, axis_2) * vec3<f32>((sign_0 * comp3_0(b_27.half_2, axis_2)));
}

fn box_to_local_0( b_28 : Box_0,  r_9 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(dot(r_9, b_28.axis0_0), dot(r_9, b_28.axis1_0), dot(r_9, b_28.axis2_0));
}

fn hull_signed_distance_0( b_29 : Box_0,  local_2 : vec3<f32>,  face_0 : ptr<function, u32>) -> f32
{
    (*face_0) = u32(0);
    var best_0 : f32 = -1.00000001504746622e+30f;
    var f_11 : u32 = u32(0);
    loop
    {
        if(f_11 < (b_29.hull_f_0))
        {
        }
        else
        {
            break;
        }
        var plane_0 : vec4<f32> = loads_0[b_29.hull_at_0 + b_29.hull_v_0 + f_11];
        var d_3 : f32 = dot(plane_0.xyz, local_2) - plane_0.w;
        if(d_3 > best_0)
        {
            (*face_0) = f_11;
            best_0 = d_3;
        }
        f_11 = f_11 + u32(1);
    }
    return best_0;
}

fn penetration_0( b_30 : Box_0,  p_4 : vec3<f32>,  depth_0 : ptr<function, f32>,  normal_2 : ptr<function, vec3<f32>>) -> bool
{
    (*depth_0) = 0.0f;
    (*normal_2) = vec3<f32>(0.0f);
    var r_10 : vec3<f32> = p_4 - b_30.center_1;
    var _S364 : vec3<f32> = b_30.half_2;
    if((dot(r_10, r_10)) > (dot(_S364, _S364) * 1.00001001358032227f))
    {
        return false;
    }
    if((b_30.hull_v_0) != u32(0))
    {
        var face_1 : u32;
        var d_4 : f32 = hull_signed_distance_0(b_30, box_to_local_0(b_30, r_10), &(face_1));
        if(!(d_4 < 0.0f))
        {
            return false;
        }
        (*depth_0) = - d_4;
        (*normal_2) = box_to_world_0(b_30, loads_0[b_30.hull_at_0 + b_30.hull_v_0 + face_1].xyz);
        return true;
    }
    var best_1 : f32 = 1.00000001504746622e+30f;
    var axis_3 : u32 = u32(0);
    var side_0 : f32 = 1.0f;
    var k_12 : u32 = u32(0);
    loop
    {
        if(k_12 < u32(3))
        {
        }
        else
        {
            break;
        }
        var local_3 : f32 = dot(r_10, box_axis_0(b_30, k_12));
        var d_5 : f32 = comp3_0(b_30.half_2, k_12) - abs(local_3);
        if(d_5 <= 0.0f)
        {
            return false;
        }
        if(d_5 < best_1)
        {
            var _S365 : f32;
            if(local_3 >= 0.0f)
            {
                _S365 = 1.0f;
            }
            else
            {
                _S365 = -1.0f;
            }
            best_1 = d_5;
            axis_3 = k_12;
            side_0 = _S365;
        }
        k_12 = k_12 + u32(1);
    }
    (*depth_0) = best_1;
    (*normal_2) = box_axis_0(b_30, axis_3) * vec3<f32>(side_0);
    return true;
}

fn penetration_1( b_31 : Box_0,  p_5 : vec3<f32>,  depth_1 : ptr<function, f32>,  normal_3 : ptr<function, vec3<f32>>) -> bool
{
    (*depth_1) = 0.0f;
    (*normal_3) = vec3<f32>(0.0f);
    var r_11 : vec3<f32> = p_5 - b_31.center_1;
    var _S366 : vec3<f32> = b_31.half_2;
    if((dot(r_11, r_11)) > (dot(_S366, _S366) * 1.00001001358032227f))
    {
        return false;
    }
    if((b_31.hull_v_0) != u32(0))
    {
        var face_2 : u32;
        var d_6 : f32 = hull_signed_distance_0(b_31, box_to_local_0(b_31, r_11), &(face_2));
        if(!(d_6 < 0.0f))
        {
            return false;
        }
        (*depth_1) = - d_6;
        (*normal_3) = box_to_world_0(b_31, loads_0[b_31.hull_at_0 + b_31.hull_v_0 + face_2].xyz);
        return true;
    }
    var best_2 : f32 = 1.00000001504746622e+30f;
    var axis_4 : u32 = u32(0);
    var side_1 : f32 = 1.0f;
    var k_13 : u32 = u32(0);
    loop
    {
        if(k_13 < u32(3))
        {
        }
        else
        {
            break;
        }
        var local_4 : f32 = dot(r_11, box_axis_0(b_31, k_13));
        var d_7 : f32 = comp3_0(b_31.half_2, k_13) - abs(local_4);
        if(d_7 <= 0.0f)
        {
            return false;
        }
        if(d_7 < best_2)
        {
            var _S367 : f32;
            if(local_4 >= 0.0f)
            {
                _S367 = 1.0f;
            }
            else
            {
                _S367 = -1.0f;
            }
            best_2 = d_7;
            axis_4 = k_13;
            side_1 = _S367;
        }
        k_13 = k_13 + u32(1);
    }
    (*depth_1) = best_2;
    (*normal_3) = box_axis_0(b_31, axis_4) * vec3<f32>(side_1);
    return true;
}

fn pair_point_0( ba_0 : Box_0,  bb_0 : Box_0,  na_0 : u32,  e_2 : u32,  p_6 : ptr<function, vec3<f32>>,  n_2 : ptr<function, vec3<f32>>,  d_8 : ptr<function, f32>) -> bool
{
    if(e_2 < na_0)
    {
        var _S368 : vec3<f32> = sample_point_0(ba_0, e_2);
        (*p_6) = _S368;
        var _S369 : bool = penetration_1(bb_0, _S368, &((*d_8)), &((*n_2)));
        return _S369;
    }
    var _S370 : vec3<f32> = sample_point_0(bb_0, e_2 - na_0);
    (*p_6) = _S370;
    var _S371 : bool = penetration_1(ba_0, _S370, &((*d_8)), &((*n_2)));
    if(!_S371)
    {
        return false;
    }
    (*n_2) = (vec3<f32>(0) - (*n_2));
    return true;
}

fn is_nan_0( x_15 : f32) -> bool
{
    return (((bitcast<u32>((x_15))) & (u32(2147483647)))) > u32(2139095040);
}

fn pair_count_sum_0( x_16 : u32,  lanes_2 : u32) -> u32
{
    return x_16;
}

fn half_thickness_and_area_0( b_32 : Box_0,  d_9 : vec3<f32>) -> vec2<f32>
{
    var k_14 : u32 = u32(0);
    var h_1 : f32 = 0.0f;
    var area_1 : f32 = 0.0f;
    loop
    {
        if(k_14 < u32(3))
        {
        }
        else
        {
            break;
        }
        var c_20 : f32 = abs(dot(d_9, box_axis_0(b_32, k_14)));
        var h_2 : f32 = h_1 + c_20 * comp3_0(b_32.half_2, k_14);
        var _S372 : u32 = k_14 + u32(1);
        var area_2 : f32 = area_1 + c_20 * 4.0f * comp3_0(b_32.half_2, _S372 % u32(3)) * comp3_0(b_32.half_2, (k_14 + u32(2)) % u32(3));
        k_14 = _S372;
        h_1 = h_2;
        area_1 = area_2;
    }
    return vec2<f32>(h_1, area_1);
}

fn contact_stiffness_0( ea_0 : f32,  a_23 : Box_0,  eb_0 : f32,  b_33 : Box_0,  dir_2 : vec3<f32>) -> f32
{
    var d_10 : vec3<f32> = safe_normalize_0(dir_2);
    var ta_1 : vec2<f32> = half_thickness_and_area_0(a_23, d_10);
    var tb_1 : vec2<f32> = half_thickness_and_area_0(b_33, d_10);
    return min(ta_1.y, tb_1.y) / (ta_1.x / ea_0 + tb_1.x / eb_0);
}

fn penalty_force_0( k_15 : f32,  m_red_0 : f32,  friction_0 : f32,  depth_2 : f32,  normal_4 : vec3<f32>,  rel_velocity_0 : vec3<f32>,  dt_11 : f32,  points_0 : u32,  stored_4 : ptr<function, f32>,  dissipated_8 : ptr<function, f32>) -> vec3<f32>
{
    var c_max_0 : f32 = 1.0f / max(f32(points_0), 10.0f) * m_red_0 / dt_11;
    var vn_0 : f32 = dot(rel_velocity_0, normal_4);
    var _S373 : f32 = k_15 * depth_2;
    var _S374 : f32 = min(2.0f * params_0.zeta_0 * sqrt(k_15 * m_red_0), c_max_0) * vn_0;
    var _S375 : f32 = _S373 - _S374;
    var _S376 : f32 = max(_S375, 0.0f);
    var vt_0 : vec3<f32> = rel_velocity_0 - normal_4 * vec3<f32>(vn_0);
    var vt_mag_0 : f32 = length(vt_0);
    var _S377 : f32 = friction_0 * _S376;
    var _S378 : f32 = min(_S377, min(c_max_0, _S377 / 0.00100000004749745f) * vt_mag_0);
    var ft_0 : vec3<f32>;
    if(vt_mag_0 > 0.0f)
    {
        ft_0 = (vec3<f32>(0) - vt_0) * vec3<f32>((_S378 / vt_mag_0));
    }
    else
    {
        ft_0 = vec3<f32>(0.0f);
    }
    (*stored_4) = 0.5f * k_15 * depth_2 * depth_2;
    var damping_power_0 : f32;
    if(_S375 > 0.0f)
    {
        damping_power_0 = _S374 * vn_0;
    }
    else
    {
        damping_power_0 = _S373 * max(vn_0, 0.0f);
    }
    (*dissipated_8) = (damping_power_0 + length(ft_0) * vt_mag_0) * dt_11;
    return normal_4 * vec3<f32>(_S376) + ft_0;
}

fn penalty_force_1( k_16 : f32,  m_red_1 : f32,  friction_1 : f32,  depth_3 : f32,  normal_5 : vec3<f32>,  rel_velocity_1 : vec3<f32>,  dt_12 : f32,  points_1 : u32,  stored_5 : ptr<function, f32>,  dissipated_9 : ptr<function, f32>) -> vec3<f32>
{
    var c_max_1 : f32 = 1.0f / max(f32(points_1), 10.0f) * m_red_1 / dt_12;
    var vn_1 : f32 = dot(rel_velocity_1, normal_5);
    var _S379 : f32 = k_16 * depth_3;
    var _S380 : f32 = min(2.0f * params_0.zeta_0 * sqrt(k_16 * m_red_1), c_max_1) * vn_1;
    var _S381 : f32 = _S379 - _S380;
    var _S382 : f32 = max(_S381, 0.0f);
    var vt_1 : vec3<f32> = rel_velocity_1 - normal_5 * vec3<f32>(vn_1);
    var vt_mag_1 : f32 = length(vt_1);
    var _S383 : f32 = friction_1 * _S382;
    var _S384 : f32 = min(_S383, min(c_max_1, _S383 / 0.00100000004749745f) * vt_mag_1);
    var ft_1 : vec3<f32>;
    if(vt_mag_1 > 0.0f)
    {
        ft_1 = (vec3<f32>(0) - vt_1) * vec3<f32>((_S384 / vt_mag_1));
    }
    else
    {
        ft_1 = vec3<f32>(0.0f);
    }
    (*stored_5) = 0.5f * k_16 * depth_3 * depth_3;
    var damping_power_1 : f32;
    if(_S381 > 0.0f)
    {
        damping_power_1 = _S380 * vn_1;
    }
    else
    {
        damping_power_1 = _S379 * max(vn_1, 0.0f);
    }
    (*dissipated_9) = (damping_power_1 + length(ft_1) * vt_mag_1) * dt_12;
    return normal_5 * vec3<f32>(_S382) + ft_1;
}

fn penalty_force_2( k_17 : f32,  m_red_2 : f32,  friction_2 : f32,  depth_4 : f32,  normal_6 : vec3<f32>,  rel_velocity_2 : vec3<f32>,  dt_13 : f32,  points_2 : u32,  stored_6 : ptr<function, f32>,  dissipated_10 : ptr<function, f32>) -> vec3<f32>
{
    var c_max_2 : f32 = 1.0f / max(f32(points_2), 10.0f) * m_red_2 / dt_13;
    var vn_2 : f32 = dot(rel_velocity_2, normal_6);
    var _S385 : f32 = k_17 * depth_4;
    var _S386 : f32 = min(2.0f * params_0.zeta_0 * sqrt(k_17 * m_red_2), c_max_2) * vn_2;
    var _S387 : f32 = _S385 - _S386;
    var _S388 : f32 = max(_S387, 0.0f);
    var vt_2 : vec3<f32> = rel_velocity_2 - normal_6 * vec3<f32>(vn_2);
    var vt_mag_2 : f32 = length(vt_2);
    var _S389 : f32 = friction_2 * _S388;
    var _S390 : f32 = min(_S389, min(c_max_2, _S389 / 0.00100000004749745f) * vt_mag_2);
    var ft_2 : vec3<f32>;
    if(vt_mag_2 > 0.0f)
    {
        ft_2 = (vec3<f32>(0) - vt_2) * vec3<f32>((_S390 / vt_mag_2));
    }
    else
    {
        ft_2 = vec3<f32>(0.0f);
    }
    (*stored_6) = 0.5f * k_17 * depth_4 * depth_4;
    var damping_power_2 : f32;
    if(_S387 > 0.0f)
    {
        damping_power_2 = _S386 * vn_2;
    }
    else
    {
        damping_power_2 = _S385 * max(vn_2, 0.0f);
    }
    (*dissipated_10) = (damping_power_2 + length(ft_2) * vt_mag_2) * dt_13;
    return normal_6 * vec3<f32>(_S388) + ft_2;
}

fn pair_sum3_0( x_17 : vec3<f32>,  lanes_3 : u32) -> vec3<f32>
{
    return x_17;
}

fn pair_sum_0( x_18 : f32,  lanes_4 : u32) -> f32
{
    return x_18;
}

fn comp_add1_3( sum_6 : ptr<function, f32>,  err_5 : ptr<function, f32>,  x_19 : f32)
{
    var t_14 : f32 = (*sum_6) + x_19;
    if((abs((*sum_6))) >= (abs(x_19)))
    {
        (*err_5) = (*err_5) + ((*sum_6) - t_14 + x_19);
    }
    else
    {
        (*err_5) = (*err_5) + (x_19 - t_14 + (*sum_6));
    }
    (*sum_6) = t_14;
    return;
}

fn comp_add1_4( sum_7 : ptr<function, f32>,  err_6 : ptr<function, f32>,  x_20 : f32)
{
    var t_15 : f32 = (*sum_7) + x_20;
    if((abs((*sum_7))) >= (abs(x_20)))
    {
        (*err_6) = (*err_6) + ((*sum_7) - t_15 + x_20);
    }
    else
    {
        (*err_6) = (*err_6) + (x_20 - t_15 + (*sum_7));
    }
    (*sum_7) = t_15;
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

fn pair_contact_0( i_9 : u32,  lane_1 : u32,  lanes_5 : u32)
{
    var at_3 : u32 = params_0.pair_index_0 + u32(6) * i_9;
    var ca_0 : u32 = index_0[at_3];
    var cb_0 : u32 = index_0[at_3 + u32(1)];
    var _S391 : u32 = index_0[at_3 + u32(3)];
    var _S392 : f32 = (bitcast<f32>((index_0[at_3 + u32(4)])));
    var _S393 : f32 = (bitcast<f32>((index_0[at_3 + u32(5)])));
    var _S394 : f32 = params_0.dt_0;
    var out_0 : u32 = params_0.slot_base_0 + u32(2) * index_0[at_3 + u32(2)];
    var cb_rel_0 : vec3<f32> = world_diff_0(chunk_world_0(cb_0), chunk_world_0(ca_0));
    var touching_0 : bool = !((length(cb_rel_0)) > (chunks_0[ca_0].half_0.w + chunks_0[cb_0].half_0.w));
    var ledger_1 : vec4<f32> = scratch_0[params_0.ledger_base_0 + i_9];
    var flags_1 : u32 = (bitcast<u32>((scratch_0[params_0.ledger_base_0 + i_9].w)));
    var _S395 : vec3<f32> = vec3<f32>(0.0f);
    var ba_1 : Box_0 = chunk_box_0(ca_0, _S395);
    var bb_1 : Box_0 = chunk_box_0(cb_0, cb_rel_0);
    var _S396 : u32 = sample_count_0(ba_1);
    var _S397 : u32 = _S396 + sample_count_0(bb_1);
    var has_state_0 : bool;
    var e_3 : u32;
    if(!touching_0)
    {
        if(((flags_1 & (u32(1)))) != u32(0))
        {
            e_3 = lane_1;
            loop
            {
                if(e_3 < _S397)
                {
                }
                else
                {
                    break;
                }
                contact_state_0[_S391 + e_3] = vec4<f32>((bitcast<f32>((u32(2143289344)))), 0.0f, 0.0f, 0.0f);
                e_3 = e_3 + lanes_5;
            }
        }
        if(lane_1 == u32(0))
        {
            if(((flags_1 & (u32(2)))) != u32(0))
            {
                e_3 = u32(0);
                loop
                {
                    if(e_3 < u32(4))
                    {
                    }
                    else
                    {
                        break;
                    }
                    scratch_0[out_0 + e_3] = vec4<f32>(0.0f);
                    e_3 = e_3 + u32(1);
                }
            }
            if(flags_1 != u32(0))
            {
                has_state_0 = true;
            }
            else
            {
                has_state_0 = (ledger_1.x) != 0.0f;
            }
            if(has_state_0)
            {
                ledger_1[i32(0)] = 0.0f;
                ledger_1[i32(3)] = (bitcast<f32>((u32(0))));
                scratch_0[params_0.ledger_base_0 + i_9] = ledger_1;
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
    var fa_1 : vec3<f32>;
    var ta_2 : vec3<f32>;
    var fb_0 : vec3<f32>;
    var tb_2 : vec3<f32>;
    var stored_sum_0 : f32;
    var diss_sum_0 : f32;
    if(has_state_0)
    {
        var va0_0 : vec3<f32>;
        var wa0_0 : vec3<f32>;
        chunk_velocity_0(ca_0, &(va0_0), &(wa0_0));
        var vb0_0 : vec3<f32>;
        var wb0_0 : vec3<f32>;
        chunk_velocity_0(cb_0, &(vb0_0), &(wb0_0));
        e_3 = lane_1;
        count_3 = u32(0);
        var engaged_0 : u32 = u32(0);
        loop
        {
            if(e_3 < _S397)
            {
            }
            else
            {
                break;
            }
            var p_7 : vec3<f32>;
            var n_3 : vec3<f32>;
            var d_11 : f32;
            var _S398 : bool = pair_point_0(ba_1, bb_1, _S396, e_3, &(p_7), &(n_3), &(d_11));
            if(!_S398)
            {
                var _S399 : u32 = _S391 + e_3;
                if(!is_nan_0(contact_state_0[_S399].x))
                {
                    contact_state_0[_S399] = vec4<f32>((bitcast<f32>((u32(2143289344)))), 0.0f, 0.0f, 0.0f);
                }
                e_3 = e_3 + lanes_5;
                continue;
            }
            var _S400 : u32 = count_3 + u32(1);
            var _S401 : u32 = _S391 + e_3;
            var entry_2 : vec4<f32> = contact_state_0[_S401];
            if(is_nan_0(contact_state_0[_S401].x))
            {
                has_state_0 = true;
            }
            else
            {
                has_state_0 = (dot(entry_2.yzw, n_3)) < 0.99000000953674316f;
            }
            if(has_state_0)
            {
                if(d_11 > (2.0f * abs(dot(va0_0 + cross(wa0_0, p_7) - (vb0_0 + cross(wb0_0, p_7 - bb_1.center_1)), n_3)) * _S394 + 9.99999971718068537e-10f))
                {
                    stored_sum_0 = d_11;
                }
                else
                {
                    stored_sum_0 = 0.0f;
                }
                entry_2 = vec4<f32>(stored_sum_0, n_3);
            }
            entry_2[i32(0)] = min(entry_2.x, d_11);
            contact_state_0[_S401] = entry_2;
            var engaged_1 : u32;
            if((d_11 - entry_2.x) > 0.0f)
            {
                engaged_1 = engaged_0 + u32(1);
            }
            else
            {
                engaged_1 = engaged_0;
            }
            count_3 = _S400;
            engaged_0 = engaged_1;
            e_3 = e_3 + lanes_5;
        }
        var engaged_2 : u32 = pair_count_sum_0(engaged_0, lanes_5);
        if((pair_count_sum_0(count_3, lanes_5)) > u32(0))
        {
            var _S402 : f32 = contact_stiffness_0(chunks_0[ca_0].cmat_0.x, ba_1, chunks_0[cb_0].cmat_0.x, bb_1, bb_1.center_1 - ba_1.center_1) / max(f32(engaged_2), 10.0f);
            e_3 = lane_1;
            fa_1 = _S395;
            ta_2 = _S395;
            fb_0 = _S395;
            tb_2 = _S395;
            stored_sum_0 = 0.0f;
            diss_sum_0 = 0.0f;
            loop
            {
                if(e_3 < _S397)
                {
                }
                else
                {
                    break;
                }
                var p_8 : vec3<f32>;
                var n_4 : vec3<f32>;
                var d_12 : f32;
                var _S403 : bool = pair_point_0(ba_1, bb_1, _S396, e_3, &(p_8), &(n_4), &(d_12));
                if(!_S403)
                {
                    e_3 = e_3 + lanes_5;
                    continue;
                }
                var eff_0 : f32 = d_12 - contact_state_0[_S391 + e_3].x;
                if(eff_0 <= 0.0f)
                {
                    e_3 = e_3 + lanes_5;
                    continue;
                }
                var stored_7 : f32;
                var diss_3 : f32;
                var f_12 : vec3<f32> = penalty_force_1(_S402, _S392, _S393, eff_0, n_4, va0_0 + cross(wa0_0, p_8) - (vb0_0 + cross(wb0_0, p_8 - bb_1.center_1)), _S394, engaged_2, &(stored_7), &(diss_3));
                var ta_3 : vec3<f32> = ta_2 + cross(p_8, f_12);
                var _S404 : vec3<f32> = (vec3<f32>(0) - f_12);
                var fb_1 : vec3<f32> = fb_0 + _S404;
                var tb_3 : vec3<f32> = tb_2 + cross(p_8 - bb_1.center_1, _S404);
                var stored_sum_1 : f32 = stored_sum_0 + stored_7;
                var diss_sum_1 : f32 = diss_sum_0 + diss_3;
                fa_1 = fa_1 + f_12;
                ta_2 = ta_3;
                fb_0 = fb_1;
                tb_2 = tb_3;
                stored_sum_0 = stored_sum_1;
                diss_sum_0 = diss_sum_1;
                e_3 = e_3 + lanes_5;
            }
            var _S405 : vec3<f32> = pair_sum3_0(fa_1, lanes_5);
            var _S406 : vec3<f32> = pair_sum3_0(ta_2, lanes_5);
            var _S407 : vec3<f32> = pair_sum3_0(fb_0, lanes_5);
            var _S408 : vec3<f32> = pair_sum3_0(tb_2, lanes_5);
            var _S409 : f32 = pair_sum_0(stored_sum_0, lanes_5);
            var _S410 : f32 = pair_sum_0(diss_sum_0, lanes_5);
            has_state_0 = true;
            diss_sum_0 = _S410;
            fa_1 = _S405;
            ta_2 = _S406;
            fb_0 = _S407;
            tb_2 = _S408;
            stored_sum_0 = _S409;
        }
        else
        {
            has_state_0 = false;
            diss_sum_0 = 0.0f;
            fa_1 = _S395;
            ta_2 = _S395;
            fb_0 = _S395;
            tb_2 = _S395;
            stored_sum_0 = 0.0f;
        }
    }
    else
    {
        has_state_0 = false;
        diss_sum_0 = 0.0f;
        fa_1 = _S395;
        ta_2 = _S395;
        fb_0 = _S395;
        tb_2 = _S395;
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
        e_3 = lane_1;
        loop
        {
            if(e_3 < _S397)
            {
            }
            else
            {
                break;
            }
            contact_state_0[_S391 + e_3] = vec4<f32>((bitcast<f32>((u32(2143289344)))), 0.0f, 0.0f, 0.0f);
            e_3 = e_3 + lanes_5;
        }
    }
    if(lane_1 != u32(0))
    {
        return;
    }
    if(has_state_0)
    {
        var _S411 : f32 = ledger_1[i32(1)];
        var _S412 : f32 = ledger_1[i32(2)];
        comp_add1_4(&(_S411), &(_S412), diss_sum_0);
        ledger_1[i32(1)] = _S411;
        ledger_1[i32(2)] = _S412;
    }
    var _S413 : vec3<f32> = vec3<f32>(0.0f);
    if((any((fa_1 != _S413))))
    {
        loaded_0 = true;
    }
    else
    {
        loaded_0 = (any((ta_2 != _S413)));
    }
    if(loaded_0)
    {
        loaded_0 = true;
    }
    else
    {
        loaded_0 = (any((fb_0 != _S413)));
    }
    if(loaded_0)
    {
        loaded_0 = true;
    }
    else
    {
        loaded_0 = (any((tb_2 != _S413)));
    }
    var _S414 : bool;
    if(loaded_0)
    {
        _S414 = true;
    }
    else
    {
        _S414 = ((flags_1 & (u32(2)))) != u32(0);
    }
    if(_S414)
    {
        scratch_0[out_0] = vec4<f32>(fa_1, 0.0f);
        scratch_0[out_0 + u32(1)] = vec4<f32>(ta_2, 0.0f);
        scratch_0[out_0 + u32(2)] = vec4<f32>(fb_0, 0.0f);
        scratch_0[out_0 + u32(3)] = vec4<f32>(tb_2, 0.0f);
    }
    ledger_1[i32(0)] = stored_sum_0;
    if(has_state_0)
    {
        e_3 = u32(1);
    }
    else
    {
        e_3 = u32(0);
    }
    if(loaded_0)
    {
        count_3 = u32(2);
    }
    else
    {
        count_3 = u32(0);
    }
    ledger_1[i32(3)] = (bitcast<f32>(((e_3 | (count_3)))));
    scratch_0[params_0.ledger_base_0 + i_9] = ledger_1;
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
    var b_34 : Box_0;
    b_34.center_1 = center_3;
    b_34.axis0_0 = rotate_0(q_10, vec3<f32>(1.0f, 0.0f, 0.0f));
    b_34.axis1_0 = rotate_0(q_10, vec3<f32>(0.0f, 1.0f, 0.0f));
    b_34.axis2_0 = rotate_0(q_10, vec3<f32>(0.0f, 0.0f, 1.0f));
    b_34.half_2 = half_3;
    b_34.hull_at_0 = u32(0);
    b_34.hull_v_0 = u32(0);
    b_34.hull_f_0 = u32(0);
    return b_34;
}

fn impactor_slots_0( imp_1 : ptr<function, Impactor_std430_0>,  b_35 : Box_0) -> u32
{
    var _S415 : u32;
    if(((*imp_1).shape_0.x) == 0.0f)
    {
        _S415 = u32(1);
    }
    else
    {
        _S415 = sample_count_0(b_35) + u32(14);
    }
    return _S415;
}

fn impactor_slots_1( imp_2 : ptr<function, Impactor_std430_0>,  b_36 : Box_0) -> u32
{
    var _S416 : u32;
    if(((*imp_2).shape_0.x) == 0.0f)
    {
        _S416 = u32(1);
    }
    else
    {
        _S416 = sample_count_0(b_36) + u32(14);
    }
    return _S416;
}

fn sphere_contact_0( b_37 : Box_0,  center_4 : vec3<f32>,  radius_0 : f32,  point_0 : ptr<function, vec3<f32>>,  normal_7 : ptr<function, vec3<f32>>,  depth_5 : ptr<function, f32>) -> bool
{
    var _S417 : vec3<f32> = vec3<f32>(0.0f);
    (*point_0) = _S417;
    (*normal_7) = _S417;
    (*depth_5) = 0.0f;
    var r_12 : vec3<f32> = center_4 - b_37.center_1;
    var local_5 : vec3<f32> = vec3<f32>(dot(r_12, b_37.axis0_0), dot(r_12, b_37.axis1_0), dot(r_12, b_37.axis2_0));
    if((b_37.hull_v_0) != u32(0))
    {
        var face_3 : u32;
        var d_13 : f32 = hull_signed_distance_0(b_37, local_5, &(face_3));
        if(d_13 >= radius_0)
        {
            return false;
        }
        var _S418 : vec3<f32> = box_to_world_0(b_37, loads_0[b_37.hull_at_0 + b_37.hull_v_0 + face_3].xyz);
        (*normal_7) = _S418;
        (*point_0) = center_4 - _S418 * vec3<f32>(max(d_13, 0.0f));
        (*depth_5) = radius_0 - d_13;
        return true;
    }
    var q_11 : vec3<f32> = clamp(local_5, (vec3<f32>(0) - b_37.half_2), b_37.half_2);
    var d_14 : vec3<f32> = local_5 - q_11;
    var dist_0 : f32 = length(d_14);
    if(dist_0 > 9.999999960041972e-13f)
    {
        if(dist_0 >= radius_0)
        {
            return false;
        }
        var dn_0 : vec3<f32> = d_14 / vec3<f32>(dist_0);
        (*normal_7) = b_37.axis0_0 * vec3<f32>(dn_0.x) + b_37.axis1_0 * vec3<f32>(dn_0.y) + b_37.axis2_0 * vec3<f32>(dn_0.z);
        (*point_0) = b_37.center_1 + b_37.axis0_0 * vec3<f32>(q_11.x) + b_37.axis1_0 * vec3<f32>(q_11.y) + b_37.axis2_0 * vec3<f32>(q_11.z);
        (*depth_5) = radius_0 - dist_0;
        return true;
    }
    var inside_0 : f32;
    var n_5 : vec3<f32>;
    var _S419 : bool = penetration_0(b_37, center_4, &(inside_0), &(n_5));
    if(!_S419)
    {
        return false;
    }
    (*normal_7) = n_5;
    (*point_0) = center_4 - n_5 * vec3<f32>(min(radius_0, inside_0));
    (*depth_5) = radius_0 + inside_0;
    return true;
}

fn impactor_contact_0( imp_3 : ptr<function, Impactor_std430_0>,  crush_depth_0 : f32,  shrunk_0 : Box_0,  b_38 : Box_0,  j_2 : u32,  p_9 : ptr<function, vec3<f32>>,  n_6 : ptr<function, vec3<f32>>,  d_15 : ptr<function, f32>) -> bool
{
    var _S420 : vec3<f32> = vec3<f32>(0.0f);
    (*p_9) = _S420;
    (*n_6) = _S420;
    (*d_15) = 0.0f;
    var _S421 : vec4<f32> = (*imp_3).shape_0;
    if(((*imp_3).shape_0.x) == 0.0f)
    {
        var _S422 : bool = sphere_contact_0(b_38, _S420, _S421.y - crush_depth_0, &((*p_9)), &((*n_6)), &((*d_15)));
        if(!_S422)
        {
            return false;
        }
        (*n_6) = (vec3<f32>(0) - (*n_6));
        return true;
    }
    var cb_1 : u32 = sample_count_0(b_38);
    if(j_2 < cb_1)
    {
        var _S423 : vec3<f32> = sample_point_0(b_38, j_2);
        (*p_9) = _S423;
        var _S424 : bool = penetration_0(shrunk_0, _S423, &((*d_15)), &((*n_6)));
        return _S424;
    }
    var _S425 : vec3<f32> = sample_point_0(shrunk_0, j_2 - cb_1);
    (*p_9) = _S425;
    var _S426 : bool = penetration_0(b_38, _S425, &((*d_15)), &((*n_6)));
    if(!_S426)
    {
        return false;
    }
    (*n_6) = (vec3<f32>(0) - (*n_6));
    return true;
}

fn impactor_contact_1( imp_4 : ptr<function, Impactor_std430_0>,  crush_depth_1 : f32,  shrunk_1 : Box_0,  b_39 : Box_0,  j_3 : u32,  p_10 : ptr<function, vec3<f32>>,  n_7 : ptr<function, vec3<f32>>,  d_16 : ptr<function, f32>) -> bool
{
    var _S427 : vec3<f32> = vec3<f32>(0.0f);
    (*p_10) = _S427;
    (*n_7) = _S427;
    (*d_16) = 0.0f;
    var _S428 : vec4<f32> = (*imp_4).shape_0;
    if(((*imp_4).shape_0.x) == 0.0f)
    {
        var _S429 : bool = sphere_contact_0(b_39, _S427, _S428.y - crush_depth_1, &((*p_10)), &((*n_7)), &((*d_16)));
        if(!_S429)
        {
            return false;
        }
        (*n_7) = (vec3<f32>(0) - (*n_7));
        return true;
    }
    var cb_2 : u32 = sample_count_0(b_39);
    if(j_3 < cb_2)
    {
        var _S430 : vec3<f32> = sample_point_0(b_39, j_3);
        (*p_10) = _S430;
        var _S431 : bool = penetration_0(shrunk_1, _S430, &((*d_16)), &((*n_7)));
        return _S431;
    }
    var _S432 : vec3<f32> = sample_point_0(shrunk_1, j_3 - cb_2);
    (*p_10) = _S432;
    var _S433 : bool = penetration_0(b_39, _S432, &((*d_16)), &((*n_7)));
    if(!_S433)
    {
        return false;
    }
    (*n_7) = (vec3<f32>(0) - (*n_7));
    return true;
}

fn impactor_point_0( _S434 : u32) -> WorldPoint_0
{
    var wi_0 : WorldPoint_0;
    wi_0.hi_3 = impactors_0[_S434].position_1.xyz;
    wi_0.lo_3 = impactors_0[_S434].position_err_1.xyz;
    wi_0.rel_0 = vec3<f32>(0.0f);
    return wi_0;
}

fn impactor_box_1( _S435 : u32,  _S436 : vec3<f32>,  _S437 : vec3<f32>) -> Box_0
{
    var q_12 : Quat_0 = quat_of_0(impactors_0[_S435].rotation_1);
    var b_40 : Box_0;
    b_40.center_1 = _S436;
    b_40.axis0_0 = rotate_0(q_12, vec3<f32>(1.0f, 0.0f, 0.0f));
    b_40.axis1_0 = rotate_0(q_12, vec3<f32>(0.0f, 1.0f, 0.0f));
    b_40.axis2_0 = rotate_0(q_12, vec3<f32>(0.0f, 0.0f, 1.0f));
    b_40.half_2 = _S437;
    b_40.hull_at_0 = u32(0);
    b_40.hull_v_0 = u32(0);
    b_40.hull_f_0 = u32(0);
    return b_40;
}

fn impactor_shrunk_0( _S438 : u32,  _S439 : f32,  _S440 : Box_0) -> Box_0
{
    var shrunk_2 : Box_0 = _S440;
    shrunk_2.half_2 = _S440.half_2 - min(vec3<f32>(_S439), _S440.half_2 * vec3<f32>(0.5f));
    return shrunk_2;
}

fn impactor_contact_2( _S441 : u32,  _S442 : f32,  _S443 : Box_0,  _S444 : Box_0,  _S445 : u32,  _S446 : ptr<function, vec3<f32>>,  _S447 : ptr<function, vec3<f32>>,  _S448 : ptr<function, f32>) -> bool
{
    var _S449 : Impactor_std430_0 = impactors_0[_S441];
    var _S450 : vec3<f32> = vec3<f32>(0.0f);
    (*_S446) = _S450;
    (*_S447) = _S450;
    (*_S448) = 0.0f;
    if((_S449.shape_0.x) == 0.0f)
    {
        var _S451 : bool = sphere_contact_0(_S444, _S450, _S449.shape_0.y - _S442, &((*_S446)), &((*_S447)), &((*_S448)));
        if(!_S451)
        {
            return false;
        }
        (*_S447) = (vec3<f32>(0) - (*_S447));
        return true;
    }
    var cb_3 : u32 = sample_count_0(_S444);
    if(_S445 < cb_3)
    {
        var _S452 : vec3<f32> = sample_point_0(_S444, _S445);
        (*_S446) = _S452;
        var _S453 : bool = penetration_0(_S443, _S452, &((*_S448)), &((*_S447)));
        return _S453;
    }
    var _S454 : vec3<f32> = sample_point_0(_S443, _S445 - cb_3);
    (*_S446) = _S454;
    var _S455 : bool = penetration_0(_S444, _S454, &((*_S448)), &((*_S447)));
    if(!_S455)
    {
        return false;
    }
    (*_S447) = (vec3<f32>(0) - (*_S447));
    return true;
}

fn impactor_candidate_forces_0( k_18 : u32,  lane_2 : u32,  lanes_6 : u32)
{
    var _S456 : u32;
    var _S457 : bool;
    var _S458 : u32 = u32(3) * k_18;
    var at_4 : u32 = params_0.cand_index_0 + _S458;
    var c_21 : u32 = index_0[at_4];
    var slot_1 : u32 = index_0[at_4 + u32(1)];
    var _S459 : u32 = index_0[at_4 + u32(2)];
    var _S460 : Impactor_std430_0 = impactors_0[_S459];
    var _S461 : f32 = params_0.dt_0;
    var _S462 : vec3<f32> = vec3<f32>(0.0f);
    var data_1 : vec4<f32> = scratch_0[params_0.cand_base_0 + _S458];
    var hit_0 : bool;
    var f_sum_0 : vec3<f32>;
    var t_sum_0 : vec3<f32>;
    var imp_f_0 : vec3<f32>;
    var imp_t_0 : vec3<f32>;
    if((_S460.cand_0.z) == u32(0))
    {
        var rel_1 : vec3<f32> = world_diff_0(chunk_world_0(c_21), impactor_point_0(_S459));
        var _S463 : vec4<f32> = _S460.half_1;
        if(!((length(rel_1)) > (_S460.half_1.w + chunks_0[c_21].half_0.w)))
        {
            var _S464 : Box_0 = impactor_box_1(_S459, _S462, _S463.xyz);
            var b_41 : Box_0 = chunk_box_0(c_21, rel_1);
            var _S465 : vec4<f32> = _S460.mat_0;
            var kc_0 : f32 = contact_stiffness_0(_S460.mat_0.x, _S464, chunks_0[c_21].cmat_0.x, b_41, b_41.center_1 - _S464.center_1);
            var _S466 : vec4<f32> = _S460.geom_0;
            var _S467 : f32 = _S460.geom_0.y;
            var _S468 : Box_0 = impactor_shrunk_0(_S459, _S467, _S464);
            var j_4 : u32 = lane_2;
            var n_8 : u32 = u32(0);
            loop
            {
                var _S469 : u32 = impactor_slots_0(&(_S460), b_41);
                _S456 = _S469;
                if(j_4 < _S469)
                {
                }
                else
                {
                    break;
                }
                var p_11 : vec3<f32>;
                var nrm_0 : vec3<f32>;
                var dep_0 : f32;
                if(impactor_contact_2(_S459, _S467, _S468, b_41, j_4, &(p_11), &(nrm_0), &(dep_0)))
                {
                    n_8 = n_8 + u32(1);
                }
                j_4 = j_4 + lanes_6;
            }
            var n_9 : u32 = pair_count_sum_0(n_8, lanes_6);
            var _S470 : bool = (_S460.shape_0.x) == 0.0f;
            var _S471 : f32;
            if(_S470)
            {
                _S471 = kc_0;
            }
            else
            {
                _S471 = kc_0 / max(f32(n_9), 10.0f);
            }
            if(_S470)
            {
                j_4 = u32(1);
            }
            else
            {
                j_4 = n_9;
            }
            var m_4 : f32 = chunks_0[c_21].center_0.w;
            var _S472 : f32 = _S465.z;
            var _S473 : f32 = m_4 * _S472 / (m_4 + _S472);
            var _S474 : f32;
            if((params_0.pair_friction_0) >= 0.0f)
            {
                _S474 = params_0.pair_friction_0;
            }
            else
            {
                _S474 = min(_S465.y, chunks_0[c_21].cmat_0.y);
            }
            var _S475 : vec3<f32> = _S460.velocity_1.xyz + _S460.velocity_err_1.xyz;
            var vc_0 : vec3<f32>;
            var wc_0 : vec3<f32>;
            chunk_velocity_0(c_21, &(vc_0), &(wc_0));
            hit_0 = false;
            var j_5 : u32 = lane_2;
            f_sum_0 = _S462;
            t_sum_0 = _S462;
            imp_f_0 = _S462;
            imp_t_0 = _S462;
            var diss_sum_2 : f32 = 0.0f;
            loop
            {
                var _S476 : bool = n_9 > u32(0);
                _S457 = _S476;
                var _S477 : bool;
                if(_S476)
                {
                    _S477 = j_5 < _S456;
                }
                else
                {
                    _S477 = false;
                }
                if(_S477)
                {
                }
                else
                {
                    break;
                }
                var p_12 : vec3<f32>;
                var nrm_1 : vec3<f32>;
                var dep_1 : f32;
                var _S478 : bool = impactor_contact_0(&(_S460), _S467, _S468, b_41, j_5, &(p_12), &(nrm_1), &(dep_1));
                if(!_S478)
                {
                    j_5 = j_5 + lanes_6;
                    continue;
                }
                var stored_8 : f32;
                var diss_4 : f32;
                var f_13 : vec3<f32> = penalty_force_1(_S471, _S473, _S474, dep_1 * _S466.x, nrm_1, vc_0 + cross(wc_0, p_12 - b_41.center_1) - (_S475 + cross(_S460.angular_velocity_1.xyz, p_12)), _S461, j_4, &(stored_8), &(diss_4));
                var f_sum_1 : vec3<f32> = f_sum_0 + f_13;
                var t_sum_1 : vec3<f32> = t_sum_0 + cross(p_12 - b_41.center_1, f_13);
                var imp_f_1 : vec3<f32> = imp_f_0 - f_13;
                var imp_t_1 : vec3<f32> = imp_t_0 - cross(p_12, f_13);
                var diss_sum_3 : f32 = diss_sum_2 + diss_4;
                hit_0 = true;
                f_sum_0 = f_sum_1;
                t_sum_0 = t_sum_1;
                imp_f_0 = imp_f_1;
                imp_t_0 = imp_t_1;
                diss_sum_2 = diss_sum_3;
                j_5 = j_5 + lanes_6;
            }
            var _S479 : vec3<f32> = pair_sum3_0(f_sum_0, lanes_6);
            var _S480 : vec3<f32> = pair_sum3_0(t_sum_0, lanes_6);
            var _S481 : vec3<f32> = pair_sum3_0(imp_f_0, lanes_6);
            var _S482 : vec3<f32> = pair_sum3_0(imp_t_0, lanes_6);
            var diss_sum_4 : f32 = pair_sum_0(diss_sum_2, lanes_6);
            var _S483 : bool = pair_any_0(hit_0, lanes_6);
            if(_S457)
            {
                var _S484 : f32 = data_1[i32(2)];
                var _S485 : f32 = data_1[i32(3)];
                comp_add1_4(&(_S484), &(_S485), diss_sum_4);
                data_1[i32(2)] = _S484;
                data_1[i32(3)] = _S485;
            }
            hit_0 = _S483;
            f_sum_0 = _S479;
            t_sum_0 = _S480;
            imp_f_0 = _S481;
            imp_t_0 = _S482;
        }
        else
        {
            hit_0 = false;
            f_sum_0 = _S462;
            t_sum_0 = _S462;
            imp_f_0 = _S462;
            imp_t_0 = _S462;
        }
    }
    else
    {
        hit_0 = false;
        f_sum_0 = _S462;
        t_sum_0 = _S462;
        imp_f_0 = _S462;
        imp_t_0 = _S462;
    }
    if(lane_2 != u32(0))
    {
        return;
    }
    var _S486 : u32 = u32(2) * slot_1;
    scratch_0[params_0.slot_base_0 + _S486] = vec4<f32>(f_sum_0, 0.0f);
    scratch_0[params_0.slot_base_0 + _S486 + u32(1)] = vec4<f32>(t_sum_0, 0.0f);
    scratch_0[params_0.cand_base_0 + _S458] = data_1;
    scratch_0[params_0.cand_base_0 + _S458 + u32(1)] = vec4<f32>(imp_f_0, 0.0f);
    scratch_0[params_0.cand_base_0 + _S458 + u32(2)] = vec4<f32>(imp_t_0, 0.0f);
    if(hit_0)
    {
        sentinel_mark_0(u32(2));
    }
    return;
}

fn travel_check_0( c_22 : u32)
{
    var _S487 : vec4<u32> = chunks_0[c_22].cinfo_0;
    if((chunks_0[c_22].cinfo_0.z) == u32(0))
    {
        return;
    }
    var _S488 : u32 = u32(4) * c_22;
    var at_5 : u32 = params_0.cplan_base_0 + _S488;
    var start_hi_0 : vec4<f32> = scratch_0[at_5];
    var start_rot_0 : vec4<f32> = scratch_0[at_5 + u32(2)];
    var wp_0 : WorldPoint_0 = chunk_world_0(c_22);
    var moved_0 : vec3<f32> = wp_0.hi_3 - scratch_0[at_5].xyz + (wp_0.lo_3 - scratch_0[at_5 + u32(1)].xyz) + wp_0.rel_0;
    var _S489 : u32 = chunks_0[c_22].info_1.y;
    var _S490 : vec3<f32> = vec3<f32>((params_0.t_hi_0 - scratch_0[at_5 + u32(3)].w + (params_0.t_lo_0 - scratch_0[at_5 + u32(1)].w) + f32(islands_0[_S489].info_0.w - params_0.step_start_0) * params_0.dt_0));
    var _S491 : vec3<f32> = scratch_0[at_5 + u32(3)].xyz * _S490;
    var _S492 : u32 = _S487.w;
    var _S493 : vec3<f32>;
    if(((_S492 & (u32(2)))) != u32(0))
    {
        _S493 = vec3<f32>(0.5f) * params_0.gravity_0.xyz * _S490 * _S490;
    }
    else
    {
        _S493 = vec3<f32>(0.0f);
    }
    var travel_0 : f32 = length(moved_0 - (_S491 + _S493));
    var travel_1 : f32;
    if(((_S492 & (u32(1)))) != u32(0))
    {
        var th_4 : vec3<f32> = state_0[_S488 + u32(1)].xyz;
        var turn_0 : Quat_0 = quat_mul_0(quat_of_0(vec4<f32>((vec3<f32>(0) - start_rot_0.xyz), start_rot_0.w)), quat_mul_0(quat_of_0(islands_0[_S489].rotation_0), from_axis_angle_0(th_4, length(th_4))));
        travel_1 = travel_0 + 2.0f * atan2(length(vec3<f32>(turn_0.x_0, turn_0.y_0, turn_0.z_0)), abs(turn_0.w_0)) * chunks_0[c_22].half_0.w;
    }
    else
    {
        travel_1 = travel_0;
    }
    if(travel_1 > (start_hi_0.w))
    {
        islands_0[params_0.halt_index_0].info_0[i32(2)] = ((islands_0[params_0.halt_index_0].info_0.z) | (u32(1)));
        sentinel_mark_0(u32(1048576));
    }
    return;
}

fn impactor_travel_check_0( ii_0 : u32)
{
    var at_6 : u32 = params_0.iplan_base_0 + u32(4) * ii_0;
    var start_hi_1 : vec4<f32> = scratch_0[at_6];
    var _S494 : vec4<u32> = impactors_0[ii_0].cand_0;
    var _S495 : bool;
    if((impactors_0[ii_0].cand_0.z) != u32(0))
    {
        _S495 = true;
    }
    else
    {
        _S495 = (start_hi_1.w) <= 0.0f;
    }
    if(_S495)
    {
        return;
    }
    var _S496 : vec3<f32> = vec3<f32>((params_0.t_hi_0 - scratch_0[at_6 + u32(3)].w + (params_0.t_lo_0 - scratch_0[at_6 + u32(1)].w) + f32(_S494.w - params_0.step_start_0) * params_0.dt_0));
    var turn_1 : Quat_0 = quat_mul_0(quat_of_0(vec4<f32>((vec3<f32>(0) - scratch_0[at_6 + u32(2)].xyz), scratch_0[at_6 + u32(2)].w)), quat_of_0(impactors_0[ii_0].rotation_1));
    if((length(impactors_0[ii_0].position_1.xyz - start_hi_1.xyz + (impactors_0[ii_0].position_err_1.xyz - scratch_0[at_6 + u32(1)].xyz) - (scratch_0[at_6 + u32(3)].xyz * _S496 + vec3<f32>(0.5f) * params_0.gravity_0.xyz * _S496 * _S496)) + 2.0f * atan2(length(vec3<f32>(turn_1.x_0, turn_1.y_0, turn_1.z_0)), abs(turn_1.w_0)) * impactors_0[ii_0].half_1.w) > (start_hi_1.w))
    {
        islands_0[params_0.halt_index_0].info_0[i32(2)] = ((islands_0[params_0.halt_index_0].info_0.z) | (u32(1)));
        sentinel_mark_0(u32(1048576));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn contact_forces(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var i_10 : u32 = id_0.x;
    var lanes_7 : u32 = pair_lane_count_0();
    var pair_threads_0 : u32 = params_0.pair_count_0 * lanes_7;
    if(i_10 < pair_threads_0)
    {
        if(!pair_any_0(stopped_0(), lanes_7))
        {
            var _S497 : u32 = i_10 / lanes_7;
            var _S498 : u32 = i_10 % lanes_7;
            pair_contact_0(_S497, _S498, lanes_7);
        }
        return;
    }
    var j_6 : u32 = i_10 - pair_threads_0;
    var cand_threads_0 : u32 = params_0.cand_count_0 * lanes_7;
    if(j_6 < cand_threads_0)
    {
        if(!pair_any_0(stopped_0(), lanes_7))
        {
            var _S499 : u32 = j_6 / lanes_7;
            var _S500 : u32 = j_6 % lanes_7;
            impactor_candidate_forces_0(_S499, _S500, lanes_7);
        }
        return;
    }
    var _S501 : u32 = j_6 - cand_threads_0;
    var _S502 : bool;
    if(_S501 < (params_0.chunk_count_0))
    {
        _S502 = !stopped_0();
    }
    else
    {
        _S502 = false;
    }
    if(_S502)
    {
        travel_check_0(_S501);
    }
    else
    {
        if((_S501 - params_0.chunk_count_0) < (params_0.impactor_count_0))
        {
            _S502 = !stopped_0();
        }
        else
        {
            _S502 = false;
        }
        if(_S502)
        {
            impactor_travel_check_0(_S501 - params_0.chunk_count_0);
        }
    }
    return;
}

fn impactor_contact_count_0( _S503 : u32,  _S504 : f32,  _S505 : Box_0,  _S506 : Box_0) -> u32
{
    var _S507 : Impactor_std430_0 = impactors_0[_S503];
    var j_7 : u32 = u32(0);
    var count_4 : u32 = u32(0);
    loop
    {
        var _S508 : u32 = impactor_slots_0(&(_S507), _S506);
        if(j_7 < _S508)
        {
        }
        else
        {
            break;
        }
        var p_13 : vec3<f32>;
        var n_10 : vec3<f32>;
        var d_17 : f32;
        if(impactor_contact_2(_S503, _S504, _S505, _S506, j_7, &(p_13), &(n_10), &(d_17)))
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
    var k_19 : u32 = id_1.x;
    var _S509 : bool;
    if(k_19 >= (params_0.cand_count_0))
    {
        _S509 = true;
    }
    else
    {
        _S509 = stopped_0();
    }
    if(_S509)
    {
        return;
    }
    var _S510 : u32 = u32(3) * k_19;
    var at_7 : u32 = params_0.cand_index_0 + _S510;
    var c_23 : u32 = index_0[at_7];
    var _S511 : u32 = index_0[at_7 + u32(2)];
    var _S512 : Impactor_std430_0 = impactors_0[_S511];
    if((_S512.cand_0.z) == u32(0))
    {
        _S509 = (_S512.crush_0.x) > 0.0f;
    }
    else
    {
        _S509 = false;
    }
    var total_1 : f32;
    var ksum_0 : f32;
    if(_S509)
    {
        var rel_2 : vec3<f32> = world_diff_0(chunk_world_0(c_23), impactor_point_0(_S511));
        var _S513 : vec4<f32> = _S512.half_1;
        if(!((length(rel_2)) > (_S512.half_1.w + chunks_0[c_23].half_0.w)))
        {
            var _S514 : Box_0 = impactor_box_1(_S511, vec3<f32>(0.0f), _S513.xyz);
            var b_42 : Box_0 = chunk_box_0(c_23, rel_2);
            var kc_1 : f32 = contact_stiffness_0(_S512.mat_0.x, _S514, chunks_0[c_23].cmat_0.x, b_42, b_42.center_1 - _S514.center_1);
            var _S515 : f32 = _S512.crush_0.w;
            var _S516 : Box_0 = impactor_shrunk_0(_S511, _S515, _S514);
            var _S517 : u32 = impactor_contact_count_0(_S511, _S515, _S516, b_42);
            var _S518 : f32;
            if((_S512.shape_0.x) == 0.0f)
            {
                _S518 = kc_1;
            }
            else
            {
                _S518 = kc_1 / max(f32(_S517), 10.0f);
            }
            var j_8 : u32 = u32(0);
            total_1 = 0.0f;
            ksum_0 = 0.0f;
            loop
            {
                if(_S517 > u32(0))
                {
                    var _S519 : u32 = impactor_slots_1(&(_S512), b_42);
                    _S509 = j_8 < _S519;
                }
                else
                {
                    _S509 = false;
                }
                if(_S509)
                {
                }
                else
                {
                    break;
                }
                var p_14 : vec3<f32>;
                var nrm_2 : vec3<f32>;
                var dep_2 : f32;
                var _S520 : bool = impactor_contact_1(&(_S512), _S515, _S516, b_42, j_8, &(p_14), &(nrm_2), &(dep_2));
                if(!_S520)
                {
                    j_8 = j_8 + u32(1);
                    continue;
                }
                var ksum_1 : f32 = ksum_0 + _S518;
                total_1 = total_1 + _S518 * dep_2;
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
    scratch_0[params_0.cand_base_0 + _S510] = vec4<f32>(total_1, ksum_0, scratch_0[params_0.cand_base_0 + _S510].z, scratch_0[params_0.cand_base_0 + _S510].w);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn impactor_crush(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var ii_1 : u32 = group_0.x;
    var tid_12 : u32 = thread_0.x;
    var _S521 : bool;
    if(ii_1 >= (params_0.impactor_count_0))
    {
        _S521 = true;
    }
    else
    {
        _S521 = stopped_0();
    }
    if(_S521)
    {
        return;
    }
    var imp_5 : Impactor_0;
    imp_5.position_1 = impactors_0[ii_1].position_1;
    imp_5.position_err_1 = impactors_0[ii_1].position_err_1;
    imp_5.velocity_1 = impactors_0[ii_1].velocity_1;
    imp_5.velocity_err_1 = impactors_0[ii_1].velocity_err_1;
    imp_5.angular_velocity_1 = impactors_0[ii_1].angular_velocity_1;
    imp_5.rotation_1 = impactors_0[ii_1].rotation_1;
    imp_5.inertia0_2 = impactors_0[ii_1].inertia0_2;
    imp_5.inertia1_2 = impactors_0[ii_1].inertia1_2;
    imp_5.inertia2_2 = impactors_0[ii_1].inertia2_2;
    imp_5.inv0_2 = impactors_0[ii_1].inv0_2;
    imp_5.inv1_2 = impactors_0[ii_1].inv1_2;
    imp_5.inv2_2 = impactors_0[ii_1].inv2_2;
    imp_5.shape_0 = impactors_0[ii_1].shape_0;
    imp_5.half_1 = impactors_0[ii_1].half_1;
    imp_5.mat_0 = impactors_0[ii_1].mat_0;
    imp_5.crush_0 = impactors_0[ii_1].crush_0;
    imp_5.geom_0 = impactors_0[ii_1].geom_0;
    imp_5.rotation_err_1 = impactors_0[ii_1].rotation_err_1;
    imp_5.ledger_0 = impactors_0[ii_1].ledger_0;
    imp_5.cand_0 = impactors_0[ii_1].cand_0;
    imp_5.momentum_1 = impactors_0[ii_1].momentum_1;
    imp_5.momentum_err_1 = impactors_0[ii_1].momentum_err_1;
    var _S522 : vec4<f32> = vec4<f32>(0.0f);
    var shares_0 : vec4<f32> = _S522;
    var unused_2 : vec4<f32> = _S522;
    var k_20 : u32 = imp_5.cand_0.x + tid_12;
    loop
    {
        if(k_20 < (imp_5.cand_0.y))
        {
        }
        else
        {
            break;
        }
        shares_0 = shares_0 + scratch_0[params_0.cand_base_0 + u32(3) * k_20];
        k_20 = k_20 + u32(256);
    }
    group_sum2_1(tid_12, &(shares_0), &(unused_2));
    if(tid_12 != u32(0))
    {
        _S521 = true;
    }
    else
    {
        _S521 = (imp_5.cand_0.z) != u32(0);
    }
    if(_S521)
    {
        return;
    }
    imp_5.geom_0 = vec4<f32>(1.0f, imp_5.crush_0.w, 0.0f, 0.0f);
    var total_2 : f32 = shares_0.x;
    if((imp_5.crush_0.x) > 0.0f)
    {
        _S521 = (imp_5.crush_0.z) < (imp_5.crush_0.y);
    }
    else
    {
        _S521 = false;
    }
    if(_S521)
    {
        _S521 = total_2 > (imp_5.crush_0.x);
    }
    else
    {
        _S521 = false;
    }
    if(_S521)
    {
        var extra_0 : f32 = (total_2 - imp_5.crush_0.x) / shares_0.y;
        imp_5.crush_0[i32(3)] = imp_5.crush_0[i32(3)] + extra_0;
        imp_5.crush_0[i32(2)] = imp_5.crush_0[i32(2)] + imp_5.crush_0.x * extra_0;
        var _S523 : f32 = imp_5.crush_0.x * extra_0;
        var _S524 : f32 = imp_5.ledger_0[i32(2)];
        var _S525 : f32 = imp_5.ledger_0[i32(3)];
        comp_add1_0(&(_S524), &(_S525), _S523);
        imp_5.ledger_0[i32(2)] = _S524;
        imp_5.ledger_0[i32(3)] = _S525;
        var _S526 : f32 = imp_5.crush_0.x * extra_0;
        var _S527 : f32 = imp_5.ledger_0[i32(0)];
        var _S528 : f32 = imp_5.ledger_0[i32(1)];
        comp_add1_0(&(_S527), &(_S528), _S526);
        imp_5.ledger_0[i32(0)] = _S527;
        imp_5.ledger_0[i32(1)] = _S528;
        imp_5.geom_0[i32(0)] = imp_5.crush_0.x / total_2;
    }
    impactors_0[ii_1].position_1 = imp_5.position_1;
    impactors_0[ii_1].position_err_1 = imp_5.position_err_1;
    impactors_0[ii_1].velocity_1 = imp_5.velocity_1;
    impactors_0[ii_1].velocity_err_1 = imp_5.velocity_err_1;
    impactors_0[ii_1].angular_velocity_1 = imp_5.angular_velocity_1;
    impactors_0[ii_1].rotation_1 = imp_5.rotation_1;
    impactors_0[ii_1].inertia0_2 = imp_5.inertia0_2;
    impactors_0[ii_1].inertia1_2 = imp_5.inertia1_2;
    impactors_0[ii_1].inertia2_2 = imp_5.inertia2_2;
    impactors_0[ii_1].inv0_2 = imp_5.inv0_2;
    impactors_0[ii_1].inv1_2 = imp_5.inv1_2;
    impactors_0[ii_1].inv2_2 = imp_5.inv2_2;
    impactors_0[ii_1].shape_0 = imp_5.shape_0;
    impactors_0[ii_1].half_1 = imp_5.half_1;
    impactors_0[ii_1].mat_0 = imp_5.mat_0;
    impactors_0[ii_1].crush_0 = imp_5.crush_0;
    impactors_0[ii_1].geom_0 = imp_5.geom_0;
    impactors_0[ii_1].rotation_err_1 = imp_5.rotation_err_1;
    impactors_0[ii_1].ledger_0 = imp_5.ledger_0;
    impactors_0[ii_1].cand_0 = imp_5.cand_0;
    impactors_0[ii_1].momentum_1 = imp_5.momentum_1;
    impactors_0[ii_1].momentum_err_1 = imp_5.momentum_err_1;
    return;
}

struct WideGroup_0
{
     island_2 : u32,
     begin_0 : u32,
     end_0 : u32,
     first_1 : u32,
};

fn wide_group_0( table_0 : u32,  g_2 : u32) -> WideGroup_0
{
    var w_8 : WideGroup_0;
    var _S529 : u32 = table_0 + u32(4) * g_2;
    w_8.island_2 = index_0[_S529];
    w_8.begin_0 = index_0[_S529 + u32(1)];
    w_8.end_0 = index_0[_S529 + u32(2)];
    w_8.first_1 = index_0[_S529 + u32(3)];
    return w_8;
}

fn contact_stopped_1( isl_11 : ptr<function, Island_std430_0>) -> bool
{
    var _S530 : vec4<u32> = islands_0[params_0.halt_index_0].info_0;
    var _S531 : bool;
    if((((islands_0[params_0.halt_index_0].info_0.z) & (u32(1)))) != u32(0))
    {
        _S531 = true;
    }
    else
    {
        var _S532 : u32 = _S530.y;
        if(_S532 != u32(0))
        {
            _S531 = _S532 <= ((*isl_11).info_0.w);
        }
        else
        {
            _S531 = false;
        }
    }
    return _S531;
}

fn contact_stopped_2( isl_12 : ptr<function, Island_std430_0>) -> bool
{
    var _S533 : vec4<u32> = islands_0[params_0.halt_index_0].info_0;
    var _S534 : bool;
    if((((islands_0[params_0.halt_index_0].info_0.z) & (u32(1)))) != u32(0))
    {
        _S534 = true;
    }
    else
    {
        var _S535 : u32 = _S533.y;
        if(_S535 != u32(0))
        {
            _S534 = _S535 <= ((*isl_12).info_0.w);
        }
        else
        {
            _S534 = false;
        }
    }
    return _S534;
}

fn wide_runs_0( isl_13 : ptr<function, Island_std430_0>) -> bool
{
    var _S536 : vec4<u32> = (*isl_13).info_0;
    var _S537 : bool;
    if(((((*isl_13).info_0.z) & (u32(1)))) != u32(0))
    {
        _S537 = true;
    }
    else
    {
        _S537 = (_S536.y) == u32(0);
    }
    if(_S537)
    {
        return false;
    }
    if((((_S536.x) & (u32(4)))) == u32(0))
    {
        _S537 = true;
    }
    else
    {
        var _S538 : bool = contact_stopped_1(&((*isl_13)));
        _S537 = !_S538;
    }
    return _S537;
}

fn wide_runs_1( isl_14 : ptr<function, Island_std430_0>) -> bool
{
    var _S539 : vec4<u32> = (*isl_14).info_0;
    var _S540 : bool;
    if(((((*isl_14).info_0.z) & (u32(1)))) != u32(0))
    {
        _S540 = true;
    }
    else
    {
        _S540 = (_S539.y) == u32(0);
    }
    if(_S540)
    {
        return false;
    }
    if((((_S539.x) & (u32(4)))) == u32(0))
    {
        _S540 = true;
    }
    else
    {
        var _S541 : bool = contact_stopped_2(&((*isl_14)));
        _S540 = !_S541;
    }
    return _S540;
}

var<workgroup> g_wide_run_0 : u32;

fn wide_enter_0( tid_13 : u32,  isl_15 : ptr<function, Island_std430_0>) -> bool
{
    seen_clear_0(tid_13);
    if(tid_13 == u32(0))
    {
        var _S542 : bool = wide_runs_1(&((*isl_15)));
        var _S543 : i32;
        if(_S542)
        {
            _S543 = i32(1);
        }
        else
        {
            _S543 = i32(0);
        }
        g_wide_run_0 = u32(_S543);
    }
    workgroupBarrier();
    return g_wide_run_0 != u32(0);
}

fn wide_step_0( isl_16 : ptr<function, Island_std430_0>) -> u32
{
    return (*isl_16).info_0.w - params_0.step_start_0;
}

fn wide_step_1( isl_17 : ptr<function, Island_std430_0>) -> u32
{
    return (*isl_17).info_0.w - params_0.step_start_0;
}

fn wide_step_2( isl_18 : ptr<function, Island_std430_0>) -> u32
{
    return (*isl_18).info_0.w - params_0.step_start_0;
}

fn wide_step_3( isl_19 : Island_0) -> u32
{
    return isl_19.info_0.w - params_0.step_start_0;
}

fn wide_enter_1( _S544 : u32,  _S545 : u32) -> bool
{
    var _S546 : Island_std430_0 = islands_0[_S545];
    seen_clear_0(_S544);
    if(_S544 == u32(0))
    {
        var _S547 : bool = wide_runs_0(&(_S546));
        var _S548 : i32;
        if(_S547)
        {
            _S548 = i32(1);
        }
        else
        {
            _S548 = i32(0);
        }
        g_wide_run_0 = u32(_S548);
    }
    workgroupBarrier();
    return g_wide_run_0 != u32(0);
}

@compute
@workgroup_size(256, 1, 1)
fn wide_wake(@builtin(workgroup_id) group_1 : vec3<u32>, @builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var tid_14 : u32 = thread_1.x;
    var _S549 : u32 = group_1.x;
    var wg_0 : WideGroup_0 = wide_group_0(params_0.wide_chunk_table_0, _S549);
    if(_S549 != (wg_0.first_1))
    {
        return;
    }
    var _S550 : Island_std430_0 = islands_0[wg_0.island_2];
    var _S551 : vec4<u32> = _S550.info_0;
    var _S552 : u32 = _S550.info_0.x;
    var _S553 : bool;
    if(((_S552 & (u32(16)))) == u32(0))
    {
        _S553 = true;
    }
    else
    {
        var _S554 : bool = wide_enter_1(tid_14, wg_0.island_2);
        _S553 = !_S554;
    }
    if(_S553)
    {
        return;
    }
    var _S555 : Quat_0 = quat_of_0(_S550.rotation_0);
    var _S556 : bool = ((_S552 & (u32(4)))) != u32(0);
    var _S557 : vec3<f32> = vec3<f32>(0.0f);
    var norm_2 : vec3<f32> = _S557;
    var unused_3 : vec3<f32> = _S557;
    var _S558 : vec4<u32> = _S550.range_0;
    var c_24 : u32 = _S550.range_0.x + tid_14;
    loop
    {
        if(c_24 < (_S558.y))
        {
        }
        else
        {
            break;
        }
        var _S559 : u32 = wide_step_2(&(_S550));
        var _S560 : f32 = settled_chunk_load_0(c_24, _S555, _S559, params_0.dt_0, _S556);
        norm_2[i32(0)] = norm_2[i32(0)] + _S560;
        c_24 = c_24 + u32(256);
    }
    group_sum3_3(tid_14, &(norm_2), &(unused_3));
    if(tid_14 != u32(0))
    {
        return;
    }
    var cov_10 : u32 = (u32(1024) | ((seen_bits_0())));
    if((params_0.solve_mode_0) == u32(1))
    {
        _S553 = (abs(norm_2.x - _S550.energy_0.z)) > (_S550.energy_0.w);
    }
    else
    {
        _S553 = false;
    }
    var cov_11 : u32;
    if(_S553)
    {
        islands_0[wg_0.island_2].info_0[i32(0)] = (_S552 & (u32(4294967279)));
        islands_0[wg_0.island_2].info_0[i32(2)] = ((_S551.z) | (u32(4)));
        cov_11 = (cov_10 | (u32(2048)));
    }
    else
    {
        cov_11 = cov_10;
    }
    cov_mark_0(wg_0.island_2, _S550.coverage_0, cov_11);
    return;
}

fn wide_partials_0( tid_15 : u32,  first_2 : u32,  count_5 : u32,  p_15 : u32,  a_24 : ptr<function, vec3<f32>>,  b_43 : ptr<function, vec3<f32>>)
{
    var _S561 : vec4<f32> = vec4<f32>(0.0f);
    var x_21 : vec4<f32> = _S561;
    var y_5 : vec4<f32> = _S561;
    var s_9 : u32 = tid_15;
    loop
    {
        if(s_9 < count_5)
        {
        }
        else
        {
            break;
        }
        var _S562 : u32 = u32(8) * (first_2 + s_9);
        x_21 = x_21 + scratch_0[params_0.wide_base_0 + _S562 + p_15];
        y_5 = y_5 + scratch_0[params_0.wide_base_0 + _S562 + p_15 + u32(1)];
        s_9 = s_9 + u32(256);
    }
    group_sum2_0(tid_15, &(x_21), &(y_5));
    (*a_24) = x_21.xyz;
    (*b_43) = y_5.xyz;
    return;
}

fn wide_partials_1( tid_16 : u32,  first_3 : u32,  count_6 : u32,  p_16 : u32,  a_25 : ptr<function, vec3<f32>>,  b_44 : ptr<function, vec3<f32>>)
{
    var _S563 : vec4<f32> = vec4<f32>(0.0f);
    var x_22 : vec4<f32> = _S563;
    var y_6 : vec4<f32> = _S563;
    var s_10 : u32 = tid_16;
    loop
    {
        if(s_10 < count_6)
        {
        }
        else
        {
            break;
        }
        var _S564 : u32 = u32(8) * (first_3 + s_10);
        x_22 = x_22 + scratch_0[params_0.wide_base_0 + _S564 + p_16];
        y_6 = y_6 + scratch_0[params_0.wide_base_0 + _S564 + p_16 + u32(1)];
        s_10 = s_10 + u32(256);
    }
    group_sum2_1(tid_16, &(x_22), &(y_6));
    (*a_25) = x_22.xyz;
    (*b_44) = y_6.xyz;
    return;
}

fn drift_angular_1( c_25 : u32,  wcom_4 : vec3<f32>,  tr_5 : vec3<f32>,  dv_5 : vec3<f32>,  lu_3 : ptr<function, vec3<f32>>,  lv_3 : ptr<function, vec3<f32>>)
{
    var r_13 : vec3<f32> = chunks_0[c_25].center_0.xyz - wcom_4;
    var _S565 : u32 = u32(4) * c_25;
    var _S566 : vec3<f32> = vec3<f32>(chunks_0[c_25].center_0.w);
    var _S567 : vec4<f32> = chunks_0[c_25].inertia0_1;
    var _S568 : vec4<f32> = chunks_0[c_25].inertia1_1;
    var _S569 : vec4<f32> = chunks_0[c_25].inertia2_1;
    var _S570 : vec3<f32> = vec3<f32>(chunks_0[c_25].scale_0.x);
    (*lu_3) = (*lu_3) + (cross(r_13, state_0[_S565].xyz - tr_5) * _S566 + rows_mul_0(chunks_0[c_25].inertia0_1, chunks_0[c_25].inertia1_1, chunks_0[c_25].inertia2_1, state_0[_S565 + u32(1)].xyz)) * _S570;
    (*lv_3) = (*lv_3) + (cross(r_13, state_0[_S565 + u32(2)].xyz - dv_5) * _S566 + rows_mul_0(_S567, _S568, _S569, state_0[_S565 + u32(3)].xyz)) * _S570;
    return;
}

fn wide_store_0( slot_2 : u32,  p_17 : u32,  a_26 : vec3<f32>,  b_45 : vec3<f32>)
{
    var _S571 : u32 = u32(8) * slot_2;
    scratch_0[params_0.wide_base_0 + _S571 + p_17] = vec4<f32>(a_26, 0.0f);
    scratch_0[params_0.wide_base_0 + _S571 + p_17 + u32(1)] = vec4<f32>(b_45, 0.0f);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn wide_drift(@builtin(workgroup_id) group_2 : vec3<u32>, @builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var tid_17 : u32 = thread_2.x;
    var _S572 : u32 = group_2.x;
    var wg_1 : WideGroup_0 = wide_group_0(params_0.wide_chunk_table_0, _S572);
    var isl_20 : Island_std430_0 = islands_0[wg_1.island_2];
    var _S573 : bool;
    if((((islands_0[wg_1.island_2].info_0.x) & (u32(17)))) != u32(0))
    {
        _S573 = true;
    }
    else
    {
        var _S574 : bool = wide_enter_1(tid_17, wg_1.island_2);
        _S573 = !_S574;
    }
    if(_S573)
    {
        return;
    }
    var tu_3 : vec3<f32>;
    var pv_3 : vec3<f32>;
    wide_partials_1(tid_17, wg_1.first_1, isl_20.wide_groups_0, u32(2), &(tu_3), &(pv_3));
    var _S575 : vec3<f32> = vec3<f32>(isl_20.wcom_0.w);
    var tr_6 : vec3<f32> = tu_3 / _S575;
    var dv_6 : vec3<f32> = pv_3 / _S575;
    var _S576 : vec3<f32> = vec3<f32>(0.0f);
    var lu_4 : vec3<f32> = _S576;
    var lv_4 : vec3<f32> = _S576;
    var c_26 : u32 = wg_1.begin_0 + tid_17;
    if(c_26 < (wg_1.end_0))
    {
        drift_angular_1(c_26, isl_20.wcom_0.xyz, tr_6, dv_6, &(lu_4), &(lv_4));
    }
    group_sum3_3(tid_17, &(lu_4), &(lv_4));
    if(tid_17 == u32(0))
    {
        wide_store_0(_S572, u32(4), lu_4, lv_4);
    }
    return;
}

fn rigid_of_1( isl_21 : ptr<function, Island_std430_0>) -> Rigid_0
{
    var rg_9 : Rigid_0;
    rg_9.rot_0 = quat_of_0((*isl_21).rotation_0);
    rg_9.rot_err_0 = (*isl_21).rotation_err_0;
    rg_9.pos_0 = (*isl_21).position_0.xyz;
    rg_9.pos_err_0 = (*isl_21).position_err_0.xyz;
    rg_9.vel_0 = (*isl_21).velocity_0.xyz;
    rg_9.vel_err_0 = (*isl_21).velocity_err_0.xyz;
    rg_9.w_1 = (*isl_21).angular_velocity_0.xyz;
    rg_9.l_0 = (*isl_21).momentum_0.xyz;
    rg_9.l_err_0 = (*isl_21).momentum_err_0.xyz;
    var _S577 : vec3<f32> = vec3<f32>(0.0f);
    rg_9.torque_0 = _S577;
    rg_9.a_0 = _S577;
    rg_9.alpha_0 = _S577;
    return rg_9;
}

fn rigid_of_2( isl_22 : ptr<function, Island_std430_0>) -> Rigid_0
{
    var rg_10 : Rigid_0;
    rg_10.rot_0 = quat_of_0((*isl_22).rotation_0);
    rg_10.rot_err_0 = (*isl_22).rotation_err_0;
    rg_10.pos_0 = (*isl_22).position_0.xyz;
    rg_10.pos_err_0 = (*isl_22).position_err_0.xyz;
    rg_10.vel_0 = (*isl_22).velocity_0.xyz;
    rg_10.vel_err_0 = (*isl_22).velocity_err_0.xyz;
    rg_10.w_1 = (*isl_22).angular_velocity_0.xyz;
    rg_10.l_0 = (*isl_22).momentum_0.xyz;
    rg_10.l_err_0 = (*isl_22).momentum_err_0.xyz;
    var _S578 : vec3<f32> = vec3<f32>(0.0f);
    rg_10.torque_0 = _S578;
    rg_10.a_0 = _S578;
    rg_10.alpha_0 = _S578;
    return rg_10;
}

fn rigid_of_3( isl_23 : ptr<function, Island_std430_0>) -> Rigid_0
{
    var rg_11 : Rigid_0;
    rg_11.rot_0 = quat_of_0((*isl_23).rotation_0);
    rg_11.rot_err_0 = (*isl_23).rotation_err_0;
    rg_11.pos_0 = (*isl_23).position_0.xyz;
    rg_11.pos_err_0 = (*isl_23).position_err_0.xyz;
    rg_11.vel_0 = (*isl_23).velocity_0.xyz;
    rg_11.vel_err_0 = (*isl_23).velocity_err_0.xyz;
    rg_11.w_1 = (*isl_23).angular_velocity_0.xyz;
    rg_11.l_0 = (*isl_23).momentum_0.xyz;
    rg_11.l_err_0 = (*isl_23).momentum_err_0.xyz;
    var _S579 : vec3<f32> = vec3<f32>(0.0f);
    rg_11.torque_0 = _S579;
    rg_11.a_0 = _S579;
    rg_11.alpha_0 = _S579;
    return rg_11;
}

fn rigid_acceleration_1( isl_24 : ptr<function, Island_std430_0>,  rg_12 : ptr<function, Rigid_0>,  f_14 : vec3<f32>,  t_16 : vec3<f32>)
{
    var iw_w_1 : vec3<f32> = world_mul_0((*rg_12).rot_0, (*isl_24).inertia0_0, (*isl_24).inertia1_0, (*isl_24).inertia2_0, (*rg_12).w_1);
    (*rg_12).a_0 = f_14 / vec3<f32>((*isl_24).com_0.w);
    (*rg_12).alpha_0 = world_mul_0((*rg_12).rot_0, (*isl_24).inv0_0, (*isl_24).inv1_0, (*isl_24).inv2_0, t_16 - cross((*rg_12).w_1, iw_w_1));
    (*rg_12).torque_0 = t_16;
    return;
}

fn rigid_acceleration_2( isl_25 : ptr<function, Island_std430_0>,  rg_13 : ptr<function, Rigid_0>,  f_15 : vec3<f32>,  t_17 : vec3<f32>)
{
    var iw_w_2 : vec3<f32> = world_mul_0((*rg_13).rot_0, (*isl_25).inertia0_0, (*isl_25).inertia1_0, (*isl_25).inertia2_0, (*rg_13).w_1);
    (*rg_13).a_0 = f_15 / vec3<f32>((*isl_25).com_0.w);
    (*rg_13).alpha_0 = world_mul_0((*rg_13).rot_0, (*isl_25).inv0_0, (*isl_25).inv1_0, (*isl_25).inv2_0, t_17 - cross((*rg_13).w_1, iw_w_2));
    (*rg_13).torque_0 = t_17;
    return;
}

fn wide_rigid_frame_0( tid_18 : u32,  isl_26 : ptr<function, Island_std430_0>,  wg_2 : WideGroup_0) -> Rigid_0
{
    var _S580 : Rigid_0 = rigid_of_1(&((*isl_26)));
    var rg_14 : Rigid_0 = _S580;
    if(((((*isl_26).info_0.x) & (u32(1)))) == u32(0))
    {
        var f_16 : vec3<f32>;
        var t_18 : vec3<f32>;
        wide_partials_0(tid_18, wg_2.first_1, (*isl_26).wide_groups_0, u32(0), &(f_16), &(t_18));
        rigid_acceleration_1(&((*isl_26)), &(rg_14), f_16, t_18);
    }
    return rg_14;
}

fn wide_rigid_frame_1( tid_19 : u32,  isl_27 : ptr<function, Island_std430_0>,  wg_3 : WideGroup_0) -> Rigid_0
{
    var _S581 : Rigid_0 = rigid_of_3(&((*isl_27)));
    var rg_15 : Rigid_0 = _S581;
    if(((((*isl_27).info_0.x) & (u32(1)))) == u32(0))
    {
        var f_17 : vec3<f32>;
        var t_19 : vec3<f32>;
        wide_partials_1(tid_19, wg_3.first_1, (*isl_27).wide_groups_0, u32(0), &(f_17), &(t_19));
        rigid_acceleration_2(&((*isl_27)), &(rg_15), f_17, t_19);
    }
    return rg_15;
}

fn comp_add_1( sum_8 : ptr<function, vec3<f32>>,  err_7 : ptr<function, vec3<f32>>,  x_23 : vec3<f32>)
{
    var t_20 : vec3<f32> = (*sum_8) + x_23;
    var _S582 : vec3<f32> = abs(x_23);
    (*err_7) = (*err_7) + (select(x_23, (*sum_8), (abs((*sum_8))) >= _S582) - t_20 + select((*sum_8), x_23, (abs((*sum_8))) >= _S582));
    (*sum_8) = t_20;
    return;
}

fn turn_left_1( hi_4 : ptr<function, Quat_0>,  lo_4 : ptr<function, vec4<f32>>,  omega_2 : vec3<f32>,  dt_14 : f32)
{
    var _S583 : f32 = length(omega_2);
    var angle_5 : f32 = _S583 * dt_14;
    if(angle_5 < 1.00000000317107685e-30f)
    {
        return;
    }
    var d_18 : vec4<f32> = turn_minus_one_0(omega_2 / vec3<f32>(_S583), angle_5);
    var dq_2 : Quat_0;
    dq_2.x_0 = d_18.x;
    dq_2.y_0 = d_18.y;
    dq_2.z_0 = d_18.z;
    dq_2.w_0 = d_18.w;
    quat_accumulate_0(&((*hi_4)), &((*lo_4)), quat_vec_0(quat_mul_0(dq_2, (*hi_4))));
    return;
}

fn integrate_rigid_1( isl_28 : ptr<function, Island_std430_0>,  rg_16 : ptr<function, Rigid_0>,  dt_15 : f32)
{
    var _S584 : vec3<f32> = vec3<f32>(dt_15);
    var _S585 : vec3<f32> = (*rg_16).torque_0 * _S584;
    var _S586 : vec3<f32> = (*rg_16).l_0;
    var _S587 : vec3<f32> = (*rg_16).l_err_0;
    comp_add_0(&(_S586), &(_S587), _S585);
    (*rg_16).l_0 = _S586;
    (*rg_16).l_err_0 = _S587;
    var l_5 : vec3<f32> = _S586 + _S587;
    var _S588 : vec3<f32> = (*rg_16).a_0 * _S584;
    var _S589 : vec3<f32> = (*rg_16).vel_0;
    var _S590 : vec3<f32> = (*rg_16).vel_err_0;
    comp_add_0(&(_S589), &(_S590), _S588);
    (*rg_16).vel_0 = _S589;
    (*rg_16).vel_err_0 = _S590;
    var _S591 : vec4<f32> = (*isl_28).inv0_0;
    var _S592 : vec4<f32> = (*isl_28).inv1_0;
    var _S593 : vec4<f32> = (*isl_28).inv2_0;
    var w_mid_1 : vec3<f32> = world_mul_0((*rg_16).rot_0, (*isl_28).inv0_0, (*isl_28).inv1_0, (*isl_28).inv2_0, l_5);
    var delta_1 : vec3<f32> = (_S589 + _S590) * _S584 - turn_difference_0(w_mid_1, dt_15, rotate_0((*rg_16).rot_0, (*isl_28).com_0.xyz));
    var _S594 : vec3<f32> = (*rg_16).pos_0;
    var _S595 : vec3<f32> = (*rg_16).pos_err_0;
    comp_add_0(&(_S594), &(_S595), delta_1);
    (*rg_16).pos_0 = _S594;
    (*rg_16).pos_err_0 = _S595;
    var _S596 : Quat_0 = (*rg_16).rot_0;
    var _S597 : vec4<f32> = (*rg_16).rot_err_0;
    turn_left_0(&(_S596), &(_S597), w_mid_1, dt_15);
    (*rg_16).rot_0 = _S596;
    (*rg_16).rot_err_0 = _S597;
    (*rg_16).w_1 = world_mul_0(_S596, _S591, _S592, _S593, l_5);
    return;
}

fn drift_rigid_1( isl_29 : ptr<function, Island_std430_0>,  rg_17 : ptr<function, Rigid_0>,  tr_7 : vec3<f32>,  phi_5 : vec3<f32>,  dv_7 : vec3<f32>,  dw_4 : vec3<f32>)
{
    var wcom_5 : vec3<f32> = (*isl_29).wcom_0.xyz;
    var rot_3 : Quat_0 = (*rg_17).rot_0;
    var _S598 : vec3<f32> = rotate_0((*rg_17).rot_0, tr_7 - turn_difference_0(phi_5, 1.0f, wcom_5));
    var _S599 : vec3<f32> = (*rg_17).pos_0;
    var _S600 : vec3<f32> = (*rg_17).pos_err_0;
    comp_add_0(&(_S599), &(_S600), _S598);
    (*rg_17).pos_0 = _S599;
    (*rg_17).pos_err_0 = _S600;
    var b_46 : vec3<f32> = inverse_rotate_0((*rg_17).rot_0, (*rg_17).w_1);
    var _S601 : vec4<f32> = (*isl_29).inertia0_0;
    var _S602 : vec4<f32> = (*isl_29).inertia1_0;
    var _S603 : vec4<f32> = (*isl_29).inertia2_0;
    var dl_turn_1 : vec3<f32> = rotate_0((*rg_17).rot_0, cross(phi_5, rows_mul_0((*isl_29).inertia0_0, (*isl_29).inertia1_0, (*isl_29).inertia2_0, b_46)) - rows_mul_0((*isl_29).inertia0_0, (*isl_29).inertia1_0, (*isl_29).inertia2_0, cross(phi_5, b_46)));
    var _S604 : Quat_0 = (*rg_17).rot_0;
    var _S605 : vec4<f32> = (*rg_17).rot_err_0;
    turn_right_0(&(_S604), &(_S605), phi_5);
    (*rg_17).rot_0 = _S604;
    (*rg_17).rot_err_0 = _S605;
    var _S606 : vec3<f32> = rotate_0(rot_3, dv_7 + cross(dw_4, (*isl_29).com_0.xyz - wcom_5));
    var _S607 : vec3<f32> = (*rg_17).vel_0;
    var _S608 : vec3<f32> = (*rg_17).vel_err_0;
    comp_add_0(&(_S607), &(_S608), _S606);
    (*rg_17).vel_0 = _S607;
    (*rg_17).vel_err_0 = _S608;
    var _S609 : vec3<f32> = rotate_0(rot_3, dw_4);
    (*rg_17).w_1 = (*rg_17).w_1 + _S609;
    var _S610 : vec3<f32> = dl_turn_1 + world_mul_0(_S604, _S601, _S602, _S603, _S609);
    var _S611 : vec3<f32> = (*rg_17).l_0;
    var _S612 : vec3<f32> = (*rg_17).l_err_0;
    comp_add_0(&(_S611), &(_S612), _S610);
    (*rg_17).l_0 = _S611;
    (*rg_17).l_err_0 = _S612;
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn wide_rigid(@builtin(workgroup_id) group_3 : vec3<u32>, @builtin(local_invocation_id) thread_3 : vec3<u32>)
{
    var tid_20 : u32 = thread_3.x;
    var _S613 : u32 = group_3.x;
    var wg_4 : WideGroup_0 = wide_group_0(params_0.wide_chunk_table_0, _S613);
    var _S614 : Island_std430_0 = islands_0[wg_4.island_2];
    var _S615 : u32 = _S614.info_0.x;
    var _S616 : bool;
    if(((_S615 & (u32(1)))) != u32(0))
    {
        _S616 = true;
    }
    else
    {
        var _S617 : bool = wide_enter_1(tid_20, wg_4.island_2);
        _S616 = !_S617;
    }
    if(_S616)
    {
        return;
    }
    if(((_S615 & (u32(16)))) != u32(0))
    {
        if(_S613 != (wg_4.first_1))
        {
            return;
        }
        var _S618 : Rigid_0 = wide_rigid_frame_1(tid_20, &(_S614), wg_4);
        var rs_0 : Rigid_0 = _S618;
        if(tid_20 != u32(0))
        {
            _S616 = true;
        }
        else
        {
            _S616 = ((_S615 & (u32(2)))) != u32(0);
        }
        if(_S616)
        {
            return;
        }
        integrate_rigid_1(&(_S614), &(rs_0), params_0.dt_0);
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
    var tu_4 : vec3<f32>;
    var pv_4 : vec3<f32>;
    wide_partials_1(tid_20, wg_4.first_1, _S614.wide_groups_0, u32(2), &(tu_4), &(pv_4));
    var lu_5 : vec3<f32>;
    var lv_5 : vec3<f32>;
    wide_partials_1(tid_20, wg_4.first_1, _S614.wide_groups_0, u32(4), &(lu_5), &(lv_5));
    var _S619 : vec4<f32> = _S614.wcom_0;
    var _S620 : vec3<f32> = vec3<f32>(_S614.wcom_0.w);
    var tr_8 : vec3<f32> = tu_4 / _S620;
    var dv_8 : vec3<f32> = pv_4 / _S620;
    var phi_6 : vec3<f32> = rows_mul_0(_S614.winv0_0, _S614.winv1_0, _S614.winv2_0, lu_5);
    var dw_5 : vec3<f32> = rows_mul_0(_S614.winv0_0, _S614.winv1_0, _S614.winv2_0, lv_5);
    var c_27 : u32 = wg_4.begin_0 + tid_20;
    if(c_27 < (wg_4.end_0))
    {
        drift_apply_0(c_27, _S619.xyz, tr_8, phi_6, dv_8, dw_5);
    }
    if(_S613 != (wg_4.first_1))
    {
        return;
    }
    var _S621 : Rigid_0 = wide_rigid_frame_1(tid_20, &(_S614), wg_4);
    var rg_18 : Rigid_0 = _S621;
    if(tid_20 != u32(0))
    {
        return;
    }
    if(!(((_S615 & (u32(2)))) != u32(0)))
    {
        integrate_rigid_1(&(_S614), &(rg_18), params_0.dt_0);
        drift_rigid_1(&(_S614), &(rg_18), tr_8, phi_6, dv_8, dw_5);
    }
    islands_0[wg_4.island_2].rotation_0 = quat_vec_0(rg_18.rot_0);
    islands_0[wg_4.island_2].rotation_err_0 = rg_18.rot_err_0;
    islands_0[wg_4.island_2].position_0 = vec4<f32>(rg_18.pos_0, 0.0f);
    islands_0[wg_4.island_2].position_err_0 = vec4<f32>(rg_18.pos_err_0, 0.0f);
    islands_0[wg_4.island_2].velocity_0 = vec4<f32>(rg_18.vel_0, 0.0f);
    islands_0[wg_4.island_2].velocity_err_0 = vec4<f32>(rg_18.vel_err_0, 0.0f);
    islands_0[wg_4.island_2].angular_velocity_0 = vec4<f32>(rg_18.w_1, 0.0f);
    islands_0[wg_4.island_2].momentum_0 = vec4<f32>(rg_18.l_0, 0.0f);
    islands_0[wg_4.island_2].momentum_err_0 = vec4<f32>(rg_18.l_err_0, 0.0f);
    return;
}

var<workgroup> g_label_0 : array<u32, i32(4096)>;

var<workgroup> g_changed_0 : u32;

fn connected_3( st_7 : ptr<function, JointState_std430_0>,  has_rebar_4 : bool) -> bool
{
    var _S622 : bool;
    if(((*st_7).damage_0) < 1.0f)
    {
        _S622 = true;
    }
    else
    {
        if(has_rebar_4)
        {
            _S622 = ((*st_7).rebar_broken_0) == 0.0f;
        }
        else
        {
            _S622 = false;
        }
    }
    return _S622;
}

fn connected_4( st_8 : ptr<function, JointState_std430_0>,  has_rebar_5 : bool) -> bool
{
    var _S623 : bool;
    if(((*st_8).damage_0) < 1.0f)
    {
        _S623 = true;
    }
    else
    {
        if(has_rebar_5)
        {
            _S623 = ((*st_8).rebar_broken_0) == 0.0f;
        }
        else
        {
            _S623 = false;
        }
    }
    return _S623;
}

fn island_connected_0( tid_21 : u32,  isl_30 : Island_0) -> bool
{
    var _S624 : u32 = isl_30.range_0.x;
    var n_11 : u32 = isl_30.range_0.y - _S624;
    var converged_2 : bool;
    if(n_11 > u32(4096))
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
    var c_28 : u32 = tid_21;
    loop
    {
        if(c_28 < n_11)
        {
        }
        else
        {
            break;
        }
        g_label_0[c_28] = c_28;
        c_28 = c_28 + u32(256);
    }
    workgroupBarrier();
    converged_2 = false;
    var pass_1 : u32 = u32(0);
    loop
    {
        if(pass_1 <= n_11)
        {
        }
        else
        {
            break;
        }
        if(tid_21 == u32(0))
        {
            g_changed_0 = u32(0);
        }
        workgroupBarrier();
        c_28 = tid_21;
        loop
        {
            if(c_28 < n_11)
            {
            }
            else
            {
                break;
            }
            var gc_1 : u32 = _S624 + c_28;
            var _S625 : u32 = index_0[gc_1];
            var l_6 : u32 = g_label_0[c_28];
            var e_4 : u32 = _S625;
            loop
            {
                if(e_4 < index_0[gc_1 + u32(1)])
                {
                }
                else
                {
                    break;
                }
                var entry_3 : u32 = index_0[e_4];
                var bond_1 : u32 = (entry_3 >> (u32(1)));
                var _S626 : JointState_std430_0 = bond_dyn_0[bond_1].js_0;
                var _S627 : bool = connected_3(&(_S626), (bonds_0[bond_1].law_0.stiff1_0.w) != 0.0f);
                if(!_S627)
                {
                    e_4 = e_4 + u32(1);
                    continue;
                }
                var other_1 : u32;
                if(((entry_3 & (u32(1)))) == u32(0))
                {
                    other_1 = bonds_0[bond_1].law_0.ids_0.z;
                }
                else
                {
                    other_1 = bonds_0[bond_1].law_0.ids_0.y;
                }
                l_6 = min(l_6, g_label_0[other_1 - _S624]);
                e_4 = e_4 + u32(1);
            }
            var _S628 : u32 = min(l_6, g_label_0[l_6]);
            if(_S628 < (g_label_0[c_28]))
            {
                g_label_0[c_28] = _S628;
                g_changed_0 = u32(1);
            }
            c_28 = c_28 + u32(256);
        }
        workgroupBarrier();
        var converged_3 : bool = g_changed_0 == u32(0);
        workgroupBarrier();
        if(converged_3)
        {
            converged_2 = converged_3;
            break;
        }
        var _S629 : u32 = pass_1 + u32(1);
        converged_2 = converged_3;
        pass_1 = _S629;
    }
    if(!converged_2)
    {
        return false;
    }
    if(tid_21 == u32(0))
    {
        g_changed_0 = u32(0);
    }
    workgroupBarrier();
    c_28 = tid_21;
    loop
    {
        if(c_28 < n_11)
        {
        }
        else
        {
            break;
        }
        if((g_label_0[c_28]) != u32(0))
        {
            g_changed_0 = u32(1);
        }
        c_28 = c_28 + u32(256);
    }
    workgroupBarrier();
    return g_changed_0 == u32(0);
}

@compute
@workgroup_size(256, 1, 1)
fn wide_end(@builtin(workgroup_id) group_4 : vec3<u32>, @builtin(local_invocation_id) thread_4 : vec3<u32>)
{
    var tid_22 : u32 = thread_4.x;
    var _S630 : u32 = group_4.x;
    var wg_5 : WideGroup_0 = wide_group_0(params_0.wide_chunk_table_0, _S630);
    if(_S630 != (wg_5.first_1))
    {
        return;
    }
    var _S631 : Island_std430_0 = islands_0[wg_5.island_2];
    var isl_31 : Island_0;
    isl_31.range_0 = _S631.range_0;
    isl_31.info_0 = _S631.info_0;
    isl_31.com_0 = _S631.com_0;
    isl_31.inertia0_0 = _S631.inertia0_0;
    isl_31.inertia1_0 = _S631.inertia1_0;
    isl_31.inertia2_0 = _S631.inertia2_0;
    isl_31.inv0_0 = _S631.inv0_0;
    isl_31.inv1_0 = _S631.inv1_0;
    isl_31.inv2_0 = _S631.inv2_0;
    isl_31.wcom_0 = _S631.wcom_0;
    isl_31.winv0_0 = _S631.winv0_0;
    isl_31.winv1_0 = _S631.winv1_0;
    isl_31.winv2_0 = _S631.winv2_0;
    isl_31.rotation_0 = _S631.rotation_0;
    isl_31.position_0 = _S631.position_0;
    isl_31.position_err_0 = _S631.position_err_0;
    isl_31.velocity_0 = _S631.velocity_0;
    isl_31.velocity_err_0 = _S631.velocity_err_0;
    isl_31.angular_velocity_0 = _S631.angular_velocity_0;
    isl_31.done_0 = _S631.done_0;
    isl_31.coverage_0 = _S631.coverage_0;
    isl_31.wide_groups_0 = _S631.wide_groups_0;
    isl_31.done_pad_0 = _S631.done_pad_0;
    isl_31.probes_0 = _S631.probes_0;
    isl_31.energy_0 = _S631.energy_0;
    isl_31.rotation_err_0 = _S631.rotation_err_0;
    isl_31.momentum_0 = _S631.momentum_0;
    isl_31.momentum_err_0 = _S631.momentum_err_0;
    var _S632 : bool = wide_enter_0(tid_22, &(_S631));
    if(!_S632)
    {
        return;
    }
    var work_4 : vec3<f32>;
    var flags_2 : vec3<f32>;
    wide_partials_1(tid_22, wg_5.first_1, isl_31.wide_groups_0, u32(6), &(work_4), &(flags_2));
    var halt_0 : bool = (((isl_31.info_0.z) & (u32(2)))) != u32(0);
    var whole_0 : bool;
    if(halt_0)
    {
        var _S633 : bool = island_connected_0(tid_22, isl_31);
        whole_0 = _S633;
    }
    else
    {
        whole_0 = false;
    }
    if(tid_22 != u32(0))
    {
        return;
    }
    var halt_1 : bool;
    if(whole_0)
    {
        isl_31.info_0[i32(2)] = ((isl_31.info_0[i32(2)]) & (u32(4294967293)));
        isl_31.coverage_0 = ((isl_31.coverage_0) | (u32(2097280)));
        halt_1 = false;
    }
    else
    {
        halt_1 = halt_0;
    }
    var k_21 : u32 = wide_step_3(isl_31);
    if((isl_31.probes_0.y) > (isl_31.probes_0.x))
    {
        record_probes_0(isl_31, rigid_of_0(isl_31), k_21);
    }
    if(halt_1)
    {
        whole_0 = (((isl_31.info_0.x) & (u32(4)))) != u32(0);
    }
    else
    {
        whole_0 = false;
    }
    if(whole_0)
    {
        contact_split_at_0(isl_31.info_0.w + u32(1));
    }
    var _S634 : f32 = work_4.x;
    var _S635 : f32 = isl_31.energy_0[i32(0)];
    var _S636 : f32 = isl_31.energy_0[i32(1)];
    comp_add1_0(&(_S635), &(_S636), _S634);
    isl_31.energy_0[i32(0)] = _S635;
    isl_31.energy_0[i32(1)] = _S636 + work_4.y;
    isl_31.done_0 = isl_31.done_0 + u32(1);
    isl_31.info_0[i32(1)] = isl_31.info_0[i32(1)] - u32(1);
    isl_31.info_0[i32(3)] = isl_31.info_0[i32(3)] + u32(1);
    if(halt_1)
    {
        isl_31.info_0[i32(2)] = ((((isl_31.info_0.z) & (u32(4294967293)))) | (u32(1)));
    }
    var _S637 : u32;
    if(halt_1)
    {
        _S637 = u32(128);
    }
    else
    {
        _S637 = u32(0);
    }
    var _S638 : u32 = (u32(256) | (_S637));
    if((isl_31.probes_0.y) > (isl_31.probes_0.x))
    {
        _S637 = u32(262144);
    }
    else
    {
        _S637 = u32(0);
    }
    var _S639 : u32 = (_S638 | (_S637));
    if((((isl_31.info_0.x) & (u32(4)))) != u32(0))
    {
        _S637 = u32(524288);
    }
    else
    {
        _S637 = u32(0);
    }
    var _S640 : u32 = (_S639 | (_S637));
    if((flags_2.x) > 0.0f)
    {
        _S637 = u32(32768);
    }
    else
    {
        _S637 = u32(0);
    }
    var _S641 : u32 = (_S640 | (_S637));
    if((flags_2.y) > 0.0f)
    {
        _S637 = u32(65536);
    }
    else
    {
        _S637 = u32(0);
    }
    var _S642 : u32 = (_S641 | (_S637));
    if((flags_2.z) > 0.0f)
    {
        _S637 = u32(131072);
    }
    else
    {
        _S637 = u32(0);
    }
    isl_31.coverage_0 = ((isl_31.coverage_0) | (((_S642 | (_S637)))));
    islands_0[wg_5.island_2].range_0 = isl_31.range_0;
    islands_0[wg_5.island_2].info_0 = isl_31.info_0;
    islands_0[wg_5.island_2].com_0 = isl_31.com_0;
    islands_0[wg_5.island_2].inertia0_0 = isl_31.inertia0_0;
    islands_0[wg_5.island_2].inertia1_0 = isl_31.inertia1_0;
    islands_0[wg_5.island_2].inertia2_0 = isl_31.inertia2_0;
    islands_0[wg_5.island_2].inv0_0 = isl_31.inv0_0;
    islands_0[wg_5.island_2].inv1_0 = isl_31.inv1_0;
    islands_0[wg_5.island_2].inv2_0 = isl_31.inv2_0;
    islands_0[wg_5.island_2].wcom_0 = isl_31.wcom_0;
    islands_0[wg_5.island_2].winv0_0 = isl_31.winv0_0;
    islands_0[wg_5.island_2].winv1_0 = isl_31.winv1_0;
    islands_0[wg_5.island_2].winv2_0 = isl_31.winv2_0;
    islands_0[wg_5.island_2].rotation_0 = isl_31.rotation_0;
    islands_0[wg_5.island_2].position_0 = isl_31.position_0;
    islands_0[wg_5.island_2].position_err_0 = isl_31.position_err_0;
    islands_0[wg_5.island_2].velocity_0 = isl_31.velocity_0;
    islands_0[wg_5.island_2].velocity_err_0 = isl_31.velocity_err_0;
    islands_0[wg_5.island_2].angular_velocity_0 = isl_31.angular_velocity_0;
    islands_0[wg_5.island_2].done_0 = isl_31.done_0;
    islands_0[wg_5.island_2].coverage_0 = isl_31.coverage_0;
    islands_0[wg_5.island_2].wide_groups_0 = isl_31.wide_groups_0;
    islands_0[wg_5.island_2].done_pad_0 = isl_31.done_pad_0;
    islands_0[wg_5.island_2].probes_0 = isl_31.probes_0;
    islands_0[wg_5.island_2].energy_0 = isl_31.energy_0;
    islands_0[wg_5.island_2].rotation_err_0 = isl_31.rotation_err_0;
    islands_0[wg_5.island_2].momentum_0 = isl_31.momentum_0;
    islands_0[wg_5.island_2].momentum_err_0 = isl_31.momentum_err_0;
    return;
}

fn sv_0( c_29 : u32,  slot_3 : u32) -> u32
{
    return params_0.statics_base_0 + u32(23) * c_29 + slot_3;
}

fn project_load_slot_0( tid_23 : u32,  isl_32 : ptr<function, Island_std430_0>,  slot_4 : u32)
{
    var _S643 : u32;
    var _S644 : vec3<f32> = vec3<f32>(0.0f);
    var net_f_0 : vec3<f32> = _S644;
    var net_m_0 : vec3<f32> = _S644;
    var _S645 : vec4<u32> = (*isl_32).range_0;
    var _S646 : u32 = (*isl_32).range_0.x + tid_23;
    var c_30 : u32 = _S646;
    loop
    {
        var _S647 : u32 = _S645.y;
        _S643 = _S647;
        if(c_30 < _S647)
        {
        }
        else
        {
            break;
        }
        var fi_1 : vec3<f32> = scratch_0[sv_0(c_30, slot_4)].xyz;
        net_f_0 = net_f_0 + fi_1;
        net_m_0 = net_m_0 + (cross(chunks_0[c_30].center_0.xyz - (*isl_32).com_0.xyz, fi_1) + scratch_0[sv_0(c_30, slot_4 + u32(1))].xyz);
        c_30 = c_30 + u32(256);
    }
    group_sum3_3(tid_23, &(net_f_0), &(net_m_0));
    var _S648 : vec4<f32> = (*isl_32).com_0;
    var _S649 : vec3<f32> = net_f_0 / vec3<f32>((*isl_32).com_0.w);
    var _S650 : vec3<f32> = rows_mul_0((*isl_32).inv0_0, (*isl_32).inv1_0, (*isl_32).inv2_0, net_m_0);
    c_30 = _S646;
    loop
    {
        if(c_30 < _S643)
        {
        }
        else
        {
            break;
        }
        var _S651 : u32 = sv_0(c_30, slot_4);
        scratch_0[_S651] = vec4<f32>(scratch_0[_S651].xyz - (_S649 + cross(_S650, chunks_0[c_30].center_0.xyz - _S648.xyz)) * vec3<f32>(chunks_0[c_30].center_0.w), 0.0f);
        var _S652 : u32 = sv_0(c_30, slot_4 + u32(1));
        scratch_0[_S652] = vec4<f32>(scratch_0[_S652].xyz - rows_mul_0(chunks_0[c_30].inertia0_1, chunks_0[c_30].inertia1_1, chunks_0[c_30].inertia2_1, _S650), 0.0f);
        c_30 = c_30 + u32(256);
    }
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    return;
}

fn island_dot_0( tid_24 : u32,  c0_0 : u32,  c1_0 : u32,  sa_0 : u32,  sb_0 : u32) -> f32
{
    var _S653 : vec4<f32> = vec4<f32>(0.0f);
    var acc_0 : vec4<f32> = _S653;
    var unused_4 : vec4<f32> = _S653;
    var c_31 : u32 = c0_0 + tid_24;
    loop
    {
        if(c_31 < c1_0)
        {
        }
        else
        {
            break;
        }
        acc_0[i32(0)] = acc_0[i32(0)] + (dot(scratch_0[sv_0(c_31, sa_0)].xyz, scratch_0[sv_0(c_31, sb_0)].xyz) + dot(scratch_0[sv_0(c_31, sa_0 + u32(1))].xyz, scratch_0[sv_0(c_31, sb_0 + u32(1))].xyz));
        c_31 = c_31 + u32(256);
    }
    group_sum2_2(tid_24, &(acc_0), &(unused_4));
    return acc_0.x;
}

fn stress_measures_1( b_47 : ptr<function, JointBond_std430_0>,  q_lin_3 : vec3<f32>,  q_ang_3 : vec3<f32>) -> Measures_0
{
    var area_3 : f32 = (*b_47).geom0_0.x;
    var _S654 : f32 = q_lin_3.z;
    var axial_1 : f32 = fdiv_0(_S654, area_3);
    var bending_1 : f32 = fdiv_0(abs(q_ang_3.x), (*b_47).geom1_0.x) + fdiv_0(abs(q_ang_3.y), (*b_47).geom1_0.y);
    var _S655 : f32 = q_lin_3.x;
    var _S656 : f32 = q_lin_3.y;
    var shear_2 : f32 = fdiv_0(fsqrt_0(_S655 * _S655 + _S656 * _S656), area_3) + fdiv_0(abs(q_ang_3.z), (*b_47).geom0_0.w);
    var m_5 : Measures_0;
    m_5.tension_0 = axial_1 + bending_1;
    m_5.shear_0 = shear_2;
    var _S657 : f32 = - axial_1;
    m_5.normal_compression_0 = max(_S657, 0.0f);
    m_5.compression_0 = _S657 + bending_1;
    m_5.compressive_force_0 = max(- _S654, 0.0f);
    return m_5;
}

fn dif_factor_1( mat_10 : ptr<function, JointMaterial_std140_0>,  strain_rate_2 : f32) -> f32
{
    var r_14 : f32 = abs(strain_rate_2);
    var _S658 : vec4<f32> = (*mat_10).dif_0;
    var ref_1 : f32 = (*mat_10).dif_0.x;
    if(r_14 <= ref_1)
    {
        return 1.0f;
    }
    var _S659 : f32 = _S658.z;
    var f_18 : f32;
    if(r_14 <= _S659)
    {
        f_18 = fpow_0(fdiv_0(r_14, ref_1), _S658.y);
    }
    else
    {
        f_18 = fpow_0(fdiv_0(_S659, ref_1), _S658.y) * fpow_0(fdiv_0(r_14, _S659), _S658.w);
    }
    return clamp(f_18, 1.0f, (*mat_10).misc_0.x);
}

fn fatigue_factor_2( mat_11 : ptr<function, JointMaterial_std140_0>,  life_3 : f32) -> f32
{
    if(((((*mat_11).kind_flags_0.y) & (u32(64)))) == u32(0))
    {
        return 1.0f;
    }
    return fpow_0(clamp(life_3, 0.0f, 1.0f), fdiv_0(1.0f, (*mat_11).misc_0.y - 2.0f));
}

fn failure_indices_1( mat_12 : ptr<function, JointMaterial_std140_0>,  b_48 : ptr<function, JointBond_std430_0>,  m_6 : Measures_0,  multiplier_2 : f32) -> vec4<f32>
{
    var fc_4 : f32 = (*mat_12).strength_0.y * multiplier_2;
    var _S660 : f32 = min((*mat_12).strength_0.z * multiplier_2 + (*mat_12).strength_0.w * m_6.normal_compression_0, (*mat_12).energy_1.x * multiplier_2);
    var idx_1 : vec4<f32>;
    idx_1[i32(0)] = max(fdiv_0(m_6.tension_0, (*mat_12).strength_0.x * multiplier_2), 0.0f);
    var _S661 : f32;
    if(_S660 > 0.0f)
    {
        _S661 = fdiv_0(m_6.shear_0, _S660);
    }
    else
    {
        _S661 = infinity_0();
    }
    idx_1[i32(1)] = _S661;
    idx_1[i32(2)] = max(fdiv_0(m_6.compression_0, fc_4), 0.0f);
    var _S662 : f32 = (*b_48).stiff1_0.y;
    if(_S662 > 0.0f)
    {
        _S661 = fdiv_0(m_6.compressive_force_0, _S662);
    }
    else
    {
        _S661 = 0.0f;
    }
    idx_1[i32(3)] = _S661;
    return idx_1;
}

fn contact_normal_1( mat_13 : ptr<function, JointMaterial_std140_0>,  b_49 : ptr<function, JointBond_std430_0>,  crush_5 : f32,  d_lin_5 : vec3<f32>,  d_ang_4 : vec3<f32>) -> vec4<f32>
{
    var kn_4 : f32 = (*b_49).stiff0_0.x;
    var w0_4 : f32 = (*b_49).geom0_0.y;
    var w1_4 : f32 = (*b_49).geom0_0.z;
    var nc_sum_5 : f32;
    var m1_4 : f32;
    var m2_4 : f32;
    var energy_7 : f32;
    if(((((*mat_13).kind_flags_0.y) & (u32(4)))) != u32(0))
    {
        var p_18 : vec4<f32> = no_tension_patch_0(kn_4 * (1.0f - crush_5), w0_4, w1_4, d_lin_5.z, d_ang_4.x, d_ang_4.y);
        var _S663 : f32 = p_18.y;
        var _S664 : f32 = p_18.z;
        var _S665 : f32 = p_18.w;
        nc_sum_5 = p_18.x;
        m1_4 = _S663;
        m2_4 = _S664;
        energy_7 = _S665;
    }
    else
    {
        var ki_2 : f32 = kn_4 * (1.0f - crush_5) / 36.0f;
        var _S666 : f32 = d_ang_4.x;
        var _S667 : f32 = d_ang_4.y;
        var spread_2 : f32 = abs(_S666) * 0.4166666567325592f * w1_4 + abs(_S667) * 0.4166666567325592f * w0_4;
        var _S668 : f32 = d_lin_5.z;
        var slack_2 : f32 = 9.99999997475242708e-07f * (abs(_S668) + spread_2);
        if((_S668 - spread_2) > slack_2)
        {
            nc_sum_5 = 0.0f;
            m1_4 = 0.0f;
            m2_4 = 0.0f;
            energy_7 = 0.0f;
        }
        else
        {
            if((_S668 + spread_2) < (- slack_2))
            {
                var i1_1 : f32 = 2.91666650772094727f * w0_4 * w0_4;
                var i2_1 : f32 = 2.91666650772094727f * w1_4 * w1_4;
                var _S669 : f32 = ki_2 * _S666 * i2_1;
                var _S670 : f32 = ki_2 * _S667 * i1_1;
                var _S671 : f32 = 0.5f * ki_2 * (36.0f * _S668 * _S668 + _S666 * _S666 * i2_1 + _S667 * _S667 * i1_1);
                nc_sum_5 = ki_2 * 36.0f * _S668;
                m1_4 = _S669;
                m2_4 = _S670;
                energy_7 = _S671;
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
                    var _S672 : f32 = SPRING_AT_0[i_11] * w0_4;
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
                        var di_2 : f32 = _S668 + _S666 * s2_1 - _S667 * _S672;
                        if(di_2 < 0.0f)
                        {
                            var f_19 : f32 = ki_2 * di_2;
                            var m1_6 : f32 = m1_4 + f_19 * s2_1;
                            var m2_6 : f32 = m2_4 - f_19 * _S672;
                            var energy_9 : f32 = energy_7 + 0.5f * ki_2 * di_2 * di_2;
                            nc_sum_5 = nc_sum_5 + f_19;
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

fn contact_part_1( mat_14 : ptr<function, JointMaterial_std140_0>,  b_50 : ptr<function, JointBond_std430_0>,  crush_6 : f32,  plastic_5 : vec3<f32>,  d_lin_6 : vec3<f32>,  d_ang_5 : vec3<f32>) -> Contact_0
{
    var c_32 : Contact_0;
    var _S673 : vec3<f32> = vec3<f32>(0.0f);
    c_32.q_lin_1 = _S673;
    c_32.q_ang_1 = _S673;
    c_32.energy_5 = 0.0f;
    c_32.diss_0 = 0.0f;
    c_32.plastic_0 = plastic_5;
    if(((((*mat_14).kind_flags_0.y) & (u32(2)))) == u32(0))
    {
        return c_32;
    }
    var ks_3 : f32 = (*b_50).stiff0_0.y;
    var kt_3 : f32 = (*b_50).stiff1_0.x;
    var _S674 : vec4<f32> = contact_normal_1(&((*mat_14)), &((*b_50)), crush_6, d_lin_6, d_ang_5);
    var nc_sum_7 : f32 = _S674.x;
    var m1_7 : f32 = _S674.y;
    var m2_7 : f32 = _S674.z;
    var energy_10 : f32 = _S674.w;
    var nc_2 : f32 = - nc_sum_7;
    var p_19 : vec3<f32> = plastic_5;
    c_32.q_lin_1 = vec3<f32>(0.0f, 0.0f, nc_sum_7);
    c_32.q_ang_1 = vec3<f32>(m1_7, m2_7, 0.0f);
    var slide_cap_2 : f32 = (*mat_14).strength_0.w * nc_2;
    var _S675 : f32 = ks_3 * (d_lin_6.x - plastic_5.x);
    var _S676 : f32 = ks_3 * (d_lin_6.y - plastic_5.y);
    var tn_2 : f32 = fsqrt_0(_S675 * _S675 + _S676 * _S676);
    var _S677 : bool;
    if(tn_2 > slide_cap_2)
    {
        _S677 = tn_2 > 0.0f;
    }
    else
    {
        _S677 = false;
    }
    var diss_5 : f32;
    if(_S677)
    {
        var _S678 : f32 = fdiv_0(_S675, tn_2);
        var _S679 : f32 = fdiv_0(_S676, tn_2);
        var dslip_2 : f32 = fdiv_0(tn_2 - slide_cap_2, ks_3);
        p_19[i32(0)] = p_19[i32(0)] + _S678 * dslip_2;
        p_19[i32(1)] = p_19[i32(1)] + _S679 * dslip_2;
        c_32.q_lin_1[i32(0)] = _S678 * slide_cap_2;
        c_32.q_lin_1[i32(1)] = _S679 * slide_cap_2;
        diss_5 = slide_cap_2 * dslip_2;
    }
    else
    {
        c_32.q_lin_1[i32(0)] = _S675;
        c_32.q_lin_1[i32(1)] = _S676;
        diss_5 = 0.0f;
    }
    var tq_1 : vec2<f32> = return_map_0(kt_3, d_ang_5.z, p_19.z, slide_cap_2 * (*b_50).geom1_0.z);
    var _S680 : f32 = tq_1.x;
    var _S681 : f32 = tq_1.y;
    var diss_6 : f32 = diss_5 + abs(_S680) * abs(_S681);
    p_19[i32(2)] = p_19[i32(2)] + _S681;
    c_32.q_ang_1[i32(2)] = _S680;
    c_32.energy_5 = energy_10 + 0.5f * (fdiv_0(sq_0(c_32.q_lin_1.x), ks_3) + fdiv_0(sq_0(c_32.q_lin_1.y), ks_3) + fdiv_0(sq_0(_S680), kt_3));
    c_32.diss_0 = diss_6;
    c_32.plastic_0 = p_19;
    return c_32;
}

fn contact_offsets_1( mat_15 : ptr<function, JointMaterial_std140_0>,  b_51 : ptr<function, JointBond_std430_0>,  crush_7 : f32,  plastic_6 : vec3<f32>,  d_lin_7 : vec3<f32>,  d_ang_6 : vec3<f32>) -> vec3<f32>
{
    var _S682 : u32 = (*mat_15).kind_flags_0.y;
    if(((_S682 & (u32(2)))) == u32(0))
    {
        return plastic_6;
    }
    var kn_5 : f32 = (*b_51).stiff0_0.x;
    var ks_4 : f32 = (*b_51).stiff0_0.y;
    var kt_4 : f32 = (*b_51).stiff1_0.x;
    var w0_5 : f32 = (*b_51).geom0_0.y;
    var w1_5 : f32 = (*b_51).geom0_0.z;
    var nc_sum_8 : f32;
    if(((_S682 & (u32(4)))) != u32(0))
    {
        var _S683 : vec4<f32> = no_tension_patch_0(kn_5 * (1.0f - crush_7), w0_5, w1_5, d_lin_7.z, d_ang_6.x, d_ang_6.y);
        nc_sum_8 = _S683.x;
    }
    else
    {
        var ki_3 : f32 = kn_5 * (1.0f - crush_7) / 36.0f;
        var _S684 : f32 = d_ang_6.x;
        var _S685 : f32 = d_ang_6.y;
        var spread_3 : f32 = abs(_S684) * 0.4166666567325592f * w1_5 + abs(_S685) * 0.4166666567325592f * w0_5;
        var _S686 : f32 = d_lin_7.z;
        var slack_3 : f32 = 9.99999997475242708e-07f * (abs(_S686) + spread_3);
        if((_S686 - spread_3) > slack_3)
        {
            nc_sum_8 = 0.0f;
        }
        else
        {
            if((_S686 + spread_3) < (- slack_3))
            {
                nc_sum_8 = ki_3 * 36.0f * _S686;
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
                    var _S687 : f32 = SPRING_AT_0[i_12] * w0_5;
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
                        var di_3 : f32 = _S686 + _S684 * (SPRING_AT_0[j_10] * w1_5) - _S685 * _S687;
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
    var p_20 : vec3<f32> = plastic_6;
    var slide_cap_3 : f32 = (*mat_15).strength_0.w * nc_3;
    var _S688 : f32 = ks_4 * (d_lin_7.x - plastic_6.x);
    var _S689 : f32 = ks_4 * (d_lin_7.y - plastic_6.y);
    var tn_3 : f32 = fsqrt_0(_S688 * _S688 + _S689 * _S689);
    var _S690 : bool;
    if(tn_3 > slide_cap_3)
    {
        _S690 = tn_3 > 0.0f;
    }
    else
    {
        _S690 = false;
    }
    if(_S690)
    {
        var _S691 : f32 = fdiv_0(_S689, tn_3);
        var dslip_3 : f32 = fdiv_0(tn_3 - slide_cap_3, ks_4);
        p_20[i32(0)] = p_20[i32(0)] + fdiv_0(_S688, tn_3) * dslip_3;
        p_20[i32(1)] = p_20[i32(1)] + _S691 * dslip_3;
    }
    p_20[i32(2)] = p_20[i32(2)] + return_map_0(kt_4, d_ang_6.z, p_20.z, slide_cap_3 * (*b_51).geom1_0.z).y;
    return p_20;
}

fn life_rate_1( mat_16 : ptr<function, JointMaterial_std140_0>,  s_11 : f32) -> f32
{
    if(s_11 <= 0.0f)
    {
        return 0.0f;
    }
    var _S692 : f32 = (*mat_16).misc_0.y;
    return fdiv_0((_S692 + 1.0f) * fpow_0(s_11, _S692), (*mat_16).misc_0.z);
}

fn joint_evaluate_1( mat_17 : ptr<function, JointMaterial_std140_0>,  b_52 : ptr<function, JointBond_std430_0>,  state_3 : ptr<function, JointState_std430_0>,  d_lin_8 : vec3<f32>,  d_ang_7 : vec3<f32>,  dt_16 : f32,  fracture_3 : bool) -> JointResponse_0
{
    var kn_6 : f32 = (*b_52).stiff0_0.x;
    var ks_5 : f32 = (*b_52).stiff0_0.y;
    var kb1_1 : f32 = (*b_52).stiff0_0.z;
    var kb2_1 : f32 = (*b_52).stiff0_0.w;
    var _S693 : vec4<f32> = (*b_52).stiff1_0;
    var kt_5 : f32 = (*b_52).stiff1_0.x;
    var has_rebar_6 : bool = ((*b_52).stiff1_0.w) != 0.0f;
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
    var _S694 : bool = connected_4(&((*state_3)), has_rebar_6);
    var qe_lin_1 : vec3<f32> = d_lin_8 * vec3<f32>(ks_5, ks_5, kn_6);
    var qe_ang_1 : vec3<f32> = d_ang_7 * vec3<f32>(kb1_1, kb2_1, kt_5);
    var _S695 : Measures_0 = stress_measures_1(&((*b_52)), qe_lin_1, qe_ang_1);
    var _S696 : f32 = max(max(_S695.tension_0, _S695.shear_0), _S695.compression_0);
    var _S697 : bool = dt_16 > 0.0f;
    var dif_2 : f32;
    if(_S697)
    {
        var raw_1 : f32 = fdiv_0(max(fdiv_0(_S696 - st_9.governing_stress_0, dt_16), 0.0f), (*mat_17).misc_0.w);
        var tau_3 : f32 = _S693.z;
        if(((flags_3 & (u32(16)))) != u32(0))
        {
            dif_2 = - expm1_accurate_0(- fdiv_0(dt_16, tau_3));
        }
        else
        {
            dif_2 = min(fdiv_0(dt_16, tau_3), 1.0f);
        }
        st_9.strain_rate_0 = st_9.strain_rate_0 + (raw_1 - st_9.strain_rate_0) * dif_2;
        st_9.governing_stress_0 = _S696;
    }
    if(((flags_3 & (u32(32)))) != u32(0))
    {
        var _S698 : f32 = dif_factor_1(&((*mat_17)), st_9.strain_rate_0);
        dif_2 = _S698;
    }
    else
    {
        dif_2 = 1.0f;
    }
    var weibull_1 : f32 = (*b_52).geom1_0.w;
    var _S699 : f32 = fatigue_factor_2(&((*mat_17)), st_9.life_0);
    var multiplier_3 : f32 = weibull_1 * dif_2 * _S699;
    var _S700 : vec4<f32> = failure_indices_1(&((*mat_17)), &((*b_52)), _S695, multiplier_3);
    var _S701 : f32 = _S700.x;
    var _S702 : f32 = _S700.y;
    st_9.utilization_0 = max(max(_S701, _S702), max(_S700.z, _S700.w));
    var _S703 : f32 = d_lin_8.x;
    var _S704 : f32 = d_lin_8.y;
    var _S705 : f32 = d_ang_7.z;
    var _S706 : f32 = ks_5 * (sq_0(_S703) + sq_0(_S704)) + kb1_1 * sq_0(d_ang_7.x) + kb2_1 * sq_0(d_ang_7.y) + kt_5 * sq_0(_S705);
    var _S707 : f32 = d_lin_8.z;
    var _S708 : bool = _S707 > 0.0f;
    if(_S708)
    {
        dif_2 = kn_6 * sq_0(_S707);
    }
    else
    {
        dif_2 = 0.0f;
    }
    var psi_ts_1 : f32 = 0.5f * (_S706 + dif_2);
    var psi_c_1 : f32;
    if(_S707 < 0.0f)
    {
        psi_c_1 = 0.5f * kn_6 * sq_0(_S707);
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
        var _S709 : vec4<f32> = contact_normal_1(&((*mat_17)), &((*b_52)), (*state_3).crush_1, d_lin_8, d_ang_7);
        patch_1 = _S709.w + 0.5f * (ks_5 * (sq_0(_S703 - (*state_3).plastic_x_0) + sq_0(_S704 - (*state_3).plastic_y_0)) + kt_5 * sq_0(_S705 - (*state_3).plastic_t_0));
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
        rebar_1 = 0.5f * ((*b_52).rebar0_0.x * sq_0(_S707 - (*state_3).rebar_plastic_0) + (*b_52).rebar0_0.y * (sq_0(_S703 - (*state_3).rebar_slip0_0) + sq_0(_S704 - (*state_3).rebar_slip1_0)));
    }
    else
    {
        rebar_1 = 0.0f;
    }
    var _S710 : f32 = (*state_3).damage_0;
    var _S711 : f32 = 1.0f - (*state_3).damage_0;
    var _S712 : f32 = (*state_3).crush_1;
    var frozen_1 : f32 = _S711 * (psi_ts_1 + (1.0f - (*state_3).crush_1) * psi_c_1) + (*state_3).damage_0 * patch_1 + rebar_1;
    var diss_contact_1 : f32;
    var psi_contact_1 : f32;
    var intact_normal_1 : f32;
    var dissipated_11 : f32;
    var qc_lin_1 : vec3<f32>;
    if(fracture_3)
    {
        var _S713 : bool = _S701 >= _S702;
        if(_S713)
        {
            diss_contact_1 = _S701;
        }
        else
        {
            diss_contact_1 = _S702;
        }
        var mode_ts_1 : u32;
        if(_S713)
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
                intact_normal_1 = fdiv_0(psi_contact_1 * (*b_52).geom0_0.x * diss_contact_1 * diss_contact_1, psi_ts_1);
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
            var _S714 : f32 = inc_2.x;
            if(_S714 > (st_9.damage_0))
            {
                var _S715 : Contact_0 = contact_part_1(&((*mat_17)), &((*b_52)), st_9.crush_1, plastic_7, d_lin_8, d_ang_7);
                var _S716 : f32 = max(inc_2.y - max(_S715.energy_5 - (1.0f - st_9.crush_1) * psi_c_1, 0.0f) * (_S714 - st_9.damage_0), 0.0f);
                st_9.damage_0 = _S714;
                st_9.mode_0 = mode_ts_1;
                dissipated_11 = _S716;
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
        if(_S710 > 0.0f)
        {
            var _S717 : Contact_0 = contact_part_1(&((*mat_17)), &((*b_52)), _S712, vec3<f32>((*state_3).plastic_x_0, (*state_3).plastic_y_0, (*state_3).plastic_t_0), d_lin_8, d_ang_7);
            qc_lin_1 = qe_ang_1 * vec3<f32>(_S711) + _S717.q_ang_1 * vec3<f32>(_S710);
        }
        else
        {
            qc_lin_1 = qe_ang_1;
        }
        var _S718 : Measures_0 = stress_measures_1(&((*b_52)), vec3<f32>(0.0f, 0.0f, min(qe_lin_1.z, 0.0f)), qc_lin_1);
        var _S719 : vec4<f32> = failure_indices_1(&((*mat_17)), &((*b_52)), _S718, multiplier_3);
        var _S720 : f32 = _S719.z;
        var _S721 : f32 = _S719.w;
        var _S722 : bool = _S720 >= _S721;
        if(_S722)
        {
            psi_contact_1 = _S720;
        }
        else
        {
            psi_contact_1 = _S721;
        }
        if(_S722)
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
                intact_normal_1 = fdiv_0((*mat_17).energy_1.w * (*b_52).geom0_0.x * psi_contact_1 * psi_contact_1, psi_c_1);
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
            var _S723 : f32 = inc_3.x;
            if(_S723 > (st_9.crush_1))
            {
                var dissipated_12 : f32 = dissipated_11 + inc_3.y;
                st_9.crush_1 = _S723;
                st_9.mode_0 = mode_c_1;
                if(_S723 >= 1.0f)
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
    var _S724 : vec3<f32> = vec3<f32>(0.0f);
    var qc_ang_1 : vec3<f32>;
    if((st_9.damage_0) == 0.0f)
    {
        if(((flags_3 & (u32(8)))) == u32(0))
        {
            var _S725 : vec3<f32> = contact_offsets_1(&((*mat_17)), &((*b_52)), st_9.crush_1, plastic_7, d_lin_8, d_ang_7);
            st_9.plastic_x_0 = _S725.x;
            st_9.plastic_y_0 = _S725.y;
            st_9.plastic_t_0 = _S725.z;
        }
        diss_contact_1 = 0.0f;
        qc_lin_1 = _S724;
        qc_ang_1 = _S724;
        psi_contact_1 = 0.0f;
    }
    else
    {
        var _S726 : Contact_0 = contact_part_1(&((*mat_17)), &((*b_52)), st_9.crush_1, plastic_7, d_lin_8, d_ang_7);
        st_9.plastic_x_0 = _S726.plastic_0.x;
        st_9.plastic_y_0 = _S726.plastic_0.y;
        st_9.plastic_t_0 = _S726.plastic_0.z;
        diss_contact_1 = _S726.diss_0;
        qc_lin_1 = _S726.q_lin_1;
        qc_ang_1 = _S726.q_ang_1;
        psi_contact_1 = _S726.energy_5;
    }
    var dissipated_14 : f32 = dissipated_11 + dmg_1 * diss_contact_1;
    if(_S708)
    {
        intact_normal_1 = kn_6 * _S707;
    }
    else
    {
        intact_normal_1 = (1.0f - st_9.crush_1) * kn_6 * _S707;
    }
    var _S727 : f32 = 1.0f - dmg_1;
    var force_lin_4 : vec3<f32> = vec3<f32>(_S727 * qe_lin_1.x + dmg_1 * qc_lin_1.x, _S727 * qe_lin_1.y + dmg_1 * qc_lin_1.y, _S727 * intact_normal_1 + dmg_1 * qc_lin_1.z);
    var force_ang_3 : vec3<f32> = qe_ang_1 * vec3<f32>(_S727) + qc_ang_1 * vec3<f32>(dmg_1);
    var stored_9 : f32 = _S727 * (psi_ts_1 + (1.0f - st_9.crush_1) * psi_c_1) + dmg_1 * psi_contact_1;
    if(has_rebar_6)
    {
        changed_1 = (st_9.rebar_broken_0) == 0.0f;
    }
    else
    {
        changed_1 = false;
    }
    var stored_10 : f32;
    var force_lin_5 : vec3<f32>;
    if(changed_1)
    {
        var k_axial_1 : f32 = (*b_52).rebar0_0.x;
        var k_dowel_1 : f32 = (*b_52).rebar0_0.y;
        var yield_force_1 : f32 = (*b_52).rebar0_0.z;
        var dowel_capacity_1 : f32 = (*b_52).rebar0_0.w;
        var nr_1 : vec2<f32> = return_map_0(k_axial_1, _S707, st_9.rebar_plastic_0, yield_force_1);
        var v1_1 : vec2<f32> = return_map_0(k_dowel_1, _S703, st_9.rebar_slip0_0, dowel_capacity_1);
        var v2_1 : vec2<f32> = return_map_0(k_dowel_1, _S704, st_9.rebar_slip1_0, dowel_capacity_1);
        var _S728 : f32 = nr_1.y;
        var _S729 : f32 = v1_1.y;
        var _S730 : f32 = v2_1.y;
        var work_5 : f32 = yield_force_1 * abs(_S728) + dowel_capacity_1 * (abs(_S729) + abs(_S730));
        st_9.rebar_plastic_0 = st_9.rebar_plastic_0 + _S728;
        st_9.rebar_slip0_0 = st_9.rebar_slip0_0 + _S729;
        st_9.rebar_slip1_0 = st_9.rebar_slip1_0 + _S730;
        st_9.rebar_work_0 = st_9.rebar_work_0 + work_5;
        var dissipated_15 : f32 = dissipated_14 + work_5;
        var _S731 : f32 = nr_1.x;
        var _S732 : f32 = v1_1.x;
        var _S733 : f32 = v2_1.x;
        var elastic_1 : f32 = 0.5f * (fdiv_0(sq_0(_S731), k_axial_1) + fdiv_0(sq_0(_S732) + sq_0(_S733), k_dowel_1));
        if(fracture_3)
        {
            changed_1 = (st_9.rebar_work_0) >= ((*b_52).rebar1_0.x);
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
            stored_10 = stored_9;
            dissipated_11 = dissipated_16;
        }
        else
        {
            var stored_11 : f32 = stored_9 + elastic_1;
            force_lin_5 = force_lin_4 + vec3<f32>(_S732, _S733, _S731);
            stored_10 = stored_11;
            dissipated_11 = dissipated_15;
        }
    }
    else
    {
        force_lin_5 = force_lin_4;
        stored_10 = stored_9;
        dissipated_11 = dissipated_14;
    }
    if(fracture_3)
    {
        changed_1 = _S697;
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
        var _S734 : Measures_0 = stress_measures_1(&((*b_52)), force_lin_5, force_ang_3);
        var _S735 : vec4<f32> = failure_indices_1(&((*mat_17)), &((*b_52)), _S734, weibull_1);
        var _S736 : f32 = life_rate_1(&((*mat_17)), max(max(_S735.x, _S735.y), _S735.z));
        st_9.life_0 = max(st_9.life_0 - _S736 * dt_16, 0.0f);
    }
    if((st_9.damage_0) != _S710)
    {
        changed_1 = true;
    }
    else
    {
        changed_1 = (st_9.crush_1) != _S712;
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
        released_1 = frozen_1 - stored_10;
    }
    else
    {
        released_1 = 0.0f;
    }
    var _S737 : f32 = max(min(dissipated_11, released_1), 0.0f);
    var overshoot_2 : f32 = released_1 - _S737;
    st_9.dissipated_0 = st_9.dissipated_0 + _S737;
    var resp_1 : JointResponse_0;
    resp_1.force_lin_1 = force_lin_5;
    resp_1.force_ang_1 = force_ang_3;
    resp_1.state_1 = st_9;
    resp_1.dissipated_1 = _S737;
    resp_1.overshoot_0 = overshoot_2;
    resp_1.stored_0 = stored_10;
    if(_S694)
    {
        changed_1 = !connected_2(st_9, has_rebar_6);
    }
    else
    {
        changed_1 = false;
    }
    resp_1.disconnected_0 = changed_1;
    resp_1.fatigue_before_0 = _S699;
    resp_1.measures_0 = _S695;
    return resp_1;
}

fn joint_evaluate_2( mat_18 : ptr<function, JointMaterial_std140_0>,  b_53 : ptr<function, JointBond_std430_0>,  state_4 : JointState_0,  d_lin_9 : vec3<f32>,  d_ang_8 : vec3<f32>,  dt_17 : f32,  fracture_4 : bool) -> JointResponse_0
{
    var kn_7 : f32 = (*b_53).stiff0_0.x;
    var ks_6 : f32 = (*b_53).stiff0_0.y;
    var kb1_2 : f32 = (*b_53).stiff0_0.z;
    var kb2_2 : f32 = (*b_53).stiff0_0.w;
    var _S738 : vec4<f32> = (*b_53).stiff1_0;
    var kt_6 : f32 = (*b_53).stiff1_0.x;
    var has_rebar_7 : bool = ((*b_53).stiff1_0.w) != 0.0f;
    var kind_6 : u32 = (*mat_18).kind_flags_0.x;
    var flags_4 : u32 = (*mat_18).kind_flags_0.y;
    var softening_2 : bool = ((flags_4 & (u32(1)))) != u32(0);
    var st_10 : JointState_0 = state_4;
    var was_connected_0 : bool = connected_2(state_4, has_rebar_7);
    var qe_lin_2 : vec3<f32> = d_lin_9 * vec3<f32>(ks_6, ks_6, kn_7);
    var qe_ang_2 : vec3<f32> = d_ang_8 * vec3<f32>(kb1_2, kb2_2, kt_6);
    var _S739 : Measures_0 = stress_measures_1(&((*b_53)), qe_lin_2, qe_ang_2);
    var _S740 : f32 = max(max(_S739.tension_0, _S739.shear_0), _S739.compression_0);
    var _S741 : bool = dt_17 > 0.0f;
    var dif_3 : f32;
    if(_S741)
    {
        var raw_2 : f32 = fdiv_0(max(fdiv_0(_S740 - st_10.governing_stress_0, dt_17), 0.0f), (*mat_18).misc_0.w);
        var tau_4 : f32 = _S738.z;
        if(((flags_4 & (u32(16)))) != u32(0))
        {
            dif_3 = - expm1_accurate_0(- fdiv_0(dt_17, tau_4));
        }
        else
        {
            dif_3 = min(fdiv_0(dt_17, tau_4), 1.0f);
        }
        st_10.strain_rate_0 = st_10.strain_rate_0 + (raw_2 - st_10.strain_rate_0) * dif_3;
        st_10.governing_stress_0 = _S740;
    }
    if(((flags_4 & (u32(32)))) != u32(0))
    {
        var _S742 : f32 = dif_factor_1(&((*mat_18)), st_10.strain_rate_0);
        dif_3 = _S742;
    }
    else
    {
        dif_3 = 1.0f;
    }
    var weibull_2 : f32 = (*b_53).geom1_0.w;
    var _S743 : f32 = fatigue_factor_2(&((*mat_18)), st_10.life_0);
    var multiplier_4 : f32 = weibull_2 * dif_3 * _S743;
    var _S744 : vec4<f32> = failure_indices_1(&((*mat_18)), &((*b_53)), _S739, multiplier_4);
    var _S745 : f32 = _S744.x;
    var _S746 : f32 = _S744.y;
    st_10.utilization_0 = max(max(_S745, _S746), max(_S744.z, _S744.w));
    var _S747 : f32 = d_lin_9.x;
    var _S748 : f32 = d_lin_9.y;
    var _S749 : f32 = d_ang_8.z;
    var _S750 : f32 = ks_6 * (sq_0(_S747) + sq_0(_S748)) + kb1_2 * sq_0(d_ang_8.x) + kb2_2 * sq_0(d_ang_8.y) + kt_6 * sq_0(_S749);
    var _S751 : f32 = d_lin_9.z;
    var _S752 : bool = _S751 > 0.0f;
    if(_S752)
    {
        dif_3 = kn_7 * sq_0(_S751);
    }
    else
    {
        dif_3 = 0.0f;
    }
    var psi_ts_2 : f32 = 0.5f * (_S750 + dif_3);
    var psi_c_2 : f32;
    if(_S751 < 0.0f)
    {
        psi_c_2 = 0.5f * kn_7 * sq_0(_S751);
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
        var _S753 : vec4<f32> = contact_normal_1(&((*mat_18)), &((*b_53)), state_4.crush_1, d_lin_9, d_ang_8);
        patch_2 = _S753.w + 0.5f * (ks_6 * (sq_0(_S747 - state_4.plastic_x_0) + sq_0(_S748 - state_4.plastic_y_0)) + kt_6 * sq_0(_S749 - state_4.plastic_t_0));
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
        rebar_2 = 0.5f * ((*b_53).rebar0_0.x * sq_0(_S751 - state_4.rebar_plastic_0) + (*b_53).rebar0_0.y * (sq_0(_S747 - state_4.rebar_slip0_0) + sq_0(_S748 - state_4.rebar_slip1_0)));
    }
    else
    {
        rebar_2 = 0.0f;
    }
    var _S754 : f32 = 1.0f - state_4.damage_0;
    var frozen_2 : f32 = _S754 * (psi_ts_2 + (1.0f - state_4.crush_1) * psi_c_2) + state_4.damage_0 * patch_2 + rebar_2;
    var diss_contact_2 : f32;
    var psi_contact_2 : f32;
    var intact_normal_2 : f32;
    var dissipated_17 : f32;
    var qc_lin_2 : vec3<f32>;
    if(fracture_4)
    {
        var _S755 : bool = _S745 >= _S746;
        if(_S755)
        {
            diss_contact_2 = _S745;
        }
        else
        {
            diss_contact_2 = _S746;
        }
        var mode_ts_2 : u32;
        if(_S755)
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
                intact_normal_2 = fdiv_0(psi_contact_2 * (*b_53).geom0_0.x * diss_contact_2 * diss_contact_2, psi_ts_2);
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
            var _S756 : f32 = inc_4.x;
            if(_S756 > (st_10.damage_0))
            {
                var _S757 : Contact_0 = contact_part_1(&((*mat_18)), &((*b_53)), st_10.crush_1, plastic_8, d_lin_9, d_ang_8);
                var _S758 : f32 = max(inc_4.y - max(_S757.energy_5 - (1.0f - st_10.crush_1) * psi_c_2, 0.0f) * (_S756 - st_10.damage_0), 0.0f);
                st_10.damage_0 = _S756;
                st_10.mode_0 = mode_ts_2;
                dissipated_17 = _S758;
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
            var _S759 : Contact_0 = contact_part_1(&((*mat_18)), &((*b_53)), state_4.crush_1, vec3<f32>(state_4.plastic_x_0, state_4.plastic_y_0, state_4.plastic_t_0), d_lin_9, d_ang_8);
            qc_lin_2 = qe_ang_2 * vec3<f32>(_S754) + _S759.q_ang_1 * vec3<f32>(state_4.damage_0);
        }
        else
        {
            qc_lin_2 = qe_ang_2;
        }
        var _S760 : Measures_0 = stress_measures_1(&((*b_53)), vec3<f32>(0.0f, 0.0f, min(qe_lin_2.z, 0.0f)), qc_lin_2);
        var _S761 : vec4<f32> = failure_indices_1(&((*mat_18)), &((*b_53)), _S760, multiplier_4);
        var _S762 : f32 = _S761.z;
        var _S763 : f32 = _S761.w;
        var _S764 : bool = _S762 >= _S763;
        if(_S764)
        {
            psi_contact_2 = _S762;
        }
        else
        {
            psi_contact_2 = _S763;
        }
        if(_S764)
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
                intact_normal_2 = fdiv_0((*mat_18).energy_1.w * (*b_53).geom0_0.x * psi_contact_2 * psi_contact_2, psi_c_2);
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
            var _S765 : f32 = inc_5.x;
            if(_S765 > (st_10.crush_1))
            {
                var dissipated_18 : f32 = dissipated_17 + inc_5.y;
                st_10.crush_1 = _S765;
                st_10.mode_0 = mode_c_2;
                if(_S765 >= 1.0f)
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
    var _S766 : vec3<f32> = vec3<f32>(0.0f);
    var qc_ang_2 : vec3<f32>;
    if((st_10.damage_0) == 0.0f)
    {
        if(((flags_4 & (u32(8)))) == u32(0))
        {
            var _S767 : vec3<f32> = contact_offsets_1(&((*mat_18)), &((*b_53)), st_10.crush_1, plastic_8, d_lin_9, d_ang_8);
            st_10.plastic_x_0 = _S767.x;
            st_10.plastic_y_0 = _S767.y;
            st_10.plastic_t_0 = _S767.z;
        }
        diss_contact_2 = 0.0f;
        qc_lin_2 = _S766;
        qc_ang_2 = _S766;
        psi_contact_2 = 0.0f;
    }
    else
    {
        var _S768 : Contact_0 = contact_part_1(&((*mat_18)), &((*b_53)), st_10.crush_1, plastic_8, d_lin_9, d_ang_8);
        st_10.plastic_x_0 = _S768.plastic_0.x;
        st_10.plastic_y_0 = _S768.plastic_0.y;
        st_10.plastic_t_0 = _S768.plastic_0.z;
        diss_contact_2 = _S768.diss_0;
        qc_lin_2 = _S768.q_lin_1;
        qc_ang_2 = _S768.q_ang_1;
        psi_contact_2 = _S768.energy_5;
    }
    var dissipated_20 : f32 = dissipated_17 + dmg_2 * diss_contact_2;
    if(_S752)
    {
        intact_normal_2 = kn_7 * _S751;
    }
    else
    {
        intact_normal_2 = (1.0f - st_10.crush_1) * kn_7 * _S751;
    }
    var _S769 : f32 = 1.0f - dmg_2;
    var force_lin_6 : vec3<f32> = vec3<f32>(_S769 * qe_lin_2.x + dmg_2 * qc_lin_2.x, _S769 * qe_lin_2.y + dmg_2 * qc_lin_2.y, _S769 * intact_normal_2 + dmg_2 * qc_lin_2.z);
    var force_ang_4 : vec3<f32> = qe_ang_2 * vec3<f32>(_S769) + qc_ang_2 * vec3<f32>(dmg_2);
    var stored_12 : f32 = _S769 * (psi_ts_2 + (1.0f - st_10.crush_1) * psi_c_2) + dmg_2 * psi_contact_2;
    if(has_rebar_7)
    {
        changed_2 = (st_10.rebar_broken_0) == 0.0f;
    }
    else
    {
        changed_2 = false;
    }
    var stored_13 : f32;
    var force_lin_7 : vec3<f32>;
    if(changed_2)
    {
        var k_axial_2 : f32 = (*b_53).rebar0_0.x;
        var k_dowel_2 : f32 = (*b_53).rebar0_0.y;
        var yield_force_2 : f32 = (*b_53).rebar0_0.z;
        var dowel_capacity_2 : f32 = (*b_53).rebar0_0.w;
        var nr_2 : vec2<f32> = return_map_0(k_axial_2, _S751, st_10.rebar_plastic_0, yield_force_2);
        var v1_2 : vec2<f32> = return_map_0(k_dowel_2, _S747, st_10.rebar_slip0_0, dowel_capacity_2);
        var v2_2 : vec2<f32> = return_map_0(k_dowel_2, _S748, st_10.rebar_slip1_0, dowel_capacity_2);
        var _S770 : f32 = nr_2.y;
        var _S771 : f32 = v1_2.y;
        var _S772 : f32 = v2_2.y;
        var work_6 : f32 = yield_force_2 * abs(_S770) + dowel_capacity_2 * (abs(_S771) + abs(_S772));
        st_10.rebar_plastic_0 = st_10.rebar_plastic_0 + _S770;
        st_10.rebar_slip0_0 = st_10.rebar_slip0_0 + _S771;
        st_10.rebar_slip1_0 = st_10.rebar_slip1_0 + _S772;
        st_10.rebar_work_0 = st_10.rebar_work_0 + work_6;
        var dissipated_21 : f32 = dissipated_20 + work_6;
        var _S773 : f32 = nr_2.x;
        var _S774 : f32 = v1_2.x;
        var _S775 : f32 = v2_2.x;
        var elastic_2 : f32 = 0.5f * (fdiv_0(sq_0(_S773), k_axial_2) + fdiv_0(sq_0(_S774) + sq_0(_S775), k_dowel_2));
        if(fracture_4)
        {
            changed_2 = (st_10.rebar_work_0) >= ((*b_53).rebar1_0.x);
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
            stored_13 = stored_12;
            dissipated_17 = dissipated_22;
        }
        else
        {
            var stored_14 : f32 = stored_12 + elastic_2;
            force_lin_7 = force_lin_6 + vec3<f32>(_S774, _S775, _S773);
            stored_13 = stored_14;
            dissipated_17 = dissipated_21;
        }
    }
    else
    {
        force_lin_7 = force_lin_6;
        stored_13 = stored_12;
        dissipated_17 = dissipated_20;
    }
    if(fracture_4)
    {
        changed_2 = _S741;
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
        var _S776 : Measures_0 = stress_measures_1(&((*b_53)), force_lin_7, force_ang_4);
        var _S777 : vec4<f32> = failure_indices_1(&((*mat_18)), &((*b_53)), _S776, weibull_2);
        var _S778 : f32 = life_rate_1(&((*mat_18)), max(max(_S777.x, _S777.y), _S777.z));
        st_10.life_0 = max(st_10.life_0 - _S778 * dt_17, 0.0f);
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
        released_2 = frozen_2 - stored_13;
    }
    else
    {
        released_2 = 0.0f;
    }
    var _S779 : f32 = max(min(dissipated_17, released_2), 0.0f);
    var overshoot_3 : f32 = released_2 - _S779;
    st_10.dissipated_0 = st_10.dissipated_0 + _S779;
    var resp_2 : JointResponse_0;
    resp_2.force_lin_1 = force_lin_7;
    resp_2.force_ang_1 = force_ang_4;
    resp_2.state_1 = st_10;
    resp_2.dissipated_1 = _S779;
    resp_2.overshoot_0 = overshoot_3;
    resp_2.stored_0 = stored_13;
    if(was_connected_0)
    {
        changed_2 = !connected_2(st_10, has_rebar_7);
    }
    else
    {
        changed_2 = false;
    }
    resp_2.disconnected_0 = changed_2;
    resp_2.fatigue_before_0 = _S743;
    resp_2.measures_0 = _S739;
    return resp_2;
}

fn static_kinematics_0( _S780 : u32,  _S781 : ptr<function, vec3<f32>>,  _S782 : ptr<function, vec3<f32>>)
{
    var ca_1 : u32 = bonds_0[_S780].law_0.ids_0.y;
    var cb_4 : u32 = bonds_0[_S780].law_0.ids_0.z;
    var _S783 : u32 = u32(4) * cb_4;
    var _S784 : u32 = u32(4) * ca_1;
    var _S785 : u32 = _S783 + u32(1);
    var _S786 : u32 = _S784 + u32(1);
    var _S787 : u32 = sv_0(cb_4, u32(22));
    var _S788 : u32 = sv_0(ca_1, u32(22));
    var dth_0 : vec3<f32> = state_0[_S785].xyz - state_0[_S786].xyz + (scratch_0[_S787].xyz - scratch_0[_S788].xyz);
    (*_S781) = to_local_0(_S780, state_0[_S783].xyz - state_0[_S784].xyz + (scratch_0[sv_0(cb_4, u32(21))].xyz - scratch_0[sv_0(ca_1, u32(21))].xyz) + (cross(state_0[_S785].xyz + scratch_0[_S787].xyz, bonds_0[_S780].rb_0.xyz) - cross(state_0[_S786].xyz + scratch_0[_S788].xyz, bonds_0[_S780].ra_0.xyz)));
    (*_S782) = to_local_0(_S780, dth_0);
    return;
}

fn static_response_0( i_13 : u32) -> JointResponse_0
{
    var d_lin_10 : vec3<f32>;
    var d_ang_9 : vec3<f32>;
    static_kinematics_0(i_13, &(d_lin_10), &(d_ang_9));
    var _S789 : JointBond_std430_0 = bonds_0[i_13].law_0;
    var _S790 : JointMaterial_std140_0 = materials_0.m_0.data_0[_S789.ids_0.x];
    var _S791 : JointState_std430_0 = bond_dyn_0[i_13].js_0;
    var _S792 : JointResponse_0 = joint_evaluate_1(&(_S790), &(_S789), &(_S791), d_lin_10, d_ang_9, 0.0f, false);
    return _S792;
}

fn gather_loads_0( c_33 : u32,  fi_2 : ptr<function, vec3<f32>>,  mi_3 : ptr<function, vec3<f32>>)
{
    var _S793 : vec3<f32> = vec3<f32>(0.0f);
    (*fi_2) = _S793;
    (*mi_3) = _S793;
    var e_5 : u32 = index_0[c_33];
    loop
    {
        if(e_5 < index_0[c_33 + u32(1)])
        {
        }
        else
        {
            break;
        }
        var entry_4 : u32 = index_0[e_5];
        var bond_2 : u32 = (entry_4 >> (u32(1)));
        if(((entry_4 & (u32(1)))) == u32(0))
        {
            var _S794 : u32 = u32(3) * bond_2;
            (*fi_2) = (*fi_2) + scratch_0[_S794].xyz;
            (*mi_3) = (*mi_3) + scratch_0[_S794 + u32(1)].xyz;
        }
        else
        {
            var _S795 : u32 = u32(3) * bond_2;
            (*fi_2) = (*fi_2) - scratch_0[_S795].xyz;
            (*mi_3) = (*mi_3) + scratch_0[_S795 + u32(2)].xyz;
        }
        e_5 = e_5 + u32(1);
    }
    return;
}

fn bond_load_magnitude2_0( c_34 : u32) -> f32
{
    var e_6 : u32 = index_0[c_34];
    var m_7 : f32 = 0.0f;
    loop
    {
        if(e_6 < index_0[c_34 + u32(1)])
        {
        }
        else
        {
            break;
        }
        var entry_5 : u32 = index_0[e_6];
        var _S796 : u32 = u32(3) * ((entry_5 >> (u32(1))));
        var f_20 : vec3<f32> = scratch_0[_S796].xyz;
        var t_21 : vec3<f32>;
        if(((entry_5 & (u32(1)))) == u32(0))
        {
            t_21 = scratch_0[_S796 + u32(1)].xyz;
        }
        else
        {
            t_21 = scratch_0[_S796 + u32(2)].xyz;
        }
        var m_8 : f32 = m_7 + (dot(f_20, f_20) + dot(t_21, t_21));
        e_6 = e_6 + u32(1);
        m_7 = m_8;
    }
    return m_7;
}

fn fixed_mask_0( c_35 : u32) -> u32
{
    var support_1 : u32 = chunks_0[c_35].info_1.x;
    var _S797 : u32;
    if(support_1 == u32(1))
    {
        _S797 = u32(63);
    }
    else
    {
        if(support_1 == u32(2))
        {
            _S797 = u32(7);
        }
        else
        {
            _S797 = u32(0);
        }
    }
    return _S797;
}

fn hold_0( mask_0 : u32,  lin_0 : ptr<function, vec4<f32>>,  ang_0 : ptr<function, vec4<f32>>,  keep_lin_0 : vec4<f32>,  keep_ang_0 : vec4<f32>)
{
    var d_19 : u32 = u32(0);
    loop
    {
        if(d_19 < u32(3))
        {
        }
        else
        {
            break;
        }
        if(((mask_0 & (((u32(1) << (d_19)))))) != u32(0))
        {
            (*lin_0)[d_19] = keep_lin_0[d_19];
        }
        if(((mask_0 & (((u32(1) << ((d_19 + u32(3)))))))) != u32(0))
        {
            (*ang_0)[d_19] = keep_ang_0[d_19];
        }
        d_19 = d_19 + u32(1);
    }
    return;
}

fn is_finite_0( x_24 : f32) -> bool
{
    return (((bitcast<u32>((x_24))) & (u32(2147483647)))) < u32(2139095040);
}

fn secant_factors_1( b_54 : ptr<function, JointBond_std430_0>,  st_11 : ptr<function, JointState_std430_0>,  d_lin_11 : vec3<f32>,  f_lin_2 : ptr<function, vec3<f32>>,  f_ang_2 : ptr<function, vec3<f32>>)
{
    var _S798 : f32 = (*st_11).damage_0;
    var compressed_1 : bool = (d_lin_11.z) < 0.0f;
    var contact_4 : f32;
    if(compressed_1)
    {
        contact_4 = _S798;
    }
    else
    {
        contact_4 = 0.0f;
    }
    var _S799 : f32 = 1.0f - _S798;
    var _S800 : f32 = max(_S799 + contact_4, 9.99999997475242708e-07f);
    var normal_8 : f32;
    if(compressed_1)
    {
        normal_8 = max(1.0f - (*st_11).crush_1, 9.99999997475242708e-07f);
    }
    else
    {
        normal_8 = max(_S799, 9.99999997475242708e-07f);
    }
    (*f_lin_2) = vec3<f32>(_S800, _S800, normal_8);
    (*f_ang_2) = vec3<f32>(_S800);
    var _S801 : bool;
    if(((*b_54).stiff1_0.w) != 0.0f)
    {
        _S801 = ((*st_11).rebar_broken_0) == 0.0f;
    }
    else
    {
        _S801 = false;
    }
    if(_S801)
    {
        var _S802 : vec4<f32> = (*b_54).rebar0_0;
        var _S803 : vec4<f32> = (*b_54).stiff0_0;
        (*f_lin_2)[i32(2)] = (*f_lin_2)[i32(2)] + fdiv_0((*b_54).rebar0_0.x, (*b_54).stiff0_0.x);
        var _S804 : f32 = fdiv_0(_S802.y, _S803.y);
        (*f_lin_2)[i32(0)] = (*f_lin_2)[i32(0)] + _S804;
        (*f_lin_2)[i32(1)] = (*f_lin_2)[i32(1)] + _S804;
    }
    return;
}

fn statics_bond_slot_0( i_14 : u32) -> u32
{
    return params_0.statics_base_0 + u32(23) * params_0.chunk_count_0 + u32(2) * i_14;
}

fn store_inverse_0( c_36 : u32,  a_27 : array<f32, i32(36)>)
{
    var j_11 : u32;
    var sum_9 : f32;
    var l_7 : array<f32, i32(36)>;
    var k_22 : u32 = u32(0);
    loop
    {
        if(k_22 < u32(36))
        {
        }
        else
        {
            break;
        }
        l_7[k_22] = 0.0f;
        k_22 = k_22 + u32(1);
    }
    var spd_0 : bool = true;
    var i_15 : u32 = u32(0);
    loop
    {
        var _S805 : bool;
        if(i_15 < u32(6))
        {
            _S805 = spd_0;
        }
        else
        {
            _S805 = false;
        }
        if(_S805)
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
            var _S806 : u32 = i_15 * u32(6);
            var _S807 : u32 = _S806 + j_11;
            k_22 = u32(0);
            sum_9 = a_27[_S807];
            loop
            {
                if(k_22 < j_11)
                {
                }
                else
                {
                    break;
                }
                var sum_10 : f32 = sum_9 - l_7[_S806 + k_22] * l_7[j_11 * u32(6) + k_22];
                k_22 = k_22 + u32(1);
                sum_9 = sum_10;
            }
            if(i_15 == j_11)
            {
                if(sum_9 <= 0.0f)
                {
                    spd_0 = false;
                    break;
                }
                l_7[_S806 + i_15] = sqrt(sum_9);
            }
            else
            {
                l_7[_S807] = sum_9 / l_7[j_11 * u32(6) + j_11];
            }
            j_11 = j_11 + u32(1);
        }
        i_15 = i_15 + u32(1);
    }
    var inv_0 : array<f32, i32(36)>;
    if(!spd_0)
    {
        k_22 = u32(0);
        loop
        {
            if(k_22 < u32(36))
            {
            }
            else
            {
                break;
            }
            inv_0[k_22] = 0.0f;
            k_22 = k_22 + u32(1);
        }
        k_22 = u32(0);
        loop
        {
            if(k_22 < u32(6))
            {
            }
            else
            {
                break;
            }
            var _S808 : u32 = k_22 * u32(6) + k_22;
            if((a_27[_S808]) > 0.0f)
            {
                sum_9 = 1.0f / a_27[_S808];
            }
            else
            {
                sum_9 = 0.0f;
            }
            inv_0[_S808] = sum_9;
            k_22 = k_22 + u32(1);
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
                    sum_9 = 1.0f;
                }
                else
                {
                    sum_9 = 0.0f;
                }
                k_22 = u32(0);
                var s_12 : f32 = sum_9;
                loop
                {
                    if(k_22 < i_15)
                    {
                    }
                    else
                    {
                        break;
                    }
                    var s_13 : f32 = s_12 - l_7[i_15 * u32(6) + k_22] * y_7[k_22];
                    k_22 = k_22 + u32(1);
                    s_12 = s_13;
                }
                y_7[i_15] = s_12 / l_7[i_15 * u32(6) + i_15];
                i_15 = i_15 + u32(1);
            }
            var x_25 : array<f32, i32(6)>;
            x_25[i32(0)] = 0.0f;
            x_25[i32(1)] = 0.0f;
            x_25[i32(2)] = 0.0f;
            x_25[i32(3)] = 0.0f;
            x_25[i32(4)] = 0.0f;
            x_25[i32(5)] = 0.0f;
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
                k_22 = i_16 + u32(1);
                sum_9 = y_7[i_16];
                loop
                {
                    if(k_22 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var s_14 : f32 = sum_9 - l_7[k_22 * u32(6) + i_16] * x_25[k_22];
                    k_22 = k_22 + u32(1);
                    sum_9 = s_14;
                }
                x_25[i_16] = sum_9 / l_7[i_16 * u32(6) + i_16];
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
                inv_0[i_17 * u32(6) + j_11] = x_25[i_17];
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
        var _S809 : u32 = u32(4) * j_11;
        scratch_0[sv_0(c_36, u32(12) + j_11)] = vec4<f32>(inv_0[_S809], inv_0[_S809 + u32(1)], inv_0[_S809 + u32(2)], inv_0[_S809 + u32(3)]);
        j_11 = j_11 + u32(1);
    }
    return;
}

fn assemble_block_0( c_37 : u32)
{
    var p_21 : u32;
    var r_15 : u32;
    var a_28 : array<f32, i32(36)>;
    var k_23 : u32 = u32(0);
    loop
    {
        if(k_23 < u32(36))
        {
        }
        else
        {
            break;
        }
        a_28[k_23] = 0.0f;
        k_23 = k_23 + u32(1);
    }
    var e_7 : u32 = index_0[c_37];
    loop
    {
        if(e_7 < index_0[c_37 + u32(1)])
        {
        }
        else
        {
            break;
        }
        var entry_6 : u32 = index_0[e_7];
        var i_18 : u32 = (entry_6 >> (u32(1)));
        var _S810 : bool = ((entry_6 & (u32(1)))) != u32(0);
        var _S811 : u32 = statics_bond_slot_0(i_18);
        var _S812 : vec4<f32> = scratch_0[_S811];
        var _S813 : vec4<f32> = scratch_0[_S811 + u32(1)];
        p_21 = u32(0);
        loop
        {
            if(p_21 < u32(6))
            {
            }
            else
            {
                break;
            }
            var _S814 : u32 = p_21 % u32(3);
            var t_22 : vec3<f32>;
            if(_S814 == u32(0))
            {
                t_22 = bonds_0[i_18].t1_0.xyz;
            }
            else
            {
                if(_S814 == u32(1))
                {
                    t_22 = bonds_0[i_18].t2_0.xyz;
                }
                else
                {
                    t_22 = bonds_0[i_18].normal_0.xyz;
                }
            }
            var _S815 : bool = p_21 < u32(3);
            var row_u_0 : vec3<f32>;
            var row_t_0 : vec3<f32>;
            if(_S815)
            {
                if(_S810)
                {
                    row_u_0 = t_22;
                }
                else
                {
                    row_u_0 = (vec3<f32>(0) - t_22);
                }
                if(_S810)
                {
                    row_t_0 = cross(bonds_0[i_18].rb_0.xyz, t_22);
                }
                else
                {
                    row_t_0 = (vec3<f32>(0) - cross(bonds_0[i_18].ra_0.xyz, t_22));
                }
            }
            else
            {
                var _S816 : vec3<f32> = vec3<f32>(0.0f);
                if(_S810)
                {
                    row_u_0 = t_22;
                }
                else
                {
                    row_u_0 = (vec3<f32>(0) - t_22);
                }
                var _S817 : vec3<f32> = row_u_0;
                row_u_0 = _S816;
                row_t_0 = _S817;
            }
            var kp_0 : f32;
            if(_S815)
            {
                kp_0 = _S812[p_21];
            }
            else
            {
                kp_0 = _S813[p_21 - u32(3)];
            }
            if(kp_0 == 0.0f)
            {
                p_21 = p_21 + u32(1);
                continue;
            }
            var _S818 : array<f32, i32(6)> = array<f32, i32(6)>( row_u_0.x, row_u_0.y, row_u_0.z, row_t_0.x, row_t_0.y, row_t_0.z );
            r_15 = u32(0);
            loop
            {
                if(r_15 < u32(6))
                {
                }
                else
                {
                    break;
                }
                var q_13 : u32 = u32(0);
                loop
                {
                    if(q_13 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    a_28[r_15 * u32(6) + q_13] = a_28[r_15 * u32(6) + q_13] + kp_0 * _S818[r_15] * _S818[q_13];
                    q_13 = q_13 + u32(1);
                }
                r_15 = r_15 + u32(1);
            }
            p_21 = p_21 + u32(1);
        }
        e_7 = e_7 + u32(1);
    }
    var _S819 : u32 = fixed_mask_0(c_37);
    p_21 = u32(0);
    loop
    {
        if(p_21 < u32(6))
        {
        }
        else
        {
            break;
        }
        if(((_S819 & (((u32(1) << (p_21)))))) != u32(0))
        {
            r_15 = u32(0);
            loop
            {
                if(r_15 < u32(6))
                {
                }
                else
                {
                    break;
                }
                a_28[p_21 * u32(6) + r_15] = 0.0f;
                a_28[r_15 * u32(6) + p_21] = 0.0f;
                r_15 = r_15 + u32(1);
            }
            a_28[p_21 * u32(6) + p_21] = 1.0f;
        }
        p_21 = p_21 + u32(1);
    }
    p_21 = u32(0);
    loop
    {
        if(p_21 < u32(6))
        {
        }
        else
        {
            break;
        }
        if((a_28[p_21 * u32(6) + p_21]) == 0.0f)
        {
            a_28[p_21 * u32(6) + p_21] = 1.0f;
        }
        p_21 = p_21 + u32(1);
    }
    store_inverse_0(c_37, a_28);
    return;
}

fn block_get_0( c_38 : u32,  i_19 : u32,  j_12 : u32) -> f32
{
    var k_24 : u32 = i_19 * u32(6) + j_12;
    return scratch_0[sv_0(c_38, u32(12) + k_24 / u32(4))][k_24 % u32(4)];
}

fn precondition_0( c_39 : u32)
{
    var _S820 : array<f32, i32(6)> = array<f32, i32(6)>( scratch_0[sv_0(c_39, u32(4))].x, scratch_0[sv_0(c_39, u32(4))].y, scratch_0[sv_0(c_39, u32(4))].z, scratch_0[sv_0(c_39, u32(5))].x, scratch_0[sv_0(c_39, u32(5))].y, scratch_0[sv_0(c_39, u32(5))].z );
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
        var s_15 : f32 = 0.0f;
        loop
        {
            if(j_13 < u32(6))
            {
            }
            else
            {
                break;
            }
            var s_16 : f32 = s_15 + block_get_0(c_39, i_20, j_13) * _S820[j_13];
            j_13 = j_13 + u32(1);
            s_15 = s_16;
        }
        z_1[i_20] = s_15;
        i_20 = i_20 + u32(1);
    }
    scratch_0[sv_0(c_39, u32(6))] = vec4<f32>(z_1[i32(0)], z_1[i32(1)], z_1[i32(2)], 0.0f);
    scratch_0[sv_0(c_39, u32(7))] = vec4<f32>(z_1[i32(3)], z_1[i32(4)], z_1[i32(5)], 0.0f);
    return;
}

fn project_displacement_slot_0( tid_25 : u32,  isl_33 : ptr<function, Island_std430_0>,  slot_5 : u32)
{
    var _S821 : u32;
    var _S822 : vec3<f32> = vec3<f32>(0.0f);
    var p_22 : vec3<f32> = _S822;
    var l_8 : vec3<f32> = _S822;
    var _S823 : vec4<u32> = (*isl_33).range_0;
    var _S824 : u32 = (*isl_33).range_0.x + tid_25;
    var c_40 : u32 = _S824;
    loop
    {
        var _S825 : u32 = _S823.y;
        _S821 = _S825;
        if(c_40 < _S825)
        {
        }
        else
        {
            break;
        }
        var u_3 : vec3<f32> = scratch_0[sv_0(c_40, slot_5)].xyz;
        var th_5 : vec3<f32> = scratch_0[sv_0(c_40, slot_5 + u32(1))].xyz;
        var r_16 : vec3<f32> = chunks_0[c_40].center_0.xyz - (*isl_33).com_0.xyz;
        var _S826 : vec3<f32> = vec3<f32>(chunks_0[c_40].center_0.w);
        p_22 = p_22 + u_3 * _S826;
        l_8 = l_8 + (cross(r_16, u_3) * _S826 + rows_mul_0(chunks_0[c_40].inertia0_1, chunks_0[c_40].inertia1_1, chunks_0[c_40].inertia2_1, th_5));
        c_40 = c_40 + u32(256);
    }
    group_sum3_3(tid_25, &(p_22), &(l_8));
    var _S827 : vec4<f32> = (*isl_33).com_0;
    var _S828 : vec3<f32> = p_22 / vec3<f32>((*isl_33).com_0.w);
    var _S829 : vec3<f32> = rows_mul_0((*isl_33).inv0_0, (*isl_33).inv1_0, (*isl_33).inv2_0, l_8);
    c_40 = _S824;
    loop
    {
        if(c_40 < _S821)
        {
        }
        else
        {
            break;
        }
        var _S830 : u32 = sv_0(c_40, slot_5);
        scratch_0[_S830] = vec4<f32>(scratch_0[_S830].xyz - _S828 - cross(_S829, chunks_0[c_40].center_0.xyz - _S827.xyz), 0.0f);
        var _S831 : u32 = sv_0(c_40, slot_5 + u32(1));
        scratch_0[_S831] = vec4<f32>(scratch_0[_S831].xyz - _S829, 0.0f);
        c_40 = c_40 + u32(256);
    }
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    return;
}

fn statics_result_slot_0( island_3 : u32) -> u32
{
    return params_0.statics_base_0 + u32(23) * params_0.chunk_count_0 + u32(2) * params_0.statics_bonds_0 + island_3;
}

fn write_bond_loads_0( _S832 : u32,  _S833 : u32,  _S834 : vec3<f32>,  _S835 : vec3<f32>,  _S836 : f32)
{
    var _S837 : vec3<f32> = to_body_0(_S833, _S834);
    var _S838 : vec3<f32> = to_body_0(_S833, _S835);
    var _S839 : u32 = u32(3) * _S832;
    scratch_0[_S839] = vec4<f32>(_S837, _S836);
    scratch_0[_S839 + u32(1)] = vec4<f32>(_S838 + cross(bonds_0[_S833].ra_0.xyz, _S837), 0.0f);
    scratch_0[_S839 + u32(2)] = vec4<f32>((vec3<f32>(0) - _S838) + cross(bonds_0[_S833].rb_0.xyz, (vec3<f32>(0) - _S837)), 0.0f);
    return;
}

fn static_kinematics_1( _S840 : u32,  _S841 : ptr<function, vec3<f32>>,  _S842 : ptr<function, vec3<f32>>)
{
    var ca_2 : u32 = bonds_0[_S840].law_0.ids_0.y;
    var cb_5 : u32 = bonds_0[_S840].law_0.ids_0.z;
    var _S843 : u32 = u32(4) * cb_5;
    var _S844 : u32 = u32(4) * ca_2;
    var _S845 : u32 = _S843 + u32(1);
    var _S846 : u32 = _S844 + u32(1);
    var _S847 : u32 = sv_0(cb_5, u32(22));
    var _S848 : u32 = sv_0(ca_2, u32(22));
    var dth_1 : vec3<f32> = state_0[_S845].xyz - state_0[_S846].xyz + (scratch_0[_S847].xyz - scratch_0[_S848].xyz);
    (*_S841) = to_local_0(_S840, state_0[_S843].xyz - state_0[_S844].xyz + (scratch_0[sv_0(cb_5, u32(21))].xyz - scratch_0[sv_0(ca_2, u32(21))].xyz) + (cross(state_0[_S845].xyz + scratch_0[_S847].xyz, bonds_0[_S840].rb_0.xyz) - cross(state_0[_S846].xyz + scratch_0[_S848].xyz, bonds_0[_S840].ra_0.xyz)));
    (*_S842) = to_local_0(_S840, dth_1);
    return;
}

fn bond_kinematics_0( _S849 : u32,  _S850 : vec3<f32>,  _S851 : vec3<f32>,  _S852 : vec3<f32>,  _S853 : vec3<f32>,  _S854 : ptr<function, vec3<f32>>,  _S855 : ptr<function, vec3<f32>>)
{
    (*_S854) = to_local_0(_S849, _S852 + cross(_S853, bonds_0[_S849].rb_0.xyz) - (_S850 + cross(_S851, bonds_0[_S849].ra_0.xyz)));
    (*_S855) = to_local_0(_S849, _S853 - _S851);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn island_statics(@builtin(workgroup_id) group_5 : vec3<u32>, @builtin(local_invocation_id) thread_5 : vec3<u32>)
{
    var i_21 : u32;
    var converged_4 : bool;
    var c_41 : u32;
    var tid_26 : u32 = thread_5.x;
    var _S856 : u32 = group_5.x;
    var _S857 : Island_std430_0 = islands_0[_S856];
    var _S858 : vec4<u32> = _S857.info_0;
    var _S859 : u32 = _S857.info_0.z;
    if(((_S859 & (u32(8)))) == u32(0))
    {
        return;
    }
    var free_0 : bool = (((_S858.x) & (u32(1)))) == u32(0);
    var c0_1 : u32 = _S857.range_0.x;
    var c1_1 : u32 = _S857.range_0.y;
    var b0_0 : u32 = _S857.range_0.z;
    var _S860 : u32 = _S857.range_0.w;
    var _S861 : u32 = c0_1 + tid_26;
    var c_42 : u32 = _S861;
    loop
    {
        if(c_42 < c1_1)
        {
        }
        else
        {
            break;
        }
        var _S862 : vec4<f32> = vec4<f32>(0.0f);
        scratch_0[sv_0(c_42, u32(21))] = _S862;
        scratch_0[sv_0(c_42, u32(22))] = _S862;
        c_42 = c_42 + u32(256);
    }
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    if(free_0)
    {
        project_load_slot_0(tid_26, &(_S857), u32(0));
    }
    var _S863 : f32 = island_dot_0(tid_26, c0_1, c1_1, u32(0), u32(0));
    var _S864 : f32 = max(sqrt(_S863), 1.00000000317107685e-30f);
    var _S865 : f32 = params_0.statics_tol_0;
    var _S866 : u32 = min(params_0.statics_cg_0, u32(20) * (c1_1 - c0_1) * u32(6) + u32(200));
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
        var _S867 : u32 = b0_0 + tid_26;
        i_21 = _S867;
        loop
        {
            if(i_21 < _S860)
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
        var _S868 : vec4<f32> = vec4<f32>(0.0f);
        var magnitude_0 : vec4<f32> = _S868;
        var unused_m_0 : vec4<f32> = _S868;
        c_42 = _S861;
        loop
        {
            if(c_42 < c1_1)
            {
            }
            else
            {
                break;
            }
            var fi_3 : vec3<f32>;
            var mi_4 : vec3<f32>;
            gather_loads_0(c_42, &(fi_3), &(mi_4));
            magnitude_0[i32(0)] = magnitude_0[i32(0)] + bond_load_magnitude2_0(c_42);
            var r_lin_0 : vec4<f32> = vec4<f32>(scratch_0[sv_0(c_42, u32(0))].xyz + fi_3, 0.0f);
            var r_ang_0 : vec4<f32> = vec4<f32>(scratch_0[sv_0(c_42, u32(1))].xyz + mi_4, 0.0f);
            hold_0(fixed_mask_0(c_42), &(r_lin_0), &(r_ang_0), _S868, _S868);
            scratch_0[sv_0(c_42, u32(4))] = r_lin_0;
            scratch_0[sv_0(c_42, u32(5))] = r_ang_0;
            c_42 = c_42 + u32(256);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        group_sum2_1(tid_26, &(magnitude_0), &(unused_m_0));
        if(free_0)
        {
            project_load_slot_0(tid_26, &(_S857), u32(4));
        }
        var _S869 : f32 = island_dot_0(tid_26, c0_1, c1_1, u32(4), u32(4));
        var residual_1 : f32 = sqrt(_S869) / _S864;
        var _S870 : f32 = max(0.00100000004749745f, 3.83999986297567375e-06f * sqrt(magnitude_0.x) / _S864);
        if(newton_0 == u32(0))
        {
            start_0 = residual_1;
        }
        var _S871 : bool = is_finite_0(residual_1);
        var _S872 : bool;
        if(_S871)
        {
            if(residual_1 <= _S865)
            {
                converged_4 = true;
            }
            else
            {
                if(residual_1 <= _S870)
                {
                    converged_4 = residual_1 > (0.5f * previous_1);
                }
                else
                {
                    converged_4 = false;
                }
                if(converged_4)
                {
                    _S872 = residual_1 <= start_0;
                }
                else
                {
                    _S872 = false;
                }
                converged_4 = _S872;
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
        if(!_S871)
        {
            residual_0 = residual_1;
            converged_4 = false;
            break;
        }
        var i_22 : u32 = _S867;
        loop
        {
            if(i_22 < _S860)
            {
            }
            else
            {
                break;
            }
            var d_lin_12 : vec3<f32>;
            var d_ang_10 : vec3<f32>;
            static_kinematics_1(i_22, &(d_lin_12), &(d_ang_10));
            var _S873 : JointBond_std430_0 = bonds_0[i_22].law_0;
            var _S874 : JointState_std430_0 = bond_dyn_0[i_22].js_0;
            var f_lin_3 : vec3<f32>;
            var f_ang_3 : vec3<f32>;
            secant_factors_1(&(_S873), &(_S874), d_lin_12, &(f_lin_3), &(f_ang_3));
            var _S875 : u32 = statics_bond_slot_0(i_22);
            var _S876 : f32 = _S873.stiff0_0.y;
            scratch_0[_S875] = vec4<f32>(_S876 * f_lin_3.x, _S876 * f_lin_3.y, _S873.stiff0_0.x * f_lin_3.z, 0.0f);
            scratch_0[_S875 + u32(1)] = vec4<f32>(_S873.stiff0_0.z * f_ang_3.x, _S873.stiff0_0.w * f_ang_3.y, _S873.stiff1_0.x * f_ang_3.z, 0.0f);
            i_22 = i_22 + u32(256);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        var c_43 : u32 = _S861;
        loop
        {
            if(c_43 < c1_1)
            {
            }
            else
            {
                break;
            }
            assemble_block_0(c_43);
            scratch_0[sv_0(c_43, u32(2))] = _S868;
            scratch_0[sv_0(c_43, u32(3))] = _S868;
            c_43 = c_43 + u32(256);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        var c_44 : u32 = _S861;
        loop
        {
            if(c_44 < c1_1)
            {
            }
            else
            {
                break;
            }
            precondition_0(c_44);
            c_44 = c_44 + u32(256);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        if(free_0)
        {
            project_displacement_slot_0(tid_26, &(_S857), u32(6));
        }
        var c_45 : u32 = _S861;
        loop
        {
            if(c_45 < c1_1)
            {
            }
            else
            {
                break;
            }
            scratch_0[sv_0(c_45, u32(8))] = scratch_0[sv_0(c_45, u32(6))];
            scratch_0[sv_0(c_45, u32(9))] = scratch_0[sv_0(c_45, u32(7))];
            c_45 = c_45 + u32(256);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        var _S877 : f32 = island_dot_0(tid_26, c0_1, c1_1, u32(4), u32(6));
        var _S878 : f32 = island_dot_0(tid_26, c0_1, c1_1, u32(4), u32(4));
        var _S879 : f32 = sqrt(_S878);
        var rz_0 : f32 = _S877;
        var k_25 : u32 = u32(0);
        var cg_total_1 : u32 = cg_total_0;
        loop
        {
            if(k_25 < _S866)
            {
                _S872 = _S879 > 0.0f;
            }
            else
            {
                _S872 = false;
            }
            if(_S872)
            {
            }
            else
            {
                cg_total_0 = cg_total_1;
                break;
            }
            var i_23 : u32 = _S867;
            loop
            {
                if(i_23 < _S860)
                {
                }
                else
                {
                    break;
                }
                var ca_3 : u32 = bonds_0[i_23].law_0.ids_0.y;
                var cb_6 : u32 = bonds_0[i_23].law_0.ids_0.z;
                var d_lin_13 : vec3<f32>;
                var d_ang_11 : vec3<f32>;
                bond_kinematics_0(i_23, scratch_0[sv_0(ca_3, u32(8))].xyz, scratch_0[sv_0(ca_3, u32(9))].xyz, scratch_0[sv_0(cb_6, u32(8))].xyz, scratch_0[sv_0(cb_6, u32(9))].xyz, &(d_lin_13), &(d_ang_11));
                var _S880 : u32 = statics_bond_slot_0(i_23);
                write_bond_loads_0(i_23, i_23, d_lin_13 * scratch_0[_S880].xyz, d_ang_11 * scratch_0[_S880 + u32(1)].xyz, 0.0f);
                i_23 = i_23 + u32(256);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
            c_41 = _S861;
            loop
            {
                if(c_41 < c1_1)
                {
                }
                else
                {
                    break;
                }
                var fi_4 : vec3<f32>;
                var mi_5 : vec3<f32>;
                gather_loads_0(c_41, &(fi_4), &(mi_5));
                var ap_lin_0 : vec4<f32> = vec4<f32>((vec3<f32>(0) - fi_4), 0.0f);
                var ap_ang_0 : vec4<f32> = vec4<f32>((vec3<f32>(0) - mi_5), 0.0f);
                hold_0(fixed_mask_0(c_41), &(ap_lin_0), &(ap_ang_0), scratch_0[sv_0(c_41, u32(8))], scratch_0[sv_0(c_41, u32(9))]);
                scratch_0[sv_0(c_41, u32(10))] = ap_lin_0;
                scratch_0[sv_0(c_41, u32(11))] = ap_ang_0;
                c_41 = c_41 + u32(256);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
            var pap_0 : f32 = island_dot_0(tid_26, c0_1, c1_1, u32(8), u32(10));
            var _S881 : u32 = cg_total_1 + u32(1);
            if(pap_0 <= 0.0f)
            {
                cg_total_0 = _S881;
                break;
            }
            var _S882 : f32 = rz_0 / pap_0;
            var c_46 : u32 = _S861;
            loop
            {
                if(c_46 < c1_1)
                {
                }
                else
                {
                    break;
                }
                var _S883 : u32 = sv_0(c_46, u32(2));
                var _S884 : vec3<f32> = vec3<f32>(_S882);
                scratch_0[_S883] = vec4<f32>(scratch_0[_S883].xyz + _S884 * scratch_0[sv_0(c_46, u32(8))].xyz, 0.0f);
                var _S885 : u32 = sv_0(c_46, u32(3));
                scratch_0[_S885] = vec4<f32>(scratch_0[_S885].xyz + _S884 * scratch_0[sv_0(c_46, u32(9))].xyz, 0.0f);
                var _S886 : u32 = sv_0(c_46, u32(4));
                scratch_0[_S886] = vec4<f32>(scratch_0[_S886].xyz - _S884 * scratch_0[sv_0(c_46, u32(10))].xyz, 0.0f);
                var _S887 : u32 = sv_0(c_46, u32(5));
                scratch_0[_S887] = vec4<f32>(scratch_0[_S887].xyz - _S884 * scratch_0[sv_0(c_46, u32(11))].xyz, 0.0f);
                c_46 = c_46 + u32(256);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
            var _S888 : f32 = island_dot_0(tid_26, c0_1, c1_1, u32(4), u32(4));
            if((sqrt(_S888)) <= (0.00009999999747379f * _S879))
            {
                cg_total_0 = _S881;
                break;
            }
            var c_47 : u32 = _S861;
            loop
            {
                if(c_47 < c1_1)
                {
                }
                else
                {
                    break;
                }
                precondition_0(c_47);
                c_47 = c_47 + u32(256);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
            if(free_0)
            {
                project_displacement_slot_0(tid_26, &(_S857), u32(6));
            }
            var rz_new_0 : f32 = island_dot_0(tid_26, c0_1, c1_1, u32(4), u32(6));
            var _S889 : f32 = rz_new_0 / rz_0;
            var c_48 : u32 = _S861;
            loop
            {
                if(c_48 < c1_1)
                {
                }
                else
                {
                    break;
                }
                var _S890 : u32 = sv_0(c_48, u32(8));
                var _S891 : vec3<f32> = vec3<f32>(_S889);
                scratch_0[_S890] = vec4<f32>(scratch_0[sv_0(c_48, u32(6))].xyz + _S891 * scratch_0[_S890].xyz, 0.0f);
                var _S892 : u32 = sv_0(c_48, u32(9));
                scratch_0[_S892] = vec4<f32>(scratch_0[sv_0(c_48, u32(7))].xyz + _S891 * scratch_0[_S892].xyz, 0.0f);
                c_48 = c_48 + u32(256);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
            var _S893 : u32 = k_25 + u32(1);
            rz_0 = rz_new_0;
            k_25 = _S893;
            cg_total_1 = _S881;
        }
        c_41 = _S861;
        loop
        {
            if(c_41 < c1_1)
            {
            }
            else
            {
                break;
            }
            var _S894 : u32 = u32(4) * c_41;
            var u_4 : vec3<f32> = state_0[_S894].xyz;
            var _S895 : u32 = sv_0(c_41, u32(21));
            var u_lo_0 : vec3<f32> = scratch_0[_S895].xyz;
            var _S896 : u32 = _S894 + u32(1);
            var th_6 : vec3<f32> = state_0[_S896].xyz;
            var _S897 : u32 = sv_0(c_41, u32(22));
            var th_lo_0 : vec3<f32> = scratch_0[_S897].xyz;
            comp_add_0(&(u_4), &(u_lo_0), scratch_0[sv_0(c_41, u32(2))].xyz);
            comp_add_0(&(th_6), &(th_lo_0), scratch_0[sv_0(c_41, u32(3))].xyz);
            state_0[_S894] = vec4<f32>(u_4, state_0[_S894].w);
            state_0[_S896] = vec4<f32>(th_6, state_0[_S896].w);
            scratch_0[_S895] = vec4<f32>(u_lo_0, 0.0f);
            scratch_0[_S897] = vec4<f32>(th_lo_0, 0.0f);
            c_41 = c_41 + u32(256);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        var _S898 : u32 = newton_0 + u32(1);
        previous_1 = residual_1;
        residual_0 = residual_1;
        newton_0 = _S898;
    }
    i_21 = b0_0 + tid_26;
    loop
    {
        if(i_21 < _S860)
        {
        }
        else
        {
            break;
        }
        var resp_4 : JointResponse_0 = static_response_0(i_21);
        var _S899 : JointState_0 = JointState_0( bond_dyn_0[i_21].js_0.damage_0, bond_dyn_0[i_21].js_0.crush_1, bond_dyn_0[i_21].js_0.kappa_0, bond_dyn_0[i_21].js_0.kappa_c_0, bond_dyn_0[i_21].js_0.ductility_0, bond_dyn_0[i_21].js_0.ductility_c_0, bond_dyn_0[i_21].js_0.life_0, bond_dyn_0[i_21].js_0.plastic_x_0, bond_dyn_0[i_21].js_0.plastic_y_0, bond_dyn_0[i_21].js_0.plastic_t_0, bond_dyn_0[i_21].js_0.rebar_plastic_0, bond_dyn_0[i_21].js_0.rebar_slip0_0, bond_dyn_0[i_21].js_0.rebar_slip1_0, bond_dyn_0[i_21].js_0.rebar_work_0, bond_dyn_0[i_21].js_0.rebar_broken_0, bond_dyn_0[i_21].js_0.strain_rate_0, bond_dyn_0[i_21].js_0.governing_stress_0, bond_dyn_0[i_21].js_0.dissipated_0, bond_dyn_0[i_21].js_0.utilization_0, bond_dyn_0[i_21].js_0.mode_0 );
        var bd_1 : BondDyn_0;
        bd_1.js_0 = _S899;
        bd_1.force_lin_0 = bond_dyn_0[i_21].force_lin_0;
        bd_1.force_ang_0 = bond_dyn_0[i_21].force_ang_0;
        bd_1.sums_0 = bond_dyn_0[i_21].sums_0;
        bd_1.comps_0 = bond_dyn_0[i_21].comps_0;
        bd_1.events_0 = bond_dyn_0[i_21].events_0;
        bd_1.force_lin_0 = vec4<f32>(resp_4.force_lin_1, resp_4.stored_0);
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
    c_42 = _S861;
    loop
    {
        if(c_42 < c1_1)
        {
        }
        else
        {
            break;
        }
        var fi_5 : vec3<f32>;
        var mi_6 : vec3<f32>;
        gather_loads_0(c_42, &(fi_5), &(mi_6));
        var reaction_1 : vec3<f32>;
        if((fixed_mask_0(c_42)) != u32(0))
        {
            reaction_1 = (vec3<f32>(0) - (scratch_0[sv_0(c_42, u32(0))].xyz + fi_5));
        }
        else
        {
            var _S900 : u32 = u32(4) * c_42;
            reaction_1 = vec3<f32>(state_0[_S900 + u32(1)].w, state_0[_S900 + u32(2)].w, state_0[_S900 + u32(3)].w);
        }
        var _S901 : u32 = u32(4) * c_42;
        var _S902 : u32 = _S901 + u32(1);
        state_0[_S902] = vec4<f32>(state_0[_S902].xyz, reaction_1.x);
        state_0[_S901 + u32(2)] = vec4<f32>(0.0f, 0.0f, 0.0f, reaction_1.y);
        state_0[_S901 + u32(3)] = vec4<f32>(0.0f, 0.0f, 0.0f, reaction_1.z);
        c_42 = c_42 + u32(256);
    }
    if(tid_26 == u32(0))
    {
        var _S903 : f32 = (bitcast<f32>((newton_0)));
        var _S904 : f32 = (bitcast<f32>((cg_total_0)));
        if(converged_4)
        {
            previous_1 = 1.0f;
        }
        else
        {
            previous_1 = 0.0f;
        }
        scratch_0[statics_result_slot_0(_S856)] = vec4<f32>(residual_0, _S903, _S904, previous_1);
        islands_0[_S856].info_0[i32(2)] = (_S859 & (u32(4294967287)));
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn settled_fatigue(@builtin(workgroup_id) group_6 : vec3<u32>, @builtin(local_invocation_id) thread_6 : vec3<u32>)
{
    var tid_27 : u32 = thread_6.x;
    var _S905 : u32 = group_6.x;
    var isl_34 : Island_std430_0 = islands_0[_S905];
    var _S906 : bool;
    if((((islands_0[_S905].info_0.x) & (u32(16)))) == u32(0))
    {
        _S906 = true;
    }
    else
    {
        _S906 = (isl_34.range_0.w) == (isl_34.range_0.z);
    }
    if(_S906)
    {
        return;
    }
    seen_clear_0(tid_27);
    if(tid_27 == u32(0))
    {
        g_halt_0 = u32(0);
        g_run_0 = u32(0);
    }
    workgroupBarrier();
    var _S907 : u32 = isl_34.info_0.w;
    var i_24 : u32 = isl_34.range_0.z + tid_27;
    loop
    {
        if(i_24 < (isl_34.range_0.w))
        {
        }
        else
        {
            break;
        }
        var _S908 : JointState_0 = JointState_0( bond_dyn_0[i_24].js_0.damage_0, bond_dyn_0[i_24].js_0.crush_1, bond_dyn_0[i_24].js_0.kappa_0, bond_dyn_0[i_24].js_0.kappa_c_0, bond_dyn_0[i_24].js_0.ductility_0, bond_dyn_0[i_24].js_0.ductility_c_0, bond_dyn_0[i_24].js_0.life_0, bond_dyn_0[i_24].js_0.plastic_x_0, bond_dyn_0[i_24].js_0.plastic_y_0, bond_dyn_0[i_24].js_0.plastic_t_0, bond_dyn_0[i_24].js_0.rebar_plastic_0, bond_dyn_0[i_24].js_0.rebar_slip0_0, bond_dyn_0[i_24].js_0.rebar_slip1_0, bond_dyn_0[i_24].js_0.rebar_work_0, bond_dyn_0[i_24].js_0.rebar_broken_0, bond_dyn_0[i_24].js_0.strain_rate_0, bond_dyn_0[i_24].js_0.governing_stress_0, bond_dyn_0[i_24].js_0.dissipated_0, bond_dyn_0[i_24].js_0.utilization_0, bond_dyn_0[i_24].js_0.mode_0 );
        var bd_2 : BondDyn_0;
        bd_2.js_0 = _S908;
        bd_2.force_lin_0 = bond_dyn_0[i_24].force_lin_0;
        bd_2.force_ang_0 = bond_dyn_0[i_24].force_ang_0;
        bd_2.sums_0 = bond_dyn_0[i_24].sums_0;
        bd_2.comps_0 = bond_dyn_0[i_24].comps_0;
        bd_2.events_0 = bond_dyn_0[i_24].events_0;
        var _S909 : JointBond_std430_0 = bonds_0[i_24].law_0;
        var _S910 : u32 = u32(4) * _S909.ids_0.y;
        var _S911 : u32 = u32(4) * _S909.ids_0.z;
        var d_lin_14 : vec3<f32>;
        var d_ang_12 : vec3<f32>;
        bond_kinematics_0(i_24, state_0[_S910].xyz, state_0[_S910 + u32(1)].xyz, state_0[_S911].xyz, state_0[_S911 + u32(1)].xyz, &(d_lin_14), &(d_ang_12));
        var _S912 : JointMaterial_std140_0 = materials_0.m_0.data_0[_S909.ids_0.x];
        var previous_2 : JointState_0 = bd_2.js_0;
        var _S913 : JointResponse_0 = joint_evaluate_2(&(_S912), &(_S909), bd_2.js_0, d_lin_14, d_ang_12, params_0.dt_0, (params_0.fracture_0) != u32(0));
        if((_S913.state_1.damage_0) > (bd_2.js_0.damage_0 + 9.99999971718068537e-10f))
        {
            _S906 = true;
        }
        else
        {
            _S906 = (_S913.state_1.crush_1) > (previous_2.crush_1 + 9.99999971718068537e-10f);
        }
        var flags_5 : u32;
        if(_S906)
        {
            flags_5 = u32(16);
        }
        else
        {
            flags_5 = u32(0);
        }
        var _S914 : f32 = bd_2.sums_0[i32(0)];
        var _S915 : f32 = bd_2.comps_0[i32(0)];
        comp_add1_0(&(_S914), &(_S915), _S913.dissipated_1);
        bd_2.sums_0[i32(0)] = _S914;
        bd_2.comps_0[i32(0)] = _S915;
        var _S916 : f32 = bd_2.sums_0[i32(1)];
        var _S917 : f32 = bd_2.comps_0[i32(1)];
        comp_add1_0(&(_S916), &(_S917), _S913.overshoot_0);
        bd_2.sums_0[i32(1)] = _S916;
        bd_2.comps_0[i32(1)] = _S917;
        bd_2.force_lin_0 = vec4<f32>(_S913.force_lin_1, _S913.stored_0);
        bd_2.force_ang_0 = vec4<f32>(_S913.force_ang_1, max(bd_2.force_ang_0.w, _S913.state_1.utilization_0));
        var _S918 : bool;
        if(!is_damaged_1(previous_2))
        {
            _S918 = is_damaged_1(_S913.state_1);
        }
        else
        {
            _S918 = false;
        }
        var _S919 : bool;
        if(_S918)
        {
            _S919 = (bd_2.events_0.x) == u32(0);
        }
        else
        {
            _S919 = false;
        }
        if(_S919)
        {
            bd_2.events_0[i32(0)] = _S907;
            bd_2.events_0[i32(3)] = _S913.state_1.mode_0;
            g_seen_0[u32(3)] = u32(1);
        }
        var _S920 : bool;
        if((bd_2.events_0.y) == u32(0))
        {
            _S920 = (_S913.state_1.life_0) != (previous_2.life_0);
        }
        else
        {
            _S920 = false;
        }
        var _S921 : bool;
        if(_S920)
        {
            _S921 = (_S913.fatigue_before_0) > 0.99000000953674316f;
        }
        else
        {
            _S921 = false;
        }
        var _S922 : bool;
        if(_S921)
        {
            var _S923 : f32 = fatigue_factor_2(&(_S912), _S913.state_1.life_0);
            _S922 = _S923 <= 0.99000000953674316f;
        }
        else
        {
            _S922 = false;
        }
        if(_S922)
        {
            bd_2.events_0[i32(1)] = _S907;
            g_seen_0[u32(4)] = u32(1);
        }
        var flags_6 : u32;
        if(_S913.disconnected_0)
        {
            bd_2.events_0[i32(2)] = _S907;
            flags_6 = (flags_5 | (u32(32)));
        }
        else
        {
            flags_6 = flags_5;
        }
        bd_2.js_0 = _S913.state_1;
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
        write_bond_loads_0(i_24, i_24, _S913.force_lin_1, _S913.force_ang_1, max(_S913.measures_0.tension_0, _S913.measures_0.compression_0));
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
    if(tid_27 != u32(0))
    {
        return;
    }
    if(((g_halt_0 | (g_run_0))) != u32(0))
    {
        var _S924 : u32 = isl_34.info_0.z;
        if(g_halt_0 != u32(0))
        {
            i_24 = u32(16);
        }
        else
        {
            i_24 = u32(0);
        }
        var _S925 : u32 = (_S924 | (i_24));
        if(g_run_0 != u32(0))
        {
            i_24 = u32(32);
        }
        else
        {
            i_24 = u32(0);
        }
        islands_0[_S905].info_0[i32(2)] = (_S925 | (i_24));
    }
    var _S926 : u32 = ((((isl_34.coverage_0) | (u32(16384)))) | ((seen_bits_0())));
    if(g_run_0 != u32(0))
    {
        i_24 = u32(128);
    }
    else
    {
        i_24 = u32(0);
    }
    islands_0[_S905].coverage_0 = (_S926 | (i_24));
    return;
}

fn impactor_integrate_at_0( ii_3 : u32,  tid_28 : u32)
{
    var p_23 : vec3<f32>;
    if(ii_3 >= (params_0.impactor_count_0))
    {
        return;
    }
    var imp_6 : Impactor_0;
    imp_6.position_1 = impactors_0[ii_3].position_1;
    imp_6.position_err_1 = impactors_0[ii_3].position_err_1;
    imp_6.velocity_1 = impactors_0[ii_3].velocity_1;
    imp_6.velocity_err_1 = impactors_0[ii_3].velocity_err_1;
    imp_6.angular_velocity_1 = impactors_0[ii_3].angular_velocity_1;
    imp_6.rotation_1 = impactors_0[ii_3].rotation_1;
    imp_6.inertia0_2 = impactors_0[ii_3].inertia0_2;
    imp_6.inertia1_2 = impactors_0[ii_3].inertia1_2;
    imp_6.inertia2_2 = impactors_0[ii_3].inertia2_2;
    imp_6.inv0_2 = impactors_0[ii_3].inv0_2;
    imp_6.inv1_2 = impactors_0[ii_3].inv1_2;
    imp_6.inv2_2 = impactors_0[ii_3].inv2_2;
    imp_6.shape_0 = impactors_0[ii_3].shape_0;
    imp_6.half_1 = impactors_0[ii_3].half_1;
    imp_6.mat_0 = impactors_0[ii_3].mat_0;
    imp_6.crush_0 = impactors_0[ii_3].crush_0;
    imp_6.geom_0 = impactors_0[ii_3].geom_0;
    imp_6.rotation_err_1 = impactors_0[ii_3].rotation_err_1;
    imp_6.ledger_0 = impactors_0[ii_3].ledger_0;
    imp_6.cand_0 = impactors_0[ii_3].cand_0;
    imp_6.momentum_1 = impactors_0[ii_3].momentum_1;
    imp_6.momentum_err_1 = impactors_0[ii_3].momentum_err_1;
    var _S927 : vec4<u32> = islands_0[params_0.halt_index_0].info_0;
    var _S928 : bool;
    if((((islands_0[params_0.halt_index_0].info_0.z) & (u32(1)))) != u32(0))
    {
        _S928 = true;
    }
    else
    {
        var _S929 : u32 = _S927.y;
        if(_S929 != u32(0))
        {
            _S928 = (imp_6.cand_0.w) >= _S929;
        }
        else
        {
            _S928 = false;
        }
    }
    if(_S928)
    {
        return;
    }
    if((imp_6.cand_0.z) != u32(0))
    {
        if(tid_28 == u32(0))
        {
            impactors_0[ii_3].cand_0[i32(3)] = imp_6.cand_0.w + u32(1);
        }
        return;
    }
    var _S930 : vec4<f32> = vec4<f32>(0.0f);
    var rf_1 : vec4<f32> = _S930;
    var rt_0 : vec4<f32> = _S930;
    var k_26 : u32 = imp_6.cand_0.x + tid_28;
    loop
    {
        if(k_26 < (imp_6.cand_0.y))
        {
        }
        else
        {
            break;
        }
        var _S931 : u32 = u32(3) * k_26;
        rf_1 = rf_1 + scratch_0[params_0.cand_base_0 + _S931 + u32(1)];
        rt_0 = rt_0 + scratch_0[params_0.cand_base_0 + _S931 + u32(2)];
        k_26 = k_26 + u32(256);
    }
    group_sum2_3(tid_28, &(rf_1), &(rt_0));
    if(tid_28 != u32(0))
    {
        return;
    }
    var dt_18 : f32 = params_0.dt_0;
    var _S932 : vec3<f32> = vec3<f32>(0.0f);
    var load_f_0 : vec3<f32>;
    var load_t_0 : vec3<f32>;
    if((params_0.has_ground_0) != u32(0))
    {
        var ib_0 : Box_0 = impactor_box_0(imp_6, _S932, imp_6.half_1.xyz);
        var _S933 : vec3<f32> = imp_6.velocity_1.xyz + imp_6.velocity_err_1.xyz;
        const _S934 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 1.0f);
        var kc_2 : f32 = contact_stiffness_0(params_0.ground_modulus_0, ib_0, imp_6.mat_0.x, ib_0, _S934);
        var _S935 : f32 = imp_6.position_1.z - params_0.ground_hi_0 + (imp_6.position_err_1.z - params_0.ground_lo_0);
        var total_points_0 : u32;
        if((imp_6.shape_0.x) == 0.0f)
        {
            total_points_0 = u32(1);
        }
        else
        {
            total_points_0 = u32(14);
        }
        var _S936 : f32 = kc_2 / f32(min(total_points_0, u32(5)));
        var s_17 : u32 = u32(0);
        var below_0 : u32 = u32(0);
        loop
        {
            if(s_17 < total_points_0)
            {
            }
            else
            {
                break;
            }
            if((imp_6.shape_0.x) == 0.0f)
            {
                p_23 = vec3<f32>(0.0f, 0.0f, - imp_6.shape_0.y);
            }
            else
            {
                p_23 = sample_point_0(ib_0, s_17);
            }
            if((_S935 + p_23.z) < 0.0f)
            {
                below_0 = below_0 + u32(1);
            }
            s_17 = s_17 + u32(1);
        }
        if(below_0 > u32(0))
        {
            sentinel_mark_0(u32(8));
        }
        s_17 = u32(0);
        load_f_0 = _S932;
        load_t_0 = _S932;
        loop
        {
            if(s_17 < total_points_0)
            {
            }
            else
            {
                break;
            }
            if((imp_6.shape_0.x) == 0.0f)
            {
                p_23 = vec3<f32>(0.0f, 0.0f, - imp_6.shape_0.y);
            }
            else
            {
                p_23 = sample_point_0(ib_0, s_17);
            }
            var depth_6 : f32 = - (_S935 + p_23.z);
            if(depth_6 <= 0.0f)
            {
                s_17 = s_17 + u32(1);
                continue;
            }
            var stored_15 : f32;
            var diss_7 : f32;
            var f_21 : vec3<f32> = penalty_force_0(_S936, imp_6.mat_0.z, params_0.ground_friction_0, depth_6, _S934, _S933 + cross(imp_6.angular_velocity_1.xyz, p_23), dt_18, below_0, &(stored_15), &(diss_7));
            var load_f_1 : vec3<f32> = load_f_0 + f_21;
            var load_t_1 : vec3<f32> = load_t_0 + cross(p_23, f_21);
            var _S937 : f32 = imp_6.ledger_0[i32(0)];
            var _S938 : f32 = imp_6.ledger_0[i32(1)];
            comp_add1_3(&(_S937), &(_S938), diss_7);
            imp_6.ledger_0[i32(0)] = _S937;
            imp_6.ledger_0[i32(1)] = _S938;
            load_f_0 = load_f_1;
            load_t_0 = load_t_1;
            s_17 = s_17 + u32(1);
        }
    }
    else
    {
        load_f_0 = _S932;
        load_t_0 = _S932;
    }
    var load_f_2 : vec3<f32> = rf_1.xyz + load_f_0;
    var load_t_2 : vec3<f32> = rt_0.xyz + load_t_0;
    var m_9 : f32 = imp_6.mat_0.z;
    var vel_1 : vec3<f32> = imp_6.velocity_1.xyz;
    var vel_err_1 : vec3<f32> = imp_6.velocity_err_1.xyz;
    var _S939 : vec3<f32> = vec3<f32>(dt_18);
    comp_add_1(&(vel_1), &(vel_err_1), (load_f_2 / vec3<f32>(m_9) + params_0.gravity_0.xyz) * _S939);
    var _S940 : Quat_0 = quat_of_0(imp_6.rotation_1);
    var q_14 : Quat_0 = _S940;
    var q_err_0 : vec4<f32> = imp_6.rotation_err_1;
    var l_hi_0 : vec3<f32> = imp_6.momentum_1.xyz;
    var l_err_1 : vec3<f32> = imp_6.momentum_err_1.xyz;
    comp_add_1(&(l_hi_0), &(l_err_1), load_t_2 * _S939);
    var l_9 : vec3<f32> = l_hi_0 + l_err_1;
    var w_mid_2 : vec3<f32> = world_mul_0(_S940, imp_6.inv0_2, imp_6.inv1_2, imp_6.inv2_2, l_9);
    var pos_1 : vec3<f32> = imp_6.position_1.xyz;
    var pos_err_1 : vec3<f32> = imp_6.position_err_1.xyz;
    comp_add_1(&(pos_1), &(pos_err_1), (vel_1 + vel_err_1) * _S939);
    turn_left_1(&(q_14), &(q_err_0), w_mid_2, dt_18);
    imp_6.angular_velocity_1 = vec4<f32>(world_mul_0(q_14, imp_6.inv0_2, imp_6.inv1_2, imp_6.inv2_2, l_9), 0.0f);
    imp_6.rotation_1 = quat_vec_0(q_14);
    imp_6.rotation_err_1 = q_err_0;
    imp_6.momentum_1 = vec4<f32>(l_hi_0, 0.0f);
    imp_6.momentum_err_1 = vec4<f32>(l_err_1, 0.0f);
    imp_6.position_1 = vec4<f32>(pos_1, 0.0f);
    imp_6.position_err_1 = vec4<f32>(pos_err_1, 0.0f);
    imp_6.velocity_1 = vec4<f32>(vel_1, 0.0f);
    imp_6.velocity_err_1 = vec4<f32>(vel_err_1, 0.0f);
    imp_6.cand_0[i32(3)] = imp_6.cand_0[i32(3)] + u32(1);
    imp_6.geom_0 = vec4<f32>(1.0f, imp_6.crush_0.w, 0.0f, 0.0f);
    impactors_0[ii_3].position_1 = imp_6.position_1;
    impactors_0[ii_3].position_err_1 = imp_6.position_err_1;
    impactors_0[ii_3].velocity_1 = imp_6.velocity_1;
    impactors_0[ii_3].velocity_err_1 = imp_6.velocity_err_1;
    impactors_0[ii_3].angular_velocity_1 = imp_6.angular_velocity_1;
    impactors_0[ii_3].rotation_1 = imp_6.rotation_1;
    impactors_0[ii_3].inertia0_2 = imp_6.inertia0_2;
    impactors_0[ii_3].inertia1_2 = imp_6.inertia1_2;
    impactors_0[ii_3].inertia2_2 = imp_6.inertia2_2;
    impactors_0[ii_3].inv0_2 = imp_6.inv0_2;
    impactors_0[ii_3].inv1_2 = imp_6.inv1_2;
    impactors_0[ii_3].inv2_2 = imp_6.inv2_2;
    impactors_0[ii_3].shape_0 = imp_6.shape_0;
    impactors_0[ii_3].half_1 = imp_6.half_1;
    impactors_0[ii_3].mat_0 = imp_6.mat_0;
    impactors_0[ii_3].crush_0 = imp_6.crush_0;
    impactors_0[ii_3].geom_0 = imp_6.geom_0;
    impactors_0[ii_3].rotation_err_1 = imp_6.rotation_err_1;
    impactors_0[ii_3].ledger_0 = imp_6.ledger_0;
    impactors_0[ii_3].cand_0 = imp_6.cand_0;
    impactors_0[ii_3].momentum_1 = imp_6.momentum_1;
    impactors_0[ii_3].momentum_err_1 = imp_6.momentum_err_1;
    var k_27 : u32 = imp_6.cand_0.w - u32(1) - params_0.step_start_0;
    if(k_27 < (params_0.record_stride_0))
    {
        var at_8 : u32 = params_0.record_base_0 + u32(2) * (ii_3 * params_0.record_stride_0 + k_27);
        scratch_0[at_8] = vec4<f32>(vel_1 + vel_err_1, 0.0f);
        scratch_0[at_8 + u32(1)] = vec4<f32>(pos_1 + pos_err_1, 0.0f);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn impactor_integrate(@builtin(workgroup_id) group_7 : vec3<u32>, @builtin(local_invocation_id) thread_7 : vec3<u32>)
{
    impactor_integrate_at_0(group_7.x, thread_7.x);
    return;
}

fn ground_contact_0( c_49 : u32,  account_0 : bool,  f_22 : ptr<function, vec3<f32>>,  t_23 : ptr<function, vec3<f32>>)
{
    var wp_1 : WorldPoint_0 = chunk_world_0(c_49);
    var above_0 : f32 = wp_1.hi_3.z - params_0.ground_hi_0 + (wp_1.lo_3.z - params_0.ground_lo_0) + wp_1.rel_0.z;
    if((above_0 - chunks_0[c_49].half_0.w) > 0.0f)
    {
        return;
    }
    var b_55 : Box_0 = chunk_box_0(c_49, vec3<f32>(0.0f));
    const _S941 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 1.0f);
    var _S942 : f32 = contact_stiffness_0(params_0.ground_modulus_0, b_55, chunks_0[c_49].cmat_0.x, b_55, _S941);
    var _S943 : u32 = sample_count_0(b_55);
    var s_18 : u32 = u32(0);
    var n_12 : u32 = u32(0);
    loop
    {
        if(s_18 < _S943)
        {
        }
        else
        {
            break;
        }
        if((above_0 + sample_point_0(b_55, s_18).z) < 0.0f)
        {
            n_12 = n_12 + u32(1);
        }
        s_18 = s_18 + u32(1);
    }
    if(n_12 == u32(0))
    {
        return;
    }
    sentinel_mark_0(u32(4));
    var vc_1 : vec3<f32>;
    var wc_1 : vec3<f32>;
    chunk_velocity_1(c_49, &(vc_1), &(wc_1));
    var ledger_2 : vec4<f32> = scratch_0[params_0.ledger_base_0 + params_0.pair_count_0 + c_49];
    s_18 = u32(0);
    loop
    {
        if(s_18 < _S943)
        {
        }
        else
        {
            break;
        }
        var p_24 : vec3<f32> = sample_point_0(b_55, s_18);
        var _S944 : f32 = above_0 + p_24.z;
        if(!(_S944 < 0.0f))
        {
            s_18 = s_18 + u32(1);
            continue;
        }
        var stored_16 : f32;
        var diss_8 : f32;
        var g_3 : vec3<f32> = penalty_force_2(_S942 / f32(max(n_12, u32(5))), chunks_0[c_49].center_0.w, params_0.ground_friction_0, - _S944, _S941, vc_1 + cross(wc_1, p_24), params_0.dt_0, n_12, &(stored_16), &(diss_8));
        (*f_22) = (*f_22) + g_3;
        (*t_23) = (*t_23) + cross(p_24, g_3);
        var _S945 : f32 = ledger_2[i32(1)];
        var _S946 : f32 = ledger_2[i32(2)];
        comp_add1_0(&(_S945), &(_S946), diss_8);
        ledger_2[i32(1)] = _S945;
        ledger_2[i32(2)] = _S946;
        s_18 = s_18 + u32(1);
    }
    if(account_0)
    {
        scratch_0[params_0.ledger_base_0 + params_0.pair_count_0 + c_49] = ledger_2;
    }
    return;
}

fn contact_sum_at_0( g_4 : u32)
{
    var _S947 : bool;
    if(g_4 >= (params_0.seg_count_0))
    {
        _S947 = true;
    }
    else
    {
        _S947 = stopped_0();
    }
    if(_S947)
    {
        return;
    }
    var _S948 : u32 = u32(3) * g_4;
    var _S949 : u32 = index_0[params_0.seg_index_0 + _S948];
    var begin_1 : u32 = index_0[params_0.seg_index_0 + _S948 + u32(1)];
    var _S950 : u32 = index_0[params_0.seg_index_0 + _S948 + u32(2)];
    var _S951 : vec3<f32> = vec3<f32>(0.0f);
    var f_23 : vec3<f32> = _S951;
    var t_24 : vec3<f32> = _S951;
    var e_8 : u32 = begin_1;
    loop
    {
        if(e_8 < _S950)
        {
        }
        else
        {
            break;
        }
        var entry_7 : u32 = index_0[e_8];
        if(entry_7 == u32(2147483648))
        {
            ground_contact_0(_S949, true, &(f_23), &(t_24));
            e_8 = e_8 + u32(1);
            continue;
        }
        var _S952 : u32 = u32(2) * entry_7;
        f_23 = f_23 + scratch_0[params_0.slot_base_0 + _S952].xyz;
        t_24 = t_24 + scratch_0[params_0.slot_base_0 + _S952 + u32(1)].xyz;
        e_8 = e_8 + u32(1);
    }
    var _S953 : u32 = u32(2) * g_4;
    scratch_0[params_0.seg_base_0 + _S953] = vec4<f32>(f_23, 0.0f);
    scratch_0[params_0.seg_base_0 + _S953 + u32(1)] = vec4<f32>(t_24, 0.0f);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn contact_sums(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    contact_sum_at_0(id_2.x);
    return;
}

fn net_load_1( c_50 : u32,  isl_35 : ptr<function, Island_std430_0>,  rg_19 : Rigid_0,  k_28 : u32,  dt_19 : f32,  contact_5 : bool,  f_24 : ptr<function, vec3<f32>>,  t_25 : ptr<function, vec3<f32>>)
{
    var fl_1 : vec3<f32>;
    var tl_1 : vec3<f32>;
    var rep_f_2 : vec3<f32>;
    var rep_m_2 : vec3<f32>;
    chunk_loads_1(c_50, c_50, rg_19.rot_0, k_28, dt_19, contact_5, 0.0f, &(fl_1), &(tl_1), &(rep_f_2), &(rep_m_2));
    var fc_5 : vec3<f32> = fl_1 + params_0.gravity_0.xyz * vec3<f32>(chunks_0[c_50].center_0.w);
    var _S954 : vec3<f32> = chunks_0[c_50].center_0.xyz;
    var _S955 : vec3<f32> = (*isl_35).com_0.xyz;
    var r_17 : vec3<f32> = rotate_0(rg_19.rot_0, _S954 + state_0[u32(4) * c_50].xyz - _S955);
    (*f_24) = (*f_24) + fc_5;
    (*t_25) = (*t_25) + (cross(r_17, fc_5) + tl_1);
    if((chunks_0[c_50].load_range_0.y) > (chunks_0[c_50].load_range_0.x))
    {
        var fw_3 : vec3<f32> = rotate_0(rg_19.rot_0, rep_f_2);
        (*f_24) = (*f_24) + fw_3;
        (*t_25) = (*t_25) + (cross(rotate_0(rg_19.rot_0, _S954 - _S955), fw_3) + rotate_0(rg_19.rot_0, rep_m_2));
    }
    return;
}

fn chunk_update_1( c_51 : u32,  isl_36 : ptr<function, Island_std430_0>,  rg_20 : Rigid_0,  dt_20 : f32,  rml_1 : bool,  step_1 : u32,  contact_6 : bool,  work_7 : ptr<function, f32>,  work_err_3 : ptr<function, f32>)
{
    var _S956 : vec3<f32> = vec3<f32>(0.0f);
    var _S957 : u32 = index_0[c_51];
    var peak_1 : f32 = 0.0f;
    var e_9 : u32 = _S957;
    var fi_6 : vec3<f32> = _S956;
    var mi_7 : vec3<f32> = _S956;
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
        var _S958 : u32 = u32(3) * ((entry_8 >> (u32(1))));
        var fa_2 : vec4<f32> = scratch_0[_S958];
        if(((entry_8 & (u32(1)))) == u32(0))
        {
            var mi_8 : vec3<f32> = mi_7 + scratch_0[_S958 + u32(1)].xyz;
            fi_6 = fi_6 + fa_2.xyz;
            mi_7 = mi_8;
        }
        else
        {
            var mi_9 : vec3<f32> = mi_7 + scratch_0[_S958 + u32(2)].xyz;
            fi_6 = fi_6 + (vec3<f32>(0) - fa_2.xyz);
            mi_7 = mi_9;
        }
        var _S959 : f32 = max(peak_1, fa_2.w);
        var _S960 : u32 = e_9 + u32(1);
        peak_1 = _S959;
        e_9 = _S960;
    }
    var _S961 : u32 = u32(4) * c_51;
    var u_5 : vec3<f32> = state_0[_S961].xyz;
    var _S962 : u32 = _S961 + u32(1);
    var th_7 : vec3<f32> = state_0[_S962].xyz;
    var _S963 : u32 = _S961 + u32(2);
    var v_13 : vec3<f32> = state_0[_S963].xyz;
    var _S964 : u32 = _S961 + u32(3);
    var w_9 : vec3<f32> = state_0[_S964].xyz;
    var mass_1 : f32 = chunks_0[c_51].center_0.w;
    var _S965 : vec3<f32> = chunks_0[c_51].center_0.xyz;
    var _S966 : vec3<f32> = (*isl_36).com_0.xyz;
    var r_world_1 : vec3<f32> = rotate_0(rg_20.rot_0, _S965 + u_5 - _S966);
    var f_load_1 : vec3<f32>;
    var t_load_1 : vec3<f32>;
    var rep_f_3 : vec3<f32>;
    var rep_m_3 : vec3<f32>;
    chunk_loads_1(c_51, c_51, rg_20.rot_0, step_1, dt_20, contact_6, dt_20, &(f_load_1), &(t_load_1), &(rep_f_3), &(rep_m_3));
    record_chunk_load_0(c_51, f_load_1, t_load_1);
    var _S967 : vec3<f32> = vec3<f32>(mass_1);
    var f_world_2 : vec3<f32> = f_load_1 + params_0.gravity_0.xyz * _S967;
    var t_world_3 : vec3<f32> = t_load_1;
    var f_world_3 : vec3<f32>;
    var t_world_4 : vec3<f32>;
    if(rml_1)
    {
        var t_world_5 : vec3<f32> = t_world_3 - (world_mul_0(rg_20.rot_0, chunks_0[c_51].inertia0_1, chunks_0[c_51].inertia1_1, chunks_0[c_51].inertia2_1, rg_20.alpha_0) + cross(rg_20.w_1, world_mul_0(rg_20.rot_0, chunks_0[c_51].inertia0_1, chunks_0[c_51].inertia1_1, chunks_0[c_51].inertia2_1, rg_20.w_1)));
        f_world_3 = f_world_2 - (rg_20.a_0 + cross(rg_20.alpha_0, r_world_1) + cross(rg_20.w_1, cross(rg_20.w_1, r_world_1))) * _S967;
        t_world_4 = t_world_5;
    }
    else
    {
        f_world_3 = f_world_2;
        t_world_4 = t_world_3;
    }
    var f_ext_2 : vec3<f32> = inverse_rotate_0(rg_20.rot_0, f_world_3);
    var m_ext_3 : vec3<f32> = inverse_rotate_0(rg_20.rot_0, t_world_4);
    var f_ext_3 : vec3<f32>;
    var m_ext_4 : vec3<f32>;
    if(rml_1)
    {
        var wb_2 : vec3<f32> = inverse_rotate_0(rg_20.rot_0, rg_20.w_1);
        var i_w_1 : vec3<f32> = rows_mul_0(chunks_0[c_51].inertia0_1, chunks_0[c_51].inertia1_1, chunks_0[c_51].inertia2_1, w_9);
        var m_ext_5 : vec3<f32> = m_ext_3 - (cross(wb_2, i_w_1) + cross(w_9, rows_mul_0(chunks_0[c_51].inertia0_1, chunks_0[c_51].inertia1_1, chunks_0[c_51].inertia2_1, wb_2)) + cross(w_9, i_w_1));
        f_ext_3 = f_ext_2 - cross(wb_2, v_13) * vec3<f32>((2.0f * mass_1));
        m_ext_4 = m_ext_5;
    }
    else
    {
        f_ext_3 = f_ext_2;
        m_ext_4 = m_ext_3;
    }
    var f_25 : vec3<f32> = f_ext_3 + rep_f_3 + fi_6;
    var m_10 : vec3<f32> = m_ext_4 + rep_m_3 + mi_7;
    var support_2 : u32 = chunks_0[c_51].info_1.x;
    var _S968 : vec3<f32> = vec3<f32>(state_0[_S962].w, state_0[_S963].w, state_0[_S964].w);
    var reaction_2 : vec3<f32>;
    var u_6 : vec3<f32>;
    var th_8 : vec3<f32>;
    var v_14 : vec3<f32>;
    var w_10 : vec3<f32>;
    if(support_2 == u32(1))
    {
        reaction_2 = (vec3<f32>(0) - f_25);
        u_6 = u_5;
        th_8 = th_7;
        v_14 = _S956;
        w_10 = _S956;
    }
    else
    {
        var _S969 : vec4<f32> = chunks_0[c_51].scale_0;
        var w_11 : vec3<f32> = w_9 + rows_mul_0(chunks_0[c_51].inv0_1, chunks_0[c_51].inv1_1, chunks_0[c_51].inv2_1, m_10) * vec3<f32>((dt_20 * chunks_0[c_51].scale_0.z));
        var _S970 : vec3<f32> = vec3<f32>(dt_20);
        var th_9 : vec3<f32> = th_7 + w_11 * _S970;
        if(support_2 == u32(2))
        {
            reaction_2 = (vec3<f32>(0) - f_25);
            u_6 = u_5;
            th_8 = _S956;
        }
        else
        {
            var v_15 : vec3<f32> = v_13 + f_25 * vec3<f32>((dt_20 * _S969.y));
            var u_7 : vec3<f32> = u_5 + v_15 * _S970;
            reaction_2 = _S968;
            u_6 = u_7;
            th_8 = v_15;
        }
        var _S971 : vec3<f32> = th_8;
        th_8 = th_9;
        v_14 = _S971;
        w_10 = w_11;
    }
    state_0[_S961] = vec4<f32>(u_6, peak_1);
    state_0[_S962] = vec4<f32>(th_8, reaction_2.x);
    state_0[_S963] = vec4<f32>(v_14, reaction_2.y);
    state_0[_S964] = vec4<f32>(w_10, reaction_2.z);
    comp_add1_0(&((*work_7)), &((*work_err_3)), (dot(f_load_1, rg_20.vel_0 + rg_20.vel_err_0 + cross(rg_20.w_1, rotate_0(rg_20.rot_0, _S965 + u_6 - _S966)) + rotate_0(rg_20.rot_0, v_14)) + dot(t_load_1, rg_20.w_1 + rotate_0(rg_20.rot_0, w_10))) * dt_20);
    return;
}

fn drift_moments_1( c_52 : u32,  tu_5 : ptr<function, vec3<f32>>,  pv_5 : ptr<function, vec3<f32>>)
{
    var _S972 : u32 = u32(4) * c_52;
    var _S973 : vec3<f32> = vec3<f32>((chunks_0[c_52].center_0.w * chunks_0[c_52].scale_0.x));
    (*tu_5) = (*tu_5) + state_0[_S972].xyz * _S973;
    (*pv_5) = (*pv_5) + state_0[_S972 + u32(2)].xyz * _S973;
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn island_frame(@builtin(workgroup_id) group_8 : vec3<u32>, @builtin(local_invocation_id) thread_8 : vec3<u32>)
{
    island_frame_at_1(group_8.x, thread_8.x);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn contact_islands(@builtin(workgroup_id) group_9 : vec3<u32>, @builtin(local_invocation_id) thread_9 : vec3<u32>)
{
    island_frame_at_0(group_9.x, thread_9.x);
    return;
}

fn wide_chunks_at_0( gx_0 : u32,  tid_29 : u32)
{
    var wg_6 : WideGroup_0 = wide_group_0(params_0.wide_chunk_table_0, gx_0);
    var _S974 : Island_std430_0 = islands_0[wg_6.island_2];
    var _S975 : bool = wide_enter_1(tid_29, wg_6.island_2);
    if(!_S975)
    {
        return;
    }
    var _S976 : u32 = _S974.info_0.x;
    var anchored_0 : bool = ((_S976 & (u32(1)))) != u32(0);
    if(((_S976 & (u32(16)))) != u32(0))
    {
        if(tid_29 == u32(0))
        {
            scratch_0[params_0.wide_base_0 + u32(8) * gx_0 + u32(6)] = vec4<f32>(0.0f);
        }
        return;
    }
    var _S977 : Rigid_0 = wide_rigid_frame_0(tid_29, &(_S974), wg_6);
    var work_8 : f32 = 0.0f;
    var work_err_4 : f32 = 0.0f;
    var _S978 : vec3<f32> = vec3<f32>(0.0f);
    var tu_6 : vec3<f32> = _S978;
    var pv_6 : vec3<f32> = _S978;
    var c_53 : u32 = wg_6.begin_0 + tid_29;
    if(c_53 < (wg_6.end_0))
    {
        var _S979 : bool = (params_0.rigid_motion_loads_0) != u32(0);
        var _S980 : u32 = wide_step_0(&(_S974));
        chunk_update_1(c_53, &(_S974), _S977, params_0.dt_0, _S979, _S980, ((_S976 & (u32(4)))) != u32(0), &(work_8), &(work_err_4));
        if(!anchored_0)
        {
            drift_moments_1(c_53, &(tu_6), &(pv_6));
        }
    }
    var wsum_2 : vec3<f32> = vec3<f32>(work_8, work_err_4, 0.0f);
    var unused_5 : vec3<f32> = _S978;
    group_sum3_1(tid_29, &(wsum_2), &(unused_5));
    var _S981 : bool = !anchored_0;
    if(_S981)
    {
        group_sum3_1(tid_29, &(tu_6), &(pv_6));
    }
    if(tid_29 == u32(0))
    {
        var _S982 : u32 = u32(8) * gx_0;
        scratch_0[params_0.wide_base_0 + _S982 + u32(6)] = vec4<f32>(wsum_2, 0.0f);
        if(_S981)
        {
            wide_store_0(gx_0, u32(2), tu_6, pv_6);
        }
        scratch_0[params_0.wide_base_0 + _S982 + u32(7)] = vec4<f32>(f32(g_seen_0[u32(0)]), f32(g_seen_0[u32(1)]), f32(g_seen_0[u32(2)]), 0.0f);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn wide_chunks(@builtin(workgroup_id) group_10 : vec3<u32>, @builtin(local_invocation_id) thread_10 : vec3<u32>)
{
    wide_chunks_at_0(group_10.x, thread_10.x);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn sums_and_impactors(@builtin(workgroup_id) group_11 : vec3<u32>, @builtin(local_invocation_id) thread_11 : vec3<u32>)
{
    var sums_groups_0 : u32 = (params_0.seg_count_0 + u32(255)) / u32(256);
    var _S983 : u32 = group_11.x;
    if(_S983 < sums_groups_0)
    {
        contact_sum_at_0(_S983 * u32(256) + thread_11.x);
    }
    else
    {
        impactor_integrate_at_0(_S983 - sums_groups_0, thread_11.x);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn islands_and_wide_chunks(@builtin(workgroup_id) group_12 : vec3<u32>, @builtin(local_invocation_id) thread_12 : vec3<u32>)
{
    var _S984 : u32 = group_12.x;
    if(_S984 < (params_0.halt_index_0))
    {
        island_frame_at_1(_S984, thread_12.x);
    }
    else
    {
        wide_chunks_at_0(_S984 - params_0.halt_index_0, thread_12.x);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn contact_islands_and_wide_chunks(@builtin(workgroup_id) group_13 : vec3<u32>, @builtin(local_invocation_id) thread_13 : vec3<u32>)
{
    var _S985 : u32 = group_13.x;
    if(_S985 < (params_0.halt_index_0))
    {
        island_frame_at_0(_S985, thread_13.x);
    }
    else
    {
        wide_chunks_at_0(_S985 - params_0.halt_index_0, thread_13.x);
    }
    return;
}

fn wide_bonds_at_0( gx_1 : u32,  tid_30 : u32)
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
    var _S986 : WideGroup_0 = wg_7;
    var _S987 : Island_std430_0 = islands_0[wg_7.island_2];
    var _S988 : bool = wide_enter_1(tid_30, wg_7.island_2);
    if(!_S988)
    {
        return;
    }
    var _S989 : u32 = wide_step_1(&(_S987));
    if(bond_group_0)
    {
        var _S990 : vec4<u32> = _S987.info_0;
        if((((_S987.info_0.x) & (u32(16)))) != u32(0))
        {
            return;
        }
        var i_25 : u32 = wg_7.begin_0 + tid_30;
        var _S991 : bool;
        if(i_25 < (wg_7.end_0))
        {
            var _S992 : bool = bond_update_0(i_25, params_0.dt_0, (params_0.fracture_0) != u32(0), _S990.w + u32(1), true);
            _S991 = _S992;
        }
        else
        {
            _S991 = false;
        }
        if(_S991)
        {
            islands_0[_S986.island_2].info_0[i32(2)] = ((_S990.z) | (u32(2)));
        }
        return;
    }
    var _S993 : u32 = _S987.info_0.x;
    if(((_S993 & (u32(1)))) != u32(0))
    {
        return;
    }
    var _S994 : vec3<f32> = vec3<f32>(0.0f);
    var f_26 : vec3<f32> = _S994;
    var t_26 : vec3<f32> = _S994;
    var c_54 : u32 = wg_7.begin_0 + tid_30;
    if(c_54 < (wg_7.end_0))
    {
        var _S995 : Rigid_0 = rigid_of_2(&(_S987));
        net_load_1(c_54, &(_S987), _S995, _S989, params_0.dt_0, ((_S993 & (u32(4)))) != u32(0), &(f_26), &(t_26));
    }
    group_sum3_2(tid_30, &(f_26), &(t_26));
    if(tid_30 == u32(0))
    {
        wide_store_0(gx_1 - params_0.wide_bond_groups_0, u32(0), f_26, t_26);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn wide_bonds(@builtin(workgroup_id) group_14 : vec3<u32>, @builtin(local_invocation_id) thread_14 : vec3<u32>)
{
    wide_bonds_at_0(group_14.x, thread_14.x);
    return;
}

fn bond_group_at_0( table_1 : u32,  gx_2 : u32,  tid_31 : u32)
{
    var wg_8 : WideGroup_0 = wide_group_0(table_1, gx_2);
    var isl_37 : Island_std430_0 = islands_0[wg_8.island_2];
    var _S996 : bool = wide_enter_1(tid_31, wg_8.island_2);
    var _S997 : bool;
    if(!_S996)
    {
        _S997 = true;
    }
    else
    {
        _S997 = (((isl_37.info_0.x) & (u32(16)))) != u32(0);
    }
    if(_S997)
    {
        return;
    }
    var i_26 : u32 = wg_8.begin_0 + tid_31;
    if(i_26 < (wg_8.end_0))
    {
        var _S998 : bool = bond_update_0(i_26, params_0.dt_0, (params_0.fracture_0) != u32(0), isl_37.info_0.w + u32(1), true);
        _S997 = _S998;
    }
    else
    {
        _S997 = false;
    }
    if(_S997)
    {
        islands_0[wg_8.island_2].info_0[i32(2)] = ((isl_37.info_0.z) | (u32(2)));
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn sums_impactors_bonds(@builtin(workgroup_id) group_15 : vec3<u32>, @builtin(local_invocation_id) thread_15 : vec3<u32>)
{
    var sums_groups_1 : u32 = (params_0.seg_count_0 + u32(255)) / u32(256);
    var _S999 : u32 = max(params_0.impactor_count_0, u32(1));
    var _S1000 : u32 = group_15.x;
    if(_S1000 < sums_groups_1)
    {
        contact_sum_at_0(_S1000 * u32(256) + thread_15.x);
    }
    else
    {
        if(_S1000 < (sums_groups_1 + _S999))
        {
            impactor_integrate_at_0(_S1000 - sums_groups_1, thread_15.x);
        }
        else
        {
            var gb_0 : u32 = _S1000 - sums_groups_1 - _S999;
            var narrow_0 : bool = gb_0 < (params_0.contact_bond_groups_0);
            var _S1001 : u32;
            if(narrow_0)
            {
                _S1001 = params_0.contact_bond_table_0;
            }
            else
            {
                _S1001 = params_0.wide_bond_table_0;
            }
            var _S1002 : u32;
            if(narrow_0)
            {
                _S1002 = gb_0;
            }
            else
            {
                _S1002 = gb_0 - params_0.contact_bond_groups_0;
            }
            bond_group_at_0(_S1001, _S1002, thread_15.x);
        }
    }
    return;
}

