struct SplitParams_std140_0
{
    @align(16) parent_count_0 : u32,
    @align(4) out_base_0 : u32,
    @align(8) pad0_0 : u32,
    @align(4) pad1_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : SplitParams_std140_0;
@binding(7) @group(0) var<storage, read_write> parents_0 : array<vec4<u32>>;

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

@binding(5) @group(0) var<storage, read> islands_0 : array<Island_std430_0>;

@binding(8) @group(0) var<storage, read_write> work_0 : array<u32>;

@binding(2) @group(0) var<storage, read> index_0 : array<u32>;

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

@binding(4) @group(0) var<storage, read> bond_dyn_0 : array<BondDyn_std430_0>;

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

@binding(3) @group(0) var<storage, read> bonds_0 : array<BondStatic_std430_0>;

@binding(9) @group(0) var<storage, read_write> out_0 : array<vec4<f32>>;

@binding(10) @group(0) var<storage, read> frames_0 : array<vec4<f32>>;

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

@binding(1) @group(0) var<storage, read> chunks_0 : array<ChunkStatic_std430_0>;

@binding(6) @group(0) var<storage, read> state_0 : array<vec4<f32>>;

fn total_chunks_0() -> u32
{
    return parents_0[params_0.parent_count_0].y;
}

fn device_group_barrier_0()
{
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    return;
}

var<workgroup> g_changed_0 : u32;

fn connected_0( st_0 : ptr<function, JointState_std430_0>,  has_rebar_0 : bool) -> bool
{
    var _S1 : bool;
    if(((*st_0).damage_0) < 1.0f)
    {
        _S1 = true;
    }
    else
    {
        if(has_rebar_0)
        {
            _S1 = ((*st_0).rebar_broken_0) == 0.0f;
        }
        else
        {
            _S1 = false;
        }
    }
    return _S1;
}

@compute
@workgroup_size(256, 1, 1)
fn split_components(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var children_0 : u32;
    var k_0 : u32;
    var at_0 : u32;
    var tid_0 : u32 = thread_0.x;
    var p_0 : u32 = group_0.x;
    if(p_0 >= (params_0.parent_count_0))
    {
        return;
    }
    var _S2 : u32 = parents_0[p_0].x;
    var _S3 : u32 = islands_0[_S2].range_0.x;
    var _S4 : u32 = islands_0[_S2].range_0.y - _S3;
    var _S5 : u32 = parents_0[p_0].y;
    var _S6 : u32 = total_chunks_0();
    var c_0 : u32 = tid_0;
    loop
    {
        if(c_0 < _S4)
        {
        }
        else
        {
            break;
        }
        work_0[_S5 + c_0] = c_0;
        c_0 = c_0 + u32(256);
    }
    device_group_barrier_0();
    var pass_0 : u32 = u32(0);
    loop
    {
        if(pass_0 <= _S4)
        {
        }
        else
        {
            break;
        }
        if(tid_0 == u32(0))
        {
            g_changed_0 = u32(0);
        }
        workgroupBarrier();
        c_0 = tid_0;
        loop
        {
            if(c_0 < _S4)
            {
            }
            else
            {
                break;
            }
            var _S7 : u32 = _S5 + c_0;
            var gc_0 : u32 = _S3 + c_0;
            var _S8 : u32 = index_0[gc_0];
            children_0 = work_0[_S7];
            k_0 = _S8;
            loop
            {
                if(k_0 < index_0[gc_0 + u32(1)])
                {
                }
                else
                {
                    break;
                }
                var entry_0 : u32 = index_0[k_0];
                var bond_0 : u32 = (entry_0 >> (u32(1)));
                var _S9 : JointState_std430_0 = bond_dyn_0[bond_0].js_0;
                var _S10 : bool = connected_0(&(_S9), (bonds_0[bond_0].law_0.stiff1_0.w) != 0.0f);
                if(!_S10)
                {
                    k_0 = k_0 + u32(1);
                    continue;
                }
                if(((entry_0 & (u32(1)))) == u32(0))
                {
                    at_0 = bonds_0[bond_0].law_0.ids_0.z;
                }
                else
                {
                    at_0 = bonds_0[bond_0].law_0.ids_0.y;
                }
                var _S11 : bool;
                if(at_0 < _S3)
                {
                    _S11 = true;
                }
                else
                {
                    _S11 = at_0 >= (_S3 + _S4);
                }
                if(_S11)
                {
                    k_0 = k_0 + u32(1);
                    continue;
                }
                children_0 = min(children_0, work_0[_S5 + at_0 - _S3]);
                k_0 = k_0 + u32(1);
            }
            var _S12 : u32 = min(children_0, work_0[_S5 + children_0]);
            if(_S12 < work_0[_S7])
            {
                work_0[_S7] = _S12;
                g_changed_0 = u32(1);
            }
            c_0 = c_0 + u32(256);
        }
        device_group_barrier_0();
        var done_1 : bool = g_changed_0 == u32(0);
        workgroupBarrier();
        if(done_1)
        {
            break;
        }
        pass_0 = pass_0 + u32(1);
    }
    if(tid_0 != u32(0))
    {
        return;
    }
    c_0 = u32(0);
    children_0 = u32(0);
    loop
    {
        if(c_0 < _S4)
        {
        }
        else
        {
            break;
        }
        if(work_0[_S5 + c_0] == c_0)
        {
            work_0[_S6 + _S5 + c_0] = children_0;
            children_0 = children_0 + u32(1);
        }
        c_0 = c_0 + u32(1);
    }
    k_0 = u32(0);
    loop
    {
        if(k_0 < children_0)
        {
        }
        else
        {
            break;
        }
        var _S13 : f32 = (bitcast<f32>((u32(0))));
        out_0[params_0.out_base_0 + u32(8) * (_S5 + k_0)] = vec4<f32>((bitcast<f32>((p_0))), _S13, _S13, _S13);
        k_0 = k_0 + u32(1);
    }
    c_0 = u32(0);
    loop
    {
        if(c_0 < _S4)
        {
        }
        else
        {
            break;
        }
        var _S14 : u32 = u32(8) * (_S5 + work_0[_S6 + _S5 + work_0[_S5 + c_0]]);
        var rec_0 : vec4<f32> = out_0[params_0.out_base_0 + _S14];
        rec_0[i32(2)] = (bitcast<f32>(((bitcast<u32>((out_0[params_0.out_base_0 + _S14].z))) + u32(1))));
        out_0[params_0.out_base_0 + _S14] = rec_0;
        c_0 = c_0 + u32(1);
    }
    k_0 = u32(0);
    at_0 = u32(0);
    loop
    {
        if(k_0 < children_0)
        {
        }
        else
        {
            break;
        }
        var _S15 : u32 = u32(8) * (_S5 + k_0);
        var rec_1 : vec4<f32> = out_0[params_0.out_base_0 + _S15];
        rec_1[i32(1)] = (bitcast<f32>((at_0)));
        var at_1 : u32 = at_0 + (bitcast<u32>((rec_1.z)));
        rec_1[i32(2)] = (bitcast<f32>((u32(0))));
        out_0[params_0.out_base_0 + _S15] = rec_1;
        k_0 = k_0 + u32(1);
        at_0 = at_1;
    }
    c_0 = u32(0);
    loop
    {
        if(c_0 < _S4)
        {
        }
        else
        {
            break;
        }
        var _S16 : u32 = _S5 + c_0;
        work_0[_S16] = work_0[_S6 + _S5 + work_0[_S16]];
        c_0 = c_0 + u32(1);
    }
    c_0 = u32(0);
    loop
    {
        if(c_0 < _S4)
        {
        }
        else
        {
            break;
        }
        var _S17 : u32 = u32(8) * (_S5 + work_0[_S5 + c_0]);
        var rec_2 : vec4<f32> = out_0[params_0.out_base_0 + _S17];
        var _S18 : u32 = (bitcast<u32>((out_0[params_0.out_base_0 + _S17].z)));
        work_0[_S6 + _S5 + ((bitcast<u32>((out_0[params_0.out_base_0 + _S17].y))) + _S18)] = c_0;
        rec_2[i32(2)] = (bitcast<f32>((_S18 + u32(1))));
        out_0[params_0.out_base_0 + _S17] = rec_2;
        c_0 = c_0 + u32(1);
    }
    parents_0[p_0] = vec4<u32>(_S2, _S5, children_0, u32(0));
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

fn rotate_0( q_1 : Quat_0,  v_0 : vec3<f32>) -> vec3<f32>
{
    var qv_0 : vec3<f32> = vec3<f32>(q_1.x_0, q_1.y_0, q_1.z_0);
    var t_0 : vec3<f32> = cross(qv_0, v_0) * vec3<f32>(2.0f);
    return v_0 + t_0 * vec3<f32>(q_1.w_0) + cross(qv_0, t_0);
}

fn inverse_rotate_0( q_2 : Quat_0,  v_1 : vec3<f32>) -> vec3<f32>
{
    var c_1 : Quat_0;
    c_1.w_0 = q_2.w_0;
    c_1.x_0 = - q_2.x_0;
    c_1.y_0 = - q_2.y_0;
    c_1.z_0 = - q_2.z_0;
    return rotate_0(c_1, v_1);
}

fn mat_mul_0( r0_0 : vec3<f32>,  r1_0 : vec3<f32>,  r2_0 : vec3<f32>,  v_2 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(dot(r0_0, v_2), dot(r1_0, v_2), dot(r2_0, v_2));
}

fn rotate_matrix_0( q_3 : Quat_0,  m0_0 : vec3<f32>,  m1_0 : vec3<f32>,  m2_0 : vec3<f32>,  o0_0 : ptr<function, vec3<f32>>,  o1_0 : ptr<function, vec3<f32>>,  o2_0 : ptr<function, vec3<f32>>)
{
    var k0_0 : vec3<f32> = rotate_0(q_3, mat_mul_0(m0_0, m1_0, m2_0, inverse_rotate_0(q_3, vec3<f32>(1.0f, 0.0f, 0.0f))));
    var k1_0 : vec3<f32> = rotate_0(q_3, mat_mul_0(m0_0, m1_0, m2_0, inverse_rotate_0(q_3, vec3<f32>(0.0f, 1.0f, 0.0f))));
    var k2_0 : vec3<f32> = rotate_0(q_3, mat_mul_0(m0_0, m1_0, m2_0, inverse_rotate_0(q_3, vec3<f32>(0.0f, 0.0f, 1.0f))));
    (*o0_0) = vec3<f32>(k0_0.x, k1_0.x, k2_0.x);
    (*o1_0) = vec3<f32>(k0_0.y, k1_0.y, k2_0.y);
    (*o2_0) = vec3<f32>(k0_0.z, k1_0.z, k2_0.z);
    return;
}

fn inverse3_0( r0_1 : vec3<f32>,  r1_1 : vec3<f32>,  r2_1 : vec3<f32>,  i0_0 : ptr<function, vec3<f32>>,  i1_0 : ptr<function, vec3<f32>>,  i2_0 : ptr<function, vec3<f32>>)
{
    var c0_0 : vec3<f32> = cross(r1_1, r2_1);
    var c1_0 : vec3<f32> = cross(r2_1, r0_1);
    var c2_0 : vec3<f32> = cross(r0_1, r1_1);
    var det_0 : f32 = dot(r0_1, c0_0);
    var inv_0 : f32;
    if(det_0 != 0.0f)
    {
        inv_0 = 1.0f / det_0;
    }
    else
    {
        inv_0 = 0.0f;
    }
    (*i0_0) = vec3<f32>(c0_0.x, c1_0.x, c2_0.x) * vec3<f32>(inv_0);
    (*i1_0) = vec3<f32>(c0_0.y, c1_0.y, c2_0.y) * vec3<f32>(inv_0);
    (*i2_0) = vec3<f32>(c0_0.z, c1_0.z, c2_0.z) * vec3<f32>(inv_0);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn split_children(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var anchored_0 : bool;
    var slot_0 : u32 = id_0.x;
    var total_0 : u32 = total_chunks_0();
    if(slot_0 >= total_0)
    {
        return;
    }
    var p_1 : u32 = u32(0);
    loop
    {
        var _S19 : u32 = p_1 + u32(1);
        if(_S19 < (params_0.parent_count_0))
        {
            anchored_0 = (parents_0[_S19].y) <= slot_0;
        }
        else
        {
            anchored_0 = false;
        }
        if(anchored_0)
        {
        }
        else
        {
            break;
        }
        p_1 = _S19;
    }
    var parent_0 : vec4<u32> = parents_0[p_1];
    var _S20 : u32 = parents_0[p_1].y;
    if((slot_0 - _S20) >= (parents_0[p_1].z))
    {
        return;
    }
    var at_2 : u32 = params_0.out_base_0 + u32(8) * slot_0;
    var offset_0 : u32 = (bitcast<u32>((out_0[at_2].y)));
    var count_0 : u32 = (bitcast<u32>((out_0[at_2].z)));
    var _S21 : u32 = islands_0[parent_0.x].range_0.x;
    var _S22 : u32 = u32(5) * p_1;
    var q_4 : Quat_0 = quat_of_0(frames_0[_S22]);
    var _S23 : vec3<f32> = frames_0[_S22 + u32(1)].xyz;
    var _S24 : vec3<f32> = frames_0[_S22 + u32(2)].xyz;
    var _S25 : vec3<f32> = frames_0[_S22 + u32(3)].xyz;
    var _S26 : vec3<f32> = vec3<f32>(0.0f);
    anchored_0 = false;
    var j_0 : u32 = u32(0);
    var mass_0 : f32 = 0.0f;
    var moment_0 : vec3<f32> = _S26;
    loop
    {
        if(j_0 < count_0)
        {
        }
        else
        {
            break;
        }
        var _S27 : f32 = chunks_0[_S21 + work_0[total_0 + _S20 + offset_0 + j_0]].center_0.w;
        var mass_1 : f32 = mass_0 + _S27;
        var moment_1 : vec3<f32> = moment_0 + chunks_0[_S21 + work_0[total_0 + _S20 + offset_0 + j_0]].center_0.xyz * vec3<f32>(_S27);
        if(anchored_0)
        {
            anchored_0 = true;
        }
        else
        {
            anchored_0 = (chunks_0[_S21 + work_0[total_0 + _S20 + offset_0 + j_0]].info_1.x) != u32(0);
        }
        j_0 = j_0 + u32(1);
        mass_0 = mass_1;
        moment_0 = moment_1;
    }
    var com_1 : vec3<f32> = moment_0 / vec3<f32>(mass_0);
    j_0 = u32(0);
    var i0_1 : vec3<f32> = _S26;
    var i1_1 : vec3<f32> = _S26;
    var i2_1 : vec3<f32> = _S26;
    var pm_0 : vec3<f32> = _S26;
    var l_0 : vec3<f32> = _S26;
    loop
    {
        if(j_0 < count_0)
        {
        }
        else
        {
            break;
        }
        var c_2 : u32 = _S21 + work_0[total_0 + _S20 + offset_0 + j_0];
        var _S28 : vec3<f32> = chunks_0[c_2].center_0.xyz;
        var r_1 : vec3<f32> = _S28 - com_1;
        var rr_0 : f32 = dot(r_1, r_1);
        var _S29 : vec3<f32> = chunks_0[c_2].inertia0_1.xyz;
        var _S30 : vec3<f32> = vec3<f32>(chunks_0[c_2].center_0.w);
        var i0_2 : vec3<f32> = i0_1 + (_S29 + (vec3<f32>(rr_0, 0.0f, 0.0f) - vec3<f32>(r_1.x) * r_1) * _S30);
        var _S31 : vec3<f32> = chunks_0[c_2].inertia1_1.xyz;
        var i1_2 : vec3<f32> = i1_1 + (_S31 + (vec3<f32>(0.0f, rr_0, 0.0f) - vec3<f32>(r_1.y) * r_1) * _S30);
        var _S32 : vec3<f32> = chunks_0[c_2].inertia2_1.xyz;
        var i2_2 : vec3<f32> = i2_1 + (_S32 + (vec3<f32>(0.0f, 0.0f, rr_0) - vec3<f32>(r_1.z) * r_1) * _S30);
        var _S33 : u32 = u32(4) * c_2;
        var v_3 : vec3<f32> = _S23 + cross(_S24, rotate_0(q_4, _S28 + state_0[_S33].xyz - _S25)) + rotate_0(q_4, state_0[_S33 + u32(2)].xyz);
        var w_1 : vec3<f32> = _S24 + rotate_0(q_4, state_0[_S33 + u32(3)].xyz);
        var pm_1 : vec3<f32> = pm_0 + v_3 * _S30;
        var rw_0 : vec3<f32> = rotate_0(q_4, r_1);
        var ci0_0 : vec3<f32>;
        var ci1_0 : vec3<f32>;
        var ci2_0 : vec3<f32>;
        rotate_matrix_0(q_4, _S29, _S31, _S32, &(ci0_0), &(ci1_0), &(ci2_0));
        var l_1 : vec3<f32> = l_0 + (cross(rw_0, v_3) * _S30 + mat_mul_0(ci0_0, ci1_0, ci2_0, w_1));
        j_0 = j_0 + u32(1);
        i0_1 = i0_2;
        i1_1 = i1_2;
        i2_1 = i2_2;
        pm_0 = pm_1;
        l_0 = l_1;
    }
    var cv_0 : vec3<f32>;
    var cw_0 : vec3<f32>;
    if(!anchored_0)
    {
        var w0_0 : vec3<f32>;
        var w1_0 : vec3<f32>;
        var w2_0 : vec3<f32>;
        rotate_matrix_0(q_4, i0_1, i1_1, i2_1, &(w0_0), &(w1_0), &(w2_0));
        var v0_0 : vec3<f32>;
        var v1_0 : vec3<f32>;
        var v2_0 : vec3<f32>;
        inverse3_0(w0_0, w1_0, w2_0, &(v0_0), &(v1_0), &(v2_0));
        var _S34 : vec3<f32> = mat_mul_0(v0_0, v1_0, v2_0, l_0);
        cv_0 = pm_0 / vec3<f32>(mass_0);
        cw_0 = _S34;
    }
    else
    {
        cv_0 = _S26;
        cw_0 = _S26;
    }
    var _S35 : f32 = (bitcast<f32>((p_1)));
    var _S36 : f32 = (bitcast<f32>((offset_0)));
    var _S37 : f32 = (bitcast<f32>((count_0)));
    if(anchored_0)
    {
        p_1 = u32(1);
    }
    else
    {
        p_1 = u32(0);
    }
    out_0[at_2] = vec4<f32>(_S35, _S36, _S37, (bitcast<f32>((p_1))));
    out_0[at_2 + u32(1)] = vec4<f32>(com_1, mass_0);
    out_0[at_2 + u32(2)] = vec4<f32>(i0_1, 0.0f);
    out_0[at_2 + u32(3)] = vec4<f32>(i1_1, 0.0f);
    out_0[at_2 + u32(4)] = vec4<f32>(i2_1, 0.0f);
    out_0[at_2 + u32(5)] = vec4<f32>(cv_0, 0.0f);
    out_0[at_2 + u32(6)] = vec4<f32>(cw_0, 0.0f);
    out_0[at_2 + u32(7)] = vec4<f32>(0.0f);
    return;
}

