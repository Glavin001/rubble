@binding(13) @group(0) var<storage, read_write> control_0 : array<u32>;

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

struct SplitParams_std140_0
{
    @align(16) parent_count_0 : u32,
    @align(4) out_base_0 : u32,
    @align(8) anchored_pose_0 : u32,
    @align(4) pad1_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : SplitParams_std140_0;
@binding(9) @group(0) var<storage, read_write> out_0 : array<vec4<f32>>;

@binding(10) @group(0) var<storage, read_write> frames_0 : array<vec4<f32>>;

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

@binding(12) @group(0) var<storage, read> parent_bodies_0 : array<vec4<f32>>;

@binding(11) @group(0) var<storage, read> engine_bodies_0 : array<vec4<f32>>;

fn total_chunks_0() -> u32
{
    return parents_0[control_0[u32(1)]].y;
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
    if(p_0 >= control_0[u32(1)])
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
        if(_S19 < control_0[u32(1)])
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

fn quat_mul_0( a_0 : Quat_0,  o_0 : Quat_0) -> Quat_0
{
    var r_2 : Quat_0;
    r_2.w_0 = a_0.w_0 * o_0.w_0 - a_0.x_0 * o_0.x_0 - a_0.y_0 * o_0.y_0 - a_0.z_0 * o_0.z_0;
    r_2.x_0 = a_0.w_0 * o_0.x_0 + a_0.x_0 * o_0.w_0 + a_0.y_0 * o_0.z_0 - a_0.z_0 * o_0.y_0;
    r_2.y_0 = a_0.w_0 * o_0.y_0 - a_0.x_0 * o_0.z_0 + a_0.y_0 * o_0.w_0 + a_0.z_0 * o_0.x_0;
    r_2.z_0 = a_0.w_0 * o_0.z_0 + a_0.x_0 * o_0.y_0 - a_0.y_0 * o_0.x_0 + a_0.z_0 * o_0.w_0;
    return r_2;
}

fn normalized_0( q_5 : Quat_0) -> Quat_0
{
    var _S38 : f32 = q_5.w_0;
    var _S39 : f32 = q_5.x_0;
    var _S40 : f32 = q_5.y_0;
    var _S41 : f32 = q_5.z_0;
    var n_0 : f32 = sqrt(_S38 * _S38 + _S39 * _S39 + _S40 * _S40 + _S41 * _S41);
    var r_3 : Quat_0;
    r_3.w_0 = q_5.w_0 / n_0;
    r_3.x_0 = q_5.x_0 / n_0;
    r_3.y_0 = q_5.y_0 / n_0;
    r_3.z_0 = q_5.z_0 / n_0;
    return r_3;
}

fn quat_vec_0( q_6 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_6.x_0, q_6.y_0, q_6.z_0, q_6.w_0);
}

@compute
@workgroup_size(64, 1, 1)
fn split_frames(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var p_2 : u32 = id_1.x;
    if(p_2 >= control_0[u32(1)])
    {
        return;
    }
    var _S42 : bool;
    if((((islands_0[parents_0[p_2].x].info_0.x) & (u32(1)))) != u32(0))
    {
        _S42 = (params_0.anchored_pose_0) != u32(0);
    }
    else
    {
        _S42 = false;
    }
    if(_S42)
    {
        var _S43 : u32 = u32(5) * p_2;
        frames_0[_S43] = islands_0[parents_0[p_2].x].rotation_0;
        var _S44 : vec4<f32> = vec4<f32>(0.0f);
        frames_0[_S43 + u32(1)] = _S44;
        frames_0[_S43 + u32(2)] = _S44;
        frames_0[_S43 + u32(3)] = vec4<f32>(islands_0[parents_0[p_2].x].com_0.xyz, 0.0f);
        frames_0[_S43 + u32(4)] = vec4<f32>(islands_0[parents_0[p_2].x].position_0.xyz + islands_0[parents_0[p_2].x].position_err_0.xyz, 0.0f);
        return;
    }
    var _S45 : u32 = u32(2) * p_2;
    var head_0 : vec4<f32> = parent_bodies_0[_S45];
    var body_0 : u32 = (bitcast<u32>((head_0.x)));
    if(body_0 == u32(4294967295))
    {
        return;
    }
    var com_2 : vec3<f32> = head_0.yzw;
    var principal_0 : Quat_0 = quat_of_0(parent_bodies_0[_S45 + u32(1)]);
    var _S46 : u32 = u32(7) * body_0;
    var b0_0 : vec4<f32> = engine_bodies_0[_S46];
    var orientation_0 : Quat_0 = quat_of_0(engine_bodies_0[_S46 + u32(1)]);
    var lin_0 : vec3<f32> = engine_bodies_0[_S46 + u32(3)].xyz;
    var ang_0 : vec3<f32> = engine_bodies_0[_S46 + u32(4)].xyz;
    var axes_inv_0 : Quat_0;
    axes_inv_0.w_0 = 0.70710676908493042f;
    axes_inv_0.x_0 = 0.70710676908493042f;
    axes_inv_0.y_0 = 0.0f;
    axes_inv_0.z_0 = 0.0f;
    var p_inv_0 : Quat_0;
    p_inv_0.w_0 = principal_0.w_0;
    p_inv_0.x_0 = - principal_0.x_0;
    p_inv_0.y_0 = - principal_0.y_0;
    p_inv_0.z_0 = - principal_0.z_0;
    var rot_0 : Quat_0 = normalized_0(quat_mul_0(normalized_0(quat_mul_0(axes_inv_0, orientation_0)), p_inv_0));
    var com_world_0 : vec3<f32> = vec3<f32>(b0_0.x, - b0_0.z, b0_0.y);
    var _S47 : u32 = u32(5) * p_2;
    frames_0[_S47] = quat_vec_0(rot_0);
    frames_0[_S47 + u32(1)] = vec4<f32>(lin_0.x, - lin_0.z, lin_0.y, 0.0f);
    frames_0[_S47 + u32(2)] = vec4<f32>(ang_0.x, - ang_0.z, ang_0.y, 0.0f);
    frames_0[_S47 + u32(3)] = vec4<f32>(com_2, 0.0f);
    frames_0[_S47 + u32(4)] = vec4<f32>(com_world_0 - rotate_0(rot_0, com_2), 0.0f);
    return;
}

var<workgroup> g_anchor_0 : u32;

var<workgroup> g_a_0 : array<vec4<f32>, i32(64)>;

var<workgroup> g_b_0 : array<vec4<f32>, i32(64)>;

var<workgroup> g_c_0 : array<vec4<f32>, i32(64)>;

var<workgroup> g_d_0 : array<vec4<f32>, i32(64)>;

var<workgroup> g_e_0 : array<vec4<f32>, i32(64)>;

fn tree5_0( tid_1 : u32)
{
    workgroupBarrier();
    var s_0 : u32 = u32(32);
    loop
    {
        if(s_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_1 < s_0)
        {
            var _S48 : u32 = tid_1 + s_0;
            g_a_0[tid_1] = g_a_0[tid_1] + g_a_0[_S48];
            g_b_0[tid_1] = g_b_0[tid_1] + g_b_0[_S48];
            g_c_0[tid_1] = g_c_0[tid_1] + g_c_0[_S48];
            g_d_0[tid_1] = g_d_0[tid_1] + g_d_0[_S48];
            g_e_0[tid_1] = g_e_0[tid_1] + g_e_0[_S48];
        }
        workgroupBarrier();
        s_0 = (s_0 >> (u32(1)));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn split_children_group(@builtin(workgroup_id) group_1 : vec3<u32>, @builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var slot_1 : u32 = group_1.x;
    var tid_2 : u32 = thread_1.x;
    var total_1 : u32 = total_chunks_0();
    var _S49 : bool;
    if(slot_1 >= total_1)
    {
        _S49 = true;
    }
    else
    {
        _S49 = control_0[u32(1)] == u32(0);
    }
    if(_S49)
    {
        return;
    }
    var _S50 : u32 = control_0[u32(1)] - u32(1);
    var p_3 : u32 = u32(0);
    var hi_0 : u32 = _S50;
    loop
    {
        if(p_3 < hi_0)
        {
        }
        else
        {
            break;
        }
        var mid_0 : u32 = (p_3 + hi_0 + u32(1)) / u32(2);
        if((parents_0[mid_0].y) <= slot_1)
        {
            p_3 = mid_0;
        }
        else
        {
            hi_0 = mid_0 - u32(1);
        }
    }
    var parent_1 : vec4<u32> = parents_0[p_3];
    var _S51 : u32 = parents_0[p_3].y;
    if((slot_1 - _S51) >= (parents_0[p_3].z))
    {
        return;
    }
    var at_3 : u32 = params_0.out_base_0 + u32(8) * slot_1;
    var offset_1 : u32 = (bitcast<u32>((out_0[at_3].y)));
    var count_1 : u32 = (bitcast<u32>((out_0[at_3].z)));
    var _S52 : u32 = islands_0[parent_1.x].range_0.x;
    var _S53 : u32 = u32(5) * p_3;
    var q_7 : Quat_0 = quat_of_0(frames_0[_S53]);
    var _S54 : vec3<f32> = frames_0[_S53 + u32(1)].xyz;
    var _S55 : vec3<f32> = frames_0[_S53 + u32(2)].xyz;
    var _S56 : vec3<f32> = frames_0[_S53 + u32(3)].xyz;
    if(tid_2 == u32(0))
    {
        g_anchor_0 = u32(0);
    }
    var _S57 : vec4<f32> = vec4<f32>(0.0f);
    var anchored_1 : u32 = u32(0);
    var j_1 : u32 = tid_2;
    var mm_0 : vec4<f32> = _S57;
    loop
    {
        if(j_1 < count_1)
        {
        }
        else
        {
            break;
        }
        var _S58 : f32 = chunks_0[_S52 + work_0[total_1 + _S51 + offset_1 + j_1]].center_0.w;
        var mm_1 : vec4<f32> = mm_0 + vec4<f32>(chunks_0[_S52 + work_0[total_1 + _S51 + offset_1 + j_1]].center_0.xyz * vec3<f32>(_S58), _S58);
        if((chunks_0[_S52 + work_0[total_1 + _S51 + offset_1 + j_1]].info_1.x) != u32(0))
        {
            anchored_1 = u32(1);
        }
        j_1 = j_1 + u32(64);
        mm_0 = mm_1;
    }
    g_a_0[tid_2] = mm_0;
    g_b_0[tid_2] = _S57;
    g_c_0[tid_2] = _S57;
    g_d_0[tid_2] = _S57;
    g_e_0[tid_2] = _S57;
    tree5_0(tid_2);
    if(anchored_1 != u32(0))
    {
        g_anchor_0 = u32(1);
    }
    var mass_2 : f32 = g_a_0[i32(0)].w;
    var _S59 : vec3<f32> = vec3<f32>(mass_2);
    var com_3 : vec3<f32> = g_a_0[i32(0)].xyz / _S59;
    workgroupBarrier();
    var _S60 : vec3<f32> = vec3<f32>(0.0f);
    j_1 = tid_2;
    var i0_3 : vec3<f32> = _S60;
    var i1_3 : vec3<f32> = _S60;
    var i2_3 : vec3<f32> = _S60;
    var pm_2 : vec3<f32> = _S60;
    var l_2 : vec3<f32> = _S60;
    loop
    {
        if(j_1 < count_1)
        {
        }
        else
        {
            break;
        }
        var c_3 : u32 = _S52 + work_0[total_1 + _S51 + offset_1 + j_1];
        var _S61 : vec3<f32> = chunks_0[c_3].center_0.xyz;
        var r_4 : vec3<f32> = _S61 - com_3;
        var rr_1 : f32 = dot(r_4, r_4);
        var _S62 : vec3<f32> = chunks_0[c_3].inertia0_1.xyz;
        var _S63 : vec3<f32> = vec3<f32>(chunks_0[c_3].center_0.w);
        var i0_4 : vec3<f32> = i0_3 + (_S62 + (vec3<f32>(rr_1, 0.0f, 0.0f) - vec3<f32>(r_4.x) * r_4) * _S63);
        var _S64 : vec3<f32> = chunks_0[c_3].inertia1_1.xyz;
        var i1_4 : vec3<f32> = i1_3 + (_S64 + (vec3<f32>(0.0f, rr_1, 0.0f) - vec3<f32>(r_4.y) * r_4) * _S63);
        var _S65 : vec3<f32> = chunks_0[c_3].inertia2_1.xyz;
        var i2_4 : vec3<f32> = i2_3 + (_S65 + (vec3<f32>(0.0f, 0.0f, rr_1) - vec3<f32>(r_4.z) * r_4) * _S63);
        var _S66 : u32 = u32(4) * c_3;
        var v_4 : vec3<f32> = _S54 + cross(_S55, rotate_0(q_7, _S61 + state_0[_S66].xyz - _S56)) + rotate_0(q_7, state_0[_S66 + u32(2)].xyz);
        var w_2 : vec3<f32> = _S55 + rotate_0(q_7, state_0[_S66 + u32(3)].xyz);
        var pm_3 : vec3<f32> = pm_2 + v_4 * _S63;
        var rw_1 : vec3<f32> = rotate_0(q_7, r_4);
        var ci0_1 : vec3<f32>;
        var ci1_1 : vec3<f32>;
        var ci2_1 : vec3<f32>;
        rotate_matrix_0(q_7, _S62, _S64, _S65, &(ci0_1), &(ci1_1), &(ci2_1));
        var l_3 : vec3<f32> = l_2 + (cross(rw_1, v_4) * _S63 + mat_mul_0(ci0_1, ci1_1, ci2_1, w_2));
        j_1 = j_1 + u32(64);
        i0_3 = i0_4;
        i1_3 = i1_4;
        i2_3 = i2_4;
        pm_2 = pm_3;
        l_2 = l_3;
    }
    g_a_0[tid_2] = vec4<f32>(i0_3, 0.0f);
    g_b_0[tid_2] = vec4<f32>(i1_3, 0.0f);
    g_c_0[tid_2] = vec4<f32>(i2_3, 0.0f);
    g_d_0[tid_2] = vec4<f32>(pm_2, 0.0f);
    g_e_0[tid_2] = vec4<f32>(l_2, 0.0f);
    tree5_0(tid_2);
    if(tid_2 != u32(0))
    {
        return;
    }
    var i0_5 : vec3<f32> = g_a_0[i32(0)].xyz;
    var i1_5 : vec3<f32> = g_b_0[i32(0)].xyz;
    var i2_5 : vec3<f32> = g_c_0[i32(0)].xyz;
    var pm_4 : vec3<f32> = g_d_0[i32(0)].xyz;
    var l_4 : vec3<f32> = g_e_0[i32(0)].xyz;
    var anch_0 : bool = g_anchor_0 != u32(0);
    var cv_1 : vec3<f32>;
    var cw_1 : vec3<f32>;
    if(!anch_0)
    {
        var w0_1 : vec3<f32>;
        var w1_1 : vec3<f32>;
        var w2_1 : vec3<f32>;
        rotate_matrix_0(q_7, i0_5, i1_5, i2_5, &(w0_1), &(w1_1), &(w2_1));
        var v0_1 : vec3<f32>;
        var v1_1 : vec3<f32>;
        var v2_1 : vec3<f32>;
        inverse3_0(w0_1, w1_1, w2_1, &(v0_1), &(v1_1), &(v2_1));
        var _S67 : vec3<f32> = mat_mul_0(v0_1, v1_1, v2_1, l_4);
        cv_1 = pm_4 / _S59;
        cw_1 = _S67;
    }
    else
    {
        cv_1 = _S60;
        cw_1 = _S60;
    }
    var _S68 : f32 = (bitcast<f32>((p_3)));
    var _S69 : f32 = (bitcast<f32>((offset_1)));
    var _S70 : f32 = (bitcast<f32>((count_1)));
    if(anch_0)
    {
        p_3 = u32(1);
    }
    else
    {
        p_3 = u32(0);
    }
    out_0[at_3] = vec4<f32>(_S68, _S69, _S70, (bitcast<f32>((p_3))));
    out_0[at_3 + u32(1)] = vec4<f32>(com_3, mass_2);
    out_0[at_3 + u32(2)] = vec4<f32>(i0_5, 0.0f);
    out_0[at_3 + u32(3)] = vec4<f32>(i1_5, 0.0f);
    out_0[at_3 + u32(4)] = vec4<f32>(i2_5, 0.0f);
    out_0[at_3 + u32(5)] = vec4<f32>(cv_1, 0.0f);
    out_0[at_3 + u32(6)] = vec4<f32>(cw_1, 0.0f);
    out_0[at_3 + u32(7)] = _S57;
    return;
}

