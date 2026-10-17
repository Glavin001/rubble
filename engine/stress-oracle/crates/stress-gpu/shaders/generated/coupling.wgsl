struct IngestParams_std140_0
{
    @align(16) row_offset_0 : u32,
    @align(4) max_rows_0 : u32,
    @align(8) substep_0 : u32,
    @align(4) chunk_count_0 : u32,
    @align(16) record_count_0 : u32,
    @align(4) max_bodies_0 : u32,
    @align(8) h_0 : f32,
    @align(4) pad_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : IngestParams_std140_0;
struct Record_std430_0
{
    @align(16) ids_0 : vec4<u32>,
    @align(16) point_0 : vec4<f32>,
    @align(16) impulse_0 : vec4<f32>,
    @align(16) other_0 : vec4<f32>,
};

@binding(7) @group(0) var<storage, read_write> records_0 : array<Record_std430_0>;

@binding(5) @group(0) var<storage, read> counters_0 : array<u32>;

struct EngineContactRow_std430_0
{
    @align(16) ids_1 : vec4<u32>,
    @align(16) normal_pen_0 : vec4<f32>,
    @align(16) anchor_a_0 : vec4<f32>,
    @align(16) anchor_b_0 : vec4<f32>,
    @align(16) t1_lambda_n_0 : vec4<f32>,
    @align(16) t2_lambda_t1_0 : vec4<f32>,
    @align(16) kn_kt_lt2_c0t1_0 : vec4<f32>,
    @align(16) tail_0 : vec4<f32>,
};

@binding(1) @group(0) var<storage, read> contacts_0 : array<EngineContactRow_std430_0>;

struct EngineCollider_std430_0
{
    @align(16) ids_2 : vec4<u32>,
    @align(16) local_pos_0 : vec4<f32>,
    @align(16) local_rot_0 : vec4<f32>,
    @align(16) params_1 : vec4<f32>,
};

@binding(2) @group(0) var<storage, read> colliders_0 : array<EngineCollider_std430_0>;

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

@binding(3) @group(0) var<storage, read> bodies_0 : array<EngineBody_std430_0>;

struct ColliderInfo_std430_0
{
    @align(16) a_0 : vec4<u32>,
    @align(16) b_0 : vec4<f32>,
};

@binding(6) @group(0) var<storage, read> collider_info_0 : array<ColliderInfo_std430_0>;

@binding(4) @group(0) var<storage, read> start_bodies_0 : array<EngineBody_std430_0>;

@binding(10) @group(0) var<storage, read_write> counts_0 : array<atomic<u32>>;

@binding(8) @group(0) var<storage, read_write> sort_0 : array<u32>;

@binding(9) @group(0) var<storage, read_write> manifolds_0 : array<Record_std430_0>;

fn isinf_0( x_0 : f32) -> bool
{
    var _S1 : u32 = (bitcast<u32>((x_0)));
    var _S2 : u32 = (_S1 & (u32(8388607)));
    var _S3 : bool;
    if(((((_S1 >> (u32(23)))) & (u32(255)))) == u32(255))
    {
        _S3 = _S2 == u32(0);
    }
    else
    {
        _S3 = false;
    }
    return _S3;
}

fn isnan_0( x_1 : f32) -> bool
{
    var _S4 : u32 = (bitcast<u32>((x_1)));
    var _S5 : u32 = (_S4 & (u32(8388607)));
    var _S6 : bool;
    if(((((_S4 >> (u32(23)))) & (u32(255)))) == u32(255))
    {
        _S6 = _S5 != u32(0);
    }
    else
    {
        _S6 = false;
    }
    return _S6;
}

fn isinf_1( x_2 : vec3<f32>) -> vec3<bool>
{
    var result_0 : vec3<bool>;
    var i_0 : i32 = i32(0);
    loop
    {
        if(i_0 < i32(3))
        {
        }
        else
        {
            break;
        }
        result_0[i_0] = isinf_0(x_2[i_0]);
        i_0 = i_0 + i32(1);
    }
    return result_0;
}

fn isnan_1( x_3 : vec3<f32>) -> vec3<bool>
{
    var result_1 : vec3<bool>;
    var i_1 : i32 = i32(0);
    loop
    {
        if(i_1 < i32(3))
        {
        }
        else
        {
            break;
        }
        result_1[i_1] = isnan_0(x_3[i_1]);
        i_1 = i_1 + i32(1);
    }
    return result_1;
}

fn rotate_q_0( q_0 : vec4<f32>,  v_0 : vec3<f32>) -> vec3<f32>
{
    var u_0 : vec3<f32> = q_0.xyz;
    var t_0 : vec3<f32> = vec3<f32>(2.0f) * cross(u_0, v_0);
    return v_0 + vec3<f32>(q_0.w) * t_0 + cross(u_0, t_0);
}

struct ColliderInfo_0
{
     a_0 : vec4<u32>,
     b_0 : vec4<f32>,
};

fn point_velocity_0( info_0 : ColliderInfo_0,  body_0 : u32,  p_0 : vec3<f32>) -> vec3<f32>
{
    if((info_0.a_0.z) != u32(2))
    {
        return vec3<f32>(0.0f);
    }
    var s_0 : EngineBody_std430_0 = start_bodies_0[body_0];
    return s_0.vel_sleep_0.xyz + cross(s_0.ang_vel_gen_0.xyz, p_0 - s_0.pos_inv_mass_0.xyz);
}

fn to_zup_0( v_1 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(v_1.x, - v_1.z, v_1.y);
}

struct Record_0
{
     ids_0 : vec4<u32>,
     point_0 : vec4<f32>,
     impulse_0 : vec4<f32>,
     other_0 : vec4<f32>,
};

@compute
@workgroup_size(64, 1, 1)
fn ingest_rows(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var i_2 : u32 = id_0.x;
    if(i_2 >= (params_0.max_rows_0))
    {
        return;
    }
    var base_0 : u32 = u32(2) * (params_0.substep_0 * params_0.max_rows_0 + i_2);
    var none_0 : Record_0;
    none_0.ids_0 = vec4<u32>(u32(4294967295), u32(4294967295), base_0, u32(0));
    var _S7 : vec4<f32> = vec4<f32>(0.0f);
    none_0.point_0 = _S7;
    none_0.impulse_0 = _S7;
    none_0.other_0 = _S7;
    records_0[base_0].ids_0 = none_0.ids_0;
    records_0[base_0].point_0 = none_0.point_0;
    records_0[base_0].impulse_0 = none_0.impulse_0;
    records_0[base_0].other_0 = none_0.other_0;
    var _S8 : u32 = base_0 + u32(1);
    none_0.ids_0[i32(2)] = _S8;
    records_0[_S8].ids_0 = none_0.ids_0;
    records_0[_S8].point_0 = none_0.point_0;
    records_0[_S8].impulse_0 = none_0.impulse_0;
    records_0[_S8].other_0 = none_0.other_0;
    if(i_2 >= counters_0[i32(2)])
    {
        return;
    }
    var c_0 : EngineContactRow_std430_0 = contacts_0[params_0.row_offset_0 + i_2];
    if((c_0.ids_1.w) == u32(0))
    {
        return;
    }
    var n_0 : vec3<f32> = c_0.normal_pen_0.xyz;
    var jb_0 : vec3<f32> = (vec3<f32>(0) - (n_0 * vec3<f32>(c_0.t1_lambda_n_0.w) + c_0.t1_lambda_n_0.xyz * vec3<f32>(c_0.t2_lambda_t1_0.w) + c_0.t2_lambda_t1_0.xyz * vec3<f32>(c_0.kn_kt_lt2_c0t1_0.z))) * vec3<f32>(params_0.h_0);
    var _S9 : bool;
    if((all((jb_0 == vec3<f32>(0.0f)))))
    {
        _S9 = true;
    }
    else
    {
        _S9 = (any((isnan_1(jb_0))));
    }
    if(_S9)
    {
        _S9 = true;
    }
    else
    {
        _S9 = (any((isinf_1(jb_0))));
    }
    if(_S9)
    {
        return;
    }
    var _S10 : u32 = c_0.ids_1.x;
    var _S11 : u32 = c_0.ids_1.y;
    var _S12 : u32 = colliders_0[_S10].ids_2.x;
    var ba_0 : EngineBody_std430_0 = bodies_0[_S12];
    var _S13 : u32 = colliders_0[_S11].ids_2.x;
    var bb_0 : EngineBody_std430_0 = bodies_0[_S13];
    var _S14 : vec3<f32> = (ba_0.pos_inv_mass_0.xyz + rotate_q_0(ba_0.rot_0, c_0.anchor_a_0.xyz) + (bb_0.pos_inv_mass_0.xyz + rotate_q_0(bb_0.rot_0, c_0.anchor_b_0.xyz))) * vec3<f32>(0.5f);
    var _S15 : ColliderInfo_std430_0 = collider_info_0[_S10];
    var _S16 : ColliderInfo_std430_0 = collider_info_0[_S11];
    var side_0 : u32 = u32(0);
    loop
    {
        if(side_0 < u32(2))
        {
        }
        else
        {
            break;
        }
        var _S17 : bool = side_0 == u32(0);
        var mine_0 : ColliderInfo_0;
        if(_S17)
        {
            mine_0.a_0 = _S15.a_0;
            mine_0.b_0 = _S15.b_0;
        }
        else
        {
            mine_0.a_0 = _S16.a_0;
            mine_0.b_0 = _S16.b_0;
        }
        var other_1 : ColliderInfo_0;
        if(_S17)
        {
            other_1.a_0 = _S16.a_0;
            other_1.b_0 = _S16.b_0;
        }
        else
        {
            other_1.a_0 = _S15.a_0;
            other_1.b_0 = _S15.b_0;
        }
        var mine_body_0 : u32;
        if(_S17)
        {
            mine_body_0 = _S12;
        }
        else
        {
            mine_body_0 = _S13;
        }
        var other_body_0 : u32;
        if(_S17)
        {
            other_body_0 = _S13;
        }
        else
        {
            other_body_0 = _S12;
        }
        var _S18 : ColliderInfo_0 = mine_0;
        var _S19 : u32 = mine_0.a_0.x;
        if(_S19 == u32(4294967295))
        {
            _S9 = true;
        }
        else
        {
            _S9 = (_S18.a_0.w) == u32(0);
        }
        var _S20 : bool;
        if(_S9)
        {
            _S20 = true;
        }
        else
        {
            _S20 = (other_1.a_0.z) == u32(0);
        }
        if(_S20)
        {
            side_0 = side_0 + u32(1);
            continue;
        }
        var j_0 : vec3<f32>;
        if(_S17)
        {
            j_0 = (vec3<f32>(0) - jb_0);
        }
        else
        {
            j_0 = jb_0;
        }
        var push_0 : vec3<f32>;
        if(_S17)
        {
            push_0 = (vec3<f32>(0) - n_0);
        }
        else
        {
            push_0 = n_0;
        }
        var _S21 : f32 = max(dot(point_velocity_0(other_1, other_body_0, _S14) - point_velocity_0(mine_0, mine_body_0, _S14), push_0), 0.0f);
        var r_0 : Record_0;
        var _S22 : ColliderInfo_0 = other_1;
        var _S23 : u32;
        if((other_1.a_0.z) == u32(2))
        {
            _S23 = _S22.a_0.y;
        }
        else
        {
            _S23 = u32(4294967295);
        }
        var _S24 : u32 = base_0 + side_0;
        r_0.ids_0 = vec4<u32>(_S19, _S23, _S24, u32(1));
        var jz_0 : vec3<f32> = to_zup_0(j_0);
        r_0.point_0 = vec4<f32>(to_zup_0(_S14), length(jz_0));
        r_0.impulse_0 = vec4<f32>(jz_0, _S21);
        r_0.other_0 = other_1.b_0;
        records_0[_S24].ids_0 = r_0.ids_0;
        records_0[_S24].point_0 = r_0.point_0;
        records_0[_S24].impulse_0 = r_0.impulse_0;
        records_0[_S24].other_0 = r_0.other_0;
        side_0 = side_0 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn ingest_clear(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var _S25 : u32 = id_1.x;
    if(_S25 < (u32(2) * params_0.chunk_count_0))
    {
        atomicStore(&(counts_0[_S25]), u32(0));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn ingest_count(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var _S26 : u32 = id_2.x;
    if(_S26 >= (params_0.record_count_0))
    {
        return;
    }
    var r_1 : Record_std430_0 = records_0[_S26];
    if((records_0[_S26].ids_0.w) != u32(0))
    {
        var _S27 : u32 = atomicAdd(&(counts_0[r_1.ids_0.x]), u32(1));
    }
    return;
}

var<workgroup> g_scan_0 : array<u32, i32(1024)>;

@compute
@workgroup_size(1024, 1, 1)
fn ingest_scan(@builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var tid_0 : u32 = thread_0.x;
    var n_1 : u32 = params_0.chunk_count_0;
    var base_1 : u32 = u32(0);
    var carry_0 : u32 = u32(0);
    loop
    {
        if(base_1 < n_1)
        {
        }
        else
        {
            break;
        }
        var k_0 : u32 = base_1 + tid_0;
        var _S28 : bool = k_0 < n_1;
        var v_2 : u32;
        if(_S28)
        {
            var _S29 : u32 = atomicLoad(&(counts_0[k_0]));
            v_2 = _S29;
        }
        else
        {
            v_2 = u32(0);
        }
        g_scan_0[tid_0] = v_2;
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
            var add_0 : u32;
            if(tid_0 >= step_0)
            {
                add_0 = g_scan_0[tid_0 - step_0];
            }
            else
            {
                add_0 = u32(0);
            }
            workgroupBarrier();
            g_scan_0[tid_0] = g_scan_0[tid_0] + add_0;
            workgroupBarrier();
            step_0 = (step_0 << (u32(1)));
        }
        if(_S28)
        {
            sort_0[k_0] = carry_0 + g_scan_0[tid_0] - v_2;
        }
        var total_0 : u32 = g_scan_0[i32(1023)];
        workgroupBarrier();
        var carry_1 : u32 = carry_0 + total_0;
        base_1 = base_1 + u32(1024);
        carry_0 = carry_1;
    }
    if(tid_0 == u32(0))
    {
        sort_0[n_1] = carry_0;
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn ingest_scatter(@builtin(global_invocation_id) id_3 : vec3<u32>)
{
    var _S30 : u32 = id_3.x;
    if(_S30 >= (params_0.record_count_0))
    {
        return;
    }
    var r_2 : Record_std430_0 = records_0[_S30];
    if((records_0[_S30].ids_0.w) == u32(0))
    {
        return;
    }
    var n_2 : u32 = params_0.chunk_count_0;
    var _S31 : u32 = r_2.ids_0.x;
    var at_0 : u32 = atomicAdd(&(counts_0[params_0.chunk_count_0 + _S31]), u32(1));
    sort_0[u32(2) * n_2 + u32(1) + sort_0[_S31] + at_0] = _S30;
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn ingest_merge(@builtin(global_invocation_id) id_4 : vec3<u32>)
{
    var b_1 : u32;
    var m_0 : u32;
    var k_1 : u32 = id_4.x;
    var n_3 : u32 = params_0.chunk_count_0;
    if(k_1 >= (params_0.chunk_count_0))
    {
        return;
    }
    var begin_0 : u32 = sort_0[k_1];
    var _S32 : u32 = sort_0[k_1 + u32(1)];
    var _S33 : u32 = u32(2) * n_3 + u32(1);
    var a_1 : u32 = sort_0[k_1] + u32(1);
    loop
    {
        if(a_1 < _S32)
        {
        }
        else
        {
            break;
        }
        var v_3 : u32 = sort_0[_S33 + a_1];
        b_1 = a_1;
        loop
        {
            var _S34 : bool;
            if(b_1 > begin_0)
            {
                _S34 = sort_0[_S33 + b_1 - u32(1)] > v_3;
            }
            else
            {
                _S34 = false;
            }
            if(_S34)
            {
            }
            else
            {
                break;
            }
            var _S35 : u32 = _S33 + b_1;
            sort_0[_S35] = sort_0[_S35 - u32(1)];
            b_1 = b_1 - u32(1);
        }
        sort_0[_S33 + b_1] = v_3;
        a_1 = a_1 + u32(1);
    }
    a_1 = begin_0;
    b_1 = u32(0);
    loop
    {
        if(a_1 < _S32)
        {
        }
        else
        {
            break;
        }
        var r_3 : Record_std430_0 = records_0[sort_0[_S33 + a_1]];
        var q_1 : u32 = u32(0);
        loop
        {
            if(q_1 < b_1)
            {
            }
            else
            {
                m_0 = u32(4294967295);
                break;
            }
            if((manifolds_0[begin_0 + q_1].ids_0.y) == (r_3.ids_0.y))
            {
                m_0 = q_1;
                break;
            }
            q_1 = q_1 + u32(1);
        }
        if(m_0 == u32(4294967295))
        {
            manifolds_0[begin_0 + b_1] = r_3;
            b_1 = b_1 + u32(1);
            a_1 = a_1 + u32(1);
            continue;
        }
        var _S36 : u32 = begin_0 + m_0;
        var acc_0 : Record_0;
        acc_0.ids_0 = manifolds_0[_S36].ids_0;
        acc_0.point_0 = manifolds_0[_S36].point_0;
        acc_0.impulse_0 = manifolds_0[_S36].impulse_0;
        acc_0.other_0 = manifolds_0[_S36].other_0;
        var w0_0 : f32 = acc_0.point_0.w;
        var w_0 : f32 = r_3.point_0.w;
        var total_1 : f32 = w0_0 + w_0;
        if(total_1 > 0.0f)
        {
            var _S37 : vec3<f32> = (acc_0.point_0.xyz * vec3<f32>(w0_0) + r_3.point_0.xyz * vec3<f32>(w_0)) / vec3<f32>(total_1);
            acc_0.point_0.x = _S37.x;
            acc_0.point_0.y = _S37.y;
            acc_0.point_0.z = _S37.z;
            acc_0.impulse_0[i32(3)] = (acc_0.impulse_0.w * w0_0 + r_3.impulse_0.w * w_0) / total_1;
        }
        var _S38 : vec3<f32> = acc_0.impulse_0.xyz + r_3.impulse_0.xyz;
        acc_0.impulse_0.x = _S38.x;
        acc_0.impulse_0.y = _S38.y;
        acc_0.impulse_0.z = _S38.z;
        acc_0.point_0[i32(3)] = total_1;
        manifolds_0[_S36].ids_0 = acc_0.ids_0;
        manifolds_0[_S36].point_0 = acc_0.point_0;
        manifolds_0[_S36].impulse_0 = acc_0.impulse_0;
        manifolds_0[_S36].other_0 = acc_0.other_0;
        a_1 = a_1 + u32(1);
    }
    sort_0[n_3 + u32(1) + k_1] = b_1;
    return;
}

