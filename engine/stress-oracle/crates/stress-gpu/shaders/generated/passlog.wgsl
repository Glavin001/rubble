@binding(1) @group(0) var<storage, read> control_0 : array<u32>;

@binding(2) @group(0) var<storage, read> parents_0 : array<vec4<u32>>;

@binding(9) @group(0) var<storage, read_write> log_0 : array<u32>;

struct LogParams_std140_0
{
    @align(16) log_words_0 : u32,
    @align(4) pad0_0 : u32,
    @align(8) pad1_0 : u32,
    @align(4) pad2_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : LogParams_std140_0;
@binding(10) @group(0) var<storage, read> status_0 : array<u32>;

@binding(4) @group(0) var<storage, read> children_0 : array<vec4<f32>>;

@binding(5) @group(0) var<storage, read> slots_0 : array<vec4<u32>>;

@binding(6) @group(0) var<storage, read> extras_0 : array<vec4<f32>>;

@binding(3) @group(0) var<storage, read> work_0 : array<u32>;

@binding(7) @group(0) var<storage, read> bodies_0 : array<vec4<f32>>;

@binding(8) @group(0) var<storage, read> colliders_0 : array<vec4<f32>>;

var<workgroup> g_children_0 : u32;

var<workgroup> g_size_0 : u32;

var<workgroup> g_at_0 : u32;

@compute
@workgroup_size(256, 1, 1)
fn pass_log(@builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var tid_0 : u32 = thread_0.x;
    var n_0 : u32 = control_0[u32(1)];
    if(n_0 == u32(0))
    {
        return;
    }
    var total_0 : u32 = control_0[u32(6)];
    var aborted_0 : bool = control_0[u32(0)] != u32(0);
    var first_body_0 : u32 = control_0[u32(4)];
    var first_collider_0 : u32 = control_0[u32(5)];
    var nb_0 : u32;
    if(aborted_0)
    {
        nb_0 = u32(0);
    }
    else
    {
        nb_0 = control_0[u32(8)] - first_body_0;
    }
    var nc_0 : u32;
    if(aborted_0)
    {
        nc_0 = u32(0);
    }
    else
    {
        nc_0 = control_0[u32(9)] - first_collider_0;
    }
    var _S1 : bool = tid_0 == u32(0);
    var j_0 : u32;
    var p_0 : u32;
    if(_S1)
    {
        j_0 = u32(0);
        p_0 = u32(0);
        loop
        {
            if(j_0 < n_0)
            {
            }
            else
            {
                break;
            }
            var c_0 : u32 = p_0 + parents_0[j_0].z;
            j_0 = j_0 + u32(1);
            p_0 = c_0;
        }
        var size_0 : u32 = u32(16) + u32(2) * n_0 + u32(14) * p_0 + total_0 + u32(28) * nb_0 + u32(16) * nc_0;
        var _S2 : u32 = max(log_0[i32(0)], u32(16));
        g_children_0 = p_0;
        if((_S2 + size_0) > (params_0.log_words_0))
        {
            log_0[i32(2)] = u32(1);
            g_size_0 = u32(0);
        }
        else
        {
            g_size_0 = size_0;
        }
        g_at_0 = _S2;
    }
    workgroupBarrier();
    var at_0 : u32 = g_at_0;
    var size_1 : u32 = g_size_0;
    var c_1 : u32 = g_children_0;
    if(g_size_0 == u32(0))
    {
        return;
    }
    if(_S1)
    {
        log_0[at_0] = size_1;
        log_0[at_0 + u32(1)] = n_0;
        log_0[at_0 + u32(2)] = c_1;
        if(aborted_0)
        {
            j_0 = u32(1);
        }
        else
        {
            j_0 = u32(0);
        }
        log_0[at_0 + u32(3)] = j_0;
        log_0[at_0 + u32(4)] = status_0[i32(1)] - u32(1);
        log_0[at_0 + u32(5)] = first_body_0;
        log_0[at_0 + u32(6)] = first_collider_0;
        log_0[at_0 + u32(7)] = first_body_0 + nb_0;
        log_0[at_0 + u32(8)] = first_collider_0 + nc_0;
        log_0[at_0 + u32(9)] = control_0[u32(2)];
        log_0[at_0 + u32(10)] = control_0[u32(3)];
        log_0[at_0 + u32(11)] = total_0;
        j_0 = u32(12);
        loop
        {
            if(j_0 < u32(16))
            {
            }
            else
            {
                break;
            }
            log_0[at_0 + j_0] = u32(0);
            j_0 = j_0 + u32(1);
        }
        var _S3 : u32 = at_0 + u32(16);
        var _S4 : u32 = _S3 + u32(2) * n_0;
        p_0 = u32(0);
        var h_0 : u32 = _S4;
        loop
        {
            if(p_0 < n_0)
            {
            }
            else
            {
                break;
            }
            var parent_0 : vec4<u32> = parents_0[p_0];
            var _S5 : u32 = _S3 + u32(2) * p_0;
            log_0[_S5] = parent_0.x;
            var _S6 : u32 = parent_0.z;
            log_0[_S5 + u32(1)] = _S6;
            var k_0 : u32 = u32(0);
            loop
            {
                if(k_0 < _S6)
                {
                }
                else
                {
                    break;
                }
                var slot_0 : u32 = parent_0.y + k_0;
                var head_0 : vec4<f32> = children_0[u32(8) * slot_0];
                var s_0 : vec4<u32>;
                if(aborted_0)
                {
                    s_0 = vec4<u32>(vec4<i32>(i32(0)));
                }
                else
                {
                    s_0 = slots_0[slot_0];
                }
                var _S7 : u32 = u32(2) * slot_0;
                var com_0 : vec4<f32> = extras_0[_S7];
                var rot_0 : vec4<f32> = extras_0[_S7 + u32(1)];
                log_0[h_0] = (bitcast<u32>((head_0.z)));
                log_0[h_0 + u32(1)] = (bitcast<u32>((head_0.w)));
                log_0[h_0 + u32(2)] = s_0.x;
                log_0[h_0 + u32(3)] = s_0.y;
                log_0[h_0 + u32(4)] = s_0.z;
                log_0[h_0 + u32(5)] = s_0.w;
                log_0[h_0 + u32(6)] = (bitcast<u32>((com_0.x)));
                log_0[h_0 + u32(7)] = (bitcast<u32>((com_0.y)));
                log_0[h_0 + u32(8)] = (bitcast<u32>((com_0.z)));
                log_0[h_0 + u32(9)] = (bitcast<u32>((com_0.w)));
                log_0[h_0 + u32(10)] = (bitcast<u32>((rot_0.x)));
                log_0[h_0 + u32(11)] = (bitcast<u32>((rot_0.y)));
                log_0[h_0 + u32(12)] = (bitcast<u32>((rot_0.z)));
                log_0[h_0 + u32(13)] = (bitcast<u32>((rot_0.w)));
                var h_1 : u32 = h_0 + u32(14);
                k_0 = k_0 + u32(1);
                h_0 = h_1;
            }
            p_0 = p_0 + u32(1);
        }
    }
    var m_0 : u32 = at_0 + u32(16) + u32(2) * n_0 + u32(14) * c_1;
    j_0 = tid_0;
    loop
    {
        if(j_0 < total_0)
        {
        }
        else
        {
            break;
        }
        log_0[m_0 + j_0] = work_0[total_0 + j_0];
        j_0 = j_0 + u32(256);
    }
    var b_0 : u32 = m_0 + total_0;
    j_0 = tid_0;
    loop
    {
        if(j_0 < (u32(7) * nb_0))
        {
        }
        else
        {
            break;
        }
        var v_0 : vec4<f32> = bodies_0[u32(7) * first_body_0 + j_0];
        var _S8 : u32 = b_0 + u32(4) * j_0;
        log_0[_S8] = (bitcast<u32>((v_0.x)));
        log_0[_S8 + u32(1)] = (bitcast<u32>((v_0.y)));
        log_0[_S8 + u32(2)] = (bitcast<u32>((v_0.z)));
        log_0[_S8 + u32(3)] = (bitcast<u32>((v_0.w)));
        j_0 = j_0 + u32(256);
    }
    var _S9 : u32 = b_0 + u32(28) * nb_0;
    j_0 = tid_0;
    loop
    {
        if(j_0 < (u32(4) * nc_0))
        {
        }
        else
        {
            break;
        }
        var v_1 : vec4<f32> = colliders_0[u32(4) * first_collider_0 + j_0];
        var _S10 : u32 = _S9 + u32(4) * j_0;
        log_0[_S10] = (bitcast<u32>((v_1.x)));
        log_0[_S10 + u32(1)] = (bitcast<u32>((v_1.y)));
        log_0[_S10 + u32(2)] = (bitcast<u32>((v_1.z)));
        log_0[_S10 + u32(3)] = (bitcast<u32>((v_1.w)));
        j_0 = j_0 + u32(256);
    }
    workgroupBarrier();
    if(_S1)
    {
        log_0[i32(0)] = at_0 + size_1;
        log_0[i32(1)] = log_0[i32(1)] + u32(1);
    }
    return;
}

