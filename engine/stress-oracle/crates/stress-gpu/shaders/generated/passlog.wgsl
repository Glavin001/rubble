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

@binding(11) @group(0) var<storage, read> old_islands_0 : array<vec4<f32>>;

@binding(12) @group(0) var<storage, read> old_chunks_0 : array<vec4<f32>>;

@binding(4) @group(0) var<storage, read> children_0 : array<vec4<f32>>;

@binding(5) @group(0) var<storage, read> slots_0 : array<vec4<u32>>;

@binding(6) @group(0) var<storage, read> extras_0 : array<vec4<f32>>;

@binding(3) @group(0) var<storage, read> work_0 : array<u32>;

@binding(7) @group(0) var<storage, read> bodies_0 : array<vec4<f32>>;

@binding(8) @group(0) var<storage, read> colliders_0 : array<vec4<f32>>;

var<workgroup> g_group_uniform_0 : u32;

fn group_uniform_0( value_0 : u32,  thread_0 : u32) -> u32
{
    if(thread_0 == u32(0))
    {
        g_group_uniform_0 = value_0;
    }
    var _S1 : u32 = (workgroupUniformLoad(&((g_group_uniform_0))));
    return _S1;
}

struct LogEntry_0
{
     n_0 : u32,
     total_0 : u32,
     aborted_0 : bool,
     first_body_0 : u32,
     first_collider_0 : u32,
     nb_0 : u32,
     nc_0 : u32,
};

fn log_entry_0( n_1 : u32) -> LogEntry_0
{
    var e_0 : LogEntry_0;
    e_0.n_0 = n_1;
    e_0.total_0 = control_0[u32(6)];
    var _S2 : bool = control_0[u32(0)] != u32(0);
    e_0.aborted_0 = _S2;
    e_0.first_body_0 = control_0[u32(4)];
    e_0.first_collider_0 = control_0[u32(5)];
    var _S3 : u32;
    if(_S2)
    {
        _S3 = u32(0);
    }
    else
    {
        _S3 = control_0[u32(8)] - e_0.first_body_0;
    }
    e_0.nb_0 = _S3;
    if(e_0.aborted_0)
    {
        _S3 = u32(0);
    }
    else
    {
        _S3 = control_0[u32(9)] - e_0.first_collider_0;
    }
    e_0.nc_0 = _S3;
    return e_0;
}

var<workgroup> g_children_0 : u32;

var<workgroup> g_size_0 : u32;

var<workgroup> g_at_0 : u32;

fn log_place_0( e_1 : LogEntry_0,  tid_0 : u32,  mark_0 : bool)
{
    var _S4 : u32;
    if(tid_0 == u32(0))
    {
        var p_0 : u32 = u32(0);
        var c_0 : u32 = u32(0);
        loop
        {
            _S4 = e_1.n_0;
            if(p_0 < (e_1.n_0))
            {
            }
            else
            {
                break;
            }
            var c_1 : u32 = c_0 + parents_0[p_0].z;
            p_0 = p_0 + u32(1);
            c_0 = c_1;
        }
        var size_0 : u32 = u32(16) + u32(4) * _S4 + u32(14) * c_0 + e_1.total_0 + u32(28) * e_1.nb_0 + u32(16) * e_1.nc_0;
        var _S5 : u32 = max(log_0[i32(0)], u32(16));
        g_children_0 = c_0;
        if((_S5 + size_0) > (params_0.log_words_0))
        {
            if(mark_0)
            {
                log_0[i32(2)] = u32(1);
            }
            g_size_0 = u32(0);
        }
        else
        {
            g_size_0 = size_0;
        }
        g_at_0 = _S5;
    }
    workgroupBarrier();
    return;
}

fn log_head_0( e_2 : LogEntry_0,  at_0 : u32,  size_1 : u32,  c_2 : u32,  tid_1 : u32)
{
    var j_0 : u32;
    var p_1 : u32;
    if(tid_1 == u32(0))
    {
        log_0[at_0] = size_1;
        log_0[at_0 + u32(1)] = e_2.n_0;
        log_0[at_0 + u32(2)] = c_2;
        if(e_2.aborted_0)
        {
            j_0 = u32(1);
        }
        else
        {
            j_0 = u32(0);
        }
        log_0[at_0 + u32(3)] = j_0;
        log_0[at_0 + u32(4)] = status_0[i32(1)] - u32(1);
        log_0[at_0 + u32(5)] = e_2.first_body_0;
        log_0[at_0 + u32(6)] = e_2.first_collider_0;
        log_0[at_0 + u32(7)] = e_2.first_body_0 + e_2.nb_0;
        log_0[at_0 + u32(8)] = e_2.first_collider_0 + e_2.nc_0;
        log_0[at_0 + u32(9)] = control_0[u32(2)];
        log_0[at_0 + u32(10)] = control_0[u32(3)];
        log_0[at_0 + u32(11)] = e_2.total_0;
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
        var _S6 : u32 = at_0 + u32(16);
        var _S7 : u32 = _S6 + u32(4) * e_2.n_0;
        p_1 = u32(0);
        var h_0 : u32 = _S7;
        loop
        {
            if(p_1 < (e_2.n_0))
            {
            }
            else
            {
                break;
            }
            var parent_0 : vec4<u32> = parents_0[p_1];
            var _S8 : u32 = parent_0.x;
            var _S9 : u32 = u32(25) * _S8;
            var info_0 : vec4<u32> = (bitcast<vec4<u32>>((old_chunks_0[u32(16) * (bitcast<vec4<u32>>((old_islands_0[_S9]))).x + u32(8)])));
            var _S10 : u32 = _S6 + u32(4) * p_1;
            log_0[_S10] = _S8;
            var _S11 : u32 = parent_0.z;
            log_0[_S10 + u32(1)] = _S11;
            log_0[_S10 + u32(2)] = (bitcast<vec4<u32>>((old_islands_0[_S9 + u32(19)]))).w;
            log_0[_S10 + u32(3)] = info_0.z;
            var k_0 : u32 = u32(0);
            loop
            {
                if(k_0 < _S11)
                {
                }
                else
                {
                    break;
                }
                var slot_0 : u32 = parent_0.y + k_0;
                var head_0 : vec4<f32> = children_0[u32(8) * slot_0];
                var s_0 : vec4<u32>;
                if(e_2.aborted_0)
                {
                    s_0 = vec4<u32>(vec4<i32>(i32(0)));
                }
                else
                {
                    s_0 = slots_0[slot_0];
                }
                var _S12 : u32 = u32(2) * slot_0;
                var com_0 : vec4<f32> = extras_0[_S12];
                var rot_0 : vec4<f32> = extras_0[_S12 + u32(1)];
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
            p_1 = p_1 + u32(1);
        }
    }
    var _S13 : u32 = at_0 + u32(16) + u32(4) * e_2.n_0 + u32(14) * c_2;
    j_0 = tid_1;
    loop
    {
        if(j_0 < (e_2.total_0))
        {
        }
        else
        {
            break;
        }
        p_1 = u32(0);
        loop
        {
            var _S14 : u32 = p_1 + u32(1);
            var _S15 : bool;
            if(_S14 < (e_2.n_0))
            {
                _S15 = (parents_0[_S14].y) <= j_0;
            }
            else
            {
                _S15 = false;
            }
            if(_S15)
            {
            }
            else
            {
                break;
            }
            p_1 = _S14;
        }
        log_0[_S13 + j_0] = (bitcast<vec4<u32>>((old_chunks_0[u32(16) * ((bitcast<vec4<u32>>((old_islands_0[u32(25) * parents_0[p_1].x]))).x + work_0[e_2.total_0 + j_0]) + u32(8)]))).w;
        j_0 = j_0 + u32(256);
    }
    return;
}

fn log_engine_0( e_3 : LogEntry_0,  at_1 : u32,  size_2 : u32,  c_3 : u32,  tid_2 : u32)
{
    var _S16 : u32;
    var b_0 : u32 = at_1 + u32(16) + u32(4) * e_3.n_0 + u32(14) * c_3 + e_3.total_0;
    var j_1 : u32 = tid_2;
    loop
    {
        _S16 = e_3.nb_0;
        if(j_1 < (u32(7) * e_3.nb_0))
        {
        }
        else
        {
            break;
        }
        var v_0 : vec4<f32> = bodies_0[u32(7) * e_3.first_body_0 + j_1];
        var _S17 : u32 = b_0 + u32(4) * j_1;
        log_0[_S17] = (bitcast<u32>((v_0.x)));
        log_0[_S17 + u32(1)] = (bitcast<u32>((v_0.y)));
        log_0[_S17 + u32(2)] = (bitcast<u32>((v_0.z)));
        log_0[_S17 + u32(3)] = (bitcast<u32>((v_0.w)));
        j_1 = j_1 + u32(256);
    }
    var _S18 : u32 = b_0 + u32(28) * _S16;
    j_1 = tid_2;
    loop
    {
        if(j_1 < (u32(4) * e_3.nc_0))
        {
        }
        else
        {
            break;
        }
        var v_1 : vec4<f32> = colliders_0[u32(4) * e_3.first_collider_0 + j_1];
        var _S19 : u32 = _S18 + u32(4) * j_1;
        log_0[_S19] = (bitcast<u32>((v_1.x)));
        log_0[_S19 + u32(1)] = (bitcast<u32>((v_1.y)));
        log_0[_S19 + u32(2)] = (bitcast<u32>((v_1.z)));
        log_0[_S19 + u32(3)] = (bitcast<u32>((v_1.w)));
        j_1 = j_1 + u32(256);
    }
    workgroupBarrier();
    if(tid_2 == u32(0))
    {
        log_0[i32(0)] = at_1 + size_2;
        log_0[i32(1)] = log_0[i32(1)] + u32(1);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn pass_log(@builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var tid_3 : u32 = thread_1.x;
    var n_2 : u32 = group_uniform_0(control_0[u32(1)], tid_3);
    if(n_2 == u32(0))
    {
        return;
    }
    var e_4 : LogEntry_0 = log_entry_0(n_2);
    log_place_0(e_4, tid_3, true);
    var at_2 : u32 = g_at_0;
    var size_3 : u32 = group_uniform_0(g_size_0, tid_3);
    var c_4 : u32 = g_children_0;
    if(size_3 == u32(0))
    {
        return;
    }
    log_head_0(e_4, at_2, size_3, c_4, tid_3);
    log_engine_0(e_4, at_2, size_3, c_4, tid_3);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn pass_log_head(@builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var tid_4 : u32 = thread_2.x;
    var n_3 : u32 = group_uniform_0(control_0[u32(1)], tid_4);
    if(n_3 == u32(0))
    {
        return;
    }
    var e_5 : LogEntry_0 = log_entry_0(n_3);
    log_place_0(e_5, tid_4, true);
    var at_3 : u32 = g_at_0;
    var size_4 : u32 = group_uniform_0(g_size_0, tid_4);
    var c_5 : u32 = g_children_0;
    if(size_4 == u32(0))
    {
        return;
    }
    log_head_0(e_5, at_3, size_4, c_5, tid_4);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn pass_log_tail(@builtin(local_invocation_id) thread_3 : vec3<u32>)
{
    var tid_5 : u32 = thread_3.x;
    var n_4 : u32 = group_uniform_0(control_0[u32(1)], tid_5);
    if(n_4 == u32(0))
    {
        return;
    }
    var e_6 : LogEntry_0 = log_entry_0(n_4);
    log_place_0(e_6, tid_5, false);
    var at_4 : u32 = g_at_0;
    var size_5 : u32 = group_uniform_0(g_size_0, tid_5);
    var c_6 : u32 = g_children_0;
    if(size_5 == u32(0))
    {
        return;
    }
    log_engine_0(e_6, at_4, size_5, c_6, tid_5);
    return;
}

