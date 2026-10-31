struct LogParams_std140_0
{
    @align(16) chunk_count_0 : u32,
    @align(4) bond_count_0 : u32,
    @align(8) body_count_0 : u32,
    @align(4) step_0 : u32,
    @align(16) w_alive_0 : u32,
    @align(4) cap_bonds_0 : u32,
    @align(8) cap_chunks_0 : u32,
    @align(4) cap_frames_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : LogParams_std140_0;
@binding(1) @group(0) var<storage, read> work_0 : array<u32>;

@binding(3) @group(0) var<storage, read_write> track_0 : array<u32>;

@binding(4) @group(0) var<storage, read_write> log_0 : array<atomic<u32>>;

@binding(2) @group(0) var<storage, read> frames_0 : array<vec4<f32>>;

fn claim_0( s_0 : u32,  cap_0 : u32,  at_0 : ptr<function, u32>) -> bool
{
    var i_0 : u32 = atomicAdd(&(log_0[s_0]), u32(1));
    var acked_0 : u32 = atomicLoad(&(log_0[u32(4) + s_0]));
    var _S1 : u32 = i_0 % cap_0;
    (*at_0) = _S1;
    if((i_0 - acked_0) >= cap_0)
    {
        atomicStore(&(log_0[i32(3)]), u32(1));
        return false;
    }
    return true;
}

fn l_bonds_0() -> u32
{
    return u32(8);
}

@compute
@workgroup_size(64, 1, 1)
fn log_bonds(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var j_0 : u32 = id_0.x;
    if(j_0 >= (params_0.bond_count_0))
    {
        return;
    }
    var alive_0 : u32;
    if(work_0[params_0.w_alive_0 + j_0] != u32(0))
    {
        alive_0 = u32(1);
    }
    else
    {
        alive_0 = u32(0);
    }
    if(track_0[j_0] == alive_0)
    {
        return;
    }
    track_0[j_0] = alive_0;
    var at_1 : u32;
    var _S2 : bool;
    if(alive_0 == u32(0))
    {
        var _S3 : bool = claim_0(u32(0), params_0.cap_bonds_0, &(at_1));
        _S2 = _S3;
    }
    else
    {
        _S2 = false;
    }
    if(_S2)
    {
        var _S4 : u32 = l_bonds_0();
        atomicStore(&(log_0[_S4 + u32(2) * at_1]), j_0);
        atomicStore(&(log_0[_S4 + u32(2) * at_1 + u32(1)]), params_0.step_0);
    }
    return;
}

fn t_body_0() -> u32
{
    return params_0.bond_count_0;
}

fn l_chunks_0() -> u32
{
    return l_bonds_0() + u32(2) * params_0.cap_bonds_0;
}

@compute
@workgroup_size(64, 1, 1)
fn log_chunks(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var c_0 : u32 = id_1.x;
    if(c_0 >= (params_0.chunk_count_0))
    {
        return;
    }
    var body_0 : u32 = work_0[c_0];
    var _S5 : u32 = t_body_0() + c_0;
    if(track_0[_S5] == body_0)
    {
        return;
    }
    track_0[_S5] = body_0;
    var at_2 : u32;
    var _S6 : bool = claim_0(u32(1), params_0.cap_chunks_0, &(at_2));
    if(_S6)
    {
        var _S7 : u32 = l_chunks_0();
        atomicStore(&(log_0[_S7 + u32(3) * at_2]), c_0);
        atomicStore(&(log_0[_S7 + u32(3) * at_2 + u32(1)]), body_0);
        atomicStore(&(log_0[_S7 + u32(3) * at_2 + u32(2)]), params_0.step_0);
    }
    return;
}

fn t_frame_0() -> u32
{
    return params_0.bond_count_0 + params_0.chunk_count_0;
}

fn l_frames_0() -> u32
{
    return l_chunks_0() + u32(3) * params_0.cap_chunks_0;
}

@compute
@workgroup_size(64, 1, 1)
fn log_frames(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var b_0 : u32 = id_2.x;
    if(b_0 >= (params_0.body_count_0))
    {
        return;
    }
    var _S8 : u32 = u32(2) * b_0;
    var f0_0 : vec4<u32> = (bitcast<vec4<u32>>((frames_0[_S8])));
    var f1_0 : vec4<u32> = (bitcast<vec4<u32>>((frames_0[_S8 + u32(1)])));
    var t_0 : u32 = t_frame_0() + u32(8) * b_0;
    var same_0 : bool;
    if((all((vec4<u32>(track_0[t_0], track_0[t_0 + u32(1)], track_0[t_0 + u32(2)], track_0[t_0 + u32(3)]) == f0_0))))
    {
        same_0 = (all((vec4<u32>(track_0[t_0 + u32(4)], track_0[t_0 + u32(5)], track_0[t_0 + u32(6)], track_0[t_0 + u32(7)]) == f1_0)));
    }
    else
    {
        same_0 = false;
    }
    if(same_0)
    {
        return;
    }
    var w_0 : u32 = u32(0);
    loop
    {
        if(w_0 < u32(4))
        {
        }
        else
        {
            break;
        }
        track_0[t_0 + w_0] = f0_0[w_0];
        track_0[t_0 + u32(4) + w_0] = f1_0[w_0];
        w_0 = w_0 + u32(1);
    }
    var at_3 : u32;
    var _S9 : bool = claim_0(u32(2), params_0.cap_frames_0, &(at_3));
    if(_S9)
    {
        var e_0 : u32 = l_frames_0() + u32(10) * at_3;
        atomicStore(&(log_0[e_0]), b_0);
        atomicStore(&(log_0[e_0 + u32(1)]), params_0.step_0);
        w_0 = u32(0);
        loop
        {
            if(w_0 < u32(4))
            {
            }
            else
            {
                break;
            }
            atomicStore(&(log_0[e_0 + u32(2) + w_0]), f0_0[w_0]);
            atomicStore(&(log_0[e_0 + u32(6) + w_0]), f1_0[w_0]);
            w_0 = w_0 + u32(1);
        }
    }
    return;
}

