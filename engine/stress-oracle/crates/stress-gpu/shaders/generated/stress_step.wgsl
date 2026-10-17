struct StepParams_std140_0
{
    @align(16) chunk_count_0 : u32,
    @align(4) bond_count_0 : u32,
    @align(8) dt_0 : f32,
    @align(4) substeps_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : StepParams_std140_0;
struct Bond_std430_0
{
    @align(16) chunks_0 : vec4<u32>,
    @align(16) t1_0 : vec4<f32>,
    @align(16) t2_0 : vec4<f32>,
    @align(16) normal_0 : vec4<f32>,
    @align(16) ra_0 : vec4<f32>,
    @align(16) rb_0 : vec4<f32>,
    @align(16) k_lin_0 : vec4<f32>,
    @align(16) k_ang_0 : vec4<f32>,
    @align(16) c_lin_0 : vec4<f32>,
    @align(16) c_ang_0 : vec4<f32>,
    @align(16) section_0 : vec4<f32>,
};

@binding(1) @group(0) var<storage, read> bonds_0 : array<Bond_std430_0>;

@binding(5) @group(0) var<storage, read_write> state_0 : array<vec4<f32>>;

@binding(6) @group(0) var<storage, read_write> bond_loads_0 : array<vec4<f32>>;

@binding(3) @group(0) var<storage, read> chunk_bond_start_0 : array<u32>;

@binding(4) @group(0) var<storage, read> chunk_bonds_0 : array<u32>;

struct Chunk_std430_0
{
    @align(16) inv_inertia0_0 : vec4<f32>,
    @align(16) inv_inertia1_0 : vec4<f32>,
    @align(16) inv_inertia2_0 : vec4<f32>,
    @align(16) force_0 : vec4<f32>,
    @align(16) moment_0 : vec4<f32>,
    @align(16) support_0 : vec4<u32>,
};

@binding(2) @group(0) var<storage, read> chunks_1 : array<Chunk_std430_0>;

@binding(7) @group(0) var<storage, read> islands_0 : array<vec4<u32>>;

struct ChunkState_0
{
     u_0 : vec3<f32>,
     theta_0 : vec3<f32>,
     v_0 : vec3<f32>,
     w_0 : vec3<f32>,
};

fn load_state_0( c_0 : u32) -> ChunkState_0
{
    var s_0 : ChunkState_0;
    var _S1 : u32 = u32(4) * c_0;
    s_0.u_0 = state_0[_S1].xyz;
    s_0.theta_0 = state_0[_S1 + u32(1)].xyz;
    s_0.v_0 = state_0[_S1 + u32(2)].xyz;
    s_0.w_0 = state_0[_S1 + u32(3)].xyz;
    return s_0;
}

struct BondLoads_0
{
     fa_0 : vec4<f32>,
     ma_0 : vec3<f32>,
     mb_0 : vec3<f32>,
};

fn to_local_0( _S2 : u32,  _S3 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(dot(_S3, bonds_0[_S2].t1_0.xyz), dot(_S3, bonds_0[_S2].t2_0.xyz), dot(_S3, bonds_0[_S2].normal_0.xyz));
}

fn to_body_0( _S4 : u32,  _S5 : vec3<f32>) -> vec3<f32>
{
    return bonds_0[_S4].t1_0.xyz * vec3<f32>(_S5.x) + bonds_0[_S4].t2_0.xyz * vec3<f32>(_S5.y) + bonds_0[_S4].normal_0.xyz * vec3<f32>(_S5.z);
}

fn bond_math_0( _S6 : u32,  _S7 : ChunkState_0,  _S8 : ChunkState_0) -> BondLoads_0
{
    var ra_1 : vec3<f32> = bonds_0[_S6].ra_0.xyz;
    var rb_1 : vec3<f32> = bonds_0[_S6].rb_0.xyz;
    var _S9 : vec3<f32> = bonds_0[_S6].k_lin_0.xyz * to_local_0(_S6, _S8.u_0 + cross(_S8.theta_0, rb_1) - (_S7.u_0 + cross(_S7.theta_0, ra_1)));
    var _S10 : vec3<f32> = bonds_0[_S6].k_ang_0.xyz * to_local_0(_S6, _S8.theta_0 - _S7.theta_0);
    var _S11 : vec3<f32> = to_body_0(_S6, _S9 + bonds_0[_S6].c_lin_0.xyz * to_local_0(_S6, _S8.v_0 + cross(_S8.w_0, rb_1) - (_S7.v_0 + cross(_S7.w_0, ra_1))));
    var _S12 : vec3<f32> = to_body_0(_S6, _S10 + bonds_0[_S6].c_ang_0.xyz * to_local_0(_S6, _S8.w_0 - _S7.w_0));
    var o_0 : BondLoads_0;
    o_0.fa_0 = vec4<f32>(_S11, abs(_S9.z) / bonds_0[_S6].section_0.x + abs(_S10.x) / bonds_0[_S6].section_0.y + abs(_S10.y) / bonds_0[_S6].section_0.z);
    o_0.ma_0 = _S12 + cross(ra_1, _S11);
    o_0.mb_0 = (vec3<f32>(0) - _S12) + cross(rb_1, (vec3<f32>(0) - _S11));
    return o_0;
}

fn bond_phase_0( i_0 : u32)
{
    var _S13 : BondLoads_0 = bond_math_0(i_0, load_state_0(bonds_0[i_0].chunks_0.x), load_state_0(bonds_0[i_0].chunks_0.y));
    var _S14 : u32 = u32(3) * i_0;
    bond_loads_0[_S14] = _S13.fa_0;
    bond_loads_0[_S14 + u32(1)] = vec4<f32>(_S13.ma_0, 0.0f);
    bond_loads_0[_S14 + u32(2)] = vec4<f32>(_S13.mb_0, 0.0f);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn bond_forces(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S15 : u32 = id_0.x;
    if(_S15 < (params_0.bond_count_0))
    {
        bond_phase_0(_S15);
    }
    return;
}

fn chunk_math_0( ch_0 : ptr<function, Chunk_std430_0>,  st_0 : ChunkState_0,  fi_0 : vec3<f32>,  mi_0 : vec3<f32>,  dt_1 : f32) -> ChunkState_0
{
    var _S16 : ChunkState_0 = st_0;
    var _S17 : vec4<f32> = (*ch_0).force_0;
    var f_0 : vec3<f32> = (*ch_0).force_0.xyz + fi_0;
    var m_0 : vec3<f32> = (*ch_0).moment_0.xyz + mi_0;
    var alpha_0 : vec3<f32> = vec3<f32>(dot((*ch_0).inv_inertia0_0.xyz, m_0), dot((*ch_0).inv_inertia1_0.xyz, m_0), dot((*ch_0).inv_inertia2_0.xyz, m_0)) * vec3<f32>((*ch_0).moment_0.w);
    var support_1 : u32 = (*ch_0).support_0.x;
    if(support_1 == u32(1))
    {
        var _S18 : vec3<f32> = vec3<f32>(0.0f);
        _S16.v_0 = _S18;
        _S16.w_0 = _S18;
    }
    else
    {
        var _S19 : vec3<f32> = vec3<f32>(dt_1);
        var _S20 : vec3<f32> = _S16.w_0 + alpha_0 * _S19;
        _S16.w_0 = _S20;
        _S16.theta_0 = _S16.theta_0 + _S20 * _S19;
        if(support_1 == u32(2))
        {
            _S16.v_0 = vec3<f32>(0.0f);
        }
        else
        {
            var _S21 : vec3<f32> = _S16.v_0 + f_0 * vec3<f32>((dt_1 * _S17.w));
            _S16.v_0 = _S21;
            _S16.u_0 = _S16.u_0 + _S21 * _S19;
        }
    }
    return _S16;
}

fn chunk_math_1( ch_1 : ptr<function, Chunk_std430_0>,  st_1 : ChunkState_0,  fi_1 : vec3<f32>,  mi_1 : vec3<f32>,  dt_2 : f32) -> ChunkState_0
{
    var _S22 : ChunkState_0 = st_1;
    var _S23 : vec4<f32> = (*ch_1).force_0;
    var f_1 : vec3<f32> = (*ch_1).force_0.xyz + fi_1;
    var m_1 : vec3<f32> = (*ch_1).moment_0.xyz + mi_1;
    var alpha_1 : vec3<f32> = vec3<f32>(dot((*ch_1).inv_inertia0_0.xyz, m_1), dot((*ch_1).inv_inertia1_0.xyz, m_1), dot((*ch_1).inv_inertia2_0.xyz, m_1)) * vec3<f32>((*ch_1).moment_0.w);
    var support_2 : u32 = (*ch_1).support_0.x;
    if(support_2 == u32(1))
    {
        var _S24 : vec3<f32> = vec3<f32>(0.0f);
        _S22.v_0 = _S24;
        _S22.w_0 = _S24;
    }
    else
    {
        var _S25 : vec3<f32> = vec3<f32>(dt_2);
        var _S26 : vec3<f32> = _S22.w_0 + alpha_1 * _S25;
        _S22.w_0 = _S26;
        _S22.theta_0 = _S22.theta_0 + _S26 * _S25;
        if(support_2 == u32(2))
        {
            _S22.v_0 = vec3<f32>(0.0f);
        }
        else
        {
            var _S27 : vec3<f32> = _S22.v_0 + f_1 * vec3<f32>((dt_2 * _S23.w));
            _S22.v_0 = _S27;
            _S22.u_0 = _S22.u_0 + _S27 * _S25;
        }
    }
    return _S22;
}

fn store_state_0( c_1 : u32,  s_1 : ChunkState_0,  stress_0 : f32)
{
    var _S28 : u32 = u32(4) * c_1;
    state_0[_S28] = vec4<f32>(s_1.u_0, stress_0);
    state_0[_S28 + u32(1)] = vec4<f32>(s_1.theta_0, 0.0f);
    state_0[_S28 + u32(2)] = vec4<f32>(s_1.v_0, 0.0f);
    state_0[_S28 + u32(3)] = vec4<f32>(s_1.w_0, 0.0f);
    return;
}

fn chunk_phase_0( c_2 : u32,  dt_3 : f32)
{
    var _S29 : vec3<f32> = vec3<f32>(0.0f);
    var _S30 : u32 = chunk_bond_start_0[c_2];
    var stress_1 : f32 = 0.0f;
    var k_0 : u32 = _S30;
    var fi_2 : vec3<f32> = _S29;
    var mi_2 : vec3<f32> = _S29;
    loop
    {
        if(k_0 < chunk_bond_start_0[c_2 + u32(1)])
        {
        }
        else
        {
            break;
        }
        var e_0 : u32 = chunk_bonds_0[k_0];
        var _S31 : u32 = u32(3) * ((e_0 >> (u32(1))));
        var fa_1 : vec4<f32> = bond_loads_0[_S31];
        if(((e_0 & (u32(1)))) == u32(0))
        {
            var mi_3 : vec3<f32> = mi_2 + bond_loads_0[_S31 + u32(1)].xyz;
            fi_2 = fi_2 + fa_1.xyz;
            mi_2 = mi_3;
        }
        else
        {
            var mi_4 : vec3<f32> = mi_2 + bond_loads_0[_S31 + u32(2)].xyz;
            fi_2 = fi_2 + (vec3<f32>(0) - fa_1.xyz);
            mi_2 = mi_4;
        }
        var _S32 : f32 = max(stress_1, fa_1.w);
        var _S33 : u32 = k_0 + u32(1);
        stress_1 = _S32;
        k_0 = _S33;
    }
    var _S34 : Chunk_std430_0 = chunks_1[c_2];
    var _S35 : ChunkState_0 = chunk_math_0(&(_S34), load_state_0(c_2), fi_2, mi_2, dt_3);
    store_state_0(c_2, _S35, stress_1);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn chunk_integrate(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var _S36 : u32 = id_1.x;
    if(_S36 < (params_0.chunk_count_0))
    {
        chunk_phase_0(_S36, params_0.dt_0);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn island_substeps(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var _S37 : vec4<u32> = islands_0[group_0.x];
    var _S38 : f32 = params_0.dt_0;
    var s_2 : u32 = u32(0);
    loop
    {
        if(s_2 < (params_0.substeps_0))
        {
        }
        else
        {
            break;
        }
        var _S39 : u32 = thread_0.x;
        var i_1 : u32 = _S37.z + _S39;
        loop
        {
            if(i_1 < (_S37.w))
            {
            }
            else
            {
                break;
            }
            bond_phase_0(i_1);
            i_1 = i_1 + u32(256);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        var c_3 : u32 = _S37.x + _S39;
        loop
        {
            if(c_3 < (_S37.y))
            {
            }
            else
            {
                break;
            }
            chunk_phase_0(c_3, _S38);
            c_3 = c_3 + u32(256);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        s_2 = s_2 + u32(1);
    }
    return;
}

var<workgroup> g_u_0 : array<vec3<f32>, i32(160)>;

var<workgroup> g_theta_0 : array<vec3<f32>, i32(160)>;

var<workgroup> g_v_0 : array<vec3<f32>, i32(160)>;

var<workgroup> g_w_0 : array<vec3<f32>, i32(160)>;

fn shared_state_0( local_0 : u32) -> ChunkState_0
{
    var s_3 : ChunkState_0;
    s_3.u_0 = g_u_0[local_0];
    s_3.theta_0 = g_theta_0[local_0];
    s_3.v_0 = g_v_0[local_0];
    s_3.w_0 = g_w_0[local_0];
    return s_3;
}

var<workgroup> g_fa_0 : array<vec4<f32>, i32(400)>;

var<workgroup> g_ma_0 : array<vec3<f32>, i32(400)>;

var<workgroup> g_mb_0 : array<vec3<f32>, i32(400)>;

@compute
@workgroup_size(256, 1, 1)
fn island_shared_substeps(@builtin(workgroup_id) group_1 : vec3<u32>, @builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var island_0 : vec4<u32> = islands_0[group_1.x];
    var chunk0_0 : u32 = island_0.x;
    var chunk_count_1 : u32 = island_0.y - chunk0_0;
    var _S40 : u32 = island_0.z;
    var _S41 : u32 = island_0.w - _S40;
    var _S42 : f32 = params_0.dt_0;
    var _S43 : u32 = thread_1.x;
    var c_4 : u32 = _S43;
    loop
    {
        if(c_4 < chunk_count_1)
        {
        }
        else
        {
            break;
        }
        var s_4 : ChunkState_0 = load_state_0(chunk0_0 + c_4);
        g_u_0[c_4] = s_4.u_0;
        g_theta_0[c_4] = s_4.theta_0;
        g_v_0[c_4] = s_4.v_0;
        g_w_0[c_4] = s_4.w_0;
        c_4 = c_4 + u32(256);
    }
    workgroupBarrier();
    var peak_stress_0 : f32 = 0.0f;
    var s_5 : u32 = u32(0);
    loop
    {
        if(s_5 < (params_0.substeps_0))
        {
        }
        else
        {
            break;
        }
        var i_2 : u32 = _S43;
        loop
        {
            if(i_2 < _S41)
            {
            }
            else
            {
                break;
            }
            var _S44 : u32 = _S40 + i_2;
            var _S45 : BondLoads_0 = bond_math_0(_S44, shared_state_0(bonds_0[_S44].chunks_0.x - chunk0_0), shared_state_0(bonds_0[_S44].chunks_0.y - chunk0_0));
            g_fa_0[i_2] = _S45.fa_0;
            g_ma_0[i_2] = _S45.ma_0;
            g_mb_0[i_2] = _S45.mb_0;
            i_2 = i_2 + u32(256);
        }
        workgroupBarrier();
        if(_S43 < chunk_count_1)
        {
            var _S46 : vec3<f32> = vec3<f32>(0.0f);
            var _S47 : u32 = chunk0_0 + _S43;
            var _S48 : u32 = chunk_bond_start_0[_S47];
            var peak_0 : f32 = 0.0f;
            var k_1 : u32 = _S48;
            var fi_3 : vec3<f32> = _S46;
            var mi_5 : vec3<f32> = _S46;
            loop
            {
                if(k_1 < chunk_bond_start_0[_S47 + u32(1)])
                {
                }
                else
                {
                    break;
                }
                var e_1 : u32 = chunk_bonds_0[k_1];
                var bond_0 : u32 = ((e_1 >> (u32(1)))) - _S40;
                var fa_2 : vec4<f32> = g_fa_0[bond_0];
                if(((e_1 & (u32(1)))) == u32(0))
                {
                    var mi_6 : vec3<f32> = mi_5 + g_ma_0[bond_0];
                    fi_3 = fi_3 + fa_2.xyz;
                    mi_5 = mi_6;
                }
                else
                {
                    var mi_7 : vec3<f32> = mi_5 + g_mb_0[bond_0];
                    fi_3 = fi_3 + (vec3<f32>(0) - fa_2.xyz);
                    mi_5 = mi_7;
                }
                var _S49 : f32 = max(peak_0, fa_2.w);
                var _S50 : u32 = k_1 + u32(1);
                peak_0 = _S49;
                k_1 = _S50;
            }
            var _S51 : Chunk_std430_0 = chunks_1[_S47];
            var _S52 : ChunkState_0 = chunk_math_1(&(_S51), shared_state_0(_S43), fi_3, mi_5, _S42);
            g_u_0[_S43] = _S52.u_0;
            g_theta_0[_S43] = _S52.theta_0;
            g_v_0[_S43] = _S52.v_0;
            g_w_0[_S43] = _S52.w_0;
            peak_stress_0 = peak_0;
        }
        workgroupBarrier();
        s_5 = s_5 + u32(1);
    }
    if(_S43 < chunk_count_1)
    {
        var _S53 : u32 = chunk0_0 + _S43;
        var _S54 : ChunkState_0 = shared_state_0(_S43);
        if((params_0.substeps_0) > u32(0))
        {
        }
        else
        {
            peak_stress_0 = state_0[u32(4) * _S53].w;
        }
        store_state_0(_S53, _S54, peak_stress_0);
    }
    return;
}

