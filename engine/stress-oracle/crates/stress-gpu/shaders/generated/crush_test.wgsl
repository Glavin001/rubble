struct TestParams_std140_0
{
    @align(16) count_0 : u32,
    @align(4) pad0_0 : u32,
    @align(8) pad1_0 : u32,
    @align(4) pad2_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : TestParams_std140_0;
@binding(1) @group(0) var<storage, read> inputs_0 : array<vec4<f32>>;

@binding(2) @group(0) var<storage, read_write> outputs_0 : array<vec4<f32>>;

struct Sym_0
{
     d_0 : vec3<f32>,
     o_0 : vec3<f32>,
};

fn crush_invariants_0( v_0 : Sym_0,  volume_0 : f32) -> vec2<f32>
{
    var inv_0 : f32 = 1.0f / volume_0;
    var xx_0 : f32 = v_0.d_0.x * inv_0;
    var yy_0 : f32 = v_0.d_0.y * inv_0;
    var zz_0 : f32 = v_0.d_0.z * inv_0;
    var xy_0 : f32 = v_0.o_0.x * inv_0;
    var xz_0 : f32 = v_0.o_0.y * inv_0;
    var yz_0 : f32 = v_0.o_0.z * inv_0;
    var p_0 : f32 = - (xx_0 + yy_0 + zz_0) / 3.0f;
    var dx_0 : f32 = xx_0 + p_0;
    var dy_0 : f32 = yy_0 + p_0;
    var dz_0 : f32 = zz_0 + p_0;
    return vec2<f32>(p_0, sqrt(1.5f * max(dx_0 * dx_0 + dy_0 * dy_0 + dz_0 * dz_0 + 2.0f * (xy_0 * xy_0 + xz_0 * xz_0 + yz_0 * yz_0), 0.0f)));
}

fn crush_excess_0( pq_0 : vec2<f32>,  m0_0 : vec4<f32>) -> f32
{
    var _S1 : f32 = pq_0.x;
    if(!(_S1 > 0.0f))
    {
        return 0.0f;
    }
    var _S2 : f32 = max(pq_0.y - (m0_0.y + m0_0.z * _S1), _S1 - m0_0.x);
    var _S3 : f32;
    if(_S2 > 0.0f)
    {
        _S3 = _S2;
    }
    else
    {
        _S3 = 0.0f;
    }
    return _S3;
}

fn crush_step_0( v_1 : Sym_0,  volume_1 : f32,  dt_0 : f32,  m0_1 : vec4<f32>,  viscosity_0 : f32,  damage_0 : f32) -> vec4<f32>
{
    var pq_1 : vec2<f32> = crush_invariants_0(v_1, volume_1);
    var excess_0 : f32 = crush_excess_0(pq_1, m0_1);
    if(!(excess_0 > 0.0f))
    {
        return vec4<f32>(damage_0, pq_1, 0.0f);
    }
    var d_1 : f32 = damage_0 + excess_0 * excess_0 * dt_0 / (viscosity_0 * m0_1.w);
    if(d_1 >= 1.0f)
    {
        return vec4<f32>(1.0f, pq_1, 1.0f);
    }
    return vec4<f32>(d_1, pq_1, 0.0f);
}

@compute
@workgroup_size(64, 1, 1)
fn crush_step_test(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var i_0 : u32 = id_0.x;
    if(i_0 >= (params_0.count_0))
    {
        return;
    }
    var _S4 : u32 = u32(4) * i_0;
    var a_0 : vec4<f32> = inputs_0[_S4];
    var b_0 : vec4<f32> = inputs_0[_S4 + u32(1)];
    var m0_2 : vec4<f32> = inputs_0[_S4 + u32(2)];
    var c_0 : vec4<f32> = inputs_0[_S4 + u32(3)];
    var v_2 : Sym_0;
    v_2.d_0 = a_0.xyz;
    v_2.o_0 = b_0.xyz;
    outputs_0[i_0] = crush_step_0(v_2, a_0.w, b_0.w, m0_2, c_0.x, c_0.y);
    return;
}

