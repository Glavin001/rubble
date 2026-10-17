struct Params_std140_0
{
    @align(16) count_0 : u32,
    @align(4) scale_0 : f32,
};

@binding(0) @group(0) var<uniform> params_0 : Params_std140_0;
@binding(1) @group(0) var<storage, read> a_0 : array<vec4<f32>>;

@binding(2) @group(0) var<storage, read> b_0 : array<vec4<f32>>;

@binding(3) @group(0) var<storage, read_write> result_0 : array<vec4<f32>>;

@compute
@workgroup_size(64, 1, 1)
fn smoke(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var i_0 : u32 = id_0.x;
    if(i_0 >= (params_0.count_0))
    {
        return;
    }
    var c_0 : vec3<f32> = cross(a_0[i_0].xyz, b_0[i_0].xyz) * vec3<f32>(params_0.scale_0);
    result_0[i_0] = vec4<f32>(c_0, sqrt(dot(c_0, c_0)));
    return;
}

