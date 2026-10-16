struct Hud_std140_0
{
    @align(16) rect_0 : vec4<f32>,
    @align(16) target_0 : vec4<f32>,
};

@binding(0) @group(0) var<uniform> hud_0 : Hud_std140_0;
@binding(1) @group(0) var<storage, read> pixels_0 : array<u32>;

struct VertexOut_0
{
    @builtin(position) position_0 : vec4<f32>,
};

@vertex
fn hud_vs(@builtin(vertex_index) vertex_0 : u32) -> VertexOut_0
{
    const _S1 : vec2<f32> = vec2<f32>(0.0f, 0.0f);
    const _S2 : vec2<f32> = vec2<f32>(1.0f, 1.0f);
    var quad_0 : array<vec2<f32>, i32(6)> = array<vec2<f32>, i32(6)>( _S1, vec2<f32>(1.0f, 0.0f), _S2, _S1, _S2, vec2<f32>(0.0f, 1.0f) );
    var p_0 : vec2<f32> = hud_0.rect_0.xy + quad_0[vertex_0] * hud_0.rect_0.zw;
    var o_0 : VertexOut_0;
    o_0.position_0 = vec4<f32>(2.0f * p_0.x / hud_0.target_0.x - 1.0f, 1.0f - 2.0f * p_0.y / hud_0.target_0.y, 0.0f, 1.0f);
    return o_0;
}

struct pixelOutput_0
{
    @location(0) output_0 : vec4<f32>,
};

@fragment
fn hud_fs(@builtin(position) position_1 : vec4<f32>) -> pixelOutput_0
{
    var _S3 : vec2<u32> = vec2<u32>(position_1.xy - hud_0.rect_0.xy);
    var w_0 : u32 = u32(hud_0.rect_0.z);
    var c_0 : u32 = pixels_0[min(_S3.y, u32(hud_0.rect_0.w) - u32(1)) * w_0 + min(_S3.x, w_0 - u32(1))];
    var _S4 : pixelOutput_0 = pixelOutput_0( vec4<f32>(f32((c_0 & (u32(255)))), f32((((c_0 >> (u32(8)))) & (u32(255)))), f32((((c_0 >> (u32(16)))) & (u32(255)))), f32((c_0 >> (u32(24))))) / vec4<f32>(255.0f) );
    return _S4;
}

