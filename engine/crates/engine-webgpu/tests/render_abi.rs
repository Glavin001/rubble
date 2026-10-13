//! The demo renderer reads the same body buffer the solver writes. A shorter
//! WGSL `Body` packs instances at the wrong stride and the canvas fills with
//! one face instead of the scene.

use engine_schema::{BodyState, Collider};
use std::mem::size_of;

fn wgsl_struct(src: &str, name: &str) -> String {
    let marker = format!("struct {name} {{");
    let start = src
        .find(&marker)
        .unwrap_or_else(|| panic!("demo renderer is missing `struct {name}`"));
    let body = &src[start + marker.len()..];
    let end = body
        .find("};")
        .unwrap_or_else(|| panic!("`struct {name}` is not closed"));
    body[..end].to_string()
}

fn wgsl_type_layout(ty: &str) -> (u32, u32) {
    match ty {
        "f32" | "u32" | "i32" => (4, 4),
        "vec2f" | "vec2u" | "vec2i" => (8, 8),
        "vec3f" | "vec3u" | "vec3i" => (16, 12),
        "vec4f" | "vec4u" | "vec4i" => (16, 16),
        other => panic!("unsupported WGSL field type `{other}`"),
    }
}

/// WGSL struct size: each member is aligned, then the whole struct is rounded
/// up to its largest member alignment. `vec3` plus a scalar packs into 16 bytes.
fn wgsl_struct_size(body: &str) -> u32 {
    let mut offset = 0u32;
    let mut align = 4u32;
    for raw in body.split([',', '\n']) {
        let line = raw.split("//").next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        let ty = line
            .split(':')
            .nth(1)
            .unwrap_or_else(|| panic!("field `{line}` has no type"))
            .trim()
            .trim_end_matches(';');
        let (field_align, field_size) = wgsl_type_layout(ty);
        align = align.max(field_align);
        offset = offset.div_ceil(field_align) * field_align + field_size;
    }
    offset.div_ceil(align) * align
}

#[test]
fn demo_renderer_body_matches_solver_record() {
    let src = include_str!("../demo.html");
    let body = wgsl_struct(src, "Body");
    let collider = wgsl_struct(src, "Collider");
    assert_eq!(
        wgsl_struct_size(&body),
        size_of::<BodyState>() as u32,
        "demo.html Body is a different stride from the solver buffer"
    );
    assert_eq!(
        wgsl_struct_size(&collider),
        size_of::<Collider>() as u32,
        "demo.html Collider is a different stride from the solver buffer"
    );
}
