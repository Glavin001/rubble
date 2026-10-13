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

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Unit-cube corners in (u, v), then the two triangles. Negative faces must
/// reverse the order: the same CCW-(u,v) walk is clockwise when viewed from
/// outside a negative-axis face, and backface culling drops it.
fn cube_triangle(sign: f32, tri: usize, corner: usize) -> [f32; 2] {
    let winding = if sign > 0.0 {
        [[0, 1, 2], [0, 2, 3]]
    } else {
        [[0, 2, 1], [0, 3, 2]]
    };
    let uv = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    let [a, b] = uv[winding[tri][corner]];
    [a, b]
}

#[test]
fn negative_cube_faces_point_outward() {
    let src = include_str!("../demo.html");
    assert!(
        src.contains("sign > 0 ? [[0, 1, 2], [0, 2, 3]] : [[0, 2, 1], [0, 3, 2]]"),
        "demo cube keeps one winding for both face signs, so backface culling hides half the faces"
    );
    let axes = [[0, 1, 2], [1, 2, 0], [2, 0, 1]];
    for axis in axes {
        for &sign in &[-1.0f32, 1.0] {
            let mut n = [0.0, 0.0, 0.0];
            n[axis[2]] = sign;
            let corner = |i: usize| {
                let [a, b] = cube_triangle(sign, 0, i);
                let mut p = [0.0, 0.0, 0.0];
                p[axis[0]] = a * 0.5;
                p[axis[1]] = b * 0.5;
                p[axis[2]] = sign * 0.5;
                p
            };
            let p0 = corner(0);
            let p1 = corner(1);
            let p2 = corner(2);
            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let geom = cross(e1, e2);
            assert!(
                dot(geom, n) > 0.0,
                "face axis {:?} sign {sign} winds inward: geom {geom:?} stored {n:?}",
                axis
            );
        }
    }
}
