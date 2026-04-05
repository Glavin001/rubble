use bytemuck::{Pod, Zeroable};
use std::f32::consts::PI;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

impl Vertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
        ],
    };
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

/// Unit sphere (radius 1) via icosphere subdivision.
pub fn icosphere(subdivisions: u32) -> Mesh {
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let mut positions: Vec<[f32; 3]> = vec![
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];
    for p in &mut positions {
        let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        p[0] /= len;
        p[1] /= len;
        p[2] /= len;
    }

    let mut indices: Vec<u32> = vec![
        0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7,
        1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9,
        8, 1,
    ];

    for _ in 0..subdivisions {
        let mut new_indices = Vec::with_capacity(indices.len() * 4);
        let mut midpoint_cache = std::collections::HashMap::new();
        for tri in indices.chunks(3) {
            let mut mids = [0u32; 3];
            for (i, &(a, b)) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
                .iter()
                .enumerate()
            {
                let key = if a < b { (a, b) } else { (b, a) };
                mids[i] = *midpoint_cache.entry(key).or_insert_with(|| {
                    let pa = positions[a as usize];
                    let pb = positions[b as usize];
                    let mut m = [
                        (pa[0] + pb[0]) * 0.5,
                        (pa[1] + pb[1]) * 0.5,
                        (pa[2] + pb[2]) * 0.5,
                    ];
                    let len = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
                    m[0] /= len;
                    m[1] /= len;
                    m[2] /= len;
                    let idx = positions.len() as u32;
                    positions.push(m);
                    idx
                });
            }
            let (a, b, c) = (tri[0], tri[1], tri[2]);
            new_indices.extend_from_slice(&[a, mids[0], mids[2]]);
            new_indices.extend_from_slice(&[b, mids[1], mids[0]]);
            new_indices.extend_from_slice(&[c, mids[2], mids[1]]);
            new_indices.extend_from_slice(&[mids[0], mids[1], mids[2]]);
        }
        indices = new_indices;
    }

    let vertices = positions
        .iter()
        .map(|p| Vertex {
            position: *p,
            normal: *p,
        })
        .collect();

    Mesh { vertices, indices }
}

/// Unit cube (half-extent 1) with per-face normals.
pub fn unit_cube() -> Mesh {
    let faces: &[([f32; 3], [[f32; 3]; 4])] = &[
        // +X
        (
            [1.0, 0.0, 0.0],
            [
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0],
            ],
        ),
        // -X
        (
            [-1.0, 0.0, 0.0],
            [
                [-1.0, -1.0, 1.0],
                [-1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
            ],
        ),
        // +Y
        (
            [0.0, 1.0, 0.0],
            [
                [-1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
        ),
        // -Y
        (
            [0.0, -1.0, 0.0],
            [
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
            ],
        ),
        // +Z
        (
            [0.0, 0.0, 1.0],
            [
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0],
            ],
        ),
        // -Z
        (
            [0.0, 0.0, -1.0],
            [
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, -1.0],
            ],
        ),
    ];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (normal, corners) in faces {
        let base = vertices.len() as u32;
        for &pos in corners {
            vertices.push(Vertex {
                position: pos,
                normal: *normal,
            });
        }
        indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
    }
    Mesh { vertices, indices }
}

/// Capsule aligned along Y with given half-height (cylinder) and radius.
/// Returns a unit capsule (half_height=1, radius=1) that gets scaled via instance transform.
pub fn unit_capsule(rings: u32, segments: u32) -> Mesh {
    assert!(rings >= 2, "capsule needs at least two hemisphere rings");
    assert!(
        segments >= 3,
        "capsule needs at least three radial segments"
    );

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let half_h = 1.0_f32;
    let radius = 1.0_f32;
    let mut ring_indices: Vec<Vec<u32>> = Vec::new();

    let top_pole = vertices.len() as u32;
    vertices.push(Vertex {
        position: [0.0, half_h + radius, 0.0],
        normal: [0.0, 1.0, 0.0],
    });

    let mut push_ring = |y: f32, ring_radius: f32, normal_y: f32| -> Vec<u32> {
        let mut ring = Vec::with_capacity(segments as usize);
        for seg in 0..segments {
            let theta = (seg as f32 / segments as f32) * 2.0 * PI;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();
            ring.push(vertices.len() as u32);
            vertices.push(Vertex {
                position: [ring_radius * cos_theta, y, ring_radius * sin_theta],
                normal: [ring_radius * cos_theta, normal_y, ring_radius * sin_theta],
            });
        }
        ring
    };

    for ring in 1..rings {
        let phi = (ring as f32 / rings as f32) * PI * 0.5;
        ring_indices.push(push_ring(
            phi.cos() * radius + half_h,
            phi.sin() * radius,
            phi.cos(),
        ));
    }

    ring_indices.push(push_ring(half_h, radius, 0.0));
    ring_indices.push(push_ring(-half_h, radius, 0.0));

    for ring in 1..rings {
        let phi = PI * 0.5 + (ring as f32 / rings as f32) * PI * 0.5;
        ring_indices.push(push_ring(
            phi.cos() * radius - half_h,
            phi.sin() * radius,
            phi.cos(),
        ));
    }

    let bottom_pole = vertices.len() as u32;
    vertices.push(Vertex {
        position: [0.0, -half_h - radius, 0.0],
        normal: [0.0, -1.0, 0.0],
    });

    let first_ring = ring_indices
        .first()
        .expect("capsule should have at least one ring");
    for seg in 0..segments as usize {
        let next = (seg + 1) % segments as usize;
        indices.extend_from_slice(&[top_pole, first_ring[next], first_ring[seg]]);
    }

    for rings in ring_indices.windows(2) {
        let upper = &rings[0];
        let lower = &rings[1];
        for seg in 0..segments as usize {
            let next = (seg + 1) % segments as usize;
            indices.extend_from_slice(&[upper[seg], upper[next], lower[seg]]);
            indices.extend_from_slice(&[upper[next], lower[next], lower[seg]]);
        }
    }

    let last_ring = ring_indices
        .last()
        .expect("capsule should have at least one ring");
    for seg in 0..segments as usize {
        let next = (seg + 1) % segments as usize;
        indices.extend_from_slice(&[bottom_pole, last_ring[seg], last_ring[next]]);
    }

    Mesh { vertices, indices }
}

/// 2D circle approximated as a triangle fan (flat, in XY plane, z=0).
pub fn circle_2d(segments: u32) -> Mesh {
    let mut vertices = vec![Vertex {
        position: [0.0, 0.0, 0.0],
        normal: [0.0, 0.0, 1.0],
    }];
    let mut indices = Vec::new();
    for i in 0..=segments {
        let theta = (i as f32 / segments as f32) * 2.0 * PI;
        vertices.push(Vertex {
            position: [theta.cos(), theta.sin(), 0.0],
            normal: [0.0, 0.0, 1.0],
        });
        if i > 0 {
            indices.push(0);
            indices.push(i);
            indices.push(i + 1);
        }
    }
    Mesh { vertices, indices }
}

/// 2D unit quad (half-extent 1) in XY plane.
pub fn quad_2d() -> Mesh {
    let vertices = vec![
        Vertex {
            position: [-1.0, -1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        },
        Vertex {
            position: [1.0, -1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        },
        Vertex {
            position: [1.0, 1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        },
        Vertex {
            position: [-1.0, 1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        },
    ];
    let indices = vec![0, 1, 2, 0, 2, 3];
    Mesh { vertices, indices }
}

#[cfg(test)]
mod tests {
    use super::{unit_capsule, unit_cube, Mesh};

    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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

    fn length_sq(v: [f32; 3]) -> f32 {
        dot(v, v)
    }

    fn triangle_winding_matches_normals(mesh: &Mesh) {
        for triangle in mesh.indices.chunks_exact(3) {
            let a = mesh.vertices[triangle[0] as usize].position;
            let b = mesh.vertices[triangle[1] as usize].position;
            let c = mesh.vertices[triangle[2] as usize].position;
            let na = mesh.vertices[triangle[0] as usize].normal;
            let nb = mesh.vertices[triangle[1] as usize].normal;
            let nc = mesh.vertices[triangle[2] as usize].normal;

            let triangle_normal = cross(sub(b, a), sub(c, a));
            let avg_normal = [
                na[0] + nb[0] + nc[0],
                na[1] + nb[1] + nc[1],
                na[2] + nb[2] + nc[2],
            ];

            assert!(
                dot(triangle_normal, avg_normal) > 0.0,
                "triangle winding should match outward face normal",
            );
        }
    }

    #[test]
    fn unit_cube_triangles_face_outward() {
        triangle_winding_matches_normals(&unit_cube());
    }

    #[test]
    fn unit_capsule_triangles_face_outward() {
        triangle_winding_matches_normals(&unit_capsule(8, 16));
    }

    #[test]
    fn unit_capsule_has_no_degenerate_triangles() {
        let capsule = unit_capsule(8, 16);

        assert!(
            !capsule.vertices.is_empty(),
            "capsule should produce vertices"
        );
        assert!(
            !capsule.indices.is_empty(),
            "capsule should produce indices"
        );

        for triangle in capsule.indices.chunks_exact(3) {
            let a = capsule.vertices[triangle[0] as usize].position;
            let b = capsule.vertices[triangle[1] as usize].position;
            let c = capsule.vertices[triangle[2] as usize].position;
            let triangle_normal = cross(sub(b, a), sub(c, a));

            assert!(
                length_sq(triangle_normal) > 1.0e-8,
                "capsule should not emit degenerate triangles",
            );
        }
    }
}
