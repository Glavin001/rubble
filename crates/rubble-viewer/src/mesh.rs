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
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    Mesh { vertices, indices }
}

/// Capsule aligned along Y with given half-height (cylinder) and radius.
/// Returns a unit capsule (half_height=1, radius=1) that gets scaled via instance transform.
pub fn unit_capsule(rings: u32, segments: u32) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let half_h = 1.0_f32;
    let radius = 1.0_f32;

    // Top hemisphere
    for ring in 0..=rings {
        let phi = (ring as f32 / rings as f32) * PI * 0.5;
        let y = phi.cos() * radius + half_h;
        let r = phi.sin() * radius;
        for seg in 0..=segments {
            let theta = (seg as f32 / segments as f32) * 2.0 * PI;
            let x = r * theta.cos();
            let z = r * theta.sin();
            let nx = phi.sin() * theta.cos();
            let ny = phi.cos();
            let nz = phi.sin() * theta.sin();
            vertices.push(Vertex {
                position: [x, y, z],
                normal: [nx, ny, nz],
            });
        }
    }

    // Cylinder body (two rings at +half_h and -half_h)
    for &cy in &[half_h, -half_h] {
        for seg in 0..=segments {
            let theta = (seg as f32 / segments as f32) * 2.0 * PI;
            let x = radius * theta.cos();
            let z = radius * theta.sin();
            vertices.push(Vertex {
                position: [x, cy, z],
                normal: [theta.cos(), 0.0, theta.sin()],
            });
        }
    }

    // Bottom hemisphere
    for ring in 0..=rings {
        let phi = (ring as f32 / rings as f32) * PI * 0.5 + PI * 0.5;
        let y = phi.cos() * radius - half_h;
        let r = phi.sin() * radius;
        for seg in 0..=segments {
            let theta = (seg as f32 / segments as f32) * 2.0 * PI;
            let x = r * theta.cos();
            let z = r * theta.sin();
            let nx = phi.sin() * theta.cos();
            let ny = phi.cos();
            let nz = phi.sin() * theta.sin();
            vertices.push(Vertex {
                position: [x, y, z],
                normal: [nx, ny, nz],
            });
        }
    }

    // Generate indices: connect successive rings
    let verts_per_ring = segments + 1;
    let total_rings = (rings + 1) + 2 + (rings + 1);
    for ring in 0..(total_rings - 1) {
        for seg in 0..segments {
            let a = ring * verts_per_ring + seg;
            let b = a + verts_per_ring;
            indices.push(a);
            indices.push(b);
            indices.push(a + 1);
            indices.push(a + 1);
            indices.push(b);
            indices.push(b + 1);
        }
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
