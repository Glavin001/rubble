//! GPU skinning of render meshes bound to soft bodies (`slang/kernels/soft_skin.slang`).
//!
//! An integration binds its mesh once (`SoftHost::bind_mesh`, per part), then
//! encodes [`SoftSkinner::encode`] after each step. The output buffers feed any
//! renderer directly: positions (w = 1 for a torn vertex), smooth normals (w = the
//! largest crumple strain around the vertex) and per-triangle normal/strain
//! (w = −1 for a torn triangle). Nothing is read back to the CPU.

use crate::{WgpuSim, SOFT_SKIN_WGSL};
use engine_softbody::SkinBinding;
use glam::Vec3;
use wgpu::util::DeviceExt;

pub struct SoftSkinner {
    pipelines: [wgpu::ComputePipeline; 3],
    groups: [wgpu::BindGroup; 3],
    pub positions: wgpu::Buffer,
    pub normals: wgpu::Buffer,
    pub faces: wgpu::Buffer,
    vertices: u32,
    triangles: u32,
    params: wgpu::Buffer,
    /// Holds 1: copied into `SkinParams.skipSleeping` after the first encode.
    one: wgpu::Buffer,
    first: std::cell::Cell<bool>,
}

impl SoftSkinner {
    /// `bindings` from `SoftHost::bind_mesh`, `rest` the same vertices' rest
    /// positions (for crumple strain), `triangles` the mesh's faces.
    pub fn new(
        sim: &WgpuSim,
        bindings: &[SkinBinding],
        rest: &[Vec3],
        triangles: &[[u32; 3]],
    ) -> Self {
        let device = sim.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soft_skin"),
            source: wgpu::ShaderSource::Wgsl(SOFT_SKIN_WGSL.into()),
        });
        let pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipelines = [
            pipe("soft_skin_vertices"),
            pipe("soft_skin_faces"),
            pipe("soft_skin_normals"),
        ];
        let nv = bindings.len();
        let nt = triangles.len();
        let words: Vec<u32> = bindings.iter().flat_map(|b| b.to_words(0)).collect();
        let rest4: Vec<[f32; 4]> = rest.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let tris4: Vec<[u32; 4]> = triangles.iter().map(|t| [t[0], t[1], t[2], 0]).collect();
        // Vertex → triangles (CSR): starts, then the lists.
        let mut deg = vec![0u32; nv];
        for t in triangles {
            for &v in t {
                deg[v as usize] += 1;
            }
        }
        let mut adj = vec![0u32; nv + 1 + nt * 3];
        let mut acc = (nv + 1) as u32;
        for v in 0..nv {
            adj[v] = acc;
            acc += deg[v];
        }
        adj[nv] = acc;
        let mut fill: Vec<u32> = adj[..nv].to_vec();
        for (t, tri) in triangles.iter().enumerate() {
            for &v in tri {
                adj[fill[v as usize] as usize] = t as u32;
                fill[v as usize] += 1;
            }
        }
        let storage = wgpu::BufferUsages::STORAGE;
        let init = |label: &str, contents: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };
        let pad = |b: &[u8]| {
            if b.is_empty() {
                vec![0u8; 16]
            } else {
                b.to_vec()
            }
        };
        let bind = init("skin-bind", &pad(bytemuck::cast_slice(&words)), storage);
        let restb = init("skin-rest", &pad(bytemuck::cast_slice(&rest4)), storage);
        let trisb = init("skin-tris", &pad(bytemuck::cast_slice(&tris4)), storage);
        let adjb = init("skin-adj", &pad(bytemuck::cast_slice(&adj)), storage);
        let out = |label: &str, n: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n.max(1) * 16) as u64,
                usage: storage | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let positions = out("skin-positions", nv);
        let faces = out("skin-faces", nt);
        let normals = out("skin-normals", nv);
        let params = init(
            "skin-params",
            bytemuck::cast_slice(&[nv as u32, nt as u32, 1.8f32.to_bits(), 0]),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let asleep = init(
            "skin-asleep",
            &pad(bytemuck::cast_slice(&vec![0u32; nv.max(1)])),
            storage,
        );
        let one = init("skin-one", bytemuck::cast_slice(&[1u32]), wgpu::BufferUsages::COPY_SRC);
        let group = |p: &wgpu::ComputePipeline, list: &[(u32, &wgpu::Buffer)]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &p.get_bind_group_layout(0),
                entries: &list
                    .iter()
                    .map(|&(binding, b)| wgpu::BindGroupEntry {
                        binding,
                        resource: b.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        };
        let groups = [
            group(
                &pipelines[0],
                &[
                    (0, sim.body_buffer()),
                    (1, &bind),
                    (5, &positions),
                    (8, &params),
                    (9, &asleep),
                ],
            ),
            group(
                &pipelines[1],
                &[
                    (2, &restb),
                    (3, &trisb),
                    (5, &positions),
                    (6, &faces),
                    (8, &params),
                    (9, &asleep),
                ],
            ),
            group(
                &pipelines[2],
                &[(4, &adjb), (6, &faces), (7, &normals), (8, &params)],
            ),
        ];
        Self {
            pipelines,
            groups,
            positions,
            normals,
            faces,
            vertices: nv as u32,
            triangles: nt as u32,
            params,
            one,
            first: std::cell::Cell::new(true),
        }
    }

    /// Skin the mesh from the current body buffer (after a step).
    pub fn encode(&self, enc: &mut wgpu::CommandEncoder) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("soft_skin"),
            timestamp_writes: None,
        });
        for (i, n) in [self.vertices, self.triangles, self.vertices]
            .into_iter()
            .enumerate()
        {
            pass.set_pipeline(&self.pipelines[i]);
            pass.set_bind_group(0, &self.groups[i], &[]);
            pass.dispatch_workgroups(n.div_ceil(64).max(1), 1, 1);
        }
        drop(pass);
        // Every vertex skinned once: from now on sleeping ones keep their output.
        if self.first.replace(false) {
            enc.copy_buffer_to_buffer(&self.one, 0, &self.params, 12, 4);
        }
    }

    /// Skin now and read the result back (tests, CPU consumers): positions and
    /// normals, xyzw per vertex.
    pub fn skin_and_read(&self, sim: &WgpuSim) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
        let device = sim.device();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode(&mut enc);
        let size = (self.vertices.max(1) * 16) as u64;
        let staging = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let (sp, sn) = (staging("skin-read-p"), staging("skin-read-n"));
        enc.copy_buffer_to_buffer(&self.positions, 0, &sp, 0, size);
        enc.copy_buffer_to_buffer(&self.normals, 0, &sn, 0, size);
        sim.queue().submit([enc.finish()]);
        let read = |b: &wgpu::Buffer| {
            let slice = b.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            rx.recv().expect("map").expect("map ok");
            let v: Vec<[f32; 4]> =
                bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
            b.unmap();
            v
        };
        let p = read(&sp);
        let n = read(&sn);
        (
            p[..self.vertices as usize].to_vec(),
            n[..self.vertices as usize].to_vec(),
        )
    }
}
