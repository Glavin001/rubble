//! Explicit beams on the GPU (`slang/kernels/explicit_beams.slang`): every vehicle
//! in one dispatch per frame, one 256-lane workgroup each, running the origin rebase
//! and all of the frame's fixed steps. The rules are
//! [`engine_softbody::explicit_beams`]'s (the CPU reference); the GPU differs only in
//! summation order, in the self-collision forces (accumulated in 1/32 N fixed point)
//! and in the turbulence noise (a per-node hash instead of the CPU sequence).
//!
//! Build it from spawned [`ExplicitBeams`] (their current state is uploaded), call
//! [`ExplicitBeamsGpu::frame`] once per frame with that frame's steps
//! ([`engine_softbody::explicit_beams::FrameClock`]), and read state back with
//! [`ExplicitBeamsGpu::positions`], [`ExplicitBeamsGpu::velocities`] and
//! [`ExplicitBeamsGpu::beams`].

use crate::{EXPLICIT_BEAMS_WGSL, EXPLICIT_BEAMS_WIDE_WGSL};
use engine_softbody::explicit_beams::ExplicitBeams;
use glam::Vec3;
use wgpu::util::DeviceExt;

/// A beam's state as read back: broken, and its rest length (m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuBeam {
    pub broken: bool,
    pub rest: f32,
    /// Spawn rest length (m).
    pub rest0: f32,
}

impl GpuBeam {
    /// Yielded, as [`engine_softbody::explicit_beams::Beam::bent`].
    pub fn bent(&self) -> bool {
        !self.broken && (self.rest - self.rest0).abs() > 1e-3 * self.rest0
    }
}

#[derive(Clone, Copy, Debug)]
struct Span {
    node_off: u32,
    nodes: u32,
    beam_off: u32,
    beams: u32,
}

pub struct ExplicitBeamsGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    params: wgpu::Buffer,
    nodes: wgpu::Buffer,
    beams: wgpu::Buffer,
    misc: wgpu::Buffer,
    spans: Vec<Span>,
    rest0: Vec<f32>,
    words: [u32; 12],
    timing: Option<FrameTiming>,
}

/// Timestamps around each frame's pass ([`ExplicitBeamsGpu::enable_frame_timing`]).
struct FrameTiming {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    capacity: u32,
    used: u32,
}

impl ExplicitBeamsGpu {
    /// Upload `vehicles` (their current state: positions, velocities, forces, beam
    /// state, throttles, origin).
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, vehicles: &[ExplicitBeams]) -> Self {
        let nv = vehicles.len();
        let total_nodes: usize = vehicles.iter().map(|v| v.nodes.len()).sum();
        let total_beams: usize = vehicles.iter().map(|v| v.beams.len()).sum();
        let total_tris: usize = vehicles.iter().map(|v| v.triangles.len()).sum();
        let node_info_base = nv * 4;
        let beam_base = node_info_base + total_nodes;
        let tri_base = beam_base + 3 * total_beams;
        let box_base = tri_base + total_tris;

        let mut vehicle_recs: Vec<[u32; 4]> = Vec::with_capacity(nv * 4);
        let mut node_info: Vec<[u32; 4]> = Vec::with_capacity(total_nodes);
        let mut beam_recs: Vec<[u32; 4]> = Vec::with_capacity(total_beams * 3);
        let mut tri_recs: Vec<[u32; 4]> = Vec::with_capacity(total_tris);
        let mut box_recs: Vec<[u32; 4]> = Vec::new();
        let mut lists: Vec<u32> = Vec::new();
        let mut nodes: Vec<[f32; 4]> = Vec::with_capacity(total_nodes * 3);
        let mut beams: Vec<[f32; 4]> = Vec::with_capacity(total_beams * 3);
        let mut groups: Vec<u32> = Vec::new();
        let mut misc: Vec<[u32; 4]> = Vec::with_capacity(nv + total_tris);
        let mut tri_misc: Vec<[u32; 4]> = Vec::with_capacity(total_tris);
        let mut spans = Vec::with_capacity(nv);
        let mut rest0 = Vec::with_capacity(total_beams);
        let (mut node_off, mut beam_off, mut tri_off, mut box_off) = (0u32, 0u32, 0u32, 0u32);
        let bits = |v: Vec3| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits(), 0];
        // Per beam, the lists positions of its end a's and end b's incidence entries.
        let mut slots: Vec<[u32; 2]> = vec![[u32::MAX; 2]; total_beams];
        for v in vehicles {
            let group_count = v
                .beams
                .iter()
                .map(|b| b.group.unsigned_abs() + 1)
                .max()
                .filter(|&g| g > 1)
                .unwrap_or(0);
            let group_off = groups.len() as u32;
            groups.extend(std::iter::repeat_n(u32::MAX, group_count as usize));
            // Incidence lists, then the candidates.
            for (i, inc) in v.incidence.iter().enumerate() {
                let n = &v.nodes[i];
                node_info.push([n.flags, lists.len() as u32, inc.len() as u32, 0]);
                for &j in inc {
                    let b = &v.beams[j as usize];
                    let is_b = (b.b == i as u32 && b.a != i as u32) as u32;
                    let slot = &mut slots[(beam_off + j) as usize];
                    // A beam from a node to itself has both entries in one list.
                    let end = if is_b == 1 || slot[0] != u32::MAX {
                        1
                    } else {
                        0
                    };
                    slot[end] = lists.len() as u32;
                    lists.push(((beam_off + j) << 1) | is_b);
                }
            }
            let cand_off = lists.len() as u32;
            lists.extend(v.candidates.iter().map(|&c| node_off + c));
            let flags = v.settings.fast_inv_sqrt as u32
                | (v.settings.drag as u32) << 1
                | (v.settings.turbulence as u32) << 2;
            vehicle_recs.push([
                node_off,
                v.nodes.len() as u32,
                beam_off,
                v.beams.len() as u32,
            ]);
            vehicle_recs.push([
                tri_off,
                v.triangles.len() as u32,
                box_off,
                v.boxes.len() as u32,
            ]);
            vehicle_recs.push([cand_off, v.candidates.len() as u32, group_off, group_count]);
            vehicle_recs.push([
                flags,
                v.ground.to_bits(),
                v.collision_range.to_bits(),
                grid_cell(v).to_bits(),
            ]);
            for n in &v.nodes {
                nodes.push([n.pos.x, n.pos.y, n.pos.z, n.mass]);
                nodes.push([n.vel.x, n.vel.y, n.vel.z, n.friction]);
                nodes.push([n.force.x, n.force.y, n.force.z, 0.0]);
            }
            for b in &v.beams {
                beam_recs.push([
                    node_off + b.a,
                    node_off + b.b,
                    b.kind.code() | b.ty.code() << 8,
                    b.group as u32,
                ]);
                beam_recs.push([
                    b.k.to_bits(),
                    b.c.to_bits(),
                    b.limit_k.to_bits(),
                    b.limit_c.to_bits(),
                ]);
                beam_recs.push([
                    b.short_bound.to_bits(),
                    b.long_bound.to_bits(),
                    b.plastic.to_bits(),
                    b.rest0.to_bits(),
                ]);
                beams.push([b.rest, b.tc, b.tt, b.gate]);
                beams.push([b.strength, b.force, b.broken as u32 as f32, 0.0]);
                beams.push([0.0; 4]);
                rest0.push(b.rest0);
            }
            for t in &v.triangles {
                tri_recs.push([
                    node_off + t.nodes[0],
                    node_off + t.nodes[1],
                    node_off + t.nodes[2],
                    0,
                ]);
                tri_misc.push([t.wait as u32, t.idle as u32, 0, 0]);
            }
            for bx in &v.boxes {
                box_recs.push(bits(bx.lo));
                box_recs.push(bits(bx.hi));
            }
            let mut m = bits(v.origin);
            m[3] = v.steps as u32;
            misc.push(m);
            spans.push(Span {
                node_off,
                nodes: v.nodes.len() as u32,
                beam_off,
                beams: v.beams.len() as u32,
            });
            node_off += v.nodes.len() as u32;
            beam_off += v.beams.len() as u32;
            tri_off += v.triangles.len() as u32;
            box_off += v.boxes.len() as u32;
        }
        misc.extend(tri_misc);
        debug_assert!(slots.iter().flatten().all(|&s| s != u32::MAX));
        let inc_base = beams.len() as u32;
        beams.extend(std::iter::repeat_n([0.0f32; 4], lists.len()));
        // The self-collision grid's sorted runs, at each vehicle's candidate offset.
        let grid_base = beams.len() as u32;
        beams.extend(std::iter::repeat_n([0.0f32; 4], lists.len()));
        let grid = vehicles.iter().any(|v| grid_cell(v) > 0.0) as u32;
        let slot_base = lists.len() as u32;
        lists.extend(slots.iter().flatten());
        let mut statics = vehicle_recs;
        statics.extend(node_info);
        statics.extend(beam_recs);
        statics.extend(tri_recs);
        statics.extend(box_recs);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("explicit_beams"),
            // Large vehicles get twice the lanes (more loads in flight per step).
            source: wgpu::ShaderSource::Wgsl(
                if vehicles.iter().any(|v| v.beams.len() > WIDE_BEAMS) {
                    EXPLICIT_BEAMS_WIDE_WGSL
                } else {
                    EXPLICIT_BEAMS_WGSL
                }
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("explicit_beams_frame"),
            layout: None,
            module: &module,
            entry_point: Some("explicit_beams_frame"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pad = |b: &[u8]| {
            if b.is_empty() {
                vec![0u8; 16]
            } else {
                b.to_vec()
            }
        };
        // Two parities of group slots.
        let group_total = groups.len() as u32;
        let groups: Vec<u32> = groups.iter().chain(groups.iter()).copied().collect();
        let init = |label: &str, contents: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &pad(contents),
                usage,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let state = storage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let words = [
            0,
            nv as u32,
            node_info_base as u32,
            beam_base as u32,
            tri_base as u32,
            box_base as u32,
            group_total,
            inc_base,
            slot_base,
            grid_base,
            grid,
            0,
        ];
        let params = init(
            "explicit-beams-params",
            bytemuck::cast_slice(&words),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let statics = init(
            "explicit-beams-static",
            bytemuck::cast_slice(&statics),
            storage,
        );
        let list_len = lists.len();
        let lists = init(
            "explicit-beams-lists",
            bytemuck::cast_slice(&lists),
            storage,
        );
        let nodes = init("explicit-beams-nodes", bytemuck::cast_slice(&nodes), state);
        let beams = init("explicit-beams-beams", bytemuck::cast_slice(&beams), state);
        let acc = init(
            "explicit-beams-acc",
            bytemuck::cast_slice(&vec![0i32; total_nodes * 3]),
            storage,
        );
        let groups = init(
            "explicit-beams-groups",
            bytemuck::cast_slice(&groups),
            storage,
        );
        let misc = init("explicit-beams-misc", bytemuck::cast_slice(&misc), state);
        let next = init(
            "explicit-beams-grid-next",
            bytemuck::cast_slice(&vec![0u32; list_len.max(1)]),
            storage,
        );
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("explicit-beams"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                (0, &params),
                (1, &statics),
                (2, &lists),
                (3, &nodes),
                (4, &beams),
                (5, &acc),
                (6, &groups),
                (7, &misc),
                (8, &next),
            ]
            .map(|(binding, b)| wgpu::BindGroupEntry {
                binding,
                resource: b.as_entire_binding(),
            }),
        });
        Self {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            group,
            params,
            nodes,
            beams,
            misc,
            spans,
            rest0,
            words,
            timing: None,
        }
    }

    /// Record the GPU time of each of the next `frames` frames (read with
    /// [`Self::frame_times_ms`]). False when the device lacks timestamp queries.
    pub fn enable_frame_timing(&mut self, frames: u32) -> bool {
        if !self
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return false;
        }
        let count = 2 * frames.max(1);
        self.timing = Some(FrameTiming {
            set: self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("explicit-beams-frame-timing"),
                ty: wgpu::QueryType::Timestamp,
                count,
            }),
            resolve: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("explicit-beams-frame-timing"),
                size: u64::from(count) * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            capacity: frames.max(1),
            used: 0,
        });
        true
    }

    /// GPU milliseconds of each timed frame since [`Self::enable_frame_timing`]
    /// (waits for them); timing restarts empty.
    pub fn frame_times_ms(&mut self) -> Vec<f64> {
        let Some(t) = &mut self.timing else {
            return Vec::new();
        };
        let used = std::mem::take(&mut t.used);
        if used == 0 {
            return Vec::new();
        }
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.resolve_query_set(&t.set, 0..2 * used, &t.resolve, 0);
        self.queue.submit([enc.finish()]);
        let period = f64::from(self.queue.get_timestamp_period()) * 1e-6;
        let raw = self.read(&self.timing.as_ref().unwrap().resolve);
        let ticks: &[u64] = bytemuck::cast_slice(&raw[..used as usize * 16]);
        ticks
            .chunks(2)
            .map(|p| p[1].saturating_sub(p[0]) as f64 * period)
            .collect()
    }

    /// Vehicles simulated.
    pub fn vehicle_count(&self) -> usize {
        self.spans.len()
    }

    /// Submit one frame: the origin rebase, then `steps` fixed steps, for every
    /// vehicle. Does not wait ([`Self::wait`]).
    pub fn frame(&mut self, steps: u32) {
        self.submit(steps);
    }

    fn submit(&mut self, steps: u32) {
        if self.spans.is_empty() {
            return;
        }
        self.words[0] = steps;
        self.queue
            .write_buffer(&self.params, 0, bytemuck::cast_slice(&self.words));
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let timestamps = self.timing.as_mut().and_then(|t| {
                (t.used < t.capacity).then(|| {
                    t.used += 1;
                    wgpu::ComputePassTimestampWrites {
                        query_set: &t.set,
                        beginning_of_pass_write_index: Some(2 * t.used - 2),
                        end_of_pass_write_index: Some(2 * t.used - 1),
                    }
                })
            });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("explicit_beams"),
                timestamp_writes: timestamps,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(self.spans.len() as u32, 1, 1);
        }
        self.queue.submit([enc.finish()]);
    }

    /// Submit `dispatches` dispatches of `steps` steps each in one compute pass (the
    /// same simulation as one frame of their product): measures what a dispatch per
    /// step costs. Does not wait.
    pub fn frame_split(&mut self, steps: u32, dispatches: u32) {
        if self.spans.is_empty() {
            return;
        }
        self.words[0] = steps;
        self.queue
            .write_buffer(&self.params, 0, bytemuck::cast_slice(&self.words));
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("explicit_beams_split"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            for _ in 0..dispatches {
                pass.dispatch_workgroups(self.spans.len() as u32, 1, 1);
            }
        }
        self.queue.submit([enc.finish()]);
    }

    /// Block until submitted frames have run.
    pub fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Set one node's velocity (vehicle index, node index within it).
    pub fn set_velocity(&self, vehicle: usize, node: u32, v: Vec3) {
        let s = self.spans[vehicle];
        let offset = ((s.node_off + node) as u64 * 3 + 1) * 16;
        self.queue
            .write_buffer(&self.nodes, offset, bytemuck::cast_slice(&v.to_array()));
    }

    fn read(&self, buffer: &wgpu::Buffer) -> Vec<u8> {
        let size = buffer.size();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-read"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map").expect("map ok");
        let out = slice.get_mapped_range().expect("mapped").to_vec();
        staging.unmap();
        out
    }

    fn node_field(&self, field: usize) -> Vec<Vec<Vec3>> {
        let raw = self.read(&self.nodes);
        let v: &[[f32; 4]] = bytemuck::cast_slice(&raw);
        self.spans
            .iter()
            .map(|s| {
                (0..s.nodes)
                    .map(|i| {
                        let r = v[(s.node_off + i) as usize * 3 + field];
                        Vec3::new(r[0], r[1], r[2])
                    })
                    .collect()
            })
            .collect()
    }

    /// Per vehicle, the origins nodes are relative to.
    pub fn origins(&self) -> Vec<Vec3> {
        let raw = self.read(&self.misc);
        let m: &[[u32; 4]] = bytemuck::cast_slice(&raw);
        (0..self.spans.len())
            .map(|i| {
                Vec3::new(
                    f32::from_bits(m[i][0]),
                    f32::from_bits(m[i][1]),
                    f32::from_bits(m[i][2]),
                )
            })
            .collect()
    }

    /// Per vehicle, world node positions.
    pub fn positions(&self) -> Vec<Vec<Vec3>> {
        let origins = self.origins();
        let mut rel = self.node_field(0);
        for (o, nodes) in origins.iter().zip(&mut rel) {
            for p in nodes.iter_mut() {
                *p += *o;
            }
        }
        rel
    }

    /// Per vehicle, node velocities.
    pub fn velocities(&self) -> Vec<Vec<Vec3>> {
        self.node_field(1)
    }

    /// Per vehicle, every beam's broken flag and rest length.
    pub fn beams(&self) -> Vec<Vec<GpuBeam>> {
        let raw = self.read(&self.beams);
        let v: &[[f32; 4]] = bytemuck::cast_slice(&raw);
        self.spans
            .iter()
            .map(|s| {
                (0..s.beams)
                    .map(|j| {
                        let g = (s.beam_off + j) as usize;
                        GpuBeam {
                            broken: v[g * 3 + 1][2] != 0.0,
                            rest: v[g * 3][0],
                            rest0: self.rest0[g],
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

/// Beams above which a vehicle runs on the wide (512-lane) module.
const WIDE_BEAMS: usize = 2048;

/// Contacters above which self-collision uses the grid (below, testing every pair
/// spreads better over the lanes).
const GRID_MIN_CANDIDATES: usize = 64;

/// The self-collision grid's cell size for a vehicle (0: all pairs): its median
/// collision triangle's box at spawn, padded by the contact range, so a typical
/// triangle covers at most 2 cells a side (a large one more, up to the kernel's
/// limit, past which it tests every contacter).
fn grid_cell(v: &ExplicitBeams) -> f32 {
    if v.candidates.len() <= GRID_MIN_CANDIDATES || v.triangles.is_empty() {
        return 0.0;
    }
    let mut extents: Vec<f32> = v
        .triangles
        .iter()
        .map(|t| {
            let p = t.nodes.map(|n| v.nodes[n as usize].pos);
            let span = p[0].max(p[1]).max(p[2]) - p[0].min(p[1]).min(p[2]);
            span.max_element()
        })
        .collect();
    extents.sort_by(f32::total_cmp);
    (extents[extents.len() / 2] + 2.0 * v.collision_range).max(0.05)
}

/// What a renderer needs besides a trace of `copies` vehicles of one `.truck` file,
/// their nodes concatenated in order (as a trace of [`ExplicitBeamsGpu::positions`]
/// writes them): `{"vehicles", "nodes", "triangles", "tyres", "boxes"}` with every
/// `cab` triangle (collision or not) and each tyre's node rings, offset per copy, and
/// the static boxes given (centre, half extents). `extra` is appended inside the
/// object (`,"key":…`).
pub fn render_scene_json(
    text: &str,
    nodes_per_vehicle: usize,
    copies: usize,
    boxes: &[(Vec3, Vec3)],
    extra: &str,
) -> String {
    let desc = engine_softbody::truck::load_truck(text);
    let file = engine_softbody::truck::parse_truck(text);
    let mut tris = Vec::new();
    let mut tyres = Vec::new();
    for k in 0..copies {
        let off = k * nodes_per_vehicle;
        tris.extend(desc.triangles.iter().map(|t| {
            format!(
                "[{},{},{}]",
                off + t.nodes[0] as usize,
                off + t.nodes[1] as usize,
                off + t.nodes[2] as usize
            )
        }));
        // Tyre nodes follow the truck's nodes, two rings of `rays` per wheel.
        let mut base = desc.nodes.len();
        for w in &file.wheels {
            tyres.push(format!(
                "{{\"axle\":[{},{}],\"first\":{},\"rays\":{}}}",
                off + w.node1 as usize,
                off + w.node2 as usize,
                off + base,
                w.rays
            ));
            base += 2 * w.rays as usize;
        }
    }
    let boxes: Vec<String> = boxes
        .iter()
        .map(|(c, h)| format!("[{},{},{},{},{},{}]", c.x, c.y, c.z, h.x, h.y, h.z))
        .collect();
    format!(
        "{{\"vehicles\":{copies},\"nodes\":{},\"triangles\":[{}],\"tyres\":[{}],\"boxes\":[{}]{extra}}}",
        nodes_per_vehicle * copies,
        tris.join(","),
        tyres.join(","),
        boxes.join(",")
    )
}
