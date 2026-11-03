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

use crate::{
    EXPLICIT_BEAMS_AERO_WGSL, EXPLICIT_BEAMS_AERO_WIDE_WGSL, EXPLICIT_BEAMS_WGSL,
    EXPLICIT_BEAMS_WIDE_PASSES, EXPLICIT_BEAMS_WIDE_WGSL,
};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitKind};
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
    words: [u32; PARAM_WORDS],
    /// gBeams record of the aero state and per vehicle its first wing and count.
    aero_state: u32,
    wing_spans: Vec<(u32, u32)>,
    /// The wide passes (one thread per element over every vehicle), when used.
    wide: Option<Wide>,
    total: [u32; 5],
    timing: Option<FrameTiming>,
}

/// The wide passes' pipelines and bind group ([`ExplicitBeamsGpu::set_wide`]).
struct Wide {
    passes: Vec<wgpu::ComputePipeline>,
    group: wgpu::BindGroup,
    enabled: bool,
    /// The kernel's vehicle table (constant memory): static records, then per
    /// vehicle (origin, steps) copied from the misc buffer each frame.
    table: wgpu::Buffer,
}

/// Wide pass entry points, in [`EXPLICIT_BEAMS_WIDE_PASSES`] order.
const WIDE_BEGIN: usize = 0;
const WIDE_REBASE: usize = 1;
const WIDE_NODES: usize = 2;
const WIDE_BEAMS_PASS: usize = 3;
const WIDE_TRIS: usize = 4;
const WIDE_END: usize = 5;
const WIDE_CLOSE: usize = 6;

/// Parameter copies for the wide passes: one per step of a frame (its index in the
/// step word) plus the frame's end, at the uniform offset alignment.
const PARAM_STRIDE: u64 = 256;
const PARAM_COPIES: u32 = 129;
/// Words of the kernel's Params.
const PARAM_WORDS: usize = 31;
/// The kernel's WIDE_MAX_VEHICLES: more vehicles run one workgroup each.
const WIDE_MAX_VEHICLES: usize = 256;

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
        // Beam records carry the vehicle index in 16 bits.
        assert!(nv <= 1 << 16, "explicit beams GPU: at most 65536 vehicles, got {nv}");
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
        let mut beams: Vec<[f32; 4]> = Vec::with_capacity(total_beams * 2);
        let mut groups: Vec<u32> = Vec::new();
        let mut misc: Vec<[u32; 4]> = Vec::with_capacity(nv + total_tris);
        let mut tri_misc: Vec<[u32; 4]> = Vec::with_capacity(total_tris);
        let mut spans = Vec::with_capacity(nv);
        let mut rest0 = Vec::with_capacity(total_beams);
        let (mut node_off, mut beam_off, mut tri_off, mut box_off) = (0u32, 0u32, 0u32, 0u32);
        // shocks2 rates (EB_AERO modules), 2 records per shock, global indices.
        let mut shock_recs: Vec<[u32; 4]> = Vec::new();
        let bits = |v: Vec3| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits(), 0];
        // Per beam, the lists positions of its end a's and end b's incidence entries.
        let mut slots: Vec<[u32; 2]> = vec![[u32::MAX; 2]; total_beams];
        let mut cand_table: Vec<u32> = Vec::new();
        for (vi, v) in vehicles.iter().enumerate() {
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
                node_info.push([n.flags, lists.len() as u32, inc.len() as u32, vi as u32]);
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
            cand_table.extend(cand_off..cand_off + v.candidates.len() as u32);
            lists.extend(v.candidates.iter().map(|&c| node_off + c));
            let flags = v.settings.fast_inv_sqrt as u32
                | (v.settings.drag as u32) << 1
                | (v.settings.turbulence as u32) << 2
                | (v.fusedrag.is_some() as u32) << 3;
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
            let shock_off = (shock_recs.len() / 2) as u32;
            for sh in &v.shocks2 {
                shock_recs.push([
                    sh.k_out.to_bits(),
                    sh.c_out.to_bits(),
                    sh.pk_in.to_bits(),
                    sh.pc_in.to_bits(),
                ]);
                shock_recs.push([
                    sh.pk_out.to_bits(),
                    sh.pc_out.to_bits(),
                    (sh.soft as u32 as f32).to_bits(),
                    0,
                ]);
            }
            for b in &v.beams {
                beam_recs.push([
                    node_off + b.a,
                    node_off + b.b,
                    b.kind.code() | b.ty.code() << 8 | (vi as u32) << 16,
                    b.group as u32,
                ]);
                beam_recs.push([
                    b.k.to_bits(),
                    b.c.to_bits(),
                    b.limit_k.to_bits(),
                    b.limit_c.to_bits(),
                ]);
                // A shocks2 beam has no plastic coefficient: its shock's index.
                let third = if b.kind == ExplicitKind::Shock2 {
                    shock_off + b.shock
                } else {
                    b.plastic.to_bits()
                };
                beam_recs.push([
                    b.short_bound.to_bits(),
                    b.long_bound.to_bits(),
                    third,
                    b.rest0.to_bits(),
                ]);
                beams.push([b.rest, b.tc, b.tt, b.gate]);
                beams.push([b.strength, b.force, b.broken as u32 as f32, 0.0]);
                rest0.push(b.rest0);
            }
            for t in &v.triangles {
                tri_recs.push([
                    node_off + t.nodes[0],
                    node_off + t.nodes[1],
                    node_off + t.nodes[2],
                    vi as u32,
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
        // Wings, the fuselage drag, self-centering commands and shocks2 (the EB_AERO
        // modules).
        let aero = vehicles.iter().any(|v| {
            !v.wings.is_empty() || v.fusedrag.is_some() || !v.centering.is_empty() || !v.shocks2.is_empty()
        });
        let mut aero_recs: Vec<[u32; 4]> = Vec::new();
        let mut wing_recs: Vec<[u32; 4]> = Vec::new();
        let mut cmd_recs: Vec<[u32; 4]> = Vec::new();
        let mut table_recs_aero: Vec<[u32; 4]> = Vec::new();
        let mut wing_state: Vec<[f32; 4]> = Vec::new();
        let mut cmd_state: Vec<[f32; 4]> = Vec::new();
        let mut fuse_state: Vec<[f32; 4]> = Vec::new();
        let mut wing_spans = Vec::with_capacity(nv);
        let base_static = (box_base + box_recs.len()) as u32;
        let aero_base = base_static;
        let wing_base = aero_base + 2 * nv as u32;
        let total_wings: usize = vehicles.iter().map(|v| v.wings.len()).sum();
        let total_cmds: usize = vehicles.iter().map(|v| v.centering.len()).sum();
        let cmd_base = wing_base + 4 * total_wings as u32;
        let table_base = cmd_base + 2 * total_cmds as u32;
        // Airfoil tables shared by name across vehicles.
        let mut tables: Vec<(String, u32)> = Vec::new();
        let (mut node_off, mut beam_off) = (0u32, 0u32);
        for v in vehicles {
            let table_of = |t: u32, tables: &mut Vec<(String, u32)>, recs: &mut Vec<[u32; 4]>| {
                let a = &v.airfoils[t as usize];
                if let Some((_, at)) = tables.iter().find(|(n, _)| *n == a.name) {
                    return *at;
                }
                let at = table_base + recs.len() as u32;
                for i in 0..engine_softbody::explicit_aero::AIRFOIL_ENTRIES {
                    recs.push([a.cl[i].to_bits(), a.cd[i].to_bits(), a.cm[i].to_bits(), 0]);
                }
                tables.push((a.name.clone(), at));
                at
            };
            aero_recs.push([
                (wing_recs.len() / 4) as u32,
                v.wings.len() as u32,
                (cmd_recs.len() / 2) as u32,
                v.centering.len() as u32,
            ]);
            wing_spans.push(((wing_recs.len() / 4) as u32, v.wings.len() as u32));
            let fd = v.fusedrag.as_ref();
            aero_recs.push([
                node_off + fd.map_or(0, |f| f.front),
                fd.map_or(0.0f32, |f| f.width).to_bits(),
                v.nodes.len() as u32,
                0,
            ]);
            fuse_state.push([v.fuse_force.x, v.fuse_force.y, v.fuse_force.z, 0.0]);
            for w in &v.wings {
                let g = |s: usize| node_off + w.nodes[s];
                wing_recs.push([g(0), g(1), g(2), g(3)]);
                wing_recs.push([g(4), g(5), g(6), g(7)]);
                let table = table_of(w.airfoil, &mut tables, &mut table_recs_aero);
                let (side, span, area) = match w.induced {
                    None => (0, 0.0f32, 0.0f32),
                    Some(i) => (
                        if i.side == engine_softbody::explicit_aero::WingSide::Left { 1 } else { 2 },
                        i.span,
                        i.area,
                    ),
                };
                wing_recs.push([w.efficacy.to_bits(), w.break_ref.to_bits(), table, side]);
                wing_recs.push([span.to_bits(), area.to_bits(), 0, 0]);
                wing_state.push([w.broken as u32 as f32, 0.0, 0.0, 0.0]);
                wing_state.extend([[0.0f32; 4]; 3]);
            }
            for c in &v.centering {
                cmd_recs.push([beam_off + c.beam, c.contract_first as u32, c.centre.to_bits(), 0]);
                cmd_recs.push([
                    c.shorten_rate.to_bits(),
                    c.lengthen_rate.to_bits(),
                    c.max_contraction.to_bits(),
                    c.max_extension.to_bits(),
                ]);
                cmd_state.push([c.mode as f32, 0.0, 0.0, 0.0]);
            }
            node_off += v.nodes.len() as u32;
            beam_off += v.beams.len() as u32;
        }
        debug_assert!(slots.iter().flatten().all(|&s| s != u32::MAX));
        let inc_base = beams.len() as u32;
        beams.extend(std::iter::repeat_n([0.0f32; 4], lists.len()));
        // The self-collision grid's sorted runs, at each vehicle's candidate offset.
        let grid_base = beams.len() as u32;
        beams.extend(std::iter::repeat_n([0.0f32; 4], lists.len()));
        let grid = vehicles.iter().any(|v| grid_cell(v) > 0.0) as u32;
        let slot_base = lists.len() as u32;
        lists.extend(slots.iter().flatten());
        let cand_table_base = lists.len() as u32;
        let total_cands = cand_table.len() as u32;
        lists.extend(cand_table);
        // The wide passes' step counter and per-vehicle rebase shift.
        let wide_misc = misc.len() as u32;
        misc.extend(std::iter::repeat_n([0u32; 4], 1 + nv));
        // gAcc: the accumulators, then the wide passes' grid counts and tested
        // counts (atomic).
        let grid_count_base = (total_nodes * 3) as u32;
        let tested_base = grid_count_base + (nv * WIDE_GRID_BUCKETS) as u32;
        let overflow_base = tested_base + 2 * nv as u32;
        let acc_len = overflow_base as usize + nv;
        // gBeams: the wide grid's bucket slots, for vehicles on the wide passes.
        let slot_base16 = beams.len() as u32;
        if vehicles.iter().any(|v| v.beams.len() > WIDE_BEAMS) {
            beams.extend(std::iter::repeat_n(
                [0.0f32; 4],
                nv * WIDE_GRID_BUCKETS * WIDE_SLOTS,
            ));
        }
        // gBeams: the aero state (fuselage drag, wings, commands).
        let aero_state = beams.len() as u32;
        beams.extend(fuse_state);
        beams.extend(wing_state);
        beams.extend(cmd_state);
        // gNext: per list position the grid rank, then the wide passes' grid
        // offsets and tested lists (plain words: only the counters are atomic).
        let grid_off_base = lists.len() as u32;
        let tested_list_base = grid_off_base + (nv * (WIDE_GRID_BUCKETS + 1)) as u32;
        let next_len = tested_list_base as usize + total_tris;
        let table_recs = vehicle_recs.clone();
        let mut statics = vehicle_recs;
        statics.extend(node_info);
        statics.extend(beam_recs);
        statics.extend(tri_recs);
        statics.extend(box_recs);
        debug_assert_eq!(statics.len() as u32, aero_base);
        statics.extend(aero_recs);
        statics.extend(wing_recs);
        statics.extend(cmd_recs);
        statics.extend(table_recs_aero);
        let shock2_base = statics.len() as u32;
        let shock2_count = (shock_recs.len() / 2) as u32;
        statics.extend(shock_recs);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("explicit_beams"),
            // Large vehicles get twice the lanes (more loads in flight per step).
            source: wgpu::ShaderSource::Wgsl(
                match (aero, vehicles.iter().any(|v| v.beams.len() > WIDE_BEAMS)) {
                    (false, true) => EXPLICIT_BEAMS_WIDE_WGSL,
                    (false, false) => EXPLICIT_BEAMS_WGSL,
                    (true, true) => EXPLICIT_BEAMS_AERO_WIDE_WGSL,
                    (true, false) => EXPLICIT_BEAMS_AERO_WGSL,
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
            total_nodes as u32,
            total_beams as u32,
            total_tris as u32,
            total_cands,
            wide_misc,
            grid_count_base,
            grid_off_base,
            tested_base,
            tested_list_base,
            cand_table_base,
            slot_base16,
            overflow_base,
            aero_base,
            wing_base,
            cmd_base,
            aero_state,
            total_wings as u32,
            shock2_base,
            shock2_count,
        ];
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-params"),
            size: PARAM_STRIDE * u64::from(PARAM_COPIES),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params, 0, bytemuck::cast_slice(&words));
        let statics = init(
            "explicit-beams-static",
            bytemuck::cast_slice(&statics),
            storage,
        );
        let lists = init(
            "explicit-beams-lists",
            bytemuck::cast_slice(&lists),
            storage,
        );
        let nodes = init("explicit-beams-nodes", bytemuck::cast_slice(&nodes), state);
        let beams = init("explicit-beams-beams", bytemuck::cast_slice(&beams), state);
        let acc = init(
            "explicit-beams-acc",
            bytemuck::cast_slice(&vec![0i32; acc_len]),
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
            bytemuck::cast_slice(&vec![0u32; next_len.max(1)]),
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
        let buffers = [
            &params, &statics, &lists, &nodes, &beams, &acc, &groups, &misc, &next,
        ];
        // The wide passes do not run wings, the fuselage drag, commands or shocks2.
        let wide = (nv <= WIDE_MAX_VEHICLES && !aero)
            .then(|| Wide::new(device, queue, &buffers, &table_recs));
        let total = [
            total_nodes as u32,
            total_beams as u32,
            total_tris as u32,
            total_cands,
            group_total,
        ];
        // Every vehicle must have run the same steps for the wide passes (one
        // step parity for the group slots).
        let same_steps = vehicles.windows(2).all(|w| w[0].steps == w[1].steps);
        let wide = wide.map(|mut w| {
            w.enabled = same_steps && vehicles.iter().any(|v| v.beams.len() > WIDE_BEAMS);
            w
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
            aero_state,
            wing_spans,
            timing: None,
            wide,
            total,
        }
    }

    /// Run frames as the wide passes (one thread per element over every vehicle,
    /// a dispatch per pass) or as one workgroup per vehicle. Wide is the default
    /// when a vehicle has more than [`WIDE_BEAMS`] beams. Returns whether wide is on.
    pub fn is_wide(&self) -> bool {
        self.wide.as_ref().is_some_and(|w| w.enabled)
    }

    /// Run frames as the wide passes or one workgroup per vehicle (see
    /// [`Self::is_wide`]); returns whether wide is on.
    pub fn set_wide(&mut self, on: bool) -> bool {
        if let Some(w) = &mut self.wide {
            w.enabled = on;
        }
        self.wide.as_ref().is_some_and(|w| w.enabled)
    }

    /// One wide frame with each dispatch in its own timed pass (a diagnostic: the
    /// passes add their own cost): GPU milliseconds per pass kind, summed over the
    /// frame, in [`EXPLICIT_BEAMS_WIDE_PASSES`] order. Empty unless wide is on and
    /// the device has timestamp queries.
    pub fn profile_wide_frame(&mut self, steps: u32) -> Vec<(&'static str, f64)> {
        if !self.is_wide()
            || !self
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return Vec::new();
        }
        self.write_params(steps);
        let plan = self.wide_plan(steps);
        let w = self.wide.as_ref().unwrap();
        let count = 2 * plan.len() as u32;
        let set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("explicit-beams-wide-profile"),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-wide-profile"),
            size: u64::from(count) * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for (i, &(step, kind, x)) in plan.iter().enumerate() {
            if i == 2 {
                let nv = self.spans.len() as u64;
                enc.copy_buffer_to_buffer(
                    &self.misc,
                    0,
                    &w.table,
                    (WIDE_MAX_VEHICLES * 4 * 16) as u64,
                    nv * 16,
                );
            }
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("explicit_beams_wide_profile"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &set,
                    beginning_of_pass_write_index: Some(2 * i as u32),
                    end_of_pass_write_index: Some(2 * i as u32 + 1),
                }),
            });
            pass.set_pipeline(&w.passes[kind]);
            pass.set_bind_group(0, &w.group, &[step * PARAM_STRIDE as u32]);
            pass.dispatch_workgroups(x, 1, 1);
        }
        enc.resolve_query_set(&set, 0..count, &resolve, 0);
        self.queue.submit([enc.finish()]);
        let period = f64::from(self.queue.get_timestamp_period()) * 1e-6;
        let raw = self.read(&resolve);
        let ticks: &[u64] = bytemuck::cast_slice(&raw);
        let mut out: Vec<(&'static str, f64)> = EXPLICIT_BEAMS_WIDE_PASSES
            .iter()
            .map(|(n, _)| (*n, 0.0))
            .collect();
        for (i, &(_, kind, _)) in plan.iter().enumerate() {
            out[kind].1 += ticks[2 * i + 1].saturating_sub(ticks[2 * i]) as f64 * period;
        }
        out
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

    /// Resolve the timed frames since [`Self::enable_frame_timing`] into a new
    /// mappable buffer and restart timing empty, without waiting: the buffer
    /// holds a (begin, end) pair of ticks per frame, the count is the frames.
    /// Map it once the queue reaches it (browsers: `mapAsync`) and scale the
    /// ticks by [`Self::timestamp_period_ms`]. `None` with no timing or no frames.
    pub fn take_frame_times(&mut self) -> Option<(wgpu::Buffer, u32)> {
        let t = self.timing.as_mut()?;
        let used = std::mem::take(&mut t.used);
        if used == 0 {
            return None;
        }
        let size = u64::from(used) * 16;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-frame-times"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.resolve_query_set(&t.set, 0..2 * used, &t.resolve, 0);
        enc.copy_buffer_to_buffer(&t.resolve, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        Some((staging, used))
    }

    /// Milliseconds per timestamp tick.
    pub fn timestamp_period_ms(&self) -> f64 {
        f64::from(self.queue.get_timestamp_period()) * 1e-6
    }

    /// The node state buffer, 3 `vec4<f32>` per node in vehicle order: position
    /// relative to the vehicle's origin and mass, velocity and friction, force.
    /// Vehicle `v`'s nodes start at [`Self::node_offset`]`(v)`.
    pub fn nodes_buffer(&self) -> &wgpu::Buffer {
        &self.nodes
    }

    /// The beam state buffer: 2 `vec4<f32>` per beam in vehicle order, (rest
    /// length, compression and tension yield, gate) and (strength, last force,
    /// broken as 0 or 1, 0), then the kernel's scratch. Vehicle `v`'s beams start
    /// at [`Self::beam_offset`]`(v)`.
    pub fn beams_buffer(&self) -> &wgpu::Buffer {
        &self.beams
    }

    /// The per-vehicle buffer: vehicle `v`'s record (`vec4<u32>`) at index `v` holds
    /// its origin's bits (xyz) and the steps it has run (w); triangle and wide
    /// pass scratch follows.
    pub fn misc_buffer(&self) -> &wgpu::Buffer {
        &self.misc
    }

    /// The first node of vehicle `v` in [`Self::nodes_buffer`] (in nodes).
    pub fn node_offset(&self, v: usize) -> u32 {
        self.spans[v].node_off
    }

    /// The first beam of vehicle `v` in [`Self::beams_buffer`] (in beams).
    pub fn beam_offset(&self, v: usize) -> u32 {
        self.spans[v].beam_off
    }

    /// Spawn rest lengths of every beam, in buffer order.
    pub fn rest0(&self) -> &[f32] {
        &self.rest0
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
        self.write_params(steps);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        // Timestamps at the start of the frame's first pass and the end of its last.
        let slot = self.timing.as_mut().and_then(|t| {
            (t.used < t.capacity).then(|| {
                t.used += 1;
                2 * t.used - 2
            })
        });
        let set = self.timing.as_ref().map(|t| &t.set);
        let stamps = |begin: bool, end: bool| {
            set.zip(slot)
                .map(|(q, k)| wgpu::ComputePassTimestampWrites {
                    query_set: q,
                    beginning_of_pass_write_index: begin.then_some(k),
                    end_of_pass_write_index: end.then_some(k + 1),
                })
        };
        if let Some(w) = self.wide.as_ref().filter(|w| w.enabled) {
            let plan = self.wide_plan(steps);
            // Begin and rebase, then the vehicles' origins and steps into the
            // table, then the steps.
            let (head, rest) = plan.split_at(2);
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("explicit_beams_wide_begin"),
                    timestamp_writes: stamps(true, false),
                });
                for &(k, kind, x) in head {
                    pass.set_pipeline(&w.passes[kind]);
                    pass.set_bind_group(0, &w.group, &[k * PARAM_STRIDE as u32]);
                    pass.dispatch_workgroups(x, 1, 1);
                }
            }
            let nv = self.spans.len() as u64;
            enc.copy_buffer_to_buffer(
                &self.misc,
                0,
                &w.table,
                (WIDE_MAX_VEHICLES * 4 * 16) as u64,
                nv * 16,
            );
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("explicit_beams_wide"),
                timestamp_writes: stamps(false, true),
            });
            for &(k, kind, x) in rest {
                pass.set_pipeline(&w.passes[kind]);
                pass.set_bind_group(0, &w.group, &[k * PARAM_STRIDE as u32]);
                pass.dispatch_workgroups(x, 1, 1);
            }
        } else {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("explicit_beams"),
                timestamp_writes: stamps(true, true),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(self.spans.len() as u32, 1, 1);
        }
        self.queue.submit([enc.finish()]);
    }

    /// The parameters for a frame of `steps` steps: copy k (at k × PARAM_STRIDE)
    /// carries step k of the frame (k = steps: the frame's end).
    fn write_params(&mut self, steps: u32) {
        assert!(
            steps < PARAM_COPIES,
            "at most {} steps per frame",
            PARAM_COPIES - 1
        );
        self.words[0] = steps;
        let mut bytes = vec![0u8; (PARAM_STRIDE * u64::from(steps + 1)) as usize];
        for k in 0..=steps {
            let mut w = self.words;
            w[11] = k;
            let at = (PARAM_STRIDE * u64::from(k)) as usize;
            bytes[at..at + 4 * PARAM_WORDS].copy_from_slice(bytemuck::cast_slice(&w));
        }
        self.queue.write_buffer(&self.params, 0, &bytes);
    }

    /// The wide passes of a frame: (parameter copy, pass, workgroups).
    fn wide_plan(&self, steps: u32) -> Vec<(u32, usize, u32)> {
        let groups = |n: u32| n.div_ceil(256).max(1);
        let nv = self.spans.len() as u32;
        let [nodes, beams, tris, cands, slots] = self.total;
        let buckets = nv * WIDE_GRID_BUCKETS as u32;
        let mut plan = vec![(0, WIDE_BEGIN, groups(nv)), (0, WIDE_REBASE, groups(nodes))];
        for k in 0..steps {
            plan.push((k, WIDE_NODES, groups(nodes + slots + buckets)));
            plan.push((k, WIDE_BEAMS_PASS, groups(beams + cands + tris)));
            plan.push((k, WIDE_TRIS, groups(tris * WIDE_SUBS)));
        }
        plan.push((steps, WIDE_END, groups(tris + beams)));
        plan.push((steps, WIDE_CLOSE, groups(nv)));
        plan
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

    /// Per vehicle, its wings' broken flags (file order).
    pub fn wings_broken(&self) -> Vec<Vec<bool>> {
        let raw = self.read(&self.beams);
        let v: &[[f32; 4]] = bytemuck::cast_slice(&raw);
        let nv = self.spans.len() as u32;
        self.wing_spans
            .iter()
            .map(|&(first, count)| {
                (first..first + count)
                    .map(|gw| v[(self.aero_state + nv + 4 * gw) as usize][0] != 0.0)
                    .collect()
            })
            .collect()
    }

    /// Per vehicle, the fuselage drag force on each node (the last step's D).
    pub fn fuse_forces(&self) -> Vec<Vec3> {
        let raw = self.read(&self.beams);
        let v: &[[f32; 4]] = bytemuck::cast_slice(&raw);
        (0..self.spans.len())
            .map(|i| {
                let r = v[self.aero_state as usize + i];
                Vec3::new(r[0], r[1], r[2])
            })
            .collect()
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
                            broken: v[g * 2 + 1][2] != 0.0,
                            rest: v[g * 2][0],
                            rest0: self.rest0[g],
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

/// Beams above which a vehicle runs on the 512-lane module, and wide passes are the
/// default.
const WIDE_BEAMS: usize = 2048;
/// The kernel's GRID_BUCKETS and GRID_SUBS.
const WIDE_GRID_BUCKETS: usize = 1024;
const WIDE_SUBS: u32 = 8;
/// The kernel's WIDE_SLOTS (grid bucket slots of the wide passes).
const WIDE_SLOTS: usize = 16;

impl Wide {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffers: &[&wgpu::Buffer; 9],
        vehicle_recs: &[[u32; 4]],
    ) -> Self {
        let table = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-vehicle-table"),
            size: (WIDE_MAX_VEHICLES * 5 * 16) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let n = vehicle_recs.len().min(WIDE_MAX_VEHICLES * 4);
        queue.write_buffer(&table, 0, bytemuck::cast_slice(&vehicle_recs[..n]));
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                // The parameters: one copy per step, picked by the offset.
                has_dynamic_offset: binding == 0,
                min_binding_size: None,
            },
            count: None,
        };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("explicit-beams-wide"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, ro),
                entry(2, ro),
                entry(3, rw),
                entry(4, rw),
                entry(5, rw),
                entry(6, rw),
                entry(7, rw),
                entry(8, rw),
                entry(9, wgpu::BufferBindingType::Uniform),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("explicit-beams-wide"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let passes = EXPLICIT_BEAMS_WIDE_PASSES
            .iter()
            .map(|(entry, wgsl)| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(entry),
                    source: wgpu::ShaderSource::Wgsl((*wgsl).into()),
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("explicit-beams-wide"),
            layout: &layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: if i == 0 {
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: b,
                            offset: 0,
                            size: std::num::NonZeroU64::new(PARAM_STRIDE),
                        })
                    } else {
                        b.as_entire_binding()
                    },
                })
                .chain(std::iter::once(wgpu::BindGroupEntry {
                    binding: 9,
                    resource: table.as_entire_binding(),
                }))
                .collect::<Vec<_>>(),
        });
        Self {
            passes,
            group,
            enabled: false,
            table,
        }
    }
}

/// The self-collision grid's cell size for a vehicle (0: all pairs): the
/// solver's own ([`ExplicitBeams::grid_cell`]), so CPU and GPU share it.
fn grid_cell(v: &ExplicitBeams) -> f32 {
    v.grid_cell
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
