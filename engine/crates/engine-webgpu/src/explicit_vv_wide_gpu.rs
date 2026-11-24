//! A fleet with vehicle contact on the wide passes while the contact plan holds no group,
//! on the WebGPU host (a part of `explicit_beams_gpu.rs`, its parent module):
//! `engine_exec::explicit_plan::vv_wide`'s launches. The gate's pipeline and buffer, the
//! node and close passes with the solo vehicles' frame-bounds check; every other launch
//! is a pipeline this host has already (the wide passes, the list kernel, the surface
//! prep and plan, the self split's vehicles' step and solo rest), sized from the gate's
//! buffer.

use super::*;
use plan::vv_wide::{pass_module, WideOp, WideSlot, WIDE_ARGS_BYTES};

/// The largest fleet whose frames without a contact group this host runs as the wide
/// passes by default. Measured on Metal (M3 Max; vehicle-perf-first-principles.md
/// §8.57): 2 to 8 cars apart are 6 to 20 % faster on them, 16 and 20 within 5 % either
/// way, 32 are 5 to 10 % slower (a workgroup a vehicle costs that GPU the same for 2
/// cars as for 32, the wide passes' work grows with the cars).
const DEFAULT_MOST_VEHICLES: usize = 8;

/// [`FramePlanner::set_vv_wide`]'s default on this host for a fleet of `vehicles`
/// unless `RUBBLE_VV_WIDE` says (`plan::vv_wide::vv_wide_default`): on natively up to
/// [`DEFAULT_MOST_VEHICLES`]; off in a browser (wasm32). Chrome's A/B (AGENTS.md, rung 3)
/// held it bit for bit wherever it ran (8 cars in the derby with surface contact's
/// vehicle-contact part on), but with that part off, as it is there, and in the derby
/// without surface contact, its frames were never taken in Chrome, so there is no
/// speed to judge it by (vehicle-perf-first-principles.md §8.64).
pub(super) fn default_on(vehicles: usize) -> bool {
    cfg!(not(target_arch = "wasm32")) && vehicles <= DEFAULT_MOST_VEHICLES
}

/// Vehicle contact's frames without a group on the wide passes where the host starts
/// with them ([`ExplicitBeamsGpu::compute_specs_for`]): the gate and the two passes with
/// the bounds check (VvWide::new's descriptors), deferred (the schedule waits for every
/// module it launches, plan::vv_wide's gate on them). Only beside the wide passes.
pub(super) fn push_specs(specs: &mut Vec<crate::ComputeSpec>, vehicles: usize, u: &dyn Fn(&str) -> String) {
    let wide_listed = specs.iter().any(|s| s.module == plan::WIDE_PASSES[0]);
    if vehicles > 1 && wide_listed && plan::vv_wide::vv_wide_default(default_on(vehicles)) {
        for ((module, entry), layout) in [
            (plan::VV_GATE, fleet_args_entries()),
            (plan::VV_WIDE_NODES, wide_entries()),
            (plan::VV_WIDE_CLOSE, wide_entries()),
        ] {
            specs.push(crate::ComputeSpec { label: entry, module, wgsl: u(module_wgsl(module)), entry, layout: Some(layout), deferred: true });
        }
    }
}

pub(super) struct VvWide {
    gate: wgpu::ComputePipeline,
    /// The node and close passes with the bounds check (the wide passes' layout and bind
    /// group).
    nodes: wgpu::ComputePipeline,
    close: wgpu::ComputePipeline,
    /// The gate's layout (the state bindings, then bindings 10 and 11) and its bind
    /// groups by mode, made at first use: binding 10 the contact plan's arguments, the
    /// self split's, the surface pass's; binding 11 the gate's buffer.
    layout: wgpu::BindGroupLayout,
    binds: [std::sync::OnceLock<wgpu::BindGroup>; 3],
    /// The gate's buffer ([`schedule::ArgBuf::Wide`]): storage at the gate's binding 11,
    /// the indirect arguments of the launches it sizes.
    args: wgpu::Buffer,
}

impl VvWide {
    fn new(device: &wgpu::Device, wide: &Wide, bit_exact: bool) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("explicit-beams-vv-gate"),
            entries: &fleet_args_entries(),
        });
        let gate_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("explicit-beams-vv-gate"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipe = |(name, entry): (&str, &str), layout: &wgpu::PipelineLayout| {
            let module = explicit_shader_module(device, wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(host_wgsl(module_wgsl(name), bit_exact).into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(layout),
                module: &module,
                entry_point: Some(entry),
                // (As the wide passes: no zeroing of workgroup memory. The bounds check
                // reads none; the passes they call write theirs before reading it.)
                compilation_options: wgpu::PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: std::env::var("RUBBLE_WIDE_ZERO").as_deref() == Ok("1"),
                    ..Default::default()
                },
                cache: None,
            })
        };
        let args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-vv-wide-args"),
            size: WIDE_ARGS_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        Self {
            gate: pipe(plan::VV_GATE, &gate_layout),
            nodes: pipe(plan::VV_WIDE_NODES, &wide.pipeline_layout),
            close: pipe(plan::VV_WIDE_CLOSE, &wide.pipeline_layout),
            layout,
            binds: Default::default(),
            args,
        }
    }
}

impl ExplicitBeamsGpu {
    fn vv_wide_pipes(&self) -> &VvWide {
        self.vv_wide.get_or_init(|| {
            let wide = self.wide.as_ref().expect("explicit beams: wide passes not built");
            VvWide::new(&self.device, wide, self.bit_exact)
        })
    }

    /// The gate's bind group of `mode` ([`WideOp::Gate`]).
    fn vv_gate_bind(&self, mode: u32) -> &wgpu::BindGroup {
        let v = self.vv_wide_pipes();
        v.binds[mode as usize].get_or_init(|| {
            let first = match mode {
                0 => &self.contact.args,
                1 => &self.self_split_pipes().args,
                _ => self.split_pipes().args(),
            };
            let mut entries: Vec<wgpu::BindGroupEntry> = (0..9)
                .map(|i| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: if i == 0 { params_window(&self.bufs[0]) } else { self.bufs[i].as_entire_binding() },
                })
                .collect();
            entries.push(wgpu::BindGroupEntry { binding: 10, resource: first.as_entire_binding() });
            entries.push(wgpu::BindGroupEntry { binding: 11, resource: v.args.as_entire_binding() });
            self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("explicit-beams-vv-gate"), layout: &v.layout, entries: &entries })
        })
    }

    /// One launch of the wide frame under vehicle contact ([`GroupOp::Wide`]).
    pub(super) fn run_vv_wide_op(&self, pass: &mut wgpu::ComputePass, op: WideOp) {
        let v = self.vv_wide_pipes();
        let at = |k: u32| [k * PARAM_STRIDE as u32];
        let slot = match op {
            WideOp::Gate { mode } => {
                pass.set_pipeline(&v.gate);
                pass.set_bind_group(0, self.vv_gate_bind(mode), &at(mode));
                pass.dispatch_workgroups(1, 1, 1);
                return;
            }
            WideOp::Pass { kind, copy, slot } => {
                let w = self.wide.as_ref().expect("explicit beams: wide passes not built");
                let module = pass_module(kind).0;
                pass.set_pipeline(if module == plan::VV_WIDE_NODES.0 {
                    &v.nodes
                } else if module == plan::VV_WIDE_CLOSE.0 {
                    &v.close
                } else {
                    &w.passes[kind as usize]
                });
                pass.set_bind_group(0, &w.group, &at(copy));
                slot
            }
            WideOp::List { copy, slot } => {
                let p = self.self_split_pipes();
                pass.set_pipeline(p.list_pipe());
                pass.set_bind_group(0, &p.step_bind, &at(copy));
                slot
            }
            WideOp::SurfBind { copy, pass: i } => {
                let (pipes, args_group) = self.split_pipes().fast();
                pass.set_pipeline(&pipes[i as usize]);
                pass.set_bind_group(0, args_group, &at(copy));
                WideSlot::Nv
            }
            WideOp::SelfMember(s) => {
                let p = self.self_split_pipes();
                pass.set_pipeline(&p.vv_pipes()[0]);
                pass.set_bind_group(0, &p.step_bind, &at(s));
                WideSlot::Old
            }
            WideOp::SoloRest { copy } => {
                let p = self.self_split_pipes();
                let (aero, _, lanes) = self.module_kind;
                pass.set_pipeline(p.rest_vv_pipe(aero, lanes));
                pass.set_bind_group(0, &p.pairs_bind, &at(copy));
                WideSlot::Old
            }
        };
        pass.dispatch_workgroups_indirect(&v.args, slot.offset());
    }
}

/// Frames between this host's reads of the contact plan's group count once one is known.
const SEEN_EVERY: u32 = 4;

/// The host's read of the contact plan's groups of two or more members
/// ([`FramePlanner::set_vv_wide_seen`], [`plan::vv_wide::plan_groups_of_two`]): two words
/// of the plan's arguments copied after a frame's passes and mapped without waiting, taken
/// at a later frame's start.
#[derive(Default)]
pub(super) struct Seen {
    staging: std::sync::OnceLock<wgpu::Buffer>,
    /// 0 idle, 1 the copy encoded (mapped after its submission), 2 the map requested, 3
    /// mapped, 4 failed.
    state: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// Whether a count was ever read, and the frames since the last copy.
    known: std::cell::Cell<bool>,
    age: std::cell::Cell<u32>,
    /// Whether the last frame held the wide schedule ([`ExplicitBeamsGpu::vv_wide_used`]).
    pub(super) used: std::cell::Cell<bool>,
}

impl ExplicitBeamsGpu {
    /// A fleet with vehicle contact runs the frames whose contact plan holds no group as
    /// the wide passes (on), or a workgroup a vehicle (off): the same results bit for bit
    /// (`--ab-switch vvwide`; vehicle-perf-first-principles.md §8.57). By default on
    /// natively for a fleet of at most eight vehicles ([`default_on`]; `RUBBLE_VV_WIDE=1`
    /// any fleet, 0 none). It acts where the wide passes with the reused pair list could
    /// run the fleet but for its contact ([`FramePlanner::vv_wide`]), once this host has
    /// read a plan without a group back ([`FramePlanner::set_vv_wide_seen`]).
    pub fn set_vv_wide(&mut self, on: bool) {
        self.planner.set_vv_wide(on);
    }

    /// The contact groups of two or more vehicles of the current plan (blocking read; 0
    /// when contact is not active): the plan's indirect-arguments word 15, less one.
    pub fn contact_groups(&self) -> u32 {
        if !self.contact_active() {
            return 0;
        }
        let raw = self.read(&self.contact.args);
        bytemuck::cast_slice::<u8, u32>(&raw)[15].saturating_sub(1)
    }

    /// Diagnostic (blocking): the member count of each contact group of the current
    /// plan but the groups of one (0: past VV_MAX_GROUP members, its vehicles alone).
    #[doc(hidden)]
    pub fn contact_group_sizes(&self) -> Vec<u32> {
        if !self.contact_active() {
            return Vec::new();
        }
        // The plan's group count (indirect-arguments word 3; word 4 is the frame's
        // workgroups, vehicles + groups).
        let n = bytemuck::cast_slice::<u8, u32>(&self.read(&self.contact.args))[3] as usize;
        let misc: Vec<u32> = bytemuck::cast_slice(&self.read(&self.misc)).to_vec();
        let at = self.planner.words[46] as usize;
        (0..n).map(|g| misc[(at + g) * 4 + 1]).filter(|&m| m != 1).collect()
    }

    /// Whether the last frame held the wide schedule beside today's (tests: the path
    /// was taken; which of the two ran is the plan's choice on the GPU).
    #[doc(hidden)]
    pub fn vv_wide_used(&self) -> bool {
        self.vv_seen.used.get()
    }

    /// Tests: the contact plan's group count as if read back
    /// ([`FramePlanner::set_vv_wide_seen`]); the host's own reads replace it.
    #[doc(hidden)]
    pub fn set_vv_wide_seen(&mut self, groups: Option<u32>) {
        self.planner.set_vv_wide_seen(groups);
    }

    /// A split group path dispatch the plan's arguments size at byte `offset`: in a frame
    /// holding the wide schedule from the gate's slot ([`plan::vv_wide::gated_slot`]: the
    /// plan's count with a group of two or more members, none without), else the plan's.
    pub(super) fn dispatch_plan_args(&self, pass: &mut wgpu::ComputePass, offset: u64) {
        if self.vv_seen.used.get() {
            pass.dispatch_workgroups_indirect(&self.vv_wide_pipes().args, plan::vv_wide::gated_slot(offset).offset());
        } else {
            pass.dispatch_workgroups_indirect(&self.contact.args, offset);
        }
    }

    /// A frame's start: a finished read handed to the planner.
    pub(super) fn vv_wide_seen(&mut self) {
        use std::sync::atomic::Ordering::{Acquire, Release};
        let s = &self.vv_seen;
        match s.state.load(Acquire) {
            3 => {
                let b = s.staging.get().expect("explicit beams: the plan's count is mapped");
                let groups = {
                    let view = b.slice(..).get_mapped_range().expect("mapped");
                    let w = |k: usize| u32::from_le_bytes(view[4 * k..4 * k + 4].try_into().unwrap());
                    // (Word 0 the plan's word PLAN_GROUPS_WORD, word 1 its
                    // PLAN_GROUPS_OF_TWO_WORD.)
                    let mut args = [0u32; plan::vv_wide::PLAN_GROUPS_OF_TWO_WORD + 1];
                    args[plan::vv_wide::PLAN_GROUPS_WORD] = w(0);
                    args[plan::vv_wide::PLAN_GROUPS_OF_TWO_WORD] = w(1);
                    plan::vv_wide::plan_groups_of_two(&args).expect("both words")
                };
                b.unmap();
                s.state.store(0, Release);
                s.known.set(true);
                self.planner.set_vv_wide_seen(Some(groups));
            }
            4 => s.state.store(0, Release),
            _ => {}
        }
    }

    /// After a frame's passes: the plan's group count copied for a read, when the setting
    /// can act and no read is in flight (at once until a count is known, then every
    /// [`SEEN_EVERY`] frames).
    pub(super) fn vv_wide_read(&self, enc: &mut wgpu::CommandEncoder) {
        use std::sync::atomic::Ordering::{Acquire, Release};
        let s = &self.vv_seen;
        if !self.planner.vv_wide() {
            return;
        }
        s.age.set(s.age.get() + 1);
        if s.state.load(Acquire) != 0 || (s.known.get() && s.age.get() < SEEN_EVERY) {
            return;
        }
        let b = s.staging.get_or_init(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("explicit-beams-vv-plan-seen"),
                size: 8,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        // (The kernel's vvArgs words 3 and 15: the plan's groups, and its groups of two
        // or more plus one.)
        for (k, word) in [plan::vv_wide::PLAN_GROUPS_WORD, plan::vv_wide::PLAN_GROUPS_OF_TWO_WORD].into_iter().enumerate() {
            enc.copy_buffer_to_buffer(&self.contact.args, 4 * word as u64, b, 4 * k as u64, 4);
        }
        s.age.set(0);
        s.state.store(1, Release);
    }

    /// After the submission holding [`Self::vv_wide_read`]'s copy: its map requested.
    pub(super) fn vv_wide_map(&self) {
        use std::sync::atomic::Ordering::{Acquire, Release};
        let s = &self.vv_seen;
        if s.state.load(Acquire) != 1 {
            return;
        }
        s.state.store(2, Release);
        let state = s.state.clone();
        let b = s.staging.get().expect("explicit beams: the plan's count has a buffer");
        b.slice(..).map_async(wgpu::MapMode::Read, move |r| state.store(if r.is_ok() { 3 } else { 4 }, Release));
    }
}
