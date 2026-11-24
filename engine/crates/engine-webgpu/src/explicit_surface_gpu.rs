//! Surface contact's pipelines on the WebGPU host (a part of `explicit_beams_gpu.rs`, its
//! parent module): the frame in segments with the surface pass's pipelines between them
//! (`Kernel::FrameSeg`, `Kernel::Surf`), and the pass's pipelines of the fast schedules
//! (`FramePlanner::set_surface_fast`: `plan::SURF_FAST_PASSES`, the solo vehicles' rest).

use super::*;

/// The split frame's bind group layout entries ([`Kernel::FrameSeg`], [`Kernel::Surf`]):
/// the wide passes' without their vehicle table (binding 0: the segment's parameter
/// copy, by the dynamic offset).
pub(super) fn split_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut e = wide_entries();
    e.retain(|x| x.binding != BIND_TABLE);
    e
}

/// Whether hosts run surface contact split from the frame (on unless
/// `RUBBLE_SURFACE_SPLIT=0`, a measurement: the frame with the pass in it).
pub fn surface_split_default() -> bool {
    std::env::var("RUBBLE_SURFACE_SPLIT").as_deref() != Ok("0")
}

/// Measurement (mask bit 16 of [`explicit_shader_checks_off`], never in a default): the
/// surface pass's pipelines with their workgroup memory zeroed at each workgroup's start,
/// as naga does unless told otherwise. Natively they are made without it: naga's MSL
/// zeroes from one lane, 0.16 to 0.19 ms a dispatch on an M3 Max for the coupling's
/// workgroup copies (2,080 words), nine dispatches a step, 52 of the 70 ms of a one-car
/// frame in segments (vehicle-perf-first-principles.md §8.53). Every workgroup value
/// these entries read is written in the same dispatch before the barrier the reads
/// follow: the coupling's header copies and the empty sphere table (coupleBegin,
/// cplGsClear in surfPipeSkip), the prep's pair-path flag, the sleep rule's flag
/// (sleepSkip). `--ab-switch checks=21:5` holds the two bit for bit.
#[doc(hidden)]
pub fn surface_pipes_zeroed(device: &wgpu::Device) -> bool {
    explicit_shader_checks_off(device) & 16 != 0
}

/// The split frame's pipelines and bind group ([`FramePlanner::set_surface_split`]); the
/// surface pass's pipelines also serve the fast schedules
/// ([`FramePlanner::set_surface_fast`]), which never make the segments' pipeline.
pub(super) struct Split {
    frame: std::sync::OnceLock<wgpu::ComputePipeline>,
    pub(super) surf: Vec<wgpu::ComputePipeline>,
    pub(super) group: wgpu::BindGroup,
    /// The fast schedules' pipelines (`plan::SURF_FAST_PASSES`), made at their first
    /// use: the prep and the plan with the surface arguments at binding 10 (storage),
    /// the others sized from them (indirect; bound as `group`).
    fast: std::sync::OnceLock<(Vec<wgpu::ComputePipeline>, wgpu::BindGroup)>,
    args: wgpu::Buffer,
    bufs: [wgpu::Buffer; 9],
    device: wgpu::Device,
    pipeline_layout: wgpu::PipelineLayout,
    frame_module: String,
    bit_exact: bool,
}

impl Split {
    /// The surface pass's arguments on the fast schedules ([`schedule::ArgBuf::Surf`]).
    pub(super) fn args(&self) -> &wgpu::Buffer {
        &self.args
    }

    /// The segments' pipeline ([`Kernel::FrameSeg`]), made at its first use.
    pub(super) fn frame(&self) -> &wgpu::ComputePipeline {
        self.frame.get_or_init(|| {
            let module = explicit_shader_module(&self.device, wgpu::ShaderModuleDescriptor {
                label: Some(&self.frame_module),
                source: wgpu::ShaderSource::Wgsl(host_wgsl(module_wgsl(&self.frame_module), self.bit_exact).into()),
            });
            self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&self.frame_module),
                layout: Some(&self.pipeline_layout),
                module: &module,
                entry_point: Some(plan::FRAME_ENTRY),
                compilation_options: explicit_compilation_options(&self.device),
                cache: None,
            })
        })
    }

    /// The fast schedules' pipelines and the bind group of those that write the surface
    /// arguments (see `fast`). Without the zeroing of workgroup memory, as the others
    /// ([`surface_pipes_zeroed`]).
    pub(super) fn fast(&self) -> &(Vec<wgpu::ComputePipeline>, wgpu::BindGroup) {
        self.fast.get_or_init(|| {
            let device = &self.device;
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("explicit-beams-surface-args"),
                entries: &self_split_step_entries(),
            });
            let args_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("explicit-beams-surface-args"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let zero = surface_pipes_zeroed(device) && explicit_compilation_options(device).zero_initialize_workgroup_memory;
            let pipes = plan::SURF_FAST_PASSES
                .iter()
                .enumerate()
                .map(|(i, &(name, entry))| {
                    let module = explicit_shader_module(device, wgpu::ShaderModuleDescriptor {
                        label: Some(name),
                        source: wgpu::ShaderSource::Wgsl(host_wgsl(module_wgsl(name), self.bit_exact).into()),
                    });
                    let binds = schedule::surf_args_offset(0, i as u8).is_none();
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(name),
                        layout: Some(if binds { &args_layout } else { &self.pipeline_layout }),
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: wgpu::PipelineCompilationOptions { zero_initialize_workgroup_memory: zero, ..Default::default() },
                        cache: None,
                    })
                })
                .collect();
            let mut entries: Vec<wgpu::BindGroupEntry> = (0..9)
                .map(|i| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: if i == 0 { params_window(&self.bufs[0]) } else { self.bufs[i].as_entire_binding() },
                })
                .collect();
            entries.push(wgpu::BindGroupEntry { binding: 10, resource: self.args.as_entire_binding() });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("explicit-beams-surface-args"), layout: &layout, entries: &entries });
            (pipes, group)
        })
    }

    pub(super) fn new(device: &wgpu::Device, bufs: &[wgpu::Buffer; 9], frame_module: &str, bit_exact: bool) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("explicit-beams-split"),
            entries: &split_entries(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("explicit-beams-split"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        // (The surface pass's pipelines without the zeroing of workgroup memory unless a
        // measurement asks: surface_pipes_zeroed.)
        let zero = surface_pipes_zeroed(device) && explicit_compilation_options(device).zero_initialize_workgroup_memory;
        let pipe = |module_name: &str, entry: &str| {
            let module = explicit_shader_module(&device, wgpu::ShaderModuleDescriptor {
                label: Some(module_name),
                source: wgpu::ShaderSource::Wgsl(host_wgsl(module_wgsl(module_name), bit_exact).into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(module_name),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: zero,
                    ..Default::default()
                },
                cache: None,
            })
        };
        let surf = plan::SURF_PASSES.iter().map(|&(m, e)| pipe(m, e)).collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("explicit-beams-split"),
            layout: &layout,
            entries: &std::array::from_fn::<_, 9, _>(|i| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: if i == 0 { params_window(&bufs[0]) } else { bufs[i].as_entire_binding() },
            }),
        });
        let args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-surface-args"),
            size: schedule::SURF_ARGS_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        Self {
            frame: std::sync::OnceLock::new(),
            surf,
            group,
            fast: std::sync::OnceLock::new(),
            args,
            bufs: bufs.clone(),
            device: device.clone(),
            pipeline_layout,
            frame_module: frame_module.to_string(),
            bit_exact,
        }
    }
}

/// Whether an upload's surface contact runs split from the frame
/// ([`surface_split_default`]): only with the World's coupling. Without it the frame
/// with the pass in it (the `_nocpl` module) is small enough, and one pipeline.
pub(super) fn surface_split_for(coupled: bool) -> bool {
    coupled && surface_split_default()
}

impl ExplicitBeamsGpu {
    /// A fleet with surface contact on the fast schedules (on: the wide passes with
    /// their reused pair list, self-collision split with vehicle contact), the surface
    /// pass as its own pipelines after each step's self-collision hits; or (off) a
    /// workgroup a vehicle or a group with the pass inside it, or the frame in segments.
    /// The same results bit for bit (`--ab-switch surffast`;
    /// vehicle-perf-first-principles.md §8.53). By default natively a wide frame's
    /// alone ([`plan::schedule::surface_fast_default`],
    /// [`plan::schedule::surface_fast_contact_default`]; `RUBBLE_SURFACE_FAST=1` both
    /// parts, 0 neither); this sets both. No effect without surface contact.
    pub fn set_surface_fast(&mut self, on: bool) {
        self.planner.set_surface_fast(on);
    }

    /// Whether the fleet's surface contact runs as pipelines on the fast schedules
    /// ([`Self::set_surface_fast`], and the fleet has surface contact).
    pub fn surface_fast(&self) -> bool {
        self.planner.surface_piped()
    }

    /// A/B harnesses (`vehicle_lab --ab-switch NAME`): the host switch of that name set,
    /// or false for a name that is none. `surffast`: [`Self::set_surface_fast`]; `vvwide`:
    /// [`FramePlanner::set_vv_wide`].
    #[doc(hidden)]
    pub fn set_named_switch(&mut self, name: &str, on: bool) -> bool {
        match name {
            "surffast" => self.set_surface_fast(on),
            "vvwide" => self.set_vv_wide(on),
            _ => return false,
        }
        true
    }

    /// What [`Self::set_named_switch`]'s `name` switches (`None`: no such switch).
    #[doc(hidden)]
    pub fn named_switch(name: &str) -> Option<&'static str> {
        match name {
            "surffast" => Some("surface contact on the fast schedules"),
            "vvwide" => Some("vehicle contact's frames without a group on the wide passes"),
            "surffast_wide" => Some("surface contact on the fast schedules, a wide frame's part alone"),
            "surffast_contact" => Some("surface contact on the fast schedules, its vehicle-contact part (the wide part on)"),
            _ => None,
        }
    }

    /// A browser's A/B of a host switch (`WebWorld.explicitAbSwitch`): `surffast` (both
    /// parts of [`Self::set_surface_fast`]), `surffast_wide` (its wide frames' part alone
    /// against neither), `surffast_contact` (its vehicle-contact part, the wide part on
    /// either side), `vvwide`, on or off. False for a name that is none. Put back with
    /// [`Self::reset_schedule_switches`].
    #[doc(hidden)]
    pub fn set_ab_switch(&mut self, name: &str, on: bool) -> bool {
        match name {
            "surffast_wide" => {
                self.planner.set_surface_fast(on);
                self.planner.set_surface_fast_contact(false);
            }
            "surffast_contact" => {
                self.planner.set_surface_fast(true);
                self.planner.set_surface_fast_contact(on);
            }
            _ => return self.set_named_switch(name, on),
        }
        true
    }

    /// The schedule switches as this host starts them
    /// ([`plan::schedule::host_setting`] and the defaults [`Self::with_coupling`]
    /// applies): after an A/B of [`Self::set_ab_switch`], or once a host setting has
    /// changed.
    pub fn reset_schedule_switches(&mut self) {
        self.planner.set_surface_fast(plan::schedule::surface_fast_default());
        self.planner.set_surface_fast_contact(plan::schedule::surface_fast_contact_default(false));
        let n = self.layout.spans.len();
        self.planner.set_vv_wide(plan::vv_wide::vv_wide_default(vv_wide_gpu::default_on(n)));
    }

    /// Whether a browser host compiled the modules the schedule switches take as the
    /// host settings now say ([`Self::reset_schedule_switches`]): false when a switch is
    /// on now that was off when this host's modules were listed
    /// ([`Self::compute_specs_for`]), so the page builds again to compile them. Always
    /// true without the pipeline gate (native hosts create pipelines at first use).
    pub fn schedule_modules_listed(&self) -> bool {
        let n = self.layout.spans.len();
        let now = (plan::schedule::surface_fast_default(), plan::vv_wide::vv_wide_default(vv_wide_gpu::default_on(n)));
        !crate::pipeline_gated() || ((!now.0 || self.schedule_listed.0) && (!now.1 || self.schedule_listed.1))
    }

    /// The fast schedules' surface launches ([`GroupOp::SoloRest`], [`GroupOp::Surf`]).
    pub(super) fn run_surface_op(&self, pass: &mut wgpu::ComputePass, op: GroupOp) {
        let args = &self.contact.args;
        let at = |k: u32| [k * PARAM_STRIDE as u32];
        match op {
            GroupOp::SoloRest { copy } => {
                // Surface contact as pipelines: the solo vehicles' hits, a workgroup a
                // vehicle as their step's (a member's returns at once).
                let p = self.self_split_pipes();
                let (aero, _, lanes) = self.module_kind;
                pass.set_pipeline(p.rest_vv_pipe(aero, lanes));
                pass.set_bind_group(0, &p.pairs_bind, &at(copy));
                pass.dispatch_workgroups_indirect(args, SPLIT_ALL_ARGS);
            }
            GroupOp::Surf { copy, pass: i, groups } => {
                // The prep and the plan with the surface arguments bound (they size the
                // others), the others from the step's slot of them.
                let s = self.split_pipes();
                let (pipes, args_group) = s.fast();
                pass.set_pipeline(&pipes[i as usize]);
                match schedule::surf_args_offset(copy, i) {
                    None => {
                        pass.set_bind_group(0, args_group, &at(copy));
                        pass.dispatch_workgroups(groups, 1, 1);
                    }
                    Some(offset) => {
                        pass.set_bind_group(0, &s.group, &at(copy));
                        pass.dispatch_workgroups_indirect(&s.args, offset);
                    }
                }
            }
            _ => unreachable!("not a surface launch"),
        }
    }
}

/// The fast schedules' modules of a fleet with surface contact, for a browser host's
/// compiles ahead ([`ExplicitBeamsGpu::compute_specs`]; `specs` so far, `u` the host's
/// WGSL of a module): the pass's pipelines and, with vehicle contact (`several`), the
/// solo vehicles' rest. Deferred: frames keep the pass in the kernels until they are
/// compiled.
pub(super) fn fast_specs(specs: &mut Vec<crate::ComputeSpec>, aero: bool, lanes: u32, several: bool, u: &dyn Fn(&str) -> String) {
    {
        let rest: &'static str = crate::EXPLICIT_BEAMS_SELF_SPLIT
            .iter()
            .map(|(n, _)| *n)
            .find(|n| *n == plan::solo_rest_module_name(aero, lanes, true))
            .expect("explicit beams: the solo rest module");
        let pipes = plan::SURF_PASSES
            .iter()
            .chain(plan::SURF_FAST_PASSES.iter())
            .copied()
            .chain(several.then_some((rest, plan::SOLO_REST_ENTRY)));
        for (module, entry) in pipes {
            if specs.iter().any(|s| s.module == module) {
                continue;
            }
            // (The fast prep and plan bind the surface arguments at binding 10.)
            let binds = module == plan::SURF_FAST_PASSES[0].0 || module == plan::SURF_FAST_PASSES[2].0;
            let layout = if binds { self_split_step_entries() } else { split_entries() };
            specs.push(crate::ComputeSpec { label: entry, module, wgsl: u(module_wgsl(module)), entry, layout: Some(layout), deferred: true });
        }
    }
}
