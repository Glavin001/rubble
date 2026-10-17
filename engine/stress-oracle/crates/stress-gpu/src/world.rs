//! The standalone world on the GPU: stress-ref `World::step_frame` (explicit, adaptive
//! and quasi-static modes) with the stress solve, scripted loads and probes on the GPU
//! (`GpuSolver`).
//!
//! Adaptive and quasi-static modes: settled clusters move rigidly on the GPU (and, in
//! adaptive mode, wake there when their loads change); the frame's static solves (settle,
//! settled fatigue, the quasi-static cascade) run the reference's own code on the mirror
//! with the external loads the GPU recorded, as the plan keeps static solves on the host
//! until they move to the GPU.
//!
//! A stress-ref `World` is kept as the host shell: it supplies the prestressed initial
//! state, the substep schedule (`substep_dt`) and the observation assembly, run on the
//! GPU solver's mirror. Scripted structural events (removals) run the reference's own
//! code on the mirror at the start of the substep they fall in; segments start there and
//! where a blast first acts, so blast faces are cached with that moment's geometry.
//!
//! Coupled to a rigid-body engine (`GpuWorld::with_engine`), the world runs the
//! reference's coupled frame (world_coupling.rs) with the substeps on the GPU: the
//! reference's own code selects the impact islands, steps the engine with the island
//! bodies frozen and takes its contacts on the clusters it drove, on the mirror; the GPU
//! then simulates the islands with its own chunk-level contact, and the frame is
//! verified and redone as the reference's is. Approximation (named, measured against the
//! reference's coupled world): driven clusters keep their start-of-frame motion through
//! the frame's substeps (the reference interpolates it to the engine's end state); it
//! enters only their rigid-frame loads, since nothing in an island touches a driven body
//! within the frame.

use stress_ref::math::Vec3;
use stress_ref::observation::{Observation, ProbeSeries};
use stress_ref::scene::{EventDesc, LoadDesc, ProbeKind, Scene, SectionComponent, SolveMode};
use stress_ref::solver::{Activity, ChunkLoads, ReferenceSolver};
use stress_ref::statics::StaticOptions;
use stress_ref::engine::{BodyKey, RigidEngine};
use stress_ref::world::{fractured_since, World};

use crate::gpu::Gpu;
use crate::static_inputs::StaticInputs;
use crate::loads::WorldLoads;
use crate::solver::GpuSolver;

/// Probe accumulator (world.rs `ProbeAcc`): envelope since the last sample.
#[derive(Clone, Debug, Default)]
struct ProbeAcc {
    series: ProbeSeries,
    lo: f64,
    hi: f64,
}

pub struct GpuWorld {
    pub scene: Scene,
    /// The host shell (its `solver` is a placeholder; the mirror lives in `solver`).
    shell: World,
    pub solver: GpuSolver,
    pub frame: u64,
    events_done: Vec<bool>,
    blasts_active: Vec<bool>,
    probes: Vec<ProbeAcc>,
    next_sample: f64,
    sample_interval: f64,
    /// Host time (s) in the frame tails (static solves on the mirror, rebuilds).
    pub tail_seconds: f64,
    /// Coupled to an engine: its contacts on driven clusters, as GPU load terms.
    filter: crate::coupled::LoadFilter,
    /// Coupled to an engine, in a frame without impact islands (no contact in the stress
    /// solve): substeps at the stress step alone, as the engine-coupled solver takes them
    /// (`coupled.rs`), instead of the reference coupled world's substep, which keeps the
    /// global contact bound in every frame. A step change (measured against the
    /// reference by its refinement); on by default, off for like-for-like comparisons.
    pub stress_step_outside_islands: bool,
    /// Coupled to an engine, in a frame with impact islands: the contact bound over the
    /// contacts that can engage within the frame (the contact plan's chunk pairs,
    /// impactor candidates and ground chunks, each against the stiffest material it can
    /// touch; world_coupling.rs `engaged_contact_dt`) instead of the reference's global
    /// bound, which presses every chunk of the scene with the scene's stiffest material.
    /// A step change (measured against the reference); on by default.
    pub engaged_contact_step: bool,
    /// Coupled to an engine, with `sim.methods.true_step`: each structure's certified
    /// stress step from the dashpots its bonds actually have
    /// (`stability::certified_damped_step`) instead of the oracle's, which pairs the
    /// largest dashpot-to-stiffness ratio of any bond with the stiffest mode. Exact class
    /// (a sharper certificate of the same scheme); on by default.
    pub own_dashpot_step: bool,
    /// Those steps by structure (computed when the world is coupled).
    damped_steps: Vec<f64>,
    /// Measured costs of island frames (seconds, count): per substep, and per segment
    /// set-up (snapshot, contact plan, its bound). They choose how far a contact-bound
    /// segment plans (`step_frame_coupled`).
    cost_substep: (f64, u64),
    cost_segment: (f64, u64),
    /// Coupled to an engine: solve a frame again, with the bodies in islands, when a body
    /// the engine moved fractured during it (the reference's verification). Off: the
    /// fracture stands at the end of the frame (no rewind), and the frame is counted in
    /// `unverified_fractures`.
    pub redo_fractured_frames: bool,
    /// Frames in which a body the engine moved fractured and no redo was taken.
    pub unverified_fractures: u64,
    /// Promoted structures (no redo) and the quiet frames they have left.
    promoted: std::collections::HashMap<usize, u32>,
    /// Clusters left out of the GPU tables at both refreshes of the last frame (so their
    /// mirror state did not change during it): the next frame's
    /// `ReferenceSolver::unchanged`.
    unchanged_next: std::collections::HashSet<u64>,
    /// Island segments solved again at a smaller contact step (a plan made within the
    /// segment needed it): local retries of the stress solve, not frame rewinds.
    pub segment_retries: u64,
    /// The tiered coupling (`set_tiered`, quasi-static mode only): impact islands take
    /// the frame's explicit substeps; every other cluster is settled and takes the
    /// frame's static solve with its cascade (Blast's model outside impacts). A
    /// structure's island promotion lasts `PROMOTE_FRAMES` after a fracture under the
    /// engine's loads only, not after its own island's fractures, so islands end with
    /// their impacts. A model change against the explicit coupling (measured, opt-in).
    pub tiered: bool,
    /// Coupled quasi-static frames: each cluster's (by id) last solve inputs that left its
    /// state unchanged (`static_inputs`), and whether that solve converged.
    static_repeats: std::collections::HashMap<u64, (StaticInputs, bool)>,
}

/// `STRESS_GPU_QS_EVERY_SOLVE=1`: the coupled quasi-static frame without its lossless
/// shortcuts (repeated solves skipped, no rebuild without a damage change).
fn qs_every_solve() -> bool {
    std::env::var("STRESS_GPU_QS_EVERY_SOLVE").is_ok_and(|v| v == "1")
}

/// Frames a structure stays promoted (its clusters island seeds) after its last fracture.
const PROMOTE_FRAMES: u32 = 6;

/// Segments of an island frame, each checked against the contact plans made within it
/// (`step_frame_coupled`): a plan that needs a smaller step costs one segment again.
///
/// `STRESS_GPU_FIXED_HORIZON=1` (switchable, read per segment): every island frame runs
/// these segments, without the cost model's shorter horizons. The cost model weighs
/// wall-clock costs measured so far, so its horizons, and with them the substeps, plans
/// and outcome, differ from run to run; fixed horizons make a run reproducible (the GPU
/// is otherwise deterministic), e.g. to bisect a fault to its frame.
const FRAME_SEGMENTS: usize = 4;

/// Why a scene cannot run on the GPU yet.
pub fn unsupported(scene: &Scene) -> Option<String> {
    if scene.sim.solve_mode == SolveMode::Implicit {
        return Some("implicit solve mode".into());
    }
    // Every switch the GPU would not honour (`methods::TABLE` says why).
    let unhonoured = crate::methods::unhonoured(&scene.sim.methods, scene.sim.solve_mode);
    if !unhonoured.is_empty() {
        let list: Vec<String> = unhonoured
            .iter()
            .map(|c| match c.honoured {
                crate::methods::Honoured::InertWithout(crate::methods::Prerequisite::Method(other)) => format!("{} (with {other})", c.name),
                _ => c.name.to_string(),
            })
            .collect();
        return Some(format!("methods not on the GPU: {}", list.join(", ")));
    }
    if scene.sim.refine_utilization.is_some() {
        return Some("refinement".into());
    }
    if scene.bodies.iter().any(|b| b.chunks.iter().any(|c| c.compound.is_some())) {
        return Some("compound chunks (the GPU contact samples one box or hull per chunk)".into());
    }
    None
}

impl GpuWorld {
    pub fn new(gpu: &Gpu, scene: &Scene) -> Result<GpuWorld, String> {
        if let Some(why) = unsupported(scene) {
            return Err(why);
        }
        // The gravity prestress (world.rs `prestress`) runs on the GPU below, unless the
        // static solves stay on the host (`STRESS_GPU_HOST_STATICS=1`).
        let host_statics = std::env::var("STRESS_GPU_HOST_STATICS").is_ok_and(|v| v == "1");
        let mut shell_scene = scene.clone();
        shell_scene.sim.gravity_prestress &= host_statics;
        let mut shell = World::new(&shell_scene);
        let mirror = shell.solver.clone();
        let loads = WorldLoads::new(scene, &mirror);
        let mut solver = GpuSolver::new(gpu, mirror, Some(loads), shell.impactors.clone())?;
        if scene.sim.gravity_prestress && !host_statics {
            prestress(gpu, &mut solver, scene.sim.solve_mode)?;
        }
        // The shell keeps a light placeholder; the mirror is swapped in when needed.
        shell.solver = ReferenceSolver::new(&Scene { bodies: Vec::new(), ..scene.clone() });
        let mut w = GpuWorld {
            scene: scene.clone(),
            shell,
            solver,
            frame: 0,
            events_done: vec![false; scene.events.len()],
            blasts_active: vec![false; scene.loads.len()],
            probes: vec![ProbeAcc::default(); scene.probes.len()],
            next_sample: 0.0,
            sample_interval: scene.sim.sample_interval.unwrap_or(scene.sim.frame_dt),
            tail_seconds: 0.0,
            filter: crate::coupled::LoadFilter::default(),
            stress_step_outside_islands: true,
            engaged_contact_step: true,
            own_dashpot_step: true,
            damped_steps: Vec::new(),
            cost_substep: (0.0, 0),
            cost_segment: (0.0, 0),
            redo_fractured_frames: true,
            unverified_fractures: 0,
            promoted: std::collections::HashMap::new(),
            unchanged_next: std::collections::HashSet::new(),
            segment_retries: 0,
            tiered: false,
            static_repeats: std::collections::HashMap::new(),
        };
        let values: Vec<f64> = (0..scene.probes.len()).map(|p| w.probe_value(p)).collect();
        w.record(w.solver.mirror.time, &values, true);
        Ok(w)
    }

    /// Run `f` with the mirror in the shell `World` (e.g. its energy terms,
    /// `World::mechanical_energy`; the mirror's bonds must be synced for those).
    pub fn with_shell<R>(&mut self, f: impl FnOnce(&World) -> R) -> R {
        self.shell.impactors = self.solver.impactors.clone();
        self.shell.contact.dissipated = self.solver.contact_dissipated;
        self.shell.contact.crush = self.solver.crush_energy;
        std::mem::swap(&mut self.shell.solver, &mut self.solver.mirror);
        let r = f(&self.shell);
        std::mem::swap(&mut self.shell.solver, &mut self.solver.mirror);
        r
    }

    /// The world coupled to a rigid-body engine (world_coupling.rs `World::with_engine`):
    /// the engine owns rigid motion and contact except in impact islands, which the GPU
    /// simulates with its own chunk-level contact.
    pub fn with_engine(gpu: &Gpu, scene: &Scene, engine: Box<dyn RigidEngine>) -> Result<GpuWorld, String> {
        if !matches!(scene.sim.solve_mode, SolveMode::Explicit | SolveMode::Adaptive | SolveMode::QuasiStatic) {
            return Err("the engine-coupled world runs the explicit, adaptive or quasi-static solve".into());
        }
        let mut w = GpuWorld::new(gpu, scene)?;
        w.solver.sync(gpu);
        if scene.sim.methods.true_step {
            let m = &w.solver.mirror;
            w.damped_steps = crate::stability::certified_damped_steps(m);
        }
        w.with_shell_mut(|shell| shell.attach_engine(engine));
        Ok(w)
    }

    /// Add an impactor between frames (a projectile fired into the world), coupled to
    /// the engine at once when there is one (world_coupling.rs `add_impactor`). Returns
    /// its index.
    pub fn add_impactor(&mut self, gpu: &Gpu, imp: stress_ref::world::Impactor) -> Result<usize, String> {
        let i = self.with_shell_mut(|w| w.add_impactor(imp));
        self.solver.refresh(gpu)?;
        Ok(i)
    }

    /// Coupled mode: impact islands on (the default) or off (the engine drives every
    /// body and its contacts reach the stress solve only as loads; `World::impact_islands`).
    pub fn set_impact_islands(&mut self, on: bool) {
        self.shell.impact_islands = on;
    }

    /// The tiered coupling (`tiered`); quasi-static mode only.
    pub fn set_tiered(&mut self, on: bool) -> Result<(), String> {
        if on && self.scene.sim.solve_mode != SolveMode::QuasiStatic {
            return Err("the tiered coupling runs in quasi-static mode".into());
        }
        self.tiered = on;
        self.shell.qs_islands = on;
        self.solver.mirror.config.tiered = on;
        Ok(())
    }

    /// Whether the world is coupled to an engine.
    pub fn coupled(&self) -> bool {
        self.shell.engine_name().is_some()
    }

    /// Run `f` on the shell `World` holding the mirror and the impactors (the
    /// reference's own code acting on the GPU world's host state).
    fn with_shell_mut<R>(&mut self, f: impl FnOnce(&mut World) -> R) -> R {
        self.shell.impactors = std::mem::take(&mut self.solver.impactors);
        std::mem::swap(&mut self.shell.solver, &mut self.solver.mirror);
        let r = f(&mut self.shell);
        std::mem::swap(&mut self.shell.solver, &mut self.solver.mirror);
        self.solver.impactors = std::mem::take(&mut self.shell.impactors);
        r
    }

    /// One engine-coupled frame (world_coupling.rs `step_frame_coupled`): islands and the
    /// engine's step by the reference's code on the mirror, the substeps on the GPU,
    /// verification (the frame solved again with more islands when a body the engine
    /// moved fractured), and the write-back to the engine.
    fn step_frame_coupled(&mut self, gpu: &Gpu) -> Result<(), String> {
        let fdt = self.scene.sim.frame_dt;
        let mut forced: std::collections::HashSet<BodyKey> = std::collections::HashSet::new();
        // Promotion (no redo): every cluster of a structure that fractured under the
        // engine's loads is an island seed until the structure has been quiet for
        // `PROMOTE_FRAMES` frames, so the release that follows a fracture is resolved by
        // the stress solve's own dynamics from the next frame on.
        if !self.redo_fractured_frames {
            // A scripted structural event (a support or member removal) this frame, or one
            // still ramping, is a sudden change of the structure's support: its structure
            // is promoted from the frame it starts, before any rigid response to it.
            let t = self.solver.mirror.time;
            for (ei, ev) in self.scene.events.iter().enumerate() {
                let (time, duration, body) = match ev {
                    EventDesc::RemoveChunks { time, duration, body, .. } | EventDesc::RemoveSupports { time, duration, body, .. } => (*time, *duration, body),
                };
                let starts = !self.events_done[ei] && time <= t + fdt + 1e-12;
                let ramping = time <= t && t < time + duration;
                if starts || ramping {
                    if let Some(si) = self.scene.bodies.iter().position(|b| &b.name == body) {
                        self.promoted.insert(si, PROMOTE_FRAMES);
                    }
                }
            }
            self.promoted.retain(|_, left| *left > 0);
            for cl in &self.solver.mirror.clusters {
                if self.promoted.contains_key(&cl.structure) {
                    forced.insert(BodyKey::Cluster(cl.id));
                }
            }
        }
        // Quiet clusters at this frame's first refresh (see `unchanged_next`).
        let mut excluded_mid: std::collections::HashSet<u64>;
        let frozen = loop {
            // The rewind's snapshot only with the redo on; without it, the frame-start
            // cluster of every chunk (for counting fractures of driven bodies) suffices.
            let saved = if self.redo_fractured_frames {
                Some(self.solver.snapshot(gpu))
            } else {
                self.solver.download(gpu);
                None
            };
            let events_at_start = self.solver.mirror.events.len();
            let start_cluster = if saved.is_none() { stress_ref::world::chunk_clusters(&self.solver.mirror) } else { Vec::new() };
            let saved_host = (self.frame, self.events_done.clone(), self.blasts_active.clone(), self.probes.clone(), self.next_sample, self.filter.clone());
            let saved_counts = (self.shell.island_frames, self.shell.max_island_bodies, self.shell.engine_body_frames, self.shell.island_body_frames);
            if self.redo_fractured_frames {
                self.with_shell_mut(|w| w.engine_save());
            }
            self.solver.mirror_check("frame start");
            self.solver.mirror.unchanged = self.unchanged_next.clone();
            let f = self.with_shell_mut(|w| w.coupled_prepare(&forced));
            self.solver.mirror_check("after coupled_prepare (engine step)");
            self.filter.ingest(&self.solver.mirror, f.t0, fdt, &f.impulses);
            // Quasi-static (no islands, nothing integrated explicitly): the substeps only
            // sample the loads for the frame's average. With the engine's loads as their
            // exact frame averages and no scripted loads or events (which take substep
            // times), one substep gives that average exactly.
            let qs_one_step = self.scene.sim.solve_mode == SolveMode::QuasiStatic && f.frozen.is_empty() && self.scene.loads.is_empty() && self.scene.events.is_empty();
            self.solver.extra_terms = if qs_one_step { self.filter.frame_average_terms(&self.solver.mirror, f.t0, fdt) } else { self.filter.terms(&self.solver.mirror) };
            if self.tiered {
                // Island clusters integrate explicitly; the rest keep the state of their
                // last static solve (one released from an island this frame, its own
                // state: the frame's static solve takes it over).
                let active_time = self.solver.mirror.config.active_time;
                for cl in &mut self.solver.mirror.clusters {
                    if cl.bonds.is_empty() {
                        continue;
                    }
                    if !cl.driven {
                        if cl.activity != Activity::Active {
                            cl.activity = Activity::Active;
                            cl.active_timer = active_time;
                        }
                    } else {
                        cl.activity = Activity::Settled;
                    }
                }
            }
            self.solver.refresh(gpu)?;
            excluded_mid = self.solver.excluded_ids().iter().copied().collect();
            let (mut n, mut h) = (f.n, f.h);
            if qs_one_step {
                (n, h) = (1, fdt);
            } else if self.stress_step_outside_islands && f.frozen.is_empty() {
                let dt = self.stress_dt().min(fdt);
                n = (fdt / dt).ceil().max(1.0) as usize;
                h = fdt / n as f64;
            }
            let engaged = self.engaged_contact_step && !f.frozen.is_empty();
            if std::env::var("STRESS_GPU_COUPLED_TRACE").is_ok() {
                eprintln!("coupled frame {}: start, {} frozen{}", self.frame, f.frozen.len(), if engaged { String::new() } else { format!(", {n} substeps of {h:.2e} s") });
            }
            self.begin_frame_loads();
            let mut taken = 0;
            if !engaged {
                self.advance_substeps(gpu, h, n)?;
                taken = n;
            } else {
                // Island frames run in segments, each with its contacts planned over the
                // segment and its step bounding that plan (budgets, and so candidates and
                // the directions pairs can press along, shrink with the horizon). Plans
                // made within a segment (a chunk outran its budget, a split) let other
                // contacts touch: a segment whose step one of them does not bound is
                // solved again from its start at the step it needs.
                let t_end = self.solver.mirror.time + fdt;
                let span = fdt / FRAME_SEGMENTS as f64;
                let mut cap = f64::INFINITY;
                self.solver.stop_at_plan = true;
                while t_end - self.solver.mirror.time > 1e-3 * span {
                    let rest = t_end - self.solver.mirror.time;
                    let mut horizon = if rest < 1.5 * span { rest } else { span };
                    let t_setup = std::time::Instant::now();
                    let snap = self.solver.snapshot_segment(gpu);
                    let host = (self.events_done.clone(), self.blasts_active.clone(), self.probes.clone(), self.next_sample, self.filter.clone());
                    let seg_taken = taken;
                    let stress = self.stress_dt();
                    // A contact-bound segment weighs shorter horizons (host-only plans)
                    // while that lowers the predicted cost per simulated second,
                    // (cost per substep) / step + (cost per segment) / horizon, both as
                    // measured so far: candidates, sweeps and the directions pairs can press
                    // along shrink with the horizon (a fast fragment sweeps past fewer
                    // chunks), at the price of more segments. The segment then ends there.
                    // (`STRESS_GPU_FIXED_HORIZON=1`: segments of FRAME_SEGMENTS per frame,
                    // independent of measured costs, for reproducible runs.)
                    if self.cost_substep.1 > 0 && self.cost_segment.1 > 0 && std::env::var("STRESS_GPU_FIXED_HORIZON").map_or(true, |v| v != "1") {
                        let per_sub = self.cost_substep.0 / self.cost_substep.1 as f64;
                        let per_seg = self.cost_segment.0 / self.cost_segment.1 as f64;
                        let cost = |w: &Self, h: f64, c: f64| per_sub / w.combined_dt(stress, c, fdt) + per_seg / h;
                        let mut contact = self.plan_contact_dt(&self.solver.plan_preview(horizon)).min(cap);
                        while contact < stress {
                            let shorter = 0.5 * horizon;
                            if shorter < self.combined_dt(stress, contact, fdt) {
                                break;
                            }
                            let c = self.plan_contact_dt(&self.solver.plan_preview(shorter)).min(cap);
                            if cost(self, shorter, c) < cost(self, horizon, contact) {
                                (horizon, contact) = (shorter, c);
                            } else {
                                break;
                            }
                        }
                    }
                    // The plan the GPU enforces, and its bound (the step's certificate).
                    self.solver.plan_for(gpu, horizon)?;
                    let view = self.solver.plan_view();
                    let mut contact = self.plan_contact_dt(&view).min(cap);
                    let setup = t_setup.elapsed().as_secs_f64();
                    let seg_end = self.solver.mirror.time + horizon;
                    self.cost_segment.0 += setup;
                    self.cost_segment.1 += 1;
                    let mut violated = None;
                    // The segment, re-stepped whenever the solver stops at a new plan
                    // (nothing has run under it yet).
                    loop {
                        let left = seg_end - self.solver.mirror.time;
                        if left <= 1e-6 * horizon {
                            break;
                        }
                        let dt = self.combined_dt(stress, contact, fdt);
                        let len = (left / dt).ceil().max(1.0) as usize;
                        let step = left / len as f64;
                        self.solver.plan_log = Some(Vec::new());
                        let t_run = std::time::Instant::now();
                        let ran = self.advance_substeps(gpu, step, len)?;
                        if self.solver.mirror_check("after island substeps") {
                            eprintln!("    (segment: {ran} of {len} substeps of {step:.3e} s, stress {stress:.3e}, contact {contact:.3e}, horizon {horizon:.3e})");
                        }
                        if ran > 0 {
                            self.cost_substep.0 += t_run.elapsed().as_secs_f64();
                            self.cost_substep.1 += ran as u64;
                        }
                        taken += ran;
                        let log = self.solver.plan_log.take().unwrap_or_default();
                        let need = log.iter().map(|v| self.plan_contact_dt(v)).fold(f64::INFINITY, f64::min);
                        if std::env::var("STRESS_GPU_COUPLED_TRACE").is_ok() {
                            eprintln!("coupled frame {}: {ran} substeps of {step:.2e} s ({} pairs){}", self.frame, view.pairs.len(), if self.solver.stopped_at_plan { format!(", stopped at a plan needing {need:.2e} s") } else { String::new() });
                        }
                        if self.solver.stopped_at_plan {
                            // Only the last plan stands for what follows; those before it
                            // were run under and must hold for this step.
                            let (last, before) = log.split_last().map_or((f64::INFINITY, f64::INFINITY), |(l, b)| (self.plan_contact_dt(l), b.iter().map(|v| self.plan_contact_dt(v)).fold(f64::INFINITY, f64::min)));
                            if self.combined_dt(stress, before.min(cap), fdt) < step * (1.0 - 1e-9) {
                                violated = Some(before);
                                break;
                            }
                            contact = last.min(cap);
                            continue;
                        }
                        if self.combined_dt(stress, need.min(cap), fdt) < step * (1.0 - 1e-9) {
                            violated = Some(need);
                        }
                        break;
                    }
                    if let Some(need) = violated {
                        if std::env::var("STRESS_GPU_COUPLED_TRACE").is_ok() {
                            eprintln!("coupled frame {}: a plan within a segment needs a contact step of {need:.2e} s: solved again", self.frame);
                        }
                        self.solver.restore(gpu, snap)?;
                        (self.events_done, self.blasts_active, self.probes, self.next_sample, self.filter) = host;
                        // A rejected segment (a plan within it needs a smaller contact step):
                        // a local numerical retry, not a frame rewind.
                        self.segment_retries += 1;
                        taken = seg_taken;
                        cap = cap.min(need);
                        continue;
                    }
                    cap = f64::INFINITY;
                }
                self.solver.stop_at_plan = false;
            }
            self.solver.plan_log = None;
            self.solver.mirror_check("before finish_frame");
            self.finish_frame(gpu, taken)?;
            self.solver.mirror_check("after finish_frame");
            self.solver.sync(gpu);
            let since = &self.solver.mirror.events[events_at_start..];
            let broke = match &saved {
                Some(saved) => fractured_since(saved.mirror(), since, &f.driven),
                None => stress_ref::world::fractured_since_clusters(&start_cluster, &self.solver.mirror, since, &f.driven),
            };
            if std::env::var("STRESS_GPU_COUPLED_TRACE").is_ok() {
                let broken = since.iter().filter(|e| matches!(e, stress_ref::solver::SolverEvent::Broken { .. })).count();
                eprintln!(
                    "coupled frame {} (forced {}): {} substeps of {:.2e} s, {} frozen, {} driven clusters, {} engine impulses, {broken} bonds broken, {} driven fractured; host splits {}, batches {}",
                    self.frame,
                    forced.len(),
                    f.n,
                    f.h,
                    f.frozen.len(),
                    f.driven.len(),
                    f.impulses.len(),
                    broke.len(),
                    self.solver.host_splits,
                    self.solver.profile.batches
                );
            }
            if broke.is_empty() || broke.is_subset(&forced) || !self.redo_fractured_frames {
                if !broke.is_empty() && !broke.is_subset(&forced) {
                    // No redo (`redo_fractured_frames` off): the fracture of a body the
                    // engine moved stands as computed from the engine's loads; counted.
                    self.unverified_fractures += 1;
                }
                if !self.redo_fractured_frames {
                    for left in self.promoted.values_mut() {
                        *left = left.saturating_sub(1);
                    }
                    // A promotion starts where a fracture happened under the engine's loads
                    // (a driven cluster; fractures inside an island were resolved by its
                    // window) and lasts while the structure keeps breaking anywhere.
                    for e in since {
                        if let stress_ref::solver::SolverEvent::Broken { structure, bond, .. } = e {
                            let chunk = self.solver.mirror.bonds[*structure][*bond].geometry.a;
                            let key = BodyKey::Cluster(start_cluster.get(*structure).and_then(|c| c.get(chunk)).copied().unwrap_or(u64::MAX));
                            if f.driven.contains(&key) || (self.promoted.contains_key(structure) && !self.tiered) {
                                self.promoted.insert(*structure, PROMOTE_FRAMES);
                            }
                        }
                    }
                }
                if self.redo_fractured_frames {
                    self.with_shell_mut(|w| w.engine_discard());
                }
                break f.frozen;
            }
            // The engine's assumption failed for these bodies: solve the frame again with
            // them in islands.
            self.solver.restore(gpu, saved.expect("snapshot taken with the redo on"))?;
            (self.frame, self.events_done, self.blasts_active, self.probes, self.next_sample, self.filter) = saved_host;
            (self.shell.island_frames, self.shell.max_island_bodies, self.shell.engine_body_frames, self.shell.island_body_frames) = saved_counts;
            self.with_shell_mut(|w| w.engine_restore());
            self.shell.redone_frames += 1;
            forced.extend(broke);
        };
        // Driven bodies at the engine's end of the frame; island bodies back to the engine.
        self.solver.download(gpu);
        self.with_shell_mut(|w| {
            w.set_driven_motion(1.0);
            w.coupled_write_back(&frozen);
        });
        self.solver.mirror_check("after write-back");
        self.solver.refresh(gpu)?;
        self.unchanged_next = self.solver.excluded_ids().iter().copied().filter(|id| excluded_mid.contains(id)).collect();
        self.solver.mirror.unchanged.clear();
        Ok(())
    }

    /// The reference's `stable_dt` with each structure's certified step from its own
    /// dashpots (`own_dashpot_step`): the larger of the chunk Gershgorin step and that,
    /// over the clusters integrated explicitly, times the safety.
    fn stress_dt(&self) -> f64 {
        let m = &self.solver.mirror;
        if !self.own_dashpot_step || self.damped_steps.is_empty() {
            return m.stable_dt();
        }
        let mut dt = f64::INFINITY;
        for ci in 0..m.clusters.len() {
            let cl = &m.clusters[ci];
            let settled = (m.config.mode == SolveMode::Adaptive || self.tiered) && cl.activity != stress_ref::solver::Activity::Active;
            if cl.bonds.is_empty() || settled {
                continue;
            }
            let gershgorin = m.chunk_stable_dts(ci).iter().map(|&(_, d)| d).fold(f64::INFINITY, f64::min);
            dt = dt.min(gershgorin.max(self.damped_steps[cl.structure]));
        }
        dt * m.config.courant_safety
    }

    /// The contact bound over a contact plan (`engaged_contact_step`): every pair,
    /// impactor candidate and ground chunk the plan lets touch (world_coupling.rs
    /// `engaged_contact_dt`). The GPU enforces the plan's travel budgets (it re-plans
    /// when a chunk leaves its budget), so the plan bounds every contact while it stands.
    fn plan_contact_dt(&mut self, view: &crate::contacts::PlanView) -> f64 {
        use stress_ref::world::ContactPartner;
        let mut partners: std::collections::BTreeMap<(usize, usize), Vec<ContactPartner>> = std::collections::BTreeMap::new();
        for &(sa, ca, sb, cb) in &view.pairs {
            partners.entry((sa, ca)).or_default().push(ContactPartner::Chunk(sb, cb));
            partners.entry((sb, cb)).or_default().push(ContactPartner::Chunk(sa, ca));
        }
        let mut impactors = Vec::new();
        for (i, list) in view.impactor_chunks.iter().enumerate() {
            for &sc in list {
                partners.entry(sc).or_default().push(ContactPartner::Impactor(i));
            }
            if !list.is_empty() {
                impactors.push((i, list.clone()));
            }
        }
        if self.scene.ground.is_some() {
            for &sc in &view.ground {
                partners.entry(sc).or_default().push(ContactPartner::Ground);
            }
        }
        let chunks: Vec<((usize, usize), Vec<ContactPartner>)> = partners.into_iter().collect();
        self.with_shell_mut(|w| w.engaged_contact_dt(&chunks, &impactors, &view.poses, &view.impactor_poses))
    }

    /// The substep from the stress and contact bounds (`scaled_step_bound` combines
    /// them as the reference does), capped by the frame and `sim.max_substep`.
    fn combined_dt(&self, stress: f64, contact: f64, fdt: f64) -> f64 {
        let dt = if self.scene.sim.methods.scaled_step_bound { 1.0 / (stress.powi(-2) + contact.powi(-2)).sqrt() } else { stress.min(contact) };
        let mut dt = dt.min(fdt);
        if let Some(m) = self.scene.sim.max_substep {
            dt = dt.min(m);
        }
        dt
    }

    /// Apply scripted events due at time `t` (world.rs `process_events`). True if any.
    fn process_events(&mut self, gpu: &Gpu, t: f64) -> bool {
        let mut any = false;
        for ei in 0..self.scene.events.len() {
            if self.events_done[ei] {
                continue;
            }
            let due = match &self.scene.events[ei] {
                EventDesc::RemoveChunks { time, .. } | EventDesc::RemoveSupports { time, .. } => t + 1e-12 >= *time,
            };
            if due {
                self.solver.sync(gpu);
            }
            let m = &mut self.solver.mirror;
            match self.scene.events[ei].clone() {
                EventDesc::RemoveChunks { time, duration, body, chunks } => {
                    if t + 1e-12 < time {
                        continue;
                    }
                    let s = m.structure_index(&body).expect("event body");
                    let sel = select_with_descendants(&chunks, &self.scene, s);
                    m.remove_chunks(s, &sel, duration);
                }
                EventDesc::RemoveSupports { time, duration, body, chunks } => {
                    if t + 1e-12 < time {
                        continue;
                    }
                    let s = m.structure_index(&body).expect("event body");
                    let sel = chunks.select(&self.scene.bodies[s]);
                    m.remove_supports(s, &sel, duration);
                }
            }
            self.events_done[ei] = true;
            any = true;
        }
        any
    }

    /// Whether a blast starts acting at time `t` (its faces are then cached).
    fn blasts_starting(&mut self, t: f64) -> bool {
        let mut any = false;
        for (li, l) in self.scene.loads.iter().enumerate() {
            if let LoadDesc::Blast { time, .. } = l {
                if !self.blasts_active[li] && t + 1e-12 >= *time {
                    self.blasts_active[li] = true;
                    any = true;
                }
            }
        }
        any
    }

    /// Advance one frame (`sim.frame_dt`).
    /// Frames solved again by the coupled frame's verification (a body the engine moved
    /// fractured).
    pub fn redone_frames(&self) -> u64 {
        self.shell.redone_frames
    }

    pub fn step_frame(&mut self, gpu: &Gpu) -> Result<(), String> {
        if self.coupled() {
            return self.step_frame_coupled(gpu);
        }
        let dt = self.with_shell(|w| w.substep_dt());
        let n = (self.scene.sim.frame_dt / dt).round().max(1.0) as usize;
        self.advance(gpu, dt, n)
    }

    /// A frame of `n` substeps of `dt` on the GPU, with its events, probes and tail.
    fn advance(&mut self, gpu: &Gpu, dt: f64, n: usize) -> Result<(), String> {
        self.begin_frame_loads();
        self.advance_substeps(gpu, dt, n)?;
        self.finish_frame(gpu, n)
    }

    fn begin_frame_loads(&mut self) {
        let sum = &mut self.solver.frame_load_sum;
        sum.ensure_shape(&self.solver.mirror);
        sum.clear();
    }

    /// `n` substeps of `dt` within the frame (events, blasts, probes). Returns the
    /// substeps run: fewer when the solver stopped at a contact plan (`stop_at_plan`).
    fn advance_substeps(&mut self, gpu: &Gpu, dt: f64, n: usize) -> Result<usize, String> {
        let mut done = 0;
        while done < n {
            let t = self.solver.mirror.time;
            let changed = self.process_events(gpu, t);
            if changed {
                self.solver.coverage |= crate::coverage::REBUILD_EVENT;
            }
            let blast = self.blasts_starting(t);
            if changed || blast {
                self.solver.sync(gpu);
                if let Some(l) = self.solver.loads.as_mut() {
                    l.topology_version += changed as u64;
                }
                if changed {
                    // The removal may have split clusters: the reference re-forms them
                    // inside `remove_chunks`/`remove_supports`.
                }
                self.solver.rebuild(gpu, 0.0)?;
            }
            // The segment runs to the next substep at which an event or blast is due.
            let mut len = 1;
            while done + len < n {
                let tk = t + len as f64 * dt;
                let due = self.scene.events.iter().enumerate().any(|(ei, e)| {
                    !self.events_done[ei]
                        && match e {
                            EventDesc::RemoveChunks { time, .. } | EventDesc::RemoveSupports { time, .. } => tk + 1e-12 >= *time,
                        }
                }) || self.scene.loads.iter().enumerate().any(|(li, l)| matches!(l, LoadDesc::Blast { time, .. } if !self.blasts_active[li] && tk + 1e-12 >= *time));
                if due {
                    break;
                }
                len += 1;
            }
            let start = self.solver.mirror.substeps as usize;
            let values = self.solver.step(gpu, dt, len)?;
            let ran = self.solver.mirror.substeps as usize - start;
            for k in 0..ran {
                let t = self.solver.time_after(start + k + 1);
                let row: Vec<f64> = values.iter().map(|v| v[k]).collect();
                self.record(t, &row, false);
            }
            done += ran;
            if self.solver.stopped_at_plan {
                break;
            }
        }
        Ok(done)
    }

    /// The end of a frame of `n` substeps: the solve mode's tail (statics, settling).
    fn finish_frame(&mut self, gpu: &Gpu, n: usize) -> Result<(), String> {
        self.frame += 1;
        let tail = std::time::Instant::now();
        // The host's static work needs the mirror's bonds current: always in quasi-static
        // mode, in adaptive mode only when a cluster may settle or the host does the
        // settled fatigue.
        let fdt = self.scene.sim.frame_dt;
        let host_statics = std::env::var("STRESS_GPU_HOST_STATICS").is_ok_and(|v| v == "1");
        let may_settle = self.solver.mirror.clusters.iter().any(|c| c.activity == Activity::Active && !c.bonds.is_empty() && c.active_timer - fdt <= 0.0);
        match self.scene.sim.solve_mode {
            SolveMode::QuasiStatic => self.solver.sync(gpu),
            SolveMode::Adaptive if may_settle || host_statics => self.solver.sync(gpu),
            _ => {}
        }
        let changed = match self.scene.sim.solve_mode {
            SolveMode::QuasiStatic => {
                // world.rs: the frame's average loads, equilibrium with same-step cascade.
                let mut fl = self.solver.frame_load_sum.clone();
                fl.scale(1.0 / n as f64);
                let (converged, structural, solved) = self.solve_static_all(gpu, &fl)?;
                if !converged {
                    self.shell.static_unconverged_frames += 1;
                }
                // Coupled: without a damage change or split only the solved bonds'
                // history moved (the GPU holds the equilibrium): upload it, no rebuild.
                if self.coupled() && !structural && !qs_every_solve() {
                    self.solver.upload_bonds(gpu, &solved);
                    false
                } else {
                    true
                }
            }
            SolveMode::Adaptive if host_statics => {
                let settled = self.settle_quiet_clusters(gpu);
                let (fatigued, structural) = self.advance_settled_fatigue();
                if !(settled || structural) {
                    // Only settled bonds' history moved: no rebuild.
                    self.solver.upload_bonds(gpu, &fatigued);
                }
                settled || structural
            }
            SolveMode::Adaptive => self.solver.adaptive_tail(gpu, fdt)?,
            _ => false,
        };
        if changed {
            if let Some(l) = self.solver.loads.as_mut() {
                l.topology_version += 1;
            }
            self.solver.rebuild(gpu, 0.0)?;
        }
        self.tail_seconds += tail.elapsed().as_secs_f64();
        Ok(())
    }

    /// Static equilibrium of `clusters` (statics.rs `equilibrate_or_keep`): on the GPU, or
    /// with `STRESS_GPU_HOST_STATICS=1` by the reference on the mirror. Whether each
    /// converged.
    fn equilibrate(&mut self, gpu: &Gpu, clusters: &[usize], loads: &ChunkLoads) -> Vec<bool> {
        if std::env::var("STRESS_GPU_HOST_STATICS").is_ok_and(|v| v == "1") {
            let opts = StaticOptions::default();
            return clusters.iter().map(|&ci| self.solver.mirror.equilibrate_or_keep(ci, loads, &opts).converged).collect();
        }
        self.solver.equilibrate(gpu, clusters, loads).iter().map(|e| e.converged).collect()
    }

    /// statics.rs `solve_static_all` with the same-step cascade: every cluster to
    /// equilibrium (one GPU dispatch), damage committed at it, splits, and again until no
    /// bond changes. Whether every solve converged, whether any damage changed or a
    /// cluster split, and the clusters of the first pass.
    fn solve_static_all(&mut self, gpu: &Gpu, loads: &ChunkLoads) -> Result<(bool, bool, Vec<usize>), String> {
        let dt = self.scene.sim.frame_dt;
        let mut converged = true;
        let mut structural = false;
        let mut first = Vec::new();
        // Coupled: a cluster whose solve inputs repeat those of a solve that left its
        // state unchanged is not solved again (`static_inputs`; lossless).
        // (`STRESS_GPU_QS_EVERY_SOLVE=1`, diagnostic: every solve and rebuild, to check
        // that the shortcuts change nothing.)
        let skip_repeats = self.coupled() && !qs_every_solve();
        for pass in 0..StaticOptions::default().max_cascade.max(1) {
            if pass > 0 {
                self.solver.coverage |= crate::coverage::QS_CASCADE;
            }
            // Tiered: island clusters keep their explicit state (they are not in
            // equilibrium); the static solve takes the rest.
            let tiered = self.tiered;
            let clusters: Vec<usize> = (0..self.solver.mirror.clusters.len()).filter(|&ci| !self.solver.mirror.clusters[ci].bonds.is_empty() && !(tiered && !self.solver.mirror.clusters[ci].driven)).collect();
            if pass == 0 {
                first = clusters.clone();
            }
            let results = if skip_repeats {
                let m = &self.solver.mirror;
                let inputs: Vec<StaticInputs> = clusters.iter().map(|&ci| StaticInputs::of(m, ci, loads)).collect();
                let mut results: Vec<Option<bool>> = clusters.iter().zip(&inputs).map(|(&ci, k)| self.static_repeats.get(&m.clusters[ci].id).filter(|(was, _)| was == k).map(|(_, ok)| *ok)).collect();
                let todo: Vec<usize> = clusters.iter().zip(&results).filter(|(_, r)| r.is_none()).map(|(&ci, _)| ci).collect();
                let solved = if todo.is_empty() { Vec::new() } else { self.equilibrate(gpu, &todo, loads) };
                let m = &self.solver.mirror;
                for (k, &ci) in clusters.iter().enumerate() {
                    if let Some(j) = todo.iter().position(|&t| t == ci) {
                        results[k] = Some(solved[j]);
                        // Converged or not (an unconverged solve keeps its start state),
                        // a solve that ended where it started repeats bit for bit.
                        if inputs[k].same_state(m, ci) {
                            self.static_repeats.insert(m.clusters[ci].id, (inputs[k].clone(), solved[j]));
                        } else {
                            self.static_repeats.remove(&m.clusters[ci].id);
                        }
                    }
                }
                results.into_iter().map(|r| r.expect("solved or repeated")).collect()
            } else {
                self.equilibrate(gpu, &clusters, loads)
            };
            let m = &mut self.solver.mirror;
            let mut any = false;
            let mut to_split = Vec::new();
            for (&ci, &ok) in clusters.iter().zip(&results) {
                converged &= ok;
                m.clusters[ci].activity = Activity::Settled;
                if ok {
                    // Static fatigue advances once per call, not per cascade pass.
                    let (changed, disconnected) = m.commit_damage(ci, if pass == 0 { dt } else { 0.0 });
                    any |= changed || disconnected;
                    if disconnected {
                        to_split.push(ci);
                    }
                }
            }
            for &ci in to_split.iter().rev() {
                m.split_cluster(ci);
            }
            if !any {
                break;
            }
            structural = true;
            // Damage (and topology) changed on the mirror: the next pass solves from it.
            if let Some(l) = self.solver.loads.as_mut() {
                l.topology_version += 1;
            }
            self.solver.rebuild(gpu, 0.0)?;
        }
        Ok((converged, structural, first))
    }

    /// world.rs `settle_quiet_clusters` (`GpuSolver::settle_quiet_clusters`).
    fn settle_quiet_clusters(&mut self, gpu: &Gpu) -> bool {
        let host = std::env::var("STRESS_GPU_HOST_STATICS").is_ok_and(|v| v == "1");
        self.solver.settle_quiet_clusters(gpu, self.scene.sim.frame_dt, host)
    }

    /// world.rs `advance_settled_fatigue`: settled clusters' bonds evaluated at their
    /// equilibrium with the frame's duration; damage wakes the cluster, a disconnection
    /// splits it. The clusters evaluated, and whether any woke or split.
    fn advance_settled_fatigue(&mut self) -> (Vec<usize>, bool) {
        let fdt = self.scene.sim.frame_dt;
        let m = &mut self.solver.mirror;
        let mut any = false;
        let mut fatigued = Vec::new();
        for ci in (0..m.clusters.len()).rev() {
            let cl = &m.clusters[ci];
            if cl.activity != Activity::Settled || cl.bonds.is_empty() {
                continue;
            }
            let (changed, disconnected) = m.commit_damage(ci, fdt);
            fatigued.push(ci);
            if changed || disconnected {
                any = true;
                m.clusters[ci].activity = Activity::Active;
                m.clusters[ci].active_timer = m.config.active_time;
            }
            if disconnected {
                m.split_cluster(ci);
            }
        }
        (fatigued, any)
    }

    /// Run the scene to its duration and return the observation.
    pub fn run(&mut self, gpu: &Gpu) -> Result<Observation, String> {
        let frames = (self.scene.sim.duration / self.scene.sim.frame_dt).round() as u64;
        while self.frame < frames {
            self.step_frame(gpu)?;
        }
        self.solver.sync(gpu);
        Ok(self.observation())
    }

    /// The observation (world.rs `observation`, on the GPU state), with the GPU's probes.
    /// The mirror's bonds must be synced (`GpuSolver::sync`; `run` does).
    pub fn observation(&mut self) -> Observation {
        let mut obs = self.with_shell(|w| w.observation());
        obs.solver = "stress-gpu".into();
        for (p, acc) in self.scene.probes.iter().zip(&self.probes) {
            obs.probes.insert(p.name.clone(), acc.series.clone());
        }
        obs
    }

    /// world.rs `record_probes`.
    fn record(&mut self, t: f64, values: &[f64], force_sample: bool) {
        let sample = force_sample || t + 1e-12 >= self.next_sample;
        for (acc, &v) in self.probes.iter_mut().zip(values) {
            if acc.series.t.is_empty() && !force_sample {
                continue;
            }
            if acc.series.t.is_empty() || force_sample {
                acc.lo = v;
                acc.hi = v;
            } else {
                acc.lo = acc.lo.min(v);
                acc.hi = acc.hi.max(v);
            }
            if sample {
                acc.series.push(t, v, acc.lo, acc.hi);
                acc.lo = v;
                acc.hi = v;
            }
        }
        if sample {
            self.next_sample = t + self.sample_interval;
        }
    }

    /// A probe's value from the mirror (world.rs `probe_value`), for the initial sample.
    fn probe_value(&self, p: usize) -> f64 {
        let m = &self.solver.mirror;
        let initial = &self.solver.loads.as_ref().expect("loads").initial_positions;
        match &self.scene.probes[p].kind {
            ProbeKind::ChunkDisplacement { body, chunk, axis } => {
                let s = m.structure_index(body).expect("probe body");
                if !m.chunks[s][*chunk].active {
                    return f64::NAN;
                }
                (m.chunk_position(s, *chunk) - initial[s][*chunk]).dot(Vec3::from_array(*axis).normalized())
            }
            ProbeKind::ChunkVelocity { body, chunk, axis } => {
                let s = m.structure_index(body).expect("probe body");
                if !m.chunks[s][*chunk].active {
                    return f64::NAN;
                }
                m.chunk_velocity(s, *chunk).0.dot(Vec3::from_array(*axis).normalized())
            }
            ProbeKind::SectionForce { body, point, normal, region, component } => {
                let s = m.structure_index(body).expect("probe body");
                let pt = Vec3::from_array(*point);
                let n = Vec3::from_array(*normal).normalized();
                let st = &m.structures[s];
                let (mut f, mut mo) = (Vec3::ZERO, Vec3::ZERO);
                for b in m.bonds[s].iter().filter(|b| b.alive) {
                    let g = &b.geometry;
                    if region.as_ref().is_some_and(|r| !r.contains(g.centroid)) {
                        continue;
                    }
                    let (sa, sb) = ((st.chunks[g.a].center - pt).dot(n), (st.chunks[g.b].center - pt).dot(n));
                    if sa * sb >= 0.0 {
                        continue;
                    }
                    let (fa, ma, fb, mb) = g.chunk_loads(&b.force);
                    let (fc, mc, xc) = if sa > 0.0 { (fa, ma, st.chunks[g.a].center) } else { (fb, mb, st.chunks[g.b].center) };
                    f += fc;
                    mo += mc + (xc - pt).cross(fc);
                }
                match component {
                    SectionComponent::Normal => -f.dot(n),
                    SectionComponent::Force(a) => f.dot(Vec3::from_array(*a).normalized()),
                    SectionComponent::Moment(a) => mo.dot(Vec3::from_array(*a).normalized()),
                }
            }
            ProbeKind::Reaction { body, chunks, axis } => {
                let s = m.structure_index(body).expect("probe body");
                let a = Vec3::from_array(*axis).normalized();
                chunks.select(&self.scene.bodies[s]).into_iter().map(|c| m.reaction_world(s, c).0.dot(a)).sum()
            }
            ProbeKind::ImpactorVelocity { impactor, axis } => {
                let imp = self.solver.impactors.iter().find(|i| &i.name == impactor).expect("probe impactor");
                imp.velocity.dot(Vec3::from_array(*axis).normalized())
            }
            ProbeKind::ImpactorPosition { impactor, axis } => {
                let imp = self.solver.impactors.iter().find(|i| &i.name == impactor).expect("probe impactor");
                imp.pose.position.dot(Vec3::from_array(*axis).normalized())
            }
        }
    }
}

/// world.rs `prestress` on the GPU: supported structures settle under gravity and the
/// scripted loads at t = 0 (all in one solve); in the adaptive and quasi-static modes
/// they start settled.
fn prestress(gpu: &Gpu, solver: &mut GpuSolver, mode: SolveMode) -> Result<(), String> {
    let loads = solver.loads.as_mut().expect("loads").chunk_loads(&solver.mirror, 0.0);
    let m = &solver.mirror;
    let clusters: Vec<usize> = (0..m.clusters.len()).filter(|&ci| m.clusters[ci].anchored && !m.clusters[ci].bonds.is_empty()).collect();
    if clusters.is_empty() {
        return Ok(());
    }
    let results = solver.equilibrate(gpu, &clusters, &loads);
    for (&ci, r) in clusters.iter().zip(&results) {
        if !r.converged {
            return Err(format!("prestress did not converge (cluster {ci}, residual {:.3e})", r.residual));
        }
        if mode != SolveMode::Explicit {
            let m = &mut solver.mirror;
            m.clusters[ci].activity = Activity::Settled;
            let cl = &m.clusters[ci];
            m.clusters[ci].settled_load_norm = cl.chunks.iter().map(|&c| loads.force[cl.structure][c].norm()).sum();
        }
    }
    solver.rebuild(gpu, 0.0)
}

fn select_with_descendants(sel: &stress_ref::scene::ChunkSelector, scene: &Scene, s: usize) -> Vec<usize> {
    let body = &scene.bodies[s];
    let mut out = sel.select(body);
    let mut i = 0;
    while i < out.len() {
        for (k, c) in body.chunks.iter().enumerate() {
            if c.parent == Some(out[i]) && !out.contains(&k) {
                out.push(k);
            }
        }
        i += 1;
    }
    out
}
