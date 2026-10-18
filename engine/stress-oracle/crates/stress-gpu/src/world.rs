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

mod resident_frame;

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
    /// Whether a fracture under the engine's loads promotes its structure (no redo; on by
    /// default, `Profile::promote_fractures`, `STRESS_PROMOTE=0|1`).
    pub promote_fractures: bool,
    /// Routed resident frames (`routing.slang`, M5): with the tiered coupling and a
    /// resident engine, a frame runs resident unless the last frame's routing check found
    /// an impactor that may seed an island (`route_next`); such a frame takes the coupled
    /// path, whose host routing decides exactly. On by default (`STRESS_ROUTED_RESIDENT=0`:
    /// every tiered frame coupled).
    pub routed_resident: bool,
    /// The next frame may have an island seed (the last frame's routing check).
    pub(crate) route_next: bool,
    /// The last frame ran resident (the next coupled frame takes the engine's motion first).
    pub(crate) last_resident: bool,
    /// The impact window (s; 0 off, `STRESS_IMPACT_WINDOW`): a routed frame with an island
    /// runs the island explicitly for this long only, then the engine steps the rest of the
    /// frame resident with whatever the impact broke off (forward, one tick).
    pub impact_window: f64,
    /// With the impact window: the frame's time before the estimated first contact runs
    /// resident, so the explicit part starts at the contact (`STRESS_IMPACT_WINDOW_LEAD=1`).
    pub impact_window_lead: bool,
    /// Frames that ran coupled (not resident).
    pub coupled_frames: u64,
    /// The routing check's kernel and buffers (`routing.slang`).
    pub(crate) routing: Option<crate::routing::Routing>,
    /// The GPU's predictive punch (`punch.slang`, `STRESS_IMPACT_PUNCH=gpu`).
    pub(crate) punch: Option<crate::routing::Punch>,
    /// Clusters left out of the GPU tables at both refreshes of the last frame (so their
    /// mirror state did not change during it): the next frame's
    /// `ReferenceSolver::unchanged`.
    unchanged_next: std::collections::HashSet<u64>,
    /// Island segments solved again at a smaller contact step (a plan made within the
    /// segment needed it): local retries of the stress solve, not frame rewinds.
    pub segment_retries: u64,
    /// Host seconds setting up island segments (snapshot, contact plan, its bound).
    pub segment_setup_seconds: f64,
    /// The tiered coupling (`set_tiered`, quasi-static mode only): impact islands take
    /// the frame's explicit substeps; every other cluster is settled and takes the
    /// frame's static solve with its cascade (Blast's model outside impacts). A
    /// structure's island promotion lasts `PROMOTE_FRAMES` after a fracture under the
    /// engine's loads only, not after its own island's fractures, so islands end with
    /// their impacts. A model change against the explicit coupling (measured, opt-in).
    pub tiered: bool,
    /// Host seconds in the coupled frame's phases: the engine's step and the islands
    /// (`coupled_prepare`), the solver refresh, and the write-back with its refresh.
    /// The fast quasi-static frame (`STRESS_QS_FAST`, opt-in): in an engine-coupled
    /// quasi-static frame with no islands and no scripted loads (the solve works in each
    /// cluster's body frame, its loads from the poses the engine wrote back),
    /// the static solves and their damage commit stay on the GPU (`equilibrate_resident`,
    /// `commit_resident`) with the frame-average engine loads computed on the host; the
    /// tables are rebuilt and the mirror synced only when a cluster splits. f32 commits
    /// against the host path's f64 ones: gated by outcome, not bit for bit.
    pub fast_qs: bool,
    fast_now: bool,
    /// The resident frame (`with_resident_engine`, `resident_frame.rs`): the engine
    /// stepped and its contacts turned into static loads on the GPU, no host round trip.
    resident_engine: Option<std::rc::Rc<std::cell::RefCell<dyn crate::resident::ResidentEngine>>>,
    /// The engine's part table on the GPU (M3's body creation), built at the first split.
    part_table: Option<std::rc::Rc<crate::gpu_split::PartTable>>,
    /// Impactors added that the solver's tables do not hold yet (`add_impactor` in a
    /// resident frame).
    impactors_stale: bool,
    resident: Option<resident_frame::ResidentState>,
    /// The collider table's lookups of the current layout (resident_frame.rs).
    table_lookup: Option<resident_frame::TableLookup>,
    /// A frame's GPU splits the engine has adopted but the host's mirror and layout not yet
    /// followed (resident_frame.rs `flush_pending`), and whether the engine's write-back waits
    /// with them.
    pending_splits: Vec<resident_frame::PendingSplit>,
    pending_write_back: bool,
    /// Frames taken on the resident path.
    pub resident_frames: u64,
    pub prepare_seconds: f64,
    pub refresh_seconds: f64,
    pub write_back_seconds: f64,
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
            promote_fractures: std::env::var("STRESS_PROMOTE").map_or(true, |v| v != "0"),
            routed_resident: std::env::var("STRESS_ROUTED_RESIDENT").map_or(true, |v| v != "0"),
            route_next: true,
            last_resident: false,
            impact_window_lead: std::env::var("STRESS_IMPACT_WINDOW_LEAD").is_ok_and(|v| v == "1"),
            impact_window: std::env::var("STRESS_IMPACT_WINDOW").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0),
            coupled_frames: 0,
            routing: None,
            punch: None,
            unchanged_next: std::collections::HashSet::new(),
            segment_retries: 0,
            segment_setup_seconds: 0.0,
            tiered: false,
            fast_qs: std::env::var("STRESS_QS_FAST").is_ok_and(|v| v == "1"),
            fast_now: false,
            resident_engine: None,
            part_table: None,
            impactors_stale: false,
            resident: None,
            table_lookup: None,
            pending_splits: Vec::new(),
            pending_write_back: false,
            resident_frames: 0,
            prepare_seconds: 0.0,
            refresh_seconds: 0.0,
            write_back_seconds: 0.0,
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
        w.solver.quiet_quasi_static = scene.sim.solve_mode == SolveMode::QuasiStatic;
        // `STRESS_RESIDENT_ISLANDS=1` (opt-in): island segments planned on the GPU, their
        // snapshots without the chunk state or bond download (`GpuSolver::resident_contacts`).
        w.solver.resident_contacts = std::env::var("STRESS_RESIDENT_ISLANDS").is_ok_and(|v| v == "1");
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
        self.flush_pending(gpu)?;
        let i = self.with_shell_mut(|w| w.add_impactor(imp));
        // The resident quasi-static frame takes impactors as the engine's bodies (and their
        // properties from the host's list): the solver's tables need no refresh for one.
        // Any other frame refreshes first (`impactors_stale`).
        if self.resident_now() {
            self.impactors_stale = true;
        } else {
            self.solver.refresh(gpu)?;
        }
        Ok(i)
    }

    /// Coupled mode: impact islands on (the default) or off (the engine drives every
    /// body and its contacts reach the stress solve only as loads; `World::impact_islands`).
    /// A capability profile's solver switches (`stress_ref::capability`): its per-tick
    /// static solves and whether impact islands run. The scene's switches (`apply_scene`)
    /// and the resident engine (`with_resident_engine`) are the caller's, before the world
    /// is built.
    pub fn apply_profile(&mut self, profile: stress_ref::capability::Profile) {
        profile.apply_config(&mut self.solver.mirror.config);
        self.set_impact_islands(profile.impact_islands());
        if std::env::var("STRESS_PROMOTE").is_err() {
            self.promote_fractures = profile.promote_fractures();
        }
        // The profile's resident frame runs the GPU's split path.
        if profile.resident() {
            crate::GPU_FRAME_PROFILE.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub fn set_impact_islands(&mut self, on: bool) {
        self.shell.impact_islands = on;
    }

    /// Strict mode: fail when a structure's certified explicit stress step is below
    /// `min_step` (s), naming the chunks that set it, so the asset is fixed at its source
    /// (for example its slivers re-exported) instead of the solver taking thousands of
    /// substeps per frame. Checked now, for the structures as they are.
    pub fn set_min_stress_step(&mut self, min_step: f64) -> Result<(), String> {
        let m = &self.solver.mirror;
        let safety = m.config.courant_safety;
        let frame_dt = self.scene.sim.frame_dt;
        let mut report = Vec::new();
        for (ci, cl) in m.clusters.iter().enumerate() {
            if cl.bonds.is_empty() {
                continue;
            }
            let mut dts = m.chunk_stable_dts(ci);
            let gershgorin = dts.iter().map(|&(_, d)| d).fold(f64::INFINITY, f64::min);
            let damped = if self.own_dashpot_step { self.damped_steps.get(cl.structure).copied().unwrap_or(0.0) } else { 0.0 };
            let step = gershgorin.max(damped) * safety;
            if step >= min_step {
                continue;
            }
            dts.sort_by(|a, b| a.1.total_cmp(&b.1));
            let s = &m.structures[cl.structure];
            let worst: Vec<String> = dts
                .iter()
                .take(10)
                .map(|&(c, d)| {
                    let ch = &s.chunks[c];
                    let h = ch.half_extents;
                    format!(
                        "    chunk {c}: {:.2e} s ({:.3} kg, {:.3}x{:.3}x{:.3} m, {}{})",
                        d * safety,
                        ch.mass,
                        2.0 * h.x,
                        2.0 * h.y,
                        2.0 * h.z,
                        ch.material,
                        if ch.groups.is_empty() { String::new() } else { format!(", {}", ch.groups.join(" ")) }
                    )
                })
                .collect();
            report.push(format!(
                "structure '{}' (cluster {}, {} chunks): certified step {:.2e} s < {:.2e} s ({} substeps per {:.4} s frame); its lightest-stepping chunks:\n{}",
                s.name,
                cl.id,
                cl.chunks.len(),
                step,
                min_step,
                (frame_dt / step).ceil(),
                frame_dt,
                worst.join("\n")
            ));
        }
        if !report.is_empty() {
            return Err(format!("strict step: {} structure(s) below the minimum step\n{}", report.len(), report.join("\n")));
        }
        Ok(())
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
        if self.resident_now() {
            self.last_resident = true;
            return self.resident_frame(gpu);
        }
        let fdt = self.scene.sim.frame_dt;
        // `STRESS_IMPACT_PUNCH=1` (a model under measurement): a routed frame breaks the
        // impact's plug out before the engine steps when the punch's force beats the bonds
        // holding it (`World::predict_punch`), then runs resident, the engine's rigid contact
        // meeting the free plug. The prediction takes the engine's motion for a moment (the
        // mirror's poses are restored when nothing breaks out).
        if self.tiered && self.routed_resident && self.resident_engine.is_some() && self.qs_frame_conditions() && std::env::var("STRESS_IMPACT_PUNCH").is_ok_and(|v| v == "1") {
            let m = &self.solver.mirror;
            let saved: Vec<_> = m.clusters.iter().map(|c| (c.pose, c.velocity, c.angular_velocity)).collect();
            let saved_imps: Vec<_> = self.solver.impactors.iter().map(|i| (i.pose, i.velocity, i.angular_velocity)).collect();
            self.resident_sync_motion();
            let bonds = self.with_shell_mut(|w| w.predict_punch(fdt));
            if bonds.is_empty() {
                for (cl, s) in self.solver.mirror.clusters.iter_mut().zip(saved) {
                    (cl.pose, cl.velocity, cl.angular_velocity) = s;
                }
                for (imp, s) in self.solver.impactors.iter_mut().zip(saved_imps) {
                    (imp.pose, imp.velocity, imp.angular_velocity) = s;
                }
            } else {
                self.coupled_frames += 1;
                let t0 = std::time::Instant::now();
                self.solver.download(gpu);
                self.resident_sync_motion();
                let t1 = t0.elapsed().as_secs_f64() * 1e3;
                self.solver.mirror.break_bonds(&bonds);
                let t2 = t0.elapsed().as_secs_f64() * 1e3;
                self.solver.rebuild(gpu, 0.0)?;
                let t3 = t0.elapsed().as_secs_f64() * 1e3;
                for cl in &mut self.solver.mirror.clusters {
                    cl.driven = true;
                }
                self.with_shell_mut(|w| w.coupled_write_back(&[]));
                if std::env::var("STRESS_IMPACT_PUNCH_TRACE").is_ok() {
                    eprintln!("punch frame {}: {} bonds; download {t1:.1} ms, break {:.1}, rebuild {:.1}, engine bodies {:.1}", self.frame, bonds.len(), t2 - t1, t3 - t2, t0.elapsed().as_secs_f64() * 1e3 - t3);
                }
            }
            self.last_resident = true;
            return self.resident_frame(gpu);
        }
        self.coupled_frames += 1;
        // The impact window: the island's explicit part of the frame cut to the window, the
        // rest a resident frame (`impact_window`).
        let mut window = self.tiered && self.routed_resident && self.impact_window > 0.0 && self.resident_engine.is_some() && self.qs_frame_conditions();
        if std::mem::take(&mut self.last_resident) {
            // After resident frames the mirror's motion is the GPU islands' at the last
            // frame's start: every body's taken from the engine (a read), into the tables.
            self.solver.download(gpu);
            self.with_shell_mut(|w| w.take_engine_motion());
            self.solver.refresh(gpu)?;
        }
        // The frame's time before the window (a resident lead, `impact_window_lead`).
        let mut lead = 0.0;
        if window {
            // From the frame's start to the window past the first contact (estimated early).
            // `STRESS_IMPACT_WINDOW_NEW_ONLY=1` (a measurement): a window only for a new
            // approach (pairs already overlapping, an impact that went on past its window,
            // left out); a frame with none runs resident, the engine taking the contact.
            let new_only = std::env::var("STRESS_IMPACT_WINDOW_NEW_ONLY").is_ok_and(|v| v == "1");
            let first = self.with_shell_mut(|w| w.first_contact_time_with(fdt, new_only));
            if new_only && first >= fdt {
                self.last_resident = true;
                return self.resident_frame(gpu);
            }
            // `impact_window_lead`: the frame's time before the estimated contact stepped
            // resident (rigid motion only: nothing touches the island yet), the explicit
            // part starting at the contact.
            if self.impact_window_lead && first > 1e-4 && first + self.impact_window < 0.9 * fdt {
                lead = first;
                self.set_frame_dt(lead);
                self.last_resident = true;
                let t_lead = std::time::Instant::now();
                let r = self.resident_frame(gpu);
                if std::env::var("STRESS_GPU_COUPLED_TRACE").is_ok() {
                    eprintln!("impact window frame {}: lead (resident) {:.1} ms", self.frame - 1, t_lead.elapsed().as_secs_f64() * 1e3);
                }
                self.frame -= 1;
                if let Err(e) = r {
                    self.set_frame_dt(fdt);
                    return Err(e);
                }
                self.last_resident = false;
                // The lead's GPU splits followed first (their bookkeeping waits for the next
                // frame otherwise), so the mirror is the GPU's layout.
                self.flush_pending(gpu)?;
                self.solver.download(gpu);
                self.with_shell_mut(|w| w.take_engine_motion());
                self.solver.refresh(gpu)?;
            }
            let len = first - lead + self.impact_window;
            if std::env::var("STRESS_GPU_COUPLED_TRACE").is_ok() {
                eprintln!("impact window frame {}: first contact at {:.2} ms, resident lead {:.2} ms, explicit part {:.2} ms", self.frame, first * 1e3, lead * 1e3, len.min(fdt - lead) * 1e3);
            }
            if lead + len < 0.9 * fdt {
                self.set_frame_dt(len);
            } else if lead > 0.0 {
                self.set_frame_dt(fdt - lead);
                window = false;
            } else {
                window = false;
            }
        }
        let t_window = std::time::Instant::now();
        let r = self.step_frame_tiered(gpu);
        if window && std::env::var("STRESS_GPU_COUPLED_TRACE").is_ok() {
            eprintln!("impact window frame {}: explicit part {:.1} ms", self.frame - 1, t_window.elapsed().as_secs_f64() * 1e3);
        }
        if !window && lead > 0.0 {
            self.set_frame_dt(fdt);
        }
        if window {
            let w = self.scene.sim.frame_dt;
            self.set_frame_dt(fdt - lead - w);
            let t_rest = std::time::Instant::now();
            let rest = r.and_then(|_| {
                self.last_resident = true;
                self.resident_frame(gpu)
            });
            if std::env::var("STRESS_GPU_COUPLED_TRACE").is_ok() {
                eprintln!("impact window frame {}: rest (resident) {:.1} ms", self.frame - 1, t_rest.elapsed().as_secs_f64() * 1e3);
            }
            self.set_frame_dt(fdt);
            // One tick: the window and the rest are one frame.
            self.frame -= 1;
            return rest;
        }
        r?;
        // The routed resident frame (M5): the next frame's routing predicted exactly now,
        // from the state it starts with.
        if self.tiered && self.routed_resident && self.resident_engine.is_some() {
            let fdt = self.scene.sim.frame_dt;
            let none = std::collections::HashSet::new();
            self.route_next = !self.promoted.is_empty() || self.with_shell_mut(|w| w.predict_islands(fdt, &none)) > 0;
        }
        Ok(())
    }

    /// The frame's duration, in the world and its shell (an impact window's parts).
    fn set_frame_dt(&mut self, dt: f64) {
        self.scene.sim.frame_dt = dt;
        self.shell.scene.sim.frame_dt = dt;
    }

    /// The coupled frame on the host's routing (`step_frame_coupled` without a resident one).
    fn step_frame_tiered(&mut self, gpu: &Gpu) -> Result<(), String> {
        if self.impactors_stale {
            self.impactors_stale = false;
            self.solver.refresh(gpu)?;
        }
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
            self.fast_now = self.fast_frame();
            let saved = if self.redo_fractured_frames {
                Some(self.solver.snapshot(gpu))
            } else {
                if !self.fast_now {
                    self.solver.download(gpu);
                }
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
            let t_prepare = std::time::Instant::now();
            let f = self.with_shell_mut(|w| w.coupled_prepare(&forced));
            self.prepare_seconds += t_prepare.elapsed().as_secs_f64();
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
            let t_refresh = std::time::Instant::now();
            if !self.fast_now {
                self.solver.refresh(gpu)?;
            }
            self.refresh_seconds += t_refresh.elapsed().as_secs_f64();
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
                    self.segment_setup_seconds += setup;
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
                            if self.promote_fractures && (f.driven.contains(&key) || (self.promoted.contains_key(structure) && !self.tiered)) {
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
        let t_back = std::time::Instant::now();
        if !self.fast_now {
            self.solver.download(gpu);
        }
        self.with_shell_mut(|w| {
            w.set_driven_motion(1.0);
            w.coupled_write_back(&frozen);
        });
        self.solver.mirror_check("after write-back");
        // Quasi-static with no island and every cluster anchored: the write-back moved
        // nothing the tables hold, and the next frame refreshes them anyway (or, fast,
        // keeps them).
        let unmoved = self.scene.sim.solve_mode == SolveMode::QuasiStatic && frozen.is_empty() && self.solver.mirror.clusters.iter().all(|c| c.anchored);
        if !unmoved && !self.fast_now {
            self.solver.refresh(gpu)?;
        }
        self.unchanged_next = self.solver.excluded_ids().iter().copied().filter(|id| excluded_mid.contains(id)).collect();
        self.write_back_seconds += t_back.elapsed().as_secs_f64();
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
        // Implicit contact (`STRESS_REF_IMPLICIT_CONTACT=1`) sets no step bound.
        let contact = if stress_ref::world::implicit_contact() { f64::INFINITY } else { contact };
        // `STRESS_IMPACT_WINDOW_STEP_SCALE=k` (a lossy measurement, off): an island frame's
        // certified stress step times k (past the certificate: may lose stability).
        let stress = stress * std::env::var("STRESS_IMPACT_WINDOW_STEP_SCALE").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0);
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
        // A frame other than a resident one follows any GPU splits first.
        if !(self.coupled() && self.resident_now()) {
            self.flush_pending(gpu)?;
        }
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
            SolveMode::QuasiStatic if self.fast_now => {}
            SolveMode::QuasiStatic => self.solver.sync(gpu),
            SolveMode::Adaptive if may_settle || host_statics => self.solver.sync(gpu),
            _ => {}
        }
        let changed = match self.scene.sim.solve_mode {
            SolveMode::QuasiStatic if self.fast_now => {
                self.fast_static_tail(gpu)?;
                false
            }
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

    /// Whether this frame takes the fast quasi-static path (`fast_qs`).
    fn fast_frame(&self) -> bool {
        self.fast_qs && self.fast_qs_conditions()
    }

    /// The scene and coupling conditions of the fast quasi-static frame (and of the
    /// resident one).
    fn fast_qs_conditions(&self) -> bool {
        self.qs_frame_conditions() && !self.tiered
    }

    /// `fast_qs_conditions` but the tiered coupling (a routed resident frame's).
    pub(crate) fn qs_frame_conditions(&self) -> bool {
        let m = &self.solver.mirror;
        self.frame > 0
            && self.coupled()
            && !self.redo_fractured_frames
            && self.scene.sim.solve_mode == SolveMode::QuasiStatic
            && self.scene.loads.is_empty()
            && self.scene.events.is_empty()
            && self.scene.probes.is_empty()
            && !m.config.methods.tangent_statics
    }

    /// The frame-average engine loads (`extra_terms`, constant point loads in a
    /// quasi-static one-step frame) as chunk loads, as the GPU's `chunk_loads` records
    /// them: the force, and its moment about the chunk's displaced centre.
    fn host_frame_loads(&self) -> ChunkLoads {
        let m = &self.solver.mirror;
        let mut l = ChunkLoads::new(m);
        for t in &self.solver.extra_terms {
            let crate::loads::Function::Constant(v) = t.function else { continue };
            if t.kind != crate::loads::LOAD_POINT {
                continue;
            }
            let cl = &m.clusters[m.chunks[t.structure][t.chunk].cluster];
            let f = t.dir * v;
            let lever = cl.pose.transform_vector(t.arm - m.chunks[t.structure][t.chunk].u);
            l.force[t.structure][t.chunk] += f;
            l.torque[t.structure][t.chunk] += lever.cross(f);
        }
        l
    }

    /// The fast quasi-static frame's tail (`fast_qs`): solves and commits on the GPU, the
    /// cascade until no damage grows; a split syncs the mirror, splits and rebuilds.
    fn fast_static_tail(&mut self, gpu: &Gpu) -> Result<(), String> {
        let trace = std::env::var("STRESS_GPU_QS_TRACE").is_ok();
        let t0 = std::time::Instant::now();
        let loads = self.host_frame_loads();
        if trace {
            eprintln!("fast qs: loads {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        let dt = self.scene.sim.frame_dt;
        let max = StaticOptions::default().max_cascade.max(1);
        let mut unconverged = false;
        for pass in 0..max {
            let excluded = self.solver.excluded_ids().to_vec();
            let m = &self.solver.mirror;
            let clusters: Vec<usize> = (0..m.clusters.len()).filter(|&ci| !m.clusters[ci].bonds.is_empty() && excluded.binary_search(&m.clusters[ci].id).is_err()).collect();
            let t_pass = std::time::Instant::now();
            let converged = self.solver.equilibrate_resident(gpu, &clusters, &loads);
            let t_solve = t_pass.elapsed().as_secs_f64() * 1e3;
            unconverged |= converged.iter().any(|&c| !c);
            let solved: Vec<usize> = clusters.iter().zip(&converged).filter(|(_, &c)| c).map(|(&ci, _)| ci).collect();
            for &ci in &clusters {
                self.solver.mirror.clusters[ci].activity = Activity::Settled;
            }
            let (damaged, disconnected) = self.solver.commit_resident(gpu, &solved, if pass == 0 { dt } else { 0.0 });
            if trace {
                eprintln!("fast qs pass {pass}: {} clusters ({} converged), solve {t_solve:.1} ms, commit {:.1} ms, {} damaged, {} disconnected", clusters.len(), solved.len(), t_pass.elapsed().as_secs_f64() * 1e3 - t_solve, damaged.len(), disconnected.len());
            }
            if !disconnected.is_empty() {
                // A split: the host's mirror, split and tables.
                self.solver.sync(gpu);
                let mut pending = disconnected.clone();
                pending.sort_unstable();
                for &ci in pending.iter().rev() {
                    self.solver.mirror.split_cluster(ci);
                }
                if let Some(l) = self.solver.loads.as_mut() {
                    l.topology_version += 1;
                }
                self.solver.rebuild(gpu, 0.0)?;
                continue;
            }
            if damaged.is_empty() {
                break;
            }
        }
        if unconverged {
            self.shell.static_unconverged_frames += 1;
        }
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
        // Pass 0's clusters whose solve repeated (from `static_repeats`), and of those the
        // ones its damage commit left exactly as they were: their next solve repeats too,
        // so the tiered coupling may leave them out of the GPU tables (`static_stable`).
        let mut cached: Vec<usize> = Vec::new();
        let mut stable: Vec<u64> = Vec::new();
        for pass in 0..StaticOptions::default().max_cascade.max(1) {
            if pass > 0 {
                self.solver.coverage |= crate::coverage::QS_CASCADE;
            }
            // Tiered: island clusters keep their explicit state (they are not in
            // equilibrium); the static solve takes the rest.
            let tiered = self.tiered;
            // Quiet clusters (left out of the GPU tables) neither: their solve would repeat.
            let excluded = self.solver.excluded_ids();
            let clusters: Vec<usize> = (0..self.solver.mirror.clusters.len())
                .filter(|&ci| {
                    let cl = &self.solver.mirror.clusters[ci];
                    !cl.bonds.is_empty() && !(tiered && !cl.driven) && !(skip_repeats && excluded.binary_search(&cl.id).is_ok())
                })
                .collect();
            if pass == 0 {
                first = clusters.clone();
            }
            // STRESS_GPU_QS_TRACE=1: each cascade pass's time and work on stderr.
            let trace = std::env::var("STRESS_GPU_QS_TRACE").is_ok();
            let t_pass = std::time::Instant::now();
            let results = if skip_repeats {
                let m = &self.solver.mirror;
                let inputs: Vec<StaticInputs> = clusters.iter().map(|&ci| StaticInputs::of(m, ci, loads)).collect();
                let mut results: Vec<Option<bool>> = clusters.iter().zip(&inputs).map(|(&ci, k)| self.static_repeats.get(&m.clusters[ci].id).filter(|(was, _)| was == k).map(|(_, ok)| *ok)).collect();
                if pass == 0 {
                    cached = clusters.iter().zip(&results).filter(|(_, r)| r.is_some()).map(|(&ci, _)| ci).collect();
                }
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
                    if changed || disconnected {
                        continue;
                    }
                }
                // A repeated solve (converged or not) whose commit left its inputs as they
                // were: the next one repeats too.
                if pass == 0 && skip_repeats && cached.contains(&ci) {
                    let id = m.clusters[ci].id;
                    if self.static_repeats.get(&id).is_some_and(|(was, _)| *was == StaticInputs::of(m, ci, loads)) {
                        stable.push(id);
                    }
                }
            }
            let t_solved = t_pass.elapsed().as_secs_f64() * 1e3;
            for &ci in to_split.iter().rev() {
                m.split_cluster(ci);
            }
            if trace {
                let ok = results.iter().filter(|&&r| r).count();
                eprintln!("qs pass {pass}: {} clusters ({ok} converged), solve and commit {t_solved:.1} ms, damage {any}, {} splits", clusters.len(), to_split.len());
            }
            if !any {
                break;
            }
            // A later pass may change any cluster: none is stable.
            stable.clear();
            structural = true;
            // Damage (and topology) changed on the mirror: the next pass solves from it.
            if let Some(l) = self.solver.loads.as_mut() {
                l.topology_version += 1;
            }
            let t_rebuild = std::time::Instant::now();
            self.solver.rebuild(gpu, 0.0)?;
            if trace {
                eprintln!("qs pass {pass}: rebuild {:.1} ms", t_rebuild.elapsed().as_secs_f64() * 1e3);
            }
        }
        stable.sort_unstable();
        self.solver.static_stable = stable;
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
