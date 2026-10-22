//! Runs a scene with the reference [`World`], one frame at a time, and keeps a compact
//! single-precision copy of every snapshot that will be rendered.
//!
//! Rendering happens after the whole run, so colour scales can be fixed over the
//! complete recording (and across compared variants) and the time-series strip can
//! show the full curves.

use std::collections::BTreeMap;
use std::time::Instant;

use stress_ref::math::{Mat3, Vec3};
use stress_ref::observation::ProbeSeries;
use stress_ref::scene::{ImpactorShape, ProbeKind, Scene, SectionComponent};
use stress_ref::snapshot::{symmetric_eigenvalues, BondStatus, Snapshot};
use stress_ref::world::World;

use crate::camera::Aabb;

/// One chunk box with the fields the views colour by.
#[derive(Clone, Debug)]
pub struct ChunkRec {
    pub center: [f32; 3],
    /// Rotation matrix rows (world = R * local).
    pub rotation: [[f32; 3]; 3],
    pub half_extents: [f32; 3],
    pub cluster: u64,
    pub anchored: bool,
    pub velocity: [f32; 3],
    pub deformation: [f32; 3],
    pub von_mises: f32,
    /// Dominant principal stress: the principal value of largest magnitude, signed
    /// (Pa, tension positive), so compression shows as negative.
    pub principal: f32,
    pub utilization: f32,
    /// Faces of a hull chunk (chunk frame), shared by every frame; `None` for a box.
    pub hull: Option<HullMesh>,
}

/// Outward normal and counter-clockwise polygon of each face of a hull chunk.
pub type HullMesh = std::sync::Arc<Vec<([f32; 3], Vec<[f32; 3]>)>>;

/// Hull meshes per (body, chunk), built once.
fn hull_meshes(world: &World) -> Vec<Vec<Option<HullMesh>>> {
    let f = |v: Vec3| [v.x as f32, v.y as f32, v.z as f32];
    world
        .solver
        .structures
        .iter()
        .map(|st| {
            st.chunks
                .iter()
                .map(|ch| {
                    ch.hull.as_ref().map(|h| {
                        std::sync::Arc::new(h.faces.iter().map(|face| (f(face.normal), face.vertices.iter().map(|&i| f(h.vertices[i])).collect())).collect())
                    })
                })
                .collect()
        })
        .collect()
}

/// A damaged (cracked or broken) bond patch; intact bonds are not kept.
#[derive(Clone, Debug)]
pub struct BondRec {
    pub centroid: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub width: [f32; 2],
    pub broken: bool,
    /// `max(damage, crush)`.
    pub damage: f32,
}

#[derive(Clone, Debug)]
pub struct ImpactorRec {
    pub shape: ImpactorShape,
    pub center: [f32; 3],
    pub rotation: [[f32; 3]; 3],
}

#[derive(Clone, Debug)]
pub struct Frame {
    pub time: f64,
    pub chunks: Vec<ChunkRec>,
    pub bonds: Vec<BondRec>,
    pub impactors: Vec<ImpactorRec>,
    pub broken_bonds: usize,
    pub mechanical: f64,
    pub dissipated: f64,
}

/// A recorded run.
pub struct Recording {
    pub scene: Scene,
    pub frames: Vec<Frame>,
    /// Probe series of the scene (full sampling resolution) with their units.
    pub probes: BTreeMap<String, (ProbeSeries, &'static str)>,
    /// Bounds of the initial chunks and impactors.
    pub bounds: Aabb,
    /// Smallest chunk half extent of the initial state (sets overlay depth bias).
    pub min_half_extent: f64,
    /// Median chunk size (edge length), for the deformation exaggeration.
    pub typical_chunk: f64,
    /// Wall-clock seconds the simulation took.
    pub wall_seconds: f64,
}

#[derive(Clone, Debug)]
pub struct RecordOptions {
    /// Keep every `every`-th simulated frame (the first and last are always kept).
    pub every: usize,
    /// Simulated time to run to (s).
    pub duration: f64,
    /// Prefix for progress lines on stderr (empty = quiet).
    pub label: String,
    /// Rigid-body engine.
    pub engine: Engine,
    /// `Engine::Learned`: the model's checkpoint and device (`cpu` | `mps`).
    pub checkpoint: Option<std::path::PathBuf>,
    pub model_device: String,
    /// The model's Python package directory with its `.venv` (default: this checkout's).
    pub model_python: Option<std::path::PathBuf>,
}

/// Which rigid-body engine moves the bodies: the standalone world's own (the oracle) or
/// rubble's AVBD GPU engine (`stress_rubble::RubbleEngine`, `World::with_engine`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Engine {
    #[default]
    Standalone,
    Rubble,
    /// rubble's engine, every routed island's frame from the learned model (closed loop).
    Learned,
    /// The same handoff with the oracle as the model (P0's boundary ceiling, the "perfect
    /// model"): what the learned model would give if it were exact.
    Ceiling,
    /// Harness A: the standalone oracle with every routed island's frame from the learned
    /// model (`World::step_frame_island_provider`; the model's decoded state applied to the
    /// island on a copy of the frame-start world, as `learned_p0 --host a`).
    ALearned,
    /// The CPU `blast_parity` profile (quasi-static stress, P1's QS baseline), standalone.
    Qs,
}

impl Engine {
    /// The routed islands' frames come from a provider at the 1/60 s call boundary.
    pub fn model_coupled(self) -> bool {
        matches!(self, Engine::Learned | Engine::Ceiling | Engine::ALearned)
    }
}

impl std::str::FromStr for Engine {
    type Err = String;
    fn from_str(s: &str) -> Result<Engine, String> {
        match s {
            "standalone" | "oracle" | "reference" => Ok(Engine::Standalone),
            "rubble" if cfg!(feature = "rubble") => Ok(Engine::Rubble),
            "rubble" => Err("engine rubble: build stress-viz with --features rubble".into()),
            "learned" | "model" if cfg!(feature = "learned") => Ok(Engine::Learned),
            "ceiling" if cfg!(feature = "learned") => Ok(Engine::Ceiling),
            "a_learned" | "harness_a" if cfg!(feature = "learned") => Ok(Engine::ALearned),
            "learned" | "model" | "ceiling" | "a_learned" | "harness_a" => Err(format!("engine {s}: build stress-viz with --features learned")),
            "qs" | "blast_parity" => Ok(Engine::Qs),
            other => Err(format!("unknown engine '{other}' (standalone | rubble | learned | ceiling | a_learned | qs)")),
        }
    }
}

pub fn new_world(scene: &Scene, engine: Engine) -> World {
    try_new_world(scene, engine).expect("world")
}

/// The world for `scene` with `engine` as its rigid-body engine.
pub fn try_new_world(scene: &Scene, engine: Engine) -> Result<World, String> {
    match engine {
        Engine::Standalone | Engine::ALearned => Ok(World::new(scene)),
        Engine::Qs => {
            let mut s = scene.clone();
            stress_ref::capability::Profile::BlastParity.apply_scene(&mut s);
            let mut w = World::new(&s);
            stress_ref::capability::Profile::BlastParity.apply_config(&mut w.solver.config);
            Ok(w)
        }
        #[cfg(feature = "rubble")]
        Engine::Rubble => {
            use stress_ref::scene::SolveMode;
            if !matches!(scene.sim.solve_mode, SolveMode::Explicit | SolveMode::Adaptive) {
                return Err(format!("the rubble coupling runs the explicit or adaptive solve, not {:?}", scene.sim.solve_mode));
            }
            // rubble's gravity is along its up axis only (the oracle's z).
            let engine = stress_rubble::RubbleEngine::new(Vec3::from_array(scene.gravity)).map_err(|e| format!("rubble engine: {e}"))?;
            Ok(World::with_engine(scene, Box::new(engine)))
        }
        #[cfg(not(feature = "rubble"))]
        Engine::Rubble => Err("engine rubble: build stress-viz with --features rubble".into()),
        Engine::Learned | Engine::Ceiling => try_new_world(scene, Engine::Rubble),
    }
}

fn f3(a: [f64; 3]) -> [f32; 3] {
    [a[0] as f32, a[1] as f32, a[2] as f32]
}

fn m3(m: [[f64; 3]; 3]) -> [[f32; 3]; 3] {
    [f3(m[0]), f3(m[1]), f3(m[2])]
}

/// The principal stress of largest magnitude, with its sign.
fn dominant_principal(stress: [[f64; 3]; 3]) -> f64 {
    let [e1, _, e3] = symmetric_eigenvalues(&Mat3 { m: stress });
    if e1.abs() >= e3.abs() {
        e1
    } else {
        e3
    }
}

fn compact(s: &Snapshot, hulls: &[Vec<Option<HullMesh>>]) -> Frame {
    let chunks = s
        .chunks
        .iter()
        .map(|c| ChunkRec {
            center: f3(c.center),
            rotation: m3(c.rotation),
            half_extents: f3(c.half_extents),
            cluster: c.cluster,
            anchored: c.anchored,
            velocity: f3(c.velocity),
            deformation: f3(c.deformation),
            von_mises: c.von_mises as f32,
            principal: dominant_principal(c.stress) as f32,
            utilization: c.utilization as f32,
            hull: hulls[c.body][c.chunk].clone(),
        })
        .collect();
    let bonds = s
        .bonds
        .iter()
        .filter(|b| b.status != BondStatus::Intact)
        .map(|b| BondRec {
            centroid: f3(b.centroid),
            normal: f3(b.normal),
            tangent: f3(b.tangent),
            width: [b.width[0] as f32, b.width[1] as f32],
            broken: b.status == BondStatus::Broken,
            damage: b.damage.max(b.crush) as f32,
        })
        .collect();
    let impactors = s
        .impactors
        .iter()
        .map(|i| ImpactorRec { shape: i.shape.clone(), center: f3(i.center), rotation: m3(i.rotation) })
        .collect();
    Frame {
        time: s.time,
        chunks,
        bonds,
        impactors,
        broken_bonds: s.bonds.iter().filter(|b| b.status == BondStatus::Broken).count(),
        mechanical: s.energy.mechanical,
        dissipated: s.energy.dissipated,
    }
}

/// Unit of a probe's values, from its kind.
fn probe_unit(kind: &ProbeKind) -> &'static str {
    match kind {
        ProbeKind::ChunkDisplacement { .. } | ProbeKind::ImpactorPosition { .. } => "m",
        ProbeKind::ChunkVelocity { .. } | ProbeKind::ImpactorVelocity { .. } => "m/s",
        ProbeKind::SectionForce { component: SectionComponent::Moment(_), .. } => "N·m",
        _ => "N",
    }
}

/// Captures compact frames of a running world: the hull meshes are built once, and the
/// initial bounds and chunk sizes are measured from the first frame.
pub struct Recorder {
    hulls: Vec<Vec<Option<HullMesh>>>,
    /// Bounds of the initial chunks and impactors.
    pub bounds: Aabb,
    /// Smallest chunk half extent of the initial state (sets overlay depth bias).
    pub min_half_extent: f64,
    /// Median chunk size (edge length), for the deformation exaggeration.
    pub typical_chunk: f64,
}

impl Recorder {
    /// A recorder for `world` and its first frame.
    pub fn new(world: &World) -> (Recorder, Frame) {
        let first = world.snapshot();
        let mut bounds = Aabb::EMPTY;
        let mut halves = Vec::with_capacity(first.chunks.len());
        for c in &first.chunks {
            let (center, r, h) = (Vec3::from_array(c.center), c.rotation, c.half_extents);
            // World half-extent of the rotated box along each axis.
            let ext = Vec3::new(
                (0..3).map(|k| r[0][k].abs() * h[k]).sum(),
                (0..3).map(|k| r[1][k].abs() * h[k]).sum(),
                (0..3).map(|k| r[2][k].abs() * h[k]).sum(),
            );
            bounds.include(center - ext);
            bounds.include(center + ext);
            halves.extend_from_slice(&h);
        }
        for i in &first.impactors {
            let r = match i.shape {
                ImpactorShape::Sphere { radius } => Vec3::splat(radius),
                ImpactorShape::Box { half_extents } => Vec3::splat(Vec3::from_array(half_extents).norm()),
            };
            bounds.include(Vec3::from_array(i.center) - r);
            bounds.include(Vec3::from_array(i.center) + r);
        }
        halves.sort_by(f64::total_cmp);
        let min_half_extent = halves.first().copied().unwrap_or(0.05);
        let typical_chunk = 2.0 * halves.get(halves.len() / 2).copied().unwrap_or(0.05);
        let hulls = hull_meshes(world);
        let frame = compact(&first, &hulls);
        (Recorder { hulls, bounds, min_half_extent, typical_chunk }, frame)
    }

    /// The current state of `world` as a frame.
    pub fn capture(&self, world: &World) -> Frame {
        compact(&world.snapshot(), &self.hulls)
    }
}

/// Simulates `scene` and records the frames to render.
/// Steps one frame with the run's engine: the world's own step, or (learned, ceiling) the
/// closed loop where each routed island's frame comes from a provider.
struct FrameStepper {
    engine: Engine,
    #[cfg(feature = "learned")]
    model: Option<stress_learned::provider_client::ModelProvider>,
    calls: usize,
    call_ms: f64,
}

impl FrameStepper {
    fn new(scene: &Scene, opts: &RecordOptions) -> FrameStepper {
        #[cfg(feature = "learned")]
        let model = matches!(opts.engine, Engine::Learned | Engine::ALearned).then(|| {
            let ckpt = opts.checkpoint.clone().expect("engine learned needs --checkpoint");
            let mut spec = stress_learned::provider_client::ModelSpec::new(ckpt);
            spec.device = opts.model_device.clone();
            if let Some(d) = &opts.model_python {
                spec.python_dir = d.clone();
            }
            let dir = std::env::temp_dir().join(format!("stress-viz-model-{}-{}", std::process::id(), opts.label.replace(['/', ' ', ':'], "_")));
            stress_learned::provider_client::ModelProvider::spawn(&spec, &dir, &scene.name).unwrap_or_else(|e| panic!("model provider: {e}"))
        });
        #[cfg(not(feature = "learned"))]
        let _ = scene;
        FrameStepper {
            engine: opts.engine,
            #[cfg(feature = "learned")]
            model,
            calls: 0,
            call_ms: 0.0,
        }
    }

    fn step(&mut self, world: &mut World) {
        if !self.engine.model_coupled() {
            world.step_frame();
            return;
        }
        #[cfg(feature = "learned")]
        {
            let frame = world.frame as i64;
            let t = Instant::now();
            let engine = self.engine;
            let model = &mut self.model;
            let mut provider = |view: &World, keys: &stress_ref::world::IslandKeys| -> Result<stress_ref::world::DecodedState, String> {
                match engine {
                    Engine::Ceiling => {
                        let mut oracle = view.standalone_copy();
                        oracle.step_frame();
                        Ok(view.decoded_from_oracle(&oracle, keys, frame, 0))
                    }
                    _ => model.as_mut().expect("model").outcome(view, keys, frame),
                }
            };
            let keys = if engine == Engine::ALearned {
                let mut a_provider = |start: &World, m: &stress_ref::world::IslandMembers| -> Result<stress_ref::world::IslandState, String> {
                    // The island's view (every other body driven) for the call, the decoded
                    // state applied on a copy of the frame-start world, the island's state taken.
                    let mut view = start.clone();
                    let island: std::collections::HashSet<u64> =
                        view.solver.clusters.iter().filter(|c| c.chunks.first().is_some_and(|&k| m.chunks.contains(&(c.structure, k)))).map(|c| c.id).collect();
                    for cl in &mut view.solver.clusters {
                        cl.driven = !island.contains(&cl.id);
                    }
                    for (i, imp) in view.impactors.iter_mut().enumerate() {
                        imp.driven = !m.impactors.contains(&i);
                    }
                    let keys = view.routed_island_keys();
                    let d = provider(&view, &keys)?;
                    let mut x = start.clone();
                    x.solver.reset_peak_utilizations();
                    x.apply_decoded(&d)?;
                    Ok(x.extract_island(m))
                };
                let m = world.step_frame_island_provider(&mut a_provider).unwrap_or_else(|e| panic!("frame {frame}: {e}"));
                stress_ref::world::IslandKeys { chunks: m.chunks.iter().map(|&(s, c)| stress_ref::world::stable_key(s, c)).collect(), impactors: m.impactors.iter().map(|&i| i as i64).collect(), ..Default::default() }
            } else {
                world.step_frame_with_provider(&mut provider).unwrap_or_else(|e| panic!("frame {frame}: {e}"))
            };
            if !keys.chunks.is_empty() || !keys.impactors.is_empty() {
                self.calls += 1;
                self.call_ms += t.elapsed().as_secs_f64() * 1e3;
            }
        }
    }
}

pub fn record(scene: &Scene, opts: &RecordOptions) -> Recording {
    let start = Instant::now();
    let mut world = new_world(scene, opts.engine);
    let (recorder, first) = Recorder::new(&world);

    let frame_dt = scene.sim.frame_dt;
    let total = (opts.duration / frame_dt).round().max(1.0) as u64;
    let every = opts.every.max(1) as u64;
    let mut frames = vec![first];
    let mut last_report = Instant::now();
    let mut stepper = FrameStepper::new(scene, opts);
    // Stop on simulated time: a frame never advances less than one substep.
    while world.solver.time < opts.duration - 0.5 * frame_dt {
        stepper.step(&mut world);
        let done = world.solver.time >= opts.duration - 0.5 * frame_dt;
        if world.frame.is_multiple_of(every) || done {
            frames.push(recorder.capture(&world));
        }
        if !opts.label.is_empty() && (last_report.elapsed().as_secs_f64() > 5.0 || done) {
            last_report = Instant::now();
            eprintln!(
                "{}: simulated frame {}/{} (t = {:.4} s, {:.0} s wall)",
                opts.label,
                world.frame,
                total,
                world.solver.time,
                start.elapsed().as_secs_f64()
            );
        }
    }
    if stepper.engine.model_coupled() && !opts.label.is_empty() {
        eprintln!("{}: {} model calls, {:.0} ms per frame with a call (call record + model + apply)",
                  opts.label, stepper.calls, stepper.call_ms / stepper.calls.max(1) as f64);
    }
    if !opts.label.is_empty() {
        // One machine-readable line per run (for the comparison videos' index).
        let speeds: Vec<String> = world.impactors.iter().map(|i| format!("{:.4}", i.velocity.norm())).collect();
        eprintln!(
            "stress-viz summary: {{\"label\": {:?}, \"t\": {:.6}, \"broken_bonds\": {}, \"pieces\": {}, \"impactor_speeds\": [{}], \"wall_s\": {:.1}}}",
            opts.label,
            world.solver.time,
            world.solver.broken_bond_count(),
            world.solver.clusters.len(),
            speeds.join(", "),
            start.elapsed().as_secs_f64()
        );
    }
    let obs = world.observation();
    let probes = scene
        .probes
        .iter()
        .filter_map(|p| obs.probes.get(&p.name).map(|s| (p.name.clone(), (s.clone(), probe_unit(&p.kind)))))
        .collect();
    Recording {
        scene: scene.clone(),
        frames,
        probes,
        bounds: recorder.bounds,
        min_half_extent: recorder.min_half_extent,
        typical_chunk: recorder.typical_chunk,
        wall_seconds: start.elapsed().as_secs_f64(),
    }
}

impl Recording {
    /// Mean simulated time between recorded frames.
    pub fn frame_interval(&self) -> f64 {
        match (self.frames.first(), self.frames.last()) {
            (Some(a), Some(b)) if self.frames.len() > 1 => (b.time - a.time) / (self.frames.len() - 1) as f64,
            _ => self.scene.sim.frame_dt,
        }
    }

    pub fn end_time(&self) -> f64 {
        self.frames.last().map_or(0.0, |f| f.time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dominant_principal_keeps_the_sign_of_the_largest_magnitude() {
        let uniaxial = |s: f64| [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, s]];
        assert_eq!(dominant_principal(uniaxial(-5.0)), -5.0);
        assert_eq!(dominant_principal(uniaxial(3.0)), 3.0);
        // Pure shear: equal magnitudes, tension wins the tie.
        let shear = [[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        assert!((dominant_principal(shear) - 2.0).abs() < 1e-12);
    }
}
