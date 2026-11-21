//! The convex and joint scenes of `engine-scenes` on CUDA, through `rigid_host`'s shared
//! joint and hull handling: Convex Pile (10k), Ragdoll Pile (10k), Hinge Chains (10k).
//! The gate is `engine_scenes::invariants::gate`, the same one engine-webgpu's
//! `scene_gate.rs` runs (no cross-backend bit identity: CUDA and Metal round
//! differently). `ENGINE_CUDA_DEVICE=1` runs them; Modal sets it.
use engine_cuda::CudaSim;
use engine_scenes::invariants::{gate, GateHost};
use engine_schema::{BodyState, StepCounters};

struct Host(CudaSim);

impl GateHost for Host {
    fn step(&mut self) -> anyhow::Result<StepCounters> {
        Ok(self.0.step()?)
    }
    fn bodies(&mut self) -> Vec<BodyState> {
        self.0.download_bodies()
    }
    fn validate_coloring(&mut self) -> Result<(), String> {
        self.0.validate_contact_coloring()
    }
    fn submit(&mut self) -> anyhow::Result<()> {
        Ok(self.0.step_submit()?)
    }
    fn drain(&mut self) -> anyhow::Result<StepCounters> {
        Ok(self.0.drain()?)
    }
}

fn run(name: &str, max_gap: f32) {
    if std::env::var("ENGINE_CUDA_DEVICE").as_deref() != Ok("1") {
        eprintln!("skipped: set ENGINE_CUDA_DEVICE=1 on an NVIDIA host");
        return;
    }
    let (scene, info) = engine_scenes::build_host(name).expect("scene");
    let mut sim = CudaSim::new(info.capacity.clone()).expect("CudaSim");
    sim.set_params(scene.params);
    sim.set_materials(&scene.materials).expect("materials");
    sim.set_hulls(&scene.hulls).expect("hulls");
    sim.add_bodies(&scene.bodies);
    sim.add_colliders(&scene.colliders);
    sim.add_joints(&scene.joints).expect("joints");
    eprintln!(
        "{name} on {}: {} bodies, {} joints",
        sim.device_name(),
        scene.bodies.len(),
        scene.joints.len()
    );
    gate(name, &mut Host(sim), &scene.joints, 300, max_gap);
}

#[test]
fn cuda_convex_pile_10k() {
    run(
        "Convex Pile (10k)",
        engine_scenes::invariants::JOINT_GAP_MAX,
    );
}

#[test]
fn cuda_ragdoll_pile_10k() {
    run(
        "Ragdoll Pile (10k)",
        engine_scenes::invariants::JOINT_GAP_MAX,
    );
}

#[test]
fn cuda_hinge_chains_10k() {
    run(
        "Hinge Chains (10k)",
        engine_scenes::invariants::JOINT_GAP_MAX,
    );
}
