//! The convex and joint scenes of `engine-scenes` on WebGPU, held to the same gate as
//! engine-cuda's `scenes.rs` (`engine_scenes::invariants::gate`): the reference the CUDA
//! host is compared with.
use engine_scenes::invariants::{gate, GateHost, JOINT_GAP_MAX};
use engine_schema::{BodyState, StepCounters};
use engine_webgpu::WgpuSim;

struct Host(WgpuSim);

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

fn run(name: &str) {
    let (mut scene, info) = engine_scenes::build_host(name).expect("scene");
    let mut sim = WgpuSim::new(info.capacity.clone()).expect("GPU adapter");
    sim.set_params(scene.params);
    sim.set_materials(&scene.materials);
    sim.set_hulls(&mut scene.hulls);
    sim.add_bodies(&scene.bodies);
    sim.add_colliders(&scene.colliders);
    sim.add_joints(&scene.joints);
    gate(name, &mut Host(sim), &scene.joints, 300, JOINT_GAP_MAX);
}

#[test]
fn convex_pile_10k_gate() {
    run("Convex Pile (10k)");
}

#[test]
fn ragdoll_pile_10k_gate() {
    run("Ragdoll Pile (10k)");
}

#[test]
fn hinge_chains_10k_gate() {
    run("Hinge Chains (10k)");
}
