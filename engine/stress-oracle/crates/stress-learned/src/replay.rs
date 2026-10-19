//! World-space sidecars. Read-only capture for visualization and explicit,
//! versioned frame-start learned inputs. Never substitute an end snapshot.
use serde::Serialize;
use stress_ref::{math::Quat, scene::ImpactorShape, world::World};

#[derive(Serialize)]
pub struct ReplayFrame {
    pub time: f64,
    pub centers: Vec<[f64; 3]>,
    pub rotations: Vec<[[f64; 3]; 3]>,
    pub cluster_rotations: Vec<[[f64; 3]; 3]>,
    pub active: Vec<bool>,
    pub clusters: Vec<u64>,
    pub velocities: Vec<[f64; 3]>,
    pub angular_velocities: Vec<[f64; 3]>,
    pub mechanical_energy: f64,
    pub dissipated_energy: f64,
    pub impactors: Vec<ReplayImpactor>,
}
#[derive(Serialize)]
pub struct ReplayImpactor {
    pub name: String,
    pub shape: ImpactorShape,
    pub center: [f64; 3],
    pub rotation: [[f64; 3]; 3],
    pub velocity: [f64; 3],
    pub angular_velocity: [f64; 3],
    pub mass: f64,
    pub inertia: [[f64; 3]; 3],
    pub material: stress_ref::material::Material,
    pub crush: Option<(f64, f64)>,
    pub crush_used: f64,
    pub crush_depth: f64,
    pub driven: bool,
}
#[derive(Serialize)]
pub struct Replay {
    pub format: &'static str,
    pub world_state_format: &'static str,
    pub scene: String,
    pub ground: Option<f64>,
    pub blasts: Vec<serde_json::Value>,
    pub frames: Vec<ReplayFrame>,
}
impl Replay {
    pub fn new(world: &World, keys: &[(usize, usize)]) -> Self {
        Self {
            format: "learned-impact-replay/1",
            world_state_format: "learned-world-state/1",
            scene: world.scene.name.clone(),
            ground: world.scene.ground.as_ref().map(|g| g.height),
            blasts: world.scene.loads.iter().filter_map(|l| if let stress_ref::scene::LoadDesc::Blast {position,tnt_mass,time}=l {
                Some(serde_json::json!({"position":position,"tnt_mass":tnt_mass,"time":time}))
            } else {None}).collect(),
            frames: vec![capture(world, keys)],
        }
    }
    pub fn push(&mut self, world: &World, keys: &[(usize, usize)]) {
        self.frames.push(capture(world, keys));
    }
}
pub fn capture(world: &World, keys: &[(usize, usize)]) -> ReplayFrame {
    let sv = &world.solver;
    let mut centers = Vec::new();
    let mut rotations = Vec::new();
    let mut cluster_rotations = Vec::new();
    let mut active = Vec::new();
    let mut clusters = Vec::new();
    let mut velocities = Vec::new();
    let mut angular_velocities = Vec::new();
    for &(s, c) in keys {
        let state = &sv.chunks[s][c];
        let ch = &sv.structures[s].chunks[c];
        let cl = &sv.clusters[state.cluster];
        centers.push(sv.chunk_position(s, c).to_array());
        cluster_rotations.push(cl.rotation().m);
        rotations.push(
            (cl.rotation()
                * Quat::from_axis_angle(state.th, state.th.norm()).to_mat3()
                * ch.rotation)
                .m,
        );
        active.push(state.active);
        clusters.push(cl.id);
        let (v, w) = sv.chunk_velocity(s, c);
        velocities.push(v.to_array());
        angular_velocities.push(w.to_array());
    }
    ReplayFrame {
        time: sv.time,
        centers,
        rotations,
        cluster_rotations,
        active,
        clusters,
        velocities,
        angular_velocities,
        mechanical_energy: world.mechanical_energy(),
        dissipated_energy: world.dissipated_energy(),
        impactors: world
            .impactors
            .iter()
            .map(|p| ReplayImpactor {
                name: p.name.clone(),
                shape: p.shape.clone(),
                center: p.pose.position.to_array(),
                rotation: p.pose.rotation.to_mat3().m,
                velocity: p.velocity.to_array(),
                angular_velocity: p.angular_velocity.to_array(),
                mass: p.mass,
                inertia: p.inertia.m,
                material: p.material.clone(),
                crush: p.crush,
                crush_used: p.crush_used,
                crush_depth: p.crush_depth,
                driven: p.driven,
            })
            .collect(),
    }
}
