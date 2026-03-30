use glam::{Quat, Vec3, Vec4};
use rubble_broadphase3d::overlap_pairs;
use rubble_math::{Aabb3D, BodyHandle, CollisionEvent, RigidBodyState3D};
use rubble_narrowphase3d::generate_contacts;
use rubble_shapes3d::{compute_aabb, ShapeDesc};
use rubble_solver3d::{integrate, solve_contacts, SolverParams};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct SimConfig {
    pub gravity: Vec3,
    pub dt: f32,
    pub solver_iterations: u32,
    pub friction_default: f32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 1.0 / 60.0,
            solver_iterations: 6,
            friction_default: 0.5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RigidBodyDesc {
    pub shape: ShapeDesc,
    pub position: Vec3,
    pub rotation: Quat,
    pub linear_velocity: Vec3,
    pub mass: f32,
}

pub struct World {
    config: SimConfig,
    states: Vec<RigidBodyState3D>,
    shapes: Vec<ShapeDesc>,
    generations: Vec<u32>,
    free: Vec<u32>,
    live: HashSet<u32>,
    prev_contact_pairs: HashSet<(u32, u32)>,
    events: Vec<CollisionEvent>,
}

impl World {
    pub fn new(config: SimConfig) -> Self {
        Self {
            config,
            states: Vec::new(),
            shapes: Vec::new(),
            generations: Vec::new(),
            free: Vec::new(),
            live: HashSet::new(),
            prev_contact_pairs: HashSet::new(),
            events: Vec::new(),
        }
    }

    pub fn add_body(&mut self, desc: RigidBodyDesc) -> BodyHandle {
        let inv_mass = if desc.mass <= 0.0 {
            0.0
        } else {
            1.0 / desc.mass
        };
        let state = RigidBodyState3D {
            position_inv_mass: desc.position.extend(inv_mass),
            orientation: desc.rotation,
            lin_vel: desc.linear_velocity.extend(0.0),
            ang_vel: Vec4::ZERO,
        };

        if let Some(index) = self.free.pop() {
            let i = index as usize;
            self.states[i] = state;
            self.shapes[i] = desc.shape;
            self.live.insert(index);
            BodyHandle {
                index,
                generation: self.generations[i],
            }
        } else {
            let index = self.states.len() as u32;
            self.states.push(state);
            self.shapes.push(desc.shape);
            self.generations.push(0);
            self.live.insert(index);
            BodyHandle {
                index,
                generation: 0,
            }
        }
    }

    pub fn remove_body(&mut self, handle: BodyHandle) {
        if self.is_valid(handle) {
            self.live.remove(&handle.index);
            self.generations[handle.index as usize] += 1;
            self.free.push(handle.index);
        }
    }

    pub fn step(&mut self) {
        let indices: Vec<usize> = self.live.iter().map(|v| *v as usize).collect();
        let mut remap = HashMap::new();
        let mut compact_states = Vec::with_capacity(indices.len());
        let mut compact_shapes = Vec::with_capacity(indices.len());

        for (compact, &orig) in indices.iter().enumerate() {
            remap.insert(orig as u32, compact as u32);
            compact_states.push(self.states[orig]);
            compact_shapes.push(self.shapes[orig].clone());
        }

        integrate(&mut compact_states, self.config.gravity, self.config.dt);

        let aabbs: Vec<Aabb3D> = compact_states
            .iter()
            .zip(compact_shapes.iter())
            .map(|(s, sh)| compute_aabb(sh, s.position_inv_mass.truncate(), s.orientation))
            .collect();
        let pairs = overlap_pairs(&aabbs);
        let contacts = generate_contacts(
            &pairs,
            &compact_states
                .iter()
                .map(|s| s.position_inv_mass.truncate())
                .collect::<Vec<_>>(),
            &compact_shapes,
        );
        solve_contacts(
            &mut compact_states,
            &contacts,
            SolverParams {
                iterations: self.config.solver_iterations,
                baumgarte: 0.2,
            },
        );

        for (compact, &orig) in indices.iter().enumerate() {
            self.states[orig] = compact_states[compact];
        }

        let current_pairs: HashSet<(u32, u32)> = contacts
            .iter()
            .map(|c| {
                let oa = indices[c.body_a as usize] as u32;
                let ob = indices[c.body_b as usize] as u32;
                if oa < ob {
                    (oa, ob)
                } else {
                    (ob, oa)
                }
            })
            .collect();
        self.events.clear();
        for &(a, b) in current_pairs.difference(&self.prev_contact_pairs) {
            self.events.push(CollisionEvent::Started {
                body_a: BodyHandle {
                    index: a,
                    generation: self.generations[a as usize],
                },
                body_b: BodyHandle {
                    index: b,
                    generation: self.generations[b as usize],
                },
            });
        }
        for &(a, b) in self.prev_contact_pairs.difference(&current_pairs) {
            self.events.push(CollisionEvent::Ended {
                body_a: BodyHandle {
                    index: a,
                    generation: self.generations[a as usize],
                },
                body_b: BodyHandle {
                    index: b,
                    generation: self.generations[b as usize],
                },
            });
        }
        self.prev_contact_pairs = current_pairs;
    }

    pub fn get_body_state(&self, handle: BodyHandle) -> Option<RigidBodyState3D> {
        self.is_valid(handle)
            .then(|| self.states[handle.index as usize])
    }

    pub fn drain_collision_events(&mut self) -> Vec<CollisionEvent> {
        std::mem::take(&mut self.events)
    }

    pub fn body_count(&self) -> u32 {
        self.live.len() as u32
    }

    fn is_valid(&self, handle: BodyHandle) -> bool {
        self.live.contains(&handle.index)
            && self.generations.get(handle.index as usize).copied() == Some(handle.generation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn gravity_matches_analytic_free_fall() {
        let mut world = World::new(SimConfig::default());
        let h = world.add_body(RigidBodyDesc {
            shape: ShapeDesc::Sphere { radius: 0.5 },
            position: Vec3::new(0.0, 10.0, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            mass: 1.0,
        });

        let steps = 100.0;
        for _ in 0..steps as usize {
            world.step();
        }
        let y = world.get_body_state(h).unwrap().position_inv_mass.y;
        let dt = world.config.dt;
        let expected = 10.0 + world.config.gravity.y * dt * dt * (steps * (steps + 1.0) * 0.5);
        assert_relative_eq!(y, expected, epsilon = 1e-4);
    }

    #[test]
    fn collision_events_started_and_ended_once() {
        let mut world = World::new(SimConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let _a = world.add_body(RigidBodyDesc {
            shape: ShapeDesc::Sphere { radius: 1.0 },
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            mass: 1.0,
        });
        let b = world.add_body(RigidBodyDesc {
            shape: ShapeDesc::Sphere { radius: 1.0 },
            position: Vec3::new(1.5, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            mass: 1.0,
        });
        world.step();
        let started = world.drain_collision_events();
        assert!(started
            .iter()
            .any(|e| matches!(e, CollisionEvent::Started { .. })));
        assert!(world.drain_collision_events().is_empty());

        if let Some(mut s) = world.get_body_state(b) {
            s.position_inv_mass.x = 10.0;
            world.states[b.index as usize] = s;
        }
        world.step();
        let ended = world.drain_collision_events();
        assert!(ended
            .iter()
            .any(|e| matches!(e, CollisionEvent::Ended { .. })));
    }

    #[test]
    fn handle_generation_prevents_stale_access() {
        let mut world = World::new(SimConfig::default());
        let h = world.add_body(RigidBodyDesc {
            shape: ShapeDesc::Sphere { radius: 1.0 },
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            mass: 1.0,
        });
        world.remove_body(h);
        let h2 = world.add_body(RigidBodyDesc {
            shape: ShapeDesc::Sphere { radius: 1.0 },
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            mass: 1.0,
        });
        assert_eq!(h.index, h2.index);
        assert_ne!(h.generation, h2.generation);
        assert!(world.get_body_state(h).is_none());
        assert!(world.get_body_state(h2).is_some());
    }
}

#[cfg(test)]
mod e2e_invariants {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn center_of_mass_invariant_without_external_forces() {
        let mut world = World::new(SimConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let h1 = world.add_body(RigidBodyDesc {
            shape: ShapeDesc::Sphere { radius: 1.0 },
            position: Vec3::new(-0.2, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            mass: 2.0,
        });
        let h2 = world.add_body(RigidBodyDesc {
            shape: ShapeDesc::Sphere { radius: 1.0 },
            position: Vec3::new(1.2, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            mass: 1.0,
        });

        let com_before = {
            let a = world
                .get_body_state(h1)
                .unwrap()
                .position_inv_mass
                .truncate();
            let b = world
                .get_body_state(h2)
                .unwrap()
                .position_inv_mass
                .truncate();
            (a * 2.0 + b) / 3.0
        };

        for _ in 0..10 {
            world.step();
        }

        let com_after = {
            let a = world
                .get_body_state(h1)
                .unwrap()
                .position_inv_mass
                .truncate();
            let b = world
                .get_body_state(h2)
                .unwrap()
                .position_inv_mass
                .truncate();
            (a * 2.0 + b) / 3.0
        };

        assert_relative_eq!(com_before.x, com_after.x, epsilon = 1e-5);
        assert_relative_eq!(com_before.y, com_after.y, epsilon = 1e-5);
        assert_relative_eq!(com_before.z, com_after.z, epsilon = 1e-5);
    }

    #[test]
    fn penetration_resolves_monotonically() {
        let mut world = World::new(SimConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let h1 = world.add_body(RigidBodyDesc {
            shape: ShapeDesc::Sphere { radius: 1.0 },
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            mass: 1.0,
        });
        let h2 = world.add_body(RigidBodyDesc {
            shape: ShapeDesc::Sphere { radius: 1.0 },
            position: Vec3::new(1.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            mass: 1.0,
        });

        let mut prev_penetration = 1.0;
        for _ in 0..8 {
            world.step();
            let p1 = world
                .get_body_state(h1)
                .unwrap()
                .position_inv_mass
                .truncate();
            let p2 = world
                .get_body_state(h2)
                .unwrap()
                .position_inv_mass
                .truncate();
            let penetration = (2.0 - p1.distance(p2)).max(0.0);
            assert!(
                penetration <= prev_penetration + 1e-6,
                "penetration increased from {prev_penetration} to {penetration}"
            );
            prev_penetration = penetration;
        }
        assert!(prev_penetration < 1e-2);
    }
}
