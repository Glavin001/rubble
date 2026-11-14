//! The vehicle lab on the CPU reference: the explicit solver (`ExplicitBeams`) with its
//! rigid proxies (`explicit_rigid::ProxySet`, the coupling's reference) and **no rigid
//! solver**. Between their contacts with the car the lab's bodies move on their own:
//! dynamic ones fall under gravity and stop on the flat scene's ground, kinematic ones
//! keep their velocity, static ones stand. Bodies never touch each other.
//!
//! That is the World's coupling exactly for what the lab measures on one car (a
//! projectile's strike, a plate's press, the mountain's static rock); it is not the
//! World for piles (a brick stack does not stack).

use super::monitor::{BodyView, CarRead, ProxyFrame, ShapeView, ViewGeom};
use super::{lab_car, Bench, LabGeom, LabOptions, Motion, ObjDef, Rig, Scene, GROUND};
use engine_schema::convex::ConvexHull;
use engine_softbody::explicit_beams::{ExplicitBeams, GRAVITY};
use engine_softbody::explicit_drive::DriveInput;
use engine_softbody::explicit_rigid::{ProxyGeom, ProxySet, ProxyShape, RigidProxy};
use glam::{Quat, Vec3};
use serde_json::Value;

struct Obj {
    key: u32,
    geom: ProxyGeom,
    view: ViewGeom,
    radius: f32,
    pos: Vec3,
    rot: Quat,
    vel: Vec3,
    ang_vel: Vec3,
    inv_mass: f32,
    inv_inertia: Vec3,
    motion: Motion,
}

fn geoms(g: &LabGeom) -> anyhow::Result<(ProxyGeom, ViewGeom)> {
    Ok(match g {
        LabGeom::Box { half } => (
            ProxyGeom::Box { half: *half },
            ViewGeom::Box { half: *half },
        ),
        LabGeom::Sphere { radius } => (
            ProxyGeom::Sphere { radius: *radius },
            ViewGeom::Sphere { radius: *radius },
        ),
        LabGeom::Capsule { radius, half_len } => (
            ProxyGeom::Capsule {
                radius: *radius,
                half_len: *half_len,
            },
            ViewGeom::Capsule {
                radius: *radius,
                half_len: *half_len,
            },
        ),
        LabGeom::Cylinder { .. } | LabGeom::Hull { .. } => {
            let points = match g {
                LabGeom::Cylinder { radius, half_len } => {
                    LabGeom::cylinder_points(*radius, *half_len)
                }
                LabGeom::Hull { points } => points.to_vec(),
                _ => unreachable!(),
            };
            let hull = ConvexHull::from_points(&points)
                .map_err(|e| anyhow::anyhow!("vehicle lab: hull: {e:?}"))?;
            let planes: Vec<(Vec3, f32)> = hull.faces.iter().map(|f| (f.normal, f.d)).collect();
            let radius = hull.vertices.iter().fold(0.0f32, |m, p| m.max(p.length()));
            (
                ProxyGeom::Hull {
                    planes: planes.clone(),
                    radius,
                },
                ViewGeom::Hull { planes },
            )
        }
    })
}

struct CpuRig<'a> {
    objs: &'a mut Vec<Obj>,
    next: &'a mut u32,
    sim: &'a mut ExplicitBeams,
}

impl Rig for CpuRig<'_> {
    fn spawn(&mut self, d: &ObjDef) -> anyhow::Result<u32> {
        let (geom, view) = geoms(&d.geom)?;
        let mass = d.density * d.geom.volume();
        let (inv_mass, inv_inertia) = if d.motion == Motion::Dynamic && mass > 0.0 {
            (1.0 / mass, Vec3::ONE / (d.geom.unit_inertia() * mass))
        } else {
            (0.0, Vec3::ZERO)
        };
        let key = *self.next;
        *self.next += 1;
        self.objs.push(Obj {
            key,
            geom,
            view,
            radius: d.geom.radius(),
            pos: d.pos.as_vec3(),
            rot: d.rot,
            vel: d.vel,
            ang_vel: d.ang_vel,
            inv_mass,
            inv_inertia,
            motion: d.motion,
        });
        Ok(key)
    }

    fn set_velocity(&mut self, key: u32, vel: Vec3) -> anyhow::Result<()> {
        if let Some(o) = self.objs.iter_mut().find(|o| o.key == key) {
            o.vel = vel;
        }
        Ok(())
    }

    fn remove(&mut self, key: u32) {
        self.objs.retain(|o| o.key != key);
    }

    fn set_drive(&mut self, input: DriveInput) {
        self.sim.set_drive_input(input);
    }
}

/// The lab on the CPU reference.
pub struct CpuLab {
    pub bench: Bench,
    pub sim: ExplicitBeams,
    objs: Vec<Obj>,
    next: u32,
    /// The plan's reach past the car's box (m), as the World's coupling margin.
    pub margin: f32,
}

impl CpuLab {
    pub fn new(template: &ExplicitBeams, opts: LabOptions) -> anyhow::Result<Self> {
        anyhow::ensure!(!opts.parked, "vehicle lab: the CPU reference runs one car");
        let sim = lab_car(template, &opts, None);
        let mut lab = Self {
            bench: Bench::new(opts, &sim, 1.0 / 60.0),
            sim,
            objs: Vec::new(),
            next: 1,
            margin: 0.1,
        };
        lab.bench.begin(&mut CpuRig {
            objs: &mut lab.objs,
            next: &mut lab.next,
            sim: &mut lab.sim,
        })?;
        lab.read(&ProxySet::default(), &[]);
        Ok(lab)
    }

    pub fn command(&mut self, cmd: &Value) -> anyhow::Result<Value> {
        self.bench.command(
            &mut CpuRig {
                objs: &mut self.objs,
                next: &mut self.next,
                sim: &mut self.sim,
            },
            cmd,
        )
    }

    fn read(&mut self, set: &ProxySet, which: &[usize]) {
        let o = self.sim.origin;
        let frame = ProxyFrame {
            bodies: set
                .bodies
                .iter()
                .map(|b| BodyView {
                    pos: b.pos - o,
                    rot: b.rot,
                    vel: b.vel,
                    ang_vel: b.ang_vel,
                    inv_mass: b.inv_mass,
                })
                .collect(),
            shapes: set
                .shapes
                .iter()
                .zip(which)
                .map(|(s, &i)| {
                    let b = &set.bodies[s.body as usize];
                    ShapeView {
                        collider: self.objs[i].key,
                        body: s.body as usize,
                        geom: self.objs[i].view.clone(),
                        pos: b.pos - o,
                        rot: b.rot,
                        impulse: s.impulse,
                        peak: s.peak,
                    }
                })
                .collect(),
            planes: 0,
        };
        let pos: Vec<Vec3> = self.sim.nodes.iter().map(|n| n.pos).collect();
        let vel: Vec<Vec3> = self.sim.nodes.iter().map(|n| n.vel).collect();
        let beams: Vec<(bool, f32)> = self.sim.beams.iter().map(|b| (b.broken, b.rest)).collect();
        self.bench.read(
            &CarRead {
                pos: &pos,
                vel: &vel,
                beams: Some(&beams),
                origin: o.as_dvec3(),
            },
            &frame,
        );
    }

    /// One World window (1/60 s).
    pub fn step(&mut self) -> anyhow::Result<()> {
        self.window(None)
    }

    /// One World window, calling `each` after every 2 kHz step of the car (a
    /// measurement's view of the car between steps; `examples/hinge_breaks.rs`). The
    /// steps are the window's own ([`ExplicitBeams::frame`]'s, one by one).
    pub fn step_observed(&mut self, each: &mut dyn FnMut(&ExplicitBeams)) -> anyhow::Result<()> {
        self.window(Some(each))
    }

    fn window(&mut self, each: Option<&mut dyn FnMut(&ExplicitBeams)>) -> anyhow::Result<()> {
        self.bench.before_step(&mut CpuRig {
            objs: &mut self.objs,
            next: &mut self.next,
            sim: &mut self.sim,
        })?;
        let dt = self.bench.dt;
        // The window's proxies: every body whose bounding sphere, swept over the window,
        // reaches the car's box grown by the margin and the car's own travel.
        let o = self.sim.origin;
        let (mut lo, mut hi, mut fast) = (Vec3::MAX, Vec3::MIN, 0.0f32);
        // Retired debris is not where the car is (the GPU's cpl_bounds skips it too).
        for (_, n) in self.sim.nodes.iter().enumerate().filter(|(i, _)| !self.sim.is_retired(*i)) {
            lo = lo.min(o + n.pos);
            hi = hi.max(o + n.pos);
            fast = fast.max(n.vel.length());
        }
        let grow = self.margin + fast * dt;
        let mut set = ProxySet::default();
        let mut which = Vec::new();
        for (i, ob) in self.objs.iter().enumerate() {
            let r = ob.radius + grow + ob.vel.length() * dt;
            if (ob.pos + Vec3::splat(r)).cmplt(lo).any()
                || (ob.pos - Vec3::splat(r)).cmpgt(hi).any()
            {
                continue;
            }
            let j = set.bodies.len() as u32;
            set.bodies.push(RigidProxy {
                body: j,
                pos: ob.pos,
                rot: ob.rot,
                vel: ob.vel,
                ang_vel: ob.ang_vel,
                inv_mass: ob.inv_mass,
                inv_inertia: ob.inv_inertia,
                share: 1.0,
                count: 0,
            });
            set.shapes.push(ProxyShape::new(
                j,
                Vec3::ZERO,
                Quat::IDENTITY,
                ob.geom.clone(),
                ob.key,
                None,
            ));
            which.push(i);
        }
        let cap = self.bench.opts.capacity;
        anyhow::ensure!(
            set.shapes.len() <= cap.shapes as usize,
            "vehicle lab: more colliders than the shapes room ({} of {})",
            set.shapes.len(),
            cap.shapes
        );
        set.begin_window();
        self.sim.rigid = Some(Box::new(set));
        match each {
            None => {
                self.sim.frame();
            }
            Some(each) => {
                // `frame`, step by step.
                self.sim.check_wings();
                self.sim.rebase_origin();
                for _ in 0..self.sim.clock.next() {
                    self.sim.step();
                    each(&self.sim);
                }
            }
        }
        let set = *self.sim.rigid.take().expect("the window's proxies");
        let mut proxied = vec![false; self.objs.len()];
        for (b, &i) in set.bodies.iter().zip(&which) {
            let ob = &mut self.objs[i];
            (ob.pos, ob.rot, ob.vel, ob.ang_vel) = (b.pos, b.rot, b.vel, b.ang_vel);
            proxied[i] = true;
        }
        // The lab's stand-in for the rigid solver.
        let flat = self.bench.opts.scene == Scene::Flat;
        for (ob, proxied) in self.objs.iter_mut().zip(proxied) {
            if ob.motion == Motion::Static {
                continue;
            }
            if !proxied {
                ob.pos += ob.vel * dt;
            }
            if ob.motion == Motion::Dynamic {
                ob.vel.y += GRAVITY * dt;
                let rest = GROUND
                    + ob.radius.min(match &ob.view {
                        ViewGeom::Box { half } => half.min_element(),
                        ViewGeom::Sphere { radius } | ViewGeom::Capsule { radius, .. } => *radius,
                        ViewGeom::Hull { .. } => ob.radius,
                    });
                if flat && ob.pos.y < rest && ob.vel.y < 0.0 {
                    ob.pos.y = rest;
                    ob.vel = Vec3::new(0.8 * ob.vel.x, 0.0, 0.8 * ob.vel.z);
                    ob.ang_vel *= 0.8;
                }
            }
        }
        self.read(&set, &which);
        // The debris time to live's decision, for the next window (as the GPU lab's).
        if let Some(r) = self.bench.retire_due.take() {
            self.sim.retired = r;
        }
        anyhow::ensure!(
            !self.bench.car.non_finite,
            "vehicle lab: non-finite node state at t = {:.3} s",
            self.bench.t
        );
        Ok(())
    }
}
