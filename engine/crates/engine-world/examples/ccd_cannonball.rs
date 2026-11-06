//! Fast rigid bodies fired into a soft-body vehicle's door (`engine/docs/jbeam-semantics-spec.md`
//! §38.4): a steel sphere (6 cm or 15 cm radius) or a steel box (20 cm cube) at 30, 100,
//! 300 and 600 m/s, horizontally into the side of a parked car at door height, surface
//! contact on. Stepped standalone on the CPU reference (`ExplicitBeams::rigid`), on the
//! GPU (`ExplicitBeamsGpu` with the proxy written by the host) and in the World (a dynamic
//! body, the coupling plan, the GPU frame).
//!
//! Per run: the momentum the projectile gives the car (a fraction of its own), the
//! deepest any collision triangle gets inside the projectile, and the outcome: `tunnel`
//! when the projectile ends past the car's far side having given it less than 2 % of its
//! momentum, else `hit`.
//!
//! ```text
//! perl -e 'alarm 900; exec @ARGV' cargo run --release -p engine-world --example ccd_cannonball \
//!     [-- --car sedan1|sedan] [--modes cpu,gpu,world] [--speeds 30,100,300,600]
//! ```

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_softbody::explicit_rigid::{ProxyGeom, ProxySet, ProxyShape, RigidProxy};
use engine_webgpu::explicit_coupling::{read_proxies, write_proxies};
use engine_webgpu::{CouplingCapacity, ExplicitBeamsGpu};
use glam::{Quat, Vec3};
use rubble::*;

const STEEL: f32 = 7850.0;
const STEPS: u32 = 17;

fn arg(k: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1).cloned())
}

fn car(name: &str) -> ExplicitBeams {
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        drag: false,
        turbulence: false,
        triangle_throttle: true,
    };
    let text = match name {
        "sedan1" => {
            use engine_softbody::authoring::{build, style, BuildParams, Resolution};
            build(
                &style("sedan1").expect("style"),
                &BuildParams {
                    resolution: Resolution::parse("medium").unwrap(),
                    strength: 1.0,
                    paint: None,
                },
            )
            .expect("build")
            .truck
        }
        _ => std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
        ))
        .expect("sedan"),
    };
    let mut sim = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    // Long axis along z, lowest node 2 cm above the ground at y = 0, centred on x = 0.
    let b = |s: &ExplicitBeams| {
        s.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| {
            (a.min(n.pos), b.max(n.pos))
        })
    };
    let (lo, hi) = b(&sim);
    if hi.x - lo.x > hi.z - lo.z {
        for n in &mut sim.nodes {
            n.pos = Vec3::new(-n.pos.z, n.pos.y, n.pos.x);
        }
    }
    let (lo, hi) = b(&sim);
    let shift = Vec3::new(-0.5 * (lo.x + hi.x), 0.02 - lo.y, -0.5 * (lo.z + hi.z));
    for n in &mut sim.nodes {
        n.pos += shift;
    }
    sim.ground = 0.0;
    sim.surface_contact = true;
    sim.dissipative_contact = true;
    sim
}

#[derive(Clone, Copy, Debug)]
enum Shot {
    Sphere(f32),
    Box(f32),
}

impl Shot {
    fn mass(self) -> f32 {
        match self {
            Shot::Sphere(r) => STEEL * 4.0 / 3.0 * std::f32::consts::PI * r * r * r,
            Shot::Box(h) => STEEL * 8.0 * h * h * h,
        }
    }
    fn inv_inertia(self) -> Vec3 {
        let m = self.mass();
        match self {
            Shot::Sphere(r) => Vec3::splat(1.0 / (0.4 * m * r * r)),
            Shot::Box(h) => Vec3::splat(1.0 / (m / 3.0 * 2.0 * h * h)),
        }
    }
    fn geom(self) -> ProxyGeom {
        match self {
            Shot::Sphere(r) => ProxyGeom::Sphere { radius: r },
            Shot::Box(h) => ProxyGeom::Box {
                half: Vec3::splat(h),
            },
        }
    }
    fn label(self) -> String {
        match self {
            Shot::Sphere(r) => format!("sphere {:.0} cm", r * 100.0),
            Shot::Box(h) => format!("box {:.0} cm", 2.0 * h * 100.0),
        }
    }
    /// How deep point p is inside the projectile at c (0 outside).
    fn depth(self, c: Vec3, p: Vec3) -> f32 {
        match self {
            Shot::Sphere(r) => (r - (p - c).length()).max(0.0),
            Shot::Box(h) => {
                let d = Vec3::splat(h) - (p - c).abs();
                d.min_element().max(0.0)
            }
        }
    }
}

/// The aim: door height (the car's mid height), its middle along z, 1 m off its +x side.
fn aim(sim: &ExplicitBeams) -> (Vec3, f32, f32) {
    let (lo, hi) = sim.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| {
        (a.min(n.pos), b.max(n.pos))
    });
    (
        Vec3::new(hi.x + 1.0, 0.5 * (lo.y + hi.y), 0.5 * (lo.z + hi.z)),
        lo.x,
        hi.x,
    )
}

/// The deepest any collision triangle gets inside the projectile at c (sampled: the
/// triangle's closest point to c, its vertices and centroid).
fn deepest(sim: &ExplicitBeams, pos: &[Vec3], shot: Shot, c: Vec3) -> f32 {
    let mut m = 0.0f32;
    for t in &sim.triangles {
        let x = t.nodes.map(|i| pos[i as usize]);
        let (q, _) = engine_schema::feature_query::closest_on_triangle(c, x[0], x[1], x[2]);
        for p in [q, x[0], x[1], x[2], (x[0] + x[1] + x[2]) / 3.0] {
            m = m.max(shot.depth(c, p));
        }
    }
    m
}

struct Out {
    transfer: f32,
    depth: f32,
    end_x: f32,
}

fn proxy(shot: Shot, at: Vec3, speed: f32) -> ProxySet {
    let mut set = ProxySet {
        bodies: vec![RigidProxy {
            body: 1,
            pos: at,
            rot: Quat::IDENTITY,
            vel: Vec3::new(-speed, 0.0, 0.0),
            ang_vel: Vec3::ZERO,
            inv_mass: 1.0 / shot.mass(),
            inv_inertia: shot.inv_inertia(),
            share: 1.0,
            count: 0,
        }],
        shapes: vec![ProxyShape::new(
            0,
            Vec3::ZERO,
            Quat::IDENTITY,
            shot.geom(),
            1,
            None,
        )],
        frozen: false,
    };
    set.begin_window();
    set
}

fn frames_for(speed: f32) -> u32 {
    // Long enough to cross 6 m (or 0.1 s).
    ((6.0 / speed / (STEPS as f32 * 0.0005)).ceil() as u32).clamp(4, 12)
}

fn run_cpu(sim: &ExplicitBeams, shot: Shot, speed: f32) -> Out {
    let (at, _, _) = aim(sim);
    let mut cpu = sim.clone();
    cpu.rigid = Some(Box::new(proxy(shot, at, speed)));
    let mut depth = 0.0f32;
    for _ in 0..frames_for(speed) {
        cpu.rigid.as_mut().unwrap().begin_window();
        for _ in 0..STEPS {
            cpu.step();
            let c = cpu.rigid.as_ref().unwrap().bodies[0].pos;
            let pos: Vec<Vec3> = cpu.nodes.iter().map(|n| cpu.origin + n.pos).collect();
            depth = depth.max(deepest(&cpu, &pos, shot, c));
            if std::env::var("CCD_TRACE").is_ok() {
                let b = &cpu.rigid.as_ref().unwrap().bodies[0];
                println!(
                    "x {:.3} v {:?} w {:.1} hits {}",
                    b.pos.x,
                    b.vel,
                    b.ang_vel.length(),
                    cpu.surface_last
                );
            }
        }
    }
    let b = &cpu.rigid.as_ref().unwrap().bodies[0];
    Out {
        transfer: 1.0 - (-b.vel.x / speed),
        depth,
        end_x: b.pos.x,
    }
}

fn run_gpu(sim: &ExplicitBeams, shot: Shot, speed: f32) -> Out {
    let (_, device, queue) = engine_webgpu::default_device().unwrap();
    let (at, _, _) = aim(sim);
    let cap = CouplingCapacity {
        bodies: 2,
        shapes: 2,
        planes: 8,
    };
    let mut gpu =
        ExplicitBeamsGpu::with_coupling(&device, &queue, std::slice::from_ref(sim), Some(cap));
    let mut set = proxy(shot, at, speed);
    gpu.set_coupling_on(true);
    let mut depth = 0.0f32;
    for _ in 0..frames_for(speed) {
        set.begin_window();
        write_proxies(&gpu, &queue, 0, &set).unwrap();
        gpu.frame(STEPS);
        read_proxies(&gpu, &device, &queue, 0, &mut set);
        let pos = gpu.positions();
        depth = depth.max(deepest(sim, &pos[0], shot, set.bodies[0].pos));
    }
    Out {
        transfer: 1.0 - (-set.bodies[0].vel.x / speed),
        depth,
        end_x: set.bodies[0].pos.x,
    }
}

fn run_world(sim: &ExplicitBeams, shot: Shot, speed: f32) -> Out {
    let (at, _, _) = aim(sim);
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef {
            gravity: [0.0, engine_softbody::explicit_beams::GRAVITY, 0.0],
            enable_sleep: false,
            ..WorldDef::default()
        },
    )
    .unwrap();
    // The vehicles' clock starts a World step after they are added: the shot starts two
    // steps' travel farther out, so it meets the car inside the vehicles' windows.
    let at = at + Vec3::X * (2.0 * speed * w.solver().dt);
    let b = w
        .create_body(&BodyDef {
            linear_velocity: [-speed, 0.0, 0.0],
            ..BodyDef::dynamic(at.to_array())
        })
        .unwrap();
    let def = ShapeDef {
        density: STEEL,
        ..ShapeDef::default()
    };
    match shot {
        Shot::Sphere(r) => w.create_sphere_shape(b, &def, [0.0; 3], r).unwrap(),
        Shot::Box(h) => w.create_box_shape(b, &def, [h; 3]).unwrap(),
    };
    w.add_explicit_vehicles(std::slice::from_ref(sim), &ExplicitVehiclesDef::default())
        .unwrap();
    let mut depth = 0.0f32;
    let mut last = Vec3::from(at);
    let frames = ((6.0 / speed / w.solver().dt).ceil() as u32 + 2).clamp(4, 14);
    for _ in 0..frames {
        w.step().unwrap();
        w.poll_blocking().unwrap();
        let st = w.read_body_state(b).unwrap();
        last = Vec3::from(st.transform.position);
        let pos = w.explicit_vehicles().unwrap().positions();
        depth = depth.max(deepest(sim, &pos[0], shot, last));
    }
    let v = w.read_body_state(b).unwrap().linear_velocity;
    Out {
        transfer: 1.0 - (-v[0] / speed),
        depth,
        end_x: last.x,
    }
}

fn main() {
    let name = arg("--car").unwrap_or_else(|| "sedan1".into());
    let modes = arg("--modes").unwrap_or_else(|| "cpu,gpu,world".into());
    let speeds: Vec<f32> = arg("--speeds")
        .unwrap_or_else(|| "30,100,300,600".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let sim = car(&name);
    let (_, far, _) = aim(&sim);
    println!(
        "{name}: {} nodes, {} triangles; far side x = {far:.2}",
        sim.nodes.len(),
        sim.triangles.len()
    );
    println!(
        "| car | mode | projectile | m/s | outcome | momentum given | deepest (mm) | end x (m) |"
    );
    println!("|---|---|---|---|---|---|---|---|");
    for shot in [Shot::Sphere(0.06), Shot::Sphere(0.15), Shot::Box(0.1)] {
        for &v in &speeds {
            for mode in modes.split(',') {
                let o = match mode {
                    "cpu" => run_cpu(&sim, shot, v),
                    "gpu" => run_gpu(&sim, shot, v),
                    "world" => run_world(&sim, shot, v),
                    _ => panic!("--modes cpu,gpu,world"),
                };
                let outcome = if o.end_x < far && o.transfer < 0.02 {
                    "tunnel"
                } else {
                    "hit"
                };
                println!(
                    "| {name} | {mode} | {} | {v:.0} | {outcome} | {:.1} % | {:.1} | {:.2} |",
                    shot.label(),
                    100.0 * o.transfer,
                    1000.0 * o.depth,
                    o.end_x
                );
            }
        }
    }
}
