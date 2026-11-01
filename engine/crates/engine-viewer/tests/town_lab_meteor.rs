//! Probe (ignored): Vibe Town's pack in the fracture lab's own step plan on Metal, a reference
//! for `rubble::World` (engine-world/tests/town_solver_probe.rs, TOWN_METEOR_R=0.5): one CCD
//! projectile of TOWN_METEOR_KG (default 22 t) straight down at 50 m/s onto house-89.
//!   RUBBLE_TOWN_PACK=/path/vibe-town.json cargo test --release -p engine-viewer \
//!     --test town_lab_meteor -- --ignored --nocapture

use engine_fracture::pack::build_pack_scene;
use engine_schema::JOINT_FLAG_BROKEN;
use engine_viewer::fracture::FractureSim;
use metal::Device;

#[test]
#[ignore]
fn town_meteor_in_the_lab() {
    let Ok(path) = std::env::var("RUBBLE_TOWN_PACK") else { return };
    let (mut scene, _) = build_pack_scene(&std::fs::read(&path).unwrap()).unwrap();
    scene.projectile_mass = std::env::var("TOWN_METEOR_KG").ok().and_then(|v| v.parse().ok()).unwrap_or(22_000.0);
    scene.projectile_arc = false;
    let joints = scene.joints.len();
    let device = Device::system_default().expect("Metal");
    let queue = device.new_command_queue();
    let mut sim = FractureSim::new(&device, &queue, &scene).unwrap();
    let hz = (1.0 / scene.params.dt).round() as u32;
    sim.step(hz).unwrap();
    sim.plan.set_projectile_ccd(true);
    sim.plan.set_projectile_speed_scale(50.0 / sim.plan.shot_speed()).unwrap();
    sim.launch([-34.2, 50.0, 15.5], [0.0, -1.0, 0.0]).unwrap();
    let broken = |sim: &mut FractureSim| {
        sim.wait().unwrap();
        let words = sim.joint_buffer().contents() as *const u32;
        (0..joints)
            .filter(|&j| unsafe { *words.add(j * 288 / 4 + 2) } & JOINT_FLAG_BROKEN != 0)
            .count()
    };
    for tenth in 1..=30 {
        sim.step(hz / 10).unwrap();
        if tenth % 3 == 0 {
            println!("  t {:.1} s: bonds broken {}, awake {}", tenth as f32 / 10.0, broken(&mut sim), sim.last.awake);
        }
    }
}
