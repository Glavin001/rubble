//! Run a named demo scene on Metal (Slang → MSL).
//!
//! ```bash
//! cargo run -p engine-scenes --release --bin demo_viewer -- --list
//! cargo run -p engine-scenes --release --bin demo_viewer -- "Wall Smash (2k)"
//! cargo run -p engine-scenes --release --bin demo_viewer -- wall-smash-2k --steps 300
//! ```

use engine_exec::PhysicsBackend;
use engine_scenes::{build_world, list_scenes};

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() || args.iter().any(|a| a == "--list" || a == "-l") {
        println!("Available scenes (three-avbd inspired, Y-up):\n");
        let mut group = "";
        for s in list_scenes() {
            if s.info.group != group {
                group = s.info.group;
                println!("  {group}");
            }
            println!(
                "    - {}  (iters={}, max_bodies={})",
                s.info.name, s.info.iterations, s.info.capacity.max_bodies
            );
        }
        if args.is_empty() {
            println!("\nDefaulting to Wall Smash (2k). Pass a name or --list.");
            args.push("Wall Smash (2k)".into());
        } else {
            return Ok(());
        }
    }

    let mut steps = 300u32;
    let mut name = String::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--steps" {
            i += 1;
            steps = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(300);
        } else if !args[i].starts_with('-') {
            if !name.is_empty() {
                name.push(' ');
            }
            name.push_str(&args[i]);
        }
        i += 1;
    }
    if name.is_empty() {
        name = "Wall Smash (2k)".into();
    }

    let (mut world, info) = build_world(&name)?;
    println!(
        "demo_viewer scene=\"{}\" group=\"{}\" chip={} bodies={} iters={}",
        info.name,
        info.group,
        world.chip,
        world.body_count(),
        info.iterations
    );
    println!("poses buffer resident on GPU (no hot-path readback)");

    for step in 0..steps {
        match world.step() {
            Ok(_) => {}
            Err(e) => {
                eprintln!("step {step} failed: {e:?}");
                break;
            }
        }
        if step % 60 == 0 || step + 1 == steps {
            let t = world.last_timings();
            let c = world.last_counters();
            let bodies = world.download_bodies();
            let mut awake = 0u32;
            let mut sleeping = 0u32;
            let mut max_y = f32::NEG_INFINITY;
            let mut min_y = f32::INFINITY;
            for b in &bodies {
                if b.is_sleeping() {
                    sleeping += 1;
                } else if b.is_dynamic() {
                    awake += 1;
                }
                max_y = max_y.max(b.position[1]);
                min_y = min_y.min(b.position[1]);
            }
            println!(
                "step={step} gpu={:.3}ms cpu_submit={:.3}ms contacts={} joints={} awake={awake} sleeping={sleeping} y=[{min_y:.2},{max_y:.2}] overflow=0x{:x}",
                t.total_gpu,
                t.cpu_submit,
                c.contact_count,
                c.joint_count,
                c.overflow_flags
            );
        }
    }
    Ok(())
}
