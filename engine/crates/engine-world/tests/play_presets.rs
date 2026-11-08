//! Every preset whose content is placed in Rust builds through one entry, natively: its
//! scene from the preset's query ([`SceneDef::from_query`]), its world, its placement,
//! and a few steps without overflow. The same scenes the pages build (checked byte for
//! byte in the browser) and the C API runs.

use rubble::play::{presets, Playground};
use rubble::GpuContext;

#[test]
fn every_placed_preset_builds_and_steps() {
    // PLAY_PRESETS=a,b runs only those; PLAY_SKIP=a,b leaves those out.
    let only = std::env::var("PLAY_PRESETS").ok();
    let skip = std::env::var("PLAY_SKIP").unwrap_or_default();
    let mut ran = Vec::new();
    for p in presets() {
        if only
            .as_deref()
            .is_some_and(|o| !o.split(',').any(|n| n == p.name))
        {
            continue;
        }
        if skip.split(',').any(|n| n == p.name) {
            continue;
        }
        let Some(def) = p.scene_def() else { continue };
        let def = def.unwrap_or_else(|e| panic!("{}: {e}", p.name));
        // Each phase's time: making the world, placing the scene, its first steps.
        let t = std::time::Instant::now();
        let wd = def
            .world_def()
            .unwrap_or_else(|e| panic!("{}: {e:#}", p.name));
        let world = rubble::World::new(GpuContext::Default, &wd)
            .unwrap_or_else(|e| panic!("{}: {e:#}", p.name));
        let made = t.elapsed();
        if only.is_some() {
            eprintln!("{}: world made, placing", p.name);
        }
        let mut world = world;
        let placed = def
            .place(&mut world)
            .unwrap_or_else(|e| panic!("{}: {e:#}", p.name));
        let placing = t.elapsed() - made;
        if only.is_some() {
            eprintln!(
                "{}: placed in {:.0} ms, stepping",
                p.name,
                placing.as_secs_f64() * 1e3
            );
        }
        let mut play = Playground {
            world,
            scene: def,
            placed,
        };
        let bodies = play.stats().bodies;
        assert!(bodies > 1, "{}: {bodies} bodies", p.name);
        play.step(3).unwrap_or_else(|e| panic!("{}: {e:#}", p.name));
        let stepping = t.elapsed() - made - placing;
        let s = play.stats();
        assert_eq!(s.overflow, 0, "{}: overflow {:#x}", p.name, s.overflow);
        assert_eq!(s.steps, 3, "{}", p.name);
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        eprintln!(
            "{:<16} {:>7} bodies  world {:>6.0} ms  place {:>7.0} ms  3 steps {:>6.0} ms",
            p.name,
            bodies,
            ms(made),
            ms(placing),
            ms(stepping)
        );
        ran.push(p.name);
    }
    eprintln!("{} presets", ran.len());
    assert!(
        only.is_some() || !skip.is_empty() || ran.len() > 30,
        "{} presets ran",
        ran.len()
    );
}
