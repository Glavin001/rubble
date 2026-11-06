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
        if matches!(def, rubble::play::SceneDef::Range(_)) {
            // A range scene builds its own world from its GLB.
            let mut play = Playground::new(GpuContext::Default, def)
                .unwrap_or_else(|e| panic!("{}: {e:#}", p.name));
            play.step(3).unwrap_or_else(|e| panic!("{}: {e:#}", p.name));
            assert_eq!(play.stats().overflow, 0, "{}", p.name);
            eprintln!(
                "{:16} {:7} bodies (range scene)",
                p.name,
                play.stats().bodies
            );
            ran.push(p.name);
            continue;
        }
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
        let mut play = Playground::of_placed(def, world, placed)
            .unwrap_or_else(|e| panic!("{}: {e:#}", p.name));
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

/// Each scene's director natively, through [`Playground`]: its actions and its own script
/// change the world as the pages' did (a volley flies, a crater digs, the T-bone ram
/// launches at its step, the range's volleys land and score, the run sends meteors).
#[test]
fn directors_act_and_script() {
    use rubble::play::{Action, Aim};
    let aim = Aim {
        eye: [0.0, 3.0, -10.0],
        dir: [0.0, 0.0, 1.0],
    };
    let events = |p: &mut Playground| -> Vec<&'static str> {
        p.director
            .take_events()
            .into_iter()
            .map(|e| e.what)
            .collect()
    };
    let dt = 1.0 / 60.0;

    // Motion: the gallery's volley (F, and on its own at step 20), the playground's rocket.
    let mut p = Playground::preset(GpuContext::Default, "ccd", "autofire=0").unwrap();
    p.step(5).unwrap();
    p.act(&Action::Fire { aim: None }).unwrap();
    p.step(10).unwrap();
    assert_eq!(events(&mut p), ["volley"]);
    let mut p = Playground::preset(GpuContext::Default, "ccd", "").unwrap();
    p.step(25).unwrap();
    assert_eq!(
        events(&mut p),
        ["volley"],
        "the gallery fires on its own at step 20"
    );
    let mut p = Playground::preset(GpuContext::Default, "character", "").unwrap();
    p.step(4).unwrap();
    p.act(&Action::Cannon { aim }).unwrap();
    // (Reset here hits main's "forcing a body to sleep" gap, as motion.html's R does.)
    assert_eq!(events(&mut p), ["rocket"]);
    assert!(
        p.act(&Action::Crater { at: None }).is_err(),
        "no craters in motion scenes"
    );

    // Terrain: craters (random spots continue placement's sequence), rain, reset.
    let mut p = Playground::preset(GpuContext::Default, "terrain", "n=500").unwrap();
    p.step(30).unwrap();
    p.act(&Action::Crater {
        at: Some([0.0, 0.0]),
    })
    .unwrap();
    p.act(&Action::Crater { at: None }).unwrap();
    p.act(&Action::Rain).unwrap();
    p.step(30).unwrap();
    assert_eq!(p.director.state()["craters"], 2);
    p.act(&Action::Reset).unwrap();
    assert_eq!(p.director.state()["craters"], 0);
    assert_eq!(events(&mut p), ["crater", "crater", "rain", "reset"]);

    // Soft: the T-bone ram launches at step 60 by itself; the yard's cannon.
    let mut p = Playground::preset(GpuContext::Default, "soft-tbone", "").unwrap();
    for _ in 0..20 {
        p.frame(4, dt).unwrap();
    }
    assert_eq!(events(&mut p), ["ram"]);
    let mut p = Playground::preset(GpuContext::Default, "soft-yard", "").unwrap();
    p.act(&Action::Cannon { aim }).unwrap();
    p.act(&Action::Ram).unwrap();
    assert_eq!(events(&mut p), ["cannon", "ram"]);

    // Vehicles: the range's volleys land on the pads and score; the playground's cannon.
    let mut p = Playground::preset(GpuContext::Default, "meteors", "").unwrap();
    for _ in 0..300 {
        p.frame(1, dt).unwrap();
    }
    let ev = p.director.take_events();
    let volleys = ev.iter().filter(|e| e.what == "volley").count();
    let downs = ev.iter().filter(|e| e.what == "touchdown").count();
    assert!(volleys >= 2, "{volleys} volleys in 5 s");
    assert!(downs > 0, "no touchdowns");
    assert!(p.director.state()["zones"].as_u64().unwrap() > 0);
    // The city: every house placed, its scenario started, a shell fired.
    let mut p = Playground::preset(GpuContext::Default, "city", "k=16&scenario=war").unwrap();
    p.step(30).unwrap();
    p.act(&Action::Start).unwrap();
    p.act(&Action::Shell {
        origin: [0.0, 40.0, 0.0],
        velocity: [0.0, -60.0, 0.0],
        radius: 0.5,
    })
    .unwrap();
    p.step(30).unwrap();
    assert_eq!(events(&mut p), ["start", "shell"]);
    // A range scene: its 24-pounder along the aim.
    let mut p = Playground::preset(GpuContext::Default, "timber-yard", "level=1").unwrap();
    p.act(&Action::Cannon { aim }).unwrap();
    p.step(10).unwrap();
    assert!(p.act(&Action::Rain).is_err());
    let mut p = Playground::preset(GpuContext::Default, "playground", "").unwrap();
    p.act(&Action::Cannon { aim }).unwrap();
    assert_eq!(events(&mut p), ["cannon"]);
}
