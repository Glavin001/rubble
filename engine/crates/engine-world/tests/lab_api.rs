//! The fracture lab's Rust API (`rubble::lab`): a scene placed under every bond model with
//! the same settings, actions queued at steps, the stats a host shows; and the combinations
//! that cannot work failing with their reason.
//!
//!   cargo test --release -p engine-world --test lab_api -- --test-threads=1 --nocapture
#![cfg(feature = "stress")]

use rubble::lab::*;
use rubble::*;
use std::sync::Arc;

fn house_1story() -> LabSource {
    LabSource::Pack {
        bytes: Arc::new(include_bytes!("../../../stress-oracle/packs/house-1story.json").to_vec()),
    }
}

/// Build, settle (D6), place: the world and the lab.
fn place(def: &LabDef) -> anyhow::Result<(World, Lab)> {
    let mut plan = LabPlan::new(def)?;
    let mut world = World::new(GpuContext::Default, &plan.world_def()?)?;
    if plan.needs_settle() {
        plan.settle(world.gpu_context())?;
        world = World::new(world.gpu_context(), &plan.world_def()?)?;
    }
    let lab = Lab::place(&mut world, &plan)?;
    Ok((world, lab))
}

fn run(world: &mut World, lab: &mut Lab, steps: u32) -> anyhow::Result<LabStats> {
    for _ in 0..steps {
        lab.step(world)?;
        world.poll_blocking()?;
    }
    lab.stats(world)
}

/// The scene's centre from straight in front (the lab's click at the middle of the view).
fn centre_shot(lab: &Lab) -> LabAction {
    let s = lab.scene();
    let c = s.center;
    LabAction::Shot {
        origin: [c[0], c[1], c[2] + 3.0 * s.extent.max(1.0)],
        direction: [0.0, 0.0, -1.0],
    }
}

/// Every bond model takes the same scene and the same shot; the breakable ones break it.
#[test]
fn every_bond_model_runs_the_cuboid_house_and_the_breakable_ones_break_it() {
    for bonds in LabBonds::ALL {
        let def = LabDef {
            source: LabSource::Cuboid,
            arrangement: LabArrangement::One,
            settings: LabSettings {
                bonds,
                // The analytic strike holes a wall only above its ballistic limit.
                ball_speed_scale: if bonds == LabBonds::Model(StrengthModel::ImpactLocal) { 6.0 } else { 1.0 },
                ..LabSettings::default()
            },
        };
        let (mut world, mut lab) = place(&def).unwrap_or_else(|e| panic!("{}: {e:#}", bonds.name()));
        assert_eq!(lab.settings().physics_hz, DEFAULT_PHYSICS_HZ, "{}", bonds.name());
        let at_rest = run(&mut world, &mut lab, 30).unwrap();
        lab.queue(&world, &centre_shot(&lab), 0).unwrap();
        let st = run(&mut world, &mut lab, 150).unwrap();
        eprintln!(
            "{}: {} shots · {} bodies · {} of {} bonds broken (at rest {}) · {} awake · {} contacts",
            bonds.name(),
            st.shots,
            st.bodies,
            st.bonds - st.bonds_intact,
            st.bonds,
            at_rest.bonds - at_rest.bonds_intact,
            st.awake,
            st.contacts
        );
        assert_eq!(st.shots, 1, "{}: the shot fired", bonds.name());
        assert_eq!(st.steps, 180);
        if bonds != LabBonds::None {
            assert!(st.bonds > 0, "{}: bonds", bonds.name());
            assert!(st.bonds_intact < st.bonds, "{}: the shot breaks bonds", bonds.name());
        }
        assert!(lab.queue(&world, &LabAction::Volley, 0).is_err(), "{}: no volley without the block", bonds.name());
        if bonds.stress().is_some() {
            assert!(lab.draws_poses());
            let poses = lab.poses(&world).unwrap();
            assert_eq!(poses.len(), RECORD_FLOATS * lab.scene().bodies.len());
        }
    }
}

/// The district buildings (engine-fracture `district.rs`) under the threshold model: each
/// stands under its own weight with no bond broken, and a cannonball breaks it.
#[test]
fn the_district_buildings_stand_and_a_shot_breaks_them() {
    use engine_fracture::district::DistrictScene;
    for scene in [DistrictScene::Block, DistrictScene::Frame] {
        let def = LabDef {
            source: LabSource::district(scene).unwrap(),
            arrangement: LabArrangement::One,
            settings: LabSettings {
                bonds: LabBonds::Model(StrengthModel::Threshold),
                ..LabSettings::default()
            },
        };
        let (mut world, mut lab) = place(&def).unwrap_or_else(|e| panic!("{}: {e:#}", scene.key()));
        let at_rest = run(&mut world, &mut lab, 120).unwrap();
        assert!(at_rest.bonds > 1000, "{}: {} bonds", scene.key(), at_rest.bonds);
        assert_eq!(at_rest.bonds_intact, at_rest.bonds, "{}: it stands whole", scene.key());
        lab.queue(&world, &centre_shot(&lab), 0).unwrap();
        let st = run(&mut world, &mut lab, 150).unwrap();
        eprintln!(
            "{}: {} chunks · {} of {} bonds broken · {} bodies · {} awake",
            scene.key(),
            st.chunks,
            st.bonds - st.bonds_intact,
            st.bonds,
            st.bodies,
            st.awake
        );
        assert_eq!(st.shots, 1, "{}", scene.key());
        assert!(st.bonds_intact < st.bonds, "{}: the shot breaks bonds", scene.key());
    }
}

/// A rocket's charge goes off where its ray meets the building, with blast loading and
/// support loss on: it breaks bonds, and more TNT breaks more. Without blast loading a blast
/// is refused, with its reason.
#[test]
fn a_rocket_breaks_a_blast_loaded_building() {
    use engine_fracture::district::DistrictScene;
    let def = |blast_loading| LabDef {
        source: LabSource::district(DistrictScene::Block).unwrap(),
        arrangement: LabArrangement::One,
        settings: LabSettings {
            bonds: LabBonds::Model(StrengthModel::Threshold),
            support_loss: true,
            blast_loading,
            ..LabSettings::default()
        },
    };
    let rocket = |lab: &Lab, tnt_kg| {
        let LabAction::Shot { origin, direction } = centre_shot(lab) else { unreachable!() };
        LabAction::BlastRay { origin, direction, tnt_kg }
    };
    let mut broken = Vec::new();
    for tnt_kg in [5.0, 50.0] {
        let (mut world, mut lab) = place(&def(true)).unwrap();
        let at_rest = run(&mut world, &mut lab, 60).unwrap();
        assert_eq!(at_rest.bonds_intact, at_rest.bonds, "it stands whole with support loss on");
        lab.queue(&world, &rocket(&lab, tnt_kg), 0).unwrap();
        assert_eq!(lab.queued(), 1);
        let st = run(&mut world, &mut lab, 120).unwrap();
        assert_eq!(lab.queued(), 0, "the charge went off");
        eprintln!("{tnt_kg} kg TNT: {} of {} bonds broken · {} bodies", st.bonds - st.bonds_intact, st.bonds, st.bodies);
        broken.push(st.bonds - st.bonds_intact);
    }
    assert!(broken[0] > 0, "5 kg breaks bonds");
    assert!(broken[1] > broken[0], "50 kg breaks more than 5 kg: {broken:?}");
    let (world, mut lab) = place(&def(false)).unwrap();
    let e = lab.queue(&world, &rocket(&lab, 5.0), 0).unwrap_err();
    assert!(format!("{e:#}").contains("blast loading"), "{e:#}");
}

/// The tower block of house-1story (four copies, the city's cars and pedestrians) under
/// `blast_gpu`: the volley fires every shot at its step and breaks the towers.
#[test]
fn the_tower_block_runs_its_volley_under_blast_gpu() {
    let def = LabDef {
        source: house_1story(),
        arrangement: LabArrangement::TowerBlock { meteor: false },
        settings: LabSettings {
            bonds: LabBonds::Model(StrengthModel::BlastGpu),
            ccd: true,
            ..LabSettings::default()
        },
    };
    let (mut world, mut lab) = place(&def).unwrap();
    assert_eq!(lab.destructibles().len(), 4, "four towers");
    assert!(lab.props().is_some(), "the city's props");
    lab.queue(&world, &LabAction::Volley, 0).unwrap();
    let queued = lab.queued();
    assert!(queued > 0);
    let (_, _, every) = engine_fracture::towers::VOLLEY;
    let steps = 60 + every * queued as u32;
    let st = run(&mut world, &mut lab, steps).unwrap();
    eprintln!(
        "tower block (house-1story) blast_gpu volley: {} shots · {} bodies · {} of {} bonds broken · {} awake",
        st.shots,
        st.bodies,
        st.bonds - st.bonds_intact,
        st.bonds,
        st.awake
    );
    assert_eq!(lab.queued(), 0, "every shot fired");
    assert_eq!(st.shots as usize, queued);
    assert!(st.bonds_intact < st.bonds, "the volley breaks the towers");
    // The topple and meteor are the block's too; a cuboid house has neither.
    lab.queue(&world, &LabAction::Topple, 0).unwrap();
    lab.queue(&world, &LabAction::Meteor, 0).unwrap();
}

/// What cannot work fails with its reason; nothing is substituted.
#[test]
fn combinations_that_cannot_work_say_why() {
    let err = |def: LabDef| match LabPlan::new(&def) {
        Ok(_) => panic!("{def:?} should fail"),
        Err(e) => e.to_string(),
    };
    let base = LabDef {
        source: LabSource::Cuboid,
        arrangement: LabArrangement::One,
        settings: LabSettings::default(),
    };
    let e = err(LabDef {
        settings: LabSettings {
            crush: LabCrush::Scene,
            ..LabSettings::default()
        },
        ..base.clone()
    });
    assert!(e.contains("crushing is a stress model's"), "{e}");
    let e = err(LabDef {
        arrangement: LabArrangement::TowerBlock { meteor: false },
        ..base.clone()
    });
    assert!(e.contains("tower block"), "{e}");
    let e = err(LabDef {
        settings: LabSettings {
            physics_hz: 0.0,
            ..LabSettings::default()
        },
        ..base.clone()
    });
    assert!(e.contains("physics rate"), "{e}");
    let e = err(LabDef {
        settings: LabSettings {
            fracture: FractureOptions {
                contact_fracture: Some(true),
                ..FractureOptions::default()
            },
            ..LabSettings::default()
        },
        ..base.clone()
    });
    assert!(e.contains("contact fracture"), "{e}");
    // A scene's suggested settings change nothing unless applied.
    let block = LabDef {
        source: house_1story(),
        arrangement: LabArrangement::TowerBlock { meteor: false },
        settings: LabSettings {
            bonds: LabBonds::Model(StrengthModel::Threshold),
            ..LabSettings::default()
        },
    };
    let suggested = block.suggested_settings(&block.settings);
    assert_eq!(suggested.bonds, LabBonds::Model(StrengthModel::Bonds));
    assert_eq!(block.settings.bonds, LabBonds::Model(StrengthModel::Threshold));
    assert_eq!(LabPlan::new(&block).unwrap().def().settings.bonds, LabBonds::Model(StrengthModel::Threshold));
}

/// The cross-host scenario (engine/docs/lab-world-plan.md, Checks): a pack's tower block
/// under `blast_gpu`, the volley, 480 steps; one stats line, as the C example (`lab.c`) and
/// the web page print it, for `engine/scripts/lab-cross-host.mjs` to compare. The pack:
/// `LAB_CROSS_HOST_PACK`, else the fractured tower when generated, else house-1story.
#[test]
fn cross_host_tower_block_reference() {
    let generated = concat!(env!("CARGO_MANIFEST_DIR"), "/../engine-webgpu/generated/vibe-city/fracturedTower.json");
    let path = std::env::var("LAB_CROSS_HOST_PACK").unwrap_or_else(|_| {
        if std::path::Path::new(generated).exists() {
            generated.to_string()
        } else {
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../stress-oracle/packs/house-1story.json").to_string()
        }
    });
    let bonds = std::env::var("LAB_CROSS_HOST_BONDS").unwrap_or_else(|_| "blast_gpu".into());
    let steps: u32 = std::env::var("LAB_CROSS_HOST_STEPS").map_or(480, |v| v.parse().unwrap());
    let def = LabDef {
        source: LabSource::Pack {
            bytes: Arc::new(std::fs::read(&path).unwrap()),
        },
        arrangement: LabArrangement::TowerBlock { meteor: false },
        settings: LabSettings {
            bonds: LabBonds::parse(&bonds).expect("a bond model"),
            ccd: true,
            ..LabSettings::default()
        },
    };
    let (mut world, mut lab) = place(&def).unwrap();
    lab.queue(&world, &LabAction::Volley, 0).unwrap();
    let s = run(&mut world, &mut lab, steps).unwrap();
    println!(
        "LAB_STATS host=native pack={} model={bonds} steps={} shots={} bodies={} broken={} crushed={} bonds={} chunks={}",
        path.rsplit('/').next().unwrap(),
        s.steps,
        s.shots,
        s.bodies,
        s.bonds - s.bonds_intact,
        s.crushed,
        s.bonds,
        s.chunks
    );
    assert!(s.bonds_intact < s.bonds, "the volley breaks the block");
}

