//! The equipment, hardware and trim (`equipment`): every part a closed solid
//! with real mass on its own nodes, or carried by its host; the shell's
//! fixed-trim allowance shrinks by what became parts, so the kerb mass stays.
//! (Closed solids that cut through nothing, and mounts on real nodes: the
//! checks of `tests/mechanical.rs` run over these parts too.)

use engine_vehiclegen::designs::sedan::{Sedan, SedanParams};
use engine_vehiclegen::mechanical::MechPart;
use engine_vehiclegen::Design;

fn sedan(p: &str) -> Sedan {
    let p: SedanParams = serde_json::from_str(if p.is_empty() { "{}" } else { p }).unwrap();
    Sedan::new(p)
}

fn part<'a>(m: &'a [MechPart], name: &str) -> &'a MechPart {
    m.iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("no part {name}"))
}

/// (part, nodes, kg on its nodes, kg carried by its host)
const PARTS: &[(&str, usize, f64, f64)] = &[
    // 1. engine bay
    ("battery", 4, 17.0, 0.0),
    ("airbox", 4, 2.4, 0.0),
    ("fan_shroud", 4, 4.2, 0.0),
    ("condenser", 4, 2.6, 0.0),
    ("coolant_reservoir", 4, 1.6, 0.0),
    ("washer_reservoir", 4, 3.0, 0.0),
    ("brake_booster", 4, 4.4, 0.0),
    ("fuse_box", 4, 1.4, 0.0),
    ("tower_brace", 2, 2.4, 0.0),
    // 5. body hardware
    ("hinge_door_FL_upper", 2, 0.8, 0.0),
    ("hinge_door_FL_lower", 2, 0.8, 0.0),
    ("hinge_door_RL_upper", 2, 0.8, 0.0),
    ("hinge_door_RL_lower", 2, 0.8, 0.0),
    ("hinge_door_FR_upper", 2, 0.8, 0.0),
    ("hinge_door_FR_lower", 2, 0.8, 0.0),
    ("hinge_door_RR_upper", 2, 0.8, 0.0),
    ("hinge_door_RR_lower", 2, 0.8, 0.0),
    ("latch_door_FL", 2, 0.9, 0.0),
    ("latch_door_RL", 2, 0.9, 0.0),
    ("latch_door_FR", 2, 0.9, 0.0),
    ("latch_door_RR", 2, 0.9, 0.0),
    ("door_beam_FL", 0, 0.0, 2.2),
    ("door_beam_RL", 0, 0.0, 2.2),
    ("door_beam_FR", 0, 0.0, 2.2),
    ("door_beam_RR", 0, 0.0, 2.2),
    ("window_regulator_FL", 0, 0.0, 1.8),
    ("window_regulator_RL", 0, 0.0, 1.8),
    ("window_regulator_FR", 0, 0.0, 1.8),
    ("window_regulator_RR", 0, 0.0, 1.8),
    ("hinge_hood_L", 2, 1.1, 0.0),
    ("hinge_hood_R", 2, 1.1, 0.0),
    ("gas_strut_L", 2, 0.35, 0.0),
    ("gas_strut_R", 2, 0.35, 0.0),
    ("hinge_trunk_L", 2, 0.9, 0.0),
    ("hinge_trunk_R", 2, 0.9, 0.0),
    ("latch_hood", 2, 0.8, 0.0),
    ("latch_trunk", 2, 0.7, 0.0),
    ("absorber_front", 4, 0.7, 0.0),
    ("absorber_rear", 4, 0.7, 0.0),
    ("undertray", 4, 2.6, 0.0),
    ("fuel_filler", 3, 1.3, 0.0),
    ("fuel_flap", 0, 0.0, 0.25),
    ("plate_front", 0, 0.0, 0.2),
    ("plate_rear", 0, 0.0, 0.2),
    // 2. brakes and anti-roll bars (the coil springs are part of the struts)
    ("brake_caliper_FL", 0, 0.0, 4.5),
    ("brake_caliper_FR", 0, 0.0, 4.5),
    ("brake_caliper_RL", 0, 0.0, 3.2),
    ("brake_caliper_RR", 0, 0.0, 3.2),
    ("antiroll_bar_front", 4, 5.5, 0.0),
    ("antiroll_bar_rear", 4, 3.5, 0.0),
];

#[test]
fn every_part_has_its_nodes_and_mass() {
    let s = sedan("");
    let st = s.unibody();
    let m = s.mechanical(&st);
    for &(name, n, kg, carried) in PARTS {
        let p = part(&m, name);
        assert_eq!(p.nodes.len(), n, "{name}");
        assert!((p.mass() - kg).abs() < 1e-6, "{name}: {} kg", p.mass());
        assert!(
            (p.carried_kg - carried).abs() < 1e-9,
            "{name}: carries {} kg",
            p.carried_kg
        );
        assert!(!p.mesh.v.is_empty(), "{name} has no solid");
        if n == 0 {
            // a part without nodes follows its host panel or other parts' nodes
            assert!(
                p.host.is_some() || p.binds.len() >= 3,
                "{name} follows nothing"
            );
            assert!(carried > 0.0 && !p.from_trim, "{name}");
        } else {
            assert!(p.from_trim && p.host.is_none(), "{name}");
            // no node lighter than half the study's lightest (a hub cap's 0.2 kg)
            let least = p.nodes.iter().map(|x| x.mass).fold(f64::INFINITY, f64::min);
            assert!(least >= 0.1, "{name}: a node of {least} kg");
            // it can leave the car: every mount breaks
            assert!(
                p.mounts.iter().all(|x| x.v.strength.is_finite()) && !p.mounts.is_empty(),
                "{name}"
            );
        }
    }
    // every part of the equipment module is in the table
    let listed: Vec<&str> = PARTS.iter().map(|p| p.0).collect();
    for p in m.iter().filter(|p| p.from_trim || p.carried_kg > 0.0) {
        assert!(
            listed.contains(&p.name.as_str()),
            "{} is not in the table",
            p.name
        );
    }
}

#[test]
fn the_fixed_trim_shrinks_and_the_kerb_mass_stays() {
    let s = sedan("");
    let st = s.unibody();
    let m = s.mechanical(&st);
    let new: f64 = m.iter().filter(|p| p.from_trim).map(|p| p.mass()).sum();
    assert!((st.trim_removed - new).abs() < 1e-9);
    // the shell's nodes and the parts together weigh what they did before
    // the equipment was parts: 487.1 kg of shell and 700.9 kg of the study's
    let all: f64 = st.node_mass() + m.iter().map(|p| p.mass()).sum::<f64>();
    assert!((all - (487.1 + 700.9)).abs() < 0.2, "{all}");
    // what is left on the shell beside the body-in-white: sealer, wiring,
    // insulation, paint, the heater, fluids
    let biw = s.p.physics.biw as f64;
    let trim = st.fixed_trim(biw);
    println!(
        "fixed trim {trim:.1} kg (was {:.1}), parts from it {new:.1} kg",
        trim + new
    );
    assert!((50.0..=150.0).contains(&trim), "{trim}");
    let table = s.mass_table();
    for (class, kg, nodes) in &table {
        println!("{class:<30} {kg:>7.1} kg  {nodes:>3} nodes");
    }
    let sum: f64 = table.iter().map(|r| r.1).sum();
    assert!((sum - s.p.physics.kerb as f64).abs() < 1e-6);
    // panels, glass, lamps and the driver: unchanged by the equipment
    assert!(
        (table.last().unwrap().1 - 262.0).abs() < 0.3,
        "{}",
        table.last().unwrap().1
    );
}

#[test]
fn every_preset_builds_the_same_parts() {
    let n = sedan("").mechanical(&sedan("").unibody()).len();
    for p in [
        r#"{"wheelbase":2.55,"front_overhang":0.80,"rear_overhang":0.85,"width":1.78,"height":1.42,"track":1.53,"tyre_d":0.64,"tyre_w":0.205,"rim_in":16,"rim_w_in":6.5}"#,
        r#"{"ride":0.15,"height":1.52,"tyre_d":0.78,"tyre_w":0.265,"rim_in":17,"rim_w_in":8,"track":1.66,"width":1.90,"rim_style":"steel"}"#,
        r#"{"wheelbase":3.10,"front_overhang":0.95,"rear_overhang":1.15,"width":1.92,"height":1.48,"track":1.64,"tyre_d":0.70,"tyre_w":0.255,"rim_in":19,"rim_w_in":8.5}"#,
        r#"{"wheelbase":2.80,"width":2.00,"height":1.30,"ride":-0.02,"track":1.76,"tyre_d":0.68,"tyre_w":0.295,"rim_in":20,"rim_w_in":10}"#,
    ] {
        let s = sedan(p);
        let st = s.unibody();
        let m = s.mechanical(&st);
        assert_eq!(m.len(), n, "{p}");
        for q in m.iter().filter(|q| !q.mesh.v.is_empty()) {
            let low = q.mesh.v.iter().map(|v| v.y).fold(f64::INFINITY, f64::min);
            assert!(
                low > 0.10 || q.class == "wheel",
                "{p}: {} is {low:.3} m above the road",
                q.name
            );
        }
    }
}
