//! The reference destruction rigs end to end (`engine/docs/destruction-reactions.md`,
//! "Tests and oracle"): the punch-through wall and the reference building, ported from
//! blast-stress-solver as scene packs, run as threshold destructibles with every reaction
//! counted (`support/reactions.rs`). Gates are physical invariants, not another engine's
//! numbers.
#![cfg(feature = "stress")]

#[path = "support/reactions.rs"]
mod reactions;

use reactions::*;
use rubble::*;

fn spec(scene: &str, shot: Option<&str>, seconds: f64) -> RunSpec {
    RunSpec {
        scene: scene.into(),
        shot: shot.map(str::to_string),
        seconds,
        ..RunSpec::default()
    }
}

#[test]
fn both_reference_scenes_load_with_their_crush_blocks_and_shots() {
    for (key, bytes) in SCENES {
        let json: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        let nodes = json["scenario"]["nodes"].as_array().unwrap().len();
        let bonds = json["scenario"]["bonds"].as_array().unwrap().len();
        // The authored crush parameters survive the port (v3 packs).
        let crush = json["defaults"]["solver"]["materials"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| m.get("crush").is_some())
            .count();
        assert_eq!(
            crush > 0,
            json["version"] == 3,
            "{key}: crush blocks iff v3"
        );
        // The World loader takes them (unknown fields are not an error).
        let asset = DestructibleAssetDesc::from_pack(bytes).unwrap();
        assert!(asset.name.is_some(), "{key}");
        let stress = asset.stress.as_ref().unwrap();
        if let Err(e) = stress.scene() {
            panic!("{key}: the stress import failed: {e}");
        }
        let scene = stress
            .threshold_scene(1.0, asset.solver_for(StrengthModel::Threshold).unwrap().dt)
            .unwrap_or_else(|e| panic!("{key}: {e}"));
        let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
        assert_eq!(chunks, nodes, "{key}: one stress chunk per pack node");
        assert_eq!(asset.chunks.len() + asset.bonds.len() > 0, true);
        eprintln!(
            "{key}: {nodes} nodes, {bonds} bonds, {} world chunks, {} world bonds",
            asset.chunks.len(),
            asset.bonds.len()
        );
        let shots = engine_fracture::pack::pack_shots(bytes).unwrap();
        assert!(!shots.is_empty(), "{key}: authored shots");
        for s in &shots {
            assert!(s.mass > 0.0 && s.radius > 0.0, "{key} {}", s.label);
            // Every shot flies toward the structure (+x, the source's launcher side).
            assert!(
                s.velocity[0] > 0.0 && s.origin[0] < -5.0,
                "{key} {}",
                s.label
            );
        }
    }
}

#[test]
fn reference_building_stands_under_gravity_for_5_s() {
    let r = run(&spec("reference-building-crush", None, 5.0)).unwrap();
    assert_eq!(r.total_broken(), 0, "gravity alone breaks no bond");
    assert_eq!(r.total_splits(), 0);
    assert_eq!(r.bonds_intact_end, r.bonds);
    assert_eq!(
        r.total_crushed(),
        0,
        "a structure under its own weight never crushes"
    );
}

/// Mass is conserved through every split, frame by frame (no crushing removes it: the
/// charter's default `CrushMass::Conserve`).
fn assert_mass_conserved(r: &Report) {
    let m0 = r.frames[0].system_mass;
    let shot = r.shot.as_ref().map_or(0.0, |s| s.mass);
    for f in &r.frames {
        let expect = if f.impactor_speed.is_some() {
            r.structure_mass_start + shot
        } else {
            r.structure_mass_start
        };
        assert!(
            (f.system_mass - expect).abs() <= 1e-4 * expect,
            "{}: frame {} system mass {} vs {expect} (first {m0})",
            r.scene,
            f.frame,
            f.system_mass
        );
    }
    assert!((r.structure_mass_end - r.structure_mass_start).abs() <= 1e-4 * r.structure_mass_start);
}

#[test]
fn an_ordinary_shot_knocks_chunks_off_and_conserves_mass() {
    for (scene, shot) in [
        ("reference-building-crush", "ordinary"),
        ("crush-wall", "1.0x"),
    ] {
        let r = run(&spec(scene, Some(shot), 4.0)).unwrap();
        eprintln!("{}", to_markdown(std::slice::from_ref(&r)));
        assert!(
            r.total_broken() > 0,
            "{scene} {shot}: the shot breaks bonds"
        );
        assert!(
            r.total_splits() > 0 && r.actors_end() > 1,
            "{scene} {shot}: pieces come off"
        );
        // No break before the shot arrives: everything before launch is gravity.
        let s = r.shot.as_ref().unwrap();
        assert!(
            r.frames[..s.launch_frame as usize]
                .iter()
                .all(|f| f.bonds_broken == 0),
            "{scene}: bonds broke before the shot"
        );
        // The impactor pays for the damage: it leaves slower than it came.
        assert!(
            s.speed_out < s.speed_in && s.ke_out < s.ke_in,
            "{scene}: {s:?}"
        );
        assert_mass_conserved(&r);
    }
}

#[test]
fn two_runs_produce_identical_events() {
    let a = run(&spec("crush-wall", Some("1.4x"), 3.0)).unwrap();
    let b = run(&spec("crush-wall", Some("1.4x"), 3.0)).unwrap();
    assert!(!a.events.is_empty());
    assert_eq!(a.events, b.events, "the same events in the same frames");
    assert_eq!(
        a.frames, b.frames,
        "the same counts, speeds and masses every frame"
    );
}

/// Every reaction on (`crush`, `support_loss`, `blast_loading`), gravity alone for 5 s:
/// the reference building stands, nothing breaks and nothing crushes (its columns carry
/// the flow at about 1 MPa against the reinforced concrete's 14.4 MPa cohesion; the
/// drywall infill carries a share of it far under its 0.48 MPa).
#[test]
fn reference_building_with_every_reaction_on_stands_under_gravity() {
    let r = run(&RunSpec {
        crush: true,
        support_loss: true,
        blast_loading: true,
        ..spec("reference-building-crush", None, 5.0)
    })
    .unwrap();
    assert_eq!(r.total_broken(), 0, "gravity alone breaks no bond");
    assert_eq!(r.total_splits(), 0);
    assert_eq!(
        r.total_crushed(),
        0,
        "a structure under its own weight never crushes"
    );
    assert_eq!(r.bonds_intact_end, r.bonds);
    assert_mass_conserved(&r);
}

/// The ground-floor charge of the tall building partially destroyed: 20 kg of TNT 0.30 m
/// outside the ground-floor drywall infill on the -z wall (chunk 40, its outer face at
/// z = -2.06; the charge at its centre's height).
///
/// Why 20 kg (hand calculation, `engine/docs/destruction-reactions.md` section 1 "Blast"):
/// at R = 0.30 m, Z = R / W^(1/3) = 0.110 m/kg^(1/3); Kinney-Graham gives the reflected
/// peak 242 MPa, positive phase 15.7 ms, Friedlander decay b = 314. The drywall crushes
/// when `integral excess^2 dt >= crushViscosity crushEnergy = 2e5 x 2e6 = 4e11 Pa^2 s`.
/// Loaded on one face with its reactions at its centre (the uniaxial bound, sigma = -P/2,
/// p = P/6, q = P/2, excess = 0.3 P - 0.48 MPa) it takes D = 0.32 at 20 kg (0.13 at 10 kg,
/// 1.0 at 50 kg). The pack's bonds put the infill's reaction centroid 0.11 m behind its
/// mid-plane (its bond centroids are the chunk centres' midpoints), so the lever doubles
/// the normal stress and adds shear: sigma_zz = -(0.06 + 0.11) P A / V = -1.41 P, about
/// 9x the bound's excess^2, D of order 3: crushed in the first window with margin. The
/// reinforced-concrete columns (cohesion 14.4 MPa, crushEnergy 120 MJ/m^3) take D < 0.01
/// at 1 m: they are knocked off, not crushed.
const CHARGE: BlastSpec = BlastSpec {
    position: [-0.9, 2.1, -2.36],
    tnt_kg: 20.0,
    at: 2.0,
};

fn tall_building(seconds: f64) -> Report {
    run(&RunSpec {
        crush: true,
        support_loss: true,
        blast: Some(CHARGE),
        ..spec("reference-building-crush", None, seconds)
    })
    .unwrap()
}

#[test]
fn tall_building_partially_destroyed_by_a_ground_floor_charge() {
    let r = tall_building(6.0);
    eprintln!("{}", to_markdown(std::slice::from_ref(&r)));
    for (c, b) in &r.broken_by_cause {
        eprintln!("{c:?}: {} bonds {b:?}", b.len());
    }
    eprintln!("crushed {:?}", r.crushed);
    let charge = glam::Vec3::from(CHARGE.position);
    let blast_frame = r.blast_frame.unwrap() as usize;
    assert!(
        r.frames[..blast_frame].iter().all(|f| f.bonds_broken == 0),
        "nothing breaks before the charge"
    );
    // Crushed by the blast, near the charge.
    let near: Vec<u32> = r
        .crushed
        .iter()
        .filter(|c| c.1 == CrushCause::Blast)
        .map(|c| c.0)
        .collect();
    assert!(!near.is_empty(), "the charge crushes a chunk");
    for &c in &near {
        let d = (glam::Vec3::from(r.chunk_centres_start[c as usize]) - charge).length();
        assert!(
            d < 1.0,
            "chunk {c} crushed by the blast {d} m from the charge"
        );
    }
    // Knocked off: the blast breaks bonds and pieces come off.
    assert!(r.total_blast_breaks() > 0, "the blast breaks bonds");
    assert!(
        r.total_splits() > 0 && r.actors_end() > 1,
        "pieces come off"
    );
    // A crushed chunk's bonds are reported as the crush's.
    let crush_bonds = r
        .broken_by_cause
        .iter()
        .find(|c| c.0 == BreakCause::Crush)
        .map_or(0, |c| c.1.len());
    assert!(
        crush_bonds > 0,
        "the crushed chunk's bonds break with cause Crush"
    );
    // Support loss: the bonds that carried the floors above to the footings break once
    // the blast has cut their load path, above the charge's level, never in a footing.
    let support = &r
        .broken_by_cause
        .iter()
        .find(|c| c.0 == BreakCause::SupportLoss)
        .expect("support-loss breaks")
        .1;
    assert!(!support.is_empty());
    let pack: serde_json::Value =
        serde_json::from_slice(pack_bytes("reference-building-crush").unwrap()).unwrap();
    let bonds = pack["scenario"]["bonds"].as_array().unwrap();
    for &b in support {
        let bond = &bonds[b as usize];
        let (a, o) = (
            bond["node0"].as_u64().unwrap() as usize,
            bond["node1"].as_u64().unwrap() as usize,
        );
        let top = r.chunk_centres_start[a][1].max(r.chunk_centres_start[o][1]);
        assert!(
            top > CHARGE.position[1],
            "support-loss bond {b} ({a}-{o}) below the charge"
        );
        for c in [a, o] {
            assert!(
                pack["scenario"]["nodes"][c]["mass"].as_f64().unwrap() > 0.0,
                "bond {b} on a footing"
            );
        }
    }
    // The support-loss breaks come after the blast's (forward: the flow runs on the
    // pieces the blast's breaks left).
    let first_support = r
        .frames
        .iter()
        .position(|f| f.support_loss_breaks > 0)
        .unwrap();
    let first_blast = r.frames.iter().position(|f| f.blast_breaks > 0).unwrap();
    assert!(first_support >= first_blast);
    // Debris lands: no chunk centre (or crush fragment) ever below the ground, and the
    // crushed chunks' fragments end on it.
    let lowest = r.lowest_above_ground().unwrap();
    assert!(
        lowest > 0.0,
        "a chunk centre went {lowest} m below the ground"
    );
    let ground = r.ground.unwrap();
    for &c in &near {
        let y = r.chunk_centres_end[c as usize][1] - ground;
        assert!(y < 0.3, "chunk {c}'s fragments end {y} m above the ground");
    }
    // Mass kept every frame: the crushed chunks' mass is in their fragments.
    assert_mass_conserved(&r);
    assert!(r.total_crushed_mass() > 0.0);
}

#[test]
fn tall_building_blast_is_deterministic() {
    let a = tall_building(3.0);
    let b = tall_building(3.0);
    assert!(a.total_crushed() > 0 && a.total_support_loss() > 0);
    assert_eq!(a.events, b.events, "the same events in the same frames");
    assert_eq!(a.frames, b.frames, "the same counts and masses every frame");
    assert_eq!(a.chunk_centres_end, b.chunk_centres_end, "the same poses");
}
