//! Sections, triggers, key-driven commands, slide nodes and animators in the explicit
//! solver, from `engine/docs/explicit-beams-sections-triggers-slidenodes-spec.md`'s
//! definitions and verification plan (§8): synthetic files only.

use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, ExplicitKind, ExplicitType, GRAVITY, STEP,
};
use engine_softbody::explicit_triggers::segment_point;
use engine_softbody::truck::{parse_truck, parse_truck_config};
use glam::Vec3;

fn settings() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings {
        fast_inv_sqrt: false,
        drag: false,
        turbulence: false,
        triangle_throttle: true,
        centred_damping: false,
        centred_objective: false,
    }
}

fn spawn(text: &str) -> ExplicitBeams {
    ExplicitBeams::from_truck(text, Vec3::ZERO, Vec3::ZERO, settings()).unwrap()
}

fn ends(s: &ExplicitBeams) -> Vec<(u32, u32)> {
    s.beams.iter().map(|b| (b.a, b.b)).collect()
}

// --- §1 sections ---

/// Root lines around two modules: `zeta` (first in the file, unselected by
/// default) sets the beam scale and node friction for every later line; `alpha`
/// adds a node and a beam, then a root node and beam follow it.
const SECTIONS: &str = "sections
globals
100, 0, mat
nodes
0, 0, 1, 0
1, 1, 1, 0
2, 0, 2, 0
beams
0, 1
sectionconfig 0 zeta
sectionconfig 0 alpha
section 0 zeta
set_beam_defaults_scale 3, 3, 3, 3
set_node_defaults -1, 1.5, -1, -1
beams
1, 2
end_section
beams
0, 2
section 0 alpha
nodes
4, 2, 2, 0
beams
2, 4
set_beam_defaults_scale 2, 2, 2, 2
end_section
nodes
5, 3, 3, 0
beams
1, 5
end
";

#[test]
fn explicit_sections_default_is_first_name_in_byte_order() {
    let f = parse_truck(SECTIONS);
    assert_eq!(f.section_configs, ["alpha", "zeta"]);
    assert_eq!(f.section_config.as_deref(), Some("alpha"));
    let s = spawn(SECTIONS);
    // Root nodes (0, 1, 2 and the later 5) first, then alpha's node 4.
    assert_eq!(s.nodes.len(), 5);
    assert_eq!(s.nodes[3].pos, Vec3::new(3.0, 3.0, 0.0));
    assert_eq!(s.nodes[4].pos, Vec3::new(2.0, 2.0, 0.0));
    // Root beams in file order, then alpha's.
    assert_eq!(ends(&s), [(0, 1), (0, 2), (1, 3), (2, 4)]);
    // The unselected module's scale applies to the root and alpha lines after it,
    // alpha's to the root line after end_section.
    let k: Vec<f32> = s.beams.iter().map(|b| b.k).collect();
    assert_eq!(k, [9.0e6, 27.0e6, 18.0e6, 27.0e6]);
    // So does the unselected module's node friction.
    let fr: Vec<f32> = s.nodes.iter().map(|n| n.friction).collect();
    assert_eq!(fr, [1.0, 1.0, 1.0, 1.5, 1.5]);
}

#[test]
fn explicit_sections_named_empty_and_unknown_configs() {
    let s = ExplicitBeams::from_truck_config(
        SECTIONS,
        Some("zeta"),
        Vec3::ZERO,
        Vec3::ZERO,
        settings(),
    )
    .unwrap();
    assert_eq!(s.nodes.len(), 4);
    assert_eq!(ends(&s), [(0, 1), (0, 2), (1, 3), (1, 2)]);
    for config in ["", "nope"] {
        let f = parse_truck_config(SECTIONS, Some(config));
        assert_eq!(f.section_config, None);
        let s = ExplicitBeams::from_truck_config(
            SECTIONS,
            Some(config),
            Vec3::ZERO,
            Vec3::ZERO,
            settings(),
        )
        .unwrap();
        assert_eq!(s.nodes.len(), 4, "{config}: the root alone");
        assert_eq!(ends(&s), [(0, 1), (0, 2), (1, 3)]);
    }
    assert!(parse_truck_config(SECTIONS, Some("nope"))
        .warnings
        .iter()
        .any(|w| w.contains("not found")));
}

#[test]
fn explicit_sections_first_name_counts_and_sectionconfig_is_ignored() {
    let text = "names
nodes
0, 0, 1, 0
1, 1, 1, 0
sectionconfig 0 b
sectionconfig 0 a
section 0 b a
beams
0, 1
end_section
end
";
    let f = parse_truck(text);
    assert_eq!(f.section_configs, ["b"]);
    assert_eq!(f.beams.len(), 1);
}

#[test]
fn explicit_beam_and_node_default_directives() {
    let text = "defaults
nodes
0, 0, 1, 0
1, 1, 1, 0
beams
set_beam_defaults_scale 2, 2, 2
0, 1
set_beam_defaults_scale 2, 3, 1, 1, 0.1
0, 1
set_beam_defaults 1000, 20, 500000, 2000000
0, 1
set_beam_defaults 2000
0, 1
set_beam_defaults -1
0, 1
end
";
    let f = parse_truck(text);
    let b = &f.beams;
    // Three numbers: ignored.
    assert_eq!((b[0].spring, b[0].damp), (9.0e6, 12000.0));
    // Five: the fifth is ignored.
    assert_eq!((b[1].spring, b[1].damp), (18.0e6, 36000.0));
    assert_eq!(
        (b[2].spring, b[2].damp, b[2].strength),
        (2000.0, 60.0, 2.0e6)
    );
    // From the current values.
    assert_eq!(
        (b[3].spring, b[3].damp, b[3].strength),
        (4000.0, 60.0, 2.0e6)
    );
    // Negative: the built-in default.
    assert_eq!((b[4].spring, b[4].damp), (18.0e6, 60.0));

    let nodes = "node defaults
set_node_defaults -1, 1.3
nodes
0, 0, 1, 0
set_node_defaults
1, 1, 1, 0
set_node_defaults 10
2, 1, 2, 0
end
";
    let f = parse_truck(nodes);
    let fr: Vec<f32> = f.nodes.iter().map(|n| n.friction).collect();
    // A line without the load weight is ignored; an omitted friction is 1.
    assert_eq!(fr, [1.3, 1.3, 1.0]);
}

// --- §2 triggers ---

/// Nodes 0 (a) and 1 (b), 1 m apart along x, immovable, with `lines` in a
/// `triggers` section (and anything after).
fn trigger_rig(head: &str, lines: &str) -> String {
    format!(
        "trigger rig
globals
100, 0, mat
{head}
nodes
0, 0, 1, 0
1, 1, 1, 0
2, 0, 2, 0
3, 1, 2, 0
fixes
0
1
2
3
beams
0, 2
1, 3
triggers
{lines}
end
"
    )
}

#[test]
fn explicit_trigger_spawn_bounds_and_beam() {
    let text = trigger_rig(
        "set_beam_defaults_scale 0.05, 0.005, 0.01, 0.5",
        "0, 1, 0.6, 0.4, 84, 83, ci\n1, 0, 0.58, 0.5, 17, 16, ci",
    );
    let s = spawn(&text);
    let t: Vec<_> = s
        .beams
        .iter()
        .filter(|b| b.kind == ExplicitKind::Trigger)
        .collect();
    assert_eq!(t.len(), 2);
    let bits = |b: &&engine_softbody::explicit_beams::Beam| {
        (b.short_bound.to_bits(), b.long_bound.to_bits())
    };
    assert_eq!(bits(&t[0]), (0x3ECC_CCCC, 0xBF19_999A));
    assert_eq!(bits(&t[1]), (0x3ED7_0A3E, 0xBF00_0000));
    for b in &t {
        assert_eq!((b.k, b.c), (0.0, 0.0));
        assert_eq!(b.ty, ExplicitType::Actuator);
        // S unscaled, G the scaled deform threshold.
        assert_eq!(b.strength, 1.0e6);
        assert_eq!(b.gate, 4000.0);
        assert_eq!(b.rest, 1.0);
    }
    assert_eq!((t[0].a, t[0].b), (0, 1));
    assert_eq!((s.triggers[0].short_key, s.triggers[0].long_key), (84, 83));
    assert_eq!((s.triggers[1].short_key, s.triggers[1].long_key), (17, 16));
}

#[test]
fn explicit_trigger_validation_and_masses() {
    let base = spawn(&trigger_rig("minimass\n1", ""));
    for bad in ["0, 1, 0.5, 1.5, 0, 4", "0, 1, 0.5, 1.5, 85, 4"] {
        let s = spawn(&trigger_rig("minimass\n1", bad));
        assert_eq!(s.beams.len(), base.beams.len(), "{bad}: dropped");
        let m: Vec<f32> = s.nodes.iter().map(|n| n.mass).collect();
        let m0: Vec<f32> = base.nodes.iter().map(|n| n.mass).collect();
        assert_eq!(m, m0, "{bad}: no mass");
    }
    // A trigger beam takes its share of the dry mass.
    let s = spawn(&trigger_rig("minimass\n1", "0, 1, 0.5, 1.5, 3, 4"));
    assert!(s.nodes[0].mass > base.nodes[0].mass);
    assert!(s.nodes[2].mass < base.nodes[2].mass);
    // Unsupported options and key blockers are refused.
    for refused in [
        "0, 1, 0.5, 1.5, 3, 4, x",
        "0, 1, 0.5, 1.5, 3, 4, ct",
        "0, 1, 0.5, 1.5, 3, -1",
    ] {
        let r = ExplicitBeams::from_truck(
            &trigger_rig("minimass\n1", refused),
            Vec3::ZERO,
            Vec3::ZERO,
            settings(),
        );
        assert!(r.is_err(), "{refused}");
    }
}

/// Place node a so that the trigger's length is `len` (node b at x = 1).
fn place(s: &mut ExplicitBeams, len: f32) {
    s.nodes[0].pos = Vec3::new(1.0 - len, 1.0, 0.0);
}

fn held(s: &ExplicitBeams) -> Vec<usize> {
    (0..s.key_input.len())
        .filter(|&k| s.key_input[k] != 0.0)
        .collect()
}

#[test]
fn explicit_trigger_sides_boundaries_and_zero_force() {
    // No `c`: sb = 0.5, lb = 1.0 (L = 1): long past ℓ = 2, short below ℓ = 0.5.
    let mut s = spawn(&trigger_rig("minimass\n1", "0, 1, 0.5, 1.0, 3, 4"));
    let cases = [
        (1.0, vec![]),
        (2.0, vec![]), // e = lb·L exactly: the short test applies, and fails
        (2.25, vec![4]),
        (0.5, vec![]), // e = −sb·L exactly
        (0.25, vec![3]),
    ];
    for (len, want) in cases {
        place(&mut s, len);
        s.step();
        assert_eq!(held(&s), want, "length {len}");
        // The trigger adds nothing: the force is the restart's gravity.
        assert_eq!(
            s.nodes[0].force,
            Vec3::new(0.0, s.nodes[0].mass * GRAVITY, 0.0)
        );
    }
    // `c` with lb < −sb: sb = 0.5, lb = −0.8, no inside region; the long side wins
    // where both hold (−0.8 < e < −0.5).
    let mut s = spawn(&trigger_rig("minimass\n1", "0, 1, 0.5, 0.2, 3, 4, c"));
    for (len, want) in [(1.0, vec![4]), (0.375, vec![4]), (0.125, vec![3])] {
        place(&mut s, len);
        s.step();
        assert_eq!(held(&s), want, "length {len}");
    }
}

#[test]
fn explicit_trigger_drives_commands_one_step_later() {
    // Command (2, 3): contract on the long key 4 (B 1.0), extend on the short key 3
    // (B 1.5), rates 0.6 / 0.8; command (2, 3) again with both entries on key 3.
    let text = trigger_rig(
        "",
        "0, 1, 0.5, 1.0, 3, 4\ncommands2\n2, 3, 0.6, 0.8, 1.0, 1.5, 4, 3, i\n2, 3, 10, 10, 1.0, 1.5, 3, 3, i",
    );
    let mut s = spawn(&text);
    assert_eq!(s.commands.len(), 2);
    let (j, j2) = (s.commands[0].beam as usize, s.commands[1].beam as usize);
    let rest0 = s.beams[j].rest0;
    // Inside: nothing moves.
    s.step();
    assert_eq!(s.beams[j].rest, rest0);
    // Short side in step n: the commands start in step n + 1.
    place(&mut s, 0.25);
    s.step();
    assert_eq!(s.beams[j].rest, rest0);
    s.step();
    let step1 = s.beams[j].rest - rest0;
    assert!((step1 - 0.8 * STEP).abs() < 1e-6, "{step1}");
    // Contract key = extend key: contraction first (ρ = 1 does not pass B = 1.0),
    // then the extension at 10 m/s.
    assert!((s.beams[j2].rest - rest0 - 10.0 * STEP).abs() < 1e-5);
    // It stops on the first step past ρ = 1.5.
    let mut steps = 1;
    let mut last = s.beams[j].rest;
    while steps < 100_000 {
        s.step();
        steps += 1;
        if s.beams[j].rest == last {
            break;
        }
        last = s.beams[j].rest;
    }
    assert!(
        last / rest0 >= 1.5 && (last - 0.8 * STEP) / rest0 < 1.5,
        "{}",
        last / rest0
    );
    // Back on the long side: it retracts at 0.6 m/s down to ρ ≤ 1.0.
    place(&mut s, 2.25);
    s.step();
    assert_eq!(s.beams[j].rest, last);
    s.step();
    assert!((last - s.beams[j].rest - 0.6 * STEP).abs() < 1e-6);
    for _ in 0..100_000 {
        s.step();
    }
    let r = s.beams[j].rest / rest0;
    assert!(r <= 1.0 && r > 1.0 - 0.6 * STEP / rest0 - 1e-6, "{r}");
}

// --- §3 slide nodes ---

/// Rail nodes 0–3 along x at y = 1 on immovable nodes, beams (0, 1), (2, 1)
/// written in reverse, (2, 3); node 4 the slide node at `x`, 1 cm above the rail.
fn slide_rig(slide_line: &str, x: f32, fixed_slide: bool) -> ExplicitBeams {
    let text = format!(
        "slide rig
globals
100, 0, mat
minimass
1
nodes
0, 0, 1, 0
1, 1, 1, 0
2, 2, 1, 0
3, 3, 1, 0
4, {x}, 1.01, 0
fixes
0
1
2
3
{}
beams
0, 1
2, 1
2, 3
slidenodes
{slide_line}
end
",
        if fixed_slide { "4" } else { "" }
    );
    spawn(&text)
}

#[test]
fn explicit_slide_rail_segments_and_spring() {
    let mut s = slide_rig("4, 0, 1, 2, 3, s9000", 0.5, true);
    assert_eq!(s.segments.len(), 3);
    // The second pair resolves to the beam written (2, 1): its direction.
    let b = s.beams[s.segments[1].beam as usize];
    assert_eq!((b.a, b.b), (2, 1));
    assert_eq!(s.segments[0].prev, None);
    assert_eq!(s.segments[1].prev, Some(0));
    assert_eq!(s.segments[1].next, Some(2));
    assert_eq!(s.slides[0].segment, Some(0));
    s.step();
    // A spring of 9000 N/m over 1 cm toward the rail; the reaction split in half.
    let m = s.nodes[4].mass;
    let g = Vec3::new(0.0, m * GRAVITY, 0.0);
    let f = s.nodes[4].force - g;
    assert!((f - Vec3::new(0.0, -90.0, 0.0)).length() < 1e-2, "{f}");
    for end in [0, 1] {
        let e = s.nodes[end].force - Vec3::new(0.0, s.nodes[end].mass * GRAVITY, 0.0);
        assert!(
            (e - Vec3::new(0.0, 45.0, 0.0)).length() < 1e-2,
            "{end}: {e}"
        );
    }
    // The geometry's clamps.
    let (q, _, r) = segment_point(Vec3::ZERO, Vec3::X, Vec3::new(-0.5, 1.0, 0.0));
    assert_eq!((q, r), (Vec3::ZERO, 0.0));
    let (q, _, r) = segment_point(Vec3::ZERO, Vec3::X, Vec3::new(1.5, 1.0, 0.0));
    assert_eq!((q, r), (Vec3::X, 1.0));
    let (q, d, r) = segment_point(Vec3::ONE, Vec3::ONE, Vec3::ZERO);
    assert_eq!((q, r), (Vec3::ONE, 0.0));
    assert!((d - 3f32.sqrt()).abs() < 1e-6);
}

#[test]
fn explicit_slide_hops_one_segment_per_step() {
    let mut s = slide_rig("4, 0, 1, 2, 3", 0.5, true);
    // Moved onto the last segment: one hop per step.
    s.nodes[4].pos = Vec3::new(2.5, 1.0, 0.0);
    s.step();
    assert_eq!(s.slides[0].segment, Some(1));
    s.step();
    assert_eq!(s.slides[0].segment, Some(2));
    s.step();
    assert_eq!(s.slides[0].segment, Some(2));
    // Past the end: clamped to the last segment's end, pulled back along x.
    s.nodes[4].pos = Vec3::new(3.5, 1.0, 0.0);
    s.step();
    let f = s.nodes[4].force - Vec3::new(0.0, s.nodes[4].mass * GRAVITY, 0.0);
    assert!(f.x < -4.0e6 && f.y.abs() < 1.0, "{f}");
}

#[test]
fn explicit_slide_loop_missing_rail_break_and_stuck() {
    // A loop: the first and last segments are each other's neighbours.
    let text = "loop
nodes
0, 0, 1, 0
1, 1, 1, 0
2, 0, 2, 0
3, 0.4, 1.4, 0
beams
0, 1
1, 2
2, 0
slidenodes
3, 0, 1, 2, 0
end
";
    let s = spawn(text);
    assert_eq!(s.segments[0].prev, Some(2));
    assert_eq!(s.segments[2].next, Some(0));
    // A missing rail beam: inert.
    let mut s = slide_rig("4, 0, 3", 0.5, true);
    assert_eq!(s.slides[0].segment, None);
    s.step();
    assert_eq!(
        s.nodes[4].force,
        Vec3::new(0.0, s.nodes[4].mass * GRAVITY, 0.0)
    );
    // A 50 N break force: the breaking step's force applies, then none.
    let mut s = slide_rig("4, 0, 1, 2, 3, b50", 0.5, true);
    s.step();
    assert!(s.slides[0].broken_away);
    let g = Vec3::new(0.0, s.nodes[4].mass * GRAVITY, 0.0);
    assert!((s.nodes[4].force - g).length() > 1.0e4);
    s.step();
    assert_eq!(s.nodes[4].force, g);
    // Its current segment's beam broken: stuck, no force.
    let mut s = slide_rig("4, 0, 1, 2, 3", 0.5, true);
    let j = s.segments[0].beam as usize;
    s.beams[j].broken = true;
    s.nodes[4].pos = Vec3::new(1.5, 1.0, 0.0);
    s.step();
    assert_eq!(s.slides[0].segment, Some(0));
    assert_eq!(
        s.nodes[4].force,
        Vec3::new(0.0, s.nodes[4].mass * GRAVITY, 0.0)
    );
    // Unsupported options are refused.
    for line in [
        "4, 0, 1, r2",
        "4, 0, 1, d0.2",
        "4, 0, 1, g1",
        "4, 0, 1, ca",
        "4, 0",
    ] {
        let text = format!(
            "x\nnodes\n0, 0, 1, 0\n1, 1, 1, 0\n4, 0.5, 1, 0\nbeams\n0, 1\nslidenodes\n{line}\nend\n"
        );
        assert!(
            ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings()).is_err(),
            "{line}"
        );
    }
}

#[test]
fn explicit_slide_holds_a_free_node() {
    // A free slide node on the default spring rides the rail under gravity.
    let mut s = slide_rig("4, 0, 1, 2, 3", 0.5, false);
    s.nodes[4].vel = Vec3::new(2.0, 0.0, 0.0);
    for _ in 0..600 {
        s.step();
    }
    let p = s.nodes[4].pos;
    // Undamped: it swings about the rail by its 1 cm start offset, never falls.
    assert!((p.y - 1.0).abs() < 0.0101 && (p.x - 1.1).abs() < 1e-3, "{p}");
}

// --- §4 animators, §5 visual-only sections ---

#[test]
fn explicit_tacho_animator_is_a_constant_actuator() {
    let text = trigger_rig(
        "enable_advanced_deformation\nset_beam_defaults_scale 0.05, 0.005, 0.01, 0.5",
        "0, 1, 0.5, 1.5, 3, 4\nanimators\n0, 1, 0.5, tacho | inv\n1, 0, 0.5, tacho | shortlimit: 0.02 | longlimit: 0.05",
    );
    let mut s = spawn(&text);
    let n = s.beams.len();
    // The animators are the last beams, after the trigger.
    assert_eq!(s.beams[n - 3].kind, ExplicitKind::Trigger);
    for b in &s.beams[n - 2..] {
        assert_eq!(
            (b.kind, b.ty),
            (ExplicitKind::Plain, ExplicitType::Actuator)
        );
        assert_eq!((b.k, b.c), (450_000.0, 60.0));
        assert_eq!((b.strength, b.gate), (500_000.0, 4000.0));
    }
    for _ in 0..1000 {
        s.step();
    }
    for b in &s.beams[n - 2..] {
        assert_eq!(b.rest, b.rest0);
    }
    let speedo = trigger_rig("", "0, 1, 0.5, 1.5, 3, 4\nanimators\n0, 1, 0.5, speedo");
    assert!(ExplicitBeams::from_truck(&speedo, Vec3::ZERO, Vec3::ZERO, settings()).is_err());
}

#[test]
fn explicit_lockgroups_videocamera_flexbodies_change_nothing() {
    let plain = trigger_rig("", "0, 1, 0.5, 1.5, 3, 4");
    let extra = plain.replace(
        "triggers\n",
        "lockgroups\n1, 0, 1\nvideocamera\n0, 1, 2, 3, 0, 0.0, 0.0, 0.0, 0, -25, -90, 90, 512, 512, 0.1, 100, 1, -2, mirror\n\
         flexbodies\n0, 1, 2, 0.5, 0.0, 0.0, 25, 180, 0, body.mesh\nforset 0-3\ntriggers\n",
    );
    let (a, b) = (spawn(&plain), spawn(&extra));
    assert_eq!(a.nodes, b.nodes);
    assert_eq!(a.beams, b.beams);
}
