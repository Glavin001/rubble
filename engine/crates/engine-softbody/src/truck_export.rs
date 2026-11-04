//! A [`SoftVehicleDesc`] (from any loader: BeamNG JBeam, Rubble's generators) written
//! as a Rigs of Rods `.truck` file, so the same vehicle runs in Rigs of Rods and in
//! Rubble's `.truck` path. Written from the `.truck` format as Rubble's loader
//! ([`crate::truck`]) reads it; no third-party code.
//!
//! What carries over exactly: node positions and masses (every node loaded with its
//! own weight, no dry mass, the minimum mass under the lightest node), ground
//! contact and friction, contacters, fixed nodes, every beam's spring, damping,
//! yield threshold, strength and plastic coefficient (`set_beam_defaults` before
//! each run of equal values), support and rope beams, break groups (as detacher
//! groups), collision triangles (`cab … c`, others as visual `cab`), the reference
//! frame (`cameras`) and the self-collision range.
//!
//! What changes, reported in the warnings: bounded beams become `shocks` (their
//! limit spring and damping, bounds and precompression), hydros `hydros` with
//! their factor; anisotropic and pressured beams plain beams; passive break groups
//! active ones; and BeamNG's pressure wheels become `.truck` soft tyres (`wheels`)
//! on the same axle nodes, radius and width, with [`TyreDefaults`] (usually
//! [`TyreDefaults::for_vehicle`]): Rubble does not
//! model BeamNG's pressurised tyre.

use crate::desc::{BeamKind, SoftVehicleDesc};
use std::fmt::Write;

/// Soft tyre parameters for converted wheels (a 12-ray tyre that is stable at the
/// fixed 0.5 ms step with light rim nodes).
#[derive(Clone, Copy, Debug)]
pub struct TyreDefaults {
    pub rays: u32,
    pub mass: f32,
    pub spring: f32,
    pub damp: f32,
}

impl Default for TyreDefaults {
    fn default() -> Self {
        Self {
            rays: 12,
            mass: 15.0,
            spring: 800_000.0,
            damp: 300.0,
        }
    }
}

impl TyreDefaults {
    /// The defaults (sized for a 1,300 kg car) with tyre mass and spring scaled by the
    /// vehicle's mass over that (never below 1): the same deflection under load and
    /// the same spring over rim-node mass, so the same stability at the fixed step.
    pub fn for_vehicle(desc: &SoftVehicleDesc) -> Self {
        let d = Self::default();
        let scale = (desc.nodes.iter().map(|n| n.mass).sum::<f32>() / 1300.0).max(1.0);
        Self {
            mass: d.mass * scale,
            spring: d.spring * scale,
            ..d
        }
    }
}

/// A value `.truck` reads back as the same f32 (infinite as a huge finite value:
/// never reached).
fn num(v: f32) -> String {
    if v.is_finite() {
        format!("{v}")
    } else {
        "1e30".into()
    }
}

/// The `.truck` text and what did not carry over exactly.
pub fn desc_to_truck(desc: &SoftVehicleDesc, tyres: TyreDefaults) -> (String, Vec<String>) {
    let mut warnings = Vec::new();
    let mut t = String::new();
    let name = if desc.name.is_empty() {
        "Converted vehicle"
    } else {
        &desc.name
    };
    writeln!(t, "{}", name.lines().next().unwrap_or(name)).unwrap();
    writeln!(
        t,
        "; Converted by engine-softbody::truck_export from a Rubble vehicle description."
    )
    .unwrap();
    writeln!(t, "fileformatversion 450").unwrap();
    // Masses come from the nodes alone.
    writeln!(t, "globals\n0, 0, tracks/semi\n").unwrap();
    let lightest = desc
        .nodes
        .iter()
        .map(|n| n.mass)
        .fold(f32::INFINITY, f32::min);
    let minimass = if lightest.is_finite() {
        0.999 * lightest
    } else {
        1.0
    };
    writeln!(t, "minimass\n{}\n", num(minimass)).unwrap();
    writeln!(t, "enable_advanced_deformation").unwrap();
    if desc.collision_range > 0.0 {
        writeln!(t, "set_collision_range {}", num(desc.collision_range)).unwrap();
    }

    // Nodes, friction by `set_node_defaults` runs.
    let mut friction = f32::NAN;
    writeln!(t, "\nnodes").unwrap();
    for (i, n) in desc.nodes.iter().enumerate() {
        if n.friction != friction {
            friction = n.friction;
            writeln!(t, "set_node_defaults -1, {}, -1, n", num(friction)).unwrap();
        }
        let opts = if n.collision { "l" } else { "lc" };
        let p = n.position;
        writeln!(
            t,
            "{i}, {}, {}, {}, {opts}, {}",
            p[0],
            p[1],
            p[2],
            num(n.mass)
        )
        .unwrap();
    }

    // Beams, then the kinds `.truck` keeps in their own sections.
    let mut defaults: Option<(f32, f32, f32, f32, f32)> = None;
    let mut group = 0u32;
    let mut set_defaults = |t: &mut String, d: (f32, f32, f32, f32, f32)| {
        if defaults != Some(d) {
            defaults = Some(d);
            writeln!(
                t,
                "set_beam_defaults {}, {}, {}, {}, 0.05, tracks/beam, {}",
                num(d.0),
                num(d.1),
                num(d.2),
                num(d.3),
                num(d.4)
            )
            .unwrap();
        }
    };
    let mut set_group = |t: &mut String, g: u32| {
        if g != group {
            group = g;
            if g == 0 {
                writeln!(t, "detacher_group end").unwrap();
            } else {
                writeln!(t, "detacher_group {g}").unwrap();
            }
        }
    };
    let (mut aniso, mut passive) = (0, 0);
    writeln!(t, "\nbeams").unwrap();
    for b in &desc.beams {
        let opts = match b.kind {
            BeamKind::Normal => "n".to_string(),
            BeamKind::Support => format!("s, {}", num(b.long_bound)),
            BeamKind::Rope => "r".to_string(),
            BeamKind::Anisotropic => {
                aniso += 1;
                "n".to_string()
            }
            BeamKind::Bounded | BeamKind::Hydro => continue,
        };
        passive += b.break_group_passive as usize;
        set_defaults(
            &mut t,
            (b.spring, b.damp, b.deform, b.strength, b.plastic_coef),
        );
        set_group(&mut t, b.break_group);
        let opts = if b.visible {
            opts
        } else {
            opts.replacen('n', "i", 1)
        };
        writeln!(t, "{}, {}, {opts}", b.a, b.b).unwrap();
    }
    if desc.beams.iter().any(|b| b.kind == BeamKind::Bounded) {
        writeln!(t, "\nshocks").unwrap();
        for b in desc.beams.iter().filter(|b| b.kind == BeamKind::Bounded) {
            // Inside the bounds the shock's own spring and damping; past them the
            // beam defaults (the limit spring and damping).
            set_defaults(
                &mut t,
                (b.deform, b.limit_damp, f32::INFINITY, b.strength, 0.0),
            );
            set_group(&mut t, b.break_group);
            writeln!(
                t,
                "{}, {}, {}, {}, {}, {}, {}, n",
                b.a,
                b.b,
                num(b.spring),
                num(b.damp),
                num(b.short_bound),
                num(b.long_bound),
                num(b.precompression)
            )
            .unwrap();
        }
        warnings.push("bounded beams written as shocks".into());
    }
    if desc.beams.iter().any(|b| b.kind == BeamKind::Hydro) {
        writeln!(t, "\nhydros").unwrap();
        for b in desc.beams.iter().filter(|b| b.kind == BeamKind::Hydro) {
            set_defaults(&mut t, (b.spring, b.damp, f32::INFINITY, b.strength, 0.0));
            set_group(&mut t, b.break_group);
            writeln!(t, "{}, {}, {}, n", b.a, b.b, num(b.hydro_factor)).unwrap();
        }
        warnings.push("hydros written with their factor; no steering input drives them".into());
    }
    set_group(&mut t, 0);
    if aniso > 0 {
        warnings.push(format!("{aniso} anisotropic beams written as plain beams"));
    }
    if passive > 0 {
        warnings.push(format!(
            "{passive} beams in passive break groups written as active detacher groups"
        ));
    }

    // Triangles.
    if !desc.triangles.is_empty() {
        writeln!(t, "\nsubmesh\ncab").unwrap();
        for tr in &desc.triangles {
            let opts = if tr.collision { "c" } else { "n" };
            writeln!(
                t,
                "{}, {}, {}, {opts}",
                tr.nodes[0], tr.nodes[1], tr.nodes[2]
            )
            .unwrap();
        }
    }
    let contacters: Vec<String> = desc
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.contacter)
        .map(|(i, _)| i.to_string())
        .collect();
    if !contacters.is_empty() {
        writeln!(t, "\ncontacters").unwrap();
        for c in &contacters {
            writeln!(t, "{c}").unwrap();
        }
    }
    let fixed: Vec<String> = desc
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.fixed)
        .map(|(i, _)| i.to_string())
        .collect();
    if !fixed.is_empty() {
        writeln!(t, "\nfixes").unwrap();
        for f in &fixed {
            writeln!(t, "{f}").unwrap();
        }
    }
    if let Some(r) = &desc.refs {
        writeln!(t, "\ncameras\n{}, {}, {}", r.center, r.back, r.left).unwrap();
    }

    // Wheels: soft tyres on the same axles (their spokes and rings take the tyre
    // values, not the body's).
    if !desc.wheels.is_empty() {
        writeln!(t, "\nset_beam_defaults -1, -1, -1, -1\nwheels").unwrap();
        for w in &desc.wheels {
            writeln!(
                t,
                "{}, {}, {}, {}, {}, 9999, {}, {}, {}, {}, {}, {}, tracks/wheelface, tracks/wheelband1",
                num(w.radius),
                num(w.width),
                tyres.rays,
                w.node1,
                w.node2,
                if w.braked { 1 } else { 0 },
                if w.propulsion > 0 { 1 } else { 0 },
                w.arm.map_or("9999".to_string(), |a| a.to_string()),
                num(tyres.mass),
                num(tyres.spring),
                num(tyres.damp)
            )
            .unwrap();
        }
        warnings.push(format!(
            "{} pressure wheels written as {}-ray soft tyres ({} kg, {} N/m, {} N·s/m)",
            desc.wheels.len(),
            tyres.rays,
            tyres.mass,
            tyres.spring,
            tyres.damp
        ));
    }
    writeln!(t, "\nend").unwrap();
    (t, warnings)
}
