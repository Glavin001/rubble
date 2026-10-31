//! Size and cost of impact and explosion regions at the scale of the browser
//! demo (`impact.html`): the π-scaling crater, the region's cell and box, an
//! estimate of its particles, and the substeps of a 60 Hz frame at physical
//! stiffness (default sand, E = 50 MPa).
//!
//! `cargo run --release -p engine-terrain --example impact_scale`

use engine_terrain::plan::source_peak_speed;
use engine_terrain::zmodel::point_source_z;
use engine_terrain::*;
use glam::DVec3;

fn main() {
    let sand = SandMaterial::default();
    let c = sand.p_wave_speed();
    let g = 9.81;
    println!(
        "sand: E {:.0} MPa, density {} kg/m³, P-wave {c:.0} m/s",
        sand.youngs_modulus / 1e6,
        sand.density
    );
    let row = |name: String, cr: CraterPrediction, peak: f32, cpr: f32, margin: f64| {
        let cfg = TerrainConfig {
            cells_per_radius: cpr,
            max_blocks: 65535,
            ..TerrainConfig::default()
        };
        let plan = cr.region(&cfg, glam::Vec3::ZERO, margin, 0, 0).unwrap();
        let span = plan.end() - plan.origin;
        let s = plan.dx / cfg.particles_per_cell_axis as f32;
        let particles = (span.x / s) * (span.z / s) * (-plan.origin.y / s);
        let first = substeps_for(&cfg, 1.0 / 60.0, plan.dx, c, 1.5 * peak + 20.0)
            .unwrap()
            .0;
        let calm = substeps_for(&cfg, 1.0 / 60.0, plan.dx, c, 40.0).unwrap().0;
        println!(
            "{name}: R {:.1} m, depth {:.1} m, final rim {:.1} m, formation {:.1} s, peak {peak:.0} m/s \
             | {cpr} cells/R, margin {margin}: dx {:.2} m, {} blocks, ~{:.0}k particles, \
             substeps {first} at the source, {calm} at 20 m/s",
            cr.radius,
            cr.depth,
            cr.final_rim_radius,
            cr.formation_time,
            plan.dx,
            plan.blocks.x * plan.blocks.y * plan.blocks.z,
            particles / 1000.0
        );
    };
    for (a, u, d) in [
        (0.25f64, 12_000.0, 3000.0),
        (0.5, 12_000.0, 3000.0),
        (1.0, 12_000.0, 3000.0),
        (1.0, 20_000.0, 7800.0),
        (2.0, 20_000.0, 3000.0),
        // impact.html's default event.
        (8.0, 20_000.0, 3000.0),
    ] {
        let src = ImpactSource {
            impactor_radius: a,
            impactor_density: d,
            velocity: DVec3::new(0.0, -u, 0.0),
            target: CraterTarget::sand(&sand).unwrap(),
        };
        let cr = src.crater(g).unwrap();
        let f = src.field(g, point_source_z(src.target.ejecta.mu)).unwrap();
        let peak = source_peak_speed(&f);
        for (cpr, margin) in [(16.0, 0.6), (16.0, 1.0), (16.0, 2.0)] {
            row(
                format!(
                    "impactor {a} m, {u} m/s, {d} kg/m³ ({:.1e} J)",
                    src.kinetic_energy()
                ),
                cr,
                peak,
                cpr,
                margin,
            );
        }
    }
    for (kg, depth) in [
        (100.0f64, 0.5f64),
        (1000.0, 1.0),
        (10_000.0, 2.0),
        (100_000.0, 5.0),
        // impact.html's default charge: 1 kt at 8 m.
        (1.0e6, 8.0),
    ] {
        let ex = ExplosionSource::tnt(kg, depth, CraterTarget::sand(&sand).unwrap());
        match (
            ex.crater(g),
            ex.field(g, point_source_z(ex.target.ejecta.mu)),
        ) {
            (Ok(cr), Ok(f)) => row(
                format!("TNT {kg} kg at {depth} m"),
                cr,
                source_peak_speed(&f),
                16.0,
                0.6,
            ),
            (a, b) => println!("TNT {kg} kg at {depth} m: {:?} {:?}", a.err(), b.err()),
        }
    }
}
