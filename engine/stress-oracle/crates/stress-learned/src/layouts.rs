//! Independent source geometries for layout-held-out impact experiments.
//! Every variant is built at the same 0.1 m chunk resolution; solver accuracy
//! settings are inherited unchanged. Openings are physical design parameters.
use crate::generate::Job;
use serde_json::json;
use stress_ref::{builders, math::Vec3, scene::Support};

#[derive(Clone, Copy, Debug)]
pub struct WallLayout {
    pub counts: [usize; 3],
    /// Half-open grid ranges [x_start,x_end,z_start,z_end]. Foundation is retained.
    pub opening: Option<[usize; 4]>,
}

pub fn wall_layouts() -> Vec<WallLayout> {
    vec![
        WallLayout {
            counts: [16, 1, 16],
            opening: None,
        },
        WallLayout {
            counts: [16, 2, 21],
            opening: None,
        },
        WallLayout {
            counts: [20, 3, 16],
            opening: None,
        },
        WallLayout {
            counts: [24, 1, 26],
            opening: None,
        },
        WallLayout {
            counts: [24, 3, 21],
            opening: None,
        },
        WallLayout {
            counts: [28, 2, 26],
            opening: None,
        },
        WallLayout {
            counts: [16, 2, 21],
            opening: Some([6, 10, 1, 13]),
        },
        WallLayout {
            counts: [20, 2, 21],
            opening: Some([4, 8, 1, 15]),
        },
        WallLayout {
            counts: [24, 2, 26],
            opening: Some([9, 15, 7, 17]),
        },
        WallLayout {
            counts: [24, 3, 26],
            opening: Some([4, 10, 7, 16]),
        },
        WallLayout {
            counts: [28, 2, 21],
            opening: Some([16, 23, 6, 15]),
        },
        WallLayout {
            counts: [20, 3, 26],
            opening: Some([7, 13, 1, 18]),
        },
    ]
}

pub fn layout_job(layout: WallLayout, layout_id: &str, case: usize, seed: u64, frames: u32) -> Job {
    let [nx, ny, nz] = layout.counts;
    assert!(
        nx > 2 && ny > 0 && nz > 2 && frames > 0,
        "invalid wall grid or duration"
    );
    if let Some([x0, x1, z0, z1]) = layout.opening {
        assert!(
            x0 > 0 && x0 < x1 && x1 < nx && z0 > 0 && z0 < z1 && z1 < nz,
            "opening must preserve piers, foundation and lintel"
        );
    }
    // Repeat these same load recipes for every layout, avoiding layout/load leakage.
    let (speed, mass, x_fraction, height_fraction, yaw): (f64, f64, f64, f64, f64) = match case {
        0 => (0., 1000., 0., 0.5, 0.),
        1 => (8., 250., -0.25, 0.35, 0.),
        2 => (15., 1000., 0.25, 0.65, -25.),
        3 => (25., 3000., 0., 0.5, 25.),
        _ => panic!("unknown impact recipe"),
    };
    let id = format!("v4_layout_{layout_id}_c{case}_s{seed}");
    let mut scene = builders::wall_impact(speed, &id);
    scene.benchmark = None;
    scene.description="Independent concrete wall layout; fixed chunk resolution and unchanged oracle solver settings".into();
    scene.sim.seed = seed;
    scene.sim.duration = frames as f64 * scene.sim.frame_dt;
    let width = nx as f64 * 0.1;
    let height = (nz - 1) as f64 * 0.1;
    let mut chunks = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if layout
                    .opening
                    .is_some_and(|[x0, x1, z0, z1]| x >= x0 && x < x1 && z >= z0 && z < z1)
                {
                    continue;
                }
                let mut c = builders::chunk(
                    Vec3::new(
                        -width * 0.5 + (x as f64 + 0.5) * 0.1,
                        (y as f64 + 0.5) * 0.1,
                        (z as f64 - 0.5) * 0.1,
                    ),
                    Vec3::splat(0.05),
                    "concrete",
                );
                if z == 0 {
                    c.support = Support::Fixed;
                    c.groups.push("foundation".into());
                }
                chunks.push(c);
            }
        }
    }
    let bonds = builders::auto_bonds(&chunks, |_, _| "concrete".into());
    scene.bodies = vec![builders::body("wall", chunks, bonds)];
    let p = &mut scene.impactors[0];
    p.mass = mass;
    p.position = [x_fraction * width, -0.251, height_fraction * height];
    let a = yaw.to_radians();
    p.velocity = [speed * a.sin(), speed * a.cos(), 0.];
    // The benchmark-specific regions/analytic metrics describe the original wall.
    // This catalog is evaluated from recorded stress/state arrays instead.
    scene.metrics.clear();
    scene.oracle.insert(
        "learned_impact".into(),
        json!({
            "catalog":"layouts-v4","split_group":id,"layout_group":layout_id,
            "parameters":{"counts":layout.counts,"chunk_size":0.1,"opening_grid":layout.opening,
                "case":case,"speed":speed,"mass":mass,"impact_x_fraction":x_fraction,
                "impact_height_fraction":height_fraction,"yaw_degrees":yaw,"seed":seed}
        }),
    );
    Job {
        event_id: id,
        structure_type: "concrete_wall_layout".into(),
        load_type: "layout_impact".into(),
        frames,
        qs_probe: false,
        scene,
    }
}

pub fn layout_jobs(seed: u64) -> Vec<Job> {
    wall_layouts()
        .iter()
        .enumerate()
        .flat_map(|(i, &layout)| {
            (0..4).map(move |case| layout_job(layout, &format!("{i:02}"), case, seed, 8))
        })
        .collect()
}

/// Separate scale diagnostic: 10,400 chunks, 10,240 free structural chunks.
/// Never truncate to fit a model; this is not part of the small-layout training set.
pub fn scale_layout_job(seed: u64) -> Job {
    layout_job(
        WallLayout {
            counts: [40, 4, 65],
            opening: None,
        },
        "scale10k",
        2,
        seed,
        1,
    )
}
