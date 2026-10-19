//! Parameterized scene variations; never change oracle accuracy settings.
use crate::generate::Job;
use serde_json::{json, Value};
use stress_ref::{
    builders,
    scene::{ImpactorShape, LoadDesc, Scene},
    showcases,
};

fn materials(scene: &mut Scene, strength: f64, toughness: f64, stiffness: f64) {
    // Projectile/ground steel stays identical. Vary the structural material law.
    for (name, m) in &mut scene.materials {
        if name == "steel" {
            continue;
        }
        m.tensile_strength *= strength;
        m.compressive_strength *= strength;
        m.cohesion *= strength;
        if let Some(cap) = &mut m.shear_cap {
            *cap *= strength;
        }
        m.fracture_energy.tension *= toughness;
        m.fracture_energy.shear *= toughness;
        m.fracture_energy.compression *= toughness;
        m.youngs_modulus *= stiffness;
    }
}
fn make(id: String, family: &str, mut scene: Scene, frames: u32, parameters: Value) -> Job {
    scene.name = id.clone();
    scene.description = "Learned stress diversity v3; explicit oracle accuracy unchanged".into();
    scene.oracle.insert(
        "learned_impact".into(),
        json!({"catalog":"permutations-v3","parameters":parameters,"split_group":id}),
    );
    Job {
        event_id: id,
        structure_type: family.into(),
        load_type: "parameter_sweep".into(),
        frames,
        qs_probe: false,
        scene,
    }
}
/// Shorten the recording/prediction interval without shortening the physical clip.
/// This is an explicit dataset experiment; the engine's default cadence and all
/// stability/accuracy settings remain unchanged. Related cadences keep split_group.
pub fn subdivide_frames(job: &mut Job, factor: u32) -> Result<(), String> {
    if factor == 0 || job.frames == 0 || !job.scene.sim.frame_dt.is_finite() || job.scene.sim.frame_dt <= 0.0 {
        return Err("frame subdivision requires positive factor, frame count and finite dt".into());
    }
    if factor == 1 { return Ok(()); }
    let frames = job.frames.checked_mul(factor).ok_or("subdivided frame count overflows u32")?;
    let original_dt = job.scene.sim.frame_dt;
    let dt = original_dt / factor as f64;
    if dt <= 0.0 { return Err("subdivided dt underflow".into()); }
    let duration = original_dt * job.frames as f64;
    job.scene.oracle.get_mut("learned_impact").ok_or("subdivision requires recorded recipe provenance")?["recording"] =
        json!({"frame_subdivision": factor, "original_frame_dt": original_dt, "duration": duration});
    job.scene.sim.frame_dt = dt;
    job.scene.sim.duration = duration;
    job.frames = frames;
    job.event_id = format!("{}_sub{factor}", job.event_id);
    job.scene.name = job.event_id.clone();
    Ok(())
}

/// 80 events: paired sweeps plus combined factors. Seed replicas are opt-in.
pub fn permutation_jobs(seed: u64) -> Vec<Job> {
    let mut jobs = Vec::new();
    // 32 impacts: location/height, mass, material, geometry and approach angle.
    for i in 0..32usize {
        let mut s = builders::wall_impact(15.0, "v3_wall");
        s.sim.seed = seed;
        let mut x = 0.0;
        let mut z = 1.0;
        let mut mass = 1000.0;
        let mut speed = 15.0;
        let mut strength = 1.0;
        let mut toughness = 1.0;
        let mut size = 1.0;
        let mut yaw = 0.0_f64;
        match i {
            0..=5 => {
                x = [-0.65, 0., 0.65][i % 3];
                z = [0.45, 1.55][i / 3];
            }
            6..=10 => mass = [100., 300., 600., 2000., 4000.][i - 6],
            11..=15 => strength = [0.3, 0.5, 0.8, 1.5, 2.5][i - 11],
            16..=19 => {
                toughness = [0.3, 0.6, 1.8, 3.0][i - 16];
            }
            20..=23 => {
                size = [0.5, 0.75, 1.25, 1.5][i - 20];
            }
            24..=27 => {
                yaw = [-35., -15., 15., 35.][i - 24];
                x = if i % 2 == 0 { -0.35 } else { 0.35 };
            }
            _ => {
                mass = [250., 500., 2000., 3000.][i - 28];
                speed = [8., 25., 12., 30.][i - 28];
                strength = [0.5, 1.5, 2., 0.8][i - 28];
                x = [-0.55, 0.55, -0.3, 0.3][i - 28];
                z = [0.6, 1.4, 1.6, 0.4][i - 28];
            }
        }
        let a = yaw.to_radians();
        let p = &mut s.impactors[0];
        p.mass = mass;
        p.shape = ImpactorShape::Box {
            half_extents: [0.2 * size, 0.25 * size, 0.2 * size],
        };
        p.position = [x, -0.25 * size - 0.001, z];
        p.velocity = [speed * a.sin(), speed * a.cos(), 0.];
        materials(&mut s, strength, toughness, 1.0);
        jobs.push(make(format!("v3_wall_{i:02}_s{seed}"),"wall_b5",s,8,json!({"impact_x":x,"impact_z":z,"mass":mass,"speed":speed,"strength_scale":strength,"toughness_scale":toughness,"projectile_size_scale":size,"yaw_degrees":yaw,"seed":seed})));
    }
    // 20 house impacts target facade edges, lintels, lower and upper wall bands.
    for i in 0..20usize {
        let speed = if i < 12 {
            15.0
        } else {
            [6., 10., 20., 30.][i % 4]
        };
        let mut s = showcases::masonry_house(showcases::HouseEvent::Impact { speed }, "v3_house");
        s.sim.seed = seed;
        let y = [0.35, 1.5, 2.65][i % 3];
        let z = [0.4, 0.85, 1.4, 1.9][(i / 3) % 4];
        let mass = if i < 12 {
            1500.0
        } else {
            [400., 800., 2500., 4000.][i % 4]
        };
        let strength = if i < 12 {
            1.0
        } else {
            [0.5, 0.8, 1.5, 2.0][i % 4]
        };
        s.impactors[0].position[1] = y;
        s.impactors[0].position[2] = z;
        s.impactors[0].mass = mass;
        if i >= 16 {
            s.impactors[0].crush = None;
        }
        materials(&mut s, strength, 1., 1.);
        jobs.push(make(format!("v3_house_{i:02}_s{seed}"),"masonry_house",s,24,json!({"impact_y":y,"impact_z":z,"mass":mass,"speed":speed,"strength_scale":strength,"crushable":i<16,"seed":seed})));
    }
    // 16 blasts vary charge, standoff, lateral origin, height and arrival time.
    for i in 0..16usize {
        let tnt = [0.25, 0.75, 2., 5.][i % 4];
        let distance = [1., 2., 4., 6.][(i / 4) % 4];
        let x = [0.5, 1.5, 2.5, 1.5][i % 4];
        let z = if i % 2 == 0 { 0.4 } else { 1.8 };
        let time = if i >= 12 { 0.004 } else { 0. };
        let mut s = showcases::masonry_house(showcases::HouseEvent::Blast { tnt }, "v3_blast");
        s.sim.seed = seed;
        for l in &mut s.loads {
            if let LoadDesc::Blast {
                position, time: t, ..
            } = l
            {
                *position = [x, -distance, z];
                *t = time;
            }
        }
        jobs.push(make(format!("v3_blast_{i:02}_s{seed}"),"masonry_house",s,24,json!({"blast_tnt_mass":tnt,"standoff":distance,"origin":[x,-distance,z],"arrival":time,"seed":seed})));
    }
    // Low and near-threshold cases preserve negative coverage across constitutive changes.
    for i in 0..12usize {
        let speed = [0., 1., 3., 6.][i % 4];
        let mut s = builders::masonry_wall(speed, "v3_masonry");
        s.sim.seed = seed;
        let strength = [0.5, 1., 2.][i / 4];
        let stiffness = [0.5, 1., 2.][(i / 4 + i % 4) % 3];
        materials(&mut s, strength, 1., stiffness);
        jobs.push(make(
            format!("v3_masonry_{i:02}_s{seed}"),
            "masonry_wall",
            s,
            8,
            json!({"speed":speed,"strength_scale":strength,"modulus_scale":stiffness,"seed":seed}),
        ));
    }
    jobs
}

/// 52 paired spin experiments on four existing wall recipes. All descendants
/// retain their source split group: these are new loads, not new geometries.
/// At seed 101 these four source recipes belong to the established train set.
pub fn projectile_spin_jobs(seed: u64) -> Vec<Job> {
    let bases = permutation_jobs(seed);
    let mut jobs = Vec::new();
    for source_index in [0usize, 6, 11, 19] {
        let source = &bases[source_index];
        let mut variants = vec![("zero".to_string(), [0.0; 3])];
        for (axis, axis_name) in ["x", "y", "z"].iter().enumerate() {
            for rate in [-20.0_f64, -10.0, 10.0, 20.0] {
                let mut spin = [0.0; 3];
                spin[axis] = rate;
                variants.push((format!("{axis_name}_{}{:.0}", if rate < 0.0 { "m" } else { "p" }, rate.abs()), spin));
            }
        }
        for (variant, spin) in variants {
            let mut job = source.clone();
            job.event_id = format!("v22_spin_wall_{source_index:02}_{variant}_s{seed}");
            job.scene.name = job.event_id.clone();
            job.scene.description = "Paired projectile-spin sweep; source geometry, material and solver accuracy retained".into();
            job.scene.impactors[0].angular_velocity = spin;
            job.scene.oracle.insert("learned_impact".into(), json!({
                "catalog":"projectile-spin-v22", "split_group":source.event_id,
                "parameters":{"source_event":source.event_id,
                    "source_parameters":source.scene.oracle["learned_impact"]["parameters"],
                    "angular_velocity_radps":spin, "seed":seed}
            }));
            jobs.push(job);
        }
    }
    jobs
}
