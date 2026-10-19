//! Balanced prescribed-load recipes on existing, explicitly grouped layouts.
//! Changes physical loading only; geometry, material and solver settings are retained.
use crate::{
    generate::Job,
    layouts::{layout_job, wall_layouts},
};
use serde_json::json;
use stress_ref::scene::{ChunkSelector, LoadDesc, ProbeKind, Support, TimeFunction};

pub fn forcing_curve(kind: &str, peak: f64, duration: f64) -> TimeFunction {
    let start = 0.008;
    match kind {
        "constant" => TimeFunction::Constant { value: peak },
        "ramp" => TimeFunction::Ramp {
            t0: start,
            t1: start + duration,
            value: peak,
        },
        "half_sine" => TimeFunction::HalfSine {
            start,
            duration,
            peak,
        },
        "friedlander" => TimeFunction::Friedlander {
            arrival: start,
            duration,
            peak,
            decay: 1.0,
        },
        // Two separated triangular pulses, with every knot retained in scene/replay inputs.
        "table" => TimeFunction::Table {
            points: vec![
                [start, 0.],
                [start + duration * 0.15, peak],
                [start + duration * 0.30, 0.],
                [start + duration * 0.55, 0.],
                [start + duration * 0.75, peak * 0.6],
                [start + duration, 0.],
            ],
        },
        _ => panic!("unknown forcing curve: {kind}"),
    }
}

/// Four existing layout groups: 02/08 train, 01 validation, 07 test in the
/// predeclared v4 split. This catalog adds forcing diversity, not new geometry.
pub fn forcing_jobs(seed: u64) -> Vec<Job> {
    let layouts = wall_layouts();
    let mut jobs = Vec::new();
    for layout_index in [1usize, 2, 7, 8] {
        for load_kind in ["pressure", "point_force"] {
            for curve_kind in ["constant", "ramp", "half_sine", "friedlander", "table"] {
                for level in 0..2 {
                    // Constant forcing has no duration parameter: record it once.
                    for duration_index in 0..if curve_kind == "constant" { 1 } else { 2 } {
                        let duration = [0.015, 0.060][duration_index];
                        let peak = if load_kind == "pressure" {
                            [2e4, 2e5][level]
                        } else {
                            [1e3, 1e4][level]
                        };
                        let layout_id = format!("{layout_index:02}");
                        let mut job = layout_job(layouts[layout_index], &layout_id, 0, seed, 8);
                        let id = format!("v8_forcing_{layout_id}_{load_kind}_{curve_kind}_a{level}_d{duration_index}_s{seed}");
                        job.event_id = id.clone();
                        job.scene.name = id.clone();
                        job.load_type = format!("prescribed_{load_kind}");
                        job.scene.description = "Balanced prescribed loading on an existing wall layout; unchanged oracle accuracy settings".into();
                        job.scene.impactors.clear();
                        // The base wall's ram probe is invalid once the ram is removed.
                        // Retain structural diagnostics such as foundation reactions.
                        job.scene.probes.retain(|p| {
                            !matches!(
                                p.kind,
                                ProbeKind::ImpactorVelocity { .. }
                                    | ProbeKind::ImpactorPosition { .. }
                            )
                        });
                        job.scene.loads.clear();
                        let body = &job.scene.bodies[0];
                        let free: Vec<_> = body
                            .chunks
                            .iter()
                            .enumerate()
                            .filter(|(_, c)| c.support == Support::None)
                            .map(|(i, _)| i)
                            .collect();
                        let wave = forcing_curve(curve_kind, peak, duration);
                        let mut application = None;
                        if load_kind == "pressure" {
                            job.scene.loads.push(LoadDesc::Pressure {
                                body: "wall".into(),
                                chunks: ChunkSelector::Indices(free),
                                face_normal: [0., -1., 0.],
                                pressure: wave,
                            });
                        } else {
                            let [nx, _, nz] = layouts[layout_index].counts;
                            let target =
                                [0.25 * nx as f64 * 0.1, 0.05, 0.65 * (nz - 1) as f64 * 0.1];
                            let distance = |i: usize| {
                                body.chunks[i]
                                    .center
                                    .iter()
                                    .zip(target)
                                    .map(|(a, b)| (a - b).powi(2))
                                    .sum::<f64>()
                            };
                            let chunk = *free
                                .iter()
                                .min_by(|&&a, &&b| distance(a).total_cmp(&distance(b)))
                                .unwrap();
                            let mut point = body.chunks[chunk].center;
                            point[2] += 0.04;
                            application =
                                Some(json!({"chunk":chunk,"point":point,"direction":[0.,1.,0.]}));
                            job.scene.loads.push(LoadDesc::PointForce {
                                body: "wall".into(),
                                chunk,
                                direction: [0., 1., 0.],
                                magnitude: wave,
                                point: Some(point),
                            });
                        }
                        job.scene.oracle.insert("learned_impact".into(),json!({
                            "catalog":"forcing-v8","split_group":id,"layout_group":layout_id,"layout_catalog":"layouts-v4",
                            "parameters":{"layout_index":layout_index,"load_kind":load_kind,"curve_kind":curve_kind,
                                "peak":peak,"units":if load_kind=="pressure" {"Pa"} else {"N"},
                                "duration":if curve_kind=="constant" {None} else {Some(duration)},
                                "application":application,"seed":seed}
                        }));
                        jobs.push(job);
                    }
                }
            }
        }
    }
    jobs
}

/// Magnitude/location pilot on an already-declared TRAINING geometry only.
/// Do not treat these as an independent test set or tune held-out layouts here.
pub fn point_force_pilot_jobs(seed: u64) -> Vec<Job> {
    let base=forcing_jobs(seed).into_iter().find(|j|
        j.event_id==format!("v8_forcing_02_point_force_constant_a1_d0_s{seed}")).unwrap();
    let layout=wall_layouts()[2];
    let width=layout.counts[0] as f64*0.1;
    let height=(layout.counts[2]-1) as f64*0.1;
    let mut jobs=Vec::new();
    for (position_index,(x_fraction,height_fraction)) in
        [(0.25,0.65),(0.,0.5),(-0.35,0.2)].into_iter().enumerate() {
        for (amplitude_index,peak) in [1e4,1e5,1e6].into_iter().enumerate() {
            for curve in ["constant","half_sine"] {
                let mut job=base.clone();
                let target=[x_fraction*width,0.05,height_fraction*height];
                let body=&job.scene.bodies[0];
                let chunk=body.chunks.iter().enumerate().filter(|(_,c)|c.support==Support::None)
                    .min_by(|(_,a),(_,b)| {
                        let distance=|v:&[f64;3]|v.iter().zip(target).map(|(x,y)|(x-y).powi(2)).sum::<f64>();
                        distance(&a.center).total_cmp(&distance(&b.center))
                    }).unwrap().0;
                let mut point=body.chunks[chunk].center;point[2]+=0.04;
                let id=format!("v9_point_pilot_02_p{position_index}_a{amplitude_index}_{curve}_s{seed}");
                job.event_id=id.clone();job.scene.name=id.clone();
                job.scene.description="Training-layout point-force magnitude/location pilot; unchanged geometry, materials and accuracy settings".into();
                job.scene.loads=vec![LoadDesc::PointForce{body:"wall".into(),chunk,direction:[0.,1.,0.],
                    magnitude:forcing_curve(curve,peak,0.06),point:Some(point)}];
                job.scene.oracle.insert("learned_impact".into(),json!({
                    "catalog":"point-force-pilot-v9","split_group":id,"layout_group":"02","layout_catalog":"layouts-v4",
                    "partition_intent":"train_only_pilot",
                    "parameters":{"layout_index":2,"load_kind":"point_force","curve_kind":curve,
                        "peak":peak,"units":"N","duration":if curve=="constant" {None} else {Some(0.06)},
                        "application":{"chunk":chunk,"point":point,"direction":[0.,1.,0.],
                            "x_fraction":x_fraction,"height_fraction":height_fraction},"seed":seed}
                }));
                jobs.push(job);
            }
        }
    }
    jobs
}

/// Finite, impulse-controlled pulses on training layout 02. Peak and duration
/// vary independently at fixed impulse; the force always ends before the clip.
/// These are range-finding cases, not independent held-out geometries.
pub fn point_impulse_pilot_jobs(seed: u64) -> Vec<Job> {
    point_impulse_jobs(seed, [12., 36., 96.], "point-impulse-pilot-v10", "v10_point_impulse")
}

/// Second predeclared range after the full 108-case pilot yielded three breaks.
/// Retain every factorial outcome; this remains training-layout range finding.
pub fn point_impulse_range_jobs(seed: u64) -> Vec<Job> {
    point_impulse_jobs(seed, [192., 384., 768.], "point-impulse-range-v11", "v11_point_impulse")
}

fn point_impulse_jobs(seed: u64, impulses: [f64; 3], catalog: &str, prefix: &str) -> Vec<Job> {
    let bases: Vec<_> = point_force_pilot_jobs(seed).into_iter().filter(|j| {
        let p = &j.scene.oracle["learned_impact"]["parameters"];
        p["curve_kind"] == "constant" && p["peak"].as_f64() == Some(1e4)
    }).collect();
    let mut jobs = Vec::new();
    for (position_index, base) in bases.into_iter().enumerate() {
        for (impulse_index, impulse) in impulses.into_iter().enumerate() {
            for (duration_index, duration) in [0.001, 0.004, 0.012].into_iter().enumerate() {
                for (direction_index, direction) in [[0., 1., 0.], [1., 0., 0.]].into_iter().enumerate() {
                    for (offset_index, offset) in [0., 0.04].into_iter().enumerate() {
                        let mut job = base.clone();
                        let LoadDesc::PointForce { chunk, .. } = &base.scene.loads[0] else { unreachable!() };
                        let mut point = base.scene.bodies[0].chunks[*chunk].center;
                        point[2] += offset;
                        // Integral of peak*sin(pi*t/duration) is 2*peak*duration/pi.
                        let peak = impulse * std::f64::consts::PI / (2. * duration);
                        let id = format!("{prefix}_02_p{position_index}_j{impulse_index}_d{duration_index}_n{direction_index}_o{offset_index}_s{seed}");
                        job.event_id = id.clone();
                        job.scene.name = id.clone();
                        job.scene.description = "Finite impulse-controlled point-force pilot; unchanged geometry, materials and oracle accuracy".into();
                        job.scene.loads = vec![LoadDesc::PointForce {
                            body: "wall".into(), chunk: *chunk, direction,
                            magnitude: forcing_curve("half_sine", peak, duration), point: Some(point),
                        }];
                        let old = &base.scene.oracle["learned_impact"]["parameters"]["application"];
                        job.scene.oracle.insert("learned_impact".into(), json!({
                            "catalog":catalog, "split_group":id,
                            "layout_group":"02", "layout_catalog":"layouts-v4", "partition_intent":"train_only_pilot",
                            "parameters":{"layout_index":2,"load_kind":"point_force","curve_kind":"half_sine",
                                "peak":peak,"units":"N","duration":duration,"start":0.008,
                                "impulse":impulse,"impulse_units":"N s",
                                "application":{"chunk":chunk,"point":point,"direction":direction,"offset_z":offset,
                                    "x_fraction":old["x_fraction"],"height_fraction":old["height_fraction"]},"seed":seed}
                        }));
                        jobs.push(job);
                    }
                }
            }
        }
    }
    jobs
}
