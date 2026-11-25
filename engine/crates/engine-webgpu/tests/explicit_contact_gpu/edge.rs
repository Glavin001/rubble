//! Edge vehicle contact and contact offsets on the GPU against the CPU reference (a
//! module of explicit_contact_gpu.rs).

use super::*;

/// A free tetrahedron (no beams: four contact points) with its faces wound outward.
fn tetra(points: [Vec3; 4], vel: Vec3) -> ExplicitBeams {
    let mut s = vehicle(&points);
    let c = (points[0] + points[1] + points[2] + points[3]) / 4.0;
    for f in [[0u32, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
        let [a, b, d] = f.map(|i| points[i as usize]);
        let n = (b - a).cross(d - a);
        let nodes = if n.dot((a + b + d) / 3.0 - c) < 0.0 { [f[0], f[2], f[1]] } else { f };
        s.triangles.push(Triangle { nodes, wait: 0, idle: 0 });
    }
    for n in &mut s.nodes {
        n.vel = vel;
        n.flags |= NODE_NO_GROUND | NODE_CONTACTER;
    }
    s.settings.triangle_throttle = false;
    s.finish();
    s.fixed_contact = true;
    s.two_body_contact = true;
    s.dissipative_contact = true;
    s.edge_contact = true;
    s
}

/// Edge vehicle contact (`ExplicitBeams::edge_contact`, explicit_beams_vv_edges.slang) on
/// the GPU as on the CPU, bit for bit with bit-exact results on: two tetrahedra whose
/// ridges cross (edge against edge, no node over a face at first), the upper one falling
/// onto the lower; then a third whose corner comes down onto the lower ridge (node against
/// edge). One-step frames, then 17-step ones.
#[test]
fn edge_contact_steps_like_the_cpu() {
    edge_scene(false, false);
}

/// [`edge_contact_steps_like_the_cpu`] on the split group path (the vehicle lab's: a group's
/// tests across workgroups, its levels after them; the fleet-wide schedules off with edge
/// contact, `PERF_FLEET_FAMILY`).
#[test]
fn edge_contact_on_the_split_path_steps_like_the_cpu() {
    edge_scene(true, false);
}

/// [`edge_contact_steps_like_the_cpu`] in a contact group of five (the larger groups'
/// owner-by-owner pass, vvPass): two more tetrahedra coming down on the lower ridge's far end.
#[test]
fn edge_contact_in_a_large_group_steps_like_the_cpu() {
    edge_scene(false, true);
}

fn edge_scene(split: bool, large: bool) {
    let lower = tetra(
        [Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0), Vec3::new(0.0, -0.4, 0.4), Vec3::new(0.0, -0.4, -0.4)],
        Vec3::ZERO,
    );
    let upper = tetra(
        [Vec3::new(0.01, 0.02, -0.5), Vec3::new(-0.01, 0.02, 0.5), Vec3::new(0.4, 0.42, 0.0), Vec3::new(-0.4, 0.42, 0.0)],
        Vec3::new(0.0, -2.0, 0.0),
    );
    // A corner over the lower ridge's far part (x 0.35), outside both of its faces' prisms.
    let corner = tetra(
        [Vec3::new(0.35, 0.03, 0.0), Vec3::new(0.55, 0.5, 0.2), Vec3::new(0.25, 0.5, 0.25), Vec3::new(0.3, 0.5, -0.25)],
        Vec3::new(0.0, -1.5, 0.0),
    );
    let mut fleet = vec![lower, upper, corner];
    if large {
        // (Past the kernels' VV_PART, 8 members: twelve bodies over one ridge.)
        for i in 0..9 {
            let (x, z) = (-0.4 + 0.07 * i as f32, if i % 2 == 0 { 0.02f32 } else { -0.03 });
            fleet.push(tetra(
                [Vec3::new(x, 0.03, z), Vec3::new(x - 0.2, 0.5, z + 0.2), Vec3::new(x + 0.1, 0.5, z + 0.25), Vec3::new(x + 0.05, 0.5, z - 0.25)],
                Vec3::new(0.0, -1.2, 0.0),
            ));
        }
    }
    let Some(mut g) = gpu(&fleet) else {
        eprintln!("no GPU: skipped");
        return;
    };
    g.set_bit_exact(true);
    if split {
        g.set_group_split(true);
        g.set_group_split_tests(true);
    }
    let mut off = fleet.clone();
    for v in &mut off {
        v.edge_contact = false;
    }
    let mut w = ExplicitBeamsWorld::new(fleet);
    // (EDGE_DEBUG=1: one-step frames against the CPU with and without edge contact, to the
    // first step that differs.)
    if std::env::var("EDGE_DEBUG").is_ok() {
        let mut wo = ExplicitBeamsWorld::new(off);
        for step in 0..60 {
            w.frame(1);
            wo.frame(1);
            g.frame(1);
            let pos = g.positions();
            let nv = w.vehicles.len();
            let d: Vec<f32> = (0..nv).map(|k| rms(&pos[k], &w.vehicles[k].positions())).collect();
            let doff: Vec<f32> = (0..nv).map(|k| rms(&pos[k], &wo.vehicles[k].positions())).collect();
            let cpu: Vec<f32> = (0..nv).map(|k| rms(&wo.vehicles[k].positions(), &w.vehicles[k].positions())).collect();
            eprintln!("step {step}: GPU-CPU {d:?}  GPU-CPU(off) {doff:?} CPU on-off {cpu:?} edge hits {}", w.edge_stats[5]);
            if d.iter().any(|&x| x != 0.0) {
                break;
            }
        }
        panic!("EDGE_DEBUG: stepped to the first difference");
    }
    for (frames, steps) in [(60, 1), (6, 17)] {
        for _ in 0..frames {
            w.frame(steps);
            g.frame(steps);
        }
    }
    g.wait();
    g.contact_status().unwrap();
    if large {
        assert!(g.contact_groups() >= 1, "contact groups {}", g.contact_groups());
        assert!(w.vehicles.len() > 8, "a group past VV_PART");
    }
    eprintln!("edge pass stats (tested, end, parallel, facing, deep, hits, crossed): {:?}", w.edge_stats);
    assert!(w.edge_stats[5] > 10, "edge hits {:?}", w.edge_stats);
    let pos = g.positions();
    for k in 0..w.vehicles.len() {
        let d = rms(&pos[k], &w.vehicles[k].positions());
        eprintln!("vehicle {k}: GPU–CPU {d:.2e} m after {} steps", w.steps);
        assert_eq!(d, 0.0, "vehicle {k}: GPU–CPU {d} m");
    }
    // And edge contact acted: the CPU with it off ends elsewhere.
    let mut wo = ExplicitBeamsWorld::new(off);
    for (frames, steps) in [(60, 1), (6, 17)] {
        for _ in 0..frames {
            wo.frame(steps);
        }
    }
    let moved = (0..wo.vehicles.len()).map(|k| rms(&wo.vehicles[k].positions(), &pos[k])).fold(0.0f32, f32::max);
    eprintln!("with edge contact off the CPU ends {moved:.3} m away");
    assert!(moved > 1e-3, "edge contact made no difference ({moved} m)");
}

/// Edge contact between two whole cars (the procedural sedan, exact settings, the BeamNG
/// profile's contact: two-body, dissipative, fixed-point sums, anti-clip): one drives into
/// the other's rear at 15 m/s, the other backing at 5 m/s; GPU == CPU bit for bit through
/// the crash (the edge grid finds the same pairs as testing all of them).
#[test]
fn edge_contact_two_sedans_step_like_the_cpu() {
    let text = sedan_text();
    let make = |at: Vec3, v: Vec3| {
        let mut s = ExplicitBeams::from_truck(&text, at, v, EXACT).unwrap();
        s.roll_tyres(v);
        s.two_body_contact = true;
        s.dissipative_contact = true;
        s.fixed_contact = true;
        s.anti_clip = true;
        s.edge_contact = true;
        s
    };
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let (zmin, zmax) = probe.nodes.iter().fold((f32::MAX, f32::MIN), |(a, b), n| (a.min(n.pos.z), b.max(n.pos.z)));
    let fleet = vec![
        make(Vec3::new(0.0, 0.01 - low, zmax - zmin + 0.3), Vec3::new(0.0, 0.0, -5.0)),
        make(Vec3::new(0.0, 0.01 - low, 0.0), Vec3::new(0.0, 0.0, 15.0)),
    ];
    let Some(mut g) = gpu(&fleet) else {
        eprintln!("no GPU: skipped");
        return;
    };
    g.set_bit_exact(true);
    let mut w = ExplicitBeamsWorld::new(fleet);
    if std::env::var("EDGE_STEP_DEBUG").is_ok() {
        for step in 0..720 {
            let hits = w.edge_stats[5];
            w.frame(1);
            g.frame(1);
            let pos = g.positions();
            let d: Vec<f32> = (0..2).map(|k| rms(&pos[k], &w.vehicles[k].positions())).collect();
            let dump = std::env::var("EDGE_DUMP_STEP").ok().and_then(|x| x.parse::<usize>().ok()) == Some(step + 1);
            if dump && d.iter().all(|&x| x == 0.0) {
                let pt = g.portable(0.0, "");
                let words: Vec<u32> = bytemuck::cast_slice(&pt.bindings[0]).to_vec();
                let bm: Vec<f32> = bytemuck::cast_slice(&pt.bindings[4]).to_vec();
                let snap = words[72] as usize;
                let n0 = w.vehicles[0].nodes.len();
                for (lab, gi) in [("v0 14", 14), ("v0 8", 8), ("v1 166", n0 + 166), ("v1 162", n0 + 162)] {
                    let at = (snap + gi) * 4;
                    eprintln!("gpu snap {lab} {:?}", &bm[at..at + 3]);
                }
            }
            if d.iter().any(|&x| x != 0.0) {
                for k in 0..2 {
                    let cpu = w.vehicles[k].positions();
                    let mut v: Vec<(f32, usize)> = pos[k].iter().zip(&cpu).enumerate().map(|(i, (a, b))| ((*a - *b).length(), i)).filter(|x| x.0 > 0.0).collect();
                    v.sort_by(|a, b| b.0.total_cmp(&a.0));
                    eprintln!("vehicle {k}: {} nodes differ; top {:?}", v.len(), &v[..v.len().min(6)]);
                }
                let pt = g.portable(0.0, "");
                let words: Vec<u32> = bytemuck::cast_slice(&pt.bindings[0]).to_vec();
                let bm: Vec<f32> = bytemuck::cast_slice(&pt.bindings[4]).to_vec();
                let snap = words[72] as usize;
                let n0 = w.vehicles[0].nodes.len();
                for (lab, gi) in [("v0 14", 14), ("v0 8", 8), ("v1 166", n0 + 166), ("v1 162", n0 + 162)] {
                    let at = (snap + gi) * 4;
                    eprintln!("gpu snap {lab} {:?}", &bm[at..at + 3]);
                }
                panic!("first difference at step {step}: {d:?}; edge hits this step {}, total {:?}", w.edge_stats[5] - hits, w.edge_stats);
            }
        }
    }
    for (frames, steps) in [(20, 17), (40, 1), (20, 17)] {
        for _ in 0..frames {
            w.frame(steps);
            g.frame(steps);
        }
    }
    g.wait();
    g.contact_status().unwrap();
    eprintln!("edge pass stats (tested, end, parallel, facing, deep, hits, crossed): {:?}; node hits {}", w.edge_stats, w.total.hits);
    assert!(w.edge_stats[5] > 100, "edge hits {:?}", w.edge_stats);
    let pos = g.positions();
    for k in 0..2 {
        let d = rms(&pos[k], &w.vehicles[k].positions());
        eprintln!("sedan {k}: GPU–CPU {d:.2e} m after {} steps", w.steps);
        assert_eq!(d, 0.0, "sedan {k}: GPU–CPU {d} m");
    }
}

/// [`edge_contact_two_sedans_step_like_the_cpu`] with four sedans: two head-on pairs side by
/// side, overlapping, so the group's candidate lists live long and are rebuilt often.
#[test]
fn edge_contact_four_sedans_step_like_the_cpu() {
    four_sedans(false);
}

/// [`edge_contact_four_sedans_step_like_the_cpu`] on the split group path (the lab's).
#[test]
fn edge_contact_four_sedans_on_the_split_path_step_like_the_cpu() {
    four_sedans(true);
}

fn four_sedans(split: bool) {
    let text = sedan_text();
    let make = |at: Vec3, v: Vec3| {
        let mut s = ExplicitBeams::from_truck(&text, at, v, EXACT).unwrap();
        s.roll_tyres(v);
        s.two_body_contact = true;
        s.dissipative_contact = true;
        s.fixed_contact = true;
        s.anti_clip = true;
        s.edge_contact = true;
        s
    };
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let (zmin, zmax) = probe.nodes.iter().fold((f32::MAX, f32::MIN), |(a, b), n| (a.min(n.pos.z), b.max(n.pos.z)));
    let (xmin, xmax) = probe.nodes.iter().fold((f32::MAX, f32::MIN), |(a, b), n| (a.min(n.pos.x), b.max(n.pos.x)));
    let (len, wid) = (zmax - zmin, xmax - xmin);
    let y = 0.01 - low;
    let fleet = vec![
        make(Vec3::new(0.0, y, 0.0), Vec3::new(0.0, 0.0, 12.0)),
        make(Vec3::new(0.0, y, len + 0.4), Vec3::new(0.0, 0.0, -8.0)),
        make(Vec3::new(0.7 * wid, y, 0.3), Vec3::new(-1.0, 0.0, 10.0)),
        make(Vec3::new(0.7 * wid, y, len + 0.6), Vec3::new(1.0, 0.0, -10.0)),
    ];
    let Some(mut g) = gpu(&fleet) else {
        eprintln!("no GPU: skipped");
        return;
    };
    g.set_bit_exact(true);
    if split {
        g.set_group_split(true);
        g.set_group_split_tests(true);
    }
    let mut w = ExplicitBeamsWorld::new(fleet);
    let mut steps = 0;
    for f in 0..90 {
        w.frame(17);
        g.frame(17);
        steps += 17;
        if f % 10 == 9 {
            let pos = g.positions();
            for k in 0..4 {
                let d = rms(&pos[k], &w.vehicles[k].positions());
                assert_eq!(d, 0.0, "sedan {k}: GPU–CPU {d} m after {steps} steps (edge hits {:?})", w.edge_stats);
            }
        }
    }
    g.wait();
    g.contact_status().unwrap();
    eprintln!("edge stats {:?} after {steps} steps", w.edge_stats);
    assert!(w.edge_stats[5] > 100);
}

/// Contact offsets (`ExplicitBeams::contact_offset`: the triangles at their nodes pushed out
/// along the vertex normals) in the shipped kernels: two procedural sedans, 4 cm on every
/// node, one driving into the other's rear; GPU == CPU bit for bit, and the offsets acted.
#[test]
fn contact_offsets_step_like_the_cpu() {
    contact_offsets_scene(false);
}

/// [`contact_offsets_step_like_the_cpu`] on the split group path with the tests across
/// workgroups: the fleet-wide schedules (their kept candidate lists padded by the
/// largest offset, the offset vectors from each vehicle's step).
#[test]
fn contact_offsets_on_the_fleet_path_step_like_the_cpu() {
    contact_offsets_scene(true);
}

fn contact_offsets_scene(split: bool) {
    let text = sedan_text();
    let make = |at: Vec3, v: Vec3, off: f32| {
        let mut s = ExplicitBeams::from_truck(&text, at, v, EXACT).unwrap();
        s.roll_tyres(v);
        s.two_body_contact = true;
        s.dissipative_contact = true;
        s.fixed_contact = true;
        s.anti_clip = true;
        s.contact_offset = vec![off; s.nodes.len()];
        s
    };
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let (zmin, zmax) = probe.nodes.iter().fold((f32::MAX, f32::MIN), |(a, b), n| (a.min(n.pos.z), b.max(n.pos.z)));
    let fleet_with = |off: f32| {
        vec![
            make(Vec3::new(0.0, 0.01 - low, zmax - zmin + 0.4), Vec3::new(0.0, 0.0, -3.0), off),
            make(Vec3::new(0.0, 0.01 - low, 0.0), Vec3::new(0.0, 0.0, 12.0), off),
        ]
    };
    let fleet = fleet_with(0.04);
    let Some(mut g) = gpu(&fleet) else {
        eprintln!("no GPU: skipped");
        return;
    };
    g.set_bit_exact(true);
    if split {
        g.set_group_split(true);
        g.set_group_split_tests(true);
    }
    let mut w = ExplicitBeamsWorld::new(fleet);
    let mut wo = ExplicitBeamsWorld::new(fleet_with(0.0));
    if std::env::var("OFFSET_DEBUG").is_ok() {
        for step in 0..800 {
            let before: Vec<Vec<Vec3>> = w.vehicles.iter().map(|v| v.nodes.iter().map(|n| n.pos).collect()).collect();
            w.frame(1);
            g.frame(1);
            let pos = g.positions();
            let d: Vec<f32> = (0..2).map(|k| rms(&pos[k], &w.vehicles[k].positions())).collect();
            if d.iter().any(|&x| x != 0.0) {
                // The offset vectors at this state, GPU against a CPU replica.
                let pt = g.portable(0.0, "");
                let st: Vec<u32> = bytemuck::cast_slice(&pt.bindings[1]).to_vec();
                let bm: Vec<f32> = bytemuck::cast_slice(&pt.bindings[4]).to_vec();
                let words: Vec<u32> = bytemuck::cast_slice(&pt.bindings[0]).to_vec();
                let ob = words[107] as usize;
                let vecs = st[(ob - 1) * 4] as usize;
                let mut node0 = 0;
                for (k, v) in w.vehicles.iter().enumerate() {
                    let mut n = vec![Vec3::ZERO; v.nodes.len()];
                    for (t, tri) in v.triangles.iter().enumerate() {
                        if v.tri_dead.get(t).copied().unwrap_or(false) || v.tri_anti_only(t) || !v.tri_outer.get(t).copied().unwrap_or(true) {
                            continue;
                        }
                        let [a, b, c] = tri.nodes.map(|i| v.nodes[i as usize].pos);
                        let nn = (b - a).cross(c - a).normalize_or_zero();
                        for &i in &tri.nodes {
                            n[i as usize] += nn;
                        }
                    }
                    let mut worst = (0.0f32, 0usize);
                    for i in 0..v.nodes.len() {
                        let cpu = n[i].normalize_or_zero() * v.contact_offset[i];
                        let at = (vecs + node0 + i) * 4;
                        let gpu = Vec3::new(bm[at], bm[at + 1], bm[at + 2]);
                        let e = (cpu - gpu).abs().max_element();
                        if e > worst.0 {
                            worst = (e, i);
                        }
                    }
                    eprintln!("vehicle {k}: offset vectors differ by up to {:e} (node {}); origins cpu {:?} gpu {:?}", worst.0, worst.1, v.origin, g.origins()[k]);
                    let i = worst.1;
                    let at = (vecs + node0 + i) * 4;
                    let cnt = v.triangles.iter().enumerate().filter(|(t, tr)| tr.nodes.contains(&(i as u32)) && !v.tri_dead.get(*t).copied().unwrap_or(false)).count();
                    eprintln!("  cpu {:?} gpu {:?} sum {:?} ({cnt} tris)", n[i].normalize_or_zero() * v.contact_offset[i], [bm[at], bm[at + 1], bm[at + 2]], n[i]);
                    node0 += v.nodes.len();
                }
                panic!("first difference at step {step}: {d:?}, hits {} (first hit step {:?})", w.total.hits, w.first_hit_step);
            }
        }
    }
    for (frames, steps) in [(20, 17), (40, 1), (20, 17)] {
        for _ in 0..frames {
            w.frame(steps);
            wo.frame(steps);
            g.frame(steps);
        }
    }
    g.wait();
    g.contact_status().unwrap();
    let pos = g.positions();
    for k in 0..2 {
        let d = rms(&pos[k], &w.vehicles[k].positions());
        eprintln!("sedan {k}: GPU–CPU {d:.2e} m after {} steps ({} hits)", w.steps, w.total.hits);
        assert_eq!(d, 0.0, "sedan {k}: GPU–CPU {d} m");
    }
    let moved = (0..2).map(|k| rms(&wo.vehicles[k].positions(), &pos[k])).fold(0.0f32, f32::max);
    eprintln!("without offsets the CPU ends {moved:.4} m away");
    assert!(moved > 1e-4, "the offsets made no difference");
}

/// A browser host compiles the modules `compute_specs` lists and runs a pass only once its
/// module is compiled: a fleet with edge contact must list the edge pass's (it did not until
/// 2026-11-25, so pages never ran edge contact), one without must not.
#[test]
fn a_browser_host_compiles_the_edge_pass() {
    let fleet = |edge: bool| -> Vec<ExplicitBeams> {
        (0..2)
            .map(|k| {
                let x = 2.0 * k as f32;
                let mut s = tetra([Vec3::new(x, 0.0, 0.0), Vec3::new(x + 1.0, 0.0, 0.0), Vec3::new(x, 1.0, 0.0), Vec3::new(x, 0.0, 1.0)], Vec3::ZERO);
                s.edge_contact = edge;
                s
            })
            .collect()
    };
    let has = |edge: bool| ExplicitBeamsGpu::compute_specs(&fleet(edge), None).iter().any(|s| s.module.starts_with("explicit_beams_split_edges"));
    assert!(has(true), "the edge pass's module is not listed");
    assert!(!has(false), "the edge pass's module without edge contact");
}

/// The car the lab drives (the generated sedan, rigged from code, raycast wheels, surface contact
/// and contact offsets on, as `surface=1`): two of them head on at 60 km/h each with edge contact,
/// through the hit (0.4 s, the lab's 34-step frames), GPU == CPU bit for bit.
#[test]
fn edge_contact_generated_sedans_head_on_step_like_the_cpu() {
    use engine_softbody_authoring::authoring::rig;
    let built = rig::rig_vehicle("sedan", None, None).expect("rig the generated sedan");
    let files = rig::mod_files(&built);
    let txt: Vec<(String, String)> = files.iter().filter(|(p, _)| p.ends_with(".jbeam")).map(|(p, b)| (p.clone(), String::from_utf8_lossy(b).into_owned())).collect();
    let pc = files.iter().find(|(p, _)| p.ends_with(".pc")).map(|(_, b)| String::from_utf8_lossy(b).into_owned());
    let refs: Vec<(&str, &str)> = txt.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let d = engine_softbody::jbeam::load_jbeam(&refs, pc.as_deref()).expect("load");
    let settings = engine_softbody_authoring::vehicle_compare::settings();
    let mut turned = d.clone();
    turned.nodes.iter_mut().for_each(|n| n.position = [-n.position[0], n.position[1], -n.position[2]]);
    let ext = |d: &engine_softbody::desc::SoftVehicleDesc, k: usize| d.nodes.iter().map(|n| n.position[k]).fold((f32::MAX, f32::MIN), |(a, b), x| (a.min(x), b.max(x)));
    let make = |d: &engine_softbody::desc::SoftVehicleDesc, z: f32, v: f32| {
        let (mut s, _) = ExplicitBeams::from_desc_jbeam_raycast(d, Vec3::new(0.0, 0.02 - ext(d, 1).0, z), Vec3::Z * v, settings).expect("spawn");
        s.surface_contact = true;
        s.edge_contact = true;
        s.finish();
        s
    };
    let speed = 60.0 / 3.6;
    let fleet = vec![make(&d, -0.125 - ext(&d, 2).1, speed), make(&turned, 0.125 - ext(&turned, 2).0, -speed)];
    assert!(fleet.iter().all(|s| s.edge_contact && !s.contact_offset.is_empty()), "edge contact and offsets on");
    let Some(mut g) = gpu(&fleet) else {
        eprintln!("no GPU: skipped");
        return;
    };
    g.set_bit_exact(true);
    g.set_vehicle_contact(true);
    let mut w = ExplicitBeamsWorld::new(fleet);
    for f in 0..24 {
        w.frame(34);
        g.frame(34);
        let pos = g.positions();
        for k in 0..2 {
            let dd = rms(&pos[k], &w.vehicles[k].positions());
            assert_eq!(dd, 0.0, "sedan {k}: GPU–CPU {dd} m after frame {f} ({} steps); edge hits {:?}", w.steps, w.edge_stats);
        }
    }
    g.contact_status().unwrap();
    eprintln!("edge pass stats {:?}; node hits {}", w.edge_stats, w.total.hits);
    assert!(w.edge_stats[5] > 100, "the cars met: edge hits {:?}", w.edge_stats);
}
