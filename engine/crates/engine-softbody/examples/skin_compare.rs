//! The showcase sedan's skin under four bindings on the SAME node
//! trajectories (CPU only; engine/docs/measurements/skin-shards.md, "Four
//! bindings"): the triangle binding, node frames with the earlier stretch /
//! fold cuts, node frames cut only where beams broke (the default), and
//! BeamNG-style per-vertex frames. Each scene runs once; at each checkpoint
//! every binding is measured with the definitions of
//! `tests/showcase_skin_integrity.rs` (shards, stretch, flipped, saw-tooth,
//! coverage and cuts no broken beam explains), plus the binding lint.
//!
//! `cargo run --release -p engine-softbody --example skin_compare -- [scene…]`
//! scenes: wall70 wall90x3 side60 grab (default: all). `PARTS=1`: per part
//! shards, flips and coverage for the default binding. `LINT=1`: the lint.
use engine_softbody::authoring::node_frames::{bind_beamng, eval_full, Binding};
use engine_softbody::authoring::skin::{
    deform_beams, raycast_visual, showcase_visual_lod, showcase_visual_triangles_lod, unit_states,
    vertex_tears, BeamState, ShowcaseVisual,
};
use engine_softbody::authoring::skin_lint::lint;
use engine_softbody::authoring::{build, style, BuildParams, Built, Resolution};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, Grab, StaticBox, STEP};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use glam::{Quat, Vec3};
use std::collections::{BTreeMap, HashMap, HashSet};

fn spawn(b: &Built, at: Vec3, yaw: f32, vel: Vec3) -> ExplicitBeams {
    let st = ExplicitBeamsSettings::default();
    let (p, _) = ExplicitBeams::from_truck_raycast(&b.truck, None, Vec3::ZERO, yaw, Vec3::ZERO, st.clone()).unwrap();
    let d = p.drive.as_ref().unwrap();
    let low = d
        .config
        .wheels
        .iter()
        .map(|w| 0.5 * (p.nodes[w.node1 as usize].pos.y + p.nodes[w.node2 as usize].pos.y) - w.radius)
        .fold(f32::MAX, f32::min);
    let (lo, hi) = p.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, c), n| (a.min(n.pos), c.max(n.pos)));
    let mid = 0.5 * (lo + hi);
    let pos = Vec3::new(at.x - mid.x, 0.02 - low, at.z - mid.z);
    let (mut s, _) = ExplicitBeams::from_truck_raycast(&b.truck, None, pos, yaw, vel, st).unwrap();
    let sp = vel.length();
    if let Some(d) = s.drive.as_mut() {
        let r: Vec<f32> = d.config.wheels.iter().map(|w| w.radius).collect();
        for (w, r) in d.wheels.iter_mut().zip(r) {
            w.omega = sp / r;
        }
    }
    s
}

fn components(s: &ExplicitBeams) -> Vec<u32> {
    let mut p: Vec<u32> = (0..s.nodes.len() as u32).collect();
    fn find(p: &mut [u32], mut i: u32) -> u32 {
        while p[i as usize] != i {
            p[i as usize] = p[p[i as usize] as usize];
            i = p[i as usize];
        }
        i
    }
    for b in s.beams.iter().filter(|b| !b.broken) {
        let (x, y) = (find(&mut p, b.a), find(&mut p, b.b));
        if x != y {
            p[x as usize] = y;
        }
    }
    (0..s.nodes.len() as u32).map(|i| find(&mut p, i)).collect()
}

fn drivers(v: &ShowcaseVisual, i: usize, used: Option<u8>) -> Vec<u32> {
    if let Some(ids) = v.bl_node.get(i).filter(|b| b[0] != u32::MAX) {
        return ids
            .iter()
            .enumerate()
            .filter(|&(k, &j)| j != u32::MAX && used.map_or(true, |u| (u >> k) & 1 == 1))
            .map(|(_, &j)| j)
            .collect();
    }
    if v.rigid[i] >= 2 {
        return Vec::new();
    }
    v.bind_nodes[i].to_vec()
}

fn drawn(len: f32, ta: f32, tb: f32) -> f32 {
    match (ta < 0.5, tb < 0.5) {
        (true, true) => len,
        (false, false) => 0.0,
        _ => {
            let (lo, hi) = (ta.min(tb), ta.max(tb));
            len * (0.5 - lo) / (hi - lo).max(1e-9)
        }
    }
}

#[derive(Default, Clone)]
struct M {
    tris: usize,
    uncut: usize,
    shards: usize,
    longest: f32,
    p99: f32,
    max: f32,
    flipped: usize,
    jumpy: usize,
    spans: usize,
    cut_unexplained: usize,
    lattice_flip: f32,
    /// Per part: [rest tris, uncut, shards, flipped, cut unexplained].
    parts: BTreeMap<String, [usize; 5]>,
}

fn measure(b: &Built, v: &ShowcaseVisual, s: &ExplicitBeams) -> M {
    let nodes = s.positions();
    let bs: Vec<BeamState> = s.beams.iter().map(|b| BeamState { broken: b.broken, rest: b.rest }).collect();
    let us = unit_states(v, &bs);
    let (pos, nrm) = deform_beams(v, &nodes, Some(&us), Some(&bs));
    let tear = vertex_tears(v, &nodes, Some(&bs), &pos);
    let sk = eval_full(v, &nodes, Some(&bs));
    let comp = components(s);
    let broken: HashSet<(u32, u32)> =
        s.beams.iter().filter(|b| b.broken).map(|b| (b.a.min(b.b), b.a.max(b.b))).collect();
    let at_break = |i: usize| {
        let d = drivers(v, i, None);
        d.iter().any(|&a| d.iter().any(|&c| a < c && broken.contains(&(a, c))))
    };
    let shattered = |i: usize| {
        let u = v.unit.get(i).copied().unwrap_or(u32::MAX);
        u != u32::MAX && us[u as usize].broken
    };
    let part_name = |i: usize| match v.part.get(i).copied().unwrap_or(u16::MAX) {
        u16::MAX => "structure".to_string(),
        p => b.skeleton.parts[p as usize].clone(),
    };
    let rp = |i: u32| Vec3::from_array(v.positions[i as usize]);
    let n = v.positions.len();
    let mut m = M::default();
    let (mut nb, mut cnt, mut edge, mut ring) = (vec![Vec3::ZERO; n], vec![0u32; n], vec![0f32; n], vec![0u32; n]);
    for t in v.indices.chunks_exact(3) {
        for &i in t {
            ring[i as usize] += 2;
        }
    }
    let mut st = Vec::new();
    for t in v.indices.chunks_exact(3) {
        let ids = [t[0] as usize, t[1] as usize, t[2] as usize];
        let name = part_name(ids[0]);
        m.tris += 1;
        let p = ids.map(|i| pos[i]);
        let tv = ids.map(|i| tear[i]);
        let cut = p.iter().any(|q| !q.x.is_finite()) || tv.iter().any(|&x| x >= 0.5);
        let ok = !cut || ids.iter().any(|&i| shattered(i) || at_break(i));
        let e = m.parts.entry(name).or_default();
        e[0] += 1;
        e[1] += !cut as usize;
        e[4] += !ok as usize;
        m.uncut += !cut as usize;
        m.cut_unexplained += !ok as usize;
        if p.iter().any(|q| !q.x.is_finite()) {
            continue;
        }
        for (a, c) in [(0, 1), (1, 2), (2, 0)] {
            if tv[a] < 0.5 && tv[c] < 0.5 {
                nb[ids[a]] += p[c] - rp(t[c]);
                nb[ids[c]] += p[a] - rp(t[a]);
                cnt[ids[a]] += 1;
                cnt[ids[c]] += 1;
                let l0 = rp(t[a]).distance(rp(t[c]));
                edge[ids[a]] += l0;
                edge[ids[c]] += l0;
            }
        }
        if tv.iter().all(|&x| x >= 0.5) {
            continue;
        }
        let (mut shard, mut worst) = (false, 0.0f32);
        for (a, c) in [(0, 1), (1, 2), (2, 0)] {
            let l0 = rp(t[a]).distance(rp(t[c]));
            let l = drawn(p[a].distance(p[c]), tv[a], tv[c]);
            worst = worst.max(l / l0.max(1e-4));
            if l > 3.0 * l0 && l > l0 + 0.05 {
                shard = true;
                m.longest = m.longest.max(l);
            }
        }
        st.push(worst);
        m.shards += shard as usize;
        e[2] += shard as usize;
        let mut c: Vec<u32> = ids
            .iter()
            .flat_map(|&i| drivers(v, i, sk[i].map(|x| x.used)))
            .map(|j| comp[j as usize])
            .collect();
        c.sort_unstable();
        c.dedup();
        m.spans += (c.len() > 1) as usize;
        let r0 = ids.map(|i| rp(i as u32));
        let f0 = (r0[1] - r0[0]).cross(r0[2] - r0[0]);
        let f = (p[1] - p[0]).cross(p[2] - p[0]);
        let n0: Vec3 = ids.iter().map(|&i| Vec3::from_array(v.normals[i])).sum();
        let n1: Vec3 = ids.iter().map(|&i| nrm[i]).sum();
        if f0.dot(n0) > 0.0 && f.length_squared() > 1e-18 && f.dot(n1) < 0.0 {
            m.flipped += 1;
            e[3] += 1;
        }
    }
    for i in 0..n {
        if ring[i] < 6 || cnt[i] < ring[i] || tear[i] >= 0.5 || !pos[i].x.is_finite() {
            continue;
        }
        let d = pos[i] - rp(i as u32);
        let mean = nb[i] / cnt[i] as f32;
        let e = (edge[i] / cnt[i] as f32).max(0.005);
        m.jumpy += ((d - mean).length() > 3.0 * e) as usize;
    }
    st.sort_by(f32::total_cmp);
    if !st.is_empty() {
        m.max = *st.last().unwrap();
        m.p99 = st[(st.len() as f32 * 0.99) as usize];
    }
    // The skeleton's own skin folded over (the integrity test's measure).
    let rest = |i: u32| Vec3::from_array(v.node_rest[i as usize]);
    let (mut flip, mut all) = (0usize, 0usize);
    for t in &v.normal_tris {
        let f0 = (rest(t[1]) - rest(t[0])).cross(rest(t[2]) - rest(t[0]));
        let q = t.map(|k| nodes[k as usize]);
        let f = (q[1] - q[0]).cross(q[2] - q[0]);
        all += 1;
        let mut mm = Vec3::ZERO;
        for &k in t {
            let k = k as usize;
            for &tj in &v.node_tri_list[v.node_tri_start[k] as usize..v.node_tri_start[k + 1] as usize] {
                let u = v.normal_tris[tj as usize];
                let qu = u.map(|x| nodes[x as usize]);
                let ru = u.map(rest);
                mm += (qu[1] - qu[0]).cross(qu[2] - qu[0]).normalize_or_zero()
                    * (ru[1] - ru[0]).cross(ru[2] - ru[0]).dot(f0).signum();
            }
        }
        flip += (f.dot(mm) < 0.0) as usize;
    }
    m.lattice_flip = flip as f32 / all.max(1) as f32;
    m
}

const NAMES: [&str; 4] = ["triangles", "frames+cuts", "frames, breaks", "beamng"];

fn report(tag: &str, b: &Built, vs: &[ShowcaseVisual], s: &ExplicitBeams) {
    let nb = s.beams.iter().filter(|x| x.broken).count();
    println!("{tag} ({nb} beams broken)");
    for (k, v) in vs.iter().enumerate() {
        let m = measure(b, v, s);
        println!(
            "  | {:<14} | {:>6} | {:.2} m | {:.2} / {:.0} | {:>5} ({:.2} %; lattice {:.2} %) | {:>4} | {:>4} | {:.1} % | {:>5} |",
            NAMES[k],
            m.shards,
            m.longest,
            m.p99,
            m.max,
            m.flipped,
            100.0 * m.flipped as f32 / m.uncut.max(1) as f32,
            100.0 * m.lattice_flip,
            m.jumpy,
            m.spans,
            100.0 * m.uncut as f32 / m.tris as f32,
            m.cut_unexplained,
        );
        if std::env::var("PARTS").is_ok() && k == 2 {
            for (name, [all, uncut, sh, fl, bad]) in &m.parts {
                if uncut < all || *sh > 0 || *fl > 0 {
                    println!(
                        "      {name}: {uncut}/{all} uncut ({:.1} %), {sh} shards, {fl} flipped, {bad} cut without a break",
                        100.0 * *uncut as f32 / *all as f32
                    );
                }
            }
        }
    }
}

fn wall(b: &Built, vs: &[ShowcaseVisual], kmh: f32, hits: u32, tag: &str) {
    let vel = Vec3::new(0.0, 0.0, kmh / 3.6);
    let mut s = spawn(b, Vec3::ZERO, 0.0, vel);
    let reach = |s: &ExplicitBeams| s.positions().iter().map(|q| q.z).fold(f32::MIN, f32::max);
    let face = reach(&s) + 0.6;
    s.boxes.push(StaticBox::centered(Vec3::new(0.0, 1.5, face + 0.3), Vec3::new(8.0, 1.5, 0.3)));
    for hit in 0..hits {
        if hit > 0 {
            let back = Vec3::Z * (reach(&s) - (face - 1.0));
            for n in &mut s.nodes {
                n.pos -= back;
                n.vel = vel;
            }
        }
        for f in 0..90 {
            s.frame();
            if f == 20 && hit == 0 {
                report(&format!("{tag} hit {} t 0.33 s", hit + 1), b, vs, &s);
            }
        }
        report(&format!("{tag} hit {} t 1.5 s", hit + 1), b, vs, &s);
    }
}

fn side(b: &Built, vs: &[ShowcaseVisual]) {
    let struck = spawn(b, Vec3::ZERO, 0.0, Vec3::ZERO);
    let p0 = struck.positions();
    let rear = b.skeleton.parts.iter().position(|p| p == "door_r_l").unwrap() as u16;
    let door: Vec<Vec3> = (0..b.skeleton.nodes.len()).filter(|&i| b.skeleton.nodes[i].part == rear).map(|i| p0[i]).collect();
    let zd = door.iter().map(|p| p.z).sum::<f32>() / door.len() as f32;
    let hw = p0.iter().map(|p| p.x).fold(f32::MIN, f32::max);
    let striker = spawn(b, Vec3::new(hw + 0.5 * b.def.profile.length + 0.3, 0.0, zd), -90.0, Vec3::new(-60.0 / 3.6, 0.0, 0.0));
    let mut w = ExplicitBeamsWorld::new(vec![struck, striker]);
    for f in 0..120u32 {
        w.frame_60();
        if f == 15 || f == 119 {
            report(&format!("side 60 struck f {f}"), b, vs, &w.vehicles[0]);
            report(&format!("side 60 striker f {f}"), b, vs, &w.vehicles[1]);
        }
    }
}

fn grab(b: &Built, vs: &[ShowcaseVisual], part: &str, dist: f32) {
    let k = 5.0e4;
    let mut s = spawn(b, Vec3::ZERO, 0.0, Vec3::ZERO);
    s.run(0.5);
    let pi = b.skeleton.parts.iter().position(|p| p == part).unwrap() as u16;
    let p = s.positions();
    let c = p.iter().copied().sum::<Vec3>() / p.len() as f32;
    let node = (0..b.skeleton.nodes.len())
        .filter(|&i| b.skeleton.nodes[i].part == pi)
        .max_by(|&x, &y| (p[x] - c).length().total_cmp(&(p[y] - c).length()))
        .unwrap();
    let dir = ((p[node] - c) * Vec3::new(1.0, 0.3, 1.0)).normalize();
    let m = s.nodes[node].mass;
    let cd = (2.0 * (k * m).sqrt()).min(0.5 * m / STEP);
    for f in 0..150u32 {
        s.grab = (f < 90).then(|| Grab { node: node as u32, target: p[node] + dir * dist * (f as f32 / 30.0).min(1.0), k, c: cd });
        s.frame();
        if f == 89 {
            report(&format!("grab {part} {dist} m held"), b, vs, &s);
        }
    }
    report(&format!("grab {part} {dist} m released"), b, vs, &s);
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let all = a.is_empty();
    let on = |s: &str| all || a.iter().any(|x| x == s);
    let b = build(&style("sedan").unwrap(), &BuildParams { resolution: Resolution::Medium, ..Default::default() }).unwrap();
    let mut tri = showcase_visual_triangles_lod(&b, 1.0).unwrap();
    raycast_visual(&mut tri, &b.truck).unwrap();
    let mut breaks = showcase_visual_lod(&b, 1.0).unwrap();
    raycast_visual(&mut breaks, &b.truck).unwrap();
    let mut stretch = breaks.clone();
    stretch.frames.mode = Binding::Stretch;
    let mut beamng = breaks.clone();
    bind_beamng(&mut beamng).unwrap();
    let vs = [tri, stretch, breaks, beamng];
    if std::env::var("LINT").is_ok() {
        for (k, v) in vs.iter().enumerate() {
            let r = lint(v, &b, 7);
            println!(
                "lint {}: A1 {} A3 max {:.3} over {} B3 max {:.2} p99 {:.2} A6 {} A7 {:?} B1 {} / {:.1e} m B4 {} A4 max {:.1} over {} | B2 affine {:.1e} m bend {:.1e} m | A2 angle min {:.0}° p1 {:.0}° det min {:.3}",
                NAMES[k], r.cross_group, r.weight_norm_max, r.weight_norm_over, r.gain_max, r.gain_p99, r.incoherent_edges,
                r.bad_groups, r.leaks, r.rigid_err, r.bridges, r.reach_max, r.reach_over, r.affine_err, r.bend_err,
                r.frame_angle_min, r.frame_angle_p1, r.frame_det_min
            );
        }
    }
    println!("  | binding | shards | longest | stretch p99 / max | flipped (share; skeleton's) | saw-tooth | spans | uncut | cut w/o break |");
    if on("wall70") {
        wall(&b, &vs, 70.0, 1, "wall 70");
    }
    if on("wall90x3") {
        wall(&b, &vs, 90.0, 3, "wall 90 x3");
    }
    if on("side60") {
        side(&b, &vs);
    }
    if on("grab") {
        for (part, d) in [("door_f_l", 10.0), ("fender_l", 2.0), ("bumper_f", 2.0), ("bootlid", 2.0), ("bonnet", 2.0)] {
            grab(&b, &vs, part, d);
        }
    }
    let _ = (Quat::IDENTITY, HashMap::<u8, u8>::new());
}
