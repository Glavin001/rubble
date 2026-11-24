//! The car's self-collision, triangle and beam census printouts (split from `vehicle_lab.rs`).

use super::*;

/// The self-collision walk's load for the car `s` with its nodes at `pos` and beams
/// `broken`: live collision triangles (none of the beams that kill it broken) by the
/// largest side of their box, how many cover more than the grid's 64 cells (they test
/// every candidate), and the (triangle, candidate) pairs inside the boxes (the kernel's
/// pair list holds 8,192; anti-clip's speed padding left out, so a lower bound).
pub(super) fn tri_census(s: &ExplicitBeams, pos: &[glam::Vec3], vel: &[glam::Vec3], broken: &[bool]) -> String {
    let (r, cell) = (s.collision_range, s.grid_cell);
    // Anti-clip pads every triangle's box by its own and the car's fastest node's travel.
    let vmax = vel.iter().map(|v| v.length()).fold(0.0f32, f32::max);
    let fast = vel.iter().filter(|v| v.length() > 100.0).count();
    let (mut padded_pairs, mut padded_every, mut own_pairs) = (0u64, 0u32, 0u64);
    let (mut live, mut dead, mut over1, mut over3, mut every_c, mut cells, mut pairs, mut longest) =
        (0u32, 0u32, 0u32, 0u32, 0u32, 0u64, 0u64, 0.0f32);
    let (mut never, mut big_pairs) = (0u32, 0u64);
    for (t, tri) in s.triangles.iter().enumerate() {
        let kills = s.tri_breaks.get(t).copied().unwrap_or([u32::MAX; 4]);
        if kills.iter().any(|&b| b != u32::MAX && broken.get(b as usize).copied().unwrap_or(false)) {
            dead += 1;
            continue;
        }
        live += 1;
        let p = tri.nodes.map(|i| pos[i as usize]);
        let lo = p[0].min(p[1]).min(p[2]) - glam::Vec3::splat(r);
        let hi = p[0].max(p[1]).max(p[2]) + glam::Vec3::splat(r);
        let ext = (hi - lo).max_element() - 2.0 * r;
        longest = longest.max(ext);
        over1 += (ext > 1.0) as u32;
        over3 += (ext > 3.0) as u32;
        never += (ext > 1.0 && kills.iter().all(|&x| x == u32::MAX)) as u32;
        let fs = ((hi / cell).floor() - (lo / cell).floor() + glam::Vec3::ONE).max(glam::Vec3::ONE);
        let c = fs.x * fs.y * fs.z;
        if c > 64.0 || fs.max_element() > 64.0 {
            every_c += 1;
        } else {
            cells += c as u64;
        }
        let n = s.candidates.iter().filter(|&&h| {
            let q = pos[h as usize];
            !(q.cmplt(lo).any() || q.cmpgt(hi).any())
        }).count() as u64;
        pairs += n;
        if ext > 1.0 {
            big_pairs += n;
        }
        if s.anti_clip {
            let tv = tri.nodes.iter().map(|&i| vel[i as usize].length()).fold(0.0f32, f32::max);
            let pad = glam::Vec3::splat((tv + vmax) * engine_softbody::explicit_beams::STEP + engine_softbody::explicit_contact::CLIP_DEPTH);
            let (lo, hi) = (lo - pad, hi + pad);
            let fs = ((hi / cell).floor() - (lo / cell).floor() + glam::Vec3::ONE).max(glam::Vec3::ONE);
            padded_every += (fs.x * fs.y * fs.z > 64.0 || fs.max_element() > 64.0) as u32;
            padded_pairs += s.candidates.iter().filter(|&&h| {
                let q = pos[h as usize];
                !(q.cmplt(lo).any() || q.cmpgt(hi).any())
            }).count() as u64;
            // Each candidate against the box padded by its own travel, not the fastest node's.
            let (lo0, hi0) = (lo + pad, hi - pad);
            own_pairs += s.candidates.iter().filter(|&&h| {
                let q = pos[h as usize];
                let pad = glam::Vec3::splat((tv + vel[h as usize].length()) * engine_softbody::explicit_beams::STEP + engine_softbody::explicit_contact::CLIP_DEPTH);
                !(q.cmplt(lo0 - pad).any() || q.cmpgt(hi0 + pad).any())
            }).count() as u64;
        }
    }
    format!(
        "anti-clip {}: fastest node {vmax:.0} m/s ({fast} over 100 m/s), padded boxes hold {padded_pairs} pairs ({own_pairs} with each node's own travel), {padded_every} past 64 cells; triangles live {live} dead {dead}; box side > 1 m: {over1}, > 3 m: {over3} (longest {longest:.1} m; {never} that no beam can kill); past 64 cells {every_c}; cells walked {cells}; pairs in boxes {pairs} ({big_pairs} from the > 1 m boxes); candidates {}; broken beams {}",
        s.anti_clip,
        s.candidates.len(),
        broken.iter().filter(|&&b| b).count()
    )
}

/// What the reused pair list's movement bound sees: the tracked nodes (contacters and live
/// collision triangles' nodes) split into the main body (the heaviest piece the unbroken
/// beams leave) and the rest; per group, the per-axis spread of their velocities (m/s) and
/// the steps a 0.05 m list survives at that spread (M / (spread · h)); the detached ones'
/// speeds and whether they carry live triangles.
pub(super) fn reuse_census(s: &ExplicitBeams, pos: &[glam::Vec3], vel: &[glam::Vec3], broken: &[bool]) -> String {
    let n = s.nodes.len().min(pos.len());
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for (j, b) in s.beams.iter().enumerate() {
        if !broken.get(j).copied().unwrap_or(false) {
            let (x, y) = (find(&mut parent, b.a as usize), find(&mut parent, b.b as usize));
            if x != y {
                parent[x] = y;
            }
        }
    }
    let mut mass = vec![0.0f32; n];
    for i in 0..n {
        let r = find(&mut parent, i);
        mass[r] += s.nodes[i].mass;
    }
    let main = (0..n).fold(0, |b, i| if mass[i] > mass[b] { i } else { b });
    let mut tracked = vec![false; n];
    for &c in &s.candidates {
        tracked[c as usize] = true;
    }
    let mut live_tri_detached = 0u32;
    for (t, tri) in s.triangles.iter().enumerate() {
        let kills = s.tri_breaks.get(t).copied().unwrap_or([u32::MAX; 4]);
        let dead = kills.iter().any(|&b| b != u32::MAX && broken.get(b as usize).copied().unwrap_or(false));
        for &v in &tri.nodes {
            tracked[v as usize] = true;
        }
        if !dead && tri.nodes.iter().any(|&v| find(&mut parent, v as usize) != main) {
            live_tri_detached += 1;
        }
    }
    let spread = |sel: &dyn Fn(usize) -> bool| -> (f32, u32) {
        let (mut lo, mut hi, mut k) = (glam::Vec3::MAX, glam::Vec3::MIN, 0u32);
        for i in (0..n).filter(|&i| tracked[i] && sel(i)) {
            lo = lo.min(vel[i]);
            hi = hi.max(vel[i]);
            k += 1;
        }
        (if k > 0 { (hi - lo).max_element() } else { 0.0 }, k)
    };
    let mut is_main = vec![false; n];
    for i in 0..n {
        is_main[i] = find(&mut parent, i) == main;
    }
    let (all, ka) = spread(&|_| true);
    let (m0, k0) = spread(&|i| is_main[i]);
    let det: Vec<f32> = (0..n).filter(|&i| tracked[i] && !is_main[i]).map(|i| vel[i].length()).collect();
    let pieces = {
        let mut r: Vec<usize> = (0..n).filter(|&i| !is_main[i]).map(|i| find(&mut parent, i)).collect();
        r.sort();
        r.dedup();
        r.len()
    };
    let steps = |sp: f32| if sp > 0.0 { 0.05 / (sp * STEP) } else { f32::INFINITY };
    // The pieces off the main body: (nodes, kg), heaviest first.
    let mut shed: Vec<(u32, f32)> = Vec::new();
    {
        let mut seen = vec![false; n];
        for i in 0..n {
            let r = find(&mut parent, i);
            if r != main && !seen[r] {
                seen[r] = true;
                let k = (0..n).filter(|&j| find(&mut parent, j) == r).count() as u32;
                shed.push((k, mass[r]));
            }
        }
        shed.sort_by(|a, b| b.1.total_cmp(&a.1));
    }
    let shed: Vec<String> = shed.iter().map(|(k, m)| format!("{k}n {m:.2}kg")).collect();
    // The pair list's volatile triangles under the lab's labels: live and dead.
    let (vol_live, vol_dead, vol_big) = {
        use engine_softbody::explicit_debris::{piece_labels, Pieces};
        let pieces = Pieces::of(s, broken);
        let labels = piece_labels(s, &pieces, broken);
        let (mut l, mut d, mut big) = (0u32, 0u32, 0u32);
        for (t, tri) in s.triangles.iter().enumerate() {
            let a = labels[tri.nodes[0] as usize];
            if !(a == 7 || tri.nodes.iter().any(|&v| labels[v as usize] != a)) {
                continue;
            }
            let kills = s.tri_breaks.get(t).copied().unwrap_or([u32::MAX; 4]);
            if kills.iter().any(|&b| b != u32::MAX && broken.get(b as usize).copied().unwrap_or(false)) {
                d += 1;
            } else {
                l += 1;
                let p = tri.nodes.map(|i| pos[i as usize]);
                big += ((p[0].max(p[1]).max(p[2]) - p[0].min(p[1]).min(p[2])).max_element() > 3.0) as u32;
            }
        }
        (l, d, big)
    };
    format!(
        "volatile triangles live {vol_live} ({vol_big} over 3 m) dead {vol_dead}; pieces off [{}]; tracked {ka} (main {k0}); detached nodes {} in {pieces} pieces ({} tracked, fastest {:.0} m/s, {} over 20 m/s), live triangles off the main body {live_tri_detached}; velocity spread all {all:.1} m/s ({:.0} steps a list), main only {m0:.1} m/s ({:.0} steps)",
        shed.join(", "),
        (0..n).filter(|&i| !is_main[i]).count(),
        det.len(),
        det.iter().fold(0.0f32, |a, &b| a.max(b)),
        det.iter().filter(|&&v| v > 20.0).count(),
        steps(all),
        steps(m0)
    )
}

/// What spreads the main body's (label 0) tracked nodes' displacements, the pair list's
/// reuse bound: their velocity spread (the bound's rate), the same after removing the best
/// rigid motion (translation and rotation: what rotation alone costs), the nodes far off
/// the rigid motion, and the spread among nodes within 1 m of each other (what a local
/// bound would see).
pub(super) fn reuse_cause(s: &ExplicitBeams, pos: &[glam::Vec3], vel: &[glam::Vec3], broken: &[bool]) -> String {
    use engine_softbody::explicit_debris::{piece_labels, Pieces};
    let n = s.nodes.len().min(pos.len());
    let pieces = Pieces::of(s, broken);
    let labels = piece_labels(s, &pieces, broken);
    let mut tracked = vec![false; n];
    for &c in &s.candidates {
        tracked[c as usize] = true;
    }
    for tri in &s.triangles {
        for &v in &tri.nodes {
            tracked[v as usize] = true;
        }
    }
    let m: Vec<usize> = (0..n).filter(|&i| tracked[i] && labels[i] == 0).collect();
    if m.is_empty() {
        return "no main nodes".into();
    }
    let k = m.len() as f32;
    let c = m.iter().map(|&i| pos[i]).sum::<glam::Vec3>() / k;
    let v0 = m.iter().map(|&i| vel[i]).sum::<glam::Vec3>() / k;
    let mut inertia = glam::Mat3::ZERO;
    let mut l = glam::Vec3::ZERO;
    for &i in &m {
        let r = pos[i] - c;
        inertia += glam::Mat3::from_diagonal(glam::Vec3::splat(r.length_squared()))
            - glam::Mat3::from_cols(r * r.x, r * r.y, r * r.z);
        l += r.cross(vel[i] - v0);
    }
    let w = if inertia.determinant().abs() > 1e-6 { inertia.inverse() * l } else { glam::Vec3::ZERO };
    let spread = |f: &dyn Fn(usize) -> glam::Vec3| {
        let (mut lo, mut hi) = (glam::Vec3::MAX, glam::Vec3::MIN);
        for &i in &m {
            lo = lo.min(f(i));
            hi = hi.max(f(i));
        }
        (hi - lo).max_element()
    };
    let all = spread(&|i| vel[i]);
    let rot = spread(&|i| w.cross(pos[i] - c));
    let res = |i: usize| vel[i] - v0 - w.cross(pos[i] - c);
    let resid = spread(&res);
    let mut far: Vec<f32> = m.iter().map(|&i| res(i).abs().max_element()).collect();
    far.sort_by(|a, b| b.total_cmp(a));
    let over = |x: f32| far.iter().filter(|&&f| f > x).count();
    // Without the nodes more than 2 m/s off the rigid motion: the rest's spread.
    let (mut lo, mut hi) = (glam::Vec3::MAX, glam::Vec3::MIN);
    for &i in &m {
        if res(i).abs().max_element() <= 2.0 {
            lo = lo.min(vel[i]);
            hi = hi.max(vel[i]);
        }
    }
    let calm = (hi - lo).max_element();
    let mut local = 0.0f32;
    for (a, &i) in m.iter().enumerate() {
        for &j in &m[a + 1..] {
            if (pos[i] - pos[j]).abs().max_element() < 1.0 {
                local = local.max((vel[i] - vel[j]).abs().max_element());
            }
        }
    }
    let steps = |sp: f32| if sp > 0.0 { 0.1 / (sp * STEP) } else { f32::INFINITY };
    format!(
        "main tracked {}: spread {all:.2} m/s ({:.0} steps at 0.1 m), rotation {:.2} rad/s gives {rot:.2}, off the rigid motion {resid:.2} (nodes over 0.5 / 2 / 10 m/s: {} / {} / {}; top {:.1} {:.1} {:.1}), spread without those over 2 m/s {calm:.2} ({:.0} steps), within 1 m {local:.2} ({:.0} steps)",
        m.len(),
        steps(all),
        w.length(),
        over(0.5),
        over(2.0),
        over(10.0),
        far.first().copied().unwrap_or(0.0),
        far.get(1).copied().unwrap_or(0.0),
        far.get(2).copied().unwrap_or(0.0),
        steps(calm),
        steps(local)
    )
}

/// The page's mountain (`vehicle-lab.html?mode=mountain&…`) for its cost: per second of
/// simulated time, the wall time of a World step (each step waited on: its GPU time and
/// the readback), with the car's speed, node box, coupling use and the surface pair
/// counts. `--steepness 55 --roughness 0.25 --launch 12 --rockslide 1 --surface 1` is
/// the owner's report. Measurement switches (one at a time): `--ablate MASK` from
/// `--ablate-at T` (results meaningless), `--surface-pairs 0`, `--surface-culls 0`.
/// The beam phase's shape on the GPU (`--measure beam-census`, CPU only): beams by kind
/// code, extension records and damping filters, and per 32-beam SIMD group (lane j takes
/// beam j, j + LANES, …) the distinct code paths it runs; the nodes' incidence counts
/// (a node lane's loads in its beam sum).
pub(super) fn beam_census(template: &ExplicitBeams) {
    let path = |j: usize| {
        let b = &template.beams[j];
        let ext = template.ext_of.get(j).is_some_and(|&x| x != 0);
        (b.kind.code(), ext, b.shock != 0 && b.kind.is_jbeam())
    };
    let mut kinds = std::collections::BTreeMap::<(u32, bool, bool), usize>::new();
    for j in 0..template.beams.len() {
        *kinds.entry(path(j)).or_default() += 1;
    }
    println!("beams {} by (kind code, extension, filter): {kinds:?}", template.beams.len());
    let groups = template.beams.len().div_ceil(32);
    let (mut paths, mut mixed) = (0usize, 0usize);
    for g in 0..groups {
        let mut s: Vec<_> = (g * 32..((g + 1) * 32).min(template.beams.len())).map(path).collect();
        s.sort();
        s.dedup();
        paths += s.len();
        mixed += usize::from(s.len() > 1);
    }
    println!(
        "SIMD groups of 32 beams: {groups}, {:.2} distinct paths each on average, {mixed} mixed",
        paths as f64 / groups as f64
    );
    let mut inc: Vec<usize> = template.incidence.iter().map(Vec::len).collect();
    let total: usize = inc.iter().sum();
    inc.sort_unstable();
    println!(
        "nodes {}: incidence entries {total}, mean {:.1}, median {}, p90 {}, max {}",
        inc.len(),
        total as f64 / inc.len() as f64,
        inc[inc.len() / 2],
        inc[inc.len() * 9 / 10],
        inc[inc.len() - 1]
    );
}
