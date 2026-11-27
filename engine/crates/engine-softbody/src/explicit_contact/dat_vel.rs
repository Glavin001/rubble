//! Velocity contact's pass (`Dat::velocity`; `ccd_vel`): the response over the listed pairs
//! and the spheres the vehicles' rigid proxies carry, the listing grown when the response
//! moved nodes past its reach.

use super::{list_pairs, Dat, DatStats, Rec, Surface, STEP};
use crate::explicit_beams::ExplicitBeams;
use glam::Vec3;

/// The response (positions `x1` and velocities `vs` moved): the CCD's `pairs`, the rigid
/// proxies' spheres as bodies of their own, and [`relist_velocity`].
#[allow(clippy::too_many_arguments)]
pub(super) fn respond(vehicles: &mut [ExplicitBeams], dat: &mut Dat, recs: &mut Vec<Rec>, own_reach: &mut [f32], surf: &[Surface], off: &[Vec<Vec3>], x0: &[Vec<Vec3>], x1: &mut Vec<Vec<Vec3>>, vs: &mut Vec<Vec<Vec3>>, m: &[Vec<f32>], im: &[Vec<f32>], vpre: &[Vec<Vec3>], reach_v: &[f32], pairs: &[super::super::ccd::CcdPair], st: &mut DatStats) {
    // The spheres the vehicles' rigid proxies carry are bodies of the solve too (a
    // fast rigid body into a car): one node each after the vehicles', its centre
    // against the vehicle's collision triangles with the gap grown by its radius.
    let balls = sphere_bodies(vehicles);
    let nv = x0.len();
    let (mut x0e, mut x1e, mut vse, mut me, mut ime, mut vpe) = (x0.to_vec(), x1.clone(), vs.clone(), m.to_vec(), im.to_vec(), vpre.to_vec());
    for b in &balls {
        x0e.push(vec![b.c]);
        x1e.push(vec![b.c + b.v * STEP]);
        vse.push(vec![b.v]);
        me.push(vec![b.m]);
        ime.push(vec![1.0 / b.m]);
        if !vpe.is_empty() {
            vpe.push(vec![b.v]);
        }
    }
    let extra = sphere_pairs(&balls, surf, &x0e, &x1e, nv, dat.min_gap.max(0.002));
    let all: Vec<super::super::ccd::CcdPair> = pairs.iter().cloned().chain(extra.iter().cloned()).collect();
    // The kicked velocities: the impact law's guard's reference (`ccd::resolve`).
    dat.warm.vf = vse.clone();
    super::super::ccd::resolve(&all, &x0e, &mut x1e, &mut vse, &me, &ime, STEP, dat.ccd_rounds, dat.soft_zones, dat.zone_rounds, &mut st.ccd, &vpe, dat.velocity_restitution, Some(&mut dat.warm));
    relist_velocity(dat, recs, own_reach, surf, off, &x0e, &mut x1e, &mut vse, &me, &ime, &vpe, reach_v, all.len(), &extra, st);
    if std::env::var("DAT_MISS").is_ok() {
        miss_dbg(surf, &x0e, &x1e, &all, recs);
    }
    for (i, b) in balls.iter().enumerate() {
        if let Some(r) = vehicles[b.veh].rigid.as_mut() {
            r.bodies[b.body].vel = vse[nv + i][0];
        }
    }
    x1e.truncate(nv);
    vse.truncate(nv);
    *x1 = x1e;
    *vs = vse;
}

/// A sphere a vehicle's rigid proxies carry, as a body of velocity contact.
struct Ball {
    veh: usize,
    body: usize,
    r: f32,
    c: Vec3,
    v: Vec3,
    m: f32,
}

fn sphere_bodies(vehicles: &[ExplicitBeams]) -> Vec<Ball> {
    let mut out = Vec::new();
    for (veh, v) in vehicles.iter().enumerate() {
        if let Some(rs) = v.rigid.as_ref() {
            for (_, body, r) in rs.row_spheres() {
                let b = &rs.bodies[body];
                out.push(Ball { veh, body, r, c: b.pos, v: b.vel, m: 1.0 / (b.inv_mass * b.share.max(1.0)) });
            }
        }
    }
    out
}

/// Each sphere's centre (node `nv + i`) against its vehicle's collision triangles near its
/// path: a point-against-triangle pair whose gap is the radius and `gap`.
fn sphere_pairs(balls: &[Ball], surf: &[Surface], x0: &[Vec<Vec3>], x1: &[Vec<Vec3>], nv: usize, gap: f32) -> Vec<super::super::ccd::CcdPair> {
    let mut out = Vec::new();
    for (i, b) in balls.iter().enumerate() {
        let (c0, c1) = (x0[nv + i][0], x1[nv + i][0]);
        let v = b.veh;
        for t in &surf[v].tris {
            let p0 = t.map(|k| x0[v][k as usize]);
            let reach = (c1 - c0).length() + t.iter().map(|&k| (x1[v][k as usize] - x0[v][k as usize]).length()).fold(0.0f32, f32::max);
            let near = super::super::closest_on_triangle(c0, p0[0], p0[1], p0[2]).distance(c0);
            if near < b.r + gap + 2.0 * reach + 0.02 {
                out.push(super::super::ccd::CcdPair { k: [(nv + i, 0), (v, t[0]), (v, t[1]), (v, t[2])], nn: 1, xi: b.r + gap });
            }
        }
    }
    out
}

/// The CCD's pairs among the records: within twice the two sides' reach `q` this step.
/// Each pair's gap is `share` of its thickness `r.d` (its nodes' paint offsets along its normal,
/// at least `min_gap`): `xi_share` (half) where the speculative pass holds the rest, the whole of
/// it for velocity contact, which has none (with half, the paint stood 107 mm into the other car
/// at head 100; with all of it 0.8 mm, contact at the drawn surface the authored offsets give).
pub(super) fn ccd_pairs(recs: &[Rec], q: &[Vec<f32>], min_gap: f32, penalty: f32, share: f32) -> Vec<super::super::ccd::CcdPair> {
    let side = |r: &Rec, neg: bool| {
        let js = if neg { 0..r.nn } else { r.nn..r.len };
        js.map(|j| q[r.k[j].0][r.k[j].1 as usize]).fold(0.0f32, f32::max)
    };
    recs.iter()
        .filter(|r| r.len == 4 && r.g - r.d < 2.0 * (side(r, true) + side(r, false)))
        .map(|r| super::super::ccd::CcdPair { k: r.k, nn: r.nn, xi: if penalty > 0.0 { min_gap } else { (share * r.d).max(min_gap) } })
        .collect()
}

/// Velocity contact moves nodes by impulses, which can carry a node farther than the reach its
/// pairs and records were listed for (a bounce, a push from its neighbours' rows; and the records
/// are listed relative to each vehicle's first node, which the response may move too): it could
/// then meet a pair not listed. Each vehicle's reach the response grew lists the records again,
/// the pairs the final reach adds are listed, and the response runs again from here, until the
/// listing covers the motion (at most four times).
#[allow(clippy::too_many_arguments)]
fn relist_velocity(dat: &mut Dat, recs: &mut Vec<Rec>, own_reach: &mut [f32], surf: &[Surface], off: &[Vec<Vec3>], x0: &[Vec<Vec3>], x1: &mut Vec<Vec<Vec3>>, vs: &mut Vec<Vec<Vec3>>, m: &[Vec<f32>], im: &[Vec<f32>], vpre: &[Vec<Vec3>], reach_v: &[f32], listed: usize, extra: &[super::super::ccd::CcdPair], st: &mut DatStats) {
    let mut reach_l = reach_v.to_vec();
    let mut n = listed;
    for _ in 0..4 {
        let fin: Vec<f32> = surf.iter().enumerate().map(|(v, s)| {
            let r0 = x1[v][0] - x0[v][0];
            s.nodes.iter().map(|&i| ((x1[v][i as usize] - x0[v][i as usize]) - r0).length()).fold(0.0f32, f32::max)
        }).collect();
        let mut grew = false;
        for (l, f) in reach_l.iter_mut().zip(&fin) {
            if *f > *l {
                *l = *f;
                grew = true;
            }
        }
        if grew {
            let refs_l: Vec<(Vec3, Vec3)> = (0..surf.len()).map(|v| (x1[v][0] - x0[v][0], vs[v][0])).collect();
            let mut own_l = vec![0.0f32; surf.len()];
            *recs = list_pairs(dat, &x0[..surf.len()], surf, off, &reach_l, &refs_l, &mut own_l, st);
            for (o, l) in own_reach.iter_mut().zip(own_l) {
                *o = o.max(l);
            }
        }
        let q: Vec<Vec<f32>> = x1.iter().zip(x0).map(|(a, b)| a.iter().zip(b).map(|(p, o)| (*p - *o).length()).collect()).collect();
        let mut more = ccd_pairs(recs, &q, dat.min_gap, dat.penalty, 1.0);
        more.extend(extra.iter().cloned());
        if !grew && more.len() <= n {
            break;
        }
        n = more.len();
        st.relisted += 1;
        let z0 = st.ccd.zones;
        super::super::ccd::resolve(&more, x0, x1, vs, m, im, STEP, dat.ccd_rounds, dat.soft_zones, dat.zone_rounds, &mut st.ccd, vpre, dat.velocity_restitution, Some(&mut dat.warm));
        st.ccd.zones_relist += st.ccd.zones - z0;
    }
}

/// A measurement (`DAT_MISS=1`): each vehicle node that ends the step on the other side of one of
/// its own triangles' planes, inside it, having started outside on the other side; whether that
/// node–triangle pair was a record and a CCD pair this step.
fn miss_dbg(surf: &[Surface], x0: &[Vec<Vec3>], x1: &[Vec<Vec3>], pairs: &[super::super::ccd::CcdPair], recs: &[Rec]) {
    let vol = |a: Vec3, b: Vec3, c: Vec3, p: Vec3| ((b - a).cross(c - a)).dot(p - a);
    for v in 0..surf.len() {
        for t in &surf[v].tris {
            let (a0, a1) = (t.map(|k| x0[v][k as usize]), t.map(|k| x1[v][k as usize]));
            for &i in &surf[v].nodes {
                if t.contains(&i) {
                    continue;
                }
                let (p0, p1) = (x0[v][i as usize], x1[v][i as usize]);
                let (s0, s1) = (vol(a0[0], a0[1], a0[2], p0), vol(a1[0], a1[1], a1[2], p1));
                if s0 * s1 >= 0.0 {
                    continue;
                }
                // Inside the triangle at the end and at the start (projected), and close.
                let c1 = super::super::closest_on_triangle(p1, a1[0], a1[1], a1[2]);
                let c0 = super::super::closest_on_triangle(p0, a0[0], a0[1], a0[2]);
                let n1 = (a1[1] - a1[0]).cross(a1[2] - a1[0]).normalize_or_zero();
                let n0 = (a0[1] - a0[0]).cross(a0[2] - a0[0]).normalize_or_zero();
                if (p1 - c1).cross(n1).length() > 1e-4 || (p0 - c0).cross(n0).length() > 1e-4 {
                    continue;
                }
                let key = |k: &[(usize, u32); 4]| k.contains(&(v, i)) && t.iter().all(|&j| k.contains(&(v, j)));
                let pair = pairs.iter().find(|q| key(&q.k));
                let rec = recs.iter().find(|r| r.len == 4 && key(&r.k));
                if let Some(q) = pair {
                    use super::super::ccd::{accd, closest};
                    let a: [Vec3; 4] = q.k.map(|(v, i)| x0[v][i as usize]);
                    let dp: [Vec3; 4] = q.k.map(|(v, i)| x1[v][i as usize] - x0[v][i as usize]);
                    let d0 = closest(q.nn, a).1.length();
                    let cl = |t: f32| { let at = [0, 1, 2, 3].map(|j| a[j] + dp[j] * t); let (w, gv) = closest(q.nn, at); let n = gv.normalize_or_zero(); (0..4).map(|j| w[j] * dp[j].dot(n)).sum::<f32>() };
                    let (t1, t2) = (accd(q.nn, a, dp, q.xi), accd(q.nn, a, dp, 0.5 * d0));
                    let samples: Vec<String> = (0..=10).map(|k| { let t = k as f32 / 10.0; let at = [0, 1, 2, 3].map(|j| a[j] + dp[j] * t); format!("{:.2}", closest(q.nn, at).1.length() * 1e3) }).collect();
                    eprintln!("MISS detail nn {} d0 {:.3} mm | accd xi {:?} closing {:?} | accd half {:?} closing {:?} | gap mm over t {:?}", q.nn, d0 * 1e3, t1, t1.map(cl), t2, t2.map(cl), samples);
                }
                eprintln!("MISS car {v} node {i} through {t:?}: start {:.1} mm end {:.1} mm, travel {:.1} mm | record {} (g {:?}, d {:?}) | pair {} (xi {:?})", (p0 - c0).length() * 1e3, (p1 - c1).length() * 1e3, (p1 - p0).length() * 1e3, rec.is_some(), rec.map(|r| r.g), rec.map(|r| r.d), pair.is_some(), pair.map(|q| q.xi));
            }
        }
    }
}
