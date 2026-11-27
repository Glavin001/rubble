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
    super::super::ccd::resolve(&all, &x0e, &mut x1e, &mut vse, &me, &ime, STEP, dat.ccd_rounds, dat.soft_zones, dat.zone_rounds, &mut st.ccd, &vpe, dat.restitution, Some(&mut dat.warm));
    relist_velocity(dat, recs, own_reach, surf, off, &x0e, &mut x1e, &mut vse, &me, &ime, &vpe, reach_v, all.len(), &extra, st);
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
        super::super::ccd::resolve(&more, x0, x1, vs, m, im, STEP, dat.ccd_rounds, dat.soft_zones, dat.zone_rounds, &mut st.ccd, vpre, dat.restitution, Some(&mut dat.warm));
    }
}
