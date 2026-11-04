//! City war, first shell step: which sleeping bodies each ccd_wake rule wakes (probe).
#![cfg(feature = "stress")]
use glam::{Quat, Vec3};
use rubble::*;

fn city(by_collider: bool) -> World {
    let house = rubble::city::city_house().unwrap();
    let def = rubble::city::CityDef {
        houses: 16,
        sleep: true,
        ..Default::default()
    };
    let mut wd = def.world_def(&house).unwrap();
    wd.features.ccd_by_collider = by_collider;
    let mut w = World::new(GpuContext::Default, &wd).unwrap();
    let mut c = rubble::city::City::begin(&mut w, &house, def).unwrap();
    c.place(&mut w, 16).unwrap();
    for _ in 0..rubble::city::CITY_SETTLE {
        w.step().unwrap();
        w.poll_blocking().unwrap();
    }
    c.start(
        &mut w,
        rubble::city::CityScenario::from_name("war").unwrap(),
    )
    .unwrap();
    w
}

fn extent(c: &engine_schema::Collider) -> f32 {
    let p = c.params;
    let r = match c.shape {
        0 => p[0],
        1 => p[0] + p[1],
        3 | 5 => p[3],
        _ => Vec3::new(p[0], p[1], p[2]).length(),
    };
    Vec3::from(c.local_pos).length() + r
}

#[test]
#[ignore]
fn ccd_wake_rules() {
    let steps: u32 = std::env::var("CCDW_STEP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let mut a = city(true);
    let mut b = city(false);
    for _ in 1..steps {
        for w in [&mut a, &mut b] {
            w.step().unwrap();
            w.poll_blocking().unwrap();
        }
    }
    let bodies = a.sim().download_bodies();
    assert!(
        bodies
            .iter()
            .zip(b.sim().download_bodies())
            .all(|(x, y)| x.position == y.position && x.flags == y.flags),
        "worlds agree before the step"
    );
    let cols = a.sim().download_colliders();
    let p = a.solver();
    let (dt, off) = (p.dt, p.contact_offset);
    let nc = cols.len();
    let first = |body: u32| cols.partition_point(|c| c.body < body);
    let run = |body: u32| {
        (first(body)..nc)
            .take_while(|&c| cols[c].body == body)
            .collect::<Vec<_>>()
    };
    let all = |body: u32| {
        (0..nc)
            .filter(|&c| cols[c].body == body)
            .collect::<Vec<_>>()
    };
    let sorted = cols.windows(2).all(|w| w[0].body <= w[1].body);
    eprintln!(
        "ccdw: {} colliders, sorted by body: {sorted}; dt {dt} contactOffset {off}",
        nc
    );
    let dynamic = |s: &engine_schema::BodyState| s.inv_mass > 0.0 && s.flags & 1 != 0;
    let sleeping = |s: &engine_schema::BodyState| s.flags & 4 != 0;
    let fast: Vec<u32> = (0..bodies.len() as u32)
        .filter(|&i| {
            let s = &bodies[i as usize];
            s.flags & engine_schema::BODY_CCD != 0
                && dynamic(s)
                && !sleeping(s)
                && Vec3::from(s.linear_vel).length() * dt > 4.0 * off
        })
        .collect();
    let reach = |ids: &[usize]| ids.iter().map(|&c| extent(&cols[c])).fold(0.0f32, f32::max);
    let shells: Vec<(u32, Vec3, Vec3, f32, f32)> = fast
        .iter()
        .map(|&f| {
            let s = &bodies[f as usize];
            (
                f,
                Vec3::from(s.position),
                Vec3::from(s.linear_vel) * dt,
                reach(&run(f)),
                reach(&all(f)),
            )
        })
        .collect();
    for s in &shells {
        eprintln!(
            "ccdw: shell {} r main {} r all {} run {:?} all {:?}",
            s.0,
            s.3,
            s.4,
            run(s.0),
            all(s.0)
        );
    }
    let near = |pos: Vec3, own: f32, main: bool| {
        shells.iter().find_map(|&(f, a, d, rm, ra)| {
            let r = if main { rm } else { ra };
            let t = ((pos - a).dot(d) / d.dot(d).max(1e-20)).clamp(0.0, 1.0);
            let dist = (a + d * t - pos).length();
            (dist <= 1.25 * r + own).then_some((f, dist, 1.25 * r + own))
        })
    };
    let mut extra = Vec::new();
    let (mut n_main, mut n_all) = (0, 0);
    for i in 0..bodies.len() as u32 {
        let s = &bodies[i as usize];
        if !dynamic(s) || !sleeping(s) {
            continue;
        }
        let pos = Vec3::from(s.position);
        let wm = near(pos, reach(&run(i)), true).is_some();
        let wa = all(i)
            .iter()
            .any(|&c| near(pos, extent(&cols[c]), false).is_some());
        n_main += wm as u32;
        n_all += wa as u32;
        if wa && !wm {
            extra.push(i);
        }
        if wm && !wa {
            eprintln!("ccdw: body {i} woken by main's rule only");
        }
    }
    eprintln!(
        "ccdw: woken by main's rule {n_main}, by the per-collider rule {n_all}, extra {extra:?}"
    );
    for &i in &extra {
        let s = &bodies[i as usize];
        let pos = Vec3::from(s.position);
        let q = Quat::from_array(s.orientation);
        let r = run(i);
        let al = all(i);
        eprintln!(
            "ccdw: body {i} at {pos:?}: first {} run {:?} all colliders {:?}",
            first(i),
            r,
            al
        );
        for &c in &al {
            let col = &cols[c];
            let wpos = pos + q * Vec3::from(col.local_pos);
            let e = extent(col);
            for &(f, a, d, rm, ra) in &shells {
                let t = ((pos - a).dot(d) / d.dot(d).max(1e-20)).clamp(0.0, 1.0);
                let dist = (a + d * t - pos).length();
                let tc = ((wpos - a).dot(d) / d.dot(d).max(1e-20)).clamp(0.0, 1.0);
                let gap = (a + d * tc - wpos).length() - e - ra;
                // The collider's box (shape 2: params.xyz half extents) against the shell's
                // swept sphere: the closest of 257 points along the sweep.
                let crot = q * Quat::from_array(col.local_rot);
                let half = Vec3::new(col.params[0], col.params[1], col.params[2]);
                let obb = (0..=256)
                    .map(|k| {
                        let x = crot.inverse() * (a + d * (k as f32 / 256.0) - wpos);
                        (x.abs() - half).max(Vec3::ZERO).length()
                    })
                    .fold(f32::MAX, f32::min);
                if dist <= 1.25 * ra + e {
                    eprintln!("ccdw:   collider {c} shape {} half {half:?}: box to sweep {obb:.3} m, shell radius {ra:.3}", col.shape);
                    eprintln!(
                        "ccdw:   collider {c} (in main's run: {}) extent {e:.3}: shell {f} (r main {rm:.3}, r {ra:.3}) body-centre distance {dist:.3} <= {:.3}; collider centre to sweep minus extents {gap:.3}",
                        r.contains(&c), 1.25 * ra + e
                    );
                }
            }
        }
    }
    // The GPU's answer: bodies awake after the step in each world.
    for w in [&mut a, &mut b] {
        w.step().unwrap();
        w.poll_blocking().unwrap();
    }
    let (ba, bb) = (a.sim().download_bodies(), b.sim().download_bodies());
    let awake = |v: &[engine_schema::BodyState]| {
        v.iter()
            .enumerate()
            .filter(|(_, s)| dynamic(s) && !sleeping(s))
            .map(|(i, _)| i as u32)
            .collect::<std::collections::BTreeSet<_>>()
    };
    let (aa, ab) = (awake(&ba), awake(&bb));
    eprintln!("ccdw: GPU awake after the step: per-collider {} per-body {}; only per-collider {:?}; only per-body {:?}", aa.len(), ab.len(), aa.difference(&ab).collect::<Vec<_>>(), ab.difference(&aa).collect::<Vec<_>>());
}
