//! Measurement (ignored): the contacts `city.html`'s war barrage needs.
//!
//! The page's dense town (1,024 houses on 16 m lots, a 131,072-slot piece pool, asleep
//! once settled) takes the war scenario's 20 shots, then a barrage: two shells (0.5 m,
//! 120 m/s, from 200 m at a slant) every BARRAGE_EVERY steps onto lots near the map's
//! centre, at most 48 in the world, until the pool has under 2,000 slots left. Prints
//! the engine's contacts as it goes and their peak.
//!
//! cargo test -p engine-world --release --test city_barrage -- --ignored --nocapture
//! BARRAGE_K (houses, 1024), BARRAGE_STEPS (most steps, 8000), BARRAGE_EVERY (steps
//! between volleys, 18: the deployed page's 603 shells in 5,568 steps),
//! BARRAGE_CONTACTS (contact capacity, the city's own by default).
#![cfg(feature = "stress")]

use rubble::city::*;
use rubble::*;

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The pool sizes the contacts: 8 a slot within the contact ingest, never fewer than
/// the chunks' own.
#[test]
fn city_contacts_follow_the_pool() {
    let asset = city_house().unwrap();
    let contacts = |pool| {
        let def = CityDef {
            houses: 1024,
            sleep: true,
            pool,
            ..CityDef::default()
        };
        def.capacity(&asset).unwrap().max_contacts
    };
    assert_eq!(contacts(Some(131_072)), 1 << 20);
    assert_eq!(
        contacts(Some(1 << 20)),
        stress_rubble::threshold::INGEST_CONTACTS
    );
    assert_eq!(contacts(Some(32_768)), 1 << 19);
    assert_eq!(contacts(None), 1 << 19);
}

#[test]
#[ignore]
fn city_barrage() {
    let k: usize = env("BARRAGE_K", 1024);
    let steps: u32 = env("BARRAGE_STEPS", 8000);
    let every: u32 = env("BARRAGE_EVERY", 18);
    let pool = 131_072u32;
    let def = CityDef {
        houses: k,
        map: (k as f64).sqrt().ceil() as f32 * 16.0,
        sleep: true,
        pool: Some(pool),
    };
    let asset = city_house().unwrap();
    let mut wd = def.world_def(&asset).unwrap();
    if let Ok(n) = std::env::var("BARRAGE_CONTACTS") {
        wd.capacity.max_contacts = n.parse().unwrap();
        wd.capacity.max_candidates = wd.capacity.max_candidates.max(wd.capacity.max_contacts);
    }
    let cap = wd.capacity.max_contacts;
    println!("{k} houses, contact capacity {cap}");
    let mut w = World::new(GpuContext::Default, &wd).unwrap();
    let mut c = City::begin(&mut w, &asset, def).unwrap();
    c.place(&mut w, k).unwrap();
    let step = |w: &mut World| {
        w.step().unwrap();
        w.poll_blocking().unwrap();
        w.counters().unwrap()
    };
    for _ in 0..CITY_SETTLE {
        step(&mut w);
    }
    let mut shells = c.start(&mut w, CityScenario::War).unwrap();
    let start_bodies = w.body_count() - shells.len();
    // The page's barrage: lots within reach of the view's centre (the map's).
    let near: Vec<[f32; 2]> = c
        .lots
        .iter()
        .copied()
        .filter(|l| l[0].hypot(l[1]) < 120.0)
        .collect();
    let mut seed = 0x9e37_79b9u32;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed as f32 / u32::MAX as f32
    };
    let (mut peak, mut peak_at, mut fired) = (0u32, 0u32, shells.len());
    for s in 0..steps {
        let pieces = w.body_count() - start_bodies - shells.len();
        if s % every == 0 && (pool as usize).saturating_sub(pieces) >= 2000 {
            for _ in 0..2 {
                let l = near[(rnd() * near.len() as f32) as usize % near.len()];
                let at = [
                    l[0] + (rnd() - 0.5) * 8.0,
                    c.ground + 3.0,
                    l[1] + (rnd() - 0.5) * 6.0,
                ];
                let a = rnd() * std::f32::consts::TAU;
                let from = [
                    at[0] + 60.0 * a.cos(),
                    c.ground + 200.0,
                    at[2] + 60.0 * a.sin(),
                ];
                let d = [at[0] - from[0], at[1] - from[1], at[2] - from[2]];
                let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let v = d.map(|x| x / n * 120.0);
                shells.push(city_shell(&mut w, from, v, 0.5).unwrap());
                fired += 1;
            }
            while shells.len() > 48 {
                w.destroy_body(shells.remove(0)).unwrap();
            }
        }
        let n = step(&mut w);
        assert_eq!(
            n.overflow_flags, 0,
            "step {s}: overflow {:#x}",
            n.overflow_flags
        );
        if n.contact_count > peak {
            (peak, peak_at) = (n.contact_count, s);
        }
        if s % 250 == 0 {
            println!(
                "step {s}: fired {fired} pieces {pieces} bodies {} awake {} contacts {} (peak {peak} at {peak_at})",
                n.body_count, n.awake_count, n.contact_count
            );
        }
    }
    println!("peak contacts {peak} at step {peak_at} of capacity {cap}; {fired} shells");
}
