//! gen3's node pattern against the BeamNG census (round 9): per part the
//! structural node count, node mass p10 / p50 / p90, nearest-neighbour
//! spacing, member length and k·L, and the share of structural nodes **at the
//! bound**: a node is at the bound when the step's stability bound (the row
//! cap or the power iteration) set any of its own members' springs below 95 %
//! of the census spring its role asks for. The census's Gershgorin row (as a
//! multiple of sedan1's bound) is printed beside it.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example gen3_census
//! ```
use engine_softbody::authoring::gen2::build::{Part, Role, SPRING_LENGTH};
use engine_softbody::authoring::gen2::{model, rig};

fn pct(v: &mut [f32], q: f32) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f32 * q).round() as usize]
}

/// A member: a beam the bound may soften that carries the structure (not a
/// lane or edge tie, hardware, a stop or a seal).
fn member(r: Role) -> bool {
    !matches!(
        r,
        Role::Tie
            | Role::Backing
            | Role::EdgeTie
            | Role::EdgeStop
            | Role::Hinge
            | Role::Bolt
            | Role::Clip
            | Role::Bearing
            | Role::Stop
            | Role::Susp
            | Role::Mount
            | Role::Engine
    )
}

fn main() {
    let t0 = std::time::Instant::now();
    let m = model::build();
    let r = rig::rig(&m);
    let car = &r.car;
    let hub = rig::hub_extra(car);
    let rows = car.rows(&hub);
    let n = car.nodes.len();
    let stiff = engine_softbody::authoring::gen2::build::stiff_mult();
    // Per node: the lowest share of its census spring any of its members kept.
    let mut kept = vec![f32::INFINITY; n];
    for b in &car.beams {
        if !member(b.role) || b.break_group.is_some() {
            continue;
        }
        let sp = b.role.spec(b.part);
        let l = car.p(b.a).distance(car.p(b.b)).max(0.02);
        let nom = stiff
            * b.section
            * if sp.k_fixed > 0.0 {
                sp.k_fixed
            } else {
                sp.kl / l.max(SPRING_LENGTH)
            };
        if nom <= 0.0 {
            continue;
        }
        let f = b.k / nom;
        kept[b.a] = kept[b.a].min(f);
        kept[b.b] = kept[b.b].min(f);
    }
    let structural = |i: usize| {
        let nd = &car.nodes[i];
        !nd.contact
            && !nd.edge
            && !nd.no_group
            && nd.part != Part::Engine
            && !nd.name.starts_with("hub")
            && !nd.name.starts_with("rim")
    };
    println!(
        "rigged in {:.1} s: {} nodes ({} contact-only, {} edge), {} beams; omega_h {:.3}; mass {:.1} kg",
        t0.elapsed().as_secs_f32(),
        n,
        car.nodes.iter().filter(|x| x.contact).count(),
        car.nodes.iter().filter(|x| x.edge).count(),
        car.beams.len(),
        car.omega_h,
        car.nodes.iter().map(|x| x.mass).sum::<f32>() + hub.values().sum::<f32>()
    );
    println!("part | structural (all) | kg | node kg p10/p50/p90 | spacing p50 | member L p50 | member k·L p50 | at the bound | kept p50 | row p50/p90");
    let mut all_m = Vec::new();
    let mut all_bound = (0usize, 0usize);
    for part in Part::ALL {
        let ids: Vec<usize> = (0..n).filter(|&i| car.nodes[i].part == part).collect();
        let st: Vec<usize> = ids.iter().copied().filter(|&i| structural(i)).collect();
        if st.is_empty() {
            continue;
        }
        let kg: f32 = ids.iter().map(|&i| car.nodes[i].mass).sum();
        let mut ms: Vec<f32> = st.iter().map(|&i| car.nodes[i].mass).collect();
        all_m.extend(ms.iter().copied());
        let mut sp: Vec<f32> = st
            .iter()
            .map(|&i| {
                st.iter()
                    .filter(|&&j| j != i)
                    .map(|&j| car.p(i).distance(car.p(j)))
                    .fold(f32::MAX, f32::min)
            })
            .collect();
        let mb: Vec<_> = car
            .beams
            .iter()
            .filter(|b| b.part == part && member(b.role) && b.break_group.is_none())
            .filter(|b| structural(b.a) && structural(b.b))
            .collect();
        let mut l: Vec<f32> = mb.iter().map(|b| car.p(b.a).distance(car.p(b.b))).collect();
        let mut kl: Vec<f32> = mb
            .iter()
            .map(|b| b.k * car.p(b.a).distance(car.p(b.b)))
            .collect();
        let with: Vec<usize> = st
            .iter()
            .copied()
            .filter(|&i| kept[i].is_finite())
            .collect();
        let bound = with.iter().filter(|&&i| kept[i] < 0.95).count();
        all_bound.0 += bound;
        all_bound.1 += with.len();
        let mut kp: Vec<f32> = with.iter().map(|&i| kept[i]).collect();
        let mut rw: Vec<f32> = st.iter().map(|&i| rows[i]).collect();
        let mut ends: Vec<f32> = st
            .iter()
            .map(|&i| {
                car.beams
                    .iter()
                    .filter(|b| (b.a == i || b.b == i) && b.k > 1.0 && !b.role.spec(b.part).support)
                    .count() as f32
            })
            .collect();
        println!(
            "{:?} | {} ({}) | {kg:.1} | {:.2}/{:.2}/{:.2} | {:.3} | {:.3} | {:.2e} | {:.0} % | {:.2} | {:.2}/{:.2} | ends {:.0}",
            part,
            st.len(),
            ids.len(),
            pct(&mut ms, 0.1),
            pct(&mut ms, 0.5),
            pct(&mut ms, 0.9),
            pct(&mut sp, 0.5),
            pct(&mut l, 0.5),
            pct(&mut kl, 0.5),
            100.0 * bound as f32 / with.len().max(1) as f32,
            pct(&mut kp, 0.5),
            pct(&mut rw, 0.5),
            pct(&mut rw, 0.9),
            pct(&mut ends, 0.5),
        );
    }
    let mut every: Vec<f32> = (0..n)
        .filter(|&i| !car.nodes[i].no_group)
        .map(|i| car.nodes[i].mass)
        .collect();
    println!(
        "structural nodes: {} at the bound of {} ({:.0} %); node mass p50 {:.2} kg (structural), {:.2} kg (every node)",
        all_bound.0,
        all_bound.1,
        100.0 * all_bound.0 as f32 / all_bound.1.max(1) as f32,
        pct(&mut all_m, 0.5),
        pct(&mut every, 0.5)
    );
    if std::env::var("GEN3_CENSUS_NODES").is_ok() {
        for i in 0..n {
            if structural(i) {
                println!(
                    "  {} {:?} {:.2} kg kept {:.2}",
                    car.nodes[i].name, car.nodes[i].part, car.nodes[i].mass, kept[i]
                );
            }
        }
    }
}
