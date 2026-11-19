//! What the reference does for the lab's sedan: counts, kinds and every optional path's
//! switch, so the floor kernel ports exactly the paths in use.
use engine_softbody::explicit_beams::*;
use gpu_floor::car;
use std::collections::BTreeMap;

fn main() -> anyhow::Result<()> {
    let wheels = if std::env::args().any(|a| a == "--pressure") { JbeamWheels::Pressure } else { JbeamWheels::Raycast };
    let t = car::load_template(&car::default_car_dir(), wheels)?;
    let s = car::place(&t, 0.3, 0.0);
    println!("nodes {} beams {} triangles {} candidates {} boxes {} shapes {}", s.nodes.len(), s.beams.len(), s.triangles.len(), s.candidates.len(), s.boxes.len(), s.shapes.len());
    println!("settings {:?}", s.settings);
    println!("collision_range {} grid_cell {} contact_every_node {} two_body {} anti_clip {} own_travel {} fixed {} dissipative {} surface {} exact_friction {} plastic_strength_drop {} beamng {} damp_scheme {} rest_pose {:?}",
        s.collision_range, s.grid_cell, s.contact_every_node, s.two_body_contact, s.anti_clip, s.own_travel_reach, s.fixed_contact, s.dissipative_contact, s.surface_contact, s.exact_friction, s.plastic_strength_drop, s.beamng_profile, s.damp_scheme, s.rest_pose);
    println!("drive {} wings {} fusedrag {} centering {} shocks2 {} triggers {} commands {} slides {} brakes {} tyres {} tread {} damp_filters {} torsion_watch {} beam_ext {} ext_of_nonzero {} tri_breaks {} tri_dead {} deform_state {} contact_points {} self_groups {} tri_excl_total {} anti_only {} contact_throttle {} contact_force {} retired {} grab {}",
        s.drive.is_some(), s.wings.len(), s.fusedrag.is_some(), s.centering.len(), s.shocks2.len(), s.triggers.len(), s.commands.len(), s.slides.len(), s.brakes.len(), s.tyres.len(), s.tread.len(), s.damp_filters.len(), s.torsion_watch.len(), s.beam_ext.len(), s.ext_of.iter().filter(|&&x| x != 0).count(), s.tri_breaks.len(), s.tri_dead.len(), s.deform_state.len(), s.contact_points.len(), s.self_groups.len(), s.tri_excl.iter().map(|v| v.len()).sum::<usize>(), s.anti_only_tris.iter().filter(|&&x| x).count(), s.contact_throttle.len(), s.contact_force.is_some(), s.retired.len(), s.grab.is_some());
    let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
    for b in &s.beams {
        let key = format!("{:?}/{:?} grp{} filt{} plastic{} lk{} brk{}", b.kind, b.ty, (b.group != 0) as u8, (b.shock != 0) as u8, (b.plastic != 0.0) as u8, (b.limit_k != 0.0) as u8, b.strength.is_finite() as u8);
        *kinds.entry(key).or_default() += 1;
    }
    for (k, n) in kinds { println!("  {n:5} {k}"); }
    let mut flags: BTreeMap<u32, usize> = BTreeMap::new();
    for n in &s.nodes { *flags.entry(n.flags).or_default() += 1; }
    println!("node flags {:?}", flags);
    let inc: Vec<usize> = s.incidence.iter().map(|v| v.len()).collect();
    println!("incidence max {} mean {:.2}", inc.iter().max().unwrap(), inc.iter().sum::<usize>() as f32 / inc.len() as f32);
    let mut ext: BTreeMap<String, usize> = BTreeMap::new();
    for x in &s.beam_ext {
        let b = &s.beams[x.beam as usize];
        let key = format!("{:?} mask{} deform{:?} ramp{}/{} extra{:?} grp{}", b.kind, x.mask, x.deform.map(|d| (d != 0) as u8), x.ramp_left, x.ramp_steps, x.extra_groups.map(|g| (g != 0) as u8), b.group);
        *ext.entry(key).or_default() += 1;
    }
    for (k, n) in ext { println!("  ext {n:4} {k}"); }
    let groups: std::collections::BTreeSet<i32> = s.beams.iter().map(|b| b.group.abs()).filter(|&g| g != 0).collect();
    println!("groups {} max {:?}; torsion_watch {:?}", groups.len(), groups.iter().max(), s.torsion_watch);
    println!("tri_breaks nonempty {}", s.tri_breaks.iter().filter(|t| t.iter().any(|&j| j != u32::MAX)).count());
    let selfg: std::collections::BTreeSet<u32> = s.self_groups.iter().copied().collect();
    println!("self groups {:?}", selfg.len());
    // Incidence order and self-loops.
    let mut sorted = true; let mut loops = 0;
    for (i, l) in s.incidence.iter().enumerate() { if l.windows(2).any(|w| w[0] > w[1]) { sorted = false; } for &j in l { let b = &s.beams[j as usize]; if b.a == b.b && b.a == i as u32 { loops += 1; } } }
    println!("incidence sorted {sorted} self-loop entries {loops}");
    // Parameter redundancy among the JBeam plain and support beams.
    let uniq = |f: &dyn Fn(&Beam) -> Vec<u32>| s.beams.iter().filter(|b| matches!(b.kind, ExplicitKind::JbeamPlain | ExplicitKind::JbeamSupport)).map(f).collect::<std::collections::BTreeSet<_>>().len();
    println!("distinct (k, c) {}, k {}, c {}, tc {}, tt {}, gate {}, strength {}, limit_k {}, short/rest0 {}, long/rest0 {}, plastic {}, (k,c,tc,tt,strength,limit_k) {}",
        uniq(&|b| vec![b.k.to_bits(), b.c.to_bits()]), uniq(&|b| vec![b.k.to_bits()]), uniq(&|b| vec![b.c.to_bits()]), uniq(&|b| vec![b.tc.to_bits()]), uniq(&|b| vec![b.tt.to_bits()]),
        uniq(&|b| vec![b.gate.to_bits()]), uniq(&|b| vec![b.strength.to_bits()]), uniq(&|b| vec![b.limit_k.to_bits()]),
        uniq(&|b| vec![((b.short_bound / b.rest0) * 1e4).round() as u32]), uniq(&|b| vec![((b.long_bound / b.rest0) * 1e4).round() as u32]), uniq(&|b| vec![b.plastic.to_bits()]),
        uniq(&|b| vec![b.k.to_bits(), b.c.to_bits(), b.tc.to_bits(), b.tt.to_bits(), b.strength.to_bits(), b.limit_k.to_bits()]));
    let mut deg: Vec<usize> = s.incidence.iter().map(|v| v.len()).collect();
    deg.sort();
    println!("degree percentiles: min {} p50 {} p90 {} max {}; sum of each 32-node group's max (sorted) {}", deg[0], deg[deg.len() / 2], deg[deg.len() * 9 / 10], deg[deg.len() - 1], deg.chunks(32).map(|c| c[c.len() - 1]).sum::<usize>());
    let first = |f: &dyn Fn(&Beam) -> bool| s.beams.iter().position(f);
    let last = |f: &dyn Fn(&Beam) -> bool| s.beams.iter().rposition(f);
    println!("first aero record {:?}, last non-aero {:?}; couplers {:?}..{:?}, torsion {:?}..{:?}", first(&|b| b.kind == ExplicitKind::JbeamAeroTri), last(&|b| !matches!(b.kind, ExplicitKind::JbeamAeroTri | ExplicitKind::JbeamAeroTri2)), first(&|b| b.kind == ExplicitKind::JbeamCoupler), last(&|b| b.kind == ExplicitKind::JbeamCoupler), first(&|b| b.kind == ExplicitKind::JbeamTorsion), last(&|b| b.kind == ExplicitKind::JbeamTorsion));
    let sup: Vec<f32> = s.beams.iter().filter(|b| b.kind == ExplicitKind::JbeamSupport).map(|b| b.long_bound / b.rest0).collect();
    println!("support long/rest0 min {:?} max {:?}; gate/k*rest0 ...", sup.iter().cloned().fold(f32::MAX, f32::min), sup.iter().cloned().fold(f32::MIN, f32::max));
    // Ownership: how many two-node beams a node's lane could own (at most `cap`) if the
    // owned ones must be the first or the last entries of the node's ordered add list.
    {
        let scalar = |k: ExplicitKind| matches!(k, ExplicitKind::JbeamPlain | ExplicitKind::JbeamSupport | ExplicitKind::JbeamBounded | ExplicitKind::JbeamAniso | ExplicitKind::Plain);
        let nn = s.nodes.len();
        let mut lists: Vec<Vec<usize>> = vec![Vec::new(); nn];
        let mut all_scalar = 0;
        for (j, b) in s.beams.iter().enumerate() {
            if matches!(b.kind, ExplicitKind::JbeamTorsionPair | ExplicitKind::JbeamAeroTri2) { continue; }
            if scalar(b.kind) { all_scalar += 1; }
            for n in [b.a, b.b] { lists[n as usize].push(j); }
            if b.kind == ExplicitKind::JbeamAeroTri { /* third node: not a two-node entry */ }
        }
        for cap in [4usize, 5, 6, 8] {
            for both_ends in [false, true] {
                let mut owner = vec![usize::MAX; s.beams.len()];
                let mut pre = vec![0usize; nn];
                let mut suf = vec![0usize; nn];
                let mut open_pre = vec![true; nn];
                let mut open_suf = vec![both_ends; nn];
                loop {
                    let mut changed = false;
                    for i in 0..nn {
                        if open_pre[i] && pre[i] + suf[i] < cap && pre[i] + suf[i] < lists[i].len() {
                            let j = lists[i][pre[i]];
                            if scalar(s.beams[j].kind) && owner[j] == usize::MAX { owner[j] = i; pre[i] += 1; changed = true; } else { open_pre[i] = false; }
                        }
                        if open_suf[i] && pre[i] + suf[i] < cap && pre[i] + suf[i] < lists[i].len() {
                            let j = lists[i][lists[i].len() - 1 - suf[i]];
                            if scalar(s.beams[j].kind) && owner[j] == usize::MAX { owner[j] = i; suf[i] += 1; changed = true; } else { open_suf[i] = false; }
                        }
                    }
                    if !changed { break; }
                }
                let owned = owner.iter().filter(|&&o| o != usize::MAX).count();
                println!("ownership: cap {cap}, {}: {owned} of {all_scalar} two-node beams owned in order", if both_ends { "first or last entries" } else { "first entries only" });
            }
        }
    }
    let tot: f32 = s.nodes.iter().map(|n| n.mass).sum();
    println!("mass {tot}");
    if let Some(d) = &s.drive { println!("drive: {:?}", d.config.wheels.len()); }
    Ok(())
}
