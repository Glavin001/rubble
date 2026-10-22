//! Decoded state /2 (`stress_learned::decoded_full`): the oracle's own frame-end bonds and
//! chunks, flattened by P1's rule into columns and written back over a stale island, are the
//! oracle's island bit for bit.

use std::collections::BTreeSet;

use stress_learned::decoded_full::{apply_full, to_value, FullOut};
use stress_learned::full_state_calls::flatten;
use stress_learned::relabel::state_bytes;
use stress_ref::world::{oracle_island_step, World};

fn columns_of(rows: &[Vec<(String, f64)>]) -> Vec<String> {
    rows.iter().flatten().map(|(c, _)| c.clone()).collect::<BTreeSet<_>>().into_iter().collect()
}

fn matrix(rows: &[Vec<(String, f64)>], cols: &[String]) -> Vec<f64> {
    rows.iter()
        .flat_map(|r| cols.iter().map(move |c| r.iter().find(|(n, _)| n == c).map_or(if c.contains('=') { 0.0 } else { f64::NAN }, |(_, x)| *x)))
        .collect()
}

#[test]
fn the_oracles_own_columns_round_trip_bit_for_bit() {
    let mut scene = stress_ref::builders::wall_impact(20.0, "b5_wall_impact_v20");
    scene.sim.duration = 0.2;
    let mut w = World::new(&scene);
    let mut checked = 0;
    for frame in 0..8 {
        let m = w.island_members();
        if !m.chunks.is_empty() && frame >= 2 {
            let start = w.extract_island(&m);
            let native = oracle_island_step(&w, &m).unwrap();
            // the model's columns: every bond and chunk field of the oracle's frame end
            let mut bkeys = Vec::new();
            let mut brows = Vec::new();
            for ((s, _), b) in &native.solver.bonds {
                bkeys.push(stress_ref::keys::bond_key(*s, b.source));
                let mut r = Vec::new();
                flatten(&to_value(b).unwrap(), "", &mut r);
                brows.push(r);
            }
            let mut ckeys = Vec::new();
            let mut crows = Vec::new();
            for ((s, c), ch) in &native.solver.chunks {
                ckeys.push(stress_ref::keys::chunk_key(*s, *c));
                let mut r = Vec::new();
                flatten(&to_value(ch).unwrap(), "", &mut r);
                crows.push(r);
            }
            let (bc, cc) = (columns_of(&brows), columns_of(&crows));
            let outs = vec![
                FullOut { kind: "bond".into(), keys: bkeys, columns: bc.clone(), rows: matrix(&brows, &bc) },
                FullOut { kind: "chunk".into(), keys: ckeys, columns: cc.clone(), rows: matrix(&crows, &cc) },
            ];
            // a stale island: the frame start's bonds and chunks, the decode's topology
            let mut stale = native.clone();
            for ((k, b), (_, s0)) in stale.solver.bonds.iter_mut().zip(&start.solver.bonds) {
                let alive = b.alive;
                *b = s0.clone();
                b.alive = alive;
                let _ = k;
            }
            for ((_, ch), (_, c0)) in stale.solver.chunks.iter_mut().zip(&start.solver.chunks) {
                *ch = *c0;
            }
            let out = apply_full(&stale, &native, &outs).unwrap();
            let flat = |v: rmpv::Value| { let mut r = Vec::new(); flatten(&v, "", &mut r); r };
            let mut diffs = BTreeSet::new();
            for ((_, a), (_, b)) in out.solver.bonds.iter().zip(&native.solver.bonds) {
                for ((n, x), (_, y)) in flat(to_value(a).unwrap()).into_iter().zip(flat(to_value(b).unwrap())) {
                    if x.to_bits() != y.to_bits() { diffs.insert(format!("bond {n}")); }
                }
                if to_value(a).unwrap() != to_value(b).unwrap() { diffs.insert("bond (value)".into()); }
            }
            for ((_, a), (_, b)) in out.solver.chunks.iter().zip(&native.solver.chunks) {
                for ((n, x), (_, y)) in flat(to_value(a).unwrap()).into_iter().zip(flat(to_value(b).unwrap())) {
                    if x.to_bits() != y.to_bits() { diffs.insert(format!("chunk {n}")); }
                }
                if to_value(a).unwrap() != to_value(b).unwrap() { diffs.insert("chunk (value)".into()); }
            }
            assert!(diffs.is_empty(), "frame {frame}: {diffs:?}");
            assert_eq!(state_bytes(&out).unwrap(), state_bytes(&native).unwrap(), "frame {frame}: the round trip differs");
            checked += 1;
        }
        w.solver.reset_peak_utilizations();
        w.step_frame();
    }
    assert!(checked >= 3, "routed frames checked: {checked}");
}
