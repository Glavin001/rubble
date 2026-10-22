//! The event's cumulative energy budget (planning's fix of the per-frame energy bound; the host
//! and P2's decode compute it identically): a per-frame `+TOL` bound compounds over an event's
//! frames (`1.05^100`, about 130x), so the bound is carried across the event's routed calls.
//!
//! The ledger, host state over one event's routed calls in order:
//! - `E_ref`: the largest frame-start `KE + E_elastic` of any call so far;
//! - `gain`: the signed sum, over the calls so far, of each call's change of its island's
//!   mechanical energy `KE + E_elastic + PE` (gravity's work included through `PE`, so a
//!   falling island gains nothing).
//!
//! A call may change its island's mechanical energy by at most the remaining budget
//! `TOL E_ref - gain` (never below 0): its frame-end `KE + E_elastic + PE` at most its start's
//! plus that. The sum is signed: energy the island has dissipated earlier is budget a later
//! frame may spend (an island gains energy only by contact with bodies outside it, and the
//! oracle's own trajectories never come out ahead of what they dissipated: on all of P1's
//! calls-v5, 604 events and 40,713 calls, the largest `gain / E_ref` is <= 0, while a frame
//! alone gains up to 3.7x its start's energy, so neither a per-frame bound nor a sum of the
//! frames' gains alone is oracle-neutral).
//!
//! Recorded calls (`targets`) give the ledger from their own energy scalars
//! (`ledger_of_calls`); a live call carries the host's ledger as inputs
//! (`energy.E_budget_ref`, `energy.E_budget_gain`, `energy.E_budget_remaining`).

use std::collections::BTreeMap;

pub const TOL: f64 = 0.05;

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct Ledger {
    pub e_ref: f64,
    pub gain: f64,
    pub calls: usize,
}

impl Ledger {
    /// A call opens with its island's frame-start `KE + E_elastic`: `E_ref` updated, the
    /// remaining budget returned (the energy the call may add to its island).
    pub fn open(&mut self, ke_elastic_start: f64) -> f64 {
        self.e_ref = self.e_ref.max(ke_elastic_start);
        self.remaining()
    }

    pub fn remaining(&self) -> f64 {
        (TOL * self.e_ref - self.gain).max(0.0)
    }

    /// A call closes with its island's change of `KE + E_elastic + PE` over the frame.
    pub fn close(&mut self, delta: f64) {
        self.gain += delta;
        self.calls += 1;
    }

    /// The ledger as a live call's energy inputs.
    pub fn insert(&self, energy: &mut BTreeMap<String, f64>) {
        energy.insert("E_budget_ref".into(), self.e_ref);
        energy.insert("E_budget_gain".into(), self.gain);
        energy.insert("E_budget_remaining".into(), self.remaining());
    }
}

/// A recorded call's change of its island's mechanical energy, from its energy scalars:
/// `KE_out + E_elastic_end - KE_in - E_elastic_start - W_gravity`.
pub fn call_delta(e: &BTreeMap<String, f64>) -> Result<f64, String> {
    let g = |k: &str| e.get(k).copied().ok_or_else(|| format!("call energy lacks {k}"));
    Ok(g("KE_out")? + g("E_elastic_end")? - g("KE_in")? - g("E_elastic_start")? - g("W_gravity")?)
}

/// The ledger before each of an event's recorded calls (`energy` per call, in call order:
/// the calls' frames ascending), each after its `open`: (ledger, remaining budget).
pub fn ledger_of_calls(energy: &[&BTreeMap<String, f64>]) -> Result<Vec<(Ledger, f64)>, String> {
    let mut l = Ledger::default();
    let mut out = Vec::with_capacity(energy.len());
    for e in energy {
        let s = e.get("KE_in").copied().ok_or("call energy lacks KE_in")? + e.get("E_elastic_start").copied().ok_or("call energy lacks E_elastic_start")?;
        let r = l.open(s);
        out.push((l, r));
        l.close(call_delta(e)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A creep of +5% per frame (each frame within the old per-frame bound) is held to 5% of
    /// `E_ref` over the event, not `1.05^n`.
    #[test]
    fn a_creep_is_bounded_over_the_event() {
        let mut l = Ledger::default();
        let mut e = 100.0;
        for _ in 0..100 {
            let allowed = l.open(e);
            let d = (0.05 * e).min(allowed);
            l.close(d);
            e += d;
        }
        // E_ref follows the island's energy up, so the fixed point is 100 / (1 - TOL)
        assert!(e <= 100.0 / (1.0 - TOL) + 1e-9, "{e}");
    }

    /// Energy dissipated earlier is budget: an island that lost half its energy may regain it
    /// (contact with bodies outside it), plus the tolerance, and no more.
    #[test]
    fn dissipation_is_credit() {
        let mut l = Ledger::default();
        l.open(100.0);
        l.close(-50.0);
        assert_eq!(l.open(50.0), 55.0);
    }
}
