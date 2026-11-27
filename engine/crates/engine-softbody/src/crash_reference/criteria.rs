//! Acceptance criteria as data: a metric's name, the band it must fall in, how hard the
//! requirement is, and what the band rests on.
//!
//! **A criterion can fail a run only when its band has a [`Basis`]**: a law (conservation, a
//! closed form) or cited data measured under matching conditions, with the band set to the
//! data's whole spread. A band that is an estimate, or read off reference footage, is
//! measured and reported and asserts nothing, whatever its tier: a wrong bound would be a
//! wrong goal. Where only a direction is certain (the struck side absorbs more), assert the
//! direction (a ratio above 1), not a size.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Tier 0: holds in every crash (conservation, nothing passes through, finite).
    Invariant,
    /// Tier 1: what physics fixes for this crash whatever the car (wide bands).
    Physics,
    /// Tier 2: what the reference frames show of this particular car. Reported, never a
    /// failure.
    Shape,
}

impl Tier {
    pub fn name(self) -> &'static str {
        match self {
            Tier::Invariant => "invariant",
            Tier::Physics => "physics",
            Tier::Shape => "shape",
        }
    }
}

/// What a band rests on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "source")]
pub enum Basis {
    /// A conservation law or closed form; the derivation.
    Law(&'static str),
    /// Measured data under matching conditions; the source and what it measured.
    Cited(&'static str),
    /// An estimate or a reading of reference footage: reported only.
    Estimate,
}

/// `metric` must lie in [`lo`, `hi`] (either end open when None).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Criterion {
    pub name: &'static str,
    pub metric: &'static str,
    pub lo: Option<f64>,
    pub hi: Option<f64>,
    pub tier: Tier,
    /// The reason for the band.
    pub why: &'static str,
    pub basis: Basis,
}

impl Criterion {
    pub const fn new(tier: Tier, name: &'static str, metric: &'static str, lo: Option<f64>, hi: Option<f64>, why: &'static str) -> Self {
        Self { name, metric, lo, hi, tier, why, basis: Basis::Estimate }
    }

    /// The band follows from a law: the derivation.
    pub const fn law(mut self, derivation: &'static str) -> Self {
        self.basis = Basis::Law(derivation);
        self
    }

    /// The band is measured data under matching conditions: the source.
    pub const fn cited(mut self, source: &'static str) -> Self {
        self.basis = Basis::Cited(source);
        self
    }

    /// Whether failing it fails a run: tier 0 or 1 with a basis.
    pub fn asserted(&self) -> bool {
        self.tier != Tier::Shape && self.basis != Basis::Estimate
    }

    /// Judged on a run's metrics. A metric that is missing or not finite fails.
    pub fn judge(&self, m: &BTreeMap<String, f64>) -> CriterionResult {
        let value = m.get(self.metric).copied();
        let pass = value.is_some_and(|v| v.is_finite() && self.lo.is_none_or(|l| v >= l) && self.hi.is_none_or(|h| v <= h));
        CriterionResult { name: self.name.into(), metric: self.metric.into(), value, lo: self.lo, hi: self.hi, tier: self.tier, pass, asserted: self.asserted(), why: self.why.into(), basis: match self.basis { Basis::Law(x) => format!("law: {x}"), Basis::Cited(x) => format!("cited: {x}"), Basis::Estimate => "estimate".into() } }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CriterionResult {
    pub name: String,
    pub metric: String,
    pub value: Option<f64>,
    pub lo: Option<f64>,
    pub hi: Option<f64>,
    pub tier: Tier,
    pub pass: bool,
    /// Failing it fails the run ([`Criterion::asserted`]).
    pub asserted: bool,
    pub why: String,
    pub basis: String,
}

impl CriterionResult {
    pub fn shown(&self) -> String {
        self.value.map_or("not measured".into(), |v| format!("{v:.3}"))
    }
    pub fn band(&self) -> String {
        match (self.lo, self.hi) {
            (Some(l), Some(h)) => format!("[{l}, {h}]"),
            (Some(l), None) => format!(">= {l}"),
            (None, Some(h)) => format!("<= {h}"),
            (None, None) => "any".into(),
        }
    }
}
