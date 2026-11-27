//! Acceptance criteria as data: a metric's name, the band it must fall in, and how hard the
//! requirement is.

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

/// `metric` must lie in [`lo`, `hi`] (either end open when None).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Criterion {
    pub name: &'static str,
    pub metric: &'static str,
    pub lo: Option<f64>,
    pub hi: Option<f64>,
    pub tier: Tier,
    /// The first-principles reason for the band.
    pub why: &'static str,
}

impl Criterion {
    pub const fn new(tier: Tier, name: &'static str, metric: &'static str, lo: Option<f64>, hi: Option<f64>, why: &'static str) -> Self {
        Self { name, metric, lo, hi, tier, why }
    }

    /// Judged on a run's metrics. A metric that is missing or not finite fails.
    pub fn judge(&self, m: &BTreeMap<String, f64>) -> CriterionResult {
        let value = m.get(self.metric).copied();
        let pass = value.is_some_and(|v| v.is_finite() && self.lo.is_none_or(|l| v >= l) && self.hi.is_none_or(|h| v <= h));
        CriterionResult { name: self.name.into(), metric: self.metric.into(), value, lo: self.lo, hi: self.hi, tier: self.tier, pass, why: self.why.into() }
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
    pub why: String,
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
