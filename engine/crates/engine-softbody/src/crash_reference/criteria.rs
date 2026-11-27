//! Acceptance criteria as data: a metric's name, the band it must fall in, how hard the
//! requirement is, and what the band rests on.
//!
//! **A criterion can fail a run only when its band has a [`Basis`]**: a law (conservation, a
//! closed form) or cited data measured under matching conditions, with the band set to the
//! data's whole spread. A band that is an estimate, or read off reference footage, is
//! measured and reported and asserts nothing, whatever its tier: a wrong bound would be a
//! wrong goal. Where only a direction is certain (the struck side absorbs more), assert the
//! direction (a ratio above 1), not a size.
//!
//! **Known failures are data, with a ratchet.** An asserted criterion that fails today in a
//! contact setting is recorded on it ([`Criterion::known_failure`], [`Criterion::known_intermittent`])
//! with its measured value, the date and where it is documented. A known failure that fails
//! is reported as such and does not fail the run; one recorded as failing in every run that
//! passes in all of a setting's runs fails the verdict ([`super::verdict`]: "remove the
//! entry"), so the list only shrinks. The band, tier and basis are never touched by it.

use super::Contact;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// How a known failure shows across the jittered runs ([`super::Jitter::run`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ts_rs::TS)]
#[serde(rename_all = "snake_case")]
pub enum Occurs {
    /// Fails in every run: passing in all of them fails the verdict (remove the entry).
    EveryRun,
    /// Fails in some runs: passing in all of them is only noted.
    SomeRuns,
}

/// An asserted criterion's failure that is known in some contact settings: why, the
/// measured value, the date and where it is documented.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KnownFailure {
    pub contacts: Vec<Contact>,
    pub occurs: Occurs,
    pub note: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ts_rs::TS)]
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
    /// Where it fails today ([`KnownFailure`]); empty for most.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub known: Vec<KnownFailure>,
}

impl Criterion {
    pub const fn new(tier: Tier, name: &'static str, metric: &'static str, lo: Option<f64>, hi: Option<f64>, why: &'static str) -> Self {
        Self { name, metric, lo, hi, tier, why, basis: Basis::Estimate, known: Vec::new() }
    }

    /// It fails today in every run in `contacts`: `note` says why, the measured value, the
    /// date and where it is documented. Passing in all runs of a setting then fails the
    /// verdict, so the entry is removed when it is fixed.
    pub fn known_failure(self, contacts: &[Contact], note: &'static str) -> Self {
        self.known_as(contacts, Occurs::EveryRun, note)
    }

    /// It fails today in some of the runs in `contacts` (`note` as [`Self::known_failure`]).
    pub fn known_intermittent(self, contacts: &[Contact], note: &'static str) -> Self {
        self.known_as(contacts, Occurs::SomeRuns, note)
    }

    fn known_as(mut self, contacts: &[Contact], occurs: Occurs, note: &'static str) -> Self {
        self.known.push(KnownFailure { contacts: contacts.to_vec(), occurs, note });
        self
    }

    /// The known failure recorded for `contact`, if any.
    pub fn known_in(&self, contact: Contact) -> Option<&KnownFailure> {
        self.known.iter().find(|k| k.contacts.contains(&contact))
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

    /// Judged on a run's metrics, without its known failures (no contact setting). A metric
    /// that is missing or not finite fails.
    pub fn judge(&self, m: &BTreeMap<String, f64>) -> CriterionResult {
        let value = m.get(self.metric).copied();
        let pass = value.is_some_and(|v| v.is_finite() && self.lo.is_none_or(|l| v >= l) && self.hi.is_none_or(|h| v <= h));
        CriterionResult {
            name: self.name.into(),
            metric: self.metric.into(),
            value,
            lo: self.lo,
            hi: self.hi,
            tier: self.tier,
            pass,
            asserted: self.asserted(),
            why: self.why.into(),
            basis: match self.basis {
                Basis::Law(x) => format!("law: {x}"),
                Basis::Cited(x) => format!("cited: {x}"),
                Basis::Estimate => "estimate".into(),
            },
            known: false,
            known_occurs: None,
            known_note: None,
        }
    }

    /// Judged on a run in `contact`, with the known failure recorded for that setting.
    pub fn judge_in(&self, m: &BTreeMap<String, f64>, contact: Contact) -> CriterionResult {
        let mut r = self.judge(m);
        if let Some(k) = self.known_in(contact) {
            r.known = true;
            r.known_occurs = Some(k.occurs);
            r.known_note = Some(k.note.into());
        }
        r
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, ts_rs::TS)]
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
    /// A known failure is recorded for this run's contact setting ([`Criterion::known_in`]):
    /// failing it does not fail the run.
    #[serde(default)]
    pub known: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub known_occurs: Option<Occurs>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub known_note: Option<String>,
}

impl CriterionResult {
    /// It fails the run: asserted, failed, and not a known failure.
    pub fn fails_run(&self) -> bool {
        self.asserted && !self.pass && !self.known
    }
    /// A known failure that failed.
    pub fn known_failed(&self) -> bool {
        self.asserted && !self.pass && self.known
    }
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
