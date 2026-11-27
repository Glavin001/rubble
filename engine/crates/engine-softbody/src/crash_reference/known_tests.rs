//! Known failures' semantics ([`super::verdict`], `criteria.rs`): an unknown failure fails;
//! a known one does not; a known failure recorded for every run that passes in all of them
//! fails ("remove the entry"); an intermittent one never fails for passing.

use super::*;

fn criterion() -> Criterion {
    Criterion::new(Tier::Invariant, "no energy created", "all.energy_gain_j", None, Some(1.0), "test").law("test")
}

/// A scenario's runs in `contact`, run k with `value(k)`.
fn reports(c: &Criterion, contact: Contact, value: impl Fn(usize) -> f64) -> Vec<CrashReport> {
    (0..RUNS)
        .map(|k| {
            let metrics: BTreeMap<String, f64> = [("all.energy_gain_j".to_string(), value(k))].into();
            let results = vec![c.judge_in(&metrics, contact)];
            CrashReport { id: "T".into(), contact, run: k, metrics, results, series: None }
        })
        .collect()
}

#[test]
fn an_unknown_failure_fails() {
    let v = verdict(&reports(&criterion(), Contact::Edge, |k| if k == 3 { 5.0 } else { 0.0 }));
    assert!(!v.ok());
    assert_eq!(v.failed.len(), 1, "{v:?}");
    assert!(v.failed[0].contains("run 3"), "{v:?}");
}

#[test]
fn a_known_failure_passes_the_run() {
    let c = criterion().known_failure(&[Contact::Edge], "measured 5 J");
    let r = reports(&c, Contact::Edge, |_| 5.0);
    assert!(r.iter().all(|x| x.failures().is_empty() && x.known_failures().len() == 1));
    let v = verdict(&r);
    assert!(v.ok(), "{v:?}");
    assert_eq!(v.known.len(), 1);
    assert!(v.known[0].contains("8 of 8"), "{v:?}");
}

#[test]
fn a_known_failure_elsewhere_still_fails() {
    let c = criterion().known_failure(&[Contact::Edge], "measured 5 J");
    let v = verdict(&reports(&c, Contact::Discrete, |_| 5.0));
    assert_eq!(v.failed.len(), RUNS, "{v:?}");
}

#[test]
fn an_every_run_known_failure_that_passes_fails() {
    let c = criterion().known_failure(&[Contact::Edge, Contact::Continuous], "measured 5 J");
    let v = verdict(&reports(&c, Contact::Continuous, |_| 0.0));
    assert!(!v.ok());
    assert!(v.failed[0].contains("known failure now passes in all 8 runs: remove the entry"), "{v:?}");
}

#[test]
fn an_every_run_known_failure_passing_in_some_is_noted() {
    let c = criterion().known_failure(&[Contact::Edge], "measured 5 J");
    let v = verdict(&reports(&c, Contact::Edge, |k| if k < 2 { 0.0 } else { 5.0 }));
    assert!(v.ok(), "{v:?}");
    assert!(v.notes[0].contains("now intermittent"), "{v:?}");
}

#[test]
fn an_intermittent_known_failure_that_passes_is_fine() {
    let c = criterion().known_intermittent(&[Contact::Edge], "5 of 8 runs");
    let v = verdict(&reports(&c, Contact::Edge, |_| 0.0));
    assert!(v.ok() && v.known.is_empty(), "{v:?}");
    assert_eq!(v.notes.len(), 1);
    let v = verdict(&reports(&c, Contact::Edge, |k| if k % 2 == 0 { 5.0 } else { 0.0 }));
    assert!(v.ok() && v.known[0].contains("4 of 8"), "{v:?}");
}

/// The recorded known failures are on asserted criteria (only they can fail a run), each
/// contact setting at most once a criterion, each with a note.
#[test]
fn recorded_known_failures_are_well_formed() {
    for s in all() {
        for c in (s.criteria)() {
            assert!(c.known.is_empty() || c.asserted(), "{} {}: a known failure on a reported criterion", s.id, c.name);
            for contact in Contact::ALL {
                assert!(c.known.iter().filter(|k| k.contacts.contains(&contact)).count() <= 1, "{} {}: {} twice", s.id, c.name, contact.name());
            }
            assert!(c.known.iter().all(|k| !k.note.is_empty() && !k.contacts.is_empty()), "{} {}", s.id, c.name);
        }
    }
}
