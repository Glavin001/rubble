//! Shared helpers for the integration tests.
#![allow(dead_code)]

use stress_ref::metrics::{compare, format_table, Comparison};
use stress_ref::observation::Observation;
use stress_ref::scene::Scene;
use stress_ref::world::World;

pub fn run(scene: &Scene) -> Observation {
    World::new(scene).run()
}

/// Run a scene and assert every analytic expectation passes.
pub fn assert_analytic(scene: &Scene) -> Vec<Comparison> {
    let obs = run(scene);
    let rows = compare(scene, &obs, &[]);
    let table = format_table(&scene.name, &rows);
    println!("{table}");
    for r in &rows {
        if r.reference_source == "analytic" {
            assert_eq!(r.pass, Some(true), "{}: {} failed\n{table}", scene.name, r.metric);
        }
    }
    rows
}

pub fn metric_number(scene: &Scene, obs: &Observation, name: &str) -> f64 {
    let m = scene.metrics.iter().find(|m| m.name == name).unwrap_or_else(|| panic!("no metric {name}"));
    stress_ref::metrics::evaluate(scene, m, obs).unwrap().as_f64().unwrap()
}

pub fn metric_value(scene: &Scene, obs: &Observation, name: &str) -> stress_ref::metrics::MetricValue {
    let m = scene.metrics.iter().find(|m| m.name == name).unwrap_or_else(|| panic!("no metric {name}"));
    stress_ref::metrics::evaluate(scene, m, obs).unwrap()
}

pub fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}
