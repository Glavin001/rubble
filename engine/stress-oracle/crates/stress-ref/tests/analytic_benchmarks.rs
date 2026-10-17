//! Benchmarks 1-4 of the spec against closed-form solutions.

mod common;

use stress_ref::builders;

#[test]
fn b1_bond_tension_failure_force_and_fracture_energy() {
    common::assert_analytic(&builders::bond_tension());
}

#[test]
fn b1_bond_shear_mohr_coulomb_failure_force() {
    common::assert_analytic(&builders::bond_shear());
}

#[test]
fn b2_cantilever_matches_euler_bernoulli() {
    for n in [10, 20, 40] {
        common::assert_analytic(&builders::cantilever(n, "cantilever"));
    }
}

#[test]
fn b3_bar_wave_speed_and_reflection_true_and_scaled_stiffness() {
    common::assert_analytic(&builders::bar_wave(1.0, "bar"));
    common::assert_analytic(&builders::bar_wave(0.01, "bar_scaled"));
}

#[test]
fn b4_sudden_support_loss_doubles_the_overload_gradual_does_not() {
    common::assert_analytic(&builders::support_loss(0.0, "sudden"));
    common::assert_analytic(&builders::support_loss(0.1, "gradual"));
}
