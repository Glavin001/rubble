//! Moreau's impact law over continuous contact's response (AGENTS.md, "Contact never stands
//! in for metal"; `docs/continuous-response-parallel-design.md`), the CPU twin of the kernel's
//! `ccRestitute`.

use super::dat::DatStats;
use crate::explicit_beams::ExplicitBeams;
use glam::Vec3;

/// The impact law's restitution e by default: 0 (inelastic). At e = 1 the law as written here
/// (the whole response's change given back each step) CREATES energy: a pressed contact
/// bounces back every step and its beams push it in again; two elastic bars head on gained
/// 403 J on 275 J of kinetic energy (`tests/contact_known_solutions.rs`). The design's per-row
/// share of closing at the step's start (σ, `continuous-response-parallel-design.md`) is what
/// keeps it to true impacts; until it exists e stays 0. The GPU's tables carry the same value.
pub const CONTACT_RESTITUTION: f32 = 0.0;

/// The restitution guard's sums are striped over this many lanes in a fixed order (the
/// kernel's `ccRestitute`, whatever its workgroup's width), then added in lane order.
pub const RESTITUTION_STRIPES: usize = 64;

/// Moreau's impact law over the response (AGENTS.md, "Contact never stands in for metal"):
/// the response's velocity change Δ = v_p − v⁻ (from equal and opposite impulses: momentum
/// kept) is given back e times, v⁺ = v_p + θ·Δ, θ = e, or less where that would raise the
/// vehicles' kinetic energy above its value before the pass: θ the larger root of
/// Σm|v_p + θΔ|² = Σm|v⁻|² (a quadratic), so no energy is created. f32 in the kernel's order
/// (`ccRestitute`): per stripe of [`RESTITUTION_STRIPES`] lanes over the vehicles' nodes in
/// order, then the stripes in order.
pub(super) fn restitute(vehicles: &[ExplicitBeams], vp: &mut [Vec<Vec3>], e: f32, near: &[(usize, usize)], st: &mut DatStats) {
    if e <= 0.0 {
        return;
    }
    // The guard's groups: vehicles joined by the listing's near test, each labelled by its
    // smallest member (the kernel's labels, lane 0 over its group's members in order).
    let nv = vehicles.len();
    let mut label: Vec<usize> = (0..nv).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for &(a, b) in near {
            let m = label[a].min(label[b]);
            if label[a] != m || label[b] != m {
                (label[a], label[b]) = (m, m);
                changed = true;
            }
        }
    }
    let dot = |a: Vec3, b: Vec3| (a.x * b.x + a.y * b.y) + a.z * b.z;
    for c in (0..nv).filter(|&c| label[c] == c) {
        let members: Vec<usize> = (0..nv).filter(|&v| label[v] == c).collect();
        let mut part = [[0.0f32; 4]; RESTITUTION_STRIPES];
        for (l, s) in part.iter_mut().enumerate() {
            for &v in &members {
                let veh = &vehicles[v];
                for i in (l..veh.nodes.len()).step_by(RESTITUTION_STRIPES) {
                    let (m, vm, p) = (veh.nodes[i].mass, veh.nodes[i].vel, vp[v][i]);
                    let d = p - vm;
                    s[0] += m * dot(p, d);
                    s[1] += m * dot(d, d);
                    s[2] += m * dot(vm, vm);
                    s[3] += m * dot(p, p);
                }
            }
        }
        let mut t = [0.0f32; 4];
        for s in &part {
            for j in 0..4 {
                t[j] += s[j];
            }
        }
        let [a, b, k0, kp] = t;
        let mut theta = 0.0f32;
        if b > 0.0 {
            // kp + 2θa + θ²b ≤ k0: the larger root.
            let disc = (a * a - b * (kp - k0)).max(0.0);
            let most = (-a + disc.sqrt()) / b;
            theta = e.min(most.max(0.0));
            if theta < e {
                st.guarded += 1;
            }
        }
        for &v in &members {
            for (i, n) in vehicles[v].nodes.iter().enumerate() {
                let d = vp[v][i] - n.vel;
                vp[v][i] += d * theta;
            }
        }
    }
}
