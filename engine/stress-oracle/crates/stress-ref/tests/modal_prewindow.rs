//! The modal pre-window (`world::modal`, feature `modal`): its linear system is the
//! explicit solver's, its closed-form update is exact, and on a small wall it reaches the
//! explicit solver's verdicts. The full gate is `examples/modal_prewindow.rs`.
#![cfg(feature = "modal")]

use stress_ref::builders::*;
use stress_ref::math::Vec3;
use stress_ref::scene::*;
use stress_ref::solver::SolverEvent;
use stress_ref::world::modal::{self, ModalBasis, ModalConfig, Verdict};
use stress_ref::world::World;

/// A concrete wall of `nx x 2 x nz` 0.1 m chunks, its bottom row fixed (anchored) or
/// free, optionally struck by a 1000 kg steel ram at `speed` m/s.
fn wall(nx: usize, nz: usize, anchored: bool, ram: Option<f64>) -> Scene {
    let mut s = new_scene("modal_wall", "modal pre-window test wall");
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("steel".into(), elastic_steel());
    let counts = [nx, 2, nz];
    let mut chunks = grid(Vec3::new(-0.05 * nx as f64, 0.0, -0.1), counts, Vec3::splat(0.1), "concrete");
    if anchored {
        for i in 0..nx {
            for j in 0..2 {
                chunks[grid_index(counts, i, j, 0)].support = Support::Fixed;
            }
        }
    }
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("wall", chunks, bonds));
    if let Some(v) = ram {
        s.impactors.push(ImpactorDesc {
            name: "ram".into(),
            shape: ImpactorShape::Box { half_extents: [0.15, 0.25, 0.15] },
            mass: 1000.0,
            position: [0.0, -0.251, 0.05 * nz as f64],
            orientation: [1.0, 0.0, 0.0, 0.0],
            velocity: [0.0, v, 0.0],
            angular_velocity: [0.0; 3],
            material: "steel".into(),
            crush: None,
        });
    }
    if !anchored {
        s.gravity = [0.0; 3];
    }
    s.sim.methods = Methods::established();
    s.sim.duration = 0.01;
    s
}

/// Deterministic pseudo-random numbers in [-1, 1).
fn noise(n: usize, seed: u64) -> Vec<f64> {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((x >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn mat_vec(a: &faer::Mat<f64>, x: &[f64]) -> Vec<f64> {
    (0..a.nrows()).map(|i| (0..a.ncols()).map(|j| a[(i, j)] * x[j]).sum()).collect()
}

/// `K` and `C` are the explicit solver's bond forces: for any hidden displacement and
/// velocity, `-(K x + C v)` equals the sum of every bond's `chunk_loads` of its elastic
/// force and dashpot (intact joints, as `bond_response` applies them).
#[test]
fn assembly_is_the_explicit_bond_force() {
    for anchored in [true, false] {
        let world = World::new(&wall(4, 5, anchored, None));
        let asm = modal::assemble(&world, 0).unwrap();
        let n = asm.n;
        let (x, v) = (noise(n, 1).iter().map(|a| a * 1e-5).collect::<Vec<_>>(), noise(n, 2).iter().map(|a| a * 1e-2).collect::<Vec<_>>());
        let s = 0;
        let sv = &world.solver;
        let at = |c: usize, y: &[f64]| -> (Vec3, Vec3) {
            let mut out = [0.0; 6];
            for d in 0..6 {
                if let Some(i) = asm.dof[c][d] {
                    out[d] = y[i];
                }
            }
            (Vec3::new(out[0], out[1], out[2]), Vec3::new(out[3], out[4], out[5]))
        };
        let mut f = vec![0.0; n];
        for b in &sv.bonds[s] {
            let g = &b.geometry;
            let ((ua, ta), (ub, tb)) = (at(g.a, &x), at(g.b, &x));
            let ((va, wa), (vb, wb)) = (at(g.a, &v), at(g.b, &v));
            let q = g.kinematics(ua, ta, ub, tb).mul_elem(&b.stiffness.as_local()).add(&g.kinematics(va, wa, vb, wb).mul_elem(&b.damping));
            let (fa, ma, fb, mb) = g.chunk_loads(&q);
            for d in 0..6 {
                let comp = |f: Vec3, m: Vec3| if d < 3 { f.to_array()[d] } else { m.to_array()[d - 3] };
                if let Some(i) = asm.dof[g.a][d] {
                    f[i] += comp(fa, ma);
                }
                if let Some(i) = asm.dof[g.b][d] {
                    f[i] += comp(fb, mb);
                }
            }
        }
        let (kx, cv) = (mat_vec(&asm.k, &x), mat_vec(&asm.c, &v));
        let scale = f.iter().map(|a| a.abs()).fold(0.0, f64::max);
        let err = (0..n).map(|i| (f[i] + kx[i] + cv[i]).abs()).fold(0.0, f64::max);
        assert!(err <= 1e-12 * scale, "anchored {anchored}: |f + Kx + Cv| = {err:.3e} of {scale:.3e}");
    }
}

/// The closed-form modal update is exact for the linear system: an initial velocity and
/// a constant load, advanced in one step and in 7 unequal steps, match RK4 at a step of
/// 1/100 of the fastest period's radian (RK4's own error about 1e-10). The anchored wall
/// has non-proportional dashpots (bonds to the fixed row see twice the reduced mass), so
/// this exercises the state-space modes; the free wall checks the rigid modes' removal
/// with a self-equilibrated load and momentum-free velocity.
#[test]
fn modal_update_is_exact() {
    for anchored in [true, false] {
        let world = World::new(&wall(3, 4, anchored, None));
        let asm = modal::assemble(&world, 0).unwrap();
        let basis = ModalBasis::new(&world, 0).unwrap();
        let n = asm.n;
        assert!(basis.nonclassical > 1e-3 || !anchored, "anchored wall should have non-classical damping: {}", basis.nonclassical);
        let mut v0: Vec<f64> = noise(n, 3).iter().map(|a| a * 0.05).collect();
        let mut f: Vec<f64> = noise(n, 4).iter().map(|a| a * 1e4).collect();
        if !anchored {
            // Remove the rigid components: M-orthogonal projection on the rigid modes.
            let st = &world.solver.structures[0];
            let cl = &world.solver.clusters[0];
            let mut r = vec![vec![0.0; n]; 6];
            for &c in &cl.chunks {
                let x = st.chunks[c].center - cl.com;
                for k in 0..3 {
                    let e = Vec3::new((k == 0) as u8 as f64, (k == 1) as u8 as f64, (k == 2) as u8 as f64);
                    let rot_u = e.cross(x);
                    for d in 0..6 {
                        if let Some(i) = asm.dof[c][d] {
                            r[k][i] = if d < 3 { e.to_array()[d] } else { 0.0 };
                            r[3 + k][i] = if d < 3 { rot_u.to_array()[d] } else { e.to_array()[d - 3] };
                        }
                    }
                }
            }
            let mr: Vec<Vec<f64>> = r.iter().map(|ri| mat_vec(&asm.m, ri)).collect();
            let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
            let g = faer::Mat::<f64>::from_fn(6, 6, |a, b| dot(&r[a], &mr[b]));
            let ginv = g.partial_piv_lu();
            use faer::linalg::solvers::DenseSolveCore;
            let ginv = ginv.inverse();
            // v <- v - R G^-1 R^T M v ; f <- f - M R G^-1 R^T f.
            let rtmv: Vec<f64> = (0..6).map(|a| dot(&mr[a], &v0)).collect();
            let rtf: Vec<f64> = (0..6).map(|a| dot(&r[a], &f)).collect();
            for a in 0..6 {
                let (ca, cf) = ((0..6).map(|b| ginv[(a, b)] * rtmv[b]).sum::<f64>(), (0..6).map(|b| ginv[(a, b)] * rtf[b]).sum::<f64>());
                for i in 0..n {
                    v0[i] -= r[a][i] * ca;
                    f[i] -= mr[a][i] * cf;
                }
            }
        }
        // Reference: RK4 on M x'' = f - K x - C x'.
        let t_end = 2e-4;
        let w_max = basis.omega_range.1;
        let steps = (t_end * w_max * 100.0).ceil() as usize;
        let dt = t_end / steps as f64;
        let minv = {
            use faer::linalg::solvers::DenseSolveCore;
            asm.m.partial_piv_lu().inverse()
        };
        let accel = |x: &[f64], v: &[f64]| -> Vec<f64> {
            let (kx, cv) = (mat_vec(&asm.k, x), mat_vec(&asm.c, v));
            let r: Vec<f64> = (0..n).map(|i| f[i] - kx[i] - cv[i]).collect();
            mat_vec(&minv, &r)
        };
        let (mut x, mut v) = (vec![0.0; n], v0.clone());
        for _ in 0..steps {
            let a1 = accel(&x, &v);
            let (x2, v2): (Vec<f64>, Vec<f64>) = (0..n).map(|i| (x[i] + 0.5 * dt * v[i], v[i] + 0.5 * dt * a1[i])).unzip();
            let a2 = accel(&x2, &v2);
            let (x3, v3): (Vec<f64>, Vec<f64>) = (0..n).map(|i| (x[i] + 0.5 * dt * v2[i], v[i] + 0.5 * dt * a2[i])).unzip();
            let a3 = accel(&x3, &v3);
            let (x4, v4): (Vec<f64>, Vec<f64>) = (0..n).map(|i| (x[i] + dt * v3[i], v[i] + dt * a3[i])).unzip();
            let a4 = accel(&x4, &v4);
            for i in 0..n {
                x[i] += dt / 6.0 * (v[i] + 2.0 * v2[i] + 2.0 * v3[i] + v4[i]);
                v[i] += dt / 6.0 * (a1[i] + 2.0 * a2[i] + 2.0 * a3[i] + a4[i]);
            }
        }
        let mv0 = mat_vec(&asm.m, &v0);
        let g = basis.input(&f);
        for split in [vec![1.0], vec![0.05, 0.2, 0.1, 0.15, 0.3, 0.12, 0.08]] {
            let (mut cr, mut ci) = basis.input(&mv0);
            for frac in &split {
                basis.advance((&mut cr, &mut ci), (&g.0, &g.1), frac * t_end);
            }
            let (xm, vm) = basis.state((&cr, &ci));
            let rel = |a: &[f64], b: &[f64]| {
                let num = a.iter().zip(b).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
                num / b.iter().map(|p| p.abs()).fold(0.0, f64::max)
            };
            let (ex, ev) = (rel(&xm, &x), rel(&vm, &v));
            assert!(ex < 1e-9 && ev < 1e-9, "anchored {anchored}, {} steps: displacement error {ex:.2e}, velocity {ev:.2e}", split.len());
            eprintln!("anchored {anchored}, {} steps: displacement error {ex:.2e}, velocity {ev:.2e} (RK4 {steps} steps)", split.len());
        }
    }
}

/// First crack time of a run, if any.
fn first_crack(world: &World) -> Option<f64> {
    world.solver.events.iter().find_map(|e| match e {
        SolverEvent::Cracked { time, .. } => Some(*time),
        _ => None,
    })
}

/// On a small anchored wall the pre-window reaches the explicit solver's verdicts: a slow
/// ram damages nothing in either (certified or run out by the window), a fast ram's first
/// crack comes within 0.1 ms of the explicit one.
#[test]
fn small_wall_matches_explicit() {
    let mut basis: Option<ModalBasis> = None;
    for speed in [0.3, 10.0] {
        let scene = wall(8, 8, true, Some(speed));
        let mut truth = World::new(&scene);
        truth.run();
        let mut world = World::new(&scene);
        if basis.is_none() {
            basis = Some(ModalBasis::new(&world, 0).unwrap());
        }
        let report = modal::run(&mut world, basis.as_ref().unwrap(), &ModalConfig::default()).unwrap();
        let frames = (scene.sim.duration / scene.sim.frame_dt).round() as u64;
        while world.frame < frames {
            world.step_frame();
        }
        let (t_truth, t_modal) = (first_crack(&truth), first_crack(&world));
        eprintln!("speed {speed}: truth crack {t_truth:?}, modal {t_modal:?}, verdict {:?}, {} steps", report.verdict, report.steps);
        assert_eq!(t_truth.is_some(), report.verdict.damages(), "speed {speed}: verdict differs");
        if let (Some(a), Some(b)) = (t_truth, t_modal) {
            assert!((a - b).abs() <= 1e-4, "speed {speed}: first crack {a} vs {b}");
        }
        if let Verdict::Certified { bound, .. } = report.verdict {
            assert!(bound <= 0.95);
        }
    }
}

/// A light, stiff hit (a 2 cm steel cube at 5 m/s) damages nothing: the window runs to
/// the end with the explicit solver's peak failure index (within 1%) and impactor speed.
#[test]
fn light_debris_peak_matches_explicit() {
    let mut scene = wall(8, 8, true, Some(5.0));
    let h = 0.01;
    scene.impactors[0].shape = ImpactorShape::Box { half_extents: [h, h, h] };
    scene.impactors[0].mass = 7850.0 * 8.0 * h * h * h;
    scene.impactors[0].position = [0.0, -h - 0.001, 0.4];
    scene.sim.duration = 0.004;
    scene.sim.frame_dt = 0.001;
    let mut truth = World::new(&scene);
    truth.run();
    let mut world = World::new(&scene);
    let basis = ModalBasis::new(&world, 0).unwrap();
    let r = modal::run(&mut world, &basis, &ModalConfig { certify: false, ..ModalConfig::default() }).unwrap();
    let (a, b) = (truth.solver.max_utilization, r.max_utilization);
    eprintln!("explicit peak {a:.4}, modal {b:.4}; impactor {:.4} vs {:.4} m/s", truth.impactors[0].velocity.y, world.impactors[0].velocity.y);
    assert!(first_crack(&truth).is_none() && !r.verdict.damages());
    assert!((a - b).abs() <= 0.01 * a, "peak failure index {a} vs {b}");
    assert!((truth.impactors[0].velocity.y - world.impactors[0].velocity.y).abs() <= 0.01);
}

/// The basis is accurate: eigen residuals and the explicit inverse at round-off; the
/// anchored wall's dashpots are non-classical (bonds to the fixed row see twice the
/// reduced mass) and the free wall's classical (equal chunks); an instantaneous impulse
/// `M v` gives back the velocity `v` (anchored: no rigid part to drop) at zero
/// displacement.
#[test]
fn basis_is_accurate() {
    for anchored in [true, false] {
        let world = World::new(&wall(3, 4, anchored, None));
        let asm = modal::assemble(&world, 0).unwrap();
        let basis = ModalBasis::new(&world, 0).unwrap();
        assert!(basis.residuals.iter().all(|&r| r < 1e-12), "anchored {anchored}: residuals {:?}", basis.residuals);
        if anchored {
            assert!(basis.nonclassical > 1e-2, "{}", basis.nonclassical);
        } else {
            assert!(basis.nonclassical < 1e-12 && basis.rigid_dropped == 6, "{}", basis.nonclassical);
        }
        let n = asm.n;
        let v0: Vec<f64> = noise(n, 3).iter().map(|a| a * 0.05).collect();
        let (cr, ci) = basis.input(&mat_vec(&asm.m, &v0));
        let (x, v) = basis.state((&cr, &ci));
        let xm = x.iter().map(|a| a.abs()).fold(0.0, f64::max);
        assert!(xm < 1e-15, "anchored {anchored}: x(0) {xm:.3e}");
        if anchored {
            let dv = v.iter().zip(&v0).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(dv < 1e-12, "|v - v0| {dv:.3e}");
        }
    }
}
