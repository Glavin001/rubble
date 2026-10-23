//! CPU reference of the particle and beam solver: the same VBD block solve,
//! beam model, penalty ramp and post-step plasticity as `soft_step.slang`, with
//! planes as the only obstacles. Used for unit tests, the convergence check and
//! as a regression oracle for the GPU kernels (contact-free scenes match closely).

use crate::color::color_graph;
use crate::desc::SoftVehicleDesc;
use crate::layout::SoftSettings;
use crate::model::*;
use crate::pack::{pack_soft, Placement, SpawnOptions};
use glam::{Mat3, Quat, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub normal: Vec3,
    pub offset: f32,
    pub friction: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StepReport {
    pub yielded: u32,
    pub broke: u32,
}

#[derive(Clone, Debug)]
pub struct CpuSoft {
    pub x: Vec<Vec3>,
    pub v: Vec<Vec3>,
    pub mass: Vec<f32>,
    pub fixed: Vec<bool>,
    pub collide: Vec<bool>,
    pub beams: Vec<Beam>,
    pub adjacency: Vec<Vec<u32>>,
    pub colors: Vec<u32>,
    pub color_count: u32,
    pub planes: Vec<Plane>,
    pub radius: f32,
    pub settings: SoftSettings,
    pub h: f32,
    pub iterations: u32,
    pub gravity: Vec3,
    /// Contact penalty (N/m) for node–plane contacts.
    pub contact_stiffness: f32,
    pub steps: u32,
    /// Chebyshev spectral radius estimate ρ (0 = off).
    pub chebyshev_rho: f32,
    /// Last step's velocities (adaptive initial guess).
    pub v_prev: Vec<Vec3>,
}

impl CpuSoft {
    /// Particles and beams of `desc` (no rigid core), placed in the world.
    pub fn from_desc(desc: &SoftVehicleDesc, position: Vec3, velocity: Vec3) -> Self {
        let opts = SpawnOptions {
            rigid_core: false,
            ..Default::default()
        };
        let p = pack_soft(
            desc,
            Placement {
                position,
                rotation: Quat::IDENTITY,
            },
            &opts,
            0,
            -1,
        );
        let edges: Vec<(u32, u32)> = p.beams.iter().map(|b| (b.a, b.b)).collect();
        let (colors, color_count) = color_graph(desc.nodes.len(), &edges);
        Self {
            x: p.bodies
                .iter()
                .map(|b| Vec3::from_array(b.position))
                .collect(),
            v: desc
                .nodes
                .iter()
                .map(|n| if n.fixed { Vec3::ZERO } else { velocity })
                .collect(),
            mass: p.masses,
            fixed: desc.nodes.iter().map(|n| n.fixed).collect(),
            collide: desc.nodes.iter().map(|n| n.collision).collect(),
            beams: p.beams,
            adjacency: p.adjacency,
            colors,
            color_count,
            planes: Vec::new(),
            radius: opts.node_radius,
            settings: SoftSettings::default(),
            h: 1.0 / 120.0,
            iterations: 10,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            contact_stiffness: 1.0e7,
            steps: 0,
            chebyshev_rho: 0.0,
            v_prev: vec![Vec3::ZERO; desc.nodes.len()],
        }
    }

    pub fn step(&mut self) -> StepReport {
        let h = self.h;
        let n = self.x.len();
        let x0 = self.x.clone();
        // Inertial target y = x + h v + h² g. Initial guess (as soft_begin):
        // VBD's adaptive x + h v + h² g ã, ã = last step's acceleration along g.
        let y: Vec<Vec3> = (0..n)
            .map(|i| {
                if self.fixed[i] {
                    x0[i]
                } else {
                    x0[i] + self.v[i] * h + self.gravity * h * h
                }
            })
            .collect();
        let g2 = self.gravity.length_squared().max(1e-6);
        for i in 0..n {
            if self.fixed[i] {
                self.x[i] = x0[i];
                continue;
            }
            let w = ((self.v[i] - self.v_prev[i]) / h).dot(self.gravity) / g2;
            self.x[i] = x0[i] + self.v[i] * h + self.gravity * (h * h * w.clamp(0.0, 1.0));
        }
        self.v_prev.clone_from(&self.v);
        let order: Vec<Vec<usize>> = {
            let mut o = vec![Vec::new(); self.color_count as usize];
            for i in 0..n {
                if !self.fixed[i] {
                    o[self.colors[i] as usize].push(i);
                }
            }
            o
        };
        let mut prev2 = self.x.clone();
        let mut prev1 = self.x.clone();
        let mut omega = 1.0f32;
        let rho2 = self.chebyshev_rho * self.chebyshev_rho;
        for it in 0..self.iterations {
            for list in &order {
                for &i in list {
                    self.solve_particle(i, &x0, &y);
                }
            }
            if rho2 > 0.0 {
                // Chebyshev semi-iterative acceleration (VBD §3.6), off for particles in contact.
                omega = match it {
                    0 => 1.0,
                    1 => 2.0 / (2.0 - rho2),
                    _ => 4.0 / (4.0 - rho2 * omega),
                };
                for i in 0..n {
                    let in_contact = self.collide[i]
                        && self
                            .planes
                            .iter()
                            .any(|p| p.normal.dot(self.x[i]) - p.offset - self.radius < 0.0);
                    if !self.fixed[i] && it > 0 && !in_contact {
                        self.x[i] = prev2[i] + (self.x[i] - prev2[i]) * omega;
                    }
                }
                prev2.clone_from(&prev1);
                prev1.clone_from(&self.x);
            }
            // Penalty ramp (soft beams): k ← min(k*, k + β|C|).
            for b in &mut self.beams {
                if b.broken() {
                    continue;
                }
                let len = (self.x[b.a as usize] - self.x[b.b as usize]).length();
                let c = (len - b.rest).abs();
                b.penalty = (b.penalty + self.settings.beta * c).min(b.spring);
            }
        }
        let mut rep = StepReport::default();
        let mut groups = Vec::new();
        for b in &mut self.beams {
            if b.broken() {
                continue;
            }
            let len = (self.x[b.a as usize] - self.x[b.b as usize]).length();
            let len0 = (x0[b.a as usize] - x0[b.b as usize]).length();
            let f = readout_force(b, len, len0, h, self.settings.damping_scale);
            let o = plastic_update(
                b,
                len,
                f,
                self.settings.min_length,
                self.settings.plasticity,
                self.settings.breaking,
            );
            rep.yielded += u32::from(o.yielded);
            if o.broke {
                rep.broke += 1;
                if b.group != 0 {
                    groups.push(b.group);
                }
            }
        }
        if !groups.is_empty() {
            for b in &mut self.beams {
                if b.group != 0 && !b.broken() && groups.contains(&b.group) {
                    b.flags |= crate::layout::BEAM_BROKEN;
                    rep.broke += 1;
                }
            }
        }
        for i in 0..n {
            self.v[i] = if self.fixed[i] {
                Vec3::ZERO
            } else {
                (self.x[i] - x0[i]) / h
            };
        }
        self.steps += 1;
        rep
    }

    fn solve_particle(&mut self, i: usize, x0: &[Vec3], y: &[Vec3]) {
        let h = self.h;
        let m = self.mass[i];
        let xi = self.x[i];
        let mut hess = Mat3::from_diagonal(Vec3::splat(m / (h * h)));
        let mut grad = (xi - y[i]) * (m / (h * h));
        for &e in &self.adjacency[i] {
            let b = &self.beams[e as usize];
            if b.broken() {
                continue;
            }
            // The block is symmetric in the ends: evaluate with this particle first.
            let other = if b.a as usize == i {
                b.b as usize
            } else {
                b.a as usize
            };
            let (g, hb) = beam_block_plastic(
                b,
                xi,
                self.x[other],
                x0[i],
                x0[other],
                h,
                self.settings.damping_scale,
                self.settings.plasticity.then_some(1.0),
            );
            grad += g;
            hess += hb;
        }
        if self.collide[i] {
            for p in &self.planes {
                let c = p.normal.dot(xi) - p.offset - self.radius;
                if c < 0.0 {
                    let k = self.contact_stiffness;
                    let fn_ = -k * c;
                    grad += p.normal * (k * c);
                    hess += Mat3::from_cols(
                        p.normal * p.normal.x,
                        p.normal * p.normal.y,
                        p.normal * p.normal.z,
                    ) * k;
                    // Lagged Coulomb friction toward the step-start tangential position.
                    let slip = (xi - x0[i]) - p.normal * p.normal.dot(xi - x0[i]);
                    let ks = k.min(m / (h * h) * 10.0);
                    let ft = slip * ks;
                    let cap = p.friction * fn_;
                    let (ft, kt) = if ft.length() > cap && slip.length() > 1e-9 {
                        (slip.normalize() * cap, cap / slip.length())
                    } else {
                        (ft, ks)
                    };
                    grad += ft;
                    let tt = Mat3::IDENTITY
                        - Mat3::from_cols(
                            p.normal * p.normal.x,
                            p.normal * p.normal.y,
                            p.normal * p.normal.z,
                        );
                    hess += tt * kt;
                }
            }
        }
        let det = hess.determinant();
        if det.abs() < 1e-12 {
            return;
        }
        let dx = -(hess.inverse() * grad);
        if dx.is_finite() {
            self.x[i] = xi + dx;
        }
    }

    pub fn kinetic_energy(&self) -> f32 {
        self.v
            .iter()
            .zip(&self.mass)
            .map(|(v, m)| 0.5 * m * v.length_squared())
            .sum()
    }

    pub fn broken_beams(&self) -> usize {
        self.beams.iter().filter(|b| b.broken()).count()
    }

    pub fn yielded_beams(&self) -> usize {
        self.beams
            .iter()
            .filter(|b| b.flags & crate::layout::BEAM_YIELDED != 0)
            .count()
    }
}
