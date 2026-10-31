//! `physics/mpm_material.slang` on the GPU (through `mpm_material_probe`) against
//! its engine-terrain CPU mirror: the SVD, the material update (the sand return
//! maps, cohesive Mohr–Coulomb soil and rock damage) and the Kirchhoff stress agree
//! within f32 tolerance on random deformation gradients, and the input checks agree
//! on NaN and ±inf.

use engine_terrain::material::{
    hencky_kirchhoff, principal_kirchhoff, rock_damage, sand_input_valid, sand_project,
    sand_project_mohr_coulomb, soil_project_mohr_coulomb, svd3, svd_input_valid, Svd3,
    YieldSurface,
};
use engine_terrain::{RockMaterial, SandMaterial};
use engine_webgpu::{default_device, MPM_MATERIAL_PROBE_WGSL};
use glam::{Mat3, Quat, Vec3};

/// `MpmProbeCase`: F's columns, (friction or f_t, cohesion word or E·G_f/f_t,
/// lambda, mu), (logJp, model, damage, dx).
type ProbeCase = [[f32; 4]; 5];
/// `MpmProbeResult`: U, V, sigma (+ valid), (sigmaOut, logJpOut), τ.
type ProbeResult = [[f32; 4]; 11];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next() >> 40) as f32 / (1u64 << 24) as f32)
    }

    fn rotation(&mut self) -> Mat3 {
        let q = Quat::from_xyzw(
            self.range(-1.0, 1.0),
            self.range(-1.0, 1.0),
            self.range(-1.0, 1.0),
            self.range(-1.0, 1.0),
        );
        Mat3::from_quat(q.normalize())
    }

    fn matrix(&mut self, lo: f32, hi: f32) -> Mat3 {
        Mat3::from_cols_array(&std::array::from_fn(|_| self.range(lo, hi)))
    }
}

/// What a case runs after the SVD.
#[derive(Clone, Copy)]
enum Model {
    /// Sand or soil: Drucker–Prager (cohesion as its apex stretch), dry
    /// Mohr–Coulomb, or cohesive Mohr–Coulomb (stress space), as the engine
    /// picks them (`terrain_set_material`).
    Sand(SandMaterial),
    /// Rock with its damage before the update and the cell size.
    Rock {
        rock: RockMaterial,
        damage: f32,
        dx: f32,
    },
}

impl Model {
    /// The probe's material and model words.
    fn words(&self) -> ([f32; 4], f32, f32, f32) {
        match *self {
            Model::Sand(m) => {
                let (code, friction, cohesion) = match m.yield_surface {
                    YieldSurface::DruckerPrager => (0.0, m.alpha(), m.apex_strain()),
                    YieldSurface::MohrCoulomb if m.cohesion == 0.0 => {
                        (1.0, m.friction_angle.sin(), 0.0)
                    }
                    YieldSurface::MohrCoulomb => {
                        (2.0, m.friction_angle.sin(), m.cohesion_cos_phi())
                    }
                };
                ([friction, cohesion, m.lambda(), m.mu()], code, 0.0, 0.0)
            }
            Model::Rock { rock, damage, dx } => (
                [
                    rock.tensile_strength,
                    rock.band_stress(),
                    rock.lambda(),
                    rock.mu(),
                ],
                3.0,
                damage,
                dx,
            ),
        }
    }

    fn lame(&self) -> (f32, f32) {
        match self {
            Model::Sand(m) => (m.lambda(), m.mu()),
            Model::Rock { rock, .. } => (rock.lambda(), rock.mu()),
        }
    }
}

struct Case {
    f: Mat3,
    model: Model,
    log_jp: f32,
}

fn cases() -> Vec<Case> {
    let mut rng = Rng(23);
    let mut out = Vec::new();
    for i in 0..4096u32 {
        let f = match i % 8 {
            0 => rng.matrix(-2.0, 2.0),
            // MPM-like: a rotation of a small strain, compressed or stretched.
            1 => rng.rotation() * (Mat3::IDENTITY + rng.matrix(-0.05, 0.05)),
            2 => {
                rng.rotation() * Mat3::from_diagonal(Vec3::splat(rng.range(0.7, 1.0)))
                    + rng.matrix(-0.02, 0.02)
            }
            3 => {
                rng.rotation() * Mat3::from_diagonal(Vec3::splat(rng.range(1.0, 1.3)))
                    + rng.matrix(-0.02, 0.02)
            }
            // Near-singular, and every other one nearly rank 1.
            4 => {
                // σ₂ ≤ √ε·σ₁: FᵀF cannot resolve the small plane; the trailing
                // 2×2 SVD must.
                let s1 = if i % 16 == 4 {
                    10f32.powf(rng.range(-7.0, -3.5))
                } else {
                    rng.range(0.05, 1.0)
                };
                let d = Vec3::new(rng.range(0.5, 2.0), s1, s1.min(1e-6));
                rng.rotation() * Mat3::from_diagonal(d) * rng.rotation().transpose()
            }
            // det < 0.
            5 => -rng.rotation() * (Mat3::IDENTITY + rng.matrix(-0.1, 0.1)),
            // Repeated singular values.
            6 => {
                let a = rng.range(0.5, 1.5);
                let d = Vec3::new(a, a, rng.range(0.5, 1.5));
                rng.rotation() * Mat3::from_diagonal(d) * rng.rotation().transpose()
            }
            _ => match i % 64 {
                7 => Mat3::IDENTITY,
                15 => Mat3::ZERO,
                _ => rng.matrix(-2.0, 2.0) * 1e3,
            },
        };
        let mut mat = SandMaterial {
            youngs_modulus: rng.range(1e6, 1e8),
            poisson_ratio: rng.range(0.2, 0.4),
            friction_angle: rng.range(0.0, 45.0).to_radians(),
            yield_surface: if i % 2 == 0 {
                YieldSurface::MohrCoulomb
            } else {
                YieldSurface::DruckerPrager
            },
            ..SandMaterial::default()
        };
        // Cohesive (soil) cases: an apex stretch of up to 0.05 (as the log-strain
        // offsets before), c = apex·(3λ + 2μ)·tan φ.
        let apex = rng.range(0.0, 0.05);
        if i % 5 == 0 || i % 5 == 2 {
            mat.friction_angle = mat.friction_angle.max(1f32.to_radians());
            mat.cohesion = apex * (3.0 * mat.lambda() + 2.0 * mat.mu()) * mat.friction_angle.tan();
        }
        let log_jp = if i % 3 == 0 { rng.range(0.0, 0.2) } else { 0.0 };
        out.push(Case {
            f,
            model: Model::Sand(mat),
            log_jp,
        });
    }
    // Rock: strains around its tensile strain (f_t/E ≈ 1e-4 to 1e-3), well past
    // it, and the general families above; intact, partly and fully damaged.
    for i in 0..2048u32 {
        let rock = RockMaterial {
            youngs_modulus: rng.range(5e9, 7e10),
            poisson_ratio: rng.range(0.15, 0.3),
            tensile_strength: rng.range(1e6, 1.5e7),
            fracture_energy: rng.range(30.0, 200.0),
            ..RockMaterial::default()
        };
        let ft_strain = rock.tensile_strength / rock.youngs_modulus;
        let f = match i % 4 {
            0 | 1 => {
                let k = if i % 4 == 0 { 3.0 } else { 30.0 };
                let d = Vec3::new(
                    (k * ft_strain * rng.range(-1.0, 1.0)).exp(),
                    (k * ft_strain * rng.range(-1.0, 1.0)).exp(),
                    (k * ft_strain * rng.range(-1.0, 1.0)).exp(),
                );
                rng.rotation() * Mat3::from_diagonal(d) * rng.rotation().transpose()
            }
            2 => rng.rotation() * (Mat3::IDENTITY + rng.matrix(-0.05, 0.05)),
            _ => rng.matrix(-2.0, 2.0),
        };
        let damage = match i % 3 {
            0 => 0.0,
            1 => rng.range(0.0, 1.0),
            _ => 1.0,
        };
        let dx = rng.range(0.05, 0.95) * rock.max_cell_size();
        out.push(Case {
            f,
            model: Model::Rock { rock, damage, dx },
            log_jp: 0.0,
        });
    }
    // Sand at rest: a rotation by microradians (repeated singular values; the
    // trailing 2×2 block's off-diagonal squared underflows), and a bed particle's
    // F (rows, bit for bit) whose stress once came out NaN on the GPU.
    for k in 0..64 {
        let axis = Vec3::new(
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        );
        let angle = 10f32.powf(rng.range(-7.0, -4.0));
        out.push(Case {
            f: Mat3::from_axis_angle(axis.normalize_or(Vec3::X), angle),
            model: Model::Sand(SandMaterial::default()),
            log_jp: if k % 2 == 0 { 3.7e-5 } else { 0.0 },
        });
    }
    let rows: [[u32; 3]; 3] = [
        [0x3f800000, 0x368594c2, 0x36d338d8],
        [0xb68594a0, 0x3f800001, 0xb7cef85d],
        [0xb6d2e165, 0x37cef067, 0x3f7fffff],
    ];
    out.push(Case {
        f: Mat3::from_cols_array_2d(&rows.map(|r| r.map(f32::from_bits))).transpose(),
        model: Model::Sand(SandMaterial::default()),
        log_jp: f32::from_bits(0x381bc050),
    });
    // Input checks: NaN and ±inf in F.
    for bad in [f32::NAN, f32::INFINITY, -f32::INFINITY] {
        let mut f = Mat3::IDENTITY;
        f.z_axis.y = bad;
        out.push(Case {
            f,
            model: Model::Sand(SandMaterial::default()),
            log_jp: 0.0,
        });
    }
    out
}

fn run_probe(cases: &[ProbeCase]) -> Vec<ProbeResult> {
    use wgpu::util::DeviceExt;
    let (_, device, queue) = default_device().expect("wgpu");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mpm_material_probe"),
        source: wgpu::ShaderSource::Wgsl(MPM_MATERIAL_PROBE_WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mpm_material_probe"),
        layout: None,
        module: &module,
        entry_point: Some("mpm_material_probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let storage = wgpu::BufferUsages::STORAGE;
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("probe-cases"),
        contents: bytemuck::cast_slice(cases),
        usage: storage,
    });
    let count = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("probe-count"),
        contents: bytemuck::cast_slice(&[cases.len() as u32]),
        usage: storage,
    });
    let size = (cases.len() * std::mem::size_of::<ProbeResult>()) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe-results"),
        size,
        usage: storage | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe-read"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: count.as_entire_binding(),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((cases.len() as u32).div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&output, 0, &staging, 0, size);
    queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map").expect("map");
    let data = slice.get_mapped_range().expect("mapped range");
    let results = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    results
}

fn v3(a: [f32; 4]) -> Vec3 {
    Vec3::new(a[0], a[1], a[2])
}

fn frob(m: Mat3) -> f32 {
    m.to_cols_array().iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[test]
fn gpu_material_matches_the_cpu_mirror() {
    let cases = cases();
    let packed: Vec<ProbeCase> = cases
        .iter()
        .map(|c| {
            let (material, code, damage, dx) = c.model.words();
            [
                c.f.x_axis.extend(0.0).to_array(),
                c.f.y_axis.extend(0.0).to_array(),
                c.f.z_axis.extend(0.0).to_array(),
                material,
                [c.log_jp, code, damage, dx],
            ]
        })
        .collect();
    let results = run_probe(&packed);
    assert_eq!(results.len(), cases.len());

    let (mut d_sigma, mut d_uv, mut d_proj, mut d_tau, mut gpu_rec) =
        (0f32, 0f32, 0f32, 0f32, 0f32);
    let (mut d_damage, mut soil_cases, mut rock_cases, mut rock_damaged) = (0f32, 0, 0, 0);
    let mut uv_compared = 0;
    for (c, r) in cases.iter().zip(&results) {
        let cpu_valid = svd_input_valid(c.f);
        let cpu = svd3(c.f);
        let gpu_valid = r[6][3] == 1.0;
        let cpu_valid = cpu_valid && sand_input_valid(cpu.sigma, c.log_jp);
        assert_eq!(gpu_valid, cpu_valid, "input check differs for {:?}", c.f);
        if !cpu_valid {
            continue;
        }
        let gpu = Svd3 {
            u: Mat3::from_cols(v3(r[0]), v3(r[1]), v3(r[2])),
            sigma: v3(r[6]),
            v: Mat3::from_cols(v3(r[3]), v3(r[4]), v3(r[5])),
        };
        let scale = frob(c.f).max(1e-30);

        // The GPU's own decomposition reconstructs F with rotations.
        let rec = gpu.u * Mat3::from_diagonal(gpu.sigma) * gpu.v.transpose();
        let rel = if frob(c.f) == 0.0 {
            frob(rec)
        } else {
            frob(rec - c.f) / scale
        };
        gpu_rec = gpu_rec.max(rel);
        assert!(rel < 1e-5, "GPU ‖UΣVᵀ − F‖/‖F‖ = {rel} for {:?}", c.f);
        for m in [gpu.u, gpu.v] {
            assert!((m.determinant() - 1.0).abs() < 1e-5, "{m:?}");
        }

        let ds = (gpu.sigma - cpu.sigma).abs().max_element() / scale;
        d_sigma = d_sigma.max(ds);
        assert!(ds < 1e-5, "sigma {:?} vs {:?}", gpu.sigma, cpu.sigma);

        // U and V are unique (with the sign rule) only for distinct singular values.
        let s = cpu.sigma.abs();
        if (s.x - s.y).min(s.y - s.z) > 0.05 * s.x {
            uv_compared += 1;
            let d = frob(gpu.u - cpu.u).max(frob(gpu.v - cpu.v));
            d_uv = d_uv.max(d);
            assert!(d < 1e-4, "U/V differ by {d} for {:?}", c.f);
        }

        // Each later stage is compared on the GPU's own inputs, so a difference
        // is that stage's alone (ln σ amplifies SVD rounding where σ is small).
        let (lambda, mu) = c.model.lame();
        let gpu_out = v3(r[7]);
        let gpu_tau = Mat3::from_cols(v3(r[8]), v3(r[9]), v3(r[10]));
        let tau = match c.model {
            Model::Rock { rock, damage, dx } => {
                let p = rock_damage(
                    gpu.sigma,
                    damage,
                    lambda,
                    mu,
                    rock.tensile_strength,
                    rock.band_stress(),
                    dx,
                );
                let dd = (r[7][3] - p.damage).abs();
                d_damage = d_damage.max(dd);
                rock_cases += 1;
                rock_damaged += usize::from(p.damage > damage);
                assert!(dd < 5e-5, "damage {} vs {p:?}", r[7][3]);
                principal_kirchhoff(gpu.u, p.tau)
            }
            Model::Sand(m) => {
                let p = match m.yield_surface {
                    YieldSurface::DruckerPrager => {
                        sand_project(gpu.sigma, c.log_jp, m.alpha(), m.apex_strain(), lambda, mu)
                    }
                    YieldSurface::MohrCoulomb if m.cohesion == 0.0 => sand_project_mohr_coulomb(
                        gpu.sigma,
                        c.log_jp,
                        m.friction_angle.sin(),
                        0.0,
                        lambda,
                        mu,
                    ),
                    YieldSurface::MohrCoulomb => {
                        soil_cases += 1;
                        soil_project_mohr_coulomb(
                            gpu.sigma,
                            c.log_jp,
                            m.friction_angle.sin(),
                            m.cohesion_cos_phi(),
                            lambda,
                            mu,
                        )
                    }
                };
                let dp = ((gpu_out - p.sigma).abs() / p.sigma)
                    .max_element()
                    .max((r[7][3] - p.log_jp).abs());
                d_proj = d_proj.max(dp);
                assert!(dp < 1e-5, "projection {:?} vs {p:?}", r[7]);
                hencky_kirchhoff(gpu.u, gpu_out, lambda, mu)
            }
        };

        // τ compared as a strain: |Δτ| / (3λ + 2μ), relative to the strain when
        // it exceeds 1.
        let modulus = 3.0 * lambda + 2.0 * mu;
        let dt = frob(gpu_tau - tau) / modulus.max(frob(tau));
        d_tau = d_tau.max(dt);
        assert!(
            dt < 1e-5,
            "τ differs by strain {dt}: {gpu_tau:?} vs {tau:?}"
        );
    }
    eprintln!(
        "mpm_material GPU vs CPU over {} cases: sigma {d_sigma:.2e} (rel ‖F‖), U/V {d_uv:.2e} \
         ({uv_compared} distinct), projection {d_proj:.2e} (rel; {soil_cases} cohesive), \
         damage {d_damage:.2e} ({rock_cases} rock, {rock_damaged} damaged), τ {d_tau:.2e} \
         (strain); GPU reconstruction {gpu_rec:.2e}",
        cases.len()
    );
    assert!(
        soil_cases > 500 && rock_damaged > 500,
        "too few soil or damaging rock cases"
    );
}
