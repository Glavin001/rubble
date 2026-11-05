# Vehicle test strategy: small analytic cases and physical invariants

Added 2026-11-05 (branch `feat/vehicle-invariants`).

**Why.** Large end-to-end crash tests let subtle errors cascade, and they cannot say *why* a crash looks wrong. This strategy adds two layers below them:

1. **Small analytic cases.** Each checks one feature against a closed form, on the CPU and the GPU.
2. **Physical invariants of full cars.** Each is a reusable checker, with a threshold taken from measured data or exact physics. Anything else is labelled a **Choice**.

The user's two complaints were these:
- In a frontal or car-to-car hit, the boot at the far end deforms or moves a lot.
- Panels ripple afterwards.

Layer 2 now measures both, and layer 1 and the panel diagnostics explain them.

| File | Contents |
|---|---|
| `engine/crates/engine-softbody/tests/invariants_small.rs` | Layer 1 on the CPU |
| `engine/crates/engine-softbody/tests/invariants_common/mod.rs` | Shared bodies and closed forms |
| `engine/crates/engine-webgpu/tests/invariants_small_gpu.rs` | Layer 1 on the GPU kernel (includes the common module by `#[path]`) |
| `engine/crates/engine-softbody/tests/crash_invariants.rs` | Layer 2 |

Commands:
- `cargo test --release -p engine-softbody --test invariants_small -- --nocapture`
- `cargo test --release -p engine-softbody --test crash_invariants -- --nocapture --include-ignored`
- Community cars: set `RUBBLE_BEAMNG_MODS`.

Every test prints a report row: `| name | car or sim | measured | bound | pass/FAIL |`.

Failing invariants that are real bugs are `#[ignore]`d. Their measured values are in their doc comments, so the owners can un-ignore them when the bugs are fixed.

## Layer 1: small analytic cases

### The step's exact closed forms

The solver integrates with symplectic Euler: v' = v + h·F(x, v)/m, then x' = x + h·v'.

For one degree of freedom (spring k, dashpot c, mass m), the step is a linear map A with:
- det A = 1 − hc/m;
- tr A = 2 − h²k/m − hc/m.

Its eigenvalues are r·e^{±iθ}. This gives exact discrete targets:
- **frequency:** ω_d = acos(tr/2r)/h;
- **decay:** σ_d = −ln√(1 − hc/m)/h.

The continuous √(k/m) and ζω differ from these by the step's known shift, about (ωh)²/24. That is 0.6 % at the skin's ωh = 0.38.

Undamped, the step conserves E_h = ½mv² + ½kx² − ½hkxv exactly. The energy itself oscillates by at most ω_max·h/2 of the system's energy. ω_max is bounded by the Gershgorin rows, as in `Skeleton::bound_rows`.

### Results (CPU, 2026-11-05: all pass)

| Test | Guards | Measured | Closed form, tolerance |
|---|---|---|---|
| `beam_natural_frequency` | spring, integrator | 758.61 rad/s (skin k 8.7e5, 1.53 kg) | ω_d 758.61 (0.1 %); √(k/m) 754.07 |
| `beam_damping_and_log_decrement` | dashpot | σ 20.97/s, δ 0.1728 (ζ 0.0275); σ 43.51/s, δ 0.3564 (ζ 0.0567) | σ_d 20.96, 43.70 (1 %); 2πζ/√(1−ζ²) (3 %) |
| `beam_yield_plastic_set_and_unloading` | yield rule, hardening, RoR and JBeam | Fy 1000.0 N, set 19.843 mm, unload slope 1e5, zero force at the set | set = (k·x − Fy)/(k + Fy/L₀) = 19.842 mm (1 %); slope k (0.1 %) |
| `beam_break_force` | break rule | brittle JBeam: x 20.002 mm at 1998 N; ductile RoR: 1010 N | S/k = 20.000 mm; where Fy·L/L₀ meets S − k(L − L₀) (1 %); RoR with infinite deform never breaks |
| `beam_energy_stored_and_dissipated` | energy accounting | U 3.2087 J; work 25.239 J | F²/2k (10⁻⁶); U + ∫Tt dL (1 %) |
| `support_beam_pushes_only_and_breaks_by_length` | SUPPORT §4 | force error 0.006 N; break at +0.5000 m | 0; (1 + b)·ℓ₀ |
| `bounded_beam_follows_its_blended_law` | BOUNDED §5 | worst relative error 6e-4 | k + (k_lim − k)·min(o/z, 1) (2e-3) |
| `anisotropic_beam_is_a_bilinear_oscillator` | ANISOTROPIC §20 | 7.3874 ms | step 7.3875 ms, π√(μ/k) + π√(μ/kₑ) = 7.4509 ms |
| `pressured_beam_follows_the_gas_law` | PRESSURED §21 | worst 3.8e-7 | (p_abs,0·ℓ₀/(ℓ₀ − c(ℓ₀ − ℓ)) − p_atm)·A |
| `lbeam_conserves_momentum_angular_momentum_and_energy` | LBEAM §24 | Δp 1.3e-5, ΔL/L 1.7e-5, drift −7e-5 J, swing 5e-4 J | M·g·t; 0; no drift; ≤ ω_max·h/2·E |
| `winter_effective_width_closed_form` | sheet buckling yield | 1.9e-7; 47.8 MPa | independent f64 evaluation; the doc's 48 MPa |
| `node_on_ground_never_tunnels_and_rebounds_at_most_a_fifth` | ground contact | depth/v·h 0.969; rebound/v 0.193 | ≤ 1 and ≤ 0.2, from the recurrence v' = 0.2v + 0.2p/h (crossing at depth φvh leaves at 0.2φv) |
| `two_node_bar_on_ground_never_tunnels_or_gains_energy` | contact with a beam | 0.993; 0.158; energy rise 0 | as above; never rises |
| `node_into_box_stops_inside_its_tunnelling_bound` | static boxes (no depth push) | none tunnels at ≤ 0.9× the bound; at 1.5×, 2 of 4 phases tunnel | depth ≤ 1.25·v·h, so safe while v < 0.8·H/h |
| `node_never_passes_a_triangle_below_the_band_speed` | vehicle contact band | 0 passes at ≤ 0.95·r/h; above r/h, 30 of 60 pass | normal speed ≤ r/h = 40 m/s (the node must land in the near half of the ±r band) |
| `struck_bar_wave_speed` | wave propagation | 157.6 m/s (skin k); 408 m/s (at 0.95 of the bound) | l·√(k/m) = 158.4 and 409.4 (2 %); steel's √(E/ρ) = 5,170 m/s |
| `bar_lowest_mode_decays_at_the_stiffness_proportional_rate` | modal damping | 74.4 Hz, σ 5.62/s, ζ₁ 0.0120 (1 % in 820 ms) | the step's; ζ₁ = ζ_b·ω₁/ω_pair = 0.0120 |
| `latch_releases_at_its_statically_shared_strength` | break groups, statics | 4086 N; whole group broken; hinges hold | F = (S + m·g·x̄/a)·a/p = 4088 N (2 %), a plate on three supports (statically determinate) |
| `free_flight_conserves_momentum_and_never_gains_energy` | plain + damping, support, bounded, anisotropic, break group | Δp ≤ 5e-5; internal-energy rise ≤ bound | M·g·t (10⁻⁵); ≤ ω_max·h/2·E |

The GPU file runs the dynamic bodies on `ExplicitBeamsGpu` with the same tolerances:
- frequency and damping;
- yield and break;
- anisotropic period;
- wave speed and mode decay;
- ground contact;
- latch release.

The GPU has no beam-force readback. It measures forces from positions and rest lengths, and holds drivers kinematic by resetting their velocity every step.

On the M3 Max's wgpu adapter (2026-11-05) all six GPU tests pass with the CPU's tolerances:
- ω and σ at ζ 0.0275 and 0.0567: 762.37 against 762.35, and 765.75 against 765.77 rad/s;
- yield and set: 19.843 mm;
- breaks: 20.002 mm;
- bar mode: ω 467.7, σ 5.62;
- anisotropic period, ground contact and latch release.

**What layer 1 tells us about ripple.**
- **Wave speed.** Rubble's discretised skin carries stress waves at l·√(k/m) ≈ 160 m/s, 3 % of steel's speed. Waves cross a 4.7 m car in about 30 ms instead of 1 ms.
- **Damping.** Per-beam dashpots are stiffness-proportional. A low mode is therefore damped by ζ_b·ω₁/ω_pair, not by ζ_b. With the BX's ζ_b = 0.0275, a 74 Hz mode of a 1 m span has ζ₁ = 0.012 and takes 0.8 s to decay to 1 %.

## Layer 2: crash invariants (sedan and sedan1, Medium)

### Scenes and checkers

Each scene runs once per car and is cached.

| Scene | Setup |
|---|---|
| rest | 6 s |
| drive | 15 m/s, 2 s |
| wall | 56 km/h, 1.5 s |
| head-on | 56 km/h into an identical standing car, front to front, two-body contact, 1.5 s |
| side pole | 32 km/h into a 254 mm pole at the front door, 1 s |
| no-drag variants | for diagnosis |

The ledger (`ExplicitBeamsWorld::enable_ledger`) runs in every scene. The checkers:
- **Rigid fit:** Horn's quaternion fit, solved with a 4×4 Jacobi eigen-solver, and tested to recover a rigid motion to 1e-9.
- **Plastic census:** beams past 0.2 % strain, by member group and sign (+ stretched, − shortened, x broken).
- **Hardware:** the `Built.hardware` groups.
- **Per-part ripple:** the RMS over a part's nodes of √2·σ of each node's displacement about the part's rigid fit. Sampled at 1 kHz, in 0.1 s windows.
- **Contact speed:** node samples within 0.3 m of the other car, against the band speed r/h.
- **Visual drift:** each exterior visual vertex's distance from the collision skin, against its rest distance.
- **Creep:** a straight-line fit of the centre of mass over time.

### Thresholds

| Threshold | Value | Source |
|---|---|---|
| Plastic strain | 0.2 % | ASTM E8 offset yield |
| Contact band speed | r/h = 40 m/s | layer 1 |
| Energy | ledger closure ≤ 10⁻³ of start; rise ≤ 0.5 % | `energy_ledger.rs` |
| Creep | drift ≤ sway + 1 mm | `explicit_jbeam_ext.rs` |
| Visual drift | ≤ the contact band, 2 cm | contact band |
| Lid vs rear body | 37 mm | exact from measured data: a/ω² at a = 47 g (the NHTSA peak's top) for a lid at the lowest measured production hood mode, 17.7 Hz |
| Rear-body RMS | 5 mm | **Choice**: millimetres. The rear third carries only its own inertia, ≤ 47 g. |
| Ripple visible | 1 mm | **Choice**: about half a pixel at 2 m |
| Lids' (bonnet, boot lid) first mode | 17.7–41.5 Hz | measured production hoods on the car: first torsion 17.7 Hz (experimental modal analysis, MDPI *Machines* 12(12) 862, 2024) to 41.5 Hz (the least first mode of a lightweight hood's design variants); design targets quoted 25–30 Hz and "above 40 Hz". The decklid takes the same range (same construction, a Choice). Replaces the earlier ≥ 48 Hz, which no measured hood reaches. Doors and fenders are reported |

### Results

| Invariant | sedan | sedan1 | Status |
|---|---|---|---|
| Rest: no plastic or breaks, no creep, ledger | (0, 0); drift 1.07 ≤ sway 3.27 + 1 mm | (0, 0); 0.46 ≤ 3.03 + 1 | **pass** (live) |
| Ledgers close, energy never rises (wall, head-on, pole, rest, drive) | rise ≤ 9.9 J of 195 kJ | pole rise 223 J of 65 kJ (0.34 %) | **pass** (live) |
| Drive 15 m/s: no plastic or breaks | (0, 0) | 2 tension yields (bonnet_inner_1, rail_f_r); 0 with drag off | FAIL, ignored |
| Frontal rear third elastic (wall / head-on) | 21 past 0.2 %, 0 broken / 0 | 131 past, 24 broken / 37 past | FAIL, ignored |
| Boot lid vs rear body, peak/end (wall) | 631/348 mm, no hardware broken | 564/486 mm | FAIL, ignored |
| Rear body RMS/max (wall) | 41/238 mm | 100/518 mm | FAIL, ignored |
| Lid / rear body (head-on) | 172/18 mm; 9.9/70 mm | 253/126 mm; 29.6/173 mm | FAIL, ignored |
| Ripple from 0.5 s, worst part | bootlid 17.9 mm, bonnet 14.7, body 7.4 | bumper_r 15.0, bonnet 7.4, bootlid 6.4 | FAIL, ignored |
| Head-on contacts faster than r/h | 15 of 300,040 (59.9 m/s) | 15 of 197,234 (59.6 m/s) | FAIL, ignored |
| Visual drift from the collision skin (wall / pole), sampled every 0.1 s | 151.3 / 102.9 mm | 172.1 / 150.0 mm | FAIL, ignored. Panel edges past their last node row; waits on hem rows (see 5). |
| Detached door: no drawn triangle joins it to the car (0.3 s after it comes off) | n/a | 0 (263 without the fallback) | **pass** (live) |
| Pole: far side elastic, far doors closed | 40 past, 22 broken; hardware holds | 89 past, 10 broken; door_f_r.latch_secondary and door_r_r.hinge_upper break | FAIL, ignored |
| Lids' first mode 17.7–41.5 Hz (`lids_first_mode_in_the_measured_range`) | bootlid 5.4, bonnet 6.7 Hz | bootlid 6.7, bonnet 8.7 Hz (2026-11-05, later) | FAIL, ignored |
| Body sheet edges yield in tension at ≥ (√3/4)σ_y·t·L | 1029 of 1267 below (median 0.06) | 1132 of 1651 below (median 0.155) | FAIL, ignored. Fixed upstream at 1444ea75 (`rubble_tension_yield`); un-ignore after merging. |

**Community cars** on Rubble's solver (56 km/h wall, rear third), as a control (`community_wall_56_report`):
- **Monster supercar:** 0 beams past 0.2 %, 0 broken; rear RMS/max 137/455 mm. The car bounced back at 3.2 m/s.
- **Astral R:** 4 past, 0 broken; rear 17/88 mm.

The solver lets BeamNG content keep its rear elastic. The authored car's plastic rear is content. Large elastic rear motion appears in the community cars too: their panels are also lattices.

### Mechanisms found

**1. The boot lid and rear body.**
- As a rigid body on its hinges, latch and seals, the lid would move only 2.5–4.5 mm at 40 g (`boot_lid_attachment_report`).
- Its own panel's lowest mode is 2.5–5.4 Hz (`panel_modes_report`: the part cut out, its neighbours held, struck at 0.1 m/s).
- Under the pulse it deflects a/ω²: 0.40 m for 5.4 Hz at 47 g, against 0.63 m measured.
- The lid's rear corner swings 0.62 m forward at 0.2 s. Then its middle sinks 0.33 m into the boot and stays.
- The rear deck (parcel shelf) nodes move about 0.24 m at 0.1 s.
- Cause: single-layer lattice panels have no bending stiffness out of plane, and they sit on light nodes (0.1–0.9 kg).
- Drag is not the cause: the results are the same with drag off.

**2. Plastic rear.** The skin edges' tension yield is far below sheet steel's:
- sedan: 236–377 N, less than one node's own inertia at 47 g (690 N);
- sedan1: the Winter compression value is used in tension too.

So the floppy rear panels yield where they stretch.

**3. Ripple.**
- Lid and bonnet modes are 2.5–7 Hz; doors are 17–37 Hz.
- For a given struck speed the amplitude is v/ω. A lid in the measured 17.7–41.5 Hz would ring 2–8× smaller and decay ζω-proportionally faster.
- Stiffness-proportional damping lowers the modal ratio below the per-beam ζ (layer 1). The measured modal ζ is 0.017–0.055.
- A real steel panel's lowest mode cannot be lower than that of a flat plate of its gauge and spans. Curvature and an inner skin only raise it. The flat simply supported plate gives f₁₁ = (π/2)(1/a² + 1/b²)·√(D/ρt): 11–19 Hz for these lids. Ours are 2.5–5.4 Hz, so for the same struck speed their amplitude is 3.4–4.5× a real lid's, at least.

**4. Fast contacts.**
- Vehicle contact acts in a ±2 cm band, so it can be crossed in one step above 40 m/s normal speed.
- In a 56 km/h head-on, 5–8 per 100,000 node samples near the other car exceed that, up to 60 m/s.
- The lossless fix is a swept crossing test for fast nodes (solver).

**5. Visual drift.** Binding fixed (feat/showcase-car-visual). The remaining drift is content: it needs node rows at the panels' edges.
- The worst exterior visual vertices were bound to skin triangles 0.22–0.37 m away along the normal.
- When such a triangle rotates, the vertex swings 15–18 cm off the collision skin.
- Root cause: each vertex could bind only to its voted part's single-part skin triangles. Every triangle across a shut line, and every one-row part (no single-part triangle), was excluded. A vertex there took the part's nearest remaining triangle, often another sheet of the same part.
- Fix (`authoring::skin`):
  - A vertex within the contact band of its part's own skin rides its part's triangle under it.
  - It no longer rides shut-line triangles that span two parts. On identical physics (`bind=legacy` A/B) those triangles drew ribbons when a door came off and a sawtooth along the bonnet's rear edge.
  - Off the skin, a vertex rides a frame of three of its own part's nodes, as a BeamNG flexbody does. The frame is the part's nearest well-shaped, compact one, with at most one node across the part's border.
  - Structure-group pieces (dash, console, aprons) ride their groups' nodes.
  - The build fails if a vertex lies beyond `BIND_MAX` (0.3 m), or `GROUP_BIND_MAX` (0.45 m) on structure groups.
- Shut-line fallback (`ShowcaseVisual::alt`): a vertex riding a frame across a part border (a one-row part's node frame) also gets its own part's frame at build time. It switches to that frame for good once the frame dies, which happens when:
  - a beam on its death list breaks: its collision triangle's §18 break list, else its sides' cross-part beams, else the part's hinges and latches;
  - or the frame tears.
- The renderer's `skin()` and `deform_beams` do the same.
- Without the fallback, a pulled-off door drew 263 triangles of skin joining it to the car; with the fallback it draws 0.
- Once a lid pops or a latch breaks, its edge rides the lid's own nodes. Those nodes are up to 0.15 m inboard, so the drift grows (above).
- Measuring: `showcaseCar({ bind: 'legacy' })` (and `?bind=legacy` on softbody.html and showcase-car.html, and `crash-lab-record.mjs --query bind=legacy`) selects the binding from before this work on the same build. `alt: false` (`?alt=0`) drops the fallback.
- On identical physics, the doorpull's torn door edge and sill shards appear with the legacy binding too, so they are physics. They differ from the 4e9db94a recording because the physics changed in the merge.
- C5: 10 cars at 16 m (`perf_n10`, Apple M-series) take 4.0 ms (sedan) and 4.1 ms (sedan1), about 0.41 ms a car. Legacy takes 4.2–4.3 ms. The panels' drawn edges lie up to 0.15 m past their last node row. About 31 % of body vertices ride a shut-line triangle. In the crashes, up to 215 of about 1700 such samples stretch past 1.5×, and the renderer tears them at 2.5×. The physics' own collision triangles stretch the same way.

**6. Per-node air drag.**
- The drag is 0.05·|v|·v on every node, whatever its mass: 112 m/s² on a 0.1 kg node at 15 m/s.
- Driving at 15 m/s, it moves the sedan's lid 161 mm relative to the rear body, against 28 mm with drag off.
- It yields 2 beams in sedan1.

Findings were sent to their owners:
- car content (1, 2, 3, 5): the showcase-car agent;
- solver (3's damping, 4, 6): the solver agent.

## Coverage gaps

- **GPU:**
  - Layer 2 has no GPU run (the crash scenes are CPU only).
  - Layer 1's GPU file covers the dynamic bodies only. The force-law sweeps (BOUNDED, PRESSURED, SUPPORT) need beam-force readback, which the GPU does not give.
- **Momentum in crashes:** asserted exactly only in free flight (layer 1). Crash scenes have ground and wall impulses, which the ledger records as work, not impulse. A ground and wall impulse log would let momentum be checked to rounding.
- **Measured thresholds:**
  - The lid, rear-body and ripple-visibility thresholds are Choices. Hinge and latch stiffness and real panel modal data, from measurement, would ground them.
  - The flat-plate frequency is a lower bound, not a measured panel frequency.
- **Rear-third definition:** geometric (a third of the length) rather than by member. The boot floor and the rear rails are counted with the deck.
- **Two-car scenes:** only front to front. No rear impact or offset, and no T-bone invariants (`showcase_car.rs` `t` covers the T-bone leak).
- **Small cases not yet written:**
  - hinge torsion;
  - triangle breaking and deform groups;
  - rails and slidenodes, pressure wheels;
  - self-collision;
  - a sheet-panel test on a synthetic plate with a closed-form mode (only the cut-out car panels are measured).
- **Community cars:** only the wall is reported, with no per-part ripple (their parts are not mapped). The forklift is not run.
