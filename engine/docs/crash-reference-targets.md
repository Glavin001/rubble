# Crash reference targets for the showcase sedan

Status: research note, 2026-11-04. Each number below comes from a public measurement, a regulation, or an
equation. Where nothing open was available, the gap is stated rather than filled.

Subject: the procedurally authored sedan, 1,450 kg, about 4.7 m long and 1.8 m wide.

## 0. Where the numbers come from

**Primary data: the NHTSA Vehicle Crash Test Database API.** This is public-domain JSON, read directly.
- OpenAPI spec: `https://nrd.api.nhtsa.dot.gov/nhtsa/vehicle/v3/api-docs`
- Search endpoint: `.../api/v1/vehicle-database-test-results/by-search`
- Per-vehicle measurements: `.../get-vehicle-detail-info/1/{testNo}`

**Frontal population.** The frontal statistics use this population:
- test type "NEW CAR ASSESSMENT TEST" (all NCAP variants);
- configuration `VEHICLE INTO BARRIER` (rigid);
- impact angle 0 and offset 0;
- closing speed 55.5–57.5 km/h;
- model years 2010–2025;
- test mass 1,300–1,700 kg;
- body type `FOUR DOOR SEDAN`.

That gives **n = 82 tests**: median mass 1,511 kg, median length 4,622 mm, median speed 56.3 km/h.

**Side pole population.** Configuration `VEHICLE INTO POLE`, NCAP, 31–33 km/h, 1,300–1,750 kg, four-door sedans. That gives **n = 116 tests**: median mass 1,566 kg, crab angle 285° (that is, 75°).

**How the statistics are reported.** As median, IQR and p10–p90, because the database contains obvious transcription errors. One example is a door-opening change of 1,860 mm. Means and standard deviations are distorted by these errors, so robust statistics are used throughout.

**What the API does not give: dynamic crush, rebound and pulse.** These need the accelerometer time histories, which are separate data files. Those count as downloads, so they were not fetched (see section 7).

## 1. Reference table

| Quantity | Measured range | Central value for a 1,450 kg sedan | Source / derivation |
|---|---|---|---|
| **Frontal 56 km/h: residual (static) max crush** at bumper level | p10–p90 **380–580 mm**; IQR 431–543; min–max 217–738 | **491 mm** (median) | NHTSA DB, n=82 sedans, field `maximumCrushDistance` (largest of the 6-zone profile C1..C6) |
| Frontal: residual average crush (NHTSA 6-point trapezoid) | p10–p90 261–512 mm; IQR 365–487 | 428 mm | Same 82 tests: C̄ = (C1/2 + C2..C5 + C6/2)/5 |
| Frontal: overall length loss | p10–p90 311–602 mm | 470 mm | Pre-test minus post-test overall length |
| Frontal: crush (damage) width ÷ vehicle width | p10–p90 0.63–0.96 | 0.775 | `totalLengthofIndentation` / `vehicleWidth` |
| Frontal: Campbell **b1** = (v − b0)/C̄, with b0 = 8 km/h | p10–p90 **22.9–39.2 s⁻¹**; IQR 27.2–35.5 | 30.4 s⁻¹ | Derived per test (§2.2). Normalised for mass and width, so it transfers to our car |
| Frontal: Campbell **A**, per unit crush width (b0 = 8 km/h) | p10–p90 48.9–105 kN/m; IQR 69.5–91.2 | 79.4 kN/m | Derived per test: A = m·b0·b1/L |
| Frontal: Campbell **B**, per unit crush width | p10–p90 813–2,049 kN/m²; IQR 961–1,563 | 1,126 kN/m² | Derived per test: B = m·b1²/L |
| Frontal: v ÷ C_max | p10–p90 27.0–41.2 s⁻¹ | 31.9 s⁻¹ | Derived per test |
| Frontal: **coefficient of restitution** e = v_rebound/v_impact | about 0.03–0.2 for barrier tests; falls as speed rises | about 0.11 at 56 km/h | Secondary sources only, from search snippets of papers not opened (see §6). One gives about 0.03–0.2 and a working value of 0.1. An ASME lift-truck paper (page returned 403) states NCAP cars see Δv ≈ 39 mph from a 35 mph impact, which implies rebound ≈ 4 mph ≈ 1.8 m/s and e ≈ 0.11. Monson & Germane, SAE 1999-01-0097 (abstract read): restitution generally decreases with impact speed |
| Frontal: **rebound velocity** | about 0.5–3.1 m/s (e = 0.03–0.2) | about 1.8 m/s | e × 15.56 m/s; same caveat as the row above |
| Frontal: pulse duration | about 90–115 ms | — | Secondary, ASME snippet (unverified). Locey et al. 2012 (AAAM, abstract read): durations are shortening by 0.74 ms per model year for mid-size cars; across the fleet, the 2009–10 durations span only 13.4 ms. Absolute values are not in the abstract |
| Frontal: peak deceleration, occupant-compartment accelerometer | about 36–46 g (fleet average range) | — | Secondary, ASME snippet (unverified). Locey et al.: peak deceleration is rising by 0.40 g per model year for mid-size cars; across the fleet, the 2009–10 peaks span 10.7 g |
| Frontal: mean deceleration over the whole pulse | 15.5–19.7 g | about 17.5 g | Derived: Δv / duration = 17.4 m/s ÷ (0.115…0.090 s). Inherits the secondary-source caveat |
| Frontal: **maximum dynamic crush** | **not openly available** | — | Needs time histories (§7). Physics bound in §2.4 |
| Frontal: time of maximum dynamic crush | about 77 ms (one test) | — | Secondary: snippet of an NHTSA ESV paper on frontal stiffness accuracy (PDF not opened) |
| **Cabin, frontal 56 km/h: door-aperture length change** (leading to trailing edge, upper and lower, left and right) | median **0–1 mm**; IQR about −2…+3 mm; p10–p90 about **−9…+9 mm** | about 0 mm | NHTSA DB pre/post door-edge measurements, n=82. In real cars the aperture is essentially unchanged |
| Cabin: A-pillar base rearward movement | left: median 2, p10–p90 −7…17 mm; right: median 1, p10–p90 −11…15 mm | about 2 mm | NHTSA DB, `RearSurfaceofVehicletoBottomofApost` pre minus post |
| Cabin: firewall rearward movement | left: median 20, p10–p90 1…67 mm; right: median 15, p10–p90 −15…85 mm | about 15–20 mm | NHTSA DB |
| Cabin: steering-column position change | median −40 mm (column moves *forward* when it strokes); p10–p90 −85…+11 | — | NHTSA DB. Not structural; do not assert against it |
| **Side pole, 32 km/h, 75°, 254 mm: residual max crush at the pole** | p10–p90 **297–445 mm**; IQR 336–399; min–max 169–537 | **356 mm** | NHTSA DB, n=116 sedans, `maximumCrushDistance` on the struck side |
| Comparable 1,400–1,500 kg sedans at the pole | 264–388 mm | — | Examples: 2012 Impreza 1,450 kg 388; 2016 Cruze 1,417 kg 320; 2018 Accord 1,508 kg 337; 2020 Sentra 1,475 kg 297; 2025 Corolla Hybrid 1,443 kg 320 |
| **FMVSS 214 static door crush**: required | initial ≥ **10,000 N** (average over 0–152 mm); intermediate ≥ **15,569 N** (average over 0–305 mm); peak ≥ **min(2 × curb weight, 31,138 N)** over 457 mm | For 1,450 kg: peak ≥ 28,449 N | 49 CFR 571.214 S6.1 (seats removed). The alternative S6.2 (seats installed) requires 10,000 N / 19,460 N / min(3.5 × CW, 53,378 N). Ram: rigid 305 mm cylinder, vertical axis, centred 127 mm above the lowest point of the door, ≤ 12.7 mm/s |
| FMVSS 214 static door crush: typical measured | **not openly available** | — | Values exist only in compliance-report PDFs (§7) |
| **FMVSS 216a roof**: required | ≥ **3.0 × unloaded vehicle weight** (GVWR ≤ 6,000 lb) before 127 mm of platen travel, each side; head-form load ≤ 222 N | For 1,450 kg: ≥ 42.7 kN | 49 CFR 571.216a S5. Platen 762 × 1,829 mm, pitched 5° and rolled 25° |
| IIHS roof (rating discontinued) | Good ≥ 4.0; Acceptable 3.25; Marginal 2.5; peak before 5 in (127 mm) | Good for 1,450 kg: ≥ 56.9 kN | iihs.org roof-strength page: "virtually all vehicles earned good ratings" |
| Roof: typical compact sedans | strength-to-weight ratio about 4.5–5.3 | about 5 | Secondary press reports of IIHS results: 2011 Mazda3 sedan 5.32; 2010 Corolla 5.1; 2006–11 Civic 4.48 |

**Comparable frontal tests (NHTSA DB, 1,400–1,520 kg, 4.55–4.90 m sedans).** Values are test number, model, mass, and maximum residual crush:
- 8407, 2014 Corolla, 1,480 kg: 521 mm
- 9065, 2015 Corolla, 1,475 kg: 567 mm
- 9593, 2016 Civic, 1,432 kg: 445 mm
- 9764, 2016 Cruze, 1,506 kg: 488 mm
- 9793, 2017 Elantra, 1,480 kg: 451 mm
- 10688, 2019 Forte, 1,455 kg: 421 mm
- 11574, 2021 Elantra, 1,452 kg: 530 mm
- 8294, 2014 Forte, 1,455 kg: 548 mm
- 9984, 2017 Corolla, 1,488 kg: 738 mm (the population maximum)

## 2. Physics relations, worked for m = 1,450 kg at 56 km/h

v₀ = 56/3.6 = **15.556 m/s**; KE₀ = ½mv₀² = **175.4 kJ**.

### 2.1 Energy balance and restitution

```
½ m v₀² = W_crush(loading) − W_returned(unloading) + ½ m v_r²
                                                    (+ friction, rotation and vibration KE, damping)
e = v_r / v₀                       (rigid, immovable wall)
fraction of KE₀ returned as rebound KE = e²
```

Worked values:

| e | v_r | Rebound KE | Share of KE₀ |
|---|---|---|---|
| 0.03 | 0.47 m/s | 0.16 kJ | — |
| 0.10 | 1.56 m/s | 1.75 kJ | 1.0% |
| 0.20 | 3.11 m/s | 7.0 kJ | 4.0% |
| 0.122 (our sim) | 1.9 m/s | 2.62 kJ | 1.5% |

So 96–99.9% of KE₀ must be dissipated: plastic work, damping and friction.

### 2.2 Campbell linear crush model (CRASH3 form)

Force per unit crush width is linear in residual crush C:

```
F/L = A + B·C
```

Integrating, the crush energy for a uniform profile is:

```
E = L · (A·C + B·C²/2 + G),   with G = A²/(2B)
```

Here G is the energy absorbed with no residual crush. In a rigid-barrier test, E is set equal to ½mv₀², using the impact speed rather than Δv, by CRASH3 convention.

Equivalently, the impact speed is linear in crush above a no-damage speed b0:

```
v = b0 + b1·C
A = m·b0·b1 / L
B = m·b1² / L
```

The b0 = 8 km/h used here is the reconstruction-practice value quoted alongside e ≈ 0.1 in the J.S. Held stiffness-trends paper (seen via a search snippet).

Per-test derivation from the 82 sedans (§1):
- **b1 median 30.4 s⁻¹** (p10–p90 22.9–39.2);
- **A median 79 kN/m**;
- **B median 1,126 kN/m²**;
- **G = A²/2B ≈ 2.8 kJ/m**.

Worked for our car:
- L = 0.775 × 1.8 m = 1.39 m;
- b1 = √(B·L/m) = 32.8 s⁻¹;
- C̄ = (v₀ − b0)/b1 = (15.556 − 2.222)/32.8 = **0.41 m** average residual crush;
- energy check: L(A·C + B·C²/2 + G) = 177 kJ ≈ KE₀ (✓ consistent to 1%).

**Speed scaling.** Residual crush is linear in v above b0, so C(v₂)/C(v₁) = (v₂ − b0)/(v₁ − b0). For example, 50 vs 56 km/h gives C₅₀/C₅₆ = 41.9/47.8 = **0.875**. Energy scales as v², so (50/56)² = 0.797.

### 2.3 Momentum and impulse versus the pulse

```
∫ F_wall dt = m(v₀ + v_r)  = 1,450 × (15.556 + 1.9) = 25.3 kN·s   (sim's v_r)
mean deceleration          = (v₀ + v_r) / T_pulse
deceleration to max crush  = v₀² / (2·D_dyn)
```

Mean deceleration over the pulse: 17.4 m/s ÷ 90–115 ms gives 15.5–19.7 g. The deceleration to maximum crush is an average over displacement.

For any pulse that is symmetric in time, whether constant or half-sine, **D_dyn = v₀·t_m/2**. Here t_m is the time of maximum crush, when the velocity reaches 0. For example, t_m = 77 ms gives D_dyn ≈ 0.60 m, and D_dyn = 0.76 m implies t_m ≈ 98 ms. A front-loaded pulse gives a smaller D for the same t_m.

### 2.4 Elastic fraction (springback)

Springback is D_dyn − C_static. The rebound KE cannot exceed the elastic energy released on unloading. With unloading treated as linear from the wall force at maximum crush, F_D:

```
½ m v_r² ≤ ½ F_D (D_dyn − C_static)
```

From the data, e² ≈ 1–4%, so the elastic part of the energy is small.

**What the open data does not settle.** The openly available data does not give D_dyn − C_static directly. It needs the time histories in §7: double-integrate the rear-crossmember acceleration and subtract the measured static crush.

**Rough estimate (an illustration, not a target).** Assume e = 0.1–0.2 and F_D at about 25 g, which is 356 kN; the 25 g is not sourced. Linear unloading then gives springback δ = e²·m·v₀²/F_D ≈ **10–40 mm**.

### 2.5 Other configurations

- **2 × 50 km/h head-on, identical cars.** By symmetry the contact plane is stationary, so each car behaves as in a **50 km/h rigid-barrier test**. Expect the crush of each car ≈ the barrier crush at 50 km/h ≈ 0.875 × the crush at 56 km/h. Total momentum is 0 before and after. KE per car is 139.9 kJ.
- **Side pole at 32 km/h.** KE = 57.3 kJ. Mean force over a 0.356 m crush is 161 kN, about 11 g.
- **15 m drop.** v = √(2gh) = 17.16 m/s (61.8 km/h); E = mgh = 213.4 kJ. No open test reference exists. Use it only as an energy-balance and stability test.

## 3. Current simulated results versus the references

| Sim result | Reference | Verdict | Reasoning |
|---|---|---|---|
| Permanent crush about 0.60 m (56 km/h) | 380–580 mm p10–p90, median 491; max of 82 tests 738 | ✓ (soft end) | Inside the observed range, but only 6 of 82 sedans reach 600 mm or more (about the 93rd percentile). Measure it the NHTSA way before tuning: residual crush at bumper height, 6 zones over the damage width, maximum and trapezoid average |
| Peak dynamic crush 0.76 m | No open measurement; kinematics gives D = v₀·t_m/2 = 0.60 m at t_m = 77 ms (one secondary data point) | ? (probably soft) | 0.76 m needs t_m ≈ 98 ms and a mean deceleration to maximum crush of 16.2 g. That is at or below the low end of the 15.5–19.7 g whole-pulse mean, and the pulse-to-maximum-crush mean should be higher than the whole-pulse mean. Not decidable until the §7 time histories are analysed |
| Springback 0.76 − 0.60 = 0.16 m, with rebound 1.9 m/s | Physics: ½mv_r² ≤ ½F_D·δ. Illustrative springback for real cars is 10–40 mm | ✗ (suspicious) | The rebound KE of 2.6 kJ over 160 mm of springback implies an average unloading force of only about 33 kN (2.3 g). A real structure unloads from a wall force of hundreds of kN over a few centimetres. The 160 mm is more likely low-stiffness recovery, such as panels or the measuring node bouncing, than structural springback. Check which node defines "crush" |
| Rebound 1.9 m/s (e = 0.122) | e about 0.03–0.2; about 0.11 implied by "Δv ≈ 39 mph" | ✓ | Mid-range. Both sources are secondary |
| Door aperture change about 20 mm | Median 0–1 mm, p10–p90 about ±9 mm | ✗ | More than 2× the p90 of real sedans. Real cabins are effectively rigid in this test |
| Cabin pillar deformation 9–12 mm | A-pillar base rearward: median 1–2 mm, p90 15–17 mm | ✓ (marginal) | Within p90, if "pillar deformation" means A-pillar base displacement. If it means pillar *bending*, there is no direct NHTSA measure |
| Side pole intrusion about 0.13 m, relative to the cabin's far side | Residual exterior crush at the pole: p10–p90 297–445 mm, median 356; the minimum of all 116 is 169 mm | ✗ (or the metric doesn't match) | Below the smallest of 116 real sedans. Mean wall force would be 441 kN (31 g), against about 161 kN (11 g) at the median. Measure as NHTSA does (residual crush of the struck-side exterior relative to its pre-test line, at the pole centreline) before concluding |

## 4. Recommended test assertions

Bands are p10–p90 of the NHTSA sedan population unless noted. Hard-fail bounds are the observed min–max. Exact-physics assertions use numerical tolerances, and those tolerances should be set from the sim's own drift, not from crash data.

**Exact physics (any crash):**

1. **Energy ledger.** KE₀ = KE_final + E_plastic + E_damping + E_friction + E_elastic_stored, closed to within the solver's measured energy drift. At 56 km/h, KE₀ = 175.4 kJ.
2. **Impulse.** ∫F_wall dt = m·(v₀ + v_r), using the wall contact force sum and the CG velocity after separation.
3. **Momentum in the head-on.** Total momentum is ≈ 0 throughout. Each car's CG Δv is ≈ 13.9 m/s plus its rebound.
4. **Kinematic consistency.** D_dyn equals ∫v_CG dt up to t(v = 0). The sim then reports t_m and checks D_dyn ≈ v₀·t_m/2 for a near-symmetric pulse (a soft check; record the ratio).

**Frontal, 56 km/h, rigid wall:**

5. Residual max crush at bumper height: **380–580 mm**; hard fail outside 217–738 mm.
6. Campbell b1 = (v₀ − 2.22 m/s)/C̄ (NHTSA 6-zone trapezoid C̄): **22.9–39.2 s⁻¹**. Normalised for mass and width, so it stays valid if the car's mass changes.
7. Restitution e = v_r/v₀: **0.03–0.20**. Secondary sources; tighten once the §7 data is analysed.
8. Springback bound: ½m·v_r² ≤ ½·F_wall(t_m)·(D_dyn − C_static), and springback ≤ 0.1·C_static. The 10% figure is from the §2.4 estimate, about 40 mm ÷ 491 mm; treat it as provisional.
9. Door-aperture length change: **|Δ| ≤ 9 mm** (p90). A-pillar base rearward ≤ 17 mm. Firewall rearward ≤ 85 mm.

**Speed scaling (Campbell linearity):**

10. C̄(50)/C̄(56) ≈ (50 − 8)/(56 − 8) = 0.875. Also, each car in the 2 × 50 km/h head-on ≈ the 50 km/h barrier crush. The tolerance is not set by data: open sources give no scatter for this ratio, so use the sim's run-to-run spread and report the value.

**Side pole, 32 km/h, 75°, 254 mm:**

11. Residual struck-side exterior crush at the pole: **297–445 mm**; hard fail outside 169–537 mm.

**Component tests (regulatory minimums, pass/fail):**

12. Door crush (FMVSS 214 S6.1 rig): the mean force must be:
    - at least 10,000 N over 0–152 mm;
    - at least 15,569 N over 0–305 mm;
    - with a peak of at least 28,449 N over 457 mm (2 × curb weight for 1,450 kg; use the actual curb mass).
13. Roof (FMVSS 216a rig): force at ≤ 127 mm platen travel ≥ 3.0 × UVW = 42.7 kN on each side. IIHS "good" (≥ 4.0 × UVW, 56.9 kN) is a stretch target. Typical compact sedans reach about 4.5–5.3 × UVW.

## 5. Limitations

**Data not openly available.**
- Dynamic crush, rebound velocity, pulse duration and peak deceleration for specific tests exist only as time-history files (§7).
- The pulse and restitution values above come from search-engine snippets of papers that could not be opened (403, or a PDF). They are labelled as secondary.
- Measured FMVSS 214 static door-crush forces exist only in compliance-report PDFs.

**Mismatched metrics.**
- NHTSA "crush" is a residual exterior measurement against the pre-test profile, at bumper height, in 6 zones.
- "Pole intrusion relative to the far side" and "pillar deformation" are not NHTSA quantities. The sim needs probes that reproduce the NHTSA definitions before ✓ and ✗ are final.

**Database noise.** Hand-measured fields carry about ±2–5 mm resolution, plus transcription errors. That is why medians and percentiles are used. Door-aperture "changes" under about 5 mm are measurement noise in the reference too.

**Population.** These are 2010–2025 US-market sedans, mostly compacts. The sim is a generic sedan, so it should land inside the spread, not on a particular car.

**What a node–beam model can't match.**
- A node–beam car at roughly 10 cm node spacing cannot reproduce local sheet-metal folding. Its crush is a homogenised beam-plasticity response.
- Compare global quantities, such as energy, b1, crush depth, restitution and pulse, rather than local deformation shapes.
- Accelerometer pulses in the sim will carry beam-vibration noise. Filter them with SAE J211 CFC 60 before comparing peaks, as NHTSA practice does for vehicle-structure channels.

**Campbell model assumptions.** It assumes a uniform vertical profile, ignores rebound in the energy (CRASH3 convention), and depends on b0. The 8 km/h b0 value is a convention, not a measurement. Changing b0 shifts A and B but not b1's role in assertion 6.

## 6. Sources

**Read directly**
- NHTSA Vehicle Crash Test Database API (OpenAPI): `https://nrd.api.nhtsa.dot.gov/nhtsa/vehicle/v3/api-docs`. Endpoints used: `by-search`, `get-vehicle-detail-info`, `get-instrumentation-info`, `get-instrumentation-detail-info`.
- 49 CFR 571.214 (side impact, door crush S6, pole S9/S10): https://www.law.cornell.edu/cfr/text/49/571.214
- 49 CFR 571.216a (roof crush): https://www.law.cornell.edu/cfr/text/49/571.216a
- IIHS roof-strength test description: https://www.iihs.org/ratings/about-our-tests/roof-strength
- Monson & Germane, "Determination and Mechanisms of Motor Vehicle Structural Restitution from Crash Test Data", SAE 1999-01-0097 (abstract): https://saemobilus.sae.org/papers/determination-mechanisms-motor-vehicle-structural-restitution-crash-test-data-1999-01-0097
- Locey et al., "Homogenization of vehicle fleet frontal crash pulses from 2000–2010", AAAM 2012, PMC3503406 (abstract via Europe PMC).

**Secondary (search snippets only, not opened)**
- J.S. Held, "Vehicle Frontal Crush Stiffness Coefficients Trends" (e ≈ 0.03–0.2, b0 = 8 km/h): https://jsheld.com/uploads/Vehicle-Frontal-Crush-Stiffness-Coefficients-Trends.pdf
- "Frontal Crash Testing of a Class V Lift Truck", ASME Open Engineering (NCAP Δv ≈ 39 mph, 90–115 ms, peak 36–46 g): https://asmedigitalcollection.asme.org/openengineering/article/doi/10.1115/1.4065105/1199076/Frontal-Crash-Testing-of-a-Class-V-Lift-Truck
- NHTSA ESV, "Accuracy of Vehicle Frontal Stiffness Estimates…" (maximum dynamic crush at about 77 ms): https://www-nrd.nhtsa.dot.gov/pdf/ESV/Proceedings/20/07-0513-O.pdf
- IIHS roof results for the 2011 Mazda3, 2010 Corolla and 2006–11 Civic: press coverage (autoblog.com, pressroom.toyota.com, autosafetymsi.blogspot.com)

## 7. Files that would close the gaps (each needs the user's permission to download)

**Time histories.** The rear-seat crossmember X accelerations (channels 093 and 094) of the comparable sedans:
- from these we can compute dynamic crush (by double integration), rebound velocity, peak and mean deceleration (CFC 60), pulse duration and t_m;
- format: ASCII TSV, a few thousand rows each, roughly 100 KB;
- URL pattern: `https://nrd-static.nhtsa.dot.gov/tsv/vehdb/v0XX00/v0XXX0/v0NNNN/v0NNNNtsv.093` (and `.094`).

| Test | Car | Files |
|---|---|---|
| 8407 | 2014 Corolla | `.../v00000/v08400/v08407/v08407tsv.093`, `.094` |
| 9065 | 2015 Corolla | `.../v00000/v09000/v09065/v09065tsv.093`, `.094` |
| 9079 | 2015 Sentra | `.../v00000/v09000/v09079/v09079tsv.093`, `.094` |
| 9593 | 2016 Civic | `.../v00000/v09500/v09593/v09593tsv.093`, `.094` |
| 9764 | 2016 Cruze | `.../v00000/v09700/v09764/v09764tsv.093`, `.094` |
| 9793 | 2017 Elantra | `.../v00000/v09700/v09793/v09793tsv.093`, `.094` |
| 10688 | 2019 Forte | `.../v10000/v10600/v10688/v10688tsv.093`, `.094` |
| 11574 | 2021 Elantra | `.../v10000/v11500/v11574/v11574tsv.093`, `.094` |

All of these live under `https://nrd-static.nhtsa.dot.gov/tsv/vehdb`. The barrier load-cell channels (about 104 onwards) in the same tests would also give the wall force–deflection curve. That allows a direct comparison against F/L = A + B·C.

**Papers (PDFs).**
- J.S. Held trends paper (above): tabulated A, B and e by class.
- Wiacek et al., ESV 2015, "Evaluation of frontal crash stiffness measures from the U.S. NCAP": https://www-nrd.nhtsa.dot.gov/departments/esv/24th/files/24ESV-000257.PDF. It covers dynamic crush and pulse statistics for MY2002–2014.
- JH Scientific, "Crush Analysis with Under-rides and the Coefficient of Restitution": https://www.jhscientific.com/downloads/CrushAndRestitution.pdf. It has a table of mean and SD of barrier e at 30/35 mph, after Prasad.
- FMVSS 214 static door-crush compliance reports, for measured door forces. These are found through NHTSA's compliance test-report search; no specific file was identified.

## 8. Measured pulses (NHTSA time histories, computed 2026-11-04)

Rear-seat crossmember X accelerometers (channels 093 and 094, G's, global X, averaged
per test) of the eight 56 km/h NCAP sedan tests in section 7, integrated from 56.3 km/h
by `engine/scripts/nhtsa_pulse.py` (velocity and displacement from the raw signal;
peak through a 100 Hz two-pole Butterworth run both ways, close to CFC 60). The files
were downloaded with the user's permission into the gitignored
`inspiration/nhtsa-pulses/` and are not committed.

| Test | Car | Dynamic crush (m) | t_m (ms) | Rebound (m/s) | e | Peak (g) | Mean to t_m (g) |
|---|---|---|---|---|---|---|---|
| 8407 | 2014 Corolla | 0.723 | 69 | 2.52 | 0.161 | 45.3 | 23.0 |
| 9065 | 2015 Corolla | 0.753 | 73 | 2.65 | 0.169 | 46.9 | 22.0 |
| 9079 | 2015 Sentra | (1.105) | (91) | (7.47) | (0.48) | 56.6 | (17.6) |
| 9593 | 2016 Civic | 0.680 | 76 | 3.10 | 0.198 | 34.3 | 21.0 |
| 9764 | 2016 Cruze | 0.630 | 68 | 1.98 | 0.127 | 39.4 | 23.3 |
| 9793 | 2017 Elantra | 0.647 | 65 | 2.40 | 0.154 | 43.9 | 24.6 |
| 10688 | 2019 Forte | 0.617 | 63 | 2.88 | 0.184 | 44.0 | 25.5 |
| 11574 | 2021 Elantra | 0.632 | 69 | 2.30 | 0.147 | 39.6 | 23.1 |

9079 is excluded: its integral drifts (7.5 m/s rebound and 1.1 m crush are not
physical for that car's measured residual crush), likely an accelerometer offset or
saturation.

Over the other seven:
- **dynamic crush** 0.617–0.753 m (median 0.647 m);
- **time of maximum crush** 63–76 ms (median 69 ms);
- **rebound** 1.98–3.10 m/s (median 2.52 m/s);
- **restitution** 0.127–0.198 (median 0.161);
- **peak** 34–47 g (median 43.9 g);
- **mean deceleration to t_m** 21–25.5 g.

What this corrects:
- **Restitution.** The measured e is 0.13–0.20. Section 3's "about 0.11" was low, and a rebound target below 1.5 m/s would be wrong.
- **Springback.** Dynamic crush (median 0.65 m) minus residual crush (section 2, median 0.49 m) is about 0.16 m, roughly 25 % of the dynamic crush. Springback of 0.15–0.2 m is therefore normal for these cars, and a "≤ 10 %" bound would be wrong.
- **What the simulation should match** at 56 km/h, 1,450 kg:
  - dynamic crush 0.62–0.75 m;
  - t_m 63–76 ms;
  - rebound 2.0–3.1 m/s (e 0.13–0.20);
  - peak 34–47 g at the rear seat crossmember (filtered as above);
  - residual crush 380–580 mm (section 2).

Caveat: these are 7 small and compact sedans of 1,250–1,400 kg. Mid-size sedans of
about 1,450–1,550 kg sit in the same NCAP band (section 2's statistics cover them).

## 9. Door hardware strength (FMVSS 206 compliance reports)

Added 2026-11-05. These are measured door latch and hinge loads from NHTSA's FMVSS 206 compliance test reports. They replace the "1.5 × FMVSS 206 minimum" estimate.

**Where the reports come from.** NHTSA's compliance report search (nhtsa.gov/compliance, under "Other" equipment, FMVSS 206) is backed by a public JSON API, `https://ovsc.api.nhtsa.dot.gov/other?fmvss_code=206`. It returns 427 FMVSS 206 reports, each with a PDF at `static.nhtsa.gov/odi/ctr/...`.
- 41 reports were downloaded: 40 passenger cars from model years 2005–2023, and one compact SUV (the Bronco Sport, not used here).
- They were downloaded with the user's permission into the gitignored `inspiration/fmvss206/` and are not committed.

**What the test is.** It is a quasi-static bench pull on a production latch and striker, or on the hinge pair, mounted in a fixture (TP-206). One test per position, left front door.
- Latch "Load Test One" (formerly called "longitudinal"): required 11,000 N fully latched and 4,450 N (4,500 N from 2007) secondary.
- Latch "Load Test Two" (formerly "transverse"): required 8,900 N (9,000 N from 2007) fully latched and 4,450/4,500 N secondary.
- Hinge system (upper and lower hinge together): required 11,000 N longitudinal and 8,900/9,000 N transverse.

**Vehicle mass is not reported.** The reports test parts on a bench, not whole vehicles, so no mass is given. The cars span compact sedans (Civic, Corolla, Forte, Mazda3, Verano, Jetta) and mid-size or upper mid-size sedans (Camry, Accord, Altima, Passat, Maxima, LaCrosse, ES 350, C300). That brackets the 1,450 kg showcase sedan.

**Most reports only prove the minimum.** Whether a test went to failure depends on the laboratory and the year. The end state of each test was checked against its remarks and, where noted, its force curve.
- **KARCO Engineering, 2005–2012.** These tests load the latch until it separates: the force peaks and then drops, and the remarks say "teeth bent open", "hardware broke" or "striker broke in half". Curves were checked for six of the 11 reports (Jetta, Forte, Mazda3, 2008 Altima, C300, ES 350), and they show the peak and drop. The hinges, by contrast, were stopped at about 12–13 kN longitudinal and 10–11 kN transverse with "no damage".
- **MGA Research, 2005–2006.** Each test was pushed well past the requirement. Some latches failed: the Camry and Corolla curves peak and drop, and the Civic and Accord remarks record a broken striker bolt or latch and striker pulled apart. Other tests were stopped while the force was still rising. The hinges reached up to 36 kN, with no separation.
- **MGA Research, 2007–2024.** These are proof tests only, stopped while the force is still rising. Curves were checked for the 2011 Cruze and the 2022 Bronco Sport. Since about 2016 MGA has stopped at fixed loads: about 13.9 / 7.1 / 11.7 / 7.1 kN for the latch and 13.9 / 11.7 kN for the hinges. These are lower bounds only.

**Inertial test and striker.**
- **Inertial (30 g) test.** None of the reports contain one. The TP-206I sled test is a separate procedure and is not part of these reports. Manufacturers usually certify the inertial requirement by calculation.
- **Striker.** It is not tested on its own. Its failures show up as the end state of the fully latched Load Test One: in 9 of the 13 failures, the striker or its mounting fasteners broke.

### 9.1 Latches loaded to failure (maximum load at separation, N)

"≥" means that test stopped without separating, so the value is a lower bound. Hinge values are the maximum applied. No hinge system separated in any report.

| NHTSA no. | Lab no. | Vehicle (MY) | Primary, Load Test One | Secondary, Load Test One | Primary, Load Test Two | Secondary, Load Test Two | Hinge longitudinal | Hinge transverse | Latch end state |
|---|---|---|---|---|---|---|---|---|---|
| 638716 | 206-KAR-07-002 | 2007 Nissan Altima | 23,555 | 19,608 | 18,879 | 10,968 | ≥12,059 | ≥10,705 | striker bolts pulled out; latch released |
| 639635 | 206-KAR-08-009 | 2008 Nissan Altima | 37,936 | 20,834 | 16,547 | 9,507 | ≥12,217 | ≥10,737 | striker hardware broke; peak then drop |
| 639290 | 206-KAR-08-013 | 2008 Mercedes C300 | 27,967 | 19,968 | 22,408 | 16,051 | ≥12,126 | ≥10,542 | striker broke in half; peak then drop |
| 640327 | 206-KAR-09-014 | 2009 VW Jetta | 24,245 | 8,700 | 19,648 | 9,986 | ≥12,642 | ≥10,683 | teeth bent open; peak then drop |
| 640328 | 206-KAR-09-015 | 2009 VW Passat | 30,650 | 13,960 | 23,214 | 11,530 | ≥12,262 | ≥10,842 | teeth bent open (all four) |
| 640262 | 206-KAR-09-001 | 2009 Nissan Maxima | 35,165 | 15,486 | 25,752 | 10,609 | ≥12,462 | ≥10,517 | striker hardware broke; teeth bent open |
| 641195 | 206-KAR-10-009 | 2010 Kia Forte | ≥39,918 | 18,180 | 20,278 | 9,670 | ≥12,226 | ≥10,123 | primary Load Test One plateaued at the rig's about 40 kN without breaking |
| 641049 | 206-KAR-10-005 | 2010 Mazda3 (5-door) | 36,803 | 22,753 | 27,651 | 11,634 | ≥12,644 | ≥10,410 | striker hardware broke; teeth bent open |
| 641050 | 206-KAR-10-006 | 2010 Buick LaCrosse | 28,035 | 23,793 | 21,001 | 13,344 | ≥12,670 | ≥10,672 | striker hardware broke; teeth bent open |
| 642464 | 206-KAR-83-003 | 2012 Buick Verano | 33,506 | 23,920 | 20,621 | 13,336 | ≥12,523 | ≥10,432 | striker broke in half; teeth bent open |
| 642316 | 206-KAR-83-002 | 2012 Lexus ES 350 | 32,320 | 14,230 | 29,169 | 8,241 | ≥12,873 | ≥10,641 | hardware broke; teeth bent open |
| 637726 | 206-MGA-05-012 | 2005 Toyota Camry | 22,953 | 16,351 | ≥15,913 | 7,554 | ≥36,179 | ≥29,813 | three of four latch curves peak then drop; hinges slightly distorted |
| 637782 | 206-MGA-05-015 | 2005 Toyota Corolla | 20,168 | 13,762 | 18,576 | 7,469 | ≥28,956 | ≥25,203 | all four latch curves peak then drop; hinges distorted |
| 637586 | 206-MGA-05-001 | 2005 Honda Civic | 26,784 | 28,278 | 19,650 | 5,351 | ≥28,682 | ≥28,548 | striker bolt broke ×3; latch and striker pulled apart |
| 637593 | 206-MGA-05-004 | 2005 Honda Accord | ≥27,397 | ≥10,560 | 13,614 | ≥10,034 | ≥27,817 | ≥20,846 | only primary Load Test Two failed (striker bolt sheared) |

### 9.2 Proof-only reports (test stopped above the requirement, N)

All of these are lower bounds. Order of values: primary Load Test One / secondary Load Test One / primary Load Test Two / secondary Load Test Two; hinge longitudinal / hinge transverse.

| NHTSA no. | Vehicle (MY) | Latch | Hinge |
|---|---|---|---|
| 637594 | 2005 Ford Focus | 20,098 / 10,652 / 17,799 / 8,901 | 22,131 / 19,736 |
| 638044 | 2006 Ford Taurus | 18,372 / 12,328 / 18,502 / 12,929 | 21,302 / 18,170 |
| 638125 | 2006 Chevrolet Impala | 15,127 / 12,238 / 15,119 / 12,831 | 16,501 / 18,384 |
| 638676 | 2007 Ford Fusion | 16,747 / 15,045 / 12,956 / 11,935 | 16,518 / 15,858 |
| 638693 | 2007 Saturn Aura | 17,635 / 15,341 / 16,611 / 12,589 | 18,109 / 11,776 |
| 639279 | 2008 Ford Focus | 14,147 / 11,399 / 12,276 / 7,496 | 14,601 / 10,689 |
| 640326 | 2009 Mazda6 | 14,436 / 6,854 / 12,470 / 6,830 | 14,288 / 10,422 |
| 640325 | 2009 Acura TSX | 14,512 / 8,570 / 9,379 / 6,284 | 14,905 / 12,252 |
| 641797 | 2011 Honda Civic | 14,313 / 9,211 / 12,507 / 8,445 | 14,528 / 12,419 |
| 641767 | 2011 Chevrolet Cruze | 15,658 / 10,774 / 14,653 / 10,666 | 16,667 / 13,643 |
| 641809 | 2011 Hyundai Sonata | 15,532 / 8,578 / 13,602 / 8,266 | 15,554 / 14,652 |
| 641810 | 2011 Kia Optima | 15,345 / 9,124 / 14,303 / 8,636 | 15,271 / 15,129 |
| 645328, 645332, 647066, 647368, 647371, 647375, 647376, 647754, 206-2360623, 206-2360631, 206-2360621, 206-2360637 | 2016 Fusion, 2016 Civic, 2020 Mazda6, 2021 Malibu, 2021 Sonata, 2021 Legacy, 2021 Corolla, 2022 Sentra, 2022 Camry, 2023 Accord, 2023 Altima, 2023 Model 3 | 13.8–13.9 k / 7.03–7.19 k / 11.65–11.83 k / 7.07–7.38 k | 13.8–14.2 k / 11.6–11.9 k |

The 2005 Chevrolet Malibu report (637911, KARCO) was also downloaded. Its result cells are too degraded to read, so it is not used.

### 9.3 Medians and ranges (failures only, 2005–2012 hardware)

| Quantity | n | Median | IQR | Range | Median ÷ requirement |
|---|---|---|---|---|---|
| Primary latch, Load Test One (longitudinal) | 13 (+1 ≥39.9 kN) | **28.0 kN** | 23.9–34.3 kN | 20.2–37.9 kN | 2.5 |
| Secondary latch, Load Test One | 14 | **18.9 kN** | 14.2–23.0 kN | 8.7–28.3 kN | 4.2 |
| Primary latch, Load Test Two (transverse) | 14 | **20.4 kN** | 18.8–23.8 kN | 13.6–29.2 kN | 2.3 |
| Secondary latch, Load Test Two | 14 | **10.3 kN** | 8.1–12.1 kN | 5.4–16.1 kN | 2.3 |
| Hinge system, longitudinal | 0 failures | — | — | maximum without separation 36.2 kN; median of the seven 2005–06 tests ≥ 27.8 kN | ≥ 2.5 |
| Hinge system, transverse | 0 failures | — | — | maximum without separation 29.8 kN; median of the seven 2005–06 tests ≥ 20.8 kN | ≥ 2.3 |

Notes on the medians:
- **Primary longitudinal is slightly understated.** The Forte's ≥39.9 kN is left out of that median, so the true median is a little higher.
- **Later hardware is consistent.** The 2016–2023 proof loads (13.9 / 7.1 / 11.7 / 7.1 kN) sit below every failure value except one, the weakest secondary Load Test Two latch (2005 Civic, 5.4 kN). So the later cars contradict nothing above. They only show that MGA stopped at about 126–158 % of the requirement.
- **Displacement at the latch peak** in the KARCO curves is 16–40 mm (mostly 19–29 mm), which includes fixture compliance. The tests run at about 0.15 in/min. That is quasi-static, so there are no strain-rate effects.

### 9.4 Recommended ultimate strengths for the 1,450 kg showcase sedan

These are per door. Each value is the median from §9.3, with the IQR as the spread.

| Component | Recommended ultimate | Spread (IQR) | Replaces (1.5 × minimum) |
|---|---|---|---|
| Primary latch, longitudinal (Load Test One) | **28 kN** | 24–34 kN (range 20–38 kN) | 16.5 kN |
| Primary latch, transverse (Load Test Two) | **20.5 kN** | 19–24 kN (range 14–29 kN) | 13.5 kN |
| Secondary latch, longitudinal | **19 kN** | 14–23 kN (range 9–28 kN) | 6.7 kN |
| Secondary latch, transverse | **10.3 kN** | 8–12 kN (range 5–16 kN) | 6.7 kN |
| Hinge system, longitudinal (both hinges) | **≥ 28 kN** (lower bound; use 28 kN, or about 14 kN per hinge if they share equally) | no separation seen up to 36 kN | 16.5 kN |
| Hinge system, transverse (both hinges) | **≥ 21 kN** (lower bound; use 21 kN, or about 10.5 kN per hinge) | no separation seen up to 30 kN | 13.5 kN |

What the recommendation implies:
- **The old estimate was too weak.** The 1.5 × factor made the hardware 1.5–2.8 × weaker than measured. Measured primary latches fail at a median of 2.3–2.5 × the requirement, and secondary latches at 2.3–4.2 ×.
- **Hinges should not break first.** No hinge pair separated in any report, including seven pairs loaded to 16.5–36 kN, while latches separated at 5–38 kN. So a simulated door under these loads should open at the latch, usually at the striker or its fasteners, before a hinge separates.
- **Hinge strength is not measured.** The hinge values are floors, not strengths: no report loaded a hinge system to separation. If a hinge break value is needed, take it at or above the latch value for the same direction. That is about 28 kN longitudinal and 21 kN transverse, which matches the medians of the 2005–06 hinge floors.

**Sources.** NHTSA OVSC compliance test reports, FMVSS 206. The index is at `https://ovsc.api.nhtsa.dot.gov/other?fmvss_code=206`. PDFs:
- KARCO latch-failure reports:
  - 638716 `https://static.nhtsa.gov/odi/ctr/2007/TRTR-638716-2007.PDF`
  - 639635 `https://static.nhtsa.gov/odi/ctr/2008/TRTR-639635-2008.pdf`
  - 639290 `https://static.nhtsa.gov/odi/ctr/2008/TRTR-639290-2008.pdf`
  - 640327 `https://static.nhtsa.gov/odi/ctr/2009/TRTR-640327-2009-001.pdf`
  - 640328 `https://static.nhtsa.gov/odi/ctr/2009/TRTR-640328-2009-001.pdf`
  - 640262 `https://static.nhtsa.gov/odi/ctr/2009/TRTR-640262-2009-001.pdf`
  - 641195 `https://static.nhtsa.gov/odi/ctr/2010/TRTR-641195-2010-001.pdf`
  - 641049 `https://static.nhtsa.gov/odi/ctr/2010/TRTR-641049-2010-001.pdf`
  - 641050 `https://static.nhtsa.gov/odi/ctr/2010/TRTR-641050-2010-001.pdf`
  - 642464 `https://static.nhtsa.gov/odi/ctr/2012/TRTR-642464-2012-001.pdf`
  - 642316 `https://static.nhtsa.gov/odi/ctr/2012/TRTR-642316-2012-001.pdf`
- MGA 2005–06 reports:
  - 637726 `https://static.nhtsa.gov/odi/ctr/9999/TRTR-637726-20605.pdf`
  - 637782 `https://static.nhtsa.gov/odi/ctr/9999/TRTR-637782-20605.pdf`
  - 637586 `https://static.nhtsa.gov/odi/ctr/9999/TRTR-637586-20605.pdf`
  - 637593 `https://static.nhtsa.gov/odi/ctr/2005/TRTR-637593-2005.PDF`
  - 637594 `https://static.nhtsa.gov/odi/ctr/2005/TRTR-637594-2005.PDF`
  - 638044 `https://static.nhtsa.gov/odi/ctr/2005/TRTR-638044-2005.PDF`
  - 638125 `https://static.nhtsa.gov/odi/ctr/2006/TRTR-638125-2006.PDF`
- MGA proof reports:
  - 638676 `https://static.nhtsa.gov/odi/ctr/2007/TRTR-638676-2007.PDF`
  - 638693 `https://static.nhtsa.gov/odi/ctr/2007/TRTR-638693-2007.PDF`
  - 639279 `https://static.nhtsa.gov/odi/ctr/2008/TRTR-639279-2008.pdf`
  - 640326 `https://static.nhtsa.gov/odi/ctr/2009/TRTR-640326-2009-001.pdf`
  - 640325 `https://static.nhtsa.gov/odi/ctr/2009/TRTR-640325-2009-001.pdf`
  - 641797, 641767, 641809, 641810 `https://static.nhtsa.gov/odi/ctr/2011/TRTR-<no>-2011-001.pdf`
  - 645328, 645332, 647066, 647368, 647371, 647375, 647376, 647754 `https://static.nhtsa.gov/odi/ctr/9999/TRTR-<no>-<year>-001.pdf`
  - 206-2360623 and 206-2360621 `https://static.nhtsa.gov/odi/ctr/2023/<no>.pdf`
  - 206-2360631 and 206-2360637 `https://static.nhtsa.gov/odi/ctr/2024/<no>.pdf`
- Test procedures: TP-206-08 `https://www.nhtsa.gov/sites/nhtsa.gov/files/tp-206-08_19_feb_2010.pdf`; 49 CFR 571.206.

## 10. Crash-lab rigs: the FMVSS 214 barrier and the door pull (2026-11-05)

`engine_softbody::crash_rigs`, scenes `mdb214` and `doorpull` in
`engine-world-wasm/src/explicit.rs`, lab presets `mdb214` and `doorpull`.

**The moving deformable barrier** (a JBeam vehicle Rubble generates, run in JBeam mode):

| quantity | value | source |
|---|---|---|
| mass | 1,367.6 kg (3,015 lb, configuration 2) | 49 CFR 587.6(c) |
| wheelbase, track | 2.591 m (102 in), 1.880 m (74 in) | 587.6(c) |
| centre of mass | 1.123 m (44.2 in) behind the front axle, 0.500 m (19.7 in) up | 587.6(d) |
| face | 1,676 mm wide, 559 mm high, 279 mm ground clearance | NHTSA TP-214D (via a search summary of TP-214D-09; the PDF returned 403) and the AFL FMVSS 214 barrier data sheet |
| depth | 381 mm main block, 483 mm at the bumper (bumper protrudes 102 mm) | AFL and AXAC data sheets |
| crush strength | main block 310 kPa (45 psi ± 2.5), bumper element 1,689 kPa (245 psi ± 15) | TP-214D; AFL data sheet; 49 CFR 587.15 gives the sampling |
| test | 53 ± 1 km/h (33.5 mph), wheels crabbed 27° ± 1°, its forward edge's plane on the impact reference line (940 mm ahead of the wheelbase's centre up to a 2,896 mm wheelbase, else 508 mm behind the front axle) | 49 CFR 571.214 S7, S8.10, S8.11.1 |

Choices (the drawings DSL-1278 … DSL-1290 that 587.6 incorporates are not openly
available): the bumper element covers the face's lowest third; the face is 17 × 7
nodes of 0.8 kg (and 17 × 3 bumper nodes) with tributary areas summing to the face
(a coarser 9 × 4 face of 1.6 kg nodes put 33 kN on each bumper node and punched
through the sedan's door skin: the barrier passed through the car); the honeycomb is columns of
SUPPORT beams that yield at the crush stress times their area with no hardening
(`deformLimitStress` = the crush force) and densify at 80 % crush; elastic strain at
the crush stress 10 % (main) and 20 % (bumper), set so the columns integrate at the
2 kHz step on 0.8 kg nodes; the front axle 0.6 m behind the face; frictionless skids
for the crabbed, free-rolling wheels. Structure (2026-11-05, after the barrier
collapsed to a sheet in the lab): the carriage's beams each as stiff as the 2 kHz
step allows their two nodes (k = 0.25·min(m/n)/h², n a node's carriage beams; the
first version took one stiffness from the lightest node, 0.8 MN/m on 90 kg frame
nodes, and the carriage was soft), its mass 45 / 10 / 45 % at the back plate, mid and
rear frames; shear diagonals from each face node to the neighbouring back-plate
nodes yielding at a tenth of the column's crush force (the column carries the rest,
so a node still crushes at σ·A; without them the columns tilted and the face folded
flat); a densification stop behind the bumper at 20 % of the full depth.

Check (`engine-softbody/tests/crash_rigs.rs`): mass and centre of mass to the
standard's; the face's crush force is bounded by σ·A = 310 kPa × 1.676 × 0.559 m =
290 kN (+5 %); with its front skin held and the carriage at 8 m/s, the mean over the
crush is 101 % of σ·A, the carriage's length within 1.3 % and its back plate within
2.3 cm of flat. Into the procedural sedan (GPU): the carriage within 0.9 %, the face
crushed at most 48 %. Into the showcase car (with and without full contact) the
car diverges about 5 frames after the barrier meets it (its nodes reach 82 m/s, then
921 m/s; the barrier follows through contact): **open**, reported by
`explicit_scenes.rs`. Into the procedural sedan's left side
(`engine-world-wasm/tests/explicit_scenes.rs`, GPU): the barrier strikes at about
0.6 s, the car leaves at 2.9 m/s, the barrier at 1.5 m/s across (rebounding), with
0.26 m of crush and intrusion between their centres of mass; no pass-through.

**Door pull** (`doorpull`): the struck-side (left) front door's latch edge, the skin
nodes nearest a point at 60 % of the body's height and 0.2 m ahead of the wheelbase's
middle (**Choice**), pulled outward by one rig node (5 kg) on 200 kN/m cables, its
grab target moving at `pullSpeed` (default 0.5 m/s) on a `pullStiffness` spring
(default 100 kN/m). It loads the door's hinges and latch together as FMVSS 206's
pull does; it is not the 206 rig (that pulls the hinge and latch separately, with
specified loads).
