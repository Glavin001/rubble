# BeamNG authoring census (2026-11-06, measured)

**Decision: measurement only.** No engine or authoring code changed. This page
measures how five real BeamNG cars are authored, puts sedan1 beside them, and
ends with the rules the generator should adopt. It does not adopt them.

The cars:
- **BX**: BeamNG's BX, its slot defaults (the hatch body).
- **ETK K**: the community ETK K-series, `kr6.pc`.
- **CCF**: the community CCF, `f4_sport_M.pc`.
- **Vertex**: the community Gavril Vertex, `gavril_vertex.pc`.
- **fullsuv**: the community full-size SUV, `mid-sport.pc` (body on frame).
- **sedan1**: Rubble's generated car, Medium, at 645d0b9a, from its `.truck` as
  the solver loads it.

Clean room: only the cars' `.jbeam` rows (as data) and their `.dae` geometry
were read. No BeamNG or RoR code, no Lua.

## How to run it

`engine/crates/engine-softbody/tests/authoring_census.rs`, opt-in and ignored:

```bash
RUBBLE_CENSUS_CARS="bx=<dir>/vehicles,bx,;etkk=<dir>/vehicles,etkk,kr6.pc;..." \
  cargo test --release -p engine-softbody --test authoring_census -- --ignored --nocapture authoring_census
```

- Each entry is `label=<vehicles dir>,<vehicle>,<.pc or empty>`. A car whose
  directory is absent is skipped; with no variable only sedan1 runs.
- `census_parts` lists every part with its slot, category, nodes and mass.
- `authoring_census` prints tagged lines: `[part]`, `[beam]`, `[types]`,
  `[hold]`, `[tris]`, `[flex]`, `[mass]`.

Method:
- **Parts** are mapped to categories by part and slot names (`door`, `hood`,
  `trunk`/`tailgate`/`boot`, `fender`/`wing`, `bumper`, `frame`/`radsupport`,
  `body`). A door and its inner panel are one door (the BX keeps its door's
  structure in `doorpanel`). Numbers are per part instance (per door, per
  fender).
- **Beams of a part** are the rows that part defines (sedan1: beams whose first
  node is the part's). Structural beams are NORMAL, SUPPORT and ANISOTROPIC.
- **ε_y** = beamDeform / (k·L); springback after 20 % plastic strain is
  ε_y / (ε_y + 0.2); **ε_b** = beamStrength / (k·L) over finite strengths;
  **ζ** = c / 2√(k·μ), μ the pair's reduced mass.
- **Spring ends per node** count every spring beam at the node, from any part.
- **Row** is the node's Gershgorin row over sedan1's step bound,
  Σ (h²k + 2hc)(1/mᵢ + 1/√(mᵢmⱼ)) / (`G_LIMIT`·h²) at h = 0.5 ms. sedan1 is
  authored to row ≤ 1. "At the bound" is the share of nodes with row > 0.95.
  (The BX's body gives 82 % here and 74 % in `beamng_topology_census`, which
  uses the solver's beams after loading; the BX's configuration also differs:
  that test uses the coupe.)

## 1. Per part


### Body shell

| Car | nodes | kg | node mass p10 / p50 / p90 (kg) | spacing p50 (m) | spring ends p50 | at the bound | row p50 / p90 / max |
|---|---|---|---|---|---|---|---|
| BX | 111 | 257.2 | 0.30 / 2.50 / 4.00 | 0.184 | 17 | 82 % | 1.51 / 2.64 / 3.86 |
| ETK K | 124 | 515.0 | 2.60 / 4.00 / 5.50 | 0.175 | 15 | 74 % | 1.28 / 2.21 / 3.15 |
| CCF | 138 | 425.8 | 1.50 / 3.00 / 5.00 | 0.159 | 18 | 85 % | 1.32 / 2.14 / 2.74 |
| Vertex | 118 | 455.3 | 1.40 / 3.70 / 7.00 | 0.216 | 18 | 80 % | 1.40 / 2.04 / 4.12 |
| fullsuv | 96 | 265.9 | 2.00 / 2.44 / 3.90 | 0.287 | 18 | 72 % | 1.44 / 2.47 / 3.49 |
| **sedan1** | 592 | 798.5 | 0.46 / 1.02 / 2.52 | 0.154 | 12 | 87 % | 1.00 / 1.00 / 1.25 |

| Car | beams | L p50 (m) | k p50 (N/m) | k·L p50 (N) | deform p50 (N) | ε_y p10 / p50 / p90 (%) | springback at 20 % | FLT_MAX strength | finite strength p50 (N) | ε_b p50 (%) | ζ p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BX | 850 | 0.563 | 1.7e+06 | 8.2e+05 | 8e+03 | 0.23 / 1.00 / 4.88 | 5 % | 97 % | 5e+04 | 3.7e+02 | 0.028 |
| ETK K | 677 | 0.461 | 1.6e+06 | 9.4e+05 | 1.4e+04 | 0.29 / 1.33 / 9.29 | 6 % | 98 % | 5e+04 | 9.3e+02 | 0.033 |
| CCF | 863 | 0.526 | 1.5e+06 | 8e+05 | 1.1e+04 | 0.18 / 1.32 / 6.35 | 6 % | 98 % | 2e+04 | 6.1e+02 | 0.041 |
| Vertex | 895 | 0.606 | 2e+06 | 1e+06 | 1.4e+04 | 0.30 / 1.30 / 7.80 | 6 % | 97 % | 1.1e+04 | 69 | 0.036 |
| fullsuv | 665 | 0.681 | 1.9e+06 | 1.1e+06 | 1.1e+04 | 0.19 / 0.97 / 11.23 | 5 % | 94 % | 3.4e+04 | 8 | 0.038 |
| **sedan1** | 3992 | 0.359 | 5.2e+05 | 2.1e+05 | 5.7e+03 | 0.67 / 2.95 / 30.12 | 13 % | 3 % | 1.2e+05 | 68 | 0.027 |

### Doors (per door)

| Car | nodes | kg | node mass p10 / p50 / p90 (kg) | spacing p50 (m) | spring ends p50 | at the bound | row p50 / p90 / max |
|---|---|---|---|---|---|---|---|
| BX | 11 | 16.9 | 1.30 / 1.50 / 1.50 | 0.253 | 11 | 73 % | 1.09 / 1.44 / 2.10 |
| ETK K | 13 | 21.5 | 1.50 / 1.70 / 1.70 | 0.225 | 12 | 69 % | 1.06 / 1.73 / 2.64 |
| CCF | 13 | 18.1 | 1.20 / 1.20 / 1.90 | 0.216 | 13 | 92 % | 1.59 / 2.02 / 2.12 |
| Vertex | 10 | 18.0 | 1.20 / 1.40 / 1.70 | 0.276 | 13 | 80 % | 1.26 / 2.02 / 2.13 |
| fullsuv | 9 | 21.1 | 1.87 / 1.88 / 5.64 | 0.307 | 12 | 81 % | 1.38 / 2.03 / 2.31 |
| **sedan1** | 26 | 26.6 | 0.44 / 0.96 / 1.60 | 0.180 | 12 | 92 % | 1.00 / 1.00 / 1.02 |

| Car | beams | L p50 (m) | k p50 (N/m) | k·L p50 (N) | deform p50 (N) | ε_y p10 / p50 / p90 (%) | springback at 20 % | FLT_MAX strength | finite strength p50 (N) | ε_b p50 (%) | ζ p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BX | 142 | 0.580 | 1.1e+06 | 6.4e+05 | 8e+03 | 0.49 / 1.25 / 3.95 | 6 % | 54 % | 2e+05 | 30 | 0.033 |
| ETK K | 152 | 0.540 | 1e+06 | 5.4e+05 | 7e+03 | 0.42 / 1.42 / 6.96 | 7 % | 74 % | 3e+04 | 2 | 0.033 |
| CCF | 174 | 0.454 | 1e+06 | 4e+05 | 1.5e+04 | 0.92 / 3.35 / 18.30 | 14 % | 67 % | 2e+05 | 44 | 0.087 |
| Vertex | 384 | 0.514 | 1.1e+06 | 6.3e+05 | 9e+03 | 0.33 / 1.57 / 9.82 | 7 % | 61 % | 1e+05 | 19 | 0.031 |
| fullsuv | 311 | 0.601 | 1.9e+06 | 1e+06 | 1.3e+04 | 0.61 / 1.39 / 3.91 | 7 % | 68 % | 9.8e+04 | 19 | 0.053 |
| **sedan1** | 708 | 0.288 | 4.8e+05 | 1.3e+05 | 5.2e+03 | 0.77 / 3.76 / 78.35 | 16 % | 0 % | 7.5e+04 | 71 | 0.034 |

### Bonnet

| Car | nodes | kg | node mass p10 / p50 / p90 (kg) | spacing p50 (m) | spring ends p50 | at the bound | row p50 / p90 / max |
|---|---|---|---|---|---|---|---|
| BX | 24 | 14.5 | 0.55 / 0.55 / 0.80 | 0.240 | 10 | 92 % | 1.19 / 1.63 / 2.11 |
| ETK K | 23 | 34.8 | 1.50 / 1.50 / 1.60 | 0.294 | 8 | 4 % | 0.60 / 0.72 / 2.25 |
| CCF | 21 | 22.3 | 0.95 / 0.95 / 1.60 | 0.292 | 10 | 95 % | 1.42 / 1.87 / 2.58 |
| Vertex | 14 | 11.4 | 0.60 / 0.80 / 0.80 | 0.343 | 10 | 79 % | 1.16 / 1.70 / 2.33 |
| fullsuv | 30 | 36.8 | 1.20 / 1.20 / 1.20 | 0.250 | 12 | 73 % | 1.17 / 1.53 / 2.42 |
| **sedan1** | 26 | 17.9 | 0.50 / 0.53 / 0.62 | 0.208 | 13 | 96 % | 1.00 / 1.01 / 1.01 |

| Car | beams | L p50 (m) | k p50 (N/m) | k·L p50 (N) | deform p50 (N) | ε_y p10 / p50 / p90 (%) | springback at 20 % | FLT_MAX strength | finite strength p50 (N) | ε_b p50 (%) | ζ p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BX | 148 | 0.436 | 5.2e+05 | 2.2e+05 | 1.4e+04 | 1.34 / 5.53 / 24.23 | 22 % | 68 % | 1.2e+06 | 5.3e+02 | 0.059 |
| ETK K | 106 | 0.457 | 8e+05 | 3.4e+05 | 9e+03 | 0.20 / 3.67 / 11.77 | 16 % | 72 % | 1.1e+06 | 3.7e+02 | 0.032 |
| CCF | 146 | 0.454 | 1e+06 | 4.4e+05 | 8e+03 | 0.49 / 2.51 / 15.96 | 11 % | 55 % | 1e+05 | 81 | 0.044 |
| Vertex | 171 | 0.463 | 7e+05 | 3.2e+05 | 1.3e+04 | 0.66 / 4.54 / 19.81 | 19 % | 49 % | 7.8e+04 | 34 | 0.057 |
| fullsuv | 191 | 0.560 | 7.9e+05 | 4.7e+05 | 6.8e+03 | 0.24 / 1.46 / 13.13 | 7 % | 86 % | 1e+06 | 1.5e+02 | 0.071 |
| **sedan1** | 189 | 0.355 | 2.8e+05 | 9.4e+04 | 3e+03 | 1.15 / 3.07 / 5.71 | 13 % | 5 % | 3.7e+04 | 46 | 0.059 |

### Boot lid / tailgate

| Car | nodes | kg | node mass p10 / p50 / p90 (kg) | spacing p50 (m) | spring ends p50 | at the bound | row p50 / p90 / max |
|---|---|---|---|---|---|---|---|
| BX | 22 | 14.8 | 0.60 / 0.60 / 1.10 | 0.203 | 14 | 77 % | 1.44 / 2.00 / 2.16 |
| ETK K | 15 | 13.9 | 0.90 / 0.90 / 0.90 | 0.243 | 9 | 100 % | 1.27 / 1.61 / 1.76 |
| CCF | 12 | 11.6 | 0.80 / 0.80 / 1.60 | 0.280 | 11 | 100 % | 1.38 / 1.88 / 2.06 |
| Vertex | 17 | 13.6 | 0.80 / 0.80 / 0.80 | 0.212 | 12 | 94 % | 1.46 / 1.62 / 2.17 |
| fullsuv | 12 | 18.7 | 1.17 / 1.60 / 2.00 | 0.295 | 12 | 50 % | 1.13 / 1.91 / 1.95 |
| **sedan1** | 16 | 12.4 | 0.58 / 0.64 / 0.91 | 0.215 | 11 | 94 % | 1.00 / 1.00 / 1.01 |

| Car | beams | L p50 (m) | k p50 (N/m) | k·L p50 (N) | deform p50 (N) | ε_y p10 / p50 / p90 (%) | springback at 20 % | FLT_MAX strength | finite strength p50 (N) | ε_b p50 (%) | ζ p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BX | 190 | 0.434 | 5e+05 | 2.2e+05 | 7e+03 | 0.73 / 2.98 / 12.13 | 13 % | 69 % | 1.2e+06 | 4.8e+02 | 0.057 |
| ETK K | 84 | 0.389 | 1e+06 | 3.4e+05 | 1.6e+04 | 0.35 / 4.59 / 15.70 | 19 % | 68 % | 1.1e+06 | 4.6e+02 | 0.047 |
| CCF | 129 | 0.424 | 8e+05 | 3.1e+05 | 3.8e+04 | 3.55 / 13.73 / 44.91 | 41 % | 81 % | 3.5e+04 | 2.8e+02 | 0.053 |
| Vertex | 122 | 0.413 | 8e+05 | 3e+05 | 1.4e+04 | 0.97 / 5.22 / 19.84 | 21 % | 64 % | 1.2e+06 | 4.5e+02 | 0.053 |
| fullsuv | 208 | 0.545 | 9e+05 | 5e+05 | 7.5e+03 | 0.70 / 2.08 / 9.73 | 9 % | 76 % | 3.4e+04 | 8.9 | 0.060 |
| **sedan1** | 108 | 0.346 | 3.6e+05 | 1.2e+05 | 3e+03 | 0.66 / 2.13 / 4.76 | 10 % | 19 % | 4.2e+04 | 33 | 0.057 |

### Fenders (per fender)

| Car | nodes | kg | node mass p10 / p50 / p90 (kg) | spacing p50 (m) | spring ends p50 | at the bound | row p50 / p90 / max |
|---|---|---|---|---|---|---|---|
| BX | 10 | 4.2 | 0.40 / 0.40 / 0.40 | 0.237 | 14 | 100 % | 1.53 / 1.66 / 1.99 |
| ETK K | 13 | 6.5 | 0.50 / 0.50 / 0.50 | 0.235 | 10 | 100 % | 1.37 / 1.45 / 1.50 |
| CCF | 10 | 8.6 | 0.80 / 0.80 / 0.80 | 0.227 | 9 | 95 % | 1.20 / 1.56 / 2.09 |
| Vertex | 12 | 9.1 | 0.75 / 0.75 / 0.75 | 0.264 | 11 | 100 % | 1.57 / 1.85 / 1.85 |
| fullsuv | 10 | 8.8 | 0.80 / 0.80 / 0.80 | 0.193 | 10 | 80 % | 1.13 / 1.62 / 1.78 |
| **sedan1** | 10 | 5.1 | 0.49 / 0.53 / 0.55 | 0.198 | 10 | 80 % | 1.00 / 1.00 / 1.00 |

| Car | beams | L p50 (m) | k p50 (N/m) | k·L p50 (N) | deform p50 (N) | ε_y p10 / p50 / p90 (%) | springback at 20 % | FLT_MAX strength | finite strength p50 (N) | ε_b p50 (%) | ζ p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BX | 208 | 0.366 | 4e+05 | 1.5e+05 | 7e+03 | 1.95 / 4.55 / 11.84 | 19 % | 62 % | 4e+04 | 48 | 0.042 |
| ETK K | 136 | 0.410 | 4e+05 | 1.8e+05 | 6e+03 | 1.19 / 4.43 / 13.29 | 18 % | 62 % | 1.2e+04 | 10 | 0.071 |
| CCF | 101 | 0.386 | 1e+06 | 3.6e+05 | 5e+03 | 0.47 / 1.67 / 7.06 | 8 % | 55 % | 4.5e+04 | 14 | 0.037 |
| Vertex | 162 | 0.431 | 8e+05 | 3.9e+05 | 9.9e+03 | 0.92 / 2.54 / 6.26 | 11 % | 52 % | 3.3e+05 | 59 | 0.040 |
| fullsuv | 104 | 0.531 | 8e+05 | 4.3e+05 | 6e+03 | 0.43 / 1.36 / 4.56 | 6 % | 58 % | 4.5e+04 | 13 | 0.039 |
| **sedan1** | 118 | 0.264 | 4e+05 | 1e+05 | 2.9e+03 | 1.55 / 2.40 / 3.95 | 11 % | 0 % | 3.9e+04 | 41 | 0.042 |

### Front bumper

| Car | nodes | kg | node mass p10 / p50 / p90 (kg) | spacing p50 (m) | spring ends p50 | at the bound | row p50 / p90 / max |
|---|---|---|---|---|---|---|---|
| BX | 22 | 11.9 | 0.30 / 0.30 / 1.20 | 0.121 | 13 | 100 % | 1.37 / 1.78 / 2.00 |
| ETK K | 25 | 17.5 | 0.70 / 0.70 / 0.70 | 0.157 | 11 | 76 % | 1.18 / 1.35 / 1.71 |
| CCF | 31 | 29.4 | 0.70 / 0.70 / 1.45 | 0.142 | 13 | 87 % | 1.41 / 1.78 / 2.52 |
| Vertex | 24 | 9.1 | 0.10 / 0.30 / 0.60 | 0.143 | 10 | 60 % | 1.24 / 1.79 / 2.15 |
| fullsuv | 38 | 36.0 | 0.90 / 0.90 / 1.10 | 0.141 | 12 | 58 % | 1.07 / 1.44 / 1.94 |
| **sedan1** | 14 | 2.6 | 0.09 / 0.18 / 0.28 | 0.203 | 9 | 86 % | 1.00 / 1.01 / 1.01 |

| Car | beams | L p50 (m) | k p50 (N/m) | k·L p50 (N) | deform p50 (N) | ε_y p10 / p50 / p90 (%) | springback at 20 % | FLT_MAX strength | finite strength p50 (N) | ε_b p50 (%) | ζ p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BX | 374 | 0.336 | 2e+05 | 4.6e+04 | 5.6e+03 | 2.96 / 12.09 / 95.08 | 38 % | 24 % | 3.4e+04 | 72 | 0.089 |
| ETK K | 175 | 0.291 | 6e+05 | 1.2e+05 | 5.5e+03 | 3.18 / 5.80 / 12.45 | 22 % | 0 % | 5.3e+04 | 48 | 0.065 |
| CCF | 210 | 0.394 | 5e+05 | 1.1e+05 | 4.8e+03 | 0.70 / 2.66 / 28.65 | 12 % | 40 % | 5.9e+04 | 86 | 0.095 |
| Vertex | 310 | 0.342 | 1.5e+05 | 4e+04 | 8.4e+03 | 2.23 / 21.29 / 228.35 | 52 % | 26 % | 2.5e+04 | 55 | 0.173 |
| fullsuv | 268 | 0.487 | 1.5e+05 | 9.6e+04 | 9.8e+03 | 2.09 / 7.42 / 28.17 | 27 % | 42 % | 3.9e+04 | 51 | 0.288 |
| **sedan1** | 59 | 0.225 | 1.3e+05 | 3.4e+04 | 1.4e+03 | 1.82 / 4.65 / 12.32 | 19 % | 0 % | 1.4e+04 | 42 | 0.090 |

### Rear bumper

| Car | nodes | kg | node mass p10 / p50 / p90 (kg) | spacing p50 (m) | spring ends p50 | at the bound | row p50 / p90 / max |
|---|---|---|---|---|---|---|---|
| BX | 24 | 7.2 | 0.30 / 0.30 / 0.30 | 0.196 | 13 | 79 % | 1.19 / 2.14 / 2.53 |
| ETK K | 19 | 9.5 | 0.50 / 0.50 / 0.50 | 0.079 | 10 | 58 % | 1.34 / 1.50 / 1.81 |
| CCF | 38 | 28.8 | 0.50 / 0.50 / 1.45 | 0.157 | 9 | 84 % | 1.27 / 1.62 / 2.49 |
| Vertex | 29 | 14.4 | 0.30 / 0.60 / 0.60 | 0.141 | 10 | 72 % | 1.12 / 1.79 / 2.05 |
| fullsuv | 23 | 16.1 | 0.70 / 0.70 / 0.70 | 0.224 | 12 | 91 % | 1.13 / 1.29 / 1.59 |
| **sedan1** | 10 | 1.5 | 0.13 / 0.15 / 0.15 | 0.243 | 9 | 80 % | 1.00 / 1.00 / 1.01 |

| Car | beams | L p50 (m) | k p50 (N/m) | k·L p50 (N) | deform p50 (N) | ε_y p10 / p50 / p90 (%) | springback at 20 % | FLT_MAX strength | finite strength p50 (N) | ε_b p50 (%) | ζ p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BX | 230 | 0.359 | 1e+05 | 4e+04 | 5e+03 | 2.10 / 14.53 / 77.64 | 42 % | 7 % | 8e+04 | 1.1e+02 | 0.091 |
| ETK K | 104 | 0.343 | 2.2e+05 | 7.9e+04 | 5.5e+03 | 2.50 / 9.56 / 43.97 | 32 % | 18 % | 6e+04 | 74 | 0.213 |
| CCF | 209 | 0.349 | 5e+05 | 1.3e+05 | 7.5e+03 | 1.22 / 4.27 / 211.78 | 18 % | 41 % | 5.9e+04 | 56 | 0.085 |
| Vertex | 207 | 0.392 | 2e+05 | 7.3e+04 | 9.8e+03 | 3.93 / 11.15 / 40.34 | 36 % | 12 % | 5.9e+04 | 59 | 0.173 |
| fullsuv | 154 | 0.452 | 1.5e+05 | 6.4e+04 | 4.8e+03 | 2.37 / 11.14 / 39.64 | 36 % | 19 % | 5.9e+04 | 69 | 0.326 |
| **sedan1** | 58 | 0.238 | 1.2e+05 | 1.9e+04 | 1.4e+03 | 1.52 / 2.66 / 13.40 | 12 % | 0 % | 1.1e+04 | 38 | 0.090 |

### Rails, subframe, frame

| Car | nodes | kg | node mass p10 / p50 / p90 (kg) | spacing p50 (m) | spring ends p50 | at the bound | row p50 / p90 / max |
|---|---|---|---|---|---|---|---|
| BX | 7 | 10.5 | 1.50 / 1.50 / 1.50 | 0.241 | 20 | 100 % | 2.05 / 2.06 / 2.31 |
| fullsuv | 52 | 609.3 | 8.95 / 12.90 / 13.28 | 0.160 | 21 | 65 % | 1.25 / 1.96 / 2.55 |

| Car | beams | L p50 (m) | k p50 (N/m) | k·L p50 (N) | deform p50 (N) | ε_y p10 / p50 / p90 (%) | springback at 20 % | FLT_MAX strength | finite strength p50 (N) | ε_b p50 (%) | ζ p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BX | 49 | 0.474 | 1.7e+06 | 8e+05 | 4e+03 | 0.17 / 0.47 / 1.85 | 2 % | 100 % | nan | nan | 0.032 |
| fullsuv | 517 | 0.661 | 3.8e+06 | 2.1e+06 | 2.1e+04 | 0.26 / 1.02 / 8.36 | 5 % | 85 % | 1.4e+05 | 8.4 | 0.026 |

**What the part tables say:**
- **Fewer, heavier nodes on the body and doors.** Body: 96–138 nodes at
  2.4–4.0 kg (p50) against sedan1's 592 at 1.0 kg. Door: 9–13 nodes at
  1.2–1.9 kg against 26 at 0.96 kg. The lids, fenders and bumpers are close to
  sedan1 in count and mass.
- **Longer, stiffer beams.** Body L 0.46–0.68 m and k 1.5–2.0e6 N/m, so k·L
  0.8–1.1e6 N: **4–5× sedan1's 2.1e5**. Doors k·L 0.40–1.05e6: **3–8×**
  sedan1's 1.3e5. Bonnet and boot 2–4×, fenders 1.5–4×.
- **Low yield strain on the structure.** Body ε_y 1.0–1.3 % (sedan1 3.0 %),
  so springback after 20 % crush is 5–6 % (sedan1 13 %). Doors 1.3–1.6 % (the
  CCF 3.4 %; sedan1 3.8 %). The skins are higher: bonnet 1.5–5.5 %, boot
  2–5 % (CCF 14 %), fenders 1.4–4.6 %.
- **Bumpers are soft, foam-like**: k·L 0.4–1.3e5, ε_y 3–21 %, springback
  12–52 %, ζ 0.07–0.33.
- **The structure does not break.** 94–98 % of body beams have FLT_MAX
  strength (sedan1: 3 %); doors 54–74 %, lids 49–86 %. Separation happens at
  break groups (hinges, latches, panel mounts), not inside a panel.
- **Every BeamNG part is past sedan1's bound.** Row p50 1.1–1.6, p90 1.4–2.6
  and max 2–4 (one exception, the ETK's bonnet at 0.6). sedan1 is capped at
  exactly 1.00 on every part. Rubble's JBeam mode runs all five cars at these
  rows; the ETK is the car the user just saw in the crash lab.
- **Damping per beam is similar in ratio.** ζ p50 0.03–0.04 on bodies and
  doors, like sedan1's 0.027–0.034. In N·s/m the BeamNG beams are heavier
  (body c 80–130 against 27) because their nodes are.

## 2. Beam types and options

Shares of all beam rows (from `[types]`):

| Car | NORMAL | SUPPORT | BOUNDED | ANISOTROPIC | damping-only | LBEAM | torsion bars | in a break group | deformLimitExpansion | precompression | FLT_MAX strength |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BX | 80.9 % | 16.9 % | 2.2 % | 0 | 0 | 0 | 13 | 31.8 % | 27.6 % | 10.8 % | 55 % |
| ETK K | 85.4 % | 10.9 % | 2.4 % | 1.2 % | 0 | 0 | 41 | 26.1 % | 34.6 % | 6.0 % | 57 % |
| CCF | 92.0 % | 6.3 % | 1.7 % | 0 | 0 | 0 | 39 | 16.7 % | 30.4 % | 2.7 % | 62 % |
| Vertex | 83.3 % | 14.5 % | 2.2 % | 0 | 0 | 0 | 8 | 18.8 % | 27.2 % | 6.8 % | 58 % |
| fullsuv | 89.5 % | 8.9 % | 1.7 % | 0 | 0 | 0 | 25 | 23.1 % | 26.1 % | 4.4 % | 63 % |
| **sedan1** | 66.4 % | 4.6 % | 1.4 % | 0 | **27.6 %** | 0 | 0 | 0.5 % | 0 | 0 | 21 % |

- **No damping-only beams.** sedan1 spends 28 % of its beams (30 % of the
  body's) on dashpots; BeamNG damps every spring beam and adds no dashpots.
  Each dashpot takes part of a node's row (the 2hc term) without stiffness.
- **SUPPORT** is 5–7 % of the body (fullsuv 0), 6–45 % of doors, 9–38 % of
  bonnets, 8–25 % of boot lids: the seals and stops between a closure and its
  opening.
- **deformLimitExpansion** is set on 15–49 % of beams (body 37–49 %), always
  1.1 or 1.2. sedan1's `.truck` carries none. deformLimit and
  deformLimitStress are not used by any of the five.
- **beamPrecompression** is on 3–11 % of beams, mostly door and lid SUPPORT
  stops (the BX's door rows 46 %): the closure is preloaded against its stops.
- **LBEAM**: none. **Torsion bars**: 8–41, all in the suspension and
  steering, none on the body.
- **deformGroup** (glass and lamp breakage triggers) is on 10–18 % of beams.

## 3. How doors, bonnet and boot are held

From `[hold]`: every beam from a closure's nodes to the rest of the car
(its own glass excluded), grouped by break group.

| | Hinges | Latch | Seals and stops |
|---|---|---|---|
| **Door** (BeamNG, 5 cars) | 2 break groups (upper `a`, lower `b`; the CCF one) of 4–14 beams each, 11–22 per door; NORMAL plus 3–4 BOUNDED per upper hinge; k 0.5–3e6 N/m, deform 10–77 kN, strength 30–98 kN per beam; finite sum 0.6–2.4 MN per door | 1 coupler: `couplerNodes` at 35–85 kN (BX, CCF, fullsuv) or a node `couplerStrength` 35–40 kN (ETK, Vertex) | 4–30 SUPPORT beams to the opening, strength 100–280 kN, ungrouped or in their own `*_supportBeams` group; often precompressed |
| **Door** (sedan1) | 2 groups of 2–3 NORMAL beams; k 3.1–4.0e5; strength = deform 33–56 kN; sum 0.07–0.09 MN per hinge | 1–3 NORMAL beams at 16–32 kN plus a secondary of 2 beams at 19–21 kN | 49–57 SUPPORT beams (strength ~57 kN) and 22–24 dashpots |
| **Bonnet** (BeamNG) | 2 groups of 2–7 beams; the BX and Vertex hinge beams are soft (k 8e3), the others 0.6–1.2e6; strength 35–155 kN | 2 coupler nodes at 30 kN (latch and catch) | 7 BOUNDED rest bumpers and 6–33 SUPPORT rests, strength 90 kN–1.25 MN |
| **Bonnet** (sedan1) | 2 groups of 2 beams, k 3.2e5, 36 kN | 2 beams, 29 kN | 53 SUPPORT, 25 dashpots |
| **Boot** (BeamNG) | 2 groups of 5–7 beams (fullsuv's lower tailgate one group of 30), strength 10–35 kN | 1 coupler node 40–50 kN, or 2 `couplerNodes` 20–40 kN (fullsuv) | 10–30 SUPPORT rests, strength up to 1.25 MN |
| **Boot** (sedan1) | 2 groups of 2 beams, 47–48 kN | 2 beams, 36 kN | 43 SUPPORT, 22 dashpots |

Break groups are small and named per hinge: 5–14 beams. A door's mirror is
its own group of 12–22 beams.

## 4. Collision triangles and contacters

| Car | triangles (collidable) | body: triangle edge p50 | body: triangle area p50 | nodes on a triangle | contacters |
|---|---|---|---|---|---|
| BX | 599 (560) | 0.43 m | 0.063 m² | 73 % | 90 % |
| ETK K | 519 (476) | 0.36 m | 0.045 m² | 70 % | 90 % |
| CCF | 529 (483) | 0.39 m | 0.050 m² | 60 % | 89 % |
| Vertex | 556 (531) | 0.45 m | 0.069 m² | 74 % | 92 % |
| fullsuv | 611 (571) | 0.54 m | 0.079 m² | 68 % | 88 % |
| **sedan1** | 944 (944) | 0.29 m | 0.029 m² | 46 % | 62 % |

- Each triangle spans the part's own nodes: one triangle per 2–3 nodes. Doors
  carry 20–29 triangles each; sedan1's doors 140 in all.
- 77–100 % of every panel's nodes are on a triangle.
- The engine, suspension and wheel nodes are contacters with no triangle
  (64–95 % of chassis nodes). They hit the world and other cars' triangles but
  carry no surface.
- 61–74 % of BeamNG nodes set `selfCollision` (sedan1's `.truck` does not
  carry the flag, so it is not compared).

## 5. Flexbodies against their nodes

From `[flex]`, panel meshes only (≥ 100 vertices). A mesh's panel nodes are its
groups' nodes within 10 cm of the mesh. The "even" columns give the same
distance for points spread evenly over the mesh: where a node placed at random
on the panel would sit. A crease is an edge between faces more than 30° apart.
The meshes are closed (hemmed), so a panel edge shows as a crease.

| Part | vertex → nearest node p50 (BeamNG / sedan1) | panel nodes per m² | node → surface p50 | node → crease p50 (even) BeamNG | node → crease p50 (even) sedan1 |
|---|---|---|---|---|---|
| body | 0.11–0.18 m / 0.10 m | 4–25 / 35 | 2.0–3.9 cm / 4.9 cm | 3.2–5.2 cm (3.2–7.9) | 8.1 cm (2.1) |
| doors | 0.13–0.20 / 0.09 | 5.5–9.6 / 31 | 2.7–4.4 / 4.5 | 3.1–5.3 (2.0–6.6) | 7.7 (1.2) |
| bonnet | 0.08–0.16 / 0.11 | 4.5–11 / 20 | 1.7–2.3 / 2.2 | **2.3–4.0 (5.4–19.3)** | 7.7 (1.2) |
| boot | 0.10–0.13 / 0.11 | 7–11 / 20 | 1.0–3.1 / 1.6 | **1.2–3.4 (1.0–5.6)** | 3.7 (1.1) |
| fenders | 0.09–0.13 / 0.09 | 8.7–15 / 25 | 1.7–3.4 / 2.0 | **2.3–4.0 (4.9–6.0)** | 8.0 (3.5) |
| bumpers | 0.08–0.13 / 0.11–0.13 | 6–63 / 23–51 | 0.8–3.7 / 7.1–7.8 | **0.8–4.5 (2.6–5.7)** | 7.4–7.8 (1.4–1.6) |

- **Nodes sit on the panel's creases and edges** (lids, fenders, bumpers;
  the body and doors are no nearer than random). On every lid, fender and
  bumper the BeamNG nodes are 1–4 cm from a crease, against 3–19 cm for a
  random place on the panel. On the bonnet the gap is 5×. The node lattice
  is drawn on the panel's own feature lines: its edges, its shut lines and its
  style lines.
- **Nodes sit on the surface**: 1–4 cm off the mesh, the panel's thickness.
  sedan1's body and bumper nodes are 5–8 cm off it.
- **Few nodes per panel**: 4.5–11 per m² on lids and doors, so each vertex
  rides nodes 8–20 cm away. sedan1 has 20–35 per m² and its vertices are no
  closer (9–13 cm): the mesh does not gain from the denser lattice.
- **Each flexbody binds to its own part's group**, one group per mesh (the
  body 1–3: its shell plus shock tops or pillars). A door mesh binds to 9–19
  nodes, its door and inner panel and window frame.
- sedan1's own mesh is crease-dense (an even point is 1–3.5 cm from a crease),
  so its nodes (4–8 cm from one) are off its feature lines rather than on them.

## 6. Mass

| Car | total | node mass p10 / p50 / p90 | body shell | body + panels + glass | powertrain | chassis (suspension, wheels) | nodes ≥ 10 kg |
|---|---|---|---|---|---|---|---|
| BX | 1,139 kg | 0.30 / 1.05 / 4.75 | 257 (23 %) | 44 % | 18 % | 25 % | 10 nodes, 15 % |
| ETK K | 1,345 | 0.45 / 1.50 / 6.00 | 515 (38 %) | 54 % | 21 % | 24 % | 17, 23 % |
| CCF | 1,385 | 0.50 / 1.20 / 4.50 | 426 (31 %) | 53 % | 20 % | 21 % | 15, 21 % |
| Vertex | 1,245 | 0.40 / 1.40 / 6.10 | 455 (37 %) | 61 % | 14 % | 22 % | 9, 9 % |
| fullsuv | 2,954 | 0.70 / 2.00 / 12.90 | 266 + 609 frame (30 %) | 67 % | 12 % | 16 % | 93, 53 % |
| **sedan1** | 1,450 | 0.48 / 0.94 / 2.64 | 798 (55 %) | 65 % | 20 % | 7 % | 0 |

- **The engine is about 10 lumped nodes** of 10–23 kg (104–233 kg). The
  transmission is 1–13 nodes, the differential 2–3 heavy nodes.
- **The suspension carries a fifth to a quarter of the mass**: 100–215 kg per
  axle, on 17–28 nodes. sedan1 puts 7 % there and more into the body shell.
- The fullsuv also hangs 400 kg of ballast on 16 side-skirt nodes.

## 7. What each car does that sedan1 does not

The five cars share their method; their differences are in degree. Against
sedan1, what most plausibly explains each quality:

**Solidity: little ripple, low springback.**
1. **Heavy nodes on long beams.** The body's node is 2.4–4 kg on 0.46–0.68 m
   beams; the door's 1.2–1.9 kg on 0.45–0.60 m. At the step's bound
   k ∝ m/(h²·n), so k·L ∝ m·L/n: 4–5× sedan1's on the body, 3–8× on the doors.
   Their lowest panel modes are few and stiff.
2. **Rows past sedan1's cap.** Every part runs at 1.1–1.6× sedan1's bound at
   the median and up to 2–4× at the stiffest node, and runs in Rubble's JBeam
   mode. sedan1 is clamped at 1.0 everywhere, so the same node masses would
   give it ~1.4× less stiffness than BeamNG's authors use.
3. **No dashpots.** BeamNG gives every node's row to spring beams with their
   own damping (ζ 0.03–0.05). sedan1's 28 % dashpots take row without adding
   stiffness.
4. **Yield at about 1 %.** Body ε_y 1.0–1.3 % gives 5–6 % springback; sedan1's
   3 % gives 13 %.
5. **Preloaded closures.** Doors and lids sit on 4–30 precompressed SUPPORT
   stops against one strong coupler, so they cannot rattle in their openings.

**Clean crumple.**
1. **The structure never tears**: 94–98 % of body beams and 54–86 % of panel
   beams are FLT_MAX. Crumple is plastic folding.
2. **Yield is low and stretch is capped**: ε_y ~1 % with deformLimitExpansion
   1.1–1.2 on 26–35 % of beams, so folds form early and are not drawn out.
3. **SUPPORT beams across sections** (5–7 % of the body, more on closures)
   resist crush without holding parts together in tension.
4. **Parts leave by break groups**: hinges of 4–14 beams, a coupler latch, a
   mirror of 12–22 beams.

**A mesh that follows the physics.**
1. **Nodes on the panel's feature lines**, 1–4 cm from its creases and shut
   lines and on its surface, so a crease bends where a node is.
2. **A coarse, stiff lattice**: 5–11 nodes per m², each vertex on nodes
   8–20 cm away. With few low modes there is no lattice ripple for the mesh to
   show.
3. **One group per flexbody**, the part's own nodes, so a detached or folded
   panel carries exactly its mesh.

## Authoring rules

Numbers are the range over the five cars (the median of each car's part);
outliers are named. "Row" is the Gershgorin row of the census.

- **Body shell: 100–140 nodes** (96–138), at **2.5–4 kg each** (p50; p10
  0.3–2.6, p90 3.9–7). Spacing 0.16–0.29 m; **beams 0.45–0.70 m** long,
  k 1.5–2.0e6 N/m, **k·L 0.8–1.1e6 N**, 15–18 spring ends per node.
  **beamDeform 8–14 kN, ε_y 1.0–1.3 %**. **Strength FLT_MAX** on 94–98 %.
  ζ 0.03–0.04. SUPPORT 5–7 %. deformLimitExpansion 1.1–1.2 on 37–49 %.
- **Door: 9–13 nodes** (structure and inner panel together; on the
  surface, 3–5 cm from a crease: the doors are not measurably nearer their
  feature lines than random), **1.2–1.9 kg each, 17–22 kg** a
  door. Beams 0.45–0.60 m, k 1.0–1.9e6, **k·L 0.4–1.0e6**, deform 7–15 kN,
  **ε_y 1.3–1.6 %** (CCF 3.4 %), FLT_MAX on 54–74 %.
  - **Hinges**: two break groups (upper, lower) of **4–14 beams each**, k
    0.5–3e6, **30–98 kN per beam**, upper with 3–4 BOUNDED stops.
  - **Latch**: **one coupler, 35–85 kN**.
  - **Seals**: 4–30 SUPPORT stops to the opening, 100–280 kN, preloaded.
- **Bonnet: 14–30 nodes, 0.55–1.5 kg** each (11–37 kg), spacing 0.24–0.34 m,
  k·L 2.2–4.7e5, ε_y 1.5–5.5 %. Two hinge groups of 2–7 beams, 35–155 kN;
  **latch as 2 coupler nodes at 30 kN**; 7 BOUNDED and 6–33 SUPPORT rests.
- **Boot lid: 12–22 nodes, 0.6–1.6 kg** (12–19 kg), k·L 2.2–5e5, ε_y 2–5 %
  (CCF 14 %). Two hinge groups of 5–7 beams at 10–35 kN; **latch a coupler
  at 40–50 kN**; 10–30 SUPPORT rests.
- **Fender: 10–13 nodes, 0.4–0.8 kg** (4–9 kg), k·L 1.5–4.3e5, ε_y 1.4–4.6 %.
- **Bumper cover: 19–38 nodes, 0.3–0.9 kg**, soft: k 1.5–6e5, k·L 0.4–1.3e5,
  ε_y 2.7–21 %, ζ 0.07–0.33. A separate bumper bar of ~10 nodes.
- **Rows**: BeamNG's parts run at row 1.1–1.6 (p50), up to 2–4. Adopting this
  means relaxing sedan1's cap of 1.0, an engine-wide limit, so it is the
  user's decision; the census only shows that real content runs past it in
  Rubble's own JBeam mode.
- **No damping-only beams**: damp each spring beam (ζ 0.03–0.06 on the
  structure).
- **Collision triangles**: one per 2–3 of a panel's nodes, on the part's own
  nodes, edges 0.35–0.55 m. Make 85–90 % of nodes contacters, including the
  engine and suspension nodes, which carry no triangle.
- **Flexbody binding**: place each panel node on the panel's surface (within
  4 cm) and on a crease, panel edge or shut line (within 4 cm); bind each mesh
  to its own part's group only; 5–11 nodes per m² of panel.
- **Mass**: engine as ~10 nodes of 10–23 kg; suspension 100–215 kg per axle
  (a fifth of the car); body shell 23–38 % of the kerb mass, not 55 %.

## Not measured

- Damping per node (Σc/m): `light_node_damping` in `showcase_car.rs` already
  gives it for the BX.
- Any run on the reference scenes: this page is static data only.
