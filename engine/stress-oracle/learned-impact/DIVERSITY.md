# Diversity v1 — decent starting catalog

Goal: enough **structure × load × time** coverage that holdouts mean something (not “never breaks → F1=1”).
Rates measured on this Mac (2026-10-19): **wall ≈6 s/frame**, **house ≈8 s/frame**, small scenes ≪1 s/frame.
Train: **~15–40 s** for 600–800 steps on prove-sized sets (MPS).

## What you asked for → what we put in v1

| Theme | Coverage in v1 | Explicitly deferred |
|---|---|---|
| Per-brick / per-bond labels + wave across graph | Already in schema; keep | — |
| Wall push / ram / punch | Speed matrix + 1 soft no-break | Mass/angle/height (v1.1) |
| Soft tap / “not enough” | 1–2 low-speed or weak-ram no-break walls | Full bounce physics suite |
| House w/ openings + ram | 2 speeds × longer rollouts | Full vibe-town packs |
| Crackling cantilevers | **Fracture on**; load×length×gravity until root/mid break | Full analytic sweep |
| Loop (arch) | Keystone remove + stand | Multi-span arches |
| Building frame | Sudden column removal | Gradual + multi-column |
| Time / inertia / ring-out | **8–16 frames** on impacts (not 1–2) | True 90-frame (costly) |
| Holdouts | Whole structure + whole load band | — |

## Catalog (counts)

### A. Walls — push / ram / punch / none (14 events)

| Load | Speeds (m/s) | Frames | Events |
|---|---|---|---|
| Soft / no-break | 1, 2, 3, 4 | 4 | 4 |
| Push | 5, 8 | 8 | 2 |
| Ram | 12, 15, 18 | 8 | 3 |
| Punch | 25, 35, 45 | 8 | 3 |
| Long ring-out | 10, 20 | **16** | 2 |

**Frames ≈** 4×4 + 2×8 + 3×8 + 3×8 + 2×16 = **116** → ETA **~12 min** @ 6 s/frame.

### B. Houses — openings + car (~6 events)

| Load | Speeds | Frames | Events |
|---|---|---|---|
| Soft / settle-ish | settlement | 8 | 1 |
| Car impact | 8, 12, 20 | 8 | 3 |
| Hard punch-ish | 30 | 8 | 1 |
| Long ring-out | 15 | **16** | 1 |

**Frames ≈** 8+24+8+16 = **56** → ETA **~7–8 min** @ 8 s/frame.

### C. Crackling cantilevers — must fracture (~9 events)

Need a **fracture-enabled** cantilever variant (current builder has `fracture=false` — fix first).

| Arm / chunks | Tip load scale | Gravity | Frames | Notes |
|---|---|---|---|---|
| short / mid / long (3) | light (no break) | off | 4 | negative class |
| short / mid / long (3) | break-at-root | off | 8 | classic |
| mid / long (2) | break-midspan-ish | on | 8 | gravity + tip |
| mid (1) | overload crush | off | 8 | heavy |

**Frames ≈** ~60 of small graphs → ETA **~2–4 min** (cheap per frame once fracture works).

### D. Loop + frame + anchors (~6 events)

| Structure | Load | Frames |
|---|---|---|
| Arch | keystone remove | 8 |
| Arch | stand (no remove) | 4 |
| Frame column | sudden remove | 8 |
| Frame | gradual remove | 8 |
| Shear / tension | fail + no-fail | 4 each |

ETA **~3–5 min**.

## Totals

| | Events (approx) | Frames (approx) | Generate ETA |
|---|---|---|---|
| **Diversity v1** | **35** | **~264** | **~25–35 min** wall (less with Rayon) |
| Current `--prove` | 17 | 43 | ~1.4 min (fracture cantilevers enabled) |

Parallel CPU later can cut wall time; single-threaded oracle is the limiter today.

## Train ETA (after data exists)

| Job | Steps | ETA | Gate |
|---|---|---|---|
| Overfit one house 16-frame | 300 | ~1–2 min | loss ↓ ≫ |
| Train all, hold out **punch** loads | 800–1500 | **~1–3 min** | fail F1 holdout ≥ ~0.7 |
| Train all, hold out **masonry_house** | 800–1500 | **~1–3 min** | F1 within ~0.1 of wall-seen |
| Train all, hold out **cantilever** | 800 | ~1–2 min | Only valid once cantilevers *break* |
| Optional rollout FT (4–8 unroll) | +same order | **+2–5 min** | less drift on 16-frame clips |
| **Full train pass suite** | — | **~10–20 min** | write finding |

Data gen dominates (**½ hour**); training is cheap at this size.

## Holdout matrix (prove generalization)

1. Hold out **punch** wall speeds (train push/ram/soft).
2. Hold out **house** structure (train walls+frame+cantilever+arch).
3. Hold out **long** cantilever (train short/mid).
4. Never random-frame split.

## Implementation status (2026-10-19)

| Step | Status |
|---|---|
| Fracture cantilever (`builders::cantilever_crackling`) | **done** — tip load held, `fracture=true`; analytic `cantilever` unchanged |
| `diversity_jobs()` + `--diversity` / `--diversity-v1` + `--confirm-diversity` | **done** |
| Parallel generate + resume + per-job ETA | **done** (`run_jobs_with`) |
| Generate diversity v1 (~30 min / less parallel) | **not run here** — opt-in CLI |
| Train + holdouts + finding | **pending** |
| Stretch 90-frame clips | deferred |

```sh
cargo run --release -p stress-learned --bin learned_impact_generate -- \
  --diversity --confirm-diversity --out /tmp/li-diversity-v1
```

`--prove` still 17 events; cantilevers there now use `cantilever_crackling` (fracture on).

## Storage cost (measured, uncompressed `learned-impact/1`)

| Row | Size |
|---|---|
| Wall frame (840 chunks / 2018 bonds) | **~0.70 MB**/frame + **~0.25 MB** static once |
| House frame (665 / 1299) | **~0.48 MB**/frame + **~0.17 MB** static |
| Diversity v1 (~230 frames, mix) | **~150–200 MB** raw |
| Plan full set (5k × 60-frame houses) | plan ~60 GB; compresses well (fp16+zstd) |

A “training row” ≈ one **(event, frame)** with full per-chunk / per-bond arrays (schema already stores QS force, peak util, fail bins, joint start+end, Δstate). Sub-bin continuous trajectories are **not** stored (only K=4 bin averages) — that is the intentional trade.

`--always-qs` forces a real QS solve every frame (stores it in `bond_qs_force`); without it, impact jobs fill that field from `bond_force_prev` (~1.7× cheaper on walls).
