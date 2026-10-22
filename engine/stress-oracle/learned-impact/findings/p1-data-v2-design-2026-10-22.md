# P1 data v2: design draft for the rented Linux server (2026-10-22)

This drafts planning's data-v2 request
(`docs/learned-stress-solver/generalization-plan.md`, step 0 and step 1). Nothing here
has run on the server yet.

## Step 0: the cross-machine determinism parity check (first thing on the server)

The oracle is deterministic per machine and per thread count (`par.rs`), but it is
untested across architectures: Apple M-series here, x86-64 there. The kit is ready.

**The kit** is built by `learned-impact/parity/make_kit.py` on this machine (74 MB):
- 50 viability events spread over all ten families, each with its recipe
  (`replay.json`), scene, context, outcome and recorded end fingerprint;
- for 5 of them, the full state at every boundary and the sha256 of their `/5` calls.

**The check,** `learned-impact/parity/check.py --kit KIT --bin target/release` on the
server, runs:
1. the pinned tests: stress-ref goldens, the M5 scene pins (`m5_scenes_are_p0s`), and the
   full-state completeness gate (`tests/full_state.rs`);
2. all 50 recipes replayed by the oracle, each end fingerprint against the recorded one;
3. for the 5 full events: every boundary restored and stepped equals the next, and the
   calls rebuilt from the recording hash the same.

**Outcome rule** (plan v2):
- all pass: the Mac and the server are interchangeable;
- any fails: the server becomes the single reference machine, and every evaluation
  reference, ensemble and bracket is regenerated there. The evaluation sets are
  regenerable from their recipes; reclaimed events carry `reclaimed.json` with the
  command.

The kit passes on this machine against itself first (`runs/dev/parity-self-check.log`).

## Step 1: the factor space and the Sobol design

One new catalog, `p2-walls`, with one scrambled Sobol point per distinct configuration
(Owen-scrambled, Joe–Kuo direction numbers, 20 dimensions). Categorical factors take
quantile bins of their dimension; continuous ones are log-uniform where the physics is
multiplicative.

| # | factor | range / levels | notes |
|---|---|---|---|
| 1–2 | width × height (chunks) | 6–28 × 6–26 | the bulk ranges |
| 3 | depth (chunks through the wall) | 1, 2, 3 | the axis the ballistic split missed |
| 4 | chunk size | 0.10, 0.15, 0.20 m | |
| 5 | opening | none 75%; a window or door 25% | piers, foundation and lintel kept |
| 6 | supports | standing on the ground, base, base and sides | gravity on for all three |
| 7 | gravity | on 90%; zero gravity, free (the M5 setting) 10% | keeps the M5 regime covered |
| 8–10 | material scales | strength 0.5–2, toughness 0.5–2, stiffness 0.7–1.5 (log) | concrete presets, as now |
| 11 | Weibull scatter | off 70%; modulus 6–12 30% | |
| 12 | impactor shape | sphere, box | |
| 13 | impactor mass | 50–4000 kg (log) | |
| 14 | impactor size | 0.05–0.30 m (log), at most 0.45 of the face | |
| 15 | speed | transition-targeted (below) | |
| 16 | incidence angle | 0–35° (yaw or pitch) | |
| 17–18 | aim point | face fractions; 10% grazing an edge, 15% at a corner | |
| 19 | spin | none 70%; up to 20 rad/s | |
| 20 | impactor count | 1 (70%), 2–4 (30%) | the extra impactors drawn independently |

Materials beyond concrete and irregular (Shardwright) chunks stay out of data v2's first
10k. They need their own pilot (planning's earlier decision), and data v2 would then add
them as factors with their own tiers.

## Speeds: transition-targeted

Bracketing every configuration costs about 9 extra oracle runs each, too many at 30k. So
data v2 does it in two stages:
1. **Bracket a Sobol subsample** of about 5% of the configurations (`--no-full-state`,
   9 speeds over 0.02–2× the punch threshold, about 4 MB per run). This records each
   one's perforation transition: the last rebound/arrest speed and the first perforation.
2. **Fit a transition model:** log v_transition as a smooth function of the factors
   (thickness·chunk, supports, mass, size, shape, material scales: a small ridge or GP
   regression), validated by leave-out on the brackets.

Each configuration's speed is then drawn around its predicted transition:
- 60% log-uniform in 0.5–2× (the transition band);
- 20% in 0.05–0.5× (sub-threshold: rebound, arrest, cracking);
- 20% in 2–20× (clean perforation and the high-speed regime).

The model's error is measured on the bracketed subsample and reported. The share of
events whose outcome lands in the predicted band is checked on the 10k checkpoint.

## Ensembles, splits and tiers

**Ensembles:** about 10% of training configurations get 8-member ensembles (±1 mm,
±0.1% speed, their own Weibull seed, as now). Every val/test configuration is an ensemble.

**Tiers** are assigned at design time, before recording, using P2's coverage check
(`python/learned_impact/outcome/coverage.py`, the single implementation) on the planned
recipes:
- **Tier 1:** held-out configurations of training layouts. Data v2 gives each training
  layout 3 impactor configurations by design, so every layout can hold one out.
- **Tier 2:** new layout groups whose every factor is covered by training. Drawn from the
  same Sobol sequence; the coverage check rejects any that falls outside.
- **Tier 3:** designed extrapolation, `parameters.extrapolation` set: speed above
  training, mass beyond, size beyond, more loads. Reported separately; never used for
  selection.

The tier goes into each recipe (`parameters.tier`), and from there into every call
manifest. Sealed sets stay untouched.

## Sizes, cost and disk

Every event is recorder revision 9: every frame, full state. Brackets are recorded
without full state.

| stage | configurations | events | disk (full state) | oracle core-hours |
|---|---:|---:|---:|---:|
| brackets (5%) | 500 / 1,500 | 4,500 / 13,500 single runs | about 18 / 55 GB | about 150 / 450 |
| 10k checkpoint | about 7,000 (10% × 8 ensembles) | 10,000 | about 200 GB | about 330 |
| 30k target | about 21,000 | 30,000 | about 600 GB | about 1,000 |

- **Per-event cost and size:** this machine's viability runs give 6–16 MB and 0.5–2.5
  core-minutes per wall event at pilot sizes. The bulk sizes (up to 28 × 3 × 26 chunks)
  push both up, hence about 20 MB and 2 core-minutes as the plan assumes.
- **Calls:** val/test/ensembles resident; training calls on demand into a bounded cache.
  With 2 TB of disk, the training calls of the 10k checkpoint (about 30 GB) can also stay
  resident.
- **Wall time on 128 cores:** the 10k checkpoint takes about 3 h, the 30k about 8 h.

## The portable job runner

`learned-impact/jobrun.py` (to build), Linux and many cores:
- **Queue:** a spec (JSON) lists generator jobs; each is a `learned_impact_outcome_generate`
  invocation with its own `--jobs K`. The runner keeps up to `P` jobs running, so P × K
  ≤ cores, and each job resumes (finished events are skipped).
- **Whole-footprint budget guard** (block-counted, as `du`): checked before every job
  starts and before any plan is written, fixing the overshoot seen here, where each
  stopped job still wrote its `recipes.json`. The tree under the guard includes caches and
  any data outside it named in the spec.
- **Control and state:** a `PAUSE` file holds new events; `status.json` per job and one
  runner status file; heartbeats every 30 s; one log per job.
- **Supervision:** systemd or tmux.
- **Thread rule:** the GPU trainers get their own reserved cores; the oracle runs at a
  lower nice level.
- **Recorded provenance:** source commit, binary hash and machine (CPU model, core count,
  OS), so the reference-machine rule can be audited.

## Open decisions (planning and the user)

1. Gravity off for 10% of free walls, to keep the M5 zero-gravity regime covered: yes or no?
2. Materials and irregular chunks in data v2: after their pilot, as an extension of this
   design?
3. The transition model's band shares (60 / 20 / 20).
4. Disk on the server: about 1 TB covers the 30k at full state plus brackets and resident
   evaluation calls.
