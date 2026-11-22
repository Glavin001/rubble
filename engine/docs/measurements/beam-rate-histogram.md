# How much of a car needs 2 kHz (2026-11-16)

**Conclusion: drop "rates by material".** On the lab's generated sedan about 88 % of
the nodes and 97 % of the beams need the 2 kHz step with no safety margin at all, and
nothing is left to slow down once a margin is asked for. The stiff nodes are not a core:
they are the whole car, in one connected piece, in every part. Stepping parts of a car at
1 kHz or 500 Hz would save about 1 % of the beam work on the lab sedan and at most 6 % on
the repo's low-resolution sedan, before the cost of the interface.

A measurement on the spawn state, CPU only. Nothing is stepped and the solver is not
changed.

```bash
cd engine
cargo run --release -p engine-softbody-lab --example beam_rate_histogram -- \
    --car ~/Development/rubble-artifacts/sedan-live/mods/sedan_new     # the lab sedan
cargo run --release -p engine-softbody-lab --example beam_rate_histogram  # the repo sedan
```

Options: `--generated low|medium|high`, `--wheels pressure` (JBeam pressure wheels),
`--worst` (shocks and tyre spokes at their limit rates instead of inside their bounds).
Each run prints every table below in full (all parts, all break groups).

## The two cars

| | lab sedan | repo sedan |
|---|---|---|
| source | `rubble-artifacts/sedan-live/mods/sedan_new`, as `vehicle_lab --car <dir>` loads it | `authoring::build(style("sedan"), Resolution::Low)`, as `group_split_gpu.rs` spawns it |
| mode | JBeam, raycast wheels, BeamNG profile (centred damping) | `.truck`, soft tyres, Rigs of Rods profile |
| nodes / beam records / linear springs measured | 548 / 4,251 / 3,557 | 1,099 / 7,313 / 7,305 |
| mass | 1,448 kg | 1,450 kg |

Left out (not a spring between two nodes): the lab sedan's 670 aerodynamic triangle
records and 24 torsion records; the repo sedan's 8 limited springs with no rate.

## The limits

The solver's step is `STEP` = 0.5 ms (`engine-softbody/src/explicit_beams.rs`). Its node
update is v += h·F/m, then x += h·v, with the forces of the beam pass before it
(symplectic Euler). For one mode of stiffness k and damping c on mass m, with
ω = √(k/m) and ζ = c / (2·√(k·m)):

| case | stable while |
|---|---|
| undamped | h < 2/ω |
| damped, damping read from the step's start velocity (`.truck` kinds) | ω·h < 2·(√(1 + ζ²) − ζ) |
| damped, centred damping (JBeam kinds, jbeam-semantics-spec.md §39) | the smallest Ω = ω·h with (1 − ζΩ)·(Ω² + 4ζΩ) ≥ 4, and ζΩ < 1 |
| a damper alone (k = 0) | c·h/m < 2 |

The centred rule reproduces §39's table (ζ = 0.15 gives 2.148).

* **A beam alone:** m is the reduced mass m₁·m₂/(m₁ + m₂) of its two nodes.
* **A node:** k and c summed over all its beams, m its own mass.
* **The assembled car:** the largest ω of M⁻¹K at rest (K = Σ k·d̂d̂ᵀ per beam), by power
  iteration. This is the true undamped limit of the linear car.

A rate r is allowed when limit ≥ safety / r. "Under margin" means the element does not
reach that safety factor even at 2 kHz.

## Where the cars stand at 2 kHz today

| limit / step | lab sedan | repo sedan |
|---|---|---|
| tightest beam alone, damped | 1.47 | 1.35 |
| tightest node (Σk, Σc), damped | 0.85 | 0.97 |
| the assembled car, undamped | **1.07** (ω·h = 1.86) | **1.42** (ω·h = 1.41) |

The lab sedan runs within 7 % of the step's limit, as BeamNG content does (§39). So a
safety factor of 2 or 4 cannot be met by the car itself at 2 kHz, and the tables with
those factors mostly show "under margin". The per-node sum is a little pessimistic (0.85
where the true figure is 1.07): beams in different directions do not add fully. The last
table in this note therefore repeats the question with the assembled stiffness.

## Share by the slowest rate allowed

Lab sedan:

| what | safety | under margin at 2 kHz | 2 kHz | 1 kHz | 500 Hz | ≤ 250 Hz | **needs 2 kHz** |
|---|---|---|---|---|---|---|---|
| beams, each alone | 1 | 0.0 % | 1.0 % | 76.7 % | 17.3 % | 4.9 % | 1.0 % |
| nodes (Σk, Σc) | 1 | 12.0 % | 75.5 % | 11.3 % | 1.1 % | 0.0 % | **87.6 %** |
| beams, at their faster end node's rate | 1 | 24.3 % | 72.7 % | 3.0 % | 0.0 % | 0.0 % | **97.0 %** |
| mass, by node | 1 | 3.3 % | 79.1 % | 9.6 % | 8.0 % | 0.0 % | 82.4 % |
| beams, each alone | 2 | 1.0 % | 76.7 % | 17.3 % | 3.9 % | 1.0 % | 77.8 % |
| nodes (Σk, Σc) | 2 | 87.6 % | 11.3 % | 1.1 % | 0.0 % | 0.0 % | **98.9 %** |
| beams, at their faster end node's rate | 2 | 97.0 % | 3.0 % | 0.0 % | 0.0 % | 0.0 % | **100 %** |
| beams, each alone | 4 | 77.8 % | 17.3 % | 3.9 % | 1.0 % | 0.0 % | 95.1 % |
| nodes (Σk, Σc) | 4 | 98.9 % | 1.1 % | 0.0 % | 0.0 % | 0.0 % | **100 %** |

Repo sedan:

| what | safety | under margin at 2 kHz | 2 kHz | 1 kHz | 500 Hz | ≤ 250 Hz | **needs 2 kHz** |
|---|---|---|---|---|---|---|---|
| beams, each alone | 1 | 0.0 % | 1.1 % | 29.8 % | 43.9 % | 25.2 % | 1.1 % |
| nodes (Σk, Σc) | 1 | 0.9 % | 84.2 % | 14.5 % | 0.5 % | 0.0 % | **85.1 %** |
| beams, at their faster end node's rate | 1 | 0.7 % | 93.2 % | 6.0 % | 0.1 % | 0.0 % | **93.9 %** |
| mass, by node | 1 | 0.1 % | 72.6 % | 26.7 % | 0.6 % | 0.0 % | 72.7 % |
| beams, each alone | 2 | 1.1 % | 29.8 % | 43.9 % | 17.8 % | 7.4 % | 30.9 % |
| nodes (Σk, Σc) | 2 | 85.1 % | 14.5 % | 0.5 % | 0.0 % | 0.0 % | **99.5 %** |
| beams, at their faster end node's rate | 2 | 93.9 % | 6.0 % | 0.1 % | 0.0 % | 0.0 % | **99.9 %** |
| beams, each alone | 4 | 30.9 % | 43.9 % | 17.8 % | 4.1 % | 3.4 % | 74.8 % |
| nodes (Σk, Σc) | 4 | 99.5 % | 0.5 % | 0.0 % | 0.0 % | 0.0 % | **100 %** |

**The per-beam figure misleads.** Taken one at a time, 99 % of the beams look fine at
1 kHz. But a node carries 12 or 13 beams, their stiffnesses add on its one mass, and a
beam has to step with its faster end. That is why 1 % of beams becomes 97 % of beams.

Work per frame if every node ran at its own rate and every beam at its faster end's
(1.00 = everything at 2 kHz):

| safety | lab sedan: node updates / beam evaluations | repo sedan |
|---|---|---|
| 1 | 0.94 / 0.99 | 0.92 / 0.97 |
| 2 | 0.99 / 1.00 | 1.00 / 1.00 |
| 4 | 1.00 / 1.00 | 1.00 / 1.00 |

## By part (lab sedan, nodes at safety 1)

The lab sedan carries 42 parts. "Detached" is the part alone as a free piece (its own
nodes and the beams between them): its largest ω and the slowest rate that allows with
no margin.

| part | nodes | kg | need 2 kHz | 1 kHz | 500 Hz | detached: largest ω (rad/s) | detached: slowest rate |
|---|---|---|---|---|---|---|---|
| body | 197 | 487 | 95.9 % | 4.1 % | 0 % | 3,271 | 2 kHz |
| hood | 28 | 18 | 100 % | 0 % | 0 % | 3,606 | 2 kHz |
| front bumper | 24 | 11 | 70.8 % | 25.0 % | 4.2 % | 2,258 | 2 kHz |
| trunk | 20 | 12 | 100 % | 0 % | 0 % | 3,666 | 2 kHz |
| rear bumper | 19 | 9 | 68.4 % | 21.1 % | 10.5 % | 2,432 | 2 kHz |
| front doors (each) | 19 | 30 | 100 % | 0 % | 0 % | 3,596 | 2 kHz |
| rear doors (each) | 19 | 24 | 94.7 % | 5.3 % | 0 % | 3,534 | 2 kHz |
| exhaust | 13 | 18 | 38.5 % | 61.5 % | 0 % | 2,914 | 2 kHz |
| fenders (each) | 12 | 5 | 100 % | 0 % | 0 % | 3,322 | 2 kHz |
| engine | 10 | 170 | 100 % | 0 % | 0 % | 3,142 | 2 kHz |
| impact beams, front / rear | 10 / 10 | 8 / 6 | 70 % / 100 % | 30 % / 0 % | 0 % | 2,178 / 2,192 | 2 kHz |
| fuel tank, seats, dash, console, radiator | 4–8 each | | 86–100 % | 0 % | 0–14 % | 2,380–2,630 | 2 kHz |
| subframes, front / rear | 7 / 6 | 47 / 42 | 100 % | 0 % | 0 % | 2,397 / 2,171 | 2 kHz |
| wheels (each) | 5 | 19 | 20–40 % | 60–80 % | 0 % | 2,189 | 2 kHz |
| hubs (each) | 4 | 17–24 | 100 % | 0 % | 0 % | 2,000–2,376 | 2 kHz |
| differential | 3 | 42 | 0 % | 100 % | 0 % | 1,014 | 1 kHz |
| hub caps (each) | 3 | 0.3 | 0 % | 100 % | 0 % | 1,304 | 1 kHz |
| driveshaft, steering rack | 2 each | 1, 10 | 100 % | 0 % | 0 % | 1,844 | 1 kHz |
| transmission | 2 | 45 | 0 % | 50 % | 50 % | 657 | 500 Hz |
| glass (windshield, rear) | 1 each | 12, 8 | 0 % | 50 % | 50 % | – | – |

* **Detached pieces do not help.** 32 of the 42 parts, holding 525 of the 548 nodes,
  need 2 kHz even alone and with no margin. The ones that could step slower are the
  differential, hub caps, driveshaft, steering rack and transmission: 21 nodes in all.
  A bumper, door, hood or trunk lying on the road still needs 2 kHz.
* At safety 2 every part is at least 85 % at 2 kHz except the transmission (one of its
  two nodes) and the one-node windshield.
* **Break groups** (121 groups, 604 beams): at safety 2 every group is wholly at 2 kHz;
  none is slower or mixed.

The repo sedan has no part names, only its 80 structural member groups, and no break
group names. All but one of those groups are at least 99 % at 2 kHz at safety 2 (the
radiator: 2 of 6 nodes). Its 112 tyre nodes all need 2 kHz.

## Which beam kinds set the requirement

Share of the summed stiffness on the nodes that need 2 kHz (safety 2):

| kind | lab sedan: beams | k median (N/m) | share of Σk | repo sedan: beams | k median (N/m) | share of Σk |
|---|---|---|---|---|---|---|
| normal | 3,164 | 1.22·10⁶ | **95.3 %** | 5,616 | 2.1·10⁵ | **85.4 %** |
| support | 334 | 6.0·10⁵ | 4.2 % | 380 | 1.2·10⁵ | 2.3 % |
| anisotropic | 33 | 8.0·10⁵ | 0.4 % | – | | |
| bounded (shocks, limiters) | 15 | 1.4·10⁵ | 0.0 % | 4 | 3.0·10⁴ | 0.0 % |
| coupler | 7 | 8.0·10⁵ | 0.1 % | – | | |
| wheel rigidity (virtual) | 4 | 8.7·10⁵ | 0.1 % | – | | |
| tyre: tread to tread (normal) | – | | | 224 | 8.0·10⁵ | 5.9 % |
| tyre: tread to rim (normal) | – | | | 112 | 8.0·10⁵ | 3.0 % |
| tyre: tread to rim (bounded spokes) | – | | | 112 | 8.0·10⁵ | 3.0 % |
| rope | – | | | 46 | 3.7·10⁵ | 0.4 % |
| damper only | – | | | 811 | 0 | 0.0 % |

* **Ordinary NORMAL beams set it:** sheet metal and frame members, everywhere. Not the
  suspension, the shocks, the hydros (the lab sedan has none: raycast steering) or the
  wheels.
* Tyre beams add 12 % on the repo sedan and matter only for the tyre nodes themselves.
* With the lab sedan on JBeam pressure wheels (804 nodes) the picture is the same:
  75.6 % of nodes and 92.1 % of beams need 2 kHz at safety 1. The tyre nodes are the
  softer ones (16.7 % of nodes at 500 Hz by the per-node rule), but they are not part
  of the driving configuration, which uses raycast wheels.
* With shocks and tyre spokes at their limit rates (`--worst`) the lab sedan does not
  change. The repo sedan's tyre spokes then exceed the linear limit at 2 kHz, which only
  says the limit state is not a state the car sits in.

What a 2 kHz node looks like (safety 2):

| | lab sedan | repo sedan |
|---|---|---|
| mass, min / median / max | 0.10 / 1.40 / 20 kg | 0.05 / 0.79 / 21 kg |
| Σk, median | 1.5·10⁷ N/m | 4.0·10⁶ N/m |
| beams per node, median | 13 | 12 |
| ζ, median | 0.11 | 0.23 |

Light and heavy nodes alike: the generator sizes each beam's stiffness to its nodes'
mass (`Built::stiffness_scale`, "scaled so the explicit step stays stable"), so every
node ends up near the same limit. That is by design, and it is why no soft region exists.

## Where the 2 kHz nodes are

| | safety | 2 kHz nodes | their box / the car's box | connected pieces (largest) | slower nodes with a 2 kHz neighbour | slower pieces once the 2 kHz nodes are cut out (largest) | beams joining two rates |
|---|---|---|---|---|---|---|---|
| lab sedan | 1 | 480 (87.6 %) | 1.00 on every axis | 1 (480) | 91.2 % of 68 | 21 (11 nodes) | 361 (10.1 %) |
| lab sedan | 2 | 542 (98.9 %) | 1.00 | 1 (542) | 100 % of 6 | 6 (1 node) | 50 (1.4 %) |
| repo sedan | 1 | 935 (85.1 %) | 1.00 | 8 (820) | 100 % of 164 | 19 (104 nodes) | 1,728 (23.7 %) |
| repo sedan | 2 | 1,094 (99.5 %) | 1.00 | 1 (1,094) | 100 % of 5 | 2 (4 nodes) | 56 (0.8 %) |

**One stiff piece that is the whole car, with the soft nodes scattered through it.** The
stiff set spans the car's full length, width and height and is connected. The slower
nodes are 19 to 21 small islands, almost every one touching a 2 kHz node. Along the
length in ten slices, every slice is 95–100 % at 2 kHz (safety 2). A multi-rate step
would have an interface at nearly every slow node: on the repo sedan the beams joining
two rates (24 %) outnumber the beams that would run slower (6 %).

## The same question from the assembled stiffness

The per-node rule is an estimate, so this is the check that does not rely on it. Nodes are
taken out (held, as if stepping at 2 kHz) from the stiffest mode of what is left until
the rest's largest ω is under the limit. A greedy search, so the sets are lower bounds.
A beam is slow when both its ends are.

| what the slow set must satisfy | lab sedan: slow nodes | slow beams | repo sedan: slow nodes | slow beams |
|---|---|---|---|---|
| 1 kHz with the margin the car has today (1.07 / 1.42) | 53 (9.7 %) | 44 (1.2 %) | 274 (24.9 %) | 928 (12.7 %) |
| 1 kHz at safety 2, or 500 Hz at safety 1 | 3 (0.5 %) | 0 | 106 (9.6 %) | 213 (2.9 %) |
| 500 Hz with the car's own margin | 3 (0.5 %) | 0 | 25 (2.3 %) | 16 (0.2 %) |
| 1 kHz at safety 4, or 500 Hz at safety 2 | 1 (0.2 %) | 0 | 11 (1.0 %) | 3 (0.0 %) |

Best case, keeping only today's margin: half the cost of 1.2 % of the lab sedan's beams
(0.6 % of the beam work) and of 12.7 % of the repo sedan's (6.4 %). The repo sedan at
medium resolution is the same within a few points (32 % of nodes, 19.5 % of beams at
its own margin).

## What it means

* The plan's rule was to drop the bet if about 90 % of the car needs 2 kHz. Measured:
  88 % of nodes and 97 % of beams on the lab sedan, 85 % and 94 % on the repo sedan, with
  no safety margin; 99–100 % with a safety factor of 2. **Dropped.**
* No meaningful fraction of the car can step at half or a quarter of the rate. The
  cars are built to sit at the 2 kHz limit everywhere, so a lower rate for some parts
  means softer or heavier parts, which is a content decision and not a solver one.
* Detached pieces are the same sheet metal and need the same rate. Their cost is better
  cut by not simulating them (vehicle sleep, debris retirement), which already exists.
* Not measured: contact stiffness (ground, self-collision, vehicle contact), which adds
  to a node's stiffness while it touches and only tightens these limits; and the
  yielded or broken car, where beams lose stiffness. A wreck's loose parts could be
  softer than at spawn, but the detached-part column says the intact panels are not.
