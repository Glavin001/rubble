# Explicit beams: sections, triggers, slide nodes and animators

This is a behavioural specification for four `.truck` features that the 1982 Ford Mustang GT 351W (`inspiration/ror-repo/mustang/1982-Ford-Mustang-GT-351W.truck`) needs in Rubble's explicit node–beam solver:

* `section` / `sectionconfig` / `end_section`;
* `triggers`, and the command motion they drive;
* `slidenodes`;
* `animators`.

It also confirms that `lockgroups`, `videocamera` and `flexbodies` have no physical effect.

It extends these documents:

* `explicit-beams.md` (the solver, step and beam pass);
* `explicit-beams-aero-commands-spec.md` (commands, including self-centering, and their length-update arithmetic);
* `explicit-beams-wheels2-shocks2-spec.md` (`meshwheels`, node defaults on wheel nodes, the dry-mass distribution).

Terms not defined here mean what they mean there. That includes:

* the step h = 0.0005 s;
* the node pass and the beam pass;
* the gate G, the strength S, and the thresholds Tc and Tt;
* the beam kinds and types;
* "normalise v" and the f32 / [f64] notation of the aero spec.

The reference is Rigs of Rods (RoR), judged as a black box. The goal is agreement to float tolerance in spawn tables (node and beam order, parameters) and in trajectories. Behaviours that are artefacts of the reference rather than physics are marked **(quirk)**; the RoR-compatible profile reproduces them.

Notation:

* All arithmetic is IEEE single precision (f32) unless marked [f64].
* `min(a, b)` means "b if b < a, else a"; `max(a, b)` means "b if a < b, else a".
* "World position" of a node means origin + relative position, as the node pass recomputes it after integration (component-wise f32 sum).

---

## 0. What the Mustang uses

| feature | the car's lines | physical effect |
|---|---|---|
| sections | three modules, `ats_wheels`, `saleen_wheels` and `sve_series_1_wheels`, each holding 4 `meshwheels` lines and a `flexbodies` line; three `sectionconfig` lines | picks which 4 wheels spawn and their tyre-node friction |
| `triggers` | `421, 422, 0.6, 0.4, 84, 83, ci` and `235, 421, 0.58, 0.5, 17, 16, ci` | two zero-force beams that hold command keys 83/84 and 16/17 |
| commands driven by triggers | `279, 422, …, 16, 17, i ign …`, `426, 428, …, 83, 84, i …`, `423, 427, …, 83, 84, i …` | move only when a trigger's short bound is crossed (§2.6) |
| `slidenodes` | 12 lines, spring `s9000`, `s900000` or the default; no other options | spring forces every step |
| `animators` | `279, 421, 0.5, tacho \| inv` | an ordinary actuator beam; its rest length never changes with no engine running (§4) |
| `lockgroups`, `videocamera`, `flexbodies` | present | none (§5) |

---

## 1. `section`, `sectionconfig`, `end_section`

### 1.1 Parsing

* **`section <version> <name> …`** switches the parser's current module to the module named by the **third token**.
  * The line needs at least 3 tokens; otherwise it is ignored.
  * The version (second token) is unused.
  * Tokens after the name are ignored. **(quirk)** The docs suggest one section line can belong to several configurations; in RoR only the first name counts.
  * If no module of that name exists, it is created; if it exists, the parser re-enters it and appends to it.
  * Naming the module the parser is already in is ignored.
* **`end_section`** switches back to the root module. In the root module it is ignored.
* Both lines also end the current block (as any new keyword does). Data lines after them are ignored until the next block keyword.
* **`sectionconfig` lines are ignored entirely. (quirk)** They neither create modules nor set the default configuration. Their number and order have no effect. Data lines after a `sectionconfig` line are swallowed until the next keyword.
* Every block line between `section` and `end_section` goes into that module, in file order. That covers `nodes`, `beams`, `meshwheels`, `commands2`, `slidenodes` and so on.

### 1.2 Which modules spawn

The spawned vehicle is the **root module**, followed by **at most one** user module, the selected configuration.

* The available configurations are the module names **sorted in ascending byte-wise string order**. RoR keeps modules in an ordered map, so the order is that of the names, not of the file.
* With no configuration requested, RoR's spawn paths (the vehicle selector, preselected and scripted spawns) choose **the first name in that order**. Spawning with an empty configuration name selects no user module; no normal spawn path does that.
* If the requested name does not exist, only the root spawns (RoR logs a warning).
* **The Mustang's default is `ats_wheels`**: `ats_wheels` < `saleen_wheels` < `sve_series_1_wheels`.

Rubble's loader takes an optional configuration name and applies the same rules. It must report the available names and the one it chose.

### 1.3 Spawn order: modules interleave per section keyword

RoR spawns keyword by keyword, in a fixed keyword order (`explicit-beams-wheels2-shocks2-spec.md` §2.1, extended by §2.7 and §3.6 below). For each keyword, it first spawns **all of the root module's lines** of that keyword, then **all of the selected module's lines**. So a module never spawns as a contiguous block; its lines interleave with the root's, keyword by keyword:

* nodes: root `nodes`, then module `nodes`;
* then cinecam (root, module);
* then `wheels` (root, module), `wheels2` (root, module), `meshwheels` (root, module), `meshwheels2` (root, module), `flexbodywheels` (root, module);
* then `beams` (root, module), `shocks` (root, module), and so on through the whole order.

For the Mustang with `ats_wheels`:

* **Nodes** (527 in total):
  * truck nodes 0–428;
  * cinecam nodes 429, 430;
  * the four `ats_wheels` meshwheels, 24 nodes each, in their file order inside the module: 431–454, 455–478, 479–502, 503–526.
* **Beams**, in this order:
  * the 16 cinecam beams;
  * the 4 × 108 meshwheel beams (12 rays, with a rigidity node, so 9 beams per ray);
  * root `beams`;
  * root `shocks` (12);
  * `commands2` (8);
  * `hydros` (2);
  * `triggers` (2);
  * `animators` (1, the last beam; §4.2).

Lines of an **unselected** module do nothing at spawn.

### 1.4 Legacy node numbers across modules

The parser hands out legacy numbers for generated nodes in **file order over all modules, selected or not**. The count of generated nodes per wheel keyword also includes **every** module.

Resolving a number uses the section offsets of `explicit-beams-wheels2-shocks2-spec.md` §2.1, computed from those whole-file counts. **(quirk)** A number that names a generated wheel node is therefore only reliable when a single module generates wheels. With unselected modules it can resolve to the wrong node.

The Mustang references no node above 430 in any physical section, so this does not arise for it. Rubble must:

* resolve numbers ≤ the last truck/cinecam node exactly;
* **fail loudly** on a reference to a generated node when more than one module generates nodes. Exact parity there is not worth the risk.

### 1.5 `set_` directives carry across modules in file order

The following are **parser state**, not module state:

* `set_beam_defaults`;
* `set_beam_defaults_scale`;
* `set_node_defaults`;
* `set_inertia_defaults`;
* `detacher_group`.

The parser updates them as it reads the file, top to bottom. `section`, `end_section` and module selection never reset or save them. Each spawning line captures the values in force at **its position in the file**. So:

* a directive inside a module affects the root lines that follow `end_section`;
* a directive in an unselected module still affects every later line, including root lines and lines of the selected module. The directive does not need to be spawned to take effect.

For the Mustang this changes nothing physical. Each wheel module resets the node defaults before its wheels (`set_node_defaults` with −1s) and sets the beam scale itself. The root lines after the modules (engine, slidenodes, cab) do not use beam or node defaults.

`enable_advanced_deformation` is a whole-document flag, captured by each later `set_beam_defaults`.

Exact directive semantics (Rubble's current loader differs; fix it):

* **`set_beam_defaults_scale s_k, s_c, s_deform, s_break`**
  * It needs **all four** numbers. With fewer than four, the whole line is ignored and the previous scales remain.
  * A fifth number and anything after it are ignored. The Mustang has several lines with five numbers; the fifth (0.1) has no effect.
  * It replaces all four scales. It keeps the unscaled defaults.
* **`set_beam_defaults k [, c [, deform [, break [, diameter [, material [, plastic]]]]]]`**
  * It starts from the **current** defaults, scales included; omitted fields keep their current values.
  * A negative k, c, deform or break means 9 000 000, 12 000, 400 000 or 1 000 000.
  * It marks the defaults as user-defined. That matters for the deform-threshold rule in `explicit-beams-wheels2-shocks2-spec.md` §2.4.
* **`set_node_defaults load_weight [, friction [, volume [, surface [, options]]]]`**
  * It needs at least the load weight.
  * Each of the four numbers that is negative **or omitted** becomes RoR's default, **not** the previous value. The defaults are: load weight unset, friction 1, volume 1, surface 1.
  * The options are replaced only if given.

The Mustang's per-module node defaults determine tyre-node friction (`explicit-beams-wheels2-shocks2-spec.md` §2.3):

| module | front wheels (31/32, 39/40) | rear wheels (61/62, 53/54) |
|---|---|---|
| `ats_wheels` (default) | friction 1 | friction 1.1 |
| `saleen_wheels` | friction 1.05 | friction 1.1 |
| `sve_series_1_wheels` | friction 1, volume 1.05 | friction 1.1 |

Volume and surface matter only for buoyancy (water), which is out of scope.

---

## 2. `triggers`

### 2.1 File format

`n1, n2, contraction_limit, expansion_limit, short_action, long_action [, options [, boundary_timer]]`

* The line needs at least 6 fields; otherwise it is ignored.
* Options are single characters; unknown characters are ignored with a warning:
  * `i` invisible (no physical effect);
  * `c` command-style bounds;
  * `x` start disabled;
  * `b` key blocker at start;
  * `B` trigger blocker;
  * `A` inverted trigger blocker;
  * `s` command-number switch;
  * `h` / `H` hook unlock / lock;
  * `t` continuous;
  * `E` engine trigger.
* `boundary_timer` defaults to 1.0. It is used only by `s`.

The Mustang uses only `c` and `i`. Section 2.7 tells Rubble what to do with the other options.

### 2.2 Validation (plain trigger: none of `B`, `A`, `h`, `H`, `E`)

* `short_action` must be in 1…84 (84 is the number of command keys). Otherwise the line is **dropped entirely**: no beam, no node-mass contribution. RoR logs an error.
* `long_action`:
  * −1 makes the trigger a command-key blocker (out of scope, §2.7);
  * 0…84 is accepted. Key 0 exists but is never processed by the command pass, so a long action of 0 does nothing;
  * a value above 84 is undefined in RoR. Rubble rejects it.
* If either node does not resolve, the line is dropped with a warning.

### 2.3 Bounds

Let sb₀ = contraction_limit and lb₀ = expansion_limit, as parsed f32 values.

* **Without `c`**: sb = sb₀, lb = lb₀ ("shock style": fractions of the rest length).
* **With `c`**: sb = |sb₀ − 1| and lb = lb₀ − 1. Both are f32 operations; the absolute value is the float absolute value. **(quirk)** lb is **not** made absolute, so an expansion limit below 1 gives a negative lb.

The Mustang's values:

| trigger | L₀ (m, from spawn positions) | sb | lb |
|---|---|---|---|
| T1 = (421, 422) | 0.300 | 0.39999998 (0x3ECCCCCC) | −0.60000002 (0xBF19999A) |
| T2 = (235, 421) | 0.763 | 0.42000002 (0x3ED70A3E) | −0.5 (0xBF000000) |

*Verification flag*: RoR calls an unqualified `abs` on a float here. On the toolchains checked this is the float overload. If the harness ever shows sb = 0 for T1, the integer overload was picked; record it and follow the harness.

### 2.4 The beam each trigger spawns

Triggers spawn **after `hydros`** and before `ropes` (root lines first, then the selected module's), in file order. One beam per line:

| property | value |
|---|---|
| ends | (n1, n2); node a = n1 |
| kind | **trigger** (a new kind; it is not plain) |
| type | actuator (never yields) |
| stiffness k, damping c | **0 and 0** |
| strength S | the beam-defaults break threshold at the line, **unscaled** |
| G, Tc, Tt | the ordinary deform threshold at the line, **scaled** by the deform scale (as for `beams`) |
| plastic coefficient | from the defaults (unused) |
| rest length L = L₀ | \|x_n1 − x_n2\| from the spawned f32 positions |
| short bound, long bound | sb, lb (§2.3) |
| detacher group | the current group at the line |

Mustang (scale 0.05, 0.005, 0.01, 1.0 in force; no `set_beam_defaults`): S = 1 000 000 and G = Tc = Tt = 4000.

Consequences:

* **The beam's force is always zero.** F = −0·e − 0·u is a signed zero. It adds nothing to either node and never passes the gate, so a trigger beam never breaks on its own force.
* A detacher-group cascade can still disable it (group > 0 masters, < 0 followers, as for any beam). A disabled trigger is skipped by the beam pass, so it stops firing.
* It **does** take part in the dry-mass distribution like any non-virtual beam: L₀/2 per non-tyre end.
* It does **not** count as a live plain beam for the collision-triangle break exception, because its kind is not plain.
* It can serve as a rail segment for a slide node (§3.2).

Per-trigger state at spawn:

* *enabled* = true (false with `x`);
* *short key* = short_action, *long key* = long_action.

Per command key k (1…84), at spawn:

* *blocked* = false;
* *trigger input* = 0;
* *command value* = 0.

Each trigger line also writes the blocked flag of both its keys at spawn. Without `b`, both become false. With a later trigger on the same keys, the last line wins.

### 2.5 The per-step trigger test (inside the beam pass)

In the beam pass, in beam-index order, a live trigger beam computes ℓ and e = ℓ − L exactly like every beam (same inverse square root, same L). If the trigger is enabled:

1. **Long side**: if e > lb·L (f32 product, strict), and the long key is not blocked: set that key's *trigger input* to 1.
2. **Short side**: else, if e < −sb·L (strict), and the short key is not blocked: set the short key's *trigger input* to 1.
3. Otherwise ("inside the bounds"), a plain trigger does nothing.

Notes:

* **There is no hysteresis and no latch.** Each step's test stands alone, and a trigger "holds" its key only while the condition holds.
* Writing 1 is idempotent. Several triggers on one key act as a logical OR.
* The test order matters only when both conditions are true. That happens when lb < −sb, which `c` with an expansion limit below 1 − |1 − contraction limit| produces. Then **the long side wins**.

For the Mustang this gives two regions (L the trigger's rest length, ℓ its current length):

| trigger | ℓ > 0.4·L (T1) / 0.5·L (T2) | ℓ ≤ that |
|---|---|---|
| T1 (421, 422), L = 0.300 | key **83** held | key **84** held |
| T2 (235, 421), L = 0.763 | key **16** held | key **17** held |

The exact boundaries are e > lb·L and e < −sb·L as given above, with the f32 constants from §2.3. At the boundary itself (e = lb·L), the short test applies.

There is no inside region: each trigger holds exactly one of its keys in every step from the first beam pass on. At spawn both triggers sit on their long side, so keys 83 and 16 are held continuously.

### 2.6 What a held key does: the command pass with input

The command pass (`explicit-beams-aero-commands-spec.md` §1.4) runs after the hydro/animator pass and before the beam pass (§6). It is extended with key input as follows.

**Key values.** At the start of the pass, for each key k = 1…84 in order:

* *command value*(k) = max(player input, *trigger input*(k)). Player input is 0 in every black-box scene.
* Then *trigger input*(k) is reset to 0.

So a trigger set in step n's beam pass drives commands in step n+1. **There is one step of latency.** In the first step after spawn every command value is 0.

**Self-centering lock.** Before moving anything, every self-centering beam is unlocked. Then, for every key whose command value is ≥ 0.5, each command entry on that key is locked. If the entry's command line is self-centering, its mode (shared by both of the line's entries) is set to 0. A locked self-centering line skips the centering steps 1–3 of the aero spec in this step, but still runs the move test (step 4 with this value v). The Mustang has no self-centering commands.

**Motion.** Keys are visited 1…84 in ascending order. Within a key, its entries are visited in spawn order: command lines in spawn order, root module then selected module. Each command line has a contraction entry on its contract key and an extension entry on its extend key. For each entry with direction σ (−1 contraction, +1 extension), rate r (shorten_rate or lengthen_rate) and boundary B (max_contraction or max_extension):

1. If L_ref = 0 or L = 0: skip the entry.
2. ρ = L / L_ref. The entry may move only if (σ = +1 and ρ < B) or (σ = −1 and ρ > B), both strict, tested **before** the move. So L may pass the boundary by up to one step's change and then stops.
3. v = the key's command value. The inertia filter is the identity when the line has no inertia. It has none when the line's start and stop delays are 0 and no `set_inertia_defaults` set positive delays; that holds for every Mustang command. With `p`, `o` or a self-centering mode pointing this way, v may be forced to 1; out of scope here.
4. **needs_engine gate**: if the line needs the engine, and the vehicle has an engine that is not running (or not at hydraulic pressure), skip the move. With no engine object at all, nothing is gated.
   * The `commands2` default is needs_engine = **true**. This is an edge case: "engine object present but off" freezes such commands, while "no engine" does not. All Mustang commands set it to 0.
5. cf = 1 unless affect_engine > 0, in which case cf is the engine's crank factor. That factor is 1 with no engine object.
   * With option `f`, the running crank factor of the pass is clamped to ≤ 1 for this and **every later entry in the step**. **(quirk)**
   * All Mustang commands have affect_engine = 0, so cf = 1.
6. L ← L × (1 + σ·r·v·cf·h / L), with the exact f32/[f64] evaluation of the aero spec §1.4 step 4. With v = 1: ΔL ≈ σ·r·h per step.

A broken command beam still has its L updated, with no physical effect.

**The Mustang's trigger-driven commands** (scale 0.05, 0.005, 0.01, 1.0 in force; options `i`; delays 0; affect_engine 0; needs_engine 0):

| line | contract key / rate / B | extend key / rate / B | k | c | S | G |
|---|---|---|---|---|---|---|
| (279, 422) | 16 / 0.6 / 1.0 | 17 / 0.8 / 1.8 | 450 000 | 60 | 1 000 000 | 4000 |
| (426, 428) | 83 / 10 / 1.0 | 84 / 10 / 1.7 | 450 000 | 60 | 1 000 000 | 4000 |
| (423, 427) | 83 / 10 / 1.0 | 84 / 10 / 1.7 | 450 000 | 60 | 1 000 000 | 4000 |

At rest the held keys are 16 and 83, both contraction keys with B = 1.0. ρ = 1 exactly, which is not > 1.0, so **nothing moves**.

These beams move only after a trigger's short side is reached:

* T2 short (node 421 within 0.5·L of node 235) holds key 17, which extends (279, 422) at 0.8 m/s up to ρ ≥ 1.8;
* T1 short (nodes 421 and 422 within 0.4·0.3 m) holds key 84, which extends the two front commands at 10 m/s up to ρ ≥ 1.7.

When the trigger returns to its long side, key 16 or 83 retracts the beams at their contraction rate down to ρ ≤ 1.0. (The mechanism is meant to be driven by the tacho animator with the engine revving; with no engine running only large deformation can reach these bounds.)

The other five Mustang commands (keys 1–8, doors, bonnet and boot) are never driven by a trigger. With no input they hold their spawn length.

### 2.7 Trigger options the Mustang does not use

Rubble must **refuse with a clear error** a trigger line with any of `x`, `b`, `B`, `A`, `s`, `h`, `H`, `t`, `E`, or with long_action = −1, until each is specified separately. The current harness has not verified them. In brief, for whoever specifies them next:

* `x` sets enabled = false at spawn.
* `B` and `A` enable or disable the next N trigger beams in beam-index order.
* `s` swaps the short and long keys of other triggers on the same key pair, latched by the boundary timer.
* `t` writes a 0…1 value proportional to the position between the bounds.
* `h` and `H` lock or unlock hook groups.
* `E` drives engine controls.
* `b` and long_action = −1 make key blockers.

---

## 3. `slidenodes`

### 3.1 File format

A line is split on commas and spaces (empty tokens dropped). The tokens are:

1. the first token: the slide node;
2. then rail nodes: every following token **until the first option token**. A rail-node token may also be a node range;
3. then options. Each option is a letter (either case) followed by its number, with no separator:
   * `S` spring rate (N/m), default 9 000 000;
   * `B` break force (N), default +∞;
   * `T` tolerance (m), default 0;
   * `R` attachment rate (m/s), default 1;
   * `D` attachment distance (m), default 0.1;
   * `G` rail-group id (instead of a node list);
   * `C` followed by `a`, `f`, `s` or `n`: attachment constraints.

Details of option tokens:

* A token is treated as an option if its first character, upper-cased, is one of S, B, T, R, G, D, C. **(quirk)** A *named* rail node starting with one of these letters is misread as an option. Rubble rejects such names in slide-node lines.
* An unknown letter after the options have started is ignored. Before that point it is taken as a node name.
* The values S, B, T, R and D are stored as absolute values.
* The line needs at least 2 tokens.

The Mustang uses only rail lists and `s` (spring 9000 or 900 000). Lines without `s` use 9 000 000.

### 3.2 Rail construction (at spawn)

Slide nodes are processed **after every beam-creating section**: after cinecam, wheels, beams, shocks, commands, hydros, triggers, ropes, ties and animators. Root lines come first, then the selected module's, in file order.

For each line:

1. **Collect the rail's node list** in the order written. A range a–b expands to every index from min(a, b) to max(a, b).
2. **For each consecutive pair (pᵢ, pᵢ₊₁)**, find the **lowest-index beam** among all beams spawned so far whose ends are {pᵢ, pᵢ₊₁}, in either order. This can be a plain beam, shock, command, hydro, trigger, animator, cinecam or wheel beam.
   * The segment is that beam.
   * Its direction is **the beam's own (a, b)**, not the rail list's order. The Mustang's rail (46, 28) resolves to the shock written `28, 46`, so its segment runs from 28 to 46.
3. If any pair has no beam, the rail is not built. The slide node is still created but has no rail and is **inert** for ever: no forces. RoR logs an error. Rubble matches this and warns.
4. A rail list with one node is undefined in RoR. Rubble rejects it.
5. Segments link to their neighbours in list order (previous and next).
6. If the first and last listed node are the same, the rail is a loop: the first and last segments are each other's neighbours.

Every slide node line builds its own rail. Rails are never shared, except through `railgroups`, which the Mustang does not use and Rubble refuses.

**Initial segment.** At spawn, the current segment is the segment nearest to the slide node's spawned position. Ties go to the earliest segment. The distance is defined in §3.3.

**Attachment state at spawn.** The node is attached to its rail from spawn. The current tolerance equals the initial tolerance (T, 0 by default). `slidenode_connect_instantly` and the attachment toggles (the C constraints, R, D) only change attachment through the user's slide-node lock action. That action is out of scope, so Rubble refuses those options. With the action never used, R and D have no effect.

### 3.3 Geometry

All positions are **world positions**: origin + relative position, f32.

For a segment with ends P1 (beam end a) and P2 (beam end b), and a point X:

* d = P2 − P1; β = |d|. Normalise d: if β > 0, d ← d·(1/β), with the reciprocal computed first; if β = 0, d stays zero.
* τ = max(0, min((X − P1)·d, β)).
* The nearest point is Q = P1 + d·τ.
* Its distance is |Q − X| (f32 correctly rounded square root).
* The ratio is ϱ = τ/β if β > 0, else 0.

### 3.4 The per-step update (after self-collision)

Each step, after the beam pass and self-collision, for each slide node in spawn order:

1. **Skip conditions.** If the node has no rail, or its current segment's beam is broken, skip it. It produces no force, and it never changes segment again. **(quirk)**
2. **Segment hop.** Let D₀ be the distance to the current segment. If the current segment has a previous neighbour whose distance is strictly less than D₀, it becomes the candidate. Then, if there is a next neighbour whose distance is strictly less than the candidate's distance, the next neighbour becomes the candidate. The candidate becomes the current segment.
   * The node moves at most one segment per step.
   * Whether a neighbour's beam is broken is not checked.
3. **Ideal point.** Compute the ideal point Q and the ratio ϱ on the (possibly new) current segment.
4. If the node's *broken-away* flag is set, skip the force: steps 5–7.
5. **Force.** w = Q − X, where X is the node's world position. δ = |w|; normalise w. f_len = −k·max(0, δ − θ), where k is the spring rate and θ the current tolerance. P = w·f_len.
6. **Break test.** If |P| > the break force, set *broken-away*. The force below is still applied in this step.
7. **Apply.**
   * The slide node gets −P, a pull of magnitude k·(δ − θ) toward Q.
   * The segment's end a gets P·(1 − ϱ).
   * The segment's end b gets P·ϱ.
   * These are evaluated as the vector times the f32 scalar.

There is **no damping term** and no velocity dependence. Forces accumulate like beam forces and are integrated in the next step's node pass.

The tolerance decreases only while it is above the initial tolerance (θ ← θ − R·h each step), and only after an attachment action. With no such action θ stays constant.

### 3.5 Mustang slide nodes

| slide node | rail (segments as found) | spring k |
|---|---|---|
| 48 | shock (28, 46) | 9 000 000 |
| 49 | shock (37, 38) | 9 000 000 |
| 31 | beam (32, 51) | 9 000 000 |
| 39 | beam (40, 50) | 9 000 000 |
| 54 | beam (53, 61) | 9 000 000 |
| 62 | beam (53, 61) | 9 000 000 |
| 408 | shock (63, 59) | 9000 |
| 409 | shock (68, 56) | 9000 |
| 45 | beam (407, 406) | 9000 |
| 47 | beam (407, 406) | 9000 |
| 421 | beams (279, 269), (269, 257), (257, 255), (255, 235) | 900 000 |
| 422 | the same four beams | 900 000 |
| 427 | beam (423, 424) | 9000 |
| 428 | beam (426, 425) | 9000 |

For each "beam" entry, the segment direction is the (a, b) of the first matching line in `beams`. Check it from the spawn table, not this listing.

None of these has a break force, so a slide node breaks away only when its rail segment's beam breaks.

Note that wheel axle nodes (31, 39, 54, 62) are slide nodes. Their spring acts on the axle every step.

### 3.6 Spawn-order position

Slide nodes create no nodes and no beams. They do not affect node masses.

---

## 4. `animators`

### 4.1 File format

`n1, n2, factor, options`

* The line is split on commas only; it needs 4 fields.
* The fourth field is split on `|`, and each token is trimmed.
* Recognised tokens:
  * `tacho`, `speedo`, `accel`, `brakes`, `clutch` and so on: data sources;
  * `inv` invisible;
  * `vis`;
  * `shortlimit: x` and `longlimit: x`.
* Unknown tokens are ignored.

The Mustang's line is `279, 421, 0.5, tacho | inv`: factor 0.5, source tacho, invisible.

### 4.2 The beam

Animators spawn **after triggers, ropes, hooks, ties and ropables**, so after every other beam-creating section in this document. The Mustang's animator is the last beam.

| property | value |
|---|---|
| kind | plain (it counts as a live plain beam for the break exception) |
| type | actuator (never yields) |
| k | springiness × spring scale |
| c | damping × damping scale |
| S | break × break scale (**scaled**) |
| G = Tc = Tt | the ordinary scaled deform threshold |
| L = L₀ = L_ref | spawn length |
| short limit | 0.99999 (f32), unless `shortlimit` is given |
| long limit | 1 000 000, unless `longlimit` is given |
| node masses | takes part in the dry-mass distribution |

Mustang (scale 0.05, 0.005, 0.01, 1.0 in force): k = 450 000, c = 60, S = 1 000 000, G = 4000.

So it breaks only when |F| > 1 000 000.

### 4.3 Per step (the hydro pass, before commands)

The animator is appended to the vehicle's list of hydros, after all `hydros` lines. Each step, in the hydro pass, which runs after the wheel and shock passes and before the command pass:

* Start with s = 0 and a divisor n = 0. The animator has no hydro input flags, so it adds nothing here.
* **`tacho`** applies **only if the vehicle has an engine object**: s ← s − rpm / max_rpm, and n ← n + 1.
  * max_rpm is |the `engine` line's second argument|, 5800 for the Mustang.
  * Other sources follow the same pattern; they are out of scope and Rubble refuses them.
* If n = 0: **L is not touched.**
* If n > 0:
  * s ← s / n;
  * the inertia filter is the identity, as no `set_inertia_defaults` with both delays > 0 precedes the line;
  * φ = 1.0 − s·factor, computed as [f64] 1.0 minus the f32 product, rounded to f32;
  * clamp φ: if φ < 1 − short limit (f32), φ ← 1 − short limit; if φ > 1 + long limit, φ ← 1 + long limit;
  * L ← L_ref·φ.

### 4.4 What the Mustang's tacho does with no engine running

**It has no effect on its rest length**, in both cases that "no engine running" can mean:

* **No engine object.** This is Rubble's current model and the reference harness's. n = 0, so L is never written.
* **An engine object that is off** (rpm = 0). s = −0/5800 = 0 and φ = 1 exactly, so L = L_ref·1 = L_ref every step.

The animator therefore behaves as a plain actuator beam of constant rest length (k = 450 000, c = 60) between nodes 279 and 421. It is not physically inert: its spring–damper force acts every step, and its mass contribution counts.

In real RoR gameplay the engine is normally **started at spawn**, with rpm = the idle rpm, 600 from `engoption`. Then φ = 1 + 0.5·rpm/5800 lengthens the beam by about 5.2% at idle. That pushes node 421 along its rail toward the trigger bounds of §2. Engine-on parity is out of scope. If it is added later, rpm becomes an input to this formula, used in exactly this place.

---

## 5. Sections without physical effect (confirmed)

* **`lockgroups`** sets each listed node's lock group (`lockgroup_default_nolock` sets the default to "no lock"). The value is read only when a vehicle's **hook** looks for nodes to lock onto. With no hooks in the scene it changes nothing. Rubble parses and ignores it.
* **`videocamera`** is created during graphics setup: a render-to-texture mirror camera that reads node positions only. Rubble ignores it.
* **`flexbodies`** (and `forset` / `forvert`) are visual meshes deformed from node positions. Their node references are resolved, and a missing node only produces a placeholder mesh. They add no nodes, beams, mass or forces. Rubble ignores them, including inside modules.

Also ignored, as visual only:

* the meshwheels' mesh, side and material;
* the `i` and `inv` invisibility flags;
* trigger and command descriptions.

---

## 6. Position in the step (consolidated)

Per step, per vehicle. The RoR order, restricted to what Rubble simulates:

1. **Node pass**: contacts, integration, force restart (existing).
2. **Wings**, then **fuselage drag** (aero spec).
3. **Wheels and shocks**: no force with no input and no active shocks.
4. **Hydro pass**: `hydros` (with no input L = L_ref, an exact no-op), then **animators** (§4.3).
5. **Command pass** (§2.6). It reads the trigger inputs set by the **previous** step's beam pass, then clears them.
6. **Beam pass** (existing), including the **trigger test** (§2.5) at each trigger beam's index. Triggers read this step's lengths and set the inputs for the next step.
7. **Self-collision** (existing).
8. **Slide nodes** (§3.4), in spawn order. They read the world positions from this step's node pass and add forces for the next step's integration.

Per frame there is nothing new. The origin rebase leaves world positions unchanged, so slide nodes are unaffected by it.

**GPU notes.**

* A trigger is a beam with k = c = 0 and a per-beam side record holding sb, lb, the two keys and *enabled*. The trigger test writes a per-key flag, an idempotent store of 1; no atomics are needed beyond a plain store.
* The command pass reads and clears those flags in one lane after a barrier.
* Slide nodes are a separate stride over a slide-node table after self-collision. Their forces go into per-node accumulators like contacts: the fixed-point integer atomics of `explicit-beams.md`, or a dedicated incidence slot. The sums are order-independent.
* The segment hop is per-lane data, and each slide node's state (current segment, broken-away flag) is owned by one lane. No barrier sits under a data-dependent branch.

---

## 7. Edge cases

* **Trigger bounds.**
  * A trigger with lb < −sb has no inside region (§2.5).
  * A trigger with sb = 0 and lb = 0 (shock style) fires the long key whenever e > 0 and the short key whenever e < 0. At e = 0 exactly it fires neither.
* **Disabled or broken trigger beams** are skipped. Their key simply stops being held one step later.
* **Two commands on the same key**: their entries move in spawn order. A command with contract key = extend key gets both entries on one key; the contraction entry runs first, then the extension entry.
* **Rail beams.**
  * A rail segment whose beam is a trigger beam is valid: a geometric segment with zero force.
  * A rail segment whose beam later breaks stops the slide node only if it is the current segment at that time (§3.4 step 1).
* **Zero-length segment.** If β = 0: the ideal point is P1, ϱ = 0, and all of the reaction goes to end a.
* **Slide node exactly at its ideal point**: δ = 0, w stays zero, and the force is zero.
* **Modules with `nodes` or `beams`** interleave as in §1.3. Their legacy node numbers follow §1.4. A module's `beams` can reference root nodes, and root slidenodes can use a module's beams as rails, because slidenodes are processed after all beams of all selected modules.

---

## 8. Verification plan (black-box against `inspiration/ror-harness/bin/ror_ref`)

**Harness work first** (its owner, RoR code verbatim; in progress now):

* Accept a section-configuration name, defaulting to RoR's rule: the first module name in byte order. Spawn the selected modules through RoR's own keyword-by-keyword flow.
* Stop refusing `triggers`, `slidenodes` and `animators`. Run, verbatim:
  * the trigger, slide-node, animator and command spawn;
  * the hydro pass with animators;
  * the command pass with trigger input;
  * the trigger test inside the beam pass;
  * the slide-node update.
* Keep `engine` as "no engine object" by default. Optionally add an `--engine-off` mode that creates the engine object with rpm 0 and never runs it.
* Export:
  * the node table and the beam table after spawn, including the beam kind (trigger), short and long bounds, and each beam's command-entry key, rate and boundary;
  * per step: beam L for command, hydro and animator beams; each key's command value; each trigger's side (long, short, inside); each slide node's current segment, ratio ϱ and broken-away flag; per-node force after the slide-node pass.

Then Rubble must match:

1. **Section selection and order.**
   * Load the Mustang with no configuration, then with each of the three names. Compare:
     * the chosen name (`ats_wheels` by default);
     * the node count (527);
     * node order and positions;
     * tyre-node friction (the §1.5 table);
     * beam order and parameters, bit-exact.
   * A synthetic file with a module whose `nodes` and `beams` sit between root sections. Check the per-keyword interleave.
   * A synthetic file whose `set_beam_defaults_scale` and `set_node_defaults` sit inside an **unselected** module and precede root lines. Check that the root lines carry them.
   * `set_beam_defaults_scale` with 3 numbers (ignored) and with 5 numbers (fifth ignored).
   * `sectionconfig` lines listed in reverse order: the default is unchanged.
   * A `section 0 a b` line: only `a` exists.
2. **Trigger spawn.**
   * The Mustang's two triggers: sb and lb bit patterns (0x3ECCCCCC, 0xBF19999A, 0x3ED70A3E, 0xBF000000), k = c = 0, S = 1 000 000, G = 4000, L.
   * A trigger with short_action 0 or 85: dropped, so no beam and no mass.
   * Node masses with and without the trigger lines: shows they enter the dry-mass distribution.
3. **Trigger logic, element level.**
   * Two 1 kg nodes, gravity and ground off. Node b immovable. Node a driven along the axis at ±0.1 m/s through both bounds.
   * Each trigger option set: none, `c`, and `c` with lb < −sb.
   * Check per step:
     * which key's command value is 1 in the next step (one-step latency; long side wins in the overlap);
     * the exact boundary step;
     * zero force on both nodes.
4. **Trigger → command.**
   * The same rig, with a `commands2` beam on the trigger's keys (contract on the long key with B = 1.0, extend on the short key with B = 1.5), rates 0.6/0.8 and 10.
   * Check L(t) per step: the start one step after crossing, ΔL per step, the stop on the first step past the boundary, the retract after leaving.
   * Then the same with two commands sharing a key, and with contract key = extend key.
5. **Slide nodes.**
   * A 3-segment rail (4 rail nodes on a fixed frame) whose second rail pair resolves to a beam written in reverse order. The slide node starts 1 cm off the rail; the spring is the default 9 000 000, then 9000.
   * Check per step the segment, ϱ, the force on the node and on both segment ends.
   * Check the hop at segment joints (one segment per step) and the end clamps (τ = 0 and τ = β).
   * A loop rail.
   * A missing rail beam: the slide node is inert.
   * A break force of 50 N: the force is applied on the breaking step, then zero for ever.
   * Breaking the current segment's beam: the node is stuck from then on.
6. **Animator.**
   * The Mustang's (279, 421) beam: its spawn row, and L constant for 10 s in both no-engine and engine-off modes.
   * An animator with `shortlimit: 0.02` / `longlimit: 0.05` in engine-off mode: L = L_ref.
7. **Lockgroups, videocamera, flexbodies.** The Mustang, and the same file with those three sections deleted, must give identical spawn tables and 5 s trajectories in RoR.
8. **Integration.**
   * The Mustang (`ats_wheels`, no engine, no input) dropped from 0.3 m onto flat ground for 5 s:
     * all node trajectories within the existing RoR-compatible tolerance;
     * command L unchanged; keys 16 and 83 held from step 2 on; no trigger reaching its short side.
   * Then a 50 km/h frontal wall strike for 2 s:
     * compare trajectories, broken and bent beam counts, and slide-node broken-away flags;
     * record whether any trigger flips. If one does, the command L(t) must match step by step.
   * Repeat the drop with `saleen_wheels`: front tyre friction 1.05.
