# Draft: calls `/5`, the oracle's full dynamic state (2026-10-22, to agree with P1 and P0)

**Principle (user, via the planning session).** Accuracy first, with the oracle's complete
contract.
- A call's inputs and outputs are the oracle's whole mutable state for the island.
- Every carried field gets a frame-start input, an output head, a frame-end target and a
  loss, and the decode passes it through to the engine.
- Simplification (dropping fields or priors) comes only after a model passes the gates,
  and as measured cost work.

**What "full" means.**
- P1's recorder writes every mutable `World` field, with a bit-exact one-frame resume test
  as the completeness proof.
- P0's `ceiling_full` (the full state carried between calls) is the closed-loop reference.
- The field list below follows `stress-ref`. P1's resume test makes it final, and **it also
  classifies each field** as one of:
  - **state**: resetting it breaks the bit-exact resume. It gets an input, a head, a target
    and a loss.
  - **derived**: recomputed from state, e.g. caches. It is an input feature only if it
    helps, never a head.
  - **diagnostic**: nothing reads it. It gets no input and no head.

## Norms and losses, by the physics of the field

Every loss is ÷ the call's `E_in` (J), and then normalised by its zero-predictor value, as
now. A field whose conjugate energy is known is scored in that energy:
- **rates:** the KE norm;
- **displacements and offsets:** the strain-energy norm with the stiffness that loads them;
- **damage-like history:** the energy its change dissipates;
- **dimensionless history with no clean energy:** its physical scale (e.g. κ ÷ κ_fail),
  weighted by the bond's fracture energy G_f·A ÷ E_in.

Heads predict **increments**. Monotone fields use non-negative increments (softplus), so
monotonicity is exact by construction; the damage head already works this way.
Discrete fields use cross-entropy or BCE.

## Per entity

### Chunk (per chunk, in its cluster's body frame)

| field | input | head / decode | target and loss |
|---|---|---|---|
| u, θ (`ChunkState.u/th`) | ÷ u_c, θ·R ÷ u_c (as `/4`) | Δu, Δθ; decode adds them | `y_chunk_def_end`, exact. Strain-energy norm over the chunk's live bonds: Σ_b ½ k_b (Δelongation error)² |
| v, w (`ChunkState.v/w`) and the cluster's v, ω (via the per-chunk world v, w) | as `/4` | as now (per-chunk v, w; decode keeps them unreduced) | KE norm, `motion` + `def_motion` |
| inertia_scale | as is | pass-through unless the resume test calls it state that changes in a frame | — |
| a, α, step_start (implicit mode only) | only if the profile is implicit (`blast_parity` is explicit: absent) | — | — |
| active, removed, cluster | topology (graph and body id) | breaks → fragments; `removed` as BCE if it can change in a frame | fragments as now |
| reaction (support force/moment, last substep) | ÷ (m·g) and (m·g·R) | head if state | KE-consistent: impulse norm (reaction·dt)² ÷ (m·E_in) |

### Cluster (body)

| field | input | head | loss |
|---|---|---|---|
| pose, com, mass, inertia, v, ω | derived from the chunks (P0's momentum-exact fit) | no head: the decode derives them | — |
| activity (Active/Settled), explicit_window, driven | flags | BCE heads | BCE |
| active_timer | ÷ dt | Δ head | Huber on Δ ÷ dt |
| window_utilization, settled_load_norm | log1p, and ÷ (m·g) | heads if state | Huber in log space |

### Bond (per live bond: every `JointState` field, plus the `RtBond` runtime fields)

| field | input scale | head | loss |
|---|---|---|---|
| damage D, crush Dc | as is | softplus Δ (D already) | energy dissipated: G_f·A·(ΔD error) ÷ E_in, Huber |
| kappa, kappa_c (equivalent-strain history) | ÷ κ_fail | softplus Δ | G_f·A·(Δκ error ÷ κ_fail) ÷ E_in |
| ductility, ductility_c (∞ at rest) | 1 ÷ (1 + ductility) (∞ → 0) | Δ on that encoding | Huber, weighted G_f·A ÷ E_in |
| fatigue ω | as is (0–1) | softplus Δ | Huber, weighted G_f·A ÷ E_in |
| plastic (5 used: slide x, y; rock x, y; twist) | ÷ u_c (lin), ÷ θ_c (ang) | Δ | strain-energy norm ½ k_t (Δ error)² per component ÷ E_in |
| rebar_plastic, rebar_slip[2], rebar_work, rebar_broken | ÷ u_c; rebar_work ÷ GA | Δ heads; rebar_broken BCE | strain-energy norm with the rebar stiffness; rebar_work in J ÷ E_in |
| strain_rate | sign · log1p(\|ε̇\|·dt) | Δ in that encoding | Huber |
| governing_stress | ÷ f_t | Δ | Huber |
| mode (failure mode) | one-hot | categorical head (broken bonds) | CE (as `fail_mode` now) |
| dissipated | ÷ GA | softplus Δ | in J ÷ E_in |
| utilization, peak_utilization, fail_ts maxima | log1p | log r head (now) | as now (demand / margin) |
| force, elastic (Local6), measures, stored, damped | derived or diagnostic, per the resume test | head only if state | strain-energy norm with the bond stiffness (`force` via k⁻¹) |

### Contacts (persistent per-pair state)

| field | representation | head | loss |
|---|---|---|---|
| `pair_offsets[ChunkPair]`: pre-existing overlap per sample point | on the structure-contact edges: a fixed-size slot list per pair (P1 to give the max sample count) with a validity mask | Δoffset per slot, plus "still in contact at frame end" (BCE) | strain-energy norm ½ k_c (Δ error)² per slot ÷ E_in |
| `ground_offsets` (chunk or impactor vs ground) | a ground-contact edge type (new) | same | same |
| `sticks[ContactKey]`: shear-layer slip and twist | on the contact edge | Δslip, Δtwist, and survival BCE | ½ k_t (Δslip error)² + ½ k_θ (Δtwist error)² ÷ E_in |
| neighbours (broadphase skin list) | derived (rebuilt from poses) | — | — |
| `ContactLedger` (energy accounting) | diagnostic for the ledger; energy targets as now | — | — |

New pairs created during the frame are frame-end outputs without a frame-start input. They
come from the routing sweep's predicted contacts: a head on the predicted
structure-contact and impactor-contact edges gives their frame-end slot state.

### Impactor

| field | input | head | loss |
|---|---|---|---|
| pose, v, ω | as now | as now | KE norm |
| crush_used | ÷ crush energy | softplus Δ | J ÷ E_in |
| crush_depth | ÷ radius | softplus Δ | Huber |

### World-level

Frame counter, events_done, probe accumulators, implicit loads, exposure, blast caches,
the topology version: scene bookkeeping and caches. The resume test decides whether any of
them is state *for an island call*. Expected: none, except blast caches if a blast spans
frames.

## Decode and decoded state

- `learned-impact-decoded-state/2` carries every **state** field per chunk, bond, contact
  slot and impactor. The host writes them into the `World`, as P0's `ceiling_full` does
  from the oracle.
- Momentum projection and the KE bound stay as now. The bound counts kinetic energy
  only, as P0 found sufficient. Energy-consistent losses keep the stored-energy fields
  honest.

## Order of work

1. P1: the resume test's classification of every field (state / derived / diagnostic),
   and for contacts the max sample points per pair and the key sets.
2. P0: confirm `ceiling_full` restores exactly the state set, and say whether anything
   outside `World` (e.g. the engine's own contact cache) also has to be carried.
3. P2: freeze the `/5` columns, heads and losses from that set, then the loader, model,
   decode and provider, with a synthetic test per field.
