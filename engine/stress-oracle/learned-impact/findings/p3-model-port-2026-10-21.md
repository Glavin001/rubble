# P3 groundwork: P2's model in Rust and Slang, with parity on fixed weights (2026-10-21)

Request (planning session): port the trained model's exact architecture (P2's
`learned_impact/outcome/model.py`) to Slang starting from the P0 `gnn_probe`. Load P2's
exported weights, check against PyTorch on fixed weights, include the per-node
first-layer projection lever, and defer timing while P2 trains on the GPU. No timing was
taken; the GPU checks below are one correctness run each.

## What exists

| Piece | Where | Checked against |
|---|---|---|
| Call-record reader (`learned-impact-outcome/3`, plain or zstd) | `stress_learned::outcome_model::read_event` | P2's export calls |
| Weights loader (`learned-impact-weights/1`) | `outcome_model::Weights` | P2's `sweep-all3` export (288 tensors) |
| BSMS hierarchy: bi-stride selection with P2's rotation-stable seeds (`tie_tol` 1e-5), two-hop support, `live_edges` crack masking, compact per-call ids | `outcome_model::Hierarchy`, `call_hierarchy` | PyTorch outputs (through the U) |
| Forward pass: encoders, typed fine rounds, BSMS U (pool, coarse rounds with body-frame fibres, unpool, skips), global node, symmetric bond head, node heads with the variant-B residual, energy head | `outcome_model::Model::forward` | PyTorch CPU outputs |
| Decode: `to_physical`, causal freedom, free-island momentum projection (6×6), energy bound (free and anchored), breaks (log q ≥ log τ) and bins, survivor damage clamp | `stress_learned::outcome_decode::decode` → `stress_ref::world::DecodedState` | P2's `decoded.json` |
| The model as a GPU program: ops over an arena, packed k-major weights and a gather/CSR table | `stress_learned::outcome_program::compile`, `Program::run_cpu` | the Rust reference |
| Slang kernels `gnn_mlp`, `gnn_csr`, `gnn_fibre` and their executor | `stress-gpu/shaders/outcome_gnn.slang`, `stress_gpu::outcome_gnn` | the Rust reference, on the GPU |

The fixture `crates/stress-learned/tests/fixtures/p3-model/` (1.9 MB) holds P2's export
(`weights.bin` / `.json`) and 15 calls:
- **0–3:** P2's own parity calls: early free-wall calls, every bond intact, no cracked or
  structure-contact edges.
- **x0–x4:** P1 evaluation calls that cover the rest:
  - 15 cracked and 44 structure-contact edges;
  - broken bonds;
  - anchored walls;
  - 0, 3 and 4 impactors.
- **g0–g2:** zero-gravity variants, so the decode's momentum projection runs on islands
  of several bodies.
- **k0–k2:** P2's outputs with the node heads' Δv and Δω scaled 30× before decoding, so
  the energy bound acts on both a free and an anchored island. Only the decode is checked
  on these.

`learned-impact/scripts/p3_fixture.py` builds the fixture, running P2's checkpoint through
P2's own package (`--extra`).

## Results

- **Forward pass, Rust (f32) against PyTorch (CPU):**
  - worst relative difference 1.6e-6 over all heads on the 12 unmodified calls;
  - identical break sets.
- **Decode on PyTorch's outputs against P2's decoded states:**
  - motion to 1e-9 (f64);
  - identical chunk and impactor keys, break sets and time bins, damage values and flags;
  - covers 10 free islands (projection) and 3 energy-bound activations.
- **Rust forward then Rust decode against P2's decoded states:**
  - identical breaks and flags;
  - motion within 1e-5 of each field's largest value (f32 forward noise).
- **Node-projection lever** (an edge MLP's first layer as `W_e·e + (W_s·h)[src] + (W_d·h)[dst]`,
  the node blocks projected once per node): the same to 1e-6, on the CPU and on the GPU.
- **Folding the input standardisation into the first layer is not lossless in f32.** Some
  bond columns have σ ≈ 8.5e-6 against |μ| up to 12, so `W/σ` grows about 1e5×, and the
  folded dot product loses 4.4e-3 to cancellation. This contradicts the model's own comment
  ("the GPU port folds it into the first layer"). Standardisation stays a separate step (one
  subtraction and one multiply per input, staged with the row).
- **The GPU program on the CPU** (`run_cpu`, the kernels' arithmetic) matches the
  reference to 1e-5 on every call, with the lever on and off.
- **The GPU, on all three shader paths** (Metal library, MSL, WGSL):
  - matches the reference to 2e-6, with identical break sets;
  - 68–126 dispatches per call; the lever adds the projection dispatches.

Tests:
- `cargo test --release -p stress-learned --test outcome_model`: 5 tests.
- `cargo test --release -p stress-rubble --features gpu --test outcome_gnn_gpu`: 1 test,
  also run with `STRESS_GPU_SHADERS=wgsl` and `=msl`.
- `scripts/compile_shaders.sh --check`: current.

## What is not done (next)

- **Timing**, deferred while P2 trains. The kernels are written for correctness:
  - one row per slot of W lanes, with sequential k loops;
  - one dispatch per MLP, CSR or fibre op;
  - a uniform buffer per op.

  The P0 probe's tiled GEMM layout (`gnn_mlp_tiled`) is the starting point for the fast
  path, and the program's op list is what it would consume. The lever's value has to be
  measured: it replaces 2·W·W multiply-adds per edge with the same per node, at the cost
  of two dispatches.
- **Batching several calls in one program.** The ops already carry per-row gathers. The
  compiler emits one call, so the global node's per-call broadcast and means need
  per-call offsets.
- **Decode on the GPU.** The decode is host-side Rust (f64); its per-island 6×6 solve is
  small.
- **Wiring into the closed loop** (`--provider` in-process Rust model). The pieces exist:
  `live_call` builds the `CallEvent`, then `Model::forward` (or the GPU program), then
  `decode`, then `DecodedState`.
- **Only `mlp_hidden_layers` 1 and widths 32 and 64 are compiled to the GPU.** The CPU
  reference takes any depth. P2's default and its sweep winner are W32 with one hidden
  layer.
