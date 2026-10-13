# Local oracle baselines (this machine)

**Host:** Apple M3 Max, macOS (darwin 23.x).  
**Policy:** Accuracy first. Speed bars come from **local** runs only — never published numbers from other GPUs.

## Top oracles selected

| Rank | Oracle | Role | Local status |
|------|--------|------|--------------|
| 1 | `inspiration/pinned/three-avbd` | Primary GPU AVBD + CPU f64/CpuRef | CPU tests **81/81 pass** (Node 24). Stack scene settles (xy≈0, v≈0 after 900 frames). Headless Dawn GPU **blocked**: dawn.node built for macOS 26. GPU timing → Chrome WebGPU later. |
| 2 | `inspiration/pinned/avbd-demo3d` | Official C++ numerical / params | Collide/manifold source for our SAT port. |
| 3 | (optional) `avbd-sandbox` GPU | Second GPU orchestration check | Only after three-avbd accuracy + Chrome GPU baseline. |

## Commands (local)

```bash
# three-avbd CPU accuracy (requires Node ≥22.7 with --experimental-transform-types)
export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"; nvm use 24
cd inspiration/pinned/three-avbd && corepack enable && pnpm test
# Stack only:
node --experimental-transform-types -e "..." # see agent notes / avbd3d-ref.test.ts

# our engine
cd engine
cargo test -p engine-metal --release --test oracle_accuracy
cargo run -p engine-metal --example stack_diag --release -- 5 200
```

## Accuracy vs three-avbd (local)

| Case | three-avbd CPU (local) | engine Metal (local) |
|------|------------------------|----------------------|
| Stack 10, 900 frames, spaced drop | PASS (z≈i+1−margin, xy=0) | FAIL — tips after impact (Jacobi + per-frame contact regen) |
| Pre-stacked 5 boxes, 200 frames | n/a (their scene is spaced) | PASS gate in `oracle_accuracy` |
| Pyramid finite | PASS | PASS |
| Rope joint bound | PASS | PASS |

### Root gaps vs three-avbd CPU (not perf)

1. **Solver schedule:** they use sequential GS over bodies; we use graph-colored Jacobi.
2. **Manifold persistence:** they keep `Manifold` + stick anchors across frames; we rebuild contacts each step (warmstart match helps but stick anchors not retained).
3. **Ground:** their Stack uses a static box; plane path is OK for short stacks.

## three-avbd CPU log

See `LOCAL_three_avbd_cpu_test.log` (81 passed, 2026-10-13).

## Speed baselines

_Empty until Chrome/WebGPU or newer Dawn runs on this Mac._ Fill `ours_ms` vs `oracle_ms` only from that local run.
