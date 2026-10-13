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
| Stack 10, 900 frames, spaced drop | PASS (z≈i+1−margin, xy=0) | **PASS** (`oracle_accuracy`, 40 iters, sleep off) |
| Pyramid finite | PASS | PASS |
| Rope joint bound | PASS | PASS |
| Seated sphere on plane | n/a | PASS (`avbd_slice`); fast drops → CCD |

### Root gaps vs three-avbd CPU (not perf)

1. **Fixed:** step order NP→warmStart (C0 at x⁻); Contact ABI 128 B; cReg + margin; massPenalty; small-scene GS.
2. **Still open:** high-speed sphere–plane (CCD stretch); beat local GPU oracle ms (after accuracy).

## three-avbd CPU log

See `LOCAL_three_avbd_cpu_test.log` (81 passed, 2026-10-13).

## Speed baselines

_Empty until Chrome/WebGPU or newer Dawn runs on this Mac._ Fill `ours_ms` vs `oracle_ms` only from that local run.
