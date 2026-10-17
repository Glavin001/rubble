# Engine toolchain

Pinned for the MacBook M3 Max proof path (Slang → Metal).

## Host

| Tool | Version / revision |
|------|--------------------|
| Chip | Apple M3 Max (record `sysctl -n machdep.cpu.brand_string` with every bench) |
| macOS SDK | 14.5+ |
| Rust | stable (workspace `rust-toolchain.toml` inherits repo root or pins stable) |
| slangc | **2026.19** at `~/.local/opt/slang-2026.19/bin/slangc` (symlink `~/.local/bin/slangc`) |

Install slangc:

```bash
curl -sL -o /tmp/slang.zip \
  https://github.com/shader-slang/slang/releases/download/v2026.19/slang-2026.19-macos-aarch64.zip
mkdir -p ~/.local/opt/slang-2026.19 ~/.local/bin
unzip -q /tmp/slang.zip -d ~/.local/opt/slang-2026.19
ln -sfn ~/.local/opt/slang-2026.19/bin/slangc ~/.local/bin/slangc
export PATH="$HOME/.local/bin:$PATH"
slangc -v
```

Linux x86_64 (CI, the Modal build image, `engine-cuda`'s PTX): the pinned
glibc-2.28 slangc plus nvcc only, no driver:

```bash
sudo engine/tools/cuda/install_linux_toolchain.sh
export SLANGC=/opt/slang/bin/slangc NVCC=/usr/local/cuda-12.8/bin/nvcc
```

## CUDA tests on Modal

When nvcc is present at build time, `engine-cuda` embeds a fatbin per kernel:
SASS for `ENGINE_CUDA_SASS` (default `75,89`: T4, and the Ada L4 / L40S / RTX
4090) so those GPUs never JIT, plus PTX for `ENGINE_CUDA_ARCH` (default
`compute_75`) for anything else. Its binaries then need only the NVIDIA driver.
On macOS it builds without nvcc: the CUDA sources are still generated and
checked, and the GPU tests skip. `ENGINE_CUDA_DEVICE=1` makes them run (and fail
when no device loads); `ENGINE_CUDA_PERF=1` adds the timed 10k-body run.

The build fails if Slang's CUDA output reads a buffer element after storing to
it (a Slang miscompile, DECISIONS 8ez); move the read before the store in the
`.slang` source.

Modal only runs the GPU part: nothing is compiled or uploaded there except the
test archive and the runner script, and the binaries run in a single-use GPU
container that exits as soon as they finish (a correctness run is about 6 s of
a T4, ~$0.001). Builds are always free: native on Linux x86_64, the GitHub
runner in CI, and locally in Docker:
`engine/tools/cuda/build_tests_in_docker.sh` compiles in an x86_64 Ubuntu 22.04
container (the CI toolchain; emulated on Apple silicon) with cargo's registry
and target dir in Docker volumes. The first run builds the image (~2 min) and
compiles (~70 s); after that, an unchanged tree takes ~5 s and a Slang edit
~70 s (every kernel recompiles).

`engine/tools/modal/modal.sh` runs the pinned Modal CLI in Docker, so nothing
is installed on the host. It mounts the repo, and keeps the login token in the
`rubble-modal-config` volume. Run from the repo root:

```bash
engine/tools/modal/modal.sh setup                              # once: open the printed link
engine/tools/modal/run_cuda_tests.sh                           # local build, T4 run
engine/tools/modal/run_cuda_tests.sh --perf --gpu L40S         # timing on the RTX 4090's die
```

`run_cuda_tests.sh` skips the GPU run when the same archive (packaged
byte-reproducibly) already passed with the same arguments and build-time
`ENGINE_*` variables (`engine/target/cuda-green`); `--force` reruns it. CI
skips the same way by input hash. On Linux x86_64 with slangc and nvcc,
`modal.sh run engine/tools/modal/cuda_tests.py` builds natively instead of
taking `--archive`.

Debugging a GPU failure, cheapest first (each test runs in its own process, so
one CUDA fault does not fail the rest):

| Switch | Effect |
|---|---|
| `--filter NAME` | Only tests whose names contain NAME; prints their output |
| `--env K=V,...` | `ENGINE_*` variables for the build and the tests, e.g. the WebGPU host's control values (`ENGINE_COLOR_HEADROOM=full`, `ENGINE_REPAIR=rounds`, ...) or `ENGINE_CUDA_TRACE=1` (per-step counters) |
| `--sync` | `ENGINE_CUDA_SYNC=1`: wait after every kernel, so a fault names its kernel and step |
| `--env ENGINE_CUDA_STOP_AFTER=kernel@step` | End that step right after the kernel, so the counters and buffers show its output |
| `--sanitize memcheck` (or `racecheck`, `synccheck`, `initcheck`) | Run under compute-sanitizer: faulting access with its Slang source line |

`.github/workflows/engine-cuda.yml` builds on the free runner and calls Modal on
pushes to main, PRs labelled `cuda`, and manual runs (repo secrets
`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`), skipping inputs that already passed.

## Inspiration (gitignored `inspiration/pinned/`)

| Repo | Pinned revision | Role |
|------|-----------------|------|
| sbobyn/three-avbd | `2bb1a318e9aad03640b4bce5b5d222484f0925f7` | Primary 3D GPU AVBD oracle |
| savant117/avbd-demo3d | `7701bd427d55ca5d03ea1fdf331912ded9169f4b` | Numerical definition |
| itchyfeetleech/avbd-sandbox | `44cdeea0a5134f5f909a4475c12c67ae07665dd1` | Anchors / tangent transport |
| Jaisiero/beat-box | `b212b1af546750a0cd50c43652f073d06836062a` | Slang kernel structure |
| newton-physics/newton | `fb92f6d84e4a72796d95cefbb378ba6b6be72f04` | Joint semantics |
| jrouwe/JoltPhysics | `v5.3.0` (`0373ec0dd762e4bc2f6acdb08371ee84fa23c6db`) | Independent rigid-body oracle |

Do not copy solvers. Read, then reimplement in `engine/slang/`.

## Compile targets

```bash
# Metal (primary proof)
slangc -target metal -stage compute -entry <name> -o out.metal in.slang

# SPIR-V (WebGPU via naga → WGSL; Slang WGSL atomics are limited)
slangc -target spirv -stage compute -entry <name> -o out.spv in.slang

# CUDA (NVIDIA Linux; engine-cuda adds -DENGINE_WGSL=1, then nvcc -ptx)
slangc -target cuda -stage compute -entry <name> -o out.cu in.slang
```

`engine-metal` / `engine-webgpu` / `engine-cuda` invoke these from `build.rs` and embed artifacts. The slang compiler is **not** linked into shipped binaries.

## Units and frames

- SI: meters, kilograms, seconds, radians
- Right-handed, **Y-up**
- Quaternions `(x, y, z, w)`
- Official demo Z-up scenes are converted in oracle adapters
