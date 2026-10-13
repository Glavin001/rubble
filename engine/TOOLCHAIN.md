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

# CUDA (NVIDIA Linux)
slangc -target cuda -stage compute -entry <name> -o out.cu in.slang
```

`engine-metal` / `engine-webgpu` / `engine-cuda` invoke these from `build.rs` and embed artifacts. The slang compiler is **not** linked into shipped binaries.

## Units and frames

- SI: meters, kilograms, seconds, radians
- Right-handed, **Y-up**
- Quaternions `(x, y, z, w)`
- Official demo Z-up scenes are converted in oracle adapters
