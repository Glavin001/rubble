# WebGPU ≤10 storage buffers (inventory → packing)

Chrome `maxStorageBuffersPerShaderStage` = **10**. Native Metal allows 31.

## Inventory (2026-10-19)

| Shader set | Storage binds | Status |
|---|---:|---|
| `layout.slang` | 26 | **must pack** (GPU split layout) |
| `pass.slang` | 24 | **must pack** |
| `spawn.slang` / `colliders.slang` | 16 | must pack |
| `split.slang` | 13 | must pack |
| `passlog.slang` | 12 | must pack |
| `coupling.slang` / `punch.slang` | 11 | −1 each |
| `world.slang` (`WORLD_BINDINGS`) | **10** | already at limit (2 uniforms + 10 storage) |

Pipelines over 10 are rejected in the browser → stress mode steps but **nothing breaks**.

## Approach

Merge buffers with **base offsets** in Slang + host bind lists. Do **not** use per-kernel bind groups (tried; shared set still exceeds 10; reverted).

First targets: layout `old_*` / live float4 tables → two packed buffers; then pass; then spawn/split/colliders; then coupling/punch 11→10.

## Implementation status

Packing in progress (follow-on agent). Update this file with before/after counts when packs land.
