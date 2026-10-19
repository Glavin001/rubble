# Learned-impact weight fixture (Slang / WebGPU)

Format: `learned-impact-slang/1` — flat little-endian `f32` blob + JSON index.

| File | Role |
|---|---|
| `weights.bin` | Packed floats for `engine/slang/kernels/learned_impact.slang` |
| `weights.json` | `h`, `p`, `n_floats`, `n_bytes`, per-tensor offsets |

Source checkpoint for this fixture: house prove-train (`H=48`, `P=16`). Re-export after training:

```sh
python -c "from pathlib import Path; import torch; from learned_impact.export import export_slang_weights; \
  export_slang_weights(torch.load('state_dict.pt', map_location='cpu', weights_only=True), \
  Path('.'), h=48, p=16)"
```

See `fracture/README.md` (Bond modes → Learned impact) and `learned_impact/export.py`.
