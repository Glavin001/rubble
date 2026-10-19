"""Flatten a state_dict to a binary blob + JSON config for Slang / WebGPU load."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .constants import K

# Must match engine/slang/kernels/learned_impact.slang and fracture/learned-impact.mjs
SLANG_FORMAT = "learned-impact-slang/1"
H_DEFAULT = 48
P_DEFAULT = 16
NODE_IN = 46
EDGE_IN = 28
FFN_MULT = 4
MP_LAYERS = 4
ATTN_BLOCKS = 1
FAIL_BINS = K + 1


def slang_weight_count(
    h: int = H_DEFAULT,
    p: int = P_DEFAULT,
    *,
    node_in: int = NODE_IN,
    edge_in: int = EDGE_IN,
    attn_blocks: int = ATTN_BLOCKS,
    mp_layers: int = MP_LAYERS,
) -> int:
    """Float count of the packed Slang blob (encode + MP + attn + bond heads)."""
    h2 = FFN_MULT * h
    enc_n = h * node_in + h + h * h + h
    enc_e = h * edge_in + h + h * h + h
    mp_e = h * 3 * h + h + h * h + h
    mp_n = h * 2 * h + h + h * h + h
    attn = p * h + p + h + 1 + 3 * h * h + 3 * h + h * h + h + h2 * h + h2 + h * h2 + h
    dec = FAIL_BINS * h + FAIL_BINS + h + 1
    return enc_n + enc_e + mp_layers * (mp_e + mp_n) + attn_blocks * attn + dec


def _append(blob: bytearray, order: list[dict], name: str, tensor: torch.Tensor) -> None:
    arr = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
    order.append({"name": name, "shape": list(arr.shape), "dtype": "f32", "offset": len(blob)})
    blob.extend(arr.tobytes(order="C"))


def _linear_w_b(sd: dict[str, torch.Tensor], prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    return sd[f"{prefix}.weight"], sd[f"{prefix}.bias"]


def _mlp_pack(sd: dict[str, torch.Tensor], prefix: str) -> list[tuple[str, torch.Tensor]]:
    """Pack MLP Sequential: Linear → act → Linear as W0,b0,W1,b1 (row-major)."""
    w0, b0 = _linear_w_b(sd, f"{prefix}.net.0")
    w1, b1 = _linear_w_b(sd, f"{prefix}.net.2")
    return [
        (f"{prefix}.net.0.weight", w0),
        (f"{prefix}.net.0.bias", b0),
        (f"{prefix}.net.2.weight", w1),
        (f"{prefix}.net.2.bias", b1),
    ]


def pack_slang_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    h: int,
    p: int,
    attn_blocks: int = ATTN_BLOCKS,
) -> tuple[bytes, dict]:
    """
    Pack a LearnedImpactNet state_dict into the flat float layout the Slang kernels read.

    Layout (floats, little-endian):
      node_enc MLP, edge_enc MLP,
      for each of 4 MP layers: edge MLP, node MLP,
      for each attn block: assign_w[P,H], assign_b[P], tau_w[H], tau_b,
                           qkv_w (3×[H,H]), qkv_b[3H], out_w, out_b, ffn_up, ffn_down,
      bond_fail [FAIL_BINS,H]+b, bond_util [1,H]+b

    LayerNorm affine and chunk_delta heads are training-only (not in the Slang blob).
    Attention temperature (tau) is initialized so softplus(tau_logit)≈0.5 → tau≈1.
    """
    sd = state_dict
    blob = bytearray()
    order: list[dict] = []
    expected = slang_weight_count(h, p, attn_blocks=attn_blocks)

    for name, t in _mlp_pack(sd, "encode_n"):
        _append(blob, order, name, t)
    for name, t in _mlp_pack(sd, "encode_e"):
        _append(blob, order, name, t)

    for li in range(MP_LAYERS):
        for name, t in _mlp_pack(sd, f"mp.{li}.edge"):
            _append(blob, order, name, t)
        for name, t in _mlp_pack(sd, f"mp.{li}.node"):
            _append(blob, order, name, t)

    # tau: prefer unit temperature (softplus(x)=0.5 ⇒ x = log(e^0.5 - 1))
    tau_b = torch.tensor([float(torch.log(torch.exp(torch.tensor(0.5)) - 1.0))])
    tau_w = torch.zeros(h)

    for bi in range(attn_blocks):
        # Train model has a single PhysicsAttention; repeat its weights if attn_blocks>1.
        aw, ab = _linear_w_b(sd, "attn.assign")
        if aw.shape != (p, h):
            raise ValueError(f"attn.assign.weight expected [{p},{h}], got {tuple(aw.shape)}")
        _append(blob, order, f"attn.{bi}.assign.weight", aw)
        _append(blob, order, f"attn.{bi}.assign.bias", ab)
        _append(blob, order, f"attn.{bi}.tau.weight", tau_w)
        _append(blob, order, f"attn.{bi}.tau.bias", tau_b)

        qw, qb = _linear_w_b(sd, "attn.qkv")
        # PyTorch [3H, H] → three [H, H] blocks already contiguous
        _append(blob, order, f"attn.{bi}.qkv.weight", qw.reshape(3, h, h))
        _append(blob, order, f"attn.{bi}.qkv.bias", qb.reshape(3, h))

        ow, ob = _linear_w_b(sd, "attn.out")
        _append(blob, order, f"attn.{bi}.out.weight", ow)
        _append(blob, order, f"attn.{bi}.out.bias", ob)

        # ffn: Sequential Linear(h,4h), SiLU, Linear(4h,h)
        fw0, fb0 = _linear_w_b(sd, "attn.ffn.0")
        fw1, fb1 = _linear_w_b(sd, "attn.ffn.2")
        _append(blob, order, f"attn.{bi}.ffn.0.weight", fw0)
        _append(blob, order, f"attn.{bi}.ffn.0.bias", fb0)
        _append(blob, order, f"attn.{bi}.ffn.2.weight", fw1)
        _append(blob, order, f"attn.{bi}.ffn.2.bias", fb1)

    fw, fb = _linear_w_b(sd, "bond_fail")
    uw, ub = _linear_w_b(sd, "bond_util")
    if fw.shape != (FAIL_BINS, h):
        raise ValueError(f"bond_fail.weight expected [{FAIL_BINS},{h}], got {tuple(fw.shape)}")
    _append(blob, order, "bond_fail.weight", fw)
    _append(blob, order, "bond_fail.bias", fb)
    _append(blob, order, "bond_util.weight", uw)
    _append(blob, order, "bond_util.bias", ub)

    n_floats = len(blob) // 4
    if n_floats != expected:
        raise RuntimeError(
            f"slang pack size mismatch: packed {n_floats} floats, expected {expected} "
            f"(h={h} p={p} attn_blocks={attn_blocks})"
        )

    cfg = {
        "format": SLANG_FORMAT,
        "h": h,
        "p": p,
        "node_in": NODE_IN,
        "edge_in": EDGE_IN,
        "ffn_mult": FFN_MULT,
        "mp_layers": MP_LAYERS,
        "attn_blocks": attn_blocks,
        "fail_bins": FAIL_BINS,
        "n_floats": n_floats,
        "n_bytes": len(blob),
        "tensors": order,
        "notes": (
            "Flat f32 LE blob for learned_impact.slang. "
            "LayerNorm affine / chunk_delta omitted. "
            "Native Metal loads the same weights.bin via a storage buffer bind."
        ),
    }
    return bytes(blob), cfg


def export_slang_weights(
    model: torch.nn.Module | dict[str, torch.Tensor],
    out_dir: Path | str,
    *,
    h: int,
    p: int,
    attn_blocks: int = ATTN_BLOCKS,
) -> Path:
    """Write weights.bin + weights.json in learned-impact-slang/1 layout."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sd = model if isinstance(model, dict) else model.state_dict()
    blob, cfg = pack_slang_state_dict(sd, h=h, p=p, attn_blocks=attn_blocks)
    bin_path = out / "weights.bin"
    bin_path.write_bytes(blob)
    (out / "weights.json").write_text(json.dumps(cfg, indent=2))
    return bin_path


def export_weights(model: torch.nn.Module, out_dir: Path | str, *, h: int, p: int) -> Path:
    """Export state_dict.pt plus raw PyTorch blob; Slang-pack when the net is MGN-T."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()
    torch.save(sd, out / "state_dict.pt")
    # Raw concatenation (training debug / Python reload) — not the Slang layout.
    blob = bytearray()
    order: list[dict] = []
    for name, tensor in sd.items():
        _append(blob, order, name, tensor)
    (out / "weights_pytorch.bin").write_bytes(blob)
    raw_cfg = {
        "format": "learned-impact-weights/1",
        "h": h,
        "p": p,
        "n_bytes": len(blob),
        "tensors": order,
        "notes": "Raw state_dict concatenation. Use weights.bin (slang/1) for GPU inference.",
    }
    (out / "weights_pytorch.json").write_text(json.dumps(raw_cfg, indent=2))
    # Slang pack needs encode/mp/attn/bond heads (LearnedImpactNet). MLP baseline skips.
    if "encode_n.net.0.weight" in sd and "attn.assign.weight" in sd:
        return export_slang_weights(sd, out, h=h, p=p)
    # Keep a copy named weights.bin for older smoke checks (pytorch layout).
    (out / "weights.bin").write_bytes(blob)
    (out / "weights.json").write_text(json.dumps({**raw_cfg, "notes": "MLP baseline — not Slang-packed."}, indent=2))
    return out / "weights.bin"
