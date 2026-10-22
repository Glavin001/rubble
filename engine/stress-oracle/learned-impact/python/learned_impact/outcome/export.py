"""Export a trained checkpoint for the Slang port (P3): `learned-impact-weights/1`.

    python -m learned_impact.outcome.export --checkpoint ../runs/<job>/best.pt \
        --out ../runs/<job>/export [--dataset <calls> --parity 4]

Writes, under `--out`:
- `weights.bin`: every tensor of the model's state dict (parameters and the input
  standardisation buffers), little-endian f32, row-major, each 16-byte aligned;
- `weights.json`: the manifest. It holds `schema` / `calls_schema`, `model_config` (width,
  rounds, levels, variant, …), `tensors` (name → shape, byte offset, element count; names are
  PyTorch's), `decode` (τ on the failure margin, p_b temperature, energy tolerance, prior
  correction or none), `feature_columns` (the calls schema's input orders), and `layers`, a
  description of every block's computation, so a port need not read the PyTorch;
- `parity/<k>/`: real validation calls (the single-call `learned-impact-outcome/4` record)
  with `outputs.npz` (every head output, f32, CPU) and `decoded.json` (the deployed decoded
  state). A port that reproduces these within fp32 tolerance, with identical break sets,
  passes the plan's parity criterion on these calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import decode as D
from . import schema as S
from . import state_io as IO
from .batch import HierarchyCache, collate
from .train import Data, decode_config, load_model

VERSION = "learned-impact-weights/1"

LAYERS = {
    "MeshGraphMLP": "Linear(in→hidden) · SiLU · [Linear(hidden→hidden) · SiLU]×(hidden_layers−1) · "
                    "Linear(hidden→out) · LayerNorm(out, eps=1e-5) (no LayerNorm on heads: "
                    "`*_head`); state names `model.<i>.weight|bias` in Sequential order",
    "input": "each input array x (chunk: chunk_feat ‖ chunk_qs_feat for variant B; imp: "
             "impactor_feat ‖ impactor_qs_feat; glob: global_feat; edge_<type>: <type>_feat, "
             "bond ‖ bond_qs_feat for B) → (x − in_mean_<k>) / in_std_<k> when input_norm",
    "encode": "h[chunk] = enc_chunk(x), h[impactor] = enc_imp(x), g = enc_glob(x), "
              "e[type] = enc_edge.<type>(x)",
    "FineRound": "for each edge type: e' = e + edge.<type>.edge_mlp([e, h[src], h[dst]]); "
                 "h' = h + node.node_mlp([Σ_dst e'_bond, Σ e'_cracked, Σ e'_impactor_contact, "
                 "Σ e'_structure_contact, h, g[call]]); g' = g + glob.mlp([g, mean_call h'])",
    "U": "fine_pre rounds; for each level l: pool (BSMS WeightedEdgeConv weights on level l's "
         "live edges, then select h_ids[l+1]), down[l] CoarseRound×r; up: unpool, "
         "edge_conv(aggregating=False), up rounds, + skip; fine level: h = skip0 + unpooled; "
         "fine_post rounds (see mgn/bsms.py and outcome/model.py)",
    "CoarseRound": "m = gmp.mlp_edge([fibre, h[i], h[j]]), fibre = (R_bodyᵀ(p_i − p_j)/L, |·|); "
                   "h' = h + gmp.mlp_node([h, Σ_j m, g[call]]); g' = g + glob.mlp([g, mean h'])",
    "heads": "bond: ½(bond_head([e_fwd, e_rev, h_s, h_d, g]) + bond_head([e_rev, e_fwd, h_d, h_s, g])) "
             "→ [log r, break logit, softplus→Δdamage, log q, time logits×8] (variant B adds "
             "bond_qs col 0 to log r); chunk/imp: *_head([h, g]) → 12 body-frame outputs (+ "
             "perforation logit for impactors; B adds *_qs to the first 6); energy_head(g)",
}


def export(ckpt: str | Path, out: str | Path, dataset: str | None = None, parity: int = 4) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model, ck = load_model(ckpt, "cpu")
    model.eval()
    sd = model.state_dict()
    blob, tensors, off = [], [], 0
    for name, t in sd.items():
        a = t.detach().cpu().contiguous().numpy().astype("<f4")
        raw = a.tobytes()
        pad = (-len(raw)) % 16
        tensors.append({"name": name, "shape": list(a.shape), "offset": off, "count": int(a.size)})
        blob.append(raw + b"\0" * pad)
        off += len(raw) + pad
    (out / "weights.bin").write_bytes(b"".join(blob))
    dcfg = decode_config(ck)
    manifest = {
        "schema": VERSION, "calls_schema": S.SCHEMA_VERSION, "checkpoint": str(ckpt),
        "model_config": ck["model_config"],
        "decode": {"tau": dcfg.tau, "temperature": dcfg.temperature,
                   "energy_tolerance": dcfg.energy_tolerance, "break_mode": dcfg.break_mode,
                   "sample_seed": dcfg.sample_seed,
                   "break_rule": ("log q ≥ log tau" if dcfg.break_mode == "threshold" else
                                  "hash_uniform(bond_key, sample_seed) < sigmoid(break_logit / "
                                  "temperature) (decode.hash_uniform)"),
                   "prior_correction": ck.get("priors")},
        "feature_columns": {k: list(v) for k, v in S.FEATURE_COLUMNS.items()},
        "tensors": tensors, "layers": LAYERS,
    }
    (out / "weights.json").write_text(json.dumps(manifest, indent=1))
    if dataset and parity:
        data = Data(dataset)
        calls = [c for c in data.val if c.phase in ("contact", "secondary")][:parity]
        cache = HierarchyCache(model.cfg.levels)
        for k, c in enumerate(calls):
            d = out / "parity" / f"{k}"
            ev = next(e for e in data.events if e.manifest["event_id"] == c.event_id)
            S.write_event(d / "call", {**ev.manifest, "event_id": c.event_id}, [c], ev.structures)
            b = collate([c], data.structures, cache)
            with torch.no_grad():
                o = model(b)
            np.savez(d / "outputs.npz", **{k2: v.numpy().astype(np.float32) for k2, v in o.items()})
            dec = D.decode(o, b, dcfg)
            IO.write(d / "decoded.json", IO.decoded_states(b, o, dec, {"checkpoint": str(ckpt)})[0])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=None, help="calls dataset for parity fixtures")
    ap.add_argument("--parity", type=int, default=4)
    a = ap.parse_args(argv)
    print(export(a.checkpoint, a.out, a.dataset, a.parity))


if __name__ == "__main__":
    main()
