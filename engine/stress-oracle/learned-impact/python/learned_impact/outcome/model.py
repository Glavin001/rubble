"""Impact-outcome GNN, model specification v1 (plan, "Model specification v1").

Encode (per node type and per edge type, the HybridMGN pattern) → U-processor (typed
MeshGraphNets rounds at the fine level; BSMS bi-stride levels on the masked bond graph;
a GraphNet global node throughout) → heads (bond demand / break / time / Δdamage, per-node
Δv Δω Δx Δθ in body frames, P(perforate), energy).

Every head output is nondimensional; `decode.py` turns it into physical quantities.
Variant A predicts the outcome; variant B adds the `blast_parity` QS frame outcome as an
input and predicts a residual over it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..mgn.bsms import GraphMessagePassing, Unpool, WeightedEdgeConv
from ..mgn.mesh_blocks import GlobalBlock, MeshEdgeBlock, TypedNodeBlock
from ..mgn.mesh_graph_mlp import MeshGraphMLP
from . import schema as S
from .batch import SCALE_INDEX, Batch
from .geometry import quat_rotate_inv

NODE_OUT = 12  # Δv(3) Δω(3) Δx(3) Δθ(3), body frame, nondimensional


@dataclass
class ModelConfig:
    # Accuracy first (planning session, 2026-10-22): the default is the reference W64 with
    # (3, 2, 2, 2, 3) = 12 rounds; `large()` is W128 / 16 rounds; `cost()` is P0's
    # cost-feasible W32 / 8 rounds, to shrink to only after a model passes.
    width: int = 64
    mlp_hidden_layers: int = 1  # MeshGraphMLP hidden layers: 1 → two Linear layers
    # Rounds per U stage: [fine, L1 down, …, bottom, …, L1 up, fine]; len = 2·levels + 1.
    rounds: tuple[int, ...] = (3, 2, 2, 2, 3)
    levels: int = 2
    variant: str = "B"  # "B" residual over the QS frame outcome (default since the P0 step-5 pilot), "A" impact-only
    n_time_bins: int = S.N_TIME_BINS
    use_global: bool = True
    # Fixed per-column standardisation of every input array, from training-set statistics
    # stored in the checkpoint (a global affine map, not per-graph normalisation). The port
    # applies it as a separate (x − mean) / std step: folding it into the first Linear is
    # lossy in f32 (P0 measured 4.4e-3 on bond columns with σ ≈ 8.5e-6).
    input_norm: bool = True
    edge_types: tuple[str, ...] = S.EDGE_TYPES
    # Input columns held at 0 after standardisation ("<array>:<column>", e.g.
    # "bond_feat:log_qs_hertz_util"): an input ablation that training, evaluation, the
    # provider and the export all apply the same way, since it lives in the config.
    drop_features: tuple[str, ...] = ()
    # The break margin as frame-start margin × a non-negative increment (see `_heads`);
    # False only for checkpoints that predate it.
    monotone_margin: bool = True
    # `/5`: the oracle's full dynamic bond and chunk state as inputs and heads (fullstate.py)
    full_state: bool = False

    def __post_init__(self):
        self.rounds = tuple(self.rounds)
        self.drop_features = tuple(self.drop_features)
        for f in self.drop_features:
            arr, _, col = f.partition(":")
            if arr not in S.FEATURE_COLUMNS or col not in S.FEATURE_COLUMNS[arr]:
                raise ValueError(f"drop_features {f!r}: no such input column")
            if arr.endswith("_qs_feat") and self.variant != "B":
                raise ValueError(f"drop_features {f!r}: variant A has no QS-frame inputs")
        if len(self.rounds) != 2 * self.levels + 1:
            raise ValueError(f"rounds {self.rounds} need {2 * self.levels + 1} stages")
        if self.variant not in ("A", "B"):
            raise ValueError(f"variant {self.variant!r}")

    @staticmethod
    def reference() -> "ModelConfig":
        return ModelConfig(width=64, rounds=(3, 2, 2, 2, 3), levels=2)

    @staticmethod
    def large() -> "ModelConfig":
        return ModelConfig(width=128, rounds=(4, 3, 2, 3, 4), levels=2)

    @staticmethod
    def cost() -> "ModelConfig":
        return ModelConfig(width=32, rounds=(2, 1, 2, 1, 2), levels=2)

    def to_dict(self) -> dict:
        return asdict(self)


def _mlp(i: int, o: int, w: int, hl: int, norm: bool = True) -> MeshGraphMLP:
    return MeshGraphMLP(i, o, w, hl, nn.SiLU(), "LayerNorm" if norm else None)


class FineRound(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        w, hl = cfg.width, cfg.mlp_hidden_layers
        self.types = cfg.edge_types
        self.edge = nn.ModuleDict({t: MeshEdgeBlock(w, w, w, w, hl, nn.SiLU())
                                   for t in self.types})
        self.node = TypedNodeBlock(len(self.types), w, hl, nn.SiLU())
        self.glob = GlobalBlock(w, hl, nn.SiLU()) if cfg.use_global else None

    def forward(self, h: Tensor, e: dict[str, Tensor], g: Tensor, b: Batch) -> tuple:
        for t in self.types:
            e[t], _ = self.edge[t](e[t], h, b.edges[t]["src"], b.edges[t]["dst"])
        h = self.node([e[t] for t in self.types], [b.edges[t]["dst"] for t in self.types], h,
                      g[b.t["node_call"]])
        if self.glob is not None:
            g = self.glob(g, h, b.t["node_call"])
        return h, e, g


class CoarseRound(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gmp = GraphMessagePassing(cfg.width, cfg.mlp_hidden_layers, pos_dim=3,
                                       extra_dim=cfg.width)
        self.glob = GlobalBlock(cfg.width, cfg.mlp_hidden_layers) if cfg.use_global else None

    def forward(self, h: Tensor, edges: Tensor, fiber: Tensor, g: Tensor, call: Tensor):
        h = self.gmp.forward_fiber(h, edges, fiber, g[call])
        if self.glob is not None:
            g = self.glob(g, h, call)
        return h, g


class ImpactOutcomeModel(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        w, hl = cfg.width, cfg.mlp_hidden_layers
        b = cfg.variant == "B"
        from . import fullstate as FS
        nfc = len(S.CHUNK_FEAT_COLUMNS) + (len(S.CHUNK_QS_FEAT_COLUMNS) if b else 0) \
            + (FS.N_CHUNK_IN if cfg.full_state else 0)
        nfi = len(S.IMPACTOR_FEAT_COLUMNS) + (len(S.IMPACTOR_QS_FEAT_COLUMNS) if b else 0)
        self.enc_chunk = _mlp(nfc, w, w, hl)
        self.enc_imp = _mlp(nfi, w, w, hl)
        self.enc_glob = _mlp(len(S.GLOBAL_FEAT_COLUMNS), w, w, hl)
        edims = {t: len(S.FEATURE_COLUMNS[f"{t}_feat"]) for t in cfg.edge_types}
        if b:
            edims["bond"] += len(S.BOND_QS_FEAT_COLUMNS)
        if cfg.full_state:
            edims["bond"] += FS.N_BOND_IN
        self.enc_edge = nn.ModuleDict({t: _mlp(edims[t], w, w, hl) for t in cfg.edge_types})
        r = cfg.rounds
        self.fine_pre = nn.ModuleList(FineRound(cfg) for _ in range(r[0]))
        self.fine_post = nn.ModuleList(FineRound(cfg) for _ in range(r[-1]))
        self.down = nn.ModuleList(
            nn.ModuleList(CoarseRound(cfg) for _ in range(r[1 + i])) for i in range(cfg.levels))
        self.up = nn.ModuleList(
            nn.ModuleList(CoarseRound(cfg) for _ in range(r[cfg.levels + 1 + i]))
            for i in range(cfg.levels - 1))
        self.edge_conv = WeightedEdgeConv()
        dims = {"chunk": nfc, "imp": nfi, "glob": len(S.GLOBAL_FEAT_COLUMNS),
                **{f"edge_{t}": d for t, d in edims.items()}}
        for k, d in dims.items():  # identity until set_input_stats (kept in the state dict)
            self.register_buffer(f"in_mean_{k}", torch.zeros(d))
            self.register_buffer(f"in_std_{k}", torch.ones(d))
        # input key and column offset of every feature array (variant B appends the QS frame)
        where = {"chunk_feat": ("chunk", 0), "impactor_feat": ("imp", 0), "global_feat": ("glob", 0),
                 "chunk_qs_feat": ("chunk", len(S.CHUNK_FEAT_COLUMNS)),
                 "impactor_qs_feat": ("imp", len(S.FEATURE_COLUMNS["impactor_feat"])),
                 "bond_qs_feat": ("edge_bond", len(S.FEATURE_COLUMNS["bond_feat"])),
                 **{f"{t}_feat": (f"edge_{t}", 0) for t in cfg.edge_types}}
        self._keep: dict[str, str] = {}
        for f in cfg.drop_features:
            arr, _, col = f.partition(":")
            k, off = where[arr]
            name = f"in_keep_{k}"
            if k not in self._keep:
                self.register_buffer(name, torch.ones(dims[k]), persistent=False)
                self._keep[k] = name
            getattr(self, name)[off + S.FEATURE_COLUMNS[arr].index(col)] = 0.0
        self.unpool = Unpool()
        nt = cfg.n_time_bins
        fs_b = FS.N_BOND_OUT if cfg.full_state else 0
        fs_c = FS.N_CHUNK_OUT if cfg.full_state else 0
        self.bond_head = _mlp(4 * w + w, 4 + nt + fs_b, w, hl, norm=False)
        self.chunk_head = _mlp(2 * w, NODE_OUT + fs_c, w, hl, norm=False)
        self.imp_head = _mlp(2 * w, NODE_OUT + 1, w, hl, norm=False)
        self.energy_head = _mlp(w, 4, w, hl, norm=False)
        # Start every head at (almost) the zero predictor, so each normalised loss term
        # starts near 1: small final weights, and softplus channels biased toward 0.
        for head in (self.bond_head, self.chunk_head, self.imp_head, self.energy_head):
            last = head.model[-1]
            nn.init.normal_(last.weight, std=1e-3)
            nn.init.zeros_(last.bias)
        with torch.no_grad():
            self.bond_head.model[-1].bias[2] = -6.0  # Δdamage
            self.energy_head.model[-1].bias[0] = -6.0  # D_bond / E_in
            self.energy_head.model[-1].bias[2] = -6.0  # E_elastic / E_in

    def _norm(self, key: str, x: Tensor) -> Tensor:
        if self.cfg.input_norm:
            x = (x - getattr(self, f"in_mean_{key}")) / getattr(self, f"in_std_{key}")
        if key in self._keep:
            x = x * getattr(self, self._keep[key])
        return x

    @torch.no_grad()
    def set_input_stats(self, stats: dict[str, tuple[Tensor, Tensor]]) -> None:
        """`stats[key] = (mean, std)`; a column with (near-)zero spread keeps std 1."""
        for k, (m, sd) in stats.items():
            getattr(self, f"in_mean_{k}").copy_(m)
            getattr(self, f"in_std_{k}").copy_(torch.where(sd > 1e-6, sd, torch.ones_like(sd)))

    # ------------------------------------------------------------------------------
    def _fiber(self, pos: Tensor, chunk_idx: Tensor, edges: Tensor, b: Batch) -> Tensor:
        q = b.t["chunk_q_world"][chunk_idx]
        call = b.t["chunk_call"][chunk_idx]
        lscale = b.t["scales"][call, SCALE_INDEX["L"]].to(pos.dtype).unsqueeze(-1)
        i, j = edges[0], edges[1]
        d = quat_rotate_inv(q[i], pos[i] - pos[j]) / lscale[i]
        return torch.cat([d, torch.linalg.norm(d, dim=-1, keepdim=True)], dim=-1)

    def forward(self, b: Batch) -> dict[str, Tensor]:
        cfg = self.cfg
        t = b.t
        cx, ix = t["chunk_x"], t["imp_x"]
        if cfg.variant == "B":
            cx = torch.cat([cx, t["chunk_qs"]], -1)
            ix = torch.cat([ix, t["imp_qs"]], -1)
        if cfg.full_state:
            cx = torch.cat([cx, t["chunk_fs_in"]], -1)
        cx, ix = self._norm("chunk", cx), self._norm("imp", ix)
        n = int(t["n_nodes"])
        h = cx.new_zeros((n, cfg.width))
        h = h.index_copy(0, t["chunk_node"], self.enc_chunk(cx))
        h = h.index_copy(0, t["imp_node"], self.enc_imp(ix))
        g = self.enc_glob(self._norm("glob", t["glob_x"]))
        e = {}
        for et in cfg.edge_types:
            x = b.edges[et]["x"]
            if et == "bond" and cfg.variant == "B":
                x = torch.cat([x, t["bond_qs"]], -1)
            if et == "bond" and cfg.full_state:
                x = torch.cat([x, b.edges["bond"]["fs"]], -1)
            e[et] = self.enc_edge[et](self._norm(f"edge_{et}", x))

        for rnd in self.fine_pre:
            h, e, g = rnd(h, e, g, b)

        if cfg.levels > 0:
            chunk_node = t["chunk_node"]
            skip0 = h
            hl = h[chunk_node]
            pos = t["chunk_pos"]
            w = pos.new_ones((pos.shape[0], 1))
            skips, ews, poss, calls = [], [], [], []
            for li in range(cfg.levels):
                gl = b.h_edges[li]
                ew, w = self.edge_conv.cal_ew(w, gl)
                ews.append(ew)
                if li > 0:
                    skips.append(hl)
                poss.append(pos)
                hl = self.edge_conv(hl, gl, ew)[b.h_ids[li + 1]]
                with torch.no_grad():
                    pos = self.edge_conv(pos, gl, ew)[b.h_ids[li + 1]]
                w = w[b.h_ids[li + 1]]
                ch = b.h_chunk[li + 1]
                call = t["chunk_call"][ch]
                calls.append(call)
                fib = self._fiber(pos, ch, b.h_edges[li + 1], b)
                for rnd in self.down[li]:
                    hl, g = rnd(hl, b.h_edges[li + 1], fib, g, call)
            for k, li in enumerate(range(cfg.levels - 1, -1, -1)):
                gl = b.h_edges[li]
                n_prev = poss[li].shape[0]
                hl = self.unpool(hl, n_prev, b.h_ids[li + 1])
                hl = self.edge_conv(hl, gl, ews[li], aggragating=False)
                if li > 0:  # BSMS order at coarse levels: GMP, then the skip
                    ch = b.h_chunk[li]
                    fib = self._fiber(poss[li], ch, gl, b)
                    for rnd in self.up[k]:
                        hl, g = rnd(hl, gl, fib, g, calls[li - 1])
                    hl = hl + skips[li - 1]
            # fine level: skip first so impactors (not in the hierarchy) keep their state
            h = skip0.index_add(0, chunk_node, hl)

        for rnd in self.fine_post:
            h, e, g = rnd(h, e, g, b)

        return self._heads(h, e, g, b)

    def _heads(self, h: Tensor, e: dict[str, Tensor], g: Tensor, b: Batch) -> dict[str, Tensor]:
        t = b.t
        nt = self.cfg.n_time_bins
        be = b.edges["bond"]
        ef, er = e["bond"][0::2], e["bond"][1::2]
        hs, hd = h[be["src"][0::2]], h[be["dst"][0::2]]
        gb = g[be["call"][0::2]]
        out = 0.5 * (self.bond_head(torch.cat([ef, er, hs, hd, gb], -1))
                     + self.bond_head(torch.cat([er, ef, hd, hs, gb], -1)))
        log_r = out[:, 0]
        if self.cfg.variant == "B":
            log_r = log_r + t["bond_qs"][0::2, 0]
        if self.cfg.monotone_margin:
            # q = κ / κ_fail with κ a running maximum: a surviving bond's frame peak is never
            # below its frame-start margin (bulk v1: 0 of 8,987), and in quiet frames it stays
            # put. So q̂ = q_start · e^softplus(·): "unchanged" is representable to any
            # precision, where a free regression of log q sat ±5% around near-critical damaged
            # bonds and crossed τ = 1 (the real-data overfit's false breaks).
            q0 = be["x"][0::2, S.col("bond_feat", "fail_margin")].clamp(min=1e-4)
            log_q = torch.log(q0) + F.softplus(out[:, 3] - 2.0)
        else:
            log_q = out[:, 3]
        chunk_h = h[t["chunk_node"]]
        imp_h = h[t["imp_node"]]
        c_out = self.chunk_head(torch.cat([chunk_h, g[t["chunk_call"]]], -1))
        i_raw = self.imp_head(torch.cat([imp_h, g[t["imp_call"]]], -1))
        i_out = i_raw[:, :NODE_OUT]
        if self.cfg.variant == "B":
            pad_c = F.pad(t["chunk_qs"], (0, c_out.shape[-1] - 6))
            pad_i = F.pad(t["imp_qs"], (0, NODE_OUT - 6))
            c_out = c_out + pad_c
            i_out = i_out + pad_i
        en = self.energy_head(g)
        extra = {}
        if self.cfg.full_state:
            from . import fullstate as FS
            nf = len(FS.BOND_FIELDS)
            raw = out[:, 4 + nt:4 + nt + nf]
            mono = torch.as_tensor(FS.MONO, device=raw.device)
            start = t["y_bond_fs_start"].to(raw.dtype)
            extra["bond_fs"] = start + torch.where(mono, F.softplus(raw - 2.0), raw)
            extra["bond_fs_mode_logits"] = out[:, 4 + nt + nf:4 + nt + nf + 3]
            extra["chunk_fs"] = t["chunk_fs_in"].to(c_out.dtype) + c_out[:, NODE_OUT:]
        return {**extra, 
            "bond_log_r": log_r,
            "bond_break_logit": out[:, 1],
            "bond_log_q": log_q,  # failure margin: the break decision (q ≥ 1)
            "bond_time_logits": out[:, 4:4 + nt],
            "bond_ddamage": F.softplus(out[:, 2]),
            "chunk_out": c_out[:, :NODE_OUT],
            "imp_out": i_out,
            "imp_perf_logit": i_raw[:, NODE_OUT],
            # ÷ E_in: D_bond ≥ 0, its joint (vs dashpot) fraction, E_elastic_end ≥ 0, KE_loss (signed)
            "energy_d_bond": F.softplus(en[:, 0]),
            "energy_frac_joint": torch.sigmoid(en[:, 1]),
            "energy_e_elastic": F.softplus(en[:, 2]),
            "energy_ke_loss": en[:, 3],
            "latent_global": g,
        }


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


