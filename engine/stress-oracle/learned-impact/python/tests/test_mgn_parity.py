"""The blocks copied into learned_impact/mgn reproduce PhysicsNeMo @ c8f68428 for fixed
weights. The reference outputs are a committed fixture (scripts/gen_physicsnemo_fixture.py,
run once); nothing here imports inspiration/."""

from pathlib import Path

import numpy as np
import pytest
import torch

from learned_impact.mgn.bsms import BistrideGraphMessagePassing
from learned_impact.mgn.bsms_graph import build_hierarchy
from learned_impact.mgn.mesh_blocks import MeshEdgeBlock, MeshNodeBlock
from learned_impact.mgn.mesh_graph_mlp import MeshGraphMLP

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "physicsnemo_parity.npz"


@pytest.fixture(scope="module")
def fx():
    return np.load(FIX)


def _load(module: torch.nn.Module, fx, prefix: str) -> None:
    sd = {k[len(prefix) + 7:]: torch.from_numpy(fx[k]) for k in fx.files
          if k.startswith(prefix + "/state/")}
    missing, unexpected = module.load_state_dict(sd, strict=True), None
    assert unexpected is None and not missing.missing_keys


def test_fixture_commit(fx):
    assert str(fx["commit"]) == "c8f68428840a559f4b86ef56e96000b80a8d9594"


@pytest.mark.parametrize("hl", [1, 2])
def test_mesh_graph_mlp(fx, hl):
    m = MeshGraphMLP(10, 16, 24, hl, torch.nn.SiLU(), "LayerNorm")
    _load(m, fx, f"mlp{hl}")
    y = m(torch.from_numpy(fx[f"mlp{hl}/x"]))
    np.testing.assert_allclose(y.detach().numpy(), fx[f"mlp{hl}/y"], rtol=1e-5, atol=1e-6)


def test_edge_and_node_blocks(fx):
    eb = MeshEdgeBlock(16, 16, 16, 16, 1, torch.nn.SiLU())
    nb = MeshNodeBlock("sum", 16, 16, 16, 16, 1, torch.nn.SiLU())
    _load(eb, fx, "edge")
    _load(nb, fx, "node")
    ei = torch.from_numpy(fx["graph/edge_index"]).long()
    e_new, _ = eb(torch.from_numpy(fx["edge/efeat"]), torch.from_numpy(fx["edge/nfeat"]),
                  ei[0], ei[1])
    np.testing.assert_allclose(e_new.detach().numpy(), fx["edge/y"], rtol=1e-5, atol=1e-5)
    _, n_new = nb(e_new, torch.from_numpy(fx["edge/nfeat"]), ei[1])
    np.testing.assert_allclose(n_new.detach().numpy(), fx["node/y"], rtol=1e-5, atol=1e-5)


def test_bsms_hierarchy_matches(fx):
    pos = fx["graph/pos"]
    ei = fx["graph/edge_index"]
    half = ei.shape[1] // 2
    h = build_hierarchy(len(pos), ei[:, :half].T, pos, 2, tie_tol=0.0)
    for li, lv in enumerate(h.levels):
        np.testing.assert_array_equal(lv.ids, fx[f"bsms/ids{li}"])
        ours = {tuple(e) for e in lv.edges.T.tolist()}
        theirs = {tuple(e) for e in fx[f"bsms/gs{li + 1}"].T.tolist()}
        assert ours == theirs, f"level {li + 1} edges differ"


def test_bsms_u_processor(fx):
    up = BistrideGraphMessagePassing(unet_depth=2, latent_dim=16, hidden_layer=1, pos_dim=3)
    _load(up, fx, "bsms")
    gs = [torch.from_numpy(fx[f"bsms/gs{i}"]).long() for i in range(3)]
    ids = [torch.from_numpy(fx[f"bsms/ids{i}"]).long() for i in range(2)]
    y = up(torch.from_numpy(fx["bsms/h"]), ids, gs, torch.from_numpy(fx["graph/pos"]))
    np.testing.assert_allclose(y.detach().numpy(), fx["bsms/y"], rtol=1e-4, atol=1e-5)
