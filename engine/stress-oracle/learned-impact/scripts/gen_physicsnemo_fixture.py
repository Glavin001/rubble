"""Generate `fixtures/physicsnemo_parity.npz` ONCE from an upstream PhysicsNeMo checkout.

The committed fixture lets `tests/test_mgn_parity.py` prove that the blocks copied into
`learned_impact/mgn/` reproduce PhysicsNeMo's outputs for fixed weights, without the tests
ever importing `inspiration/` (AGENTS.md): only this generator does, and it is not run in
CI. Run it in a throwaway venv with `torch warp-lang torch_geometric jaxtyping packaging
scipy` installed:

    python scripts/gen_physicsnemo_fixture.py --physicsnemo <path>/physicsnemo \
        --out fixtures/physicsnemo_parity.npz

PhysicsNeMo needs `torch_scatter` for its PyG path; no wheel exists for this torch, so the
generator installs a stub package whose `scatter(..., reduce="sum")` is `scatter_add_`
(the same sum; the stub is for the import only). The upstream commit is recorded in the
fixture and must be c8f68428840a559f4b86ef56e96000b80a8d9594.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

STUB = '''
import torch
__version__ = "2.1.2"
def scatter(src, index, dim=0, out=None, dim_size=None, reduce="sum"):
    if reduce != "sum":
        raise NotImplementedError(reduce)
    if dim < 0:
        dim = src.dim() + dim
    size = list(src.shape)
    size[dim] = dim_size if dim_size is not None else int(index.max()) + 1
    idx = index
    while idx.dim() < src.dim():
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)
    return torch.zeros(size, dtype=src.dtype, device=src.device).scatter_add_(dim, idx, src)
'''


def install_stub() -> str:
    d = Path(tempfile.mkdtemp())
    (d / "torch_scatter").mkdir()
    (d / "torch_scatter" / "__init__.py").write_text(STUB)
    di = d / "torch_scatter-2.1.2.dist-info"
    di.mkdir()
    (di / "METADATA").write_text("Metadata-Version: 2.1\nName: torch-scatter\nVersion: 2.1.2\n")
    (di / "RECORD").write_text("")
    return str(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--physicsnemo", required=True, help="path to the PhysicsNeMo repo root")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.physicsnemo).resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                     text=True).strip()
    sys.path.insert(0, install_stub())
    sys.path.insert(0, str(root))
    import torch
    from torch_geometric.data import Data

    from physicsnemo.datapipes.gnn.bsms import BistrideMultiLayerGraph
    from physicsnemo.nn.module.gnn_layers.bsms import BistrideGraphMessagePassing
    from physicsnemo.nn.module.gnn_layers.mesh_edge_block import MeshEdgeBlock
    from physicsnemo.nn.module.gnn_layers.mesh_graph_mlp import MeshGraphMLP
    from physicsnemo.nn.module.gnn_layers.mesh_node_block import MeshNodeBlock

    torch.manual_seed(1234)
    out: dict[str, np.ndarray] = {"commit": np.array(commit)}

    def put(prefix: str, module: torch.nn.Module) -> None:
        for k, v in module.state_dict().items():
            out[f"{prefix}/state/{k}"] = v.detach().numpy()

    # 1. MeshGraphMLP (hidden_layers 1 and 2)
    for hl in (1, 2):
        m = MeshGraphMLP(10, 16, 24, hl, torch.nn.SiLU(), "LayerNorm")
        x = torch.randn(7, 10)
        put(f"mlp{hl}", m)
        out[f"mlp{hl}/x"] = x.numpy()
        out[f"mlp{hl}/y"] = m(x).detach().numpy()

    # a small 3-D grid graph (both directions), as a bond graph
    nx, ny, nz = 5, 2, 4
    pos = np.array([[i, j, k] for k in range(nz) for j in range(ny) for i in range(nx)],
                   dtype=np.float32) * 0.1
    idx = lambda i, j, k: (k * ny + j) * nx + i  # noqa: E731
    und = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if i + 1 < nx:
                    und.append((idx(i, j, k), idx(i + 1, j, k)))
                if j + 1 < ny:
                    und.append((idx(i, j, k), idx(i, j + 1, k)))
                if k + 1 < nz:
                    und.append((idx(i, j, k), idx(i, j, k + 1)))
    und = np.array(und)
    ei = np.concatenate([und.T, und[:, ::-1].T], axis=1)
    n = len(pos)
    graph = Data(edge_index=torch.from_numpy(ei).long(), num_nodes=n, pos=torch.from_numpy(pos))
    out["graph/pos"] = pos
    out["graph/edge_index"] = ei

    # 2. MeshEdgeBlock and 3. MeshNodeBlock
    eb = MeshEdgeBlock(16, 16, 16, 16, 1, torch.nn.SiLU(), "LayerNorm")
    nb = MeshNodeBlock("sum", 16, 16, 16, 16, 1, torch.nn.SiLU(), "LayerNorm")
    ef = torch.randn(ei.shape[1], 16)
    nf = torch.randn(n, 16)
    e_new, _ = eb(ef, nf, graph)
    _, n_new = nb(e_new, nf, graph)
    put("edge", eb)
    put("node", nb)
    out["edge/efeat"] = ef.numpy()
    out["edge/nfeat"] = nf.numpy()
    out["edge/y"] = e_new.detach().numpy()
    out["node/y"] = n_new.detach().numpy()

    # 4. BSMS hierarchy and U-processor
    ms = BistrideMultiLayerGraph(graph, 2)
    _, m_gs, m_ids = ms.get_multi_layer_graphs()
    for li, g in enumerate(m_gs):
        out[f"bsms/gs{li}"] = np.asarray(g, dtype=np.int64)
    for li, ids in enumerate(m_ids):
        out[f"bsms/ids{li}"] = np.asarray(ids, dtype=np.int64)
    up = BistrideGraphMessagePassing(unet_depth=2, latent_dim=16, hidden_layer=1, pos_dim=3)
    h = torch.randn(n, 16)
    y = up(h, [torch.tensor(i).long() for i in m_ids],
           [torch.tensor(np.asarray(g)).long() for g in m_gs], torch.from_numpy(pos))
    put("bsms", up)
    out["bsms/h"] = h.numpy()
    out["bsms/y"] = y.detach().numpy()
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out} ({len(out)} arrays) from PhysicsNeMo {commit}")


if __name__ == "__main__":
    main()
