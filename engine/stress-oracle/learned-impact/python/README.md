# Learned-impact PyTorch (smoke only)

Trains the residual MGN-T impact model later; this package is scaffolding + CPU smoke.

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python -m learned_impact.train --smoke
```

Full dataset generation and long training are **not** in scope here.
Default smoke uses `H=8`, `P=4`. Lean production shape is `H=48`, `P=64`.
