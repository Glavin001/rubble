"""Copy a stratified real subset of P1's calls into a scratch dir, for pack_data.py (the
sources are only read):

    python3 subset_data.py <P1 runs dir> <out> <n_train_events> <n_val_ensembles>

Train: `n_train` events of `calls-v3-train`, evenly spaced within each family in proportion
to its size. Val: `n_val_ensembles` complete 8-member ensembles of `calls-v3-eval`'s val
split, round-robin over the families. Fails if a split group would be in both.
(The 2026-10-22 trial: `… sub100 100 3` → 100 train events, 24 val events.)"""
import collections
import json
import os
import shutil
import sys

SRC, out, n_train, n_ens = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])


def fam(e):
    return "_".join(e.split("_")[:4])


def man(d, e):
    with open(f"{SRC}/{d}/{e}/manifest.json") as f:
        return json.load(f)


tr = sorted(e for e in os.listdir(f"{SRC}/calls-v3-train")
            if os.path.exists(f"{SRC}/calls-v3-train/{e}/manifest.json"))
by = collections.defaultdict(list)
for e in tr:
    by[fam(e)].append(e)
pick = []
for f, evs in sorted(by.items()):
    k = max(1, round(n_train * len(evs) / len(tr)))
    pick += [evs[int(i * len(evs) / k)] for i in range(k)]
ens = collections.defaultdict(list)
for e in sorted(os.listdir(f"{SRC}/calls-v3-eval")):
    m = man("calls-v3-eval", e)
    if m["split"] == "val":
        ens[(fam(e), m["ensemble_id"])].append(e)
vf = collections.defaultdict(list)
for (f, _), evs in sorted(ens.items()):
    if len(evs) == 8:
        vf[f].append(evs)
val = []
fams = sorted(vf)
for i in range(n_ens):
    f = fams[i % len(fams)]
    val += vf[f][(i // len(fams) * 7) % len(vf[f])]
g_tr = {man("calls-v3-train", e)["split_group"] for e in pick}
g_va = {man("calls-v3-eval", e)["split_group"] for e in val}
assert not g_tr & g_va, g_tr & g_va
for sub, src, evs in (("train", "calls-v3-train", pick), ("val", "calls-v3-eval", val)):
    os.makedirs(f"{out}/{sub}", exist_ok=True)
    for e in evs:
        if not os.path.exists(f"{out}/{sub}/{e}"):
            shutil.copytree(f"{SRC}/{src}/{e}", f"{out}/{sub}/{e}")
print(json.dumps({"train": len(pick), "train_by_family": collections.Counter(map(fam, pick)),
                  "val": len(val), "val_by_family": collections.Counter(map(fam, val))}))
