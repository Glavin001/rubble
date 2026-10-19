"""Reproducible collection and all-data training for recorded-frame comparisons."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from .dataset import list_events, iter_event_frames
from .export import export_weights
from .losses import total_loss
from .model import LearnedImpactNet
from .normalize import normalize_batch
from .schema import load_event
from .train import _pick_device, _batch_to_device, eval_mean_loss


def event_digest(path: Path) -> str:
    man = json.loads((path / 'manifest.json').read_text())
    h = hashlib.sha256(json.dumps({k: man[k] for k in
        ('schema_version', 'n_frames', 'n_chunks', 'n_bonds', 'K', 'dt', 'seed')}, sort_keys=True).encode())
    context_path = path / 'context.json'
    if context_path.exists():
        context = json.loads(context_path.read_text())
        # These change the meaning/features of a sample even if its raw force
        # bytes happen to match. Names and visualization/provenance do not.
        semantics = {k: context.get(k) for k in ('format', 'bond_frames', 'qs_probe', 'load_basis', 'recorder_revision')}
        # Preserve historical fingerprints when optional new contracts are absent.
        semantics.update({k: context[k] for k in ('load_preview', 'solver_methods') if k in context})
        h.update(json.dumps(semantics, sort_keys=True).encode())
    for cat_path in [path / 'catalog.json', *sorted(path.glob('frames/*/catalog.json'))]:
        for name, entry in sorted(json.loads(cat_path.read_text())['arrays'].items()):
            h.update(json.dumps([name, entry['dtype'], entry['shape']]).encode())
            h.update((cat_path.parent / entry['path']).read_bytes())
    if context_path.exists() and context.get('joint_history'):
        from .joint_history import load_joint_history
        load_joint_history(path,context,man) # Reject incomplete or ambiguous history before fingerprinting.
        directory=path/'joint_history';meta_path=directory/'manifest.json'
        h.update(meta_path.read_bytes())
        for snapshot in json.loads(meta_path.read_text())['snapshots']:
            for key in ('values','flags'):h.update((directory/snapshot[key]).read_bytes())
    if context_path.exists() and context.get('bond_response'):
        from .response_history import load_response_history
        load_response_history(path,context,man)
        directory=path/'bond_response';meta_path=directory/'manifest.json'
        h.update(meta_path.read_bytes())
        for snapshot in json.loads(meta_path.read_text())['snapshots']:
            for key in ('values','flags'):h.update((directory/snapshot[key]).read_bytes())
    return h.hexdigest()


def collection_digest(path: Path) -> str:
    """Conservative duplicate identity including observations outside array files.

    Keep event_digest unchanged for existing provenance/checkpoints. Two clips
    with identical local labels can still have different projectile/forcing
    inputs or world targets, especially no-contact negative examples.
    """
    h=hashlib.sha256(b'learned-collection-observations/2')
    h.update(event_digest(path).encode())
    ignored={'scene.json':('name','description','oracle'),
             'context.json':('source_scene','generator'),
             'replay.json':('scene','source_sha256','verification')}
    for name,keys in ignored.items():
        source=path/name;h.update(name.encode())
        if not source.exists():h.update(b'absent');continue
        data=json.loads(source.read_text())
        for key in keys:data.pop(key,None)
        h.update(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    return h.hexdigest()


def collect(roots: list[Path], out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f'{out} already exists; use a new output directory')
    unique = {}
    for root in roots:
        for path in list_events(root):
            ev = load_event(path)
            for array in [*ev.static.values(), *(v for f in ev.frames for v in f.values())]:
                if not np.isfinite(array).all():
                    raise ValueError(f'non-finite array in {path}')
            digest = collection_digest(path)
            if digest not in unique:
                unique[digest] = {'source': path, 'manifest': ev.manifest, 'aliases': [], 'array_digest':event_digest(path)}
            unique[digest]['aliases'].append(str(path.resolve()))
    if not unique:
        raise ValueError('no valid events')
    out.mkdir(parents=True)
    entries = []
    for digest, record in unique.items():
        src, man = record['source'], record['manifest']
        array_digest=record['array_digest']
        name = f"{man['event_id']}--{digest[:12]}"
        shutil.copytree(src, out / name)
        replay_path = out / name / 'replay.json'
        if replay_path.exists():
            replay = json.loads(replay_path.read_text())
            replay['source_sha256'] = array_digest
            replay['verification'] = 'World poses captured by OracleSession alongside these input and label arrays.'
            replay_path.write_text(json.dumps(replay, allow_nan=False))
        entries.append({**{k: man[k] for k in ('event_id', 'structure_type', 'load_type', 'n_frames', 'n_chunks', 'n_bonds')},
                        'path': name, 'sha256': array_digest, 'collection_sha256':digest, 'sources': record['aliases']})
    manifest = {'schema_version': 'learned-impact/1', 'set': 'combined', 'events': entries,
                'source_roots': [str(p.resolve()) for p in roots],
                'deduplication': 'learned-collection-observations/2: exact arrays/history plus physical scene, world replay and context; descriptive aliases ignored; partial overlapping clips retained',
                'n_frames': sum(e['n_frames'] for e in entries)}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def fit(dataset: Path, out: Path, epochs: int, device: str, lr: float, seed: int) -> dict:
    if out.exists():
        raise FileExistsError(f'{out} already exists; refusing to overwrite a checkpoint')
    if epochs < 1:
        raise ValueError('epochs must be positive')
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    rng = random.Random(seed)
    dev = _pick_device(device)
    events = list_events(dataset)
    raw = [b for ev in events for b in iter_event_frames(ev)]
    batches = [_batch_to_device(normalize_batch(b), dev) for b in raw]
    model = LearnedImpactNet(h=48, p=16).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    out.mkdir(parents=True)
    config = {'dataset': str(dataset.resolve()), 'events': len(events), 'frames': len(batches),
              'epochs': epochs, 'steps': epochs * len(batches), 'seed': seed, 'lr': lr,
              'device': str(dev), 'h': 48, 'p': 16, 'parameters': sum(p.numel() for p in model.parameters()),
              'evaluation': 'in-sample teacher-forced; all collected data used for training',
              'graph_policy': 'remove bonds that failed in earlier oracle frames',
              'gpu_export': 'packed for development only; Slang numerical parity not established',
              'manifest_sha256': hashlib.sha256((dataset / 'manifest.json').read_bytes()).hexdigest()}
    (out / 'run.json').write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps(config), flush=True)
    started = time.monotonic()
    history = []
    for epoch in range(epochs):
        order = list(range(len(batches)))
        rng.shuffle(order)
        model.train()
        summed = 0.
        for idx in order:
            opt.zero_grad(set_to_none=True)
            loss = total_loss(model(batches[idx]), batches[idx])['loss']
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite loss epoch={epoch} frame={idx}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            summed += float(loss.detach())
        row = {'epoch': epoch + 1, 'mean_online_loss': summed / len(batches), 'seconds': time.monotonic() - started}
        history.append(row)
        print(json.dumps(row), flush=True)
        (out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    model.cpu().eval()
    export_weights(model, out, h=48, p=16)
    metrics = eval_mean_loss(model, raw)
    metrics['scope'] = config['evaluation']
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics), flush=True)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--dataset', type=Path, action='append', default=[])
    c.add_argument('--discover', type=Path, help='discover li-* dataset roots immediately below this directory')
    c.add_argument('--out', type=Path, required=True)
    t = sub.add_parser('train')
    t.add_argument('--dataset', type=Path, required=True)
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--epochs', type=int, default=12)
    t.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    t.add_argument('--lr', type=float, default=3e-4)
    t.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.command == 'collect':
        roots = args.dataset
        if args.discover:
            roots += sorted(x for x in args.discover.glob('li-*') if x.is_dir() and (x / 'manifest.json').exists())
        m = collect(roots, args.out)
        print(f"collected {len(m['events'])} unique events / {m['n_frames']} frames from {len(roots)} roots into {args.out}")
    else:
        fit(args.dataset, args.out, args.epochs, args.device, args.lr, args.seed)


if __name__ == '__main__':
    main()
