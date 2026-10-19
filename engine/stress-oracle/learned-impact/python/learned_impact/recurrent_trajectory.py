"""Experimental recurrent multiscale interval predictor.

Reuses audited causal feature definitions and the native response wire decoder,
with a new processor and explicit caller-owned memory. It still forecasts once
per interval; this experiment isolates spatial/temporal representation, not live
within-interval contact feedback or accelerated integration. Never auto-promoted.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .multiscale import RecurrentGraphProcessor, build_hierarchy
from .trajectory_response import (
    CompleteTrajectoryNet, records, masks, query, loss, current_batch, decode_forecast,
)
from .stress_phases import PhaseHistory


class RecurrentTrajectoryNet(CompleteTrajectoryNet):
    architecture = 'physics-recurrent-multiscale-trajectory-v36'

    def __init__(self, h=48, memory=16, levels=8, recurrent=True, knots=16):
        super().__init__(h=h, p=4, knots=knots)
        self.levels, self.memory_width, self.recurrent = levels, memory, recurrent
        # Only the audited feature encoders and current complete-response heads
        # are reused. Legacy processors and unused endpoint heads do not ship.
        retained = {'encode_n', 'encode_e', 'world_encoder', 'impactor_encoder',
                    'forcing_encoder', 'forcing_table_encoder',
                    'constitutive_node_encoder', 'constitutive_edge_encoder',
                    'response_encoder', 'dynamic_projectile_encoder',
                    'boundary_encoder', 'complete_head', 'flag_head'}
        for name in list(self._modules):
            if name not in retained:
                delattr(self, name)
        self.graph_processor = RecurrentGraphProcessor(h, memory, recurrent)

    def hierarchy(self, batch):
        node_ids = torch.arange(len(batch['chunk_mass']), dtype=torch.long,
                                device=batch['bond_endpoints'].device)
        return [level.to(node_ids.device) for level in build_hierarchy(
            node_ids, batch['bond_endpoints'], batch['bond_failure_eligible'], self.levels)]

    def step(self, raw, *, scene, time, state=None, hierarchy=None, dt=None):
        batch = self.prepare_batch(raw)
        dt = float(batch['dt']) if dt is None else dt
        if not torch.equal(batch['dt'], batch['dt'].new_tensor(dt)):
            raise ValueError('sequence timestep disagrees with physical input')
        if hierarchy is None:
            hierarchy = self.hierarchy(batch)
        _, edge, next_state = self.graph_processor(
            self.encode_nodes(batch), self.encode_edges(batch), batch['bond_endpoints'],
            batch['bond_failure_eligible'], hierarchy, batch['bond_ids'],
            scene=scene, time=time, dt=dt, state=state)
        return self.edge_outputs(edge, batch), next_state

    def forward(self, batch):
        raise RuntimeError('recurrent inference requires step with explicit scene, clock and memory')


def sequences(items):
    """Whole parent episodes; reject missing prefixes, clock gaps and leakage."""
    grouped = {}
    for record in items:
        grouped.setdefault(record['event'], []).append(record)
    result = []
    for event, rows in sorted(grouped.items()):
        rows.sort(key=lambda r: r['frame'])
        if [r['frame'] for r in rows] != list(range(len(rows))):
            raise ValueError(f'episode {event} requires its complete contiguous warmup prefix')
        if len({r['partition'] for r in rows}) != 1:
            raise ValueError('episode partition leakage')
        for a, b in zip(rows, rows[1:]):
            if abs(a['start_time'] + a['dt'] - b['start_time']) > 1e-9:
                raise ValueError('episode clock gap')
        result.append(rows)
    return result


@torch.no_grad()
def fit_scales(model, items):
    if not items or any(r['partition'] != 'train' for r in items):
        raise ValueError('scales require training parents only')
    sums = torch.zeros(42, dtype=torch.float64)
    counts = torch.zeros_like(sums)
    for r in items:
        weight = masks(r).double()
        delta = r['target'].double() - r['batch']['trajectory_base'][None].double()
        sums += (delta.square() * weight).sum((0, 1))
        counts += weight.sum((0, 1))
    variance = sums / counts.clamp_min(1e-20)
    scale = torch.where(variance > 0, variance.sqrt(), torch.ones_like(variance))
    for sl in (slice(21, 24), slice(24, 27), slice(27, 30), slice(30, 33)):
        value = float((sums[sl].sum() / counts[sl].sum()).sqrt())
        scale[sl] = value if value > 0 else 1.
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('invalid training scales')
    model.full_scale.copy_(scale.float())
    model.trajectory_scale.copy_(model.full_scale[21:27])
    model.fit_boundary_scale(items)


@torch.inference_mode()
def evaluate(model, episodes):
    model.eval()
    objective = squared = mass = 0.
    frames = 0
    by_event = []
    for episode in episodes:
        memory = None
        event_squared = event_mass = 0.
        for r in episode:
            pred, memory = model.step(r['batch'], scene=r['event'], time=r['start_time'],
                                      state=memory, hierarchy=r['hierarchy'], dt=r['dt'])
            objective += float(loss(pred, r, model.full_scale))
            values, _, _ = query(pred, r)
            weight = (r['weights'][:, None] * r['mask']).double()
            error = values[:, :, 21:24].double() - r['target'][:, :, 21:24].double()
            event_squared += float((error.square() * weight[:, :, None]).sum())
            event_mass += float(weight.sum())
            frames += 1
        by_event.append({'event': episode[0]['event'], 'force_rmse_n':
                         (event_squared / (3 * event_mass)) ** .5 if event_mass else None})
        squared += event_squared
        mass += event_mass
    if not frames or not mass:
        raise ValueError('empty evaluation')
    return {'frames': frames, 'objective': objective / frames,
            'force_rmse_n': (squared / (3 * mass)) ** .5, 'by_event': by_event,
            'scope': 'Chronological oracle-state inputs with recurrent memory; not independent motion.'}


def load_model(path):
    path = Path(path)
    config = json.loads((path / 'config.json').read_text())
    if config['architecture'] != RecurrentTrajectoryNet.architecture:
        raise ValueError('recurrent trajectory checkpoint required')
    model = RecurrentTrajectoryNet(**{key: config[key] for key in
                                     ('h', 'memory', 'levels', 'recurrent', 'knots')})
    model.load_state_dict(torch.load(path / 'state_dict.pt', map_location='cpu', weights_only=True))
    model.eval()
    return model, {'sha256': PhaseHistory._hash(path / 'state_dict.pt'),
                   'architecture': model.architecture, 'run': config, 'h': model.h, 'p': 0}


@torch.inference_mode()
def forecast(model, graph, snapshot, kinematics, state=None, *, hierarchy_cache=None):
    batch = current_batch(graph, snapshot, kinematics)
    hierarchy = None
    if hierarchy_cache is not None:
        hierarchy = hierarchy_cache.get(torch.arange(graph.n, dtype=torch.long),
                                         batch['bond_endpoints'], batch['bond_failure_eligible'], model.levels)
    pred, state = model.step(batch, scene=graph.scene['name'],
                             time=snapshot['world']['time'], state=state, dt=graph.dt, hierarchy=hierarchy)
    return decode_forecast(graph, snapshot, batch, pred), state


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir()
    started = time.monotonic()
    frozen = out / 'sources'
    frozen.mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source, frozen / source.name)
    (out / 'source-hashes.json').write_text(json.dumps({p.name: PhaseHistory._hash(p)
                                                      for p in frozen.iterdir()}, indent=2) + '\n')
    items, sources, boundary_sources = records(args.dataset, args.boundary_root, args.knots)
    model = RecurrentTrajectoryNet(args.h, args.memory, args.levels, args.recurrent, args.knots)
    fit_scales(model, [r for r in items if r['partition'] == 'train'])
    for r in items:
        r['hierarchy'] = model.hierarchy(r['batch'])
    episodes = sequences(items)
    groups = {part: [s for s in episodes if s[0]['partition'] == part]
              for part in ('train', 'validation', 'test')}
    if not all(groups.values()):
        raise ValueError('all partitions required')
    config = {**vars(args), 'architecture': model.architecture, 'scope': __doc__,
              'phase_sources': sources, 'boundary_sources': boundary_sources,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'initialization': 'Random from saved seed; no pretrained checkpoint.',
              'selection': 'Minimum chronological validation complete-response objective.',
              'training': 'Whole training-parent episodes; truncated backpropagation; physical inputs remain oracle states.',
              'promoted': False}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed)
    baseline = evaluate(model, groups['validation'])
    best, selected, updates, history = float('inf'), 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.
        frame_count = 0
        for index in rng.permutation(len(groups['train'])):
            episode = groups['train'][index]
            memory = None
            for start in range(0, len(episode), args.unroll):
                window = episode[start:start + args.unroll]
                optimizer.zero_grad()
                terms = []
                for r in window:
                    pred, memory = model.step(r['batch'], scene=r['event'], time=r['start_time'],
                                              state=memory, hierarchy=r['hierarchy'], dt=r['dt'])
                    terms.append(loss(pred, r, model.full_scale))
                value = torch.stack(terms).mean()
                if not torch.isfinite(value):
                    raise ValueError('nonfinite recurrent training loss')
                value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                memory = memory.detach()
                updates += 1
                train_sum += float(value.detach()) * len(window)
                frame_count += len(window)
        validation = evaluate(model, groups['validation'])
        if validation['objective'] < best:
            best, selected = validation['objective'], epoch
            torch.save(model.state_dict(), out / 'state_dict.pt')
        row = {'epoch': epoch, 'updates': updates, 'train_loss': train_sum / frame_count,
               'validation': validation, 'seconds': time.monotonic() - started}
        history.append(row)
        (out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row), flush=True)
    model.load_state_dict(torch.load(out / 'state_dict.pt', map_location='cpu', weights_only=True))
    report = {'status': 'completed', 'selected_epoch': selected, 'updates': updates,
              'seconds': time.monotonic() - started, 'baseline_validation': baseline,
              'checkpoint_sha256': PhaseHistory._hash(out / 'state_dict.pt'),
              'metrics': {part: evaluate(model, rows) for part, rows in groups.items()},
              'promoted': False}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'boundary-root', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--h', type=int, default=48)
    parser.add_argument('--memory', type=int, default=16)
    parser.add_argument('--levels', type=int, default=8)
    parser.add_argument('--recurrent', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--knots', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--unroll', type=int, default=4)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=.001)
    args = parser.parse_args()
    if args.h < 2 or min(args.memory, args.epochs, args.unroll, args.threads) < 1 or args.levels < 0 or args.knots < 2 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('invalid training settings')
    train(args)


if __name__ == '__main__':
    main()
