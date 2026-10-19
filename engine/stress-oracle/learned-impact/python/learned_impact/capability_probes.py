"""Analytic wave-transition probe and component latency, not a destruction benchmark.

The free 1-D spring chain has an exact modal solution. Learned pair impulses and
pair displacement corrections conserve its total linear momentum and center of
mass by construction. This isolates communication/transition learning without
claiming angular momentum, fracture, contact or native-engine scale validation.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import torch
from torch import nn

from .model import MLP
from .multiscale import MultiscaleProcessor, build_hierarchy, HierarchyCache


def chain(n):
    return torch.stack((torch.arange(n-1), torch.arange(1, n)), -1)


def exact_transition(displacement, velocity, stiffness=100., dt=.8):
    """Unit masses, free boundaries, continuous-time linear dynamics in float64."""
    n = displacement.shape[-1]
    incidence = torch.zeros(n, n-1, dtype=torch.float64)
    incidence[torch.arange(n-1), torch.arange(n-1)] = -1
    incidence[torch.arange(1, n), torch.arange(n-1)] = 1
    matrix = stiffness * incidence @ incidence.T
    values, vectors = torch.linalg.eigh(matrix)
    omega = values.clamp_min(0).sqrt()
    cosine = torch.cos(omega * dt)
    # torch.sinc avoids division by the rigid-translation zero mode.
    sine_over_omega = dt * torch.sinc(omega * dt / torch.pi)
    sine_times_omega = torch.sin(omega * dt) * omega
    u, v = displacement.double() @ vectors, velocity.double() @ vectors
    return ((u * cosine + v * sine_over_omega) @ vectors.T,
            (v * cosine - u * sine_times_omega) @ vectors.T)


def incidence_sum(values, endpoints, size):
    result = values.new_zeros(size)
    result.index_add_(0, endpoints[:, 0], -values)
    result.index_add_(0, endpoints[:, 1], values)
    return result


class WaveTransition(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.node = MLP(3, width, width)
        self.edge = MLP(2, width, width)
        self.processor = MultiscaleProcessor(width)
        self.output = nn.Linear(width, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, u, v, dt, endpoints, hierarchy):
        flat_u, flat_v = u.flatten(), v.flatten()
        a, b = endpoints.T
        node = self.node(torch.stack((flat_u / .05, flat_v, torch.full_like(flat_u, dt)), -1))
        edge = self.edge(torch.stack(((flat_u[b]-flat_u[a]) / .05, flat_v[b]-flat_v[a]), -1))
        _, edge = self.processor(node, edge, endpoints,
                                 torch.ones(len(edge), dtype=torch.bool, device=edge.device), hierarchy)
        exchange = self.output(edge)
        du = incidence_sum(exchange[:, 0], endpoints, len(flat_u)).reshape_as(u)
        dv = incidence_sum(exchange[:, 1], endpoints, len(flat_v)).reshape_as(v)
        return u + dt * v + du, v + dv


def graph(n, batch, levels):
    endpoints = torch.cat([chain(n) + i*n for i in range(batch)])
    hierarchy = build_hierarchy(torch.arange(n*batch), endpoints,
                                torch.ones(len(endpoints), dtype=torch.bool), levels)
    return endpoints, hierarchy


def wave_data(seed, cases=96, n=32, dt=.8):
    generator = torch.Generator().manual_seed(seed)
    u = torch.randn(cases, n, generator=generator) * .02
    v = torch.randn(cases, n, generator=generator) * .5
    target_u, target_v = exact_transition(u, v, dt=dt)
    return u, v, target_u.float(), target_v.float()


@torch.inference_mode()
def wave_metrics(model, dataset, levels, dt, scales):
    u, v, target_u, target_v = dataset
    endpoints, hierarchy = graph(u.shape[1], len(u), levels)
    out_u, out_v = model(u, v, dt, endpoints, hierarchy)
    def score(x, y):
        return float((((x-target_u)/scales[0]).square().mean() +
                      ((y-target_v)/scales[1]).square().mean()) / 2)
    # This probe uses unit masses and stiffness 100. Momentum structure does
    # not imply conservation of mechanical energy; report that separately.
    def energy(x, y):
        return .5 * y.double().square().sum(1) + 50. * x.double().diff().square().sum(1)
    relative_energy_change = (energy(out_u, out_v)-energy(u, v)) / energy(u, v)
    return {'normalized_mse': score(out_u, out_v),
            'ballistic_normalized_mse': score(u + dt*v, v),
            'position_rmse': float((out_u-target_u).square().mean().sqrt()),
            'velocity_rmse': float((out_v-target_v).square().mean().sqrt()),
            'mean_absolute_relative_energy_change': float(relative_energy_change.abs().mean()),
            'mean_signed_relative_energy_change': float(relative_energy_change.mean()),
            'max_momentum_error': float((out_v.double().sum(1)-v.double().sum(1)).abs().max()),
            'max_center_of_mass_error': float((out_u.double().mean(1)-
                                               (u.double()+dt*v.double()).mean(1)).abs().max())}


def run_wave(args):
    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir()
    for name in ('capability_probes.py', 'multiscale.py', 'model.py'):
        shutil.copy2(Path(__file__).parent / name, out / name)
    dt, n, batch = .8, 32, 8
    train = wave_data(601)
    validation = wave_data(602, 24)
    test = wave_data(603, 24)
    scales = ((train[2] - train[0] - dt * train[1]).square().mean().sqrt(),
              (train[3] - train[1]).square().mean().sqrt())
    results = {}
    for name, levels in (('local', 0), ('multiscale', 8)):
        torch.manual_seed(args.seed)
        model = WaveTransition(args.width)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        generator = torch.Generator().manual_seed(args.seed + 71)
        endpoints, hierarchy = graph(n, batch, levels)
        best, selected, history = float('inf'), 0, []
        started = time.monotonic()
        for step in range(1, args.steps + 1):
            indices = torch.randint(len(train[0]), (batch,), generator=generator)
            u, v, target_u, target_v = [x[indices] for x in train]
            optimizer.zero_grad()
            out_u, out_v = model(u, v, dt, endpoints, hierarchy)
            value = (((out_u-target_u)/scales[0]).square().mean() +
                     ((out_v-target_v)/scales[1]).square().mean()) / 2
            if not torch.isfinite(value):
                raise ValueError('wave probe diverged')
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if step % 50 == 0 or step == args.steps:
                metrics = wave_metrics(model, validation, levels, dt, scales)
                history.append({'step': step, 'validation': metrics})
                if metrics['normalized_mse'] < best:
                    best, selected = metrics['normalized_mse'], step
                    torch.save(model.state_dict(), out / (name + '.pt'))
        model.load_state_dict(torch.load(out / (name + '.pt'), weights_only=True))
        results[name] = {'selected_step': selected, 'seconds': time.monotonic()-started,
                         'history': history, 'test': wave_metrics(model, test, levels, dt, scales)}
        print(json.dumps({name: results[name]['test']}), flush=True)
    report = {'scope': __doc__, 'steps': args.steps, 'seed': args.seed, 'width': args.width,
              'nodes_per_chain': n, 'dt': dt, 'stiffness': 100.,
              'selection': 'Validation only; independent initial-state seeds; fixed geometry and material.',
              'results': results, 'native_fracture_validated': False}
    (out / 'review.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


@torch.inference_mode()
def run_benchmark(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('Metal is unavailable; no silent CPU substitution')
    device = torch.device(args.device)
    def sync():
        if device.type == 'mps':
            torch.mps.synchronize()
    rows = []
    for n in (1000, 10000):
        endpoints = chain(n)
        # Add regular local cross-links, not all pairs.
        endpoints = torch.cat((endpoints, torch.stack((torch.arange(n-10), torch.arange(10, n)), -1)))
        connected = torch.ones(len(endpoints), dtype=torch.bool)
        cache = HierarchyCache()
        node_ids = torch.arange(n)
        start = time.perf_counter()
        hierarchy = cache.get(node_ids, endpoints, connected, 8)
        build_ms = (time.perf_counter()-start)*1000
        start = time.perf_counter()
        for _ in range(10):
            cache.get(node_ids, endpoints, connected, 8)
        cached_ms = (time.perf_counter()-start)*100
        node, edge = torch.randn(n, args.width), torch.randn(len(endpoints), args.width)
        processor = MultiscaleProcessor(args.width).eval()
        reference = processor(node, edge, endpoints, connected, hierarchy)
        start = time.perf_counter()
        node, edge, endpoints, connected = [x.to(device) for x in (node, edge, endpoints, connected)]
        hierarchy = [level.to(device) for level in hierarchy]
        processor.to(device)
        sync()
        transfer_ms = (time.perf_counter()-start)*1000
        observed = processor(node, edge, endpoints, connected, hierarchy)
        parity = max(float((x.cpu()-y).abs().max()) for x, y in zip(observed, reference))
        for x, y in zip(observed, reference):
            torch.testing.assert_close(x.cpu(), y, rtol=2e-4, atol=2e-4)
        for _ in range(3):
            processor(node, edge, endpoints, connected, hierarchy)
        sync()
        timings = []
        for _ in range(10):
            start = time.perf_counter()
            processor(node, edge, endpoints, connected, hierarchy)
            sync()
            timings.append((time.perf_counter()-start)*1000)
        timings.sort()
        rows.append({'nodes': n, 'edges': len(endpoints), 'levels': len(hierarchy),
                     'hierarchy_build_ms': build_ms, 'cached_hierarchy_lookup_ms': cached_ms,
                     'initial_transfer_ms': transfer_ms, 'max_abs_cpu_difference': parity,
                     'processor_median_ms':
                     (timings[4]+timings[5])/2, 'processor_max_ms': max(timings)})
    result = {'scope': 'Synthetic graph COMPONENT benchmark; CPU hierarchy construction and synchronized device inference. Not active rigid bodies or full engine; no realtime claim.',
              'device': args.device, 'cpu_threads': 1, 'width': args.width,
              'torch_version': torch.__version__, 'measurements': rows}
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('wave', 'benchmark'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    args = parser.parse_args()
    if args.width < 2 or args.steps < 1:
        parser.error('invalid probe dimensions/budget')
    if args.mode == 'wave' and args.device != 'cpu':
        parser.error('the wave training probe currently supports CPU only')
    (run_wave if args.mode == 'wave' else run_benchmark)(args)


if __name__ == '__main__':
    main()
