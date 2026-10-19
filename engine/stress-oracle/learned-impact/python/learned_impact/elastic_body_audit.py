"""Audit six-DOF propagation on read-only f64 native linear graph exports."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from .elastic_body_graph import ElasticBodyGraph, ElasticBodyTransition, exact_body_transition
from .physical_transition import ChainState
from .wave_operator import CoupledWaveOperator


def run(args):
    torch.set_num_threads(1)
    checkpoint = torch.load(args.weights, weights_only=True)
    learned = CoupledWaveOperator(checkpoint['coefficients'], checkpoint['dt'], checkpoint['stiffness'])
    results = []
    for case, source in enumerate(args.graph):
        path = Path(source)
        data = json.loads(path.read_text())
        if data['schema'] != 'native-linear-elastic-graph/1':
            raise ValueError('unsupported native elastic export')
        for index, cluster in enumerate(data['clusters']):
            graph = ElasticBodyGraph.from_native(cluster)
            probe = torch.tensor(cluster['probe_displacement'], dtype=torch.float64).flatten()
            response = graph.response(ChainState(probe, torch.zeros_like(probe)))
            targets = {'deformation': torch.tensor([b['probe_deformation'] for b in cluster['bonds']], dtype=torch.float64).reshape(-1, 6),
                       'bond_force': torch.tensor([b['probe_force'] for b in cluster['bonds']], dtype=torch.float64).reshape(-1, 6),
                       'internal_nodal_force': torch.tensor(cluster['probe_nodal_force'], dtype=torch.float64).flatten()}
            parity = {}
            for name, target in targets.items():
                error = float((response[name]-target).abs().max()) if target.numel() else 0.
                scale = float(target.abs().max()) if target.numel() else 0.
                parity[name] = {'max_abs_error': error, 'target_max_abs': scale}
                if error > 1e-12*max(scale, 1e-15):
                    raise ValueError(f'native {name} mismatch: {error} of {scale}')
            bound = graph.spectral_bound()
            max_dt = learned.dt*math.sqrt(4*learned.stiffness/bound) if bound else None
            dt = args.dt if args.dt is not None else .9*max_dt
            model = ElasticBodyTransition(learned, graph, dt)
            rng = torch.Generator().manual_seed(args.seed+case*19+index)
            free = (~graph.fixed).flatten()
            initial = ChainState(torch.randn(12, graph.n*6, generator=rng, dtype=torch.float64)*1e-7*free,
                                 torch.randn(12, graph.n*6, generator=rng, dtype=torch.float64)*.005*free)
            actual, expected = initial, initial
            base_energy = graph.energy(initial)
            work = torch.zeros_like(base_energy)
            work_scale = base_energy.clone()
            defect = state_error = 0.
            for step in range(32):
                force = torch.randn(12, graph.n*6, generator=rng, dtype=torch.float64)*.01
                old = actual
                actual = model.step(actual, force)
                expected = exact_body_transition(graph, expected, dt, force)
                dw = (force*(actual.displacement-old.displacement)).sum(-1)
                work += dw
                work_scale += dw.abs()
                defect = max(defect, float(((graph.energy(actual)-base_energy-work)/work_scale).abs().max()))
                delta = ChainState(actual.displacement-expected.displacement, actual.velocity-expected.velocity)
                state_error = max(state_error, float((graph.energy(delta)/graph.energy(expected)).max()))
            row = {'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'cluster': index, 'chunks': graph.n, 'bonds': len(graph.stiffness), 'dof': int(free.sum()),
                   'parity': parity, 'spectral_bound': bound, 'degree': learned.degree,
                   'steps': 32, 'dt_s': dt, 'spectral_domain_max_dt_s': max_dt,
                   'calls_to_cover_16_67ms_using_spectral_bound': math.ceil((1/60)/max_dt) if max_dt else 1,
                   'max_relative_work_balance_defect': defect, 'max_relative_state_energy_error': state_error,
                   'translation_rmse_m': float((actual.displacement-expected.displacement).reshape(-1, graph.n, 6)[..., :3].square().mean().sqrt()),
                   'rotation_rmse_rad': float((actual.displacement-expected.displacement).reshape(-1, graph.n, 6)[..., 3:].square().mean().sqrt()),
                   'velocity_rmse_m_s': float((actual.velocity-expected.velocity).reshape(-1, graph.n, 6)[..., :3].square().mean().sqrt()),
                   'angular_velocity_rmse_rad_s': float((actual.velocity-expected.velocity).reshape(-1, graph.n, 6)[..., 3:].square().mean().sqrt())}
            results.append(row)
            print(json.dumps(row), flush=True)
    report = {'status': 'completed', 'scope': 'Frozen native six-DOF linear mechanics and exact modal teacher. No damping, contact, fracture, corotation, native rollout or speed claim.',
              'operator_sha256': hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(), 'results': results}
    with Path(args.out).open('x') as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', action='append', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--dt', type=float)
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--out', required=True)
    run(parser.parse_args())
