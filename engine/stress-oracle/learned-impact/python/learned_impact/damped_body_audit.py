"""Measure native damping and risky simplifications on frozen small graphs.

Uses every free physical DOF. No production setting, material or timestep is
changed. The two deliberately inaccurate controls only diagnose representation:
ignore damping, or delete coupling between undamped modes. Neither is adopted.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .damped_body_graph import DampedBodyGraph, DenseDampedReference
from .physical_transition import ChainState


def error_energy(body, actual, expected, initial_energy):
    return float((body.energy(ChainState(actual.displacement-expected.displacement,
                                        actual.velocity-expected.velocity))/initial_energy).max())


def run(args):
    torch.set_num_threads(1)
    rows = []
    for case, source in enumerate(args.graph):
        path = Path(source)
        data = json.loads(path.read_text())
        if data['schema'] != 'native-linear-elastic-graph/2':
            raise ValueError('damping audit requires v2 export including actual chunk mass damping')
        for index, cluster in enumerate(data['clusters']):
            graph = DampedBodyGraph.from_native(cluster)
            body, ref = graph.body, DenseDampedReference(graph)
            probe = ChainState(torch.tensor(cluster['probe_displacement'], dtype=torch.float64).flatten(),
                               torch.tensor(cluster['probe_velocity'], dtype=torch.float64).flatten())
            response = graph.response(probe)
            parity = {}
            for name, target in (
                ('bond_damping_force', torch.tensor([b['probe_damping_force'] for b in cluster['bonds']], dtype=torch.float64).reshape(-1, 6)),
                ('damping_nodal_force', torch.tensor(cluster['probe_damping_nodal_force'], dtype=torch.float64).flatten()),
            ):
                error = float((response[name]-target).abs().max())
                scale = float(target.abs().max())
                parity[name] = dict(max_abs_error=error, target_max_abs=scale)
                if error > 1e-12*max(scale, 1e-15):
                    raise ValueError(f'native damping mismatch: {name}, {error} of {scale}')
            rng = torch.Generator().manual_seed(args.seed+case*19+index)
            free = (~body.fixed).flatten()
            initial = ChainState(torch.randn(8, body.n*6, generator=rng, dtype=torch.float64)*1e-7*free,
                                 torch.randn(8, body.n*6, generator=rng, dtype=torch.float64)*.005*free)
            energy = body.energy(initial)
            encoded = ref.encode(initial)
            short_dt = .5/ref.omega
            short = ref.step(initial, short_dt)
            # Independently staged NumPy RK4 checks the matrix exponential.
            a = ref.generator.numpy()
            rk4 = []
            for subdivisions in (8, 16):
                state = encoded.numpy().copy()
                h = short_dt/subdivisions
                for _ in range(subdivisions):
                    k1 = state@a.T
                    k2 = (state+.5*h*k1)@a.T
                    k3 = (state+.5*h*k2)@a.T
                    k4 = (state+h*k3)@a.T
                    state += h/6*(k1+2*k2+2*k3+k4)
                rk4.append(error_energy(body, ref.decode(torch.from_numpy(state)), short, energy))
            if rk4[1] > 1e-12 or rk4[1] > rk4[0]:
                raise ValueError(f'RK4 reference check failed: {rk4}')

            # Independent power quadrature over the short window.
            p, _ = ref.propagator(short_dt/128)
            z = encoded.clone()
            integral = torch.zeros_like(energy)
            for step in range(129):
                weight = 1 if step in (0, 128) else (4 if step % 2 else 2)
                integral += weight*graph.dissipated_power(ref.decode(z).velocity)
                z = z@p.T
            integral *= short_dt/(3*128)
            balance = float(((body.energy(short)-energy+integral)/energy).abs().max())
            if balance > 1e-10:
                raise ValueError(f'damping work balance mismatch: {balance}')

            dt = 1/120
            transition, _ = ref.propagator(dt)
            half, _ = ref.propagator(dt/2)
            semigroup = error_energy(body, ref.decode(encoded@transition.T), ref.decode(encoded@half.T@half.T), energy)
            values, modes = torch.linalg.eigh(ref.k)
            modal_c = modes.T@ref.c@modes
            diagonal_c = modes@torch.diag(modal_c.diagonal())@modes.T
            offdiag = float(torch.linalg.norm(modal_c-torch.diag(modal_c.diagonal()))/torch.linalg.norm(modal_c))
            commutator = float(torch.linalg.norm(ref.k@ref.c-ref.c@ref.k)/(torch.linalg.norm(ref.k)*torch.linalg.norm(ref.c)))
            controls = {
                'ignore_damping': torch.linalg.matrix_exp(dt*ref.make_generator(ref.k, torch.zeros_like(ref.c))),
                'diagonal_modal_damping': torch.linalg.matrix_exp(dt*ref.make_generator(ref.k, diagonal_c)),
            }
            trajectory = []
            z = encoded.clone()
            alternatives = {name: encoded.clone() for name in controls}
            for frame in range(1, 5):
                z = z@transition.T
                expected = ref.decode(z)
                metrics = {}
                for name, control in controls.items():
                    alternatives[name] = alternatives[name]@control.T
                    metrics[name] = error_energy(body, ref.decode(alternatives[name]), expected, energy)
                trajectory.append({'time_s': frame*dt, 'reference_energy_fraction_max': float((body.energy(expected)/energy).max()),
                                   'control_max_error_energy_over_initial': metrics})
            # All individual undamped modes are retained in this diagnostic.
            # Their initial velocity is a unit-energy state. Full C can couple them.
            modal_initial = ref.k.new_zeros(ref.dof, ref.dof*2)
            modal_initial[:, ref.dof:] = modes.T
            modal_final = ref.decode(modal_initial@transition.T)
            retention = body.energy(modal_final)/.5
            high = values >= .5*values.max()
            row = {'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'cluster': index, 'chunks': body.n, 'bonds': len(body.stiffness), 'free_dof': ref.dof,
                   'mass_damping_alpha_per_s': cluster['mass_damping_alpha_per_s'], 'native_parity': parity,
                   'highest_angular_frequency_per_s': ref.omega,
                   'normalized_stiffness_damping_commutator': commutator,
                   'damping_offdiagonal_fraction_in_undamped_basis': offdiag,
                   'rk4_error_energy_over_initial_8_16_subdivisions': rk4,
                   'short_window_relative_work_balance_defect': balance,
                   'semigroup_error_energy_over_initial': semigroup, 'trajectory_120hz': trajectory,
                   'high_frequency_mode_count': int(high.sum()),
                   'high_frequency_max_energy_retained_after_8_33ms': float(retention[high].max())}
            rows.append(row)
            print(json.dumps(row), flush=True)
    with Path(args.out).open('x') as file:
        json.dump({'scope': 'Frozen small-motion native linear damping audit. Dense reference only; no learned accuracy, fracture, contact, native rollout or speed claim.',
                   'results': rows}, file, indent=2, allow_nan=False)
        file.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', action='append', required=True)
    parser.add_argument('--seed', type=int, default=1201)
    parser.add_argument('--out', required=True)
    run(parser.parse_args())
