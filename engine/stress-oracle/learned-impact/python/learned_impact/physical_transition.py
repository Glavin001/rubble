"""Explicit physical-state transition experiment on free linear spring chains.

Only displacement and velocity persist. Bond extension, force and stored energy
are derived. The optional decoder projects each independent chain onto its known
momentum/energy surface. This is valid only for the undamped, unforced, fixed
topology probe; it is not a fracture/contact decoder or a native-engine change.
Energy projection alone does not guarantee correct phase, local energy transport,
time composition, or physical accuracy. Those are measured separately.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time

import torch

from .capability_probes import WaveTransition, exact_transition, graph, wave_data
from .stress_phases import PhaseHistory


@dataclass(frozen=True)
class ChainState:
    displacement: torch.Tensor
    velocity: torch.Tensor

    def energy(self, stiffness=100.):
        return (.5 * self.velocity.square().sum(-1)
                + .5 * stiffness * self.displacement.diff().square().sum(-1))

    def bond_response(self, stiffness=100.):
        extension = self.displacement.diff()
        return {'extension': extension, 'force': stiffness * extension,
                'stored_energy': .5 * stiffness * extension.square()}


def conservative_projection(initial, proposed, dt, stiffness=100.):
    """Per-chain physical projection, without a target or hidden temporal state.

    Strip rigid translation, preserve its exact ballistic evolution, and rescale
    the proposed internal displacement AND velocity by one common factor so their
    joint mechanical energy equals the initial internal energy. Zero-energy input
    stays rigid. A collapsed proposal uses the initial internal state as its
    explicitly defined direction. This is a nonlocal per-island projection, not a
    local material law or a claim of exact wave propagation.
    """
    u, v = initial.displacement, initial.velocity
    center, speed = u.mean(-1, keepdim=True), v.mean(-1, keepdim=True)
    original = ChainState(u-center, v-speed)
    candidate = ChainState(proposed.displacement-proposed.displacement.mean(-1, keepdim=True),
                           proposed.velocity-proposed.velocity.mean(-1, keepdim=True))
    candidate_energy = candidate.energy(stiffness)
    collapsed = candidate_energy == 0
    internal = ChainState(torch.where(collapsed[..., None], original.displacement, candidate.displacement),
                          torch.where(collapsed[..., None], original.velocity, candidate.velocity))
    old_energy, new_energy = original.energy(stiffness), internal.energy(stiffness)
    # Exact zero is a separate physical state; do not add fictitious energy eps.
    denominator = torch.where(new_energy > 0, new_energy, torch.ones_like(new_energy))
    ratio = old_energy / denominator
    # Keep sqrt's gradient finite in the rigid zero-energy case.
    scale = torch.where(old_energy > 0, torch.where(old_energy > 0, ratio, torch.ones_like(ratio)).sqrt(),
                        torch.zeros_like(ratio))
    return ChainState(center + dt*speed + scale[..., None]*internal.displacement,
                      speed + scale[..., None]*internal.velocity)


class PhysicalChainTransition(WaveTransition):
    def __init__(self, width=32, project=True):
        super().__init__(width)
        self.project = project

    def step(self, state, dt, endpoints, hierarchy):
        u, v = super().forward(state.displacement, state.velocity, dt, endpoints, hierarchy)
        proposed = ChainState(u, v)
        return conservative_projection(state, proposed, dt) if self.project else proposed


def state_loss(actual, target, scales):
    return (((actual.displacement-target.displacement)/scales[0]).square().mean()
            + ((actual.velocity-target.velocity)/scales[1]).square().mean()) / 2


@torch.inference_mode()
def evaluate(model, dataset, dt, scales, steps=1):
    state = ChainState(dataset[0].clone(), dataset[1].clone())
    initial = state
    target = ChainState(dataset[2], dataset[3])
    endpoints, hierarchy = graph(state.displacement.shape[-1], len(state.displacement), 8)
    worst_energy = max_momentum = max_center = 0.
    errors = []
    for index in range(steps):
        state = model.step(state, dt, endpoints, hierarchy)
        if not torch.isfinite(state.displacement).all() or not torch.isfinite(state.velocity).all():
            return {'status': 'nonfinite', 'failed_step': index+1}
        if index:
            target = ChainState(*exact_transition(initial.displacement, initial.velocity, dt=dt*(index+1)))
        errors.append(float(state_loss(state, target, scales)))
        energy_error = (state.energy().double()-initial.energy().double()).abs()/initial.energy().double()
        worst_energy = max(worst_energy, float(energy_error.max()))
        max_momentum = max(max_momentum, float((state.velocity.double().sum(-1)-initial.velocity.double().sum(-1)).abs().max()))
        expected_center = (initial.displacement.double()+dt*(index+1)*initial.velocity.double()).mean(-1)
        max_center = max(max_center, float((state.displacement.double().mean(-1)-expected_center).abs().max()))
    return {'status': 'finite', 'steps': steps, 'normalized_mse': errors[-1],
            'mean_step_normalized_mse': sum(errors)/len(errors),
            'max_relative_energy_error': worst_energy, 'max_momentum_error': max_momentum,
            'max_center_of_mass_error': max_center, 'step_errors': errors}


def train(args):
    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir()
    source_dir = out / 'sources'
    source_dir.mkdir()
    for filename in ('physical_transition.py', 'capability_probes.py', 'multiscale.py', 'model.py'):
        shutil.copy2(Path(__file__).parent / filename, source_dir / filename)
    (out / 'source-hashes.json').write_text(json.dumps({p.name: PhaseHistory._hash(p) for p in source_dir.iterdir()}, indent=2)+'\n')
    dt, n, batch = .8, 32, 8
    train_data, validation, test = wave_data(601), wave_data(602, 24), wave_data(603, 24)
    scales = ((train_data[2]-train_data[0]-dt*train_data[1]).square().mean().sqrt(),
              (train_data[3]-train_data[1]).square().mean().sqrt())
    endpoints, hierarchy = graph(n, batch, 8)
    results = {}
    for name, project in (('unconstrained', False), ('energy_projected', True)):
        torch.manual_seed(args.seed)
        model = PhysicalChainTransition(args.width, project)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        rng = torch.Generator().manual_seed(args.seed+71)
        best, selected, history = float('inf'), 0, []
        started = time.monotonic()
        for step in range(1, args.steps+1):
            indices = torch.randint(len(train_data[0]), (batch,), generator=rng)
            u, v, tu, tv = [x[indices] for x in train_data]
            optimizer.zero_grad()
            actual = model.step(ChainState(u, v), dt, endpoints, hierarchy)
            value = state_loss(actual, ChainState(tu, tv), scales)
            if not torch.isfinite(value):
                raise ValueError('nonfinite physical transition training loss')
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if step % 50 == 0 or step == args.steps:
                metric = evaluate(model, validation, dt, scales)
                history.append({'step': step, 'validation': metric})
                if metric['normalized_mse'] < best:
                    best, selected = metric['normalized_mse'], step
                    torch.save(model.state_dict(), out / (name+'.pt'))
        model.load_state_dict(torch.load(out / (name+'.pt'), weights_only=True))
        results[name] = {'selected_step': selected, 'seconds': time.monotonic()-started,
                         'history': history, 'test': evaluate(model, test, dt, scales),
                         'test_rollout': evaluate(model, test, dt, scales, 32),
                         'checkpoint_sha256': PhaseHistory._hash(out / (name+'.pt'))}
        print(json.dumps({name: {k: v for k, v in results[name].items() if k != 'history'}}), flush=True)
    report = {'status': 'completed', 'scope': __doc__, 'seed': args.seed, 'steps': args.steps,
              'width': args.width, 'physical_dt': dt, 'nodes': n, 'results': results,
              'selection': 'Minimum one-transition validation MSE; test and 32-step rollout used only after selection.',
              'persistent_state': ['displacement', 'velocity'], 'native_validated': False}
    (out / 'review.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--width', type=int, default=32)
    args = parser.parse_args()
    if args.steps < 1 or args.width < 2:
        parser.error('positive training budget and width >= 2 required')
    train(args)


if __name__ == '__main__':
    main()
