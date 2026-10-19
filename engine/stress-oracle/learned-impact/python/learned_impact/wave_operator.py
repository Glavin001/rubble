"""Coupled learned linear graph transition, with physical state as its only memory.

Controlled free-chain experiment, unit masses and stiffness 100. Two learned
Chebyshev polynomials C(L), S(L) advance both physical fields:
u_next = C(L) u + S(L) v; v_next = -L S(L) u + C(L) v.
The coefficients are fitted from training transitions, never from test labels.
No eigenvectors, oracle calls, temporal hidden state or native substeps are used
at inference. This is a linear-physics capability probe, not a nonlinear native
destruction solver, and polynomial degree remains sequential graph work.
"""
import argparse
import json
from pathlib import Path
import shutil

import torch

from .capability_probes import wave_data, exact_transition
from .physical_transition import ChainState, state_loss
from .stress_phases import PhaseHistory


def stiffness_apply(x, stiffness=100.):
    """B^T diag(k) B on separate free chains; all physical edges are retained."""
    tension = stiffness*x.diff()
    result = torch.zeros_like(x)
    result[..., :-1] -= tension
    result[..., 1:] += tension
    return result


def polynomial_basis(x, degree, stiffness=100.):
    # Gershgorin bound 4k contains the unit-mass free chain's entire spectrum.
    def scaled_operator(value):
        return stiffness_apply(value, stiffness)/(2*stiffness)-value
    older, previous = x, scaled_operator(x)
    result = [previous+x]  # T1(z) - T1(-1), exact zero on the rigid mode.
    for index in range(2, degree+1):
        current = 2*scaled_operator(previous)-older
        result.append(current-((-1)**index)*x)
        older, previous = previous, current
    return torch.stack(result, -1)


class CoupledWaveOperator:
    def __init__(self, coefficients, dt, stiffness=100.):
        if coefficients.ndim != 1 or len(coefficients) % 2 or len(coefficients) < 2:
            raise ValueError('two equally sized nonempty polynomial coefficient vectors required')
        if not torch.isfinite(coefficients).all() or dt <= 0 or stiffness <= 0:
            raise ValueError('invalid physical operator')
        self.coefficients, self.dt, self.stiffness = coefficients, dt, stiffness
        self.degree = len(coefficients)//2

    def design(self, state):
        u, v = state.displacement, state.velocity
        pu, pv = [polynomial_basis(x, self.degree, self.stiffness) for x in (u, v)]
        # stiffness_apply acts along the physical-node axis, not coefficient axis.
        lpu = stiffness_apply(pu.movedim(-1, -2), self.stiffness).movedim(-2, -1)
        return torch.cat((pu, pv), -1), torch.cat((pv, -lpu), -1)

    def step(self, state, dt=None):
        if dt is not None and dt != self.dt:
            raise ValueError('this fitted operator has one declared physical interval')
        u, v = state.displacement, state.velocity
        xu, xv = self.design(state)
        coefficients = self.coefficients.to(dtype=u.dtype, device=u.device)
        return ChainState(u+self.dt*v+xu@coefficients,
                          v-self.dt*stiffness_apply(u, self.stiffness)+xv@coefficients)


def fit(dataset, degree, dt, scales, *, force_consistent=False):
    u, v, tu, tv = [x.double() for x in dataset]
    model = CoupledWaveOperator(torch.zeros(2*degree, dtype=torch.float64), dt)
    xu, xv = model.design(ChainState(u, v))
    baseline = model.step(ChainState(u, v))
    design = torch.cat((xu.reshape(-1, 2*degree)/scales[0], xv.reshape(-1, 2*degree)/scales[1]))
    target = torch.cat(((tu-baseline.displacement).reshape(-1)/scales[0],
                        (tv-baseline.velocity).reshape(-1)/scales[1]))
    # A constant force's rigid mode requires R(0)=dt^2/2, where
    # R=(I-C)/L. Enforce C'(0)=-dt^2/2 by eliminating one coefficient.
    origin = torch.zeros(2*degree, dtype=torch.float64)
    nullspace = torch.eye(2*degree, dtype=torch.float64)
    if force_consistent:
        index = torch.arange(1, degree+1, dtype=torch.float64)
        derivative = 2/(4*model.stiffness)*((-1.)**(index-1))*index.square()
        origin[0] = (-.5*dt*dt)/derivative[0]
        nullspace = nullspace[:, 1:]
        nullspace[0, :degree-1] = -derivative[1:]/derivative[0]
    solution = torch.linalg.lstsq(design@nullspace, target-design@origin, driver='gelsd')
    if int(solution.rank) != nullspace.shape[1]:
        raise ValueError('training transitions do not identify the operator coefficients')
    model.coefficients = origin+nullspace@solution.solution
    return model


@torch.inference_mode()
def metrics(model, dataset, scales, steps=1):
    initial = ChainState(dataset[0].double(), dataset[1].double())
    state = initial
    errors = []
    maximum_energy = maximum_momentum = 0.
    for index in range(steps):
        state = model.step(state)
        target = ChainState(*exact_transition(initial.displacement, initial.velocity,
                                             model.stiffness, model.dt*(index+1)))
        errors.append(float(state_loss(state, target, scales)))
        maximum_energy = max(maximum_energy, float(((state.energy()-initial.energy())/initial.energy()).abs().max()))
        maximum_momentum = max(maximum_momentum, float((state.velocity.sum(-1)-initial.velocity.sum(-1)).abs().max()))
    return {'steps': steps, 'normalized_mse': errors[-1], 'mean_step_normalized_mse': sum(errors)/len(errors),
            'max_relative_energy_error': maximum_energy, 'max_momentum_error': maximum_momentum,
            'step_errors': errors}


def run(args):
    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir()
    shutil.copy2(__file__, out/'wave_operator.py')
    dt = .8
    train, validation = wave_data(601), wave_data(602, 24)
    scales = ((train[2]-train[0]-dt*train[1]).double().square().mean().sqrt(),
              (train[3]-train[1]).double().square().mean().sqrt())
    results, models = [], []
    for degree in (4, 8, 12, 16, 24):
        model = fit(train, degree, dt, scales)
        models.append(model)
        row = {'degree': degree, 'parameter_count': 2*degree,
               'train': metrics(model, train, scales), 'validation': metrics(model, validation, scales),
               'validation_rollout': metrics(model, validation, scales, 32)}
        results.append(row)
        print(json.dumps(row), flush=True)
    # Smallest validation model meeting the declared capability threshold.
    eligible = [i for i, row in enumerate(results) if row['validation_rollout']['normalized_mse'] <= 1e-6
                and row['validation_rollout']['max_relative_energy_error'] <= 1e-3]
    selected = eligible[0] if eligible else min(range(len(results)), key=lambda i: results[i]['validation_rollout']['normalized_mse'])
    model = models[selected]
    torch.save({'coefficients': model.coefficients, 'dt': dt, 'stiffness': model.stiffness}, out/'operator.pt')
    tests = {}
    for n, seed in ((32, 603), (64, 604), (128, 605)):
        data = wave_data(seed, 24, n)
        tests[str(n)] = {'one_step': metrics(model, data, scales), 'rollout': metrics(model, data, scales, 32)}
    report = {'status': 'completed', 'scope': __doc__, 'candidates': results, 'selected_degree': model.degree,
              'selection': 'Smallest validation degree with 32-step MSE <= 1e-6 and max relative energy error <= 1e-3; otherwise lowest 32-step validation MSE.',
              'test': tests, 'checkpoint_sha256': PhaseHistory._hash(out/'operator.pt'),
              'native_validated': False, 'persistent_state': ['displacement', 'velocity'],
              'test_scope': 'Unseen initial states and chain lengths; same unit masses, stiffness and dt.'}
    (out/'review.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'candidates'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    run(parser.parse_args())
