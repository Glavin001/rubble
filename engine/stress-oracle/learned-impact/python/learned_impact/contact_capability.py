"""Small 1-D impact capability probe: physical-state feedback versus frozen loads.

A projectile starts just touching a supported elastic bar. Contact is a unilateral
linear spring, without friction, damping or fracture. The C/S coefficients are the
same independently fitted elastic operator. This is not the native destruction
contact law, a trained contact model, or a runtime-speed benchmark.
"""
import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch

from .elastic_body_graph import ElasticBodyGraph, ElasticBodyTransition
from .physical_transition import ChainState
from .wave_operator import CoupledWaveOperator


@dataclass(frozen=True)
class ImpactState:
    mechanics: ChainState
    touching: bool


class ElasticImpact:
    def __init__(self, learned, masses, stiffness=100., contact_stiffness=50.):
        self.learned, self.masses = learned, np.asarray(masses, dtype=float)
        self.n, self.stiffness, self.contact_stiffness = len(masses), stiffness, contact_stiffness
        if self.n < 2 or not np.isfinite(self.masses).all() or (self.masses <= 0).any() or stiffness <= 0 or contact_stiffness <= 0:
            raise ValueError('positive finite impact properties required')
        self.free_graph, self.contact_graph = self.graph(False), self.graph(True)
        # Declared event-resolution restriction for this narrow experiment. It is
        # not silently imposed on a caller's requested interval or production.
        self.event_resolution_dt = .5/math.sqrt(self.contact_graph.spectral_bound())

    def graph(self, contact):
        pairs = [(i, i+1) for i in range(1, self.n-1)]
        k = [self.stiffness]*len(pairs)
        if contact:
            pairs.append((0, 1))
            k.append(self.contact_stiffness)
        e = len(pairs)
        fixed = torch.ones(self.n, 6, dtype=torch.bool)
        fixed[:, 0] = False
        # Two masses are the isolated collision test; longer bars have a fixed end.
        if self.n > 2:
            fixed[-1, 0] = True
        return ElasticBodyGraph(torch.tensor(pairs, dtype=torch.long).reshape(e, 2),
            torch.eye(3, dtype=torch.float64).expand(e, -1, -1),
            torch.zeros(e, 3, dtype=torch.float64), torch.zeros(e, 3, dtype=torch.float64),
            torch.tensor(k, dtype=torch.float64)[:, None].expand(-1, 6),
            torch.diag_embed(torch.tensor(self.masses, dtype=torch.float64)[:, None].expand(-1, 6)), fixed)

    def initial(self, speed=1.):
        u, v = torch.zeros(self.n*6, dtype=torch.float64), torch.zeros(self.n*6, dtype=torch.float64)
        v[0] = speed
        return ImpactState(ChainState(u, v), speed > 0)

    @staticmethod
    def penetration(state):
        return float(state.displacement[0]-state.displacement[6])

    def propagate(self, state, touching, dt, constant_force=None):
        graph = self.contact_graph if touching else self.free_graph
        return ElasticBodyTransition(self.learned, graph, dt).step(state, constant_force)

    def step(self, initial, dt, *, frozen=False):
        if dt <= 0 or not math.isfinite(dt) or dt > self.event_resolution_dt*(1+1e-12):
            raise ValueError('requested interval exceeds declared contact-event resolution')
        state, touching = initial.mechanics, initial.touching
        if frozen:
            # Controlled bad coupling: correct CURRENT force, held for the interval.
            force = torch.zeros_like(state.displacement)
            f = self.contact_stiffness*max(self.penetration(state), 0.)
            force[0], force[6] = -f, f
            result = self.propagate(state, False, dt, force)
            return ImpactState(result, self.penetration(result) > 0), []
        remaining, elapsed, events = dt, 0., []
        for _ in range(8):
            proposed = self.propagate(state, touching, remaining)
            end = self.penetration(proposed)
            crossing = end < 0 if touching else end > 0
            if not crossing:
                return ImpactState(proposed, touching), events
            # Locate the change of contact mode using this world's own trajectory.
            lo, hi = 0., remaining
            for _ in range(40):
                mid = .5*(lo+hi)
                value = self.penetration(self.propagate(state, touching, mid))
                before = value >= 0 if touching else value <= 0
                if before:
                    lo = mid
                else:
                    hi = mid
            crossing_dt = .5*(lo+hi)
            state = self.propagate(state, touching, crossing_dt)
            elapsed += crossing_dt
            remaining -= crossing_dt
            touching = not touching
            events.append({'time': elapsed, 'touching': touching})
            if remaining <= dt*1e-12:
                return ImpactState(state, touching), events
        raise ValueError('too many contact events in the declared interval')

    def energy(self, state):
        return float(self.free_graph.energy(state)+.5*self.contact_stiffness*max(self.penetration(state), 0.)**2)

    def reference(self, speed, dt, intervals, subdivisions):
        """Independent RK4 force-law reference; recomputes contact at each stage."""
        x, v = np.zeros(self.n), np.zeros(self.n)
        v[0] = speed
        def derivative(y):
            u, velocity = y[:self.n], y[self.n:]
            force = np.zeros(self.n)
            bond = self.stiffness*np.diff(u[1:])
            force[1:-1] += bond
            force[2:] -= bond
            contact = self.contact_stiffness*max(u[0]-u[1], 0.)
            force[0] -= contact
            force[1] += contact
            acceleration = force/self.masses
            if self.n > 2:
                acceleration[-1] = 0.
            return np.concatenate((velocity, acceleration))
        y, output = np.concatenate((x, v)), []
        h = dt/subdivisions
        for _ in range(intervals):
            for _ in range(subdivisions):
                k1 = derivative(y)
                k2 = derivative(y+.5*h*k1)
                k3 = derivative(y+.5*h*k2)
                k4 = derivative(y+h*k3)
                y += h/6*(k1+2*k2+2*k3+k4)
            output.append(y.copy())
        return np.asarray(output)


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    ckpt = torch.load(args.weights, weights_only=True)
    learned = CoupledWaveOperator(ckpt['coefficients'], ckpt['dt'], ckpt['stiffness'])
    rows = []
    for mass, contact_k in ((.25, 25.), (.5, 50.), (2., 100.)):
        system = ElasticImpact(learned, [mass]+[1.]*8, contact_stiffness=contact_k)
        dt = .9*system.event_resolution_dt
        intervals = math.ceil(2./dt)
        reference = system.reference(1., dt, intervals, 128)
        finer = system.reference(1., dt, intervals, 256)
        for frozen in (False, True):
            state = system.initial()
            initial_energy = system.energy(state.mechanics)
            trajectory, events, max_drift = [], [], 0.
            for step in range(intervals):
                state, changes = system.step(state, dt, frozen=frozen)
                events.extend([{**event, 'time': event['time']+step*dt} for event in changes])
                u = state.mechanics.displacement.reshape(system.n, 6)[:, 0]
                v = state.mechanics.velocity.reshape(system.n, 6)[:, 0]
                trajectory.append(torch.cat((u, v)).numpy())
                max_drift = max(max_drift, abs(system.energy(state.mechanics)/initial_energy-1))
            trajectory = np.asarray(trajectory)
            delta = trajectory-finer
            row = {'projectile_mass': mass, 'contact_stiffness': contact_k,
                   'coupling': 'frozen_current_force' if frozen else 'evolving_contact_edge',
                   'intervals': intervals, 'dt_s': dt, 'duration_s': intervals*dt,
                   'reference_convergence_max_abs': float(np.abs(reference-finer).max()),
                   'projectile_position_rmse_m': float(np.sqrt(np.mean(delta[:, 0]**2))),
                   'projectile_velocity_rmse_m_s': float(np.sqrt(np.mean(delta[:, system.n]**2))),
                   'reference_final_projectile_velocity_m_s': float(finer[-1, system.n]),
                   'model_final_projectile_velocity_m_s': float(trajectory[-1, system.n]),
                   'max_relative_energy_drift': max_drift, 'contact_events': events}
            rows.append(row)
            print(json.dumps(row), flush=True)
    with Path(args.out).open('x') as file:
        json.dump({'status': 'completed', 'scope': __doc__, 'results': rows}, file, indent=2, allow_nan=False)
        file.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--out', required=True)
    run(parser.parse_args())
