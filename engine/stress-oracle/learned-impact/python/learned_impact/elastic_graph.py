"""Apply the learned wave operator to new scalar elastic graphs and supports.

This extends the physical-state probe beyond uniform chains. Mass and stiffness
are explicit inputs, and the same learned coefficient vector is reused unchanged.
The certified Gershgorin spectrum bound must fit its declared training interval;
unsupported inputs fail instead of silently clipping or dropping interactions.
Still linear scalar mechanics, without contact, damping, fracture or 3-D rotation.
"""
from dataclasses import dataclass

import torch

from .physical_transition import ChainState


@dataclass(frozen=True)
class ElasticGraph:
    endpoints: torch.Tensor
    stiffness: torch.Tensor
    mass: torch.Tensor
    fixed: torch.Tensor

    def validate(self):
        n, e = len(self.mass), len(self.stiffness)
        if self.endpoints.dtype != torch.long or self.endpoints.shape != (e, 2):
            raise ValueError('invalid elastic graph edges')
        if self.mass.ndim != 1 or self.stiffness.ndim != 1 or self.fixed.shape != (n,) or self.fixed.dtype != torch.bool:
            raise ValueError('invalid elastic graph fields')
        if n == 0 or not torch.isfinite(self.mass).all() or not torch.isfinite(self.stiffness).all() or (self.mass <= 0).any() or (self.stiffness <= 0).any():
            raise ValueError('positive finite mass and stiffness required')
        if e and ((self.endpoints < 0).any() or (self.endpoints >= n).any() or (self.endpoints[:, 0] == self.endpoints[:, 1]).any()):
            raise ValueError('invalid edge endpoint')

    def stiffness_apply(self, displacement):
        a, b = self.endpoints.T
        force = self.stiffness*(displacement[..., b]-displacement[..., a])
        result = torch.zeros_like(displacement)
        result.index_add_(-1, a, -force)
        result.index_add_(-1, b, force)
        return result

    def apply(self, weighted_displacement):
        root = self.mass.sqrt()
        displacement = weighted_displacement*(~self.fixed)/root
        return self.stiffness_apply(displacement)*(~self.fixed)/root

    def spectral_bound(self):
        a, b = self.endpoints.T
        rows = torch.zeros_like(self.mass)
        rows.index_add_(0, a, self.stiffness/self.mass[a])
        rows.index_add_(0, b, self.stiffness/self.mass[b])
        off = self.stiffness/(self.mass[a]*self.mass[b]).sqrt()
        rows.index_add_(0, a, off*(~self.fixed[b]))
        rows.index_add_(0, b, off*(~self.fixed[a]))
        return float((rows*(~self.fixed)).max())

    def energy(self, state):
        a, b = self.endpoints.T
        return (.5*(self.mass*state.velocity.square()).sum(-1)
                + .5*(self.stiffness*(state.displacement[..., b]-state.displacement[..., a]).square()).sum(-1))

    def response(self, state):
        a, b = self.endpoints.T
        extension = state.displacement[..., b]-state.displacement[..., a]
        gradient = self.stiffness_apply(state.displacement)
        return {'bond_force': self.stiffness*extension,
                'bond_stored_energy': .5*self.stiffness*extension.square(),
                'internal_nodal_force': -gradient,
                'support_reaction': gradient*self.fixed}


class ElasticGraphTransition:
    def __init__(self, learned, graph):
        graph.validate()
        self.graph, self.learned = graph, learned
        self.spectral_max = 4*learned.stiffness
        if graph.spectral_bound() > self.spectral_max*(1+1e-12):
            raise ValueError('graph spectrum bound exceeds the fitted operator domain')

    def basis(self, field):
        def scaled(x):
            return 2*self.graph.apply(x)/self.spectral_max-x
        older, previous = field, scaled(field)
        result = [previous+field]
        for index in range(2, self.learned.degree+1):
            current = 2*scaled(previous)-older
            result.append(current-((-1)**index)*field)
            older, previous = previous, current
        return torch.stack(result, -2)

    def force_displacement_basis(self, field):
        """(T_j(2L/b-1)-(-1)^j I)/L without an inverse or zero-mode division."""
        def scaled(x):
            return 2*self.graph.apply(x)/self.spectral_max-x
        older, previous = torch.zeros_like(field), 2/self.spectral_max*field
        result = [previous]
        for index in range(2, self.learned.degree+1):
            current = 2*scaled(previous)-older+4*((-1)**(index-1))/self.spectral_max*field
            result.append(current)
            older, previous = previous, current
        return torch.stack(result, -2)

    def step(self, state, constant_force=None):
        if state.displacement.shape != state.velocity.shape or state.displacement.shape[-1] != len(self.graph.mass):
            raise ValueError('graph state shape mismatch')
        if not torch.isfinite(state.displacement).all() or not torch.isfinite(state.velocity).all():
            raise ValueError('nonfinite physical state')
        if (state.displacement[..., self.graph.fixed] != 0).any() or (state.velocity[..., self.graph.fixed] != 0).any():
            raise ValueError('fixed supports must have zero displacement and velocity')
        root = self.graph.mass.sqrt()
        u, v = state.displacement*root, state.velocity*root
        pu, pv = self.basis(u), self.basis(v)
        a, b = self.learned.coefficients.chunk(2)
        a, b = a.to(u), b.to(u)
        cu, cv = (pu*a[:, None]).sum(-2), (pv*a[:, None]).sum(-2)
        su, sv = (pu*b[:, None]).sum(-2), (pv*b[:, None]).sum(-2)
        next_u = u+self.learned.dt*v+cu+sv
        next_v = v+cv-self.graph.apply(self.learned.dt*u+su)
        if constant_force is not None:
            if constant_force.shape != state.displacement.shape or not torch.isfinite(constant_force).all():
                raise ValueError('finite constant force must match physical state shape')
            force = constant_force*(~self.graph.fixed)/root
            next_u = next_u-(self.force_displacement_basis(force)*a[:, None]).sum(-2)
            next_v = next_v+self.learned.dt*force+(self.basis(force)*b[:, None]).sum(-2)
        return ChainState(next_u/root, next_v/root)


def exact_graph_transition(graph, state, dt, constant_force=None):
    """Dense modal teacher for tests/data only; never called by learned inference."""
    n = len(graph.mass)
    matrix = graph.apply(torch.eye(n, dtype=graph.mass.dtype)).T
    values, vectors = torch.linalg.eigh(matrix)
    omega = values.clamp_min(0).sqrt()
    root = graph.mass.sqrt()
    u, v = (state.displacement*root)@vectors, (state.velocity*root)@vectors
    c, s = (omega*dt).cos(), dt*torch.sinc(omega*dt/torch.pi)
    next_u, next_v = u*c+v*s, v*c-u*omega*(omega*dt).sin()
    if constant_force is not None:
        force = (constant_force*(~graph.fixed)/root)@vectors
        r = .5*dt*dt*torch.sinc(omega*dt/(2*torch.pi)).square()
        next_u, next_v = next_u+r*force, next_v+s*force
    return ChainState((next_u@vectors.T)/root, (next_v@vectors.T)/root)
