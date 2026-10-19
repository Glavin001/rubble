"""Frozen linear damping capability and teacher, not a production propagator.

The persisted state remains physical displacement/rotation and velocity. Bond
dashpots are a separate graph operator C, which need not commute with stiffness
K. Dense exponentials here only supply small-graph validation/training targets;
they are explicitly size-limited and do not qualify as a fast learned solver.
"""
import math

import torch

from .elastic_body_graph import ElasticBodyGraph
from .physical_transition import ChainState


class DampedBodyGraph:
    def __init__(self, body, bond_damping, mass_damping):
        if (bond_damping.shape != body.stiffness.shape
                or mass_damping.shape != body.mass_matrix.shape
                or not torch.isfinite(bond_damping).all() or not torch.isfinite(mass_damping).all()
                or (bond_damping < 0).any()):
            raise ValueError('finite nonnegative bond damping and chunk damping matrices required')
        if not torch.allclose(mass_damping, mass_damping.mT, atol=1e-14, rtol=1e-12):
            raise ValueError('chunk damping must be symmetric positive semidefinite')
        if (torch.linalg.eigvalsh(mass_damping) < 0).any():
            raise ValueError('chunk damping must be symmetric positive semidefinite')
        self.body, self.bond_damping, self.mass_damping = body, bond_damping, mass_damping

    @classmethod
    def from_native(cls, cluster):
        # Do not invent missing Rayleigh terms for older exports.
        body = ElasticBodyGraph.from_native(cluster)
        return cls(body, torch.tensor([b['damping'] for b in cluster['bonds']], dtype=torch.float64).reshape(-1, 6),
                   torch.tensor([c['mass_damping_matrix'] for c in cluster['chunks']], dtype=torch.float64))

    def physical_apply(self, velocity):
        """Return C v; the damping load on a chunk is -C v."""
        body = self.body
        rate = body.deformation(velocity)
        pair = torch.einsum('eij,...ei->...ej', body.rows, self.bond_damping*rate)
        v = velocity.reshape(*velocity.shape[:-1], body.n, 6)
        result = torch.einsum('nij,...nj->...ni', self.mass_damping, v)
        a, b = body.endpoints.T
        result.index_add_(-2, a, pair[..., :6])
        result.index_add_(-2, b, pair[..., 6:])
        return result.flatten(-2)

    def apply(self, weighted_velocity):
        return self.body.weighted_force(self.physical_apply(self.body.from_weighted(weighted_velocity)))

    def dissipated_power(self, velocity):
        rate = self.body.deformation(velocity)
        v = velocity.reshape(*velocity.shape[:-1], self.body.n, 6)
        return ((self.bond_damping*rate.square()).sum((-1, -2))
                + torch.einsum('...ni,nij,...nj->...', v, self.mass_damping, v))

    def response(self, state):
        result = self.body.response(state)
        damping = self.bond_damping*self.body.deformation(state.velocity)
        result.update(bond_damping_force=damping, bond_total_force=result['bond_force']+damping,
                      damping_nodal_force=-self.physical_apply(state.velocity),
                      dissipated_power=self.dissipated_power(state.velocity))
        return result


class DenseDampedReference:
    """All physical free DOFs retained, for small frozen validation graphs only.

    z = [omega * M^(1/2) x, M^(1/2) v]. Balancing displacement against velocity
    avoids a poorly scaled first-order generator at native SI stiffness values.
    Known constant force has an exact affine response, including zero modes.
    No trajectory-dependent information or opaque temporal state persists.
    """
    def __init__(self, graph, max_free_dof=256):
        if max_free_dof < 1 or max_free_dof > 256:
            raise ValueError('dense validation is limited to 256 free physical DOFs')
        self.graph, self.body = graph, graph.body
        self.free = (~self.body.fixed).flatten()
        self.dof = int(self.free.sum())
        if not 0 < self.dof <= max_free_dof:
            raise ValueError(f'dense validation permits 1..{max_free_dof} free DOFs, got {self.dof}')
        eye = torch.eye(self.body.n*6, dtype=self.body.stiffness.dtype, device=self.body.stiffness.device)[self.free]
        self.k = self.body.apply(eye)[:, self.free].T
        self.c = graph.apply(eye)[:, self.free].T
        self.omega = max(1., float(torch.linalg.eigvalsh(self.k).max().clamp_min(0).sqrt()))
        self.generator = self.make_generator(self.k, self.c)

    def make_generator(self, k, c):
        identity = torch.eye(self.dof, dtype=k.dtype, device=k.device)
        return torch.cat((torch.cat((torch.zeros_like(k), self.omega*identity), -1),
                          torch.cat((-k/self.omega, -c), -1)), -2)

    def encode(self, state):
        u, v = state.displacement, state.velocity
        if (u.shape != v.shape or u.shape[-1] != self.body.n*6
                or not torch.isfinite(u).all() or not torch.isfinite(v).all()):
            raise ValueError('finite physical displacement and velocity required')
        if (u[..., ~self.free] != 0).any() or (v[..., ~self.free] != 0).any():
            raise ValueError('fixed support state must be zero')
        return torch.cat((self.omega*self.body.to_weighted(u)[..., self.free],
                          self.body.to_weighted(v)[..., self.free]), -1)

    def decode(self, value):
        y, w = value.chunk(2, -1)
        u = value.new_zeros(*value.shape[:-1], self.body.n*6)
        v = torch.zeros_like(u)
        u[..., self.free], v[..., self.free] = y/self.omega, w
        return ChainState(self.body.from_weighted(u), self.body.from_weighted(v))

    def propagator(self, dt, *, constant_force=False):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('finite positive interval required')
        if not constant_force:
            return torch.linalg.matrix_exp(dt*self.generator), None
        n = self.dof
        augmented = self.k.new_zeros(3*n, 3*n)
        augmented[:2*n, :2*n] = self.generator
        augmented[n:2*n, 2*n:] = torch.eye(n, dtype=self.k.dtype, device=self.k.device)
        result = torch.linalg.matrix_exp(dt*augmented)
        return result[:2*n, :2*n], result[:2*n, 2*n:]

    def step(self, state, dt, constant_force=None):
        transition, forced = self.propagator(dt, constant_force=constant_force is not None)
        result = self.encode(state)@transition.T
        if constant_force is not None:
            if constant_force.shape != state.displacement.shape or not torch.isfinite(constant_force).all():
                raise ValueError('finite constant force must match state')
            force = self.body.weighted_force(constant_force)[..., self.free]
            result += force@forced.T
        return self.decode(result)
