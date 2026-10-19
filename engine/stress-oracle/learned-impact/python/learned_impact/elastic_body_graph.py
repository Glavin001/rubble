"""Six-DOF frozen elastic graph probe using native geometry and full inertia.

Only displacement/rotation and their velocities persist. Forces and stored energy
come from the same bond kinematics. Sparse inference uses no dense global matrix
or eigenvectors. This is linear small-motion mechanics, NOT corotational fracture,
damping, contact coupling, or the engine's floating-frame transfer.
"""
import math

import torch

from .elastic_graph import ElasticGraphTransition
from .physical_transition import ChainState


class ElasticBodyGraph:
    def __init__(self, endpoints, axes, ra, rb, stiffness, mass_matrix, fixed):
        n, e = len(mass_matrix), len(stiffness)
        if n < 1 or mass_matrix.shape != (n, 6, 6) or fixed.shape != (n, 6) or fixed.dtype != torch.bool:
            raise ValueError('invalid chunk inertia/support fields')
        if (endpoints.dtype != torch.long or endpoints.shape != (e, 2) or axes.shape != (e, 3, 3)
                or ra.shape != (e, 3) or rb.shape != (e, 3) or stiffness.shape != (e, 6)):
            raise ValueError('invalid six-DOF bond fields')
        if e and ((endpoints < 0).any() or (endpoints >= n).any() or (endpoints[:, 0] == endpoints[:, 1]).any()):
            raise ValueError('invalid bond endpoints')
        if any(not torch.isfinite(x).all() for x in (axes, ra, rb, stiffness, mass_matrix)) or (stiffness <= 0).any():
            raise ValueError('finite geometry, mass and positive stiffness required')
        if not torch.allclose(mass_matrix, mass_matrix.mT, atol=1e-14, rtol=1e-12):
            raise ValueError('mass/inertia must be symmetric')
        if not torch.allclose(axes@axes.mT, torch.eye(3, dtype=axes.dtype, device=axes.device).expand(e, -1, -1), atol=1e-12, rtol=1e-12):
            raise ValueError('bond frame must be orthonormal')
        torch.linalg.cholesky(mass_matrix)  # Validate physical SPD even on fixed chunks.
        self.endpoints, self.stiffness = endpoints, stiffness
        self.mass_matrix, self.fixed, self.n = mass_matrix, fixed, n
        self.free = ~fixed
        # Restricted mass factor; held components are identity placeholders.
        restricted = mass_matrix*self.free[:, :, None]*self.free[:, None, :]+torch.diag_embed(fixed.to(mass_matrix))
        self.root = torch.linalg.cholesky(restricted)
        self.inverse_root = torch.linalg.solve_triangular(self.root, torch.eye(6, dtype=mass_matrix.dtype, device=mass_matrix.device).expand(n, -1, -1), upper=False)
        z = torch.zeros_like(axes)
        lin = torch.cat((-axes, -torch.linalg.cross(ra[:, None, :].expand_as(axes), axes),
                         axes, torch.linalg.cross(rb[:, None, :].expand_as(axes), axes)), -1)
        rot = torch.cat((z, -axes, z, axes), -1)
        self.rows = torch.cat((lin, rot), -2)  # E x local6 x (a6,b6)

    @classmethod
    def from_native(cls, cluster):
        chunks, bonds = cluster['chunks'], cluster['bonds']
        def field(rows, name, tail, dtype=torch.float64):
            return torch.tensor([row[name] for row in rows], dtype=dtype).reshape(len(rows), *tail)
        return cls(field(bonds, 'endpoints', (2,), torch.long), field(bonds, 'axes', (3, 3)),
                   field(bonds, 'ra', (3,)), field(bonds, 'rb', (3,)), field(bonds, 'stiffness', (6,)),
                   field(chunks, 'mass_matrix', (6, 6)), field(chunks, 'fixed', (6,), torch.bool))

    def to_weighted(self, value):
        value = value.reshape(*value.shape[:-1], self.n, 6)*self.free
        return torch.einsum('nji,...nj->...ni', self.root, value).flatten(-2)

    def from_weighted(self, value):
        value = value.reshape(*value.shape[:-1], self.n, 6)
        return (torch.einsum('nji,...nj->...ni', self.inverse_root, value)*self.free).flatten(-2)

    def weighted_force(self, value):
        value = value.reshape(*value.shape[:-1], self.n, 6)*self.free
        return (torch.einsum('nij,...nj->...ni', self.inverse_root, value)*self.free).flatten(-2)

    def deformation(self, displacement):
        x = displacement.reshape(*displacement.shape[:-1], self.n, 6)
        a, b = self.endpoints.T
        pair = torch.cat((x[..., a, :], x[..., b, :]), -1)
        return torch.einsum('eij,...ej->...ei', self.rows, pair)

    def stiffness_apply(self, displacement):
        q = self.stiffness*self.deformation(displacement)
        pair = torch.einsum('eij,...ei->...ej', self.rows, q)
        result = torch.zeros(*displacement.shape[:-1], self.n, 6, dtype=displacement.dtype, device=displacement.device)
        a, b = self.endpoints.T
        result.index_add_(-2, a, pair[..., :6])
        result.index_add_(-2, b, pair[..., 6:])
        return result.flatten(-2)

    def apply(self, value):
        return self.weighted_force(self.stiffness_apply(self.from_weighted(value)))

    def spectral_bound(self):
        # Sum of absolute element contributions bounds the assembled row sums,
        # including full rotational inertia and all cross-component couplings.
        a, b = self.endpoints.T
        inverse = self.inverse_root.mT*self.free[:, :, None]
        rows = torch.cat((self.rows[..., :6]@inverse[a], self.rows[..., 6:]@inverse[b]), -1).abs()
        sums = (self.stiffness[..., None]*rows*rows.sum(-1, keepdim=True)).sum(-2)
        result = torch.zeros(self.n, 6, dtype=rows.dtype, device=rows.device)
        result.index_add_(0, a, sums[..., :6])
        result.index_add_(0, b, sums[..., 6:])
        return float(result.max())

    def energy(self, state):
        v = state.velocity.reshape(*state.velocity.shape[:-1], self.n, 6)
        kinetic = .5*torch.einsum('...ni,nij,...nj->...', v, self.mass_matrix, v)
        elastic = .5*(self.stiffness*self.deformation(state.displacement).square()).sum((-1, -2))
        return kinetic+elastic

    def response(self, state):
        d = self.deformation(state.displacement)
        return {'deformation': d, 'bond_force': self.stiffness*d,
                'bond_stored_energy': .5*(self.stiffness*d.square()).sum(-1),
                'internal_nodal_force': -self.stiffness_apply(state.displacement)}


class ElasticBodyTransition(ElasticGraphTransition):
    def __init__(self, learned, graph, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('finite positive physical interval required')
        self.learned, self.body, self.dt = learned, graph, dt
        self.scale = dt/learned.dt
        self.spectral_max = 4*learned.stiffness
        if graph.spectral_bound()*self.scale**2 > self.spectral_max*(1+1e-12):
            raise ValueError('graph spectrum and interval exceed the fitted operator domain')
        # Use the identical normalized C/S polynomial code with dimensionless time.
        self.graph = self

    def apply(self, value):
        return self.scale**2*self.body.apply(value)

    def step(self, state, constant_force=None):
        u, v = state.displacement, state.velocity
        if u.shape != v.shape or u.shape[-1] != self.body.n*6 or not torch.isfinite(u).all() or not torch.isfinite(v).all():
            raise ValueError('finite six-DOF state required')
        fixed = self.body.fixed.flatten()
        if (u[..., fixed] != 0).any() or (v[..., fixed] != 0).any():
            raise ValueError('fixed supports must have zero displacement and velocity')
        y, w = self.body.to_weighted(u), self.scale*self.body.to_weighted(v)
        py, pw = self.basis(y), self.basis(w)
        a, b = [c.to(y) for c in self.learned.coefficients.chunk(2)]
        ny = y+self.learned.dt*w+(py*a[:, None]+pw*b[:, None]).sum(-2)
        nw = w+(pw*a[:, None]).sum(-2)-self.apply(self.learned.dt*y+(py*b[:, None]).sum(-2))
        if constant_force is not None:
            if constant_force.shape != u.shape or not torch.isfinite(constant_force).all():
                raise ValueError('finite constant force must match physical state shape')
            f = self.scale**2*self.body.weighted_force(constant_force)
            ny -= (self.force_displacement_basis(f)*a[:, None]).sum(-2)
            nw += self.learned.dt*f+(self.basis(f)*b[:, None]).sum(-2)
        return ChainState(self.body.from_weighted(ny), self.body.from_weighted(nw)/self.scale)


def exact_body_transition(graph, state, dt, constant_force=None):
    """Dense independent modal teacher for small validation graphs only."""
    matrix = graph.apply(torch.eye(graph.n*6, dtype=state.displacement.dtype)).T
    values, vectors = torch.linalg.eigh(matrix)
    if float(values.min()) < -1e-10*max(1., float(values.max())):
        raise ValueError('non-positive elastic operator')
    omega = values.clamp_min(0).sqrt()
    y, w = graph.to_weighted(state.displacement)@vectors, graph.to_weighted(state.velocity)@vectors
    c, s = (omega*dt).cos(), dt*torch.sinc(omega*dt/torch.pi)
    ny, nw = y*c+w*s, w*c-y*omega*(omega*dt).sin()
    if constant_force is not None:
        f = graph.weighted_force(constant_force)@vectors
        ny += .5*dt*dt*torch.sinc(omega*dt/(2*torch.pi)).square()*f
        nw += s*f
    return ChainState(graph.from_weighted(ny@vectors.T), graph.from_weighted(nw@vectors.T))
