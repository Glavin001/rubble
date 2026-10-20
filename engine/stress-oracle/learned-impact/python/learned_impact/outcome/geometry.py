"""Quaternion and rotation helpers (w-first unit quaternions, body → world)."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def quat_rotate(q: Tensor, v: Tensor) -> Tensor:
    """Rotate `v` [..., 3] by unit quaternions `q` [..., 4] (body → world)."""
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def quat_rotate_inv(q: Tensor, v: Tensor) -> Tensor:
    """World → body."""
    return quat_rotate(torch.cat([q[..., :1], -q[..., 1:]], dim=-1), v)


def quat_to_matrix(q: Tensor) -> Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def rotvec_to_matrix(r: Tensor) -> Tensor:
    """Rodrigues; exact at small angles through the series fallback."""
    th = torch.linalg.norm(r, dim=-1, keepdim=True)
    small = th < 1e-6
    ths = torch.where(small, torch.ones_like(th), th)
    k = r / ths
    kx = torch.zeros(r.shape[:-1] + (3, 3), dtype=r.dtype, device=r.device)
    kx[..., 0, 1], kx[..., 0, 2] = -k[..., 2], k[..., 1]
    kx[..., 1, 0], kx[..., 1, 2] = k[..., 2], -k[..., 0]
    kx[..., 2, 0], kx[..., 2, 1] = -k[..., 1], k[..., 0]
    s = torch.sin(ths)[..., None]
    c = (1 - torch.cos(ths))[..., None]
    eye = torch.eye(3, dtype=r.dtype, device=r.device).expand_as(kx)
    full = eye + s * kx + c * (kx @ kx)
    rx = torch.zeros_like(kx)
    rx[..., 0, 1], rx[..., 0, 2] = -r[..., 2], r[..., 1]
    rx[..., 1, 0], rx[..., 1, 2] = r[..., 2], -r[..., 0]
    rx[..., 2, 0], rx[..., 2, 1] = -r[..., 1], r[..., 0]
    approx = eye + rx
    return torch.where(small[..., None], approx, full)


def world_inertia(q: Tensor, principal: Tensor) -> Tensor:
    """I_world = R diag(I) Rᵀ."""
    r = quat_to_matrix(q)
    return r @ torch.diag_embed(principal) @ r.transpose(-1, -2)


def random_quat(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    return q


def quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_to_matrix_np(q: np.ndarray) -> np.ndarray:
    return quat_to_matrix(torch.from_numpy(np.asarray(q, dtype=np.float64))).numpy()
