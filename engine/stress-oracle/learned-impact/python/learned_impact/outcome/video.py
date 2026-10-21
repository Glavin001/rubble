"""Side-by-side videos of closed-loop runs: the oracle reference against the learned model
(and optionally QS or the punch), same scene, same projectile, same camera.

Input: one pose track per run (`Track`, read from the harness's `--poses` dump by
`load_track`). Output: an mp4 (H.264 via ffmpeg) with one panel per run. Chunks are boxes
coloured by fragment (the largest fragment grey; the others by a stable colour of their
lowest chunk key, so the same fragment has the same colour in every frame), the impactor is
dark red, and each panel shows time, impactor speed, fragment count and broken bonds.

    python -m learned_impact.outcome.video --track oracle=oracle.poses \
        --track "learned (W32, pilot)"=model.poses --out free_wall.mp4 --title "free wall, 40 m/s"
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import quat_to_matrix_np

_CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
_FACES = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]


@dataclass
class Track:
    """Poses of one run. `chunk_*` arrays are [F, N, …] over frame boundaries."""

    name: str
    t: np.ndarray  # [F]
    half_extents: np.ndarray  # [N, 3]
    chunk_key: np.ndarray  # [N]
    chunk_pos: np.ndarray  # [F, N, 3]
    chunk_quat: np.ndarray  # [F, N, 4] wxyz
    chunk_cluster: np.ndarray  # [F, N]
    imp_pos: np.ndarray  # [F, I, 3]
    imp_quat: np.ndarray  # [F, I, 4]
    imp_vel: np.ndarray  # [F, I, 3]
    imp_radius: np.ndarray  # [I] (spheres) or half extents for boxes, see imp_half
    imp_half: np.ndarray | None = None  # [I, 3] for box impactors
    broken: np.ndarray | None = None  # [F] broken bonds so far
    notes: dict = field(default_factory=dict)


def load_track(path: Path | str, name: str) -> Track:
    """Read a pose dump (JSONL: a header line with the static data, then one line per frame).
    Field names follow P0's `--poses` output; see PROVIDER_PROTOCOL.md on P0's branch."""
    lines = Path(path).read_text().splitlines()
    head = json.loads(lines[0])
    frames = [json.loads(x) for x in lines[1:] if x.strip()]
    ck = np.array(head["chunk_keys"], dtype=np.int64)
    he = np.array(head["chunk_half_extents"], float)
    imps = head.get("impactors", [])
    imp_r = np.array([i.get("radius", 0.0) for i in imps], float)
    imp_h = np.array([i.get("half_extents", [0, 0, 0]) for i in imps], float) if imps else None

    def stack(key, default_shape):
        out = []
        for f in frames:
            v = f.get(key)
            out.append(np.asarray(v, float) if v is not None else np.zeros(default_shape))
        return np.stack(out)

    n, ni = len(ck), len(imps)
    return Track(
        name=name, t=np.array([f["t"] for f in frames], float), half_extents=he, chunk_key=ck,
        chunk_pos=stack("chunk_pos", (n, 3)), chunk_quat=stack("chunk_quat", (n, 4)),
        chunk_cluster=stack("chunk_cluster", (n,)).astype(np.int64),
        imp_pos=stack("impactor_pos", (ni, 3)), imp_quat=stack("impactor_quat", (ni, 4)),
        imp_vel=stack("impactor_vel", (ni, 3)), imp_radius=imp_r,
        imp_half=imp_h if imp_h is not None and np.any(imp_h) else None,
        broken=np.array([f.get("broken_bonds", np.nan) for f in frames], float),
        notes={"scene": head.get("scene"), "variant": head.get("variant")})


def _palette(n: int) -> np.ndarray:
    import matplotlib

    cmap = matplotlib.colormaps["tab20"]
    return np.array([cmap(i % 20)[:3] for i in range(n)])


def _fragment_colours(track: Track, f: int) -> np.ndarray:
    """Grey for the largest fragment; others coloured by their lowest chunk key."""
    cl = track.chunk_cluster[f]
    ids, inv, counts = np.unique(cl, return_inverse=True, return_counts=True)
    big = int(np.argmax(counts))
    pal = _palette(20)
    cols = np.empty((len(cl), 3))
    for j in range(len(ids)):
        sel = inv == j
        if j == big:
            cols[sel] = (0.72, 0.72, 0.74)
        else:
            k = int(track.chunk_key[sel].min())
            cols[sel] = pal[(k * 2654435761) % 20]
    return cols


def _boxes(pos, quat, half):
    r = quat_to_matrix_np(quat)  # [N,3,3]
    corners = np.einsum("nij,nkj->nki", r, _CORNERS[None] * half[:, None, :]) + pos[:, None]
    return corners  # [N, 8, 3]


def _shade(polys: np.ndarray, base: np.ndarray, light=np.array([0.3, -0.5, 0.8])) -> np.ndarray:
    light = light / np.linalg.norm(light)
    n = np.cross(polys[:, 1] - polys[:, 0], polys[:, 2] - polys[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    lam = 0.45 + 0.55 * np.abs(n @ light)
    return np.clip(base * lam[:, None], 0, 1)


def scene_bounds(tracks: list[Track], margin: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """A fixed box: the structure at t0 plus where most of the motion goes (98th percentile,
    so one far-flung chunk doesn't shrink the scene)."""
    p0 = np.concatenate([t.chunk_pos[0] for t in tracks])
    lo, hi = p0.min(0), p0.max(0)
    allp = np.concatenate([t.chunk_pos.reshape(-1, 3) for t in tracks]
                          + [t.imp_pos.reshape(-1, 3) for t in tracks if t.imp_pos.size])
    lo = np.minimum(lo, np.percentile(allp, 1, axis=0))
    hi = np.maximum(hi, np.percentile(allp, 99, axis=0))
    c, h = 0.5 * (lo + hi), 0.5 * (hi - lo).max() + margin
    return c - h, c + h


def render(tracks: list[Track], out: Path | str, title: str = "", fps: int = 30,
           slowmo: float = 2.0, elev: float = 18.0, azim: float = -60.0, dpi: int = 110,
           width_per_panel: float = 5.0) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out = Path(out)
    lo, hi = scene_bounds(tracks)
    t_end = min(t.t[-1] for t in tracks)
    n_video = max(2, int(round(t_end * slowmo * fps)) + 1)
    times = np.linspace(0.0, t_end, n_video)
    tmp = Path(tempfile.mkdtemp())
    fig = plt.figure(figsize=(width_per_panel * len(tracks), width_per_panel * 0.9), dpi=dpi)
    for vi, tv in enumerate(times):
        fig.clf()
        if title:
            fig.suptitle(title, fontsize=12)
        for pi, tr in enumerate(tracks):
            f = int(np.clip(np.searchsorted(tr.t, tv, side="right") - 1, 0, len(tr.t) - 1))
            ax = fig.add_subplot(1, len(tracks), pi + 1, projection="3d")
            corners = _boxes(tr.chunk_pos[f], tr.chunk_quat[f], tr.half_extents)
            polys = corners[:, _FACES].reshape(-1, 4, 3)
            base = np.repeat(_fragment_colours(tr, f), 6, axis=0)
            pc = Poly3DCollection(polys, facecolors=_shade(polys, base), edgecolors=(0, 0, 0, 0.15),
                                  linewidths=0.2)
            ax.add_collection3d(pc)
            for i in range(tr.imp_pos.shape[1]):
                if tr.imp_half is not None:
                    ic = _boxes(tr.imp_pos[f, i:i + 1], tr.imp_quat[f, i:i + 1], tr.imp_half[i:i + 1])
                    ip = ic[:, _FACES].reshape(-1, 4, 3)
                    ax.add_collection3d(Poly3DCollection(
                        ip, facecolors=_shade(ip, np.tile([[0.55, 0.08, 0.08]], (len(ip), 1)))))
                else:
                    u, v = np.mgrid[0:2 * np.pi:16j, 0:np.pi:9j]
                    rr = tr.imp_radius[i]
                    c = tr.imp_pos[f, i]
                    ax.plot_surface(c[0] + rr * np.cos(u) * np.sin(v), c[1] + rr * np.sin(u) * np.sin(v),
                                    c[2] + rr * np.cos(v), color=(0.55, 0.08, 0.08), shade=True,
                                    linewidth=0)
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            n_frag = len(np.unique(tr.chunk_cluster[f]))
            speed = (float(np.linalg.norm(tr.imp_vel[f, 0])) if tr.imp_vel.shape[1] else float("nan"))
            broken = tr.broken[f] if tr.broken is not None else np.nan
            ax.set_title(f"{tr.name}\n t = {tr.t[f] * 1000:5.1f} ms   ram {speed:5.1f} m/s\n"
                         f"fragments {n_frag}" + ("" if np.isnan(broken) else f"   broken {int(broken)}"),
                         fontsize=9, family="monospace")
        fig.savefig(tmp / f"f{vi:05d}.png")
    plt.close(fig)
    ff = shutil.which("ffmpeg")
    if ff is None:
        raise RuntimeError(f"ffmpeg not found; frames are in {tmp}")
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                    str(tmp / "f%05d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)], check=True)
    shutil.rmtree(tmp)
    return out


def synthetic_tracks(seed: int = 0, n_frames: int = 30) -> list[Track]:
    """Two fake tracks of a 10×1×6 wall hit by a sphere (renderer test only)."""
    rng = np.random.default_rng(seed)
    grid = np.array([[i * 0.1, 0.0, k * 0.1] for k in range(6) for i in range(10)])
    n = len(grid)
    out = []
    for name, spread in (("oracle", 1.0), ("learned", 0.7)):
        t = np.arange(n_frames) / 60.0
        pos = np.repeat(grid[None], n_frames, 0)
        cl = np.zeros((n_frames, n), np.int64)
        hit = np.linalg.norm(grid - grid[25], axis=1) < 0.15
        v = np.where(hit[:, None], [0.0, 3.0 * spread, 0.0], 0.0) + rng.normal(0, 0.05, (n, 3)) * hit[:, None]
        for f in range(n_frames):
            if f >= 5:
                pos[f] = grid + v * (t[f] - t[5])
                cl[f] = np.where(hit, 1 + np.arange(n), 0)
        quat = np.tile([1.0, 0, 0, 0], (n_frames, n, 1))
        ip = np.array([[grid[25] + [0, -0.3 + 4.0 * tt, 0]] for tt in t])
        out.append(Track(name=name, t=t, half_extents=np.full((n, 3), 0.05), chunk_key=np.arange(n),
                         chunk_pos=pos, chunk_quat=quat, chunk_cluster=cl, imp_pos=ip,
                         imp_quat=np.tile([1.0, 0, 0, 0], (n_frames, 1, 1)),
                         imp_vel=np.tile([[0, 4.0, 0]], (n_frames, 1, 1)), imp_radius=np.array([0.1])))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--track", action="append", required=True, help="name=path to a pose dump")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--slowmo", type=float, default=2.0)
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    a = ap.parse_args(argv)
    tracks = [load_track(p, n) for n, p in (x.split("=", 1) for x in a.track)]
    print(render(tracks, a.out, a.title, slowmo=a.slowmo, elev=a.elev, azim=a.azim))


if __name__ == "__main__":
    main()
