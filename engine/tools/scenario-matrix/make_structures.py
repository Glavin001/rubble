#!/usr/bin/env python3
"""The scenario matrix's node-beam structures (no wheels, no drive), as .truck files.

    python3 engine/tools/scenario-matrix/make_structures.py   (writes scenes/*.truck)

- barrier.truck: a steel lattice barrier, 6 m wide, 1.2 m high, 0.4 m deep: a box truss of
  13 x 4 x 2 nodes at 0.5 m (x), 0.4 m (y) and 0.4 m (z), every edge and every face's two
  diagonals a beam, its bottom row anchored (`fixes`), both large faces covered with
  collision triangles (`cab`) so a car's nodes hit it and its nodes hit the car. Beams
  yield (deform) and break well under a 50 km/h car's load: it bends when hit.
- truss.truck: a bridge truss (Pratt) on two piers: 10 m span of 1 m panels, 1.5 m deep,
  two planes 1.5 m apart joined by cross beams and the deck's diagonals, the piers' feet
  and the deck's ends anchored, two 3 t loads at mid-span: it sags and yields under its
  load alone (row 6: a structure under load, no vehicle).

The files are committed; this script is how they were made (re-run after an edit).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "scenes")


def fmt(v):
    return f"{v:.3f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))


class Truck:
    def __init__(self, title):
        self.title = title
        self.nodes = []  # (x, y, z, options, load)
        self.beams = set()
        self.tris = []
        self.fixes = []

    def node(self, x, y, z, opt="n", load=None):
        self.nodes.append((x, y, z, opt, load))
        return len(self.nodes) - 1

    def beam(self, a, b):
        if a != b:
            self.beams.add((min(a, b), max(a, b)))

    def text(self, dry_mass, minimass, beam_defaults):
        out = [self.title, "", "globals", f"{fmt(dry_mass)}, 0, tracks/semi", "",
               "minimass", fmt(minimass), "",
               "set_beam_defaults " + ", ".join(fmt(x) for x in beam_defaults), "",
               "nodes"]
        for i, (x, y, z, opt, load) in enumerate(self.nodes):
            line = f"{i}, {fmt(x)}, {fmt(y)}, {fmt(z)}, {opt}"
            if load is not None:
                line += f", {fmt(load)}"
            out.append(line)
        out += ["", "beams"]
        out += [f"{a}, {b}" for a, b in sorted(self.beams)]
        if self.tris:
            out += ["", "submesh", "cab"]
            out += [f"{a}, {b}, {c}, c" for a, b, c in self.tris]
        if self.fixes:
            out += ["", "fixes"]
            out += [str(n) for n in self.fixes]
        out += ["", "end", ""]
        return "\n".join(out)


def barrier():
    t = Truck("scenario matrix: steel lattice barrier (node-beam structure, no wheels)")
    nx, ny, nz = 13, 4, 2
    dx, dy, dz = 0.5, 0.4, 0.4
    idx = {}
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                idx[i, j, k] = t.node(i * dx - 3.0, j * dy, k * dz)
                if j == 0:
                    t.fixes.append(idx[i, j, k])
    for (i, j, k), a in idx.items():
        # Edges and the diagonals of every face of every cell.
        for di, dj, dk in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0),
                           (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]:
            b = idx.get((i + di, j + dj, k + dk))
            if b is not None:
                t.beam(a, b)
    for k in range(nz):
        for i in range(nx - 1):
            for j in range(ny - 1):
                a, b = idx[i, j, k], idx[i + 1, j, k]
                c, d = idx[i + 1, j + 1, k], idx[i, j + 1, k]
                # Outward winding on each face (front z = 0 faces -z, back faces +z).
                if k == 0:
                    t.tris += [(a, c, b), (a, d, c)]
                else:
                    t.tris += [(a, b, c), (a, c, d)]
    # Steel-like but yielding: spring 2 MN/m, damping 400 N s/m, deform at 40 kN,
    # breaks at 400 kN (RoR's set_beam_defaults: spring, damp, deform, break).
    return t.text(dry_mass=600, minimass=4, beam_defaults=[2.0e6, 400, 4.0e4, 4.0e5])


def truss():
    t = Truck("scenario matrix: bridge truss on piers, loaded at mid-span (no wheels)")
    panels, length, depth, width, pier = 10, 1.0, 1.5, 1.5, 2.0
    bot, top = {}, {}
    for k in range(2):
        z = k * width
        for i in range(panels + 1):
            mid = i in (panels // 2,)
            bot[i, k] = t.node(i * length, pier, z, "l" if mid else "n", 3000 if mid else None)
            top[i, k] = t.node(i * length, pier + depth, z)
            if i in (0, panels):
                t.fixes.append(bot[i, k])
        for i in range(panels):
            t.beam(bot[i, k], bot[i + 1, k])
            t.beam(top[i, k], top[i + 1, k])
        for i in range(panels + 1):
            t.beam(bot[i, k], top[i, k])
        # Pratt diagonals: toward mid-span from the top.
        for i in range(panels):
            if i < panels // 2:
                t.beam(top[i, k], bot[i + 1, k])
            else:
                t.beam(bot[i, k], top[i + 1, k])
    for i in range(panels + 1):
        t.beam(bot[i, 0], bot[i, 1])
        t.beam(top[i, 0], top[i, 1])
        if i < panels:
            t.beam(bot[i, 0], bot[i + 1, 1])
            t.beam(top[i, 0], top[i + 1, 1])
    # Piers: a column under each end of each plane, its foot anchored.
    for i in (0, panels):
        for k in range(2):
            foot = t.node(i * length, 0.0, k * width)
            t.fixes.append(foot)
            t.beam(foot, bot[i, k])
    return t.text(dry_mass=400, minimass=4, beam_defaults=[2.0e6, 400, 6.0e4, 6.0e5])


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, text in [("barrier.truck", barrier()), ("truss.truck", truss())]:
        with open(os.path.join(OUT, name), "w") as f:
            f.write(text)
        print(os.path.join(OUT, name))


if __name__ == "__main__":
    main()
