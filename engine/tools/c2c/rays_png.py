#!/usr/bin/env python3
"""Front view of a car's collision-skin coverage from `vehicle_c2c_overlap`'s DUMP rays.csv.

Each 2 cm cell where the render surface is hit, coloured by how far behind it the first
collision triangle is: green <= 5 cm, yellow <= 15 cm, orange > 15 cm, red none at all.
Usage: rays_png.py rays.csv out.png   (no dependencies: PNG written with zlib)
"""
import csv, struct, sys, zlib

rows = list(csv.DictReader(open(sys.argv[1])))
g, x0, nx, ny, sc = 0.02, -1.1, 110, 80, 6
img = [[(255, 255, 255)] * nx for _ in range(ny)]
for r in rows:
    x, y, zr, zc = float(r["x"]), float(r["y"]), float(r["zr"]), float(r["zc"])
    i, j = int((x - x0) / g), int(y / g)
    if not (0 <= i < nx and 0 <= j < ny):
        continue
    if zc <= -8.9:
        c = (220, 30, 30)
    else:
        gap = zr - zc
        c = (40, 170, 60) if gap <= 0.05 else (235, 200, 40) if gap <= 0.15 else (240, 120, 20)
    img[ny - 1 - j][i] = c
raw = b"".join(
    b"\x00" + b"".join(bytes(px) * sc for px in row) for row in img for _ in range(sc)
)
def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", nx * sc, ny * sc, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
open(sys.argv[2], "wb").write(png)
