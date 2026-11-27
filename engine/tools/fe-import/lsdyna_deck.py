#!/usr/bin/env python3
"""Read an LS-DYNA keyword deck (the subset a full-vehicle crash model uses) and cache it.

Stage-1 tool of the FE-import spike (engine/docs/fe-import-spike.md). Standard library
only. The deck is untrusted data: this script only reads it.

    python3 -I engine/tools/fe-import/lsdyna_deck.py DECK_DIR/combine.key OUT_DIR

follows *INCLUDE / *INCLUDE_TRANSFORM (the transformed include, a barrier, is parsed
and kept apart as "barrier" parts), and writes OUT_DIR/deck.pickle: nodes, elements,
parts, sections, materials, curves, connections and added masses, plus counts.

Formats handled: standard fixed-width cards (I8 element/node cards, I10 others), free
format (comma separated), long format (keyword suffix " +" or "*KEYWORD LONG": every
field 20 wide, untested on a real deck), *PARAMETER / *PARAMETER_EXPRESSION values
referenced as &NAME (expressions limited to products/quotients/sums of names and
numbers), *_TITLE variants (title line first) and *_ID variants of the connection cards.
"""
import os
import pickle
import re
import sys
from array import array

W8 = {  # keywords whose cards use 8-wide integer fields
    "*NODE": [8, 16, 16, 16, 8, 8],
    "*ELEMENT_SHELL": [8] * 10,
    "*ELEMENT_SOLID": [8] * 10,
    "*ELEMENT_BEAM": [8] * 10,
    "*ELEMENT_DISCRETE": [8, 8, 8, 8, 8, 16, 8, 16],
    "*ELEMENT_MASS": [8, 8, 16, 8],
    "*ELEMENT_MASS_PART": [8, 16, 16, 8],
    "*ELEMENT_SEATBELT_ACCELEROMETER": [10] * 8,
}


class Deck:
    def __init__(self):
        self.params = {}
        self.node_id = {}            # nid -> index
        self.xyz = array("d")
        self.shell_pid = array("l"); self.shell_n = array("l")   # 4 nids each
        self.solid_pid = array("l"); self.solid_n = array("l")   # 8 nids each
        self.beam_pid = array("l"); self.beam_n = array("l")     # 2 nids each
        self.discrete = []           # (eid, pid, n1, n2)
        self.elmass = []             # (eid, nid, mass, pid, source)
        self.mass_part = []          # (pid, addmass, finmass, source)
        self.parts = {}              # pid -> dict(title, secid, mid, source)
        self.sections = {}           # secid -> dict(kind, ...)
        self.mats = {}               # mid -> dict(type, ro, e, pr, sigy, etan, lcss, title)
        self.curves = {}             # lcid -> list of (a, o)
        self.spotwelds = []          # (n1, n2)
        self.nrb = []                # (pid, nsid)
        self.extra_nodes = []        # (pid, nsid)
        self.rigid_bodies = []       # (pidm, pids)
        self.joints = {}             # type -> count
        self.contacts = []           # (keyword, first card)
        self.node_sets = {}          # sid -> list of nids
        self.part_sets = {}          # sid -> list of pids
        self.keyword_counts = {}
        self.files = []
        self.include_stack = []

    # ---------------- field handling ----------------
    def val(self, s):
        s = s.strip()
        if not s:
            return None
        if s.startswith("&"):
            return self.params.get(s[1:].strip().upper())
        if s[0] == "-" and len(s) > 1 and s[1] == "&":
            v = self.params.get(s[2:].strip().upper())
            return None if v is None else -v
        try:
            return float(s)
        except ValueError:
            return s

    def fields(self, line, widths, long_fmt):
        if "," in line:
            return [self.val(x) for x in line.split(",")]
        if long_fmt:
            widths = [20] * len(widths)
        out, p = [], 0
        for w in widths:
            out.append(self.val(line[p:p + w]))
            p += w
        return out

    def eval_expr(self, expr):
        expr = expr.split("$")[0].strip()
        tok = re.sub(r"&([A-Za-z_][A-Za-z0-9_]*)",
                     lambda m: repr(self.params[m.group(1).upper()]), expr)
        if not re.fullmatch(r"[0-9eE+\-*/(). ]+", tok):
            raise ValueError("unsupported parameter expression: " + expr)
        # arithmetic on numbers only (checked above); no names reach eval
        return float(eval(tok, {"__builtins__": {}}, {}))

    # ---------------- reading ----------------
    def read(self, path, source="vehicle"):
        path = os.path.abspath(path)
        self.files.append((path, source))
        base = os.path.dirname(path)
        kw, card, long_fmt, block = None, 0, False, []
        title_kw = False
        pending_include = None
        with open(path, "r", encoding="latin-1") as f:
            for raw in f:
                line = raw.rstrip("\r\n")
                if not line or line[0] == "$":
                    continue
                if line[0] == "*":
                    if pending_include:
                        self._flush(kw, block, source, base, long_fmt)
                    else:
                        self._flush(kw, block, source, base, long_fmt)
                    parts = line.split()
                    kw = parts[0].upper()
                    long_fmt = (len(parts) > 1 and parts[1] == "+") or kw.endswith("+")
                    kw = kw.rstrip("+")
                    if kw == "*KEYWORD" and "LONG" in line.upper():
                        self.long_all = True
                    long_fmt = long_fmt or getattr(self, "long_all", False)
                    self.keyword_counts[(source, kw)] = self.keyword_counts.get((source, kw), 0) + 1
                    block = []
                    if kw in W8 and kw.startswith("*NODE") or kw.startswith("*ELEMENT_"):
                        self._stream_kw = kw
                    else:
                        self._stream_kw = None
                    continue
                if kw is None:
                    continue
                if self._stream_kw:
                    self._element_line(kw, line, source, long_fmt)
                else:
                    block.append(line)
        self._flush(kw, block, source, base, long_fmt)

    def _element_line(self, kw, line, source, lf):
        if kw == "*NODE":
            f = self.fields(line, W8["*NODE"], lf)
            nid = int(f[0])
            if nid in self.node_id:
                raise ValueError("duplicate node %d" % nid)
            self.node_id[nid] = len(self.xyz) // 3
            self.xyz.extend([f[1] or 0.0, f[2] or 0.0, f[3] or 0.0])
            if source != "vehicle":
                self.barrier_nodes = getattr(self, "barrier_nodes", set()); self.barrier_nodes.add(nid)
        elif kw == "*ELEMENT_SHELL":
            f = self.fields(line, W8[kw], lf)
            n = [int(x) for x in f[2:6] if x is not None]
            if len(n) == 3:
                n.append(n[2])
            if len(f) > 6 and any(x for x in f[6:10] if x):
                raise ValueError("8-node shells not supported")
            self.shell_pid.append(int(f[1])); self.shell_n.extend(n)
        elif kw == "*ELEMENT_SOLID":
            f = self.fields(line, W8[kw], lf)
            if len([x for x in f if x is not None]) <= 2:
                raise ValueError("two-line *ELEMENT_SOLID format not supported")
            n = [int(x) for x in f[2:10] if x is not None]
            if len(n) == 4:
                n = [n[0], n[1], n[2], n[2], n[3], n[3], n[3], n[3]]
            elif len(n) == 6:
                n = [n[0], n[1], n[2], n[2], n[3], n[4], n[5], n[5]]
            self.solid_pid.append(int(f[1])); self.solid_n.extend(n)
        elif kw == "*ELEMENT_BEAM":
            f = self.fields(line, W8[kw], lf)
            self.beam_pid.append(int(f[1])); self.beam_n.extend([int(f[2]), int(f[3])])
        elif kw == "*ELEMENT_DISCRETE":
            f = self.fields(line, W8[kw], lf)
            self.discrete.append((int(f[0]), int(f[1]), int(f[2]), int(f[3])))
        elif kw == "*ELEMENT_MASS":
            f = self.fields(line, W8[kw], lf)
            self.elmass.append((int(f[0]), int(f[1]), float(f[2] or 0.0), f[3], source))
        elif kw == "*ELEMENT_MASS_PART":
            f = self.fields(line, W8[kw], lf)
            self.mass_part.append((int(f[0]), float(f[1] or 0.0), float(f[2] or 0.0), source))
        else:
            self.other_elements = getattr(self, "other_elements", {})
            self.other_elements[kw] = self.other_elements.get(kw, 0) + 1

    def _flush(self, kw, block, source, base, lf):
        if kw is None or not block:
            return
        F = lambda line, n=8, w=10: self.fields(line, [w] * n, lf)
        if kw == "*PARAMETER":
            for line in block:
                typ, name, v = line[0:1], line[1:10].strip().upper(), line[10:30]
                if typ.strip() in ("R", "I"):
                    self.params[name] = float(v)
                else:  # older layout: name and value in 10-wide fields
                    self.params[line[0:10].strip().upper()] = float(line[10:20])
        elif kw == "*PARAMETER_EXPRESSION":
            for line in block:
                name = line[1:10].strip().upper()
                self.params[name] = self.eval_expr(line[10:])
        elif kw in ("*INCLUDE",):
            for line in block:
                self.include_stack.append(line.strip())
                self.read(os.path.join(base, line.strip()), source)
        elif kw == "*INCLUDE_TRANSFORM":
            # first line is the file; the rest are id offsets/transform (not applied to
            # parts we keep apart). The only transform here is a translation in X.
            fname = block[0].strip()
            tranid = None
            if len(block) >= 5:
                tranid = F(block[4], 1)[0]
            self.include_transform = (fname, tranid)
            self.read(os.path.join(base, fname), "barrier")
        elif kw == "*DEFINE_TRANSFORMATION":
            self.transformations = getattr(self, "transformations", {})
            tid = int(F(block[0], 1)[0])
            self.transformations[tid] = [line for line in block[1:]]
        elif kw.startswith("*PART") and not kw.startswith("*PART_"):
            i = 0
            while i + 1 < len(block):
                title, f = block[i].strip(), F(block[i + 1])
                pid = int(f[0])
                self.parts[pid] = dict(title=title, secid=int(f[1]) if f[1] else None,
                                       mid=int(f[2]) if f[2] else None, source=source,
                                       hgid=f[4])
                i += 2
        elif kw.startswith("*SECTION_"):
            b = block[1:] if kw.endswith("_TITLE") else block
            kind = kw[len("*SECTION_"):].replace("_TITLE", "")
            c1 = F(b[0])
            sid = int(c1[0])
            d = dict(kind=kind, elform=c1[1], title=block[0].strip() if kw.endswith("_TITLE") else None)
            if kind == "SHELL":
                c2 = F(b[1])
                t = [x for x in c2[0:4] if isinstance(x, float)]
                d.update(t=sum(t) / len(t) if t else 0.0, t4=c2[0:4], nip=c1[3],
                         marea=c2[5] or 0.0)
            elif kind == "BEAM":
                c2 = F(b[1]) if len(b) > 1 else []
                d.update(cst=c1[4], nsm=c1[6] or 0.0, card2=c2)
            elif kind == "DISCRETE":
                pass
            self.sections[sid] = d
        elif kw.startswith("*MAT_"):
            is_title = kw.endswith("_TITLE")
            b = block[1:] if is_title else block
            typ = kw[len("*MAT_"):].replace("_TITLE", "")
            c1 = F(b[0])
            mid = int(c1[0])
            d = dict(type=typ, title=block[0].strip() if is_title else None, ro=None)
            if typ in ("SPRING_ELASTIC", "DAMPER_VISCOUS", "SPRING_NONLINEAR_ELASTIC"):
                d.update(k=c1[1])
            else:
                d.update(ro=c1[1], e=c1[2], pr=c1[3])
            if typ in ("PIECEWISE_LINEAR_PLASTICITY", "MODIFIED_PIECEWISE_LINEAR_PLASTICITY"):
                c2 = F(b[1]) if len(b) > 1 else [None] * 8
                d.update(sigy=c1[4], etan=c1[5], fail=c1[6], c=c2[0], p=c2[1],
                         lcss=int(c2[2]) if isinstance(c2[2], float) else None)
            elif typ == "SPOTWELD":
                d.update(sigy=c1[4], etan=c1[5])
            elif typ == "LOW_DENSITY_FOAM":
                d.update(e=c1[2], lcid=c1[3])
            elif typ == "BLATZ-KO_RUBBER":
                d.update(g=c1[2], e=None, pr=None)
            elif typ == "NULL":
                d.update(e=c1[6], pr=c1[7])
            self.mats[mid] = d
        elif kw.startswith("*DEFINE_CURVE"):
            b = block[1:] if kw.endswith("_TITLE") else block
            c1 = F(b[0])
            lcid = int(c1[0])
            sfa = c1[2] or 1.0; sfo = c1[3] or 1.0
            sfa = sfa if sfa != 0 else 1.0; sfo = sfo if sfo != 0 else 1.0
            pts = []
            for line in b[1:]:
                f = self.fields(line, [20, 20], lf)
                if isinstance(f[0], float) and isinstance(f[1], float):
                    pts.append((f[0] * sfa, f[1] * sfo))
            self.curves[lcid] = pts
        elif kw == "*CONSTRAINED_SPOTWELD_ID" or kw == "*CONSTRAINED_SPOTWELD":
            step = 2 if kw.endswith("_ID") else 1
            for i in range(0, len(block), step):
                f = F(block[i + step - 1])
                self.spotwelds.append((int(f[0]), int(f[1])))
        elif kw == "*CONSTRAINED_NODAL_RIGID_BODY":
            for line in block:
                f = F(line)
                self.nrb.append((int(f[0]), int(f[2])))
        elif kw == "*CONSTRAINED_EXTRA_NODES_SET":
            for line in block:
                f = F(line)
                self.extra_nodes.append((int(f[0]), int(f[1])))
        elif kw == "*CONSTRAINED_RIGID_BODIES":
            for line in block:
                f = F(line)
                self.rigid_bodies.append((int(f[0]), int(f[1])))
        elif kw.startswith("*CONSTRAINED_JOINT_"):
            self.joints[kw] = self.joints.get(kw, 0) + 1
        elif kw.startswith("*CONTACT_"):
            self.contacts.append((kw, block[:2]))
        elif kw == "*SET_NODE_LIST":
            sid = int(F(block[0])[0])
            ids = []
            for line in block[1:]:
                ids.extend(int(x) for x in F(line) if isinstance(x, float) and x > 0)
            self.node_sets[sid] = ids
        elif kw.startswith("*SET_PART_LIST"):
            b = block[1:] if kw.endswith("_TITLE") else block
            sid = int(F(b[0])[0])
            ids = []
            for line in b[1:]:
                ids.extend(int(x) for x in F(line) if isinstance(x, float) and x > 0)
            self.part_sets[sid] = ids


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    deck = Deck()
    deck.read(sys.argv[1])
    os.makedirs(sys.argv[2], exist_ok=True)
    out = os.path.join(sys.argv[2], "deck.pickle")
    with open(out, "wb") as f:
        pickle.dump(deck.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("nodes", len(deck.node_id), "shells", len(deck.shell_pid), "solids", len(deck.solid_pid),
          "beams", len(deck.beam_pid), "discrete", len(deck.discrete), "mass elems", len(deck.elmass),
          "parts", len(deck.parts), "sections", len(deck.sections), "mats", len(deck.mats),
          "curves", len(deck.curves), "spotwelds", len(deck.spotwelds), "nrb", len(deck.nrb))
    print("params", deck.params)
    print("other elements", getattr(deck, "other_elements", {}))
    print("wrote", out)


if __name__ == "__main__":
    main()
