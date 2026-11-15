#!/usr/bin/env python3
"""The explicit kernels' capped loops whose bodies reach a barrier.

A loop whose body calls a barrier (directly or through any function that does) must
end uniformly across the workgroup: it takes EB_END_UNIFORM / EB_PASS_UNIFORM, which
the per-invocation pass budget (explicit_loop_cap.slang) does not cut. Every other
capped loop takes EB_END / EB_PASS and is budgeted.

Usage: barrier_loops.py [--check] <slang files...>
Prints file:line of each capped loop, its form and whether it reaches a barrier.
--check: exit 1 when a barrier loop uses a budgeted form (or a barrier-free loop a
uniform one, which is allowed but reported).
"""
import re
import sys

BARRIERS = ("GroupMemoryBarrierWithGroupSync", "DeviceMemoryBarrierWithGroupSync",
            "AllMemoryBarrierWithGroupSync")
KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "uint", "int", "float",
            "float3", "float4", "uint4", "uint3", "uint2", "int3", "bool", "asfloat", "asuint",
            "asint", "min", "max", "abs", "all", "any", "floor", "dot", "cross", "length",
            "normalize", "clamp", "sqrt", "select"}


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), s, flags=re.S)
    return re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), s)


def match_brace(s, i):
    """Index after the block that opens at s[i] == '{'."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    return len(s)


def match_paren(s, i):
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "(":
            depth += 1
        elif s[j] == ")":
            depth -= 1
            if depth == 0:
                return j + 1
    return len(s)


def functions(src):
    """name -> list of bodies (overloads and #ifdef variants all counted)."""
    out = {}
    for m in re.finditer(r"\n[A-Za-z_][\w<>]*\s+([A-Za-z_]\w*)\s*\(", src):
        name = m.group(1)
        p = match_paren(src, m.end() - 1)
        rest = src[p:p + 200]
        k = re.match(r"\s*(?::\s*\w+\s*)?\{", rest)
        if not k:
            continue
        b0 = p + k.end() - 1
        out.setdefault(name, []).append(src[b0:match_brace(src, b0)])
    return out


def calls(body):
    return {c for c in re.findall(r"\b([A-Za-z_]\w*)\s*\(", body) if c not in KEYWORDS}


def barrier_functions(funcs):
    reach = {"sync"} | set(BARRIERS)
    changed = True
    while changed:
        changed = False
        for name, bodies in funcs.items():
            if name in reach:
                continue
            if any(calls(b) & reach for b in bodies):
                reach.add(name)
                changed = True
    return reach


def loop_body(src, i):
    """The body of the for/while loop whose header starts at src[i]."""
    p = src.index("(", i)
    e = match_paren(src, p)
    rest = src[e:]
    k = re.match(r"\s*\{", rest)
    if k:
        b0 = e + k.end() - 1
        return src[b0:match_brace(src, b0)]
    # One statement (possibly a nested loop or if with its own block).
    j = e
    depth = 0
    while j < len(src):
        c = src[j]
        if c in "({":
            depth += 1
        elif c in ")}":
            depth -= 1
        elif c == ";" and depth == 0:
            return src[e:j + 1]
        j += 1
    return src[e:]


def main():
    args = sys.argv[1:]
    check = "--check" in args
    files = [a for a in args if a != "--check"]
    srcs = {f: strip_comments(open(f).read()) for f in files}
    funcs = {}
    for s in srcs.values():
        for k, v in functions(s).items():
            funcs.setdefault(k, []).extend(v)
    reach = barrier_functions(funcs)
    bad = 0
    for f, s in srcs.items():
        for m in re.finditer(r"\b(for|while)\s*\(", s):
            header_end = match_paren(s, s.index("(", m.start()))
            header = s[m.start():header_end]
            body = loop_body(s, m.start())
            uses = re.findall(r"EB_(END|PASS)(_UNIFORM)?\s*\(", header + body[:400])
            if not uses:
                continue
            # The loop's own cap: its header's EB_END, or the first EB_PASS in its body.
            hm = re.search(r"EB_END(_UNIFORM)?\s*\(", header)
            if hm:
                form, uniform = "END", bool(hm.group(1))
            else:
                pm = re.search(r"EB_PASS(_UNIFORM)?\s*\(", body)
                if not pm:
                    continue
                form, uniform = "PASS", bool(pm.group(1))
            barrier = bool(calls(body) & reach)
            line = s.count("\n", 0, m.start()) + 1
            status = "ok"
            if barrier and not uniform:
                status = "NEEDS_UNIFORM"
                bad += 1
            elif uniform and not barrier:
                status = "uniform-without-barrier"
            if not check or status != "ok":
                print(f"{f}:{line}: EB_{form}{'_UNIFORM' if uniform else ''} barrier={barrier} {status}")
    if check and bad:
        sys.exit(1)


main()
