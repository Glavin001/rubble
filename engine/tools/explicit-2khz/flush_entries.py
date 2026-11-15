#!/usr/bin/env python3
"""Every compute entry point of a Slang file flushes the loop-cap flag
(explicit_loop_cap.slang): `return;` becomes `return ebLoopCapFlush();` and the body
ends with `ebLoopCapFlush();`. In place; entries that already flush are left.
Usage: flush_entries.py <file.slang>
"""
import re
import sys

path = sys.argv[1]
s = open(path).read()
out = []
i = 0
for m in re.finditer(r'\[shader\("compute"\)\]', s):
    if m.start() < i:
        continue
    open_ = s.index("{", m.end())
    depth = 0
    for j in range(open_, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                close = j
                break
    body = s[open_:close]
    if "ebLoopCapFlush" not in body:
        body = body.replace("return;", "return ebLoopCapFlush();")
        body = body.rstrip() + "\n    ebLoopCapFlush();\n"
    out.append(s[i:open_])
    out.append(body)
    i = close
out.append(s[i:])
open(path, "w").write("".join(out))
