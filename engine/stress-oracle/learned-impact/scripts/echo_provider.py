#!/usr/bin/env python3
"""A file-protocol provider that answers every request with the oracle's own decoded state
(`oracle.json`, written by the harness with `--echo-oracle`): checks the closed loop's file
path end to end. Exits when `<dir>/done` appears.

    echo_provider.py <dir>
"""
import os
import sys
import time

d = sys.argv[1]
req, resp = os.path.join(d, "req"), os.path.join(d, "resp")
os.makedirs(resp, exist_ok=True)
answered = set()
while not os.path.exists(os.path.join(d, "done")):
    for name in sorted(os.listdir(req)) if os.path.isdir(req) else []:
        if name in answered or not os.path.exists(os.path.join(req, name, "ready")):
            continue
        body = open(os.path.join(req, name, "oracle.json")).read()
        tmp = os.path.join(resp, name + ".json.tmp")
        open(tmp, "w").write(body)
        os.replace(tmp, os.path.join(resp, name + ".json"))
        answered.add(name)
    time.sleep(0.002)
print(f"echo provider: answered {len(answered)} calls")
