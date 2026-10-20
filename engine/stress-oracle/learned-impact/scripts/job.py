#!/usr/bin/env python3
"""Run a long command as a checkable job (plan: "Progress logging for every long-running job").

    job.py <name> -- <command...>

Writes `learned-impact/runs/<name>/progress.log` (every output line, flushed, timestamped)
and `status.json` (state, elapsed, line count, last line, exit code), rewritten at least
every 30 s. A heartbeat line goes to the log whenever the command is silent for 30 s.
Exit code is the command's.
"""
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), "runs")
BEAT = 30.0


def main():
    if "--" not in sys.argv:
        sys.exit(__doc__)
    i = sys.argv.index("--")
    name, cmd = sys.argv[1], sys.argv[i + 1:]
    d = os.path.join(RUNS, name)
    os.makedirs(d, exist_ok=True)
    log = open(os.path.join(d, "progress.log"), "a", buffering=1)
    st = {"job": name, "cmd": cmd, "state": "running", "started": time.time(), "lines": 0, "last_line": "", "exit": None, "last_error": None}
    lock = threading.Lock()
    last_out = [time.time()]

    def write_status():
        st["updated"] = time.time()
        st["elapsed_s"] = round(st["updated"] - st["started"], 1)
        tmp = os.path.join(d, "status.json.tmp")
        with open(tmp, "w") as f:
            json.dump(st, f, indent=1)
        os.replace(tmp, os.path.join(d, "status.json"))

    def emit(line):
        stamp = time.strftime("%H:%M:%S")
        with lock:
            log.write(f"[{stamp}] {line}\n")
            sys.stdout.write(f"[{name} {stamp}] {line}\n")
            sys.stdout.flush()

    write_status()
    emit("start: " + " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    done = threading.Event()

    def beat():
        while not done.wait(BEAT / 3):
            with lock:
                write_status()
            if time.time() - last_out[0] >= BEAT:
                emit(f"heartbeat: {time.time() - st['started']:.0f} s elapsed, {st['lines']} lines, last: {st['last_line'][:160]}")
                last_out[0] = time.time()

    t = threading.Thread(target=beat, daemon=True)
    t.start()
    for line in p.stdout:
        line = line.rstrip("\n")
        with lock:
            st["lines"] += 1
            st["last_line"] = line
            if "error" in line.lower() or "panicked" in line:
                st["last_error"] = line
        last_out[0] = time.time()
        emit(line)
    p.wait()
    done.set()
    st["exit"] = p.returncode
    st["state"] = "done" if p.returncode == 0 else "failed"
    with lock:
        write_status()
    emit(f"exit {p.returncode} after {time.time() - st['started']:.0f} s")
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
