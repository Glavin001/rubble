#!/usr/bin/env python3
"""Run a command as the machine's one GPU job, with a hard time limit.

    engine/scripts/with-gpu-lock.py --limit 600 -- node crash-orbit.mjs capture sedan

Takes an exclusive flock on ~/.rubble/gpu.lock (a RUBBLE_GPU_LOCK other than 0 or 1 is
the lock file's path), the lock the repo's cargo runs take in hold_gpu_lock()
(.cargo/config.toml sets RUBBLE_GPU_LOCK=1), waiting while another job holds it. The
command runs with RUBBLE_GPU_LOCK=0 (it must not wait on the lock this process holds) in
its own process group (headless Chrome and its helpers included); after --limit seconds
(default 900) the whole group is killed by its id. The lock is released when this
process exits.
"""
import fcntl
import os
import signal
import subprocess
import sys
import time


def lock_path(env):
    """The lock file: RUBBLE_GPU_LOCK when it is a path (not unset, empty, 0 or 1)."""
    if env in (None, "", "0", "1"):
        return os.path.expanduser("~/.rubble/gpu.lock")
    return env


def main() -> int:
    args = sys.argv[1:]
    limit = 900.0
    if args[:1] == ["--limit"]:
        limit = float(args[1])
        args = args[2:]
    if args[:1] == ["--"]:
        args = args[1:]
    if not args:
        print(__doc__, file=sys.stderr)
        return 2
    path = lock_path(os.environ.get("RUBBLE_GPU_LOCK"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock = open(path, "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"with-gpu-lock: waiting for {path} (another GPU job is running)", file=sys.stderr, flush=True)
        t0 = time.time()
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(f"with-gpu-lock: got the lock after {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    child = subprocess.Popen(args, start_new_session=True, env={**os.environ, "RUBBLE_GPU_LOCK": "0"})
    try:
        return child.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        print(f"with-gpu-lock: time limit ({limit:.0f} s), killing process group {child.pid}", file=sys.stderr, flush=True)
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        return 124
    except KeyboardInterrupt:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        return 130
    finally:
        # Anything the command left in its group goes too.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # Gone, or (macOS) only exited members left in the group.
            pass


if __name__ == "__main__":
    sys.exit(main())
