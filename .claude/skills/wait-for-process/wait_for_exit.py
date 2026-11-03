#!/usr/bin/env python3
"""Block until the given processes exit, or a timeout passes.

Waits on PIDs with the kernel's exit notification: kqueue EVFILT_PROC/NOTE_EXIT on
macOS, pidfd_open + poll on Linux. Nothing is matched by command-line text, so the
waiter can never match itself or another waiter (the `while pgrep -f NAME` bug).

    wait_for_exit.py --timeout SECONDS PID [PID ...]
    wait_for_exit.py --timeout SECONDS --name EXACT_PROCESS_NAME

--name resolves PIDs once, by exact executable name (`pgrep -x`, never `-f`), and
excludes this process and its ancestors. It fails (exit 3) when nothing matches,
unless --allow-none.

Exit codes: 0 every process has exited (including "was not running"); 124 timeout
(the processes are still running and are NOT killed); 2 bad arguments; 3 --name
matched nothing.
"""
import argparse
import os
import select
import subprocess
import sys
import time


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def ancestors() -> set:
    out, pid = {os.getpid()}, os.getppid()
    while pid > 1 and pid not in out:
        out.add(pid)
        try:
            ps = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True)
            pid = int(ps.stdout.strip() or 1)
        except ValueError:
            break
    return out


def by_name(name: str) -> list:
    r = subprocess.run(["pgrep", "-x", name], capture_output=True, text=True)
    mine = ancestors()
    return [int(p) for p in r.stdout.split() if int(p) not in mine]


def wait_kqueue(pids: set, deadline: float) -> set:
    kq = select.kqueue()
    live = set()
    for p in pids:
        ev = select.kevent(p, filter=select.KQ_FILTER_PROC, flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                           fflags=select.KQ_NOTE_EXIT)
        try:
            kq.control([ev], 0, 0)
            live.add(p)
        except ProcessLookupError:
            pass  # already gone
    while live:
        left = deadline - time.monotonic()
        if left <= 0:
            break
        for ev in kq.control(None, len(live), left):
            live.discard(ev.ident)
    return live


def wait_pidfd(pids: set, deadline: float) -> set:
    poller, fds = select.poll(), {}
    for p in pids:
        try:
            fd = os.pidfd_open(p)
        except ProcessLookupError:
            continue
        fds[fd] = p
        poller.register(fd, select.POLLIN)
    while fds:
        left = deadline - time.monotonic()
        if left <= 0:
            break
        for fd, _ in poller.poll(left * 1000):
            poller.unregister(fd)
            os.close(fd)
            fds.pop(fd)
    return set(fds.values())


def wait_poll(pids: set, deadline: float) -> set:
    live = {p for p in pids if alive(p)}
    while live and time.monotonic() < deadline:
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        live = {p for p in live if alive(p)}
    return live


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeout", type=float, required=True, help="seconds; required, there is no unbounded wait")
    ap.add_argument("--name", help="exact process (executable) name, resolved once")
    ap.add_argument("--allow-none", action="store_true", help="with --name: no match is success")
    ap.add_argument("pids", nargs="*", type=int)
    a = ap.parse_args()
    if a.timeout <= 0 or (not a.pids) == (not a.name):
        ap.error("give --timeout > 0 and either PIDs or --name (not both)")
    pids = set(a.pids)
    if a.name:
        pids = set(by_name(a.name))
        if not pids:
            print(f"wait_for_exit: no process named {a.name!r}", file=sys.stderr)
            return 0 if a.allow_none else 3
    deadline = time.monotonic() + a.timeout
    if hasattr(select, "kqueue"):
        live = wait_kqueue(pids, deadline)
    elif hasattr(os, "pidfd_open"):
        live = wait_pidfd(pids, deadline)
    else:
        live = wait_poll(pids, deadline)
    if live:
        print(f"wait_for_exit: timeout after {a.timeout:g} s; still running: {sorted(live)}", file=sys.stderr)
        return 124
    print(f"wait_for_exit: exited: {sorted(pids)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
