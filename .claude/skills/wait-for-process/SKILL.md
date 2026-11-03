---
name: wait-for-process
description: How to wait for, gate on, or stop background processes (benchmarks, builds, test runs, GPU jobs) without hangs. Use before writing any wait loop, "while pgrep", sleep-poll, quiet-machine gate, or pkill, and before ending a turn with background work running.
---

# Waiting on processes

A wait that matches processes by command-line text can match itself. On 2026-11-03 four
waiters (`while pgrep -f "explicit_beams_scaling" ...; do sleep 10; done`) each matched
the others' command lines and kept one another alive for 3 hours. Follow these rules
exactly.

## 1. Prefer not waiting at all

Run the work itself in the background (the Bash tool's `run_in_background`). The
harness notifies you when it exits. Put steps that depend on each other in one
command or script (`build && bench && summarize`), not in a second waiter.

## 2. Wait on PIDs, never on names or patterns

Use the script next to this file. It waits on the kernel's exit event (kqueue on
macOS, pidfd on Linux), and `--timeout` is required:

```bash
python3 -I .claude/skills/wait-for-process/wait_for_exit.py --timeout 900 "$PID"
```

- Get the PID where the process starts: `$!` after `cmd &`, or a pidfile the
  launcher writes (`echo $$ > run.pid` before `exec`).
- For a process you did not start, resolve it once by exact executable name:
  `--name explicit_beams_scaling` (uses `pgrep -x`, excludes itself and its
  ancestors, exits 3 if nothing matches).
- Exit codes: 0 means exited; 124 means timeout, with the process still running
  and not killed; 3 means `--name` matched nothing. Act on 124 and 3 explicitly.

**Never** use `pgrep -f` / `pkill -f` / `ps | grep` in a loop condition or to kill.
The pattern also matches the waiter's own shell, other waiters, and other sessions'
processes. Kill only a PID you recorded, after checking it with `ps -o pid,etime,command -p PID`.

## 3. Every wait has a bound

Every loop that waits on a condition, such as "GPU quiet" or "file appears", has a
deadline. When the deadline passes, the loop does something explicit: skip and log,
or fail. For example, a quiet-machine gate:

```sh
# Wait up to 30 min for a quiet machine; skip the piece if it never comes.
end=$(( $(date +%s) + 1800 ))
until [ "$(gpu_busy)" -lt 10 ] && [ "$(load_ok)" = 1 ]; do
  [ "$(date +%s)" -ge "$end" ] && { echo "skip: machine never quiet"; continue 2; }
  sleep 30
done
```

The work itself has a kill timer too, for example `perl -e 'alarm 900; exec @ARGV' cmd`.
It also saves partial results as it goes, so a timeout loses nothing.

## 4. Before ending a turn

List the background tasks you started. Stop any that are no longer needed, using
their task ID or recorded PID. A waiter left from earlier work blocks goal
evaluation and wastes the user's time.
