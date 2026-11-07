#!/bin/sh
# Runs a GPU job with a hard kill timer, holding the machine's shared GPU lock:
#   bounded.sh <seconds> <program> [args...]
# The lock is a directory (mkdir is atomic): $GPU_LOCK (default
# ~/Development/rubble-artifacts/.gpu.lock), with the holder's pid and command in it.
# It is waited for (a lock whose holder is gone is taken over), held for the whole run
# and released on exit. SIGALRM kills the program when the timer runs out (at most 20
# minutes). GPU_LOCK_HELD=1 skips taking it (a caller that holds it already).
secs="$1"
shift
[ "$secs" -le 1200 ] || secs=1200
lock="${GPU_LOCK:-$HOME/Development/rubble-artifacts/.gpu.lock}"
if [ "${GPU_LOCK_HELD:-0}" != 1 ]; then
  until mkdir "$lock" 2>/dev/null; do
    holder=$(cat "$lock/pid" 2>/dev/null)
    if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then
      echo "bounded.sh: taking over a stale GPU lock (pid $holder gone)" >&2
      rm -rf "$lock"
      continue
    fi
    sleep 2
  done
  echo $$ > "$lock/pid"
  echo "$*" > "$lock/cmd"
  trap 'rm -rf "$lock"' EXIT INT TERM
fi
perl -e 'alarm shift; exec @ARGV or die "exec: $!"' "$secs" "$@"
