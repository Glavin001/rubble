#!/bin/sh
# Runs a GPU measurement with a hard kill timer: bounded.sh <seconds> <program> [args...]
# (SIGALRM kills the process when the timer runs out; never longer than 20 minutes.)
secs="$1"
shift
[ "$secs" -le 1200 ] || secs=1200
exec perl -e 'alarm shift; exec @ARGV or die "exec: $!"' "$secs" "$@"
