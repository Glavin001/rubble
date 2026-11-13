#!/bin/sh
# Runs a GPU job with a hard kill timer, holding the machine's GPU lock:
#   bounded.sh <seconds> <program> [args...]
# The lock is engine-webgpu's (`hold_gpu_lock`): an exclusive flock on
# ~/.rubble/gpu.lock (a $RUBBLE_GPU_LOCK other than 0 or 1 is the lock file's path). This wrapper waits for it first (the
# wait does not count against the timer), holds it for the whole run and runs the
# program with RUBBLE_GPU_LOCK=0 (it must not wait on the lock its parent holds). The
# timer then kills the program when it runs out (at most 20 minutes).
secs="$1"
shift
[ "$secs" -le 1200 ] || secs=1200
case "${RUBBLE_GPU_LOCK:-}" in
  ""|0|1) lock="$HOME/.rubble/gpu.lock" ;;
  *) lock="$RUBBLE_GPU_LOCK" ;;
esac
mkdir -p "$(dirname "$lock")"
exec perl -e '
  use Fcntl qw(:flock);
  my ($lock, $secs, @cmd) = @ARGV;
  open(my $fh, ">>", $lock) or die "bounded.sh: open $lock: $!";
  unless (flock($fh, LOCK_EX | LOCK_NB)) {
    print STDERR "bounded.sh: waiting for the GPU lock $lock\n";
    flock($fh, LOCK_EX) or die "bounded.sh: lock: $!";
  }
  $ENV{RUBBLE_GPU_LOCK} = "0";
  my $pid = fork() // die "fork: $!";
  if ($pid == 0) { exec @cmd or die "exec: $!"; }
  local $SIG{ALRM} = sub { print STDERR "bounded.sh: timer ($secs s) ran out: killing $pid\n"; kill "KILL", $pid; };
  alarm $secs;
  waitpid($pid, 0);
  my $st = $?;
  alarm 0;
  exit($st & 127 ? 128 + ($st & 127) : $st >> 8);
' "$lock" "$secs" "$@"
