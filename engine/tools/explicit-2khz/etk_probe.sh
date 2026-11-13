#!/bin/sh
# The ETK K-series alone in the crash town (coupling off), its explicit frame timed by
# explicit_frame_probe after 60 World steps at 60 Hz. Extra env (EV_ABLATE, ...) passes
# through. Usage: etk_probe.sh [probe binary] [cars]
ETK_DIR="${ETK_DIR:-/Users/glavin/Development/rubble-community/engine/crates/engine-webgpu/__mods/etkk_tex}"
bin="${1:-target/release/examples/explicit_frame_probe}"
cars="${2:-1}"
here="$(dirname "$0")"
EV_ETK="$ETK_DIR" EV_NO_CONTACT=1 EV_HZ=60 "$here/bounded.sh" 900 "$bin" "$cars" 60 2>&1 | grep -v 'warn' | grep -v 'contact off\|coupling off'
