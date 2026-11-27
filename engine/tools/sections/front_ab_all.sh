#!/usr/bin/env bash
# Every comparison of the front rules' study (lessons §25): all off, each rule alone, the rules
# cumulative in the audit's order, all on; the real_crash character with and without rule 6.
# From engine/; each variant is tools/sections/front_ab.sh (a few minutes each). Pass tags to
# run only those.
set -u
R1=subframe_mounts_off_crash_box R2=crash_box_bay_alone R3=crash_box_stroke R4=yielding_stops R5=hood_fold_row R6=crush_from_section
run() { tools/sections/front_ab.sh "$@" || echo "variant $1 failed"; }
sel=" $* "
go() { if [ "$sel" = "  " ] || [[ "$sel" == *" $1 "* ]]; then run "$@"; fi; }
go off
go r1 $R1
go r2 $R2
go r3 $R3
go r4 $R4
go r5 $R5
go c12 $R1 $R2
go c123 $R1 $R2 $R3
go c1234 $R1 $R2 $R3 $R4
go c12345 $R1 $R2 $R3 $R4 $R5
go all all
go rc_off crash=real_crash
go rc_r6 crash=real_crash $R6
go rc_all crash=real_crash all
