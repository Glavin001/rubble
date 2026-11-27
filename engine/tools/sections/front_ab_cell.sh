#!/usr/bin/env bash
# The six front rules re-measured with the crash box's closed cell on (lessons §26): the cell
# alone, each rule with it, cumulative, all; real_crash with and without rule 6. From engine/;
# pass tags to run only those (each tools/sections/front_ab.sh, a few minutes).
set -u
C=crash_box_cell R1=subframe_mounts_off_crash_box R2=crash_box_bay_alone R3=crash_box_stroke R4=yielding_stops R5=hood_fold_row R6=crush_from_section
sel=" $* "
go() { if [ "$sel" = "  " ] || [[ "$sel" == *" $1 "* ]]; then tools/sections/front_ab.sh "$@" || echo "variant $1 failed"; fi; }
go cell $C
go cr1 $C $R1
go cr2 $C $R2
go cr3 $C $R3
go cr4 $C $R4
go cr5 $C $R5
go cc12 $C $R1 $R2
go cc123 $C $R1 $R2 $R3
go cc1234 $C $R1 $R2 $R3 $R4
go cc12345 $C $R1 $R2 $R3 $R4 $R5
go call all
go rc_cell crash=real_crash $C
go rc_cr6 crash=real_crash $C $R6
go rc_call crash=real_crash all
