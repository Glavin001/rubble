#!/bin/sh
# Build designs' .glb and rig.json (no renders) into out/<design>/, for the
# vehicle rigger (engine/docs/generated-vehicles.md). Usage, from
# engine/tools/vehicle-gen:  sh tools/build_rigs.sh police_sedan wagon ...
B=${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}
for d in "$@"; do
  "$B" -b --python build.py -- "$d" --out "out/$d" --no-render > "out/build_$d.log" 2>&1
  echo "$d: exit $? ($(grep -c 'Error' "out/build_$d.log") error lines); $(ls "out/$d" | tr '\n' ' ')"
done
