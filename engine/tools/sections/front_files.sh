#!/usr/bin/env bash
# Which generated files each front rule changes, on every registered design: generates every
# design with TAG's rules into out/gen-rules/<TAG>/ and lists the files that differ from
# out/gen-before/ (the designs generated before the rules, tools/sections/gen_all.sh). From
# engine/:   tools/sections/front_files.sh TAG [rule ...] [all] [crash=PRESET]
set -u
tag=${1:?tag}
shift
pd=out/gen-rules/params_$tag
mkdir -p "$pd"
for d in $(target/release/examples/vehicle_gen_rig --list </dev/null); do
  perl -e 'alarm 120; exec @ARGV' python3 -I tools/sections/front_params.py "$d" "$pd/$d.json" "$@" </dev/null || exit 1
done
tools/sections/gen_all.sh "out/gen-rules/$tag" "$pd" >/dev/null
echo "== $tag ($*): files that differ from out/gen-before"
diff <(cut -c67- out/gen-before.sha256 | paste -d' ' - <(cut -c1-64 out/gen-before.sha256)) \
     <(cut -c67- "out/gen-rules/$tag.sha256" | paste -d' ' - <(cut -c1-64 "out/gen-rules/$tag.sha256")) \
  | grep '^>' | cut -c3- | awk '{print $1}' | sort
