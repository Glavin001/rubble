#!/bin/zsh
# fetch_refs.sh <design> <dir>: download a design's reference photos
d=$1; out=$2; mkdir -p $out
python3 -c "import json,sys; j=json.load(open('refs/$d/photos.json')); [print(k, v) for k, v in j.items() if k != 'note']" | while read k u; do
  curl -sf -o $out/$k.jpg "$u" && echo "$k ok"
done
