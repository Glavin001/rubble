#!/bin/sh
S=$SCRATCH
I=/Users/glavin/Development/rubble-beamng-parity/inspiration
export RUBBLE_AIRFOIL_DIR=$I/rigs-of-rods/resources/airfoils
for car in viper:ror-repo/viper/ViperGTS.car e36:ror-repo/e36/E36Sedan.truck mustang:ror-repo/mustang/1982-Ford-Mustang-GT-351W.truck; do
  name=${car%%:*}; file=${car#*:}
  for tag in before after; do
    perl -e 'alarm 900; exec @ARGV' $S/before/explicit_beams_scaling_$tag $I/$file scenarios=mixed counts=1,64 rubble_counts=1024 fps=120 frames=360 speed=27.78 drag=0 trials=3 nudges=0 quiet_wait=800 max_load=8 out=$S/speed2/$tag/$name > $S/speed2/$tag/$name.log 2>&1
    echo "$name $tag exit $?"
  done
done
