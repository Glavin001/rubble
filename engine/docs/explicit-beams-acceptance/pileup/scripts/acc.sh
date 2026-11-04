#!/bin/sh
# Two-car head-on and T-bone accuracy ensembles (nudges=24) against RoR, per car.
S=$SCRATCH
I=/Users/glavin/Development/rubble-beamng-parity/inspiration
export RUBBLE_AIRFOIL_DIR=$I/rigs-of-rods/resources/airfoils
BIN=${BIN:-$S/acc/ebs}
for car in ${CARS:-viper:ror-repo/viper/ViperGTS.car e36:ror-repo/e36/E36Sedan.truck mustang:ror-repo/mustang/1982-Ford-Mustang-GT-351W.truck}; do
  name=${car%%:*}; file=${car#*:}
  for sc in ${SCS:-headon tbone}; do
    perl -e 'alarm 900; exec @ARGV' $BIN $I/$file ror=$I/ror-harness/bin/ror_ref scenarios=$sc counts=2 rubble_counts= fps=120 frames=360 speed=27.78 drag=0 trials=1 nudges=24 quiet_wait=800 max_load=8 out=$S/acc/$name-$sc > $S/acc/$name-$sc.log 2>&1
    echo "$name $sc exit $?"
  done
done
