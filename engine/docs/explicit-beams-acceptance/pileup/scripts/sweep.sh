#!/bin/sh
S=$SCRATCH
I=/Users/glavin/Development/rubble-beamng-parity/inspiration
export RUBBLE_AIRFOIL_DIR=$I/rigs-of-rods/resources/airfoils
for car in ${CARS:-viper:ror-repo/viper/ViperGTS.car e36:ror-repo/e36/E36Sedan.truck mustang:ror-repo/mustang/1982-Ford-Mustang-GT-351W.truck}; do
  name=${car%%:*}; file=${car#*:}
  perl -e 'alarm 900; exec @ARGV' $S/acc/ebs $I/$file ror=$I/ror-harness/bin/ror_ref scenarios=pileup counts=1,2,4,8,16,32,64,128,256,512,1024 rubble_counts=2048,4096,8192 fps=120 frames=360 speed=27.78 drag=0 trials=3 nudges=0 quiet_wait=1800 max_load=${MAXLOAD:-8} ceiling_ms=33.3 ceiling_frames=10 deadline_s=780 out=$S/sweep/$name > $S/sweep/$name.log 2>&1
  echo "$name exit $?"
done
