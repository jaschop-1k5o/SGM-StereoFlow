#!/bin/bash

for imgId in $(seq -f "%06g" $1 $2)
do
  echo "[eval.sh] ./sgm_stereoflow $imgId"
#  ./sgm_stereoflow $imgId
  mkdir -p ../outputs/$imgId
  mv ../*.png ../outputs/$imgId
  mv ../timers.txt ../outputs/$imgId
done
