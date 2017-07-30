#!/bin/bash
#USAGE: place in same folder as sgm_stereoflow executeable
#PARAMETERS: start & end KITTI image nr, e.g. "bash eval.sh 0 20" (zero-padding done by script)
#WHAT IT DOES: excutes ./sgm_stereoflow with consecutive KITTI imgIDs, and moves outputs into ../outputs/{$imgId}

for imgId in $(seq -f "%06g" $1 $2)
do
  echo "[eval.sh] ./sgm_stereoflow $imgId"
  ./sgm_stereoflow $imgId
  mkdir -p ../outputs/$imgId
  mv ../*.png ../outputs/$imgId
  mv ../timers.txt ../outputs/$imgId
done
