#!/bin/bash
#USAGE: place in {KITTI Benchmark results directory}/data
#PARAMETERS: start & end KITTI image nr & SGM outputs folder, e.g. "bash eval.sh 0 20 ~/my/local/StereoFlow/path/outputs"
#WHAT IT DOES: copies&renames stereoflow images from outputs folder ($3) into . for evaluation by KITTI Benchmark

for imgId in $(seq -f "%06g" $1 $2)
do
  cp ${3}/${imgId}/disparityStereoFlow.png ${imgId}_10.png
done
