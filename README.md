# SGM-StereoFlow
This code is based on a SGM-based algorithm for calculating disparity images.
Stereo and Flow Images can be calculated. We added computation of combined StereoFlow Images.

# Installation/Compilation
linked libraries:

-- OpenCV (used: latest version from their github, including opencv_contrib)

-- PCL (used: Release 1.8.0 from their github)

-- libpng 1.6[currently not needed](used: Release 1.6.29 from https://sourceforge.net/projects/libpng/files/libpng16/)

Adjust CMakeLists.txt to point it to your OpenCV/PCL install directories.

# Usage
Adjust "kittiDir" in sgm_stereoflow.cpp to point to your local KITTI data folder & recompile.

Pass ID of KITTI image (i.e. 000045) as parameter.

->Stereo-disparity, Flow-disparity & StereoFlow-disparity are calculated successively

(Image-loading is currently limited to convinient usage of KITTI images, but adjusting it to take arbitrary files should be no problem at all.)

# Outputs (currently saved in ".."):

-disparityStereo.png

-disparityFlow.png

-disFlowFlag.png (indicates area, where Flow-disparity could be evaluated fully)

-disparityStereoFlow.png
