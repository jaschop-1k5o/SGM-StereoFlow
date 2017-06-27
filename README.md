# SGM-StereoFlow
This code is based on a SGM-based algorithm for calculating disparity images.
Stereo and Flow Images can be calculated. We added computation of combined StereoFlow Images.

# Installation/Compilation
required libraries
  -OpenCV (used: latest version from their github, including opencv_contrib)
  -PCL (used: Release 1.8.0 from their github)

Adjust CMakeLists.txt to point it to your OpenCV/PCL install directories.

# Usage
Please check sgm_stereoflow.cpp, and possibly update any hardcoded file paths.
