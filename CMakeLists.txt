#Project
cmake_minimum_required(VERSION 2.8)
project( DisplayImage )

#Set OpenCV_DIR to the folder of OpenCVConfig.cmake.
#Therefore, find_package will automatically read the cmake file for configuration.
#Several varaibles will be set up. More detail : OpenCVConfig.cmake
#Those included files have been done by OpenCVConfig.cmake as well.

#----------------------------------------------------------------------------------
SET(OpenCV_DIR "/home/johann/TUM/17S/Seminar-HWSWCodesign/OpenCvRepo/opencv/build")
SET(PCL_DIR "/home/johann/TUM/17S/Seminar-HWSWCodesign/PCLLocal/lib/share/pcl-1.8")

find_package( PCL REQUIRED )
find_package( OpenCV REQUIRED )

file(GLOB PNG_LIBS "/home/johann/TUM/17S/Seminar-HWSWCodesign/png-library/libpng-1.6.29/build/libpng16.so")
set (PNG_LINK_LIBS ${PNG_LIBS})

message("-- OpenCVConfig.cmake :${OpenCV_CONFIG}")
message("-- OpenCV_INCLUDE_DIRS :${OpenCV_INCLUDE_DIRS}")
message("-- OpenCV_LIBS_PATH :${OpenCV_LIBS_PATH}")
message("-- PCLConfig.cmake :${PCL_CONFIG}")
message("-- PCL_INCLUDE_DIRS :${PCL_INCLUDE_DIRS}")
message("-- PCL_LIBRARY_DIRS :${PCL_LIBRARY_DIRS}")
message("-- PCL_SAMPLE_CONSENSUS_FOUND :${PCL_SAMPLE_CONSENSUS_FOUND}")
message("-- PCL_SAMPLE_CONSENSUS_LIBRARIES :${PCL_SAMPLE_CONSENSUS_LIBRARIES}")

include_directories(include ${PCL_INCLUDE_DIRS} ${PCL_LIBRARY_DIRS} ${OpenCV_INCLUDE_DIRS})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -O3")        ## Optimize
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

add_executable( sgm_stereoflow sgm_stereoflow.cpp SGM.cpp )
target_link_libraries( sgm_stereoflow ${OpenCV_LIBS})
target_link_libraries( sgm_stereoflow ${PCL_SAMPLE_CONSENSUS_LIBRARIES})
target_link_libraries( sgm_stereoflow ${PNG_LINK_LIBS} )
