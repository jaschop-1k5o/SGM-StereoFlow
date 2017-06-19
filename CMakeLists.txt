#Project
cmake_minimum_required(VERSION 2.8)
project( DisplayImage )

#Set OpenCV_DIR to the folder of OpenCVConfig.cmake.
#Therefore, find_package will automatically read the cmake file for configuration.
#Several varaibles will be set up. More detail : OpenCVConfig.cmake
#Those included files have been done by OpenCVConfig.cmake as well.

#----------------------------------------------------------------------------------
#set(OpenCV_DIR "/usr/share/OpenCV") #system level
SET(OpenCV_DIR "/home/johann/TUM/17S/Seminar-HWSWCodesign/OpenCvRepo/opencv/build") #local level
#SET(PCL_DIR "/home/johann/TUM/17S/Seminar-HWSWCodesign/PCLLocal/lib/share/pcl-1.8") #local level

#find_package( PCL REQUIRED )
find_package( OpenCV REQUIRED )

message("-- OpenCVConfig.cmake :${OpenCV_CONFIG}")
message("-- OpenCV_INCLUDE_DIRS :${OpenCV_INCLUDE_DIRS}")
message("-- OpenCV_INSTALL_PATH :${OpenCV_INSTALL_PATH}")
message("-- OpenCV_LIBS_PATH :${OpenCV_LIBS_PATH}")
message("-- PCLConfig.cmake :${PCL_CONFIG}")
message("-- PCL_INCLUDE_DIRS :${PCL_INCLUDE_DIRS}")
message("-- PCL_INSTALL_PATH :${PCL_INSTALL_PATH}")

#set(CMAKE_CXX_FLAGS "-O3")        ## Optimize

include_directories(include ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -O3")        ## Optimize
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

#add_executable( sgm_flow sgm_flow.cpp )
#target_link_libraries( sgm_flow ${OpenCV_LIBS} )

add_executable( sgm_stereo sgm_stereo.cpp SGM.cpp )
target_link_libraries( sgm_stereo ${OpenCV_LIBS})
target_link_libraries( sgm_stereo ${PCL_LIBS})
