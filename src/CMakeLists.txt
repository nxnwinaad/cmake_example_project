# Tutorial CMake project
# Rose Hendrix
# 16FEB2017

cmake_minimum_required (VERSION 2.8)
project (cmake_tutorial)

file(GLOB SOURCES 
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_executable(cmake_tutorial ${SOURCES})

include( ${CMAKE_CURRENT_SOURCE_DIR}/../libs/ParallelUtils.cmake ${OpenCV_DIR}/OpenCVConfig.cmake  )
find_package( OpenCV REQUIRED )
target_link_libraries( cmake_tutorial ${OpenCV_LIBS} )
include_directories(${OpenCV_DIR}/include/opencv2 $ENV{Eigen_DIR} )

if(NOT CMAKE_VERSION VERSION_LESS "3.6.0")
  set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT cmake_tutorial)
endif()