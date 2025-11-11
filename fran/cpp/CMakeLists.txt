cmake_minimum_required(VERSION 3.22)
project(fran_cc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(ITK REQUIRED COMPONENTS
  ITKCommon
  ITKIOImageBase
  ITKIONIFTI
  ITKImageIntensity
  ITKConnectedComponents
  ITKLabelMap
)
include(${ITK_USE_FILE})

find_package(Torch REQUIRED)

add_executable(bbox_stats_itk
  bbox_stats_itk.cpp
  ${ITKFactoryRegistration}
  /home/ub/code/slicer_cpp/protot/utils.cxx
  /home/ub/code/slicer_cpp/protot/LabelmapGeometry.cxx
)

target_include_directories(bbox_stats_itk PRIVATE
  /home/ub/code/slicer_cpp
  /home/ub/code/slicer_cpp/protot
)

target_link_libraries(bbox_stats_itk PRIVATE
  ${ITK_LIBRARIES}
  ${TORCH_LIBRARIES}
)

