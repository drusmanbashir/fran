cmake_minimum_required(VERSION 3.22)
project(fran_cc LANGUAGES CXX)

# Global C++ standard — fine for classic CMake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set (CMAKE_POSITION_INDEPENDENT_CODE ON)
# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------

# pybind11 (only needed if you’re compiling pybind modules)
find_package(pybind11 REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGLWidgets Gui)
set(PYBIND11_FINDPYTHON ON)

# ITK with classic include/use file — this provides ITK_LIBRARIES, ITK_INCLUDE_DIRS, ITKFactoryRegistration, etc.
find_package(ITK REQUIRED COMPONENTS
  ITKCommon
  ITKIOImageBase
  ITKIONIFTI
  ITKImageIntensity
  ITKConnectedComponents
  ITKLabelMap
)
include(${ITK_USE_FILE})

# --- VTK (with Qt support) ---
find_package(VTK REQUIRED COMPONENTS
  CommonCore
  CommonColor
  FiltersCore
  RenderingCore
  RenderingOpenGL2
  RenderingVolumeOpenGL2
  InteractionStyle
  GUISupportQt
)
# Torch (LibTorch)
find_package(Torch REQUIRED)




# ------------------- executables -------------------
option(WITH_TORCH "Build Torch based components" ON)
option (BUILD_VIEWER_ONLY "Only bbox_stats_itk +Qt" OFF)

if (BUILD_VIEWER_ONLY)
message(STATUS "Fast viewer-only mode enabled")
else()
message(STATUS "Full build enabled")
endif()
if (NOT SKBUILD)
add_executable(bbox_stats_itk
  bboxes_info.cxx
  bbox_stats_itk.cpp
  bboxviewer.cxx
  bboxviewer.ui 
  vtk_labelmap_view.cxx

  ${ITKFactoryRegistration}
  /home/ub/code/slicer_cpp/protot/utils.cxx
  /home/ub/code/slicer_cpp/protot/LabelmapGeometry.cxx
)

target_include_directories(bbox_stats_itk PRIVATE
  /home/ub/code/slicer_cpp
  /home/ub/code/slicer_cpp/protot
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${VTK_INCLUDE_DIRS}
)
target_link_libraries(bbox_stats_itk PRIVATE
  ${ITK_LIBRARIES}
  # ${TORCH_LIBRARIES}
 Qt6::Widgets
 fran_cpp_lib
 ${VTK_LIBRARIES}
)

vtk_module_autoinit(
    TARGETS bbox_stats_itk
    MODULES ${VTK_LIBRARIES}
  )

add_executable(torch_load_check tnsr_check.cpp)
target_link_libraries(torch_load_check PRIVATE ${TORCH_LIBRARIES})
endif()
# ------------------- core library -------------------
# NO bbox_stats_itk.cpp here (it has main()).
add_library(fran_cpp_lib
  /home/ub/code/slicer_cpp/protot/utils.cxx
  /home/ub/code/slicer_cpp/protot/LabelmapGeometry.cxx
)

target_include_directories(fran_cpp_lib PRIVATE
  /home/ub/code/slicer_cpp
  /home/ub/code/slicer_cpp/protot
)

target_link_libraries(fran_cpp_lib PRIVATE
  ${ITK_LIBRARIES}
  ${TORCH_LIBRARIES}
)

# ------------------- pybind module -------------------
# This is your file with:
#  - int add(int,int)
#  - process_file_py(...)
#  - PYBIND11_MODULE(fran_hello, m) { ... }
pybind11_add_module(fran_hello MODULE
  pybind_bboxes.cpp
  ${ITKFactoryRegistration}
)

target_include_directories(fran_hello PRIVATE
  /home/ub/code/slicer_cpp
  /home/ub/code/slicer_cpp/protot
)

target_compile_options(fran_hello PRIVATE ${TORCH_CXX_FLAGS})
target_link_libraries(fran_hello PRIVATE
  fran_cpp_lib
  ${ITK_LIBRARIES}
)


install (TARGETS fran_hello DESTINATION fran)

