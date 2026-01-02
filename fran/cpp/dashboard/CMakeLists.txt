cmake_minimum_required(VERSION 3.25)
project(dashboard LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC OFF)

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
endif()

# find packages
find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package (Xlnt  CONFIG REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(pybind11 REQUIRED)

find_package(CUDAToolkit REQUIRED)

#=================================

# file (GLOB_RECURSE PROJECT_HEADERS
#      "${CMAKE_SOURCE_DIR}/include/*.h")

add_library(fran_cpp STATIC
src/franproject.cpp
src/inference.cxx
src/training.cxx
)
add_library(cuda_utils  STATIC
src/cuda_utils.cxx
)
target_include_directories(cuda_utils
PRIVATE
    ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(cuda_utils PRIVATE CUDA::cudart)

target_include_directories(fran_cpp
PRIVATE
    ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(fran_cpp pybind11::embed Qt6::Widgets xlnt::xlnt)

add_executable(dashboard
    src/main.cpp
    src/mainwindow.cpp
    src/plansmodel.cpp
    src/mainwindow.ui
    src/trainingcontroller.cxx
    include/mainwindow.h
    include/trainingcontroller.h
    include/plansmodel.h
)

target_include_directories(dashboard
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}

)

target_link_libraries(dashboard PRIVATE Qt6::Widgets yaml-cpp fran_cpp cuda_utils)
