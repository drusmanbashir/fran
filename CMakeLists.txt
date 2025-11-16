cmake_minimum_required(VERSION 3.24)
project(fran LANGUAGES CXX)

# set(FRAN_BUILD_TYPE "Release" CACHE STRING "Build type")
add_subdirectory(fran/cpp)
# Option
option(FRAN_BUILD_CPP "Build C++ extensions" ON)

# Auto-detect ITK: if ITK not found → disable extension
find_package(ITK QUIET COMPONENTS ITKCommon)

if(NOT ITK_FOUND)
    message(WARNING "ITK not found → disabling C++ extensions.")
    set(FRAN_BUILD_CPP OFF)
endif()

if(FRAN_BUILD_CPP)
    add_subdirectory(cpp)
else()
    message(STATUS "FRAN_BUILD_CPP=OFF → skipping C++ extension build.")
endif()
