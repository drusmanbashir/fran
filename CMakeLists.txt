cmake_minimum_required(VERSION 3.24)
option(FRAN_BUILD_CPP "Build FRAN C++ extensions" ON)
if(FRAN_BUILD_CPP)
    project(fran LANGUAGES CXX)
    message(STATUS "Building FRAN C++ extensions.")
    add_subdirectory(fran/cpp/dashboard)
else()
    message(STATUS "FRAN_BUILD_CPP=OFF → skipping C++ extension build.")
endif()

# Option to build C++ extensions (default = ON for local dev)

# Try to find ITK quietly
find_package(ITK QUIET COMPONENTS ITKCommon)

# Auto-disable if ITK missing
if(NOT ITK_FOUND)
    message(STATUS "ITK not found → disabling FRAN C++ extensions.")
    set(FRAN_BUILD_CPP OFF)
endif()

# Only build C++ if enabled AND ITK found

