cmake_minimum_required(VERSION 3.25)
project(dashboard LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
endif()

# find packages
find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(yaml-cpp REQUIRED)

find_package(pybind11 REQUIRED)


#=================================
file(GLOB_RECURSE PROJECT_SOURCES
     "${CMAKE_SOURCE_DIR}/src/*.cpp"
     "${CMAKE_SOURCE_DIR}/src/*.cxx"
   "${CMAKE_SOURCE_DIR}/src/*.ui"
)
file (GLOB_RECURSE PROJECT_HEADERS
     "${CMAKE_SOURCE_DIR}/include/*.h")

add_executable(dashboard
    ${PROJECT_SOURCES}
    ${PROJECT_HEADERS}
    # src/mainwindow.ui
)

target_include_directories(dashboard
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include

)

target_link_libraries(dashboard PRIVATE Qt6::Widgets yaml-cpp pybind11::embed)
