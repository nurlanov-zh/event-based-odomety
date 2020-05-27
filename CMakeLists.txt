cmake_minimum_required(VERSION 3.10)

project(event-based-odometry)
include(CTest)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" ${CMAKE_MODULE_PATH})
set(EIGEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen")
set(CERES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/build-ceres-solver")
set(PANGOLIN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/build-Pangolin/src")
set(OPENGV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/build-opengv")

# Set default build type if not specified otherwise.
# See https://cmake.org/pipermail/cmake/2012-May/050243.html
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified.")
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# Configure CCache if available
if (NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  else()
    message(STATUS "Dind't find ccache")
  endif()
else()
  message(STATUS "Compiler launcher already set. Not configuring ccache.")
  message(STATUS "CMAKE_C_COMPILER_LAUNCHER: ${CMAKE_C_COMPILER_LAUNCHER}")
  message(STATUS "CMAKE_CXX_COMPILER_LAUNCHER: ${CMAKE_CXX_COMPILER_LAUNCHER}")
endif()

# See comment in ./build_submodules.sh on the relation of arch=native
# and Eigen's alignment. TLDR: Passing arch=native for all build types
# is currently the only viable option to avoid subtle bugs with Eigen.
SET(CXX_MARCH native)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# for some reason clang doesn't include them in Wall whereas gcc does, so include them explicitly.
SET(EXTRA_WARNING_FLAGS "-Wsign-compare")

# clang-specific compile flags to avoid warnings from thirdpary includes
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(EXTRA_WARNING_FLAGS "${EXTRA_WARNING_FLAGS} -Wno-exceptions")
endif()

set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")                   # cmake default: "-g"
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -DNDEBUG -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")  # cmake default: "-O2 -g -DNDEBUG"
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")                                               # cmake default: "-O3 -DNDEBUG"
SET(CMAKE_CXX_FLAGS " -ftemplate-backtrace-limit=0 -Wall -Wextra ${EXTRA_WARNING_FLAGS} -march=${CXX_MARCH} ${CMAKE_CXX_FLAGS}")


set(EIGEN_INCLUDE_DIR_HINTS ${EIGEN_ROOT})
find_package(Eigen3 3.3.7 EXACT REQUIRED MODULE)
set(EIGEN_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR})  # for ceres / opengv
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "Found Eigen headers in: ${EIGEN3_INCLUDE_DIR}")

include_directories(thirdparty/Sophus)
include_directories(thirdparty/CLI11/include)
include_directories(include)

# to use link to target "ceres"
# Note: always link to ceres first, as otherwise a system-wide install might
# be found before our custom build. For example on macOS a both pangolin and
# TBB have /usr/local/include as part of their include directories.
find_package(Ceres 2.0 REQUIRED CONFIG HINTS "${CERES_ROOT}" NO_DEFAULT_PATH)

# to use link to target "pangolin"
set(Pangolin_DIR ${PANGOLIN_ROOT})
find_package(Pangolin REQUIRED CONFIG NO_DEFAULT_PATH)
message(STATUS "Found Pangolin headers in: ${Pangolin_INCLUDE_DIR}")

# to use link to target "opengv"
find_package(opengv REQUIRED CONFIG HINTS ${OPENGV_ROOT} NO_DEFAULT_PATH)
message(STATUS "Found OpenGV headers in: ${opengv_DIR}")

# to use link to target "TBB"
find_package(TBB REQUIRED)
message(STATUS "Found TBB headers in: ${TBB_INCLUDE_DIRS}")

# to use link to target "OpenCV"
# NOTE: not specifying version, since 2, 3 or 4 is fine
find_package(OpenCV REQUIRED COMPONENTS core features2d)
message(STATUS "Found OpenCV ${OpenCV_VERSION} headers in: ${OpenCV_INCLUDE_DIRS}")
add_library(OpenCV INTERFACE IMPORTED)
set_property(TARGET OpenCV PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OpenCV_INCLUDE_DIRS})
set_property(TARGET OpenCV PROPERTY INTERFACE_LINK_LIBRARIES ${OpenCV_LIBS})
if(OpenCV_VERSION VERSION_LESS 3)
  set_property(TARGET OpenCV PROPERTY INTERFACE_COMPILE_DEFINITIONS DBATK_OPENCV2)
endif()

find_program(CTEST_MEMORYCHECK_COMMAND REQUIRED valgrind)
message(STATUS "Found Valgrind: ${CTEST_MEMORYCHECK_COMMAND}")

enable_testing()
add_subdirectory(src)
add_subdirectory(thirdparty/googletest)