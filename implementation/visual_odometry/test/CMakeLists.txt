cmake_minimum_required(VERSION 3.10)

enable_testing()
include(GoogleTest)

# Add a testing executable
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "-DSAVE_IMAGES")

add_executable(visual_odometry_test visual_odometry_test.cpp triangulation_test.cpp)

target_link_libraries(visual_odometry_test
    gtest
    gtest_main
    common
    visual_odometry
)

gtest_discover_tests(visual_odometry_test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} DISCOVERY_TIMEOUT 120)
