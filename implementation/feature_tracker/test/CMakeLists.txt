cmake_minimum_required(VERSION 3.10)

enable_testing()
include(GoogleTest)

# Add a testing executable
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "-DSAVE_IMAGES")

add_executable(feature_tracker_test feature_detector_test.cpp)

target_link_libraries(feature_tracker_test
    gtest
    gtest_main
    common
    dataset_reader
    feature_detector
)

add_executable(patch_test patch_test.cpp)
target_link_libraries(patch_test
    gtest
    gtest_main
    common
    feature_detector
)

gtest_discover_tests(feature_tracker_test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} DISCOVERY_TIMEOUT 120)
gtest_discover_tests(patch_test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} DISCOVERY_TIMEOUT 120)
