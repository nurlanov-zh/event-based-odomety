cmake_minimum_required(VERSION 3.10)

enable_testing()
include(GoogleTest)

add_executable(camera_model_test camera_model_test.cpp)

target_link_libraries(camera_model_test
    gtest
    gtest_main
    common
)

gtest_discover_tests(camera_model_test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} DISCOVERY_TIMEOUT 120)