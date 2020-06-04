cmake_minimum_required(VERSION 3.10)

enable_testing()
include(GoogleTest)

# Add a testing executable
add_executable(replayer_test replayer_test.cpp)

target_link_libraries(replayer_test
    gtest
    gtest_main
    replayer
)

gtest_discover_tests(replayer_test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} DISCOVERY_TIMEOUT 120)
