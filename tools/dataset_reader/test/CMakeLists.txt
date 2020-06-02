cmake_minimum_required(VERSION 3.10)

enable_testing()
include(GoogleTest)

# Add a testing executable
add_executable(dataset_reader_test davis240c_reader_test.cpp)

target_link_libraries(dataset_reader_test
    gtest
    gtest_main
    dataset_reader
    spdlog::spdlog
)

gtest_discover_tests(dataset_reader_test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} DISCOVERY_TIMEOUT 120)
