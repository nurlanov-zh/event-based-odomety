cmake_minimum_required(VERSION 3.10)

enable_testing()
include(GoogleTest)

# Add a testing executable
add_executable(evaluator_test evaluator_test.cpp)

target_link_libraries(evaluator_test
    gtest
    gtest_main
    evaluator
)

gtest_discover_tests(evaluator_test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} DISCOVERY_TIMEOUT 120)
