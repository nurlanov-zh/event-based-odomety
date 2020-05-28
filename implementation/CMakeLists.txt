cmake_minimum_required(VERSION 3.10)

#add_subdirectory(googletest EXCLUDE_FROM_ALL)
# ensure gtest include directory is added before any of the
# system-wide include directories that might contain gtest
include_directories(BEFORE ${gtest_SOURCE_DIR}/include)

include(GoogleTest)

enable_testing()

add_executable(dummy_test test/dummy_test.cpp)
target_link_libraries(dummy_test gtest gtest_main)

gtest_discover_tests(dummy_test DISCOVERY_TIMEOUT 120)
