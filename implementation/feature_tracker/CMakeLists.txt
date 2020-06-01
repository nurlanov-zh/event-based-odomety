cmake_minimum_required(VERSION 3.10)

project(feature_tracker)

include_directories(include)

add_library(feature_detector
    src/feature_detector.cpp
    src/patch.cpp
    src/flow_estimator.cpp)

target_link_libraries(feature_detector common ${OpenCV_LIBRARIES})
target_include_directories(feature_detector PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(test)
