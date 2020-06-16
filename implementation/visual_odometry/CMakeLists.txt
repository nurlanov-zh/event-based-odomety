cmake_minimum_required(VERSION 3.10)

project(visual_odometry)

include_directories(include)

add_library(${PROJECT_NAME}
    src/visual_odometry.cpp
    src/keyframe.cpp
    src/triangulation.cpp)

target_link_libraries(${PROJECT_NAME} common feature_detector opengv)
target_include_directories(${PROJECT_NAME}  PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(test)
