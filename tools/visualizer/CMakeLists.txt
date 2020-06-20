cmake_minimum_required(VERSION 3.10)

project(visualizer)

include_directories(include)
include_directories(thirdparty/CLI11/include)

add_library(${PROJECT_NAME}_lib 
    src/visualizer.cpp)

target_link_libraries(${PROJECT_NAME}_lib common pangolin feature_detector visual_odometry)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} 
    ${PROJECT_NAME}_lib 
    common 
    dataset_reader 
    replayer 
    evaluator
    spdlog::spdlog)
