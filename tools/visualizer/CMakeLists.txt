cmake_minimum_required(VERSION 3.10)

project(visualizer)

include_directories(include)
include_directories(thirdparty/CLI11/include)

add_executable(${PROJECT_NAME} 
    src/visualizer.cpp)

target_link_libraries(${PROJECT_NAME} common replayer pangolin)
