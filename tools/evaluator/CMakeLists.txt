cmake_minimum_required(VERSION 3.10)

project(evaluator)

include_directories(include)

add_library(${PROJECT_NAME} 
    src/evaluator.cpp)

target_link_libraries(${PROJECT_NAME} 
    common
    feature_detector
    spdlog::spdlog)
    
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
