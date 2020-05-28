cmake_minimum_required(VERSION 3.10)

project(dataset_reader)

include_directories(include)

add_library(${PROJECT_NAME} 
    src/davis240c_reader.cpp 
    include/dataset_reader/davis240c_reader.h)

target_link_libraries(${PROJECT_NAME} common)

add_subdirectory(test)
