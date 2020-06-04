cmake_minimum_required(VERSION 3.10)

project(replayer)

include_directories(include)

add_library(${PROJECT_NAME} 
    src/replayer.cpp)

target_link_libraries(${PROJECT_NAME} common dataset_reader)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(test)