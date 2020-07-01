cmake_minimum_required(VERSION 3.10)

project(dataset_reader)

include_directories(include)

add_library(mapped_file
    include/dataset_reader/mapped_file.h
    src/mapped_file.cpp)

add_library(${PROJECT_NAME} 
    src/davis240c_reader.cpp)

target_link_libraries(${PROJECT_NAME} common mapped_file feature_detector)

# FIXME It is not needed to include whole include dir 
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)


add_subdirectory(test)
