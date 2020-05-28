cmake_minimum_required(VERSION 3.10)

project(common)

include_directories(thirdparty/Sophus)

add_library(${PROJECT_NAME} INTERFACE)

target_link_libraries(${PROJECT_NAME} INTERFACE OpenCV)
target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR}/include)