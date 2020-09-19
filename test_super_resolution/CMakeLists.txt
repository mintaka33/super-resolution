cmake_minimum_required (VERSION 2.8.12)

project(testsr)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "-O0 -g")

find_package(OpenCV REQUIRED)
find_package(InferenceEngine REQUIRED)

add_executable(testsr main.cpp)

target_link_libraries(testsr PRIVATE ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES})