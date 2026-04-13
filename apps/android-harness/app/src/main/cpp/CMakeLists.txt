cmake_minimum_required(VERSION 3.22.1)

project(zkgpu_android_bridge)

add_library(
    zkgpu_android_bridge
    SHARED
    zkgpu_android_bridge.cpp
)

find_library(log-lib log)
find_library(dl-lib dl)

target_link_libraries(
    zkgpu_android_bridge
    PRIVATE
    ${log-lib}
    ${dl-lib}
)
