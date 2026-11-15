cmake_minimum_required(VERSION 3.22)
project(tellwell_llm CXX)

set(CMAKE_CXX_STANDARD 17)
set(LIBS ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})

# Prebuilt llama.cpp (Snapdragon build with CPU, OpenCL and Hexagon backends), packaged from jniLibs.
foreach(lib llama ggml ggml-base)
    add_library(${lib} SHARED IMPORTED)
    set_target_properties(${lib} PROPERTIES IMPORTED_LOCATION ${LIBS}/lib${lib}.so)
endforeach()

add_library(tellwell_llm SHARED tellwell_llm.cpp)
target_include_directories(tellwell_llm PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tellwell_llm llama ggml ggml-base log)
