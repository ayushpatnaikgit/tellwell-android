cmake_minimum_required(VERSION 3.22)
project(tellwell_llm C CXX)

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

# Speech recognition on the Qualcomm NPU (tellwell_asr.cpp). ONNX Runtime with QNN (libonnxruntime_qnn.so) is opened at
# run time by path, so it is not linked here. Features: kaldi-native-fbank 1.23.1 (Apache 2.0) with kissfft (BSD-3).
file(GLOB KNF_SRC ${CMAKE_SOURCE_DIR}/third_party/kaldi-native-fbank/csrc/*.cc)
add_library(tellwell_asr SHARED
    tellwell_asr.cpp
    ${KNF_SRC}
    ${CMAKE_SOURCE_DIR}/third_party/kissfft/kiss_fft.c
    ${CMAKE_SOURCE_DIR}/third_party/kissfft/kiss_fftr.c)
target_include_directories(tellwell_asr PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/kissfft)
target_link_libraries(tellwell_asr log dl)

# Hindi speech (an optional language): IndicConformer Hindi on the CPU through the same ONNX Runtime, opened at run
# time by path. One model takes audio and returns CTC log-probabilities; its features are computed inside the model.
add_library(tellwell_hindi_asr SHARED tellwell_hindi_asr.cpp)
target_include_directories(tellwell_hindi_asr PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tellwell_hindi_asr log dl)
