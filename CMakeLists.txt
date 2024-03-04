cmake_minimum_required(VERSION 3.28)
project(LiteLanguageModel CUDA CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CUDA_STANDARD 14)

add_subdirectory(TensorLite)

add_executable(LiteLanguageModel main.cpp
        LLM.cpp
        LLM.h)

target_link_libraries(LiteLanguageModel PRIVATE TensorLite)

set_target_properties(LiteLanguageModel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
