cmake_minimum_required(VERSION 3.28)
project(LiteLanguageModel)

set(CMAKE_CXX_STANDARD 26)

add_subdirectory(TensorLite)

add_executable(LiteLanguageModel main.cpp
        LLM.cpp
        LLM.h)

target_link_libraries(LiteLanguageModel PRIVATE TensorLite)