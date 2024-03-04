cmake_minimum_required(VERSION 3.28)
project(LiteLanguageModel)

set(CMAKE_CXX_STANDARD 26)

add_executable(LiteLanguageModel main.cpp
        LLM.cpp
        LLM.h)

add_subdirectory(TensorLite)

target_link_libraries(LiteLanguageModel PRIVATE TensorLite)