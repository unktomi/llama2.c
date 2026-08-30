cmake_minimum_required(VERSION 3.20)
project(llama_cpp_escardo LANGUAGES C CXX ASM)

set(LLAMA_CPP_SOURCE "" CACHE PATH "Path to a llama.cpp source checkout")
if(NOT EXISTS "${LLAMA_CPP_SOURCE}/include/llama.h")
    message(FATAL_ERROR "Set LLAMA_CPP_SOURCE to a llama.cpp source checkout")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_CPP_SOURCE}" llama.cpp-build EXCLUDE_FROM_ALL)

add_executable(run_escardo_gguf run_escardo_gguf.cpp)
target_compile_features(run_escardo_gguf PRIVATE cxx_std_17)
target_compile_options(run_escardo_gguf PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(run_escardo_gguf PRIVATE
    "${LLAMA_CPP_SOURCE}/src"
    "${LLAMA_CPP_SOURCE}/ggml/include"
    "${LLAMA_CPP_SOURCE}/ggml/src"
)
target_link_libraries(run_escardo_gguf PRIVATE llama llama-common)
