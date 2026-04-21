cmake_minimum_required(VERSION 3.20)

option(LLMBoost_SKIP_CUDA "Skip CUDA kernel build for CI" OFF)

if(LLMBoost_SKIP_CUDA)
  project(LLMCompilerPasses C CXX)
else()
  project(LLMCompilerPasses C CXX CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 80 86)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT TARGET zstd::libzstd_shared)
  add_library(zstd::libzstd_shared SHARED IMPORTED)
  set_target_properties(zstd::libzstd_shared PROPERTIES
      IMPORTED_LOCATION "/usr/lib/x86_64-linux-gnu/libzstd.so.1")
endif()

if(NOT TARGET Terminfo::terminfo)
  add_library(Terminfo::terminfo INTERFACE IMPORTED)
  set_target_properties(Terminfo::terminfo PROPERTIES
      INTERFACE_LINK_LIBRARIES "tinfo")
endif()

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

if(NOT LLMBoost_SKIP_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)

if(NOT LLMBoost_SKIP_CUDA)
  add_subdirectory(kernels)
endif()
