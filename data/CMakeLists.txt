cmake_minimum_required(VERSION 3.5)
project(GNNoS C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
message(STATUS "Start configuring project ${PROJECT_NAME}")

find_package(OpenMP REQUIRED)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
find_package(Python3 REQUIRED COMPONENTS Development)

# configure torch
if(NOT DEFINED ENV{TORCH_INSTALL_PREFIX})
  execute_process(
    COMMAND
    python3 -c "import torch;print(torch.__path__[0])"
    OUTPUT_VARIABLE TORCH_INSTALL_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  set(ENV{TORCH_INSTALL_PREFIX} "${TORCH_INSTALL_PREFIX}")
endif()
list(APPEND CMAKE_PREFIX_PATH "$ENV{TORCH_INSTALL_PREFIX}/share/cmake/")

find_package(OpenMP REQUIRED)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
find_package(Python3 REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

# TODO: we should put libfennel under pybind11, not TorchScript operators
add_library(fennel SHARED csrc/fennel_ops.cpp csrc/fennel.cpp csrc/fennel_weighted.cpp)
set_property(TARGET fennel PROPERTY CXX_STANDARD 14)
target_include_directories(fennel PRIVATE "csrc")
target_link_libraries(fennel PUBLIC torch Python3::Python)

# libxTensor
add_library(xTensor SHARED csrc/ext_tensor.cpp csrc/ext_ops.cpp)
set_property(TARGET xTensor PROPERTY CXX_STANDARD 14)
target_include_directories(xTensor PRIVATE "csrc")
target_link_libraries(xTensor PUBLIC "${TORCH_LIBRARIES}" Python3::Python)

# libiplasma
option(MI_OVERRIDE OFF)
option(MI_BUILD_STATIC OFF)
option(MI_BUILD_TESTS OFF)
option(MI_BUILD_OBJECT OFF)
add_subdirectory(mimalloc)
add_library(
  iplasma SHARED
  csrc/iplasma_common.cpp
  csrc/iplasma_io.cpp
  csrc/iplasma_fling.cpp
  # csrc/iplasma_malloc.cpp
  # csrc/iplasma_buffer.cpp
)
set_property(TARGET iplasma PROPERTY CXX_STANDARD 14)
target_include_directories(iplasma PRIVATE ./mimalloc/include)
target_link_libraries(iplasma PUBLIC mimalloc)
target_link_libraries(iplasma PUBLIC torch Python3::Python)

# test iplasma with mimalloc
add_executable(iplasma_test csrc/iplasma_test.cpp)
set_property(TARGET iplasma_test PROPERTY CXX_STANDARD 14)
target_include_directories(iplasma_test PRIVATE csrc)
target_link_libraries(iplasma_test PUBLIC mimalloc)

