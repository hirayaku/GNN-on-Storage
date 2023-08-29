cmake_minimum_required(VERSION 3.5)
project(GNNoS C CXX)
message(STATUS "Start configuring project ${PROJECT_NAME}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
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
# we need this macro to link torch/lib/libshm.so
macro(append_torchlib_if_found)
  foreach (_arg ${ARGN})
    find_library(${_arg}_LIBRARY ${_arg} PATHS "${TORCH_INSTALL_PREFIX}/lib")
    if(${_arg}_LIBRARY)
      list(APPEND TORCH_LIBRARIES ${${_arg}_LIBRARY})
    else()
      message(WARNING "static library ${${_arg}_LIBRARY} not found.")
    endif()
  endforeach()
endmacro()
append_torchlib_if_found(shm)

# TODO: we should put libfennel under pybind11, not TorchScript operators
add_library(fennel SHARED csrc/fennel.cpp) #  csrc/heap.cpp
set_property(TARGET fennel PROPERTY CXX_STANDARD 14)
target_include_directories(fennel PRIVATE "csrc")
target_link_libraries(fennel PUBLIC torch Python3::Python)

# libxTensor
add_library(xTensor SHARED csrc/ext_tensor.cpp csrc/ext_ops.cpp)
set_property(TARGET xTensor PROPERTY CXX_STANDARD 14)
target_include_directories(xTensor PRIVATE "csrc")
target_link_libraries(xTensor PUBLIC "${TORCH_LIBRARIES}" Python3::Python)

# add_library(flatbuffer SHARED csrc/flat_buffer.cpp)
# set_property(TARGET flatbuffer PROPERTY CXX_STANDARD 14)
# target_link_libraries(flatbuffer PUBLIC torch Python3::Python)
