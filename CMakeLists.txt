cmake_minimum_required(VERSION 3.5)
project(GNNoS C CXX)
message(STATUS "Start configuring project ${PROJECT_NAME}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# configure torch
if(NOT DEFINED TORCH_CMAKE_DIR)
  if(DEFINED TORCH_ROOT_DIR)
    set(TORCH_CMAKE_DIR "${TORCH_ROOT_DIR}/share/cmake")
  elseif(DEFINED ENV{TORCH_ROOT_DIR})
    set(TORCH_CMAKE_DIR "$ENV{TORCH_ROOT_DIR}/share/cmake")
  else()
    execute_process(
      COMMAND
      python -c "import torch;print(torch.utils.cmake_prefix_path)"
      OUTPUT_VARIABLE TORCH_CMAKE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
  endif()
endif()

list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_DIR}")
find_package(Torch REQUIRED)
set(TORCH_CMAKE_DIR "${TORCH_CMAKE_DIR}" CACHE PATH "Torch CMake config directory")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# build library
add_library(gnnos SHARED
            csrc/tensor_io.cpp
            csrc/graph_io.cpp
            csrc/graph_partition.cpp
            )
set_property(TARGET gnnos PROPERTY CXX_STANDARD 14)
target_include_directories(gnnos PRIVATE "include")
target_link_libraries(gnnos PUBLIC "${TORCH_LIBRARIES}")

# build executable - test purposes
file(GLOB CPP_FILES csrc/main.cpp)
add_executable(dev-test ${CPP_FILES})
set_property(TARGET dev-test PROPERTY CXX_STANDARD 14)
target_include_directories(dev-test PRIVATE "include")
target_link_libraries(dev-test PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(dev-test PRIVATE gnnos)

# configure python3 for building the extension
if(NOT DEFINED PYTHON_CXX_FLAGS)
  execute_process(
    COMMAND
    python3-config --cflags
    RESULT_VARIABLE RETCODE
    OUTPUT_VARIABLE PYTHON_C_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE PYTHON_CONFIG_ERROR
  )
  if(NOT RETCODE EQUAL 0)
    message(FATAL_ERROR "Failed to python3-config ${PYTHON_CONFIG_ERROR}")
  endif()
  set(PYTHON_CXX_FLAGS ${PYTHON_C_FLAGS} CACHE STRING "Flags for building with Python")
endif()

add_library(gnnos-python SHARED csrc/extension.cpp)
set_property(TARGET gnnos-python PROPERTY CXX_STANDARD 14)
separate_arguments(PYTHON_CXX_FLAGS) # separate string into lists
target_compile_options(gnnos-python PRIVATE ${PYTHON_CXX_FLAGS})
target_link_libraries(gnnos-python PUBLIC "${TORCH_LIBRARIES}")
target_link_libraries(gnnos-python PUBLIC gnnos)

