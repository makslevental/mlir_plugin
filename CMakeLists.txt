####################
# BOILERPLATE STARTS
####################

cmake_minimum_required(VERSION 3.13.4)

if (POLICY CMP0068)
    cmake_policy(SET CMP0068 NEW)
    set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif ()

if (POLICY CMP0075)
    cmake_policy(SET CMP0075 NEW)
endif ()

if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif ()

if (POLICY CMP0116)
    cmake_policy(SET CMP0116 NEW)
endif ()

project(mlir_plugin LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

set(LLVM_INSTALL_DIR "" CACHE STRING "Path to LLVM install dir")
if (LLVM_INSTALL_DIR STREQUAL "")
    execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/llvm_url.py" --llvm-url OUTPUT_VARIABLE LLVM_URL)
    include(FetchContent)
    FetchContent_Populate(
            MLIR
            URL "${LLVM_URL}"
            SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm_install"
    )
    list(APPEND CMAKE_PREFIX_PATH)
    set(LLVM_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm_install")
endif ()
message(STATUS "LLVM_INSTALL_DIR: ${LLVM_INSTALL_DIR}")
list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}")
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

####################
# BOILERPLATE ENDS
####################

add_subdirectory(python_ext)
add_subdirectory(pure)