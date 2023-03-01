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

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

include(MLIRDetectPythonEnv)
set(MLIR_INCLUDE_TESTS 0)

include(AddMLIRPython)
mlir_configure_python_dev_packages()
mlir_detect_pybind11_install()

####################
# BOILERPLATE ENDS
####################

add_subdirectory(DummyPass)

set(MLIR_PLUGIN_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_plugin.mlir._mlir.")
declare_mlir_python_sources(MLIRPluginPythonSources)

declare_mlir_python_extension(MLIRPluginPythonSources.Extension
        MODULE_NAME _mlir_plugin
        ADD_TO_PARENT MLIRPluginPythonSources
        SOURCES
        MLIRPluginExtension.cpp

        PRIVATE_LINK_LIBS
        DummyPass
        )

set(_source_components
        MLIRPythonSources
        MLIRPythonExtension.Core
        MLIRPythonExtension.RegisterEverything
        MLIRPluginPythonSources
        )

add_mlir_python_common_capi_library(MLIRPluginMLIRAggregateCAPI
        INSTALL_COMPONENT MLIRPluginMLIRPythonModules
        INSTALL_DESTINATION _mlir/_mlir_libs
        OUTPUT_DIRECTORY "${MLIR_PLUGIN_PYTHON_PACKAGES_DIR}/mlir_plugin/_mlir/_mlir_libs"
        RELATIVE_INSTALL_ROOT "../../../../.."
        DECLARED_SOURCES ${_source_components}
        )

add_mlir_python_modules(MLIRPluginPythonModules
        ROOT_PREFIX "${MLIR_PLUGIN_PYTHON_PACKAGES_DIR}/mlir_plugin/_mlir"
        INSTALL_PREFIX "_mlir"
        DECLARED_SOURCES ${_source_components}
        COMMON_CAPI_LINK_LIBS
        MLIRPluginMLIRAggregateCAPI
        )

