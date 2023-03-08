####################
# BOILERPLATE STARTS
####################

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

include(MLIRDetectPythonEnv)
set(MLIR_INCLUDE_TESTS 0)

include(AddMLIRPython)
mlir_configure_python_dev_packages()
mlir_detect_pybind11_install()

set(MLIR_PLUGIN_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")

####################
# BOILERPLATE ENDS
####################

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_plugin.mlir._mlir.")
declare_mlir_python_sources(MLIRPluginPythonSources)
add_subdirectory(DummyPass)
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
        INSTALL_DESTINATION python_ext/mlir_plugin/_mlir/_mlir_libs
        OUTPUT_DIRECTORY "${MLIR_PLUGIN_PYTHON_PACKAGES_DIR}/python_ext/mlir_plugin/_mlir/_mlir_libs"
        RELATIVE_INSTALL_ROOT "../../../../../.."
        DECLARED_SOURCES ${_source_components}
        )

add_mlir_python_modules(MLIRPluginPythonModules
        ROOT_PREFIX "${MLIR_PLUGIN_PYTHON_PACKAGES_DIR}/python_ext/mlir_plugin/_mlir"
        INSTALL_PREFIX "python_ext/mlir_plugin/_mlir"
        DECLARED_SOURCES ${_source_components}
        COMMON_CAPI_LINK_LIBS
        MLIRPluginMLIRAggregateCAPI
        )

