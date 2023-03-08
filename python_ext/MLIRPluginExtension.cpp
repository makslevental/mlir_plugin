#include "mlir/Pass/PassRegistry.h"
#include <pybind11/pybind11.h>

#include "DummyPass/DummyPass.h"

#include <iostream>

using namespace mlir;

PYBIND11_MODULE(_mlir_plugin, m) {
    mlir_plugin::registerDummyPass();
    m.def("print_help", []() -> std::string {
        PassPipelineCLParser passPipeline("", "Compiler passes to run", "p");
        std::string dummy = "dummy";
        std::string help = "--help";
        char *argv[] = {dummy.data(), help.data()};
        llvm::cl::ParseCommandLineOptions(2, argv, "");
    });
}
