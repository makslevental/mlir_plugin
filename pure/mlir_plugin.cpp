#include <iostream>

#include "DummyPass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"


using namespace mlir;

// no clue why but without this i get a missing symbol error
namespace llvm {
    int DisableABIBreakingChecks = 1;
    int EnableABIBreakingChecks = 0;
}// namespace llvm

void runPass() {
    DialectRegistry registry;
    registerAllDialects(registry);
    registerAllPasses();
    mlir_plugin::registerDummyPass();
    MLIRContext context(registry);
    StringRef s = R"MLIR(
            func.func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
              linalg.matmul ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>)
                                 outs(%C : memref<?x?xf32>)
              return
            }
            )MLIR";
    ParserConfig config(&context);
    auto module = mlir::parseSourceString<ModuleOp>(s, config);
    std::cerr << "before:\n";
    module->dump();
    std::cerr << "\n";
    PassManager pm(&context, module.get()->getName().getStringRef());
    (void) parsePassPipeline("func.func(convert-linalg-to-loops),dummy-pass", pm);
    pm.dump();
    if (mlir::failed(pm.run(*module)))
        std::cerr << "wtfbbq";
    std::cerr << "\nafter:\n";
    module->dump();
}

int main() {
    runPass();
}
