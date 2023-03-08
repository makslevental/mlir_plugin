#include "DummyPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define DEBUG_TYPE "dummy-pass"

#include <iostream>

namespace {
    struct DummyPass
        : public PassWrapper<DummyPass, OperationPass<mlir::ModuleOp>> {

        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DummyPass)

        void getDependentDialects(DialectRegistry &registry) const override {
        }
        [[nodiscard]] StringRef getArgument() const final {
            return "dummy-pass";
        }

        void runOnOperation() override {
            getOperation()->setAttr("dummy.dummy", UnitAttr::get(&getContext()));
        }
    };
}// namespace

namespace mlir_plugin {
    void registerDummyPass() {
        PassRegistration<DummyPass>();
    }
}// namespace mlir_plugin