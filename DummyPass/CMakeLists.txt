add_mlir_library(DummyPass
        DummyPass.cpp

        DISABLE_INSTALL

        LINK_LIBS PUBLIC
        MLIRFuncDialect
        )
