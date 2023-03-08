from textwrap import dedent

from mlir_plugin import ir
from mlir_plugin import PassManager

ctx = ir.Context()
with ctx:
    uloc = ir.Location.unknown()
    with uloc:
        mod = ir.Module.parse(
            dedent(
                """\
        func.func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
          linalg.matmul ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>)
                             outs(%C : memref<?x?xf32>)
          return
        }
        """
            )
        )

print("before:")
print(mod)

with ctx:
    pm = PassManager.parse("builtin.module(func.func(convert-linalg-to-loops),dummy-pass)")
    pm.run(mod.operation)

print("after:")
print(mod)
