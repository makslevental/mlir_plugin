from mlir_plugin import ir
from mlir_plugin import PassManager

ctx = ir.Context()
with ctx:
    uloc = ir.Location.unknown()
    with uloc:
        mod = ir.Module.create()

print("before:")
print(mod)

with ctx:
    pm = PassManager.parse("builtin.module(dummy-pass)")
    pm.run(module=mod)

print("after:")
print(mod)
