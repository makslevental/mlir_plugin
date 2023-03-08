import ctypes
from _ctypes import dlopen
import os
import platform
from pathlib import Path

if platform.system() == "Darwin":
    shlib_ext = "dylib"
elif platform.system() == "Linux":
    shlib_ext = "so"
else:
    raise NotImplementedError(f"unknown platform {platform.system()}")

lib_mlir_c_path = os.environ.get("LIB_MLIR_C_PATH")
if lib_mlir_c_path is None:
    lib_mlir_c_path = (
        Path(".").absolute().parent / f"llvm_install/lib/libMLIR-C.{shlib_ext}"
    )
else:
    lib_mlir_c_path = Path(lib_mlir_c_path)

assert lib_mlir_c_path.exists(), lib_mlir_c_path


lib_mlir_plugin_path = Path(".").absolute() / f"libmlir_plugin.{shlib_ext}"
assert lib_mlir_plugin_path.exists()


dlopen(lib_mlir_c_path, ctypes.RTLD_GLOBAL)
mlir_plugin = ctypes.CDLL(lib_mlir_plugin_path)
mlir_plugin.main()
