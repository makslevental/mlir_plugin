import contextlib
import ctypes
import sys


@contextlib.contextmanager
def dl_open_guard():
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(old_flags | ctypes.RTLD_GLOBAL)
    yield
    sys.setdlopenflags(old_flags)


with dl_open_guard():
    # noinspection PyUnresolvedReferences
    from ._mlir import ir
    from ._mlir.passmanager import PassManager

# noinspection PyUnresolvedReferences
from ._mlir._mlir_libs import _mlir_plugin
