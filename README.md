# MLIR Plugin Passes

Two ways to build "plugin passes" for MLIR.

## Building

```shell
$ pip install -r python_ext/requirements.txt
$ mkdir build && pushd build
$ cmake .. -DPython3_EXECUTABLE=$(which python) -DCMAKE_INSTALL_PREFIX=$PWD/..
$ cmake --build . --target install && popd
```

This will install both versions of the "plugin".

Note that this will download a copy of LLVM from my
personal [release repo](https://github.com/makslevental/llvm-releases).
If you already have LLVM installed somewhere, you can add `-DLLVM_INSTALL_DIR=<where your LLVM is installed>` to `cmake`
and it'll skip the download. Note your LLVM distribution will need to have been compiled
with `-DMLIR_ENABLE_BINDINGS_PYTHON=ON` at minimum.
See [As a standalone shared library](#as-a-standalone-shared-library) for fancier expectations/requirements on the LLVM
distribution.

# The two paths

## As a Python extension

See [python_ext](./python_ext).

This really isn't a plugin but just a quick way to compile something that includes one of your own passes *and* uses
upstream passes without compiling LLVM/MLIR from source (i.e., using static `.a` libraries).

## As a standalone shared library

See [pure](./pure).

This is much closer to a real plugin and depends on having a distribution of LLVM compiled with symbols exported; i.e.,
your LLVM/MLIR distribution should have been compiled with

```
-DCMAKE_C_VISIBILITY_PRESET=default
-DCMAKE_CXX_VISIBILITY_PRESET=default
-DCMAKE_VISIBILITY_INLINES_HIDDEN=0
```

Again, since the included `CMakeLists.txt` downloads LLVM from my repo (which was compiled as such), everything should
work by default (but YMMV).

# Running

The python extension is straightforward:

```shell
$ python python_ext/demo.py
```

The "pure" plugin needs to preload an MLIR "aggregate" library; if you're on Mac OS:

```shell
$ LIB_MLIR_C_PATH=$PWD/llvm_install/lib/libMLIR-C.dylib python ./pure/demo.py
```

while if you're on Linux:

```shell
$ LIB_MLIR_C_PATH=$PWD/llvm_install/lib/libMLIR-C.so python ./pure/demo.py
```

# Expected results

Both paths will run `convert-linalg-to-loops` and `dummy-pass` (note the `dummy.dummy` annotation on the module):

```mlir
// before:
module {
  func.func @matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    linalg.matmul ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg2 : memref<?x?xf32>)
    return
  }
}

// after:
module attributes {dummy.dummy} {
  func.func @matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %dim_1 = memref.dim %arg1, %c1 : memref<?x?xf32>
    scf.for %arg3 = %c0 to %dim step %c1 {
      scf.for %arg4 = %c0 to %dim_1 step %c1 {
        scf.for %arg5 = %c0 to %dim_0 step %c1 {
          %0 = memref.load %arg0[%arg3, %arg5] : memref<?x?xf32>
          %1 = memref.load %arg1[%arg5, %arg4] : memref<?x?xf32>
          %2 = memref.load %arg2[%arg3, %arg4] : memref<?x?xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %arg2[%arg3, %arg4] : memref<?x?xf32>
        }
      }
    }
    return
  }
}
```
