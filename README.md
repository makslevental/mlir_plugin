# MLIR Plugin Passes

A demo for how to register your own passes, in addition to those bundled with MLIR; effectively this functions
as a pass plugin functionality.

# Building

```shell
$ pip install requirements.txt
$ mkdir build && pushd build
$ cmake .. -DPython3_EXECUTABLE=$(which python) -DCMAKE_INSTALL_PREFIX=$PWD/../mlir_plugin && \
  cmake --build . --target install && \
  popd
```

# Running

```shell
$ python demo.py

before:
module {
}

after:
module attributes {dummy.dummy} {
}
```