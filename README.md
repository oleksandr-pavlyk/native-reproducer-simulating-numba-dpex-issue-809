# Reproducer for memory corruption on interop kernel launch by Level-Zero under WSL

## Building

Assuming oneAPI is activated:

```bash
LIB=${LD_LIBRARY_PATH} CXX=icpx cmake . -B build_dir -DCMAKE_INSTALL_PREFIX=.
cmake --build build_dir --target install
```

## Running

```bash
./run_example
```