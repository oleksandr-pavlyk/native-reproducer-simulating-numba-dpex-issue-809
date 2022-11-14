# Reproducer for memory corruption on interop kernel launch by Level-Zero under WSL

## Building

Assuming oneAPI is activated:

```bash
LIB=${LD_LIBRARY_PATH} CXX=icpx cmake . -B build_dir -DCMAKE_INSTALL_PREFIX=.
cmake --build build_dir --target install
```

Building requires `cmake` of version 3.21 or higher (to support integration with DPC++ compiler).
It also requires `level-zero` and `level-zero-dev` packages to be installed. Check with

```bash
dpkg -l | grep level-zero
```


## Running

```bash
./run_example
```

Obversed outputs:

```
(dev_dpctl) user@work:~/repos/wsl_corruption_repro$ ./run_example
Execution device: Intel(R) Graphics [0x9a49]
Execution device driver_version: 1.3.24347
x: 1 1 1 1 1 1 1 1 1 1
y: 2 2 2 2 2 2 2 2 2 2
```

```
(dev_dpctl) opavlyk@opavlyk-mobl:~/repos/wsl_corruption_repro$ SYCL_DEVICE_FILTER=opencl ./run_example
Execution device: Intel(R) Graphics [0x9a49]
Execution device driver_version: 22.39.24347
x: 1 1 1 1 1 1 1 1 1 1
y: 2 2 2 2 2 2 2 2 2 2
```