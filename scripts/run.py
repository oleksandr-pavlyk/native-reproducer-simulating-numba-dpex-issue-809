import ctypes
import dpctl
import dpctl.tensor as dpt
import dpctl.program as dpp
import os.path
import numpy as np

def get_usm_ndarray_args(x):
    assert x.ndim == 1
    return [
        ctypes.c_size_t(0), # dummy
        ctypes.c_size_t(0), # dummy
        ctypes.c_longlong(x.size), # num. elements in x
        ctypes.c_longlong(x.itemsize), # size of single data element
        x.usm_data,  # USMMemory object underlying array
        ctypes.c_longlong(x.shape[0]),
        ctypes.c_longlong(x.strides[0]),
    ]


x = dpt.ones(10, dtype='i8')
y = dpt.empty_like(x)

with open(os.path.join("..", "krn.spv"), "br") as fh:
    il = fh.read()
with open(os.path.join("..", "krn_name.txt"), "r") as fh:
    name = fh.readline().strip()

q = x.sycl_queue
pr = dpp.create_program_from_spirv(q, il, "")
krn = pr.get_sycl_kernel(name)

args = []
args += get_usm_ndarray_args(x)
args += get_usm_ndarray_args(y)


e = q.submit(krn, args, [10])
e.wait()

x_np = dpt.asnumpy(x)
y_np = dpt.asnumpy(y)

print(f"Ran on : {q.sycl_device.name} with driver version: {q.sycl_device.driver_version}")
print(x_np)
print(y_np)

if (np.array_equal(x_np, np.ones(10, 'i8')) and np.array_equal(y_np, np.full(10, 2, 'i8'))):
    print("Not corruption detected")
else:
    print("Data CORRUPTION detected")
