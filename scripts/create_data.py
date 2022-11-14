import dpctl
import dpctl.tensor as dpt
import numba_dpex

x = dpt.ones(10, dtype='i8')
y = dpt.empty_like(x)

assert x.sycl_device == y.sycl_device

@numba_dpex.kernel
def foo(X, Y):
    i = numba_dpex.get_global_id(0)
    Y[i] = 2*X[i]


foo_c = foo[10, numba_dpex.DEFAULT_LOCAL_SIZE]

foo_c(x, y)

krn = list(foo_c.definitions.items())[0][1][1]

with open("krn.spv", "bw") as fh:
    fh.write(krn.spirv_bc)

with open("krn_name.txt", "w") as fh:
    fh.write(krn.kernel.get_function_name() + "\n")

