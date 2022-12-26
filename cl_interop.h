#pragma once

#include <CL/sycl.hpp>
#if __has_include(<CL/sycl/backend/opencl.hpp>)
#   include <CL/sycl/backend/opencl.hpp>
#else
#   include <sycl/backend/opencl.hpp>
#endif

sycl::kernel_bundle<sycl::bundle_state::executable> 
create_program_from_spirv_cl(
    const sycl::context &,
    const sycl::device &,
    const void *,
    size_t, 
    const char *
);

sycl::kernel 
get_kernel_cl(
    const sycl::kernel_bundle<sycl::bundle_state::executable> &,
    const char *
);
