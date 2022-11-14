#pragma once

#include <CL/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

sycl::kernel_bundle<sycl::bundle_state::executable> 
create_program_from_spirv_ze(
    const sycl::context &,
    const sycl::device &,
    const void *,
    size_t, 
    const char *
);

sycl::kernel 
get_kernel_ze(
    const sycl::kernel_bundle<sycl::bundle_state::executable> &,
    const char *
);