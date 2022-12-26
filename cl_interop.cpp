#include <CL/sycl.hpp>
#if __has_include(<CL/sycl/backend/opencl.hpp>)
#   include <CL/sycl/backend/opencl.hpp>
#else
#   include <sycl/backend/opencl.hpp>
#endif
#include <vector>
#include "cl_interop.h"

using namespace sycl;
constexpr backend cl_be = backend::opencl;

kernel_bundle<bundle_state::executable> create_program_from_spirv_cl(
    const context &ctx,
    const device &dev,
    const void *IL,
    size_t il_length, 
    const char *CompileOpts
) {
    backend_traits<cl_be>::return_type<context> clContext;
    clContext = get_native<cl_be>(ctx);

    cl_int create_err_code = CL_SUCCESS;
    cl_program clProgram =
        clCreateProgramWithIL(clContext, IL, il_length, &create_err_code);

    if (create_err_code != CL_SUCCESS) {
        throw std::runtime_error(std::to_string(create_err_code));
    }

    backend_traits<cl_be>::return_type<device> clDevice;
    clDevice = get_native<cl_be>(dev);

    cl_int build_status =
        clBuildProgram(clProgram, 1, &clDevice, CompileOpts, nullptr, nullptr);

    using ekbTy = kernel_bundle<bundle_state::executable>;
    ekbTy kb =
        make_kernel_bundle<cl_be, bundle_state::executable>(clProgram, ctx);
    return kb;
}

kernel get_kernel_cl(
    const kernel_bundle<bundle_state::executable> &kb,
    const char *kernel_name
) {

    std::vector<cl_program> oclKB = get_native<cl_be>(kb);
    bool found = false;
    cl_kernel ocl_kernel_from_kb;
    for (auto &cl_pr : oclKB) {
        cl_int create_kernel_err_code = CL_SUCCESS;
        cl_kernel try_kern =
            clCreateKernel(cl_pr, kernel_name, &create_kernel_err_code);
        if (create_kernel_err_code == CL_SUCCESS) {
            found = true;
            ocl_kernel_from_kb = try_kern;
            break;
        }
    }
    if (found) {
        try {
            context ctx = kb.get_context();

            kernel interop_kernel = make_kernel<cl_be>(ocl_kernel_from_kb, ctx);

            return interop_kernel;
        } catch (std::exception const &e) {
            throw e;
        }
    }
    else {
        throw std::runtime_error("Kernel " + std::string(kernel_name) + " not found.");
    }
}
