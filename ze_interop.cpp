#include <CL/sycl.hpp>
#include "level_zero/ze_api.h" /* Level Zero headers */
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <vector>
#include "ze_interop.h"

using namespace sycl;

constexpr backend ze_be = backend::ext_oneapi_level_zero;

#define CodeStringSuffix(code)                                                 \
    std::string(" (code=") + std::to_string(static_cast<int>(code)) + ")"

#define EnumCaseString(code)                                                   \
    case code:                                                                 \
        return std::string(#code) + CodeStringSuffix(code)

std::string _GetErrorCode_ze_impl(ze_result_t code)
{
    switch (code) {
        EnumCaseString(ZE_RESULT_ERROR_UNINITIALIZED);
        EnumCaseString(ZE_RESULT_ERROR_DEVICE_LOST);
        EnumCaseString(ZE_RESULT_ERROR_INVALID_NULL_HANDLE);
        EnumCaseString(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
        EnumCaseString(ZE_RESULT_ERROR_INVALID_ENUMERATION);
        EnumCaseString(ZE_RESULT_ERROR_INVALID_NATIVE_BINARY);
        EnumCaseString(ZE_RESULT_ERROR_INVALID_SIZE);
        EnumCaseString(ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY);
        EnumCaseString(ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY);
        EnumCaseString(ZE_RESULT_ERROR_MODULE_BUILD_FAILURE);
        EnumCaseString(ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED);
    default:
        return "<< UNRECOGNIZED ZE_RESULT_T CODE >> " + CodeStringSuffix(code);
    }
}


kernel_bundle<bundle_state::executable> create_program_from_spirv_ze(
    const context &SyclCtx,
    const device &SyclDev,
    const void *IL,
    size_t il_length, 
    const char *CompileOpts
)
{
    backend_traits<ze_be>::return_type<context> ZeContext;
    ZeContext = get_native<ze_be>(SyclCtx);

    backend_traits<ze_be>::return_type<device> ZeDevice;
    ZeDevice = get_native<ze_be>(SyclDev);

    // Specialization constants are not supported by DPCTL at the moment
    ze_module_constants_t ZeSpecConstants = {};
    ZeSpecConstants.numConstants = 0;

    // Populate the Level Zero module descriptions
    ze_module_desc_t ZeModuleDesc = {};
    ZeModuleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    ZeModuleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    ZeModuleDesc.inputSize = il_length;
    ZeModuleDesc.pInputModule = (uint8_t *)IL;
    ZeModuleDesc.pBuildFlags = CompileOpts;
    ZeModuleDesc.pConstants = &ZeSpecConstants;

    ze_module_handle_t ZeModule;

    auto ret_code = zeModuleCreate(ZeContext, ZeDevice, &ZeModuleDesc,
                                   &ZeModule, nullptr);
    if (ret_code != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(
            "Module creation failed " +
            _GetErrorCode_ze_impl(ret_code));
    }

    auto kb = make_kernel_bundle<ze_be, bundle_state::executable>(
        {ZeModule, ext::oneapi::level_zero::ownership::keep}, SyclCtx);

    return kb;
}

kernel get_kernel_ze(
    const kernel_bundle<bundle_state::executable> &kb,
    const char *kernel_name)
{
    auto ZeKernelBundle = sycl::get_native<ze_be>(kb);
    bool found = false;

    // Populate the Level Zero kernel descriptions
    ze_kernel_desc_t ZeKernelDescr = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr,
                                      0, // flags
                                      kernel_name};

    std::unique_ptr<sycl::kernel> syclInteropKern_ptr;
    ze_kernel_handle_t ZeKern;
    for (auto &ZeM : ZeKernelBundle) {
        ze_result_t ze_status = zeKernelCreate(ZeM, &ZeKernelDescr, &ZeKern);

        if (ze_status == ZE_RESULT_SUCCESS) {
            found = true;
            auto ctx = kb.get_context();
            auto k = make_kernel<ze_be>(
                {kb, ZeKern, ext::oneapi::level_zero::ownership::keep}, ctx);
            syclInteropKern_ptr = std::unique_ptr<kernel>(new kernel(k));
            break;
        }
        else {
            if (ze_status != ZE_RESULT_ERROR_INVALID_KERNEL_NAME) {
                throw std::runtime_error("zeKernelCreate failed: " +
                                  _GetErrorCode_ze_impl(ze_status));
            }
        }
    }

    if (found) {
        return *syclInteropKern_ptr;
    }
    else {
        throw std::runtime_error("Kernel named " + std::string(kernel_name) +
                          " could not be found.");
    }
}
