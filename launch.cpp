#include <CL/sycl.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cstdint>

#include "ze_interop.h"
#include "cl_interop.h"

#ifdef VERBOSE
void log_out(const std::string &s) {
  std::cout << s << std::endl;
}
#else
void log_out(const std::string &) {
  return;
}
#endif

/*
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
 */

using dataT = std::int64_t; 

sycl::event submit_interop_kernel(
  sycl::queue exec_q,
  sycl::kernel krn,
  dataT *x_ptr,
  dataT *y_ptr,
  size_t n,
  const std::vector<sycl::event> &depends = {}
) {
  sycl::event e = 
    exec_q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(depends);

      // Arguments corresponding to X
      cgh.set_arg(0, size_t(0));  // dummy variable
      cgh.set_arg(1, size_t(0));  // dummy variable
      cgh.set_arg(2, static_cast<long long>(n));  // size of the array
      cgh.set_arg(3, static_cast<long long>(8));  // 8 bytes per element
      cgh.set_arg(4, static_cast<void *>(x_ptr));
      cgh.set_arg(5, static_cast<long long>(n));
      cgh.set_arg(6, static_cast<long long>(1));

      // Arguments corresponding to Y
      cgh.set_arg(7 + 0, size_t(0));  // dummy variable
      cgh.set_arg(7 + 1, size_t(0));  // dummy variable
      cgh.set_arg(7 + 2, static_cast<long long>(n));  // size of the array
      cgh.set_arg(7 + 3, static_cast<long long>(8));  // 8 bytes per element
      cgh.set_arg(7 + 4, static_cast<void *>(y_ptr));
      cgh.set_arg(7 + 5, static_cast<long long>(n));
      cgh.set_arg(7 + 6, static_cast<long long>(1));

      cgh.parallel_for(sycl::range<1>(n), krn);
    });

  return e;
}

std::vector<char> read_spirv(void) {
  std::ifstream spirvFile;
  size_t spirvFileSize;

  if (!std::filesystem::exists("./krn.spv")) {
    throw std::runtime_error("Required file './krn.spv' is not found");
  }

  spirvFile.open("./krn.spv", std::ios::binary | std::ios::ate);
  spirvFileSize = std::filesystem::file_size("./krn.spv");
  
  std::vector<char> spirvBuffer(spirvFileSize);
  spirvFile.seekg(0, std::ios::beg);
  spirvFile.read(spirvBuffer.data(), spirvFileSize);

  spirvFile.close();

  log_out("From read_spirv, vector.size: " + std::to_string(spirvBuffer.size()));

  return spirvBuffer;
}

std::string read_name() {
  std::ifstream nameFile;
  std::string name;

  if (!std::filesystem::exists("./krn_name.txt")) {
    throw std::runtime_error("Required file './krn_name.txt' is not found");
  }

  nameFile.open("./krn_name.txt", std::ios::in);
  std::getline(nameFile, name);
  nameFile.close();

  return name;
}

void display_buffer(const std::string &prefix, dataT *data, size_t len) {
  std::cout << prefix << ": ";

  for(size_t i = 0; i < len; ++i) {
    std::cout << data[i] << " ";
  }

  std::cout << std::endl;
}

int main(void) {

  sycl::queue q;
  size_t n = 10;

  std::cout << "Execution device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "Execution device driver_version: " << q.get_device().get_info<sycl::info::device::driver_version>() << std::endl;

  auto il = read_spirv();
  auto name = read_name();

  log_out("Read SPV and names");
  log_out("Size of SPV: " + std::to_string(il.size()));
  log_out("Name of the kernel: " + name);

  dataT *x_ptr = sycl::malloc_device<dataT>(n, q);
  dataT *y_ptr = sycl::malloc_device<dataT>(n, q);

  sycl::event e1_pop = q.fill<std::int64_t>(x_ptr, dataT(1), n);
  sycl::event e2_pop = q.fill<std::int64_t>(y_ptr, dataT(0), n);

  dataT *x_host_ptr = new dataT[n];
  dataT *y_host_ptr = new dataT[n];

  std::unique_ptr<sycl::kernel> krn;

  log_out("Memory allocated");

  if (q.get_backend() == sycl::backend::ext_oneapi_level_zero) {

    log_out("In Level-zero branch");
    auto pr = create_program_from_spirv_ze(
      q.get_context(),
      q.get_device(),
      il.data(), 
      il.size(), 
      ""
    );

    krn = std::make_unique<sycl::kernel>(get_kernel_ze(pr, name.c_str()));


  } else if (q.get_backend() == sycl::backend::opencl) {
    log_out("In OpenCL branch");
    auto pr = create_program_from_spirv_cl(
      q.get_context(),
      q.get_device(),
      il.data(), 
      il.size(), 
      ""
    );

    krn = std::make_unique<sycl::kernel>(get_kernel_cl(pr, name.c_str()));
  } else {
    throw std::runtime_error("Unsupported backend");
  }

  sycl::event e = submit_interop_kernel(q, *krn, x_ptr, y_ptr, n, {e1_pop, e2_pop});

  log_out("Submitted the interop kernel");

  sycl::event e1_copy = q.copy<dataT>(x_ptr, x_host_ptr, n, {e});
  sycl::event e2_copy = q.copy<dataT>(y_ptr, y_host_ptr, n, {e});

  sycl::event::wait({e1_copy, e2_copy});
  sycl::free(x_ptr, q);
  sycl::free(y_ptr, q);

  display_buffer("x", x_host_ptr, n);
  display_buffer("y", y_host_ptr, n);

  bool x_as_expected = true;
  bool y_as_expected = true;

  for(size_t i = 0; x_as_expected && y_as_expected && i < n; ++i) {
    x_as_expected = x_host_ptr[i] == 1;
    y_as_expected = y_host_ptr[i] == 2;
  }

  delete[] x_host_ptr;
  delete[] y_host_ptr;

  return 0;
}
