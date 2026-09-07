# Checked SPECTRA native C ABI library.
cmake_minimum_required(VERSION 3.16)
project(spectra_kernel CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(spectra_kernel SHARED spectra_kernel.cpp)

# AUTO is safe: only enable AVX2 when a configure-time test can execute and the
# host reports AVX2. Cross-compiles or unknown CPUs fall back to scalar.
set(SPECTRA_AVX2 "AUTO" CACHE STRING "AVX2 backend: AUTO, ON, or OFF")
set_property(CACHE SPECTRA_AVX2 PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${SPECTRA_AVX2}" SPECTRA_AVX2_MODE)
set(SPECTRA_USE_AVX2 FALSE)

if(SPECTRA_AVX2_MODE STREQUAL "ON")
  set(SPECTRA_USE_AVX2 TRUE)
elseif(SPECTRA_AVX2_MODE STREQUAL "AUTO")
  include(CheckCXXSourceRuns)
  check_cxx_source_runs([[ 
#if defined(_MSC_VER)
#include <intrin.h>
int main() {
  int r[4] = {0,0,0,0};
  __cpuid(r, 1);
  if (!(r[2] & (1 << 27)) || !(r[2] & (1 << 28))) return 1;
  if ((_xgetbv(0) & 0x6) != 0x6) return 1;
  __cpuidex(r, 7, 0);
  return (r[1] & (1 << 5)) ? 0 : 1;
}
#elif defined(__x86_64__) || defined(__i386__)
int main() {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") ? 0 : 1;
}
#else
int main() { return 1; }
#endif
]] SPECTRA_HOST_HAS_AVX2)
  if(SPECTRA_HOST_HAS_AVX2)
    set(SPECTRA_USE_AVX2 TRUE)
  endif()
elseif(NOT SPECTRA_AVX2_MODE STREQUAL "OFF")
  message(FATAL_ERROR "SPECTRA_AVX2 must be AUTO, ON, or OFF")
endif()

if(MSVC)
  target_compile_options(spectra_kernel PRIVATE /O2)
  if(SPECTRA_USE_AVX2)
    target_compile_options(spectra_kernel PRIVATE /arch:AVX2)
  endif()
else()
  target_compile_options(spectra_kernel PRIVATE -O3 -funroll-loops)
  if(SPECTRA_USE_AVX2)
    target_compile_options(spectra_kernel PRIVATE -mavx2)
  endif()
endif()
message(STATUS "SPECTRA native backend: ${SPECTRA_USE_AVX2} (AVX2=true, scalar=false)")
