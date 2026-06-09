# Build the SPECTRA fused sparse ternary GEMV kernel (Phase 12 / section 26.3).
#
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#
# Produces a shared library loadable from Python via ctypes (see
# deploy/bitnet_cpp_adapter). Enable AVX2 on legacy x86 for the SIMD fast path.
cmake_minimum_required(VERSION 3.16)
project(spectra_kernel CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(spectra_kernel SHARED spectra_kernel.cpp)

if(NOT MSVC)
  target_compile_options(spectra_kernel PRIVATE -O3 -funroll-loops)
  # Enable AVX2 for the SIMD path (legacy x86 eval box). Comment out if the CPU
  # lacks AVX2; the scalar reference path remains correct.
  target_compile_options(spectra_kernel PRIVATE -mavx2)
else()
  target_compile_options(spectra_kernel PRIVATE /O2 /arch:AVX2)
endif()
