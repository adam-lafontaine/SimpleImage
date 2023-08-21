#pragma once

#include "src/util/types.hpp"

#include <cstddef>
#include <cassert>
#include <cstdlib>

#ifndef IS_LITTLE_ENDIAN
#define IS_LITTLE_ENDIAN 1
#endif

// Support .png image files
#define SIMAGE_PNG

// Support .bmp image files
#define SIMAGE_BMP

// Disable multithreaded image processing
//#define SIMAGE_NO_PARALLEL

// Disable std::filesystem file paths as an alternative to const char*
// Uses std::string instead
//#define SIMAGE_NO_FILESYSTEM

// Disable USB camera support
//#define SIMAGE_NO_USB_CAMERA

// Disable CUDA GPU support
#ifndef SIMAGE_NO_CUDA
#define SIMAGE_NO_CUDA
#endif

// Force enable CUDA
#ifdef SIMAGE_ENABLE_CUDA
#ifdef SIMAGE_NO_CUDA
#undef SIMAGE_NO_CUDA
#endif
#endif


// Disable SIMD support
#define SIMAGE_NO_SIMD

#ifndef SIMAGE_NO_SIMD

//#define SIMD_INTEL_128
#define SIMD_INTEL_256

#endif