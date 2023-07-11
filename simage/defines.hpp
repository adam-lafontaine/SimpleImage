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
//#define SIMAGE_NO_CUDA

// Disable API profiling
//#define SIMAGE_NO_PROFILE

#ifdef SIMAGE_NO_PARALLEL

constexpr u32 N_THREADS = 1;

#else

constexpr u32 N_THREADS = 1;

#endif //!SIMAGE_NO_PARALLEL


// Force enable CUDA
#ifdef ENABLE_CUDA
#ifdef SIMAGE_NO_CUDA
#undef SIMAGE_NO_CUDA
#endif
#endif


// Force enable API profiling
#ifdef ENABLE_PROFILE
#ifdef SIMAGE_NO_PROFILE
#undef SIMAGE_NO_PROFILE
#endif
#endif