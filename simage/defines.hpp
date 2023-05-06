#pragma once

#include <cstddef>
#include <cassert>
#include <cstdint>
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


using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using f32 = float;
using f64 = double;
using i8 = int8_t;
using i16 = short;
using i32 = int32_t;
using cstr = const char*;


#ifndef SIMAGE_NO_CUDA

// CUDA supports 16 bit half floats
using f16 = u16;

#endif

#ifdef SIMAGE_NO_PARALLEL

constexpr u32 N_THREADS = 1;

#else

constexpr u32 N_THREADS = 16;

#endif //!SIMAGE_NO_PARALLEL


template <typename T>
class Point2D
{
public:
	T x;
	T y;
};

using Point2Du32 = Point2D<u32>;
using Point2Df32 = Point2D<f32>;


// region of interest in an image
class Range2Du32
{
public:
	u32 x_begin;
	u32 x_end;  // one past last x
	u32 y_begin;
	u32 y_end;   // one past last y
};


inline Range2Du32 make_range(u32 width, u32 height)
{
	Range2Du32 r{};

	r.x_begin = 0;
	r.y_begin = 0;
	r.x_end = width;
	r.y_end = height;

	return r;
}


template <class T>
inline Range2Du32 make_range(T const& c)
{
	return make_range(c.width, c.height);
}


template <typename T>
class Matrix2D
{
public:
	T* data_ = nullptr;
	u32 width = 0;
	u32 height = 0;
};