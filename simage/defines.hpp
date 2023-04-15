#pragma once

#include <cstddef>
#include <cassert>
#include <cstdint>
#include <cstdlib>

#ifndef IS_LITTLE_ENDIAN
#define IS_LITTLE_ENDIAN 1
#endif

#define SIMAGE_PNG
#define SIMAGE_BMP

//#define SIMAGE_NO_PARALLEL
//#define SIMAGE_NO_FILESYSTEM

//#define SIMAGE_NO_CPP17

//#define RPI_3B_PLUS
//#define JETSON_NANO


#ifdef RPI_3B_PLUS

#define SIMAGE_NO_CPP17

#endif // RPI_3B_PLUS

#ifdef JETSON_NANO

#define SIMAGE_NO_CPP17

#endif // JETSON_NANO


#ifdef SIMAGE_NO_CPP17

#define SIMAGE_NO_PARALLEL
#define SIMAGE_NO_FILESYSTEM

#endif // SIMAGE_NO_CPP17


/*  types.hpp  */

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
class Matrix1D
{
public:
	T* data_ = nullptr;
	u32 length = 0;
};


template <typename T>
class Matrix2D
{
public:
	T* data_ = nullptr;
	u32 width = 0;
	u32 height = 0;
};