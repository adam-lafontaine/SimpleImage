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

//#define SIMAGE_NO_WRITE
//#define SIMAGE_NO_RESIZE
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
