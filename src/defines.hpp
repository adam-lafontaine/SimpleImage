#pragma once

#include <cstddef>
#include <cassert>
#include <cstdint>
#include <cstdlib>

//#define IS_BIG_ENDIAN (!*(unsigned char *)&(uint16_t){1})

#define IS_BIG_ENDIAN 1

#define SIMAGE_PNG
#define SIMAGE_BMP

//#define SIMAGE_NO_WRITE
//#define SIMAGE_NO_RESIZE
//#define SIMAGE_NO_PARALLEL
//#define SIMAGE_NO_FILESYSTEM

//#define SIMAGE_NO_CPP17

//#define RPI_3B_PLUS
//#define JETSON_NANO


#define SIMD_INTEL_256
//#define SIMD_INTEL_128
//#define SIMD_ARM_NEON


#ifdef RPI_3B_PLUS

#define SIMAGE_NO_CPP17
#define SIMD_ARM_NEON

#endif // RPI_3B_PLUS

#ifdef JETSON_NANO

#define SIMAGE_NO_CPP17
#define SIMD_ARM_NEON

#endif // JETSON_NANO


#ifdef SIMAGE_NO_CPP17

#define SIMAGE_NO_PARALLEL
#define SIMAGE_NO_FILESYSTEM

#endif // SIMAGE_NO_CPP17


/*  types.hpp  */

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using r32 = float;
using r64 = double;
using i8 = int8_t;
using i32 = int32_t;


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
using Point2Dr32 = Point2D<r32>;


