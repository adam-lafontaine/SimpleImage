#pragma once

#include <cstddef>
#include <cassert>
#include <cstdint>
#include <cstdlib>

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


class Point2Du32
{
public:
	u32 x;
	u32 y;
};


class Point2Dr32
{
public:
	r32 x;
	r32 y;
};


// region of interest in an image
class Range2Du32
{
public:
	u32 x_begin;
	u32 x_end;   // one past last x
	u32 y_begin;
	u32 y_end;   // one past last y
};


template <typename T>
class Matrix2D
{
public:
	T* data_ = nullptr;
	u32 width = 0;
	u32 height = 0;	

#ifndef NDEBUG

	~Matrix2D() { assert(!(bool)data_); }

#endif // !NDEBUG
};


template <typename T>
class MemoryBuffer
{
public:
	T* data_ = nullptr;
	size_t capacity_ = 0;
	size_t size_ = 0;

#ifndef NDEBUG

	~MemoryBuffer() { assert(!(bool)data_); }

#endif // !NDEBUG
};