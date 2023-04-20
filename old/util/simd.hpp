#pragma once

#include "../defines.hpp"


#ifdef SIMD_INTEL_256


#include <xmmintrin.h>
#include <immintrin.h>

namespace simd
{
	constexpr u32 VEC_LEN = 8;

	using vec_t = __m256;


	static inline vec_t load_broadcast(const f32* a)
	{
		return _mm256_broadcast_ss(a);
	}


	static inline vec_t load(const f32* a)
	{
		return _mm256_load_ps(a);
	}


	static inline void store(f32* dst, vec_t const& a)
	{
		_mm256_store_ps(dst, a);
	}


	static inline vec_t setzero()
	{
		return _mm256_setzero_ps();
	}


	static inline vec_t add(vec_t const& a, vec_t const& b)
	{
		return _mm256_add_ps(a, b);
	}


	static inline vec_t subtract(vec_t const& a, vec_t const& b)
	{
		return _mm256_sub_ps(a, b);
	}


	static inline vec_t multiply(vec_t const& a, vec_t const& b)
	{
		return _mm256_mul_ps(a, b);
	}


	static inline vec_t divide(vec_t const& a, vec_t const& b)
	{
		return _mm256_div_ps(a, b);
	}


	static inline vec_t fmadd(vec_t const& a, vec_t const& b, vec_t const& c)
	{
		return _mm256_fmadd_ps(a, b, c);
	}


	static inline vec_t sqrt(vec_t const& a)
	{
		return _mm256_sqrt_ps(a);
	}
}

#endif // SIMD_INTEL_256


#ifdef SIMD_INTEL_128

constexpr u32 VEC_LEN = 4;

using vec_t = __m128;


namespace simd
{
	static inline vec_t load_broadcast(const f32* a)
	{
		return _mm_load_ps1(a);
	}


	static inline vec_t load(const f32* a)
	{
		return _mm_load_ps(a);
	}


	static inline void store(f32* dst, vec_t const& a)
	{
		_mm_store_ps(dst, a);
	}


	static inline vec_t setzero()
	{
		return _mm_setzero_ps();
	}


	static inline vec_t add(vec_t const& a, vec_t const& b)
	{
		return _mm_add_ps(a, b);
	}


	static inline vec_t subtract(vec_t const& a, vec_t const& b)
	{
		return _mm_sub_ps(a, b);
	}


	static inline vec_t multiply(vec_t const& a, vec_t const& b)
	{
		return _mm_mul_ps(a, b);
	}


	static inline vec_t divide(vec_t const& a, vec_t const& b)
	{
		return _mm_div_ps(a, b);
	}


	static inline vec_t fmadd(vec_t const& a, vec_t const& b, vec_t const& c)
	{
		return _mm_fmadd_ps(a, b, c);
	}


	static inline vec_t sqrt(vec_t const& a)
	{
		return _mm_sqrt_ps(a);
	}
}

#endif // SIMD_INTEL_128


#ifdef SIMD_ARM_NEON

constexpr u32 VEC_LEN = 4;

using vec_t = float32x4_t;


namespace simd
{
	static inline vec_t load_broadcast(const f32* a)
	{
		return vld1q_dup_f32(a);
	}


	static inline vec_t load(const f32* a)
	{
		return vld1q_f32(a);
	}


	static inline void store(f32* dst, vec_t const& a)
	{
		vst1q_f32(dst, a);
	}


	static inline vec_t setzero()
	{
		return vmovq_n_f32(0);
	}


	static inline vec_t add(vec_t const& a, vec_t const& b)
	{
		return vaddq_f32(a, b);
	}


	static inline vec_t subtract(vec_t const& a, vec_t const& b)
	{
		return vsubq_f32(a, b);
	}


	static inline vec_t multiply(vec_t const& a, vec_t const& b)
	{
		return vmulq_f32(a, b);
	}


	static inline vec_t divide(vec_t const& a, vec_t const& b)
	{
		return vmulq_f32(a, vrecpeq_f32(b));
	}


	static inline vec_t fmadd(vec_t const& a, vec_t const& b, vec_t const& c)
	{
		return vmlaq_f32(c, b, a);
	}


	static inline vec_t sqrt(vec_t const& a)
	{
		return vrecpeq_f32(vrsqrteq_f32(a));
	}
}

#endif