#ifndef SIMAGE_NO_SIMD


namespace simd
{
    using Pixel = simage::Pixel;
    using RGBf32p = simage::RGBf32p;
    using RGBAf32p = simage::RGBAf32p;
}


#ifdef SIMD_INTEL_128

#include <emmintrin.h>
#include <immintrin.h>


namespace simd
{
    constexpr size_t LEN = 4;

    using vecf32 = __m128;
}

#endif // SIMD_INTEL_128

#ifdef SIMD_INTEL_256

#include <immintrin.h>

namespace simd
{
    constexpr size_t LEN = 8;

    using vecf32 = __m256;
}

#endif // SIMD_INTEL_256

#ifdef SIMD_INTEL_128

/* load */

namespace simd
{
    static inline vecf32 load_f32_broadcast(f32 value)
    {
        return _mm_load_ps1(&value);
    }


    static inline vecf32 load_u8_broadcast(u8 value)
    {
        auto v_int = _mm_set_epi32(value, value, value, value);
        return _mm_cvtepi32_ps(v_int);
    }


    static inline vecf32 setzero_f32()
    {
        return _mm_setzero_ps();
    }    


    static inline vecf32 load_f32(f32* src)
    {
        return _mm_loadu_ps(src);
    }


    static inline vecf32 load_gray(f32* src)
    {
        return load_f32(src);
    }


    static inline vecf32 load_gray(u8* src)
    {
        auto p = src;
        auto v_int = _mm_set_epi32(p[3], p[2], p[1], p[0]);
        return _mm_cvtepi32_ps(v_int);
    }


    template <typename T>
    static inline vecf32 load_bytes(T* src)
    {
        return load_f32((f32*)src);
    }
}


/* store */

namespace simd
{
    static void store_gray(vecf32 const& src, u8* dst)
    {
        f32 gray[LEN] = { 0 };

        _mm_store_ps(gray, src);

        for (u32 i = 0; i < LEN; ++i)
        {
            dst[i] = (u8)gray[i];
        }
    }


    static void store_gray(vecf32 const& src, f32* dst)
    {
        _mm_store_ps(dst, src);
    }


    static void store_f32(vecf32 const& src, f32* dst)
    {
        _mm_store_ps(dst, src);
    }


    template <typename T>
    static void store_bytes(vecf32 const& src, T* dst)
    {
        store_f32(src, (f32*)dst);
    }

}


/* operations */

namespace simd
{
    static inline vecf32 mul(vecf32 const& a, vecf32 const& b)
    {
        return _mm_mul_ps(a, b);
    }


    static inline vecf32 fmadd(vecf32 const& a, vecf32 const& b, vecf32 const& c)
    {
        return _mm_fmadd_ps(a, b, c);
    }
}

#endif // SIMD_INTEL_128

#ifdef SIMD_INTEL_256

/* load */

namespace simd
{
    static inline vecf32 load_f32_broadcast(f32 value)
    {
        return _mm256_broadcast_ss(&value);
    }


    static inline vecf32 load_u8_broadcast(u8 value)
    {
        auto v_int = _mm256_set_epi32(value, value, value, value, value, value, value, value);
        return _mm256_cvtepi32_ps(v_int);
    }


    static inline vecf32 setzero_f32()
    {
        return _mm256_setzero_ps();
    }    


    static inline vecf32 load_f32(f32* src)
    {
        return _mm256_loadu_ps(src);
    }


    static inline vecf32 load_gray(f32* src)
    {
        return load_f32(src);
    }


    static inline vecf32 load_gray(u8* src)
    {
        auto p = src;
        auto v_int = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
        return _mm256_cvtepi32_ps(v_int);
    }


    template <typename T>
    static inline vecf32 load_bytes(T* src)
    {
        return load_f32((f32*)src);
    }
}


/* store */

namespace simd
{
    static void store_gray(vecf32 const& src, u8* dst)
    {
        int gray32[LEN] = { 0 };

        auto v_int = _mm256_cvtps_epi32(src);

        _mm256_storeu_epi32(gray32, v_int);
        
        for (u32 i = 0; i < LEN; ++i)
        {
            dst[i] = (u8)gray32[i];
        }
    }


    static void store_gray(vecf32 const& src, f32* dst)
    {
        _mm256_store_ps(dst, src);
    }


    static void store_f32(vecf32 const& src, f32* dst)
    {
        _mm256_store_ps(dst, src);
    }


    template <typename T>
    static void store_bytes(vecf32 const& src, T* dst)
    {
        store_f32(src, (f32*)dst);
    }
}


/* operations */

namespace simd
{
    static inline vecf32 mul(vecf32 const& a, vecf32 const& b)
    {
        return _mm256_mul_ps(a, b);
    }


    static inline vecf32 fmadd(vecf32 const& a, vecf32 const& b, vecf32 const& c)
    {
        return _mm256_fmadd_ps(a, b, c);
    }
}

#endif // SIMD_INTEL_256


#endif // !SIMAGE_NO_SIMD