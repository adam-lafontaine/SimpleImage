#ifndef SIMAGE_NO_SIMD


namespace simd
{
    using Pixel = simage::Pixel;
    using RGBf32p = simage::RGBf32p;
    using RGBAf32p = simage::RGBAf32p;
}


#ifdef SIMD_INTEL_128

#include <emmintrin.h>


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
    static void load_scalar_broadcast(f32 value, vecf32& dst)
    {
        dst = _mm_load_ps1(&value);
    }


    static void load_gray(u8* src, vecf32& dst)
    {
        auto p = src;
        auto v_int = _mm_set_epi32(p[3], p[2], p[1], p[0]);
        dst = _mm_cvtepi32_ps(v_int);
    }


    static void load_gray(f32* src, vecf32& dst)
    {
        dst = _mm_loadu_ps(src);
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

}


/* operations */

namespace simd
{
    static void multiply(vecf32 const& a, vecf32 const& b, vecf32& dst)
    {
        dst = _mm_mul_ps(a, b);
    }
}

#endif // SIMD_INTEL_128

#ifdef SIMD_INTEL_256

/* load */

namespace simd
{
    union i32_u8
    {
        i32 val_i32 = 0;

        struct
        {
            u8 val_u8;
            u8 pad1;
            u8 pad2;
            u8 pad3;
        };
        
    };

    static void load_scalar_broadcast(f32 value, vecf32& dst)
    {
        dst = _mm256_broadcast_ss(&value);
    }


    static void load_gray(u8* src, vecf32& dst)
    {        
        auto p = src;

        auto v_int = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);

        dst =  _mm256_cvtepi32_ps(v_int);
    }


    static void load_gray(f32* src, vecf32& dst)
    {
        dst = _mm256_loadu_ps(src);
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
}


/* operations */

namespace simd
{
    static void multiply(vecf32 const& a, vecf32 const& b, vecf32& dst)
    {
        dst = _mm256_mul_ps(a, b);
    }
}

#endif // SIMD_INTEL_256


#endif // !SIMAGE_NO_SIMD