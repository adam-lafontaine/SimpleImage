#ifndef SIMAGE_NO_SIMD

#ifdef SIMD_INTEL_128

#include <immintrin.h>
#include <xmmintrin.h>
#include <smmintrin.h>


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


namespace simd
{
    class Gray_f32_255
    {
    public:
        vecf32 gray;
    };


    class RGB_f32_255
    {
    public:
        vecf32 red;
        vecf32 green;
        vecf32 blue;
    };


    class RGBA_f32_255
    {
    public:
        vecf32 red;
        vecf32 green;
        vecf32 blue;
        vecf32 alpha;
    };


    class Gray_f32_1
    {
    public:
        vecf32 gray;
    };


    class RGB_f32_1
    {
    public:
        vecf32 red;
        vecf32 green;
        vecf32 blue;
    };


    class RGBA_f32_1
    {
    public:
        vecf32 red;
        vecf32 green;
        vecf32 blue;
        vecf32 alpha;
    };


    using Pixel = simage::Pixel;
    using RGBf32p = simage::RGBf32p;
    using RGBAf32p = simage::RGBAf32p;
}


namespace simd
{
    template <class PROC>
    static void process_span(u32 width, PROC const& proc)
    {
        constexpr auto step = (u32)LEN;

        u32 x = 0;
        for (; x < width; x += step)
        {
            proc(x);
        }

        x = width - step;
        proc(x);
    }
    
}


#ifdef SIMD_INTEL_128

/* load */

namespace simd
{
    static vecf32 load_scalar_broadcast(f32 val)
    {
        return _mm_load_ps1(&val);
    }


    static Gray_f32_1 load_gray(f32* p)
    {
        Gray_f32_1 g{};

        g.gray = _mm_load_ps(p);

        return g;
    }


    static Gray_f32_255 load_gray(u8* p)
    {
        Gray_f32_255 g{};

        f32 gray[LEN] = { (f32)p[0], (f32)p[1], (f32)p[2], (f32)p[3] };

        //g.gray = _mm_set_ps((f32)p[3], (f32)p[2], (f32)p[1], (f32)p[0]);
        g.gray = _mm_load_ps(gray);

        return g;
    }


    static RGB_f32_255 load_rgb(Pixel* p)
    {
        RGB_f32_255 rgb{};

        auto& p0 = p[0].rgba;
        auto& p1 = p[1].rgba;
        auto& p2 = p[2].rgba;
        auto& p3 = p[3].rgba;

        rgb.red   = _mm_set_ps((f32)p3.red, (f32)p2.red, (f32)p1.red, (f32)p0.red);
        rgb.green = _mm_set_ps((f32)p3.green, (f32)p2.green, (f32)p1.green, (f32)p0.green);
        rgb.blue  = _mm_set_ps((f32)p3.blue, (f32)p2.blue, (f32)p1.blue, (f32)p0.blue);

        return rgb;
    }


    static RGBA_f32_255 load_rgba(Pixel* p)
    {
        RGBA_f32_255 rgba{};

        auto& p0 = p[0].rgba;
        auto& p1 = p[1].rgba;
        auto& p2 = p[2].rgba;
        auto& p3 = p[3].rgba;

        rgba.red   = _mm_set_ps((f32)p3.red, (f32)p2.red, (f32)p1.red, (f32)p0.red);
        rgba.green = _mm_set_ps((f32)p3.green, (f32)p2.green, (f32)p1.green, (f32)p0.green);
        rgba.blue  = _mm_set_ps((f32)p3.blue, (f32)p2.blue, (f32)p1.blue, (f32)p0.blue);
        rgba.alpha = _mm_set_ps((f32)p3.alpha, (f32)p2.alpha, (f32)p1.alpha, (f32)p0.alpha);

        return rgba;
    }


    static RGB_f32_1 load_rgb(RGBf32p p)
    {
        RGB_f32_1 rgb{};

        rgb.red   = _mm_load_ps(p.R);
        rgb.green = _mm_load_ps(p.G);
        rgb.blue  = _mm_load_ps(p.B);

        return rgb;
    }


    static RGBA_f32_1 load_rgba(RGBAf32p p)
    {
        RGBA_f32_1 rgb{};

        rgb.red   = _mm_load_ps(p.R);
        rgb.green = _mm_load_ps(p.G);
        rgb.blue  = _mm_load_ps(p.B);
        rgb.alpha = _mm_load_ps(p.A);

        return rgb;
    }
}


/* store */

namespace simd
{
    static void store_gray(Gray_f32_255 const& src, u8* dst)
    {
        f32 gray[LEN] = { 0 };

        _mm_store_ps(gray, src.gray);

        for (u32 i = 0; i < LEN; ++i)
        {
            dst[i] = (u8)gray[i];
        }
    }


    static void store_gray(Gray_f32_1 const& src, f32* dst)
    {
        _mm_store_ps(dst, src.gray);
    }


    static void store_rgb(RGB_f32_255 const& src, Pixel* dst)
    {
        f32 red[LEN]   = { 0 };
        f32 green[LEN] = { 0 };
        f32 blue[LEN]  = { 0 };

        _mm_store_ps(red, src.red);
        _mm_store_ps(green, src.green);
        _mm_store_ps(blue, src.blue);

        for (u32 i = 0; i < LEN; ++i)
        {
            auto& p = dst[i].rgba;
            p.red = (u8)red[i];
            p.green = (u8)red[i];
            p.blue = (u8)blue[i];
        }
    }


    static void store_rgba(RGBA_f32_255 const& src, Pixel* dst)
    {
        f32 red[LEN]   = { 0 };
        f32 green[LEN] = { 0 };
        f32 blue[LEN]  = { 0 };
        f32 alpha[LEN] = { 0 };

        _mm_store_ps(red, src.red);
        _mm_store_ps(green, src.green);
        _mm_store_ps(blue, src.blue);
        _mm_store_ps(alpha, src.alpha);

        for (u32 i = 0; i < LEN; ++i)
        {
            auto& p = dst[i].rgba;
            p.red   = (u8)red[i];
            p.green = (u8)green[i];
            p.blue  = (u8)blue[i];
            p.alpha = (u8)alpha[i];
        }
    }

}


namespace simd
{
    static void multiply(Gray_f32_255 const& src, vecf32 const& v_val, Gray_f32_1& dst)
    {
        dst.gray = _mm_mul_ps(src.gray, v_val);
    }


    static void multiply(Gray_f32_1 const& src, vecf32 const& v_val, Gray_f32_255& dst)
    {
        dst.gray = _mm_mul_ps(src.gray, v_val);
    }


    static void map_gray(Gray_f32_255 const& src, Gray_f32_1& dst)
    {
        constexpr f32 scalar = 1.0f / 255.0f;
        auto v_scalar = _mm_load_ps1(&scalar);

        dst.gray = _mm_mul_ps(src.gray, v_scalar);
    }


    static void map_gray(Gray_f32_1 const& src, Gray_f32_255& dst)
    {
        constexpr f32 scalar = 255.0f;
        auto v_scalar = _mm_load_ps1(&scalar);

        dst.gray = _mm_mul_ps(src.gray, v_scalar);
    }


    static void map_rgb(RGB_f32_255 const& src, RGB_f32_1& dst)
    {
        constexpr f32 scalar = 1.0f / 255.0f;
        auto v_scalar = _mm_load_ps1(&scalar);

        dst.red   = _mm_mul_ps(src.red, v_scalar);
        dst.green = _mm_mul_ps(src.green, v_scalar);
        dst.blue  = _mm_mul_ps(src.blue, v_scalar);
    }


    static void map_rgba(RGBA_f32_255 const& src, RGBA_f32_1& dst)
    {
        constexpr f32 scalar = 255.0f;
        auto v_scalar = _mm_load_ps1(&scalar);

        dst.red   = _mm_mul_ps(src.red, v_scalar);
        dst.green = _mm_mul_ps(src.green, v_scalar);
        dst.blue  = _mm_mul_ps(src.blue, v_scalar);
        dst.alpha = _mm_mul_ps(src.alpha, v_scalar);
    }
}

#endif // SIMD_INTEL_128

#ifdef SIMD_INTEL_256

/* load */

namespace simd
{
    static Gray_f32_1 load_gray(f32* p)
    {
        Gray_f32_1 g{};

        g.gray = _mm256_loadu_ps(p);

        return g;
    }


    static Gray_f32_255 load_gray(u8* p)
    {
        Gray_f32_255 g{};

        f32 gray[LEN] = 
        { 
            (f32)p[0], 
            (f32)p[1], 
            (f32)p[2], 
            (f32)p[3], 
            (f32)p[4],
            (f32)p[5], 
            (f32)p[6], 
            (f32)p[7] 
        };
        
        g.gray = _mm256_loadu_ps(gray);

        return g;
    }
}


/* store */

namespace simd
{
    static void store_gray(Gray_f32_255 const& src, u8* dst)
    {
        f32 gray[LEN] = { 0 };

        _mm256_store_ps(gray, src.gray);

        for (u32 i = 0; i < LEN; ++i)
        {
            dst[i] = (u8)gray[i];
        }
    }


    static void store_gray(Gray_f32_1 const& src, f32* dst)
    {
        _mm256_store_ps(dst, src.gray);
    }
}


/* map */

namespace simd
{
    static void map_gray(Gray_f32_255 const& src, Gray_f32_1& dst)
    {
        constexpr f32 scalar = 1.0f / 255.0f;
        auto v_scalar = _mm256_broadcast_ss(&scalar);

        dst.gray = _mm256_mul_ps(src.gray, v_scalar);
    }


    static void map_gray(Gray_f32_1 const& src, Gray_f32_255& dst)
    {
        constexpr f32 scalar = 255.0f;
        auto v_scalar = _mm256_broadcast_ss(&scalar);

        dst.gray = _mm256_mul_ps(src.gray, v_scalar);
    }
}

#endif // SIMD_INTEL_256


#endif // !SIMAGE_NO_SIMD