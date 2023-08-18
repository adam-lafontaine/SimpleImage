#include <immintrin.h>
#include <xmmintrin.h>
#include <smmintrin.h>


namespace simd
{
    constexpr size_t LEN = 4;

    using vecf32 = __m128;
}


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
}


namespace simd
{
    static Gray_f32_255 load_gray(u8* p)
    {
        Gray_f32_255 g{};

        g.gray = _mm_set_ps((f32)p[0], (f32)p[1], (f32)p[2], (f32)p[3]);

        return g;
    }


    static RGB_f32_255 load_rgb(Pixel* p)
    {
        RGB_f32_255 rgb{};

        auto& p0 = p[0].rgba;
        auto& p1 = p[1].rgba;
        auto& p2 = p[2].rgba;
        auto& p3 = p[3].rgba;

        rgb.red   = _mm_set_ps((f32)p0.red, (f32)p1.red, (f32)p2.red, (f32)p3.red);
        rgb.green = _mm_set_ps((f32)p0.green, (f32)p1.green, (f32)p2.green, (f32)p3.green);
        rgb.blue  = _mm_set_ps((f32)p0.blue, (f32)p1.blue, (f32)p2.blue, (f32)p3.blue);

        return rgb;
    }


    static RGBA_f32_255 load_rgba(Pixel* p)
    {
        RGBA_f32_255 rgba{};

        auto& p0 = p[0].rgba;
        auto& p1 = p[1].rgba;
        auto& p2 = p[2].rgba;
        auto& p3 = p[3].rgba;

        rgba.red   = _mm_set_ps((f32)p0.red, (f32)p1.red, (f32)p2.red, (f32)p3.red);
        rgba.green = _mm_set_ps((f32)p0.green, (f32)p1.green, (f32)p2.green, (f32)p3.green);
        rgba.blue  = _mm_set_ps((f32)p0.blue, (f32)p1.blue, (f32)p2.blue, (f32)p3.blue);
        rgba.alpha = _mm_set_ps((f32)p0.alpha, (f32)p1.alpha, (f32)p2.alpha, (f32)p3.alpha);

        return rgba;
    }


    static void store_gray(Gray_f32_255 const& src, u8* dst)
    {
        f32 arr[LEN] = { 0 };

        _mm_store_ps(arr, src.gray);

        for (u32 i = 0; i < LEN; ++i)
        {
            dst[i] = (u8)arr[i];
        }
    }

}