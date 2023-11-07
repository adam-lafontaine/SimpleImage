#pragma once

#include "types.hpp"

#include <cmath>
#include <array>


namespace color_space
{
    constexpr auto PI = 3.1415926536f;

    constexpr f32 CH_U8_MAX = 255.0f;

    constexpr f32 CH_U16_MAX = 255.0f * 256;

    namespace lut
    {
        static constexpr std::array<f32, 256> channel_f32()
        {
            std::array<f32, 256> lut = {};

            for (u32 i = 0; i < 256; ++i)
            {
                lut[i] = i / CH_U8_MAX;
            }

            return lut;
        }


        static constexpr std::array<f32, 256> u8_to_f32 = channel_f32();
    }
}


namespace color_space
{
    inline constexpr f32 clamp(f32 value, f32 min, f32 max)
    {
        if (value >= min && value <= max)
        {
            return value;
        }

        if (value < min)
        {
            return min;
        }

        return max;
    }


    inline constexpr f32 clamp(f32 value)
    {
        return clamp(value, 0.0f, 1.0f);
    }


    inline constexpr f32 to_channel_f32(f32 value)
    {
        return clamp(value);
    }


    inline constexpr f32 to_channel_f32(u8 value)
    {
        return lut::u8_to_f32[value];
    }


    template <typename T>
    inline constexpr T round_to_unsigned(f32 value)
    {
        return (T)(value + 0.5f);
    }


    inline constexpr u32 round_to_u32(f32 value)
    {
        return round_to_unsigned<u32>(value);
    }


    inline constexpr u8 round_to_u8(f32 value)
    {
        return round_to_unsigned<u8>(value);
    }


    inline constexpr u8 to_channel_u8(f32 value)
    {
        return round_to_u8(clamp(value) * CH_U8_MAX);
    }
}


/* types */

namespace color_space
{   
    template <typename T>
    class RGB
    {
    public:
        T red;
        T green;
        T blue;
    };


    template <typename T>
    class RGBA
    {
    public:
        T red;
        T green;
        T blue;
        T alpha;
    };


    template <typename T>
    class HSV
    {
    public:
        T hue;
        T sat;
        T val;
    };


    template <typename T>
    class LCH
    {
    public:
        T light;
        T chroma;
        T hue;
    };


    template <typename T>
    class YUV
    {
    public:
        T y;
        T u;
        T v;
    };


    using RGBf32 = RGB<f32>;
    using RGBu8 = RGB<u8>;
    using RGBAu8 = RGBA<u8>;

    using HSVf32 = HSV<f32>;
    using HSVu8 = HSV<u8>;

    using LCHf32 = LCH<f32>;
    using LCHu8 = LCH<u8>;

    using YUVf32 = YUV<f32>;
    using YUVu8 = YUV<u8>;
}


/* grayscale */

namespace gray
{
    namespace cs = color_space;

    constexpr f32 COEFF_RED = 0.299f;
    constexpr f32 COEFF_GREEN = 0.587f;
    constexpr f32 COEFF_BLUE = 0.114f;


    inline constexpr f32 f32_from_rgb_f32(f32 r, f32 g, f32 b)
    {
        return COEFF_RED * r + COEFF_GREEN * g + COEFF_BLUE * b;
    }


    inline constexpr u8 u8_from_rgb_u8(u8 r, u8 g, u8 b)
    {
        return cs::round_to_u8(COEFF_RED * r + COEFF_GREEN * g + COEFF_BLUE * b);
    }


    inline constexpr f32 f32_from_rgb_u8(u8 r, u8 g, u8 b)
    {
        auto gray = COEFF_RED * r + COEFF_GREEN * g + COEFF_BLUE * b;
        return cs::to_channel_f32(gray / cs::CH_U8_MAX);
    }


    inline constexpr u8 u8_from_rgb_f32(f32 r, f32 g, f32 b)
    {
        auto gray = COEFF_RED * r + COEFF_GREEN * g + COEFF_BLUE * b;
        return cs::to_channel_u8(gray);
    }
    
}


/* hsv f32 */

namespace hsv
{
    namespace cs = color_space;

    constexpr f32 HUE_MAX = 360.0f;
    constexpr f32 ZERO_F = 1.0f / HUE_MAX;


    inline constexpr bool equals(f32 lhs, f32 rhs)
    {
        // no constexpr std::abs
        auto diff = lhs - rhs;

        diff = diff < 0.0f ? -1.0f * diff : diff;

        return diff < ZERO_F;
    }


    inline constexpr cs::RGBf32 f32_to_rgb_f32(f32 h, f32 s, f32 v)
    {
        if (v <= ZERO_F)
        {
            return { 0.0f, 0.0f, 0.0f };
        }

        if (s <= ZERO_F || h < 0.0f)
        {
            return { v, v, v };
        }

        auto max = v;
        auto range = s * v;
        auto min = max - range;

        auto d = h * 6.0f; // 360.0f / 60.0f;

        auto h_id = (int)d;
        auto ratio = d - h_id;

        auto rise = min + ratio * range;
        auto fall = max - ratio * range;

        f32 r = 0.0f;
        f32 g = 0.0f;
        f32 b = 0.0f;

        switch (h_id)
        {
        case 0:
            r = max;
            g = rise;
            b = min;
            break;
        case 1:
            r = fall;
            g = max;
            b = min;
            break;
        case 2:
            r = min;
            g = max;
            b = rise;
            break;
        case 3:
            r = min;
            g = fall;
            b = max;
            break;
        case 4:
            r = rise;
            g = min;
            b = max;
            break;
        default:
            r = max;
            g = min;
            b = fall;
            break;
        }

        return { r, g, b };
    }


    inline constexpr cs::HSVf32 f32_from_rgb_f32(f32 r, f32 g, f32 b)
    {
        auto max = std::max({ r, g, b });
        auto min = std::min({ r, g, b });

        f32 h = 0.0f;
        f32 s = 0.0f;
        f32 v = max;

        if (equals(max, min))
        {
            h = -1.0f;
            s = 0.0f;

            return { h, s, v };
        }

        auto range = max - min;

        s = range / v;

        auto const r_is_max = equals(r, max);
        auto const r_is_min = equals(r, min);
        auto const g_is_max = equals(g, max);
        auto const g_is_min = equals(g, min);
        auto const b_is_max = equals(b, max);
        auto const b_is_min = equals(b, min);

        auto delta_h = 1.0f / 6;
        auto h_id = 0.0f;
        auto delta_c = 0.0f;

        if (r_is_max && b_is_min)
        {
            h_id = 0;
            delta_c = g - min;
        }
        else if (g_is_max && b_is_min)
        {
            h_id = 1;
            delta_c = max - r;
        }
        else if (g_is_max && r_is_min)
        {
            h_id = 2;
            delta_c = b - min;
        }
        else if (b_is_max && r_is_min)
        {
            h_id = 3;
            delta_c = max - g;
        }
        else if (b_is_max && g_is_min)
        {
            h_id = 4;
            delta_c = r - min;
        }
        else
        {
            h_id = 5;
            delta_c = max - b;
        }

        h = (delta_h * (h_id + (f32)delta_c / (max - min)));

        return { h, s, v };
    }
}


/* hsv overloads */

namespace hsv
{
    inline constexpr cs::RGBu8 f32_to_rgb_u8(f32 h, f32 s, f32 v)
    {
        if (v <= ZERO_F)
        {
            return { 0, 0, 0 };
        }

        if (s <= ZERO_F || h < 0.0f)
        {
            auto gray = cs::to_channel_u8(v);
            return { gray, gray, gray };
        }

        auto rgb = f32_to_rgb_f32(h, s, v);

        return {
            cs::to_channel_u8(rgb.red),
            cs::to_channel_u8(rgb.green),
            cs::to_channel_u8(rgb.blue)
        };
    }


    inline constexpr cs::HSVf32 f32_from_rgb_u8(u8 r, u8 g, u8 b)
    {
        auto R = cs::to_channel_f32(r);
        auto G = cs::to_channel_f32(g);
        auto B = cs::to_channel_f32(b);

        return f32_from_rgb_f32(R, G, B);
    }


    inline constexpr cs::HSVu8 u8_from_rgb_u8(u8 r, u8 g, u8 b)
    {
        auto v = std::max({ r, g, b });        

        auto R = cs::to_channel_f32(r);
        auto G = cs::to_channel_f32(g);
        auto B = cs::to_channel_f32(b);

        auto hsv = f32_from_rgb_f32(R, G, B);

        auto h = cs::to_channel_u8(hsv.hue);
        auto s = cs::to_channel_u8(hsv.sat);

        return { h, s, v };
    }


    inline constexpr cs::RGBu8 u8_to_rgb_u8(u8 h, u8 s, u8 v)
    {
        if (v == 0 || s == 0)
        {
            return { v, v, v };
        }

        auto H = cs::to_channel_f32(h);
        auto S = cs::to_channel_f32(s);
        auto V = cs::to_channel_f32(v);

        auto rgb = f32_to_rgb_f32(H, S, V);

        return {
            cs::to_channel_u8(rgb.red),
            cs::to_channel_u8(rgb.green),
            cs::to_channel_u8(rgb.blue)
        };
    }
}


/* LCH f32 */

namespace lch
{
    namespace cs = color_space;


    inline cs::LCHf32 f32_from_rgb_f32(f32 r, f32 g, f32 b)
    {
        auto l_ = 0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b;
        auto m_ = 0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b;
        auto s_ = 0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b;

        l_ = cbrtf(l_);
        m_ = cbrtf(m_);
        s_ = cbrtf(s_);

        auto L = 0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_;
        auto A = 1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_;
        auto B = 0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_;

        auto C = std::hypotf(A, B);
        auto H = std::atan2(B, A) / (2 * cs::PI) + 0.5f;

        return { L, C, H };
    }


    inline cs::RGBf32 f32_to_rgb_f32(f32 l, f32 c, f32 h)
    {
        auto H = (h - 0.5f) * 2 * cs::PI;
        auto A = c * std::cos(H);
        auto B = c * std::sin(H);

        auto l_ = l + 0.3963377774f * A + 0.2158037573f * B;
        auto m_ = l - 0.1055613458f * A - 0.0638541728f * B;
        auto s_ = l - 0.0894841775f * A - 1.2914855480f * B;

        l_ = l_ * l_ * l_;
        m_ = m_ * m_ * m_;
        s_ = s_ * s_ * s_;

        auto red = 4.0767416621f * l_ - 3.3077115913f * m_ + 0.2309699292f * s_;
        auto green = -1.2684380046f * l_ + 2.6097574011f * m_ - 0.3413193965f * s_;
        auto blue = -0.0041960863f * l_ - 0.7034186147f * m_ + 1.7076147010f * s_;

        return {
            cs::clamp(red),
            cs::clamp(green),
            cs::clamp(blue)
        };
    }
}


/* LCH overloads */

namespace lch
{
    template <typename T>
    inline constexpr cs::LCHf32 lch_f32_from_rgb(T r, T g, T b)
    {
        auto R = cs::to_channel_f32(r);
        auto G = cs::to_channel_f32(g);
        auto B = cs::to_channel_f32(b);

        return f32_from_rgb_f32(R, G, B);
    }


    template <typename T>
    inline constexpr cs::RGBf32 rgb_f32_from_lch(T l, T c, T h)
    {
        auto L = cs::to_channel_f32(l);
        auto C = cs::to_channel_f32(c);
        auto H = cs::to_channel_f32(h);

        return f32_to_rgb_f32(L, C, H);
    }


    inline cs::RGBu8 f32_to_rgb_u8(f32 l, f32 c, f32 h)
    {
        auto rgb = f32_to_rgb_f32(l, c, h);

        return {
            cs::to_channel_u8(rgb.red),
            cs::to_channel_u8(rgb.green),
            cs::to_channel_u8(rgb.blue)
        };
    }


    inline cs::LCHf32 f32_from_rgb_u8(u8 r, u8 g, u8 b)
    {
        return lch_f32_from_rgb(r, g, g);
    }


    inline cs::LCHu8 u8_from_rgb_u8(u8 r, u8 g, u8 b)
    {
        auto lch = lch_f32_from_rgb(r, g, b);

        return {
            cs::to_channel_u8(lch.light),
            cs::to_channel_u8(lch.chroma),
            cs::to_channel_u8(lch.hue)
        };
    }


    inline cs::RGBu8 u8_to_rgb_u8(u8 l, u8 c, u8 h)
    {
        auto rgb = rgb_f32_from_lch(l, c, h);

        return {
            cs::to_channel_u8(rgb.red),
            cs::to_channel_u8(rgb.green),
            cs::to_channel_u8(rgb.blue)
        };
    }
}


/* YUV */

namespace yuv
{
    namespace cs = color_space;    


    template <typename T>
    constexpr f32 rgb_to_y(T r, T g, T b)
    {
        constexpr f32 ry = 0.299f;
        constexpr f32 gy = 0.587f;
        constexpr f32 by = 0.114f;

        return (ry * r) + (gy * g) + (by * b);
    }


    template <typename T>
    constexpr f32 rgb_to_u(T r, T g, T b, f32 scale)
    {
        constexpr f32 ru = -0.14713f;
        constexpr f32 gu = -0.28886f;
        constexpr f32 bu = 0.436f;

        return (ru * r) + (gu * g) + (bu * b) + scale / 2;
    }


    template <typename T>
    constexpr f32 rgb_to_v(T r, T g, T b, f32 scale)
    {
        constexpr f32 rv = 0.615f;
        constexpr f32 gv = -0.51499f;
        constexpr f32 bv = -0.10001f;

        return (rv * r) + (gv * g) + (bv * b) + scale / 2;
    }


    template <typename T>
    constexpr f32 yuv_to_r(T y, T u, T v, f32 scale)
    {
        constexpr f32 yr = 1.0f;
        constexpr f32 ur = 0.0f;
        constexpr f32 vr = 1.13983f;

        f32 Y = (f32)y;
        f32 U = (f32)u - scale / 2;
        f32 V = (f32)v - scale / 2;

        return cs::clamp((yr * Y) + (ur * U) + (vr * V), 0.0f, scale);
    }


    template <typename T>
    constexpr f32 yuv_to_g(T y, T u, T v, f32 scale)
    {
        constexpr f32 yg = 1.0f;
        constexpr f32 ug = -0.39465f;
        constexpr f32 vg = -0.5806f;

        f32 Y = (f32)y;
        f32 U = (f32)u - scale / 2;
        f32 V = (f32)v - scale / 2;

        return cs::clamp((yg * Y) + (ug * U) + (vg * V), 0.0f, scale);
    }


    template <typename T>
    constexpr f32 yuv_to_b(T y, T u, T v, f32 scale)
    {
        constexpr f32 yb = 1.0f;
        constexpr f32 ub = 2.03211f;
        constexpr f32 vb = 0.0f;

        f32 Y = (f32)y;
        f32 U = (f32)u - scale / 2;
        f32 V = (f32)v - scale / 2;

        return cs::clamp((yb * Y) + (ub * U) + (vb * V), 0.0f, scale);
    }


    inline constexpr cs::YUVf32 f32_from_rgb_f32(f32 r, f32 g, f32 b)
    {
        f32 y = rgb_to_y(r, g, b);
        f32 u = rgb_to_u(r, g, b, 1.0f);
        f32 v = rgb_to_v(r, g, b, 1.0f);

        return { y, u, v };
    }


    inline constexpr cs::YUVf32 f32_from_rgb_u8(u8 r, u8 g, u8 b)
    {
        constexpr f32 scale = 255.0;

        f32 y = rgb_to_y(r, g, b);
        f32 u = rgb_to_u(r, g, b, scale);
        f32 v = rgb_to_v(r, g, b, scale);

        return { 
            cs::to_channel_f32(y / scale),
            cs::to_channel_f32(u / scale),
            cs::to_channel_f32(v / scale)
        };
    }


    inline constexpr cs::YUVu8 u8_from_rgb_u8(u8 r, u8 g, u8 b)
    {
        constexpr f32 scale = 255.0;

        f32 y = rgb_to_y(r, g, b);
        f32 u = rgb_to_u(r, g, b, scale);
        f32 v = rgb_to_v(r, g, b, scale);

        return {
            cs::round_to_u8(y),
            cs::round_to_u8(u),
            cs::round_to_u8(v)
        };
    }


    inline constexpr cs::RGBf32 f32_to_rgb_f32(f32 y, f32 u, f32 v)
    {
        constexpr f32 scale = 1.0f;

        auto R = yuv_to_r(y, u, v, scale);
        auto G = yuv_to_g(y, u, v, scale);
        auto B = yuv_to_b(y, u, v, scale);

        return { R, G, B };
    }


    inline constexpr cs::RGBu8 f32_to_rgb_u8(f32 y, f32 u, f32 v)
    {
        constexpr f32 scale = 1.0f;

        auto R = yuv_to_r(y, u, v, scale);
        auto G = yuv_to_g(y, u, v, scale);
        auto B = yuv_to_b(y, u, v, scale);

        return {
            cs::to_channel_u8(R),
            cs::to_channel_u8(G),
            cs::to_channel_u8(B)
        };
    }


    inline constexpr cs::RGBf32 u8_to_rgb_f32(u8 y, u8 u, u8 v)
    {
        constexpr f32 scale = 255.0;

        auto R = yuv_to_r(y, u, v, scale);
        auto G = yuv_to_g(y, u, v, scale);
        auto B = yuv_to_b(y, u, v, scale);

        return {
            cs::to_channel_f32(R / scale),
            cs::to_channel_f32(G / scale),
            cs::to_channel_f32(B / scale)
        };
 
    }


    inline constexpr cs::RGBu8 u8_to_rgb_u8(u8 y, u8 u, u8 v)
    {
        constexpr f32 scale = 255.0;

        auto R = yuv_to_r(y, u, v, scale);
        auto G = yuv_to_g(y, u, v, scale);
        auto B = yuv_to_b(y, u, v, scale);

        return {
            cs::round_to_u8(R),
            cs::round_to_u8(G),
            cs::round_to_u8(B)
        };
    }
}