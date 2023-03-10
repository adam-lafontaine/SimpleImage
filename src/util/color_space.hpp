#pragma once

#include "../defines.hpp"

#include <cmath>
#include <array>


namespace color_space
{
    constexpr auto PI = 3.1415926536f;

    constexpr r32 CHANNEL_U8_MAX = 255.0f;

    constexpr r32 CHANNEL_U16_MAX = 255.0f * 256;

    namespace lut
    {
        static constexpr std::array<r32, 256> channel_r32()
        {
            std::array<r32, 256> lut = {};

            for (u32 i = 0; i < 256; ++i)
            {
                lut[i] = i / CHANNEL_U8_MAX;
            }

            return lut;
        }
    }
}


namespace color_space
{

    inline constexpr r32 to_channel_r32(u8 value)
    {
        constexpr auto lut = lut::channel_r32();

        return lut[value];
    }


    inline constexpr r32 to_channel_r32(u16 value)
    {
        return value / CHANNEL_U16_MAX;
    }


    inline constexpr r32 clamp(r32 value)
    {
        if (value < 0.0f)
        {
            value = 0.0f;
        }
        else if (value > 1.0f)
        {
            value = 1.0f;
        }

        return value;
    }


    inline constexpr u32 round_to_u32(r32 value)
    {
        return (u32)(value + 0.5f);
    }


    inline constexpr u8 to_channel_u8(r32 value)
    {
        return (u8)round_to_u32(clamp(value) * CHANNEL_U8_MAX);
    }


    inline constexpr u8 to_channel_u8(u16 value)
    {
        return (u8)(value / 256);
    }


    inline constexpr u16 to_channel_u16(r32 value)
    {
        return (u16)round_to_u32(clamp(value) * CHANNEL_U16_MAX);
    }


    inline constexpr u16 to_channel_u16(u8 value)
    {
        return (u16)value * 256;
    }
}


/* types */

namespace color_space
{   
    class RGBr32
    {
    public:
        r32 red;
        r32 green;
        r32 blue;
    };


    class RGBu8
    {
    public:
        u8 red;
        u8 green;
        u8 blue;
    };


    class RGBu16
    {
    public:
        u16 red;
        u16 green;
        u16 blue;
    };


    class HSVr32
    {
    public:
        r32 hue;
        r32 sat;
        r32 val;
    };


    class HSVu8
    {
    public:
        u8 hue;
        u8 sat;
        u8 val;
    };


    class HSVu16
    {
    public:
        u16 hue;
        u16 sat;
        u16 val;
    };


    class LCHr32
    {
    public:
        r32 light;
        r32 chroma;
        r32 hue;
    };


    class LCHu8
    {
    public:
        u8 light;
        u8 chroma;
        u8 hue;
    };


    class LCHu16
    {
    public:
        u16 light;
        u16 chroma;
        u16 hue;
    };


    class YUVr32
    {
    public:
        r32 y;
        r32 u;
        r32 v;
    };


    class YUVu8
    {
    public:
        u8 y;
        u8 u;
        u8 v;
    };


    class YUVu16
    {
    public:
        u16 y;
        u16 u;
        u16 v;
    };
}


/* grayscale */

namespace gray
{
    constexpr r32 COEFF_RED = 0.299f;
    constexpr r32 COEFF_GREEN = 0.587f;
    constexpr r32 COEFF_BLUE = 0.114f;


    inline constexpr r32 r32_from_rgb_r32(r32 r, r32 g, r32 b)
    {
        return COEFF_RED * r + COEFF_GREEN * g + COEFF_BLUE * b;
    }


    inline constexpr u8 u8_from_rgb_u8(u8 r, u8 g, u8 b)
    {
        return (u8)(COEFF_RED * r + COEFF_GREEN * g + COEFF_BLUE * b);
    }


    inline constexpr u16 u16_from_rgb_u16(u16 r, u16 g, u16 b)
    {
        return (u16)(COEFF_RED * r + COEFF_GREEN * g + COEFF_BLUE * b);
    }
    
}


/* hsv r32 */

namespace hsv
{
    namespace cs = color_space;

    constexpr r32 HUE_MAX = 360.0f;
    constexpr r32 ZERO_F = 1.0f / HUE_MAX;


    inline constexpr bool equals(r32 lhs, r32 rhs)
    {
        // no constexpr std::abs
        auto diff = lhs - rhs;

        diff = diff < 0.0f ? -1.0f * diff : diff;

        return diff < ZERO_F;
    }


    inline constexpr cs::RGBr32 r32_to_rgb_r32(r32 h, r32 s, r32 v)
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

        r32 r = 0.0f;
        r32 g = 0.0f;
        r32 b = 0.0f;

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


    inline constexpr cs::HSVr32 r32_from_rgb_r32(r32 r, r32 g, r32 b)
    {
        auto max = std::max({ r, g, b });
        auto min = std::min({ r, g, b });

        r32 h = 0.0f;
        r32 s = 0.0f;
        r32 v = max;

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

        h = (delta_h * (h_id + (r32)delta_c / (max - min)));

        return { h, s, v };
    }
}


/* hsv overloads */

namespace hsv
{
    inline constexpr cs::RGBu8 r32_to_rgb_u8(r32 h, r32 s, r32 v)
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

        auto rgb = r32_to_rgb_r32(h, s, v);

        return {
            cs::to_channel_u8(rgb.red),
            cs::to_channel_u8(rgb.green),
            cs::to_channel_u8(rgb.blue)
        };
    }


    inline constexpr cs::HSVr32 r32_from_rgb_u8(u8 r, u8 g, u8 b)
    {
        auto R = cs::to_channel_r32(r);
        auto G = cs::to_channel_r32(g);
        auto B = cs::to_channel_r32(b);

        return r32_from_rgb_r32(R, G, B);
    }


    inline constexpr cs::HSVu8 u8_from_rgb_u8(u8 r, u8 g, u8 b)
    {
        auto v = std::max({ r, g, b });        

        auto R = cs::to_channel_r32(r);
        auto G = cs::to_channel_r32(g);
        auto B = cs::to_channel_r32(b);

        auto hsv = r32_from_rgb_r32(R, G, B);

        auto h = cs::to_channel_u8(hsv.hue);
        auto s = cs::to_channel_u8(hsv.sat);

        return { h, s, v };
    }


    inline constexpr cs::HSVu16 u16_from_rgb_u16(u16 r, u16 g, u16 b)
    {
        auto v = std::max({ r, g, b });

        auto R = cs::to_channel_r32(r);
        auto G = cs::to_channel_r32(g);
        auto B = cs::to_channel_r32(b);

        auto hsv = r32_from_rgb_r32(R, G, B);

        auto h = cs::to_channel_u16(hsv.hue);
        auto s = cs::to_channel_u16(hsv.sat);

        return { h, s, v };
    }


    inline constexpr cs::HSVu16 u16_from_rgb_u8(u8 r, u8 g, u8 b)
    {
        auto max = std::max({ r, g, b });

        auto v = cs::to_channel_u16(max);

        auto R = cs::to_channel_r32(r);
        auto G = cs::to_channel_r32(g);
        auto B = cs::to_channel_r32(b);

        auto hsv = r32_from_rgb_r32(R, G, B);

        auto h = cs::to_channel_u16(hsv.hue);
        auto s = cs::to_channel_u16(hsv.sat);

        return { h, s, v };
    }


    inline constexpr cs::RGBu8 u8_to_rgb_u8(u8 h, u8 s, u8 v)
    {
        if (v == 0 || s == 0)
        {
            return { v, v, v };
        }

        auto H = cs::to_channel_r32(h);
        auto S = cs::to_channel_r32(s);
        auto V = cs::to_channel_r32(v);

        auto rgb = r32_to_rgb_r32(H, S, V);

        return {
            cs::to_channel_u8(rgb.red),
            cs::to_channel_u8(rgb.green),
            cs::to_channel_u8(rgb.blue)
        };
    }


    inline constexpr cs::RGBu16 u16_to_rgb_u16(u16 h, u16 s, u16 v)
    {
        if (v == 0 || s == 0)
        {
            return { v, v, v };
        }

        auto H = cs::to_channel_r32(h);
        auto S = cs::to_channel_r32(s);
        auto V = cs::to_channel_r32(v);

        auto rgb = r32_to_rgb_r32(H, S, V);

        return {
            cs::to_channel_u16(rgb.red),
            cs::to_channel_u16(rgb.green),
            cs::to_channel_u16(rgb.blue)
        };
    }


    inline constexpr cs::RGBu8 u16_to_rgb_u8(u16 h, u16 s, u16 v)
    {
        if (v == 0 || s == 0)
        {
            auto gray = cs::to_channel_u8(v);
            return { gray, gray, gray };
        }
        
        auto H = cs::to_channel_r32(h);
        auto S = cs::to_channel_r32(s);
        auto V = cs::to_channel_r32(v);

        auto rgb = r32_to_rgb_r32(H, S, V);

        return {
            cs::to_channel_u8(rgb.red),
            cs::to_channel_u8(rgb.green),
            cs::to_channel_u8(rgb.blue)
        };
    }
}


/* LCH r32 */

namespace lch
{
    namespace cs = color_space;


    inline cs::LCHr32 r32_from_rgb_r32(r32 r, r32 g, r32 b)
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


    inline cs::RGBr32 r32_to_rgb_r32(r32 l, r32 c, r32 h)
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
    inline cs::RGBu8 r32_to_rgb_u8(r32 l, r32 c, r32 h)
    {
        auto rgb = r32_to_rgb_r32(l, c, h);

        return {
            cs::to_channel_u8(rgb.red),
            cs::to_channel_u8(rgb.green),
            cs::to_channel_u8(rgb.blue)
        };
    }


    inline cs::LCHr32 r32_from_rgb_u8(u8 r, u8 g, u8 b)
    {
        auto R = cs::to_channel_r32(r);
        auto G = cs::to_channel_r32(g);
        auto B = cs::to_channel_r32(b);

        return r32_from_rgb_r32(R, G, B);
    }


    inline cs::LCHu8 u8_from_rgb_u8(u8 r, u8 g, u8 b)
    {
        auto lch = r32_from_rgb_u8(r, g, b);

        return {
            cs::to_channel_u8(lch.light),
            cs::to_channel_u8(lch.chroma),
            cs::to_channel_u8(lch.hue)
        };
    }


    inline cs::RGBu8 u8_to_rgb_u8(u8 l, u8 c, u8 h)
    {
        auto L = cs::to_channel_r32(l);
        auto C = cs::to_channel_r32(c);
        auto H = cs::to_channel_r32(h);

        return r32_to_rgb_u8(L, C, H);
    }
}


/* YUV r32 */

namespace yuv
{
    namespace cs = color_space;


    inline constexpr cs::YUVr32 r32_from_rgb_r32(r32 r, r32 g, r32 b)
    {
        constexpr r32 ry = 0.299f;
        constexpr r32 gy = 0.587f;
        constexpr r32 by = 0.114f;

        constexpr r32 ru = -0.14713f;
        constexpr r32 gu = -0.28886f;
        constexpr r32 bu = 0.436f;

        constexpr r32 rv = 0.615f;
        constexpr r32 gv = -0.51499f;
        constexpr r32 bv = -0.10001f;

        r32 y = (ry * r) + (gy * g) + (by * b);
        r32 u = (ru * r) + (gu * g) + (bu * b) + 0.5f;
        r32 v = (rv * r) + (gv * g) + (bv * b) + 0.5f;

        return { y, u, v };
    }


    inline constexpr cs::RGBr32 r32_to_rgb_r32(r32 y, r32 u, r32 v)
    {
        constexpr r32 yr = 1.0f;
        constexpr r32 ur = 0.0f;
        constexpr r32 vr = 1.13983f;

        constexpr r32 yg = 1.0f;
        constexpr r32 ug = -0.39465f;
        constexpr r32 vg = -0.5806f;

        constexpr r32 yb = 1.0f;
        constexpr r32 ub = 2.03211f;
        constexpr r32 vb = 0.0f;

        u -= 0.5f;
        v -= 0.5f;

        auto R = (yr * y) + (ur * u) + (vr * v);
        auto G = (yg * y) + (ug * u) + (vg * v);
        auto B = (yb * y) + (ub * u) + (vb * v);

        return {
            cs::clamp(R),
            cs::clamp(G),
            cs::clamp(B)
        };
    }
}


/* YUV overloads */

namespace yuv
{
    inline constexpr cs::RGBu8 u8_to_rgb_u8(u8 y, u8 u, u8 v)
    {
        auto Y = cs::to_channel_r32(y);
        auto U = cs::to_channel_r32(u); // -0.5f;
        auto V = cs::to_channel_r32(v); // -0.5f;

        auto rgb = r32_to_rgb_r32(Y, U, V);

        return {
            cs::to_channel_u8(rgb.red),
            cs::to_channel_u8(rgb.green),
            cs::to_channel_u8(rgb.blue)
        };
    }


    inline constexpr cs::YUVu8 u8_from_rgb_u8(u8 r, u8 g, u8 b)
    {
        constexpr r32 ry = 0.299f;
        constexpr r32 gy = 0.587f;
        constexpr r32 by = 0.114f;

        constexpr r32 ru = -0.14713f;
        constexpr r32 gu = -0.28886f;
        constexpr r32 bu = 0.436f;

        constexpr r32 rv = 0.615f;
        constexpr r32 gv = -0.51499f;
        constexpr r32 bv = -0.10001f;

        auto Y = (ry * r) + (gy * g) + (by * b);
        auto U = (ru * r) + (gu * g) + (bu * b) + 128.0f;
        auto V = (rv * r) + (gv * g) + (bv * b) + 128.0f;

        return { 
            (u8)cs::round_to_u32(Y),
            (u8)cs::round_to_u32(U),
            (u8)cs::round_to_u32(V)
        };
    }


    inline constexpr cs::RGBr32 u8_to_rgb_r32(u8 y, u8 u, u8 v)
    {
        auto U = (r32)u - 128.0f;
        auto V = (r32)v - 128.0f;

        auto R = y + 1.402f * V;
        auto G = y - 0.344f * U - 0.714f * V;
        auto B = y + 1.722f * U;

        return { 
            cs::clamp(R),
            cs::clamp(G),
            cs::clamp(B)
        };
    }

}