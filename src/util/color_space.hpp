#pragma once

#include "../defines.hpp"

#include <cmath>
#include <array>


namespace color_space
{
    namespace lut
    {
        static constexpr std::array<r32, 256> channel_r32()
        {
            std::array<r32, 256> lut = {};

            for (u32 i = 0; i < 256; ++i)
            {
                lut[i] = i / 255.0f;
            }

            return lut;
        }
    }
}


namespace color_space
{

    static constexpr r32 to_channel_r32(u8 value)
    {
        constexpr auto lut = lut::channel_r32();

        return lut[value];
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

        constexpr auto val = (u8)513;

        constexpr auto val2 = 513 % 256;

        return value;
    }


    inline constexpr u8 round_to_u8(r32 value)
    {
        return (u8)(u32)(value + 0.5f);
    }


    static constexpr u8 to_channel_u8(r32 value)
    {
        return round_to_u8(clamp(value) * 255);
    }


    static constexpr r32 lerp_to_r32(u8 value, r32 min, r32 max)
    {
        assert(min < max);

        return min + (value / 255.0f) * (max - min);
    }


    static constexpr u8 lerp_to_u8(r32 value, r32 min, r32 max)
    {
        assert(min < max);
        assert(value >= min);
        assert(value <= max);

        if (value < min)
        {
            value = min;
        }
        else if (value > max)
        {
            value = max;
        }

        auto ratio = (value - min) / (max - min);

        return round_to_u8(ratio * 255);
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


    class RGBAu8
    {
    public:
        u8 red;
        u8 green;
        u8 blue;
        u8 alpha;
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
        u8 pad;
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
        u8 pad;
    };
}


namespace hsv
{
    namespace cs = color_space;

    constexpr r32 HUE_MAX = 360.0f;
    constexpr r32 zerof = 1.0f / HUE_MAX;


    inline constexpr bool equals(r32 lhs, r32 rhs)
    {
        auto diff = lhs - rhs;

        diff = diff < 0.0f ? -1.0f * diff : diff;

        return diff < zerof;
    }


    inline constexpr cs::RGBr32 r32_to_rgb_r32(r32 h, r32 s, r32 v)
    {
        if (v <= zerof)
        {
            return { 0.0f, 0.0f, 0.0f };
        }

        if (s <= zerof || h < 0.0f)
        {
            return { v, v, v };
        }

        auto max = v;
        auto range = s * v;
        auto min = max - range;

        auto d = h * 6.0f; // 360.0f / 60.0f;

        auto h_id = (int)d;

        auto rise = min + (d - h_id) * range;
        auto fall = max - (d - h_id) * range;

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


    inline constexpr cs::RGBAu8 r32_to_rgba_u8(r32 h, r32 s, r32 v)
    {
        if (v <= zerof)
        {
            return { 0, 0, 0, 255 };
        }

        if (s <= zerof || h < 0.0f)
        {
            auto gray = cs::round_to_u8(v * 255);
            return { gray, gray, gray, 255 };
        }
        
        auto rgb = r32_to_rgb_r32(h, s, v);

        auto r = cs::round_to_u8(rgb.red * 255);
        auto g = cs::round_to_u8(rgb.green * 255);
        auto b = cs::round_to_u8(rgb.blue * 255);

        return { r, g, b, 255 };
    }


    inline constexpr cs::RGBAu8 u8_to_rgba_u8(u8 h, u8 s, u8 v)
    {
        if (!s)
        {
            return { v, v, v, 255 };
        }

        auto H = cs::to_channel_r32(h);
        auto S = cs::to_channel_r32(s);
        auto V = cs::to_channel_r32(v);

        return r32_to_rgba_u8(H, S, V);
    }


    inline constexpr cs::HSVr32 r32_from_rgb_r32(r32 r, r32 g, r32 b)
    {
        auto max = std::max({r, g, b});
        auto min = std::min({r, g, b});

        auto const r_max = equals(r, max);
        auto const r_min = equals(r, min);
        auto const g_max = equals(g, max);
        auto const g_min = equals(g, min);
        auto const b_max = equals(b, max);
        auto const b_min = equals(b, min);

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

        int h_id =
            (r_max && g_min && b_min) ? -3 :
            (r_min && g_max && b_min) ? -2 :
            (r_min && g_min && b_max) ? -1 :

            (r_max && b_min) ? 0 :
            (g_max && b_min) ? 1 :
            (g_max && r_min) ? 2 :
            (b_max && r_min) ? 3 :
            (b_max && g_min) ? 4 : 5;
            //(g_min && r_max) ? 5;

        auto h_360 = h_id * 60.0f;

        switch (h_id)
        {
        case -3:
            h_360 = 0.0f;
            break;
        case -2:
            h_360 = 120.0f;
            break;
        case -1:
            h_360 = 240.0f;
            break;

        case 0:
            h_360 += 60.0f * (g - min) / range;
            break;
        case 1:
            h_360 += 60.0f * (max - r) / range;
            break;
        case 2:
            h_360 += 60.0f * (b - min) / range;
            break;
        case 3:
            h_360 += 60.0f * (max - g) / range;
            break;
        case 4:
            h_360 += 60.0f * (r - min) / range;
            break;
        case 5:
            h_360 += 60.0f * (max - b) / range;
            break;
        default:
            h_360 = -360.0f;
        }

        h = h_360 / 360.0f;

        return { h, s, v };
    }


    inline constexpr cs::HSVr32 r32_from_rgb_u8(u8 r, u8 g, u8 b)
    {
        auto R = cs::to_channel_r32(r);
        auto G = cs::to_channel_r32(g);
        auto B = cs::to_channel_r32(b);

        return r32_from_rgb_r32(R, G, B);
    }


    inline constexpr cs::HSVu8 u8_from_rgb_r32(r32 r, r32 g, r32 b)
    {
        auto R = cs::clamp(r);
        auto G = cs::clamp(g);
        auto B = cs::clamp(b);
        
        auto hsv = r32_from_rgb_r32(R, G, B);

        auto h = cs::round_to_u8(hsv.hue * 255);
        auto s = cs::round_to_u8(hsv.sat * 255);
        auto v = cs::round_to_u8(hsv.val * 255);

        return { h, s, v };
    }


    inline constexpr cs::HSVu8 u8_from_rgb_u8(u8 r, u8 g, u8 b)
    {
        auto min = std::min({ r, g, b });
        auto max = std::max({ r, g, b });

        auto v = max;
        u8 s = max == min ? 0 : cs::round_to_u8(255.0f * (max - min) / max);

        auto R = (r32)(r - min) / (max - min);
        auto G = (r32)(g - min) / (max - min);
        auto B = (r32)(b - min) / (max - min);

        auto hsv = r32_from_rgb_r32(R, G, B);

        auto h = cs::round_to_u8(hsv.hue * 255);

        return { h, s, v };
    }
}


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
        u -= 0.5f;
        v -= 0.5f;

        constexpr r32 yr = 1.0f;
        constexpr r32 ur = 0.0f;
        constexpr r32 vr = 1.13983f;

        constexpr r32 yg = 1.0f;
        constexpr r32 ug = -0.39465f;
        constexpr r32 vg = -0.5806f;

        constexpr r32 yb = 1.0f;
        constexpr r32 ub = 2.03211f;
        constexpr r32 vb = 0.0f;

        r32 r = cs::clamp((yr * y) + (ur * u) + (vr * v));
        r32 g = cs::clamp((yg * y) + (ug * u) + (vg * v));
        r32 b = cs::clamp((yb * y) + (ub * u) + (vb * v));

        return { r, g, b };
    }


    inline constexpr cs::RGBAu8 u8_to_rgba_u8(u8 y, u8 u, u8 v)
    {
        auto U = (r32)u - 128.0f;
        auto V = (r32)v - 128.0f;

        auto R = y + 1.402f * V;
        auto G = y - 0.344f * U - 0.714f * V;
        auto B = y + 1.722f * U;

        auto r = cs::round_to_u8(R);
        auto g = cs::round_to_u8(G);
        auto b = cs::round_to_u8(B);

        return { r, g, b, 255 };
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

        auto y = cs::round_to_u8(Y);
        auto u = cs::round_to_u8(U);
        auto v = cs::round_to_u8(V);

        return { y, u, v };
    }


    inline constexpr cs::RGBr32 u8_to_rgb_r32(u8 y, u8 u, u8 v)
    {
        auto rgba = u8_to_rgba_u8(y, u, v);

        auto r = cs::to_channel_r32(rgba.red);
        auto g = cs::to_channel_r32(rgba.green);
        auto b = cs::to_channel_r32(rgba.blue);

        return { r, g, b };
    }

}