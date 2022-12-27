#pragma once

#include "../defines.hpp"

#include <cmath>
#include <array>


namespace color_space
{
    static constexpr std::array<r32, 256> channel_r32_lut()
    {
        std::array<r32, 256> lut = {};

        for (u32 i = 0; i < 256; ++i)
        {
            lut[i] = i / 255.0f;
        }

        return lut;
    }


    static constexpr r32 to_channel_r32(u8 value)
    {
        constexpr auto lut = channel_r32_lut();

        return lut[value];
    }


    static constexpr u8 to_channel_u8(r32 value)
    {
        if (value < 0.0f)
        {
            value = 0.0f;
        }
        else if (value > 1.0f)
        {
            value = 1.0f;
        }

        return (u8)(u32)(value * 255 + 0.5f);
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

        return (u8)(u32)(ratio * 255 + 0.5f);
    }


    class RGBr32
    {
    public:
        r32 red;
        r32 green;
        r32 blue;
    };


    class HSVr32
    {
    public:
        r32 hue;
        r32 sat;
        r32 val;
    };


    class YUVr32
    {
    public:
        r32 y;
        r32 u;
        r32 v;
    };
}


namespace hsv
{
    namespace cs = color_space;

    using RGBr32 = cs::RGBr32;
    //using RGBu8 = cs::RGBu8;
    using HSVr32 = cs::HSVr32;


    constexpr r32 HUE_MAX = 360.0f;
    constexpr r32 zerof = 1.0f / HUE_MAX;


    inline constexpr bool equals(r32 lhs, r32 rhs)
    {
        auto diff = lhs - rhs;

        diff = diff < 0.0f ? -1.0f * diff : diff;

        return diff < zerof;
    }


    inline constexpr RGBr32 to_rgb(r32 h, r32 s, r32 v)
    {
        if (v <= zerof)
        {
            return { 0.0f, 0.0f, 0.0f };
        }

        if (s <= zerof)
        {
            return { v, v, v };
        }

        if (h < 0)
        {
            //assert(false);
            return { 0.0f, 0.0f, 0.0f };
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


    inline constexpr HSVr32 from_rgb(r32 r, r32 g, r32 b)
    {
        auto max = std::max(r, std::max(g, b));
        auto min = std::min(r, std::min(g, b));

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


    inline constexpr HSVr32 from_rgb(u8 r, u8 g, u8 b)
    {
        auto R = cs::to_channel_r32(r);
        auto G = cs::to_channel_r32(g);
        auto B = cs::to_channel_r32(b);

        return from_rgb(R, G, B);
    }
}


namespace yuv
{
    namespace cs = color_space;

    using YUVr32 = cs::YUVr32;
    using RGBr32 = cs::RGBr32;
    using HSVr32 = cs::HSVr32;


    inline constexpr YUVr32 from_rgb(r32 r, r32 g, r32 b)
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
        r32 u = (ru * r) + (gu * g) + (bu * b);
        r32 v = (rv * r) + (gv * g) + (bv * b);

        return { y, u, v };
    }


    inline constexpr RGBr32 to_rgb(r32 y, r32 u, r32 v)
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

        r32 r = (yr * y) + (ur * u) + (vr * v);
        r32 g = (yg * y) + (ug * u) + (vg * v);
        r32 b = (yb * y) + (ub * u) + (vb * v);

        return { r, g, b };
    }


    static constexpr std::array<r32, 256> uv_channel_r32_lut()
    {
        std::array<r32, 256> lut = {};

        for (u32 i = 0; i < 256; ++i)
        {
            lut[i] = cs::to_channel_r32((u8)i) - 0.5f;
        }

        return lut;
    }


    inline constexpr r32 to_uv_channel_r32(u8 value)
    {
        constexpr auto lut = uv_channel_r32_lut();

        return lut[value];
    }


    inline constexpr RGBr32 to_rgb(u8 y, u8 u, u8 v)
    {
        auto Y = cs::to_channel_r32(y);
        auto U = to_uv_channel_r32(u);
        auto V = to_uv_channel_r32(v);

        return to_rgb(Y, U, V);
    }


    inline constexpr HSVr32 to_hsv(u8 y, u8 u, u8 v)
    {
        auto rgb = to_rgb(y, u, v);

        return hsv::from_rgb(rgb.red, rgb.green, rgb.blue);
    }

}