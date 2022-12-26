#pragma once

#include "../defines.hpp"

#include <cmath>


// https://github.com/QuantitativeBytes/qbColor/blob/main/qbColor.cpp
// https://www.youtube.com/watch?v=I8i0W8ve-JI





namespace hsv
{
    class HSVr32
    {
    public:
        r32 hue;
        r32 sat;
        r32 val;
    };


    class RGBr32
    {
    public:
        r32 red;
        r32 green;
        r32 blue;
    };


    constexpr r32 HUE_MAX = 360.0f;
    constexpr r32 zerof = 1.0f / HUE_MAX;



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


    inline HSVr32 from_rgb(r32 r, r32 g, r32 b)
    {
        auto max = std::max(r, std::max(g, b));
        auto min = std::min(r, std::min(g, b));

        auto constexpr equals = [](r32 lhs, r32 rhs) { return std::abs(lhs - rhs) < zerof; };

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

        if (r_max && g_min && b_min)
        {
            h = 0.0f;
            return { h, s, v };
        }

        if (r_min && g_max && b_min)
        {
            h = 120.0f / 360.0f;
            return { h, s, v };
        }

        if (r_min && g_min && b_max)
        {
            h = 240.0f / 360.0f;
            return { h, s, v };
        }

        u32 h_id =
            (r_max && b_min) ? 0 :
            (g_max && b_min) ? 1 :
            (g_max && r_min) ? 2 :
            (b_max && r_min) ? 3 :
            (b_max && g_min) ? 4 : 5;
        //(g_min && r_max) ? 5;

        auto h_360 = h_id * 60.0f;

        switch (h_id)
        {
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
}