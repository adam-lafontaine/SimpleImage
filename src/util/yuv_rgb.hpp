#pragma once

#include "../defines.hpp"

#include <cmath>


namespace yuv
{
    class RGBr32
    {
    public:
        r32 red;
        r32 green;
        r32 blue;
    };


    class YUVr32
    {
    public:
        r32 y;
        r32 u;
        r32 v;
    };


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

        r32 y = ry * r + gy * g + by * b;
        r32 u = ru * r + gu * g + bu * b;
        r32 v = rv * r + gv * g + bv * b;

        return { y, u, v };
    }


    inline RGBr32 to_rgb(r32 y, r32 u, r32 v)
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

        r32 r = yr * y + ur * u + vr * v;
        r32 g = yg * y + ug * u + vg * v;
        r32 b = yb * y + ub * u + vb * v;

        return { r, g, b };
    }
}