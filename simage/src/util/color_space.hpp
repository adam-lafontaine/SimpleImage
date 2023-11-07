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


    inline constexpr f32 clamp_channel_f32(f32 value)
    {
        return clamp(value, 0.0f, 1.0f);
    }


    inline constexpr f32 to_channel_f32(f32 value)
    {
        return clamp_channel_f32(value);
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
        return round_to_u8(clamp_channel_f32(value) * CH_U8_MAX);
    }
}


/* types */

namespace color_space
{   
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


    static inline void hsv_to_rgb(f32 h, f32 s, f32 v, f32* pr, f32* pg, f32* pb)
    {
        /*if (v <= ZERO_F)
        {
            return { 0.0f, 0.0f, 0.0f };
        }

        if (s <= ZERO_F || h < 0.0f)
        {
            return { v, v, v };
        }*/

        assert(h >= 0.0f);
        assert(v > ZERO_F);
        assert(s > ZERO_F);        

        auto max = v;
        auto range = s * v;
        auto min = max - range;

        auto d = h * 6.0f; // 360.0f / 60.0f;

        auto h_id = (int)d;
        auto ratio = d - h_id;

        auto rise = min + ratio * range;
        auto fall = max - ratio * range;

        auto& r = *pr;
        auto& g = *pg;
        auto& b = *pb;

        r = 0.0f;
        g = 0.0f;
        b = 0.0f;

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
    }


    static inline void rgb_to_hsv(f32 r, f32 g, f32 b, f32* ph, f32* ps, f32* pv)
    {
        auto max = std::max({ r, g, b });
        auto min = std::min({ r, g, b });

        auto& h = *ph;
        auto& s = *ps;
        auto& v = *pv;

        h = 0.0f;
        s = 0.0f;
        v = max;

        if (equals(max, min))
        {
            h = -1.0f;
            s = 0.0f;

            return;
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
    }
}


/* hsv overloads */

namespace hsv
{
    static inline void f32_from_rgb_f32(f32 r, f32 g, f32 b, f32* ph, f32* ps, f32* pv)
    {
        rgb_to_hsv(r, g, b, ph, ps, pv);
    }


    static inline void f32_from_rgb_u8(u8 r, u8 g, u8 b, f32* ph, f32* ps, f32* pv)
    {
        auto R = cs::to_channel_f32(r);
        auto G = cs::to_channel_f32(g);
        auto B = cs::to_channel_f32(b);

        rgb_to_hsv(R, G, B, ph, ps, pv);
    }


    static inline void u8_from_rgb_u8(u8 r, u8 g, u8 b, u8* ph, u8* ps, u8* pv)
    {
        auto R = cs::to_channel_f32(r);
        auto G = cs::to_channel_f32(g);
        auto B = cs::to_channel_f32(b);

        f32 H = 0.0f;
        f32 S = 0.0f;
        f32 V = 0.0f;

        rgb_to_hsv(R, G, B, &H, &S, &V);

        *ph = cs::to_channel_u8(H);
        *ps = cs::to_channel_u8(S);
        *pv = cs::to_channel_u8(V);
    }


    static inline void f32_to_rgb_f32(f32 h, f32 s, f32 v, f32* pr, f32* pg, f32* pb)
    {
        if (v <= ZERO_F)
        {
            *pr = *pg = *pb = 0.0f;
            return;
        }

        if (s <= ZERO_F || h < 0.0f)
        {
            *pr = *pg = *pb = v;
            return;
        }

        hsv_to_rgb(h, s, v, pr, pg, pb);
    }    


    static inline void f32_to_rgb_u8(f32 h, f32 s, f32 v, u8* pr, u8* pg, u8* pb)
    {
        if (v <= ZERO_F)
        {
            *pr = *pg = *pb = 0;
            return;
        }

        if (s <= ZERO_F || h < 0.0f)
        {
            auto gray = cs::to_channel_u8(v);
            *pr = *pg = *pb = gray;
            return;
        }

        f32 R = 0.0f;
        f32 G = 0.0f;
        f32 B = 0.0f;

        hsv_to_rgb(h, s, v, &R, &G, &B);

        *pr = cs::to_channel_u8(R);
        *pg = cs::to_channel_u8(G);
        *pb = cs::to_channel_u8(B);
    }


    static inline void u8_to_rgb_u8(u8 h, u8 s, u8 v, u8* pr, u8* pg, u8* pb)
    {
        if (v == 0 || s == 0)
        {
            *pr = *pg = *pb = v;
            return;
        }

        auto H = cs::to_channel_f32(h);
        auto S = cs::to_channel_f32(s);
        auto V = cs::to_channel_f32(v);

        f32 R = 0.0f;
        f32 G = 0.0f;
        f32 B = 0.0f;

        hsv_to_rgb(H, S, V, &R, &G, &B);

        *pr = cs::to_channel_u8(R);
        *pg = cs::to_channel_u8(G);
        *pb = cs::to_channel_u8(B);
    }
}


/* LCH f32 */

namespace lch
{
    namespace cs = color_space;


    static inline void rgb_to_lch(f32 r, f32 g, f32 b, f32* pl, f32* pc, f32* ph)
    {
        auto l_ = 0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b;
        auto m_ = 0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b;
        auto s_ = 0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b;

        l_ = cbrtf(l_);
        m_ = cbrtf(m_);
        s_ = cbrtf(s_);

        *pl = 0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_;

        auto A = 1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_;
        auto B = 0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_;

        *pc = std::hypotf(A, B);
        *ph = std::atan2(B, A) / (2 * cs::PI) + 0.5f;
    }


    static inline void lch_to_rgb(f32 l, f32 c, f32 h, f32* pr, f32* pg, f32* pb)
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

        *pr = cs::clamp_channel_f32(4.0767416621f * l_ - 3.3077115913f * m_ + 0.2309699292f * s_);
        *pg = cs::clamp_channel_f32(-1.2684380046f * l_ + 2.6097574011f * m_ - 0.3413193965f * s_);
        *pb = cs::clamp_channel_f32(-0.0041960863f * l_ - 0.7034186147f * m_ + 1.7076147010f * s_);
    }
}


/* LCH overloads */

namespace lch
{
    static inline void f32_from_rgb_f32(f32 r, f32 g, f32 b, f32* pl, f32* pc, f32* ph)
    {
        rgb_to_lch(r, g, b, pl, pc, ph);
    }


    static inline void f32_from_rgb_u8(u8 r, u8 g, u8 b, f32* pl, f32* pc, f32* ph)
    {
        auto R = cs::to_channel_f32(r);
        auto G = cs::to_channel_f32(g);
        auto B = cs::to_channel_f32(b);

        rgb_to_lch(R, G, B, pl, pc, ph);
    }


    static inline void u8_from_rgb_u8(u8 r, u8 g, u8 b, u8* pl, u8* pc, u8* ph)
    {
        auto R = cs::to_channel_f32(r);
        auto G = cs::to_channel_f32(g);
        auto B = cs::to_channel_f32(b);

        f32 L = 0.0f;
        f32 C = 0.0f;
        f32 H = 0.0f;

        rgb_to_lch(R, G, B, &L, &C, &H);

        *pl = cs::to_channel_u8(L);
        *pc = cs::to_channel_u8(C);
        *ph = cs::to_channel_u8(H);
    }


    static inline void f32_to_rgb_f32(f32 l, f32 c, f32 h, f32* pr, f32* pg, f32* pb)
    {
        lch_to_rgb(l, c, h, pr, pg, pb);
    }


    static inline void f32_to_rgb_u8(f32 l, f32 c, f32 h, u8* pr, u8* pg, u8* pb)
    {
        f32 R = 0.0f;
        f32 G = 0.0f;
        f32 B = 0.0f;

        lch_to_rgb(l, c, h, &R, &G, &B);

        *pr = cs::to_channel_u8(R);
        *pg = cs::to_channel_u8(G);
        *pb = cs::to_channel_u8(B);
    }


    inline cs::RGBu8 u8_to_rgb_u8(u8 l, u8 c, u8 h, u8* pr, u8* pg, u8* pb)
    {
        auto L = cs::to_channel_f32(l);
        auto C = cs::to_channel_f32(c);
        auto H = cs::to_channel_f32(h);

        f32 R = 0.0f;
        f32 G = 0.0f;
        f32 B = 0.0f;

        lch_to_rgb(L, C, H, &R, &G, &B);

        *pr = cs::to_channel_u8(R);
        *pg = cs::to_channel_u8(G);
        *pb = cs::to_channel_u8(B);
    }
}


/* YUV */

namespace yuv
{
    namespace cs = color_space;


    static inline void rgb_to_yuv(f32 r, f32 g, f32 b, f32* py, f32* pu, f32* pv)
    {
        constexpr f32 ry = 0.299f;
        constexpr f32 gy = 0.587f;
        constexpr f32 by = 0.114f;

        constexpr f32 ru = -0.14713f;
        constexpr f32 gu = -0.28886f;
        constexpr f32 bu = 0.436f;

        constexpr f32 rv = 0.615f;
        constexpr f32 gv = -0.51499f;
        constexpr f32 bv = -0.10001f;

        *py = (ry * r) + (gy * g) + (by * b);
        *pu = (ru * r) + (gu * g) + (bu * b) + 0.5f;
        *pv = (rv * r) + (gv * g) + (bv * b) + 0.5f;
    }


    static inline void yuv_to_rgb(f32 y, f32 u, f32 v, f32* pr, f32* pg, f32* pb)
    {
        constexpr f32 yr = 1.0f;
        constexpr f32 ur = 0.0f;
        constexpr f32 vr = 1.13983f;

        constexpr f32 yg = 1.0f;
        constexpr f32 ug = -0.39465f;
        constexpr f32 vg = -0.5806f;

        constexpr f32 yb = 1.0f;
        constexpr f32 ub = 2.03211f;
        constexpr f32 vb = 0.0f;

        u -= 0.5f;
        v -= 0.5f;

        *pr = cs::clamp((yr * y) + (ur * u) + (vr * v), 0.0f, 1.0f);
        *pg = cs::clamp((yg * y) + (ug * u) + (vg * v), 0.0f, 1.0f);
        *pb = cs::clamp((yb * y) + (ub * u) + (vb * v), 0.0f, 1.0f);
    }


    static inline void f32_from_rgb_f32(f32 r, f32 g, f32 b, f32* py, f32* pu, f32* pv)
    {
        rgb_to_yuv(r, g, b, py, pu, pv);
    }


    static inline void f32_from_rgb_u8(u8 r, u8 g, u8 b, f32* py, f32* pu, f32* pv)
    {
        auto R = cs::to_channel_f32(r);
        auto G = cs::to_channel_f32(g);
        auto B = cs::to_channel_f32(b);

        rgb_to_yuv(R, G, B, py, pu, pv);
    }


    static inline void u8_from_rgb_u8(u8 r, u8 g, u8 b, u8* py, u8* pu, u8* pv)
    {
        auto R = cs::to_channel_f32(r);
        auto G = cs::to_channel_f32(g);
        auto B = cs::to_channel_f32(b);

        f32 Y = 0.0f;
        f32 U = 0.0f;
        f32 V = 0.0f;

        rgb_to_yuv(R, G, B, &Y, &U, &V);

        *py = cs::to_channel_u8(Y);
        *pu = cs::to_channel_u8(U);
        *pv = cs::to_channel_u8(V);
    }


    static inline void f32_to_rgb_f32(f32 y, f32 u, f32 v, f32* pr, f32* pg, f32* pb)
    {
        yuv_to_rgb(y, u, v, pr, pg, pb);
    }


    static inline void f32_to_rgb_u8(f32 y, f32 u, f32 v, u8* pr, u8* pg, u8* pb)
    {
        f32 R = 0.0f;
        f32 G = 0.0f;
        f32 B = 0.0f;

        yuv_to_rgb(y, u, v, &R, &G, &B);

        *pr = cs::to_channel_u8(R);
        *pg = cs::to_channel_u8(G);
        *pb = cs::to_channel_u8(B);
    }


    static inline void u8_to_rgb_f32(u8 y, u8 u, u8 v, f32* pr, f32* pg, f32* pb)
    {
        auto Y = cs::to_channel_f32(y);
        auto U = cs::to_channel_f32(u);
        auto V = cs::to_channel_f32(v);

        yuv_to_rgb(Y, U, V, pr, pg, pb);
    }


    static inline void u8_to_rgb_u8(u8 y, u8 u, u8 v, u8* pr, u8* pg, u8* pb)
    {
        auto Y = cs::to_channel_f32(y);
        auto U = cs::to_channel_f32(u);
        auto V = cs::to_channel_f32(v);

        f32 R = 0.0f;
        f32 G = 0.0f;
        f32 B = 0.0f;

        yuv_to_rgb(Y, U, V, &R, &G, &B);

        *pr = cs::to_channel_u8(R);
        *pg = cs::to_channel_u8(G);
        *pb = cs::to_channel_u8(B);
    }
}