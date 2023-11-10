#include "../../tests_include.hpp"
#include "../../../simage/src/util/color_space.hpp"


static bool equals(f32 lhs, f32 rhs)
{
    return std::abs(lhs - rhs) < (1.0f / 255.0f);
}


bool hsv_conversion_test()
{
    printf("hsv_conversion_test: ");

    f32 R = 0.0f;
    f32 G = 0.0f;
    f32 B = 0.0f;

    f32 H = 0.0f;
    f32 S = 0.0f;
    f32 V = 0.0f;

    for (u32 r = 0; r < 256; ++r)
    {
        auto red = r / 255.0f;

        for (u32 g = 0; g < 256; ++g)
        {
            auto green = g / 255.0f;

            for (u32 b = 0; b < 256; ++b)
            {
                auto blue = b / 255.0f;

                hsv::f32_from_rgb_f32(red, green, blue, &H, &S, &V);
                hsv::f32_to_rgb_f32(H, S, V, &R, &G, &B);

                if (!equals(red, R) || !equals(green, G) || !equals(blue, B))
                {
                    printf("FAIL\n");
                    return false;
                }
            }
        }
    }

    printf("OK\n");
    return true;
}


void hsv_draw_test(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;

    auto buffer = img::create_buffer32(width * height * 3);

    auto hsv = img::make_view_3(width, height, buffer);

    auto hue = img::select_channel(hsv, img::HSV::H);
    auto sat = img::select_channel(hsv, img::HSV::S);
    auto val = img::select_channel(hsv, img::HSV::V);

    img::for_each_xy(hue, [width](u32 x, u32 y) { return (f32)x / width; });

    img::for_each_xy(sat, [height](u32 x, u32 y) { return (f32)y / height; });

    img::fill(val, 1.0f);

    img::map_hsv_rgba(hsv, out);
    
    mb::destroy_buffer(buffer);
}