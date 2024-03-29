#include "../../tests_include.hpp"
#include "../../../simage/src/util/color_space.hpp"


static bool equals(f32 lhs, f32 rhs);


bool lch_conversion_test()
{
    printf("lch_conversion_test: ");

    f32 R = 0.0f;
    f32 G = 0.0f;
    f32 B = 0.0f;

    f32 L = 0.0f;
    f32 C = 0.0f;
    f32 H = 0.0f;

    for (u32 r = 0; r < 256; ++r)
    {
        auto red = r / 255.0f;

        for (u32 g = 0; g < 256; ++g)
        {
            auto green = g / 255.0f;

            for (u32 b = 0; b < 256; ++b)
            {
                auto blue = b / 255.0f;

                lch::f32_from_rgb_f32(red, green, blue, &L, &C, &H);
                lch::f32_to_rgb_f32(L, C, H, &R, &G, &B);

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


void lch_draw_test(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;
    
    auto buffer = img::create_buffer32(width * height * 3);

    auto lch = img::make_view_3(width, height, buffer);

    auto view_L = img::select_channel(lch, img::LCH::L);
    auto view_C = img::select_channel(lch, img::LCH::C);
    auto view_H = img::select_channel(lch, img::LCH::H);

    img::for_each_xy(view_H, [width](u32 x, u32 y) { return (f32)x / width; });

    img::for_each_xy(view_C, [height](u32 x, u32 y) { return (f32)y / height; });

    img::fill(view_L, 1.0f);

    img::map_lch_rgba(lch, out);
    
    mb::destroy_buffer(buffer);
}