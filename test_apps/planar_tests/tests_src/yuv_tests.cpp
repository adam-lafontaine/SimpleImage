#include "../../tests_include.hpp"
#include "../../../simage/src/util/color_space.hpp"


static bool equals(f32 lhs, f32 rhs);


bool yuv_conversion_test()
{
    printf("yuv_conversion_test: ");

    f32 R = 0.0f;
    f32 G = 0.0f;
    f32 B = 0.0f;

    f32 Y = 0.0f;
    f32 U = 0.0f;
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

                yuv::f32_from_rgb_f32(red, green, blue, &Y, &U, &V);
                yuv::f32_to_rgb_f32(Y, U, V, &R, &G, &B);

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


void yuv_draw_test(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;

    img::Buffer32 buffer;
    mb::create_buffer(buffer, width * height * 3);

    auto yuv = img::make_view_3(width, height, buffer);

    auto view_Y = img::select_channel(yuv, img::YUV::Y);
    auto view_U = img::select_channel(yuv, img::YUV::U);
    auto view_V = img::select_channel(yuv, img::YUV::V);

    img::for_each_xy(view_V, [height](u32 x, u32 y) { return (f32)y / height; });

    img::for_each_xy(view_U, [width](u32 x, u32 y) { return (f32)x / width; });

    img::fill(view_Y, 0.5f);

    img::map_yuv_rgba(yuv, out);
    
    mb::destroy_buffer(buffer);
}