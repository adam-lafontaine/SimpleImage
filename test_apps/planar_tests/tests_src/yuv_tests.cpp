#include "../../tests_include.hpp"
#include "../../../simage/src/util/color_space.hpp"


static bool equals(f32 lhs, f32 rhs);


bool yuv_conversion_test()
{
    printf("yuv_conversion_test: ");

    for (u32 r = 0; r < 256; ++r)
    {
        auto red = r / 255.0f;

        for (u32 g = 0; g < 256; ++g)
        {
            auto green = g / 255.0f;

            for (u32 b = 0; b < 256; ++b)
            {
                auto blue = b / 255.0f;

                auto yuv = yuv::f32_from_rgb_f32(red, green, blue);
                auto rgb = yuv::f32_to_rgb_f32(yuv.y, yuv.u, yuv.v);

                if (!equals(red, rgb.red) || !equals(green, rgb.green) || !equals(blue, rgb.blue))
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

    auto r = make_range(width, height);
    for (u32 x = 0; x < width; ++x)
    {
        r.x_begin = x;
        r.x_end = x + 1;
        img::fill(img::sub_view(view_U, r), (f32)x / width);
    }

    r = make_range(width, height);
    for (u32 y = 0; y < height; ++y)
    {
        r.y_begin = y;
        r.y_end = y + 1;

        img::fill(img::sub_view(view_V, r), (f32)y / height);
    }

    img::fill(view_Y, 0.5f);

    img::map_yuv_rgb(yuv, out);
    
    mb::destroy_buffer(buffer);
}