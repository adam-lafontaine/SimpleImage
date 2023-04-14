#include "../../tests_include.hpp"
#include "../../../simage/src/util/color_space.hpp"


static bool equals(f32 lhs, f32 rhs)
{
    return std::abs(lhs - rhs) < (1.0f / 255.0f);
}


bool hsv_conversion_test()
{
    printf("hsv converstion_f32_test: ");

    for (u32 r = 0; r < 256; ++r)
    {
        auto red = r / 255.0f;

        for (u32 g = 0; g < 256; ++g)
        {
            auto green = g / 255.0f;

            for (u32 b = 0; b < 256; ++b)
            {
                auto blue = b / 255.0f;

                auto hsv = hsv::f32_from_rgb_f32(red, green, blue);
                auto rgb = hsv::f32_to_rgb_f32(hsv.hue, hsv.sat, hsv.val);

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


void hsv_draw_test(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;

    img::Buffer16 buffer;
    mb::create_buffer(buffer, width * height * 3);

    auto hsv = img::make_view_3(width, height, buffer);

    auto hue = img::select_channel(hsv, img::HSV::H);
    auto sat = img::select_channel(hsv, img::HSV::S);
    auto val = img::select_channel(hsv, img::HSV::V);

    auto r = make_range(width, height);
    auto w = width / 256;
    for (u32 x = 0; x < 256; ++x)
    {
        r.x_begin = x * w;
        r.x_end = std::min(width, r.x_begin + w);

        img::fill(img::sub_view(hue, r), (u8)(255 - x));
    }

    r = make_range(width, height);
    auto h = height / 256;
    for (u32 y = 0; y < 256; ++y)
    {
        r.y_begin = y * h;
        r.y_end = std::min(height, r.y_begin + h);

        img::fill(img::sub_view(sat, r), (u8)(255 - y));
    }

    img::fill(val, 255);

    img::map_hsv_rgb(hsv, out);
    
    mb::destroy_buffer(buffer);
}