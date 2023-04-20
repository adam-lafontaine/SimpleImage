#include "../../tests_include.hpp"
#include "../../../simage/src/util/color_space.hpp"


static bool equals(f32 lhs, f32 rhs)
{
    return std::abs(lhs - rhs) < (1.0f / 255.0f);
}


bool hsv_conversion_test()
{
    printf("hsv_conversion_test: ");

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

    auto buffer = img::create_buffer32(width * height * 3);

    auto hsv = img::make_view_3(width, height, buffer);

    auto hue = img::select_channel(hsv, img::HSV::H);
    auto sat = img::select_channel(hsv, img::HSV::S);
    auto val = img::select_channel(hsv, img::HSV::V);

    auto r = make_range(width, height);
    for (u32 x = 0; x < width; ++x)
    {
        r.x_begin = x;
        r.x_end = x + 1;
        img::fill(img::sub_view(hue, r), (f32)x / width);
    }

    r = make_range(width, height);
    for (u32 y = 0; y < height; ++y)
    {
        r.y_begin = y;
        r.y_end = y + 1;

        img::fill(img::sub_view(sat, r), (f32)y / height);
    }

    img::fill(val, 1.0f);

    img::map_hsv_rgb(hsv, out);
    
    mb::destroy_buffer(buffer);
}